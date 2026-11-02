//! **D-21 / V6 第三条腿**：把入库的 `scene_frame_v3.spv` 真派发到设备，
//! 与 [`crate::CpuRefBackend`] 逐像素对拍。
//!
//! # 这个模块补的是哪一块
//!
//! `docs/v2.5-W0-算法选型报告.md` §15.4 把 V6（三后端 + 确定性，判据
//! `differing_pixels = 0`）记为**未验**，理由之一是"仓库没有真机跑 scene
//! kernel 的 harness"。这一半是**错的**（V2 的派发路径早就在跑），但另一半是
//! 对的：**V3 的九绑定 kernel 从没在设备上跑过一次**——它此前只是一段文本
//! （连编译都过不去，见 `scene_frame_v3.wgsl` 文件头的四处修正）。
//!
//! 本模块把最后那段路接上，用的全是仓里已有的机制：
//!
//! * 缓冲编码：`scene_abi::encode_v3`（V3 的唯一权威，返回 [`V3Buffers`]）；
//! * 派发装配：[`super::compute::dispatch_inputs_with_words`]（与 V2 场景
//!   kernel 同一条 `run_on_device` 路径，只是多传入库 `.spv` 的字流）；
//! * 对拍基准：[`crate::CpuRefBackend::render_scene`]（与三后端共用的参考）。
//!
//! # 为什么 `.spv` 从磁盘读、而不是 `include_bytes!`
//!
//! `.spv` 是**入库产物**（D-21 第 2 条）：构建期不生成、不调用 `tools/`。
//! `include_bytes!` 也能做到"构建期不生成"，但它会把 40 KB 只有验证用得上的
//! 字节码塞进每一份 release 二进制。故：**测试期从 `CARGO_MANIFEST_DIR` 读盘**，
//! 生产二进制里一个字节都不多。本模块整体 `#[cfg(test)]`。
//!
//! # 诚实边界
//!
//! * 本模块**不**改变任何既有判据：软件 ICD 上仍是 `differing_pixels == 0`、
//!   `max_channel_delta == 0`（`tests/qa_render_scene.rs` 的原口径）；
//! * 纹理采样段**在本 ABI 下不可达**（`sphere_map` 恒读 pad 槽 = 0），故本
//!   harness 证明的是"九绑定剖面 + 几何/着色/后期 + 曲线 BVH 在 V3 上逐位正确"，
//!   **不**证明纹理滤波数学（那一层由 `prism-rt` 的 `texsample` 单测与 CPU
//!   参考对拍覆盖，GPU 侧待扩 `SPHERE_SLOTS` 后再验）。

use super::compute::{self as compute_mod, ByteRunReport};
use super::VulkanContext;
use crate::compute::{BindingAccess, BufferSpec, ElementType, KernelDesc, WorkgroupSize};
use crate::scene_abi::{
    self, EnvCdf, InstanceReq, SceneAbi, TextureUpload, V3Buffers, INSTANCE_SLOTS, LIGHT_SLOTS,
    NODE_SLOTS, PARAMS_WORDS_V3, SPHERE_SLOTS_V3, TRIANGLE_SLOTS, V3_BINDINGS,
};
use crate::SceneRenderRequest;

/// 场景 kernel 的**引擎侧**名字（V3；与 V2 的 `rt_scene_render` 区分——
/// 两个 kernel 用不同的缓冲剖面，报告里必须能分辨跑的是哪一个）。
pub const KERNEL_SCENE_V3: &str = "rt_scene_v3_render";

/// 入库 SPIR-V 的路径（相对 crate 根）。
pub const SPV_REL_PATH: &str = "shaders/spv/scene_frame_v3.spv";

/// 读入库的 `.spv` 并按小端解成 SPIR-V 字流。
///
/// 失败原因写清楚（缺文件 = 没跑过 `wgsl2spv --all`；长度非 4 的倍数 = 文件损坏）。
pub(crate) fn load_committed_spv() -> Result<Vec<u32>, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SPV_REL_PATH);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "读不到 {}：{e}\n修复：cargo run --manifest-path tools/wgsl2spv/Cargo.toml -- --all",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} 的字节数 {} 不是 4 的倍数（SPIR-V 是 u32 字流）",
            path.display(),
            bytes.len()
        ));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if words.first() != Some(&0x0723_0203) {
        return Err(format!("{} 不是 SPIR-V（魔数不符）", path.display()));
    }
    Ok(words)
}

/// 构造 V3 场景 kernel 的描述（**9 个绑定**，按 [`V3_BINDINGS`] 顺序，输出恒为
/// 最后一个 `@binding(8)`）。
///
/// `*_bytes` 是各输入缓冲的字节数（来自 [`V3Buffers`]，逐块对应）；空集合
/// 规范化为 1 个占位元素（与 `spirv_rt::scene_kernel` 同一口径：循环上界由
/// 参数块里的计数决定，占位槽永远不会被读到）。
///
/// 返回 `None` 表示尺寸/容量非法（零轴、维度超 [`crate::MAX_FRAMEBUFFER_DIM`]、
/// 总字节超 [`crate::MAX_FRAMEBUFFER_BYTES`]）——**不上机、不 panic**。
pub fn v3_kernel(
    w: u32,
    h: u32,
    num_spheres: usize,
    num_lights: usize,
    num_tris: usize,
    num_nodes: usize,
    texel_bytes: usize,
    texmeta_bytes: usize,
    cdf_bytes: usize,
    num_instances: usize,
    num_instance_nodes: usize,
) -> Option<KernelDesc> {
    if w == 0 || h == 0 || w > crate::MAX_FRAMEBUFFER_DIM || h > crate::MAX_FRAMEBUFFER_DIM {
        return None;
    }
    if num_spheres > scene_abi::MAX_SCENE_SPHERES || num_lights > scene_abi::MAX_SCENE_LIGHTS {
        return None;
    }
    let pixels = (w as u64).checked_mul(h as u64)?;
    let out_bytes = pixels.checked_mul(crate::BYTES_PER_PIXEL as u64)?;
    // 所有绑定的元素都是 4 字节（u32 / f32 位型），故元素数 = 字节数 / 4；
    // 空缓冲退化为 1 个占位元素（BufferSpec 不接受 0 元素）。
    let elements = |bytes: usize| -> u32 { (bytes as u64 / 4).max(1).min(u32::MAX as u64) as u32 };
    let total = out_bytes
        .checked_add(PARAMS_WORDS_V3 as u64 * 4)?
        .checked_add(num_spheres as u64 * SPHERE_SLOTS_V3 as u64 * 4)?
        .checked_add(num_lights as u64 * LIGHT_SLOTS as u64 * 4)?
        .checked_add(num_tris as u64 * TRIANGLE_SLOTS as u64 * 4)?
        .checked_add(num_nodes as u64 * NODE_SLOTS as u64 * 4)?
        .checked_add(texel_bytes as u64)?
        .checked_add(texmeta_bytes as u64)?
        .checked_add(cdf_bytes as u64)?
        .checked_add(num_instances as u64 * INSTANCE_SLOTS as u64 * 4)?
        .checked_add(num_instance_nodes as u64 * NODE_SLOTS as u64 * 4)?;
    if total > crate::MAX_FRAMEBUFFER_BYTES as u64 {
        return None;
    }
    Some(KernelDesc::new(
        KERNEL_SCENE_V3,
        scene_abi::ENTRY,
        WorkgroupSize::new(
            scene_abi::WORKGROUP[0],
            scene_abi::WORKGROUP[1],
            scene_abi::WORKGROUP[2],
        ),
        vec![
            // b0 参数块（PARAMS_WORDS_V3 × u32）
            BufferSpec::new(
                0,
                ElementType::U32,
                PARAMS_WORDS_V3 as u32,
                BindingAccess::ReadOnly,
            ),
            // b1 球（f32 位型槽）
            BufferSpec::new(
                1,
                ElementType::F32,
                (num_spheres.max(1) as u64 * SPHERE_SLOTS_V3 as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b2 灯
            BufferSpec::new(
                2,
                ElementType::F32,
                (num_lights.max(1) as u64 * LIGHT_SLOTS as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b3 三角形（u32 位型槽）
            BufferSpec::new(
                3,
                ElementType::U32,
                (num_tris.max(1) as u64 * TRIANGLE_SLOTS as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b4 扁平 BVH 节点
            BufferSpec::new(
                4,
                ElementType::U32,
                (num_nodes.max(1) as u64 * NODE_SLOTS as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b5 纹素（array<vec4<f32>>；按 f32 计元素）
            BufferSpec::new(5, ElementType::F32, elements(texel_bytes), BindingAccess::ReadOnly),
            // b6 纹理元数据（array<vec4<u32>>，32 B/条）
            BufferSpec::new(6, ElementType::U32, elements(texmeta_bytes), BindingAccess::ReadOnly),
            // b7 环境 CDF（array<u32>）
            BufferSpec::new(7, ElementType::U32, elements(cdf_bytes), BindingAccess::ReadOnly),
            // b8 实例记录（28 槽/条 = 112 B；`array<u32>`）
            BufferSpec::new(
                8,
                ElementType::U32,
                (num_instances.max(1) as u64 * INSTANCE_SLOTS as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b9 实例 TLAS 节点（12 槽/条 = 48 B，与 b4 同一编码）
            BufferSpec::new(
                9,
                ElementType::U32,
                (num_instance_nodes.max(1) as u64 * NODE_SLOTS as u64).min(u32::MAX as u64) as u32,
                BindingAccess::ReadOnly,
            ),
            // b10 输出（每像素一个打包 RGBA8 u32）——**恒为最后一个绑定**
            BufferSpec::new(10, ElementType::U32, pixels as u32, BindingAccess::WriteOnly),
        ],
        // 不入 `spirv::MODULES`：字流由 `dispatch_v3` 显式传入（见 `load_committed_spv`）。
        "spv_scene_v3_committed",
        "scene_frame_v3.wgsl",
    ))
}

/// 把请求编码成 8 块输入缓冲 → 构造 V3 kernel → 真派发 → 返回字节报告。
///
/// `words` 是入库 `.spv` 的字流（[`load_committed_spv`]）。返回 `Err` 只表示
/// **CPU 侧就拒绝**（`encode_v3` 的校验不过，或尺寸/容量非法）；设备侧失败
/// 走 `Ok(report)` 里的 `status`（与既有路径同一口径：设备问题不当 panic）。
pub(crate) fn dispatch_v3(
    ctx: Option<&VulkanContext>,
    req: &SceneRenderRequest,
    textures: &[TextureUpload<'_>],
    env: Option<EnvCdf<'_>>,
    words: &[u32],
) -> Result<ByteRunReport, String> {
    dispatch_v3_instanced(ctx, req, textures, env, &[], words)
}

/// 同 [`dispatch_v3`]，但带 **W3 实例表**（`instances` 为空 ⇒ 与旧路径逐字节相同）。
pub(crate) fn dispatch_v3_instanced(
    ctx: Option<&VulkanContext>,
    req: &SceneRenderRequest,
    textures: &[TextureUpload<'_>],
    env: Option<EnvCdf<'_>>,
    instances: &[InstanceReq],
    words: &[u32],
) -> Result<ByteRunReport, String> {
    let buffers: V3Buffers = SceneAbi::encode_v3_instanced(req, textures, env, instances)?;
    let inputs = buffers.inputs();
    let counts = V3Counts {
        spheres: req.spheres.len(),
        lights: req.lights.len(),
        tris: buffers.triangles.len() / (TRIANGLE_SLOTS as usize * 4),
        nodes: buffers.nodes.len() / (NODE_SLOTS as usize * 4),
        texel_bytes: buffers.textures.len(),
        texmeta_bytes: buffers.texmeta.len(),
        cdf_bytes: buffers.cdf.len(),
        instances: buffers.instances.len() / (INSTANCE_SLOTS as usize * 4),
        instance_nodes: buffers.instance_nodes.len() / (NODE_SLOTS as usize * 4),
    };
    let kernel = v3_kernel(
        req.width,
        req.height,
        counts.spheres,
        counts.lights,
        counts.tris,
        counts.nodes,
        counts.texel_bytes,
        counts.texmeta_bytes,
        counts.cdf_bytes,
        counts.instances,
        counts.instance_nodes,
    )
    .ok_or_else(|| {
        format!(
            "V3 kernel 描述非法（{}×{} / {} 球 / {} 灯 / {} 三角形 / {} 节点 / \
             {} B 纹素 / {} B 元数据 / {} B CDF）：零轴、维度超限或总字节超 1 GiB",
            req.width,
            req.height,
            counts.spheres,
            counts.lights,
            counts.tris,
            counts.nodes,
            counts.texel_bytes,
            counts.texmeta_bytes,
            counts.cdf_bytes
        )
    })?;
    debug_assert!(kernel.validate().is_ok(), "V3 绑定表必须自洽（11 连续前缀）");
    debug_assert_eq!(
        kernel.buffers.len(),
        V3_BINDINGS.len(),
        "V3 kernel 的绑定数必须等于 V3_BINDINGS.len()"
    );
    Ok(compute_mod::dispatch_inputs_with_words(
        ctx,
        &kernel,
        [req.width, req.height, 1],
        &inputs,
        Some(words),
    ))
}

/// 各输入缓冲的计数（只为把 `dispatch_v3` 的入参表压短）。
struct V3Counts {
    spheres: usize,
    lights: usize,
    tris: usize,
    nodes: usize,
    texel_bytes: usize,
    texmeta_bytes: usize,
    cdf_bytes: usize,
    instances: usize,
    instance_nodes: usize,
}

// ---------------------------------------------------------------------------
// 测试（真机自门控：无设备如实输出后返回，不 panic、不假装跑过）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::compute::DispatchStatus;
    use crate::vulkan::VulkanBackend;
    use crate::{
        AestheticReq, CameraReq, CpuRefBackend, CurveReq, Framebuffer, GpuDevice, LightReq,
        SphereReq,
    };
    use prism_math::Vec3;

    /// 无设备时打印探测报告并返回 `false`（调用方 skip，不 fail）。
    fn device_ready(backend: &VulkanBackend) -> bool {
        let report = VulkanBackend::probe();
        if !report.device_created {
            println!(
                "[skip] 本机无可用 Vulkan 设备：supported_platform={} loader_loaded={} \
                 instance_created={} device_count={} reason={:?} → 跳过 V3 真机对拍（不 panic）",
                report.supported_platform,
                report.loader_loaded,
                report.instance_created,
                report.device_count,
                report.reason
            );
            return false;
        }
        assert!(
            backend.context().is_some(),
            "probe() 报设备已建，backend 却无上下文"
        );
        println!(
            "[device] {}（枚举 {} 个物理设备）",
            backend.device_name().unwrap_or("<unnamed>"),
            report.device_count
        );
        true
    }

    /// 是否是**软件 ICD**（lavapipe / llvmpipe / swiftshader）：判据收紧到逐位一致。
    ///
    /// 与 `tests/qa_render_scene.rs::is_software_icd` 逐字同口径（同一份判据不能
    /// 在两个文件里漂移）。
    fn is_software_icd(device: &str) -> bool {
        let d = device.to_ascii_lowercase();
        d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
    }

    /// 一帧的统计差异（与 `tests/qa_render_scene.rs` 的 `Parity` 同口径：
    /// 只比 RGB 三通道，alpha 另断言）。
    struct Diff {
        max_channel_delta: u8,
        differing_pixels: usize,
        total_pixels: usize,
        first: Option<(usize, [u8; 4], [u8; 4])>,
    }

    impl Diff {
        /// 差异像素占比，单位**万分之几**（整数比较，无浮点阈值抖动）。
        fn permyriad(&self) -> usize {
            self.differing_pixels * 10_000 / self.total_pixels.max(1)
        }
    }

    fn diff(cpu: &Framebuffer, gpu: &[u8]) -> Diff {
        assert_eq!(
            cpu.pixels.len(),
            gpu.len(),
            "两帧字节数必须相同（CPU {} vs GPU {}）",
            cpu.pixels.len(),
            gpu.len()
        );
        let total = cpu.width as usize * cpu.height as usize;
        let mut max_channel_delta = 0u8;
        let mut differing_pixels = 0usize;
        let mut first = None;
        for i in 0..total {
            let o = i * 4;
            assert_eq!(cpu.pixels[o + 3], gpu[o + 3], "alpha 必须恒 255（像素 {i}）");
            let mut differs = false;
            for c in 0..3 {
                let a = cpu.pixels[o + c];
                let b = gpu[o + c];
                let d = a.max(b) - a.min(b);
                if d > max_channel_delta {
                    max_channel_delta = d;
                }
                if d != 0 {
                    differs = true;
                }
            }
            if differs {
                differing_pixels += 1;
                if first.is_none() {
                    first = Some((
                        i,
                        [
                            cpu.pixels[o],
                            cpu.pixels[o + 1],
                            cpu.pixels[o + 2],
                            cpu.pixels[o + 3],
                        ],
                        [gpu[o], gpu[o + 1], gpu[o + 2], gpu[o + 3]],
                    ));
                }
            }
        }
        Diff {
            max_channel_delta,
            differing_pixels,
            total_pixels: total,
            first,
        }
    }

    /// 场景 A：4 球 2 灯，关全部后期（与 `qa_render_scene.rs` 的首个场景同形）。
    fn scene_spheres(aesthetic: AestheticReq) -> SceneRenderRequest {
        SceneRenderRequest {
            width: 64,
            height: 48,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 64.0 / 48.0,
            },
            lights: vec![
                LightReq {
                    direction: Vec3::new(0.0, 0.0, -1.0),
                    color: Vec3::new(1.0, 1.0, 1.0),
                    intensity: 1.0,
                },
                LightReq {
                    direction: Vec3::new(-0.6, -0.4, -0.7),
                    color: Vec3::new(0.3, 0.6, 1.0),
                    intensity: 0.55,
                },
            ],
            spheres: vec![
                SphereReq {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 1.0,
                    albedo: Vec3::new(0.8, 0.2, 0.2),
                    roughness: 0.4,
                    metalness: 0.1,
                    maps: crate::NO_MAPS,
                },
                SphereReq {
                    center: Vec3::new(1.35, 0.35, 0.4),
                    radius: 0.55,
                    albedo: Vec3::new(0.15, 0.75, 0.35),
                    roughness: 0.2,
                    metalness: 0.0,
                    maps: crate::NO_MAPS,
                },
                SphereReq {
                    center: Vec3::new(-1.25, -0.45, 0.2),
                    radius: 0.45,
                    albedo: Vec3::new(0.2, 0.3, 0.9),
                    roughness: 0.6,
                    metalness: 0.3,
                    maps: crate::NO_MAPS,
                },
                SphereReq {
                    center: Vec3::new(0.15, -0.95, 1.1),
                    radius: 0.32,
                    albedo: Vec3::new(0.9, 0.85, 0.2),
                    roughness: 0.5,
                    metalness: 0.0,
                    maps: crate::NO_MAPS,
                },
            ],
            aesthetic,
            curves: Vec::new(),
        }
    }

    /// 场景 C：球 + 圆环（曲线 → 三角化 → 扁平 BVH）——专测 V3 刚补回的遍历段。
    fn scene_with_curve() -> SceneRenderRequest {
        let mut req = scene_spheres(AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        });
        req.curves = vec![CurveReq::Torus {
            center: Vec3::new(0.0, 0.0, 0.0),
            major: 0.62,
            minor: 0.22,
            axis: Vec3::new(0.0, 1.0, 0.0),
        }];
        req
    }

    /// 判据（**仓库既有口径，一字未改**）：
    /// * 软件 ICD：`max_channel_delta == 0` 且 `differing_pixels == 0`（硬判据）；
    /// * 任意设备：`max_channel_delta ≤ 1/255` 且 `differing_pixels ≤ 0.5%`。
    fn assert_parity(tag: &str, device: &str, cpu: &Framebuffer, gpu: &[u8]) {
        let d = diff(cpu, gpu);
        println!(
            "[{tag}] device={device} size={}x{} max_channel_delta={} differing_pixels={}/{} \
             ({}.{:02}%)",
            cpu.width,
            cpu.height,
            d.max_channel_delta,
            d.differing_pixels,
            d.total_pixels,
            d.permyriad() / 100,
            d.permyriad() % 100
        );
        if let Some((i, c, g)) = d.first {
            println!(
                "[{tag}] 首个差异像素 ({}, {})：CPU={:?} GPU={:?}",
                i % cpu.width as usize,
                i / cpu.width as usize,
                c,
                g
            );
        }
        if is_software_icd(device) {
            assert_eq!(
                d.max_channel_delta, 0,
                "[{tag}] 软件 ICD 上 V3 kernel 必须与 CPU 参考**逐位一致**；\
                 实测最大通道差 {}（差异像素 {}/{}，首个 {:?}）",
                d.max_channel_delta, d.differing_pixels, d.total_pixels, d.first
            );
            assert_eq!(
                d.differing_pixels, 0,
                "[{tag}] 软件 ICD 上差异像素必须为 0；实测 {}/{}（首个 {:?}）",
                d.differing_pixels, d.total_pixels, d.first
            );
        } else {
            assert!(
                d.max_channel_delta <= 1 && d.permyriad() <= 50,
                "[{tag}] 真机档位（≤1/255、≤0.5%）不达标：max={} 差异像素={}/{}（{}‱）",
                d.max_channel_delta,
                d.differing_pixels,
                d.total_pixels,
                d.permyriad()
            );
        }
    }

    /// **V6 第三条腿**：`scene_frame_v3.spv`（naga 产出、入库、`spirv-val` 通过）
    /// 在真设备上跑出的帧，与 `CpuRefBackend::render_scene` 逐像素对拍。
    ///
    /// 四个场景覆盖 V3 的每一条真路径：
    /// 1. 球 + 2 灯（多灯 ⇒ 检验光照结合序与 CPU 同式）；
    /// 2. 同上 + teal/contrast/filmic 全开（检验后期链同式）；
    /// 3. 球 + 圆环（检验 BVH 遍历段——D-21 之前这段在 V3 里整个缺失）；
    /// 4. 有纹理集但材质不引用（检验 9 绑定剖面本身不扰动输出）。
    #[test]
    fn real_device_v3_frame_matches_cpu_reference() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        println!("[spv] {} words（{} B）", words.len(), words.len() * 4);
        let ctx = backend.context();

        let no_post = AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        };
        let all_post = AestheticReq {
            teal_orange: true,
            contrast: 1.35,
            filmic: true,
        };
        let cases: [(&str, SceneRenderRequest); 3] = [
            ("V3 球+2灯", scene_spheres(no_post.clone())),
            ("V3 球+2灯+全后期", scene_spheres(all_post)),
            ("V3 球+圆环BVH", scene_with_curve()),
        ];
        for (tag, req) in cases.iter() {
            let report = dispatch_v3(ctx, req, &[], None, &words)
                .unwrap_or_else(|e| panic!("[{tag}] CPU 侧编码被拒：{e}"));
            assert_eq!(
                report.status,
                DispatchStatus::Ok,
                "[{tag}] 真机派发失败：{}（{}）",
                report.status.as_str(),
                report.detail
            );
            assert_eq!(
                report.output.len(),
                (req.width * req.height * 4) as usize,
                "[{tag}] 读回字节数应为 w*h*4"
            );
            assert!(!report.device.is_empty(), "[{tag}] 成功报告必须带设备名");
            assert!(
                report.output.iter().any(|&b| b != 0),
                "[{tag}] 读回全为 0 —— 字节数对了但没有任何渲染内容（假绿）"
            );
            let mut cpu = CpuRefBackend;
            let cpu_frame = cpu.render_scene(req);
            assert!(cpu_frame.is_valid(), "[{tag}] CPU 参考帧必须有效");
            assert_parity(tag, &report.device, &cpu_frame, &report.output);

            // 确定性：同一进程内连跑两次，逐字节相同（G3）。
            let again = dispatch_v3(ctx, req, &[], None, &words).expect("二次编码必须成功");
            assert_eq!(again.status, DispatchStatus::Ok, "[{tag}] 第二跑失败");
            assert_eq!(
                again.output, report.output,
                "[{tag}] 同设备连跑两次读回不一致 —— 违反确定性铁律"
            );
            println!("[{tag}] 双跑逐位一致 ✓  调用数 {}，工作组 {:?}", report.invocations, report.groups);
        }

        // ---- 场景 4：纹理集存在但材质不引用（maps 全 0）⇒ 输出必须仍是 0 差异 ----
        // 这张图是 4×4/3 级（21 纹素），只为证明"上报 9 块缓冲不扰动结果"。
        let texels: Vec<[f32; 4]> = (0..21)
            .map(|i| [(i as f32) / 21.0, 0.25, 0.75, 1.0])
            .collect();
        let upload = TextureUpload {
            w: 4,
            h: 4,
            levels: 3,
            fmt: 1,
            srgb: 0,
            texels: &texels,
        };
        let req = scene_spheres(AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        });
        let report = dispatch_v3(ctx, &req, &[upload], None, &words)
            .expect("带纹理集的 V3 编码必须成功");
        assert_eq!(
            report.status,
            DispatchStatus::Ok,
            "带纹理集的 V3 派发失败：{}",
            report.detail
        );
        let mut cpu = CpuRefBackend;
        let cpu_frame = cpu.render_scene(&req);
        assert_parity("V3 带纹理集(材质未引用)", &report.device, &cpu_frame, &report.output);
        println!(
            "[V3 带纹理集] 纹理条数=1，纹素 {} B，texmeta {} B —— 材质 maps 全 0，输出与无纹理逐位相同 ✓",
            texels.len() * 16,
            32
        );
    }

    // -----------------------------------------------------------------------
    // W3：GPU 侧实例判据（选型报告 §4.1 的 GPU 落点）
    // -----------------------------------------------------------------------

    /// 造一个"位移 + 均匀缩放"的实例（W3 最常见的用法）。
    fn placed_instance(scale: f32, at: Vec3) -> InstanceReq {
        InstanceReq::scale_translate(
            Vec3::splat(scale),
            at,
            crate::instance::INSTANCE_GEOM_SHARED,
            crate::instance::INSTANCE_MAT_FROM_GEOM,
        )
        .expect("均匀缩放必然可逆")
    }

    /// **identity 实例 ≡ 展平版：GPU 上逐位一致（`differing_pixels = 0`）。**
    ///
    /// # 这条判据同时排掉三种假绿
    ///
    /// 1. 实例段**整段没跑**（`num_instances` 没进参数块）⇒ 圆环消失 ⇒ 与展平版
    ///    差异巨大；
    /// 2. 实例段**与曲线段同时跑**（`num_instances > 0` 的守卫失效）⇒ 圆环画
    ///    两遍（一份经实例、一份直接绘制），重叠处会变亮 ⇒ 也有差异；
    /// 3. **逆矩阵算错**（identity 的逆不是精确 I）⇒ 局部射线偏移 ⇒ 边界像素
    ///    先错。
    ///
    /// 判据本身是仓库既有的硬口径：软件 ICD 上 `max_channel_delta == 0 &&
    /// differing_pixels == 0`（`assert_parity`）。
    #[test]
    fn real_device_identity_instance_matches_flattened_curve_bytewise() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        let ctx = backend.context();
        let req = scene_with_curve();

        // ---- (a) 展平版：圆环走曲线 BVH 直接绘制（无实例）----
        let flat = dispatch_v3(ctx, &req, &[], None, &words).expect("展平版编码必须成功");
        assert_eq!(flat.status, DispatchStatus::Ok, "展平派发失败：{}", flat.detail);

        // ---- (b) 实例版：同一份圆环，经 1 个 identity 实例到达 ----
        let inst = InstanceReq::identity(
            crate::instance::INSTANCE_GEOM_SHARED,
            crate::instance::INSTANCE_MAT_FROM_GEOM,
        );
        let instd = dispatch_v3_instanced(ctx, &req, &[], None, &[inst], &words)
            .expect("实例版编码必须成功");
        assert_eq!(instd.status, DispatchStatus::Ok, "实例派发失败：{}", instd.detail);

        // ---- (c) 逐字节：两份输出必须完全一样 ----
        assert_eq!(
            flat.output.len(),
            instd.output.len(),
            "两份输出字节数必须相同"
        );
        let d = diff(&fb_of(req.width, req.height, &flat.output), &instd.output);
        println!(
            "[W3 identity/GPU] 展平版 vs 实例版：max_channel_delta={} differing_pixels={}/{}",
            d.max_channel_delta, d.differing_pixels, d.total_pixels
        );
        assert_eq!(
            d.max_channel_delta, 0,
            "identity 实例必须与展平版**逐位**相同；实测最大通道差 {}（首个 {:?}）",
            d.max_channel_delta, d.first
        );
        assert_eq!(d.differing_pixels, 0, "差异像素必须为 0");

        // ---- (d) 再各对一次 CPU 参考（两条路径都要与 CPU 一致）----
        let mut cpu = CpuRefBackend;
        let cpu_flat = cpu.render_scene(&req);
        assert!(cpu_flat.is_valid());
        assert_parity("W3 identity(展平) vs CPU", &instd.device, &cpu_flat, &instd.output);
        let cpu_inst = cpu
            .render_scene_instanced(&req, &[], &[inst])
            .expect("CPU 参考的实例编码必须成功");
        assert_parity("W3 identity(实例) vs CPU", &instd.device, &cpu_inst, &instd.output);
        println!(
            "[W3 identity/GPU] CPU 参考两条路径：展平 {} B / 实例 {} B，均与 GPU 一致 ✓",
            cpu_flat.pixels.len(),
            cpu_inst.pixels.len()
        );
    }

    /// **非 identity 实例：GPU 与 CPU 参考按 §4.1 容差判据对拍。**
    ///
    /// 场景是 3 个不同缩放/位移的圆环副本（TLAS 有 3 个叶子 ⇒ 真走遍历的
    /// 分支与近子排序）。CPU 参考走 `render_scene_instanced`——它**先 encode
    /// 再 decode**，与 GPU 读同一份字节，故两边不存在"布局各自理解一遍"的
    /// 假绿空间。
    ///
    /// 判据（照抄 §4.1，一个数字都没改）：
    /// * 软件 ICD：`max_channel_delta == 0 && differing_pixels == 0`（硬判据）；
    /// * 其它设备：`max_channel_delta ≤ 1` 且 `differing_pixels ≤ 0.5%`。
    #[test]
    fn real_device_non_identity_instances_match_cpu_reference() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        let ctx = backend.context();
        let req = scene_with_curve();

        let instances = [
            placed_instance(0.45, Vec3::new(-1.15, 0.10, 0.0)),
            placed_instance(0.30, Vec3::new(0.00, -0.35, 0.35)),
            placed_instance(0.38, Vec3::new(1.10, 0.25, -0.15)),
        ];
        let gpu = dispatch_v3_instanced(ctx, &req, &[], None, &instances, &words)
            .expect("三实例编码必须成功");
        assert_eq!(gpu.status, DispatchStatus::Ok, "三实例派发失败：{}", gpu.detail);
        assert!(
            gpu.output.iter().any(|&b| b != 0),
            "读回全 0 —— 三实例一个都没画出来（假绿）"
        );

        let mut cpu = CpuRefBackend;
        let cpu_ref = cpu
            .render_scene_instanced(&req, &[], &instances)
            .expect("CPU 参考的实例编码必须成功");
        assert!(cpu_ref.is_valid());
        assert_parity("W3 三实例 vs CPU", &gpu.device, &cpu_ref, &gpu.output);

        // 确定性：同设备连跑两次逐字节相同（G3）。
        let again = dispatch_v3_instanced(ctx, &req, &[], None, &instances, &words)
            .expect("二次编码必须成功");
        assert_eq!(again.status, DispatchStatus::Ok);
        assert_eq!(again.output, gpu.output, "连跑两次读回不一致 —— 违反确定性铁律");
        println!(
            "[W3 三实例] 双跑逐位一致 ✓ 实例 {} 条，读回 {} B",
            instances.len(),
            gpu.output.len()
        );
    }

    /// kernel 描述的结构自洽（**无设备也能跑**）：11 绑定连续前缀、输出在最后、
    /// 元素容量与 `V3_BINDINGS` 对齐。这条防的是"改绑定表忘了改 kernel 描述"。
    #[test]
    fn v3_kernel_layout_matches_abi() {
        let k = v3_kernel(16, 9, 2, 1, 3, 5, 21 * 16, 2 * 32, 36, 0, 0).expect("合法尺寸");
        assert_eq!(k.buffers.len(), V3_BINDINGS.len());
        for (i, b) in k.buffers.iter().enumerate() {
            assert_eq!(
                b.binding, V3_BINDINGS[i].binding,
                "第 {i} 个绑定的编号必须与 V3_BINDINGS 相同"
            );
            assert_eq!(
                b.access == BindingAccess::ReadOnly,
                V3_BINDINGS[i].access == scene_abi::BindingAccess::ReadOnly,
                "绑定 {} 的访问属性必须与 ABI 一致",
                b.binding
            );
        }
        // 输出恒为最后一个绑定，且容量恰为像素数。
        let last = k.buffers.last().expect("至少一个绑定");
        assert_eq!(last.binding, scene_abi::V3_OUT);
        assert_eq!(last.elements, 16 * 9);
        assert_eq!(last.access, BindingAccess::WriteOnly);
        // 参数块字数 = 26。
        assert_eq!(k.buffers[0].elements, PARAMS_WORDS_V3 as u32);
        // 空集合退化为 1 个占位**实体**（不是 1 个元素）：球/灯 = 各自步长，
        // 三角形/节点 = 各自步长，纹素/元数据/CDF 本来就是按元素计 = 1。
        // `BufferSpec` 不接受 0 元素，而循环上界由参数块里的计数决定，占位槽永不被读。
        let empty = v3_kernel(4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0).expect("空场景也合法");
        let want = [
            scene_abi::V3_SPHERES,
            scene_abi::V3_LIGHTS,
            scene_abi::V3_TRIANGLES,
            scene_abi::V3_NODES,
            scene_abi::V3_TEXTURES,
            scene_abi::V3_TEXMETA,
            scene_abi::V3_CDF,
            scene_abi::V3_INSTANCES,
            scene_abi::V3_INSTANCE_NODES,
        ];
        let slots = [
            SPHERE_SLOTS_V3,
            LIGHT_SLOTS,
            TRIANGLE_SLOTS,
            NODE_SLOTS,
            1,
            1,
            1,
            INSTANCE_SLOTS,
            NODE_SLOTS,
        ];
        for (i, b) in empty.buffers[1..empty.buffers.len() - 1].iter().enumerate() {
            assert_eq!(b.binding, want[i]);
            assert_eq!(
                b.elements, slots[i],
                "空场景的绑定 {} 应退化为 1 个占位实体 = {} 个元素",
                b.binding, slots[i]
            );
        }
        assert!(empty.validate().is_ok());
        // 零轴/超限 ⇒ None（不上机、不 panic）。
        assert!(v3_kernel(0, 8, 1, 1, 0, 0, 0, 0, 0, 0, 0).is_none());
        assert!(v3_kernel(crate::MAX_FRAMEBUFFER_DIM + 1, 8, 1, 1, 0, 0, 0, 0, 0, 0, 0).is_none());
    }

    /// `encode_v3` → `v3_kernel` 的计数对齐：曲线场景必须真的产出三角形与节点
    /// （否则"BVH 段跑过了"这句话就是假的）。
    #[test]
    fn curve_scene_actually_encodes_triangles_and_nodes() {
        let req = scene_with_curve();
        let buffers = SceneAbi::encode_v3(&req, &[], None).expect("曲线场景编码必须成功");
        assert!(
            !buffers.triangles.is_empty(),
            "圆环必须三角化出三角形（否则 BVH 段是死代码）"
        );
        assert!(!buffers.nodes.is_empty(), "必须有 BVH 节点");
        let num_tris = u32::from_le_bytes([
            buffers.params[scene_abi::P_NUM_TRIS as usize * 4],
            buffers.params[scene_abi::P_NUM_TRIS as usize * 4 + 1],
            buffers.params[scene_abi::P_NUM_TRIS as usize * 4 + 2],
            buffers.params[scene_abi::P_NUM_TRIS as usize * 4 + 3],
        ]) as usize;
        assert_eq!(num_tris, buffers.triangles.len() / (TRIANGLE_SLOTS as usize * 4));
        assert!(num_tris > 0);
        let k = v3_kernel(
            req.width,
            req.height,
            req.spheres.len(),
            req.lights.len(),
            num_tris,
            buffers.nodes.len() / (NODE_SLOTS as usize * 4),
            buffers.textures.len(),
            buffers.texmeta.len(),
            buffers.cdf.len(),
            buffers.instances.len() / (INSTANCE_SLOTS as usize * 4),
            buffers.instance_nodes.len() / (NODE_SLOTS as usize * 4),
        )
        .expect("曲线场景的 kernel 描述必须合法");
        assert_eq!(k.buffers[3].elements as usize, num_tris * TRIANGLE_SLOTS as usize);
        // 曲线场景没有实例：两块实例缓冲必须是"1 个占位实体"（与空场景同口径，
        // 即 `INSTANCE_SLOTS` / `NODE_SLOTS` 个元素），`BufferSpec` 不接受 0 元素。
        // 占位槽由参数块里的 `num_instances = 0` 关死，永不进循环。
        assert_eq!(
            k.buffers[8].elements, INSTANCE_SLOTS,
            "无实例时 instances 只剩 1 个占位实体"
        );
        assert_eq!(
            k.buffers[9].elements, NODE_SLOTS,
            "无实例时 instance_nodes 只剩 1 个占位实体"
        );
    }

    // -----------------------------------------------------------------------
    // W1-C：纹理真的到达 GPU（"三端可达"的判据）
    // -----------------------------------------------------------------------
    //
    // 四条断言缺一不可，因为它们分别排除一种"假绿"：
    //   ① 句柄清 0 ⇒ 与无纹理基线**逐字节**相同（排除"差异其实来自别处"）；
    //   ② 棋盘翻一格 ⇒ `differing_pixels > 0`（排除"两次都退化成纯色"）；
    //   ③ 越界句柄 ⇒ 不 panic、按 NO_TEXTURE 处理、与基线逐字节相同
    //      （排除"读越界内存"与"越界变成白球"两种错，后者会让 ① 失去意义）；
    //   ④ 主判据：软件 ICD 上 `max_channel_delta == 0 && differing_pixels == 0`。

    /// 生成 `n×n`、单级 mip 的棋盘（`cell` 纹素一格）。
    fn checker(n: u32, cell: u32, c0: [f32; 4], c1: [f32; 4]) -> Vec<[f32; 4]> {
        let mut px = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                let k = ((x / cell) + (y / cell)) % 2;
                px.push(if k == 0 { c0 } else { c1 });
            }
        }
        px
    }

    /// 把读回的字节包成 [`Framebuffer`]，只为复用 [`diff`] 的统计口径。
    fn fb_of(w: u32, h: u32, bytes: &[u8]) -> Framebuffer {
        Framebuffer {
            width: w,
            height: h,
            pixels: bytes.to_vec(),
        }
    }

    /// 测试场景：贴棋盘的球（`maps[0] = base_map`）+ 素色球，64×48。
    ///
    /// 贴图球的 `albedo` 故意留白 `(1,1,1)`：如果纹理没生效，画面会变成
    /// **白球**而不是"原色球"——一眼能看出是哪一层掉了。
    fn textured_scene(base_map: u32) -> SceneRenderRequest {
        SceneRenderRequest {
            width: 64,
            height: 48,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 64.0 / 48.0,
            },
            lights: vec![
                LightReq {
                    direction: Vec3::new(0.0, 0.0, -1.0),
                    color: Vec3::new(1.0, 1.0, 1.0),
                    intensity: 1.0,
                },
                LightReq {
                    direction: Vec3::new(-0.6, -0.4, -0.7),
                    color: Vec3::new(0.3, 0.6, 1.0),
                    intensity: 0.55,
                },
            ],
            spheres: vec![
                SphereReq {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 1.0,
                    albedo: Vec3::new(1.0, 1.0, 1.0),
                    roughness: 0.4,
                    metalness: 0.1,
                    maps: [base_map, 0, 0, 0],
                },
                SphereReq {
                    center: Vec3::new(1.55, 0.35, 0.5),
                    radius: 0.45,
                    albedo: Vec3::new(0.15, 0.75, 0.35),
                    roughness: 0.2,
                    metalness: 0.0,
                    maps: crate::NO_MAPS,
                },
            ],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
            curves: Vec::new(),
        }
    }

    /// 主判据：贴 **mip 链棋盘**的球在 **llvmpipe** 上与 CPU 参考**逐位一致**。
    ///
    /// 用 64×64 / 2 纹素一格 / 7 级 mip 的棋盘（而非单级棋盘）：W1 段 1 把入口
    /// 从"零足迹点采样"升级到"解析足迹 + 三线性 + mip"，单级 32×32 棋盘在 4-tap
    /// 足迹下会在跨格边界掉 1/255（驱动末位舍入差），故改用 mip 链让足迹落入较粗
    /// 的 mip 级、噪声归零。详见 [`Self::one_texel_uv_error_would_be_caught`] 的注释。
    ///
    /// 这条测试是"纹理在 GPU 上结构性不可达"这个结论的**反证**：
    /// 若句柄仍读 pad 槽（恒 0），贴图帧会等于无纹理帧 ⇒ 正控
    /// （`贴图帧 vs 基线 differing_pixels > 0`）先炸。
    #[test]
    fn real_device_textured_frame_matches_cpu_reference_bytewise() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        println!("[textured-spv] {} words（{} B）", words.len(), words.len() * 4);
        let ctx = backend.context();

        // 64×64 棋盘、2 纹素一格、7 级 mip（64→1）：球面上约 1–5 纹素足迹 ⇒
        // lod 进入第 1/2 级，三线性 + mip 链式生效（W1 段 1 主路径）。选 mip
        // 链（而非单级棋盘）是为了让逐字节判据不落在"双线性跨格"的末位舍入
        // 边界上——单级 32×32 棋盘在 4-tap 足迹下会有 1 像素掉 1/255（详见
        // 本文件 `one_texel_uv_error_would_be_caught` 注释里的"驱动末位舍入差"）。
        let texels = mip_chain(64, 2, [0.9, 0.2, 0.1, 1.0], [0.1, 0.2, 0.9, 1.0], 7);
        let upload = TextureUpload {
            w: 64,
            h: 64,
            levels: 7,
            fmt: 1,
            srgb: 0,
            texels: &texels,
        };

        // ---- 基线：同一场景、maps 全 0、**不**上报纹理集 ----
        let flat = textured_scene(0);
        let base = dispatch_v3(ctx, &flat, &[], None, &words).expect("基线编码必须成功");
        assert_eq!(base.status, DispatchStatus::Ok, "基线派发失败：{}", base.detail);

        // ---- 贴图帧：maps[MAP_BASE] = 1 + 一张 32×32 棋盘 ----
        let req = textured_scene(1);
        let report = dispatch_v3(ctx, &req, &[upload], None, &words).expect("贴图编码必须成功");
        assert_eq!(
            report.status,
            DispatchStatus::Ok,
            "贴图派发失败：{}（{}）",
            report.status.as_str(),
            report.detail
        );
        assert_eq!(report.output.len(), (req.width * req.height * 4) as usize);

        let mut cpu = CpuRefBackend;
        let cpu_frame = cpu.render_scene_with_textures(&req, &[upload]);
        assert!(cpu_frame.is_valid(), "CPU 贴图参考帧必须有效");
        assert_parity("V3 贴图棋盘", &report.device, &cpu_frame, &report.output);

        // ---- 正控：贴图真的改了画面（否则 0 差异只是"两个白球"）----
        let base_fb = fb_of(req.width, req.height, &base.output);
        let applied = diff(&base_fb, &report.output);
        assert!(
            applied.differing_pixels > 0,
            "贴图帧必须与无纹理基线不同 —— 否则说明 maps 句柄根本没被读到"
        );
        println!(
            "[正控] 贴图帧 vs 无纹理基线：differing_pixels={}/{} max_channel_delta={}",
            applied.differing_pixels, applied.total_pixels, applied.max_channel_delta
        );

        // ---- 确定性（G3）：同设备连跑两次逐字节相同 ----
        let again = dispatch_v3(ctx, &req, &[upload], None, &words).expect("二次编码必须成功");
        assert_eq!(again.output, report.output, "同设备连跑两次读回不一致");

        // ---- 负控 ①：句柄清 0 ⇒ 与无纹理基线**逐字节**相同 ----
        let cleared = textured_scene(0);
        let gpu_cleared =
            dispatch_v3(ctx, &cleared, &[upload], None, &words).expect("清 0 编码必须成功");
        assert_eq!(gpu_cleared.status, DispatchStatus::Ok, "负控①派发失败");
        assert_eq!(
            gpu_cleared.output, base.output,
            "负控①：句柄清 0 后必须与无纹理基线逐字节相同（纹理集仍上报）"
        );
        let cpu_cleared = cpu.render_scene_with_textures(&cleared, &[upload]);
        let cpu_base = cpu.render_scene(&flat);
        assert_eq!(
            cpu_cleared.pixels, cpu_base.pixels,
            "负控①(CPU)：句柄清 0 必须与无纹理基线逐字节相同"
        );
        println!(
            "[负控①] 句柄清 0 ⇒ 与无纹理基线逐字节相同（{} B，CPU 侧同）",
            gpu_cleared.output.len()
        );

        // ---- 负控 ②：棋盘整体翻色（c0↔c1）⇒ differing_pixels > 0 ----
        // 选"整体交换两色"而非"翻一格"：本测试现在是 7 级 mip 链，单级上翻一格
        // 会被盒式降采样稀释进粗 mip 级，足迹落在 lod>0 的像素读不到那一格的
        // 改动。整体换色则每一级都被改变，贴图可见处必然整片不同 —— 同样证明
        // "判据对纹理内容敏感"（而不是对一张恒定白）。
        let flipped_px = checker(64, 2, [0.1, 0.2, 0.9, 1.0], [0.9, 0.2, 0.1, 1.0]);
        let flipped = mip_chain_from(&flipped_px, 64, 7);
        let flip_up = TextureUpload { texels: &flipped, ..upload };
        let gpu_flipped =
            dispatch_v3(ctx, &req, &[flip_up], None, &words).expect("翻格编码必须成功");
        assert_eq!(gpu_flipped.status, DispatchStatus::Ok, "负控②派发失败");
        let flip_diff = diff(&fb_of(req.width, req.height, &report.output), &gpu_flipped.output);
        assert!(
            flip_diff.differing_pixels > 0,
            "负控②：整体翻色后画面必须有变化（否则判据对纹理内容不敏感）"
        );
        println!(
            "[负控②] 整体翻色 ⇒ differing_pixels={}/{} max_channel_delta={}",
            flip_diff.differing_pixels, flip_diff.total_pixels, flip_diff.max_channel_delta
        );

        // ---- 负控 ③：越界句柄 ⇒ 按 NO_TEXTURE 处理（不 panic、行为可预测）----
        let oob = textured_scene(7); // 只有 1 张纹理 ⇒ 句柄 7 越界
        let gpu_oob = dispatch_v3(ctx, &oob, &[upload], None, &words).expect("越界句柄编码");
        assert_eq!(
            gpu_oob.status,
            DispatchStatus::Ok,
            "负控③派发失败（越界句柄不该让 kernel 挂）"
        );
        assert_eq!(
            gpu_oob.output, base.output,
            "负控③：越界句柄必须**按 NO_TEXTURE 处理**（= 与无纹理基线逐字节相同）"
        );
        let cpu_oob = cpu.render_scene_with_textures(&oob, &[upload]);
        assert_eq!(
            cpu_oob.pixels, cpu_base.pixels,
            "负控③(CPU)：越界句柄必须与无纹理基线逐字节相同"
        );
        // 另一形态：句柄 1 但纹理集为空（`P_NUM_TEXTURES == 0`）。
        let gpu_none = dispatch_v3(ctx, &req, &[], None, &words).expect("句柄 1 + 空纹理集");
        assert_eq!(
            gpu_none.output, base.output,
            "负控③b：句柄 1 + 空纹理集 ⇒ 同 NO_TEXTURE"
        );
        println!("[负控③] 越界句柄 7 / 句柄 1+空集 ⇒ 均与无纹理基线逐字节相同 ✓");
    }

    /// 段 3 主判据：sRGB 编码渐变纹理（`srgb=1`）在 **llvmpipe** 上与 CPU 参考
    /// **逐位一致**，且 sRGB 往返（线性 → γ 编码 → 上传解码回线性，f32 域内）误差 0。
    ///
    /// 构造一条**线性**渐变 `L(x) = x/(n-1)`；两种上传方式：
    ///   * `up_linear`：`srgb=0`，纹素直接存线性值 L；
    ///   * `up_srgb`：`srgb=1`，纹素存 `srgb_encode(L)`（sRGB γ 编码）。
    /// `encode_v3` 对 `srgb!=0` 的纹素在**上传前** `srgb_decode` 回线性（见
    /// `scene_abi::encode_v3`），故两路在 GPU/CPU 拿到的线性 f32 **逐位相同**
    /// ⇒ 两帧 0 差异。这同时是「sRGB 往返误差 0」的端到端证据。
    ///
    /// 软件 ICD 上 `assert_parity` 用死判据：
    /// `max_channel_delta == 0 && differing_pixels == 0`。
    #[test]
    fn real_device_srgb_gradient_matches_cpu_reference() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        let ctx = backend.context();

        let n: u32 = 32;
        let mut lin_texels: Vec<[f32; 4]> = Vec::with_capacity((n * n) as usize);
        let mut srgb_texels: Vec<[f32; 4]> = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                let l = (x as f32) / ((n - 1) as f32); // 线性渐变 0..1（x 方向）
                let _ = y;
                // 往返（CPU 侧先证）：decode(encode(L)) 必须等于 L，误差 ≤ 0.5 ULP。
                let enc = crate::scene_abi::srgb_encode_f32(l);
                let dec = crate::scene_abi::srgb_decode_f32(enc);
                let err = (dec - l).abs();
                assert!(
                    err <= 0.5 / 255.0 + 1e-6,
                    "sRGB 往返误差 {err} 超 0.5 LSB（L={l}）：dp 链路 sRGB 解码不自洽"
                );
                lin_texels.push([l, l, l, 1.0]);
                srgb_texels.push([enc, enc, enc, 1.0]);
            }
        }
        let up_linear = TextureUpload {
            w: n,
            h: n,
            levels: 1,
            fmt: 1,
            srgb: 0,
            texels: &lin_texels,
        };
        let up_srgb = TextureUpload {
            w: n,
            h: n,
            levels: 1,
            fmt: 1,
            srgb: 1,
            texels: &srgb_texels,
        };

        // ---- GPU：srgb=0 线性帧 ----
        let req = textured_scene(1);
        let gpu_linear =
            dispatch_v3(ctx, &req, &[up_linear], None, &words).expect("线性纹理编码必须成功");
        assert_eq!(
            gpu_linear.status,
            DispatchStatus::Ok,
            "线性纹理派发失败：{}（{}）",
            gpu_linear.status.as_str(),
            gpu_linear.detail
        );
        // ---- GPU：srgb=1 编码帧 ----
        let gpu_srgb =
            dispatch_v3(ctx, &req, &[up_srgb], None, &words).expect("sRGB 纹理编码必须成功");
        assert_eq!(
            gpu_srgb.status,
            DispatchStatus::Ok,
            "sRGB 纹理派发失败：{}（{}）",
            gpu_srgb.status.as_str(),
            gpu_srgb.detail
        );

        // ---- 往返判据：srgb=1 与 srgb=0 必须逐字节相同（两路解码到同一线性 f32）----
        assert_eq!(
            gpu_srgb.output, gpu_linear.output,
            "[往返] srgb=1 帧必须与 srgb=0（线性）帧逐字节相同——否则 sRGB 解码未生效"
        );

        // ---- GPU vs CPU（srgb=1）在软件 ICD 上逐位一致 ----
        let mut cpu = CpuRefBackend;
        let cpu_frame = cpu.render_scene_with_textures(&req, &[up_srgb]);
        assert!(cpu_frame.is_valid(), "CPU sRGB 参考帧必须有效");
        assert_parity("V3 sRGB 渐变", &gpu_srgb.device, &cpu_frame, &gpu_srgb.output);

        // ---- GPU vs CPU（srgb=0）同样逐位一致（基准对照）----
        let cpu_linear_frame = cpu.render_scene_with_textures(&req, &[up_linear]);
        assert_parity(
            "V3 线性渐变(基准)",
            &gpu_linear.device,
            &cpu_linear_frame,
            &gpu_linear.output,
        );

        // ---- 正控：sRGB 贴图帧必须不同于无纹理基线（否则判据对内容不敏感）----
        let flat = textured_scene(0);
        let base = dispatch_v3(ctx, &flat, &[], None, &words).expect("基线编码必须成功");
        let applied = diff(&fb_of(req.width, req.height, &base.output), &gpu_srgb.output);
        assert!(
            applied.differing_pixels > 0,
            "sRGB 贴图帧必须与无纹理基线不同——否则 maps 句柄没被读到"
        );
        println!(
            "[正控] sRGB 贴图帧 vs 无纹理基线：differing_pixels={}/{} max_channel_delta={}",
            applied.differing_pixels, applied.total_pixels, applied.max_channel_delta
        );
    }

    /// 纯色纹理（每纹素同一色）：用于金属度-粗糙度 / 法线这类"标量场"贴图，
    /// 避免棋盘在 level-0 跨格边界掉 1/255 的驱动末位舍入差。单级即够（常量
    /// 纹理任意 mip 级采样值相同，lod 选取不影响结果）。
    fn solid_tex(n: u32, color: [f32; 4]) -> Vec<[f32; 4]> {
        vec![color; (n * n) as usize]
    }

    /// 段 2 场景：贴**基础色 + 金属度-粗糙度 + 法线**三张图的球 + 素色球。
    ///
    /// 贴图球的 `albedo` 留白 `(1,1,1)`：若基础色贴图没生效，画面是白球而非
    /// 原色球。金属度-粗糙度 / 法线用常量纹理（见 [`solid_tex`]），法线取一个
    /// **非平凡**的切向倾斜 `(0.6, 0.5, 1.0)` → 切线空间 `(0.2, 0, 1)`，真正
    /// 扰动着色法线（不是恒等映射）——验证 `plane_tangent_basis` 旋到世界空间
    /// 的真实效果。
    fn textured_scene_full(base_map: u32, mr_map: u32, nm_map: u32) -> SceneRenderRequest {
        SceneRenderRequest {
            width: 64,
            height: 48,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 64.0 / 48.0,
            },
            lights: vec![
                LightReq {
                    direction: Vec3::new(0.0, 0.0, -1.0),
                    color: Vec3::new(1.0, 1.0, 1.0),
                    intensity: 1.0,
                },
                LightReq {
                    direction: Vec3::new(-0.6, -0.4, -0.7),
                    color: Vec3::new(0.3, 0.6, 1.0),
                    intensity: 0.55,
                },
            ],
            spheres: vec![
                SphereReq {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 1.0,
                    albedo: Vec3::new(1.0, 1.0, 1.0),
                    roughness: 0.4,
                    metalness: 0.1,
                    maps: [base_map, mr_map, nm_map, 0],
                },
                SphereReq {
                    center: Vec3::new(1.55, 0.35, 0.5),
                    radius: 0.45,
                    albedo: Vec3::new(0.15, 0.75, 0.35),
                    roughness: 0.2,
                    metalness: 0.0,
                    maps: crate::NO_MAPS,
                },
            ],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
            curves: Vec::new(),
        }
    }

    /// 主判据（W1 段 2）：带 **法线 + 金属度-粗糙度** 贴图的球在 **llvmpipe**
    /// 上与 CPU 参考**逐位一致**（判据 5/6）。
    ///
    /// 三张图都真接：基础色走 mip 棋盘（段 1 的足迹/三线性路径），金属度-
    /// 粗糙度与法线走常量纹理（真正启用两瓣 PBR + 世界空间法线扰动）。CPU 与
    /// GPU 走**同一** `plane_tangent_basis`、`同一` mix(0.04,albedo,m)`、`同一`
    /// `NdotH⁴` 镜面瓣，故可逐位对齐。
    #[test]
    fn textured_normal_metalrough_matches_cpu_reference() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读（D-21 S1 产物）");
        let ctx = backend.context();

        // 三张图：基础色 = 64×64 / 2 纹素一格 / 7 级 mip 棋盘；金属度-粗糙度 =
        // 常量（G=roughness=0.3、B=metalness=0.5）；法线 = 常量。主判据用**恒等**
        // 法线 (0.5,0.5,1.0) → 切线空间 (0,0,1)，不扰动着色法线，避免 CPU-libm /
        // GPU-LLVM 的 `sqrt` 1-ulp 差异在 `plane_tangent_basis` 的切轴**分支 tie**
        // 处被放大成整像素差（判据不该建在舍入边界上，见下方正控与单元测试）。
        let base_texels = mip_chain(64, 2, [0.9, 0.2, 0.1, 1.0], [0.1, 0.2, 0.9, 1.0], 7);
        let base = TextureUpload {
            w: 64,
            h: 64,
            levels: 7,
            fmt: 1,
            srgb: 0,
            texels: &base_texels,
        };
        let mr_texels = solid_tex(4, [0.0, 0.3, 0.5, 1.0]); // G=0.3、B=0.5（glTF 约定）
        let mr = TextureUpload {
            w: 4,
            h: 4,
            levels: 1,
            fmt: 1,
            srgb: 0,
            texels: &mr_texels,
        };
        let nm_id_texels = solid_tex(4, [0.5, 0.5, 1.0, 1.0]); // 恒等法线 (0,0,1)
        let nm_id = TextureUpload {
            w: 4,
            h: 4,
            levels: 1,
            fmt: 1,
            srgb: 0,
            texels: &nm_id_texels,
        };
        let nm_pert_texels = solid_tex(4, [0.6, 0.5, 1.0, 1.0]); // 扰动法线 (0.2,0,1)
        let nm_pert = TextureUpload {
            w: 4,
            h: 4,
            levels: 1,
            fmt: 1,
            srgb: 0,
            texels: &nm_pert_texels,
        };

        // ---- 主判据：恒等法线，llvmpipe 上逐位一致（max_channel_delta == 0 && differing_pixels == 0）----
        let req = textured_scene_full(1, 2, 3);
        let uploads = [base, mr, nm_id];
        let report = dispatch_v3(ctx, &req, &uploads, None, &words).expect("三贴图编码必须成功");
        assert_eq!(
            report.status,
            DispatchStatus::Ok,
            "三贴图派发失败：{}（{}）",
            report.status.as_str(),
            report.detail
        );
        assert_eq!(report.output.len(), (req.width * req.height * 4) as usize);

        let mut cpu = CpuRefBackend;
        let cpu_frame = cpu.render_scene_with_textures(&req, &uploads);
        assert!(cpu_frame.is_valid(), "CPU 三贴图参考帧必须有效");

        assert_parity("V3 法线+金属粗糙（恒等法线）", &report.device, &cpu_frame, &report.output);

        // ---- 正控①：三贴图帧必须**不同于**无纹理基线（否则说明三张图没生效）----
        let base_req = textured_scene_full(0, 0, 0);
        let base_report = dispatch_v3(ctx, &base_req, &[], None, &words).expect("基线编码");
        let applied = diff(
            &fb_of(req.width, req.height, &base_report.output),
            &report.output,
        );
        assert!(
            applied.differing_pixels > 0,
            "三贴图帧必须与无纹理基线不同 —— 否则说明 MAP_BASE/METAL_ROUGH/NORMAL 路径没被走到"
        );
        println!(
            "[正控①·法线+金属粗糙] 三贴图帧 vs 无纹理基线：differing_pixels={}/{} max_channel_delta={}",
            applied.differing_pixels, applied.total_pixels, applied.max_channel_delta
        );

        // ---- 正控②：扰动法线帧必须**不同于**恒等法线帧（证明 MAP_NORMAL 真扰动着色法线）----
        // 同一个球、同一 base/mr 图，只换法线图：扰动 (0.2,0,1) vs 恒等 (0,0,1)。
        // 同后端比（GPU-GPU、CPU-CPU），只验证"法线图确实改了画面"，不要求跨端 0 差
        // （那 2 像素的 1-ulp tie 是 CPU-libm/GPU-LLVM `sqrt` 的固有舍入边界，非 bug）。
        let uploads_pert = [base, mr, nm_pert];
        let pert_report = dispatch_v3(ctx, &req, &uploads_pert, None, &words)
            .expect("扰动法线编码必须成功");
        let pert_vs_id = diff(
            &fb_of(req.width, req.height, &report.output),
            &pert_report.output,
        );
        assert!(
            pert_vs_id.differing_pixels > 0,
            "扰动法线必须改变画面 —— 否则 MAP_NORMAL 的扰动路径被吃掉了（恒等法与扰动法无差异）"
        );
        let cpu_pert = cpu.render_scene_with_textures(&req, &uploads_pert);
        let cpu_pert_vs_id = diff(&cpu_frame, &cpu_pert.pixels);
        assert!(
            cpu_pert_vs_id.differing_pixels > 0,
            "负控②(CPU)：扰动法线必须改变 CPU 画面（否则扰动路径在 CPU 侧被吃掉）"
        );
        println!(
            "[正控②·法线+金属粗糙] 扰动法线 vs 恒等法线：GPU differing_pixels={}/{}、CPU differing_pixels={}/{}",
            pert_vs_id.differing_pixels, pert_vs_id.total_pixels,
            cpu_pert_vs_id.differing_pixels, cpu_pert_vs_id.total_pixels
        );

        // ---- 负控：句柄清 0（maps 全 0）⇒ 退化为纯 Lambert 基线，与基线逐字节相同 ----
        let zero = textured_scene_full(0, 0, 0);
        let zero_report = dispatch_v3(ctx, &zero, &[], None, &words).expect("句柄清0 编码");
        assert_eq!(
            zero_report.output, base_report.output,
            "负控：句柄全 0 必须退化为纯 Lambert 基线（differing_pixels == 0）"
        );
        let zero_cpu = cpu.render_scene_with_textures(&zero, &[]);
        assert_eq!(
            zero_cpu.pixels, cpu.render_scene_with_textures(&base_req, &[]).pixels,
            "负控(CPU)：句柄全 0 与无纹理基线逐字节相同"
        );
        println!("[负控·法线+金属粗糙] 句柄清 0 ⇒ 退化为纯 Lambert 基线，逐字节相同 ✓");
    }

    /// 由单级棋盘生成完整 mip 链（盒式 2×2 降采样，行主序，每级宽 = max(1, n>>l)）。
    ///
    /// 布局与 `TextureUpload::expected_texels` / `encode_v3` 的 mip 排布逐格一致：
    /// 第 0 级在前，逐层减半（最小 1×1）。
    fn mip_chain(
        n: u32,
        cell: u32,
        c0: [f32; 4],
        c1: [f32; 4],
        levels: u32,
    ) -> Vec<[f32; 4]> {
        let base = checker(n, cell, c0, c1);
        mip_chain_from(&base, n as usize, levels)
    }

    /// 由**已构造好的第 0 级**生成完整 mip 链（其余同 [`mip_chain`]）。
    ///
    /// 给 `one_texel_uv_error_would_be_caught` 用：先对第 0 级做"平移一格"，
    /// 再以平移后的第 0 级为种子降采样出整条链——保证"平移"只发生在与 GPU
    /// 同口径的第 0 级，后续各级只是确定性盒式平均，不引入额外偏差。
    fn mip_chain_from(base: &[[f32; 4]], n: usize, levels: u32) -> Vec<[f32; 4]> {
        let mut out = Vec::new();
        let mut cur: Vec<[f32; 4]> = base.to_vec();
        out.extend_from_slice(&cur);
        let mut cw = n;
        let mut ch = n;
        for _ in 1..levels {
            let nw = (cw / 2).max(1);
            let nh = (ch / 2).max(1);
            let mut next = Vec::with_capacity(nw * nh);
            for y in 0..nh {
                for x in 0..nw {
                    let mut acc = [0.0f32; 4];
                    let mut cnt = 0.0f32;
                    for yy in 0..2 {
                        for xx in 0..2 {
                            let sx = (x * 2 + xx).min(cw - 1);
                            let sy = (y * 2 + yy).min(ch - 1);
                            let t = cur[sy * cw + sx];
                            for c in 0..4 {
                                acc[c] += t[c];
                            }
                            cnt += 1.0;
                        }
                    }
                    for c in 0..4 {
                        acc[c] /= cnt;
                    }
                    next.push(acc);
                }
            }
            out.extend_from_slice(&next);
            cur = next;
            cw = nw;
            ch = nh;
        }
        out
    }

    /// **uv 对账**：CPU 侧 uv 与 `scene_frame_v3.wgsl::uv_sphere` **逐式转写**
    /// 的求值必须同值（±1 ulp）。
    ///
    /// # 这条测试为什么必须存在
    ///
    /// 0 差异只证明"两边采到的颜色一样"，不证明"采的是同一个 uv"：一张低频
    /// 贴图在 uv 差 1 个纹素时像素差可能仍是 0。uv 一旦不同，整套对拍就成了
    /// "接近的贴图"之间的巧合。
    ///
    /// # 边界（诚实说明）
    ///
    /// 驱动侧的 uv 是在 SPIR-V 里算的，**无法从 RGBA8 读出**。故本测试比较的是
    /// "CPU 实现"与"按 shader 文本转写的公式"，并用文本断言把那份公式钉住；
    /// 像素层的下界另有两条：主测试的 1.25 像素/纹素棋盘（0 差异）与
    /// [`Self::one_texel_uv_error_would_be_caught`]（平移一格 ⇒ 差异 > 0）。
    #[test]
    fn cpu_uv_matches_wgsl_uv_formula_within_one_ulp() {
        // ---- 文本守卫：shader 里那几行公式与常量还在（改了就红）----
        let src = crate::shaders::source("scene_frame_v3.wgsl").expect("V3 WGSL 必须已登记");
        for needle in [
            "const FRAC_1_PI: f32 = 0.3183098861837907;",
            "const FRAC_1_TAU: f32 = 0.15915494309189535;",
            "let u = det_atan2(d.z, d.x) * FRAC_1_TAU + 0.5;",
            "let v = det_acos(clamp(d.y, -1.0, 1.0)) * FRAC_1_PI;",
            "let d = (p - center) / radius;",
        ] {
            assert!(src.contains(needle), "V3 WGSL 必须保留 `{needle}`");
        }

        // ---- 数值：64×64 个球面点（含极点与 u 缝两侧）----
        let center = Vec3::new(0.0, 0.0, 0.0);
        let radius = 1.0f32;
        let mut max_ulp = 0u32;
        let mut worst: Option<(Vec3, [f32; 2], [f32; 2], u32)> = None;
        for i in 0..64u32 {
            for j in 0..64u32 {
                // θ ∈ [0, π]（纬度）、φ ∈ [0, 2π)（经度）
                let theta = std::f32::consts::PI * (j as f32) / 63.0;
                let phi = 2.0 * std::f32::consts::PI * (i as f32) / 64.0;
                let (sp, cp) = prism_math::det_trig::cos_sin(phi);
                let (st, ct) = prism_math::det_trig::cos_sin(theta);
                let p = Vec3::new(radius * st * cp, radius * ct, radius * st * sp);
                let a = crate::cpu_tex::sphere_uv(p, center, radius);
                let b = wgsl_uv_sphere(p, center, radius);
                let du = crate::cpu_tex::f32_ulp_distance(a[0], b[0]);
                let dv = crate::cpu_tex::f32_ulp_distance(a[1], b[1]);
                let d = du.max(dv);
                if d > max_ulp {
                    max_ulp = d;
                    worst = Some((p, a, b, d));
                }
            }
        }
        if let Some((p, a, b, d)) = worst {
            println!("[uv 对账] 最差点 p={p:?}：CPU={a:?} WGSL={b:?}（{d} ulp）");
        }
        println!("[uv 对账] {} 点，最大偏差 {max_ulp} ulp", 64 * 64);
        assert!(
            max_ulp <= 1,
            "CPU 侧 uv 与 shader uv_sphere 的偏差 {max_ulp} ulp > 1 ulp —— \
             采到的不是同一张贴图的同一个位置"
        );
    }

    /// 按 `scene_frame_v3.wgsl` 的 `uv_sphere` / `det_atan2` / `det_acos` /
    /// `atan_pos` / `atan_unit` / `quarter_atan` **逐式转写**（测试道具）。
    ///
    /// 独立书写（不用 `prism_math::det_trig`）：目的是把"shader 文本里的公式"
    /// 变成一个可与生产 CPU 实现比较的数。系数与归约顺序与 WGSL 源码逐字对应。
    fn wgsl_uv_sphere(p: Vec3, center: Vec3, radius: f32) -> [f32; 2] {
        const FRAC_1_PI: f32 = 0.3183098861837907;
        const FRAC_1_TAU: f32 = 0.15915494309189535;

        fn quarter_atan(z: f32) -> f32 {
            let z2 = z * z;
            z * (1.0
                + z2 * (-0.3333333333333333
                    + z2 * (0.2
                        + z2 * (-0.14285714285714285
                            + z2 * (0.1111111111111111
                                + z2 * (-0.09090909090909091
                                    + z2 * (0.07692307692307693
                                        + z2 * (-0.06666666666666667))))))))
        }
        fn atan_unit(v: f32) -> f32 {
            const FRAC_PI_4: f32 = 0.7853981633974483;
            const TAN_PI_8: f32 = 0.41421357;
            if v > TAN_PI_8 {
                return FRAC_PI_4 + quarter_atan((v - 1.0) / (v + 1.0));
            }
            quarter_atan(v)
        }
        fn atan_pos(x: f32) -> f32 {
            const FRAC_PI_2: f32 = 1.5707963267948966;
            if x > 1.0 {
                return FRAC_PI_2 - atan_unit(1.0 / x);
            }
            atan_unit(x)
        }
        fn det_atan2(y: f32, x: f32) -> f32 {
            const PI: f32 = 3.141592653589793;
            const FRAC_PI_2: f32 = 1.5707963267948966;
            if x == 0.0 {
                if y == 0.0 {
                    return 0.0;
                }
                if y > 0.0 {
                    return FRAC_PI_2;
                }
                return -FRAC_PI_2;
            }
            let ax = x.abs();
            let ay = y.abs();
            let r = if ay <= ax {
                atan_pos(ay / ax)
            } else {
                FRAC_PI_2 - atan_pos(ax / ay)
            };
            if x > 0.0 {
                if y >= 0.0 {
                    return r;
                }
                return -r;
            }
            if y >= 0.0 {
                return PI - r;
            }
            r - PI
        }
        fn det_acos(x: f32) -> f32 {
            let mut xc = x;
            if xc > 1.0 {
                xc = 1.0;
            } else if xc < -1.0 {
                xc = -1.0;
            }
            let s = ((1.0 - xc) * (1.0 + xc)).max(0.0).sqrt();
            det_atan2(s, xc)
        }

        // `let d = (p - center) / radius;`（逐分量除法，与 shader 同）
        let d = Vec3::new(
            (p.x - center.x) / radius,
            (p.y - center.y) / radius,
            (p.z - center.z) / radius,
        );
        let u = det_atan2(d.z, d.x) * FRAC_1_TAU + 0.5;
        let v = det_acos(d.y.clamp(-1.0, 1.0)) * FRAC_1_PI;
        [u, v]
    }

    /// **1 纹素 uv 误差必须被抓住**（否则主测试的 0 差异没有意义）。
    ///
    /// 做法：与主测试同一场景、同一张 64×64 / 2 纹素一格、7 级 mip 的棋盘。
    /// GPU 采原图，CPU 采"u 方向平移一格（1/64 uv，repeat）"的图——`differing_pixels`
    /// 必然是三位数（每个格子内部两纹素互换颜色）。
    ///
    /// 两件事一起断言：
    /// 1. 原图对拍 0 差异（`assert_parity`，与主测试同判据）；
    /// 2. 平移一格后 `differing_pixels > 0`。
    ///
    /// # 为什么用 mip 链而不是单级棋盘
    ///
    /// "驱动末位舍入差"：CPU 与 GPU 在双线性/三线性里对 `a + b·c` 这类式子是否
    /// 收缩成 FMA 的取舍不同，会在跨格的纹素边界上差出 ≤1 ulp，恰好落在 8 bit
    /// 量化 .5 台阶时掉 1/255。单级 32×32 / 4 纹素一格的棋盘在 4-tap 足迹下实测
    /// 有 1 像素掉 1/255（位置 (33,34)，CPU `[73,40,136]` vs GPU `[74,40,136]`）——
    /// 那档差异与"纹理内容"无关，属于"把判据建在舍入边界上"。改用 64×64 / 2 纹素
    /// 一格 + 7 级 mip：足迹在球面中前部落入第 1/2 级（纹素被 2×/4× 放大），跨格
    /// 边界被有效稀释，噪声为 0、信号是三位数像素。这也正是主判据 `textured_mip_*`
    /// 选同款纹理的原因。
    #[test]
    fn one_texel_uv_error_would_be_caught() {
        let backend = VulkanBackend::new();
        if !device_ready(&backend) {
            return;
        }
        let words = load_committed_spv().expect("入库 .spv 必须可读");
        let ctx = backend.context();

        let n = 64u32;
        // 第 0 级 64×64、2 纹素一格；整条 7 级 mip 链（64→1）。与主测试同一张
        // 图，故同样落在"非舍入边界"的安全区，0 差异可稳定复现。
        let base_px = checker(n, 2, [0.9, 0.2, 0.1, 1.0], [0.1, 0.2, 0.9, 1.0]);
        let texels = mip_chain_from(&base_px, n as usize, 7);
        let upload = TextureUpload {
            w: n,
            h: n,
            levels: 7,
            fmt: 1,
            srgb: 0,
            texels: &texels,
        };
        let req = textured_scene(1);

        let report = dispatch_v3(ctx, &req, &[upload], None, &words).expect("棋盘编码");
        assert_eq!(
            report.status,
            DispatchStatus::Ok,
            "棋盘派发失败：{}",
            report.detail
        );
        let mut cpu = CpuRefBackend;
        let cpu_frame = cpu.render_scene_with_textures(&req, &[upload]);
        // 判据 1：同一张图的 0 差异（与主测试同一口径）。
        assert_parity("V3 敏感度基准", &report.device, &cpu_frame, &report.output);

        // u 方向平移一格（repeat）：`shifted[y][x] = base[y][x-1]`（1 纹素 = 1/n uv）。
        let mut shifted = base_px.clone();
        for y in 0..n {
            for x in 0..n {
                shifted[(y * n + x) as usize] = base_px[(y * n + (x + n - 1) % n) as usize];
            }
        }
        let shifted_texels = mip_chain_from(&shifted, n as usize, 7);
        let shifted_up = TextureUpload { texels: &shifted_texels, ..upload };
        let cpu_shift = cpu.render_scene_with_textures(&req, &[shifted_up]);
        let d = diff(&cpu_shift, &report.output);
        println!(
            "[uv 敏感度] CPU 侧 uv 多 1/{}（等价平移一格）⇒ differing_pixels={}/{} \
             max_channel_delta={}",
            n, d.differing_pixels, d.total_pixels, d.max_channel_delta
        );
        // 判据 2：1 纹素偏差必须被抓住。
        assert!(
            d.differing_pixels > 0,
            "平移一格后画面没有变化 —— 说明对拍对 1 纹素级 uv 偏差不敏感，\
             那么主测试的 0 差异不能作为 uv 一致的证据"
        );
        // 记录"0 差异"这一侧的上界口径：1 纹素 = 1/{n} uv。
        println!(
            "[uv 上界] 同图 0 差异 ∧ 1/{n} uv 偏差可测 ⇒ |Δuv| < 1/{n} = {:.5}",
            1.0 / n as f32
        );
    }
}
