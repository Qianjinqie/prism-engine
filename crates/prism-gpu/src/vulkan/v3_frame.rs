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
    self, EnvCdf, SceneAbi, TextureUpload, V3Buffers, LIGHT_SLOTS, NODE_SLOTS, PARAMS_WORDS_V3,
    SPHERE_SLOTS, TRIANGLE_SLOTS, V3_BINDINGS,
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
        .checked_add(num_spheres as u64 * SPHERE_SLOTS as u64 * 4)?
        .checked_add(num_lights as u64 * LIGHT_SLOTS as u64 * 4)?
        .checked_add(num_tris as u64 * TRIANGLE_SLOTS as u64 * 4)?
        .checked_add(num_nodes as u64 * NODE_SLOTS as u64 * 4)?
        .checked_add(texel_bytes as u64)?
        .checked_add(texmeta_bytes as u64)?
        .checked_add(cdf_bytes as u64)?;
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
                (num_spheres.max(1) as u64 * SPHERE_SLOTS as u64).min(u32::MAX as u64) as u32,
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
            // b8 输出（每像素一个打包 RGBA8 u32）——**恒为最后一个绑定**
            BufferSpec::new(8, ElementType::U32, pixels as u32, BindingAccess::WriteOnly),
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
    let buffers: V3Buffers = SceneAbi::encode_v3(req, textures, env)?;
    let inputs = buffers.inputs();
    let counts = V3Counts {
        spheres: req.spheres.len(),
        lights: req.lights.len(),
        tris: buffers.triangles.len() / (TRIANGLE_SLOTS as usize * 4),
        nodes: buffers.nodes.len() / (NODE_SLOTS as usize * 4),
        texel_bytes: buffers.textures.len(),
        texmeta_bytes: buffers.texmeta.len(),
        cdf_bytes: buffers.cdf.len(),
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
    debug_assert!(kernel.validate().is_ok(), "V3 绑定表必须自洽（9 连续前缀）");
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
                },
                SphereReq {
                    center: Vec3::new(1.35, 0.35, 0.4),
                    radius: 0.55,
                    albedo: Vec3::new(0.15, 0.75, 0.35),
                    roughness: 0.2,
                    metalness: 0.0,
                },
                SphereReq {
                    center: Vec3::new(-1.25, -0.45, 0.2),
                    radius: 0.45,
                    albedo: Vec3::new(0.2, 0.3, 0.9),
                    roughness: 0.6,
                    metalness: 0.3,
                },
                SphereReq {
                    center: Vec3::new(0.15, -0.95, 1.1),
                    radius: 0.32,
                    albedo: Vec3::new(0.9, 0.85, 0.2),
                    roughness: 0.5,
                    metalness: 0.0,
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

        // ---- 场景 4：纹理集存在但不可达（pad 槽恒 0）⇒ 输出必须仍是 0 差异 ----
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
        assert_parity("V3 带纹理集(不可达)", &report.device, &cpu_frame, &report.output);
        println!(
            "[V3 带纹理集] 纹理条数=1，纹素 {} B，texmeta {} B —— 材质未引用，输出与无纹理逐位相同 ✓",
            texels.len() * 16,
            32
        );
    }

    /// kernel 描述的结构自洽（**无设备也能跑**）：9 绑定连续前缀、输出在最后、
    /// 元素容量与 `V3_BINDINGS` 对齐。这条防的是"改绑定表忘了改 kernel 描述"。
    #[test]
    fn v3_kernel_layout_matches_abi() {
        let k = v3_kernel(16, 9, 2, 1, 3, 5, 21 * 16, 2 * 32, 36).expect("合法尺寸");
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
        let empty = v3_kernel(4, 4, 0, 0, 0, 0, 0, 0, 0).expect("空场景也合法");
        let want = [
            scene_abi::V3_SPHERES,
            scene_abi::V3_LIGHTS,
            scene_abi::V3_TRIANGLES,
            scene_abi::V3_NODES,
            scene_abi::V3_TEXTURES,
            scene_abi::V3_TEXMETA,
            scene_abi::V3_CDF,
        ];
        let slots = [
            SPHERE_SLOTS,
            LIGHT_SLOTS,
            TRIANGLE_SLOTS,
            NODE_SLOTS,
            1,
            1,
            1,
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
        assert!(v3_kernel(0, 8, 1, 1, 0, 0, 0, 0, 0).is_none());
        assert!(v3_kernel(crate::MAX_FRAMEBUFFER_DIM + 1, 8, 1, 1, 0, 0, 0, 0, 0).is_none());
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
        )
        .expect("曲线场景的 kernel 描述必须合法");
        assert_eq!(k.buffers[3].elements as usize, num_tris * TRIANGLE_SLOTS as usize);
    }
}
