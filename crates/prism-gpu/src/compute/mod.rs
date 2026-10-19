//! 设备无关的 compute 语义层（V2.0-G GPU compute-first 管线的基础件）。
//!
//! # 为什么要「语义层」先行
//!
//! 真机 Vulkan/Metal/WebGPU 三条路径的 **kernel 语义完全相同**，差别只在
//! 「谁来执行同一组 invocation」。把语义抽成设备无关的纯数据变换之后：
//!
//! * 引擎在**无 GPU 的构建机**上也能真跑真测 kernel 正确性（本仓库 CI 沙箱
//!   只有 loader、无 ICD，见 `crate::vulkan` 的诚实边界）；
//! * 真机 backend 只负责把 [`DispatchShape`] 三个组数喂给 `vkCmdDispatch`，
//!   不存在第二份形状/布局口径（工程四律 1/4：「绝不私定第二套口径」）；
//! * parity 判对（`prism-rt::wavefront` 的 double gate）以本层的
//!   `dispatch_cpu` 序为「CPU 模拟 GPU 数组序」的权威定义。
//!
//! # 模块构成
//!
//! * [`dispatch`]：workgroup 尺寸 / 组数展开 / CPU 等价执行器 / 尾组浪费账；
//! * [`layout`]：std430 子集 buffer 布局 + [`KernelDesc`] 结构校验；
//! * 本文件：**两个真实 kernel 语义**的 CPU 参考实现 + kernel 注册表：
//!   1. [`primary_ray_gen_cpu`] —— RT 主射线发射（相机 → 逐像素射线）；
//!   2. [`wavefront_intersect_step_cpu`] —— wavefront 求交一步（射线批 → 命中批）。
//!
//! # 与既有 `scan.rs` / `sort.rs` / `sync.rs` 的同构
//!
//! 三者的共同模式是「**设备无关语义 + CPU 参考真跑 + 真机路径标注待接入**」：
//! `scan` 用 Blelloch 双扫仿真 GPU 前缀和、`sort` 用 radix 分桶仿真全局排序、
//! `sync` 用纯数据屏障推导仿真 GPU 同步；本模块用 `dispatch_cpu` 仿真 GPU
//! invocation 序，走同一条设计约束（D-15 自研抽象层 / D-21 先数学后 backend）。
//!
//! # 禁 libm（D-29）
//!
//! 本模块只用四则运算与 `sqrt`（IEEE 精确舍入），**不调用**任何超越函数；
//! 三角量（`tan_half_fov`）由调用方一次算好传入，避免 GPU/CPU 三角实现差异
//! 污染判对。

use prism_math::Vec3;

pub mod dispatch;
pub mod layout;

pub use dispatch::{
    index_3d_to_linear, linear_to_index_3d, Coverage, DispatchShape, WorkgroupSize,
};
pub use layout::{
    align_up, BindingAccess, BindingLayout, BufferSpec, ElementType, KernelDesc, LayoutError,
    BINDING_ALIGNMENT,
};

/// 主射线命中的最小 t（与 `prism-rt::integrator` 的 sentinel 同值口径）。
pub const T_MIN: f32 = 1e-4;

/// kernel 名：主射线发射。
pub const KERNEL_PRIMARY_RAY_GEN: &str = "rt_primary_ray_gen";
/// kernel 名：wavefront 求交一步。
pub const KERNEL_WAVEFRONT_INTERSECT: &str = "rt_wavefront_intersect";
/// kernel 名：SPIR-V 装配探针（`out[i] = in[i] * 2`，手写字节码，见
/// [`crate::vulkan::spirv`]）。
pub const KERNEL_SPV_MUL2_PROBE: &str = "spv_mul2_probe";

/// 入口函数名（三个 kernel 各自的 `OpEntryPoint` / WGSL `fn` / MSL kernel）。
pub const ENTRY_PRIMARY_RAY_GEN: &str = "primary_ray_gen";
/// wavefront 求交入口名。
pub const ENTRY_WAVEFRONT_INTERSECT: &str = "wavefront_intersect";
/// SPIR-V 探针入口名。
pub const ENTRY_SPV_MUL2: &str = "mul2";

// ---------------------------------------------------------------------------
// kernel 1：RT 主射线发射（相机 → 逐像素射线）
// ---------------------------------------------------------------------------

/// 相机参数（已预计算正交基与 `tan(fov/2)`，避免在此调用三角函数）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraParams {
    /// 视点。
    pub eye: Vec3,
    /// 前向单位向量。
    pub forward: Vec3,
    /// 右向单位向量。
    pub right: Vec3,
    /// 上向单位向量。
    pub up: Vec3,
    /// `tan(fov_y / 2)`（调用方一次算好）。
    pub tan_half_fov: f32,
    /// 画幅比（width / height）。
    pub aspect: f32,
    /// 亚像素抖动（`[0,1)`；`[0,0]` = 无抖动，确定性快路径）。
    pub jitter: [f32; 2],
}

impl CameraParams {
    /// 构造无抖动相机参数。
    pub fn new(
        eye: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        tan_half_fov: f32,
        aspect: f32,
    ) -> Self {
        CameraParams {
            eye,
            forward,
            right,
            up,
            tan_half_fov,
            aspect,
            jitter: [0.0, 0.0],
        }
    }

    /// 带亚像素抖动（确定性：抖动值由调用方的样本流给出，非 RNG）。
    pub fn with_jitter(mut self, jitter: [f32; 2]) -> Self {
        self.jitter = jitter;
        self
    }
}

/// 一条主射线样本（kernel 输出元素）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimaryRay {
    /// 像素坐标 `[x, y]`。
    pub pixel: [u32; 2],
    /// 射线起点（= 视点）。
    pub origin: Vec3,
    /// 单位方向。
    pub dir: Vec3,
}

/// **kernel 1 CPU 参考实现**：逐像素发射主射线。
///
/// 语义（与真机 kernel 逐字对应）：对每个 `global_invocation_id = (x, y, 0)`
/// 且落在 `extent` 内的调用，取该像素**中心**（`+0.5`）加亚像素抖动后的
/// NDC 坐标 → 相机基下方向 → 归一化。
///
/// 输出序 = [`DispatchShape::dispatch_cpu_clamped`] 的 invocation 序
/// （x 最快 ⇒ 行主序像素序），长度恒为 `extent[0] * extent[1] * extent[2]`。
///
/// # 参数
/// * `cam`：相机参数（基向量与 `tan_half_fov` 已预计算）；
/// * `extent`：问题域规模，通常 `[width, height, 1]`。
///
/// extent 任一元为 0 时返回空向量（不 panic）。
pub fn primary_ray_gen_cpu(cam: &CameraParams, extent: [u32; 3]) -> Vec<PrimaryRay> {
    if extent.iter().any(|&e| e == 0) {
        return Vec::new();
    }
    let shape = DispatchShape::new(extent, WorkgroupSize::new(64, 1, 1));
    let mut out: Vec<PrimaryRay> = Vec::with_capacity(shape.useful_invocations() as usize);
    let w = extent[0] as f32;
    let h = extent[1] as f32;
    let tan_half = cam.tan_half_fov;
    let aspect = cam.aspect;
    shape.dispatch_cpu_clamped(|id| {
        // 像素中心（+0.5）+ 亚像素抖动。
        let px = id[0] as f32 + 0.5 + cam.jitter[0];
        let py = id[1] as f32 + 0.5 + cam.jitter[1];
        // NDC：x ∈ [-aspect·tan, +aspect·tan]，y 上下翻转（图像 y 向下）。
        let ndc_x = (2.0 * (px / w) - 1.0) * tan_half * aspect;
        let ndc_y = (1.0 - 2.0 * (py / h)) * tan_half;
        let dir = (cam.forward + cam.right * ndc_x + cam.up * ndc_y).normalize();
        out.push(PrimaryRay {
            pixel: [id[0], id[1]],
            origin: cam.eye,
            dir,
        });
    });
    out
}

/// kernel 1 的 dispatch 形状（RT 主射线：一维工作组 64）。
pub fn primary_ray_gen_dispatch(extent: [u32; 3]) -> DispatchShape {
    DispatchShape::new(extent, WorkgroupSize::new(64, 1, 1))
}

// ---------------------------------------------------------------------------
// kernel 2：wavefront 求交一步（射线批 → 命中批）
// ---------------------------------------------------------------------------

/// 球体几何（wavefront 求交的图元集合元素）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereGeom {
    /// 球心。
    pub center: Vec3,
    /// 半径。
    pub radius: f32,
    /// 材质 id（命中批里原样带出，供着色阶段取材质）。
    pub material: u32,
}

/// 一条待求交射线（wavefront 队列项的核心字段）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaySample {
    /// 起点。
    pub origin: Vec3,
    /// 单位方向。
    pub dir: Vec3,
}

/// 命中记录（`None` = 逃逸）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitRecord {
    /// 命中距离。
    pub t: f32,
    /// 命中的图元 id（`spheres` 下标）。
    pub prim: u32,
    /// 命中点。
    pub point: Vec3,
    /// 单位外法线。
    pub normal: Vec3,
    /// 材质 id。
    pub material: u32,
}

/// 单条射线对单个球的求交（`None` = 未命中 / 命中在 [`T_MIN`] 之前）。
pub fn intersect_sphere(ray: &RaySample, sphere: &SphereGeom) -> Option<f32> {
    let oc = ray.origin - sphere.center;
    let b = oc.dot(&ray.dir);
    let c = oc.dot(&oc) - sphere.radius * sphere.radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    let t1 = -b + sq;
    // 取最近的正根（起点在球内时 t0 < 0，走 t1）。
    if t1 < T_MIN {
        return None;
    }
    let t = if t0 >= T_MIN { t0 } else { t1 };
    Some(t)
}

/// **kernel 2 CPU 参考实现**：wavefront 求交一步。
///
/// 输入一批射线、输出逐条射线的最近命中（或 `None`）。语义：
/// * 遍历序与 GPU 一致（批内线性序 = invocation id 序）；
/// * 平局（同一 `t`）按图元下标升序取小者——**确定性 tie-break**，
///   使 CPU/GPU/驱动重排都不会改变结果（工程四律：确定性）；
/// * 空图元集合 ⇒ 全 `None`（不 panic）。
pub fn wavefront_intersect_step_cpu(
    spheres: &[SphereGeom],
    rays: &[RaySample],
) -> Vec<Option<HitRecord>> {
    let mut out: Vec<Option<HitRecord>> = Vec::with_capacity(rays.len());
    for ray in rays {
        let mut best: Option<HitRecord> = None;
        for (i, s) in spheres.iter().enumerate() {
            let Some(t) = intersect_sphere(ray, s) else {
                continue;
            };
            let better = match best {
                None => true,
                Some(h) => t < h.t, // 严格小于：平局保留下标更小者
            };
            if better {
                let point = ray.origin + ray.dir * t;
                best = Some(HitRecord {
                    t,
                    prim: i as u32,
                    point,
                    normal: (point - s.center).normalize(),
                    material: s.material,
                });
            }
        }
        out.push(best);
    }
    out
}

/// kernel 2 的 dispatch 形状（一批射线 = 一维 extent）。
pub fn wavefront_intersect_dispatch(ray_count: u32) -> DispatchShape {
    DispatchShape::linear(ray_count, WorkgroupSize::new(64, 1, 1))
}

// ---------------------------------------------------------------------------
// kernel 注册表（描述 + 布局 + 源/二进制锚点）
// ---------------------------------------------------------------------------

/// 全部语义 kernel 的描述（顺序即注册序，确定性）。
///
/// 注意：`spv_module` 为**空串**表示「SPIR-V 待真机 glslang 生成」——
/// 本仓库构建环境没有 `glslangValidator`/`glslc`，故只有
/// [`KERNEL_SPV_MUL2_PROBE`] 的 SPIR-V 是手写字节码真实嵌入的（见
/// [`crate::vulkan::spirv`]）。这一区分由 [`kernel_spv_status`] 如实暴露，
/// **不假装** RT kernel 已有二进制。
pub fn kernel_registry() -> Vec<KernelDesc> {
    vec![
        KernelDesc::new(
            KERNEL_PRIMARY_RAY_GEN,
            ENTRY_PRIMARY_RAY_GEN,
            WorkgroupSize::new(64, 1, 1),
            vec![
                // b0: 相机基（eye / forward / right / up 各一个 vec4 槽）
                BufferSpec::new(0, ElementType::Vec4, 4, BindingAccess::ReadOnly),
                // b1: 相机标量（tan_half_fov / aspect / width / height）
                BufferSpec::new(1, ElementType::F32, 4, BindingAccess::ReadOnly),
                // b2: 输出射线（origin + dir = 2 × vec4 每像素）
                BufferSpec::new(2, ElementType::Vec4, 2048 * 2, BindingAccess::WriteOnly),
            ],
            "",
            "rt_primary_ray_gen.wgsl",
        ),
        KernelDesc::new(
            KERNEL_WAVEFRONT_INTERSECT,
            ENTRY_WAVEFRONT_INTERSECT,
            WorkgroupSize::new(64, 1, 1),
            vec![
                // b0: 射线批（origin + dir = 2 × vec4 每条）
                BufferSpec::new(0, ElementType::Vec4, 2048 * 2, BindingAccess::ReadOnly),
                // b1: 球体批（center + r + pad = vec4 每个）
                BufferSpec::new(1, ElementType::Vec4, 256, BindingAccess::ReadOnly),
                // b2: 命中批（t / prim / material / pad = vec4 每条）
                BufferSpec::new(2, ElementType::Vec4, 2048, BindingAccess::WriteOnly),
            ],
            "",
            "rt_wavefront_intersect.wgsl",
        ),
        spirv_probe_kernel(),
    ]
}

/// 按名取 kernel 描述。
pub fn kernel_by_name(name: &str) -> Option<KernelDesc> {
    kernel_registry().into_iter().find(|k| k.name == name)
}

/// SPIR-V 装配探针 kernel：`out[i] = in[i] * 2`。
///
/// 它是**唯一的真实 SPIR-V 二进制**（手写字节码，见
/// [`crate::vulkan::spirv::MODULES`]），用途是让「SPIR-V 加载 → 描述符布局
/// → compute pipeline → 命令缓冲 → 队列提交 → 读回」整条装配路径有一个
/// 可被真机 smoke 验证的最小载体。RT 语义 kernel 的 SPIR-V 待真机工具链。
pub fn spirv_probe_kernel() -> KernelDesc {
    KernelDesc::new(
        KERNEL_SPV_MUL2_PROBE,
        ENTRY_SPV_MUL2,
        WorkgroupSize::new(64, 1, 1),
        vec![
            BufferSpec::new(0, ElementType::F32, 256, BindingAccess::ReadOnly),
            BufferSpec::new(1, ElementType::F32, 256, BindingAccess::WriteOnly),
        ],
        "spv_mul2_probe",
        "spv_mul2_probe.wgsl",
    )
}

/// kernel 的 SPIR-V 供给状态（如实二分，供探针/CI 打印）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpvStatus {
    /// 已嵌入 SPIR-V 二进制（可装配 pipeline）。
    Embedded,
    /// 待真机工具链生成（本环境无 glslang）。
    PendingToolchain,
}

impl SpvStatus {
    /// 稳定短名。
    pub fn as_str(self) -> &'static str {
        match self {
            SpvStatus::Embedded => "embedded",
            SpvStatus::PendingToolchain => "pending-toolchain",
        }
    }
}

/// 查某 kernel 的 SPIR-V 供给状态。
pub fn kernel_spv_status(kernel: &KernelDesc) -> SpvStatus {
    if kernel.spv_module.is_empty() {
        SpvStatus::PendingToolchain
    } else {
        SpvStatus::Embedded
    }
}

// ---------------------------------------------------------------------------
// 测试（kernel 语义正确性：CPU 参考实现真跑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `tan(30°)`：32 位常量（不在内核里调三角，D-29）。
    const TAN_30: f32 = 0.577_350_26;

    fn straight_camera() -> CameraParams {
        CameraParams::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            TAN_30,
            1.0,
        )
    }

    /// kernel 1：每个像素恰好一条射线，序 = 行主序（GPU invocation 序）。
    #[test]
    fn primary_ray_gen_covers_every_pixel_exactly_once() {
        // 非整除 extent（113×47 / workgroup 64）⇒ 尾组浪费必须被裁掉。
        let extent = [113u32, 47u32, 1u32];
        let cam = straight_camera();
        let rays = primary_ray_gen_cpu(&cam, extent);
        assert_eq!(rays.len(), 113 * 47, "有效调用数须等于像素数");
        let shape = primary_ray_gen_dispatch(extent);
        assert!(
            shape.coverage().wasted_invocations > 0,
            "该 extent 应触发尾组浪费（诊断量非零）"
        );
        // 序：行主序；每像素唯一。
        let mut seen: Vec<[u32; 2]> = Vec::with_capacity(rays.len());
        for (i, r) in rays.iter().enumerate() {
            let x = (i as u32) % extent[0];
            let y = (i as u32) / extent[0];
            assert_eq!(r.pixel, [x, y], "输出序应为行主序（x 最快）");
            seen.push(r.pixel);
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), rays.len(), "像素不得重复");
        // 全部单位方向、起点 = 视点。
        for r in &rays {
            assert!((r.dir.length() - 1.0).abs() < 1e-5, "方向须归一化");
            assert_eq!(r.origin, cam.eye);
        }
    }

    /// kernel 1：主点射线方向 = forward；四角方向符号正确（不镜像）。
    #[test]
    fn primary_ray_gen_geometry_is_correct() {
        // 3×3 ⇒ 中心像素 (1,1) 恰为主点。
        let cam = straight_camera();
        let rays = primary_ray_gen_cpu(&cam, [3, 3, 1]);
        let center = rays[1 * 3 + 1];
        assert!((center.dir - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5, "中心射线须正对 forward");
        // 左上角 (0,0)：x 为负、y 为正（图像 y 向下 ⇒ 顶部 +y）。
        let tl = rays[0];
        assert!(tl.dir.x < 0.0, "左列 x 应为负：{:?}", tl.dir);
        assert!(tl.dir.y > 0.0, "顶行 y 应为正：{:?}", tl.dir);
        // 右下角 (2,2)：x 正、y 负。
        let br = rays[8];
        assert!(br.dir.x > 0.0, "右列 x 应为正");
        assert!(br.dir.y < 0.0, "底行 y 应为负");
        // 关于中心对称（同一行的左右像素 x 分量互为相反数）。
        let l = rays[1 * 3 + 0];
        let r = rays[1 * 3 + 2];
        assert!((l.dir.x + r.dir.x).abs() < 1e-6, "左右像素应镜像对称");
        assert!((l.dir.y - r.dir.y).abs() < 1e-6);
    }

    /// kernel 1：确定性——同参数两遍逐位一致；抖动改变结果但保持确定性。
    #[test]
    fn primary_ray_gen_is_deterministic_and_jitter_aware() {
        let cam = straight_camera();
        let a = primary_ray_gen_cpu(&cam, [16, 9, 1]);
        let b = primary_ray_gen_cpu(&cam, [16, 9, 1]);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].dir.x.to_bits(), b[i].dir.x.to_bits(), "两遍不逐位一致 px{i}");
            assert_eq!(a[i].dir.y.to_bits(), b[i].dir.y.to_bits());
            assert_eq!(a[i].dir.z.to_bits(), b[i].dir.z.to_bits());
        }
        let jit = cam.with_jitter([0.5, 0.5]);
        let c = primary_ray_gen_cpu(&jit, [16, 9, 1]);
        let d = primary_ray_gen_cpu(&jit, [16, 9, 1]);
        assert_eq!(c, d, "带抖动也须确定性");
        assert_ne!(c[0].dir, a[0].dir, "抖动应改变射线（否则抖动无效）");
    }

    /// kernel 1：空 extent 返回空（不 panic）。
    #[test]
    fn primary_ray_gen_empty_extent_is_empty() {
        let cam = straight_camera();
        assert!(primary_ray_gen_cpu(&cam, [0, 9, 1]).is_empty());
        assert!(primary_ray_gen_cpu(&cam, [4, 0, 1]).is_empty());
    }

    /// kernel 2：单球正面命中——t / 命中点 / 法线全部正确。
    #[test]
    fn intersect_single_sphere_hit_is_exact() {
        let spheres = [SphereGeom {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            material: 7,
        }];
        let rays = [RaySample {
            origin: Vec3::new(0.0, 0.0, -3.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        }];
        let hits = wavefront_intersect_step_cpu(&spheres, &rays);
        assert_eq!(hits.len(), 1);
        let h = hits[0].expect("应命中");
        assert!((h.t - 2.0).abs() < 1e-6, "t 应为 2，实际 {}", h.t);
        assert!((h.point - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
        assert!((h.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
        assert_eq!(h.prim, 0);
        assert_eq!(h.material, 7, "材质 id 须原样带出");
    }

    /// kernel 2：最近命中优先 + 平局按下标 + 逃逸/背向为 None。
    #[test]
    fn intersect_takes_nearest_and_handles_miss() {
        let spheres = [
            SphereGeom {
                center: Vec3::new(0.0, 0.0, 5.0),
                radius: 1.0,
                material: 1,
            },
            SphereGeom {
                center: Vec3::new(0.0, 0.0, 2.0),
                radius: 1.0,
                material: 2,
            },
            // 与第二个球完全相同位置（平局：下标 2 > 1，故保留 1）
            SphereGeom {
                center: Vec3::new(0.0, 0.0, 2.0),
                radius: 1.0,
                material: 3,
            },
        ];
        let rays = [
            // 正前方：命中 z=2 的球（t=1）
            RaySample {
                origin: Vec3::new(0.0, 0.0, 0.0),
                dir: Vec3::new(0.0, 0.0, 1.0),
            },
            // 背向：全 None
            RaySample {
                origin: Vec3::new(0.0, 0.0, 0.0),
                dir: Vec3::new(0.0, 0.0, -1.0),
            },
            // 侧向：全 None
            RaySample {
                origin: Vec3::new(0.0, 0.0, 0.0),
                dir: Vec3::new(1.0, 0.0, 0.0),
            },
        ];
        let hits = wavefront_intersect_step_cpu(&spheres, &rays);
        let h = hits[0].expect("应命中最近的球");
        assert_eq!(h.prim, 1, "最近命中的是下标 1（z=2）");
        assert_eq!(h.material, 2);
        assert!((h.t - 1.0).abs() < 1e-6);
        assert!(hits[1].is_none(), "背向射线不得命中");
        assert!(hits[2].is_none(), "侧向射线不得命中");
    }

    /// kernel 2：起点在球内——取出射根（t > 0），不死循环/不返回负 t。
    #[test]
    fn intersect_from_inside_sphere_returns_exit() {
        let spheres = [SphereGeom {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            material: 0,
        }];
        let rays = [RaySample {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        }];
        let hits = wavefront_intersect_step_cpu(&spheres, &rays);
        let h = hits[0].expect("球内起点应命中出射面");
        assert!((h.t - 1.0).abs() < 1e-6, "出射 t 应为 1，实际 {}", h.t);
        assert!((h.normal - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
    }

    /// kernel 2：批量长度对齐 + 空集合全 None + 确定性。
    #[test]
    fn intersect_batch_shape_and_determinism() {
        let spheres: Vec<SphereGeom> = (0..8)
            .map(|i| SphereGeom {
                center: Vec3::new(0.0, 0.0, 1.0 + i as f32),
                radius: 0.4,
                material: i,
            })
            .collect();
        let rays: Vec<RaySample> = (0..37)
            .map(|i| RaySample {
                origin: Vec3::new((i % 5) as f32 * 0.05, 0.0, 0.0),
                dir: Vec3::new(0.0, 0.0, 1.0),
            })
            .collect();
        let a = wavefront_intersect_step_cpu(&spheres, &rays);
        let b = wavefront_intersect_step_cpu(&spheres, &rays);
        assert_eq!(a.len(), rays.len(), "输出须与输入等长（批对齐）");
        assert_eq!(a, b, "同输入两遍结果应完全一致");
        assert!(wavefront_intersect_step_cpu(&[], &rays).iter().all(|h| h.is_none()));
        assert!(wavefront_intersect_step_cpu(&spheres, &[]).is_empty());
    }

    /// kernel 1 → kernel 2 串联：射线批送进求交，命中数自洽。
    #[test]
    fn primary_rays_feed_intersect_pipeline() {
        let cam = straight_camera();
        let rays = primary_ray_gen_cpu(&cam, [8, 8, 1]);
        assert_eq!(rays.len(), 64);
        // 一个正对相机的小球（z=1.0，半径 0.2）⇒ 中心附近像素命中。
        let spheres = [SphereGeom {
            center: Vec3::new(0.0, 0.0, 1.0),
            radius: 0.2,
            material: 42,
        }];
        let batch: Vec<RaySample> = rays
            .iter()
            .map(|r| RaySample {
                origin: r.origin,
                dir: r.dir,
            })
            .collect();
        let hits = wavefront_intersect_step_cpu(&spheres, &batch);
        assert_eq!(hits.len(), 64);
        let hit_count = hits.iter().filter(|h| h.is_some()).count();
        assert!(hit_count > 0, "小球应被部分主射线命中，实际 0 命中");
        assert!(hit_count < 64, "小球不应占满整帧，实际全命中");
        for h in hits.iter().flatten() {
            assert_eq!(h.material, 42);
            assert!(h.t > 0.0);
        }
    }

    /// 注册表：描述合法、布局唯一、SPIR-V 供给状态如实。
    #[test]
    fn kernel_registry_is_valid_and_spv_status_is_honest() {
        let reg = kernel_registry();
        assert!(reg.len() >= 3, "应注册 RT 两 kernel + SPIR-V 探针");
        for k in &reg {
            assert!(k.validate().is_ok(), "{} 描述非法：{:?}", k.name, k.validate());
            assert!(!k.spv_module.is_empty() || kernel_spv_status(k) == SpvStatus::PendingToolchain);
            assert!(!k.wgsl_source.is_empty(), "{} 缺 WGSL 源锚点", k.name);
            // 布局单调：offset 递增且对齐。
            let mut prev = 0;
            for l in k.layout() {
                assert!(l.offset >= prev);
                assert_eq!(l.offset % BINDING_ALIGNMENT, 0);
                prev = l.offset + l.bytes;
            }
        }
        assert_eq!(kernel_by_name(KERNEL_PRIMARY_RAY_GEN).map(|k| k.entry), Some(ENTRY_PRIMARY_RAY_GEN));
        assert!(kernel_by_name("no_such_kernel").is_none());
        // 探针是唯一 Embedded 者；RT kernel 如实标 PendingToolchain。
        let probe = kernel_by_name(KERNEL_SPV_MUL2_PROBE).expect("探针已注册");
        assert_eq!(kernel_spv_status(&probe), SpvStatus::Embedded);
        assert_eq!(probe.spv_module, "spv_mul2_probe");
        let rt = kernel_by_name(KERNEL_PRIMARY_RAY_GEN).expect("RT kernel 已注册");
        assert_eq!(
            kernel_spv_status(&rt),
            SpvStatus::PendingToolchain,
            "本环境无 glslang：RT kernel 的 SPIR-V 必须如实标为待生成"
        );
        assert_eq!(SpvStatus::Embedded.as_str(), "embedded");
        assert_eq!(SpvStatus::PendingToolchain.as_str(), "pending-toolchain");
    }
}
