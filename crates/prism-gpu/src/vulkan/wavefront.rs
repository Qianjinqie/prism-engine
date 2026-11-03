//! **T6：整条 WGSL wavefront 的 Vulkan 派发器**（5 个 compute pipeline）。
//!
//! # 这是什么
//!
//! `crates/prism-gpu/shaders/rt_wavefront.wgsl` 有 5 个 `@compute` 入口
//! （`generate` / `shade` / `connect` / `extend` / `accumulate`）。本模块把它们
//! 装成 5 条 compute pipeline，按 `prism-rt/src/wavefront.rs` 的 CPU 语义派发：
//!
//! ```text
//! generate → [ shade → connect → extend ] × max_bounces → accumulate
//! ```
//!
//! 每个入口一个 **独立 `.spv`**（`tools/wgsl2spv --all` 产物，入库；D-21），
//! 因为 `VkPipelineShaderStageCreateInfo.p_name` 只能选中模块里的一个入口。
//!
//! # 绑定（4 个 descriptor set，与 WGSL 声明逐条对应）
//!
//! | set | binding | 资源 | 类型 |
//! |---|---|---|---|
//! | 0 | 0 | `slots`（`array<PathSlot>`） | storage rw |
//! | 0 | 1 | `params`（`WFParams`） | uniform |
//! | 0 | 2 | `accum`（`array<vec4<f32>>`） | storage rw |
//! | 1 | 0 | `sobol_dir`（`[64][32] u32`） | storage ro |
//! | 1 | 1 | `scene_buf`（占位） | storage ro |
//! | 2 | 0 | `scene_params`（`SceneParams`） | uniform |
//! | 2 | 1 | `materials` | storage ro |
//! | 2 | 2 | `prim_mat` | storage ro |
//! | 2 | 3 | `tris` | storage ro |
//! | 2 | 4 | `lights` | storage ro |
//! | 2 | 5 | `env_blob` | storage ro |
//! | 2 | 6 | `spheres`（兼容模式，平铺 8 f32/球） | storage ro |
//! | 2 | 7 | `dir_lights`（兼容模式，平铺 8 f32/灯） | storage ro |
//! | 3 | 0 | `camera`（`Camera`） | uniform |
//!
//! `extend` 的静态引用集是 `{0,1}`（set0）+ `{0,1}`（set1）+
//! `{0,1,2,3,4}`（set2），`shade` 是 `{0..7}`（set2）——**每个 set 内都是
//! 连续前缀**。本模块给 4 个 set 都建完整布局（超集），Vulkan 允许 pipeline
//! layout 含 shader 未使用的 binding。
//!
//! # 两种模式（同一份 shader，`WFParams.mode`）
//!
//! * [`WavefrontMode::Path`]（0）：W2 的路径追踪语义，输出 HDR `vec4`；
//! * [`WavefrontMode::CpuRefCompat`]（1）：`CpuRefBackend` 的**逐式移植**，
//!   输出量化后的 RGBA8 位型（W1 逐字节判据用）。
//!
//! 兼容模式不是"另一条产品路径"：它存在的唯一目的是把**派发器/绑定/队列**
//! 与 megakernel 放在同一估计量上对拍。产品代码不调用它。
//!
//! # 无设备时
//!
//! `VulkanBackend` 没有 context（无 ICD / wasm）时返回
//! [`DispatchStatus::DeviceUnavailable`]，`pixels` / `hdr` 为空——**不造帧**。

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use super::compute::DispatchStatus;
#[cfg(not(target_arch = "wasm32"))]
use super::loader as vk;
#[cfg(not(target_arch = "wasm32"))]
use super::VulkanContext;
use crate::{AestheticReq, CameraReq, SceneRenderRequest, SphereReq};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 产物路径（D-21：`.spv` 入库，构建期不生成）
// ---------------------------------------------------------------------------

/// `generate` 的入库 SPIR-V（相对 `crates/prism-gpu`）。
pub const SPV_GENERATE: &str = "shaders/spv/rt_wavefront_generate.spv";
/// `shade` 的入库 SPIR-V。
pub const SPV_SHADE: &str = "shaders/spv/rt_wavefront_shade.spv";
/// `connect` 的入库 SPIR-V。
pub const SPV_CONNECT: &str = "shaders/spv/rt_wavefront_connect.spv";
/// `extend` 的入库 SPIR-V。
pub const SPV_EXTEND: &str = "shaders/spv/rt_wavefront_extend.spv";
/// `accumulate` 的入库 SPIR-V。
pub const SPV_ACCUMULATE: &str = "shaders/spv/rt_wavefront_accumulate.spv";

/// 五个入口的 `(entry, spv 相对路径)`，顺序即派发顺序的前四段。
pub const STAGES: [(&str, &str); 5] = [
    ("generate", SPV_GENERATE),
    ("shade", SPV_SHADE),
    ("connect", SPV_CONNECT),
    ("extend", SPV_EXTEND),
    ("accumulate", SPV_ACCUMULATE),
];

/// 读入库 `.spv` 并按小端解成 SPIR-V 字流。
///
/// 失败原因写清楚（缺文件 = 没跑过 `wgsl2spv --all`；长度非 4 的倍数 = 文件损坏）。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn load_spv(rel: &str) -> Result<Vec<u32>, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "读不到 {}：{e}\n修复：cargo run --manifest-path tools/wgsl2spv/Cargo.toml -- --all",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} 长度 {} 不是 4 的倍数（SPIR-V 是 u32 字流）",
            path.display(),
            bytes.len()
        ));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(words)
}

// ---------------------------------------------------------------------------
// 输入类型
// ---------------------------------------------------------------------------

/// wavefront 的两种语义（写进 `WFParams.mode`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavefrontMode {
    /// W2 路径追踪（产品语义；输出 HDR）。
    Path,
    /// `CpuRefBackend` 逐式移植（W1 判据；输出 RGBA8 位型）。
    CpuRefCompat,
}

impl WavefrontMode {
    /// WGSL 里的 `mode` 值。
    fn as_u32(self) -> u32 {
        match self {
            WavefrontMode::Path => 0,
            WavefrontMode::CpuRefCompat => 1,
        }
    }
}

/// 路径模式的三角形（与 `rt_scene.wgsl::Tri` 的 12 个 f32 槽对应）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavefrontTri {
    /// 顶点 A。
    pub a: Vec3,
    /// 顶点 B。
    pub b: Vec3,
    /// 顶点 C。
    pub c: Vec3,
}

/// 路径模式的材质（与 `rt_scene.wgsl::Material` 的 16 个 f32 槽对应）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavefrontMaterial {
    /// 基色。
    pub base_color: Vec3,
    /// 粗糙度。
    pub roughness: f32,
    /// 菲涅耳 F0。
    pub f0: Vec3,
    /// 金属度。
    pub metalness: f32,
    /// 透射率。
    pub transmission: f32,
    /// 折射率。
    pub ior: f32,
    /// 自发光。
    pub emissive: Vec3,
}

/// 路径模式的球面灯代理（与 `rt_scene.wgsl::Light` 对应）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavefrontLight {
    /// 灯中心。
    pub position: Vec3,
    /// 灯半径（决定立体角 pdf）。
    pub radius: f32,
    /// 灯颜色。
    pub color: Vec3,
}

/// 一次 wavefront 渲染的完整输入。
#[derive(Debug, Clone)]
pub struct WavefrontRequest {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// 每像素样本数。
    pub spp: u32,
    /// 最大弹射次数。
    pub max_bounces: u32,
    /// 全局种子。
    pub seed: u32,
    /// 模式。
    pub mode: WavefrontMode,
    /// 相机。
    pub camera: CameraReq,
    /// aesthetic（兼容模式读）。
    pub aesthetic: AestheticReq,
    /// 球表（兼容模式读）。
    pub spheres: Vec<SphereReq>,
    /// 三角形（路径模式读）。
    pub triangles: Vec<WavefrontTri>,
    /// 材质（路径模式读）。
    pub materials: Vec<WavefrontMaterial>,
    /// 三角形 → 材质索引（路径模式读；缺省时按 0 填满）。
    pub prim_mat: Vec<u32>,
    /// 球面灯（路径模式读）。
    pub lights: Vec<WavefrontLight>,
    /// 环境背景（`SceneParams.ambient_*`）。
    pub ambient: Vec3,
    /// 环境 CDF blob（空 = 1 元占位 ⇒ `env_present()` 假）。
    pub env_blob: Vec<f32>,
    /// **只跑到 `stage_plan` 的第 k 段**（`None` = 跑完整计划）。
    ///
    /// 存在的理由只有一个：W1 的「差异定位到每像素每阶段」。把派发停在
    /// `generate` 之后，宿主就能读回该像素的**射线**，与 Rust 侧按同一表达式树
    /// 重算的射线逐位比对——这样「差异是生成段引入的还是着色段引入的」不再是
    /// 猜的。产品调用方不设它。
    pub stage_stop: Option<usize>,
    /// 读回 `slots` 原始字节（默认关：`96 B × 路径数` 对产品帧是纯浪费）。
    pub readback_slots: bool,
}

impl WavefrontRequest {
    /// 从 `SceneRenderRequest` 构造**兼容模式**请求（W1 判据的唯一入口）。
    ///
    /// 字段映射逐项显式，不做 `..Default::default()`：W1 的对拍语义要求
    /// 相机 / 灯 / 球 / aesthetic 全部来自同一个请求对象。
    pub fn compat_from_scene(req: &SceneRenderRequest) -> Self {
        // 兼容模式的灯：`LightReq` 的 `direction` 先归一化（megakernel 的
        // `encode_lights` 也在 CPU 侧归一化，见 `spirv_rt`）。
        let lights = req
            .lights
            .iter()
            .map(|l| WavefrontLight {
                position: l.direction.normalize(),
                radius: l.intensity,
                color: l.color,
            })
            .collect();
        WavefrontRequest {
            width: req.width,
            height: req.height,
            spp: 1,
            max_bounces: 1,
            seed: 0,
            mode: WavefrontMode::CpuRefCompat,
            camera: req.camera.clone(),
            aesthetic: req.aesthetic.clone(),
            spheres: req.spheres.clone(),
            triangles: Vec::new(),
            materials: Vec::new(),
            prim_mat: Vec::new(),
            lights,
            ambient: Vec3::splat(0.0),
            env_blob: Vec::new(),
            stage_stop: None,
            readback_slots: false,
        }
    }
}

/// 一次 wavefront 渲染的结果。
#[derive(Debug, Clone, PartialEq)]
pub struct WavefrontReport {
    /// 结果状态。
    pub status: DispatchStatus,
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 派发的 compute 次数（含每个 bounce 的三段）。
    pub dispatches: u32,
    /// 设备名（有设备时非空；失败也带出）。
    pub device: String,
    /// 失败原因（成功时为空）。
    pub detail: String,
    /// RGBA8 像素（兼容模式 = 真值；路径模式 = clamp 后的预览）。
    pub pixels: Vec<u8>,
    /// HDR 累积（路径模式；`accum` 的 `vec4` 逐像素）。
    pub hdr: Vec<[f32; 4]>,
    /// 全部绑定缓冲的字节数（诊断）。
    pub buffer_bytes: u64,
    /// `slots` 原始字节（仅 `readback_slots == true` 时非空）。
    pub slot_bytes: Vec<u8>,
}

impl WavefrontReport {
    /// 是否成功。
    pub fn is_ok(&self) -> bool {
        self.status == DispatchStatus::Ok
    }

    /// 失败报告（不造帧）。
    fn failed(status: DispatchStatus, w: u32, h: u32, detail: impl Into<String>) -> Self {
        WavefrontReport {
            status,
            width: w,
            height: h,
            dispatches: 0,
            device: String::new(),
            detail: detail.into(),
            pixels: Vec::new(),
            hdr: Vec::new(),
            buffer_bytes: 0,
            slot_bytes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// 编码（唯一权威：本模块；着色器按同一布局读取）
// ---------------------------------------------------------------------------

/// `PathSlot` 的字节数（std430：5 个 vec3 各带 4 字节填充 → 96）。
pub const PATH_SLOT_BYTES: usize = 96;
/// `WFParams` 的字节数（16 × 4）。
pub const WF_PARAMS_BYTES: usize = 64;
/// `SceneParams` 的字节数（8 × 4）。
pub const SCENE_PARAMS_BYTES: usize = 32;
/// `Material` 的字节数。
pub const MATERIAL_BYTES: usize = 64;
/// `Tri` 的字节数。
pub const TRI_BYTES: usize = 48;
/// `Light` 的字节数。
pub const LIGHT_BYTES: usize = 32;
/// `Camera` 的字节数。
pub const CAMERA_BYTES: usize = 96;
/// 平铺球 / 灯的每元素 f32 槽数。
pub const FLAT_ELEM_SLOTS: usize = 8;

/// f32 → little-endian 字节。
fn put_f32(out: &mut [u8], off: usize, v: f32) {
    out[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
}

/// u32 → little-endian 字节。
fn put_u32(out: &mut [u8], off: usize, v: u32) {
    out[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// 把 `Vec3` 写到 `off` 起的三个 f32 槽。
fn put_vec3(out: &mut [u8], off: usize, v: Vec3) {
    put_f32(out, off, v.x);
    put_f32(out, off + 4, v.y);
    put_f32(out, off + 8, v.z);
}

/// 兼容模式的球表（平铺 8 f32/球，与 megakernel `encode_spheres` 逐字节同式）。
fn encode_spheres_flat(spheres: &[SphereReq]) -> Vec<u8> {
    let mut out = vec![0u8; spheres.len().max(1) * FLAT_ELEM_SLOTS * 4];
    for (i, s) in spheres.iter().enumerate() {
        let b = i * FLAT_ELEM_SLOTS * 4;
        put_vec3(&mut out, b, s.center);
        put_f32(&mut out, b + 12, s.radius);
        put_vec3(&mut out, b + 16, s.albedo);
        // 槽 7 = pad，保持 0。
    }
    out
}

/// 兼容模式的方向光表（平铺 8 f32/灯：dir / color / intensity / pad）。
fn encode_dirlights_flat(lights: &[WavefrontLight]) -> Vec<u8> {
    let mut out = vec![0u8; lights.len().max(1) * FLAT_ELEM_SLOTS * 4];
    for (i, l) in lights.iter().enumerate() {
        let b = i * FLAT_ELEM_SLOTS * 4;
        put_vec3(&mut out, b, l.position);
        put_vec3(&mut out, b + 12, l.color);
        put_f32(&mut out, b + 24, l.radius);
    }
    out
}

/// 路径模式的灯表（`rt_scene.wgsl::Light`：position / radius / color / pad）。
fn encode_lights_rt(lights: &[WavefrontLight]) -> Vec<u8> {
    let mut out = vec![0u8; lights.len().max(1) * LIGHT_BYTES];
    for (i, l) in lights.iter().enumerate() {
        let b = i * LIGHT_BYTES;
        put_vec3(&mut out, b, l.position);
        put_f32(&mut out, b + 12, l.radius);
        put_vec3(&mut out, b + 16, l.color);
    }
    out
}

/// 材质表（`rt_scene.wgsl::Material` 的 16 个 f32 槽）。
fn encode_materials(materials: &[WavefrontMaterial]) -> Vec<u8> {
    let mut out = vec![0u8; materials.len().max(1) * MATERIAL_BYTES];
    for (i, m) in materials.iter().enumerate() {
        let b = i * MATERIAL_BYTES;
        put_vec3(&mut out, b, m.base_color);
        put_f32(&mut out, b + 12, m.roughness);
        put_vec3(&mut out, b + 16, m.f0);
        put_f32(&mut out, b + 28, m.metalness);
        put_f32(&mut out, b + 32, m.transmission);
        put_f32(&mut out, b + 36, m.ior);
        put_vec3(&mut out, b + 48, m.emissive);
    }
    out
}

/// 三角形表（`rt_scene.wgsl::Tri` 的 12 个 f32 槽）。
fn encode_tris(tris: &[WavefrontTri]) -> Vec<u8> {
    let mut out = vec![0u8; tris.len().max(1) * TRI_BYTES];
    for (i, t) in tris.iter().enumerate() {
        let b = i * TRI_BYTES;
        put_vec3(&mut out, b, t.a);
        put_vec3(&mut out, b + 16, t.b);
        put_vec3(&mut out, b + 32, t.c);
    }
    out
}

/// 相机 uniform（96 字节，`aspect` 在原填充位 @84）。
fn encode_camera(cam: &CameraReq) -> [u8; CAMERA_BYTES] {
    let basis = super::spirv_rt::camera_basis(cam);
    let mut out = [0u8; CAMERA_BYTES];
    for k in 0..3 {
        put_f32(&mut out, k * 4, basis.eye[k]);
        put_f32(&mut out, 16 + k * 4, basis.forward[k]);
        put_f32(&mut out, 32 + k * 4, basis.right[k]);
        put_f32(&mut out, 48 + k * 4, basis.up[k]);
    }
    put_f32(&mut out, 64, basis.tan_half_fov);
    put_f32(&mut out, 68, 0.0); // aperture_radius：兼容 / W1 都是针孔
    put_f32(&mut out, 72, 1.0); // focus_distance（不读，填确定性值）
    put_f32(&mut out, 76, 0.0); // shutter_open
    put_f32(&mut out, 80, 0.0); // shutter_close
    put_f32(&mut out, 84, basis.aspect);
    out
}

/// `WFParams`（64 字节）。
fn encode_params(req: &WavefrontRequest) -> [u8; WF_PARAMS_BYTES] {
    let mut out = [0u8; WF_PARAMS_BYTES];
    put_u32(&mut out, 0, req.width);
    put_u32(&mut out, 4, req.height);
    put_u32(&mut out, 8, req.spp);
    put_u32(&mut out, 12, req.max_bounces);
    put_u32(&mut out, 16, 0); // clamp_indirect_on（W1 / W3 都不钳）
    put_f32(&mut out, 20, 0.0);
    put_u32(&mut out, 24, 0); // nee_only = false（路径模式跑完整 NEE + MIS）
    put_u32(&mut out, 28, req.seed);
    put_u32(&mut out, 32, 0); // node_id
    put_u32(&mut out, 36, 0); // frame
    put_u32(&mut out, 40, req.mode.as_u32());
    put_u32(
        &mut out,
        44,
        u32::from(req.aesthetic.teal_orange),
    );
    put_f32(&mut out, 48, req.aesthetic.contrast);
    put_u32(
        &mut out,
        52,
        u32::from(req.aesthetic.contrast != 1.0),
    );
    put_u32(&mut out, 56, u32::from(req.aesthetic.filmic));
    out
}

/// `SceneParams`（32 字节）。
fn encode_scene_params(req: &WavefrontRequest) -> [u8; SCENE_PARAMS_BYTES] {
    let mut out = [0u8; SCENE_PARAMS_BYTES];
    put_u32(&mut out, 0, req.materials.len() as u32);
    put_u32(&mut out, 4, req.lights.len() as u32);
    put_u32(&mut out, 8, req.triangles.len() as u32);
    put_u32(&mut out, 12, 0); // node_id
    put_f32(&mut out, 16, req.ambient.x);
    put_f32(&mut out, 20, req.ambient.y);
    put_f32(&mut out, 24, req.ambient.z);
    put_u32(&mut out, 28, req.spheres.len() as u32);
    out
}

/// 确定性 Sobol 方向数表（`[64][32] u32`，8192 B）。
///
/// # 这不是"随便填的随机数"
///
/// 表的语义由 `rt_shade.wgsl::sobol_u32` 决定：`v = ⊕_{k: bit(index,k)} row[dim][k]`。
/// 本函数用 PCG 字节混洗填表，保证：
/// * **非零**（全零表会让所有样本退化成同一个点）；
/// * **确定性**（同 seed 同表，不依赖时间/地址）；
/// * **维度间不相关**（每行用不同种子派生）。
///
/// 它不承诺 Joe–Kuo 的 t-值最优性——W3 只比 4 vs 8 弹射的**同一张表**，
/// 表质量不进入判据；真正的低差异序列在 CPU 侧（`prism-rt::sampler`）。
fn sobol_table(seed: u32) -> Vec<u8> {
    let mut out = vec![0u8; 64 * 32 * 4];
    let mut state = seed ^ 0x9E37_79B1;
    for dim in 0..64u32 {
        for k in 0..32u32 {
            // xorshift32 + 乘混洗：整数、无平台差异。
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = state.wrapping_mul(0x2545_F491).wrapping_add(dim * 0x9E37 + k);
            put_u32(&mut out, ((dim * 32 + k) * 4) as usize, v);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 资源表
// ---------------------------------------------------------------------------

/// 一个绑定的静态描述（buffer 大小 + 初始字节）。
///
/// **跨后端单一真源**：native 的 [`run_on_device`] 与 wasm 侧
/// `web::scene_gpu` 的 wavefront 路径都消费这张表——Vulkan 用它建
/// VkBuffer/descriptor，WebGPU 用它建 GPUBuffer/bind group。两边的绑定顺序
/// 因此不可能漂移（漂移过一次：`extend` 少了 `lights` 就不会被任何静态检查发现）。
pub(crate) struct Resource {
    /// descriptor set（WebGPU 的 bind group）索引。
    pub(crate) group: u32,
    /// 组内绑定号。
    pub(crate) binding: u32,
    /// `true` = uniform buffer；`false` = storage buffer。
    pub(crate) uniform: bool,
    /// 初始字节（写入 host-visible 缓冲）。
    pub(crate) bytes: Vec<u8>,
}

impl Resource {
    /// 实际分配的字节数（Vulkan 拒绝 0 长 buffer）。
    #[cfg(not(target_arch = "wasm32"))]
    fn alloc_bytes(&self) -> u64 {
        self.bytes.len().max(4) as u64
    }
}

/// 全部资源（顺序即 `Resource` 表的顺序；descriptor 写入按此顺序）。
///
/// 顺序恒为 `(group, binding)` 升序：g0{slots,params,accum} → g1{sobol,scene_buf}
/// → g2{scene_params..dir_lights 共 8} → g3{camera}。两个后端都按「组内前 n 个」
/// 取前缀，故顺序即契约。
pub(crate) fn build_resources(req: &WavefrontRequest) -> Vec<Resource> {
    let paths = (req.width as usize)
        .saturating_mul(req.height as usize)
        .saturating_mul(req.spp as usize);
    let npix = (req.width as usize).saturating_mul(req.height as usize);

    let mut slots = vec![0u8; paths * PATH_SLOT_BYTES];
    // 初始状态 = TERMINATED（1），避免未生成的槽被 accumulate 当有效路径。
    for i in 0..paths {
        put_u32(&mut slots, i * PATH_SLOT_BYTES + 64, 1);
    }
    // `prim_mat` 缺省按 0 填满（路径模式每个三角形都有材质）。
    let mut prim_mat = vec![0u8; req.triangles.len().max(1) * 4];
    for (i, &m) in req.prim_mat.iter().take(req.triangles.len()).enumerate() {
        put_u32(&mut prim_mat, i * 4, m);
    }

    let mut env = vec![0u8; req.env_blob.len().max(1) * 4];
    for (i, &v) in req.env_blob.iter().enumerate() {
        put_f32(&mut env, i * 4, v);
    }

    let lights = match req.mode {
        WavefrontMode::CpuRefCompat => encode_dirlights_flat(&req.lights),
        WavefrontMode::Path => encode_lights_rt(&req.lights),
    };

    vec![
        // group 0
        Resource {
            group: 0,
            binding: 0,
            uniform: false,
            bytes: slots,
        },
        Resource {
            group: 0,
            binding: 1,
            uniform: true,
            bytes: encode_params(req).to_vec(),
        },
        Resource {
            group: 0,
            binding: 2,
            uniform: false,
            bytes: vec![0u8; npix * 16],
        },
        // group 1
        Resource {
            group: 1,
            binding: 0,
            uniform: false,
            bytes: sobol_table(req.seed),
        },
        Resource {
            group: 1,
            binding: 1,
            uniform: false,
            bytes: vec![0u8; 4],
        },
        // group 2
        Resource {
            group: 2,
            binding: 0,
            uniform: true,
            bytes: encode_scene_params(req).to_vec(),
        },
        Resource {
            group: 2,
            binding: 1,
            uniform: false,
            bytes: encode_materials(&req.materials),
        },
        Resource {
            group: 2,
            binding: 2,
            uniform: false,
            bytes: prim_mat,
        },
        Resource {
            group: 2,
            binding: 3,
            uniform: false,
            bytes: encode_tris(&req.triangles),
        },
        Resource {
            group: 2,
            binding: 4,
            uniform: false,
            bytes: lights.clone(),
        },
        Resource {
            group: 2,
            binding: 5,
            uniform: false,
            bytes: env,
        },
        Resource {
            group: 2,
            binding: 6,
            uniform: false,
            bytes: encode_spheres_flat(&req.spheres),
        },
        Resource {
            group: 2,
            binding: 7,
            uniform: false,
            // 路径模式下不读，但绑定必须存在；兼容模式用同一份平铺灯表。
            bytes: if matches!(req.mode, WavefrontMode::CpuRefCompat) {
                lights
            } else {
                // 路径模式的 `lights` 是 rt_scene 布局；兼容表单独给 1 灯占位。
                vec![0u8; FLAT_ELEM_SLOTS * 4]
            },
        },
        // group 3
        Resource {
            group: 3,
            binding: 0,
            uniform: true,
            bytes: encode_camera(&req.camera).to_vec(),
        },
    ]
}

// ---------------------------------------------------------------------------
// 派发
// ---------------------------------------------------------------------------

/// 5 个 stage 的 dispatch 形状（每个 64 线程/组）。
#[cfg(not(target_arch = "wasm32"))]
fn dispatch_groups(req: &WavefrontRequest, entry: &str) -> u32 {
    let n = if entry == "accumulate" {
        (req.width as u64) * (req.height as u64)
    } else {
        (req.width as u64) * (req.height as u64) * (req.spp as u64)
    };
    n.div_ceil(64).max(1) as u32
}

/// 无设备 / wasm 的失败报告。
fn unavailable(status: DispatchStatus, req: &WavefrontRequest, detail: &str) -> WavefrontReport {
    let mut r = WavefrontReport::failed(status, req.width, req.height, detail);
    r.dispatches = 0;
    r
}

/// **真机路径**（有 `VulkanContext`）。
///
/// `cleanup!` 宏在每条提前返回分支上都把句柄清零，编译器会报「赋了没读」；
/// 这是 RAII-on-failure 的固有形态，与 [`super::compute::run_on_device`] 同处理。
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_assignments)]
unsafe fn run_on_device(ctx: &VulkanContext, req: &WavefrontRequest) -> WavefrontReport {
    let fns = &ctx.device_fns;
    let device = ctx.device;

    // ---- 必备入口 ------------------------------------------------------
    let required = [
        ("vkCreateShaderModule", fns.create_shader_module.is_some()),
        ("vkDestroyShaderModule", fns.destroy_shader_module.is_some()),
        (
            "vkCreateDescriptorSetLayout",
            fns.create_descriptor_set_layout.is_some(),
        ),
        (
            "vkDestroyDescriptorSetLayout",
            fns.destroy_descriptor_set_layout.is_some(),
        ),
        ("vkCreatePipelineLayout", fns.create_pipeline_layout.is_some()),
        (
            "vkDestroyPipelineLayout",
            fns.destroy_pipeline_layout.is_some(),
        ),
        (
            "vkCreateComputePipelines",
            fns.create_compute_pipelines.is_some(),
        ),
        ("vkDestroyPipeline", fns.destroy_pipeline.is_some()),
        ("vkCreateBuffer", fns.create_buffer.is_some()),
        ("vkDestroyBuffer", fns.destroy_buffer.is_some()),
        (
            "vkGetBufferMemoryRequirements",
            fns.get_buffer_memory_requirements.is_some(),
        ),
        ("vkAllocateMemory", fns.allocate_memory.is_some()),
        ("vkFreeMemory", fns.free_memory.is_some()),
        ("vkBindBufferMemory", fns.bind_buffer_memory.is_some()),
        ("vkCreateDescriptorPool", fns.create_descriptor_pool.is_some()),
        (
            "vkDestroyDescriptorPool",
            fns.destroy_descriptor_pool.is_some(),
        ),
        (
            "vkAllocateDescriptorSets",
            fns.allocate_descriptor_sets.is_some(),
        ),
        ("vkUpdateDescriptorSets", fns.update_descriptor_sets.is_some()),
        ("vkCreateCommandPool", fns.create_command_pool.is_some()),
        (
            "vkDestroyCommandPool",
            fns.destroy_command_pool.is_some(),
        ),
        (
            "vkAllocateCommandBuffers",
            fns.allocate_command_buffers.is_some(),
        ),
        (
            "vkFreeCommandBuffers",
            fns.free_command_buffers.is_some(),
        ),
        ("vkBeginCommandBuffer", fns.begin_command_buffer.is_some()),
        ("vkEndCommandBuffer", fns.end_command_buffer.is_some()),
        ("vkCmdBindPipeline", fns.cmd_bind_pipeline.is_some()),
        (
            "vkCmdBindDescriptorSets",
            fns.cmd_bind_descriptor_sets.is_some(),
        ),
        ("vkCmdDispatch", fns.cmd_dispatch.is_some()),
        (
            "vkCmdPipelineBarrier",
            fns.cmd_pipeline_barrier.is_some(),
        ),
        ("vkQueueSubmit", fns.queue_submit.is_some()),
        ("vkCreateFence", fns.create_fence.is_some()),
        ("vkDestroyFence", fns.destroy_fence.is_some()),
        ("vkWaitForFences", fns.wait_for_fences.is_some()),
        ("vkMapMemory", fns.map_memory.is_some()),
        ("vkUnmapMemory", fns.unmap_memory.is_some()),
    ];
    let missing: Vec<&'static str> = required
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| *n)
        .collect();
    if !missing.is_empty() {
        let mut r = unavailable(
            DispatchStatus::MissingEntryPoints(missing),
            req,
            "驱动/loader 未导出 wavefront 派发必备入口",
        );
        r.device = ctx.device_name.clone();
        return r;
    }

    // ---- 读 5 份入库 SPIR-V（D-21） -----------------------------------
    let mut modules_words: Vec<Vec<u32>> = Vec::with_capacity(STAGES.len());
    for (entry, rel) in STAGES {
        match load_spv(rel) {
            Ok(w) => modules_words.push(w),
            Err(e) => {
                let mut r = WavefrontReport::failed(
                    DispatchStatus::ShaderModuleFailed,
                    req.width,
                    req.height,
                    format!("{entry}: {e}"),
                );
                r.device = ctx.device_name.clone();
                return r;
            }
        }
    }

    // ---- 资源 ----------------------------------------------------------
    let resources = build_resources(req);
    let total_bytes: u64 = resources.iter().map(|r| r.alloc_bytes()).sum();

    // ---- 清理登记 ------------------------------------------------------
    let mut shader_modules: Vec<vk::VkShaderModule> = Vec::new();
    let mut dsls: Vec<vk::VkDescriptorSetLayout> = Vec::new();
    let mut pipeline_layout: vk::VkPipelineLayout = 0;
    let mut pipelines: Vec<vk::VkPipeline> = Vec::new();
    let mut pool: vk::VkDescriptorPool = 0;
    let mut sets: [vk::VkDescriptorSet; 4] = [0; 4];
    let mut cmd_pool: vk::VkCommandPool = 0;
    let mut cmd: vk::VkCommandBuffer = std::ptr::null_mut();
    let mut fence: vk::VkFence = 0;
    let mut buffers: Vec<(vk::VkBuffer, vk::VkDeviceMemory)> = Vec::new();

    // 统一清理（逆序；每个分支都走这里，避免半建状态泄漏）。
    macro_rules! cleanup {
        () => {{
            if fence != 0 {
                (fns.destroy_fence.unwrap())(device, fence, std::ptr::null());
                fence = 0;
            }
            if cmd_pool != 0 {
                if !cmd.is_null() {
                    (fns.free_command_buffers.unwrap())(device, cmd_pool, 1, &cmd);
                    cmd = std::ptr::null_mut();
                }
                (fns.destroy_command_pool.unwrap())(device, cmd_pool, std::ptr::null());
                cmd_pool = 0;
            }
            if pool != 0 {
                (fns.destroy_descriptor_pool.unwrap())(device, pool, std::ptr::null());
                pool = 0;
            }
            for (b, m) in buffers.drain(..) {
                (fns.destroy_buffer.unwrap())(device, b, std::ptr::null());
                (fns.free_memory.unwrap())(device, m, std::ptr::null());
            }
            for p in pipelines.drain(..) {
                (fns.destroy_pipeline.unwrap())(device, p, std::ptr::null());
            }
            if pipeline_layout != 0 {
                (fns.destroy_pipeline_layout.unwrap())(device, pipeline_layout, std::ptr::null());
                pipeline_layout = 0;
            }
            for d in dsls.drain(..) {
                (fns.destroy_descriptor_set_layout.unwrap())(device, d, std::ptr::null());
            }
            for m in shader_modules.drain(..) {
                (fns.destroy_shader_module.unwrap())(device, m, std::ptr::null());
            }
        }};
    }

    // ---- 1. shader modules --------------------------------------------
    for words in &modules_words {
        let smci = vk::VkShaderModuleCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: words.len() * 4,
            p_code: words.as_ptr(),
        };
        let mut m: vk::VkShaderModule = 0;
        let r = (fns.create_shader_module.unwrap())(device, &smci, std::ptr::null(), &mut m);
        if r != vk::VK_SUCCESS || m == 0 {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::ShaderModuleFailed,
                req.width,
                req.height,
                format!("vkCreateShaderModule 失败（VkResult=0x{r:08X}）"),
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        shader_modules.push(m);
    }

    // ---- 2. descriptor set layouts（4 个 set，逐 group） ---------------
    // 每个 set 的 binding 数：g0=3, g1=2, g2=8, g3=1。
    for group in 0..4u32 {
        let bindings: Vec<vk::VkDescriptorSetLayoutBinding> = resources
            .iter()
            .filter(|r| r.group == group)
            .map(|r| vk::VkDescriptorSetLayoutBinding {
                binding: r.binding,
                descriptor_type: if r.uniform {
                    vk::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                } else {
                    vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                },
                descriptor_count: 1,
                stage_flags: vk::VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let dslci = vk::VkDescriptorSetLayoutCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut d: vk::VkDescriptorSetLayout = 0;
        let r = (fns.create_descriptor_set_layout.unwrap())(
            device,
            &dslci,
            std::ptr::null(),
            &mut d,
        );
        if r != vk::VK_SUCCESS || d == 0 {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::LayoutFailed,
                req.width,
                req.height,
                format!("vkCreateDescriptorSetLayout(set={group}) 失败（0x{r:08X}）"),
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        dsls.push(d);
    }

    // ---- 3. pipeline layout -------------------------------------------
    let plci = vk::VkPipelineLayoutCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        set_layout_count: dsls.len() as u32,
        p_set_layouts: dsls.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: std::ptr::null(),
    };
    let r = (fns.create_pipeline_layout.unwrap())(
        device,
        &plci,
        std::ptr::null(),
        &mut pipeline_layout,
    );
    if r != vk::VK_SUCCESS || pipeline_layout == 0 {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::LayoutFailed,
            req.width,
            req.height,
            "vkCreatePipelineLayout 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }

    // ---- 4. 5 条 compute pipeline --------------------------------------
    for (i, (entry, _)) in STAGES.iter().enumerate() {
        let entry_c = std::ffi::CString::new(*entry).unwrap_or_default();
        let stage = vk::VkPipelineShaderStageCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: vk::VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_modules[i],
            p_name: entry_c.as_ptr(),
            p_specialization_info: std::ptr::null(),
        };
        let cpci = vk::VkComputePipelineCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut p: vk::VkPipeline = 0;
        let r = (fns.create_compute_pipelines.unwrap())(
            device,
            0,
            1,
            &cpci,
            std::ptr::null(),
            &mut p,
        );
        if r != vk::VK_SUCCESS || p == 0 {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::PipelineFailed,
                req.width,
                req.height,
                format!("vkCreateComputePipelines({entry}) 失败（0x{r:08X}）"),
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        pipelines.push(p);
    }

    // ---- 5. host-visible buffers + 初始数据 ---------------------------
    let mut mem_props = vk::VkPhysicalDeviceMemoryPropertiesStorage::zeroed();
    if let Some(get_mem_props) = ctx.instance_fns.get_physical_device_memory_properties {
        get_mem_props(ctx.physical_device, &mut mem_props);
    }
    for res in &resources {
        let bci = vk::VkBufferCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: res.alloc_bytes(),
            usage: if res.uniform {
                vk::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            } else {
                vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            },
            sharing_mode: vk::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };
        let mut buffer: vk::VkBuffer = 0;
        if (fns.create_buffer.unwrap())(device, &bci, std::ptr::null(), &mut buffer)
            != vk::VK_SUCCESS
            || buffer == 0
        {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::BufferFailed,
                req.width,
                req.height,
                format!(
                    "vkCreateBuffer 失败（set{} binding{}）",
                    res.group, res.binding
                ),
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        let mut req_mem = vk::VkMemoryRequirements::default();
        (fns.get_buffer_memory_requirements.unwrap())(device, buffer, &mut req_mem);
        let Some(type_index) = mem_props.pick_host_visible(req_mem.memory_type_bits) else {
            (fns.destroy_buffer.unwrap())(device, buffer, std::ptr::null());
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::BufferFailed,
                req.width,
                req.height,
                "无 host-visible coherent 内存类型",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        };
        let mai = vk::VkMemoryAllocateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: req_mem.size,
            memory_type_index: type_index,
        };
        let mut memory: vk::VkDeviceMemory = 0;
        if (fns.allocate_memory.unwrap())(device, &mai, std::ptr::null(), &mut memory)
            != vk::VK_SUCCESS
            || memory == 0
        {
            (fns.destroy_buffer.unwrap())(device, buffer, std::ptr::null());
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::BufferFailed,
                req.width,
                req.height,
                "vkAllocateMemory 失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        if (fns.bind_buffer_memory.unwrap())(device, buffer, memory, 0) != vk::VK_SUCCESS {
            (fns.destroy_buffer.unwrap())(device, buffer, std::ptr::null());
            (fns.free_memory.unwrap())(device, memory, std::ptr::null());
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::BufferFailed,
                req.width,
                req.height,
                "vkBindBufferMemory 失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        // 清零 + 写入初始字节（确定性：未写到的尾部恒 0）。
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        if (fns.map_memory.unwrap())(device, memory, 0, vk::VK_WHOLE_SIZE, 0, &mut mapped)
            == vk::VK_SUCCESS
            && !mapped.is_null()
        {
            let dst = mapped as *mut u8;
            std::ptr::write_bytes(dst, 0, req_mem.size as usize);
            std::ptr::copy_nonoverlapping(
                res.bytes.as_ptr(),
                dst,
                res.bytes.len().min(req_mem.size as usize),
            );
            (fns.unmap_memory.unwrap())(device, memory);
        } else {
            (fns.destroy_buffer.unwrap())(device, buffer, std::ptr::null());
            (fns.free_memory.unwrap())(device, memory, std::ptr::null());
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::BufferFailed,
                req.width,
                req.height,
                "vkMapMemory 写初始数据失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        buffers.push((buffer, memory));
    }

    // ---- 6. descriptor pool + 4 sets + writes --------------------------
    let uniform_count = resources.iter().filter(|r| r.uniform).count() as u32;
    let storage_count = resources.len() as u32 - uniform_count;
    let pool_sizes = [
        vk::VkDescriptorPoolSize {
            ty: vk::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptor_count: uniform_count,
        },
        vk::VkDescriptorPoolSize {
            ty: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: storage_count,
        },
    ];
    let dpci = vk::VkDescriptorPoolCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        max_sets: 4,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
    };
    if (fns.create_descriptor_pool.unwrap())(device, &dpci, std::ptr::null(), &mut pool)
        != vk::VK_SUCCESS
        || pool == 0
    {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::LayoutFailed,
            req.width,
            req.height,
            "vkCreateDescriptorPool 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }
    let dsai = vk::VkDescriptorSetAllocateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        descriptor_pool: pool,
        descriptor_set_count: 4,
        p_set_layouts: dsls.as_ptr(),
    };
    if (fns.allocate_descriptor_sets.unwrap())(device, &dsai, sets.as_mut_ptr())
        != vk::VK_SUCCESS
    {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::LayoutFailed,
            req.width,
            req.height,
            "vkAllocateDescriptorSets 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }
    let buffer_infos: Vec<vk::VkDescriptorBufferInfo> = buffers
        .iter()
        .map(|(b, _)| vk::VkDescriptorBufferInfo {
            buffer: *b,
            offset: 0,
            range: vk::VK_WHOLE_SIZE,
        })
        .collect();
    let writes: Vec<vk::VkWriteDescriptorSet> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| vk::VkWriteDescriptorSet {
            s_type: vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: std::ptr::null(),
            dst_set: sets[r.group as usize],
            dst_binding: r.binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: if r.uniform {
                vk::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            } else {
                vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            },
            p_image_info: std::ptr::null(),
            p_buffer_info: &buffer_infos[i],
            p_texel_buffer_view: std::ptr::null(),
        })
        .collect();
    (fns.update_descriptor_sets.unwrap())(device, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());

    // ---- 7. command buffer --------------------------------------------
    let cpci = vk::VkCommandPoolCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_index: ctx.graphics_queue_family,
    };
    if (fns.create_command_pool.unwrap())(device, &cpci, std::ptr::null(), &mut cmd_pool)
        != vk::VK_SUCCESS
        || cmd_pool == 0
    {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::CommandFailed,
            req.width,
            req.height,
            "vkCreateCommandPool 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }
    let cbai = vk::VkCommandBufferAllocateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        command_pool: cmd_pool,
        level: vk::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        command_buffer_count: 1,
    };
    if (fns.allocate_command_buffers.unwrap())(device, &cbai, &mut cmd) != vk::VK_SUCCESS
        || cmd.is_null()
    {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::CommandFailed,
            req.width,
            req.height,
            "vkAllocateCommandBuffers 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }
    let cbbi = vk::VkCommandBufferBeginInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        p_inheritance_info: std::ptr::null(),
    };
    if (fns.begin_command_buffer.unwrap())(cmd, &cbbi) != vk::VK_SUCCESS {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::CommandFailed,
            req.width,
            req.height,
            "vkBeginCommandBuffer 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }

    // stage 序列：generate → [shade, connect, extend] × max_bounces → accumulate。
    let mut stage_plan: Vec<usize> = Vec::new();
    stage_plan.push(0); // generate
    for _ in 0..req.max_bounces.max(1) {
        stage_plan.push(1); // shade
        stage_plan.push(2); // connect
        stage_plan.push(3); // extend
    }
    stage_plan.push(4); // accumulate
    // W1 的逐阶段定位：截到第 k 段（`k >= len` 等价于完整计划）。
    if let Some(k) = req.stage_stop {
        stage_plan.truncate(k.clamp(1, stage_plan.len()));
    }

    let mut dispatches = 0u32;
    for (step, &stage) in stage_plan.iter().enumerate() {
        (fns.cmd_bind_pipeline.unwrap())(
            cmd,
            vk::VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelines[stage],
        );
        (fns.cmd_bind_descriptor_sets.unwrap())(
            cmd,
            vk::VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            4,
            sets.as_ptr(),
            0,
            std::ptr::null(),
        );
        (fns.cmd_dispatch.unwrap())(
            cmd,
            dispatch_groups(req, STAGES[stage].0),
            1,
            1,
        );
        dispatches += 1;
        if step + 1 < stage_plan.len() {
            // 全屏障：上一 stage 的 storage 写对下一 stage 可见。
            let barrier = vk::VkMemoryBarrier {
                s_type: vk::VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: vk::VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: vk::VK_ACCESS_SHADER_READ_BIT
                    | vk::VK_ACCESS_SHADER_WRITE_BIT,
            };
            (fns.cmd_pipeline_barrier.unwrap())(
                cmd,
                vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &barrier,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            );
        }
    }
    if (fns.end_command_buffer.unwrap())(cmd) != vk::VK_SUCCESS {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::CommandFailed,
            req.width,
            req.height,
            "vkEndCommandBuffer 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }

    // ---- 8. submit + fence --------------------------------------------
    let fci = vk::VkFenceCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
    };
    let used_fence = (fns.create_fence.unwrap())(device, &fci, std::ptr::null(), &mut fence)
        == vk::VK_SUCCESS
        && fence != 0;
    let submit = vk::VkSubmitInfo {
        s_type: vk::VK_STRUCTURE_TYPE_SUBMIT_INFO,
        p_next: std::ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: std::ptr::null(),
        p_wait_dst_stage_mask: std::ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        signal_semaphore_count: 0,
        p_signal_semaphores: std::ptr::null(),
    };
    if (fns.queue_submit.unwrap())(ctx.queue, 1, &submit, if used_fence { fence } else { 0 })
        != vk::VK_SUCCESS
    {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::SubmitFailed,
            req.width,
            req.height,
            "vkQueueSubmit 失败",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }
    if used_fence {
        const FENCE_TIMEOUT_NS: u64 = 10_000_000_000;
        if (fns.wait_for_fences.unwrap())(device, 1, &fence, 1, FENCE_TIMEOUT_NS)
            != vk::VK_SUCCESS
        {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::SubmitFailed,
                req.width,
                req.height,
                "vkWaitForFences 超时/失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
    } else if let Some(wait_idle) = fns.queue_wait_idle {
        if wait_idle(ctx.queue) != vk::VK_SUCCESS {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::SubmitFailed,
                req.width,
                req.height,
                "vkQueueWaitIdle 失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
    } else {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::SubmitFailed,
            req.width,
            req.height,
            "无 fence 且无 vkQueueWaitIdle：无法确认 GPU 完成",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }

    // ---- 9. 读回 accum -------------------------------------------------
    // accum 是第 3 个资源（index 2）。
    let accum_index = 2usize;
    let npix = (req.width as usize) * (req.height as usize);
    let mut hdr: Vec<[f32; 4]> = Vec::with_capacity(npix);
    let mut pixels: Vec<u8> = Vec::new();
    if let (Some(&(_b, mem)), Some(res)) = (buffers.get(accum_index), resources.get(accum_index))
    {
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        if (fns.map_memory.unwrap())(device, mem, 0, vk::VK_WHOLE_SIZE, 0, &mut mapped)
            != vk::VK_SUCCESS
            || mapped.is_null()
        {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::ReadbackFailed,
                req.width,
                req.height,
                "vkMapMemory 读回 accum 失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        let src = mapped as *const u8;
        let n = res.bytes.len().min(npix * 16);
        let bytes = std::slice::from_raw_parts(src, n);
        match req.mode {
            WavefrontMode::CpuRefCompat => {
                // 每像素 16 B；第一个 f32 承载打包后的 u32 位型。
                pixels.reserve(npix * 4);
                for i in 0..npix {
                    let o = i * 16;
                    if o + 4 > bytes.len() {
                        break;
                    }
                    let packed = u32::from_le_bytes([
                        bytes[o],
                        bytes[o + 1],
                        bytes[o + 2],
                        bytes[o + 3],
                    ]);
                    pixels.extend_from_slice(&packed.to_le_bytes());
                }
            }
            WavefrontMode::Path => {
                for i in 0..npix {
                    let o = i * 16;
                    if o + 16 > bytes.len() {
                        break;
                    }
                    let c = [
                        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
                        f32::from_le_bytes([
                            bytes[o + 4],
                            bytes[o + 5],
                            bytes[o + 6],
                            bytes[o + 7],
                        ]),
                        f32::from_le_bytes([
                            bytes[o + 8],
                            bytes[o + 9],
                            bytes[o + 10],
                            bytes[o + 11],
                        ]),
                        f32::from_le_bytes([
                            bytes[o + 12],
                            bytes[o + 13],
                            bytes[o + 14],
                            bytes[o + 15],
                        ]),
                    ];
                    hdr.push(c);
                    // 预览量化（与 `Framebuffer::set_pixel` 同式）：
                    pixels.push(quantize_u8(c[0]));
                    pixels.push(quantize_u8(c[1]));
                    pixels.push(quantize_u8(c[2]));
                    pixels.push(255);
                }
            }
        }
        (fns.unmap_memory.unwrap())(device, mem);
    } else {
        cleanup!();
        let mut rep = WavefrontReport::failed(
            DispatchStatus::ReadbackFailed,
            req.width,
            req.height,
            "accum 资源缺失",
        );
        rep.device = ctx.device_name.clone();
        return rep;
    }

    // ---- 10. 可选：读回 slots（W1 逐阶段定位） -------------------------
    let mut slot_bytes: Vec<u8> = Vec::new();
    if req.readback_slots {
        // slots 是第 1 个资源（index 0）。
        let Some(&(_sb, smem)) = buffers.first() else {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::ReadbackFailed,
                req.width,
                req.height,
                "slots 资源缺失",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        };
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        if (fns.map_memory.unwrap())(device, smem, 0, vk::VK_WHOLE_SIZE, 0, &mut mapped)
            != vk::VK_SUCCESS
            || mapped.is_null()
        {
            cleanup!();
            let mut rep = WavefrontReport::failed(
                DispatchStatus::ReadbackFailed,
                req.width,
                req.height,
                "vkMapMemory 读回 slots 失败",
            );
            rep.device = ctx.device_name.clone();
            return rep;
        }
        let n = resources[0].bytes.len();
        slot_bytes =
            std::slice::from_raw_parts(mapped as *const u8, n).to_vec();
        (fns.unmap_memory.unwrap())(device, smem);
    }

    cleanup!();
    WavefrontReport {
        status: DispatchStatus::Ok,
        width: req.width,
        height: req.height,
        dispatches,
        device: ctx.device_name.clone(),
        detail: String::new(),
        pixels,
        hdr,
        buffer_bytes: total_bytes,
        slot_bytes,
    }
}

/// `clamp(0,1) * 255` 四舍五入（与 `Framebuffer::set_pixel` 的 `channel_to_u8` 同式）。
#[cfg(not(target_arch = "wasm32"))]
fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// **派发入口**（有设备真跑；无设备如实报 `DeviceUnavailable`）。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn dispatch(
    ctx: Option<&VulkanContext>,
    req: &WavefrontRequest,
) -> WavefrontReport {
    // 先做纯数据守卫（无设备也成立）：零轴 / 超大尺寸 / 路径数溢出。
    if req.width == 0
        || req.height == 0
        || req.spp == 0
        || req.width > crate::MAX_FRAMEBUFFER_DIM
        || req.height > crate::MAX_FRAMEBUFFER_DIM
        || (req.width as u64) * (req.height as u64) * (req.spp as u64) > (1u64 << 31)
    {
        return WavefrontReport::failed(
            DispatchStatus::BufferFailed,
            req.width,
            req.height,
            "尺寸/路径数非法（零轴 / > MAX_FRAMEBUFFER_DIM / 路径数 > 2^31）",
        );
    }
    match ctx {
        None => unavailable(
            DispatchStatus::DeviceUnavailable,
            req,
            "no-vulkan-icd：loader 在位但 vkCreateInstance 失败（当前平台无 Vulkan ICD）",
        ),
        Some(ctx) => {
            // SAFETY: `ctx` 由 `VulkanBackend` 持有，device/queue 遵循类型不变量。
            unsafe { run_on_device(ctx, req) }
        }
    }
}

/// wasm32：无原生 Vulkan，如实报不支持（不 panic、不造帧）。
#[cfg(target_arch = "wasm32")]
pub(crate) fn dispatch(_ctx: (), req: &WavefrontRequest) -> WavefrontReport {
    unavailable(
        DispatchStatus::UnsupportedPlatform,
        req,
        "wasm32 无原生 Vulkan（走 WebGPU 后端）",
    )
}
