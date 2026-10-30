//! 场景渲染 ABI 单一真源（V2.2-A-1 / T03）。
//!
//! # 目的
//!
//! V2.2-A-1 之前，场景 ABI 的权威散落在多处手写源之间：
//! Vulkan 发射器（`vulkan/spirv_rt.rs`）、WebGPU WGSL（`shaders/scene_frame.wgsl`）、
//! 以及 `metal/render.rs`（历史资产，本仓不提供独立 Metal 后端）。本模块把**入口名、工作组尺寸、
//! 绑定剖面、槽位数、参数块布局**与**编码器**收拢为唯一权威，所有宿主（SPIR-V
//! 发射器、WGSL、WebGPU 胶水、测试断言）都从这里取数，禁止在任何源里写字面量。
//!
//! # 零回归
//!
//! 既有路径（`vulkan/spirv_rt.rs` 的发射器、`vulkan/scene.rs` 的 dispatch 编码、
//! `metal/render.rs` 的同构 MSL 编码、以及它们的单测）**完全不改**——
//! `spirv_rt` 用 `pub use crate::scene_abi::*` 重导出本模块的符号，语义逐位一致。
//!
//! # 绑定剖面（V1 / V2）
//!
//! 两个剖面并存（[`SceneAbiProfile`]），**输出恒为最后一个绑定**，
//! 「每个 group 的绑定须从 0 起连续」的硬约束在两者上都成立：
//!
//! | 剖面 | 绑定 | 顺序 |
//! |---|---|---|
//! | [`SceneAbiProfile::V1`] | 4 | params / spheres / lights / **out** |
//! | [`SceneAbiProfile::V2`]（A-2 起的默认） | 6 | params / spheres / lights / triangles / nodes / **out** |
//!
//! V1 是 V2.1 及以前三份源与 `vulkan::scene` dispatch 编码所使用的剖面，**永久可用、
//! 不得回归**（[`SceneAbi::bindings_for`]`(`[`SceneAbiProfile::V1`]`)`）。A-2 只把
//! **默认**剖面升到 V2（[`SceneAbiProfile::CURRENT`]），并在参数块尾部追加
//! `P_NUM_TRIS` / `P_NUM_NODES` 两字（[`PARAMS_WORDS_V2`]，V1 的 22 字前缀逐字节不变）。
//!
//! # 参数块（binding 0）
//!
//! 22 个 u32（`PARAMS_WORDS`）：浮点一律以 **f32 位型**传入（CPU 侧 `to_bits()`，
//! shader 侧 `bitcast<f32>` 还原），**不存在第二套十进制/文本往返口径**。
//!
//! # 诚实边界（G4）
//!
//! 本模块只定义「布局与编码」。能不能真跑由各自后端的「可用自检」回答；本模块
//! 不声称任何后端「可用 / 已点亮」。

use crate::{CameraReq, LightReq, SceneRenderRequest, SphereReq};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 入口 / 工作组 / 槽位（所有宿主必须与此一致）
// ---------------------------------------------------------------------------

/// 光追场景 kernel 的入口名（`OpEntryPoint` / WGSL `@compute fn`）。
///
/// WGSL（`scene_frame.wgsl`）与 SPIR-V（`spirv_rt::MODULE_SCENE`）**同名**——
/// 本仓不提供独立 Metal 后端（不登记为宿主目标）。这是 [`qa_scene_frame_abi`]
/// 的锚定基准。改这里必须同步两份真实宿主源（历史 `metal/render.rs` 沿用同名入口）。
pub const ENTRY: &str = "scene_main";

/// 工作组尺寸 `(x, y, z)`，与背景 kernel 同为 8×8 的二维 tile。
///
/// 两端同值：`scene_frame.wgsl` 的 `@workgroup_size(8, 8, 1)`、SPIR-V 的
/// `LocalSize`；Metal 无独立目标（本仓不提供）。
pub const WORKGROUP: [u32; 3] = [8, 8, 1];

/// 参数块的总字数（u32 个数）。
pub const PARAMS_WORDS: usize = 22;
/// 每球占用的 f32 槽数：`{center.xyz, radius, albedo.xyz, pad}`。
pub const SPHERE_SLOTS: u32 = 8;
/// 每灯占用的 f32 槽数：`{dir.xyz（已归一化）, color.xyz, intensity, pad}`。
pub const LIGHT_SLOTS: u32 = 8;

/// 球体/光源数量的上限（防御性：避免调用方传入天文数字导致巨额缓冲分配）。
///
/// 现实场景远小于此；超限如实拒绝（`InvalidArguments`），调用方回退 CPU。
pub const MAX_SCENE_SPHERES: usize = 1 << 16;
/// 光源数量上限（同上）。
pub const MAX_SCENE_LIGHTS: usize = 1 << 12;

/// 参数块各字的偏移（u32 槽位）。浮点走位型，shader 侧 `bitcast<f32>` 还原。
// ---------------------------------------------------------------------------
pub const P_WIDTH: u32 = 0;
/// 画幅高（像素）。
pub const P_HEIGHT: u32 = 1;
/// 相机视点 `eye.x` 的 f32 位型。
pub const P_EYE: u32 = 2;
/// 相机前向量 `forward.xyz` 的 f32 位型（起点）。
pub const P_FORWARD: u32 = 5;
/// 相机右向量 `right.xyz` 的 f32 位型（起点）。
pub const P_RIGHT: u32 = 8;
/// 相机上向量 `up.xyz` 的 f32 位型（起点）。
pub const P_UP: u32 = 11;
/// `tan(fov/2)` 的 f32 位型（CPU 侧 `det_trig::tan` 一次算好，D-29）。
pub const P_TAN_HALF_FOV: u32 = 14;
/// 画幅比 `width / height` 的 f32 位型。
pub const P_ASPECT: u32 = 15;
/// `teal_orange` 开关（0 / 1）。
pub const P_TEAL_ORANGE: u32 = 16;
/// `contrast` 的 f32 位型。
pub const P_CONTRAST: u32 = 17;
/// `contrast != 1.0` 的预计算开关（0 / 1）：CPU 侧判定，shader 不做浮点不等比较。
pub const P_CONTRAST_ON: u32 = 18;
/// `filmic` 开关（0 / 1）。
pub const P_FILMIC: u32 = 19;
/// 球体个数。
pub const P_NUM_SPHERES: u32 = 20;
/// 光源个数。
pub const P_NUM_LIGHTS: u32 = 21;
/// （V2）三角形个数。
pub const P_NUM_TRIS: u32 = 22;
/// （V2）扁平 BVH 节点个数。
pub const P_NUM_NODES: u32 = 23;

/// **V2** 参数块字数：V1 的 22 字前缀逐字节不变，尾部追加
/// `P_NUM_TRIS` / `P_NUM_NODES` 两字。
pub const PARAMS_WORDS_V2: usize = 24;

/// 每三角形占用的 f32 槽数：`{a.xyz, pad, b.xyz, pad, c.xyz, pad, pad, pad}`（12 = 48 B，
/// 满足 WGSL `vec3<f32>` 的 16 B 对齐：每个 `vec3` 后补 1 个 pad）。
pub const TRIANGLE_SLOTS: u32 = 12;

/// 每 BVH 节点占用的 f32 槽数：`{bmin.xyz, pad, bmax.xyz, pad, start, count, right_child, pad}`
/// （12 = 48 B，与设计 §1.2.2 的扁平化布局一致）。
pub const NODE_SLOTS: u32 = 12;

/// 曲线三角化的弦高容忍（世界单位）。三角化在 **CPU** 侧一次完成（
/// `prism_geometry::curve_tess`），GPU 只见三角形与扁平 BVH。
pub const CURVE_TESS_TOL: f32 = 0.02;

/// 背景色（与 [`crate::CpuRefBackend`] 的 `render_scene` 的 `(0.04, 0.05, 0.08)` 同值）。
pub const BG_COLOR: [f32; 3] = [0.04, 0.05, 0.08];
/// 环境光（与 CPU 参考的 `Vec3::splat(0.08)` 同值）。
pub const AMBIENT: f32 = 0.08;
/// `normalize` 的退化阈值（与 `prism_math::Vec3::normalize` 的 `1e-8` 同值）。
pub const NORMALIZE_EPS: f32 = 1e-8;
/// teal-orange 暗部染色（与 CPU 参考同值）。
pub const SHADOW_TINT: [f32; 3] = [0.10, 0.35, 0.45];
/// teal-orange 高光染色（与 CPU 参考同值）。
pub const HIGH_TINT: [f32; 3] = [1.0, 0.7, 0.3];

// ---------------------------------------------------------------------------
// 绑定剖面（单一真源；宿主据此派生 bind group 布局与 WGSL 声明）
// ---------------------------------------------------------------------------

/// 每个绑定的访问属性（用于 WGSL `@binding` 声明与宿主 bind group 的交叉锚定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingAccess {
    /// `var<storage, read>`。
    ReadOnly,
    /// `var<storage, read_write>`。
    ReadWrite,
}

/// 一个场景绑定的静态描述（`binding` 编号 + 名字 + 访问属性）。
///
/// 名字只是可读标签；**顺序即 binding 编号**（连续前缀 0..N-1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneBinding {
    /// 绑定编号（`@binding(k)`）。
    pub binding: u32,
    /// 可读名字（params / spheres / lights / out）。
    pub name: &'static str,
    /// 访问属性（read / read_write）。
    pub access: BindingAccess,
}

/// 场景 ABI 单一真源（V1：4 绑定连续前缀 / V2：6 绑定连续前缀）。
///
/// 所有宿主（SPIR-V 发射器、WGSL、WebGPU 胶水、测试断言）都从这里派生期望值；
/// 改绑定/槽位/入口只改本结构 + 同步三份源，禁止在源里写字面量。
pub struct SceneAbi;

/// 场景 ABI 剖面（A-2 起引入；**输出恒为最后一个绑定**，连续前缀约束不破）。
///
/// * [`SceneAbiProfile::V1`]：`{0 params, 1 spheres, 2 lights, 3 out}`（4 绑定）。
///   V2.1 及以前的三份源、`vulkan::scene` 的 dispatch 编码、既有单测全部按此剖面，
///   **不得回归**——本剖面永远可用。
/// * [`SceneAbiProfile::V2`]：`{0 params, 1 spheres, 2 lights, 3 triangles, 4 nodes, 5 out}`
///   （6 绑定）。A-2 曲线图元（三角化 → 扁平 BVH）走此剖面；参数块追加
///   `P_NUM_TRIS` / `P_NUM_NODES` 两字（[`PARAMS_WORDS_V2`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneAbiProfile {
    /// 4 绑定（params / spheres / lights / out）。
    V1,
    /// 6 绑定（params / spheres / lights / triangles / nodes / out）。
    V2,
}

impl SceneAbiProfile {
    /// 当前默认剖面（A-2 交付后为 [`SceneAbiProfile::V2`]）。
    pub const CURRENT: SceneAbiProfile = SceneAbiProfile::V2;

    /// 该剖面的绑定集（连续前缀 `0..N-1`，末位恒为输出）。
    pub fn bindings(self) -> &'static [SceneBinding] {
        match self {
            SceneAbiProfile::V1 => &[
                SceneBinding { binding: 0, name: "params", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 1, name: "spheres", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 2, name: "lights", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 3, name: "out", access: BindingAccess::ReadWrite },
            ],
            SceneAbiProfile::V2 => &[
                SceneBinding { binding: 0, name: "params", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 1, name: "spheres", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 2, name: "lights", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 3, name: "triangles", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 4, name: "nodes", access: BindingAccess::ReadOnly },
                SceneBinding { binding: 5, name: "out", access: BindingAccess::ReadWrite },
            ],
        }
    }

    /// 该剖面的绑定个数（连续前缀长度）。
    pub fn binding_count(self) -> usize {
        self.bindings().len()
    }

    /// 该剖面输出绑定的编号（**恒为最后一个**：`binding_count() - 1`）。
    pub fn out_binding(self) -> u32 {
        (self.binding_count() - 1) as u32
    }

    /// 该剖面参数块的字数（u32 个数）。
    pub fn params_words(self) -> usize {
        match self {
            SceneAbiProfile::V1 => PARAMS_WORDS,
            SceneAbiProfile::V2 => PARAMS_WORDS_V2,
        }
    }

    /// 剖面名（诊断/断言用）。
    pub fn name(self) -> &'static str {
        match self {
            SceneAbiProfile::V1 => "V1",
            SceneAbiProfile::V2 => "V2",
        }
    }
}

impl SceneAbi {
    /// 入口名（WGSL `fn scene_main` / SPIR-V `OpEntryPoint`）。
    pub const ENTRY: &'static str = ENTRY;
    /// 工作组尺寸 `[8, 8, 1]`。
    pub const WORKGROUP: [u32; 3] = WORKGROUP;
    /// 参数块字数（**V1** 契约；V2 见 [`PARAMS_WORDS_V2`]）。
    pub const PARAMS_WORDS: usize = PARAMS_WORDS;
    /// 每球 f32 槽数。
    pub const SPHERE_SLOTS: u32 = SPHERE_SLOTS;
    /// 每灯 f32 槽数。
    pub const LIGHT_SLOTS: u32 = LIGHT_SLOTS;

    /// 当前默认剖面。
    pub const PROFILE: SceneAbiProfile = SceneAbiProfile::CURRENT;

    /// 默认剖面（[`SceneAbiProfile::CURRENT`]）的绑定集。
    ///
    /// A-2 起默认升 V2（6 绑定）。V1 剖面仍可经
    /// [`SceneAbi::bindings_for`]`(`[`SceneAbiProfile::V1`]`)` 取得，**不回归**。
    pub fn bindings() -> &'static [SceneBinding] {
        Self::bindings_for(SceneAbiProfile::CURRENT)
    }

    /// 指定剖面的绑定集（连续前缀 `0..N-1`，末位恒为输出）。
    pub fn bindings_for(profile: SceneAbiProfile) -> &'static [SceneBinding] {
        profile.bindings()
    }

    /// 把一个场景渲染请求编码为三块输入缓冲（与 [`crate::CpuRefBackend`] 逐位一致）。
    ///
    /// 返回 `(params, spheres, lights)`：相机基在内部由 [`camera_basis`] 算好，
    /// 浮点一律位型（`to_bits()`），shader 侧 `bitcast<f32>` 还原。本方法是
    /// `spirv_rt` / `web::device` 等宿主的统一入口——禁止在宿主里另写字面量编码。
    ///
    /// V1 剖面（4 绑定）专用；V2 走 [`SceneAbi::encode_v2`]。
    pub fn encode(
        req: &SceneRenderRequest,
    ) -> ([u8; PARAMS_WORDS * 4], Vec<u8>, Vec<u8>) {
        let basis = camera_basis(&req.camera);
        let params = encode_params(req, &basis);
        let spheres = encode_spheres(&req.spheres);
        let lights = encode_lights(&req.lights);
        (params, spheres, lights)
    }

    /// V2 剖面编码：在 V1 三块缓冲之外追加 `triangles` / `nodes`（扁平 SAH-BVH）。
    ///
    /// 返回 `(params, spheres, lights, triangles, nodes)`；`params` 为
    /// [`PARAMS_WORDS_V2`] 个字（V1 的 22 字前缀**逐字节不变**，仅在尾部追加
    /// `P_NUM_TRIS` / `P_NUM_NODES`）。曲线为空 ⇒ `triangles`/`nodes` 皆空、
    /// 两计数为 0，WGSL 侧遍历段整体跳过 ⇒ 与 V1 输出逐位相同（A-2 的不回归
    /// 机械保证，设计 §4.3）。
    pub fn encode_v2(
        req: &SceneRenderRequest,
    ) -> (
        [u8; PARAMS_WORDS_V2 * 4],
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    ) {
        let basis = camera_basis(&req.camera);
        let mut params = [0u8; PARAMS_WORDS_V2 * 4];
        let v1 = encode_params(req, &basis);
        params[..v1.len()].copy_from_slice(&v1);
        let spheres = encode_spheres(&req.spheres);
        let lights = encode_lights(&req.lights);
        let (tris, nodes) = encode_curve_buffers(&req.curves);
        put_u32(&mut params, P_NUM_TRIS as usize, (tris.len() / (TRIANGLE_SLOTS as usize * 4)) as u32);
        put_u32(
            &mut params,
            P_NUM_NODES as usize,
            (nodes.len() / (NODE_SLOTS as usize * 4)) as u32,
        );
        (params, spheres, lights, tris, nodes)
    }
}

// ---------------------------------------------------------------------------
// CPU 侧：相机基 + 三块输入缓冲的编码（与 [`crate::CpuRefBackend`] 逐式对应）
// ---------------------------------------------------------------------------

/// 相机正交基与投影标量（CPU 一次算好，GPU 侧不重复推导）。
///
/// 与 [`crate::CpuRefBackend`] 的 `render_scene` 的基向量段**逐式对应**：
/// `forward = (target - eye).normalize()`、`right = cross(forward, (0,1,0)).normalize()`、
/// `up = cross(right, forward)`、`tan_half = tan(fov/2)`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBasis {
    /// 视点。
    pub eye: [f32; 3],
    /// 前向单位向量。
    pub forward: [f32; 3],
    /// 右向单位向量。
    pub right: [f32; 3],
    /// 上向单位向量（`cross(right, forward)`，已正交于 forward/right）。
    pub up: [f32; 3],
    /// `tan(fov/2)`。
    pub tan_half_fov: f32,
    /// 画幅比（原样取自请求）。
    pub aspect: f32,
}

/// 由相机请求算正交基（与 CPU 参考同式，D-29：三角走 `det_trig`）。
pub fn camera_basis(cam: &CameraReq) -> CameraBasis {
    let forward_v = (cam.target - cam.eye).normalize();
    let right_v = cross(forward_v, Vec3::new(0.0, 1.0, 0.0)).normalize();
    let up_v = cross(right_v, forward_v);
    CameraBasis {
        eye: [cam.eye.x, cam.eye.y, cam.eye.z],
        forward: [forward_v.x, forward_v.y, forward_v.z],
        right: [right_v.x, right_v.y, right_v.z],
        up: [up_v.x, up_v.y, up_v.z],
        tan_half_fov: prism_math::det_trig::tan(cam.fov * 0.5),
        aspect: cam.aspect,
    }
}

/// `cross(a, b)`（与 `prism_math::Vec3::cross` 同式；本模块不引入第二套口径）。
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(&b)
}

/// 把 f32 写进 `u32` 槽（位型；无十进制往返）。
fn put_f32(out: &mut [u8], slot: usize, v: f32) {
    let off = slot * 4;
    out[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
}

/// 把 u32 写进槽。
fn put_u32(out: &mut [u8], slot: usize, v: u32) {
    let off = slot * 4;
    out[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// 编码参数块（`PARAMS_WORDS` × 4 字节，little-endian）。
///
/// 相机基由 [`camera_basis`] 给出（CPU 一次算好）；三处开关（teal_orange /
/// contrast_on / filmic）在 CPU 侧判定为 u32 标志，shader 只做整数比较。
pub fn encode_params(req: &SceneRenderRequest, basis: &CameraBasis) -> [u8; PARAMS_WORDS * 4] {
    let mut out = [0u8; PARAMS_WORDS * 4];
    put_u32(&mut out, P_WIDTH as usize, req.width);
    put_u32(&mut out, P_HEIGHT as usize, req.height);
    put_f32(&mut out, P_EYE as usize, basis.eye[0]);
    put_f32(&mut out, P_EYE as usize + 1, basis.eye[1]);
    put_f32(&mut out, P_EYE as usize + 2, basis.eye[2]);
    put_f32(&mut out, P_FORWARD as usize, basis.forward[0]);
    put_f32(&mut out, P_FORWARD as usize + 1, basis.forward[1]);
    put_f32(&mut out, P_FORWARD as usize + 2, basis.forward[2]);
    put_f32(&mut out, P_RIGHT as usize, basis.right[0]);
    put_f32(&mut out, P_RIGHT as usize + 1, basis.right[1]);
    put_f32(&mut out, P_RIGHT as usize + 2, basis.right[2]);
    put_f32(&mut out, P_UP as usize, basis.up[0]);
    put_f32(&mut out, P_UP as usize + 1, basis.up[1]);
    put_f32(&mut out, P_UP as usize + 2, basis.up[2]);
    put_f32(&mut out, P_TAN_HALF_FOV as usize, basis.tan_half_fov);
    put_f32(&mut out, P_ASPECT as usize, basis.aspect);
    put_u32(&mut out, P_TEAL_ORANGE as usize, u32::from(req.aesthetic.teal_orange));
    put_f32(&mut out, P_CONTRAST as usize, req.aesthetic.contrast);
    // 与 CPU 的 `if req.aesthetic.contrast != 1.0` 同一判据（含 NaN ⇒ 应用）。
    put_u32(
        &mut out,
        P_CONTRAST_ON as usize,
        u32::from(req.aesthetic.contrast != 1.0),
    );
    put_u32(&mut out, P_FILMIC as usize, u32::from(req.aesthetic.filmic));
    put_u32(&mut out, P_NUM_SPHERES as usize, req.spheres.len() as u32);
    put_u32(&mut out, P_NUM_LIGHTS as usize, req.lights.len() as u32);
    out
}

/// 编码球体缓冲（每球 [`SPHERE_SLOTS`] 个 f32 = 32 字节）。
///
/// 槽内布局：`center.x/y/z`、`radius`、`albedo.r/g/b`、`pad`（恒 0）。
pub fn encode_spheres(spheres: &[SphereReq]) -> Vec<u8> {
    let mut out = Vec::with_capacity(spheres.len() * SPHERE_SLOTS as usize * 4);
    for s in spheres {
        let mut block = [0u8; SPHERE_SLOTS as usize * 4];
        put_f32(&mut block, 0, s.center.x);
        put_f32(&mut block, 1, s.center.y);
        put_f32(&mut block, 2, s.center.z);
        put_f32(&mut block, 3, s.radius);
        put_f32(&mut block, 4, s.albedo.x);
        put_f32(&mut block, 5, s.albedo.y);
        put_f32(&mut block, 6, s.albedo.z);
        // 槽 7 为 pad：恒 0（确定性；不留未初始化字节）。
        out.extend_from_slice(&block);
    }
    out
}

/// 编码光源缓冲（每灯 [`LIGHT_SLOTS`] 个 f32 = 32 字节）。
///
/// 槽内布局：`dir.x/y/z`（**已归一化**，与 CPU 的 `light.direction.normalize()`
/// 同值：由 CPU 调用同一 `Vec3::normalize` 得到，shader 内不再二次归一化）、
/// `color.r/g/b`、`intensity`、`pad`（恒 0）。
pub fn encode_lights(lights: &[LightReq]) -> Vec<u8> {
    let mut out = Vec::with_capacity(lights.len() * LIGHT_SLOTS as usize * 4);
    for l in lights {
        let n = l.direction.normalize();
        let mut block = [0u8; LIGHT_SLOTS as usize * 4];
        put_f32(&mut block, 0, n.x);
        put_f32(&mut block, 1, n.y);
        put_f32(&mut block, 2, n.z);
        put_f32(&mut block, 3, l.color.x);
        put_f32(&mut block, 4, l.color.y);
        put_f32(&mut block, 5, l.color.z);
        put_f32(&mut block, 6, l.intensity);
        out.extend_from_slice(&block);
    }
    out
}

// ---------------------------------------------------------------------------
// V2：曲线图元 → 三角形 + 扁平 BVH（单一权威在 `prism-geometry`）
// ---------------------------------------------------------------------------
//
// 曲线三角化与 BVH 构建**不在此处重写**：权威在叶子 crate `prism-geometry`
// （`curve_tess::tessellate_all` + `bvh::Bvh::build`），本模块只做**扁平化编码**
// （`Aabb`/`Triangle` → `array<f32>` 槽位）。依赖方向 `prism-gpu → prism-geometry`
// 单向无环：`prism-rt → prism-gpu` 已存在，故 `prism-gpu` **不得**反引 `prism-rt`。

/// 把场景里的曲线图元三角化并构建 SAH-BVH，输出两份**扁平** storeage 缓冲。
///
/// 返回 `(triangles, nodes)`：都是 little-endian 的 f32 位型字节流，
/// 步长分别为 [`TRIANGLE_SLOTS`] / [`NODE_SLOTS`] 个 f32。
///
/// 空曲线 ⇒ 两份缓冲皆空（WGSL 遍历段按 `num_tris == 0` 整段跳过）。
pub fn encode_curve_buffers(curves: &[prism_geometry::curve_tess::CurveReq]) -> (Vec<u8>, Vec<u8>) {
    if curves.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let tris = prism_geometry::curve_tess::tessellate_all(curves, CURVE_TESS_TOL);
    if tris.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let bvh = prism_geometry::bvh::Bvh::build(&tris);
    let mut tri_bytes = Vec::with_capacity(bvh.tris.len() * TRIANGLE_SLOTS as usize * 4);
    for t in &bvh.tris {
        let mut block = [0u8; TRIANGLE_SLOTS as usize * 4];
        put_f32(&mut block, 0, t.a.x);
        put_f32(&mut block, 1, t.a.y);
        put_f32(&mut block, 2, t.a.z);
        // 槽 3 = pad
        put_f32(&mut block, 4, t.b.x);
        put_f32(&mut block, 5, t.b.y);
        put_f32(&mut block, 6, t.b.z);
        // 槽 7 = pad
        put_f32(&mut block, 8, t.c.x);
        put_f32(&mut block, 9, t.c.y);
        put_f32(&mut block, 10, t.c.z);
        // 槽 11 = pad
        tri_bytes.extend_from_slice(&block);
    }
    let mut node_bytes = Vec::with_capacity(bvh.nodes.len() * NODE_SLOTS as usize * 4);
    for nd in &bvh.nodes {
        let mut block = [0u8; NODE_SLOTS as usize * 4];
        put_f32(&mut block, 0, nd.bounds.min.x);
        put_f32(&mut block, 1, nd.bounds.min.y);
        put_f32(&mut block, 2, nd.bounds.min.z);
        put_f32(&mut block, 4, nd.bounds.max.x);
        put_f32(&mut block, 5, nd.bounds.max.y);
        put_f32(&mut block, 6, nd.bounds.max.z);
        put_u32(&mut block, 8, nd.start);
        put_u32(&mut block, 9, nd.count);
        put_u32(&mut block, 10, nd.right_child);
        node_bytes.extend_from_slice(&block);
    }
    (tri_bytes, node_bytes)
}

// ---------------------------------------------------------------------------
// 单元测试（native：编码器与常数自洽；不触碰 GPU）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CurveReq;

    fn sample_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 16,
            height: 9,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 16.0 / 9.0,
            },
            lights: vec![LightReq {
                direction: Vec3::new(0.0, 0.0, -1.0),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 1.0,
            }],
            spheres: vec![SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                albedo: Vec3::new(0.8, 0.2, 0.2),
                roughness: 0.4,
                metalness: 0.1,
            }],
            aesthetic: crate::AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        curves: Vec::new(),
        }
    }

    #[test]
    fn params_block_is_exactly_params_words_times_four_bytes() {
        let req = sample_request();
        let basis = camera_basis(&req.camera);
        let p = encode_params(&req, &basis);
        assert_eq!(p.len(), PARAMS_WORDS * 4, "参数块长度必须与 PARAMS_WORDS 严格对应");
        assert_eq!(p.len(), 88);
        // 宽度/高度是 u32 字面量（非位型）：第 0/1 槽直接可还原。
        assert_eq!(u32::from_le_bytes([p[0], p[1], p[2], p[3]]), 16);
        assert_eq!(u32::from_le_bytes([p[4], p[5], p[6], p[7]]), 9);
    }

    #[test]
    fn sphere_block_is_eight_floats_per_sphere() {
        let req = sample_request();
        let s = encode_spheres(&req.spheres);
        assert_eq!(s.len(), SPHERE_SLOTS as usize * 4, "每球必须恰为 SPHERE_SLOTS 个 f32");
        assert_eq!(s.len(), 32);
        // 槽 3 = radius（位型 f32）应等于 1.0。
        let r = f32::from_le_bytes([s[12], s[13], s[14], s[15]]);
        assert_eq!(r, 1.0);
        // 槽 7 = pad 恒 0。
        assert_eq!(&s[28..32], &[0u8; 4]);
    }

    #[test]
    fn light_direction_is_normalized_before_encoding() {
        let req = sample_request();
        let l = encode_lights(&req.lights);
        assert_eq!(l.len(), LIGHT_SLOTS as usize * 4);
        let dx = f32::from_le_bytes([l[0], l[1], l[2], l[3]]);
        let dy = f32::from_le_bytes([l[4], l[5], l[6], l[7]]);
        let dz = f32::from_le_bytes([l[8], l[9], l[10], l[11]]);
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "编码前必须已归一化（与 CPU 参考同口径）");
    }

    /// **V1 剖面不回归**：4 绑定连续前缀 `{0,1,2,3}` 仍可取用、语义不变。
    ///
    /// A-2 把**默认**剖面升到 V2，但 V1 剖面必须一直可用（`vulkan::scene` 的既有
    /// dispatch 编码与 V2.1 基线按此剖面），故本断言改为显式指定 V1 剖面。
    #[test]
    fn bindings_v1_profile_is_continuous_prefix_zero_through_three() {
        // 「连续前缀」硬约束：binding 必须从 0 起连续，且输出恒为最后一个。
        let bs = SceneAbi::bindings_for(SceneAbiProfile::V1);
        assert_eq!(bs.len(), 4, "V1 剖面恒为 4 绑定");
        for (i, b) in bs.iter().enumerate() {
            assert_eq!(b.binding, i as u32, "绑定必须从 0 起连续");
        }
        assert_eq!(bs[3].name, "out");
        assert_eq!(bs[3].access, BindingAccess::ReadWrite, "输出缓冲可读写");
        assert_eq!(bs[0].access, BindingAccess::ReadOnly);
        // 剖面元数据自洽。
        assert_eq!(SceneAbiProfile::V1.binding_count(), 4);
        assert_eq!(SceneAbiProfile::V1.out_binding(), 3);
        assert_eq!(SceneAbiProfile::V1.params_words(), PARAMS_WORDS);
        assert_eq!(SceneAbiProfile::V1.name(), "V1");
    }

    /// **V2 剖面（A-2）**：6 绑定连续前缀，输出恒为最后一个（`binding(5) == out`）。
    #[test]
    fn bindings_v2_profile_is_continuous_prefix_zero_through_five() {
        let bs = SceneAbi::bindings_for(SceneAbiProfile::V2);
        assert_eq!(bs.len(), 6, "V2 剖面为 6 绑定");
        for (i, b) in bs.iter().enumerate() {
            assert_eq!(b.binding, i as u32, "绑定必须从 0 起连续（连续前缀约束）");
        }
        // 顺序即语义：params / spheres / lights / triangles / nodes / out。
        let names: Vec<&str> = bs.iter().map(|b| b.name).collect();
        assert_eq!(
            names,
            vec!["params", "spheres", "lights", "triangles", "nodes", "out"],
            "V2 绑定顺序必须与设计 §1.2.2 逐字一致"
        );
        // 输出**恒为最后一个绑定**。
        let last = bs[bs.len() - 1];
        assert_eq!(last.name, "out");
        assert_eq!(last.access, BindingAccess::ReadWrite);
        assert_eq!(SceneAbiProfile::V2.out_binding(), last.binding);
        assert_eq!(SceneAbiProfile::V2.out_binding(), 5);
        assert_eq!(SceneAbiProfile::V2.binding_count(), 6);
        // 除输出外全部只读。
        for b in &bs[..bs.len() - 1] {
            assert_eq!(b.access, BindingAccess::ReadOnly, "{} 必须只读", b.name);
        }
        assert_eq!(SceneAbiProfile::V2.params_words(), PARAMS_WORDS_V2);
        assert_eq!(PARAMS_WORDS_V2, PARAMS_WORDS + 2, "V2 仅在 V1 尾部追加两字");
    }

    /// 默认剖面 = V2（A-2 交付后）；`SceneAbi::bindings()` 跟随默认。
    #[test]
    fn default_profile_is_v2_and_agrees_with_current() {
        assert_eq!(SceneAbi::PROFILE, SceneAbiProfile::V2);
        assert_eq!(SceneAbi::PROFILE, SceneAbiProfile::CURRENT);
        assert_eq!(SceneAbi::bindings(), SceneAbi::bindings_for(SceneAbiProfile::V2));
        assert_eq!(SceneAbi::bindings().len(), 6);
    }

    /// V2 参数块的 V1 前缀**逐字节不变**（只追加 P_NUM_TRIS / P_NUM_NODES）。
    #[test]
    fn v2_params_prefix_is_bytewise_v1() {
        let req = sample_request();
        let (p1, _, _) = SceneAbi::encode(&req);
        let (p2, _, _, t2, n2) = SceneAbi::encode_v2(&req);
        assert_eq!(p2.len(), PARAMS_WORDS_V2 * 4);
        assert_eq!(&p2[..p1.len()], &p1[..], "V2 的 22 字前缀必须与 V1 逐字节相同");
        // 空曲线 ⇒ 两份曲线缓冲皆空、两计数为 0（A-2 不回归的机械保证）。
        assert!(t2.is_empty() && n2.is_empty());
        let num_tris = u32::from_le_bytes([p2[88], p2[89], p2[90], p2[91]]);
        let num_nodes = u32::from_le_bytes([p2[92], p2[93], p2[94], p2[95]]);
        assert_eq!((num_tris, num_nodes), (0, 0), "空曲线 ⇒ 计数必须为 0");
    }

    /// 曲线编码：三角化 → 扁平 BVH，槽位步长与计数自洽。
    #[test]
    fn v2_curve_buffers_are_flat_and_counted() {
        let mut req = sample_request();
        req.curves = vec![CurveReq::Torus {
            center: Vec3::new(0.0, 0.0, 0.0),
            major: 0.7,
            minor: 0.3,
            axis: Vec3::new(0.0, 1.0, 0.0),
        }];
        let (p, _, _, tris, nodes) = SceneAbi::encode_v2(&req);
        assert!(!tris.is_empty() && !nodes.is_empty(), "合法曲线必须产出三角形与节点");
        assert_eq!(tris.len() % (TRIANGLE_SLOTS as usize * 4), 0, "三角形步长自洽");
        assert_eq!(nodes.len() % (NODE_SLOTS as usize * 4), 0, "节点步长自洽");
        let num_tris = u32::from_le_bytes([p[88], p[89], p[90], p[91]]) as usize;
        let num_nodes = u32::from_le_bytes([p[92], p[93], p[94], p[95]]) as usize;
        assert_eq!(num_tris, tris.len() / (TRIANGLE_SLOTS as usize * 4));
        assert_eq!(num_nodes, nodes.len() / (NODE_SLOTS as usize * 4));
        assert!(num_nodes >= 1, "至少一个根节点");
        // 确定性：同输入两次编码逐字节相同。
        let (_, _, _, tris2, nodes2) = SceneAbi::encode_v2(&req);
        assert_eq!(tris, tris2);
        assert_eq!(nodes, nodes2);
    }

    #[test]
    fn camera_basis_is_orthonormal() {
        let req = sample_request();
        let b = camera_basis(&req.camera);
        let f = Vec3::new(b.forward[0], b.forward[1], b.forward[2]);
        let r = Vec3::new(b.right[0], b.right[1], b.right[2]);
        let u = Vec3::new(b.up[0], b.up[1], b.up[2]);
        assert!((f.length() - 1.0).abs() < 1e-5);
        assert!((r.length() - 1.0).abs() < 1e-5);
        assert!((u.length() - 1.0).abs() < 1e-5);
        // right ⟂ forward。
        assert!((f.dot(&r)).abs() < 1e-5);
        assert_eq!(b.aspect, 16.0 / 9.0);
    }
}
