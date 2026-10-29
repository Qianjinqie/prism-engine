//! 光追场景 kernel 的 SPIR-V 生成（自研发射器，零第三方依赖 — D-15）。
//!
//! # 与 `spirv_bg` 的关系
//!
//! 与 [`super::spirv_bg`] **同级**：同一套 id 分配 / 段序 / 常量去重 /
//! `NoContraction` 装饰纪律，区别只在**语义**与**控制流**：
//!
//! | | `spirv_bg`（背景帧） | 本模块（光追场景） |
//! |---|---|---|
//! | 语义 | 逐点渐变 + 噪声 + 暗角 | 逐像素主射线 → 球面求交 → 直接光 → aesthetic |
//! | 控制流 | 一层 `OpSelectionMerge`（尾组守卫） | 两层 `OpLoopMerge`（球/灯动态循环）+ 两层选择 |
//! | 循环携带值 | 无 | 局部变量（`Function` 存储类 + `OpStore`/`OpLoad`，不用 `OpPhi`） |
//! | 绑定 | 2（参数 / 输出） | 4（参数 / 球 / 灯 / 输出） |
//!
//! # 判据（与 [`crate::CpuRefBackend`] 的 `render_scene` 逐式对应，严禁代数重排）
//!
//! 每个 invocation `(gx, gy)`：
//!
//! ```text
//! if gx >= w || gy >= h: return                      // 尾组守卫
//! dir   = normalize(forward + right*ndx + up*ndy)     // 相机基由 CPU 一次算好
//! loop i < num_spheres:                              // 动态循环（不展开）
//!     disc = b*b - c
//!     if disc >= 0 && t > 0 && t < closest: closest = t; hit = i   // 用 OpSelect 落值
//! color = (0.04, 0.05, 0.08)
//! if hit < num_spheres:                              // 命中 ⇒ 着色
//!     loop j < num_lights: color += albedo*(ambient + lcolor*(diff*intensity))
//! if teal_orange: ...; if contrast_on: ...; if filmic: color = color/(color+1)
//! out[gy*w + gx] = pack_rgba8(clamp01(color))
//! ```
//!
//! 三条纪律（与 `spirv_bg` 同源，逐条有测试）：
//!
//! 1. **同式**：每个乘加都按 Rust 参考的求值树（左结合、先乘后加）发射；
//! 2. **禁收缩**：每条浮点算术结果打 `NoContraction`——否则驱动可把
//!    `a*b + c` 熔成 FMA，末位与 CPU 不同；
//! 3. **无超越函数**（D-29）：只用 `sqrt`/`floor`/`fmax`/`fclamp`
//!    （`GLSL.std.450` 的精确/正确舍入项），`tan(fov/2)` 由 CPU 用
//!    `prism_math::det_trig` 算好以 **f32 位型**传入。
//!
//! # 绑布局（4 binding，均为 std430 存储缓冲）
//!
//! | b | 内容 | 槽位 |
//! |---|---|---|
//! | 0 | Params（`PARAMS_WORDS` 个 u32；浮点走位型，shader 用 `OpBitcast` 还原） | 22 |
//! | 1 | spheres：每球 `{center.xyz, radius, albedo.xyz, pad}` | `8 × num_spheres` |
//! | 2 | lights：每灯 `{dir.xyz（已归一化）, color.xyz, intensity, pad}` | `8 × num_lights` |
//! | 3 | 输出：每像素一个 little-endian 打包 RGBA8（`r \| g<<8 \| b<<16 \| 255<<24`） | `w × h` |
//!
//! # 为什么用「Function 局部变量」而不是 `OpPhi`
//!
//! 球/灯循环的携带值（`closest_t`、`hit_idx`、计数器、颜色累加器）必须跨迭代
//! 可见。`OpPhi` 要求为**每个前驱块**列一对 `(值, 块)`——写错一个前驱就是
//! 「驱动建管线时崩溃」级别的缺陷。改用 `Function` 存储类的局部变量
//! （`OpStore`/`OpLoad`）后，语义与 Rust 参考的 `let mut` 逐句对应：
//! 不需要列前驱、不需要维护支配树，且结构校验能覆盖（见
//! [`super::spirv::validate`] 的 `check_control_flow`）。
//! 代价是依赖驱动的 mem2reg 优化——lavapipe（LLVM）会做掉。

use std::sync::OnceLock;

use super::spirv::{
    inst, BUILTIN_GLOBAL_INVOCATION_ID, DECORATION_ARRAY_STRIDE, DECORATION_BINDING,
    DECORATION_BUFFER_BLOCK, DECORATION_BUILT_IN, DECORATION_DESCRIPTOR_SET,
    DECORATION_NO_CONTRACTION, DECORATION_OFFSET, EXECUTION_MODE_LOCAL_SIZE,
    EXECUTION_MODEL_GLCOMPUTE, MAGIC, OP_ACCESS_CHAIN, OP_BITCAST, OP_BITWISE_OR, OP_BRANCH,
    OP_BRANCH_CONDITIONAL, OP_CAPABILITY, OP_COMPOSITE_EXTRACT, OP_CONSTANT, OP_CONVERT_F_TO_U,
    OP_CONVERT_U_TO_F, OP_DECORATE, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_EXT_INST,
    OP_EXT_INST_IMPORT, OP_F_ADD, OP_F_DIV, OP_F_MUL, OP_F_NEGATE, OP_F_ORD_GREATER_THAN,
    OP_F_ORD_GREATER_THAN_EQUAL, OP_F_ORD_LESS_THAN, OP_F_SUB, OP_FUNCTION, OP_FUNCTION_END,
    OP_I_ADD, OP_I_MUL, OP_I_NOT_EQUAL, OP_LABEL, OP_LOAD, OP_LOGICAL_AND, OP_LOGICAL_OR,
    OP_LOOP_MERGE, OP_MEMBER_DECORATE, OP_MEMORY_MODEL, OP_RETURN, OP_SELECT,
    OP_SELECTION_MERGE, OP_SHIFT_LEFT_LOGICAL, OP_STORE, OP_TYPE_BOOL, OP_TYPE_FLOAT,
    OP_TYPE_FUNCTION, OP_TYPE_INT, OP_TYPE_POINTER, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT,
    OP_TYPE_VECTOR, OP_TYPE_VOID, OP_U_GREATER_THAN_EQUAL, OP_U_LESS_THAN, OP_VARIABLE, SCHEMA,
    STORAGE_CLASS_FUNCTION, STORAGE_CLASS_INPUT, STORAGE_CLASS_UNIFORM, VERSION_1_0,
};
use crate::compute::{BindingAccess, BufferSpec, ElementType, KernelDesc, WorkgroupSize};
use crate::{CameraReq, LightReq, SceneRenderRequest, SphereReq, MAX_FRAMEBUFFER_BYTES};

// ---------------------------------------------------------------------------
// 模块标识（跨后端稳定契约）
// ---------------------------------------------------------------------------

/// 光追场景 kernel 的 SPIR-V 模块名（注册到 [`super::spirv::module`]）。
pub const MODULE_SCENE: &str = "spv_scene";
/// 光追场景 kernel 的入口名（`OpEntryPoint`）。
pub const ENTRY_SCENE: &str = "scene_main";
/// 工作组尺寸（x, y, z）：与背景 kernel 同为 8×8 的二维 tile。
pub const WORKGROUP_SCENE: [u32; 3] = [8, 8, 1];

// ---------------------------------------------------------------------------
// 参数块布局（binding 0）：22 个 u32
// ---------------------------------------------------------------------------
//
// 浮点一律以 **f32 位型**（`to_bits()`）传入，shader 用 `OpBitcast` 还原：
// 不存在第二套十进制/文本往返口径（同 `spirv_bg` 的 `t` 位型约定）。

/// 画幅宽（像素）。
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
/// `contrast != 1.0` 的预计算开关（0 / 1）：CPU 侧判定，shader 不做浮点不等比较
/// （避免「NaN 参与比较」在两端口径不一致）。
pub const P_CONTRAST_ON: u32 = 18;
/// `filmic` 开关（0 / 1）。
pub const P_FILMIC: u32 = 19;
/// 球体个数。
pub const P_NUM_SPHERES: u32 = 20;
/// 光源个数。
pub const P_NUM_LIGHTS: u32 = 21;

/// 参数块的总字数（u32 个数）。
pub const PARAMS_WORDS: usize = 22;
/// 每球占用的 f32 槽数：`{center.xyz, radius, albedo.xyz, pad}`。
pub const SPHERE_SLOTS: u32 = 8;
/// 每灯占用的 f32 槽数：`{dir.xyz, color.xyz, intensity, pad}`。
pub const LIGHT_SLOTS: u32 = 8;

/// 球体/光源数量的上限（防御性：避免调用方传入天文数字导致巨额缓冲分配）。
///
/// 现实场景远小于此；超限如实拒绝（`InvalidArguments`），调用方回退 CPU。
pub const MAX_SCENE_SPHERES: usize = 1 << 16;
/// 光源数量上限（同上）。
pub const MAX_SCENE_LIGHTS: usize = 1 << 12;

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

// GLSL.std.450 扩展指令号（与 `spirv_bg` 同一份公开编号）。
/// `Sqrt(x)`：正确舍入平方根。
const GLSL_SQRT: u32 = 31;
/// `FMax(x, y)`。
const GLSL_FMAX: u32 = 40;
/// `FClamp(x, min, max)`。
const GLSL_FCLAMP: u32 = 43;
/// `Floor(x)`：IEEE 精确取整。
const GLSL_FLOOR: u32 = 8;

/// `OpCapability Shader` 的 capability 号。
const CAPABILITY_SHADER: u32 = 1;
/// 逻辑寻址模型 `Logical`。
const ADDRESSING_LOGICAL: u32 = 0;
/// 内存模型 `GLSL450`。
const MEMORY_MODEL_GLSL450: u32 = 1;
/// 函数控制位 `None`。
const FUNCTION_CONTROL_NONE: u32 = 0;
/// 选择控制位 `None`。
const SELECTION_CONTROL_NONE: u32 = 0;
/// 循环控制位 `None`（无 Unroll / DontUnroll 提示：交由驱动按上界决定）。
const LOOP_CONTROL_NONE: u32 = 0;

// ---------------------------------------------------------------------------
// CPU 侧：相机基 + 三块输入缓冲的编码
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
    let right_v = cross(forward_v, prism_math::Vec3::new(0.0, 1.0, 0.0)).normalize();
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
fn cross(a: prism_math::Vec3, b: prism_math::Vec3) -> prism_math::Vec3 {
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

/// 编码参数块（[`PARAMS_WORDS`] × 4 字节，little-endian）。
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

/// 构造场景 kernel 的描述（**按帧尺寸与实体个数**：输出元素数 = 像素数）。
///
/// 返回 `None` 表示尺寸/实体数非法或超限（零轴、维度超
/// [`crate::MAX_FRAMEBUFFER_DIM`]、字节超 [`MAX_FRAMEBUFFER_BYTES`]、实体数超
/// 上限）——**不上机、不 panic**。空球/空灯集合被规范化为 1 个槽位（内容全 0），
/// 循环上界由参数块里的 0 决定，不会执行循环体。
pub fn scene_kernel(w: u32, h: u32, num_spheres: usize, num_lights: usize) -> Option<KernelDesc> {
    if w == 0 || h == 0 || w > crate::MAX_FRAMEBUFFER_DIM || h > crate::MAX_FRAMEBUFFER_DIM {
        return None;
    }
    if num_spheres > MAX_SCENE_SPHERES || num_lights > MAX_SCENE_LIGHTS {
        return None;
    }
    let pixels = (w as u64).checked_mul(h as u64)?;
    let out_bytes = pixels.checked_mul(crate::BYTES_PER_PIXEL as u64)?;
    let in_bytes = (num_spheres as u64)
        .checked_mul(SPHERE_SLOTS as u64 * 4)?
        .checked_add((num_lights as u64).checked_mul(LIGHT_SLOTS as u64 * 4)?)?;
    let total = out_bytes.checked_add(in_bytes)?.checked_add((PARAMS_WORDS * 4) as u64)?;
    if total > MAX_FRAMEBUFFER_BYTES as u64 {
        return None;
    }
    Some(KernelDesc::new(
        crate::vulkan::scene::KERNEL_SCENE_RENDER,
        ENTRY_SCENE,
        WorkgroupSize::new(
            WORKGROUP_SCENE[0],
            WORKGROUP_SCENE[1],
            WORKGROUP_SCENE[2],
        ),
        vec![
            // b0：参数块（PARAMS_WORDS × u32）
            BufferSpec::new(
                0,
                ElementType::U32,
                PARAMS_WORDS as u32,
                BindingAccess::ReadOnly,
            ),
            // b1：球体（每球 SPHERE_SLOTS 个 f32 槽；空集合退化为 1 个占位槽）
            BufferSpec::new(
                1,
                ElementType::F32,
                (num_spheres.max(1) as u32) * SPHERE_SLOTS,
                BindingAccess::ReadOnly,
            ),
            // b2：光源（同上）
            BufferSpec::new(
                2,
                ElementType::F32,
                (num_lights.max(1) as u32) * LIGHT_SLOTS,
                BindingAccess::ReadOnly,
            ),
            // b3：输出（每像素一个打包 RGBA8 u32）
            BufferSpec::new(3, ElementType::U32, pixels as u32, BindingAccess::WriteOnly),
        ],
        MODULE_SCENE,
        // WGSL 宿主**尚未建立**（V2.1 A3 评估为「条件可行、未实施」，见
        // `docs/v2.1-WebGPU可行性评估.md`）。本 kernel 现有的宿主是
        // SPIR-V（本模块发射器）与 MSL（`metal::msl::RT_SCENE_MSL`）两份；
        // 此处留空串如实表达「没有 WGSL 源」，不指向一个不存在的文件。
        "",
    ))
}

// ---------------------------------------------------------------------------
// 发射器
// ---------------------------------------------------------------------------

/// 生成（并缓存）场景 kernel 的 SPIR-V 字流。
///
/// `OnceLock` 保证**全程序唯一实例**——`vkCreateShaderModule` 拿到的指针地址
/// 在多次调用间必须一致（与 `spirv_bg::background_words` 同一教训）。
pub fn scene_words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(build_scene_module)
}

/// 发射一条指令：`operands` 为首字之外的全部字，字长由此推出。
fn push(dst: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    dst.push(inst((operands.len() + 1) as u32, opcode));
    dst.extend_from_slice(operands);
}

/// 把字符串按 SPIR-V 的 literal string 编码（UTF-8 + `\0` 终结 + 4 字节补齐）。
fn push_str(words: &mut Vec<u32>, s: &str) {
    let mut buf: Vec<u8> = s.as_bytes().to_vec();
    buf.push(0);
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
    for chunk in buf.chunks(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

/// 最小 SPIR-V 发射器（场景 kernel 专用：段序 + id 分配 + 常量去重）。
struct SceneEmitter {
    // --- 逻辑段（SPIR-V §2.4 的固定顺序）---
    caps: Vec<u32>,
    ext_imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    constants: Vec<u32>,
    globals: Vec<u32>,
    funcs: Vec<u32>,
    // --- id 分配 ---
    next_id: u32,
    // --- 类型 id ---
    t_void: u32,
    t_bool: u32,
    t_f32: u32,
    t_u32: u32,
    t_v3u32: u32,
    p_in_v3: u32,
    p_u32: u32,
    p_f32: u32,
    s_u32: u32,
    s_f32: u32,
    p_s_u32: u32,
    p_s_f32: u32,
    pf_u32: u32,
    pf_f32: u32,
    t_fn: u32,
    // --- 全局变量 id ---
    v_gid: u32,
    v_params: u32,
    v_spheres: u32,
    v_lights: u32,
    v_out: u32,
    // --- 其它 ---
    glsl: u32,
    fn_main: u32,
    // --- 函数局部变量 id（必须在首块内声明）---
    v_i: u32,
    v_closest: u32,
    v_hit: u32,
    v_j: u32,
    v_cr: u32,
    v_cg: u32,
    v_cb: u32,
    // --- 常量去重（bits → id）---
    f32_consts: Vec<(u32, u32)>,
    u32_consts: Vec<(u32, u32)>,
}

impl SceneEmitter {
    fn fresh(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn new() -> Self {
        let mut e = SceneEmitter {
            caps: Vec::new(),
            ext_imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            annotations: Vec::new(),
            types: Vec::new(),
            constants: Vec::new(),
            globals: Vec::new(),
            funcs: Vec::new(),
            next_id: 1,
            t_void: 0,
            t_bool: 0,
            t_f32: 0,
            t_u32: 0,
            t_v3u32: 0,
            p_in_v3: 0,
            p_u32: 0,
            p_f32: 0,
            s_u32: 0,
            s_f32: 0,
            p_s_u32: 0,
            p_s_f32: 0,
            pf_u32: 0,
            pf_f32: 0,
            t_fn: 0,
            v_gid: 0,
            v_params: 0,
            v_spheres: 0,
            v_lights: 0,
            v_out: 0,
            glsl: 0,
            fn_main: 0,
            v_i: 0,
            v_closest: 0,
            v_hit: 0,
            v_j: 0,
            v_cr: 0,
            v_cg: 0,
            v_cb: 0,
            f32_consts: Vec::new(),
            u32_consts: Vec::new(),
        };
        e.declare_prologue();
        e.declare_types();
        e.declare_globals();
        e
    }

    /// ① capability → ② 扩展/内存模型 → ③ 入口/执行模式 → ④ 变量注解。
    fn declare_prologue(&mut self) {
        push(&mut self.caps, OP_CAPABILITY, &[CAPABILITY_SHADER]);

        self.glsl = self.fresh();
        let mut ops = vec![self.glsl];
        push_str(&mut ops, "GLSL.std.450");
        push(&mut self.ext_imports, OP_EXT_INST_IMPORT, &ops);

        push(
            &mut self.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        self.fn_main = self.fresh();
        self.v_gid = self.fresh();
        self.v_params = self.fresh();
        self.v_spheres = self.fresh();
        self.v_lights = self.fresh();
        self.v_out = self.fresh();

        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, self.fn_main];
        push_str(&mut ops, ENTRY_SCENE);
        ops.push(self.v_gid);
        push(&mut self.entry_points, OP_ENTRY_POINT, &ops);

        push(
            &mut self.entry_points,
            OP_EXECUTION_MODE,
            &[
                self.fn_main,
                EXECUTION_MODE_LOCAL_SIZE,
                WORKGROUP_SCENE[0],
                WORKGROUP_SCENE[1],
                WORKGROUP_SCENE[2],
            ],
        );

        // ④ 注解：Input 变量必须 BuiltIn；四个存储缓冲各自绑定到 set 0。
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_gid, DECORATION_BUILT_IN, BUILTIN_GLOBAL_INVOCATION_ID],
        );
        for (var, binding) in [
            (self.v_params, 0u32),
            (self.v_spheres, 1),
            (self.v_lights, 2),
            (self.v_out, 3),
        ] {
            push(
                &mut self.annotations,
                OP_DECORATE,
                &[var, DECORATION_DESCRIPTOR_SET, 0],
            );
            push(
                &mut self.annotations,
                OP_DECORATE,
                &[var, DECORATION_BINDING, binding],
            );
        }
    }

    /// ⑤ 类型段（全部类型一次发完，避免「常量后置类型」的段序违规）。
    fn declare_types(&mut self) {
        self.t_void = self.fresh();
        push(&mut self.types, OP_TYPE_VOID, &[self.t_void]);

        self.t_bool = self.fresh();
        push(&mut self.types, OP_TYPE_BOOL, &[self.t_bool]);

        self.t_f32 = self.fresh();
        push(&mut self.types, OP_TYPE_FLOAT, &[self.t_f32, 32]);

        self.t_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_INT, &[self.t_u32, 32, 0]);

        self.t_v3u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_VECTOR,
            &[self.t_v3u32, self.t_u32, 3],
        );

        self.p_in_v3 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_in_v3, STORAGE_CLASS_INPUT, self.t_v3u32],
        );

        // Uniform 存储类的标量指针（AccessChain 的元素类型）。
        self.p_u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_u32, STORAGE_CLASS_UNIFORM, self.t_u32],
        );
        self.p_f32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_f32, STORAGE_CLASS_UNIFORM, self.t_f32],
        );

        // 运行期数组 + 单成员结构体（Vulkan SSBO 口径：Uniform + BufferBlock）。
        let rt_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_RUNTIME_ARRAY, &[rt_u32, self.t_u32]);
        let rt_f32 = self.fresh();
        push(&mut self.types, OP_TYPE_RUNTIME_ARRAY, &[rt_f32, self.t_f32]);

        self.s_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_STRUCT, &[self.s_u32, rt_u32]);
        self.s_f32 = self.fresh();
        push(&mut self.types, OP_TYPE_STRUCT, &[self.s_f32, rt_f32]);

        self.p_s_u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_s_u32, STORAGE_CLASS_UNIFORM, self.s_u32],
        );
        self.p_s_f32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_s_f32, STORAGE_CLASS_UNIFORM, self.s_f32],
        );

        // Function 存储类的标量指针（循环携带值用）。
        self.pf_u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.pf_u32, STORAGE_CLASS_FUNCTION, self.t_u32],
        );
        self.pf_f32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.pf_f32, STORAGE_CLASS_FUNCTION, self.t_f32],
        );

        self.t_fn = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_FUNCTION,
            &[self.t_fn, self.t_void],
        );

        // SSBO 装饰：BufferBlock + 成员偏移 + 数组步长。
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.s_u32, DECORATION_BUFFER_BLOCK],
        );
        push(
            &mut self.annotations,
            OP_MEMBER_DECORATE,
            &[self.s_u32, 0, DECORATION_OFFSET, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.s_f32, DECORATION_BUFFER_BLOCK],
        );
        push(
            &mut self.annotations,
            OP_MEMBER_DECORATE,
            &[self.s_f32, 0, DECORATION_OFFSET, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[rt_u32, DECORATION_ARRAY_STRIDE, 4],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[rt_f32, DECORATION_ARRAY_STRIDE, 4],
        );
    }

    /// ⑤ 全局变量段（必须在类型之后）。
    fn declare_globals(&mut self) {
        let (p_in, v_gid) = (self.p_in_v3, self.v_gid);
        push(
            &mut self.globals,
            OP_VARIABLE,
            &[p_in, v_gid, STORAGE_CLASS_INPUT],
        );
        for (ptr, var) in [
            (self.p_s_u32, self.v_params),
            (self.p_s_f32, self.v_spheres),
            (self.p_s_f32, self.v_lights),
            (self.p_s_u32, self.v_out),
        ] {
            push(
                &mut self.globals,
                OP_VARIABLE,
                &[ptr, var, STORAGE_CLASS_UNIFORM],
            );
        }
    }

    // ------------------------------------------------------------------
    // 常量（同 bits 去重）
    // ------------------------------------------------------------------

    fn f32_const(&mut self, v: f32) -> u32 {
        let bits = v.to_bits();
        if let Some(&(_, id)) = self.f32_consts.iter().find(|(b, _)| *b == bits) {
            return id;
        }
        let id = self.fresh();
        push(&mut self.constants, OP_CONSTANT, &[self.t_f32, id, bits]);
        self.f32_consts.push((bits, id));
        id
    }

    fn u32_const(&mut self, v: u32) -> u32 {
        if let Some(&(_, id)) = self.u32_consts.iter().find(|(b, _)| *b == v) {
            return id;
        }
        let id = self.fresh();
        push(&mut self.constants, OP_CONSTANT, &[self.t_u32, id, v]);
        self.u32_consts.push((v, id));
        id
    }

    // ------------------------------------------------------------------
    // 浮点指令（全部打 NoContraction：禁止驱动把 mul+add 熔成 FMA）
    // ------------------------------------------------------------------

    fn fbin(&mut self, opcode: u32, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, opcode, &[self.t_f32, id, a, b]);
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn fadd(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_ADD, a, b)
    }

    fn fsub(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_SUB, a, b)
    }

    fn fmul(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_MUL, a, b)
    }

    fn fdiv(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_DIV, a, b)
    }

    /// `OpFNegate`（与 Rust 的 `-x` 逐位一致）。
    fn fneg(&mut self, a: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_F_NEGATE, &[self.t_f32, id, a]);
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    /// `GLSL.std.450` 一元扩展指令。
    fn ext1(&mut self, inst_no: u32, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, inst_no, x],
        );
        id
    }

    /// `GLSL.std.450` 二元扩展指令。
    fn ext2(&mut self, inst_no: u32, x: u32, y: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, inst_no, x, y],
        );
        id
    }

    /// `GLSL.std.450` 三元扩展指令（`FClamp`）。
    fn ext3(&mut self, inst_no: u32, x: u32, y: u32, z: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, inst_no, x, y, z],
        );
        id
    }

    fn fmax(&mut self, x: u32, y: u32) -> u32 {
        self.ext2(GLSL_FMAX, x, y)
    }

    fn fclamp(&mut self, x: u32, lo: f32, hi: f32) -> u32 {
        let lo = self.f32_const(lo);
        let hi = self.f32_const(hi);
        self.ext3(GLSL_FCLAMP, x, lo, hi)
    }

    fn flt(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_F_ORD_LESS_THAN, &[self.t_bool, id, a, b]);
        id
    }

    fn fgt(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_F_ORD_GREATER_THAN,
            &[self.t_bool, id, a, b],
        );
        id
    }

    /// 有序浮点 ≥（NaN 为假；与 Rust 的 `a >= b` 同语义）。
    fn fge(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_F_ORD_GREATER_THAN_EQUAL,
            &[self.t_bool, id, a, b],
        );
        id
    }

    fn land(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOGICAL_AND, &[self.t_bool, id, a, b]);
        id
    }

    fn lor(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOGICAL_OR, &[self.t_bool, id, a, b]);
        id
    }

    /// `u != 0`（把 u32 标志位变成布尔谓词）。
    fn flag(&mut self, v: u32) -> u32 {
        let zero = self.u32_const(0);
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_I_NOT_EQUAL,
            &[self.t_bool, id, v, zero],
        );
        id
    }

    fn select_f32(&mut self, cond: u32, if_true: u32, if_false: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_SELECT,
            &[self.t_f32, id, cond, if_true, if_false],
        );
        id
    }

    fn select_u32(&mut self, cond: u32, if_true: u32, if_false: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_SELECT,
            &[self.t_u32, id, cond, if_true, if_false],
        );
        id
    }

    // ------------------------------------------------------------------
    // 整数指令 / 类型转换 / 内存访问
    // ------------------------------------------------------------------

    fn ubin(&mut self, opcode: u32, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, opcode, &[self.t_u32, id, a, b]);
        id
    }

    fn uadd(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_I_ADD, a, b)
    }

    fn umul(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_I_MUL, a, b)
    }

    fn ushl(&mut self, a: u32, shift: u32) -> u32 {
        let s = self.u32_const(shift);
        self.ubin(OP_SHIFT_LEFT_LOGICAL, a, s)
    }

    fn uor(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_BITWISE_OR, a, b)
    }

    fn uless(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_U_LESS_THAN, &[self.t_bool, id, a, b]);
        id
    }

    fn uge(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_U_GREATER_THAN_EQUAL,
            &[self.t_bool, id, a, b],
        );
        id
    }

    fn u2f(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_CONVERT_U_TO_F,
            &[self.t_f32, id, x],
        );
        id
    }

    fn f2u(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_CONVERT_F_TO_U,
            &[self.t_u32, id, x],
        );
        id
    }

    fn bitcast_f32(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_BITCAST, &[self.t_f32, id, x]);
        id
    }

    fn load_u32(&mut self, ptr: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_u32, id, ptr]);
        id
    }

    fn load_f32(&mut self, ptr: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_f32, id, ptr]);
        id
    }

    fn store(&mut self, ptr: u32, value: u32) {
        push(&mut self.funcs, OP_STORE, &[ptr, value]);
    }

    /// `var ptr`：函数局部变量（必须在首块、且在其它指令之前声明）。
    fn local_var(&mut self, ptr_ty: u32, id: u32) {
        push(
            &mut self.funcs,
            OP_VARIABLE,
            &[ptr_ty, id, STORAGE_CLASS_FUNCTION],
        );
    }

    /// `base[index]`（存储缓冲：先取结构体成员 0（运行期数组），再按下标取元素）。
    fn access(&mut self, elem_ptr_ty: u32, base: u32, index: u32) -> u32 {
        let c0 = self.u32_const(0);
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_ACCESS_CHAIN,
            &[elem_ptr_ty, id, base, c0, index],
        );
        id
    }

    /// 读参数块第 `slot` 个 u32。
    fn param_u32(&mut self, slot: u32) -> u32 {
        let idx = self.u32_const(slot);
        let p_u32 = self.p_u32;
        let v = self.v_params;
        let ptr = self.access(p_u32, v, idx);
        self.load_u32(ptr)
    }

    /// 读参数块第 `slot` 个 f32（位型 → bitcast 还原）。
    fn param_f32(&mut self, slot: u32) -> u32 {
        let bits = self.param_u32(slot);
        self.bitcast_f32(bits)
    }

    /// 读 `var[base + off]` 的 f32（球/灯缓冲的槽位访问）。
    fn slot_f32(&mut self, var: u32, base: u32, off: u32) -> u32 {
        let off_id = self.u32_const(off);
        let idx = self.uadd(base, off_id);
        let p_f32 = self.p_f32;
        let ptr = self.access(p_f32, var, idx);
        self.load_f32(ptr)
    }

    // ------------------------------------------------------------------
    // 语义件
    // ------------------------------------------------------------------

    /// `Vec3::dot` 同式：`x*x + y*y + z*z`（左结合）。
    fn dot3(&mut self, x1: u32, y1: u32, z1: u32, x2: u32, y2: u32, z2: u32) -> u32 {
        let xx = self.fmul(x1, x2);
        let yy = self.fmul(y1, y2);
        let s1 = self.fadd(xx, yy);
        let zz = self.fmul(z1, z2);
        self.fadd(s1, zz)
    }

    /// `Vec3::normalize` 同式：`len = sqrt(dot)`，`len < 1e-8 ⇒ 零向量`，否则逐分量除以 `len`。
    fn normalize3(&mut self, x: u32, y: u32, z: u32) -> (u32, u32, u32) {
        let dot = self.dot3(x, y, z, x, y, z);
        let len = self.ext1(GLSL_SQRT, dot);
        let eps = self.f32_const(NORMALIZE_EPS);
        let degenerate = self.flt(len, eps);
        let qx = self.fdiv(x, len);
        let qy = self.fdiv(y, len);
        let qz = self.fdiv(z, len);
        let zero = self.f32_const(0.0);
        (
            self.select_f32(degenerate, zero, qx),
            self.select_f32(degenerate, zero, qy),
            self.select_f32(degenerate, zero, qz),
        )
    }

    /// `a * 0.6 + shadow_comp + high_comp`（teal-orange，单通道）。
    fn teal_orange_channel(
        &mut self,
        c: u32,
        luma: u32,
        one_minus_luma: u32,
        tint_lo: f32,
        tint_hi: f32,
    ) -> u32 {
        let lo_tint = self.f32_const(tint_lo);
        let sh = self.fmul(lo_tint, c);
        let sh2 = self.fmul(sh, one_minus_luma);
        let half = self.f32_const(0.5);
        let sh3 = self.fmul(sh2, half);
        let hi_tint = self.f32_const(tint_hi);
        let hi = self.fmul(hi_tint, c);
        let hi2 = self.fmul(hi, luma);
        let hi3 = self.fmul(hi2, half);
        let sixty = self.f32_const(0.6);
        let base = self.fmul(c, sixty);
        let s1 = self.fadd(base, sh3);
        self.fadd(s1, hi3)
    }

    /// 量化一通道：`clamp01 * 255` 后 `floor(x + 0.5)`（= Rust 的 `.round()`，非负域）。
    fn quantize(&mut self, c: u32) -> u32 {
        let clamped = self.fclamp(c, 0.0, 1.0);
        let c255 = self.f32_const(255.0);
        let scaled = self.fmul(clamped, c255);
        let half = self.f32_const(0.5);
        let shifted = self.fadd(scaled, half);
        let floored = self.ext1(GLSL_FLOOR, shifted);
        self.f2u(floored)
    }

    /// ⑥ 函数体：一像素一 invocation。
    fn build_body(&mut self) {
        // --- 标签（先分配，保证引用处不借用 self）---
        let l_entry = self.fresh();
        let l_oob_merge = self.fresh();
        let l_body = self.fresh();
        let l_sph_hdr = self.fresh();
        let l_sph_body = self.fresh();
        let l_sph_cont = self.fresh();
        let l_sph_merge = self.fresh();
        let l_hit = self.fresh();
        let l_hit_merge = self.fresh();
        let l_lt_hdr = self.fresh();
        let l_lt_body = self.fresh();
        let l_lt_cont = self.fresh();
        let l_lt_merge = self.fresh();

        // --- 局部变量 id（首块内声明）---
        self.v_i = self.fresh();
        self.v_closest = self.fresh();
        self.v_hit = self.fresh();
        self.v_j = self.fresh();
        self.v_cr = self.fresh();
        self.v_cg = self.fresh();
        self.v_cb = self.fresh();

        let t_void = self.t_void;
        let t_fn = self.t_fn;
        let fn_main = self.fn_main;
        push(
            &mut self.funcs,
            OP_FUNCTION,
            &[t_void, fn_main, FUNCTION_CONTROL_NONE, t_fn],
        );
        push(&mut self.funcs, OP_LABEL, &[l_entry]);

        // 局部变量必须集中在函数首块、且先于其它指令。
        for (ptr_ty, id) in [
            (self.pf_u32, self.v_i),
            (self.pf_f32, self.v_closest),
            (self.pf_u32, self.v_hit),
            (self.pf_u32, self.v_j),
            (self.pf_f32, self.v_cr),
            (self.pf_f32, self.v_cg),
            (self.pf_f32, self.v_cb),
        ] {
            self.local_var(ptr_ty, id);
        }

        // --- 入口块：取 invocation id + 尺寸，做尾组守卫 ---
        let t_v3u32 = self.t_v3u32;
        let v_gid = self.v_gid;
        let gid = self.load_u32_vector(v_gid, t_v3u32);
        let t_u32 = self.t_u32;
        let gx = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[t_u32, gx, gid, 0]);
        let gy = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[t_u32, gy, gid, 1]);

        let w = self.param_u32(P_WIDTH);
        let h = self.param_u32(P_HEIGHT);
        let cond_x = self.uge(gx, w);
        let cond_y = self.uge(gy, h);
        let oob = self.lor(cond_x, cond_y);
        push(
            &mut self.funcs,
            OP_SELECTION_MERGE,
            &[l_oob_merge, SELECTION_CONTROL_NONE],
        );
        push(
            &mut self.funcs,
            OP_BRANCH_CONDITIONAL,
            &[oob, l_oob_merge, l_body],
        );

        // --- 主体：主射线 ---
        push(&mut self.funcs, OP_LABEL, &[l_body]);
        let (ex, ey, ez) = (
            self.param_f32(P_EYE),
            self.param_f32(P_EYE + 1),
            self.param_f32(P_EYE + 2),
        );
        let (fx, fy, fz) = (
            self.param_f32(P_FORWARD),
            self.param_f32(P_FORWARD + 1),
            self.param_f32(P_FORWARD + 2),
        );
        let (rx, ry, rz) = (
            self.param_f32(P_RIGHT),
            self.param_f32(P_RIGHT + 1),
            self.param_f32(P_RIGHT + 2),
        );
        let (ux, uy, uz) = (
            self.param_f32(P_UP),
            self.param_f32(P_UP + 1),
            self.param_f32(P_UP + 2),
        );
        let tan_half = self.param_f32(P_TAN_HALF_FOV);
        let aspect = self.param_f32(P_ASPECT);

        let one = self.f32_const(1.0);
        let two = self.f32_const(2.0);
        let half = self.f32_const(0.5);

        // nx = ((2*(gx+0.5)/w) - 1) * tan_half * aspect
        let gxf = self.u2f(gx);
        let gxf_half = self.fadd(gxf, half);
        let gxf_2 = self.fmul(two, gxf_half);
        let wf = self.u2f(w);
        let nx_div = self.fdiv(gxf_2, wf);
        let nx_m1 = self.fsub(nx_div, one);
        let nx_t = self.fmul(nx_m1, tan_half);
        let nx = self.fmul(nx_t, aspect);

        // ny = (1 - 2*(gy+0.5)/h) * tan_half
        let gyf = self.u2f(gy);
        let gyf_half = self.fadd(gyf, half);
        let gyf_2 = self.fmul(two, gyf_half);
        let hf = self.u2f(h);
        let ny_div = self.fdiv(gyf_2, hf);
        let ny_sub = self.fsub(one, ny_div);
        let ny = self.fmul(ny_sub, tan_half);

        // dir = ((forward + right*nx) + up*ny).normalize()
        let rnx = self.fmul(rx, nx);
        let rny = self.fmul(ry, nx);
        let rnz = self.fmul(rz, nx);
        let dx0 = self.fadd(fx, rnx);
        let dy0 = self.fadd(fy, rny);
        let dz0 = self.fadd(fz, rnz);
        let unx = self.fmul(ux, ny);
        let uny = self.fmul(uy, ny);
        let unz = self.fmul(uz, ny);
        let dx1 = self.fadd(dx0, unx);
        let dy1 = self.fadd(dy0, uny);
        let dz1 = self.fadd(dz0, unz);
        let (dx, dy, dz) = self.normalize3(dx1, dy1, dz1);

        // --- 球体循环初始化 ---
        let inf = self.f32_const(f32::INFINITY);
        let v_closest = self.v_closest;
        self.store(v_closest, inf);
        let num_spheres = self.param_u32(P_NUM_SPHERES);
        let v_hit = self.v_hit;
        self.store(v_hit, num_spheres);
        let zero_u = self.u32_const(0);
        let v_i = self.v_i;
        self.store(v_i, zero_u);

        push(&mut self.funcs, OP_BRANCH, &[l_sph_hdr]);

        // --- 球体循环头 ---
        push(&mut self.funcs, OP_LABEL, &[l_sph_hdr]);
        let i_cur = self.load_u32(v_i);
        let i_in_range = self.uless(i_cur, num_spheres);
        push(
            &mut self.funcs,
            OP_LOOP_MERGE,
            &[l_sph_merge, l_sph_cont, LOOP_CONTROL_NONE],
        );
        push(
            &mut self.funcs,
            OP_BRANCH_CONDITIONAL,
            &[i_in_range, l_sph_body, l_sph_merge],
        );

        // --- 球体循环体：最近命中 ---
        push(&mut self.funcs, OP_LABEL, &[l_sph_body]);
        let v_spheres = self.v_spheres;
        let slots = self.u32_const(SPHERE_SLOTS);
        let base = self.umul(i_cur, slots);
        let cx = self.slot_f32(v_spheres, base, 0);
        let cy = self.slot_f32(v_spheres, base, 1);
        let cz = self.slot_f32(v_spheres, base, 2);
        let radius = self.slot_f32(v_spheres, base, 3);
        // oc = eye - center
        let ocx = self.fsub(ex, cx);
        let ocy = self.fsub(ey, cy);
        let ocz = self.fsub(ez, cz);
        // b = dot(oc, dir); c = dot(oc, oc) - r*r; disc = b*b - c
        let b = self.dot3(ocx, ocy, ocz, dx, dy, dz);
        let oc_sq = self.dot3(ocx, ocy, ocz, ocx, ocy, ocz);
        let r2 = self.fmul(radius, radius);
        let cc = self.fsub(oc_sq, r2);
        let b2 = self.fmul(b, b);
        let disc = self.fsub(b2, cc);
        // 先钳到非负再开方：disc < 0 时结果必被 select 丢弃，但开方参数保持
        // 在定义域内（不让 NaN 进入数据通路）。
        let zero_f = self.f32_const(0.0);
        let disc_pos = self.fmax(disc, zero_f);
        let sq = self.ext1(GLSL_SQRT, disc_pos);
        let neg_b = self.fneg(b);
        let t = self.fsub(neg_b, sq);
        // if disc >= 0 && t > 0 && t < closest { closest = t; hit = i }
        let c_disc = self.fge(disc, zero_f);
        let c_tpos = self.fgt(t, zero_f);
        let closest_cur = self.load_f32(v_closest);
        let c_near = self.flt(t, closest_cur);
        let c_dt = self.land(c_disc, c_tpos);
        let hit_ok = self.land(c_dt, c_near);
        let new_closest = self.select_f32(hit_ok, t, closest_cur);
        self.store(v_closest, new_closest);
        let hit_cur = self.load_u32(v_hit);
        let new_hit = self.select_u32(hit_ok, i_cur, hit_cur);
        self.store(v_hit, new_hit);
        push(&mut self.funcs, OP_BRANCH, &[l_sph_cont]);

        // --- 球体循环 continue ---
        push(&mut self.funcs, OP_LABEL, &[l_sph_cont]);
        let one_u = self.u32_const(1);
        let i_next = self.uadd(i_cur, one_u);
        self.store(v_i, i_next);
        push(&mut self.funcs, OP_BRANCH, &[l_sph_hdr]);

        // --- 球体循环出口：命中与否 ---
        push(&mut self.funcs, OP_LABEL, &[l_sph_merge]);
        let hit_idx = self.load_u32(v_hit);
        let has_hit = self.uless(hit_idx, num_spheres);
        // 背景色先落到颜色累加器（未命中时原样输出）。
        let bg_r = self.f32_const(BG_COLOR[0]);
        let bg_g = self.f32_const(BG_COLOR[1]);
        let bg_b = self.f32_const(BG_COLOR[2]);
        let (v_cr, v_cg, v_cb) = (self.v_cr, self.v_cg, self.v_cb);
        self.store(v_cr, bg_r);
        self.store(v_cg, bg_g);
        self.store(v_cb, bg_b);
        push(
            &mut self.funcs,
            OP_SELECTION_MERGE,
            &[l_hit_merge, SELECTION_CONTROL_NONE],
        );
        push(
            &mut self.funcs,
            OP_BRANCH_CONDITIONAL,
            &[has_hit, l_hit, l_hit_merge],
        );

        // --- 命中块：法线 + 直接光累加 ---
        push(&mut self.funcs, OP_LABEL, &[l_hit]);
        let closest_t = self.load_f32(v_closest);
        let ptx = self.fmul(dx, closest_t);
        let pty = self.fmul(dy, closest_t);
        let ptz = self.fmul(dz, closest_t);
        let px = self.fadd(ex, ptx);
        let py = self.fadd(ey, pty);
        let pz = self.fadd(ez, ptz);
        let hbase = self.umul(hit_idx, slots);
        let hcx = self.slot_f32(v_spheres, hbase, 0);
        let hcy = self.slot_f32(v_spheres, hbase, 1);
        let hcz = self.slot_f32(v_spheres, hbase, 2);
        let alb_r = self.slot_f32(v_spheres, hbase, 4);
        let alb_g = self.slot_f32(v_spheres, hbase, 5);
        let alb_b = self.slot_f32(v_spheres, hbase, 6);
        let nx_v = self.fsub(px, hcx);
        let ny_v = self.fsub(py, hcy);
        let nz_v = self.fsub(pz, hcz);
        let (n_x, n_y, n_z) = self.normalize3(nx_v, ny_v, nz_v);

        let zero_f2 = self.f32_const(0.0);
        self.store(v_cr, zero_f2);
        self.store(v_cg, zero_f2);
        self.store(v_cb, zero_f2);
        let num_lights = self.param_u32(P_NUM_LIGHTS);
        let v_j = self.v_j;
        let zero_u2 = self.u32_const(0);
        self.store(v_j, zero_u2);
        push(&mut self.funcs, OP_BRANCH, &[l_lt_hdr]);

        // --- 光源循环头 ---
        push(&mut self.funcs, OP_LABEL, &[l_lt_hdr]);
        let j_cur = self.load_u32(v_j);
        let j_in_range = self.uless(j_cur, num_lights);
        push(
            &mut self.funcs,
            OP_LOOP_MERGE,
            &[l_lt_merge, l_lt_cont, LOOP_CONTROL_NONE],
        );
        push(
            &mut self.funcs,
            OP_BRANCH_CONDITIONAL,
            &[j_in_range, l_lt_body, l_lt_merge],
        );

        // --- 光源循环体 ---
        push(&mut self.funcs, OP_LABEL, &[l_lt_body]);
        let v_lights = self.v_lights;
        let lslots = self.u32_const(LIGHT_SLOTS);
        let lbase = self.umul(j_cur, lslots);
        let ldx = self.slot_f32(v_lights, lbase, 0);
        let ldy = self.slot_f32(v_lights, lbase, 1);
        let ldz = self.slot_f32(v_lights, lbase, 2);
        let lc_r = self.slot_f32(v_lights, lbase, 3);
        let lc_g = self.slot_f32(v_lights, lbase, 4);
        let lc_b = self.slot_f32(v_lights, lbase, 5);
        let lint = self.slot_f32(v_lights, lbase, 6);
        let ndotl = self.dot3(n_x, n_y, n_z, ldx, ldy, ldz);
        let zero_f3 = self.f32_const(0.0);
        let diff = self.fmax(ndotl, zero_f3);
        let scaled = self.fmul(diff, lint);
        let amb = self.f32_const(AMBIENT);
        // 每通道：acc += albedo * (ambient + lcolor * (diff * intensity))
        for (acc_var, alb, lc) in [(v_cr, alb_r, lc_r), (v_cg, alb_g, lc_g), (v_cb, alb_b, lc_b)] {
            let tint = self.fmul(lc, scaled);
            let term = self.fadd(amb, tint);
            let prod = self.fmul(alb, term);
            let acc_cur = self.load_f32(acc_var);
            let acc_next = self.fadd(acc_cur, prod);
            self.store(acc_var, acc_next);
        }
        push(&mut self.funcs, OP_BRANCH, &[l_lt_cont]);

        // --- 光源循环 continue / 出口 ---
        push(&mut self.funcs, OP_LABEL, &[l_lt_cont]);
        let j_next = self.uadd(j_cur, one_u);
        self.store(v_j, j_next);
        push(&mut self.funcs, OP_BRANCH, &[l_lt_hdr]);

        push(&mut self.funcs, OP_LABEL, &[l_lt_merge]);
        push(&mut self.funcs, OP_BRANCH, &[l_hit_merge]);

        // --- 命中/未命中汇合：aesthetic → 量化 → 打包 → 写回 ---
        push(&mut self.funcs, OP_LABEL, &[l_hit_merge]);
        let mut cr = self.load_f32(v_cr);
        let mut cg = self.load_f32(v_cg);
        let mut cb = self.load_f32(v_cb);

        // if teal_orange { ... }
        let ta_flag = self.param_u32(P_TEAL_ORANGE);
        let teal = self.flag(ta_flag);
        // luma = 0.299*r + 0.587*g + 0.114*b（左结合，与 CPU 同序）
        let k_r = self.f32_const(0.299);
        let lr = self.fmul(k_r, cr);
        let k_g = self.f32_const(0.587);
        let lg = self.fmul(k_g, cg);
        let lrg = self.fadd(lr, lg);
        let k_b = self.f32_const(0.114);
        let lb = self.fmul(k_b, cb);
        let luma = self.fadd(lrg, lb);
        let oml = self.fsub(one, luma);
        let tr = self.teal_orange_channel(cr, luma, oml, SHADOW_TINT[0], HIGH_TINT[0]);
        let tg = self.teal_orange_channel(cg, luma, oml, SHADOW_TINT[1], HIGH_TINT[1]);
        let tb = self.teal_orange_channel(cb, luma, oml, SHADOW_TINT[2], HIGH_TINT[2]);
        cr = self.select_f32(teal, tr, cr);
        cg = self.select_f32(teal, tg, cg);
        cb = self.select_f32(teal, tb, cb);

        // if contrast != 1.0 { c = (c - 0.5) * contrast + 0.5 }
        let contrast_on = self.param_u32(P_CONTRAST_ON);
        let contrast = self.param_f32(P_CONTRAST);
        let con = self.flag(contrast_on);
        let half_c = self.f32_const(0.5);
        let mut nc = [0u32; 3];
        for (slot, c) in [cr, cg, cb].into_iter().enumerate() {
            let d = self.fsub(c, half_c);
            let m = self.fmul(d, contrast);
            let a = self.fadd(m, half_c);
            nc[slot] = self.select_f32(con, a, c);
        }
        cr = nc[0];
        cg = nc[1];
        cb = nc[2];

        // if filmic { c = c / (c + 1) }
        let filmic_flag = self.param_u32(P_FILMIC);
        let filmic = self.flag(filmic_flag);
        let one_f = self.f32_const(1.0);
        let mut nf = [0u32; 3];
        for (slot, c) in [cr, cg, cb].into_iter().enumerate() {
            let den = self.fadd(c, one_f);
            let q = self.fdiv(c, den);
            nf[slot] = self.select_f32(filmic, q, c);
        }

        // clamp01 → *255 → floor(x+0.5) → u32（= CPU 的 `(v.clamp*255).round() as u8`）
        let q_r = self.quantize(nf[0]);
        let q_g = self.quantize(nf[1]);
        let q_b = self.quantize(nf[2]);
        let alpha = self.u32_const(0xFF00_0000);
        let g_shift = self.ushl(q_g, 8);
        let b_shift = self.ushl(q_b, 16);
        let pack0 = self.uor(alpha, b_shift);
        let pack1 = self.uor(pack0, g_shift);
        let packed = self.uor(pack1, q_r);

        // out[gy * w + gx] = packed
        let row = self.umul(gy, w);
        let idx = self.uadd(row, gx);
        let p_u32 = self.p_u32;
        let v_out = self.v_out;
        let out_ptr = self.access(p_u32, v_out, idx);
        self.store(out_ptr, packed);

        push(&mut self.funcs, OP_BRANCH, &[l_oob_merge]);
        push(&mut self.funcs, OP_LABEL, &[l_oob_merge]);
        push(&mut self.funcs, OP_RETURN, &[]);
        push(&mut self.funcs, OP_FUNCTION_END, &[]);
    }

    /// `OpLoad %v3u32 %v_gid`（Input 内建变量）。
    fn load_u32_vector(&mut self, var: u32, vec_ty: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[vec_ty, id, var]);
        id
    }

    /// 拼装完整模块：5 字头 + 九个逻辑段。
    fn assemble(self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        out.push(MAGIC);
        out.push(VERSION_1_0);
        out.push(0); // generator = 0（自研手写，非工具生成 —— 如实标注）
        out.push(self.next_id); // bound = max id + 1
        out.push(SCHEMA);
        for seg in [
            self.caps,
            self.ext_imports,
            self.memory_model,
            self.entry_points,
            self.annotations,
            self.types,
            self.constants,
            self.globals,
            self.funcs,
        ] {
            out.extend_from_slice(&seg);
        }
        out
    }
}

/// 构建场景 kernel 的完整 SPIR-V 模块（确定性：两次构建逐字相同）。
pub fn build_scene_module() -> Vec<u32> {
    let mut e = SceneEmitter::new();
    e.build_body();
    e.assemble()
}

// ---------------------------------------------------------------------------
// 测试（无设备即可执行：结构校验 / 确定性 / 编码口径 / 尺寸守卫）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::spirv::{disassemble, validate, SpvExpect};
    use crate::AestheticReq;
    use prism_math::Vec3;

    fn scene_expect() -> SpvExpect {
        SpvExpect {
            entry: ENTRY_SCENE,
            local_size: WORKGROUP_SCENE,
            bindings: &[0, 1, 2, 3],
        }
    }

    fn request(w: u32, h: u32) -> SceneRenderRequest {
        SceneRenderRequest {
            width: w,
            height: h,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: w as f32 / h as f32,
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
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        }
    }

    /// 结构验证：入口 / LocalSize / 绑定 / 单函数 / 动态循环 / 指令铺满字流。
    #[test]
    fn scene_module_is_structurally_valid() {
        let words = scene_words();
        assert!(words.len() > 400, "场景 kernel 不应是玩具模块");
        let summary = validate(words, &scene_expect()).expect("发射器产出的模块须通过结构验证");
        assert_eq!(summary.entry, "scene_main");
        assert_eq!(summary.local_size, [8, 8, 1]);
        assert_eq!(summary.bindings, vec![0, 1, 2, 3]);
        assert_eq!(summary.functions, 1, "模块应恰有一个函数");

        // 指令字长之和必须铺满整个字流。
        let mut i = 5usize;
        let mut total = 0usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            assert!(wc > 0, "零字长指令 @{i}");
            total += wc;
            i += wc;
        }
        assert_eq!(5 + total, words.len(), "指令字长之和须铺满字流");

        // 关键件必须在场：两层动态循环 + 两层选择 + 求交/归一化/量化件。
        for op in [
            OP_LOOP_MERGE,
            OP_SELECTION_MERGE,
            OP_BRANCH_CONDITIONAL,
            OP_U_LESS_THAN,
            OP_F_ORD_GREATER_THAN_EQUAL,
            OP_LOGICAL_AND,
            OP_I_NOT_EQUAL,
            OP_F_NEGATE,
            OP_EXT_INST,
            OP_CONVERT_F_TO_U,
            OP_BITCAST,
            OP_VARIABLE,
            OP_FUNCTION_END,
        ] {
            assert!(
                summary.opcodes.contains(&op),
                "缺指令 {}",
                crate::vulkan::spirv::opcode_name(op)
            );
        }
        // 循环数：球 + 灯 = 2 个 OpLoopMerge（按指令首字精确计数）。
        let mut loop_count = 0usize;
        let mut i = 5usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if words[i] & 0xFFFF == OP_LOOP_MERGE {
                loop_count += 1;
            }
            i += wc;
        }
        assert_eq!(loop_count, 2, "应有球体与光源两层动态循环");

        // 反汇编不得出现未知 OpCode（白名单与发射器同步）。
        let text = disassemble(words);
        assert!(!text.contains("OpUnknown"), "白名单漏了 OpCode：\n{text}");
        println!(
            "场景 SPIR-V：{} 字 / {} 条指令 / bound={}",
            words.len(),
            summary.instruction_count,
            summary.bound
        );
    }

    /// 发射确定性：两次构建逐字相同（`OnceLock` 缓存与构建结果一致）。
    #[test]
    fn build_is_deterministic() {
        let a = build_scene_module();
        let b = build_scene_module();
        assert_eq!(a, b, "同输入两次构建必须逐字相同");
        assert_eq!(scene_words(), a.as_slice(), "缓存实例须与构建结果一致");
        // 注册表可见（`spirv::module` 是唯一取用口）。
        assert_eq!(
            crate::vulkan::spirv::module(MODULE_SCENE).map(|w| w.len()),
            Some(a.len())
        );
    }

    /// **OpCode 编号钉死**（本仓曾因编号错位静默算错图：`OpFMul` 写成 128）。
    ///
    /// 期望值来自 SPIR-V 1.0 官方工具链实测（`spirv-as` + `spirv-dis`），
    /// 不是「照着实现抄一遍」——若有人把常量改回错值，本断言立刻红。
    #[test]
    fn opcode_numbers_match_spec() {
        use crate::vulkan::spirv::*;
        assert_eq!(OP_F_NEGATE, 127);
        assert_eq!(OP_I_ADD, 128);
        assert_eq!(OP_F_ADD, 129);
        assert_eq!(OP_F_SUB, 131);
        assert_eq!(OP_F_MUL, 133);
        assert_eq!(OP_F_DIV, 136);
        assert_eq!(OP_LOGICAL_OR, 166);
        assert_eq!(OP_LOGICAL_AND, 167);
        assert_eq!(OP_LOGICAL_NOT, 168);
        assert_eq!(OP_SELECT, 169);
        assert_eq!(OP_I_NOT_EQUAL, 171);
        assert_eq!(OP_U_GREATER_THAN_EQUAL, 174);
        assert_eq!(OP_U_LESS_THAN, 176);
        assert_eq!(OP_F_ORD_LESS_THAN, 184);
        assert_eq!(OP_F_ORD_GREATER_THAN, 186);
        assert_eq!(OP_F_ORD_GREATER_THAN_EQUAL, 190);
        assert_eq!(OP_LOOP_MERGE, 246);
        assert_eq!(OP_SELECTION_MERGE, 247);
        assert_eq!(OP_LABEL, 248);
        assert_eq!(OP_BRANCH, 249);
        assert_eq!(OP_BRANCH_CONDITIONAL, 250);
        // 三个曾被写错的编号不得回归（勘误留痕）。
        assert_ne!(OP_U_LESS_THAN, 170, "170 是 OpIEqual，不是 OpULessThan");
        assert_ne!(OP_LOGICAL_NOT, 167, "167 是 OpLogicalAnd，不是 OpLogicalNot");
    }

    /// 结构化控制流校验抓得住真实缺陷（负向用例，非恒真）。
    #[test]
    fn control_flow_validator_catches_broken_blocks() {
        let good = scene_words();
        let expect = scene_expect();
        validate(good, &expect).expect("原始模块应合法");

        // ① 把某条 OpBranch 的目标改成不存在的 id ⇒ 分支目标非法。
        let mut bad = good.to_vec();
        let mut i = 5usize;
        let mut patched = false;
        while i < good.len() {
            let wc = (good[i] >> 16) as usize;
            if good[i] & 0xFFFF == OP_BRANCH {
                bad[i + 1] = 0xFFFF_FF00; // 不存在的 label
                patched = true;
                break;
            }
            i += wc;
        }
        assert!(patched, "模块中应存在 OpBranch");
        assert!(matches!(
            validate(&bad, &expect),
            Err(crate::vulkan::spirv::SpvError::MalformedControlFlow { .. })
        ));

        // ② 把某个 OpLoopMerge 的 continue 目标改成不存在的 label ⇒ 目标非法。
        let mut bad2 = good.to_vec();
        let mut i = 5usize;
        let mut patched2 = false;
        while i < good.len() {
            let wc = (good[i] >> 16) as usize;
            if good[i] & 0xFFFF == OP_LOOP_MERGE {
                bad2[i + 2] = 0xFFFF_FF01;
                patched2 = true;
                break;
            }
            i += wc;
        }
        assert!(patched2, "模块中应存在 OpLoopMerge");
        assert!(matches!(
            validate(&bad2, &expect),
            Err(crate::vulkan::spirv::SpvError::MalformedControlFlow { .. })
        ));

        // ③ 把某条 OpBranch 改写成 OpLabel（同为 2 字）⇒ 上一块失去终结符。
        let mut bad3 = good.to_vec();
        let mut i = 5usize;
        let mut patched3 = false;
        while i < good.len() {
            let wc = (good[i] >> 16) as usize;
            if good[i] & 0xFFFF == OP_BRANCH && wc == 2 {
                bad3[i] = crate::vulkan::spirv::inst(2, OP_LABEL);
                bad3[i + 1] = 0xABCD;
                patched3 = true;
                break;
            }
            i += wc;
        }
        assert!(patched3, "模块中应存在 2 字的 OpBranch");
        assert!(matches!(
            validate(&bad3, &expect),
            Err(crate::vulkan::spirv::SpvError::MalformedControlFlow { .. })
        ));
    }

    /// 参数块编码：字段位型、标志位、实体计数逐槽对齐。
    #[test]
    fn params_encode_layout_is_stable() {
        let req = request(64, 48);
        let basis = camera_basis(&req.camera);
        let p = encode_params(&req, &basis);
        let word = |slot: u32| -> u32 {
            let o = slot as usize * 4;
            u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]])
        };
        assert_eq!(p.len(), PARAMS_WORDS * 4);
        assert_eq!(word(P_WIDTH), 64);
        assert_eq!(word(P_HEIGHT), 48);
        assert_eq!(word(P_EYE), 0.0f32.to_bits());
        assert_eq!(word(P_EYE + 2), (-4.0f32).to_bits());
        // forward 指向 +Z；right = cross(forward, world_up) = (-1, 0, 0)（与
        // `CpuRefBackend` 同一公式、同一符号约定：镜像由相机基决定，不在渲染侧翻转）；
        // up = cross(right, forward) = (0, 1, 0)。
        assert_eq!(word(P_FORWARD + 2), 1.0f32.to_bits());
        assert_eq!(word(P_RIGHT), (-1.0f32).to_bits());
        assert_eq!(word(P_UP + 1), 1.0f32.to_bits());
        assert_eq!(
            word(P_TAN_HALF_FOV),
            prism_math::det_trig::tan(req.camera.fov * 0.5).to_bits(),
            "tan_half 必须走 det_trig（D-29），不得用平台 tan"
        );
        assert_eq!(word(P_ASPECT), (64.0f32 / 48.0).to_bits());
        assert_eq!(word(P_TEAL_ORANGE), 0);
        assert_eq!(word(P_CONTRAST), 1.0f32.to_bits());
        assert_eq!(word(P_CONTRAST_ON), 0, "contrast == 1.0 ⇒ 开关关闭");
        assert_eq!(word(P_FILMIC), 0);
        assert_eq!(word(P_NUM_SPHERES), 1);
        assert_eq!(word(P_NUM_LIGHTS), 1);

        // 开关口径：contrast != 1.0（含 NaN）⇒ 1。
        let mut r2 = request(8, 8);
        r2.aesthetic.contrast = 1.5;
        r2.aesthetic.teal_orange = true;
        r2.aesthetic.filmic = true;
        let p2 = encode_params(&r2, &camera_basis(&r2.camera));
        let w2 = |slot: u32| -> u32 {
            let o = slot as usize * 4;
            u32::from_le_bytes([p2[o], p2[o + 1], p2[o + 2], p2[o + 3]])
        };
        assert_eq!(w2(P_TEAL_ORANGE), 1);
        assert_eq!(w2(P_CONTRAST_ON), 1);
        assert_eq!(w2(P_FILMIC), 1);
        let mut r3 = request(8, 8);
        r3.aesthetic.contrast = f32::NAN;
        let p3 = encode_params(&r3, &camera_basis(&r3.camera));
        let o = P_CONTRAST_ON as usize * 4;
        assert_eq!(u32::from_le_bytes([p3[o], p3[o + 1], p3[o + 2], p3[o + 3]]), 1);
    }

    /// 球/灯编码：槽位布局（centre/radius/albedo/pad、dir/color/intensity/pad）。
    #[test]
    fn sphere_and_light_encoding_is_slot_exact() {
        let req = request(4, 4);
        let s = encode_spheres(&req.spheres);
        assert_eq!(s.len(), SPHERE_SLOTS as usize * 4);
        let f32_at = |b: &[u8], i: usize| -> f32 {
            f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
        };
        assert_eq!(f32_at(&s, 0), 0.0); // center.x
        assert_eq!(f32_at(&s, 3), 1.0); // radius
        assert_eq!(f32_at(&s, 4), 0.8); // albedo.r
        assert_eq!(f32_at(&s, 7), 0.0); // pad

        let l = encode_lights(&req.lights);
        assert_eq!(l.len(), LIGHT_SLOTS as usize * 4);
        // 方向已归一化：(0,0,-1) 本身就是单位向量。
        assert_eq!(f32_at(&l, 2), -1.0);
        assert_eq!(f32_at(&l, 6), 1.0); // intensity
        assert_eq!(f32_at(&l, 7), 0.0); // pad

        // 非单位方向 ⇒ 编码时归一化（与 CPU 的 normalize 同值）。
        let l2 = vec![LightReq {
            direction: Vec3::new(0.0, 0.0, -3.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 2.0,
        }];
        let e2 = encode_lights(&l2);
        assert_eq!(f32_at(&e2, 2), -1.0, "方向须已归一化");
        assert_eq!(f32_at(&e2, 6), 2.0);

        // 空集合 ⇒ 空缓冲（kernel 描述另用 1 个占位槽，循环上界为 0）。
        assert!(encode_spheres(&[]).is_empty());
        assert!(encode_lights(&[]).is_empty());
    }

    /// kernel 描述：绑定/布局/工作组/SPIR-V 状态自洽；尺寸与实体数守卫。
    #[test]
    fn scene_kernel_desc_and_guards() {
        let k = scene_kernel(64, 48, 3, 2).expect("合法尺寸");
        assert_eq!(k.entry, ENTRY_SCENE);
        assert_eq!(k.spv_module, MODULE_SCENE);
        assert_eq!((k.workgroup.x, k.workgroup.y, k.workgroup.z), (8, 8, 1));
        k.validate().expect("kernel 描述须通过布局校验");
        let layout = k.layout();
        assert_eq!(layout.len(), 4);
        assert_eq!(layout[0].bytes, PARAMS_WORDS as u32 * 4);
        assert_eq!(layout[1].bytes, 3 * SPHERE_SLOTS * 4);
        assert_eq!(layout[2].bytes, 2 * LIGHT_SLOTS * 4);
        assert_eq!(layout[3].bytes, 64 * 48 * 4);
        assert_eq!(
            crate::compute::kernel_spv_status(&k),
            crate::compute::SpvStatus::Embedded,
            "场景 kernel 的 SPIR-V 必须是发射器产物（已嵌入）"
        );
        // 空实体集合：占位槽为 1，避免零元素绑定（validate 会拒）。
        let empty = scene_kernel(4, 4, 0, 0).expect("空场景仍应可派发（背景色）");
        assert!(empty.validate().is_ok());
        assert_eq!(empty.layout()[1].bytes, SPHERE_SLOTS * 4);
        assert_eq!(empty.layout()[2].bytes, LIGHT_SLOTS * 4);

        // 守卫：零轴 / 维度超限 / 字节超限 / 实体数超限 ⇒ None（不 panic）。
        assert!(scene_kernel(0, 8, 1, 1).is_none());
        assert!(scene_kernel(8, 0, 1, 1).is_none());
        assert!(scene_kernel(crate::MAX_FRAMEBUFFER_DIM + 1, 8, 1, 1).is_none());
        assert!(scene_kernel(65_536, 65_536, 1, 1).is_none());
        assert!(scene_kernel(u32::MAX, u32::MAX, 1, 1).is_none());
        assert!(scene_kernel(8, 8, MAX_SCENE_SPHERES + 1, 1).is_none());
        assert!(scene_kernel(8, 8, 1, MAX_SCENE_LIGHTS + 1).is_none());
        assert!(scene_kernel(1920, 1080, 4, 2).is_some());
    }

    /// 相机基与 CPU 参考同式（用 `CpuRefBackend` 渲染一帧做端到端锚定在集成
    /// 测试里；此处只钉「基向量正交归一 + tan 走 det_trig」）。
    #[test]
    fn camera_basis_is_orthonormal() {
        let req = request(32, 18);
        let b = camera_basis(&req.camera);
        let f = Vec3::new(b.forward[0], b.forward[1], b.forward[2]);
        let r = Vec3::new(b.right[0], b.right[1], b.right[2]);
        let u = Vec3::new(b.up[0], b.up[1], b.up[2]);
        assert!((f.length() - 1.0).abs() < 1e-6);
        assert!((r.length() - 1.0).abs() < 1e-6);
        assert!(f.dot(&r).abs() < 1e-6, "forward ⟂ right");
        assert!(f.dot(&u).abs() < 1e-6, "forward ⟂ up");
        assert!(r.dot(&u).abs() < 1e-6, "right ⟂ up");
        assert_eq!(
            b.tan_half_fov.to_bits(),
            prism_math::det_trig::tan(req.camera.fov * 0.5).to_bits()
        );
    }
}
