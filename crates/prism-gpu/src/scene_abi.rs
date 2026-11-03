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
//! | [`SceneAbiProfile::V3`]（W1-A 定桩，**尚无宿主**） | 9 | params / spheres / lights / triangles / nodes / textures / texmeta / cdf / **out** |
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
/// **V2 剖面**每球占用的 f32 槽数：`{center.xyz, radius, albedo.xyz, pad}`（8 = 32 B）。
///
/// 手写 SPIR-V 发射器（`vulkan/spirv_rt.rs`）与 V2 场景 kernel **按此剖面**，
/// 一个字节都不许动（V2 逐位兼容的机械保证）。
pub const SPHERE_SLOTS_V2: u32 = 8;
/// **V3 剖面**每球占用的 f32 槽数：V2 的 8 槽**前缀逐字节不变**，尾部追加
/// `maps[0..4]`（槽 8..11，四个 u32 句柄）= 12 槽 = 48 B。
///
/// 48 B 必须是 4 B 的整数倍（`array<u32>` 的元素步长）——由编译期断言钉住。
pub const SPHERE_SLOTS_V3: u32 = 12;
/// 球块里四张贴图句柄的起始槽位（V3）：`SPHERE_SLOTS_V2`。
///
/// 写成派生式而不是字面量 8：改 V2 步长时 V3 的句柄区必须跟着挪，
/// 否则 shader 会从 albedo 里读出"句柄"。
pub const SPHERE_MAP_SLOT_BASE: u32 = SPHERE_SLOTS_V2;
/// 球块里四张贴图句柄的槽数（= [`TEXMAP_SLOTS`]，`usize` 以便直接索引
/// [`SphereReq::maps`]）。
pub const SPHERE_MAP_SLOTS: usize = TEXMAP_SLOTS as usize;
/// 每灯占用的 f32 槽数：`{dir.xyz（已归一化）, color.xyz, intensity, pad}`。
pub const LIGHT_SLOTS: u32 = 8;

/// 历史别名（= [`SPHERE_SLOTS_V2`]）。
///
/// 保留它是因为既有宿主（`web/scene_gpu.rs` 的缓冲规划、V2 发射器单测）按
/// 「V2 步长」取数；语义**恒为 V2**。V3 宿主必须显式取 [`SPHERE_SLOTS_V3`]。
pub const SPHERE_SLOTS: u32 = SPHERE_SLOTS_V2;

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

// ---------------------------------------------------------------------------
// V3 剖面（W1-A 定桩）：参数块再追加两字
// ---------------------------------------------------------------------------

/// （V3）纹理条数（= `Scene::textures.len()`，同时是 `texmeta` 绑定的元素数）。
pub const P_NUM_TEXTURES: u32 = 24;

/// （V3）环境 CDF 的行数 H（`cdf` 绑定的长度口径 = `(W+1)·H·4` 字节，
/// W 由环境图 `TextureRef.w` 给出）。
///
/// 之所以把 `H` 单独放进参数块、而不让 shader 从 `texmeta` 反推：shader 侧
/// 需要的是一次**整数除法**就得到行跨度 `(W+1)`，而 `W` 与 `H` 分属两个
/// 结构体时，反推要两次间接寻址 + 一次分支（"有没有环境图"）。参数块里
/// 放一个 `H` 把这条依赖压成常量，且让"没有 CDF"时取 0 即可整段跳过。
pub const P_CFD_ROWS: u32 = 25;

/// （V3 / W3）实例个数（= `Scene::instances.len()`）。
pub const P_NUM_INSTANCES: u32 = 26;

/// （V3 / W3）实例 TLAS 的节点个数（`NODE_SLOTS` 步长，与曲线 BVH 同一编码）。
pub const P_NUM_INST_NODES: u32 = 27;

/// **V3** 参数块字数：V2 的 24 字前缀逐字节不变，尾部追加
/// `P_NUM_TEXTURES` / `P_CFD_ROWS`（W1-B）与 `P_NUM_INSTANCES` /
/// `P_NUM_INST_NODES`（W3）四字。
pub const PARAMS_WORDS_V3: usize = 28;

/// 环境 CDF 的字节数口径：`(W+1)·H·4`。
///
/// 与 `prism_rt::texture::cdf_len_bytes` **同式同源**（本 crate 不依赖
/// `prism-rt`，故公式各写一份，由各自的编译期断言钉在同一个数上：
/// 2048×1024 → 8,392,704 B）。
pub const fn cdf_len_bytes(w: u32, h: u32) -> usize {
    ((w as usize) + 1) * (h as usize) * 4
}

/// V3 的 `texmeta` 元素步长（字节）：`array<vec4<u32>>`，两个 `vec4` = 32 B。
///
/// 与 `prism_rt::texture::TEXTURE_REF_BYTES` 同值（编译期断言互钉）。
pub const TEXMETA_STRIDE_BYTES: u32 = 32;

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
/// * [`SceneAbiProfile::V3`]（**W1-A 定桩，尚无宿主**）：9 绑定
///   `{0 params, 1 spheres, 2 lights, 3 triangles, 4 nodes, 5 textures,
///   6 texmeta, 7 cdf, 8 out}`。参数块再追加 `P_NUM_TEXTURES` / `P_CFD_ROWS`
///   两字（[`PARAMS_WORDS_V3`]），总 26 字。
///
///   **本剖面在 W0-A 只有表，没有消费者**：没有 `encode_v3`，没有 WGSL/SPIR-V
///   声明，没有 bind group。它的存在是为了让 W1-B（纹理）与 W2（环境光照）
///   不必各自再改一次「槽位从哪儿开始」——那是并行开工前的唯一串行点。
///   默认剖面仍是 V2（[`SceneAbiProfile::CURRENT`]），故任何既有输出的字节
///   都不受影响。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneAbiProfile {
    /// 4 绑定（params / spheres / lights / out）。
    V1,
    /// 6 绑定（params / spheres / lights / triangles / nodes / out）。
    V2,
    /// 9 绑定（params / spheres / lights / triangles / nodes / textures / texmeta / cdf / out）。
    V3,
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
            // W1-A 定桩：表在这里，消费者（WGSL / SPIR-V / bind group）在 W1-B/W2。
            SceneAbiProfile::V3 => &V3_BINDINGS,
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
            SceneAbiProfile::V3 => PARAMS_WORDS_V3,
        }
    }

    /// 剖面名（诊断/断言用）。
    pub fn name(self) -> &'static str {
        match self {
            SceneAbiProfile::V1 => "V1",
            SceneAbiProfile::V2 => "V2",
            SceneAbiProfile::V3 => "V3",
        }
    }
}

// ---------------------------------------------------------------------------
// V3 绑定表 + 编译期断言（W1-A 定桩的全部产出之一）
// ---------------------------------------------------------------------------

/// V3 剖面的绑定表：**顺序即 binding 编号**，末位恒为输出。
///
/// 选型报告 §11 的槽位顺序逐字落地（W3 在尾部追加两块，输出后移一位）：
/// `{params, spheres, lights, triangles, nodes, textures, texmeta, cdf,
///   instances, instance_nodes, out}`。
///
/// # 为什么实例两块**追加在 cdf 之后**（而不是插在 nodes 后面）
///
/// 追加 = 既有 8 个绑定的编号**一个都没动**（0..7 逐格不变），于是 W1-B/W2 的
/// 编码器、`.spv` 声明与宿主代码只改"输出绑定号 + 尾部两块"，不改既有语义。
/// 插在中间会让 `textures/texmeta/cdf` 三块整体后移，等于把 W1/W2 的判据全部
/// 重新对一遍——没有收益。
///
/// # 输出恒为最后一个绑定
///
/// 这是三份源（ABI / WGSL / 宿主）共用的硬约束。W3 把它从 8 挪到 10，
/// 所有消费者都从 [`SceneAbiProfile::out_binding`] 派生，**不许写字面量**。
pub const V3_BINDINGS: [SceneBinding; 11] = [
    SceneBinding { binding: 0, name: "params", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 1, name: "spheres", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 2, name: "lights", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 3, name: "triangles", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 4, name: "nodes", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 5, name: "textures", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 6, name: "texmeta", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 7, name: "cdf", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 8, name: "instances", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 9, name: "instance_nodes", access: BindingAccess::ReadOnly },
    SceneBinding { binding: 10, name: "out", access: BindingAccess::ReadWrite },
];

/// 实例记录的 ABI 常量从 [`crate::instance`] 再导出：`scene_abi` 是宿主查 ABI 的
/// 唯一入口，宿主不该为了 `INSTANCE_BYTES` 再 import 一个模块。
pub use crate::instance::{
    InstanceReq, INSTANCE_AFFINE_WORDS, INSTANCE_BYTES, INSTANCE_FLAG_IDENTITY,
    INSTANCE_FLAG_MIRRORED, INSTANCE_GEOM_SHARED, INSTANCE_INVERSE_WORDS, INSTANCE_MAT_FROM_GEOM,
    INSTANCE_SLOTS,
};

/// V3 各绑定的编号（宿主据此声明 `@binding(k)`，禁止写字面量）。
pub const V3_PARAMS: u32 = 0;
/// 见 [`V3_PARAMS`]。
pub const V3_SPHERES: u32 = 1;
/// 见 [`V3_PARAMS`]。
pub const V3_LIGHTS: u32 = 2;
/// 见 [`V3_PARAMS`]。
pub const V3_TRIANGLES: u32 = 3;
/// 见 [`V3_PARAMS`]。
pub const V3_NODES: u32 = 4;
/// 纹理像素（`array<vec4<f32>>`；**禁** `array<vec3<f32>>`）。
pub const V3_TEXTURES: u32 = 5;
/// 纹理元数据（`array<vec4<u32>>`，[`TEXMETA_STRIDE_BYTES`] 字节/条）。
pub const V3_TEXMETA: u32 = 6;
/// 环境 CDF（`array<u32>`，`(W+1)·H·4` 字节）。
pub const V3_CDF: u32 = 7;
/// 实例记录（`array<u32>`，[`INSTANCE_BYTES`] 字节/条 = 28 槽）。
pub const V3_INSTANCES: u32 = 8;
/// 实例 TLAS 节点（`array<u32>`，[`NODE_SLOTS`] 槽/节点 = 48 B）。
pub const V3_INSTANCE_NODES: u32 = 9;
/// 输出（**恒为最后一个绑定**）。
pub const V3_OUT: u32 = 10;

/// `Bsdf::maps` 的槽数（与 `prism_rt::bsdf::MAP_SLOTS` 同值）。
///
/// 两份常量不能互相 import（`prism-gpu` 不得反引 `prism-rt`），故各写一份
/// 并由各自的断言钉在 4；真正的绑定关系在 W1-B 的编码器单测里对拍。
pub const TEXMAP_SLOTS: u32 = 4;

/// [`SphereReq::maps`] 槽 0：基础色贴图（= `prism_rt::bsdf::MAP_BASE`）。
///
/// 四张贴图里**只有这一个**在本版着色模型里有落点（命中点 uv 采样替换
/// albedo）。其余三槽的常量照样定义，是为了让"槽序"这件事只有一个真源：
/// 顺序错一位的图像症状（"颜色对、光照全歪"）在真机上极难定位。
pub const MAP_BASE_SLOT: usize = 0;
/// 槽 1：金属度-粗糙度（glTF：G = roughness、B = metalness）。**本版不读**。
pub const MAP_METAL_ROUGH_SLOT: usize = 1;
/// 槽 2：切线空间法线。**本版不读**。
pub const MAP_NORMAL_SLOT: usize = 2;
/// 槽 3：自发光。**本版不读**。
pub const MAP_EMISSIVE_SLOT: usize = 3;

/// `const fn`：取绑定的访问属性（供 `const _` 断言用；`PartialEq` 不是 const）。
const fn access_is_read_write(a: BindingAccess) -> bool {
    match a {
        BindingAccess::ReadWrite => true,
        BindingAccess::ReadOnly => false,
    }
}

/// `const fn` 字符串相等（`==` 在 const 上下文不可用）。
const fn str_eq(a: &str, b: &str) -> bool {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    if ab.len() != bb.len() {
        return false;
    }
    let mut i = 0;
    while i < ab.len() {
        if ab[i] != bb[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// **V3 槽位顺序断言**（选型报告 §11 的槽位表逐格对账）。
///
/// 改错顺序（比如把 `texmeta` 和 `cdf` 对调）在这里就编不过——而不是等到
/// W2 接线后 shader 从 CDF 里读出纹理元数据、表现出"环境光照成了一张贴图"。
const _: () = {
    let bs = V3_BINDINGS;
    assert!(bs.len() == 11);
    let mut i = 0;
    while i < bs.len() {
        // 连续前缀：binding 编号必须等于下标（宿主据此派生 bind group）。
        assert!(bs[i].binding as usize == i);
        i += 1;
    }
    // 顺序即语义（逐格比对，不靠"看起来对"）。
    assert!(str_eq(bs[0].name, "params"));
    assert!(str_eq(bs[1].name, "spheres"));
    assert!(str_eq(bs[2].name, "lights"));
    assert!(str_eq(bs[3].name, "triangles"));
    assert!(str_eq(bs[4].name, "nodes"));
    assert!(str_eq(bs[5].name, "textures"));
    assert!(str_eq(bs[6].name, "texmeta"));
    assert!(str_eq(bs[7].name, "cdf"));
    assert!(str_eq(bs[8].name, "instances"));
    assert!(str_eq(bs[9].name, "instance_nodes"));
    assert!(str_eq(bs[10].name, "out"));
    // **输出恒为最后一个绑定**（这是三份源共用的硬约束）。
    assert!(str_eq(bs[10].name, "out"));
    assert!(access_is_read_write(bs[10].access));
    // 除输出外全部只读（写错的后果是驱动直接拒绝创建 bind group layout）。
    let mut j = 0;
    while j < 10 {
        assert!(!access_is_read_write(bs[j].access));
        j += 1;
    }
};

/// 参数块尾部追加两字：V3 = V2 + 2，且两字的编号紧接 V2 末字。
const _: () = assert!(PARAMS_WORDS_V3 == PARAMS_WORDS_V2 + 4);
const _: () = assert!(P_NUM_TEXTURES == PARAMS_WORDS_V2 as u32);
const _: () = assert!(P_CFD_ROWS == P_NUM_TEXTURES + 1);
/// 参数块末字编号 + 1 == 字数（不多不少，`encode_v3` 才可能写满）。
const _: () = assert!(PARAMS_WORDS_V3 == P_NUM_INST_NODES as usize + 1);
/// 四个 V3 追加字**紧挨且有序**（乱序会让 shader 与编码器读到对方的计数）。
const _: () = assert!(P_NUM_TEXTURES + 1 == P_CFD_ROWS);
const _: () = assert!(P_CFD_ROWS + 1 == P_NUM_INSTANCES);
const _: () = assert!(P_NUM_INSTANCES + 1 == P_NUM_INST_NODES);
/// CDF 长度口径：2048×1024 → 8,392,704 B（选型报告 §3.1 第 2 条）。
const _: () = assert!(cdf_len_bytes(2048, 1024) == 8_392_704);
/// `texmeta` 步长：`array<vec4<u32>>` 的 2 格 = 32 B（必须是 16 的倍数）。
const _: () = assert!(TEXMETA_STRIDE_BYTES == 32);
const _: () = assert!(TEXMETA_STRIDE_BYTES % 16 == 0);
/// 四张贴图句柄（`base / metal-rough / normal / emissive`）。
const _: () = assert!(TEXMAP_SLOTS == 4);

// ---- V3 球块剖面（W1-C 段 1）：V2 的 8 槽 + 四个句柄 ----

/// V3 球块 = V2 球块 + 4 槽（**不是**"另一套布局"，只是尾部加长）。
const _: () = assert!(SPHERE_SLOTS_V3 == SPHERE_SLOTS_V2 + 4);
/// 句柄区起点**恒等于** V2 步长（改 V2 步长则句柄区必须跟着挪）。
const _: () = assert!(SPHERE_MAP_SLOT_BASE == SPHERE_SLOTS_V2);
/// 球块的句柄槽数 = `Bsdf::maps` 的槽数（两份常量各写一份，钉在同一个数上）。
const _: () = assert!(SPHERE_MAP_SLOTS == TEXMAP_SLOTS as usize);
/// V3 球块 48 B：**恰好**是 12 个 u32 的宽度，没有尾部 padding。
///
/// 这条防的是"改成 13 槽但缓冲仍按 12 槽分配"这类错——那会让最后一个球的
/// `maps[3]` 读到下一个球的 `center.x`。
const _: () = assert!(SPHERE_SLOTS_V3 as usize * core::mem::size_of::<u32>() == 48);
const _: () = assert!(SPHERE_SLOTS_V2 as usize * core::mem::size_of::<u32>() == 32);
/// V2 球块 32 B 的既有契约不许动（手写 SPIR-V 按此为步长）。
const _: () = assert!(SPHERE_SLOTS == SPHERE_SLOTS_V2);

// ---- 四个句柄槽的序号（槽序 = `Bsdf::maps` = WGSL `MAP_*`）----
/// 四个槽必须**恰好**铺满 0..3（漏一个或重一个都是 ABI 错）。
const _: () = assert!(MAP_BASE_SLOT == 0);
const _: () = assert!(MAP_METAL_ROUGH_SLOT == 1);
const _: () = assert!(MAP_NORMAL_SLOT == 2);
const _: () = assert!(MAP_EMISSIVE_SLOT == 3);
const _: () = assert!(MAP_EMISSIVE_SLOT + 1 == SPHERE_MAP_SLOTS);

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

    /// **V3 剖面编码**（W1-B）：在 V2 五块缓冲之外追加 `textures` / `texmeta` / `cdf`。
    ///
    /// 返回 [`V3Buffers`]（8 块输入缓冲；第 9 个绑定 `out` 由 shader 写）。
    /// `params` 为 [`PARAMS_WORDS_V3`] 个字：V2 的 24 字前缀**逐字节不变**，
    /// 尾部追加 `P_NUM_TEXTURES` / `P_CFD_ROWS`。
    ///
    /// # 逐位兼容（V2 不回归）
    ///
    /// `textures` 为空 + `env` 为 `None` ⇒ `textures`/`texmeta`/`cdf` 三块皆空、
    /// 两计数为 0，且 `params[..24]` 与 [`SceneAbi::encode_v2`] 的返回值逐字节
    /// 相同。V3 只是"同一个场景 + 三块可选缓冲"，不是另一套语义。
    ///
    /// # 球块（W1-C 段 1 起）
    ///
    /// `spheres` 走 [`encode_spheres_v3`]：每球 [`SPHERE_SLOTS_V3`] 槽 = 48 B，
    /// 其中**前 32 B 与 [`encode_spheres`] 逐字节相同**，尾部 16 B 是四个贴图
    /// 句柄。故 V3 的球块长度是 V2 的 1.5 倍——这是本剖面唯一的缓冲布局差异
    /// （shader 侧的 `SPHERE_STRIDE` 同步为 12）。
    ///
    /// # 为什么纹理在这里"展开"而不是让宿主自己拼
    ///
    /// `texmeta` 的 `offset` 是**纹理在扁平纹素流里的起点**，只有把所有纹理
    /// 按序累加才算得对。让每个宿主各算一遍 = 每个宿主一处下标 bug；本函数
    /// 是唯一入口（与 `encode_v2` 的曲线缓冲同一纪律）。
    ///
    /// # 错误（不 panic，C18）
    ///
    /// 尺寸为 0 / 级数越界 / 纹素条数与描述不符 / CDF 字节数不符 / 纹理太多
    /// ⇒ `Err`（带可定位的说明）。这些错误在真机上表现为"驱动读到垃圾"或
    /// "整帧黑"，故必须在 CPU 侧拦死。
    pub fn encode_v3(
        req: &SceneRenderRequest,
        textures: &[TextureUpload<'_>],
        env: Option<EnvCdf<'_>>,
    ) -> Result<V3Buffers, String> {
        Self::encode_v3_with_instances(req, textures, env, &[])
    }

    /// **V3 + W3 实例**：同 [`SceneAbi::encode_v3`]，另外编码实例表与实例 TLAS。
    ///
    /// # 语义（与 `prism-rt` 的 `ShapeRef::Instance` 对齐）
    ///
    /// * `triangles` / `nodes` 两块 = **共享几何那一个 BLAS**（原型，局部坐标）；
    /// * `instances` / `instance_nodes` = 1 万级实例 + TLAS（世界坐标 AABB）；
    /// * `P_NUM_INSTANCES > 0` 时原型**不再作为世界几何绘制**（它只能经实例
    ///   到达）——这是 `prism-rt` 把 `instance_geoms` 与 `meshes` 分开的同一条
    ///   纪律（选型报告 §17.3）：否则"沿 x 摆 10 份"会变成"10 份正确 + 1 份
    ///   莫名钉在原点"。
    ///
    /// # 逆矩阵不在后端算
    ///
    /// 调用方传进来的 [`InstanceReq`] 必须已由 [`InstanceReq::from_affine`]
    /// 算好逆（CPU 一次）；本函数**不做求逆**，只编码。
    ///
    /// # 错误（不 panic）
    ///
    /// `geom_id != 0`（单 BLAS 剖面）/ 实例表非空但原型几何为空 / 退化实例
    /// ⇒ `Err`（带可定位说明）。
    pub fn encode_v3_instanced(
        req: &SceneRenderRequest,
        textures: &[TextureUpload<'_>],
        env: Option<EnvCdf<'_>>,
        instances: &[InstanceReq],
    ) -> Result<V3Buffers, String> {
        Self::encode_v3_with_instances(req, textures, env, instances)
    }

    /// [`SceneAbi::encode_v3`] / [`SceneAbi::encode_v3_instanced`] 的共同实现。
    ///
    /// `instances` 为空 ⇒ 实例两块缓冲为空、两计数为 0，WGSL 的实例遍历段整段
    /// 跳过 ⇒ 与 W1-C/W2 的 V3 输出**逐位相同**（W3 的不回归机械保证）。
    fn encode_v3_with_instances(
        req: &SceneRenderRequest,
        textures: &[TextureUpload<'_>],
        env: Option<EnvCdf<'_>>,
        instances: &[InstanceReq],
    ) -> Result<V3Buffers, String> {
        if textures.len() > MAX_V3_TEXTURES {
            return Err(format!(
                "纹理条数 {} 超过上限 {MAX_V3_TEXTURES}；建议：合并图集，或分批提交",
                textures.len()
            ));
        }
        let mut params = [0u8; PARAMS_WORDS_V3 * 4];
        let basis = camera_basis(&req.camera);
        let v1 = encode_params(req, &basis);
        params[..v1.len()].copy_from_slice(&v1);
        // V3 球块：V2 的 8 槽前缀逐字节不变 + 尾部 4 个贴图句柄。
        let spheres = encode_spheres_v3(&req.spheres);
        let lights = encode_lights(&req.lights);
        let (tris, nodes) = encode_curve_buffers(&req.curves);
        put_u32(
            &mut params,
            P_NUM_TRIS as usize,
            (tris.len() / (TRIANGLE_SLOTS as usize * 4)) as u32,
        );
        put_u32(
            &mut params,
            P_NUM_NODES as usize,
            (nodes.len() / (NODE_SLOTS as usize * 4)) as u32,
        );

        // ---- textures / texmeta（offset 必须按"纹素"累加，不是按字节）----
        let mut texel_bytes: Vec<u8> = Vec::new();
        let mut texmeta: Vec<u8> = Vec::new();
        let mut cursor: u32 = 0;
        for (i, t) in textures.iter().enumerate() {
            t.validate()
                .map_err(|e| format!("纹理 {i} 非法：{e}"))?;
            let count = t.expected_texels() as u32;
            let end = cursor.checked_add(count).ok_or_else(|| {
                format!("纹理 {i} 的起始纹素 {cursor} + {count} 溢出 u32（纹素流太大）")
            })?;
            texmeta.extend_from_slice(&t.meta_bytes(cursor));
            // 纹素：`array<vec4<f32>>` 逐格小端 f32（4 × 4 B）。
            // `srgb != 0`：在**上传前**把 RGB 解码到线性域（alpha 不动），
            // 与 `cpu_tex::RefTextureTable::new` 逐纹素同式 ⇒ 三后端拿到同一份线性 f32。
            let decode = t.srgb != 0;
            texel_bytes.reserve(t.texels.len() * 16);
            for c in t.texels {
                let rgb = if decode {
                    [
                        srgb_decode_f32(c[0]),
                        srgb_decode_f32(c[1]),
                        srgb_decode_f32(c[2]),
                        c[3],
                    ]
                } else {
                    *c
                };
                for ch in rgb {
                    texel_bytes.extend_from_slice(&ch.to_le_bytes());
                }
            }
            cursor = end;
        }
        debug_assert_eq!(texel_bytes.len(), cursor as usize * 16);

        // ---- cdf ----
        let (cdf, cdf_rows) = match env {
            Some(e) => {
                let want = cdf_len_bytes(e.w, e.h);
                if e.w == 0 || e.h == 0 {
                    return Err(format!(
                        "环境 CDF 尺寸非法（{}×{}）；建议：W,H ≥ 1",
                        e.w, e.h
                    ));
                }
                if e.bytes.len() != want {
                    return Err(format!(
                        "环境 CDF 字节数 {} ≠ (W+1)·H·4 = ({} + 1)·{}·4 = {want}；\
                         差一格会让整条前缀和错位（行内二分找到邻居的台阶）",
                        e.bytes.len(),
                        e.w,
                        e.h
                    ));
                }
                (e.bytes.to_vec(), e.h)
            }
            None => (Vec::new(), 0),
        };

        put_u32(&mut params, P_NUM_TEXTURES as usize, textures.len() as u32);
        put_u32(&mut params, P_CFD_ROWS as usize, cdf_rows);

        // ---- W3：实例 + TLAS ----
        let (instances_bytes, instance_nodes, num_instances, num_inst_nodes) =
            if instances.is_empty() {
                (Vec::new(), Vec::new(), 0u32, 0u32)
            } else {
                for (i, inst) in instances.iter().enumerate() {
                    if inst.geom_id != INSTANCE_GEOM_SHARED {
                        return Err(format!(
                            "实例 {i} 的 geom_id = {} ≠ {INSTANCE_GEOM_SHARED}：\
                             V3 剖面只有**一个** BLAS（`triangles`/`nodes` 就是它）；\
                             多原型需要再开两块缓冲，W3 未做（不是漏：判据只要一个共享原型）",
                            inst.geom_id
                        ));
                    }
                }
                let (pmin, pmax) = prototype_bounds(&tris).ok_or_else(|| {
                    "实例表非空但原型几何为空：实例需要有东西可实例化\
                     （V3 剖面用 `triangles`/`nodes` 承载那一个 BLAS）"
                        .to_string()
                })?;
                let set = crate::instance::build_instance_set(instances, pmin, pmax);
                let nodes_len = (set.nodes.len() / (NODE_SLOTS as usize * 4)) as u32;
                (
                    crate::instance::encode_instances(&set.instances),
                    set.nodes,
                    set.instances.len() as u32,
                    nodes_len,
                )
            };
        put_u32(&mut params, P_NUM_INSTANCES as usize, num_instances);
        put_u32(&mut params, P_NUM_INST_NODES as usize, num_inst_nodes);

        Ok(V3Buffers {
            params,
            spheres,
            lights,
            triangles: tris,
            nodes,
            textures: texel_bytes,
            texmeta,
            cdf,
            instances: instances_bytes,
            instance_nodes,
        })
    }
}

/// 从已编码的三角形槽位里取原型几何的局部 AABB（`None` = 空）。
///
/// 直接读 `triangles` 字节而不是重新三角化：`encode_curve_buffers` 已经把
/// 曲线烘成三角形（并建了 BVH），再算一遍既慢又可能与 BVH 里的顶点不同源。
///
/// 公开的理由：宿主要建 TLAS 就得知道原型的世界包围盒，而"原型是谁"这个
/// 判断的权威在本函数（`INSTANCE_GEOM_SHARED` 指向的就是 `triangles` 那一块）。
pub fn prototype_bounds(tris: &[u8]) -> Option<(Vec3, Vec3)> {
    let stride = TRIANGLE_SLOTS as usize * 4;
    if tris.is_empty() || tris.len() < stride {
        return None;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for block in tris.chunks_exact(stride) {
        for base in [0usize, 4, 8] {
            let x = f32::from_le_bytes([
                block[base * 4],
                block[base * 4 + 1],
                block[base * 4 + 2],
                block[base * 4 + 3],
            ]);
            let y = f32::from_le_bytes([
                block[base * 4 + 4],
                block[base * 4 + 5],
                block[base * 4 + 6],
                block[base * 4 + 7],
            ]);
            let z = f32::from_le_bytes([
                block[base * 4 + 8],
                block[base * 4 + 9],
                block[base * 4 + 10],
                block[base * 4 + 11],
            ]);
            min = Vec3::new(min.x.min(x), min.y.min(y), min.z.min(z));
            max = Vec3::new(max.x.max(x), max.y.max(y), max.z.max(z));
        }
    }
    if min.x.is_finite() && min.x <= max.x {
        Some((min, max))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// V3 剖面（W1-B）：纹理上传描述 + 编码产物
// ---------------------------------------------------------------------------

/// V3 纹理条数上限（`P_NUM_TEXTURES` 是 u32，但 bind group 与内存都有实际
/// 上限；4096 条已远超任何单帧场景，超出即视为调用方 bug）。
pub const MAX_V3_TEXTURES: usize = 4096;

/// 单个纹素的字节数（`array<vec4<f32>>` 的元素 = 4 × f32）。
///
/// **禁 `array<vec3<f32>>`**：`vec3` 在 std430 里步长 16 B、但 WGSL 只保证
/// 12 B 的成员尺寸，三后端对"最后一个元素的数组步长"处理不一致（D-15 的
/// 老账）。`vec4` 无歧义，且与 [`crate::scene_abi::TEXMETA_STRIDE_BYTES`]
/// 的保守写法同源。
pub const TEXEL_BYTES: u32 = 16;
const _: () = assert!(TEXEL_BYTES == 4 * core::mem::size_of::<f32>() as u32);
const _: () = assert!(TEXEL_BYTES % 16 == 0);

/// 一张待上传纹理（V3）：元数据 + **线性域**扁平纹素（含 mip 链，级序递增）。
///
/// # 为什么纹素必须已经是线性的
///
/// sRGB 源在 CPU 侧（`prism_rt::texture::build_rgba8`）一次性解码，三后端
/// 上传的是同一份线性位型 ⇒ "三后端读同一份 f32 位"是结构性的，而不是
/// "三份 shader 各写一遍解码、指望它们一致"。滤波也因此发生在物理正确的
/// 线性域（在线性域做双线性会让暗部偏亮）。
#[derive(Debug, Clone, Copy)]
pub struct TextureUpload<'a> {
    /// 宽度（纹素，≥ 1）。
    pub w: u32,
    /// 高度（纹素，≥ 1）。
    pub h: u32,
    /// mip 级数（≥ 1；`1` = 无 mip 链）。
    pub levels: u32,
    /// 像素格式（`prism_rt::texture::TEX_FMT_*` 同值）。
    pub fmt: u32,
    /// 传输曲线标记（`SRGB_LINEAR` / `SRGB_ENCODED` 同值）。
    pub srgb: u32,
    /// 扁平纹素（含 mip 链；长度必须等于 [`TextureUpload::expected_texels`]）。
    pub texels: &'a [[f32; 4]],
}

impl<'a> TextureUpload<'a> {
    /// 该描述应占的纹素数：`Σ_{l<levels} max(1, w>>l)·max(1, h>>l)`。
    ///
    /// 与 `prism_rt::texture::TextureRef::texel_count` 同式（本 crate 不依赖
    /// `prism-rt`，故各写一份；对拍在编码器单测里做）。
    pub fn expected_texels(&self) -> usize {
        let mut n = 0usize;
        let mut l = 0u32;
        let levels = self.levels.max(1);
        while l < levels {
            let w = ((self.w.max(1)) >> l.min(31)).max(1) as usize;
            let h = ((self.h.max(1)) >> l.min(31)).max(1) as usize;
            n += w * h;
            l += 1;
        }
        n
    }

    /// `texmeta` 的 32 字节（`array<vec4<u32>>` 的两格，字段顺序与
    /// `prism_rt::texture::TextureRef` 的 `repr(C)` 逐格一致）。
    pub fn meta_bytes(&self, offset: u32) -> [u8; TEXMETA_STRIDE_BYTES as usize] {
        let mut out = [0u8; TEXMETA_STRIDE_BYTES as usize];
        put_u32(&mut out, 0, self.w);
        put_u32(&mut out, 1, self.h);
        put_u32(&mut out, 2, self.levels.max(1));
        put_u32(&mut out, 3, offset);
        put_u32(&mut out, 4, self.fmt);
        put_u32(&mut out, 5, self.srgb);
        // 槽 6/7 = `reserved`，恒 0（不承载语义；改它等于改 ABI）。
        out
    }

    /// 校验（不 panic）：尺寸、级数、纹素条数。
    pub fn validate(&self) -> Result<(), String> {
        if self.w == 0 || self.h == 0 {
            return Err(format!("尺寸 {}×{} 含 0；建议：空图用 1×1 占位", self.w, self.h));
        }
        if self.levels == 0 || self.levels > 32 {
            return Err(format!(
                "mip 级数 {} 越界；合法范围 1..=32（2^32 已超过任何纹理尺寸）",
                self.levels
            ));
        }
        let want = self.expected_texels();
        if self.texels.len() != want {
            return Err(format!(
                "{}×{} / {} 级应有 {want} 个纹素，收到 {}；\
                 mip 链必须**完整**（缺级会让三线性在末级读到下一张图的像素）",
                self.w,
                self.h,
                self.levels,
                self.texels.len()
            ));
        }
        Ok(())
    }
}

/// sRGB → 线性（IEC 61966-2-1 **EOTF**，编码值 → 线性光）。
///
/// 纹素在 CPU 侧**上传前**解码一次（见 [`TextureUpload::srgb`] 与 `encode_v3` /
/// `cpu_tex::RefTextureTable::new`）；解码后的 f32 **同时**送给三后端，故滤波
/// 发生在物理正确的线性域（在线性域做双线性才会让暗部偏亮）。
///
/// 本函数与 [`srgb_encode_f32`] 互逆，往返误差 ≤ 0.5 LSB（8-bit），见
/// `scene_abi::tests::srgb_round_trip_is_lossless_within_half_lsb`。
///
/// 仅 CPU 侧使用（`prism-gpu` 不得反引 `prism-rt`，故不调 `prism_rt::texture`）；
/// GPU 收到的是**已解码的线性 f32**，shader 不再做 sRGB 解码。
pub fn srgb_decode_f32(u: f32) -> f32 {
    let c = u.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        // D-29：GPU/生产源码禁 libm 超越函数；sRGB EOTF 的 `x^2.4` 走
        // `prism_math::det_pow`（exp2(p·log2 x)，纯 IEEE 确定性，逐位一致）。
        prism_math::det_pow::det_pow((c + 0.055) / 1.055, 2.4)
    }
}

/// 线性 → sRGB（IEC 61966-2-1 **逆 EOTF**）。
///
/// 仅用于构造 sRGB 编码的**测试数据**（把一条线性渐变编码成贴图再走解码路径）；
/// 生产路径只用 [`srgb_decode_f32`]，shader 永远收到线性域纹素。
pub fn srgb_encode_f32(l: f32) -> f32 {
    let c = l.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        // D-29：同上，`x^(1/2.4)` 走 `prism_math::det_pow`（仅测试数据构造用）。
        1.055 * prism_math::det_pow::det_pow(c, 1.0 / 2.4) - 0.055
    }
}

/// 环境图 CDF（V3）：`(W+1)·H·4` 字节的行主序前缀和。
///
/// 语义与 `prism_rt::texture::assert_cdf_len` 同口径：每行 `W+1` 个 u32
/// （末列是行总和，便于行内二分），行数 `H`。
#[derive(Debug, Clone, Copy)]
pub struct EnvCdf<'a> {
    /// 环境图宽度（纹素）。
    pub w: u32,
    /// 环境图高度（纹素）。
    pub h: u32,
    /// `(W+1)·H·4` 字节。
    pub bytes: &'a [u8],
}

/// V3 编码产物：**10 块输入缓冲**，下标即 `@binding` 编号（0..10，`out` 除外）。
///
/// 用结构体而不是 10 元组：V3 的绑定顺序是 ABI（`V3_BINDINGS`），元组下标
/// 会让"第 5 个是哪块"变成阅读理解题。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3Buffers {
    /// 绑定 0：参数块（[`PARAMS_WORDS_V3`] × 4 B）。
    pub params: [u8; PARAMS_WORDS_V3 * 4],
    /// 绑定 1：球。
    pub spheres: Vec<u8>,
    /// 绑定 2：灯。
    pub lights: Vec<u8>,
    /// 绑定 3：三角形。
    pub triangles: Vec<u8>,
    /// 绑定 4：BVH 节点。
    pub nodes: Vec<u8>,
    /// 绑定 5：纹素（`array<vec4<f32>>`）。
    pub textures: Vec<u8>,
    /// 绑定 6：纹理元数据（`array<vec4<u32>>`，32 B/条）。
    pub texmeta: Vec<u8>,
    /// 绑定 7：环境 CDF（`array<u32>`）。
    pub cdf: Vec<u8>,
    /// 绑定 8：实例记录（`array<u32>`，[`INSTANCE_BYTES`] 字节/条；已按 TLAS 叶子序重排）。
    pub instances: Vec<u8>,
    /// 绑定 9：实例 TLAS 节点（`array<u32>`，[`NODE_SLOTS`] 槽/条）。
    pub instance_nodes: Vec<u8>,
}

impl V3Buffers {
    /// 按 `@binding` 顺序返回的输入缓冲（长度 = `V3_BINDINGS.len() - 1`）。
    ///
    /// 宿主建 bind group 时按下标取用，杜绝"手写 5/6 号绑定搞反"。
    pub fn inputs(&self) -> [&[u8]; 10] {
        [
            &self.params,
            &self.spheres,
            &self.lights,
            &self.triangles,
            &self.nodes,
            &self.textures,
            &self.texmeta,
            &self.cdf,
            &self.instances,
            &self.instance_nodes,
        ]
    }

    /// 纹理条数（= `params[P_NUM_TEXTURES]`）。
    pub fn num_textures(&self) -> u32 {
        u32::from_le_bytes([
            self.params[P_NUM_TEXTURES as usize * 4],
            self.params[P_NUM_TEXTURES as usize * 4 + 1],
            self.params[P_NUM_TEXTURES as usize * 4 + 2],
            self.params[P_NUM_TEXTURES as usize * 4 + 3],
        ])
    }

    /// 环境 CDF 行数（= `params[P_CFD_ROWS]`；无环境图为 0）。
    pub fn cdf_rows(&self) -> u32 {
        u32::from_le_bytes([
            self.params[P_CFD_ROWS as usize * 4],
            self.params[P_CFD_ROWS as usize * 4 + 1],
            self.params[P_CFD_ROWS as usize * 4 + 2],
            self.params[P_CFD_ROWS as usize * 4 + 3],
        ])
    }

    /// 实例个数（= `params[P_NUM_INSTANCES]`）。
    pub fn num_instances(&self) -> u32 {
        u32::from_le_bytes([
            self.params[P_NUM_INSTANCES as usize * 4],
            self.params[P_NUM_INSTANCES as usize * 4 + 1],
            self.params[P_NUM_INSTANCES as usize * 4 + 2],
            self.params[P_NUM_INSTANCES as usize * 4 + 3],
        ])
    }

    /// 实例 TLAS 节点数（= `params[P_NUM_INST_NODES]`）。
    pub fn num_instance_nodes(&self) -> u32 {
        u32::from_le_bytes([
            self.params[P_NUM_INST_NODES as usize * 4],
            self.params[P_NUM_INST_NODES as usize * 4 + 1],
            self.params[P_NUM_INST_NODES as usize * 4 + 2],
            self.params[P_NUM_INST_NODES as usize * 4 + 3],
        ])
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

/// 编码球体缓冲（每球 [`SPHERE_SLOTS_V2`] 个 f32 = 32 字节）。
///
/// 槽内布局：`center.x/y/z`、`radius`、`albedo.r/g/b`、`pad`（恒 0）。
///
/// # 这是 **V2** 编码器（纹理句柄放不下）
///
/// V2 球块只有 8 槽，槽 7 是 pad ⇒ [`SphereReq::maps`] 在这里**不被写出**。
/// 走 V3 剖面（要纹理）必须用 [`encode_spheres_v3`]；本函数的字节流与
/// W1-B 之前**逐字节相同**（V2 通路不回归的机械保证）。
pub fn encode_spheres(spheres: &[SphereReq]) -> Vec<u8> {
    let mut out = Vec::with_capacity(spheres.len() * SPHERE_SLOTS_V2 as usize * 4);
    for s in spheres {
        let mut block = [0u8; SPHERE_SLOTS_V2 as usize * 4];
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

/// **V3** 球体缓冲编码（每球 [`SPHERE_SLOTS_V3`] 个 u32 槽 = 48 字节）。
///
/// 槽内布局：
///
/// | 槽 | 内容 | 编码 |
/// |---|---|---|
/// | 0..3 | `center.xyz` / `radius` | f32 **位型**（与 [`encode_spheres`] 逐字节相同） |
/// | 4..7 | `albedo.rgb` / `pad`（恒 0） | f32 **位型** / 0 |
/// | 8..11 | `maps[0..4]`（base / metal-rough / normal / emissive） | **u32 句柄原值**（0 = 无贴图） |
///
/// # 两个不能含糊的点
///
/// 1. **槽 0..7 与 V2 逐字节相同**：句柄只是"尾部追加"，既有 8 槽的字节序、
///    位型、pad 位置一律照旧。故同一场景的 V3 球块前 32 B 恒等于 V2 球块。
/// 2. **句柄是 u32 原值，不是 f32 位型**：shader 侧读
///    `spheres[i * SPHERE_SLOTS_V3 + 8 + slot]`（`array<u32>`）直接用，
///    不做 `bitcast`。把句柄当 f32 写会产生 `1.4e-45` 这类指数下溢值。
///
/// 句柄**不做范围校验**（这一层不知道纹理表多长）：越界句柄由 shader 侧的
/// `handle <= P_NUM_TEXTURES` 守卫降级为"无贴图"（见 `scene_frame_v3.wgsl`），
/// CPU 参考走同一判据。
pub fn encode_spheres_v3(spheres: &[SphereReq]) -> Vec<u8> {
    let mut out = Vec::with_capacity(spheres.len() * SPHERE_SLOTS_V3 as usize * 4);
    for s in spheres {
        let mut block = [0u8; SPHERE_SLOTS_V3 as usize * 4];
        put_f32(&mut block, 0, s.center.x);
        put_f32(&mut block, 1, s.center.y);
        put_f32(&mut block, 2, s.center.z);
        put_f32(&mut block, 3, s.radius);
        put_f32(&mut block, 4, s.albedo.x);
        put_f32(&mut block, 5, s.albedo.y);
        put_f32(&mut block, 6, s.albedo.z);
        // 槽 7：V2 的 pad，恒 0（保持 0..7 与 V2 逐字节相同的唯一做法）。
        // 槽 8..11：四张贴图句柄，u32 原值。
        for slot in 0..SPHERE_MAP_SLOTS {
            put_u32(
                &mut block,
                SPHERE_MAP_SLOT_BASE as usize + slot,
                s.maps[slot],
            );
        }
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
                maps: crate::NO_MAPS,
            }],
            aesthetic: crate::AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        curves: Vec::new(),
    ..Default::default()
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

    /// **V3 剖面（W1-A 定桩）**：9 绑定连续前缀，槽位顺序与选型报告 §11 逐字一致，
    /// 输出恒为最后一个绑定，参数块尾部追加 `P_NUM_TEXTURES` / `P_CFD_ROWS`。
    ///
    /// 编译期断言（`const _`）已经把这张表钉住；本测试是它的**运行时镜像**，
    /// 目的不是重复，而是让"断言被误删"这件事在测试层也留一道痕
    /// （`const _` 被删掉时编译照样过）。
    #[test]
    fn bindings_v3_profile_matches_spec_slot_order() {
        let bs = SceneAbi::bindings_for(SceneAbiProfile::V3);
        assert_eq!(bs.len(), 11, "V3 剖面恒为 11 绑定（W3 在 cdf 之后追加实例两块）");
        let names: Vec<&str> = bs.iter().map(|b| b.name).collect();
        assert_eq!(
            names,
            vec![
                "params", "spheres", "lights", "triangles", "nodes", "textures", "texmeta",
                "cdf", "instances", "instance_nodes", "out"
            ],
            "V3 绑定顺序必须与选型报告 §11 的槽位表逐字一致（W3 只往后追加）"
        );
        for (i, b) in bs.iter().enumerate() {
            assert_eq!(b.binding, i as u32, "绑定必须从 0 起连续（连续前缀约束）");
        }
        // 输出恒为最后一个绑定。
        let last = bs[bs.len() - 1];
        assert_eq!(last.name, "out");
        assert_eq!(last.access, BindingAccess::ReadWrite);
        assert_eq!(SceneAbiProfile::V3.out_binding(), last.binding);
        assert_eq!(SceneAbiProfile::V3.out_binding(), V3_OUT);
        assert_eq!(SceneAbiProfile::V3.out_binding(), 10);
        // 除输出外全部只读。
        for b in &bs[..bs.len() - 1] {
            assert_eq!(b.access, BindingAccess::ReadOnly, "{} 必须只读", b.name);
        }
        // 绑定编号常量与表逐格对应（宿主据此声明 `@binding(k)`）。
        assert_eq!(
            [
                V3_PARAMS, V3_SPHERES, V3_LIGHTS, V3_TRIANGLES, V3_NODES, V3_TEXTURES, V3_TEXMETA,
                V3_CDF, V3_INSTANCES, V3_INSTANCE_NODES, V3_OUT
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    /// V3 参数块：`V2 前缀逐字节不变`，尾部只多四字。
    ///
    /// 追加顺序固定：W1-B 加 `P_NUM_TEXTURES` / `P_CFD_ROWS`（+2），W3 加
    /// `P_NUM_INSTANCES` / `P_NUM_INST_NODES`（+2）——**只往后加**，既有下标一个
    /// 不动，故 V2 的 24 字前缀搬过来逐字节相同。
    #[test]
    fn v3_params_words_and_cdf_len() {
        assert_eq!(SceneAbiProfile::V3.params_words(), PARAMS_WORDS_V3);
        assert_eq!(PARAMS_WORDS_V3, PARAMS_WORDS_V2 + 4);
        assert_eq!(P_NUM_TEXTURES, PARAMS_WORDS_V2 as u32);
        assert_eq!(P_CFD_ROWS, P_NUM_TEXTURES + 1);
        assert_eq!(SceneAbiProfile::V3.name(), "V3");
        // CDF 长度口径与 prism-rt 侧同式：2048×1024 → 8,392,704 B。
        assert_eq!(cdf_len_bytes(2048, 1024), 8_392_704);
        // texmeta 步长必须是 16 的倍数（`array<vec4<u32>>` 的 stride 要求）。
        assert_eq!(TEXMETA_STRIDE_BYTES % 16, 0);
        assert_eq!(TEXMETA_STRIDE_BYTES, 32);
        // 四张贴图句柄。
        assert_eq!(TEXMAP_SLOTS, 4);
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

    // -----------------------------------------------------------------
    // W1-B：encode_v3（纹理 / texmeta / cdf）
    // -----------------------------------------------------------------

    /// 一张 4×4 的线性纹理（含完整 mip 链：16+4+1 = 21 纹素）。
    fn tex_upload<'a>(texels: &'a [[f32; 4]]) -> TextureUpload<'a> {
        TextureUpload { w: 4, h: 4, levels: 3, fmt: 1, srgb: 0, texels }
    }

    #[test]
    fn v3_texel_count_matches_mip_chain_formula() {
        let t = tex_upload(&[]);
        // 4×4 → 2×2 → 1×1 = 21
        assert_eq!(t.expected_texels(), 21);
        // 非等比：8×2 → 4×1 → 2×1 → 1×1 = 16+4+2+1 = 23
        let t2 = TextureUpload { w: 8, h: 2, levels: 4, fmt: 1, srgb: 0, texels: &[] };
        assert_eq!(t2.expected_texels(), 23);
        // 与 prism-rt 的口径对拍（同式两份，各写一遍的最容易漂）。
        assert_eq!(t.expected_texels(), 21);
    }

    #[test]
    fn v3_empty_textures_keep_v2_prefix_byte_identical() {
        let req = sample_request();
        let v2 = SceneAbi::encode_v2(&req);
        let v3 = SceneAbi::encode_v3(&req, &[], None).expect("空纹理集必须合法");
        // 24 字前缀逐字节相同（V2 不回归）。
        assert_eq!(&v3.params[..PARAMS_WORDS_V2 * 4], &v2.0[..]);
        // 球块：**每球前 32 B** 与 V2 逐字节相同（V3 只在尾部加 4 个句柄）。
        // 逐球对拍而不是整块对拍——整块长度不同是设计使然（8 → 12 槽）。
        assert_eq!(v3.spheres.len(), v2.1.len() / 8 * SPHERE_SLOTS_V3 as usize);
        for (i, chunk) in v3.spheres.chunks(SPHERE_SLOTS_V3 as usize * 4).enumerate() {
            let v2_chunk = &v2.1[i * SPHERE_SLOTS_V2 as usize * 4..(i + 1) * SPHERE_SLOTS_V2 as usize * 4];
            assert_eq!(
                &chunk[..SPHERE_SLOTS_V2 as usize * 4],
                v2_chunk,
                "第 {i} 个球的前 32 B 必须与 V2 球块逐字节相同"
            );
            assert_eq!(
                &chunk[SPHERE_SLOTS_V2 as usize * 4..],
                &[0u8; 16],
                "无贴图球的四个句柄槽（8..11）必须恒为 0"
            );
        }
        assert_eq!(v3.lights, v2.2);
        assert_eq!(v3.triangles, v2.3);
        assert_eq!(v3.nodes, v2.4);
        // 三块新缓冲全空、两计数为 0。
        assert!(v3.textures.is_empty() && v3.texmeta.is_empty() && v3.cdf.is_empty());
        assert_eq!(v3.num_textures(), 0);
        assert_eq!(v3.cdf_rows(), 0);
        // 尾部四字（W1-B 的两字 + W3 的两字）必须显式写 0
        // （否则是未初始化字节，跨后端读到垃圾）。
        assert_eq!(&v3.params[P_NUM_TEXTURES as usize * 4..], &[0u8; 16]);
        assert_eq!(v3.num_instances(), 0);
        assert_eq!(v3.num_instance_nodes(), 0);
    }

    /// **V3 球块 = V2 球块 + 四个句柄**（W1-C 段 1 的逐格对账）。
    ///
    /// 三件事一起钉：
    /// 1. 步长 12 槽 / 48 B（不是 13、不是 8）；
    /// 2. 槽 0..7 与 V2 **逐字节**相同（V2 前缀不回归）；
    /// 3. 槽 8..11 = `maps[0..4]` 的 **u32 原值**，按下标逐格对应。
    #[test]
    fn v3_sphere_block_is_v2_prefix_plus_four_u32_handles() {
        let mut req = sample_request();
        req.spheres[0].maps = [1, 2, 3, 4];
        let v3 = encode_spheres_v3(&req.spheres);
        let v2 = encode_spheres(&req.spheres);
        assert_eq!(v3.len(), SPHERE_SLOTS_V3 as usize * 4);
        assert_eq!(v3.len(), 48);
        assert_eq!(v2.len(), SPHERE_SLOTS_V2 as usize * 4);
        assert_eq!(v2.len(), 32);
        assert_eq!(&v3[..32], &v2[..], "前 32 B 必须与 V2 逐字节相同");
        // 逐格：槽 8+k 是 maps[k] 的 u32 原值（不是 f32 位型）。
        for k in 0..SPHERE_MAP_SLOTS {
            let off = (SPHERE_MAP_SLOT_BASE as usize + k) * 4;
            let got = u32::from_le_bytes([v3[off], v3[off + 1], v3[off + 2], v3[off + 3]]);
            assert_eq!(got, req.spheres[0].maps[k], "槽 {} 必须是 maps[{k}]", SPHERE_MAP_SLOT_BASE + k as u32);
        }
        // 句柄当 f32 读会是指数下溢的垃圾值——顺带钉住"这里不能 bitcast"。
        let off = SPHERE_MAP_SLOT_BASE as usize * 4;
        let as_f32 = f32::from_le_bytes([v3[off], v3[off + 1], v3[off + 2], v3[off + 3]]);
        assert_eq!(as_f32.to_bits(), 1, "句柄 1 的位型必须原样是 u32 1（不是 f32 1.0）");
        // 两个球的块互不串位：第 2 个球的句柄落在 48 + 8 起。
        req.spheres.push(SphereReq { maps: [5, 6, 7, 8], ..req.spheres[0].clone() });
        let two = encode_spheres_v3(&req.spheres);
        assert_eq!(two.len(), 2 * 48);
        let off2 = 48 + SPHERE_MAP_SLOT_BASE as usize * 4;
        let h5 = u32::from_le_bytes([two[off2], two[off2 + 1], two[off2 + 2], two[off2 + 3]]);
        assert_eq!(h5, 5, "第 2 个球的 MAP_BASE 句柄不能读成第 1 个球的字段");
    }

    /// **句柄槽序 = `Bsdf::maps` 的槽序 = WGSL 的 `MAP_*`**（逐格对账）。
    ///
    /// 三份常量各有各的写法（`prism_rt::bsdf::MAP_*` 是 `usize` 且本 crate
    /// 不能 import 它；WGSL 里是 `u32` 字面量），所以这条测试**从 WGSL 源里
    /// 抠出四个数字**，再与 ABI 的槽位逐格比——不是"看起来一样就算"。
    ///
    /// 顺序错了的后果：球拿 base map 的句柄去做法线贴图，图像上表现为
    /// "颜色对了但光照全歪"，最难在真机上定位的一类。
    #[test]
    fn v3_map_slot_order_matches_bsdf_and_wgsl_cell_by_cell() {
        // ---- ABI 侧：句柄区起点 + 槽数 ----
        assert_eq!(SPHERE_MAP_SLOT_BASE, 8);
        assert_eq!(SPHERE_MAP_SLOTS, 4);
        assert_eq!(SPHERE_SLOTS_V3, 12);
        // ---- WGSL 侧：常量值逐个抠出来 ----
        let src = crate::shaders::source("scene_frame_v3.wgsl").expect("V3 WGSL 必须已登记");
        let want = [
            ("MAP_BASE", 0u32),
            ("MAP_ROUGH", 1),
            ("MAP_NORMAL", 2),
            ("MAP_EMISSIVE", 3),
        ];
        for (name, value) in want {
            let needle = format!("const {name}: u32 = {value}u;");
            assert!(
                src.contains(&needle),
                "WGSL 必须声明 `{needle}`（槽序即语义，改一个数字就与 Bsdf::maps 错位）"
            );
        }
        // 槽序逐格：ABI 槽 8+k ↔ WGSL 的 MAP_* 第 k 个。
        let map_names = ["MAP_BASE", "MAP_ROUGH", "MAP_NORMAL", "MAP_EMISSIVE"];
        for (k, name) in map_names.iter().enumerate() {
            assert_eq!(
                SPHERE_MAP_SLOT_BASE as usize + k,
                8 + k,
                "{name} 的 ABI 槽位必须是 8+{k}"
            );
        }
        // ---- shader 取数：`8 + slot`，且步长 12 ----
        assert!(
            src.contains("const SPHERE_STRIDE: u32 = 12u;"),
            "WGSL 的 SPHERE_STRIDE 必须等于 SPHERE_SLOTS_V3 = 12"
        );
        assert!(
            src.contains("spheres[i * SPHERE_STRIDE + 8u + slot]"),
            "sphere_map 必须读槽 8+slot（读 7+slot 会读到 pad/albedo）"
        );
    }

    #[test]
    fn v3_encodes_texmeta_and_flattens_texels_with_running_offset() {
        let req = sample_request();
        let px_a: Vec<[f32; 4]> = (0..21).map(|i| [i as f32, 0.5, 0.25, 1.0]).collect();
        let px_b: Vec<[f32; 4]> = (0..21).map(|i| [i as f32 + 100.0, 0.0, 0.0, 1.0]).collect();
        let a = tex_upload(&px_a);
        let b = TextureUpload { w: 4, h: 4, levels: 3, fmt: 1, srgb: 1, texels: &px_b };
        let out = SceneAbi::encode_v3(&req, &[a, b], None).expect("合法");
        assert_eq!(out.num_textures(), 2);
        // 纹素字节 = 21 × 16 × 2
        assert_eq!(out.textures.len(), 21 * 16 * 2);
        // texmeta = 2 × 32 B；第一条 offset 0，第二条 offset 21（**纹素**，非字节）
        assert_eq!(out.texmeta.len(), 64);
        assert_eq!(&out.texmeta[0..16], &[4, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&out.texmeta[16..32], &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            &out.texmeta[32..48],
            &[4, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 21, 0, 0, 0]
        );
        // 第二条的 srgb 标记（槽 5）必须落在 20..24 字节里。
        assert_eq!(&out.texmeta[52..56], &[1, 0, 0, 0]);
        // 纹素按小端 f32 展开：第一条第 3 个纹素的 R = 2.0。
        let off = 2 * 16;
        assert_eq!(&out.textures[off..off + 4], &2.0f32.to_le_bytes());
        // 第二条从第 21×16 字节起。其第 4 个纹素（字节偏移 = 21×16 + 4×4）的 R 源值是
        // 104.0，但 `b.srgb == 1` ⇒ 编码侧把 RGB 钳到 [0,1] 再走 sRGB→线性解码：
        // clamp(104.0) = 1.0、`srgb_decode_f32(1.0) = 1.0`。故存进 `textures` 的是 1.0
        // 而非 104.0/101.0——这正是「sRGB 解码已在上传前生效」的证据（与
        // `cpu_tex::RefTextureTable::new` 同式）。texmeta 槽 5 的 srgb 标记见上方断言。
        let off_b = 21 * 16 + 16;
        assert_eq!(&out.textures[off_b..off_b + 4], &1.0f32.to_le_bytes());
        // inputs() 的绑定顺序 = V3_BINDINGS（8 块输入，末位不是 out）。
        let ins = out.inputs();
        assert_eq!(ins.len(), V3_BINDINGS.len() - 1);
        assert_eq!(ins[5], &out.textures[..]);
        assert_eq!(ins[6], &out.texmeta[..]);
        assert_eq!(ins[7], &out.cdf[..]);
    }

    #[test]
    fn v3_env_cdf_length_is_enforced_before_device_access() {
        let req = sample_request();
        // 正确长度：(2+1)·3·4 = 36 B。
        let good = vec![0u8; 36];
        let out = SceneAbi::encode_v3(&req, &[], Some(EnvCdf { w: 2, h: 3, bytes: &good }))
            .expect("长度正确必须通过");
        assert_eq!(out.cdf_rows(), 3);
        assert_eq!(out.cdf.len(), 36);
        // 差一格（少 4 B）⇒ Err，且消息点明公式。
        let bad = vec![0u8; 32];
        let err = SceneAbi::encode_v3(&req, &[], Some(EnvCdf { w: 2, h: 3, bytes: &bad }))
            .expect_err("长度不符必须被拒");
        assert!(err.contains("36"), "错误消息应给出期望字节数，实际：{err}");
        // 0 尺寸 ⇒ Err（不 panic）。
        let err2 = SceneAbi::encode_v3(&req, &[], Some(EnvCdf { w: 0, h: 3, bytes: &good }))
            .expect_err("0 宽必须被拒");
        assert!(err2.contains("尺寸"), "实际：{err2}");
    }

    #[test]
    fn v3_rejects_malformed_texture_uploads_without_panicking() {
        let req = sample_request();
        // 纹素条数不符（缺 mip 链末级）。
        let short: Vec<[f32; 4]> = vec![[1.0; 4]; 20];
        let err = SceneAbi::encode_v3(&req, &[tex_upload(&short)], None)
            .expect_err("短缺的 mip 链必须被拒");
        assert!(err.contains("21"), "错误消息应给出期望纹素数，实际：{err}");
        // 尺寸 0。
        let zero = TextureUpload { w: 0, h: 4, levels: 1, fmt: 1, srgb: 0, texels: &[] };
        assert!(SceneAbi::encode_v3(&req, &[zero], None).is_err());
        // 级数 0。
        let lv0 = TextureUpload { w: 4, h: 4, levels: 0, fmt: 1, srgb: 0, texels: &short };
        assert!(SceneAbi::encode_v3(&req, &[lv0], None).is_err());
    }

    /// V3 的 WGSL 与 `V3_BINDINGS` **逐格对账**：绑定编号、名字、访问属性、
    /// 参数块字数、纹素元素类型。改一处漏一处就红——这是"三份源同一真源"
    /// 的机械保证（V2 的同款测试见 `shaders::tests`）。
    #[test]
    fn v3_wgsl_bindings_and_types_match_abi() {
        let src = crate::shaders::source("scene_frame_v3.wgsl").expect("V3 WGSL 必须已登记");
        // 每个绑定的编号与名字（声明行形如 `@binding(5) var<storage, read> textures:`）。
        for b in V3_BINDINGS.iter() {
            // ABI 名 → WGSL 变量名。唯一的别名：输出绑定在 ABI 里叫 `out`，
            // 在源里叫 `outbuf`（`out` 是 WGSL 保留字，不能作标识符）。
            let var = if b.name == "out" { "outbuf" } else { b.name };
            let needle = format!("@binding({}) var<storage,", b.binding);
            let line = src
                .lines()
                .find(|l| l.contains(needle.as_str()))
                .unwrap_or_else(|| panic!("V3 WGSL 缺少 @binding({}) 的声明", b.binding));
            assert!(
                line.replace(' ', "").contains(format!(">{var}:").as_str()),
                "@binding({}) 的变量名应为 {var}（ABI 名 {}），实际行：{line}",
                b.binding,
                b.name
            );
            let want_rw = b.access == BindingAccess::ReadWrite;
            assert_eq!(
                line.contains("read_write"),
                want_rw,
                "@binding({}) 的访问属性不符（应 {}）：{line}",
                b.binding,
                if want_rw { "read_write" } else { "read" }
            );
        }
        // 参数块字数必须是 V3 的 `PARAMS_WORDS_V3`（写小了会让尾部计数读越界）。
        assert!(
            src.contains(&format!("const PARAMS_WORDS_: u32 = {}u;", PARAMS_WORDS_V3)),
            "V3 WGSL 的参数块字数必须是 {PARAMS_WORDS_V3}（= PARAMS_WORDS_V3）"
        );
        // 纹素元素类型：`array<vec4<f32>>`，且**禁** `array<vec3<f32>>`。
        // 判据扫**去注释后的代码**：注释里写"禁 array<vec3<f32>>"是说明文字，
        // 不是声明（与 `shaders::check_shader` 的去注释同款理由）。
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            src.contains("var<storage, read>       textures: array<vec4<f32>>"),
            "textures 必须是 array<vec4<f32>>（禁 vec3：std430 步长歧义）"
        );
        assert!(
            !code.contains("array<vec3<f32>>"),
            "V3 WGSL 不得出现 array<vec3<f32>>（D-15：三后端步长不一致）"
        );
        // 采样口径必须与 CPU 同式（四个关键常量/形态）。
        assert!(code.contains("uv.x * wf - 0.5"), "双线性必须用纹素中心映射");
        assert!(code.contains("MAX_ANISO_TAPS"), "必须有 tap 上限常量");
        assert!(
            code.contains("max(s_min, s_max / f32(taps))"),
            "tap 布局必须是并集=长轴"
        );
        assert!(
            code.contains("if (m < 0) { m = m + n; }"),
            "平铺必须是 repeat 取模，不是钳制"
        );
    }
}
