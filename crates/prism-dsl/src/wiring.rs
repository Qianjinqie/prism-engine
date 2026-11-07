//! V3.0 W2：39 项能力接线注册表（唯一事实源）。
//!
//! # 它是什么
//!
//! [`CAPABILITY_ROWS`] 是 `docs/v3.0-触达规格表.md` 39 行的运行时镜像：
//! 每行的 `api_path:line`、`api_params` 全集、目标级、`exempt_ruling` 与文档
//! 逐项对齐。`coverage_gate` / `w2_wiring` 两个静态测试**双向核对**代码表与
//! 文档表，任一侧漂移即判红——规格表给人看，本表给求值链用，不出现两份
//! 各自演化的事实源。
//!
//! # 红线合规
//!
//! - **红线 1（语言层不得成第二事实源）**：本文件不实现任何数学 / 渲染 /
//!   物理 / 音频逻辑，只做「声明 → 参数表 → 既有 API 锚点」的搬运。
//!   [`CapabilityRow::consumer`] 指向 `prism-rt` / `prism-core` / `prism-audio`
//!   等既有 crate 的公开求值器；真正的调用在 `prism-app::w2_eval`。
//! - **红线 3（不取默认、不静默降级）**：一行 `cap` 块必须显式声明该行
//!   `api_params` 全集；漏填 / 未知字段 / 非法值 / 超范围 / 未知编号 /
//!   重复声明一律 `P8xxx` 三段式 + 行号，绝不补缺省。
//! - **红线 4（生成源头注入点）**：`cap` 块的参数进 `Node.params` 与
//!   `PropertyChannel` 绑定，在**渲染前**由 `prism-render::movie::wiring_frame_values`
//!   逐帧解算；本模块不建任何帧后效果节点。`codegen.rs:2810` 的
//!   `post { grain }` S4 帧后节点是 §3.3 的负控样本，不是本波的范式。
//!
//! # W3 注入点（见 `docs/v3.0-W2-实录.md`）
//!
//! 1. [`bind_source`]：把任意参数绑定成 `PropertyChannel` 的唯一入口，
//!    W3 的痕迹维度在此按帧 / 按节点声明即可注入生成源头。
//! 2. [`WiringFrame`] / [`resolve_frame`]：逐帧解算出口，W3 的帧值对拍接此。
//!
//! # 确定性
//!
//! 全部整数 / 位型存储；f32 以 `to_bits()` 保存，不做平台浮点运算；
//! `BTreeMap` 固定序；零线程、零系统时间、零 libm。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::PropertyChannel;

use crate::ast::{CapAst, FieldValue, NamedField};
use crate::codegen::{build_channel, NamedChannel};
use crate::{err_at, DslError};

// ---------------------------------------------------------------------------
// 静态表：39 行
// ---------------------------------------------------------------------------

/// 能力所属的求值族（「每族一个端到端样例」的分组依据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapFamily {
    /// 路径积分 / BSDF / 介质 / 毛发 / 纹理等渲染域。
    Render,
    /// 图元 / 网格 / 曲线 / CSG / 细分 / 位移 / 实例等几何域。
    Geometry,
    /// 音频 / 文本 / 色彩 / 工程化域。
    Engineering,
    /// 时间线（核心项 L3）。
    Timeline,
    /// 后处理（核心项 L3；W3 观感直接依赖）。
    Post,
    /// 光源（核心项 L3）。
    Light,
    /// 相机 / 材质（核心项 L3）。
    CameraMaterial,
    /// 粒子（R5010 销号族）。
    Particles,
    /// 物理仿真（R5010 销号族）。
    Sim,
    /// 合成 / 形变（R5010 销号族）。
    ComposeMorph,
    /// 音频（R5010 销号族）。
    Audio,
    /// G5 平台域（正式裁定「不适用」）。
    Platform,
}

impl CapFamily {
    /// 族标识（规格表 / 报告里的稳定短名）。
    pub fn tag(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Geometry => "geometry",
            Self::Engineering => "engineering",
            Self::Timeline => "timeline",
            Self::Post => "post",
            Self::Light => "light",
            Self::CameraMaterial => "camera_material",
            Self::Particles => "particles",
            Self::Sim => "sim",
            Self::ComposeMorph => "compose_morph",
            Self::Audio => "audio",
            Self::Platform => "platform",
        }
    }
}

/// 规格表的一行（与文档 7+1 列逐项对应）。
#[derive(Debug, Clone, Copy)]
pub struct CapabilityRow {
    /// 编号（R-1…P-6）。
    pub id: &'static str,
    /// 引擎侧公开 API 定义锚点（`file:line`）。
    pub api_path: &'static str,
    /// 该锚点的公开参数全集（结构体 pub 字段 / 枚举变体 / 函数形参）。
    pub api_params: &'static [&'static str],
    /// 目标级（`L2` / `L3` / `不适用`）。
    pub target: &'static str,
    /// 正式裁定文本；仅 P-1…P-6 平台域非空（G5 / D-19）。
    pub exempt: &'static str,
    /// 逐行可查的具体接线裁定编号（W2-R-01 … W2-P-06）。
    pub ruling: &'static str,
    /// 既有求值器锚点（`file:line`）；`w2_eval` 在此调用既有 crate API。
    pub consumer: &'static str,
    /// 求值族。
    pub family: CapFamily,
    /// `api_params` 是枚举变体名（G-3 / G-4 / P-3 / P-6）：DSL 逐变体声明
    /// 0/1 选择标志，且必须恰有一个选中（不取默认）。
    pub enum_variants: bool,
}

macro_rules! row {
    ($id:expr, $path:expr, [$($p:expr),* $(,)?], $target:expr, $exempt:expr,
     $ruling:expr, $consumer:expr, $family:expr) => {
        CapabilityRow {
            id: $id,
            api_path: $path,
            api_params: &[$($p),*],
            target: $target,
            exempt: $exempt,
            ruling: $ruling,
            consumer: $consumer,
            family: $family,
            enum_variants: false,
        }
    };
}

macro_rules! enum_row {
    ($id:expr, $path:expr, [$($p:expr),* $(,)?], $target:expr, $exempt:expr,
     $ruling:expr, $consumer:expr, $family:expr) => {
        CapabilityRow {
            id: $id,
            api_path: $path,
            api_params: &[$($p),*],
            target: $target,
            exempt: $exempt,
            ruling: $ruling,
            consumer: $consumer,
            family: $family,
            enum_variants: true,
        }
    };
}

/// G5 平台域正式裁定（规格表 `exempt_ruling` 列的唯一合法值）。
const G5_RULING: &str = "不适用（G5 平台域，§2.1 / D-19）";

/// 39 行能力接线表。**顺序即规格表顺序**，不得增删、不得改序。
pub const CAPABILITY_ROWS: &[CapabilityRow] = &[
    row!("R-1", "crates/prism-rt/src/integrator.rs:3723",
        ["spp", "max_bounces", "clamp_indirect", "nee_only"], "L3", "",
        "W2-R-01", "crates/prism-render/src/rt_backend.rs:150", CapFamily::Render),
    row!("R-2", "crates/prism-rt/src/bsdf.rs:420",
        ["base_color", "metalness", "roughness", "ior", "transmission", "subsurface",
         "subsurface_radius", "alpha_t", "alpha_b", "rotation", "maps"], "L3", "",
        "W2-R-02", "crates/prism-rt/src/bsdf.rs:420", CapFamily::CameraMaterial),
    row!("R-3", "crates/prism-rt/src/caustics.rs:39",
        ["center", "radius", "ior"], "L2", "",
        "W2-R-03", "crates/prism-rt/src/caustics.rs:39", CapFamily::Render),
    row!("R-4", "crates/prism-rt/src/sss.rs:46",
        ["p", "n", "weight"], "L2", "",
        "W2-R-04", "crates/prism-rt/src/sss.rs:46", CapFamily::Render),
    row!("R-5", "crates/prism-rt/src/hair_scatter.rs:169",
        ["beta_m", "k_azimuth", "eta", "sigma_a", "radius", "w_r", "w_tt", "w_trt",
         "phi_tt_shift"], "L2", "",
        "W2-R-05", "crates/prism-rt/src/hair_scatter.rs:169", CapFamily::Render),
    row!("R-6", "crates/prism-core/src/attach.rs:380",
        ["points", "radii", "params"], "L2", "",
        "W2-R-06", "crates/prism-core/src/attach.rs:380", CapFamily::Geometry),
    row!("R-7", "crates/prism-core/src/attach.rs:53",
        ["bounds_min", "bounds_max", "sigma_t", "albedo", "g", "priority"], "L2", "",
        "W2-R-07", "crates/prism-core/src/attach.rs:53", CapFamily::Render),
    row!("R-8", "crates/prism-rt/src/caustics.rs:50",
        ["wi", "pdf_w", "weight"], "L2", "",
        "W2-R-08", "crates/prism-rt/src/caustics.rs:50", CapFamily::Render),
    row!("R-9", "crates/prism-rt/src/gi/mod.rs:112",
        ["quality", "probe_res", "env_radiance", "min", "max"], "L2", "",
        "W2-R-09", "crates/prism-rt/src/gi/mod.rs:112", CapFamily::Render),
    row!("R-10", "crates/prism-rt/src/env.rs:216",
        ["w", "h", "spots", "p_spot", "cond_bytes", "marg_bytes", "res_cond_bytes",
         "spot_bytes"], "L2", "",
        "W2-R-10", "crates/prism-rt/src/env.rs:216", CapFamily::Render),
    row!("R-11", "crates/prism-core/src/attach.rs:295",
        ["width", "height", "srgb", "rgba8"], "L3", "",
        "W2-R-11", "crates/prism-core/src/attach.rs:295", CapFamily::Render),
    row!("R-12", "crates/prism-rt/src/spectral.rs:26",
        ["degenerate", "lambda_min", "lambda_max", "cauchy_a", "cauchy_b"], "L2", "",
        "W2-R-12", "crates/prism-rt/src/spectral.rs:26", CapFamily::Render),
    row!("R-13", "crates/prism-postfx/src/adaptive.rs:85",
        ["n0", "total", "mantissa_bits", "tile", "rounds"], "L2", "",
        "W2-R-13", "crates/prism-postfx/src/adaptive.rs:85", CapFamily::Post),
    row!("G-1", "crates/prism-rt/src/integrator.rs:885",
        ["spheres", "boxes", "paraboloids", "planes", "meshes", "cylinders", "cones",
         "toruses", "revolutions", "lights", "medium", "curve_bvh", "textures",
         "instances", "instance_geoms", "instance_bvh", "texels", "env_cdf", "media"],
        "L3", "", "W2-G-01", "crates/prism-rt/src/integrator.rs:885", CapFamily::Geometry),
    row!("G-2", "crates/prism-core/src/attach.rs:159",
        ["positions", "indices", "uv", "displacement", "crease", "corners"], "L3", "",
        "W2-G-02", "crates/prism-core/src/attach.rs:159", CapFamily::Geometry),
    enum_row!("G-3", "crates/prism-geometry/src/curve_tess.rs:55",
        ["Cylinder", "Cone", "Torus", "Revolution", "Strand", "TriSoup"], "L2", "",
        "W2-G-03", "crates/prism-geometry/src/curve_tess.rs:55", CapFamily::Geometry),
    enum_row!("G-4", "crates/prism-geometry/src/csg.rs:12",
        ["Sphere", "Box", "Plane", "Cylinder", "Torus", "Capsule", "Union",
         "Intersection", "Subtraction", "SmoothUnion", "SmoothIntersection",
         "SmoothSubtraction", "Translate", "Displace"], "L2", "",
        "W2-G-04", "crates/prism-geometry/src/csg.rs:12", CapFamily::Geometry),
    row!("G-5", "crates/prism-geometry/src/lod.rs:38",
        ["trace", "max_octaves", "falloff", "eps_per_unit"], "L2", "",
        "W2-G-05", "crates/prism-geometry/src/lod.rs:38", CapFamily::Geometry),
    row!("G-6", "crates/prism-geometry/src/gltf.rs:458",
        ["positions", "normals", "uvs", "indices", "material"], "L2", "",
        "W2-G-06", "crates/prism-geometry/src/gltf.rs:458", CapFamily::Geometry),
    row!("G-7", "crates/prism-geometry/src/sdf.rs:75",
        ["t_max", "hit_eps", "max_steps", "safety"], "L2", "",
        "W2-G-07", "crates/prism-geometry/src/sdf.rs:75", CapFamily::Geometry),
    row!("G-8", "crates/prism-core/src/attach.rs:250",
        ["geom_id", "affine", "mat_id"], "L3", "",
        "W2-G-08", "crates/prism-core/src/attach.rs:250", CapFamily::Geometry),
    row!("G-9", "crates/prism-geometry/src/subdiv.rs:68",
        ["positions", "indices", "uv", "normals", "levels"], "L2", "",
        "W2-G-09", "crates/prism-geometry/src/subdiv.rs:68", CapFamily::Geometry),
    row!("G-10", "crates/prism-core/src/attach.rs:119",
        ["texture", "amplitude", "scale", "levels"], "L2", "",
        "W2-G-10", "crates/prism-core/src/attach.rs:119", CapFamily::Geometry),
    row!("E-1", "crates/prism-core/src/eval.rs:136",
        ["graph", "registry", "plane", "sig", "ctx"], "L2", "",
        "W2-E-01", "crates/prism-core/src/eval.rs:136", CapFamily::Engineering),
    row!("E-2", "crates/prism-render/src/tile.rs:55",
        ["width", "height", "tile_w", "tile_h"], "L2", "",
        "W2-E-02", "crates/prism-render/src/tile.rs:55", CapFamily::Engineering),
    row!("E-3", "crates/prism-render/src/farm/merge.rs:49",
        ["frames", "values", "config_hash"], "L2", "",
        "W2-E-03", "crates/prism-render/src/farm/merge.rs:49", CapFamily::Engineering),
    row!("E-4", "crates/prism-core/src/signature.rs:453",
        ["width", "height", "spp_tier", "tile", "engine_version", "frame"], "L2", "",
        "W2-E-04", "crates/prism-core/src/signature.rs:453", CapFamily::Timeline),
    row!("E-5", "crates/prism-audio/src/mixer.rs:15",
        ["buffer", "gain", "fade_in_secs", "fade_out_secs"], "L2", "",
        "W2-E-05", "crates/prism-audio/src/mixer.rs:34", CapFamily::Audio),
    row!("E-6", "crates/prism-text/src/layout.rs:44",
        ["font_size", "letter_spacing", "line_spacing", "max_width", "align"], "L3", "",
        "W2-E-06", "crates/prism-text/src/layout.rs:44", CapFamily::Engineering),
    row!("E-7", "crates/prism-math/src/ocio.rs:667",
        ["name", "family", "bitdepth", "allocation", "isdata", "to_reference",
         "from_reference"], "L3", "",
        "W2-E-07", "crates/prism-math/src/ocio.rs:667", CapFamily::Engineering),
    row!("E-8", "crates/prism-dsl/src/codegen.rs:364",
        ["title", "width", "height", "fps", "duration", "format", "scenes",
         "transitions", "graph", "channels", "packs", "theme", "fonts", "formulas",
         "wiring"],
        "L2", "", "W2-E-08", "crates/prism-dsl/src/codegen.rs:364", CapFamily::Engineering),
    row!("E-9", "crates/prism-infra/src/mcp.rs:33",
        ["name", "description", "input_schema"], "L2", "",
        "W2-E-09", "crates/prism-infra/src/mcp.rs:33", CapFamily::Engineering),
    row!("E-10", "crates/prism-app/src/editor/async_preview.rs:35",
        ["graph", "defs", "quality", "width", "height", "engine_version", "frame"],
        "L2", "", "W2-E-10", "crates/prism-app/src/editor/async_preview.rs:35",
        CapFamily::Engineering),
    row!("P-1", "crates/prism-gpu/src/vulkan/compute.rs:242",
        ["kernel", "entry", "shape", "groups", "invocations", "wasted_invocations",
         "bindings", "buffer_bytes"], "不适用", G5_RULING,
        "W2-P-01", "crates/prism-gpu/src/vulkan/compute.rs:242", CapFamily::Platform),
    row!("P-2", "crates/prism-gpu/src/web/mod.rs:186",
        ["adapter_name", "has_shader_f16", "has_timestamp_query"], "不适用", G5_RULING,
        "W2-P-02", "crates/prism-gpu/src/web/mod.rs:186", CapFamily::Platform),
    enum_row!("P-3", "crates/prism-gpu/src/web/scene_gpu.rs:184",
        ["NotWasmTarget", "EmptyFrame", "TooLarge", "Step", "NoFreeFrameSlot",
         "NotSettledYet"], "不适用", G5_RULING,
        "W2-P-03", "crates/prism-gpu/src/web/scene_gpu.rs:184", CapFamily::Platform),
    row!("P-4", "crates/prism-gpu/src/metal/mod.rs:208",
        ["supported_platform", "device_present", "device_count", "reason"], "不适用",
        G5_RULING, "W2-P-04", "crates/prism-gpu/src/metal/mod.rs:208", CapFamily::Platform),
    row!("P-5", "crates/prism-ffi-hwenc/src/lib.rs:301",
        ["backend", "codec", "status", "reason"], "不适用", G5_RULING,
        "W2-P-05", "crates/prism-ffi-hwenc/src/lib.rs:301", CapFamily::Platform),
    enum_row!("P-6", "crates/prism-gpu/src/web/mod.rs:154",
        ["Available", "NoNavigatorGpu", "NotWasmTarget"], "不适用", G5_RULING,
        "W2-P-06", "crates/prism-gpu/src/web/mod.rs:154", CapFamily::Platform),
];

/// 按编号取行。
pub fn row_by_id(id: &str) -> Option<&'static CapabilityRow> {
    CAPABILITY_ROWS.iter().find(|r| r.id == id)
}

/// 规格表 39 个编号全集。
pub fn expected_ids() -> BTreeSet<String> {
    CAPABILITY_ROWS.iter().map(|r| r.id.to_string()).collect()
}

/// 平台域（G5 裁定）行集合。
pub fn platform_ids() -> BTreeSet<String> {
    CAPABILITY_ROWS
        .iter()
        .filter(|r| r.family == CapFamily::Platform)
        .map(|r| r.id.to_string())
        .collect()
}

/// 39 行参数全集总数：W1 实录 225；W2 因 E-8 新增 `wiring` 装配字段（引擎侧真实扩容）变为 226。
pub fn total_param_count() -> usize {
    CAPABILITY_ROWS.iter().map(|r| r.api_params.len()).sum()
}

// ---------------------------------------------------------------------------
// 值域规则（红线 3：超范围必报错，不取默认）
// ---------------------------------------------------------------------------

/// 一条具体的数值作用域规则（`lo..=hi`，闭区间）。
#[derive(Debug, Clone, Copy)]
pub struct RangeRule {
    pub row: &'static str,
    pub param: &'static str,
    pub lo: f64,
    pub hi: f64,
}

/// 显式值域规则表。每条都对应规格表里一个真实参数，由静态门禁回读核对。
///
/// 未列出的参数仍受通用校验约束（数值必须有限、文本必须非空、枚举变体行
/// 必须恰选中一个），但**核心项与七族的关键参数**在此给出可断言的硬边界，
/// 供负控注入「故意写超范围值 ⇒ FAIL 带行号」使用。
pub const RANGE_RULES: &[RangeRule] = &[
    RangeRule { row: "R-1", param: "spp", lo: 1.0, hi: 4096.0 },
    RangeRule { row: "R-1", param: "max_bounces", lo: 1.0, hi: 64.0 },
    RangeRule { row: "R-1", param: "clamp_indirect", lo: 0.0, hi: 1.0e6 },
    RangeRule { row: "R-2", param: "metalness", lo: 0.0, hi: 1.0 },
    RangeRule { row: "R-2", param: "roughness", lo: 0.0, hi: 1.0 },
    RangeRule { row: "R-2", param: "transmission", lo: 0.0, hi: 1.0 },
    RangeRule { row: "R-2", param: "ior", lo: 1.0, hi: 4.0 },
    RangeRule { row: "R-7", param: "sigma_t", lo: 0.0, hi: 100.0 },
    RangeRule { row: "R-7", param: "albedo", lo: 0.0, hi: 1.0 },
    RangeRule { row: "R-7", param: "g", lo: -1.0, hi: 1.0 },
    RangeRule { row: "R-11", param: "width", lo: 1.0, hi: 16384.0 },
    RangeRule { row: "R-11", param: "height", lo: 1.0, hi: 16384.0 },
    RangeRule { row: "R-13", param: "n0", lo: 1.0, hi: 65536.0 },
    RangeRule { row: "R-13", param: "rounds", lo: 1.0, hi: 64.0 },
    RangeRule { row: "G-5", param: "max_octaves", lo: 1.0, hi: 16.0 },
    RangeRule { row: "G-7", param: "max_steps", lo: 1.0, hi: 4096.0 },
    RangeRule { row: "G-9", param: "levels", lo: 0.0, hi: 8.0 },
    RangeRule { row: "G-10", param: "amplitude", lo: 0.0, hi: 100.0 },
    RangeRule { row: "E-2", param: "tile_w", lo: 1.0, hi: 4096.0 },
    RangeRule { row: "E-2", param: "tile_h", lo: 1.0, hi: 4096.0 },
    RangeRule { row: "E-4", param: "spp_tier", lo: 0.0, hi: 16.0 },
    RangeRule { row: "E-5", param: "gain", lo: 0.0, hi: 1.0 },
    RangeRule { row: "E-5", param: "fade_in_secs", lo: 0.0, hi: 3600.0 },
    RangeRule { row: "E-6", param: "font_size", lo: 1.0, hi: 1024.0 },
    RangeRule { row: "E-8", param: "fps", lo: 1.0, hi: 240.0 },
];

/// 取某行某参数的作用域规则。
pub fn range_rule(row_id: &str, param: &str) -> Option<&'static RangeRule> {
    RANGE_RULES.iter().find(|r| r.row == row_id && r.param == param)
}

/// 整数参数（写小数即非法，防止「取默认」式的静默截断）。
const INTEGER_PARAMS: &[&str] = &[
    "spp", "max_bounces", "w", "h", "width", "height", "n0", "total", "mantissa_bits",
    "tile", "rounds", "probe_res", "levels", "count", "fps", "frame", "spp_tier",
    "max_octaves", "max_steps", "priority", "font_size", "tile_w", "tile_h",
    "device_count", "groups", "invocations", "wasted_invocations", "segments",
];

/// 布尔参数（只接受 `true` / `false` 裸标识符）。
const BOOL_PARAMS: &[&str] = &[
    "nee_only", "srgb", "degenerate", "isdata", "has_shader_f16",
    "has_timestamp_query", "device_present", "supported_platform", "filmic",
];

/// 文本 / 句柄 / 数组类参数（必须写字符串，不得写裸数字）。
const TEXT_PARAMS: &[&str] = &[
    "maps", "points", "radii", "params", "bounds_min", "bounds_max", "wi", "pdf_w",
    "weight", "p", "n", "center", "env_radiance", "min", "max", "spots", "p_spot",
    "cond_bytes", "marg_bytes", "res_cond_bytes", "spot_bytes", "rgba8", "quality",
    "spheres", "boxes", "paraboloids", "planes", "meshes", "cylinders", "cones",
    "toruses", "revolutions", "lights", "medium", "curve_bvh", "textures", "instances",
    "instance_geoms", "instance_bvh", "texels", "env_cdf", "media", "positions",
    "indices", "uv", "uvs", "normals", "displacement", "crease", "corners", "trace",
    "material", "geom_id", "affine", "mat_id", "texture", "graph", "registry", "plane",
    "sig", "ctx", "frames", "values", "config_hash", "engine_version", "buffer",
    "name", "family", "allocation", "to_reference", "from_reference", "title",
    "format", "scenes", "transitions", "channels", "packs", "theme", "fonts",
    "formulas", "description", "input_schema", "defs", "align", "kernel", "entry",
    "shape", "bindings", "buffer_bytes", "adapter_name", "reason", "backend",
    "codec", "status", "bitdepth",
];

/// 参数的值形态（供校验与诊断提示用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// 非负整数（写小数即 P8003）。
    Int,
    /// 有限数值（可带小数）。
    Num,
    /// `true` / `false`。
    Bool,
    /// 非空字符串（句柄 / 路径 / 数组编码 / 枚举选择器）。
    Text,
    /// 枚举变体选择标志（0 / 1）。
    Variant,
}

/// 判定参数形态：`api_params` 全集逐项有确定的期望形态，不靠「猜」。
pub fn param_kind(row: &CapabilityRow, param: &str) -> ParamKind {
    if row.enum_variants {
        return ParamKind::Variant;
    }
    if BOOL_PARAMS.contains(&param) {
        return ParamKind::Bool;
    }
    if INTEGER_PARAMS.contains(&param) {
        return ParamKind::Int;
    }
    if TEXT_PARAMS.contains(&param) {
        return ParamKind::Text;
    }
    ParamKind::Num
}

// ---------------------------------------------------------------------------
// 编译产物：WiringPlan
// ---------------------------------------------------------------------------

/// 确定性标量值（f32 以位型保存，杜绝平台浮点差异）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringValue {
    /// 整数。
    Int(i64),
    /// 有限 f32（位型）。
    Num(u32),
    /// 布尔。
    Bool(bool),
    /// 非空文本 / 句柄。
    Text(String),
    /// vec3（三个 f32 位型）。
    Vec3([u32; 3]),
}

impl WiringValue {
    /// 数值形态的 f64 视图（仅用于值域校验，不进产物）。
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Num(bits) => Some(f32::from_bits(*bits) as f64),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Self::Vec3(_) | Self::Text(_) => None,
        }
    }

    /// 规范字节序列（同源对拍 / 两跑逐字节的粒度）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(16);
        match self {
            Self::Int(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Self::Num(bits) => {
                out.push(2);
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Self::Bool(b) => {
                out.push(3);
                out.push(if *b { 1 } else { 0 });
            }
            Self::Text(s) => {
                out.push(4);
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Self::Vec3(v) => {
                out.push(5);
                for b in v {
                    out.extend_from_slice(&b.to_le_bytes());
                }
            }
        }
        out
    }
}

/// 一个参数：静态常量或时序绑定（L3 统一通道）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringParam {
    /// 静态值。
    Const(WiringValue),
    /// 绑定到命名 `PropertyChannel`（逐帧解算；「任意参数随时序变」）。
    Channel(String),
}

/// 一个 `cap` 块的编译产物。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WiringEntry {
    /// 能力编号。
    pub row_id: String,
    /// 声明行号（三段式报错 / 报告定位）。
    pub line: usize,
    /// 场景内声明时的场景下标（顶层为 `None`）。
    pub scene: Option<usize>,
    /// 参数全集（键集合 ≡ 该行 `api_params`）。
    pub params: BTreeMap<String, WiringParam>,
}

/// 一次编译的全部接线声明。**旧文件不含 `cap` 块 ⇒ 空 ⇒ 逐字节零损失**。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WiringPlan {
    /// 按声明序（同编号重复已被 `build_wiring` 判红）。
    pub entries: Vec<WiringEntry>,
}

impl WiringPlan {
    /// 是否为空（零损失判据：旧文件恒空）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按编号取条目。
    pub fn entry(&self, row_id: &str) -> Option<&WiringEntry> {
        self.entries.iter().find(|e| e.row_id == row_id)
    }

    /// 已接线的参数总数：33 个非平台域行全覆盖时为 198；运行时表 39 行分母 226。
    pub fn param_count(&self) -> usize {
        self.entries.iter().map(|e| e.params.len()).sum()
    }

    /// 已接线的非平台域行数（目标级 L2/L3 的行）。
    pub fn wired_rows(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| row_by_id(&e.row_id).map(|r| r.family != CapFamily::Platform).unwrap_or(false))
            .count()
    }

    /// 规范字节序列：同源对拍、两跑逐字节、换序反例的统一粒度。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for e in &self.entries {
            out.extend_from_slice(&(e.row_id.len() as u64).to_le_bytes());
            out.extend_from_slice(e.row_id.as_bytes());
            // `e.line` 是诊断元数据（声明行号），不进产物：同源对拍的
            // 「直接构造同一场景」一路没有源码行号，语义相同即应逐字节相同。
            out.push(match e.scene {
                None => 0,
                Some(_) => 1,
            });
            if let Some(si) = e.scene {
                out.extend_from_slice(&(si as u64).to_le_bytes());
            }
            out.extend_from_slice(&(e.params.len() as u64).to_le_bytes());
            for (k, v) in &e.params {
                out.extend_from_slice(&(k.len() as u64).to_le_bytes());
                out.extend_from_slice(k.as_bytes());
                match v {
                    WiringParam::Const(val) => {
                        out.push(1);
                        out.extend_from_slice(&val.canonical_bytes());
                    }
                    WiringParam::Channel(name) => {
                        out.push(2);
                        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
                        out.extend_from_slice(name.as_bytes());
                    }
                }
            }
        }
        out
    }
}

/// 一帧的接线解算结果（`row.param → f32`），L3 时序对拍的执行体。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WiringFrame {
    /// 逐参数值（BTreeMap 固定序；数值以位型保存）。
    pub values: BTreeMap<String, u32>,
}

impl WiringFrame {
    /// 取某参数的 f32 值（位型还原）。
    pub fn value(&self, key: &str) -> Option<f32> {
        self.values.get(key).copied().map(f32::from_bits)
    }

    /// 规范字节序列。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(self.values.len() as u64).to_le_bytes());
        for (k, v) in &self.values {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// 按帧解算一份 `WiringPlan`（生成源头求值：渲染前调用，不是帧后节点）。
///
/// `channels` 是 `CompileV2Output::channels` 的名字 → 通道表；静态常量直接
/// 转位型，时序绑定走既有 `PropertyChannel::sample_ctx`（与
/// `prism_core::sample_binding` 同一致求入口，不另造求值器）。
///
/// 这是 W3 的**生成源头注入点之一**：痕迹维度只要把参数声明为
/// `WiringParam::Channel`，就会在每一帧、按节点 `EvalCtx` 在出帧前解算。
pub fn resolve_frame(
    plan: &WiringPlan,
    channels: &BTreeMap<&str, &PropertyChannel>,
    t: f32,
    frame: u64,
    global_seed: u64,
    node_id: u64,
) -> WiringFrame {
    use prism_core::EvalCtx;
    let ctx = EvalCtx::new(global_seed, node_id, frame);
    let mut out = WiringFrame::default();
    for e in &plan.entries {
        for (param, binding) in &e.params {
            let key = format!("{}.{}", e.row_id, param);
            let bits: u32 = match binding {
                WiringParam::Const(v) => match v {
                    WiringValue::Int(i) => (*i as f32).to_bits(),
                    WiringValue::Num(b) => *b,
                    WiringValue::Bool(b) => (if *b { 1.0_f32 } else { 0.0_f32 }).to_bits(),
                    WiringValue::Text(s) => (s.len() as f32).to_bits(),
                    WiringValue::Vec3(v) => f32::from_bits(v[0]).to_bits(),
                },
                WiringParam::Channel(name) => channels
                    .get(name.as_str())
                    .map(|ch| ch.sample_ctx(t, &ctx).to_bits())
                    .unwrap_or_else(|| f32::NAN.to_bits()),
            };
            out.values.insert(key, bits);
        }
    }
    out
}

/// 把关键帧 / 解析式绑定登记为 `PropertyChannel` 的**唯一入口**。
///
/// W3 的痕迹维度按帧 / 按节点声明时走这里：函数只做「AST 值 → 既有
/// `prism_anim` 通道」的搬运，不实现任何求值数学（红线 1）。
pub(crate) fn bind_source(
    name: &str,
    value: &FieldValue,
    fps: u32,
    desc: &str,
    line: usize,
) -> Result<PropertyChannel, DslError> {
    match value {
        FieldValue::Keys(keys) => build_channel(keys, None, fps, desc, line),
        FieldValue::Block(fields) => {
            let expr_field: Option<&NamedField> = fields.iter().find(|f| f.name == "expr");
            let expr_field: &NamedField = expr_field.ok_or_else(|| {
                wiring_err(
                    line,
                    "P8008",
                    &format!("通道 '{name}'（{desc}）的时序子块缺少 'expr' 字段"),
                    "写成 { key 0s 0 linear key 1s 1 linear } 或 { expr \"t * 2 + 1\" }",
                )
            })?;
            let src: &str = match expr_field.value.as_ref() {
                Some(FieldValue::Str(s)) => s.as_str(),
                _ => {
                    return Err(wiring_err(
                        expr_field.line,
                        "P8008",
                        &format!("{desc} 的 expr 需要字符串解析式"),
                        "写成 { expr \"t * 2 + 1\" }，解析式支持 t / + - * / 与括号",
                    ));
                }
            };
            let expr = prism_anim::Expr::parse(src).map_err(|e| {
                wiring_err(
                    expr_field.line,
                    "P8008",
                    &format!("{desc} 的解析式 '{src}' 非法（{e:?}）"),
                    "只用 t 与 + - * / ( )，例如 \"t * 2 + 1\"",
                )
            })?;
            Ok(PropertyChannel::Expr(expr))
        }
        _ => Err(wiring_err(
            line,
            "P8008",
            &format!("{desc} 的时序值必须是关键帧块或解析式子块"),
            "写成 { key 0s 0 linear key 1s 1 linear } 或 { expr \"t * 2 + 1\" }",
        )),
    }
}

// ---------------------------------------------------------------------------
// 编译期校验 + 构造（红线 3）
// ---------------------------------------------------------------------------

/// 三段式错误（P8xxx 码段，W2 新增；与 v1/v2 的 P1xxx–P4xxx、v3 的
/// P6xxx/P7xxx 互不冲突）。
fn wiring_err(line: usize, code: &str, what: &str, fix: &str) -> DslError {
    err_at(line, format!("[{code}] 第 {line} 行：{what}\n  建议：{fix}"))
}

/// 把 `FieldValue` 转成确定性 `WiringValue`；形态不符即三段式报错。
fn scalar_value(
    row: &CapabilityRow,
    param: &str,
    field: &NamedField,
) -> Result<WiringValue, DslError> {
    let kind = param_kind(row, param);
    let value: &FieldValue = field.value.as_ref().ok_or_else(|| {
        wiring_err(
            field.line,
            "P8003",
            &format!("cap \"{}\" 的参数 '{}' 缺少值", row.id, param),
            &format!("按 {} 形态显式给出 '{}' 的值", kind_name(kind), param),
        )
    })?;
    let bad = |what: &str| -> DslError {
        wiring_err(
            field.line,
            "P8003",
            &format!(
                "cap \"{}\" 的参数 '{}' 期望 {}，却写了{}",
                row.id, param, kind_name(kind), what
            ),
            &format!("把 '{}' 改写成 {} 形态；参数全集见规格表 {} 行", param, kind_name(kind), row.id),
        )
    };
    match kind {
        ParamKind::Bool => match value {
            FieldValue::Ident(s) if s == "true" => Ok(WiringValue::Bool(true)),
            FieldValue::Ident(s) if s == "false" => Ok(WiringValue::Bool(false)),
            _ => Err(bad("非 true/false 的值")),
        },
        ParamKind::Int | ParamKind::Variant => {
            let n: f32 = match value {
                FieldValue::Num(v) => *v,
                FieldValue::Percent(v) => *v,
                FieldValue::Px(v) => *v,
                _ => return Err(bad("非整数值")),
            };
            if !n.is_finite() || n.fract() != 0.0 || n < 0.0 {
                return Err(bad("非有限或非负整数"));
            }
            if kind == ParamKind::Variant && n > 1.0 {
                return Err(bad("枚举选择标志只能是 0 或 1"));
            }
            Ok(WiringValue::Int(n as i64))
        }
        ParamKind::Num => {
            let n: f32 = match value {
                FieldValue::Num(v) => *v,
                FieldValue::Percent(v) => *v / 100.0,
                FieldValue::Px(v) => *v,
                FieldValue::Angle(a) => a.value,
                FieldValue::Time(tv) => tv.value,
                _ => return Err(bad("非数值")),
            };
            if !n.is_finite() {
                return Err(bad("非有限数值"));
            }
            Ok(WiringValue::Num(n.to_bits()))
        }
        ParamKind::Text => match value {
            FieldValue::Str(s) if !s.is_empty() => Ok(WiringValue::Text(s.clone())),
            FieldValue::Vec3(v) => Ok(WiringValue::Vec3([
                v.x.to_bits(),
                v.y.to_bits(),
                v.z.to_bits(),
            ])),
            FieldValue::Color(r, g, b) => Ok(WiringValue::Text(format!("#{r:02x}{g:02x}{b:02x}"))),
            FieldValue::Ident(s) if !s.is_empty() => Ok(WiringValue::Text(s.clone())),
            _ => Err(bad("空字符串或非法形态（文本参数需要字符串 / vec3 / 颜色）")),
        },
    }
}

fn kind_name(kind: ParamKind) -> &'static str {
    match kind {
        ParamKind::Int => "非负整数",
        ParamKind::Num => "有限数值",
        ParamKind::Bool => "true/false",
        ParamKind::Text => "非空字符串",
        ParamKind::Variant => "0/1 变体选择标志",
    }
}

/// 校验值域（显式 `RANGE_RULES` + 通用有限性）。
fn check_range(row: &CapabilityRow, param: &str, v: &WiringValue, line: usize) -> Result<(), DslError> {
    let Some(rule) = range_rule(row.id, param) else {
        return Ok(());
    };
    let Some(n) = v.as_f64() else {
        return Err(wiring_err(
            line,
            "P8004",
            &format!("cap \"{}\" 的参数 '{}' 有值域 [{}, {}]，却不是数值", row.id, param, rule.lo, rule.hi),
            &format!("把 '{}' 写成 [{}, {}] 内的数值", param, rule.lo, rule.hi),
        ));
    };
    if !(rule.lo..=rule.hi).contains(&n) {
        return Err(wiring_err(
            line,
            "P8004",
            &format!(
                "cap \"{}\" 的参数 '{}' = {} 超出值域 [{}, {}]",
                row.id, param, n, rule.lo, rule.hi
            ),
            &format!("把 '{}' 改到 [{}, {}] 内；超范围不取默认、不钳制", param, rule.lo, rule.hi),
        ));
    }
    Ok(())
}

/// 把 DSL 的 `cap` 块序列编译成 `WiringPlan` + 新增命名通道。
///
/// 判红集合（红线 3 / §1.3 负控）：
/// - `P8001` 漏必填（列出缺项）；
/// - `P8002` 未知字段；
/// - `P8003` 值形态 / 类型错；
/// - `P8004` 超范围；
/// - `P8005` 未知能力编号；
/// - `P8006` 同编号重复声明；
/// - `P8007` 平台域（G5 裁定行）被声明；
/// - `P8008` 时序绑定非法；
/// - `P8009` 枚举变体行未恰选中一个。
///
/// 全部带行号 + 三段式 + 可行动建议。
pub(crate) fn build_wiring(
    caps: &[CapAst],
    fps: u32,
) -> Result<(WiringPlan, Vec<NamedChannel>), DslError> {
    let mut plan = WiringPlan::default();
    let mut channels: Vec<NamedChannel> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for cap in caps {
        let row: &CapabilityRow = row_by_id(&cap.row_id).ok_or_else(|| {
            wiring_err(
                cap.line,
                "P8005",
                &format!("未知能力编号 '{}'", cap.row_id),
                "编号取规格表 39 行之一（R-1…R-13 / G-1…G-10 / E-1…E-10 / P-1…P-6）",
            )
        })?;
        if row.family == CapFamily::Platform {
            return Err(wiring_err(
                cap.line,
                "P8007",
                &format!("能力 {} 属 G5 平台域（{}），语言层不暴露", row.id, row.exempt),
                "删除该 cap 块；平台后端由 render 命令与 D-19 双轨决定",
            ));
        }
        if !seen.insert(row.id.to_string()) {
            return Err(wiring_err(
                cap.line,
                "P8006",
                &format!("能力 {} 重复声明", row.id),
                "同一编号只声明一个 cap 块；多组参数合并到同一块内",
            ));
        }

        let mut params: BTreeMap<String, WiringParam> = BTreeMap::new();
        let mut declared: BTreeSet<&str> = BTreeSet::new();
        for f in &cap.fields {
            if !row.api_params.contains(&f.name.as_str()) {
                return Err(wiring_err(
                    f.line,
                    "P8002",
                    &format!("cap \"{}\" 出现未知字段 '{}'", row.id, f.name),
                    &format!(
                        "该行只允许参数全集：{}；删掉 '{}' 或改到对应行",
                        row.api_params.join(" / "),
                        f.name
                    ),
                ));
            }
            if !declared.insert(f.name.as_str()) {
                return Err(wiring_err(
                    f.line,
                    "P8006",
                    &format!("cap \"{}\" 的字段 '{}' 重复声明", row.id, f.name),
                    "每个参数只声明一次",
                ));
            }
            let binding: WiringParam = match f.value.as_ref() {
                Some(FieldValue::Keys(_)) | Some(FieldValue::Block(_)) => {
                    let ch_name: String = format!("w2/{}/{}/{}", row.id, cap.scene_index.unwrap_or(0), f.name);
                    let desc: String = format!("cap \"{}\" 的时序参数 '{}'", row.id, f.name);
                    let channel: PropertyChannel =
                        bind_source(&ch_name, f.value.as_ref().expect("值已匹配"), fps, &desc, f.line)?;
                    channels.push(NamedChannel { name: ch_name.clone(), channel });
                    WiringParam::Channel(ch_name)
                }
                _ => {
                    let v: WiringValue = scalar_value(row, &f.name, f)?;
                    check_range(row, &f.name, &v, f.line)?;
                    WiringParam::Const(v)
                }
            };
            params.insert(f.name.clone(), binding);
        }

        // 全集必填：差集非空即判红，列出缺项（不取默认、不静默补值）。
        let missing: Vec<&str> = row
            .api_params
            .iter()
            .copied()
            .filter(|p| !params.contains_key(*p))
            .collect();
        if !missing.is_empty() {
            return Err(wiring_err(
                cap.line,
                "P8001",
                &format!(
                    "cap \"{}\" 漏填 {} 个必填参数：{}",
                    row.id,
                    missing.len(),
                    missing.join(" / ")
                ),
                &format!(
                    "按规格表 {} 行的参数全集逐项显式声明：{}",
                    row.id,
                    row.api_params.join(" / ")
                ),
            ));
        }

        // 枚举变体行：必须恰选中一个（不取默认、不允许全 0）。
        if row.enum_variants {
            let selected: usize = row
                .api_params
                .iter()
                .filter(|p| matches!(params.get(**p), Some(WiringParam::Const(WiringValue::Int(1)))))
                .count();
            if selected != 1 {
                return Err(wiring_err(
                    cap.line,
                    "P8009",
                    &format!(
                        "cap \"{}\" 是枚举变体行，应恰有一个变体写 1，实际 {selected} 个",
                        row.id
                    ),
                    &format!("把选中的变体写 1、其余写 0，例如 {} 1", row.api_params[0]),
                ));
            }
        }

        plan.entries.push(WiringEntry {
            row_id: row.id.to_string(),
            line: cap.line,
            scene: cap.scene_index,
            params,
        });
    }
    Ok((plan, channels))
}

// ---------------------------------------------------------------------------
// 样例源生成（每族端到端样例 / 39 行全覆盖测试的单一来源）
// ---------------------------------------------------------------------------

/// 为一个非平台域行生成一个**参数全集显式声明**的 `cap` 块文本。
///
/// 值是确定性的固定样例值（不是缺省值）：每个字段都来自 DSL 显式声明，
/// 走完整校验链，红线 3「不取默认」不被绕过。
pub fn example_cap_block(row_id: &str) -> Result<String, String> {
    let row: &CapabilityRow = row_by_id(row_id).ok_or_else(|| format!("未知编号 {row_id}"))?;
    if row.family == CapFamily::Platform {
        return Err(format!("{row_id} 属平台域，不生成 cap 块"));
    }
    let mut out: String = format!("cap \"{}\" {{\n", row.id);
    for (i, p) in row.api_params.iter().enumerate() {
        let val: String = match param_kind(row, p) {
            ParamKind::Bool => "false".to_string(),
            ParamKind::Int => {
                if let Some(rule) = range_rule(row.id, p) {
                    format!("{}", (rule.lo.ceil() as i64).max(1))
                } else {
                    "1".to_string()
                }
            }
            ParamKind::Num => {
                if let Some(rule) = range_rule(row.id, p) {
                    format!("{}", (rule.lo + rule.hi) / 2.0)
                } else {
                    "0.5".to_string()
                }
            }
            ParamKind::Text => format!("\"w2-{}-{}\"", row.id.to_lowercase(), p.to_lowercase()),
            ParamKind::Variant => {
                if i == 0 {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
        };
        out.push_str(&format!("    {} {}\n", p, val));
    }
    out.push('}');
    Ok(out)
}

/// 生成 39 行中全部 33 个非平台域行的 `cap` 声明文本（覆盖率 / 对拍夹具）。
pub fn full_declaration_blocks() -> String {
    CAPABILITY_ROWS
        .iter()
        .filter(|r| r.family != CapFamily::Platform)
        .filter_map(|r| example_cap_block(r.id).ok())
        .collect::<Vec<String>>()
        .join("\n\n")
}
