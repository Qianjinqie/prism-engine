//! V3.0 W3：观感档位 `finish` 与六族痕迹维度（**只做声明 + 接线 + 调度**）。
//!
//! # 治本口径（路线图 §3.1 / §3.3）
//!
//! 塑料感的根因是生成逻辑「解析的、完美的、无介质的」。本模块**不在帧缓冲上贴
//! 任何东西**：它把用户显式声明的六族维度编译成 [`FinishPlan`]，挂到
//! [`crate::wiring::WiringPlan`] 上，由 `wiring.rs::resolve_frame` 在**渲染前**
//! 逐帧解算成 `w3.<族>.signal` 信号，再由
//! `prism-render/src/movie.rs::render_scene_frame` 在组装
//! `Material` / `Light` / `Scene` **之前**消费。`codegen.rs` 的
//! `post { grain }` S4 帧后颗粒节点是 §3.3 点名的负控样本，本模块不建、
//! 不调、不放大任何帧后节点。
//!
//! # 一等字段（D-30）
//!
//! `// prism-dsl 2` 文档必须显式写 `finish = analytic` 或
//! `finish = crafted { ... }`：漏填 / 非法值 / 全部为 0 / 帧后贴皮 / 真随机
//! 一律 `P9xxx` 三段式 + 行号，`不替用户默认选`。`// prism-dsl 1` 与无头
//! 旧文件不声明 `finish`，按 D-24 版本化语义恒等 `analytic`（不是取默认），
//! 产物 `WiringPlan.finish = None` ⇒ 求值链零变化 ⇒ 逐字节零损失。
//!
//! # 确定性（§3.3 禁令 2）
//!
//! 所有不规则只走既有 `prism_core::hash_mix` / `hash_mix_unit_f32` 与
//! `prism_noise::pcg::hash2_f32`（即 `generators.rs:145` 用的同一基元）。
//! 零真随机、零系统时间、零平台 libm、零线程；两跑逐字节一致。
//!
//! # 红线 1
//!
//! 本文件不实现任何数学 / 渲染 / 质感求值：缓动、Voronoi、胶片曲线、
//! critic 评分、配乐编排全在既有 crate，由 `prism-app::w3_eval` 调用。

use std::collections::BTreeSet;

use prism_core::eval::{hash_mix, hash_mix_bytes, hash_mix_unit_f32};

use crate::ast::NamedField;
use crate::eval::EvalEvent;
use crate::wiring::WiringFrame;
use crate::{err_at, DslError};

/// 强度定点上限：DSL 写 `0..1000`，求值链换算为 `0.000..1.000`。
pub const STRENGTH_MAX: u32 = 1000;

/// W3 命名空间常量（进产物哈希、进噪声链，与 W1 的 `SCHEMA_TAG` 不冲突）。
pub const W3_TAG: u64 = 0x7733_5eed_0000_0001;

/// 观感档位（一等字段，两档都是交付物）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishMode {
    /// 纯净匀质：六族强度全 0，逐字节 ≡ 纯解析基线。
    Analytic,
    /// 像人搓的：按声明的维度与强度把痕迹注入生成源头。
    Crafted,
}

impl FinishMode {
    /// 档位短名（DSL 文本 / 报告 / 产物哈希）。
    pub fn tag(self) -> &'static str {
        match self {
            Self::Analytic => "analytic",
            Self::Crafted => "crafted",
        }
    }

    /// 档位判别字节（进规范字节序列，禁止两端同错）。
    pub fn code(self) -> u8 {
        match self {
            Self::Analytic => 0,
            Self::Crafted => 1,
        }
    }

    /// 从 DSL 标识符解析；非法值返回 `None`（调用方报 P9002，不取默认）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "analytic" => Some(Self::Analytic),
            "crafted" => Some(Self::Crafted),
            _ => None,
        }
    }
}

/// 六族痕迹维度（§3.4 冻结表的一等枚举）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceFamily {
    /// 运动：惯性 / 犹豫 / 微抖 / 呼吸。
    Motion,
    /// 几何：受控的手工不完美 / 微不对称。
    Geometry,
    /// 材质光照：微表面 / 漏光 / 色偏。
    MaterialLight,
    /// 时间镜头：快门 / 辉光溢出 / 帧间规律性。
    TimeCamera,
    /// 构图色彩：有意图的取舍 / 色调倾向。
    CompositionColor,
    /// 剪辑音画：不等长呼吸 / 人在操作的取舍。
    EditAudio,
}

impl TraceFamily {
    /// 六族全集（固定顺序 = 求值顺序 = 凭证顺序，确定性来源）。
    pub const ALL: [TraceFamily; 6] = [
        Self::Motion,
        Self::Geometry,
        Self::MaterialLight,
        Self::TimeCamera,
        Self::CompositionColor,
        Self::EditAudio,
    ];

    /// DSL 字段名（也是 `w3.<tag>.*` 键名）。
    pub fn tag(self) -> &'static str {
        match self {
            Self::Motion => "motion",
            Self::Geometry => "geometry",
            Self::MaterialLight => "material_light",
            Self::TimeCamera => "time_camera",
            Self::CompositionColor => "composition_color",
            Self::EditAudio => "edit_audio",
        }
    }

    /// 从 DSL 标识符解析族；未知族返回 `None`。
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.tag() == s)
    }

    /// 噪声链维度号（替代字符串哈希，纯整数）。
    pub fn ordinal(self) -> u64 {
        match self {
            Self::Motion => 1,
            Self::Geometry => 2,
            Self::MaterialLight => 3,
            Self::TimeCamera => 4,
            Self::CompositionColor => 5,
            Self::EditAudio => 6,
        }
    }

    /// 引擎触点（既有 crate 的公开能力，W3 只接线不重实现）。
    pub fn engine_touchpoint(self) -> &'static str {
        match self {
            Self::Motion => "crates/prism-author/src/easing.rs:28 (EasingKind::apply)",
            Self::Geometry => {
                "crates/prism-genart/src/generators.rs:145 (pcg::hash2_f32 jitter) / :371 (VoronoiArt.site_jitter)"
            }
            Self::MaterialLight => {
                "crates/prism-postfx/src/effects/film.rs:71 (FilmCurve::eval) / :283 (Halation) / :227 (LayeredBloom) / :442 (FilmGrain)"
            }
            Self::TimeCamera => {
                "crates/prism-postfx/src/effects/{ca.rs:23, dof_full.rs:26, bloom.rs:35, vignette.rs:21}"
            }
            Self::CompositionColor => {
                "crates/prism-aesthetic/src/critic.rs:40 (Dim) / :151 (Rubric) / critic_loop.rs / cmd_critique.rs"
            }
            Self::EditAudio => {
                "crates/prism-audio/src/score.rs:127 (compose) / :253 (mood_scale) / :266 (compose_pieces) / beats.rs:59 / harmony.rs:197 / reactive.rs:59"
            }
        }
    }

    /// 生成源头注入点（**渲染前**链路；禁止填 postfx 末端）。
    pub fn injection_point(self) -> &'static str {
        match self {
            Self::Motion => "wiring.rs::bind_source/resolve_frame → movie.rs::wiring_frame_values → render_scene_frame(相机轨道半径，Material 之前)",
            Self::Geometry => "wiring.rs::resolve_frame(pcg::hash2_f32) → movie.rs::render_scene_frame(球体半径，Scene 组装之前)",
            Self::MaterialLight => "wiring.rs::resolve_frame(hash_mix) → movie.rs::render_scene_frame(roughness/metalness/Light，Material::new 之前)",
            Self::TimeCamera => "wiring.rs::resolve_frame(hash_mix) → movie.rs::render_scene_frame(fov_rad，Camera::new 之前)",
            Self::CompositionColor => "wiring.rs::resolve_frame(hash_mix) → movie.rs::render_scene_frame(orbit_center/Aesthetic.contrast，Aesthetic::new 附近)",
            Self::EditAudio => "wiring.rs::resolve_frame(hash_mix) → movie.rs::render_scene_frame(dur_local，eval_camera_orbit 之前)",
        }
    }

    /// 强度范围（定点 0..1000，全族一致；`0` 合法 = 该族绝对匀质）。
    pub fn strength_range(self) -> (u32, u32) {
        let _ = self;
        (0, STRENGTH_MAX)
    }

    /// `analytic` 档该族的值（恒 0 ⇒ 零意外噪点）。
    pub fn analytic_value(self) -> u32 {
        let _ = self;
        0
    }
}

/// 一个痕迹维度的声明（独立开关 + 强度，均可为 0）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceDim {
    /// 独立开关。
    pub enabled: bool,
    /// 强度定点 `0..1000`。
    pub strength: u32,
    /// 声明行号（三段式报错 / 负控定位）。
    pub line: usize,
}

impl TraceDim {
    /// 关闭态（analytic 档 / 未启用族）。
    pub const OFF: TraceDim = TraceDim { enabled: false, strength: 0, line: 0 };

    /// 是否真的产生痕迹（声明一致性判据的口径）。
    pub fn active(self) -> bool {
        self.enabled && self.strength > 0
    }
}

/// critic 五维目标名（顺序与 `prism_aesthetic::critic::Dim::ALL` 一致）。
pub const RUBRIC_DIMS: [&str; 5] = ["exposure", "contrast", "color", "sharpness", "composition"];

/// 从 DSL 可声明的 `Rubric` / `Target` 规格（f32 以位型保存，零平台差异）。
///
/// 声明可选；未声明时 `w3_eval` 调 `Rubric::default()`（既有引擎 API 的合法
/// 输入，不是替用户选 `finish` 档位）。一旦声明，五维 `target/tol/hard`
/// 与 `color_cast_max` 必须全部显式给出，不补缺省。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RubricPlan {
    /// 五维 `Target { target, tol, hard }` 的 f32 位型。
    pub targets: [[u32; 3]; 5],
    /// 色偏硬上限的 f32 位型。
    pub color_cast_max: u32,
}

/// 一份编译完成的观感声明（进 `WiringPlan`，逐帧解算的输入）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishPlan {
    /// 档位。
    pub mode: FinishMode,
    /// 六族维度（顺序 = `TraceFamily::ALL`）。
    pub dims: [TraceDim; 6],
    /// 一等种子（来自 v3 `seed`，进产物哈希与噪声链）。
    pub seed: u64,
    /// `finish` 声明行号。
    pub line: usize,
    /// 可选 critic 规格（构图色彩族）。
    pub rubric: Option<RubricPlan>,
}

impl FinishPlan {
    /// 构造 `analytic` 档：六族恒 0、无噪声、无 rubric。
    pub fn analytic(seed: u64, line: usize) -> Self {
        Self {
            mode: FinishMode::Analytic,
            dims: [TraceDim::OFF; 6],
            seed,
            line,
            rubric: None,
        }
    }

    /// 是否 `crafted`（只有该档允许注入痕迹）。
    pub fn is_crafted(&self) -> bool {
        self.mode == FinishMode::Crafted
    }

    /// 取一族维度。
    pub fn dim(&self, family: TraceFamily) -> TraceDim {
        self.dims[family.ordinal() as usize - 1]
    }

    /// 实际启用（enabled && strength > 0）的族。
    pub fn active_families(&self) -> Vec<TraceFamily> {
        TraceFamily::ALL
            .iter()
            .copied()
            .filter(|f| self.dim(*f).active())
            .collect()
    }

    /// 规范字节序列（进 `WiringPlan::canonical_bytes` / 产物哈希 / 同源对拍）。
    ///
    /// **诊断行号不进产物**：`self.line` 与 `TraceDim::line` 只用于三段式报错
    /// 与求值轨迹定位，属诊断元数据。与 `W2 WiringPlan::canonical_bytes`
    /// 同口径——「直接调引擎 API 构造同一观感配置」一路没有源码行号，语义
    /// 相同即应逐字节相同（§3.5 V5，禁止两端同错）。换档 / 换维度 / 换强度 /
    /// 换种子 / 换 rubric 全部仍可检出（判据 V1 / V2⑤ 的执行体）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.push(self.mode.code());
        out.extend_from_slice(&self.seed.to_le_bytes());
        for d in &self.dims {
            out.push(if d.enabled { 1 } else { 0 });
            out.extend_from_slice(&(d.strength as u64).to_le_bytes());
        }
        match &self.rubric {
            None => out.push(0),
            Some(r) => {
                out.push(1);
                for t in &r.targets {
                    for b in t {
                        out.extend_from_slice(&b.to_le_bytes());
                    }
                }
                out.extend_from_slice(&r.color_cast_max.to_le_bytes());
            }
        }
        out
    }

    /// 构造 `critic::Rubric` 所需的位型视图（供 `w3_eval` 直接 `Target::new`）。
    pub fn rubric_or_engine_default(&self) -> Option<RubricPlan> {
        self.rubric
    }
}

/// 一帧的六族注入信号（渲染层消费面，全 f32 位型来源，`analytic` 恒 0）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FrameSignals {
    /// 是否 `crafted` 且已注入（false ⇒ 渲染层一行不改）。
    pub crafted: bool,
    pub motion: f32,
    pub geometry: f32,
    pub material_light: f32,
    pub time_camera: f32,
    pub composition_color: f32,
    pub edit_audio: f32,
}

impl FrameSignals {
    /// 取某族信号（按固定枚举分派，不做字符串查表）。
    pub fn of(self, family: TraceFamily) -> f32 {
        match family {
            TraceFamily::Motion => self.motion,
            TraceFamily::Geometry => self.geometry,
            TraceFamily::MaterialLight => self.material_light,
            TraceFamily::TimeCamera => self.time_camera,
            TraceFamily::CompositionColor => self.composition_color,
            TraceFamily::EditAudio => self.edit_audio,
        }
    }
}

/// 从 `WiringFrame` 读出六族信号（渲染层唯一消费入口）。
pub fn signals_of(wf: &WiringFrame) -> FrameSignals {
    let crafted: bool = wf
        .value("w3.mode")
        .map(|v| v.to_bits() == 1.0_f32.to_bits())
        .unwrap_or(false);
    let get = |f: TraceFamily| wf.value(&format!("w3.{}.signal", f.tag())).unwrap_or(0.0);
    FrameSignals {
        crafted,
        motion: get(TraceFamily::Motion),
        geometry: get(TraceFamily::Geometry),
        material_light: get(TraceFamily::MaterialLight),
        time_camera: get(TraceFamily::TimeCamera),
        composition_color: get(TraceFamily::CompositionColor),
        edit_audio: get(TraceFamily::EditAudio),
    }
}

/// W3 注入键全集（静态门禁逐族核对实现是否真消费）。
pub const W3_INJECTION_KEYS: [&str; 6] = [
    "w3.motion.signal",
    "w3.geometry.signal",
    "w3.material_light.signal",
    "w3.time_camera.signal",
    "w3.composition_color.signal",
    "w3.edit_audio.signal",
];

// ---------------------------------------------------------------------------
// 生成源头逐帧注入（wiring.rs::resolve_frame 调用，渲染前）
// ---------------------------------------------------------------------------

/// 计算一族在某一帧的种子化确定性噪声 `[0,1)`。
///
/// 几何族走 `prism_noise::pcg::hash2_f32`（`generators.rs:145` 同一基元），
/// 其余走 `prism_core::hash_mix` 链 + `hash_mix_unit_f32`
/// （`eval.rs` 的 splitmix64 终化，与 W1 `rand(tag)` 同口径）。
/// **零真随机、零系统时间、零 libm**；同 (seed, frame, node_id, 族) 同值。
pub fn dim_noise(plan: &FinishPlan, family: TraceFamily, frame: u64, global_seed: u64, node_id: u64) -> f32 {
    let base: u64 = hash_mix(hash_mix(plan.seed, W3_TAG), global_seed ^ node_id);
    if family == TraceFamily::Geometry {
        prism_noise::pcg::hash2_f32(base, frame, family.ordinal())
    } else {
        let h: u64 = hash_mix(hash_mix(base, frame), family.ordinal());
        hash_mix_unit_f32(h)
    }
}

/// 生成源头注入：把已声明维度的强度与种子化噪声写进 `WiringFrame`。
///
/// 由 `wiring.rs::resolve_frame` 在**渲染前**逐帧调用；`analytic` 恒等返回，
/// 不插入任何键 ⇒ `WiringFrame` 与未声明 `finish` 的纯解析基线逐字节相同
/// （V1 / V4「零意外噪点」）。`movie.rs::render_scene_frame` 在组装
/// `Material` / `Light` / `Scene` 之前消费这些值，**不是帧后效果节点**。
pub fn inject_frame_values(
    wf: &mut WiringFrame,
    plan: &FinishPlan,
    frame: u64,
    global_seed: u64,
    node_id: u64,
) {
    if !plan.is_crafted() {
        return;
    }
    for family in TraceFamily::ALL {
        let dim: TraceDim = plan.dim(family);
        if !dim.active() {
            continue;
        }
        let raw: f32 = dim_noise(plan, family, frame, global_seed, node_id);
        let strength: f32 = dim.strength as f32 / STRENGTH_MAX as f32;
        let signal: f32 = (raw * 2.0 - 1.0) * strength;
        wf.values
            .insert(format!("w3.{}.strength", family.tag()), strength.to_bits());
        wf.values
            .insert(format!("w3.{}.noise", family.tag()), raw.to_bits());
        wf.values
            .insert(format!("w3.{}.signal", family.tag()), signal.to_bits());
    }
    wf.values.insert("w3.mode".to_string(), 1.0_f32.to_bits());
}

/// 帧值对拍（V1）：逐族重算注入量并与 `WiringFrame` 逐位比较。
///
/// 返回 Err 时带族名与声明行号，供负控 ⑤「字段没接」判红。
pub fn assert_frame_value_pairs(
    wf: &WiringFrame,
    plan: &FinishPlan,
    frame: u64,
    global_seed: u64,
    node_id: u64,
) -> Result<(), String> {
    if !plan.is_crafted() {
        for key in W3_INJECTION_KEYS {
            if wf.values.contains_key(key) {
                return Err(format!("第 {} 行：analytic 档不得出现注入键 {key}", plan.line));
            }
        }
        return Ok(());
    }
    for family in TraceFamily::ALL {
        let dim: TraceDim = plan.dim(family);
        if !dim.active() {
            continue;
        }
        let raw: f32 = dim_noise(plan, family, frame, global_seed, node_id);
        let strength: f32 = dim.strength as f32 / STRENGTH_MAX as f32;
        let signal: f32 = (raw * 2.0 - 1.0) * strength;
        let pairs: [(&str, f32); 3] = [
            ("strength", strength),
            ("noise", raw),
            ("signal", signal),
        ];
        for (name, want) in pairs {
            let key: String = format!("w3.{}.{name}", family.tag());
            let got: u32 = *wf
                .values
                .get(&key)
                .ok_or_else(|| format!("第 {} 行：{} 族的 {key} 未进求值链（字段没接）", dim.line, family.tag()))?;
            if got != want.to_bits() {
                return Err(format!(
                    "第 {} 行：{} 族的 {key} 帧值对拍不一致（声明强度 {}，注入量 {want:?}，实际位型 {got}）",
                    dim.line, family.tag(), dim.strength
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 三段式错误（P9xxx：W3 观感档位码段，与 P1xxx–P8xxx 互不冲突）
// ---------------------------------------------------------------------------

/// 三段式错误构造（在哪 / 什么事 / 怎么改，全部带行号）。
pub fn finish_err(line: usize, code: &str, what: &str, fix: &str) -> DslError {
    err_at(line, format!("[{code}] 第 {line} 行：{what}\n  建议：{fix}"))
}

/// 漏填 `finish`（红线 3 / 红线 4：不替用户默认选档）。
pub fn missing_finish_error(line: usize) -> DslError {
    finish_err(
        line,
        "P9001",
        "`// prism-dsl 2` 文档缺少一等字段 `finish`，编译器不替用户选档",
        "在第 2 行显式写 `finish = analytic`（纯净匀质）或 `finish = crafted { ... }`（像人搓的）；旧 `// prism-dsl 1` 文件按版本契约恒等 analytic，无需声明",
    )
}

const FINISH_SYNTAX_HINT: &str =
    "crafted 块逐族写 `<族> enabled true|false strength 0..1000`；六族是 motion / geometry / material_light / time_camera / composition_color / edit_audio";

// ---------------------------------------------------------------------------
// v3 一等字段扫描（行式、确定性；不改动 W1 的 LL(1)+Pratt 前端）
// ---------------------------------------------------------------------------

/// `finish` 一等字段的扫描结果。
#[derive(Debug, Clone)]
pub struct FinishScan {
    /// 已解析的声明（None = 全文未声明，由 `compile_v3` 在成功编译后报 P9001）。
    pub decl: Option<FinishPlan>,
    /// 把 `finish` 语句所在行替换为等长空行后的源码（行号逐字不变）。
    pub masked_source: String,
}

/// 禁止出现在 W3 观感实现/声明里的真随机与非确定输入（§3.3 禁令 2）。
pub const FORBIDDEN_RANDOM_TOKENS: [&str; 12] = [
    "true_random", "SystemTime", "thread_rng", "getrandom", "rand::", "random(",
    "now()", "clock()", "date()", "Instant::now", "/dev/urandom", "OsRng",
];

/// 帧后贴皮关键字（§3.3 禁令 1；`post { grain }` 是标准负控样本）。
/// 用于 `finish` 声明块体内：块里出现任何贴皮效果名即判红。
pub const FORBIDDEN_POSTHOC_TOKENS: [&str; 8] = [
    "post {", "post{", "grain", "vignette", "chromatic", "bloom", "filter", "lut",
];

/// 全文帧后贴皮门禁的块标记（`post { ... }` 声明起点）。
/// 只认块声明，不认裸效果名——W1 的 `rand("grain")` 是语言层确定性采样，
/// 不是帧后节点，不能误判（零损失契约）。
pub const POSTHOC_BLOCK_MARKERS: [&str; 3] = ["post {", "post{", "postfx"];

/// 真随机静态门禁：全文命中任一禁用 token ⇒ `P9012` + 行号。
pub fn true_random_gate(source: &str) -> Result<(), DslError> {
    for (idx, raw) in source.lines().enumerate() {
        for token in FORBIDDEN_RANDOM_TOKENS {
            if raw.contains(token) {
                return Err(finish_err(
                    idx + 1,
                    "P9012",
                    &format!("观感不规则出现非确定性随机源 '{token}'（§3.3 禁令 2）"),
                    "删掉真随机 / 系统时间 / IO 随机源；抖动与呼吸只走 hash_mix 或 pcg::hash2_f32 的种子化确定性噪声",
                ));
            }
        }
    }
    Ok(())
}

/// 帧后贴皮静态门禁：`// prism-dsl 2` 影片出现 postfx 贴皮块 ⇒ `P9011` + 行号。
///
/// 与用户选哪档无关：`analytic` 档同样判红，不许因选匀质而失效。
pub fn posthoc_gate(source: &str) -> Result<(), DslError> {
    for (idx, raw) in source.lines().enumerate() {
        let line: &str = raw.trim();
        if line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        for token in POSTHOC_BLOCK_MARKERS {
            if line.contains(token) {
                return Err(finish_err(
                    idx + 1,
                    "P9011",
                    &format!("痕迹从生成源挪到帧后贴皮：命中 '{token}'（§3.3 禁令 1）"),
                    "删掉该 postfx 贴皮声明；六族痕迹必须在 wiring.rs::resolve_frame → movie.rs::render_scene_frame 的渲染前链路注入",
                ));
            }
        }
    }
    Ok(())
}

/// 扫描 `// prism-dsl 2` 源码里的 `finish = ...` 一等字段。
///
/// 返回遮罩后的源码（被扫描行替换为空行，行号不变），交给 W1 的
/// `tokenize3` / `Parser3` 原样处理。非法值 / 漏族 / 全 0 / 贴皮 / 真随机
/// 一律三段式 + 行号；**未声明**不在本函数报错，由 `compile_v3` 在
/// 其余编译环节成功后统一报 `P9001`（保证 W1 的四类错误域判据不退化）。
pub fn scan_finish(source: &str) -> Result<FinishScan, DslError> {
    let raw_lines: Vec<&str> = source.lines().collect();
    let mut masked: Vec<String> = raw_lines.iter().map(|s| (*s).to_string()).collect();
    let mut decl: Option<FinishPlan> = None;
    let mut idx: usize = 0;
    while idx < raw_lines.len() {
        let line_no: usize = idx + 1;
        let trimmed: &str = raw_lines[idx].trim();
        if !trimmed.starts_with("finish") {
            idx += 1;
            continue;
        }
        if decl.is_some() {
            return Err(finish_err(
                line_no,
                "P9005",
                "`finish` 一等字段重复声明",
                "一份文档只写一个 finish 声明；六族维度合并到同一个块内",
            ));
        }
        let eq: &str = trimmed.strip_prefix("finish").unwrap_or("").trim_start();
        let Some(rest) = eq.strip_prefix('=') else {
            return Err(finish_err(
                line_no,
                "P9002",
                "`finish` 一等字段必须写成 `finish = analytic | crafted`",
                "写成 `finish = analytic` 或 `finish = crafted { ... }`",
            ));
        };
        let rest: &str = rest.trim();
        let (mode_text, has_block): (&str, bool) = match rest.strip_suffix('{') {
            Some(head) => (head.trim(), true),
            None => (rest, false),
        };
        let mode: FinishMode = FinishMode::parse(mode_text).ok_or_else(|| {
            finish_err(
                line_no,
                "P9002",
                &format!("`finish` 档位 '{mode_text}' 非法（只接受 analytic / crafted）"),
                "写 `finish = analytic` 或 `finish = crafted { ... }`；编译器不替用户默认选档",
            )
        })?;
        masked[idx].clear();
        if mode == FinishMode::Analytic {
            if has_block {
                return Err(finish_err(
                    line_no,
                    "P9007",
                    "analytic 档不允许声明痕迹维度块（语义上六族恒 0）",
                    "删掉花括号块，只写 `finish = analytic`；要逐族声明请选 crafted",
                ));
            }
            decl = Some(FinishPlan::analytic(0, line_no));
            idx += 1;
            continue;
        }
        if !has_block {
            return Err(finish_err(
                line_no,
                "P9003",
                "crafted 档必须带六族维度块",
                FINISH_SYNTAX_HINT,
            ));
        }

        // ---- crafted 块体：逐族声明 + 可选 rubric ----
        let mut dims: [TraceDim; 6] = [TraceDim::OFF; 6];
        let mut seen_dims: BTreeSet<&'static str> = BTreeSet::new();
        let mut rubric_targets: [[u32; 3]; 5] = [[0; 3]; 5];
        let mut rubric_cast: Option<u32> = None;
        let mut seen_rubric: BTreeSet<&'static str> = BTreeSet::new();
        let start_line: usize = line_no;
        let mut closed: bool = false;
        idx += 1;
        while idx < raw_lines.len() {
            let body_line: usize = idx + 1;
            let body: &str = raw_lines[idx].trim();
            masked[idx].clear();
            if body == "}" {
                closed = true;
                idx += 1;
                break;
            }
            if body.is_empty() || body.starts_with("//") || body.starts_with('#') {
                idx += 1;
                continue;
            }
            for token in FORBIDDEN_RANDOM_TOKENS {
                if body.contains(token) {
                    return Err(finish_err(
                        body_line,
                        "P9012",
                        &format!("观感不规则出现非确定性随机源 '{token}'（§3.3 禁令 2）"),
                        "抖动与呼吸只走 hash_mix 或 pcg::hash2_f32 的种子化确定性噪声",
                    ));
                }
            }
            for token in FORBIDDEN_POSTHOC_TOKENS {
                if body.contains(token) {
                    return Err(finish_err(
                        body_line,
                        "P9011",
                        &format!("痕迹从生成源挪到帧后贴皮：命中 '{token}'（§3.3 禁令 1）"),
                        "六族痕迹必须在生成源头注入，不许出帧后贴 grain / 滤镜 / 色调模板",
                    ));
                }
            }
            let words: Vec<&str> = body.split_whitespace().collect();
            if words[0] == "rubric" {
                parse_rubric_line(&words, body_line, &mut rubric_targets, &mut rubric_cast, &mut seen_rubric)?;
            } else {
                let family: TraceFamily = TraceFamily::parse(words[0]).ok_or_else(|| {
                    finish_err(
                        body_line,
                        "P9003",
                        &format!("未知痕迹维度 '{}'", words[0]),
                        FINISH_SYNTAX_HINT,
                    )
                })?;
                if !seen_dims.insert(family.tag()) {
                    return Err(finish_err(
                        body_line,
                        "P9005",
                        &format!("痕迹维度 '{}' 重复声明", family.tag()),
                        "每族只声明一次 `enabled` + `strength`",
                    ));
                }
                if words.len() != 5 || words[1] != "enabled" || words[3] != "strength" {
                    return Err(finish_err(
                        body_line,
                        "P9003",
                        &format!("'{}' 维度声明形态非法", family.tag()),
                        &format!("写成 `{} enabled true|false strength 0..1000`", family.tag()),
                    ));
                }
                let enabled: bool = match words[2] {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(finish_err(
                            body_line,
                            "P9004",
                            &format!("'{}' 的 enabled 只接受 true / false，实际 '{other}'", family.tag()),
                            "写 `enabled true` 或 `enabled false`；关闭时 strength 必须为 0",
                        ));
                    }
                };
                let strength: u32 = words[4].parse::<u32>().map_err(|_| {
                    finish_err(
                        body_line,
                        "P9004",
                        &format!("'{}' 的 strength '{}' 不是 0..1000 的整数", family.tag(), words[4]),
                        "写 0..1000 的定点整数；0 = 该族绝对匀质",
                    )
                })?;
                if strength > STRENGTH_MAX {
                    return Err(finish_err(
                        body_line,
                        "P9004",
                        &format!("'{}' 的 strength {strength} 超出 [0, {STRENGTH_MAX}]", family.tag()),
                        &format!("把 strength 改到 [0, {STRENGTH_MAX}] 内；超范围不取默认、不钳制"),
                    ));
                }
                if !enabled && strength != 0 {
                    return Err(finish_err(
                        body_line,
                        "P9004",
                        &format!("'{}' 关闭却写了非零强度 {strength}（声明自相矛盾）", family.tag()),
                        "enabled false 时 strength 必须写 0；要该族生效请写 enabled true",
                    ));
                }
                dims[family.ordinal() as usize - 1] = TraceDim { enabled, strength, line: body_line };
            }
            idx += 1;
        }
        if !closed {
            return Err(finish_err(
                start_line,
                "P9003",
                "crafted 块缺少右花括号 '}'",
                FINISH_SYNTAX_HINT,
            ));
        }
        let missing: Vec<&str> = TraceFamily::ALL
            .iter()
            .map(|f| f.tag())
            .filter(|t| !seen_dims.contains(*t))
            .collect();
        if !missing.is_empty() {
            return Err(finish_err(
                start_line,
                "P9003",
                &format!("crafted 档漏填 {} 个痕迹维度：{}", missing.len(), missing.join(" / ")),
                FINISH_SYNTAX_HINT,
            ));
        }
        let active: usize = dims.iter().filter(|d| d.active()).count();
        if active == 0 {
            return Err(finish_err(
                start_line,
                "P9006",
                "声明 `crafted` 却六族全 0（声明与产物不符，§3.5 V2 负控 ③）",
                "至少给一族写 `enabled true strength > 0`；若确实要匀质请显式选 `finish = analytic`",
            ));
        }
        let rubric: Option<RubricPlan> = if seen_rubric.is_empty() {
            None
        } else {
            let missing_r: Vec<&str> = RUBRIC_DIMS
                .iter()
                .copied()
                .chain(std::iter::once("cast"))
                .filter(|t| !seen_rubric.contains(*t))
                .collect();
            if !missing_r.is_empty() {
                return Err(finish_err(
                    start_line,
                    "P9003",
                    &format!("rubric 声明不全，缺：{}", missing_r.join(" / ")),
                    "rubric 要么整组不写（用引擎既有默认规格），要么五维 target/tol/hard + cast 全写",
                ));
            }
            Some(RubricPlan {
                targets: rubric_targets,
                color_cast_max: rubric_cast.unwrap_or(0),
            })
        };
        decl = Some(FinishPlan {
            mode,
            dims,
            seed: 0, // 由 compile_v3 用一等字段 seed 回填
            line: start_line,
            rubric,
        });
    }
    let mut masked_source: String = masked.join("\n");
    if source.ends_with('\n') {
        masked_source.push('\n');
    }
    Ok(FinishScan { decl, masked_source })
}

/// 解析一行 `rubric <维> <target> <tol> <hard>` 或 `rubric cast <值>`。
fn parse_rubric_line(
    words: &[&str],
    line: usize,
    targets: &mut [[u32; 3]; 5],
    cast: &mut Option<u32>,
    seen: &mut BTreeSet<&'static str>,
) -> Result<(), DslError> {
    if words.len() < 2 {
        return Err(finish_err(line, "P9003", "rubric 行缺少维度名", "写 `rubric exposure 0.18 0.6 2.4` 或 `rubric cast 0.12`"));
    }
    let name: &str = words[1];
    if name == "cast" {
        if words.len() != 3 {
            return Err(finish_err(line, "P9003", "rubric cast 需要 1 个数值", "写 `rubric cast 0.12`"));
        }
        let v: f32 = parse_finite(words[2], line, "rubric cast")?;
        if !seen.insert("cast") {
            return Err(finish_err(line, "P9005", "rubric cast 重复声明", "每个 rubric 项只声明一次"));
        }
        *cast = Some(v.to_bits());
        return Ok(());
    }
    let Some(pos) = RUBRIC_DIMS.iter().position(|d| *d == name) else {
        return Err(finish_err(
            line,
            "P9003",
            &format!("未知 rubric 维度 '{name}'"),
            "rubric 维度取 exposure / contrast / color / sharpness / composition，或 cast",
        ));
    };
    if words.len() != 5 {
        return Err(finish_err(
            line,
            "P9003",
            &format!("rubric {name} 需要 target / tol / hard 三个数值"),
            &format!("写 `rubric {name} <target> <tol> <hard>`"),
        ));
    }
    if !seen.insert(RUBRIC_DIMS[pos]) {
        return Err(finish_err(line, "P9005", &format!("rubric {name} 重复声明"), "每个 rubric 项只声明一次"));
    }
    for (i, text) in words[2..5].iter().enumerate() {
        targets[pos][i] = parse_finite(text, line, &format!("rubric {name}"))?.to_bits();
    }
    Ok(())
}

/// 有限 f32 解析（禁止 NaN / inf 混进产物；`f32::to_bits` 位型存储）。
fn parse_finite(text: &str, line: usize, what: &str) -> Result<f32, DslError> {
    let v: f32 = text.parse::<f32>().map_err(|_| {
        finish_err(line, "P9004", &format!("{what} 的取值 '{text}' 不是合法数值"), "写有限十进制数值")
    })?;
    if !v.is_finite() {
        return Err(finish_err(line, "P9004", &format!("{what} 的取值 '{text}' 非有限"), "写有限十进制数值"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// v2 `finish { ... }` 块（compile_v3 降级出来的声明文本也走这条路径）
// ---------------------------------------------------------------------------

/// 从 v2 扁平字段流编译 `FinishPlan`（不取默认、全集必填、逐项校验）。
///
/// 扁平字段名：
/// `mode` / `seed` / `<族>_enabled` / `<族>_strength` /
/// `rubric_<维>_target|tol|hard` / `rubric_cast`。
pub(crate) fn compile_v2_finish(fields: &[NamedField], line: usize) -> Result<FinishPlan, DslError> {
    use crate::ast::FieldValue;
    let mut mode: Option<FinishMode> = None;
    let mut seed: Option<u64> = None;
    let mut dims: [TraceDim; 6] = [TraceDim::OFF; 6];
    let mut enabled_seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut strength_seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut rubric_targets: [[u32; 3]; 5] = [[0; 3]; 5];
    let mut rubric_cast: Option<u32> = None;
    let mut rubric_seen: BTreeSet<&'static str> = BTreeSet::new();

    for f in fields {
        let value: &FieldValue = f.value.as_ref().ok_or_else(|| {
            finish_err(f.line, "P9003", &format!("finish 字段 '{}' 缺少值", f.name), "给每个字段显式写值")
        })?;
        if f.name == "mode" {
            let s: &str = match value {
                FieldValue::Ident(s) => s.as_str(),
                _ => return Err(finish_err(f.line, "P9002", "finish.mode 需要裸标识符", "写 `mode analytic` 或 `mode crafted`")),
            };
            if mode.replace(FinishMode::parse(s).ok_or_else(|| {
                finish_err(f.line, "P9002", &format!("finish.mode '{s}' 非法"), "只接受 analytic / crafted")
            })?).is_some()
            {
                return Err(finish_err(f.line, "P9005", "finish.mode 重复声明", "mode 只写一次"));
            }
            continue;
        }
        if f.name == "seed" {
            let n: f32 = match value {
                FieldValue::Num(v) => *v,
                _ => return Err(finish_err(f.line, "P9004", "finish.seed 需要非负整数", "写 `seed 42`")),
            };
            if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
                return Err(finish_err(f.line, "P9004", "finish.seed 必须是非负整数", "写 `seed 42`"));
            }
            if seed.replace(n as u64).is_some() {
                return Err(finish_err(f.line, "P9005", "finish.seed 重复声明", "seed 只写一次"));
            }
            continue;
        }
        if let Some(family) = f
            .name
            .strip_suffix("_enabled")
            .and_then(TraceFamily::parse)
        {
            let b: bool = match value {
                FieldValue::Ident(s) if s == "true" => true,
                FieldValue::Ident(s) if s == "false" => false,
                _ => return Err(finish_err(f.line, "P9004", &format!("{} 的 enabled 只接受 true/false", family.tag()), "写 `enabled true` 或 `enabled false`")),
            };
            if !enabled_seen.insert(family.tag()) {
                return Err(finish_err(f.line, "P9005", &format!("{} enabled 重复", family.tag()), "每族只写一次"));
            }
            dims[family.ordinal() as usize - 1].enabled = b;
            dims[family.ordinal() as usize - 1].line = f.line;
            continue;
        }
        if let Some(family) = f
            .name
            .strip_suffix("_strength")
            .and_then(TraceFamily::parse)
        {
            let n: f32 = match value {
                FieldValue::Num(v) => *v,
                _ => return Err(finish_err(f.line, "P9004", &format!("{} 的 strength 需要整数", family.tag()), "写 0..1000 的整数")),
            };
            if !n.is_finite() || n < 0.0 || n.fract() != 0.0 || n > STRENGTH_MAX as f32 {
                return Err(finish_err(
                    f.line,
                    "P9004",
                    &format!("{} 的 strength {n} 超出 [0, {STRENGTH_MAX}]", family.tag()),
                    "写 0..1000 的定点整数；超范围不取默认、不钳制",
                ));
            }
            if !strength_seen.insert(family.tag()) {
                return Err(finish_err(f.line, "P9005", &format!("{} strength 重复", family.tag()), "每族只写一次"));
            }
            dims[family.ordinal() as usize - 1].strength = n as u32;
            continue;
        }
        if let Some(rest) = f.name.strip_prefix("rubric_") {
            let n: f32 = match value {
                FieldValue::Num(v) => *v,
                _ => return Err(finish_err(f.line, "P9004", &format!("{rest} 需要有限数值"), "写有限十进制数值")),
            };
            if !n.is_finite() {
                return Err(finish_err(f.line, "P9004", &format!("{rest} 非有限"), "写有限十进制数值"));
            }
            if rest == "cast" {
                if rubric_cast.replace(n.to_bits()).is_some() {
                    return Err(finish_err(f.line, "P9005", "rubric_cast 重复", "只写一次"));
                }
                rubric_seen.insert("cast");
                continue;
            }
            let (dim_name, part): (&str, &str) = if let Some(x) = rest.strip_suffix("_target") {
                (x, "target")
            } else if let Some(x) = rest.strip_suffix("_tol") {
                (x, "tol")
            } else if let Some(x) = rest.strip_suffix("_hard") {
                (x, "hard")
            } else {
                return Err(finish_err(f.line, "P9003", &format!("未知 finish 字段 '{}'", f.name), "字段名见 docs/v3.0-痕迹维度表.md"));
            };
            let Some(pos) = RUBRIC_DIMS.iter().position(|d| *d == dim_name) else {
                return Err(finish_err(f.line, "P9003", &format!("未知 rubric 维度 '{dim_name}'"), "五维名见 critic::Dim::ALL"));
            };
            let slot: usize = match part {
                "target" => 0,
                "tol" => 1,
                _ => 2,
            };
            rubric_targets[pos][slot] = n.to_bits();
            rubric_seen.insert(RUBRIC_DIMS[pos]);
            continue;
        }
        return Err(finish_err(
            f.line,
            "P9003",
            &format!("未知 finish 字段 '{}'", f.name),
            "字段名见 docs/v3.0-痕迹维度表.md（mode / seed / 六族 enabled|strength / rubric_*）",
        ));
    }

    let mode: FinishMode = mode.ok_or_else(|| {
        finish_err(line, "P9002", "finish 块缺少 mode", "写 `mode analytic` 或 `mode crafted`")
    })?;
    let seed: u64 = seed.ok_or_else(|| {
        finish_err(line, "P9004", "finish 块缺少 seed", "写 `seed <非负整数>`；种子进产物哈希与噪声链")
    })?;
    if mode == FinishMode::Analytic {
        let extra: bool = enabled_seen.len() + strength_seen.len() + rubric_seen.len() > 0;
        if extra {
            return Err(finish_err(
                line,
                "P9007",
                "analytic 档不允许声明痕迹维度 / rubric（六族语义恒 0）",
                "删掉维度字段，只保留 mode 与 seed；要逐族声明请选 crafted",
            ));
        }
        return Ok(FinishPlan::analytic(seed, line));
    }
    let missing: Vec<&str> = TraceFamily::ALL
        .iter()
        .map(|f| f.tag())
        .filter(|t| !enabled_seen.contains(*t) || !strength_seen.contains(*t))
        .collect();
    if !missing.is_empty() {
        return Err(finish_err(
            line,
            "P9003",
            &format!("crafted 档漏填 {} 个维度的 enabled/strength：{}", missing.len(), missing.join(" / ")),
            FINISH_SYNTAX_HINT,
        ));
    }
    for d in &dims {
        if !d.enabled && d.strength != 0 {
            return Err(finish_err(
                d.line,
                "P9004",
                "关闭的维度写了非零 strength（声明自相矛盾）",
                "enabled false 时 strength 必须为 0",
            ));
        }
    }
    if !dims.iter().any(|d| d.active()) {
        return Err(finish_err(
            line,
            "P9006",
            "声明 `crafted` 却六族全 0（声明与产物不符，§3.5 V2 负控 ③）",
            "至少给一族 enabled true + strength > 0；要匀质请显式选 analytic",
        ));
    }
    let rubric: Option<RubricPlan> = if rubric_seen.is_empty() {
        None
    } else {
        let missing_r: Vec<&str> = RUBRIC_DIMS
            .iter()
            .copied()
            .chain(std::iter::once("cast"))
            .filter(|t| !rubric_seen.contains(*t))
            .collect();
        if !missing_r.is_empty() {
            return Err(finish_err(
                line,
                "P9003",
                &format!("rubric 声明不全，缺：{}", missing_r.join(" / ")),
                "rubric 要么整组不写，要么五维 target/tol/hard + cast 全写",
            ));
        }
        Some(RubricPlan { targets: rubric_targets, color_cast_max: rubric_cast.unwrap_or(0) })
    };
    Ok(FinishPlan { mode, dims, seed, line, rubric })
}

/// 生成等价的 v2 `finish { ... }` 声明文本（`compile_v3` 降级用）。
///
/// 该文本进 `lowered_source` ⇒ 同源对拍 `compile_v2(&lowered_source)` 与
/// `CompileV3Output::base` 逐字段相等；同时 `finish` 进产物哈希。
pub fn finish_block_text(plan: &FinishPlan) -> String {
    let mut out: String = String::from("finish {\n");
    out.push_str(&format!("    mode {}\n", plan.mode.tag()));
    out.push_str(&format!("    seed {}\n", plan.seed));
    if plan.is_crafted() {
        for f in TraceFamily::ALL {
            let d: TraceDim = plan.dim(f);
            out.push_str(&format!("    {}_enabled {}\n", f.tag(), if d.enabled { "true" } else { "false" }));
            out.push_str(&format!("    {}_strength {}\n", f.tag(), d.strength));
        }
        if let Some(r) = &plan.rubric {
            for (i, name) in RUBRIC_DIMS.iter().enumerate() {
                for (slot, part) in ["target", "tol", "hard"].iter().enumerate() {
                    let v: f32 = f32::from_bits(r.targets[i][slot]);
                    out.push_str(&format!("    rubric_{name}_{part} {v}\n"));
                }
            }
            out.push_str(&format!(
                "    rubric_cast {}\n",
                f32::from_bits(r.color_cast_max)
            ));
        }
    }
    out.push('}');
    out
}

/// `finish` 进产物哈希（W3 对 W1 §5 F-8 的正式扩链，逐字节可复现）。
pub fn mix_artifact_hash(h: u64, plan: &FinishPlan) -> u64 {
    hash_mix_bytes(hash_mix(h, W3_TAG), &plan.canonical_bytes())
}

/// `finish` 的求值轨迹事件（F-5；一等字段的声明次序先于语句求值）。
pub fn trace_event(plan: &FinishPlan) -> EvalEvent {
    EvalEvent {
        line: plan.line,
        tag: hash_mix(W3_TAG, plan.mode.code() as u64),
        value_bits: hash_mix_bytes(plan.seed, &plan.canonical_bytes()),
    }
}

/// 声明文本示例（`docs` / 测试夹具的单一来源，不编数字）。
pub fn example_crafted_finish() -> String {
    let mut plan: FinishPlan = FinishPlan::analytic(0, 2);
    plan.mode = FinishMode::Crafted;
    plan.dims = [
        TraceDim { enabled: true, strength: 620, line: 3 },
        TraceDim { enabled: false, strength: 0, line: 4 },
        TraceDim { enabled: true, strength: 400, line: 5 },
        TraceDim { enabled: true, strength: 300, line: 6 },
        TraceDim { enabled: true, strength: 250, line: 7 },
        TraceDim { enabled: true, strength: 500, line: 8 },
    ];
    String::from("finish = crafted {\n")
        + &TraceFamily::ALL
            .iter()
            .map(|f| {
                let d: TraceDim = plan.dim(*f);
                format!(
                    "    {} enabled {} strength {}\n",
                    f.tag(),
                    if d.enabled { "true" } else { "false" },
                    d.strength
                )
            })
            .collect::<String>()
        + "}"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_scan_has_no_dims_and_masks_line() {
        let src: &str = "// prism-dsl 2\nseed = 7\nfinish = analytic\nemit \"x\"\n";
        let scan: FinishScan = scan_finish(src).expect("analytic 应可扫描");
        let plan: FinishPlan = scan.decl.expect("应有声明");
        assert_eq!(plan.mode, FinishMode::Analytic);
        assert!(plan.active_families().is_empty());
        assert_eq!(scan.masked_source.lines().count(), src.lines().count());
        assert_eq!(scan.masked_source.lines().nth(2).unwrap(), "");
    }

    #[test]
    fn crafted_all_zero_is_rejected_with_line() {
        let src: &str = "finish = crafted {\n motion enabled false strength 0\n geometry enabled false strength 0\n material_light enabled false strength 0\n time_camera enabled false strength 0\n composition_color enabled false strength 0\n edit_audio enabled false strength 0\n}\n";
        let err: DslError = scan_finish(src).expect_err("全 0 crafted 必须判红");
        assert_eq!(err.line, 1);
        assert!(err.message.contains("P9006"));
        assert!(err.message.contains("建议："));
    }

    #[test]
    fn injection_is_deterministic_and_seed_sensitive() {
        let mut plan: FinishPlan = FinishPlan::analytic(9, 2);
        plan.mode = FinishMode::Crafted;
        plan.dims[0] = TraceDim { enabled: true, strength: 620, line: 3 };
        let mut a: WiringFrame = WiringFrame::default();
        let mut b: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut a, &plan, 5, 0, 0);
        inject_frame_values(&mut b, &plan, 5, 0, 0);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_frame_value_pairs(&a, &plan, 5, 0, 0).expect("帧值应与注入量逐位对拍");
        plan.seed = 10;
        let mut c: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut c, &plan, 5, 0, 0);
        assert_ne!(a.canonical_bytes(), c.canonical_bytes(), "换种子必须可检出");
    }

    #[test]
    fn analytic_never_injects_any_key() {
        let plan: FinishPlan = FinishPlan::analytic(1, 2);
        let mut wf: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut wf, &plan, 0, 0, 0);
        assert!(wf.values.is_empty());
        assert!(!signals_of(&wf).crafted);
    }
}
