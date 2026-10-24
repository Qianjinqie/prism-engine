//! prism-render 节点注册（D-27 静态汇入制）。
//!
//! `DirtyRange` 是增量渲染（C3）的脏区间闭包特解：给定"参数实际变化的帧
//! 区间"，`apply` 返回同一区间在祖先链上的**闭包**语义等价表示（此处以
//! `(start, end, span)` 标量三元组表达）——确定性纯函数，把"哪些帧需要
//! 重算"从工程数据结构提升为可注册、可签名、可复用的数学特解（D-22：
//! 新功能 = 新特解；D-27：宏展开为 `__prism_collect`，静态汇入单一注册表）。
//!
//! 语义边界：脏区间**推导**（签名比对 + 祖先传播）在 [`crate::incremental`]；
//! 本特解只承载"区间闭包"这一可复用内核，供调度器按 StructureId 求值。
//!
//! # W1-A3：DSL v2 新节点消费面（本文件下半部分）
//! 消费上一波 DSL 新落、已进 `graph` 但不进 `SceneV2::node_ids`（render 消费面
//! 清单）的三种节点——缺口对齐 codegen 的登记约定（`AudioInfo::node_ids` /
//! `PostInfo::node_ids` / 节点图内显式登记）：
//! - **audio → S6 谱解**（[`AudioBed`]）：`audio { music { bpm/gain/duck } }`
//!   投影的 `SpectralDecomposition` 节点，消费 = 校验后取回静态参数（render
//!   管线出图不出声，声音合成归音频 / 混流阶段）；
//! - **post → S2/S4 效果链**（[`PostChain`]）：bloom / vignette / chromatic
//!   归 S2 卷积族、grain 归 S4 双尺度颗粒特解，`Composition::Chain` 按声明序
//!   串行（f∘g∘h）；链输入**隐式挂场景合成输出**（AE / Natron / Blender 通行
//!   约定——不要求用户声明输入，不开显式输入字段，将来多层渲染再开）；
//! - **light v2 → S8 源项**（[`LightTerm`]）：`stage3d light` 投影的
//!   `RadiativeTransfer` 场景源项节点（不进任何 Info 清单，按图扫判别：
//!   `dir_x` 存在且非 material 的 `albedo_r` 判别，两者互斥）。
//!
//! 入口 [`consume_scene`]（逐场景）；消费不到（悬空 / 结构不符 / 参数非法）
//! 一律三段式 [`ConsumeError`] 显式报错，绝不静默跳过或近似。

use prism_core::node::{
    Composition, DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, NodeId,
    OpCapabilities, ParamValue, Payload, StructureId, StructureMeta,
};
use prism_core::{Graph, Light, Params};
use prism_dsl::{CompileV2Output, SceneV2};
use prism_gpu::Framebuffer;
use prism_math::Vec3;

/// S15@prism_render：脏帧区间闭包特解。
///
/// `apply(Vec3(start, end, child_span)) → Vec3(start, end, span)`：
/// - 输入 x = 变更起始帧，y = 变更结束帧，z = 子变更跨度（用于取闭包并集）；
/// - 输出 = 归一化后的闭区间三元组（start ≤ end，span = end − start + 1）。
///
/// 确定性：纯整数/浮点运算，无隐藏状态；非法输入（start > end）安全归一化
/// 为单帧区间（取 start）。
#[derive(Debug)]
pub struct DirtyRange {
    meta: StructureMeta,
}

impl Default for DirtyRange {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(15, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for DirtyRange {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    /// `Vec3(start, end, child_span) → Vec3(start, end, span)`（闭区间归一化）。
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        let Payload::Vec3(v) = x else { return Payload::Empty };
        let start = v.x;
        let end = v.y;
        // 归一化：start ≤ end；非法取 start 单帧。
        let (s, e) = if end >= start { (start, end) } else { (start, start) };
        Payload::Vec3(prism_math::Vec3::new(s, e, e - s + 1.0))
    }
}

prism_core::registry! {
    structure DirtyRange => StructureId::custom(15, env!("CARGO_CRATE_NAME"));
}

// ===========================================================================
// W1-A3：DSL v2 新节点消费面
// （audio→S6 谱解 / post→S2·S4 效果链 / light v2→S8 源项）
// ===========================================================================

// ---------------------------------------------------------------------------
// 消费错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）
// ---------------------------------------------------------------------------

/// 新节点消费错误（render 侧 W1 消费层）。
///
/// 错误码取 `R50xx` 段中未占用的 `R5030`–`R5038`（movie 成片路径占用
/// `R5001`–`R5016` / `R5020`，不冲突）。字段全部公开：movie 侧可按字段
/// 字面量转写为 `movie::RenderError`（该类型构造器不公开，逐字段映射）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumeError {
    /// 错误码（R50xx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：场景 / 节点 / 阶段。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl ConsumeError {
    /// 构造三段式错误。
    fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { code, location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl std::fmt::Display for ConsumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for ConsumeError {}

/// 新节点消费 `Result` 别名。
pub type ConsumeResult<T> = Result<T, ConsumeError>;

// ---------------------------------------------------------------------------
// 参数读取小助手（缺失 / 类型不符 / 非有限 → 缺省；与 movie 侧 p_f32 同口径）
// ---------------------------------------------------------------------------

/// 读 f32 参数（缺失 / 非 F32 / 非有限 → 缺省值）。
///
/// 值域校验（bpm > 0、gain ∈ [0,1] 等）留在 DSL codegen（P1018/P1019/P1022
/// 已拦）；本层只做"取得到、算得动"的防御，与 movie 侧材质钳制同纪律。
fn p_f32_or(params: &Params, key: &str, default: f32) -> f32 {
    match params.get(key) {
        Some(ParamValue::F32(v)) if v.is_finite() => *v,
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// audio → S6 谱解节点
// ---------------------------------------------------------------------------

/// 音频床（audio → S6 谱解节点）的消费描述。
///
/// DSL `audio { music { bpm gain duck } }` 投影为
/// `StructureId::SpectralDecomposition` + `Composition::Feed` 节点（「audio =
/// 谱解 + 流」模式标签）。渲染侧消费 = 从节点图取回校验后的静态参数：
/// render 管线出图不出声，声音合成归音频 / 混流阶段，本描述即两阶段的
/// 数据契约。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBed {
    /// 音乐床节点 id（S6）。
    pub node_id: NodeId,
    /// 每分钟拍数（BPM；缺省 120，与 codegen `DEFAULT_BPM` 同源）。
    pub bpm: f32,
    /// 音乐床增益（0..1；缺省 0.8，与 `DEFAULT_GAIN` 同源）。
    pub gain: f32,
    /// 闪避深度（dB；缺省 12，与 `DEFAULT_DUCK_DB` 同源）。
    pub duck: f32,
}

/// 单个 audio 节点 → 音乐床描述（S6 谱解；悬空 / 结构不符显式报错）。
fn consume_audio_bed(graph: &Graph, scene: &SceneV2, id: NodeId) -> ConsumeResult<AudioBed> {
    let node = graph.node(id).ok_or_else(|| {
        ConsumeError::new(
            "R5031",
            format!("场景 '{}' 的 audio 节点 {id}", scene.name),
            "AudioInfo 登记的节点 id 不在节点图中（悬空引用）".to_string(),
            "检查编译产物一致性；audio 块应投影出 S6 谱解节点并登记 node_ids",
        )
    })?;
    if node.structure != StructureId::SpectralDecomposition {
        return Err(ConsumeError::new(
            "R5032",
            format!("场景 '{}' 的 audio 节点 {id}", scene.name),
            format!(
                "audio 节点应为 S6 谱解结构（SpectralDecomposition），却得到 {}",
                node.structure
            ),
            "audio 块的节点由 codegen 投影为 SpectralDecomposition；手工构造产物请改用该结构",
        ));
    }
    Ok(AudioBed {
        node_id: id,
        bpm: p_f32_or(&node.params, "bpm", 120.0),
        gain: p_f32_or(&node.params, "gain", 0.8),
        duck: p_f32_or(&node.params, "duck", 12.0),
    })
}

// ---------------------------------------------------------------------------
// post → S2/S4 效果链（Chain 语义：按声明序串行 f∘g∘h）
// ---------------------------------------------------------------------------

/// post 效果种类（DSL post 块字段白名单 ↔ 结构族）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostEffectKind {
    /// 镜头光学泛光（S2 卷积族效果特解）。
    Bloom,
    /// 暗角（S2 卷积族效果特解）。
    Vignette,
    /// 色差（S2 卷积族效果特解）。
    Chromatic,
    /// 胶片颗粒（S4 双尺度颗粒特解）。
    Grain,
}

impl PostEffectKind {
    /// DSL 字段名 → 效果种类（表外 → `None`）。
    pub fn from_field(name: &str) -> Option<Self> {
        match name {
            "bloom" => Some(Self::Bloom),
            "vignette" => Some(Self::Vignette),
            "chromatic" => Some(Self::Chromatic),
            "grain" => Some(Self::Grain),
            _ => None,
        }
    }

    /// DSL 字段名（诊断 / 回写用）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bloom => "bloom",
            Self::Vignette => "vignette",
            Self::Chromatic => "chromatic",
            Self::Grain => "grain",
        }
    }

    /// 该效果在节点图中的结构（codegen 映射：grain → S4，其余 → S2）。
    fn expected_structure(self) -> StructureId {
        match self {
            Self::Grain => StructureId::DeterministicSampling,
            Self::Bloom | Self::Vignette | Self::Chromatic => StructureId::Convolution,
        }
    }
}

/// post 链单项：一个效果节点（声明序 = 链序）。
#[derive(Debug, Clone, PartialEq)]
pub struct PostStep {
    /// 效果节点 id。
    pub node_id: NodeId,
    /// 效果种类。
    pub kind: PostEffectKind,
    /// 效果强度（0..1；DSL 语义 0 = 关闭，1 = 最强）。
    pub strength: f32,
}

/// post 效果链（`Composition::Chain` 语义：按声明序串行 f∘g∘h）。
///
/// # 输入约定（已拍板决策）
/// 链的输入**隐式挂场景合成输出**——AE / Natron / Blender 通行约定：不要求
/// 用户在 DSL 声明输入，链首节点在图内无入边（codegen 留空不发明）。
/// **不开显式输入字段**；将来多层渲染（多合成层各自挂链）再开。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostChain {
    steps: Vec<PostStep>,
}

impl PostChain {
    /// 链上效果（声明序 = 应用序）。
    pub fn steps(&self) -> &[PostStep] {
        &self.steps
    }

    /// 效果个数。
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// 是否空链（未声明 post 块）。
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// 串行应用到帧缓冲（场景合成输出）：按声明序依次求值，第 k 步的输入
    /// 即第 k−1 步的输出（f∘g∘h，`Composition::Chain` 语义）。
    ///
    /// # 参数
    /// - `frame`：全局帧号（grain 的帧键：帧间去相关、同帧逐位复现、
    ///   任意 seek 可重算；其余效果与帧无关）。
    ///
    /// # 防御
    /// 形态非法的帧缓冲（`pixels.len() != w*h*4`，仅可能来自手工构造）
    /// 原样返回——不 panic、不写越界（`Framebuffer::new` 恒合法）。
    pub fn apply(&self, fb: &mut Framebuffer, frame: u64) {
        if fb.pixels.len() != (fb.width as usize) * (fb.height as usize) * 4 {
            return;
        }
        for step in &self.steps {
            // 0 = 关闭（DSL 语义）：强度 0 的步骤恒为恒等变换，跳过不改变
            // 结果（各效果在强度 0 时亦逐位恒等，跳过只为省算力）。
            if !(step.strength > 0.0) {
                continue;
            }
            match step.kind {
                PostEffectKind::Bloom => apply_bloom(fb, step.strength),
                PostEffectKind::Vignette => apply_vignette(fb, step.strength),
                PostEffectKind::Chromatic => apply_chromatic(fb, step.strength),
                PostEffectKind::Grain => apply_grain(fb, step.strength, step.node_id, frame),
            }
        }
    }
}

/// post 块 → 效果链（声明序 = 链序；逐节点校验结构 / 组合 / 参数 / 链边）。
fn consume_post_chain(graph: &Graph, scene: &SceneV2) -> ConsumeResult<PostChain> {
    let mut steps: Vec<PostStep> = Vec::new();
    for &id in &scene.post.node_ids {
        let node = graph.node(id).ok_or_else(|| {
            ConsumeError::new(
                "R5031",
                format!("场景 '{}' 的 post 节点 {id}", scene.name),
                "PostInfo 登记的节点 id 不在节点图中（悬空引用）".to_string(),
                "检查编译产物一致性；post 块应逐字段投影效果节点并登记 node_ids",
            )
        })?;
        // 效果判别：节点参数名 ∈ post 字段白名单（codegen 每节点恰一个，
        // 以字段名命名）；多个同时存在 = 手工产物歧义，显式拒绝。
        let mut found: Option<PostEffectKind> = None;
        for name in ["bloom", "grain", "vignette", "chromatic"] {
            if node.params.get(name).is_some() {
                if found.is_some() {
                    return Err(ConsumeError::new(
                        "R5033",
                        format!("场景 '{}' 的 post 节点 {id}", scene.name),
                        format!("post 效果节点同时带多个效果参数（已见 '{}' 与 '{name}'）", found.expect("已置位").as_str()),
                        "一个效果节点只承载一个效果（codegen 逐字段建点）；拆分或删除多余参数",
                    ));
                }
                found = PostEffectKind::from_field(name);
            }
        }
        let kind: PostEffectKind = found.ok_or_else(|| {
            ConsumeError::new(
                "R5033",
                format!("场景 '{}' 的 post 节点 {id}", scene.name),
                format!(
                    "post 效果节点没有可识别的效果参数（允许：bloom / grain / vignette / chromatic；实有：{}）",
                    node.params.names().iter().map(|n| n.as_str()).collect::<Vec<_>>().join(" / ")
                ),
                "post 效果节点的参数名必须是四种效果字段之一（codegen 以字段名建参）",
            )
        })?;
        // 结构族校验（grain → S4；bloom/vignette/chromatic → S2）。
        if node.structure != kind.expected_structure() {
            return Err(ConsumeError::new(
                "R5034",
                format!("场景 '{}' 的 post 节点 {id}（{}）", scene.name, kind.as_str()),
                format!(
                    "效果 '{}' 的节点结构应为 {}，却得到 {}",
                    kind.as_str(),
                    kind.expected_structure(),
                    node.structure
                ),
                "结构族映射：grain → DeterministicSampling（S4）；bloom/vignette/chromatic → Convolution（S2）",
            ));
        }
        // 组合校验：效果链节点必须是 Chain（串行 f∘g∘h）。
        if node.composition != Composition::Chain {
            return Err(ConsumeError::new(
                "R5035",
                format!("场景 '{}' 的 post 节点 {id}（{}）", scene.name, kind.as_str()),
                format!(
                    "post 效果节点的组合方式应为 Chain（串行链），却得到 {:?}",
                    node.composition
                ),
                "post 效果节点由 codegen 建为 Composition::Chain；手工构造产物请改用之",
            ));
        }
        // 强度：有限 + [0,1] 钳制（钳制口径与 movie 侧材质同纪律；
        // 非有限 / 类型不符显式拒绝——钳不了 NaN）。
        let strength: f32 = match node.params.get(kind.as_str()) {
            Some(ParamValue::F32(v)) if v.is_finite() => v.clamp(0.0, 1.0),
            Some(ParamValue::F32(v)) => {
                return Err(ConsumeError::new(
                    "R5036",
                    format!("场景 '{}' 的 post 节点 {id}（{}）", scene.name, kind.as_str()),
                    format!("效果强度必须为有限数，却得到 {v}"),
                    "post 强度是归一化比例（0 = 关闭，1 = 最强）；传有限数字",
                ));
            }
            other => {
                return Err(ConsumeError::new(
                    "R5036",
                    format!("场景 '{}' 的 post 节点 {id}（{}）", scene.name, kind.as_str()),
                    format!("效果强度应为 F32 数字，却得到 {:?}", other.map(|v| v.clone())),
                    "post 强度是归一化比例（0 = 关闭，1 = 最强）；传数字",
                ));
            }
        };
        // 链边校验：第 k>0 步的 0 号入边必须是上一步（声明序 = 链序的
        // 图内坐实；链首无入边——输入隐式挂场景合成输出）。
        if let Some(prev) = steps.last() {
            let wired: bool = graph
                .inputs_of(id)
                .iter()
                .any(|&(slot, src)| slot == 0 && src == prev.node_id);
            if !wired {
                return Err(ConsumeError::new(
                    "R5037",
                    format!("场景 '{}' 的 post 节点 {id}（{}）", scene.name, kind.as_str()),
                    format!(
                        "链边断链：声明序上本步应接在节点 {} 之后，但图内没有 (槽位 0) 的入边",
                        prev.node_id
                    ),
                    "Chain 语义 = 声明序串行；检查节点图连线（前一效果节点 → 本节点，槽位 0）",
                ));
            }
        }
        steps.push(PostStep { node_id: id, kind, strength });
    }
    // 摘要一致性：params 声明的效果必须都有节点——防"DSL 声明了效果却因
    // 节点丢失而静默不渲染"（codegen 逐字段建点，仅节点丢失时触发）。
    for key in scene.post.params.keys() {
        if !steps.iter().any(|s| s.kind.as_str() == key.as_str()) {
            return Err(ConsumeError::new(
                "R5038",
                format!("场景 '{}' 的 post 摘要", scene.name),
                format!("post 摘要声明了效果 '{key}'，但节点清单中没有对应效果节点"),
                "检查编译产物一致性；post 块应逐字段投影效果节点并登记 node_ids",
            ));
        }
    }
    Ok(PostChain { steps })
}

// ---------------------------------------------------------------------------
// post 效果求值（u8 显示域、整数/定点运算、零超越函数——D-29 确定性纪律）
// ---------------------------------------------------------------------------

/// 域口径（诚实声明）：效果作用于**场景合成输出**（RGBA8 帧缓冲）的显示域，
/// 非 `prism-postfx` 的 f16 线性域——后者经 `TileEffect` 挂在 tile 编排层，
/// 且 postfx → render 的依赖方向使本层不能反向依赖（两域并存，不静默混用）。
/// 映射强度 → 效果参数为固定标定常量（确定性，无隐藏状态）。

/// bloom 亮部阈值（u8；0.6 × 255）。
const BLOOM_THRESHOLD: u8 = 153;
/// bloom 扩散半径（px；全分辨率可分离 box blur——平移不变，与 postfx 同口径）。
const BLOOM_RADIUS: u32 = 2;
/// chromatic 边缘最大位移（px；强度 1 = 最强）。
const CHROMA_MAX_SHIFT: i64 = 4;
/// grain 强度 1 对应的幅度（u8；±64/255 ≈ ±25% 亮度扰动）。
const GRAIN_AMPLITUDE: f32 = 64.0;

/// bloom：亮部提取（逐通道 `max(0, c − t)`）→ 可分离 box blur 扩散 →
/// 显示域整数加算 `out = c + strength · blurred`。
fn apply_bloom(fb: &mut Framebuffer, strength: f32) {
    let w: usize = fb.width as usize;
    let h: usize = fb.height as usize;
    // 步 1：亮部提取。
    let mut bright: Vec<u8> = vec![0u8; w * h * 4];
    for (b, p) in bright.chunks_exact_mut(4).zip(fb.pixels.chunks_exact(4)) {
        for c in 0..3 {
            b[c] = p[c].saturating_sub(BLOOM_THRESHOLD);
        }
        b[3] = 0;
    }
    // 步 2：扩散（两趟固定序；窗口边缘钳制复制）。
    let blurred: Vec<u8> = box_blur_u8(&bright, w, h, BLOOM_RADIUS);
    // 步 3：加算合成。
    for (p, q) in fb.pixels.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
        for c in 0..3 {
            p[c] = (p[c] as f32 + strength * q[c] as f32).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// 可分离 box blur（RGBA 四通道独立；边缘钳制复制；两趟固定序 = 确定性）。
fn box_blur_u8(src: &[u8], w: usize, h: usize, radius: u32) -> Vec<u8> {
    let r: i64 = radius as i64;
    let win: f32 = (2 * r + 1) as f32;
    let mut tmp: Vec<u8> = vec![0u8; w * h * 4];
    let mut out: Vec<u8> = vec![0u8; w * h * 4];
    if w == 0 || h == 0 {
        return out;
    }
    // 横向（x ± r 钳制）。
    for y in 0..h {
        for x in 0..w {
            let mut acc: [f32; 4] = [0.0; 4];
            for k in -r..=r {
                let sx: usize = (x as i64 + k).clamp(0, w as i64 - 1) as usize;
                let base: usize = (y * w + sx) * 4;
                for c in 0..4 {
                    acc[c] += src[base + c] as f32;
                }
            }
            let o: usize = (y * w + x) * 4;
            for c in 0..4 {
                tmp[o + c] = (acc[c] / win).round() as u8;
            }
        }
    }
    // 纵向（y ± r 钳制）。
    for y in 0..h {
        for x in 0..w {
            let mut acc: [f32; 4] = [0.0; 4];
            for k in -r..=r {
                let sy: usize = (y as i64 + k).clamp(0, h as i64 - 1) as usize;
                let base: usize = (sy * w + x) * 4;
                for c in 0..4 {
                    acc[c] += tmp[base + c] as f32;
                }
            }
            let o: usize = (y * w + x) * 4;
            for c in 0..4 {
                out[o + c] = (acc[c] / win).round() as u8;
            }
        }
    }
    out
}

/// vignette：径向平方衰减 `out = c · (1 − strength · r²)`（r² 归一化到角点
/// = 1；免 sqrt——只用精确运算）。alpha 不动。
fn apply_vignette(fb: &mut Framebuffer, strength: f32) {
    let w: f32 = fb.width as f32;
    let h: f32 = fb.height as f32;
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    for y in 0..fb.height {
        for x in 0..fb.width {
            let nx: f32 = (x as f32 + 0.5) / w - 0.5;
            let ny: f32 = (y as f32 + 0.5) / h - 0.5;
            let r2: f32 = (nx * nx + ny * ny) / 0.5; // 角点 = 1
            let factor: f32 = (1.0 - strength * r2).max(0.0);
            let idx: usize = ((y * fb.width + x) * 4) as usize;
            for c in 0..3 {
                fb.pixels[idx + c] =
                    (fb.pixels[idx + c] as f32 * factor).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// chromatic：径向通道错位（R 外移 / B 内移，位移量 ∝ 强度，最大
/// [`CHROMA_MAX_SHIFT`] px；最近邻采样 + 坐标钳制 = 确定性）。G / alpha 不动。
fn apply_chromatic(fb: &mut Framebuffer, strength: f32) {
    if fb.width == 0 || fb.height == 0 {
        return;
    }
    let k: i64 = (strength * CHROMA_MAX_SHIFT as f32).round() as i64;
    if k <= 0 {
        return;
    }
    let w: i64 = fb.width as i64;
    let h: i64 = fb.height as i64;
    let src: Vec<u8> = fb.pixels.clone();
    for y in 0..h {
        for x in 0..w {
            let dx: f32 = (x as f32 + 0.5) / w as f32 - 0.5;
            let dy: f32 = (y as f32 + 0.5) / h as f32 - 0.5;
            let ox: i64 = (dx * 2.0 * k as f32).round() as i64;
            let oy: i64 = (dy * 2.0 * k as f32).round() as i64;
            let idx: usize = ((y * w + x) * 4) as usize;
            let rx: i64 = (x + ox).clamp(0, w - 1);
            let ry: i64 = (y + oy).clamp(0, h - 1);
            let bx: i64 = (x - ox).clamp(0, w - 1);
            let by: i64 = (y - oy).clamp(0, h - 1);
            fb.pixels[idx] = src[((ry * w + rx) * 4) as usize];
            fb.pixels[idx + 2] = src[((by * w + bx) * 4 + 2) as usize];
        }
    }
}

/// grain：加性颗粒 `out = c + strength · A · n(x, y, frame)`，n ∈ [−1, 1)
/// 由整数 hash 派生（seed = 节点 id：每颗粒节点独立图案；帧号进 hash：
/// 帧间去相关、同帧逐位复现、任意 seek 可重算）。RGB 同噪（亮度颗粒）。
fn apply_grain(fb: &mut Framebuffer, strength: f32, seed: u64, frame: u64) {
    for y in 0..fb.height {
        for x in 0..fb.width {
            let n: f32 = hash3_f32(seed, x as u64, y as u64, frame) * 2.0 - 1.0;
            let delta: f32 = (strength * GRAIN_AMPLITUDE * n).round();
            let idx: usize = ((y * fb.width + x) * 4) as usize;
            for c in 0..3 {
                fb.pixels[idx + c] =
                    (fb.pixels[idx + c] as f32 + delta).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// 确定性整数 hash（splitmix64 终结器，纯整数运算）→ `[0, 1)` f32。
///
/// 与 `prism-postfx` grain 的哈希形状（seed, x, y, frame）一致；本地实现
/// 而非引 `prism-noise`——本 crate 零新增依赖纪律（orchestrator 测试助手
/// 同构造先例）。无私开 RNG：随机性一律整数 hash 派生（D-29）。
fn hash3_f32(seed: u64, x: u64, y: u64, frame: u64) -> f32 {
    let mut z: u64 = seed
        ^ x.wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ y.wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0 // 24 位尾数 → [0, 1)
}

// ---------------------------------------------------------------------------
// light v2 → S8 源项节点
// ---------------------------------------------------------------------------

/// 光源源项（light v2 → S8 辐射传输场景源项节点）的消费描述。
///
/// DSL `stage3d light` 投影为 `RadiativeTransfer` + `Feed` 节点（「light =
/// 场景源项 g」模式标签）：direction 逐分量透传（不取模不归一化——v1 纪律），
/// color 与 material.albedo 同族按 0..1 归一化（C17），intensity 原值。
#[derive(Debug, Clone, PartialEq)]
pub struct LightTerm {
    /// 光源节点 id（S8）。
    pub node_id: NodeId,
    /// 入射方向（透传，不归一化）。
    pub direction: Vec3,
    /// 颜色（0..1）。
    pub color: Vec3,
    /// 强度（原值）。
    pub intensity: f32,
}

impl LightTerm {
    /// 折算为渲染管线的原生光源（[`prism_core::Light`]）。
    pub fn to_core_light(&self) -> Light {
        Light::new(self.direction, self.color, self.intensity)
    }
}

/// 节点 → 光源源项（判别式，与 movie 侧 material 判别互斥）：
/// S8 辐射传输结构 + 带 `dir_x`（光源签名）+ 不带 `albedo_r`（材质签名）。
fn light_term_of(node: &prism_core::Node) -> Option<LightTerm> {
    if node.structure != StructureId::RadiativeTransfer {
        return None;
    }
    if node.params.get("albedo_r").is_some() {
        return None; // material（S8 传输介质），不是光源
    }
    if node.params.get("dir_x").is_none() {
        return None; // 无光源签名
    }
    Some(LightTerm {
        node_id: node.id,
        direction: Vec3::new(
            p_f32_or(&node.params, "dir_x", 0.0),
            p_f32_or(&node.params, "dir_y", 0.0),
            p_f32_or(&node.params, "dir_z", 0.0),
        ),
        color: Vec3::new(
            p_f32_or(&node.params, "color_r", 1.0),
            p_f32_or(&node.params, "color_g", 1.0),
            p_f32_or(&node.params, "color_b", 1.0),
        ),
        intensity: p_f32_or(&node.params, "intensity", 1.0),
    })
}

// ---------------------------------------------------------------------------
// 场景消费入口
// ---------------------------------------------------------------------------

/// 单场景的三种新节点消费结果。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneConsumption {
    /// audio 块投影的音乐床（S6；无 audio 块 = 空）。
    pub audio: Vec<AudioBed>,
    /// post 效果链（S2/S4；无 post 块 = 空链）。
    pub post: PostChain,
    /// light v2 源项（S8；按节点 id 升序 = 声明序）。
    pub lights: Vec<LightTerm>,
}

impl SceneConsumption {
    /// 光源源项 → 原生 [`Light`] 序列（按节点 id 升序 = 声明序）。
    pub fn core_lights(&self) -> Vec<Light> {
        self.lights.iter().map(LightTerm::to_core_light).collect()
    }
}

/// 消费单场景的三种新节点（audio / post / light v2）——W1 消费面入口。
///
/// # 发现路径（与 codegen 登记缺口对齐，不静默丢弃）
/// - audio：`SceneV2::audio.node_ids`（AudioInfo 登记）→ 图内取节点校验；
/// - post：`SceneV2::post.node_ids`（PostInfo 登记，声明序 = 链序）→ 逐节点
///   校验效果参数 / 结构族 / Chain 组合 / 链边；
/// - light v2：节点图扫描（S8 源项不进任何 Info 清单）→ 场景 id 窗口归属
///   （窗口推演见 [`scene_id_window`]）。
///
/// # 错误（显式，不静默）
/// 下标越界（R5030）/ 登记悬空（R5031）/ 结构不符（R5032、R5034）/ 参数
/// 不可识别或非法（R5033、R5036）/ 组合非 Chain（R5035）/ 链边断链
/// （R5037）/ 摘要与节点清单不一致（R5038）。
///
/// # 边界（诚实记录）
/// - 本层**不**消费 `SceneV2::node_ids` 内的节点（camera / material / sdf /
///   text 归 movie 侧 `build_models`；particles / sim / compose 等暂不支持
///   节点由彼处按 R5010 报错）——本层只取三种新族，图内其他节点原样忽略；
/// - audio / post 节点按 codegen 现状无通道绑定（静态参数）；若将来通道
///   化，绑定解析需随 movie 侧通道表下沉到本层（W1 不做）。
pub fn consume_scene(compile: &CompileV2Output, scene_index: usize) -> ConsumeResult<SceneConsumption> {
    let scene: &SceneV2 = compile.scenes.get(scene_index).ok_or_else(|| {
        ConsumeError::new(
            "R5030",
            format!("compile.scenes[{scene_index}]"),
            format!("场景下标越界：编译产物只有 {} 个场景", compile.scenes.len()),
            "scene_index 取 0..scenes.len()；消费逐场景进行",
        )
    })?;

    // ---- audio → S6 谱解节点 ----
    let mut audio: Vec<AudioBed> = Vec::new();
    if let Some(info) = scene.audio.as_ref() {
        for &id in &info.node_ids {
            audio.push(consume_audio_bed(&compile.graph, scene, id)?);
        }
    }

    // ---- post → S2/S4 效果链 ----
    let post: PostChain = consume_post_chain(&compile.graph, scene)?;

    // ---- light v2 → S8 源项（图扫 + 场景 id 窗口归属）----
    let (lo, hi) = scene_id_window(compile, scene_index);
    // 首占优先：更早场景的窗口已认领的 light 不重复归属（hi 单调不减 ⇒
    // 等效下界 = max(lo, 上一场景 hi)）。
    let prev_hi: NodeId =
        if scene_index == 0 { 0 } else { scene_id_window(compile, scene_index - 1).1 };
    let lo: NodeId = lo.max(prev_hi);
    let mut lights: Vec<LightTerm> = Vec::new();
    if lo < hi {
        for node in compile.graph.nodes() {
            if node.id < lo || node.id >= hi {
                continue;
            }
            if let Some(term) = light_term_of(node) {
                lights.push(term);
            }
        }
    }

    Ok(SceneConsumption { audio, post, lights })
}

/// 场景在编译产物中的显式登记 id（node_ids ∪ audio.node_ids ∪ post.node_ids）。
fn registered_ids(scene: &SceneV2) -> impl Iterator<Item = NodeId> + '_ {
    scene
        .node_ids
        .iter()
        .copied()
        .chain(scene.audio.as_ref().into_iter().flat_map(|a| a.node_ids.iter().copied()))
        .chain(scene.post.node_ids.iter().copied())
}

/// 场景 id 窗口推演：light v2 节点不进任何 Info 清单，按 id 区间归属场景。
///
/// # 不变式（codegen 分配纪律）
/// 节点 id 全局递增分配且按场景顺序连续：场景 i 的全部节点 id 小于场景
/// j>i 的全部节点 id。于是：
/// - 下界 `lo` = 1 + 更早场景的最大登记 id——sound（不丢本场景领先的
///   light 节点：本场景块起点 > 更早场景全部 id ≥ 其登记 id）；
/// - 上界 `hi` = 更晚场景的最小登记 id，无更晚场景则图内最大 id + 1——
///   sound（本场景全部 id < 更晚场景全部 id）。
///
/// # 已知边界（诚实记录，不静默）
/// 退化形态「某场景的 stage3d 只含 light（无 camera/sdf/material）且声明在
/// 图层之前」时，其 light 节点可能被归到前一场景窗口（两个相邻场景都无登记
/// id 时不可判别）。DSL 常见形态（stage3d 带 camera，camera 先于 light 分配
/// 且登记在 node_ids）下归属精确。
fn scene_id_window(compile: &CompileV2Output, scene_index: usize) -> (NodeId, NodeId) {
    let mut lo: NodeId = 1;
    for earlier in &compile.scenes[..scene_index] {
        for id in registered_ids(earlier) {
            lo = lo.max(id + 1);
        }
    }
    let mut hi: NodeId = compile.graph.nodes().map(|n| n.id).max().unwrap_or(0) + 1;
    for later in &compile.scenes[scene_index + 1..] {
        for id in registered_ids(later) {
            hi = hi.min(id);
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::NodeRegistry;
    use prism_math::Vec3;

    #[test]
    fn dirty_range_closure_is_deterministic_and_normalized() {
        let spec = DirtyRange::default();
        let ctx = EvalCtx::new(0, 1, 0);
        // 区间 [40, 40] → span 1。
        let out = spec.apply(&Payload::Vec3(Vec3::new(40.0, 40.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(40.0, 40.0, 1.0)));
        // 区间 [10, 19] → span 10。
        let out = spec.apply(&Payload::Vec3(Vec3::new(10.0, 19.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(10.0, 19.0, 10.0)));
        // 非法（start > end）→ 归一化为单帧 start。
        let out = spec.apply(&Payload::Vec3(Vec3::new(9.0, 3.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(9.0, 9.0, 1.0)));
        // 非 Vec3 输入 → Empty（类型安全，不 panic）。
        assert_eq!(spec.apply(&Payload::Empty, &ctx), Payload::Empty);
    }

    #[test]
    fn registry_collects_dirty_range() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 1);
        let id = StructureId::custom(15, "prism_render");
        assert!(reg.contains(&id), "DirtyRange 应按 id 汇入注册表");
        let inst = reg.resolve_by_name("DirtyRange").expect("应按名解析");
        let ctx = EvalCtx::new(0, 1, 0);
        assert_eq!(
            inst.apply(&Payload::Vec3(Vec3::new(0.0, 4.0, 0.0)), &ctx),
            Payload::Vec3(Vec3::new(0.0, 4.0, 5.0))
        );
    }

    // -----------------------------------------------------------------------
    // W1-A3：post 效果求值单元测试（u8 显示域、确定性、零超越函数）
    // -----------------------------------------------------------------------

    use prism_gpu::Framebuffer;

    /// 由步骤直接构链（测试与消费层同模块，可触达私有字段）。
    fn chain(steps: Vec<PostStep>) -> PostChain {
        PostChain { steps }
    }

    fn step(id: NodeId, kind: PostEffectKind, strength: f32) -> PostStep {
        PostStep { node_id: id, kind, strength }
    }

    /// 非均匀测试帧：双轴渐变 + 右上角 6×6 亮块（bloom 的输入成分）。
    fn test_fb(w: u32, h: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                fb.pixels[i] = (x * 255 / w.max(1)) as u8;
                fb.pixels[i + 1] = (y * 255 / h.max(1)) as u8;
                fb.pixels[i + 2] = ((x + y) * 255 / (w + h).max(1)) as u8;
                fb.pixels[i + 3] = 255;
                if x >= w - 6 && y < 6 {
                    fb.pixels[i] = 250;
                    fb.pixels[i + 1] = 250;
                    fb.pixels[i + 2] = 250;
                }
            }
        }
        fb
    }

    #[test]
    fn vignette_darkens_corners_more_than_center() {
        let mut fb = test_fb(33, 33);
        let before_center = fb.pixels[((16 * 33 + 16) * 4) as usize];
        let before_near_corner = fb.pixels[((2 * 33 + 2) * 4) as usize];
        assert!(before_center > 100, "夹具应中心亮");
        assert!(before_near_corner > 0, "近角点应有非零输入");
        chain(vec![step(1, PostEffectKind::Vignette, 1.0)]).apply(&mut fb, 0);
        // 强度 1：近角点因子 = 1 − r²（r² ≈ 0.72）≈ 0.28 → 显著变暗。
        let after_near_corner = fb.pixels[((2 * 33 + 2) * 4) as usize];
        assert!(
            after_near_corner < before_near_corner,
            "近角点应变暗（{before_near_corner} → {after_near_corner}）"
        );
        // 正中心径向距离恰为 0 ⇒ 因子 1 ⇒ 逐位不变。
        assert_eq!(
            fb.pixels[((16 * 33 + 16) * 4) as usize],
            before_center,
            "正中心应逐位不变"
        );
        // alpha 不动。
        assert_eq!(fb.pixels[3], 255);
    }

    #[test]
    fn bloom_brightens_region_around_bright_block() {
        let mut fb = test_fb(33, 33);
        // 亮块左邻一像素（夹具中该处较暗，泛光应把它提亮）。
        let probe = ((3 * 33 + 26) * 4) as usize; // 亮块在 x≥27, y<6；取 (26,3)
        let before = fb.pixels[probe];
        chain(vec![step(1, PostEffectKind::Bloom, 1.0)]).apply(&mut fb, 0);
        assert!(
            fb.pixels[probe] > before,
            "亮块邻域应被泛光提亮（{before} → {}）",
            fb.pixels[probe]
        );
        // 块内本身也更亮（加算自己的扩散）。
        let inside = ((2 * 33 + 29) * 4) as usize;
        assert!(fb.pixels[inside] >= 250, "块内应保持高亮（得到 {}）", fb.pixels[inside]);
    }

    #[test]
    fn grain_is_deterministic_and_frame_keyed() {
        let mut a = test_fb(24, 24);
        let mut b = test_fb(24, 24);
        let c0 = chain(vec![step(7, PostEffectKind::Grain, 0.5)]);
        c0.apply(&mut a, 0);
        c0.apply(&mut b, 0);
        assert_eq!(a.pixels, b.pixels, "同帧两遍必须逐位一致（无私开 RNG）");
        // 帧号进哈希 ⇒ 帧间去相关（颗粒"活"）。
        let mut c = test_fb(24, 24);
        c0.apply(&mut c, 1);
        assert_ne!(a.pixels, c.pixels, "不同帧的颗粒图案必须不同");
        // 节点 id 进种子 ⇒ 不同节点不同图案。
        let mut d = test_fb(24, 24);
        chain(vec![step(8, PostEffectKind::Grain, 0.5)]).apply(&mut d, 0);
        assert_ne!(a.pixels, d.pixels, "不同种子（节点 id）的颗粒图案必须不同");
        // alpha 不受影响。
        for p in a.pixels.chunks_exact(4) {
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn chromatic_shifts_channels_and_keeps_center_intact() {
        // 奇数边长 ⇒ 正中心像素径向偏移恰为 0（dx = dy = 0）。
        let mut fb = test_fb(33, 33);
        let center = ((16 * 33 + 16) * 4) as usize;
        let before_center = [fb.pixels[center], fb.pixels[center + 1], fb.pixels[center + 2]];
        chain(vec![step(1, PostEffectKind::Chromatic, 1.0)]).apply(&mut fb, 0);
        assert_eq!(
            [fb.pixels[center], fb.pixels[center + 1], fb.pixels[center + 2]],
            before_center,
            "正中心像素径向偏移为 0，应逐位不变"
        );
        assert_ne!(fb.pixels, test_fb(33, 33).pixels, "边缘区应发生通道错位");
        // 确定性：两遍一致。
        let mut again = test_fb(33, 33);
        chain(vec![step(1, PostEffectKind::Chromatic, 1.0)]).apply(&mut again, 0);
        assert_eq!(fb.pixels, again.pixels);
    }

    #[test]
    fn post_chain_zero_strength_is_identity() {
        // 0 = 关闭（DSL 语义）：全零强度链必须逐位恒等。
        let mut fb = test_fb(20, 12);
        let before = fb.pixels.clone();
        chain(vec![
            step(1, PostEffectKind::Bloom, 0.0),
            step(2, PostEffectKind::Grain, 0.0),
            step(3, PostEffectKind::Vignette, 0.0),
            step(4, PostEffectKind::Chromatic, 0.0),
        ])
        .apply(&mut fb, 3);
        assert_eq!(fb.pixels, before, "强度 0 的效果必须恒等（关闭）");
    }

    #[test]
    fn post_chain_apply_is_inert_on_malformed_framebuffer() {
        // 手工构造的畸形帧缓冲（pixels 与 w*h*4 不符）：不 panic、原样返回。
        let mut fb = Framebuffer { width: 8, height: 8, pixels: vec![1, 2, 3, 4] };
        chain(vec![
            step(1, PostEffectKind::Bloom, 0.5),
            step(2, PostEffectKind::Grain, 0.5),
        ])
        .apply(&mut fb, 0);
        assert_eq!(fb.pixels, vec![1, 2, 3, 4]);
    }

    #[test]
    fn hash3_f32_stays_in_unit_range() {
        for seed in [0u64, 1, 42, u64::MAX] {
            for x in 0..8u64 {
                for y in 0..8u64 {
                    for frame in [0u64, 1, 999] {
                        let v = hash3_f32(seed, x, y, frame);
                        assert!((0.0..1.0).contains(&v), "hash 输出须在 [0,1)：{v}");
                    }
                }
            }
        }
    }
}
