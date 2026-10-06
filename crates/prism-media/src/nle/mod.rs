//! NLE 非线性编辑（U14；账六）—— 多轨时间线求值。
//!
//! 语义权威：兼容性宪章对照表 #4（Timeline / NLE，剪辑 = 时间线上的裁剪与
//! 叠加）、架构总设计 v2.4 S5（重参数化）/ S9（轨道编排）、白皮书 §19（视频
//! 时间映射）、§18（合成/遮罩族）、§26（固定形状归约）。
//!
//! # 数学路径（账本已定）
//!
//! | 编辑语义 | 数学路径 | 落地文件 |
//! |---|---|---|
//! | 裁剪 / 变速 / 倒放 | **S5 时间映射特解**（重参数化） | [`timemap`] |
//! | 转场 | **S2 卷积 / 混合特解**（重叠区间混合） | [`transition`] |
//! | 多轨编排 | **S9 轨道编排 + 音频固定形状归约** | 本文件 + [`audio`] |
//!
//! 求值链：`t_tl --TimeMap2--> t_src --MediaClipSource--> Frame --composite--> Frame`。
//! 全链纯函数、无状态、无 libm、无网络——**同输入逐位可复现**（确定性第一铁律）。
//!
//! # 遍历顺序的确定（消序用排序，§26 工程四律 2）
//!
//! 两处"本可以乱序"的地方都被钉死：
//!
//! 1. **轨道**：`Vec<Track>` 按 `TrackIndex` 升序（结构序即索引序）；
//! 2. **同轨 Clip**：`add_clip` 按**规范键**（`timeline_in → src_in → src_out →
//!    倒放 → 源 id → 不透明度 → 混合模式`）插入有序位置，故**与添加顺序无关**。
//!    规范键相同 ⟹ 两 Clip 的可观测状态完全相同（同一源 id 契约下），故并存的
//!    tie-break 不改变求值结果——这是"不同添加顺序构造等价 timeline 结果一致"
//!    的根据（源 id 由实现者给定的内容派生值，非插入序号）。
//!
//! # 独奏 / 静音语义
//!
//! 全局独奏（任一轨 `solo`）：**仅** `solo` 轨参与，其余轨等价静音；无任何
//! `solo` 时全部轨参与。判定为对 `TrackIndex` 升序的一次线性扫描（无序集合先
//! 排序再判定），故确定。
//!
//! # 边界（C18 永不 panic）
//!
//! 零长度 clip、零/负速率、超范围求值、非法轨道下标、空源列表 → 全部返回
//! 三段式 [`NleError`]（`NLE12xx` 段），不 `unwrap`、不越界、不 `panic!`。
//!
//! 本模块零第三方依赖，只用工作区内部 crate + Rust 标准库。

use std::fmt;
use std::rc::Rc;

use prism_audio::PcmBuffer;

pub mod audio;
pub mod composite;
pub mod frame;
pub mod timemap;
pub mod transition;

pub use audio::{mixdown, AudioTrackInput};
pub use composite::{composite, BlendMode, Transform};
pub use frame::Frame;
pub use timemap::{RateCurve, TimeMap2};
pub use transition::{Transition, TransitionEasing, TransitionKind, WipeDirection};

// ---------------------------------------------------------------------------
// 错误（C18 三段式：在哪 / 什么事 / 怎么改）
// ---------------------------------------------------------------------------

/// NLE 域错误（宪章 C18 三段式；C19 必含建议段）。
///
/// 错误码分段：`NLE10xx` 时间映射退化点（[`timemap`]）、`NLE11xx` 音频归约
/// （[`audio`]）、`NLE12xx` 时间线编排与求值（本文件）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NleError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl NleError {
    /// 构造一个三段式 NLE 错误。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { code, location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl fmt::Display for NleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for NleError {}

/// NLE 域 `Result` 别名。
pub type NleResult<T> = Result<T, NleError>;

// ---------------------------------------------------------------------------
// 标识
// ---------------------------------------------------------------------------

/// 轨道索引（结构序即确定序：升序遍历）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackIndex(pub u32);

impl TrackIndex {
    /// 裸索引值。
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Clip 标识（轨道内唯一，单调发放；用于转场引用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClipId(pub u32);

/// 转场标识（时间线内唯一，单调发放）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransitionId(pub u32);

/// 轨道类型（视频轨参与画面合成，音频轨只进 mixdown）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// 视频轨（可同时带音频，进画面合成）。
    Video,
    /// 音频轨（只进 mixdown，不参与画面合成）。
    Audio,
}

// ---------------------------------------------------------------------------
// 源抽象
// ---------------------------------------------------------------------------

/// Clip 源：按**源内时间**（秒）产出一帧。
///
/// # `source_id` 契约
///
/// 返回值必须是**内容派生**的稳定 id（如素材哈希、路径哈希），**不得**取插入
/// 序号或随机值——[`Timeline::add_clip`] 的规范排序键含该项：若两个不同源返回
/// 同一 id，"不同添加顺序结果一致"不再成立（诚实的边界：契约由实现者遵守，
/// 类型系统无法强制）。
pub trait MediaClipSource {
    /// 内容派生的稳定源 id（见类型文档的契约）。
    fn source_id(&self) -> u32;

    /// 源素材时长（秒，`> 0`）。
    fn duration(&self) -> f32;

    /// 源内时间 `t_src`（秒）处的帧；越界由实现自行钳制。
    fn frame_at(&self, t_src: f32) -> NleResult<Frame>;
}

/// 纯色源（全片恒定色）——最小可用源，测试与占位用。
#[derive(Debug, Clone, PartialEq)]
pub struct SolidSource {
    id: u32,
    duration: f32,
    frame: Frame,
}

impl SolidSource {
    /// 构造纯色源（`duration ≤ 0` 钳到 `f32::MIN_POSITIVE` 以免除零）。
    pub fn new(id: u32, duration: f32, width: u32, height: u32, rgba: [f32; 4]) -> Self {
        let mut frame = Frame::new(width, height);
        for y in 0..height {
            for x in 0..width {
                frame.set(x, y, rgba);
            }
        }
        Self { id, duration: if duration > 0.0 { duration } else { f32::MIN_POSITIVE }, frame }
    }
}

impl MediaClipSource for SolidSource {
    fn source_id(&self) -> u32 {
        self.id
    }

    fn duration(&self) -> f32 {
        self.duration
    }

    fn frame_at(&self, _t_src: f32) -> NleResult<Frame> {
        Ok(self.frame.clone())
    }
}

/// 斜坡源：帧的 R 通道编码**归一化源内时间** `t_src / duration`。
///
/// 用途：让裁剪 / 变速 / 倒放的求值结果**可从像素直接读出**——`eval_at` 的
/// 断言无需窥探内部映射即可验证时间映射正确性。
#[derive(Debug, Clone, PartialEq)]
pub struct RampSource {
    id: u32,
    duration: f32,
    width: u32,
    height: u32,
}

impl RampSource {
    /// 构造斜坡源（`duration ≤ 0` 钳到 `f32::MIN_POSITIVE`）。
    pub fn new(id: u32, duration: f32, width: u32, height: u32) -> Self {
        Self {
            id,
            duration: if duration > 0.0 { duration } else { f32::MIN_POSITIVE },
            width,
            height,
        }
    }
}

impl MediaClipSource for RampSource {
    fn source_id(&self) -> u32 {
        self.id
    }

    fn duration(&self) -> f32 {
        self.duration
    }

    fn frame_at(&self, t_src: f32) -> NleResult<Frame> {
        let u = if self.duration > 0.0 { (t_src / self.duration).clamp(0.0, 1.0) } else { 0.0 };
        let mut f = Frame::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                f.set(x, y, [u, 0.0, 0.0, 1.0]);
            }
        }
        Ok(f)
    }
}

// ---------------------------------------------------------------------------
// Clip / Track
// ---------------------------------------------------------------------------

/// 时间线上的一段剪辑：源引用 + 时间映射 + 变换。
#[derive(Clone)]
pub struct Clip {
    id: ClipId,
    source: Rc<dyn MediaClipSource>,
    map: TimeMap2,
    transform: Transform,
}

impl Clip {
    /// Clip 标识。
    pub fn id(&self) -> ClipId {
        self.id
    }

    /// 时间映射（裁剪 / 变速 / 倒放）。
    pub fn time_map(&self) -> &TimeMap2 {
        &self.map
    }

    /// 变换（不透明度 + 混合模式）。
    pub fn transform(&self) -> Transform {
        self.transform
    }

    /// 源引用。
    pub fn source(&self) -> &Rc<dyn MediaClipSource> {
        &self.source
    }

    /// 时间线区间终点（秒）。
    pub fn timeline_end(&self) -> f32 {
        self.map.timeline_in() + self.map.timeline_duration()
    }

    /// 时刻 `t` 是否落在本 Clip 的时间线区间 `[in, end)` 内。
    pub fn covers(&self, t: f32) -> bool {
        t >= self.map.timeline_in() && t < self.timeline_end()
    }

    /// 求值：时间线时间 → 源内时间 → 帧。
    pub fn eval(&self, t: f32) -> NleResult<Frame> {
        let t_src = self.map.source_at(t);
        self.source.frame_at(t_src)
    }

    /// 规范排序键（`f32` 经定点量化转 `i64` 以取得**全序**，避免 `partial_cmp`
    /// 的 `None` 分支；NaN 归到最小）。含 `source_id` ⟹ 键相同即可观测状态相同。
    fn canonical_key(&self) -> (i64, i64, i64, u32, u32, i64, u32) {
        (
            key_f32(self.map.timeline_in()),
            key_f32(self.map.src_in()),
            key_f32(self.map.src_out()),
            u32::from(self.map.is_reversed()),
            self.source.source_id(),
            key_f32(self.transform.opacity),
            self.transform.blend as u32,
        )
    }
}

/// `f32` → 全序 `i64` 键（定点量化：乘 4096 后四舍五入；NaN → `i64::MIN`）。
#[inline]
fn key_f32(v: f32) -> i64 {
    if v.is_nan() {
        i64::MIN
    } else {
        // `as i64` 对越界 f32 是饱和转换（Rust 定义行为，非 UB）。
        (v.clamp(-1.0e6, 1.0e6) * 4096.0).round() as i64
    }
}

impl fmt::Debug for Clip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Clip")
            .field("id", &self.id)
            .field("source_id", &self.source.source_id())
            .field("timeline_in", &self.map.timeline_in())
            .field("timeline_dur", &self.map.timeline_duration())
            .field("src_in", &self.map.src_in())
            .field("src_out", &self.map.src_out())
            .field("reversed", &self.map.is_reversed())
            .field("transform", &self.transform)
            .finish()
    }
}

/// 一条轨道：若干 Clip（**恒按规范键有序**）+ 静音 / 独奏 / 合成参数。
#[derive(Debug, Clone)]
pub struct Track {
    index: TrackIndex,
    kind: TrackKind,
    muted: bool,
    solo: bool,
    transform: Transform,
    clips: Vec<Clip>,
    audio: Option<AudioTrackInput>,
}

impl Track {
    /// 轨道索引。
    pub fn index(&self) -> TrackIndex {
        self.index
    }

    /// 轨道类型。
    pub fn kind(&self) -> TrackKind {
        self.kind
    }

    /// 是否静音。
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// 是否独奏。
    pub fn is_solo(&self) -> bool {
        self.solo
    }

    /// 轨道级变换（叠加到本轨合成结果之上）。
    pub fn transform(&self) -> Transform {
        self.transform
    }

    /// Clip 列表（按规范键升序的只读视图）。
    pub fn clips(&self) -> &[Clip] {
        &self.clips
    }

    /// 轨道音频输入（`None` = 本轨不参与 mixdown）。
    pub fn audio(&self) -> Option<&AudioTrackInput> {
        self.audio.as_ref()
    }

    /// 按规范键插入位置（稳定插入：相等键落在既有相等键之后）。
    fn insert_index_for(&self, key: &(i64, i64, i64, u32, u32, i64, u32)) -> usize {
        self.clips.partition_point(|c| c.canonical_key() <= *key)
    }
}

/// 一个已注册转场：轨道 + 前后两 Clip + 转场参数。
#[derive(Debug, Clone, Copy, PartialEq)]
struct TransitionEntry {
    id: TransitionId,
    track: TrackIndex,
    from: ClipId,
    to: ClipId,
    spec: Transition,
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// 多轨时间线：轨道集合 + 转场集合（U14 编排主体）。
///
/// 构造用 [`Timeline::new`]；编辑用 [`Timeline::add_track`] / [`add_clip`] /
/// [`add_transition`] / 静音独 setters；求值用 [`Timeline::eval_at`]。
#[derive(Debug, Clone)]
pub struct Timeline {
    fps: f32,
    tracks: Vec<Track>,
    transitions: Vec<TransitionEntry>,
    next_clip_id: u32,
    next_transition_id: u32,
}

impl Timeline {
    /// 构造空时间线（默认 30 fps；`with_fps` 可改）。
    pub fn new() -> Self {
        Self {
            fps: 30.0,
            tracks: Vec::new(),
            transitions: Vec::new(),
            next_clip_id: 0,
            next_transition_id: 0,
        }
    }

    /// 构造指定帧率的时间线（`fps ≤ 0` 或非有限 → 结构化错误）。
    pub fn with_fps(fps: f32) -> NleResult<Self> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(NleError::new(
                "NLE1201",
                "Timeline::with_fps",
                format!("帧率非法：fps={fps}（要求 0 < fps < ∞）"),
                "提供正的有限帧率（如 24.0 / 25.0 / 30.0）",
            ));
        }
        let mut tl = Self::new();
        tl.fps = fps;
        Ok(tl)
    }

    /// 时间线帧率。
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// 轨道数。
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// 轨道列表（只读；结构序 = `TrackIndex` 升序）。
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// 转场数。
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// 新增一条轨道，返回其索引（索引 = 结构序，确定）。
    pub fn add_track(&mut self, kind: TrackKind) -> TrackIndex {
        let index = TrackIndex(self.tracks.len() as u32);
        self.tracks.push(Track {
            index,
            kind,
            muted: false,
            solo: false,
            transform: Transform::identity(),
            clips: Vec::new(),
            audio: None,
        });
        index
    }

    /// 轨道只读访问（越界返回 `None`）。
    pub fn track(&self, index: TrackIndex) -> Option<&Track> {
        self.tracks.get(index.0 as usize)
    }

    /// 轨道可变访问（越界返回 `None`）。
    pub fn track_mut(&mut self, index: TrackIndex) -> Option<&mut Track> {
        self.tracks.get_mut(index.0 as usize)
    }

    /// 置静音。
    pub fn set_muted(&mut self, index: TrackIndex, muted: bool) -> NleResult<()> {
        match self.tracks.get_mut(index.0 as usize) {
            Some(t) => {
                t.muted = muted;
                Ok(())
            }
            None => Err(Self::bad_track(index, "Timeline::set_muted")),
        }
    }

    /// 置独奏。
    pub fn set_solo(&mut self, index: TrackIndex, solo: bool) -> NleResult<()> {
        match self.tracks.get_mut(index.0 as usize) {
            Some(t) => {
                t.solo = solo;
                Ok(())
            }
            None => Err(Self::bad_track(index, "Timeline::set_solo")),
        }
    }

    /// 置轨道级变换（不透明度 / 混合模式）。
    pub fn set_track_transform(
        &mut self,
        index: TrackIndex,
        transform: Transform,
    ) -> NleResult<()> {
        match self.tracks.get_mut(index.0 as usize) {
            Some(t) => {
                t.transform = transform;
                Ok(())
            }
            None => Err(Self::bad_track(index, "Timeline::set_track_transform")),
        }
    }

    /// 置轨道音频输入（进 mixdown）。
    pub fn set_track_audio(
        &mut self,
        index: TrackIndex,
        audio: AudioTrackInput,
    ) -> NleResult<()> {
        match self.tracks.get_mut(index.0 as usize) {
            Some(t) => {
                t.audio = Some(audio);
                Ok(())
            }
            None => Err(Self::bad_track(index, "Timeline::set_track_audio")),
        }
    }

    /// 新增 Clip：源引用 + 时间映射 + 变换。
    ///
    /// 插入位置由**规范键**决定（与调用顺序无关，见模块文档）。退化点
    /// （非法轨道下标、源时长与映射不符、零长度区间 / 零速率——后者由
    /// [`TimeMap2::new`] 返回）→ 三段式 [`NleError`]。
    pub fn add_clip(
        &mut self,
        track: TrackIndex,
        source: Rc<dyn MediaClipSource>,
        map: TimeMap2,
        transform: Transform,
    ) -> NleResult<ClipId> {
        let idx = track.0 as usize;
        let slot = match self.tracks.get_mut(idx) {
            Some(t) => t,
            None => return Err(Self::bad_track(track, "Timeline::add_clip")),
        };
        // 源时长必须覆盖裁剪区间（否则映射指向不存在的素材）。
        if map.src_out() > source.duration() {
            return Err(NleError::new(
                "NLE1202",
                "Timeline::add_clip",
                format!(
                    "裁剪区间越出源时长：src_out={} > src_dur={}",
                    map.src_out(),
                    source.duration()
                ),
                "把 src_out 收进 [0, 源时长]，或换一个更长的源",
            ));
        }
        let id = ClipId(self.next_clip_id);
        self.next_clip_id = self.next_clip_id.saturating_add(1);
        let clip = Clip { id, source, map, transform };
        let at = slot.insert_index_for(&clip.canonical_key());
        slot.clips.insert(at, clip);
        Ok(id)
    }

    /// 置某个 Clip 的变换（不透明度 / 混合模式）。
    ///
    /// 插错位置不影响求值：Clip 一经插入即按规范键有序，本操作只改其变换
    /// （规范键中的不透明度分量随之更新，但**不重排**——重排会移动 Clip 覆盖
    /// 的求值次序，编辑语义上"改透明度"不应重排同轨片段）。
    pub fn set_clip_transform(
        &mut self,
        track: TrackIndex,
        clip: ClipId,
        transform: Transform,
    ) -> NleResult<()> {
        let slot = match self.tracks.get_mut(track.0 as usize) {
            Some(t) => t,
            None => return Err(Self::bad_track(track, "Timeline::set_clip_transform")),
        };
        match slot.clips.iter_mut().find(|c| c.id == clip) {
            Some(c) => {
                c.transform = transform;
                Ok(())
            }
            None => Err(NleError::new(
                "NLE1203",
                "Timeline::set_clip_transform",
                format!("Clip {clip:?} 不在轨道 {track:?} 上"),
                "用 add_clip 返回的 ClipId，并确认其所属轨道",
            )),
        }
    }

    /// 按 `ClipId` 定位轨道下标与 Clip 在轨内的下标（线性扫描 `TrackIndex` 升序）。
    fn locate_clip(&self, id: ClipId) -> Option<(usize, usize)> {
        for (ti, t) in self.tracks.iter().enumerate() {
            if let Some(ci) = t.clips.iter().position(|c| c.id == id) {
                return Some((ti, ci));
            }
        }
        None
    }

    /// 注册转场：`from` → `to` 在 `overlap` 区间内按 `spec` 混合（S2）。
    ///
    /// 转场不移动 Clip、不改时间线时长——它只是**同一轨相邻两 Clip 在重叠区间
    /// 内的混合规则**（真实的 NLE 拖拽重排属交互层，不在本层）。
    pub fn add_transition(
        &mut self,
        track: TrackIndex,
        from: ClipId,
        to: ClipId,
        spec: Transition,
    ) -> NleResult<TransitionId> {
        let idx = track.0 as usize;
        if !matches!(self.tracks.get(idx), Some(t) if t.kind == TrackKind::Video) {
            return Err(Self::bad_track(track, "Timeline::add_transition"));
        }
        if !spec.is_valid() {
            return Err(NleError::new(
                "NLE1203",
                "Timeline::add_transition",
                format!("转场参数非法：start={} duration={}", spec.start, spec.duration),
                "提供有限的 start 与正的 duration（重叠区间须非空）",
            ));
        }
        if from == to {
            return Err(NleError::new(
                "NLE1203",
                "Timeline::add_transition",
                "转场两端为同一 Clip".to_string(),
                "选择两个不同的 Clip 作为转场前后段",
            ));
        }
        // 两端必须同在该视频轨。
        for end in [from, to] {
            match self.locate_clip(end) {
                Some((ti, _)) if ti == idx => {}
                _ => {
                    return Err(NleError::new(
                        "NLE1203",
                        "Timeline::add_transition",
                        format!("Clip {end:?} 不在轨道 {track:?} 上"),
                        "用 add_clip 返回的 ClipId，且两端属同一视频轨",
                    ))
                }
            }
        }
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id = self.next_transition_id.saturating_add(1);
        self.transitions.push(TransitionEntry { id, track, from, to, spec });
        Ok(id)
    }

    /// 转场标识（注册序，确定）。
    pub fn transition(&self, id: TransitionId) -> Option<&Transition> {
        self.transitions.iter().find(|e| e.id == id).map(|e| &e.spec)
    }

    /// 时间线总时长（秒）= 所有 Clip 区间终点与所有转场终点的最大值；空时间线为 0。
    pub fn duration(&self) -> f32 {
        let mut d: f32 = 0.0;
        for t in &self.tracks {
            for c in &t.clips {
                let e = c.timeline_end();
                if e.is_finite() && e > d {
                    d = e;
                }
            }
        }
        for e in &self.transitions {
            let end = e.spec.end();
            if end.is_finite() && end > d {
                d = end;
            }
        }
        d
    }

    /// 在时间线时刻 `t` 求值画面（秒）。
    ///
    /// 流程：范围校验 → 独奏/静音筛选（索引序）→ 逐轨求值（同轨 Clip 规范序，
    /// 相邻两 Clip 命中转场则按进度混合）→ 轨道间自下而上 `composite` 叠加。
    ///
    /// 退化点：`t` 非有限或超出 `[0, duration()]` → `NLE1204`；源求值失败 →
    /// 透传源错误码。
    pub fn eval_at(&self, t: f32) -> NleResult<Frame> {
        if !t.is_finite() {
            return Err(NleError::new(
                "NLE1204",
                "Timeline::eval_at",
                format!("求值时刻非有限：t={t}"),
                "传入有限的秒数（0 ≤ t ≤ duration()）",
            ));
        }
        let total = self.duration();
        if t < 0.0 || t > total {
            return Err(NleError::new(
                "NLE1204",
                "Timeline::eval_at",
                format!("求值时刻越界：t={t} ∉ [0, {total}]"),
                "把 t 钳进 [0, duration()]，或先用 add_clip 铺设足够长的内容",
            ));
        }
        // 独奏：任一轨 solo → 仅 solo 轨参与（索引序线性扫描，确定）。
        let any_solo = self.tracks.iter().any(|t| t.solo);
        let mut acc: Option<Frame> = None;
        for track in &self.tracks {
            if track.kind != TrackKind::Video {
                continue;
            }
            if track.muted || (any_solo && !track.solo) {
                continue;
            }
            let layer = match self.eval_track(track, t)? {
                Some(f) => f,
                None => continue,
            };
            // 轨道级变换：轨道结果作为"上层"合成到已有画面之上。
            acc = Some(match acc {
                None => composite(&Frame::new(0, 0), &layer, track.transform),
                Some(base) => composite(&base, &layer, track.transform),
            });
        }
        Ok(acc.unwrap_or_else(|| Frame::new(0, 0)))
    }

    /// 单轨求值（同轨 Clip 按规范序；相邻两 Clip 命中转场则混合）。
    ///
    /// **Clip 级变换在本层生效**：非转场分支把该 Clip 的帧按其自身
    /// [`Transform`] 合成到轨内已累积的画面之上（首个 Clip 以空帧为下层）。
    /// 转场分支由转场自身的进度曲线决定两段的混合比例，故**不再**叠加两端
    /// Clip 的不透明度（转场进度即权威比例，避免"淡入淡出 × 不透明度"双重
    /// 淡入的歧义语义）。轨级 [`Track::transform`] 随后在 [`Self::eval_at`]
    /// 中作用于整轨结果。
    fn eval_track(&self, track: &Track, t: f32) -> NleResult<Option<Frame>> {
        // 覆盖 t 的 Clip 下标（规范序即确定序）。
        let idx: Vec<usize> = (0..track.clips.len()).filter(|i| track.clips[*i].covers(t)).collect();
        if idx.is_empty() {
            return Ok(None);
        }
        let mut acc: Option<Frame> = None;
        let mut i = 0;
        while i < idx.len() {
            // 相邻两覆盖 Clip 之间是否有转场在 t 处生效？
            let pair = idx.get(i + 1).copied().and_then(|j| {
                let a = &track.clips[idx[i]];
                let b = &track.clips[j];
                self.transitions
                    .iter()
                    .find(|e| {
                        e.track == track.index
                            && ((e.from == a.id && e.to == b.id) || (e.from == b.id && e.to == a.id))
                            && e.spec.covers(t)
                    })
                    .map(|e| e.spec)
            });
            match pair {
                Some(spec) => {
                    // 转场语义：from = 前段（下层 A），to = 后段（上层 B）。
                    let fa = track.clips[idx[i]].eval(t)?;
                    let fb = track.clips[idx[i + 1]].eval(t)?;
                    let mixed = spec.blend(t, &fa, &fb);
                    acc = Some(match acc {
                        None => mixed,
                        Some(base) => composite(&base, &mixed, Transform::identity()),
                    });
                    i += 2;
                }
                None => {
                    let clip = &track.clips[idx[i]];
                    i += 1;
                    // 不透明度 0 = 完全透明 = 不参与合成：跳过（若它是本轨首个
                    // 贡献者，则整轨视为无内容 ⟹ 上层轨道不遮挡下层轨道）。
                    // 否则零帧会以"不透明黑"的身份盖住下层 —— 违背透明语义。
                    if clip.transform.opacity <= 0.0 {
                        continue;
                    }
                    let f = clip.eval(t)?;
                    // Clip 级不透明度 / 混合模式在此生效。
                    acc = Some(match acc {
                        None => composite(&Frame::new(0, 0), &f, clip.transform),
                        Some(base) => composite(&base, &f, clip.transform),
                    });
                }
            }
        }
        Ok(acc)
    }

    /// 多轨音频 mixdown（**白皮书 §26 固定形状归约**，见 [`audio`]）。
    ///
    /// 收集各轨 [`AudioTrackInput`]（`TrackIndex` 升序），按时间线的静音 / 独奏
    /// 语义折算进每轨的 `muted`，再交 `audio::mixdown` 的平衡 pairwise 树归约
    /// ——**本层不写浮点累加**。无任何轨道带音频 → `NLE1102`。
    pub fn mixdown(&self, peak_guard: bool) -> NleResult<PcmBuffer> {
        let any_solo = self.tracks.iter().any(|t| t.solo);
        let mut inputs: Vec<AudioTrackInput> = Vec::new();
        for t in &self.tracks {
            if let Some(a) = &t.audio {
                let mut a = a.clone();
                if t.muted || (any_solo && !t.solo) {
                    a.muted = true;
                }
                inputs.push(a);
            }
        }
        if inputs.is_empty() {
            return Err(NleError::new(
                "NLE1102",
                "Timeline::mixdown",
                "没有任何轨道携带音频输入".to_string(),
                "先用 set_track_audio 至少给一条轨道设置 AudioTrackInput",
            ));
        }
        // 静音 / 独奏已折算进每轨 muted ⟹ 此处 solo 集为空（避免二次判定）。
        mixdown(&inputs, &[], peak_guard)
    }

    /// 非法轨道下标的标准三段式错误。
    fn bad_track(index: TrackIndex, op: &str) -> NleError {
        NleError::new(
            "NLE1205",
            op,
            format!("轨道下标越界：TrackIndex({}) ，现有轨道数 = ?", index.0),
            "用 add_track 返回的 TrackIndex（本时间线当前轨道数见 track_count()）",
        )
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    /// 求值序列的逐位指纹（确定性断言的统一工具）。
    fn bits_of(frames: &[Frame]) -> Vec<u32> {
        let mut v = Vec::new();
        for f in frames {
            for p in &f.pixels {
                v.push(p.to_bits());
            }
            v.push(f.width);
            v.push(f.height);
        }
        v
    }

    fn sample(tl: &Timeline, t: f32) -> Vec<u32> {
        bits_of(&[tl.eval_at(t).unwrap_or_else(|e| panic!("{e}"))])
    }

    /// 时间线求值序列（步进 0.25s，含终点）。
    fn sweep(tl: &Timeline) -> Vec<u32> {
        let d = tl.duration();
        let steps = (d / 0.25).ceil().max(0.0) as usize;
        let mut v = Vec::new();
        for i in 0..=steps {
            v.extend(sample(tl, i as f32 * 0.25));
        }
        v
    }

    fn ramp(id: u32, dur: f32) -> Rc<dyn MediaClipSource> {
        Rc::new(RampSource::new(id, dur, 1, 1))
    }

    fn solid_src(id: u32, dur: f32, c: [f32; 4]) -> Rc<dyn MediaClipSource> {
        Rc::new(SolidSource::new(id, dur, 1, 1, c))
    }

    fn ident() -> Transform {
        Transform::identity()
    }

    // -- 基础 API --------------------------------------------------------

    #[test]
    fn new_timeline_is_empty() {
        let tl = Timeline::new();
        assert_eq!(tl.track_count(), 0);
        assert_eq!(tl.transition_count(), 0);
        assert_eq!(tl.duration(), 0.0);
        // 空时间线求值 0 秒 → 空帧（单位元），不报错。
        assert!(tl.eval_at(0.0).unwrap_or_else(|e| panic!("{e}")).is_empty());
    }

    #[test]
    fn add_track_and_clip_counts() {
        let mut tl = Timeline::new();
        let a = tl.add_track(TrackKind::Video);
        let b = tl.add_track(TrackKind::Audio);
        assert_eq!(a, TrackIndex(0));
        assert_eq!(b, TrackIndex(1));
        assert_eq!(tl.track_count(), 2);
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(1.0), false).unwrap();
        let id = tl.add_clip(a, ramp(7, 4.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(id, ClipId(0));
        assert_eq!(tl.track(a).map(|t| t.clips().len()), Some(1));
    }

    // -- 裁剪 / 变速 / 倒放（S5 特解，经由求值链验证）--------------------

    #[test]
    fn trim_boundaries_visible_through_ramp() {
        // 源 10s 裁 [2,6) 落时间线 [1,5)：t=1 → u=0.2；t=5 → u=0.6。
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::new(10.0, 2.0, 6.0, 1.0, 4.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 10.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        let mid = tl.eval_at(3.0).unwrap_or_else(|e| panic!("{e}"));
        assert!((mid.pixel(0, 0)[0] - 0.4).abs() < 1e-5, "t=3 → u=(2+2)/10=0.4");
        let head = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}"));
        assert!((head.pixel(0, 0)[0] - 0.2).abs() < 1e-5, "trim in 边界");
    }

    #[test]
    fn speed_2x_halves_timeline_duration() {
        // 源 4s，2× 速 → 时间线 2s。
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 2.0, RateCurve::Constant(2.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 4.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        assert!((tl.duration() - 2.0).abs() < 1e-6, "2× 速时长减半");
        let mid = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}"));
        assert!((mid.pixel(0, 0)[0] - 0.5).abs() < 1e-5, "t=1 → 源 2s → u=0.5");
    }

    #[test]
    fn reverse_swaps_head_and_tail() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(1.0), true)
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 4.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        let head = tl.eval_at(0.25).unwrap_or_else(|e| panic!("{e}"));
        let tail = tl.eval_at(3.75).unwrap_or_else(|e| panic!("{e}"));
        assert!(head.pixel(0, 0)[0] > 0.9, "倒放首 ≈ 源尾 u≈0.94");
        assert!(tail.pixel(0, 0)[0] < 0.1, "倒放尾 ≈ 源首 u≈0.06");
    }

    // -- 多轨合成 --------------------------------------------------------

    #[test]
    fn upper_track_opacity_0_and_1_select_layers() {
        let mut tl = Timeline::new();
        let lower = tl.add_track(TrackKind::Video);
        let upper = tl.add_track(TrackKind::Video);
        let m = TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(lower, solid_src(10, 2.0, [0.0, 0.0, 1.0, 1.0]), m.clone(), ident())
            .unwrap_or_else(|e| panic!("{e}"));
        let top = tl.add_clip(
            upper,
            solid_src(11, 2.0, [1.0, 0.0, 0.0, 1.0]),
            m,
            Transform::new(0.0, BlendMode::Normal),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let c0 = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((c0[2] - 1.0).abs() < 1e-6 && c0[0].abs() < 1e-6, "Clip 透明度 0 → 只见下层蓝");

        // 轨级不透明度同样生效（1 → 整轨不透明，红盖蓝）。
        tl.add_clip(
            upper,
            solid_src(12, 2.0, [1.0, 0.0, 0.0, 1.0]),
            TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}")),
            ident(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        tl.set_track_transform(upper, Transform::new(1.0, BlendMode::Normal))
            .unwrap_or_else(|e| panic!("{e}"));
        let c1 = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((c1[0] - 1.0).abs() < 1e-6 && c1[2].abs() < 1e-6, "轨级透明度 1 → 只见上层红");

        // Clip 级不透明度 1 也一样（经 set_clip_transform 改同一 Clip）。
        tl.set_clip_transform(upper, top, Transform::new(1.0, BlendMode::Normal))
            .unwrap_or_else(|e| panic!("{e}"));
        let c2 = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((c2[0] - 1.0).abs() < 1e-6, "Clip 透明度 1 → 只见上层红");
    }

    #[test]
    fn solo_and_mute_gate_tracks() {
        let mut tl = Timeline::new();
        let lower = tl.add_track(TrackKind::Video);
        let upper = tl.add_track(TrackKind::Video);
        let m = TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(lower, solid_src(10, 2.0, [0.0, 0.0, 1.0, 1.0]), m.clone(), ident())
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(upper, solid_src(11, 2.0, [1.0, 0.0, 0.0, 1.0]), m, ident())
            .unwrap_or_else(|e| panic!("{e}"));

        tl.set_solo(upper, true).unwrap_or_else(|e| panic!("{e}"));
        let soloed = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((soloed[0] - 1.0).abs() < 1e-6, "独奏上层 → 下层被屏蔽");

        tl.set_solo(upper, false).unwrap_or_else(|e| panic!("{e}"));
        tl.set_muted(upper, true).unwrap_or_else(|e| panic!("{e}"));
        let muted = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((muted[2] - 1.0).abs() < 1e-6, "静音上层 → 回落下层");
    }

    // -- 转场（S2 特解）--------------------------------------------------

    #[test]
    fn dissolve_midpoint_is_fifty_fifty_on_timeline() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        // 两 Clip 在 [0.5, 1.5) 重叠：A 蓝、B 红。
        let ma = TimeMap2::new(4.0, 0.0, 2.0, 0.0, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        let mb = TimeMap2::new(4.0, 0.0, 2.0, 0.5, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        let a = tl.add_clip(tr, solid_src(10, 4.0, [0.0, 0.0, 1.0, 1.0]), ma, ident())
            .unwrap_or_else(|e| panic!("{e}"));
        let b = tl.add_clip(tr, solid_src(11, 4.0, [1.0, 0.0, 0.0, 1.0]), mb, ident())
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_transition(
            tr,
            a,
            b,
            Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.5, 1.0),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        // 重叠中点 t=1.0 → p=0.5 → 两 Clip 各 50%。
        let c = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0);
        assert!((c[0] - 0.5).abs() < 1e-6 && (c[2] - 0.5).abs() < 1e-6, "中点各 50%: {c:?}");
        // 重叠外：t=0.25 只有 A（蓝），t=1.75 只有 B（红）。
        assert!((tl.eval_at(0.25).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0)[2] - 1.0).abs() < 1e-6);
        assert!((tl.eval_at(1.75).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0)[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn wipe_transition_produces_hard_edge() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let ma = TimeMap2::new(4.0, 0.0, 2.0, 0.0, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        let mb = TimeMap2::new(4.0, 0.0, 2.0, 0.5, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        let a = tl.add_clip(
            tr,
            Rc::new(SolidSource::new(10, 4.0, 4, 1, [0.0, 0.0, 1.0, 1.0])),
            ma,
            ident(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let b = tl.add_clip(
            tr,
            Rc::new(SolidSource::new(11, 4.0, 4, 1, [1.0, 0.0, 0.0, 1.0])),
            mb,
            ident(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        tl.add_transition(
            tr,
            a,
            b,
            Transition::new(
                TransitionKind::Wipe(WipeDirection::LeftToRight),
                TransitionEasing::Linear,
                0.5,
                1.0,
            ),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let f = tl.eval_at(1.0).unwrap_or_else(|e| panic!("{e}"));
        // p=0.5、4 像素宽：x=0 的 u=0 < 0.5 → 已擦入 B（红）；x=3 的 u=0.75 ≥ 0.5 → 仍 A（蓝）。
        assert_eq!(f.pixel(0, 0), [1.0, 0.0, 0.0, 1.0], "左端已擦入 B");
        assert_eq!(f.pixel(3, 0), [0.0, 0.0, 1.0, 1.0], "右端仍 A（硬边，非渐变）");
    }

    #[test]
    fn keyframed_speed_curve_on_timeline_is_monotone() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::new(
            8.0,
            0.0,
            8.0,
            0.0,
            4.0,
            RateCurve::Keyframed(vec![(0.0, 1.0), (2.0, 1.0), (4.0, 3.0)]),
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 8.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        // 只扫 Clip 区间内 [0, 4)：t=4 处 Clip 已结束（半开区间），不参与。
        let mut prev = -1.0_f32;
        for i in 0..32 {
            let t = i as f32 * 0.125;
            let u = tl.eval_at(t).unwrap_or_else(|e| panic!("{e}")).pixel(0, 0)[0];
            assert!(u >= prev - 1e-5, "关键帧速率下求值应单调不减 (t={t})");
            prev = u;
        }
    }

    // -- 确定性（最高铁律）----------------------------------------------

    #[test]
    fn same_timeline_two_passes_bitwise_identical() {
        let mut tl = Timeline::new();
        let v0 = tl.add_track(TrackKind::Video);
        let v1 = tl.add_track(TrackKind::Video);
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(v0, ramp(1, 4.0), m.clone(), ident()).unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(
            v1,
            solid_src(2, 4.0, [0.3, 0.6, 0.9, 1.0]),
            m.clone(),
            Transform::new(0.25, BlendMode::Screen),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let first = sweep(&tl);
        for _ in 0..50 {
            assert_eq!(sweep(&tl), first, "同 timeline 两遍必须字节级一致");
        }
    }

    #[test]
    fn equivalent_timeline_regardless_of_add_order() {
        // 两 Clip 落点不同 ⟹ 规范键不同 ⟹ 与添加顺序无关。
        // 一个实质性的时间线：两轨 + 叠加 + 不透明度。
        let build = |swap: bool| {
            let mut tl = Timeline::new();
            let v0 = tl.add_track(TrackKind::Video);
            let v1 = tl.add_track(TrackKind::Video);
            let ramp_clip = |id: u32, tl_in: f32, op: f32| {
                let m = TimeMap2::new(4.0, 0.0, 4.0, tl_in, 2.0, RateCurve::Constant(1.0), false)
                    .unwrap_or_else(|e| panic!("{e}"));
                (id, m, Transform::new(op, BlendMode::Normal))
            };
            let a = ramp_clip(20, 0.0, 1.0);
            let b = ramp_clip(21, 1.0, 0.5);
            if swap {
                tl.add_clip(v1, ramp(21, 4.0), b.1, b.2).unwrap_or_else(|e| panic!("{e}"));
                tl.add_clip(v0, ramp(20, 4.0), a.1, a.2).unwrap_or_else(|e| panic!("{e}"));
            } else {
                tl.add_clip(v0, ramp(20, 4.0), a.1, a.2).unwrap_or_else(|e| panic!("{e}"));
                tl.add_clip(v1, ramp(21, 4.0), b.1, b.2).unwrap_or_else(|e| panic!("{e}"));
            }
            tl
        };
        assert_eq!(
            sweep(&build(false)),
            sweep(&build(true)),
            "不同添加顺序构造的等价 timeline 结果应一致"
        );
    }

    #[test]
    fn canonical_order_is_insertion_order_independent() {
        // 直接断言轨道内 Clip 的规范序：timeline_in 升序，与插入顺序相反。
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let late = TimeMap2::new(4.0, 0.0, 4.0, 2.0, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        let early = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 2.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 4.0), late, ident()).unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(2, 4.0), early, ident()).unwrap_or_else(|e| panic!("{e}"));
        let ins: Vec<f32> =
            tl.track(tr).unwrap_or_else(|| panic!("无轨")).clips().iter().map(|c| c.time_map().timeline_in()).collect();
        assert_eq!(ins, vec![0.0, 2.0], "后插入的早落点 clip 应排到前面");
    }

    // -- 音频（§26 固定形状归约）----------------------------------------

    #[test]
    fn timeline_mixdown_is_bitwise_reproducible() {
        let mut tl = Timeline::new();
        let a = tl.add_track(TrackKind::Audio);
        let b = tl.add_track(TrackKind::Audio);
        let buf = |k: f32| {
            PcmBuffer::from_mono(8_000, (0..32).map(|i| (i as f32) * 0.01 * k).collect())
                .unwrap_or_else(|e| panic!("{e}"))
        };
        tl.set_track_audio(a, AudioTrackInput::new(buf(1.0), 1.0)).unwrap_or_else(|e| panic!("{e}"));
        tl.set_track_audio(b, AudioTrackInput::new(buf(-0.5), 0.7)).unwrap_or_else(|e| panic!("{e}"));
        let first = tl.mixdown(false).unwrap_or_else(|e| panic!("{e}"));
        let want: Vec<u32> = first.as_slice().iter().map(|v| v.to_bits()).collect();
        for _ in 0..100 {
            let got = tl.mixdown(false).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                got.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want,
                "时间线 mixdown 两遍必须逐位一致"
            );
        }
    }

    #[test]
    fn timeline_mixdown_respects_solo_and_mute() {
        let mut tl = Timeline::new();
        let a = tl.add_track(TrackKind::Audio);
        let b = tl.add_track(TrackKind::Audio);
        let ones = PcmBuffer::from_mono(1_000, vec![1.0; 8]).unwrap_or_else(|e| panic!("{e}"));
        let half = PcmBuffer::from_mono(1_000, vec![0.5; 8]).unwrap_or_else(|e| panic!("{e}"));
        tl.set_track_audio(a, AudioTrackInput::new(ones.clone(), 1.0)).unwrap_or_else(|e| panic!("{e}"));
        tl.set_track_audio(b, AudioTrackInput::new(half, 1.0)).unwrap_or_else(|e| panic!("{e}"));
        let both = tl.mixdown(false).unwrap_or_else(|e| panic!("{e}"));
        assert!((both.sample(0, 0).unwrap_or(0.0) - 1.5).abs() < 1e-6, "两轨相加");

        tl.set_solo(a, true).unwrap_or_else(|e| panic!("{e}"));
        let soloed = tl.mixdown(false).unwrap_or_else(|e| panic!("{e}"));
        assert!((soloed.sample(0, 0).unwrap_or(0.0) - 1.0).abs() < 1e-6, "独奏轨 0");

        tl.set_solo(a, false).unwrap_or_else(|e| panic!("{e}"));
        tl.set_muted(a, true).unwrap_or_else(|e| panic!("{e}"));
        let muted = tl.mixdown(false).unwrap_or_else(|e| panic!("{e}"));
        assert!((muted.sample(0, 0).unwrap_or(0.0) - 0.5).abs() < 1e-6, "静音轨 0");
    }

    #[test]
    fn mixdown_without_any_track_audio_errors() {
        let tl = Timeline::new();
        assert_eq!(tl.mixdown(false).unwrap_err().code, "NLE1102");
    }

    // -- 退化点（C18 永不 panic）----------------------------------------

    #[test]
    fn degenerate_construction_errors_structured() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        // 非法轨道下标。
        let m = TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            tl.add_clip(TrackIndex(99), ramp(1, 2.0), m.clone(), ident()).unwrap_err().code,
            "NLE1205"
        );
        // 零长度 clip（tl_dur = 0）。
        let zero = TimeMap2::new(2.0, 0.0, 2.0, 0.0, 0.0, RateCurve::Constant(1.0), false).unwrap_err();
        assert_eq!(zero.code, "NLE1003", "零长度区间在 TimeMap2 构造期即被拒");
        // 零速率变速。
        let zero_rate = TimeMap2::new(2.0, 0.0, 2.0, 0.0, 2.0, RateCurve::Constant(0.0), false)
            .unwrap_err();
        assert_eq!(zero_rate.code, "NLE1004");
        // 裁剪区间越出源时长。
        let over = TimeMap2::new(10.0, 0.0, 10.0, 0.0, 10.0, RateCurve::Constant(1.0), false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tl.add_clip(tr, ramp(1, 2.0), over, ident()).unwrap_err().code, "NLE1202");
        // 非法帧率。
        assert_eq!(Timeline::with_fps(0.0).unwrap_err().code, "NLE1201");
    }

    #[test]
    fn out_of_range_and_non_finite_eval_error_structured() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
        tl.add_clip(tr, ramp(1, 2.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tl.eval_at(-0.1).unwrap_err().code, "NLE1204", "负时刻");
        assert_eq!(tl.eval_at(2.5).unwrap_err().code, "NLE1204", "超时长");
        assert_eq!(tl.eval_at(f32::NAN).unwrap_err().code, "NLE1204", "NaN");
        assert_eq!(tl.eval_at(f32::INFINITY).unwrap_err().code, "NLE1204", "∞");
        // 端点本身合法。
        assert!(tl.eval_at(0.0).is_ok());
        assert!(tl.eval_at(2.0).is_ok());
    }

    #[test]
    fn invalid_transition_errors_structured() {
        let mut tl = Timeline::new();
        let tr = tl.add_track(TrackKind::Video);
        let m = TimeMap2::direct(2.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
        let a = tl.add_clip(tr, ramp(1, 2.0), m.clone(), ident()).unwrap_or_else(|e| panic!("{e}"));
        let b = tl.add_clip(tr, ramp(2, 2.0), m, ident()).unwrap_or_else(|e| panic!("{e}"));
        // 零时长重叠。
        assert_eq!(
            tl.add_transition(
                tr,
                a,
                b,
                Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.0, 0.0)
            )
            .unwrap_err()
            .code,
            "NLE1203"
        );
        // 两端同一 Clip。
        assert_eq!(
            tl.add_transition(
                tr,
                a,
                a,
                Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.0, 1.0)
            )
            .unwrap_err()
            .code,
            "NLE1203"
        );
        // Clip 不在该轨。
        let other = tl.add_track(TrackKind::Video);
        assert_eq!(
            tl.add_transition(
                other,
                a,
                b,
                Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.0, 1.0)
            )
            .unwrap_err()
            .code,
            "NLE1203"
        );
    }

    #[test]
    fn setters_reject_bad_track_index() {
        let mut tl = Timeline::new();
        assert_eq!(tl.set_muted(TrackIndex(3), true).unwrap_err().code, "NLE1205");
        assert_eq!(tl.set_solo(TrackIndex(3), true).unwrap_err().code, "NLE1205");
        assert_eq!(
            tl.set_track_transform(TrackIndex(3), ident()).unwrap_err().code,
            "NLE1205"
        );
        assert_eq!(
            tl.set_track_audio(TrackIndex(3), AudioTrackInput::new(
                PcmBuffer::from_mono(1_000, vec![0.0]).unwrap_or_else(|e| panic!("{e}")),
                1.0
            ))
            .unwrap_err()
            .code,
            "NLE1205"
        );
    }

    #[test]
    fn error_display_has_three_segments() {
        let e = tl_err();
        let s = e.to_string();
        assert!(s.contains("NLE1205") && s.contains("什么事") && s.contains("建议"), "{s}");
    }

    fn tl_err() -> NleError {
        Timeline::new().set_muted(TrackIndex(7), true).unwrap_err()
    }
}
