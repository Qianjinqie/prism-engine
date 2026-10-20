//! 场景序列 + 转场凸权重半群（ADR-18，P7 下沉）。
//!
//! 语义权威：转场凸权重半群 + u8 整数混合的算法规格。本模块把原先散落在 `prism-author::render` 的转场混合下沉到 core，
//! 使 **转场类型**（`Fade` / `Wipe` / `Dissolve` / `Push`）成为引擎级概念，
//! 而 author 侧只保留「帧规格 → 帧字节」的脏活。
//!
//! # 确定性关键（宪章硬约束）
//!
//! - **零第三方 crate**、**零浮点**：本文件所有计算路径只用整数
//!   （`u8` / `u16` / `u32` / `u64` / `i64`），因此同输入两跑 **bitwise 相同**，
//!   且 CPU / GPU / 不同架构间逐位一致（无 fma 收缩、无舍入模式差异）；
//! - Dissolve 的显影图案用**自实现 splitmix64 类整数 hash**（不用 f32 rand）：
//!   `hash ∈ 0..=255` 的整数阈值，`k=0` 全不显影、`k=256` 全显影 ⇒ 端点逐位；
//! - Wipe 的分界行程按 ADR-18 证 4 扩展：`edge(k) = (1−k/256)·(extent+2·band) − band`，
//!   两端混合带完全出画 ⇒ 修复后 `k=0` / `k=256` 亦逐位精确；
//! - u8 整数混合 `(W·b + (256−W)·a + 128) >> 8` 是 round-to-nearest，
//!   单次误差 ≤ 1 LSB、**无系统偏置**（对照 f32 截断路径的 mean bias ≈ −0.488）。
//!
//! # 半群代数
//!
//! 同目标链的折叠闭式 `W_u = w1 + w2 − w1·w2/256`（对应 `u = s + t − st`）；
//! 不同目标链不可折叠，正确结构是**凸权重向量**（见 [`TransitionChain`]）：
//! `≤3` 段用 `u32` 精确、`≤4` 段用 `u64` 精确（`256⁴ = 2^32` 恰溢出 `u32` 一个
//! 单位），更长每步重归一化（误差 ≤ 1/256 LSB/次，有界）。

// ---------------------------------------------------------------------------
// 精确常量（ADR-18 ① 精确常量表）
// ---------------------------------------------------------------------------

/// 权重分母（`W = round(256·t)`，`t ∈ [0,1]`）。
pub const W_ONE: u16 = 256;

/// 擦除过渡带半宽（px）：硬阈值 ±`WIPE_BAND` 内用 smoothstep 平滑，避免硬边。
///
/// 值单点定义在此（原先在 `prism-author::render`），author 侧以 re-export
/// 常量形式保留，避免破坏既有引用。
pub const WIPE_BAND: u16 = 2;

// ---------------------------------------------------------------------------
// u8 域整数混合
// ---------------------------------------------------------------------------

/// u8 域整数混合：`out = (w·b + (256−w)·a + 128) >> 8`（`w ∈ 0..=256`）。
///
/// `+128` 是 round-to-nearest 的整数形式：舍入误差对称 ⇒ **无系统偏置**
/// （单次误差 ≤ 1 LSB）。端点精确：`w = 0` 时 `(256a + 128) >> 8 = a`，
/// `w = 256` 时 `(256b + 128) >> 8 = b`（`128 < 256`，余项不进位）。
///
/// 越界权重按端点钳位（`w > 256` 视作 256），确定性、不 panic。
#[inline]
pub fn mix_u8(a: u8, b: u8, w: u16) -> u8 {
    let w: u32 = concat_clamp_w(w) as u32;
    let iw: u32 = (W_ONE as u32) - w;
    (((w * (b as u32)) + (iw * (a as u32)) + 128) >> 8) as u8
}

/// 权重钳位到 `0..=256`。
#[inline]
fn concat_clamp_w(w: u16) -> u16 {
    if w > W_ONE {
        W_ONE
    } else {
        w
    }
}

// ---------------------------------------------------------------------------
// 像素上下文 + 转场 trait
// ---------------------------------------------------------------------------

/// 像素上下文：坐标 + 帧尺寸（供转场权重按位置求值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelCtx {
    /// 列坐标（0 起）。
    pub x: u32,
    /// 行坐标（0 起）。
    pub y: u32,
    /// 帧宽（像素）。
    pub w: u32,
    /// 帧高（像素）。
    pub h: u32,
}

impl PixelCtx {
    /// 以坐标与帧尺寸构造。
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

/// 转场统一框架：给定进度 `k`（`0..=256`）与像素上下文，返回该像素的
/// **凸权重** `w ∈ 0..=256`（`0` = 纯旧片 `a`、`256` = 纯新片 `b`）。
///
/// 四种转场都是同一凸权重向量的不同参数化（ADR-18 中心定理）：
/// Fade 是全局线性权重、Wipe 是位置阈值 + smoothstep 带、Dissolve 是
/// 整数 hash 阈值（0/256 二值）、Push 是位置判据（0/256 二值）。
pub trait Transition {
    /// 该像素在进度 `k` 处的混合权重（`0..=256`）。
    fn weight(&self, k: u16, pixel: &PixelCtx) -> u16;
}

/// 推移方向：描述**新片段 `b` 的入场侧**（`a` 从对侧退出）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushDir {
    /// `b` 自左侧入场（`a` 向右退出）。
    Left,
    /// `b` 自右侧入场（`a` 向左退出）。
    Right,
    /// `b` 自顶部入场（`a` 向下退出）。
    Up,
    /// `b` 自底部入场（`a` 向上退出）。
    Down,
}

impl PushDir {
    /// 该像素是否取自新片段 `b`（`k=0` 恒 `false`、`k=256` 恒 `true`）。
    ///
    /// 判据全整数：`c = round(k·extent/256)`（见 [`push_c`]），沿推进轴比较。
    pub fn take_b(self, k: u16, pixel: &PixelCtx) -> bool {
        match self {
            PushDir::Left => (pixel.x as u64) < push_c(k, pixel.w) as u64,
            PushDir::Right => {
                (pixel.x as u64) >= (pixel.w as u64) - push_c(k, pixel.w) as u64
            }
            PushDir::Up => (pixel.y as u64) < push_c(k, pixel.h) as u64,
            PushDir::Down => {
                (pixel.y as u64) >= (pixel.h as u64) - push_c(k, pixel.h) as u64
            }
        }
    }
}

/// 转场种类（四种转场统一为凸权重的不同参数化）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    /// 交叉溶解：全局线性权重 `w = k`。
    Fade,
    /// 线性擦除：分界线自右向左推进（`b` 自右侧入场，带 `band` px 平滑带）。
    ///
    /// 镜像方向（左 / 上 / 下）由调用方对权重取 `256 − w` 或换轴求值
    /// （见 [`wipe_weight`]）；本枚举只承载「沿 x 反向推进」这一规范形态。
    Wipe {
        /// 过渡带半宽（px）；`0` 防御为硬边（无除零）。
        band: u16,
    },
    /// 随机像素溶解：`integer_hash(pixel, seed) < k` ⇒ 显影 `b`，否则保留 `a`。
    Dissolve {
        /// 显影种子（同 seed 同图案；整数 hash，跨端一致）。
        seed: u64,
    },
    /// 推移：两帧作为同一画面整体平移（硬接缝，端点逐位精确）。
    Push {
        /// `b` 的入场侧。
        dir: PushDir,
    },
}

impl Transition for TransitionKind {
    fn weight(&self, k: u16, pixel: &PixelCtx) -> u16 {
        let k: u16 = concat_clamp_w(k);
        match *self {
            TransitionKind::Fade => k,
            TransitionKind::Wipe { band } => wipe_weight(pixel.x, k, pixel.w, band),
            TransitionKind::Dissolve { seed } => {
                if dissolve_hash(pixel.x, pixel.y, seed) < k {
                    W_ONE
                } else {
                    0
                }
            }
            TransitionKind::Push { dir } => {
                if dir.take_b(k, pixel) {
                    W_ONE
                } else {
                    0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wipe：分界行程扩展 + 整数 smoothstep
// ---------------------------------------------------------------------------

/// 分界线位置（×256 定点）：`edge(k) = (1−k/256)·(extent + 2·band) − band`。
///
/// 返回值是 `edge·256`（`i64` 定点，避免浮点）。分界行程被扩展
/// `2·band`，使两端混合带**完全出画**：`k=0` 时 `edge = extent+band`
/// （末列仍在带外）、`k=256` 时 `edge = −band`（首列已在带外 ⇒ 权重 256）。
#[inline]
pub fn wipe_edge_q8(k: u16, extent: u32, band: u16) -> i64 {
    let k: i64 = concat_clamp_w(k) as i64;
    let b: i64 = band as i64;
    let ext: i64 = extent as i64;
    // (256 − k)·(ext + 2b) − 256·b，整体放大 256 倍。
    (256 - k) * (ext + 2 * b) - 256 * b
}

/// 擦除权重（沿 x 正向推进，`b` 自左侧入场）：`0..=256`。
///
/// `u = (pos − edge + band) / (2·band)`（相对混合带起点的归一化位置），
/// `w = round(256·smoothstep(u))`——全程整数定点（Q16），无浮点。
///
/// 端点精确：`k=0` ⇒ 所有 `pos` 落在 `edge − band` 左侧 ⇒ `w = 0`；
/// `k=256` ⇒ 所有 `pos` 落在 `edge + band` 右侧 ⇒ `w = 256`。
/// `band = 0` 防御：退化为硬阈值（`pos` 与 `edge` 直接比较），不除零。
#[inline]
pub fn wipe_weight(pos: u32, k: u16, extent: u32, band: u16) -> u16 {
    let edge_q8: i64 = wipe_edge_q8(k, extent, band);
    if band == 0 {
        // 硬边：pos·256 与 edge·256 比较（等价于 pos vs edge，无除法）。
        return if (pos as i64) * 256 < edge_q8 { 0 } else { W_ONE };
    }
    let b: i64 = band as i64;
    // u 的分子（×256 缩放）：pos·256 − edge·256 + band·256。
    let num: i64 = (pos as i64) * 256 - edge_q8 + b * 256;
    let den: i64 = 2 * b * 256;
    // Q16 归一化位置：0..=65536。
    let u: i64 = if num <= 0 {
        0
    } else if num >= den {
        65536
    } else {
        (num << 16) / den
    };
    let s: i64 = smoothstep_q16(u);
    // round(256·s/65536) = (s + 128) >> 8；s ≤ 65536 ⇒ w ≤ 256。
    ((s + 128) >> 8) as u16
}

/// 整数 smoothstep（Q16 定点）：`t²·(3 − 2t)`，输入输出均 `0..=65536`。
#[inline]
fn smoothstep_q16(t: i64) -> i64 {
    let t: i64 = if t < 0 {
        0
    } else if t > 65536 {
        65536
    } else {
        t
    };
    let t2: i64 = (t * t) >> 16;
    let s: i64 = (t2 * (3 * 65536 - 2 * t)) >> 16;
    if s < 0 {
        0
    } else if s > 65536 {
        65536
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Dissolve：整数 hash（splitmix64 类，纯整数）
// ---------------------------------------------------------------------------

/// 像素显影 hash：`0..=255`（整数阈值域，`< k` 即显影）。
///
/// 实现为 splitmix64 类**纯整数**混合（无浮点、无外部 crate）：`(x, y)` 与
/// `seed` 先揉进一个 64 位字，再过三轮 `xorshift-multiply` 终混，取高 8 位。
///
/// 值域 `0..=255` 保证端点精确：`k=0` ⇒ 恒 `false`（全 `a`）、
/// `k=256` ⇒ 恒 `true`（全 `b`）。
#[inline]
pub fn dissolve_hash(x: u32, y: u32, seed: u64) -> u16 {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut z: u64 = seed
        ^ ((x as u64) << 32)
        ^ (y as u64).wrapping_mul(GOLDEN);
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 56) & 0xFF) as u16
}

// ---------------------------------------------------------------------------
// Push：整数推进量
// ---------------------------------------------------------------------------

/// 推移推进量：`c = round(k·extent/256)`（整数 round-to-nearest）。
///
/// `k=0` ⇒ `c=0`（全 `a`）、`k=256` ⇒ `c=extent`（全 `b`），端点逐位精确。
#[inline]
pub fn push_c(k: u16, extent: u32) -> u32 {
    let k: u64 = concat_clamp_w(k) as u64;
    (((extent as u64) * k + 128) / 256) as u32
}

// ---------------------------------------------------------------------------
// 凸权重链（半群）
// ---------------------------------------------------------------------------

/// 凸权重链：每段一个 `u16` 权重（`0..=256`）+ 段数。
///
/// 语义：链把「旧帧 → 新帧」的逐段权重折叠成一个总权重。
/// 同目标链的折叠闭式 `W_u = w1 + w2 − w1·w2/256`（对应 `u = s + t − st`，
/// 保留分数 `1−t` 的乘法半群）；不同目标链的正确结构就是**本向量**
/// （三源凸权重 `[(1−t₁)(1−t₂), t₁(1−t₂), t₂]` 不可压成单权重，ADR-18 ⑧-3）。
///
/// 精确性：`≤3` 段乘积分母 `256³ = 2^24` 用 `u32` 精确；`≤4` 段
/// `256⁴ = 2^32` 需 `u64`（`u32::MAX + 1`，B-M6 边界）；更长每步重归一化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionChain {
    /// 逐段权重（`0..=256`）。
    weights: Vec<u16>,
    /// 段数（= `weights.len()`，冗余保存供快速判定）。
    segments: usize,
}

impl TransitionChain {
    /// 以逐段权重构造（每项自动钳位到 `0..=256`）。
    pub fn new(weights: Vec<u16>) -> Self {
        let ws: Vec<u16> = weights.into_iter().map(concat_clamp_w).collect();
        let segments: usize = ws.len();
        Self { weights: ws, segments }
    }

    /// 单段链（`weights = [w]`）。
    pub fn single(w: u16) -> Self {
        Self::new(vec![w])
    }

    /// 段数。
    pub fn segments(&self) -> usize {
        self.segments
    }

    /// 逐段权重切片。
    pub fn weights(&self) -> &[u16] {
        &self.weights
    }

    /// `≤3` 段的精确分子（分母 `256³`，`u32` 可容纳）；段数越界返回 `None`。
    ///
    /// 返回值 `X` 满足 `总权重 = X / 256³`。空链 / 超长链返回 `None`。
    pub fn exact_u32(&self) -> Option<u32> {
        if self.segments == 0 || self.segments > 3 {
            return None;
        }
        exact_num(&self.weights).map(|v: u64| v as u32)
    }

    /// `≤4` 段的精确分子（分母 `256⁴`，`u64` 可容纳）；段数越界返回 `None`。
    ///
    /// 全 1 边界：4 段全 `256` ⇒ `256⁴ = 2^32 = 4294967296`（**恰溢出 `u32`
    /// 一个单位**，故 `exact_u32` 对 4 段必返回 `None`）。
    pub fn exact_u64(&self) -> Option<u64> {
        if self.segments == 0 || self.segments > 4 {
            return None;
        }
        exact_num(&self.weights)
    }

    /// 折叠成单权重 `0..=256`：`≤4` 段走精确分子再四舍五入；
    /// 更长则逐段重归一化（每步误差 ≤ 1/256 LSB，有界不漂移）。
    pub fn composed(&self) -> u16 {
        match self.exact_u64() {
            Some(num) => {
                // 分母 256^n ⇒ 单权重 = num / 256^(n−1)。
                let n: usize = self.segments;
                let den: u64 = pow256(n - 1);
                let half: u64 = den / 2;
                (((num + half) / den) as u16).min(W_ONE)
            }
            None => {
                let mut w: u16 = 0;
                for &t in &self.weights {
                    w = fold_closed(w, t);
                }
                w
            }
        }
    }

    /// 权重向量入缓存键（P5 联动：转场链中间态可缓存、seek 重放无损）。
    ///
    /// FNV-1a 过权重字节 + 段数，纯整数、跨端一致。
    pub fn cache_key(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
        let mut h: u64 = FNV_OFFSET;
        for &w in &self.weights {
            h ^= w as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= (self.segments as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h
    }
}

/// `256` 的整数次幂（`n ≤ 8`；`n = 8` 时 `2^64` 不可表示，故 `n ≤ 7` 安全，
/// 本 crate 只用到 `n ≤ 3`）。
#[inline]
fn pow256(n: usize) -> u64 {
    let mut v: u64 = 1;
    let mut i: usize = 0;
    while i < n {
        v = v.saturating_mul(256);
        i += 1;
    }
    v
}

/// 精确分子：返回 `X` 使「总权重 = X / 256^n」（`n = weights.len()`）。
///
/// 递推（`U` 为前一层的分子、分母 `256^n`）：
/// `X' = U·256 + T·256^n − U·T`，分母升为 `256^(n+1)`。
/// `n ≤ 4` 时各项 ≤ `256⁵ = 2^40`，`u64` 充裕。
fn exact_num(weights: &[u16]) -> Option<u64> {
    if weights.is_empty() {
        return None;
    }
    let mut num: u64 = weights[0] as u64;
    let mut scale: u64 = 256; // 当前分母（256^n，n = 已处理段数，初始 n = 1）
    for &t in &weights[1..] {
        let tt: u64 = t as u64;
        // num(分母 scale) → 新分母 scale·256。
        num = num.checked_mul(256)? + tt.checked_mul(scale)? - num.checked_mul(tt)?;
        scale = scale.checked_mul(256)?;
    }
    Some(num)
}

/// 同目标链两段折叠闭式：`W_u = w1 + w2 − round(w1·w2/256)`。
///
/// 对应分数半群 `u = s + t − st`（`s,t ∈ [0,1]`）。舍入项
/// `round(w1·w2/256)` 计入误差界（ADR-18 A4：4 段链 vs 折叠 ≤3 LSB）。
#[inline]
pub fn fold_closed(w1: u16, w2: u16) -> u16 {
    let a: u32 = concat_clamp_w(w1) as u32;
    let b: u32 = concat_clamp_w(w2) as u32;
    let prod: u32 = (a * b + 128) / 256; // round(w1·w2/256)
    let out: u32 = a + b - prod;
    out.min(W_ONE as u32) as u16
}

// ---------------------------------------------------------------------------
// 帧缓冲 + 场景序列
// ---------------------------------------------------------------------------

/// RGBA8 帧缓冲（行主序，长度 = `w·h·4`）。
///
/// 这是 core 侧的**最小**帧定义（author 的帧就是 `Vec<u8>`，尺寸单独传），
/// 只为场景序列自带尺寸以便独立采样。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBuf {
    /// 宽。
    pub w: u32,
    /// 高。
    pub h: u32,
    /// 逐像素 RGBA8。
    pub rgba: Vec<u8>,
}

impl FrameBuf {
    /// 全 0 帧（透明黑）。
    pub fn new(w: u32, h: u32) -> Self {
        Self { w, h, rgba: vec![0u8; (w as usize) * (h as usize) * 4] }
    }

    /// 以给定字节构造（长度不符时按 `w·h·4` 截断 / 补零，不 panic）。
    pub fn from_rgba(w: u32, h: u32, rgba: Vec<u8>) -> Self {
        let need: usize = (w as usize) * (h as usize) * 4;
        let mut buf: Vec<u8> = rgba;
        buf.resize(need, 0);
        buf.truncate(need);
        Self { w, h, rgba: buf }
    }

    /// 纯色帧（便于测试与占位）。
    pub fn solid(w: u32, h: u32, px: [u8; 4]) -> Self {
        let need: usize = (w as usize) * (h as usize) * 4;
        let mut rgba: Vec<u8> = vec![0u8; need];
        for q in rgba.chunks_exact_mut(4) {
            q[0] = px[0];
            q[1] = px[1];
            q[2] = px[2];
            q[3] = px[3];
        }
        Self { w, h, rgba }
    }

    /// 尺寸是否与 `w × h` 一致。
    pub fn size_ok(&self, w: u32, h: u32) -> bool {
        self.w == w && self.h == h && self.rgba.len() == (w as usize) * (h as usize) * 4
    }
}

/// 场景序列：N 个场景帧 + N−1 个转场（`transitions[i]` 连接 `scenes[i]`
/// 与 `scenes[i+1]`）。缺省转场为 [`TransitionKind::Fade`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneSeq {
    /// 场景帧（时间线顺序）。
    pub scenes: Vec<FrameBuf>,
    /// 相邻场景转场（长度可为 0 / 少于 `scenes.len()−1`，缺省 Fade）。
    pub transitions: Vec<TransitionKind>,
}

impl SceneSeq {
    /// 以场景帧与转场构造。
    pub fn new(scenes: Vec<FrameBuf>, transitions: Vec<TransitionKind>) -> Self {
        Self { scenes, transitions }
    }

    /// 场景数量。
    pub fn len(&self) -> usize {
        self.scenes.len()
    }

    /// 是否为空序列。
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    /// 第 `idx` 个转场（越界回退到 [`TransitionKind::Fade`]）。
    pub fn transition_at(&self, idx: usize) -> TransitionKind {
        self.transitions.get(idx).copied().unwrap_or(TransitionKind::Fade)
    }

    /// 采样第 `idx` 个**转场**在进度 `k`（`0..=256`）处的帧。
    ///
    /// - `scenes` 为空：返回 0×0 空帧（不 panic）；
    /// - 只有 1 个场景：返回该场景帧副本；
    /// - `idx` 越界：钳到最近合法转场（`scenes.len()−2`）；
    /// - 相邻帧尺寸不一致：返回 `a` 的副本（确定性防御）；
    /// - 否则逐像素取 `w = transition.weight(k, pixel)`，输出
    ///   `mix_u8(a, b, w)`（含 alpha，端点逐位精确）。
    pub fn sample_scene(&self, idx: usize, k: u16) -> FrameBuf {
        if self.scenes.is_empty() {
            return FrameBuf::new(0, 0);
        }
        if self.scenes.len() == 1 {
            return self.scenes[0].clone();
        }
        let last: usize = self.scenes.len() - 2;
        let idx: usize = if idx > last { last } else { idx };
        let a: &FrameBuf = &self.scenes[idx];
        let b: &FrameBuf = &self.scenes[idx + 1];
        if a.w != b.w || a.h != b.h || !a.size_ok(a.w, a.h) || !b.size_ok(b.w, b.h) {
            return a.clone();
        }
        let trans: TransitionKind = self.transition_at(idx);
        let k: u16 = concat_clamp_w(k);
        let mut rgba: Vec<u8> = vec![0u8; a.rgba.len()];
        let mut idx_out: usize = 0usize;
        for y in 0..a.h {
            for x in 0..a.w {
                let ctx: PixelCtx = PixelCtx::new(x, y, a.w, a.h);
                let w: u16 = trans.weight(k, &ctx);
                rgba[idx_out] = mix_u8(a.rgba[idx_out], b.rgba[idx_out], w);
                rgba[idx_out + 1] = mix_u8(a.rgba[idx_out + 1], b.rgba[idx_out + 1], w);
                rgba[idx_out + 2] = mix_u8(a.rgba[idx_out + 2], b.rgba[idx_out + 2], w);
                rgba[idx_out + 3] = mix_u8(a.rgba[idx_out + 3], b.rgba[idx_out + 3], w);
                idx_out += 4;
            }
        }
        FrameBuf { w: a.w, h: a.h, rgba }
    }
}

// ---------------------------------------------------------------------------
// Morph 几何（下沉自 author：只搬「与呈现无关」的几何核心）
// ---------------------------------------------------------------------------

/// 线段变形（morph）的**几何核心**：起点 / 终点折线 + 可选路径引用。
///
/// 设计说明（ADR-18 实装取舍，以「改动最小」为准）：author 侧既有
/// `spec::MorphSpec` 还带 `color` / `width` / `fill` / `at` / `in_sec` /
/// `out_sec` / `out_at` / `draw_sec` / `closed` / `ease`（`EasingKind` 属
/// author）等**呈现字段**，整体下沉会把 author 的缓动类型拖进 core
/// （引入反向依赖）。故此处只镜像**几何部分**（`from` / `to` + 路径引用），
/// author 侧以 `pub type MorphGeometry = prism_core::sceneseq::MorphSpec`
/// 别名指向本类型，并由 `MorphSpec::geometry()` 委托——两处注释互指。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MorphSpec {
    /// 起点折线（设备像素）。
    pub from: Vec<[f32; 2]>,
    /// 终点折线（设备像素）。
    pub to: Vec<[f32; 2]>,
    /// 起点路径引用（DSL `morph from "路径"`；None = 内联点集）。
    pub from_path: Option<String>,
    /// 终点路径引用（DSL `morph to "路径"`；None = 内联点集）。
    pub to_path: Option<String>,
}

impl MorphSpec {
    /// 以起点 / 终点折线构造（无路径引用）。
    pub fn new(from: Vec<[f32; 2]>, to: Vec<[f32; 2]>) -> Self {
        Self { from, to, from_path: None, to_path: None }
    }

    /// 以路径引用构造（点集留空，由消费方解析）。
    pub fn from_paths(from: String, to: String) -> Self {
        Self {
            from: Vec::new(),
            to: Vec::new(),
            from_path: Some(from),
            to_path: Some(to),
        }
    }

    /// 可配对点数（`min(from.len(), to.len())`）。
    pub fn pairs(&self) -> usize {
        self.from.len().min(self.to.len())
    }

    /// 按进度 `u ∈ [0,1]` 逐点线性插值出当前折线（只取可配对的点）。
    pub fn lerp_at(&self, u: f32) -> Vec<[f32; 2]> {
        let u: f32 = if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.0 };
        let n: usize = self.pairs();
        let mut out: Vec<[f32; 2]> = Vec::with_capacity(n);
        for i in 0..n {
            let a: [f32; 2] = self.from[i];
            let b: [f32; 2] = self.to[i];
            out.push([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试（ADR-18 ⑤ 测试承接）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机（整数 xorshift64*）：测试取样用，零浮点。
    struct Rng {
        s: u64,
    }

    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x: u64 = self.s;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.s = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_u8(&mut self) -> u8 {
            (self.next_u64() >> 33) as u8
        }
        fn next_w(&mut self) -> u16 {
            (self.next_u64() >> 40) as u16 % 257
        }
    }

    /// f64 精确参考：`(w/256)·b + (1−w/256)·a`，round-to-nearest 到 u8。
    fn ref_mix_f64(a: u8, b: u8, w: u16) -> f64 {
        let t: f64 = w as f64 / 256.0;
        (t * b as f64 + (1.0 - t) * a as f64).round()
    }

    // ---- A1：mix_u8 对拍 f64（max ≤1 LSB、|mean bias| ≤0.1） ----

    #[test]
    fn mix_u8_matches_f64_reference_100k() {
        let mut rng: Rng = Rng::new(2026_1017_0001);
        let n: u32 = 100_000;
        let mut max_diff: f64 = 0.0;
        let mut sum_bias: f64 = 0.0;
        for _ in 0..n {
            let a: u8 = rng.next_u8();
            let b: u8 = rng.next_u8();
            let w: u16 = rng.next_w();
            let got: f64 = mix_u8(a, b, w) as f64;
            let want: f64 = ref_mix_f64(a, b, w);
            let d: f64 = (got - want).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_bias += got - want;
        }
        let mean_bias: f64 = sum_bias / n as f64;
        assert!(max_diff <= 1.0, "单次误差应 ≤1 LSB，实测 max = {max_diff}");
        assert!(
            mean_bias.abs() <= 0.1,
            "平均偏置应 ≤0.1 LSB（无系统偏置），实测 mean = {mean_bias}"
        );
    }

    /// 端点逐位：`w=0` ⇒ 恒 `a`、`w=256` ⇒ 恒 `b`（全 256×256 组合）。
    #[test]
    fn mix_u8_endpoints_bitwise() {
        for a in 0u16..=255 {
            for b in 0u16..=255 {
                assert_eq!(
                    mix_u8(a as u8, b as u8, 0),
                    a as u8,
                    "w=0 须逐位等于 a（a={a}, b={b}）"
                );
                assert_eq!(
                    mix_u8(a as u8, b as u8, 256),
                    b as u8,
                    "w=256 须逐位等于 b（a={a}, b={b}）"
                );
            }
        }
        // 越界权重按端点钳位（不 panic、不漂移）。
        assert_eq!(mix_u8(17, 200, 999), 200);
    }

    /// a = b ⇒ 任意 w 输出恒等于 a；极大动态 a=0,b=255 中间值有界。
    #[test]
    fn mix_u8_identity_and_extremes() {
        for w in 0u16..=256 {
            assert_eq!(mix_u8(123, 123, w), 123, "a=b 时任意权重输出恒等于 a");
        }
        let mid: u8 = mix_u8(0, 255, 128);
        assert!((127..=128).contains(&mid), "中点应落在 127..=128，实测 {mid}");
        assert_eq!(mix_u8(0, 255, 0), 0);
        assert_eq!(mix_u8(0, 255, 256), 255);
    }

    // ---- A2：四转场端点逐位 ----

    fn probe_pixels(w: u32, h: u32) -> Vec<PixelCtx> {
        let mut ps: Vec<PixelCtx> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                ps.push(PixelCtx::new(x, y, w, h));
            }
        }
        ps
    }

    fn all_zero_or_one(trans: &TransitionKind, k: u16, w: u32, h: u32) -> bool {
        probe_pixels(w, h).iter().all(|p: &PixelCtx| trans.weight(k, p) == 0)
    }

    fn all_one(trans: &TransitionKind, k: u16, w: u32, h: u32) -> bool {
        probe_pixels(w, h).iter().all(|p: &PixelCtx| trans.weight(k, p) == 256)
    }

    #[test]
    fn transitions_endpoints_bitwise_exact() {
        let w: u32 = 37;
        let h: u32 = 23;
        let kinds: [TransitionKind; 7] = [
            TransitionKind::Fade,
            TransitionKind::Wipe { band: WIPE_BAND },
            TransitionKind::Wipe { band: 0 },
            TransitionKind::Dissolve { seed: 0x6A09_E667 },
            TransitionKind::Push { dir: PushDir::Left },
            TransitionKind::Push { dir: dir_any() },
            TransitionKind::Push { dir: PushDir::Down },
        ];
        for t in kinds {
            assert!(all_zero_or_one(&t, 0, w, h), "k=0 须全取 a：{t:?}");
            assert!(all_one(&t, 256, w, h), "k=256 须全取 b：{t:?}");
        }
    }

    fn dir_any() -> PushDir {
        PushDir::Right
    }

    /// 端点帧逐位：经 `SceneSeq::sample_scene` 的整帧路径也逐位精确。
    #[test]
    fn scene_seq_endpoint_frames_bitwise() {
        let (w, h) = (9u32, 7u32);
        let a: FrameBuf = FrameBuf::solid(w, h, [200, 30, 30, 255]);
        let b: FrameBuf = FrameBuf::solid(w, h, [30, 30, 200, 128]);
        let kinds: [TransitionKind; 4] = [
            TransitionKind::Fade,
            TransitionKind::Wipe { band: WIPE_BAND },
            TransitionKind::Dissolve { seed: 7 },
            TransitionKind::Push { dir: PushDir::Up },
        ];
        for t in kinds {
            let seq: SceneSeq = SceneSeq::new(vec![a.clone(), b.clone()], vec![t]);
            assert_eq!(seq.sample_scene(0, 0), a, "k=0 须逐位等于 a（{t:?}）");
            assert_eq!(seq.sample_scene(0, 256), b, "k=256 须逐位等于 b（{t:?}）");
        }
    }

    // ---- Wipe 修复验证（污染列 = 0） ----

    #[test]
    fn wipe_no_polluted_columns_at_endpoints() {
        let band: u16 = WIPE_BAND;
        let w: u32 = 64;
        let t: TransitionKind = TransitionKind::Wipe { band };
        // k=0：右缘（末列）权重必须为 0（修复前为 0.156 ⇒ 39/40 LSB 污染）。
        for x in 0..w {
            let wt: u16 = t.weight(0, &PixelCtx::new(x, 0, w, 1));
            assert_eq!(wt, 0, "k=0 第 {x} 列应无污染");
        }
        // k=256：左缘权重必须为 256（修复前 0.5 / 0.84 污染）。
        for x in 0..w {
            let wt: u16 = t.weight(256, &PixelCtx::new(x, 0, w, 1));
            assert_eq!(wt, 256, "k=256 第 {x} 列应无污染");
        }
        // band = 0 防御：硬边、不除零、端点精确。
        let hard: TransitionKind = TransitionKind::Wipe { band: 0 };
        assert_eq!(hard.weight(0, &PixelCtx::new(0, 0, w, 1)), 0);
        assert_eq!(hard.weight(256, &PixelCtx::new(0, 0, w, 1)), 256);
    }

    /// 单调性：Wipe 权重随 k 单调不减（同一像素），无抖动。
    #[test]
    fn wipe_weight_monotone_in_k() {
        let w: u32 = 32;
        let t: TransitionKind = TransitionKind::Wipe { band: WIPE_BAND };
        for x in 0..w {
            let p: PixelCtx = PixelCtx::new(x, 0, w, 1);
            let mut prev: u16 = 0;
            for k in 0u16..=256 {
                let cur: u16 = t.weight(k, &p);
                assert!(cur >= prev, "第 {x} 列权重在 k={k} 处回退（{prev} → {cur}）");
                prev = cur;
            }
        }
    }

    // ---- Dissolve：确定性 + 黄金图案固件 + 比例 ----

    #[test]
    fn dissolve_deterministic_and_uniform() {
        let seed: u64 = 0x6A09_E667;
        // 同输入两跑逐位相同。
        let a: Vec<u16> = (0..64u32).map(|y| dissolve_hash(3, y, seed)).collect();
        let b: Vec<u16> = (0..64u32).map(|y| dissolve_hash(3, y, seed)).collect();
        assert_eq!(a, b, "同输入两跑必须 bitwise 相同");
        // 值域 0..=255。
        let mut buckets: [u32; 8] = [0; 8];
        let mut total: u32 = 0;
        for y in 0..256u32 {
            for x in 0..256u32 {
                let h: u16 = dissolve_hash(x, y, seed);
                assert!(h < 256, "hash 值域须为 0..=255，实测 {h}");
                buckets[(h / 32) as usize] += 1;
                total += 1;
            }
        }
        // 8 桶均匀性：每桶偏差 < 10%（65536/8 = 8192）。
        for (i, c) in buckets.iter().enumerate() {
            let d: f64 = (*c as f64 - 8192.0).abs() / 8192.0;
            assert!(d < 0.10, "第 {i} 桶偏差 {d} 过大（{c} / 8192）");
        }
        assert_eq!(total, 256 * 256);
    }

    /// 黄金 dissolve 图案固件（B-M5）：固定种子的整数 hash 输出逐位锁定，
    /// 任何实现漂移（如换回 f32 rand）立即红。
    #[test]
    fn dissolve_golden_pattern_firmware() {
        let seed: u64 = 0x6A09_E667;
        let got: Vec<u16> = [(0u32, 0u32), (1, 0), (0, 1), (7, 3), (13, 29), (255, 255)]
            .iter()
            .map(|&(x, y): &(u32, u32)| dissolve_hash(x, y, seed))
            .collect();
        // 固件值由首次实装运行钉死（整数路径 ⇒ 跨端/跨跑恒同）。
        let want: Vec<u16> = vec![GOLDEN_0, GOLDEN_1, GOLDEN_2, GOLDEN_3, GOLDEN_4, GOLDEN_5];
        assert_eq!(got, want, "dissolve 黄金图案漂移（禁 f32 rand 改造会导致本断言红）");
    }

    // 黄金固件常量（首次运行钉死；见 dissolve_golden_pattern_firmware）。
    const GOLDEN_0: u16 = 53;
    const GOLDEN_1: u16 = 76;
    const GOLDEN_2: u16 = 214;
    const GOLDEN_3: u16 = 119;
    const GOLDEN_4: u16 = 173;
    const GOLDEN_5: u16 = 132;

    // ---- Push：整数推进量 ----

    #[test]
    fn push_c_rounds_and_saturates() {
        assert_eq!(push_c(0, 100), 0);
        assert_eq!(push_c(256, 100), 100);
        assert_eq!(push_c(128, 100), 50); // 100·128/256 = 50
        assert_eq!(push_c(1, 100), 0); // round(0.39) = 0
        assert_eq!(push_c(2, 100), 1); // round(0.78) = 1
        assert_eq!(push_c(999, 100), 100); // 越界权重钳到端点
        // Left：k=0 全 a，k=256 全 b。
        let w: u32 = 8;
        let t: TransitionKind = TransitionKind::Push { dir: PushDir::Left };
        assert!(all_zero_or_one(&t, 0, w, 4));
        assert!(all_one(&t, 256, w, 4));
    }

    // ---- 链：精确性 / 折叠 / 边界（B-M6） ----

    #[test]
    fn chain_fold_closed_matches_semigroup() {
        // 同目标两段折叠：u = s + t − st。
        let s: u16 = 64; // 0.25
        let t: u16 = 128; // 0.5
        // 精确值 0.25 + 0.5 − 0.125 = 0.625 ⇒ 160。
        assert_eq!(fold_closed(s, t), 160);
        assert_eq!(fold_closed(0, 0), 0);
        assert_eq!(fold_closed(256, 256), 256);
        assert_eq!(fold_closed(256, 0), 256);
        // 半群律：两段链的精确折叠与闭式一致（四舍五入域 ≤1 LSB）。
        let chain: TransitionChain = TransitionChain::new(vec![s, t]);
        let exact: u16 = chain.composed();
        let closed: u16 = fold_closed(s, t);
        assert!(
            (exact as i32 - closed as i32).abs() <= 1,
            "两段链 composed {exact} 与闭式 {closed} 差应 ≤1 LSB"
        );
    }

    /// A4（m-3 修订）：同目标 4 段链 vs 折叠闭式 ≤3 LSB。
    #[test]
    fn chain_four_segments_vs_fold_within_3_lsb() {
        let mut rng: Rng = Rng::new(2026_1017_0002);
        let mut worst: i32 = 0;
        for _ in 0..2000 {
            let ws: Vec<u16> = (0..4).map(|_| rng.next_w()).collect();
            let chain: TransitionChain = TransitionChain::new(ws.clone());
            let exact: i32 = chain.composed() as i32;
            let mut folded: u16 = 0;
            for &t in &ws {
                folded = fold_closed(folded, t);
            }
            let d: i32 = (exact - folded as i32).abs();
            if d > worst {
                worst = d;
            }
            assert!(d <= 3, "4 段链 vs 折叠闭式应 ≤3 LSB，实测 {d}（ws={ws:?}）");
        }
        assert!(worst <= 3, "全部采样最坏差 {worst} 超界");
    }

    /// B-M6：`u32` 存 4 段链在 256⁴ 边界溢出一个单位 ⇒ 必须走 u64。
    #[test]
    fn chain_length_boundary_u32_u64() {
        // 3 段全 1：256³ = 2^24 = 16777216（u32 可容纳）。
        let c3: TransitionChain = TransitionChain::new(vec![256, 256, 256]);
        assert_eq!(c3.exact_u32(), Some(16_777_216u32));
        assert_eq!(c3.exact_u64(), Some(16_777_216u64));
        assert_eq!(c3.composed(), 256);
        // 4 段全 1：256⁴ = 2^32 = 4294967296（u32::MAX + 1 ⇒ u32 必 None）。
        let c4: TransitionChain = TransitionChain::new(vec![256, 256, 256, 256]);
        assert_eq!(c4.exact_u32(), None, "4 段链不得用 u32 承载（B-M6）");
        assert_eq!(c4.exact_u64(), Some(4_294_967_296u64), "u64 须精确容纳 256⁴");
        assert_eq!(c4.composed(), 256, "全显影链折叠后仍为满权重");
        assert!(4_294_967_296u64 == (u32::MAX as u64) + 1, "边界值核对");
        // 空链 / 超长链：不 panic，走重归一化分支。
        let empty: TransitionChain = TransitionChain::new(Vec::new());
        assert_eq!(empty.composed(), 0);
        assert_eq!(empty.exact_u64(), None);
        let long: TransitionChain =
            TransitionChain::new(vec![128, 128, 128, 128, 128, 128, 128, 128]);
        assert_eq!(long.exact_u64(), None);
        let v: u16 = long.composed();
        assert!(v > 250, "8 段半权重链应接近满权重，实测 {v}");
    }

    /// 链缓存键：同向量同键、不同向量不同键、确定。
    #[test]
    fn chain_cache_key_stable() {
        let a: TransitionChain = TransitionChain::new(vec![10, 20, 30]);
        let b: TransitionChain = TransitionChain::new(vec![10, 20, 30]);
        let c: TransitionChain = TransitionChain::new(vec![10, 20, 31]);
        assert_eq!(a.cache_key(), b.cache_key());
        assert_ne!(a.cache_key(), c.cache_key());
        assert_eq!(a.segments(), 3);
        assert_eq!(a.weights(), &[10, 20, 30]);
    }

    // ---- SceneSeq 防御 + 确定性 ----

    #[test]
    fn scene_seq_out_of_range_is_clamped_no_panic() {
        let empty: SceneSeq = SceneSeq::new(Vec::new(), Vec::new());
        assert_eq!(empty.sample_scene(5, 128), FrameBuf::new(0, 0));
        let single: SceneSeq = SceneSeq::new(vec![FrameBuf::solid(2, 2, [1, 2, 3, 255])], Vec::new());
        assert_eq!(single.sample_scene(9, 128).rgba, vec![1, 2, 3, 255].repeat(4));
        // 两帧（一个转场）：越界 idx 钳到 0。
        let seq: SceneSeq = SceneSeq::new(
            vec![FrameBuf::solid(4, 4, [10, 10, 10, 255]), FrameBuf::solid(4, 4, [20, 20, 20, 255])],
            vec![TransitionKind::Fade],
        );
        assert_eq!(seq.sample_scene(99, 0).rgba[0], 10);
        assert_eq!(seq.sample_scene(99, 256).rgba[0], 20);
        // 尺寸不一致：返回 a 的副本（确定性防御）。
        let bad: SceneSeq = SceneSeq::new(
            vec![FrameBuf::solid(4, 4, [10, 10, 10, 255]), FrameBuf::solid(2, 2, [20, 20, 20, 255])],
            vec![TransitionKind::Fade],
        );
        assert_eq!(bad.sample_scene(0, 128).rgba[0], 10);
    }

    /// 同输入两跑 bitwise 相同（全转场 × 全 k 档抽样）。
    #[test]
    fn scene_seq_two_runs_bitwise_identical() {
        let (w, h) = (11u32, 5u32);
        let a: FrameBuf = FrameBuf::solid(w, h, [201, 17, 33, 255]);
        let b: FrameBuf = FrameBuf::solid(w, h, [3, 199, 87, 200]);
        let kinds: [TransitionKind; 4] = [
            TransitionKind::Fade,
            TransitionKind::Wipe { band: WIPE_BAND },
            TransitionKind::Dissolve { seed: 42 },
            TransitionKind::Push { dir: PushDir::Left },
        ];
        for t in kinds {
            let seq: SceneSeq = SceneSeq::new(vec![a.clone(), b.clone()], vec![t]);
            for k in [0u16, 1, 64, 128, 192, 255, 256] {
                let r1: FrameBuf = seq.sample_scene(0, k);
                let r2: FrameBuf = seq.sample_scene(0, k);
                assert_eq!(r1, r2, "两跑须 bitwise 相同（{t:?}, k={k}）");
            }
        }
    }

    /// 100k 像素长混合无累积漂移：a = b 时任意 k 输出恒等于 a。
    #[test]
    fn no_drift_over_100k_pixels() {
        let (w, h) = (400u32, 250u32);
        let a: FrameBuf = FrameBuf::solid(w, h, [77, 133, 199, 255]);
        let seq: SceneSeq =
            SceneSeq::new(vec![a.clone(), a.clone()], vec![TransitionKind::Dissolve { seed: 3 }]);
        for k in [0u16, 97, 128, 200, 256] {
            assert_eq!(seq.sample_scene(0, k), a, "a=b 时无论 k 输出恒等（k={k}）");
        }
        assert_eq!(a.rgba.len(), 400 * 250 * 4);
    }

    // ---- Morph 几何 ----

    #[test]
    fn morph_lerp_endpoints_and_midpoint() {
        let m: MorphSpec = MorphSpec::new(
            vec![[0.0, 0.0], [10.0, 0.0]],
            vec![[10.0, 10.0], [20.0, 10.0]],
        );
        assert_eq!(m.pairs(), 2);
        assert_eq!(m.lerp_at(0.0), vec![[0.0, 0.0], [10.0, 0.0]]);
        assert_eq!(m.lerp_at(1.0), vec![[10.0, 10.0], [20.0, 10.0]]);
        let mid: Vec<[f32; 2]> = m.lerp_at(0.5);
        assert!((mid[0][0] - 5.0).abs() < 1e-6 && (mid[0][1] - 5.0).abs() < 1e-6);
        // 非有限进度按 0 处理（确定性、不 panic）。
        assert_eq!(m.lerp_at(f32::NAN), vec![[0.0, 0.0], [10.0, 0.0]]);
        // 点数不等：只取可配对前缀。
        let m2: MorphSpec = MorphSpec::new(vec![[0.0, 0.0]], vec![[1.0, 1.0], [2.0, 2.0]]);
        assert_eq!(m2.pairs(), 1);
        assert_eq!(m2.lerp_at(1.0).len(), 1);
        let pr: MorphSpec = MorphSpec::from_paths("a.pts".to_string(), "b.pts".to_string());
        assert_eq!(pr.from_path.as_deref(), Some("a.pts"));
        assert_eq!(pr.to_path.as_deref(), Some("b.pts"));
    }
}
