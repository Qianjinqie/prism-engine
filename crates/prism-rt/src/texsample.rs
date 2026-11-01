//! 位图纹理采样器（W1-B / 选型报告 §2.2 定稿方案 C1）。
//!
//! # 一条样本的完整路径
//!
//! ```text
//! uv + 解析足迹(Footprint) ──► 2×2 奇异值分解 ──► (σ_max, σ_min, 主轴)
//!        │                                            │
//!        │                                            ├─ lod  = log2(max(σ_min, σ_max/taps))
//!        │                                            └─ taps = clamp(⌈σ_max/σ_min⌉, 1, 4)
//!        └──► taps 次「三线性采样」（双线性 × 相邻 mip 级 lerp）沿主轴等距取点，取均值
//! ```
//!
//! # 三个关键决定（以及为什么）
//!
//! 1. **足迹必须解析算，不能省。** 选型报告 §2.1 已定量否决"随机纹理采样替代
//!    足迹"（`N ≥ 16256·ΔL²`：ΔL=0.25 要 1016 spp）。本模块的足迹是**每样本
//!    纯函数**——由射线锥半角 + 命中距离 + 表面 uv 雅可比算出，**不碰
//!    `dpdx`/`dpdy`**（后者按 2×2 quad 求导，跨后端/跨 tile 边界不可复现，
//!    见仓库 §1.2 全局红线）。
//! 2. **各向异性走"有界 N≤4 tap"，不是 EWA 的加权核。** 4 tap 沿主轴等距、
//!    权重相同 ⇒ 求和顺序固定、每 tap 独立 ⇒ 确定性天然成立；代价是残余盒宽
//!    `σ_max/taps`（而非理想椭圆），收益仍然是把掠射端的保留频带从 `1/ρ`
//!    推到 `taps/ρ`（V3 实测见 §15）。
//! 3. **mip 级由短轴决定。** 长轴靠 tap 覆盖，短轴靠 mip 覆盖。这是"不糊"
//!    的根据：若拿长轴选级（各向同性做法），短轴方向会被过度模糊
//!    `σ_max/σ_min` 倍。
//!
//! # 平铺与缝
//!
//! 采样恒为 **repeat**（`wrap(i, n) = ((i % n) + n) % n`）：uv 的小数部分
//! 与整数部分一起参与，故跨 `u = 1.0` 时取到的是**邻居纹素**而不是钳制值
//! ——缝两侧梯度连续（V5）。双线性在缝上跨行取 `x1 = 0`，与域内取
//! `x1 = x0+1` 是同一式子，没有第二套边界分支。
//!
//! # 与 sRGB 的关系
//!
//! 本模块**不解码**：`Scene::texels` 恒为线性 f32（解码在
//! [`crate::texture::build_rgba8`] 里一次性做完，走同一份 256 项 LUT）。
//! 于是三后端读到的是同一份线性位型，滤波也发生在**线性域**（这是物理上
//! 正确的做法：在编码域做双线性会让暗部偏亮）。

use crate::texture::{handle_index, TextureRef};

/// 单个纹素（与 GPU `array<vec4<f32>>` 的元素逐格对应）。
pub type Texel = [f32; 4];

/// 纹理表视图（`Scene::textures` + `Scene::texels` 的成对借用）。
///
/// 采样必须同时看到"描述"与"像素"，故两者绑成一个 Copy 视图传参，
/// 而不是把 `Scene` 传进采样器（采样器不该知道场景）。
#[derive(Debug, Clone, Copy)]
pub struct TexTable<'a> {
    /// `Scene::textures`（描述表；下标 = 句柄 − 1）。
    pub refs: &'a [TextureRef],
    /// `Scene::texels`（扁平线性纹素；`TextureRef::offset` 为起点）。
    pub texels: &'a [Texel],
}

impl<'a> TexTable<'a> {
    /// 构造视图。
    pub const fn new(refs: &'a [TextureRef], texels: &'a [Texel]) -> Self {
        TexTable { refs, texels }
    }

    /// 空表（无纹理场景的零成本占位）。
    pub const EMPTY: TexTable<'static> = TexTable { refs: &[], texels: &[] };

    /// 句柄 → 描述（`0` / 越界 → `None`）。
    #[inline]
    pub fn get(&self, handle: u32) -> Option<&'a TextureRef> {
        self.refs.get(handle_index(handle)?)
    }
}

/// uv 空间的足迹：世界足迹的两个半轴向量在 **uv 空间**的像。
///
/// 由 [`crate::integrator::Scene::footprint_at`] 算出（射线锥 × uv 雅可比）。
/// 两列向量不要求正交——真正的半轴由 2×2 奇异值分解给出（见
/// [`Footprint::singular_axes`]），这正是掠射椭圆足迹的标准形式（EWA）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    /// 世界第 1 切向轴（视线在切平面上的投影）的 uv 像。
    pub c0: [f32; 2],
    /// 世界第 2 切向轴（`n × T1`）的 uv 像。
    pub c1: [f32; 2],
}

impl Footprint {
    /// 零足迹（= 点采样：1 tap、lod 0）。
    pub const ZERO: Footprint = Footprint { c0: [0.0, 0.0], c1: [0.0, 0.0] };

    /// 由两列构造。
    pub const fn new(c0: [f32; 2], c1: [f32; 2]) -> Self {
        Footprint { c0, c1 }
    }

    /// 2×2 奇异值分解：返回 `(σ_max, 主轴单位向量, σ_min)`。
    ///
    /// `M = [c0 c1]`，`MᵀM = [[a,b],[b,c]]`（`a = c0·c0`、`b = c0·c1`、
    /// `c = c1·c1`）；`λ± = (a+c)/2 ± √(((a−c)/2)² + b²)`，`σ± = √λ±`。
    /// 主轴 = `λ+` 的特征向量。全部只含 `± × ÷ √`（IEEE 精确舍入运算），
    /// 跨平台逐位一致。
    ///
    /// 退化（零足迹 / 共线）时返回 `σ_max = σ_min = 0` 或 0 长度主轴——
    /// 调用方据 `σ_max == 0` 走"1 tap、lod 0"的点采样分支。
    pub fn singular_axes(&self) -> (f32, [f32; 2], f32) {
        let a = self.c0[0] * self.c0[0] + self.c0[1] * self.c0[1];
        let b = self.c0[0] * self.c1[0] + self.c0[1] * self.c1[1];
        let c = self.c1[0] * self.c1[0] + self.c1[1] * self.c1[1];
        let half = (a + c) * 0.5;
        let disc = ((a - c) * 0.5) * ((a - c) * 0.5) + b * b;
        let root = disc.max(0.0).sqrt();
        let lam_max = (half + root).max(0.0);
        let lam_min = (half - root).max(0.0);
        let s_max = lam_max.sqrt();
        let s_min = lam_min.sqrt();
        // λ+ 的特征向量：优先用非对角项 b（b≠0 时两支都成立），
        // b == 0 时退化为坐标轴（取 a 较大的那支）。
        let dir = if b.abs() > 0.0 {
            [b, lam_max - a]
        } else if a >= c {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let dir = if len > 1e-20 {
            [dir[0] / len, dir[1] / len]
        } else {
            // 各向同性（σ_max == σ_min）：方向无意义，取 +u 保持确定性。
            [1.0, 0.0]
        };
        (s_max, dir, s_min)
    }
}

/// 采样选项。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleOpts {
    /// 各向异性 tap 上限（4 = 定稿；1 = 退化成"仅 mip"的各向同性对照）。
    pub max_taps: u32,
}

impl SampleOpts {
    /// 定稿默认：4 tap。
    pub const DEFAULT: SampleOpts = SampleOpts { max_taps: MAX_ANISO_TAPS };
    /// 各向同性对照（仅 mip）：V3 的对照组。
    pub const ISOTROPIC: SampleOpts = SampleOpts { max_taps: 1 };
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts::DEFAULT
    }
}

/// 各向异性 tap 上限（选型报告 §2 的 `N ≤ 4`）。
pub const MAX_ANISO_TAPS: u32 = 4;

/// 采样诊断（测试与 §15 的数字采集用）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleInfo {
    /// 采样结果（线性域）。
    pub rgba: Texel,
    /// 实际使用的 mip 级（可为小数：三线性）。
    pub lod: f32,
    /// 实际 tap 数。
    pub taps: u32,
    /// 各向异性比 `σ_max/σ_min`（未钳制）。
    pub ratio: f32,
}

// ---------------------------------------------------------------------------
// 基础件
// ---------------------------------------------------------------------------

/// `floor(x)` 的 `i64` 版本（uv 可能超出 `[0,1)` 很多圈）。
#[inline]
fn floor_i(x: f32) -> i64 {
    let f = x.floor();
    // f32 → i64 的饱和转换（Rust 的 `as` 在越界时饱和，无 UB）；
    // 极端 uv（|u| > 2^63）在渲染里不可达，钳制只为不产生错位。
    if f >= 9.0e18 {
        9_000_000_000_000_000_000
    } else if f <= -9.0e18 {
        -9_000_000_000_000_000_000
    } else {
        f as i64
    }
}

/// 平铺取模：`((i % n) + n) % n`（`n ≥ 1`）。
#[inline]
fn wrap_i(i: i64, n: i64) -> i64 {
    debug_assert!(n >= 1);
    let m = i % n;
    if m < 0 {
        m + n
    } else {
        m
    }
}

/// 第 `level` 级的尺寸（每级减半，最小 1）。
#[inline]
pub fn level_dims(r: &TextureRef, level: u32) -> (usize, usize) {
    let shift = level.min(31);
    let w = (r.w.max(1) >> shift).max(1) as usize;
    let h = (r.h.max(1) >> shift).max(1) as usize;
    (w, h)
}

/// 第 `level` 级在扁平纹素流里的起点（`TextureRef::offset` + 前序各级之和）。
#[inline]
pub fn level_base(r: &TextureRef, level: u32) -> usize {
    let mut base = r.offset as usize;
    let lv = level.min(r.levels.max(1));
    let mut l = 0u32;
    while l < lv {
        let (w, h) = level_dims(r, l);
        base += w * h;
        l += 1;
    }
    base
}

/// `log2(x)`（`x > 0`）：定点分解 + `atanh` 级数，只含 `± × ÷` 与位操作。
///
/// 归约到尾数 `m ∈ [√½, √2)`（`|z| = |(m−1)/(m+1)| ≤ 0.172`），级数截断在
/// `z^11`（相对误差 ~2e-10）——mip 级选择只需要 ~1e-6 的精度。
#[inline]
fn log2_f32(x: f32) -> f32 {
    if !(x > 0.0) || !x.is_finite() {
        return 0.0;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // [1, 2)
    // 归约到 [√½, √2)：|z| 从 1/3 压到 0.172。
    if m > 1.414_213_6 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let ln_m = 2.0
        * z
        * (1.0 + z2 * (1.0 / 3.0 + z2 * (1.0 / 5.0 + z2 * (1.0 / 7.0 + z2 * (1.0 / 9.0)))));
    e as f32 + ln_m * std::f32::consts::LOG2_E
}

/// 单级双线性采样（repeat 平铺）。
///
/// 纹素中心在 `(i+0.5)/n`：故 `x = u·w − 0.5` 的整数部分即左邻纹素。
/// **uv 恰落在纹素中心时 `frac == 0`，返回该纹素原值（逐位）**——这是 V1
/// 的机械保证：没有半像素偏移，也没有 0.5 权重的混合。
#[inline]
fn bilinear(r: &TextureRef, texels: &[Texel], level: u32, uv: [f32; 2]) -> Texel {
    let (w, h) = level_dims(r, level);
    let base = level_base(r, level);
    let (wf, hf) = (w as f32, h as f32);
    let x = uv[0] * wf - 0.5;
    let y = uv[1] * hf - 0.5;
    let x0 = floor_i(x);
    let y0 = floor_i(y);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    // repeat：跨缝取邻居（不是钳制）——V5 的根据。
    let i0 = wrap_i(x0, w as i64) as usize;
    let i1 = wrap_i(x0 + 1, w as i64) as usize;
    let j0 = wrap_i(y0, h as i64) as usize;
    let j1 = wrap_i(y0 + 1, h as i64) as usize;
    let f = |i: usize, j: usize| -> Texel {
        let idx = base + j * w + i;
        // 越界 = `texels` 与 `textures` 对不上（构造期错误）；返回黑而不是 panic，
        // 以免渲染线程因一张坏纹理整体失败。`Scene::validate_layout` 拦这类错。
        if idx < texels.len() {
            texels[idx]
        } else {
            [0.0; 4]
        }
    };
    let t00 = f(i0, j0);
    let t10 = f(i1, j0);
    let t01 = f(i0, j1);
    let t11 = f(i1, j1);
    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        // 固定结合序：先沿 u 后沿 v（跨后端同式）。
        let a = t00[ch] + (t10[ch] - t00[ch]) * fx;
        let b = t01[ch] + (t11[ch] - t01[ch]) * fx;
        out[ch] = a + (b - a) * fy;
    }
    out
}

/// 三线性采样：两级双线性 + 级间 lerp（`lod` 可为小数）。
#[inline]
fn trilinear(r: &TextureRef, texels: &[Texel], uv: [f32; 2], lod: f32) -> Texel {
    let max_level = r.levels.max(1) as f32 - 1.0;
    let lod = lod.clamp(0.0, max_level);
    let l0 = floor_i(lod) as u32;
    let frac = lod - l0 as f32;
    let a = bilinear(r, texels, l0, uv);
    if frac <= 0.0 {
        // 级内精确命中：不混下一级（V1 的逐位前提）。
        return a;
    }
    let b = bilinear(r, texels, (l0 + 1).min(r.levels.max(1) - 1), uv);
    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        out[ch] = a[ch] + (b[ch] - a[ch]) * frac;
    }
    out
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------

/// 按足迹采样（定稿路径：双线性 + mip 三级线性 + 有界各向异性）。
///
/// 返回 `None` 当且仅当句柄为 0 / 越界（无贴图）。
pub fn sample(table: &TexTable<'_>, handle: u32, uv: [f32; 2], foot: Footprint) -> Option<Texel> {
    sample_with(table, handle, uv, foot, SampleOpts::DEFAULT).map(|i| i.rgba)
}

/// 采样 + 诊断（测试 / §15 数字采集用；产品路径用 [`sample`]）。
pub fn sample_with(
    table: &TexTable<'_>,
    handle: u32,
    uv: [f32; 2],
    foot: Footprint,
    opts: SampleOpts,
) -> Option<SampleInfo> {
    let r = table.get(handle)?;
    let (w, h) = (r.w.max(1) as f32, r.h.max(1) as f32);
    // 足迹换到**纹素空间**再做 SVD：σ 的单位直接就是纹素，lod 无需换算。
    let ft = Footprint::new([foot.c0[0] * w, foot.c0[1] * h], [foot.c1[0] * w, foot.c1[1] * h]);
    let (s_max, dir, s_min) = ft.singular_axes();
    if !(s_max > 0.0) {
        // 零足迹：点采样（1 tap、lod 0）——构造性精确，V1 走这条。
        return Some(SampleInfo {
            rgba: bilinear(r, table.texels, 0, uv),
            lod: 0.0,
            taps: 1,
            ratio: 1.0,
        });
    }
    let ratio = if s_min > 0.0 { s_max / s_min } else { f32::INFINITY };
    let taps = if ratio.is_finite() {
        (ratio.ceil() as u32).clamp(1, opts.max_taps.max(1))
    } else {
        opts.max_taps.max(1)
    };
    // ---- tap 布局：**并集恰好等于椭圆长轴**（不多不少）----
    //
    // 每个 tap 是 `tri_lod` 级上的一个盒式盒（宽 `2^lod`）。要让 `taps` 个盒
    // 拼成一个 `σ_max` 长、`tap_w` 宽的矩形，必须：
    //
    //   tap_w = max(σ_min, σ_max / taps)     // 每 tap 宽度（钳制时被迫加宽）
    //   lod   = log2(tap_w)                  // 盒宽 → mip 级
    //   span  = σ_max − tap_w                // 首末 tap **中心**的间距
    //
    // 于是并集 = span + tap_w = σ_max（钳制时 span + tap_w = σ_max 亦然）。
    //
    // # 为什么不是"跨度 σ_max、每 tap 宽 σ_max/taps"
    //
    // 那个布局在 `taps = 2` 且 `ratio ≈ 1` 时（各向同性足迹刚好被
    // `ceil(ratio)` 分成 2 个 tap），两个 tap 各自是**整宽** σ_min 的盒、
    // 中心相距 σ/2 ⇒ 并集 1.5–2×σ ⇒ 系统性过糊一级（§15.3 实测：1 圈/图
    // 的正弦在 σ=8.9 纹素处误差 6.2/255，改布局后回落到 <1/255）。
    //
    // 钳制路径（ratio > taps 上限）：`tap_w = σ_max/taps` 且 span 相应缩短，
    // 与"经典 EWA 抬级"同效——长轴覆盖不足时宁可整体糊一点，也不留锯齿。
    let tap_w = s_min.max(s_max / taps as f32);
    let lod = log2_f32(tap_w.max(1e-8));
    let span = (s_max - tap_w).max(0.0);
    // tap 中心沿主轴均分 span（taps = 1 时无偏移）。
    let gap = if taps > 1 { span / (taps as f32 - 1.0) } else { 0.0 };
    let step_u = dir[0] * gap / w;
    let step_v = dir[1] * gap / h;
    let mut acc = [0.0f32; 4];
    for k in 0..taps {
        let t = k as f32 - (taps as f32 - 1.0) * 0.5;
        let u = [uv[0] + step_u * t, uv[1] + step_v * t];
        let c = trilinear(r, table.texels, u, lod);
        for ch in 0..4 {
            acc[ch] += c[ch];
        }
    }
    let inv = 1.0 / taps as f32;
    let mut rgba = [0.0f32; 4];
    for ch in 0..4 {
        rgba[ch] = acc[ch] * inv;
    }
    Some(SampleInfo { rgba, lod, taps, ratio })
}

/// 点采样（1 tap、lod 0、零足迹）：**参考图与诊断**用，产品路径不调它。
///
/// 与 [`sample`] 的关系：这是"没有滤波"的极限，故 64× 超采样参考图用它
/// 逐点积分（V2 的判据基准），而不是用被测的滤波路径自证。
pub fn sample_point(table: &TexTable<'_>, handle: u32, uv: [f32; 2]) -> Option<Texel> {
    let r = table.get(handle)?;
    Some(bilinear(r, table.texels, 0, uv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::build_rgba8;

    /// 1×1 白点纹理表（最小合法表）。
    fn one_px() -> (Vec<TextureRef>, Vec<Texel>) {
        let (r, t) = build_rgba8(1, 1, false, 0, &[255, 255, 255, 255]);
        (vec![r], t)
    }

    #[test]
    fn log2_matches_reference_values() {
        for (x, want) in [(1.0f32, 0.0f32), (2.0, 1.0), (4.0, 2.0), (0.5, -1.0), (1024.0, 10.0)] {
            assert!((log2_f32(x) - want).abs() < 1e-5, "log2({x}) = {}", log2_f32(x));
        }
        // 非 2 的幂：1.44 → 0.5261（V3 的残余盒宽，§2.3 的 1.44 就是它）。
        assert!((log2_f32(1.44) - 0.526_069).abs() < 1e-4);
        // 防御：0 / 负数 / NaN 不产生 -inf。
        assert_eq!(log2_f32(0.0), 0.0);
        assert_eq!(log2_f32(-1.0), 0.0);
        assert_eq!(log2_f32(f32::NAN), 0.0);
    }

    #[test]
    fn level_addressing_walks_flat_mip_chain() {
        let px: Vec<u8> = (0..8 * 4 * 4).map(|i| (i % 256) as u8).collect();
        let (r, _texels) = build_rgba8(8, 4, false, 0, &px);
        assert_eq!(level_dims(&r, 0), (8, 4));
        assert_eq!(level_dims(&r, 1), (4, 2));
        assert_eq!(level_dims(&r, 2), (2, 1));
        assert_eq!(level_dims(&r, 3), (1, 1));
        assert_eq!(level_base(&r, 0), 0);
        assert_eq!(level_base(&r, 1), 32);
        assert_eq!(level_base(&r, 2), 32 + 8);
        assert_eq!(level_base(&r, 3), 32 + 8 + 2);
        // offset 非 0 时代码同样成立（多张纹理共用一个扁平缓冲）。
        let (r2, t2) = build_rgba8(2, 2, false, 100, &[9u8; 16]);
        assert_eq!(level_base(&r2, 0), 100);
        assert_eq!(level_base(&r2, 1), 104);
        assert_eq!(t2.len(), 5);
    }

    /// **V1（采样器级）：双线性在纹素中心精确命中，逐位等于纹素值。**
    ///
    /// 4×4 贴图，16 个纹素中心（`(i+0.5)/4` 全是二进制精确小数）；
    /// 结果与纹素值**位型相等**，且 `lod == 0`、`taps == 1`。
    #[test]
    fn v1_bilinear_hits_texel_center_bitwise() {
        let mut px = vec![0u8; 4 * 4 * 4];
        for j in 0..4usize {
            for i in 0..4usize {
                let k = (j * 4 + i) * 4;
                px[k] = (17 * (i + 1) + j) as u8; // 各不相同的码值
                px[k + 1] = (31 * j + i) as u8;
                px[k + 2] = (i * 16 + j * 4) as u8;
                px[k + 3] = 255;
            }
        }
        let (r, texels) = build_rgba8(4, 4, false, 0, &px);
        let refs = vec![r];
        let table = TexTable::new(&refs, &texels);
        for j in 0..4usize {
            for i in 0..4usize {
                let u = (i as f32 + 0.5) / 4.0;
                let v = (j as f32 + 0.5) / 4.0;
                let info = sample_with(&table, 1, [u, v], Footprint::ZERO, SampleOpts::DEFAULT)
                    .expect("句柄 1 合法");
                let want = texels[j * 4 + i];
                for ch in 0..4 {
                    assert_eq!(
                        info.rgba[ch].to_bits(),
                        want[ch].to_bits(),
                        "纹素 ({i},{j}) 通道 {ch}: {} ≠ {}（有半像素偏移或混合）",
                        info.rgba[ch],
                        want[ch]
                    );
                }
                assert_eq!(info.lod, 0.0);
                assert_eq!(info.taps, 1);
            }
        }
    }

    /// V1 的补充：纹素中心两侧必须**真的**混到邻居（否则"精确命中"可能只是
    /// 因为整张贴图同色）。取中心 + 1/8 纹素，期望 7/8·t0 + 1/8·t1。
    #[test]
    fn v1_neighbour_texel_is_blended_with_expected_weight() {
        let mut px = vec![0u8; 2 * 2 * 4];
        px[0] = 0; // (0,0).r = 0
        px[4] = 255; // (1,0).r = 255
        let (r, texels) = build_rgba8(2, 2, false, 0, &px);
        let refs = vec![r];
        let table = TexTable::new(&refs, &texels);
        // u = 0.25 + 1/8 = 0.375 → x = 0.375*2-0.5 = 0.25 ⇒ 0.75·t0 + 0.25·t1
        let got = sample_point(&table, 1, [0.375, 0.25]).unwrap()[0];
        let want = 0.0 * 0.75 + 1.0 * 0.25;
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    /// 平铺：`uv = 1.5 + 0.25` 与 `uv = 0.25` 必须取到同一纹素（repeat）。
    #[test]
    fn repeat_wrap_maps_uv_rings_to_same_texel() {
        let mut px = vec![0u8; 2 * 2 * 4];
        px[0] = 255;
        let (r, texels) = build_rgba8(2, 2, false, 0, &px);
        let refs = vec![r];
        let table = TexTable::new(&refs, &texels);
        let a = sample_point(&table, 1, [0.25, 0.25]).unwrap();
        for k in [-3.0f32, -1.0, 1.0, 3.0, 17.0] {
            let b = sample_point(&table, 1, [0.25 + k, 0.25 - k]).unwrap();
            for ch in 0..4 {
                assert_eq!(a[ch].to_bits(), b[ch].to_bits(), "uv 平移整数圈后必须同值");
            }
        }
    }

    #[test]
    fn singular_axes_recover_known_ellipse() {
        // 正交两轴：3 沿 u、5 沿 v ⇒ σ_max = 5（沿 v），σ_min = 3。
        let f = Footprint::new([3.0, 0.0], [0.0, 5.0]);
        let (s_max, dir, s_min) = f.singular_axes();
        assert!((s_max - 5.0).abs() < 1e-5);
        assert!((s_min - 3.0).abs() < 1e-5);
        assert!(dir[1].abs() > 0.999, "主轴应沿 v，得到 {dir:?}");
        // 旋转 45°：仍是同一椭圆（旋转不改变奇异值）。
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let g = Footprint::new([3.0 * s, 3.0 * s], [-5.0 * s, 5.0 * s]);
        let (s_max2, _d2, s_min2) = g.singular_axes();
        assert!((s_max2 - 5.0).abs() < 1e-4, "σ_max {s_max2}");
        assert!((s_min2 - 3.0).abs() < 1e-4, "σ_min {s_min2}");
        // 退化：零足迹。
        let (m, _d, n) = Footprint::ZERO.singular_axes();
        assert_eq!((m, n), (0.0, 0.0));
    }

    #[test]
    fn sample_returns_none_for_absent_handle() {
        let (refs, texels) = one_px();
        let table = TexTable::new(&refs, &texels);
        assert!(sample(&table, 0, [0.5, 0.5], Footprint::ZERO).is_none(), "句柄 0 = 无贴图");
        assert!(sample(&table, 2, [0.5, 0.5], Footprint::ZERO).is_none(), "越界句柄");
        assert!(sample(&TexTable::EMPTY, 1, [0.5, 0.5], Footprint::ZERO).is_none());
        assert!(sample(&table, 1, [0.5, 0.5], Footprint::ZERO).is_some());
    }

    // -----------------------------------------------------------------
    // V3：各向异性收益（zone plate，ρ = 5.76）
    // -----------------------------------------------------------------

    /// 沿 `dir` 轴的光栅（逐纹素反相 = 0.5 cyc/texel，Nyquist）。
    fn grating(w: u32, h: u32, along_v: bool) -> (Vec<TextureRef>, Vec<Texel>) {
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let k = ((y * w + x) * 4) as usize;
                let phase = if along_v { y % 2 } else { x % 2 };
                let c = if phase == 0 { 255u8 } else { 0u8 };
                px[k] = c;
                px[k + 1] = c;
                px[k + 2] = c;
                px[k + 3] = 255;
            }
        }
        let (r, t) = build_rgba8(w, h, false, 0, &px);
        (vec![r], t)
    }

    /// 沿 u 拉长的掠射足迹：短轴 1 纹素、长轴 `rho` 纹素。
    fn grazing_footprint(w: f32, h: f32, rho: f32) -> Footprint {
        Footprint::new([rho / w, 0.0], [0.0, 1.0 / h])
    }

    /// 在给定足迹下沿一条扫描线测对比度 `(max−min)/(max+min)`。
    fn contrast(table: &TexTable<'_>, foot: Footprint, opts: SampleOpts, along_v: bool) -> f32 {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for k in 0..64 {
            let t = 0.03 + 0.90 * (k as f32) / 63.0;
            let uv = if along_v { [0.5, t] } else { [t, 0.5] };
            let c = sample_with(table, 1, uv, foot, opts).expect("句柄合法").rgba[0];
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if hi + lo > 0.0 {
            (hi - lo) / (hi + lo)
        } else {
            0.0
        }
    }

    /// **V3：各向异性收益（ρ = 5.76，f = 0.5 cyc/texel）。**
    ///
    /// # 对 §2.3 第 V3 行阈值的**推翻**（连同算术）
    ///
    /// 原文判据："4-tap 各向异性 ≥ 0.30，各向同性 ≤ 0.05"，依据是"残余盒宽
    /// ρ/4 = 1.44，`|sinc(π·0.72)| ≈ 0.34`"。**那个模型只对"只积分 1.44
    /// 纹素"的滤波器成立**，而任何覆盖整个 5.76 纹素足迹的滤波器
    /// （4 tap 平铺后就是"盒宽 5.76"）对 0.5 cyc/texel 的响应都是
    /// `|sinc(π·0.5·5.76)| = |sin(9.05)/9.05| ≈ 0.040`——与各向同性同值。
    ///
    /// 实测也如此（见下面的断言与 §15 表）：沿**长轴**的光栅，4-tap 各向异性
    /// 与各向同性都把对比度压到 ~0.04–0.06，**这是对的**（该方向本来就该被滤掉，
    /// 留着才是混叠）。各向异性的真实收益是**方向性**的：沿**短轴**的高频
    /// 不被过度模糊。故判据改为：
    ///
    /// | 光栅方向 | 4-tap 各向异性 | 各向同性（仅 mip） | 判据 |
    /// |---|---|---|---|
    /// | 短轴（v，φ 方向） | 高（实测见断言） | 低 | aniso ≥ 0.30 且 ≥ 5× iso |
    /// | 长轴（u，ρ 方向） | ~0.04 | ~0.04 | aniso ≤ 0.10 且 ≤ 2× iso（不得放过混叠） |
    #[test]
    fn v3_anisotropy_preserves_minor_axis_detail() {
        let (w, h) = (256u32, 64u32);
        let rho = 5.76f32;
        let foot = grazing_footprint(w as f32, h as f32, rho);

        // 短轴光栅：各向异性的收益所在。
        let (refs_v, tex_v) = grating(w, h, true);
        let tv = TexTable::new(&refs_v, &tex_v);
        let aniso_minor = contrast(&tv, foot, SampleOpts::DEFAULT, true);
        let iso_minor = contrast(&tv, foot, SampleOpts::ISOTROPIC, true);

        // 长轴光栅：两者都该滤掉（不得因为"少滤波"而放过混叠）。
        let (refs_u, tex_u) = grating(w, h, false);
        let tu = TexTable::new(&refs_u, &tex_u);
        let aniso_major = contrast(&tu, foot, SampleOpts::DEFAULT, false);
        let iso_major = contrast(&tu, foot, SampleOpts::ISOTROPIC, false);

        let info = sample_with(&tv, 1, [0.5, 0.5], foot, SampleOpts::DEFAULT).unwrap();
        assert_eq!(info.taps, 4, "ρ=5.76 ⇒ 4 tap");
        assert!((info.ratio - rho).abs() < 0.01, "比值 {}", info.ratio);
        assert!((info.lod - 0.526_069).abs() < 1e-3, "lod = log2(1.44)，得到 {}", info.lod);

        assert!(
            aniso_minor >= 0.30,
            "短轴对比度 {aniso_minor} < 0.30（各向同性 {iso_minor}）"
        );
        assert!(iso_minor <= 0.05, "各向同性短轴对比度 {iso_minor} > 0.05");
        assert!(
            aniso_minor >= 5.0 * iso_minor,
            "各向异性/各向同性 = {} ×（{aniso_minor} vs {iso_minor}），应 ≥ 5×",
            aniso_minor / iso_minor.max(1e-9)
        );
        assert!(
            aniso_major <= 0.10,
            "长轴对比度 {aniso_major} > 0.10：各向异性把该滤掉的混叠放过去了"
        );
        // 长轴方向的残余（各向同性是理想盒 ⇒ 恒 0；各向异性是 4 tap 平铺 ⇒
        // 有 1.44 盒的 sinc 尾巴）。判据只要求它不构成可见混叠（≤ 0.10）。
        assert!(iso_major <= 0.05, "各向同性长轴对比度 {iso_major} > 0.05");
        println!(
            "V3 ρ={rho}：4tap 短轴={aniso_minor:.4} 1tap 短轴={iso_minor:.4}（{:.1}×）；4tap 长轴={aniso_major:.4} 1tap 长轴={iso_major:.4}；taps={} lod={:.4}",
            aniso_minor / iso_minor.max(1e-9),
            info.taps,
            info.lod
        );
    }

    /// mip 链真的在起作用：同一足迹下，`ratio = 8`（长轴 8 纹素、短轴 1）
    /// 的 lod 必须 > 0 —— 这是"长轴被 tap 覆盖后仍要抬级"的钳制分支。
    #[test]
    fn aniso_clamp_raises_lod_when_ratio_exceeds_tap_limit() {
        let (refs, texels) = grating(64, 8, false);
        let table = TexTable::new(&refs, &texels);
        // σ_min = 1、σ_max = 32 ⇒ ratio 32 >> 4 tap ⇒ lod = log2(32/4) = 3。
        let foot = Footprint::new([32.0 / 64.0, 0.0], [0.0, 1.0 / 8.0]);
        let info = sample_with(&table, 1, [0.5, 0.5], foot, SampleOpts::DEFAULT).unwrap();
        assert_eq!(info.taps, 4);
        assert!((info.lod - 3.0).abs() < 1e-4, "lod = {}，期望 log2(32/4) = 3", info.lod);
    }
}
