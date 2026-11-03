//! A5 逐像素自适应采样 —— 选型报告 §6 定稿 **C1：两级固定预算**
//! （`N0 = 8` pilot → `n_i ∝ σ̂^0.5` → [`allocate_spp`] 归一）。
//!
//! # 一句话
//!
//! **只改"每个像素跑多少个样本"，不改"每个像素怎么由样本得到值"。** 于是有一条
//! 强不变式可以直接当回归测试（§6.2）：
//!
//! ```text
//! adaptive[p] == render_pixel(p, spp = n_p)[p]      （逐位）
//! ```
//!
//! 它不是"数值上很接近"：走到 [`mean_samples`] 的那条序列与 `render_pixel` 内部
//! 的序列**是同一串**（同样的 `SampleStream::new(ctx, pixel, s)`、同样的 `s = 0..n_p`
//! 顺序、同一个固定形状归约）。
//!
//! # 三条硬化要求（一条都不许打折，逐条对应报告 §6.1 / §6.2）
//!
//! 1. **样本索引必须连续**：pilot 用 `0..N0`，追加用 `N0..n_p`，**不跳、不重排**。
//!    Sobol 的前缀性质（任意前缀仍是低差异序列）只在连续索引下成立；跳采样会让
//!    实测收益退出 §6 给出的 2.05× 区间。
//! 2. **分支只允许落在整数上**：`σ̂` 用 f32 **位型截断**成桶（取尾数高 `k` 位，
//!    默认 `k = 8`），桶号是 `u32`；权重由**桶号**（而不是 `σ̂` 的浮点比较）决定。
//!    这样 CPU / GPU 只要 `σ̂` 的位型一致就必然同桶号 ⇒ 分配逐像素一致。
//!    **禁止**对 `σ̂` 做阈值比较（一个 ULP 翻转控制流，跨后端放大成整幅差异）。
//! 3. **存逐样本值，最后归约**：`prism_core::reduce` 是**固定形状树**
//!    （`reduction_shape(items.len())`），`mean(N0+k)` **不能**由 `mean(N0)` 折叠得到。
//!    故 pilot 的 N0 个样本值必须留着，与追加样本拼成长度 `n_p` 的一条序列再归约。
//!
//! # 与 `allocate_spp` 的关系：分层，不是替代
//!
//! 本模块产出"每像素想要多少"（权重），[`allocate_spp`]（`prism-postfx` 的
//! 归一化层）把总预算按权重切成**整数**（largest-remainder，tie-break 固定，
//! `Σ == total` 有 `debug_assert`）。**`allocate_spp` 一行没改**：它是预算的
//! 归一化层，改了会破既有的 `Σ == total` 保证。
//!
//! # 预算不变式（V2）
//!
//! `total ≥ N0 · 像素数` 时：先给每像素钉死 `N0`，余量 `total − N0·P` 再按权重
//! 分配 ⇒ `Σ n_p == total` 且 `n_p ≥ N0` **在构造上成立**（不靠事后检查）。
//!
//! # 内存（tile 内，tile 结束即释放）
//!
//! * **pilot 缓冲**：每个 tile 一段 `tile_px × N0 × 16 B`（64×64 tile、N0=8 ⇒ 512 KB），
//!   tile 精化完即 `drop`（`pilots[ti] = Vec::new()`），所以峰值随 tile 逐个退潮；
//!   *全部 tile 同时存活*时的上界 = `P × N0 × 16 B`（320×240、N0=8 ⇒ **9.8 MB**）。
//! * **精化暂存**：每 tile 一个可复用的 `Vec<[f32;4]>`，长度 = 该像素当前 `n_p`
//!   （单像素上界 `max n_p × 16 B`）。
//! * 归约后只留 `P × 16 B` 的成品 + `P × 4 B` 的 `n_p` + 两份 `P × 4 B` 的 `σ̂`
//!   （pilot 版供分配、最终版供收敛判据；两者语义不同，见 [`AdaptiveImage`]）。
//!
//! # 三条明确"未做"（不许读成"已做"）
//!
//! * **不做反馈迭代**（§6 的 C3 已被否：迭代到收敛的停止条件是浮点分支，跨后端
//!   放大；且收益不可控）。本模块只有 **pilot → 分配 → 定量追加** 两段。
//! * **不做跨帧/跨 tile 的全局状态**：分配只依赖本帧 pilot 的 `σ̂`。
//! * **不按误差提前终止**：`n_p` 一旦定下就整段跑完（提前终止会破坏"逐位等于
//!   `render_pixel(spp=n_p)`"这条不变式）。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::accum::mean_samples;
use prism_rt::camera::Camera;
use prism_rt::integrator::{trace_sample, RenderParams, Scene};

use crate::denoise::allocate_spp;

/// Rec. 709 亮度权重（**固定**口径，与 `displace.rs` 的通道口径同源）。
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// pilot 样本数默认值（§6 定稿 `N0 = 8`）。
pub const DEFAULT_N0: u32 = 8;

/// `σ̂` 分桶时保留的**尾数高位**数（默认 8 ⇒ 每个二进制阶 256 个桶，
/// 量化相对误差 ≤ 2^-8 = 0.39%，开方后 ≤ 0.2%）。
pub const DEFAULT_MANTISSA_BITS: u32 = 8;

/// tile 边长默认值（64×64）。
pub const DEFAULT_TILE: u32 = 64;

/// 自适应采样的参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveParams {
    /// 每像素 pilot 样本数（`0..n0`）。必须 ≥ 2（1 个样本的样本方差不定义）。
    pub n0: u32,
    /// **全图总预算**（`Σ n_p`）。必须 ≥ `n0 × 像素数`，且 ≤ 2²⁴
    /// （与 `allocate_spp` 的上界一致：f32 舍入会破坏 largest-remainder 的前提）。
    pub total: u32,
    /// `σ̂` 分桶保留的尾数高位数 `k ∈ [1, 23]`；桶号 = `σ̂.to_bits() >> (23 − k)`。
    pub mantissa_bits: u32,
    /// tile 边长（pilot 缓冲与精化暂存的作用域）。
    pub tile: u32,
}

impl AdaptiveParams {
    /// 以总预算构造默认参数（`N0 = 8`、`k = 8`、tile 64）。
    pub fn new(total: u32) -> Self {
        AdaptiveParams {
            n0: DEFAULT_N0,
            total,
            mantissa_bits: DEFAULT_MANTISSA_BITS,
            tile: DEFAULT_TILE,
        }
    }

    /// 分桶右移位数（`23 − k`）。
    #[inline]
    pub fn bucket_shift(&self) -> u32 {
        23 - self.mantissa_bits
    }
}

/// 自适应渲染的产物。
#[derive(Debug, Clone, Default)]
pub struct AdaptiveImage {
    /// 帧宽。
    pub width: u32,
    /// 帧高。
    pub height: u32,
    /// 成品像素（行主序 `width × height`，rgba）。
    pub pixels: Vec<[f32; 4]>,
    /// 逐像素实际样本数 `n_p`（行主序）。
    pub spp: Vec<u32>,
    /// 逐像素 `σ̂`（**pilot** 的 8 个样本，亮度口径，行主序）。
    ///
    /// 它是**分配**的依据（`w_p = σ̂^0.5`），不是收敛判据的依据 —— 8 个样本的
    /// 样本标准差相对误差约 27%，用它做"这个像素收敛了吗"会把 2% 上下的像素
    /// 判错（见 [`AdaptiveImage::sigma`] 的说明与 `tests/qa_adaptive.rs` 的 V4）。
    pub sigma_pilot: Vec<f32>,
    /// 逐像素 `σ̂`（**最终** `n_p` 个样本，亮度口径，行主序）。
    ///
    /// 收敛判据该用这一个：真实的"跑够了没有"是在每像素已有样本上现算
    /// `σ̂/√n_p`，手上从来不止 8 个样本。用 pilot 的 8 样本算等于丢掉
    /// `n_p − 8` 个样本的信息，`σ̂` 的相对误差会一直停在 27%。
    pub sigma: Vec<f32>,
    /// 实际追踪的样本总数（`== Σ spp`，**含 pilot**）。
    pub traced: u64,
}

impl AdaptiveImage {
    /// 逐像素样本数之和（`u64`，防溢出）。
    pub fn sample_count(&self) -> u64 {
        self.spp.iter().map(|&v| v as u64).sum()
    }
}

/// 单样本的亮度（Rec. 709，固定求和顺序 `R → G → B`）。
///
/// 口径**必须固定**：σ̂ 的数值、进而桶号、进而分配，全由它决定。换口径就是换
/// 一版分配（会破坏跨版本的可复现性）。
#[inline]
pub fn sample_luma(s: [f32; 4]) -> f32 {
    let mut v = s[0] * LUMA_R;
    v = v + s[1] * LUMA_G;
    v = v + s[2] * LUMA_B;
    v
}

/// 由逐样本值算 `σ̂`（**样本标准差**，分母 `n − 1`）。
///
/// 两遍扫描（先均值再离差平方和），顺序固定 ⇒ 同输入逐位可复现。
/// `n < 2` 或非有限 ⇒ `0.0`（确定性兜底，不产 NaN）。
pub fn sigma_of(samples: &[[f32; 4]]) -> f32 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let mut sum = 0.0f32;
    for s in samples {
        sum += sample_luma(*s);
    }
    let mean = sum / nf;
    let mut acc = 0.0f32;
    for s in samples {
        let d = sample_luma(*s) - mean;
        acc += d * d;
    }
    let var = acc / (nf - 1.0);
    // **整数**退化判定（不看 `var > 0.0`）：符号位 / 指数全 1 / 位型 0 都退回 0。
    let bits = var.to_bits();
    if bits & 0x8000_0000 != 0 || bits >= 0x7F80_0000 || bits == 0 {
        return 0.0;
    }
    var.sqrt()
}

/// `σ̂` → **整数桶号**：f32 位型右移 `23 − k` 位（取尾数高 `k` 位，含指数位）。
///
/// 这是 §6.1 的"分支只落在整数上"落地处：桶号是 `u32`，跨后端只要 `σ̂` 的位型
/// 一致就必然同桶号。**不对 `σ̂` 做任何浮点阈值比较。**
///
/// `σ̂ ≤ 0` / 非有限 ⇒ 桶 `0`（不是 NaN 桶：位型 `0x0000_0000`）。
///
/// **退化判定也走整数**（2026-11 复核时改的）：上一版写的是
/// `if !(sigma > 0.0) || !sigma.is_finite()` —— 那是**浮点比较**，正是本节
/// 要禁的东西。现在只看位型：
///
/// * 符号位 `0x8000_0000` 置位 ⇒ 负数（含 `-0.0`、`-∞`）⇒ 桶 0；
/// * 位型 `>= 0x7F80_0000` ⇒ 指数全 1（`±∞` / NaN，负号已被上一条拦下）⇒ 桶 0；
/// * `+0.0` 的位型就是 `0`，右移仍是 0 ⇒ 天然落在桶 0。
///
/// `+∞` 单靠 `!(sigma > 0.0)` 拦不住（它确实 `> 0`），而 `0x7F80_0000 >> 15`
/// 是一个巨大的非零桶 ⇒ 权重 `sqrt(∞) = ∞` 会让 `allocate_spp` 的权重断言炸掉。
/// 故显式按**位型有限性**兜底。
#[inline]
pub fn sigma_bucket(sigma: f32, mantissa_bits: u32) -> u32 {
    let bits = sigma.to_bits();
    if bits & 0x8000_0000 != 0 || bits >= 0x7F80_0000 {
        return 0;
    }
    bits >> (23 - mantissa_bits.clamp(1, 23))
}

/// 桶号 → 权重 `w = σ̂_q^0.5`，`σ̂_q` = 桶号左移回 f32 位型（**尾数低位清零**）。
///
/// 这里刻意**不是** `(bucket as f32).sqrt()`：那样会让权重跨二进制阶失真
/// （桶号每个阶跳 2^k，而 `σ̂` 只跳 2×）。左移回去得到的是"量化后的 σ̂"，
/// 于是 `w ∝ σ̂^0.5` 在全体量级上成立，相对误差只有 2^-k 量级 —— 而桶号仍是整数，
/// CPU/GPU 同桶号 ⇒ 同权重。
#[inline]
pub fn bucket_weight(bucket: u32, mantissa_bits: u32) -> f32 {
    if bucket == 0 {
        return 0.0;
    }
    let bits = bucket << (23 - mantissa_bits.clamp(1, 23));
    // 位型越界（指数全 1 或符号位）⇒ 0：这是**整数**判断。合法的 `σ̂` 桶号
    // 由 [`sigma_bucket`] 保证 `< 0x7F80_0000 >> shift`，越界只会来自调用方
    // 手搓的桶号，此时宁可退化成"不分配"也不要把 `∞` 权重喂给 `allocate_spp`。
    if bits & 0x8000_0000 != 0 || bits >= 0x7F80_0000 {
        return 0.0;
    }
    f32::from_bits(bits).sqrt()
}

/// 逐像素样本数分配（**纯函数**：`σ̂` → `n_p`，与渲染顺序、tile 划分无关）。
///
/// 三步：
/// 1. 每像素钉 `N0` 个（pilot 的样本要重放，不能丢）；
/// 2. 余量 `total − N0·P` 按 `w_p = σ̂_q^0.5` 交给 [`allocate_spp`] 归一成整数；
/// 3. `n_p = N0 + extra_p`。
///
/// 全零权重（所有像素 `σ̂ = 0`，例如纯黑场景）⇒ 权重退化为全 1（等分余量），
/// 这是**整数**判断（全部桶号 == 0），不是浮点分支。
///
/// # Panics
/// `total < N0·P`、`total > 2²⁴`、`n0 < 2`、`mantissa_bits ∉ [1, 23]`。
pub fn allocate_per_pixel(sigma: &[f32], n0: u32, total: u32, mantissa_bits: u32) -> Vec<u32> {
    assert!(n0 >= 2, "N0 必须 ≥ 2（样本方差不定义于 1 个样本）");
    assert!(
        (1..=23).contains(&mantissa_bits),
        "mantissa_bits 必须在 [1, 23]"
    );
    assert!(total <= 1 << 24, "总预算过大：f32 舍入会破坏精确分配");
    let npix = sigma.len();
    let base = (npix as u64) * (n0 as u64);
    assert!(
        (total as u64) >= base,
        "总预算 {total} 不够每像素 N0={n0}（需要 {base}）"
    );
    if npix == 0 {
        return Vec::new();
    }
    let k = mantissa_bits;
    let all_zero = sigma.iter().all(|s| sigma_bucket(*s, k) == 0);
    let mut w: Vec<f32> = Vec::with_capacity(npix);
    for s in sigma {
        let b = sigma_bucket(*s, k);
        w.push(if all_zero { 1.0 } else { bucket_weight(b, k) });
    }
    let extra = allocate_spp(&w, total - base as u32);
    let mut out = extra;
    for v in out.iter_mut() {
        *v += n0;
    }
    debug_assert_eq!(
        out.iter().map(|&v| v as u64).sum::<u64>(),
        total as u64,
        "Σ n_p 必须等于总预算"
    );
    debug_assert!(out.iter().all(|&v| v >= n0), "n_p 必须 ≥ N0");
    out
}

/// **自适应渲染入口**：pilot → 分配 → 定量追加 → 固定形状归约。
///
/// 每个像素的样本序列是 `s = 0, 1, …, n_p − 1`（**连续、不跳、不重排**），
/// 经 [`mean_samples`]（`prism_core::reduce` 固定形状树）归约 ⇒ 与
/// `render_pixel(p, spp = n_p)` 逐位相同。
///
/// # Panics
/// 参数不合法（见 [`allocate_per_pixel`]）或 `resolution` 为 0。
pub fn render_adaptive(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    ap: &AdaptiveParams,
) -> AdaptiveImage {
    let (w, h) = resolution;
    assert!(w > 0 && h > 0, "分辨率不得为 0");
    let npix = (w as usize) * (h as usize);
    let n0 = ap.n0;
    let base = (npix as u64) * (n0 as u64);
    assert!(
        (ap.total as u64) >= base,
        "总预算 {} 不够每像素 N0={n0}（需要 {base}）",
        ap.total
    );
    assert!(ap.total <= 1 << 24, "总预算过大：f32 舍入会破坏精确分配");

    let tile = ap.tile.max(1);
    let tiles_x = (w + tile - 1) / tile;
    let tiles_y = (h + tile - 1) / tile;

    // ---- 第一段：逐 tile 跑 pilot，只留 σ̂；样本值暂存在**该 tile 自己的缓冲**里 ----
    let mut sigma = vec![0.0f32; npix];
    let mut pilots: Vec<Vec<[f32; 4]>> = Vec::with_capacity((tiles_x * tiles_y) as usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let tw = tile.min(w - x0);
            let th = tile.min(h - y0);
            let mut buf: Vec<[f32; 4]> =
                Vec::with_capacity((tw as usize) * (th as usize) * n0 as usize);
            for ly in 0..th {
                for lx in 0..tw {
                    let (gx, gy) = (x0 + lx, y0 + ly);
                    for s in 0..n0 {
                        buf.push(trace_sample(
                            scene,
                            cam,
                            ctx,
                            resolution,
                            (gx, gy),
                            s,
                            params,
                        ));
                    }
                    let off = ((ly as usize) * (tw as usize) + (lx as usize)) * (n0 as usize);
                    let sl = &buf[off..off + (n0 as usize)];
                    sigma[(gy as usize) * (w as usize) + (gx as usize)] = sigma_of(sl);
                }
            }
            pilots.push(buf);
        }
    }

    // ---- 第二段：全局分配（与 tile 划分无关的纯函数） ----
    let spp = allocate_per_pixel(&sigma, n0, ap.total, ap.mantissa_bits);

    // ---- 第三段：逐 tile 定量追加 + 归约；tile 结束即释放该 tile 的 pilot 缓冲 ----
    let mut image = AdaptiveImage {
        width: w,
        height: h,
        pixels: vec![[0.0; 4]; npix],
        spp: spp.clone(),
        sigma_pilot: sigma,
        sigma: vec![0.0f32; npix],
        traced: 0,
    };
    let mut scratch: Vec<[f32; 4]> = Vec::new();
    let mut ti = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let tw = tile.min(w - x0);
            let th = tile.min(h - y0);
            let pilot = std::mem::take(&mut pilots[ti]);
            for ly in 0..th {
                for lx in 0..tw {
                    let (gx, gy) = (x0 + lx, y0 + ly);
                    let p = (ly as usize) * (tw as usize) + (lx as usize);
                    let gi = (gy as usize) * (w as usize) + (gx as usize);
                    let n = spp[gi];
                    scratch.clear();
                    scratch.reserve(n as usize);
                    scratch.extend_from_slice(&pilot[p * n0 as usize..(p + 1) * n0 as usize]);
                    // 连续索引：`N0 .. n_p`（不跳、不重排）。
                    for s in n0..n {
                        scratch.push(trace_sample(
                            scene,
                            cam,
                            ctx,
                            resolution,
                            (gx, gy),
                            s,
                            params,
                        ));
                    }
                    image.pixels[gi] = mean_samples(&scratch);
                    // 收敛判据用的 σ̂：**全部 n_p 个**样本现算（不是 pilot 的 8 个）。
                    image.sigma[gi] = sigma_of(&scratch);
                    image.traced += n as u64;
                }
            }
            drop(pilot); // tile 结束即释放（逐样本值不留跨 tile 状态）
            ti += 1;
        }
    }
    debug_assert_eq!(
        image.traced, ap.total as u64,
        "实际追踪样本数必须等于总预算"
    );
    image
}

/// 均匀对照：每像素 `spp` 个样本（与 [`render_adaptive`] 走**同一个**逐样本入口）。
///
/// 存在的意义只有一个：V3/V5 的"自适应 vs 均匀"必须在**同一份采样器/同一份归约**
/// 上比，否则比的是两套实现，不是两套分配。
pub fn render_uniform(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    spp: u32,
) -> Vec<[f32; 4]> {
    let (w, h) = resolution;
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let mut scratch: Vec<[f32; 4]> = Vec::with_capacity(spp as usize);
    for y in 0..h {
        for x in 0..w {
            scratch.clear();
            for s in 0..spp {
                scratch.push(trace_sample(scene, cam, ctx, resolution, (x, y), s, params));
            }
            out.push(mean_samples(&scratch));
        }
    }
    out
}

/// 两个像素序列的**逐位**比较：返回位差个数（0 ⇒ 逐位一致）。
pub fn bitwise_differences(a: &[[f32; 4]], b: &[[f32; 4]]) -> usize {
    assert_eq!(a.len(), b.len(), "比较双方长度必须一致");
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            if x[c].to_bits() != y[c].to_bits() {
                n += 1;
                break;
            }
        }
    }
    n
}

/// PSNR（dB）：`10·log10( peak² / MSE )`，峰值取 `peak`，在 **RGB 三通道**上平均。
///
/// 只用 `log10`（诊断量，不进任何判据的浮点分支）。MSE 为 0 ⇒ `f32::INFINITY`。
pub fn psnr(a: &[[f32; 4]], b: &[[f32; 4]], peak: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return f32::INFINITY;
    }
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        for c in 0..3 {
            let d = x[c] as f64 - y[c] as f64;
            acc += d * d;
        }
    }
    let mse = acc / (a.len() as f64 * 3.0);
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    let p = peak as f64;
    // 中间量保持 f64（避免 8 位量级下 pow 的相消），仅在末端收窄回 f32。
    (10.0 * (p * p / mse).log10()) as f32
}

/// 逐像素亮度（用于误差/相关性诊断）。
pub fn luma_of(p: [f32; 4]) -> f32 {
    sample_luma(p)
}

/// 亮度向量（诊断用）。
pub fn lumas(px: &[[f32; 4]]) -> Vec<f32> {
    px.iter().map(|p| luma_of(*p)).collect()
}

/// `Vec3` 版本亮度（测试便利）。
#[inline]
pub fn luma_vec3(v: Vec3) -> f32 {
    v.x * LUMA_R + v.y * LUMA_G + v.z * LUMA_B
}

// ---------------------------------------------------------------------------
// 单元测试：只测**纯函数**（σ̂ 分桶、权重、分配、PSNR）。
//
// 需要真渲染的那几条（V1 估计量不变、V3 效率、V4 相关性、V5 降噪后收益、
// V6 确定性）在 `tests/qa_adaptive.rs`：它们要一个真场景与 4096 spp 参考图，
// 不适合塞进 `cargo test --lib` 的快回路。
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 桶号必须是**整数**且由位型唯一决定：同 `σ̂` 位型 ⇒ 同桶号；
    /// `σ̂` 单调增 ⇒ 桶号单调不减（这是"分支只落在整数上"的可测形式）。
    #[test]
    fn sigma_bucket_is_integer_and_monotone() {
        let k = DEFAULT_MANTISSA_BITS;
        // 遍历 [2^-20, 2^20] 上的 4096 个几何级数点（含同阶内的不同尾数）。
        let mut prev = 0u32;
        let mut prev_s = 0.0f32;
        for i in 0..4096u32 {
            let s = (2.0f32).powf(-20.0 + 40.0 * (i as f32) / 4095.0);
            let b = sigma_bucket(s, k);
            assert!(b >= prev, "桶号必须随 σ̂ 单调不减（{prev_s} -> {s}）");
            assert_eq!(b, sigma_bucket(s, k), "同输入必须同桶号");
            prev = b;
            prev_s = s;
        }
        assert!(prev > 0, "上界处的桶号不应退化到 0");
    }

    /// 桶号是**位型截断**而不是四舍五入：只差低位尾数的两个 `σ̂` 必须同桶。
    #[test]
    fn sigma_bucket_truncates_low_mantissa_bits() {
        let k = DEFAULT_MANTISSA_BITS;
        let shift = 23 - k;
        let base = 0.5f32; // 位型 0x3F00_0000，尾数全 0
        let b0 = sigma_bucket(base, k);
        // 打开最低 shift 位 ⇒ 桶号不变（截断而非 round-to-nearest）。
        let low = f32::from_bits(base.to_bits() | ((1 << shift) - 1));
        assert_eq!(sigma_bucket(low, k), b0, "低位尾数必须被截断掉");
        // 打开最高位尾数（1 << shift）⇒ 桶号 +1。
        let hi = f32::from_bits(base.to_bits() | (1 << shift));
        assert_eq!(sigma_bucket(hi, k), b0 + 1, "进一位尾数必须换桶");
    }

    /// `σ̂ ≤ 0` / NaN / `±∞` ⇒ 桶 0（确定性兜底，不产 NaN 桶也不产 ∞ 权重）。
    #[test]
    fn sigma_bucket_degenerate_inputs() {
        for k in [1u32, 8, 23] {
            assert_eq!(sigma_bucket(0.0, k), 0);
            assert_eq!(sigma_bucket(-0.0, k), 0);
            assert_eq!(sigma_bucket(-1.5, k), 0);
            assert_eq!(sigma_bucket(f32::NAN, k), 0);
            assert_eq!(sigma_bucket(f32::INFINITY, k), 0);
            assert_eq!(sigma_bucket(f32::NEG_INFINITY, k), 0);
            // 桶 0 ⇒ 权重 0（不是 NaN）。
            assert_eq!(bucket_weight(0, k), 0.0);
            assert!(bucket_weight(0, k).is_finite());
        }
    }

    /// 权重必须是**量化后 σ̂ 的开方**（左移回位型），不是桶号开方。
    /// 判据：相对误差 ≤ 2^-k（k=8 ⇒ ≤ 0.39%）。
    #[test]
    fn bucket_weight_tracks_quantized_sigma() {
        for k in [8u32, 12] {
            let tol = (2.0f32).powi(-(k as i32)) * 1.01;
            for i in 0..512u32 {
                let s = (2.0f32).powf(-16.0 + 32.0 * (i as f32) / 511.0);
                let w = bucket_weight(sigma_bucket(s, k), k);
                let want = s.sqrt();
                let rel = (w - want).abs() / want;
                assert!(rel <= tol, "k={k} σ̂={s} 权重 {w} 偏离 √σ̂={want}（{rel}）");
            }
        }
    }

    /// 权重随桶号单调不减（跨二进制阶也成立：左移回的是合法 f32 位型）。
    #[test]
    fn bucket_weight_is_monotone() {
        let k = DEFAULT_MANTISSA_BITS;
        let mut prev = 0.0f32;
        for b in 0..4096u32 {
            // 只取正规数区（指数位不越界）。
            let bits = b << (23 - k);
            if f32::from_bits(bits).is_finite() && bits < 0x7F80_0000 {
                let w = bucket_weight(b, k);
                assert!(w >= prev, "桶 {b} 权重 {w} 小于前一桶 {prev}");
                prev = w;
            }
        }
        assert!(prev > 0.0);
    }

    /// `sigma_of` 对已知样本给出**样本标准差**（分母 `n − 1`）。
    #[test]
    fn sigma_of_matches_analytic() {
        let px = |l: f32| {
            // 把给定亮度编码进 R 通道：`sample_luma` 里 R 的系数是 LUMA_R，
            // 故先除回去 ⇒ 该样本的亮度恰为 `l`（相对误差 1 ULP 量级）。
            [l / LUMA_R, 0.0, 0.0, 1.0]
        };
        // 两样本 {0, 1}：样本标准差 = |Δ| / √2。
        let s2 = sigma_of(&[px(0.0), px(1.0)]);
        assert!(
            (s2 - std::f32::consts::FRAC_1_SQRT_2).abs() <= 1e-6,
            "两样本 {s2}"
        );
        // 四样本 {0,0,1,1}：样本标准差 = √(1/3)。
        let s4 = sigma_of(&[px(0.0), px(0.0), px(1.0), px(1.0)]);
        let want = (1.0f32 / 3.0).sqrt();
        assert!((s4 - want).abs() <= 1e-6, "四样本 {s4} vs {want}");
        // 常数序列 ⇒ 0（两个样本也 0）。
        assert_eq!(sigma_of(&[px(0.25), px(0.25), px(0.25)]), 0.0);
    }

    /// `sigma_of` 的退化输入（`n < 2`、非有限值）⇒ 0，绝不产 NaN。
    #[test]
    fn sigma_of_degenerate_inputs() {
        assert_eq!(sigma_of(&[]), 0.0);
        assert_eq!(sigma_of(&[[0.1, 0.2, 0.3, 1.0]]), 0.0);
        let nan = sigma_of(&[[f32::NAN, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]);
        assert_eq!(nan, 0.0, "NaN 输入必须兜底成 0");
        let inf = sigma_of(&[[f32::INFINITY, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]);
        assert_eq!(inf, 0.0, "∞ 输入必须兜底成 0");
    }

    /// 分配：`Σ n_p == total` 且 `n_p ≥ N0`（V2 的纯函数版本，构造性）。
    #[test]
    fn allocate_pins_n0_and_conserves_budget() {
        let npix = 97usize;
        // 造一个跨 6 个数量级的 σ 场（含 0）。
        let sigma: Vec<f32> = (0..npix)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    (i as f32 * 0.37).sin().abs()
                }
            })
            .collect();
        for avg in [8u32, 9, 16, 23, 64, 255] {
            let total = (npix as u32) * avg;
            let n = allocate_per_pixel(&sigma, 8, total, DEFAULT_MANTISSA_BITS);
            assert_eq!(n.len(), npix);
            assert_eq!(n.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
            assert!(n.iter().all(|&v| v >= 8), "n_p 必须 ≥ N0");
        }
    }

    /// 全零 σ̂（纯黑 / 常数场景）⇒ 余量等分，且是整数判断（不是浮点分支）。
    #[test]
    fn allocate_all_zero_is_equal_split() {
        let npix = 40usize;
        let sigma = vec![0.0f32; npix];
        let total = 40 * 11 + 7; // 故意不整除
        let n = allocate_per_pixel(&sigma, 8, total, DEFAULT_MANTISSA_BITS);
        assert_eq!(n.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
        let lo = *n.iter().min().unwrap();
        let hi = *n.iter().max().unwrap();
        assert!(hi - lo <= 1, "全零权重必须等分（差 {lo}..{hi}）");
    }

    /// 高 σ̂ 像素必须拿到更多样本（`n_p ∝ σ̂^0.5` 的方向性）；
    /// 且 σ̂ = 0 的像素恰好拿 N0（一分不多）。
    #[test]
    fn allocate_gives_more_to_noisier_pixels() {
        let mut sigma = vec![0.0f32; 64];
        sigma[10] = 1.0; // 最吵
        sigma[20] = 0.0625; // 次吵
        let total = 64 * 16;
        let n = allocate_per_pixel(&sigma, 8, total, DEFAULT_MANTISSA_BITS);
        assert_eq!(n[0], 8, "σ̂=0 的像素必须恰好拿 N0");
        assert!(
            n[10] > n[20],
            "更吵的像素必须拿更多（{} vs {}）",
            n[10],
            n[20]
        );
        assert!(n[20] > 8, "次吵的像素也必须有追加样本");
        assert_eq!(n.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
    }

    /// 分桶 `k` 改变 ⇒ 分配可能变，但**同一 k 下同输入必须逐位同分配**（纯函数）。
    #[test]
    fn allocate_is_a_pure_function_of_sigma() {
        let sigma: Vec<f32> = (0..50)
            .map(|i| (i as f32 * 0.11).cos().abs() * 0.3)
            .collect();
        let a = allocate_per_pixel(&sigma, 8, 50 * 20, DEFAULT_MANTISSA_BITS);
        let b = allocate_per_pixel(&sigma, 8, 50 * 20, DEFAULT_MANTISSA_BITS);
        assert_eq!(a, b);
        // σ̂ 只动最低尾数位（被截断）⇒ 分配不变（分桶的吸收性）。
        let mut sigma2 = sigma.clone();
        sigma2[3] = f32::from_bits(sigma[3].to_bits() | 0x7);
        let c = allocate_per_pixel(&sigma2, 8, 50 * 20, DEFAULT_MANTISSA_BITS);
        assert_eq!(a, c, "被截断的低位不得影响分配");
    }

    /// 参数边界：`N0 < 2` / `k ∉ [1,23]` / 预算不足 ⇒ panic（防呆，不静默）。
    #[test]
    #[should_panic(expected = "N0 必须 ≥ 2")]
    fn allocate_rejects_tiny_n0() {
        let _ = allocate_per_pixel(&[1.0, 1.0], 1, 100, DEFAULT_MANTISSA_BITS);
    }

    /// 预算不足 `N0·P` ⇒ panic（不是静默少采）。
    #[test]
    #[should_panic(expected = "不够每像素")]
    fn allocate_rejects_insufficient_budget() {
        let _ = allocate_per_pixel(&[1.0; 16], 8, 100, DEFAULT_MANTISSA_BITS);
    }

    /// 亮度口径 = Rec. 709（固定和序 `R → G → B`）。
    #[test]
    fn luma_is_rec709_fixed_order() {
        assert_eq!(sample_luma([1.0, 0.0, 0.0, 1.0]), LUMA_R);
        assert_eq!(sample_luma([0.0, 1.0, 0.0, 1.0]), LUMA_G);
        assert_eq!(sample_luma([0.0, 0.0, 1.0, 1.0]), LUMA_B);
        // 灰 1.0 ⇒ 权重和（f32 下应恰为 1.0）。
        let g = sample_luma([1.0, 1.0, 1.0, 1.0]);
        assert!((g - 1.0).abs() <= 1e-7, "灰必须映射到 1.0，实得 {g}");
        // α 通道不参与。
        assert_eq!(
            sample_luma([0.3, 0.4, 0.5, 1.0]),
            sample_luma([0.3, 0.4, 0.5, 0.0])
        );
    }

    /// PSNR：同图 ⇒ ∞；常数偏 δ ⇒ `10·log10(1/δ²)`（peak = 1）。
    #[test]
    fn psnr_known_values() {
        let a = vec![[0.5f32, 0.25, 0.75, 1.0]; 4];
        assert!(psnr(&a, &a, 1.0).is_infinite(), "同图 PSNR 必须为 ∞");
        let d = 0.01f32;
        let b: Vec<[f32; 4]> = a
            .iter()
            .map(|p| [p[0] + d, p[1] + d, p[2] + d, 1.0])
            .collect();
        let got = psnr(&a, &b, 1.0);
        let want = 10.0 * (1.0f32 / (d * d)).log10();
        assert!((got - want).abs() <= 1e-3, "PSNR {got} vs {want}");
        assert!(psnr(&[], &[], 1.0).is_infinite(), "空图不产 NaN");
    }

    /// 参数默认值与派生量一致（`bucket_shift = 23 − k`）。
    #[test]
    fn params_defaults_and_derived() {
        let p = AdaptiveParams::new(1000);
        assert_eq!(p.n0, DEFAULT_N0);
        assert_eq!(p.n0, 8);
        assert_eq!(p.mantissa_bits, DEFAULT_MANTISSA_BITS);
        assert_eq!(p.bucket_shift(), 23 - DEFAULT_MANTISSA_BITS);
        assert_eq!(p.tile, DEFAULT_TILE);
        assert_eq!(p.total, 1000);
    }

    /// 逐位比较器：只差 1 ULP 也必须被计数（否则 V1 是假绿）。
    #[test]
    fn bitwise_comparator_catches_one_ulp() {
        let a = [[0.5f32, 0.5, 0.5, 1.0]];
        let mut b = a;
        b[0][0] = f32::from_bits(a[0][0].to_bits() + 1);
        assert_eq!(bitwise_differences(&a, &a), 0);
        assert_eq!(bitwise_differences(&a, &b), 1);
        // -0.0 vs +0.0 是**不同位型**（本比较器按位，不按值）。
        assert_eq!(bitwise_differences(&[[0.0; 4]], &[[-0.0; 4]]), 1);
    }
}
