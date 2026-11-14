//! HDR 环境光照 / IBL（A2 / W2）：等距柱状 + 立体角测度 2D CDF + 亮斑列表 + MIS。
//!
//! 定稿口径：`docs/v2.5-W0-算法选型报告.md` §3（C1）。本模块是 **CPU 权威**实现，
//! WGSL wavefront（`rt_env.wgsl`）按同一份语义镜像。
//!
//! # 一、为什么极点在立体角测度下解析消失
//!
//! 等距柱状图的行 `j` 覆盖极角带 `[θ_j, θ_{j+1}]`，其立体角是**精确球冠**：
//!
//! ```text
//! Ω_j = 2π(cos θ_j − cos θ_{j+1})            （j = 0 与 j = H−1 同式，无需特判）
//! ```
//!
//! 行内 `φ` 均匀 ⇒ 列立体角 `Ω_ij = Ω_j / W`（**不是** `sinθ·Δθ·Δφ` 的梯形近似）。
//! 采样概率取 `p_ij ∝ L_ij·Ω_ij`、行内位置均匀 ⇒
//!
//! ```text
//! pdf_ω(dir) = p_ij / Ω_ij = L_ij / Σ(L·Ω)
//! ```
//!
//! 分子分母的 `sinθ` **精确相消**：`pdf_ω` 与 `θ` 无关，极点不发散。
//! 常数环境（`L ≡ 1`）下退化为 `pdf_ω ≡ 1/4π`（V5 判据：`pdf_ω·4π` 偏差 ≤ 1e-3）。
//!
//! # 二、为什么前缀和是整数（D-29 / §1.2 求和顺序红线）
//!
//! `L·Ω` 先按 `2^40 / max` 量化成 **u64** 整数，前缀和全程整数累加，
//! 最后用 `u128` 精确归一化到 u32 存储。整数加法**可交换** ⇒ 求和顺序无关，
//! 比"固定求值顺序"强一档；SIMD 重排、分块归约都不会改变任何一个字节。
//! 运行期只做整数二分 + 两次 f32 除法。
//!
//! # 三、为什么亮斑要单独列表（§3 定稿的"两级拆分"）
//!
//! 太阳只占几个 texel ⇒ 单一 CDF 里它的权重比天空大 5~6 个数量级，
//! 残差那一段的 u32 分辨率被吃干净（尾部 texel 被量化成 0 宽 = 永不采样）。
//! 两级拆分把分布写成**混合**：
//!
//! ```text
//! p(ω) = P_s·(w_k/W_S) + (1−P_s)·(w_k/W_R)      k ∈ S / k ∈ R
//!      = w_k / (T·Ω_k)                            （两式恒等，代数上完全一致）
//! ```
//!
//! 于是拆分**只改采样路径，不改目标密度**：`pdf_ω` 仍是 `L_ij/Σ(L·Ω)`。
//! 好处是残差 CDF 用它自己的总量归一化 ⇒ 全 u32 分辨率留给"非太阳"的那部分。
//!
//! # 四、不做预过滤（§3 定稿的核心优势）
//!
//! CDF 采样只做**离散 texel 反演 + 双线性取值**，没有预过滤 cubemap、
//! 没有 split-sum 的 `F·L` 分离、没有 LOD 三线性混合。代价是每个样本一次
//! 二分（十几步整数比较），收益是无近似（V1 白炉 1e-2、V3 无偏 0.5%）。
//!
//! # 五、确定性
//!
//! 只用 `± × ÷ √` 与 [`prism_math::det_trig`] 的确定性三角件；无 `libm`、
//! 无全局 RNG、无哈希遍历顺序（`Vec` 行主序）。同一份纹素两次构建得到
//! **逐字节相同**的 CDF（`build_is_bitwise_deterministic` 钉住）。

use std::f32::consts::{FRAC_1_PI, PI, TAU};

use prism_math::{det_trig, Vec3};

use crate::bsdf::luminance;
use crate::texsample::{level_base, level_dims, TexTable, Texel};
use crate::texture::TextureRef;

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/// 量化尺度 `2^40`（f64 精确可表示）：每个 texel 的权重按 `2^40 / max` 量化成 u64。
///
/// 40 位 = 1.1e12 级，对应"行内最暗 texel 相对最亮 texel"的可表示比 1e12；
/// 行内 `W ≤ 8192` ⇒ 行内累计 `≤ 2^53`，u64 有 11 位余量（不会被 `cum_u32`
/// 的 `u128` 归一化吃掉精度）。
const QUANT_SCALE: f64 = 1_099_511_627_776.0;

/// 亮斑判据：单 texel 的立体角权重超过**全环境平均权重**多少倍算"亮斑"。
///
/// 64 = 太阳/天空对比 1e5 时，太阳 texel 超均值 ~1e4 倍（太阳只占 1e-5 立体角），
/// 天空 texel 约 0.2 倍；取 64 是这两档之间的一刀，两侧都留 1.5 个数量级。
/// 阈值只影响**采样路径**，不影响目标密度（见模块文档第三节）。
pub const SPOT_FACTOR: f64 = 64.0;

/// 立体角总量（球面）。`Ω_total` 是 `pdf_ω` 的归一化常数。
pub const OMEGA_SPHERE: f32 = 4.0 * PI;

// ---------------------------------------------------------------------------
// 等距柱状映射（CPU 与 WGSL 必须逐位同式）
// ---------------------------------------------------------------------------

/// 方向 → 等距柱状 `(u, v)`。
///
/// * `u = atan2(z, x)/(2π) + 0.5`，归一到 `[0, 1)`（`u` 方向**平铺**）；
/// * `v = acos(y)/π`，`v = 0` 在 +Y（北极）、`v = 1` 在 −Y（南极）。
///
/// `atan2` / `acos` 走 [`prism_math::det_trig`]（只含 `± × ÷ √`）——
/// 落平台 `libm` 会让同一方向在不同机器上取到不同 texel，
/// 而 CDF 的 pdf 必须与采样严格同源（V5）。
#[inline]
/// 把 CDF 的全部状态打成一个 u32 blob（布局 = `rt_env.wgsl` 文件头那张表）。
///
/// **这是 blob 布局的唯一权威**：WGSL 侧只按偏移读，偏移由本函数写入。
/// （原为 `tests/qa_w2_env_gpu_parity.rs` 的测试内实现；R-10 设备采数也要打
/// 同一布局的 blob，故正名为公开 API——布局权威不该住在测试文件里。）
pub fn blob_words(cdf: &EnvCdf, pixels: &[[f32; 4]]) -> Vec<u32> {
    let (w, h) = cdf.dims();
    let cond = cdf.cond_words().to_vec();
    let marg = cdf.marg_res_words().to_vec();
    let res_cond = cdf.res_cond_words().to_vec();
    let row_off = cdf.res_row_off_words().to_vec();
    let spot_prefix = cdf.spot_prefix_words().to_vec();
    let spot_texels = cdf.spot_texel_words();
    let stats = cdf.stats();
    assert_eq!(cond.len(), ((w + 1) * h) as usize);
    assert_eq!(marg.len(), (h + 1) as usize);
    assert_eq!(row_off.len(), h as usize);
    assert_eq!(spot_prefix.len(), stats.spots as usize + 1);
    assert_eq!(spot_texels.len(), stats.spots as usize);

    let mut blob = vec![0u32; 24];
    let push = |blob: &mut Vec<u32>, data: &[u32]| -> u32 {
        let off = blob.len() as u32;
        blob.extend_from_slice(data);
        off
    };
    blob[0] = w;
    blob[1] = h;
    blob[2] = stats.spots;
    blob[3] = stats.p_spot.to_bits();
    blob[4] = cdf.omega_total().to_bits();
    blob[5] = if cdf.total_weight() > 0.0 { 1 } else { 0 };
    // 残差权重是否非零：`p_spot < 1` 且总能量非零（与 CPU sample 的分支同源）。
    blob[6] = if cdf.total_weight() > 0.0 && stats.p_spot < 1.0 {
        1
    } else {
        0
    };
    blob[7] = cond.len() as u32;
    blob[8] = marg.len() as u32;
    blob[9] = res_cond.len() as u32;
    blob[10] = row_off.len() as u32;
    blob[11] = spot_prefix.len() as u32;
    blob[12] = spot_texels.len() as u32;
    blob[13] = (4 * w * h) as u32;

    blob[14] = 0;
    let off_cond = push(&mut blob, &cond);
    let off_marg = push(&mut blob, &marg);
    let off_res = push(&mut blob, &res_cond);
    let off_row = push(&mut blob, &row_off);
    let off_sp = push(&mut blob, &spot_prefix);
    let off_st = push(&mut blob, &spot_texels);
    let off_tex = blob.len() as u32;
    for p in pixels {
        blob.push(p[0].to_bits());
        blob.push(p[1].to_bits());
        blob.push(p[2].to_bits());
        blob.push(p[3].to_bits());
    }
    blob[14] = off_cond;
    blob[15] = off_marg;
    blob[16] = off_res;
    blob[17] = off_row;
    blob[18] = off_sp;
    blob[19] = off_st;
    blob[20] = off_tex;
    blob
}

pub fn dir_to_uv(d: Vec3) -> (f32, f32) {
    // u = atan2(z, x)/(2π) + 0.5；`FRAC_1_PI * 0.5` 是 1/(2π)，与 WGSL 侧同式。
    let mut u = det_trig::atan2(d.z, d.x) * (FRAC_1_PI * 0.5) + 0.5;
    u -= u.floor();
    if !(u < 1.0) {
        u = 0.0; // NaN / 恰为 1.0 的防御（atan2 非有限时返回 0，此处兜底）
    }
    let v = det_trig::acos(d.y.clamp(-1.0, 1.0)) * FRAC_1_PI;
    (u, v.clamp(0.0, 1.0))
}

/// 等距柱状 `(u, v)` → 方向（[`dir_to_uv`] 的逆）。
///
/// `v` 决定极角：`θ = v·π`；`u` 决定方位角：`φ = 2π(u − 0.5)`。
#[inline]
pub fn uv_to_dir(u: f32, v: f32) -> Vec3 {
    let theta = v.clamp(0.0, 1.0) * PI;
    let (c, s) = det_trig::cos_sin(theta);
    let phi = TAU * (u - 0.5);
    let (cp, sp) = det_trig::cos_sin(phi);
    Vec3::new(s * cp, c, s * sp)
}

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

/// 一个亮斑：单个 texel 的线性下标 + 其立体角权重（f64，构建期用）。
#[derive(Debug, Clone, Copy)]
struct Spot {
    /// texel 线性下标（行主序：`j * W + i`）。
    texel: u32,
    /// `L·Ω`（构建期权重；只在建表时用，运行期用前缀差）。
    weight: f64,
}

/// 一次环境采样的结果（辐亮度由调用方按 `wi` 再取一次双线性）。
#[derive(Debug, Clone, Copy)]
pub struct EnvSample {
    /// 采样方向（单位向量）。
    pub wi: Vec3,
    /// 该方向的立体角 pdf（`pdf_ω`）。
    pub pdf_w: f32,
}

/// 环境图统计（诊断 / 报告用；不参与像素值）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvStats {
    /// 宽度（texel）。
    pub w: u32,
    /// 高度（texel）。
    pub h: u32,
    /// 亮斑 texel 数。
    pub spots: u32,
    /// 亮斑占总能量的比例 `P_s`。
    pub p_spot: f32,
    /// 条件 CDF 字节数（`(W+1)·H·4`）。
    pub cond_bytes: usize,
    /// 边际 CDF 字节数（`H·4`）。
    pub marg_bytes: usize,
    /// 残差条件 CDF 字节数（只有含亮斑的行）。
    pub res_cond_bytes: usize,
    /// 亮斑列表字节数（每项 8 B：texel + 权重前缀）。
    pub spot_bytes: usize,
}

/// 等距柱状环境图的立体角 2D CDF（不可变；构建一次、只读）。
///
/// 见模块文档。所有字段私有：外部只能经 [`EnvCdf::radiance`] /
/// [`EnvCdf::pdf_w`] / [`EnvCdf::sample`] 使用，避免第二套寻址口径。
#[derive(Debug, Clone)]
pub struct EnvCdf {
    /// 环境图句柄（[`crate::light::Light::Environment`] 的 `map`）。
    handle: u32,
    /// 宽度（texel）。
    w: u32,
    /// 高度（texel）。
    h: u32,
    /// 条件 CDF：`(W+1)·H` 个 u32，第 `j` 行归一化到 `u32::MAX`。
    cond: Vec<u32>,
    /// 边际 CDF：`H+1` 个 u32（跨行归一化）。
    marg: Vec<u32>,
    /// 残差边际 CDF：`H+1` 个 u32（剔除亮斑后的行权重）。
    marg_res: Vec<u32>,
    /// 含亮斑行的残差条件 CDF（压缩存放，每行 `W+1` 项）。
    res_cond: Vec<u32>,
    /// 行 → [`EnvCdf::res_cond`] 的起始偏移；`u32::MAX` = 该行无亮斑。
    res_row_off: Vec<u32>,
    /// 亮斑列表（按 texel 升序）。
    spots: Vec<Spot>,
    /// 亮斑前缀 CDF：`spots.len()+1` 个 u32（归一化到 `u32::MAX`）。
    spot_prefix: Vec<u32>,
    /// 亮斑总能量占比 `P_s`。
    p_spot: f32,
    /// 每行立体角 `Ω_j`。
    omega_row: Vec<f32>,
    /// `Ω_total = Σ Ω_j`（≈ 4π；用实际求和值，保证 pdf 归一化与表一致）。
    omega_total: f32,
    /// `Σ(L·Ω)`（f64；诊断 + 空环境判定）。
    total_weight: f64,
    /// 残差总能量（f64；判定残差分支是否可用）。
    residual_weight: f64,
}

impl EnvCdf {
    /// 由环境图句柄 + 纹理表构建 CDF。
    ///
    /// 返回 `None` 当且仅当：句柄为 0 / 越界、尺寸为 0、或纹素区间越界
    /// （场景装配错误；[`crate::integrator::Scene::validate_layout`] 会在
    /// 构造期先拦一遍，本函数是防御性二次判定）。
    ///
    /// **全黑环境**可以成功构建（`total_weight == 0`）：此时
    /// [`EnvCdf::sample`] 返回 `None`、[`EnvCdf::pdf_w`] 返回 0、
    /// [`EnvCdf::radiance`] 返回黑——语义等价于"没有环境灯"，而不是非法输入。
    pub fn build(handle: u32, table: TexTable<'_>) -> Option<EnvCdf> {
        let r: TextureRef = *table.get(handle)?;
        let w = r.w.max(1) as usize;
        let h = r.h.max(1) as usize;
        let base = level_base(&r, 0);
        let n = w.checked_mul(h)?;
        let end = base.checked_add(n)?;
        if end > table.texels.len() {
            return None;
        }
        let lv0: &[Texel] = &table.texels[base..end];

        // ---- 行立体角（精确球冠）----
        let mut omega_row = Vec::with_capacity(h);
        let mut omega_total = 0.0f64;
        for j in 0..h {
            let t0 = PI * (j as f32) / (h as f32);
            let t1 = PI * ((j + 1) as f32) / (h as f32);
            let o = (2.0 * PI) * (det_trig::cos(t0) - det_trig::cos(t1));
            let o = if o.is_finite() && o > 0.0 { o } else { 0.0 };
            omega_row.push(o);
            omega_total += o as f64;
        }
        let omega_total_f32 = omega_total as f32;

        // ---- 亮度表 + 行权重（f64，行主序固定顺序）----
        let mut lum = vec![0.0f32; n];
        let mut row_weight = vec![0.0f64; h];
        for j in 0..h {
            let mut s = 0.0f64;
            for i in 0..w {
                let t = lv0[j * w + i];
                let l = luminance(Vec3::new(t[0], t[1], t[2]));
                let l = if l.is_finite() && l > 0.0 { l } else { 0.0 };
                lum[j * w + i] = l;
                s += l as f64;
            }
            // 行权重 = Σ_k L_ik·Ω_ik，其中 Ω_ik = Ω_row/W（列立体角）。
            row_weight[j] = (s / w as f64) * omega_row[j] as f64;
        }
        let total_weight: f64 = row_weight.iter().sum();

        // ---- 亮斑检测（阈值 = SPOT_FACTOR × **中位数**权重）----
        //
        // 为什么不用均值：均值会被亮斑自己拉高。设 N 个 texel 里 k 个太阳、
        // 亮度 S、其余 1，则 `均值 ≈ kS/N`，而单个太阳 texel 的权重
        // `≈ S·Ω_total/N`；判据 `S·Ω_total/N > 64·kS/N` 化成 `Ω_total > 64k`
        // ——**与 S 无关**且 4π < 64 ⇒ 无论太阳多亮都判不出来。
        // 中位数对少数离群值免疫（太阳占 texel 数 < 50% 时恒等于天空权重），
        // 于是判据退化为"比天空亮 64 倍"，正是要抓的东西。
        let mut spots: Vec<Spot> = Vec::new();
        if total_weight > 0.0 && omega_total > 0.0 {
            let median_w = median_texel_weight(&lum, &omega_row, w, h);
            let thr = SPOT_FACTOR * median_w;
            for j in 0..h {
                let ow = omega_row[j] as f64 / w as f64;
                for i in 0..w {
                    let wk = lum[j * w + i] as f64 * ow;
                    if wk > thr {
                        spots.push(Spot { texel: (j * w + i) as u32, weight: wk });
                    }
                }
            }
        }
        let spot_weight: f64 = spots.iter().map(|s| s.weight).sum();
        let p_spot = if total_weight > 0.0 {
            (spot_weight / total_weight) as f32
        } else {
            0.0
        };
        let residual_weight = total_weight - spot_weight;

        // ---- 条件 CDF（逐行归一化到 u32::MAX）----
        let mut cond = vec![0u32; (w + 1) * h];
        for j in 0..h {
            let row = &lum[j * w..(j + 1) * w];
            let q = quantize_f32(row);
            let prefix = cum_u32(&q);
            cond[j * (w + 1)..j * (w + 1) + (w + 1)].copy_from_slice(&prefix);
        }

        // ---- 边际 CDF（跨行归一化）----
        let marg = cum_u32(&quantize(&row_weight));

        // ---- 残差：行权重、含亮斑行的条件 CDF ----
        let mut row_weight_res = row_weight.clone();
        let mut spot_rows: Vec<bool> = vec![false; h];
        for s in &spots {
            let j = s.texel as usize / w;
            spot_rows[j] = true;
            row_weight_res[j] -= s.weight;
            if row_weight_res[j] < 0.0 {
                row_weight_res[j] = 0.0;
            }
        }
        let marg_res = cum_u32(&quantize(&row_weight_res));
        let mut res_cond: Vec<u32> = Vec::new();
        let mut res_row_off: Vec<u32> = vec![u32::MAX; h];
        for j in 0..h {
            if !spot_rows[j] {
                continue;
            }
            // 该行去掉亮斑后的亮度序列（亮斑 texel 权重清零）。
            let mut row: Vec<f32> = lum[j * w..(j + 1) * w].to_vec();
            for s in &spots {
                let k = s.texel as usize;
                if k / w == j {
                    row[k % w] = 0.0;
                }
            }
            let q = quantize_f32(&row);
            let prefix = cum_u32(&q);
            res_row_off[j] = res_cond.len() as u32;
            res_cond.extend_from_slice(&prefix);
        }

        // ---- 亮斑前缀（归一化到 u32::MAX）----
        let spot_w: Vec<f64> = spots.iter().map(|s| s.weight).collect();
        let spot_prefix = cum_u32(&quantize(&spot_w));

        let cdf = EnvCdf {
            handle,
            w: w as u32,
            h: h as u32,
            cond,
            marg,
            marg_res,
            res_cond,
            res_row_off,
            spots,
            spot_prefix,
            p_spot,
            omega_row,
            omega_total: omega_total_f32,
            total_weight,
            residual_weight,
        };
        // 构造即自检：CDF 必须单调不减且末端为 u32::MAX（空行除外）。
        debug_assert!(cdf.marg.windows(2).all(|p| p[0] <= p[1]), "边际 CDF 非单调");
        debug_assert!(
            cdf.cond.chunks(w + 1).all(|row| row.windows(2).all(|p| p[0] <= p[1])),
            "条件 CDF 非单调"
        );
        Some(cdf)
    }

    /// 环境图句柄。
    #[inline]
    pub fn handle(&self) -> u32 {
        self.handle
    }

    /// 宽度 / 高度（texel）。
    #[inline]
    pub fn dims(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// `Ω_total`（`pdf_ω` 的归一化常数；V5 用 `pdf_ω·Ω_total` 判 1）。
    #[inline]
    pub fn omega_total(&self) -> f32 {
        self.omega_total
    }

    /// 能量总量 `Σ(L·Ω)`（f64；全黑环境为 0）。
    #[inline]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// 统计（诊断）。
    pub fn stats(&self) -> EnvStats {
        EnvStats {
            w: self.w,
            h: self.h,
            spots: self.spots.len() as u32,
            p_spot: self.p_spot,
            cond_bytes: self.cond.len() * 4,
            marg_bytes: self.marg.len() * 4,
            res_cond_bytes: self.res_cond.len() * 4,
            spot_bytes: self.spots.len() * 8,
        }
    }

    /// 条件 CDF 的原始字节（GPU 上传用；`(W+1)·H` 个 LE u32）。
    pub fn cond_words(&self) -> &[u32] {
        &self.cond
    }

    /// 边际 CDF 的原始字（`H+1` 个 u32）。
    pub fn marg_words(&self) -> &[u32] {
        &self.marg
    }

    /// 残差边际 CDF 的原始字（`H+1` 个 u32）。
    pub fn marg_res_words(&self) -> &[u32] {
        &self.marg_res
    }

    /// 亮斑列表的原始字：`[texel, weight_prefix, texel, weight_prefix, …]`。
    pub fn spot_words(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.spots.len() * 2);
        for (k, s) in self.spots.iter().enumerate() {
            out.push(s.texel);
            out.push(self.spot_prefix[k + 1]);
        }
        out
    }

    /// 残差条件 CDF 的原始字（压缩行；只有含亮斑的行占一段）。
    ///
    /// GPU 侧上传用：与 [`Self::cond_words`] 同口径（`u32` 定点，
    /// 行内 `W+1` 个）。`res_row_off[j] == u32::MAX` 时该行读主表。
    pub fn res_cond_words(&self) -> &[u32] {
        &self.res_cond
    }

    /// 残差行偏移表（`H` 个 u32；`u32::MAX` = 该行无亮斑 ⇒ 读 [`Self::cond_words`]）。
    pub fn res_row_off_words(&self) -> &[u32] {
        &self.res_row_off
    }

    /// 亮斑 texel 线性下标列表（`j·W + i`，按 texel 升序；与 `spot_prefix` 同序）。
    pub fn spot_texel_words(&self) -> Vec<u32> {
        self.spots.iter().map(|s| s.texel).collect()
    }

    /// 亮斑权重前缀（`spot_count+1` 个 u32；首项 0、末项 = 亮斑总权重）。
    pub fn spot_prefix_words(&self) -> &[u32] {
        &self.spot_prefix
    }

    /// 该方向落在**亮斑 texel** 上时返回其下标。
    #[inline]
    fn spot_index_of(&self, texel: u32) -> Option<usize> {
        self.spots
            .binary_search_by_key(&texel, |s| s.texel)
            .ok()
    }

    /// 扁平 texel 下标的亮斑归属（`None` = 走残差支路）。
    ///
    /// **对拍/诊断用**：WGSL 侧 `env_spot_index` 判的就是同一件事，
    /// V6 用它区分"pdf 公式漂移"与"纹素边界两侧分类不同"。
    #[inline]
    pub fn spot_index_of_texel(&self, texel: u32) -> Option<usize> {
        self.spot_index_of(texel)
    }

    /// 第 `j` 行的立体角（`Ω_j`）。
    #[inline]
    fn row_omega(&self, j: usize) -> f32 {
        self.omega_row.get(j).copied().unwrap_or(0.0)
    }

    /// 单个 texel 的立体角（`Ω_j / W`）。
    #[inline]
    fn texel_omega(&self, j: usize) -> f32 {
        let w = self.w.max(1) as f32;
        self.row_omega(j) / w
    }

    /// 方向 → texel `(i, j)`（与 [`dir_to_uv`] 同式；越界钳制）。
    fn texel_of(&self, dir: Vec3) -> (usize, usize) {
        let (u, v) = dir_to_uv(dir);
        let w = self.w.max(1) as usize;
        let h = self.h.max(1) as usize;
        let i = ((u * (w as f32)) as usize).min(w - 1);
        let j = ((v * (h as f32)) as usize).min(h - 1);
        (i, j)
    }

    /// 环境辐亮度（**双线性、不做预过滤**；`u` 平铺、`v` 钳制）。
    ///
    /// 纹素中心在 `(i+0.5)/W`：方向恰落在纹素中心时返回该纹素原值（逐位），
    /// 这是 V2「峰值辐亮度 == 亮斑值」的机械保证。
    pub fn radiance(&self, table: TexTable<'_>, dir: Vec3) -> Vec3 {
        let Some(r) = table.get(self.handle) else {
            return Vec3::splat(0.0);
        };
        let (u, v) = dir_to_uv(dir);
        let (w, h) = level_dims(&r, 0);
        if w == 0 || h == 0 {
            return Vec3::splat(0.0);
        }
        let base = level_base(&r, 0);
        if base + w * h > table.texels.len() {
            return Vec3::splat(0.0);
        }
        let tex = &table.texels[base..base + w * h];
        let x = u * (w as f32) - 0.5;
        let y = v * (h as f32) - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        // u 平铺（跨缝取邻居）；v 钳制（极点不是环绕）。
        let i0 = wrap_i(x0 as i64, w as i64) as usize;
        let i1 = wrap_i(x0 as i64 + 1, w as i64) as usize;
        let j0 = (y0 as i64).clamp(0, h as i64 - 1) as usize;
        let j1 = (y0 as i64 + 1).clamp(0, h as i64 - 1) as usize;
        let t00 = tex[j0 * w + i0];
        let t10 = tex[j0 * w + i1];
        let t01 = tex[j1 * w + i0];
        let t11 = tex[j1 * w + i1];
        let mut c = [0.0f32; 3];
        for ch in 0..3 {
            // 固定结合序（先 u 后 v），跨后端同式。
            let a = t00[ch] + (t10[ch] - t00[ch]) * fx;
            let b = t01[ch] + (t11[ch] - t01[ch]) * fx;
            c[ch] = a + (b - a) * fy;
        }
        Vec3::new(c[0], c[1], c[2])
    }

    /// 方向的立体角 pdf（**与采样严格同源**：读的是同一份量化表）。
    ///
    /// `pdf_ω = p(texel) / Ω_texel`，其中 `p` 由两级混合给出（见模块文档第三节）。
    /// 落在量化成 0 宽的 texel 上时返回 0（该 texel 永不被 CDF 采到；
    /// 此时 BSDF 策略的 MIS 权重为 1，仍然无偏）。
    pub fn pdf_w(&self, dir: Vec3) -> f32 {
        if !(self.total_weight > 0.0) {
            return 0.0;
        }
        let (i, j) = self.texel_of(dir);
        let omega = self.texel_omega(j);
        if !(omega > 0.0) {
            return 0.0;
        }
        let texel = (j * self.w.max(1) as usize + i) as u32;
        let w = self.w.max(1) as usize;
        if let Some(k) = self.spot_index_of(texel) {
            let den = self.spot_prefix[self.spot_prefix.len() - 1];
            if den == 0 {
                return 0.0;
            }
            let p = self.spot_prefix[k + 1] - self.spot_prefix[k];
            return self.p_spot * (p as f32 / den as f32) / omega;
        }
        if !(self.residual_weight > 0.0) {
            return 0.0;
        }
        // 残差行概率。
        let p_row = if self.marg_res[j + 1] > self.marg_res[j] {
            (self.marg_res[j + 1] - self.marg_res[j]) as f32 / u32::MAX as f32
        } else {
            return 0.0;
        };
        // 残差列概率（含亮斑的行读压缩表，其余读主表）。
        let row = if self.res_row_off[j] != u32::MAX {
            let off = self.res_row_off[j] as usize;
            &self.res_cond[off..off + w + 1]
        } else {
            &self.cond[j * (w + 1)..(j + 1) * (w + 1)]
        };
        let lo = row[i];
        let hi = row[i + 1];
        if hi <= lo {
            return 0.0;
        }
        let p_col = (hi - lo) as f32 / u32::MAX as f32;
        (1.0 - self.p_spot) * p_row * p_col / omega
    }

    /// 采样一个方向（两级混合：亮斑列表 / 残差 2D CDF）。
    ///
    /// `u1` 承担"混合判决 + 边际反演"，`u2` 承担"行内反演 / 亮斑内方位角"。
    /// 两个输入均匀数被**完整**用掉（反演的分数部分同样均匀），
    /// 不会出现"维度抽了不用"的偏置。
    ///
    /// 返回 `None` 当且仅当环境全黑（`Σ(L·Ω) == 0`）——那等价于没有环境灯。
    pub fn sample(&self, u1: f32, u2: f32) -> Option<EnvSample> {
        if !(self.total_weight > 0.0) {
            return None;
        }
        let u1 = clamp_unit(u1);
        let u2 = clamp_unit(u2);
        let w = self.w.max(1) as usize;
        let h = self.h.max(1) as usize;
        let has_spots = !self.spots.is_empty() && self.p_spot > 0.0;
        let has_res = self.residual_weight > 0.0;
        let choose_spot = has_spots && (!has_res || u1 < self.p_spot);

        if choose_spot {
            // ---- 亮斑分支：按能量选 texel，内部均匀（立体角测度）----
            let us = if self.p_spot >= 1.0 { u1 } else { u1 / self.p_spot };
            let (k, frac) = pick_u32(&self.spot_prefix, us);
            let texel = self.spots[k].texel as usize;
            let (i, j) = (texel % w, texel / w);
            let dir = self.texel_dir(j, i, frac, u2);
            let den = self.spot_prefix[self.spot_prefix.len() - 1];
            let omega = self.texel_omega(j);
            if den == 0 || !(omega > 0.0) {
                return None;
            }
            let p = self.spot_prefix[k + 1] - self.spot_prefix[k];
            let pdf = self.p_spot * (p as f32 / den as f32) / omega;
            return Some(EnvSample { wi: dir, pdf_w: pdf });
        }

        // ---- 残差分支：跨行边际 → 行内条件 ----
        let ur = if self.p_spot >= 1.0 {
            u1
        } else {
            (u1 - self.p_spot) / (1.0 - self.p_spot)
        };
        let (j, fv) = pick_u32(&self.marg_res, ur);
        let (j, fv) = (j.min(h - 1), fv);
        let row = if self.res_row_off[j] != u32::MAX {
            let off = self.res_row_off[j] as usize;
            &self.res_cond[off..off + w + 1]
        } else {
            &self.cond[j * (w + 1)..(j + 1) * (w + 1)]
        };
        let (i, fu) = pick_u32(row, u2);
        let (i, fu) = (i.min(w - 1), fu);
        let dir = self.texel_dir(j, i, fv, fu);
        let omega = self.texel_omega(j);
        if !(omega > 0.0) {
            return None;
        }
        let p_row = (self.marg_res[j + 1] - self.marg_res[j]) as f32 / u32::MAX as f32;
        let lo = row[i];
        let hi = row[i + 1];
        if hi <= lo {
            return None;
        }
        let p_col = (hi - lo) as f32 / u32::MAX as f32;
        let pdf = (1.0 - self.p_spot) * p_row * p_col / omega;
        Some(EnvSample { wi: dir, pdf_w: pdf })
    }

    /// 行 `j`、列 `i` 内按 `(fv, fu)` 取一个方向（**立体角测度内均匀**）。
    ///
    /// * 方位角 `φ` 均匀：`dω ∝ dφ`，故 `φ` 直接线性插值；
    /// * 极角按 `sinθ` 加权：`cosθ` 在 `[cosθ_{j+1}, cosθ_j]` 上均匀，
    ///   即 `cosθ = cosθ_j + fv·(cosθ_{j+1} − cosθ_j)`，再取 `sinθ = √(1−cos²θ)`。
    ///
    /// 这两步让"texel 内均匀"是**真均匀**（而不是 uv 空间的均匀）——
    /// 否则 `pdf_ω = p/Ω` 就不成立，V5 会立刻红。
    fn texel_dir(&self, j: usize, i: usize, fv: f32, fu: f32) -> Vec3 {
        let w = self.w.max(1) as f32;
        let h = self.h.max(1) as f32;
        let c0 = det_trig::cos(PI * (j as f32) / h);
        let c1 = det_trig::cos(PI * ((j + 1) as f32) / h);
        let c = (c0 + (c1 - c0) * fv.clamp(0.0, 1.0)).clamp(-1.0, 1.0);
        let s = (1.0 - c * c).max(0.0).sqrt();
        let u = (i as f32 + fu.clamp(0.0, 1.0)) / w;
        let phi = TAU * (u - 0.5);
        let (cp, sp) = det_trig::cos_sin(phi);
        Vec3::new(s * cp, c, s * sp)
    }
}

// ---------------------------------------------------------------------------
// 整数 CDF 构建件
// ---------------------------------------------------------------------------

/// 把一组非负 f64 权重按 `2^QUANT_BITS / max` 量化成 u64（整数加法可交换）。
fn median_texel_weight(lum: &[f32], omega_row: &[f32], w: usize, h: usize) -> f64 {
    let n = w * h;
    if n == 0 || h == 0 {
        return 0.0;
    }
    // 固定步长子采样（≤ 4096 项）：**确定性**（同一份输入恒取同一批下标），
    // 且中位数对这种规则采样足够稳（亮斑是空间连通的，不会恰好只落在步长外）。
    let stride = (n / 4096).max(1);
    let mut sample: Vec<f64> = Vec::with_capacity(n / stride + 1);
    let mut idx = 0usize;
    while idx < n {
        let j = idx / w;
        let ow = omega_row.get(j).copied().unwrap_or(0.0) as f64 / w as f64;
        sample.push(lum[idx] as f64 * ow);
        idx += stride;
    }
    if sample.is_empty() {
        return 0.0;
    }
    sample.sort_by(|a, b| a.total_cmp(b));
    sample[sample.len() / 2]
}

fn quantize_f32(vals: &[f32]) -> Vec<u64> {
    let wide: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    quantize(&wide)
}

/// [`quantize`] 的 f64 版本（行权重 / 亮斑权重用）。
fn quantize(vals: &[f64]) -> Vec<u64> {
    let mut max = 0.0f64;
    for &v in vals {
        if v.is_finite() && v > max {
            max = v;
        }
    }
    if !(max > 0.0) {
        return vec![0u64; vals.len()];
    }
    let scale = QUANT_SCALE / max;
    let mut out = Vec::with_capacity(vals.len());
    for &v in vals {
        let q = if v.is_finite() && v > 0.0 { v * scale } else { 0.0 };
        // 截断（不是四舍五入）：`as u64` 对 NaN 给 0、对超界饱和，
        // 两个行为都在规范里，跨平台一致。
        out.push(q as u64);
    }
    out
}

/// 整数前缀和 → 归一化 u32（长度 `n+1`，首 0 尾 `u32::MAX`；总量为 0 时全 0）。
///
/// 归一化用 `u128` 精确乘除：`cum ≤ 2^53`、`× (2^32−1)` 只到 2^85，
/// u128 装得下 ⇒ 无溢出、无精度损失、与求和顺序无关。
fn cum_u32(q: &[u64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(q.len() + 1);
    out.push(0u32);
    let total: u64 = q.iter().sum();
    if total == 0 {
        out.extend(std::iter::repeat(0u32).take(q.len()));
        return out;
    }
    let mut cum = 0u64;
    let max = u32::MAX as u128;
    for &v in q {
        cum += v;
        let norm = (cum as u128 * max) / total as u128;
        out.push(norm as u32);
    }
    // 末项严格为 u32::MAX（上面的除法在 cum == total 时给出 max）。
    if let Some(last) = out.last_mut() {
        *last = u32::MAX;
    }
    out
}

/// 在归一化 u32 前缀表里反演：返回 `(下标, 区间内分数)`。
///
/// 二分找**第一个** `prefix[k+1] > target` 的 `k`——零宽区间（量化后坍缩的
/// 尾部 texel）自然被跳过。分数部分 `(target − prefix[k]) / (prefix[k+1] − prefix[k])`
/// 在 `[0,1)` 上均匀（逆 CDF 性质），直接用作 texel 内的抖动。
fn pick_u32(prefix: &[u32], u: f32) -> (usize, f32) {
    let n = prefix.len().saturating_sub(1);
    if n == 0 {
        return (0, 0.0);
    }
    let u = clamp_unit(u);
    // f32 → u32：`u32::MAX as f32` 向上舍入到 2^32，`as u32` 饱和到 MAX，
    // 故 u = 1−ε 时 target 落在最后一个非空区间里（不会越界）。
    let target = (u * (u32::MAX as f32)) as u32;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if prefix[mid + 1] > target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let k = lo.min(n - 1);
    let a = prefix[k];
    let b = prefix[k + 1];
    let frac = if b > a {
        ((target.saturating_sub(a)) as f32 / (b - a) as f32).clamp(0.0, 0.999_999_94)
    } else {
        0.0
    };
    (k, frac)
}

/// `[0, 1)` 钳制（含 NaN 防御：NaN → 0）。
#[inline]
fn clamp_unit(u: f32) -> f32 {
    if u.is_finite() {
        u.clamp(0.0, 0.999_999_94)
    } else {
        0.0
    }
}

/// 平铺取模（与采样器同式）。
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

// ---------------------------------------------------------------------------
// 测试：映射 / 立体角测度 / 整数 CDF / 两级拆分的中性 / 确定性
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::{TextureRef, TEX_FMT_RGBA16F, SRGB_LINEAR};

    /// 构造一张只有 level 0 的环境图（`offset = 0`；CDF 只读 level 0）。
    fn env_table(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> (Vec<TextureRef>, Vec<Texel>) {
        let mut tex = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                tex.push(f(i, j));
            }
        }
        let r = TextureRef::new(w, h, 1, 0, TEX_FMT_RGBA16F, SRGB_LINEAR);
        (vec![r], tex)
    }

    fn cdf_of(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> (EnvCdf, Vec<TextureRef>, Vec<Texel>) {
        let (refs, texels) = env_table(w, h, f);
        let table = TexTable::new(&refs, &texels);
        let cdf = EnvCdf::build(1, table).expect("合法环境图必须可构建");
        (cdf, refs, texels)
    }

    #[test]
    fn dir_uv_roundtrip() {
        // 方向 → uv → 方向：往返误差 ≤ 1e-5（det 三角件 ~1e-7/步）。
        let mut worst = 0.0f32;
        for j in 1..31u32 {
            for i in 0..47u32 {
                let u = (i as f32 + 0.31) / 47.0;
                let v = (j as f32 + 0.63) / 32.0;
                let d = uv_to_dir(u, v);
                let (u2, v2) = dir_to_uv(d);
                let du = (u2 - u).abs().min(1.0 - (u2 - u).abs()); // u 周期
                worst = worst.max(du).max((v2 - v).abs());
            }
        }
        assert!(worst < 1e-5, "uv 往返最差误差 {worst} 超过 1e-5");
    }

    #[test]
    fn row_solid_angles_are_exact_caps() {
        // 首末行必须是精确球冠 2π(1−cosΔθ)，其余行 2π(cosθ_j − cosθ_{j+1})；
        // 全表求和 = 4π（§3.1 第 1 条的"极点相消"前提）。
        for &h in &[4u32, 16, 128, 1024] {
            let (cdf, _, _) = cdf_of(2, h, |_, _| [1.0, 1.0, 1.0, 1.0]);
            let dt = PI / h as f32;
            let cap = 2.0 * PI * (1.0 - det_trig::cos(dt));
            let first = cdf.row_omega(0);
            let last = cdf.row_omega(h as usize - 1);
            assert!(
                (first - cap).abs() / cap < 1e-6,
                "h={h} 首行立体角 {first} vs 球冠 {cap}"
            );
            assert!(
                (last - cap).abs() / cap < 1e-6,
                "h={h} 末行立体角 {last} vs 球冠 {cap}"
            );
            let sum: f32 = (0..h as usize).map(|j| cdf.row_omega(j)).sum();
            assert!(
                (sum - OMEGA_SPHERE).abs() / OMEGA_SPHERE < 1e-6,
                "h={h} 立体角求和 {sum} vs 4π"
            );
        }
    }

    #[test]
    fn constant_env_pdf_is_uniform_over_every_texel() {
        // V5 的单元级版本：常数环境 L=1 时 `pdf_ω ≡ 1/4π`，
        // **逐个 texel 中心**检查（全 W×H 行）。
        let (w, h) = (64u32, 32u32);
        let (cdf, refs, texels) = cdf_of(w, h, |_, _| [1.0, 1.0, 1.0, 1.0]);
        let table = TexTable::new(&refs, &texels);
        let mut worst = 0.0f32;
        for j in 0..h {
            for i in 0..w {
                let d = uv_to_dir((i as f32 + 0.5) / w as f32, (j as f32 + 0.5) / h as f32);
                let p = cdf.pdf_w(d);
                assert!(p > 0.0, "texel({i},{j}) 的 pdf 为 0");
                worst = worst.max((p * cdf.omega_total() - 1.0).abs());
                // 辐亮度也必须精确为 1（双线性在纹素中心是恒等）。
                let l = cdf.radiance(table, d);
                assert_eq!(l.x.to_bits(), 1.0f32.to_bits(), "texel({i},{j}) 辐亮度非 1");
            }
        }
        assert!(worst <= 1e-3, "pdf·Ω_total 最大偏差 {worst} 超过 1e-3");
    }

    #[test]
    fn pdf_equals_radiance_over_total_without_and_with_spots() {
        // 两级拆分**只改采样路径、不改目标密度**：
        // 无论有没有亮斑，`pdf_ω(texel) == L/Σ(L·Ω)`。
        for &(spot, label) in &[(false, "无亮斑"), (true, "有亮斑")] {
            let (w, h) = (48u32, 24u32);
            let (cdf, refs, texels) = cdf_of(w, h, move |i, j| {
                // 太阳：3 个 texel 极亮；其余为渐变天空。
                if spot && (5..8).contains(&i) && j == 4 {
                    [8.0e3, 7.6e3, 6.0e3, 1.0]
                } else {
                    let t = j as f32 / h as f32;
                    let c = 0.05 + 0.9 * t;
                    [c, c * 1.1, c * 1.4, 1.0]
                }
            });
            // 构造函数本身是「表数据合法」的断言；此处只核 pdf，故不绑定变量名。
            let _table = TexTable::new(&refs, &texels);
            let total = cdf.total_weight();
            let mut worst = 0.0f32;
            let mut n = 0u32;
            for j in 0..h {
                for i in 0..w {
                    // 用 uv 空间的纹素中心（双线性在这里是恒等），
                    // 不能 texel_dir（那是立体角中心，v 会偏离纹素中心）。
                    let d = uv_to_dir((i as f32 + 0.5) / w as f32, (j as f32 + 0.5) / h as f32);
                    let p = cdf.pdf_w(d);
                    if p <= 0.0 {
                        continue; // 量化坍缩的尾部 texel（见模块文档）
                    }
                    // 目标密度是**离散分布**的密度 `L_k/Σ(L·Ω)`：
                    // 比的是 texel 自己的值，不是双线性插值值——后者在亮斑
                    // 边界处会掺进邻居 1e4 倍的太阳（那是 1 ulp 的 u 反演误差
                    // 放大出来的假差异，不是 pdf 的问题）。
                    let t = texels[(j * w + i) as usize];
                    let want = (luminance(Vec3::new(t[0], t[1], t[2])) as f64 / total) as f32;
                    worst = worst.max((p - want).abs() / want);
                    n += 1;
                }
            }
            assert!(n > (w * h) * 9 / 10, "{label}：可用 texel 过少 {n}");
            assert!(worst < 1e-3, "{label}：pdf 与 L/Σ(L·Ω) 最大相对偏差 {worst}");
        }
    }

    #[test]
    fn spot_list_catches_the_sun_and_keeps_pdf_consistent() {
        let (w, h) = (64u32, 32u32);
        let (cdf, _, _) = cdf_of(w, h, |i, j| {
            if (10..12).contains(&i) && j == 8 {
                [1.0e4, 1.0e4, 1.0e4, 1.0]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        });
        let st = cdf.stats();
        assert_eq!(st.spots, 2, "2 个 texel 的太阳应全部进亮斑列表");
        assert!(st.p_spot > 0.0 && st.p_spot < 1.0, "P_s 应在 (0,1)：{}", st.p_spot);
        // 采样必须两个分支都走得到（各 2000 次里都出现）。
        let mut n_spot = 0u32;
        for k in 0..4000u32 {
            let u1 = (k as f32 + 0.5) / 4000.0;
            let u2 = ((k * 7 % 4000) as f32 + 0.5) / 4000.0;
            let s = cdf.sample(u1, u2).expect("非全黑环境必须可采样");
            assert!(s.pdf_w > 0.0, "采样返回非正 pdf");
            let (i, j) = cdf.texel_of(s.wi);
            if (10..12).contains(&i) && j == 8 {
                n_spot += 1;
            }
        }
        assert!(n_spot > 0, "亮斑分支未被采样到");
        assert!(n_spot < 4000, "残差分支未被采样到");
    }

    #[test]
    fn sampling_frequency_matches_cdf_widths() {
        // 逆 CDF 的机械校验：N 个均匀数的落点频率必须与区间宽度成正比。
        let q: Vec<u64> = vec![1, 3, 0, 7, 9, 2];
        let prefix = cum_u32(&q);
        let total: u64 = q.iter().sum();
        let n = 60_000u32;
        let mut hits = vec![0u32; q.len()];
        for k in 0..n {
            let u = (k as f32 + 0.5) / n as f32;
            let (idx, frac) = pick_u32(&prefix, u);
            assert!((0.0..1.0).contains(&frac), "分数越界：{frac}");
            hits[idx] += 1;
        }
        for (i, &qi) in q.iter().enumerate() {
            let want = qi as f32 / total as f32;
            let got = hits[i] as f32 / n as f32;
            // 3σ：σ = √(n·p·(1−p))/n
            let p = want as f64;
            let sigma = (p * (1.0 - p) / n as f64).sqrt() as f32;
            assert!(
                (got - want).abs() <= 3.0 * sigma + 1.0 / n as f32,
                "箱 {i}：期望 {want} 实测 {got}（3σ={}）",
                3.0 * sigma
            );
        }
    }

    #[test]
    fn texel_dir_roundtrips_to_same_texel() {
        let (w, h) = (32u32, 16u32);
        let (cdf, _, _) = cdf_of(w, h, |_, _| [1.0, 1.0, 1.0, 1.0]);
        for j in 0..h as usize {
            for i in 0..w as usize {
                for &f in &[0.05f32, 0.5, 0.95] {
                    let d = cdf.texel_dir(j, i, f, f);
                    let (i2, j2) = cdf.texel_of(d);
                    assert_eq!((i2, j2), (i, j), "texel({i},{j},f={f}) 反查落到 ({i2},{j2})");
                }
            }
        }
    }

    #[test]
    fn build_is_bitwise_deterministic() {
        let (a, _, _) = cdf_of(40, 20, |i, j| {
            let t = ((i * 13 + j * 7) % 17) as f32;
            [t, t * 0.5, t * 0.25, 1.0]
        });
        let (b, _, _) = cdf_of(40, 20, |i, j| {
            let t = ((i * 13 + j * 7) % 17) as f32;
            [t, t * 0.5, t * 0.25, 1.0]
        });
        assert_eq!(a.cond, b.cond, "条件 CDF 两次构建不一致");
        assert_eq!(a.marg, b.marg, "边际 CDF 两次构建不一致");
        assert_eq!(a.marg_res, b.marg_res, "残差边际两次构建不一致");
        assert_eq!(a.res_cond, b.res_cond, "残差条件两次构建不一致");
        assert_eq!(a.p_spot.to_bits(), b.p_spot.to_bits());
        // 交换纹素流的遍历顺序（列优先 → 行优先）不应改变任何一个 CDF 字节：
        // 整数前缀和可交换，这是"顺序无关"的机械证据。
        let (refs, texels) = env_table(40, 20, |i, j| {
            let t = ((i * 13 + j * 7) % 17) as f32;
            [t, t * 0.5, t * 0.25, 1.0]
        });
        let table = TexTable::new(&refs, &texels);
        let c = EnvCdf::build(1, table).unwrap();
        assert_eq!(c.marg, a.marg, "同一份输入重建后边际 CDF 变了");
    }

    #[test]
    fn black_env_is_well_defined() {
        let (cdf, refs, texels) = cdf_of(8, 4, |_, _| [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(cdf.total_weight(), 0.0);
        assert!(cdf.sample(0.5, 0.5).is_none(), "全黑环境不应产出样本");
        assert_eq!(cdf.pdf_w(Vec3::new(0.0, 1.0, 0.0)).to_bits(), 0.0f32.to_bits());
        let table = TexTable::new(&refs, &texels);
        let l = cdf.radiance(table, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(l.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(cdf.stats().spots, 0);
    }

    #[test]
    fn invalid_handle_and_bounds_are_rejected() {
        let (refs, texels) = env_table(4, 2, |_, _| [1.0, 1.0, 1.0, 1.0]);
        let table = TexTable::new(&refs, &texels);
        assert!(EnvCdf::build(0, table).is_none(), "句柄 0 = 无贴图");
        assert!(EnvCdf::build(2, table).is_none(), "越界句柄必须拒绝");
        // 纹素区间越界（描述说 4×2，实际只给 3 个纹素）。
        let short = vec![refs[0]; 1];
        let bad = TexTable::new(&short, &texels[..3]);
        assert!(EnvCdf::build(1, bad).is_none(), "纹素区间越界必须拒绝");
    }
}
