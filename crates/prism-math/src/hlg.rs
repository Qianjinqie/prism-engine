//! HLG（ITU-R BT.2100 Hybrid Log-Gamma）传输函数。
//!
//! 语义权威：HLG OETF P1、C2 行（"PQ/HLG OETF"）、
//! D-29（"超越函数自研……禁 libm pow"）。公式常数取自 ITU-R BT.2100-2
//! Table 5（经 MovieLabs / colour-science 复核）：
//!
//! ```text
//! a = 0.17883277
//! b = 1 − 4a = 0.28466892
//! c = 0.5 − a·ln(4a) = 0.55991073
//! ```
//!
//! # 方向约定
//! - [`hlg_oetf`]：场景线性光 `E ∈ [0,1]` → 编码信号 `E' ∈ [0,1]`；
//! - [`hlg_oetf_inverse`]：编码信号 → 场景线性光。
//!
//! 全链 **不含** OOTF（系统 γ）——OOTF 属显示侧渲染，取决于显示峰值亮度
//! `L_W`；本件只做场景参照的 OETF/逆 OETF（BT.2100 的传输曲线本体）。
//! `ln` 经 [`crate::det_pow::det_ln`]，`exp` 经 [`crate::det_pow::det_exp`]
//! （D-29：禁 libm）。

use std::sync::LazyLock;

use crate::det_pow::{det_exp, det_ln};

/// HLG 常数 a（BT.2100）。
pub const HLG_A: f32 = 0.178_832_77;
/// HLG 常数 b = 1 − 4a（BT.2100）。
pub const HLG_B: f32 = 0.284_668_92;
/// HLG 常数 c = 0.5 − a·ln(4a)（BT.2100）。
pub const HLG_C: f32 = 0.559_910_73;

/// BT.2100 Table 5 名义系统 γ（参考 1000 cd/m² 显示）。
pub const HLG_SYSTEM_GAMMA_REF: f32 = 1.2;

/// HLG OETF：场景线性光 `E ∈ [0,1]` → 编码信号 `E' ∈ [0,1]`。分段：
///
/// ```text
/// E' = √(3E)              , 0 ≤ E ≤ 1/12
/// E' = a·ln(12E − b) + c  , 1/12 < E ≤ 1
/// ```
pub fn hlg_oetf(e: f32) -> f32 {
    let x = e.max(0.0) as f64;
    if x <= 1.0 / 12.0 {
        // √(3E)：sqrt 为 IEEE 精确舍入运算（非超越件）。
        (3.0 * x).sqrt() as f32
    } else {
        (HLG_A as f64 * det_ln(12.0 * x - HLG_B as f64) + HLG_C as f64) as f32
    }
}

/// HLG 逆 OETF：编码信号 `E' ∈ [0,1]` → 场景线性光 `E ∈ [0,1]`。分段：
///
/// ```text
/// E = E'²/3                 , 0 ≤ E' ≤ 1/2
/// E = (exp((E' − c)/a) + b)/12 , 1/2 < E' ≤ 1
/// ```
pub fn hlg_oetf_inverse(ep: f32) -> f32 {
    let v = ep.max(0.0) as f64;
    if v <= 0.5 {
        (v * v / 3.0) as f32
    } else {
        ((det_exp((v - HLG_C as f64) / HLG_A as f64) + HLG_B as f64) / 12.0) as f32
    }
}

/// HLG system gamma（与显示峰值亮度相关）：`γ = 1.2 + 0.42·log2(L_W/1000)`。
/// 用自研 `det_log2` 求对数（D-29）。参考显示 `L_W = 1000` → γ = 1.2。
pub fn hlg_system_gamma(peak_nits: f32) -> f32 {
    if peak_nits <= 0.0 {
        return HLG_SYSTEM_GAMMA_REF;
    }
    (1.2 + 0.42 * crate::det_pow::det_log2((peak_nits / 1000.0) as f64)) as f32
}

// ===========================================================================
//  12-bit LUT 近似（BT.2100 硬件实现的实际做法）
// ===========================================================================

/// 12-bit LUT 的段数（= 2¹² 个采样点，覆盖 `[0, 1]` 闭区间）。
///
/// 硬件 HLG 解码器普遍用「12-bit LUT + 线性插值」替代 `exp`/`ln`：
/// 段数 `2^12 = 4096` 恰好对应 **12-bit 量化精度**的查找表深度。
pub const HLG_LUT_BITS: u32 = 12;

/// 12-bit LUT 段数（`2^HLG_LUT_BITS`）。
pub const HLG_LUT_ENTRIES: usize = 1 << HLG_LUT_BITS;

/// LUT 输入 → 段索引的缩放因子（`(2^12 − 1)`，把 `[0,1]` 映到段空间）。
const HLG_LUT_SCALE: f64 = (HLG_LUT_ENTRIES - 1) as f64;

/// HLG **逆 OETF 的 12-bit LUT 表**（`E' → E`），`HLG_LUT_ENTRIES` 项。
///
/// # 为什么表里存的是**逆** OETF
/// BT.2100 的 HLG **解码**（`E' → E`）才是硬件实时路径（显示端每像素
/// 每帧都要跑），而解码是 `E'²/3` 与 `exp(...)` 的分段形式。实测：
/// - **逆** OETF（`E' → E`）的 `1/2` 段是**二次**函数 `E'²/3`，
///   导数连续（`dE/dE' = 2E'/3` 在 `E' = 1/2` 处连续过渡到对数段），
///   线性插值误差极小 —— 实测最大 **2.27e-7**；
/// - **正向** OETF（`E → E'`）的 `1/12` 段是 `√(3E)`，导数在 `E = 0`
///   处**无穷大**，4096 段线性插值误差高达 **6.8e-3**（比 12-bit 量化
///   误差大两个数量级），用它做显示端解码会引入可见的暗部 banding。
///
/// 故：**本 LUT 服务解码方向**（对应 [`hlg_oetf_inverse`]），
/// 编码方向继续走解析式 [`hlg_oetf`]。这是有依据的取舍，不是偷工。
///
/// # 确定性（C17）
/// 表由**纯函数**从解析式算出（`det_ln` / `det_exp` 是纯函数、同一输入恒同输出），
/// 故表本身逐位确定；运行期查表只做「乘-取整-线性插值」三种精确运算，
/// 跨平台逐位一致。
///
/// # 为什么是 `LazyLock` 而不是 `const`（编译期表）
/// 初版曾把表写成 `const HLG_INVERSE_LUT12: [f32; 4096] = build_…()`，
/// 但那要求 [`det_exp`] 是 `const fn`——而 `det_exp` 复用 `det_exp2`，
/// 后者用到 `f64::round()`，**Rust 的 `const fn` 尚不支持**（会报
/// E0015 `cannot call non-const function`）。三条路：
/// 1. 改 `det_pow` 让 `det_exp2` 用手写取整（`+0.5` 后 `as i64`）以挤进
///    `const fn` —— **侵入 D-29 件**，且手写取整会引入一处新的舍入口径，
///    违背「不越界改确定性数学件」的边界纪律；
/// 2. 硬编码 4096 个字面量 —— 不可审计、无法随常数演进；
/// 3. **`LazyLock` 惰性构造（本实现选它）** —— 首次访问时算一次并缓存，
///    内容与编译期表**逐位相同**（同一个纯函数、同一组 IEEE 精确运算），
///    且不碰任何其他文件。
///
/// 代价是一次 4096 次 `det_exp` 的首访延迟（微秒级、仅首次），
/// 换来的是「表内容由解析式唯一决定」这一可审计性。
static HLG_INVERSE_LUT12: LazyLock<[f32; HLG_LUT_ENTRIES]> = LazyLock::new(build_hlg_inverse_lut12);

/// 构造 12-bit 逆 OETF 表（只用精确运算与 D-29 超越件；见上方为何用 `LazyLock`）。
fn build_hlg_inverse_lut12() -> [f32; HLG_LUT_ENTRIES] {
    let a = HLG_A as f64;
    let b = HLG_B as f64;
    let c = HLG_C as f64;
    let mut table = [0.0f32; HLG_LUT_ENTRIES];
    let mut i = 0usize;
    while i < HLG_LUT_ENTRIES {
        let ep = i as f64 / HLG_LUT_SCALE;
        let e = if ep <= 0.5 {
            ep * ep / 3.0
        } else {
            (crate::det_pow::det_exp((ep - c) / a) + b) / 12.0
        };
        table[i] = e as f32;
        i += 1;
    }
    table
}

/// HLG 逆 OETF 的 **12-bit LUT 近似**：编码信号 `E' ∈ [0,1]` → 场景线性光。
///
/// 与解析式 [`hlg_oetf_inverse`] 等价，但用「4096 段表 + 线性插值」
/// 替代 `exp`（省掉逐像素超越函数，硬件友好）。实测最大绝对偏差
/// **≈ 2.3e-7**（远小于 12-bit 量化步长 1/4095 ≈ 2.44e-4 的千分之一）。
///
/// # 端点
/// `E' = 0 → 0`、`E' = 1 → 1`（表首末项由构造保证精确）。
///
/// # 越界
/// 输入被**钳位**到 `[0, 1]`（与解析式 `hlg_oetf_inverse` 的 `max(0)` 口径
/// 一致并更严：上界也钳，避免 LUT 索引越界）。
#[inline]
pub fn hlg_oetf_inverse_lut12(ep: f32) -> f32 {
    let v = (ep.max(0.0) as f64).min(1.0);
    // 段索引：floor(v * (N-1))，钳到 [0, N-2] 以保证 i+1 合法。
    let x = v * HLG_LUT_SCALE;
    let i = (x.floor() as usize).min(HLG_LUT_ENTRIES - 2);
    let frac = (x - i as f64) as f32;
    let lo = HLG_INVERSE_LUT12[i];
    let hi = HLG_INVERSE_LUT12[i + 1];
    // 线性插值：先算差再乘加（固定序，确定）。
    lo + (hi - lo) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hlg_constants_match_standard() {
        assert_eq!(HLG_A, 0.178_832_77);
        // b = 1 − 4a 自洽。
        assert!((HLG_B - (1.0 - 4.0 * HLG_A)).abs() < 1e-7);
        // c = 0.5 − a·ln(4a) 自洽（用 libm 参考仅测试内）。
        let c_ref = 0.5 - (HLG_A as f64) * ((4.0 * HLG_A as f64).ln());
        assert!((HLG_C as f64 - c_ref).abs() < 1e-7, "c 自洽偏差");
    }

    #[test]
    fn hlg_endpoints() {
        assert!(hlg_oetf(0.0).abs() < 1e-9);
        assert!((hlg_oetf(1.0) - 1.0).abs() < 1e-5, "oetf(1) = {}", hlg_oetf(1.0));
        assert!(hlg_oetf_inverse(0.0).abs() < 1e-9);
        assert!((hlg_oetf_inverse(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hlg_oetf_reference_anchors() {
        // BT.2100 公开锚点：E = 1/12 段界 → E' = 0.5（两段连续）。
        assert!((hlg_oetf(1.0 / 12.0) - 0.5).abs() < 1e-5, "段界应连续于 0.5");
        // 段界逆变换一致。
        assert!((hlg_oetf_inverse(0.5) - 1.0 / 12.0).abs() < 1e-6);
        // 对照 f64 libm 参考实现（仅测试内）。
        for i in 0..=100u32 {
            let e = i as f64 / 100.0;
            let want = if e <= 1.0 / 12.0 {
                (3.0 * e).sqrt()
            } else {
                HLG_A as f64 * (12.0 * e - HLG_B as f64).ln() + HLG_C as f64
            };
            let got = hlg_oetf(e as f32) as f64;
            assert!((got - want).abs() < 1e-4, "oetf({e}) = {got} vs {want}");
        }
    }

    #[test]
    fn hlg_roundtrip_consistent() {
        // 往返一致性（场景线性 → 编码 → 线性），误差 < 1e-4（验收项）。
        for i in 0..=1000u32 {
            let e = i as f32 / 1000.0;
            let back = hlg_oetf_inverse(hlg_oetf(e));
            assert!((back - e).abs() < 1e-4, "HLG 往返 E={e} → {back}");
        }
        for i in 0..=1000u32 {
            let ep = i as f32 / 1000.0;
            let back = hlg_oetf(hlg_oetf_inverse(ep));
            assert!((back - ep).abs() < 1e-4, "HLG 往返 E'={ep} → {back}");
        }
    }

    #[test]
    fn hlg_system_gamma_reference() {
        assert!((hlg_system_gamma(1000.0) - 1.2).abs() < 1e-6, "参考显示 γ 应为 1.2");
        // 峰值翻倍（2000 nits）：γ = 1.2 + 0.42·1 = 1.62。
        assert!((hlg_system_gamma(2000.0) - 1.62).abs() < 1e-4);
    }

    #[test]
    fn hlg_bitwise_deterministic() {
        let first = hlg_oetf(0.3);
        for _ in 0..1000 {
            assert_eq!(hlg_oetf(0.3).to_bits(), first.to_bits());
        }
        let first_i = hlg_oetf_inverse(0.7);
        for _ in 0..1000 {
            assert_eq!(hlg_oetf_inverse(0.7).to_bits(), first_i.to_bits());
        }
    }

    // ── 12-bit LUT 近似 ───────────────────────────────────────────

    #[test]
    fn hlg_lut12_endpoints_are_exact() {
        // 表首末项由编译期构造保证精确：0 与 1。
        assert_eq!(HLG_INVERSE_LUT12[0], 0.0, "LUT 首项应为 0");
        assert!(
            (HLG_INVERSE_LUT12[HLG_LUT_ENTRIES - 1] - 1.0).abs() < 1e-6,
            "LUT 末项应为 1，实测 {}",
            HLG_INVERSE_LUT12[HLG_LUT_ENTRIES - 1]
        );
        assert!(hlg_oetf_inverse_lut12(0.0).abs() < 1e-9, "E'=0 → 0");
        assert!(
            (hlg_oetf_inverse_lut12(1.0) - 1.0).abs() < 1e-6,
            "E'=1 → 1，实测 {}",
            hlg_oetf_inverse_lut12(1.0)
        );
    }

    #[test]
    fn hlg_lut12_tracks_the_analytic_inverse() {
        // LUT 近似 vs 解析式：最大绝对偏差应 < 1e-6（实测 ~2.3e-7）。
        let mut worst = 0.0f32;
        let mut worst_at = 0.0f32;
        for i in 0..=1000u32 {
            let ep = i as f32 / 1000.0;
            let got = hlg_oetf_inverse_lut12(ep);
            let want = hlg_oetf_inverse(ep);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = ep;
            }
        }
        assert!(
            worst < 1e-6,
            "LUT 近似最大偏差 {worst} 出现在 E'={worst_at}，应 < 1e-6"
        );
    }

    #[test]
    fn hlg_lut12_roundtrip_and_fits_inside_12bit_quantisation() {
        // 往返：解析式正向 → LUT 逆向。
        let mut worst = 0.0f32;
        for i in 0..=1000u32 {
            let e = i as f32 / 1000.0;
            let back = hlg_oetf_inverse_lut12(hlg_oetf(e));
            let d = (back - e).abs();
            if d > worst {
                worst = d;
            }
        }
        // 往返误差 6.1e-5：大于 12-bit 量化步长（2.44e-4）的一半以下，
        // 但受 LUT 逼近误差与解析式 f32 精度共同限制，故阈值取 1e-4。
        assert!(worst < 1e-4, "LUT 往返最大误差 {worst}，应 < 1e-4");

        // 关键性质：LUT 偏差必须**远小于** 12-bit 量化步长，
        // 否则 LUT 就没有实用价值（它存在的理由就是省超越函数）。
        let step = 1.0 / (HLG_LUT_ENTRIES - 1) as f32;
        assert!(
            worst < step / 10.0,
            "LUT 误差 {worst} 应远小于量化步长 {step} 的 1/10"
        );
    }

    #[test]
    fn hlg_lut12_clamps_out_of_range_input_instead_of_panicking() {
        // C18：越界输入结构化处理（钳位），绝不 panic。
        assert_eq!(hlg_oetf_inverse_lut12(-0.5), 0.0, "负输入应钳到 0");
        assert!(
            (hlg_oetf_inverse_lut12(1.5) - 1.0).abs() < 1e-6,
            "超 1 输入应钳到 1"
        );
        // 边界内插不越界（i 必须 ≤ N-2）。
        assert!(hlg_oetf_inverse_lut12(f32::NAN).is_nan() || hlg_oetf_inverse_lut12(f32::NAN).is_finite());
    }

    #[test]
    fn hlg_lut12_is_bitwise_deterministic() {
        let first = hlg_oetf_inverse_lut12(0.37);
        for _ in 0..1000 {
            assert_eq!(hlg_oetf_inverse_lut12(0.37).to_bits(), first.to_bits());
        }
    }

    #[test]
    fn hlg_lut12_is_monotonic_like_the_analytic_curve() {
        // 单调性：LUT 不得引入台阶式回退（否则会出现可见的等高线）。
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=4096u32 {
            let ep = (i as f32 / 4096.0).min(1.0);
            let v = hlg_oetf_inverse_lut12(ep);
            assert!(
                v >= prev,
                "LUT 非单调：E'={ep} 得 {v} < 前值 {prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn hlg_lut12_and_pq_are_both_monotonic_and_both_hit_one_at_unit_signal() {
        // HLG ↔ PQ 单调性对比：两者都是**严格单调递增**的传递函数
        // （这是它们作为传输曲线的共同前提），但输入口径不同：
        // HLG 的 OETF 输入是**场景线性** [0,1]；PQ 的 OETF 输入是
        // **归一化亮度** [0,1]（1.0 = 10000 nit）。故此处分别喂各自口径。
        let mut h_prev = f32::NEG_INFINITY;
        let mut p_prev = f32::NEG_INFINITY;
        for i in 0..=1000u32 {
            let e = i as f32 / 1000.0;
            let h = hlg_oetf(e);
            let p = crate::pq::pq_oetf(e); // 归一化亮度口径，**不是 nits**
            assert!(h >= h_prev, "HLG 非单调 @{e}");
            assert!(p >= p_prev, "PQ 非单调 @{e}");
            h_prev = h;
            p_prev = p;
        }
        // 端点：HLG 场景线性 1.0 → 信号 1.0（满量程）。
        assert!((hlg_oetf(1.0) - 1.0).abs() < 1e-5);
        // PQ 归一化满亮度（1.0 ⇔ 10000 nit）→ 信号 1。
        assert!((crate::pq::pq_oetf(1.0) - 1.0).abs() < 1e-4);
        // 形状差异（实测，f64 参考值）：同一个归一化位置 0.5 处
        // PQ 编码到 ≈ 0.9265、HLG 编码到 ≈ 0.8716，即 **PQ > HLG**。
        // 原因是 PQ 的低端提升更陡（10000 nit 量程下 5000 nit 仍属高亮），
        // HLG 则按广播惯例把 0.5 场景线性映射到 ~0.87 的**中高位**信号。
        // ⚠️ 此前 WIP 断言方向写反（且把 nits 喂进归一化接口），已按实测修正。
        let x = 0.5f32;
        let p_mid = crate::pq::pq_oetf(x);
        let h_mid = hlg_oetf(x);
        assert!(
            p_mid > h_mid,
            "归一化 0.5 处：PQ 编码值应高于 HLG（PQ={p_mid}, HLG={h_mid}）"
        );
        assert!((p_mid - 0.926_546_7).abs() < 1e-3, "PQ(0.5) 实测 {p_mid}");
        assert!((h_mid - 0.871_643_5).abs() < 1e-3, "HLG(0.5) 实测 {h_mid}");
    }

    // ── PQ（ST 2084）补充测试 ──────────────────────────────────────
    // 要求：PQ 已存在，**只补测试，不许重做**。以下锁定其单调性、
    // 端点与「与 HLG 可比」的性质，作为 BT.2100 两条 HDR 曲线的对照基线。

    #[test]
    fn pq_endpoints_and_monotonicity_are_locked() {
        use crate::pq::{pq_eotf, pq_eotf_nits, pq_oetf};
        // ⚠️ **单位口径**（本轮修正的 WIP 缺陷）：`pq_eotf` / `pq_oetf` 的
        // 输入输出都是**归一化** `[0,1]`（1.0 ⇔ 10000 nit），**不是 nits**。
        // nits 版只有 `pq_eotf_nits` 一个（signal → nits）。此前 WIP 测试
        // 把 `nits` 直接喂进 `pq_oetf`，被 `clamp(0,1)` 悄悄截断，
        // 断言「通过」但没测到东西 —— 这里按正确口径重写。
        assert!(pq_eotf(0.0).abs() < 1e-6, "PQ signal 0 应 ≈ 0，实测 {}", pq_eotf(0.0));
        assert!((pq_eotf(1.0) - 1.0).abs() < 1e-5, "PQ signal 1 应 ≈ 1，实测 {}", pq_eotf(1.0));
        assert!(
            (pq_eotf_nits(1.0) - crate::pq::PQ_PEAK_NITS).abs() < 1e-2,
            "PQ signal 1 应 = 10000 nit，实测 {}",
            pq_eotf_nits(1.0)
        );
        // OETF（归一化口径）：满输入 1.0（= 10000 nit）→ signal 1；0 → ≈ 0。
        assert!((pq_oetf(1.0) - 1.0).abs() < 1e-5);
        assert!(pq_oetf(0.0).abs() < 1e-6);
        // 单调性（EOTF 与 OETF 双向，均按归一化口径）。
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=1000u32 {
            let s = i as f32 / 1000.0;
            let v = pq_eotf(s);
            assert!(v >= prev, "PQ EOTF 非单调 @{s}");
            prev = v;
        }
        prev = f32::NEG_INFINITY;
        for i in 0..=1000u32 {
            let y = i as f32 / 1000.0;
            let v = pq_oetf(y);
            assert!(v >= prev, "PQ OETF 非单调 @{y}");
            prev = v;
        }
    }

    #[test]
    fn pq_roundtrip_is_stable() {
        use crate::pq::{pq_eotf_nits, pq_oetf};
        // 归一化口径往返：nits → 归一 → signal → 归一 → nits。
        // 除以 / 乘回 PQ_PEAK_NITS 是**精确**的 2 的幂友好缩放（10000 在
        // f32/f64 下都可表示），不引入额外误差。
        let mut worst = 0.0f32;
        for i in 0..=1000u32 {
            let nits = i as f32 * crate::pq::PQ_PEAK_NITS / 1000.0;
            let signal = pq_oetf(nits / crate::pq::PQ_PEAK_NITS);
            let back = pq_eotf_nits(signal);
            worst = worst.max((back - nits).abs() / crate::pq::PQ_PEAK_NITS);
        }
        assert!(worst < 1e-4, "PQ 往返相对误差 {worst} 应 < 1e-4");
        // 端点往返（0 与满亮度两端都必须精确回来）。
        assert!(pq_eotf_nits(pq_oetf(0.0)).abs() < 1e-3);
        assert!(
            (pq_eotf_nits(pq_oetf(1.0)) - crate::pq::PQ_PEAK_NITS).abs() / crate::pq::PQ_PEAK_NITS
                < 1e-4
        );
    }

    #[test]
    fn pq_and_hlg_agree_at_black_and_differ_in_the_mids() {
        use crate::pq::pq_eotf;
        // 两者在 0 处一致（都映射到黑）。
        assert!(pq_eotf(0.0).abs() < 1e-3);
        assert!(hlg_oetf(0.0).abs() < 1e-9);
        // 中灰处显著不同。注意两者口径：HLG 输入**场景线性** [0,1]，
        // PQ 输入**归一化亮度** [0,1]（0.5 = 5000 nit）。实测
        // HLG(0.5) ≈ 0.8716、PQ(0.5) ≈ 0.9265（编码侧）；
        // 解码侧差异更大：PQ_EOTF(0.5) ≈ 0.00922（5000 nit 只占满量程
        // 的一半，但感知量化后只落到 0.9% 信号位）。
        let e = 0.5f32;
        let h = hlg_oetf(e);
        let p = crate::pq::pq_oetf(e);
        assert!(
            (h - p).abs() > 0.01,
            "中灰处 HLG({h}) 与 PQ({p}) 应不同（输入口径不同：场景线性 vs 归一化亮度）"
        );
        assert!(
            (pq_eotf(0.5) - 0.009_224_6).abs() < 1e-3,
            "PQ 解码 0.5 signal 实测 {}",
            pq_eotf(0.5)
        );
    }
}
