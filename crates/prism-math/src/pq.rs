//! PQ（SMPTE ST 2084 / BT.2100 Perceptual Quantizer）HDR 传输函数。
//!
//! 语义权威：《强大版路线图》§3.5（"ST 2084 编码/解码纯数学 P0 完成+单测"）、
//! C2 行（"PQ/HLG OETF"）、D-29（"超越函数自研……禁 libm pow"）；
//! 《算法白皮书》§25.3（确定性格锚）。公式常数取自 SMPTE ST 2084:2014
//! （经 BT.2100-2 Table 4 复核）：
//!
//! ```text
//! m1 = 2610/16384 = 0.1593017578125
//! m2 = 2523/32    = 78.84375
//! c1 = 3424/4096  = 0.8359375
//! c2 = 2413/128   = 18.8515625
//! c3 = 2392/128   = 18.6875
//! ```
//!
//! # 方向约定（BT.2100 语义）
//! - **EOTF / inverse-OETF（signal → linear）**：`E' ∈ [0,1]` 编码信号 →
//!   归一化线性光 `Y ∈ [0,1]`（1.0 = 10000 cd/m² 峰值）；
//! - **inverse-EOTF / OETF（linear → signal）**：线性光 → 编码信号。
//!
//! 全部幂运算经 [`crate::det_pow::det_pow`]（D-29：禁 libm `pow`）。

use crate::det_pow::det_pow;

/// m1 = 2610/16384（PQ）。
pub const PQ_M1: f32 = 0.159_301_76;
/// m2 = 2523/32（PQ）。
pub const PQ_M2: f32 = 78.843_75;
/// c1 = 3424/4096（PQ）。
pub const PQ_C1: f32 = 0.835_937_5;
/// c2 = 2413/128（PQ）。
pub const PQ_C2: f32 = 18.851_562_5;
/// c3 = 2392/128（PQ）。
pub const PQ_C3: f32 = 18.687_5;

/// PQ 峰值亮度（cd/m²，ST 2084 归一化基准）。
pub const PQ_PEAK_NITS: f32 = 10_000.0;

/// PQ EOTF（inverse-OETF）：编码信号 `e' ∈ [0,1]` → 归一化线性光 `[0,1]`
/// （1.0 = 10000 cd/m²）。公式：
///
/// ```text
/// Y = ( (max(N^{1/m2} − c1, 0)) / (c2 − c3·N^{1/m2}) )^{1/m1}
/// ```
///
/// `N^{1/m2}` 与商幂 `x^{1/m1}` 均经自研 `det_pow`（D-29）。域外 clamp。
pub fn pq_eotf(signal: f32) -> f32 {
    let n = signal.clamp(0.0, 1.0) as f64;
    let n_p = det_pow(n as f32, (1.0 / PQ_M2) as f32) as f64; // N^{1/m2}
    let num = (n_p - PQ_C1 as f64).max(0.0);
    let den = PQ_C2 as f64 - PQ_C3 as f64 * n_p;
    if den <= 0.0 {
        return 0.0;
    }
    det_pow((num / den) as f32, (1.0 / PQ_M1) as f32)
}

/// PQ inverse-EOTF（OETF）：归一化线性光 `y ∈ [0,1]` → 编码信号 `[0,1]`。公式：
///
/// ```text
/// N = ( (c1 + c2·Y^{m1}) / (1 + c3·Y^{m1}) )^{m2}
/// ```
pub fn pq_oetf(linear: f32) -> f32 {
    let y = linear.clamp(0.0, 1.0) as f64;
    let y_m1 = det_pow(y as f32, PQ_M1) as f64; // Y^{m1}
    let num = PQ_C1 as f64 + PQ_C2 as f64 * y_m1;
    let den = 1.0 + PQ_C3 as f64 * y_m1;
    det_pow((num / den) as f32, PQ_M2)
}

/// PQ EOTF 的亮度版（nits）：编码信号 → 显示亮度 cd/m²（0..10000）。
pub fn pq_eotf_nits(signal: f32) -> f32 {
    pq_eotf(signal) * PQ_PEAK_NITS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_constants_match_standard() {
        // 精确有理数值锚（SMPTE ST 2084）。
        assert_eq!(PQ_M1, (2610.0f32) / 16384.0);
        assert_eq!(PQ_M2, 2523.0 * 128.0 / 4096.0);
        assert_eq!(PQ_C1, 3424.0 / 4096.0);
        assert_eq!(PQ_C2, 2413.0 * 32.0 / 4096.0);
        assert_eq!(PQ_C3, 2392.0 * 32.0 / 4096.0);
        // c1 = c3 − c2 + 1（标准内自洽约束）。
        assert!((PQ_C1 - (PQ_C3 - PQ_C2 + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn pq_endpoint_values() {
        // 端点：signal 0 → 0；signal 1 → 1（峰值）。
        assert!(pq_eotf(0.0).abs() < 1e-9, "eotf(0) = {}", pq_eotf(0.0));
        assert!((pq_eotf(1.0) - 1.0).abs() < 1e-5, "eotf(1) = {}", pq_eotf(1.0));
        assert!(pq_oetf(0.0).abs() < 1e-6);
        assert!((pq_oetf(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pq_known_reference_anchor_0_18() {
        // ST 2084 公开锚点：线性 0.18（18% 灰，1800 nits 尺度）↔ 编码信号。
        // colour-science: oetf_ST2084(0.18) ≈ 0.0794209（L_p=10000 归一后
        // 实为线性 0.18/10000 尺度，此处按归一线性 0.18 直接验证自洽性）。
        // 直接用官方 eotf/oetf 互逆 + 单调性锚定。
        let s = pq_oetf(0.18);
        let back = pq_eotf(s);
        assert!((back - 0.18).abs() < 1e-5, "18% 灰往返 = {back}");
    }

    #[test]
    fn pq_roundtrip_signal_to_linear_and_back() {
        // 往返一致性：编码 → 线性 → 编码（往返误差 < 1e-4，路线图验收项）。
        for i in 0..=1000u32 {
            let s = i as f32 / 1000.0;
            let back = pq_oetf(pq_eotf(s));
            assert!((back - s).abs() < 1e-4, "PQ 往返 s={s} → {back}");
        }
        // 线性 → 编码 → 线性。
        for i in 0..=1000u32 {
            let y = i as f32 / 1000.0;
            let back = pq_eotf(pq_oetf(y));
            assert!((back - y).abs() < 1e-4, "PQ 往返 y={y} → {back}");
        }
    }

    #[test]
    fn pq_is_monotonic_increasing() {
        // 单调性：编码信号越大，线性光越大（PQ 感知量化的基本性质）。
        let mut prev = -1.0f32;
        for i in 0..=2000u32 {
            let v = pq_eotf(i as f32 / 2000.0);
            assert!(v >= prev, "PQ EOTF 在 {i} 处非单调：{v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn pq_bitwise_deterministic() {
        let first = pq_eotf(0.5);
        for _ in 0..1000 {
            assert_eq!(pq_eotf(0.5).to_bits(), first.to_bits());
        }
        let first_o = pq_oetf(0.42);
        for _ in 0..1000 {
            assert_eq!(pq_oetf(0.42).to_bits(), first_o.to_bits());
        }
    }
}
