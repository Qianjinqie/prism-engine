//! 声像器：恒功率声像 + 自动化曲线（立体声重塑）。
//!
//! 单声道 → 立体声，`pan ∈ [−1, +1]`（−1 全左，+1 全右）。采用 **恒功率** 律
//! `gL = cos(θ), gR = sin(θ)`，`θ = (pan+1)·π/4`——`gL² + gR² = 1`（能量守恒，
//! 不随声像位置忽大忽小）。三角函数走 [`crate::det`]（D-29）。
//!
//! **自动化**：`pan_at(t)` 为自动化曲线（起始/结束声像 + 形状），是采样索引/
//! 时间的纯函数——seek 即求值，无状态机。

use crate::buffer::PcmBuffer;
use crate::det;

/// 声像自动化曲线形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanCurve {
    /// 线性从 `from` 到 `to`。
    Linear,
    /// 保持 `from` 恒定（忽略 `to`）。
    Hold,
    /// 平滑 S 形（smoothstep）从 `from` 到 `to`。
    Smooth,
}

/// 声像自动化：在持续 `duration_secs` 内从 `from` 移到 `to`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanAutomation {
    /// 起始声像 ∈ [−1, 1]。
    pub from: f32,
    /// 结束声像 ∈ [−1, 1]。
    pub to: f32,
    /// 自动化时长（秒；≤0 表示瞬时到位）。
    pub duration_secs: f32,
    /// 曲线形状。
    pub curve: PanCurve,
}

impl PanAutomation {
    /// 常量声像（无自动化）。
    pub fn fixed(pan: f32) -> Self {
        Self { from: pan, to: pan, duration_secs: 0.0, curve: PanCurve::Hold }
    }

    /// 在音符/片段内 `t` 秒处的声像值（纯函数，`t` 超时长后钳到 `to`）。
    pub fn at(&self, t: f32, total_secs: f32) -> f32 {
        let denom = if self.duration_secs > 0.0 { self.duration_secs } else { total_secs };
        let u = if denom > 0.0 { (t / denom).clamp(0.0, 1.0) } else { 1.0 };
        let shaped = match self.curve {
            PanCurve::Linear => u,
            PanCurve::Hold => 0.0,
            PanCurve::Smooth => u * u * (3.0 - 2.0 * u), // smoothstep
        };
        (self.from + (self.to - self.from) * shaped).clamp(-1.0, 1.0)
    }
}

/// 恒功率声像增益对 `(gL, gR)`，`gL² + gR² = 1`。
pub fn constant_power_gains(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    let theta = (p + 1.0) * std::f32::consts::FRAC_PI_4 as f32 / 1.0;
    // θ = (pan+1)·π/4，走确定性 cos/sin（f64 内部）。
    let theta_f64 = ((p + 1.0) as f64) * std::f64::consts::FRAC_PI_4;
    let gl = det::cos_f64(theta_f64) as f32;
    let gr = det::sin_f64(theta_f64) as f32;
    let _ = theta; // 保持 f64 口径，避免 f32 中间量泄漏
    (gl, gr)
}

/// 把 `pan` 应用于单声道样本 → 立体声交错缓冲。
pub fn pan_to_stereo(input: &[f32], sample_rate: u32, pan: f32) -> PcmBuffer {
    let (gl, gr) = constant_power_gains(pan);
    let mut data = Vec::with_capacity(input.len() * 2);
    for &x in input {
        data.push(x * gl);
        data.push(x * gr);
    }
    PcmBuffer::from_interleaved(sample_rate, 2, data).expect("2 声道交错长度必然合法")
}

/// 自动化声像：沿时间逐帧求 `pan_at(t)` 后恒功率成型（纯函数，逐帧独立）。
pub fn pan_automated(input: &[f32], sample_rate: u32, auto: &PanAutomation) -> PcmBuffer {
    let total = input.len() as f32 / sample_rate as f32;
    let mut data = Vec::with_capacity(input.len() * 2);
    for (i, &x) in input.iter().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let (gl, gr) = constant_power_gains(auto.at(t, total));
        data.push(x * gl);
        data.push(x * gr);
    }
    PcmBuffer::from_interleaved(sample_rate, 2, data).expect("2 声道交错长度必然合法")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 恒功率：任意声像位置 gL² + gR² = 1（能量守恒断言）。
    #[test]
    fn constant_power_energy_preserved() {
        for pan in [-1.0_f32, -0.5, 0.0, 0.25, 1.0] {
            let (gl, gr) = constant_power_gains(pan);
            let e = gl * gl + gr * gr;
            assert!((e - 1.0).abs() < 1e-5, "pan={pan} 能量 {e} 应为 1");
        }
        let (gl, gr) = constant_power_gains(0.0);
        assert!((gl - gr).abs() < 1e-6 && (gl - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    /// 极值声像：−1 全左（右≈0），+1 全右（左≈0）。
    #[test]
    fn extreme_pan_isolation() {
        let (gl, gr) = constant_power_gains(-1.0);
        assert!((gl - 1.0).abs() < 1e-5 && gr.abs() < 1e-5);
        let (gl, gr) = constant_power_gains(1.0);
        assert!(gl.abs() < 1e-5 && (gr - 1.0).abs() < 1e-5);
    }

    /// 立体声输出：样本 = 输入 × 增益对，逐帧确定性。
    #[test]
    fn pan_to_stereo_scales_and_orders() {
        let buf = pan_to_stereo(&[1.0, -0.5], 8_000, 0.0);
        assert_eq!(buf.channels(), 2);
        assert_eq!(buf.frames(), 2);
        let l = buf.sample(0, 0).unwrap();
        let r = buf.sample(0, 1).unwrap();
        assert!((l - r).abs() < 1e-6, "居中时左右相等");
        assert_eq!(pan_to_stereo(&[1.0], 1, 0.0).as_slice(), pan_to_stereo(&[1.0], 1, 0.0).as_slice());
    }

    /// 自动化曲线锚点：线性中点为均值、Hold 恒定、Smooth 端点到位。
    #[test]
    fn automation_curve_anchors() {
        let auto = PanAutomation { from: -1.0, to: 1.0, duration_secs: 1.0, curve: PanCurve::Linear };
        assert!((auto.at(0.0, 1.0) + 1.0).abs() < 1e-6);
        assert!(auto.at(0.5, 1.0).abs() < 1e-6, "线性中点应为 0");
        assert!((auto.at(1.0, 1.0) - 1.0).abs() < 1e-6);
        let hold = PanAutomation::fixed(0.3);
        assert!((hold.at(0.9, 1.0) - 0.3).abs() < 1e-6);
        let smooth = PanAutomation { from: 0.0, to: 1.0, duration_secs: 1.0, curve: PanCurve::Smooth };
        assert!((smooth.at(0.0, 1.0)).abs() < 1e-6);
        assert!((smooth.at(1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((smooth.at(0.5, 1.0) - 0.5).abs() < 1e-6, "smoothstep 中点 0.5");
    }
}
