//! `prism-anim` — 属性通道数学件库（工作流 A2； Spring 解析解 /
//!  噪声； S7 闭式线性响应 + S9 分段多项式的特解宿主）。
//!
//! 内容：
//!
//! - [`easing`] — 缓动函数表（linear/quad/cubic/…/elastic，CSS 同名惯例，
//!   宪章 C5/C14 行业词汇）；
//! - [`track`] — 关键帧轨道（插值 + 边界规则）与循环模式（direct/loop/pingpong）；
//! - [`spring`] — 阻尼弹簧**闭式解析解**（欠阻尼/临界/过阻尼三分支，O(1)
//!  无迭代，seek 即求值）；
//! - [`expr`] — 受限表达式求值骨架（最小算术 + 变量 t/p/w/h；完整语法随
//!   A4 DSL 落地）；
//! - [`presets`] — 预置特解参数包（弹簧预设，命名对照 React Spring 惯例）；
//! - [`Wiggle`] — 噪声驱动抖动通道（接 `prism-noise`）。
//!
//! # 确定性契约
//!
//! 一切公开函数都是其参数的纯函数：无内部可变状态、无隐藏 RNG、无 libm
//! 超越函数。sin/cos/exp 由本 crate 内 [`det`] 模块以**固定系数多项式**实现
//! （内部 f64 运算，IEEE 逐运算确定）——D-29 的 crate 内自含过渡实现，待
//! `prism-math` 确定性数学件落地后统一收编。因此**同参数跨平台逐位一致**。

pub mod camera_presets;
pub mod easing;
pub mod expr;
pub mod presets;
pub mod spring;
pub mod track;

pub use camera_presets::{CameraMove, CameraPose, DollyZoom, Handheld, LensSpec};
pub use easing::EasingKind;
pub use expr::{Expr, ExprEnv, ExprError};
pub use presets::SpringPreset;
pub use spring::Spring;
pub use track::{Keyframe, LoopMode, Track};

use prism_noise::fbm::Fbm;

// ---------------------------------------------------------------------------
// Wiggle 通道（噪声驱动抖动；A2 文件清单内置于 lib.rs，体量小不单开文件）
// ---------------------------------------------------------------------------

/// Wiggle 通道：噪声驱动的参数抖动（AE wiggle 同概念）。
///
/// 值 = `amplitude × fbm(seed, t·frequency)`：value-noise 基 + 指定倍频程
/// （lacunarity 2、gain 0.5），fbm 输出归一化 `[-1, 1]`，故值域为
/// `[-amplitude, amplitude]`。纯函数：同 seed 同 t 同值（逐位）。
#[derive(Debug, Clone, PartialEq)]
pub struct Wiggle {
    /// 振幅（输出值域 `[-amplitude, amplitude]`）。
    pub amplitude: f32,
    /// 基频（Hz）：噪声坐标 = `t × frequency`。
    pub frequency: f32,
    /// fbm 倍频程数（0 → 恒 0）。
    pub octaves: u32,
    /// 通道种子（格 Zᵏ；EvalCtx 衔接时经 [`mix_seeds`] 与全局上下文混合）。
    pub seed: u64,
}

impl Wiggle {
    /// 以振幅 / 基频 / 倍频程 / 种子构造 Wiggle 通道。
    pub fn new(amplitude: f32, frequency: f32, octaves: u32, seed: u64) -> Self {
        Self { amplitude, frequency, octaves, seed }
    }

    /// 纯 f(t) 求值：以通道自身 `seed` 派生（同 seed 同 t 同值）。
    pub fn sample(&self, t: f32) -> f32 {
        self.sample_with_seed(t, self.seed)
    }

    /// 以外部派生种子求值（`prism-core` 的 EvalCtx 衔接路径）。
    pub fn sample_with_seed(&self, t: f32, seed: u64) -> f32 {
        let fbm = Fbm::new(self.octaves, 2.0, 0.5);
        self.amplitude * fbm.value2(seed, t * self.frequency, 0.0)
    }
}

/// EvalCtx 种子混合：`seed = hash(通道种子， 全局种子， 节点id)`。
///
/// 整数哈希派生 = 全引擎唯一随机源；通道层不私开 RNG。PCG 整数运算跨平台
/// 逐位一致。第 4 格放通道域分隔常量（PCG 按槽位加盐，混入
/// 0 也不会与低元哈希混淆）。
pub fn mix_seeds(channel_seed: u64, global_seed: u64, node_id: u64) -> u64 {
    const ANIM_CHANNEL_TAG: u64 = 0xA11A_0000_0000_0001;
    prism_noise::pcg::hash3(channel_seed, global_seed, node_id, ANIM_CHANNEL_TAG)
}

// ---------------------------------------------------------------------------
// 确定性超越函数（D-29 crate 内自含过渡实现）
// ---------------------------------------------------------------------------

/// 固定系数确定性数学件：sin / cos / exp 的区间逼近多项式实现。
///
/// 为什么自含：D-29 规定超越函数一律"区间逼近多项式 + 固定系数"自研、禁
/// libm（跨平台差 1–2 ulp 击穿逐位一致）；`prism-math` 的确定性数学件尚未
/// 落地且不在当前文件清单内，故先在 crate 内自含，待其落地后收编替换。
///
/// 全部只用 `+ - × ÷`、`floor`/`round`（IEEE 精确取整运算）与位构造——
/// IEEE 754 对每个基本运算的结果都有唯一定义，因此**跨平台逐位一致**。
pub(crate) mod det {
    /// 2/π（f64 精确常量）。
    const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;
    /// π/2（f64 精确常量）。
    const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;
    /// log₂e。
    const LOG_2_E: f64 = std::f64::consts::LOG2_E;

    /// sin 在 `[-π/4, π/4]` 的 Taylor 展开（Horner 于 r²，截断 < 1e-11）。
    #[inline]
    fn poly_sin(r: f64) -> f64 {
        let r2 = r * r;
        r * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0
                    + r2 * (-1.0 / 5040.0
                        + r2 * (1.0 / 3_628_80.0 + r2 * (-1.0 / 39_916_800.0))))))
    }

    /// cos 在 `[-π/4, π/4]` 的 Taylor 展开（截断 < 1e-12）。
    #[inline]
    fn poly_cos(r: f64) -> f64 {
        let r2 = r * r;
        1.0 + r2 * (-0.5
            + r2 * (1.0 / 24.0
                + r2 * (-1.0 / 720.0
                    + r2 * (1.0 / 40_320.0 + r2 * (-1.0 / 3_628_800.0)))))
    }

    /// sin(x)，f64 全精度。区间折叠到 `[-π/4, π/4]` 后查象限。
    /// 精度界：`|x| ≲ 1e8` 时误差 ≲ 1e-10（动画时间尺度远小于此）。
    pub fn sin_f64(x: f64) -> f64 {
        let k = (x * FRAC_2_PI).round();
        let r = x - k * FRAC_PI_2;
        let q = (k as i64).rem_euclid(4);
        let (s, c) = (poly_sin(r), poly_cos(r));
        match q {
            0 => s,
            1 => c,
            2 => -s,
            _ => -c,
        }
    }

    /// cos(x)，f64 全精度（与 [`sin_f64`] 同一折叠）。
    pub fn cos_f64(x: f64) -> f64 {
        let k = (x * FRAC_2_PI).round();
        let r = x - k * FRAC_PI_2;
        let q = (k as i64).rem_euclid(4);
        let (s, c) = (poly_sin(r), poly_cos(r));
        match q {
            0 => c,
            1 => -s,
            2 => -c,
            _ => s,
        }
    }

    /// sin(x)，f32 出入口（内部 f64 运算后一次舍入）。
    #[inline]
    pub fn sin_f32(x: f32) -> f32 {
        sin_f64(x as f64) as f32
    }

    /// cos(x)，f32 出入口。
    #[inline]
    pub fn cos_f32(x: f32) -> f32 {
        cos_f64(x as f64) as f32
    }

    /// 2^f 在 `[-0.5, 0.5)` 的 Taylor 展开：`2^f = Σ (ln2)^k / k! · f^k`。
    ///
    /// 系数在 const 上下文按 IEEE 递推编译期求值（固定值，IEEE 754 对乘除
    /// 结果唯一定义 → 跨平台逐位一致）；对称折叠使 `|f| ≤ 0.5`，12 项截断
    /// 相对误差 < 1e-14（若不折叠、f→1 处 7 项截断 ~4e-7，不可接受）。
    #[inline]
    fn poly_exp2(f: f64) -> f64 {
        const C: [f64; 12] = {
            let mut c = [0.0_f64; 12];
            c[0] = 1.0;
            let mut i = 1_usize;
            while i < 12 {
                c[i] = c[i - 1] * std::f64::consts::LN_2 / (i as f64);
                i += 1;
            }
            c
        };
        let mut acc = 0.0_f64;
        for c in C.iter().rev() {
            acc = acc * f + c;
        }
        acc
    }

    /// exp(x) = 2^(x·log₂e)，f64 全精度。
    ///
    /// 下溢（x ≲ −745）按 0 返回、上溢按 +∞ 返回—— 附注"钳为
    /// 收敛态"的确定性形态；次正规区间精度降级按文档化舍入处理。
    pub fn exp_f64(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x == f64::INFINITY {
            return f64::INFINITY;
        }
        if x == f64::NEG_INFINITY {
            return 0.0;
        }
        let y = x * LOG_2_E;
        let n0 = y.floor();
        let f0 = y - n0; // ∈ [0, 1)
        // 对称折叠：f0 ≥ 0.5 时借一位指数，使多项式自变量落在 [-0.5, 0.5)。
        let (n, f) = if f0 >= 0.5 { (n0 + 1.0, f0 - 1.0) } else { (n0, f0) };
        if n > 1023.0 {
            return f64::INFINITY;
        }
        if n < -1022.0 {
            return 0.0;
        }
        // 2^n 经位构造精确得到（指数域整数平移，无舍入）。
        let two_n = f64::from_bits(((n as i64 + 1023) as u64) << 52);
        poly_exp2(f) * two_n
    }

    /// exp(x)，f32 出入口。
    #[inline]
    pub fn exp_f32(x: f32) -> f32 {
        exp_f64(x as f64) as f32
    }

    /// 2^x，f32 出入口（`exp2(y) = e^(y·ln2)`）。
    #[inline]
    pub fn exp2_f32(x: f32) -> f32 {
        exp_f64((x as f64) * std::f64::consts::LN_2) as f32
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 1 附属：确定性数学件精度与逐位复现；关卡 4：Wiggle 种子确定性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// det_sin 对照 std f64 sin（**仅测试内**用 libm 参考做精度对拍）。
    #[test]
    fn det_sin_matches_reference() {
        // 常规幅度：覆盖正负、跨象限、大参数。
        let mut x = -100.0_f64;
        while x <= 100.0 {
            let diff = (det::sin_f64(x) - x.sin()).abs();
            assert!(diff < 1e-9, "det_sin({x}) 偏差 {diff}");
            x += 0.037;
        }
        // 特殊点。
        assert_eq!(det::sin_f64(0.0), 0.0);
        assert!((det::sin_f64(std::f64::consts::FRAC_PI_6) - 0.5).abs() < 1e-12);
        assert!((det::sin_f64(std::f64::consts::PI) - 0.0).abs() < 1e-12);
    }

    /// det_cos 对照 std f64 cos。
    #[test]
    fn det_cos_matches_reference() {
        let mut x = -100.0_f64;
        while x <= 100.0 {
            let diff = (det::cos_f64(x) - x.cos()).abs();
            assert!(diff < 1e-9, "det_cos({x}) 偏差 {diff}");
            x += 0.041;
        }
        assert_eq!(det::cos_f64(0.0), 1.0);
        assert!((det::cos_f64(std::f64::consts::FRAC_PI_3) - 0.5).abs() < 1e-12);
    }

    /// det_exp 对照 std f64 exp；下溢钳 0、上溢钳 +∞（附注）。
    #[test]
    fn det_exp_matches_reference_and_saturates() {
        let mut x = -50.0_f64;
        while x <= 50.0 {
            let diff = (det::exp_f64(x) - x.exp()).abs() / x.exp();
            assert!(diff < 1e-9, "det_exp({x}) 相对偏差 {diff}");
            x += 0.043;
        }
        assert_eq!(det::exp_f64(0.0), 1.0);
        assert_eq!(det::exp_f64(-800.0), 0.0, "深度下溢应钳为 0（收敛态）");
        assert_eq!(det::exp_f64(1000.0), f64::INFINITY, "上溢应为 +∞");
    }

    /// 关卡 1（逐位复现）：同参数重复调用 10000 次逐位一致。
    #[test]
    fn det_functions_bitwise_reproducible() {
        let x = 2.718_281_828_f64;
        let expected = (
            det::sin_f64(x),
            det::cos_f64(x),
            det::exp_f64(x),
            det::sin_f32(x as f32),
            det::cos_f32(x as f32),
            det::exp_f32(x as f32),
            det::exp2_f32(0.37_f32),
        );
        for _ in 0..10_000 {
            assert_eq!(
                (
                    det::sin_f64(x),
                    det::cos_f64(x),
                    det::exp_f64(x),
                    det::sin_f32(x as f32),
                    det::cos_f32(x as f32),
                    det::exp_f32(x as f32),
                    det::exp2_f32(0.37_f32),
                ),
                expected
            );
        }
    }

    /// 关卡 4：同 seed 同 t 同值（逐位，10000 次重复）。
    #[test]
    fn wiggle_same_seed_same_t_bitwise() {
        let wiggle = Wiggle::new(2.0, 3.0, 4, 0x1234_5678);
        for t in [0.0_f32, 0.5, 1.0, 7.25, 100.0] {
            let expected = wiggle.sample(t);
            for _ in 0..10_000 {
                assert_eq!(wiggle.sample(t).to_bits(), expected.to_bits());
            }
        }
    }

    /// 关卡 4：不同 seed 产生不同流；同 seed 与 mix_seeds 派生种子结果可对上。
    #[test]
    fn wiggle_seed_sensitivity_and_mix() {
        let a = Wiggle::new(1.0, 2.0, 3, 1);
        let b = Wiggle::new(1.0, 2.0, 3, 2);
        let differing = (0..16)
            .filter(|i| a.sample(*i as f32) != b.sample(*i as f32))
            .count();
        assert!(differing >= 12, "不同 seed 的流应不同，仅 {differing}/16 处不同");

        // mix_seeds 路径：sample_with_seed(t, mix_seeds(...)) 是确定性的，
        // 且与任意外部种子逐位可复算。
        let mixed = mix_seeds(1, 42, 9);
        let v1 = a.sample_with_seed(1.5, mixed);
        let v2 = a.sample_with_seed(1.5, mixed);
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    /// 关卡 4：值域 `[-amplitude, amplitude]`；零振幅恒 0；零倍频程恒 0。
    #[test]
    fn wiggle_range_and_degenerate_params() {
        let wiggle = Wiggle::new(3.0, 5.0, 4, 0xFEED);
        for i in 0..2000_u64 {
            let t = i as f32 * 0.01;
            let v = wiggle.sample(t);
            assert!(
                (-3.0..=3.0).contains(&v),
                "Wiggle 越界：t={t} v={v}（值域 ±amplitude）"
            );
        }
        let silent = Wiggle::new(0.0, 5.0, 4, 0xFEED);
        assert_eq!(silent.sample(1.234), 0.0, "零振幅恒 0");
        let flat = Wiggle::new(3.0, 5.0, 0, 0xFEED);
        assert_eq!(flat.sample(1.234), 0.0, "零倍频程为空和恒 0");
    }
}
