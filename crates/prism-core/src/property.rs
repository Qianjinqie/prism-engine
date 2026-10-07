//! 属性通道 PropertyChannel（工作流 A2）。
//!
//! 语义权威：v2.0 Phase 1 表 A2 行、
//!  类图 `PropertyChannel`（`eval(t) → SolvedParams`）、D-7（Spring 解析
//! 解 + 自研受限表达式求值器，纯函数契约）。
//!
//! **一切节点参数皆通道**：`PropertyChannel` 把"写死一个数"与"随时间变化的
//! 曲线 / 弹簧 / 表达式 / 噪声"统一为 **f(t) 纯函数**——同 t 同值、无隐藏
//! 状态（求值纯函数契约）。通道值经 [`sample_binding`] 解算并写
//! 入节点 [`Params`]，这就是"属性通道喂 Params"的路径（节点三元
//! 组的特解参数位）。
//!
//! 各通道的数学件宿主是 `prism-anim`（track/spring/easing/expr）；
//! 通道模型、两个组合子（Remap / Loop）与求值分派。
//!
//! # 确定性
//! - 全部通道是 t 的纯函数：`sample` 不读全局状态、不私开 RNG；
//! - 唯一需要上下文的是 Wiggle（噪声种子派生）：[`PropertyChannel::sample_ctx`]
//!  把 `EvalCtx` 的 (全局种子， 节点id) 经整数哈希混入噪声种子（
//!   hash 派生 = 唯一随机源）——同节点同帧必同值，不同节点必不同流。

use prism_anim::{Expr, LoopMode, Spring, Track, Wiggle};

use crate::node::{EvalCtx, NodeBinding, ParamValue, Params};

// ---------------------------------------------------------------------------
// 通道模型（类图 PropertyChannel）
// ---------------------------------------------------------------------------

/// 属性通道（全家桶）：一切变体都是 t 的纯函数，t 单位为秒。
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyChannel {
    /// 常量通道：f(t) ≡ v。
    Const(f32),
    /// 关键帧轨道（S9 分段多项式；easing 函数表在 `prism-anim::easing`）。
    Track(Track),
    /// 阻尼弹簧闭式解（S7 闭式线性响应；：O(1) 无迭代、seek 零成本）。
    Spring(Spring),
    /// 受限表达式通道（本批最小算术 + 变量 t/p/w/h；完整语法随 A4 DSL 落地）。
    Expr(Expr),
    /// 噪声驱动抖动（接 `prism-noise` 的 PCG/value-noise/fbm）。
    Wiggle(Wiggle),
    /// 时间重映射组合子：inner 的时间轴从 `out` 区间线性映射到 `in` 区间。
    /// 线性映射保序（单调），端点严格对齐（t = out_hi 时取 inner(in_hi)）。
    Remap {
        /// 内层通道。
        inner: Box<PropertyChannel>,
        /// inner 自身时间轴的起点（秒）。
        in_lo: f32,
        /// inner 自身时间轴的终点（秒）。
        in_hi: f32,
        /// 引擎时间轴的起点（秒）。
        out_lo: f32,
        /// 引擎时间轴的终点（秒）。
        out_hi: f32,
    },
    /// 循环组合子：把 inner 在 `[0, duration]` 上的表现按模式铺满整条时间轴。
    Loop {
        /// 内层通道。
        inner: Box<PropertyChannel>,
        /// 循环模式（direct / loop / pingpong，见 `prism_anim::LoopMode`）。
        mode: LoopMode,
        /// 循环周期（秒，> 0）。
        duration: f32,
    },
}

impl PropertyChannel {
    /// 纯 f(t) 求值：同 t 同值（逐位一致）。
    ///
    /// 上下文无关通道的直接入口；Wiggle 以其自身 `seed` 求值（同 seed 同 t
    /// 同值）。需要按节点派生噪声流时走 [`Self::sample_ctx`]。
    pub fn sample(&self, t: f32) -> f32 {
        match self {
            Self::Const(v) => *v,
            Self::Track(track) => track.sample(t),
            Self::Spring(spring) => spring.sample(t),
            Self::Expr(expr) => expr.sample(t),
            Self::Wiggle(wiggle) => wiggle.sample(t),
            Self::Remap { inner, in_lo, in_hi, out_lo, out_hi } => {
                inner.sample(remap_time(t, *in_lo, *in_hi, *out_lo, *out_hi))
            }
            Self::Loop { inner, mode, duration } => inner.sample(mode.map(t, *duration)),
        }
    }

    /// EvalCtx 衔接入口。与 [`Self::sample`] 的唯一差别：
    /// Wiggle 的噪声种子由 (通道 seed, 全局种子， 节点id) 整数哈希派生——
    /// 同节点同帧必同值、不同节点不同流（防网格伪影）；其余通道是纯 f(t)，
    /// 忽略 ctx。
    pub fn sample_ctx(&self, t: f32, ctx: &EvalCtx) -> f32 {
        match self {
            Self::Wiggle(wiggle) => {
                let seed = prism_anim::mix_seeds(wiggle.seed, ctx.global_seed, ctx.node_id);
                wiggle.sample_with_seed(t, seed)
            }
            other => other.sample(t),
        }
    }
}

/// 线性时间重映射：把 t 从 `out` 区间映到 `in` 区间（纯函数，保序）。
///
/// 退化 `out_hi == out_lo` 时恒映射到 `in_lo`——文档化行为；DSL 层将
/// 以防呆规则拦截退化区间（宪章 C18 防呆规则库）。
fn remap_time(t: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = out_hi - out_lo;
    if span == 0.0 {
        return in_lo;
    }
    in_lo + (t - out_lo) / span * (in_hi - in_lo)
}

/// 通道 → Params 的喂给路径：按 t 解算绑定的通道，把结果以
/// `ParamValue::F32` 写入 `params[param]`。
///
/// 名字引用（`node.rs::ChannelRef`）到通道实例的解析属 A4（anim/expr 块
/// codegen）；本批通道随绑定显式携带。
pub fn sample_binding(
    binding: &NodeBinding,
    channel: &PropertyChannel,
    t: f32,
    ctx: &EvalCtx,
    params: &mut Params,
) {
    let value = channel.sample_ctx(t, ctx);
    params.set(binding.param.as_str(), ParamValue::F32(value));
}

// ---------------------------------------------------------------------------
// 测试（关卡 2/3/4：Track 边界、Loop 三模式、Remap 单调性、Const 直通、
// Wiggle 种子确定性——数学件自身的关卡断言在 prism-anim 各模块内）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_anim::{EasingKind, Keyframe};

    /// 测试用上升轨道：0s:0 → 1s:10（线性）。
    fn rising_track() -> Track {
        Track::new(vec![
            Keyframe::new(0.0, 0.0, EasingKind::Linear),
            Keyframe::new(1.0, 10.0, EasingKind::Linear),
        ])
    }

    #[test]
    fn const_channel_passthrough() {
        let ch = PropertyChannel::Const(2.5);
        for t in [-1.0_f32, 0.0, 0.5, 7.25, 1e6] {
            assert_eq!(ch.sample(t), 2.5, "Const 直通：任意 t 同值");
            let ctx = EvalCtx::new(7, 1, 3);
            assert_eq!(ch.sample_ctx(t, &ctx), 2.5, "Const 经 ctx 仍直通");
        }
    }

    #[test]
    fn track_channel_matches_direct_track() {
        let track = rising_track();
        let ch = PropertyChannel::Track(track.clone());
        for t in [0.0_f32, 0.25, 0.5, 0.99, 1.0, 5.0] {
            assert_eq!(ch.sample(t), track.sample(t), "通道分派与直调逐位一致 t={t}");
        }
    }

    #[test]
    fn spring_channel_start_and_convergence() {
        // 阶跃弹簧：x0=0 → target=1，欠阻尼。
        let ch = PropertyChannel::Spring(prism_anim::SpringPreset::WOBBLY.spring(0.0, 1.0));
        assert_eq!(ch.sample(0.0), 0.0, "t=0 精确取初值");
        let settled = ch.sample(30.0);
        assert!(
            (settled - 1.0).abs() < 1e-6,
            "t=30s 应收敛到 target，实际 {settled}"
        );
    }

    #[test]
    fn expr_channel_evaluates() {
        let ch = PropertyChannel::Expr(Expr::parse("t * 10 + 1").unwrap());
        assert!((ch.sample(0.3) - 4.0).abs() < 1e-6, "t*10+1 @ t=0.3 应为 4");
    }

    #[test]
    fn wiggle_channel_seed_ctx_determinism() {
        let wiggle = Wiggle::new(1.0, 2.0, 3, 0xAB_CD);
        let ch = PropertyChannel::Wiggle(wiggle);
        let ctx_a = EvalCtx::new(42, 9, 5);
        // 同 ctx 同 t：逐位一致。
        for t in [0.0_f32, 0.1, 1.7, 33.3] {
            let a = ch.sample_ctx(t, &ctx_a);
            let b = ch.sample_ctx(t, &ctx_a);
            assert_eq!(a.to_bits(), b.to_bits(), "同 seed 同 t 必逐位一致 t={t}");
        }
        // 不同节点 id：流必不同（8 个 t 里至少 6 个不同，规避巧合碰撞）。
        let ctx_b = EvalCtx::new(42, 10, 5);
        let differing = (0..8)
            .filter(|i| ch.sample_ctx(*i as f32, &ctx_a) != ch.sample_ctx(*i as f32, &ctx_b))
            .count();
        assert!(differing >= 6, "不同节点 id 的噪声流应不同，仅 {differing}/8 处不同");
    }

    #[test]
    fn remap_channel_monotone_and_endpoints() {
        // inner 在 [0,1] 上 0→10；引擎时间轴 [2,6] 重映射回去。
        let ch = PropertyChannel::Remap {
            inner: Box::new(PropertyChannel::Track(rising_track())),
            in_lo: 0.0,
            in_hi: 1.0,
            out_lo: 2.0,
            out_hi: 6.0,
        };
        // 端点严格对齐。
        assert_eq!(ch.sample(2.0), 0.0, "t=out_lo 应取 inner(in_lo)");
        assert!((ch.sample(6.0) - 10.0).abs() < 1e-5, "t=out_hi 应取 inner(in_hi)");
        // 单调性：t 递增 ⇒ 值不减（线性重映射保序）。
        let mut prev = ch.sample(1.0);
        for i in 1..=100 {
            let t = 1.0 + i as f32 * 0.05;
            let v = ch.sample(t);
            assert!(v >= prev, "Remap 单调性破坏 @ t={t}: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn loop_channel_three_modes() {
        // inner：0..1s 线性 0→10；duration = 1s。
        let make = |mode: LoopMode| PropertyChannel::Loop {
            inner: Box::new(PropertyChannel::Track(rising_track())),
            mode,
            duration: 1.0,
        };
        let direct = make(LoopMode::Direct);
        let loopy = make(LoopMode::Loop);
        let pingpong = make(LoopMode::PingPong);

        // Direct：首前取首、末后取末（钳制）。
        assert_eq!(direct.sample(-1.0), 0.0);
        assert_eq!(direct.sample(0.0), 0.0);
        assert!((direct.sample(0.5) - 5.0).abs() < 1e-6);
        assert!((direct.sample(1.0) - 10.0).abs() < 1e-6, "Direct 末点取末帧");
        assert!((direct.sample(2.5) - 10.0).abs() < 1e-6, "Direct 超时钳末帧");

        // Loop：周期回绕；t=duration 整回绕到首帧（u mod 1）。
        assert_eq!(loopy.sample(0.0), 0.0);
        assert!((loopy.sample(0.5) - 5.0).abs() < 1e-6);
        assert_eq!(loopy.sample(1.0), 0.0, "Loop 在 t=duration 回绕到首帧");
        assert!((loopy.sample(1.5) - 5.0).abs() < 1e-6);
        assert_eq!(loopy.sample(2.0), 0.0);

        // PingPong：三角折返，t=duration 为折返点（末帧）。
        assert_eq!(pingpong.sample(0.0), 0.0);
        assert!((pingpong.sample(0.5) - 5.0).abs() < 1e-6);
        assert!((pingpong.sample(1.0) - 10.0).abs() < 1e-6, "PingPong 在 t=duration 折返（末帧）");
        assert!((pingpong.sample(1.5) - 5.0).abs() < 1e-6);
        assert_eq!(pingpong.sample(2.0), 0.0, "PingPong 一个整周期后回到首帧");
    }

    #[test]
    fn sample_binding_writes_f32_param() {
        let binding = NodeBinding {
            param: "opacity".into(),
            channel: crate::node::ChannelRef { name: "fade".into() },
        };
        let ch = PropertyChannel::Const(0.75);
        let mut params = Params::new();
        let ctx = EvalCtx::new(1, 1, 1);
        sample_binding(&binding, &ch, 3.0, &ctx, &mut params);
        assert_eq!(
            params.get("opacity"),
            Some(&ParamValue::F32(0.75)),
            "通道值应以 F32 写入对应参数槽"
        );
    }
}
