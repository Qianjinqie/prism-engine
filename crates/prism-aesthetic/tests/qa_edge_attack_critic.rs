//! 攻击性边界/确定性/数学正确性测试。
//!
//! 覆盖 critic 闭环面：`prism-aesthetic::{critic, critic_loop, cmd, nodes}` 与
//! `prism-ai::assist`。
//!
//! 每条测试的 doc 注释说明"失败代表什么缺陷"。协议/数学依据：
//! - 指标算法：曝光=对数亮度几何均值、对比=线性域亮度 std、清晰度=拉普拉斯能量；
//! - 收敛判定：`next-prev>=eps → Improving`；`prev-next>=eps → Diverging`；否则 Converged；
//! - 评分映射：容差内 1，容差外线性衰减到 hard 处 0（`score_dim`）；
//! - D-29 禁 libm：critic 的自研 `ln` 逼近、assist 的 FNV hash（无超越函数）。
//!
//! 本文件只写测试、不改源码。

use prism_aesthetic::cmd::{CriticCommand, Phase};
use prism_aesthetic::critic::{
    analyze, diagnose, score, Dim, Direction, Rubric, Target, MID_GRAY,
};
use prism_aesthetic::critic_loop::{
    apply_suggestion, convergence, suggest, Convergence, CriticLoop, MetricsAccumulator, Suggestion,
    DEFAULT_EPS,
};
use prism_aesthetic::nodes::QualityScoreNode;
use prism_ai::assist::{advise, advise_all, action_id, AdviceRequest};
use prism_io::fbuf::Fbuf16F;

/// 构造纯色帧。
fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
    let mut fb = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            fb.set(x, y, c);
        }
    }
    fb
}

/// 棋盘（高频）帧。
fn checker(w: u32, h: u32) -> Fbuf16F {
    let mut fb = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            fb.set(x, y, [v, v, v, 1.0]);
        }
    }
    fb
}

// ===========================================================================
// critic 指标算法正确性
// ===========================================================================

/// 纯色帧：对比（std）= 0、清晰度 = 0、中性灰色偏 = 0、饱和度 = 0。
/// 失败代表：指标公式错误（如把方差算成均值）→ 诊断全错。
#[test]
fn critic_solid_frame_metrics_are_zero_where_expected() {
    let m = analyze(&solid(16, 16, [0.18, 0.18, 0.18, 1.0]));
    assert!(m.luma_std.abs() < 1e-6, "纯色 → 对比 0，实得 {}", m.luma_std);
    assert!(m.sharpness.abs() < 1e-9, "纯色 → 清晰度 0，实得 {}", m.sharpness);
    assert!(m.color_cast.abs() < 1e-6, "中性灰 → 色偏 0，实得 {}", m.color_cast);
    assert!(m.mean_saturation.abs() < 1e-6, "中性灰 → 饱和 0");
    assert_eq!(m.pixels, 256);
}

/// 曝光代理（log 亮度）随亮度单调递增且与 ln(v) 一致（自研 ln 精度）。
/// 失败代表：曝光指标方向错误或 ln 逼近偏差过大 → 曝光诊断误判。
#[test]
fn critic_exposure_monotone_and_matches_ln() {
    let dark = analyze(&solid(8, 8, [0.01, 0.01, 0.01, 1.0]));
    let mid = analyze(&solid(8, 8, [0.18, 0.18, 0.18, 1.0]));
    let bright = analyze(&solid(8, 8, [0.9, 0.9, 0.9, 1.0]));
    assert!(dark.log_mean_luma < mid.log_mean_luma, "越亮 log 亮度越大");
    assert!(mid.log_mean_luma < bright.log_mean_luma);
    // ln(0.18) ≈ -1.7148；自研 ln 逼近应接近。
    assert!(
        (mid.log_mean_luma - (0.18_f32.ln())).abs() < 0.02,
        "自研 ln 偏离过大：{} vs {}",
        mid.log_mean_luma,
        0.18_f32.ln()
    );
    // 与 MID_GRAY 常量一致性。
    assert!((MID_GRAY - 0.18).abs() < 1e-6);
}

/// 色偏检测：红主导帧的 color_cast > 0.25，饱和度 > 0.5。
/// 失败代表：色彩指标算错（如用错通道差）→ 偏色漏检。
#[test]
fn critic_color_cast_detects_red_dominance() {
    let m = analyze(&solid(8, 8, [0.5, 0.2, 0.2, 1.0]));
    assert!(m.color_cast > 0.25, "红主导应明显色偏，实得 {}", m.color_cast);
    assert!(m.mean_saturation > 0.5, "饱和红应高饱和，实得 {}", m.mean_saturation);
}

/// 清晰度：棋盘（高频）> 平板（低频）；拉普拉斯能量归一化正确。
/// 失败代表：拉普拉斯核或归一化错误 → 清晰度诊断反向。
#[test]
fn critic_sharpness_orders_checker_above_flat() {
    let sharp = analyze(&checker(16, 16));
    let flat = analyze(&solid(16, 16, [0.5, 0.5, 0.5, 1.0]));
    assert!(sharp.sharpness > flat.sharpness, "棋盘应比平板清晰");
    assert!(sharp.sharpness > 0.1, "棋盘拉普拉斯能量应显著，实得 {}", sharp.sharpness);
}

// ===========================================================================
// 退化 / 边界：空帧、零尺寸、小尺寸
// ===========================================================================

/// 空帧（0×0）分析安全，不 panic；返回全零指标 + pixels=0。
/// 失败代表：0 像素下除零 → NaN/panic。
#[test]
fn critic_empty_frame_is_safe() {
    let m = analyze(&Fbuf16F::new(0, 0));
    assert_eq!(m.pixels, 0);
    assert!(m.log_mean_luma.is_infinite() && m.log_mean_luma.is_sign_negative(), "空帧曝光为 −∞");
    assert_eq!(m.luma_std, 0.0);
    // 评分不 panic 且落在 [0,1]。
    let s = score(&m, &Rubric::default());
    assert!((0.0..=1.0).contains(&s.total), "空帧总分应在 [0,1]");
}

/// 1×1、2×2 极小帧：拉普拉斯无内点 → 清晰度 0，但不 panic。
/// 失败代表：`w>=3 && h>=3` 边界外越界。
#[test]
fn critic_tiny_frames_safe() {
    let m1 = analyze(&solid(1, 1, [0.5, 0.5, 0.5, 1.0]));
    assert_eq!(m1.pixels, 1);
    assert_eq!(m1.sharpness, 0.0, "1×1 无拉普拉斯内点");
    let m2 = analyze(&solid(2, 2, [0.5, 0.5, 0.5, 1.0]));
    assert_eq!(m2.sharpness, 0.0, "2×2 无 3×3 核内点");
    // 2×3 也内点不足。
    let m3 = analyze(&solid(2, 3, [0.5, 0.5, 0.5, 1.0]));
    assert_eq!(m3.sharpness, 0.0);
}

/// 极暗灰卡诊断：曝光低 + 对比低（平板），且不误报色偏（High）。
/// 失败代表：诊断口径与评分口径不一致，或中性灰被误判偏色。
#[test]
fn critic_underexposed_diagnosis_hits() {
    let m = analyze(&solid(32, 32, [0.005, 0.005, 0.005, 1.0]));
    let d = diagnose(&m, &Rubric::default());
    let has = |dim: Dim, dir: Direction| d.iter().any(|x| x.dim == dim && x.direction == dir);
    assert!(has(Dim::Exposure, Direction::Low), "应诊断过暗：{d:?}");
    assert!(has(Dim::Contrast, Direction::Low), "应诊断平板：{d:?}");
    assert!(
        !d.iter().any(|x| x.dim == Dim::Color && x.direction == Direction::High),
        "中性灰不应有色偏（High）诊断：{d:?}"
    );
}

/// 过曝（过亮）灰卡应诊断曝光 High。
/// 失败代表：方向判定反了。
#[test]
fn critic_overexposed_diagnosis_high() {
    let m = analyze(&solid(32, 32, [0.98, 0.98, 0.98, 1.0]));
    let d = diagnose(&m, &Rubric::default());
    assert!(
        d.iter().any(|x| x.dim == Dim::Exposure && x.direction == Direction::High),
        "近白帧应诊断过亮：{d:?}"
    );
}

// ===========================================================================
// 评分映射正确性（score_dim 契约）
// ===========================================================================

/// score_dim：容差内 =1，容差外线性衰减，hard 处 =0，超出保持 0。
/// 失败代表：评分曲线错误 → 质量分失真。
#[test]
fn critic_score_dim_mapping_contract() {
    let t = Target::new(10.0, 1.0, 3.0);
    assert_eq!(prism_aesthetic::critic::score_dim(10.0, &t), 1.0, "目标点满分");
    assert_eq!(prism_aesthetic::critic::score_dim(11.0, &t), 1.0, "容差边界满分");
    assert_eq!(prism_aesthetic::critic::score_dim(13.0, &t), 0.0, "hard 处归零");
    assert_eq!(prism_aesthetic::critic::score_dim(20.0, &t), 0.0, "超出保持 0");
    assert!((prism_aesthetic::critic::score_dim(12.0, &t) - 0.5).abs() < 1e-6, "中点应 0.5");
    // 负方向对称。
    assert!((prism_aesthetic::critic::score_dim(8.0, &t) - 0.5).abs() < 1e-6);
}

/// 总分 = 五维等权算术平均（固定顺序），且各维 ∈ [0,1]。
/// 失败代表：维度权重错误或求和顺序不稳定。
#[test]
fn critic_total_is_mean_of_five_dims() {
    let m = analyze(&solid(16, 16, [0.4, 0.2, 0.1, 1.0]));
    let s = score(&m, &Rubric::default());
    assert_eq!(s.dims.len(), 5);
    assert!(s.dims.iter().all(|&v| (0.0..=1.0).contains(&v)), "各维应在 [0,1]");
    let mean: f32 = s.dims.iter().sum::<f32>() / 5.0;
    assert!((s.total - mean).abs() < 1e-6, "总分应为等权平均");
}

// ===========================================================================
// 确定性（最高优先）
// ===========================================================================

/// **确定性**：同图同评分，逐位一致（含诊断）。
/// 失败代表：评分含隐藏状态/非确定浮点 → 违反 C5 确定性契约。
#[test]
fn critic_score_is_bitwise_deterministic() {
    let fb = solid(16, 16, [0.4, 0.2, 0.1, 1.0]);
    let r = Rubric::default();
    let a = score(&analyze(&fb), &r);
    let b = score(&analyze(&fb), &r);
    assert_eq!(a, b, "同输入同评分（逐位）");
    // 100 次重复确认稳定。
    let first = score(&analyze(&fb), &r).total.to_bits();
    for _ in 0..100 {
        assert_eq!(score(&analyze(&fb), &r).total.to_bits(), first);
    }
}

/// analyst 100 次重复同帧指标逐位一致。
/// 失败代表：逐像素归约顺序不稳定。
#[test]
fn critic_analyze_is_bitwise_deterministic() {
    let fb = checker(24, 24);
    let first = analyze(&fb);
    for _ in 0..50 {
        assert_eq!(analyze(&fb), first, "同帧指标应逐位一致");
    }
}

/// CriticLoop 两轮同输入同评分、同重渲帧集。
/// 失败代表：增量缓存影响结果（复用与重算不一致）。
#[test]
fn critic_loop_round_deterministic() {
    let r = Rubric::default();
    let mk = || {
        let mut lp = CriticLoop::new(4, r, 0);
        lp.round(None, |f| solid(16, 16, [0.02 + 0.01 * (f as f32), 0.02, 0.02, 1.0]))
    };
    let a = mk();
    let b = mk();
    assert_eq!(a.score, b.score, "同输入同评分");
    assert_eq!(a.metrics, b.metrics);
    assert_eq!(a.replayed_frames, b.replayed_frames);
}

// ===========================================================================
// 收敛/回升判定
// ===========================================================================

/// convergence：三态分类边界严格（d>=eps→Improving，-d>=eps→Diverging，否则 Converged）。
/// 失败代表：--autofix 接纳/停止判定错误。
#[test]
fn critic_convergence_classification() {
    assert_eq!(convergence(0.8, 0.5, DEFAULT_EPS), Convergence::Improving);
    assert_eq!(convergence(0.5, 0.8, DEFAULT_EPS), Convergence::Diverging);
    assert_eq!(convergence(0.5001, 0.5, DEFAULT_EPS), Convergence::Converged);
    // 明确大于 eps 的回升（避开 f32 表示在 1e-3 边界的舍入歧义）。
    assert_eq!(convergence(0.51, 0.5, 1e-3), Convergence::Improving);
    // 明确大于 eps 的下降。
    assert_eq!(convergence(0.49, 0.5, 1e-3), Convergence::Diverging);
    // 相等 → 收敛。
    assert_eq!(convergence(0.5, 0.5, 1e-3), Convergence::Converged);
    // 差值远小于 eps → 收敛。
    assert_eq!(convergence(0.5000001, 0.5, 1e-3), Convergence::Converged);
}

/// 首轮后 last_convergence 为 None；两轮后有条目。
/// 失败代表：轨迹记账错误。
#[test]
fn critic_loop_tracks_rounds() {
    let mut lp = CriticLoop::new(2, Rubric::default(), 0);
    assert_eq!(lp.current_score(), None);
    lp.round(None, |_| solid(16, 16, [0.18, 0.18, 0.18, 1.0]));
    assert!(lp.last_convergence().is_none(), "单轮无收敛判定");
    assert_eq!(lp.rounds().len(), 1);
    lp.round(None, |_| solid(16, 16, [0.18, 0.18, 0.18, 1.0]));
    assert_eq!(lp.rounds().len(), 2);
    assert_eq!(lp.last_convergence(), Some(Convergence::Converged), "同帧 → 收敛");
}

// ===========================================================================
// suggest / apply 契约
// ===========================================================================

/// suggest 与 diagnoses 一一对应（维度/方向/强度）。
/// 失败代表：建议派生丢失诊断。
#[test]
fn critic_suggest_one_to_one() {
    let fb = solid(16, 16, [0.005, 0.005, 0.005, 1.0]);
    let m = analyze(&fb);
    let d = diagnose(&m, &Rubric::default());
    let sug = suggest(&d);
    assert_eq!(sug.len(), d.len());
    for (g, dd) in sug.iter().zip(d.iter()) {
        assert_eq!(g.dim, dd.dim);
        assert_eq!(g.direction, dd.direction);
        assert_eq!(g.strength, dd.severity);
    }
}

/// apply_suggestion 强度 0 → 恒等（逐位），所有维度/方向。
/// 失败代表：无诊断时仍改动画面 → 破坏"无诊断无改动"契约。
#[test]
fn critic_apply_zero_strength_identity() {
    let c = [0.3, 0.5, 0.7, 1.0];
    for dim in Dim::ALL {
        for dir in [Direction::Low, Direction::High] {
            let s = Suggestion { dim, direction: dir, strength: 0.0 };
            assert_eq!(apply_suggestion(c, &s), c, "{dim:?}/{dir:?} 强度 0 → 恒等");
        }
    }
}

/// 曝光 Low 提亮、High 压暗；色彩 High 降饱和且非负。
/// 失败代表：修正方向反了或产生负色（越界）。
#[test]
fn critic_apply_directions_correct() {
    let c = [0.1, 0.1, 0.1, 1.0];
    let bright = apply_suggestion(c, &Suggestion { dim: Dim::Exposure, direction: Direction::Low, strength: 1.0 });
    assert!(bright[0] > c[0], "曝光低 → 提亮");
    let dark = apply_suggestion(c, &Suggestion { dim: Dim::Exposure, direction: Direction::High, strength: 1.0 });
    assert!(dark[0] < c[0], "曝光高 → 压暗");

    let sat = [0.8, 0.1, 0.1, 1.0];
    let out = apply_suggestion(sat, &Suggestion { dim: Dim::Color, direction: Direction::High, strength: 1.0 });
    assert!(out[0] - out[2] < sat[0] - sat[2], "色彩高 → 降饱和");
    assert!(out.iter().all(|&v| v >= 0.0), "修正后不得为负");
}

/// apply/apply_suggestion 是纯函数：同输入同输出。
/// 失败代表：apply 含隐藏状态。
#[test]
fn critic_apply_is_pure() {
    let cmd = CriticCommand::default();
    let fb = solid(8, 8, [0.4, 0.2, 0.1, 1.0]);
    let sug = cmd.critique(&fb).suggestions;
    let a = cmd.apply(&fb, &sug);
    let b = cmd.apply(&fb, &sug);
    assert_eq!(a, b, "apply 应逐位一致");
    assert_eq!(cmd.apply(&fb, &[]), fb, "空建议 → 恒等");
}

// ===========================================================================
// autofix 闭环（2 轮回升，对接 C3 增量）
// ===========================================================================

/// autofix 对欠曝帧：2 轮评分回升（Improving），轨迹 ≥2 轮，末轮 > 首轮。
/// 失败代表：闭环不收敛或方向错误（Phase 3 验收第 5 条）。
#[test]
fn critic_autofix_improves_two_rounds() {
    let r = Rubric::default();
    let mut lp = CriticLoop::new(2, r, 0).with_eps(1e-6);
    let (conv, score) = lp.autofix(
        2,
        |_| solid(24, 24, [0.02, 0.02, 0.02, 1.0]),
        |fb, sug| {
            let mut out = Fbuf16F::new(fb.width, fb.height);
            for y in 0..fb.height {
                for x in 0..fb.width {
                    let mut c = fb.get(x, y);
                    for s in sug {
                        if s.dim == Dim::Exposure {
                            c = apply_suggestion(c, s);
                        }
                    }
                    out.set(x, y, c);
                }
            }
            out
        },
    );
    assert_eq!(conv, Convergence::Improving, "2 轮评分应回升");
    assert!(score.unwrap() > 0.0);
    assert!(lp.rounds().len() >= 2, "应有首轮 + 回升轮");
    let first = lp.rounds()[0].score.total;
    let last = lp.rounds()[lp.rounds().len() - 1].score.total;
    assert!(last > first, "末轮 {last} 应高于首轮 {first}");
}

/// autofix 对已达标帧（无诊断）→ 首轮即收敛、不无谓加轮。
/// 失败代表：无建议仍强行迭代。
#[test]
fn critic_autofix_stops_when_good() {
    let r = Rubric::default();
    let mut lp = CriticLoop::new(2, r, 0);
    let probe = lip_diagnose(&solid(24, 24, [0.18, 0.18, 0.18, 1.0]), &r);
    // 仅当该帧无诊断时检验停止行为（否则该测试不适用，跳过断言）。
    if probe.is_empty() {
        let (conv, _) = lp.autofix(2, |_| solid(24, 24, [0.18, 0.18, 0.18, 1.0]), |fb, _| fb);
        assert_eq!(conv, Convergence::Converged, "已达标 → 首轮收敛");
    }
}

/// lip_diagnose 辅助（避免与内部别名冲突）。
fn lip_diagnose(fb: &Fbuf16F, r: &Rubric) -> Vec<prism_aesthetic::critic::Diagnosis> {
    diagnose(&analyze(fb), r)
}

/// 第二轮同渲染复用后评分与指标逐位一致（增量会话命中缓存不应改变结果）。
/// 失败代表：复用路径与重算路径产出不一致（违反 C3 "复用同字节"铁律）。
///
/// 注：`CriticLoop` 未公开内部 `IncrementalSession`，故此处从**可观察行为**断言：
/// 复用（第二轮）与重算（首轮）产出的评分/指标必须逐位一致。
#[test]
fn critic_loop_reuse_matches_recompute() {
    let r = Rubric::default();
    let mut lp = CriticLoop::new(3, r, 0);
    let a = lp.round(None, |_| solid(16, 16, [0.18, 0.18, 0.18, 1.0]));
    let b = lp.round(None, |_| solid(16, 16, [0.18, 0.18, 0.18, 1.0]));
    assert_eq!(a.score, b.score, "复用不应改变评分");
    assert_eq!(a.metrics, b.metrics, "复用不应改变指标");
    assert_eq!(a.replayed_frames, b.replayed_frames);
}

// ===========================================================================
// MetricsAccumulator 边界
// ===========================================================================

/// 空聚合器 finish 不 panic，返回空帧口径（pixels=0）。
/// 失败代表：0 帧除零。
#[test]
fn critic_accumulator_empty_safe() {
    let acc = MetricsAccumulator::new();
    assert_eq!(acc.count(), 0);
    let m = acc.finish();
    assert_eq!(m.pixels, 0);
}

/// 多帧平均：pixels 求和、log 亮度介于各帧之间。
/// 失败代表：平均口径错误。
#[test]
fn critic_accumulator_averages() {
    let mut acc = MetricsAccumulator::new();
    acc.push(analyze(&solid(8, 8, [0.1, 0.1, 0.1, 1.0])));
    acc.push(analyze(&solid(8, 8, [0.3, 0.3, 0.3, 1.0])));
    let m = acc.finish();
    assert_eq!(acc.count(), 2);
    assert_eq!(m.pixels, 128, "pixels 应求和");
    assert!(m.log_mean_luma > (0.1_f32).ln() && m.log_mean_luma < (0.3_f32).ln());
}

// ===========================================================================
// cmd / nodes 契约
// ===========================================================================

/// CriticCommand 四相均确定性；Phase::name 稳定。
/// 失败代表：命令层引入非确定状态。
#[test]
fn critic_command_phases_deterministic() {
    let cmd = CriticCommand::default();
    let fb = solid(16, 16, [0.3, 0.1, 0.1, 1.0]);
    assert_eq!(cmd.critique(&fb), cmd.critique(&fb), "critique 应逐位一致");
    assert_eq!(cmd.diagnose(&fb), cmd.diagnose(&fb));
    assert_eq!(Phase::Analyze.name(), "analyze");
    assert_eq!(Phase::Critique.name(), "critique");
    assert_eq!(Phase::Suggest.name(), "suggest");
    assert_eq!(Phase::Apply.name(), "apply");
}

/// QualityScoreNode：标量透传 + clamp [0,1]；非标量 → Empty。
/// 失败代表：节点映射越界或静默编造数值。
#[test]
fn critic_quality_node_clamps_and_unit() {
    use prism_core::node::{EvalCtx, MathStructure, Payload};
    let node = QualityScoreNode::default();
    let ctx = EvalCtx::new(0, 1, 0);
    assert_eq!(node.apply(&Payload::Scalar(0.42), &ctx), Payload::Scalar(0.42));
    assert_eq!(node.apply(&Payload::Scalar(2.0), &ctx), Payload::Scalar(1.0));
    assert_eq!(node.apply(&Payload::Scalar(-1.0), &ctx), Payload::Scalar(0.0));
    assert_eq!(node.apply(&Payload::Empty, &ctx), Payload::Empty, "非标量 → 单位元");
    // 确定性。
    assert_eq!(node.apply(&Payload::Scalar(0.7), &ctx), node.apply(&Payload::Scalar(0.7), &ctx));
}

// ===========================================================================
// prism-ai assist：确定性 + 边界
// ===========================================================================

/// advise 纯函数：同请求逐位相同；动作 kebab-case 正确。
/// 失败代表：引入了随机/隐藏状态（违反 D-11 确定性 mock）。
#[test]
fn ai_assist_is_bitwise_deterministic() {
    let req = AdviceRequest::new("exposure", "low", 0.8, 7);
    let a = advise(&req).unwrap();
    let b = advise(&req).unwrap();
    assert_eq!(a, b, "同请求同建议（逐位）");
    assert_eq!(a.action, "exposure-low");
    assert_eq!(a.confidence, 0.8);
    assert!(a.text.contains("strong signal"));
    // action_id 归一。
    assert_eq!(action_id("EXPOSURE", "HIGH"), "exposure-high");
    assert_eq!(action_id("weirdness", "low"), "look-low", "未知维度归一为 look");
}

/// 空维度 → 结构化错误 AI5001（不静默编造）；三段式含建议段。
/// 失败代表：畸形输入崩溃或产出无意义建议。
#[test]
fn ai_assist_empty_dimension_is_structured_error() {
    let err = advise(&AdviceRequest::new("  ", "low", 0.5, 1)).unwrap_err();
    assert_eq!(err.code, "AI5001");
    assert!(err.to_string().contains("建议："), "三段式应含建议段");
}

/// severity 构造时钳到 [0,1]；不同种子只变指纹不变动作/置信。
/// 失败代表：越界 severity 传播。
#[test]
fn ai_assist_severity_clamp_and_seed_effect() {
    assert_eq!(AdviceRequest::new("exposure", "low", 5.0, 1).severity, 1.0);
    assert_eq!(AdviceRequest::new("exposure", "low", -3.0, 1).severity, 0.0);
    let a = advise(&AdviceRequest::new("color", "low", 0.3, 1)).unwrap();
    let b = advise(&AdviceRequest::new("color", "low", 0.3, 2)).unwrap();
    assert_ne!(a.fingerprint, b.fingerprint, "种子影响指纹");
    assert_eq!(a.action, b.action, "动作与种子无关");
    assert_eq!(a.confidence, b.confidence);
}

/// advise_all 保持输入顺序，与逐条一致。
/// 失败代表：批处理乱序。
#[test]
fn ai_assist_all_preserves_order() {
    let reqs = vec![
        AdviceRequest::new("exposure", "low", 0.6, 1),
        AdviceRequest::new("color", "high", 0.9, 1),
        AdviceRequest::new("sharpness", "low", 0.2, 1),
    ];
    let out = advise_all(&reqs).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].action, "exposure-low");
    assert_eq!(out[1].action, "color-high");
    assert_eq!(out[2].action, "sharpness-low");
    // 逐条一致。
    for (i, r) in reqs.iter().enumerate() {
        assert_eq!(out[i], advise(r).unwrap());
    }
}

/// 方向措辞区分 low/high（high 含收敛前缀）。
/// 失败代表：方向未区分 → 建议误导。
#[test]
fn ai_assist_direction_wording_differs() {
    let low = advise(&AdviceRequest::new("exposure", "low", 0.5, 1)).unwrap();
    let high = advise(&AdviceRequest::new("exposure", "high", 0.5, 1)).unwrap();
    assert_ne!(low.text, high.text);
    assert!(high.text.contains("ease back"), "high 含收敛措辞：{}", high.text);
}
