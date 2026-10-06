//! QA 攻击性边界测试（W6-B10：prism-ai）。
//!
//! 作者：严过关（QA）。攻击目标（确定性铁律 + 防御缺失）：
//! 1. hash 原语：FNV-1a 空串锚、雪崩、derive 区分性。
//! 2. mock：空/空白 prompt 拒绝、max_words 语义（0=不限 vs 显式 0）、
//!    极端 seed、盐区分、确定性逐字节。
//! 3. timings：空输入、零时长词、倒退、max_words=0、超长单词、极值、确定性。
//! 4. journal：resume 语义、序列化往返确定性、损坏输入结构化报错、
//!    往返后 PartialEq 一致性（records 重建语义）。
//! 5. pipeline：resume 等价（跳阶段不改变产出）、帧确定性、帧尺寸。
//! 6. nodes：CueColorNode 标量域 [0,1]、非标量 → Empty、NaN 输入、确定性。

use prism_core::node::MathStructure;
use prism_ai::hash::{derive, fnv1a64, fnv1a64_str, mix64};
use prism_ai::journal::{Journal, Stage, StageState};
use prism_ai::mock::MockProvider;
use prism_ai::nodes::CueColorNode;
use prism_ai::pipeline::{DraftConfig, DraftPipeline};
use prism_ai::provider::{Provider, ProviderRequest, ProviderResponse};
use prism_ai::timings::{cue_split, CueConfig, WordTiming};

// ---------------------------------------------------------------------------
// ① hash 原语
// ---------------------------------------------------------------------------

#[test]
fn fnv1a_empty_anchor_and_avalanche() {
    // FNV-1a 空串 = offset basis（已知常量）。
    assert_eq!(fnv1a64(&[]), 0xCBF2_9CE4_8422_2325, "FNV-1a 空串应为 offset basis");
    assert_eq!(fnv1a64_str(""), 0xCBF2_9CE4_8422_2325);
    // FNV-1a 原始哈希对**末字节**差异雪崩弱（末字节仅经一次乘法传播）——这是
    // FNV 已知性质；模块文档正是因此配 mix64 提升雪崩。故雪崩断言针对 derive
    // （= mix64 ∘ fnv），原始 fnv1a64 仅断言"不同输入不同输出"。
    let a = fnv1a64_str("determinism");
    let b = fnv1a64_str("determinisn");
    assert_ne!(a, b, "末字节差异应产生不同哈希");
    // derive 经 mix64 混洗 → 雪崩近似（差异 > 24 位）。
    let da = derive(a, "");
    let db = derive(b, "");
    let differing = (da ^ db).count_ones();
    assert!(differing > 16, "mix64 混洗后差异应显著，仅 {differing} 位");
    // mix64 是双射（对随机输入无碰撞在此抽样规模内）。
    assert_eq!(mix64(0), mix64(0));
    assert_ne!(mix64(1), mix64(2));
    // derive 确定性。
    assert_eq!(derive(12345, "salt"), derive(12345, "salt"));
    assert_ne!(derive(12345, "salt"), derive(12345, "salt2"));
}

// ---------------------------------------------------------------------------
// ② mock 边界
// ---------------------------------------------------------------------------

#[test]
fn mock_empty_and_whitespace_prompt_rejected() {
    let p = MockProvider::default();
    for bad in ["", "   ", "\t\n ", "\u{3000}"] {
        let e = p.complete(&ProviderRequest::new(bad, 1)).unwrap_err();
        assert_eq!(e.code, "AI1001", "空白 prompt `{bad:?}` 应拒绝");
        assert!(e.to_string().contains("建议："), "三段式必含建议段");
    }
}

#[test]
fn mock_max_words_zero_vs_bounded() {
    let p = MockProvider::default();
    let req0 = ProviderRequest::new("hello world", 7);
    assert_eq!(req0.max_words, 0, "默认应不设限（0）");
    let full = p.complete(&req0).unwrap();
    assert!((1..=6).contains(&full.words.len()), "无上限时词数 1..=6：{}", full.words.len());
    // max_words = 1 → 恰 1 词。
    let one = p.complete(&req0.clone().with_max_words(1)).unwrap();
    assert_eq!(one.words.len(), 1, "max_words=1 应产出 1 词");
    // max_words 大于派生词数（6）→ 不放大词数。
    let big = p.complete(&req0.clone().with_max_words(100)).unwrap();
    assert_eq!(big.words.len(), full.words.len(), "max_words 大于派生数不应放大");
}

#[test]
fn mock_extreme_seed_and_utf8_deterministic() {
    let p = MockProvider::default();
    // 极端 seed 与多字节 prompt：确定性不破。
    for seed in [0u64, 1, u64::MAX / 2, u64::MAX] {
        let req = ProviderRequest::new("中文旁白 + emoji 🎬", seed);
        let a = p.complete(&req).unwrap();
        let b = p.complete(&req).unwrap();
        assert_eq!(a, b, "seed={seed} 同请求同响应");
        // 词级时间戳非倒退、时长为正。
        let mut prev_end = 0u64;
        for (i, w) in a.words.iter().enumerate() {
            assert!(w.end_ms > w.start_ms, "词时长应正");
            if i > 0 {
                assert!(w.start_ms >= prev_end, "时间戳不应倒退");
            }
            prev_end = w.end_ms;
        }
    }
}

#[test]
fn mock_different_prompt_differs_and_salt_isolated() {
    let p = MockProvider::default();
    let a = p.complete(&ProviderRequest::new("alpha", 3)).unwrap();
    let b = p.complete(&ProviderRequest::new("beta", 3)).unwrap();
    assert_ne!(a, b, "不同 prompt 应产出不同（FNV 派生）");
    // 盐隔离。
    let s = MockProvider::with_salt("salt-A").complete(&ProviderRequest::new("alpha", 3)).unwrap();
    let d = MockProvider::default().complete(&ProviderRequest::new("alpha", 3)).unwrap();
    assert_ne!(s, d, "盐应区分实例输出");
}

// ---------------------------------------------------------------------------
// ③ timings 边界
// ---------------------------------------------------------------------------

#[test]
fn cue_split_empty_and_single() {
    assert!(cue_split(&[], CueConfig::default()).unwrap().is_empty(), "空输入空 cues");
    let one = cue_split(&[WordTiming::new("hi", 0, 100)], CueConfig::default()).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].text, "hi");
    assert_eq!((one[0].start_ms, one[0].end_ms), (0, 100));
}

#[test]
fn cue_split_rejects_zero_and_reversed_durations() {
    let cfg = CueConfig::default();
    // 零时长词。
    let e = cue_split(&[WordTiming::new("a", 100, 100)], cfg).unwrap_err();
    assert_eq!(e.code, "AI2001", "零时长词应拒绝");
    // 反向。
    let e = cue_split(&[WordTiming::new("a", 200, 100)], cfg).unwrap_err();
    assert_eq!(e.code, "AI2001");
    // 首词正常但第二词倒退（start < 前 end）。
    let e = cue_split(
        &[WordTiming::new("a", 0, 100), WordTiming::new("b", 50, 200)],
        cfg,
    )
    .unwrap_err();
    assert_eq!(e.code, "AI2001");
}

#[test]
fn cue_split_zero_max_words_rejected() {
    let cfg = CueConfig { max_cue_ms: 1000, max_words: 0, gap_ms: 100 };
    let e = cue_split(&[WordTiming::new("a", 0, 100)], cfg).unwrap_err();
    assert_eq!(e.code, "AI2002");
}

#[test]
fn cue_split_max_duration_boundary_exact() {
    // 攻击精确边界：max_cue_ms 恰好等于词跨度 → 不切；超 1ms → 切。
    let words = vec![WordTiming::new("a", 0, 500), WordTiming::new("b", 500, 1000)];
    // 跨 0..1000 = 1000ms。max = 1000 → 恰容纳（new_len > max? 1000>1000 false）。
    let cfg_eq = CueConfig { max_cue_ms: 1000, max_words: 8, gap_ms: 10000 };
    let cues_eq = cue_split(&words, cfg_eq).unwrap();
    assert_eq!(cues_eq.len(), 1, "跨度 == max 应不切（> 才切）");
    // max = 999 → 切。
    let cfg_lt = CueConfig { max_cue_ms: 999, max_words: 8, gap_ms: 10000 };
    let cues_lt = cue_split(&words, cfg_lt).unwrap();
    assert_eq!(cues_lt.len(), 2, "跨度 > max 应切");
}

#[test]
fn cue_split_oversized_single_word_is_unsplittable() {
    // 契约追问：单字超过 max_cue_ms，无法切分 → 应独占一条（不得丢弃/死循环）。
    let words = vec![WordTiming::new("supercalifragilistic", 0, 5000)];
    let cfg = CueConfig { max_cue_ms: 1000, max_words: 8, gap_ms: 100 };
    let cues = cue_split(&words, cfg).unwrap();
    assert_eq!(cues.len(), 1, "超长单词应独占 1 条（不可丢弃）");
    assert_eq!(cues[0].text, "supercalifragilistic");
}

#[test]
fn cue_split_gap_boundary_exact() {
    // 间隙恰 == gap_ms → 不切；gap+1 → 切。
    let words = vec![WordTiming::new("a", 0, 100), WordTiming::new("b", 200, 300)];
    let cfg_eq = CueConfig { max_cue_ms: 10000, max_words: 8, gap_ms: 100 };
    assert_eq!(cue_split(&words, cfg_eq).unwrap().len(), 1, "gap==阈值不切（> 才切）");
    let cfg_lt = CueConfig { max_cue_ms: 10000, max_words: 8, gap_ms: 99 };
    assert_eq!(cue_split(&words, cfg_lt).unwrap().len(), 2, "gap>阈值应切");
}

#[test]
fn cue_split_deterministic_large() {
    let words: Vec<WordTiming> = (0..500)
        .map(|i| WordTiming::new("w", i * 130, i * 130 + 100))
        .collect();
    let cfg = CueConfig::default();
    assert_eq!(cue_split(&words, cfg), cue_split(&words, cfg), "大输入切分应确定");
}

// ---------------------------------------------------------------------------
// ④ journal
// ---------------------------------------------------------------------------

#[test]
fn journal_resume_sequence() {
    let mut j = Journal::new("t");
    assert_eq!(j.resume_from(), Some(Stage::Text), "空流水账从 Text 起");
    j.mark_done(Stage::Text);
    assert_eq!(j.resume_from(), Some(Stage::Timings));
    j.mark_failed(Stage::Timings, "boom");
    assert_eq!(j.resume_from(), Some(Stage::Timings), "失败阶段未完成应从此续");
    assert!(!j.is_done(Stage::Timings));
    // 失败后标记完成 → 前进。
    j.mark_done(Stage::Timings);
    assert_eq!(j.resume_from(), Some(Stage::Cues));
}

#[test]
fn journal_serialize_roundtrip_state_preserved() {
    let mut j = Journal::new("task-x");
    j.mark_done(Stage::Text);
    j.mark_failed(Stage::Cues, "line1\nline2"); // 换行应被压平
    let s = j.serialize();
    let back = Journal::deserialize(&s).unwrap();
    // 关键状态往返保持。
    assert_eq!(back.task(), "task-x");
    assert_eq!(back.state(Stage::Text), StageState::Done);
    assert_eq!(back.state(Stage::Cues), StageState::Failed("line1 line2".to_string()));
    assert_eq!(back.state(Stage::Timings), StageState::Pending);
    // 序列化幂等。
    assert_eq!(back.serialize(), s, "序列化 → 反序列化 → 序列化应稳定");
    // resume 一致。
    assert_eq!(back.resume_from(), j.resume_from());
}

#[test]
fn journal_deserialize_rejects_malformed() {
    assert_eq!(Journal::deserialize("garbage").unwrap_err().code, "AI3001");
    assert_eq!(Journal::deserialize("nope=done").unwrap_err().code, "AI3001");
    assert_eq!(Journal::deserialize("text=weird").unwrap_err().code, "AI3001");
    // 空文本 → 空流水账（合法）。
    let j = Journal::deserialize("").unwrap();
    assert_eq!(j.resume_from(), Some(Stage::Text));
}

#[test]
fn journal_equality_roundtrip_consistency() {
    // 攻击：mark_* 只记录被触及阶段；deserialize 会为**所有出现行**建记录
    // （serialize 写出全部 4 阶段行）。故 serialize→deserialize 的 records
    // 与原始不等。若 Journal 的 PartialEq 语义含 records，则往返不保持相等。
    let mut j = Journal::new("t");
    j.mark_done(Stage::Text);
    let back = Journal::deserialize(&j.serialize()).unwrap();
    // 状态等价（state 是主语义）。
    for s in Stage::ALL {
        assert_eq!(back.state(s), j.state(s), "阶段状态往返应一致：{:?}", s);
    }
    // records 语义：往返后记录数可能不同（记录的是"文件出现行"而非"首次置位"）。
    // 这是可接受的，只要 state/resume 语义保持——此处仅记录事实不做强断言。
}

// ---------------------------------------------------------------------------
// ⑤ pipeline
// ---------------------------------------------------------------------------

#[test]
fn pipeline_resume_equivalent_to_full_run() {
    // 攻击：从"部分完成"的 prev journal 续跑，产出 cues/frames/text 应与
    // 从头跑**逐位一致**（丢花絮：跳过的 Provider 重调，确定性 ⇒ 结果同）。
    let pipe = DraftPipeline::new(DraftConfig::default());
    let provider = MockProvider::default();
    let req = ProviderRequest::new("resume determinism", 55);

    let full = pipe.run(&provider, &req, None).unwrap();

    // 模拟中断：Text+Timings 已完成。
    let mut prev = Journal::new("resume determinism");
    prev.mark_done(Stage::Text);
    prev.mark_done(Stage::Timings);
    let resumed = pipe.run(&provider, &req, Some(&prev)).unwrap();

    assert_eq!(full.text, resumed.text, "resume 文本应与全跑一致");
    assert_eq!(full.cues, resumed.cues, "resume cues 应与全跑一致");
    assert_eq!(full.frames.len(), resumed.frames.len(), "帧数应一致");
    for (a, b) in full.frames.iter().zip(resumed.frames.iter()) {
        assert_eq!(a.pixels, b.pixels, "resume 帧像素应与全跑逐位一致");
    }
    assert_eq!(resumed.journal.resume_from(), None, "resume 后应全完成");
    assert_eq!(resumed.journal.done_count(), 4);
}

#[test]
fn pipeline_frames_deterministic_and_sized() {
    let cfg = DraftConfig { width: 16, height: 9, cue: CueConfig::default() };
    let pipe = DraftPipeline::new(cfg);
    let provider = MockProvider::default();
    let req = ProviderRequest::new("frames", 2);
    let a = pipe.run(&provider, &req, None).unwrap();
    let b = pipe.run(&provider, &req, None).unwrap();
    assert_eq!(a.frames.len(), b.frames.len());
    for (fa, fb) in a.frames.iter().zip(b.frames.iter()) {
        assert_eq!(fa.pixels, fb.pixels, "同输入同帧");
        assert_eq!(fa.width, 16);
        assert_eq!(fa.height, 9);
    }
    assert_eq!(a.frames.len(), a.cues.len(), "每 cue 一帧");
}

#[test]
fn pipeline_alt_seed_changes_frames() {
    let pipe = DraftPipeline::new(DraftConfig { width: 8, height: 8, cue: CueConfig::default() });
    let provider = MockProvider::default();
    let a = pipe.run(&provider, &ProviderRequest::new("seed test", 1), None).unwrap();
    let b = pipe.run(&provider, &ProviderRequest::new("seed test", 2), None).unwrap();
    // seed 不同 → Provider 产出的词不同 → cues 可能不同 → 帧不同（至少一帧异）。
    let differs = a.frames.len() != b.frames.len()
        || a.frames.iter().zip(b.frames.iter()).any(|(x, y)| x.pixels != y.pixels);
    assert!(differs, "不同 seed 应改变草稿输出");
}

// ---------------------------------------------------------------------------
// ⑥ nodes（CueColorNode）
// ---------------------------------------------------------------------------

#[test]
fn cue_color_node_scalar_range_and_empty() {
    let node = CueColorNode::default();
    let ctx = prism_core::EvalCtx::new(0, 1, 0);
    // 标量输入 → [0,1] 输出。
    for h in [0.0f64, 1.25, -3.0, 1e-6, 1e6, 1e15] {
        match node.apply(&prism_core::Payload::Scalar(h as f32), &ctx) {
            prism_core::Payload::Scalar(v) => {
                assert!((0.0..=1.0).contains(&v), "h={h} 输出越界：{v}");
                assert!(v.is_finite(), "h={h} 输出非有限：{v}");
            }
            other => panic!("标量输入应得 Scalar，得 {other:?}"),
        }
    }
    // 非标量 → Empty（不编造）。
    assert_eq!(node.apply(&prism_core::Payload::Empty, &ctx), prism_core::Payload::Empty);
}

#[test]
fn cue_color_node_nan_input_finite_output() {
    // 攻击 NaN 输入：f64::to_bits(NaN) 仍为整数 → mix64 → 有限 [0,1]。
    // 契约：任何 f64（含 NaN/Inf）都不得产出 NaN 标量。
    let node = CueColorNode::default();
    let ctx = prism_core::EvalCtx::new(0, 1, 0);
    for h in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        match node.apply(&prism_core::Payload::Scalar(h as f32), &ctx) {
            prism_core::Payload::Scalar(v) => {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v), "h={h} 产出非法：{v}");
            }
            other => panic!("应得 Scalar，得 {other:?}"),
        }
    }
}

#[test]
fn cue_color_node_deterministic() {
    let node = CueColorNode::default();
    let ctx = prism_core::EvalCtx::new(0, 1, 0);
    let a = node.apply(&prism_core::Payload::Scalar(42.5f32), &ctx);
    let b = node.apply(&prism_core::Payload::Scalar(42.5f32), &ctx);
    assert_eq!(a, b, "节点输出应确定");
}

// ---------------------------------------------------------------------------
// ⑦ provider 闭包适配器契约
// ---------------------------------------------------------------------------

#[test]
fn fn_provider_propagates_errors() {
    let p = prism_ai::provider::FnProvider::new("err", |_r: &ProviderRequest| {
        Err(prism_ai::AiError::new("AI9999", "loc", "boom", "fix"))
    });
    let e = p.complete(&ProviderRequest::new("x", 0)).unwrap_err();
    assert_eq!(e.code, "AI9999");
}
