//! V3.0 W3 语言层出口判据（§3.5 V1/V2/V4/V5/V6 + §5.4 错误注入集）。
//!
//! 靶子 = **声明一致性 + 实现合规**，不是主观审美：
//! - V1 档位忠实：`analytic` 零注入；`crafted` 每族静态检出 + 帧值对拍；
//! - V2 五个负控全 FAIL 且带行号（与选哪档无关）；
//! - V4 匀质同样是能力：`analytic` 不因「不够搓」被判失败、无全局加噪；
//! - V5 同源对拍：语言声明 ≡ 直接调引擎 API 构造同一观感配置，逐字节；
//! - V6 确定性 + 零损失：两跑逐字节、3 个 `.prism` md5、旧入口等价。
//!
//! 静态注入断言读实现源码（`include_str!`），帧值对拍走渲染前唯一注入函数
//! [`prism_dsl::inject_frame_values`]（与 `wiring.rs::resolve_frame` 同一函数）。

use prism_dsl::{
    assert_frame_value_pairs, compile_auto, compile_v2, compile_v3, dim_noise,
    example_crafted_finish, finish_block_text, inject_frame_values, mix_artifact_hash,
    posthoc_gate, scan_finish, signals_of, trace_event, true_random_gate, AutoCompile,
    CompileV3Output, FinishMode, FinishPlan, FinishScan, FrameSignals, TraceDim, TraceFamily,
    WiringFrame, WiringPlan, FORBIDDEN_POSTHOC_TOKENS, FORBIDDEN_RANDOM_TOKENS,
    POSTHOC_BLOCK_MARKERS, RUBRIC_DIMS, STRENGTH_MAX, W3_INJECTION_KEYS,
};


// ---------------------------------------------------------------------------
// 样例：与 W1 同源对拍同构的最小 v3 场景（32×24、1s、2fps = 2 帧）
// ---------------------------------------------------------------------------

const SCENE_EMIT: &str = r#"
r = 0.75
emit f"""
movie "g" { duration 1s fps 2 resolution 32x24 format png }
scene "s" {
    duration 1s
    stage3d {
        camera main { orbit { center 0,0,0 radius 4.5 } fov 55 }
        sdf ball { kind sphere radius {{r}} }
        material hero { albedo #c0392b roughness 0.4 metalness 0.1 }
        light key { direction(0.4,-1,-0.6) color #ffffff intensity 3.2 }
    }
}
output { resolution 32x24 format png }
"""
"#;

/// 六族维度声明（只有 motion 带呼吸 + 其余四族启用，geometry 绝对匀质）。
const CRAFTED_DIMS: &str = r#"finish = crafted {
    motion enabled true strength 620
    geometry enabled false strength 0
    material_light enabled true strength 400
    time_camera enabled true strength 300
    composition_color enabled true strength 250
    edit_audio enabled true strength 500
}"#;

fn v3_with_finish(finish: &str) -> String {
    format!("// prism-dsl 2\nseed = 42\n{finish}\n{SCENE_EMIT}")
}

const ANALYTIC: &str = "// prism-dsl 2\nseed = 42\nfinish = analytic\n";

fn analytic_source() -> String {
    format!("{ANALYTIC}{SCENE_EMIT}")
}

fn crafted_source() -> String {
    v3_with_finish(CRAFTED_DIMS)
}

fn ok(src: &str) -> CompileV3Output {
    compile_v3(src).unwrap_or_else(|e| panic!("应编译成功，却在第 {} 行失败：{}", e.line, e.message))
}

/// 与 `CRAFTED_DIMS` 语义相同的「直接调引擎 API」观感配置（V5 的另一侧）。
fn direct_finish_plan() -> FinishPlan {
    let mut plan: FinishPlan = FinishPlan::analytic(42, 0);
    plan.mode = FinishMode::Crafted;
    plan.dims = [
        TraceDim { enabled: true, strength: 620, line: 0 },
        TraceDim { enabled: false, strength: 0, line: 0 },
        TraceDim { enabled: true, strength: 400, line: 0 },
        TraceDim { enabled: true, strength: 300, line: 0 },
        TraceDim { enabled: true, strength: 250, line: 0 },
        TraceDim { enabled: true, strength: 500, line: 0 },
    ];
    plan
}

fn expect_fail(src: &str, code: &str) -> prism_dsl::DslError {
    let err = compile_v3(src).err().unwrap_or_else(|| {
        panic!("负控样例必须编译期判红（{code}），实际编译成功：\n{src}")
    });
    assert!(
        err.message.contains(code),
        "错误码应为 {code}，实际第 {} 行：{}",
        err.line,
        err.message
    );
    assert!(err.line >= 1, "错误必须带行号（{code}）");
    assert!(err.message.contains("建议："), "{code} 必须是三段式（含建议）");
    err
}

// ---------------------------------------------------------------------------
// V1 档位忠实：analytic 零噪点；crafted 逐族静态检出 + 帧值对拍
// ---------------------------------------------------------------------------

#[test]
fn v1_analytic_is_byte_identical_baseline_and_has_zero_unexpected_noise() {
    let a = ok(&analytic_source());
    // analytic 不注入任何 v2 声明块 ⇒ 降级文本与 W1 完全一致、wiring 恒空。
    assert!(!a.lowered_source.contains("finish {"));
    assert!(a.base.wiring.is_empty(), "analytic 档 wiring 必须为空（零损失）");
    assert!(a.base.wiring.finish.is_none(), "analytic 不挂 crafted 声明块");
    // 逐帧 0..8 全部零意外噪点。
    for frame in 0..8u64 {
        let mut wf: WiringFrame = WiringFrame::default();
        let plan: FinishPlan = FinishPlan::analytic(42, 2);
        inject_frame_values(&mut wf, &plan, frame, 42, 0);
        assert!(wf.values.is_empty(), "第 {frame} 帧出现意外噪点");
        assert!(!signals_of(&wf).crafted, "第 {frame} 帧误开 crafted");
        assert_frame_value_pairs(&wf, &plan, frame, 42, 0)
            .unwrap_or_else(|e| panic!("第 {frame} 帧零噪点对拍失败：{e}"));
    }
}

#[test]
fn v1_crafted_every_declared_dimension_is_statically_detectable_and_pairs_match() {
    let out = ok(&crafted_source());
    let plan: &FinishPlan = out
        .base
        .wiring
        .finish_plan()
        .expect("crafted 档必须把 FinishPlan 挂进 WiringPlan");
    assert!(plan.is_crafted());
    // 逐族：静态检出（编译产物里该族 active）+ 帧值对拍（逐帧逐位）。
    let active = plan.active_families();
    assert_eq!(active.len(), 5, "geometry 绝对匀质 ⇒ 恰好 5 族启用");
    assert!(!active.contains(&TraceFamily::Geometry), "geometry 必须匀质");
    for frame in 0..8u64 {
        let mut wf: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut wf, plan, frame, 42, 0);
        let sig: FrameSignals = signals_of(&wf);
        assert!(sig.crafted, "第 {frame} 帧应为 crafted");
        for fam in TraceFamily::ALL {
            let key: String = format!("w3.{}.signal", fam.tag());
            assert!(
                wf.values.contains_key(&key) == active.contains(&fam),
                "第 {frame} 帧 {} 族的注入键与声明不一致",
                fam.tag()
            );
        }
        assert_frame_value_pairs(&wf, plan, frame, 42, 0)
            .unwrap_or_else(|e| panic!("第 {frame} 帧帧值对拍失败：{e}"));
        // 每族 signal 与声明强度、种子化噪声逐位一致。
        for fam in &active {
            let want: f32 = (dim_noise(plan, *fam, frame, 42, 0) * 2.0 - 1.0)
                * (plan.dim(*fam).strength as f32 / STRENGTH_MAX as f32);
            assert_eq!(
                sig.of(*fam),
                want,
                "第 {frame} 帧 {} 族 signal 与注入量不一致",
                fam.tag()
            );
        }
    }
}

#[test]
fn v1_finish_is_first_class_in_artifact_hash_and_trace() {
    let a = ok(&analytic_source());
    let c = ok(&crafted_source());
    assert_ne!(a.artifact_hash, c.artifact_hash, "换档必须进产物哈希");
    // 换强度 / 换种子 / 换 rubric 全部可检出（V1 / V2⑤ 的执行体）。
    let d = ok(&crafted_source().replace("strength 620", "strength 300"));
    assert_ne!(c.artifact_hash, d.artifact_hash, "换维度强度必须可检出");
    let s = ok(&crafted_source().replace("seed = 42", "seed = 43"));
    assert_ne!(c.artifact_hash, s.artifact_hash, "换种子必须可检出");
    let r = ok(&crafted_source().replace(
        "    edit_audio enabled true strength 500",
        "    edit_audio enabled true strength 500\n    rubric exposure 0.18 0.6 2.4\n    rubric contrast 0.18 0.08 0.32\n    rubric color 0.35 0.12 0.45\n    rubric sharpness 0.02 0.01 0.05\n    rubric composition 0.5 0.2 0.5\n    rubric cast 0.12",
    ));
    assert_ne!(c.artifact_hash, r.artifact_hash, "换 rubric 必须可检出");
    // F-5 求值轨迹首事件是一等字段声明（先于语句求值）。
    assert_eq!(a.trace[0], trace_event(&FinishPlan::analytic(42, 3)));
    assert_eq!(
        c.trace[0].value_bits,
        trace_event(c.base.wiring.finish_plan().unwrap()).value_bits
    );
    // `example_crafted_finish()` 是文档 / 测试夹具的单一来源，本身合法可编译。
    let example: String = format!("// prism-dsl 2\nseed = 42\n{}\n{SCENE_EMIT}", example_crafted_finish());
    let ex = ok(&example);
    assert_eq!(
        ex.base.wiring.finish_plan().unwrap().canonical_bytes(),
        c.base.wiring.finish_plan().unwrap().canonical_bytes(),
        "示例声明语义须与 crafted 样例一致"
    );
    // F-8 扩链：`mix_artifact_hash` 对同一 plan 幂等、对不同 plan 可区分。
    let p: FinishPlan = FinishPlan::analytic(42, 3);
    assert_eq!(mix_artifact_hash(7, &p), mix_artifact_hash(7, &p));
    assert_ne!(mix_artifact_hash(7, &p), mix_artifact_hash(8, &p));
}

#[test]
fn v1_dimension_zero_is_a_legal_first_class_choice() {
    // 「只有运动带呼吸、几何绝对匀质」必须能表达，且强度 0 = 该族恒等 0。
    let src = v3_with_finish(
        r#"finish = crafted {
    motion enabled true strength 620
    geometry enabled false strength 0
    material_light enabled false strength 0
    time_camera enabled false strength 0
    composition_color enabled false strength 0
    edit_audio enabled false strength 0
}"#,
    );
    let out = ok(&src);
    let plan: &FinishPlan = out.base.wiring.finish_plan().unwrap();
    assert_eq!(plan.active_families(), vec![TraceFamily::Motion]);
    let mut wf: WiringFrame = WiringFrame::default();
    inject_frame_values(&mut wf, plan, 3, 42, 0);
    assert!(wf.values.contains_key("w3.motion.signal"));
    assert!(!wf.values.contains_key("w3.geometry.signal"));
    assert_eq!(signals_of(&wf).geometry, 0.0, "绝对匀质族恒等 0");
}

// ---------------------------------------------------------------------------
// V2 五个负控全 FAIL 且带行号（与用户选哪档无关）
// ---------------------------------------------------------------------------

#[test]
fn v2_neg1_posthoc_skin_is_rejected_even_on_analytic() {
    // 标准负控样本：`post { grain }` 帧后贴皮节点。全文静态门禁，
    // `analytic` 档同样判红（§3.3 禁令 1，不因选匀质而失效）。
    let src: &str = "// prism-dsl 2\nseed = 42\nfinish = analytic\npost { grain 0.3 }\n";
    let err = posthoc_gate(src).expect_err("帧后贴皮必须静态判红");
    assert!(err.message.contains("P9011"));
    assert_eq!(err.line, 4, "必须带行号");
    assert!(err.message.contains("建议："));
    // 另一路：`postfx` 块标记同样判红。
    let err2 = posthoc_gate("// prism-dsl 2\nseed = 1\nfinish = crafted { }\npostfx film\n")
        .expect_err("postfx 标记必须判红");
    assert!(err2.message.contains("P9011"));
    // 块标记常量是门禁自身的唯一事实源，不许被绕过。
    assert_eq!(POSTHOC_BLOCK_MARKERS.len(), 3);
    assert!(FORBIDDEN_POSTHOC_TOKENS.contains(&"grain"));
}

#[test]
fn v2_neg2_true_random_is_rejected_even_on_analytic() {
    // 真随机 / 系统时间 / IO 随机源 ⇒ 静态判红（§3.3 禁令 2）。
    for token in FORBIDDEN_RANDOM_TOKENS {
        // `analytic` 档同样判红：全文门禁与用户选哪档无关。
        let src = format!("// prism-dsl 2\nseed = 42\nfinish = analytic\nlet jitter = {token}\n");
        let bad = true_random_gate(&src)
            .expect_err(&format!("token '{token}' 必须静态判红"));
        assert!(bad.message.contains("P9012"));
        assert_eq!(bad.line, 4, "真随机门禁必须带行号");
        assert!(bad.message.contains("建议："));
    }
    // 两条门禁确实接在 `compile_v3` 编译链路上（实现级静态检出，不是只在测试里调）。
    let lib_src: &str = include_str!("../src/lib.rs");
    assert!(lib_src.contains("finish::posthoc_gate(source)?"));
    assert!(lib_src.contains("finish::true_random_gate(source)?"));
    // 两跑逐字节：种子化噪声同 (seed, frame, node, 族) 恒等，真随机不进入实现。
    let plan: FinishPlan = direct_finish_plan();
    let mut a: WiringFrame = WiringFrame::default();
    let mut b: WiringFrame = WiringFrame::default();
    inject_frame_values(&mut a, &plan, 11, 42, 0);
    inject_frame_values(&mut b, &plan, 11, 42, 0);
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    // 几何族走 pcg::hash2_f32，其余走 hash_mix 链；换帧必换值（不是常量伪装）。
    let g0 = dim_noise(&plan, TraceFamily::Geometry, 0, 42, 0);
    let g1 = dim_noise(&plan, TraceFamily::Geometry, 1, 42, 0);
    assert_ne!(g0, g1, "几何族必须是种子化逐帧噪声");
    assert_eq!(g0, dim_noise(&plan, TraceFamily::Geometry, 0, 42, 0));
}

#[test]
fn v2_neg3_crafted_all_zero_is_rejected_with_line() {
    let src = v3_with_finish(
        r#"finish = crafted {
    motion enabled false strength 0
    geometry enabled false strength 0
    material_light enabled false strength 0
    time_camera enabled false strength 0
    composition_color enabled false strength 0
    edit_audio enabled false strength 0
}"#,
    );
    let err = expect_fail(&src, "P9006");
    assert!(err.message.contains("全 0"));
}

#[test]
fn v2_neg4_missing_or_illegal_finish_is_three_part_error_with_line() {
    // ④ 漏填 finish：不替用户默认选档，P9001 + 行号 + 三段式。
    let missing = format!("// prism-dsl 2\nseed = 42\n{SCENE_EMIT}");
    let err = expect_fail(&missing, "P9001");
    assert_eq!(err.line, 1);
    assert!(err.message.contains("不替用户选档"));
    // ④ 非法档位：P9002。
    let illegal = v3_with_finish("finish = plastic");
    let err2 = expect_fail(&illegal, "P9002");
    assert_eq!(err2.line, 3);
    // ④ 缺 '='：P9002。
    expect_fail(&v3_with_finish("finish crafted { }"), "P9002");
}

#[test]
fn v2_neg5_field_not_wired_is_detected_by_frame_value_pairing() {
    // ⑤ 改某维度强度而帧不变（字段没接）⇒ 判红且带声明行号。
    let plan = direct_finish_plan();
    let mut wf: WiringFrame = WiringFrame::default();
    inject_frame_values(&mut wf, &plan, 6, 42, 0);
    // 正向：改强度后帧值必须变（证明字段真的接进求值链）。
    let mut weaker = plan.clone();
    weaker.dims[0] = TraceDim { enabled: true, strength: 300, line: 0 };
    let mut wf2: WiringFrame = WiringFrame::default();
    inject_frame_values(&mut wf2, &weaker, 6, 42, 0);
    assert_ne!(
        wf.value("w3.motion.signal"),
        wf2.value("w3.motion.signal"),
        "改强度而帧不变 ⇒ 字段没接"
    );
    // 负控：把某一族的注入键抹掉（等价于字段没接）⇒ 对拍必须报行号 + 族名。
    let mut unwired = wf.clone();
    unwired.values.remove("w3.motion.signal");
    let err = assert_frame_value_pairs(&unwired, &plan, 6, 42, 0)
        .expect_err("字段没接必须判红");
    assert!(err.contains("motion"), "报错须带族名：{err}");
    assert!(err.contains("行"), "报错须带行号：{err}");
    // 负控：用错强度声明去对拍实际帧值 ⇒ 判红。
    let err2 = assert_frame_value_pairs(&wf, &weaker, 6, 42, 0)
        .expect_err("声明强度与帧值不符必须判红");
    assert!(err2.contains("对拍不一致"), "报错须指出对拍不一致：{err2}");
}

// ---------------------------------------------------------------------------
// V4 匀质同样是能力：不因「不够搓」被判失败
// ---------------------------------------------------------------------------

#[test]
fn v4_analytic_is_a_first_class_success_not_a_failure() {
    let a = ok(&analytic_source());
    // 旧入口零损失：无头 / 版本 1 恒等 analytic（D-24 版本契约，非取默认）。
    let v1: &str = include_str!("../../../examples/w2d_showcase.prism");
    let t0: &str = include_str!("../../../examples/t0_media_mesh.prism");
    for src in [v1, t0] {
        assert!(matches!(compile_auto(src), Ok(AutoCompile::V2(_))));
        let direct = compile_v2(src).map(|o| format!("{o:?}")).map_err(|e| e.message);
        let auto = compile_auto(src)
            .map(|o| format!("{:?}", o.into_v2_output()))
            .map_err(|e| e.message);
        assert_eq!(direct, auto, "旧文件分派必须逐字段等价");
    }
    // analytic 不触发任何六族注入键，且不会被盲测判为失败（本档根本不跑盲测）。
    assert!(a.base.wiring.finish.is_none());
    let mut wf: WiringFrame = WiringFrame::default();
    inject_frame_values(&mut wf, &FinishPlan::analytic(42, 2), 0, 42, 0);
    for key in W3_INJECTION_KEYS {
        assert!(!wf.values.contains_key(key), "analytic 不得出现 {key}");
    }
}

// ---------------------------------------------------------------------------
// V5 同源对拍：语言声明 ≡ 直接调引擎 API 构造同一观感配置，逐字节
// ---------------------------------------------------------------------------

#[test]
fn v5_language_declaration_equals_direct_engine_api_byte_for_byte() {
    let out = ok(&crafted_source());
    let via_language: &FinishPlan = out.base.wiring.finish_plan().unwrap();
    let via_api: FinishPlan = direct_finish_plan();
    // 诊断行号不同（语言侧来自源码，API 侧为 0）⇒ 规范字节序列必须逐字节相同。
    assert_ne!(via_language.line, via_api.line);
    assert_eq!(
        via_language.canonical_bytes(),
        via_api.canonical_bytes(),
        "FinishPlan 规范字节序列必须逐字节相同（禁止两端同错）"
    );
    // WiringPlan 全量规范字节（含 0xF3 段）同样逐字节。
    let direct_wiring: WiringPlan = WiringPlan {
        entries: Vec::new(),
        finish: Some(via_api.clone()),
    };
    assert_eq!(out.base.wiring.canonical_bytes(), direct_wiring.canonical_bytes());
    // 展开文本独立编译 ≡ base（W1 同源对拍口径），且 finish 块可还原同一 plan。
    let lowered = compile_v2(&out.lowered_source).expect("展开文本应可独立编译");
    assert_eq!(format!("{lowered:?}"), format!("{:?}", out.base));
    assert_eq!(
        lowered.wiring.finish_plan().unwrap().canonical_bytes(),
        via_api.canonical_bytes()
    );
    // 两路逐帧帧值逐字节对拍（渲染前的真实注入量）。
    for frame in 0..8u64 {
        let mut a: WiringFrame = WiringFrame::default();
        let mut b: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut a, via_language, frame, 42, 0);
        inject_frame_values(&mut b, &via_api, frame, 42, 0);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes(), "第 {frame} 帧逐字节");
    }
    // `finish_block_text` 是降级文本的单一来源，与 v2 编译器互逆。
    // 注意：scan 阶段 seed 尚未回填（由 compile_v3 用一等字段回填），故此处
    // 只比对维度与档位；完整 plan 的逐字节对拍在上面已完成。
    let scan: FinishScan = scan_finish(&crafted_source()).expect("扫描应成功");
    let scanned = scan.decl.unwrap();
    assert_eq!(scanned.mode, via_api.mode);
    assert_eq!(scanned.dims.iter().map(|d| (d.enabled, d.strength)).collect::<Vec<_>>(),
               via_api.dims.iter().map(|d| (d.enabled, d.strength)).collect::<Vec<_>>());
    let block: String = finish_block_text(&via_api);
    assert!(block.starts_with("finish {\n"));
    assert!(block.contains("mode crafted"));
    assert!(block.contains("seed 42"));
    for fam in TraceFamily::ALL {
        assert!(block.contains(&format!("{}_enabled", fam.tag())));
        assert!(block.contains(&format!("{}_strength", fam.tag())));
    }
}

// ---------------------------------------------------------------------------
// V6 确定性 + 零损失：两跑逐字节、3 个 .prism md5、旧入口等价
// ---------------------------------------------------------------------------

#[test]
fn v6_two_runs_are_byte_identical_and_golden_files_are_unchanged() {
    for src in [analytic_source(), crafted_source()] {
        let a = ok(&src);
        let b = ok(&src);
        assert_eq!(a.artifact_hash, b.artifact_hash);
        assert_eq!(a.lowered_source.as_bytes(), b.lowered_source.as_bytes());
        assert_eq!(a.trace, b.trace);
        assert_eq!(a.base.wiring.canonical_bytes(), b.base.wiring.canonical_bytes());
    }
    // 3 个黄金 .prism md5 与 W1/W2 基线逐字节一致（复用 W1 的纯整数 md5）。
    let cases: &[(&[u8], &str)] = &[
        (
            include_bytes!("../../../examples/spin_red_ball.prism"),
            "e75b03e4bc7fcc090561eb1163f8ad24",
        ),
        (
            include_bytes!("../../../examples/t0_media_mesh.prism"),
            "46e29136de352ea75341823f2b69bf18",
        ),
        (
            include_bytes!("../../../examples/w2d_showcase.prism"),
            "88f735505eab09c6763b644197eee51b",
        ),
    ];
    for (bytes, want) in cases {
        assert_eq!(&md5::hex(bytes), want, "黄金 .prism 内容发生变化");
    }
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
}

// ---------------------------------------------------------------------------
// 错误注入集：P9001–P9012 逐码三段式 + 行号（§5.4「每波必跑」）
// ---------------------------------------------------------------------------

#[test]
fn error_injection_set_p9001_to_p9012_all_fail_with_line_and_three_parts() {
    let six_zero = r#"finish = crafted {
    motion enabled false strength 0
    geometry enabled false strength 0
    material_light enabled false strength 0
    time_camera enabled false strength 0
    composition_color enabled false strength 0
    edit_audio enabled false strength 0
}"#;
    let cases: Vec<(String, &str)> = vec![
        (format!("// prism-dsl 2\nseed = 42\n{SCENE_EMIT}"), "P9001"), // 漏填
        (v3_with_finish("finish = plastic"), "P9002"),                  // 非法档位
        (v3_with_finish("finish crafted { }"), "P9002"),                // 缺 '='
        (v3_with_finish("finish = crafted"), "P9003"),                  // crafted 无块
        (v3_with_finish("finish = crafted {\n    motion enabled true strength 1\n}"), "P9003"), // 漏族
        (v3_with_finish("finish = crafted {\n    foobar enabled true strength 1\n}"), "P9003"), // 未知族
        (v3_with_finish("finish = crafted {\n    motion enabled true strength 1"), "P9003"), // 缺 '}'
        (v3_with_finish("finish = crafted {\n    motion enabled yes strength 1\n}"), "P9004"),
        (v3_with_finish("finish = crafted {\n    motion enabled true strength abc\n}"), "P9004"),
        (v3_with_finish("finish = crafted {\n    motion enabled true strength 1001\n}"), "P9004"),
        (v3_with_finish("finish = crafted {\n    motion enabled false strength 5\n}"), "P9004"),
        (v3_with_finish("finish = crafted {\n    motion enabled true strength 1\n    motion enabled true strength 2\n}"), "P9005"),
        (v3_with_finish("finish = analytic\nfinish = crafted { }"), "P9005"), // 重复声明
        (v3_with_finish(six_zero), "P9006"),                                    // 全 0
        (v3_with_finish("finish = analytic {\n    motion enabled true strength 1\n}"), "P9007"),
    ];
    for (src, code) in &cases {
        expect_fail(src, code);
    }
    assert_eq!(cases.len(), 15, "语言层错误注入：P9001–P9007 全码 + 变体");
    // P9011 / P9012 是**编译期全文静态门禁**（§3.3 两条治本禁令）：必须直接
    // 对源码判红，与选哪档无关；不塞进 v3 语法树（否则先被词法拦截，掩盖门禁）。
    let posthoc = "// prism-dsl 2\nseed = 1\nfinish = analytic\npost { grain 0.3 }\n";
    let err = posthoc_gate(posthoc).expect_err("帧后贴皮必须静态判红");
    assert!(err.message.contains("P9011") && err.line == 4 && err.message.contains("建议："));
    let truerand = "// prism-dsl 2\nseed = 1\nfinish = crafted { }\nlet jitter = SystemTime::now()\n";
    let err = true_random_gate(truerand).expect_err("真随机必须静态判红");
    assert!(err.message.contains("P9012") && err.line == 4 && err.message.contains("建议："));
    // 实现级源码门禁：W3 注入链路里不得出现帧缓冲贴皮 / 真随机 API。
    let finish_src: &str = include_str!("../src/finish.rs");
    assert!(finish_src.contains("hash_mix_unit_f32") && finish_src.contains("hash2_f32"));
    assert!(!finish_src.contains("thread_rng("));
    assert!(!finish_src.contains("SystemTime::now("));
    // rubric 必须整组声明（不许补缺省）。
    let partial = v3_with_finish(
        "finish = crafted {\n    motion enabled true strength 1\n    geometry enabled false strength 0\n    material_light enabled false strength 0\n    time_camera enabled false strength 0\n    composition_color enabled false strength 0\n    edit_audio enabled false strength 0\n    rubric exposure 0.18 0.6 2.4\n}",
    );
    expect_fail(&partial, "P9003");
    // rubric 五维全集常量与 critic::Dim::ALL 同序。
    assert_eq!(RUBRIC_DIMS, ["exposure", "contrast", "color", "sharpness", "composition"]);
}

// ---------------------------------------------------------------------------
// 六族：每族一条静态注入断言 + 一条帧值对拍断言（§3.4）
// ---------------------------------------------------------------------------

#[test]
fn six_families_each_have_static_injection_and_frame_value_pair_assertions() {
    // 静态检出：注入实现源码里逐族注册键 + wiring.rs::resolve_frame 调用注入 +
    // movie.rs::render_scene_frame 在 Material / Light / Scene 组装之前消费。
    let finish_src: &str = include_str!("../src/finish.rs");
    let wiring_src: &str = include_str!("../src/wiring.rs");
    let movie_src: &str = include_str!("../../prism-render/src/movie.rs");
    assert!(wiring_src.contains("crate::finish::inject_frame_values"), "resolve_frame 必须调用注入");
    assert!(wiring_src.contains("fn bind_source"), "注入点须落在 bind_source 链路");
    assert!(movie_src.contains("fn wiring_frame_values"));
    assert!(movie_src.contains("fn render_scene_frame"));
    assert!(movie_src.contains("prism_dsl::signals_of"), "渲染前须消费六族信号");
    // 实现级静态门禁：注入函数只写 WiringFrame 键，不碰帧缓冲 / 帧后节点。
    assert!(finish_src.contains("pub fn inject_frame_values"));
    assert!(!finish_src.contains("eval_window("));
    assert!(!finish_src.contains("TileEffect"));
    assert!(finish_src.contains("wf.values"), "注入目标是生成源头的帧值表");

    for fam in TraceFamily::ALL {
        // 静态注入断言 ①：键名、引擎触点、注入点（生成源，禁止 postfx 末端）。
        let key: String = format!("w3.{}.signal", fam.tag());
        assert!(W3_INJECTION_KEYS.contains(&key.as_str()), "{key} 未登记");
        assert!(fam.engine_touchpoint().contains("crates/"), "{} 缺引擎触点", fam.tag());
        let point: &str = fam.injection_point();
        assert!(point.contains("wiring.rs"), "{} 注入点未在生成源", fam.tag());
        assert!(point.contains("movie.rs"), "{} 注入点未进渲染前链路", fam.tag());
        assert!(!point.contains("postfx"), "{} 注入点不得是帧后节点", fam.tag());
        assert_eq!(fam.strength_range(), (0, STRENGTH_MAX));
        assert_eq!(fam.analytic_value(), 0, "{} 的 analytic 值须恒 0", fam.tag());
        // 静态注入断言 ②：movie.rs 消费该族信号（渲染前，非帧后）。
        let field: &str = match fam {
            TraceFamily::Motion => "w3.motion",
            TraceFamily::Geometry => "w3.geometry",
            TraceFamily::MaterialLight => "w3.material_light",
            TraceFamily::TimeCamera => "w3.time_camera",
            TraceFamily::CompositionColor => "w3.composition_color",
            TraceFamily::EditAudio => "w3.edit_audio",
        };
        assert!(movie_src.contains(field), "{} 信号未在 render_scene_frame 消费", fam.tag());

        // 帧值对拍断言：只启用该族，逐帧逐位与声明强度 / 噪声对拍。
        let mut plan: FinishPlan = FinishPlan::analytic(42, 2);
        plan.mode = FinishMode::Crafted;
        plan.dims[fam.ordinal() as usize - 1] =
            TraceDim { enabled: true, strength: 640, line: 3 };
        for frame in 0..8u64 {
            let mut wf: WiringFrame = WiringFrame::default();
            inject_frame_values(&mut wf, &plan, frame, 42, 0);
            assert!(wf.values.contains_key(&key), "{} 第 {frame} 帧未注入", fam.tag());
            assert_frame_value_pairs(&wf, &plan, frame, 42, 0)
                .unwrap_or_else(|e| panic!("{} 第 {frame} 帧对拍失败：{e}", fam.tag()));
            // 其它族恒等 0（独立开关）。
            for other in TraceFamily::ALL {
                if other != fam {
                    assert_eq!(signals_of(&wf).of(other), 0.0);
                }
            }
        }
    }
}

/// 精简 RFC 1321 MD5（纯整数、零第三方；与 W1 `v3_language.rs` 同口径）。
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
        5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee, 0xf57c_0faf, 0x4787_c62a, 0xa830_4613,
        0xfd46_9501, 0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be, 0x6b90_1122, 0xfd98_7193,
        0xa679_438e, 0x49b4_0821, 0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa, 0xd62f_105d,
        0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8, 0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
        0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a, 0xfffa_3942, 0x8771_f681, 0x6d9d_6122,
        0xfde5_380c, 0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70, 0x289b_7ec6, 0xeaa1_27fa,
        0xd4ef_3085, 0x0488_1d05, 0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665, 0xf429_2244,
        0x432a_ff97, 0xab94_23a7, 0xfc93_a039, 0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
        0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1, 0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb,
        0xeb86_d391,
    ];

    pub fn hex(data: &[u8]) -> String {
        let mut msg: Vec<u8> = data.to_vec();
        let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_le_bytes());
        let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        for chunk in msg.chunks_exact(64) {
            let mut m: [u32; 16] = [0; 16];
            for (i, slot) in m.iter_mut().enumerate() {
                *slot = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let [mut a, mut b, mut c, mut d] = state;
            for i in 0..64 {
                let (f, g): (u32, usize) = match i {
                    0..=15 => ((b & c) | (!b & d), i),
                    16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let sum: u32 = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
                let next: u32 = b.wrapping_add(sum.rotate_left(S[i]));
                a = d;
                d = c;
                c = b;
                b = next;
            }
            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in state {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }
}
