//! V3.0 W3 出口判据（app / render 侧）：§3.5 V1–V6 的端到端凭证。
//!
//! - V1 档位忠实：`analytic` 出片逐字节 ≡ 纯解析基线；`crafted` 六族逐帧注入；
//! - V2 负控：帧后贴皮 / 真随机静态门禁在 CLI 入口同样判红；
//! - V3 成对盲测：确定性可计算判官 + 冻结门槛 + 反假绿负控 + critic 客观侧证；
//! - V4 匀质：`analytic` 与旧文件零观感凭证、不打印、出片不变；
//! - V5 同源对拍：语言声明 ≡ 直接调引擎 API 构造同一观感配置；
//! - V6 确定性 + 零损失：两跑逐字节、3 个 `.prism` 旧入口、W1/W2 门禁不退化。
//!
//! 「只接线、不重实现」：`w3_eval` 把六族帧值作为实参交给既有 crate API，
//! 真实返回值进 `digest`；本测试不复制任何数学 / 渲染 / 质感求值。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use prism_dsl::{
    compile_auto, compile_v2, compile_v3, FinishMode, FinishPlan, TraceDim, TraceFamily,
    WiringFrame,
};
use prism_gpu::CpuRefBackend;
use prism_render::movie::wiring_frame_values;

use prism_app::w3_blind::{
    analyticity_metrics, critic_side_evidence, critic_side_report, family_tags_match_dsl,
    judge, observation, run_control, run_protocol, AnalyticityMetrics, BLIND_FAMILY_TAGS,
    BLIND_SAMPLE_COUNT, BLIND_THRESHOLD, BLIND_THRESHOLD_BASIS, CRITIC_REGRESSION_TOL, Verdict,
};
use prism_app::w3_eval::{evaluate, evaluate_plan, receipts_bytes, W3Receipt};

// ---------------------------------------------------------------------------
// 样例（32×24、1s、2fps = 2 帧；与 W1 同源对拍同构，保证逐字节比较快速）
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

const CRAFTED_DIMS: &str = r#"finish = crafted {
    motion enabled true strength 620
    geometry enabled false strength 0
    material_light enabled true strength 400
    time_camera enabled true strength 300
    composition_color enabled true strength 250
    edit_audio enabled true strength 500
}"#;

fn source(finish: &str) -> String {
    format!("// prism-dsl 2\nseed = 42\n{finish}\n{SCENE_EMIT}")
}

fn analytic_source() -> String {
    source("finish = analytic")
}

fn crafted_source() -> String {
    source(CRAFTED_DIMS)
}

fn compile(src: &str) -> prism_dsl::CompileV2Output {
    compile_auto(src)
        .unwrap_or_else(|e| panic!("应编译成功，却在第 {} 行失败：{}", e.line, e.message))
        .into_v2_output()
}

/// 渲染产物的规范字节序列（标题 / 画幅 / 帧率 / 时长 / 每帧像素）。
fn render_bytes(c: &prism_dsl::CompileV2Output) -> Vec<u8> {
    let mut device = CpuRefBackend;
    let deliverable = prism_render::render_movie(c, &mut device)
        .expect("render_movie 应成功出片");
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(deliverable.title.as_bytes());
    bytes.extend_from_slice(&deliverable.width.to_le_bytes());
    bytes.extend_from_slice(&deliverable.height.to_le_bytes());
    bytes.extend_from_slice(&deliverable.fps.to_le_bytes());
    bytes.extend_from_slice(&deliverable.duration.to_le_bytes());
    bytes.extend_from_slice(&(deliverable.frames.len() as u64).to_le_bytes());
    for frame in &deliverable.frames {
        bytes.extend_from_slice(&(frame.width as u64).to_le_bytes());
        bytes.extend_from_slice(&(frame.height as u64).to_le_bytes());
        bytes.extend_from_slice(&frame.pixels);
    }
    bytes
}

/// 直接调引擎 API 构造同一观感配置（V5 的另一侧）。
fn direct_plan() -> FinishPlan {
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

// ---------------------------------------------------------------------------
// 1. V1 / V4：档位忠实 + 匀质零噪点 + 生成源头（非帧后）渲染前注入
// ---------------------------------------------------------------------------

#[test]
fn v1_analytic_render_equals_pure_baseline_and_crafted_is_injected_at_source() {
    let a = compile(&analytic_source());
    let c = compile(&crafted_source());
    // analytic 出片逐字节 ≡ 纯解析基线（手写等价 v2，直接调引擎 API）。
    const V2_EQUIVALENT: &str = r#"// prism-dsl 1
movie "g" { duration 1s fps 2 resolution 32x24 format png }
scene "s" {
    duration 1s
    stage3d {
        camera main { orbit { center 0,0,0 radius 4.5 } fov 55 }
        sdf ball { kind sphere radius 0.75 }
        material hero { albedo #c0392b roughness 0.4 metalness 0.1 }
        light key { direction(0.4,-1,-0.6) color #ffffff intensity 3.2 }
    }
}
output { resolution 32x24 format png }
"#;
    let baseline = compile_v2(V2_EQUIVALENT).expect("等价 v2 应可编译");
    assert_eq!(render_bytes(&a), render_bytes(&baseline), "analytic ≡ 纯解析基线逐字节");
    // crafted 确实在**渲染前**逐帧注入 ⇒ 帧值与 analytic 不同（不是帧后贴皮）。
    let wf_a: WiringFrame = wiring_frame_values(&a, 0);
    let wf_c: WiringFrame = wiring_frame_values(&c, 0);
    assert!(wf_a.values.keys().all(|k| !k.starts_with("w3.")), "analytic 零注入键");
    assert_eq!(wf_c.value("w3.mode"), Some(1.0), "crafted 渲染前已开档");
    assert!(wf_c.values.contains_key("w3.motion.signal"));
    assert!(!wf_c.values.contains_key("w3.geometry.signal"), "geometry 绝对匀质");
    assert_ne!(wf_a.canonical_bytes(), wf_c.canonical_bytes());
    // 逐帧帧值确定性（两跑逐字节）。
    assert_eq!(
        wiring_frame_values(&c, 1).canonical_bytes(),
        wiring_frame_values(&c, 1).canonical_bytes()
    );
}

#[test]
fn v4_analytic_and_old_files_emit_no_w3_receipts() {
    assert!(evaluate(&compile(&analytic_source()), 0).is_empty(), "analytic 零观感凭证");
    // 版本 1 旧文件 ⇒ compile_auto 恒等 compile_v2 ⇒ wiring.finish = None ⇒ 空凭证。
    for src in [
        include_str!("../../../examples/t0_media_mesh.prism"),
        include_str!("../../../examples/w2d_showcase.prism"),
    ] {
        let c = compile(src);
        assert!(c.wiring.finish_plan().is_none());
        assert!(evaluate(&c, 0).is_empty(), "旧文件不得产生观感凭证");
    }
    // 无头黄金文件属 DSL v1 语法，既有产品路径走 `compile()`（W1 口径）；
    // 它不经过 v3 执行器，天然没有任何 W3 声明 / 凭证。
    let no_head: &str = include_str!("../../../examples/spin_red_ball.prism");
    let legacy = prism_dsl::compile(no_head).expect("v1 黄金示例必须能被 compile() 编译");
    assert_eq!(legacy.width, 480);
    assert!(!no_head.contains("w3.") && !no_head.contains("finish"), "旧文件无 W3 一等字段");
}

// ---------------------------------------------------------------------------
// 2. V3：成对盲测（确定性可计算判官，非主观自评、非 critic 分数当胜率）
// ---------------------------------------------------------------------------

#[test]
fn v3_blind_protocol_passes_frozen_threshold_with_two_runs_identical() {
    let report = run_protocol();
    let again = run_protocol();
    assert_eq!(report.canonical_bytes(), again.canonical_bytes(), "两跑逐字节一致");
    assert_eq!(report.total, BLIND_SAMPLE_COUNT as usize);
    assert_eq!(report.samples.len(), BLIND_SAMPLE_COUNT as usize);
    assert!(report.passed, "胜率 {} 应过冻结门槛 {}", report.win_rate, BLIND_THRESHOLD);
    assert!(report.win_rate > 0.5, "胜率须显著 > 0.5");
    assert_eq!(report.threshold, BLIND_THRESHOLD);
    assert_eq!(report.threshold_basis, BLIND_THRESHOLD_BASIS);
    // 逐对输出完整可复现（脚本 / 文档共用 report_text）。
    let text: String = report.report_text();
    assert!(text.contains("盲测样本数 N=64"));
    assert!(text.contains(&format!("B 胜={}/{}", report.b_wins, report.total)));
    let lines: usize = text.lines().count();
    assert_eq!(lines, 3 + BLIND_SAMPLE_COUNT as usize, "表头 + 64 行逐对输出");
    // 每个样本：种子 / 强度 / 顺序 / 判选 / B胜 都是确定性记录。
    for s in &report.samples {
        assert!(s.strengths.iter().any(|v| *v > 0), "至少一族启用");
        assert!(s.strengths.iter().all(|v| *v <= 1000));
    }
    // 判官对同一对强制二选一、且认数值不认位置。
    let a: Vec<f32> = observation(42, &[620, 0, 400, 300, 250, 500], false);
    let b: Vec<f32> = observation(42, &[620, 0, 400, 300, 250, 500], true);
    let (v1, _) = judge(&a, &b);
    let (v2, _) = judge(&b, &a);
    assert_ne!(v1, v2);
    assert!(matches!(v1, Verdict::Second) && matches!(v2, Verdict::First));
}

#[test]
fn v3_control_shows_judge_is_not_biased_towards_b() {
    let control = run_control();
    assert!(control.win_rate < control.threshold, "负控（B=A）应失去区分力：{}", control.win_rate);
    assert!(control.ties > 0, "A/A 对必出现指标完全相同");
    let m: AnalyticityMetrics = analyticity_metrics(&observation(7, &[0; 6], false));
    assert!(m.second_difference_energy < 1e-4);
    assert!(m.temporal_regularity > 0.99);
}

#[test]
fn v3_critic_five_dims_are_side_evidence_only_not_the_win_rate() {
    // critic 五维只做「画质不退化」侧证，不进盲测判官、不换算胜率。
    let ev = critic_side_evidence(3);
    assert!(ev.no_regression, "画质不退化：\n{}", ev.report_text());
    assert!(ev.analytic_total > 0.0 && ev.crafted_total > 0.0);
    assert_eq!(ev.analytic_dims.len(), 5);
    assert_eq!(ev.crafted_dims.len(), 5);
    assert!(ev.report_text().contains("exposure"));
    assert_eq!(CRITIC_REGRESSION_TOL, 0.25, "侧证容差冻结，不事后调整");
    // 全池侧证两跑逐字节 + 零退化。
    let all = critic_side_report();
    let all2 = critic_side_report();
    assert_eq!(all.len(), BLIND_SAMPLE_COUNT as usize);
    for (a, b) in all.iter().zip(all2.iter()) {
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert!(a.no_regression);
    }
    // 判官胜率与 critic 总分是两把不同的尺子（维度不错配）。
    let report = run_protocol();
    let total_critic: f32 = all.iter().map(|e| e.crafted_total - e.analytic_total).sum();
    assert_ne!(report.win_rate, total_critic as f64, "critic 分数不得等同盲测胜率");
    // 六族标签与 DSL 一一对应（静态注入断言的读数）。
    assert!(family_tags_match_dsl());
    assert_eq!(BLIND_FAMILY_TAGS.len(), TraceFamily::ALL.len());
}

// ---------------------------------------------------------------------------
// 3. 六族既有 crate 接线（D-32：只接线、不重实现）
// ---------------------------------------------------------------------------

#[test]
fn six_families_are_wired_to_existing_crate_apis_with_deterministic_receipts() {
    let c = compile(&crafted_source());
    let receipts: Vec<W3Receipt> = evaluate(&c, 0);
    assert_eq!(receipts.len(), 6, "六族接线齐全");
    let expected: [&str; 6] = [
        "motion", "geometry", "material_light", "time_camera", "composition_color", "edit_audio",
    ];
    for (r, tag) in receipts.iter().zip(expected.iter()) {
        assert_eq!(r.family, *tag);
        assert!(r.engine_api.starts_with("crates/"), "{} 缺既有 API 锚点", tag);
        assert!(r.engine_api.contains(".rs:"), "{} 锚点须带行号", tag);
        assert!(!r.digest.is_empty(), "{} 的既有 API 输出不得为空", tag);
        // **绝对匀质族**（geometry，enabled false + strength 0）按 D-30 不注入
        // 任何 `w3.geometry.*` 键 ⇒ param_keys 为空是正确语义；但凭证仍须调用
        // 既有 API（digest 非空）以证明六族接线齐全，而不是只接启用族。
        if *tag == "geometry" {
            assert!(r.param_keys.is_empty(), "绝对匀质族不得出现生成源头注入键");
        } else {
            assert!(!r.param_keys.is_empty(), "{} 未消费到生成源头帧值", tag);
            assert!(r.param_keys.iter().all(|k| k.starts_with(&format!("w3.{tag}."))));
        }
    }
    // 两跑逐字节 + 逐族锚点覆盖六个既有 crate。
    let again: Vec<W3Receipt> = evaluate(&c, 0);
    assert_eq!(receipts_bytes(&receipts), receipts_bytes(&again));
    let apis: String = receipts.iter().map(|r| r.engine_api).collect::<Vec<_>>().join("|");
    for crate_name in [
        "prism-author", "prism-genart", "prism-postfx", "prism-aesthetic", "prism-audio",
    ] {
        assert!(apis.contains(crate_name), "{crate_name} 未接线");
    }
    // V5：语言声明 ≡ 直接调引擎 API 构造同一观感配置，凭证逐字节。
    let plan: FinishPlan = direct_plan();
    let wf: WiringFrame = wiring_frame_values(&c, 0);
    let direct: Vec<W3Receipt> = evaluate_plan(&plan, &wf, 0);
    assert_eq!(receipts_bytes(&receipts), receipts_bytes(&direct), "V5 同源对拍逐字节");
    // 即使某族强度为 0 也接了既有 API（证明六族接线齐全，非只接启用族）。
    assert_eq!(direct.iter().find(|r| r.family == "geometry").unwrap().digest.len() > 0, true);
}

// ---------------------------------------------------------------------------
// 4. CLI 端到端：观感凭证（crafted 打印 / analytic 与旧文件零输出）+ 静态门禁
// ---------------------------------------------------------------------------

fn run_render(prism: &PathBuf, out: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args([
            "render",
            prism.to_str().expect("路径应可转 str"),
            "--codec",
            "mjpeg",
            "--fast",
            "--out",
            out.to_str().expect("路径应可转 str"),
        ])
        .output()
        .expect("子进程应可启动")
}

fn write_temp(tag: &str, src: &str) -> (PathBuf, PathBuf) {
    let dir: PathBuf = std::env::temp_dir().join(format!("prism-w3-e2e-{tag}"));
    fs::create_dir_all(&dir).expect("临时目录应可创建");
    let prism: PathBuf = dir.join("case.prism");
    let out: PathBuf = dir.join("out.mp4");
    fs::write(&prism, src).expect("临时 .prism 应可写");
    let _ = fs::remove_file(&out);
    (prism, out)
}

#[test]
fn cli_prints_w3_receipts_only_for_crafted_and_gates_reject_negative_sources() {
    // crafted：六族观感凭证逐行打印，且成片成功。
    let (prism, out) = write_temp("crafted", &crafted_source());
    let crafted = run_render(&prism, &out);
    let stdout: String = String::from_utf8_lossy(&crafted.stdout).into_owned();
    assert_eq!(crafted.status.code(), Some(0), "crafted 应出片成功：{stdout}");
    assert!(stdout.contains("观感凭证   : finish = crafted / 6 族"), "缺观感凭证：{stdout}");
    for tag in ["motion", "geometry", "material_light", "time_camera", "composition_color", "edit_audio"] {
        assert!(stdout.contains(tag), "缺 {tag} 族凭证：{stdout}");
    }
    assert!(out.exists(), "应落盘 MP4");

    // analytic：观感凭证零输出（零损失 stdout），仍成功出片。
    let (prism_a, out_a) = write_temp("analytic", &analytic_source());
    let analytic = run_render(&prism_a, &out_a);
    let stdout_a: String = String::from_utf8_lossy(&analytic.stdout).into_owned();
    assert_eq!(analytic.status.code(), Some(0), "analytic 应出片成功：{stdout_a}");
    assert!(!stdout_a.contains("观感凭证"), "analytic 不得打印观感凭证：{stdout_a}");

    // 旧文件：观感凭证零输出。
    let (prism_o, out_o) = write_temp("old", include_str!("../../../examples/w2d_showcase.prism"));
    let old = run_render(&prism_o, &out_o);
    let stdout_o: String = String::from_utf8_lossy(&old.stdout).into_owned();
    assert_eq!(old.status.code(), Some(0), "旧文件应出片成功：{stdout_o}");
    assert!(!stdout_o.contains("观感凭证"), "旧文件不得打印观感凭证：{stdout_o}");

    // V2 静态门禁：帧后贴皮 / 真随机在 CLI 入口同样判红，且带行号 + 三段式。
    // 负控样例必须能先过 v3 前端（门禁位于 `compile_v3` 的 codegen_v3 之后）：
    // `post { grain }` 放进 `emit` 的 v2 场景文本内 —— v3 视作 f-string 字面量，
    // 降级出的 v2 认识 scene 内 post 块 ⇒ 编译成功 ⇒ posthoc_gate 全文命中 `post {`。
    // 这与 §3.3「现成反例 post { bloom, grain, ... }」是同一个帧后贴皮样本。
    let posthoc = r#"// prism-dsl 2
seed = 42
finish = analytic
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
    post { grain 0.3 }
}
output { resolution 32x24 format png }
"""
"#;
    let (prism_p, out_p) = write_temp("posthoc", posthoc);
    let p = run_render(&prism_p, &out_p);
    let p_err: String = String::from_utf8_lossy(&p.stderr).into_owned();
    assert_ne!(p.status.code(), Some(0), "帧后贴皮必须判红");
    assert!(p_err.contains("P9011") && p_err.contains("第") && p_err.contains("行"), "门禁须带行号：{p_err}");

    // 真随机：写在注释行。`true_random_gate` 全文逐行扫描（不跳注释）⇒ P9012；
    // v3 lexer 跳过注释 ⇒ 场景编译照常成功 ⇒ 确实由静态门禁判红，而非语法错误抢先。
    let truerand = format!(
        "// prism-dsl 2\nseed = 42\nfinish = analytic\n// jitter from SystemTime::now()\n{SCENE_EMIT}"
    );
    let (prism_r, out_r) = write_temp("truerand", &truerand);
    let r = run_render(&prism_r, &out_r);
    let r_err: String = String::from_utf8_lossy(&r.stderr).into_owned();
    assert_ne!(r.status.code(), Some(0), "真随机必须判红");
    assert!(r_err.contains("P9012") && r_err.contains("行"), "门禁须带行号：{r_err}");
}

// ---------------------------------------------------------------------------
// 5. V5 / V6：同源对拍逐字节 + 两跑确定性 + W1/W2 门禁不退化
// ---------------------------------------------------------------------------

#[test]
fn v5_language_output_equals_direct_engine_call_byte_for_byte() {
    let v3 = compile_v3(&crafted_source())
        .unwrap_or_else(|e| panic!("v3 应编译成功，第 {} 行：{}", e.line, e.message));
    // 展开文本独立编译与 base 逐字段一致（W1 同源对拍口径）。
    let lowered = compile_v2(&v3.lowered_source).expect("展开文本应可独立编译");
    assert_eq!(format!("{lowered:?}"), format!("{:?}", v3.base));
    // 语言声明的 FinishPlan ≡ 直接构造的 FinishPlan（诊断行号不进产物）。
    let via_language: &FinishPlan = v3.base.wiring.finish_plan().unwrap();
    let via_api: FinishPlan = direct_plan();
    assert_eq!(via_language.canonical_bytes(), via_api.canonical_bytes());
    // 成片逐字节（语言产物 ≡ 展开文本独立编译）。
    assert_eq!(render_bytes(&v3.base), render_bytes(&lowered));
    // 两跑逐字节。
    let v3b = compile_v3(&crafted_source()).unwrap();
    assert_eq!(v3.artifact_hash, v3b.artifact_hash);
    assert_eq!(render_bytes(&v3.base), render_bytes(&v3b.base));
}

#[test]
fn v6_zero_loss_and_w1_w2_gates_do_not_regress() {
    // 3 个黄金 .prism 旧入口零损失（compile_auto ≡ compile_v2 逐字段）。
    for src in [
        include_str!("../../../examples/spin_red_ball.prism"),
        include_str!("../../../examples/t0_media_mesh.prism"),
        include_str!("../../../examples/w2d_showcase.prism"),
    ] {
        let direct = compile_v2(src).map(|o| format!("{o:?}")).map_err(|e| e.message);
        let auto = compile_auto(src)
            .map(|o| format!("{:?}", o.into_v2_output()))
            .map_err(|e| e.message);
        assert_eq!(direct, auto, "旧文件分派逐字段等价");
    }
    // W1/W2 门禁不退化：39 行接线表、226 参数分母、R5010 七族、覆盖率 100%。
    assert_eq!(prism_dsl::CAPABILITY_ROWS.len(), 39, "W2 39 行规格表不变");
    assert_eq!(prism_dsl::total_param_count(), 226, "W2 参数分母 226 不变");
    let w2 = prism_app::w2_eval::evaluate(&compile(&crafted_source()), 0);
    assert!(w2.iter().all(|r| !r.engine_api.is_empty()), "W2 七族凭证仍可产出");
    // W3 不引入任何第三方依赖：Cargo.lock 的 `source =` 计数为 0。
    let lock: &str = include_str!("../../../Cargo.lock");
    assert_eq!(lock.lines().filter(|l: &&str| l.starts_with("source = ")).count(), 0);
    // 零 libm / 零真随机：实现源码只走 hash_mix / pcg::hash2_f32。
    let finish_src: &str = include_str!("../../prism-dsl/src/finish.rs");
    assert!(finish_src.contains("hash_mix_unit_f32"));
    assert!(finish_src.contains("prism_noise::pcg::hash2_f32"));
}
