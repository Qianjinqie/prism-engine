//! V3.0 W2 出口判据（app / render 侧）：七族端到端样例、同源对拍 0 差异、
//! 引擎 API 直算核对、L3 两帧值、39 行接线落到求值器、负控、零损失。
//!
//! 「每族一个端到端样例」的族指 §5.3 R5010 七族：
//! `particles / sim / compose / morph / light / audio / post`。
//! 每个样例都带一个 `cap "R-1"` 声明（使 `compile.wiring` 非空，接线通路
//! 启用），并含该族 DSL 节点；旧文件不声明 `cap` ⇒ 返回空凭证 ⇒ 零损失。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use prism_core::{Light, Material, PropertyChannel};
use prism_dsl::{
    compile_auto, compile_v2, example_cap_block, full_declaration_blocks, param_kind, range_rule,
    CapFamily, ParamKind, WiringEntry, WiringParam, WiringPlan, WiringValue, CAPABILITY_ROWS,
};
use prism_math::Vec3;
use prism_render::movie::wiring_frame_values;

use prism_app::w2_eval::{
    evaluate, family_node_ids, r5010_families, receipts_bytes, R5010_FAMILIES,
};

const FPS: u32 = 8;

/// 一个族样例：顶层 `cap "R-1"` + 该族 DSL 节点。
fn sample(tag: &str, family_block: &str) -> String {
    let cap: String = example_cap_block("R-1").expect("R-1 样例块应可生成");
    format!(
        "// prism-dsl 1\nmovie \"w2-{tag}\" {{ duration 0.5s fps 4 resolution 64x36 }}\n{cap}\nscene \"s\" {{\n  duration 0.5s\n  {family_block}\n}}\n"
    )
}

/// 七族样例源（族标签 → DSL 块）。
fn family_samples() -> Vec<(&'static str, String)> {
    let blocks: &[(&str, &str)] = &[
        ("particles", "layer particles { count 12 speed 1 }"),
        ("sim", "sim cloth { grid 4 stiffness 0.9 damping 0.02 }"),
        (
            "compose",
            "layer text a { content \"A\" x 8 y 8 size 10px color #ffffff }\n  compose { add layer:a }",
        ),
        (
            "morph",
            "layer text a { morph from \"a.svg\" to \"b.svg\" content \"A\" x 8 y 8 size 10px color #ffffff }",
        ),
        (
            "light",
            "stage3d { light key { direction (1,-1,-0.5) color #ffffff intensity 2 } }",
        ),
        ("audio", "audio { music { bpm 120 gain 0.8 duck 12 } }"),
        ("post", "post { bloom 0.2 }"),
    ];
    blocks
        .iter()
        .map(|(tag, block)| (*tag, sample(tag, block)))
        .collect()
}

fn compile(src: &str) -> prism_dsl::CompileV2Output {
    compile_auto(src)
        .unwrap_or_else(|e| panic!("样例应编译成功，实际第 {} 行：{}", e.line, e.message))
        .into_v2_output()
}

/// 「直接调用引擎 API 构造同一场景」：不经 DSL 文本，用公开类型逐项 new
/// 出与 `example_cap_block` 语义相同的接线表。
fn direct_plan() -> WiringPlan {
    let mut entries: Vec<WiringEntry> = Vec::new();
    for row in CAPABILITY_ROWS {
        if row.family == CapFamily::Platform {
            continue;
        }
        let mut params: BTreeMap<String, WiringParam> = BTreeMap::new();
        for (i, p) in row.api_params.iter().enumerate() {
            let value: WiringValue = match param_kind(row, p) {
                ParamKind::Bool => WiringValue::Bool(false),
                ParamKind::Int => WiringValue::Int(
                    range_rule(row.id, p)
                        .map(|r| (r.lo.ceil() as i64).max(1))
                        .unwrap_or(1),
                ),
                ParamKind::Num => {
                    let text: String = range_rule(row.id, p)
                        .map(|r| format!("{}", (r.lo + r.hi) / 2.0))
                        .unwrap_or_else(|| "0.5".to_string());
                    WiringValue::Num(text.parse::<f32>().expect("样例数值应可解析").to_bits())
                }
                ParamKind::Text => WiringValue::Text(format!(
                    "w2-{}-{}",
                    row.id.to_lowercase(),
                    p.to_lowercase()
                )),
                ParamKind::Variant => WiringValue::Int(if i == 0 { 1 } else { 0 }),
            };
            params.insert((*p).to_string(), WiringParam::Const(value));
        }
        entries.push(WiringEntry {
            row_id: row.id.to_string(),
            line: 0,
            scene: None,
            params,
        });
    }
    WiringPlan { entries, finish: None }
}

/// 起 CLI 子进程跑 `prism-app render`。
fn run_render(prism: &PathBuf, out: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args([
            "render",
            prism.to_str().expect("路径应可转 str"),
            "--codec",
            "mjpeg",
            "--out",
            out.to_str().expect("路径应可转 str"),
        ])
        .output()
        .expect("子进程应可启动")
}

fn write_temp(tag: &str, src: &str) -> (PathBuf, PathBuf) {
    let dir: PathBuf = std::env::temp_dir().join(format!("prism-w2-e2e-{tag}"));
    fs::create_dir_all(&dir).expect("临时目录应可创建");
    let prism: PathBuf = dir.join("case.prism");
    let out: PathBuf = dir.join("out.mp4");
    fs::write(&prism, src).expect("临时 .prism 应可写");
    let _ = fs::remove_file(&out);
    (prism, out)
}

/// 检查一段 needle 是否出现在 haystack 的字节窗口里。
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ---------------------------------------------------------------------------
// 1. 七族端到端样例 + R5010 销号（编译 / 凭证 / 真实成片）
// ---------------------------------------------------------------------------

#[test]
fn w2_seven_families_each_have_end_to_end_sample_and_no_r5010() {
    let samples = family_samples();
    assert_eq!(samples.len(), 7);
    assert_eq!(R5010_FAMILIES.len(), 7);
    let family_tag: BTreeMap<&str, &str> = BTreeMap::from([
        ("particles", "particles"),
        ("sim", "sim"),
        ("compose", "compose_morph"),
        ("morph", "compose_morph"),
        ("light", "light"),
        ("audio", "audio"),
        ("post", "post"),
    ]);
    for (tag, src) in &samples {
        let tag: &str = *tag;
        let out = compile(src);
        assert!(!out.wiring.is_empty(), "{tag} 样例应启用接线通路");
        let tags: BTreeSet<&'static str> = r5010_families(&out);
        assert!(tags.contains(tag), "{tag} 族节点应被识别，实际：{tags:?}");
        let want: &str = family_tag[tag];
        let fam: CapFamily = match want {
            "particles" => CapFamily::Particles,
            "sim" => CapFamily::Sim,
            "compose_morph" => CapFamily::ComposeMorph,
            "light" => CapFamily::Light,
            "audio" => CapFamily::Audio,
            "post" => CapFamily::Post,
            other => panic!("未知族 {other}"),
        };
        assert!(
            !family_node_ids(&out, fam).is_empty(),
            "{tag} 族节点应进求值链"
        );
        let receipts = evaluate(&out, 0);
        assert!(
            receipts.iter().any(|r| r.family == want),
            "{tag} 样例应产出 {want} 族凭证，实际：{:?}",
            receipts.iter().map(|r| r.family).collect::<Vec<&str>>()
        );
        for r in &receipts {
            assert!(!r.engine_api.is_empty(), "{tag}/{want} 的既有 API 锚点不能为空");
            assert!(!r.digest.is_empty(), "{tag}/{want} 的既有 API 输出不能为空");
        }

        // 真实 CLI 成片：不拒绝、不静默降级。
        let (prism, mp4) = write_temp(tag, src);
        let cli: Output = run_render(&prism, &mp4);
        let stderr = String::from_utf8_lossy(&cli.stderr);
        assert!(cli.status.success(), "{tag} 样例应出片：{stderr}");
        assert!(!stderr.contains("R5010"), "{tag} 不应再出现 R5010：{stderr}");
        let bytes: Vec<u8> = fs::read(&mp4).expect("{tag} 应写出 MP4");
        assert!(bytes.len() >= 12 && &bytes[4..8] == b"ftyp", "{tag} 应是 MP4 容器");
    }
}

#[test]
fn w2_combined_sample_covers_all_seven_families_and_all_39_rows() {
    let cap: String = full_declaration_blocks();
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-all\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{cap}\nscene \"s\" {{\n  duration 1s\n  layer particles {{ count 12 speed 1 }}\n  layer text a {{ morph from \"a.svg\" to \"b.svg\" content \"A\" x 8 y 8 size 10px color #ffffff }}\n  compose {{ add layer:a }}\n  sim cloth {{ grid 4 stiffness 0.9 damping 0.02 }}\n  stage3d {{ light key {{ direction (1,-1,-0.5) color #ffffff intensity 2 }} }}\n  audio {{ music {{ bpm 120 gain 0.8 duck 12 }} }}\n  post {{ bloom 0.2 }}\n}}\n"
    );
    let out = compile(&src);
    assert_eq!(out.wiring.entries.len(), 33, "39 行中 33 个非平台域行全部接线");
    assert_eq!(out.wiring.param_count(), 198);
    let tags: BTreeSet<&'static str> = r5010_families(&out);
    assert_eq!(tags.len(), 7, "七族销号 7/7：{tags:?}");
    for t in R5010_FAMILIES {
        assert!(tags.contains(t), "缺 {t} 族");
    }
    let receipts = evaluate(&out, 0);
    // 11 个 CapFamily 全部由 cap 行或 R5010 节点触发，无死分支。
    assert_eq!(receipts.len(), 11, "族凭证数：{:?}", receipts.iter().map(|r| r.family).collect::<Vec<&str>>());
    let mut keys: BTreeSet<String> = BTreeSet::new();
    for r in &receipts {
        keys.extend(r.param_keys.iter().cloned());
    }
    for row in CAPABILITY_ROWS.iter().filter(|r| r.family != CapFamily::Platform) {
        for p in row.api_params {
            assert!(
                keys.contains(&format!("{}.{}", row.id, p)),
                "{} 的 {p} 未进求值凭证",
                row.id
            );
        }
    }
    assert!(keys.iter().any(|k| k.starts_with("node.")), "R5010 节点参数未进凭证");
}

// ---------------------------------------------------------------------------
// 2. 同源对拍 0 差异 + 既有引擎 API 直算核对
// ---------------------------------------------------------------------------

#[test]
fn w2_dsl_vs_direct_api_same_scene_receipts_are_byte_identical() {
    let cap: String = full_declaration_blocks();
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-pair\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{cap}\nscene \"s\" {{ duration 1s layer particles {{ count 12 speed 1 }} }}\n"
    );
    let dsl_out = compile(&src);
    // 直接构造同一场景：保留同一份节点图 / 通道 / 时间线，仅把接线声明
    // 换成用公开类型逐项 new 出的等价 WiringPlan。
    let mut api_out = dsl_out.clone();
    api_out.wiring = direct_plan();
    assert_eq!(
        dsl_out.wiring.canonical_bytes(),
        api_out.wiring.canonical_bytes(),
        "声明产物与直接构造产物应逐字节相同"
    );
    for frame in 0..3_u64 {
        let a = evaluate(&dsl_out, frame);
        let b = evaluate(&api_out, frame);
        assert_eq!(
            receipts_bytes(&a),
            receipts_bytes(&b),
            "第 {frame} 帧同源对拍应 0 差异"
        );
    }
}

#[test]
fn w2_receipt_digests_contain_real_engine_api_outputs() {
    let cap: String = full_declaration_blocks();
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-api\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{cap}\nscene \"s\" {{ duration 1s layer particles {{ count 12 speed 1 }} stage3d {{ light key {{ direction (1,-1,-0.5) color #ffffff intensity 2 }} }} }}\n"
    );
    let out = compile(&src);
    let receipts = evaluate(&out, 0);
    let by_family: BTreeMap<&str, &Vec<u8>> = receipts
        .iter()
        .map(|r| (r.family, &r.digest))
        .collect();

    // 直接调用既有引擎 API，逐项核对凭证 digest 里的真实返回值。
    let rt = prism_render::rt_backend::RtBackend::new(1, 1);
    assert!(contains(
        by_family["render"],
        &(rt.spp as u64).to_le_bytes()
    ), "RtBackend::new 的 spp 应进 Render 族凭证");
    let mat: Material = Material::new(Vec3::new(1.0, 1.0, 1.0), 0.5, 0.5);
    assert!(contains(
        by_family["camera_material"],
        &mat.roughness.to_bits().to_le_bytes()
    ), "Material::new 的 roughness 应进凭证");
    let mut emitter = prism_particles::Emitter::default();
    emitter.rate = 12.0;
    emitter.seed = 0x5eed;
    assert!(contains(
        by_family["particles"],
        &emitter.rate.to_bits().to_le_bytes()
    ), "Emitter::rate 应进 Particles 族凭证");
    assert!(contains(
        by_family["particles"],
        &(prism_particles::ParticlePool::new(12).capacity() as u64).to_le_bytes()
    ), "ParticlePool::capacity 应进凭证");
    let light: Light = Light::new(Vec3::new(1.0, -1.0, -0.5), Vec3::new(1.0, 1.0, 1.0), 2.0);
    assert!(contains(
        by_family["light"],
        &light.intensity.to_bits().to_le_bytes()
    ), "Light::new 的 intensity 应进 Light 族凭证");
    let reg = prism_postfx::registry::default_registry();
    for n in reg.names() {
        assert!(contains(by_family["post"], n.as_bytes()), "postfx 注册表 {n} 应进凭证");
    }
    // Audio 族的 gain 来自 `cap "E-5"` 的显式声明值（不是引擎缺省），
    // 用同一值直接调用既有混音 API，核对输出确实进凭证。
    let gain: f32 = wiring_frame_values(&out, 0)
        .value("E-5.gain")
        .expect("E-5.gain 应已由生成源头解算");
    let audio_buf = prism_audio::PcmBuffer::from_interleaved(48_000, 1, vec![0.0_f32; 4])
        .expect("4 个单声道样本应可构造");
    let audio_track = prism_audio::mixer::MixTrack::new(audio_buf, gain);
    let audio_mixed = prism_audio::mixer::mix_tracks(&[audio_track])
        .expect("单轨混音应成功");
    assert!(contains(
        by_family["audio"],
        &gain.to_bits().to_le_bytes()
    ), "MixTrack::new 的 gain 应进 Audio 族凭证");
    assert!(contains(
        by_family["audio"],
        format!("{audio_mixed:?}").as_bytes()
    ), "mix_tracks 的既有输出应进 Audio 族凭证");

    // 每个族的 engine_api 锚点必须指向真实存在的源码文件。
    for r in &receipts {
        let path: &str = r.engine_api.split('@').next().unwrap_or(r.engine_api);
        let file: &str = path.split(':').next().unwrap_or(path).trim();
        if file.starts_with("crates/") {
            let full: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(file);
            assert!(full.exists(), "{} 的锚点文件不存在：{}", r.family, full.display());
        }
    }
}

// ---------------------------------------------------------------------------
// 3. 核心项 L3 两帧值（渲染路径 wiring_frame_values）
// ---------------------------------------------------------------------------

#[test]
fn w2_core_l3_two_frames_differ_and_match_analytic_channel() {
    // 六类核心项各取一行参数，声明成解析式子块（生成源头时序绑定）。
    let cases: &[(&str, &str, &str)] = &[
        ("G-1", "revolutions", "t * 0.5 + 1"),
        ("R-2", "roughness", "t * 0.1 + 0.2"),
        ("R-1", "clamp_indirect", "t * 0.5 + 1"),
        ("G-2", "displacement", "t * 0.2 + 0.3"),
        ("E-4", "frame", "t * 2 + 1"),
        ("R-13", "n0", "t * 4 + 8"),
    ];
    let mut blocks: Vec<String> = Vec::new();
    for (row, param, expr) in cases {
        let block: String = example_cap_block(row).expect("样例块应可生成");
        let mut out: String = String::new();
        for line in block.lines() {
            if line.trim_start().starts_with(&format!("{param} ")) {
                out.push_str(&format!("    {param} {{ expr \"{expr}\" }}\n"));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        blocks.push(out);
    }
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-l3\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{}\nscene \"s\" {{ duration 1s }}\n",
        blocks.join("\n")
    );
    let out = compile(&src);
    let f0 = wiring_frame_values(&out, 0);
    let f4 = wiring_frame_values(&out, 4);
    let t4: f32 = 4.0 / FPS as f32;
    for (row, param, _expr) in cases {
        let key: String = format!("{row}.{param}");
        let v0: f32 = f0.value(&key).unwrap_or_else(|| panic!("{key} 第 0 帧未解算"));
        let v4: f32 = f4.value(&key).unwrap_or_else(|| panic!("{key} 第 4 帧未解算"));
        assert_ne!(v0.to_bits(), v4.to_bits(), "{key} 两帧必须取到不同参数值");
        let ch: &PropertyChannel = out
            .channels
            .iter()
            .find(|c| c.name == format!("w2/{row}/0/{param}"))
            .map(|c| &c.channel)
            .unwrap_or_else(|| panic!("{key} 的时序通道未登记"));
        assert_eq!(v0.to_bits(), ch.sample(0.0).to_bits(), "{key} 第 0 帧解析式对拍");
        assert_eq!(v4.to_bits(), ch.sample(t4).to_bits(), "{key} 第 4 帧解析式对拍");
    }
    // 帧凭证本身也逐帧不同（L3 通道确实进了 Deliverable 的求值链）。
    assert_ne!(f0.canonical_bytes(), f4.canonical_bytes());
}

// ---------------------------------------------------------------------------
// 4. 反假绿运行时负控 / 零损失
// ---------------------------------------------------------------------------

#[test]
fn w2_drop_one_wiring_entry_changes_receipts_and_lowers_param_count() {
    let cap: String = full_declaration_blocks();
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-drop\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{cap}\nscene \"s\" {{ duration 1s }}\n"
    );
    let full = compile(&src);
    let full_receipts = evaluate(&full, 0);
    let full_keys: usize = full_receipts.iter().map(|r| r.param_keys.len()).sum();
    for id in ["R-1", "R-2", "G-3", "E-8", "R-13"] {
        let mut cut = full.clone();
        let n: usize = cut.wiring.entries.len();
        cut.wiring.entries.retain(|e| e.row_id != id);
        assert_eq!(cut.wiring.entries.len(), n - 1, "删 {id} 应少一行");
        let cut_receipts = evaluate(&cut, 0);
        let cut_keys: usize = cut_receipts.iter().map(|r| r.param_keys.len()).sum();
        let row = CAPABILITY_ROWS.iter().find(|r| r.id == id).unwrap();
        assert_eq!(cut_keys, full_keys - row.api_params.len(), "删 {id} 参数数应下降");
        assert_ne!(
            receipts_bytes(&full_receipts),
            receipts_bytes(&cut_receipts),
            "删 {id} 后凭证必须变化（同源对拍可检出）"
        );
    }
}

#[test]
fn w2_negative_controls_fail_end_to_end_with_line_and_three_part() {
    let bad_cases: &[(&str, &str)] = &[
        (
            "missing",
            "cap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 }",
        ),
        ("range", "cap \"R-1\" { spp 99999 max_bounces 1 clamp_indirect 0.5 nee_only false }"),
        ("unknown", "cap \"X-9\" { a 1 }"),
        ("platform", "cap \"P-1\" { }"),
    ];
    for (tag, cap) in bad_cases {
        let src: String = format!(
            "// prism-dsl 1\nmovie \"x\" {{ duration 0.5s fps 4 resolution 64x36 }}\n{cap}\nscene \"s\" {{ duration 0.5s }}\n"
        );
        let (prism, mp4) = write_temp(&format!("neg-{tag}"), &src);
        let cli: Output = run_render(&prism, &mp4);
        assert!(!cli.status.success(), "{tag} 负控应以退出码 1 终止");
        let stderr = String::from_utf8_lossy(&cli.stderr);
        assert!(stderr.contains("[P80"), "{tag} 应带 P8xxx 码：{stderr}");
        assert!(stderr.contains("第 3 行"), "{tag} 应带行号：{stderr}");
        assert!(stderr.contains("建议："), "{tag} 应带三段式建议：{stderr}");
        assert!(!mp4.exists(), "{tag} 失败时不应留下输出文件");
    }
}

#[test]
fn w2_zero_loss_old_files_have_no_receipts_and_two_runs_are_identical() {
    // 旧文件不声明 cap ⇒ evaluate 返回空凭证 ⇒ CLI 不打印接线凭证。
    let old: &[&str] = &[
        include_str!("../../../examples/t0_media_mesh.prism"),
        include_str!("../../../examples/w2d_showcase.prism"),
    ];
    for src in old {
        let out = compile_v2(src).expect("旧 v2 黄金文件应编译成功");
        assert!(out.wiring.is_empty());
        assert!(evaluate(&out, 0).is_empty(), "旧文件不得产生接线凭证");
    }
    // 新文件两跑逐字节。
    let src: String = format!(
        "// prism-dsl 1\nmovie \"w2-two\" {{ duration 0.5s fps 4 resolution 64x36 }}\n{}\nscene \"s\" {{ duration 0.5s layer particles {{ count 12 speed 1 }} }}\n",
        full_declaration_blocks()
    );
    let a = compile(&src);
    let b = compile(&src);
    assert_eq!(a.wiring.canonical_bytes(), b.wiring.canonical_bytes());
    assert_eq!(receipts_bytes(&evaluate(&a, 0)), receipts_bytes(&evaluate(&b, 0)));
    let (prism, mp4) = write_temp("two-runs", &src);
    let first: Output = run_render(&prism, &mp4);
    assert!(first.status.success(), "两跑首跑应成功");
    let first_bytes: Vec<u8> = fs::read(&mp4).expect("首跑 MP4 应存在");
    let second: Output = run_render(&prism, &mp4);
    assert!(second.status.success(), "两跑次跑应成功");
    let second_bytes: Vec<u8> = fs::read(&mp4).expect("次跑 MP4 应存在");
    assert_eq!(first_bytes, second_bytes, "同文件同种子两跑应逐字节一致");
    // 旧文件 CLI stdout 不打印接线凭证（零损失）。
    let (old_prism, old_mp4) = write_temp("old", old[1]);
    let old_cli: Output = run_render(&old_prism, &old_mp4);
    assert!(old_cli.status.success(), "旧文件应照常出片");
    let stdout = String::from_utf8_lossy(&old_cli.stdout);
    assert!(!stdout.contains("接线凭证"), "旧文件 stdout 不得出现新凭证：{stdout}");
}
