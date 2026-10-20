//! QA 攻击性测试 —— prism-dsl note 块。
//!
//! 攻击 note 块的字段白名单 / 数值防呆 / 音阶白名单 / 缺省回填 / 多块。
//! 断言以文档规格（codegen.rs 注释与 note 块字段规格）为准，失败即偏差证据。

use prism_dsl::{compile_v2, NoteInfo};

const HEADER: &str = "// prism-dsl 1\n";

/// 编译含单 note 块的场景，返回该 NoteInfo。
fn compile_one_note(note_body: &str) -> Result<NoteInfo, prism_dsl::DslError> {
    let src = format!("{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"n\" {{ {note_body} }} }}\n");
    let out = compile_v2(&src)?;
    Ok(out.scenes[0].notes[0].clone())
}

/// 缺陷：缺省回填应符合文档（major / root 60 / bpm 120 / bars 4 / seed 0）。
#[test]
fn note_defaults_match_docs() {
    let info = compile_one_note("").expect("空 note 块应可编译并回填缺省");
    assert_eq!(info.scale, "major", "缺省音阶 major");
    assert_eq!(info.root, 60, "缺省根音 C4=60");
    assert!((info.bpm - 120.0).abs() < 1e-6, "缺省 120 BPM");
    assert_eq!(info.bars, 4, "缺省 4 小节");
    assert_eq!(info.seed, 0, "缺省种子 0");
    assert!(!info.arpeggio, "缺省不琶音");
}

/// 缺陷：音阶白名单——8 个合法标识应全部接受，且回填正确。
#[test]
fn note_scale_whitelist_accepts_all() {
    for s in ["major", "minor", "pentatonic", "minor-pentatonic", "dorian", "phrygian", "lydian", "mixolydian"] {
        let info = compile_one_note(&format!("scale {s}")).unwrap_or_else(|e| panic!("音阶 {s} 应被接受：{e}"));
        assert_eq!(info.scale, s, "音阶回填 {s}");
    }
}

/// 缺陷：未知音阶应报 P1023（拒绝静默容错）。
#[test]
fn note_unknown_scale_rejected() {
    let err = compile_one_note("scale bebop").unwrap_err();
    assert!(err.message.contains("P1023"), "未知音阶应 P1023：{}", err.message);
}

/// 缺陷：root 越界 [0,127] 外应报 P1024。
#[test]
fn note_root_bounds() {
    assert!(compile_one_note("root 0").is_ok());
    assert!(compile_one_note("root 127").is_ok());
    let err = compile_one_note("root 128").unwrap_err();
    assert!(err.message.contains("P1024"), "越界 root 应 P1024：{}", err.message);
    let err2 = compile_one_note("root -1").unwrap_err();
    assert!(err2.message.contains("P1024"), "负 root 应 P1024：{}", err2.message);
}

/// 缺陷：bpm ≤ 0 应报 P1019。
#[test]
fn note_bpm_must_be_positive() {
    let err = compile_one_note("bpm 0").unwrap_err();
    assert!(err.message.contains("P1019"), "bpm=0 应 P1019：{}", err.message);
    let err2 = compile_one_note("bpm -10").unwrap_err();
    assert!(err2.message.contains("P1019"), "负 bpm 应 P1019：{}", err2.message);
}

/// 缺陷：bars < 1 应报 P1025；bars=1 合法。
#[test]
fn note_bars_minimum() {
    assert!(compile_one_note("bars 1").is_ok());
    let err = compile_one_note("bars 0").unwrap_err();
    assert!(err.message.contains("P1025"), "bars=0 应 P1025：{}", err.message);
}

/// 缺陷：seed < 0 应报 P1026；seed=0 合法。
#[test]
fn note_seed_non_negative() {
    assert!(compile_one_note("seed 0").is_ok());
    let err = compile_one_note("seed -1").unwrap_err();
    assert!(err.message.contains("P1026"), "负 seed 应 P1026：{}", err.message);
}

/// 缺陷：未知字段应报 P1002（白名单外拒绝）。
#[test]
fn note_unknown_field_rejected() {
    let err = compile_one_note("tempo 99").unwrap_err();
    assert!(err.message.contains("P1002"), "未知字段应 P1002：{}", err.message);
}

/// 缺陷：arpeggio 布尔数字与标识符语义（值形态，可达）。
#[test]
fn note_arpeggio_flag_semantics() {
    assert!(compile_one_note("arpeggio 1").unwrap().arpeggio, "arpeggio 1 = true");
    assert!(!compile_one_note("arpeggio 0").unwrap().arpeggio, "arpeggio 0 = false");
    assert!(!compile_one_note("arpeggio off").unwrap().arpeggio, "arpeggio off = false");
    assert!(compile_one_note("arpeggio on").unwrap().arpeggio, "arpeggio on = true");
    assert!(!compile_one_note("arpeggio false").unwrap().arpeggio, "arpeggio false = false");
}

/// 缺陷（P1，文档/解析器不符）：codegen `build_note`（codegen.rs:2033-2035）
/// 明确支持「标志字段（无值 = true）」——`arpeggio` 裸写应为 true；但
/// parser `parse_named_fields` 对无值字段一律报 P2107（缺值），致使该
/// 文档化形态**不可达**。文档（codegen.rs:2031 注释、note 块规格）与解析器
/// 行为矛盾。
///
/// 失败代表缺陷：`arpeggio` 裸标志应被解析为 true（或文档须移除该形态）。
#[test]
fn note_arpeggio_bare_flag_documented_but_unreachable() {
    let r = compile_one_note("arpeggio");
    assert!(
        r.is_ok(),
        "文档声明裸 arpeggio = true，但解析器拒绝：{:?}",
        r.err().map(|e| e.message)
    );
    assert!(r.unwrap().arpeggio);
}

/// 缺陷：root 非整数（小数）应按四舍五入（文档 round），且需确定。
#[test]
fn note_root_rounds() {
    assert_eq!(compile_one_note("root 60.4").unwrap().root, 60);
    assert_eq!(compile_one_note("root 60.6").unwrap().root, 61);
}

/// 缺陷：多个 note 块应按出现序全部收集（不被覆盖）。
#[test]
fn note_multiple_blocks_collected() {
    let src = format!(
        "{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"a\" {{ seed 1 }} note \"b\" {{ seed 2 }} }}\n"
    );
    let out = compile_v2(&src).unwrap();
    let notes = &out.scenes[0].notes;
    assert_eq!(notes.len(), 2, "两个 note 块应都在");
    assert_eq!(notes[0].name, "a");
    assert_eq!(notes[1].name, "b");
    assert_eq!(notes[0].seed, 1);
    assert_eq!(notes[1].seed, 2);
}

/// 缺陷：note 块确定性——同源码两遍编译产物逐字段一致。
#[test]
fn note_compile_deterministic() {
    let src = format!(
        "{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"theme\" {{ scale dorian root 62 bpm 96 bars 8 seed 42 arpeggio 1 }} }}\n"
    );
    let a = compile_v2(&src).unwrap();
    let b = compile_v2(&src).unwrap();
    assert_eq!(a.scenes[0].notes, b.scenes[0].notes, "note 块编译应逐字段一致");
}

/// 缺陷：note 关键字缺名称（无字符串）应报错而非 panic。
#[test]
fn note_missing_name_errors() {
    let src = format!("{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note {{ scale major }} }}\n");
    assert!(compile_v2(&src).is_err(), "缺 note 名称应报错");
}

/// 缺陷：note 块缺右花括号应报错而非 panic。
#[test]
fn note_missing_brace_errors() {
    let src = format!("{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"n\" {{ scale major }}\n");
    assert!(compile_v2(&src).is_err(), "缺右花括号应报错");
}
