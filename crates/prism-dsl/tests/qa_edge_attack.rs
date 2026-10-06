//! QA 边界反例攻击（W3-A4 验收清单 C 项：DSL 侧）。
//!
//! 与 src/lib.rs 内单测的分工：单测覆盖"实现者预期的行为"，本文件从
//! **对抗视角**攻击边界与规格缝隙——空输入、数值溢出、重名歧义、版本头
//! 错位、装包边界。断言一律以文档规格为准（兼容性宪章 C11/C17/C18/C20、
//! 架构总设计 §2.3/§8.3），失败即视为被验实现与规格的偏差证据。

use prism_dsl::{compile_v2, compile_v2_with_diagnostics};

/// 构造断言助手：v2 编译必须失败，返回 (行号, 消息)。
fn expect_err_v2(src: &str, want_line: usize, needle: &str) {
    match compile_v2(src) {
        Ok(_) => panic!("以下源码应当编译失败（攻击未被发现）：\n{}", src),
        Err(e) => {
            assert_eq!(e.line, want_line, "错误行号不符：{}", e);
            assert!(
                e.message.contains(needle),
                "错误信息应包含 {:?}，实际为：{}",
                needle,
                e.message
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 攻击 1：空文件与仅版本头（C18 P1003 空文档防呆）
// ---------------------------------------------------------------------------

#[test]
fn empty_source_reports_p1003() {
    // 完全空文件：不得 panic、不得静默产出空图，必须 P1003 三段式。
    match compile_v2("") {
        Ok(_) => panic!("空文件应被拒绝"),
        Err(e) => {
            assert!(e.message.contains("P1003"), "应含 P1003：{}", e.message);
            assert!(e.message.contains("建议："), "应含建议段（C19）：{}", e.message);
        }
    }
}

#[test]
fn version_head_only_document_reports_p1003() {
    // 仅版本头、无任何块：同为空文档，P1003。
    expect_err_v2("// prism-dsl 1\n", 1, "P1003");
}

// ---------------------------------------------------------------------------
// 攻击 2：单位数值大值与溢出（C17 单位字面量 + C20 禁静默容错）
// ---------------------------------------------------------------------------

#[test]
fn large_millisecond_duration_is_accepted_and_faithful() {
    // 999999999ms（≈ 1e6 s）：合法大值，f32 可表示（精度内），应编译成功
    // 且归一化值与 1e6 s 的偏差在 f32 精度内（999999999 在 f32 中最近
    // 可表示值为 1e9，除以 1000 后 = 1e6）。
    let src: &str = "// prism-dsl 1\nmovie \"m\" { duration 999999999ms }\nscene \"s\" {}\n";
    let out = compile_v2(src).unwrap_or_else(|e| panic!("大值毫秒时长应可编译：{}", e));
    assert!(
        (out.duration - 1_000_000.0).abs() < 0.5,
        "999999999ms 应归一化到 ≈1e6 s，实际 {}",
        out.duration
    );
    assert!(out.duration.is_finite(), "时长必须有限（C20）");
}

#[test]
fn overflow_duration_inf_is_rejected() {
    // ≈1e39 s（39 个 9，超出 f32 上限 ≈3.4e38）：解析为 +inf。
    // inf 时长若直通产物，下游帧数 = inf → NaN 灾难。按 C20（禁止静默
    // 容错）与 C18（首发防呆），inf 必须在编译期显式拒绝。
    let src: &str = concat!(
        "// prism-dsl 1\n",
        "movie \"m\" { duration 999999999999999999999999999999999999999s }\n",
        "scene \"s\" {}\n"
    );
    match compile_v2(src) {
        Ok(out) => panic!(
            "inf 时长应被拒绝，实际编译成功且 duration = {}（is_finite = {}）",
            out.duration,
            out.duration.is_finite()
        ),
        Err(e) => {
            assert!(e.message.contains("建议："), "拒绝消息应含建议段（C19）：{}", e.message);
        }
    }
}

// ---------------------------------------------------------------------------
// 攻击 3：重名歧义（C18 防呆——transition 按名匹配，重名 = 匹配歧义）
// ---------------------------------------------------------------------------

#[test]
fn duplicate_scene_names_are_rejected() {
    // 两个同名 scene：transition from/to 按场景名匹配，重名即歧义，
    // 必须编译期拒绝（同 P3008 同场景块重名同级别防呆）。
    let src: &str = concat!(
        "// prism-dsl 1\n",
        "scene \"a\" {}\n",
        "scene \"a\" {}\n",
        "transition \"t\" { from \"a\" to \"a\" duration 0.5s }\n"
    );
    match compile_v2(src) {
        Ok(_) => panic!(
            "重名 scene 应被拒绝（transition 按名匹配会歧义），实际编译成功"
        ),
        Err(e) => {
            assert!(e.message.contains("建议："), "应含建议段（C19）：{}", e.message);
            assert!(e.message.contains("[P"), "应含错误码（C18）：{}", e.message);
        }
    }
}

#[test]
fn duplicate_import_packs_are_rejected() {
    // 同名 import packs 两次：装包清单重复登记 = 加载歧义，应拒绝。
    // （最小语义下包间无引用，"循环 import"以"重复同名"为最近可测形态。）
    let src: &str = concat!(
        "// prism-dsl 1\n",
        "import packs \"kit\" {}\n",
        "import packs \"kit\" {}\n",
        "scene \"s\" {}\n"
    );
    match compile_v2(src) {
        Ok(out) => panic!(
            "同名 import packs 应被拒绝，实际编译成功且登记 {} 条清单",
            out.packs.len()
        ),
        Err(e) => {
            assert!(e.message.contains("建议："), "应含建议段（C19）：{}", e.message);
        }
    }
}

// ---------------------------------------------------------------------------
// 攻击 4：版本头错位（C11：版本头在文档首行）
// ---------------------------------------------------------------------------

#[test]
fn version_head_on_non_first_line_warns_per_c11() {
    // 版本头不在首行（首行是空行）：按 C11"文档首行 // prism-dsl 1"，
    // 非首行版本头 = 未按规格声明 → 应降级为 W1001 警告（而非静默当作
    // 已声明）。断言：必须出现 W1001 诊断。
    let src: &str = "\n// prism-dsl 1\nscene \"s\" {}\n";
    let (out, diags) =
        compile_v2_with_diagnostics(src).expect("含错位版本头的文档应能编译");
    assert_eq!(out.scenes.len(), 1);
    let has_w1001 = diags.iter().any(|d| d.code == "W1001");
    assert!(
        has_w1001,
        "版本头不在首行应触发 W1001（C11 首行规格），实际诊断：{diags:?}"
    );
}

// ---------------------------------------------------------------------------
// 攻击 5：装包 engine 版本边界语义（C12/V9：区间含/不含边界的精确性）
// ---------------------------------------------------------------------------

#[test]
fn pack_engine_boundary_semantics() {
    // 当前引擎版本 0.1.0（见 prism-dsl/Cargo.toml version）。
    // 通过侧：闭边界 >=0.1 / ==0.1 / <=0.1 / <0.2 全部满足。
    for engine in [">=0.1", "==0.1", "=0.1", "<=0.1", "<0.2", ">=0.1, <0.2"] {
        let src: &str = &format!(
            "// prism-dsl 1\nimport packs \"kit\" {{ engine \"{engine}\" }}\nscene \"s\" {{}}\n"
        );
        let out = compile_v2(src)
            .unwrap_or_else(|e| panic!("engine \"{engine}\" 应满足（当前 0.1.0）：{}", e));
        assert_eq!(out.packs.len(), 1, "engine \"{engine}\" 应登记装包清单");
    }
    // 拒绝侧：严格大于下界（0.1.0 不满足 >0.1）与不可能区间 → P4001。
    expect_err_v2(
        "// prism-dsl 1\nimport packs \"kit\" { engine \">0.1\" }\nscene \"s\" {}\n",
        2,
        "P4001",
    );
    expect_err_v2(
        "// prism-dsl 1\nimport packs \"kit\" { engine \">=99.0, <99.1\" }\nscene \"s\" {}\n",
        2,
        "P4001",
    );
}
