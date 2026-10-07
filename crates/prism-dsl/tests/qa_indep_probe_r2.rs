//! 第 2 轮独立复核探针—— DSL 裸标志分支。
//! 验证 P1 修复（value=None 分支）未引入新问题：原本应报错的空值字段是否被误当 true。

use prism_dsl::{compile_v2, DslError};

const HEADER: &str = "// prism-dsl 1\n";

fn compile_one_note(body: &str) -> Result<bool, DslError> {
    let src = format!("{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"n\" {{ {body} }} }}\n");
    Ok(compile_v2(&src)?.scenes[0].notes[0].arpeggio)
}

/// 裸 arpeggio 应为 true（缺陷 3 修复点）。
#[test]
fn dsl_bare_arpeggio_true() {
    assert!(compile_one_note("arpeggio").unwrap(), "裸 arpeggio 应 = true");
}

/// 裸标志仅在「名 后紧跟 }」生效；其他裸标志字段（scale 等）应为「无值」——不能误当 true。
/// scale 无值应报错而非静默当作某种真值。
#[test]
fn dsl_bare_non_flag_field_not_silently_true() {
    // scale 是取字符串的字段，裸写不应被 None 分支吞掉成「合法」。
    let r = compile_one_note("scale");
    assert!(r.is_err(), "裸 scale 无值应报错，实际 {r:?}");
}

/// 多字段混合：`scale major arpeggio` 末尾裸标志=true，前序字段正常。
#[test]
fn dsl_trailing_bare_flag_with_prior_fields() {
    assert!(compile_one_note("scale major root 62 arpeggio").unwrap());
    assert!(!compile_one_note("scale major root 62 arpeggio off").unwrap());
}

/// 裸 root（无值）应报错，不得被 None 分支误当合法。
#[test]
fn dsl_bare_root_errors() {
    assert!(compile_one_note("root").is_err(), "裸 root 应报错");
}
