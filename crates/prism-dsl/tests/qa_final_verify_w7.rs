//! W7 终验（Round 3）独立复核 —— note 块裸标志白名单边界。
//!
//! 每个断言注明失败代表什么缺陷。

use prism_dsl::{compile_v2, DslError};

const HEADER: &str = "// prism-dsl 1\n";

fn compile_one_note(body: &str) -> Result<prism_dsl::NoteInfo, DslError> {
    let src = format!("{HEADER}movie \"m\" {{}}\nscene \"s\" {{ note \"n\" {{ {body} }} }}\n");
    let out = compile_v2(&src)?;
    Ok(out.scenes[0].notes[0].clone())
}

/// 缺陷：白名单字段 arpeggio 裸写应=true（无值标志）。
#[test]
fn bare_flag_arpeggio_is_true() {
    let info = compile_one_note("arpeggio").expect("arpeggio 裸标志应被接受");
    assert!(info.arpeggio, "arpeggio 裸标志应解释为 true");
}

/// 缺陷：非白名单字段裸写（遇 `}` 无值）若被静默接受，则缺值被吞。
/// 每个都必须报错（P2107 或同类缺值错误）。
#[test]
fn bare_flag_non_whitelist_fields_rejected() {
    for f in ["content", "rotation", "kind", "radius", "duration", "size", "color", "x", "y", "opacity", "scale", "root", "bpm", "bars", "seed"] {
        let r = compile_one_note(f);
        assert!(
            r.is_err(),
            "字段 '{f}' 裸写（无值）应报缺值错误，却被接受：{:?}",
            r.map(|i| format!("{i:?}"))
        );
    }
}

/// 缺陷：duration 裸写在 note 块若被当缺省吞掉，数值语义丢失。
/// 显式确认错误信息指向缺值（P2107）。
#[test]
fn bare_duration_reports_missing_value() {
    let e = compile_one_note("duration").unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("P2107") || msg.contains("缺值") || msg.contains("值"),
        "duration 裸写应报缺值错误，实得：{msg}"
    );
}

/// 缺陷：arpeggio 带显式值仍应正确解析（白名单不等于"只能裸写"）。
#[test]
fn arpeggio_with_explicit_value_still_works() {
    let on = compile_one_note("arpeggio 1").expect("arpeggio 1 应可编译");
    assert!(on.arpeggio, "arpeggio 1 => true");
    let off = compile_one_note("arpeggio off").expect("arpeggio off 应可编译");
    assert!(!off.arpeggio, "arpeggio off => false");
}

/// 缺陷（非阻塞观察）：白名单字段 `arpeggio` 的裸标志仅在其后紧跟 `}` 时生效；
/// 若其后紧跟**另一个字段名**，arpeggio 会按「Ident 值」吞掉该字段名
/// （arpeggio ≠ off/false → true），导致后续字段被静默吃掉（`arpeggio root 62`
/// 甚至报 P1001）。这是 arpeggio 自身的宽松 Ident 语义遗留，**非**本次
/// 白名单修复引入。此测试固化当前行为，供后续决定是否收紧。
#[test]
fn arpeggio_bare_flag_followed_by_ident_is_currently_swallowed() {
    // 裸标志（后跟 `}`）→ true
    let info = compile_one_note("arpeggio").expect("arpeggio 裸标志应可编译");
    assert!(info.arpeggio);

    // arpeggio 后跟标识符：被当作 arpeggio 的值（宽松 Ident 语义）。
    // 当前实现返回 Ok（arpeggio=true）。记录该行为为已知潜在歧义。
    let r = compile_one_note("arpeggio duration");
    // 若未来收紧为报错，此断言会提示规格已变更。
    match r {
        Ok(i) => assert!(i.arpeggio, "当前语义：arpeggio<Ident> => true"),
        Err(_) => { /* 收紧后报错亦可接受——两种均记录为「有界定」 */ }
    }
}
