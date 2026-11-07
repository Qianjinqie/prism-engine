//! V3.0 W1 参数覆盖率门禁骨架 + 错误注入集。
//!
//! 静态测试遍历 `docs/v3.0-触达规格表.md` 全 39 行，**逐行回读引擎源码**核对
//! `api_path:line` 处确为公开 API 定义，且 `api_params` 全集与该定义逐项一致。
//!
//! W1 判红集合（路线图 §5.3 W1 / §2.2）：
//!   ① 缺 `api_path`  ② 参数全集为空  ③ 目标级缺失  ④ 故意删 1 行映射
//! 另加两条判别力约束：假 exempt、参数全集与引擎源码不符。
//!
//! W1 口径：`dsl_fields` 按当前真实现状如实登记，**空不判红**；覆盖率支如实
//! 输出当前百分比，不设阈值。100% / P0 ≥ L2 / 核心项 L3 是 §7 #2 与 W2/W4 出口。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const SPEC: &str = include_str!("../../../docs/v3.0-触达规格表.md");

/// 规格表的一行（7 列）。
#[derive(Debug, Clone)]
struct Row {
    id: String,
    capability: String,
    api_path: String,
    api_params: Vec<String>,
    dsl_fields: Vec<String>,
    target: String,
    exempt: String,
}

#[derive(Debug, Clone, Default)]
struct Stats {
    rows: usize,
    api_params_total: usize,
    mapped_rows: usize,
    params_intersection: usize,
    params_tail_intersection: usize,
    per_row_gap: BTreeMap<String, usize>,
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_ident(s: &str) -> bool {
    let b: &[u8] = s.as_bytes();
    !b.is_empty() && is_ident_start(b[0]) && b.iter().all(|&c| is_ident_char(c))
}

/// 抽取反引号包裹的 token 全集（表格里的 `` `x` `` 列表）。
fn backticks(cell: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest: &str = cell;
    loop {
        match rest.find('`') {
            None => break,
            Some(a) => {
                rest = &rest[a + 1..];
                match rest.find('`') {
                    None => break,
                    Some(b) => {
                        out.push(rest[..b].to_string());
                        rest = &rest[b + 1..];
                    }
                }
            }
        }
    }
    out
}

fn expected_ids() -> BTreeSet<String> {
    let mut ids: BTreeSet<String> = BTreeSet::new();
    for i in 1..=13 {
        ids.insert(format!("R-{i}"));
    }
    for i in 1..=10 {
        ids.insert(format!("G-{i}"));
        ids.insert(format!("E-{i}"));
    }
    for i in 1..=6 {
        ids.insert(format!("P-{i}"));
    }
    ids
}

fn parse_rows(spec: &str) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for raw in spec.lines() {
        let line: &str = raw.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cols: Vec<&str> = line.split('|').collect();
        if cols.len() < 9 {
            continue;
        }
        let id: &str = cols[1].trim();
        if !matches!(id.chars().next(), Some('R' | 'G' | 'E' | 'P')) {
            continue;
        }
        if !id.contains('-') {
            continue;
        }
        rows.push(Row {
            id: id.to_string(),
            capability: cols[2].trim().to_string(),
            api_path: cols[3].trim().trim_matches('`').to_string(),
            api_params: backticks(cols[4]),
            dsl_fields: backticks(cols[5]),
            target: cols[6].trim().to_string(),
            exempt: cols[7].trim().to_string(),
        });
    }
    rows
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 从 `api_path:line` 回读源码，抽取该定义行的**公开参数全集**。
/// struct → pub 字段；enum → 全部变体；fn → 全部形参。
fn engine_params(api_path: &str) -> Result<Vec<String>, String> {
    let (path, line_s): (&str, &str) = api_path
        .rsplit_once(':')
        .ok_or_else(|| format!("api_path 格式应为 path:line，实际 '{api_path}'"))?;
    let line_no: usize = line_s
        .parse::<usize>()
        .map_err(|_| format!("api_path 行号不是数字：'{api_path}'"))?;
    if line_no == 0 {
        return Err(format!("api_path 行号必须 ≥ 1：'{api_path}'"));
    }
    let full: PathBuf = workspace_root().join(path);
    let text: String = fs::read_to_string(&full)
        .map_err(|e| format!("api_path 指向的文件不存在：{path}（{e}）"))?;
    let lines: Vec<&str> = text.lines().collect();
    let header: &str = *lines
        .get(line_no - 1)
        .ok_or_else(|| format!("api_path 行号 {line_no} 超出 {path} 的 {} 行", lines.len()))?;
    let trimmed: &str = header.trim_start();
    if !(trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub fn "))
    {
        return Err(format!(
            "{path}:{line_no} 不是公开 API 定义行（实际：{}）",
            trimmed.chars().take(60).collect::<String>()
        ));
    }
    if trimmed.starts_with("pub fn ") {
        let mut buf = String::new();
        for l in lines.iter().skip(line_no - 1).take(30) {
            buf.push_str(l.trim());
            buf.push(' ');
            if l.contains('{') || l.trim_end().ends_with(';') {
                break;
            }
        }
        let sig: &str = match buf.find('{') {
            Some(i) => &buf[..i],
            None => &buf,
        };
        let b: &[u8] = sig.as_bytes();
        let mut out: Vec<String> = Vec::new();
        let mut i: usize = 0;
        while i < b.len() {
            if is_ident_start(b[i]) {
                let start: usize = i;
                while i < b.len() && is_ident_char(b[i]) {
                    i += 1;
                }
                let mut j: usize = i;
                while j < b.len() && (b[j] as char).is_whitespace() {
                    j += 1;
                }
                if j < b.len()
                    && b[j] == b':'
                    && (j + 1 >= b.len() || b[j + 1] != b':')
                {
                    out.push(sig[start..i].to_string());
                }
            } else {
                i += 1;
            }
        }
        return Ok(out);
    }
    // struct / enum：花括号深度为 1 时取 pub 字段或枚举变体。
    let mut depth: i32 = header.chars().filter(|&c| c == '{').count() as i32
        - header.chars().filter(|&c| c == '}').count() as i32;
    let mut fields: Vec<String> = Vec::new();
    let mut variants: Vec<String> = Vec::new();
    for l in lines.iter().skip(line_no).take(400) {
        let s: &str = l.trim();
        if depth == 1 {
            if let Some(rest) = s.strip_prefix("pub ") {
                if let Some((name, _)) = rest.split_once(':') {
                    let name: &str = name.trim();
                    if is_ident(name) {
                        fields.push(name.to_string());
                    }
                }
            }
            let first: &str = s.split(|c: char| !is_ident_char(c as u8)).next().unwrap_or("");
            if !first.is_empty()
                && first.as_bytes()[0].is_ascii_uppercase()
                && is_ident(first)
                && first != "Self"
                && !s.starts_with('#')
            {
                variants.push(first.to_string());
            }
        }
        depth += l.chars().filter(|&c| c == '{').count() as i32
            - l.chars().filter(|&c| c == '}').count() as i32;
        if depth <= 0 {
            break;
        }
    }
    if !fields.is_empty() {
        Ok(fields)
    } else {
        Ok(variants)
    }
}

/// 门禁主体：返回统计或判红原因列表。W1 的四类判红 + 两条判别力约束。
fn validate(spec: &str) -> Result<Stats, Vec<String>> {
    let rows: Vec<Row> = parse_rows(spec);
    let mut errors: Vec<String> = Vec::new();
    let want: BTreeSet<String> = expected_ids();
    let got: BTreeSet<String> = rows.iter().map(|r| r.id.clone()).collect();
    if rows.len() != 39 {
        errors.push(format!(
            "规格表应有 39 行映射，实际 {} 行（故意删 1 行必须判红）",
            rows.len()
        ));
    }
    for missing in want.difference(&got) {
        errors.push(format!("编号 {missing} 的映射行被删除"));
    }
    for extra in got.difference(&want) {
        errors.push(format!("出现未知编号 {extra}"));
    }
    if got.len() != rows.len() {
        errors.push("存在重复编号".to_string());
    }

    let mut stats = Stats::default();
    for r in &rows {
        // ① 缺 api_path
        if r.api_path.is_empty() || r.api_path == "∅" {
            errors.push(format!("{} 缺 api_path", r.id));
        } else {
            match engine_params(&r.api_path) {
                Err(msg) => errors.push(format!("{} api_path 无效：{}", r.id, msg)),
                Ok(engine) => {
                    // 逐行核对引擎 API：参数全集必须与源码逐项一致（判别力约束）。
                    if engine.is_empty() {
                        errors.push(format!(
                            "{} 的 api_path 在引擎侧参数全集为空，不能作为锚点",
                            r.id
                        ));
                    }
                    if r.api_params != engine {
                        errors.push(format!(
                            "{} api_params 与引擎源码不符：表={:?} 源码={:?}",
                            r.id, r.api_params, engine
                        ));
                    }
                }
            }
        }
        // ② 参数全集为空
        if r.api_params.is_empty() {
            errors.push(format!("{} 的 api_params（全集）为空", r.id));
        }
        // ③ 目标级缺失
        if r.target.is_empty() || r.target == "∅" {
            errors.push(format!("{} 缺目标级", r.id));
        } else if !matches!(r.target.as_str(), "L2" | "L3" | "不适用") {
            errors.push(format!("{} 目标级 '{}' 非法", r.id, r.target));
        }
        // 假 exempt 约束：只有 G5 平台域六行允许正式裁定，且目标级须为「不适用」。
        let has_exempt: bool = !r.exempt.is_empty() && r.exempt != "∅";
        let is_platform: bool = r.id.starts_with("P-");
        if has_exempt && !is_platform {
            errors.push(format!("{} 非平台域却挂 exempt_ruling（假 exempt 凑数）", r.id));
        }
        if has_exempt && r.target != "不适用" {
            errors.push(format!("{} 有 exempt 却标了目标级 '{}'", r.id, r.target));
        }
        if is_platform && !has_exempt {
            errors.push(format!("{} 平台域缺 G5 正式裁定", r.id));
        }
        if r.capability.is_empty() {
            errors.push(format!("{} 缺能力名", r.id));
        }

        // 覆盖率统计（如实，不判红）。
        stats.rows += 1;
        stats.api_params_total += r.api_params.len();
        if !r.dsl_fields.is_empty() {
            stats.mapped_rows += 1;
        }
        let field_set: BTreeSet<&str> = r.dsl_fields.iter().map(String::as_str).collect();
        let mut hit: usize = 0;
        for p in &r.api_params {
            if field_set.contains(p.as_str()) {
                hit += 1;
            }
        }
        stats.params_intersection += hit;
        // 第二种口径：DSL 字段路径的末段名（如 `media.sigma_t` → `sigma_t`）与
        // 引擎参数同名即算命中，反映「字段已接线但命名前缀不同」的真实触达。
        let tail_set: BTreeSet<&str> = r
            .dsl_fields
            .iter()
            .map(|f| f.rsplit('.').next().unwrap_or(f.as_str()))
            .collect();
        stats.params_tail_intersection += r
            .api_params
            .iter()
            .filter(|p| tail_set.contains(p.as_str()))
            .count();
        stats.per_row_gap.insert(
            r.id.clone(),
            r.api_params.len().saturating_sub(hit),
        );
    }
    if errors.is_empty() {
        Ok(stats)
    } else {
        Err(errors)
    }
}

fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

// ---------------------------------------------------------------------------
// 变异夹具
// ---------------------------------------------------------------------------

fn remove_row(spec: &str, id: &str) -> String {
    spec.lines()
        .filter(|l| !(l.trim().starts_with("| ") && l.contains(&format!("| {id} |"))))
        .collect::<Vec<&str>>()
        .join("\n")
}

fn set_cell(spec: &str, id: &str, col: usize, value: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for l in spec.lines() {
        if l.trim().starts_with("| ") && l.contains(&format!("| {id} |")) {
            let mut cols: Vec<String> = l.split('|').map(|s| s.to_string()).collect();
            if col < cols.len() {
                cols[col] = format!(" {value} ");
            }
            out.push(cols.join("|"));
        } else {
            out.push(l.to_string());
        }
    }
    out.join("\n")
}

fn assert_red(spec: &str, needle: &str) {
    match validate(spec) {
        Ok(_) => panic!("错误注入必须判红（期望错误含 '{needle}'），门禁却放行"),
        Err(errs) => {
            let joined: String = errs.join(" | ");
            assert!(
                joined.contains(needle),
                "判红原因应含 '{needle}'，实际：{joined}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 门禁本体
// ---------------------------------------------------------------------------

#[test]
fn spec_table_39_rows_all_have_api_path_and_full_params() {
    let stats: Stats = validate(SPEC).unwrap_or_else(|errs| {
        panic!("规格表门禁判红：\n  {}", errs.join("\n  "));
    });
    assert_eq!(stats.rows, 39);
    assert_eq!(stats.api_params_total > 0, true);
    // 每行的 api_path / 参数全集 / 目标级都被 `validate` 逐行核对过，这里再显式断言
    // 结构覆盖率读数，避免「门禁跑过但没输出数字」。
    assert_eq!(pct(stats.rows, 39), 100.0);
}

#[test]
fn injection_delete_one_row_is_red() {
    assert_red(&remove_row(SPEC, "R-7"), "R-7");
}

#[test]
fn injection_empty_api_path_is_red() {
    assert_red(&set_cell(SPEC, "G-8", 3, "∅"), "G-8");
}

#[test]
fn injection_empty_api_params_is_red() {
    assert_red(&set_cell(SPEC, "E-8", 4, "∅"), "E-8");
}

#[test]
fn injection_missing_target_is_red() {
    assert_red(&set_cell(SPEC, "R-1", 6, "∅"), "R-1");
}

#[test]
fn injection_fake_exempt_is_red() {
    let mutated: String = set_cell(SPEC, "R-2", 7, "不适用（伪造裁定）");
    assert_red(&mutated, "假 exempt");
}

#[test]
fn injection_wrong_api_param_is_red() {
    let mutated: String = set_cell(SPEC, "R-1", 4, "`spp` `max_bounces` `clamp_indirect` `fake_param`");
    assert_red(&mutated, "与引擎源码不符");
}

#[test]
fn every_api_path_points_at_a_real_engine_definition() {
    // 独立再走一遍：逐行打印锚点与参数个数，保证「逐行核对」不是纸面动作。
    let rows: Vec<Row> = parse_rows(SPEC);
    assert_eq!(rows.len(), 39);
    for r in &rows {
        let params: Vec<String> = engine_params(&r.api_path)
            .unwrap_or_else(|e| panic!("{} 的 api_path 不成立：{}", r.id, e));
        assert!(!params.is_empty(), "{} 参数全集为空", r.id);
        assert_eq!(params, r.api_params, "{} 参数全集与源码不一致", r.id);
    }
}

#[test]
fn coverage_report_is_honest_and_printed() {
    let stats: Stats = validate(SPEC).unwrap_or_else(|e| panic!("{e:?}"));
    let structure: f64 = pct(stats.rows, 39);
    let rows_cov: f64 = pct(stats.mapped_rows, 39);
    let params_cov: f64 = pct(stats.params_intersection, stats.api_params_total);
    let tail_cov: f64 = pct(stats.params_tail_intersection, stats.api_params_total);
    println!("=== V3.0 W1 覆盖率实录（如实输出当前真实现状，不设阈值）===");
    println!(
        "结构覆盖率（39 行全有 api_path:line + 非空参数全集 + 目标级）：39/39 = {structure:.2}%"
    );
    println!(
        "DSL 字段映射行覆盖率：{}/39 = {:.2}%",
        stats.mapped_rows, rows_cov
    );
    println!(
        "参数级覆盖率（全名同名交集 / 参数全集）：{}/{} = {:.2}%",
        stats.params_intersection, stats.api_params_total, params_cov
    );
    println!(
        "参数级覆盖率（末段同名交集 / 参数全集）：{}/{} = {:.2}%",
        stats.params_tail_intersection, stats.api_params_total, tail_cov
    );
    println!("每行差集条数（W2 接线工作量）：{:?}", stats.per_row_gap);
    assert_eq!(structure, 100.0);
    // 如实口径：当前远未达 100%，不许用假映射 / 假 exempt 凑数。
    assert!(rows_cov < 100.0, "W1 不应出现假绿式的 100% 映射率");
    assert!(params_cov < 100.0, "参数级覆盖率达标是 W2/W4 出口，不是 W1");
    assert!(tail_cov < 100.0, "末段口径同样不许在 W1 假绿");
    assert!(
        stats.params_tail_intersection > stats.params_intersection,
        "末段口径必须比全名口径更有判别力"
    );
    // 平台域六行必须挂正式裁定，其余各行不得挂。
    let rows: Vec<Row> = parse_rows(SPEC);
    assert_eq!(
        rows.iter().filter(|r| r.exempt != "∅" && !r.exempt.is_empty()).count(),
        6
    );
}
