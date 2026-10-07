//! import packs 装包：清单模型 + 最小 TOML 解析骨架 + engine 版本范围校验。
//!
//! # 最小可用语义
//! - DSL 内联清单：`import packs "名" { engine ">=0.4, <0.5" provides { ... } }`；
//! - TOML 清单骨架：`PackManifest::parse_toml` 支持最小子集
//!   （`key = "value"` 行 + `[provides]` 段），包文件加载留 Phase 2/3；
//! - engine 范围校验（兼容性宪章 C12 / 验收 V9）：清单约束与当前引擎版本
//!   （`env!("CARGO_PKG_VERSION")`）比对，不满足 → P4001；格式非法 → P4002。

use crate::ast::{FieldValue, NamedField};
use crate::err_at;
use crate::DslError;

/// 语义化版本三元组（装包清单比对用的最小子集）。
pub type SemVer = (u32, u32, u32);

/// 版本比较算子。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VersionOp {
    /// >=
    Ge,
    /// >
    Gt,
    /// <=
    Le,
    /// <
    Lt,
    /// ==（也接受 `=` 写法）
    Eq,
}

/// 单条版本约束：`>=0.4` 之类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VersionReq {
    op: VersionOp,
    ver: SemVer,
}

impl VersionReq {
    /// 当前版本是否满足本约束。
    fn satisfied_by(self, current: SemVer) -> bool {
        match self.op {
            VersionOp::Ge => current >= self.ver,
            VersionOp::Gt => current > self.ver,
            VersionOp::Le => current <= self.ver,
            VersionOp::Lt => current < self.ver,
            VersionOp::Eq => current == self.ver,
        }
    }
}

/// 解析 `x.y` / `x.y.z` 语义化版本；缺省段补 0，非法输入返回 None。
pub(crate) fn parse_semver(text: &str) -> Option<SemVer> {
    let mut parts: [u32; 3] = [0, 0, 0];
    for (i, seg) in text.trim().split('.').enumerate() {
        if i >= 3 {
            return None;
        }
        parts[i] = seg.parse::<u32>().ok()?;
    }
    Some((parts[0], parts[1], parts[2]))
}

/// 解析 engine 约束串（如 `">=0.4, <0.5"`）为约束列表。
/// 返回 Err(描述) 表示清单格式非法（调用方转 P4002）。
pub(crate) fn parse_version_reqs(range: &str) -> Result<Vec<VersionReq>, String> {
    let mut reqs: Vec<VersionReq> = Vec::new();
    for cond in range.split(',') {
        let cond: &str = cond.trim();
        if cond.is_empty() {
            continue;
        }
        // 算子：两字符优先（>= / <= / ==），再单字符（> / < / =）。
        let (op, rest): (VersionOp, &str) = if let Some(rest) = cond.strip_prefix(">=") {
            (VersionOp::Ge, rest)
        } else if let Some(rest) = cond.strip_prefix("<=") {
            (VersionOp::Le, rest)
        } else if let Some(rest) = cond.strip_prefix("==") {
            (VersionOp::Eq, rest)
        } else if let Some(rest) = cond.strip_prefix('>') {
            (VersionOp::Gt, rest)
        } else if let Some(rest) = cond.strip_prefix('<') {
            (VersionOp::Lt, rest)
        } else if let Some(rest) = cond.strip_prefix('=') {
            (VersionOp::Eq, rest)
        } else {
            return Err(format!("约束 '{}' 缺少比较算子（>= / > / <= / < / ==）", cond));
        };
        let ver: SemVer = match parse_semver(rest) {
            Some(v) => v,
            None => return Err(format!("约束 '{}' 的版本段 '{}' 不是合法的 x.y[.z] 版本", cond, rest.trim())),
        };
        reqs.push(VersionReq { op, ver });
    }
    if reqs.is_empty() {
        return Err("engine 约束为空".to_string());
    }
    Ok(reqs)
}

// ---------------------------------------------------------------------------
// PackManifest
// ---------------------------------------------------------------------------

/// import packs 装包清单（最小可用语义：解析 + 校验 + 登记到编译产物；
/// 包文件加载与组件实例化留 Phase 2/3）。
#[derive(Debug, Clone)]
pub struct PackManifest {
    /// 包名（`import packs "名"`）。
    pub name: String,
    /// engine 版本约束（C12：形如 `">=0.4, <0.5"`）；None = 无约束。
    pub engine: Option<String>,
    /// 声明提供的组件名（占位；Phase 2/3 接入组件实例化）。
    pub provides: Vec<String>,
    /// import 声明在 DSL 中的行号（TOML 路径下为 TOML 内行号）。
    pub line: usize,
}

impl PackManifest {
    /// 从 DSL `import packs` 块的字段列表构建清单。
    /// 识别 `engine`（字符串）与 `provides`（子块，字段名为组件名）；
    /// 其余字段登记忽略（宽容读取，Phase 2/3 扩展）。
    pub(crate) fn from_fields(
        name: &str,
        fields: &[NamedField],
        line: usize,
    ) -> Result<PackManifest, DslError> {
        let mut engine: Option<String> = None;
        let mut provides: Vec<String> = Vec::new();
        for f in fields {
            match (&f.value, f.name.as_str()) {
                (Some(FieldValue::Str(s)), "engine") => engine = Some(s.clone()),
                (Some(FieldValue::Block(sub)), "provides") => {
                    for sf in sub {
                        provides.push(sf.name.clone());
                    }
                }
                (Some(FieldValue::Str(_)), other) => {
                    // 其他字符串字段：登记忽略（宽容读取）。
                    let _ = other;
                }
                // engine 字段类型错误 → P4002（清单格式错误）。
                (_, "engine") => {
                    return Err(err_at(
                        f.line,
                        "[P4002] import packs 块中字段 'engine' 应为字符串\n  建议：写成 engine \">=0.4, <0.5\"（约束当前引擎版本范围）",
                    ));
                }
                _ => {}
            }
        }
        Ok(PackManifest { name: name.to_string(), engine, provides, line })
    }

    /// 当前引擎版本（编译期取自 crate 版本）。
    pub fn engine_version() -> SemVer {
        parse_semver(env!("CARGO_PKG_VERSION")).unwrap_or((0, 0, 0))
    }

    /// 校验 engine 约束（C12 / V9）。
    ///
    /// # Errors
    /// - 约束格式非法 → P4002（清单格式错误）；
    /// - 当前引擎版本不满足约束 → P4001（版本范围不满足）。
    pub fn check_engine(&self) -> Result<(), DslError> {
        let range: &str = match &self.engine {
            Some(r) => r,
            None => return Ok(()),
        };
        let reqs: Vec<VersionReq> = match parse_version_reqs(range) {
            Ok(r) => r,
            Err(detail) => {
                return Err(err_at(
                    self.line,
                    format!(
                        "[P4002] 包 '{}' 的 engine 约束 '{}' 格式非法：{}\n  建议：写成 \">=0.4, <0.5\" 形式（逗号分隔多条约束，每条 = 算子 + x.y[.z] 版本）",
                        self.name, range, detail
                    ),
                ));
            }
        };
        let current: SemVer = Self::engine_version();
        for req in reqs {
            if !req.satisfied_by(current) {
                return Err(err_at(
                    self.line,
                    format!(
                        "[P4001] 包 '{}' 要求 engine {}，当前引擎版本 {}.{}.{} 不满足\n  建议：升级引擎到要求的范围，或放宽该包清单的 engine 约束",
                        self.name, range, current.0, current.1, current.2
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 从最小 TOML 子集解析清单（包文件加载骨架）。
    ///
    /// 支持的形态：
    /// ```text
    /// name = "particle-kit"
    /// engine = ">=0.4, <0.5"
    ///
    /// [provides]
    /// emitter = "1.0"
    /// noise = "1.0"
    /// ```
    ///
    /// # Errors
    /// 行不是 `key = "value"` / `[段]` / 空行 / `#` 注释 → P4002。
    pub fn parse_toml(src: &str) -> Result<PackManifest, DslError> {
        let mut name: Option<String> = None;
        let mut engine: Option<String> = None;
        let mut provides: Vec<String> = Vec::new();
        let mut section: String = String::new();
        for (idx, raw) in src.lines().enumerate() {
            let line_no: usize = idx + 1;
            let line: &str = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                // [段] 头。
                let end: Option<usize> = rest.find(']');
                let sec: &str = match end {
                    Some(e) => &rest[..e],
                    None => {
                        return Err(err_at(
                            line_no,
                            "[P4002] 清单段头缺少右括号 ']'\n  建议：写成 [provides] 形式",
                        ));
                    }
                };
                section = sec.trim().to_string();
                continue;
            }
            let (key, value): (String, String) = match line.split_once('=') {
                Some((k, v)) => (k.trim().to_string(), v.trim().to_string()),
                None => {
                    return Err(err_at(
                        line_no,
                        format!(
                            "[P4002] 清单行 '{}' 不是 'key = \"value\"' 形式\n  建议：最小清单子集只支持 key = \"value\"、[段] 与 # 注释",
                            line
                        ),
                    ));
                }
            };
            // 值必须是双引号字符串（最小子集不支持数字 / 布尔）。
            let value: String = if value.len() >= 2 && value.starts_with('"') && value.ends_with('"')
            {
                value[1..value.len() - 1].to_string()
            } else {
                return Err(err_at(
                    line_no,
                    format!(
                        "[P4002] 清单键 '{}' 的值 {} 应为双引号字符串\n  建议：写成 {} = \"...\" 形式",
                        key, value, key
                    ),
                ));
            };
            if section == "provides" {
                provides.push(key);
            } else if key == "name" {
                name = Some(value);
            } else if key == "engine" {
                engine = Some(value);
            }
            // 其余键登记忽略（宽容读取）。
        }
        let name: String = name.unwrap_or_else(|| "unnamed-pack".to_string());
        Ok(PackManifest { name, engine, provides, line: 1 })
    }
}

// ---------------------------------------------------------------------------
// packs 模块自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_semver_variants() {
        assert_eq!(parse_semver("0.4"), Some((0, 4, 0)));
        assert_eq!(parse_semver("0.4.2"), Some((0, 4, 2)));
        assert_eq!(parse_semver(" 1.10.3 "), Some((1, 10, 3)));
        assert_eq!(parse_semver("abc"), None);
        assert_eq!(parse_semver("1.2.3.4"), None);
        assert_eq!(parse_semver(""), None);
    }

    #[test]
    fn version_reqs_parse_and_check() {
        let reqs: Vec<VersionReq> = parse_version_reqs(">=0.4, <0.5").expect("约束应可解析");
        assert_eq!(reqs.len(), 2);
        // 当前 (0,4,0)：满足 >=0.4，满足 <0.5。
        for r in &reqs {
            assert!(r.satisfied_by((0, 4, 0)), "约束 {:?} 应满足", r);
        }
        // (0,5,0)：满足 >=0.4 但不满足 <0.5。
        assert!(!reqs[1].satisfied_by((0, 5, 0)));
        // 格式非法。
        assert!(parse_version_reqs("0.4").is_err());
        assert!(parse_version_reqs(">=x.y").is_err());
    }

    #[test]
    fn manifest_from_fields_and_engine_check() {
        // 无约束：直接通过。
        let m = PackManifest::from_fields("kit", &[], 3).expect("空清单应可构建");
        assert!(m.check_engine().is_ok());
        assert_eq!(m.name, "kit");
        assert_eq!(m.line, 3);

        // engine 字段 + provides 子块。
        let fields: Vec<NamedField> = vec![
            NamedField {
                name: "engine".to_string(),
                value: Some(FieldValue::Str(">=0.1".to_string())),
                line: 4,
            },
            NamedField {
                name: "provides".to_string(),
                value: Some(FieldValue::Block(vec![
                    NamedField { name: "emitter".to_string(), value: Some(FieldValue::Str("1.0".to_string())), line: 5 },
                    NamedField { name: "noise".to_string(), value: Some(FieldValue::Str("1.0".to_string())), line: 6 },
                ])),
                line: 5,
            },
        ];
        let m = PackManifest::from_fields("fx-kit", &fields, 3).expect("清单应可构建");
        assert_eq!(m.engine.as_deref(), Some(">=0.1"));
        assert_eq!(m.provides, vec!["emitter".to_string(), "noise".to_string()]);
        assert!(m.check_engine().is_ok(), "当前引擎版本应满足 >=0.1");

        // engine 字段类型错误 → P4002。
        let bad: Vec<NamedField> = vec![NamedField {
            name: "engine".to_string(),
            value: Some(FieldValue::Num(1.0)),
            line: 4,
        }];
        let err: DslError =
            PackManifest::from_fields("kit", &bad, 3).expect_err("engine 类型错误应报 P4002");
        assert!(err.message.contains("P4002"), "应含 P4002，实际：{}", err.message);
    }

    #[test]
    fn manifest_engine_range_reject_and_format_error() {
        // 不可能满足的范围：当前版本不可能同时 >=99.0 且 <99.1（除非引擎真是 99）。
        let m = PackManifest {
            name: "future-kit".to_string(),
            engine: Some(">=99.0, <99.1".to_string()),
            provides: Vec::new(),
            line: 2,
        };
        let result: Result<(), DslError> = m.check_engine();
        match result {
            Ok(()) => panic!("当前引擎版本不应满足 >=99.0, <99.1"),
            Err(e) => {
                assert!(e.message.contains("P4001"), "应含 P4001，实际：{}", e.message);
                assert!(e.message.contains("建议"), "应含建议（C19），实际：{}", e.message);
            }
        }

        // 格式非法 → P4002。
        let m = PackManifest {
            name: "broken-kit".to_string(),
            engine: Some("what?".to_string()),
            provides: Vec::new(),
            line: 2,
        };
        let err: DslError = m.check_engine().expect_err("格式非法应报 P4002");
        assert!(err.message.contains("P4002"), "应含 P4002，实际：{}", err.message);
    }

    #[test]
    fn manifest_parse_toml_minimal() {
        let src: &str = concat!(
            "# 注释行\n",
            "name = \"particle-kit\"\n",
            "engine = \">=0.1, <0.2\"\n",
            "\n",
            "[provides]\n",
            "emitter = \"1.0\"\n",
            "noise = \"1.0\"\n"
        );
        let m: PackManifest = PackManifest::parse_toml(src).expect("最小 TOML 应可解析");
        assert_eq!(m.name, "particle-kit");
        assert_eq!(m.engine.as_deref(), Some(">=0.1, <0.2"));
        assert_eq!(m.provides, vec!["emitter".to_string(), "noise".to_string()]);
        assert!(m.check_engine().is_ok());

        // 非法行 → P4002。
        let err: DslError = PackManifest::parse_toml("name = particle\n")
            .expect_err("非引号值应报 P4002");
        assert!(err.message.contains("P4002"), "应含 P4002，实际：{}", err.message);
    }
}
