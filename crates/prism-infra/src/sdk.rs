//! SDK 稳定化：数据包（节点 / 插件）接口的语义化版本协商。
//!
//! 语义权威：D-8（插件 = 数据包，无动态代码加载）+ 插件接口语义化版本
//! （版本协商，不兼容明确报错）。本模块是 [`crate::plugin`] 数据包市场的
//! **接口版本闸门**：manifest 的 `engine` 约束在 add 时按当前引擎版本求值，
//! 不满足 ⇒ 明确拒绝（宪章 C20：不静默容错；C18 三段式 + C19 建议段）。
//!
//! # 版本口径
//! - [`ENGINE_VERSION`]：引擎**接口**版本（pack 契约的协商基准；唯一真源），
//!   与 DSL 侧 `prism_dsl::PackManifest::engine_version()` 保持同号——
//!   同一份 manifest 在市场侧与编译侧的协商结论必须一致，否则"能 add 的包
//!   编译不过"或反之。
//! - **兼容带**（semver，含 0.x 特例）：主版本不同 ⇒ 不兼容；主版本为 0
//!   时次版本不同 ⇒ 不兼容（0.x 的次版本就是破坏轴）；否则兼容。
//! - **约束文法**：逗号分隔的比较子合取，每条 = 算子（`>=` / `>` / `<=` /
//!   `<` / `=` / `==`，两字符优先）+ 版本（`x` / `x.y` / `x.y.z`，缺段补 0）。
//!   裸版本（无算子）**不合法**——与 DSL 侧 `parse_version_reqs` 同口径，
//!   防止市场能 add 的包在 DSL 侧编译不过。
//!
//! # 错误码
//! 与 DSL 侧装包校验同码同义：`P4001` = 版本范围不满足；`P4002` = 清单 /
//! 约束格式非法（见 `prism-dsl` 的 `packs.rs`）。错误码域是宪章级共享的，
//! 跨 crate 同码要求同义——故此处不另起段号。

use crate::plugin::Manifest;

/// 引擎接口版本（pack 契约的协商基准；与 workspace 各 crate 同号，当前
/// `0.1.0`）。改此常量等于宣告一次引擎接口发布——其测试与文档口径须同步。
pub const ENGINE_VERSION: SdkVersion = SdkVersion { major: 0, minor: 1, patch: 0 };

/// 语义化版本 `x.y.z`（定序可比较：数值比较，非字典序）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SdkVersion {
    /// 主版本（破坏轴）。
    pub major: u32,
    /// 次版本（主版本为 0 时充当破坏轴）。
    pub minor: u32,
    /// 修订号。
    pub patch: u32,
}

impl SdkVersion {
    /// 直接构造。
    pub const fn new(major: u32, minor: u32, patch: u32) -> SdkVersion {
        SdkVersion { major, minor, patch }
    }

    /// 宽松解析 `x` / `x.y` / `x.y.z`（缺段补 0；约束文法用，非法 ⇒ P4002）。
    ///
    /// 与 [`crate::plugin::parse_semver`]（严格 `x.y.z`）分工：清单 `version`
    /// 字段必须严格三段；`engine` 约束里的版本按惯例可写 `>=0.4` 两段。
    ///
    /// # Errors
    /// 空串 / 空段 / 非数字段 / 超过三段 → [`SdkError`]（P4002）。
    pub fn parse(text: &str) -> Result<SdkVersion, SdkError> {
        let trimmed: &str = text.trim();
        if trimmed.is_empty() {
            return Err(sdk_err("[P4002] 版本为空串\n  建议：写成 x.y.z（如 0.1.0）或 x.y（如 0.4）"));
        }
        let mut parts: [u32; 3] = [0, 0, 0];
        for (i, seg) in trimmed.split('.').enumerate() {
            if i >= 3 {
                return Err(sdk_err(format!(
                    "[P4002] 版本 '{}' 段数过多（最多 x.y.z 三段）",
                    trimmed
                )));
            }
            if seg.is_empty() {
                return Err(sdk_err(format!(
                    "[P4002] 版本 '{}' 含空段（形如 '0..1'）\n  建议：每段都是非负整数",
                    trimmed
                )));
            }
            parts[i] = seg
                .parse::<u32>()
                .map_err(|_| sdk_err(format!("[P4002] 版本段 '{}' 不是非负整数", seg)))?;
        }
        Ok(SdkVersion { major: parts[0], minor: parts[1], patch: parts[2] })
    }

    /// semver 兼容带判定：`self`（如 pack 面向版本）与 `other`（如引擎版本）
    /// 是否同带（主版本相同，且主版本为 0 时次版本也相同）。
    ///
    /// 0.x 特例的语义依据：semver 规范对 0.x 期的约定——次版本递增即可含
    /// 破坏变更，故 0.x 的兼容轴是次版本而非主版本。
    pub fn is_compatible_with(&self, other: &SdkVersion) -> bool {
        if self.major != other.major {
            return false;
        }
        if self.major == 0 {
            return self.minor == other.minor;
        }
        true
    }
}

impl std::fmt::Display for SdkVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// SDK / 版本协商错误（三段式消息；P4xxx 段，与 DSL 侧同码同义）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdkError {
    /// 面向人类的完整消息（含"建议："）。
    pub message: String,
}

impl std::fmt::Display for SdkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SdkError {}

fn sdk_err(message: impl Into<String>) -> SdkError {
    SdkError { message: message.into() }
}

/// 比较算子。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `>=`
    Ge,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `<`
    Lt,
    /// `=` / `==`
    Eq,
}

/// 单条版本约束（算子 + 版本），如 `>=0.1`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comparator {
    /// 比较算子。
    pub op: CmpOp,
    /// 比较基准版本（宽松解析，缺段补 0）。
    pub version: SdkVersion,
}

impl Comparator {
    /// 当前版本是否满足本比较子。
    fn satisfied_by(&self, current: SdkVersion) -> bool {
        match self.op {
            CmpOp::Ge => current >= self.version,
            CmpOp::Gt => current > self.version,
            CmpOp::Le => current <= self.version,
            CmpOp::Lt => current < self.version,
            CmpOp::Eq => current == self.version,
        }
    }

    /// 规范化文本（如 `>=0.1.0`）。
    pub fn describe(&self) -> String {
        let op: &str = match self.op {
            CmpOp::Ge => ">=",
            CmpOp::Gt => ">",
            CmpOp::Le => "<=",
            CmpOp::Lt => "<",
            CmpOp::Eq => "=",
        };
        format!("{op}{}", self.version)
    }
}

/// engine 版本约束（比较子合取：全部满足才算匹配）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VersionConstraint {
    /// 比较子列表（声明顺序）。
    pub comparators: Vec<Comparator>,
}

impl VersionConstraint {
    /// 解析约束串（如 `">=0.4, <0.5"`）。
    ///
    /// 文法：逗号分隔的比较子合取；每条 = 算子（`>=` / `<=` / `==` / `>` /
    /// `<` / `=`，两字符优先）+ 版本。裸版本（无算子）不合法（与 DSL 侧
    /// `parse_version_reqs` 同口径）。空约束（无任何比较子）⇒ [`SdkError`]
    /// ——声明了约束却是空串，属坏清单，不放行。
    ///
    /// # Errors
    /// 缺算子 / 版本段非法 / 空约束 → [`SdkError`]（P4002）。
    pub fn parse(text: &str) -> Result<VersionConstraint, SdkError> {
        let mut comparators: Vec<Comparator> = Vec::new();
        for cond in text.split(',') {
            let cond: &str = cond.trim();
            if cond.is_empty() {
                continue;
            }
            let (op, rest): (CmpOp, &str) = if let Some(rest) = cond.strip_prefix(">=") {
                (CmpOp::Ge, rest)
            } else if let Some(rest) = cond.strip_prefix("<=") {
                (CmpOp::Le, rest)
            } else if let Some(rest) = cond.strip_prefix("==") {
                (CmpOp::Eq, rest)
            } else if let Some(rest) = cond.strip_prefix('>') {
                (CmpOp::Gt, rest)
            } else if let Some(rest) = cond.strip_prefix('<') {
                (CmpOp::Lt, rest)
            } else if let Some(rest) = cond.strip_prefix('=') {
                (CmpOp::Eq, rest)
            } else {
                return Err(sdk_err(format!(
                    "[P4002] 约束 '{}' 缺少比较算子\n  建议：每条约束 = 算子 + 版本，如 \">=0.1, <0.2\"（算子限 >= / > / <= / < / =）",
                    cond
                )));
            };
            let version: SdkVersion = SdkVersion::parse(rest).map_err(|e| {
                sdk_err(format!("[P4002] 约束 '{}' 的版本段非法：{}\n  建议：版本写 x.y 或 x.y.z（缺段补 0）", cond, e.message))
            })?;
            comparators.push(Comparator { op, version });
        }
        if comparators.is_empty() {
            return Err(sdk_err(format!(
                "[P4002] engine 约束 '{}' 为空（没有任何比较子）\n  建议：删除 engine 字段（无约束），或写成 \">=0.1, <0.2\" 形式",
                text.trim()
            )));
        }
        Ok(VersionConstraint { comparators })
    }

    /// 某版本是否满足整条约束（合取：全部比较子通过）。
    pub fn matches(&self, version: SdkVersion) -> bool {
        self.comparators.iter().all(|c| c.satisfied_by(version))
    }

    /// 是否空约束（永真；当前 `parse` 拒绝空串，故此恒为 false，保留供未来
    /// "无约束" 语义显式化）。
    pub fn is_empty(&self) -> bool {
        self.comparators.is_empty()
    }

    /// 规范化文本（比较子以"、"连接，如 `>=0.1.0、<0.2.0`）。
    pub fn describe(&self) -> String {
        self.comparators.iter().map(Comparator::describe).collect::<Vec<_>>().join("、")
    }
}

/// 版本冲突（C18 三段式 + C19 建议段；P4001）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionConflict {
    /// 三段式一【在哪】。
    pub location: String,
    /// 三段式二【什么事】。
    pub detail: String,
    /// 三段式三【建议】。
    pub suggestion: String,
}

impl std::fmt::Display for VersionConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error[P4001] {location}\n  \
             什么事：{detail}\n  \
             建议：{suggestion}",
            location = self.location,
            detail = self.detail,
            suggestion = self.suggestion
        )
    }
}

/// 版本协商结论。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    /// manifest 未声明 `engine` 约束——隐式"任意版本"，放行。
    Unconstrained,
    /// 当前引擎接口版本满足约束。
    Compatible,
    /// 不满足：调用方**必须**明确报错（不得静默放行）。
    Incompatible(VersionConflict),
}

/// 约束文本 × 当前引擎版本 → 协商结论（无包名前缀，供直接复用）。
///
/// # Errors
/// 约束格式非法 → [`SdkError`]（P4002）。
pub fn check_constraint(
    pack: &str,
    constraint: &str,
    current: SdkVersion,
) -> Result<Compatibility, SdkError> {
    let parsed: VersionConstraint = VersionConstraint::parse(constraint)?;
    if parsed.matches(current) {
        return Ok(Compatibility::Compatible);
    }
    // 首个失败比较子（定位用）：全部失败时报第一个，符合"报最近事实"。
    let first_fail: &Comparator = parsed
        .comparators
        .iter()
        .find(|c| !c.satisfied_by(current))
        .unwrap_or(&parsed.comparators[0]);
    let band_note: &str = if current.is_compatible_with(&first_fail.version) {
        "（同兼容带内不满足——多半是约束窗口写窄了）"
    } else {
        "（约束窗口与当前版本不在同一兼容带——面向更高/更低接口版本的包）"
    };
    Ok(Compatibility::Incompatible(VersionConflict {
        location: format!("插件市场 add 闸门：包 '{}'", pack),
        detail: format!(
            "engine 约束 '{}'（规范化 {}）不被当前引擎接口版本 {} 满足——首个失败比较子：{}{}",
            constraint.trim(),
            parsed.describe(),
            current,
            first_fail.describe(),
            band_note
        ),
        suggestion: format!(
            "三选一：① 把引擎升级到满足 '{}' 的版本；② 放宽该包 manifest 的 engine \
             约束以包含 {}；③ 确认该包面向其他接口版本、暂不入库——数据包是纯数据，\
             接口不兼容时其预设 / 子图可能引用不存在的字段，故显式拒绝而非静默放行",
            constraint.trim(),
            current
        ),
    }))
}

/// 按 manifest 的 `engine` 约束协商当前引擎版本（[`ENGINE_VERSION`]）。
///
/// - 未声明约束 ⇒ [`Compatibility::Unconstrained`]；
/// - 约束非法 ⇒ `Err`（P4002：坏清单不许进库）；
/// - 不满足 ⇒ [`Compatibility::Incompatible`]（P4001：调用方明确报错）。
///
/// # Errors
/// 约束格式非法 → [`SdkError`]。
pub fn check_manifest(manifest: &Manifest) -> Result<Compatibility, SdkError> {
    match &manifest.engine {
        None => Ok(Compatibility::Unconstrained),
        Some(text) => check_constraint(&manifest.name, text, ENGINE_VERSION),
    }
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_accepts_lenient_segments() {
        assert_eq!(SdkVersion::parse("0.1.0").unwrap(), SdkVersion::new(0, 1, 0));
        assert_eq!(SdkVersion::parse(" 1.10.3 ").unwrap(), SdkVersion::new(1, 10, 3));
        // 缺段补 0（约束文法惯例）。
        assert_eq!(SdkVersion::parse("0.4").unwrap(), SdkVersion::new(0, 4, 0));
        assert_eq!(SdkVersion::parse("2").unwrap(), SdkVersion::new(2, 0, 0));
        // 非法一律 P4002。
        for bad in ["", "  ", "abc", "1.2.3.4", "0..1", "1.-2"] {
            let err: SdkError = SdkVersion::parse(bad).expect_err("坏版本必须拒绝");
            assert!(err.message.contains("P4002"), "'{bad}' 应报 P4002：{}", err.message);
        }
    }

    #[test]
    fn version_order_is_numeric_not_lexicographic() {
        assert!(SdkVersion::new(0, 9, 0) < SdkVersion::new(0, 10, 0));
        assert!(SdkVersion::new(0, 1, 0) < SdkVersion::new(0, 1, 1));
        assert!(SdkVersion::new(0, 9, 9) < SdkVersion::new(1, 0, 0));
    }

    #[test]
    fn compat_band_follows_semver_zero_x_rule() {
        let engine: SdkVersion = SdkVersion::new(0, 1, 0);
        // 同 0.1.x 带内兼容（次版本同轴）。
        assert!(engine.is_compatible_with(&SdkVersion::new(0, 1, 7)));
        // 0.x：次版本不同即破坏。
        assert!(!engine.is_compatible_with(&SdkVersion::new(0, 2, 0)));
        // 主版本为 1+：次版本 / 修订号不影响兼容。
        let v1: SdkVersion = SdkVersion::new(1, 0, 0);
        assert!(v1.is_compatible_with(&SdkVersion::new(1, 9, 9)));
        assert!(!v1.is_compatible_with(&SdkVersion::new(2, 0, 0)));
        assert!(!v1.is_compatible_with(&SdkVersion::new(0, 9, 9)));
    }

    #[test]
    fn constraint_parse_matches_conjunction() {
        let c: VersionConstraint = VersionConstraint::parse(">=0.1, <0.2").expect("应可解析");
        assert!(c.matches(SdkVersion::new(0, 1, 0)));
        assert!(c.matches(SdkVersion::new(0, 1, 99)));
        assert!(!c.matches(SdkVersion::new(0, 2, 0)), "上界不含");
        assert!(!c.matches(SdkVersion::new(0, 0, 9)), "下界不含");
        // 五种算子 + 裸版本（=）。
        assert!(VersionConstraint::parse(">0.1.0").unwrap().matches(SdkVersion::new(0, 1, 1)));
        assert!(!VersionConstraint::parse(">0.1.0").unwrap().matches(SdkVersion::new(0, 1, 0)));
        assert!(VersionConstraint::parse("<=0.2.0").unwrap().matches(SdkVersion::new(0, 2, 0)));
        assert!(VersionConstraint::parse("<0.2.0").unwrap().matches(SdkVersion::new(0, 1, 99)));
        assert!(VersionConstraint::parse("=0.1.0").unwrap().matches(SdkVersion::new(0, 1, 0)));
        assert!(!VersionConstraint::parse("=0.1.0").unwrap().matches(SdkVersion::new(0, 1, 1)));
        // 多约束合取：任一不满足即不匹配。
        let multi: VersionConstraint =
            VersionConstraint::parse(">=0.1, <0.2, <0.1.5").expect("应可解析");
        assert!(!multi.matches(SdkVersion::new(0, 1, 5)), "<0.1.5 不含上界");
        assert!(multi.describe().starts_with(">=0.1.0"), "规范化文本：{}", multi.describe());
    }

    #[test]
    fn constraint_parse_rejects_malformed_with_p4002() {
        // "0.1" = 裸版本（无算子）也不合法——与 DSL 侧 parse_version_reqs 同口径。
        for bad in ["0.1", ">=x.y", "what?", "", "  ", ", ,", ">= 0.1.0.0"] {
            let err: SdkError = VersionConstraint::parse(bad).expect_err("坏约束必须拒绝");
            assert!(err.message.contains("P4002"), "'{bad}' 应报 P4002：{}", err.message);
            assert!(err.message.contains("建议"), "'{bad}' 必含建议段（C19）：{}", err.message);
        }
    }

    #[test]
    fn negotiation_gates_incompatible_and_malformed_packs() {
        let current: SdkVersion = ENGINE_VERSION;
        // 兼容窗含当前版本。
        assert_eq!(
            check_constraint("ok-pack", ">=0.1, <0.2", current).unwrap(),
            Compatibility::Compatible
        );
        // 不满足 ⇒ Incompatible 且三段式带 P4001 + 建议。
        let conflict: VersionConflict = match check_constraint("future-pack", ">=99.0", current)
            .expect("格式合法不应 Err")
        {
            Compatibility::Incompatible(c) => c,
            other => panic!("应不兼容，实际 {other:?}"),
        };
        let text: String = conflict.to_string();
        assert!(text.contains("error[P4001]"), "须带错误码：{text}");
        assert!(text.contains("99.0.0"), "须含失败比较子：{text}");
        assert!(text.contains("建议："), "必含建议段（C19）：{text}");
        // 格式非法 ⇒ Err（P4002），不是 Incompatible。
        let err: SdkError =
            check_constraint("broken-pack", "banana", current).expect_err("坏约束必须 Err");
        assert!(err.message.contains("P4002"));
    }

    #[test]
    fn check_manifest_reads_optional_engine_field() {
        let base: Manifest = Manifest {
            name: "p".to_string(),
            version: (0, 1, 0),
            license: "MIT".to_string(),
            provides: vec!["a".to_string()],
            engine: None,
            assets: Vec::new(),
        };
        // 未声明 ⇒ Unconstrained（放行）。
        assert_eq!(check_manifest(&base).unwrap(), Compatibility::Unconstrained);
        // 声明且满足 ⇒ Compatible。
        let mut ok: Manifest = base.clone();
        ok.engine = Some(">=0.1, <0.2".to_string());
        assert_eq!(check_manifest(&ok).unwrap(), Compatibility::Compatible);
        // 声明且不满足 ⇒ Incompatible。
        let mut bad: Manifest = base.clone();
        bad.engine = Some(">=2.0".to_string());
        assert!(matches!(
            check_manifest(&bad).unwrap(),
            Compatibility::Incompatible(_)
        ));
        // 声明但空串 ⇒ Err（坏清单）。
        let mut broken: Manifest = base;
        broken.engine = Some("".to_string());
        assert!(check_manifest(&broken).is_err(), "空约束是坏清单，不许放行");
    }
}
