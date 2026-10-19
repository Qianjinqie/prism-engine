//! 数据包市场：manifest 校验（semver / 许可证 / 资产清单）+ 本地库管理
//! （add / list / remove）。
//!
//! 语义权威：（"数据包节点不是代码，是预制子图 + 参数预设
//! + 资产的纯数据"；"市场/分发 = 校验 manifest（semver、许可证、资产清单）后
//! 落库，无动态代码加载，安全面极小"）与 C6（"数据包市场校验 add/list/remove"）、
//! D-8（插件=数据包，无动态代码加载）、D-23（数据即组件）。
//!
//! # 包目录布局
//! ```text
//! mypack/
//!   manifest.toml          # name(kebab-case)/semver/license/provides/engine
//!   presets/*.prism        # 参数预设
//!   subgraphs/*.prism      # 预制子图
//!   assets/                # 字体子集 / .cube LUT / 分词词典 / cues
//! ```
//!
//! # 安全面
//! - **无动态代码加载**：本模块只读文件、只做校验与登记（记录包目录路径），
//!   绝不执行包内任何内容；
//! - manifest 校验失败（非法名 / 非法 semver / 许可证不在白名单 / 资产清单与
//!   实际不符）→ 结构化拒绝，不落库；
//! - **接口版本协商**（EC-3）：`engine` 约束按 `crate::sdk::ENGINE_VERSION`
//!   求值，不兼容 ⇒ P4001 明确拒绝（见 [`crate::sdk`]）。

use std::fs;
use std::path::{Path, PathBuf};

/// 许可证白名单（SPDX 标识符子集；"许可证白名单"）。
pub const LICENSE_ALLOWLIST: &[&str] = &[
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MPL-2.0",
    "Unlicense",
];

/// 数据包 manifest（解析自 `manifest.toml`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// 包名（kebab-case）。
    pub name: String,
    /// 语义化版本 `x.y.z`。
    pub version: (u32, u32, u32),
    /// 许可证（SPDX 标识符，须在白名单）。
    pub license: String,
    /// 提供的能力名（组件/预设名）。
    pub provides: Vec<String>,
    /// engine 版本约束（可空）。
    pub engine: Option<String>,
    /// 声明的资产相对路径（资产清单；须实际存在）。
    pub assets: Vec<String>,
}

/// 数据包校验 / 库操作错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackError {
    /// 面向人类的消息（含"建议："）。
    pub message: String,
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PackError {}

fn pack_err(message: impl Into<String>) -> PackError {
    PackError { message: message.into() }
}

/// 校验包名：kebab-case（小写字母 / 数字 / 连字符，字母开头）。
pub fn validate_name(name: &str) -> Result<(), PackError> {
    let ok: bool = !name.is_empty()
        && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if ok {
        Ok(())
    } else {
        Err(pack_err(format!(
            "包名 '{}' 非法：须为 kebab-case（小写字母开头，仅含小写字母/数字/连字符）\n  建议：改名如 'cinematic-basic'",
            name
        )))
    }
}

/// 解析并校验 semver `x.y.z`。
pub fn parse_semver(text: &str) -> Result<(u32, u32, u32), PackError> {
    let mut it = text.trim().split('.');
    let parse = |s: Option<&str>| -> Result<u32, PackError> {
        s.ok_or_else(|| pack_err(format!("版本 '{}' 段数不足（须 x.y.z）", text)))?
            .parse()
            .map_err(|_| pack_err(format!("版本 '{}' 含非数字段", text)))
    };
    let major = parse(it.next())?;
    let minor = parse(it.next())?;
    let patch = parse(it.next())?;
    if it.next().is_some() {
        return Err(pack_err(format!("版本 '{}' 段数过多（须恰好 x.y.z）", text)));
    }
    Ok((major, minor, patch))
}

/// 校验许可证是否在白名单内。
pub fn validate_license(license: &str) -> Result<(), PackError> {
    if LICENSE_ALLOWLIST.contains(&license) {
        Ok(())
    } else {
        Err(pack_err(format!(
            "许可证 '{}' 不在白名单内\n  建议：改用 {} 之一",
            license,
            LICENSE_ALLOWLIST.join(" / ")
        )))
    }
}

impl Manifest {
    /// 从 `manifest.toml` 文本解析（最小 TOML 子集：`key = "value"` / `[段]` / 数组）。
    ///
    /// 支持键：`name` / `version` / `license` / `engine`（字符串）；
    /// `provides`（数组）、`assets`（数组）。`[provides]` 段内键名亦计入 provides。
    ///
    /// # Errors
    /// 语法非法 / 缺必填键 → [`PackError`]。
    pub fn parse_toml(src: &str) -> Result<Manifest, PackError> {
        let mut name: Option<String> = None;
        let mut version: Option<(u32, u32, u32)> = None;
        let mut license: Option<String> = None;
        let mut engine: Option<String> = None;
        let mut provides: Vec<String> = Vec::new();
        let mut assets: Vec<String> = Vec::new();
        let mut section: String = String::new();

        for (idx, raw) in src.lines().enumerate() {
            let line_no: usize = idx + 1;
            let line: &str = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                let end: usize = rest
                    .find(']')
                    .ok_or_else(|| pack_err(format!("第 {} 行: 段头缺少 ']'", line_no)))?;
                section = rest[..end].trim().to_string();
                continue;
            }
            let (key, value): (&str, &str) = line
                .split_once('=')
                .ok_or_else(|| pack_err(format!("第 {} 行: '{}' 不是 'key = value' 形式", line_no, line)))?;
            let key: &str = key.trim();
            let value: &str = value.trim();
            if section == "provides" {
                provides.push(key.to_string());
                continue;
            }
            if value.starts_with('[') {
                let arr: Vec<String> = parse_string_array(value, line_no)?;
                match key {
                    "provides" => provides = arr,
                    "assets" => assets = arr,
                    other => return Err(pack_err(format!("第 {} 行: 未知数组键 '{}'", line_no, other))),
                }
            } else {
                let s: String = parse_toml_string(value, line_no)?;
                match key {
                    "name" => name = Some(s),
                    "version" => version = Some(parse_semver(&s)?),
                    "license" => license = Some(s),
                    "engine" => engine = Some(s),
                    other => return Err(pack_err(format!("第 {} 行: 未知键 '{}'", line_no, other))),
                }
            }
        }

        let name: String = name.ok_or_else(|| pack_err("manifest 缺少必填键 'name'"))?;
        let version = version.ok_or_else(|| pack_err("manifest 缺少必填键 'version'"))?;
        let license: String = license.ok_or_else(|| pack_err("manifest 缺少必填键 'license'"))?;
        Ok(Manifest { name, version, license, provides, engine, assets })
    }

    /// 校验 manifest 自身的静态字段（名字 / semver / 许可证 / provides 非空）。
    ///
    /// # Errors
    /// 任一字段非法 → [`PackError`]。
    pub fn validate_static(&self) -> Result<(), PackError> {
        validate_name(&self.name)?;
        validate_license(&self.license)?;
        // version 已在 parse 时校验；此处再核 provides 命名。
        if self.provides.is_empty() {
            return Err(pack_err(format!(
                "包 '{}' 未声明任何 provides 能力\n  建议：在 manifest 中列出该包提供的组件/预设名",
                self.name
            )));
        }
        for p in &self.provides {
            validate_name(p).map_err(|e| pack_err(format!("provides 项非法：{}", e.message)))?;
        }
        Ok(())
    }

    /// 校验资产清单：声明的每个资产路径必须真实存在（相对包根）。
    ///
    /// # Errors
    /// 资产缺失 → [`PackError`]。
    pub fn validate_assets(&self, pack_root: &Path) -> Result<(), PackError> {
        for asset in &self.assets {
            let path: PathBuf = pack_root.join(asset);
            if !path.exists() {
                return Err(pack_err(format!(
                    "包 '{}' 声明的资产 '{}' 不存在（期望路径 {}）\n  建议：补齐文件或从 assets 清单移除",
                    self.name,
                    asset,
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

fn parse_toml_string(value: &str, line_no: usize) -> Result<String, PackError> {
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        Ok(value[1..value.len() - 1].to_string())
    } else {
        Err(pack_err(format!("第 {} 行: 值 {} 应为双引号字符串", line_no, value)))
    }
}

fn parse_string_array(value: &str, line_no: usize) -> Result<Vec<String>, PackError> {
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(pack_err(format!("第 {} 行: 数组格式非法", line_no)));
    }
    let inner: &str = &value[1..value.len() - 1];
    let mut out: Vec<String> = Vec::new();
    for part in inner.split(',') {
        let p: &str = part.trim();
        if p.is_empty() {
            continue;
        }
        if p.len() >= 2 && p.starts_with('"') && p.ends_with('"') {
            out.push(p[1..p.len() - 1].to_string());
        } else {
            return Err(pack_err(format!("第 {} 行: 数组元素 '{}' 应为双引号字符串", line_no, p)));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 本地库管理（add / list / remove）
// ---------------------------------------------------------------------------

/// 本地库中的一条记录（包名 → 版本 + 安装路径）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryEntry {
    /// 包名。
    pub name: String,
    /// 版本 `x.y.z`。
    pub version: (u32, u32, u32),
    /// 许可证。
    pub license: String,
    /// 安装目录（绝对或相对路径）。
    pub path: PathBuf,
}

/// 本地包库（一份 `registry.toml` 索引 + 包目录）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackLibrary {
    /// 库根目录。
    pub root: PathBuf,
}

impl PackLibrary {
    /// 打开 / 创建库（不立即写盘，写盘在 [`add`] 时惰性发生）。
    pub fn new(root: impl Into<PathBuf>) -> Self {
        PackLibrary { root: root.into() }
    }

    /// 索引文件路径（`<root>/registry.toml`）。
    pub fn index_path(&self) -> PathBuf {
        self.root.join("registry.toml")
    }

    /// add：校验包目录（manifest 静态字段 + 资产清单）后登记到库索引。
    ///
    /// **无动态代码加载**：只读 manifest 与资产存在性，不执行包内容。
    ///
    /// 版本协商闸门（EC-3 SDK 稳定化）：manifest 的 `engine` 约束按
    /// [`crate::sdk::ENGINE_VERSION`] 求值——不满足 ⇒ 明确拒绝（P4001，三段式
    /// 含建议段；C20 不静默容错）；约束格式非法 ⇒ P4002。未声明 `engine`
    /// ⇒ 放行（隐式"任意版本"）。
    ///
    /// # Errors
    /// 包目录不存在 / manifest 缺失或非法 / 资产缺失 / 版本不兼容 /
    /// 约束格式非法 / 同名包已存在 → [`PackError`]。
    pub fn add(&self, pack_dir: &Path) -> Result<LibraryEntry, PackError> {
        if !pack_dir.is_dir() {
            return Err(pack_err(format!("包目录 '{}' 不存在", pack_dir.display())));
        }
        let manifest_path: PathBuf = pack_dir.join("manifest.toml");
        let text: String = fs::read_to_string(&manifest_path)
            .map_err(|e| pack_err(format!("无法读取 '{}': {}", manifest_path.display(), e)))?;
        let manifest: Manifest = Manifest::parse_toml(&text)?;
        manifest.validate_static()?;
        manifest.validate_assets(pack_dir)?;
        // 接口版本协商闸门：不兼容 / 坏约束都显式拒绝（不放行）。
        match crate::sdk::check_manifest(&manifest).map_err(|e| pack_err(e.message))? {
            crate::sdk::Compatibility::Incompatible(conflict) => {
                return Err(pack_err(conflict.to_string()));
            }
            crate::sdk::Compatibility::Compatible | crate::sdk::Compatibility::Unconstrained => {}
        }

        let mut entries: Vec<LibraryEntry> = self.list();
        if entries.iter().any(|e| e.name == manifest.name) {
            return Err(pack_err(format!(
                "包 '{}' 已存在于库中\n  建议：先 remove 旧版本再 add",
                manifest.name
            )));
        }
        let entry: LibraryEntry = LibraryEntry {
            name: manifest.name.clone(),
            version: manifest.version,
            license: manifest.license.clone(),
            path: pack_dir.to_path_buf(),
        };
        entries.push(entry.clone());
        self.write_index(&entries)?;
        Ok(entry)
    }

    /// list：读取库索引（不存在则空）。
    pub fn list(&self) -> Vec<LibraryEntry> {
        let path: PathBuf = self.index_path();
        let text: String = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        parse_index(&text)
    }

    /// remove：按包名从库索引移除。
    ///
    /// # Errors
    /// 包名不在库中 → [`PackError`]。
    pub fn remove(&self, name: &str) -> Result<LibraryEntry, PackError> {
        let mut entries: Vec<LibraryEntry> = self.list();
        let pos: usize = entries
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| pack_err(format!("包 '{}' 不在库中", name)))?;
        let removed: LibraryEntry = entries.remove(pos);
        self.write_index(&entries)?;
        Ok(removed)
    }

    fn write_index(&self, entries: &[LibraryEntry]) -> Result<(), PackError> {
        fs::create_dir_all(&self.root)
            .map_err(|e| pack_err(format!("无法创建库目录 '{}': {}", self.root.display(), e)))?;
        let mut text: String = String::from("# Prism 数据包库索引（自动生成）\n");
        for e in entries {
            text.push_str("[[pack]]\n");
            text.push_str(&format!("name = \"{}\"\n", e.name));
            text.push_str(&format!(
                "version = \"{}.{}.{}\"\n",
                e.version.0, e.version.1, e.version.2
            ));
            text.push_str(&format!("license = \"{}\"\n", e.license));
            text.push_str(&format!("path = \"{}\"\n\n", e.path.to_string_lossy()));
        }
        fs::write(self.index_path(), text)
            .map_err(|e| pack_err(format!("写入库索引失败: {}", e)))?;
        Ok(())
    }
}

fn parse_index(text: &str) -> Vec<LibraryEntry> {
    let mut entries: Vec<LibraryEntry> = Vec::new();
    let mut cur: Option<(String, (u32, u32, u32), String, PathBuf)> = None;
    for raw in text.lines() {
        let line: &str = raw.trim();
        if line == "[[pack]]" {
            if let Some((n, v, l, p)) = cur.take() {
                entries.push(LibraryEntry { name: n, version: v, license: l, path: p });
            }
            cur = Some((String::new(), (0, 0, 0), String::new(), PathBuf::new()));
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value): (&str, &str) = match line.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        let key: &str = key.trim();
        let value: &str = value.trim().trim_matches('"');
        if let Some((n, v, l, p)) = cur.as_mut() {
            match key {
                "name" => *n = value.to_string(),
                "version" => *v = parse_semver(value).unwrap_or((0, 0, 0)),
                "license" => *l = value.to_string(),
                "path" => *p = PathBuf::from(value),
                _ => {}
            }
        }
    }
    if let Some((n, v, l, p)) = cur.take() {
        entries.push(LibraryEntry { name: n, version: v, license: l, path: p });
    }
    entries
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
name = "cinematic-basic"
version = "0.4.2"
license = "MIT"
engine = ">=0.1, <0.2"
provides = ["grade-film", "bloom"]
assets = []
"#;

    #[test]
    fn parse_valid_manifest() {
        let m = Manifest::parse_toml(GOOD).expect("应可解析");
        assert_eq!(m.name, "cinematic-basic");
        assert_eq!(m.version, (0, 4, 2));
        assert_eq!(m.license, "MIT");
        assert_eq!(m.provides, vec!["grade-film", "bloom"]);
        m.validate_static().expect("静态校验应通过");
    }

    #[test]
    fn name_kebab_case_enforced() {
        assert!(validate_name("cinematic-basic").is_ok());
        assert!(validate_name("Cinematic").is_err(), "大写应拒绝");
        assert!(validate_name("a_b").is_err(), "下划线应拒绝");
        assert!(validate_name("").is_err());
    }

    #[test]
    fn semver_strict() {
        assert_eq!(parse_semver("1.2.3").unwrap(), (1, 2, 3));
        assert!(parse_semver("1.2").is_err());
        assert!(parse_semver("1.2.3.4").is_err());
        assert!(parse_semver("a.b.c").is_err());
    }

    #[test]
    fn license_allowlist_enforced() {
        assert!(validate_license("MIT").is_ok());
        assert!(validate_license("Proprietary").is_err());
    }

    #[test]
    fn bad_manifest_rejected() {
        let bad_license = "name = \"x-pack\"\nversion = \"1.0.0\"\nlicense = \"Nope\"\nprovides = [\"a\"]\n";
        let m = Manifest::parse_toml(bad_license).unwrap();
        assert!(m.validate_static().is_err(), "非法许可证应拒绝");

        let missing = "version = \"1.0.0\"\nlicense = \"MIT\"\n";
        assert!(Manifest::parse_toml(missing).is_err(), "缺 name 应拒绝");

        let no_provides = "name = \"x-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\n";
        let m = Manifest::parse_toml(no_provides).unwrap();
        assert!(m.validate_static().is_err(), "无 provides 应拒绝");
    }

    #[test]
    fn library_add_list_remove_round_trip() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-lib-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let pack_dir = tmp.join("mypack");
        fs::create_dir_all(&pack_dir).unwrap();
        fs::write(pack_dir.join("manifest.toml"), GOOD).unwrap();

        let lib = PackLibrary::new(tmp.join("lib"));
        assert!(lib.list().is_empty());

        let entry = lib.add(&pack_dir).expect("add 应成功");
        assert_eq!(entry.name, "cinematic-basic");
        let listed = lib.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].version, (0, 4, 2));

        // 重复 add 应拒绝。
        assert!(lib.add(&pack_dir).is_err());

        let removed = lib.remove("cinematic-basic").expect("remove 应成功");
        assert_eq!(removed.name, "cinematic-basic");
        assert!(lib.list().is_empty());
        assert!(lib.remove("cinematic-basic").is_err(), "二次 remove 应失败");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_rejects_missing_asset() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-asset-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let pack_dir = tmp.join("p");
        fs::create_dir_all(&pack_dir).unwrap();
        let manifest = "name = \"asset-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nprovides = [\"a\"]\nassets = [\"assets/font.ttf\"]\n";
        fs::write(pack_dir.join("manifest.toml"), manifest).unwrap();

        let lib = PackLibrary::new(tmp.join("lib"));
        let err = lib.add(&pack_dir).expect_err("缺资产应拒绝");
        assert!(err.message.contains("不存在"), "错误应指明资产缺失");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_accepts_present_asset() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-ok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let pack_dir = tmp.join("p");
        fs::create_dir_all(pack_dir.join("assets")).unwrap();
        fs::write(pack_dir.join("assets/font.ttf"), b"font").unwrap();
        let manifest = "name = \"asset-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nprovides = [\"a\"]\nassets = [\"assets/font.ttf\"]\n";
        fs::write(pack_dir.join("manifest.toml"), manifest).unwrap();

        let lib = PackLibrary::new(tmp.join("lib"));
        assert!(lib.add(&pack_dir).is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }

    // -----------------------------------------------------------------------
    // EC-3：接口版本协商闸门（add 时对 engine 约束求值）
    // -----------------------------------------------------------------------

    /// 写一个最小包（manifest + 可空资产）到临时目录。
    fn write_pack(tmp: &Path, dir_name: &str, manifest: &str) -> PathBuf {
        let pack_dir: PathBuf = tmp.join(dir_name);
        fs::create_dir_all(&pack_dir).unwrap();
        fs::write(pack_dir.join("manifest.toml"), manifest).unwrap();
        pack_dir
    }

    #[test]
    fn add_rejects_engine_incompatible_pack_explicitly() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-ver-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // 约束窗口不含当前引擎接口版本（>=99.0 永不满足）。
        let pack_dir: PathBuf = write_pack(
            &tmp,
            "future",
            "name = \"future-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nengine = \">=99.0\"\nprovides = [\"a\"]\n",
        );
        let lib = PackLibrary::new(tmp.join("lib"));
        let err = lib.add(&pack_dir).expect_err("版本不兼容必须拒绝");
        assert!(err.message.contains("P4001"), "须带版本错误码：{}", err.message);
        assert!(err.message.contains("建议"), "必含建议段（C19）：{}", err.message);
        // 拒绝即不落库（闸门在 write_index 之前）。
        assert!(lib.list().is_empty(), "被拒的包不得落库");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn add_rejects_malformed_engine_constraint() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-badver-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let pack_dir: PathBuf = write_pack(
            &tmp,
            "broken",
            "name = \"broken-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nengine = \"banana\"\nprovides = [\"a\"]\n",
        );
        let lib = PackLibrary::new(tmp.join("lib"));
        let err = lib.add(&pack_dir).expect_err("坏约束必须拒绝");
        assert!(err.message.contains("P4002"), "须带格式错误码：{}", err.message);
        assert!(lib.list().is_empty(), "被拒的包不得落库");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn add_accepts_engine_compatible_and_unconstrained_packs() {
        let tmp = std::env::temp_dir().join(format!("prism-pack-verok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let lib = PackLibrary::new(tmp.join("lib"));
        // 显式兼容窗口（含当前引擎版本）。
        let bounded: PathBuf = write_pack(
            &tmp,
            "bounded",
            "name = \"bounded-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nengine = \">=0.1, <0.2\"\nprovides = [\"a\"]\n",
        );
        assert!(lib.add(&bounded).is_ok(), "兼容窗口应放行");
        // 未声明 engine ⇒ 放行（GOOD 常量即此形态，主流程测试已覆盖，此处显式钉一次）。
        let open: PathBuf = write_pack(
            &tmp,
            "open",
            "name = \"open-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nprovides = [\"a\"]\n",
        );
        assert!(lib.add(&open).is_ok(), "无约束应放行");
        assert_eq!(lib.list().len(), 2);
        let _ = fs::remove_dir_all(&tmp);
    }
}
