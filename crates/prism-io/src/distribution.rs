//! 自定义生态分发层：数据包市场 + 特解包签名。
//!
//! 语义权威：D-23 自定义四级光谱、D-8（插件=数据包，
//! 无动态代码加载）、宪章 C13/R3（schema 版本 → 旧包只失效重建、永不误复用）、
//! C18/C19（错误三段式 + 必含建议段）、C20（默认全部报错，白名单豁免）、
//! D-16（Mini Code）、D-29（确定性数学件）。
//!
//! # 职责
//! - **包格式**：包名 / 语义化版本三元组 / 作者 / 依赖 / 能力 / 内容 + 内容哈希，
//!   自研二进制序列化（**不引 serde**——铁律）；
//! - **签名树**：复用 `prism-core/src/signature.rs` 的**签名树结构语义**
//!   （见下「签名树复用点」）；
//! - **注册表**：扫描目录 → 解析清单 → 建索引；按名 / 版本 / 能力查询；
//!   **索引序确定**（包名升序 + 版本降序，见 [`PackageIndex`]）；
//! - **依赖图**：循环依赖检测 + 拓扑序（确定性）。
//!
//! # 签名树复用点（与的依赖约束）
//!  规定依赖方向 `io∈L4 → core∈L2`（io **不得**依赖 core），
//! `prism-core/src/signature.rs` 的模块头亦明确记载了这条约束（并因此在 core
//! 内部自带一份 SHA-256）。故本模块**不 import core**，而是复用其**结构语义**：
//!
//! | `prism-core::signature` | 本模块 `distribution` 对应物 |
//! |---|---|
//! | canonical 字节：magic 头 + schema 版本 + 定长/长度前缀 + 大端 | [`Package::canonical_bytes`] 同构 |
//! | 域分隔标签 `PRISM-SIG-NODE` / `-GRAPH` | `PRISM-PKG-ENTRY` / `-ROOT` / `-DEP`（防跨域拼接歧义） |
//! | 自底向上子树哈希（`graph_signatures` → `graph_signature`） | 资产条目签名 → 包根签名（[`package_signature`]） |
//! | **增量性**：改一个节点只影响其自身到根的祖先链，兄弟子树逐位不变 | [`PackageIndex::root_signatures`] 同性质（见测试） |
//! | 子签名按**槽位升序**拼接 | 子签名按**条目名 / 依赖名升序**拼接（BTreeMap 键序，确定） |
//! | 哈希原语 SHA-256 | **直接 [`crate::sha256::sha256`]**（同 crate，直接复用） |
//!
//! # 诚实边界（勿高估）
//! - **不是密码学签名**。[`PackageSignature`] 是 `sha256(域标签 ++ 内容哈希 ++
//!   签名者标识)` 的**完整性 + 作者身份绑定校验值**。无公钥/私钥、无非对称
//!   算法、无证书链——那需要第三方 crate 或手写椭圆曲线（远超 D-16 范围）。
//!   它能发现**内容篡改**与**冒名**（冒用他人签名者标识可被发现，但攻击者
//!   可自称任何标识符，故**不防抵赖**）。真正的发布者认证需后续轮次。
//! - **不压缩、不加密**、无二进制资产分块（整包字节进内存）。
//! - **无网络协议**：注册表只扫本地目录，无远程市场、无索引服务。
//! - **无 SemVer 预发布/构建元数据**（`1.0.0-alpha` 不可表达；`parse` 遇到
//!   非纯数字段**拒收**，不做宽容解析——C20 白名单外）。
//! - 能力（capability）是**自由字符串**，无注册中心校验——D-23 术语不泄漏，
//!   故不预设能力词表；查不到即 `None`，不静默近似匹配。
//!
//! # 确定性契约
//! 全部序列化大端定长 / 显式长度前缀；依赖与能力在**构造时排序**（语义集合
//! 无序 → canonical 输出与插入序无关）；索引遍历走 `BTreeMap`（禁 HashMap
//! 迭代序决定运算顺序）。同输入两遍 `to_bytes()` 逐字节一致。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::sha256::sha256;

/// 256 位哈希（与 `prism-core::node::Hash256` 位模式同一）。
pub type Hash256 = [u8; 32];

// ===========================================================================
//  版本头与域分隔（宪章 C8/C13）
// ===========================================================================

/// 包格式 schema 版本。任何字节布局变更都必须递增（C13：旧包只失效重建，
/// 永不误复用）。
pub const PACKAGE_SCHEMA_VERSION: u32 = 1;

/// 包文件魔数（宪章 C8：内部格式带版本头；静默失效重建 ≠ 用户数据丢失）。
const PACKAGE_MAGIC: &[u8; 10] = b"PRISMPKG\0\0";

/// 数据包文件扩展名（注册表扫描时只认此扩展名）。
pub const PACKAGE_EXTENSION: &str = "prismpkg";

/// 域分隔标签：不同用途的哈希输入各挂唯一前缀，防跨域拼接歧义
/// （与 `signature.rs` 的 `NODE_TAG` / `GRAPH_TAG` 同思路）。
const ENTRY_TAG: &[u8; 15] = b"PRISM-PKG-ENTRY";
const ROOT_TAG: &[u8; 14] = b"PRISM-PKG-ROOT";
const DEPENDENCY_TAG: &[u8; 13] = b"PRISM-PKG-DEP";

// ===========================================================================
//  错误（宪章 C18 三段式；在哪 / 什么事 / 怎么改；C19：必含建议段）
// ===========================================================================

/// 分发层错误。错误码取 PD 段（prism-distribution 族；与 cache 清单的 PF2xxx
/// 独立前缀，避免两族混淆——同 `cache_manifest` 的分族理由）。
#[derive(Debug, Clone, PartialEq)]
pub struct PackageError {
    /// 错误码（PD1xxx 清单格式 / PD2xxx 完整性签名 / PD3xxx 依赖图）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 包名/路径。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for PackageError {}

/// 构造 PD1xxx（清单格式）错误。
fn format_error(
    location: impl Into<String>,
    problem: impl Into<String>,
    fix: impl Into<String>,
) -> PackageError {
    PackageError {
        code: "PD1001",
        location: location.into(),
        problem: problem.into(),
        fix: fix.into(),
    }
}

/// 构造 PD2xxx（完整性/签名）错误。
fn integrity_error(
    location: impl Into<String>,
    problem: impl Into<String>,
    fix: impl Into<String>,
) -> PackageError {
    PackageError {
        code: "PD2001",
        location: location.into(),
        problem: problem.into(),
        fix: fix.into(),
    }
}

/// 构造 PD3xxx（依赖图）错误。
fn dependency_error(
    location: impl Into<String>,
    problem: impl Into<String>,
    fix: impl Into<String>,
) -> PackageError {
    PackageError {
        code: "PD3001",
        location: location.into(),
        problem: problem.into(),
        fix: fix.into(),
    }
}

// ===========================================================================
//  语义化版本
// ===========================================================================

/// 语义化版本三元组 `(major, minor, patch)`（D-8 数据包版本）。
///
/// **只支持纯数字三段式**：预发布（`1.0.0-alpha`）与构建元数据
/// （`1.0.0+build`）**不可表达**，[`SemVer::parse`] 遇到即拒收（C20：
/// 白名单外不宽容解析）。比较用标准 SemVer 字典序：先 major，再 minor，再 patch。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SemVer {
    /// 主版本：不兼容改动。
    pub major: u32,
    /// 次版本：向后兼容的新增。
    pub minor: u32,
    /// 修订号：向后兼容的修复。
    pub patch: u32,
}

impl SemVer {
    /// 构造版本三元组。
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        SemVer {
            major,
            minor,
            patch,
        }
    }

    /// 解析 `major.minor.patch`（**必须**恰三段纯十进制数字，无前导 `+`/
    /// 无空白/无预发布后缀）。溢出 → 拒收。
    pub fn parse(text: &str) -> Result<Self, PackageError> {
        let location = "distribution.SemVer.parse";
        let bad = |problem: String| {
            format_error(
                location,
                problem,
                "用纯三段十进制数字，如 1.0.0；本实现不支持预发布/构建元数据后缀",
            )
        };
        let parts: Vec<&str> = text.split('.').collect();
        if parts.len() != 3 {
            return Err(bad(format!(
                "版本串 {text:?} 有 {} 段，应恰为 3 段",
                parts.len()
            )));
        }
        let mut nums = [0u32; 3];
        for (i, p) in parts.iter().enumerate() {
            if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad(format!(
                    "版本段 {:?}（第 {} 段）不是纯十进制数字",
                    p,
                    i + 1
                )));
            }
            nums[i] = p
                .parse::<u32>()
                .map_err(|_| bad(format!("版本段 {:?} 超出 u32 范围", p)))?;
        }
        Ok(SemVer::new(nums[0], nums[1], nums[2]))
    }

    /// 笛卡尔兼容判断（caret 语义，`^req`）：`major` 相同即兼容。
    ///
    /// `req.major == 0` 时按 minor 锁定（`^0.3.0` 接受 `0.3.x`，不接受 `0.4.0`），
    /// 与 Cargo 的前导零约定一致。
    pub fn is_compatible_with(&self, req: &SemVer) -> bool {
        if self.major != req.major {
            return false;
        }
        if req.major == 0 && self.minor != req.minor {
            return false;
        }
        self >= req
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// 版本要求 = 最低版本 + caret 兼容上界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct VersionReq {
    /// 最低可接受版本（含）。
    pub min: SemVer,
}

impl VersionReq {
    /// 构造版本要求。
    pub const fn at_least(min: SemVer) -> Self {
        VersionReq { min }
    }

    /// 候选版本是否满足本要求。
    pub fn matches(&self, v: &SemVer) -> bool {
        v.is_compatible_with(&self.min)
    }
}

// ===========================================================================
//  包内容与清单
// ===========================================================================

/// 资产条目：数据包内一个命名 blob（字体 / LUT / 词典 / 预置子图字节 …）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct AssetEntry {
    /// 条目名（包内唯一；注册表按名升序拼子签名）。
    pub name: String,
    /// 条目字节。
    pub data: Vec<u8>,
}

impl AssetEntry {
    /// 构造资产条目。
    pub fn new(name: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        AssetEntry {
            name: name.into(),
            data: data.into(),
        }
    }

    /// 条目 canonical 字节（供签名树叶子用）。
    ///
    /// 布局：`长度前缀(u32 BE) ++ 名 UTF-8 ++ 长度前缀(u64 BE) ++ 数据`。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.name.len() + 8 + self.data.len());
        out.extend_from_slice(&(self.name.len() as u32).to_be_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&(self.data.len() as u64).to_be_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// 条目签名 = `sha256(ENTRY_TAG ++ schema ++ canonical 字节)`（签名树叶子）。
    pub fn signature(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(ENTRY_TAG.len() + 4 + self.data.len() + 16);
        buf.extend_from_slice(ENTRY_TAG);
        buf.extend_from_slice(&PACKAGE_SCHEMA_VERSION.to_be_bytes());
        buf.extend_from_slice(&self.canonical_bytes());
        sha256(&buf)
    }
}

/// 依赖声明：本包需要的另一包 + 版本要求。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dependency {
    /// 被依赖的包名。
    pub name: String,
    /// 版本要求（caret 兼容）。
    pub req: VersionReq,
}

impl Dependency {
    /// 构造依赖声明。
    pub fn new(name: impl Into<String>, min: SemVer) -> Self {
        Dependency {
            name: name.into(),
            req: VersionReq::at_least(min),
        }
    }
}

/// 数据包清单（D-8「数据包 = 纯数据不含可执行代码」）。
///
/// **构造即规范化**：依赖与能力在构造时排序（语义集合无序 → canonical 输出
/// 与插入序无关），资产条目在 [`Package::new`] 中按名升序落位。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Package {
    /// 包名（市场唯一标识）。
    pub name: String,
    /// 版本。
    pub version: SemVer,
    /// 作者 / 发布者标识（签名者标识的默认值）。
    pub author: String,
    /// 描述（可空串）。
    pub description: String,
    /// 依赖（按名升序）。
    pub dependencies: Vec<Dependency>,
    /// 能力标签（字典序；D-23 术语不泄漏，故为自由字符串无词表校验）。
    pub capabilities: Vec<String>,
    /// 资产条目（按名升序）。
    pub assets: Vec<AssetEntry>,
    /// 内容哈希 = `sha256(canonical_bytes())`。
    pub content_hash: Hash256,
}

impl Package {
    /// 构造数据包并计算内容哈希（依赖/能力/资产全部规范化排序）。
    pub fn new(
        name: impl Into<String>,
        version: SemVer,
        author: impl Into<String>,
        description: impl Into<String>,
        dependencies: Vec<Dependency>,
        capabilities: Vec<String>,
        assets: Vec<AssetEntry>,
    ) -> Self {
        let mut pkg = Package {
            name: name.into(),
            version,
            author: author.into(),
            description: description.into(),
            dependencies,
            capabilities,
            assets,
            content_hash: [0u8; 32],
        };
        pkg.normalize();
        pkg.content_hash = sha256(&pkg.canonical_bytes());
        pkg
    }

    /// 规范化：依赖按名升序去重同名、能力去重排序、资产按名升序。
    ///
    /// 同名依赖取**版本要求较高者**（`max`）——两个声明要求不同版本时，
    /// 满足较高者必然满足较低者，故取高者是语义无损的合并。
    fn normalize(&mut self) {
        self.dependencies.sort_by(|a, b| a.name.cmp(&b.name));
        let mut merged: Vec<Dependency> = Vec::with_capacity(self.dependencies.len());
        for d in self.dependencies.drain(..) {
            match merged.iter_mut().find(|m| m.name == d.name) {
                Some(existing) => {
                    if d.req.min > existing.req.min {
                        existing.req = d.req;
                    }
                }
                None => merged.push(d),
            }
        }
        self.dependencies = merged;

        let caps: BTreeSet<String> = self.capabilities.iter().cloned().collect();
        self.capabilities = caps.into_iter().collect();

        self.assets.sort_by(|a, b| a.name.cmp(&b.name));
        self.assets.dedup_by(|a, b| a.name == b.name);
    }

    /// 清单 canonical 字节（**不含** `content_hash` 本身——否则自指）。
    ///
    /// 布局（与 `signature.rs::graph_canonical_bytes` 同构：magic + schema +
    /// 计数 + 各字段长度前缀/大端）：
    /// ```text
    /// magic(10) ++ schema(u32)
    /// ++ name_len(u32) ++ name
    /// ++ major(u32) ++ minor(u32) ++ patch(u32)
    /// ++ author_len(u32) ++ author
    /// ++ desc_len(u32) ++ description
    /// ++ dep_count(u32) ++ 各依赖(name_len ++ name ++ min 三段 u32)
    /// ++ cap_count(u32) ++ 各能力(name_len ++ name)
    /// ++ asset_count(u32) ++ 各资产 canonical 字节
    /// ```
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PACKAGE_MAGIC);
        out.extend_from_slice(&PACKAGE_SCHEMA_VERSION.to_be_bytes());
        append_str(&mut out, &self.name);
        out.extend_from_slice(&self.version.major.to_be_bytes());
        out.extend_from_slice(&self.version.minor.to_be_bytes());
        out.extend_from_slice(&self.version.patch.to_be_bytes());
        append_str(&mut out, &self.author);
        append_str(&mut out, &self.description);
        out.extend_from_slice(&(self.dependencies.len() as u32).to_be_bytes());
        for d in &self.dependencies {
            append_str(&mut out, &d.name);
            out.extend_from_slice(&d.req.min.major.to_be_bytes());
            out.extend_from_slice(&d.req.min.minor.to_be_bytes());
            out.extend_from_slice(&d.req.min.patch.to_be_bytes());
        }
        out.extend_from_slice(&(self.capabilities.len() as u32).to_be_bytes());
        for c in &self.capabilities {
            append_str(&mut out, c);
        }
        out.extend_from_slice(&(self.assets.len() as u32).to_be_bytes());
        for a in &self.assets {
            out.extend_from_slice(&a.canonical_bytes());
        }
        out
    }

    /// 导出确定性字节（`canonical_bytes()` ++ `content_hash`）。
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.canonical_bytes();
        out.extend_from_slice(&self.content_hash);
        out
    }

    /// 从字节解析并**校验内容哈希**（哈希不符 → 结构化错误，绝不静默接受）。
    ///
    /// 这就是「内容被改过」的检出点：任何人改动包内任一字节（清单字段或资产
    /// 数据）都会使重算哈希与尾部存储值不符。
    pub fn from_bytes(data: &[u8]) -> Result<Self, PackageError> {
        let location = "distribution.Package.from_bytes";
        let bad = |problem: String, fix: &str| format_error(location, problem, fix);

        let mut r = Reader::new(data, location);
        r.expect_magic(PACKAGE_MAGIC, "包文件魔数不匹配，非本引擎数据包")?;
        r.expect_schema(PACKAGE_SCHEMA_VERSION)?;

        let name = r.read_str("包名")?;
        let version = SemVer::new(
            r.read_u32("major")?,
            r.read_u32("minor")?,
            r.read_u32("patch")?,
        );
        let author = r.read_str("作者")?;
        let description = r.read_str("描述")?;

        let dep_count = r.read_u32("依赖数")?;
        let mut dependencies = Vec::with_capacity((dep_count as usize).min(1024));
        for i in 0..dep_count {
            let dep_name = r.read_str("依赖名")?;
            let min = SemVer::new(
                r.read_u32("依赖 major")?,
                r.read_u32("依赖 minor")?,
                r.read_u32("依赖 patch")?,
            );
            if dep_name.is_empty() {
                return Err(bad(
                    format!("第 {i} 条依赖名为空串"),
                    "依赖名不得为空字符串",
                ));
            }
            dependencies.push(Dependency {
                name: dep_name,
                req: VersionReq::at_least(min),
            });
        }

        let cap_count = r.read_u32("能力数")?;
        let mut capabilities = Vec::with_capacity((cap_count as usize).min(1024));
        for _ in 0..cap_count {
            capabilities.push(r.read_str("能力名")?);
        }

        let asset_count = r.read_u32("资产数")?;
        let mut assets = Vec::with_capacity((asset_count as usize).min(1024));
        for _ in 0..asset_count {
            let aname = r.read_str("资产名")?;
            let alen_u64 = r.read_u64("资产数据长度")?;
            // u64 → usize 用 checked 转换（宪章 C18：禁 `unwrap()` 转换）。
            let alen = usize::try_from(alen_u64).map_err(|_| {
                format_error(
                    location,
                    format!("资产 `{aname}` 数据长度 {alen_u64} 超出本平台 usize 范围"),
                    "数据包文件已损坏或来自不兼容平台；删除后从可信来源重新获取",
                )
            })?;
            let bytes = r.read_bytes(alen, "资产数据")?;
            assets.push(AssetEntry {
                name: aname,
                data: bytes.to_vec(),
            });
        }

        let stored = r.read_exact_32("内容哈希")?;
        r.finish()?;

        let pkg = Package {
            name,
            version,
            author,
            description,
            dependencies,
            capabilities,
            assets,
            content_hash: stored,
        };
        // 解析出的依赖/能力保持文件内序（往返必须逐字节一致），故**不** normalize；
        // 只重算哈希做完整性校验。
        let recomputed = sha256(&pkg.canonical_bytes());
        if recomputed != stored {
            return Err(integrity_error(
                location,
                format!(
                    "包 `{}` 内容哈希不匹配：存储 {:02x?}，重算 {:02x?}（内容已被改动或损坏）",
                    pkg.name,
                    &stored[..8],
                    &recomputed[..8]
                ),
                "重新从可信来源获取该数据包；不要使用哈希不符的包（宪章 C20：不静默接受）",
            ));
        }
        Ok(pkg)
    }

    /// 验签：内容哈希 + 签名者标识 → 签名值（**完整性 + 作者绑定**，非密码学签名）。
    ///
    /// `PackageSignature::compute` 与本方法同式，故本方法恒真——它存在的意义
    /// 是让调用方有一处**语义明确**的验签入口（未来接入真签名时可在此替换，
    /// 签名树结构不变）。
    pub fn verify(&self, signer: &str) -> PackageSignature {
        PackageSignature::compute(&self.content_hash, signer)
    }

    /// 包根签名（签名树内部节点）=
    /// `sha256(ROOT_TAG ++ schema ++ name ++ version ++ Σ 资产条目签名（名升序) )`。
    ///
    /// 对应 `signature.rs::graph_signature` 的自底向上汇总。
    pub fn root_signature(&self) -> Hash256 {
        let mut buf =
            Vec::with_capacity(ROOT_TAG.len() + 4 + self.name.len() + 12 + self.assets.len() * 32);
        buf.extend_from_slice(ROOT_TAG);
        buf.extend_from_slice(&PACKAGE_SCHEMA_VERSION.to_be_bytes());
        append_str(&mut buf, &self.name);
        buf.extend_from_slice(&self.version.major.to_be_bytes());
        buf.extend_from_slice(&self.version.minor.to_be_bytes());
        buf.extend_from_slice(&self.version.patch.to_be_bytes());
        for a in &self.assets {
            buf.extend_from_slice(&a.signature());
        }
        sha256(&buf)
    }
}

/// 包签名：内容哈希与签名者标识的绑定校验值。
///
/// **诚实边界**：这是 `sha256(域标签 ++ content_hash ++ signer)`，**不是**公钥
/// 签名——无密钥体系、无非对称算法。作用范围：检出内容篡改 + 绑定声明的签名者
/// 标识。**不防抵赖**（可自称任意标识符）。见模块头诚实边界。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageSignature {
    /// 签名者标识（作者 / 发布者键 id）。
    pub signer: Hash256,
    /// 签名值。
    pub value: Hash256,
}

impl PackageSignature {
    /// 由内容哈希 + 签名者标识计算签名。
    pub fn compute(content_hash: &Hash256, signer_id: &str) -> Self {
        let signer = sha256(signer_id.as_bytes());
        let mut buf = Vec::with_capacity(ROOT_TAG.len() + 32 + 32);
        buf.extend_from_slice(ROOT_TAG);
        buf.extend_from_slice(content_hash);
        buf.extend_from_slice(&signer);
        PackageSignature {
            signer,
            value: sha256(&buf),
        }
    }

    /// 验签：与给定内容哈希 + 签名者标识重算并比对（恒等比较，无短路）。
    pub fn verify_against(&self, content_hash: &Hash256, signer_id: &str) -> bool {
        let expected = PackageSignature::compute(content_hash, signer_id);
        // 逐字节比较（不依赖 `==` 的提前退出语义，满足确定性铁律的可审计性）。
        let mut diff = 0u8;
        for (a, b) in self.value.iter().zip(expected.value.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

// ===========================================================================
//  字节读写辅助（自研序列化；零依赖、不 panic）
// ===========================================================================

/// 追加长度前缀字符串（`len(u32 BE) ++ UTF-8`）。
fn append_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// 字节游标：越界一律返回结构化错误，**绝不 panic / 越界索引**
/// （宪章 C18；`data[a..b]` 形式会 panic，故全部走 `get`）。
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    location: &'static str,
}

impl<'a> Reader<'a> {
    /// 构造游标。
    fn new(data: &'a [u8], location: &'static str) -> Self {
        Reader {
            data,
            pos: 0,
            location,
        }
    }

    /// 截断错误。
    fn truncated(&self, need: usize) -> PackageError {
        format_error(
            self.location,
            format!(
                "字节不足：偏移 {} 处需 {need} 字节，仅剩 {} 字节",
                self.pos,
                self.data.len().saturating_sub(self.pos)
            ),
            "数据包文件被截断或不是本格式；删除后从可信来源重新获取",
        )
    }

    /// 读取 `n` 字节。
    fn read_bytes(&mut self, n: usize, what: &str) -> Result<&'a [u8], PackageError> {
        let end = self.pos.checked_add(n).ok_or_else(|| self.truncated(n))?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| {
            format_error(
                self.location,
                format!(
                    "{what}区越界：偏移 {} 读 {n} 字节，数据全长 {}",
                    self.pos,
                    self.data.len()
                ),
                "数据包文件被截断或不是本格式；删除后从可信来源重新获取",
            )
        })?;
        self.pos = end;
        Ok(slice)
    }

    /// 校验魔数。
    fn expect_magic(&mut self, expect: &[u8], msg: &str) -> Result<(), PackageError> {
        let got = self.read_bytes(expect.len(), "魔数")?;
        if got != expect {
            return Err(format_error(
                self.location,
                format!(
                    "{msg}（读到 {:02x?}，期望 {:02x?}）",
                    &got[..got.len().min(4)],
                    &expect[..4]
                ),
                "确认文件是本引擎数据包；非本格式直接删除后重新获取",
            ));
        }
        Ok(())
    }

    /// 校验 schema 版本（不符 → 拒收，C13：旧包只失效重建）。
    fn expect_schema(&mut self, expect: u32) -> Result<(), PackageError> {
        let got = self.read_u32("schema 版本")?;
        if got != expect {
            return Err(format_error(
                self.location,
                format!("schema 版本不符：文件 {got}，当前 {expect}"),
                "旧版本数据包不误复用（宪章 C13）；用当前引擎重新导出该包",
            ));
        }
        Ok(())
    }

    /// 读取大端 u32。
    fn read_u32(&mut self, what: &str) -> Result<u32, PackageError> {
        let b = self.read_bytes(4, what)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(b);
        Ok(u32::from_be_bytes(a))
    }

    /// 读取大端 u64。
    fn read_u64(&mut self, what: &str) -> Result<u64, PackageError> {
        let b = self.read_bytes(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    /// 读取长度前缀字符串。
    fn read_str(&mut self, what: &str) -> Result<String, PackageError> {
        let len = self.read_u32(what)? as usize;
        let bytes = self.read_bytes(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            format_error(
                self.location,
                format!("{what}不是合法 UTF-8（{len} 字节）"),
                "数据包文件已损坏；删除后从可信来源重新获取",
            )
        })
    }

    /// 读取定长 32 字节。
    fn read_exact_32(&mut self, what: &str) -> Result<Hash256, PackageError> {
        let b = self.read_bytes(32, what)?;
        let mut a = [0u8; 32];
        a.copy_from_slice(b);
        Ok(a)
    }

    /// 收尾：必须恰好读完（尾部多余字节 → 拒收，防拼接歧义）。
    fn finish(&self) -> Result<(), PackageError> {
        if self.pos != self.data.len() {
            return Err(format_error(
                self.location,
                format!("尾部多余 {} 字节未消费", self.data.len() - self.pos),
                "数据包文件格式不合法；删除后从可信来源重新获取",
            ));
        }
        Ok(())
    }
}

// ===========================================================================
//  注册表与索引
// ===========================================================================

/// 索引条目：一个已解析并验过哈希的包 + 其路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    /// 源文件路径。
    pub path: PathBuf,
    /// 包清单。
    pub package: Package,
}

/// 包索引：**遍历序确定** = (包名升序, 版本降序)。
///
/// 版本降序意味着「同名包的最新版本在前」——查询 [`PackageIndex::latest`] 取
/// 首项即可，无需二次排序。底层用 [`BTreeMap`] + 逆序 `Vec`，**禁 HashMap
/// 迭代序**决定运算顺序（确定性铁律）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PackageIndex {
    /// 包名 → 该名各版本（版本降序）。
    by_name: BTreeMap<String, Vec<Package>>,
    /// 能力 → 提供该能力的包（(名, 版本) 升序）。
    by_capability: BTreeMap<String, BTreeSet<(String, SemVer)>>,
    /// 根签名 → (包名, 版本)，用于增量性验证。
    by_root_sig: BTreeMap<Hash256, Vec<(String, SemVer)>>,
}

/// 索引中的一个位置引用。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackageId {
    /// 包名。
    pub name: String,
    /// 版本。
    pub version: SemVer,
}

impl PackageId {
    /// 构造位置引用。
    pub fn new(name: impl Into<String>, version: SemVer) -> Self {
        PackageId {
            name: name.into(),
            version,
        }
    }
}

impl PackageIndex {
    /// 构造空索引。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一个包（**规范化入表**：同 `(名, 版本)` 重复 → 结构化报错）。
    ///
    /// # 错误
    /// 同名同版本重复登记 → PD1001（不静默覆盖，C20）。
    pub fn insert(&mut self, entry: IndexEntry) -> Result<(), PackageError> {
        let id = PackageId::new(entry.package.name.clone(), entry.package.version);
        let versions = self.by_name.entry(id.name.clone()).or_default();
        if versions.iter().any(|p| p.version == id.version) {
            return Err(format_error(
                format!(
                    "distribution.PackageIndex.insert（包 {} {}）",
                    id.name, id.version
                ),
                format!(
                    "包 `{}` 版本 {} 已在索引中（来源 {:?}）",
                    id.name, id.version, entry.path
                ),
                "同名同版本只应有一份；移除重复的数据包文件，或给其中一个改版本号",
            ));
        }
        versions.push(entry.package.clone());
        // 版本降序（最新在前）。
        versions.sort_by(|a, b| b.version.cmp(&a.version));

        for cap in &entry.package.capabilities {
            self.by_capability
                .entry(cap.clone())
                .or_default()
                .insert((id.name.clone(), id.version));
        }
        self.by_root_sig
            .entry(entry.package.root_signature())
            .or_default()
            .push((id.name.clone(), id.version));
        Ok(())
    }

    /// 索引内包总数（同名多版本各计一次）。
    pub fn len(&self) -> usize {
        self.by_name.values().map(Vec::len).sum()
    }

    /// 索引是否为空。
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// 不同包名数。
    pub fn distinct_names(&self) -> usize {
        self.by_name.len()
    }

    /// 全部位置引用，**按 (名升序, 版本降序)**——索引序的规范表述。
    pub fn ids(&self) -> Vec<PackageId> {
        let mut out = Vec::with_capacity(self.len());
        for (name, versions) in &self.by_name {
            for p in versions {
                out.push(PackageId::new(name.clone(), p.version));
            }
        }
        out
    }

    /// 按包名取全部版本（版本降序）。
    pub fn versions_of(&self, name: &str) -> &[Package] {
        self.by_name.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// 按精确 (名, 版本) 取包。
    pub fn get(&self, id: &PackageId) -> Option<&Package> {
        self.by_name
            .get(&id.name)?
            .iter()
            .find(|p| p.version == id.version)
    }

    /// 按名取**最新**版本。
    pub fn latest(&self, name: &str) -> Option<&Package> {
        self.by_name.get(name).and_then(|v| v.first())
    }

    /// 按能力查询全部包（结果按 (名升序, 版本降序)）。
    ///
    /// 能力是自由字符串，**精确匹配**——查不到即空，**不静默近似/前缀匹配**
    /// （C20 白名单外；D-23 术语不泄漏，不预设能力词表）。
    pub fn find_by_capability(&self, cap: &str) -> Vec<PackageId> {
        let Some(set) = self.by_capability.get(cap) else {
            return Vec::new();
        };
        // `BTreeSet` 按 (名, 版本) 升序 → 需按版本降序重排以符合索引序规范。
        let mut ids: Vec<PackageId> = set
            .iter()
            .map(|(n, v)| PackageId::new(n.clone(), *v))
            .collect();
        ids.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| b.version.cmp(&a.version)));
        ids
    }

    /// 全部能力标签（字典序）。
    pub fn capabilities(&self) -> Vec<&str> {
        self.by_capability.keys().map(String::as_str).collect()
    }

    /// 解析依赖闭包中的全部位置引用（按索引序返回）。
    ///
    /// 语义：从 `roots` 出发按依赖名取**最新**版本，逐层展开。缺依赖 /
    /// 版本不兼容 → 结构化报错（PD3xxx），**不静默跳过**（C20）。
    pub fn resolve_closure(&self, roots: &[PackageId]) -> Result<Vec<PackageId>, PackageError> {
        let mut seen: BTreeSet<PackageId> = BTreeSet::new();
        let mut queue: Vec<PackageId> = roots.to_vec();
        queue.sort_unstable();
        while let Some(id) = queue.pop() {
            if seen.contains(&id) {
                continue;
            }
            let pkg = self.get(&id).ok_or_else(|| {
                dependency_error(
                    "distribution.PackageIndex.resolve_closure",
                    format!("依赖闭包中找不到包 `{}` 版本 {}", id.name, id.version),
                    format!(
                        "安装该包，或从 roots 中移除对它的引用；可用能力查询：{:?}",
                        self.capabilities()
                    ),
                )
            })?;
            for dep in &pkg.dependencies {
                // 依赖名解析到最新版本，再检查 caret 兼容性。
                let candidate = self.latest(&dep.name).ok_or_else(|| {
                    dependency_error(
                        "distribution.PackageIndex.resolve_closure",
                        format!("包 `{}` 依赖 `{}`，但索引中无此包", id.name, dep.name),
                        format!("安装 `{}`，或移除该依赖", dep.name),
                    )
                })?;
                if !dep.req.matches(&candidate.version) {
                    return Err(dependency_error(
                        "distribution.PackageIndex.resolve_closure",
                        format!(
                            "包 `{}` 依赖 `{}` 要求 ^{}，但索引最新为 {}（不兼容）",
                            id.name, dep.name, dep.req.min, candidate.version
                        ),
                        format!(
                            "安装兼容版本 `{}` {}，或放宽依赖要求",
                            dep.name, dep.req.min
                        ),
                    ));
                }
                queue.push(PackageId::new(dep.name.clone(), candidate.version));
            }
            seen.insert(id);
        }
        let mut out: Vec<PackageId> = seen.into_iter().collect();
        out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| b.version.cmp(&a.version)));
        Ok(out)
    }

    /// 依赖图拓扑序（Kahn，**按名升序**取入度为 0 者 → 序确定）。
    ///
    /// # 错误
    /// 存在循环依赖 → PD3001，错误信息含**具体环路**（可行动，C19）。
    pub fn topological_order(&self) -> Result<Vec<PackageId>, PackageError> {
        // 依赖边：被依赖包 → 依赖方包（被依赖者须先出现）。
        let mut indegree: BTreeMap<PackageId, usize> = BTreeMap::new();
        let mut dependents: BTreeMap<PackageId, BTreeSet<PackageId>> = BTreeMap::new();
        // 第一遍：把全部节点登记入图（**必须先建全节点**，否则查前驱边时会
        // 漏掉尚未登记的 id——按名升序遍历时 a→b 的前驱 b 尚未入图）。
        for id in self.ids() {
            indegree.entry(id).or_insert(0);
        }
        // 第二遍：建边。
        for id in self.ids() {
            let Some(pkg) = self.get(&id) else { continue };
            for dep in &pkg.dependencies {
                let Some(dep_id) = self.resolve_dep_id(dep) else {
                    continue;
                };
                if !indegree.contains_key(&dep_id) {
                    // 依赖不在索引内：不在本图内建边（由 resolve_closure 负责报错）。
                    continue;
                }
                let set = dependents.entry(dep_id).or_default();
                if set.insert(id.clone()) {
                    *indegree.entry(id.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut ready: Vec<PackageId> = indegree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(k, _)| k.clone())
            .collect();
        ready.sort_unstable();
        let mut out: Vec<PackageId> = Vec::with_capacity(indegree.len());
        while let Some(id) = ready.first().cloned() {
            ready.remove(0);
            if let Some(set) = dependents.get(&id) {
                let mut newly: Vec<PackageId> = Vec::new();
                for dep in set {
                    if let Some(d) = indegree.get_mut(dep) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            newly.push(dep.clone());
                        }
                    }
                }
                ready.extend(newly);
                ready.sort_unstable();
            }
            out.push(id);
        }

        if out.len() != indegree.len() {
            // 残余入度 > 0 的包在环上——显式报出环路（不静默截断，C20）。
            let cycle: Vec<String> = indegree
                .iter()
                .filter(|(_, &d)| d > 0)
                .map(|(k, _)| format!("{}@{}", k.name, k.version))
                .collect();
            return Err(dependency_error(
                "distribution.PackageIndex.topological_order",
                format!(
                    "依赖图存在循环依赖，涉及 {} 个包：{}",
                    cycle.len(),
                    cycle.join(" -> ")
                ),
                "打断环路：把环上某个包的依赖改为可选，或移除该依赖边后重新导出",
            ));
        }
        Ok(out)
    }

    /// 把一条依赖声明解析为索引内的确切位置引用（取该名的最新版本）。
    fn resolve_dep_id(&self, dep: &Dependency) -> Option<PackageId> {
        self.latest(&dep.name)
            .map(|p| PackageId::new(p.name.clone(), p.version))
    }

    /// 各包根签名（按索引序）——签名树的全图视图。
    pub fn root_signatures(&self) -> Vec<(PackageId, Hash256)> {
        self.ids()
            .iter()
            .filter_map(|id| self.get(id).map(|p| (id.clone(), p.root_signature())))
            .collect()
    }

    /// 依赖树签名 = `sha256(DEPENDENCY_TAG ++ schema ++ 各被依赖包根签名（名升序) )`。
    ///
    /// 对应 `signature.rs::snapshot_signature` 的「签名链」语义：某包的依赖侧
    /// 签名变化会沿依赖图向上传播，而**与该包无关的包签名逐位不变**。
    pub fn dependency_signature(&self, id: &PackageId) -> Hash256 {
        let mut buf = Vec::new();
        buf.extend_from_slice(DEPENDENCY_TAG);
        buf.extend_from_slice(&PACKAGE_SCHEMA_VERSION.to_be_bytes());
        let Some(pkg) = self.get(id) else {
            return sha256(&buf);
        };
        let mut dep_sigs: Vec<Hash256> = pkg
            .dependencies
            .iter()
            .filter_map(|d| self.resolve_dep_id(d))
            .filter_map(|did| self.get(&did).map(|p| p.root_signature()))
            .collect();
        dep_sigs.sort_unstable();
        buf.extend_from_slice(&(dep_sigs.len() as u32).to_be_bytes());
        for s in &dep_sigs {
            buf.extend_from_slice(s);
        }
        // 依赖树签名与包自身根签名绑定（否则无依赖的包签名全同）。
        buf.extend_from_slice(&pkg.root_signature());
        sha256(&buf)
    }
}

/// 扫描结果：索引 + 逐包问题（不因单个坏包中断整次扫描）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScanReport {
    /// 成功入索引的包。
    pub index: PackageIndex,
    /// 被拒收的包（路径 + 原因）——**逐条上报，绝不静默丢弃**（C20）。
    pub rejected: Vec<(PathBuf, PackageError)>,
}

/// 扫描目录，建立包索引。
///
/// 流程：列目录 → 只取 `*.prismpkg` 普通文件 → **按文件名升序**处理（目录
/// 项迭代序未定义，必须显式排序才确定）→ 逐个 [`Package::from_bytes`]（含哈希
/// 校验）→ 入索引。同名同版本重复 → 记入 `rejected`。
///
/// 单个坏包**不**中断整次扫描（如实收集到 [`ScanReport::rejected`]），调用方
/// 决定是否致命——但坏包绝不会进入索引被误用。
pub fn scan_directory(dir: &Path) -> Result<ScanReport, PackageError> {
    let mut report = ScanReport::default();
    let entries = fs::read_dir(dir).map_err(|e| {
        format_error(
            "distribution.scan_directory",
            format!("读取目录 {:?} 失败：{e}", dir),
            "确认目录存在且可读；用绝对路径重试",
        )
    })?;

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                report.rejected.push((
                    dir.to_path_buf(),
                    format_error(
                        "distribution.scan_directory",
                        format!("目录项读取失败：{e}"),
                        "检查目录权限与文件系统状态后重试",
                    ),
                ));
                continue;
            }
        };
        let path = entry.path();
        // 只取普通文件 + 匹配扩展名。
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some(PACKAGE_EXTENSION) {
            continue;
        }
        files.push(path);
    }
    // 目录迭代序未定义 → 显式按路径字符串排序，保证处理序确定。
    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));

    for path in files {
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                report.rejected.push((
                    path,
                    format_error(
                        "distribution.scan_directory",
                        format!("读取数据包文件失败：{e}"),
                        "检查文件权限；损坏或不可读的文件应删除后重新获取",
                    ),
                ));
                continue;
            }
        };
        match Package::from_bytes(&bytes) {
            Ok(pkg) => {
                if let Err(e) = report.index.insert(IndexEntry {
                    path: path.clone(),
                    package: pkg,
                }) {
                    report.rejected.push((path, e));
                }
            }
            Err(e) => report.rejected.push((path, e)),
        }
    }
    Ok(report)
}

/// 把一个包写入目录（`name@version.prismpkg`）。
pub fn write_package(dir: &Path, pkg: &Package) -> Result<PathBuf, PackageError> {
    fs::create_dir_all(dir).map_err(|e| {
        format_error(
            "distribution.write_package",
            format!("创建目录 {:?} 失败：{e}", dir),
            "确认路径可写；用绝对路径重试",
        )
    })?;
    let file_name = format!("{}@{}.{}", pkg.name, pkg.version, PACKAGE_EXTENSION);
    let path = dir.join(file_name);
    fs::write(&path, pkg.to_bytes()).map_err(|e| {
        format_error(
            "distribution.write_package",
            format!("写入数据包 {:?} 失败：{e}", path),
            "检查目录权限与磁盘空间后重试",
        )
    })?;
    Ok(path)
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 临时目录名（按测试名隔离，避免并行测试互相干扰）。
    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("prism_dist_{tag}"));
        let _ = fs::remove_dir_all(&d);
        d
    }

    /// 构造一个最小可用包。
    fn sample(name: &str, version: SemVer) -> Package {
        Package::new(
            name,
            version,
            "alice",
            "示例数据包",
            vec![],
            vec!["text".to_string()],
            vec![AssetEntry::new("lut", vec![1u8, 2, 3])],
        )
    }

    /// 清理临时目录。
    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    // ------------------------------------------------------------------
    // 关卡 0：数据表不变式
    // ------------------------------------------------------------------

    #[test]
    fn package_magic_is_ten_bytes() {
        // 魔数长度是解析偏移的前提——变了必须同步递增 schema（C13）。
        assert_eq!(PACKAGE_MAGIC.len(), 10);
    }

    // ------------------------------------------------------------------
    // 关卡 1：语义化版本
    // ------------------------------------------------------------------

    #[test]
    fn semver_compares_each_dimension() {
        // major 维度优先。
        assert!(
            SemVer::new(2, 0, 0) > SemVer::new(1, 9, 9),
            "major 优先于 minor/patch"
        );
        // minor 维度次之。
        assert!(
            SemVer::new(1, 2, 0) > SemVer::new(1, 1, 9),
            "minor 优先于 patch"
        );
        // patch 维度最后。
        assert!(
            SemVer::new(1, 1, 2) > SemVer::new(1, 1, 1),
            "patch 末位比较"
        );
        // 相等版本互相等价。
        assert_eq!(SemVer::new(1, 2, 3), SemVer::new(1, 2, 3));
        assert!(SemVer::new(1, 0, 0) <= SemVer::new(1, 0, 0));
        // 默认版本为零。
        assert_eq!(SemVer::default(), SemVer::new(0, 0, 0));
    }

    #[test]
    fn semver_parses_and_rejects_malformed() {
        assert_eq!(
            SemVer::parse("1.2.3").unwrap_or_else(|e| panic!("{e}")),
            SemVer::new(1, 2, 3)
        );
        assert_eq!(
            SemVer::parse("0.0.0").unwrap_or_else(|e| panic!("{e}")),
            SemVer::new(0, 0, 0)
        );
        for bad in [
            "1.2",
            "1.2.3.4",
            "1.2.x",
            "",
            "a.b.c",
            "1.2.-3",
            " 1.2.3",
            "1.2.3-alpha",
        ] {
            let err = SemVer::parse(bad).unwrap_err();
            assert!(
                err.to_string().contains("建议："),
                "拒收须含建议段（C19）：{bad:?}"
            );
        }
    }

    #[test]
    fn version_req_caret_semantics() {
        // ^1.2.3 接受 1.x，不接受 0.x / 2.x。
        let req = VersionReq::at_least(SemVer::new(1, 2, 3));
        assert!(req.matches(&SemVer::new(1, 2, 3)));
        assert!(req.matches(&SemVer::new(1, 9, 0)));
        assert!(!req.matches(&SemVer::new(1, 2, 2)), "低于最低版本不匹配");
        assert!(!req.matches(&SemVer::new(2, 0, 0)), "major 变更不兼容");
        // ^0.3.0 按 minor 锁定。
        let req0 = VersionReq::at_least(SemVer::new(0, 3, 0));
        assert!(req0.matches(&SemVer::new(0, 3, 7)));
        assert!(!req0.matches(&SemVer::new(0, 4, 0)), "^0.3.0 不接受 0.4.0");
    }

    // ------------------------------------------------------------------
    // 关卡 2：清单序列化往返
    // ------------------------------------------------------------------

    #[test]
    fn manifest_roundtrip_is_byte_exact() {
        let pkg = Package::new(
            "luts.warm",
            SemVer::new(1, 2, 3),
            "carol",
            "暖色 LUT 包",
            vec![Dependency::new("base.core", SemVer::new(0, 1, 0))],
            vec!["color".to_string(), "lut".to_string()],
            vec![
                AssetEntry::new("neutral.cube", vec![9u8; 16]),
                AssetEntry::new("warm.cube", vec![7u8; 8]),
            ],
        );
        let bytes = pkg.to_bytes();
        let back = Package::from_bytes(&bytes).unwrap_or_else(|e| panic!("往返失败：{e}"));
        assert_eq!(back, pkg, "反序列化必须与原清单全等");
        assert_eq!(back.to_bytes(), bytes, "再序列化必须逐字节一致");
        assert_eq!(back.content_hash, pkg.content_hash, "内容哈希往返一致");
    }

    #[test]
    fn manifest_serialization_is_insertion_order_independent() {
        // 依赖/能力/资产都是语义集合：装配序不得影响字节。
        let a = Package::new(
            "p",
            SemVer::new(1, 0, 0),
            "a",
            "",
            vec![
                Dependency::new("z", SemVer::new(1, 0, 0)),
                Dependency::new("y", SemVer::new(2, 0, 0)),
            ],
            vec!["b".to_string(), "a".to_string(), "c".to_string()],
            vec![
                AssetEntry::new("s2", vec![2u8]),
                AssetEntry::new("s1", vec![1u8]),
            ],
        );
        let b = Package::new(
            "p",
            SemVer::new(1, 0, 0),
            "a",
            "",
            vec![
                Dependency::new("y", SemVer::new(2, 0, 0)),
                Dependency::new("z", SemVer::new(1, 0, 0)),
            ],
            vec!["c".to_string(), "b".to_string(), "a".to_string()],
            vec![
                AssetEntry::new("s1", vec![1u8]),
                AssetEntry::new("s2", vec![2u8]),
            ],
        );
        assert_eq!(
            a.to_bytes(),
            b.to_bytes(),
            "集合装配序不得影响 canonical 字节"
        );
        assert_eq!(a.content_hash, b.content_hash, "内容哈希亦与装配序无关");
    }

    #[test]
    fn manifest_is_byte_level_deterministic() {
        let mk = || {
            Package::new(
                "det",
                SemVer::new(2, 1, 0),
                "dave",
                "确定性",
                vec![Dependency::new("core", SemVer::new(1, 0, 0))],
                vec!["x".to_string(), "y".to_string()],
                vec![
                    AssetEntry::new("a", vec![1u8, 2]),
                    AssetEntry::new("b", vec![3u8]),
                ],
            )
        };
        assert_eq!(mk().to_bytes(), mk().to_bytes(), "同输入两遍必须逐字节一致");
    }

    #[test]
    fn content_hash_detects_any_field_tampering() {
        let pkg = sample("t", SemVer::new(1, 0, 0));
        let mut bytes = pkg.to_bytes();
        // 定位作者字段 "alice" 并改其首字节——保持结构合法（长度前缀不变），
        // 故只能靠内容哈希检出（这正是哈希的职责：结构合法但内容被改）。
        let pos = bytes
            .windows(5)
            .position(|w| w == b"alice")
            .unwrap_or_else(|| panic!("未找到作者字段"));
        if let Some(slot) = bytes.get_mut(pos) {
            *slot = b'A';
        }
        let err = Package::from_bytes(&bytes).unwrap_err();
        assert_eq!(err.code, "PD2001", "结构合法但内容被改 → 必须靠哈希检出");
        assert!(err.problem.contains("t"), "报错应指出包名：{}", err.problem);
        assert!(err.to_string().contains("建议："), "必含建议段（C19）");
    }

    #[test]
    fn content_hash_detects_asset_data_tampering() {
        let pkg = Package::new(
            "asset-tamper",
            SemVer::new(1, 0, 0),
            "eve",
            "",
            vec![],
            vec![],
            vec![AssetEntry::new("blob", vec![0u8; 4])],
        );
        let mut bytes = pkg.to_bytes();
        // 资产数据（4 个零字节）紧邻尾部 32 字节哈希之前，故偏移 = len - 32 - 4。
        // 精确定位而非搜索零区——搜索会误命中长度前缀里的零字节。
        let target = bytes.len().saturating_sub(32 + 4);
        assert_eq!(
            bytes.get(target..target + 4),
            Some(&[0u8; 4][..]),
            "定位到的应是资产数据区"
        );
        if let Some(slot) = bytes.get_mut(target) {
            *slot = 0xFF;
        }
        let err = Package::from_bytes(&bytes).unwrap_err();
        assert_eq!(err.code, "PD2001", "资产数据改动必须被哈希检出");
    }

    #[test]
    fn truncated_and_oversized_payloads_are_rejected_without_panic() {
        let pkg = sample("trunc", SemVer::new(1, 0, 0));
        let bytes = pkg.to_bytes();
        // 各种截断点：绝不 panic（宪章 C18），一律结构化错误。
        for cut in [
            0usize,
            1,
            5,
            9,
            11,
            20,
            33,
            47,
            bytes.len() / 2,
            bytes.len() - 1,
        ] {
            let err = Package::from_bytes(&bytes[..cut.min(bytes.len())]).unwrap_err();
            assert_eq!(err.code, "PD1001", "截断到 {cut} 字节应报格式错误");
        }
        // 尾部多余字节 → 拒收（防拼接歧义）。
        let mut extra = bytes.clone();
        extra.push(0);
        let err = Package::from_bytes(&extra).unwrap_err();
        assert!(
            err.problem.contains("多余"),
            "多余尾字节须被检出：{}",
            err.problem
        );
    }

    #[test]
    fn wrong_magic_and_schema_are_rejected() {
        let pkg = sample("m", SemVer::new(1, 0, 0));
        let mut bad_magic = pkg.to_bytes();
        if let Some(s) = bad_magic.get_mut(0) {
            *s = b'X';
        }
        let err = Package::from_bytes(&bad_magic).unwrap_err();
        assert_eq!(err.code, "PD1001");
        assert!(
            err.problem.contains("魔数"),
            "须指出魔数不匹配：{}",
            err.problem
        );

        // schema 版本不符 → 拒收（C13：旧包只失效重建，永不误复用）。
        let mut bad_schema = pkg.to_bytes();
        let off = PACKAGE_MAGIC.len();
        if let Some(s) = bad_schema.get_mut(off) {
            *s = 9;
        }
        let err = Package::from_bytes(&bad_schema).unwrap_err();
        assert!(
            err.problem.contains("schema"),
            "须指出 schema 不符：{}",
            err.problem
        );
    }

    // ------------------------------------------------------------------
    // 关卡 3：签名（完整性 + 作者绑定）
    // ------------------------------------------------------------------

    #[test]
    fn signature_binds_content_and_signer() {
        let pkg = sample("sig", SemVer::new(1, 0, 0));
        let sig = pkg.verify("alice");
        assert!(
            sig.verify_against(&pkg.content_hash, "alice"),
            "同内容同签名者应验签通过"
        );
        // 换签名者 → 验签失败（作者身份绑定生效）。
        assert!(
            !sig.verify_against(&pkg.content_hash, "mallory"),
            "他人标识不得验签通过"
        );
        // 换内容 → 验签失败（完整性绑定生效）。
        let other = sample("sig", SemVer::new(1, 0, 1));
        assert!(
            !sig.verify_against(&other.content_hash, "alice"),
            "内容变更必须验签失败"
        );
    }

    #[test]
    fn signature_is_deterministic_and_domain_separated() {
        let h = [7u8; 32];
        assert_eq!(
            PackageSignature::compute(&h, "alice"),
            PackageSignature::compute(&h, "alice"),
            "同输入同签名"
        );
        // 域分隔：签名值不等于裸内容哈希（防跨域混用）。
        let sig = PackageSignature::compute(&h, "alice");
        assert_ne!(sig.value, h, "签名必须带域分隔，不得等于裸哈希");
        // 签名者摘要就是 sha256(标识)（确定性派生）。
        assert_eq!(
            sig.signer.to_vec(),
            sha256(b"alice").to_vec(),
            "签名者摘要应等于 sha256(标识)"
        );
        assert_ne!(sig.signer, sig.value, "签名者摘要与签名值必须是不同域");
    }

    // ------------------------------------------------------------------
    // 关卡 4：签名树（增量性）
    // ------------------------------------------------------------------

    #[test]
    fn root_signature_depends_on_every_asset() {
        let base = Package::new(
            "tree",
            SemVer::new(1, 0, 0),
            "a",
            "",
            vec![],
            vec![],
            vec![
                AssetEntry::new("x", vec![1u8]),
                AssetEntry::new("y", vec![2u8]),
            ],
        );
        // 改一个资产内容 → 根签名必变。
        let mut changed = base.clone();
        changed.assets[0].data = vec![9u8];
        changed.content_hash = sha256(&changed.canonical_bytes());
        assert_ne!(
            base.root_signature(),
            changed.root_signature(),
            "资产改动必须改变根签名"
        );

        // 改资产**名**（内容不变）→ 根签名也必变（名进 canonical）。
        let mut renamed = base.clone();
        renamed.assets[0].name = "z".to_string();
        renamed.content_hash = sha256(&renamed.canonical_bytes());
        assert_ne!(
            base.root_signature(),
            renamed.root_signature(),
            "资产改名必须改变根签名"
        );

        // 完全相同 → 签名逐位一致。
        let same = base.clone();
        assert_eq!(base.root_signature(), same.root_signature());
    }

    #[test]
    fn entry_signature_ignores_sibling_order() {
        // 条目签名按名排序拼接 → 资产装配序不影响根签名。
        let a = Package::new(
            "ord",
            SemVer::new(1, 0, 0),
            "a",
            "",
            vec![],
            vec![],
            vec![
                AssetEntry::new("b", vec![2u8]),
                AssetEntry::new("a", vec![1u8]),
            ],
        );
        let b = Package::new(
            "ord",
            SemVer::new(1, 0, 0),
            "a",
            "",
            vec![],
            vec![],
            vec![
                AssetEntry::new("a", vec![1u8]),
                AssetEntry::new("b", vec![2u8]),
            ],
        );
        assert_eq!(
            a.root_signature(),
            b.root_signature(),
            "资产序不影响根签名（已规范化）"
        );
    }

    // ------------------------------------------------------------------
    // 关卡 5：注册表索引
    // ------------------------------------------------------------------

    /// 构造含三包（两版本同名）的索引。
    fn demo_index() -> PackageIndex {
        let mut idx = PackageIndex::new();
        for (n, v) in [
            ("b.pkg", SemVer::new(1, 0, 0)),
            ("a.pkg", SemVer::new(2, 1, 0)),
        ] {
            idx.insert(IndexEntry {
                path: PathBuf::from(n),
                package: sample(n, v),
            })
            .unwrap_or_else(|e| panic!("{e}"));
        }
        idx.insert(IndexEntry {
            path: PathBuf::from("a.pkg"),
            package: sample("a.pkg", SemVer::new(1, 5, 0)),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        idx
    }

    #[test]
    fn index_order_is_name_ascending_version_descending() {
        let idx = demo_index();
        let ids = idx.ids();
        assert_eq!(
            ids,
            vec![
                PackageId::new("a.pkg", SemVer::new(2, 1, 0)),
                PackageId::new("a.pkg", SemVer::new(1, 5, 0)),
                PackageId::new("b.pkg", SemVer::new(1, 0, 0)),
            ],
            "索引序必须是 (名升序, 版本降序)"
        );
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.distinct_names(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn duplicate_name_and_version_is_rejected() {
        let mut idx = PackageIndex::new();
        idx.insert(IndexEntry {
            path: PathBuf::from("a"),
            package: sample("dup", SemVer::new(1, 0, 0)),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let err = idx
            .insert(IndexEntry {
                path: PathBuf::from("b"),
                package: sample("dup", SemVer::new(1, 0, 0)),
            })
            .unwrap_err();
        assert_eq!(err.code, "PD1001", "同名同版本须拒收");
        assert!(err.to_string().contains("建议："), "必含建议段（C19）");
        assert_eq!(idx.len(), 1, "拒收后索引保持干净（不静默覆盖，C20）");
    }

    #[test]
    fn query_by_name_version_and_capability() {
        let idx = demo_index();
        // 按名 → 版本降序。
        let versions: Vec<SemVer> = idx.versions_of("a.pkg").iter().map(|p| p.version).collect();
        assert_eq!(versions, vec![SemVer::new(2, 1, 0), SemVer::new(1, 5, 0)]);
        // 取最新。
        assert_eq!(
            idx.latest("a.pkg").map(|p| p.version),
            Some(SemVer::new(2, 1, 0))
        );
        // 精确取。
        let id = PackageId::new("a.pkg", SemVer::new(1, 5, 0));
        assert!(idx.get(&id).is_some(), "精确版本应可取");
        // 缺失返回 None（不 panic、不编造）。
        assert!(idx
            .get(&PackageId::new("a.pkg", SemVer::new(9, 9, 9)))
            .is_none());
        assert!(idx.latest("nope").is_none());
        // 按能力查询 → 结果按索引序。
        let found = idx.find_by_capability("text");
        assert_eq!(found.len(), 3, "三包都声明 text 能力");
        assert_eq!(
            found[0],
            PackageId::new("a.pkg", SemVer::new(2, 1, 0)),
            "能力查询亦按索引序"
        );
        // 未知能力 → 空（不近似匹配）。
        assert!(
            idx.find_by_capability("video").is_empty(),
            "未知能力不得近似匹配"
        );
        assert_eq!(idx.capabilities(), vec!["text"]);
    }

    // ------------------------------------------------------------------
    // 关卡 6：目录扫描
    // ------------------------------------------------------------------

    #[test]
    fn scan_directory_is_deterministic_and_order_independent() {
        let dir = tmp_dir("scan");
        // 故意按「逆索引序」写盘：b 先写、a 后写。
        write_package(&dir, &sample("b.pkg", SemVer::new(1, 0, 0)))
            .unwrap_or_else(|e| panic!("{e}"));
        write_package(&dir, &sample("a.pkg", SemVer::new(2, 0, 0)))
            .unwrap_or_else(|e| panic!("{e}"));
        // 放一个非数据包文件，应被扩展名过滤忽略。
        fs::write(dir.join("readme.txt"), b"not a package").unwrap_or_else(|e| panic!("{e}"));

        let r1 = scan_directory(&dir).unwrap_or_else(|e| panic!("{e}"));
        let r2 = scan_directory(&dir).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r1.index.ids(), r2.index.ids(), "多次扫描索引必须一致");
        assert_eq!(
            r1.index.ids(),
            vec![
                PackageId::new("a.pkg", SemVer::new(2, 0, 0)),
                PackageId::new("b.pkg", SemVer::new(1, 0, 0))
            ],
            "扫描索引序与写盘序无关"
        );
        assert!(
            r1.rejected.is_empty(),
            "非数据包文件应被过滤而非拒收：{:?}",
            r1.rejected
        );
        cleanup(&dir);
    }

    #[test]
    fn scan_reports_bad_packages_without_aborting_or_ingesting_them() {
        let dir = tmp_dir("bad");
        write_package(&dir, &sample("good", SemVer::new(1, 0, 0)))
            .unwrap_or_else(|e| panic!("{e}"));
        // 坏包：哈希被篡改。
        let mut pkg = sample("bad", SemVer::new(1, 0, 0));
        pkg.content_hash = [0xAB; 32];
        fs::write(
            dir.join(format!("bad@1.0.0.{}", PACKAGE_EXTENSION)),
            pkg.to_bytes(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        // 坏包：非本格式。
        fs::write(
            dir.join(format!("junk@1.0.0.{}", PACKAGE_EXTENSION)),
            b"garbage!!",
        )
        .unwrap_or_else(|e| panic!("{e}"));

        let report = scan_directory(&dir).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(report.index.len(), 1, "只应索引一个好包");
        assert_eq!(
            report.index.latest("good").map(|p| p.name.as_str()),
            Some("good")
        );
        assert_eq!(report.rejected.len(), 2, "两个坏包都应被如实上报");
        // 拒收原因有序确定（按文件名排序）：bad 先于 junk。
        let names: Vec<String> = report
            .rejected
            .iter()
            .map(|(p, _)| p.to_string_lossy().into_owned())
            .collect();
        assert!(
            names[0].contains("bad@"),
            "拒收序应确定，首项为 bad：{names:?}"
        );
        // 坏包绝不入索引（不静默接受，C20）。
        assert!(report.index.latest("bad").is_none());
        assert!(report.index.latest("junk").is_none());
        cleanup(&dir);
    }

    #[test]
    fn write_then_scan_roundtrip_preserves_content_hash() {
        let dir = tmp_dir("roundtrip");
        let pkg = Package::new(
            "fonts.prism",
            SemVer::new(3, 2, 1),
            "frank",
            "字体包",
            vec![],
            vec!["font".to_string()],
            vec![AssetEntry::new("regular.ttf", vec![0xAA; 32])],
        );
        let original_hash = pkg.content_hash;
        let path = write_package(&dir, &pkg).unwrap_or_else(|e| panic!("{e}"));
        assert!(path.exists());
        let report = scan_directory(&dir).unwrap_or_else(|e| panic!("{e}"));
        let loaded = report
            .index
            .latest("fonts.prism")
            .unwrap_or_else(|| panic!("未索引到"));
        assert_eq!(loaded.content_hash, original_hash, "落盘往返内容哈希不变");
        assert_eq!(loaded, &pkg, "落盘往返清单全等");
        cleanup(&dir);
    }

    #[test]
    fn scan_missing_directory_returns_structured_error() {
        let err = scan_directory(Path::new("/nonexistent/prism/dist/dir")).unwrap_err();
        assert_eq!(err.code, "PD1001");
        let text = err.to_string();
        assert!(text.starts_with("error[PD1001] "), "三段式首段：{text}");
        assert!(text.contains("建议："), "必含建议段（C19）：{text}");
    }

    // ------------------------------------------------------------------
    // 关卡 7：依赖图（循环检测 + 拓扑序 + 闭包）
    // ------------------------------------------------------------------

    /// 构造线性依赖链 a → b → c 的索引（`x` 依赖 `deps` 列出的包）。
    fn chain_index_with(c_asset: u8) -> PackageIndex {
        let mk = |name: &str, deps: Vec<Dependency>| IndexEntry {
            path: PathBuf::from(name),
            package: Package::new(
                name,
                SemVer::new(1, 0, 0),
                "gina",
                "",
                deps,
                vec![],
                vec![AssetEntry::new("f", vec![c_asset])],
            ),
        };
        let mut idx = PackageIndex::new();
        // 声明方向：a 依赖 b，b 依赖 c。
        idx.insert(mk("a", vec![Dependency::new("b", SemVer::new(1, 0, 0))]))
            .unwrap_or_else(|e| panic!("{e}"));
        idx.insert(mk("b", vec![Dependency::new("c", SemVer::new(1, 0, 0))]))
            .unwrap_or_else(|e| panic!("{e}"));
        idx.insert(mk("c", vec![]))
            .unwrap_or_else(|e| panic!("{e}"));
        idx
    }

    /// 构造线性依赖链 a → b → c 的索引。
    fn chain_index() -> PackageIndex {
        chain_index_with(1)
    }

    #[test]
    fn topological_order_puts_dependencies_first() {
        let idx = chain_index();
        let order = idx.topological_order().unwrap_or_else(|e| panic!("{e}"));
        let pos = |n: &str| order.iter().position(|x| x.name == n).unwrap_or(usize::MAX);
        assert!(pos("c") < pos("b"), "c 必须先于 b");
        assert!(pos("b") < pos("a"), "b 必须先于 a");
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn cyclic_dependency_is_detected_with_the_cycle_reported() {
        let mk = |name: &str, dep: &str| IndexEntry {
            path: PathBuf::from(name),
            package: Package::new(
                name,
                SemVer::new(1, 0, 0),
                "hank",
                "",
                vec![Dependency::new(dep, SemVer::new(1, 0, 0))],
                vec![],
                vec![],
            ),
        };
        let mut idx = PackageIndex::new();
        idx.insert(mk("p", "q")).unwrap_or_else(|e| panic!("{e}"));
        idx.insert(mk("q", "r")).unwrap_or_else(|e| panic!("{e}"));
        idx.insert(mk("r", "p")).unwrap_or_else(|e| panic!("{e}")); // 成环

        let err = idx.topological_order().unwrap_err();
        assert_eq!(err.code, "PD3001", "循环依赖须报 PD3001");
        let text = err.to_string();
        assert!(text.contains("循环依赖"), "须指出循环依赖：{text}");
        // 环路须点名具体包（可行动，C19）。
        for n in ["p", "q", "r"] {
            assert!(text.contains(n), "环路信息应含包名 {n}：{text}");
        }
        assert!(text.contains("建议："), "必含建议段（C19）：{text}");
    }

    #[test]
    fn self_dependency_is_detected() {
        let mut idx = PackageIndex::new();
        idx.insert(IndexEntry {
            path: PathBuf::from("self"),
            package: Package::new(
                "self",
                SemVer::new(1, 0, 0),
                "i",
                "",
                vec![Dependency::new("self", SemVer::new(1, 0, 0))],
                vec![],
                vec![],
            ),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let err = idx.topological_order().unwrap_err();
        assert_eq!(err.code, "PD3001", "自依赖须被检出为环");
    }

    #[test]
    fn resolve_closure_expands_transitively() {
        let idx = chain_index();
        let closure = idx
            .resolve_closure(&[PackageId::new("a", SemVer::new(1, 0, 0))])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(closure.len(), 3, "a 的闭包应含 a/b/c");
        // 闭包结果按索引序（名升序）。
        assert_eq!(closure[0].name, "a");
        assert_eq!(closure[1].name, "b");
        assert_eq!(closure[2].name, "c");
    }

    #[test]
    fn resolve_closure_reports_missing_and_incompatible_deps() {
        let mut idx = PackageIndex::new();
        // 依赖一个不存在的包。
        idx.insert(IndexEntry {
            path: PathBuf::from("needy"),
            package: Package::new(
                "needy",
                SemVer::new(1, 0, 0),
                "j",
                "",
                vec![Dependency::new("absent", SemVer::new(1, 0, 0))],
                vec![],
                vec![],
            ),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let err = idx
            .resolve_closure(&[PackageId::new("needy", SemVer::new(1, 0, 0))])
            .unwrap_err();
        assert_eq!(err.code, "PD3001");
        assert!(
            err.problem.contains("absent"),
            "须指出缺失依赖名：{}",
            err.problem
        );

        // 依赖存在但版本不兼容。
        let mut idx2 = PackageIndex::new();
        idx2.insert(IndexEntry {
            path: PathBuf::from("picky"),
            package: Package::new(
                "picky",
                SemVer::new(1, 0, 0),
                "k",
                "",
                vec![Dependency::new("lib", SemVer::new(2, 0, 0))],
                vec![],
                vec![],
            ),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        idx2.insert(IndexEntry {
            path: PathBuf::from("lib"),
            package: sample("lib", SemVer::new(1, 0, 0)),
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let err = idx2
            .resolve_closure(&[PackageId::new("picky", SemVer::new(1, 0, 0))])
            .unwrap_err();
        assert_eq!(err.code, "PD3001");
        assert!(
            err.problem.contains("不兼容"),
            "须指出版本不兼容：{}",
            err.problem
        );
    }

    // ------------------------------------------------------------------
    // 关卡 8：签名树的图级增量性（对应 signature.rs 的核心性质）
    // ------------------------------------------------------------------

    #[test]
    fn dependency_signature_propagates_only_along_the_chain() {
        // 链：c ← b ← a（b 依赖 c，a 依赖 b）。
        let base = chain_index();
        let a = PackageId::new("a", SemVer::new(1, 0, 0));
        let b = PackageId::new("b", SemVer::new(1, 0, 0));
        let c = PackageId::new("c", SemVer::new(1, 0, 0));
        let base_a = base.dependency_signature(&a);
        let base_b = base.dependency_signature(&b);
        let base_c = base.dependency_signature(&c);

        // 改 c 的资产（重建索引以避免同名同版本重复登记）→ c/b/a 签名全变。
        let changed = chain_index_with(99);
        assert_ne!(
            changed.dependency_signature(&c),
            base_c,
            "叶子自身签名必须变"
        );
        assert_ne!(changed.dependency_signature(&b), base_b, "父节点签名必须变");
        assert_ne!(
            changed.dependency_signature(&a),
            base_a,
            "祖父节点签名必须变"
        );
        // 确定性：同一份输入两次求值逐位一致（传播本身无隐藏状态）。
        assert_eq!(
            changed.dependency_signature(&a),
            changed.dependency_signature(&a),
            "依赖树签名必须是纯函数"
        );
    }

    #[test]
    fn unrelated_package_signatures_are_bit_identical_after_a_change() {
        // 兄弟子树逐位不变（signature.rs 增量渲染根基的镜像性质）。
        let mut idx = PackageIndex::new();
        for n in ["base", "left", "right"] {
            idx.insert(IndexEntry {
                path: PathBuf::from(n),
                package: Package::new(
                    n,
                    SemVer::new(1, 0, 0),
                    "m",
                    "",
                    vec![],
                    vec![],
                    vec![AssetEntry::new("f", vec![1u8])],
                ),
            })
            .unwrap_or_else(|e| panic!("{e}"));
        }
        let right_before = idx.dependency_signature(&PackageId::new("right", SemVer::new(1, 0, 0)));
        // 改 left 的资产。
        let new_left = Package::new(
            "left",
            SemVer::new(1, 0, 0),
            "m",
            "",
            vec![],
            vec![],
            vec![AssetEntry::new("f", vec![42u8])],
        );
        // 同名同版本不能重复插入，故重建索引。
        let mut idx2 = PackageIndex::new();
        for (n, pkg) in [
            (
                "base",
                idx.latest("base")
                    .unwrap_or_else(|| panic!("无 base"))
                    .clone(),
            ),
            ("left", new_left),
            (
                "right",
                idx.latest("right")
                    .unwrap_or_else(|| panic!("无 right"))
                    .clone(),
            ),
        ] {
            idx2.insert(IndexEntry {
                path: PathBuf::from(n),
                package: pkg,
            })
            .unwrap_or_else(|e| panic!("{e}"));
        }
        let right_after = idx2.dependency_signature(&PackageId::new("right", SemVer::new(1, 0, 0)));
        assert_eq!(right_before, right_after, "与改动无关的包签名必须逐位不变");
    }

    #[test]
    fn root_signatures_follow_index_order() {
        let idx = demo_index();
        let sigs = idx.root_signatures();
        assert_eq!(sigs.len(), 3);
        assert_eq!(sigs[0].0, PackageId::new("a.pkg", SemVer::new(2, 1, 0)));
        assert_eq!(sigs[2].0, PackageId::new("b.pkg", SemVer::new(1, 0, 0)));
    }

    #[test]
    fn empty_registry_operations_are_safe() {
        let idx = PackageIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.ids().is_empty());
        assert!(idx.latest("any").is_none());
        assert!(idx.find_by_capability("any").is_empty());
        assert!(idx.capabilities().is_empty());
        // 空图拓扑序为空 Ok（不是错误）。
        assert_eq!(
            idx.topological_order()
                .unwrap_or_else(|e| panic!("{e}"))
                .len(),
            0
        );
    }
}
