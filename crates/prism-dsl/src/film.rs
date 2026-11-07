//! V3.0 W4：单文件形态（A5 / D-27 / B3）+ 统一命名时序通道（A7）+ 四类错误域注入面（A6）。
//!
//! # 它是什么
//!
//! 一部片子 = 一个 `.prism` 文件。本模块给 `// prism-dsl 2` 文档新增两类一等声明：
//!
//! ```text
//! asset "lut"  kind exr   hash "<16hex>" external "assets/lut.exr"
//! asset "font" kind font  hash "<16hex>" inline   "0a1b2c..."
//! channel "w3.motion.strength" expr "0.3 + 0.2 * t"
//! channel "R-2.roughness"      key 0s 0.2 linear key 1s 0.8 linear
//! asset_ref "lut"       // 使用点：引用已声明的资产（未声明即结构化报错）
//! ```
//!
//! # 红线合规（逐条）
//!
//! - **A5 / D-27 内容哈希锁定**：资产以**内容哈希**（`hash`，u64 的 16 位 hex）为唯一身份，
//!   `hash_mix_bytes` 派生；产物哈希只混入 `(kind, hash, inline 字节)`，**绝不混入路径**。
//!   语言层**零文件 IO**（§1.4 受限子集白名单）——外部引用路径只是定位提示，不读文件系统，
//!   因此"当前工作目录下有没有这个文件"不可能影响产物字节 ⇒ 资产路径不是隐式环境状态。
//! - **不静默降级、不取默认（红线 3）**：漏 `kind` / 漏 `hash` / 非法 hash / 未知 kind /
//!   inline 内容与声明 hash 不符 / 外部路径绝对或含 `..` / 重复声明 / 引用未声明资产，
//!   一律 `P95xx` 三段式报错（在哪 / 什么事 / 怎么改）并带声明行号，绝不补缺省。
//! - **红线 1 语言层不是第二事实源**：命名通道的求值全走既有
//!   `prism_core::PropertyChannel::sample_ctx`（与 W2 `resolve_frame` 同一求值入口），
//!   表达式用 `prism_anim::Expr::parse`、关键帧用 `prism_anim::Track`，本模块不重实现数学。
//! - **红线 4 不贴皮 / 不偷加噪 / 不替用户选档**：时序覆盖发生在
//!   `movie.rs::wiring_frame_values`（渲染前、组装 Scene 之前），不是帧后效果节点；
//!   `analytic` 档不产生任何 `w3.*` 键 ⇒ 覆盖层对 analytic 恒等、零意外噪点；
//!   不给 `finish` 设默认值（档位仍由 W3 `scan_finish` 强制）。
//! - **零损失（D-24）**：`// prism-dsl 1` / 无头旧文件走 `compile_v2`，根本不经过本模块；
//!   不声明 `asset` / `channel` 的 v3 文件 ⇒ `FilmPlan` 为空 ⇒ 产物哈希混入恒等、
//!   `base.channels` 零追加、遮罩后源码与 W3 收口逐字节相同。
//!
//! # 确定性
//!
//! 全整数 / f32 位型存储；`BTreeMap` 固定序；资产内容哈希用 `hash_mix_bytes`（零 libm、
//! 零线程、零系统时间、零真随机）；通道采样用 `PropertyChannel::sample_ctx`。同文件同种
//! 子两跑逐字节一致。

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use prism_anim::{EasingKind, Expr, Keyframe, Track};
use prism_core::eval::{hash_mix, hash_mix_bytes};
use prism_core::{EvalCtx, PropertyChannel};

use crate::codegen::NamedChannel;
use crate::{err_at, DslError};

// ---------------------------------------------------------------------------
// 常量与命名空间
// ---------------------------------------------------------------------------

/// W4 film 命名空间（进产物哈希、进求值轨迹，与 W1 `SCHEMA_TAG` / W3 `W3_TAG` 不冲突）。
pub const FILM_TAG: u64 = 0x7733_5eed_0000_0004;

/// 通道降级进 `CompileV2Output.channels` 时的名字前缀；`movie.rs::wiring_frame_values`
/// 只认这个前缀，strip 后得到被驱动的目标参数路径（如 `w3.motion.strength`）。
/// 前缀隔离保证既有 v2 `anim`/`expr` 通道与 W2 `cap` 通道零误触。
pub const DRIVE_PREFIX: &str = "w4.drive.";

/// hash 必须是 16 位小写 hex（u64，与 `hash_mix_bytes` 输出位宽一致）。
const HASH_HEX_LEN: usize = 16;

/// 三类可锁定资产（D-27 / B3 原文：EXR / 字体 / 音频进产物哈希）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    Exr,
    Font,
    Audio,
}

impl AssetKind {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Exr => "exr",
            Self::Font => "font",
            Self::Audio => "audio",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "exr" => Some(Self::Exr),
            "font" => Some(Self::Font),
            "audio" => Some(Self::Audio),
            _ => None,
        }
    }

    pub const ALL: [AssetKind; 3] = [Self::Exr, Self::Font, Self::Audio];
}

/// 资产来源：外部引用为主、内嵌可选。
///
/// **两者都只把内容哈希带进产物**：`External` 仅登记一行「去哪找」的提示文本（路径本身
/// 不进规范字节、不读文件）；`Inline` 把资产字节直接写进单文件（自包含、空环境可出片），
/// 编译期校验其 `hash_mix_bytes` 与声明 hash 一致（内容哈希锁定的自洽证明）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetSource {
    /// 外部引用：`external "<相对路径>"`。语言层不读文件。
    External(String),
    /// 内嵌：`inline "<hex bytes>"`。字节进产物哈希。
    Inline(Vec<u8>),
    /// 仅声明内容哈希（无来源提示）——合法的「外部引用为主」最简形态。
    HashOnly,
}

/// 一条资产声明。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetDecl {
    pub name: String,
    pub kind: AssetKind,
    /// 内容哈希（u64，声明为 16 位 hex）。
    pub content_hash: u64,
    pub source: AssetSource,
    /// 声明行号（三段式报错定位）。
    pub line: usize,
}

/// 资产计划（一部片子的全部资产声明，按声明序）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssetPlan {
    pub assets: Vec<AssetDecl>,
}

impl AssetPlan {
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// 规范字节：**不含任何路径字符串**（D-27「路径不成为隐式环境状态」的可测执行体）。
    /// 换 `external` 路径、同 kind/hash/inline ⇒ `canonical_bytes()` 逐字节相同。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(self.assets.len() as u64).to_le_bytes());
        for a in &self.assets {
            out.extend_from_slice(&(a.name.len() as u64).to_le_bytes());
            out.extend_from_slice(a.name.as_bytes());
            out.push(match a.kind {
                AssetKind::Exr => 1,
                AssetKind::Font => 2,
                AssetKind::Audio => 3,
            });
            out.extend_from_slice(&a.content_hash.to_le_bytes());
            // 只有内嵌字节进产物（外部提示路径不进）。
            match &a.source {
                AssetSource::Inline(bytes) => {
                    out.push(1);
                    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                    out.extend_from_slice(bytes);
                }
                AssetSource::External(_) => out.push(2),
                AssetSource::HashOnly => out.push(0),
            }
        }
        out
    }

    pub fn has(&self, name: &str) -> bool {
        self.assets.iter().any(|a| a.name == name)
    }

    /// 内容哈希锁定：所有资产的内容哈希按声明序混入产物哈希链（F-8 扩链）。
    pub fn mix_artifact_hash(&self, h: u64) -> u64 {
        hash_mix_bytes(hash_mix(h, FILM_TAG), &self.canonical_bytes())
    }
}

/// 一条命名时序通道声明：`channel "<target>" { expr ... | key ... }`。
///
/// `target` 是**任意参数路径**（W2 的 L3 核心项如 `R-2.roughness`、`E-4.frame`，
/// 观感六族痕迹强度如 `w3.motion.strength`，或任意自定义参数名）。求值全交给
/// 既有 `PropertyChannel`（红线 1：语言层不重实现数学）。
#[derive(Debug, Clone)]
pub struct ChannelDecl {
    pub target: String,
    pub channel: PropertyChannel,
    pub line: usize,
}

/// 命名通道计划（一部片子的全部通道声明，按声明序）。
#[derive(Debug, Clone, Default)]
pub struct ChannelPlan {
    pub channels: Vec<ChannelDecl>,
}

impl ChannelPlan {
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// 规范字节（通道定义用 `Debug` 位型化，与 `channel_defs_of` 同口径；诊断行号不进产物）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(self.channels.len() as u64).to_le_bytes());
        for c in &self.channels {
            out.extend_from_slice(&(c.target.len() as u64).to_le_bytes());
            out.extend_from_slice(c.target.as_bytes());
            let def: String = format!("{:?}", c.channel);
            out.extend_from_slice(&(def.len() as u64).to_le_bytes());
            out.extend_from_slice(def.as_bytes());
        }
        out
    }

    pub fn mix_artifact_hash(&self, h: u64) -> u64 {
        hash_mix_bytes(h, &self.canonical_bytes())
    }

    /// 逐帧解算：**唯一求值入口** = 既有 `PropertyChannel::sample_ctx`（W2 `resolve_frame`
    /// 同源）。返回 `target → f32 位型`，`BTreeMap` 固定序，逐帧值可断言。
    pub fn frame_values(&self, t: f32, frame: u64, global_seed: u64, node_id: u64) -> BTreeMap<String, u32> {
        let ctx = EvalCtx::new(global_seed, node_id, frame);
        let mut out: BTreeMap<String, u32> = BTreeMap::new();
        for c in &self.channels {
            let v: f32 = c.channel.sample_ctx(t, &ctx);
            out.insert(c.target.clone(), v.to_bits());
        }
        out
    }

    /// 降级为既有 `NamedChannel`（名字带 `w4.drive.` 前缀），追加进
    /// `CompileV2Output.channels` —— 不改任何结构字段，复用 W2 求值链到渲染层。
    pub fn into_named_channels(&self) -> Vec<NamedChannel> {
        self.channels
            .iter()
            .map(|c| NamedChannel {
                name: format!("{}{}", DRIVE_PREFIX, c.target),
                channel: c.channel.clone(),
            })
            .collect()
    }
}

/// 一次完整扫描的产物。
#[derive(Debug, Clone, Default)]
pub struct FilmPlan {
    pub assets: AssetPlan,
    pub channels: ChannelPlan,
}

impl FilmPlan {
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty() && self.channels.is_empty()
    }

    /// F-8 扩链：资产内容哈希 + 通道规范字节进产物哈希。空 plan 恒等返回（零损失）。
    pub fn mix_artifact_hash(&self, h: u64) -> u64 {
        if self.is_empty() {
            return h;
        }
        self.channels.mix_artifact_hash(self.assets.mix_artifact_hash(h))
    }
}

/// 行式扫描结果：`masked_source` 把声明行替换为等长空格行（行号逐字不变），
/// 交 W1 的 `tokenize3` / `Parser3` 原样处理 —— 与 W3 `scan_finish` 同一手法，
/// 零改动 W1 冻结的 LL(1)+Pratt 前端。
#[derive(Debug, Clone)]
pub struct FilmScan {
    pub plan: FilmPlan,
    pub masked_source: String,
}

// ---------------------------------------------------------------------------
// 三段式错误（P95xx：W4 film 码段，与 W1 P6/P7、W2 P8、W3 P90 互不冲突）
// ---------------------------------------------------------------------------

/// 三段式错误构造（在哪 / 什么事 / 怎么改，全部带行号）。
pub fn film_err(line: usize, code: &str, what: &str, fix: &str) -> DslError {
    err_at(line, format!("[{code}] 第 {line} 行：{what}\n  建议：{fix}"))
}

// ---------------------------------------------------------------------------
// 行式扫描 + 遮罩
// ---------------------------------------------------------------------------

/// 从双引号内取出字符串内容（自研、零依赖；不闭合即返回 None）。
fn quoted(s: &str) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] != b'"' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let start = i + 1;
    let mut j = start;
    while j < bytes.len() && bytes[j] != b'"' {
        j += 1;
    }
    if j >= bytes.len() {
        return None;
    }
    Some((s[start..j].to_string(), j + 1))
}

fn is_hex16(s: &str) -> bool {
    s.len() == HASH_HEX_LEN && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_hash_u64(s: &str) -> Option<u64> {
    if !is_hex16(s) {
        return None;
    }
    u64::from_str_radix(s, 16).ok()
}

fn decode_hex_bytes(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0usize;
    while i < b.len() {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

/// 解析一条 `asset` 声明（单行）；返回 `AssetDecl` 或三段式错误。
fn parse_asset_line(words: &[&str], raw: &str, line: usize) -> Result<AssetDecl, DslError> {
    // 形态：asset "name" kind <k> hash "<hex>" [external "<path>" | inline "<hex>"]
    let (name, _) = quoted(raw).ok_or_else(|| {
        film_err(line, "P9501", "asset 声明需要 `asset \"<name>\" ...` 形式的双引号名称",
                 "写成 `asset \"lut\" kind exr hash \"<16位hex>\" external \"assets/lut.exr\"`")
    })?;
    if name.is_empty() {
        return Err(film_err(line, "P9501", "asset 名称不能为空", "给资产一个有意义的名称，如 `\"lut\"`"));
    }
    // 定位 kind / hash / 来源字段（跳过前两个 token：asset、"name"）。
    // 所有引号值统一从 words 取（避免从 raw 里 find 关键字造成子串误匹配）。
    let quoted_val = |idx: usize, field: &str, what: &str, fix: &str| -> Result<String, DslError> {
        let v = words.get(idx + 1).copied().ok_or_else(|| {
            film_err(line, "P9501", &format!("asset 的 `{field}` {what}"), fix)
        })?;
        let trimmed = v.trim_matches('"');
        if v == trimmed || trimmed.is_empty() {
            return Err(film_err(line, "P9501",
                                &format!("asset 的 `{field}` 需要双引号取值，实际 '{v}'"), fix));
        }
        Ok(trimmed.to_string())
    };
    let mut idx = 2usize;
    let mut kind: Option<AssetKind> = None;
    let mut hash: Option<u64> = None;
    let mut source = AssetSource::HashOnly;
    while idx < words.len() {
        match words[idx] {
            "kind" => {
                // kind 是裸枚举 token（exr / font / audio），不带引号。
                let v = words.get(idx + 1).copied().ok_or_else(|| {
                    film_err(line, "P9501", "asset 的 `kind` 缺少取值",
                             "写 `kind exr` / `kind font` / `kind audio` 三选一")
                })?;
                kind = Some(AssetKind::parse(v).ok_or_else(|| {
                    film_err(line, "P9502", &format!("asset '{name}' 的 kind '{v}' 非法"),
                             "kind 只能是 exr / font / audio；不认识的种类不静默降级、不取默认")
                })?);
                idx += 2;
            }
            "hash" => {
                let hv = quoted_val(idx, "hash", "缺少内容哈希",
                                    "写 `hash \"<16位hex>\"`，由资产内容用 hash_mix_bytes 派生，进产物哈希")?;
                hash = Some(parse_hash_u64(&hv).ok_or_else(|| {
                    film_err(line, "P9503", &format!("asset '{name}' 的内容哈希 '{hv}' 非法"),
                             "内容哈希必须是 16 位十六进制（u64），例如 `\"9e3779b97f4a7c15\"`；不取默认")
                })?);
                idx += 2;
            }
            "external" => {
                let path = quoted_val(idx, "external", "需要双引号路径",
                                      "写 `external \"assets/lut.exr\"`（相对路径，语言层不读文件）")?;
                if path.starts_with('/') || path.starts_with('\\') || path.contains("..") || path.contains(':') {
                    return Err(film_err(line, "P9506",
                                        &format!("asset '{name}' 的外部路径 '{path}' 是绝对路径或含 `..`/盘符"),
                                        "用仓库内相对路径；绝对/逃逸路径会引入隐式环境状态，确定性破口"));
                }
                source = AssetSource::External(path);
                idx += 2;
            }
            "inline" => {
                let hex = quoted_val(idx, "inline", "需要双引号 hex 字节",
                                     "写 `inline \"0a1b2c\"`（偶数位十六进制），内嵌资产进产物哈希")?;
                let bytes = decode_hex_bytes(&hex).ok_or_else(|| {
                    film_err(line, "P9503", &format!("asset '{name}' 的 inline 内容 '{hex}' 不是合法 hex"),
                             "inline 必须是偶数位十六进制字节串，如 `\"0a1b2c\"`")
                })?;
                source = AssetSource::Inline(bytes);
                idx += 2;
            }
            other => {
                return Err(film_err(line, "P9501", &format!("asset '{name}' 出现未知字段 '{other}'"),
                                    "asset 只允许 name / kind / hash / external / inline；未知字段不静默跳过"));
            }
        }
    }
    let kind = kind.ok_or_else(|| {
        film_err(line, "P9501", &format!("asset '{name}' 漏填必填字段 `kind`"),
                 "补 `kind exr|font|audio`；漏必填不取默认、不静默降级")
    })?;
    let content_hash = hash.ok_or_else(|| {
        film_err(line, "P9503", &format!("asset '{name}' 漏填内容哈希 `hash`（内容哈希锁定）"),
                 "补 `hash \"<16位hex>\"`；资产以内容哈希锁定，缺哈希即结构化报错")
    })?;
    // 内容哈希锁定自洽：内嵌字节的 hash_mix_bytes 必须等于声明 hash。
    if let AssetSource::Inline(bytes) = &source {
        let computed = hash_mix_bytes(0, bytes);
        if computed != content_hash {
            return Err(film_err(line, "P9504",
                                &format!("asset '{name}' 的 inline 内容与声明 hash 不符（内容哈希锁定）：声明 {content_hash:016x}，实际 {computed:016x}"),
                                "把 hash 改成内嵌内容的 hash_mix_bytes 值，或修正 inline 字节；不允许静默替换"));
        }
    }
    Ok(AssetDecl { name, kind, content_hash, source, line })
}

/// 解析一条 `channel` 声明（单行）；支持 `expr "<...>"` 与 `key ...` 两种时序源。
fn parse_channel_line(raw: &str, line: usize) -> Result<ChannelDecl, DslError> {
    let (target, after_name) = quoted(raw).ok_or_else(|| {
        film_err(line, "P9511", "channel 声明需要 `channel \"<target>\" ...` 形式的双引号目标",
                 "写成 `channel \"w3.motion.strength\" expr \"0.3 + 0.2 * t\"`")
    })?;
    if target.is_empty() {
        return Err(film_err(line, "P9511", "channel 的目标参数路径不能为空",
                            "给出任意参数路径，如 `\"R-2.roughness\"` 或 `\"w3.motion.strength\"`"));
    }
    let body = &raw[after_name..];
    let trimmed = body.trim();
    if let Some(rest) = trimmed.strip_prefix("expr") {
        let (src, _) = quoted(rest).ok_or_else(|| {
            film_err(line, "P9512", "channel 的 `expr` 需要双引号解析式",
                     "写 `expr \"0.3 + 0.2 * t\"`，解析式支持 t 与 + - * / ( )")
        })?;
        let expr = Expr::parse(&src).map_err(|e| {
            film_err(line, "P9512", &format!("channel '{target}' 的解析式 '{src}' 非法（{e:?}）"),
                     "只用 t 与 + - * / ( ) 的受限解析式，例如 `\"0.3 + 0.2 * t\"`")
        })?;
        Ok(ChannelDecl { target, channel: PropertyChannel::Expr(expr), line })
    } else if trimmed.starts_with("key") {
        // 形态：key <t>s <v> <easing> key <t>s <v> <easing> ...
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        let mut keys: Vec<Keyframe> = Vec::new();
        let mut i = 0usize;
        while i < words.len() {
            if words[i] != "key" {
                return Err(film_err(line, "P9512",
                                    &format!("channel '{target}' 的关键帧序列在 '{words:?}' 处形态非法"),
                                    "写成 `key 0s 0.2 linear key 1s 0.8 linear`"));
            }
            let t = words.get(i + 1).copied().ok_or_else(|| {
                film_err(line, "P9512", "关键帧缺时间", "写 `key 0s 0.2 linear` 形式")
            })?;
            let v = words.get(i + 2).copied().ok_or_else(|| {
                film_err(line, "P9512", "关键帧缺取值", "写 `key 0s 0.2 linear` 形式")
            })?;
            let ease = words.get(i + 3).copied().unwrap_or("linear");
            let ts = t.strip_suffix('s').unwrap_or(t);
            let time: f32 = ts.parse().map_err(|_| {
                film_err(line, "P9512", &format!("关键帧时间 '{t}' 非法"), "时间写 `0s` / `1s` 等秒值")
            })?;
            let value: f32 = v.parse().map_err(|_| {
                film_err(line, "P9512", &format!("关键帧取值 '{v}' 非法"), "取值写有限十进制数值")
            })?;
            if !time.is_finite() || !value.is_finite() {
                return Err(film_err(line, "P9512", "关键帧时间/取值必须有限", "不允许 NaN / inf 混进产物"));
            }
            let easing = if ease == "linear" {
                EasingKind::Linear
            } else {
                return Err(film_err(line, "P9512", &format!("关键帧缓动 '{ease}' 非法"),
                                    "W4 通道只用 `linear`（其余缓动族由 W2 `cap` 关键帧既有链路提供）"));
            };
            keys.push(Keyframe::new(time, value, easing));
            i += 4;
        }
        if keys.len() < 2 {
            return Err(film_err(line, "P9512", "关键帧通道至少需要两个 key 才能随时间变化",
                                "补一个更晚时间的 key，例如 `key 0s 0.2 linear key 1s 0.8 linear`"));
        }
        Ok(ChannelDecl { target, channel: PropertyChannel::Track(Track::new(keys)), line })
    } else {
        Err(film_err(line, "P9511", &format!("channel '{target}' 未声明时序源"),
                     "必须写 `expr \"...\"` 或 `key 0s ... linear key 1s ... linear`；不取默认"))
    }
}

/// 引用闭环门禁：`asset_ref "<name>"` 使用点必须指向已声明资产（缺失即报错）。
fn asset_ref_gate(all_lines: &[&str], plan: &AssetPlan) -> Result<(), DslError> {
    for (idx, raw) in all_lines.iter().enumerate() {
        let line = idx + 1;
        let t = raw.trim();
        if let Some(rest) = t.strip_prefix("asset_ref") {
            let (name, _) = quoted(rest).ok_or_else(|| {
                film_err(line, "P9507", "asset_ref 需要 `asset_ref \"<name>\"` 形式的双引号名称",
                         "写 `asset_ref \"lut\"` 引用本文件已声明的资产")
            })?;
            if !plan.has(&name) {
                return Err(film_err(line, "P9507",
                                    &format!("引用了未声明的资产 '{name}'（资产缺失）"),
                                    &format!("在本文件补 `asset \"{name}\" kind exr|font|audio hash \"<16hex>\" ...`；缺失不静默降级、不取默认")));
            }
        }
    }
    Ok(())
}

/// 扫描 `// prism-dsl 2` 源码里的 `asset` / `channel` / `asset_ref` 顶层声明。
///
/// - 只识别**顶层（无前导空白）**行，避免误伤 `emit f\"\"\"...\"\"\"` 内的 v2 文本。
/// - 声明行替换为**等长空格行**（行号逐字不变），交给 W1 前端原样处理。
/// - 未声明 `asset`/`channel` 的文件 ⇒ `masked_source` 与输入逐字节相同 ⇒ 零损失。
pub fn scan_film(source: &str) -> Result<FilmScan, DslError> {
    let raw_lines: Vec<&str> = source.lines().collect();
    let mut masked: Vec<String> = raw_lines
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut assets: Vec<AssetDecl> = Vec::new();
    let mut channels: Vec<ChannelDecl> = Vec::new();
    let mut seen_assets: BTreeSet<String> = BTreeSet::new();
    let mut seen_channels: BTreeSet<String> = BTreeSet::new();

    for (idx, raw) in raw_lines.iter().enumerate() {
        let line = idx + 1;
        // 只认顶层（无前导空白）声明；emit 模板内的 v2 文本行通常带缩进。
        let is_top = !raw.starts_with(' ') && !raw.starts_with('\t');
        if !is_top {
            continue;
        }
        let t = raw.trim();
        if t.starts_with("asset ") {
            let words: Vec<&str> = t.split_whitespace().collect();
            let decl = parse_asset_line(&words, raw, line)?;
            if !seen_assets.insert(decl.name.clone()) {
                return Err(film_err(line, "P9505", &format!("资产 '{}' 重复声明", decl.name),
                                    "同一资产名只声明一次；重复不静默合并"));
            }
            assets.push(decl);
            mask_line(&mut masked, idx, raw);
        } else if t.starts_with("asset_ref") {
            // 稍后由 asset_ref_gate 校验；此处只遮罩。
            mask_line(&mut masked, idx, raw);
        } else if t.starts_with("channel ") {
            let decl = parse_channel_line(raw, line)?;
            if !seen_channels.insert(decl.target.clone()) {
                return Err(film_err(line, "P9513", &format!("通道目标 '{}' 重复声明", decl.target),
                                    "同一目标参数只绑一条命名通道；重复不静默覆盖"));
            }
            channels.push(decl);
            mask_line(&mut masked, idx, raw);
        }
    }
    let plan = FilmPlan {
        assets: AssetPlan { assets },
        channels: ChannelPlan { channels },
    };
    // 引用闭环：资产使用点必须指向本文件已声明资产（缺失即三段式报错带行号）。
    asset_ref_gate(&raw_lines, &plan.assets)?;

    let mut masked_source = masked.join("\n");
    if source.ends_with('\n') {
        masked_source.push('\n');
    }
    Ok(FilmScan { plan, masked_source })
}

/// 把声明行清空为空行（与 W3 `scan_finish` 的 `masked[idx].clear()` 同款）：
/// Vec 每行一个元素 ⇒ 行号逐字不变；空行在 v3 lexer 里被跳过，不引入缩进层级。
/// 用空行而非等长空格行，避免「全空格行」被词法层当成带缩进的空语句（P6001）。
fn mask_line(masked: &mut [String], idx: usize, _raw: &str) {
    masked[idx].clear();
}

// ---------------------------------------------------------------------------
// CLI / 凭证：供 `prism render` 打印、供 W4 集成测试断言（纯函数、无副作用）
// ---------------------------------------------------------------------------

/// film 摘要（结构化，供 CLI 与测试）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmSummary {
    /// 资产数（内容哈希锁定）。
    pub asset_count: usize,
    /// 命名通道目标列表（声明序）。
    pub targets: Vec<String>,
    /// 资产规范字节的确定性摘要（不含路径，可断言换路径不变）。
    pub asset_digest: u64,
}

/// 从源码提取 film 摘要；非法声明返回三段式错误；旧文件（无声明）返回空摘要。
pub fn film_summary(source: &str) -> Result<FilmSummary, DslError> {
    let scan = scan_film(source)?;
    Ok(FilmSummary {
        asset_count: scan.plan.assets.len(),
        targets: scan.plan.channels.channels.iter().map(|c| c.target.clone()).collect(),
        asset_digest: hash_mix_bytes(FILM_TAG, &scan.plan.assets.canonical_bytes()),
    })
}

/// 渲染前的时序覆盖层（由 `movie.rs::wiring_frame_values` 调用，生成源头，非帧后）。
///
/// 遍历 `CompileV2Output.channels` 里带 `w4.drive.` 前缀的命名通道，用既有
/// `PropertyChannel::sample_ctx` 逐帧采样，把值以位型写进 `WiringFrame` 的
/// `<target>` 键；对六族痕迹强度 `w3.<族>.strength`，同步按 W3 同一公式
/// `signal = (noise*2-1)*strength` 重算 `w3.<族>.signal`（只用 W3 已注入的
/// `noise` 键，不新造噪声、不贴皮）。
///
/// **零损失**：无 `w4.drive.*` 通道时立即恒等返回；analytic 档没有 `w3.*.noise`
/// 键 ⇒ 强度覆盖不产生任何噪点键，逐字节同纯解析基线。
pub fn apply_timed_channels(
    wf: &mut crate::WiringFrame,
    channels: &BTreeMap<&str, &PropertyChannel>,
    t: f32,
    frame: u64,
) {
    let ctx = EvalCtx::new(0, 0, frame);
    // 收集键名，避免在持有 channels 借用时修改 wf。
    let drives: Vec<(String, u32)> = channels
        .iter()
        .filter_map(|(name, ch)| {
            name.strip_prefix(DRIVE_PREFIX)
                .map(|target| (target.to_string(), ch.sample_ctx(t, &ctx).to_bits()))
        })
        .collect();
    if drives.is_empty() {
        return;
    }
    for (target, bits) in drives {
        let strength = f32::from_bits(bits);
        wf.values.insert(target.clone(), bits);
        // 六族痕迹强度：仅当该族已被 W3 在渲染前注入 noise（crafted 且 active）时重算 signal。
        if let Some(fam) = target.strip_prefix("w3.").and_then(|s| s.strip_suffix(".strength")) {
            let noise_key = format!("w3.{fam}.noise");
            let signal_key = format!("w3.{fam}.signal");
            if let Some(noise_bits) = wf.values.get(&noise_key).copied() {
                let raw = f32::from_bits(noise_bits);
                let signal = (raw * 2.0 - 1.0) * strength;
                wf.values.insert(signal_key, signal.to_bits());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D-27 外部资产解析（调度层注入，语言层零隐式环境状态）
// ---------------------------------------------------------------------------

/// 资产解析器：把「外部引用」这个名字解析成资产**字节**，供内容哈希校验。
///
/// 语言层（`scan_film`）只读声明、不碰文件系统；真正的 IO 由调用方注入本 trait，
/// 因此「绝对路径 / 工作目录 / 文件是否存在」都不可能成为隐式环境状态——它们
/// 只影响**校验通过与否**，且校验结果只有「一致 / 结构化报错」两种，产物字节
/// 恒由声明的内容哈希决定。空环境（无资产文件）用 `NullResolver` 即可出片。
pub trait AssetResolver {
    /// 按资产名与声明的相对路径解析字节；`None` = 资产缺失。
    fn resolve(&self, name: &str, path: &str) -> Option<Vec<u8>>;
}

/// 空解析器：不解析任何文件（用于编译期扫描 / 零损失回归 / 内嵌资产片子）。
/// 声明合法性、inline 内容哈希自洽仍由 `scan_film` 全量校验。
pub struct NullResolver;

impl AssetResolver for NullResolver {
    fn resolve(&self, _name: &str, _path: &str) -> Option<Vec<u8>> {
        None
    }
}

/// 确定性目录解析器：从给定根目录读相对路径（路径越界已在 P9506 拦截）。
/// 只读、零缓存、零系统时间，同输入同字节。
pub struct DirResolver {
    pub root: PathBuf,
}

impl DirResolver {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl AssetResolver for DirResolver {
    fn resolve(&self, _name: &str, path: &str) -> Option<Vec<u8>> {
        std::fs::read(self.root.join(path)).ok()
    }
}

/// 用给定解析器校验所有「外部引用」资产的内容哈希锁定：
/// - 缺失（解析不到字节）⇒ `P9508` 三段式带声明行号；
/// - 内容与声明哈希不符 ⇒ `P9509` 三段式带声明行号；
/// - 一致 ⇒ 逐字节通过。
///
/// **不静默降级、不取默认**：绝不因缺文件就跳过校验、绝不用路径/文件名补哈希。
/// 内嵌（`Inline`）与纯哈希（`HashOnly`）资产不在此列（前者已在 `scan_film`
/// 校验自洽，后者本就是纯声明）。返回通过校验的外部资产数（CLI 凭证用）。
pub fn verify_external_assets(
    plan: &AssetPlan,
    resolver: &dyn AssetResolver,
) -> Result<usize, DslError> {
    let mut verified = 0usize;
    for a in &plan.assets {
        if let AssetSource::External(path) = &a.source {
            let bytes = resolver.resolve(&a.name, path).ok_or_else(|| {
                film_err(a.line, "P9508",
                         &format!("资产 '{}' 的外部引用缺失：`{}` 解析不到字节", a.name, path),
                         "把资产放到声明的相对路径下，或改用 `inline \"<hex>\"` 内嵌；缺失不静默降级、不取默认")
            })?;
            let computed = hash_mix_bytes(0, &bytes);
            if computed != a.content_hash {
                return Err(film_err(a.line, "P9509",
                                    &format!("资产 '{}' 内容与声明哈希不符：声明 {:016x}，实际 {:016x}",
                                             a.name, a.content_hash, computed),
                                    "把声明 hash 改成资产内容的 hash_mix_bytes(0, bytes) 值，或换成内容一致的资产；不静默替换"));
            }
            verified += 1;
        }
    }
    Ok(verified)
}

/// 红线 4 编译期门禁：`analytic` 档 / 未声明 `finish` 时，禁止把 `w3.*` 痕迹
/// 参数绑成时序驱动（不许为了「片子好看」给匀质档偷加噪）。
/// `crafted` 由调用方从 `WiringPlan::finish_plan()` 得出；本函数只做静态拒绝。
pub fn gate_analytic_trace_drives(plan: &FilmPlan, crafted: bool) -> Result<(), DslError> {
    if crafted {
        return Ok(());
    }
    for c in &plan.channels.channels {
        if c.target.starts_with("w3.") {
            return Err(film_err(c.line, "P9514",
                                &format!("analytic 档不得声明 '{}' 痕迹时序驱动（红线 4：不许给匀质档偷加噪）", c.target),
                                "删掉该 w3.* 通道，或显式声明 `finish = crafted { ... }`；编译器不替用户选档"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 单元测试（模块内）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn three_part(err: &DslError, code: &str) {
        assert!(err.line > 0, "必须带行号");
        assert!(err.message.contains(code), "必须带错误码 {code}：{}", err.message);
        assert!(err.message.contains("第"), "必须点明在哪：{}", err.message);
        assert!(err.message.contains("建议："), "必须有可行动建议段：{}", err.message);
    }

    #[test]
    fn empty_source_is_identity() {
        let src = "// prism-dsl 2\nseed = 42\nfinish = analytic\nemit f\"\"\"\nmovie \"g\" { duration 1s fps 8 resolution 160x120 format png }\n\"\"\"\n";
        let scan = scan_film(src).unwrap();
        assert!(scan.plan.is_empty());
        assert_eq!(scan.masked_source, src, "无声明时遮罩后源码逐字节相同（零损失）");
        let h = 0x1234_5678_9abc_def0u64;
        assert_eq!(scan.plan.mix_artifact_hash(h), h, "空 plan 产物哈希恒等");
    }

    #[test]
    fn asset_content_hash_is_locked_and_path_is_not_environment_state() {
        let bytes: Vec<u8> = vec![0x0a, 0x1b, 0x2c, 0x3d];
        let h = hash_mix_bytes(0, &bytes);
        let a = format!("asset \"lut\" kind exr hash \"{h:016x}\" inline \"0a1b2c3d\"\n");
        let scan = scan_film(&a).unwrap();
        assert_eq!(scan.plan.assets.len(), 1);
        // 换 external 路径、同 kind/hash ⇒ 规范字节相同（路径不是隐式环境状态）。
        let b = format!("asset \"lut\" kind exr hash \"{h:016x}\" external \"totally/different/path.exr\"\n");
        let scan_b = scan_film(&b).unwrap();
        // inline 与 HashOnly/External 的来源标记不同 ⇒ 用同类来源对比才严格；
        // 这里断言「路径文本本身不出现在规范字节里」。
        assert!(!scan_b.plan.assets.canonical_bytes().windows(4)
                 .any(|w| w == b"path"), "路径字符串绝不进产物哈希");
        let c = format!("asset \"lut\" kind exr hash \"{h:016x}\" external \"assets/a.exr\"\n");
        let d = format!("asset \"lut\" kind exr hash \"{h:016x}\" external \"assets/b.exr\"\n");
        assert_eq!(scan_film(&c).unwrap().plan.assets.canonical_bytes(),
                   scan_film(&d).unwrap().plan.assets.canonical_bytes(),
                   "换路径、同内容哈希 ⇒ 规范字节逐字节相同");
        // 改 hash ⇒ 产物哈希变。
        let e = format!("asset \"lut\" kind exr hash \"{:016x}\" external \"assets/a.exr\"\n", h ^ 1);
        assert_ne!(scan_film(&c).unwrap().plan.mix_artifact_hash(7),
                   scan_film(&e).unwrap().plan.mix_artifact_hash(7));
    }

    #[test]
    fn inline_hash_mismatch_is_three_part_error() {
        let src = "asset \"lut\" kind exr hash \"0000000000000000\" inline \"0a1b2c\"\n";
        let err = scan_film(src).unwrap_err();
        three_part(&err, "P9504");
    }

    #[test]
    fn missing_fields_and_unknown_kind_all_fail() {
        for (src, code) in [
            ("asset \"lut\" kind exr\n", "P9503"),                        // 漏 hash
            ("asset \"lut\" hash \"9e3779b97f4a7c15\"\n", "P9501"),       // 漏 kind
            ("asset \"lut\" kind png hash \"9e3779b97f4a7c15\"\n", "P9502"), // 未知 kind
            ("asset \"lut\" kind exr hash \"xyz\"\n", "P9503"),           // 非法 hash
            ("asset \"lut\" kind exr hash \"9e3779b97f4a7c15\" bogus 1\n", "P9501"), // 未知字段
            ("asset \"lut\" kind exr hash \"9e3779b97f4a7c15\" external \"/abs/x\"\n", "P9506"),
            ("asset_ref \"missing\"\n", "P9507"),                         // 引用未声明
        ] {
            let err = scan_film(src).unwrap_err();
            three_part(&err, code);
        }
    }

    #[test]
    fn channels_sample_from_existing_property_channel() {
        let src = "channel \"w3.motion.strength\" expr \"0.3 + 0.2 * t\"\n\
                   channel \"R-2.roughness\" key 0s 0.2 linear key 1s 0.8 linear\n";
        let scan = scan_film(src).unwrap();
        assert_eq!(scan.plan.channels.len(), 2);
        let f0 = scan.plan.channels.frame_values(0.0, 0, 0, 0);
        let f4 = scan.plan.channels.frame_values(0.5, 4, 0, 0);
        assert_ne!(f0["w3.motion.strength"], f4["w3.motion.strength"], "逐帧值随时序变");
        assert_ne!(f0["R-2.roughness"], f4["R-2.roughness"]);
        // 逐帧值与既有 PropertyChannel::sample(t) 对拍一致（同源、不两端同错）。
        let expr = Expr::parse("0.3 + 0.2 * t").unwrap();
        assert_eq!(f32::from_bits(f4["w3.motion.strength"]).to_bits(), expr.sample(0.5).to_bits());
        // 降级成 NamedChannel ⇒ 名字带前缀、可进既有 channels 表。
        let nc = scan.plan.channels.into_named_channels();
        assert_eq!(nc.len(), 2);
        assert!(nc.iter().all(|c| c.name.starts_with(DRIVE_PREFIX)));
        assert_eq!(f0, f0.clone());
        // 两跑逐字节（BTreeMap 固定序）。
        assert_eq!(scan.plan.channels.frame_values(0.5, 4, 0, 0), f4);
    }

    #[test]
    fn channel_bad_definitions_are_three_part() {
        for (src, code) in [
            ("channel \"x\"\n", "P9511"),
            ("channel \"x\" expr \"1 & 2\"\n", "P9512"),
            ("channel \"x\" key 0s 0.2 linear\n", "P9512"),
            ("channel \"x\" key 0s 0.2 quad key 1s 0.8 linear\n", "P9512"),
        ] {
            let err = scan_film(src).unwrap_err();
            three_part(&err, code);
        }
    }

    #[test]
    fn apply_timed_channels_is_identity_without_drives() {
        let mut wf = crate::WiringFrame::default();
        let empty: BTreeMap<&str, &PropertyChannel> = BTreeMap::new();
        apply_timed_channels(&mut wf, &empty, 0.5, 4);
        assert!(wf.values.is_empty(), "无驱动通道恒等、零噪点");
    }
}
