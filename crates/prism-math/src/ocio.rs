//! V2.5-B1：自研 OCIO `config.ocio` 解析（零第三方依赖）。
//!
//! # 为什么自研
//! `config.ocio` 是 **YAML 风格**的配置，但本项目零第三方依赖（连 `serde_yaml`
//! / `yaml-rust` 都不许）。本模块自带一个**缩进敏感的最小 YAML 子集解析器**
//! （块映射 / 块序列 / 行内 `{...}` / `[...]` / `!<Tag>` / 引号串 / 注释），
//! 再把解析出的 [`Yaml`] 树逐字段校验成 [`OcioConfig`]。
//!
//! # 覆盖范围（需求池 V2.5-B1 / 矩阵 E-7）
//! - 顶层：`ocio_profile_version` / `search_path` / `strictparsing` /
//!   `roles` / `displays` + `views` / `looks` / `colorspaces`；
//! - 色彩空间：`to_scene_reference` / `from_scene_reference` 变换链；
//! - transform 链六类：[`Transform::File`]（LUT）、[`Transform::Matrix`]、
//!   [`Transform::Exponent`]、[`Transform::Log`]、[`Transform::Range`]、
//!   [`Transform::Group`]（`children` 递归）；
//! - display / view 名字、look 的 `process_space` 与 transform 全保留。
//!
//! # 错误纪律（C18/C20）
//! 任何解析不下来的语法、未知 transform、错矩阵长度、错 LUT 尺寸、非有限数、
//! 结构不自洽都返回 [`OcioError`]（`OC1xxx` 三段式：在哪 / 什么事 / 怎么改），
//! **不 panic、不静默吞**：未识别的顶层键进 [`OcioConfig::ignored_keys`] 显式记账。
//!
//! # 确定性
//! 所有集合按**源文件出现顺序**保存（`Vec`），不做哈希迭代；数值解析走
//! `f64::from_str`（整数/十进制字面量，固定求值），不调用任何平台 libm。
//! 同输入解析两遍逐字段一致（判据 V5）。

use std::fmt;

/// OCIO 解析 / 装配错误（三段式）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcioError {
    /// 错误码（`OC1xxx`）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 行号。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for OcioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for OcioError {}

impl OcioError {
    /// 构造错误（四字段全填）。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }

    /// OC1001：词法/缩进错误。
    pub fn syntax(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1001",
            format!("ocio::parse 第 {line} 行"),
            problem,
            "检查缩进是否用空格（本解析器不接受 Tab）、`key: value` 是否成对、\
             行内 `{...}` / `[...]` 是否闭合；参考 OCIO config.ocio 的缩进层级",
        )
    }

    /// OC1002：数值解析失败 / 非有限数。
    pub fn number(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1002",
            format!("ocio::number 第 {line} 行"),
            problem,
            "把该值写成有限十进制数（如 `0.45`、`1e-3`）；不接受 NaN / Inf / 十六进制",
        )
    }

    /// OC1003：结构不自洽（缺键 / 键类型错 / 未知顶层键在严格模式下）。
    pub fn structure(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1003",
            format!("ocio::structure 第 {line} 行"),
            problem,
            "核对 config.ocio 的层级：`colorspaces` / `looks` 是序列，\
             `roles` / `displays` 是映射，transform 用 `!<Tag>` 标注",
        )
    }

    /// OC1004：未知 transform 类型。
    pub fn unknown_transform(line: usize, tag: &str) -> Self {
        Self::new(
            "OC1004",
            format!("ocio::transform 第 {line} 行"),
            format!("未注册的 transform 类型 `!<{tag}>`"),
            "只允许 FileTransform / MatrixTransform / ExponentTransform / \
             LogTransform / RangeTransform / GroupTransform 六类；\
             其余 transform 必须先在 ocio.rs 注册并实现再启用，禁止静默跳过",
        )
    }

    /// OC1005：矩阵长度 / 数值不合法。
    fn matrix(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1005",
            format!("ocio::MatrixTransform 第 {line} 行"),
            problem,
            "`matrix` 必须是 16 个数（4×4 行主序），`offset` 若给出必须是 4 个数",
        )
    }

    /// OC1006：LUT 文件尺寸不合法。
    pub fn lut_size(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1006",
            format!("ocio::FileTransform 第 {line} 行"),
            problem,
            "`.cube` 须声明 `LUT_3D_SIZE n` 且恰好给 n³ 行三元组；\
             `.spi1d` 的 `Length` 须等于数据行数；检查 LUT 文件是否被截断",
        )
    }

    /// OC1007：LUT 源不可达 / 解析失败。
    pub fn lut_source(line: usize, problem: impl Into<String>) -> Self {
        Self::new(
            "OC1007",
            format!("ocio::FileTransform 第 {line} 行"),
            problem,
            "确认 `src` 指向的 LUT 文件存在于 config.ocio 同目录（或 `search_path`），\
             且后缀是 `.cube` / `.spi1d`",
        )
    }
}

/// OCIO `Result` 别名。
pub type OcioResult<T> = Result<T, OcioError>;

// ===========================================================================
//  最小 YAML 子集（缩进敏感）
// ===========================================================================

/// 解析出的 YAML 节点（只保留 OCIO 需要的四种形态）。
#[derive(Debug, Clone, PartialEq)]
pub enum Yaml {
    /// 标量（引号已剥离、`#` 注释已剥离）。
    Scalar(String),
    /// 序列。
    Seq(Vec<Yaml>),
    /// 映射（保留源顺序，键不排序）。
    Map(Vec<(String, Yaml)>),
    /// `!<Tag>` 标注的节点（tag 已去掉 `!<>`, body 是紧随其后的值）。
    Tagged { tag: String, body: Box<Yaml> },
    /// 显式空值。
    Null,
}

impl Yaml {
    /// 作为标量文本（否则 `None`）。
    pub fn as_scalar(&self) -> Option<&str> {
        match self {
            Yaml::Scalar(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// 作为映射（否则 `None`）。
    pub fn as_map(&self) -> Option<&[(String, Yaml)]> {
        match self {
            Yaml::Map(m) => Some(m.as_slice()),
            _ => None,
        }
    }

    /// 作为序列（否则 `None`）。
    pub fn as_seq(&self) -> Option<&[Yaml]> {
        match self {
            Yaml::Seq(s) => Some(s.as_slice()),
            _ => None,
        }
    }

    /// 取映射键（若本节点是 Tagged，穿透到 body）。
    pub fn get(&self, key: &str) -> Option<&Yaml> {
        match self {
            Yaml::Tagged { body, .. } => body.get(key),
            Yaml::Map(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// 取映射键并解成标量文本。
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Yaml::as_scalar)
    }
}

/// 一行（已去注释、已算缩进、保留行号）。
#[derive(Debug, Clone)]
struct Line {
    indent: usize,
    text: String,
    number: usize,
}

/// 去掉行内注释（`#` 只在引号外生效），并去掉首尾空白。
fn strip_comment(raw: &str) -> String {
    let mut quote: Option<char> = None;
    let mut out = String::with_capacity(raw.len());
    let mut prev: char = '\0';
    for ch in raw.chars() {
        match quote {
            Some(q) => {
                out.push(ch);
                if ch == q && prev != '\\' {
                    quote = None;
                }
            }
            None => match ch {
                '"' | '\'' => {
                    quote = Some(ch);
                    out.push(ch);
                }
                '#' => break,
                _ => out.push(ch),
            },
        }
        prev = ch;
    }
    out.trim().to_string()
}

fn lexical_lines(text: &str) -> OcioResult<Vec<Line>> {
    let mut out: Vec<Line> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let number: usize = idx + 1;
        if raw.contains('\t') {
            return Err(OcioError::syntax(number, "缩进里出现 Tab 字符"));
        }
        let body: String = strip_comment(raw);
        if body.is_empty() {
            continue;
        }
        let indent: usize = raw.chars().take_while(|c| *c == ' ').count();
        out.push(Line { indent, text: body, number });
    }
    Ok(out)
}

/// 行内流值解析器（`{...}` / `[...]` / 标量 / `!<Tag>`）。
struct Flow {
    chars: Vec<char>,
    pos: usize,
    line: usize,
}

impl Flow {
    fn new(s: &str, line: usize) -> Self {
        Self { chars: s.chars().collect(), pos: 0, line }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r' | '\n')) {
            self.pos += 1;
        }
    }

    fn tag(&mut self) -> OcioResult<Option<String>> {
        if self.peek() != Some('!') {
            return Ok(None);
        }
        self.pos += 1;
        if self.bump() != Some('<') {
            return Err(OcioError::syntax(self.line, "`!<Tag>` 缺少 `<`"));
        }
        let mut name = String::new();
        loop {
            match self.bump() {
                Some('>') => break,
                Some(c) => name.push(c),
                None => return Err(OcioError::syntax(self.line, "`!<Tag>` 未闭合（缺 `>`）")),
            }
        }
        self.skip_ws();
        Ok(Some(name))
    }

    fn quoted_or_bare(&mut self, stop: &[char]) -> OcioResult<String> {
        self.skip_ws();
        match self.peek() {
            Some('"') | Some('\'') => {
                let q: char = self.bump().unwrap_or('"');
                let mut s = String::new();
                let mut prev: char = '\0';
                loop {
                    match self.bump() {
                        Some(c) if c == q && prev != '\\' => break,
                        Some(c) => {
                            s.push(c);
                            prev = c;
                        }
                        None => {
                            return Err(OcioError::syntax(self.line, "引号字符串未闭合"));
                        }
                    }
                }
                Ok(s)
            }
            _ => {
                let mut s = String::new();
                while let Some(c) = self.peek() {
                    if stop.contains(&c) {
                        break;
                    }
                    s.push(c);
                    self.pos += 1;
                }
                Ok(s.trim().to_string())
            }
        }
    }

    fn value(&mut self) -> OcioResult<Yaml> {
        self.skip_ws();
        let tag: Option<String> = self.tag()?;
        let node: Yaml = match self.peek() {
            Some('{') => {
                self.pos += 1;
                let mut m: Vec<(String, Yaml)> = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek() == Some('}') {
                        self.pos += 1;
                        break;
                    }
                    if self.peek().is_none() {
                        return Err(OcioError::syntax(self.line, "行内 `{` 未闭合"));
                    }
                    let key: String = self.quoted_or_bare(&[':', ',', '}'])?;
                    self.skip_ws();
                    if self.bump() != Some(':') {
                        return Err(OcioError::syntax(
                            self.line,
                            format!("行内映射键 `{key}` 后缺 `:`"),
                        ));
                    }
                    let val: Yaml = self.value()?;
                    m.push((key, val));
                    self.skip_ws();
                    match self.peek() {
                        Some(',') => {
                            self.pos += 1;
                        }
                        Some('}') => {
                            self.pos += 1;
                            break;
                        }
                        None => return Err(OcioError::syntax(self.line, "行内 `{` 未闭合")),
                        Some(c) => {
                            return Err(OcioError::syntax(
                                self.line,
                                format!("行内映射里出现意外字符 `{c}`（期望 `,` 或 `}}`）"),
                            ))
                        }
                    }
                }
                Yaml::Map(m)
            }
            Some('[') => {
                self.pos += 1;
                let mut s: Vec<Yaml> = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek() == Some(']') {
                        self.pos += 1;
                        break;
                    }
                    if self.peek().is_none() {
                        return Err(OcioError::syntax(self.line, "行内 `[` 未闭合"));
                    }
                    s.push(self.value()?);
                    self.skip_ws();
                    match self.peek() {
                        Some(',') => {
                            self.pos += 1;
                        }
                        Some(']') => {
                            self.pos += 1;
                            break;
                        }
                        None => return Err(OcioError::syntax(self.line, "行内 `[` 未闭合")),
                        Some(c) => {
                            return Err(OcioError::syntax(
                                self.line,
                                format!("行内序列里出现意外字符 `{c}`（期望 `,` 或 `]`）"),
                            ))
                        }
                    }
                }
                Yaml::Seq(s)
            }
            Some(_) => {
                let s: String = self.quoted_or_bare(&[',', '}', ']'])?;
                if s.is_empty() || s == "null" || s == "~" {
                    Yaml::Null
                } else {
                    Yaml::Scalar(s)
                }
            }
            None => {
                if tag.is_some() {
                    Yaml::Null
                } else {
                    return Err(OcioError::syntax(self.line, "期望一个值，实际到行尾"));
                }
            }
        };
        Ok(match tag {
            Some(t) => Yaml::Tagged { tag: t, body: Box::new(node) },
            None => node,
        })
    }

    fn parse_all(mut self) -> OcioResult<Yaml> {
        let v: Yaml = self.value()?;
        self.skip_ws();
        if self.pos != self.chars.len() {
            return Err(OcioError::syntax(
                self.line,
                format!("行内值后有残留字符 `{}`", self.chars[self.pos..].iter().collect::<String>()),
            ));
        }
        Ok(v)
    }
}

/// 缩进敏感解析器：块映射 / 块序列 / `!<Tag>`。
struct BlockParser {
    lines: Vec<Line>,
    pos: usize,
}

impl BlockParser {
    fn new(lines: Vec<Line>) -> Self {
        Self { lines, pos: 0 }
    }

    fn peek(&self) -> Option<&Line> {
        self.lines.get(self.pos)
    }

    /// 解析一个块（indent 由调用方指定，取当前行缩进）。
    fn parse_block(&mut self, indent: usize) -> OcioResult<Yaml> {
        let Some(first) = self.peek().cloned() else {
            return Ok(Yaml::Null);
        };
        if first.indent < indent {
            return Ok(Yaml::Null);
        }
        if first.text.starts_with("- ") || first.text == "-" {
            self.parse_seq(indent)
        } else {
            self.parse_map(indent)
        }
    }

    fn parse_seq(&mut self, indent: usize) -> OcioResult<Yaml> {
        let mut items: Vec<Yaml> = Vec::new();
        loop {
            let Some(line) = self.peek().cloned() else { break };
            if line.indent != indent || !(line.text.starts_with("- ") || line.text == "-") {
                break;
            }
            self.pos += 1;
            let rest: String = line.text.strip_prefix("- ").unwrap_or("").trim().to_string();
            let mut node: Yaml = if rest.is_empty() {
                self.parse_child_block(indent + 2)?
            } else {
                Flow::new(&rest, line.number).parse_all()?
            };
            // 序列项后若有更深缩进的续行（`- !<ColorSpace>` 后跟字段），并入映射。
            node = self.merge_continuation(node, indent, line.number)?;
            items.push(node);
        }
        Ok(Yaml::Seq(items))
    }

    fn parse_map(&mut self, indent: usize) -> OcioResult<Yaml> {
        let mut entries: Vec<(String, Yaml)> = Vec::new();
        loop {
            let Some(line) = self.peek().cloned() else { break };
            if line.indent != indent || line.text.starts_with("- ") || line.text == "-" {
                break;
            }
            let (key, rest) = split_key_value(&line.text, line.number)?;
            self.pos += 1;
            let node: Yaml = if rest.is_empty() {
                self.parse_child_block(indent + 2)?
            } else {
                Flow::new(&rest, line.number).parse_all()?
            };
            let node: Yaml = self.merge_continuation(node, indent, line.number)?;
            entries.push((key, node));
        }
        Ok(Yaml::Map(entries))
    }

    /// 解析下一层（缩进严格大于父层）；若下一行没有更深缩进则空值。
    fn parse_child_block(&mut self, parent_indent: usize) -> OcioResult<Yaml> {
        match self.peek() {
            Some(next) if next.indent > parent_indent - 2 => {
                let child_indent: usize = next.indent;
                self.parse_block(child_indent)
            }
            _ => Ok(Yaml::Null),
        }
    }

    /// 把「更深缩进的续行」合并进当前节点（用于 `- !<Tag>` 后跟字段、
    /// `key: !<Tag>` 后跟字段这两种常见 OCIO 写法）。
    fn merge_continuation(&mut self, node: Yaml, indent: usize, line: usize) -> OcioResult<Yaml> {
        let Some(next) = self.peek().cloned() else { return Ok(node) };
        if next.indent <= indent {
            return Ok(node);
        }
        let cont: Yaml = self.parse_block(next.indent)?;
        let extra: Vec<(String, Yaml)> = match cont {
            Yaml::Map(m) => m,
            Yaml::Null => Vec::new(),
            other => {
                return Err(OcioError::structure(
                    line,
                    format!("缩进续行应是 `key: value` 映射，实际是 {other:?}"),
                ))
            }
        };
        Ok(match node {
            Yaml::Tagged { tag, body } => {
                let mut merged: Vec<(String, Yaml)> = match *body {
                    Yaml::Map(m) => m,
                    Yaml::Null => Vec::new(),
                    _ => Vec::new(),
                };
                merged.extend(extra);
                Yaml::Tagged { tag, body: Box::new(Yaml::Map(merged)) }
            }
            Yaml::Map(mut m) => {
                m.extend(extra);
                Yaml::Map(m)
            }
            Yaml::Null => Yaml::Map(extra),
            other => other,
        })
    }
}

/// 把 `key: value` 拆开（值里含 `:` 时只认第一个，且 `!<Tag>` 内的 `:` 不拆）。
fn split_key_value(text: &str, line: usize) -> OcioResult<(String, String)> {
    let mut depth: i32 = 0;
    let mut quote: Option<char> = None;
    let mut colon: Option<usize> = None;
    for (i, c) in text.char_indices() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                ':' if depth == 0 => {
                    colon = Some(i);
                    break;
                }
                _ => {}
            },
        }
    }
    let at: usize = colon.ok_or_else(|| {
        OcioError::syntax(line, format!("映射行 `{text}` 缺 `:`（期望 `key: value`）"))
    })?;
    let key: String = text[..at].trim().to_string();
    if key.is_empty() {
        return Err(OcioError::syntax(line, "映射键为空"));
    }
    Ok((key, text[at + 1..].trim().to_string()))
}

/// 解析 `config.ocio` 文本为 [`Yaml`] 树（公开供 golden 逐字段对拍）。
pub fn parse_yaml(text: &str) -> OcioResult<Yaml> {
    let lines: Vec<Line> = lexical_lines(text)?;
    let Some(first) = lines.first() else { return Ok(Yaml::Null) };
    let indent: usize = first.indent;
    let mut p = BlockParser::new(lines);
    let tree: Yaml = p.parse_block(indent)?;
    if p.pos != p.lines.len() {
        let l: &Line = &p.lines[p.pos];
        return Err(OcioError::syntax(
            l.number,
            format!("第 {} 行起有无法归层的残留内容：`{}`", l.number, l.text),
        ));
    }
    Ok(tree)
}

// ===========================================================================
//  配置数据模型
// ===========================================================================

/// 一条 transform（六类全覆盖）。
#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    /// `!<FileTransform>`：外部 LUT（`.cube` / `.spi1d`）。
    File {
        /// LUT 相对路径。
        src: String,
        /// 插值方式（`linear` / `nearest` / `tetrahedral`）。
        interpolation: String,
        /// 方向（`forward` / `inverse`）。
        direction: String,
    },
    /// `!<MatrixTransform>`：4×4 行主序 + 4 分量 offset。
    Matrix { matrix: [f64; 16], offset: [f64; 4] },
    /// `!<ExponentTransform>`：4 分量 gamma。
    Exponent { value: [f64; 4] },
    /// `!<LogTransform>`：以 base 为底的 log（OCIO 的 log-to-linear 侧由 direction 决定）。
    Log { base: f64, direction: String },
    /// `!<RangeTransform>`：输入/输出区间线性映射。
    Range {
        min_in: f64,
        max_in: f64,
        min_out: f64,
        max_out: f64,
    },
    /// `!<GroupTransform>`：`children` 递归。
    Group(Vec<Transform>),
}

/// 色彩空间定义。
#[derive(Debug, Clone, PartialEq)]
pub struct ColorSpaceDef {
    /// 名字（`roles` / `displays` 引用它）。
    pub name: String,
    /// 族。
    pub family: String,
    /// 位深（`32f` / `16f` / `8ui` …）。
    pub bitdepth: String,
    /// allocation（`uniform` / `lg2`）。
    pub allocation: String,
    /// `isdata`。
    pub isdata: bool,
    /// color space → scene reference 的链。
    pub to_reference: Vec<Transform>,
    /// scene reference → color space 的链。
    pub from_reference: Vec<Transform>,
}

/// 一个 view（`displays` 的 `- !<View>`，或顶层 `views` 的条目）。
#[derive(Debug, Clone, PartialEq)]
pub struct ViewDef {
    /// view 名。
    pub name: String,
    /// 目标色彩空间名。
    pub colorspace: String,
    /// 该 view 挂的 look 名（升序，源顺序保留）。
    pub looks: Vec<String>,
}

/// 一个 display。
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayDef {
    /// display 名。
    pub name: String,
    /// 其下所有 views（源顺序）。
    pub views: Vec<ViewDef>,
}

/// 一个 look。
#[derive(Debug, Clone, PartialEq)]
pub struct LookDef {
    /// look 名。
    pub name: String,
    /// 处理色彩空间。
    pub process_space: String,
    /// look 的 transform 链。
    pub transform: Vec<Transform>,
}

/// 完整 `config.ocio` 解析结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OcioConfig {
    /// `ocio_profile_version`（缺省 `"1"`）。
    pub version: String,
    /// `search_path`（原样保留）。
    pub search_path: String,
    /// `strictparsing`。
    pub strictparsing: bool,
    /// `roles`（源顺序）。
    pub roles: Vec<(String, String)>,
    /// `colorspaces`（源顺序）。
    pub colorspaces: Vec<ColorSpaceDef>,
    /// `displays`（源顺序）。
    pub displays: Vec<DisplayDef>,
    /// 独立 `views` 段（源顺序）。
    pub views: Vec<ViewDef>,
    /// `looks`（源顺序）。
    pub looks: Vec<LookDef>,
    /// 已识别但不参与装配的顶层键（**显式记账，不静默吞**）。
    pub ignored_keys: Vec<String>,
}

impl OcioConfig {
    /// 空配置（没有任何色彩空间 / display / look）：产品路径按「无 config」处理。
    pub fn is_empty(&self) -> bool {
        self.colorspaces.is_empty() && self.displays.is_empty() && self.looks.is_empty()
    }

    /// 解析 `config.ocio` 文本。
    pub fn parse(text: &str) -> OcioResult<Self> {
        let root: Yaml = parse_yaml(text)?;
        let map: &[(String, Yaml)] = root.as_map().ok_or_else(|| {
            OcioError::structure(1, "config.ocio 顶层应是 `key: value` 映射")
        })?;
        let mut cfg = OcioConfig::default();
        for (key, val) in map {
            match key.as_str() {
                "ocio_profile_version" => {
                    cfg.version = scalar_or(val, "1", key)?;
                }
                "search_path" => {
                    cfg.search_path = scalar_or(val, "", key)?;
                }
                "strictparsing" => {
                    cfg.strictparsing = matches!(
                        scalar_or(val, "false", key)?.to_ascii_lowercase().as_str(),
                        "true" | "1" | "yes"
                    );
                }
                "roles" => {
                    cfg.roles = parse_roles(val)?;
                }
                "colorspaces" => {
                    cfg.colorspaces = parse_colorspaces(val)?;
                }
                "displays" => {
                    cfg.displays = parse_displays(val)?;
                }
                "views" => {
                    cfg.views = parse_views(val)?;
                }
                "looks" => {
                    cfg.looks = parse_looks(val)?;
                }
                other => {
                    // 顶层其它键（description / environment / luma / active_views …）
                    // 显式记账；strictparsing=true 时报错（不静默吞）。
                    cfg.ignored_keys.push(other.to_string());
                    if cfg.strictparsing {
                        return Err(OcioError::structure(
                            1,
                            format!("strictparsing=true 时不允许未注册的顶层键 `{other}`"),
                        ));
                    }
                }
            }
        }
        // strictparsing 可能在 roles 之后才出现：二次校验，保证顺序无关。
        if cfg.strictparsing && !cfg.ignored_keys.is_empty() {
            return Err(OcioError::structure(
                1,
                format!(
                    "strictparsing=true 时不允许未注册的顶层键：{:?}",
                    cfg.ignored_keys
                ),
            ));
        }
        Ok(cfg)
    }

    /// 从磁盘读 `config.ocio`（产品路径入口）。
    pub fn from_path(path: &str) -> OcioResult<Self> {
        let text: String = std::fs::read_to_string(path).map_err(|e| {
            OcioError::new(
                "OC1008",
                format!("ocio::from_path({path})"),
                format!("读取 config.ocio 失败：{e}"),
                "确认 --ocio 指向的路径存在且为 UTF-8 文本",
            )
        })?;
        Self::parse(&text)
    }

    /// 按名字查色彩空间。
    pub fn colorspace(&self, name: &str) -> Option<&ColorSpaceDef> {
        self.colorspaces.iter().find(|c| c.name == name)
    }

    /// 解析 role → 色彩空间名。
    pub fn role(&self, name: &str) -> Option<&str> {
        self.roles
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// 选 display / view：未指定时取第一个（确定性，源顺序）。
    pub fn view(&self, display: Option<&str>, view: Option<&str>) -> OcioResult<(DisplayDef, ViewDef)> {
        let d: &DisplayDef = match display {
            Some(name) => self.displays.iter().find(|d| d.name == name).ok_or_else(|| {
                OcioError::structure(1, format!("display `{name}` 不存在于 config.ocio"))
            })?,
            None => self.displays.first().ok_or_else(|| {
                OcioError::structure(1, "config.ocio 没有 displays 段，无法选 view")
            })?,
        };
        let v: &ViewDef = match view {
            Some(name) => d.views.iter().find(|v| v.name == name).ok_or_else(|| {
                OcioError::structure(1, format!("display `{}` 下没有 view `{name}`", d.name))
            })?,
            None => d.views.first().ok_or_else(|| {
                OcioError::structure(1, format!("display `{}` 下没有 views", d.name))
            })?,
        };
        Ok((d.clone(), v.clone()))
    }

    /// 装配 view transform 链（look 链 + 色彩空间链）。
    ///
    /// 口径（自研等价管线，写死并落档）：
    /// ① 选中 display/view 指定的 look（`v.looks` 优先，其次 `look` 参数）；
    /// ② look 的 transform 链在前；
    /// ③ 视图色彩空间优先用 `from_scene_reference`（reference → view），
    ///    没给则按声明序取 `to_scene_reference`；
    /// ④ 链为空 ⇒ 返回空链（产品路径按「无 config」逐字节走 ACES）。
    pub fn view_chain(
        &self,
        display: Option<&str>,
        view: Option<&str>,
        look: Option<&str>,
    ) -> OcioResult<Vec<Transform>> {
        let (_d, v): (DisplayDef, ViewDef) = self.view(display, view)?;
        let mut chain: Vec<Transform> = Vec::new();
        let mut wanted: Vec<String> = v.looks.clone();
        if let Some(l) = look {
            wanted.push(l.to_string());
        }
        for name in wanted {
            let def: &LookDef = self.looks.iter().find(|l| l.name == name).ok_or_else(|| {
                OcioError::structure(1, format!("view `{}` 引用的 look `{name}` 未定义", v.name))
            })?;
            chain.extend(def.transform.iter().cloned());
        }
        let cs: &ColorSpaceDef = self.colorspace(&v.colorspace).ok_or_else(|| {
            OcioError::structure(
                1,
                format!("view `{}` 引用了未定义的色彩空间 `{}`", v.name, v.colorspace),
            )
        })?;
        if !cs.from_reference.is_empty() {
            chain.extend(cs.from_reference.iter().cloned());
        } else {
            chain.extend(cs.to_reference.iter().cloned());
        }
        Ok(chain)
    }

    /// 展平 GroupTransform（确定性，前序）。
    pub fn flatten_chain(chain: &[Transform]) -> Vec<Transform> {
        let mut out: Vec<Transform> = Vec::new();
        for t in chain {
            match t {
                Transform::Group(children) => out.extend(Self::flatten_chain(children)),
                other => out.push(other.clone()),
            }
        }
        out
    }
}

fn scalar_or(val: &Yaml, default: &str, key: &str) -> OcioResult<String> {
    match val {
        Yaml::Scalar(s) => Ok(s.clone()),
        Yaml::Null => Ok(default.to_string()),
        _ => Err(OcioError::structure(
            1,
            format!("顶层键 `{key}` 应是标量，实际是 {val:?}"),
        )),
    }
}

fn parse_roles(val: &Yaml) -> OcioResult<Vec<(String, String)>> {
    let m: &[(String, Yaml)] = val.as_map().ok_or_else(|| {
        OcioError::structure(1, "`roles` 应是 `role: colorspace` 映射")
    })?;
    let mut out: Vec<(String, String)> = Vec::new();
    for (k, v) in m {
        let name: String = v.as_scalar().ok_or_else(|| {
            OcioError::structure(1, format!("role `{k}` 的值应是色彩空间名字符串"))
        })?.to_string();
        out.push((k.clone(), name));
    }
    Ok(out)
}

fn parse_colorspaces(val: &Yaml) -> OcioResult<Vec<ColorSpaceDef>> {
    let items: &[Yaml] = val.as_seq().ok_or_else(|| {
        OcioError::structure(1, "`colorspaces` 应是 `- !<ColorSpace>` 序列")
    })?;
    let mut out: Vec<ColorSpaceDef> = Vec::new();
    for item in items {
        let tag: &str = match item {
            Yaml::Tagged { tag, .. } => tag.as_str(),
            _ => "",
        };
        if !tag.is_empty() && tag != "ColorSpace" {
            return Err(OcioError::structure(
                1,
                format!("colorspaces 序列里出现 `!<{tag}>`，只允许 `!<ColorSpace>`"),
            ));
        }
        let get = |k: &str| item.get(k);
        let name: String = item.get_str("name").unwrap_or("").to_string();
        if name.is_empty() {
            return Err(OcioError::structure(1, "ColorSpace 缺 `name`"));
        }
        let to_reference: Vec<Transform> = match get("to_scene_reference") {
            Some(v) => parse_transform_node(v)?,
            None => Vec::new(),
        };
        let from_reference: Vec<Transform> = match get("from_scene_reference") {
            Some(v) => parse_transform_node(v)?,
            None => Vec::new(),
        };
        out.push(ColorSpaceDef {
            name,
            family: item.get_str("family").unwrap_or("").to_string(),
            bitdepth: item.get_str("bitdepth").unwrap_or("").to_string(),
            allocation: item.get_str("allocation").unwrap_or("").to_string(),
            isdata: matches!(
                item.get_str("isdata").unwrap_or("false").to_ascii_lowercase().as_str(),
                "true" | "1" | "yes"
            ),
            to_reference,
            from_reference,
        });
    }
    Ok(out)
}

fn parse_displays(val: &Yaml) -> OcioResult<Vec<DisplayDef>> {
    let m: &[(String, Yaml)] = val.as_map().ok_or_else(|| {
        OcioError::structure(1, "`displays` 应是 `displayName:` → views 序列的映射")
    })?;
    let mut out: Vec<DisplayDef> = Vec::new();
    for (name, v) in m {
        let views: Vec<ViewDef> = parse_views(v)?;
        out.push(DisplayDef { name: name.clone(), views });
    }
    Ok(out)
}

fn parse_views(val: &Yaml) -> OcioResult<Vec<ViewDef>> {
    let items: &[Yaml] = val.as_seq().ok_or_else(|| {
        OcioError::structure(1, "`views`（或 display 下的 views）应是 `- !<View>` 序列")
    })?;
    let mut out: Vec<ViewDef> = Vec::new();
    for item in items {
        if let Yaml::Tagged { tag, .. } = item {
            if tag != "View" {
                return Err(OcioError::structure(
                    1,
                    format!("views 序列里出现 `!<{tag}>`，只允许 `!<View>`"),
                ));
            }
        }
        let name: String = item.get_str("name").unwrap_or("").to_string();
        let colorspace: String = item
            .get_str("colorspace")
            .or_else(|| item.get_str("colorspaceName"))
            .unwrap_or("")
            .to_string();
        if name.is_empty() || colorspace.is_empty() {
            return Err(OcioError::structure(
                1,
                "View 必须同时给 `name` 与 `colorspace`",
            ));
        }
        let looks: Vec<String> = match item.get("looks") {
            Some(Yaml::Scalar(s)) => s
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect(),
            Some(Yaml::Seq(v)) => v.iter().filter_map(Yaml::as_scalar).map(str::to_string).collect(),
            _ => Vec::new(),
        };
        out.push(ViewDef { name, colorspace, looks });
    }
    Ok(out)
}

fn parse_looks(val: &Yaml) -> OcioResult<Vec<LookDef>> {
    let items: &[Yaml] = val.as_seq().ok_or_else(|| {
        OcioError::structure(1, "`looks` 应是 `- !<Look>` 序列")
    })?;
    let mut out: Vec<LookDef> = Vec::new();
    for item in items {
        if let Yaml::Tagged { tag, .. } = item {
            if tag != "Look" {
                return Err(OcioError::structure(
                    1,
                    format!("looks 序列里出现 `!<{tag}>`，只允许 `!<Look>`"),
                ));
            }
        }
        let name: String = item.get_str("name").unwrap_or("").to_string();
        if name.is_empty() {
            return Err(OcioError::structure(1, "Look 缺 `name`"));
        }
        let transform: Vec<Transform> = match item.get("transform") {
            Some(v) => parse_transform_node(v)?,
            None => Vec::new(),
        };
        out.push(LookDef {
            name,
            process_space: item.get_str("process_space").unwrap_or("").to_string(),
            transform,
        });
    }
    Ok(out)
}

/// 把一个 transform 节点（可能带 `!<Tag>` 或 `!<GroupTransform>{children:[...]}`）
/// 解析成链（GroupTransform 的 children 保持为一个 Group 项，便于 golden 逐字段对拍）。
fn parse_transform_node(node: &Yaml) -> OcioResult<Vec<Transform>> {
    match node {
        Yaml::Tagged { tag, body } => Ok(vec![parse_transform_tagged(tag, body)?]),
        Yaml::Map(m) => {
            // `children:` 形式（无显式 tag）也按 Group 收。
            if m.iter().any(|(k, _)| k == "children") {
                let children: Vec<Transform> = parse_children(node)?;
                Ok(vec![Transform::Group(children)])
            } else {
                Err(OcioError::structure(
                    1,
                    "transform 节点缺 `!<Tag>` 标注（无法判定类型，拒绝静默跳过）",
                ))
            }
        }
        Yaml::Seq(items) => {
            let mut out: Vec<Transform> = Vec::new();
            for it in items {
                out.extend(parse_transform_node(it)?);
            }
            Ok(out)
        }
        Yaml::Null => Ok(Vec::new()),
        other => Err(OcioError::structure(
            1,
            format!("transform 节点形态不合法：{other:?}"),
        )),
    }
}

fn parse_children(node: &Yaml) -> OcioResult<Vec<Transform>> {
    let seq: &[Yaml] = node
        .get("children")
        .and_then(Yaml::as_seq)
        .ok_or_else(|| OcioError::structure(1, "GroupTransform 缺 `children` 序列"))?;
    let mut out: Vec<Transform> = Vec::new();
    for it in seq {
        out.extend(parse_transform_node(it)?);
    }
    Ok(out)
}

fn numbers(val: Option<&Yaml>, line: usize, name: &str) -> OcioResult<Vec<f64>> {
    let Some(v) = val else { return Ok(Vec::new()) };
    let seq: &[Yaml] = v.as_seq().ok_or_else(|| {
        OcioError::number(line, format!("`{name}` 应是 `[...]` 数值数组"))
    })?;
    let mut out: Vec<f64> = Vec::with_capacity(seq.len());
    for s in seq {
        let raw: &str = s.as_scalar().ok_or_else(|| {
            OcioError::number(line, format!("`{name}` 数组含非标量元素 {s:?}"))
        })?;
        let n: f64 = raw.parse::<f64>().map_err(|_| {
            OcioError::number(line, format!("`{name}` 里的 `{raw}` 不是合法有限十进制数"))
        })?;
        if !n.is_finite() {
            return Err(OcioError::number(line, format!("`{name}` 里的 `{raw}` 非有限数")));
        }
        out.push(n);
    }
    Ok(out)
}

fn finite_or(val: Option<&Yaml>, line: usize, name: &str, default: f64) -> OcioResult<f64> {
    match val.and_then(Yaml::as_scalar) {
        None => Ok(default),
        Some(raw) => {
            let n: f64 = raw.parse::<f64>().map_err(|_| {
                OcioError::number(line, format!("`{name}` 的 `{raw}` 不是合法有限十进制数"))
            })?;
            if !n.is_finite() {
                return Err(OcioError::number(line, format!("`{name}` 的 `{raw}` 非有限数")));
            }
            Ok(n)
        }
    }
}

fn parse_transform_tagged(tag: &str, body: &Yaml) -> OcioResult<Transform> {
    let line: usize = 1; // 行内节点无法回溯精确行号；错误内容仍三段式可定位。
    match tag {
        "FileTransform" => {
            let src: String = body.get_str("src").unwrap_or("").to_string();
            if src.is_empty() {
                return Err(OcioError::lut_source(line, "FileTransform 缺 `src`"));
            }
            if !(src.ends_with(".cube") || src.ends_with(".spi1d")) {
                return Err(OcioError::lut_source(
                    line,
                    format!("FileTransform 的 `src` 后缀不受支持：`{src}`"),
                ));
            }
            Ok(Transform::File {
                src,
                interpolation: body.get_str("interpolation").unwrap_or("linear").to_string(),
                direction: body.get_str("direction").unwrap_or("forward").to_string(),
            })
        }
        "MatrixTransform" => {
            let m: Vec<f64> = numbers(body.get("matrix"), line, "matrix")?;
            if m.len() != 16 {
                return Err(OcioError::matrix(
                    line,
                    format!("matrix 应有 16 个元素，实际 {} 个", m.len()),
                ));
            }
            let mut matrix: [f64; 16] = [0.0; 16];
            matrix.copy_from_slice(&m);
            let off: Vec<f64> = numbers(body.get("offset"), line, "offset")?;
            if !off.is_empty() && off.len() != 4 {
                return Err(OcioError::matrix(
                    line,
                    format!("offset 应有 4 个元素，实际 {} 个", off.len()),
                ));
            }
            let mut offset: [f64; 4] = [0.0; 4];
            if off.len() == 4 {
                offset.copy_from_slice(&off);
            }
            Ok(Transform::Matrix { matrix, offset })
        }
        "ExponentTransform" => {
            let v: Vec<f64> = numbers(body.get("value"), line, "value")?;
            if v.len() != 4 {
                return Err(OcioError::number(
                    line,
                    format!("ExponentTransform 的 `value` 应有 4 个元素，实际 {} 个", v.len()),
                ));
            }
            let mut value: [f64; 4] = [1.0; 4];
            value.copy_from_slice(&v);
            Ok(Transform::Exponent { value })
        }
        "LogTransform" => {
            let base: f64 = finite_or(body.get("base"), line, "base", 10.0)?;
            if base <= 0.0 || (base - 1.0).abs() < 1e-12 {
                return Err(OcioError::number(
                    line,
                    format!("LogTransform 的 base={base} 不合法（须 >0 且 ≠1）"),
                ));
            }
            Ok(Transform::Log {
                base,
                direction: body.get_str("direction").unwrap_or("forward").to_string(),
            })
        }
        "RangeTransform" => {
            let min_in: f64 = finite_or(body.get("min_in_value"), line, "min_in_value", 0.0)?;
            let max_in: f64 = finite_or(body.get("max_in_value"), line, "max_in_value", 1.0)?;
            let min_out: f64 = finite_or(body.get("min_out_value"), line, "min_out_value", 0.0)?;
            let max_out: f64 = finite_or(body.get("max_out_value"), line, "max_out_value", 1.0)?;
            if min_in >= max_in {
                return Err(OcioError::number(
                    line,
                    format!("RangeTransform 的 min_in_value={min_in} ≥ max_in_value={max_in}"),
                ));
            }
            Ok(Transform::Range { min_in, max_in, min_out, max_out })
        }
        "GroupTransform" => Ok(Transform::Group(parse_children(body)?)),
        other => Err(OcioError::unknown_transform(line, other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_flow_and_block_roundtrip() {
        let y = parse_yaml("a: {b: [1, 2, 3], c: \"hi\"}\nd:\n  - !<View> {name: Raw}\n")
            .unwrap();
        assert_eq!(y.get("a").unwrap().get_str("c"), Some("hi"));
        assert_eq!(y.get("a").unwrap().get("b").unwrap().as_seq().unwrap().len(), 3);
        assert_eq!(y.get("d").unwrap().as_seq().unwrap().len(), 1);
        // 二次解析同一文本逐字段一致（确定性）。
        let y2 = parse_yaml("a: {b: [1, 2, 3], c: \"hi\"}\nd:\n  - !<View> {name: Raw}\n")
            .unwrap();
        assert_eq!(y, y2);
    }

    #[test]
    fn unknown_top_level_key_is_recorded_not_swallowed() {
        let cfg = OcioConfig::parse("description: hello\nroles:\n  default: raw\n").unwrap();
        assert_eq!(cfg.ignored_keys, vec!["description".to_string()]);
        assert_eq!(cfg.role("default"), Some("raw"));
    }

    #[test]
    fn bad_matrix_is_structured_error() {
        let e = OcioConfig::parse(
            "colorspaces:\n  - !<ColorSpace>\n    name: raw\n    to_scene_reference: !<MatrixTransform> {matrix: [1, 2, 3]}\n",
        )
        .unwrap_err();
        assert_eq!(e.code, "OC1005");
        assert!(e.problem.contains("16"));
    }
}
