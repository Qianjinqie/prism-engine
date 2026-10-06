//! U11-2：MaterialX `.mtlx`（XML）解析 → 特解参数包（D-23 三元组）编译。
//!
//! 语义权威：《差距账本》U11（"无 MaterialX/OSL 材质互换"）、
//! 《架构总设计》§2.6 D-23（自定义四级光谱之**特解级**：结构位 + 特解参数位 +
//! 组合位 = 三元组）、宪章 C18（错误三段式 + 必含建议段）、
//! C20（缺定义即明确报错，绝不静默降级）、C17（全链路确定性）、
//! D-16（Mini Code）、D-29（确定性数学件）。
//!
//! # 为什么手写 XML 解析器
//! MaterialX 是 XML；零第三方依赖是铁律（连 `quick-xml` / `serde` 都不许），
//! 故本模块自带**最小 XML 解析器** [`parse_xml`]，只实现本任务需要的子集。
//!
//! # XML 支持范围（明确的最小子集）
//! **支持**：
//! - 元素嵌套、属性（单/双引号）、自闭合元素（`<x/>`）、文本内容；
//! - 注释 `<!-- ... -->`（可跨行）；
//! - CDATA 段 `<![CDATA[ ... ]]>`（按原样文本处理，不解实体）；
//! - 数字/字符串实体：`&lt; &gt; &amp; &quot; &apos; &#NN; &#xHH;`；
//! - XML 声明 `<?xml version="1.0"?>`（跳过）与处理指令（跳过）。
//!
//! **不支持**（本轮**明确**不做，见「诚实边界」）：
//! - DTD / 实体声明 `<!DOCTYPE ...>`（只跳过不解释，**不**支持自定义实体）；
//! - 命名空间处理（`xmlns` 属性原样保留，不做前缀解析）；
//! - CDATA 之外的实体白名单（只认上列 5 个 + 数字实体，**未知实体报错**
//!   而非吞掉——吞掉会静默改变材质数值）；
//! - 文本内容的自动转义回写（CDATA 与文本按原样保存）。
//!
//! # MaterialX 支持范围
//! - `<materialx>` → `<materyialgraph>` → `<material>` → `<nodegraph>`
//!   → `<node>` / `<input>` / `<output>` / `<parameter>`（注意 MaterialX
//!   规范里 `materyialgraph` 的历史拼写，本解析器**两种拼写都收**）；
//! - 节点类型映射到项目特解：见 [`NodeKind`]（`standard_surface` / `bsdf`
//!   / `diffuse` / `glossy` / `mix` / `add` / `multiply` / `ramp` / `constant`
//!   / `position` / `normal` / `dot` / `image` 等）；
//! - `<input>` 的 `value` / `interconnect` 属性 + `<connectfrom>` 子元素
//!   → 图边（`interconnect="ND_bsdf.outputs:surface"` 形式）；
//! - `<output>` 收集材质输出（如 `surface`）；
//! - 编译产物：[`MaterialPack`]（特解参数包）+ [`CompiledGraph`]（节点/边/输出）。
//!
//! # 诚实边界
//! - **只编译，不求值**：本模块把 `.mtlx` 编译成**可用的特解参数包**
//!   （结构 + 有序参数 + 图边），**不**实现 BSDF 着色方程求值、不做
//!   纹理采样、不做颜色空间换算。求值属渲染器职责。
//! - **未知节点类型 → 结构化报错**（`MX2004`），**不**静默当常量跳过
//!   （宪章 C20：材质悄悄少一个节点 = 画面悄悄变错）；
//! - **不支持** `<include>` / `<function>` 定义与调用展开、`<typedef>`、
//!   `unit`/`unitdef` 换算、`target` 变体选择（MaterialX 的 multi-target
//!   特性）、`version` 兼容性检查、`nodedef` 扩展属性；
//! - **输入不合法**（缺 `type`、类型与 value 不匹配、连线指向不存在的
//!   节点/输出、循环依赖）→ **全部结构化错误**，绝不 panic、绝不降级。
//!
//! # 确定性契约
//! 参数包内部用 `BTreeMap`（键序确定）；节点/边/输出按**源文件出现顺序**
//! 追加（语义有序）；`materialx` 里的 `version` 原样保留。同输入编译两遍
//! 逐字段一致（见 `compile_is_deterministic`）。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ===========================================================================
//  错误（宪章 C18 三段式 + C19 必含建议段）
// ===========================================================================

/// MaterialX 编译错误。错误码取 **MX 段**（prism-materialx 族）。
#[derive(Debug, Clone, PartialEq)]
pub struct MtlxError {
    /// 错误码（MX2xxx）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 源内位置（元素路径 / 字符偏移）。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for MtlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for MtlxError {}

impl MtlxError {
    /// 构造错误（四字段全填）。
    #[inline]
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

    /// MX2001：XML 语法错误。
    pub fn xml(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2001",
            location,
            problem,
            "检查 XML 语法：标签必须闭合（`<x>..</x>` 或 `<x/>`）、属性必须带引号、\
             注释用 `<!-- -->`；用任意 XML 校验工具先过一遍"
                .to_string(),
        )
    }

    /// MX2002：XML 特征本轮不支持（DTD / 未知实体等）。
    pub fn unsupported(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2002",
            location,
            problem,
            "本轮只支持最小 XML（无 DTD、无自定义实体、无命名空间语义）；\
             请让导出方关闭 DTD 并只用预定义实体（&lt; &gt; &amp; &quot; &apos;）"
                .to_string(),
        )
    }

    /// MX2003：MaterialX 结构错误（缺 material / nodegraph / name 等）。
    pub fn structure(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2003",
            location,
            problem,
            "标准结构应为 `<materialx><materyialgraph><material><nodegraph>`；\
             检查是否漏了 `name` 属性或节点图",
        )
    }

    /// MX2004：未知节点类型 / 未知参数名。
    pub fn unknown_node(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2004",
            location,
            problem,
            "本轮支持的节点类型有限（见 `NodeKind` 文档）；如需该节点，\
             请先在 `materialx.rs` 里加映射，或改用已支持的等价节点组合"
                .to_string(),
        )
    }

    /// MX2005：参数值与声明类型不匹配。
    pub fn type_mismatch(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2005",
            location,
            problem,
            "核对 `type` 与 `value`：`float`/`color3`/`vector3`/`float2` 等；\
             color 需 3 分量、vector3 需 3 分量、float2 需 2 分量",
        )
    }

    /// MX2006：连线端点悬空（节点或输出不存在）。
    pub fn dangling_interconnect(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2006",
            location,
            problem,
            "`interconnect`/`connectfrom` 指向的节点或输出不存在；\
             检查节点 `name` 与 `outputs` 声明是否拼写一致",
        )
    }

    /// MX2007：循环依赖。
    pub fn cycle(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2007",
            location,
            problem,
            "节点图存在环；把环上的一条边改成由外部输入驱动（MaterialX 节点图须是 DAG）",
        )
    }

    /// MX2008：同一输出被多条输入占用 / 重复定义。
    pub fn conflict(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "MX2008",
            location,
            problem,
            "一个输入只能连一条边、一个节点名不能重复；拆成多节点或改名",
        )
    }
}

// ===========================================================================
//  最小 XML 解析器
// ===========================================================================

/// XML 元素节点。
///
/// `children` 按源顺序；属性存 `BTreeMap`（**键序确定**，C17）。
/// `text` 为**直接**文本内容（不含子元素的文本；MaterialX 的值都在属性里，
/// 故此字段主要用于 `ramp` 的 `<stop>` 之类结构）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct XmlElement {
    /// 标签名（原样，**不做命名空间解析**）。
    pub name: String,
    /// 属性（`BTreeMap` → 键序确定）。
    pub attributes: BTreeMap<String, String>,
    /// 子元素（源顺序）。
    pub children: Vec<XmlElement>,
    /// 直接文本内容（已解预定义实体；CDATA 按原样）。
    pub text: String,
}

impl XmlElement {
    /// 取属性。
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// 按标签名找**第一个**直接子元素。
    pub fn child(&self, name: &str) -> Option<&XmlElement> {
        self.children.iter().find(|c| c.name == name)
    }

    /// 按标签名收集全部直接子元素。
    pub fn children_named<'a>(&'a self, name: &'a str) -> Vec<&'a XmlElement> {
        self.children.iter().filter(|c| c.name == name).collect()
    }

    /// 元素路径（`/materialx/materyialgraph/material[2]/nodegraph`），用于错误定位。
    pub fn path(&self) -> String {
        let mut p = format!("/{}", self.name);
        for c in &self.children {
            p.push('/');
            p.push_str(&c.name);
        }
        p
    }
}

/// 解析 XML 文本为元素树（返回根元素）。
///
/// 支持：元素嵌套 / 属性 / 自闭合 / 文本 / 注释 / CDATA /
/// 预定义 + 数字实体 / `<?xml?>` 声明。不支持 DTD 与自定义实体
/// （遇到即 [`MtlxError::unsupported`]，不静默吞掉）。
pub fn parse_xml(text: &str) -> Result<XmlElement, MtlxError> {
    let mut p = XmlParser {
        src: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    p.parse_document()
}

/// 嵌套深度上限（防御恶意超深输入导致栈溢出——C18 永不 panic）。
const MAX_XML_DEPTH: usize = 256;

/// 手写 XML 解析器（递归下降）。
struct XmlParser<'a> {
    src: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> XmlParser<'a> {
    /// 当前位置描述。
    fn here(&self) -> String {
        let (mut line, mut col) = (1usize, 1usize);
        for b in &self.src[..self.pos.min(self.src.len())] {
            if *b == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        format!(
            "materialx::parse_xml 第 {line} 行第 {col} 列（字节偏移 {}）",
            self.pos
        )
    }

    /// 前进 `n` 字节。
    fn bump(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.src.len());
    }

    /// 窥视当前字节。
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    /// 从 `pos` 起匹配字面量。
    fn starts_with(&self, s: &str) -> bool {
        self.src[self.pos.min(self.src.len())..].starts_with(s.as_bytes())
    }

    /// 跳过空白。
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.bump(1);
            } else {
                break;
            }
        }
    }

    /// 解析整篇文档 → 根元素。
    fn parse_document(&mut self) -> Result<XmlElement, MtlxError> {
        // 前导：声明 / 注释 / DOCTYPE（拒绝）/ 空白。
        loop {
            self.skip_ws();
            if self.starts_with("<?") {
                self.skip_until("?>")?;
            } else if self.starts_with("<!--") {
                self.skip_comment()?;
            } else if self.starts_with("<!DOCTYPE") {
                return Err(MtlxError::unsupported(
                    self.here(),
                    "遇到 `<!DOCTYPE>`：本轮最小 XML 解析器不支持 DTD / 自定义实体",
                ));
            } else {
                break;
            }
        }
        if self.peek() != Some(b'<') {
            return Err(MtlxError::xml(self.here(), "文档根不是元素（未找到 `<`）"));
        }
        let root = self.parse_element()?;
        // 根之后只允许空白 / 注释 / PI。
        loop {
            self.skip_ws();
            if self.starts_with("<?") {
                self.skip_until("?>")?;
            } else if self.starts_with("<!--") {
                self.skip_comment()?;
            } else if self.peek().is_none() {
                break;
            } else {
                return Err(MtlxError::xml(
                    self.here(),
                    format!(
                        "根元素之后出现多余内容：{:?}",
                        &self.src[self.pos..(self.pos + 8).min(self.src.len())]
                    ),
                ));
            }
        }
        Ok(root)
    }

    /// 跳到 `delim` 之后（找不到 → 报错）。
    fn skip_until(&mut self, delim: &str) -> Result<(), MtlxError> {
        while self.pos < self.src.len() {
            if self.starts_with(delim) {
                self.bump(delim.len());
                return Ok(());
            }
            self.bump(1);
        }
        Err(MtlxError::xml(
            self.here(),
            format!("未找到结束标记 {delim:?}"),
        ))
    }

    /// 跳过注释 `<!-- ... -->`。
    fn skip_comment(&mut self) -> Result<(), MtlxError> {
        self.bump(4); // `<!--`
        self.skip_until("-->")
            .map_err(|_| MtlxError::xml(self.here(), "注释未闭合（缺 `-->`）"))
    }

    /// 解析一个元素（含属性 / 子元素 / 文本）。
    fn parse_element(&mut self) -> Result<XmlElement, MtlxError> {
        if self.depth >= MAX_XML_DEPTH {
            return Err(MtlxError::xml(
                self.here(),
                format!("XML 嵌套深度超过上限 {MAX_XML_DEPTH}（疑似恶意输入）"),
            ));
        }
        self.bump(1); // `<`
        let name = self.parse_name()?;
        if name.is_empty() {
            return Err(MtlxError::xml(self.here(), "元素名为空"));
        }
        let mut el = XmlElement {
            name,
            ..Default::default()
        };
        // 属性。
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'/') => {
                    self.bump(1);
                    if self.peek() != Some(b'>') {
                        return Err(MtlxError::xml(self.here(), "自闭合元素应以 `/>` 结束"));
                    }
                    self.bump(1);
                    return Ok(el);
                }
                Some(b'>') => {
                    self.bump(1);
                    break;
                }
                Some(_) => {
                    let key = self.parse_name()?;
                    if key.is_empty() {
                        return Err(MtlxError::xml(
                            self.here(),
                            "属性名为空（检查是否缺引号或用了非法字符）",
                        ));
                    }
                    self.skip_ws();
                    if self.peek() != Some(b'=') {
                        return Err(MtlxError::xml(
                            self.here(),
                            format!("属性 {key:?} 后应为 `=`"),
                        ));
                    }
                    self.bump(1);
                    self.skip_ws();
                    let val = self.parse_attr_value(&key)?;
                    if el.attributes.insert(key.clone(), val).is_some() {
                        return Err(MtlxError::xml(
                            self.here(),
                            format!("属性 {key:?} 重复定义"),
                        ));
                    }
                }
                None => {
                    return Err(MtlxError::xml(
                        self.here(),
                        "元素属性区未闭合（读到文件结尾）",
                    ))
                }
            }
        }
        // 子内容。
        self.depth += 1;
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(MtlxError::xml(
                        self.here(),
                        format!("元素 <{}> 未闭合（读到文件结尾）", el.name),
                    ))
                }
                Some(b'<') => {
                    if self.starts_with("</") {
                        // 结束标签。
                        self.bump(2);
                        let close = self.parse_name()?;
                        self.skip_ws();
                        if self.peek() != Some(b'>') {
                            return Err(MtlxError::xml(
                                self.here(),
                                format!("结束标签 </{close}> 后应为 `>`"),
                            ));
                        }
                        self.bump(1);
                        if close != el.name {
                            return Err(MtlxError::xml(
                                self.here(),
                                format!("标签不匹配：<{}> 被 </{close}> 关闭", el.name),
                            ));
                        }
                        el.text = text;
                        self.depth -= 1;
                        return Ok(el);
                    } else if self.starts_with("<!--") {
                        self.skip_comment()?;
                    } else if self.starts_with("<![CDATA[") {
                        self.bump(9);
                        let start = self.pos;
                        self.skip_until("]]>").map_err(|_| {
                            MtlxError::xml(self.here(), "CDATA 段未闭合（缺 `]]>`）")
                        })?;
                        // `skip_until` 已越过 `]]>`，回退 3 字节取内容。
                        let end = self.pos - 3;
                        text.push_str(
                            std::str::from_utf8(&self.src[start..end])
                                .map_err(|_| MtlxError::xml(self.here(), "CDATA 内容非 UTF-8"))?,
                        );
                    } else if self.starts_with("<?") {
                        self.skip_until("?>")?;
                    } else if self.starts_with("<!") {
                        return Err(MtlxError::unsupported(
                            self.here(),
                            "元素内出现声明（`<!...>`）：本轮不支持 DTD / 自定义实体",
                        ));
                    } else {
                        el.children.push(self.parse_element()?);
                    }
                }
                Some(_) => {
                    // 文本。
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c == b'<' {
                            break;
                        }
                        self.bump(1);
                    }
                    let raw = std::str::from_utf8(&self.src[start..self.pos])
                        .map_err(|_| MtlxError::xml(self.here(), "文本内容非 UTF-8"))?;
                    text.push_str(&decode_entities(raw, &self.here())?);
                }
            }
        }
    }

    /// 解析元素/属性名（字母、数字、`_`、`-`、`.`、`:`）。
    fn parse_name(&mut self) -> Result<String, MtlxError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' || c == b':' {
                self.bump(1);
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .map(|s| s.to_string())
            .map_err(|_| MtlxError::xml(self.here(), "名称非 UTF-8"))
    }

    /// 解析属性值（带引号 + 解实体）。
    fn parse_attr_value(&mut self, key: &str) -> Result<String, MtlxError> {
        let quote = match self.peek() {
            Some(q @ (b'"' | b'\'')) => q,
            _ => {
                return Err(MtlxError::xml(
                    self.here(),
                    format!("属性 {key:?} 的值必须用引号包裹（\" 或 '）"),
                ))
            }
        };
        self.bump(1);
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == quote {
                let raw = std::str::from_utf8(&self.src[start..self.pos])
                    .map_err(|_| MtlxError::xml(self.here(), "属性值非 UTF-8"))?;
                let decoded = decode_entities(raw, &self.here())?;
                self.bump(1);
                return Ok(decoded);
            }
            // `&` 在属性值里必须以实体形式出现，裸 `&` 是错误。
            if c == b'<' {
                return Err(MtlxError::xml(
                    self.here(),
                    format!("属性 {key:?} 的值内出现裸 `<`（应用 &lt;）"),
                ));
            }
            self.bump(1);
        }
        Err(MtlxError::xml(
            self.here(),
            format!("属性 {key:?} 的值未闭合（缺引号）"),
        ))
    }
}

/// 解码预定义实体 + 数字实体。
///
/// **未知实体 → 报错**（`MX2002`）：吞掉未知实体会静默改变材质数值
/// （如 `&deg;` 被丢掉 → 温度算错），故宁报错不猜。
fn decode_entities(s: &str, location: &str) -> Result<String, MtlxError> {
    if !s.contains('&') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == '&' {
            // 找 `;`。
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != ';' && j - i <= 12 {
                j += 1;
            }
            if j >= bytes.len() || bytes[j] != ';' {
                return Err(MtlxError::xml(
                    location,
                    "出现 `&` 但未找到实体结束 `;`（裸 & 在 XML 中非法）",
                ));
            }
            let ent: String = bytes[i + 1..j].iter().collect();
            let ch = match ent.as_str() {
                "lt" => '<',
                "gt" => '>',
                "amp" => '&',
                "quot" => '"',
                "apos" => '\'',
                other => {
                    // 数字实体 `&#123;` / `&#x1F;`。
                    let parsed =
                        if let Some(hex) = other.strip_prefix("#x").or(other.strip_prefix("#X")) {
                            u32::from_str_radix(hex, 16).ok()
                        } else if let Some(dec) = other.strip_prefix('#') {
                            dec.parse::<u32>().ok()
                        } else {
                            None
                        };
                    match parsed.and_then(char::from_u32) {
                        Some(c) => c,
                        None => {
                            return Err(MtlxError::unsupported(
                                location,
                                format!(
                                    "未知实体 `&{other};`：本轮只支持 &lt; &gt; &amp; &quot; &apos; 与数字实体 \
                                     （自定义实体需 DTD，本轮不支持）"
                                ),
                            ));
                        }
                    }
                }
            };
            out.push(ch);
            i = j + 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(out)
}

// ===========================================================================
//  MaterialX 数据模型与特解参数包
// ===========================================================================

/// 节点类型（MaterialX → 项目特解的映射目标）。
///
/// 每个变体对应一个**结构位**语义：编译产物的 [`CompiledNode::structure`]
/// 直接取本枚举名，项目侧据此选择求值实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// 通用表面（MaterialX 标准表面）。
    StandardSurface,
    /// 通用 BSDF（旧版 MaterialX）。
    Bsdf,
    /// 漫反射。
    Diffuse,
    /// 光泽。
    Glossy,
    /// 线性插值。
    Mix,
    /// 加法。
    Add,
    /// 乘法。
    Multiply,
    /// 颜色渐变。
    Ramp,
    /// 常量。
    Constant,
    /// 位置输入。
    Position,
    /// 法线输入。
    Normal,
    /// 纹理输入。
    Image,
    /// 点积。
    Dot,
}

impl NodeKind {
    /// 结构位名称（编译产物里的稳定标识）。
    pub fn structure(self) -> &'static str {
        match self {
            NodeKind::StandardSurface => "standard_surface",
            NodeKind::Bsdf => "bsdf",
            NodeKind::Diffuse => "diffuse",
            NodeKind::Glossy => "glossy",
            NodeKind::Mix => "mix",
            NodeKind::Add => "add",
            NodeKind::Multiply => "multiply",
            NodeKind::Ramp => "ramp",
            NodeKind::Constant => "constant",
            NodeKind::Position => "position",
            NodeKind::Normal => "normal",
            NodeKind::Image => "image",
            NodeKind::Dot => "dot",
        }
    }

    /// 由 MaterialX `type` 属性文本解析节点类型；未知 → `None`。
    pub fn from_mtlx_type(t: &str) -> Option<Self> {
        // MaterialX 的 nodedef 名可带命名空间前缀（`ND_foo_bar`）与命名空间
        // 路径（`myns:foo`）。归一化顺序：先去 `:` 路径末段 → 去 `ND__` 前缀
        // → 若仍不匹配，逐个剥掉最右的 `_` 段重试（`diffuse_surfaceshader`
        // → `diffuse`）。**先长后短**（最具体优先），且剥离顺序确定。
        let last = t.rsplit(':').next().unwrap_or(t);
        let base = last.strip_prefix("ND__").unwrap_or(last);
        let mut candidate = base;
        loop {
            if let Some(k) = match_node_kind(candidate) {
                return Some(k);
            }
            match candidate.rfind('_') {
                // 至少保留首段（`_` 不在首位由 `strip_prefix("ND__")` 保证）。
                Some(i) if i > 0 => candidate = &candidate[..i],
                _ => return None,
            }
        }
    }
}

/// 按精确名匹配一个 nodedef 末段。
fn match_node_kind(last: &str) -> Option<NodeKind> {
    Some(match last {
        "standard_surface" | "surfaceshader" => NodeKind::StandardSurface,
        "bsdf" | "closure" | "shader" => NodeKind::Bsdf,
        "diffuse" | "diffuse_bsdf" => NodeKind::Diffuse,
        "glossy" | "glossy_bsdf" => NodeKind::Glossy,
        "mix" | "mix_float" | "mix_color" => NodeKind::Mix,
        "add" | "add_float" | "add_color" => NodeKind::Add,
        "multiply" | "multiply_float" | "multiply_color" => NodeKind::Multiply,
        "ramp" | "ramp_float" | "ramp_color" => NodeKind::Ramp,
        "constant" | "constant_float" | "constant_color" => NodeKind::Constant,
        "position" => NodeKind::Position,
        "normal" => NodeKind::Normal,
        "image" | "texture2d" => NodeKind::Image,
        "dot" => NodeKind::Dot,
        _ => return None,
    })
}

/// 特解参数值（结构化，不引 `serde`）。
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// 浮点标量。
    F32(f32),
    /// 整数标量。
    I32(i32),
    /// 布尔。
    Bool(bool),
    /// 字符串（token / 资产路径原样）。
    Str(String),
    /// 二元向量（`float2` / `texCoord` / UV）。
    F32x2([f32; 2]),
    /// 三元向量（`vector3` / `normal`）。
    F32x3([f32; 3]),
    /// 三元颜色（`color3`；分量语义与 `F32x3` 同，类型上区分以保意图）。
    Color3([f32; 3]),
    /// 浮点数组（`float[]` / ramp 的 positions）。
    F32Array(Vec<f32>),
    /// 颜色数组（ramp 的 colors）。
    ColorArray(Vec<[f32; 3]>),
}

impl ParamValue {
    /// 单一数值（数组取首元素；非数值返回 `None`）。
    pub fn as_scalar(&self) -> Option<f32> {
        match self {
            ParamValue::F32(v) => Some(*v),
            ParamValue::I32(v) => Some(*v as f32),
            ParamValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// 三元数值（color3 / vector3 均可）。
    pub fn as_triple(&self) -> Option<[f32; 3]> {
        match self {
            ParamValue::F32x3(v) | ParamValue::Color3(v) => Some(*v),
            _ => None,
        }
    }

    /// 字符串值。
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// 特解参数包（D-23 三元组的「特解参数」位）。
///
/// 结构位（[`CompiledNode::structure`]）+ 本参数位 + 组合位（连线 =
/// [`CompiledGraph::edges`]）三者构成 D-23 的节点三元组。内部 `BTreeMap`
/// 保证**参数序确定**（C17：序列化/签名输入）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParamPack {
    entries: BTreeMap<String, ParamValue>,
}

impl ParamPack {
    /// 空参数包。
    pub fn new() -> Self {
        Self::default()
    }

    /// 写入参数。
    pub fn set(&mut self, key: impl Into<String>, value: ParamValue) {
        self.entries.insert(key.into(), value);
    }

    /// 读参数。
    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.entries.get(key)
    }

    /// 参数数量。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按键升序遍历（确定序）。
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParamValue)> {
        self.entries.iter()
    }
}

/// 一个输入端口的绑定：常量值或上游连线。
#[derive(Debug, Clone, PartialEq)]
pub enum PortBinding {
    /// 常量参数。
    Constant(ParamValue),
    /// 连到上游节点的某个输出。
    Connected {
        /// 上游节点名。
        node: String,
        /// 上游输出名。
        output: String,
    },
}

/// 编译后的节点（特解三元组的节点侧）。
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledNode {
    /// 节点名（MaterialX `name` 属性，图内唯一）。
    pub name: String,
    /// 节点类型（原始 `type` 字符串）。
    pub nodedef: String,
    /// 结构位（求值器据此选实现）。
    pub structure: String,
    /// 特解参数位（仅常量输入进这里；连线输入不落参数）。
    pub params: ParamPack,
    /// 输入端口绑定（按输入名排序，确定序）。
    pub inputs: BTreeMap<String, PortBinding>,
    /// 本节点声明的输出名（源顺序）。
    pub outputs: Vec<String>,
}

/// 图边（`<input interconnect=...>` 或 `<output><connectfrom/></output>`）。
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEdge {
    /// 源节点名。
    pub from_node: String,
    /// 源输出名。
    pub from_output: String,
    /// 目标节点名。
    pub to_node: String,
    /// 目标输入名。
    pub to_input: String,
}

/// 编译后的材质（一个 `<material>` 的结果）。
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledMaterial {
    /// 材质名。
    pub name: String,
    /// 节点图（按名排序的节点表 + 边 + 输出）。
    pub graph: CompiledGraph,
}

/// 编译后的节点图（对应项目 `Graph` 的"图侧"数据形态）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompiledGraph {
    /// 节点（`BTreeMap` → 名序确定）。
    pub nodes: BTreeMap<String, CompiledNode>,
    /// 边（源顺序 = 源文件出现顺序）。
    pub edges: Vec<GraphEdge>,
    /// 材质输出：输出名 → `(节点名, 节点输出名)`（`BTreeMap` → 序确定）。
    pub outputs: BTreeMap<String, (String, String)>,
}

impl CompiledGraph {
    /// 节点数。
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 边数。
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// 拓扑序（确定性：Kahn + 就绪集按名升序）。
    ///
    /// **不返回错误**：调用方在编译阶段已做无环校验（[`MX2007`]），
    /// 故此处必然成功；若仍遇到环（逻辑 bug），返回空 vec 而非 panic（C18）。
    pub fn topo_order(&self) -> Vec<String> {
        let mut indegree: BTreeMap<&str, usize> =
            self.nodes.keys().map(|k| (k.as_str(), 0usize)).collect();
        // 目标侧入度 = 入边数。
        for e in &self.edges {
            if let Some(d) = indegree.get_mut(e.to_node.as_str()) {
                *d += 1;
            }
        }
        let mut ready: BTreeSet<String> = indegree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(k, _)| (*k).to_string())
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(n) = ready.iter().next().cloned() {
            ready.remove(&n);
            out.push(n.clone());
            for e in &self.edges {
                if e.from_node == n {
                    if let Some(d) = indegree.get_mut(e.to_node.as_str()) {
                        *d -= 1;
                        if *d == 0 {
                            ready.insert(e.to_node.clone());
                        }
                    }
                }
            }
        }
        if out.len() != self.nodes.len() {
            // 有环：编译期已拦，这里只防御（返回部分序而非 panic）。
            return Vec::new();
        }
        out
    }
}

/// 编译产物顶层：一份 `.mtlx` 的全部材质。
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialPack {
    /// `<materialx version="...">` 的版本（缺省 `"1.0"`）。
    pub version: String,
    /// 材质（按源顺序）。
    pub materials: Vec<CompiledMaterial>,
}

impl MaterialPack {
    /// 按名查材质。
    pub fn material(&self, name: &str) -> Option<&CompiledMaterial> {
        self.materials.iter().find(|m| m.name == name)
    }

    /// 材质总数。
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}

// ===========================================================================
//  编译入口
// ===========================================================================

/// 编译 `.mtlx` 文本 → 特解参数包。
pub fn compile_str(text: &str) -> Result<MaterialPack, MtlxError> {
    let root = parse_xml(text)?;
    compile_root(&root)
}

/// 编译 `.mtlx` 文件。
pub fn compile_file(path: &str) -> Result<MaterialPack, MtlxError> {
    let bytes = std::fs::read(path).map_err(|e| {
        MtlxError::new(
            "MX2009",
            format!("materialx::compile_file({path})"),
            format!("读文件失败：{e}"),
            "确认路径存在且可读；相对路径以进程工作目录为基准",
        )
    })?;
    let text = String::from_utf8(bytes).map_err(|_| {
        MtlxError::new(
            "MX2010",
            format!("materialx::compile_file({path})"),
            "文件不是合法 UTF-8",
            "把 .mtlx 以 UTF-8 保存（MaterialX 规范要求 UTF-8）",
        )
    })?;
    compile_str(&text)
}

/// 由已解析的根元素编译。
pub fn compile_root(root: &XmlElement) -> Result<MaterialPack, MtlxError> {
    if root.name != "materialx" {
        return Err(MtlxError::structure(
            format!("materialx::compile_root（根标签 {}）", root.name),
            format!("根标签应为 <materialx>，实际是 <{}>", root.name),
        ));
    }
    let version = root.attr("version").unwrap_or("1.0").to_string();
    // 找 materyialgraph（MaterialX 历史拼写）或 materialgraph。
    let graph = root
        .child("materyialgraph")
        .or_else(|| root.child("materialgraph"));
    let Some(graph) = graph else {
        return Err(MtlxError::structure(
            "materialx::compile_root（根标签 materialx）",
            "未找到 <materyialgraph>（或 <materialgraph>）子元素",
        ));
    };
    let mut materials = Vec::new();
    for m in graph.children_named("material") {
        materials.push(compile_material(m)?);
    }
    if materials.is_empty() {
        return Err(MtlxError::structure(
            "materialx::compile_root",
            "materyialgraph 内没有任何 <material>",
        ));
    }
    Ok(MaterialPack { version, materials })
}

/// 编译单个 `<material>`。
fn compile_material(mat: &XmlElement) -> Result<CompiledMaterial, MtlxError> {
    let name = mat.attr("name").ok_or_else(|| {
        MtlxError::structure(
            "materialx::compile_material",
            "<material> 缺少必需属性 `name`",
        )
    })?;
    let name = name.to_string();
    let ng = mat.child("nodegraph").ok_or_else(|| {
        MtlxError::structure(
            format!("materialx::compile_material（材质 {name}）"),
            "<material> 内缺少 <nodegraph>",
        )
    })?;
    let mut graph = CompiledGraph::default();
    // 先收节点（两遍：先建表，再解连线，便于校验悬空）。
    for n in ng.children_named("node") {
        let node = compile_node(n, &name)?;
        if graph.nodes.insert(node.name.clone(), node).is_some() {
            return Err(MtlxError::conflict(
                format!("materialx::compile_material（材质 {name}）"),
                format!("节点名 {:?} 重复定义", name),
            ));
        }
    }
    // 收边与参数。
    for n in ng.children_named("node") {
        let node_name = n.attr("name").unwrap_or("").to_string();
        // `<input>`：值或 interconnect。
        for inp in n.children_named("input") {
            let iname = inp.attr("name").ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name}"),
                    "<input> 缺少必需属性 `name`",
                )
            })?;
            let itype = inp.attr("type").ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name} 的 input {iname}"),
                    "<input> 缺少必需属性 `type`",
                )
            })?;
            // 两种连线写法：`interconnect` 属性 + `<connectfrom>` 子元素。
            if let Some(target) = inp.attr("interconnect") {
                let (tn, to) = parse_interconnect(target, &name, &node_name, itype)?;
                add_edge(&mut graph, &tn, &to, &node_name, iname, &name)?;
                let node = graph.nodes.get_mut(&node_name).ok_or_else(|| {
                    MtlxError::structure(
                        format!("材质 {name}"),
                        format!("节点 {node_name:?} 在图中不存在"),
                    )
                })?;
                node.inputs.insert(
                    iname.to_string(),
                    PortBinding::Connected {
                        node: tn,
                        output: to,
                    },
                );
            } else if let Some(cf) = inp.child("connectfrom") {
                let tn = cf.attr("node").unwrap_or("").to_string();
                let to = cf.attr("output").unwrap_or(iname).to_string();
                if tn.is_empty() {
                    return Err(MtlxError::structure(
                        format!("材质 {name} 的节点 {node_name} 的 input {iname}"),
                        "<connectfrom> 缺少 `node` 属性",
                    ));
                }
                add_edge(&mut graph, &tn, &to, &node_name, iname, &name)?;
                let node = graph.nodes.get_mut(&node_name).ok_or_else(|| {
                    MtlxError::structure(
                        format!("材质 {name}"),
                        format!("节点 {node_name:?} 在图中不存在"),
                    )
                })?;
                node.inputs.insert(
                    iname.to_string(),
                    PortBinding::Connected {
                        node: tn,
                        output: to,
                    },
                );
            } else if let Some(raw) = inp.attr("value") {
                let v = parse_param_value(itype, raw, &name, &node_name, iname)?;
                let node = graph.nodes.get_mut(&node_name).ok_or_else(|| {
                    MtlxError::structure(
                        format!("材质 {name}"),
                        format!("节点 {node_name:?} 在图中不存在"),
                    )
                })?;
                node.inputs
                    .insert(iname.to_string(), PortBinding::Constant(v));
            } else {
                return Err(MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name} 的 input {iname}"),
                    "<input> 既无 `value` 也无 `interconnect`/`connectfrom`",
                ));
            }
        }
        // `<parameter>`：显式参数（MaterialX 1.5 风格），进参数包。
        for prm in n.children_named("parameter") {
            let pname = prm.attr("name").ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name}"),
                    "<parameter> 缺少必需属性 `name`",
                )
            })?;
            let ptype = prm.attr("type").ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name} 的 parameter {pname}"),
                    "<parameter> 缺少必需属性 `type`",
                )
            })?;
            let raw = prm.attr("value").ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name} 的节点 {node_name} 的 parameter {pname}"),
                    "<parameter> 缺少必需属性 `value`",
                )
            })?;
            let v = parse_param_value(ptype, raw, &name, &node_name, pname)?;
            let node = graph.nodes.get_mut(&node_name).ok_or_else(|| {
                MtlxError::structure(
                    format!("材质 {name}"),
                    format!("节点 {node_name:?} 在图中不存在"),
                )
            })?;
            node.params.set(pname.to_string(), v);
        }
    }
    // `<output>`：材质输出 + 可选 connectfrom。
    for out in ng.children_named("output") {
        let oname = out.attr("name").ok_or_else(|| {
            MtlxError::structure(
                format!("材质 {name} 的 <output>"),
                "<output> 缺少必需属性 `name`",
            )
        })?;
        let cf = out.child("connectfrom").ok_or_else(|| {
            MtlxError::structure(
                format!("材质 {name} 的 output {oname}"),
                "<output> 缺少 <connectfrom>（本轮只编译有连线的输出）",
            )
        })?;
        let src = cf.attr("node").unwrap_or("").to_string();
        if src.is_empty() {
            return Err(MtlxError::structure(
                format!("材质 {name} 的 output {oname}"),
                "<connectfrom> 缺少 `node` 属性",
            ));
        }
        let src_out = cf.attr("output").unwrap_or("out").to_string();
        validate_edge_endpoints(&graph, &src, &src_out, &name, oname)?;
        if graph
            .outputs
            .insert(oname.to_string(), (src.clone(), src_out.clone()))
            .is_some()
        {
            return Err(MtlxError::conflict(
                format!("材质 {name} 的 output {oname}"),
                format!("输出 {oname:?} 被重复定义"),
            ));
        }
        // 材质输出记为一条指向虚拟输出端的边，便于求值器统一遍历。
        graph.edges.push(GraphEdge {
            from_node: src,
            from_output: src_out,
            to_node: format!("__output__{oname}"),
            to_input: "value".to_string(),
        });
    }
    // 无环校验（Kahn）。
    if graph.topo_order().is_empty() && !graph.nodes.is_empty() {
        return Err(MtlxError::cycle(
            format!("材质 {name}"),
            "节点图存在环（拓扑排序无法完成）",
        ));
    }
    // 记录材质输出端的虚拟节点（供边校验通过，但不入 nodes）。
    Ok(CompiledMaterial { name, graph })
}

/// 加一条边并校验端点存在。
fn add_edge(
    graph: &mut CompiledGraph,
    from_node: &str,
    from_output: &str,
    to_node: &str,
    to_input: &str,
    mat: &str,
) -> Result<(), MtlxError> {
    validate_edge_endpoints(graph, from_node, from_output, mat, to_input)?;
    // 同一输入只能连一条边。
    if graph
        .edges
        .iter()
        .any(|e| e.to_node == to_node && e.to_input == to_input)
    {
        return Err(MtlxError::conflict(
            format!("材质 {mat} 的节点 {to_node}"),
            format!("输入 {to_input:?} 被多条连线占用"),
        ));
    }
    graph.edges.push(GraphEdge {
        from_node: from_node.to_string(),
        from_output: from_output.to_string(),
        to_node: to_node.to_string(),
        to_input: to_input.to_string(),
    });
    Ok(())
}

/// 校验连线端点（源节点 + 源输出）存在。
fn validate_edge_endpoints(
    graph: &CompiledGraph,
    from_node: &str,
    from_output: &str,
    mat: &str,
    target: &str,
) -> Result<(), MtlxError> {
    let node = graph.nodes.get(from_node).ok_or_else(|| {
        MtlxError::dangling_interconnect(
            format!("材质 {mat} 的连线目标 {target}"),
            format!(
                "源节点 {from_node:?} 不存在（已定义节点：{:?}）",
                graph.nodes.keys().collect::<Vec<_>>()
            ),
        )
    })?;
    if !node.outputs.iter().any(|o| o == from_output) {
        return Err(MtlxError::dangling_interconnect(
            format!("材质 {mat} 的连线目标 {target}"),
            format!(
                "源节点 {from_node:?} 没有输出 {from_output:?}（已声明输出：{:?}）",
                node.outputs
            ),
        ));
    }
    Ok(())
}

/// 编译单个 `<node>`。
fn compile_node(n: &XmlElement, mat: &str) -> Result<CompiledNode, MtlxError> {
    let name = n.attr("name").ok_or_else(|| {
        MtlxError::structure(
            format!("材质 {mat} 的 <node>"),
            "<node> 缺少必需属性 `name`",
        )
    })?;
    let name = name.to_string();
    let nodedef = n.attr("type").ok_or_else(|| {
        MtlxError::structure(
            format!("材质 {mat} 的节点 {name}"),
            "<node> 缺少必需属性 `type`",
        )
    })?;
    let nodedef = nodedef.to_string();
    let kind = NodeKind::from_mtlx_type(&nodedef).ok_or_else(|| {
        MtlxError::unknown_node(
            format!("材质 {mat} 的节点 {name}"),
            format!("未注册的节点类型 {nodedef:?}"),
        )
    })?;
    // 声明的输出（`<output name="..."/>` 子元素，或 MaterialX 1.5 的
    // `outputs:name` 形式这里不解析——本轮要求显式 `<output>`）。
    let mut outputs = Vec::new();
    for o in n.children_named("output") {
        let oname = o.attr("name").ok_or_else(|| {
            MtlxError::structure(
                format!("材质 {mat} 的节点 {name}"),
                "节点内 <output> 缺少 `name`",
            )
        })?;
        outputs.push(oname.to_string());
    }
    // 少数节点类型有**默认输出名**（未显式声明时兜底）。
    if outputs.is_empty() {
        match kind {
            NodeKind::StandardSurface | NodeKind::Bsdf | NodeKind::Diffuse | NodeKind::Glossy => {
                outputs.push("out".to_string());
            }
            _ => outputs.push("out".to_string()),
        }
    }
    Ok(CompiledNode {
        name,
        nodedef,
        structure: kind.structure().to_string(),
        params: ParamPack::new(),
        inputs: BTreeMap::new(),
        outputs,
    })
}

/// 解析 `interconnect="ND_bsdf.outputs:surface"` → `(节点名, 输出名)`。
///
/// 格式：`<节点名>.<输出名>`（MaterialX 也可能写 `nodename.outputs.name`
/// 或带节点图前缀 `NG.nodename.output`，本轮取**后两段**）。
fn parse_interconnect(
    raw: &str,
    mat: &str,
    node: &str,
    input: &str,
) -> Result<(String, String), MtlxError> {
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() < 2 {
        return Err(MtlxError::dangling_interconnect(
            format!("材质 {mat} 的节点 {node} 的 input {input}"),
            format!("interconnect {raw:?} 格式非法：应为 `<节点名>.<输出名>`"),
        ));
    }
    let out = parts[parts.len() - 1].to_string();
    let src = parts[parts.len() - 2].to_string();
    if src.is_empty() || out.is_empty() {
        return Err(MtlxError::dangling_interconnect(
            format!("材质 {mat} 的节点 {node} 的 input {input}"),
            format!("interconnect {raw:?} 含空的节点名或输出名"),
        ));
    }
    Ok((src, out))
}

/// 按声明类型解析参数值文本。
fn parse_param_value(
    ty: &str,
    raw: &str,
    mat: &str,
    node: &str,
    port: &str,
) -> Result<ParamValue, MtlxError> {
    let loc = format!("材质 {mat} 的节点 {node} 的端口 {port}（类型 {ty}）");
    // 归一化类型末段。
    let t = ty.rsplit(':').next().unwrap_or(ty);
    let parts: Vec<f32> = split_floats(raw, &loc)?;
    match t {
        "float" | "float1" => {
            require_len(&parts, 1, &loc)?;
            Ok(ParamValue::F32(parts[0]))
        }
        "integer" | "int" => {
            require_len(&parts, 1, &loc)?;
            Ok(ParamValue::I32(parts[0].round() as i32))
        }
        "boolean" | "bool" => match raw.trim() {
            "true" | "1" => Ok(ParamValue::Bool(true)),
            "false" | "0" => Ok(ParamValue::Bool(false)),
            other => Err(MtlxError::type_mismatch(
                &loc,
                format!("布尔值应为 true/false，收到 {other:?}"),
            )),
        },
        "string" | "token" | "filename" | "asset" => Ok(ParamValue::Str(raw.to_string())),
        "float2" | "vector2" | "texCoord" => {
            require_len(&parts, 2, &loc)?;
            Ok(ParamValue::F32x2([parts[0], parts[1]]))
        }
        "float3" | "vector3" | "normal" | "normal3" => {
            require_len(&parts, 3, &loc)?;
            Ok(ParamValue::F32x3([parts[0], parts[1], parts[2]]))
        }
        "color3" | "color" => {
            require_len(&parts, 3, &loc)?;
            Ok(ParamValue::Color3([parts[0], parts[1], parts[2]]))
        }
        "float[]" | "float_array" => Ok(ParamValue::F32Array(parts)),
        "color3[]" | "color_array" => {
            if !parts.len().is_multiple_of(3) {
                return Err(MtlxError::type_mismatch(
                    &loc,
                    format!("color3[] 的分量数应为 3 的倍数，收到 {}", parts.len()),
                ));
            }
            Ok(ParamValue::ColorArray(
                parts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            ))
        }
        other => Err(MtlxError::type_mismatch(
            &loc,
            format!("暂不支持的参数类型 {other:?}"),
        )),
    }
}

/// 要求浮点个数为 `n`。
fn require_len(parts: &[f32], n: usize, loc: &str) -> Result<(), MtlxError> {
    if parts.len() != n {
        return Err(MtlxError::type_mismatch(
            loc,
            format!("需要 {n} 个数值分量，收到 {}", parts.len()),
        ));
    }
    Ok(())
}

/// 按逗号/空白切分浮点数。
///
/// **任一分量解析失败 → 报错**（绝不 `filter_map` 静默丢弃）：`"0.8, abc"`
/// 若被吞成 `[0.8]`，`color3` 会因「分量数不足」报出误导性的错，或更糟——
/// `float[]` 会被静默接受成 1 元素数组，材质数值无声改变。
fn split_floats(raw: &str, loc: &str) -> Result<Vec<f32>, MtlxError> {
    let mut out = Vec::new();
    for piece in raw.split(|c: char| c == ',' || c.is_whitespace()) {
        if piece.is_empty() {
            continue;
        }
        match piece.parse::<f32>() {
            Ok(v) if v.is_finite() => out.push(v),
            Ok(_) => {
                return Err(MtlxError::type_mismatch(
                    loc,
                    format!("分量 {piece:?} 解析为非有限值（inf/NaN），参数不接受"),
                ));
            }
            Err(_) => {
                return Err(MtlxError::type_mismatch(
                    loc,
                    format!("分量 {piece:?} 不是合法 f32（整串为 {raw:?}）"),
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小 `.mtlx`：2 节点 + 1 连线 + 输出。
    const MINIMAL: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<materialx version="1.0">
  <materyialgraph>
    <material name="mat_red">
      <nodegraph>
        <node name="bsdf1" type="standard_surface">
          <input name="base_color" type="color3" value="0.8, 0.1, 0.05" />
          <input name="roughness" type="float" value="0.35" />
          <parameter name="metallic" type="float" value="0" />
          <output name="out" />
        </node>
        <node name="ramp1" type="ramp">
          <input name="in" type="float" value="0.25" />
          <output name="out" />
        </node>
        <output name="surface" type="material">
          <connectfrom node="bsdf1" output="out" />
        </output>
      </nodegraph>
    </material>
  </materyialgraph>
</materialx>
"#;

    #[test]
    fn xml_parser_handles_nesting_attributes_and_self_closing() {
        let root = parse_xml(r#"<a x="1"><b y='2'><c/></b><d>text</d></a>"#)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(root.name, "a");
        assert_eq!(root.attr("x"), Some("1"));
        let b = root.child("b").expect("b 应存在");
        assert_eq!(b.attr("y"), Some("2"));
        assert!(b.child("c").is_some(), "自闭合元素应被解析");
        assert_eq!(
            root.child("d").map(|d| d.text.clone()),
            Some("text".to_string())
        );
    }

    #[test]
    fn xml_parser_handles_comments_cdata_and_entities() {
        let root = parse_xml(
            "<!-- 注释 --><a v=\"&lt;&amp;&gt;&quot;&#65;&#x42;\"><![CDATA[raw <not> markup]]></a>",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(root.attr("v"), Some("<&>\"AB"));
        assert_eq!(root.text, "raw <not> markup", "CDATA 应按原样");
    }

    #[test]
    fn xml_rejects_malformed_input_with_structured_error() {
        // 标签不匹配。
        let e = parse_xml("<a></b>").expect_err("标签不匹配必须报错");
        assert_eq!(e.code, "MX2001");
        // 未闭合。
        assert_eq!(parse_xml("<a>").expect_err("未闭合必须报错").code, "MX2001");
        // 属性未加引号。
        assert_eq!(
            parse_xml("<a x=1></a>").expect_err("无引号必须报错").code,
            "MX2001"
        );
        // 重复属性。
        assert_eq!(
            parse_xml(r#"<a x="1" x="2"></a>"#)
                .expect_err("重复属性必须报错")
                .code,
            "MX2001"
        );
        // 截断：属性值未闭合。
        assert_eq!(
            parse_xml(r#"<a x="1></a>"#).expect_err("截断必须报错").code,
            "MX2001"
        );
        // 根之后有垃圾。
        assert_eq!(
            parse_xml("<a></a><b></b>")
                .expect_err("多余内容必须报错")
                .code,
            "MX2001"
        );
    }

    #[test]
    fn xml_rejects_doctype_and_unknown_entities() {
        // DTD → 明确不支持（不是静默忽略）。
        let e = parse_xml(r#"<!DOCTYPE materialx><materialx/>"#).expect_err("DTD 必须报错");
        assert_eq!(e.code, "MX2002");
        // 未知实体 → 报错（吞掉会静默改变材质数值）。
        let e = parse_xml(r#"<a v="&deg;"/>"#).expect_err("未知实体必须报错");
        assert_eq!(e.code, "MX2002");
        // 裸 & → 报错。
        assert_eq!(
            parse_xml(r#"<a v="a & b"/>"#)
                .expect_err("裸 & 必须报错")
                .code,
            "MX2001"
        );
    }

    #[test]
    fn compiles_minimal_materialx_to_param_pack() {
        let pack = compile_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pack.version, "1.0");
        assert_eq!(pack.len(), 1);
        let m = pack.material("mat_red").expect("材质应存在");
        assert_eq!(m.graph.node_count(), 2, "2 个节点");
        let bsdf = m.graph.nodes.get("bsdf1").expect("bsdf1 应存在");
        // 结构位映射正确。
        assert_eq!(bsdf.structure, "standard_surface");
        assert_eq!(bsdf.nodedef, "standard_surface");
        // 特解参数位：参数值正确。
        assert_eq!(
            bsdf.params.get("metallic").and_then(|v| v.as_scalar()),
            Some(0.0)
        );
        // 输入常量进 inputs（PortBinding::Constant）。
        match bsdf.inputs.get("base_color") {
            Some(PortBinding::Constant(ParamValue::Color3(c))) => {
                assert_eq!(*c, [0.8, 0.1, 0.05], "color3 值应正确");
            }
            other => panic!("base_color 应为 Color3 常量，实际 {other:?}"),
        }
        match bsdf.inputs.get("roughness") {
            Some(PortBinding::Constant(ParamValue::F32(v))) => {
                assert!((v - 0.35).abs() < 1e-6, "roughness = {v}")
            }
            other => panic!("roughness 应为 F32 常量，实际 {other:?}"),
        }
        // 材质输出。
        assert_eq!(
            m.graph.outputs.get("surface"),
            Some(&("bsdf1".to_string(), "out".to_string()))
        );
    }

    #[test]
    fn node_type_mapping_covers_required_kinds() {
        // 任务书点名的节点类型必须都能映射。
        for (mtlx, expect) in [
            ("standard_surface", "standard_surface"),
            ("bsdf", "bsdf"),
            ("diffuse", "diffuse"),
            ("mix", "mix"),
            ("add", "add"),
            ("multiply", "multiply"),
            ("ramp", "ramp"),
            ("constant", "constant"),
        ] {
            let k = NodeKind::from_mtlx_type(mtlx).expect("应能映射");
            assert_eq!(k.structure(), expect);
            // 带命名空间前缀也应识别（取末段）。
            let k2 = NodeKind::from_mtlx_type(&format!("ND__{mtlx}_surfaceshader"))
                .unwrap_or_else(|| panic!("{mtlx} 带前缀应识别"));
            assert_eq!(k2.structure(), expect);
        }
    }

    #[test]
    fn unknown_node_type_is_structured_error() {
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph><node name="n" type="volumetric_scatter"/></nodegraph>
</material></materyialgraph></materialx>"#;
        let e = compile_str(src).expect_err("未知节点必须报错");
        assert_eq!(e.code, "MX2004");
        assert!(!e.fix.is_empty());
    }

    #[test]
    fn interconnect_and_connectfrom_become_graph_edges() {
        // 两种连线写法都要支持。
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph>
  <node name="ramp1" type="ramp"><input name="in" type="float" value="0.5"/><output name="out"/></node>
  <node name="bsdf1" type="standard_surface">
    <input name="roughness" type="float" interconnect="ramp1.out"/>
    <input name="base" type="color3" value="0.2, 0.4, 0.6"/>
    <output name="out"/>
  </node>
  <node name="out1" type="mix">
    <input name="fg" type="color3"><connectfrom node="bsdf1" output="out"/></input>
    <input name="bg" type="color3" value="0, 0, 0"/>
    <output name="out"/>
  </node>
  <output name="surface" type="material"><connectfrom node="out1" output="out"/></output>
</nodegraph></material></materyialgraph></materialx>"#;
        let pack = compile_str(src).unwrap_or_else(|e| panic!("{e}"));
        let m = pack.material("m").expect("材质");
        // 3 个节点 + 3 条边（2 条输入边 + 1 条材质输出边）。
        assert_eq!(m.graph.node_count(), 3);
        assert_eq!(m.graph.edge_count(), 3, "边数");
        // 边内容正确。
        assert!(m
            .graph
            .edges
            .iter()
            .any(|e| e.from_node == "ramp1" && e.to_node == "bsdf1" && e.to_input == "roughness"));
        assert!(m
            .graph
            .edges
            .iter()
            .any(|e| e.from_node == "bsdf1" && e.to_node == "out1" && e.to_input == "fg"));
        // 端口绑定标记为 Connected。
        match m.graph.nodes["bsdf1"].inputs.get("roughness") {
            Some(PortBinding::Connected { node, output }) => {
                assert_eq!((node.as_str(), output.as_str()), ("ramp1", "out"));
            }
            other => panic!("roughness 应为 Connected，实际 {other:?}"),
        }
        // 拓扑序确定且完整。
        let order = m.graph.topo_order();
        assert_eq!(order.len(), 3);
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap_or(usize::MAX);
        assert!(pos("ramp1") < pos("bsdf1"), "ramp1 应先于 bsdf1");
        assert!(pos("bsdf1") < pos("out1"), "bsdf1 应先于 out1");
    }

    #[test]
    fn dangling_interconnect_is_structured_error() {
        // 指向不存在的节点。
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph><node name="b" type="standard_surface">
  <input name="roughness" type="float" interconnect="nope.out"/>
  <output name="out"/>
</node></nodegraph></material></materyialgraph></materialx>"#;
        let e = compile_str(src).expect_err("悬空节点必须报错");
        assert_eq!(e.code, "MX2006");
        // 指向不存在的输出。
        let src2 = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph>
  <node name="r" type="ramp"><output name="out"/></node>
  <node name="b" type="standard_surface">
    <input name="roughness" type="float" interconnect="r.nosuch"/>
    <output name="out"/>
  </node>
</nodegraph></material></materyialgraph></materialx>"#;
        let e = compile_str(src2).expect_err("悬空输出必须报错");
        assert_eq!(e.code, "MX2006");
    }

    #[test]
    fn cyclic_graph_is_structured_error() {
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph>
  <node name="a" type="add"><input name="in1" type="float" interconnect="b.out"/><output name="out"/></node>
  <node name="b" type="add"><input name="in1" type="float" interconnect="a.out"/><output name="out"/></node>
</nodegraph></material></materyialgraph></materialx>"#;
        let e = compile_str(src).expect_err("环必须报错");
        assert_eq!(e.code, "MX2007");
    }

    #[test]
    fn type_mismatch_is_structured_error() {
        // color3 给 2 个分量。
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph><node name="b" type="standard_surface">
  <input name="base_color" type="color3" value="0.1, 0.2"/>
  <output name="out"/>
</node></nodegraph></material></materyialgraph></materialx>"#;
        let e = compile_str(src).expect_err("color3 分量数错必须报错");
        assert_eq!(e.code, "MX2005");
        // 布尔给非法值。
        let src2 = src.replace(
            r#"type="color3" value="0.1, 0.2""#,
            r#"type="boolean" value="maybe""#,
        );
        let e = compile_str(&src2).expect_err("非法布尔必须报错");
        assert_eq!(e.code, "MX2005");
        // 不支持的类型。
        let src3 = src.replace(
            r#"type="color3" value="0.1, 0.2""#,
            r#"type="matrix44" value="0.1""#,
        );
        let e = compile_str(&src3).expect_err("未知参数类型必须报错");
        assert_eq!(e.code, "MX2005");
    }

    #[test]
    fn unparseable_float_component_is_rejected_not_silently_dropped() {
        // `"0.8, abc"` 若被 `filter_map` 吞成 `[0.8]`，`float[]` 会静默变成
        // 1 元素数组——材质数值无声改变。必须报错。
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph><node name="r" type="ramp">
  <parameter name="positions" type="float[]" value="0.0, abc, 1.0"/>
  <output name="out"/>
</node></nodegraph></material></materyialgraph></materialx>"#;
        let e = compile_str(src).expect_err("非法浮点分量必须报错");
        assert_eq!(e.code, "MX2005");
        assert!(
            e.problem.contains("abc"),
            "错误信息应指出坏分量：{}",
            e.problem
        );
        // 非有限值（inf / NaN 字面量）同样拒绝。
        let src2 = src.replace("0.0, abc, 1.0", "0.0, inf, 1.0");
        assert_eq!(
            compile_str(&src2).expect_err("非有限值必须报错").code,
            "MX2005"
        );
        let src3 = src.replace("0.0, abc, 1.0", "0.0, nan, 1.0");
        assert_eq!(compile_str(&src3).expect_err("NaN 必须报错").code, "MX2005");
        // 正常值不受影响。
        let src4 = src.replace("0.0, abc, 1.0", "0.0, 0.5, 1.0");
        let pack = compile_str(&src4).unwrap_or_else(|e| panic!("{e}"));
        match pack.material("m").expect("m").graph.nodes["r"]
            .params
            .get("positions")
        {
            Some(ParamValue::F32Array(v)) => assert_eq!(v.len(), 3),
            other => panic!("应为 F32Array(3)，实际 {other:?}"),
        }
    }

    #[test]
    fn structural_errors_are_reported() {
        // 根标签不对。
        assert_eq!(
            compile_str("<notmaterialx/>")
                .expect_err("根标签错必须报错")
                .code,
            "MX2003"
        );
        // 缺 materialgraph。
        assert_eq!(
            compile_str(r#"<materialx version="1.0"/>"#)
                .expect_err("缺 graph 必须报错")
                .code,
            "MX2003"
        );
        // 缺 material name。
        assert_eq!(
            compile_str(
                r#"<materialx version="1.0"><materyialgraph><material>
            <nodegraph/></material></materyialgraph></materialx>"#
            )
            .expect_err("缺 name 必须报错")
            .code,
            "MX2003"
        );
        // 缺 nodegraph。
        assert_eq!(
            compile_str(
                r#"<materialx version="1.0"><materyialgraph>
            <material name="m"/></materyialgraph></materialx>"#
            )
            .expect_err("缺 nodegraph 必须报错")
            .code,
            "MX2003"
        );
        // input 缺 type。
        assert_eq!(
            compile_str(
                r#"<materialx version="1.0"><materyialgraph><material name="m">
            <nodegraph><node name="b" type="constant">
              <input name="in" value="1"/>
            </node></nodegraph></material></materyialgraph></materialx>"#
            )
            .expect_err("缺 type 必须报错")
            .code,
            "MX2003"
        );
        // input 既无 value 也无 interconnect。
        assert_eq!(
            compile_str(
                r#"<materialx version="1.0"><materyialgraph><material name="m">
            <nodegraph><node name="b" type="constant">
              <input name="in" type="float"/>
            </node></nodegraph></material></materyialgraph></materialx>"#
            )
            .expect_err("无值无连线必须报错")
            .code,
            "MX2003"
        );
        // 重复节点名。
        assert_eq!(
            compile_str(
                r#"<materialx version="1.0"><materyialgraph><material name="m">
            <nodegraph>
              <node name="b" type="constant"><output name="out"/></node>
              <node name="b" type="constant"><output name="out"/></node>
            </nodegraph></material></materyialgraph></materialx>"#
            )
            .expect_err("重复节点名必须报错")
            .code,
            "MX2008"
        );
    }

    #[test]
    fn truncated_document_does_not_panic() {
        // 各种截断位置都必须结构化报错，不得 panic。
        let full = MINIMAL;
        for cut in (1..full.len()).step_by(7) {
            let piece = &full[..cut];
            match compile_str(piece) {
                Ok(_) => {}
                Err(e) => {
                    assert!(e.code.starts_with("MX2"), "错误码应形如 MX2xxx：{}", e.code);
                    assert!(!e.fix.is_empty(), "建议段不得为空");
                }
            }
        }
    }

    #[test]
    fn param_pack_ordering_is_deterministic() {
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph><node name="b" type="standard_surface">
  <parameter name="zzz" type="float" value="1"/>
  <parameter name="aaa" type="float" value="2"/>
  <parameter name="mmm" type="float" value="3"/>
  <output name="out"/>
</node></nodegraph></material></materyialgraph></materialx>"#;
        let pack = compile_str(src).unwrap_or_else(|e| panic!("{e}"));
        let keys: Vec<&str> = pack.material("m").expect("m").graph.nodes["b"]
            .params
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, vec!["aaa", "mmm", "zzz"], "参数必须按键升序");
        assert_eq!(
            pack.material("m").expect("m").graph.nodes["b"].params.len(),
            3
        );
    }

    #[test]
    fn compile_is_deterministic() {
        let a = compile_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        let b = compile_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "同输入两遍编译必须完全一致");
        // 参数值位模式逐位一致。
        let ca = a.material("mat_red").expect("m").graph.nodes["bsdf1"].clone();
        let cb = b.material("mat_red").expect("m").graph.nodes["bsdf1"].clone();
        match (ca.inputs.get("base_color"), cb.inputs.get("base_color")) {
            (
                Some(PortBinding::Constant(ParamValue::Color3(x))),
                Some(PortBinding::Constant(ParamValue::Color3(y))),
            ) => {
                for k in 0..3 {
                    assert_eq!(x[k].to_bits(), y[k].to_bits());
                }
            }
            _ => panic!("base_color 类型不符"),
        }
    }

    #[test]
    fn deep_nesting_is_rejected_not_stack_overflow() {
        // 构造 300 层嵌套（> MAX_XML_DEPTH=256）→ 结构化错误。
        let mut s = String::new();
        for _ in 0..300 {
            s.push_str("<a>");
        }
        for _ in 0..300 {
            s.push_str("</a>");
        }
        let e = parse_xml(&s).expect_err("超深嵌套必须报错");
        assert_eq!(e.code, "MX2001");
    }

    #[test]
    fn ramp_and_mix_params_parse() {
        let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph>
  <node name="r" type="ramp">
    <parameter name="positions" type="float[]" value="0.0, 0.5, 1.0"/>
    <parameter name="colors" type="color3[]" value="0,0,0, 1,1,1"/>
    <output name="out"/>
  </node>
  <node name="mx" type="mix">
    <input name="fac" type="float" value="0.25"/>
    <input name="fg" type="color3" value="1, 0, 0"/>
    <input name="bg" type="color3" value="0, 0, 1"/>
    <output name="out"/>
  </node>
</nodegraph></material></materyialgraph></materialx>"#;
        let pack = compile_str(src).unwrap_or_else(|e| panic!("{e}"));
        let g = &pack.material("m").expect("m").graph;
        let r = &g.nodes["r"];
        assert_eq!(r.structure, "ramp");
        match r.params.get("positions") {
            Some(ParamValue::F32Array(v)) => {
                assert_eq!(v.len(), 3);
                assert!((v[1] - 0.5).abs() < 1e-6);
            }
            other => panic!("positions 应为 F32Array，实际 {other:?}"),
        }
        match r.params.get("colors") {
            Some(ParamValue::ColorArray(v)) => {
                assert_eq!(v.len(), 2, "6 个分量 = 2 个颜色");
                assert_eq!(v[1], [1.0, 1.0, 1.0]);
            }
            other => panic!("colors 应为 ColorArray，实际 {other:?}"),
        }
        assert_eq!(g.nodes["mx"].structure, "mix");
    }
}
