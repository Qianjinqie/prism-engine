//! 复杂脚本整形（U10）：字素簇图 + 阿拉伯系连接 + 印度系重排。
//!
//! 语义权威：《架构总设计》v2.4 §16（文字管线：SDF 字形 + 覆盖率 + 排版规则）、
//! 宪章 D-16（Mini Code）、D-29（确定性数学件）。本模块是 §16 文字管线的**上游**
//! 整形段，补 `layout.rs`（度量/断行/禁则）缺失的「字符流 → 定位字形流」一步。
//!
//! # 为什么不用 Unicode 库（铁律：零第三方依赖）
//! Rust 生态有 `unicode-segmentation` / `unicode-bidi` / `unicode-script` 等一批库，
//! **本项目一个都不引**。UAX #29（字素簇）、UAX #9（双向）、UAX #24（脚本）所需的
//! 数据表与状态机在此**全部手写**：分类表是 `const` 升序区间数组 + 二分查找，
//! 状态机是本文件内的显式循环。连 `char::is_alphabetic()` 也不用——那也是 std
//! 内部的 Unicode 属性表；手写表才可控可审计。
//!
//! # 覆盖范围（诚实边界，勿高估）
//! | 段 | 覆盖 | 未覆盖 / 简化 |
//! |---|---|---|
//! | 阿拉伯连接 | **U+0621–U+064A 核心字母逐码位**、U+0671/0672/0673/066E/066F/06C0/06CC/06D2/06D3、U+0640 TATWEEL、U+200C/200D | U+0750–U+077F 增补、U+08A0–U+08FF 扩展-A、U+06D5–U+06ED 记号整段标 `NonJoining`（**不整形是诚实的保守默认**，错连比不连更糟） |
//! | lam-alef 连字 | U+0644 + {U+0622,0623,0625,0627,0671} | 呈现形式只给 isolated(U+FEFB)/final(U+FEFC) 两变体（medial 与 final 同形，标准实践） |
//! | 印度系 | **仅天城文 U+0900–U+097F**：i-matra 前置重排、virama conjunct 合并、nukta 吸收 | 孟加拉/古木尔蒂/泰米尔**只分类不整形**（`Script::IndicOther`，接口已留）；不做半形 GSUB、reph、rakar、连字后置元音 |
//! | 双向 | 段落基线 + **L2 反转规则**（level 0/1 两层） | **不是完整 UAX #9**：无显式嵌入控制符、无隔离符、无括号对(BD16)、数字在 RTL 段内按基线层级 |
//! | 字体 | 内嵌 Prism Sans SC 子集**不含阿拉伯/天城文字形** | 故 `gid` 回落 0、advance=0。整形产出**整形计划**（簇/形式/序），与字形替换（GSUB `init/medi/fina/isol/liga`）解耦——后者是字体的活 |
//!
//! # 确定性契约
//! `shape` 是 (字体字节, 文本, 参数) 的纯函数：无隐藏状态、无随机源、遍历序全部
//! 由输入下标与升序表决定（禁 HashMap 迭代序）。同输入两遍 `canonical_bytes()`
//! 逐字节一致（`shaping_is_byte_level_deterministic` 锁定）。

use std::cmp::Ordering;
use std::fmt;

use crate::ttf::Font;

// ===========================================================================
//  错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）
// ===========================================================================

/// 整形错误。错误码取 P1 段（防呆参数，宪章 C18 分段）。
///
/// 刻意**不含**「字形缺失」一类：内嵌字体子集本就不含阿拉伯/天城文字形，把
/// `.notdef` 升级成错误会让整形层与字体子集策略耦合。缺字形由
/// [`ShapedRun::missing_glyphs`] 如实计数暴露（诚实报告，不静默、不假装成功）。
#[derive(Debug, Clone, PartialEq)]
pub struct TextError {
    /// 错误码（P1xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 参数。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for TextError {}

/// 参数非法（字号非正/非有限、字距非有限）。
fn param_error(location: &str, problem: String, fix: &str) -> TextError {
    TextError { code: "P1101", location: location.to_string(), problem, fix: fix.to_string() }
}

// ===========================================================================
//  脚本分类（手写升序区间表；UAX #24 的最小子集）
// ===========================================================================

/// 脚本类别。**覆盖范围见模块头表格**；未列入的码位归 [`Script::Unknown`]
/// （如实标未知，不向邻脚本猜）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Script {
    /// 通用：空白、标点、数字、符号（方向中性，按上下文归附）。
    Common,
    /// 继承性记号：依附于前一个基字符（组合附加符号）。
    Inherited,
    /// 拉丁（含 Latin-1 补充、IPA、扩展附加区）。
    Latin,
    /// 希腊。
    Greek,
    /// 西里尔。
    Cyrillic,
    /// 希伯来（RTL）。
    Hebrew,
    /// 阿拉伯（RTL；连接整形已实现）。
    Arabic,
    /// 天城文（印度系；前置重排 + 辅音簇已实现）。
    Devanagari,
    /// 其他印度系脚本（**只分类不整形**——接口预留，见模块头覆盖表）。
    IndicOther,
    /// 汉字。
    Han,
    /// 假名（平/片/半角）。
    Kana,
    /// 未覆盖 / 未知脚本。
    Unknown,
}

impl Script {
    /// 是否为从右到左的强方向脚本（决定双向层级与视觉序反转）。
    pub fn is_rtl(self) -> bool {
        matches!(self, Script::Hebrew | Script::Arabic)
    }

    /// 是否为可作为簇基字符的强方向脚本（Common/Inherited 不作基字符）。
    pub fn is_strong(self) -> bool {
        !matches!(self, Script::Common | Script::Inherited | Script::Unknown)
    }
}

/// 手写脚本区间表（UAX #24 最小子集）：`(起始, 结束, 脚本)`。
///
/// **升序、互不重叠**（`script_range_table_is_sorted_and_disjoint` 锁定——这是
/// [`lookup_range`] 二分查找的唯一正确性前提）。覆盖区段见模块头表格；为控制
/// 表规模，对无关脚本（西伯利亚、美洲、东南亚、非洲）**整段归 `Unknown`**，
/// 如实表示"未覆盖"而非谎称已知。
const SCRIPT_RANGES: &[(u32, u32, Script)] = &[
    (0x0000, 0x0040, Script::Common),     // ASCII 控制/空格/标点
    (0x0041, 0x005A, Script::Latin),      // A-Z
    (0x005B, 0x0060, Script::Common),     // [ \ ] ^ _ `
    (0x0061, 0x007A, Script::Latin),      // a-z
    (0x007B, 0x00BF, Script::Common),     // { | } ~ DEL 及 Latin-1 标点
    (0x00C0, 0x02AF, Script::Latin),      // Latin-1 字母 + 扩展 A/B + IPA
    (0x02B0, 0x02FF, Script::Common),     // 修饰字母
    (0x0300, 0x036F, Script::Inherited),  // 组合附加符号
    (0x0370, 0x03FF, Script::Greek),
    (0x0400, 0x0482, Script::Cyrillic),
    (0x0483, 0x0489, Script::Inherited),  // 组合西里尔
    (0x048A, 0x052F, Script::Cyrillic),
    (0x0530, 0x058F, Script::Unknown),    // 亚美尼亚（未覆盖）
    (0x0590, 0x05FF, Script::Hebrew),
    (0x0600, 0x06FF, Script::Arabic),     // 阿拉伯文
    (0x0700, 0x074F, Script::Unknown),    // 叙利亚（未覆盖）
    (0x0750, 0x077F, Script::Arabic),     // 阿拉伯增补
    (0x0780, 0x07BF, Script::Unknown),    // 塔阿尼（未覆盖）
    (0x07C0, 0x08FF, Script::Arabic),     // 尼科 + 萨马利/曼达（阿拉伯系归类）
    (0x0900, 0x097F, Script::Devanagari), // 天城文
    (0x0980, 0x0DFF, Script::IndicOther), // 孟加拉..马拉雅拉姆（只分类不整形）
    (0x0E00, 0x1FFF, Script::Unknown),    // 东南亚/美洲/西伯利亚（未覆盖）
    (0x2000, 0x206F, Script::Common),     // 通用标点 + ZWNJ/ZWJ + 方向标记
    (0x2070, 0x20CF, Script::Common),     // 上下标 + 货币
    (0x20D0, 0x20F0, Script::Inherited),  // 符号组合附加
    (0x2100, 0x2BFF, Script::Common),     // 字母类符号 + 箭头 + 数学
    (0x2C00, 0x2DFF, Script::Common),
    (0x2E00, 0x2FDF, Script::Han),        // CJK 部首
    (0x2FF0, 0x303F, Script::Common),     // 康熙部首补充 + CJK 标点
    (0x3040, 0x30FF, Script::Kana),       // 平/片假名
    (0x3100, 0x33FF, Script::Unknown),    // 注音/谚文/兼容（未覆盖）
    (0x3400, 0x4DBF, Script::Han),        // 扩展 A
    (0x4DC0, 0x9FFF, Script::Han),        // 彝文 + 统一表意
    (0xA000, 0xABFF, Script::Unknown),    // 彝/苏巴（未覆盖）
    (0xAC00, 0xD7AF, Script::Unknown),    // 谚文音节（未覆盖）
    (0xD800, 0xF8FF, Script::Common),     // 代理区（不应出现）+ 私用
    (0xF900, 0xFAFF, Script::Han),        // CJK 兼容表意
    (0xFB00, 0xFB4F, Script::Latin),      // 拉丁 presentation forms
    (0xFB50, 0xFDFF, Script::Arabic),     // 阿拉伯 presentation forms-A
    (0xFE00, 0xFE0F, Script::Inherited),  // 变体选择符
    (0xFE10, 0xFE1F, Script::Common),     // 竖排标点
    (0xFE20, 0xFE2F, Script::Inherited),  // 组合半标记
    (0xFE30, 0xFE6F, Script::Common),     // CJK 兼容形式
    (0xFE70, 0xFEFF, Script::Arabic),     // 阿拉伯 presentation forms-B
    (0xFF00, 0xFF20, Script::Common),     // 全角形式
    (0xFF21, 0xFF3A, Script::Latin),
    (0xFF3B, 0xFF40, Script::Common),
    (0xFF41, 0xFF5A, Script::Latin),
    (0xFF5B, 0xFF65, Script::Common),
    (0xFF66, 0xFF9F, Script::Kana),       // 半角假名
    (0xFFA0, 0xFFDF, Script::Unknown),    // 谚文半角（未覆盖）
    (0xFFE0, 0xFFFF, Script::Common),     // 货币/全角符号
    (0x10000, 0x1FFFF, Script::Unknown),  // 增补平面（未覆盖）
    (0x20000, 0x2A6DF, Script::Han),      // 扩展 B
    (0x2A700, 0x2FA1F, Script::Han),      // 扩展 C-F
    (0x30000, 0x3134F, Script::Han),      // 扩展 G-H
];

/// 在升序区间表中二分查找 `cp` 所属值；未覆盖返回 `None`。
///
/// 区间表**必须升序且互不重叠**（由测试锁定），否则二分结果无意义。
fn lookup_range<T: Copy>(table: &[(u32, u32, T)], cp: u32) -> Option<T> {
    // 找第一个 `start > cp` 的下标，则其前一个是唯一可能的命中区间。
    let mut lo = 0usize;
    let mut hi = table.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let start = table.get(mid).map(|e| e.0).unwrap_or(u32::MAX);
        if start <= cp {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let entry = table.get(lo.checked_sub(1)?)?;
    if cp <= entry.1 {
        Some(entry.2)
    } else {
        None
    }
}

/// 码位 → 脚本。未覆盖 → [`Script::Unknown`]（如实标未知，不向邻脚本猜）。
pub fn script_of(ch: char) -> Script {
    lookup_range(SCRIPT_RANGES, ch as u32).unwrap_or(Script::Unknown)
}

// ===========================================================================
//  组合记号表（UAX #29 Extend 的最小子集；手写升序区间表）
// ===========================================================================

/// 手写组合记号区间表：`(起始, 结束)`，升序互不重叠。
///
/// 覆盖：U+0300–U+036F、U+0483–U+0489、U+0591–U+05BD、U+05BF、U+05C1–U+05C2、
/// U+05C4–U+05C5、U+05C7、U+0610–U+061A、U+064B–U+065F、U+0670、U+06D6–U+06DC、
/// U+06DF–U+06E4、U+06E7–U+06E8、U+06EA–U+06ED、U+0711、U+0730–U+074A、
/// U+07A6–U+07B0、U+0900–U+0902（天城文 anusvara/candrabindu——UAX #29 属
/// SpacingMark，但对天城文整形等价并入，见 [`DevanagariCategory::Sign`]）、
/// U+093A–U+094D、U+0951–U+0957、U+0962–U+0963、U+1AB0–U+1AFF、U+1DC0–U+1DFF、
/// U+20D0–U+20F0、U+FE00–U+FE0F、U+FE20–U+FE2F。
const MARK_RANGES: &[(u32, u32)] = &[
    (0x0300, 0x036F),
    (0x0483, 0x0489),
    (0x0591, 0x05BD),
    (0x05BF, 0x05BF),
    (0x05C1, 0x05C2),
    (0x05C4, 0x05C5),
    (0x05C7, 0x05C7),
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06DC),
    (0x06DF, 0x06E4),
    (0x06E7, 0x06E8),
    (0x06EA, 0x06ED),
    (0x0711, 0x0711),
    (0x0730, 0x074A),
    (0x07A6, 0x07B0),
    (0x0900, 0x0902),
    (0x093A, 0x094D),
    (0x0951, 0x0957),
    (0x0962, 0x0963),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x200C, 0x200D), // ZWNJ / ZWJ（UAX #29 属 Extend：不并入书写但并入簇）
    (0x20D0, 0x20F0),
    (0xFE00, 0xFE0F),
    (0xFE20, 0xFE2F),
];

/// 是否为组合记号（UAX #29 Extend 最小子集）。
pub fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    MARK_RANGES
        .binary_search_by(|&(s, e)| if cp < s { Ordering::Greater } else if cp > e { Ordering::Less } else { Ordering::Equal })
        .is_ok()
}

// ===========================================================================
//  字素簇图（UAX #29 最小子集：Extend + SpacingMark，无 ZWJ 序列）
// ===========================================================================

/// 字素簇类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterKind {
    /// 普通单字符簇。
    Single,
    /// 基字符 + 若干组合记号（如 `a` + U+0301）。
    Marked,
    /// lam-alef 强制连字（两码位合一）。
    LamAlefLigature,
    /// 印度系辅音簇（virama 连接的辅音串，已合并）。
    IndicConjunct,
}

impl ClusterKind {
    /// 判别值（u8，供 canonical 字节用；顺序即声明序，勿重排）。
    fn tag(self) -> u8 {
        match self {
            ClusterKind::Single => 0,
            ClusterKind::Marked => 1,
            ClusterKind::LamAlefLigature => 2,
            ClusterKind::IndicConjunct => 3,
        }
    }
}

/// 一个字素簇：不可再分的排版/上色原子。
///
/// 簇在源文本中的位置以 **char 下标**（非字节下标）记录，与 `layout.rs` 的
/// `GlyphPos::index` 语义一致（`chars().enumerate()` 序）。
#[derive(Debug, Clone, PartialEq)]
pub struct GraphemeCluster {
    /// 源文本中的起始 char 下标（含）。
    pub start: usize,
    /// 源文本中的结束 char 下标（不含）。
    pub end: usize,
    /// 簇的主脚本（由**基字符**决定，记号不改变脚本归属）。
    pub script: Script,
    /// 簇类别。
    pub kind: ClusterKind,
    /// 阿拉伯系上下文形式（非阿拉伯簇恒为 [`ArabicForm::Isolated`]）。
    pub form: ArabicForm,
    /// 双向层级：0 = LTR，1 = RTL（本模块只产出这两层，见模块头诚实边界）。
    pub level: u8,
    /// 视觉序：簇内字形从左到右的下标序列（指向 `ShapedRun.glyphs`；
    /// 前置重排的 i-matra 在此体现）。
    pub visual_glyphs: Vec<usize>,
}

impl GraphemeCluster {
    /// 簇内字符数。
    pub fn char_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// 字素簇图：一段文本的全部簇。
///
/// 簇序恒为**逻辑序**（源文本序），不做视觉反转——视觉序由
/// [`ClusterGraph::visual_cluster_order`] 依双向层级算出。两者分离使排版与
/// 显示可独立校验。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClusterGraph {
    /// 全部簇（逻辑序）。
    pub clusters: Vec<GraphemeCluster>,
}

impl ClusterGraph {
    /// 构造空簇图。
    pub fn new() -> Self {
        Self::default()
    }

    /// 簇总数（字素簇个数，非码位个数）。
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// 是否为空簇图。
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// 取第 `i` 个簇（越界返回 `None`，不 panic——宪章 C18）。
    pub fn cluster(&self, i: usize) -> Option<&GraphemeCluster> {
        self.clusters.get(i)
    }

    /// 视觉簇序（簇下标序列，从左到右的绘制顺序）。
    ///
    /// 实现 UAX #9 的 **L2 规则**（逐层反转）：从文本中的最高层级降到
    /// **最低奇数层级**（含中间未实际出现的层），对每个层级 `l` 反转所有
    /// 「层级 ≥ l」的极大连续段。层级只有 0/1/2 三档（无显式嵌入控制符、
    /// 无隔离符、无括号对——见模块头诚实边界）。
    pub fn visual_cluster_order(&self, base_level: u8) -> Vec<usize> {
        l2_reorder(self.clusters.iter().map(|c| c.level), base_level)
    }
}

/// L2 反转（UAX #9 规则 L2）：从最高层级降至最低奇数层级，逐层反转「层级 ≥ l」
/// 的极大连续段。
///
/// 抽象为自由函数，使 `ClusterGraph` 方法与 `shape` 内部共用**同一实现**
/// ——两处实现不得漂移（`cluster_graph_visual_order_matches_run_visual` 锁定）。
///
/// 最低奇数层级 = 基线层级若为奇数则取之，否则 +1。基线为 LTR(0) 时最低奇数
/// 层级是 1；基线为 RTL(1) 时是 1——后者是纯 RTL 段落能反转的关键：若误用
/// 「层级 > 基线」判据，基线 1 的段落将**永不反转**（RTL 文本会正着排）。
fn l2_reorder<I: Iterator<Item = u8>>(levels: I, base_level: u8) -> Vec<usize> {
    let levels: Vec<u8> = levels.collect();
    let mut order: Vec<usize> = (0..levels.len()).collect();
    if levels.is_empty() {
        return order;
    }
    let max_level = levels.iter().copied().max().unwrap_or(base_level);
    let lowest_odd =
        if base_level % 2 == 1 { base_level } else { base_level.saturating_add(1) };
    // 层级带下界 0：`saturating_add` 可能得 0，用 `max(1)` 保证至少反转一层。
    let lowest_odd = lowest_odd.max(1);

    let mut lvl = max_level;
    while lvl >= lowest_odd {
        let mut i = 0usize;
        while i < levels.len() {
            if levels.get(i).copied().unwrap_or(0) >= lvl {
                let start = i;
                while i < levels.len() && levels.get(i).copied().unwrap_or(0) >= lvl {
                    i += 1;
                }
                order[start..i].reverse();
            } else {
                i += 1;
            }
        }
        if lvl == 0 {
            break;
        }
        lvl -= 1;
    }
    order
}

// ===========================================================================
//  阿拉伯系连接整形
// ===========================================================================

/// 阿拉伯系连接型（ArabicShaping.txt 的 Joining_Type 语义）。
///
/// 术语按**逻辑序**（字符在源串中的先后）定义：
/// - `Right` 只连向**前一个**字符（阿语书写方向下表现为「连右」）；
/// - `Left` 只连向**后一个**字符（罕见，见覆盖说明）；
/// - `Dual` 两侧都连；`NonJoining` 两侧都不连；`Transparent` 不阻断连接链。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArabicJoiningType {
    /// 不参与连接（U+0621 HAMZA、数字、标点、ZWNJ）。
    NonJoining,
    /// 只连向逻辑序的前一字符（ALEF / DAL / REH / WAW 族）。
    Right,
    /// 只连向逻辑序的后一字符。
    Left,
    /// 两侧都连（BEH / SEEN / LAM / MEEM 族）。
    Dual,
    /// 透明：组合记号，不阻断连接链（harakat / hamza / ZWJ）。
    Transparent,
}

/// 手写连接型区间表：`(起始, 结束, 连接型)`，**升序、互不重叠**。
///
/// 覆盖区段见模块头表格。设计要点：U+0621–U+064A 这段里 DAL/REH/WAW/ALEF 是
/// `Right` 而 BEH/SEEN 是 `Dual`，**必须逐段区分**——整段一刀切会让整串字母
/// 全部同形，是最常见的错误实现。
const ARABIC_JOINING_RANGES: &[(u32, u32, ArabicJoiningType)] = &[
    (0x0600, 0x0620, ArabicJoiningType::NonJoining),  // 数字/标点/符号
    (0x0621, 0x0621, ArabicJoiningType::NonJoining),  // HAMZA
    (0x0622, 0x0623, ArabicJoiningType::Right),       // ALEF WITH MADDA / HAMZA ABOVE
    (0x0624, 0x0624, ArabicJoiningType::Right),       // WAW WITH HAMZA ABOVE
    (0x0625, 0x0625, ArabicJoiningType::Right),       // ALEF WITH HAMZA BELOW
    (0x0626, 0x0626, ArabicJoiningType::Dual),        // YEH WITH HAMZA ABOVE
    (0x0627, 0x0627, ArabicJoiningType::Right),       // ALEF
    (0x0628, 0x0628, ArabicJoiningType::Dual),        // BEH
    (0x0629, 0x0629, ArabicJoiningType::Right),       // TEH MARBUTA
    (0x062A, 0x062E, ArabicJoiningType::Dual),        // TEH THEH JEEM HAH KHAH
    (0x062F, 0x0632, ArabicJoiningType::Right),       // DAL THAL REH ZAIN
    (0x0633, 0x063F, ArabicJoiningType::Dual),        // SEEN..GHAIN
    (0x0640, 0x0640, ArabicJoiningType::Dual),        // TATWEEL（拉长连接符）
    (0x0641, 0x0647, ArabicJoiningType::Dual),        // FEH QAF KAF LAM MEEM NOON HEH
    (0x0648, 0x0648, ArabicJoiningType::Right),       // WAW
    (0x0649, 0x0649, ArabicJoiningType::Right),       // ALEF MAKSURA
    (0x064A, 0x064A, ArabicJoiningType::Dual),        // YEH
    (0x064B, 0x065F, ArabicJoiningType::Transparent), // 全部 harakat
    (0x0660, 0x066D, ArabicJoiningType::NonJoining),  // 阿拉伯数字与符号
    (0x066E, 0x066F, ArabicJoiningType::Dual),        // DOTLESS BEH / YEH
    (0x0670, 0x0670, ArabicJoiningType::Transparent), // SUPERSCRIPT ALEF
    (0x0671, 0x0673, ArabicJoiningType::Right),       // ALEF WASLA / WAVY HAMZA
    (0x0674, 0x0687, ArabicJoiningType::NonJoining),  // 高位哈姆扎等罕用字母
    (0x0688, 0x0699, ArabicJoiningType::NonJoining),
    (0x069A, 0x06BF, ArabicJoiningType::Dual),        // 波斯/库尔德常用双连字母
    (0x06C0, 0x06C0, ArabicJoiningType::Dual),        // HEH WITH YEH ABOVE
    (0x06C1, 0x06CB, ArabicJoiningType::Dual),        // HEH GOAL / LAM WITH BAR
    (0x06CC, 0x06CC, ArabicJoiningType::Dual),        // FARSI YEH
    (0x06CD, 0x06D2, ArabicJoiningType::Dual),        // YEH WITH TAIL / BARREE
    (0x06D3, 0x06D3, ArabicJoiningType::Right),       // YEH BARREE 尾形
    (0x06D4, 0x06DD, ArabicJoiningType::NonJoining),  // 句点 / End of Ayah
    (0x06DE, 0x06FF, ArabicJoiningType::NonJoining),  // 符号与未覆盖字母
    (0x0750, 0x077F, ArabicJoiningType::NonJoining),  // 增补（未整形，见覆盖表）
    (0x08A0, 0x08FF, ArabicJoiningType::NonJoining),  // 扩展-A（未整形）
    (0x200C, 0x200C, ArabicJoiningType::NonJoining),  // ZWNJ 显式断链
    (0x200D, 0x200D, ArabicJoiningType::Transparent), // ZWJ 保持链
    (0xFB50, 0xFDFF, ArabicJoiningType::NonJoining),  // presentation forms-A
    (0xFE70, 0xFEFF, ArabicJoiningType::NonJoining),  // presentation forms-B
];

/// 字符 → 阿拉伯连接型；非阿拉伯码位返回 [`ArabicJoiningType::NonJoining`]。
pub fn arabic_joining_type(ch: char) -> ArabicJoiningType {
    lookup_range(ARABIC_JOINING_RANGES, ch as u32).unwrap_or(ArabicJoiningType::NonJoining)
}

/// 阿拉伯系上下文形式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArabicForm {
    /// 孤立形（两侧都不连）：`ا`。
    Isolated,
    /// 词首形（只连向逻辑序后一字符）：`بـ`。
    Initial,
    /// 词中形（两侧都连）：`ـبـ`。
    Medial,
    /// 词尾形（只连向逻辑序前一字符）：`ـب`。
    Final,
}

impl ArabicForm {
    /// 形式判别值（u8，供 canonical 字节与测试用）。
    pub fn tag(self) -> u8 {
        match self {
            ArabicForm::Isolated => 0,
            ArabicForm::Initial => 1,
            ArabicForm::Medial => 2,
            ArabicForm::Final => 3,
        }
    }
}

/// lam-alef 连字的 alef 侧码位集合（LAM 必须紧邻其中之一）。
///
/// 覆盖：U+0622/0623/0625/0627（带各种 hamza 的 ALEF）+ U+0671（ALEF WASLA）
/// + U+0672/0673（wavy hamza）。**不含** U+0624（WAW WITH HAMZA，非 alef）。
const LAM_ALEF_SECOND: &[char] = &[
    '\u{0622}', '\u{0623}', '\u{0625}', '\u{0627}', '\u{0671}', '\u{0672}', '\u{0673}',
];

/// 是否为 lam-alef 强制连字对（`first == LAM && second ∈ LAM_ALEF_SECOND`）。
fn is_lam_alef_pair(first: char, second: char) -> bool {
    first == '\u{0644}' && LAM_ALEF_SECOND.contains(&second)
}

/// lam-alef 连字簇的呈现形式码位（U+FEFB 孤立 / U+FEFC 词尾）。
///
/// **词中型复用词尾形**（U+FEFC）——阿拉伯字体中 lam-alef 只有 isolated/final
/// 两个常用变体（medial 与 final 同形）。这是标准实践，非简化。
fn lam_alef_presentation(is_final: bool) -> char {
    if is_final {
        '\u{FEFC}'
    } else {
        '\u{FEFB}'
    }
}

// ===========================================================================
//  印度系（天城文）整形
// ===========================================================================

/// 天城文类别（U+0900–U+097F 的最小子集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevanagariCategory {
    /// 辅音（CONSONANT，含 QA..YYA 扩展）。
    Consonant,
    /// 独立元音（INDEPENDENT VOWEL，含 vocalic R/L）。
    IndependentVowel,
    /// 依附元音符号（MATRA，含 nukta 归入此类的等价处理见下）。
    Matra,
    /// 虚音符（VIRAMA，U+094D）——连接辅音成 conjunct。
    Virama,
    /// 音点（ANUSVARA / VISARGA / CANDRABINDU / OM）。
    Sign,
    /// 其它（标点、数字、符号）。
    Other,
}

/// 天城文分类（码位 → 类别）；区段外 → [`DevanagariCategory::Other`]。
///
/// 覆盖：U+0900–U+0903 sign、U+0904–U+0914 独立元音、U+0915–U+0939 辅音、
/// U+093A–U+0940 matra（U+093C nukta 在此——天城文里 nukta 依附辅音，
/// 整形上与 matra 同位置处理）、U+0941–U+094C matra、U+094D virama、
/// U+094E–U+094F matra、U+0950–U+0957 sign、U+0958–U+095F 扩展辅音、
/// U+0960–U+0961 vocalic R/L、U+0962–U+0963 matra、U+0964–U+0970 other、
/// U+0971–U+097F 扩展辅音。
fn devanagari_category(cp: u32) -> DevanagariCategory {
    match cp {
        0x0900..=0x0903 => DevanagariCategory::Sign,
        0x0904..=0x0914 => DevanagariCategory::IndependentVowel,
        0x0915..=0x0939 => DevanagariCategory::Consonant,
        0x093A..=0x0940 => DevanagariCategory::Matra,
        0x0941..=0x094C => DevanagariCategory::Matra,
        0x094D => DevanagariCategory::Virama,
        0x094E..=0x094F => DevanagariCategory::Matra,
        0x0950..=0x0957 => DevanagariCategory::Sign,
        0x0958..=0x095F => DevanagariCategory::Consonant,
        0x0960..=0x0961 => DevanagariCategory::IndependentVowel,
        0x0962..=0x0963 => DevanagariCategory::Matra,
        0x0964..=0x0970 => DevanagariCategory::Other,
        0x0971..=0x097F => DevanagariCategory::Consonant,
        _ => DevanagariCategory::Other,
    }
}

/// 天城文前置元音符号（`i-matra` U+093F）。
///
/// 前置重排是印度系整形的核心规则：书写时该符号画在**辅音左边**，但码位
/// 却在辅音**之后**（`क` + `ि` 存为 U+0915 U+093F，视觉须为 `ि` `क`）。
/// **仅 U+093F 按前置处理**；U+094E PRISHTHAMATRA 在 Unicode 定位类别上亦为
/// `Pre`，但天城文实际书写在基字下方偏右，本实现按后置处理——**这是已知简化**
/// （`devanagari_prishthamatra_treated_as_post_base` 锁定当前行为）。
const DEVANAGARI_PRE_BASE_MATRA: char = '\u{093F}';

/// 字符是否为天城文辅音。
fn is_devanagari_consonant(ch: char) -> bool {
    devanagari_category(ch as u32) == DevanagariCategory::Consonant
}

// ===========================================================================
//  整形参数与产物
// ===========================================================================

/// 双向基线方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaseDirection {
    /// 自动：由首个强方向字符决定（UAX #9 P2/P3 的简化：无隔离符与显式控制符）。
    #[default]
    Auto,
    /// 强制从左到右。
    Ltr,
    /// 强制从右到左。
    Rtl,
}

impl BaseDirection {
    /// 判别值（u8，供 canonical 字节用；顺序即声明序，勿重排）。
    fn tag(self) -> u8 {
        match self {
            BaseDirection::Auto => 0,
            BaseDirection::Ltr => 1,
            BaseDirection::Rtl => 2,
        }
    }
}

/// 整形参数。
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeParams {
    /// 字号（px，em 框高）；须为有限正数。
    pub font_size: f32,
    /// 字间追踪（px，逐字形追加）；须为有限数。
    pub letter_spacing: f32,
    /// 基线方向。
    pub base_direction: BaseDirection,
}

impl Default for ShapeParams {
    fn default() -> Self {
        ShapeParams { font_size: 24.0, letter_spacing: 0.0, base_direction: BaseDirection::Auto }
    }
}

/// 单个整形后字形。
#[derive(Debug, Clone, PartialEq)]
pub struct ShapedGlyph {
    /// 所属字素簇下标（指向 [`ShapedRun::graph`]）。
    pub cluster: usize,
    /// 该字形代表的源字符（连字取呈现形式码位）。
    pub ch: char,
    /// 字形 id（cmap 未命中 → 0 = .notdef）。
    pub gid: u16,
    /// 阿拉伯上下文形式（非阿拉伯恒 `Isolated`）。
    pub form: ArabicForm,
    /// 是否为零宽依附记号（true → 不占步进）。
    pub is_mark: bool,
    /// 笔位 x（px，画布坐标，y 向下；基线 y 由 `layout.rs` 定）。
    pub x: f32,
    /// 步进宽（px，含 letter_spacing）。
    pub advance: f32,
}

/// 整形结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapedRun {
    /// 字素簇图（逻辑序）。
    pub graph: ClusterGraph,
    /// 全部字形（**逻辑序**：按簇下标升序，簇内已按视觉序排列）。
    pub glyphs: Vec<ShapedGlyph>,
    /// 绘制序：`glyphs` 下标序列，从左到右。
    pub visual: Vec<usize>,
    /// 基线方向（已从 `Auto` 解析为具体方向）。
    pub base_direction: BaseDirection,
    /// 基线双向层级（0 = LTR，1 = RTL）。
    pub base_level: u8,
    /// 整行宽（px，视觉序上最右字形右缘）。
    pub width: f32,
    /// 未能从字体 cmap 解析到字形的**基字符**数（诚实报告，不静默）。
    ///
    /// 内嵌 Prism Sans SC 子集不含阿拉伯/天城文字形，故阿拉伯/天城文整形测试中
    /// 该值 = 基字符数——这是**预期**结果，不是缺陷。
    pub missing_glyphs: usize,
}

impl ShapedRun {
    /// 字素簇数。
    pub fn cluster_count(&self) -> usize {
        self.graph.cluster_count()
    }

    /// 确定性字节序列：整形计划的规范编码（确定性契约的验收口径）。
    ///
    /// 布局：`簇数(u32 BE) → 各簇(起 u32, 止 u32, script u8, kind u8, form u8,
    /// level u8, 视觉字形数 u32, 各下标 u32) → 字形数(u32 BE) → 各字形
    /// (簇 u32, 码位 u32, gid u16, form u8, mark u8, x 位模式 u32,
    /// advance 位模式 u32) → 绘制序长度 u32 → 各下标 u32 → 基线方向 u8 →
    /// 基线层级 u8 → 宽位模式 u32 → 缺字形数 u32`。
    ///
    /// 全程大端定长/显式长度前缀，浮点用 `to_bits()`——逐字节可复现。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.cluster_count() as u32).to_be_bytes());
        for c in &self.graph.clusters {
            out.extend_from_slice(&(c.start as u32).to_be_bytes());
            out.extend_from_slice(&(c.end as u32).to_be_bytes());
            out.push(c.script as u8);
            out.push(c.kind.tag());
            out.push(c.form.tag());
            out.push(c.level);
            out.extend_from_slice(&(c.visual_glyphs.len() as u32).to_be_bytes());
            for &g in &c.visual_glyphs {
                out.extend_from_slice(&(g as u32).to_be_bytes());
            }
        }
        out.extend_from_slice(&(self.glyphs.len() as u32).to_be_bytes());
        for g in &self.glyphs {
            out.extend_from_slice(&(g.cluster as u32).to_be_bytes());
            out.extend_from_slice(&(g.ch as u32).to_be_bytes());
            out.extend_from_slice(&g.gid.to_be_bytes());
            out.push(g.form.tag());
            out.push(u8::from(g.is_mark));
            out.extend_from_slice(&g.x.to_bits().to_be_bytes());
            out.extend_from_slice(&g.advance.to_bits().to_be_bytes());
        }
        out.extend_from_slice(&(self.visual.len() as u32).to_be_bytes());
        for &v in &self.visual {
            out.extend_from_slice(&(v as u32).to_be_bytes());
        }
        out.push(self.base_direction.tag());
        out.push(self.base_level);
        out.extend_from_slice(&self.width.to_bits().to_be_bytes());
        out.extend_from_slice(&(self.missing_glyphs as u32).to_be_bytes());
        out
    }
}

// ===========================================================================
//  整形主流程
// ===========================================================================

/// 段落基线方向解析：取首个强方向字符（UAX #9 P2/P3 的简化实现）。
///
/// `Common` / `Inherited` / 印度系 / `Unknown` **不决定**基线（跳过）——
/// 正确对应 P2「跳过中性字符」的精神（此处未实现中性括号配对 BD16）。
fn resolve_base_level(chars: &[char], requested: BaseDirection) -> (BaseDirection, u8) {
    let resolved = match requested {
        BaseDirection::Ltr => BaseDirection::Ltr,
        BaseDirection::Rtl => BaseDirection::Rtl,
        BaseDirection::Auto => {
            let mut found = BaseDirection::Ltr;
            for &ch in chars {
                let s = script_of(ch);
                if s.is_rtl() {
                    found = BaseDirection::Rtl;
                    break;
                }
                if matches!(s, Script::Latin | Script::Greek | Script::Cyrillic | Script::Han | Script::Kana) {
                    found = BaseDirection::Ltr;
                    break;
                }
            }
            found
        }
    };
    let level = match resolved {
        BaseDirection::Rtl => 1u8,
        _ => 0u8,
    };
    (resolved, level)
}

/// 校验整形参数（宪章 C18：非法参数结构化报错，**不静默 clamp**——C20 白名单外）。
fn validate_params(params: &ShapeParams) -> Result<(), TextError> {
    if !params.font_size.is_finite() || params.font_size <= 0.0 {
        return Err(param_error(
            "shape(font_size)",
            format!("字号非法：{}（须为有限正数）", params.font_size),
            "改用大于 0 的有限字号，如 24.0；不要传 0 / 负数 / NaN / inf",
        ));
    }
    if !params.letter_spacing.is_finite() {
        return Err(param_error(
            "shape(letter_spacing)",
            format!("字距非法：{}（须为有限数）", params.letter_spacing),
            "改用有限数，如 0.0；不要传 NaN / inf",
        ));
    }
    Ok(())
}

/// 切分字素簇（UAX #29 最小子集 + lam-alef 强制连字 + 印度辅音簇）。
///
/// 状态机（自左向右单遍，簇边界条件互斥且穷尽）：
/// 1. 孤立组合记号（无基字符可依附）→ 自立单字符簇（**不静默丢弃**）；
/// 2. LAM 且下一字符 ∈ `LAM_ALEF_SECOND` → 强制连字簇（两码位），跳过 alef；
/// 3. 天城文辅音 → 吞后续 matra/virama/sign；若簇尾为 virama 且下一字符是辅音，
///    续吞（conjunct 链）；
/// 4. 其余基字符 → 吸收后续组合记号。
fn build_clusters(chars: &[char]) -> Vec<GraphemeCluster> {
    let mut clusters: Vec<GraphemeCluster> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let start = i;
        let ch = chars.get(i).copied().unwrap_or('\u{0}');

        // 规则 1：孤立组合记号——无基字符可依附，自立一簇。
        if is_combining_mark(ch) {
            clusters.push(GraphemeCluster {
                start,
                end: start + 1,
                script: Script::Inherited,
                kind: ClusterKind::Single,
                form: ArabicForm::Isolated,
                level: 0,
                visual_glyphs: Vec::new(),
            });
            i += 1;
            continue;
        }

        // 规则 2：lam-alef 强制连字。
        if let Some(&next) = chars.get(i + 1) {
            if is_lam_alef_pair(ch, next) {
                clusters.push(GraphemeCluster {
                    start,
                    end: start + 2,
                    script: Script::Arabic,
                    kind: ClusterKind::LamAlefLigature,
                    form: ArabicForm::Isolated, // 稍后由连接状态机覆写
                    level: 0,
                    visual_glyphs: Vec::new(),
                });
                i += 2;
                continue;
            }
        }

        // 规则 3：天城文辅音簇（virama 链式合并）。
        if script_of(ch) == Script::Devanagari && is_devanagari_consonant(ch) {
            let mut end = start + 1;
            loop {
                // 吞掉当前辅音后的全部依附记号（matra / virama / sign）。
                while let Some(&nx) = chars.get(end) {
                    let cat = devanagari_category(nx as u32);
                    if cat == DevanagariCategory::Matra
                        || cat == DevanagariCategory::Virama
                        || cat == DevanagariCategory::Sign
                    {
                        end += 1;
                    } else {
                        break;
                    }
                }
                // 簇尾是 virama 且下一字符是辅音 → 续吞（conjunct 链）。
                let tail_is_virama = chars
                    .get(end.saturating_sub(1))
                    .map(|&t| devanagari_category(t as u32) == DevanagariCategory::Virama)
                    .unwrap_or(false);
                match chars.get(end) {
                    Some(&nx) if tail_is_virama && is_devanagari_consonant(nx) => end += 1,
                    _ => break,
                }
            }
            let kind = if end == start + 1 { ClusterKind::Single } else { ClusterKind::IndicConjunct };
            clusters.push(GraphemeCluster {
                start,
                end,
                script: Script::Devanagari,
                kind,
                form: ArabicForm::Isolated,
                level: 0,
                visual_glyphs: Vec::new(),
            });
            i = end;
            continue;
        }

        // 规则 4：任意脚本基字符吸收后续组合记号。
        let mut end = start + 1;
        while let Some(&nx) = chars.get(end) {
            if is_combining_mark(nx) {
                end += 1;
            } else {
                break;
            }
        }
        let kind = if end > start + 1 { ClusterKind::Marked } else { ClusterKind::Single };
        clusters.push(GraphemeCluster {
            start,
            end,
            script: script_of(ch),
            kind,
            form: ArabicForm::Isolated,
            level: 0,
            visual_glyphs: Vec::new(),
        });
        i = end;
    }
    clusters
}

/// 阿拉伯连接状态机：为每个**簇**求上下文形式。
///
/// 状态 = (当前簇连接型, 前一非透明簇连接型, 后一非透明簇连接型)。
/// 透明簇（纯 harakat）被跳过，不参与连接链——这是「harakat 不阻断连接」的实现点。
///
/// 形式判定（显式四分支，无 fall-through 猜测）：
/// | 连前 | 连后 | 形式 |
/// |---|---|---|
/// | 是 | 是 | `Medial` |
/// | 是 | 否 | `Final` |
/// | 否 | 是 | `Initial` |
/// | 否 | 否 | `Isolated` |
fn resolve_arabic_forms(clusters: &mut [GraphemeCluster], chars: &[char]) {
    // 每簇连接型（取簇首基字符；连字簇首为 LAM）。
    let joins: Vec<ArabicJoiningType> = clusters
        .iter()
        .map(|c| {
            chars.get(c.start).map(|&ch| arabic_joining_type(ch)).unwrap_or(ArabicJoiningType::NonJoining)
        })
        .collect();
    // 每簇是否**向前断链**：簇末字符是 ZWNJ（U+200C）时，该簇不能与后一簇连。
    // ZWNJ 已被 `is_combining_mark` 并入前一簇（UAX #29 属 Extend），故只看
    // 簇首会漏掉它——这是「显式断链」的实现点，必须查簇**末**。
    let blocks_forward: Vec<bool> = clusters
        .iter()
        .map(|c| {
            c.end
                .checked_sub(1)
                .and_then(|last| chars.get(last))
                .map(|&ch| ch == '\u{200C}')
                .unwrap_or(false)
        })
        .collect();

    for idx in 0..clusters.len() {
        let my = joins.get(idx).copied().unwrap_or(ArabicJoiningType::NonJoining);
        // 透明簇不产生字形，形式无意义。
        if my == ArabicJoiningType::Transparent {
            continue;
        }
        // 前一非透明簇。
        let mut prev = ArabicJoiningType::NonJoining;
        let mut p = idx;
        while p > 0 {
            p -= 1;
            let jt = joins.get(p).copied().unwrap_or(ArabicJoiningType::NonJoining);
            if jt != ArabicJoiningType::Transparent {
                // 前一簇以 ZWNJ 结尾 → 视为不连接（显式断链）。
                prev =
                    if blocks_forward.get(p).copied().unwrap_or(false) { ArabicJoiningType::NonJoining } else { jt };
                break;
            }
        }
        // 后一非透明簇。
        let mut next = ArabicJoiningType::NonJoining;
        for q in idx + 1..clusters.len() {
            let jt = joins.get(q).copied().unwrap_or(ArabicJoiningType::NonJoining);
            if jt != ArabicJoiningType::Transparent {
                next = jt;
                break;
            }
        }

        // 连前：我的型可连前（Dual/Right）且前一簇可连后（Dual/Left）。
        let can_prev = matches!(my, ArabicJoiningType::Dual | ArabicJoiningType::Right)
            && matches!(prev, ArabicJoiningType::Dual | ArabicJoiningType::Left);
        // 连后：我的型可连后（Dual/Left）且后一簇可连前（Dual/Right）。
        // 本簇以 ZWNJ 结尾时**不得**向后连（ZWNJ 恰位于此连接点）。
        let can_next = !blocks_forward.get(idx).copied().unwrap_or(false)
            && matches!(my, ArabicJoiningType::Dual | ArabicJoiningType::Left)
            && matches!(next, ArabicJoiningType::Dual | ArabicJoiningType::Right);

        let form = match (can_prev, can_next) {
            (true, true) => ArabicForm::Medial,
            (true, false) => ArabicForm::Final,
            (false, true) => ArabicForm::Initial,
            (false, false) => ArabicForm::Isolated,
        };

        // 仅阿拉伯簇写回形式；其它脚本保持 Isolated。
        if let Some(c) = clusters.get_mut(idx) {
            if c.script == Script::Arabic {
                c.form = form;
            }
        }
    }
}

/// 整形主入口：文本 → 定位字形流。
///
/// 管线（每步皆纯函数，遍历序由下标决定）：
/// 1. 参数校验（C18 结构化报错）；
/// 2. `chars` 收集 + 基线方向解析（P2/P3 简化）；
/// 3. 字素簇切分（UAX #29 最小子集 + lam-alef + 印度 conjunct）；
/// 4. 双向层级指派（level 0/1）；
/// 5. 阿拉伯连接状态机（上下文形式）；
/// 6. 簇内字形展开（前置重排 / 连字 / 记号零宽）；
/// 7. L2 视觉序 + 笔位前缀和累加。
///
/// # 参数
/// * `font` — 字体（仅用于 cmap 查字形与步进宽；**不解析 GSUB/GPOS**）。
/// * `text` — 源文本。
/// * `params` — 整形参数。
///
/// # 错误
/// 参数非法返回 [`TextError`]（P1101）。**空文本**返回空 `ShapedRun`（零错误）。
pub fn shape(font: &Font, text: &str, params: &ShapeParams) -> Result<ShapedRun, TextError> {
    validate_params(params)?;

    let chars: Vec<char> = text.chars().collect();
    let (base_direction, base_level) = resolve_base_level(&chars, params.base_direction);
    let mut graph = ClusterGraph::new();

    if chars.is_empty() {
        return Ok(ShapedRun { graph, base_direction, base_level, ..ShapedRun::default() });
    }

    let mut clusters = build_clusters(&chars);

    // 双向层级指派（UAX #9 规则 I1/I2 的两档简化）：
    // - RTL 脚本簇 → 奇数层级 1；
    // - LTR 强方向簇 → 偶数层级 2（当基线为 RTL 时，嵌入的 LTR 段须抬到 2，
    //   否则它与基线同级会被整段一起反转，视觉错乱）；
    // - 中性/继承簇 → 跟随基线层级。
    // 无显式嵌入控制符 / 隔离符 / 括号对（BD16），见模块头诚实边界。
    for c in clusters.iter_mut() {
        c.level = if c.script.is_rtl() {
            1
        } else if c.script.is_strong() && base_level == 1 {
            2
        } else {
            base_level
        };
    }

    resolve_arabic_forms(&mut clusters, &chars);

    // ---- 簇内字形展开（逻辑序下标分配）----
    let upem = font.units_per_em().max(1) as f32;
    let scale = params.font_size / upem;
    let mut glyphs: Vec<ShapedGlyph> = Vec::new();
    let mut missing_glyphs = 0usize;

    for (ci, c) in clusters.iter_mut().enumerate() {
        let local = expand_cluster(c, &chars);
        let mut visual: Vec<usize> = Vec::with_capacity(local.len());
        for (ch, is_mark) in local {
            let gid = font.glyph_index(ch).unwrap_or(0);
            // 缺字形只统计**基字符**（记号缺字形是常态，不重复报警）。
            if gid == 0 && !is_mark {
                missing_glyphs += 1;
            }
            let advance =
                if is_mark { 0.0 } else { font.advance(gid) as f32 * scale + params.letter_spacing };
            let gidx = glyphs.len();
            glyphs.push(ShapedGlyph {
                cluster: ci,
                ch,
                gid,
                form: if c.script == Script::Arabic { c.form } else { ArabicForm::Isolated },
                is_mark,
                x: 0.0,
                advance,
            });
            visual.push(gidx);
        }
        c.visual_glyphs = visual;
    }

    // ---- 视觉序与笔位累加（L2 反转 + 从左到右前缀和）----
    let visual_clusters = l2_reorder(clusters.iter().map(|c| c.level), base_level);
    let mut visual: Vec<usize> = Vec::with_capacity(glyphs.len());
    for ci in visual_clusters {
        if let Some(c) = clusters.get(ci) {
            visual.extend_from_slice(&c.visual_glyphs);
        }
    }

    let mut pen = 0.0f32;
    for &gi in &visual {
        if let Some(g) = glyphs.get_mut(gi) {
            g.x = pen;
            pen += g.advance;
        }
    }

    graph.clusters = clusters;
    Ok(ShapedRun { graph, glyphs, visual, base_direction, base_level, width: pen, missing_glyphs })
}

/// 簇内字形展开：把一个簇拆成 `(码位, 是否记号)` 序列，**已按视觉序排列**。
///
/// 规则：
/// - lam-alef 连字 → 单个字形，取呈现形式码位（孤立 U+FEFB / 词尾 U+FEFC）；
/// - 印度辅音簇 → **前置元音符号先出**（i-matra 前置重排），再辅音（逻辑序），
///   其余记号最后；virama 被**吸收**（不产出字形——它是连接件不是字形）；
/// - 其它 → 逻辑序，簇首为基字符、其余为记号。
fn expand_cluster(c: &GraphemeCluster, chars: &[char]) -> Vec<(char, bool)> {
    let mut out: Vec<(char, bool)> = Vec::new();
    match c.kind {
        ClusterKind::LamAlefLigature => {
            out.push((lam_alef_presentation(c.form == ArabicForm::Final), false));
        }
        ClusterKind::IndicConjunct => {
            let (mut pre, mut bases, mut post) = (Vec::new(), Vec::new(), Vec::new());
            for k in c.start..c.end {
                let Some(&ch) = chars.get(k) else { continue };
                match devanagari_category(ch as u32) {
                    DevanagariCategory::Virama => {}
                    DevanagariCategory::Matra => {
                        if ch == DEVANAGARI_PRE_BASE_MATRA {
                            pre.push(ch);
                        } else {
                            post.push(ch);
                        }
                    }
                    DevanagariCategory::Consonant | DevanagariCategory::IndependentVowel => {
                        bases.push(ch)
                    }
                    DevanagariCategory::Sign | DevanagariCategory::Other => post.push(ch),
                }
            }
            for ch in pre {
                out.push((ch, true));
            }
            for ch in bases {
                out.push((ch, false));
            }
            for ch in post {
                out.push((ch, true));
            }
        }
        ClusterKind::Marked | ClusterKind::Single => {
            for k in c.start..c.end {
                let Some(&ch) = chars.get(k) else { continue };
                out.push((ch, k > c.start && is_combining_mark(ch)));
            }
        }
    }
    out
}

/// 统计字素簇数（不需字体——纯文本结构查询）。
///
/// 便捷入口，等价于 [`shape`] 后取 [`ShapedRun::cluster_count`]，但不查 cmap、
/// 不算笔位，故可独立用于断行/光标定位的簇边界计算。
pub fn cluster_count(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    build_clusters(&chars).len()
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    /// 整形参数（默认字号；簇/形式是本模块断言主体，与字号无关）。
    fn params() -> ShapeParams {
        ShapeParams::default()
    }

    /// 整形并取各簇形式。
    fn forms_of(text: &str) -> Vec<ArabicForm> {
        let f = font_regular();
        let run = shape(&f, text, &params()).unwrap_or_else(|e| panic!("整形失败：{e}"));
        run.graph.clusters.iter().map(|c| c.form).collect()
    }

    // ------------------------------------------------------------------
    // 关卡 0：数据表自身的正确性（二分查找的前提）
    // ------------------------------------------------------------------

    #[test]
    fn script_range_table_is_sorted_and_disjoint() {
        let mut prev_end: Option<u32> = None;
        for &(s, e, _) in SCRIPT_RANGES {
            assert!(s <= e, "区间 [{s:#X}, {e:#X}] 起止倒置");
            if let Some(pe) = prev_end {
                assert!(s > pe, "区间 [{s:#X},..] 与前一区间重叠（前一止于 {pe:#X}）");
            }
            prev_end = Some(e);
        }
    }

    #[test]
    fn joining_range_table_is_sorted_and_disjoint() {
        let mut prev_end: Option<u32> = None;
        for &(s, e, _) in ARABIC_JOINING_RANGES {
            assert!(s <= e, "连接表区间 [{s:#X}, {e:#X}] 起止倒置");
            if let Some(pe) = prev_end {
                assert!(s > pe, "连接表区间 [{s:#X},..] 与前一区间重叠");
            }
            prev_end = Some(e);
        }
    }

    #[test]
    fn mark_range_table_is_sorted_and_disjoint() {
        let mut prev_end: Option<u32> = None;
        for &(s, e) in MARK_RANGES {
            assert!(s <= e, "记号表区间 [{s:#X}, {e:#X}] 起止倒置");
            if let Some(pe) = prev_end {
                assert!(s > pe, "记号表区间 [{s:#X},..] 与前一区间重叠");
            }
            prev_end = Some(e);
        }
    }

    // ------------------------------------------------------------------
    // 关卡 1：脚本与连接型分类（手写表，抽查区段边界）
    // ------------------------------------------------------------------

    #[test]
    fn script_classification_covers_expected_ranges() {
        assert_eq!(script_of('A'), Script::Latin);
        assert_eq!(script_of(' '), Script::Common);
        assert_eq!(script_of('7'), Script::Common);
        assert_eq!(script_of('\u{0301}'), Script::Inherited);
        // 阿拉伯系（含增补、扩展-A、presentation forms——已分类）。
        assert_eq!(script_of('\u{0628}'), Script::Arabic); // BEH
        assert_eq!(script_of('\u{06CC}'), Script::Arabic); // FARSI YEH
        assert_eq!(script_of('\u{0750}'), Script::Arabic); // 增补
        assert_eq!(script_of('\u{08A0}'), Script::Arabic); // 扩展-A
        assert_eq!(script_of('\u{FEFB}'), Script::Arabic); // lam-alef 孤立形
        assert_eq!(script_of('\u{05D0}'), Script::Hebrew);
        assert_eq!(script_of('\u{0915}'), Script::Devanagari);
        assert_eq!(script_of('\u{09A8}'), Script::IndicOther); // 孟加拉 NA
        assert_eq!(script_of('\u{4E2D}'), Script::Han);
        assert_eq!(script_of('\u{3042}'), Script::Kana);
        // 未覆盖区段 → Unknown（不猜）。
        assert_eq!(script_of('\u{0531}'), Script::Unknown); // 亚美尼亚
        assert_eq!(script_of('\u{0E01}'), Script::Unknown); // 泰
        assert_eq!(script_of('\u{1F600}'), Script::Unknown); // emoji
    }

    #[test]
    fn joining_types_match_arabic_letter_classes() {
        // 双连字母。
        for ch in ['\u{0628}', '\u{062A}', '\u{0633}', '\u{0644}', '\u{0645}', '\u{064A}'] {
            assert_eq!(arabic_joining_type(ch), ArabicJoiningType::Dual, "{:#X} 应为双连", ch as u32);
        }
        // 只连前（ALEF / DAL / REH / WAW 族）。
        for ch in ['\u{0627}', '\u{0622}', '\u{062F}', '\u{0631}', '\u{0648}', '\u{0649}'] {
            assert_eq!(arabic_joining_type(ch), ArabicJoiningType::Right, "{:#X} 应为只连前", ch as u32);
        }
        // 不连接：hamza / 数字 / 标点。
        for ch in ['\u{0621}', '\u{0660}', '\u{060C}'] {
            assert_eq!(
                arabic_joining_type(ch),
                ArabicJoiningType::NonJoining,
                "{:#X} 应不连接",
                ch as u32
            );
        }
        // 透明：harakat / superscript alef / ZWJ 不阻断连接链。
        for ch in ['\u{064E}', '\u{0650}', '\u{0652}', '\u{0670}', '\u{200D}'] {
            assert_eq!(
                arabic_joining_type(ch),
                ArabicJoiningType::Transparent,
                "{:#X} 应透明",
                ch as u32
            );
        }
        // ZWNJ 显式断链。
        assert_eq!(arabic_joining_type('\u{200C}'), ArabicJoiningType::NonJoining);
    }

    #[test]
    fn script_strong_and_rtl_predicates() {
        assert!(Script::Arabic.is_rtl() && Script::Hebrew.is_rtl());
        assert!(!Script::Latin.is_rtl() && !Script::Han.is_rtl());
        assert!(Script::Latin.is_strong());
        assert!(!Script::Common.is_strong() && !Script::Inherited.is_strong());
    }

    // ------------------------------------------------------------------
    // 关卡 2：阿拉伯四态（Isolated / Initial / Medial / Final）
    // ------------------------------------------------------------------

    #[test]
    fn arabic_single_letter_is_isolated() {
        assert_eq!(forms_of("\u{0628}"), vec![ArabicForm::Isolated], "孤立字母必须是 Isolated");
    }

    #[test]
    fn arabic_initial_and_final_forms() {
        // BEH BEH：首字母连后不连前 → Initial；尾字母连前不连后 → Final。
        assert_eq!(
            forms_of("\u{0628}\u{0628}"),
            vec![ArabicForm::Initial, ArabicForm::Final],
            "بـ + ـب"
        );
    }

    #[test]
    fn arabic_medial_form_in_three_letter_chain() {
        assert_eq!(
            forms_of("\u{0628}\u{0628}\u{0628}"),
            vec![ArabicForm::Initial, ArabicForm::Medial, ArabicForm::Final],
            "三连 BEH 应取 Initial/Medial/Final"
        );
    }

    #[test]
    fn arabic_all_four_states_covered_across_cases() {
        // Isolated：单字母、链尾断链。
        assert_eq!(forms_of("\u{0628}"), vec![ArabicForm::Isolated]);
        // Initial / Medial / Final：三连链。
        assert_eq!(
            forms_of("\u{0628}\u{0628}\u{0628}"),
            vec![ArabicForm::Initial, ArabicForm::Medial, ArabicForm::Final]
        );
        // 四态在**同一串**内共存：插入一个 NonJoining 字符（HAMZA U+0621）
        // 把链切断，其后三连 BEH 取首/中/尾，其前的 BEH 两侧都不连 → 孤立。
        //
        // 关键区分（易错点）：ALEF/DAL/REH 这类 **Right-joining** 字母只能
        // *接受*来自逻辑序前一字符的连接，**不能**主动连向逻辑序后一字符，
        // 所以「ALEF + BEH」里 BEH 仍取 Initial——断链必须用 NonJoining。
        let got = forms_of("\u{0628}\u{0621}\u{0628}\u{0628}\u{0628}");
        assert_eq!(
            got,
            vec![
                ArabicForm::Isolated, // BEH#1：右侧是 NonJoining，左侧无字 → 孤立
                ArabicForm::Isolated, // HAMZA 自身不连接
                ArabicForm::Initial,  // BEH#2
                ArabicForm::Medial,   // BEH#3
                ArabicForm::Final,    // BEH#4
            ],
            "BEH + HAMZA + 三连 BEH：四态在同一串内全覆盖"
        );
        // 对照一：NUN（U+0646）与 BEH 同为 Dual-joining，不停链，整串一条五连链。
        assert_eq!(
            forms_of("\u{0646}\u{0628}\u{0628}\u{0628}\u{0628}"),
            vec![
                ArabicForm::Initial,
                ArabicForm::Medial,
                ArabicForm::Medial,
                ArabicForm::Medial,
                ArabicForm::Final,
            ],
            "NUN + 四连 BEH：Dual 不断链"
        );
        // 对照二：Right-joining 的 ALEF 只能接受左连接，不能断后链。
        assert_eq!(
            forms_of("\u{0627}\u{0628}\u{0628}\u{0628}"),
            vec![
                ArabicForm::Isolated, // ALEF 左侧无字
                ArabicForm::Initial,  // BEH#1：可连向右邻的 BEH
                ArabicForm::Medial,
                ArabicForm::Final,
            ],
            "ALEF + 三连 BEH：ALEF 不阻断 BEH 之间的连接"
        );
    }

    #[test]
    fn arabic_non_joining_letter_breaks_the_chain() {
        // BEH + DAL + BEH。DAL 是「只连前」：它与**前**一簇 BEH 连（故该 BEH
        // 取 Initial），但**不**与后一簇 BEH 连（故后一簇 BEH 退化为 Isolated）。
        // DAL 自身连前不连后 → Final。
        assert_eq!(
            forms_of("\u{0628}\u{062F}\u{0628}"),
            vec![ArabicForm::Initial, ArabicForm::Final, ArabicForm::Isolated],
            "DAL 断链：首 BEH 连 DAL 取 Initial，DAL 取 Final，尾 BEH 无连取 Isolated"
        );
    }

    #[test]
    fn arabic_transparent_marks_do_not_break_joining() {
        // BEH + fatha + BEH + fatha + BEH：harakat 透明 → 链不断。
        let got = forms_of("\u{0628}\u{064E}\u{0628}\u{064E}\u{0628}");
        assert_eq!(
            got,
            vec![ArabicForm::Initial, ArabicForm::Medial, ArabicForm::Final],
            "harakat 不得并入簇、也不得阻断连接"
        );
        assert_eq!(cluster_count("\u{0628}\u{064E}\u{0628}"), 2, "harakat 不应自成簇");
    }

    #[test]
    fn arabic_zwnj_breaks_joining_explicitly() {
        // BEH + ZWNJ + BEH。ZWNJ 属 Extend（UAX #29）故并入前一簇，但其
        // 连接型是 NonJoining 且显式断链 → 首 BEH 不向后连（Isolated），
        // 尾 BEH 不向前连（Isolated）。
        assert_eq!(
            forms_of("\u{0628}\u{200C}\u{0628}"),
            vec![ArabicForm::Isolated, ArabicForm::Isolated],
            "ZWNJ 必须显式断链"
        );
        // 断链后 2 簇（ZWNJ 并入首簇，不是独立簇）。
        assert_eq!(cluster_count("\u{0628}\u{200C}\u{0628}"), 2, "ZWNJ 并入首簇");
    }

    #[test]
    fn arabic_tatweel_keeps_chain_alive() {
        // BEH + TATWEEL + BEH。TATWEEL 是双连连接件，自身成簇取 Medial，
        // 两侧 BEH 仍连 → Initial / Final。链没断。
        assert_eq!(
            forms_of("\u{0628}\u{0640}\u{0628}"),
            vec![ArabicForm::Initial, ArabicForm::Medial, ArabicForm::Final],
            "TATWEEL 应保持连接链（自身取 Medial）"
        );
    }

    #[test]
    fn arabic_alef_ends_chain_as_right_joining() {
        // BEH + ALEF：ALEF 只连前 → 它与 BEH 连（BEH 取 Initial），ALEF 取 Final。
        assert_eq!(forms_of("\u{0628}\u{0627}"), vec![ArabicForm::Initial, ArabicForm::Final]);
    }

    // ------------------------------------------------------------------
    // 关卡 3：lam-alef 连字
    // ------------------------------------------------------------------

    #[test]
    fn lam_alef_forms_mandatory_ligature() {
        assert_eq!(cluster_count("\u{0644}\u{0627}"), 1, "lam-alef 必须合一簇");
        let f = font_regular();
        let run = shape(&f, "\u{0644}\u{0627}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.cluster_count(), 1);
        assert_eq!(run.graph.clusters[0].kind, ClusterKind::LamAlefLigature);
        // 前有 BEH 时 LAM 连前 → 连字取词尾形。
        let run2 = shape(&f, "\u{0628}\u{0644}\u{0627}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run2.cluster_count(), 2, "BEH + lam-alef = 2 簇");
        assert_eq!(run2.graph.clusters[1].form, ArabicForm::Final, "连前时取词尾形");
    }

    #[test]
    fn lam_alef_ligature_uses_presentation_form_codepoint() {
        let f = font_regular();
        let iso = shape(&f, "\u{0644}\u{0627}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(iso.glyphs[0].ch, '\u{FEFB}', "孤立 lam-alef 用 U+FEFB");
        let fin =
            shape(&f, "\u{0628}\u{0644}\u{0627}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fin.glyphs[1].ch, '\u{FEFC}', "词尾 lam-alef 用 U+FEFC");
    }

    #[test]
    fn lam_alef_variants_all_supported() {
        for alef in ['\u{0622}', '\u{0623}', '\u{0625}', '\u{0627}', '\u{0671}'] {
            let text = format!("\u{0644}{alef}");
            assert_eq!(cluster_count(&text), 1, "LAM + {:#X} 应成连字", alef as u32);
        }
        // 非 alef 的 WAW+HAMZA 不得误判为连字。
        assert_eq!(cluster_count("\u{0644}\u{0624}"), 2, "LAM+WAW 不得合簇");
        // 记号夹在中间不阻断连字判定（LAM 后紧邻须是 alef）。
        assert_eq!(cluster_count("\u{0644}\u{064E}\u{0627}"), 2, "中间有记号则不成连字");
    }

    // ------------------------------------------------------------------
    // 关卡 4：天城文（i-matra 前置 + 辅音簇合并）
    // ------------------------------------------------------------------

    #[test]
    fn devanagari_i_matra_reorders_before_base() {
        // KA + i-matra：视觉上 i-matra 必须在 KA 左边。
        let f = font_regular();
        let run = shape(&f, "\u{0915}\u{093F}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.cluster_count(), 1, "辅音 + 元音符号 = 1 簇");
        let c = &run.graph.clusters[0];
        assert_eq!(c.kind, ClusterKind::IndicConjunct);
        let g0 = run.glyphs.get(c.visual_glyphs[0]).unwrap_or_else(|| panic!("无第 0 视觉字形"));
        let g1 = run.glyphs.get(c.visual_glyphs[1]).unwrap_or_else(|| panic!("无第 1 视觉字形"));
        assert_eq!(g0.ch, '\u{093F}', "i-matra 必须前置（视觉在左）");
        assert_eq!(g1.ch, '\u{0915}', "基辅音在 i-matra 之后");
        assert!(g0.is_mark, "i-matra 是记号");
        assert!(!g1.is_mark, "基辅音非记号");
        // 零宽记号与基字同笔位（无 GPOS 时不偏移——已知简化）。
        assert_eq!(g0.x.to_bits(), g1.x.to_bits(), "零宽记号与基字同笔位");
    }

    #[test]
    fn devanagari_conjunct_cluster_merges() {
        // KA + virama + SSA → 一个辅音簇，virama 被吸收。
        let text = "\u{0915}\u{094D}\u{0937}";
        assert_eq!(cluster_count(text), 1, "virama 连接的辅音应合为一簇");
        let f = font_regular();
        let run = shape(&f, text, &params()).unwrap_or_else(|e| panic!("{e}"));
        let c = &run.graph.clusters[0];
        assert_eq!(c.kind, ClusterKind::IndicConjunct);
        assert_eq!(c.visual_glyphs.len(), 2, "virama 被吸收，只剩两个辅音字形");
        assert_eq!(run.glyphs[c.visual_glyphs[0]].ch, '\u{0915}');
        assert_eq!(run.glyphs[c.visual_glyphs[1]].ch, '\u{0937}');
    }

    #[test]
    fn devanagari_conjunct_chain_merges_all_consonants() {
        // KA + virama + SSA + virama + SA：链式 conjunct 全并为一簇。
        assert_eq!(cluster_count("\u{0915}\u{094D}\u{0937}\u{094D}\u{0938}"), 1, "链式 conjunct 全并一簇");
        // 链后跟独立辅音则断簇。
        assert_eq!(cluster_count("\u{0915}\u{094D}\u{0937}\u{0915}"), 2, "链后独立辅音应断簇");
    }

    #[test]
    fn devanagari_nukta_absorbed_into_cluster() {
        let f = font_regular();
        let run = shape(&f, "\u{0915}\u{093C}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.cluster_count(), 1, "nukta 应并入辅音簇");
        let c = &run.graph.clusters[0];
        assert_eq!(c.visual_glyphs.len(), 2, "nukta 出字形但为零宽记号");
        assert!(run.glyphs[c.visual_glyphs[1]].is_mark);
    }

    #[test]
    fn devanagari_post_base_matra_stays_after_base() {
        // KA + aa-matra(U+093E)：后置记号不得前置。
        let f = font_regular();
        let run = shape(&f, "\u{0915}\u{093E}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let c = &run.graph.clusters[0];
        assert_eq!(run.glyphs[c.visual_glyphs[0]].ch, '\u{0915}', "后置记号时基字在前");
        assert_eq!(run.glyphs[c.visual_glyphs[1]].ch, '\u{093E}');
    }

    #[test]
    fn devanagari_prishthamatra_treated_as_post_base_known_simplification() {
        // U+094E 在 Unicode 定位类别上为 Pre，但天城文实际书写在基字下方偏右。
        // 本实现按后置处理——**已知简化**，此测试锁定当前行为以防无声漂移。
        let f = font_regular();
        let run = shape(&f, "\u{0915}\u{094E}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let c = &run.graph.clusters[0];
        assert_eq!(run.glyphs[c.visual_glyphs[0]].ch, '\u{0915}');
        assert_eq!(run.glyphs[c.visual_glyphs[1]].ch, '\u{094E}');
    }

    #[test]
    fn devanagari_danda_breaks_cluster() {
        assert_eq!(cluster_count("\u{0915}\u{0964}"), 2, "danda 是标点，应断簇");
    }

    // ------------------------------------------------------------------
    // 关卡 5：双向与视觉序
    // ------------------------------------------------------------------

    #[test]
    fn rtl_paragraph_base_level_and_visual_reversal() {
        let f = font_regular();
        let run =
            shape(&f, "\u{0628}\u{0628}\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.base_direction, BaseDirection::Rtl, "阿拉伯首强字符定 RTL 基线");
        assert_eq!(run.base_level, 1);
        assert_eq!(run.visual, vec![2, 1, 0], "RTL 视觉序必须反转");
    }

    #[test]
    fn ltr_paragraph_preserves_visual_order() {
        let f = font_regular();
        let run = shape(&f, "abc", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.base_direction, BaseDirection::Ltr);
        assert_eq!(run.base_level, 0);
        assert_eq!(run.visual, vec![0, 1, 2], "LTR 视觉序不得反转");
    }

    #[test]
    fn hebrew_is_rtl_too() {
        let f = font_regular();
        let run = shape(&f, "\u{05D0}\u{05D1}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.base_direction, BaseDirection::Rtl, "希伯来也是 RTL");
        assert_eq!(run.visual, vec![1, 0]);
    }

    #[test]
    fn mixed_ltr_paragraph_embeds_rtl_run_reversed_in_place() {
        // "ab" + 阿拉伯 "بب"：基线 LTR，阿拉伯段原位反转（UAX #9 L2）。
        let f = font_regular();
        let run =
            shape(&f, "ab\u{0628}\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.base_direction, BaseDirection::Ltr, "首个强字符是拉丁 → LTR 基线");
        assert_eq!(run.cluster_count(), 4);
        let chars_in_visual: String = run.visual.iter().map(|&g| run.glyphs[g].ch).collect();
        assert_eq!(chars_in_visual, "ab\u{0628}\u{0628}", "阿拉伯段应原位反转");
    }

    #[test]
    fn forced_base_direction_overrides_auto_detection() {
        // 显式 LTR 覆盖 Auto 的**基线方向判定**（本该被阿拉伯首字符定为 RTL）。
        let f = font_regular();
        let auto = shape(&f, "\u{0628}\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let forced =
            shape(&f, "\u{0628}\u{0628}", &ShapeParams { base_direction: BaseDirection::Ltr, ..params() })
                .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(auto.base_direction, BaseDirection::Rtl, "Auto 应由阿拉伯定为 RTL");
        assert_eq!(forced.base_direction, BaseDirection::Ltr, "显式 LTR 应覆盖 Auto");
        assert_eq!(forced.base_level, 0, "强制 LTR → 基线层级 0");
        // 阿拉伯段仍是 RTL 强方向 → 层级 1 → L2 仍反转该段（UBA 正确行为：
        // 段落基线方向不改变强 RTL 段的内部次序）。
        assert_eq!(forced.visual, vec![1, 0], "段内 RTL 次序不因基线方向而变");
    }

    #[test]
    fn neutral_leading_chars_do_not_decide_base_direction() {
        // 前导数字/空格不决定基线（跳过中性字符）→ 首个强字符是阿拉伯 → RTL。
        let f = font_regular();
        let run = shape(&f, " 12\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.base_direction, BaseDirection::Rtl, "中性字符应被跳过");
    }

    #[test]
    fn cluster_graph_visual_order_matches_run_visual() {
        // ClusterGraph 方法与 shape 内部共用 `l2_reorder`——不得漂移。
        let f = font_regular();
        let run = shape(&f, "a\u{0628}\u{0628}\u{0628}b", &params()).unwrap_or_else(|e| panic!("{e}"));
        let from_graph = run.graph.visual_cluster_order(run.base_level);
        let mut expanded: Vec<usize> = Vec::new();
        for ci in from_graph {
            if let Some(c) = run.graph.cluster(ci) {
                expanded.extend_from_slice(&c.visual_glyphs);
            }
        }
        assert_eq!(expanded, run.visual, "视觉簇序展开后应等于 glyphs 绘制序");
    }

    // ------------------------------------------------------------------
    // 关卡 6：字素簇图
    // ------------------------------------------------------------------

    #[test]
    fn cluster_boundaries_for_latin_marks_and_cjk() {
        assert_eq!(cluster_count("abc"), 3, "ASCII 每字一簇");
        assert_eq!(cluster_count("e\u{0301}"), 1, "组合记号应并入基字符");
        assert_eq!(cluster_count("\u{4E2D}\u{6587}"), 2, "汉字每字一簇");
        assert_eq!(cluster_count("\u{0301}"), 1, "孤立记号不得被吞掉");
    }

    #[test]
    fn cluster_graph_exposes_offsets_without_panicking() {
        let f = font_regular();
        let run = shape(&f, "a\u{0301}\u{4E2D}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let c0 = run.graph.cluster(0).unwrap_or_else(|| panic!("无第 0 簇"));
        assert_eq!((c0.start, c0.end), (0, 2), "a + U+0301 占 2 个码位");
        assert_eq!(c0.char_len(), 2);
        assert_eq!(c0.kind, ClusterKind::Marked);
        let c1 = run.graph.cluster(1).unwrap_or_else(|| panic!("无第 1 簇"));
        assert_eq!((c1.start, c1.end), (2, 3));
        assert_eq!(c1.kind, ClusterKind::Single);
        // 越界访问返回 None，不 panic（宪章 C18）。
        assert!(run.graph.cluster(99).is_none());
    }

    #[test]
    fn cluster_count_agrees_with_shape_without_font_lookup() {
        // `cluster_count` 便捷入口必须与 `shape` 的簇数一致（两处实现不得漂移）。
        for text in ["abc", "e\u{0301}", "\u{4E2D}\u{6587}", "\u{0644}\u{0627}", "\u{0915}\u{093F}"] {
            let f = font_regular();
            let run = shape(&f, text, &params()).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(cluster_count(text), run.cluster_count(), "文本 {text:?} 簇数不一致");
        }
    }

    // ------------------------------------------------------------------
    // 关卡 7：确定性契约（同输入两遍逐字节一致）
    // ------------------------------------------------------------------

    #[test]
    fn shaping_is_byte_level_deterministic() {
        let f = font_regular();
        // 混排：拉丁 + 阿拉伯 + 连字 + i-matra 前置 +  conjunct + 汉字，覆盖全部分支。
        let text = "ab\u{0628}\u{0644}\u{0627}\u{0915}\u{093F}\u{0937}\u{4E2D}";
        let a = shape(&f, text, &params()).unwrap_or_else(|e| panic!("{e}"));
        let b = shape(&f, text, &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.canonical_bytes(), b.canonical_bytes(), "同输入两遍必须逐字节一致");
        assert_eq!(a.glyphs, b.glyphs);
        assert_eq!(a.graph, b.graph);
        assert_eq!(a.width.to_bits(), b.width.to_bits());
    }

    #[test]
    fn canonical_bytes_are_input_sensitive() {
        let f = font_regular();
        let a = shape(&f, "\u{0628}\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let b = shape(&f, "\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_ne!(a.canonical_bytes(), b.canonical_bytes(), "不同输入必须产出不同规范字节");
    }

    #[test]
    fn empty_text_yields_empty_run_without_error() {
        let f = font_regular();
        let run = shape(&f, "", &params()).unwrap_or_else(|e| panic!("空文本不得报错：{e}"));
        assert_eq!(run.cluster_count(), 0);
        assert!(run.glyphs.is_empty());
        assert!(run.visual.is_empty());
        assert!(run.graph.is_empty());
        assert_eq!(run.width, 0.0);
        assert_eq!(run.missing_glyphs, 0);
    }

    // ------------------------------------------------------------------
    // 关卡 8：笔位累加与错误路径
    // ------------------------------------------------------------------

    #[test]
    fn pen_positions_accumulate_as_prefix_sum_in_visual_order() {
        let f = font_regular();
        let run = shape(&f, "abc", &params()).unwrap_or_else(|e| panic!("{e}"));
        let mut expect = 0.0f32;
        for &gi in &run.visual {
            let g = run.glyphs.get(gi).unwrap_or_else(|| panic!("视觉下标越界：{gi}"));
            assert_eq!(g.x.to_bits(), expect.to_bits(), "笔位须等于视觉序前缀和");
            expect += g.advance;
        }
        assert_eq!(run.width.to_bits(), expect.to_bits(), "行宽 = 前缀和终值");
    }

    #[test]
    fn rtl_pen_positions_still_increase_left_to_right() {
        // RTL：视觉序反转后，笔位仍从左到右递增（x 是画布坐标，与逻辑序无关）。
        let f = font_regular();
        let run = shape(&f, "\u{4E2D}\u{6587}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let mut expect = 0.0f32;
        for &gi in &run.visual {
            let g = run.glyphs.get(gi).unwrap_or_else(|| panic!("越界"));
            assert_eq!(g.x.to_bits(), expect.to_bits());
            expect += g.advance;
        }
    }

    #[test]
    fn marks_are_zero_advance() {
        let f = font_regular();
        let run = shape(&f, "e\u{0301}", &params()).unwrap_or_else(|e| panic!("{e}"));
        let mark = run.glyphs.iter().find(|g| g.is_mark).unwrap_or_else(|| panic!("无记号字形"));
        assert_eq!(mark.advance.to_bits(), 0.0f32.to_bits(), "记号步进必须为 0");
    }

    #[test]
    fn arabic_missing_glyphs_are_reported_honestly() {
        // 内嵌字体子集不含阿拉伯字形 → 基字符全部缺字形，如实计数（不静默）。
        let f = font_regular();
        let run =
            shape(&f, "\u{0628}\u{0628}\u{0628}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.missing_glyphs, 3, "缺字形必须如实计数（预期结果，非缺陷）");
        // 记号缺字形不重复计数。
        let with_mark =
            shape(&f, "\u{0628}\u{064E}", &params()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(with_mark.missing_glyphs, 1, "记号缺字形不得计入基字符统计");
    }

    #[test]
    fn invalid_params_return_structured_three_part_error() {
        let f = font_regular();
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let p = ShapeParams { font_size: bad, ..params() };
            let err = shape(&f, "a", &p).unwrap_err();
            assert_eq!(err.code, "P1101", "非法字号应报 P1101");
            let text = err.to_string();
            assert!(text.starts_with("error[P1101] "), "三段式首段【在哪】：{text}");
            assert!(text.contains("什么事："), "必含【什么事】段：{text}");
            assert!(text.contains("建议："), "必含【建议】段（C19）：{text}");
        }
        let p = ShapeParams { letter_spacing: f32::NAN, ..params() };
        let err = shape(&f, "a", &p).unwrap_err();
        assert_eq!(err.code, "P1101");
        assert!(err.location.contains("letter_spacing"), "报错应指名字距参数：{}", err.location);
    }

    #[test]
    fn layout_api_is_untouched_by_shaping() {
        // `layout.rs` 既有 API 必须仍可用（本模块只做增量、不改其签名）。
        use crate::layout::{layout_text, TextParams as LayoutParams};
        let f = font_regular();
        let l = layout_text(&f, "\u{4E2D}\u{6587}", &LayoutParams::default());
        assert_eq!(l.glyphs.len(), 2, "layout_text 行为不变");
    }
}
