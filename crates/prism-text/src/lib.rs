//! `prism-text` — 文本排版（文字管线）。
//!
//! 模块构成：
//! - [`ttf`]：TrueType 二进制解析（cmap/glyf/loca/head/hhea/hmtx/maxp 子集）；
//! - [`shaping`]：复杂脚本整形（手写字素簇图 + 阿拉伯系连接 + 天城文
//!   前置重排/辅音簇 + 双向视觉序；零 Unicode 库，分类表全部手写）；
//! - [`raster`]：字形光栅化（经 `prism-raster` 扫描线覆盖率）；
//! - [`layout`]：度量/贪心断行/行首行末禁则/对齐（GB/T 15834）；
//! - [`vertical`]：竖排 CJK（右起列、标点转向、列首禁则）；
//! - [`math_symbols`]：希腊/数学符号程序化回退表（cmap 缺失字符的
//!   合成字形；v3-A 修复「ε/δ 渲染成豆腐块」缺陷）；
//! - [`math_ast`]：公式 AST（上下标 / 分式 / 根号 / 大型算子 / 矩阵；V2-j）；
//! - [`math_layout`]：公式排版引擎（AST → 字形 + 规则线布局；V2-j）；
//! - [`cache`]：确定性字形位图缓存（纯加速，不改输出）；
//! - [`presets`]：Caption 字幕条与 Kinetic 逐字动画（接 `prism-anim`）；
//! - [`node`]：`GlyphAdvance` 特解注册（D-27 静态汇入制）。
//!
//! # 外部字体装载 + 回退链
//! [`FontSet`] 把声明字体序列（TTF/OTF 文件读入）组织成**回退链**：
//! 逐级尝试 cmap 覆盖 → 末端接 [`math_symbols`] 程序化符号表 → 全缺失
//! 返回携带缺失码位清单（U+XXXX）的三段式诊断错误，永不静默 tofu
//! （D-17）。装载期读入文件字节并记录 SHA-256（R3：外部字体漂移 →
//! 缓存键/签名扩展用）；解析是字节流的纯函数，同输入两跑逐位一致。
//!
//! # 字体资产
//! 内嵌 OFL 1.1 的 Noto Sans SC 子集（改名 Prism Sans SC；CFF→glyf 转换、
//! GB2312 全集 + ASCII + 中文标点共 6888 码位），见 `font_data/README.md`
//! 的来源/许可/合规记录。运行时零外部 crate；fontTools 仅构建期使用。
//!
//! # 确定性契约
//! 全管线是 (字体字节, 文本, 参数) 的纯函数：无隐藏状态、无随机源、
//! 浮点只用 ±×÷ 与 IEEE 精确 floor/round-to-nearest（量化出口）。

use prism_io::{sha256, Fbuf16F};

use crate::math_symbols::{advance_for_gid, fallback_gid, MATH_UNITS_PER_EM};
use crate::raster::{blit_glyph, render_glyph, GlyphBitmap};
use crate::ttf::{Font, FontError};

pub mod cache;
pub mod gsub;
pub mod layout;
pub mod layout_grid;
pub mod math_ast;
pub mod math_layout;
pub mod math_symbols;
pub mod multiscript_font;
pub mod node;
pub mod opentype;
pub mod path_text;
pub mod presets;
pub mod raster;
pub mod reveal;
pub mod sdf;
pub mod shaping;
pub mod ttf;
pub mod vertical;

/// 内嵌 Regular 子集字体（构建期产物；OFL 1.1，见 font_data/README.md）。
pub const FONT_REGULAR: &[u8] = include_bytes!("../font_data/PrismSansSC-Regular.ttf");

/// 内嵌 Bold 子集字体（构建期产物；OFL 1.1，见 font_data/README.md）。
pub const FONT_BOLD: &[u8] = include_bytes!("../font_data/PrismSansSC-Bold.ttf");

/// 内嵌**多语种**字体（本项目自研资产，零第三方 crate 生成）。
///
/// 由 [`multiscript_font::build_font`] 生成并落盘为
/// `assets/prism_multiscript.ttf`；含拉丁 A–Z + 阿拉伯 5 字母 × 四态 +
/// lam-alef 连字 + 天城文（KA / i-matra / virama）+ 真 GSUB 特征。见
/// `assets/README.md`。
pub const FONT_MULTISCRIPT: &[u8] = include_bytes!("../assets/prism_multiscript.ttf");

/// 解析内嵌 Regular 字体。
///
/// 资产在构建期已验证可解析；失败属于编程错误（panic 消息恒定）。
pub fn font_regular() -> ttf::Font<'static> {
    ttf::Font::parse(FONT_REGULAR)
        .expect("内嵌 PrismSansSC-Regular.ttf 必须可解析（构建期校验通过）")
}

/// 解析内嵌 Bold 字体。
pub fn font_bold() -> ttf::Font<'static> {
    ttf::Font::parse(FONT_BOLD)
        .expect("内嵌 PrismSansSC-Bold.ttf 必须可解析（构建期校验通过）")
}

/// 解析内嵌多语种字体（自研资产，见 [`FONT_MULTISCRIPT`]）。
///
/// 资产在构建期已验证可解析；失败属于编程错误（panic 消息恒定）。
pub fn font_multiscript() -> ttf::Font<'static> {
    ttf::Font::parse(FONT_MULTISCRIPT)
        .expect("内嵌 prism_multiscript.ttf 必须可解析（生成期校验通过）")
}

// ---------------------------------------------------------------------------
// 外部字体装载 + 回退链
// ---------------------------------------------------------------------------

/// 内嵌 Regular 字体入链时的标签（[`FontSet::push_embedded`] 写入）。
pub const EMBEDDED_FONT_LABEL: &str = "<embedded:PrismSansSC-Regular>";

/// 字体文件读取失败错误码（装载路径；T001–T007 归 `ttf.rs` 解析族）。
const FONT_IO_CODE: &str = "T008";

/// 槽位索引越界错误码（调用方 Bug：解析结果里的 origin 索引恒有效）。
const SLOT_OOB_CODE: &str = "T010";

/// 回退链中的单个字体装载槽位。
///
/// 持有装载时读入的完整字节（文件内容或内嵌资产拷贝）；解析按需进行
/// （[`FontSlot::font`]，纯函数）——不在结构体内缓存 `Font` 是因为
/// `Font<'a>` 借自身字节即自引用结构，std 无零依赖安全解法；按需解析
/// 的成本是一次表目录扫描（微秒级），渲染热路径由调用方 GlyphCache
/// memo 位图兜底。
#[derive(Debug, Clone)]
pub struct FontSlot {
    /// 字体标签（文件路径或 [`EMBEDDED_FONT_LABEL`]；诊断信息用）。
    pub label: String,
    /// 装载时读入的完整字节。
    pub bytes: Vec<u8>,
    /// 字节 SHA-256（R3：外部字体漂移 → 缓存键/签名扩展用）。
    pub sha256: [u8; 32],
}

impl FontSlot {
    /// 以字节构造槽位（即时计算 SHA-256；不解析——解析见 [`font`](FontSlot::font)）。
    fn new(label: impl Into<String>, bytes: Vec<u8>) -> Self {
        FontSlot { label: label.into(), sha256: sha256(&bytes), bytes }
    }

    /// 解析本槽位字体（字节流的纯函数：同字节必得同解析结果）。
    pub fn font(&self) -> Result<Font<'_>, FontError> {
        Font::parse(&self.bytes)
    }
}

/// 字形来源（回退深度信息；[`ResolvedGlyph`] 的伴随数据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphOrigin {
    /// 来自回退链第 `index` 个字体（0 = 首选；逐级下移即回退深度）。
    Chain(usize),
    /// 程序化符号表（[`math_symbols`]；链末端，缩放基准 = 内嵌 Regular
    /// 的 upem=1000，与 v3-A 内嵌管线行为逐位一致）。
    Synthetic,
}

/// 回退链解析结果：字形 + 来源字体 + 回退深度。
///
/// 与 `.notdef` 静默 tofu 的区别：缺失码位**不会**产出
/// `ResolvedGlyph`，而是由 [`FontSet::resolve_glyph`] 返回携带缺失码位
/// 清单的诊断错误（D-17 诚实边界）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedGlyph {
    /// 字符。
    pub ch: char,
    /// 字形 id（合成 gid 见 [`math_symbols::MATH_GID_BASE`]）。
    pub gid: u16,
    /// 来源（回退深度）。
    pub origin: GlyphOrigin,
}

/// 字体回退链。
///
/// 语义：**声明字体序列逐级尝试**（顺序 = 回退顺序，首选在前）→ 末端
/// 接 [`math_symbols`] 程序化符号表 → 全缺失才报错（错误含缺失码位
/// 清单）。链内不含隐式兜底：调用方要内嵌字体兜底就显式
/// [`push_embedded`](FontSet::push_embedded)——「全缺失才报错」的诚实
/// 边界不容忍静默的隐性字体。
///
/// # 确定性
/// 装载只做「读字节 + 算摘要 + 验解析」；解析、度量、渲染全部是
/// (字节, 字符, 参数) 的纯函数。同文件两跑 bitwise 相同。
#[derive(Debug, Clone, Default)]
pub struct FontSet {
    chain: Vec<FontSlot>,
}

impl FontSet {
    /// 构造空链（后续用 [`push_file`](FontSet::push_file) 等追加）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 从文件路径序列批量装载（顺序 = 回退链顺序，首选在前）。
    pub fn from_files<I, S>(paths: I) -> Result<Self, FontError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = Self::new();
        for p in paths {
            set.push_file(p)?;
        }
        Ok(set)
    }

    /// 单文件便捷装载（单元素链）。
    pub fn load_font_file(path: impl AsRef<str>) -> Result<Self, FontError> {
        Self::from_files([path.as_ref().to_string()])
    }

    /// 追加一个字体文件到链尾，返回槽位索引。
    ///
    /// 两条失败路径都归 **T008**（语义错误，消息必含路径）：
    /// 1. 读取失败（不存在 / 不可读）→ 路径 + 系统原因；
    /// 2. 解析失败（字节损坏 / 非 glyf 轮廓）→ 路径 + 内层错误码与事实。
    ///
    /// 包装而非透传：`ttf.rs` 的 T001–T007 描述的是「某个字节流坏了」，
    /// 缺了「哪个文件」这一半信息，对使用者不可行动（D-17）。
    pub fn push_file(&mut self, path: impl AsRef<str>) -> Result<usize, FontError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FontError {
            code: FONT_IO_CODE,
            problem: format!("字体文件读取失败：{}（{}）", path, e),
            fix: "确认路径存在且可读；file 字段应指向 .ttf（或 glyf 轮廓的 OTF）文件，CFF 轮廓 OTF 本解析器不支持（构建期转换）".to_string(),
        })?;
        let slot = FontSlot::new(path, bytes);
        if let Err(inner) = slot.font() {
            return Err(FontError {
                code: FONT_IO_CODE,
                problem: format!(
                    "字体文件解析失败：{}（内层 {}：{}）",
                    path, inner.code, inner.problem
                ),
                fix: format!("{}；同时确认该路径指向未截断的字体文件", inner.fix),
            });
        }
        self.chain.push(slot);
        Ok(self.chain.len() - 1)
    }

    /// 以字节直装（内嵌资产 / 测试数据入链用），返回槽位索引。
    ///
    /// 装载期即解析校验：坏字节在**装载期**暴露（D-17），而不是拖到
    /// 渲染期才静默产出豆腐块。
    pub fn push_bytes(&mut self, label: impl Into<String>, bytes: Vec<u8>) -> Result<usize, FontError> {
        let slot = FontSlot::new(label, bytes);
        slot.font()?;
        self.chain.push(slot);
        Ok(self.chain.len() - 1)
    }

    /// 把内嵌 Regular 字体追加为链尾一环（标签 [`EMBEDDED_FONT_LABEL`]）。
    ///
    /// 内嵌资产构建期已验证可解析，故不返回 `Result`（失败属编程错误，
    /// 由既有测试网锁定）。
    pub fn push_embedded(&mut self) -> usize {
        let idx = self.chain.len();
        self.chain.push(FontSlot::new(EMBEDDED_FONT_LABEL, FONT_REGULAR.to_vec()));
        idx
    }

    /// 链长度（声明字体个数；不含末端程序化符号表）。
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// 是否空链。
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }

    /// 第 `index` 个槽位的标签。
    pub fn label(&self, index: usize) -> Option<&str> {
        self.chain.get(index).map(|s| s.label.as_str())
    }

    /// 第 `index` 个槽位字节的 SHA-256（缓存键 / 签名扩展用，R3）。
    pub fn sha256_of(&self, index: usize) -> Option<[u8; 32]> {
        self.chain.get(index).map(|s| s.sha256)
    }

    /// 第 `index` 个槽位的字节。
    pub fn bytes_of(&self, index: usize) -> Option<&[u8]> {
        self.chain.get(index).map(|s| s.bytes.as_slice())
    }

    /// 解析第 `index` 个字体（纯函数，可重复调用）。
    pub fn font(&self, index: usize) -> Result<Font<'_>, FontError> {
        match self.chain.get(index) {
            Some(slot) => slot.font(),
            None => Err(FontError {
                code: SLOT_OOB_CODE,
                problem: format!("字体槽位索引 {} 越界（链长 {}）", index, self.chain.len()),
                fix: "origin 索引来自本 FontSet 的解析结果，越界属调用方 Bug".to_string(),
            }),
        }
    }

    /// 首个覆盖该码位的槽位索引（None = 链内全未覆盖）。
    ///
    /// cmap 映射到 gid 0（.notdef）视为**未覆盖**——那是豆腐块不是
    /// 字形，继续向下级回退（既有单字体管线不受影响：本方法只服务
    /// 回退链）。
    pub fn covering_slot(&self, ch: char) -> Option<usize> {
        for (i, slot) in self.chain.iter().enumerate() {
            if let Ok(f) = slot.font() {
                if f.glyph_index(ch).is_some_and(|g| g != 0) {
                    return Some(i);
                }
            }
        }
        None
    }

    /// 链内解析尝试（Ok(None) = 声明字体与符号表均未覆盖）。
    ///
    /// 解析错误（T001–T007）经 `?` 上抛；装载期已验解析，正常不会触发。
    fn try_resolve(&self, ch: char) -> Result<Option<ResolvedGlyph>, FontError> {
        if let Some(i) = self.covering_slot(ch) {
            let gid = self.font(i)?.glyph_index(ch).unwrap_or(0);
            return Ok(Some(ResolvedGlyph { ch, gid, origin: GlyphOrigin::Chain(i) }));
        }
        Ok(fallback_gid(ch)
            .map(|gid| ResolvedGlyph { ch, gid, origin: GlyphOrigin::Synthetic }))
    }

    /// 回退链解析（核心入口）。
    ///
    /// 声明字体序列逐级尝试 → 程序化符号表 → 全缺失返回携带缺失码位
    /// `U+XXXX` 的诊断错误（T009；D-17：永不静默 tofu）。
    pub fn resolve_glyph(&self, ch: char) -> Result<ResolvedGlyph, FontError> {
        match self.try_resolve(ch)? {
            Some(rg) => Ok(rg),
            None => {
                let labels = self.labels();
                Err(crate::math_symbols::missing_codepoint_error(&[ch], &labels))
            }
        }
    }

    /// 文本级解析：逐字符解析；全缺失时报出**全部**缺失码位清单
    /// （按首现顺序去重），而不是只报第一个。
    pub fn resolve_text(&self, text: &str) -> Result<Vec<ResolvedGlyph>, FontError> {
        let mut out: Vec<ResolvedGlyph> = Vec::with_capacity(text.chars().count());
        let mut missing: Vec<char> = Vec::new();
        for ch in text.chars() {
            match self.try_resolve(ch)? {
                Some(rg) => out.push(rg),
                None => {
                    if !missing.contains(&ch) {
                        missing.push(ch);
                    }
                }
            }
        }
        if missing.is_empty() {
            Ok(out)
        } else {
            let labels = self.labels();
            Err(crate::math_symbols::missing_codepoint_error(&missing, &labels))
        }
    }

    /// 声明字体标签序列（诊断信息用；含内嵌标记）。
    fn labels(&self) -> Vec<&str> {
        self.chain.iter().map(|s| s.label.as_str()).collect()
    }

    /// 解析结果的 em 框单位数（合成字形 = 符号表基准 1000）。
    pub fn units_per_em(&self, rg: &ResolvedGlyph) -> Result<u16, FontError> {
        match rg.origin {
            GlyphOrigin::Chain(i) => Ok(self.font(i)?.units_per_em()),
            GlyphOrigin::Synthetic => Ok(MATH_UNITS_PER_EM),
        }
    }

    /// 解析结果的步进宽（字体单位）。
    ///
    /// 真实字形走来源字体 hmtx；合成字形走符号表 advance（hmtx 对越界
    /// id 只会钳到末项给出错值，与 `layout.rs` 同纪律）。
    pub fn advance(&self, rg: &ResolvedGlyph) -> Result<u16, FontError> {
        match rg.origin {
            GlyphOrigin::Chain(i) => Ok(self.font(i)?.advance(rg.gid)),
            GlyphOrigin::Synthetic => Ok(advance_for_gid(rg.gid).unwrap_or(0)),
        }
    }

    /// 渲染解析结果为覆盖率位图（接入既有 [`raster`] 管线）。
    ///
    /// 合成字形以内嵌 Regular 为单位/轮廓基准：符号表按 upem=1000
    /// 定义，与 v3-A 内嵌管线逐位一致（外部链混排希腊/数学符号时
    /// 尺寸不随链上其他字体的 upem 漂移）。
    pub fn render(&self, rg: &ResolvedGlyph, px_size: f32) -> Result<GlyphBitmap, FontError> {
        match rg.origin {
            GlyphOrigin::Chain(i) => {
                let font = self.font(i)?;
                Ok(render_glyph(&font, rg.gid, px_size))
            }
            GlyphOrigin::Synthetic => Ok(render_glyph(&font_regular(), rg.gid, px_size)),
        }
    }

    /// 单行回退链排版 + 光栅：逐字形解析 → 渲染 → blit，笔位按**各自
    /// 来源字体**的 advance × (font_size / upem) 推进（混排不同 upem
    /// 字体时尺寸正确）。不断行——换行由调用方分行后逐行调用。
    ///
    /// 任一字码全缺失时**整体不画**并返回缺失码位清单错误（原子性：
    /// 不画半行，下游不会把半行误认成完整行）。
    ///
    /// 返回末字形之后的笔位 x（不含行尾 letter_spacing）。
    pub fn draw_resolved_line(
        &self,
        fbuf: &mut Fbuf16F,
        text: &str,
        font_size: f32,
        pen_x: f32,
        baseline_y: f32,
        letter_spacing: f32,
        color: [f32; 4],
    ) -> Result<f32, FontError> {
        let resolved = self.resolve_text(text)?;
        let mut pen = pen_x;
        for rg in &resolved {
            let bmp = self.render(rg, font_size)?;
            blit_glyph(fbuf, &bmp, pen, baseline_y, color);
            let upem = self.units_per_em(rg)? as f32;
            pen += self.advance(rg)? as f32 * (font_size / upem) + letter_spacing;
        }
        Ok(pen - letter_spacing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_fonts_are_real_and_parseable() {
        // 真实内嵌：体积量级断言（子集后 1–3MB）。
        assert!(
            FONT_REGULAR.len() > 1_000_000,
            "Regular 子集应 >1MB，实际 {} B",
            FONT_REGULAR.len()
        );
        assert!(
            FONT_BOLD.len() > 1_000_000,
            "Bold 子集应 >1MB，实际 {} B",
            FONT_BOLD.len()
        );
        // 解析 + 基本度量：upem=1000（构建脚本固定）。
        let f = font_regular();
        assert_eq!(f.units_per_em(), 1000);
        assert!(f.num_glyphs() > 6000, "GB2312 全集子集应含 6000+ 字形");
        let b = font_bold();
        assert_eq!(b.units_per_em(), 1000);
    }

    /// 回退链逐级解析：首选覆盖 → 逐级下移 → 末端程序化符号表。
    #[test]
    fn font_set_resolves_level_by_level() {
        let mut set = FontSet::new();
        // 链：多语种字体（拉丁/阿拉伯/天城文，无希腊无汉字）→ 内嵌 Regular。
        let i_ms = set
            .push_bytes("multiscript", FONT_MULTISCRIPT.to_vec())
            .expect("多语种字体应可装载");
        let i_sc = set.push_embedded();
        // 首选即覆盖：'A' 不停留。
        let a = set.resolve_glyph('A').expect("A 应解析");
        assert_eq!(a.origin, GlyphOrigin::Chain(i_ms));
        // 首选未覆盖 → 回退深度 1（内嵌）。
        let zh = set.resolve_glyph('中').expect("中 应解析");
        assert_eq!(zh.origin, GlyphOrigin::Chain(i_sc));
        assert_eq!(zh.gid, font_regular().glyph_index('中').unwrap());
        // 阿拉伯：多语种覆盖（首选即中）。
        let beh = set.resolve_glyph('ب').expect("ب 应解析");
        assert_eq!(beh.origin, GlyphOrigin::Chain(i_ms));
        // 希腊：两字体均缺 → 末端程序化符号表（合成 gid）。
        let eps = set.resolve_glyph('ε').expect("ε 应走符号表末端");
        assert_eq!(eps.origin, GlyphOrigin::Synthetic);
        assert_eq!(eps.gid, math_symbols::fallback_gid('ε').unwrap());
        // covering_slot 与 origin 索引一致（回退深度可观测）。
        assert_eq!(set.covering_slot('A'), Some(i_ms));
        assert_eq!(set.covering_slot('中'), Some(i_sc));
        assert_eq!(set.covering_slot('ε'), None, "符号表不在链内覆盖判定里");
        // 合成字形渲染与 v3-A 内嵌管线逐位一致（同 gid 同 px）。
        let via_set = set.render(&eps, 48.0).expect("渲染");
        let via_embedded = render_glyph(&font_regular(), eps.gid, 48.0);
        assert_eq!(via_set.alpha, via_embedded.alpha, "合成字形须逐位一致");
    }

    /// 缺失码位诊断：单码位含 U+XXXX；文本级汇总全部缺失且去重（D-17）。
    #[test]
    fn font_set_missing_codepoints_are_diagnosed() {
        let mut set = FontSet::new();
        set.push_bytes("multiscript", FONT_MULTISCRIPT.to_vec()).expect("应可装载");
        // 单码位。
        let err = set.resolve_glyph('☃').expect_err("☃ 应缺失");
        assert_eq!(err.code, "T009");
        assert!(err.problem.contains("U+2603"), "应含缺失码位：{}", err.problem);
        assert!(err.problem.contains("multiscript"), "应列出声明链：{}", err.problem);
        // 文本级：'A' 可解析但整行仍失败，且两个缺失码位都列出、去重。
        let err = set.resolve_text("A☃😀☃").expect_err("应报缺失");
        assert!(err.problem.contains("U+2603"), "{}", err.problem);
        assert!(err.problem.contains("U+1F600"), "{}", err.problem);
        assert_eq!(err.problem.matches("U+2603").count(), 1, "缺失码位应去重");
        assert!(err.problem.contains("共 2 个"), "应报缺失总数：{}", err.problem);
        // 空链：同样如实诊断（不伪装）。
        let err = FontSet::new().resolve_glyph('A').expect_err("空链应报错");
        assert_eq!(err.code, "T009");
        assert!(err.problem.contains("空链"), "{}", err.problem);
    }

    /// 度量与来源一致：真实字形走 hmtx，合成字形走符号表 advance。
    #[test]
    fn font_set_advance_and_upem_follow_origin() {
        let mut set = FontSet::new();
        set.push_bytes("multiscript", FONT_MULTISCRIPT.to_vec()).expect("应可装载");
        let a = set.resolve_glyph('A').unwrap();
        let f = set.font(0).unwrap();
        assert_eq!(set.advance(&a).unwrap(), f.advance(a.gid));
        assert_eq!(set.units_per_em(&a).unwrap(), f.units_per_em());
        let eps = set.resolve_glyph('ε').unwrap();
        assert_eq!(
            set.advance(&eps).unwrap(),
            math_symbols::advance_for_gid(eps.gid).unwrap()
        );
        assert_eq!(set.units_per_em(&eps).unwrap(), MATH_UNITS_PER_EM);
    }

    /// 字节直装的确定性：同字节两装 SHA-256 一致，且与 prism-io 摘要一致。
    #[test]
    fn font_set_bytes_digest_is_stable() {
        let mut a = FontSet::new();
        let mut b = FontSet::new();
        a.push_bytes("x", FONT_REGULAR.to_vec()).unwrap();
        b.push_bytes("y", FONT_REGULAR.to_vec()).unwrap();
        assert_eq!(a.sha256_of(0), b.sha256_of(0));
        assert_eq!(a.sha256_of(0).unwrap(), sha256(FONT_REGULAR));
        assert_eq!(a.bytes_of(0).unwrap(), FONT_REGULAR);
        assert!(a.sha256_of(1).is_none(), "越界索引返回 None 不 panic");
        assert_eq!(a.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 外部装载 / 回退链 / 装载错误路径 / 确定性
    // -----------------------------------------------------------------------

    /// 把字体字节写到临时目录下的确定性文件名（同进程内同 `tag` 恒定路径）。
    ///
    /// 文件名含 pid 以免并行测试进程互相覆盖；内容恒为传入字节，
    /// 因此「同字节 → 同文件 → 同解析结果」的链路可复现。
    fn write_temp_font(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p: std::path::PathBuf = std::env::temp_dir();
        p.push(format!("prism-text-w1b-{}-{}.ttf", std::process::id(), tag));
        std::fs::write(&p, bytes).expect("临时字体文件应可写入");
        p
    }

    /// 外部装载：内嵌字节落盘 → `load_font_file` 读回 → 解析成功，且
    /// 渲染结果与内嵌路径**逐位一致**（同字节同管线，装载方式不影响输出）。
    #[test]
    fn external_font_file_loads_and_renders_identically() {
        let path = write_temp_font("regular-roundtrip", FONT_REGULAR);
        let set = FontSet::load_font_file(path.to_str().expect("路径应有效 UTF-8"))
            .expect("外部字体应可装载");
        assert_eq!(set.len(), 1, "单文件装载 = 单元素链");
        // 标签即路径原文（诊断可回溯到文件）。
        assert_eq!(set.label(0), Some(path.to_str().unwrap()));
        // 摘要与内嵌字节一致（同字节 → 同 SHA-256）。
        assert_eq!(set.sha256_of(0).unwrap(), sha256(FONT_REGULAR));
        assert_eq!(set.bytes_of(0).unwrap(), FONT_REGULAR, "读回字节应逐位等于源字节");
        // 解析成功 + 度量一致。
        let font = set.font(0).expect("外部字体应可解析");
        assert_eq!(font.units_per_em(), font_regular().units_per_em());
        assert_eq!(font.num_glyphs(), font_regular().num_glyphs());
        // 渲染逐位一致：同 gid 同 px 的覆盖率位图必须完全相同。
        let rg = set.resolve_glyph('中').expect("'中' 应解析");
        assert_eq!(rg.origin, GlyphOrigin::Chain(0));
        let external: GlyphBitmap = set.render(&rg, 32.0).expect("外部字体渲染");
        let embedded: GlyphBitmap = render_glyph(&font_regular(), rg.gid, 32.0);
        assert_eq!(external.width, embedded.width);
        assert_eq!(external.height, embedded.height);
        assert_eq!(external.alpha, embedded.alpha, "外部装载与内嵌路径须逐位一致");
        // 清理（失败不影响断言结论）。
        let _ = std::fs::remove_file(&path);
    }

    /// 装载错误路径：文件不存在 / 字节损坏 → **T008**（语义错误）且消息含路径。
    #[test]
    fn load_font_file_errors_are_t008_with_path() {
        // 1) 文件不存在。
        let mut missing = std::env::temp_dir();
        missing.push(format!("prism-text-w1b-{}-no-such-font.ttf", std::process::id()));
        let path = missing.to_str().expect("路径应有效 UTF-8").to_string();
        let err = FontSet::load_font_file(&path).expect_err("不存在的文件应报错");
        assert_eq!(err.code, "T008", "读取失败归 T008：{}", err);
        assert!(err.problem.contains(&path), "错误须含路径：{}", err.problem);
        assert!(err.problem.contains("读取失败"), "{}", err.problem);
        assert!(!err.fix.is_empty(), "三段式必含建议段");

        // 2) 文件存在但字节损坏（非 sfnt）。
        let bad = write_temp_font("corrupt", b"not-a-font-at-all");
        let bad_path = bad.to_str().expect("路径应有效 UTF-8").to_string();
        let err = FontSet::load_font_file(&bad_path).expect_err("坏字节应报错");
        assert_eq!(err.code, "T008", "解析失败同样归 T008：{}", err);
        assert!(err.problem.contains(&bad_path), "错误须含路径：{}", err.problem);
        assert!(err.problem.contains("解析失败"), "{}", err.problem);
        assert!(err.problem.contains("内层 T"), "应保留内层错误码：{}", err.problem);
        // 装载失败不污染链（原子性：不留下半装槽位）。
        let mut set = FontSet::new();
        assert!(set.push_file(&bad_path).is_err());
        assert!(set.is_empty(), "失败装载不得留下槽位");
        let _ = std::fs::remove_file(&bad);
    }

    /// 回退链：首选缺码位 → 命中后面的字体（深度可观测）；全链缺失 → Err 含码位。
    #[test]
    fn font_set_fallback_hits_later_font_and_reports_missing() {
        let mut set = FontSet::new();
        // A：多语种子集（拉丁/阿拉伯/天城文，无汉字）；B：内嵌 Regular（GB2312 全集）。
        let a = set.push_bytes("a-multiscript.ttf", FONT_MULTISCRIPT.to_vec()).expect("A 应可装载");
        let b = set.push_embedded();
        // A 不含 '中'、B 含 → 解析命中 B（回退深度 1）。
        assert_eq!(set.covering_slot('中'), Some(b));
        let zh = set.resolve_glyph('中').expect("'中' 应回退到 B");
        assert_eq!(zh.origin, GlyphOrigin::Chain(b));
        assert_eq!(zh.gid, font_regular().glyph_index('中').unwrap());
        assert_eq!(a, 0);
        // U+03B5 ε：A、B 均不含（内嵌子集只收 ASCII + GB2312 汉字区），
        // 故落到链末端程序化符号表（v3-A 合成字形），不走 tofu。
        assert_eq!(set.covering_slot('ε'), None);
        assert_eq!(set.resolve_glyph('ε').unwrap().origin, GlyphOrigin::Synthetic);
        // 全链 + 符号表均缺失 → Err 含缺失码位清单（诚实边界，不静默 tofu）。
        let err = set.resolve_glyph('☃').expect_err("☃ 应报缺失");
        assert_eq!(err.code, "T009");
        assert!(err.problem.contains("缺失码位"), "{}", err.problem);
        assert!(err.problem.contains("U+2603"), "{}", err.problem);
        // 文本级：缺失码位一并列出并去重（顺序 = 首现顺序）。
        let err = set.resolve_text("中☃中😀☃").expect_err("整行应报缺失");
        assert!(err.problem.contains("U+2603") && err.problem.contains("U+1F600"), "{}", err.problem);
        assert_eq!(err.problem.matches("U+2603").count(), 1, "缺失码位应去重");
        // 缺失码位清单格式（希腊字母两枚，按传入序逗号/顿号分隔）。
        let fmt = math_symbols::missing_codepoint_error(&['ε', 'δ'], &["a.ttf", "b.ttf"]);
        assert!(fmt.problem.contains("U+03B5"), "{}", fmt.problem);
        assert!(fmt.problem.contains("U+03B4"), "{}", fmt.problem);
    }

    /// 确定性：同字节流两次装载 → 摘要 / 解析 / 渲染逐位一致（解析是纯函数）。
    #[test]
    fn font_set_loading_is_deterministic() {
        let p1 = write_temp_font("det-a", FONT_REGULAR);
        let p2 = write_temp_font("det-b", FONT_REGULAR);
        let s1 = FontSet::load_font_file(p1.to_str().unwrap()).expect("第一次装载");
        let s2 = FontSet::load_font_file(p2.to_str().unwrap()).expect("第二次装载");
        // 路径不同（标签不同）但字节相同 → 摘要相同。
        assert_eq!(s1.sha256_of(0), s2.sha256_of(0));
        for ch in ['A', '中', '，', '0'] {
            let r1 = s1.resolve_glyph(ch).expect("解析");
            let r2 = s2.resolve_glyph(ch).expect("解析");
            assert_eq!(r1, r2, "同字节须得同解析结果：{ch}");
            assert_eq!(s1.advance(&r1).unwrap(), s2.advance(&r2).unwrap());
            assert_eq!(s1.units_per_em(&r1).unwrap(), s2.units_per_em(&r2).unwrap());
            let b1 = s1.render(&r1, 24.0).expect("渲染");
            let b2 = s2.render(&r2, 24.0).expect("渲染");
            assert_eq!(b1.width, b2.width);
            assert_eq!(b1.height, b2.height);
            assert_eq!(b1.alpha, b2.alpha, "同字节渲染须逐位一致：{ch}");
        }
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }
}
