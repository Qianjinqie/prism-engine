//! `prism-text` — 文本排版（文字管线）。
//!
//! 模块构成：
//! - [`ttf`]：TrueType 二进制解析（cmap/glyf/loca/head/hhea/hmtx/maxp 子集）；
//! - [`shaping`]：复杂脚本整形（手写字素簇图 + 阿拉伯系连接 + 天城文
//!   前置重排/辅音簇 + 双向视觉序；零 Unicode 库，分类表全部手写）；
//! - [`raster`]：字形光栅化（经 `prism-raster` 扫描线覆盖率）；
//! - [`layout`]：度量/贪心断行/行首行末禁则/对齐（GB/T 15834）；
//! - [`vertical`]：竖排 CJK（右起列、标点转向、列首禁则）；
//! - [`cache`]：确定性字形位图缓存（纯加速，不改输出）；
//! - [`presets`]：Caption 字幕条与 Kinetic 逐字动画（接 `prism-anim`）；
//! - [`node`]：`GlyphAdvance` 特解注册（D-27 静态汇入制）。
//!
//! # 字体资产
//! 内嵌 OFL 1.1 的 Noto Sans SC 子集（改名 Prism Sans SC；CFF→glyf 转换、
//! GB2312 全集 + ASCII + 中文标点共 6888 码位），见 `font_data/README.md`
//! 的来源/许可/合规记录。运行时零外部 crate；fontTools 仅构建期使用。
//!
//! # 确定性契约
//! 全管线是 (字体字节, 文本, 参数) 的纯函数：无隐藏状态、无随机源、
//! 浮点只用 ±×÷ 与 IEEE 精确 floor/round-to-nearest（量化出口）。

pub mod cache;
pub mod gsub;
pub mod layout;
pub mod layout_grid;
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
}
