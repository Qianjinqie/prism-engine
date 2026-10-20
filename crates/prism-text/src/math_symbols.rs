//! 程序化数学与希腊符号回退表（任务「文本符号表修复」）。
//!
//! # 背景（缺陷根因）
//! 内嵌 PrismSansSC 子集（GB2312 全集 + ASCII）**未收录**希腊字母与
//! 数学运算符区段：fontTools 验证 ε δ π θ Δ ∂ ∑ ∏ ∫ √ ∞ ≈ ≤ ≥ ≠ → ∈
//! ∪ ∩ ± × ÷ ≡ 等字符在两套内嵌字体的 cmap 中全部缺失 ⇒ `glyph_index`
//! 返回 `None` ⇒ 旧路径 `unwrap_or(0)` 落到 `.notdef`（可见豆腐块，
//! advance 1000 units）。成片 S3 字幕「无论 ε 多小」与读数「ε = 0.41」
//! 因此渲染成豆腐块/错形。
//!
//! # 方案（没有字形数据的要补数据或程序化生成）
//! 本模块以**程序化几何**补数据：为每个缺失字符定义字体单位
//! （upem=1000，基线 y=0，数学 y↑）的填充轮廓与步进宽，经合成 gid
//! （[`MATH_GID_BASE`] 起，与真实字形 id 恒不相交——真实字体仅 6889
//! 字形）接入既有排版/光栅管线：
//!
//! - [`layout::layout_text`]：cmap 查不到时回退到 [`fallback_gid`]，
//!   advance 走 [`advance_for_gid`]（而非 `hmtx`）；
//! - [`raster::render_glyph`]：字体轮廓为空时回退到 [`outline_for_gid`]。
//!
//! 两条钩子仅影响**当前渲染为豆腐块**的字符：字体已覆盖的字符行为
//! 逐位不变（真实字形优先），黄金测试零回归。
//!
//! # 确定性契约
//! 表是编译期静态数据；轮廓构造是 (字符) 的纯函数（无随机、无状态）。
//! 每个符号的全部轮廓统一归一化为**正方向**（y 向上系 CCW）——非零
//! 环绕填充下，同向重叠轮廓取并集；若方向不一致，重叠区会相互抵消
//! 出现"镂空"（如 θ 的横杠 vs 圆身）。
//!
//! # 坐标纪律
//! 与 `symbols.rs` 矢量符号库一致：`min_x >= 0`（无左悬挂）、
//! 纵向基本落 `[-220, 820]`·units（降部符号如 γ 尾部探到基线下），
//! 全部坐标有限。
//!
//! # 笔画原语
//! - [`seg`]：直粗段（矩形轮廓，斜段自动取垂直半宽）；
//! - [`band`]：二次曲带（透镜形——两端尖、中段等粗，书法笔画感；
//!   端点尖灭落在既有轮廓内的符号（如 ε 臂尖没入右 bulge）不露接缝）；
//! - [`tilde`]：等粗谷形波（≈ / ∼ 的波长线，端帽与谷底厚度一致）；
//! - [`ellipse`] / [`dot`]：闭合椭圆 / 实心点。

use crate::ttf::{FontError, Outline, OutlineContour, OutlinePoint};

/// 合成字形 id 基址。
///
/// 真实内嵌字体 `numGlyphs = 6889`；`0xF000` 起远不相交。GlyphCache
/// 以 gid 入键，合成 gid 与真实字形永不串键。表容量上限 256 项
/// （`0xF000 + 255 = 0xF0FF`，仍在 u16 内）。
pub const MATH_GID_BASE: u16 = 0xF000;

/// 程序化符号表的 em 框单位数（upem）。
///
/// 表内全部轮廓按 upem=1000 定义（与内嵌 Regular 一致，见模块头
/// 「坐标纪律」）；回退链渲染合成字形时以此为缩放基准（见 crate 根
/// [`crate::FontSet`] 的 Synthetic 分支）。
pub const MATH_UNITS_PER_EM: u16 = 1000;

/// 缺失码位诊断错误码（回退链 + 程序化符号表全缺失）。
///
/// 码段分配：T001–T007 归 `ttf.rs` 解析族；T008 = 字体文件读取失败
/// （crate 根装载路径）；T009 = 本诊断（码位缺失）。
const MISSING_CODEPOINT_CODE: &str = "T009";

/// 缺失码位清单 → 三段式诊断错误（D-17 诚实边界：永不静默 tofu）。
///
/// 回退链（声明字体序列）与程序化符号表**全部**未覆盖某码位时，由
/// 调用方以本函数构造错误：逐一列出具体缺失码位 `U+XXXX`（含字符
/// 本身）与声明字体链，便于定位该补哪个字体——而不是静默落到
/// `.notdef` 豆腐块。
///
/// `missing` 由调用方按首现顺序去重后传入；`chain` 为声明字体标签
/// 序列（文件路径或 `<embedded:...>` 标记）。
pub fn missing_codepoint_error(missing: &[char], chain: &[&str]) -> FontError {
    let list: String = missing
        .iter()
        .map(|&ch| format!("U+{:04X} '{}'", ch as u32, ch))
        .collect::<Vec<_>>()
        .join("、");
    let chain_desc: String = if chain.is_empty() {
        "（空链：未声明任何字体）".to_string()
    } else {
        chain.join(" → ")
    };
    FontError {
        code: MISSING_CODEPOINT_CODE,
        problem: format!(
            "字体回退链全部缺失码位：{}（共 {} 个；声明字体链：{}；程序化符号表亦未覆盖）",
            list,
            missing.len(),
            chain_desc
        ),
        fix: "把一个覆盖该码位的字体文件加入回退链（DSL 侧即 font 指令的声明顺序，首选在前），或改写文本避开该码位".to_string(),
    }
}

/// 圆 / 椭圆四次贝塞尔逼近常数（与 `symbols.rs` 同款标准值）。
const K_CIRCLE: f32 = 0.5523;

/// 符号表（字符, 步进宽 units）。**有序**：下标 i ⇒ gid = `MATH_GID_BASE + i`。
///
/// 覆盖：点名集合（ε π θ Δ ∂ ∑ ∏ ∫ √ ∞ ≈ ≤ ≥ ≠ → ∈ ∪ ∩ ± × ÷）
/// + 成片 scenes_math.rs 实际用到的 δ ≡ ≥ + 常用希腊字母与运算符
/// + 上标数字与度号（防御性覆盖）。
static SYMBOLS: &[(char, u16)] = &[
    // ---- 希腊小写（x 高 ~500）----
    ('α', 520),
    ('β', 560),
    ('γ', 500),
    ('δ', 560),
    ('ε', 520),
    ('θ', 560),
    ('λ', 520),
    ('μ', 560),
    ('π', 560),
    ('σ', 520),
    ('φ', 560),
    ('ψ', 580),
    ('ω', 560),
    // ---- 希腊大写（帽高 ~700）----
    ('Γ', 500),
    ('Δ', 560),
    ('Θ', 580),
    ('Λ', 560),
    ('Ξ', 520),
    ('Π', 560),
    ('Σ', 520),
    ('Φ', 580),
    ('Ψ', 600),
    ('Ω', 600),
    // ---- 数学运算符 ----
    ('∂', 520),
    ('√', 540),
    ('∫', 480),
    ('∑', 560),
    ('∏', 560),
    ('±', 540),
    ('∓', 540),
    ('×', 540),
    ('÷', 540),
    ('⋅', 480),
    ('≈', 580),
    ('≅', 580),
    ('∼', 580),
    ('≠', 560),
    ('≡', 540),
    ('≤', 580),
    ('≥', 580),
    ('≪', 640),
    ('→', 600),
    ('←', 600),
    ('↑', 560),
    ('↓', 560),
    ('↔', 700),
    ('⇒', 640),
    ('⇔', 760),
    ('∈', 560),
    ('∪', 540),
    ('∩', 540),
    ('∅', 580),
    ('∀', 540),
    ('∃', 480),
    ('¬', 520),
    ('∧', 540),
    ('∨', 540),
    ('∞', 600),
    ('∝', 680),
    ('−', 520),
    // ---- 上标 / 度号（防御性覆盖）----
    ('¹', 380),
    ('²', 400),
    ('³', 400),
    ('°', 400),
];

// ---------------------------------------------------------------------------
// 查表 API（纯函数）
// ---------------------------------------------------------------------------

/// 字符 → 合成 gid（表外字符 → `None`；调用方回落到 `.notdef`）。
pub fn fallback_gid(ch: char) -> Option<u16> {
    SYMBOLS
        .iter()
        .position(|&(c, _)| c == ch)
        .map(|i| MATH_GID_BASE + i as u16)
}

/// 合成 gid → 表内字符（诊断 / 测试用）。
pub fn symbol_char(gid: u16) -> Option<char> {
    SYMBOLS
        .get((gid.checked_sub(MATH_GID_BASE)?) as usize)
        .map(|&(c, _)| c)
}

/// 合成 gid 的步进宽（字体单位；非合成 gid → `None`）。
pub fn advance_for_gid(gid: u16) -> Option<u16> {
    SYMBOLS
        .get((gid.checked_sub(MATH_GID_BASE)?) as usize)
        .map(|&(_, a)| a)
}

/// 合成 gid 的轮廓（字体单位；非合成 gid → `None`）。
pub fn outline_for_gid(gid: u16) -> Option<Outline> {
    let ch = symbol_char(gid)?;
    Some(build_outline(ch))
}

/// 是否为合成 gid（[`MATH_GID_BASE`] 起）。
pub fn is_math_gid(gid: u16) -> bool {
    gid >= MATH_GID_BASE && symbol_char(gid).is_some()
}

// ---------------------------------------------------------------------------
// 几何原语（字体单位，y 向上，基线 0）
// ---------------------------------------------------------------------------

/// 曲线上锚点。
fn on(x: f32, y: f32) -> OutlinePoint {
    OutlinePoint { x, y, on_curve: true }
}

/// 二次控制点。
fn off(x: f32, y: f32) -> OutlinePoint {
    OutlinePoint { x, y, on_curve: false }
}

/// 闭合椭圆（起于正上方，8 点 4 二次弧）。
fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> OutlineContour {
    vec![
        on(cx, cy + ry),
        off(cx + K_CIRCLE * rx, cy + ry),
        on(cx + rx, cy),
        off(cx + rx, cy - K_CIRCLE * ry),
        on(cx, cy - ry),
        off(cx - K_CIRCLE * rx, cy - ry),
        on(cx - rx, cy),
        off(cx - rx, cy + K_CIRCLE * ry),
    ]
}

/// 圆（实心点 / 圆身复用）。
fn dot(cx: f32, cy: f32, r: f32) -> OutlineContour {
    ellipse(cx, cy, r, r)
}

/// 直粗段（矩形轮廓，四锚点；斜段自动取垂直半宽）。
///
/// `(x0,y0)–(x1,y1)` 为**中心线**，`w` 为笔画宽（units）。
fn seg(x0: f32, y0: f32, x1: f32, y1: f32, w: f32) -> OutlineContour {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len = (dx * dx + dy * dy).sqrt();
    let (px, py) = if len > 0.0 {
        (-dy / len * w * 0.5, dx / len * w * 0.5)
    } else {
        (0.0, 0.0)
    };
    vec![
        on(x0 + px, y0 + py),
        on(x1 + px, y1 + py),
        on(x1 - px, y1 - py),
        on(x0 - px, y0 - py),
    ]
}

/// 二次曲带（透镜形）：外弧 `a → b`（控制点 `c`），内弧 `b → a`（控制点 `d`）。
///
/// 轮廓点序 `[on a, off c, on b, off d]`：`outline_to_path` 展开为
/// `quad(a,c,b)` + 收尾 `quad(b,d,a)` + close ⇒ 闭合曲带。中段厚度
/// ≈ 两弧间隙，两端尖灭——端点须落在既有轮廓内（或被 bulge 覆盖）
/// 以免露接缝。
fn band(ax: f32, ay: f32, cx: f32, cy: f32, bx: f32, by: f32, dx: f32, dy: f32) -> OutlineContour {
    vec![on(ax, ay), off(cx, cy), on(bx, by), off(dx, dy)]
}

/// 等粗谷形波（≈ / ∼ 的波长线）：从 `(x0, y_hi)` 陷至 `(xm, y_lo)`
/// 再回升到 `(x1, y_hi)`，上下缘平行（端帽与谷底厚度恒为 `t`）。
///
/// 8 点 4 二次弧：上缘三段（左缓坡 → 谷底 → 右缓坡）+ 下缘镜像。
fn tilde(x0: f32, y_hi: f32, x1: f32, y_lo: f32, t: f32) -> OutlineContour {
    let xm = (x0 + x1) * 0.5;
    let d = (x1 - x0) * 0.32; // 坡段宽度占比（控制坡度陡缓）
    vec![
        on(x0, y_hi - t * 0.5),
        off(x0 + d, y_hi - t * 0.5),
        on(xm, y_lo - t * 0.5),
        off(x1 - d, y_hi - t * 0.5),
        on(x1, y_hi - t * 0.5),
        off(x1 - d, y_hi + t * 0.5),
        on(xm, y_lo + t * 0.5),
        off(x0 + d, y_hi + t * 0.5),
    ]
}

// ---------------------------------------------------------------------------
// 符号构造（每符号：轮廓列表；构建后统一方向归一化）
// ---------------------------------------------------------------------------

/// 构造字符的填充轮廓（字体单位）。
fn build_outline(ch: char) -> Outline {
    let mut c: Vec<OutlineContour> = Vec::new();
    match ch {
        // ---- 希腊小写 ----
        // α：左环 + 右下尾。
        'α' => {
            c.push(ellipse(230.0, 250.0, 150.0, 250.0));
            c.push(band(340.0, 400.0, 470.0, 220.0, 530.0, -40.0, 380.0, 90.0));
        }
        // β：主干 + 上下两右碗。
        'β' => {
            c.push(seg(130.0, -60.0, 220.0, 720.0, 95.0));
            c.push(ellipse(330.0, 150.0, 190.0, 150.0));
            c.push(ellipse(330.0, 440.0, 190.0, 140.0));
        }
        // γ：顶弧 + 竖身 + 基线下左尾。
        'γ' => {
            c.push(band(130.0, 620.0, 330.0, 690.0, 400.0, 480.0, 250.0, 520.0));
            c.push(seg(390.0, 470.0, 385.0, -40.0, 95.0));
            c.push(band(385.0, -40.0, 300.0, -170.0, 150.0, -140.0, 330.0, -60.0));
        }
        // δ：圆身 + 右上挑笔（ascender 起笔探出圆顶右缘）。
        'δ' => {
            c.push(ellipse(260.0, 250.0, 210.0, 250.0));
            c.push(band(350.0, 420.0, 460.0, 520.0, 510.0, 560.0, 400.0, 470.0));
        }
        // ε：中横杠 + 上下臂 + 右 bulge（"反 3 加杠"形，与 σ 明确可辨）。
        'ε' => {
            c.push(seg(60.0, 255.0, 430.0, 345.0, 95.0)); // 中杠（斜上右行）
            c.push(band(90.0, 470.0, 250.0, 570.0, 420.0, 355.0, 190.0, 420.0)); // 上臂
            c.push(band(90.0, 130.0, 250.0, 30.0, 420.0, 245.0, 190.0, 180.0)); // 下臂
            c.push(ellipse(435.0, 300.0, 48.0, 108.0)); // 右端汇合 bulge
        }
        // θ：圆身 + 中横杠。
        'θ' => {
            c.push(ellipse(280.0, 270.0, 230.0, 270.0));
            c.push(seg(50.0, 235.0, 510.0, 305.0, 80.0));
        }
        // λ：左斜下 + 右斜上（顶点在基线）。
        'λ' => {
            c.push(seg(130.0, 700.0, 250.0, 0.0, 90.0));
            c.push(seg(250.0, 0.0, 470.0, 700.0, 90.0));
        }
        // μ：左竖 + 底弧 + 右竖探基线下。
        'μ' => {
            c.push(seg(150.0, 500.0, 150.0, 60.0, 90.0));
            c.push(band(150.0, 60.0, 260.0, 20.0, 370.0, 60.0, 260.0, 130.0));
            c.push(seg(370.0, 500.0, 370.0, -180.0, 90.0));
        }
        // π：双腿 + 顶横。
        'π' => {
            c.push(seg(150.0, 0.0, 150.0, 480.0, 85.0));
            c.push(seg(450.0, 0.0, 450.0, 480.0, 85.0));
            c.push(seg(90.0, 480.0, 510.0, 560.0, 85.0));
        }
        // σ：顶弧 + 右竖 + 底弧（与 ε 的"杠+臂"形明确可辨）。
        'σ' => {
            c.push(band(480.0, 460.0, 400.0, 540.0, 200.0, 500.0, 300.0, 440.0));
            c.push(seg(465.0, 460.0, 465.0, 80.0, 90.0));
            c.push(band(465.0, 80.0, 400.0, 20.0, 200.0, 50.0, 330.0, 130.0));
        }
        // φ：圆身 + 贯穿竖笔（探基线下）。
        'φ' => {
            c.push(ellipse(280.0, 300.0, 230.0, 300.0));
            c.push(seg(250.0, 720.0, 310.0, -160.0, 80.0));
        }
        // ψ：竖笔 + 左右两碗（三叉戟）。
        'ψ' => {
            c.push(seg(255.0, 720.0, 305.0, -160.0, 80.0));
            c.push(band(270.0, 520.0, 110.0, 520.0, 100.0, 260.0, 210.0, 300.0));
            c.push(band(100.0, 260.0, 120.0, 140.0, 280.0, 200.0, 190.0, 240.0));
            c.push(band(290.0, 520.0, 450.0, 520.0, 460.0, 260.0, 350.0, 300.0));
            c.push(band(460.0, 260.0, 440.0, 140.0, 280.0, 200.0, 370.0, 240.0));
        }
        // ω：双碗底连（W 形收口）。
        'ω' => {
            c.push(band(90.0, 480.0, 60.0, 180.0, 210.0, 140.0, 150.0, 200.0));
            c.push(band(210.0, 140.0, 280.0, 40.0, 350.0, 140.0, 280.0, 230.0));
            c.push(band(350.0, 140.0, 500.0, 180.0, 430.0, 480.0, 390.0, 200.0));
        }
        // ---- 希腊大写 ----
        // Γ：顶横 + 左竖。
        'Γ' => {
            c.push(seg(100.0, 620.0, 480.0, 700.0, 85.0));
            c.push(seg(100.0, 0.0, 180.0, 700.0, 85.0));
        }
        // Δ：实心三角。
        'Δ' => {
            c.push(vec![on(120.0, 0.0), on(280.0, 700.0), on(440.0, 0.0)]);
        }
        // Θ：圆身 + 中横杠。
        'Θ' => {
            c.push(ellipse(280.0, 350.0, 230.0, 350.0));
            c.push(seg(50.0, 315.0, 510.0, 385.0, 80.0));
        }
        // Λ：双斜 + 底横。
        'Λ' => {
            c.push(seg(120.0, 700.0, 280.0, 0.0, 85.0));
            c.push(seg(280.0, 0.0, 440.0, 700.0, 85.0));
            c.push(seg(150.0, 0.0, 410.0, 60.0, 60.0));
        }
        // Ξ：三横。
        'Ξ' => {
            c.push(seg(120.0, 620.0, 460.0, 690.0, 80.0));
            c.push(seg(150.0, 330.0, 430.0, 400.0, 80.0));
            c.push(seg(120.0, 10.0, 460.0, 80.0, 80.0));
        }
        // Π：双腿 + 顶横。
        'Π' => {
            c.push(seg(110.0, 0.0, 190.0, 640.0, 85.0));
            c.push(seg(370.0, 0.0, 450.0, 640.0, 85.0));
            c.push(seg(80.0, 620.0, 480.0, 700.0, 85.0));
        }
        // Σ：实心 Z 形。
        'Σ' => {
            c.push(vec![
                on(120.0, 700.0),
                on(450.0, 700.0),
                on(220.0, 380.0),
                on(450.0, 0.0),
                on(120.0, 0.0),
                on(350.0, 380.0),
            ]);
        }
        // Φ：圆身 + 贯穿竖笔。
        'Φ' => {
            c.push(ellipse(280.0, 350.0, 240.0, 350.0));
            c.push(seg(250.0, 700.0, 310.0, 0.0, 80.0));
        }
        // Ψ：竖笔 + 左右两上翘碗。
        'Ψ' => {
            c.push(seg(250.0, 700.0, 310.0, 0.0, 80.0));
            c.push(band(270.0, 560.0, 70.0, 540.0, 80.0, 300.0, 210.0, 320.0));
            c.push(band(80.0, 300.0, 110.0, 180.0, 290.0, 220.0, 190.0, 260.0));
            c.push(band(290.0, 560.0, 490.0, 540.0, 480.0, 300.0, 350.0, 320.0));
            c.push(band(480.0, 300.0, 450.0, 180.0, 270.0, 220.0, 370.0, 260.0));
        }
        // Ω：马蹄 + 两外撇足。
        'Ω' => {
            c.push(band(100.0, 600.0, 60.0, 250.0, 180.0, 90.0, 170.0, 220.0));
            c.push(band(180.0, 90.0, 280.0, 40.0, 380.0, 90.0, 280.0, 170.0));
            c.push(band(380.0, 90.0, 500.0, 250.0, 400.0, 600.0, 390.0, 220.0));
            c.push(seg(60.0, 600.0, 180.0, 660.0, 70.0));
            c.push(seg(340.0, 600.0, 460.0, 660.0, 70.0));
        }
        // ---- 数学运算符 ----
        // ∂：右竖弧 + 左下碗（反 6 形）。
        '∂' => {
            c.push(band(390.0, 700.0, 390.0, 400.0, 360.0, 180.0, 330.0, 330.0));
            c.push(band(360.0, 180.0, 200.0, 60.0, 140.0, 220.0, 280.0, 140.0));
            c.push(band(140.0, 220.0, 90.0, 400.0, 240.0, 480.0, 180.0, 320.0));
            c.push(band(240.0, 480.0, 330.0, 520.0, 390.0, 460.0, 320.0, 420.0));
        }
        // √：左下斜 + 顶横 + 右短竖。
        '√' => {
            c.push(seg(60.0, 60.0, 200.0, 560.0, 90.0));
            c.push(seg(180.0, 540.0, 470.0, 610.0, 80.0));
            c.push(seg(470.0, 610.0, 470.0, 470.0, 80.0));
        }
        // ∫：高挑 S 形（三段曲带，底探基线下）。
        '∫' => {
            c.push(band(330.0, 720.0, 140.0, 640.0, 180.0, 420.0, 270.0, 480.0));
            c.push(band(180.0, 420.0, 220.0, 200.0, 360.0, 140.0, 270.0, 260.0));
            c.push(band(360.0, 140.0, 500.0, 60.0, 430.0, -140.0, 450.0, 60.0));
        }
        // ∑：大 sigma（实心 Z，加高版）。
        '∑' => {
            c.push(vec![
                on(120.0, 700.0),
                on(480.0, 700.0),
                on(250.0, 350.0),
                on(480.0, 0.0),
                on(120.0, 0.0),
                on(350.0, 350.0),
            ]);
        }
        // ∏：大 pi（双腿 + 顶横，加高版）。
        '∏' => {
            c.push(seg(120.0, 0.0, 190.0, 640.0, 85.0));
            c.push(seg(370.0, 0.0, 440.0, 640.0, 85.0));
            c.push(seg(90.0, 620.0, 470.0, 720.0, 100.0));
        }
        // ±：中横 + 上十字（竖杠贯穿中横）。
        '±' => {
            c.push(seg(100.0, 300.0, 420.0, 370.0, 75.0));
            c.push(seg(230.0, 160.0, 290.0, 510.0, 75.0));
            c.push(seg(100.0, 60.0, 420.0, 130.0, 75.0));
        }
        // ∓：上横 + 下十字。
        '∓' => {
            c.push(seg(100.0, 500.0, 420.0, 570.0, 75.0));
            c.push(seg(230.0, 100.0, 290.0, 450.0, 75.0));
            c.push(seg(100.0, 240.0, 420.0, 310.0, 75.0));
        }
        // ×：双交叉斜笔。
        '×' => {
            c.push(seg(140.0, 580.0, 400.0, 80.0, 75.0));
            c.push(seg(140.0, 80.0, 400.0, 580.0, 75.0));
        }
        // ÷：上横 + 上下点。
        '÷' => {
            c.push(seg(100.0, 480.0, 420.0, 550.0, 75.0));
            c.push(dot(260.0, 650.0, 55.0));
            c.push(dot(260.0, 130.0, 55.0));
        }
        // ⋅：居中点。
        '⋅' => {
            c.push(dot(260.0, 300.0, 60.0));
        }
        // ≈：双波浪（上下两行等粗谷形）。
        '≈' => {
            c.push(tilde(110.0, 420.0, 240.0, 350.0, 70.0));
            c.push(tilde(370.0, 420.0, 500.0, 350.0, 70.0));
            c.push(tilde(110.0, 260.0, 240.0, 190.0, 70.0));
            c.push(tilde(370.0, 260.0, 500.0, 190.0, 70.0));
        }
        // ≅：波浪 + 双横。
        '≅' => {
            c.push(tilde(110.0, 560.0, 500.0, 490.0, 70.0));
            c.push(seg(110.0, 300.0, 470.0, 360.0, 70.0));
            c.push(seg(110.0, 180.0, 470.0, 240.0, 70.0));
        }
        // ∼：单波浪。
        '∼' => {
            c.push(tilde(110.0, 400.0, 500.0, 330.0, 70.0));
        }
        // ≠：双横 + 反斜杠。
        '≠' => {
            c.push(seg(110.0, 380.0, 410.0, 445.0, 75.0));
            c.push(seg(110.0, 215.0, 410.0, 280.0, 75.0));
            c.push(seg(150.0, 560.0, 390.0, 90.0, 75.0));
        }
        // ≡：三横。
        '≡' => {
            c.push(seg(110.0, 480.0, 410.0, 545.0, 70.0));
            c.push(seg(110.0, 315.0, 410.0, 380.0, 70.0));
            c.push(seg(110.0, 150.0, 410.0, 215.0, 70.0));
        }
        // ≤：左尖 + 底横。
        '≤' => {
            c.push(seg(450.0, 580.0, 160.0, 330.0, 75.0));
            c.push(seg(160.0, 330.0, 450.0, 80.0, 75.0));
            c.push(seg(120.0, 230.0, 460.0, 300.0, 75.0));
        }
        // ≥：右尖 + 底横。
        '≥' => {
            c.push(seg(130.0, 580.0, 420.0, 330.0, 75.0));
            c.push(seg(420.0, 330.0, 130.0, 80.0, 75.0));
            c.push(seg(120.0, 230.0, 460.0, 300.0, 75.0));
        }
        // ≪：双左尖。
        '≪' => {
            c.push(seg(500.0, 580.0, 260.0, 330.0, 70.0));
            c.push(seg(260.0, 330.0, 500.0, 80.0, 70.0));
            c.push(seg(260.0, 580.0, 40.0, 330.0, 70.0));
            c.push(seg(40.0, 330.0, 260.0, 80.0, 70.0));
        }
        // →：箭杆 + 实心箭头。
        '→' => {
            c.push(seg(60.0, 265.0, 400.0, 335.0, 80.0));
            c.push(vec![on(400.0, 420.0), on(520.0, 300.0), on(400.0, 180.0)]);
        }
        // ←：镜像。
        '←' => {
            c.push(seg(160.0, 265.0, 500.0, 335.0, 80.0));
            c.push(vec![on(160.0, 420.0), on(40.0, 300.0), on(160.0, 180.0)]);
        }
        // ↑：竖杆 + 上箭头。
        '↑' => {
            c.push(seg(220.0, 700.0, 280.0, 60.0, 80.0));
            c.push(vec![on(120.0, 420.0), on(250.0, 620.0), on(380.0, 420.0)]);
        }
        // ↓：竖杆 + 下箭头。
        '↓' => {
            c.push(seg(220.0, 0.0, 280.0, 640.0, 80.0));
            c.push(vec![on(120.0, 280.0), on(250.0, 80.0), on(380.0, 280.0)]);
        }
        // ↔：左右双箭。
        '↔' => {
            c.push(seg(180.0, 265.0, 480.0, 335.0, 75.0));
            c.push(vec![on(480.0, 430.0), on(600.0, 300.0), on(480.0, 170.0)]);
            c.push(vec![on(180.0, 430.0), on(60.0, 300.0), on(180.0, 170.0)]);
        }
        // ⇒：双杆 + 大箭头。
        '⇒' => {
            c.push(seg(60.0, 240.0, 380.0, 285.0, 70.0));
            c.push(seg(60.0, 345.0, 380.0, 390.0, 70.0));
            c.push(vec![on(380.0, 470.0), on(560.0, 315.0), on(380.0, 160.0)]);
        }
        // ⇔：双杆 + 双箭头。
        '⇔' => {
            c.push(seg(220.0, 240.0, 540.0, 285.0, 70.0));
            c.push(seg(220.0, 345.0, 540.0, 390.0, 70.0));
            c.push(vec![on(540.0, 470.0), on(720.0, 315.0), on(540.0, 160.0)]);
            c.push(vec![on(220.0, 470.0), on(40.0, 315.0), on(220.0, 160.0)]);
        }
        // ∈：ε 形右 bulge + 探出左中杠。
        '∈' => {
            c.push(seg(50.0, 255.0, 320.0, 345.0, 95.0));
            c.push(band(330.0, 460.0, 440.0, 500.0, 440.0, 360.0, 400.0, 420.0));
            c.push(band(330.0, 140.0, 440.0, 100.0, 440.0, 240.0, 400.0, 180.0));
            c.push(ellipse(445.0, 300.0, 48.0, 108.0));
        }
        // ∪：双腿 + 底弧。
        '∪' => {
            c.push(seg(130.0, 480.0, 130.0, 200.0, 85.0));
            c.push(seg(390.0, 480.0, 390.0, 200.0, 85.0));
            c.push(band(130.0, 200.0, 260.0, 60.0, 390.0, 200.0, 260.0, 280.0));
        }
        // ∩：镜像（双腿 + 顶弧）。
        '∩' => {
            c.push(seg(130.0, 0.0, 130.0, 280.0, 85.0));
            c.push(seg(390.0, 0.0, 390.0, 280.0, 85.0));
            c.push(band(130.0, 280.0, 260.0, 420.0, 390.0, 280.0, 260.0, 200.0));
        }
        // ∅：圆 + 反斜杠。
        '∅' => {
            c.push(ellipse(280.0, 300.0, 220.0, 280.0));
            c.push(seg(140.0, 520.0, 430.0, 90.0, 80.0));
        }
        // ∀：顶弧 + 双腿。
        '∀' => {
            c.push(band(110.0, 500.0, 260.0, 640.0, 410.0, 500.0, 260.0, 420.0));
            c.push(seg(110.0, 500.0, 240.0, 0.0, 80.0));
            c.push(seg(410.0, 500.0, 280.0, 0.0, 80.0));
        }
        // ∃：左竖 + 两右挑臂。
        '∃' => {
            c.push(seg(150.0, 0.0, 150.0, 700.0, 80.0));
            c.push(band(150.0, 560.0, 290.0, 540.0, 380.0, 470.0, 270.0, 480.0));
            c.push(band(150.0, 220.0, 290.0, 200.0, 380.0, 130.0, 270.0, 160.0));
        }
        // ¬：横杠 + 右下斜踢。
        '¬' => {
            c.push(seg(110.0, 300.0, 410.0, 375.0, 80.0));
            c.push(seg(360.0, 300.0, 430.0, 130.0, 75.0));
        }
        // ∧：双斜上汇。
        '∧' => {
            c.push(seg(130.0, 120.0, 260.0, 480.0, 75.0));
            c.push(seg(260.0, 480.0, 390.0, 120.0, 75.0));
        }
        // ∨：双斜下汇。
        '∨' => {
            c.push(seg(130.0, 480.0, 260.0, 120.0, 75.0));
            c.push(seg(260.0, 120.0, 390.0, 480.0, 75.0));
        }
        // ∞：双椭圆横八。
        '∞' => {
            c.push(ellipse(180.0, 300.0, 150.0, 200.0));
            c.push(ellipse(400.0, 300.0, 150.0, 200.0));
        }
        // ∝：双点 + 长波。
        '∝' => {
            c.push(dot(170.0, 470.0, 55.0));
            c.push(dot(170.0, 140.0, 55.0));
            c.push(tilde(240.0, 400.0, 630.0, 330.0, 70.0));
        }
        // −：减号（U+2212）。
        '−' => {
            c.push(seg(100.0, 265.0, 420.0, 335.0, 80.0));
        }
        // ---- 上标 / 度号 ----
        // ¹：升 1。
        '¹' => {
            c.push(seg(150.0, 380.0, 200.0, 480.0, 55.0));
            c.push(seg(200.0, 480.0, 290.0, 480.0, 55.0));
            c.push(seg(290.0, 480.0, 290.0, 380.0, 55.0));
        }
        // ²：升 2（顶弧 + 斜落 + 底横）。
        '²' => {
            c.push(band(150.0, 520.0, 180.0, 620.0, 280.0, 620.0, 220.0, 560.0));
            c.push(band(280.0, 620.0, 340.0, 560.0, 280.0, 470.0, 320.0, 520.0));
            c.push(seg(280.0, 470.0, 160.0, 380.0, 60.0));
            c.push(seg(150.0, 340.0, 320.0, 400.0, 60.0));
        }
        // ³：升 3（双右弧）。
        '³' => {
            c.push(band(150.0, 520.0, 180.0, 620.0, 280.0, 620.0, 220.0, 560.0));
            c.push(band(280.0, 620.0, 340.0, 560.0, 270.0, 490.0, 320.0, 540.0));
            c.push(band(270.0, 490.0, 330.0, 440.0, 220.0, 380.0, 290.0, 430.0));
        }
        // °：度号（升位小圆）。
        '°' => {
            c.push(dot(230.0, 620.0, 130.0));
        }
        _ => {} // 表外字符（不会发生：入口经 symbol_char 过滤）
    }
    normalize_orientation(&mut c);
    Outline { contours: c }
}

/// 轮廓方向归一化：全部转为**正方向**（y 向上系 CCW，shoelace 面积 > 0）。
///
/// 非零环绕填充下同向重叠取并集；方向冲突的重叠区会抵消成"镂空"
/// （如 θ 横杠 vs 圆身、± 竖杠 vs 横杠）。曲线轮廓以原始点列做
/// shoelace（曲率不改变简单形状的方向符号）。
fn normalize_orientation(contours: &mut [OutlineContour]) {
    for ring in contours.iter_mut() {
        if signed_area(ring) < 0.0 {
            ring.reverse();
        }
    }
}

/// 轮廓有向面积（shoelace；空环或点列不足 → 0.0，不动）。
fn signed_area(ring: &OutlineContour) -> f32 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0f32;
    for i in 0..n {
        let p = ring[i];
        let q = ring[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;
    use crate::raster::render_glyph;

    /// 表结构：字符唯一、advance 合理、下标不溢合成 gid 空间。
    #[test]
    fn table_is_well_formed() {
        assert!(SYMBOLS.len() < 256, "表容量须 < 256（合成 gid 空间）");
        let mut seen = Vec::new();
        for (i, &(ch, adv)) in SYMBOLS.iter().enumerate() {
            assert!(!seen.contains(&ch), "字符重复：{ch}");
            seen.push(ch);
            assert!(adv > 0 && adv <= 1000, "{ch} advance 越界：{adv}");
            assert_eq!(fallback_gid(ch), Some(MATH_GID_BASE + i as u16), "{ch} gid 映射");
        }
    }

    /// 查表往返：gid ↔ 字符 ↔ advance ↔ 轮廓 全链路一致。
    #[test]
    fn lookup_roundtrip() {
        for &(ch, adv) in SYMBOLS {
            let gid = fallback_gid(ch).expect("表内字符必须有 gid");
            assert_eq!(symbol_char(gid), Some(ch));
            assert_eq!(advance_for_gid(gid), Some(adv));
            assert!(outline_for_gid(gid).is_some(), "{ch} 轮廓必须可构造");
        }
        // 表外 / 非合成 gid 一律 None（不 panic）。
        assert_eq!(fallback_gid('中'), None);
        assert_eq!(fallback_gid('A'), None);
        assert_eq!(symbol_char(MATH_GID_BASE - 1), None);
        assert_eq!(advance_for_gid(0), None);
        assert_eq!(outline_for_gid(u16::MAX), None);
        assert!(!is_math_gid(3));
    }

    /// 合成 gid 与真实字形恒不相交（内嵌字体 6889 字形）。
    #[test]
    fn synthetic_gids_are_disjoint_from_real_font() {
        let f = font_regular();
        assert!(f.num_glyphs() < MATH_GID_BASE as u32, "真实字形数须远小于合成基址");
        for &(ch, _) in SYMBOLS {
            let gid = fallback_gid(ch).unwrap();
            assert!(gid as u32 >= f.num_glyphs(), "{ch} 合成 gid 不得落入真实字形区间");
        }
    }

    /// 全部符号：轮廓非空、坐标有限、方向一致、落坐标纪律带。
    #[test]
    fn all_outlines_well_formed_and_oriented() {
        for &(ch, _) in SYMBOLS {
            let out = build_outline(ch);
            assert!(!out.contours.is_empty(), "{ch} 必须有轮廓");
            let (mut x_lo, mut x_hi) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut y_lo, mut y_hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for ring in &out.contours {
                assert!(ring.len() >= 3, "{ch} 每轮廓至少 3 点");
                assert!(
                    signed_area(ring) > 0.0,
                    "{ch} 轮廓方向必须归一为正（防抵消镂空），面积 {}",
                    signed_area(ring)
                );
                for p in ring {
                    assert!(p.x.is_finite() && p.y.is_finite(), "{ch} 坐标必须有限");
                    x_lo = x_lo.min(p.x);
                    x_hi = x_hi.max(p.x);
                    y_lo = y_lo.min(p.y);
                    y_hi = y_hi.max(p.y);
                }
            }
            assert!(x_lo >= -1e-3, "{ch} 不得左悬挂：x_lo={x_lo}");
            assert!(x_hi <= 780.0, "{ch} 右缘越界：x_hi={x_hi}");
            assert!(y_lo >= -220.0, "{ch} 降部越界：y_lo={y_lo}");
            assert!(y_hi <= 820.0, "{ch} 升部越界：y_hi={y_hi}");
        }
    }

    /// 渲染集成：每个符号经 `render_glyph`（含回退钩子）产出非空墨迹。
    #[test]
    fn every_symbol_renders_visible_ink() {
        let f = font_regular();
        for &(ch, _) in SYMBOLS {
            let gid = fallback_gid(ch).expect("表内字符");
            let bmp = render_glyph(&f, gid, 48.0);
            assert!(!bmp.is_empty(), "{ch} 必须渲染出非空位图");
            let ink: u32 = bmp.alpha.iter().map(|&a| a as u32).sum();
            assert!(ink > 0, "{ch} 必须含可见墨迹（alpha 和 > 0）");
            // 确定性：两次渲染逐字节一致。
            let again = render_glyph(&f, gid, 48.0);
            assert_eq!(bmp.alpha, again.alpha, "{ch} 渲染必须确定");
        }
    }

    /// 位图互异：新符号之间、与 ASCII 参考符号之间必须可辨。
    ///
    /// 测试要求：「每个新符号渲染产出非空像素且与参考符号不同」。
    #[test]
    fn symbol_bitmaps_are_mutually_distinct() {
        let f = font_regular();
        let bmp_of = |gid: u16| render_glyph(&f, gid, 48.0).alpha;
        // ASCII 参考集（'=' 正是审计缺陷中的对照字符）。
        let ascii: Vec<(char, Vec<u8>)> = ['=', 'e', 'o', 'S', '3', '+']
            .iter()
            .map(|&c| (c, bmp_of(f.glyph_index(c).unwrap())))
            .collect();
        let mut prev: Option<(char, Vec<u8>)> = None;
        for &(ch, _) in SYMBOLS {
            let gid = fallback_gid(ch).unwrap();
            let mine = bmp_of(gid);
            for (ac, abmp) in &ascii {
                assert!(
                    mine != *abmp,
                    "{ch} 的位图不得与 ASCII '{ac}' 相同（豆腐块/错形回归）"
                );
            }
            if let Some((pch, pbmp)) = &prev {
                assert!(mine != *pbmp, "{ch} 与前一符号 {pch} 位图相同");
            }
            prev = Some((ch, mine));
        }
    }

    /// ε 与 σ 必须可辨（审计缺陷「ε 渲染成 σ」的回归锁）。
    #[test]
    fn epsilon_is_distinct_from_sigma() {
        let f = font_regular();
        let eps = fallback_gid('ε').unwrap();
        let sig = fallback_gid('σ').unwrap();
        let a = render_glyph(&f, eps, 48.0);
        let b = render_glyph(&f, sig, 48.0);
        assert_ne!(a.alpha, b.alpha, "ε 与 σ 位图必须不同");
        // ε 有明确中横杠：中部行应存在接近满覆盖的长连续段。
        let mid_y = (a.height / 2) as usize;
        let row = &a.alpha[mid_y as usize * a.width as usize..(mid_y + 1) * a.width as usize];
        let filled = row.iter().filter(|&&v| v > 128).count();
        assert!(
            filled as u32 > a.width / 2,
            "ε 中部应有贯穿横杠（实测 {filled}/{}）",
            a.width
        );
    }

    /// 排版集成：`layout_text` 对 "ε=0.41" 的 gid / advance 落位。
    #[test]
    fn layout_uses_fallback_gid_and_advance() {
        use crate::layout::{layout_text, Align, TextParams};
        let f = font_regular();
        let params = TextParams {
            font_size: 40.0,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Left,
        };
        let l = layout_text(&f, "ε=0.41", &params);
        assert_eq!(l.glyphs.len(), 6); // ε = 0 . 4 1
        // ε → 合成 gid；'=' → 真实 gid（字体已覆盖，不走回退）。
        assert_eq!(l.glyphs[0].gid, fallback_gid('ε').unwrap());
        assert_eq!(l.glyphs[1].gid, f.glyph_index('=').unwrap());
        assert!(!is_math_gid(l.glyphs[1].gid), "'=' 是真实字形");
        // ε 的 advance = 表内 units × scale（非 hmtx、非 .notdef 的 1000）。
        let eps_adv = advance_for_gid(l.glyphs[0].gid).unwrap() as f32 * (40.0 / 1000.0);
        assert!((l.glyphs[0].advance - eps_adv).abs() < 1e-4, "ε advance 须走回退表");
        assert!((l.glyphs[1].x - eps_adv).abs() < 1e-4, "后字符笔位紧随 ε advance");
        // 全宽有限且为正。
        assert!(l.width.is_finite() && l.width > 0.0);
    }

    /// 已覆盖字符零影响：'中' / 'A' / '=' 的 gid 不走回退路径。
    #[test]
    fn covered_chars_are_unaffected() {
        use crate::layout::{layout_text, Align, TextParams};
        let f = font_regular();
        let params = TextParams {
            font_size: 32.0,
            max_width: f32::INFINITY,
            align: Align::Left,
            ..TextParams::default()
        };
        let l = layout_text(&f, "中A=0", &params);
        for g in &l.glyphs {
            assert!(!is_math_gid(g.gid), "已覆盖字符不得走合成 gid：{:?}", g.ch);
        }
    }

    /// 缺失码位诊断（回退链末端；D-17）：三段式 + U+XXXX 清单。
    #[test]
    fn missing_codepoint_error_lists_codepoints_and_chain() {
        // 单码位：含具体 U+XXXX 与字符本身。
        let e = missing_codepoint_error(&['☃'], &["multiscript.ttf", "<embedded:PrismSansSC-Regular>"]);
        assert_eq!(e.code, "T009");
        assert!(e.problem.contains("U+2603"), "应含缺失码位：{}", e.problem);
        assert!(e.problem.contains('☃'), "应含缺失字符本身：{}", e.problem);
        assert!(e.problem.contains("multiscript.ttf"), "应列出声明链：{}", e.problem);
        assert!(e.problem.contains("程序化符号表亦未覆盖"), "应说明末端也未覆盖：{}", e.problem);
        assert!(!e.fix.is_empty(), "三段式必含建议段");
        // 多码位：按传入序逐列出（BMP 外码位 5 位十六进制，标准记法）。
        let e2 = missing_codepoint_error(&['☃', '😀'], &["a.ttf"]);
        assert!(e2.problem.contains("U+2603"), "{}", e2.problem);
        assert!(e2.problem.contains("U+1F600"), "BMP 外码位记法：{}", e2.problem);
        // 空链如实说明（不伪装成"有字体但缺字形"）。
        let e3 = missing_codepoint_error(&['☃'], &[]);
        assert!(e3.problem.contains("空链"), "{}", e3.problem);
    }

    /// 符号表单位基准：与内嵌 Regular upem 一致（合成字形缩放基准）。
    #[test]
    fn math_units_per_em_matches_embedded_font() {
        assert_eq!(MATH_UNITS_PER_EM, 1000);
        assert_eq!(font_regular().units_per_em() as u16, MATH_UNITS_PER_EM);
    }
}
