//! V2-j：`formula` 块端到端（DSL parse → codegen → 记法解析 → 字形布局）。
//!
//! 要求（规格 §3 / §4）：
//! - `symbols.rs` 的公式表达能经 `formula` 块**等价重写**，覆盖 ≥ 5 类结构；
//! - 每类结构都走完整链路并**渲染出字**（字形序列 / 布局输出验证）；
//! - 端到端 = `compile_v2`（parser + codegen）→
//!   `prism_text::math_ast::MathExpr::parse`（记法 → AST）→
//!   `prism_text::math_layout::layout_math`（AST → 字形）。

use prism_dsl::{compile_v2, CompileV2Output, FormulaAlign, FormulaNode};
use prism_text::math_ast::MathExpr;
use prism_text::math_layout::{MathLayout, RuleKind};

/// 带 formula 块的 v2 文档骨架。
fn doc(formula_blocks: &str) -> String {
    format!(
        "// prism-dsl 1\nmovie \"公式\" {{ duration 4s fps 30 }}\nscene \"s\" {{ }}\n{}",
        formula_blocks
    )
}

/// 编译并取出全部公式节点。
fn compile_formulas(formula_blocks: &str) -> Vec<FormulaNode> {
    let out: CompileV2Output = compile_v2(&doc(formula_blocks)).expect("文档须编译成功");
    out.formulas
}

// ---------------------------------------------------------------------------
// 1. parser + codegen：字段解析与参数归一化
// ---------------------------------------------------------------------------

/// 全字段块：notation / size / x / y / color / align 都正确落到 `FormulaNode`。
#[test]
fn formula_block_full_fields() {
    let blocks: &str = r#"
formula "dydx" {
  notation "\frac{dy}{dx} = 2x"
  size 34
  x 480
  y 300
  color #204060
  align left
}
"#;
    let f: Vec<FormulaNode> = compile_formulas(blocks);
    assert_eq!(f.len(), 1, "一个 formula 块 → 一个公式节点");
    let n: &FormulaNode = &f[0];
    assert_eq!(n.name, "dydx");
    assert_eq!(n.notation, r"\frac{dy}{dx} = 2x", "记法原文须逐字保真（反斜杠不被词法吃掉）");
    assert!((n.size - 34.0).abs() < 1e-6);
    assert!((n.x - 480.0).abs() < 1e-6);
    assert!((n.y - 300.0).abs() < 1e-6);
    assert!((n.color[0] - 0x20 as f32 / 255.0).abs() < 1e-6, "颜色归一化到 0..1");
    assert!((n.color[1] - 0x40 as f32 / 255.0).abs() < 1e-6);
    assert!((n.color[2] - 0x60 as f32 / 255.0).abs() < 1e-6);
    assert!((n.color[3] - 1.0).abs() < 1e-6, "alpha 缺省 1");
    assert_eq!(n.align, FormulaAlign::Left);
    assert_eq!(n.line, 5, "行号须指向 formula 关键字所在行（骨架 3 行 + 空行）");
}

/// 缺省参数：size 32 / x 0 / y 0 / 白色 / center；`px` 长度与 `center` 字面均可。
#[test]
fn formula_block_defaults_and_px_units() {
    let blocks: &str = r#"
formula "bare" {
  notation "x^2"
}
formula "px" {
  notation "y_1"
  size 24px
  x 100px
  align center
}
"#;
    let f: Vec<FormulaNode> = compile_formulas(blocks);
    assert_eq!(f.len(), 2);
    assert_eq!(f[0].size, 32.0, "缺省字号 32");
    assert_eq!(f[0].x, 0.0);
    assert_eq!(f[0].y, 0.0);
    assert_eq!(f[0].color, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(f[0].align, FormulaAlign::Center);
    assert_eq!(f[1].size, 24.0, "px 长度归一化为 px 数值");
    assert_eq!(f[1].x, 100.0);
    assert_eq!(f[1].align, FormulaAlign::Center);
}

/// `left_x` 按对齐方式把锚点折算为左缘（渲染侧定位入口）。
#[test]
fn formula_left_x_by_align() {
    let n: FormulaNode = FormulaNode {
        name: "n".to_string(),
        notation: "x".to_string(),
        size: 32.0,
        x: 500.0,
        y: 200.0,
        color: [1.0, 1.0, 1.0, 1.0],
        align: FormulaAlign::Center,
        line: 1,
    };
    assert!((n.left_x(200.0) - 400.0).abs() < 1e-6, "center → x − w/2");
    let mut right: FormulaNode = n.clone();
    right.align = FormulaAlign::Right;
    assert!((right.left_x(200.0) - 300.0).abs() < 1e-6, "right → x − w");
    let mut left: FormulaNode = n;
    left.align = FormulaAlign::Left;
    assert!((left.left_x(200.0) - 500.0).abs() < 1e-6, "left → x");
}

/// 诊断：缺 notation / 未知字段 / 非法 align / 空名 / 重名 —— 均带行号 + [Pxxxx]。
#[test]
fn formula_block_diagnostics() {
    // 缺 notation（第 4 行起块）。
    let err: prism_dsl::DslError = compile_v2(&doc("formula \"no_notation\" {\n  size 30\n}\n"))
        .expect_err("缺 notation 须报错");
    assert!(err.message.contains("[P2107]"), "消息 {}", err.message);
    assert!(err.message.contains("notation"), "消息须点名缺的字段");
    assert_eq!(err.line, 4, "行号须指向 formula 块");

    // 未知字段。
    let err: prism_dsl::DslError = compile_v2(&doc("formula \"x\" {\n  notation \"x\"\n  bold true\n}\n"))
        .expect_err("未知字段须报错");
    assert!(err.message.contains("[P1002]"), "消息 {}", err.message);
    assert!(err.message.contains("bold"));

    // 非法 align。
    let err: prism_dsl::DslError =
        compile_v2(&doc("formula \"x\" {\n  notation \"x\"\n  align middle\n}\n"))
            .expect_err("非法 align 须报错");
    assert!(err.message.contains("left / center / right"), "消息 {}", err.message);

    // 非正 size。
    let err: prism_dsl::DslError = compile_v2(&doc("formula \"x\" {\n  notation \"x\"\n  size 0\n}\n"))
        .expect_err("size 0 须报错");
    assert!(err.message.contains("[P1021]"), "消息 {}", err.message);

    // 空名。
    let err: prism_dsl::DslError =
        compile_v2(&doc("formula \"  \" {\n  notation \"x\"\n}\n")).expect_err("空名须报错");
    assert!(err.message.contains("[P1022]"), "消息 {}", err.message);

    // 重名。
    let err: prism_dsl::DslError = compile_v2(&doc(
        "formula \"dup\" {\n  notation \"x\"\n}\nformula \"dup\" {\n  notation \"y\"\n}\n",
    ))
    .expect_err("重名须报错");
    assert!(err.message.contains("重名"), "消息 {}", err.message);
}

// ---------------------------------------------------------------------------
// 2. 端到端：parse → codegen → 布局 → 渲染出字（≥5 类结构）
// ---------------------------------------------------------------------------

/// 五类结构各一条公式（对应 `symbols.rs` 的等价重写）。
const FIVE_STRUCTURES: &[(&str, &str)] = &[
    // 结构 1：上下标（x² / x₀ 的等价重写）。
    ("sup_sub", r"x^2 + x_1"),
    // 结构 2：分式（dy/dx 竖式分数）。
    ("frac", r"\frac{dy}{dx}"),
    // 结构 3：根号（含根指数）。
    ("sqrt", r"\sqrt[3]{x^2 + 1}"),
    // 结构 4：大型算子 + 限位（∫₀¹ / ∑ 的等价重写）。
    ("bigop", r"\int_{0}^{1} x dx"),
    // 结构 5：矩阵（2×2）。
    ("matrix", r"\matrix{a & b \\ c & d}"),
];

/// DSL 五类结构 → 全链路 → 字形与布局输出。
#[test]
fn five_structures_end_to_end_render() {
    let blocks: String = FIVE_STRUCTURES
        .iter()
        .map(|(name, notation)| format!("formula \"{name}\" {{ notation \"{notation}\" size 30 }}\n"))
        .collect();
    let formulas: Vec<FormulaNode> = compile_formulas(&blocks);
    assert_eq!(formulas.len(), 5, "五类结构各有节点");
    let font = prism_text::font_regular();
    for (i, (name, _)) in FIVE_STRUCTURES.iter().enumerate() {
        let node: &FormulaNode = &formulas[i];
        assert_eq!(node.name, *name);
        // 记法 → AST。
        let expr: MathExpr = MathExpr::parse(&node.notation)
            .unwrap_or_else(|e| panic!("formula '{name}' 记法解析失败：{e}"));
        // AST → 字形布局。
        let layout: MathLayout = prism_text::math_layout::layout_math_sized(&font, &expr, node.size);
        assert!(!layout.glyphs.is_empty(), "公式 '{name}' 必须渲染出字形");
        assert!(layout.width > 0.0, "公式 '{name}' 宽度须为正");
        assert!(layout.ascent > 0.0, "公式 '{name}' 上伸须为正");
        // 字形序列可作为「出字」证据（非空字符序列）。
        let text: String = layout.text();
        assert!(!text.trim().is_empty(), "公式 '{name}' 字形序列为空");
        // 每个字形都有 gid 与有限坐标。
        for g in &layout.glyphs {
            assert!(g.x.is_finite() && g.y.is_finite() && g.advance.is_finite(), "坐标须有限");
            // 大型算子按 big_op_scale（1.45）放大，故上限放宽到 2× 基座字号。
            assert!(g.size > 0.0 && g.size <= node.size * 2.0, "字号须在 (0, 2×size] 内");
        }
    }
}

/// 结构专属几何证据（分式线 / 根号横线）经 DSL 链路同样成立。
#[test]
fn structure_specific_geometry_via_dsl() {
    let blocks: &str = r#"
formula "frac" { notation "\frac{y}{x}" size 32 }
formula "sqrt" { notation "\sqrt{x+1}" size 32 }
"#;
    let f: Vec<FormulaNode> = compile_formulas(blocks);
    let font = prism_text::font_regular();
    let frac: MathLayout = prism_text::math_layout::layout_math_sized(
        &font,
        &MathExpr::parse(&f[0].notation).expect("分式记法"),
        f[0].size,
    );
    let bar = frac
        .rules
        .iter()
        .find(|r| r.kind == RuleKind::FracBar)
        .expect("分式必须有分数线");
    let num = frac.glyph('y').expect("分子 y");
    let den = frac.glyph('x').expect("分母 x");
    assert!(num.y > bar.top(), "分子在分数线上方");
    assert!(den.y < bar.bottom(), "分母在分数线下方");

    let sqrt: MathLayout = prism_text::math_layout::layout_math_sized(
        &font,
        &MathExpr::parse(&f[1].notation).expect("根号记法"),
        f[1].size,
    );
    let rule = sqrt
        .rules
        .iter()
        .find(|r| r.kind == RuleKind::SqrtOverline)
        .expect("根号必须有横线");
    let radicand = sqrt.glyph('+').expect("被开方体字形");
    assert!(radicand.x >= rule.x0 && radicand.x <= rule.x1, "横线覆盖被开方体");
    assert!(rule.top() > radicand.y, "横线在被开方体之上");
}

/// 无 formula 块的文档：`formulas` 为空（零回归：既有文档不受影响）。
#[test]
fn no_formula_block_keeps_empty_list() {
    let out: CompileV2Output =
        compile_v2("// prism-dsl 1\nscene \"s\" { }\n").expect("无 formula 文档仍可编译");
    assert!(out.formulas.is_empty(), "未声明 formula 时列表为空");
}
