//! 公式 AST（V2-j「公式排版引擎化」第一步）：数学结构的**纯数据**模型。
//!
//! 支撑五类结构（验收口径）：
//! 1. **上下标**：[`MathExpr::Sup`] / [`MathExpr::Sub`] / [`MathExpr::SubSup`]；
//! 2. **分式**：[`MathExpr::Frac`]（分子 / 分母）；
//! 3. **根号**：[`MathExpr::Sqrt`]（`index` = 可选根指数，`body` = 被开方体）；
//! 4. **大型算子**：[`MathExpr::BigOp`]（∑ / ∏ / ∫ + 上下限，限位堆叠语义）；
//! 5. **矩阵**：[`MathExpr::Matrix`]（rows × cols，行主序 `cells`）。
//!
//! # 两种构造入口（任务书「从符号名/字面解析到 AST 的构造器」）
//! - **符号名**：[`MathExpr::symbol`]（`"alpha"` → `α`、`"infty"` → `∞`）与
//!   [`MathExpr::from_char`]；符号字形解析复用 [`crate::math_symbols`] 的
//!   程序化符号表（[`crate::math_symbols::fallback_gid`] 可判定是否可渲染）；
//! - **字面解析**：[`MathExpr::parse`] 解析 LaTeX 风格记法
//!   （`\frac{dy}{dx}` / `x^2` / `\sqrt[3]{x}` / `\sum_{i=1}^{n}` /
//!   `\matrix{a & b \\ c & d}`），供 DSL 的 `formula` 块与测试直接复用。
//!
//! # 确定性契约
//! 解析是 (字面量) 的纯函数：同输入必得同 AST；失败返回带字符位置的
//! [`MathError`]，绝不 panic；嵌套深度上限 [`MAX_PARSE_DEPTH`] 防栈溢出。

use std::fmt;

use crate::math_symbols::fallback_gid;

/// 记号嵌套深度上限（恶意深嵌套退化为错误而非栈溢出）。
pub const MAX_PARSE_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// 大型算子
// ---------------------------------------------------------------------------

/// 大型算子种类（上下限**纵向堆叠**的算子符号）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigOpKind {
    /// 求和 ∑。
    Sum,
    /// 连乘 ∏。
    Prod,
    /// 积分 ∫。
    Integral,
}

impl BigOpKind {
    /// 命令名（与 `\sum` / `\prod` / `\int` 一致，注释与诊断复用）。
    pub fn notation(self) -> &'static str {
        match self {
            BigOpKind::Sum => "sum",
            BigOpKind::Prod => "prod",
            BigOpKind::Integral => "int",
        }
    }

    /// 字形字符（均在 [`crate::math_symbols`] 符号表内，必有合成字形兜底）。
    pub fn symbol(self) -> char {
        match self {
            BigOpKind::Sum => '∑',
            BigOpKind::Prod => '∏',
            BigOpKind::Integral => '∫',
        }
    }

    /// 命令名 → 算子（`"sum"` / `"prod"` / `"int"`；其余 → `None`）。
    pub fn from_name(name: &str) -> Option<BigOpKind> {
        match name {
            "sum" => Some(BigOpKind::Sum),
            "prod" => Some(BigOpKind::Prod),
            "int" => Some(BigOpKind::Integral),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// 符号名表
// ---------------------------------------------------------------------------

/// 符号名 → 字符表（全部落在 [`crate::math_symbols`] 合成字形表内）。
///
/// 表内字符都能经 [`fallback_gid`] 得到合成 gid（单测逐项校验），因此
/// 「符号名 → AST → 字形」链路在字体缺字时也永不出豆腐块（D-17）。
pub static SYMBOL_NAMES: &[(&str, char)] = &[
    // 希腊小写
    ("alpha", 'α'),
    ("beta", 'β'),
    ("gamma", 'γ'),
    ("delta", 'δ'),
    ("epsilon", 'ε'),
    ("theta", 'θ'),
    ("lambda", 'λ'),
    ("mu", 'μ'),
    ("pi", 'π'),
    ("sigma", 'σ'),
    ("phi", 'φ'),
    ("psi", 'ψ'),
    ("omega", 'ω'),
    // 希腊大写
    ("Gamma", 'Γ'),
    ("Delta", 'Δ'),
    ("Theta", 'Θ'),
    ("Lambda", 'Λ'),
    ("Xi", 'Ξ'),
    ("Pi", 'Π'),
    ("Sigma", 'Σ'),
    ("Phi", 'Φ'),
    ("Psi", 'Ψ'),
    ("Omega", 'Ω'),
    // 算子 / 关系 / 集合 / 逻辑
    ("partial", '∂'),
    ("sqrt", '√'),
    ("sum", '∑'),
    ("prod", '∏'),
    ("int", '∫'),
    ("pm", '±'),
    ("mp", '∓'),
    ("times", '×'),
    ("div", '÷'),
    ("cdot", '⋅'),
    ("approx", '≈'),
    ("cong", '≅'),
    ("sim", '∼'),
    ("neq", '≠'),
    ("equiv", '≡'),
    ("leq", '≤'),
    ("geq", '≥'),
    ("le", '≤'),
    ("ge", '≥'),
    ("ll", '≪'),
    ("arrow", '→'),
    ("to", '→'),
    ("rightarrow", '→'),
    ("leftarrow", '←'),
    ("uparrow", '↑'),
    ("downarrow", '↓'),
    ("leftrightarrow", '↔'),
    ("Rightarrow", '⇒'),
    ("implies", '⇒'),
    ("Leftrightarrow", '⇔'),
    ("in", '∈'),
    ("cup", '∪'),
    ("cap", '∩'),
    ("emptyset", '∅'),
    ("forall", '∀'),
    ("exists", '∃'),
    ("neg", '¬'),
    ("land", '∧'),
    ("lor", '∨'),
    ("infty", '∞'),
    ("propto", '∝'),
    ("minus", '−'),
    ("degree", '°'),
];

/// 函数名（排版为直立字母序列，不逐字符当变量处理）。
const FUNCTION_NAMES: &[&str] = &[
    "arcsin", "arccos", "arctan", "arccot", "lim", "sin", "cos", "tan", "cot", "sec", "csc", "ln",
    "log", "exp", "max", "min", "det", "sup", "inf",
];

/// 函数名判定（记法中 `\sin x` → `sin x`）。
fn is_function_name(name: &str) -> bool {
    FUNCTION_NAMES.contains(&name)
}

/// 符号名 → 字符（表外 → `None`）。
pub fn symbol_char_by_name(name: &str) -> Option<char> {
    SYMBOL_NAMES
        .iter()
        .find(|&&(n, _)| n == name)
        .map(|&(_, c)| c)
}

// ---------------------------------------------------------------------------
// 矩阵
// ---------------------------------------------------------------------------

/// 矩阵结构（行主序 `cells`：`cells[r * cols + c]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixExpr {
    /// 行数（≥ 1）。
    pub rows: usize,
    /// 列数（≥ 1）。
    pub cols: usize,
    /// 单元（长度恒为 `rows * cols`；空单元为 [`MathExpr::Row`] 空向量）。
    pub cells: Vec<MathExpr>,
}

impl MatrixExpr {
    /// 取单元（越界 → `None`）。
    pub fn cell(&self, row: usize, col: usize) -> Option<&MathExpr> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.cells.get(row * self.cols + col)
    }

    /// 每行单元迭代（排版引擎行序遍历用）。
    pub fn row_cells(&self, row: usize) -> impl Iterator<Item = &MathExpr> {
        let cols: usize = self.cols;
        let begin: usize = row * cols;
        self.cells.iter().skip(begin).take(cols)
    }
}

// ---------------------------------------------------------------------------
// 表达式
// ---------------------------------------------------------------------------

/// 公式表达式（AST 节点）。
#[derive(Debug, Clone, PartialEq)]
pub enum MathExpr {
    /// 序列（并置；空向量 = 空槽，用于矩阵空单元）。
    Row(Vec<MathExpr>),
    /// 标识符串（`x` / `dy` / `sin` 等字母序列）。
    Ident(String),
    /// 数字串（`2` / `3.14`）。
    Num(String),
    /// 数学符号（字符取自 [`crate::math_symbols`] 符号表；渲染必有字形）。
    Sym(char),
    /// 其他字面字符（`+` / `=` / `(` 等，按字体 cmap 渲染）。
    Text(char),
    /// 上标：`base^sup`。
    Sup(Box<MathExpr>, Box<MathExpr>),
    /// 下标：`base_sub`。
    Sub(Box<MathExpr>, Box<MathExpr>),
    /// 上下标同附：`base_sub^sup`。
    SubSup(Box<MathExpr>, Box<MathExpr>, Box<MathExpr>),
    /// 分式：`Frac(分子, 分母)`。
    Frac(Box<MathExpr>, Box<MathExpr>),
    /// 根号（`index = Some` 时带根指数）。
    Sqrt {
        /// 根指数（`\sqrt[3]{x}` 的 3）。
        index: Option<Box<MathExpr>>,
        /// 被开方体。
        body: Box<MathExpr>,
    },
    /// 大型算子 + 上下限（限位纵向堆叠）。
    BigOp {
        /// 算子种类。
        op: BigOpKind,
        /// 下限（`\sum_{i=1}` 的 `i=1`）。
        sub: Option<Box<MathExpr>>,
        /// 上限（`\sum^{n}` 的 `n`）。
        sup: Option<Box<MathExpr>>,
    },
    /// 矩阵。
    Matrix(MatrixExpr),
}

impl MathExpr {
    // -- 叶子构造 ---------------------------------------------------------

    /// 标识符节点。
    pub fn ident(name: impl Into<String>) -> MathExpr {
        MathExpr::Ident(name.into())
    }

    /// 数字节点。
    pub fn num(literal: impl Into<String>) -> MathExpr {
        MathExpr::Num(literal.into())
    }

    /// 符号名 → 节点（表外 → `None`）。
    pub fn symbol(name: &str) -> Option<MathExpr> {
        symbol_char_by_name(name).map(MathExpr::Sym)
    }

    /// 单字符 → 节点：符号表内字符落 [`MathExpr::Sym`]，其余落 [`MathExpr::Text`]。
    ///
    /// 判定复用 [`crate::math_symbols::fallback_gid`]（符号表是渲染兜底表，
    /// 表内字符必有合成字形）。
    pub fn from_char(ch: char) -> MathExpr {
        if fallback_gid(ch).is_some() {
            MathExpr::Sym(ch)
        } else {
            MathExpr::Text(ch)
        }
    }

    // -- 结构构造（builder） ----------------------------------------------

    /// 加上标。
    pub fn sup(self, sup: MathExpr) -> MathExpr {
        MathExpr::Sup(Box::new(self), Box::new(sup))
    }

    /// 加下标。
    pub fn sub(self, sub: MathExpr) -> MathExpr {
        MathExpr::Sub(Box::new(self), Box::new(sub))
    }

    /// 同时加上下标。
    pub fn sub_sup(self, sub: MathExpr, sup: MathExpr) -> MathExpr {
        MathExpr::SubSup(Box::new(self), Box::new(sub), Box::new(sup))
    }

    /// 分式。
    pub fn frac(num: MathExpr, den: MathExpr) -> MathExpr {
        MathExpr::Frac(Box::new(num), Box::new(den))
    }

    /// 无指数根号。
    pub fn sqrt(body: MathExpr) -> MathExpr {
        MathExpr::Sqrt { index: None, body: Box::new(body) }
    }

    /// 带指数根号（`\sqrt[index]{body}`）。
    pub fn sqrt_indexed(index: MathExpr, body: MathExpr) -> MathExpr {
        MathExpr::Sqrt { index: Some(Box::new(index)), body: Box::new(body) }
    }

    /// 大型算子（限位可缺省）。
    pub fn big_op(op: BigOpKind, sub: Option<MathExpr>, sup: Option<MathExpr>) -> MathExpr {
        MathExpr::BigOp { op, sub: sub.map(Box::new), sup: sup.map(Box::new) }
    }

    /// 逐行构造矩阵（矩形校验：各行长度必须一致且非空）。
    pub fn matrix_rows(rows: Vec<Vec<MathExpr>>) -> Result<MathExpr, MathError> {
        if rows.is_empty() {
            return Err(MathError::new(0, "矩阵至少需要一行".to_string()));
        }
        let cols: usize = rows[0].len();
        if cols == 0 {
            return Err(MathError::new(0, "矩阵每行至少需要一个单元".to_string()));
        }
        if let Some(bad) = rows.iter().position(|r| r.len() != cols) {
            return Err(MathError::new(
                0,
                format!("矩阵第 {} 行有 {} 个单元，与首行 {} 个不一致（须为矩形）", bad + 1, rows[bad].len(), cols),
            ));
        }
        let mut cells: Vec<MathExpr> = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            cells.extend(row);
        }
        Ok(MathExpr::Matrix(MatrixExpr { rows: cells.len() / cols, cols, cells }))
    }

    /// 按 rows × cols 与行主序单元构造矩阵（单元数必须等于 rows × cols）。
    pub fn matrix_cells(rows: usize, cols: usize, cells: Vec<MathExpr>) -> Result<MathExpr, MathError> {
        if rows == 0 || cols == 0 {
            return Err(MathError::new(0, "矩阵行列数必须 ≥ 1".to_string()));
        }
        if cells.len() != rows * cols {
            return Err(MathError::new(
                0,
                format!("矩阵 {}×{} 需要 {} 个单元，实得 {}", rows, cols, rows * cols, cells.len()),
            ));
        }
        Ok(MathExpr::Matrix(MatrixExpr { rows, cols, cells }))
    }

    // -- 字面解析 ---------------------------------------------------------

    /// LaTeX 风格记法 → AST。
    ///
    /// 支持：`\frac{a}{b}`、`\sqrt{x}`、`\sqrt[3]{x}`、`base_sub^sup`、
    /// `\sum` / `\prod` / `\int` + 上下限、`\matrix{a & b \\ c & d}`、
    /// 符号名命令（`\alpha` / `\infty`）、函数名命令（`\sin`）、
    /// 分组 `{...}`、字母 / 数字 / 运算符字面。空白字符被忽略。
    ///
    /// # Errors
    /// 未知命令、缺参数、非矩形矩阵、嵌套超限等 → [`MathError`]（带字符位置）。
    pub fn parse(src: &str) -> Result<MathExpr, MathError> {
        let mut p: NotationParser =
            NotationParser { chars: src.chars().collect(), pos: 0, depth: 0 };
        let expr: MathExpr = p.parse_seq()?;
        p.skip_spaces();
        if let Some(c) = p.peek() {
            return Err(MathError::new(
                p.pos,
                format!("多余的字符 '{c}'（无法归入任何公式结构）"),
            ));
        }
        Ok(expr)
    }
}

// ---------------------------------------------------------------------------
// 错误
// ---------------------------------------------------------------------------

/// 公式记法解析错误（字符级位置 + 说人话的消息）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathError {
    /// 出错字符位置（`chars()` 序；0 起）。
    pub pos: usize,
    /// 消息。
    pub message: String,
}

impl MathError {
    /// 构造错误。
    pub fn new(pos: usize, message: impl Into<String>) -> MathError {
        MathError { pos, message: message.into() }
    }
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "公式第 {} 字符：{}", self.pos, self.message)
    }
}

impl std::error::Error for MathError {}

// ---------------------------------------------------------------------------
// 手写记法解析器
// ---------------------------------------------------------------------------

/// 递归下降记法解析器（字符流 + 深度计数）。
struct NotationParser {
    chars: Vec<char>,
    pos: usize,
    depth: usize,
}

impl NotationParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.pos + k).copied()
    }

    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    /// 序列终止判定：串尾 / `}` / `]` / `&` / 行分隔 `\\`。
    fn at_seq_end(&self) -> bool {
        match self.peek() {
            None | Some('}') | Some(']') | Some('&') => true,
            Some('\\') => self.peek_at(1) == Some('\\'),
            _ => false,
        }
    }

    /// 解析序列（并置原子直到终止符；单原子不加 `Row` 包装）。
    fn parse_seq(&mut self) -> Result<MathExpr, MathError> {
        let mut items: Vec<MathExpr> = Vec::new();
        loop {
            self.skip_spaces();
            if self.at_seq_end() {
                break;
            }
            items.push(self.parse_atom()?);
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap_or(MathExpr::Row(Vec::new()))
        } else {
            MathExpr::Row(items)
        })
    }

    /// 原子 = 基座 + 可选上下标；大型算子的上下标折叠为**限位**。
    fn parse_atom(&mut self) -> Result<MathExpr, MathError> {
        if self.depth >= MAX_PARSE_DEPTH {
            return Err(MathError::new(
                self.pos,
                format!("公式嵌套超过 {MAX_PARSE_DEPTH} 层（防御性上限）"),
            ));
        }
        self.depth += 1;
        let base: MathExpr = self.parse_base()?;
        let mut sub: Option<MathExpr> = None;
        let mut sup: Option<MathExpr> = None;
        loop {
            self.skip_spaces();
            match self.peek() {
                Some('_') if sub.is_none() => {
                    self.pos += 1;
                    sub = Some(self.parse_script_arg()?);
                }
                Some('^') if sup.is_none() => {
                    self.pos += 1;
                    sup = Some(self.parse_script_arg()?);
                }
                _ => break,
            }
        }
        self.depth -= 1;
        Ok(match (base, sub, sup) {
            // 大型算子的上下标 = 限位（纵向堆叠语义，与普通上下标区分）。
            (MathExpr::BigOp { op, .. }, s, p) => MathExpr::BigOp {
                op,
                sub: s.map(Box::new),
                sup: p.map(Box::new),
            },
            (b, None, None) => b,
            (b, Some(s), None) => MathExpr::Sub(Box::new(b), Box::new(s)),
            (b, None, Some(p)) => MathExpr::Sup(Box::new(b), Box::new(p)),
            (b, Some(s), Some(p)) => MathExpr::SubSup(Box::new(b), Box::new(s), Box::new(p)),
        })
    }

    /// 基座原子（不含上下标）。
    fn parse_base(&mut self) -> Result<MathExpr, MathError> {
        self.skip_spaces();
        match self.peek() {
            None => Err(MathError::new(self.pos, "缺少表达式".to_string())),
            Some('{') => {
                self.pos += 1;
                let e: MathExpr = self.parse_seq()?;
                self.expect_char('}')?;
                Ok(e)
            }
            Some('\\') => self.parse_command(),
            Some(c) if c.is_ascii_digit() || (c == '.' && matches!(self.peek_at(1), Some(d) if d.is_ascii_digit())) => {
                let mut s: String = String::new();
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit() || (d == '.' && !s.contains('.')) {
                        s.push(d);
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                Ok(MathExpr::Num(s))
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let mut s: String = String::new();
                while let Some(d) = self.peek() {
                    if d.is_ascii_alphabetic() {
                        s.push(d);
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                Ok(MathExpr::Ident(s))
            }
            Some(c) => {
                self.pos += 1;
                Ok(MathExpr::from_char(c))
            }
        }
    }

    /// 上下标参数：`{...}` 分组或单个基座原子。
    fn parse_script_arg(&mut self) -> Result<MathExpr, MathError> {
        self.skip_spaces();
        match self.peek() {
            None => Err(MathError::new(self.pos, "上下标缺少参数".to_string())),
            Some('{') => {
                self.pos += 1;
                let e: MathExpr = self.parse_seq()?;
                self.expect_char('}')?;
                Ok(e)
            }
            _ => self.parse_base(),
        }
    }

    /// 必选参数（`\frac` / `\sqrt` 的被开方体）：`{...}` 分组或单原子。
    fn parse_arg(&mut self, what: &str) -> Result<MathExpr, MathError> {
        self.skip_spaces();
        if self.peek().is_none() {
            return Err(MathError::new(self.pos, format!("{what} 缺失")));
        }
        self.parse_script_arg()
    }

    fn expect_char(&mut self, want: char) -> Result<(), MathError> {
        self.skip_spaces();
        match self.peek() {
            Some(c) if c == want => {
                self.pos += 1;
                Ok(())
            }
            Some(c) => Err(MathError::new(self.pos, format!("期望 '{want}'，却遇到 '{c}'"))),
            None => Err(MathError::new(self.pos, format!("缺少 '{want}'（到记法末尾仍未闭合）"))),
        }
    }

    /// 反斜杠命令：结构命令（frac / sqrt / matrix / 大型算子）或符号 / 函数名。
    fn parse_command(&mut self) -> Result<MathExpr, MathError> {
        let start: usize = self.pos;
        self.pos += 1; // 跳过 '\'
        if self.peek() == Some('\\') {
            return Err(MathError::new(
                start,
                "行分隔符 '\\\\' 只能出现在 \\matrix 块内".to_string(),
            ));
        }
        let mut name: String = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphabetic() {
                name.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        if name.is_empty() {
            return Err(MathError::new(start, "孤立的反斜杠（'\\' 后缺命令名）".to_string()));
        }
        match name.as_str() {
            "frac" => {
                let num: MathExpr = self.parse_arg("\\frac 的分子")?;
                let den: MathExpr = self.parse_arg("\\frac 的分母")?;
                Ok(MathExpr::frac(num, den))
            }
            "sqrt" => {
                self.skip_spaces();
                let index: Option<Box<MathExpr>> = if self.peek() == Some('[') {
                    self.pos += 1;
                    let e: MathExpr = self.parse_seq()?;
                    self.expect_char(']')?;
                    Some(Box::new(e))
                } else {
                    None
                };
                let body: MathExpr = self.parse_arg("\\sqrt 的被开方体")?;
                Ok(MathExpr::Sqrt { index, body: Box::new(body) })
            }
            "matrix" => self.parse_matrix(start),
            _ => {
                if let Some(op) = BigOpKind::from_name(&name) {
                    Ok(MathExpr::BigOp { op, sub: None, sup: None })
                } else if let Some(e) = MathExpr::symbol(&name) {
                    Ok(e)
                } else if is_function_name(&name) {
                    Ok(MathExpr::Ident(name))
                } else if let Some(c) = symbol_char_strict(&name) {
                    // 兜底：命令名恰好是单字符符号（见表外扩展）。
                    Ok(MathExpr::Sym(c))
                } else {
                    Err(MathError::new(
                        start,
                        format!("未知命令 '\\{name}'（支持结构命令 frac / sqrt / matrix / sum / prod / int 与符号名表）"),
                    ))
                }
            }
        }
    }

    /// `\matrix{ a & b \\ c & d }`：单元 `&` 分隔、行 `\\` 分隔。
    fn parse_matrix(&mut self, start: usize) -> Result<MathExpr, MathError> {
        self.skip_spaces();
        if self.peek() != Some('{') {
            return Err(MathError::new(
                self.pos,
                "\\matrix 后须跟 '{'（如 \\matrix{a & b \\\\ c & d}）".to_string(),
            ));
        }
        self.pos += 1;
        let mut rows: Vec<Vec<MathExpr>> = vec![Vec::new()];
        loop {
            let cell: MathExpr = self.parse_seq()?;
            if let Some(last) = rows.last_mut() {
                last.push(cell);
            }
            self.skip_spaces();
            match self.peek() {
                Some('&') => {
                    self.pos += 1;
                }
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                Some('\\') if self.peek_at(1) == Some('\\') => {
                    self.pos += 2;
                    rows.push(Vec::new());
                }
                None => {
                    return Err(MathError::new(
                        start,
                        "\\matrix 缺少右花括号 '}'（到记法末尾仍未闭合）".to_string(),
                    ));
                }
                Some(c) => {
                    return Err(MathError::new(
                        self.pos,
                        format!("\\matrix 内意外的字符 '{c}'（单元用 & 分隔，行用 \\\\ 分隔）"),
                    ));
                }
            }
        }
        // 矩形校验（位置用 matrix 命令起点，指向整块）。
        let cols: usize = rows[0].len();
        if cols == 0 {
            return Err(MathError::new(start, "\\matrix 每行至少需要一个单元".to_string()));
        }
        if let Some(bad) = rows.iter().position(|r| r.len() != cols) {
            return Err(MathError::new(
                start,
                format!(
                    "\\matrix 不是矩形：第 {} 行有 {} 个单元，首行有 {} 个",
                    bad + 1,
                    rows[bad].len(),
                    cols
                ),
            ));
        }
        MathExpr::matrix_rows(rows)
    }
}

/// 表外单字符命令兜底（当前无表外项，恒 `None`——保留扩展位）。
fn symbol_char_strict(_name: &str) -> Option<char> {
    None
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// 结构 1：上标。
    #[test]
    fn parse_superscript() {
        let e: MathExpr = MathExpr::parse("x^2").expect("x^2 可解析");
        assert_eq!(
            e,
            MathExpr::Sup(Box::new(MathExpr::ident("x")), Box::new(MathExpr::num("2")))
        );
    }

    /// 结构 1：下标。
    #[test]
    fn parse_subscript() {
        let e: MathExpr = MathExpr::parse("x_1").expect("x_1 可解析");
        assert_eq!(
            e,
            MathExpr::Sub(Box::new(MathExpr::ident("x")), Box::new(MathExpr::num("1")))
        );
    }

    /// 结构 1：上下标同附 + 分组下标（`x_{n+1}`）。
    #[test]
    fn parse_subsup_with_group() {
        let e: MathExpr = MathExpr::parse("a_{n+1}^2").expect("a_{n+1}^2 可解析");
        match e {
            MathExpr::SubSup(base, sub, sup) => {
                assert_eq!(*base, MathExpr::ident("a"));
                assert_eq!(
                    *sub,
                    MathExpr::Row(vec![
                        MathExpr::ident("n"),
                        MathExpr::Text('+'),
                        MathExpr::num("1"),
                    ])
                );
                assert_eq!(*sup, MathExpr::num("2"));
            }
            other => panic!("期望 SubSup，实得 {other:?}"),
        }
    }

    /// 结构 2：分式（符号名 dy / dx 走符号表外的字母 → Ident）。
    #[test]
    fn parse_fraction() {
        let e: MathExpr = MathExpr::parse("\\frac{dy}{dx}").expect("\\frac 可解析");
        assert_eq!(
            e,
            MathExpr::frac(MathExpr::ident("dy"), MathExpr::ident("dx"))
        );
    }

    /// 结构 3：根号（无指数 / 带指数）。
    #[test]
    fn parse_sqrt_and_indexed_sqrt() {
        let plain: MathExpr = MathExpr::parse("\\sqrt{x+1}").expect("\\sqrt 可解析");
        assert_eq!(
            plain,
            MathExpr::sqrt(MathExpr::Row(vec![
                MathExpr::ident("x"),
                MathExpr::Text('+'),
                MathExpr::num("1"),
            ]))
        );
        let indexed: MathExpr = MathExpr::parse("\\sqrt[3]{x}").expect("\\sqrt[3] 可解析");
        assert_eq!(indexed, MathExpr::sqrt_indexed(MathExpr::num("3"), MathExpr::ident("x")));
    }

    /// 结构 4：大型算子 + 限位（∑ 的 `_{i=1}^{n}`）。
    #[test]
    fn parse_big_op_with_limits() {
        let e: MathExpr = MathExpr::parse("\\sum_{i=1}^{n}").expect("\\sum 可解析");
        match e {
            MathExpr::BigOp { op, sub, sup } => {
                assert_eq!(op, BigOpKind::Sum);
                assert!(sub.is_some(), "下限必须绑定到算子（限位，不是普通下标）");
                assert!(sup.is_some(), "上限必须绑定到算子");
                assert_eq!(*sup.expect("上限"), MathExpr::ident("n"));
            }
            other => panic!("期望 BigOp，实得 {other:?}"),
        }
        // ∫ 带上下限
        let i: MathExpr = MathExpr::parse("\\int_{0}^{1}").expect("\\int 可解析");
        assert!(matches!(i, MathExpr::BigOp { op: BigOpKind::Integral, .. }));
    }

    /// 结构 5：矩阵（2×2）。
    #[test]
    fn parse_matrix() {
        let e: MathExpr = MathExpr::parse("\\matrix{a & b \\\\ c & d}").expect("\\matrix 可解析");
        match e {
            MathExpr::Matrix(m) => {
                assert_eq!((m.rows, m.cols), (2, 2));
                assert_eq!(m.cell(0, 0), Some(&MathExpr::ident("a")));
                assert_eq!(m.cell(0, 1), Some(&MathExpr::ident("b")));
                assert_eq!(m.cell(1, 0), Some(&MathExpr::ident("c")));
                assert_eq!(m.cell(1, 1), Some(&MathExpr::ident("d")));
            }
            other => panic!("期望 Matrix，实得 {other:?}"),
        }
    }

    /// 复合：符号名 + 分式 + 上下标（symbols.rs 主公式的等价重写）。
    #[test]
    fn parse_symbols_equivalent_notation() {
        let e: MathExpr = MathExpr::parse("f'(x) = \\frac{dy}{dx} = 2x").expect("复合公式可解析");
        // 顶层是并置序列（5 个以上原子）。
        match e {
            MathExpr::Row(items) => assert!(items.len() >= 5, "并置原子数 {}", items.len()),
            other => panic!("期望 Row，实得 {other:?}"),
        }
        // 符号名 → 符号节点。
        let s: MathExpr = MathExpr::parse("\\infty").expect("\\infty 可解析");
        assert_eq!(s, MathExpr::Sym('∞'));
        assert_eq!(
            MathExpr::symbol("alpha").expect("alpha 在符号名表"),
            MathExpr::Sym('α')
        );
    }

    /// 符号名表的每个字符都必须有合成字形兜底（永不出豆腐块）。
    #[test]
    fn symbol_names_all_renderable() {
        for &(name, ch) in SYMBOL_NAMES {
            assert!(fallback_gid(ch).is_some(), "符号名 '{name}' 的字符 '{ch}' 不在符号表");
        }
    }

    /// 错误路径：缺参数 / 未知命令 / 非矩形矩阵 / 嵌套超限。
    #[test]
    fn parse_errors_are_diagnostic() {
        assert!(MathExpr::parse("\\frac{a}").is_err(), "缺分母须报错");
        let e: MathError = MathExpr::parse("\\foo").expect_err("未知命令须报错");
        assert!(e.message.contains("未知命令"), "消息 {}", e.message);
        let bad: MathError = MathExpr::parse("\\matrix{a & b \\\\ c}").expect_err("非矩形须报错");
        assert!(bad.message.contains("矩形"), "消息 {}", bad.message);
        assert!(MathExpr::parse("x^").is_err(), "上标缺参数须报错");
        assert!(MathExpr::parse("a \\\\ b").is_err(), "矩阵外的行分隔须报错");
        assert!(MathExpr::parse("x]").is_err(), "裸露的 ']' 须报错");
        // 深嵌套不 panic（返回错误）。
        let deep: String = format!("{}{}", "\\frac{1}{".repeat(40), "}".repeat(4));
        assert!(MathExpr::parse(&deep).is_err(), "嵌套超限须报错而非 panic");
    }

    /// 矩阵构造器校验（行主序单元数）。
    #[test]
    fn matrix_builders_validate() {
        let m: MathExpr = MathExpr::matrix_cells(
            2,
            2,
            vec![
                MathExpr::num("1"),
                MathExpr::num("2"),
                MathExpr::num("3"),
                MathExpr::num("4"),
            ],
        )
        .expect("2×2 单元数正确");
        match m {
            MathExpr::Matrix(mx) => {
                assert_eq!((mx.rows, mx.cols), (2, 2));
                assert_eq!(mx.row_cells(1).count(), 2);
                assert_eq!(mx.cell(2, 0), None, "越界 → None");
            }
            other => panic!("期望 Matrix，实得 {other:?}"),
        }
        assert!(MathExpr::matrix_cells(2, 2, vec![MathExpr::num("1")]).is_err());
        assert!(MathExpr::matrix_rows(vec![vec![], vec![]]).is_err());
    }
}
