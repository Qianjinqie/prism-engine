//! 受限表达式求值骨架（工作流 A2；D-7 自研受限求值器；架构 §3.6 类图
//! `PropertyChannel::eval(t)` 的表达式通道形态）。
//!
//! **本批范围（从简不过度设计）**：最小算术 `+ - * /`（含一元负号与括号）
//! + 变量 `t`（时间，秒）/ `p`（进度 0–1）/ `w`（宽）/ `h`（高）。完整语法
//! （函数白名单、`this.*`/节点引用、常量折叠、编译期环检测与行号）随 A4
//! DSL v2 波次落地（路线图 §3.6）。
//!
//! # 确定性与语义
//!
//! - 递归下降解析为 AST 后求值：无隐藏状态，同 env 同值；
//! - 除法沿用 IEEE 语义（除零得 ±∞ / NaN，确定性，不做静默兜底）；
//! - 数值解析走 `f32::from_str`（IEEE 正确舍入转换，跨平台逐位一致）；
//! - 解析深度守卫：括号嵌套与连续取负的递归深度上限为 [`MAX_PARSE_DEPTH`]
//!   层，超限返回 [`ExprError::DepthLimit`]（应报错不 panic，宪章 C20 精神）；
//! - 错误带字节位置并附修复建议（宪章 C19"错误可行动"精神，A4 接入
//!   三段式错误码体系）。

use std::fmt;

/// 表达式变量（本批全集：t/p/w/h）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprVar {
    /// 时间 t（秒）。
    T,
    /// 进度 p（0–1，调用方语义）。
    P,
    /// 宽 w（像素）。
    W,
    /// 高 h（像素）。
    H,
}

impl ExprVar {
    /// 变量名字面量（解析与报错共用，单一出口）。
    pub const fn name(self) -> &'static str {
        match self {
            Self::T => "t",
            Self::P => "p",
            Self::W => "w",
            Self::H => "h",
        }
    }
}

/// 表达式 AST（最小算术）。
#[derive(Debug, Clone, PartialEq)]
pub enum ExprAst {
    /// 数值字面量。
    Num(f32),
    /// 变量。
    Var(ExprVar),
    /// 一元负号。
    Neg(Box<ExprAst>),
    /// 加。
    Add(Box<ExprAst>, Box<ExprAst>),
    /// 减。
    Sub(Box<ExprAst>, Box<ExprAst>),
    /// 乘。
    Mul(Box<ExprAst>, Box<ExprAst>),
    /// 除（IEEE 语义）。
    Div(Box<ExprAst>, Box<ExprAst>),
}

/// 求值环境：变量注入（通道求值器按帧填充）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExprEnv {
    /// 时间 t（秒）。
    pub t: f32,
    /// 进度 p（0–1）。
    pub p: f32,
    /// 宽 w（像素）。
    pub w: f32,
    /// 高 h（像素）。
    pub h: f32,
}

/// 解析错误（带字节位置与修复建议）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprError {
    /// 空表达式。
    Empty,
    /// 非法字符。
    UnexpectedChar { pos: usize, ch: char },
    /// 意外到达输入末尾（缺操作数/右括号）。
    UnexpectedEnd,
    /// 未知变量。
    UnknownVar { pos: usize, name: String },
    /// 数值字面量不合法。
    BadNumber { pos: usize },
    /// 表达式结束后仍有残余字符。
    TrailingChars { pos: usize },
    /// 解析递归深度超限（括号嵌套 / 连续取负过深；防栈溢出守卫，宪章 C20）。
    DepthLimit { max: usize },
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 事件 + 位置 + 建议（宪章 C19：无建议的错误不许存在）。
        match self {
            Self::Empty => write!(
                f,
                "表达式为空。建议：写一个算术表达式，如 `t * 2` 或 `p * w`。"
            ),
            Self::UnexpectedChar { pos, ch } => write!(
                f,
                "位置 {pos}：非法字符 '{ch}'。建议：只支持数字、变量 t/p/w/h、运算符 + - * / 与括号，如 `(t + 1) * 2`。"
            ),
            Self::UnexpectedEnd => write!(
                f,
                "表达式意外结束。建议：补全操作数或右括号，如 `t * 2`、`(t + 1)`。"
            ),
            Self::UnknownVar { pos, name } => write!(
                f,
                "位置 {pos}：未知变量 '{name}'。建议：本批仅支持变量 t（时间秒）/p（进度）/w（宽）/h（高）；完整变量集随 DSL v2 波次扩展。"
            ),
            Self::BadNumber { pos } => write!(
                f,
                "位置 {pos}：数值字面量不合法。建议：使用十进制字面量，如 `1`、`0.5`。"
            ),
            Self::TrailingChars { pos } => write!(
                f,
                "位置 {pos}：表达式结束后有多余字符。建议：删除多余字符，或用括号明确优先级，如 `(t + 1) * 2`。"
            ),
            Self::DepthLimit { max } => write!(
                f,
                "表达式解析深度超过 {max} 层上限，为防解析栈溢出已中止解析。建议：减少括号嵌套或连续取负的层数至 {max} 层以内，或拆分为多个简单表达式分步计算。"
            ),
        }
    }
}

impl std::error::Error for ExprError {}

/// 编译好的表达式：源串 + AST（源串保留供通道定义哈希/调试，架构 §4.2）。
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    src: String,
    ast: ExprAst,
}

impl Expr {
    /// 解析表达式文本（递归下降；文法：`expr := term (('+'|'-') term)*`，
    /// `term := factor (('*'|'/') factor)*`，`factor := '-' factor | primary`，
    /// `primary := number | var | '(' expr ')'`）。
    ///
    /// 深度守卫（宪章 C20：应报错不 panic）：括号嵌套与连续取负的解析
    /// 深度上限为 [`MAX_PARSE_DEPTH`] 层，超限返回
    /// [`ExprError::DepthLimit`]，不会栈溢出 abort。
    pub fn parse(src: &str) -> Result<Self, ExprError> {
        let mut parser = Parser { src: src.as_bytes(), pos: 0, depth: 0 };
        parser.skip_ws();
        if parser.pos >= parser.src.len() {
            return Err(ExprError::Empty);
        }
        let ast = parser.parse_expr()?;
        parser.skip_ws();
        if parser.pos < parser.src.len() {
            let pos = parser.pos;
            let ch = parser.src[pos] as char;
            // 语义区分（宪章 C19：报错可行动）：文法外字符（如 `2 & 3` 的
            // '&'）报非法字符；文法内字符却接不上文法（如 `2 3` 的第二个
            // 数字）报多余字符。
            if !(ch.is_ascii_alphanumeric()
                || matches!(ch, '.' | '+' | '-' | '*' | '/' | '(' | ')' | '_'))
            {
                return Err(ExprError::UnexpectedChar { pos, ch });
            }
            return Err(ExprError::TrailingChars { pos });
        }
        Ok(Self { src: src.to_string(), ast })
    }

    /// 按环境求值（纯函数：同 env 同值）。
    pub fn eval(&self, env: &ExprEnv) -> f32 {
        eval_ast(&self.ast, env)
    }

    /// 纯 f(t) 求值：p/w/h 取 0（`sample(t) -> f32` 的通道接口；完整变量
    /// 注入走 [`Self::eval`]）。
    pub fn sample(&self, t: f32) -> f32 {
        self.eval(&ExprEnv { t, p: 0.0, w: 0.0, h: 0.0 })
    }

    /// 原始源串（通道定义哈希的规范化输入之一，架构 §4.2）。
    pub fn source(&self) -> &str {
        &self.src
    }
}

/// AST 求值（IEEE 语义：除零不兜底，确定性）。
fn eval_ast(ast: &ExprAst, env: &ExprEnv) -> f32 {
    match ast {
        ExprAst::Num(v) => *v,
        ExprAst::Var(v) => match v {
            ExprVar::T => env.t,
            ExprVar::P => env.p,
            ExprVar::W => env.w,
            ExprVar::H => env.h,
        },
        ExprAst::Neg(x) => -eval_ast(x, env),
        ExprAst::Add(a, b) => eval_ast(a, env) + eval_ast(b, env),
        ExprAst::Sub(a, b) => eval_ast(a, env) - eval_ast(b, env),
        ExprAst::Mul(a, b) => eval_ast(a, env) * eval_ast(b, env),
        ExprAst::Div(a, b) => eval_ast(a, env) / eval_ast(b, env),
    }
}

// ---------------------------------------------------------------------------
// 递归下降解析器（ASCII 输入；非 ASCII 一律 UnexpectedChar）
// ---------------------------------------------------------------------------

/// 解析器递归深度上限（括号嵌套与连续取负共用计数）。取 256：既容纳
/// 既有 200 层括号嵌套健壮性测试下界（QA 外部测试），又远低于栈溢出
/// 阈值——1000 层恶意嵌套必触发守卫（QA-01 修复）。
const MAX_PARSE_DEPTH: usize = 256;

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    /// 当前递归深度：每进入一层括号或一层取负 +1，离开 -1（成败皆回退；
    /// Err 向上传播时解析器整体丢弃，无需清理）。
    depth: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_char(&self) -> char {
        self.peek().unwrap_or(b'?') as char
    }

    /// 进入一层递归（深度守卫：超限即报错而非继续下钻——宪章 C20
    /// "应报错不 panic"，防栈溢出 abort）。
    fn enter_depth(&mut self) -> Result<(), ExprError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(ExprError::DepthLimit { max: MAX_PARSE_DEPTH });
        }
        Ok(())
    }

    fn parse_expr(&mut self) -> Result<ExprAst, ExprError> {
        let mut left = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let right = self.parse_term()?;
                    left = ExprAst::Add(Box::new(left), Box::new(right));
                }
                Some(b'-') => {
                    self.pos += 1;
                    let right = self.parse_term()?;
                    left = ExprAst::Sub(Box::new(left), Box::new(right));
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_term(&mut self) -> Result<ExprAst, ExprError> {
        let mut left = self.parse_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    left = ExprAst::Mul(Box::new(left), Box::new(right));
                }
                Some(b'/') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    left = ExprAst::Div(Box::new(left), Box::new(right));
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_factor(&mut self) -> Result<ExprAst, ExprError> {
        self.skip_ws();
        match self.peek() {
            Some(b'-') => {
                // 连续取负（`----1`）与括号嵌套同为无界递归，共用深度守卫。
                self.enter_depth()?;
                self.pos += 1;
                let result = self.parse_factor();
                self.depth -= 1; // 离开本层即回退（成败皆回退）
                result.map(|inner| ExprAst::Neg(Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<ExprAst, ExprError> {
        self.skip_ws();
        match self.peek() {
            None => Err(ExprError::UnexpectedEnd),
            Some(b'(') => {
                // 括号递归回溯到 parse_expr，是深嵌套栈溢出的根源（QA-01）：
                // 进入/离开各计一层，超限报 DepthLimit 而非继续下钻。
                self.enter_depth()?;
                self.pos += 1;
                let result = self.parse_expr();
                self.depth -= 1; // 离开本层即回退（成败皆回退）
                let inner = result?;
                self.skip_ws();
                match self.peek() {
                    Some(b')') => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    _ => Err(ExprError::UnexpectedEnd),
                }
            }
            Some(c) if c.is_ascii_digit() => self.parse_number(),
            Some(c) if c.is_ascii_alphabetic() => self.parse_var(),
            Some(_) => {
                let pos = self.pos;
                let ch = self.peek_char();
                Err(ExprError::UnexpectedChar { pos, ch })
            }
        }
    }

    fn parse_number(&mut self) -> Result<ExprAst, ExprError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ExprError::BadNumber { pos: start })?;
        text.parse::<f32>()
            .map(ExprAst::Num)
            .map_err(|_| ExprError::BadNumber { pos: start })
    }

    fn parse_var(&mut self) -> Result<ExprAst, ExprError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
            self.pos += 1;
        }
        let name = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ExprError::UnexpectedChar { pos: start, ch: '?' })?
            .to_string();
        match name.as_str() {
            "t" => Ok(ExprAst::Var(ExprVar::T)),
            "p" => Ok(ExprAst::Var(ExprVar::P)),
            "w" => Ok(ExprAst::Var(ExprVar::W)),
            "h" => Ok(ExprAst::Var(ExprVar::H)),
            _ => Err(ExprError::UnknownVar { pos: start, name }),
        }
    }
}

// ---------------------------------------------------------------------------
// 测试（Phase 1 验收 #3：表达式求值 vs 手写期望值 ≥10 用例）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 手写期望值用例（13 条，覆盖 Phase 1 验收线的 ≥10）。
    #[test]
    fn eval_matches_handwritten_expectations() {
        let cases: &[(&str, ExprEnv, f32)] = &[
            // (表达式, 环境, 期望)
            ("2 + 3 * 4", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 14.0),
            ("(2 + 3) * 4", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 20.0),
            ("t", ExprEnv { t: 0.5, p: 0.0, w: 0.0, h: 0.0 }, 0.5),
            ("t * 2 + 1", ExprEnv { t: 3.0, p: 0.0, w: 0.0, h: 0.0 }, 7.0),
            ("-t", ExprEnv { t: 2.0, p: 0.0, w: 0.0, h: 0.0 }, -2.0),
            ("10 / 4", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 2.5),
            ("w / h", ExprEnv { t: 0.0, p: 0.0, w: 1920.0, h: 1080.0 }, 16.0 / 9.0),
            ("p * 100", ExprEnv { t: 0.0, p: 0.25, w: 0.0, h: 0.0 }, 25.0),
            ("((1 + 2) * (3 + 4))", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 21.0),
            ("t - -1", ExprEnv { t: 1.0, p: 0.0, w: 0.0, h: 0.0 }, 2.0),
            ("0.5 * 8", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 4.0),
            ("t * w", ExprEnv { t: 2.0, p: 0.0, w: 30.0, h: 0.0 }, 60.0),
            ("1 + 2 + 3 + 4", ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 }, 10.0),
        ];
        for (src, env, want) in cases {
            let expr = Expr::parse(src).unwrap_or_else(|e| panic!("'{src}' 应可解析: {e}"));
            let got = expr.eval(env);
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "'{src}' @ {env:?} = {got}，期望 {want}"
            );
        }
    }

    /// 除零沿用 IEEE 语义（确定性，无静默兜底）。
    #[test]
    fn division_by_zero_is_ieee() {
        let expr = Expr::parse("1 / 0").unwrap();
        assert!(expr.sample(0.0).is_infinite());
        let expr = Expr::parse("0 / 0").unwrap();
        assert!(expr.sample(0.0).is_nan());
    }

    /// sample(t) 通道接口：p/w/h 取 0。
    #[test]
    fn sample_t_uses_zero_defaults() {
        let expr = Expr::parse("t + p + w + h").unwrap();
        assert!((expr.sample(2.0) - 2.0).abs() < 1e-6);
    }

    /// 错误路径：逐类断言 + 错误信息含建议段（宪章 C19 精神）。
    #[test]
    fn parse_errors_are_actionable() {
        let cases: Vec<(&str, ExprError)> = vec![
            ("", ExprError::Empty),
            ("   ", ExprError::Empty),
            ("2 +", ExprError::UnexpectedEnd),
            ("(1 + 2", ExprError::UnexpectedEnd),
            ("x + 1", ExprError::UnknownVar { pos: 0, name: "x".into() }),
            ("2 & 3", ExprError::UnexpectedChar { pos: 2, ch: '&' }),
            ("1.2.3", ExprError::BadNumber { pos: 0 }),
            ("2 3", ExprError::TrailingChars { pos: 2 }),
        ];
        for (src, want) in cases {
            let err = Expr::parse(src).expect_err(src);
            assert_eq!(err, want, "'{src}'");
            let msg = err.to_string();
            assert!(msg.contains("建议"), "'{src}' 的错误信息缺建议段: {msg}");
        }
    }

    /// 源串保留（通道定义哈希的规范化输入）。
    #[test]
    fn source_is_preserved() {
        let expr = Expr::parse("t * 10 + 1").unwrap();
        assert_eq!(expr.source(), "t * 10 + 1");
    }

    /// QA-01 深度守卫：1000 层括号嵌套返回 `DepthLimit` 而非栈溢出 abort
    /// （宪章 C20：应报错不 panic；错误信息含建议段，宪章 C19）。
    #[test]
    fn deep_nesting_1000_levels_reports_depth_limit() {
        let src = format!("{}1{}", "(".repeat(1000), ")".repeat(1000));
        let err = Expr::parse(&src).expect_err("1000 层嵌套应报 DepthLimit 而非栈溢出");
        assert_eq!(err, ExprError::DepthLimit { max: MAX_PARSE_DEPTH });
        let msg = err.to_string();
        assert!(msg.contains("建议"), "C19：深度超限错误必含建议段: {msg}");
    }

    /// QA-01 深度守卫：连续取负 `-------1` 与括号嵌套共用深度计数，
    /// 同样报错不 panic（同类无界递归一并封堵）。
    #[test]
    fn deep_negation_chain_reports_depth_limit() {
        let src = format!("-{}1", "-".repeat(1000));
        let err = Expr::parse(&src).expect_err("1000 层连续取负应报 DepthLimit 而非栈溢出");
        assert_eq!(err, ExprError::DepthLimit { max: MAX_PARSE_DEPTH });
    }

    /// QA-01 深度守卫：上限内嵌套（100 层括号）不受影响，正常解析求值。
    #[test]
    fn nesting_within_depth_limit_still_parses() {
        let depth = 100;
        let src = format!("{}t{}", "(".repeat(depth), ")".repeat(depth));
        let e = Expr::parse(&src).expect("100 层括号嵌套应正常解析");
        assert!((e.sample(2.5) - 2.5).abs() < 1e-6);
    }
}
