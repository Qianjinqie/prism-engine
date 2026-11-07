//! 词法器：`.prism` 源码 → 带行号的 token 流。
//!
//! v1（M0/M1）词法规则逐字保留；v2 新增：
//! - 单位后缀字面量：`150ms` / `36f` / `15deg` / `0.5rad` / `50%` / `120px`；
//! - 引用分隔符 `:`（compose 引用形如 `layer:text`）。
//!
//! # 与 v1 的兼容约定
//! - `3s` 仍整体词法化为 [`Tok::Duration`]；
//! - 单位后缀之后不得再跟标识符字符（因此 `3stage` 仍解析为 `Num(3)` +
//!   `Ident("stage")`，`3degrees` 不算角度，`teal-orange` 仍是一个完整 ident）；
//! - `-0.5` / `-ball` / `480x270` / `#RRGGBB` 的判定规则不变。

use crate::{DslError, err_at};

// ---------------------------------------------------------------------------
// 词法单元
// ---------------------------------------------------------------------------

/// 词法单元类别。
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tok {
    /// 标识符：字母 / 数字 / 下划线 / 连字符，如 `sphere`、`teal-orange`、`-ball`。
    Ident(String),
    /// 双引号字符串，如 `"spin_red_ball"`（不支持转义、不允许跨行）。
    Str(String),
    /// 普通数字（含负数与小数），如 `1.0`、`-0.5`、`.5`。
    Num(f32),
    /// 分辨率 `480x270`（整体词法化，中间为小写字母 x）。
    Resolution(u32, u32),
    /// 时长 `3s`（数字 + 's' 后缀，整体词法化）。
    Duration(f32),
    // ---- v2 新增：单位后缀字面量 ----
    /// 毫秒时长 `150ms`（→ 秒 = 值 / 1000）。
    Millis(f32),
    /// 帧数 `36f`（→ 秒 = 值 / fps，归一化在 codegen 完成）。
    Frames(f32),
    /// 角度 `15deg`（→ 弧度 = 值.to_radians()）。
    Deg(f32),
    /// 弧度 `0.5rad`（原值透传）。
    Rad(f32),
    /// 百分比 `50%`（→ 归一化比例 = 值 / 100）。
    Percent(f32),
    /// 像素长度 `120px`（原值透传）。
    Px(f32),
    /// 颜色 `#RRGGBB`，保留 0..255 原值，编译时再归一化到 0..1。
    Color(u8, u8, u8),
    // ---- v2 新增：符号 ----
    /// 冒号：块引用分隔符，如 `layer:text`。
    Colon,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Comma,
}

/// 带行号的词法单元（供报错定位）。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Token {
    pub(crate) tok: Tok,
    pub(crate) line: usize,
}

/// 标识符字符：字母 / 数字 / 下划线 / 连字符 `'-'`。
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// 单个十六进制字符 → 数值（调用前已校验为 hex digit，非法输入兜底返回 0）。
fn hex_digit_value(c: u8) -> u8 {
    (c as char).to_digit(16).map_or(0, |v| v as u8)
}

/// 把词法单元转成人类可读的描述（用于错误消息，如「却遇到 'sphere'」）。
pub(crate) fn describe_tok(tok: &Tok) -> String {
    match tok {
        Tok::Ident(s) => format!("'{}'", s),
        Tok::Str(s) => format!("字符串 \"{}\"", s),
        Tok::Num(v) => format!("数字 {}", v),
        Tok::Resolution(w, h) => format!("分辨率 {}x{}", w, h),
        Tok::Duration(v) => format!("时长 {}s", v),
        Tok::Millis(v) => format!("时长 {}ms", v),
        Tok::Frames(v) => format!("帧数 {}f", v),
        Tok::Deg(v) => format!("角度 {}deg", v),
        Tok::Rad(v) => format!("弧度 {}rad", v),
        Tok::Percent(v) => format!("百分比 {}%", v),
        Tok::Px(v) => format!("长度 {}px", v),
        Tok::Color(r, g, b) => format!("颜色 #{:02x}{:02x}{:02x}", r, g, b),
        Tok::Colon => "':'".to_string(),
        Tok::LBrace => "'{'".to_string(),
        Tok::RBrace => "'}'".to_string(),
        Tok::LParen => "'('".to_string(),
        Tok::RParen => "')'".to_string(),
        Tok::Comma => "','".to_string(),
    }
}

// ---------------------------------------------------------------------------
// 手写词法器
// ---------------------------------------------------------------------------

/// 手写词法器：跳过空白与 `//` 行注释，产出带行号的 token 流。
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
}

impl Lexer {
    fn new(source: &str) -> Self {
        Self { chars: source.chars().collect(), pos: 0, line: 1 }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    /// 前进一个字符；遇到换行时维护行号。
    fn bump(&mut self) -> Option<char> {
        let c: char = self.peek()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
        }
        Some(c)
    }

    /// 跳过空白与 `//` 行注释（注释止于行尾；换行符留给下一轮处理以维护行号）。
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek2() == Some('/') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, DslError> {
        let mut out: Vec<Token> = Vec::new();
        loop {
            self.skip_trivia();
            let line: usize = self.line;
            let c: char = match self.peek() {
                Some(c) => c,
                None => break,
            };
            let tok: Tok = match c {
                '{' => { self.bump(); Tok::LBrace }
                '}' => { self.bump(); Tok::RBrace }
                '(' => { self.bump(); Tok::LParen }
                ')' => { self.bump(); Tok::RParen }
                ',' => { self.bump(); Tok::Comma }
                // v2 新增：块引用分隔符 `layer:text`。
                ':' => { self.bump(); Tok::Colon }
                '#' => self.lex_color(line)?,
                '"' => self.lex_string(line)?,
                // 负号歧义：'-' 后紧跟数字或 '.' → 负数的一部分；否则属于标识符。
                '-' if matches!(self.peek2(), Some(d) if d.is_ascii_digit() || d == '.') => {
                    self.lex_number(true, line)?
                }
                ch if ch.is_ascii_digit()
                    || (ch == '.' && matches!(self.peek2(), Some(d) if d.is_ascii_digit())) =>
                {
                    self.lex_number(false, line)?
                }
                ch if is_ident_char(ch) => self.lex_ident(),
                other => return Err(err_at(line, format!("无法识别的字符 '{}'", other))),
            };
            out.push(Token { tok, line });
        }
        Ok(out)
    }

    /// 词法化数字（可选负号、小数），并处理 `480x270` / `3s` 两种 v1 复合后缀
    /// 与 v2 的单位后缀（`150ms` / `36f` / `15deg` / `0.5rad` / `50%` / `120px`）。
    fn lex_number(&mut self, negative: bool, line: usize) -> Result<Tok, DslError> {
        let mut text = String::new();
        if negative {
            text.push('-');
            self.pos += 1; // 跳过 '-'
        }
        // 整数部分。
        let mut int_digits: usize = 0;
        loop {
            match self.peek() {
                Some(d) if d.is_ascii_digit() => {
                    text.push(d);
                    self.pos += 1;
                    int_digits += 1;
                }
                _ => break,
            }
        }
        // 小数部分（'.' 后必须跟数字，避免 `1.}` 之类被误读）。
        let mut has_dot = false;
        if self.peek() == Some('.') && matches!(self.peek2(), Some(d) if d.is_ascii_digit()) {
            has_dot = true;
            text.push('.');
            self.pos += 1;
            loop {
                match self.peek() {
                    Some(d) if d.is_ascii_digit() => {
                        text.push(d);
                        self.pos += 1;
                    }
                    _ => break,
                }
            }
        }
        if int_digits == 0 && !has_dot {
            return Err(err_at(line, "无效的数字：'-' 或 '.' 后必须跟数字"));
        }
        // 复合后缀一：分辨率 `480x270`（仅限非负整数，避免与负数 / 小数混淆）。
        if !negative
            && !has_dot
            && self.peek() == Some('x')
            && matches!(self.peek2(), Some(d) if d.is_ascii_digit())
        {
            self.pos += 1; // 跳过 'x'
            let mut height_text = String::new();
            loop {
                match self.peek() {
                    Some(d) if d.is_ascii_digit() => {
                        height_text.push(d);
                        self.pos += 1;
                    }
                    _ => break,
                }
            }
            let width: u32 = text
                .parse()
                .map_err(|_| err_at(line, format!("分辨率宽度 '{}' 超出 u32 范围", text)))?;
            let height: u32 = height_text
                .parse()
                .map_err(|_| err_at(line, format!("分辨率高度 '{}' 超出 u32 范围", height_text)))?;
            return Ok(Tok::Resolution(width, height));
        }
        // 复合后缀二：时长 `3s`（'s' 后不能再跟标识符字符，如 `3seconds` 不算时长）。
        if self.peek() == Some('s') && !matches!(self.peek2(), Some(d) if is_ident_char(d)) {
            self.pos += 1; // 跳过 's'
            let value: f32 = text
                .parse()
                .map_err(|_| err_at(line, format!("无法把 '{}' 解析为时长", text)))?;
            return Ok(Tok::Duration(value));
        }
        // 复合后缀三（v2 新增）：其余单位字面量。约定与 `3s` 相同——后缀后
        // 不得再跟标识符字符（`3degrees` 不算角度，`3fps` 不算帧数）。
        let matched: Option<fn(f32) -> Tok> = if self.peek() == Some('%') {
            self.pos += 1;
            Some(Tok::Percent)
        } else if self.peek() == Some('d')
            && self.peek2() == Some('e')
            && self.chars.get(self.pos + 2).copied() == Some('g')
            && !matches!(self.chars.get(self.pos + 3).copied(), Some(c) if is_ident_char(c))
        {
            self.pos += 3;
            Some(Tok::Deg)
        } else if self.peek() == Some('r')
            && self.peek2() == Some('a')
            && self.chars.get(self.pos + 2).copied() == Some('d')
            && !matches!(self.chars.get(self.pos + 3).copied(), Some(c) if is_ident_char(c))
        {
            self.pos += 3;
            Some(Tok::Rad)
        } else if self.peek() == Some('p')
            && self.peek2() == Some('x')
            && !matches!(self.chars.get(self.pos + 2).copied(), Some(c) if is_ident_char(c))
        {
            self.pos += 2;
            Some(Tok::Px)
        } else if self.peek() == Some('m')
            && self.peek2() == Some('s')
            && !matches!(self.chars.get(self.pos + 2).copied(), Some(c) if is_ident_char(c))
        {
            self.pos += 2;
            Some(Tok::Millis)
        } else if self.peek() == Some('f')
            && !matches!(self.peek2(), Some(c) if is_ident_char(c))
        {
            self.pos += 1;
            Some(Tok::Frames)
        } else {
            None
        };
        if let Some(make) = matched {
            let value: f32 = text
                .parse()
                .map_err(|_| err_at(line, format!("无法把 '{}' 解析为数字", text)))?;
            return Ok(make(value));
        }
        let value: f32 = text
            .parse()
            .map_err(|_| err_at(line, format!("无法把 '{}' 解析为数字", text)))?;
        Ok(Tok::Num(value))
    }

    /// 词法化 `#RRGGBB` 颜色为 (r, g, b) 各 0..255。
    fn lex_color(&mut self, line: usize) -> Result<Tok, DslError> {
        self.pos += 1; // 跳过 '#'
        let mut hex = [0u8; 6];
        for slot in hex.iter_mut() {
            match self.peek() {
                Some(c) if c.is_ascii_hexdigit() => {
                    *slot = c as u8;
                    self.pos += 1;
                }
                _ => return Err(err_at(line, "颜色需为 #RRGGBB 格式（6 位十六进制）")),
            }
        }
        if matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
            return Err(err_at(line, "颜色需为 #RRGGBB 格式（十六进制位数超过 6）"));
        }
        let r: u8 = hex_digit_value(hex[0]) * 16 + hex_digit_value(hex[1]);
        let g: u8 = hex_digit_value(hex[2]) * 16 + hex_digit_value(hex[3]);
        let b: u8 = hex_digit_value(hex[4]) * 16 + hex_digit_value(hex[5]);
        Ok(Tok::Color(r, g, b))
    }

    /// 词法化双引号字符串（不支持转义；不允许跨行）。
    fn lex_string(&mut self, line: usize) -> Result<Tok, DslError> {
        self.pos += 1; // 跳过左引号
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\n') => return Err(err_at(line, "字符串不能跨行（缺少右引号 \"）")),
                None => return Err(err_at(line, "字符串缺少右引号 \"（到文件末尾仍未闭合）")),
                Some(c) => s.push(c),
            }
        }
        Ok(Tok::Str(s))
    }

    /// 词法化标识符：连续的 [字母 / 数字 / 下划线 / '-']。
    fn lex_ident(&mut self) -> Tok {
        let mut s = String::new();
        loop {
            match self.peek() {
                Some(c) if is_ident_char(c) => {
                    s.push(c);
                    self.pos += 1;
                }
                _ => break,
            }
        }
        Tok::Ident(s)
    }
}

/// 源码 → token 流（带行号）。
pub(crate) fn tokenize(source: &str) -> Result<Vec<Token>, DslError> {
    Lexer::new(source).tokenize()
}

// ---------------------------------------------------------------------------
// V3.0 W1：v3 前端词法面（D-22 自研 LL(1) + Pratt 的词法层）
//
// 与 v1/v2 词法面**物理隔离**：上面的 `tokenize` 一行未改，
// `// prism-dsl 1` 与无头文档永远走旧路径（D-24 零损失契约）。
// 冻结依据：`docs/v3.0-语言规格.md` §2.1 / §4 / §7。
// ---------------------------------------------------------------------------

/// V3.0 语言前端词法单元、词法器与行内词法器。
pub mod v3 {
    use crate::v3_err;
    use crate::DslError;

    /// v3 词法单元类别。
    #[derive(Debug, Clone, PartialEq)]
    pub enum Tok3 {
        /// 标识符或关键字：`[A-Za-z_][A-Za-z0-9_]*`。
        Ident(String),
        /// 十进制数字面量原文（精确转定点在 `crate::eval` 完成，词法层不引入浮点舍入）。
        Num(String),
        /// 普通双引号字符串（不插值、不跨行）。
        Str(String),
        /// 插值字符串 `f"..."` / `f"""..."""`；`start_line` 用于降级报错行号回折。
        Interp { text: String, start_line: usize },
        /// 缩进增加（4 空格一级）。
        Indent,
        /// 缩进减少。
        Dedent,
        /// 逻辑行结束。
        Newline,
        Assign,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        Plus,
        Minus,
        Star,
        Slash,
        Percent,
        LParen,
        RParen,
        LBracket,
        RBracket,
        LBrace,
        RBrace,
        Comma,
        Colon,
        Dot,
        /// `..` 区间运算符（`for i in 0..4`）。
        Range,
    }

    /// 带行号 / 列号的 v3 词法单元。
    #[derive(Debug, Clone, PartialEq)]
    pub struct Token3 {
        pub tok: Tok3,
        pub line: usize,
        pub col: usize,
    }

    /// 词法单元的人类可读描述（错误消息用）。
    pub fn describe(tok: &Tok3) -> String {
        match tok {
            Tok3::Ident(s) => format!("'{}'", s),
            Tok3::Num(s) => format!("数字 {}", s),
            Tok3::Str(s) => format!("字符串 \"{}\"", s),
            Tok3::Interp { .. } => "插值字符串".to_string(),
            Tok3::Indent => "缩进增加".to_string(),
            Tok3::Dedent => "缩进减少".to_string(),
            Tok3::Newline => "行尾".to_string(),
            Tok3::Assign => "'='".to_string(),
            Tok3::Eq => "'=='".to_string(),
            Tok3::Ne => "'!='".to_string(),
            Tok3::Lt => "'<'".to_string(),
            Tok3::Le => "'<='".to_string(),
            Tok3::Gt => "'>'".to_string(),
            Tok3::Ge => "'>='".to_string(),
            Tok3::Plus => "'+'".to_string(),
            Tok3::Minus => "'-'".to_string(),
            Tok3::Star => "'*'".to_string(),
            Tok3::Slash => "'/'".to_string(),
            Tok3::Percent => "'%'".to_string(),
            Tok3::LParen => "'('".to_string(),
            Tok3::RParen => "')'".to_string(),
            Tok3::LBracket => "'['".to_string(),
            Tok3::RBracket => "']'".to_string(),
            Tok3::LBrace => "'{'".to_string(),
            Tok3::RBrace => "'}'".to_string(),
            Tok3::Comma => "','".to_string(),
            Tok3::Colon => "':'".to_string(),
            Tok3::Dot => "'.'".to_string(),
            Tok3::Range => "'..'".to_string(),
        }
    }

    /// 手写 v3 词法器：按行处理缩进栈，产出 INDENT / DEDENT / NEWLINE。
    struct Lexer3 {
        chars: Vec<char>,
        pos: usize,
        line: usize,
        indents: Vec<usize>,
        out: Vec<Token3>,
    }

    impl Lexer3 {
        fn new(source: &str) -> Self {
            Self {
                chars: source.chars().collect(),
                pos: 0,
                line: 1,
                indents: vec![0],
                out: Vec::new(),
            }
        }

        fn peekn(&self, n: usize) -> Option<char> {
            self.chars.get(self.pos + n).copied()
        }

        fn peek(&self) -> Option<char> {
            self.peekn(0)
        }

        fn push(&mut self, tok: Tok3, line: usize, col: usize) {
            self.out.push(Token3 { tok, line, col });
        }

        /// 消费到行尾（含换行符），维护行号；用于空行 / 注释行 / 代码行收尾。
        fn consume_eol(&mut self) {
            while let Some(c) = self.peek() {
                self.pos += 1;
                if c == '\n' {
                    self.line += 1;
                    break;
                }
            }
        }

        fn run(mut self) -> Result<Vec<Token3>, DslError> {
            while self.pos < self.chars.len() {
                self.lex_line()?;
            }
            // 文件尾：闭合所有未收的缩进层级。
            while self.indents.len() > 1 {
                self.indents.pop();
                let line: usize = self.line;
                self.push(Tok3::Dedent, line, 0);
            }
            Ok(self.out)
        }

        fn lex_line(&mut self) -> Result<(), DslError> {
            let start_line: usize = self.line;
            // ① 行首缩进：只认空格，tab 直接 P6002。
            let mut indent: usize = 0;
            loop {
                match self.peek() {
                    Some(' ') => {
                        indent += 1;
                        self.pos += 1;
                    }
                    Some('\t') => {
                        return Err(v3_err(
                            start_line,
                            "P6002",
                            &format!("第 {} 列出现制表符缩进", indent + 1),
                            "缩进一律用 4 个空格；把 tab 换成 4 个空格",
                        ));
                    }
                    _ => break,
                }
            }
            // ② 空行 / 纯注释行不产生任何 token（也不参与缩进判定）。
            match self.peek() {
                None | Some('\n') | Some('\r') => {
                    self.consume_eol();
                    return Ok(());
                }
                Some('/') if self.peekn(1) == Some('/') => {
                    self.consume_eol();
                    return Ok(());
                }
                Some('#') => {
                    self.consume_eol();
                    return Ok(());
                }
                _ => {}
            }
            // ③ 缩进层级变化（P6001：非 4 的倍数 / 与既有层级不对齐）。
            if indent % 4 != 0 {
                return Err(v3_err(
                    start_line,
                    "P6001",
                    &format!("缩进 {} 空格不是 4 的倍数", indent),
                    "缩进按 4 空格一级递增或递减，例如块内语句缩进 4 空格",
                ));
            }
            let top: usize = *self.indents.last().unwrap_or(&0);
            if indent > top {
                self.indents.push(indent);
                self.push(Tok3::Indent, start_line, 0);
            } else if indent < top {
                while self.indents.len() > 1 && *self.indents.last().unwrap_or(&0) > indent {
                    self.indents.pop();
                    self.push(Tok3::Dedent, start_line, 0);
                }
                if *self.indents.last().unwrap_or(&0) != indent {
                    return Err(v3_err(
                        start_line,
                        "P6001",
                        &format!("缩进 {} 空格与任何外层层级都不对齐", indent),
                        "回到某个外层块的确切缩进列（4 的倍数）",
                    ));
                }
            }
            // ④ 行内词法化。
            loop {
                match self.peek() {
                    None | Some('\n') | Some('\r') => break,
                    Some('/') if self.peekn(1) == Some('/') => break,
                    Some('#') => break,
                    Some(' ') => {
                        self.pos += 1;
                    }
                    _ => self.lex_token(start_line)?,
                }
            }
            let line: usize = self.line;
            self.push(Tok3::Newline, line, 0);
            self.consume_eol();
            Ok(())
        }

        fn lex_token(&mut self, line: usize) -> Result<(), DslError> {
            let col: usize = self.pos;
            let c: char = match self.peek() {
                Some(c) => c,
                None => return Ok(()),
            };
            // 插值字符串（f" 或 f"""）。
            if c == 'f' && self.peekn(1) == Some('"') {
                let multi: bool = self.peekn(2) == Some('"') && self.peekn(3) == Some('"');
                return self.lex_interp(line, col, multi);
            }
            if c == '"' {
                return self.lex_plain_string(line, col);
            }
            let single: Option<Tok3> = match c {
                '(' => Some(Tok3::LParen),
                ')' => Some(Tok3::RParen),
                '[' => Some(Tok3::LBracket),
                ']' => Some(Tok3::RBracket),
                '{' => Some(Tok3::LBrace),
                '}' => Some(Tok3::RBrace),
                ',' => Some(Tok3::Comma),
                ':' => Some(Tok3::Colon),
                '+' => Some(Tok3::Plus),
                '-' => Some(Tok3::Minus),
                '*' => Some(Tok3::Star),
                '/' => Some(Tok3::Slash),
                '%' => Some(Tok3::Percent),
                _ => None,
            };
            if let Some(tok) = single {
                self.pos += 1;
                self.push(tok, line, col);
                return Ok(());
            }
            match c {
                '=' => {
                    if self.peekn(1) == Some('=') {
                        self.pos += 2;
                        self.push(Tok3::Eq, line, col);
                    } else {
                        self.pos += 1;
                        self.push(Tok3::Assign, line, col);
                    }
                    Ok(())
                }
                '!' => {
                    if self.peekn(1) == Some('=') {
                        self.pos += 2;
                        self.push(Tok3::Ne, line, col);
                        Ok(())
                    } else {
                        Err(v3_err(
                            line,
                            "P6003",
                            "单独的 '!' 不是合法运算符",
                            "逻辑取反写 'not x'，不等写 '!='",
                        ))
                    }
                }
                '<' => {
                    if self.peekn(1) == Some('=') {
                        self.pos += 2;
                        self.push(Tok3::Le, line, col);
                    } else {
                        self.pos += 1;
                        self.push(Tok3::Lt, line, col);
                    }
                    Ok(())
                }
                '>' => {
                    if self.peekn(1) == Some('=') {
                        self.pos += 2;
                        self.push(Tok3::Ge, line, col);
                    } else {
                        self.pos += 1;
                        self.push(Tok3::Gt, line, col);
                    }
                    Ok(())
                }
                '.' => {
                    if self.peekn(1) == Some('.') {
                        self.pos += 2;
                        self.push(Tok3::Range, line, col);
                    } else {
                        self.pos += 1;
                        self.push(Tok3::Dot, line, col);
                    }
                    Ok(())
                }
                ch if ch.is_ascii_digit() => self.lex_number(line, col),
                ch if ch.is_ascii_alphabetic() || ch == '_' => self.lex_ident(line, col),
                other => Err(v3_err(
                    line,
                    "P6003",
                    &format!("无法识别的字符 '{}'", other),
                    "检查拼写；v3 只接受标识符、数字、字符串与 §2.1 列出的运算符",
                )),
            }
        }

        fn lex_number(&mut self, line: usize, col: usize) -> Result<(), DslError> {
            let mut text = String::new();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                text.push(self.peek().unwrap_or('0'));
                self.pos += 1;
            }
            // '.' 只在后跟数字时属于小数；`0..4` 的 '..' 留给 Range。
            if self.peek() == Some('.')
                && matches!(self.peekn(1), Some(c) if c.is_ascii_digit())
            {
                text.push('.');
                self.pos += 1;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    text.push(self.peek().unwrap_or('0'));
                    self.pos += 1;
                }
            }
            self.push(Tok3::Num(text), line, col);
            Ok(())
        }

        fn lex_ident(&mut self, line: usize, col: usize) -> Result<(), DslError> {
            let mut text = String::new();
            loop {
                match self.peek() {
                    Some(c) if c.is_ascii_alphanumeric() || c == '_' => {
                        text.push(c);
                        self.pos += 1;
                    }
                    _ => break,
                }
            }
            self.push(Tok3::Ident(text), line, col);
            Ok(())
        }

        fn lex_plain_string(&mut self, line: usize, col: usize) -> Result<(), DslError> {
            self.pos += 1; // 左引号
            let mut text = String::new();
            loop {
                match self.peek() {
                    Some('"') => {
                        self.pos += 1;
                        break;
                    }
                    Some('\n') | Some('\r') | None => {
                        return Err(v3_err(
                            line,
                            "P6205",
                            "普通字符串不能跨行且缺少右引号",
                            "补上右引号；需要多行文本请用 f\"\"\"...\"\"\"",
                        ));
                    }
                    Some(c) => {
                        text.push(c);
                        self.pos += 1;
                    }
                }
            }
            self.push(Tok3::Str(text), line, col);
            Ok(())
        }

        fn lex_interp(&mut self, line: usize, col: usize, multi: bool) -> Result<(), DslError> {
            self.pos += 1; // 'f'
            self.pos += if multi { 3 } else { 1 }; // 左引号
            let start_line: usize = self.line;
            let mut text = String::new();
            loop {
                let c: char = match self.peek() {
                    Some(c) => c,
                    None => {
                        return Err(v3_err(
                            line,
                            "P6205",
                            "插值字符串缺少右引号",
                            if multi {
                                "补上右引号 \"\"\""
                            } else {
                                "补上右引号 \""
                            },
                        ));
                    }
                };
                if c == '"' {
                    if multi {
                        if self.peekn(1) == Some('"') && self.peekn(2) == Some('"') {
                            self.pos += 3;
                            break;
                        }
                    } else {
                        self.pos += 1;
                        break;
                    }
                }
                if c == '\n' {
                    if !multi {
                        return Err(v3_err(
                            line,
                            "P6205",
                            "单行插值字符串不能跨行",
                            "改用 f\"\"\"...\"\"\" 多行插值字符串",
                        ));
                    }
                    text.push('\n');
                    self.pos += 1;
                    self.line += 1;
                    continue;
                }
                text.push(c);
                self.pos += 1;
            }
            self.push(Tok3::Interp { text, start_line }, line, col);
            Ok(())
        }
    }

    /// 源码 → v3 token 流（含 INDENT / DEDENT / NEWLINE）。
    pub fn tokenize3(source: &str) -> Result<Vec<Token3>, DslError> {
        Lexer3::new(source).run()
    }

    /// 行内词法化：用于 `f"...{expr}..."` 的插值表达式片段，不产生缩进 / 行尾 token。
    pub fn tokenize_inline3(text: &str, line: usize) -> Result<Vec<Token3>, DslError> {
        let mut lx = Lexer3::new(text);
        lx.line = line;
        loop {
            match lx.peek() {
                None => break,
                Some(' ') | Some('\t') => {
                    lx.pos += 1;
                }
                _ => lx.lex_token(line)?,
            }
        }
        Ok(lx.out)
    }
}

