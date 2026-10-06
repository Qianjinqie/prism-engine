//! 词法器：`.prism` 源码 → 带行号的 token 流。
//!
//! v1（M0/M1）词法规则逐字保留；v2（W3-A4）新增：
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

