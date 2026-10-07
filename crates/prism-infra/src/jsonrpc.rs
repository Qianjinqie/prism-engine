//! 自写 JSON-RPC 2.0 编解码（零第三方依赖）。
//!
//! 语义权威："MCP stdio JSON-RPC"、"自写 JSON-RPC"与
//! JSON-RPC 2.0 规范（<https://www.jsonrpc.org/specification>）。本模块只做
//! **协议层**：值的解析 / 序列化、请求 / 通知 / 响应的建模与往返。MCP 语义
//! （工具列表 / 工具调用）见 [`crate::mcp`]。
//!
//! # 设计
//! - 值模型 [`Json`]：null / bool / number（f64）/ string / array / object。
//!   对象保留插入顺序（`Vec<(String, Json)>`），保证确定性序列化。
//! - 手写递归下降解析器（无 `serde`）；非有限浮点（NaN/Inf）在序列化时拒绝
//!   （JSON 无此表示）。
//! - 错误码遵循规范：-32700 解析错误 / -32600 非法请求 / -32601 方法不存在 /
//!   -32602 参数非法 / -32603 内部错误。
//!
//! # 硬约束
//! - 零第三方依赖（仅 `std`）；
//! - 无全局 RNG；同输入必同输出（对象键序稳定 = 逐字节可复现）。

use std::fmt;

// ---------------------------------------------------------------------------
// 值模型
// ---------------------------------------------------------------------------

/// JSON 值（对象键序稳定，保证确定性序列化）。
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`。
    Null,
    /// `true` / `false`。
    Bool(bool),
    /// 数值（统一以 f64 表示；序列化时对整数值去掉小数尾）。
    Num(f64),
    /// 字符串。
    Str(String),
    /// 数组。
    Array(Vec<Json>),
    /// 对象（键序即插入序，重复键后者覆盖前者）。
    Object(Vec<(String, Json)>),
}

impl Json {
    /// 便捷构造整数值。
    pub fn int(v: i64) -> Json {
        Json::Num(v as f64)
    }

    /// 便捷构造对象（一次性从切片构建）。
    pub fn object(fields: Vec<(&str, Json)>) -> Json {
        Json::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// 若为字符串，返回其借用。
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// 若为数值，返回其值。
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// 按键取对象字段（仅对象有效）。
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// 对象里 `id` 字段（用于响应回填）。
    pub fn id_field(&self) -> Option<&Json> {
        self.get("id")
    }

    /// 序列化为 JSON 文本（确定性：键序即插入序）。
    ///
    /// # Errors
    /// 数值为 NaN / Infinity 时返回 [`JsonError`]（JSON 无此表示）。
    pub fn to_string(&self) -> Result<String, JsonError> {
        let mut out: String = String::new();
        write_json(self, &mut out)?;
        Ok(out)
    }
}

fn write_json(value: &Json, out: &mut String) -> Result<(), JsonError> {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if !n.is_finite() {
                return Err(JsonError::new(format!(
                    "无法序列化非有限浮点数 {}（JSON 无此表示）",
                    n
                )));
            }
            // 整数值以无小数点形式输出，保证 `1.0` → `1`（确定性、可读）。
            if *n == n.trunc() && n.abs() < 1e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        Json::Str(s) => write_json_string(s, out),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out)?;
            }
            out.push(']');
        }
        Json::Object(fields) => {
            out.push('{');
            for (i, (k, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(k, out);
                out.push(':');
                write_json(v, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// 解析器（手写递归下降）
// ---------------------------------------------------------------------------

/// JSON 编解码错误。
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError {
    /// 面向人类的错误消息。
    pub message: String,
}

impl JsonError {
    /// 构造错误。
    pub fn new(message: impl Into<String>) -> Self {
        JsonError { message: message.into() }
    }
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}

/// 解析 JSON 文本为 [`Json`] 值（忽略首尾空白）。
///
/// # Errors
/// 语法非法 / 尾部有残留内容 → [`JsonError`]。
pub fn parse(text: &str) -> Result<Json, JsonError> {
    let mut p = Parser { chars: text.chars().collect(), pos: 0 };
    p.skip_ws();
    let value: Json = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(JsonError::new(format!(
            "JSON 尾部存在多余内容（位置 {}）",
            p.pos
        )));
    }
    Ok(value)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, JsonError> {
        self.skip_ws();
        match self.peek() {
            Some('n') => self.parse_literal("null", Json::Null),
            Some('t') => self.parse_literal("true", Json::Bool(true)),
            Some('f') => self.parse_literal("false", Json::Bool(false)),
            Some('"') => Ok(Json::Str(self.parse_string()?)),
            Some('[') => self.parse_array(),
            Some('{') => self.parse_object(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(JsonError::new(format!("意外的 JSON 字符 '{}'（位置 {}）", c, self.pos))),
            None => Err(JsonError::new("JSON 意外结束")),
        }
    }

    fn parse_literal(&mut self, lit: &str, value: Json) -> Result<Json, JsonError> {
        for expected in lit.chars() {
            if self.bump() != Some(expected) {
                return Err(JsonError::new(format!("字面量 '{}' 解析失败（位置 {}）", lit, self.pos)));
            }
        }
        Ok(value)
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        if self.bump() != Some('"') {
            return Err(JsonError::new("字符串缺少起始引号"));
        }
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err(JsonError::new("字符串未闭合")),
                Some('"') => break,
                Some('\\') => {
                    let esc = self.bump().ok_or_else(|| JsonError::new("转义序列未完成"))?;
                    match esc {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'b' => s.push('\u{0008}'),
                        'f' => s.push('\u{000C}'),
                        'u' => {
                            let hex: String = (0..4)
                                .map(|_| self.bump().unwrap_or('\0'))
                                .collect();
                            let code = u32::from_str_radix(&hex, 16)
                                .map_err(|_| JsonError::new(format!("非法 \\u 转义 '{}'", hex)))?;
                            let ch = char::from_u32(code)
                                .ok_or_else(|| JsonError::new("\\u 编码非法 Unicode 标量"))?;
                            s.push(ch);
                        }
                        other => {
                            return Err(JsonError::new(format!("未知转义 '\\{}'", other)));
                        }
                    }
                }
                Some(c) => s.push(c),
            }
        }
        Ok(s)
    }

    fn parse_number(&mut self) -> Result<Json, JsonError> {
        let start: usize = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let n: f64 = text
            .parse()
            .map_err(|_| JsonError::new(format!("非法数字 '{}'", text)))?;
        Ok(Json::Num(n))
    }

    fn parse_array(&mut self) -> Result<Json, JsonError> {
        self.bump(); // '['
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                _ => return Err(JsonError::new(format!("数组缺少 ',' 或 ']'（位置 {}）", self.pos))),
            }
        }
        Ok(Json::Array(items))
    }

    fn parse_object(&mut self) -> Result<Json, JsonError> {
        self.bump(); // '{'
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(Json::Object(fields));
        }
        loop {
            self.skip_ws();
            let key: String = self.parse_string()?;
            self.skip_ws();
            if self.bump() != Some(':') {
                return Err(JsonError::new(format!("对象键后缺少 ':'（位置 {}）", self.pos)));
            }
            let value: Json = self.parse_value()?;
            if let Some(slot) = fields.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                fields.push((key, value));
            }
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                _ => return Err(JsonError::new(format!("对象缺少 ',' 或 '}}'（位置 {}）", self.pos))),
            }
        }
        Ok(Json::Object(fields))
    }
}

// ---------------------------------------------------------------------------
// 协议层：请求 / 通知 / 响应 + 错误码
// ---------------------------------------------------------------------------

/// JSON-RPC 2.0 标准错误码。
pub mod error_code {
    /// 解析错误（服务端收到的不是合法 JSON）。
    pub const PARSE_ERROR: i64 = -32700;
    /// 非法请求（不是合法的请求对象）。
    pub const INVALID_REQUEST: i64 = -32600;
    /// 方法不存在。
    pub const METHOD_NOT_FOUND: i64 = -32601;
    /// 参数非法。
    pub const INVALID_PARAMS: i64 = -32602;
    /// 内部错误。
    pub const INTERNAL_ERROR: i64 = -32603;
}

/// JSON-RPC 消息：请求 / 通知 / 响应三类（按 `method` / `id` 字段区分）。
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// 请求（有 `method` 且有 `id`）。
    Request {
        /// 方法名。
        method: String,
        /// 参数（可为 null）。
        params: Json,
        /// 请求标识（回填到响应）。
        id: Json,
    },
    /// 通知（有 `method` 但无 `id`），服务端不应回响应。
    Notification {
        /// 方法名。
        method: String,
        /// 参数（可为 null）。
        params: Json,
    },
    /// 响应（有 `result` 或 `error`）。
    Response(Response),
}

/// JSON-RPC 响应。
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// 回填的请求标识。
    pub id: Json,
    /// 成功结果（与 `error` 互斥）。
    pub result: Option<Json>,
    /// 错误（与 `result` 互斥）。
    pub error: Option<RpcError>,
}

/// JSON-RPC 错误对象。
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    /// 错误码。
    pub code: i64,
    /// 错误消息。
    pub message: String,
    /// 可选补充数据（当前未用）。
    pub data: Option<Json>,
}

impl RpcError {
    /// 构造错误（无 data）。
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        RpcError { code, message: message.into(), data: None }
    }
}

impl Response {
    /// 成功响应。
    pub fn success(id: Json, result: Json) -> Self {
        Response { id, result: Some(result), error: None }
    }

    /// 错误响应。
    pub fn failure(id: Json, error: RpcError) -> Self {
        Response { id, result: None, error: Some(error) }
    }

    /// 序列化响应为 JSON 文本。
    pub fn to_json(&self) -> Json {
        let mut fields: Vec<(String, Json)> = vec![("jsonrpc".to_string(), Json::Str("2.0".to_string()))];
        if let Some(result) = &self.result {
            fields.push(("result".to_string(), result.clone()));
        }
        if let Some(err) = &self.error {
            let mut err_fields: Vec<(String, Json)> =
                vec![("code".to_string(), Json::int(err.code)), ("message".to_string(), Json::Str(err.message.clone()))];
            if let Some(data) = &err.data {
                err_fields.push(("data".to_string(), data.clone()));
            }
            fields.push(("error".to_string(), Json::Object(err_fields)));
        }
        fields.push(("id".to_string(), self.id.clone()));
        Json::Object(fields)
    }
}

impl Message {
    /// 从 JSON 值解析消息（不含版本校验，见 [`parse_request`]）。
    ///
    /// # Errors
    /// 结构非法（缺字段 / 类型错误）→ [`JsonError`]。
    pub fn from_json(value: &Json) -> Result<Message, JsonError> {
        let obj = match value {
            Json::Object(_) => value,
            _ => return Err(JsonError::new("JSON-RPC 消息必须是对象")),
        };
        let has_result: bool = obj.get("result").is_some();
        let has_error: bool = obj.get("error").is_some();
        if has_result || has_error {
            let id: Json = obj.id_field().cloned().unwrap_or(Json::Null);
            let result: Option<Json> = obj.get("result").cloned();
            let error: Option<RpcError> = match obj.get("error") {
                Some(err) => Some(parse_rpc_error(err)?),
                None => None,
            };
            return Ok(Message::Response(Response { id, result, error }));
        }
        let method: String = obj
            .get("method")
            .and_then(|m| m.as_str())
            .ok_or_else(|| JsonError::new("请求缺少字符串字段 'method'"))?
            .to_string();
        let params: Json = obj.get("params").cloned().unwrap_or(Json::Null);
        match obj.id_field() {
            Some(id) => Ok(Message::Request { method, params, id: id.clone() }),
            None => Ok(Message::Notification { method, params }),
        }
    }

    /// 序列化消息为 JSON 值。
    pub fn to_json(&self) -> Json {
        match self {
            Message::Request { method, params, id } => Json::Object(vec![
                ("jsonrpc".to_string(), Json::Str("2.0".to_string())),
                ("method".to_string(), Json::Str(method.clone())),
                ("params".to_string(), params.clone()),
                ("id".to_string(), id.clone()),
            ]),
            Message::Notification { method, params } => Json::Object(vec![
                ("jsonrpc".to_string(), Json::Str("2.0".to_string())),
                ("method".to_string(), Json::Str(method.clone())),
                ("params".to_string(), params.clone()),
            ]),
            Message::Response(resp) => resp.to_json(),
        }
    }
}

fn parse_rpc_error(value: &Json) -> Result<RpcError, JsonError> {
    let code: i64 = value
        .get("code")
        .and_then(|c| c.as_f64())
        .map(|f| f as i64)
        .ok_or_else(|| JsonError::new("错误对象缺少数值字段 'code'"))?;
    let message: String = value
        .get("message")
        .and_then(|m| m.as_str())
        .ok_or_else(|| JsonError::new("错误对象缺少字符串字段 'message'"))?
        .to_string();
    let data: Option<Json> = value.get("data").cloned();
    Ok(RpcError { code, message, data })
}

/// 解析一个 JSON-RPC 请求对象（含 `jsonrpc` 版本校验），返回请求或通知。
///
/// MCP 场景下客户端只会发请求/通知，故本函数只解析这两类；响应由客户端消费
/// （[`Message::from_json`]）。
///
/// # Errors
/// 非法 JSON / 缺 `jsonrpc` / 版本非 "2.0" / 缺 method → [`JsonError`]。
pub fn parse_request(text: &str) -> Result<Message, JsonError> {
    let value: Json = parse(text)?;
    let version: &str = value
        .get("jsonrpc")
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonError::new("请求缺少字符串字段 'jsonrpc'"))?;
    if version != "2.0" {
        return Err(JsonError::new(format!("不支持的 JSON-RPC 版本 '{}'", version)));
    }
    Message::from_json(&value)
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_primitives() {
        for text in ["null", "true", "false", "42", "-3.5", "\"hi\""] {
            let parsed: Json = parse(text).expect("应可解析");
            let back: String = parsed.to_string().expect("应可序列化");
            let reparsed: Json = parse(&back).expect("往返应可解析");
            assert_eq!(parsed, reparsed, "往返不稳定：{text}");
        }
    }

    #[test]
    fn round_trip_nested_object_preserves_order() {
        let text: &str = r#"{"b":1,"a":[1,2,{"x":"y"}],"z":null}"#;
        let parsed: Json = parse(text).expect("应可解析");
        let back: String = parsed.to_string().expect("应可序列化");
        assert_eq!(back, text, "键序应保持插入序（确定性）");
    }

    #[test]
    fn parse_rejects_trailing_garbage() {
        assert!(parse("{} x").is_err());
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("[1,2").is_err());
    }

    #[test]
    fn string_escapes_round_trip() {
        let j: Json = Json::Str("a\"b\\c\n\t\u{0001}".to_string());
        let text: String = j.to_string().unwrap();
        assert_eq!(parse(&text).unwrap(), j);
    }

    #[test]
    fn request_round_trip() {
        let text: &str = r#"{"jsonrpc":"2.0","method":"tools/list","params":{},"id":7}"#;
        let msg: Message = parse_request(text).expect("应可解析");
        match &msg {
            Message::Request { method, id, .. } => {
                assert_eq!(method, "tools/list");
                assert_eq!(id, &Json::int(7));
            }
            other => panic!("应为请求，得到 {:?}", other),
        }
        let back: String = msg.to_json().to_string().unwrap();
        assert_eq!(back, text, "请求往返应逐字节一致");
    }

    #[test]
    fn notification_has_no_id() {
        let text: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let msg: Message = parse_request(text).expect("应可解析");
        assert!(matches!(msg, Message::Notification { .. }));
    }

    #[test]
    fn response_round_trip_with_error_code() {
        let resp: Response = Response::failure(
            Json::int(1),
            RpcError::new(error_code::METHOD_NOT_FOUND, "方法不存在"),
        );
        let text: String = resp.to_json().to_string().unwrap();
        let reparsed: Message = Message::from_json(&parse(&text).unwrap()).unwrap();
        match reparsed {
            Message::Response(r) => {
                assert_eq!(r.id, Json::int(1));
                let err: RpcError = r.error.expect("应含错误");
                assert_eq!(err.code, error_code::METHOD_NOT_FOUND);
            }
            other => panic!("应为响应，得到 {:?}", other),
        }
    }

    #[test]
    fn rejects_bad_version() {
        let text: &str = r#"{"jsonrpc":"1.0","method":"x","id":1}"#;
        let err: JsonError = parse_request(text).expect_err("版本错误应拒绝");
        assert!(err.message.contains("版本"), "消息应提及版本：{}", err.message);
    }

    #[test]
    fn serialize_rejects_non_finite() {
        assert!(Json::Num(f64::NAN).to_string().is_err());
        assert!(Json::Num(f64::INFINITY).to_string().is_err());
    }

    #[test]
    fn integer_numbers_have_no_float_tail() {
        assert_eq!(Json::Num(1.0).to_string().unwrap(), "1");
        assert_eq!(Json::Num(-0.0).to_string().unwrap(), "0");
    }
}
