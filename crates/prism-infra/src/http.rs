//! 自写 HTTP/1.1 服务器（零第三方框架）。
//!
//! 语义权威：《强大版路线图》§3.6（"HTTP（自写 HTTP/1.1，局域网渲染服务）"、
//! "自写 HTTP"）与 RFC 7230/7231（HTTP/1.1 报文格式）。
//!
//! # 范围（最小可用）
//! - 请求行 `METHOD SP TARGET SP HTTP/1.1 CRLF`；
//! - 头部 `Name: Value CRLF`，以空行 `CRLF CRLF` 结束；
//! - 可选 `Content-Length` 定长正文（不支持 chunked / 多段）；
//! - 响应：状态行 + `Content-Type` / `Content-Length` / `Connection` + 正文。
//! - 路由：`GET /health`、`GET /tools`、`POST /render_frame`（JSON 入参 → PNG）、
//!   `POST /compile`（JSON → 元信息）。**默认绑 `127.0.0.1`（安全红线：不暴露
//!   公网）**。
//!
//! # 安全
//! - 畸形请求（非法请求行 / 头部缺失 CRLF / Content-Length 非法 / 方法不认识）
//!   → 400 Bad Request，且**不进入路由**；
//! - 请求行/头部总长上限（[`MAX_HEADER_BYTES`]）防内存放大；
//! - 不解析 URL query（最小面），target 按精确路径匹配。

use crate::engine::{self, RenderRequest};
use crate::jsonrpc::{parse as json_parse, Json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// 请求行 + 头部最大字节数（超限回 431）。
pub const MAX_HEADER_BYTES: usize = 16 * 1024;
/// 请求正文最大字节数（超限回 413）。
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// 解析后的 HTTP 请求（最小模型）。
#[derive(Debug, Clone, PartialEq)]
pub struct HttpRequest {
    /// 方法（大写，如 GET / POST）。
    pub method: String,
    /// 请求目标（路径，精确匹配）。
    pub target: String,
    /// 头部（名小写 → 值，保持插入序）。
    pub headers: Vec<(String, String)>,
    /// 正文（定长，可能为空）。
    pub body: Vec<u8>,
}

/// HTTP 解析错误（畸形请求；携带应回的状态码）。
#[derive(Debug, Clone, PartialEq)]
pub struct HttpError {
    /// 应回的状态码（400 / 413 / 431）。
    pub status: u16,
    /// 面向人类的原因。
    pub reason: String,
}

impl HttpError {
    fn new(status: u16, reason: impl Into<String>) -> Self {
        HttpError { status, reason: reason.into() }
    }
}

/// HTTP 响应。
#[derive(Debug, Clone, PartialEq)]
pub struct HttpResponse {
    /// 状态码。
    pub status: u16,
    /// 内容类型（如 `application/json` / `image/png`）。
    pub content_type: String,
    /// 正文。
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// 构造响应。
    pub fn new(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        HttpResponse { status, content_type: content_type.to_string(), body }
    }

    /// JSON 响应。
    pub fn json(status: u16, value: &Json) -> Self {
        let text: String = value.to_string().unwrap_or_else(|_| "{}".to_string());
        HttpResponse::new(status, "application/json", text.into_bytes())
    }

    /// 纯文本响应。
    pub fn text(status: u16, text: &str) -> Self {
        HttpResponse::new(status, "text/plain; charset=utf-8", text.as_bytes().to_vec())
    }

    /// 序列化为 HTTP/1.1 报文（含状态行 + 头部 + 空行 + 正文）。
    pub fn to_bytes(&self) -> Vec<u8> {
        let reason: &str = status_reason(self.status);
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(format!("HTTP/1.1 {} {}\r\n", self.status, reason).as_bytes());
        out.extend_from_slice(format!("Content-Type: {}\r\n", self.content_type).as_bytes());
        out.extend_from_slice(format!("Content-Length: {}\r\n", self.body.len()).as_bytes());
        out.extend_from_slice(b"Connection: close\r\n");
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(&self.body);
        out
    }
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

/// 从读取器解析一个 HTTP 请求（读请求行 + 头部 + 定长正文）。
///
/// # Errors
/// 畸形请求 → [`HttpError`]（携带应回状态码）。
pub fn parse_request<R: Read>(reader: &mut BufReader<R>) -> Result<HttpRequest, HttpError> {
    // 1) 请求行。
    let request_line: String = read_line(reader, MAX_HEADER_BYTES)?;
    let mut parts = request_line.trim_end().split(' ');
    let method: String = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| HttpError::new(400, "请求行缺少方法"))?
        .to_string();
    let target: String = parts
        .next()
        .ok_or_else(|| HttpError::new(400, "请求行缺少请求目标"))?
        .to_string();
    let version: &str = parts
        .next()
        .ok_or_else(|| HttpError::new(400, "请求行缺少 HTTP 版本"))?;
    if parts.next().is_some() {
        return Err(HttpError::new(400, "请求行字段过多"));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(HttpError::new(400, format!("不支持的 HTTP 版本 '{}'", version)));
    }
    if !method.chars().all(|c| c.is_ascii_uppercase()) || method.is_empty() {
        return Err(HttpError::new(400, format!("非法方法 '{}'", method)));
    }

    // 2) 头部（直到空行为止）。
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut total: usize = request_line.len();
    loop {
        let line: String = read_line(reader, MAX_HEADER_BYTES)?;
        total += line.len();
        if total > MAX_HEADER_BYTES {
            return Err(HttpError::new(431, "请求头过大"));
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        let (name, value): (&str, &str) = line
            .trim_end()
            .split_once(':')
            .ok_or_else(|| HttpError::new(400, format!("头部 '{}' 缺少冒号", line.trim_end())))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }

    // 3) 定长正文。
    let content_length: Option<usize> = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map(|(_, v)| v.parse::<usize>().map_err(|_| HttpError::new(400, "Content-Length 非法")))
        .transpose()?;
    let mut body: Vec<u8> = Vec::new();
    if let Some(len) = content_length {
        if len > MAX_BODY_BYTES {
            return Err(HttpError::new(413, "请求正文过大"));
        }
        body.resize(len, 0);
        reader
            .read_exact(&mut body)
            .map_err(|_| HttpError::new(400, "请求正文读取不完整"))?;
    }

    Ok(HttpRequest { method, target, headers, body })
}

fn read_line<R: Read>(reader: &mut BufReader<R>, limit: usize) -> Result<String, HttpError> {
    let mut buf: Vec<u8> = Vec::new();
    let n: usize = reader
        .read_until(b'\n', &mut buf)
        .map_err(|_| HttpError::new(400, "读取请求行失败"))?;
    if n == 0 {
        return Err(HttpError::new(400, "连接在请求完成前关闭"));
    }
    if buf.len() > limit {
        return Err(HttpError::new(431, "请求行或头部过长"));
    }
    String::from_utf8(buf).map_err(|_| HttpError::new(400, "请求行/头部不是合法 UTF-8"))
}

/// 路由：把请求映射为响应（纯函数，便于单测）。
pub fn route(req: &HttpRequest) -> HttpResponse {
    // 剥离查询串（最小面：不解析 query，仅按路径匹配）。
    let path: &str = req.target.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("GET", "/health") => HttpResponse::json(
            200,
            &Json::object(vec![
                ("status", Json::Str("ok".to_string())),
                ("engine", Json::Str("prism".to_string())),
                ("version", Json::Str(env!("CARGO_PKG_VERSION").to_string())),
            ]),
        ),
        ("GET", "/tools") => {
            let server = crate::mcp::McpServer::new();
            let tools: Vec<Json> = server
                .tools()
                .into_iter()
                .map(|t| {
                    Json::object(vec![
                        ("name", Json::Str(t.name)),
                        ("description", Json::Str(t.description)),
                    ])
                })
                .collect();
            HttpResponse::json(200, &Json::object(vec![("tools", Json::Array(tools))]))
        }
        ("POST", "/compile") => handle_compile(req),
        ("POST", "/render_frame") => handle_render_frame(req),
        ("GET", "/render_frame") | ("POST", "/health") | ("GET", "/compile") => {
            HttpResponse::text(405, "405 Method Not Allowed")
        }
        _ => HttpResponse::text(404, "404 Not Found"),
    }
}

fn handle_compile(req: &HttpRequest) -> HttpResponse {
    let body: Json = match json_parse(&String::from_utf8_lossy(&req.body)) {
        Ok(v) => v,
        Err(e) => return HttpResponse::json(400, &error_json(&format!("请求体非法 JSON: {}", e))),
    };
    let source: &str = match body.get("source").and_then(|s| s.as_str()) {
        Some(s) => s,
        None => return HttpResponse::json(400, &error_json("缺少字符串字段 'source'")),
    };
    match engine::compile_summary(source) {
        Ok(s) => HttpResponse::json(
            200,
            &Json::object(vec![
                ("width", Json::int(s.width as i64)),
                ("height", Json::int(s.height as i64)),
                ("fps", Json::int(s.fps as i64)),
                ("duration", Json::Num(s.duration as f64)),
                ("format", Json::Str(s.format)),
            ]),
        ),
        Err(e) => HttpResponse::json(400, &error_json(&e.to_string())),
    }
}

fn handle_render_frame(req: &HttpRequest) -> HttpResponse {
    let body: Json = match json_parse(&String::from_utf8_lossy(&req.body)) {
        Ok(v) => v,
        Err(e) => return HttpResponse::json(400, &error_json(&format!("请求体非法 JSON: {}", e))),
    };
    let source: &str = match body.get("source").and_then(|s| s.as_str()) {
        Some(s) => s,
        None => return HttpResponse::json(400, &error_json("缺少字符串字段 'source'")),
    };
    let t: f32 = body.get("t").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let rreq: RenderRequest = RenderRequest {
        width: body.get("width").and_then(|v| v.as_f64()).map(|f| f.max(0.0) as u32),
        height: body.get("height").and_then(|v| v.as_f64()).map(|f| f.max(0.0) as u32),
        fps: None,
    };
    match engine::render_frame_png(source, t, &rreq) {
        Ok(png) => HttpResponse::new(200, "image/png", png),
        Err(e) => HttpResponse::json(400, &error_json(&e.to_string())),
    }
}

fn error_json(message: &str) -> Json {
    Json::object(vec![("error", Json::Str(message.to_string()))])
}

/// 在给定地址上运行 HTTP 服务器（阻塞；单连接串行处理，最小可用）。
///
/// `on_request(req, resp)` 在每个请求处理后被回调（便于测试 / 日志）。
///
/// # Errors
/// 绑定失败 / 接受连接失败 → `io::Error`。
pub fn serve<F>(addr: &str, mut on_request: F) -> std::io::Result<()>
where
    F: FnMut(&HttpRequest, &HttpResponse),
{
    let listener: TcpListener = TcpListener::bind(addr)?;
    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let resp: HttpResponse = handle_connection(&mut conn);
                on_request(&HttpRequest::new_empty(), &resp);
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

impl HttpRequest {
    /// 空占位请求（用于服务器回调签名简化）。
    pub fn new_empty() -> Self {
        HttpRequest { method: String::new(), target: String::new(), headers: Vec::new(), body: Vec::new() }
    }
}

/// 处理一个 TCP 连接：读请求 → 路由 → 写响应。
pub fn handle_connection(conn: &mut TcpStream) -> HttpResponse {
    let mut reader: BufReader<&mut TcpStream> = BufReader::new(conn);
    let resp: HttpResponse = match parse_request(&mut reader) {
        Ok(req) => route(&req),
        Err(err) => HttpResponse::text(err.status, &err.reason),
    };
    let _ = reader.into_inner().write_all(&resp.to_bytes());
    resp
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_str(raw: &str) -> Result<HttpRequest, HttpError> {
        let mut r = BufReader::new(Cursor::new(raw.as_bytes().to_vec()));
        parse_request(&mut r)
    }

    #[test]
    fn parses_simple_get() {
        let req = parse_str("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("应可解析");
        assert_eq!(req.method, "GET");
        assert_eq!(req.target, "/health");
        assert_eq!(req.headers[0], ("host".to_string(), "localhost".to_string()));
    }

    #[test]
    fn parses_post_with_body() {
        let raw = "POST /compile HTTP/1.1\r\nContent-Length: 18\r\n\r\n{\"source\":\"a.pris\"}";
        let req = parse_str(raw).expect("应可解析");
        assert_eq!(req.body.len(), 18);
    }

    #[test]
    fn rejects_malformed_request_line() {
        assert!(parse_str("GET /health\r\n\r\n").is_err(), "缺版本应拒绝");
        assert!(parse_str("get /x HTTP/1.1\r\n\r\n").is_err(), "小写方法应拒绝");
        assert!(parse_str("GET /x HTTP/2\r\n\r\n").is_err(), "HTTP/2 应拒绝");
    }

    #[test]
    fn rejects_bad_header() {
        assert!(parse_str("GET / HTTP/1.1\r\nBadHeaderNoColon\r\n\r\n").is_err());
    }

    #[test]
    fn rejects_bad_content_length() {
        assert!(parse_str("POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n").is_err());
    }

    #[test]
    fn route_health_ok() {
        let req = parse_str("GET /health HTTP/1.1\r\n\r\n").unwrap();
        let resp = route(&req);
        assert_eq!(resp.status, 200);
        assert_eq!(resp.content_type, "application/json");
    }

    #[test]
    fn route_unknown_is_404() {
        let req = parse_str("GET /nope HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(route(&req).status, 404);
    }

    #[test]
    fn route_wrong_method_is_405() {
        let req = parse_str("POST /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        assert_eq!(route(&req).status, 405);
    }

    #[test]
    fn response_serializes_with_content_length() {
        let resp = HttpResponse::text(200, "hi");
        let bytes = resp.to_bytes();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.ends_with("\r\nhi"));
    }

    #[test]
    fn route_compile_returns_summary() {
        let src = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #808080 roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.5s fps 2 }\noutput { resolution 8x8 format frames }\n";
        let body = Json::object(vec![("source", Json::Str(src.to_string()))]).to_string().unwrap();
        let raw = format!("POST /compile HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
        let req = parse_str(&raw).unwrap();
        let resp = route(&req);
        assert_eq!(resp.status, 200);
        let v = json_parse(&String::from_utf8(resp.body).unwrap()).unwrap();
        assert_eq!(v.get("width").unwrap().as_f64(), Some(8.0));
    }

    #[test]
    fn route_compile_bad_json_is_400() {
        let body = "{bad";
        let raw = format!("POST /compile HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
        let req = parse_str(&raw).unwrap();
        assert_eq!(route(&req).status, 400);
    }

    #[test]
    fn render_frame_http_is_deterministic() {
        let src = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #808080 roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.5s fps 2 }\noutput { resolution 8x8 format frames }\n";
        let body = Json::object(vec![("source", Json::Str(src.to_string())), ("t", Json::Num(0.0))]).to_string().unwrap();
        let raw = format!("POST /render_frame HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
        let req = parse_str(&raw).unwrap();
        let a = route(&req);
        let b = route(&req);
        assert_eq!(a.body, b.body);
        assert_eq!(a.content_type, "image/png");
        assert_eq!(&a.body[0..4], &[0x89, b'P', b'N', b'G']);
    }
}
