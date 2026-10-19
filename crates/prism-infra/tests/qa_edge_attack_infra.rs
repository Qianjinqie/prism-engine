//! 攻击性边界/协议合规/确定性/安全测试。
//!
//! 覆盖开发者面：`prism-infra::{jsonrpc, mcp, batch, http, plugin, engine}` 与
//! `prism-app::{cmd_batch, cmd_mcp, cmd_http, cmd_pack, cmd_critique}`。
//!
//! 每条测试的 doc 注释说明"失败代表什么缺陷"。协议依据官方规范：
//! - JSON-RPC 2.0（https://www.jsonrpc.org/specification）：错误码 -32700/-32600/
//!   -32601/-32602/-32603；Request 必含 jsonrpc/method；Response 必含 jsonrpc/id 且
//!   result 与 error **互斥**；解析错误时 id 为 null；未知方法 → -32601；
//! - HTTP/1.1（RFC 7230/7231）：请求行 `METHOD SP TARGET SP HTTP/1.1 CRLF`、
//!   头部 `Name: Value CRLF` + 空行、Content-Length 定长正文；畸形 → 400/413/431；
//! - MCP（Model Context Protocol）：initialize / tools/list / tools/call / 通知。
//!
//! 本文件只写测试、不改源码。

use prism_infra::batch::{self, BatchConfig, Task};
use prism_infra::engine::{self, RenderRequest};
use prism_infra::http::{self, HttpError, HttpRequest, HttpResponse, MAX_BODY_BYTES};
use prism_infra::jsonrpc::{self, error_code, Json, Message, Response, RpcError};
use prism_infra::mcp::{McpServer, PROTOCOL_VERSION};
use prism_infra::plugin::{
    parse_semver, validate_license, validate_name, Manifest, PackLibrary,
};
use std::io::{BufReader, Cursor};

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

const SRC: &str = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #c0392b roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.2s fps 4 }\noutput { resolution 16x12 format frames }\n";

fn parse_http(raw: &str) -> Result<HttpRequest, HttpError> {
    let mut r = BufReader::new(Cursor::new(raw.as_bytes().to_vec()));
    http::parse_request(&mut r)
}

/// 调 MCP，返回响应 Json（断言有响应）。
fn mcp_call(server: &McpServer, text: &str) -> Json {
    let raw = server.handle_text(text).unwrap().expect("应为请求并返回响应");
    jsonrpc::parse(&raw).expect("响应必须是合法 JSON")
}

// ===========================================================================
// JSON-RPC 2.0 协议合规
// ===========================================================================

/// 请求往返逐字节一致：jsonrpc/method/params/id 四字段保持。
/// 失败代表：序列化丢字段或键序不稳定 → 跨端协议不兼容。
#[test]
fn jsonrpc_request_round_trip_exact() {
    let text = r#"{"jsonrpc":"2.0","method":"tools/list","params":{},"id":7}"#;
    let msg = jsonrpc::parse_request(text).expect("应可解析");
    match &msg {
        Message::Request { method, id, .. } => {
            assert_eq!(method, "tools/list");
            assert_eq!(id, &Json::int(7));
        }
        other => panic!("应为请求，得到 {other:?}"),
    }
    assert_eq!(msg.to_json().to_string().unwrap(), text, "往返应逐字节一致");
}

/// 非 "2.0" 版本必须被拒绝；缺 jsonrpc 字段必须被拒绝。
/// 失败代表：版本校验缺失 → 接受 1.0 消息导致协议歧义。
#[test]
fn jsonrpc_rejects_bad_version_and_missing_field() {
    assert!(jsonrpc::parse_request(r#"{"jsonrpc":"1.0","method":"x","id":1}"#).is_err());
    assert!(jsonrpc::parse_request(r#"{"jsonrpc":"3.0","method":"x","id":1}"#).is_err());
    assert!(jsonrpc::parse_request(r#"{"method":"x","id":1}"#).is_err(), "缺 jsonrpc 应拒绝");
}

/// 通知 = 无 id 的请求；服务端不应回响应（MCP 层验证）。
/// 失败代表：通知被当作请求回响应 → 协议违规。
#[test]
fn jsonrpc_notification_distinguished_and_not_replied() {
    let msg = jsonrpc::parse_request(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).unwrap();
    assert!(matches!(msg, Message::Notification { .. }), "无 id → 通知");
    let server = McpServer::new();
    assert!(server.handle_text(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).unwrap().is_none());
}

/// 解析错误 → -32700 且 id 为 null（规范：id 无法检测时为 null）。
/// 失败代表：错误码或 id 语义违规。
#[test]
fn jsonrpc_parse_error_returns_minus_32700() {
    let server = McpServer::new();
    let raw = server.handle_text("{not valid json").unwrap().expect("解析错误应有响应");
    let resp = Message::from_json(&jsonrpc::parse(&raw).unwrap()).unwrap();
    match resp {
        Message::Response(r) => {
            assert_eq!(r.error.unwrap().code, error_code::PARSE_ERROR);
            assert_eq!(r.id, Json::Null, "解析错误 id 必须为 null");
        }
        _ => panic!("应为响应"),
    }
}

/// 未知方法 → -32601；响应 id 回填为请求 id。
/// 失败代表：方法分发错误或 id 丢失。
#[test]
fn jsonrpc_method_not_found_code_and_id() {
    let server = McpServer::new();
    let resp = mcp_call(&server, r#"{"jsonrpc":"2.0","method":"bogus","id":"9"}"#);
    assert_eq!(
        resp.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
        error_code::METHOD_NOT_FOUND
    );
    assert_eq!(resp.get("id").unwrap().as_str(), Some("9"), "id 必须回填");
}

/// **互斥性（规范核心）**：成功响应含 result 且**不含** error；错误响应含 error 且**不含** result。
/// 失败代表：同时含 result 与 error → 违反 JSON-RPC 2.0 §5。
#[test]
fn jsonrpc_result_error_mutually_exclusive() {
    // 成功响应。
    let ok = Response::success(Json::int(1), Json::Bool(true)).to_json();
    assert!(ok.get("result").is_some());
    assert!(ok.get("error").is_none(), "成功响应不得含 error");
    // 错误响应。
    let err = Response::failure(Json::int(1), RpcError::new(error_code::INVALID_PARAMS, "bad")).to_json();
    assert!(err.get("error").is_some());
    assert!(err.get("result").is_none(), "错误响应不得含 result");
    // 两者都必须含 jsonrpc=2.0 与 id。
    for r in [&ok, &err] {
        assert_eq!(r.get("jsonrpc").unwrap().as_str(), Some("2.0"));
        assert!(r.get("id").is_some(), "响应必含 id");
    }
}

/// 非法请求（method 非字符串）→ from_json 拒绝；服务端不应崩溃。
/// 失败代表：类型校验缺失 → 后续 .as_str() 静默 None 后误判。
#[test]
fn jsonrpc_invalid_request_type_is_rejected() {
    // method 为数字。
    let v = jsonrpc::parse(r#"{"jsonrpc":"2.0","method":1,"id":1}"#).unwrap();
    assert!(Message::from_json(&v).is_err(), "method 非字符串应拒绝");
    // 顶层非对象。
    let v2 = jsonrpc::parse(r#"[1,2,3]"#).unwrap();
    assert!(Message::from_json(&v2).is_err(), "非对象消息应拒绝");
}

/// 错误对象必须含数值 code + 字符串 message；缺任一则解析失败。
/// 失败代表：错误对象结构校验缺失。
#[test]
fn jsonrpc_error_object_shape_enforced() {
    // 缺 code。
    let v = jsonrpc::parse(r#"{"jsonrpc":"2.0","error":{"message":"x"},"id":1}"#).unwrap();
    assert!(Message::from_json(&v).is_err(), "错误对象缺 code 应拒绝");
    // 缺 message。
    let v2 = jsonrpc::parse(r#"{"jsonrpc":"2.0","error":{"code":-1},"id":1}"#).unwrap();
    assert!(Message::from_json(&v2).is_err(), "错误对象缺 message 应拒绝");
}

/// **确定性**：同请求两次处理响应逐字节一致。
/// 失败代表：响应含非确定内容（如哈希随机）。
#[test]
fn jsonrpc_response_is_bytewise_deterministic() {
    let server = McpServer::new();
    let req = format!(
        r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"compile","arguments":{{"source":{}}}}},"id":1}}"#,
        Json::Str(SRC.to_string()).to_string().unwrap()
    );
    let a = server.handle_text(&req).unwrap().unwrap();
    let b = server.handle_text(&req).unwrap().unwrap();
    assert_eq!(a, b, "同请求应逐字节一致");
}

/// 序列化非有限浮点（NaN/Inf）必须被拒绝（JSON 无此表示）。
/// 失败代表：产出非法 JSON（`NaN` 字面量）。
#[test]
fn jsonrpc_rejects_non_finite_numbers() {
    assert!(Json::Num(f64::NAN).to_string().is_err());
    assert!(Json::Num(f64::INFINITY).to_string().is_err());
    assert!(Json::Num(f64::NEG_INFINITY).to_string().is_err());
}

/// JSON 解析器对畸形输入一律拒绝而非 panic（尾部残留/未闭合/非法转义）。
/// 失败代表：解析器在畸形输入下 panic（库代码 panic = 缺陷）。
#[test]
fn jsonrpc_parser_rejects_malformed_without_panic() {
    for bad in [
        "{} x", "[1,2", "{\"a\":}", "\"unterminated", "{'single':1}", "nul", "tru",
        "[1,,2]", "{\"a\":1,}", "\\uZZZZ",
    ] {
        assert!(jsonrpc::parse(bad).is_err(), "畸形输入应拒绝：{bad}");
    }
    // 深度嵌套对象不应栈溢出（合理深度）。
    let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
    assert!(jsonrpc::parse(&deep).is_ok(), "合理深度的嵌套数组应可解析");
}

/// 数字序列化：整数值无小数尾；-0.0 → "0"。
/// 失败代表：输出引入浮点尾导致往返不稳定。
#[test]
fn jsonrpc_integer_serialization_stable() {
    assert_eq!(Json::Num(1.0).to_string().unwrap(), "1");
    assert_eq!(Json::Num(-0.0).to_string().unwrap(), "0");
    assert_eq!(Json::Num(42.0).to_string().unwrap(), "42");
}

// ===========================================================================
// HTTP/1.1 报文格式与畸形拒绝
// ===========================================================================

/// 简单 GET 解析：请求行 + 头部（小写键）。
/// 失败代表：请求行/头部解析错误。
#[test]
fn http_parses_simple_get() {
    let req = parse_http("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    assert_eq!(req.method, "GET");
    assert_eq!(req.target, "/health");
    assert_eq!(req.headers[0], ("host".to_string(), "localhost".to_string()));
}

/// POST + Content-Length 定长正文解析精确。
/// 失败代表：正文长度解析错 → body 截断/溢出。
#[test]
fn http_parses_post_body_exactly() {
    let body = r#"{"source":"a.pris"}"#;
    let raw = format!("POST /compile HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
    let req = parse_http(&raw).unwrap();
    assert_eq!(req.body.len(), body.len());
    assert_eq!(req.body, body.as_bytes());
    assert_eq!(req.method, "POST");
}

/// 畸形请求行/头部/Content-Length → 400，不进入路由。
/// 失败代表：畸形请求被静默接受或 panic。
#[test]
fn http_rejects_malformed_requests() {
    // 缺版本。
    assert_eq!(parse_http("GET /health\r\n\r\n").unwrap_err().status, 400);
    // 小写方法。
    assert_eq!(parse_http("get /x HTTP/1.1\r\n\r\n").unwrap_err().status, 400);
    // 不支持版本。
    assert_eq!(parse_http("GET /x HTTP/2\r\n\r\n").unwrap_err().status, 400);
    // 头部无冒号。
    assert_eq!(parse_http("GET / HTTP/1.1\r\nBadHeaderNoColon\r\n\r\n").unwrap_err().status, 400);
    // Content-Length 非法。
    assert_eq!(parse_http("POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n").unwrap_err().status, 400);
    // 请求行字段过多。
    assert_eq!(parse_http("GET /x HTTP/1.1 extra\r\n\r\n").unwrap_err().status, 400);
}

/// 超长头部 → 431（防内存放大）。
/// 失败代表：无长度上限 → 内存爆炸（DoS）。
#[test]
fn http_oversized_header_returns_431() {
    let big = "X-Pad: ".to_string() + &"A".repeat(20 * 1024) + "\r\n";
    let raw = format!("GET / HTTP/1.1\r\n{big}\r\n");
    assert_eq!(parse_http(&raw).unwrap_err().status, 431);
}

/// 超大正文（Content-Length > MAX_BODY_BYTES）→ 413，不分配内存。
/// 失败代表：未校验 body 上限 → 巨量分配（DoS）。
#[test]
fn http_oversized_body_returns_413() {
    let raw = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY_BYTES + 1);
    // 仅头部即可触发 413（正文无需实际提供）。
    let err = parse_http(&raw).unwrap_err();
    assert_eq!(err.status, 413);
}

/// 正文缺失（声明长度但流提前结束）→ 400。
/// 失败代表：read_exact 失败未处理 → panic 或读到垃圾。
#[test]
fn http_incomplete_body_returns_400() {
    let raw = "POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort";
    assert_eq!(parse_http(raw).unwrap_err().status, 400);
}

/// 路由：/health 200、未知 404、错误方法 405。
/// 失败代表：路由状态码错误。
#[test]
fn http_routing_status_codes() {
    assert_eq!(http::route(&parse_http("GET /health HTTP/1.1\r\n\r\n").unwrap()).status, 200);
    assert_eq!(http::route(&parse_http("GET /nope HTTP/1.1\r\n\r\n").unwrap()).status, 404);
    assert_eq!(http::route(&parse_http("POST /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap()).status, 405);
}

/// 响应报文格式：状态行 + Content-Type + Content-Length + Connection + 空行 + 正文。
/// 失败代表：报文格式错误 → 客户端无法解析。
#[test]
fn http_response_serialization_format() {
    let resp = HttpResponse::text(200, "hi");
    let bytes = resp.to_bytes();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "状态行");
    assert!(text.contains("Content-Length: 2\r\n"), "Content-Length 正确");
    assert!(text.contains("Content-Type: text/plain; charset=utf-8\r\n"));
    assert!(text.contains("Connection: close\r\n"));
    assert!(text.ends_with("\r\n\r\nhi") || text.ends_with("\r\nhi"), "空行后跟正文");
    // 头部与正文之间必须有空行 CRLF CRLF。
    assert!(text.contains("\r\n\r\n"), "头部与正文间应有空行");
}

/// Content-Length 必须精确等于正文字节数（含多字节 UTF-8）。
/// 失败代表：长度按字符而非字节计 → 客户端读错边界。
#[test]
fn http_content_length_matches_body_bytes() {
    let resp = HttpResponse::json(200, &Json::object(vec![("k", Json::Str("中文".into()))]));
    let bytes = resp.to_bytes();
    let text = String::from_utf8(bytes.clone()).unwrap();
    // 提取 Content-Length，与响应体字节数比较。
    let cl = text
        .lines()
        .find(|l| l.starts_with("Content-Length:"))
        .unwrap()
        .split(':')
        .nth(1)
        .unwrap()
        .trim()
        .parse::<usize>()
        .unwrap();
    let header_end = text.find("\r\n\r\n").unwrap() + 4;
    let body_len = bytes.len() - header_end;
    assert_eq!(cl, body_len, "Content-Length 必须等于正文字节数（UTF-8）");
}

/// route 确定性：同请求同响应字节。
/// 失败代表：响应含非确定内容。
#[test]
fn http_route_is_deterministic() {
    let body = Json::object(vec![("source", Json::Str(SRC.to_string())), ("t", Json::Num(0.0))]).to_string().unwrap();
    let raw = format!("POST /render_frame HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
    let req = parse_http(&raw).unwrap();
    let a = http::route(&req);
    let b = http::route(&req);
    assert_eq!(a.body, b.body, "同请求同响应字节");
    assert_eq!(&a.body[0..4], &[0x89, b'P', b'N', b'G'], "应为 PNG");
}

/// 路径穿越尝试：目标含 `..` 不匹配任何受限路由（精确匹配），返回 404。
/// 失败代表：路径归一化漏洞导致越权访问。
#[test]
fn http_path_traversal_does_not_match_routes() {
    for target in ["/../../etc/passwd", "/health/../..", "/%2e%2e/secret"] {
        let req = parse_http(&format!("GET {target} HTTP/1.1\r\n\r\n")).unwrap();
        assert_eq!(http::route(&req).status, 404, "穿越路径 {target} 不应命中路由");
    }
}

/// query 串被剥离：`/health?x=1` 命中 /health。
/// 失败代表：查询串处理错误。
#[test]
fn http_query_string_is_stripped() {
    let req = parse_http("GET /health?verbose=1 HTTP/1.1\r\n\r\n").unwrap();
    assert_eq!(http::route(&req).status, 200);
}

/// 非法 UTF-8 请求行 → 400（不 panic）。
/// 失败代表：from_utf8 未处理 → panic。
#[test]
fn http_invalid_utf8_line_returns_400() {
    let mut raw = b"GET /he".to_vec();
    raw.push(0xFF); // 非法 UTF-8
    raw.extend_from_slice(b"alth HTTP/1.1\r\n\r\n");
    let mut r = BufReader::new(Cursor::new(raw));
    let err = http::parse_request(&mut r).unwrap_err();
    assert_eq!(err.status, 400);
}

// ===========================================================================
// MCP 工具 schema 与调用
// ===========================================================================

/// initialize 返回协议版本 + serverInfo + capabilities。
/// 失败代表：MCP 握手不合规 → 客户端拒绝连接。
#[test]
fn mcp_initialize_shape() {
    let server = McpServer::new();
    let resp = mcp_call(&server, r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#);
    let result = resp.get("result").unwrap();
    assert_eq!(result.get("protocolVersion").unwrap().as_str(), Some(PROTOCOL_VERSION));
    assert!(result.get("serverInfo").is_some(), "应含 serverInfo");
    assert!(result.get("capabilities").is_some(), "应含 capabilities");
}

/// tools/list 返回 7 个工具，每个含 name/description/inputSchema（object schema + required）。
/// 失败代表：工具 schema 缺失 → LLM 无法构造调用。
#[test]
fn mcp_tools_list_schema_complete() {
    let server = McpServer::new();
    let resp = mcp_call(&server, r#"{"jsonrpc":"2.0","method":"tools/list","id":2}"#);
    let tools = resp.get("result").unwrap().get("tools").unwrap();
    match tools {
        Json::Array(items) => {
            assert_eq!(items.len(), 7, "V2.0-M3 EC-2 后应有 7 个工具");
            for t in items {
                assert!(t.get("name").and_then(|n| n.as_str()).is_some(), "工具须有 name");
                assert!(t.get("description").and_then(|d| d.as_str()).is_some(), "须有 description");
                let schema = t.get("inputSchema").expect("须有 inputSchema");
                assert_eq!(schema.get("type").unwrap().as_str(), Some("object"));
                assert!(schema.get("required").is_some(), "schema 须有 required");
            }
        }
        _ => panic!("tools 应为数组"),
    }
    // 工具名集合（确定性）。
    let names: Vec<&str> = match &tools {
        Json::Array(items) => items.iter().filter_map(|t| t.get("name").and_then(|n| n.as_str())).collect(),
        _ => vec![],
    };
    for expected in [
        "compile", "render_frame", "render_video", "critique",
        "scene_check", "render_submit", "diagnose",
    ] {
        assert!(names.contains(&expected), "应含工具 {expected}");
    }
}

/// EC-2 三工具 round-trip 对拍（最小集）：scene_check / render_submit / diagnose
/// 各一条 请求→响应。失败代表：新工具面未按文档契约回包。
#[test]
fn mcp_ec2_tools_roundtrip() {
    let server = McpServer::new();
    let text_of = |resp: &Json| -> String {
        match resp.get("result").unwrap().get("content").unwrap() {
            Json::Array(items) => items
                .iter()
                .filter(|c| c.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<&str>>()
                .join("\n"),
            _ => panic!("content 应为数组"),
        }
    };
    // scene_check：成功场景 → 文本块含检查结论。
    let resp = mcp_call(
        &server,
        &format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"scene_check","arguments":{{"source":{}}}}},"id":1}}"#,
            Json::Str(SRC.to_string()).to_string().unwrap()
        ),
    );
    assert!(text_of(&resp).contains("场景检查通过"), "scene_check 应通过：{}", text_of(&resp));

    // diagnose：坏源码 → 工具本身成功（无 error），带回行号的诊断。
    let resp = mcp_call(
        &server,
        &format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"diagnose","arguments":{{"source":{}}}}},"id":2}}"#,
            Json::Str("bogus command\n".to_string()).to_string().unwrap()
        ),
    );
    assert!(resp.get("error").is_none(), "诊断工具不得把坏输入升级为 RPC 错误");
    let text = text_of(&resp);
    assert!(text.contains("编译失败") && text.contains("建议："), "诊断须含定位与建议：{text}");

    // render_submit：坏源码 → INVALID_PARAMS（提交侧不留情）。
    let resp = mcp_call(
        &server,
        &format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"render_submit","arguments":{{"source":{}}}}},"id":3}}"#,
            Json::Str("bogus command\n".to_string()).to_string().unwrap()
        ),
    );
    assert_eq!(
        resp.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
        error_code::INVALID_PARAMS
    );
}

/// tools/call render_frame 返回 image 内容块（base64 PNG）。
/// 失败代表：工具返回结构不合 MCP 内容块约定。
#[test]
fn mcp_tools_call_render_frame_image_block() {
    let server = McpServer::new();
    let req = format!(
        r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"render_frame","arguments":{{"source":{},"t":0.0}}}},"id":3}}"#,
        Json::Str(SRC.to_string()).to_string().unwrap()
    );
    let resp = mcp_call(&server, &req);
    let content = resp.get("result").unwrap().get("content").unwrap();
    match content {
        Json::Array(items) => {
            let img = items
                .iter()
                .find(|c| c.get("type").and_then(|t| t.as_str()) == Some("image"))
                .expect("应含 image 块");
            assert_eq!(img.get("mimeType").unwrap().as_str(), Some("image/png"));
            let data = img.get("data").unwrap().as_str().unwrap();
            assert!(!data.is_empty(), "base64 数据不可为空");
            // base64 字符集校验。
            assert!(data.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='));
        }
        _ => panic!("content 应为数组"),
    }
}

/// tools/call 缺 name → -32602；未知工具 → -32602；缺 source → -32602。
/// 失败代表：参数校验缺失 → 内部错误或崩溃。
#[test]
fn mcp_tools_call_param_errors() {
    let server = McpServer::new();
    let resp = mcp_call(&server, r#"{"jsonrpc":"2.0","method":"tools/call","params":{},"id":1}"#);
    assert_eq!(
        resp.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
        error_code::INVALID_PARAMS
    );
    let resp2 = mcp_call(
        &server,
        r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"nope","arguments":{"source":"x"}},"id":2}"#,
    );
    assert_eq!(
        resp2.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
        error_code::INVALID_PARAMS
    );
    // 有 name 但缺 source。
    let resp3 = mcp_call(
        &server,
        r#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"compile","arguments":{}},"id":3}"#,
    );
    assert_eq!(
        resp3.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
        error_code::INVALID_PARAMS
    );
}

/// 服务端收到响应消息 → -32600 非法请求（而非静默）。
/// 失败代表：协议方向校验缺失。
#[test]
fn mcp_server_rejects_incoming_response() {
    let server = McpServer::new();
    let raw = server
        .handle_text(r#"{"jsonrpc":"2.0","result":1,"id":1}"#)
        .unwrap()
        .expect("应回错误");
    let resp = Message::from_json(&jsonrpc::parse(&raw).unwrap()).unwrap();
    match resp {
        Message::Response(r) => assert_eq!(r.error.unwrap().code, error_code::INVALID_REQUEST),
        _ => panic!("应为响应"),
    }
}

// ===========================================================================
// batch 清单解析与笛卡尔积（安全/退化）
// ===========================================================================

/// 合法清单解析 + 笛卡尔积顺序稳定（width → height → fps）。
/// 失败代表：展开顺序不确定 → batch 输出非确定。
#[test]
fn batch_manifest_parse_and_stable_cartesian() {
    let manifest = r#"
[[task]]
source = "demo.prism"
out = "render/demo"
width = [320, 640]
height = [180]
fps = [15, 30]
"#;
    let tasks = batch::parse_manifest(manifest).unwrap();
    assert_eq!(tasks.len(), 1);
    let sub = tasks[0].expand();
    assert_eq!(sub.len(), 4);
    let outs: Vec<&str> = sub.iter().map(|s| s.out.as_str()).collect();
    assert_eq!(
        outs,
        vec![
            "render/demo/w320-h180-f15",
            "render/demo/w320-h180-f30",
            "render/demo/w640-h180-f15",
            "render/demo/w640-h180-f30",
        ]
    );
}

/// 空清单 / 缺 source / 缺 out / 未知键 / 非法整数 → 结构化错误，不 panic。
/// 失败代表：畸形清单崩溃或静默接受。
#[test]
fn batch_rejects_malformed_manifests() {
    assert!(batch::parse_manifest("").is_err(), "空清单应拒绝");
    assert!(batch::parse_manifest("# 只有注释\n").is_err());
    assert!(batch::parse_manifest("[[task]]\nout = \"x\"\n").is_err(), "缺 source");
    assert!(batch::parse_manifest("[[task]]\nsource = \"a\"\n").is_err(), "缺 out");
    assert!(batch::parse_manifest("[[task]]\nsource = \"a\"\nout = \"b\"\nbogus = \"c\"\n").is_err(), "未知键");
    assert!(batch::parse_manifest("[[task]]\nsource = \"a\"\nout = \"b\"\nwidth = [\"x\"]\n").is_err(), "非整数");
    // 键值不在块内。
    assert!(batch::parse_manifest("source = \"a\"\n").is_err(), "块外键值应拒绝");
}

/// 空任务清单的 BatchConfig：run_batch 返回空结果，进度不回调（不 panic）。
/// 失败代表：0 子任务除零/死循环。
#[test]
fn batch_empty_config_is_safe() {
    let cfg = BatchConfig { tasks: vec![], workers: 4 };
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let results = batch::run_batch(&cfg, |d, t| seen.push((d, t)));
    assert!(results.is_empty());
    assert!(seen.is_empty());
}

/// run_batch 对不存在源文件报错而非 panic（不启动渲染）。
/// 失败代表：IO 错误未处理 → 崩溃。
#[test]
fn batch_missing_file_reports_error() {
    let cfg = BatchConfig {
        tasks: vec![Task {
            source: "definitely-does-not-exist-w10.prism".to_string(),
            out: "/tmp/prism-w10-batch".to_string(),
            widths: vec![16],
            heights: vec![16],
            fps: vec![2],
        }],
        workers: 2,
    };
    let results = batch::run_batch(&cfg, |_, _| {});
    assert_eq!(results.len(), 1);
    assert!(results[0].error.is_some(), "缺失文件应报错");
}

/// workers=0 应被规整为 ≥1（不死锁）。
/// 失败代表：0 工作线程 → 结果槽永不回填。
#[test]
fn batch_zero_workers_does_not_deadlock() {
    let cfg = BatchConfig {
        tasks: vec![Task {
            source: "nofile.prism".to_string(),
            out: "/tmp/prism-w10-batch0".to_string(),
            widths: vec![8],
            heights: vec![8],
            fps: vec![1],
        }],
        workers: 0,
    };
    let results = batch::run_batch(&cfg, |_, _| {});
    assert_eq!(results.len(), 1, "workers=0 应规整为 1 并完成");
    assert!(results[0].error.is_some());
}

// ===========================================================================
// plugin manifest 校验（semver/许可证/资产）
// ===========================================================================

/// semver 严格：恰好 x.y.z，非数字段拒绝。
/// 失败代表：宽松 semver 接受非法版本。
#[test]
fn plugin_semver_strict() {
    assert_eq!(parse_semver("1.2.3").unwrap(), (1, 2, 3));
    assert!(parse_semver("1.2").is_err(), "两段应拒绝");
    assert!(parse_semver("1.2.3.4").is_err(), "四段应拒绝");
    assert!(parse_semver("a.b.c").is_err());
    assert!(parse_semver("").is_err());
}

/// 许可证白名单：MIT/Apache-2.0 通过；Proprietary/空 拒绝。
/// 失败代表：非白名单许可证落库（合规风险）。
#[test]
fn plugin_license_allowlist_enforced() {
    assert!(validate_license("MIT").is_ok());
    assert!(validate_license("Apache-2.0").is_ok());
    assert!(validate_license("Proprietary").is_err());
    assert!(validate_license("GPL-3.0").is_err(), "GPL 不在白名单应拒绝");
    assert!(validate_license("").is_err());
}

/// 包名 kebab-case：小写字母开头，仅小写/数字/连字符。
/// 失败代表：非法名（大写/下划线/空/数字开头）被接受。
#[test]
fn plugin_name_kebab_case_enforced() {
    assert!(validate_name("cinematic-basic").is_ok());
    assert!(validate_name("Cinematic").is_err(), "大写应拒绝");
    assert!(validate_name("a_b").is_err(), "下划线应拒绝");
    assert!(validate_name("1pack").is_err(), "数字开头应拒绝");
    assert!(validate_name("").is_err());
}

/// 缺 name / version / license 的 manifest 必须被拒绝。
/// 失败代表：必填字段校验缺失。
#[test]
fn plugin_manifest_missing_required_rejected() {
    assert!(Manifest::parse_toml("version = \"1.0.0\"\nlicense = \"MIT\"\n").is_err(), "缺 name");
    assert!(Manifest::parse_toml("name = \"x\"\nlicense = \"MIT\"\n").is_err(), "缺 version");
    assert!(Manifest::parse_toml("name = \"x\"\nversion = \"1.0.0\"\n").is_err(), "缺 license");
}

/// 无 provides 能力的包在 validate_static 被拒绝。
/// 失败代表：空能力包落库（无意义）。
#[test]
fn plugin_manifest_requires_provides() {
    let m = Manifest::parse_toml("name = \"x-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\n").unwrap();
    assert!(m.validate_static().is_err(), "无 provides 应拒绝");
}

/// 非法许可证的 manifest 通过 parse 但在 validate_static 被拒。
/// 失败代表：许可证校验被绕过。
#[test]
fn plugin_manifest_bad_license_rejected_at_validate() {
    let m = Manifest::parse_toml("name = \"x-pack\"\nversion = \"1.0.0\"\nlicense = \"Nope\"\nprovides = [\"a\"]\n").unwrap();
    assert!(m.validate_static().is_err());
}

/// library add/list/remove 往返；重复 add 拒绝；缺资产拒绝。
/// 失败代表：库索引读写或资产校验错误。
#[test]
fn plugin_library_round_trip_and_asset_check() {
    let tmp = std::env::temp_dir().join(format!("prism-w10-pack-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let pack_dir = tmp.join("mypack");
    std::fs::create_dir_all(&pack_dir).unwrap();
    let good = "name = \"w10-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nprovides = [\"a\"]\n";
    std::fs::write(pack_dir.join("manifest.toml"), good).unwrap();

    let lib = PackLibrary::new(tmp.join("lib"));
    assert!(lib.list().is_empty());
    let entry = lib.add(&pack_dir).expect("add 应成功");
    assert_eq!(entry.name, "w10-pack");
    assert_eq!(lib.list().len(), 1);
    assert!(lib.add(&pack_dir).is_err(), "重复 add 应拒绝");
    let removed = lib.remove("w10-pack").expect("remove 应成功");
    assert_eq!(removed.name, "w10-pack");
    assert!(lib.list().is_empty());
    assert!(lib.remove("w10-pack").is_err(), "二次 remove 应失败");

    // 缺资产包。
    let asset_pack = tmp.join("ap");
    std::fs::create_dir_all(&asset_pack).unwrap();
    std::fs::write(
        asset_pack.join("manifest.toml"),
        "name = \"asset-pack\"\nversion = \"1.0.0\"\nlicense = \"MIT\"\nprovides = [\"a\"]\nassets = [\"assets/font.ttf\"]\n",
    )
    .unwrap();
    assert!(lib.add(&asset_pack).is_err(), "缺资产应拒绝");

    let _ = std::fs::remove_dir_all(&tmp);
}

// ===========================================================================
// engine：确定性 + 退化
// ===========================================================================

/// 编译摘要与 DSL 一致；非法源码报行号。
/// 失败代表：引擎门面与 DSL 脱节。
#[test]
fn engine_compile_summary_and_error_line() {
    let s = engine::compile_summary(SRC).unwrap();
    assert_eq!((s.width, s.height, s.fps), (16, 12, 4));
    assert_eq!(s.format, "frames");
    let err = engine::compile_source("bogus command\n").unwrap_err();
    assert!(err.line >= 1, "应带行号");
}

/// 渲染确定性：同帧两次 PNG 逐字节一致且为 PNG 魔数。
/// 失败代表：渲染含隐藏状态。
#[test]
fn engine_render_frame_deterministic() {
    let a = engine::render_frame_png(SRC, 0.0, &RenderRequest::default()).unwrap();
    let b = engine::render_frame_png(SRC, 0.0, &RenderRequest::default()).unwrap();
    assert_eq!(a, b);
    assert_eq!(&a[0..4], &[0x89, b'P', b'N', b'G']);
}

/// 零分辨率覆盖 → 结构化错误（不 panic）。
/// 失败代表：0 尺寸渲染崩溃。
#[test]
fn engine_zero_resolution_rejected() {
    let req = RenderRequest { width: Some(0), height: Some(10), fps: None };
    assert!(engine::render_frame(SRC, 0.0, &req).is_err());
}

/// render_frames 帧数 = fps × duration；fps=0 或 duration=0 → 错误。
/// 失败代表：帧数计算错误或 0 帧静默。
#[test]
fn engine_render_frames_count_and_zero_guard() {
    let frames = engine::render_frames(SRC, &RenderRequest::default()).unwrap();
    assert_eq!(frames.len(), 1, "默认 4fps×0.2s 应产出 1 帧（引擎口径）");
    let req0 = RenderRequest { width: None, height: None, fps: Some(0) };
    assert!(engine::render_frames(SRC, &req0).is_err(), "fps=0 应拒绝");
}

// ===========================================================================
// 协议「超长输入内存爆炸」防护
// ===========================================================================

/// 超长 JSON-RPC 字符串/数组：解析不 panic（可能在合理规模内成功）。
/// 失败代表：无界解析导致栈/内存爆炸。
#[test]
fn jsonrpc_long_input_no_panic() {
    // 大数组（10 万元素）。
    let arr = format!("[{}]", vec!["1"; 100_000].join(","));
    assert!(jsonrpc::parse(&arr).is_ok(), "大数组应可解析且不 panic");
    // 长字符串。
    let long = format!("\"{}\"", "x".repeat(1_000_000));
    assert!(jsonrpc::parse(&long).is_ok(), "长字符串应可解析");
}

/// HTTP 请求行超长（无 CRLF）→ 431 或 400（不无限读）。
/// 失败代表：无长度上限读取 → 内存爆炸。
#[test]
fn http_oversized_request_line_bounded() {
    // 请求行超过 MAX_HEADER_BYTES。
    let raw = format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(20 * 1024));
    let err = parse_http(&raw).unwrap_err();
    assert!(err.status == 431 || err.status == 400, "超长请求行应被限制，实得 {}", err.status);
}
