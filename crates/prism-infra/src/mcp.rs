//! MCP server（Model Context Protocol）——自写 JSON-RPC，零第三方 SDK。
//!
//! 语义权威：《强大版路线图》§3.6（"MCP stdio JSON-RPC：compile/render_frame/
//! render_video/critique"，"自写 JSON-RPC"）与 MCP 规范
//! （<https://spec.modelcontextprotocol.io/>）。
//!
//! # 设计
//! MCP 是建立在 JSON-RPC 2.0 之上的协议：本模块实现 **stdio 传输**、**初始化
//! 握手**、**tools/list**、**tools/call** 三类消息，工具集来自引擎能力：
//! - `compile`：编译 DSL，返回场景元信息（分辨率 / 帧率 / 时长 / 格式）；
//! - `render_frame`：渲染一帧，返回 base64 PNG（MCP 图像内容块）；
//! - `render_video`：整段渲染，返回帧数 + 逐帧 base64 PNG 摘要（首帧内容块）；
//! - `critique`：确定性质量诊断（占位：返回帧的均值 / 极值等客观统计）。
//!
//! # 确定性
//! 无全局 RNG；`render_frame` 同输入同字节。工具描述（`inputSchema`）由参数
//! 定义静态生成（路线图 §2.4："schema 由节点注册表自动生成"的当前落地形态）。

use crate::engine::{self, RenderRequest};
use crate::jsonrpc::{error_code, Json, Message, Response, RpcError};

/// MCP 协议版本（与客户端 `initialize` 协商的缺省版本）。
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP 工具定义（名称 + 描述 + JSON Schema 参数）。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    /// 工具名。
    pub name: String,
    /// 面向人类 / LLM 的描述。
    pub description: String,
    /// JSON Schema（object 形态）参数说明。
    pub input_schema: Json,
}

/// MCP 服务端（无状态；每次调用独立，便于逐字节复现与测试）。
#[derive(Debug, Default, Clone)]
pub struct McpServer;

impl McpServer {
    /// 构造服务端。
    pub fn new() -> Self {
        McpServer
    }

    /// 工具清单（引擎能力面）。
    pub fn tools(&self) -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "compile".to_string(),
                description: "把 Prism DSL 源码编译为场景，返回分辨率/帧率/时长/格式元信息。"
                    .to_string(),
                input_schema: object_schema(&[("source", "string")], &["source"]),
            },
            ToolDef {
                name: "render_frame".to_string(),
                description:
                    "渲染指定时刻的单帧，返回 PNG（MCP image 内容块，base64 编码）。"
                        .to_string(),
                input_schema: object_schema(
                    &[("source", "string"), ("t", "number"), ("width", "integer"), ("height", "integer")],
                    &["source"],
                ),
            },
            ToolDef {
                name: "render_video".to_string(),
                description: "整段渲染：按 fps 均匀采样，返回帧数与首帧 PNG（base64）。"
                    .to_string(),
                input_schema: object_schema(
                    &[("source", "string"), ("width", "integer"), ("height", "integer"), ("fps", "integer")],
                    &["source"],
                ),
            },
            ToolDef {
                name: "critique".to_string(),
                description: "对指定时刻的帧做确定性客观诊断（亮度均值/极值、动态范围）。"
                    .to_string(),
                input_schema: object_schema(
                    &[("source", "string"), ("t", "number"), ("width", "integer"), ("height", "integer")],
                    &["source"],
                ),
            },
        ]
    }

    /// 处理一条 JSON-RPC 文本消息，返回可选的响应文本（通知 → None）。
    ///
    /// # Errors
    /// 仅在响应序列化失败时返回错误（协议内错误以 [`Response::failure`] 表达）。
    pub fn handle_text(&self, text: &str) -> Result<Option<String>, String> {
        let msg: Message = match crate::jsonrpc::parse_request(text) {
            Ok(m) => m,
            Err(err) => {
                // 解析失败：按 JSON-RPC 规范回 -32700（id 用 null）。
                let resp: Response =
                    Response::failure(Json::Null, RpcError::new(error_code::PARSE_ERROR, err.message));
                return resp.to_json().to_string().map(Some).map_err(|e| e.message);
            }
        };
        match msg {
            Message::Notification { .. } => Ok(None),
            Message::Request { method, params, id } => {
                let resp: Response = self.dispatch(&method, &params, id);
                resp.to_json().to_string().map(Some).map_err(|e| e.message)
            }
            Message::Response(_) => {
                // 服务端不应收到响应；按非法请求回复。
                let resp: Response = Response::failure(
                    Json::Null,
                    RpcError::new(error_code::INVALID_REQUEST, "服务端不应接收响应消息"),
                );
                resp.to_json().to_string().map(Some).map_err(|e| e.message)
            }
        }
    }

    /// 方法分发（MCP + JSON-RPC 基础方法）。
    pub fn dispatch(&self, method: &str, params: &Json, id: Json) -> Response {
        match method {
            "initialize" => Response::success(id, self.initialize_result()),
            "notifications/initialized" => Response::success(id, Json::Null),
            "ping" => Response::success(id, Json::Object(vec![])),
            "tools/list" => Response::success(id, self.tools_list_result()),
            "tools/call" => match self.call_tool(params) {
                Ok(result) => Response::success(id, result),
                Err(err) => Response::failure(id, err),
            },
            other => Response::failure(
                id,
                RpcError::new(error_code::METHOD_NOT_FOUND, format!("未知方法 '{}'", other)),
            ),
        }
    }

    fn initialize_result(&self) -> Json {
        Json::object(vec![
            ("protocolVersion", Json::Str(PROTOCOL_VERSION.to_string())),
            (
                "serverInfo",
                Json::object(vec![
                    ("name", Json::Str("prism-engine".to_string())),
                    ("version", Json::Str(env!("CARGO_PKG_VERSION").to_string())),
                ]),
            ),
            (
                "capabilities",
                Json::object(vec![("tools", Json::Object(vec![]))]),
            ),
        ])
    }

    fn tools_list_result(&self) -> Json {
        let tools: Vec<Json> = self
            .tools()
            .into_iter()
            .map(|t| {
                Json::object(vec![
                    ("name", Json::Str(t.name)),
                    ("description", Json::Str(t.description)),
                    ("inputSchema", t.input_schema),
                ])
            })
            .collect();
        Json::object(vec![("tools", Json::Array(tools))])
    }

    fn call_tool(&self, params: &Json) -> Result<Json, RpcError> {
        let name: &str = params
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| RpcError::new(error_code::INVALID_PARAMS, "tools/call 缺少 'name'"))?;
        let args: &Json = params.get("arguments").unwrap_or(&Json::Null);
        let source: &str = args
            .get("source")
            .and_then(|s| s.as_str())
            .ok_or_else(|| RpcError::new(error_code::INVALID_PARAMS, "缺少字符串参数 'source'"))?;
        let req: RenderRequest = RenderRequest {
            width: int_arg(args, "width"),
            height: int_arg(args, "height"),
            fps: int_arg(args, "fps"),
        };
        match name {
            "compile" => self.tool_compile(source),
            "render_frame" => {
                let t: f32 = float_arg(args, "t").unwrap_or(0.0);
                self.tool_render_frame(source, t, &req)
            }
            "render_video" => self.tool_render_video(source, &req),
            "critique" => {
                let t: f32 = float_arg(args, "t").unwrap_or(0.0);
                self.tool_critique(source, t, &req)
            }
            other => Err(RpcError::new(
                error_code::INVALID_PARAMS,
                format!("未知工具 '{}'", other),
            )),
        }
    }

    fn tool_compile(&self, source: &str) -> Result<Json, RpcError> {
        let s: engine::CompileSummary = engine::compile_summary(source).map_err(dsl_error)?;
        Ok(tool_text_result(&format!(
            "编译成功：{}x{} @ {}fps, {}s, format={}",
            s.width, s.height, s.fps, s.duration, s.format
        )))
    }

    fn tool_render_frame(&self, source: &str, t: f32, req: &RenderRequest) -> Result<Json, RpcError> {
        let png: Vec<u8> = engine::render_frame_png(source, t, req).map_err(dsl_error)?;
        Ok(image_result(&png, &format!("frame t={}s", t)))
    }

    fn tool_render_video(&self, source: &str, req: &RenderRequest) -> Result<Json, RpcError> {
        let frames: Vec<prism_gpu::Framebuffer> =
            engine::render_frames(source, req).map_err(dsl_error)?;
        let first_png: Vec<u8> = prism_io::encode_png(&frames[0]);
        let content: Vec<Json> = vec![
            Json::object(vec![
                ("type", Json::Str("text".to_string())),
                ("text", Json::Str(format!("渲染 {} 帧（首帧为图像内容块）", frames.len()))),
            ]),
            image_content(&first_png),
        ];
        Ok(Json::object(vec![("content", Json::Array(content))]))
    }

    fn tool_critique(&self, source: &str, t: f32, req: &RenderRequest) -> Result<Json, RpcError> {
        let fb: prism_gpu::Framebuffer = engine::render_frame(source, t, req).map_err(dsl_error)?;
        let (mean, min, max): (f64, u8, u8) = frame_stats(&fb);
        Ok(tool_text_result(&format!(
            "诊断 t={}s：均值 {:.3}，最暗 {}，最亮 {}，动态范围 {}",
            t,
            mean,
            min,
            max,
            max as i32 - min as i32
        )))
    }
}

/// 提取 `width`/`height`/`fps` 整型参数。
fn int_arg(args: &Json, key: &str) -> Option<u32> {
    args.get(key).and_then(|v| v.as_f64()).map(|f| f.max(0.0) as u32)
}

/// 提取浮点参数。
fn float_arg(args: &Json, key: &str) -> Option<f32> {
    args.get(key).and_then(|v| v.as_f64()).map(|f| f as f32)
}

/// 引擎错误 → JSON-RPC 内部错误（保留 DSL 行号信息）。
fn dsl_error(err: engine::EngineError) -> RpcError {
    RpcError::new(error_code::INVALID_PARAMS, err.to_string())
}

/// 构造 `object` 类型 JSON Schema。
fn object_schema(props: &[(&str, &str)], required: &[&str]) -> Json {
    let properties: Vec<(String, Json)> = props
        .iter()
        .map(|(name, ty)| {
            (
                name.to_string(),
                Json::object(vec![("type", Json::Str(ty.to_string()))]),
            )
        })
        .collect();
    Json::object(vec![
        ("type", Json::Str("object".to_string())),
        ("properties", Json::Object(properties)),
        (
            "required",
            Json::Array(required.iter().map(|r| Json::Str(r.to_string())).collect()),
        ),
    ])
}

/// 纯文本工具结果（MCP `content` 数组）。
fn tool_text_result(text: &str) -> Json {
    Json::object(vec![(
        "content",
        Json::Array(vec![Json::object(vec![
            ("type", Json::Str("text".to_string())),
            ("text", Json::Str(text.to_string())),
        ])]),
    )])
}

/// 图像内容块（MCP image：base64 PNG）。
fn image_content(png: &[u8]) -> Json {
    Json::object(vec![
        ("type", Json::Str("image".to_string())),
        ("data", Json::Str(base64_encode(png))),
        ("mimeType", Json::Str("image/png".to_string())),
    ])
}

/// 图像工具结果（MCP `content` 数组，单图像块 + 说明文本）。
fn image_result(png: &[u8], caption: &str) -> Json {
    Json::object(vec![(
        "content",
        Json::Array(vec![
            Json::object(vec![
                ("type", Json::Str("text".to_string())),
                ("text", Json::Str(format!("{}（PNG {} 字节，base64）", caption, png.len()))),
            ]),
            image_content(png),
        ]),
    )])
}

/// 标准 base64 编码（无第三方依赖）。
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out: String = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0: u8 = chunk[0];
        let b1: u8 = *chunk.get(1).unwrap_or(&0);
        let b2: u8 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// 帧客观统计：亮度均值（0..1）、最暗 / 最亮通道值（0..255）。
fn frame_stats(fb: &prism_gpu::Framebuffer) -> (f64, u8, u8) {
    let mut sum: u64 = 0;
    let mut min: u8 = 255;
    let mut max: u8 = 0;
    let mut count: u64 = 0;
    for px in fb.pixels.chunks_exact(4) {
        for &c in &px[0..3] {
            sum += c as u64;
            min = min.min(c);
            max = max.max(c);
            count += 1;
        }
    }
    let mean: f64 = if count == 0 { 0.0 } else { sum as f64 / count as f64 / 255.0 };
    (mean, min, max)
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonrpc::{Message, error_code};

    const SRC: &str = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #e64d1a roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.25s fps 4 }\noutput { resolution 32x24 format frames }\n";

    fn call(server: &McpServer, text: &str) -> Json {
        let raw: String = server.handle_text(text).unwrap().expect("请求应返回响应");
        crate::jsonrpc::parse(&raw).expect("响应应是合法 JSON")
    }

    #[test]
    fn initialize_returns_protocol_version() {
        let server = McpServer::new();
        let resp: Json = call(&server, r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#);
        assert_eq!(resp.get("result").unwrap().get("protocolVersion").unwrap().as_str(), Some(PROTOCOL_VERSION));
    }

    #[test]
    fn tools_list_contains_four_tools() {
        let server = McpServer::new();
        let resp: Json = call(&server, r#"{"jsonrpc":"2.0","method":"tools/list","id":2}"#);
        let tools: &Json = resp.get("result").unwrap().get("tools").unwrap();
        match tools {
            Json::Array(items) => {
                assert_eq!(items.len(), 4);
                let names: Vec<&str> = items.iter().filter_map(|t| t.get("name").and_then(|n| n.as_str())).collect();
                assert!(names.contains(&"render_frame"));
            }
            _ => panic!("tools 应为数组"),
        }
    }

    #[test]
    fn tools_call_render_frame_returns_png_image_block() {
        let server = McpServer::new();
        let req: String = format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"render_frame","arguments":{{"source":{},"t":0.0}}}},"id":3}}"#,
            Json::Str(SRC.to_string()).to_string().unwrap()
        );
        let resp: Json = call(&server, &req);
        let content: &Json = resp.get("result").unwrap().get("content").unwrap();
        match content {
            Json::Array(items) => {
                let img: &Json = items.iter().find(|c| c.get("type").and_then(|t| t.as_str()) == Some("image")).expect("应含 image 块");
                assert_eq!(img.get("mimeType").unwrap().as_str(), Some("image/png"));
                assert!(!img.get("data").unwrap().as_str().unwrap().is_empty());
            }
            _ => panic!("content 应为数组"),
        }
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let server = McpServer::new();
        let resp: Json = call(&server, r#"{"jsonrpc":"2.0","method":"bogus","id":9}"#);
        let code = resp.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64;
        assert_eq!(code, error_code::METHOD_NOT_FOUND);
    }

    #[test]
    fn notification_yields_no_response() {
        let server = McpServer::new();
        assert!(server.handle_text(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).unwrap().is_none());
    }

    #[test]
    fn malformed_json_returns_parse_error() {
        let server = McpServer::new();
        let raw: String = server.handle_text("{not json").unwrap().unwrap();
        let resp: Message = Message::from_json(&crate::jsonrpc::parse(&raw).unwrap()).unwrap();
        match resp {
            Message::Response(r) => assert_eq!(r.error.unwrap().code, error_code::PARSE_ERROR),
            _ => panic!("应为响应"),
        }
    }

    #[test]
    fn render_frame_is_deterministic_across_calls() {
        let server = McpServer::new();
        let req: String = format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"render_frame","arguments":{{"source":{},"t":0.0}}}},"id":1}}"#,
            Json::Str(SRC.to_string()).to_string().unwrap()
        );
        let a: String = server.handle_text(&req).unwrap().unwrap();
        let b: String = server.handle_text(&req).unwrap().unwrap();
        assert_eq!(a, b, "同请求应逐字节一致");
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
