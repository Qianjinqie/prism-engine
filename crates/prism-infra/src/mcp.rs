//! MCP server（Model Context Protocol）——自写 JSON-RPC，零第三方 SDK。
//!
//! 语义权威："MCP stdio JSON-RPC：compile/render_frame/
//! render_video/critique"，"自写 JSON-RPC"与 MCP 规范
//! （<https://spec.modelcontextprotocol.io/>）。
//!
//! # 设计
//! MCP 是建立在 JSON-RPC 2.0 之上的协议：本模块实现 **stdio 传输**、**初始化
//! 握手**、**tools/list**、**tools/call** 三类消息，工具集来自引擎能力：
//! - `compile`：编译 DSL，返回场景元信息（分辨率 / 帧率 / 时长 / 格式）；
//! - `render_frame`：渲染一帧，返回 base64 PNG（MCP 图像内容块）；
//! - `render_video`：整段渲染，返回帧数 + 逐帧 base64 PNG 摘要（首帧内容块）；
//! - `critique`：确定性质量诊断（占位：返回帧的均值 / 极值等客观统计）；
//! - `scene_check`：场景检查（preflight）——编译 + 首帧试渲 + 帧数预估；
//! - `render_submit`：渲染提交——确定性 job id（内容寻址）+ 逐帧 MD5 摘要
//!   + 首帧 PNG（内容指纹可回喂农场做产物校验）；
//! - `diagnose`：错误诊断——编译失败给出三段式定位（行号 / 消息 / 建议，
//!   **诊断工具本身不因输入非法而失败**），编译成功给出帧统计 + 病理旗标
//!   （全黑 / 全白 / 动态范围坍缩）。
//!
//! # 确定性
//! 无全局 RNG；`render_frame` 同输入同字节。工具描述（`inputSchema`）由参数
//! 定义静态生成（"schema 由节点注册表自动生成"的当前落地形态）。

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
            ToolDef {
                name: "scene_check".to_string(),
                description: "场景检查（提交前 preflight）：编译 + 首帧试渲 + 帧数预估，\
                              失败时带 DSL 行号。"
                    .to_string(),
                input_schema: object_schema(
                    &[
                        ("source", "string"),
                        ("width", "integer"),
                        ("height", "integer"),
                        ("fps", "integer"),
                    ],
                    &["source"],
                ),
            },
            ToolDef {
                name: "render_submit".to_string(),
                description: "渲染提交：返回确定性 job id（提交参数内容寻址）、\
                              帧数与逐帧 MD5 摘要链、首帧 PNG（MCP image 内容块）。"
                    .to_string(),
                input_schema: object_schema(
                    &[
                        ("source", "string"),
                        ("width", "integer"),
                        ("height", "integer"),
                        ("fps", "integer"),
                    ],
                    &["source"],
                ),
            },
            ToolDef {
                name: "diagnose".to_string(),
                description: "错误诊断：编译失败 → 三段式定位（行号/消息/建议，工具本身成功）；\
                              编译成功 → 帧统计（均值/极值/动态范围）+ 病理旗标\
                              （全黑/全白/动态范围坍缩）。"
                    .to_string(),
                input_schema: object_schema(
                    &[
                        ("source", "string"),
                        ("t", "number"),
                        ("width", "integer"),
                        ("height", "integer"),
                        ("fps", "integer"),
                    ],
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
            "scene_check" => self.tool_scene_check(source, &req),
            "render_submit" => self.tool_render_submit(source, &req),
            "diagnose" => {
                let t: f32 = float_arg(args, "t").unwrap_or(0.0);
                self.tool_diagnose(source, t, &req)
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

    // -----------------------------------------------------------------------
    // V2.0-M3 EC-2：场景检查 / 渲染提交 / 错误诊断
    // -----------------------------------------------------------------------

    /// **场景检查**（提交前 preflight）：编译 + 首帧试渲 + 帧数预估。
    ///
    /// 与 `compile` 的分工：`compile` 只回答"能不能编译"，`scene_check` 回答
    /// "按当前参数能不能真的出片、预计出几帧"——分辨率非法、空场景这类只在
    /// 渲染期暴露的问题被提前到提交前（C20：不静默容错）。
    ///
    /// # Errors
    /// 编译失败 → INVALID_PARAMS（携带行号）；首帧试渲失败 → INVALID_PARAMS
    /// （显式上报，不用"编译通过"蒙混）。
    fn tool_scene_check(&self, source: &str, req: &RenderRequest) -> Result<Json, RpcError> {
        let compiled: prism_dsl::CompileOutput =
            engine::compile_source(source).map_err(dsl_error)?;
        let summary: engine::CompileSummary = engine::summary_of(&compiled);
        let width: u32 = req.width.unwrap_or(summary.width);
        let height: u32 = req.height.unwrap_or(summary.height);
        let fps: u32 = req.fps.unwrap_or(summary.fps);
        if width == 0 || height == 0 {
            return Err(RpcError::new(
                error_code::INVALID_PARAMS,
                "场景检查失败：渲染分辨率必须为正（宽/高 > 0）——当前参数会产出 0 像素帧",
            ));
        }
        let frames: usize = engine::frame_count(fps, summary.duration);
        let mut text: String = format!(
            "场景检查通过：{width}x{height} @ {fps}fps, {}s, format={}\n预估帧数：{frames}（fps × duration，向上取整）",
            summary.duration, summary.format
        );
        if frames == 0 {
            // 场景合法但交不出片：如实警告（提交必败，让调用方在提交前知道）。
            text.push_str(
                "\n警告：按当前 fps/时长预估帧数为 0——提交渲染将无法产出任何帧",
            );
        } else {
            // 首帧试渲：编译通过 ≠ 能出片（空节点 / 资产缺失只在渲染期暴露）。
            let fb: prism_gpu::Framebuffer =
                engine::render_frame(source, 0.0, req).map_err(dsl_error)?;
            text.push_str(&format!(
                "\n首帧试渲：OK（{}x{} RGBA = {} 字节）",
                width,
                height,
                fb.pixels.len()
            ));
        }
        Ok(tool_text_result(&text))
    }

    /// **渲染提交**：确定性 job id（内容寻址）+ 帧数 + 逐帧 MD5 摘要链 + 首帧 PNG。
    ///
    /// `job = md5_hex("submit|source|w|h|fps")`：同源码同参数 ⇒ 同 job id，
    /// 跨进程 / 跨机器复现（内容寻址，非自增计数——自增 id 依赖提交顺序，
    /// 会破坏确定性）。逐帧 MD5 摘要是产物的内容指纹：分布式渲染（farm）
    /// 合并后可用同一公式对拍，证明并行产物 == 串行产物。
    ///
    /// # Errors
    /// 编译失败 / 分辨率非法 / 0 帧 → INVALID_PARAMS（携带行号或事实）。
    fn tool_render_submit(&self, source: &str, req: &RenderRequest) -> Result<Json, RpcError> {
        let summary: engine::CompileSummary = engine::compile_summary(source).map_err(dsl_error)?;
        let width: u32 = req.width.unwrap_or(summary.width);
        let height: u32 = req.height.unwrap_or(summary.height);
        let fps: u32 = req.fps.unwrap_or(summary.fps);
        if width == 0 || height == 0 {
            return Err(RpcError::new(
                error_code::INVALID_PARAMS,
                "渲染提交失败：渲染分辨率必须为正（宽/高 > 0）",
            ));
        }
        // 确定性 job id：提交参数的纯函数（同输入同 id，与提交次数无关）。
        let job: String = prism_render::farm::md5_hex(
            format!("submit|{source}|{width}|{height}|{fps}").as_bytes(),
        );
        let frames: Vec<prism_gpu::Framebuffer> =
            engine::render_frames(source, req).map_err(dsl_error)?;
        // 逐帧摘要链：md5(每帧 RGBA 像素)，按帧号顺序（= 渲染顺序）拼接。
        let digests: Vec<String> = frames
            .iter()
            .map(|fb| prism_render::farm::md5_hex(fb.pixels.as_slice()))
            .collect();
        let first_png: Vec<u8> = prism_io::encode_png(&frames[0]);
        let content: Vec<Json> = vec![
            Json::object(vec![
                ("type", Json::Str("text".to_string())),
                (
                    "text",
                    Json::Str(format!(
                        "渲染提交成功：job={job}\n帧数：{}（{width}x{height} @ {fps}fps, {}s）\n逐帧摘要：{}",
                        frames.len(),
                        summary.duration,
                        digests.join(",")
                    )),
                ),
            ]),
            image_content(&first_png),
        ];
        Ok(Json::object(vec![("content", Json::Array(content))]))
    }

    /// **错误诊断**：编译失败 → 三段式定位（行号 / 消息 / 建议）；
    /// 编译成功 → 帧统计 + 病理旗标（全黑 / 全白 / 动态范围坍缩）。
    ///
    /// 关键语义：诊断工具**不把"坏输入"当协议错误**——调用方要的是"哪里坏了、
    /// 怎么修"的结构化文本，而不是一个 -32602。编译失败照样返回 `result`
    /// （RPC 成功），这才配叫"诊断"。
    ///
    /// # Errors
    /// 仅当编译通过但**渲染**失败（分辨率非法等）才回 INVALID_PARAMS——
    /// 那是诊断能力内的事实，不能伪装成成功文本。
    fn tool_diagnose(&self, source: &str, t: f32, req: &RenderRequest) -> Result<Json, RpcError> {
        let compiled: prism_dsl::CompileOutput = match engine::compile_source(source) {
            Ok(out) => out,
            Err(err) => {
                // 失败路径：诊断本身成功（工具调用返回 result，非 error）。
                let (core, suggestion) = split_dsl_message(&err.message, err.line);
                let location: String = if err.line > 0 {
                    format!("第 {} 行", err.line)
                } else {
                    "引擎级（非行定位）".to_string()
                };
                return Ok(tool_text_result(&format!(
                    "诊断完成：编译失败（诊断工具本身成功——失败是诊断对象，不是协议错误）\n\
                     位置：{location}\n\
                     消息：{core}\n\
                     建议：{suggestion}"
                )));
            }
        };
        let summary: engine::CompileSummary = engine::summary_of(&compiled);
        let fb: prism_gpu::Framebuffer = engine::render_frame(source, t, req).map_err(dsl_error)?;
        let (mean, min, max): (f64, u8, u8) = frame_stats(&fb);
        let flags: Vec<&str> = pathology_flags(min, max);
        let flag_text: String = if flags.is_empty() {
            "未见明显病理（全黑 / 全白 / 动态范围坍缩均未命中）".to_string()
        } else {
            format!("命中病理：{}", flags.join("、"))
        };
        Ok(tool_text_result(&format!(
            "诊断完成：场景可编译（{width}x{height} @ {fps}fps, {}s）\n\
             t={t}s 帧统计：均值 {mean:.3}，最暗 {min}，最亮 {max}，动态范围 {range}\n\
             病理旗标：{flag_text}",
            summary.duration,
            width = summary.width,
            height = summary.height,
            fps = summary.fps,
            range = max as i32 - min as i32,
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

/// 动态范围坍缩判定阈值（0..255 通道值域）。
///
/// 全黑 / 全白之外，"近乎一片灰"（最暗与最亮差 < 16）同样是无信息帧——
/// 照度坍缩 / 材质全同 / 相机打穿都会表现为此。
const DYNAMIC_RANGE_COLLAPSE_THRESHOLD: u8 = 16;

/// 病理旗标（纯函数）：按优先级最多命中一个——全黑 ⊃ 全白 ⊃ 动态范围坍缩。
///
/// 优先级而非并列：全黑帧的"坍缩"是冗余信息，报一个最有辨识度的即可；
/// 返回 `Vec` 是为后续旗标扩展留形（当前 0..=1 个元素）。
fn pathology_flags(min: u8, max: u8) -> Vec<&'static str> {
    if max == 0 {
        vec!["全黑"]
    } else if min == 255 {
        vec!["全白"]
    } else if max - min < DYNAMIC_RANGE_COLLAPSE_THRESHOLD {
        vec!["动态范围坍缩"]
    } else {
        Vec::new()
    }
}

/// 拆分 DSL 错误消息内嵌的建议段（DSL 侧按 `消息\n  建议：…` 书写）。
///
/// 返回 `(核心消息, 建议)`。无内嵌建议时给确定性兜底（C19：诊断必含建议段，
/// 不许只丢一句"错了"给调用方）。
fn split_dsl_message(message: &str, line: usize) -> (String, String) {
    const MARKER: &str = "建议：";
    if let Some(idx) = message.find(MARKER) {
        let suggestion: &str = message[idx + MARKER.len()..].trim();
        if !suggestion.is_empty() {
            let core: String = message[..idx].trim().to_string();
            return (core, suggestion.to_string());
        }
    }
    (
        message.trim().to_string(),
        format!(
            "对照 docs/dsl-reference.md 核对第 {line} 行附近的语法（花括号 / 引号 / \
             字段名是否配对）；若无法自行定位，请附上该行及上下文反馈给引擎维护者"
        ),
    )
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
    fn tools_list_contains_seven_tools_including_ec2_faces() {
        let server = McpServer::new();
        let resp: Json = call(&server, r#"{"jsonrpc":"2.0","method":"tools/list","id":2}"#);
        let tools: &Json = resp.get("result").unwrap().get("tools").unwrap();
        match tools {
            Json::Array(items) => {
                assert_eq!(items.len(), 7, "EC-2 后应为 7 个工具");
                let names: Vec<&str> = items.iter().filter_map(|t| t.get("name").and_then(|n| n.as_str())).collect();
                for expected in [
                    "compile", "render_frame", "render_video", "critique",
                    "scene_check", "render_submit", "diagnose",
                ] {
                    assert!(names.contains(&expected), "应含工具 {expected}");
                }
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

    // -----------------------------------------------------------------------
    // EC-2 round-trip 对拍：三工具各 成功 + 失败 一对（最小充分集）
    // -----------------------------------------------------------------------

    /// 4fps × 0.4s → 2 帧（摘要链测试需要多帧）。
    const SUBMIT_SRC: &str = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #e64d1a roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.4s fps 4 }\noutput { resolution 32x24 format frames }\n";

    /// 坏输入（DSL 首行即错，行号确定 = 1）。
    const BAD_SRC: &str = "bogus command\n";

    fn call_tool(server: &McpServer, name: &str, source: &str) -> Json {
        let req: String = format!(
            r#"{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"{}","arguments":{{"source":{}}}}},"id":1}}"#,
            name,
            Json::Str(source.to_string()).to_string().unwrap()
        );
        call(server, &req)
    }

    /// 取工具结果里的全部文本块（拼接）。
    fn result_text(resp: &Json) -> String {
        match resp.get("result").unwrap().get("content").unwrap() {
            Json::Array(items) => items
                .iter()
                .filter(|c| c.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<&str>>()
                .join("\n"),
            _ => panic!("content 应为数组"),
        }
    }

    fn error_message(resp: &Json) -> String {
        resp.get("error").unwrap().get("message").unwrap().as_str().unwrap().to_string()
    }

    #[test]
    fn scene_check_roundtrip_passes_scene_and_reports_failure_with_line() {
        let server = McpServer::new();
        // 成功对拍：检查报告含元信息 + 帧数预估 + 试渲结论。
        let ok: Json = call_tool(&server, "scene_check", SRC);
        let text: String = result_text(&ok);
        assert!(text.contains("场景检查通过"), "应通过：{text}");
        assert!(text.contains("32x24 @ 4fps"), "应带生效参数：{text}");
        assert!(text.contains("预估帧数：1"), "4fps×0.25s ⇒ 1 帧：{text}");
        assert!(text.contains("首帧试渲：OK"), "应真渲一帧：{text}");
        // 失败对拍：INVALID_PARAMS + 行号（预检不许静默放过坏场景）。
        let bad: Json = call_tool(&server, "scene_check", BAD_SRC);
        let code = bad.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64;
        assert_eq!(code, error_code::INVALID_PARAMS);
        assert!(error_message(&bad).contains("第 1 行"), "必须带行号：{}", error_message(&bad));
    }

    #[test]
    fn render_submit_roundtrip_returns_content_addressed_job_and_per_frame_digests() {
        let server = McpServer::new();
        let ok: Json = call_tool(&server, "render_submit", SUBMIT_SRC);
        let text: String = result_text(&ok);
        assert!(text.contains("渲染提交成功：job="), "应带 job id：{text}");
        assert!(text.contains("帧数：2"), "4fps×0.4s ⇒ 2 帧：{text}");
        // 逐帧摘要链：2 条 32 位小写十六进制 MD5，逗号分隔。
        let digest_line: &str = text
            .lines()
            .find(|l| l.starts_with("逐帧摘要："))
            .expect("应含逐帧摘要行");
        let digests: Vec<&str> = digest_line["逐帧摘要：".len()..].split(',').collect();
        assert_eq!(digests.len(), 2, "帧数必须与摘要条数一致");
        for d in &digests {
            assert_eq!(d.len(), 32, "MD5 十六进制定长 32：{d}");
            assert!(d.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
        assert_ne!(digests[0], digests[1], "不同帧的摘要必须不同");
        // 首帧 PNG 内容块（MCP image）。
        let content: &Json = ok.get("result").unwrap().get("content").unwrap();
        match content {
            Json::Array(items) => {
                let img = items
                    .iter()
                    .find(|c| c.get("type").and_then(|t| t.as_str()) == Some("image"))
                    .expect("应含首帧 image 块");
                assert_eq!(img.get("mimeType").unwrap().as_str(), Some("image/png"));
            }
            _ => panic!("content 应为数组"),
        }
        // job id 是提交参数的纯函数：同源码同参两次调用逐字节一致。
        let again: Json = call_tool(&server, "render_submit", SUBMIT_SRC);
        assert_eq!(result_text(&ok), result_text(&again), "同提交必须同 job 同摘要");
        // 失败对拍：编译错误 → INVALID_PARAMS。
        let bad: Json = call_tool(&server, "render_submit", BAD_SRC);
        assert_eq!(
            bad.get("error").unwrap().get("code").unwrap().as_f64().unwrap() as i64,
            error_code::INVALID_PARAMS
        );
    }

    #[test]
    fn diagnose_roundtrip_reports_failure_without_rpc_error_and_stats_on_success() {
        let server = McpServer::new();
        // 失败对拍（反向语义）：诊断工具的"成功"= 给出诊断，**没有** error 字段。
        let bad: Json = call_tool(&server, "diagnose", BAD_SRC);
        assert!(bad.get("error").is_none(), "诊断失败不得升级为 RPC 错误");
        let text: String = result_text(&bad);
        assert!(text.contains("编译失败"), "应报告编译失败：{text}");
        assert!(text.contains("第 1 行"), "应带行号：{text}");
        assert!(text.contains("建议："), "诊断必含建议段（C19）：{text}");
        // 成功对拍：帧统计 + 病理旗标。
        let ok: Json = call_tool(&server, "diagnose", SRC);
        let text: String = result_text(&ok);
        assert!(text.contains("场景可编译"), "场景应可编译：{text}");
        assert!(text.contains("均值"), "应含均值统计：{text}");
        assert!(text.contains("病理旗标："), "应含病理旗标：{text}");
    }

    #[test]
    fn pathology_flags_classify_flat_frames() {
        // 优先级链：全黑 ⊃ 全白 ⊃ 动态范围坍缩（每帧最多一个旗标）。
        assert_eq!(pathology_flags(0, 0), vec!["全黑"]);
        assert_eq!(pathology_flags(255, 255), vec!["全白"]);
        assert_eq!(pathology_flags(100, 108), vec!["动态范围坍缩"]);
        assert!(pathology_flags(0, 255).is_empty(), "正常动态范围不得命中");
        // 边界：差值恰为阈值（16）不算坍缩——闸门必须确定性。
        assert!(pathology_flags(100, 116).is_empty(), "差 = 阈值 ⇒ 不坍缩");
        assert_eq!(pathology_flags(100, 115), vec!["动态范围坍缩"], "差 < 阈值 ⇒ 坍缩");
    }

    #[test]
    fn split_dsl_message_extracts_embedded_suggestion_or_falls_back() {
        // DSL 消息内嵌建议段时：拆出 core + suggestion。
        let (core, suggestion) =
            split_dsl_message("[P2103] 未知的 easing 名 'x'\n  建议：linear / quadIn", 7);
        assert_eq!(core, "[P2103] 未知的 easing 名 'x'");
        assert!(suggestion.starts_with("linear"), "应取到内嵌建议：{suggestion}");
        // 无内嵌建议时：兜底建议必须带行号（C19：诊断必含建议段）。
        let (core2, suggestion2) = split_dsl_message("裸消息", 3);
        assert_eq!(core2, "裸消息");
        assert!(suggestion2.contains("第 3 行"), "兜底建议须带行号：{suggestion2}");
    }
}
