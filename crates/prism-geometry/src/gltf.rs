//! glTF 精简导入（白皮书 §12；架构 v2.4 S12 mesh 导入特解）。
//!
//! # 范围（白皮书 §12：五块子集）
//!
//! 只解析 scenes/nodes（平移缩放）、meshes/primitives（POSITION/NORMAL/TEXCOORD_0/indices）、
//! materials 的 pbrMetallicRoughness（含 KHR_materials_ior / sheen / clearcoat 扩展）。
//! 材质与 B2 Disney BSDF 参数同构，**零转换**。访问器 componentType 支持 float/ushort/uint。
//!
//! # 诚实纪律（宪章 C7/C20）
//!
//! 未支持的扩展/字段/属性逐条写入加载报告（返回值的 warnings），**永不静默丢弃**；
//! 坏资产返回结构化 [`GltfError`]（三段式：在哪/什么事/怎么改），不 panic。
//!
//! # JSON 解析器
//!
//! 按最小需求手写的 JSON 子集递归下降解析器（零第三方，禁 serde）。

use prism_math::{Mat4, Vec3};

// ─────────────────────────────────────────────────────────────────────────────
// 手写 JSON 子集解析器
// ─────────────────────────────────────────────────────────────────────────────

/// JSON 解析错误（字节偏移 + 原因）。
#[derive(Debug, Clone, PartialEq)]
pub struct JsonErr {
    /// 出错字节偏移。
    pub offset: usize,
    /// 原因描述。
    pub msg: String,
}

/// JSON 值（数字统一 f64 存储，使用时窄化）。
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// null。
    Null,
    /// true / false。
    Bool(bool),
    /// 数值。
    Num(f64),
    /// 字符串。
    Str(String),
    /// 数组。
    Arr(Vec<Json>),
    /// 对象（保持出现顺序，重复键取首现）。
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// 解析 JSON 文本。
    pub fn parse(s: &str) -> Result<Json, JsonErr> {
        let mut p = Parser { s: s.as_bytes(), pos: 0 };
        p.skip_ws();
        let v = p.parse_value(0)?;
        p.skip_ws();
        if p.pos != p.s.len() {
            return Err(p.err("JSON 末尾有多余内容"));
        }
        Ok(v)
    }

    /// 对象取字段。
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// 数组元素。
    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// f64 值。
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// f32 值。
    pub fn f32(&self) -> Option<f32> {
        self.num().map(|n| n as f32)
    }

    /// usize 值。
    pub fn usize(&self) -> Option<usize> {
        self.num().map(|n| if n < 0.0 { 0 } else { n as usize })
    }

    /// 字符串值。
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> JsonErr {
        JsonErr { offset: self.pos, msg: msg.to_string() }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), JsonErr> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("期望 '{}'，得 {:?}", c as char, self.peek().map(|b| b as char))))
        }
    }

    fn parse_value(&mut self, depth: u32) -> Result<Json, JsonErr> {
        if depth > 64 {
            return Err(self.err("JSON 嵌套过深（>64 层）"));
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_obj(depth),
            Some(b'[') => self.parse_arr(depth),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", Json::Bool(true)),
            Some(b'f') => self.parse_lit("false", Json::Bool(false)),
            Some(b'n') => self.parse_lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => Ok(Json::Num(self.parse_number()?)),
            _ => Err(self.err("期望 JSON 值")),
        }
    }

    fn parse_lit(&mut self, lit: &str, v: Json) -> Result<Json, JsonErr> {
        if self.s[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(v)
        } else {
            Err(self.err(&format!("期望 '{lit}'")))
        }
    }

    fn parse_obj(&mut self, depth: u32) -> Result<Json, JsonErr> {
        self.expect(b'{')?;
        let mut kv: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(kv));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.parse_value(depth + 1)?;
            kv.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(kv));
                }
                _ => return Err(self.err("对象中期望 ',' 或 '}'")),
            }
        }
    }

    fn parse_arr(&mut self, depth: u32) -> Result<Json, JsonErr> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let v = self.parse_value(depth + 1)?;
            items.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("数组中期望 ',' 或 ']'")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonErr> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("字符串未闭合"))?;
            self.pos += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or_else(|| self.err("转义序列未闭合"))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.parse_hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                // 高代理：必须跟低代理，合成码点。
                                if self.peek() == Some(b'\\') {
                                    self.pos += 1;
                                } else {
                                    return Err(self.err("代理对缺少低半"));
                                }
                                if self.peek() == Some(b'u') {
                                    self.pos += 1;
                                } else {
                                    return Err(self.err("代理对缺少 \\u"));
                                }
                                let lo = self.parse_hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("低代理范围非法"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            let ch = char::from_u32(code).ok_or_else(|| self.err("非法 Unicode 码点"))?;
                            out.push(ch);
                        }
                        _ => return Err(self.err("非法转义字符")),
                    }
                }
                0x00..=0x1F => return Err(self.err("字符串中出现裸控制字符")),
                _ => {
                    // 非 ASCII：按 UTF-8 序列整段拷贝（输入已是合法 UTF-8 字符串切片）。
                    let start = self.pos - 1;
                    let len = utf8_len(c);
                    self.pos = start + len;
                    if self.pos > self.s.len() {
                        return Err(self.err("UTF-8 序列越界"));
                    }
                    out.push_str(std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| self.err("非法 UTF-8"))?);
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, JsonErr> {
        if self.pos + 4 > self.s.len() {
            return Err(self.err("\\u 需要 4 个十六进制位"));
        }
        let mut v = 0_u32;
        for _ in 0..4 {
            let c = self.s[self.pos];
            self.pos += 1;
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a' + 10) as u32,
                b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => return Err(self.err("十六进制位非法")),
            };
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn parse_number(&mut self) -> Result<f64, JsonErr> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| self.err("数字非法"))?;
        text.parse::<f64>().map_err(|_| self.err(&format!("数字 '{text}' 解析失败")))
    }
}

/// UTF-8 序列长度（首字节）。
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PBR 材质（glTF metallic-roughness ↔ B2 Disney 同构，零转换）
// ─────────────────────────────────────────────────────────────────────────────

/// alpha 模式（glTF alphaMode）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    /// 不透明（默认）。
    Opaque,
    /// 阈值镂空（携带 alphaCutoff）。
    Mask(f32),
    /// 半透明混合。
    Blend,
}

/// PBR 材质参数（字段名与 glTF/B2 Disney 参数一一对应）。
#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    /// baseColorFactor（线性 RGBA）。
    pub base_color: [f32; 4],
    /// metallicFactor。
    pub metalness: f32,
    /// roughnessFactor。
    pub roughness: f32,
    /// KHR_materials_ior 的 ior（默认 1.5）。
    pub ior: f32,
    /// KHR_materials_sheen 的 sheenColorFactor（RGB）。
    pub sheen_color: [f32; 3],
    /// KHR_materials_sheen 的 sheenRoughnessFactor。
    pub sheen_roughness: f32,
    /// KHR_materials_clearcoat 的 clearcoatFactor。
    pub clearcoat: f32,
    /// KHR_materials_clearcoat 的 clearcoatRoughnessFactor。
    pub clearcoat_roughness: f32,
    /// alphaMode。
    pub alpha_mode: AlphaMode,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        PbrMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metalness: 1.0, // glTF 规范默认 metallicFactor = 1
            roughness: 1.0,
            ior: 1.5,
            sheen_color: [0.0, 0.0, 0.0],
            sheen_roughness: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 结构化错误（宪章 C18/C19：三段式——在哪/什么事/怎么改）
// ─────────────────────────────────────────────────────────────────────────────

/// glTF 导入错误。
#[derive(Debug, Clone, PartialEq)]
pub enum GltfError {
    /// JSON 语法错误。
    Json {
        /// 字节偏移。
        offset: usize,
        /// 原因。
        msg: String,
    },
    /// 缺必需字段。
    Missing {
        /// JSON 路径。
        path: String,
        /// 缺什么。
        what: String,
    },
    /// 值非法。
    Bad {
        /// JSON 路径。
        path: String,
        /// 什么事。
        msg: String,
        /// 怎么改。
        hint: String,
    },
    /// 结构合法但超出本导入器范围。
    Unsupported {
        /// JSON 路径。
        path: String,
        /// 什么事。
        msg: String,
        /// 怎么改。
        hint: String,
    },
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfError::Json { offset, msg } => write!(
                f,
                "prism-geometry/gltf：JSON 语法错误\n  位置：字节 {offset}\n  什么事：{msg}\n  建议：用 JSON 校验器定位修复该处语法"
            ),
            GltfError::Missing { path, what } => write!(
                f,
                "prism-geometry/gltf：缺少必需字段\n  位置：{path}\n  什么事：{what}\n  建议：按 glTF 2.0 规范补齐该字段"
            ),
            GltfError::Bad { path, msg, hint } => write!(
                f,
                "prism-geometry/gltf：字段值非法\n  位置：{path}\n  什么事：{msg}\n  建议：{hint}"
            ),
            GltfError::Unsupported { path, msg, hint } => write!(
                f,
                "prism-geometry/gltf：暂不支持的结构\n  位置：{path}\n  什么事：{msg}\n  建议：{hint}"
            ),
        }
    }
}

impl std::error::Error for GltfError {}

// ─────────────────────────────────────────────────────────────────────────────
// 导入产物
// ─────────────────────────────────────────────────────────────────────────────

/// 导入的网格实例（顶点已应用 node 平移缩放链）。
#[derive(Debug, Clone, PartialEq)]
pub struct GltfMesh {
    /// POSITION 顶点。
    pub positions: Vec<Vec3>,
    /// NORMAL 法线（与顶点一一对应；资产未提供时为空）。
    pub normals: Vec<Vec3>,
    /// TEXCOORD_0（与顶点一一对应；资产未提供时为空）。
    pub uvs: Vec<[f32; 2]>,
    /// 三角形索引。
    pub indices: Vec<u32>,
    /// 材质（primitive 未引用材质时为 None）。
    pub material: Option<PbrMaterial>,
}

// ─────────────────────────────────────────────────────────────────────────────
// 主解析
// ─────────────────────────────────────────────────────────────────────────────

/// glTF 小端访问器 componentType。
const CT_FLOAT: f64 = 5126.0;
const CT_USHORT: f64 = 5123.0;
const CT_UINT: f64 = 5125.0;

/// 解析 glTF（JSON 文本 + 二进制 buffer）。
///
/// 返回（网格实例列表，加载报告 warnings——未支持字段逐条列出，永不静默）。
pub fn parse_gltf(json: &str, bin: &[u8]) -> Result<(Vec<GltfMesh>, Vec<String>), GltfError> {
    let root = Json::parse(json).map_err(|e| GltfError::Json { offset: e.offset, msg: e.msg })?;
    let mut warn: Vec<String> = Vec::new();

    // asset 版本提示（最低限度检查，规范要求存在）。
    if root.get("asset").and_then(|a| a.get("version")).is_none() {
        return Err(GltfError::Missing {
            path: "asset.version".into(),
            what: "glTF 2.0 要求 asset.version 字段".into(),
        });
    }

    let buffers = read_buffers(&root, bin)?;
    let views = read_views(&root, &buffers)?;
    let materials = read_materials(&root, &mut warn)?;
    let instances = collect_instances(&root, &mut warn)?;
    let meshes_json = root
        .get("meshes")
        .and_then(|m| m.arr())
        .ok_or_else(|| GltfError::Missing { path: "meshes".into(), what: "meshes 数组".into() })?;

    let mut out = Vec::new();
    for (mesh_idx, world) in &instances {
        let mesh = meshes_json
            .get(*mesh_idx)
            .ok_or_else(|| GltfError::Bad {
                path: format!("nodes[?].mesh = {mesh_idx}"),
                msg: "mesh 索引越界".into(),
                hint: "检查 meshes 数组长度与 node.mesh 引用".into(),
            })?;
        let prims = mesh
            .get("primitives")
            .and_then(|p| p.arr())
            .ok_or_else(|| GltfError::Missing {
                path: format!("meshes[{mesh_idx}].primitives"),
                what: "primitives 数组".into(),
            })?;
        for (pi, prim) in prims.iter().enumerate() {
            let path = format!("meshes[{mesh_idx}].primitives[{pi}]");
            // mode：只支持 TRIANGLES(4)（缺省即 4）。
            let mode = prim.get("mode").and_then(|m| m.num()).unwrap_or(4.0);
            if mode != 4.0 {
                return Err(GltfError::Unsupported {
                    path: format!("{path}.mode"),
                    msg: format!("primitive mode = {mode} 非 TRIANGLES(4)"),
                    hint: "用工具把资产转换为三角网格后重试".into(),
                });
            }
            let attrs = prim
                .get("attributes")
                .ok_or_else(|| GltfError::Missing { path: format!("{path}.attributes"), what: "attributes".into() })?;
            if let Json::Obj(kv) = attrs {
                for (k, _) in kv {
                    if !matches!(k.as_str(), "POSITION" | "NORMAL" | "TEXCOORD_0") {
                        warn.push(format!("{path}.attributes.{k}：未支持的顶点属性，已忽略"));
                    }
                }
            }
            let pos_acc = attr_index(attrs, "POSITION", &format!("{path}.attributes"))?
                .ok_or_else(|| GltfError::Missing {
                    path: format!("{path}.attributes.POSITION"),
                    what: "POSITION 访问器（必需）".into(),
                })?;
            let positions_raw = read_accessor(&root, pos_acc, &views, &format!("{path}.attributes.POSITION"))?;
            if positions_raw.ncomp != 3 {
                return Err(GltfError::Bad {
                    path: format!("{path}.attributes.POSITION"),
                    msg: format!("POSITION 分量数 {} ≠ 3", positions_raw.ncomp),
                    hint: "POSITION 必须是 VEC3".into(),
                });
            }
            let mut positions: Vec<Vec3> = positions_raw
                .floats
                .chunks_exact(3)
                .map(|c| Vec3::new(c[0], c[1], c[2]))
                .collect();
            for p in positions.iter_mut() {
                *p = world.transform_point(*p);
            }
            // NORMAL（可选）。
            let normals = match attr_index(attrs, "NORMAL", &format!("{path}.attributes"))? {
                Some(ai) => {
                    let d = read_accessor(&root, ai, &views, &format!("{path}.attributes.NORMAL"))?;
                    if d.ncomp != 3 || d.floats.len() != positions_raw.floats.len() {
                        return Err(GltfError::Bad {
                            path: format!("{path}.attributes.NORMAL"),
                            msg: "NORMAL 必须为 VEC3 且与 POSITION 等长".into(),
                            hint: "重新导出法线或移除 NORMAL 属性".into(),
                        });
                    }
                    d.floats.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
                }
                None => Vec::new(),
            };
            // TEXCOORD_0（可选）。
            let uvs = match attr_index(attrs, "TEXCOORD_0", &format!("{path}.attributes"))? {
                Some(ai) => {
                    let d = read_accessor(&root, ai, &views, &format!("{path}.attributes.TEXCOORD_0"))?;
                    if d.ncomp != 2 || d.floats.len() / 2 != positions.len() {
                        return Err(GltfError::Bad {
                            path: format!("{path}.attributes.TEXCOORD_0"),
                            msg: "TEXCOORD_0 必须为 VEC2 且与 POSITION 等长".into(),
                            hint: "重新导出 UV 或移除 TEXCOORD_0 属性".into(),
                        });
                    }
                    d.floats.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
                }
                None => Vec::new(),
            };
            // indices（可缺省：非索引网格按顺序三角化）。
            let n_verts = positions_raw.count;
            let indices: Vec<u32> = match prim.get("indices").and_then(|i| i.usize()) {
                Some(ii) => {
                    let d = read_accessor(&root, ii, &views, &format!("{path}.indices"))?;
                    if d.ncomp != 1 {
                        return Err(GltfError::Bad {
                            path: format!("{path}.indices"),
                            msg: "indices 必须为 SCALAR".into(),
                            hint: "indices 是标量访问器".into(),
                        });
                    }
                    let idx: Vec<u32> = d.uints.clone();
                    for (k, &v) in idx.iter().enumerate() {
                        if v as usize >= n_verts {
                            return Err(GltfError::Bad {
                                path: format!("{path}.indices[{k}]"),
                                msg: format!("索引 {v} 越界（顶点数 {n_verts}）"),
                                hint: "修复索引进 [0, 顶点数)".into(),
                            });
                        }
                    }
                    if idx.len() % 3 != 0 {
                        return Err(GltfError::Bad {
                            path: format!("{path}.indices"),
                            msg: format!("索引数 {} 不是 3 的倍数", idx.len()),
                            hint: "索引必须按三角形分组".into(),
                        });
                    }
                    idx
                }
                None => {
                    if n_verts % 3 != 0 {
                        return Err(GltfError::Bad {
                            path: format!("{path}.attributes.POSITION"),
                            msg: format!("非索引网格顶点数 {n_verts} 不是 3 的倍数"),
                            hint: "提供 indices 或保证顶点按三角形排布".into(),
                        });
                    }
                    (0..n_verts as u32).collect()
                }
            };
            let material = prim
                .get("material")
                .and_then(|m| m.usize())
                .map(|mi| {
                    materials.get(mi).cloned().flatten().ok_or_else(|| GltfError::Bad {
                        path: format!("{path}.material = {mi}"),
                        msg: "材质索引越界".into(),
                        hint: "检查 materials 数组长度".into(),
                    })
                })
                .transpose()?;
            out.push(GltfMesh { positions, normals, uvs, indices, material });
        }
    }
    // 未被场景引用的 mesh 逐条入报告（永不静默）。
    let referenced: Vec<usize> = instances.iter().map(|(m, _)| *m).collect();
    for mi in 0..meshes_json.len() {
        if !referenced.contains(&mi) {
            warn.push(format!("meshes[{mi}]：未被任何 scene/node 引用，未导入"));
        }
    }
    Ok((out, warn))
}

/// GLB 容器拆包：返回（JSON 文本，BIN chunk）。
pub fn read_glb(bytes: &[u8]) -> Result<(String, Vec<u8>), GltfError> {
    if bytes.len() < 12 {
        return Err(GltfError::Bad {
            path: "(GLB header)".into(),
            msg: format!("文件长度 {} 字节不足 GLB 头（12 字节）", bytes.len()),
            hint: "确认这是完整的 .glb 文件".into(),
        });
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x4654_6C67 {
        return Err(GltfError::Bad {
            path: "(GLB header).magic".into(),
            msg: format!("magic 0x{magic:08X} ≠ 'glTF'"),
            hint: "GLB 文件应以 0x46546C67 开头".into(),
        });
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 2 {
        return Err(GltfError::Unsupported {
            path: "(GLB header).version".into(),
            msg: format!("GLB 版本 {version} ≠ 2"),
            hint: "本导入器只支持 glTF 2.0".into(),
        });
    }
    let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if total > bytes.len() {
        return Err(GltfError::Bad {
            path: "(GLB header).length".into(),
            msg: format!("声明长度 {total} 超过实际 {}", bytes.len()),
            hint: "文件被截断，重新导出".into(),
        });
    }
    let mut pos = 12usize;
    let mut json_text: Option<String> = None;
    let mut bin: Vec<u8> = Vec::new();
    while pos + 8 <= total {
        let clen = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let ctype = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]);
        pos += 8;
        if pos + clen > total {
            return Err(GltfError::Bad {
                path: "(GLB chunk)".into(),
                msg: format!("chunk 长度 {clen} 越界"),
                hint: "chunk 表损坏".into(),
            });
        }
        match ctype {
            0x4E4F_534A => {
                // JSON（尾 padding 空格按规范允许）。
                let raw = &bytes[pos..pos + clen];
                let text = std::str::from_utf8(raw)
                    .map_err(|_| GltfError::Bad { path: "(GLB JSON chunk)".into(), msg: "非 UTF-8".into(), hint: "JSON chunk 必须为 UTF-8".into() })?;
                json_text = Some(text.trim_end().to_string());
            }
            0x004E_4942 => bin.extend_from_slice(&bytes[pos..pos + clen]),
            _ => {}
        }
        pos += clen;
    }
    match json_text {
        Some(t) => Ok((t, bin)),
        None => Err(GltfError::Missing { path: "(GLB JSON chunk)".into(), what: "JSON chunk".into() }),
    }
}

/// base64 解码（data URI buffer 用；表驱动，'=' padding）。
fn decode_base64(s: &str) -> Result<Vec<u8>, GltfError> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut chunk = [0_u32; 4];
    let mut n = 0usize;
    let mut pad = 0usize;
    for &c in bytes {
        if c == b'=' {
            pad += 1;
            continue;
        }
        if c == b'\n' || c == b'\r' || c == b' ' {
            continue;
        }
        let v = val(c).ok_or_else(|| GltfError::Bad {
            path: "(buffer.uri base64)".into(),
            msg: format!("非法 base64 字符 '{}'", c as char),
            hint: "data URI 必须是标准 base64".into(),
        })?;
        chunk[n] = v;
        n += 1;
        if n == 4 {
            let w = (chunk[0] << 18) | (chunk[1] << 12) | (chunk[2] << 6) | chunk[3];
            out.push((w >> 16) as u8);
            out.push((w >> 8) as u8);
            out.push(w as u8);
            n = 0;
        }
    }
    match (n, pad) {
        (0, 0) => {}
        (2, _) => {
            let w = (chunk[0] << 18) | (chunk[1] << 12);
            out.push((w >> 16) as u8);
        }
        (3, _) => {
            let w = (chunk[0] << 18) | (chunk[1] << 12) | (chunk[2] << 6);
            out.push((w >> 16) as u8);
            out.push((w >> 8) as u8);
        }
        _ => {
            return Err(GltfError::Bad {
                path: "(buffer.uri base64)".into(),
                msg: "base64 长度/padding 非法".into(),
                hint: "检查 data URI 完整性".into(),
            })
        }
    }
    Ok(out)
}

/// buffers 数组 → 每个.buffer 的字节数组（data URI 解码 / GLB bin 注入）。
fn read_buffers(root: &Json, bin: &[u8]) -> Result<Vec<Vec<u8>>, GltfError> {
    let empty = Vec::new();
    let arr = root.get("buffers").and_then(|b| b.arr()).unwrap_or(&empty);
    let mut out = Vec::new();
    for (i, b) in arr.iter().enumerate() {
        let path = format!("buffers[{i}]");
        let len = b
            .get("byteLength")
            .and_then(|v| v.usize())
            .ok_or_else(|| GltfError::Missing { path: format!("{path}.byteLength"), what: "byteLength".into() })?;
        let data = match b.get("uri").and_then(|u| u.str()) {
            None => {
                // 无 uri：数据来自 GLB BIN chunk。
                if bin.is_empty() {
                    return Err(GltfError::Missing {
                        path: format!("{path}.uri"),
                        what: "buffer 无 uri 且无 GLB BIN chunk".into(),
                    });
                }
                bin.to_vec()
            }
            Some(uri) => {
                let prefix = "data:application/octet-stream;base64,";
                if let Some(rest) = uri.strip_prefix(prefix) {
                    decode_base64(rest)?
                } else {
                    return Err(GltfError::Unsupported {
                        path: format!("{path}.uri"),
                        msg: "外部文件 buffer 引用".into(),
                        hint: format!("把 '{uri}' 内容以 base64 data URI 内嵌，或改用 GLB 容器"),
                    });
                }
            }
        };
        if data.len() < len {
            return Err(GltfError::Bad {
                path: format!("{path}"),
                msg: format!("buffer 实际 {} 字节 < 声明 {len}", data.len()),
                hint: "检查 byteLength 与数据一致性".into(),
            });
        }
        out.push(data);
    }
    Ok(out)
}

/// bufferViews 数组 → 每个视图的字节切片。
fn read_views(root: &Json, buffers: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, GltfError> {
    let empty = Vec::new();
    let arr = root.get("bufferViews").and_then(|b| b.arr()).unwrap_or(&empty);
    let mut out = Vec::new();
    for (i, v) in arr.iter().enumerate() {
        let path = format!("bufferViews[{i}]");
        let bi = v
            .get("buffer")
            .and_then(|x| x.usize())
            .ok_or_else(|| GltfError::Missing { path: format!("{path}.buffer"), what: "buffer 索引".into() })?;
        let buf = buffers.get(bi).ok_or_else(|| GltfError::Bad {
            path: format!("{path}.buffer = {bi}"),
            msg: "buffer 索引越界".into(),
            hint: "检查 buffers 数组长度".into(),
        })?;
        let off = v.get("byteOffset").and_then(|x| x.usize()).unwrap_or(0);
        let len = v
            .get("byteLength")
            .and_then(|x| x.usize())
            .ok_or_else(|| GltfError::Missing { path: format!("{path}.byteLength"), what: "byteLength".into() })?;
        if off + len > buf.len() {
            return Err(GltfError::Bad {
                path,
                msg: format!("视图区间 [{off}, {}) 超出 buffer {} 字节", off + len, buf.len()),
                hint: "检查 byteOffset/byteLength".into(),
            });
        }
        out.push(buf[off..off + len].to_vec());
    }
    Ok(out)
}

/// 访问器读取结果。
struct AccData {
    /// 解码后的浮点流（整型按值转 f32）。
    floats: Vec<f32>,
    /// 无符号整型流（indices 用）。
    uints: Vec<u32>,
    /// 元素个数（count）。
    count: usize,
    /// 每元素分量数。
    ncomp: usize,
}

/// 读取一个 accessor（float/ushort/uint 小端；byteStride 非紧凑即拒绝）。
fn read_accessor(
    root: &Json,
    idx: usize,
    views: &[Vec<u8>],
    path: &str,
) -> Result<AccData, GltfError> {
    let accessors = root
        .get("accessors")
        .and_then(|a| a.arr())
        .ok_or_else(|| GltfError::Missing { path: "accessors".into(), what: "accessors 数组".into() })?;
    let acc = accessors.get(idx).ok_or_else(|| GltfError::Bad {
        path: format!("{path} = accessor {idx}"),
        msg: "访问器索引越界".into(),
        hint: "检查 accessors 数组长度".into(),
    })?;
    let ct = acc
        .get("componentType")
        .and_then(|c| c.num())
        .ok_or_else(|| GltfError::Missing { path: format!("{path}.componentType"), what: "componentType".into() })?;
    let ncomp = match acc.get("type").and_then(|t| t.str()) {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        other => {
            return Err(GltfError::Unsupported {
                path: format!("{path}.type"),
                msg: format!("访问器类型 {other:?} 不支持"),
                hint: "支持 SCALAR/VEC2/VEC3/VEC4".into(),
            })
        }
    };
    let count = acc
        .get("count")
        .and_then(|c| c.usize())
        .ok_or_else(|| GltfError::Missing { path: format!("{path}.count"), what: "count".into() })?;
    let (comp_size, is_float, is_uint) = match ct {
        CT_FLOAT => (4usize, true, false),
        CT_USHORT => (2, false, true), // ushort 主要用于索引流 → 填充 uints
        CT_UINT => (4, false, true),
        other => {
            return Err(GltfError::Unsupported {
                path: format!("{path}.componentType"),
                msg: format!("componentType {other} 不支持（支持 5126 float / 5123 ushort / 5125 uint）"),
                hint: "导出时选择 float/ushort/uint".into(),
            })
        }
    };
    let vi = acc
        .get("bufferView")
        .and_then(|v| v.usize())
        .ok_or_else(|| GltfError::Missing { path: format!("{path}.bufferView"), what: "bufferView".into() })?;
    let view = views.get(vi).ok_or_else(|| GltfError::Bad {
        path: format!("{path}.bufferView = {vi}"),
        msg: "bufferView 索引越界".into(),
        hint: "检查 bufferViews 数组长度".into(),
    })?;
    let off = acc.get("byteOffset").and_then(|v| v.usize()).unwrap_or(0);
    let need = count * ncomp * comp_size;
    if off + need > view.len() {
        return Err(GltfError::Bad {
            path: path.to_string(),
            msg: format!("访问器需要 {need} 字节，视图仅 {}（偏移 {off}）", view.len()),
            hint: "检查 byteOffset/count/type 一致性".into(),
        });
    }
    let mut floats = Vec::with_capacity(count * ncomp);
    let mut uints: Vec<u32> = Vec::new();
    for k in 0..count * ncomp {
        let b = off + k * comp_size;
        let v = match (is_float, comp_size) {
            (true, 4) => f32::from_le_bytes([view[b], view[b + 1], view[b + 2], view[b + 3]]) as f32,
            (false, 2) => u16::from_le_bytes([view[b], view[b + 1]]) as f32,
            _ => u32::from_le_bytes([view[b], view[b + 1], view[b + 2], view[b + 3]]) as f32,
        };
        floats.push(v);
    }
    if is_uint {
        uints = floats.iter().map(|&f| f as u32).collect();
    }
    Ok(AccData { floats, uints, count, ncomp })
}

/// attributes 里取属性对应的 accessor 索引。
fn attr_index(attrs: &Json, name: &str, path: &str) -> Result<Option<usize>, GltfError> {
    match attrs.get(name) {
        Some(v) => v
            .usize()
            .map(Some)
            .ok_or_else(|| GltfError::Bad { path: format!("{path}.{name}"), msg: "属性值应为整数访问器索引".into(), hint: "attributes.{name} 是 accessor 索引".into() }),
        None => Ok(None),
    }
}

/// materials 数组 → PbrMaterial 列表（未支持字段逐条入报告）。
fn read_materials(root: &Json, warn: &mut Vec<String>) -> Result<Vec<Option<PbrMaterial>>, GltfError> {
    let empty = Vec::new();
    let arr = root.get("materials").and_then(|m| m.arr()).unwrap_or(&empty);
    let mut out = Vec::new();
    for (i, m) in arr.iter().enumerate() {
        let path = format!("materials[{i}]");
        // 顶层未支持字段逐条入报告。
        if let Json::Obj(kv) = m {
            for (k, _) in kv {
                if !matches!(
                    k.as_str(),
                    "name" | "pbrMetallicRoughness" | "alphaMode" | "alphaCutoff" | "extensions"
                ) {
                    warn.push(format!("{path}.{k}：未支持，已忽略"));
                }
            }
        }
        let mut mat = PbrMaterial::default();
        if let Some(pbr) = m.get("pbrMetallicRoughness") {
            if let Some(c) = pbr.get("baseColorFactor").and_then(|c| c.arr()) {
                mat.base_color = read_vec4(c, &format!("{path}.pbrMetallicRoughness.baseColorFactor"))?;
            }
            if let Some(v) = pbr.get("metallicFactor").and_then(|v| v.f32()) {
                mat.metalness = v;
            }
            if let Some(v) = pbr.get("roughnessFactor").and_then(|v| v.f32()) {
                mat.roughness = v;
            }
            if pbr.get("baseColorTexture").is_some() {
                warn.push(format!(
                    "{path}.pbrMetallicRoughness.baseColorTexture：纹理暂不烘焙，仅使用 baseColorFactor"
                ));
            }
            if let Json::Obj(kv) = pbr {
                for (k, _) in kv {
                    if !matches!(
                        k.as_str(),
                        "baseColorFactor" | "metallicFactor" | "roughnessFactor" | "baseColorTexture"
                    ) {
                        warn.push(format!("{path}.pbrMetallicRoughness.{k}：未支持，已忽略"));
                    }
                }
            }
        }
        if let Some(mode) = m.get("alphaMode").and_then(|v| v.str()) {
            mat.alpha_mode = match mode {
                "OPAQUE" => AlphaMode::Opaque,
                "MASK" => AlphaMode::Mask(m.get("alphaCutoff").and_then(|v| v.f32()).unwrap_or(0.5)),
                "BLEND" => AlphaMode::Blend,
                other => {
                    return Err(GltfError::Bad {
                        path: format!("{path}.alphaMode"),
                        msg: format!("alphaMode '{other}' 非法"),
                        hint: "取值 OPAQUE/MASK/BLEND".into(),
                    })
                }
            };
        }
        if let Some(ext) = m.get("extensions") {
            if let Json::Obj(kv) = ext {
                for (k, v) in kv {
                    match k.as_str() {
                        "KHR_materials_ior" => {
                            mat.ior = v.get("ior").and_then(|x| x.f32()).unwrap_or(1.5);
                        }
                        "KHR_materials_sheen" => {
                            if let Some(c) = v.get("sheenColorFactor").and_then(|c| c.arr()) {
                                let sc =
                                    read_vec3(c, &format!("{path}.extensions.{k}.sheenColorFactor"))?;
                                mat.sheen_color = [sc.x, sc.y, sc.z];
                            }
                            mat.sheen_roughness =
                                v.get("sheenRoughnessFactor").and_then(|x| x.f32()).unwrap_or(0.0);
                        }
                        "KHR_materials_clearcoat" => {
                            mat.clearcoat = v.get("clearcoatFactor").and_then(|x| x.f32()).unwrap_or(0.0);
                            mat.clearcoat_roughness =
                                v.get("clearcoatRoughnessFactor").and_then(|x| x.f32()).unwrap_or(0.0);
                        }
                        other => warn.push(format!("{path}.extensions.{other}：未支持扩展，已忽略")),
                    }
                }
            }
        }
        out.push(Some(mat));
    }
    Ok(out)
}

/// scene/node 层级 → (mesh 索引, 世界变换) 实例表（平移缩放链；rotation/matrix 入报告）。
fn collect_instances(root: &Json, warn: &mut Vec<String>) -> Result<Vec<(usize, Mat4)>, GltfError> {
    let nodes_arr = root
        .get("nodes")
        .and_then(|n| n.arr())
        .ok_or_else(|| GltfError::Missing { path: "nodes".into(), what: "nodes 数组".into() })?;
    let scenes_arr = root.get("scenes").and_then(|s| s.arr()).unwrap_or(&[]);
    if scenes_arr.is_empty() {
        return Ok(Vec::new());
    }
    let scene_idx = root.get("scene").and_then(|s| s.usize()).unwrap_or(0);
    let scene = scenes_arr.get(scene_idx).ok_or_else(|| GltfError::Bad {
        path: format!("scene = {scene_idx}"),
        msg: "场景索引越界".into(),
        hint: "检查 scenes 数组长度".into(),
    })?;
    let roots = scene
        .get("nodes")
        .and_then(|n| n.arr())
        .ok_or_else(|| GltfError::Missing { path: format!("scenes[{scene_idx}].nodes"), what: "根节点列表".into() })?;

    let mut out: Vec<(usize, Mat4)> = Vec::new();
    let mut visited = vec![false; nodes_arr.len()];
    // 迭代 DFS（防深资产爆栈）：栈元素 = (node 索引, 父变换)。
    let mut stack: Vec<(usize, Mat4)> = Vec::new();
    for r in roots.iter().rev() {
        let ri = r.usize().ok_or_else(|| GltfError::Bad {
            path: format!("scenes[{scene_idx}].nodes"),
            msg: "根节点索引应为整数".into(),
            hint: "nodes 数组元素是索引".into(),
        })?;
        stack.push((ri, Mat4::identity()));
    }
    while let Some((ni, parent)) = stack.pop() {
        if ni >= nodes_arr.len() {
            return Err(GltfError::Bad {
                path: "scenes[?].nodes".into(),
                msg: format!("node 索引 {ni} 越界"),
                hint: "检查 nodes 数组长度".into(),
            });
        }
        if visited[ni] {
            return Err(GltfError::Bad {
                path: format!("nodes[{ni}]"),
                msg: "节点图存在环".into(),
                hint: "修复资产 node 层级（glTF 禁环）".into(),
            });
        }
        visited[ni] = true;
        let node = &nodes_arr[ni];
        let path = format!("nodes[{ni}]");
        if node.get("matrix").is_some() {
            warn.push(format!("{path}.matrix：未支持（只解析平移缩放），已按单位变换处理"));
        }
        if node.get("rotation").is_some() {
            warn.push(format!("{path}.rotation：未支持（只解析平移缩放），已按单位旋转处理"));
        }
        let t = node
            .get("translation")
            .and_then(|v| v.arr())
            .map(|a| read_vec3(a, &format!("{path}.translation")))
            .transpose()?
            .unwrap_or(Vec3::splat(0.0));
        let s = node
            .get("scale")
            .and_then(|v| v.arr())
            .map(|a| read_vec3(a, &format!("{path}.scale")))
            .transpose()?
            .unwrap_or(Vec3::new(1.0, 1.0, 1.0));
        if s.x <= 0.0 || s.y <= 0.0 || s.z <= 0.0 {
            return Err(GltfError::Bad {
                path: format!("{path}.scale"),
                msg: format!("缩放分量非正：({:?})", (s.x, s.y, s.z)),
                hint: "scale 分量必须为正数".into(),
            });
        }
        let local = Mat4::translation(t).mul(&scale_mat(s));
        let world = parent.mul(&local);
        if let Some(mi) = node.get("mesh").and_then(|m| m.usize()) {
            out.push((mi, world));
        }
        if let Some(children) = node.get("children").and_then(|c| c.arr()) {
            for c in children.iter().rev() {
                let ci = c.usize().ok_or_else(|| GltfError::Bad {
                    path: format!("{path}.children"),
                    msg: "子节点索引应为整数".into(),
                    hint: "children 数组元素是索引".into(),
                })?;
                stack.push((ci, world));
            }
        }
    }
    Ok(out)
}

/// 缩放矩阵（对角阵）。
fn scale_mat(s: Vec3) -> Mat4 {
    Mat4([
        s.x, 0.0, 0.0, 0.0, //
        0.0, s.y, 0.0, 0.0, //
        0.0, 0.0, s.z, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// JSON 数组 → Vec3。
fn read_vec3(a: &[Json], path: &str) -> Result<Vec3, GltfError> {
    if a.len() != 3 {
        return Err(GltfError::Bad {
            path: path.into(),
            msg: format!("需要 3 个分量，得 {}", a.len()),
            hint: "该字段是 [x, y, z]".into(),
        });
    }
    let mut v = [0.0_f32; 3];
    for (i, x) in a.iter().enumerate() {
        v[i] = x.f32().ok_or_else(|| GltfError::Bad {
            path: format!("{path}[{i}]"),
            msg: "分量应为数字".into(),
            hint: "该字段是数字数组".into(),
        })?;
    }
    Ok(Vec3::new(v[0], v[1], v[2]))
}

/// JSON 数组 → RGBA。
fn read_vec4(a: &[Json], path: &str) -> Result<[f32; 4], GltfError> {
    if a.len() != 4 {
        return Err(GltfError::Bad {
            path: path.into(),
            msg: format!("需要 4 个分量，得 {}", a.len()),
            hint: "baseColorFactor 是 [r, g, b, a]".into(),
        });
    }
    let mut v = [0.0_f32; 4];
    for (i, x) in a.iter().enumerate() {
        v[i] = x.f32().ok_or_else(|| GltfError::Bad {
            path: format!("{path}[{i}]"),
            msg: "分量应为数字".into(),
            hint: "该字段是数字数组".into(),
        })?;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试内 base64 编码器（data URI 构造用；避免手数字符长度出错）。
    fn b64(data: &[u8]) -> String {
        const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let w = (b0 << 16) | (b1 << 8) | b2;
            out.push(TBL[(w >> 18) as usize & 63] as char);
            out.push(TBL[(w >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 { TBL[(w >> 6) as usize & 63] as char } else { '=' });
            out.push(if chunk.len() > 2 { TBL[w as usize & 63] as char } else { '=' });
        }
        out
    }

    /// data URI 前缀。
    const DATA_URI: &str = "data:application/octet-stream;base64,";

    /// 最小合法资产（1 mesh：3 顶点 1 三角 + 材质 + node 平移缩放），逐项断言。
    #[test]
    fn minimal_gltf_roundtrip() {
        let json = r#"{
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "translation": [1.0, 2.0, 3.0], "scale": [2.0, 2.0, 2.0]}],
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                    "indices": 3,
                    "material": 0
                }]
            }],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.25, 0.125, 1.0],
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.4
                },
                "alphaMode": "MASK",
                "alphaCutoff": 0.25,
                "extensions": {
                    "KHR_materials_ior": {"ior": 1.45},
                    "KHR_materials_sheen": {"sheenColorFactor": [0.1, 0.2, 0.3], "sheenRoughnessFactor": 0.35},
                    "KHR_materials_clearcoat": {"clearcoatFactor": 0.6, "clearcoatRoughnessFactor": 0.15}
                }
            }],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
                {"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0,  "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 36},
                {"buffer": 0, "byteOffset": 72, "byteLength": 24},
                {"buffer": 0, "byteOffset": 96, "byteLength": 12}
            ],
            "buffers": [{"byteLength": 108}]
        }"#;
        let mut bin = vec![0_u8; 108];
        // POSITION：3 个 VEC3 float。
        let pos: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for (i, p) in pos.iter().enumerate() {
            for (k, c) in p.iter().enumerate() {
                let b = (i * 3 + k) * 4;
                bin[b..b + 4].copy_from_slice(&c.to_le_bytes());
            }
        }
        // NORMAL（offset 36）。
        for (i, n) in [[0.0_f32, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]].iter().enumerate() {
            for (k, c) in n.iter().enumerate() {
                let b = 36 + (i * 3 + k) * 4;
                bin[b..b + 4].copy_from_slice(&c.to_le_bytes());
            }
        }
        // TEXCOORD_0（offset 72，VEC2）。
        for (i, uv) in [[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0]].iter().enumerate() {
            for (k, c) in uv.iter().enumerate() {
                let b = 72 + (i * 2 + k) * 4;
                bin[b..b + 4].copy_from_slice(&c.to_le_bytes());
            }
        }
        // indices（offset 96，uint）。
        for (i, v) in [0_u32, 1, 2].iter().enumerate() {
            let b = 96 + i * 4;
            bin[b..b + 4].copy_from_slice(&v.to_le_bytes());
        }
        let (meshes, warn) = parse_gltf(json, &bin).expect("合法资产应解析成功");
        assert!(warn.is_empty(), "最小资产不应有警告：{warn:?}");
        assert_eq!(meshes.len(), 1);
        let m = &meshes[0];
        // node 平移缩放：p*2 + (1,2,3)。
        assert_eq!(m.positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.positions[1], Vec3::new(3.0, 2.0, 3.0));
        assert_eq!(m.positions[2], Vec3::new(1.0, 4.0, 3.0));
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert_eq!(m.normals[0], Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(m.uvs[1], [1.0, 0.0]);
        let mat = m.material.as_ref().expect("应带材质");
        assert_eq!(mat.base_color, [0.5, 0.25, 0.125, 1.0]);
        assert!((mat.metalness - 0.8).abs() < 1e-6);
        assert!((mat.roughness - 0.4).abs() < 1e-6);
        assert!((mat.ior - 1.45).abs() < 1e-6);
        assert_eq!(mat.alpha_mode, AlphaMode::Mask(0.25));
        assert_eq!(mat.sheen_color, [0.1, 0.2, 0.3]);
        assert!((mat.sheen_roughness - 0.35).abs() < 1e-6);
        assert!((mat.clearcoat - 0.6).abs() < 1e-6);
        assert!((mat.clearcoat_roughness - 0.15).abs() < 1e-6);
    }

    /// GLB 容器 + base64 data URI 两种二进制通道。
    #[test]
    fn glb_and_data_uri_channels() {
        // —— base64 data URI 通道 1：12 字节全 0（1 顶点非索引）。
        //    解码通道本身应工作；1 顶点不是 3 的倍数 → 结构化 Err（不 panic）。
        let json1 = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "scene": 0,
                "scenes": [{{"nodes": [0]}}],
                "nodes": [{{"mesh": 0}}],
                "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
                "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}}],
                "bufferViews": [{{"buffer": 0, "byteLength": 12}}],
                "buffers": [{{"byteLength": 12, "uri": "{DATA_URI}{}"}}]
            }}"#,
            b64(&[0_u8; 12])
        );
        let e = parse_gltf(&json1, &[]).unwrap_err();
        assert!(e.to_string().contains("3 的倍数"), "base64 解码成功后触发三角化校验：{e}");

        // —— base64 data URI 通道 2：36 字节全 0 的合法非索引三角形。
        let json3 = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "scene": 0,
                "scenes": [{{"nodes": [0]}}],
                "nodes": [{{"mesh": 0}}],
                "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
                "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}}],
                "bufferViews": [{{"buffer": 0, "byteLength": 36}}],
                "buffers": [{{"byteLength": 36, "uri": "{DATA_URI}{}"}}]
            }}"#,
            b64(&[0_u8; 36])
        );
        let (meshes, _) = parse_gltf(&json3, &[]).expect("data URI 三角形应解析");
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].positions.len(), 3);
        assert_eq!(meshes[0].indices, vec![0, 1, 2], "非索引网格按顺序三角化");

        // —— GLB：JSON chunk + BIN chunk 手工构造。
        let inner_json = br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"buffers":[{"byteLength":42}]}"#;
        let mut bin = vec![0_u8; 42];
        let pos = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for (i, p) in pos.iter().enumerate() {
            for (k, c) in p.iter().enumerate() {
                let b = (i * 3 + k) * 4;
                bin[b..b + 4].copy_from_slice(&c.to_le_bytes());
            }
        }
        for (i, v) in [0_u16, 1, 2].iter().enumerate() {
            let b = 36 + i * 2;
            bin[b..b + 2].copy_from_slice(&v.to_le_bytes());
        }
        // 组装 GLB。
        let mut json_chunk = inner_json.to_vec();
        while json_chunk.len() % 4 != 0 {
            json_chunk.push(b' ');
        }
        let mut bin_chunk = bin.clone();
        while bin_chunk.len() % 4 != 0 {
            bin_chunk.push(0);
        }
        let mut glb = Vec::new();
        glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        glb.extend_from_slice(&2_u32.to_le_bytes());
        let total = 12 + 8 + json_chunk.len() + 8 + bin_chunk.len();
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        glb.extend_from_slice(&json_chunk);
        glb.extend_from_slice(&(bin_chunk.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        glb.extend_from_slice(&bin_chunk);
        let (text, gbin) = read_glb(&glb).expect("GLB 应拆包成功");
        let (meshes, _) = parse_gltf(&text, &gbin).expect("GLB 资产应解析");
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(meshes[0].positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    /// 坏资产：全部返回结构化 Err（不 panic），错误文本含建议段（宪章 C19）。
    #[test]
    fn bad_assets_structured_errors() {
        // JSON 语法错误。
        let e = parse_gltf("{asset:", &[]).unwrap_err();
        assert!(matches!(e, GltfError::Json { .. }), "JSON 语法 → Json 错误");
        assert!(e.to_string().contains("建议"), "三段式含建议段");
        // 缺 asset.version。
        let e = parse_gltf(r#"{"meshes":[]}"#, &[]).unwrap_err();
        assert!(matches!(e, GltfError::Missing { .. }));
        // 缺 POSITION（资产结构完整到能走到属性检查）。
        let e = parse_gltf(
            r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
                "meshes":[{"primitives":[{"attributes":{}}]}],"accessors":[],"bufferViews":[],
                "buffers":[{"byteLength":0,"uri":"data:application/octet-stream;base64,"}]}"#,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains("POSITION"), "缺 POSITION 应被点名：{e}");
        // componentType 不支持（5121 byte）。
        let e = parse_gltf(
            &format!(
                r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
                    "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
                    "accessors":[{{"bufferView":0,"componentType":5121,"count":3,"type":"VEC3"}}],
                    "bufferViews":[{{"buffer":0,"byteLength":9}}],
                    "buffers":[{{"byteLength":9,"uri":"{DATA_URI}{}"}}]}}"#,
                b64(&[0_u8; 9])
            ),
            &[],
        )
        .unwrap_err();
        assert!(matches!(e, GltfError::Unsupported { .. }), "byte 访问器 → Unsupported：{e}");
        // 索引越界（idx[0] = 999 ≥ 顶点数 3）。
        let mut bin = vec![0_u8; 48];
        bin[36..40].copy_from_slice(&999_u32.to_le_bytes());
        let e = parse_gltf(
            &format!(
                r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
                    "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],
                    "accessors":[
                        {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
                        {{"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}}],
                    "bufferViews":[{{"buffer":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":12}}],
                    "buffers":[{{"byteLength":48,"uri":"{DATA_URI}{}"}}]}}"#,
                b64(&bin)
            ),
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains("越界"), "索引越界应报错：{e}");
        // 外部 buffer 文件引用。
        let e = parse_gltf(
            r#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36,"uri":"model.bin"}]}"#,
            &[],
        )
        .unwrap_err();
        assert!(matches!(e, GltfError::Unsupported { .. }), "外部 .bin → Unsupported：{e}");
        // GLB magic 错误（12 字节起才检查 magic）。
        let e = read_glb(b"NOTGLB123456").unwrap_err();
        assert!(e.to_string().contains("magic"));
    }

    /// 未支持字段/属性/扩展进加载报告（永不静默）。
    #[test]
    fn unsupported_fields_reported() {
        let json = r#"{
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "rotation": [0.0, 0.0, 0.0, 1.0]}],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0, "TANGENT": 1}}]},
                {"primitives": [{"attributes": {"POSITION": 0}}]}
            ],
            "materials": [{"emissiveFactor": [0.0, 0.0, 0.0], "extensions": {"KHR_materials_specular": {}}}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
            ],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "buffers": [{"byteLength": 36}]
        }"#;
        let (meshes, warn) = parse_gltf(json, &vec![0_u8; 36]).expect("应解析成功");
        assert_eq!(meshes.len(), 1, "仅 node 引用的 mesh[0] 导入");
        let text = warn.join("\n");
        assert!(text.contains("TANGENT"), "未支持属性入报告：{text}");
        assert!(text.contains("rotation"), "未支持 node 字段入报告");
        assert!(text.contains("emissiveFactor"), "未支持材质字段入报告");
        assert!(text.contains("KHR_materials_specular"), "未支持扩展入报告");
        assert!(text.contains("meshes[1]"), "未引用 mesh 入报告：{text}");
    }

    /// JSON 子集：转义、Unicode 代理对、数字、深层嵌套限制。
    #[test]
    fn json_subset_semantics() {
        let v = Json::parse(r#"{"a": "é中\ud83d\ude00\n\t\"x\"", "b": [1, -2.5e2, 0.1], "c": null}"#)
            .expect("应解析");
        assert_eq!(
            v.get("a").and_then(|s| s.str()),
            Some("é中\u{1F600}\n\t\"x\""),
            "转义与代理对（😀）"
        );
        let arr = v.get("b").and_then(|b| b.arr()).unwrap();
        assert_eq!(arr[1].num(), Some(-250.0));
        assert_eq!(arr[2].f32(), Some(0.1));
        assert_eq!(v.get("c"), Some(&Json::Null));
        // 深嵌套 → 结构化错误。
        let deep = "[".repeat(70) + &"]".repeat(70);
        assert!(Json::parse(&deep).is_err(), "超过 64 层应拒绝");
        // 非法尾随。
        assert!(Json::parse("1 2").is_err());
    }
}
