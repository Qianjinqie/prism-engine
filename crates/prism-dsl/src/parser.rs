//! 解析器：token 流 → AST（递归下降 + 就地语义防呆）。
//!
//! v1（M0/M1）解析规则与错误消息逐字保留；v2 新增全块语法：
//! movie / theme / import packs / scene(v2) / transition / layer / sim /
//! stage3d / compose / anim / expr / audio / post。
//!
//! # 错误风格（兼容性宪章 C18 / C19）
//! - v1 错误：保持原样（无错误码前缀）；
//! - v2 错误：三部分格式 `[Pxxxx] 在哪+什么事\n  建议：怎么改`；
//!   码段含义：P1xxx 防呆 / P2xxx 类型 / P3xxx 图结构 / P4xxx 资产 IO。

use crate::ast::{
    AnimAst, AudioAst, CameraV2Ast, ComposeAst, ComposeOpAst, ComposeRef, DocumentAst, DocumentV2Ast,
    ExprDefAst, FieldValue, FontDecl, ImportPacksAst, KeyAst, LayerAst, LightV2Ast, MaterialV2Ast,
    MovieAst, NamedField, NoteAst, OutputAst, PostAst, SceneItem, SceneV2Ast, SdfAst, SimAst,
    Stage3dAst, TimeVal, TimeUnit, TransitionAst, AestheticAst, AngleVal, AngleUnit,
    OutputFormatAst, SceneAst, SphereAst, MaterialAst, CameraAst, LightAst, TimelineAst,
};
use crate::{DslError, err_at};
use crate::lexer::{describe_tok, Tok, Token};
use prism_math::Vec3;

/// P1011 非有限时长（C20 禁静默容错）：inf / NaN 时长显式拒绝，不许静默
/// 直通产物（下游帧数 = inf/NaN 会污染整个时间线）。
fn err_duration_not_finite(line: usize, what: &str, value: f32) -> DslError {
    err_at(
        line,
        format!(
            "[P1011] {what} 应为有限的正时长，却得到 {value}（非有限值，超出 f32 表示范围）\n  建议：缩短时长到 f32 可表示的正值（上限约 3.4e38）；时长会参与帧数计算，inf/NaN 会污染产物，必须显式拒绝（C20）"
        ),
    )
}

pub(crate) struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub(crate) fn new(toks: Vec<Token>) -> Self {
        Self { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let tok: Option<Token> = self.toks.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// "意外到达文件末尾"时报错用的行号：取最后一个 token 所在行。
    fn eof_line(&self) -> usize {
        self.toks.last().map_or(1, |t| t.line)
    }

    fn expect_punct(&mut self, want: Tok, name: &str) -> Result<usize, DslError> {
        match self.bump() {
            Some(Token { tok, line }) if tok == want => Ok(line),
            Some(Token { tok: other, line }) => {
                Err(err_at(line, format!("期望 '{}'，却遇到 {}", name, describe_tok(&other))))
            }
            None => Err(err_at(self.eof_line(), format!("期望 '{}'，但文件已结束", name))),
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<(String, usize), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Ident(s), line }) => Ok((s, line)),
            Some(Token { tok: other, line }) => {
                Err(err_at(line, format!("期望 {}，却遇到 {}", what, describe_tok(&other))))
            }
            None => Err(err_at(self.eof_line(), format!("期望 {}，但文件已结束", what))),
        }
    }

    /// 期望一个普通数字，返回 (值, 行号)。
    fn expect_num(&mut self, what: &str) -> Result<(f32, usize), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Num(v), line }) => Ok((v, line)),
            Some(Token { tok: other, line }) => {
                Err(err_at(line, format!("期望 {}，却遇到 {}", what, describe_tok(&other))))
            }
            None => Err(err_at(self.eof_line(), format!("期望 {}，但文件已结束", what))),
        }
    }

    /// 期望一个 `#RRGGBB` 颜色。
    fn expect_color(&mut self, what: &str) -> Result<(u8, u8, u8), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Color(r, g, b), .. }) => Ok((r, g, b)),
            Some(Token { tok: other, line }) => {
                Err(err_at(line, format!("期望 {}，却遇到 {}", what, describe_tok(&other))))
            }
            None => Err(err_at(self.eof_line(), format!("期望 {}，但文件已结束", what))),
        }
    }

    /// 期望一个双引号字符串（v2 新增助手；v1 的 scene 名解析保持内联原样）。
    fn expect_str(&mut self, what: &str) -> Result<(String, usize), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Str(s), line }) => Ok((s, line)),
            Some(Token { tok: other, line }) => Err(err_at(
                line,
                format!("期望 {}（双引号字符串），却遇到 {}", what, describe_tok(&other)),
            )),
            None => Err(err_at(self.eof_line(), format!("期望 {}（双引号字符串），但文件已结束", what))),
        }
    }

    /// 期望时长（形如 `3s`），给出类型不匹配时的针对性提示。
    fn expect_duration(&mut self) -> Result<f32, DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Duration(v), .. }) => Ok(v),
            Some(Token { tok: Tok::Num(v), line }) => Err(err_at(
                line,
                format!("duration 应形如 '3s'（数字后跟单位 s），却得到数字 {}", v),
            )),
            Some(Token { tok: other, line }) => {
                Err(err_at(line, format!("期望 duration（形如 3s），却遇到 {}", describe_tok(&other))))
            }
            None => Err(err_at(self.eof_line(), "期望 duration（形如 3s），但文件已结束")),
        }
    }

    /// v2：期望带单位的时间字面量（`3s` / `150ms` / `36f`）。
    /// 裸数字按 C17（单位显式）报 P2102 并给出建议。
    fn expect_time(&mut self, what: &str) -> Result<TimeVal, DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Duration(v), line }) => {
                Ok(TimeVal { value: v, unit: TimeUnit::Seconds, line })
            }
            Some(Token { tok: Tok::Millis(v), line }) => {
                Ok(TimeVal { value: v, unit: TimeUnit::Millis, line })
            }
            Some(Token { tok: Tok::Frames(v), line }) => {
                Ok(TimeVal { value: v, unit: TimeUnit::Frames, line })
            }
            Some(Token { tok: Tok::Num(v), line }) => Err(err_at(
                line,
                format!(
                    "[P2102] {} 应带时间单位，却得到裸数字 {}\n  建议：写成 3s（秒）、150ms（毫秒）或 36f（帧，按 fps 折算）",
                    what, v
                ),
            )),
            Some(Token { tok: other, line }) => Err(err_at(
                line,
                format!(
                    "[P2102] 期望 {}（带时间单位，如 3s / 150ms / 36f），却遇到 {}\n  建议：参考 3s（秒）、150ms（毫秒）、36f（帧）三种写法",
                    what,
                    describe_tok(&other)
                ),
            )),
            None => Err(err_at(
                self.eof_line(),
                format!("[P2102] 期望 {}（带时间单位），但文件已结束\n  建议：写成 3s / 150ms / 36f 之一", what),
            )),
        }
    }

    /// v2：期望带单位的角度字面量（`15deg` / `0.5rad`）。
    /// 裸数字按 C17 报 P2101 并给出建议。
    fn expect_angle(&mut self, what: &str) -> Result<AngleVal, DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Deg(v), line }) => {
                Ok(AngleVal { value: v, unit: AngleUnit::Deg, line })
            }
            Some(Token { tok: Tok::Rad(v), line }) => {
                Ok(AngleVal { value: v, unit: AngleUnit::Rad, line })
            }
            Some(Token { tok: Tok::Num(v), line }) => Err(err_at(
                line,
                format!(
                    "[P2101] {} 应带角度单位，却得到裸数字 {}\n  建议：写成 15deg（度）或 0.5rad（弧度）",
                    what, v
                ),
            )),
            Some(Token { tok: other, line }) => Err(err_at(
                line,
                format!(
                    "[P2101] 期望 {}（带角度单位，如 15deg / 0.5rad），却遇到 {}\n  建议：15deg 表示 15 度，0.5rad 表示 0.5 弧度",
                    what,
                    describe_tok(&other)
                ),
            )),
            None => Err(err_at(
                self.eof_line(),
                format!("[P2101] 期望 {}（带角度单位），但文件已结束\n  建议：写成 15deg / 0.5rad 之一", what),
            )),
        }
    }

    /// 期望分辨率（形如 `480x270`），返回 (宽, 高, 行号)。
    fn expect_resolution(&mut self) -> Result<(u32, u32, usize), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Resolution(w, h), line }) => Ok((w, h, line)),
            Some(Token { tok: Tok::Num(v), line }) => Err(err_at(
                line,
                format!(
                    "resolution 应形如 '480x270'（宽 x 高，中间为小写字母 x），却得到数字 {}",
                    v
                ),
            )),
            Some(Token { tok: other, line }) => Err(err_at(
                line,
                format!("期望 resolution（形如 480x270），却遇到 {}", describe_tok(&other)),
            )),
            None => Err(err_at(self.eof_line(), "期望 resolution（形如 480x270），但文件已结束")),
        }
    }

    /// 期望一个逗号（vec3 分量之间）。
    fn expect_comma(&mut self, what: &str, after_idx: usize) -> Result<(), DslError> {
        match self.bump() {
            Some(Token { tok: Tok::Comma, .. }) => Ok(()),
            Some(Token { tok: other, line }) => Err(err_at(
                line,
                format!(
                    "{} 需要 3 个以逗号分隔的数字：第 {}、{} 个数字之间缺少 ','，却遇到 {}",
                    what,
                    after_idx,
                    after_idx + 1,
                    describe_tok(&other)
                ),
            )),
            None => Err(err_at(
                self.eof_line(),
                format!("{} 需要 3 个以逗号分隔的数字，但文件已结束", what),
            )),
        }
    }

    /// 裸 vec3：`数字 , 数字 , 数字`（如 `center 0,0,0`，逗号两侧允许空白、允许负数）。
    fn parse_vec3(&mut self, what: &str) -> Result<Vec3, DslError> {
        let (x, _) = self.expect_num(what)?;
        self.expect_comma(what, 1)?;
        let (y, _) = self.expect_num(what)?;
        self.expect_comma(what, 2)?;
        let (z, _) = self.expect_num(what)?;
        Ok(Vec3::new(x, y, z))
    }

    // -----------------------------------------------------------------------
    // v1 文档（原样保留）
    // -----------------------------------------------------------------------

    /// 文档：`scene 块 [timeline 块] [aesthetic 块] [output 块]`。
    ///
    /// 宽容性约定：scene 必须在最前；timeline / aesthetic / output 允许任意顺序、
    /// 各至多一次（黄金示例按 scene → timeline → aesthetic → output 排列）。
    pub(crate) fn parse_document(&mut self) -> Result<DocumentAst, DslError> {
        // scene 块必需，且必须是第一个块。
        let first: &Token = match self.peek() {
            Some(t) => t,
            None => return Err(err_at(1, "空文档：至少需要一个 'scene' 块")),
        };
        let (kw, line): (String, usize) = match &first.tok {
            Tok::Ident(s) => (s.clone(), first.line),
            other => {
                return Err(err_at(
                    first.line,
                    format!("文档应以 'scene' 块开始，却遇到 {}", describe_tok(other)),
                ));
            }
        };
        if kw != "scene" {
            // v2 块出现在 v1 入口时给出升级指引；其余消息保持 v1 原样。
            let v2_hint: &str = if matches!(
                kw.as_str(),
                "movie" | "theme" | "transition" | "anim" | "expr" | "import"
            ) {
                "\n  建议：检测到 DSL v2 块语法，请改用 compile_v2 / compile_v2_with_diagnostics"
            } else {
                ""
            };
            return Err(err_at(line, format!("文档应以 'scene' 块开始，却遇到 '{}'{}", kw, v2_hint)));
        }
        self.bump(); // 消费掉已校验的 'scene' 关键字，再交给 parse_scene 解析名称与块体。
        let scene: SceneAst = self.parse_scene()?;

        let mut timeline: Option<TimelineAst> = None;
        let mut aesthetic: Option<AestheticAst> = None;
        let mut output: Option<OutputAst> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => break,
            };
            let (kw, line): (String, usize) = match t.tok {
                Tok::Ident(kw) => (kw, t.line),
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "顶层期望块（timeline / aesthetic / output），却遇到 {}",
                            describe_tok(&other)
                        ),
                    ));
                }
            };
            match kw.as_str() {
                "timeline" => {
                    if timeline.is_some() {
                        return Err(err_at(line, "'timeline' 块重复出现（最多一个）"));
                    }
                    timeline = Some(self.parse_timeline()?);
                }
                "aesthetic" => {
                    if aesthetic.is_some() {
                        return Err(err_at(line, "'aesthetic' 块重复出现（最多一个）"));
                    }
                    aesthetic = Some(self.parse_aesthetic()?);
                }
                "output" => {
                    if output.is_some() {
                        return Err(err_at(line, "'output' 块重复出现（最多一个）"));
                    }
                    output = Some(self.parse_output()?);
                }
                "scene" => {
                    return Err(err_at(line, "'scene' 块重复出现（一个文档只能有一个 scene 块）"));
                }
                other => {
                    return Err(err_at(
                        line,
                        format!("未知顶层块 '{}'（允许：timeline / aesthetic / output）", other),
                    ));
                }
            }
        }
        Ok(DocumentAst { scene, timeline, aesthetic, output })
    }

    /// scene 块：`scene 字符串 { (sphere | camera | light)* }`。
    fn parse_scene(&mut self) -> Result<SceneAst, DslError> {
        let name: String = match self.bump() {
            Some(Token { tok: Tok::Str(s), .. }) => s,
            Some(Token { tok: other, line }) => {
                return Err(err_at(
                    line,
                    format!("期望 scene 名称（字符串，如 \"my_scene\"），却遇到 {}", describe_tok(&other)),
                ));
            }
            None => return Err(err_at(self.eof_line(), "期望 scene 名称，但文件已结束")),
        };
        self.expect_punct(Tok::LBrace, "{")?;
        let mut sphere: Option<SphereAst> = None;
        let mut camera: Option<CameraAst> = None;
        let mut lights: Vec<LightAst> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "scene 块缺少右花括号 '}'")),
            };
            let (kw, line): (String, usize) = match t.tok {
                Tok::RBrace => break,
                Tok::Ident(kw) => (kw, t.line),
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "scene 块中期望 sphere / camera / light 或 '}}'，却遇到 {}",
                            describe_tok(&other)
                        ),
                    ));
                }
            };
            // 扩展点：未来加 mesh 时在此新增 `"mesh" => ...` 分支即可。
            match kw.as_str() {
                "sphere" => {
                    if sphere.is_some() {
                        return Err(err_at(line, "scene 目前最多支持一个 sphere（出现第二个）"));
                    }
                    sphere = Some(self.parse_sphere()?);
                }
                "camera" => {
                    if camera.is_some() {
                        return Err(err_at(line, "scene 目前最多支持一个 camera（出现第二个）"));
                    }
                    camera = Some(self.parse_camera()?);
                }
                "light" => lights.push(self.parse_light()?),
                other => {
                    return Err(err_at(
                        line,
                        format!("scene 块中未知条目 '{}'（允许：sphere / camera / light）", other),
                    ));
                }
            }
        }
        Ok(SceneAst { name, sphere, camera, lights })
    }

    /// sphere 块：`sphere 标识 { radius 数字 material块? }`（radius 必填）。
    fn parse_sphere(&mut self) -> Result<SphereAst, DslError> {
        let (name, _) = self.expect_ident("sphere 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut radius: Option<f32> = None;
        let mut material: Option<MaterialAst> = None;
        // 仅在遇到 '}' 时赋值（break 前必然赋值，编译器可静态判定）。
        let close_line: usize;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "sphere 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => {
                    close_line = t.line;
                    break;
                }
                Tok::Ident(field) => match field.as_str() {
                    "radius" => {
                        if radius.is_some() {
                            return Err(err_at(t.line, "sphere 块中字段 'radius' 重复出现"));
                        }
                        radius = Some(self.expect_num("sphere.radius 的数字值")?.0);
                    }
                    "material" => {
                        if material.is_some() {
                            return Err(err_at(t.line, "sphere 块中 'material' 重复出现"));
                        }
                        material = Some(self.parse_material()?);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!("sphere 块中未知字段 '{}'（允许：radius / material）", other),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("sphere 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        let radius: f32 = match radius {
            Some(r) => r,
            None => {
                return Err(err_at(close_line, format!("sphere '{}' 缺少必填字段 'radius'", name)));
            }
        };
        Ok(SphereAst { name, radius, material })
    }

    /// material 块：`material { (albedo 颜色)? (roughness 数字)? (metalness 数字)? }`。
    fn parse_material(&mut self) -> Result<MaterialAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut albedo: Option<(u8, u8, u8)> = None;
        let mut roughness: Option<f32> = None;
        let mut metalness: Option<f32> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "material 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "albedo" => {
                        if albedo.is_some() {
                            return Err(err_at(t.line, "material 块中字段 'albedo' 重复出现"));
                        }
                        albedo = Some(self.expect_color("albedo 的 #RRGGBB 颜色")?);
                    }
                    "roughness" => {
                        if roughness.is_some() {
                            return Err(err_at(t.line, "material 块中字段 'roughness' 重复出现"));
                        }
                        roughness = Some(self.expect_num("roughness 的数字值")?.0);
                    }
                    "metalness" => {
                        if metalness.is_some() {
                            return Err(err_at(t.line, "material 块中字段 'metalness' 重复出现"));
                        }
                        metalness = Some(self.expect_num("metalness 的数字值")?.0);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "material 块中未知字段 '{}'（允许：albedo / roughness / metalness）",
                                other
                            ),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("material 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        Ok(MaterialAst { albedo, roughness, metalness })
    }

    /// camera 块：`camera 标识 { orbit { ... } fov 数字 }`（orbit 必填且在前，fov 必填）。
    fn parse_camera(&mut self) -> Result<CameraAst, DslError> {
        let (name, _) = self.expect_ident("camera 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let (kw, kw_line): (String, usize) = self.expect_ident("camera 块的第一个字段 'orbit'")?;
        if kw != "orbit" {
            return Err(err_at(
                kw_line,
                format!("camera 块必须以 'orbit {{ ... }}' 开始，却遇到 '{}'", kw),
            ));
        }
        let (orbit_center, orbit_radius) = self.parse_orbit()?;
        let (fov_kw, fov_line): (String, usize) = self.expect_ident("字段 'fov'")?;
        if fov_kw != "fov" {
            return Err(err_at(
                fov_line,
                format!("camera 块中 orbit 之后期望字段 'fov'，却遇到 '{}'", fov_kw),
            ));
        }
        let (fov_deg, _) = self.expect_num("fov 的度数值（如 45 表示 45°）")?;
        self.expect_punct(Tok::RBrace, "}")?;
        Ok(CameraAst { name, orbit_center, orbit_radius, fov_deg })
    }

    /// orbit 块：`orbit { (center vec3)? (radius 数字)? }`。
    fn parse_orbit(&mut self) -> Result<(Option<Vec3>, Option<f32>), DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut center: Option<Vec3> = None;
        let mut radius: Option<f32> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "orbit 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "center" => {
                        if center.is_some() {
                            return Err(err_at(t.line, "orbit 块中字段 'center' 重复出现"));
                        }
                        center = Some(self.parse_vec3("orbit.center（vec3：数字,数字,数字）")?);
                    }
                    "radius" => {
                        if radius.is_some() {
                            return Err(err_at(t.line, "orbit 块中字段 'radius' 重复出现"));
                        }
                        radius = Some(self.expect_num("orbit.radius 的数字值")?.0);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!("orbit 块中未知字段 '{}'（允许：center / radius）", other),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("orbit 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        Ok((center, radius))
    }

    /// light 块：`light 标识 { direction '(' vec3 ')' (color 颜色)? (intensity 数字)? }`。
    fn parse_light(&mut self) -> Result<LightAst, DslError> {
        let (name, _) = self.expect_ident("light 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut direction: Option<Vec3> = None;
        let mut color: Option<(u8, u8, u8)> = None;
        let mut intensity: Option<f32> = None;
        // 仅在遇到 '}' 时赋值（break 前必然赋值，编译器可静态判定）。
        let close_line: usize;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "light 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => {
                    close_line = t.line;
                    break;
                }
                Tok::Ident(field) => match field.as_str() {
                    "direction" => {
                        if direction.is_some() {
                            return Err(err_at(t.line, "light 块中字段 'direction' 重复出现"));
                        }
                        self.expect_punct(Tok::LParen, "(")?;
                        let v: Vec3 = self.parse_vec3("light.direction（vec3：数字,数字,数字）")?;
                        self.expect_punct(Tok::RParen, ")")?;
                        direction = Some(v);
                    }
                    "color" => {
                        if color.is_some() {
                            return Err(err_at(t.line, "light 块中字段 'color' 重复出现"));
                        }
                        color = Some(self.expect_color("color 的 #RRGGBB 颜色")?);
                    }
                    "intensity" => {
                        if intensity.is_some() {
                            return Err(err_at(t.line, "light 块中字段 'intensity' 重复出现"));
                        }
                        intensity = Some(self.expect_num("intensity 的数字值")?.0);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "light 块中未知字段 '{}'（允许：direction / color / intensity）",
                                other
                            ),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("light 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        let direction: Vec3 = match direction {
            Some(v) => v,
            None => {
                return Err(err_at(
                    close_line,
                    format!("light '{}' 缺少必填字段 'direction'", name),
                ));
            }
        };
        Ok(LightAst { name, direction, color, intensity })
    }

    /// timeline 块：`timeline { (duration 时长)? (fps 整数)? }`。
    fn parse_timeline(&mut self) -> Result<TimelineAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut duration: Option<f32> = None;
        let mut fps: Option<u32> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "timeline 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "duration" => {
                        if duration.is_some() {
                            return Err(err_at(t.line, "timeline 块中字段 'duration' 重复出现"));
                        }
                        let d: f32 = self.expect_duration()?;
                        if !d.is_finite() {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "duration 应为正时长（大于 0s），却得到 {}（非有限值，超出表示范围）",
                                    d
                                ),
                            ));
                        }
                        if d <= 0.0 {
                            return Err(err_at(
                                t.line,
                                format!("duration 应为正时长（大于 0s），却得到 {}s", d),
                            ));
                        }
                        duration = Some(d);
                    }
                    "fps" => {
                        if fps.is_some() {
                            return Err(err_at(t.line, "timeline 块中字段 'fps' 重复出现"));
                        }
                        let (v, v_line) = self.expect_num("fps 的正整数值")?;
                        if v.fract() != 0.0 {
                            return Err(err_at(v_line, format!("fps 应为整数，却得到 {}", v)));
                        }
                        if v < 1.0 {
                            return Err(err_at(v_line, format!("fps 应为正整数（至少 1），却得到 {}", v)));
                        }
                        fps = Some(v as u32);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!("timeline 块中未知字段 '{}'（允许：duration / fps）", other),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("timeline 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        Ok(TimelineAst { duration, fps })
    }

    /// aesthetic 块：`aesthetic { teal-orange? (contrast 数字)? filmic? }`。
    /// v2 的 theme 块字段集相同，复用 [`Parser::parse_aesthetic_fields`]。
    fn parse_aesthetic(&mut self) -> Result<AestheticAst, DslError> {
        self.parse_aesthetic_fields("aesthetic")
    }

    // -----------------------------------------------------------------------
    // v1 / v2 共用：aesthetic / theme 字段循环（消息按块名参数化，v1 逐字一致）
    // -----------------------------------------------------------------------

    /// aesthetic / theme 块的字段循环（`teal-orange` / `contrast` / `filmic`）。
    fn parse_aesthetic_fields(&mut self, block: &str) -> Result<AestheticAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut teal_orange: bool = false;
        let mut contrast: Option<f32> = None;
        let mut filmic: bool = false;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => {
                    return Err(err_at(self.eof_line(), format!("{} 块缺少右花括号 '}}'", block)));
                }
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "teal-orange" => {
                        if teal_orange {
                            return Err(err_at(t.line, format!("{} 块中 'teal-orange' 重复出现", block)));
                        }
                        teal_orange = true;
                    }
                    "filmic" => {
                        if filmic {
                            return Err(err_at(t.line, format!("{} 块中 'filmic' 重复出现", block)));
                        }
                        filmic = true;
                    }
                    "contrast" => {
                        if contrast.is_some() {
                            return Err(err_at(
                                t.line,
                                format!("{} 块中字段 'contrast' 重复出现", block),
                            ));
                        }
                        contrast = Some(self.expect_num("contrast 的数字值")?.0);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "{} 块中未知字段 '{}'（允许：teal-orange / contrast / filmic）",
                                block, other
                            ),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "{} 块中期望字段名或 '}}'，却遇到 {}",
                            block,
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        Ok(AestheticAst { teal_orange, contrast, filmic })
    }

    // -----------------------------------------------------------------------
    // v2：output（与 v1 同构，独立方法避免交叉依赖）
    // -----------------------------------------------------------------------

    /// output 块：`output { (resolution 分辨率)? (format ('frames'|'png'))? }`。
    fn parse_output(&mut self) -> Result<OutputAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut resolution: Option<(u32, u32)> = None;
        let mut format: Option<OutputFormatAst> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "output 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "resolution" => {
                        if resolution.is_some() {
                            return Err(err_at(t.line, "output 块中字段 'resolution' 重复出现"));
                        }
                        let (w, h, r_line) = self.expect_resolution()?;
                        if w == 0 || h == 0 {
                            return Err(err_at(
                                r_line,
                                "[P1013] 分辨率宽高必须大于 0\n  建议：检查 resolution 的宽与高（最小 1x1）",
                            ));
                        }
                        resolution = Some((w, h));
                    }
                    "format" => {
                        if format.is_some() {
                            return Err(err_at(t.line, "output 块中字段 'format' 重复出现"));
                        }
                        let (value, v_line) =
                            self.expect_ident("format 的取值（'frames' 或 'png'）")?;
                        format = Some(match value.as_str() {
                            "frames" => OutputFormatAst::Frames,
                            "png" => OutputFormatAst::Png,
                            other => {
                                return Err(err_at(
                                    v_line,
                                    format!("format 只能是 'frames' 或 'png'，却遇到 '{}'", other),
                                ));
                            }
                        });
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!("output 块中未知字段 '{}'（允许：resolution / format）", other),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("output 块中期望字段名或 '}}'，却遇到 {}", describe_tok(&other)),
                    ));
                }
            }
        }
        Ok(OutputAst { resolution, format })
    }

    // -----------------------------------------------------------------------
    // v2：全块语法
    // -----------------------------------------------------------------------

    /// v2 文档：`块*`（movie / timeline / theme / import / scene / transition /
    /// anim / expr / output；各块重复规则见分支）。
    pub(crate) fn parse_document_v2(&mut self) -> Result<DocumentV2Ast, DslError> {
        let mut doc = DocumentV2Ast::default();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => break,
            };
            let (kw, line): (String, usize) = match t.tok {
                Tok::Ident(kw) => (kw, t.line),
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] 顶层期望块定义（movie / font / scene / transition 等），却遇到 {}\n  建议：顶层只能出现 movie / timeline / theme / import packs / font / scene / transition / anim / expr / output 之一",
                            describe_tok(&other)
                        ),
                    ));
                }
            };
            match kw.as_str() {
                "movie" => {
                    if doc.movie.is_some() {
                        return Err(err_at(
                            line,
                            "[P1001] 'movie' 块重复出现（最多一个）\n  建议：合并两个 movie 块的字段到一处",
                        ));
                    }
                    doc.movie = Some(self.parse_movie(line)?);
                }
                "timeline" => {
                    if doc.timeline.is_some() {
                        return Err(err_at(
                            line,
                            "[P1001] 'timeline' 块重复出现（最多一个）\n  建议：合并两个 timeline 块的字段到一处",
                        ));
                    }
                    doc.timeline = Some(self.parse_timeline()?);
                }
                "theme" => {
                    if doc.theme.is_some() {
                        return Err(err_at(
                            line,
                            "[P1001] 'theme' 块重复出现（最多一个）\n  建议：合并两个 theme 块的字段到一处",
                        ));
                    }
                    doc.theme = Some(self.parse_aesthetic_fields("theme")?);
                }
                "import" => doc.imports.push(self.parse_import(line)?),
                "font" => doc.fonts.push(self.parse_font(line)?),
                "scene" => doc.scenes.push(self.parse_scene_v2(line)?),
                "transition" => doc.transitions.push(self.parse_transition(line)?),
                "anim" => doc.anims.push(self.parse_anim(line)?),
                "expr" => doc.exprs.push(self.parse_expr_def(line)?),
                "output" => {
                    if doc.output.is_some() {
                        return Err(err_at(
                            line,
                            "[P1001] 'output' 块重复出现（最多一个）\n  建议：合并两个 output 块的字段到一处",
                        ));
                    }
                    doc.output = Some(self.parse_output()?);
                }
                "audio" | "post" => {
                    return Err(err_at(
                        line,
                        format!(
                            "[P1004] '{}' 是场景内条目，不能出现在顶层\n  建议：把 {} 块移入某个 scene 块内",
                            kw, kw
                        ),
                    ));
                }
                other => {
                    return Err(err_at(
                        line,
                        format!(
                            "[P1002] 未知顶层块 '{}'（允许：movie / timeline / theme / import packs / font / scene / transition / anim / expr / output）\n  建议：检查块名拼写；audio 与 post 应写在 scene 块内",
                            other
                        ),
                    ));
                }
            }
        }
        if doc.scenes.is_empty() {
            return Err(err_at(
                1,
                "[P1003] v2 文档至少需要一个 scene 块\n  建议：添加 scene \"场景名\" { ... }；scene 内可写 stage3d / layer / sim / compose / audio / post",
            ));
        }
        Ok(doc)
    }

    /// movie 块：`movie "标题" { duration? fps? resolution? format? }`。
    fn parse_movie(&mut self, kw_line: usize) -> Result<MovieAst, DslError> {
        let (title, _) = self.expect_str("movie 标题")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut duration: Option<TimeVal> = None;
        let mut fps: Option<u32> = None;
        let mut resolution: Option<(u32, u32)> = None;
        let mut format: Option<OutputFormatAst> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "movie 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "duration" => {
                        if duration.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] movie 块中字段 'duration' 重复出现\n  建议：删除多余 duration 字段",
                            ));
                        }
                        let tv: TimeVal = self.expect_time("movie.duration")?;
                        if !tv.value.is_finite() {
                            return Err(err_duration_not_finite(t.line, "movie.duration", tv.value));
                        }
                        if tv.value <= 0.0 {
                            return Err(err_at(
                                t.line,
                                "[P1011] movie.duration 应为正时长（大于 0）\n  建议：给出正时长，如 6s",
                            ));
                        }
                        duration = Some(tv);
                    }
                    "fps" => {
                        if fps.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] movie 块中字段 'fps' 重复出现\n  建议：删除多余 fps 字段",
                            ));
                        }
                        fps = Some(self.parse_fps_field("movie")?);
                    }
                    "resolution" => {
                        if resolution.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] movie 块中字段 'resolution' 重复出现\n  建议：删除多余 resolution 字段",
                            ));
                        }
                        let (w, h, r_line) = self.expect_resolution()?;
                        if w == 0 || h == 0 {
                            return Err(err_at(
                                r_line,
                                "[P1013] 分辨率宽高必须大于 0\n  建议：检查 resolution 的宽与高（最小 1x1）",
                            ));
                        }
                        resolution = Some((w, h));
                    }
                    "format" => {
                        if format.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] movie 块中字段 'format' 重复出现\n  建议：删除多余 format 字段",
                            ));
                        }
                        format = Some(self.parse_format_value()?);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "[P1002] movie 块中未知字段 '{}'（允许：duration / fps / resolution / format）\n  建议：检查字段名拼写",
                                other
                            ),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!("[P1001] movie 块中期望字段名或 '}}'，却遇到 {}\n  建议：movie 块字段形如 duration 6s / fps 24 / resolution 1280x720 / format png", describe_tok(&other)),
                    ));
                }
            }
        }
        Ok(MovieAst { title, duration, fps, resolution, format, line: kw_line })
    }

    /// fps 字段（movie / timeline 通用）：正整数校验 + 防呆码。
    fn parse_fps_field(&mut self, block: &str) -> Result<u32, DslError> {
        let (v, v_line) = self.expect_num("fps 的正整数值")?;
        if v.fract() != 0.0 {
            return Err(err_at(
                v_line,
                format!("[P1012] fps 应为整数，却得到 {}\n  建议：fps 表示每秒帧数，写成整数（如 24）", v),
            ));
        }
        if v < 1.0 {
            return Err(err_at(
                v_line,
                format!(
                    "[P1012] {} 块中 fps 应为正整数（至少 1），却得到 {}\n  建议：常见取值 12 / 24 / 30 / 60",
                    block, v
                ),
            ));
        }
        Ok(v as u32)
    }

    /// format 字段值：`frames` 或 `png`。
    fn parse_format_value(&mut self) -> Result<OutputFormatAst, DslError> {
        let (value, v_line) = self.expect_ident("format 的取值（'frames' 或 'png'）")?;
        match value.as_str() {
            "frames" => Ok(OutputFormatAst::Frames),
            "png" => Ok(OutputFormatAst::Png),
            other => Err(err_at(
                v_line,
                format!(
                    "[P2106] format 只能是 'frames' 或 'png'，却遇到 '{}'\n  建议：frames = 原始帧序列，png = PNG 图片序列",
                    other
                ),
            )),
        }
    }

    /// scene v2 块：`scene "名" { duration? (stage3d | layer | sim | compose | audio | post)* }`。
    fn parse_scene_v2(&mut self, kw_line: usize) -> Result<SceneV2Ast, DslError> {
        let (name, _) = self.expect_str("scene 名称")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut duration: Option<TimeVal> = None;
        let mut items: Vec<SceneItem> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "scene 块缺少右花括号 '}'")),
            };
            let (kw, line): (String, usize) = match t.tok {
                Tok::RBrace => break,
                Tok::Ident(kw) => (kw, t.line),
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] scene 块中期望条目（stage3d / layer / sim / compose / audio / note / post）或 '}}'，却遇到 {}\n  建议：v2 scene 的条目均为块；sphere / camera / light 等 v1 条目属于 stage3d 内部",
                            describe_tok(&other)
                        ),
                    ));
                }
            };
            match kw.as_str() {
                "duration" => {
                    if duration.is_some() {
                        return Err(err_at(
                            t.line,
                            "[P1001] scene 块中字段 'duration' 重复出现\n  建议：删除多余 duration 字段",
                        ));
                    }
                    let tv: TimeVal = self.expect_time("scene.duration")?;
                    if !tv.value.is_finite() {
                        return Err(err_duration_not_finite(t.line, "scene.duration", tv.value));
                    }
                    if tv.value <= 0.0 {
                        return Err(err_at(
                            t.line,
                            "[P1011] scene.duration 应为正时长（大于 0）\n  建议：给出正时长，如 2s",
                        ));
                    }
                    duration = Some(tv);
                }
                "stage3d" => items.push(SceneItem::Stage3d(self.parse_stage3d(line)?)),
                "layer" => items.push(SceneItem::Layer(self.parse_layer(line)?)),
                "sim" => items.push(SceneItem::Sim(self.parse_sim(line)?)),
                "compose" => items.push(SceneItem::Compose(self.parse_compose(line)?)),
                "audio" => items.push(SceneItem::Audio(self.parse_audio(line)?)),
                "note" => items.push(SceneItem::Note(self.parse_note(line)?)),
                "post" => items.push(SceneItem::Post(self.parse_post(line)?)),
                other => {
                    return Err(err_at(
                        line,
                        format!(
                            "[P1002] scene 块中未知条目 '{}'（允许：stage3d / layer / sim / compose / audio / note / post）\n  建议：检查条目名拼写",
                            other
                        ),
                    ));
                }
            }
        }
        Ok(SceneV2Ast { name, duration, items, line: kw_line })
    }

    /// stage3d 块：`stage3d [名] { camera? sdf* material* light* }`（名可省略）。
    fn parse_stage3d(&mut self, kw_line: usize) -> Result<Stage3dAst, DslError> {
        // 名可省略（Ident / Str / 无 → 缺省 "stage3d"）。
        let name: String = self.parse_optional_block_name("stage3d 名称")?;
        let name: String = if name.is_empty() { "stage3d".to_string() } else { name };
        self.expect_punct(Tok::LBrace, "{")?;
        let mut camera: Option<CameraV2Ast> = None;
        let mut sdfs: Vec<SdfAst> = Vec::new();
        let mut materials: Vec<MaterialV2Ast> = Vec::new();
        let mut lights: Vec<LightV2Ast> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "stage3d 块缺少右花括号 '}'")),
            };
            let (kw, line): (String, usize) = match t.tok {
                Tok::RBrace => break,
                Tok::Ident(kw) => (kw, t.line),
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] stage3d 块中期望条目（camera / sdf / material / light）或 '}}'，却遇到 {}\n  建议：stage3d 条目均为块，如 sdf ball {{ kind sphere radius 1 }}",
                            describe_tok(&other)
                        ),
                    ));
                }
            };
            match kw.as_str() {
                "camera" => {
                    if camera.is_some() {
                        return Err(err_at(
                            line,
                            "[P1001] stage3d 目前最多支持一个 camera（出现第二个）\n  建议：合并相机参数到同一 camera 块",
                        ));
                    }
                    camera = Some(self.parse_camera_v2(line)?);
                }
                "sdf" => sdfs.push(self.parse_sdf(line)?),
                "material" => materials.push(self.parse_material_v2(line)?),
                "light" => lights.push(self.parse_light_v2(line)?),
                other => {
                    return Err(err_at(
                        line,
                        format!(
                            "[P1002] stage3d 块中未知条目 '{}'（允许：camera / sdf / material / light）\n  建议：layer / sim 等合成层条目应写在 scene 块内而非 stage3d 内",
                            other
                        ),
                    ));
                }
            }
        }
        Ok(Stage3dAst { name, camera, sdfs, materials, lights, line: kw_line })
    }

    // -----------------------------------------------------------------------
    // v2：stage3d 条目与通用字段循环
    // -----------------------------------------------------------------------

    /// stage3d 内的 camera 条目：`camera 名 { orbit { ... } fov 45 }`。
    /// fov 沿用 v1 豁免：裸数字按"度"解释（C17 豁免清单）；也接受 `45deg`。
    /// fov 超界校验（P1014）在 codegen。
    fn parse_camera_v2(&mut self, kw_line: usize) -> Result<CameraV2Ast, DslError> {
        let (name, _) = self.expect_ident("camera 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("camera")?;
        Ok(CameraV2Ast { name, fields, line: kw_line })
    }

    /// stage3d 内的 sdf 条目：`sdf 名 { kind sphere radius 1 }`（kind 必填，codegen 校验）。
    fn parse_sdf(&mut self, kw_line: usize) -> Result<SdfAst, DslError> {
        let (name, _) = self.expect_ident("sdf 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("sdf")?;
        Ok(SdfAst { name, fields, line: kw_line })
    }

    /// stage3d 内的 material 条目：`material 名 { albedo #.. roughness .. metalness .. }`。
    fn parse_material_v2(&mut self, kw_line: usize) -> Result<MaterialV2Ast, DslError> {
        let (name, _) = self.expect_ident("material 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("material")?;
        Ok(MaterialV2Ast { name, fields, line: kw_line })
    }

    /// stage3d 内的 light 条目：`light 名 { direction(1,-1,-0.5) color #.. intensity .. }`。
    fn parse_light_v2(&mut self, kw_line: usize) -> Result<LightV2Ast, DslError> {
        let (name, _) = self.expect_ident("light 名称（标识符）")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("light")?;
        Ok(LightV2Ast { name, fields, line: kw_line })
    }

    /// layer 块：`layer <种类> [名] { 字段* }`（名可省略或为标识符 / 字符串，缺省同种类）。
    fn parse_layer(&mut self, kw_line: usize) -> Result<LayerAst, DslError> {
        let (kind, _) = self.expect_ident("layer 种类（如 text / particles）")?;
        let name: String = self.parse_optional_block_name("layer 名称")?;
        if name.is_empty() {
            // 名缺省同种类。
        }
        let name: String = if name.is_empty() { kind.clone() } else { name };
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("layer")?;
        Ok(LayerAst { kind, name, fields, line: kw_line })
    }

    /// 可选块名：标识符 / 字符串 / 缺省（空串）。供 layer / sim / compose / stage3d。
    fn parse_optional_block_name(&mut self, what: &str) -> Result<String, DslError> {
        match self.peek().map(|t| &t.tok) {
            Some(Tok::Ident(_)) => Ok(self.expect_ident(what)?.0),
            Some(Tok::Str(_)) => Ok(self.expect_str(what)?.0),
            _ => Ok(String::new()),
        }
    }

    /// sim 块：`sim <种类> [名] { 字段* force块? noise块? }`（名可省略，缺省同种类）。
    /// force / noise 以子块形式出现在字段列表中，codegen 白名单校验。
    fn parse_sim(&mut self, kw_line: usize) -> Result<SimAst, DslError> {
        let (kind, _) = self.expect_ident("sim 种类（如 cloth / fluid / rigid）")?;
        let name: String = self.parse_optional_block_name("sim 名称")?;
        let name: String = if name.is_empty() { kind.clone() } else { name };
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("sim")?;
        Ok(SimAst { kind, name, fields, line: kw_line })
    }

    /// compose 块：`compose [名] { 操作* }`；操作：`add layer:text layer:fx`
    /// （操作数以 `种类:名字` 引用，种类可为 layer / sim / compose）。
    fn parse_compose(&mut self, kw_line: usize) -> Result<ComposeAst, DslError> {
        // 名可省略（Ident / Str / 无 → 缺省 "compose"）。
        let name: String = self.parse_optional_block_name("compose 名称")?;
        let name: String = if name.is_empty() { "compose".to_string() } else { name };
        self.expect_punct(Tok::LBrace, "{")?;
        let mut ops: Vec<ComposeOpAst> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "compose 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(op) => {
                    // 操作数：ident 后紧跟 ':' 才是引用（否则是下一个操作名）。
                    let mut refs: Vec<ComposeRef> = Vec::new();
                    while let Some(Token { tok: Tok::Ident(kind), .. }) = self.peek().cloned() {
                        if !matches!(
                            self.toks.get(self.pos + 1).map(|x| &x.tok),
                            Some(Tok::Colon)
                        ) {
                            break;
                        }
                        self.bump(); // 消费引用种类 ident
                        self.expect_punct(Tok::Colon, ":")?;
                        let (ref_name, ref_line) = self.expect_ident("引用的块名")?;
                        refs.push(ComposeRef { kind, name: ref_name, line: ref_line });
                    }
                    if refs.is_empty() {
                        return Err(err_at(
                            t.line,
                            format!(
                                "[P1003] compose 操作 '{}' 至少需要一个操作数（如 add layer:text）\n  建议：操作数写法为 `种类:名字`，种类可为 layer / sim / compose",
                                op
                            ),
                        ));
                    }
                    ops.push(ComposeOpAst { op, refs, line: t.line });
                }
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] compose 块中期望操作名或 '}}'，却遇到 {}\n  建议：操作写法为 add layer:text layer:fx",
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        if ops.is_empty() {
            return Err(err_at(
                kw_line,
                "[P1003] compose 块为空（至少一个操作）\n  建议：添加 add layer:text 之类的合成操作",
            ));
        }
        Ok(ComposeAst { name, ops, line: kw_line })
    }

    /// audio 块：`audio { music { bpm 120 gain 0.8 duck 12 } }`（开放字段，codegen 白名单）。
    fn parse_audio(&mut self, kw_line: usize) -> Result<AudioAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("audio")?;
        Ok(AudioAst { fields, line: kw_line })
    }

    /// note 块：`note "主题" { scale major root 60 bpm 120 bars 4 seed 7 }`
    /// （开放字段，codegen 白名单）。作曲系统的 DSL 入口。
    fn parse_note(&mut self, kw_line: usize) -> Result<NoteAst, DslError> {
        let (name, _) = self.expect_str("note 名称")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("note")?;
        Ok(NoteAst { name, fields, line: kw_line })
    }

    /// post 块：`post { bloom 0.3 grain 0.1 }`（开放字段，codegen 白名单）。
    fn parse_post(&mut self, kw_line: usize) -> Result<PostAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let fields = self.parse_named_fields("post")?;
        Ok(PostAst { fields, line: kw_line })
    }

    /// transition 块：`transition "名" { from "场景A" to "场景B" duration 0.5s kind crossfade }`。
    fn parse_transition(&mut self, kw_line: usize) -> Result<TransitionAst, DslError> {
        let (name, _) = self.expect_str("transition 名称")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut from: Option<(String, usize)> = None;
        let mut to: Option<(String, usize)> = None;
        let mut duration: Option<TimeVal> = None;
        let mut kind: String = "crossfade".to_string();
        let mut kind_seen: bool = false;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "transition 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(field) => match field.as_str() {
                    "from" => {
                        if from.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'from' 重复出现\n  建议：删除多余 from 字段",
                            ));
                        }
                        let (s, _) = self.expect_str("from 的场景名")?;
                        from = Some((s, t.line));
                    }
                    "to" => {
                        if to.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'to' 重复出现\n  建议：删除多余 to 字段",
                            ));
                        }
                        let (s, _) = self.expect_str("to 的场景名")?;
                        to = Some((s, t.line));
                    }
                    "duration" => {
                        if duration.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'duration' 重复出现\n  建议：删除多余 duration 字段",
                            ));
                        }
                        let tv: TimeVal = self.expect_time("transition.duration")?;
                        if !tv.value.is_finite() {
                            return Err(err_duration_not_finite(
                                t.line,
                                "transition.duration",
                                tv.value,
                            ));
                        }
                        if tv.value <= 0.0 {
                            return Err(err_at(
                                t.line,
                                "[P1011] transition.duration 应为正时长（大于 0）\n  建议：给出正时长，如 0.5s",
                            ));
                        }
                        duration = Some(tv);
                    }
                    "kind" => {
                        if kind_seen {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'kind' 重复出现\n  建议：删除多余 kind 字段",
                            ));
                        }
                        let (k, _) = self.expect_ident("transition 种类")?;
                        if !matches!(k.as_str(), "crossfade" | "cut" | "black" | "white") {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "[P2106] 未知的 transition 种类 '{}'\n  建议：当前支持 crossfade（叠化）/ cut（硬切）/ black（黑场）/ white（白场）",
                                    k
                                ),
                            ));
                        }
                        kind = k;
                        kind_seen = true;
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "[P1002] transition 块中未知字段 '{}'（允许：from / to / duration / kind）\n  建议：检查字段名拼写",
                                other
                            ),
                        ));
                    }
                },
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] transition 块中期望字段名或 '}}'，却遇到 {}\n  建议：字段形如 from \"开场\" / to \"第二场\" / duration 0.5s / kind crossfade",
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        // from / to 必填（过渡的两端缺失会让时间线计算失去语义）。
        if from.is_none() || to.is_none() {
            let missing: &str = if from.is_none() { "from" } else { "to" };
            return Err(err_at(
                kw_line,
                format!(
                    "[P1003] transition '{}' 缺少必填字段 '{}'\n  建议：补全 from \"场景A\" 与 to \"场景B\"（场景名需与某个 scene 块的名字一致）",
                    name, missing
                ),
            ));
        }
        Ok(TransitionAst { name, from, to, duration, kind, line: kw_line })
    }

    /// anim 块：`anim "目标路径" { key 时间 值 [easing] ... loop 模式? }`。
    /// easing 名映射与 loop 语义校验在 codegen（依赖 prism-anim）。
    fn parse_anim(&mut self, kw_line: usize) -> Result<AnimAst, DslError> {
        let (target, _) = self.expect_str("anim 的目标路径")?;
        self.expect_punct(Tok::LBrace, "{")?;
        let mut keys: Vec<KeyAst> = Vec::new();
        let mut loop_mode: Option<String> = None;
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => return Err(err_at(self.eof_line(), "anim 块缺少右花括号 '}'")),
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(kw) if kw == "key" => {
                    let time: TimeVal = self.expect_time("关键帧时间")?;
                    let (value, _) = self.expect_num("关键帧数值")?;
                    // easing 可省略：下一个 ident 若不是 'key' / 'loop' 则视为 easing 名。
                    let easing: String = match self.peek() {
                        Some(Token { tok: Tok::Ident(s), .. }) if s != "key" && s != "loop" => {
                            let s = s.clone();
                            self.bump();
                            s
                        }
                        _ => "linear".to_string(),
                    };
                    keys.push(KeyAst { time, value, easing, line: t.line });
                }
                Tok::Ident(kw) if kw == "loop" => {
                    if loop_mode.is_some() {
                        return Err(err_at(
                            t.line,
                            "[P1001] anim 块中字段 'loop' 重复出现\n  建议：删除多余 loop 字段",
                        ));
                    }
                    let (m, _) = self.expect_ident("loop 模式")?;
                    if !matches!(m.as_str(), "direct" | "loop" | "pingpong") {
                        return Err(err_at(
                            t.line,
                            format!(
                                "[P2104] 未知的 loop 模式 '{}'\n  建议：direct（直接截断）/ loop（循环）/ pingpong（往复）三选一",
                                m
                            ),
                        ));
                    }
                    loop_mode = Some(m);
                }
                Tok::Ident(other) => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1002] anim 块中未知条目 '{}'（允许：key / loop）\n  建议：关键帧写法为 key 0s 0 linear（时间 / 值 / 可选 easing）",
                            other
                        ),
                    ));
                }
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] anim 块中期望 'key' / 'loop' 或 '}}'，却遇到 {}\n  建议：anim 块由 key 条目与可选 loop 字段组成",
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        if keys.is_empty() {
            return Err(err_at(
                kw_line,
                "[P1003] anim 块为空（至少一个 key 条目）\n  建议：添加 key 0s 0 linear",
            ));
        }
        Ok(AnimAst { target, keys, loop_mode, line: kw_line })
    }

    /// expr 块：`expr "名" "t*10 + p*0.5"`（受限算术表达式；表达式语法校验在 codegen）。
    fn parse_expr_def(&mut self, kw_line: usize) -> Result<ExprDefAst, DslError> {
        let (name, _) = self.expect_str("expr 名称")?;
        let (body, _) = self.expect_str("expr 表达式（如 \"t*10\"）")?;
        Ok(ExprDefAst { name, body, line: kw_line })
    }

    /// import 声明：`import packs "名" { engine ">=0.4, <0.5" ... }`（块可省略）。
    fn parse_import(&mut self, kw_line: usize) -> Result<ImportPacksAst, DslError> {
        let (kw, packs_line) = self.expect_ident("'import' 之后的 'packs'")?;
        if kw != "packs" {
            return Err(err_at(
                packs_line,
                format!(
                    "[P1002] import 之后期望 'packs'，却遇到 '{}'\n  建议：写成 import packs \"包名\"",
                    kw
                ),
            ));
        }
        let (name, _) = self.expect_str("包名")?;
        let fields: Vec<NamedField> = match self.peek() {
            Some(Token { tok: Tok::LBrace, .. }) => {
                self.bump();
                self.parse_named_fields("import packs")?
            }
            _ => Vec::new(),
        };
        Ok(ImportPacksAst { name, fields, line: kw_line })
    }

    /// font 指令：`font "首选.ttf" fallback "次级.ttf", "末级.ttf"`（V1.5-B）。
    ///
    /// 语法要点：
    /// - 首选路径必填（双引号字符串），`fallback` 子句可省略；
    /// - `fallback` 后至少一个路径，多个路径用**逗号**分隔（尾逗号显式报错）；
    /// - 声明顺序即回退顺序（首选在前），多条 font 指令串联成一条完整链。
    ///
    /// 装载（读文件 / 解析 / 缺失码位诊断）不在 parser 做——本层只负责
    /// 语法与路径非空，文件系统错误由 `prism-text` 的 `FontSet` 报 T008。
    fn parse_font(&mut self, kw_line: usize) -> Result<FontDecl, DslError> {
        let (path, path_line) = self.expect_str("font 首选字体文件路径")?;
        if path.trim().is_empty() {
            return Err(err_at(
                path_line,
                "[P4004] font 指令的首选字体路径不能为空字符串\n  建议：写成 font \"assets/MyFont.ttf\" fallback \"assets/Fallback.ttf\"；路径为运行时工作目录相对路径或绝对路径",
            ));
        }
        let mut fallbacks: Vec<String> = Vec::new();
        // 可选 fallback 子句：紧跟字符串之后、且不是 'fallback' 关键字则此链只有一项。
        let has_fallback: bool = matches!(
            self.peek(),
            Some(Token { tok: Tok::Ident(k), .. }) if k == "fallback"
        );
        if has_fallback {
            self.bump();
            loop {
                let (fb, fb_line) = self.expect_str("fallback 字体文件路径")?;
                if fb.trim().is_empty() {
                    return Err(err_at(
                        fb_line,
                        "[P4004] fallback 字体路径不能为空字符串\n  建议：删除空字符串，只保留实际存在的字体文件路径",
                    ));
                }
                fallbacks.push(fb);
                // 逗号 = 列表继续；无逗号 = 列表结束。
                let next: Option<Tok> = self.peek().map(|t| t.tok.clone());
                if matches!(next, Some(Tok::Comma)) {
                    self.bump();
                    if !matches!(self.peek(), Some(Token { tok: Tok::Str(_), .. })) {
                        let line = self.peek().map_or(self.eof_line(), |t| t.line);
                        return Err(err_at(
                            line,
                            "[P1002] fallback 列表的逗号之后缺少字体路径\n  建议：写成 fallback \"assets/B.ttf\", \"assets/C.ttf\"（逗号后必须紧跟双引号字符串，尾逗号不允许）",
                        ));
                    }
                    continue;
                }
                if matches!(next, Some(Tok::Str(_))) {
                    let line = self.peek().map_or(self.eof_line(), |t| t.line);
                    return Err(err_at(
                        line,
                        "[P1002] fallback 列表中相邻字体路径之间缺少逗号\n  建议：写成 fallback \"assets/B.ttf\", \"assets/C.ttf\"（多个回退字体以逗号分隔）",
                    ));
                }
                break;
            }
        }
        Ok(FontDecl { path, fallbacks, line: kw_line })
    }

    /// 通用字段循环（开放参数集块）：`(名 值)*`，遇 '}' 结束。
    /// 未知字段**不在 parser 拦截**——白名单校验在 codegen（按块种类给出针对性建议）。
    fn parse_named_fields(&mut self, block: &str) -> Result<Vec<NamedField>, DslError> {
        let mut fields: Vec<NamedField> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => {
                    return Err(err_at(
                        self.eof_line(),
                        format!("{} 块缺少右花括号 '}}'", block),
                    ));
                }
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(name) => {
                    let ctx: String = format!("{} 块中字段 '{}'", block, name);
                    // 裸标志：仅对**白名单内的布尔标志字段**（如 note 块的
                    // arpeggio）生效——`名` 后紧跟 `}`（无值）→ value = None，
                    // 由 codegen 解释为布尔真（缺陷 3）。
                    // 其余普通字段（content / rotation / kind / radius /
                    // duration / size / color / x / y / opacity …）遇 `}`
                    // 仍落入 parse_field_value 报 P2107（缺值），恢复 v2_guard。
                    if is_bare_flag_field(&name)
                        && matches!(self.peek().map(|x| &x.tok), Some(Tok::RBrace))
                    {
                        fields.push(NamedField { name, value: None, line: t.line });
                        continue;
                    }
                    let value: FieldValue = self.parse_field_value(&ctx)?;
                    fields.push(NamedField { name, value: Some(value), line: t.line });
                }
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] {} 块中期望字段名或 '}}'，却遇到 {}\n  建议：字段写法为 `名 值`（如 radius 1 / kind sphere / duration 3s）",
                            block,
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        Ok(fields)
    }

    /// 字段值：数字 / 字符串 / 颜色 / 单位字面量 / 标识符 / vec3 / 关键帧块 / 子块。
    fn parse_field_value(&mut self, ctx: &str) -> Result<FieldValue, DslError> {
        let t: &Token = match self.peek() {
            Some(t) => t,
            None => {
                return Err(err_at(
                    self.eof_line(),
                    format!(
                        "[P2107] {} 缺少值，但文件已结束\n  建议：给出字段值（如 radius 1 / kind sphere / duration 3s）",
                        ctx
                    ),
                ));
            }
        };
        match &t.tok {
            Tok::Num(_) => {
                // 前瞻逗号：`数字,数字,数字` → vec3（如 center 1,2,3）。
                if matches!(self.toks.get(self.pos + 1).map(|x| &x.tok), Some(Tok::Comma)) {
                    let v: Vec3 = self.parse_vec3(ctx)?;
                    return Ok(FieldValue::Vec3(v));
                }
                let (v, _) = self.expect_num(ctx)?;
                Ok(FieldValue::Num(v))
            }
            Tok::Str(_) => {
                let (s, _) = self.expect_str(ctx)?;
                Ok(FieldValue::Str(s))
            }
            Tok::Color(r, g, b) => {
                let c = (*r, *g, *b);
                self.bump();
                Ok(FieldValue::Color(c.0, c.1, c.2))
            }
            Tok::Duration(_) | Tok::Millis(_) | Tok::Frames(_) => {
                let tv: TimeVal = self.expect_time(ctx)?;
                Ok(FieldValue::Time(tv))
            }
            Tok::Deg(_) | Tok::Rad(_) => {
                let av: AngleVal = self.expect_angle(ctx)?;
                Ok(FieldValue::Angle(av))
            }
            Tok::Percent(v) => {
                let v = *v;
                self.bump();
                Ok(FieldValue::Percent(v))
            }
            Tok::Px(v) => {
                let v = *v;
                self.bump();
                Ok(FieldValue::Px(v))
            }
            Tok::Ident(_) => {
                // 带名子块前瞻：ident 后紧跟 '{' → `force wind { ... }` 形态。
                if matches!(
                    self.toks.get(self.pos + 1).map(|x| &x.tok),
                    Some(Tok::LBrace)
                ) {
                    let (name, _) = self.expect_ident(ctx)?;
                    self.bump(); // '{'
                    let fields: Vec<NamedField> = self.parse_named_fields(ctx)?;
                    return Ok(FieldValue::NamedBlock { name, fields });
                }
                let (s, _) = self.expect_ident(ctx)?;
                Ok(FieldValue::Ident(s))
            }
            Tok::LParen => {
                // `(vec3)` 括号形式（与 v1 light.direction 同规）。
                self.bump();
                let v: Vec3 = self.parse_vec3(ctx)?;
                self.expect_punct(Tok::RParen, ")")?;
                Ok(FieldValue::Vec3(v))
            }
            Tok::LBrace => {
                // '{' 后第一个 ident 是 'key' → 关键帧序列；否则为任意子块。
                let is_keys: bool = matches!(
                    self.toks.get(self.pos + 1).map(|x| &x.tok),
                    Some(Tok::Ident(s)) if s == "key"
                );
                if is_keys {
                    let keys: Vec<KeyAst> = self.parse_key_list(ctx)?;
                    Ok(FieldValue::Keys(keys))
                } else {
                    self.bump(); // '{'
                    let fields: Vec<NamedField> = self.parse_named_fields(ctx)?;
                    Ok(FieldValue::Block(fields))
                }
            }
            other => {
                let line = t.line;
                let described = describe_tok(other);
                Err(err_at(
                    line,
                    format!(
                        "[P2107] {} 缺少值，却遇到 {}\n  建议：给出字段值（数字 / 字符串 / 颜色 / 单位字面量 / 子块）",
                        ctx, described
                    ),
                ))
            }
        }
    }

    /// 关键帧序列：`{ key 时间 值 [easing] ... }`（easing 可省略，缺省 linear）。
    fn parse_key_list(&mut self, ctx: &str) -> Result<Vec<KeyAst>, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let mut keys: Vec<KeyAst> = Vec::new();
        loop {
            let t: Token = match self.bump() {
                Some(t) => t,
                None => {
                    return Err(err_at(
                        self.eof_line(),
                        format!("{} 的关键帧块缺少右花括号 '}}'", ctx),
                    ));
                }
            };
            match t.tok {
                Tok::RBrace => break,
                Tok::Ident(kw) if kw == "key" => {
                    let time: TimeVal = self.expect_time("关键帧时间")?;
                    let (value, _) = self.expect_num("关键帧数值")?;
                    let easing: String = match self.peek() {
                        Some(Token { tok: Tok::Ident(s), .. }) if s != "key" => {
                            let s = s.clone();
                            self.bump();
                            s
                        }
                        _ => "linear".to_string(),
                    };
                    keys.push(KeyAst { time, value, easing, line: t.line });
                }
                Tok::Ident(other) => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1002] 关键帧块中未知条目 '{}'（允许：key）\n  建议：关键帧写法为 key 0s 0 linear（时间 / 值 / 可选 easing）",
                            other
                        ),
                    ));
                }
                other => {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1001] 关键帧块中期望 'key' 或 '}}'，却遇到 {}\n  建议：关键帧写法为 key 0s 0 linear",
                            describe_tok(&other)
                        ),
                    ));
                }
            }
        }
        if keys.is_empty() {
            return Err(err_at(
                self.eof_line(),
                format!(
                    "[P1003] {} 的关键帧块为空（至少一个 key）\n  建议：添加 key 0s 0 linear",
                    ctx
                ),
            ));
        }
        Ok(keys)
    }
}

/// 字段是否为「裸标志（无值 = true）」型字段白名单。
///
/// 仅这些字段允许 `名` 后紧跟 `}` 的无值写法，由 codegen 解释为布尔真；
/// 其余字段（content / rotation / kind / radius / duration / size / color
/// / x / y / opacity …）遇 `}` 仍应报 P2107（缺值）。
///
/// 白名单依据 codegen.rs 的值处理分支：目前唯一明确按「无值 = true」
/// 解释的字段为 note 块的 `arpeggio`（`f.value == None => true`）。
fn is_bare_flag_field(name: &str) -> bool {
    matches!(name, "arpeggio")
}
