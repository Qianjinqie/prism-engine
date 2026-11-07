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
    AnimAst, AudioAst, CameraV2Ast, CapAst, ComposeAst, ComposeOpAst, ComposeRef, DocumentAst, DocumentV2Ast,
    ExprDefAst, FieldValue, FinishAst, FontDecl, FormulaAst, ImportPacksAst, KeyAst, LayerAst, LightV2Ast,
    MaterialV2Ast,
    MorphRefAst, MovieAst, NamedField, NoteAst, OutputAst, PostAst, SceneItem, SceneV2Ast, SdfAst,
    SimAst,
    GeoAst, Stage3dAst, TimeVal, TimeUnit, TransitionAst, AestheticAst, AngleVal, AngleUnit,
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
        let mut anisotropy: Option<f32> = None;
        let mut anisotropy_rotation: Option<f32> = None;
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
                    // V2.2-B-3（T05）：各向异性字段（与 v2 白名单对齐）。
                    "anisotropy" => {
                        if anisotropy.is_some() {
                            return Err(err_at(t.line, "material 块中字段 'anisotropy' 重复出现"));
                        }
                        let v = self.expect_num("anisotropy 的数字值（0–1）")?.0;
                        if !(0.0..=1.0).contains(&v) {
                            return Err(err_at(
                                t.line,
                                format!("material 块中字段 'anisotropy' 应在 [0,1]，却得到 {v}"),
                            ));
                        }
                        anisotropy = Some(v);
                    }
                    "anisotropy_rotation" => {
                        if anisotropy_rotation.is_some() {
                            return Err(err_at(t.line, "material 块中字段 'anisotropy_rotation' 重复出现"));
                        }
                        anisotropy_rotation = Some(self.expect_num("anisotropy_rotation 的数字值（弧度）")?.0);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "material 块中未知字段 '{}'（允许：albedo / roughness / metalness / anisotropy / anisotropy_rotation）",
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
        Ok(MaterialAst { albedo, roughness, metalness, anisotropy, anisotropy_rotation })
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
                            "[P1001] 顶层期望块定义（movie / font / scene / transition 等），却遇到 {}\n  建议：顶层只能出现 movie / timeline / theme / import packs / font / scene / transition / anim / expr / formula / output 之一",
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
                "formula" => doc.formulas.push(self.parse_formula(line)?),
                // V3.0 W2：顶层能力接线声明（39 行规格表的运行时入口）。
                "cap" => doc.caps.push(self.parse_cap(line)?),
                // V3.0 W3：顶层一等观感档位（只承载字段，校验在 finish 模块）。
                "finish" => {
                    if doc.finish.is_some() {
                        return Err(err_at(
                            line,
                            "[P9005] 'finish' 块重复出现（一份文档只声明一次观感档位）\n  建议：合并两个 finish 块的字段到一处",
                        ));
                    }
                    doc.finish = Some(self.parse_finish(line)?);
                }
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
                            "[P1002] 未知顶层块 '{}'（允许：movie / timeline / theme / import packs / font / scene / transition / anim / expr / formula / cap / finish / output）\n  建议：检查块名拼写；audio 与 post 应写在 scene 块内；cap 能力块可写在顶层或 scene 内；finish 是 // prism-dsl 2 文档降级出来的一等观感档位",
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
                // T0：附加能力块（几何 / 介质 / 纹理 / 毛发 / 实例）。
                "media" | "mesh" | "texture" | "hair" | "instance" => {
                    items.push(SceneItem::Geo(self.parse_geo(kw, line)?))
                }
                // V3.0 W2：场景内能力接线（按节点 / 按帧的声明入口）。
                "cap" => items.push(SceneItem::Cap(self.parse_cap(line)?)),
                other => {
                    return Err(err_at(
                        line,
                        format!(
                            "[P1002] scene 块中未知条目 '{}'（允许：stage3d / layer / sim / compose / audio / note / post / media / mesh / texture / hair / instance / cap）\n  建议：检查条目名拼写",
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
        // v1.5-C：可选前置 `morph from "路径A" to "路径B"` 子句（图层几何变形）。
        // 只允许出现在字段列表开头，其后仍是 `名 值` 字段流（不动通用字段解析）。
        let morph: Option<MorphRefAst> = self.parse_layer_morph_prefix()?;
        let fields = self.parse_named_fields("layer")?;
        Ok(LayerAst { kind, name, fields, morph, line: kw_line })
    }

    /// layer 块里的可选 morph 声明前缀：`morph from <路径> to <路径>`。
    ///
    /// `morph` 是保留字段名（v1.5-C），故普通字段不可再叫 `morph`；路径既可
    /// 写成字符串（`"hero_a"`）也可写成标识符（`hero_a`）。
    fn parse_layer_morph_prefix(&mut self) -> Result<Option<MorphRefAst>, DslError> {
        let is_morph: bool = matches!(self.peek().map(|t| &t.tok), Some(Tok::Ident(k)) if k == "morph");
        if !is_morph {
            return Ok(None);
        }
        let line: usize = self.peek().map(|t| t.line).unwrap_or(0);
        self.bump(); // 消费 `morph`
        let (kw, kw_line) = self.expect_ident("morph 的 from")?;
        if kw != "from" {
            return Err(err_at(
                kw_line,
                format!(
                    "[P1001] layer morph 声明期望 'from'，却遇到 '{}'\n  建议：写法为 morph from \"路径A\" to \"路径B\"",
                    kw
                ),
            ));
        }
        let from: String = self.expect_path_ref("morph.from 的路径引用")?;
        let (kw2, kw2_line) = self.expect_ident("morph 的 to")?;
        if kw2 != "to" {
            return Err(err_at(
                kw2_line,
                format!(
                    "[P1001] layer morph 声明期望 'to'，却遇到 '{}'\n  建议：写法为 morph from \"路径A\" to \"路径B\"",
                    kw2
                ),
            ));
        }
        let to: String = self.expect_path_ref("morph.to 的路径引用")?;
        Ok(Some(MorphRefAst { from, to, line }))
    }

    /// 路径引用：字符串或裸标识符（morph from/to 用）。
    fn expect_path_ref(&mut self, what: &str) -> Result<String, DslError> {
        let line: usize = self.peek().map(|t| t.line).unwrap_or_else(|| self.eof_line());
        match self.peek().map(|t| t.tok.clone()) {
            Some(Tok::Str(_)) => Ok(self.expect_str(what)?.0),
            Some(Tok::Ident(_)) => Ok(self.expect_ident(what)?.0),
            Some(other) => Err(err_at(
                line,
                format!(
                    "[P1003] 期望{}（字符串或标识符），却遇到 {}\n  建议：morph from \"路径A\" to \"路径B\"",
                    what,
                    describe_tok(&other)
                ),
            )),
            None => Err(err_at(
                line,
                format!("[P1003] 期望{}（字符串或标识符），却遇到文件结束", what),
            )),
        }
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

    /// T0 附加能力块：`media|mesh|texture|hair|instance 名 { 字段* }`。
    ///
    /// 名可省略（`media { ... }`）⇒ 缺省同种类名；引用型字段
    /// （`instance.geom`、`displacement.texture`）按名解析，见
    /// `codegen::build_geo`。
    fn parse_geo(&mut self, kind: String, kw_line: usize) -> Result<GeoAst, DslError> {
        let name: String = self.parse_optional_block_name(&format!("{kind} 名称"))?;
        let name: String = if name.is_empty() { kind.clone() } else { name };
        self.expect_punct(Tok::LBrace, "{")?;
        let fields: Vec<NamedField> = self.parse_named_fields(&kind)?;
        Ok(GeoAst { kind, name, fields, line: kw_line })
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
        // v1.5-C：引擎级转场参数（fade / wipe / dissolve / push）。
        let mut band: Option<u16> = None;
        let mut seed: Option<u64> = None;
        let mut dir: Option<String> = None;
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
                        if !matches!(
                            k.as_str(),
                            "crossfade"
                                | "cut"
                                | "black"
                                | "white"
                                | "fade"
                                | "wipe"
                                | "dissolve"
                                | "push"
                        ) {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "[P2106] 未知的 transition 种类 '{}'\n  建议：当前支持 crossfade（叠化）/ cut（硬切）/ black（黑场）/ white（白场）/ fade（交叉溶解）/ wipe（擦除）/ dissolve（像素溶解）/ push（推移）",
                                    k
                                ),
                            ));
                        }
                        kind = k;
                        kind_seen = true;
                    }
                    // v1.5-C：引擎级转场参数（ADR-18 四转场统一为凸权重）。
                    "band" => {
                        if band.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'band' 重复出现\n  建议：删除多余 band 字段",
                            ));
                        }
                        let (v, _): (f32, usize) = self.expect_num("transition.band")?;
                        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > 65535.0 {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "[P1011] transition.band 应为 0..65535 的整数像素（擦除过渡带半宽），却得到 {v}\n  建议：band 2（缺省 2px）或 band 0（硬边）"
                                ),
                            ));
                        }
                        band = Some(v as u16);
                    }
                    "seed" => {
                        if seed.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'seed' 重复出现\n  建议：删除多余 seed 字段",
                            ));
                        }
                        let (v, _): (f32, usize) = self.expect_num("transition.seed")?;
                        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > 9.007_199e15 {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "[P1011] transition.seed 应为非负整数（溶解显影种子），却得到 {v}\n  建议：seed 7（同 seed 同显影图案）"
                                ),
                            ));
                        }
                        seed = Some(v as u64);
                    }
                    "dir" => {
                        if dir.is_some() {
                            return Err(err_at(
                                t.line,
                                "[P1001] transition 块中字段 'dir' 重复出现\n  建议：删除多余 dir 字段",
                            ));
                        }
                        let (d, _) = self.expect_ident("transition 方向")?;
                        if !matches!(d.as_str(), "left" | "right" | "up" | "down") {
                            return Err(err_at(
                                t.line,
                                format!(
                                    "[P2106] 未知的 transition 方向 '{}'\n  建议：dir 取 left / right / up / down（描述新场景的入场侧）",
                                    d
                                ),
                            ));
                        }
                        dir = Some(d);
                    }
                    other => {
                        return Err(err_at(
                            t.line,
                            format!(
                                "[P1002] transition 块中未知字段 '{}'（允许：from / to / duration / kind / band / seed / dir）\n  建议：检查字段名拼写",
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
        Ok(TransitionAst { name, from, to, duration, kind, band, seed, dir, line: kw_line })
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

    /// formula 块（V2-j）：`formula "名" { notation "..." size? x? y? color? align? }`。
    ///
    /// 语法要点：
    /// - 公式名必填（双引号字符串），同一文档内重名报 P1001（codegen 校验）；
    /// - 块内字段为**开放参数集**（[`NamedField`]），`notation` 即公式记法原文，
    ///   codegen 按白名单（notation / size / x / y / color / align）校验；
    /// - 记法的结构校验（分式是否配平、矩阵是否矩形）不在 parser 做——本层只
    ///   负责语法与「notation 必填」，语义换算由 `prism-text::math_ast` 负责。
    fn parse_formula(&mut self, kw_line: usize) -> Result<FormulaAst, DslError> {
        let (name, name_line) = self.expect_str("formula 名称")?;
        if name.trim().is_empty() {
            return Err(err_at(
                name_line,
                "[P1022] formula 名称不得为空\n  建议：写成 formula \"隐函数求导\" { notation \"...\" }",
            ));
        }
        self.expect_punct(Tok::LBrace, "{")?;
        let fields: Vec<NamedField> = self.parse_named_fields("formula")?;
        // notation 从通用字段列表中提取（其余字段交给 codegen 白名单）。
        let mut notation: Option<String> = None;
        let mut rest: Vec<NamedField> = Vec::with_capacity(fields.len());
        for f in fields {
            if f.name == "notation" {
                match &f.value {
                    Some(FieldValue::Str(s)) => notation = Some(s.clone()),
                    _ => {
                        return Err(err_at(
                            f.line,
                            "[P2107] formula 块中字段 'notation' 需要字符串值（公式记法）\n  建议：写成 notation \"\\\\frac{dy}{dx} = 2x\"",
                        ));
                    }
                }
            } else {
                rest.push(f);
            }
        }
        Ok(FormulaAst { name, notation, fields: rest, line: kw_line })
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

    /// font 指令：`font "首选.ttf" fallback "次级.ttf", "末级.ttf"`。
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
    /// V3.0 W2：`cap "编号" { 名 值 ... }`。
    ///
    /// 语法层只做「编号 + 通用字段流」的读取，字段全集 / 类型 / 值域 /
    /// 时序绑定全部交给 `wiring::build_wiring` 校验（P8xxx 三段式 + 行号），
    /// 解析器不发明任何能力语义（红线 1）。
    fn parse_cap(&mut self, kw_line: usize) -> Result<CapAst, DslError> {
        let (row_id, _) = self.expect_str("cap 能力编号")?;
        if row_id.is_empty() {
            return Err(err_at(
                kw_line,
                "[P8005] cap 块的能力编号为空\n  建议：写成 cap \"R-1\" { ... }，编号取规格表 39 行之一",
            ));
        }
        self.expect_punct(Tok::LBrace, "{")?;
        let fields: Vec<NamedField> = self.parse_named_fields("cap")?;
        Ok(CapAst { row_id, fields, line: kw_line, scene_index: None })
    }

    /// V3.0 W3：`finish { ... }` 顶层块（扁平 `名 值` 字段流）。
    ///
    /// 语法糖层只做 token 搬运；`mode` / `seed` / 六族 enabled|strength /
    /// rubric_* 的全集必填、值域、全 0、贴皮与真随机门禁一律在
    /// `finish::compile_v2_finish` 里三段式校验（红线 3 不取默认）。
    fn parse_finish(&mut self, kw_line: usize) -> Result<FinishAst, DslError> {
        self.expect_punct(Tok::LBrace, "{")?;
        let fields: Vec<NamedField> = self.parse_named_fields("finish")?;
        Ok(FinishAst { line: kw_line, fields })
    }

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

// ---------------------------------------------------------------------------
// V3.0 W1：v3 前端解析器（D-22 自研 LL(1) + Pratt）
//
// 语句层是 LL(1)：每个语句的首 token 唯一决定产生式；表达式层是 Pratt
// （绑定力表见 `infix_bp`）。与 v1/v2 的 `Parser` 完全分离，旧解析器一行未改。
// 冻结依据：`docs/v3.0-语言规格.md` §2 / §4。
// ---------------------------------------------------------------------------

/// V3.0 语言前端解析器与行内表达式解析入口。
pub mod v3 {
    use crate::ast::v3::{
        BinOp3, Expr3, ForRange3, InterpPart3, Literal3, Program3, Stmt3, UnOp3,
    };
    use crate::lexer::v3::{describe, tokenize_inline3, Tok3, Token3};
    use crate::{v3_err, DslError};

    /// LL(1) 语句 + Pratt 表达式解析器。
    pub struct Parser3 {
        toks: Vec<Token3>,
        pos: usize,
    }

    impl Parser3 {
        pub fn new(toks: Vec<Token3>) -> Self {
            Self { toks, pos: 0 }
        }

        fn peek(&self) -> Option<&Tok3> {
            self.toks.get(self.pos).map(|t| &t.tok)
        }

        fn peek_line(&self) -> usize {
            self.toks.get(self.pos).map_or_else(
                || self.toks.last().map_or(1, |t| t.line),
                |t| t.line,
            )
        }

        fn bump(&mut self) -> Option<Token3> {
            let t: Option<Token3> = self.toks.get(self.pos).cloned();
            if t.is_some() {
                self.pos += 1;
            }
            t
        }

        fn at_ident(&self, word: &str) -> bool {
            matches!(self.peek(), Some(Tok3::Ident(s)) if s == word)
        }

        fn take_ident(&mut self) -> Result<(String, usize), DslError> {
            let line: usize = self.peek_line();
            match self.bump().map(|t| t.tok) {
                Some(Tok3::Ident(s)) => Ok((s, line)),
                Some(got) => Err(v3_err(
                    line,
                    "P6004",
                    &format!("期望标识符，却遇到 {}", describe(&got)),
                    "检查语句关键字与变量名的拼写",
                )),
                None => Err(v3_err(
                    line,
                    "P6004",
                    "期望标识符，却到达文件末尾",
                    "补全该语句",
                )),
            }
        }

        fn expect_tok(&mut self, want: Tok3, what: &str) -> Result<(), DslError> {
            let line: usize = self.peek_line();
            match self.peek().cloned() {
                Some(got) if got == want => {
                    self.bump();
                    Ok(())
                }
                Some(got) => Err(v3_err(
                    line,
                    "P6004",
                    &format!("期望 {}，却遇到 {}", what, describe(&got)),
                    "按语法补齐该符号；每行只写一条语句",
                )),
                None => Err(v3_err(
                    line,
                    "P6004",
                    &format!("期望 {}，却到达文件末尾", what),
                    "按语法补齐该符号",
                )),
            }
        }

        fn expect_ident(&mut self, word: &str) -> Result<(), DslError> {
            let line: usize = self.peek_line();
            if self.at_ident(word) {
                self.bump();
                Ok(())
            } else {
                Err(v3_err(
                    line,
                    "P6004",
                    &format!("期望关键字 '{}'", word),
                    "检查关键字拼写与语句结构",
                ))
            }
        }

        fn expect_newline(&mut self) -> Result<(), DslError> {
            self.expect_tok(Tok3::Newline, "行尾")
        }

        /// 解析整份程序。
        pub fn parse_program(&mut self) -> Result<Program3, DslError> {
            let mut stmts: Vec<Stmt3> = Vec::new();
            while self.pos < self.toks.len() {
                if self.peek() == Some(&Tok3::Newline) {
                    self.bump();
                    continue;
                }
                stmts.push(self.parse_stmt()?);
            }
            let line_count: usize = self.toks.last().map_or(1, |t| t.line);
            Ok(Program3 { stmts, line_count })
        }

        fn parse_stmt(&mut self) -> Result<Stmt3, DslError> {
            if self.at_ident("seed") {
                return self.parse_seed();
            }
            if self.at_ident("def") {
                return self.parse_def();
            }
            if self.at_ident("for") {
                return self.parse_for();
            }
            if self.at_ident("if") {
                return self.parse_if();
            }
            if self.at_ident("elif") || self.at_ident("else") {
                let line: usize = self.peek_line();
                return Err(v3_err(
                    line,
                    "P6108",
                    "'elif' / 'else' 没有配对的 'if'",
                    "'elif' 必须紧跟在同级 'if' 分支之后，'else' 必须是最后一个分支",
                ));
            }
            if self.at_ident("emit") {
                return self.parse_emit();
            }
            if self.at_ident("return") {
                return self.parse_return();
            }
            if matches!(self.peek(), Some(Tok3::Ident(_)))
                && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok3::Assign))
            {
                return self.parse_assign();
            }
            // 裸表达式语句：让 `now()` / `write_file(...)` 这类调用进入
            // 统一的静态拒绝面（P6101），而不是在语句层被含糊地吃掉。
            if self.can_start_expr() {
                let line: usize = self.peek_line();
                let expr: Expr3 = self.parse_expr(0)?;
                self.expect_newline()?;
                return Ok(Stmt3::Expr { expr, line });
            }
            let line: usize = self.peek_line();
            let got: String = self
                .peek()
                .map(describe)
                .unwrap_or_else(|| "文件末尾".to_string());
            Err(v3_err(
                line,
                "P6004",
                &format!("无法识别的语句开头 {}", got),
                "顶层只允许 seed / 赋值 / def / for / if / emit；块内另允许 return",
            ))
        }

        fn can_start_expr(&self) -> bool {
            matches!(
                self.peek(),
                Some(Tok3::Ident(_))
                    | Some(Tok3::Num(_))
                    | Some(Tok3::Str(_))
                    | Some(Tok3::Interp { .. })
                    | Some(Tok3::LParen)
                    | Some(Tok3::LBracket)
                    | Some(Tok3::LBrace)
                    | Some(Tok3::Minus)
            )
        }

        fn parse_seed(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'seed'
            self.expect_tok(Tok3::Assign, "'='")?;
            let value: String = match self.bump().map(|t| t.tok) {
                Some(Tok3::Num(s)) => s,
                Some(got) => {
                    return Err(v3_err(
                        line,
                        "P6112",
                        &format!("seed 必须是非负整数字面量，却遇到 {}", describe(&got)),
                        "写成 seed = 20261106 这样的十进制整数",
                    ));
                }
                None => {
                    return Err(v3_err(
                        line,
                        "P6112",
                        "seed 缺少整数字面量",
                        "写成 seed = 20261106 这样的十进制整数",
                    ));
                }
            };
            self.expect_newline()?;
            Ok(Stmt3::Seed { value, line })
        }

        fn parse_assign(&mut self) -> Result<Stmt3, DslError> {
            let (name, line): (String, usize) = self.take_ident()?;
            self.expect_tok(Tok3::Assign, "'='")?;
            let expr: Expr3 = self.parse_expr(0)?;
            self.expect_newline()?;
            Ok(Stmt3::Assign { name, expr, line })
        }

        fn parse_return(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'return'
            let expr: Expr3 = self.parse_expr(0)?;
            self.expect_newline()?;
            Ok(Stmt3::Return { expr, line })
        }

        fn parse_emit(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'emit'
            let template: Expr3 = self.parse_expr(0)?;
            self.expect_newline()?;
            Ok(Stmt3::Emit { template, line })
        }

        /// 缩进块：`:` + 换行 + INDENT + 语句* + DEDENT。
        fn parse_suite(&mut self) -> Result<Vec<Stmt3>, DslError> {
            self.expect_tok(Tok3::Colon, "':'")?;
            self.expect_newline()?;
            self.expect_tok(Tok3::Indent, "缩进块（4 空格）")?;
            let mut body: Vec<Stmt3> = Vec::new();
            loop {
                match self.peek() {
                    Some(Tok3::Dedent) => {
                        self.bump();
                        break;
                    }
                    None => {
                        let line: usize = self.peek_line();
                        return Err(v3_err(
                            line,
                            "P6001",
                            "缩进块未闭合就到达文件末尾",
                            "所有块内语句保持同一级 4 空格缩进",
                        ));
                    }
                    _ => body.push(self.parse_stmt()?),
                }
            }
            if body.is_empty() {
                let line: usize = self.peek_line();
                return Err(v3_err(
                    line,
                    "P6001",
                    "缩进块内没有任何语句",
                    "至少写一条语句；空块请删掉该分支",
                ));
            }
            Ok(body)
        }

        fn parse_def(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'def'
            let (name, _): (String, usize) = self.take_ident()?;
            self.expect_tok(Tok3::LParen, "'('")?;
            let mut params: Vec<String> = Vec::new();
            if self.peek() != Some(&Tok3::RParen) {
                loop {
                    let (p, pline): (String, usize) = self.take_ident()?;
                    if params.contains(&p) {
                        return Err(v3_err(
                            pline,
                            "P6103",
                            &format!("def '{}' 的参数 '{}' 重名", name, p),
                            "参数名在同一函数的形参表里必须唯一",
                        ));
                    }
                    params.push(p);
                    if self.peek() == Some(&Tok3::Comma) {
                        self.bump();
                        continue;
                    }
                    break;
                }
            }
            self.expect_tok(Tok3::RParen, "')'")?;
            let body: Vec<Stmt3> = self.parse_suite()?;
            Ok(Stmt3::Def { name, params, body, line })
        }

        fn parse_for(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'for'
            let (var, _): (String, usize) = self.take_ident()?;
            self.expect_ident("in")?;
            let range: ForRange3 = match self.peek().cloned() {
                Some(Tok3::Num(lo)) => {
                    let rline: usize = self.peek_line();
                    self.bump();
                    self.expect_tok(Tok3::Range, "'..'")?;
                    let hi: String = match self.bump().map(|t| t.tok) {
                        Some(Tok3::Num(s)) => s,
                        Some(got) => {
                            return Err(v3_err(
                                rline,
                                "P6106",
                                &format!("for 区间上界必须是整数字面量，却遇到 {}", describe(&got)),
                                "写成 for i in 0..4: 这样的字面量区间",
                            ));
                        }
                        None => {
                            return Err(v3_err(
                                rline,
                                "P6106",
                                "for 区间缺少上界",
                                "写成 for i in 0..4: 这样的字面量区间",
                            ));
                        }
                    };
                    ForRange3::Range { lo, hi, line: rline }
                }
                Some(Tok3::LBracket) => {
                    let items: Vec<Expr3> = self.parse_list_items()?;
                    ForRange3::List { items, line }
                }
                Some(got) => {
                    return Err(v3_err(
                        line,
                        "P6106",
                        &format!("for 的迭代源必须是字面量区间或字面量列表，却遇到 {}", describe(&got)),
                        "写成 for i in 0..4: 或 for x in [1, 2, 3]:",
                    ));
                }
                None => {
                    return Err(v3_err(
                        line,
                        "P6106",
                        "for 缺少迭代源",
                        "写成 for i in 0..4: 或 for x in [1, 2, 3]:",
                    ));
                }
            };
            let body: Vec<Stmt3> = self.parse_suite()?;
            Ok(Stmt3::For { var, range, body, line })
        }

        fn parse_if(&mut self) -> Result<Stmt3, DslError> {
            let line: usize = self.peek_line();
            self.bump(); // 'if'
            let cond: Expr3 = self.parse_expr(0)?;
            let first: Vec<Stmt3> = self.parse_suite()?;
            let mut branches: Vec<(Expr3, Vec<Stmt3>)> = vec![(cond, first)];
            let mut else_body: Vec<Stmt3> = Vec::new();
            loop {
                if self.at_ident("elif") {
                    self.bump();
                    let c: Expr3 = self.parse_expr(0)?;
                    let b: Vec<Stmt3> = self.parse_suite()?;
                    branches.push((c, b));
                    continue;
                }
                if self.at_ident("else") {
                    self.bump();
                    else_body = self.parse_suite()?;
                }
                break;
            }
            Ok(Stmt3::If { branches, else_body, line })
        }

        /// `[` 已消费前的辅助：消费 `[`、元素、`]`；尾逗号判 P6201。
        fn parse_list_items(&mut self) -> Result<Vec<Expr3>, DslError> {
            let line: usize = self.peek_line();
            self.expect_tok(Tok3::LBracket, "'['")?;
            let mut items: Vec<Expr3> = Vec::new();
            if self.peek() == Some(&Tok3::RBracket) {
                self.bump();
                return Ok(items);
            }
            loop {
                items.push(self.parse_expr(0)?);
                match self.peek() {
                    Some(Tok3::Comma) => {
                        self.bump();
                        if self.peek() == Some(&Tok3::RBracket) {
                            return Err(v3_err(
                                line,
                                "P6201",
                                "列表字面量出现尾逗号",
                                "删掉最后一个逗号，或在后面补一个元素",
                            ));
                        }
                    }
                    Some(Tok3::RBracket) => {
                        self.bump();
                        break;
                    }
                    Some(got) => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            &format!("列表元素后期望 ',' 或 ']'，却遇到 {}", describe(got)),
                            "用逗号分隔元素，用 ']' 收尾",
                        ));
                    }
                    None => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            "列表字面量缺少 ']'",
                            "补上 ']'",
                        ));
                    }
                }
            }
            Ok(items)
        }

        fn parse_dict(&mut self) -> Result<Expr3, DslError> {
            let line: usize = self.peek_line();
            self.expect_tok(Tok3::LBrace, "'{'")?;
            let mut entries: Vec<(String, Expr3)> = Vec::new();
            if self.peek() == Some(&Tok3::RBrace) {
                self.bump();
                return Ok(Expr3::Dict { entries, line });
            }
            loop {
                let key: String = match self.bump().map(|t| t.tok) {
                    Some(Tok3::Str(s)) => s,
                    Some(got) => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            &format!("字典键必须是字符串字面量，却遇到 {}", describe(&got)),
                            "写成 {{\"albedo\": 0.5}} 这样的形式",
                        ));
                    }
                    None => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            "字典缺少键",
                            "写成 {\"key\": value} 的形式",
                        ));
                    }
                };
                if entries.iter().any(|(k, _)| k == &key) {
                    return Err(v3_err(
                        line,
                        "P6202",
                        &format!("字典键 '{}' 重复", key),
                        "同一字典里每个键只能出现一次；删掉重复项",
                    ));
                }
                self.expect_tok(Tok3::Colon, "':'")?;
                let value: Expr3 = self.parse_expr(0)?;
                entries.push((key, value));
                match self.peek() {
                    Some(Tok3::Comma) => {
                        self.bump();
                        if self.peek() == Some(&Tok3::RBrace) {
                            return Err(v3_err(
                                line,
                                "P6201",
                                "字典字面量出现尾逗号",
                                "删掉最后一个逗号，或在后面补一个键值项",
                            ));
                        }
                    }
                    Some(Tok3::RBrace) => {
                        self.bump();
                        break;
                    }
                    Some(got) => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            &format!("字典项后期望 ',' 或 '}}'，却遇到 {}", describe(got)),
                            "用逗号分隔键值项，用 '}' 收尾",
                        ));
                    }
                    None => {
                        return Err(v3_err(
                            line,
                            "P6004",
                            "字典字面量缺少 '}'",
                            "补上 '}'",
                        ));
                    }
                }
            }
            Ok(Expr3::Dict { entries, line })
        }

        /// 插值字符串：单个 `{` / `}` 是字面量（v2 块语法用），`{{expr}}` 才是插值。
        fn parse_interp(
            &mut self,
            text: &str,
            start_line: usize,
            line: usize,
        ) -> Result<Expr3, DslError> {
            let mut parts: Vec<InterpPart3> = Vec::new();
            let mut buf = String::new();
            let mut line_offset: usize = 0;
            let chars: Vec<char> = text.chars().collect();
            let mut i: usize = 0;
            while i < chars.len() {
                let c: char = chars[i];
                if c == '\n' {
                    buf.push(c);
                    line_offset += 1;
                    i += 1;
                    continue;
                }
                if c == '{' && chars.get(i + 1) == Some(&'{') {
                    if !buf.is_empty() {
                        parts.push(InterpPart3::Text(std::mem::take(&mut buf)));
                    }
                    i += 2;
                    let mut frag = String::new();
                    let mut closed = false;
                    while i < chars.len() {
                        if chars[i] == '}' && chars.get(i + 1) == Some(&'}') {
                            i += 2;
                            closed = true;
                            break;
                        }
                        if chars[i] == '\n' {
                            return Err(v3_err(
                                start_line + line_offset,
                                "P6205",
                                "插值表达式不能跨行",
                                "把 {{...}} 写成单行表达式",
                            ));
                        }
                        frag.push(chars[i]);
                        i += 1;
                    }
                    if !closed {
                        return Err(v3_err(
                            start_line + line_offset,
                            "P6205",
                            "插值表达式缺少 '}}' 收尾",
                            "补上 '}}'；v2 块语法的单个 '{' 会原样保留，不需要转义",
                        ));
                    }
                    let expr: Expr3 = parse_expr_text(&frag, start_line + line_offset)?;
                    parts.push(InterpPart3::Expr(expr));
                    continue;
                }
                buf.push(c);
                i += 1;
            }
            if !buf.is_empty() {
                parts.push(InterpPart3::Text(buf));
            }
            Ok(Expr3::Interp { parts, start_line, line })
        }

        /// Pratt 表达式解析（F-1 左→右优先级，绑定力表固定）。
        pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr3, DslError> {
            let mut lhs: Expr3 = self.parse_prefix()?;
            loop {
                // 后缀下标 `[]`（绑定力 11）。
                if self.peek() == Some(&Tok3::LBracket) {
                    if 11 < min_bp {
                        break;
                    }
                    let line: usize = self.peek_line();
                    self.bump();
                    let index: Expr3 = self.parse_expr(0)?;
                    self.expect_tok(Tok3::RBracket, "']'")?;
                    lhs = Expr3::Index {
                        target: Box::new(lhs),
                        index: Box::new(index),
                        line,
                    };
                    continue;
                }
                let (l_bp, r_bp, op): (u8, u8, BinOp3) = match self.infix_bp() {
                    Some(v) => v,
                    None => break,
                };
                if l_bp < min_bp {
                    break;
                }
                let line: usize = self.peek_line();
                self.bump();
                let rhs: Expr3 = self.parse_expr(r_bp)?;
                lhs = Expr3::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    line,
                };
            }
            Ok(lhs)
        }

        fn infix_bp(&self) -> Option<(u8, u8, BinOp3)> {
            let bp: (u8, u8, BinOp3) = match self.peek() {
                Some(Tok3::Ident(s)) if s == "or" => (1, 2, BinOp3::Or),
                Some(Tok3::Ident(s)) if s == "and" => (3, 4, BinOp3::And),
                Some(Tok3::Eq) => (5, 6, BinOp3::Eq),
                Some(Tok3::Ne) => (5, 6, BinOp3::Ne),
                Some(Tok3::Lt) => (5, 6, BinOp3::Lt),
                Some(Tok3::Le) => (5, 6, BinOp3::Le),
                Some(Tok3::Gt) => (5, 6, BinOp3::Gt),
                Some(Tok3::Ge) => (5, 6, BinOp3::Ge),
                Some(Tok3::Plus) => (7, 8, BinOp3::Add),
                Some(Tok3::Minus) => (7, 8, BinOp3::Sub),
                Some(Tok3::Star) => (9, 10, BinOp3::Mul),
                Some(Tok3::Slash) => (9, 10, BinOp3::Div),
                Some(Tok3::Percent) => (9, 10, BinOp3::Mod),
                _ => return None,
            };
            Some(bp)
        }

        fn parse_prefix(&mut self) -> Result<Expr3, DslError> {
            let line: usize = self.peek_line();
            let tok: Tok3 = match self.bump().map(|t| t.tok) {
                Some(t) => t,
                None => {
                    return Err(v3_err(
                        line,
                        "P6003",
                        "表达式不完整：期望字面量 / 变量 / 左括号",
                        "补全表达式的右操作数",
                    ));
                }
            };
            match tok {
                Tok3::Num(s) => Ok(Expr3::Lit { value: Literal3::Num(s), line }),
                Tok3::Str(s) => Ok(Expr3::Lit { value: Literal3::Str(s), line }),
                Tok3::Interp { text, start_line } => {
                    self.parse_interp(&text, start_line, line)
                }
                Tok3::Ident(s) if s == "true" => {
                    Ok(Expr3::Lit { value: Literal3::Bool(true), line })
                }
                Tok3::Ident(s) if s == "false" => {
                    Ok(Expr3::Lit { value: Literal3::Bool(false), line })
                }
                Tok3::Ident(s) if s == "not" => Ok(Expr3::Unary {
                    op: UnOp3::Not,
                    rhs: Box::new(self.parse_expr(12)?),
                    line,
                }),
                Tok3::Ident(s) => {
                    if self.peek() == Some(&Tok3::LParen) {
                        self.bump();
                        let mut args: Vec<Expr3> = Vec::new();
                        if self.peek() != Some(&Tok3::RParen) {
                            loop {
                                args.push(self.parse_expr(0)?);
                                if self.peek() == Some(&Tok3::Comma) {
                                    self.bump();
                                    continue;
                                }
                                break;
                            }
                        }
                        self.expect_tok(Tok3::RParen, "')'")?;
                        Ok(Expr3::Call { name: s, args, line })
                    } else {
                        Ok(Expr3::Var { name: s, line })
                    }
                }
                Tok3::Minus => Ok(Expr3::Unary {
                    op: UnOp3::Neg,
                    rhs: Box::new(self.parse_expr(12)?),
                    line,
                }),
                Tok3::LParen => {
                    let inner: Expr3 = self.parse_expr(0)?;
                    self.expect_tok(Tok3::RParen, "')'")?;
                    Ok(inner)
                }
                Tok3::LBracket => {
                    // `parse_list_items` 自己消费 '['，这里把刚吃掉的 '[' 退回去。
                    self.pos -= 1;
                    let items: Vec<Expr3> = self.parse_list_items()?;
                    Ok(Expr3::List { items, line })
                }
                Tok3::LBrace => {
                    self.pos -= 1;
                    self.parse_dict()
                }
                got => Err(v3_err(
                    line,
                    "P6003",
                    &format!("表达式开头不能是 {}", describe(&got)),
                    "用数字 / 字符串 / 变量名 / 左括号开头",
                )),
            }
        }
    }

    /// 解析插值片段（单行、无缩进 token），供 `parse_interp` 与执行器复用。
    pub fn parse_expr_text(text: &str, line: usize) -> Result<Expr3, DslError> {
        let trimmed: &str = text.trim();
        if trimmed.is_empty() {
            return Err(v3_err(
                line,
                "P6204",
                "插值表达式为空",
                "在 '{{' 与 '}}' 之间写一个表达式，例如 {{radius}}",
            ));
        }
        let toks: Vec<Token3> = tokenize_inline3(trimmed, line)?;
        let mut parser = Parser3::new(toks);
        let expr: Expr3 = parser.parse_expr(0)?;
        if parser.pos != parser.toks.len() {
            let rest: String = parser
                .toks
                .get(parser.pos)
                .map(|t| describe(&t.tok))
                .unwrap_or_else(|| "未知符号".to_string());
            return Err(v3_err(
                line,
                "P6204",
                &format!("插值表达式末尾有多余内容 {}", rest),
                "'{{...}}' 内只写一个完整表达式",
            ));
        }
        Ok(expr)
    }
}
