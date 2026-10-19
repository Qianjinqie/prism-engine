//! `prism-dsl` — Prism 引擎的 DSL 前端（M1 + v2 全块语法）。
//!
//! 把 `.prism` 文本编译为引擎产物：
//! - **v1 入口** [`compile`]：单 scene 文档 → [`prism_core::Scene`] 与输出参数
//!   （M0/M1 语法逐字保留，零回归）；
//! - **v2 入口** [`compile_v2`] / [`compile_v2_with_diagnostics`]：全块语法
//!   （movie / theme / import packs / scene / transition / anim / expr / formula /
//!   layer / sim / stage3d / compose / audio / post / timeline / output）→
//!   prism-core 节点图（`Graph` + `Node.params` + `Node.bindings`）与属性通道
//!   （`PropertyChannel`），深语义（渲染 / 模拟步进 / 音频）留 Phase 2/3。
//!
//! # 编译流水线
//! `source → Lexer（tokenize，手写）→ Parser（递归下降 + 语义检查，产出 AST）
//!        → Codegen（映射到 prism-core / prism-anim）`
//!
//! # 版本头（v2）
//! 文档首行 `// prism-dsl 1` 声明 v2 语法（C11：版本头必须是文档第一行，
//! 见 [`detect_version_head`]）：缺失或不在首行 → W1001 警告（不阻止编译，
//! 语法本身自描述）；版本号非 1 → P1023 硬错（无论出现在哪一行）。
//!
//! # 硬约束
//! - **v1 零损失**：25 个 v1 测试原样保留、全绿；v1 错误消息 / 行号逐字不变
//!   （唯一例外：未被任何 v1 测试覆盖的 `resolution 0x0` 消息升级为带码三段式，
//!   原文语义完整保留在消息正文）。
//! - 任何词法 / 语法 / 语义错误都以 [`DslError`]（带行号、说人话的消息）返回，
//!   绝不 panic；v2 错误按兼容性宪章 C18 三部分格式
//!   `[Pxxxx] 在哪+什么事` + `\n  建议：怎么改`（C19）。
//! - 单位字面量（`3s` / `150ms` / `36f` / `15deg` / `0.5rad` / `50%` / `120px`）
//!   解析期逐字保真，codegen 期按 fps 确定性归一（C17：fov 等豁免清单除外）。
//!
//! # 模块组织
//! - [`lexer`]：token 流（v1 规则逐字 + v2 单位后缀 / 冒号）；
//! - [`ast`]：v1 文档模型 + v2 全块语法模型；
//! - [`parser`]：v1 / v2 两套文档入口，共享词法助手；
//! - [`codegen`]：v1 → [`CompileOutput`]；v2 → [`CompileV2Output`]（节点图 +
//!   通道 + 时间线 + 装包清单）；
//! - [`packs`]：`import packs` 清单模型与 engine 版本范围校验（C12/V9）。

#![forbid(unsafe_code)]

mod ast;
mod codegen;
mod lexer;
mod packs;
mod parser;
// V1.5-D：组合子库（高层语义 → 等价 v2 节点图展开）。公开模块：展开函数经
// `codegen::COMBINATOR_TABLE` 注册面查表，Rust 侧构造入口在 prism-author。
pub mod combinators;

use crate::lexer::{Token, tokenize};
use prism_core::Scene;

// ---------------------------------------------------------------------------
// 公开 API（v1，逐字保留）
// ---------------------------------------------------------------------------

/// 输出格式：`frames`（帧序列）或 `png`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// 原始帧序列（缺省）。
    Frames,
    /// PNG 图片序列。
    Png,
}

/// 一次成功编译的完整产物：场景 + 时间 / 分辨率 / 格式参数。
#[derive(Debug, Clone)]
pub struct CompileOutput {
    pub scene: Scene,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: f32,
    pub format: OutputFormat,
}

/// DSL 编译错误：带行号、面向人类的消息。
#[derive(Debug)]
pub struct DslError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for DslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DSL 第 {} 行: {}", self.line, self.message)
    }
}

impl std::error::Error for DslError {}

/// 构造 [`DslError`] 的便捷函数。
fn err_at(line: usize, message: impl Into<String>) -> DslError {
    DslError { line, message: message.into() }
}

/// 把 `.prism` 源码编译为引擎场景与输出参数（v1 语法）。
///
/// # Errors
/// 词法 / 语法 / 语义任一环节失败都返回 [`DslError`]（含出错行号）。
pub fn compile(source: &str) -> Result<CompileOutput, DslError> {
    let tokens: Vec<Token> = tokenize(source)?;
    let doc: ast::DocumentAst = parser::Parser::new(tokens).parse_document()?;
    Ok(codegen::codegen(&doc))
}

// ---------------------------------------------------------------------------
// 公开 API（v2 新增）
// ---------------------------------------------------------------------------

/// v2 非致命诊断（警告级；致命错误仍以 [`DslError`] 返回）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// 诊断定位行号。
    pub line: usize,
    /// 诊断码（当前：W1001 缺版本头）。
    pub code: String,
    /// 三部分格式的消息正文（含"建议："段，C19）。
    pub message: String,
}

/// 版本头检查结论（见 [`detect_version_head`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionHead {
    /// 文档首行声明了 `// prism-dsl 1`。
    Ok,
    /// 未声明版本头（调用方降级为 W1001 警告）。
    Missing,
    /// 版本头存在但不在文档首行（C11 规定版本头必须是文档第一行）——按缺头
    /// 策略降级为 W1001 警告，携带版本头实际所在行号。
    Misplaced(usize),
}

/// 检查 DSL v2 版本头（防呆基线；C11：版本头必须是文档首行）。
///
/// 版本头形如 `// prism-dsl 1`（`//` 与关键字之间空白任意）。判定规则：
/// - 首行（允许前导空白）为 `// prism-dsl 1` → [`VersionHead::Ok`]；
/// - 首行版本号非 1 → Err(P1023)（硬错：文档声明了未知/未来语法版本）；
/// - 首行不是版本头、但后续行出现 `// prism-dsl ...` 注释：版本号非 1 仍为
///   P1023 硬错（声明了未知版本，不许静默降级为缺头警告）；版本号恰为 1
///   → [`VersionHead::Misplaced`]（未按 C11 声明在首行，调用方降级为
///   W1001 警告引导补到首行）；
/// - 全文没有任何 `// prism-dsl` 注释 → [`VersionHead::Missing`]。
///
/// # Errors
/// 版本号非 1 时返回 P1023 硬错（三段式），无论版本头出现在哪一行。
pub fn detect_version_head(source: &str) -> Result<VersionHead, DslError> {
    // 单行是否为版本头声明；是则返回版本号文本。
    fn head_version(raw: &str) -> Option<&str> {
        let line: &str = raw.trim_start();
        let rest: &str = line.strip_prefix("//")?;
        let version: &str = rest.trim().strip_prefix("prism-dsl")?;
        Some(version.trim())
    }

    let mut lines = source.lines().enumerate();
    // ① 首行判定（C11：版本头必须在文档第一行）。
    if let Some((_, first)) = lines.next() {
        if let Some(version) = head_version(first) {
            if version == "1" {
                return Ok(VersionHead::Ok);
            }
            return Err(p1023_unsupported_version(1, version));
        }
    }
    // ② 首行不是版本头：后续行的版本头 = 错位（v1 版本号）或未知版本（硬错）。
    for (idx, raw) in lines {
        if let Some(version) = head_version(raw) {
            if version != "1" {
                return Err(p1023_unsupported_version(idx + 1, version));
            }
            return Ok(VersionHead::Misplaced(idx + 1));
        }
    }
    Ok(VersionHead::Missing)
}

/// P1023 硬错（三段式）：声明了未知语法版本，无论版本头出现在哪一行。
fn p1023_unsupported_version(line: usize, version: &str) -> DslError {
    err_at(
        line,
        format!(
            "[P1023] 版本头声明了未知语法版本 '// prism-dsl {version}'（本编译器只支持版本 1）\n  建议：把版本头写作 '// prism-dsl 1' 并置于文档首行；更高版本属于未来版本"
        ),
    )
}

/// 把 `.prism` 源码编译为 v2 全块语法产物（缺版本头只警告，不阻止编译）。
///
/// # Errors
/// 词法 / 语法 / 语义任一环节失败都返回 [`DslError`]（含出错行号）。
pub fn compile_v2(source: &str) -> Result<CompileV2Output, DslError> {
    compile_v2_with_diagnostics(source).map(|(out, _)| out)
}

/// [`compile_v2`] 的完整形态：同时返回警告诊断列表（当前仅 W1001）。
///
/// # Errors
/// 词法 / 语法 / 语义任一环节失败都返回 [`DslError`]（含出错行号）。
pub fn compile_v2_with_diagnostics(
    source: &str,
) -> Result<(CompileV2Output, Vec<Diagnostic>), DslError> {
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    match detect_version_head(source)? {
        VersionHead::Ok => {}
        VersionHead::Missing => {
            diagnostics.push(Diagnostic {
                line: 1,
                code: "W1001".to_string(),
                message: "缺少版本头 '// prism-dsl 1'\n  建议：在文件首行添加版本头，声明本文档使用 DSL v2 语法（未来版本升级时以此识别）".to_string(),
            });
        }
        VersionHead::Misplaced(line) => {
            // 非首行版本头 = 未按 C11 规格声明（不能静默当作已声明）：
            // 同码 W1001 警告，引导补到首行。
            diagnostics.push(Diagnostic {
                line,
                code: "W1001".to_string(),
                message: format!(
                    "版本头 '// prism-dsl 1' 不在文档首行（现位于第 {line} 行，C11 规定版本头必须是文档第一行）\n  建议：把版本头移到文件首行，声明本文档使用 DSL v2 语法（未来版本升级时以此识别）"
                ),
            });
        }
    }
    let tokens: Vec<Token> = tokenize(source)?;
    let doc: ast::DocumentV2Ast = parser::Parser::new(tokens).parse_document_v2()?;
    let out: CompileV2Output = codegen::codegen_v2(&doc)?;
    Ok((out, diagnostics))
}

// ---------------------------------------------------------------------------
// v2 产物类型 re-export（一次编译的完整产物 + 装包清单）
// ---------------------------------------------------------------------------

pub use crate::codegen::{
    AudioInfo, CameraOrbit, CompileV2Output, FontSlotSpec, FormulaAlign, FormulaNode, LayerInfo,
    NamedChannel, NoteInfo, PostInfo, SceneV2, SimInfo, ThemeInfo, TransitionV2,
};
pub use crate::packs::{PackManifest, SemVer};
// V1.5-D：组合子库公开面（展开函数 / 入参 / 产物 / 注册表）。
pub use crate::combinators::{
    sphere_fragment,
    expand_ken_burns, expand_lower_third, expand_morph_between, expand_split_screen,
    expand_title_card, CombinatorArgs, CombinatorError, NodeGraph,
};
pub use crate::codegen::COMBINATOR_TABLE;

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Tok, tokenize};
    use prism_core::node::{
        DualityPair, Hash256, KernelKind, LatticePlan, MathStructure, OpCapabilities, ParamValue,
        StructureMeta,
    };
    use prism_core::{
        Aesthetic, EvalCtx, Light, Node, NodeRegistry, Payload, SampleIndex, SamplePlane,
        StructureId, eval,
    };
    use prism_math::Vec3;

    /// 断言编译失败，且错误行号与消息关键字匹配。
    fn expect_err(src: &str, want_line: usize, needle: &str) {
        match compile(src) {
            Ok(_) => panic!("以下源码应当编译失败：\n{}", src),
            Err(e) => {
                assert_eq!(e.line, want_line, "错误行号不符：{}", e);
                assert!(
                    e.message.contains(needle),
                    "错误信息应包含 {:?}，实际为：{}",
                    needle,
                    e.message
                );
            }
        }
    }

    /// 断言 v2 编译失败：行号 + 关键字 + 三段式建议段（C18/C19）。
    fn expect_err_v2(src: &str, want_line: usize, needle: &str) {
        match compile_v2(src) {
            Ok(_) => panic!("以下 v2 源码应当编译失败：\n{}", src),
            Err(e) => {
                assert_eq!(e.line, want_line, "错误行号不符：{}", e);
                assert!(
                    e.message.contains(needle),
                    "错误信息应包含 {:?}，实际为：{}",
                    needle,
                    e.message
                );
                assert!(
                    e.message.contains("建议："),
                    "v2 错误必须含建议段（C19）：{}",
                    e.message
                );
                assert!(
                    e.message.contains("[P"),
                    "v2 错误必须含错误码（C18）：{}",
                    e.message
                );
            }
        }
    }

    /// 读节点 F32 参数（类型不符即 panic）。
    fn f32_param(node: &Node, key: &str) -> f32 {
        match node.params.get(key) {
            Some(ParamValue::F32(v)) => *v,
            other => panic!("参数 {key} 应为 F32，实际 {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // v1 测试（25 个，原样保留——零损失凭证）
    // -----------------------------------------------------------------------

    #[test]
    fn compiles_golden_example() {
        let src: &str = include_str!("../../../examples/spin_red_ball.prism");
        let out: CompileOutput = compile(src).expect("黄金示例应能成功编译");
        let scene: &Scene = &out.scene;

        // sphere / material：#c0392b → (0xc0, 0x39, 0x2b)/255。
        assert_eq!(scene.radius, 1.0);
        let want_albedo: Vec3 =
            Vec3::new(0xc0 as f32 / 255.0, 0x39 as f32 / 255.0, 0x2b as f32 / 255.0);
        assert!((scene.material.albedo - want_albedo).length() < 1e-5);
        assert!((scene.material.roughness - 0.3).abs() < 1e-6);
        assert!((scene.material.metalness - 0.05).abs() < 1e-6);

        // camera：fov 45° → 弧度；eye 为占位 (0,0,-orbit_radius)；target 为轨道中心。
        assert_eq!(scene.orbit_radius, 8.0);
        assert_eq!(scene.orbit_center, Vec3::new(0.0, 0.0, 0.0));
        let want_fov: f32 = 45.0_f32.to_radians();
        assert!((scene.camera.fov - want_fov).abs() < 1e-6, "fov 应为 45° 的弧度值");
        assert!((scene.camera.eye - Vec3::new(0.0, 0.0, -8.0)).length() < 1e-6);
        assert!((scene.camera.target - scene.orbit_center).length() < 1e-6);

        // light：direction 原值透传；#fff4e6 颜色；强度 3.0。
        assert_eq!(scene.lights.len(), 1);
        let light: &Light = &scene.lights[0];
        assert!((light.direction - Vec3::new(1.0, -1.0, -0.5)).length() < 1e-6);
        let want_light_color: Vec3 =
            Vec3::new(0xff as f32 / 255.0, 0xf4 as f32 / 255.0, 0xe6 as f32 / 255.0);
        assert!((light.color - want_light_color).length() < 1e-5);
        assert!((light.intensity - 3.0).abs() < 1e-6);

        // timeline / output / aesthetic。
        assert!((out.duration - 3.0).abs() < 1e-6);
        assert_eq!(out.fps, 12);
        assert_eq!(out.width, 480);
        assert_eq!(out.height, 270);
        assert_eq!(out.format, OutputFormat::Frames);
        assert!(scene.aesthetic.teal_orange);
        assert!(scene.aesthetic.filmic);
        assert!((scene.aesthetic.contrast - 1.12).abs() < 1e-6);
    }

    #[test]
    fn minimal_document_uses_defaults() {
        // 只有 scene 块（且为空）的最小文档也应能编译，全部走缺省值。
        let out: CompileOutput = compile("scene \"tiny\" {}").expect("最小文档应能编译");
        let scene: &Scene = &out.scene;

        assert_eq!(scene.radius, 1.0); // 缺省半径
        assert_eq!(scene.orbit_center, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(scene.orbit_radius, 8.0); // 缺省轨道半径
        assert!((scene.camera.fov - 45.0_f32.to_radians()).abs() < 1e-6);
        assert!(scene.lights.is_empty());
        assert!((scene.material.albedo - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-6);
        assert!((scene.material.roughness - 0.5).abs() < 1e-6);
        assert!((scene.material.metalness - 0.0).abs() < 1e-6);
        assert_eq!(scene.aesthetic, Aesthetic::new(false, 1.0, false));
        assert!((out.duration - 3.0).abs() < 1e-6);
        assert_eq!(out.fps, 12);
        assert_eq!(out.width, 480);
        assert_eq!(out.height, 270);
        assert_eq!(out.format, OutputFormat::Frames);
    }

    #[test]
    fn partial_blocks_fill_defaults() {
        // 有 sphere / timeline / output，但缺 camera、缺 aesthetic、material 只给 roughness。
        let src: &str = concat!(
            "scene \"partial\" {\n",
            "  sphere s {\n",
            "    radius 2.5\n",
            "    material {\n",
            "      roughness 0.9\n",
            "    }\n",
            "  }\n",
            "}\n",
            "timeline {\n",
            "  duration 1.5s\n",
            "}\n",
            "output {\n",
            "  format png\n",
            "}\n"
        );
        let out: CompileOutput = compile(src).expect("部分块文档应能编译");
        assert_eq!(out.scene.radius, 2.5);
        assert_eq!(out.scene.orbit_radius, 8.0); // 无 camera → 缺省轨道半径
        assert!((out.scene.camera.fov - 45.0_f32.to_radians()).abs() < 1e-6);
        assert!((out.scene.material.albedo - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-6);
        assert!((out.scene.material.roughness - 0.9).abs() < 1e-6);
        assert!((out.scene.material.metalness - 0.0).abs() < 1e-6);
        assert!((out.duration - 1.5).abs() < 1e-6);
        assert_eq!(out.fps, 12); // timeline 缺 fps → 缺省
        assert_eq!(out.format, OutputFormat::Png);
        assert_eq!(out.scene.aesthetic, Aesthetic::new(false, 1.0, false)); // 无 aesthetic 块
        assert_eq!(out.width, 480);
        assert_eq!(out.height, 270);
    }

    #[test]
    fn multiple_lights_pass_through_direction() {
        // 多盏灯逐盏收集；direction 按解析原值透传，不取模、不归一化。
        let src: &str = concat!(
            "scene \"two_lights\" {\n",
            "  light fill {\n",
            "    direction(2, 0, 0)\n", // 长度 != 1：验证不归一化
            "  }\n",
            "  light rim {\n",
            "    direction(0, -1, 0)\n",
            "    color #ff0000\n",
            "    intensity 2.0\n",
            "  }\n",
            "}\n"
        );
        let out: CompileOutput = compile(src).expect("两盏灯的文档应能编译");
        assert_eq!(out.scene.lights.len(), 2);

        let first: &Light = &out.scene.lights[0];
        assert_eq!(first.direction, Vec3::new(2.0, 0.0, 0.0)); // 原值透传
        assert_eq!(first.color, Vec3::new(1.0, 1.0, 1.0)); // 缺省白色
        assert!((first.intensity - 1.0).abs() < 1e-6); // 缺省强度

        let second: &Light = &out.scene.lights[1];
        assert_eq!(second.direction, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(second.color, Vec3::new(1.0, 0.0, 0.0));
        assert!((second.intensity - 2.0).abs() < 1e-6);
    }

    #[test]
    fn error_missing_closing_brace() {
        // 缺 '}'：文件在 radius 1.0 后结束，报错应落在最后一个 token（第 3 行）。
        expect_err("scene \"unclosed\" {\n  sphere s {\n    radius 1.0\n", 3, "}");
    }

    #[test]
    fn error_illegal_token() {
        // 非法 token '@'：词法错误，行号 3。
        expect_err("scene \"x\" {\n  sphere s {\n    radius @\n  }\n}\n", 3, "@");
    }

    #[test]
    fn error_missing_scene_block() {
        // 缺 scene 块：第一个顶层块不是 scene，报错在第 1 行。
        expect_err("timeline {\n  duration 2s\n}\n", 1, "scene");
    }

    #[test]
    fn error_unknown_field_with_typo() {
        // 未知字段 'colr'（应为 color）。
        expect_err(
            "scene \"x\" {\n  sphere s {\n    radius 1.0\n    colr #ffffff\n  }\n}\n",
            4,
            "colr",
        );
    }

    #[test]
    fn error_duplicate_field() {
        // 同一块中字段重复：报错定位在第二个（重复的）'radius' 关键字所在行。
        expect_err(
            "scene \"x\" {\n  sphere s {\n    radius 1.0\n    radius 2.0\n  }\n}\n",
            4,
            "重复",
        );
    }

    #[test]
    fn error_second_sphere_rejected() {
        // 语义约束：scene 目前最多支持一个 sphere。
        expect_err(
            concat!(
                "scene \"x\" {\n",
                "  sphere a {\n",
                "    radius 1.0\n",
                "  }\n",
                "  sphere b {\n",
                "    radius 2.0\n",
                "  }\n",
                "}\n"
            ),
            5,
            "sphere",
        );
    }

    #[test]
    fn lexer_flags_negatives_resolution_duration_color() {
        // teal-orange 是一个完整 ident；-0.5 是负数；-ball 是以 '-' 开头的 ident；
        // 480x270 / 3s / 2.5s / #c0392b 各自整体词法化。
        let toks: Vec<Token> =
            tokenize("teal-orange -0.5 -ball 480x270 3s 2.5s #c0392b").expect("tokenize 应成功");
        assert_eq!(toks.len(), 7);
        assert_eq!(toks[0], Token { tok: Tok::Ident("teal-orange".to_string()), line: 1 });
        assert_eq!(toks[1].tok, Tok::Num(-0.5));
        assert_eq!(toks[2].tok, Tok::Ident("-ball".to_string()));
        assert_eq!(toks[3].tok, Tok::Resolution(480, 270));
        assert_eq!(toks[4].tok, Tok::Duration(3.0));
        assert_eq!(toks[5].tok, Tok::Duration(2.5));
        assert_eq!(toks[6].tok, Tok::Color(0xc0, 0x39, 0x2b));
    }

    #[test]
    fn lexer_skips_comments_and_tracks_lines() {
        // 注释跳过；每个 token 记录行号。
        let toks: Vec<Token> =
            tokenize("// 顶部注释\nfoo // 行尾注释\n\nbar").expect("tokenize 应成功");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0], Token { tok: Tok::Ident("foo".to_string()), line: 2 });
        assert_eq!(toks[1], Token { tok: Tok::Ident("bar".to_string()), line: 4 });
    }

    #[test]
    fn lexer_rejects_bad_color_and_bad_char() {
        // 非法颜色（非 hex / 位数不足 / 位数超 6）与非法字符都应报词法错误。
        assert!(tokenize("#12g456").is_err());
        assert!(tokenize("#12345").is_err());
        assert!(tokenize("#1234567").is_err());
        assert!(tokenize("radius @").is_err());
        assert!(tokenize("scene \"x").is_err()); // 字符串未闭合
    }

    #[test]
    fn dsl_error_display_format() {
        let e: DslError = DslError { line: 7, message: "测试消息".to_string() };
        assert_eq!(e.to_string(), "DSL 第 7 行: 测试消息");
    }

    // -----------------------------------------------------------------------
    // 边界：缺省组合与极值参数（M1 补充）
    // -----------------------------------------------------------------------

    #[test]
    fn scene_with_only_sphere_uses_default_camera_and_no_lights() {
        // 只有 sphere、无 camera / light：相机与轨道全部走缺省，灯列表为空。
        let src: &str = concat!(
            "scene \"only_sphere\" {\n",
            "  sphere s {\n",
            "    radius 2.0\n",
            "  }\n",
            "}\n"
        );
        let out: CompileOutput = compile(src).expect("只含 sphere 的文档应能编译");
        let scene: &Scene = &out.scene;

        assert_eq!(scene.radius, 2.0);
        // 无 camera → 轨道半径 / 中心 / fov / eye 占位均为缺省。
        assert_eq!(scene.orbit_radius, 8.0);
        assert_eq!(scene.orbit_center, Vec3::new(0.0, 0.0, 0.0));
        assert!((scene.camera.fov - 45.0_f32.to_radians()).abs() < 1e-6);
        assert!((scene.camera.eye - Vec3::new(0.0, 0.0, -8.0)).length() < 1e-6);
        assert!(scene.lights.is_empty(), "无 light 条目时灯列表应为空");
    }

    #[test]
    fn three_lights_keep_declaration_order() {
        // 3 盏灯：逐盏按声明顺序收集，direction / 字段不得串位。
        let src: &str = concat!(
            "scene \"three\" {\n",
            "  light key { direction(1, -1, -0.5) intensity 3.0 }\n",
            "  light fill { direction(-2, 0, 1) intensity 1.5 }\n",
            "  light rim { direction(0, 1, 0) color #00ff00 }\n",
            "}\n"
        );
        let out: CompileOutput = compile(src).expect("三盏灯文档应能编译");
        assert_eq!(out.scene.lights.len(), 3);

        let want_dirs: [Vec3; 3] = [
            Vec3::new(1.0, -1.0, -0.5),
            Vec3::new(-2.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        for (i, want) in want_dirs.iter().enumerate() {
            assert_eq!(
                out.scene.lights[i].direction,
                *want,
                "第 {} 盏灯 direction 应保持声明顺序",
                i + 1
            );
        }
        // 顺序敏感的伴随字段：第 2 盏的强度、第 3 盏的颜色不得互相串位。
        assert!((out.scene.lights[1].intensity - 1.5).abs() < 1e-6);
        assert_eq!(out.scene.lights[2].color, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn resolution_1x1_is_accepted() {
        // 最小合法分辨率 1x1（宽高均 > 0 的下界）。
        let src: &str = "scene \"tiny\" {\n}\noutput {\n  resolution 1x1\n}\n";
        let out: CompileOutput = compile(src).expect("1x1 分辨率应能编译");
        assert_eq!(out.width, 1);
        assert_eq!(out.height, 1);
    }

    #[test]
    fn fractional_duration_0_5s_is_accepted() {
        // 小数时长 0.5s 应完整透传到编译产物。
        let src: &str = "scene \"tiny\" {\n}\ntimeline {\n  duration 0.5s\n}\n";
        let out: CompileOutput = compile(src).expect("0.5s 时长应能编译");
        assert!((out.duration - 0.5).abs() < 1e-6, "duration 应为 0.5，实际 {}", out.duration);
    }

    #[test]
    fn duration_zero_is_rejected() {
        // duration 0s 会在 eval_camera_orbit 中造成除零（NaN 相机），
        // DSL 层直接拒绝，把"静默输出空画面"变成明确报错。
        let src: &str = "scene \"tiny\" {\n}\ntimeline {\n  duration 0s\n}\n";
        let err: DslError = compile(src).expect_err("duration 0s 应被拒绝");
        assert_eq!(err.line, 4, "错误应定位在 duration 所在第 4 行");
        assert!(
            err.message.contains("duration"),
            "错误信息应提及 duration，实际：{}",
            err.message
        );
    }

    // -----------------------------------------------------------------------
    // 错误路径（M1 补充）：词法 / 语法 / 语义各环节
    // -----------------------------------------------------------------------

    #[test]
    fn error_unclosed_string() {
        // 字符串到文件末尾仍未闭合：词法错误，定位在第 1 行。
        expect_err("scene \"abc", 1, "右引号");
        // 字符串以换行"闭合"同样不允许（不能跨行）。
        expect_err("scene \"abc\n", 1, "右引号");
    }

    #[test]
    fn error_illegal_hex_color() {
        // '#1234g5'：第 5 位 'g' 不是十六进制字符，报错定位在颜色所在行。
        expect_err(
            "scene \"x\" {\n  light l {\n    color #1234g5\n  }\n}\n",
            3,
            "#RRGGBB",
        );
    }

    #[test]
    fn error_resolution_missing_height_part() {
        // `resolution 480` 缺 'x 高' 部分：期望 resolution，却得到数字，报错在数字所在行。
        expect_err("scene \"x\" {\n}\noutput {\n  resolution 480\n}\n", 4, "480x270");
    }

    #[test]
    fn error_unknown_field_in_aesthetic_block() {
        // aesthetic 块只允许 teal-orange / contrast / filmic：未知字段 'foo' 应报错。
        expect_err("scene \"x\" {\n}\naesthetic {\n  foo 1\n}\n", 4, "foo");
    }

    #[test]
    fn error_unknown_format_value_rejected() {
        // 当前仅支持 frames / png：`format video` 应报语义错误（不允许静默透传）。
        expect_err("scene \"x\" {\n}\noutput {\n  format video\n}\n", 4, "video");
    }

    // -----------------------------------------------------------------------
    // 一致性（M1 补充）：同一文档两次编译结果确定
    // -----------------------------------------------------------------------

    #[test]
    fn same_document_compiles_deterministically() {
        // 同一文档编译两次：Scene 逐字段相等（Vec3 按分量比较），输出参数逐位相等。
        let src: &str = concat!(
            "scene \"determinism\" {\n",
            "  sphere ball {\n",
            "    radius 1.5\n",
            "    material {\n",
            "      albedo #c0392b\n",
            "      roughness 0.3\n",
            "      metalness 0.05\n",
            "    }\n",
            "  }\n",
            "  camera main {\n",
            "    orbit { center 1,2,3 radius 6.0 }\n",
            "    fov 50\n",
            "  }\n",
            "  light key { direction(1, -1, -0.5) color #fff4e6 intensity 3.0 }\n",
            "  light fill { direction(0, -1, 0) intensity 0.5 }\n",
            "}\n",
            "timeline { duration 2.5s fps 10 }\n",
            "aesthetic { teal-orange contrast 1.12 filmic }\n",
            "output { resolution 320x180 format png }\n"
        );
        let a: CompileOutput = compile(src).expect("第一次编译应成功");
        let b: CompileOutput = compile(src).expect("第二次编译应成功");

        // Scene 实现了 PartialEq：逐字段、Vec3 逐分量比较。
        assert_eq!(a.scene, b.scene, "两次编译的 Scene 应逐字段一致");
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        assert_eq!(a.fps, b.fps);
        assert_eq!(a.format, b.format);
        assert_eq!(a.duration, b.duration, "同一源码解析出的 duration 应逐位相等");
    }

    // -----------------------------------------------------------------------
    // v2：词法层——单位字面量与冒号
    // -----------------------------------------------------------------------

    #[test]
    fn v2_lex_unit_literals() {
        // v2 单位后缀：ms / f / deg / rad / % / px 全部整体词法化，含负数。
        let toks: Vec<Token> =
            tokenize("150ms 36f 15deg 0.5rad 50% 120px -50% 1.5deg").expect("tokenize 应成功");
        assert_eq!(toks.len(), 8);
        assert_eq!(toks[0].tok, Tok::Millis(150.0));
        assert_eq!(toks[1].tok, Tok::Frames(36.0));
        assert_eq!(toks[2].tok, Tok::Deg(15.0));
        assert_eq!(toks[3].tok, Tok::Rad(0.5));
        assert_eq!(toks[4].tok, Tok::Percent(50.0));
        assert_eq!(toks[5].tok, Tok::Px(120.0));
        assert_eq!(toks[6].tok, Tok::Percent(-50.0));
        assert_eq!(toks[7].tok, Tok::Deg(1.5));
    }

    #[test]
    fn v2_lex_suffix_not_eaten_by_ident_and_colon() {
        // 后缀后跟标识符字符 → 不算单位（与 `3s` 同规）；冒号是独立 token。
        let toks: Vec<Token> = tokenize("3stage 3fps layer:text").expect("tokenize 应成功");
        assert_eq!(toks[0].tok, Tok::Num(3.0));
        assert_eq!(toks[1].tok, Tok::Ident("stage".to_string()));
        assert_eq!(toks[2].tok, Tok::Num(3.0));
        assert_eq!(toks[3].tok, Tok::Ident("fps".to_string()));
        assert_eq!(toks[4].tok, Tok::Ident("layer".to_string()));
        assert_eq!(toks[5].tok, Tok::Colon);
        assert_eq!(toks[6].tok, Tok::Ident("text".to_string()));
    }

    // -----------------------------------------------------------------------
    // v2：版本头三态（W1001 警告 / 正常 / P1023 硬错）
    // -----------------------------------------------------------------------

    #[test]
    fn v2_version_head_present_no_diagnostics() {
        let (_, diags) =
            compile_v2_with_diagnostics("// prism-dsl 1\nscene \"s\" {}").expect("应编译成功");
        assert!(diags.is_empty(), "有版本头时不应有警告：{diags:?}");
    }

    #[test]
    fn v2_version_head_missing_warns_w1001() {
        // 缺版本头：编译成功 + 一条 W1001 警告（不阻止编译）。
        let (out, diags) =
            compile_v2_with_diagnostics("scene \"s\" {}").expect("缺版本头仍应编译成功");
        assert_eq!(out.scenes.len(), 1);
        assert_eq!(diags.len(), 1, "应恰好一条警告");
        assert_eq!(diags[0].code, "W1001");
        assert_eq!(diags[0].line, 1);
        assert!(diags[0].message.contains("// prism-dsl 1"), "警告应给出写法：{}", diags[0].message);
        assert!(diags[0].message.contains("建议："), "警告应含建议段（C19）");
    }

    #[test]
    fn v2_version_head_wrong_version_is_hard_error() {
        // 版本号非 1：P1023 硬错（不许静默按 v2 解析）。
        match compile_v2("// prism-dsl 2\nscene \"s\" {}") {
            Ok(_) => panic!("错版本号应硬错"),
            Err(e) => {
                assert_eq!(e.line, 1);
                assert!(e.message.contains("P1023"), "应含 P1023：{}", e.message);
                assert!(e.message.contains("建议："), "应含建议段（C19）：{}", e.message);
            }
        }
    }

    // -----------------------------------------------------------------------
    // v2：黄金示例（全块语法一次到位）与时间线
    // -----------------------------------------------------------------------

    #[test]
    fn v2_golden_full_blocks() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "movie \"测试影片\" { duration 8s fps 24 resolution 640x360 format png }\n",
            "theme { teal-orange contrast 1.1 filmic }\n",
            "import packs \"fx-kit\" { engine \">=0.1\" provides { emitter \"1.0\" } }\n",
            "anim \"cam.orbit\" { key 0s 0 linear key 2s 10 linear loop pingpong }\n",
            "expr \"boost\" \"t*2 + 1\"\n",
            "scene \"开场\" {\n",
            "  duration 3s\n",
            "  stage3d {\n",
            "    camera main { orbit { center 0,0,0 radius 8 } fov 50deg }\n",
            "    sdf ball { kind sphere radius 1.5 }\n",
            "    material red { albedo #c0392b roughness 0.3 metalness 0.05 }\n",
            "    light key { direction(1,-1,-0.5) color #fff4e6 intensity 3 }\n",
            "  }\n",
            "  layer text title { content \"Hello\" x { key 0s 0 linear key 1s 5 linear } y 100 size 48px color #ffffff opacity 90% rotation 15deg }\n",
            "  layer particles { count 200 speed 0.5 }\n",
            "  compose main { add layer:title layer:particles }\n",
            "  post { bloom 0.3 grain 0.1 }\n",
            "  audio { music { bpm 120 gain 0.8 duck 12 } }\n",
            "}\n",
            "scene \"第二幕\" {\n",
            "  stage3d { camera c2 { orbit { center 1,0,0 radius 6 } fov 45 } sdf b2 { kind box } }\n",
            "  sim cloth { grid 16 stiffness 0.9 damping 0.02 force wind { strength 2 direction(1,0,0) } noise { scale 0.1 } }\n",
            "}\n",
            "transition \"cut1\" { from \"开场\" to \"第二幕\" duration 0.5s kind crossfade }\n",
            "output { resolution 640x360 format png }\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("全块语法黄金示例应编译成功");

        // 文档级参数：movie → 产物；theme / packs 登记。
        assert_eq!(out.title, "测试影片");
        assert_eq!(out.fps, 24);
        assert_eq!(out.width, 640);
        assert_eq!(out.height, 360);
        assert_eq!(out.format, OutputFormat::Png);
        assert!((out.duration - 8.0).abs() < 1e-6, "总时长取 movie.duration");
        assert_eq!(
            out.theme,
            Some(ThemeInfo { teal_orange: true, contrast: 1.1, filmic: true })
        );
        assert_eq!(out.packs.len(), 1);
        assert_eq!(out.packs[0].name, "fx-kit");
        assert_eq!(out.packs[0].provides, vec!["emitter".to_string()]);

        // 时间线：s1 显式 3s；s2 缺省 → movie.duration 平分 = 4s；重叠 0.5s。
        assert_eq!(out.scenes.len(), 2);
        assert_eq!(out.scenes[0].name, "开场");
        assert!((out.scenes[0].start - 0.0).abs() < 1e-6);
        assert!((out.scenes[0].duration - 3.0).abs() < 1e-6);
        assert!((out.scenes[1].start - 2.5).abs() < 1e-6, "start = 3 - 0.5");
        assert!((out.scenes[1].duration - 4.0).abs() < 1e-6, "8s 平分两场");
        assert_eq!(out.transitions.len(), 1);
        assert_eq!(out.transitions[0].name, "cut1");
        assert_eq!(out.transitions[0].from, "开场");
        assert_eq!(out.transitions[0].to, "第二幕");
        assert!((out.transitions[0].start - 2.5).abs() < 1e-6);
        assert!((out.transitions[0].duration - 0.5).abs() < 1e-6);
        assert_eq!(out.transitions[0].kind, "crossfade");

        // 场景 1 摘要：camera / layers / post / audio。
        let s1 = &out.scenes[0];
        let cam = s1.camera.as_ref().expect("stage3d.camera 应登记");
        assert_eq!(cam.radius, 8.0);
        assert!((cam.fov_rad - 50.0_f32.to_radians()).abs() < 1e-6, "fov 50deg 归一化为弧度");
        assert_eq!(s1.layers.len(), 2);
        assert_eq!(s1.layers[0].name, "title");
        assert_eq!(s1.layers[0].kind, "text");
        assert_eq!(s1.layers[1].kind, "particles");
        assert!(s1.sim.is_none());
        assert!((s1.post.params["bloom"] - 0.3).abs() < 1e-6);
        assert!((s1.post.params["grain"] - 0.1).abs() < 1e-6);
        let audio = s1.audio.as_ref().expect("audio 应登记");
        assert!((audio.music["bpm"] - 120.0).abs() < 1e-6);
        assert!((audio.music["gain"] - 0.8).abs() < 1e-6);
        assert!((audio.music["duck"] - 12.0).abs() < 1e-6);

        // 场景 2 摘要：sim + fov 裸数字豁免（45 = 度）。
        let s2 = &out.scenes[1];
        let sim = s2.sim.as_ref().expect("sim 应登记");
        assert_eq!(sim.kind, "cloth");
        assert!((s2.camera.as_ref().unwrap().fov_rad - 45.0_f32.to_radians()).abs() < 1e-6);

        // 节点图：s1 = camera + sdf + material + text + particles + compose = 6；
        //         s2 = camera + sdf + sim = 3。
        assert_eq!(
            out.graph.topo_order().expect("图应可拓扑排序").len(),
            9,
            "全文档节点数"
        );

        // 命名通道：字段级关键帧 + 顶层 anim + expr + sim force。
        assert_eq!(out.channels.len(), 4, "title.x / cam.orbit / boost / force strength");
        let x_ch = &out.channels.iter().find(|c| c.name == "开场/title.x").expect("x 通道应登记");
        assert!((x_ch.channel.sample(0.5) - 2.5).abs() < 1e-6, "0→5 @ 1s 的中点");
        let orbit_ch =
            &out.channels.iter().find(|c| c.name == "cam.orbit").expect("anim 通道应登记");
        // pingpong：duration 2s，t=2.5 → 相位 4-2.5=1.5 → track(1.5) = 7.5。
        assert!((orbit_ch.channel.sample(0.0) - 0.0).abs() < 1e-6);
        assert!((orbit_ch.channel.sample(1.0) - 5.0).abs() < 1e-6);
        assert!((orbit_ch.channel.sample(2.5) - 7.5).abs() < 1e-6, "pingpong 回程");
        let boost = &out.channels.iter().find(|c| c.name == "boost").expect("expr 通道应登记");
        assert!((boost.channel.sample(2.0) - 5.0).abs() < 1e-6, "t*2+1 @ t=2");
        let force_ch = &out
            .channels
            .iter()
            .find(|c| c.name == "第二幕/cloth/force/wind/strength")
            .expect("force 通道应登记");
        assert!((force_ch.channel.sample(0.0) - 2.0).abs() < 1e-6, "静态 strength 原值");
    }

    #[test]
    fn v2_multi_scene_transitions_timeline() {
        // 三场两过渡：start[i+1] = start[i] + dur[i] − overlap；总时长 = 末场景结束。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s1\" {}\n",
            "scene \"s2\" {}\n",
            "scene \"s3\" {}\n",
            "transition \"t1\" { from \"s1\" to \"s2\" duration 0.5s }\n",
            "transition \"t2\" { from \"s2\" to \"s3\" duration 1s }\n",
            "timeline { fps 24 }\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("三场两过渡应编译成功");
        assert_eq!(out.scenes.len(), 3);
        assert_eq!(out.transitions.len(), 2);
        let starts: Vec<f32> = out.scenes.iter().map(|s| s.start).collect();
        let durs: Vec<f32> = out.scenes.iter().map(|s| s.duration).collect();
        assert_eq!(durs, vec![2.0, 2.0, 2.0], "无时长来源 → 缺省 2s");
        assert_eq!(starts, vec![0.0, 1.5, 2.5], "过渡重叠折算起点");
        assert!((out.transitions[0].start - 1.5).abs() < 1e-6);
        assert!((out.transitions[1].start - 2.5).abs() < 1e-6);
        assert!((out.duration - 4.5).abs() < 1e-6, "无 timeline/movie.duration → 末场景结束");
        assert_eq!(out.fps, 24, "timeline.fps 优先");
    }

    #[test]
    fn v2_scene_duration_fallbacks() {
        // 用例 A：无任何时长来源 → 每场 2.0s，总时长 4.0s。
        let a: CompileV2Output =
            compile_v2("// prism-dsl 1\nscene \"a\" {}\nscene \"b\" {}\n").expect("用例 A 应编译");
        assert_eq!(a.scenes[0].duration, 2.0);
        assert!((a.duration - 4.0).abs() < 1e-6);

        // 用例 B：movie.duration 平分给未声明 duration 的场景。
        let b: CompileV2Output = compile_v2(
            "// prism-dsl 1\nmovie \"m\" { duration 6s }\nscene \"a\" {}\nscene \"b\" {}\nscene \"c\" {}\n",
        )
        .expect("用例 B 应编译");
        for s in &b.scenes {
            assert!((s.duration - 2.0).abs() < 1e-6, "6s / 3 场 = 2s");
        }
        assert!((b.duration - 6.0).abs() < 1e-6, "总时长取 movie.duration");

        // 用例 C：帧单位时长按 fps 折算（36f @ 24fps = 1.5s）。
        let c: CompileV2Output = compile_v2(
            "// prism-dsl 1\ntimeline { fps 24 }\nscene \"a\" { duration 36f }\n",
        )
        .expect("用例 C 应编译");
        assert!((c.scenes[0].duration - 1.5).abs() < 1e-6, "36f @ 24fps = 1.5s");
    }

    // -----------------------------------------------------------------------
    // v2：单位归一化（C17）
    // -----------------------------------------------------------------------

    #[test]
    fn v2_time_units_ms_and_frames() {
        let src: &str =
            "// prism-dsl 1\nmovie \"m\" { duration 1500ms }\ntimeline { fps 24 }\nscene \"s\" { duration 36f }\n";
        let out: CompileV2Output = compile_v2(src).expect("ms / frames 字面量应编译");
        assert!((out.scenes[0].duration - 1.5).abs() < 1e-6, "36f @ 24fps = 1.5s");
        assert!((out.duration - 1.5).abs() < 1e-6, "movie.duration 1500ms = 1.5s");
        assert_eq!(out.fps, 24);
    }

    #[test]
    fn v2_angle_units_deg_and_rad() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  layer text { rotation 15deg }\n",
            "  layer text \"b2\" { content \"b\" rotation 0.5rad }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("角度字面量应编译");
        let n1 = out.graph.node(out.scenes[0].layers[0].node_id).expect("层节点应存在");
        assert!(
            (f32_param(n1, "rotation") - 15.0_f32.to_radians()).abs() < 1e-6,
            "15deg 应归一化为弧度"
        );
        let n2 = out.graph.node(out.scenes[0].layers[1].node_id).expect("层节点应存在");
        assert_eq!(f32_param(n2, "rotation"), 0.5, "rad 原值透传");
    }

    #[test]
    fn v2_percent_and_px_units() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  layer text { size 120px opacity 50% }\n",
            "  layer particles { count 200 }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("百分比 / 像素字面量应编译");
        let t = out.graph.node(out.scenes[0].layers[0].node_id).expect("text 节点应存在");
        assert_eq!(f32_param(t, "size"), 120.0, "120px 原值透传");
        assert_eq!(f32_param(t, "opacity"), 0.5, "50% 归一化为 0.5");
        let p = out.graph.node(out.scenes[0].layers[1].node_id).expect("particles 节点应存在");
        assert_eq!(p.params.get("count"), Some(&ParamValue::U32(200)));
    }

    // -----------------------------------------------------------------------
    // v2：装包（C12/V9 engine 范围校验）
    // -----------------------------------------------------------------------

    #[test]
    fn v2_packs_collected_and_engine_ok() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "import packs \"kit-a\" {}\n",
            "import packs \"kit-b\" { engine \">=0.1\" provides { emitter \"1.0\" noise \"1.0\" } }\n",
            "scene \"s\" {}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("两个装包声明应编译");
        assert_eq!(out.packs.len(), 2);
        assert_eq!(out.packs[0].name, "kit-a");
        assert!(out.packs[0].engine.is_none());
        assert_eq!(out.packs[1].name, "kit-b");
        assert_eq!(out.packs[1].engine.as_deref(), Some(">=0.1"));
        assert_eq!(out.packs[1].provides, vec!["emitter".to_string(), "noise".to_string()]);
        assert_eq!(PackManifest::engine_version(), (0, 1, 0), "当前引擎版本（CARGO_PKG_VERSION）");
    }

    #[test]
    fn v2_packs_engine_out_of_range() {
        // 不可能满足的版本范围 → P4001。
        expect_err_v2(
            "// prism-dsl 1\nimport packs \"future\" { engine \">=99.0, <99.1\" }\nscene \"s\" {}\n",
            2,
            "P4001",
        );
    }

    #[test]
    fn v2_packs_engine_bad_format() {
        // engine 字段类型错误 → P4002。
        expect_err_v2(
            "// prism-dsl 1\nimport packs \"broken\" { engine 4 }\nscene \"s\" {}\n",
            2,
            "P4002",
        );
    }

    // -----------------------------------------------------------------------
    // v2：图结构防呆（P3005 环带行号 / P3006 悬空 / P3007 场景引用 / P3008 重名）
    // -----------------------------------------------------------------------

    #[test]
    fn v2_cycle_detection_reports_line_numbers() {
        // a → b → a 环：错误路径链带每个 compose 的定义行号（验收点名项）。
        let src: &str = concat!(
            "// prism-dsl 1\n",       // 1
            "scene \"s\" {\n",        // 2
            "  layer text {}\n",      // 3
            "  compose a {\n",        // 4
            "    add compose:b\n",    // 5
            "  }\n",                  // 6
            "  compose b {\n",        // 7
            "    add compose:a\n",    // 8
            "  }\n",                  // 9
            "}\n"                     // 10
        );
        match compile_v2(src) {
            Ok(_) => panic!("compose 引用成环应报错"),
            Err(e) => {
                assert_eq!(e.line, 8, "错误应定位在成环的引用处（第 8 行）");
                assert!(e.message.contains("P3005"), "应含 P3005：{}", e.message);
                // 行号链：环上两个 compose 的定义行 + 回边引用行。
                assert!(e.message.contains("'a'（第 4 行）"), "应含 a 的定义行号：{}", e.message);
                assert!(e.message.contains("'b'（第 7 行）"), "应含 b 的定义行号：{}", e.message);
                assert!(
                    e.message.contains("'a'（引用于第 8 行）"),
                    "应含回边引用行号：{}",
                    e.message
                );
                assert!(e.message.contains("建议："), "应含建议段（C19）：{}", e.message);
            }
        }
    }

    #[test]
    fn v2_dangling_compose_ref() {
        // 引用未定义的块 → P3006（悬空），报错在引用行。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",      // 1
                "scene \"s\" {\n",       // 2
                "  compose a {\n",       // 3
                "    add layer:ghost\n", // 4
                "  }\n",                 // 5
                "}\n"                    // 6
            ),
            4,
            "P3006",
        );
    }

    #[test]
    fn v2_transition_unknown_scene() {
        // transition.from / to 指向不存在的场景 → P3007。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",                                       // 1
                "scene \"a\" {}\n",                                       // 2
                "scene \"b\" {}\n",                                       // 3
                "transition \"t\" { from \"a\" to \"missing\" }\n"        // 4
            ),
            4,
            "P3007",
        );
    }

    #[test]
    fn v2_duplicate_block_name() {
        // 同场景 layer / sim / compose 共享命名空间：重名 → P3008（含首次定义行号）。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",     // 1
                "scene \"s\" {\n",      // 2
                "  layer text {}\n",    // 3
                "  layer text {}\n",    // 4
                "}\n"                   // 5
            ),
            4,
            "P3008",
        );
    }

    #[test]
    fn v2_top_level_audio_rejected() {
        // audio / post 是场景内条目：顶层出现 → P1004。
        expect_err_v2("// prism-dsl 1\nscene \"s\" {}\naudio {}\n", 3, "P1004");
        expect_err_v2("// prism-dsl 1\nscene \"s\" {}\npost {}\n", 3, "P1004");
    }

    // -----------------------------------------------------------------------
    // v2：端到端（DSL → prism-core 节点图 → eval 成功）
    // -----------------------------------------------------------------------

    // —— eval 特解（复刻 prism-core eval.rs 测试的 registry 模式） ——

    /// 测试用子图签名。
    const TEST_SIG: Hash256 = [0xCDu8; 32];

    fn spec_meta(id: StructureId, sample: bool) -> StructureMeta {
        StructureMeta::new(
            id,
            KernelKind::NoKernel,
            DualityPair::ContentDigest,
            LatticePlan::LatticeAnchored,
            OpCapabilities { invert: false, sample },
        )
    }

    fn make<T: MathStructure + Default + 'static>() -> Box<dyn MathStructure> {
        Box::new(T::default())
    }

    /// 常量源特解：apply(Empty) → Scalar(MILLI/1000)；其余透传。
    /// 同一类型经不同 meta 注册到不同 StructureId（meta 以注册时传入者为准）。
    #[derive(Debug)]
    struct ConstSourceSpec<const MILLI: u32> {
        meta: StructureMeta,
    }

    impl<const MILLI: u32> Default for ConstSourceSpec<MILLI> {
        fn default() -> Self {
            Self { meta: spec_meta(StructureId::ScalarField, false) }
        }
    }

    impl<const MILLI: u32> MathStructure for ConstSourceSpec<MILLI> {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Empty => Payload::Scalar(MILLI as f32 / 1000.0),
                other => *other,
            }
        }
    }

    /// splitmix64 终化函数（整数运算跨平台逐位一致）。
    fn hash_mix(seed: u64, x: u64) -> u64 {
        let mut z = seed ^ x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// 采样特解（S4）：样本流 = splitmix 纯函数派生——同 (frame, pixel, sample,
    /// dim, node_id) 同值。
    #[derive(Debug)]
    struct SampleSpec {
        meta: StructureMeta,
    }

    impl Default for SampleSpec {
        fn default() -> Self {
            Self { meta: spec_meta(StructureId::DeterministicSampling, true) }
        }
    }

    impl MathStructure for SampleSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            *x
        }
        fn sample(&self, si: SampleIndex, ctx: &EvalCtx) -> Payload {
            let mut h = hash_mix(ctx.global_seed, si.node_id);
            h = hash_mix(h, si.frame);
            h = hash_mix(h, ((si.pixel.0 as u64) << 32) | si.pixel.1 as u64);
            h = hash_mix(h, si.sample as u64);
            h = hash_mix(h, si.dimension as u64);
            Payload::Scalar((h >> 40) as f32 / 16_777_216.0)
        }
    }

    /// 模拟投影特解（S3）：状态 +1 一步（Empty = 初值 0）。
    #[derive(Debug)]
    struct StepSpec {
        meta: StructureMeta,
    }

    impl Default for StepSpec {
        fn default() -> Self {
            Self { meta: spec_meta(StructureId::ConstraintFlow, false) }
        }
    }

    impl MathStructure for StepSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Empty => Payload::Scalar(0.0),
                Payload::Scalar(v) => Payload::Scalar(v + 1.0),
                other => *other,
            }
        }
    }

    /// v2 E2E 注册表：DSL 产物图涉及的全部官方结构各一个条目。
    fn v2_test_registry() -> NodeRegistry {
        let mut reg = NodeRegistry::new();
        reg.collect(
            "v2.entity",
            spec_meta(StructureId::EntityRepresentation, false),
            make::<ConstSourceSpec<1000>>,
        )
        .unwrap();
        reg.collect(
            "v2.conv",
            spec_meta(StructureId::Convolution, false),
            make::<ConstSourceSpec<0>>,
        )
        .unwrap();
        reg.collect(
            "v2.camera",
            spec_meta(StructureId::Reparameterization, false),
            make::<ConstSourceSpec<0>>,
        )
        .unwrap();
        reg.collect(
            "v2.material",
            spec_meta(StructureId::RadiativeTransfer, false),
            make::<ConstSourceSpec<500>>,
        )
        .unwrap();
        reg.collect(
            "v2.sampler",
            spec_meta(StructureId::DeterministicSampling, true),
            make::<SampleSpec>,
        )
        .unwrap();
        reg.collect(
            "v2.step",
            spec_meta(StructureId::ConstraintFlow, false),
            make::<StepSpec>,
        )
        .unwrap();
        reg
    }

    #[test]
    fn e2e_sim_cloth_evaluates_frames() {
        // 用例 1：sim cloth → S3 节点；同 ctx 改 frame 连续 eval，快照装配推进 0→1→2。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  sim cloth { grid 8 force { strength 2 } }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("sim 场景应编译成功");
        assert_eq!(out.graph.topo_order().expect("图应可拓扑排序").len(), 1);
        let sim = out.scenes[0].sim.as_ref().expect("sim 摘要应存在");
        let node = out.graph.node(sim.node_id).expect("sim 节点应在图内");
        assert_eq!(node.params.get("force_strength"), Some(&ParamValue::F32(2.0)));
        assert_eq!(node.params.get("grid"), Some(&ParamValue::U32(8)));

        let registry = v2_test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };
        let mut ctx = EvalCtx::new(7, 0, 0);
        let o0 = eval(&out.graph, &registry, &plane, TEST_SIG, &mut ctx).expect("frame 0 应求值");
        assert_eq!(o0.get(sim.node_id), Some(Payload::Scalar(0.0)), "S3 首帧：Empty 装配初值 0");
        ctx.frame = 1;
        let o1 = eval(&out.graph, &registry, &plane, TEST_SIG, &mut ctx).expect("frame 1 应求值");
        assert_eq!(o1.get(sim.node_id), Some(Payload::Scalar(1.0)), "快照装配投影一步");
        ctx.frame = 2;
        let o2 = eval(&out.graph, &registry, &plane, TEST_SIG, &mut ctx).expect("frame 2 应求值");
        assert_eq!(o2.get(sim.node_id), Some(Payload::Scalar(2.0)), "逐帧推进 0 → 1 → 2");
    }

    #[test]
    fn e2e_compose_add_evaluates() {
        // 用例 2：layer text（S12）+ layer particles（S4 采样）→ compose add（S2
        // Merge(Plus)）：1.0 + 采样均值 ∈ [1, 2)。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  layer text { content \"hi\" }\n",
            "  layer particles { count 10 }\n",
            "  compose main { add layer:text layer:particles }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("compose 场景应编译成功");
        assert_eq!(out.scenes[0].layers.len(), 2);
        assert_eq!(out.scenes[0].node_ids.len(), 3, "text + particles + compose");
        let compose_id = *out.scenes[0].node_ids.last().expect("compose 最后建点");

        let registry = v2_test_registry();
        let plane = SamplePlane { pixel: (3, 5), spp: 2 };
        let mut ctx = EvalCtx::new(11, 0, 0);
        let o = eval(&out.graph, &registry, &plane, TEST_SIG, &mut ctx)
            .expect("compose 加算图应可求值");
        match o.get(compose_id) {
            Some(Payload::Scalar(v)) => {
                assert!((1.0..2.0).contains(&v), "1.0 + 采样均值 ∈ [1,2)，实际 {v}");
            }
            other => panic!("compose 汇点应输出标量，实际 {other:?}"),
        }
    }

    #[test]
    fn e2e_stage3d_nodes_evaluate() {
        // 用例 3：stage3d camera（S5）+ sdf（S12）+ material（S8）→ 全 Feed 节点，
        // eval 成功且相机参数逐位落盘。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  stage3d {\n",
            "    camera main { orbit { center 0,0,0 radius 6 } fov 50 }\n",
            "    sdf ball { kind sphere radius 1.5 }\n",
            "    material m { albedo #c0392b roughness 0.3 }\n",
            "  }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("stage3d 场景应编译成功");
        let ids = &out.scenes[0].node_ids;
        assert_eq!(ids.len(), 3, "camera + sdf + material");

        // 相机参数断言（fov 50 裸数字 = 度，C17 豁免）。
        let cam = out.graph.node(ids[0]).expect("camera 节点应在图内");
        assert!((f32_param(cam, "fov_rad") - 50.0_f32.to_radians()).abs() < 1e-6);
        assert_eq!(f32_param(cam, "orbit_radius"), 6.0);
        assert_eq!(
            out.scenes[0].camera.as_ref().expect("相机摘要").radius,
            6.0
        );
        // sdf / material 参数断言。
        let ball = out.graph.node(ids[1]).expect("sdf 节点应在图内");
        assert_eq!(ball.params.get("kind"), Some(&ParamValue::Str("sphere".to_string())));
        assert_eq!(f32_param(ball, "radius"), 1.5);
        let mat = out.graph.node(ids[2]).expect("material 节点应在图内");
        assert!((f32_param(mat, "roughness") - 0.3).abs() < 1e-6);

        // eval：三个 Feed 节点全部产出标量（特解常量），eval Ok。
        let registry = v2_test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };
        let mut ctx = EvalCtx::new(3, 0, 0);
        let o = eval(&out.graph, &registry, &plane, TEST_SIG, &mut ctx).expect("stage3d 图应可求值");
        assert_eq!(o.get(ids[0]), Some(Payload::Scalar(0.0)), "camera（S5 常量特解）");
        assert_eq!(o.get(ids[1]), Some(Payload::Scalar(1.0)), "sdf（S12 常量特解）");
        assert_eq!(o.get(ids[2]), Some(Payload::Scalar(0.5)), "material（S8 常量特解）");
    }

    /// V2.0-W4 曲线图元 DSL：四族 kind 编译落盘 + 字段序无关 + 错误码对拍。
    ///
    /// 覆盖：cylinder / cone / torus / revolution 的参数落盘（kind /
    /// center / radius / half_height / major / minor / profile_*）；
    /// kind 声明在字段序任意位置；P1003（缺 kind）、P1002（未知 kind /
    /// 越权字段）、P1027（缺必填）、P1028（尺寸非正）、P1029（profile
    /// 非法）。sphere 既有口径（radius 缺省 1、负值 P1015）不回归。
    #[test]
    fn v2_stage3d_sdf_curve_kinds() {
        // ---- 正路：四族同场，kind 均在字段序中段（序无关性对拍）----
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  stage3d {\n",
            "    sdf rod { radius 0.5 half_height 1.5 kind cylinder center 1,2,3 }\n",
            "    sdf spike { kind cone radius 1 half_height 2 }\n",
            "    sdf ring { center 0,1,0 kind torus major 2 minor 0.5 }\n",
            "    sdf vase { kind revolution profile { pt { r 0 y -1 } pt { r 1 y 0 } pt { r 0 y 1 } } }\n",
            "  }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("四族曲线 sdf 应编译成功");
        let ids = &out.scenes[0].node_ids;
        assert_eq!(ids.len(), 4, "四个 sdf 实体节点");
        let rod = out.graph.node(ids[0]).expect("cylinder 节点");
        assert_eq!(rod.params.get("kind"), Some(&ParamValue::Str("cylinder".to_string())));
        assert_eq!(f32_param(rod, "radius"), 0.5);
        assert_eq!(f32_param(rod, "half_height"), 1.5);
        assert_eq!(f32_param(rod, "center_x"), 1.0);
        assert_eq!(f32_param(rod, "center_y"), 2.0);
        assert_eq!(f32_param(rod, "center_z"), 3.0);
        let cone = out.graph.node(ids[1]).expect("cone 节点");
        assert_eq!(cone.params.get("kind"), Some(&ParamValue::Str("cone".to_string())));
        assert_eq!(f32_param(cone, "radius"), 1.0);
        assert_eq!(f32_param(cone, "half_height"), 2.0);
        let ring = out.graph.node(ids[2]).expect("torus 节点");
        assert_eq!(ring.params.get("kind"), Some(&ParamValue::Str("torus".to_string())));
        assert_eq!(f32_param(ring, "major"), 2.0);
        assert_eq!(f32_param(ring, "minor"), 0.5);
        assert_eq!(f32_param(ring, "center_y"), 1.0);
        let vase = out.graph.node(ids[3]).expect("revolution 节点");
        assert_eq!(vase.params.get("kind"), Some(&ParamValue::Str("revolution".to_string())));
        assert_eq!(vase.params.get("profile_n"), Some(&ParamValue::U32(3)));
        assert_eq!(f32_param(vase, "profile_r_0"), 0.0);
        assert_eq!(f32_param(vase, "profile_y_0"), -1.0);
        assert_eq!(f32_param(vase, "profile_r_1"), 1.0);
        assert_eq!(f32_param(vase, "profile_y_2"), 1.0);

        // ---- 反路：错误码 + 行号对拍（C18/C19：错误码 + 建议段）----
        // P1003：缺 kind。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { radius 1 } }\n}\n",
            3,
            "P1003",
        );
        // P1002：未知 kind。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind capsule } }\n}\n",
            3,
            "P1002",
        );
        // P1002：越权字段（torus 不收 radius；kind 前置也不影响白名单校验）。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind torus major 2 minor 0.5 radius 1 } }\n}\n",
            3,
            "P1002",
        );
        // P1027：cylinder 缺 half_height。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind cylinder radius 1 } }\n}\n",
            3,
            "P1027",
        );
        // P1028：cone 半径非正。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind cone radius 0 half_height 1 } }\n}\n",
            3,
            "P1028",
        );
        // P1029：revolution profile 单点（段数 0，rt 侧永不命中 ⇒ 构造期拒收）。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind revolution profile { pt { r 1 y 0 } } } }\n}\n",
            3,
            "P1029",
        );
        // P1029：profile 子条目名非 pt。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind revolution profile { point { r 1 y 0 } } } }\n}\n",
            3,
            "P1029",
        );
        // sphere 既有口径不回归：radius 缺省 1、负值 P1015。
        let ok: CompileV2Output = compile_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind sphere } }\n}\n",
        )
        .expect("sphere 缺省半径应编译成功");
        let ball = out_ball(&ok);
        assert_eq!(f32_param(ball, "radius"), 1.0);
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind sphere radius -1 } }\n}\n",
            3,
            "P1015",
        );
    }

    /// 取唯一 sdf 节点（单 stage3d 单实体场景的测试辅件）。
    fn out_ball(out: &CompileV2Output) -> &Node {
        let ids = &out.scenes[0].node_ids;
        assert_eq!(ids.len(), 1, "单 sdf 场景应只有一个节点");
        out.graph.node(ids[0]).expect("sdf 节点应在图内")
    }

    // -----------------------------------------------------------------------
    // v2：参数落盘与通道绑定
    // -----------------------------------------------------------------------

    #[test]
    fn v2_layer_text_params_and_channel_binding() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s1\" {\n",
            "  layer text \"标题\" {\n",
            "    content \"你好\"\n",
            "    x { key 0s 0 linear key 1s 10 linear }\n",
            "    y -5\n",
            "    size 48px\n",
            "    color #ff8800\n",
            "    opacity 80%\n",
            "    rotation 30deg\n",
            "  }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("layer text 应编译成功");
        let layer = &out.scenes[0].layers[0];
        assert_eq!(layer.name, "标题", "Str 块名");
        let node = out.graph.node(layer.node_id).expect("text 节点应在图内");

        // 静态参数。
        match node.params.get("content") {
            Some(ParamValue::Str(s)) => assert_eq!(s, "你好"),
            other => panic!("content 应为 Str，实际 {other:?}"),
        }
        assert_eq!(f32_param(node, "y"), -5.0);
        assert_eq!(f32_param(node, "size"), 48.0);
        assert_eq!(f32_param(node, "opacity"), 0.8, "80% 归一化");
        assert!((f32_param(node, "rotation") - 30.0_f32.to_radians()).abs() < 1e-6);
        assert_eq!(f32_param(node, "color_r"), 1.0);
        assert_eq!(f32_param(node, "color_g"), 136.0 / 255.0, "#ff8800 的 g 分量");
        assert_eq!(f32_param(node, "color_b"), 0.0);

        // 关键帧字段：通道登记 + 节点绑定 + t=0 静态值。
        assert_eq!(node.bindings.len(), 1, "只有 x 通道化");
        assert_eq!(node.bindings[0].param, "x");
        let ch = out
            .channels
            .iter()
            .find(|c| c.name == "s1/标题.x")
            .expect("字段级通道应登记");
        assert!((ch.channel.sample(0.5) - 5.0).abs() < 1e-6, "0→10 @ 1s 的中点");
        assert_eq!(f32_param(node, "x"), 0.0, "静态值 = t=0 采样");
    }

    #[test]
    fn v2_sim_force_named_block_channel() {
        // `force wind { strength {key...} }`：带名子块 → 通道 + 绑定。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  sim cloth {\n",
            "    grid 8\n",
            "    force wind { strength { key 0s 1 linear key 1s 3 linear } direction(1,0,0) }\n",
            "  }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("sim force 应编译成功");
        let sim = out.scenes[0].sim.as_ref().expect("sim 摘要应存在");
        let node = out.graph.node(sim.node_id).expect("sim 节点应在图内");
        assert_eq!(f32_param(node, "force_strength"), 1.0, "静态值 = t=0 采样");
        assert_eq!(f32_param(node, "force_dir_x"), 1.0);
        let ch = out
            .channels
            .iter()
            .find(|c| c.name == "s/cloth/force/wind/strength")
            .expect("force 通道应按名登记");
        assert!((ch.channel.sample(0.5) - 2.0).abs() < 1e-6, "1→3 @ 1s 的中点");
        assert_eq!(node.bindings.len(), 1);
        assert_eq!(node.bindings[0].param, "force_strength");
    }

    #[test]
    fn v2_anim_loop_modes_sampling() {
        // 三种 loop 模式的采样语义（与 prism-core property.rs 的 Loop 组合子一致）。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "anim \"a.direct\" { key 0s 0 linear key 1s 10 linear }\n",
            "anim \"a.loop\" { key 0s 0 linear key 1s 10 linear loop loop }\n",
            "anim \"a.pingpong\" { key 0s 0 linear key 1s 10 linear loop pingpong }\n",
            "scene \"s\" {}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("三种 anim 应编译成功");
        assert_eq!(out.channels.len(), 3);
        let find = |name: &str| {
            &out.channels.iter().find(|c| c.name == name).expect("通道应存在").channel
        };
        // Direct：超时钳末帧。
        assert!((find("a.direct").sample(2.0) - 10.0).abs() < 1e-6);
        // Loop：周期回绕，t=duration 整回绕首帧。
        assert!((find("a.loop").sample(1.5) - 5.0).abs() < 1e-6);
        assert_eq!(find("a.loop").sample(1.0), 0.0);
        // PingPong：t=duration 折返（末帧），回程对称。
        assert!((find("a.pingpong").sample(1.0) - 10.0).abs() < 1e-6);
        assert!((find("a.pingpong").sample(1.5) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn v2_expr_channel_sampling() {
        // expr 块：受限表达式经 prism_anim::Expr 解析后登记为通道。
        let src: &str = "// prism-dsl 1\nexpr \"boost\" \"t*2 + 1\"\nscene \"s\" {}\n";
        let out: CompileV2Output = compile_v2(src).expect("expr 应编译成功");
        let ch = &out.channels.iter().find(|c| c.name == "boost").expect("expr 通道应登记").channel;
        assert!((ch.sample(2.0) - 5.0).abs() < 1e-6, "t*2+1 @ t=2 = 5");
        assert!((ch.sample(0.0) - 1.0).abs() < 1e-6, "t*2+1 @ t=0 = 1");
    }

    // -----------------------------------------------------------------------
    // v2：防呆逐码（C18 三部分错误 + C19 建议段）
    // -----------------------------------------------------------------------

    #[test]
    fn v2_guard_movie_values() {
        // P1011：movie.duration 0。
        expect_err_v2(
            "// prism-dsl 1\nmovie \"m\" { duration 0s }\nscene \"s\" {}\n",
            2,
            "P1011",
        );
        // P1011：transition 重叠 ≥ 场景时长。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",                                    // 1
                "scene \"a\" { duration 2s }\n",                       // 2
                "scene \"b\" { duration 2s }\n",                       // 3
                "transition \"t\" { from \"a\" to \"b\" duration 2s }\n" // 4
            ),
            4,
            "P1011",
        );
        // P1012：fps 非正整数。
        expect_err_v2("// prism-dsl 1\nmovie \"m\" { fps 0 }\nscene \"s\" {}\n", 2, "P1012");
        // P1013：分辨率 0。
        expect_err_v2(
            "// prism-dsl 1\nmovie \"m\" { resolution 0x0 }\nscene \"s\" {}\n",
            2,
            "P1013",
        );
        // P2102：裸时间（缺单位）。
        expect_err_v2(
            "// prism-dsl 1\nmovie \"m\" { duration 5 }\nscene \"s\" {}\n",
            2,
            "P2102",
        );
        // P2106：format 取值白名单。
        expect_err_v2(
            "// prism-dsl 1\nmovie \"m\" { format video }\nscene \"s\" {}\n",
            2,
            "P2106",
        );
    }

    #[test]
    fn v2_guard_camera_and_material() {
        // P1014：fov 超出 (0, 180) 开区间。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",                                        // 1
                "scene \"s\" {\n",                                         // 2
                "  stage3d {\n",                                           // 3
                "    camera c { orbit { center 0,0,0 } fov 180 }\n",       // 4
                "  }\n",                                                   // 5
                "}\n"                                                      // 6
            ),
            4,
            "P1014",
        );
        // P1016：metalness 超出 [0,1]。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { material m { metalness 2 } }\n}\n",
            3,
            "P1016",
        );
        // P1017：roughness 超出 [0,1]。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { material m { roughness 1.5 } }\n}\n",
            3,
            "P1017",
        );
        // P1015：sdf.radius 负值。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" {\n  stage3d { sdf b { kind sphere radius -1 } }\n}\n",
            3,
            "P1015",
        );
    }

    #[test]
    fn v2_guard_audio_and_post() {
        // P1019：bpm 非正。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { audio { music { bpm 0 } } }\n",
            2,
            "P1019",
        );
        // P1018：duck 非正。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { audio { music { duck 0 } } }\n",
            2,
            "P1018",
        );
        // P1022：gain 超出 [0,1]。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { audio { music { gain 2 } } }\n",
            2,
            "P1022",
        );
        // P1021：post 强度超出 [0,1]。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { post { bloom 2 } }\n",
            2,
            "P1021",
        );
    }

    #[test]
    fn v2_note_block_compiles() {
        // 合法 note 块：全字段 → NoteInfo；缺省回填。
        let src = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  note \"主题\" { scale phrygian root 45 bpm 138 bars 8 seed 7 arpeggio 1 }\n",
            "  note \"空\" { scale major }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("note 块应编译成功");
        let s = &out.scenes[0];
        assert_eq!(s.notes.len(), 2);
        let n = &s.notes[0];
        assert_eq!(n.name, "主题");
        assert_eq!(n.scale, "phrygian");
        assert_eq!(n.root, 45);
        assert!((n.bpm - 138.0).abs() < 1e-6);
        assert_eq!(n.bars, 8);
        assert_eq!(n.seed, 7);
        assert!(n.arpeggio, "裸标志 = true");
        // 缺省回填。
        let d = &s.notes[1];
        assert_eq!(d.scale, "major");
        assert_eq!(d.root, 60);
        assert!((d.bpm - 120.0).abs() < 1e-6);
        assert_eq!(d.bars, 4);
        assert_eq!(d.seed, 0);
        assert!(!d.arpeggio);
    }

    #[test]
    fn v2_guard_note_block() {
        // P1023：未知音阶。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { scale bluesy } }\n",
            2,
            "P1023",
        );
        // P1024：root 越界。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { root 200 } }\n",
            2,
            "P1024",
        );
        // P1019：bpm 非正。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { bpm 0 } }\n",
            2,
            "P1019",
        );
        // P1025：bars < 1。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { bars 0 } }\n",
            2,
            "P1025",
        );
        // P1026：seed 为负。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { seed -1 } }\n",
            2,
            "P1026",
        );
        // P1002：未知字段。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { note \"n\" { tempo 100 } }\n",
            2,
            "P1002",
        );
    }

    #[test]
    fn v2_guard_anim_and_expr() {
        // P1020：关键帧时间非严格递增（顶层 anim）。
        expect_err_v2(
            concat!(
                "// prism-dsl 1\n",        // 1
                "anim \"w\" {\n",          // 2
                "  key 1s 0 linear\n",     // 3
                "  key 0s 5 linear\n",     // 4
                "}\n",                     // 5
                "scene \"s\" {}\n"         // 6
            ),
            4,
            "P1020",
        );
        // P1020：字段级关键帧同刻度。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { layer text { y { key 0s 5 linear key 0s 5 linear } } }\n",
            2,
            "P1020",
        );
        // P2103：未知 easing 名。
        expect_err_v2(
            "// prism-dsl 1\nanim \"w\" { key 0s 0 wobbling }\nscene \"s\" {}\n",
            2,
            "P2103",
        );
        // P2104：未知 loop 模式。
        expect_err_v2(
            "// prism-dsl 1\nanim \"w\" { key 0s 0 linear loop spinny }\nscene \"s\" {}\n",
            2,
            "P2104",
        );
        // P2105：expr 语法非法。
        expect_err_v2(
            "// prism-dsl 1\nexpr \"x\" \"t*)\"\nscene \"s\" {}\n",
            2,
            "P2105",
        );
    }

    #[test]
    fn v2_guard_field_values() {
        // P2101：裸角度（rotation 缺单位；fov 之外的裸数字一律拒绝，C17）。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { layer text { rotation 15 } }\n",
            2,
            "P2101",
        );
        // P2107：字段缺值。
        expect_err_v2(
            "// prism-dsl 1\nscene \"s\" { layer text { content } }\n",
            2,
            "P2107",
        );
        // P1002：未知顶层块（含建议）。
        expect_err_v2("// prism-dsl 1\nmovie2 \"m\" {}\nscene \"s\" {}\n", 2, "P1002");
        // P1003：空文档（无 scene）。
        expect_err_v2("// prism-dsl 1\n", 1, "P1003");
    }

    // -----------------------------------------------------------------------
    // v2：可选块名与入口互斥
    // -----------------------------------------------------------------------

    #[test]
    fn v2_optional_and_quoted_block_names() {
        // layer / sim / compose / stage3d 的块名可省略（缺省同种类）或为字符串。
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "scene \"s\" {\n",
            "  layer text \"title1\" { content \"hi\" }\n",
            "  sim cloth { grid 8 }\n",
            "  stage3d stage { camera main { orbit { center 0,0,0 } fov 45 } }\n",
            "  compose main { add layer:title1 sim:cloth }\n",
            "}\n"
        );
        let out: CompileV2Output = compile_v2(src).expect("可选块名形态应编译成功");
        assert_eq!(out.scenes[0].layers[0].name, "title1", "Str 块名");
        assert_eq!(out.scenes[0].sim.as_ref().expect("sim").name, "cloth", "Ident 缺省名");
        assert!(out.scenes[0].camera.is_some());
        assert_eq!(out.scenes[0].node_ids.len(), 4, "camera + text + sim + compose");
    }

    #[test]
    fn v2_blocks_in_v1_entry_get_upgrade_hint() {
        // v2 块喂给 v1 入口：消息保留 v1 原文 + 附加升级指引。
        match compile("movie \"m\" {}\n") {
            Ok(_) => panic!("v1 入口应拒绝 v2 块"),
            Err(e) => {
                assert_eq!(e.line, 1);
                assert!(
                    e.message.contains("文档应以 'scene' 块开始，却遇到 'movie'"),
                    "v1 原文应保留：{}",
                    e.message
                );
                assert!(
                    e.message.contains("compile_v2"),
                    "应给出 v2 升级指引：{}",
                    e.message
                );
            }
        }
    }

    #[test]
    fn v1_entry_message_unchanged_for_timeline() {
        // v1 兼容性关键点：timeline 开头的文档报 v1 原文（无 v2 指引混入）。
        match compile("timeline {\n  duration 2s\n}\n") {
            Ok(_) => panic!("v1 入口应拒绝 timeline 开头"),
            Err(e) => {
                assert_eq!(e.line, 1);
                assert_eq!(
                    e.message,
                    "文档应以 'scene' 块开始，却遇到 'timeline'",
                    "v1 消息必须逐字不变"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // V1.5-B：font 指令（外部字体回退链声明）
    // -----------------------------------------------------------------------

    /// font 指令往返：parse → codegen 产出 FontSet 构造序列，顺序 = 声明顺序。
    #[test]
    fn v2_font_directive_builds_declared_fallback_chain() {
        let src = "// prism-dsl 1\n\
                   font \"assets/MyFont.ttf\" fallback \"assets/B.ttf\", \"assets/C.ttf\"\n\
                   font \"assets/D.ttf\"\n\
                   scene \"s\" {}\n";
        let out: CompileV2Output = compile_v2(src).expect("font 指令应可编译");
        // 展平为单条回退链：首选在前，fallback 依次在后，多条指令按出现顺序。
        let paths: Vec<&str> = out.fonts.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["assets/MyFont.ttf", "assets/B.ttf", "assets/C.ttf", "assets/D.ttf"],
            "push 顺序必须等于声明顺序"
        );
        // index = 展平下标 = FontSet::push_file 返回的槽位号。
        assert_eq!(out.fonts.iter().map(|f| f.index).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        // 行号定位可回溯到声明行（装载失败诊断用）。
        assert_eq!(out.fonts[0].line, 2);
        assert_eq!(out.fonts[1].line, 2);
        assert_eq!(out.fonts[3].line, 3);
        // 标签保留路径原文（对接 FontSlot::label）。
        assert_eq!(out.fonts[2].label, "assets/C.ttf");
        // 无 font 指令 → 空序列（不隐式塞内嵌字体：诚实边界）。
        let plain: CompileV2Output =
            compile_v2("// prism-dsl 1\nscene \"s\" {}\n").expect("应可编译");
        assert!(plain.fonts.is_empty(), "未声明 font 时构造序列必须为空");
    }

    /// font 指令语法错误：空路径 / 缺逗号 / 尾逗号 → 带行号 + 三段式建议。
    #[test]
    fn v2_font_directive_syntax_errors_are_located() {
        // 空首选路径。
        expect_err_v2(
            "// prism-dsl 1\nfont \"\" fallback \"b.ttf\"\nscene \"s\" {}\n",
            2,
            "首选字体路径不能为空",
        );
        // 空 fallback 路径。
        expect_err_v2(
            "// prism-dsl 1\nfont \"a.ttf\" fallback \"\"\nscene \"s\" {}\n",
            2,
            "fallback 字体路径不能为空",
        );
        // 相邻路径缺逗号。
        expect_err_v2(
            "// prism-dsl 1\nfont \"a.ttf\" fallback \"b.ttf\" \"c.ttf\"\nscene \"s\" {}\n",
            2,
            "缺少逗号",
        );
        // 尾逗号后直接换到下一个顶层块（逗号后缺路径）。
        expect_err_v2(
            "// prism-dsl 1\nfont \"a.ttf\" fallback \"b.ttf\",\nscene \"s\" {}\n",
            3,
            "逗号之后缺少字体路径",
        );
        // fallback 后不是字符串（v1 风格无错误码的断言：只校验行号与措辞）。
        match compile_v2("// prism-dsl 1\nfont \"a.ttf\" fallback b\nscene \"s\" {}\n") {
            Ok(_) => panic!("fallback 后跟裸标识符应失败"),
            Err(e) => {
                assert_eq!(e.line, 2, "行号应指向 font 指令行：{}", e.message);
                assert!(
                    e.message.contains("双引号字符串"),
                    "应提示需要字符串：{}",
                    e.message
                );
            }
        }
        // 顶层未知块提示里已列出 font 关键字（可发现性）。
        match compile_v2("// prism-dsl 1\nfonts \"a.ttf\"\nscene \"s\" {}\n") {
            Ok(_) => panic!("'fonts' 应视为未知块"),
            Err(e) => assert!(e.message.contains("font"), "允许列表应含 font：{}", e.message),
        }
    }
}
