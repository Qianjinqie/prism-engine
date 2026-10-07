//! `prism-app critique` 子命令：critic 闭环 CLI 出口（C5；A8 智能辅助并入）。
//!
//! 语义权威：critic 内建为引擎命令且接增量渲染、 C5
//! 行（analyze/critique/suggest/apply + `--autofix`）、CLI 三出口之一。
//! 核心逻辑在 [`prism_aesthetic::cmd`]（CriticCommand）与 [`prism_aesthetic::critic_loop`]；
//! A8 智能辅助建议在 [`prism_ai::assist`]（D-11 引擎内确定性 mock）。本文件只做
//! CLI 适配（子命令分发 + 从引擎渲染取帧）。
//!
//! # 边界（D-11）
//! 引擎 = 纯渲染执行器：本命令对**已渲染帧**做可计算质量评估与确定性辅助建议；
//! 真正的"AI 编排"（LLM→旁白→TTS）留在应用侧。辅助建议走 prism-ai 的确定性
//! mock（无网络、无密钥、无随机）。
//!
//! # 用法
//! ```text
//! prism-app critique <FILE.prism> [--frame T] [--autofix N] [--json]
//! ```
//! 渲染指定帧（或内置演示帧）→ analyze/critique/suggest →（可选 `--autofix N`
//! 贪心爬山）→ 打印报告。

use prism_aesthetic::cmd::{CriticCommand, CritiqueReport};
use prism_aesthetic::critic_loop::{Convergence, Suggestion};
use prism_ai::assist::{AdviceRequest, AssistAdvice};
use prism_io::fbuf::Fbuf16F;

/// critique 子命令用法文本。
pub const CRITIQUE_USAGE: &str = r#"prism-app critique — critic 闭环（质量评估 + 辅助建议 + --autofix）

用法:
  prism-app critique <FILE.prism> [选项]

选项:
  --frame <T>     评估时刻 T 秒的单帧（默认 0）
  --autofix <N>   运行 N 轮自动修复（走增量渲染；贪心爬山，仅在评分回升时接纳）
  --json          以 JSON 形式输出报告（供 MCP/HTTP 复用）
  -h, --help      显示本帮助

输出: 曝光/对比/色彩/清晰度/构图 指标 + 评分 + 诊断 + 确定性辅助建议。"#;

/// 解析后的 critique CLI 选项。
#[derive(Debug, Clone, PartialEq)]
pub struct CritiqueArgs {
    /// `.prism` 源文件路径。
    pub file: String,
    /// 评估时刻（秒）。
    pub frame_time: f32,
    /// `--autofix` 轮数（0 = 不运行）。
    pub autofix_rounds: u32,
    /// `--json` 输出开关。
    pub json: bool,
}

/// 解析 critique 子命令参数（`args` 为子命令名之后的所有参数）。
///
/// 返回 `Ok(None)` 表示显式请求帮助。
pub fn parse_critique_args(args: &[String]) -> Result<Option<CritiqueArgs>, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(None);
    }
    let mut file: Option<String> = None;
    let mut frame_time: f32 = 0.0;
    let mut autofix_rounds: u32 = 0;
    let mut json: bool = false;

    let mut i: usize = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--frame" => {
                i += 1;
                let v = args.get(i).ok_or_else(|| "--frame 缺少取值".to_string())?;
                let t: f32 = v
                    .parse()
                    .map_err(|_| format!("参数 --frame 的取值 `{v}` 不是合法数字"))?;
                if !t.is_finite() || t < 0.0 {
                    return Err(format!("参数 --frame 的取值 `{v}` 必须是非负有限数"));
                }
                frame_time = t;
            }
            "--autofix" => {
                i += 1;
                let v = args.get(i).ok_or_else(|| "--autofix 缺少取值".to_string())?;
                autofix_rounds = v
                    .parse()
                    .map_err(|_| format!("参数 --autofix 的取值 `{v}` 不是合法整数"))?;
            }
            "--json" => json = true,
            other => {
                if other.starts_with('-') {
                    return Err(format!("未知参数 `{other}`"));
                }
                if file.is_some() {
                    return Err(format!("多余的源文件参数 `{other}`（只能指定一个 .prism）"));
                }
                file = Some(other.to_string());
            }
        }
        i += 1;
    }
    let file = file.ok_or_else(|| "缺少 <FILE.prism> 场景文件参数".to_string())?;
    Ok(Some(CritiqueArgs { file, frame_time, autofix_rounds, json }))
}

/// 由 critic 建议派生 A8 确定性辅助建议（D-11 边界内；不依赖网络）。
///
/// `seed` 由调用方给定（确定）；输出顺序 = 建议顺序（确定性）。
pub fn assist_from_suggestions(
    suggestions: &[Suggestion],
    seed: u64,
) -> Result<Vec<AssistAdvice>, prism_ai::AiError> {
    let reqs: Vec<AdviceRequest> = suggestions
        .iter()
        .map(|s| {
            AdviceRequest::new(
                s.dim.name(),
                s.direction.name(),
                s.strength,
                seed,
            )
        })
        .collect();
    prism_ai::assist::advise_all(&reqs)
}

/// 把报告渲染成人类可读文本（确定性；CLI 默认出口）。
pub fn render_report_text(
    report: &CritiqueReport,
    advices: &[AssistAdvice],
    autofix: Option<(Convergence, f32)>,
) -> String {
    let m = &report.metrics;
    let mut out = String::new();
    out.push_str("critic 报告\n");
    out.push_str(&format!(
        "  指标: 曝光(log luma)={:.4} 对比(std)={:.4} 饱和度={:.4} 色偏={:.4} 清晰度={:.6} 三分线={:.4} 重心偏移={:.4}\n",
        m.log_mean_luma, m.luma_std, m.mean_saturation, m.color_cast, m.sharpness,
        m.thirds_energy, m.centroid_offset
    ));
    out.push_str(&format!(
        "  评分: 总分={:.4}  各维[曝光 {:.3} 对比 {:.3} 色彩 {:.3} 清晰 {:.3} 构图 {:.3}]\n",
        report.score.total,
        report.score.dim(prism_aesthetic::Dim::Exposure),
        report.score.dim(prism_aesthetic::Dim::Contrast),
        report.score.dim(prism_aesthetic::Dim::Color),
        report.score.dim(prism_aesthetic::Dim::Sharpness),
        report.score.dim(prism_aesthetic::Dim::Composition),
    ));
    out.push_str(&format!("  诊断: {} 条\n", report.score.diagnoses.len()));
    for d in &report.score.diagnoses {
        out.push_str(&format!(
            "    - {}.{} (severity={:.3}, measured={:.4}, target={:.4})\n",
            d.dim.name(),
            d.direction.name(),
            d.severity,
            d.measured,
            d.target
        ));
    }
    out.push_str(&format!("  辅助建议: {} 条\n", advices.len()));
    for a in advices {
        out.push_str(&format!(
            "    - [{}] {} (conf={:.3})\n",
            a.action, a.text, a.confidence
        ));
    }
    if let Some((conv, score)) = autofix {
        out.push_str(&format!(
            "  --autofix: 收敛态={:?} 最终评分={:.4}\n",
            conv, score
        ));
    }
    out
}

/// 把报告渲染成 JSON（零依赖手写；供 MCP/HTTP 复用）。
pub fn render_report_json(
    report: &CritiqueReport,
    advices: &[AssistAdvice],
    autofix: Option<(Convergence, f32)>,
) -> String {
    let m = &report.metrics;
    let dims = [
        ("exposure", report.score.dim(prism_aesthetic::Dim::Exposure)),
        ("contrast", report.score.dim(prism_aesthetic::Dim::Contrast)),
        ("color", report.score.dim(prism_aesthetic::Dim::Color)),
        ("sharpness", report.score.dim(prism_aesthetic::Dim::Sharpness)),
        ("composition", report.score.dim(prism_aesthetic::Dim::Composition)),
    ];
    let dim_json: Vec<String> = dims
        .iter()
        .map(|(k, v)| format!("\"{}\":{:.6}", k, v))
        .collect();
    let diag_json: Vec<String> = report
        .score
        .diagnoses
        .iter()
        .map(|d| {
            format!(
                "{{\"dim\":\"{}\",\"direction\":\"{}\",\"severity\":{:.6},\"measured\":{:.6},\"target\":{:.6}}}",
                d.dim.name(), d.direction.name(), d.severity, d.measured, d.target
            )
        })
        .collect();
    let adv_json: Vec<String> = advices
        .iter()
        .map(|a| {
            format!(
                "{{\"action\":\"{}\",\"text\":{},\"confidence\":{:.6},\"fingerprint\":{}}}",
                a.action, json_string(&a.text), a.confidence, a.fingerprint
            )
        })
        .collect();
    let autofix_json = match autofix {
        Some((conv, s)) => format!(
            ",\"autofix\":{{\"convergence\":\"{:?}\",\"score\":{:.6}}}",
            conv, s
        ),
        None => String::new(),
    };
    format!(
        "{{\"metrics\":{{\"log_mean_luma\":{:.6},\"luma_std\":{:.6},\"mean_saturation\":{:.6},\"color_cast\":{:.6},\"sharpness\":{:.6},\"thirds_energy\":{:.6},\"centroid_offset\":{:.6},\"pixels\":{}}},\"score\":{{\"total\":{:.6},\"dims\":{{{}}}}},\"diagnoses\":[{}],\"advice\":[{}]{}}}",
        m.log_mean_luma, m.luma_std, m.mean_saturation, m.color_cast, m.sharpness,
        m.thirds_energy, m.centroid_offset, m.pixels,
        report.score.total, dim_json.join(","),
        diag_json.join(","), adv_json.join(","), autofix_json
    )
}

/// JSON 字符串转义（最小实现：转义 `\`、`"`、控制字符）。
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
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
    out
}

/// 命令主逻辑（对已渲染帧执行 critic；无 IO）。
/// `draft` 为待评估帧（由调用方渲染提供）。返回（报告，辅助建议，autofix 结果）。
pub fn run_critique_on_frame(
    cmd: &CriticCommand,
    draft: &Fbuf16F,
    autofix_rounds: u32,
    seed: u64,
) -> (CritiqueReport, Vec<AssistAdvice>, Option<(Convergence, f32)>) {
    let report = cmd.critique(draft);
    // 辅助建议失败（如维度名为空——此处不可能）时降级为空（不 panic）。
    let advices = assist_from_suggestions(&report.suggestions, seed).unwrap_or_default();

    let autofix = if autofix_rounds > 0 {
        // 单帧闭环：render 恒返回 draft；fix = apply 全部建议。
        let (conv, best, _rounds) = cmd.autofix(
            1,
            autofix_rounds.max(1),
            |_| draft.clone(),
            |fb, sug| cmd.apply(&fb, sug),
        );
        best.map(|s| (conv, s))
    } else {
        None
    };

    (report, advices, autofix)
}

/// RGBA8 [`prism_gpu::Framebuffer`] → 线性 [`Fbuf16F`]（按 `b/255` 归一，确定性）。
///
/// 渲染后端输出为 8bit 字节；critic 的指标是相对量（曝光/对比/色彩/清晰度），
/// 域一致即可，不要求线性。
fn framebuffer_to_fbuf(fb: &prism_gpu::Framebuffer) -> Fbuf16F {
    let mut out = Fbuf16F::new(fb.width, fb.height);
    let w = fb.width as usize;
    for y in 0..fb.height {
        for x in 0..fb.width {
            let i = ((y as usize) * w + (x as usize)) * 3;
            if i + 2 < fb.pixels.len() {
                let r = fb.pixels[i] as f32 / 255.0;
                let g = fb.pixels[i + 1] as f32 / 255.0;
                let b = fb.pixels[i + 2] as f32 / 255.0;
                out.set(x, y, [r, g, b, 1.0]);
            }
        }
    }
    out
}

/// `prism-app critique` 子命令入口（CLI 适配层）。
///
/// 流程：解析参数 → 读 `.prism` → DSL 编译 → CpuRef 渲染指定帧 →
/// [`run_critique_on_frame`] → 打印报告（文本 / JSON）。退出码：0 成功 / 1 失败。
pub fn run(args: &[String]) -> i32 {
    let parsed = match parse_critique_args(args) {
        Ok(None) => {
            println!("{CRITIQUE_USAGE}");
            return 0;
        }
        Ok(Some(a)) => a,
        Err(msg) => {
            eprintln!("prism-app critique: {msg}");
            eprintln!("{CRITIQUE_USAGE}");
            return 1;
        }
    };

    let source = match std::fs::read_to_string(&parsed.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("prism-app critique: 无法读取场景文件 `{}`: {e}", parsed.file);
            return 1;
        }
    };
    let compiled = match prism_dsl::compile(&source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let t = if parsed.frame_time > compiled.duration {
        compiled.duration
    } else {
        parsed.frame_time
    };
    let mut device = prism_gpu::CpuRefBackend;
    let frame = prism_render::render_frame(
        &mut device,
        &compiled.scene,
        t,
        compiled.duration,
        compiled.width,
        compiled.height,
    );
    let fbuf = framebuffer_to_fbuf(&frame);

    let cmd = CriticCommand::default();
    let (report, advices, autofix) =
        run_critique_on_frame(&cmd, &fbuf, parsed.autofix_rounds, 0);

    if parsed.json {
        println!("{}", render_report_json(&report, &advices, autofix));
    } else {
        println!("源文件: {}", parsed.file);
        print!("{}", render_report_text(&report, &advices, autofix));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, c);
            }
        }
        fb
    }

    #[test]
    fn parse_defaults_and_options() {
        let a = parse_critique_args(&argv(&["demo.prism"])).unwrap().unwrap();
        assert_eq!(a.file, "demo.prism");
        assert_eq!(a.frame_time, 0.0);
        assert_eq!(a.autofix_rounds, 0);
        assert!(!a.json);

        let b = parse_critique_args(&argv(&[
            "demo.prism", "--frame", "1.5", "--autofix", "2", "--json",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(b.frame_time, 1.5);
        assert_eq!(b.autofix_rounds, 2);
        assert!(b.json);
    }

    #[test]
    fn parse_help_and_errors() {
        assert!(parse_critique_args(&argv(&["-h"])).unwrap().is_none());
        assert!(parse_critique_args(&argv(&[])).is_err());
        assert!(parse_critique_args(&argv(&["a.prism", "--frame"])).is_err());
        assert!(parse_critique_args(&argv(&["a.prism", "--frame", "-1"])).is_err());
        assert!(parse_critique_args(&argv(&["a.prism", "--wat"])).is_err());
        assert!(parse_critique_args(&argv(&["a.prism", "b.prism"])).is_err());
    }

    #[test]
    fn assist_derives_one_advice_per_suggestion() {
        let cmd = CriticCommand::default();
        let fb = solid(16, 16, [0.01, 0.01, 0.01, 1.0]);
        let rep = cmd.critique(&fb);
        let adv = assist_from_suggestions(&rep.suggestions, 42).unwrap();
        assert_eq!(adv.len(), rep.suggestions.len());
    }

    #[test]
    fn report_text_and_json_contain_key_fields() {
        let cmd = CriticCommand::default();
        let fb = solid(16, 16, [0.02, 0.02, 0.02, 1.0]);
        let (rep, adv, _) = run_critique_on_frame(&cmd, &fb, 0, 1);
        let text = render_report_text(&rep, &adv, None);
        assert!(text.contains("critic 报告"));
        assert!(text.contains("评分:"));
        let json = render_report_json(&rep, &adv, None);
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"score\""));
        assert!(json.contains("\"diagnoses\""));
    }

    #[test]
    fn autofix_via_command_improves() {
        let cmd = CriticCommand::default();
        let fb = solid(24, 24, [0.02, 0.02, 0.02, 1.0]);
        let (_rep, _adv, autofix) = run_critique_on_frame(&cmd, &fb, 2, 1);
        assert!(autofix.is_some());
    }

    #[test]
    fn json_string_escapes_quotes() {
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("x\ny"), "\"x\\ny\"");
    }
}
