//! V3.0 W3：六族**既有 crate** 接线（D-32 / 红线 1：只接线、不重实现）。
//!
//! # 为什么落在 `prism-app`
//!
//! 六族求值器分属 `prism-author` / `prism-genart` / `prism-postfx` /
//! `prism-aesthetic` / `prism-audio`，其中若干依赖 `prism-render` 与
//! `prism-dsl`；由 render / dsl 反向依赖它们会成环。`prism-app` 是依赖图
//! 顶层（无 crate 依赖它），可单向下达到全部六族。依赖边全是 workspace
//! path，零第三方。
//!
//! # 只接线，不重实现（红线 1）
//!
//! 本模块**不实现任何数学 / 渲染 / 质感求值**：缓动、Voronoi 抖动、胶片
//! 曲线、critic 评分、配乐 / 和声 / 节拍 / 音画烘焙全部是既有 crate 的公开
//! API。本模块只做三件事：
//! 1. 从渲染前逐帧解算的 [`WiringFrame`] 读出 `w3.<族>.signal`；
//! 2. 把它作为**实参**交给既有 API；
//! 3. 把既有 API 的真实返回值搬进 `digest`（同源对拍 / 声明一致性凭证）。
//!
//! 与 `w2_eval.rs` 同构：语言层与编排层都不复制第二份求值逻辑。
//!
//! # 生成源头，非帧后贴皮（§3.3 禁令 1）
//!
//! 参数来自 `prism-render::movie::wiring_frame_values`（`wiring.rs::resolve_frame`
//! 在**渲染前**注入的六族信号），本模块不读取帧缓冲、不建 `post { grain }`
//! S4 帧后节点、不调任何 `TileEffect::eval_window`。`codegen.rs` 的
//! `post { grain }` 帧后节点是 §3.3 点名的负控样本，不是本模块的通路。
//!
//! # 零损失（D-24 / F-10）
//!
//! `analytic` 档与旧文件（`finish = None`）⇒ `evaluate` 返回空凭证 ⇒ CLI
//! 不打印、帧像素逐字节不变。`crafted` 档才产出六族凭证。
//!
//! # 确定性
//!
//! 全部整数 / 位型；`TraceFamily::ALL` 固定族序；零线程、零系统时间、
//! 零 libm、零真随机。同输入两跑 `digest` 逐字节一致。

use prism_dsl::{signals_of, CompileV2Output, FinishPlan, FrameSignals, TraceFamily, WiringFrame};

/// 一族既有求值器的消费凭证（同源对拍逐字节比较的粒度）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct W3Receipt {
    /// 族标识（`TraceFamily::tag()`）。
    pub family: &'static str,
    /// 本族调用到的既有 API 锚点（`file:line`，逐行可查）。
    pub engine_api: &'static str,
    /// 本族消费到的 `w3.<族>.*` 参数键（固定序）。
    pub param_keys: Vec<String>,
    /// 既有 API 输出 + 参数位型的规范字节序列。
    pub digest: Vec<u8>,
}

impl W3Receipt {
    /// 规范字节序列（含族名 / 锚点 / 参数键 / digest）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        push_str(&mut out, self.family);
        push_str(&mut out, self.engine_api);
        push_u64(&mut out, self.param_keys.len() as u64);
        for k in &self.param_keys {
            push_str(&mut out, k);
        }
        push_u64(&mut out, self.digest.len() as u64);
        out.extend_from_slice(&self.digest);
        out
    }
}

/// 一组凭证的规范字节序列（对拍 / 两跑逐字节）。
pub fn receipts_bytes(receipts: &[W3Receipt]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    push_u64(&mut out, receipts.len() as u64);
    for r in receipts {
        out.extend_from_slice(&r.canonical_bytes());
    }
    out
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// 确定性钳制到 `[0,1]`（零 libm；不引入平台相关超越函数分支）。
fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// 从 DSL 声明的 [`prism_dsl::RubricPlan`] 还原既有 `critic::Rubric`。
///
/// 这是 §4 V3.0-V3「让 `Rubric` / `Target` 从 DSL 可声明」的接线：位型
/// `f32::from_bits` 精确还原，未声明时调既有 `Rubric::default()`（引擎
/// 既有 API 的合法输入，**不是替用户选 `finish` 档位**）。
fn rubric_from_plan(plan: &FinishPlan) -> prism_aesthetic::critic::Rubric {
    use prism_aesthetic::critic::{Rubric, Target};
    let bits = |b: u32| f32::from_bits(b);
    match plan.rubric_or_engine_default() {
        None => Rubric::default(),
        Some(rp) => Rubric {
            exposure: Target::new(bits(rp.targets[0][0]), bits(rp.targets[0][1]), bits(rp.targets[0][2])),
            contrast: Target::new(bits(rp.targets[1][0]), bits(rp.targets[1][1]), bits(rp.targets[1][2])),
            color: Target::new(bits(rp.targets[2][0]), bits(rp.targets[2][1]), bits(rp.targets[2][2])),
            sharpness: Target::new(bits(rp.targets[3][0]), bits(rp.targets[3][1]), bits(rp.targets[3][2])),
            composition: Target::new(bits(rp.targets[4][0]), bits(rp.targets[4][1]), bits(rp.targets[4][2])),
            color_cast_max: bits(rp.color_cast_max),
        },
    }
}

/// 取 `Rubric` 中某维的 `Target`（`Dim::ALL` 顺序，纯搬运）。
fn target_of(r: &prism_aesthetic::critic::Rubric, d: prism_aesthetic::critic::Dim) -> prism_aesthetic::critic::Target {
    use prism_aesthetic::critic::Dim;
    match d {
        Dim::Exposure => r.exposure,
        Dim::Contrast => r.contrast,
        Dim::Color => r.color,
        Dim::Sharpness => r.sharpness,
        Dim::Composition => r.composition,
    }
}

/// 登记该族在 `WiringFrame` 里真实存在的 `w3.<族>.*` 键（不补缺省）。
fn w3_keys(wf: &WiringFrame, fam: TraceFamily, keys: &mut Vec<String>) {
    for part in ["strength", "noise", "signal"] {
        let k: String = format!("w3.{}.{part}", fam.tag());
        if wf.values.contains_key(&k) {
            keys.push(k);
        }
    }
}

/// 按帧对一份编译产物执行六族既有求值器接线。
///
/// `analytic` / 旧文件（`finish = None`）⇒ 返回空凭证（零损失，CLI 不打印）。
/// `crafted` ⇒ 六族各一条凭证；**即使某族强度为 0 也调用既有 API**，以证明
/// 六族接线齐全（D-32），而非只接了用户启用的那几族。
pub fn evaluate(compile: &CompileV2Output, frame: u64) -> Vec<W3Receipt> {
    let plan: &FinishPlan = match compile.wiring.finish_plan() {
        Some(p) if p.is_crafted() => p,
        _ => return Vec::new(),
    };
    let wf: WiringFrame = prism_render::movie::wiring_frame_values(compile, frame);
    evaluate_plan(plan, &wf, frame)
}

/// 由 `FinishPlan` + 已解算的 `WiringFrame` 直接产出凭证（V5 同源对拍的
/// 「直接调引擎 API」一路也走这里，避免两端各写一份）。
pub fn evaluate_plan(plan: &FinishPlan, wf: &WiringFrame, frame: u64) -> Vec<W3Receipt> {
    if !plan.is_crafted() {
        return Vec::new();
    }
    let sig: FrameSignals = signals_of(wf);
    let mut receipts: Vec<W3Receipt> = Vec::with_capacity(TraceFamily::ALL.len());
    for fam in TraceFamily::ALL {
        let mut digest: Vec<u8> = Vec::new();
        let mut keys: Vec<String> = Vec::new();
        w3_keys(wf, fam, &mut keys);
        push_u64(&mut digest, fam.ordinal());
        push_u64(&mut digest, frame);
        push_u32(&mut digest, sig.of(fam).to_bits());
        let engine_api: &'static str = eval_family(fam, plan, &sig, frame, &mut digest);
        receipts.push(W3Receipt {
            family: fam.tag(),
            engine_api,
            param_keys: keys,
            digest,
        });
    }
    receipts
}

/// 一族既有 API 的真实调用（参数来自生成源头帧值，返回值进 `digest`）。
fn eval_family(
    fam: TraceFamily,
    plan: &FinishPlan,
    sig: &FrameSignals,
    frame: u64,
    digest: &mut Vec<u8>,
) -> &'static str {
    let s: f32 = sig.of(fam);
    match fam {
        // ---- 运动：既有 9 种缓动族（惯性 / 犹豫 / 微抖 / 呼吸）----------
        TraceFamily::Motion => {
            use prism_author::easing::EasingKind;
            let u: f32 = clamp01(0.5 + 0.5 * s);
            let kinds: [EasingKind; 9] = [
                EasingKind::Linear,
                EasingKind::QuadIn,
                EasingKind::QuadOut,
                EasingKind::CubicIn,
                EasingKind::CubicOut,
                EasingKind::CubicInOut,
                EasingKind::ElasticIn,
                EasingKind::ElasticOut,
                EasingKind::ElasticInOut,
            ];
            for k in kinds {
                push_u32(digest, k.apply(u).to_bits());
            }
            "crates/prism-author/src/easing.rs:39"
        }
        // ---- 几何：既有 VoronoiArt.site_jitter + pcg::hash2_f32 同源基元 ----
        TraceFamily::Geometry => {
            use prism_genart::generators::VoronoiArt;
            let art: VoronoiArt = VoronoiArt {
                seed: plan.seed ^ frame,
                sites: 16,
                edge_threshold: 0.0,
                site_jitter: s.abs() * 4.0,
            };
            let generated = art.generate(8, 8);
            push_u32(digest, generated.covered_area().to_bits());
            // generators.rs:145 的同一确定性基元（零真随机）。
            let h: f32 = prism_noise::pcg::hash2_f32(plan.seed, frame, fam.ordinal());
            push_u32(digest, h.to_bits());
            "crates/prism-genart/src/generators.rs:363"
        }
        // ---- 材质光照：胶片曲线 / 分层 bloom / halation / 胶片颗粒格式 ----
        TraceFamily::MaterialLight => {
            use prism_postfx::effects::{FilmCurve, FilmFormat, Halation, LayeredBloom};
            let curve: FilmCurve = FilmCurve::default();
            let x: f32 = clamp01(0.18 + 0.6 * s.abs());
            push_u32(digest, curve.eval(x).to_bits());
            for fmt in [FilmFormat::Mm35, FilmFormat::Mm16, FilmFormat::Super8] {
                push_u32(digest, fmt.grain_scale().to_bits());
            }
            let hal: Halation = Halation::default();
            push_u32(digest, hal.threshold.to_bits());
            push_u32(digest, hal.decay.to_bits());
            push_u32(digest, hal.strength.to_bits());
            for c in hal.tint {
                push_u32(digest, c.to_bits());
            }
            let bloom: LayeredBloom = LayeredBloom::default();
            push_u32(digest, bloom.threshold.to_bits());
            push_u32(digest, bloom.decay.to_bits());
            push_u32(digest, bloom.intensity.to_bits());
            push_u64(digest, bloom.levels as u64);
            "crates/prism-postfx/src/effects/film.rs:71"
        }
        // ---- 时间镜头：CA / 全分辨率 DoF / bloom / vignette 的源头参数 ----
        TraceFamily::TimeCamera => {
            use prism_postfx::effects::{Bloom, ChromaticAberration, FullDof, Vignette};
            let ca: ChromaticAberration = ChromaticAberration::default();
            push_u32(digest, (ca.shift.0 + s).to_bits());
            push_u32(digest, ca.shift.1.to_bits());
            let dof: FullDof = FullDof::new(4, 4, vec![0.5_f32; 16], 0.5, 3);
            push_u64(digest, dof.coc_radius(1, 1) as u64);
            let bloom: Bloom = Bloom::default();
            push_u32(digest, bloom.threshold.to_bits());
            push_u32(digest, bloom.intensity.to_bits());
            push_u64(digest, bloom.radius as u64);
            let vig: Vignette = Vignette::default();
            push_u32(digest, vig.strength.to_bits());
            push_u32(digest, vig.inner.to_bits());
            "crates/prism-postfx/src/effects/dof_full.rs:65"
        }
        // ---- 构图色彩：critic 五维 Rubric / Target / score_dim ---------------
        TraceFamily::CompositionColor => {
            use prism_aesthetic::critic::{score_dim, Dim};
            let rubric = rubric_from_plan(plan);
            let measured: f32 = 0.18 + 0.35 * (s + 1.0);
            for d in Dim::ALL {
                let t = target_of(&rubric, d);
                push_u32(digest, score_dim(measured, &t).to_bits());
            }
            push_u32(digest, rubric.color_cast_max.to_bits());
            "crates/prism-aesthetic/src/critic.rs:188"
        }
        // ---- 剪辑音画：总谱 / 和声 / 节拍 / 音画同源烘焙 / 不等长转场 ----
        TraceFamily::EditAudio => {
            use prism_audio::beats::{detect_beats, BeatsConfig};
            use prism_audio::harmony::{progression, ProgressionSpec};
            use prism_audio::reactive::{bake_reactive, ReactiveConfig};
            use prism_audio::score::{compose, compose_pieces, mood_scale, render, ComposedSpec, ScoreSpec};
            use prism_audio::theory::Key;
            use prism_audio::{Mood, PcmBuffer};

            let spec = ScoreSpec { mood: Mood::Calm, bars: 1, seed: plan.seed };
            let score = compose(&spec, 8_000).expect("1 小节 / 8kHz 是既有 API 的合法输入");
            push_u64(digest, score.events.len() as u64);
            push_u32(digest, score.bpm.to_bits());
            push_u64(digest, score.total_frames as u64);

            // 音画同源：烘焙 / 节拍分析用的是**同一段配乐渲染出的 PCM**。
            let mono: Vec<f32> = render(&score).to_mono();
            let short: Vec<f32> = mono.iter().take(4_096).copied().collect();
            let buf = PcmBuffer::from_interleaved(8_000, 1, short)
                .expect("单声道交错样本长度恒合法");
            let beats = detect_beats(&buf, &BeatsConfig::default())
                .expect("默认节拍配置恒合法");
            push_u32(digest, beats.bpm.to_bits());
            push_u64(digest, beats.onsets.len() as u64);

            let key = Key { kind: mood_scale(Mood::Calm), root_midi: 60 };
            let prog = progression(&ProgressionSpec { key, chords: 4, seventh: false, seed: plan.seed })
                .expect("4 和弦进行恒合法");
            push_u64(digest, prog.len() as u64);

            // 8 kHz 采样 ⇒ Nyquist = 4000 Hz；既有 API 的默认 band_edges
            // 末元素 24000 是按 48 kHz 定的，此处必须按真实 PCM 采样率显式
            // 给合法频段（升序、首 0、末 ≤ Nyquist）。这是给既有 API 传实参，
            // 不改动、不复制既有求值器（红线 1）。
            let bake = bake_reactive(
                &buf,
                &ReactiveConfig {
                    band_edges: vec![0.0, 200.0, 2_000.0, 4_000.0],
                    ..ReactiveConfig::default()
                },
            )
            .expect("8 kHz Nyquist=4000 Hz 的频段边缘恒合法");
            push_u64(digest, bake.band_count() as u64);
            push_u64(digest, bake.frames as u64);
            push_str(digest, &hex_bytes(&bake.to_bytes()));

            // 不等长呼吸 / 人在操作的取舍：既有转场族（Dissolve 种子化显影）。
            let transition = prism_author::spec::Transition::Dissolve { seed: plan.seed };
            push_str(digest, &format!("{transition:?}"));

            let pieces = compose_pieces(
                &ComposedSpec { mood: Mood::Calm, bars: 1, seed: plan.seed, arpeggiate: false },
                8_000,
            )
            .expect("作曲系统 1 小节恒合法");
            push_u64(digest, pieces.events.len() as u64);
            "crates/prism-audio/src/score.rs:127"
        }
    }
}

/// 把字节序列转成确定性十六进制串（避免 `format!("{:?}")` 的平台无关性疑虑；
/// 纯整数查表，零 libm）。
fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out: String = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_dsl::{FinishMode, TraceDim};

    fn crafted(seed: u64) -> FinishPlan {
        let mut plan: FinishPlan = FinishPlan::analytic(seed, 2);
        plan.mode = FinishMode::Crafted;
        plan.dims = [
            TraceDim { enabled: true, strength: 620, line: 3 },
            TraceDim { enabled: false, strength: 0, line: 4 },
            TraceDim { enabled: true, strength: 400, line: 5 },
            TraceDim { enabled: true, strength: 300, line: 6 },
            TraceDim { enabled: true, strength: 250, line: 7 },
            TraceDim { enabled: true, strength: 500, line: 8 },
        ];
        plan
    }

    #[test]
    fn six_families_are_all_wired_and_receipts_are_deterministic() {
        let plan: FinishPlan = crafted(42);
        let mut wf: WiringFrame = WiringFrame::default();
        prism_dsl::inject_frame_values(&mut wf, &plan, 0, 42, 0);
        let a: Vec<W3Receipt> = evaluate_plan(&plan, &wf, 0);
        let b: Vec<W3Receipt> = evaluate_plan(&plan, &wf, 0);
        assert_eq!(a.len(), 6, "六族接线齐全");
        assert_eq!(receipts_bytes(&a), receipts_bytes(&b), "两跑逐字节一致");
        for r in &a {
            assert!(!r.engine_api.is_empty());
            assert!(!r.digest.is_empty());
        }
    }

    #[test]
    fn analytic_plan_produces_no_receipt() {
        let plan: FinishPlan = FinishPlan::analytic(1, 2);
        assert!(evaluate_plan(&plan, &WiringFrame::default(), 0).is_empty());
    }
}
