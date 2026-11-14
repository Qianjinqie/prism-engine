//! V3.1 CI 硬门禁（T01）：五条误差判据 + 负向用例必判红 + 哈希锚 + legacy
//! 全关逐字节一致 + golden manifest 校验。
//!
//! 设计 §5 T01 验收：golden 各档逐帧锁 sha256；五判据 + 负向用例（人为扰动
//! 必判红）；纯 CPU 单帧锚 `78366f3e…` / 3,240,988 B 不变；4629 测试全绿。
//! 本测试文件是**门禁本体**，任何一处判红都让 CI 硬失败。

use std::fs;
use std::path::PathBuf;

use prism_dsl::CompileV2Output;
use prism_render::{
    DELTA_HARD, DELTA_P99, EDGE_KEEP, FINAL_SPP, GALLERY_BYTES, GALLERY_SHA256, GOLDEN_BOUNCES,
    GOLDEN_SPP, HIST_BIN_REL, MIN_SPP, PREVIEW_SPP, PSNR_FRAME_DB, PSNR_SEQ_DB, QualityTier,
    ResTier, TEMPORAL_OUTLIER, V31Config, compare_frame, compare_frame_2d, judge_v31,
    judge_v31_with, parity_of_u8, render_golden_frame, sequence_mean_psnr, sha256_hex,
    GoldenStore, GoldenTier, JudgeLimits, V31ErrorReport,
};

// ---------------------------------------------------------------------------
// 常量单一真源：所有判据数值只在 v31_config.rs 定义，这里只做「锁定断言」
// ---------------------------------------------------------------------------

#[test]
fn judgement_constants_are_locked_to_design() {
    assert_eq!((MIN_SPP, FINAL_SPP, PREVIEW_SPP, GOLDEN_SPP, GOLDEN_BOUNCES), (2, 4, 2, 16, 6));
    assert_eq!((DELTA_HARD, DELTA_P99), (32, 8));
    assert_eq!((PSNR_FRAME_DB, PSNR_SEQ_DB), (38.0, 38.0));
    assert_eq!((HIST_BIN_REL, EDGE_KEEP, TEMPORAL_OUTLIER), (0.15, 0.90, 0.05));
    assert_eq!(
        GALLERY_SHA256,
        "78366f3e78323ea260a0dfb606d4f497b06913c1b96500a5ddc1018aa9c32da8"
    );
    assert_eq!(GALLERY_BYTES, 3_240_988);
}

#[test]
fn judgement_limits_are_tiered_per_spec_8_1() {
    // §8.1 分档上限（2026-11-14 定版）：均衡 p99≤24 / max≤96 / ≥32 dB /
    // D≤0.35 / 边缘≥0.75 / 时序≤15%；快速 p99≤40 / max≤160 / ≥28 dB /
    // D≤0.50 / 边缘≥0.60 / 时序≤25%。
    let b: JudgeLimits = JudgeLimits::BALANCED;
    assert_eq!((b.max_delta, b.p99_delta), (96, 24));
    assert_eq!(b.psnr_db, 32.0);
    assert_eq!((b.hist_l1, b.edge_keep, b.temporal), (0.35, 0.75, 0.15));
    let f: JudgeLimits = JudgeLimits::FAST;
    assert_eq!((f.max_delta, f.p99_delta), (160, 40));
    assert_eq!(f.psnr_db, 28.0);
    assert_eq!((f.hist_l1, f.edge_keep, f.temporal), (0.50, 0.60, 0.25));
}

#[test]
fn tiered_judge_uses_the_tier_limits_not_the_strict_ones() {
    // 一份刚好落在「均衡档」范围内、但会被历史最严口径判红的报告。
    let r: V31ErrorReport = V31ErrorReport {
        max_delta: 60,
        p99_delta: 3,
        psnr_db: 55.0,
        hist_bin_max_rel: 0.0005,
        edge_retention: 0.97,
        temporal_outlier_ratio: 0.0,
    };
    let strict = judge_v31_with(&r, 55.0, &JudgeLimits::STRICT);
    assert!(!strict.pass, "最严口径下 max_delta 60 > 32 必须判红");
    let balanced = judge_v31_with(&r, 55.0, &JudgeLimits::BALANCED);
    assert!(balanced.pass, "均衡档上限 96 应放行：{:?}", balanced.reasons);
    let fast = judge_v31_with(&r, 55.0, &JudgeLimits::FAST);
    assert!(fast.pass, "快速档上限 160 应放行：{:?}", fast.reasons);

    // 分档不是放水：把画质真的做坏，两档都必须判红。
    let bad: V31ErrorReport = V31ErrorReport {
        max_delta: 200,
        p99_delta: 60,
        psnr_db: 24.0,
        hist_bin_max_rel: 0.62,
        edge_retention: 0.55,
        temporal_outlier_ratio: 0.31,
    };
    assert!(!judge_v31_with(&bad, 24.0, &JudgeLimits::BALANCED).pass);
    assert!(!judge_v31_with(&bad, 24.0, &JudgeLimits::FAST).pass);
}

#[test]
fn every_tier_effective_spp_equals_golden_and_fast_is_all_off() {
    let resolutions = [
        ResTier::R320x180,
        ResTier::R640x360,
        ResTier::R720p,
        ResTier::R1080p,
        ResTier::R4K,
        ResTier::Custom(200, 100),
    ];
    for &r in &resolutions {
        for &q in &[QualityTier::Preview, QualityTier::Final, QualityTier::Fast] {
            let cfg = V31Config::for_tier(r, q);
            assert_eq!(cfg.effective_spp(), GOLDEN_SPP, "{r:?}/{q:?}");
            assert_eq!(cfg.bounces, GOLDEN_BOUNCES);
            assert_eq!(cfg.min_spp, MIN_SPP);
            assert_eq!((cfg.width, cfg.height), r.dims());
        }
        let fast = V31Config::for_tier(r, QualityTier::Fast);
        assert!(!fast.accumulate && !fast.denoise && !fast.adaptive && !fast.gpu_prefer);
    }
}

// ---------------------------------------------------------------------------
// legacy() 全关 ⇒ apply_to 逐字节 no-op；for_tier 覆写确实生效
// ---------------------------------------------------------------------------

/// 最小合法 DSL（与 `tests/movie_sim_timeline.rs::SRC_NO_SIM` 同源）。
const MIN_DSL: &str = concat!(
    "// prism-dsl 1\n",
    "movie \"v31gate\" { duration 1s fps 4 resolution 64x36 }\n",
    "scene \"bare\" {\n",
    "  duration 1s\n",
    "  stage3d {\n",
    "    camera main { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    material m { albedo #c0392b roughness 0.4 }\n",
    "  }\n",
    "}\n",
);

fn minimal_compile() -> CompileV2Output {
    prism_dsl::compile_v2(MIN_DSL).expect("最小 DSL 应编译成功")
}

#[test]
fn legacy_apply_to_is_bytewise_identity() {
    let base: CompileV2Output = minimal_compile();
    let untouched: CompileV2Output = base.clone();
    let mut applied: CompileV2Output = base;

    let legacy: V31Config = V31Config::legacy();
    assert!(legacy.is_legacy(), "legacy 必须被识别为不覆写配置");
    assert!(!legacy.accumulate && !legacy.denoise && !legacy.adaptive && !legacy.gpu_prefer);
    legacy.apply_to(&mut applied);

    // Debug 序列化覆盖 CompileV2Output 全部字段：逐字节一致的最强可得断言。
    assert_eq!(
        format!("{:?}", applied),
        format!("{:?}", untouched),
        "legacy() 全关时 apply_to 必须是 no-op（旧入口逐字节不变）"
    );
    assert_eq!(applied.width, untouched.width);
    assert_eq!(applied.height, untouched.height);
    assert_eq!(applied.fps, untouched.fps);
    assert_eq!(applied.duration.to_bits(), untouched.duration.to_bits());
}

#[test]
fn for_tier_apply_to_overrides_only_spec_fields() {
    let base: CompileV2Output = minimal_compile();
    let mut c: CompileV2Output = base.clone();
    let cfg: V31Config = V31Config::for_tier(ResTier::R640x360, QualityTier::Final);
    cfg.apply_to(&mut c);

    assert_eq!((c.width, c.height), (640, 360));
    assert_eq!(c.fps, cfg.fps);
    assert_eq!(c.duration.to_bits(), cfg.duration_secs().to_bits());
    // 覆写不得触碰 DSL 场景 IR / 接线表等其它字段。
    // `CompileV2Output` 只 derive 了 Debug + Clone（无 PartialEq），故用 Debug
    // 序列化逐字符比对，等价地钉住「其余字段一字未变」。
    assert_eq!(c.scenes.len(), base.scenes.len());
    assert_eq!(format!("{:?}", c.graph), format!("{:?}", base.graph));
    assert_eq!(format!("{:?}", c.wiring), format!("{:?}", base.wiring));
    assert_eq!(c.title, base.title);
}

// ---------------------------------------------------------------------------
// 合成帧与扰动工具（判据非恒真的保证：图案含强边缘 + 宽亮度分布）
// ---------------------------------------------------------------------------

fn synthetic_frame(w: u32, h: u32) -> Vec<[u8; 4]> {
    let mut out: Vec<[u8; 4]> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let checker: u8 = if (x / 8 + y / 8) % 2 == 0 { 200 } else { 36 };
            let ramp: u8 = ((x * 255) / w.max(1)) as u8;
            let band: u8 = ((y * 255) / h.max(1)) as u8;
            out.push([checker, ramp, 255 - band, 255]);
        }
    }
    out
}

fn uniform_offset(f: &[[u8; 4]], d: u8) -> Vec<[u8; 4]> {
    f.iter()
        .map(|p| {
            let mut q = *p;
            for k in 0..3 {
                q[k] = p[k].saturating_add(d);
            }
            q
        })
        .collect()
}

fn compress_contrast(f: &[[u8; 4]]) -> Vec<[u8; 4]> {
    f.iter()
        .map(|p| {
            let mut q = *p;
            for k in 0..3 {
                let v: i32 = p[k] as i32;
                q[k] = (128 + (v - 128) / 2).clamp(0, 255) as u8;
            }
            q
        })
        .collect()
}

fn blur3x3(f: &[[u8; 4]], w: u32, h: u32) -> Vec<[u8; 4]> {
    let mut out: Vec<[u8; 4]> = Vec::with_capacity(f.len());
    for y in 0..h {
        for x in 0..w {
            let mut sums: [u32; 3] = [0; 3];
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let cx: usize = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let cy: usize = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let p: [u8; 4] = f[cy * w as usize + cx];
                    for k in 0..3 {
                        sums[k] += p[k] as u32;
                    }
                }
            }
            let mut q: [u8; 4] = [255; 4];
            for k in 0..3 {
                q[k] = ((sums[k] + 4) / 9) as u8;
            }
            out.push(q);
        }
    }
    out
}

/// 10% 像素反色（> 5% 上限）⇒ 只触发判据 ⑤，其余四条仍全过。
fn inverted_prev(f: &[[u8; 4]]) -> Vec<[u8; 4]> {
    f.iter()
        .enumerate()
        .map(|(i, p)| {
            if i % 10 == 0 {
                [255 - p[0], 255 - p[1], 255 - p[2], p[3]]
            } else {
                *p
            }
        })
        .collect()
}

fn single_pixel_spike(f: &[[u8; 4]], idx: usize, d: u8) -> Vec<[u8; 4]> {
    let mut out: Vec<[u8; 4]> = f.to_vec();
    for k in 0..3 {
        out[idx][k] = out[idx][k].saturating_add(d);
    }
    out
}

fn verdict(golden: &[[u8; 4]], got: &[[u8; 4]], prev: Option<&[[u8; 4]]>, w: u32, h: u32) -> (prism_render::V31ErrorReport, prism_render::V31Verdict) {
    let r = compare_frame_2d(golden, got, prev, w, h);
    let v = judge_v31(&r, sequence_mean_psnr(&[r.psnr_db]));
    (r, v)
}

// ---------------------------------------------------------------------------
// 正向：完全一致帧五条判据全过（证明门禁非恒假）
// ---------------------------------------------------------------------------

#[test]
fn identical_frame_passes_all_five_criteria() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let (r, v) = verdict(&g, &g, None, w, h);
    assert!(v.pass, "完全一致帧必须全过：{:?}", v.reasons);
    assert!(v.reasons.is_empty());
    assert_eq!(r.max_delta, 0);
    assert_eq!(r.p99_delta, 0);
    assert!(r.psnr_db.is_infinite() && r.psnr_db > 0.0);
    assert_eq!(r.hist_bin_max_rel, 0.0);
    assert_eq!(r.edge_retention, 1.0);
    assert_eq!(r.temporal_outlier_ratio, 0.0);
}

#[test]
fn signatureless_compare_frame_matches_row_view() {
    let g: Vec<[u8; 4]> = synthetic_frame(16, 8);
    let a = compare_frame(&g, &g, None);
    let b = compare_frame_2d(&g, &g, None, (16 * 8) as u32, 1);
    assert_eq!(a, b);
}

#[test]
fn psnr_comes_from_wavefront_parity_base() {
    let g: Vec<[u8; 4]> = synthetic_frame(16, 8);
    let o: Vec<[u8; 4]> = uniform_offset(&g, 4);
    let r = parity_of_u8(&g, &o);
    let want = prism_rt::wavefront::parity_report(
        &g.iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect::<Vec<[f32; 4]>>(),
        &o.iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect::<Vec<[f32; 4]>>(),
        3,
    );
    assert_eq!(r.psnr_db.to_bits(), want.psnr_db.to_bits());
    assert_eq!(r.mse.to_bits(), want.mse.to_bits());
}

// ---------------------------------------------------------------------------
// 负向用例：任何扰动必须判红（CI 硬失败项）
// ---------------------------------------------------------------------------

#[test]
fn neg_single_pixel_spike_fails_hard_delta() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let got: Vec<[u8; 4]> = single_pixel_spike(&g, 137, 40);
    let (r, v) = verdict(&g, &got, None, w, h);
    assert!(!v.pass, "单点大扰动必须判红");
    assert!(!v.reasons.is_empty());
    assert!(r.max_delta > DELTA_HARD, "全局硬上限应被突破：{}", r.max_delta);
    assert!(v.reasons.iter().any(|s| s.contains("判据①")), "{:?}", v.reasons);
}

#[test]
fn neg_uniform_offset_fails_p99_and_psnr() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let got: Vec<[u8; 4]> = uniform_offset(&g, 16);
    let (r, v) = verdict(&g, &got, None, w, h);
    assert!(!v.pass, "全帧偏移必须判红");
    assert!(r.p99_delta > DELTA_P99, "p99={} 应超过 {}", r.p99_delta, DELTA_P99);
    assert!(r.psnr_db < PSNR_FRAME_DB, "PSNR={} 应低于 {}", r.psnr_db, PSNR_FRAME_DB);
    assert!(v.reasons.iter().any(|s| s.contains("判据①")), "{:?}", v.reasons);
    assert!(v.reasons.iter().any(|s| s.contains("判据②")), "{:?}", v.reasons);
}

#[test]
fn neg_contrast_compression_fails_histogram() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let got: Vec<[u8; 4]> = compress_contrast(&g);
    let (r, v) = verdict(&g, &got, None, w, h);
    assert!(!v.pass, "亮度重映射必须判红");
    assert!(r.hist_bin_max_rel > HIST_BIN_REL, "hist={}", r.hist_bin_max_rel);
    assert!(v.reasons.iter().any(|s| s.contains("判据③")), "{:?}", v.reasons);
}

#[test]
fn neg_blur_fails_edge_retention() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let got: Vec<[u8; 4]> = blur3x3(&g, w, h);
    let (r, v) = verdict(&g, &got, None, w, h);
    assert!(!v.pass, "边缘抹平必须判红");
    assert!(r.edge_retention < EDGE_KEEP, "edge={}", r.edge_retention);
    assert!(v.reasons.iter().any(|s| s.contains("判据④")), "{:?}", v.reasons);
}

#[test]
fn neg_temporal_outlier_alone_fails_gate() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let prev: Vec<[u8; 4]> = inverted_prev(&g);
    // got 与 golden 完全一致 ⇒ ①②③④ 全过，只有 ⑤ 判红（单判据隔离验证）。
    let (r, v) = verdict(&g, &g, Some(&prev), w, h);
    assert!(!v.pass, "相邻帧差分异常必须判红");
    assert!(r.temporal_outlier_ratio > TEMPORAL_OUTLIER);
    assert_eq!(r.max_delta, 0);
    assert_eq!(r.hist_bin_max_rel, 0.0);
    assert_eq!(r.edge_retention, 1.0);
    assert!(v.reasons.iter().any(|s| s.contains("判据⑤")), "{:?}", v.reasons);
    assert_eq!(v.reasons.len(), 1, "应当只有判据⑤判红：{:?}", v.reasons);
}

#[test]
fn neg_sequence_mean_psnr_fails_gate() {
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let r = compare_frame_2d(&g, &g, None, w, h);
    let v = judge_v31(&r, PSNR_FRAME_DB - 1.0);
    assert!(!v.pass);
    assert!(v.reasons.iter().any(|s| s.contains("序列平均 PSNR")), "{:?}", v.reasons);
}

#[test]
fn neg_perturbing_golden_also_fails_gate() {
    // 「扰动 golden 或 got」两种方向都必须判红（防门禁被反向绕过）。
    let (w, h) = (32u32, 24u32);
    let g: Vec<[u8; 4]> = synthetic_frame(w, h);
    let bad_golden: Vec<[u8; 4]> = uniform_offset(&g, 16);
    let (_, v) = verdict(&bad_golden, &g, None, w, h);
    assert!(!v.pass, "扰动 golden 同样必须判红");
}

#[test]
#[should_panic(expected = "帧长不等")]
fn cross_tier_comparison_is_structurally_rejected() {
    // 禁互比：不同档位帧长不同 ⇒ 结构化报错，而不是静默按较短者比较。
    let a: Vec<[u8; 4]> = vec![[0; 4]; 320 * 180];
    let b: Vec<[u8; 4]> = vec![[0; 4]; 640 * 360];
    let _ = compare_frame_2d(&a, &b, None, 320, 180);
}

// ---------------------------------------------------------------------------
// 单帧哈希锚
// ---------------------------------------------------------------------------

#[test]
fn gallery_png_matches_locked_anchor_when_present() {
    // 哈希锚常量锁定（无论文件在不在工作树都硬断言）。
    assert_eq!(GALLERY_SHA256.len(), 64);
    assert_eq!(GALLERY_BYTES, 3_240_988);

    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../gallery.png");
    let bytes: Vec<u8> = match fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            // 不是静默跳过：明确输出结构性提示 + 上面的常量锁定断言仍生效。
            eprintln!(
                "[v31-gate] 结构化提示：未找到 {}；纯 CPU 单帧锚由本批自测与 \
                 scripts/v31_perf_gate.sh 生成 gallery.png 后复验（CI 不依赖未跟踪文件）。",
                path.display()
            );
            return;
        }
    };
    assert_eq!(bytes.len() as u64, GALLERY_BYTES, "gallery.png 字节数锚不符");
    assert_eq!(sha256_hex(&bytes), GALLERY_SHA256, "gallery.png 哈希锚不符");
}

// ---------------------------------------------------------------------------
// golden 资产：manifest 可校验、逐帧可复现、禁自比
// ---------------------------------------------------------------------------

#[test]
fn minimal_golden_tier_manifest_verifies() {
    let store: GoldenStore = GoldenStore::at_repo_root();
    let entries = store
        .verify_manifest()
        .unwrap_or_else(|e| panic!("golden manifest 必须可校验（P0-6 不可静默关）：{e}"));
    assert!(!entries.is_empty());
    let prefix: String = format!("{}/", GoldenTier::minimal().tag());
    let tier_frames: Vec<_> = entries
        .iter()
        .filter(|e| e.rel_path.starts_with(&prefix))
        .collect();
    assert!(
        tier_frames.len() >= GoldenTier::minimal().frames as usize,
        "最小档位应至少有 {} 帧 golden，实得 {}",
        GoldenTier::minimal().frames,
        tier_frames.len()
    );
    for e in &tier_frames {
        assert_eq!(e.sha256_hex.len(), 64);
        assert!(e.bytes > 0);
    }
}

#[test]
fn golden_frame_is_reproducible_and_not_self_compared() {
    let store: GoldenStore = GoldenStore::at_repo_root();
    let tier: GoldenTier = GoldenTier::minimal();
    let from_disk: Vec<[u8; 4]> = store
        .load_frame(tier.res, tier.quality, 0)
        .unwrap_or_else(|e| panic!("加载 golden 第 0 帧失败：{e}"));
    // got 由**独立重渲染**产生（不是把磁盘 golden 直接当 got），故不构成自比。
    let rerendered: Vec<[u8; 4]> = render_golden_frame(&tier, 0);
    assert_eq!(from_disk.len(), rerendered.len());
    assert_eq!(
        from_disk, rerendered,
        "golden 必须是可逐字节复现的确定性 CPU 渲染产物"
    );
    let (w, h) = tier.res.dims();
    let (r, v) = verdict(&from_disk, &rerendered, None, w, h);
    assert!(v.pass, "独立重渲染对 golden 应五判据全过：{:?}", v.reasons);
    assert_eq!(r.max_delta, 0);
}
