//! V3.1 时域重投影验收（S1–S6 的画质与反作弊硬证据）。
//!
//! 驱动**出片链路同一实现** [`TemporalReprojector`]，逐帧生成独立 golden
//! （`independent_golden_u8`：该帧自身 req、逐帧独立重渲 16 spp、不锁存 /
//! 不自比 / 不消费 got 缓存），过 `v31_gate` 五条判据（判据 ⑤ 用
//! `compare_frame_2d_expected` 的「相对 golden 期望时序变化」口径，真运动
//! 不误判、重投影闪烁才判红）。
//!
//! 四组片源：
//!  - `Static`   静止（固定机位 + 固定几何）⇒ 复用比例 > 0（真加速）。
//!  - `Orbit`    相机绕行、几何固定（w2d_showcase 3D 基帧形态）⇒ 全失效回填。
//!  - `Object`   固定机位 + 物体半径脉动 ⇒ 背景复用 / 球面失效（逐像素判定）。
//!  - `Full`     S1 真运动片源（相机绕行 + 半径脉动遮挡，= examples/v31_motion.prism）。
//!
//! 反作弊硬断言：相机运动片源**表面像素**不得复用任何陈旧历史，背景像素
//! 经远点重投影 + 严格方向一致性后可无损复用（复用比例 > 0）；每帧真实新样本
//! > 0（禁止零新样本帧）；静止片源必须真的复用；物体运动必须出现「部分复用 +
//! 部分失效」。四档复用比例 new_spp ∈ {16,8,4} 全部逐帧过五条，且 got 与该帧
//! 独立 golden 逐位一致（不靠静止片让结果好看）。

use prism_gpu::{
    AestheticReq, CameraReq, LightReq, SceneGeoReq, SceneRenderRequest, SphereReq, NO_MAPS,
};
use prism_math::Vec3;
use prism_render::v31_gate::{
    compare_frame_2d_expected, judge_v31, sequence_mean_psnr, V31ErrorReport,
};
use prism_render::{independent_golden_u8, TemporalReprojector};

// ---- S1 运动序列（与 examples/v31_motion.prism 同几何）--------------------
// 相机沿 orbit_center 水平圆轨道逐帧绕行（eval_camera_orbit 口径），
// 球体半径按关键帧 0.35→2.3→0.35→2.3→0.35 脉动（物体缩放 / 遮挡变化）。

const ORBIT_RADIUS: f32 = 5.0;
const DURATION: f32 = 2.0;
const FPS: u32 = 12;

fn lerp_keys(t: f32) -> f32 {
    // key 0s 0.35 | 0.5s 2.3 | 1s 0.35 | 1.5s 2.3 | 2s 0.35，linear。
    let keys: [(f32, f32); 5] = [(0.0, 0.35), (0.5, 2.3), (1.0, 0.35), (1.5, 2.3), (2.0, 0.35)];
    for i in 0..keys.len() - 1 {
        let (t0, v0) = keys[i];
        let (t1, v1) = keys[i + 1];
        if t <= t1 {
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return v0 + (v1 - v0) * f;
        }
    }
    keys[keys.len() - 1].1
}

/// 四组片源的「相机运动 / 物体运动」开关。
#[derive(Clone, Copy)]
struct Motion {
    cam: bool,
    obj: bool,
}

const STATIC: Motion = Motion { cam: false, obj: false };
const ORBIT: Motion = Motion { cam: true, obj: false };
const OBJECT: Motion = Motion { cam: false, obj: true };
const FULL: Motion = Motion { cam: true, obj: true };

fn req_at(w: u32, h: u32, frame: u32, m: Motion) -> SceneRenderRequest {
    let t = frame as f32 / FPS as f32;
    let angle = (t / DURATION) * std::f32::consts::TAU;
    let eye = if m.cam {
        Vec3::new(angle.sin() * ORBIT_RADIUS, 0.0, -angle.cos() * ORBIT_RADIUS)
    } else {
        Vec3::new(0.0, 0.0, -ORBIT_RADIUS)
    };
    let radius = if m.obj { lerp_keys(t) } else { 1.2 };
    SceneRenderRequest {
        width: w,
        height: h,
        camera: CameraReq {
            eye,
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 55f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![LightReq {
            direction: Vec3::new(1.0, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
            intensity: 3.0,
        }],
        spheres: vec![SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius,
            albedo: Vec3::new(0.75, 0.23, 0.17),
            roughness: 0.35,
            metalness: 0.05,
            maps: NO_MAPS,
        }],
        // 中性 tone-map：got 与 golden 同 aesthetic，五判据与 aesthetic 无关。
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        curves: Vec::new(),
        geo: SceneGeoReq::default(),
    }
}

struct SeqResult {
    reports: Vec<V31ErrorReport>,
    new_samples: Vec<u64>,
    motions: Vec<f32>,
    reused: Vec<u64>,
}

/// 跑一段序列：逐帧 `TemporalReprojector` + 独立 golden + 运动感知五判据。
fn run_sequence(w: u32, h: u32, frames: u32, new_spp: u32, m: Motion) -> SeqResult {
    let mut rp = TemporalReprojector::new(w, h);
    let mut reports = Vec::new();
    let mut new_samples = Vec::new();
    let mut motions = Vec::new();
    let mut reused = Vec::new();
    let mut prev_got: Option<Vec<[u8; 4]>> = None;
    let mut prev_golden: Option<Vec<[u8; 4]>> = None;
    for f in 0..frames {
        let req = req_at(w, h, f, m);
        rp.render(&req, new_spp);
        let got: Vec<[u8; 4]> = rp.last_got().to_vec();
        // 独立 golden：该帧自身 req、逐帧独立重渲 16 spp。
        let golden = independent_golden_u8(&req);
        let r = compare_frame_2d_expected(
            &golden, &got, prev_golden.as_deref(), prev_got.as_deref(), w, h,
        );
        reports.push(r);
        new_samples.push(rp.new_samples());
        motions.push(rp.stats().mean_motion_px);
        reused.push(rp.stats().reused_pixels);
        prev_got = Some(got);
        prev_golden = Some(golden);
    }
    SeqResult { reports, new_samples, motions, reused }
}

/// 判定一段序列：五条判据逐帧全过（硬 assert）+ 真实新样本非零（硬 assert）。
/// 返回 (平均真实新 spp, 平均复用像素数, 平均运动矢量)。
fn judge_all(w: u32, h: u32, frames: u32, new_spp: u32, m: Motion, label: &str) -> (f64, f64, f64) {
    let res = run_sequence(w, h, frames, new_spp, m);
    let psnrs: Vec<f64> = res.reports.iter().map(|r| r.psnr_db).collect();
    let seq = sequence_mean_psnr(&psnrs);
    let mut max_delta = 0u8;
    let mut max_p99 = 0u8;
    let mut worst_psnr = f64::INFINITY;
    let mut max_hist = 0f64;
    let mut min_edge = 1f64;
    let mut max_outlier = 0f64;
    for (f, r) in res.reports.iter().enumerate() {
        let v = judge_v31(r, seq);
        assert!(v.pass, "[{}] new_spp={} 帧 {} 判红: {:?}", label, new_spp, f, v.reasons);
        // 反作弊硬证据：复用不得引入任何偏差，got 必须逐位等于该帧独立 golden。
        assert_eq!(r.max_delta, 0,
            "[{}] new_spp={} 帧 {} got 必须逐位等于独立 golden（maxΔ 应为 0）", label, new_spp, f);
        max_delta = max_delta.max(r.max_delta);
        max_p99 = max_p99.max(r.p99_delta);
        worst_psnr = worst_psnr.min(r.psnr_db);
        max_hist = max_hist.max(r.hist_bin_max_rel);
        min_edge = min_edge.min(r.edge_retention);
        max_outlier = max_outlier.max(r.temporal_outlier_ratio);
    }
    // 硬约束：每帧真实新样本 > 0，禁止零新样本帧。
    for (f, &n) in res.new_samples.iter().enumerate() {
        assert!(n > 0, "[{}] new_spp={} 帧 {} 出现零新样本帧", label, new_spp, f);
    }
    let npix = (w as u64) * (h as u64);
    let mean_new: f64 = res.new_samples.iter().sum::<u64>() as f64
        / (frames as f64 * npix as f64);
    let mean_reused: f64 = res.reused.iter().sum::<u64>() as f64 / frames as f64;
    let mean_motion: f64 = res.motions.iter().sum::<f32>() as f64 / frames as f64;
    let min_new = res.new_samples.iter().copied().min().unwrap_or(0);
    eprintln!(
        "  [{}] new_spp={} frames={} PASS | maxΔ={} p99={} PSNR(worst)={:.2} seq={:.2} hist={:.4} edge={:.4} outlier={:.4} | mean_new={:.2}spp min_new={} mean_reused={:.0}px mean_mv={:.2}px",
        label, new_spp, frames, max_delta, max_p99, worst_psnr, seq, max_hist, min_edge,
        max_outlier, mean_new, min_new, mean_reused, mean_motion
    );
    (mean_new, mean_reused, mean_motion)
}

#[test]
fn acceptance_five_gates_static_and_motion() {
    let (w, h) = (160u32, 90u32);
    let frames = 24u32;
    let npix = (w * h) as u64;

    eprintln!("===== Static 静止片源（固定机位 + 固定几何）=====");
    for &spp in &[16u32, 8, 4] {
        let (mean_new, mean_reused, mean_mv) =
            judge_all(w, h, frames, spp, STATIC, "static");
        assert_eq!(mean_mv, 0.0, "静止片源运动矢量必须为 0");
        if spp < 16 {
            assert!(mean_reused > 0.0, "静止片源必须真的复用历史（复用比例 > 0）");
            assert!(mean_new < 16.0, "静止片源真实新 spp 应小于 16（有复用）");
        }
    }

    eprintln!("===== Orbit 相机绕行（w2d_showcase 3D 基帧形态）=====");
    for &spp in &[16u32, 8, 4] {
        let (mean_new, mean_reused, mean_mv) = judge_all(w, h, frames, spp, ORBIT, "orbit");
        assert!(mean_mv > 0.0, "相机绕行必须产出非零运动矢量（重投影真接线）");
        if spp < 16 {
            assert!(mean_reused > 0.0, "相机运动片源背景像素必须真实复用");
            assert!(mean_reused < npix as f64, "相机运动片源表面像素必须失效回填");
            assert!(mean_new < 16.0, "有复用 ⇒ 真实新 spp 应小于 16");
        } else {
            // 一致档 new_spp=16：复用槽位为 0，逐帧仍真实全采 16 spp
            // （harness 统一走重投影器，复用标记不改变任何数值）。
            assert!((mean_new - 16.0).abs() < 1e-6, "一致档每帧真实全采 16 spp");
        }
    }

    eprintln!("===== Object 固定机位 + 物体脉动（逐像素几何判定）=====");
    for &spp in &[16u32, 8, 4] {
        let (_, mean_reused, _) = judge_all(w, h, frames, spp, OBJECT, "object");
        if spp < 16 {
            assert!(mean_reused > 0.0, "未变背景像素应被复用");
            assert!(mean_reused < npix as f64, "变化球面像素应失效回填（部分复用）");
        }
    }

    eprintln!("===== Full S1 真运动片源（相机绕行 + 半径脉动遮挡）=====");
    for &spp in &[16u32, 8, 4] {
        let (mean_new, mean_reused, mean_mv) = judge_all(w, h, frames, spp, FULL, "motion");
        assert!(mean_mv > 0.0, "S1 真运动必须产出非零运动矢量");
        if spp < 16 {
            assert!(mean_reused > 0.0, "S1 真运动片源背景像素必须真实复用");
            assert!(mean_reused < npix as f64, "S1 真运动片源表面/遮挡像素必须失效回填");
            assert!(mean_new < 16.0, "有复用 ⇒ 真实新 spp 应小于 16（不是全采）");
        } else {
            assert!((mean_new - 16.0).abs() < 1e-6, "一致档每帧真实全采 16 spp");
        }
    }
}

#[test]
fn s1_motion_quantified_adjacent_frame_change() {
    // S1 量化证据：相邻帧主体像素确实变化（mean|Δ| + 变化像素占比）。
    let (w, h) = (160u32, 90u32);
    let mut prev: Option<Vec<[u8; 4]>> = None;
    let mut max_mean = 0f64;
    let mut max_pct = 0f64;
    for f in 0..24u32 {
        let req = req_at(w, h, f, FULL);
        let got = independent_golden_u8(&req);
        if let Some(p) = prev.as_ref() {
            let n = (w * h) as usize;
            let mut sum = 0f64;
            let mut changed = 0usize;
            for i in 0..n {
                let d = (p[i][0] as i32 - got[i][0] as i32).abs()
                    .max((p[i][1] as i32 - got[i][1] as i32).abs())
                    .max((p[i][2] as i32 - got[i][2] as i32).abs());
                sum += d as f64;
                if d > 8 {
                    changed += 1;
                }
            }
            let mean = sum / (n * 3) as f64;
            let pct = changed as f64 / n as f64 * 100.0;
            max_mean = max_mean.max(mean);
            max_pct = max_pct.max(pct);
            if f <= 3 || f % 6 == 0 {
                eprintln!("  frame {}->{}: mean|Δ|={:.3} changed(>8)={:.2}%", f - 1, f, mean, pct);
            }
        }
        prev = Some(got);
    }
    eprintln!("  S1 峰值：mean|Δ|={:.3} changed(>8)={:.2}%", max_mean, max_pct);
    assert!(max_mean > 1.0, "S1 片源相邻帧主体像素必须真的变化：mean|Δ|={max_mean}");
    assert!(max_pct > 2.0, "S1 片源须有显著比例像素变化：{max_pct}%");
}
