//! T04（V2.2-B-2 SVGF 空域降噪第二阶段）验收测试钉子。
//!
//! 覆盖 PRD 验收 1–5：
//! 1. 第一阶段不回归 —— 由 `denoise.rs` 既有 5 个 checkpoint 保证（本文件不重复，
//!    但全程随 `cargo test -p prism-postfx` 一起跑）。
//! 2. 重投影退化一致性：静态相机 + 静态场景 ⇒ 引导累积与「帧 0 起重放」逐字节
//!    一致（`==`，非 PSNR）。
//! 3. 动态场景：接缝无 NaN/Inf；PSNR ≥ 40 dB；sample 数下降 ≥ 50%。
//! 4. 零超越函数：见交付报告（模块内 `grep exp|log|powf|sin|cos` 零命中）。
//! 5. G1–G9：由交付报告逐条核对。
//!
//! 无 `#[ignore]`（宪章 G5）。

use prism_io::fbuf::{f16_to_f32, Fbuf16F};
use prism_noise::pcg;
use prism_postfx::denoise::{
    atrous_pass_variance_guided, closed_form_motion, svgf_two_stage, Camera, DenoiseGuide,
    Mat4, TemporalAccumulator, VarEstimate,
};

/// 确定性噪声：hash ∈ [0,1) → `[-amp, amp]`。
fn noise_at(seed: u64, x: u32, y: u32, amp: f32) -> f32 {
    (pcg::hash2_f32(seed, x as u64, y as u64) - 0.5) * 2.0 * amp
}

// ---------------------------------------------------------------------------
// 验收 2：静态相机 + 静态场景 ⇒ 引导累积与帧 0 起重放逐字节一致
// ---------------------------------------------------------------------------

#[test]
fn phase2_static_reprojection_is_bit_identical_to_replay() {
    let w = 16u32;
    let h = 16u32;
    // 静态相机：prev == cur ⇒ 闭式 motion 严格全零。
    let cam = Camera::new(Mat4::ortho_translate(0.0, 0.0));
    let depth = vec![1.0f32; (w * h) as usize]; // 全部有效

    let frame = |f: u64| -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (pcg::hash3_f32(123, x as u64, y as u64, f) - 0.5) * 0.6 + 0.5;
                fb.set(x, y, [v, v, v, 1.0]);
            }
        }
        fb
    };

    // 路径 A：帧 0 起重放（plain accumulate）。
    let mut a = Fbuf16F::new(w, h);
    let mut st = TemporalAccumulator { samples: 0 };
    for f in 0..8u64 {
        a = st.accumulate(&a, &frame(f));
        st.samples += 1;
    }

    // 路径 B：引导累积（静态相机 ⇒ motion 全 0）。
    let mut b = Fbuf16F::new(w, h);
    let mut st2 = TemporalAccumulator { samples: 0 };
    for f in 0..8u64 {
        let motion = closed_form_motion(&cam, &cam, &depth, w, h); // prev==cur ⇒ 全 0
        let guide = DenoiseGuide {
            depth: &depth,
            motion: &motion,
            normal: None,
        };
        b = st2.accumulate_guided(&b, &frame(f), &guide);
        st2.samples += 1;
    }

    assert_eq!(
        a.pixels, b.pixels,
        "静态相机下引导累积必须与帧 0 起重放逐字节一致（重投影写错判据）"
    );
}

// ---------------------------------------------------------------------------
// 验收 3：动态场景（相机平移）—— 无 NaN/Inf；PSNR ≥ 40 dB；sample 降 ≥ 50%
// ---------------------------------------------------------------------------

#[test]
fn phase2_dynamic_psnr_and_sample_reduction() {
    let w = 64u32;
    let h = 64u32;
    let amp = 0.04f32; // 每像素每帧（1 spp）噪声幅度 → 方差 = amp²/3
    let k = 128u32; // 累积帧数（≥ w 即可：像素 x 至少获 x+1 次观测）

    // 周期性平滑真值（world 空间，周期 w），保证平移下内容一致、无真实遮挡。
    let truth = |wx: i64, wy: i64| -> f32 {
        let ux = (((wx % w as i64) + w as i64) % w as i64) as f32;
        let uy = (((wy % h as i64) + h as i64) % h as i64) as f32;
        (0.1 + 0.8 * ((ux / (w as f32 - 1.0)) * 0.6 + (uy / (h as f32 - 1.0)) * 0.4))
            .clamp(0.0, 1.0)
    };

    // 相机：每帧平移 +1 像素（x）。帧 t 的相机偏移 = t。
    let cam_at = |off: i64| -> Camera {
        let aw = 2.0 / (w as f32);
        let bh = 2.0 / (h as f32);
        let bx = aw * (off as f32 + 0.5) - 1.0;
        let by = 1.0 - bh * 0.5; // off_y = 0
        Camera::new(Mat4([
            [aw, 0.0, 0.0, bx],
            [0.0, -bh, 0.0, by],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    };

    let depth = vec![1.0f32; (w * h) as usize];

    // 先验验证闭式运动矢量正确性：prev=cam(0), cur=cam(1) ⇒ motion = (-1, 0)。
    let motion_chk = closed_form_motion(&cam_at(0), &cam_at(1), &depth, w, h);
    for &m in &motion_chk {
        assert_eq!(m, [-1.0, 0.0], "闭式运动矢量必须等于 (-1, 0)");
    }

    let mut acc = Fbuf16F::new(w, h);
    let mut st = TemporalAccumulator { samples: 0 };
    for f in 0..k {
        let cur_off = f as i64;
        // 当前帧各像素的干净真值：world = (p - off)，周期化。
        let mut cur = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let wx = (x as i64) - cur_off;
                let t = truth(wx, y as i64);
                let nz = (pcg::hash3_f32(777, x as u64, y as u64, f as u64) - 0.5) * 2.0 * amp;
                let v = (t + nz).clamp(0.0, 1.0);
                cur.set(x, y, [v, v, v, 1.0]);
            }
        }
        // 重投影：prev = cam(f-1)，cur = cam(f)。
        let motion = closed_form_motion(&cam_at(cur_off - 1), &cam_at(cur_off), &depth, w, h);
        let guide = DenoiseGuide {
            depth: &depth,
            motion: &motion,
            normal: None,
        };
        acc = svgf_two_stage(&acc, &cur, &guide, st.samples, 3, 0.1, 0.1);
        st.samples += 1;
    }

    // 验收 3a：无 NaN/Inf（接缝不破裂）。
    for c in &acc.pixels {
        assert!(f16_to_f32(*c).is_finite(), "降噪后含 NaN/Inf（接缝破裂）");
    }

    // 验收 3b：PSNR vs 真值（最终帧真值 = world (p - (k-1)) 周期化）。
    let final_off = (k - 1) as i64;
    let mut sse = 0.0f64;
    let mut cnt = 0u64;
    for y in 0..h {
        for x in 0..w {
            let wx = (x as i64) - final_off;
            let t = truth(wx, y as i64);
            let v = acc.get(x, y)[0];
            let e = (v - t) as f64;
            sse += e * e;
            cnt += 1;
        }
    }
    let mse = sse / cnt as f64;
    let psnr = -10.0 * mse.log10(); // 范围 [0,1]，MAX = 1
    println!("phase2 动态降噪 PSNR = {psnr:.3} dB (MSE = {mse:.3e})");
    assert!(psnr >= 40.0, "动态降噪 PSNR 必须 ≥ 40 dB（实得 {psnr:.3}）");

    // 验收 3c：sample 数下降 ≥ 50%。单帧需要 spp = V1/MSE 才能达同等 MSE；
    // 时序方案每帧仅花 1 spp。reduction = 1 - 1/single_spp ≥ 0.5 ⇔ single_spp ≥ 2。
    let v1 = (amp * amp) / 3.0; // uniform[-amp,amp] 方差
    let single_frame_spp = ((v1 as f64) / mse).ceil() as u32;
    let reduction = 1.0 - 1.0 / single_frame_spp as f64;
    println!(
        "phase2 单帧达同等画质需 {single_frame_spp} spp；时序每帧 1 spp；sample 下降 {:.1}%",
        reduction * 100.0
    );
    assert!(single_frame_spp >= 2, "sample 数下降必须 ≥ 50%（single_spp={single_frame_spp}）");
}

// ---------------------------------------------------------------------------
// 验收 3（补充）：方差引导 à-trous 降低平坦噪声方差且不引入 NaN
// ---------------------------------------------------------------------------

#[test]
fn phase2_variance_guided_atrous_reduces_variance_and_no_nan() {
    let w = 16u32;
    let h = 16u32;
    let mut input = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = 0.5 + noise_at(5, x, y, 0.4);
            input.set(x, y, [v, v, v, 1.0]);
        }
    }
    // 平坦区方差场（统一高方差 ⇒ 方差项 ≈ 0，平滑照常进行）。
    let var = vec![0.4f32 * 0.4 / 3.0; (w * h) as usize];
    let sigma2 = 0.1 * 0.1;
    let sigma_v2 = 0.1 * 0.1;
    let out = atrous_pass_variance_guided(&input, 1, sigma2, sigma_v2, &var);

    for c in &out.pixels {
        assert!(f16_to_f32(*c).is_finite(), "方差引导 à-trous 产生 NaN/Inf");
    }

    let mut var_in = 0.0f64;
    let mut var_out = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let di = (input.get(x, y)[0] - 0.5) as f64;
            var_in += di * di;
            let do_ = (out.get(x, y)[0] - 0.5) as f64;
            var_out += do_ * do_;
        }
    }
    let n = (w * h) as f64;
    println!(
        "phase2 方差引导 à-trous 方差：{} → {}",
        var_in / n,
        var_out / n
    );
    assert!(
        var_out / n < var_in / n,
        "方差引导 à-trous 必须降低平坦噪声方差"
    );
}

// ---------------------------------------------------------------------------
// 辅助：VarEstimate 与闭式运动在边界/退化情形下行为正确（无 NaN、零运动）
// ---------------------------------------------------------------------------

#[test]
fn phase2_var_estimate_finite_and_closed_form_static_zero() {
    let w = 8u32;
    let h = 8u32;
    let mut fb = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = 0.3 + noise_at(9, x, y, 0.2);
            fb.set(x, y, [v, v, v, 1.0]);
        }
    }
    let ve = VarEstimate::from_spatial(&fb, 1);
    let field = ve.field();
    assert_eq!(field.len(), (w * h) as usize);
    for &v in &field {
        assert!(v.is_finite() && v >= 0.0, "VarEstimate 场必须有限非负");
    }

    // 静态相机 ⇒ 全零运动（深度非有限像素亦为 0）。
    let cam = Camera::new(Mat4::ortho_translate(2.0, 3.0));
    let depth = vec![f32::NAN; (w * h) as usize];
    let motion = closed_form_motion(&cam, &cam, &depth, w, h);
    for &m in &motion {
        assert_eq!(m, [0.0, 0.0], "静态相机（含无效深度）必须零运动");
    }
}
