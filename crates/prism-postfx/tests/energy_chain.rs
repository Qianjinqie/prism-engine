//! B14 集成测试：halation 能量守恒 + 二级调色跨 crate 链路。
//!
//! - **halation 能量守恒**：加算型算子只增不减（`delta >= 0`）且注入有界
//!  （`delta <= strength · max(tint) · Σdecay^k · 高光层能量`）——
//!   "单向扩散 + 波长染色"的物理动机断言；
//! - **链路**：halation → 二级调色 → 纹理乘算 串接后逐位可复现。

use prism_aesthetic::secondary::{LiftGammaGain, Secondary};
use prism_io::fbuf::Fbuf16F;
use prism_postfx::effects::film::{halation_energy_budget, Halation};
use prism_render::tile::TileEffect;
use prism_texture::paper::Paper;

fn hdr_fixture() -> Fbuf16F {
    // HDR 输入：含超阈值高光（局部亮点）。
    let mut fb = Fbuf16F::new(24, 18);
    for y in 0..18u32 {
        for x in 0..24u32 {
            let base = (x + y) as f32 / 50.0;
            let hot = if (x == 10 || x == 11) && (y == 8 || y == 9) { 3.0 } else { 0.0 };
            fb.set(x, y, [base + hot, base + hot, base + hot, 1.0]);
        }
    }
    fb
}

#[test]
fn halation_energy_conservation() {
    let win = hdr_fixture();
    for &levels in &[1u32, 4, 8, 12] {
        for &strength in &[0.0f32, 0.2, 0.5, 1.0] {
            let h = Halation { levels, strength, ..Halation::default() };
            let (delta, bound) = halation_energy_budget(&win, &h);
            // 加算型：能量只增不减。
            assert!(delta >= 0.0, "levels={levels} strength={strength} 减能：{delta}");
            // 有界注入。
            assert!(
                delta <= bound + 1e-6,
                "levels={levels} strength={strength} 越界：{delta} > {bound}"
            );
        }
    }
    // strength 0 ⇒ 输入指纹不变（恒等）。
    let zero = Halation { strength: 0.0, ..Halation::default() };
    assert_eq!(zero.eval_window(&win), win, "strength 0 必须逐位恒等");
}

#[test]
fn halation_red_tint_halo_ordering() {
    // 纯白高光处晕圈：红 > 绿 > 蓝（红光晕）。
    let win = hdr_fixture();
    let out = Halation::default().eval_window(&win);
    let near = out.get(12, 9); // 靠近高光
    let base = win.get(12, 9);
    let dr = near[0] - base[0];
    let dg = near[1] - base[1];
    let db = near[2] - base[2];
    assert!(dr > dg && dg > db, "halation 晕圈必须偏红：Δ = {dr}/{dg}/{db}");
}

#[test]
fn film_then_secondary_then_texture_chain_reproducible() {
    let win = hdr_fixture();
    let hal = Halation::default();
    let sec = Secondary {
        lgg: LiftGammaGain { lift: 0.05, gamma: 1.3, gain: 1.05, gamma_steps: 8 },
        ..Secondary::default()
    };
    let tex = Paper { multiply: true, ..Paper::default() };

    let chain1 = tex
        .eval_window(&sec.eval_window(&hal.eval_window(&win)));
    let chain2 = tex
        .eval_window(&sec.eval_window(&hal.eval_window(&win)));
    assert_eq!(chain1, chain2, "三段链路必须逐位可复现");
    // 链路改变了画面（非恒等）。
    assert_ne!(chain1.pixels, win.pixels, "链路应产生风格化输出");
}
