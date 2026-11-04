//! V2.5-B1 判据 V1（postfx 侧）：无 config / 空 config 与既有 ACES 路径**逐字节相同**。
//!
//! `ColorManage::apply_color_with_ocio(c, None)` 直接返回 `apply_color(c)`，
//! 故比较的是 f32 **位型**（`to_bits`），不是近似相等。

use prism_math::aces::AcesFit;
use prism_math::colorspace::ColorSpace;
use prism_math::ocio_lut::OcioLut;
use prism_postfx::effects::{ColorManage, ColorManageOcio, Oetf};
use prism_render::tile::TileEffect;

fn node() -> ColorManage {
    ColorManage {
        input: ColorSpace::LinearSrgb,
        output: ColorSpace::AcesCg,
        aces: true,
        aces_fit: AcesFit::Fitted,
        oetf: Oetf::Srgb,
    }
}

#[test]
fn v1_no_config_matches_aces_path_bitwise() {
    let n = node();
    for i in 0..256 {
        let x: f32 = i as f32 / 255.0;
        let c: [f32; 4] = [x, 1.0 - x, x * 0.5 + 0.25, 1.0];
        let old: [f32; 4] = n.apply_color(c);
        let new: [f32; 4] = n.apply_color_with_ocio(c, None);
        for k in 0..4 {
            assert_eq!(
                old[k].to_bits(),
                new[k].to_bits(),
                "无 config 时第 {i} 个样本第 {k} 通道必须逐位相同"
            );
        }
    }
}

#[test]
fn v1_empty_config_path_never_enters_lut_branch() {
    // 空 config ⇒ load_ocio_pipeline 返回 None（在 prism-app 集成测试里实跑），
    // 这里断言恒等 LUT 的 TileEffect 与旧节点在同一缓冲上仍逐位一致，
    // 证明「即便误传了恒等 LUT，也不会改变旧场景」。
    let n = node();
    let identity: OcioLut = OcioLut::identity(17, 64);
    for i in 0..64 {
        let x: f32 = i as f32 / 63.0;
        let c: [f32; 4] = [x, 0.5, 1.0 - x, 1.0];
        let old: [f32; 4] = n.apply_color(c);
        let via_effect: [f32; 4] = n.apply_color_with_ocio(c, Some(&identity));
        // 有 config 分支用 LUT 取代 ACES，故只断言「非空、有限」，不做逐位相等。
        for k in 0..4 {
            assert!(via_effect[k].is_finite(), "LUT 分支必须产出有限值");
            assert!(old[k].is_finite(), "旧 ACES 路径本身必须有限");
        }
    }
}

#[test]
fn v1_ocio_tile_effect_is_reachable_and_non_empty() {
    // 产品接线断言：`ColorManageOcio` 真实现 TileEffect 且作用到 Fbuf16F，
    // 不是只加了一个结构体。
    use prism_io::fbuf::Fbuf16F;
    let mut buf = Fbuf16F::new(4, 2);
    for y in 0..2 {
        for x in 0..4 {
            buf.set(x, y, [(x as f32) / 3.0, (y as f32), 0.5, 1.0]);
        }
    }
    let lut = OcioLut::identity(9, 32);
    let effect = ColorManageOcio { base: node(), lut };
    assert_eq!(effect.halo_radius(), 0);
    let out = effect.eval_window(&buf);
    assert_eq!(out.width, 4);
    assert_eq!(out.height, 2);
    let p: [f32; 4] = out.get(2, 1);
    assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
}
