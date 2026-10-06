//! W9-C1 攻击性边界测试（QA 严过关）。找真 Bug，非复读既有断言。

use prism_core::depth::{DepthComposite, DepthLayer, DepthMap};
use prism_core::compose::{blend_pixel, merge, ComposeNode, LayerBuf, apply_fog};
use prism_core::node::MergeOp;
use prism_postfx::effects::dof_full::FullDof;
use prism_postfx::effects::mask::{combine, feather, Mask, MaskOp};
use prism_io::fbuf::Fbuf16F;

// ======================= 深度映射退化 =======================

/// 缺陷面：深度全 0（全命中在近平面）AOV 应为全 0；全 NaN 不得产生 panic 或越界。
/// 失败 = 边界映射错误或 panic。
#[test]
fn depth_map_degenerate_full_near_and_nan() {
    let mut dm = DepthMap::new(3, 1, 1.0, 11.0);
    for x in 0..3 {
        dm.set(x, 0, 1.0); // 近平面
    }
    let aov = dm.to_aov(false);
    for &v in &aov {
        assert_eq!(v, 0.0, "近平面全 0");
    }
    // NaN 深度：夹紧后既非 0 也非 1？至少不得 panic。
    let mut dn = DepthMap::new(1, 1, 1.0, 11.0);
    dn.set(0, 0, f32::NAN);
    let n = dn.to_aov(false)[0];
    // clamp(NaN) 行为：记录实际（不 panic 即视为通过防御点）。
    assert!(n.is_nan() || (0.0..=1.0).contains(&n), "NaN 深度产出 {n} 越界");
}

/// 缺陷面：深度远小于 near 或远大于 far → clamp 到 [0,1]（不得溢出）。
#[test]
fn depth_map_out_of_range_clamps() {
    let mut dm = DepthMap::new(2, 1, 1.0, 11.0);
    dm.set(0, 0, 0.0); // < near
    dm.set(1, 0, 1e6); // >> far
    let aov = dm.to_aov(false);
    assert_eq!(aov[0], 0.0, "近于 near 应钳到 0");
    assert_eq!(aov[1], 1.0, "远于 far 应钳到 1");
}

// ======================= 深度合成软权重 =======================

/// 缺陷面：tol→0（真 0）时退化硬 Z-buffer —— z_far 应被完全遮挡（权重 1 vs 0）。
#[test]
fn composite_tol_zero_degenerates_to_hard_zbuffer() {
    let dc = DepthComposite { tolerance: 0.0 };
    // z_self 明显更近（更小）→ 权重饱和 1；对方饱和 0。
    let w_near = dc.soft_weight(1.0, 5.0);
    let w_far = dc.soft_weight(5.0, 1.0);
    assert_eq!(w_near, 1.0, "tol=0 近层应满权重（硬合并）");
    assert_eq!(w_far, 0.0, "tol=0 远层应零权重（被完全遮挡）");
    // 等 z → 0.5 等权。
    assert_eq!(dc.soft_weight(3.0, 3.0), 0.5);
}

/// 缺陷面：合成结果应"近遮远"——但当前实现用【层级 z】而非逐像素 depth，
/// 若两层 layer_z 相同则完全等权，逐像素 depth 不参与 → 几何遮挡语义丢失。
/// 失败 = 逐像素遮挡语义缺失（文档 §18 声称按逐像素 z 差）。
#[test]
fn composite_ignores_per_pixel_depth() {
    let mut back = DepthLayer::new(1, 1);
    back.set(0, 0, [0.0, 0.0, 1.0, 1.0], 100.0); // 后层远
    let mut front = DepthLayer::new(1, 1);
    front.set(0, 0, [1.0, 0.0, 0.0, 1.0], 1.0); // 前层近
    // layer_z 传相同值（描述"层整体深度"），但逐像素 depth 差异巨大。
    let dc = DepthComposite { tolerance: 0.0 };
    let out = dc.composite(&back, &front, 5.0, 5.0);
    let c = out.get(0, 0);
    // 期望：近层（z=1）遮远层（z=100）→ 应偏红。实际：等权 → 各半。
    assert!(
        c[0] > c[2] + 0.3,
        "近遮远失败：逐像素 depth 被忽略，输出 {c:?}（红={}, 蓝={}）",
        c[0], c[2]
    );
}

/// 缺陷面：合成深度取"权重较大者"——wf==wb 时取 front。语义应与颜色一致。
#[test]
fn composite_depth_consistent_with_color_weight() {
    let mut back = DepthLayer::new(1, 1);
    back.set(0, 0, [0.0, 0.0, 1.0, 1.0], 9.0);
    let mut front = DepthLayer::new(1, 1);
    front.set(0, 0, [1.0, 0.0, 0.0, 1.0], 1.0);
    let out = DepthComposite { tolerance: 0.0 }.composite(&back, &front, 9.0, 1.0);
    assert_eq!(out.depth_at(0, 0), 1.0, "近层胜 → 深度取近者");
}

// ======================= 遮罩布尔真值表 =======================

fn m1(v: f32) -> Mask {
    let mut m = Mask::new(1, 1);
    m.data[0] = v;
    m
}

/// 缺陷面：布尔真值表在 0/1 端点须符合集合语义（∪=max, ∩=min, −=sat）。
#[test]
fn mask_boolean_truth_table_endpoints() {
    let cases = [(0.0f32, 0.0f32), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
    for (a, b) in cases {
        let u = combine(MaskOp::Union, &m1(a), &m1(b)).get(0, 0);
        let i = combine(MaskOp::Intersect, &m1(a), &m1(b)).get(0, 0);
        let s = combine(MaskOp::Subtract, &m1(a), &m1(b)).get(0, 0);
        assert_eq!(u, a.max(b), "∪({a},{b})");
        assert_eq!(i, a.min(b), "∩({a},{b})");
        assert_eq!(s, (a - b).max(0.0), "−({a},{b})");
    }
}

/// 缺陷面：Invert 是 1−a，两次应复原；对 b 应忽略。
#[test]
fn mask_invert_involution_and_ignores_b() {
    let inv = combine(MaskOp::Invert, &m1(0.3), &m1(0.9)).get(0, 0);
    assert!((inv - 0.7).abs() < 1e-6, "Invert(0.3)=0.7，得 {inv}");
    let twice = combine(MaskOp::Invert, &m1(inv), &m1(0.0)).get(0, 0);
    assert!((twice - 0.3).abs() < 1e-6, "Invert 对合");
}

/// 缺陷面：零尺寸遮罩（0×0）不应 panic 且 combine 尺寸一致错误应暴露。
#[test]
fn mask_zero_size_no_panic() {
    let a = Mask::new(0, 0);
    let b = Mask::new(0, 0);
    let out = combine(MaskOp::Union, &a, &b);
    assert_eq!((out.width, out.height), (0, 0));
    assert!(out.data.is_empty());
    // get 越界返回 0，不 panic。
    assert_eq!(out.get(0, 0), 0.0);
}

/// 缺陷面：feather r 大于图像尺寸时索引不得越界/panic（边界钳制正确性）。
#[test]
fn feather_radius_exceeds_image_no_panic() {
    let mut m = Mask::new(2, 2);
    m.data = vec![1.0, 0.0, 0.0, 1.0];
    let f = feather(&m, 5); // r >> 尺寸
    assert_eq!((f.width, f.height), (2, 2));
    for &v in &f.data {
        assert!((0.0..=1.0).contains(&v), "羽化值越界 {v}");
    }
}

/// 缺陷面：feather 应能量守恒近似（全 1 场羽化后仍全 1）。
#[test]
fn feather_uniform_field_invariant() {
    let mut m = Mask::new(5, 5);
    for v in m.data.iter_mut() {
        *v = 0.6;
    }
    let f = feather(&m, 2);
    for &v in &f.data {
        assert!((v - 0.6).abs() < 1e-5, "均匀场羽化非不动点 {v}");
    }
}

// ======================= DOF 退化 =======================

/// 缺陷面：合焦（CoC=0）逐位恒等；单像素帧不 panic。
#[test]
fn dof_focus_bit_exact_identity_single_pixel() {
    let mut fb = Fbuf16F::new(1, 1);
    fb.set(0, 0, [0.3, 0.7, 0.2, 1.0]);
    let dof = FullDof::new(1, 1, vec![0.5], 0.5, 4);
    assert_eq!(dof.coc_radius(0, 0), 0, "合焦 CoC=0");
    let out = dof.render(&fb);
    assert_eq!(out.get(0, 0), fb.get(0, 0), "合焦像素必须逐位恒等");
}

/// 缺陷面：CoC 对 AOV 越界（深图>1 或负）应被 coc_bias/取整鲁棒处理，
/// 不得 panic 或产出超大半径。此测试暴露 max_coc 钳制是否正确。
#[test]
fn dof_coc_clamps_extreme_aov() {
    let dof = FullDof::new(1, 1, vec![100.0], 0.0, 3);
    // |100-0|*3 = 300 → 应钳到 max_coc=3。
    assert_eq!(dof.coc_radius(0, 0), 3, "极端 AOV 应钳到 max_coc");
}

/// 缺陷面：DOF 渲染确定性（同输入两遍逐位一致）。
#[test]
fn dof_render_deterministic_repeat() {
    let mut fb = Fbuf16F::new(6, 5);
    for y in 0..5u32 {
        for x in 0..6u32 {
            fb.set(x, y, [(x as f32) / 6.0, (y as f32) / 5.0, 0.5, 1.0]);
        }
    }
    let aov: Vec<f32> = (0..30).map(|i| (i % 7) as f32 / 6.0).collect();
    let dof = FullDof::new(6, 5, aov, 0.5, 3);
    assert_eq!(dof.render(&fb).pixels, dof.render(&fb).pixels, "DOF 必须逐位确定");
}

// ======================= Compose 混合退化 =======================

/// 缺陷面：over 在 upper alpha=1 时应完全取 upper（覆盖）；alpha=0 取 lower。
#[test]
fn over_endpoints_cover_swap() {
    let lower = [1.0f32, 0.0, 0.0, 1.0];
    let upper_full = [0.0f32, 1.0, 0.0, 1.0];
    let r = blend_pixel(MergeOp::Over, lower, upper_full);
    assert!((r[1] - 1.0).abs() < 1e-6 && r[0].abs() < 1e-6, "alpha=1 应全覆盖: {r:?}");
    let upper_zero = [0.0f32, 1.0, 0.0, 0.0];
    let r0 = blend_pixel(MergeOp::Over, lower, upper_zero);
    assert!((r0[0] - 1.0).abs() < 1e-6 && r0[1].abs() < 1e-6, "alpha=0 应保下层: {r0:?}");
}

/// 缺陷面：over 的 RGB 公式对 straight alpha 是否 premultiply 正确——
/// 半透明上层 over 不透明下层，期望 RGB 应为直接线性插值（straight）。
#[test]
fn over_half_transparent_blend() {
    let lower = [1.0f32, 0.0, 0.0, 1.0];
    let upper = [0.0f32, 0.0, 1.0, 0.5];
    let r = blend_pixel(MergeOp::Over, lower, upper);
    // 文档公式：a*(1-ab)+b*ab = 0.5*lower + 0.5*upper。
    assert!((r[0] - 0.5).abs() < 1e-6, "R 应为 0.5，得 {}", r[0]);
    assert!((r[2] - 0.5).abs() < 1e-6, "B 应为 0.5，得 {}", r[2]);
    assert!((r[3] - 1.0).abs() < 1e-6, "alpha 应为 1，得 {}", r[3]);
}

/// 缺陷面：空 Compose 层列表 → from_flat 返回 None；eval_flat 也 None。
#[test]
fn compose_empty_layers_none() {
    assert!(ComposeNode::from_flat(MergeOp::Over, vec![]).is_none(), "空列表应 None");
    assert!(ComposeNode::eval_flat(MergeOp::Over, &[]).is_none(), "空归约应 None");
}

/// 缺陷面：eval_flat 对非结合 op（over）用 reduce 固定形状，是否与树形 eval 一致？
/// 这是一个可疑点：若 reduce 形状非左结合，over 结果会偏离层序语义。
#[test]
fn eval_flat_over_matches_tree_eval() {
    // 3 层不同 alpha，over 非结合，形状敏感。
    let mk = |a: f32, col: [f32; 3]| {
        let mut l = LayerBuf::new(1, 1);
        l.set(0, 0, [col[0], col[1], col[2], a]);
        l
    };
    let layers = vec![mk(1.0, [1.0, 0.0, 0.0]), mk(0.5, [0.0, 1.0, 0.0]), mk(0.5, [0.0, 0.0, 1.0])];
    let tree = ComposeNode::from_flat(MergeOp::Over, layers.clone()).unwrap().eval();
    let flat = ComposeNode::eval_flat(MergeOp::Over, &layers).unwrap();
    let tv = tree.get(0, 0);
    let fv = flat.get(0, 0);
    for i in 0..4 {
        assert!(
            (tv[i] - fv[i]).abs() < 1e-6,
            "eval_flat 与树形 eval 对 over 不一致：tree={tv:?} flat={fv:?}"
        );
    }
}

/// 缺陷面：雾合成 density=0 应恒等；aov=1 且 density=1 应全雾色。
#[test]
fn fog_endpoints() {
    let c = [0.2f32, 0.4, 0.6, 1.0];
    let fog = [1.0f32, 1.0, 1.0];
    assert_eq!(apply_fog(c, fog, 1.0, 0.0), c, "density=0 恒等");
    let full = apply_fog(c, fog, 1.0, 1.0);
    assert!((full[0] - 1.0).abs() < 1e-6 && (full[1] - 1.0).abs() < 1e-6, "全雾");
}

/// 缺陷面：merge 尺寸不一致必须 panic（契约）。
#[test]
#[should_panic]
fn merge_mismatched_size_panics() {
    let a = LayerBuf::new(1, 1);
    let b = LayerBuf::new(2, 1);
    let _ = merge(MergeOp::Over, &a, &b);
}
