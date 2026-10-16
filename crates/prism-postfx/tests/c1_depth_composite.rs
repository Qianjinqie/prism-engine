//! C1 深度合成与遮罩族——集成验收测试（Phase 3 集成波）。
//!
//! 覆盖合成域端到端：深度 AOV（prism-rt）→ 深度合成几何正确性
//! （近遮远近）→ Compose 树（层序决定论）→ 遮罩族（布尔运算 + 光栅）→
//! golden-file 指纹。全部 Cpu 可跑、确定性可复验。

use prism_core::{
    apply_fog, blend_pixel, DepthComposite, DepthLayer, DepthMap,
};
use prism_core::node::MergeOp;
use prism_core::compose::{ComposeNode, LayerBuf};
use prism_math::Vec3;
use prism_postfx::effects::{
    apply_mask, combine, feather, ChannelMask, ChromaKey, LumaKey, Mask, MaskOp, PathMask,
};
use prism_raster::geom::Point;
use prism_rt::camera::Camera;
use prism_rt::depth::{render_depth, DepthParams};
use prism_rt::integrator::{MaterialKind, Scene, Sphere};
use prism_core::EvalCtx;

/// 单像素层（便于精确断言）。
fn layer_px(rgba: [f32; 4], depth: f32) -> DepthLayer {
    let mut l = DepthLayer::new(1, 1);
    l.set(0, 0, rgba, depth);
    l
}

// ---------------------------------------------------------------------------
// 1. 深度合成几何正确性：近遮远近
// ---------------------------------------------------------------------------

#[test]
fn depth_composite_near_occludes_far_geometry() {
    // 远层蓝色（z=10），近层红色（z=1）→ 合成偏向近层（红）。
    let far = layer_px([0.0, 0.0, 1.0, 1.0], 10.0);
    let near = layer_px([1.0, 0.0, 0.0, 1.0], 1.0);
    let dc = DepthComposite { tolerance: 0.0 };
    let out = dc.composite(&far, &near, 10.0, 1.0);
    let c = out.get(0, 0);
    assert!(c[0] > c[2], "近层（红）必须遮住远层（蓝）：{c:?}");
    assert_eq!(out.depth_at(0, 0), 1.0, "合成深度取近者");

    // 反向：把近层（红 z=1）放 back 位、远层（蓝 z=10）放 front 位——
    // 几何仍由 z 决定：近层（红 z=1）遮住远层（蓝 z=10），与层摆放无关。
    let out2 = dc.composite(&layer_px([1.0, 0.0, 0.0, 1.0], 1.0), &layer_px([0.0, 0.0, 1.0, 1.0], 10.0), 1.0, 10.0);
    let c2 = out2.get(0, 0);
    assert!(c2[0] > c2[2], "近层红必须遮住远层蓝（与摆放无关）：{c2:?}");
}

#[test]
fn depth_composite_tolerance_produces_soft_edge() {
    // 两层接近（z 差 1 远小于容差 5）→ 软混合中间值（不硬切换）。
    let a = layer_px([0.0, 0.0, 0.0, 1.0], 5.0);
    let b = layer_px([1.0, 1.0, 1.0, 1.0], 6.0);
    let dc = DepthComposite { tolerance: 5.0 };
    let c = dc.composite(&a, &b, 5.0, 6.0).get(0, 0)[0];
    assert!(c > 0.0 && c < 1.0, "容差带内必须软混合：{c}");
}

// ---------------------------------------------------------------------------
// 2. Compose 树：层序决定论
// ---------------------------------------------------------------------------

#[test]
fn compose_tree_layer_order_decision() {
    let red = {
        let mut l = LayerBuf::new(1, 1);
        l.set(0, 0, [1.0, 0.0, 0.0, 1.0]);
        l
    };
    let green = {
        let mut l = LayerBuf::new(1, 1);
        l.set(0, 0, [0.0, 1.0, 0.0, 1.0]);
        l
    };
    // 上层不透明 → over 结果 = 上层色（层序决定输出）。
    let t1 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(red.clone()), ComposeNode::leaf(green.clone()));
    let t2 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(green), ComposeNode::leaf(red));
    assert!((t1.eval().get(0, 0)[1] - 1.0).abs() < 1e-6, "t1 上层绿");
    assert!((t2.eval().get(0, 0)[0] - 1.0).abs() < 1e-6, "t2 上层红");
    assert_ne!(t1.eval().get(0, 0), t2.eval().get(0, 0), "层序必须决定结果");

    // 二次求值逐位一致。
    assert_eq!(t1.eval().get(0, 0), t1.eval().get(0, 0));
}

#[test]
fn compose_tree_nested_merge_arbitrary_depth() {
    // 四层嵌套树（层数无上限）。
    let mk = |v: f32| {
        let mut l = LayerBuf::new(1, 1);
        l.set(0, 0, [v, v, v, 1.0]);
        l
    };
    let tree = ComposeNode::merge(
        MergeOp::Plus,
        ComposeNode::merge(
            MergeOp::Over,
            ComposeNode::leaf(mk(0.25)),
            ComposeNode::leaf(mk(0.0)),
        ),
        ComposeNode::leaf(mk(0.5)),
    );
    let out = tree.eval().get(0, 0);
    // (0.25 over 零) ：上层不透明（alpha=1）→ 覆盖为 0；再 + 0.5 = 0.5。
    assert!((out[0] - 0.5).abs() < 1e-6, "嵌套合成结果：{out:?}");
}

// ---------------------------------------------------------------------------
// 3. 遮罩布尔运算
// ---------------------------------------------------------------------------

#[test]
fn mask_boolean_laws() {
    let a = uniform_mask(0.3);
    let b = uniform_mask(0.7);
    assert_eq!(combine(MaskOp::Union, &a, &b).get(0, 0), 0.7);
    assert_eq!(combine(MaskOp::Intersect, &a, &b).get(0, 0), 0.3);
    assert_eq!(combine(MaskOp::Subtract, &a, &b).get(0, 0), 0.0);
    assert!((combine(MaskOp::Subtract, &b, &a).get(0, 0) - 0.4).abs() < 1e-6);
    // 反相两次还原。
    let inv = combine(MaskOp::Invert, &a, &a);
    let back = combine(MaskOp::Invert, &inv, &inv);
    assert!((back.get(0, 0) - 0.3).abs() < 1e-6, "反相反相 = 原值");
}

fn uniform_mask(v: f32) -> Mask {
    let mut m = Mask::new(1, 1);
    m.set(0, 0, v);
    m
}

#[test]
fn mask_feather_and_pathmask_pipeline() {
    // 形状遮罩光栅 → 羽化 → 应用（ROTO 基础链）。
    let pts = vec![
        Point::new(4.0, 4.0),
        Point::new(12.0, 4.0),
        Point::new(12.0, 12.0),
        Point::new(4.0, 12.0),
    ];
    let mut pm = PathMask::new(pts);
    pm.feather_radius = 1;
    let m = pm.rasterize(16, 16);
    // 内部覆盖、外部空。
    assert!(m.get(8, 8) > 0.9, "内部覆盖");
    assert!(m.get(0, 0) < 0.1, "外部空");
    // 边界带出现中间值（羽化展宽）。
    let edge = m.get(4, 8);
    assert!(edge > 0.0 && edge < 1.0, "羽化边界应为中间值：{edge}");
}

#[test]
fn mask_luma_and_chroma_keys() {
    let fb = solid_fb(2, 1, [0.5, 0.5, 0.5, 1.0]);
    let lk = LumaKey { low: 0.4, high: 0.6, invert: false };
    assert!((lk.mask(&fb).get(0, 0) - 0.5).abs() < 1e-4);

    // 绿幕键出。
    let green = solid_fb(1, 1, [0.0, 1.0, 0.0, 1.0]);
    let ck = ChromaKey::default();
    assert!(ck.mask(&green).get(0, 0) < 0.05, "绿幕应键出");

    // 通道遮罩。
    let cm = ChannelMask { channel: 3, low: 0.0, high: 1.0, invert: false };
    assert!((cm.mask(&fb).get(0, 0) - 1.0).abs() < 1e-4, "alpha 通道 = 1");
}

fn solid_fb(w: u32, h: u32, c: [f32; 4]) -> prism_io::fbuf::Fbuf16F {
    let mut fb = prism_io::fbuf::Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            fb.set(x, y, c);
        }
    }
    fb
}

#[test]
fn apply_mask_multiplies_and_feather_smooths() {
    let fb = solid_fb(4, 1, [1.0, 1.0, 1.0, 1.0]);
    let mut m = Mask::new(4, 1);
    for x in 0..4 {
        m.set(x, 0, if x < 2 { 1.0 } else { 0.0 });
    }
    let feathered = feather(&m, 1);
    let out = apply_mask(&fb, &feathered);
    // 边界处出现中间值。
    let c = out.get(2, 0)[0];
    assert!(c > 0.0 && c < 1.0, "羽化后边界半保留：{c}");
}

// ---------------------------------------------------------------------------
// 4. 深度 AOV（prism-rt）→ 雾合成 → golden-file
// ---------------------------------------------------------------------------

#[test]
fn rt_depth_aov_feeds_fog_composite() {
    // 球在 (0,0,2) 半径 0.5；相机原点看向 +Z。
    let scene = Scene {
        spheres: vec![(
            Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 0.5 },
            MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
        )],
        boxes: Vec::new(),
        lights: Vec::new(),
        medium: None,
        paraboloids: vec![],
        planes: vec![],
    };
    let cam = Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        0.0,
        2.0,
        0.0,
        0.0,
    );
    let ctx = EvalCtx::new(7, 1, 0);
    let dm = render_depth(&scene, &cam, &ctx, 8, 8, DepthParams::default());
    let aov = dm.to_aov(false);
    assert_eq!(aov.len(), 64);

    // 雾合成：背景（无命中，AOV=1）→ 全雾色；命中球（近）→ 保留原色。
    let near_color = apply_fog([1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.6], dm.aov_at(4, 4, false), 1.0);
    let bg_color = apply_fog([1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.6], dm.aov_at(0, 0, false), 1.0);
    // 球命中像素（AOV 小）比背景像素（AOV=1）更接近原色（红）。
    assert!(
        near_color[0] >= bg_color[0],
        "命中球像素应比背景更少被雾化：{near_color:?} vs {bg_color:?}"
    );
    assert!((bg_color[0] - 0.5).abs() < 1e-5, "背景全雾化：{bg_color:?}");
}

#[test]
fn depth_map_golden_linear_mapping() {
    // 深度 AOV 线性映射 golden：near=1 far=11 → d ∈ {1,6,11} → {0,0.5,1}。
    let mut dm = DepthMap::new(3, 1, 1.0, 11.0);
    dm.set(0, 0, 1.0);
    dm.set(1, 0, 6.0);
    dm.set(2, 0, 11.0);
    let aov = dm.to_aov(false);
    assert_eq!(aov, vec![0.0, 0.5, 1.0], "线性映射 golden");
}

#[test]
fn blend_pixel_reference_formulas_golden() {
    // 混合模式 golden（over/multiply/screen 参考式）。
    let a = [0.5, 0.5, 0.5, 1.0];
    let b = [0.5, 0.5, 0.5, 0.5];
    let over = blend_pixel(MergeOp::Over, a, b);
    assert!((over[0] - 0.5).abs() < 1e-6, "over: {over:?}");
    let mul = blend_pixel(MergeOp::Multiply, a, a);
    assert!((mul[0] - 0.25).abs() < 1e-6, "multiply: {mul:?}");
    let scr = blend_pixel(MergeOp::Screen, a, a);
    assert!((scr[0] - 0.75).abs() < 1e-6, "screen: {scr:?}");
}
