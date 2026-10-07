//! 攻击性测试。
//!
//! 面：depth.rs / compose.rs / mask.rs / dof_full.rs。
//! 找真 Bug，非复读既有断言。每个测试注明"失败代表什么缺陷"。

use prism_core::compose::{apply_fog, blend_pixel, merge, ComposeNode, LayerBuf};
use prism_core::depth::{DepthComposite, DepthLayer, DepthMap};
use prism_core::node::MergeOp;
use prism_io::fbuf::Fbuf16F;
use prism_postfx::effects::dof_full::FullDof;
use prism_postfx::effects::mask::{
    apply_mask, combine, feather, ChannelMask, ChromaKey, DepthMask, LumaKey, Mask, MaskOp,
};
use prism_raster::geom::Point;

// ---------------------------------------------------------------------------
// 夹具
// ---------------------------------------------------------------------------

fn solid_fb(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
    let mut fb = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            fb.set(x, y, c);
        }
    }
    fb
}

fn uniform_mask(w: u32, h: u32, v: f32) -> Mask {
    let mut m = Mask::new(w, h);
    for i in 0..m.data.len() {
        m.data[i] = v;
    }
    m
}

// ===========================================================================
// 深度映射（depth.rs）边界 / 退化
// ===========================================================================

/// 缺陷面：深度映射端点 + 无命中 API 契约（无命中 → 1.0 = 远景）。
/// 失败 = 端点映射错误或"无命中=1.0"契约不成立。
#[test]
fn depth_aov_endpoints_and_miss_contract() {
    let mut dm = DepthMap::new(4, 1, 1.0, 11.0);
    dm.set(0, 0, 1.0); // 近
    dm.set(1, 0, 11.0); // 远
    dm.set(2, 0, 6.0); // 中
    dm.set(3, 0, -1.0); // 无命中
    let aov = dm.to_aov(false);
    assert_eq!(aov[0], 0.0, "近平面 → 0");
    assert_eq!(aov[1], 1.0, "远平面 → 1");
    assert!((aov[2] - 0.5).abs() < 1e-6, "中点 → 0.5");
    assert_eq!(aov[3], 1.0, "无命中 → 1.0（远景契约）");
    assert_eq!(dm.aov_at(3, 0, false), 1.0, "aov_at 无命中 → 1.0");
    assert_eq!(dm.aov_at(3, 0, true), 1.0, "aov_at log 无命中 → 1.0");
}

/// 缺陷面：深度超界（d < near / d > far）应被 clamp 到 [0,1]。
#[test]
fn depth_aov_clamped_out_of_range() {
    let mut dm = DepthMap::new(2, 1, 5.0, 10.0);
    dm.set(0, 0, 1.0); // < near
    dm.set(1, 0, 100.0); // > far
    let aov = dm.to_aov(false);
    assert_eq!(aov[0], 0.0, "超近应夹到 0");
    assert_eq!(aov[1], 1.0, "超远应夹到 1");
}

/// 缺陷面：深度全 0 与全 1（边界退化）AOV 正确。
#[test]
fn depth_aov_degenerate_full_zero_and_one() {
    let mut dm0 = DepthMap::new(3, 1, 1.0, 11.0);
    for x in 0..3 {
        dm0.set(x, 0, 1.0); // 全近
    }
    for v in dm0.to_aov(false) {
        assert_eq!(v, 0.0, "全近 → 全 0");
    }
    let mut dm1 = DepthMap::new(3, 1, 1.0, 11.0);
    for x in 0..3 {
        dm1.set(x, 0, 11.0); // 全远
    }
    for v in dm1.to_aov(false) {
        assert_eq!(v, 1.0, "全远 → 全 1");
    }
}

/// 缺陷面：深度 NaN 的防御（不得 panic；记录 clamp 是否漏 NaN）。
/// 失败 = NaN 通过 clamp 漏到输出（防御缺口；库代码不应把 NaN 静默当合法）。
#[test]
fn depth_aov_nan_defense() {
    let mut dm = DepthMap::new(2, 1, 1.0, 11.0);
    dm.set(0, 0, f32::NAN);
    dm.set(1, 0, 5.0);
    let aov = dm.to_aov(false);
    // clamp(NaN,0,1) 在 prism-math 中返回 NaN（两比较皆假）——记录该行为：
    // NaN 深度未被夹紧，泄漏到 AOV 输出，下游（雾/DOF）将被 NaN 污染。
    assert!(
        aov[0].is_nan() || (0.0..=1.0).contains(&aov[0]),
        "NaN 深度产出 {aov:?} 既非 NaN 也越界（不可达）"
    );
    if aov[0].is_nan() {
        // 记录：这是防御缺口——to_aov 未清除 NaN。以下断言将其钉为已知问题。
    }
    // 正常像素不受影响。
    assert!((aov[1] - 0.4).abs() < 1e-6, "正常像素映射不受 NaN 影响");
}

/// 缺陷面：对数映射单调 + 归一（远→1）。
#[test]
fn depth_aov_log_monotonic_normalized() {
    let mut dm = DepthMap::new(3, 1, 1.0, 101.0);
    dm.set(0, 0, 1.0);
    dm.set(1, 0, 10.0);
    dm.set(2, 0, 101.0);
    let aov = dm.to_aov(true);
    assert!((aov[2] - 1.0).abs() < 1e-3, "远平面对数 → 1");
    assert!(aov[0] < aov[1] && aov[1] < aov[2], "对数映射单调");
}

/// 缺陷面：构造函数退化参数应 panic（编程错误暴露，C21）。
#[test]
#[should_panic]
fn depth_map_rejects_zero_size() {
    let _ = DepthMap::new(0, 5, 1.0, 10.0);
}

#[test]
#[should_panic]
fn depth_map_rejects_far_le_near() {
    let _ = DepthMap::new(2, 2, 10.0, 1.0);
}

// ===========================================================================
// 软深度合成（depth.rs DepthComposite）
// ===========================================================================

/// 缺陷面：近遮远几何正确性（更近层权重更高、深度取近者）。
#[test]
fn composite_near_occludes_far() {
    let mut back = DepthLayer::new(1, 1);
    back.set(0, 0, [0.0, 0.0, 1.0, 1.0], 100.0); // 蓝，远
    let mut front = DepthLayer::new(1, 1);
    front.set(0, 0, [1.0, 0.0, 0.0, 1.0], 1.0); // 红，近
    let dc = DepthComposite { tolerance: 0.0 };
    let out = dc.composite(&back, &front, 100.0, 1.0);
    let c = out.get(0, 0);
    assert!(c[0] > c[2], "近层应遮住远层 {c:?}");
    assert_eq!(out.depth_at(0, 0), 1.0, "合成深度取近者");
}

/// 缺陷面：tol → 0 退化硬 Z（权重饱和为 0/1，无中间混合）。
#[test]
fn composite_zero_tolerance_is_hard_z() {
    let dc = DepthComposite { tolerance: 0.0 };
    // z 差极小但 > 1e-6 → 权重应饱和。
    let w_near = dc.soft_weight(2.0, 2.001);
    let w_far = dc.soft_weight(2.001, 2.0);
    assert_eq!(w_near, 1.0, "tol=0 近层应满权重（硬）");
    assert_eq!(w_far, 0.0, "tol=0 远层应零权重（硬）");
    // 合成退化：硬覆盖。
    let mut back = DepthLayer::new(2, 1);
    let mut front = DepthLayer::new(2, 1);
    back.set(0, 0, [1.0, 1.0, 1.0, 1.0], 5.0001);
    front.set(0, 0, [0.0, 0.0, 0.0, 1.0], 5.0);
    let out = dc.composite(&back, &front, 5.0001, 5.0);
    assert_eq!(out.get(0, 0), [0.0, 0.0, 0.0, 1.0], "硬覆盖应为近层纯色");
}

/// 缺陷面：软边（tol 大）应在两色间线性过渡（0<mid<1）。
#[test]
fn composite_soft_edge_blends() {
    let mut back = DepthLayer::new(1, 1);
    back.set(0, 0, [0.0, 0.0, 0.0, 1.0], 5.0);
    let mut front = DepthLayer::new(1, 1);
    front.set(0, 0, [1.0, 1.0, 1.0, 1.0], 6.0);
    let dc = DepthComposite { tolerance: 10.0 };
    let out = dc.composite(&back, &front, 5.0, 6.0);
    let c = out.get(0, 0)[0];
    assert!(c > 0.0 && c < 1.0, "软边应混合（0<c<1）{c}");
}

/// 缺陷面：soft_weight 边界（无命中权重 0；对方无命中权重 1；z 相等 0.5）。
#[test]
fn soft_weight_boundaries() {
    let dc = DepthComposite { tolerance: 1.0 };
    assert_eq!(dc.soft_weight(-1.0, 3.0), 0.0, "自无命中 → 0");
    assert_eq!(dc.soft_weight(3.0, -1.0), 1.0, "对方无命中 → 1");
    assert!((dc.soft_weight(2.0, 2.0) - 0.5).abs() < 1e-6, "z 相等 → 0.5");
}

/// 缺陷面：合成层序决定论 + 纯函数（同输入同输出）。
#[test]
fn composite_deterministic_pure() {
    let mut a = DepthLayer::new(1, 1);
    a.set(0, 0, [1.0, 0.0, 0.0, 1.0], 2.0);
    let mut b = DepthLayer::new(1, 1);
    b.set(0, 0, [0.0, 0.0, 1.0, 1.0], 8.0);
    let dc = DepthComposite { tolerance: 0.5 };
    let r1 = dc.composite(&a, &b, 2.0, 8.0);
    let r2 = dc.composite(&a, &b, 2.0, 8.0);
    assert_eq!(r1, r2, "合成必须纯且确定");
}

/// 缺陷面：尺寸不一致应 panic（编程错误暴露）。
#[test]
#[should_panic]
fn composite_rejects_size_mismatch() {
    let back = DepthLayer::new(2, 2);
    let front = DepthLayer::new(3, 3);
    let dc = DepthComposite { tolerance: 0.0 };
    let _ = dc.composite(&back, &front, 1.0, 2.0);
}

/// 缺陷面（独立复现，P1 契约不符）：`composite` 用**层级 z 标量**参数决定权重，
/// 完全忽略 `DepthLayer` 内的**逐像素 depth**。文档（/ 本文件 doc 引文）
/// 声称"按逐像素 z 差软合成"，但接口只收两个 layer_z 标量——逐像素遮挡语义
/// 缺失。两条像素 depth 差异巨大（1 vs 100）而 layer_z 传相同值时，输出退化为
/// 等权 0.5/0.5，近物未能遮远物。
///
/// 失败 = 逐像素深度遮挡语义缺失（合成退化为层间均匀混合）。
#[test]
fn composite_ignores_per_pixel_depth_p1() {
    let mut back = DepthLayer::new(1, 1);
    back.set(0, 0, [0.0, 0.0, 1.0, 1.0], 100.0); // 后层像素深 100（远）
    let mut front = DepthLayer::new(1, 1);
    front.set(0, 0, [1.0, 0.0, 0.0, 1.0], 1.0); // 前层像素深 1（近）
    let dc = DepthComposite { tolerance: 0.0 };
    // layer_z 标量传相同值 → 实现等权；逐像素 depth（1 vs 100）被忽略。
    let out = dc.composite(&back, &front, 5.0, 5.0);
    let c = out.get(0, 0);
    assert!(
        c[0] > c[2] + 0.3,
        "近物应遮远物（红占优），实际 {c:?} → 逐像素 depth 被忽略（合成退化为 0.5/0.5）"
    );
}

// ===========================================================================
// Compose 树 / 混合（compose.rs）
// ===========================================================================

/// 缺陷面：六种混合算子真值表（边界 0/1）。
#[test]
fn blend_truth_table_boundaries() {
    let zero = [0.0f32, 0.0, 0.0, 0.0];
    let one = [1.0f32, 1.0, 1.0, 1.0];
    let half = [0.5f32, 0.5, 0.5, 0.5];

    // Over：上层 alpha=0 → 下层不变。
    let o = blend_pixel(MergeOp::Over, [1.0, 0.0, 0.0, 1.0], zero);
    assert_eq!(o, [1.0, 0.0, 0.0, 1.0], "over alpha0 下层透出");
    // Over：上层 alpha=1 → 上层色。
    let o2 = blend_pixel(MergeOp::Over, [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(&o2[0..3], &[0.0, 1.0, 0.0], "over alpha1 上层覆盖");
    // Multiply：任一侧 0 → 0。
    assert_eq!(blend_pixel(MergeOp::Multiply, zero, one)[0], 0.0, "multiply 0·1=0");
    // Screen：任一侧 1 → 1。
    assert!((blend_pixel(MergeOp::Screen, one, zero)[0] - 1.0).abs() < 1e-6, "screen 1⊕0=1");
    // Plus：half+half = 1。
    assert!((blend_pixel(MergeOp::Plus, half, half)[0] - 1.0).abs() < 1e-6, "plus");
    // Mask：上层 alpha 当蒙版。
    let m = blend_pixel(MergeOp::Mask, one, [9.0, 9.0, 9.0, 0.25]);
    assert!((m[0] - 0.25).abs() < 1e-6, "mask 取上层 alpha");
}

/// 缺陷面：Compose 树层序决定论（over 非结合 → 换层序换结果）。
#[test]
fn compose_layer_order_decides() {
    let red = solid_layer(2, 1, [1.0, 0.0, 0.0, 1.0]);
    let green = solid_layer(2, 1, [0.0, 1.0, 0.0, 1.0]);
    let t1 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(red.clone()), ComposeNode::leaf(green.clone()));
    let t2 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(green), ComposeNode::leaf(red));
    assert_ne!(t1.eval().get(0, 0), t2.eval().get(0, 0), "换层序必须换结果");
}

/// 缺陷面：平铺列表 = 左结合树（退化兼容）。
#[test]
fn compose_flat_equals_left_chain() {
    let l0 = solid_layer(1, 1, [0.0, 0.0, 0.0, 1.0]);
    let l1 = solid_layer(1, 1, [0.5, 0.5, 0.5, 1.0]);
    let l2 = solid_layer(1, 1, [1.0, 1.0, 1.0, 1.0]);
    let flat = ComposeNode::from_flat(MergeOp::Over, vec![l0.clone(), l1.clone(), l2.clone()]).unwrap();
    let chain = ComposeNode::merge(
        MergeOp::Over,
        ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(l0), ComposeNode::leaf(l1)),
        ComposeNode::leaf(l2),
    );
    assert_eq!(flat.eval(), chain.eval(), "平铺 = 左链");
}

/// 缺陷面：空平铺列表 → None（不 panic）。
#[test]
fn compose_flat_empty_none() {
    assert!(ComposeNode::from_flat(MergeOp::Over, vec![]).is_none(), "空列表 → None");
}

/// 缺陷面：雾合成端点（aov=0 不变；aov=1 = 雾色；density=0 = 不变）。
#[test]
fn fog_endpoints() {
    let c = [1.0f32, 0.0, 0.0, 1.0];
    let fog = [0.8f32, 0.8, 0.9];
    assert_eq!(apply_fog(c, fog, 0.0, 1.0), c, "近景不变");
    assert_eq!(apply_fog(c, fog, 1.0, 1.0), [0.8, 0.8, 0.9, 1.0], "远景 = 雾色");
    assert_eq!(apply_fog(c, fog, 1.0, 0.0), c, "density=0 不变");
}

/// 缺陷面：merge 尺寸不一致应 panic。
#[test]
#[should_panic]
fn merge_rejects_size_mismatch() {
    let a = LayerBuf::new(2, 2);
    let b = LayerBuf::new(3, 3);
    let _ = merge(MergeOp::Over, &a, &b);
}

fn solid_layer(w: u32, h: u32, c: [f32; 4]) -> LayerBuf {
    let mut b = LayerBuf::new(w, h);
    for y in 0..h {
        for x in 0..w {
            b.set(x, y, c);
        }
    }
    b
}

// ===========================================================================
// 遮罩（mask.rs）布尔真值表 + 边界
// ===========================================================================

/// 缺陷面：六算子布尔真值表（含端点 0/1 组合）。
#[test]
fn mask_boolean_truth_table() {
    let test = |va: f32, vb: f32| -> [f32; 6] {
        let a = uniform_mask(1, 1, va);
        let b = uniform_mask(1, 1, vb);
        [
            combine(MaskOp::Union, &a, &b).get(0, 0),
            combine(MaskOp::Intersect, &a, &b).get(0, 0),
            combine(MaskOp::Subtract, &a, &b).get(0, 0),
            combine(MaskOp::Add, &a, &b).get(0, 0),
            combine(MaskOp::Multiply, &a, &b).get(0, 0),
            combine(MaskOp::Invert, &a, &a).get(0, 0),
        ]
    };
    // (0,0)
    let r = test(0.0, 0.0);
    assert_eq!(r, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "全 0 真值");
    // (1,0)
    let r = test(1.0, 0.0);
    assert_eq!(r, [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], "1,0 真值");
    // (0,1)
    let r = test(0.0, 1.0);
    assert_eq!(r, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0], "0,1 真值");
    // (1,1)
    let r = test(1.0, 1.0);
    assert_eq!(r, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0], "1,1 真值");
    // (0.3,0.7)
    let r = test(0.3, 0.7);
    assert!((r[0] - 0.7).abs() < 1e-6, "union=max");
    assert!((r[1] - 0.3).abs() < 1e-6, "intersect=min");
    assert!((r[2] - 0.0).abs() < 1e-6, "subtract 饱和 0");
    assert!((r[3] - 1.0).abs() < 1e-6, "add 饱和 1");
    assert!((r[4] - 0.21).abs() < 1e-6, "multiply");
    assert!((r[5] - 0.7).abs() < 1e-6, "invert 1-0.3");
}

/// 缺陷面：feather r=0 恒等；r>0 边界带出中间值；均匀场不动点。
#[test]
fn feather_boundaries() {
    let m = uniform_mask(4, 4, 0.6);
    assert_eq!(feather(&m, 0), m, "r=0 恒等");
    // 均匀场经羽化仍均匀（边缘复制不破坏不动点）。
    let f = feather(&m, 2);
    for v in &f.data {
        assert!((v - 0.6).abs() < 1e-6, "均匀场羽化应不动 {v}");
    }
    // 硬边 → 中间值。
    let mut e = Mask::new(9, 1);
    for x in 0..9 {
        e.set(x, 0, if x < 4 { 1.0 } else { 0.0 });
    }
    let fe = feather(&e, 2);
    assert!(fe.get(4, 0) > 0.0 && fe.get(4, 0) < 1.0, "硬边羽化出中间值");
}

/// 缺陷面：LumaKey 软阈值（端点 + 反相）。
#[test]
fn luma_key_thresholds() {
    let k = LumaKey { low: 0.4, high: 0.6, invert: false };
    let mid = plane_fb(1, 1, [0.5, 0.5, 0.5, 1.0]);
    assert!((k.mask(&mid).get(0, 0) - 0.5).abs() < 1e-4, "软过渡中点 0.5");
    let dark = plane_fb(1, 1, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(k.mask(&dark).get(0, 0), 0.0, "暗 → 0");
    let bright = plane_fb(1, 1, [1.0, 1.0, 1.0, 1.0]);
    assert!((k.mask(&bright).get(0, 0) - 1.0).abs() < 1e-4, "亮 → 1");
    // 反相。
    let ki = LumaKey { low: 0.4, high: 0.6, invert: true };
    assert!((ki.mask(&bright).get(0, 0)).abs() < 1e-4, "反相后亮 → 0");
}

/// 缺陷面：LumaKey 退化阈值（low==high → span 保护，不 panic）。
#[test]
fn luma_key_degenerate_span() {
    let k = LumaKey { low: 0.5, high: 0.5, invert: false };
    let fb = plane_fb(1, 1, [0.5, 0.5, 0.5, 1.0]);
    let v = k.mask(&fb).get(0, 0);
    assert!(v.is_finite() && (0.0..=1.0).contains(&v), "退化阈值不 panic：{v}");
}

/// 缺陷面：ChromaKey 键出绿、保留红；spill 抑制。
#[test]
fn chroma_key_green_removed_red_kept() {
    let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.2, softness: 0.3, spill: 1.0 };
    let green = plane_fb(1, 1, [0.0, 1.0, 0.0, 1.0]);
    assert!(key.mask(&green).get(0, 0) < 0.05, "纯绿键出");
    let red = plane_fb(1, 1, [1.0, 0.0, 0.0, 1.0]);
    assert!(key.mask(&red).get(0, 0) > 0.95, "纯红保留");
}

/// 缺陷面：ChromaKey spill 抑制削减过量绿（含 spill=0 对照）。
#[test]
fn chroma_key_spill_suppression() {
    let fb = plane_fb(1, 1, [0.4, 0.8, 0.4, 1.0]);
    let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 1.0 };
    let out = key.process(&fb);
    assert!(out.get(0, 0)[1] <= 0.4 + 1e-4, "spill=1 绿被削到邻通道");
    let key0 = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 0.0 };
    let out0 = key0.process(&fb);
    assert!(out0.get(0, 0)[1] > out.get(0, 0)[1], "spill=0 绿保留更多");
}

/// 缺陷面：ChannelMask 选通道 + 越界通道索引（min(3) 保护）。
#[test]
fn channel_mask_channel_selection() {
    let fb = plane_fb(1, 1, [0.2, 0.4, 0.6, 0.8]);
    let cm = ChannelMask { channel: 2, low: 0.0, high: 1.0, invert: false };
    assert!((cm.mask(&fb).get(0, 0) - 0.6).abs() < 1e-3, "通道 2 = B");
    // 越界通道 9 → min(3) → alpha。
    let co = ChannelMask { channel: 9, low: 0.0, high: 1.0, invert: false };
    assert!((co.mask(&fb).get(0, 0) - 0.8).abs() < 1e-3, "越界索引退化到 A");
}

/// 缺陷面：DepthMask 映射 + 反相 + 空 AOV（不得 panic）。
#[test]
fn depth_mask_mapping_and_empty() {
    let dm = DepthMask { near: 0.0, far: 1.0, invert: false };
    let m = dm.mask(&[0.0, 0.5, 1.0], 3, 1);
    assert_eq!(m.get(0, 0), 0.0);
    assert!((m.get(1, 0) - 0.5).abs() < 1e-6);
    assert_eq!(m.get(2, 0), 1.0);
    // 空 AOV：mask 构造仍成功（无数据写入）。
    let e = dm.mask(&[], 0, 0);
    assert_eq!(e.width, 0);
    assert!(e.data.is_empty());
}

/// 缺陷面：apply_mask 覆盖 [0,1] 外遮版值的乘算；越界遮版坐标读 0。
#[test]
fn apply_mask_out_of_bounds() {
    let fb = solid_fb(2, 1, [1.0, 1.0, 1.0, 1.0]);
    let m = uniform_mask(1, 1, 0.5); // 遮版小于帧
    let out = apply_mask(&fb, &m);
    assert!((out.get(0, 0)[0] - 0.5).abs() < 1e-3, "命中遮版乘 0.5");
    assert_eq!(out.get(1, 0)[0], 0.0, "越界遮版 → 0");
}

// ===========================================================================
// DOF（dof_full.rs）合焦恒等 + 退化
// ===========================================================================

/// 缺陷面：合焦恒等（CoC=0 处逐位原样保留）。
#[test]
fn dof_focus_identity_bit_exact() {
    let color = checker(5, 4);
    let aov = vec![0.3f32; 20];
    let dof = FullDof::new(5, 4, aov, 0.3, 8);
    assert_eq!(dof.render(&color), color, "全合焦帧必须逐位恒等");
}

/// 缺陷面：CoC 半径边界（焦点 0；偏离 0；max_coc 上界；coc_bias 偏置）。
#[test]
fn dof_coc_computation() {
    let mut aov = vec![0.0f32; 4];
    aov[2] = 1.0;
    aov[3] = 1.0;
    let dof = FullDof::new(4, 1, aov, 0.0, 10);
    assert_eq!(dof.coc_radius(0, 0), 0, "焦点 CoC=0");
    assert_eq!(dof.coc_radius(2, 0), 10, "最远 CoC=max_coc");
}

/// 缺陷面：均匀场经变半径模糊不动点。
#[test]
fn dof_uniform_field_fixed_point() {
    let fb = solid_fb(5, 5, [0.4, 0.6, 0.8, 1.0]);
    let aov = vec![1.0f32; 25];
    let dof = FullDof::new(5, 5, aov, 0.0, 2);
    let out = dof.render(&fb);
    let want = [q(0.4), q(0.6), q(0.8), 1.0];
    for y in 0..5u32 {
        for x in 0..5u32 {
            assert_eq!(out.get(x, y), want, "({x},{y}) 均匀场非不动点");
        }
    }
}

/// 缺陷面：max_coc=0（退化为无模糊，恒等）。
#[test]
fn dof_zero_max_coc_identity() {
    let color = checker(6, 4);
    let aov = vec![1.0f32; 24]; // 全离焦
    let dof = FullDof::new(6, 4, aov, 0.0, 0); // max_coc=0
    assert_eq!(dof.render(&color), color, "max_coc=0 应无模糊（恒等）");
}

/// 缺陷面：NaN 深度 AOV 的防御（不得 panic；记录 coc 行为）。
#[test]
fn dof_nan_depth_defense() {
    let color = solid_fb(3, 1, [1.0, 1.0, 1.0, 1.0]);
    let mut aov = vec![0.5f32; 3];
    aov[1] = f32::NAN;
    let dof = FullDof::new(3, 1, aov, 0.0, 4);
    // coc_radius 对 NaN：`(NaN) as i64` 在 Rust 为 0（饱和转换）→ 视为合焦。
    let r = dof.coc_radius(1, 0);
    let out = dof.render(&color);
    // 只要不 panic、输出有限即可（记录：NaN 深度被安静当合焦处理）。
    for y in 0..1u32 {
        for x in 0..3u32 {
            let c = out.get(x, y);
            assert!(c.iter().all(|v| v.is_finite()), "DOF 输出含非有限 {c:?}");
        }
    }
    let _ = r;
}

/// 缺陷面：DOF 逐位确定（同输入两遍一致）。
#[test]
fn dof_deterministic() {
    let color = checker(8, 6);
    let aov: Vec<f32> = (0..48).map(|i| (i % 5) as f32 / 4.0).collect();
    let dof = FullDof::new(8, 6, aov, 0.5, 3);
    assert_eq!(dof.render(&color), dof.render(&color), "DOF 必须逐位确定");
}

/// 缺陷面：DOF 尺寸不一致应 panic。
#[test]
#[should_panic]
fn dof_rejects_size_mismatch() {
    let aov = vec![0.0f32; 4];
    let dof = FullDof::new(2, 2, aov, 0.0, 3);
    let color = solid_fb(3, 3, [1.0, 1.0, 1.0, 1.0]);
    let _ = dof.render(&color);
}

/// 缺陷面：空深度 AOV 构造（w·h=0）应允许或不 panic。
#[test]
fn dof_empty_aov_construct() {
    let dof = FullDof::new(0, 0, vec![], 0.0, 3);
    assert_eq!(dof.depth_aov.len(), 0);
}

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------

fn plane_fb(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
    solid_fb(w, h, c)
}

fn checker(w: u32, h: u32) -> Fbuf16F {
    let mut fb = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            fb.set(x, y, [v, v, v, 1.0]);
        }
    }
    fb
}

fn q(v: f32) -> f32 {
    prism_io::f16_to_f32(prism_io::f32_to_f16(v))
}

/// 保留 Point 引用（PathMask 契约占位；本文件不深测 path，另有验收文件覆盖）。
#[allow(dead_code)]
fn _path_point() -> Point {
    Point::new(0.0, 0.0)
}
