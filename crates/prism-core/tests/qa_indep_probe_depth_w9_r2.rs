//! W9 Round-2 独立复核探针（QA 严过关 Edward）——P1 逐像素深度合成。
//! 只读探针，不改源码。

use prism_core::depth::{DepthComposite, DepthLayer};

fn layer(w: u32, h: u32, fill: ([f32; 4], f32)) -> DepthLayer {
    let mut l = DepthLayer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            l.set(x, y, fill.0, fill.1);
        }
    }
    l
}

/// P1-a：逐像素 depth 决定权重，而非全局 layer_z。
/// 构造：back 左半近(z=1) 右半远(z=9)；front 全像素近(z=1)。
/// 若 composite 真的逐像素看 depth，则左右半结果不同（左：等距融合；
/// 右：front 近压制 back）。若忽略逐像素（旧缺陷，全用 layer_z）→ 左右相同。
#[test]
fn p1_per_pixel_depth_weights_differ_left_right() {
    let w = 4u32;
    let h = 1u32;
    let dc = DepthComposite { tolerance: 1.0 };

    let mut back = DepthLayer::new(w, h);
    let mut front = DepthLayer::new(w, h);
    for x in 0..w {
        // back: 左两像素 z=1（近），右两像素 z=9（远）
        let zb = if x < 2 { 1.0 } else { 9.0 };
        back.set(x, 0, [0.0, 0.0, 0.0, 1.0], zb);
        // front: 全部 z=1（近），颜色白
        front.set(x, 0, [1.0, 1.0, 1.0, 1.0], 1.0);
    }
    let out = dc.composite(&back, &front, 5.0, 5.0);
    let left = out.get(0, 0)[0];
    let right = out.get(3, 0)[0];
    assert!(
        (left - right).abs() > 0.1,
        "逐像素 depth 生效则左右半权重不同；实测 left={left} right={right}（相等=仍用全局 layer_z，P1 未修）"
    );
    // 右侧：front(z=1) 远近于 back(z=9)，front 应几乎全权 → 接近白 (1.0)。
    assert!(right > 0.9, "右侧 front 近应压制 back，实测 {right}");
    // 左侧：等距 → 约 0.5 融合。
    assert!((left - 0.5).abs() < 0.15, "左侧等距应约 0.5 融合，实测 {left}");
}

/// P1-b：无命中像素（depth < 0）回退 layer_z。
/// front 有命中近(z=1)，back 全像素无命中(-1) → back 回退其 layer_z。
/// layer_z_back 远(9) → front 压制 → 近白。
#[test]
fn p1_miss_falls_back_to_layer_z() {
    let dc = DepthComposite { tolerance: 1.0 };
    let mut back = DepthLayer::new(2, 1);
    for x in 0..2 {
        back.set(x, 0, [0.0, 0.0, 0.0, 1.0], -1.0); // 无命中
    }
    let front = layer(2, 1, ([1.0, 1.0, 1.0, 1.0], 1.0));
    let out = dc.composite(&back, &front, 9.0, 1.0); // back layer_z=9 远，front layer_z=1 近
    for x in 0..2 {
        assert!(out.get(x, 0)[0] > 0.9, "无命中回退 layer_z 后 front 近应胜，实测 {}", out.get(x, 0)[0]);
    }
}

/// P1-c：两层都无命中 → 权重和为 0 → 取前层（不 panic、不 NaN）。
#[test]
fn p1_both_miss_no_nan() {
    let dc = DepthComposite { tolerance: 0.0 };
    let back = layer(2, 1, ([0.0, 0.0, 0.0, 1.0], -1.0));
    let front = layer(2, 1, ([1.0, 0.0, 0.0, 1.0], -1.0));
    // layer_z 均为负 → 两 soft_weight 都 0。
    let out = dc.composite(&back, &front, -1.0, -1.0);
    for x in 0..2 {
        let c = out.get(x, 0);
        assert!(c.iter().all(|v| v.is_finite()), "不得 NaN/Inf，实测 {:?}", c);
    }
}
