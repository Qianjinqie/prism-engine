//! 合成图节点（C1）：Compose 树——双输入合成算子任意嵌套、层序决定论、
//! 混合模式（over/plus/screen/multiply/mask）。
//!
//! 语义权威："节点式无限层：Composite2D 从平铺层列表
//! 升级为 Compose 树：双输入合成算子（over/plus/screen/multiply/mask）任意
//! 嵌套；平铺列表 = 树的退化形式，完全兼容；层数无上限"；
//! `Merge(op)` 归约算子；prism-core `node::MergeOp`。
//!
//! # 确定性关键
//!
//! - 全部合并为**逐像素单遍**运算，结合序由归约形状（graph::reduce）唯一给定
//!   ——同树同遍历序、同浮点结合序，任意执行拓扑逐位一致；
//! - **层序决定论**：树的前序遍历给 A（左/下层）后 B（右/上层）；over 语义
//!   一律"B 覆盖 A"——层序固定则输出逐位固定。

use prism_math::clamp;

use crate::graph::reduce;
use crate::node::MergeOp;

/// 合成输入：一张 RGBA 图（场景线性域，行主序，长度 = w·h·4）。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBuf {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 逐像素 RGBA。
    pub rgba: Vec<f32>,
}

impl LayerBuf {
    /// 构造全透明透明层。
    ///
    /// # Panics
    /// 尺寸为 0 时 panic。
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "合成层尺寸必须为正");
        Self { width, height, rgba: vec![0.0; (width as usize) * (height as usize) * 4] }
    }

    /// 以给定逐像素 RGBA 构造（长度必须 = w·h·4）。
    ///
    /// # Panics
    /// 长度不符时 panic。
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<f32>) -> Self {
        assert_eq!(
            rgba.len(),
            (width as usize) * (height as usize) * 4,
            "RGBA 长度必须等于 w·h·4"
        );
        Self { width, height, rgba }
    }

    /// 写入一个像素。
    ///
    /// # Panics
    /// 坐标越界时 panic。
    pub fn set(&mut self, x: u32, y: u32, c: [f32; 4]) {
        assert!(x < self.width && y < self.height, "层坐标越界 ({x},{y})");
        let i = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        self.rgba[i..i + 4].copy_from_slice(&c);
    }

    /// 读取一个像素。
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        let i = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }
}

/// 逐像素混合核心：`a` 为下层、`b` 为上层，返回混合结果（固定表达式序）。
///
/// - `Over`：`a·(1−αb) + b·αb`（标准 over）；αa 同理合成 alpha；
/// - `Plus`：`a + b`（加算）；
/// - `Screen`：`1 − (1−a)(1−b)`（滤色）；
/// - `Multiply`：`a·b`（正片叠底）；
/// - `Mask`：`a·αb`（上层 alpha 当蒙版，只保留 alpha 通道）。
///
/// RGB 分量走混合式，alpha 通道固定取 `max(αa, αb)` 的覆盖式（mask 除外，
/// 取 `αa·αb`）——所有公式为有限项四则运算，逐位确定。
pub fn blend_pixel(op: MergeOp, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    match op {
        MergeOp::Over => {
            let ab = b[3];
            let aa = a[3];
            out[3] = ab + aa * (1.0 - ab);
            for i in 0..3 {
                out[i] = a[i] * (1.0 - ab) + b[i] * ab;
            }
        }
        MergeOp::Plus => {
            for i in 0..4 {
                out[i] = a[i] + b[i];
            }
        }
        MergeOp::Screen => {
            for i in 0..3 {
                out[i] = 1.0 - (1.0 - a[i]) * (1.0 - b[i]);
            }
            out[3] = a[3].max(b[3]);
        }
        MergeOp::Multiply => {
            for i in 0..3 {
                out[i] = a[i] * b[i];
            }
            out[3] = a[3].max(b[3]);
        }
        MergeOp::Mask => {
            let m = b[3];
            for i in 0..3 {
                out[i] = a[i] * m;
            }
            out[3] = a[3] * m;
        }
    }
    out
}

/// 合并两个同尺寸层（逐像素单遍，固定顺序）。
///
/// # Panics
/// 尺寸不一致时 panic。
pub fn merge(op: MergeOp, a: &LayerBuf, b: &LayerBuf) -> LayerBuf {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "合并两层尺寸必须一致"
    );
    let n = (a.width as usize) * (a.height as usize);
    let mut rgba = vec![0.0f32; n * 4];
    for i in 0..n {
        let c = i * 4;
        let ca = [a.rgba[c], a.rgba[c + 1], a.rgba[c + 2], a.rgba[c + 3]];
        let cb = [b.rgba[c], b.rgba[c + 1], b.rgba[c + 2], b.rgba[c + 3]];
        let m = blend_pixel(op, ca, cb);
        rgba[c..c + 4].copy_from_slice(&m);
    }
    LayerBuf { width: a.width, height: a.height, rgba }
}

/// 合成图节点（Compose 树）：叶子 = 输入层，内部 = 双输入合成算子。
///
/// 平铺层列表 = 退化树（左结合链）；层数无上限（求值按需
/// 递归，流式）。同树同求值序 ⇒ 逐位一致（层序决定论）。
#[derive(Debug, Clone, PartialEq)]
pub enum ComposeNode {
    /// 叶：一张输入层（携带数据，便于自包含求值与测试）。
    Leaf(LayerBuf),
    /// 内部：`op` 合并左（下）右（上）两棵子树。
    Merge {
        /// 混合模式。
        op: MergeOp,
        /// 左子树（下层）。
        left: Box<ComposeNode>,
        /// 右子树（上层）。
        right: Box<ComposeNode>,
    },
}

impl ComposeNode {
    /// 构造叶子。
    pub fn leaf(buf: LayerBuf) -> Self {
        Self::Leaf(buf)
    }

    /// 构造内部合并节点。
    pub fn merge(op: MergeOp, left: ComposeNode, right: ComposeNode) -> Self {
        Self::Merge { op, left: Box::new(left), right: Box::new(right) }
    }

    /// 从平铺层列表左结合构造（退化树；空列表返回 None）。
    pub fn from_flat(op: MergeOp, mut layers: Vec<LayerBuf>) -> Option<Self> {
        if layers.is_empty() {
            return None;
        }
        let mut acc = Self::leaf(layers.remove(0));
        for layer in layers {
            acc = Self::merge(op, acc, Self::leaf(layer));
        }
        Some(acc)
    }

    /// 求值：递归合并，返回合成结果。
    ///
    /// 归约纪律：多路同 op 的平铺链经 [`reduce`] 走固定形状
    /// 结合树——但为保持"逐层 over"的层序语义（over 非结合时结果依赖层序），
    /// 本求值对 `Merge` 节点严格按**树形**（左结合）递归，不重组；层序决定论
    /// 由此保证。返回 None 表示叶为空（不应发生）。
    pub fn eval(&self) -> LayerBuf {
        match self {
            Self::Leaf(buf) => buf.clone(),
            Self::Merge { op, left, right } => {
                let a = left.eval();
                let b = right.eval();
                merge(*op, &a, &b)
            }
        }
    }

    /// 求值——多输入版：对一组同 op 平铺层走固定形状归约（性能路径）。
    ///
    /// 语义与左结合 `eval` 在 op 结合时一致；非结合 op（over）下调用方应
    /// 用 [`ComposeNode::eval`] 的树形求值。
    pub fn eval_flat(op: MergeOp, layers: &[LayerBuf]) -> Option<LayerBuf> {
        if layers.is_empty() {
            return None;
        }
        let combined = reduce(
            layers.to_vec(),
            |a, b| merge(op, &a, &b),
            LayerBuf::new(1, 1),
        );
        Some(combined)
    }
}

/// 雾合成：按深度把雾色线性插值到颜色上（例：2D 元素进 3D 雾被
/// 正确雾化）。`density` ∈ [0,1]，1 = 完全雾色；`aov` 为该像素深度 AOV
/// （0 近 1 远）。
pub fn apply_fog(rgba: [f32; 4], fog_color: [f32; 3], aov: f32, density: f32) -> [f32; 4] {
    let t = clamp(aov * density, 0.0, 1.0);
    [
        rgba[0] * (1.0 - t) + fog_color[0] * t,
        rgba[1] * (1.0 - t) + fog_color[1] * t,
        rgba[2] * (1.0 - t) + fog_color[2] * t,
        rgba[3],
    ]
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> LayerBuf {
        let mut b = LayerBuf::new(w, h);
        for y in 0..h {
            for x in 0..w {
                b.set(x, y, c);
            }
        }
        b
    }

    #[test]
    fn over_upper_covers_lower_by_alpha() {
        // 下层不透明红，上层半透明蓝 → over = 0.5红 + 0.5蓝（RGB），alpha=1。
        let a = solid(1, 1, [1.0, 0.0, 0.0, 1.0]);
        let b = solid(1, 1, [0.0, 0.0, 1.0, 0.5]);
        let r = merge(MergeOp::Over, &a, &b).get(0, 0);
        assert!((r[0] - 0.5).abs() < 1e-6 && (r[2] - 0.5).abs() < 1e-6, "over 混合: {r:?}");
        assert!((r[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multiply_and_screen_are_reference_formulas() {
        let a = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let b = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let m = merge(MergeOp::Multiply, &a, &b).get(0, 0);
        assert!((m[0] - 0.25).abs() < 1e-6, "multiply: {m:?}");
        let s = merge(MergeOp::Screen, &a, &b).get(0, 0);
        assert!((s[0] - 0.75).abs() < 1e-6, "screen: {s:?}");
    }

    #[test]
    fn mask_keeps_only_alpha_of_upper() {
        let a = solid(1, 1, [1.0, 1.0, 1.0, 1.0]);
        let b = solid(1, 1, [9.0, 9.0, 9.0, 0.25]); // 上层通道值无关，只看 alpha
        let r = merge(MergeOp::Mask, &a, &b).get(0, 0);
        assert!((r[0] - 0.25).abs() < 1e-6 && (r[3] - 0.25).abs() < 1e-6, "mask: {r:?}");
    }

    #[test]
    fn compose_tree_layer_order_is_deterministic() {
        let red = solid(2, 1, [1.0, 0.0, 0.0, 1.0]);
        let green = solid(2, 1, [0.0, 1.0, 0.0, 1.0]);
        // (red over green) vs (green over red)：上层不同 → 结果不同（层序决定论）。
        let t1 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(red.clone()), ComposeNode::leaf(green.clone()));
        let t2 = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(green), ComposeNode::leaf(red));
        let r1 = t1.eval().get(0, 0);
        let r2 = t2.eval().get(0, 0);
        // 上层不透明 → over 结果 = 上层色。
        assert!((r1[1] - 1.0).abs() < 1e-6, "r1 上层绿: {r1:?}");
        assert!((r2[0] - 1.0).abs() < 1e-6, "r2 上层红: {r2:?}");
        assert_ne!(r1, r2, "换层序必须换结果");
    }

    #[test]
    fn compose_tree_eval_is_pure() {
        let a = solid(3, 2, [0.2, 0.4, 0.6, 1.0]);
        let b = solid(3, 2, [0.1, 0.1, 0.1, 0.5]);
        let tree = ComposeNode::merge(MergeOp::Over, ComposeNode::leaf(a), ComposeNode::leaf(b));
        assert_eq!(tree.eval(), tree.eval(), "同树两次求值逐位一致");
    }

    #[test]
    fn flat_layers_degenerate_into_left_chain() {
        // 平铺列表 = 左结合树（完全兼容）。
        let l0 = solid(1, 1, [0.0, 0.0, 0.0, 1.0]);
        let l1 = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let l2 = solid(1, 1, [1.0, 1.0, 1.0, 1.0]);
        let flat = ComposeNode::from_flat(MergeOp::Over, vec![l0, l1, l2]).unwrap();
        let chain = ComposeNode::merge(
            MergeOp::Over,
            ComposeNode::merge(
                MergeOp::Over,
                ComposeNode::leaf(solid(1, 1, [0.0, 0.0, 0.0, 1.0])),
                ComposeNode::leaf(solid(1, 1, [0.5, 0.5, 0.5, 1.0])),
            ),
            ComposeNode::leaf(solid(1, 1, [1.0, 1.0, 1.0, 1.0])),
        );
        assert_eq!(flat.eval(), chain.eval(), "平铺列表必须等价于左结合树");
    }

    #[test]
    fn apply_fog_pulls_toward_fog_color_with_distance() {
        // 远景（aov=1）→ 全雾色；近景（aov=0）→ 原色。
        let near = apply_fog([1.0, 0.0, 0.0, 1.0], [0.8, 0.8, 0.9], 0.0, 1.0);
        assert_eq!(near, [1.0, 0.0, 0.0, 1.0], "近景不被雾化");
        let far = apply_fog([1.0, 0.0, 0.0, 1.0], [0.8, 0.8, 0.9], 1.0, 1.0);
        assert_eq!(far, [0.8, 0.8, 0.9, 1.0], "远景 = 雾色");
    }

    #[test]
    fn flat_reduce_matches_left_chain_for_associative_op() {
        // plus 结合 → 固定形状归约与左链结果一致（结合序确定）。
        let layers = vec![
            solid(1, 1, [0.1, 0.0, 0.0, 0.0]),
            solid(1, 1, [0.2, 0.0, 0.0, 0.0]),
            solid(1, 1, [0.3, 0.0, 0.0, 0.0]),
        ];
        let flat = ComposeNode::eval_flat(MergeOp::Plus, &layers).unwrap();
        let v = flat.get(0, 0)[0];
        assert!((v - 0.6).abs() < 1e-6, "plus 归约 = 0.1+0.2+0.3 = 0.6，实际 {v}");
    }
}
