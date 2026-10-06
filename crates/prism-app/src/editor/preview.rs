//! 编辑器预览渲染器：**草稿档位 + 子图签名增量**的落地。
//!
//! # 核心洞察（U13）
//! "预览 = 草稿档位 + 子图签名增量"。本模块是该命题的执行端：
//!
//! 1. **草稿档位**（[`PreviewQuality`]）：`Draft` 用 1/4 内部分辨率 + 1 spp，
//!    `Final` 用全分辨率 + 16 spp。两者不仅采样数不同，**内部分辨率也不同**，
//!    因此 Draft 既快又可见地糙（放大后呈块状），Final 平滑。
//! 2. **子图签名增量**：逐节点求值时以
//!    `cache_key(节点子树签名, 渲染维度)`（[`cache_key`]，架构 §4.2 五元组 +
//!    tile + 帧，C13 版本入键）为键做**中间缓冲记忆化**。签名未变的节点直接
//!    复用上一帧缓冲——**不重算**。改叶 1 只重算 `{1, 3, 4}`，兄弟叶 2 命中缓存。
//!
//! 记忆化集合与 [`crate::editor::DirtySet`]（签名比对得出的受影响子图）**恰好
//! 相同**：两者都由"子树签名是否变化"决定，故 `recomputed` 与 `dirty` 集合相等
//! 是本设计的**不变式**（见集成测试 `incremental_recompute_set_equals_dirty_set`）。
//!
//! # 确定性（铁律：同输入逐字节可复现）
//! 预览图案**只用整数散列 + 多项式插值 + 四则运算**生成，**不调用任何超越
//! 函数**（`sin`/`cos`/`exp`/`pow` 等 libm）。理由：libm 的超越函数在不同
//! 平台/版本上末位可能不同，会击穿"同输入同 md5"。工程权威指纹
//! （[`crate::editor::canonical_bytes`] / 签名树）本身已是纯整数 SHA-256；
//! 预览若引入 libm 不确定性，会让"同图同预览"这条辅助不变量失效。
//! 代价是图案偏"几何"而非"自然"，对编辑器预览完全够用。
//!
//! # 零第三方依赖
//! 仅 `std` + workspace 内 `prism-core`（图 / 签名 / 缓存键 / 节点模型）。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::node::{MergeOp, NodeId, Params, StructureId};
use prism_core::signature::{cache_key, graph_signatures, CacheDims, ChannelDefs};
use prism_core::Graph;

use super::state::PreviewQuality;

/// 预览渲染错误（三段式，宪章 C18/C19；错误码取 E4xxx 段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreviewError {
    /// 图含环，无法求值（`topo_order` 透传）。
    Cyclic {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：底层拓扑错误文本。
        detail: String,
    },
    /// 输出尺寸越界（0 或超过 [`MAX_DIM`]）。
    BadSize {
        /// 三段式一【在哪】。
        location: String,
        /// 请求的宽。
        width: u32,
        /// 请求的高。
        height: u32,
    },
    /// 找不到根节点（图中无"无下游"节点；正常图必有，此为防御性分支）。
    NoRoot {
        /// 三段式一【在哪】。
        location: String,
    },
}

impl std::fmt::Display for PreviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreviewError::Cyclic { location, detail } => write!(
                f,
                "error[E4101] {location}\n  什么事：图含环，无法按拓扑序求值：{detail}\n  建议：先断开造成环的连线（编辑器不提供成环编辑），再重新预览"
            ),
            PreviewError::BadSize { location, width, height } => write!(
                f,
                "error[E4102] {location}\n  什么事：预览尺寸 {width}x{height} 非法（须 1..={MAX_DIM}）\n  建议：把宽高降到 {MAX_DIM} 以内再预览"
            ),
            PreviewError::NoRoot { location } => write!(
                f,
                "error[E4103] {location}\n  什么事：图中找不到根节点（无任何下游的节点）\n  建议：检查图是否为空；空图无法预览"
            ),
        }
    }
}

impl std::error::Error for PreviewError {}

/// 预览输出尺寸上限（单边）。超限返回 [`PreviewError::BadSize`] 而非截断，
/// 免得"悄悄降质"（宪章 C20：不静默）。
pub const MAX_DIM: u32 = 512;

/// Draft 档位的内部分辨率缩约因子（宽高各缩 1/DRAFT_DOWNSCALE 后求值再最近邻放大）。
pub const DRAFT_DOWNSCALE: u32 = 4;

/// 预览缓冲键：(节点 id, 缓存键)。
///
/// 缓存键已含子图签名、分辨率、spp 档、tile、帧、schema/engine 版本
/// （[`cache_key`]），故本元组足以唯一标识一份可复用的中间缓冲。
type BufKey = (NodeId, prism_core::node::Hash256);

/// 渲染统计：本次预览的增量行为（供 UI 展示与测试断言）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// 本次**重算**的节点（id 升序）——签名已变，缓存未命中。
    pub recomputed: Vec<NodeId>,
    /// 本次**命中缓存复用**的节点（id 升序）——签名未变，零重算。
    pub reused: Vec<NodeId>,
    /// 实际使用的每像素样本数。
    pub spp: u32,
    /// 内部求值分辨率宽（Draft 小于输出宽）。
    pub internal_width: u32,
    /// 内部求值分辨率高。
    pub internal_height: u32,
    /// 参与求值的节点总数（根的祖先锥）。
    pub cone_size: usize,
}

/// 一次预览渲染的产物。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// 输出宽（像素）。
    pub width: u32,
    /// 输出高（像素）。
    pub height: u32,
    /// RGBA8 像素缓冲，长度 = `width * height * 4`，行优先。
    pub rgba: Vec<u8>,
    /// 增量统计。
    pub stats: RenderStats,
}

/// 预览渲染器：持有跨帧的中间缓冲缓存（增量的"记忆"载体）。
///
/// 生命周期与编辑器一致：整个编辑会话共用一个 [`PreviewRenderer`]，每次
/// [`render`](PreviewRenderer::render) 后缓存累积；撤销/重做回到旧参数时，
/// 因签名逐位还原，**全部命中缓存**、零重算（确定性 + 增量的直接体现）。
#[derive(Debug, Clone, Default)]
pub struct PreviewRenderer {
    /// 中间缓冲缓存（键见 [`BufKey`];值为该节点的 RGB f32 平面缓冲）。
    ///
    /// 上限 [`CACHE_CAP`] 条，超出时按插入序淘汰最旧（`Vec` 头部），保持
    /// 缓存有界——wasm 线性内存不会无限增长。
    cache: BTreeMap<BufKey, Vec<f32>>,
    /// 缓存插入序（用于有界淘汰）。
    order: Vec<BufKey>,
}

/// 中间缓冲缓存条数上限。参考图 4 节点 + 编辑期新增若干节点后仍远小于此值；
/// 超出即淘汰最旧，语义仍是"签名未变可复用"，只是更早的帧不再命中。
pub const CACHE_CAP: usize = 64;

impl PreviewRenderer {
    /// 新建空渲染器（无缓存）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 缓存当前条数（UI 展示 / 测试用）。
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// 清空缓存（切图 / 切引擎版本后调用）。
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// 渲染一帧预览。
    ///
    /// # 签名参数
    /// - `graph`：当前图（由编辑器模型现建，含全部节点与连线）。
    /// - `defs`：通道定义表（进签名；空表即无通道绑定）。
    /// - `quality`：质量档位（决定内部分辨率与 spp）。
    /// - `width`/`height`：输出尺寸（1..=[`MAX_DIM`]）。
    /// - `engine_version`：引擎版本（进缓存键，C13）。
    /// - `frame`：帧号（进缓存键，D-28 帧维必须可区分）。
    ///
    /// # Errors
    /// 尺寸非法 → [`PreviewError::BadSize`]；图含环 → [`PreviewError::Cyclic`]；
    /// 图空 / 无根 → [`PreviewError::NoRoot`]。
    pub fn render(
        &mut self,
        graph: &Graph,
        defs: &ChannelDefs,
        quality: PreviewQuality,
        width: u32,
        height: u32,
        engine_version: u64,
        frame: u64,
    ) -> Result<Rendered, PreviewError> {
        let location = "PreviewRenderer::render".to_string();
        if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
            return Err(PreviewError::BadSize { location, width, height });
        }

        // 拓扑序：环 → 结构化报错，绝不死循环（C20）。
        let order = graph
            .topo_order()
            .map_err(|e| PreviewError::Cyclic { location: location.clone(), detail: e.problem })?;

        let sigs = graph_signatures(graph, defs)
            .map_err(|e| PreviewError::Cyclic { location: location.clone(), detail: e.to_string() })?;

        // 根节点：无下游者。多个时取 id 最大者（确定性 tie-break）。
        let root = pick_root(graph, &order).ok_or_else(|| PreviewError::NoRoot { location: location.clone() })?;

        // 根的祖先锥：只有锥内节点参与求值（锥外节点与输出无关，不重算也不计数）。
        let cone: BTreeSet<NodeId> = ancestor_cone(graph, &order, root);

        // 草稿档位：内部分辨率缩约 + 低采样；成片：全分辨率 + 高采样。
        let (iw, ih) = match quality {
            PreviewQuality::Draft => (
                div_ceil(width, DRAFT_DOWNSCALE).max(1),
                div_ceil(height, DRAFT_DOWNSCALE).max(1),
            ),
            PreviewQuality::Final => (width, height),
        };
        let spp = quality.spp().max(1);

        // 渲染维度进缓存键：分辨率 + spp 档 + 帧 + 引擎版本（§4.2 / C13 / D-28）。
        let dims = CacheDims {
            width: iw,
            height: ih,
            spp_tier: quality.spp_tier(),
            tile: (0, 0),
            engine_version,
            frame,
        };

        let mut recomputed: Vec<NodeId> = Vec::new();
        let mut reused: Vec<NodeId> = Vec::new();

        // 逐节点求值（拓扑序保证上游已算完）。锥外节点跳过。
        for &id in &order {
            if !cone.contains(&id) {
                continue;
            }
            let key: BufKey = (id, cache_key(&sigs[&id], &dims));
            if self.cache.contains_key(&key) {
                // 签名未变 → 复用，零重算（增量的核心收益）。
                self.touch(&key);
                reused.push(id);
                continue;
            }
            let node = match graph.node(id) {
                Some(n) => n,
                None => continue, // 拓扑序覆盖全部节点，理论不可达；防御性跳过
            };
            let inputs: Vec<(u8, NodeId)> = graph.inputs_of(id);
            let buf = self.eval_node(node, &inputs, &cone, &sigs, &dims, iw, ih, spp);
            self.insert(key, buf);
            recomputed.push(id);
        }

        // 合成输出：根缓冲 → （Draft 时最近邻放大）→ RGBA8。
        let root_key: BufKey = (root, cache_key(&sigs[&root], &dims));
        let root_buf: Vec<f32> = self.cache.get(&root_key).cloned().unwrap_or_else(|| {
            // 理论不可达（root ∈ cone 且刚被算出或命中）；兜底为黑图，不 panic。
            vec![0.0; (iw as usize) * (ih as usize) * 3]
        });
        let rgba = to_rgba(&root_buf, iw, ih, width, height);

        Ok(Rendered {
            width,
            height,
            rgba,
            stats: RenderStats {
                recomputed,
                reused,
                spp,
                internal_width: iw,
                internal_height: ih,
                cone_size: cone.len(),
            },
        })
    }

    /// 更新缓存最近使用序（命中时移到末尾，用于有界淘汰）。
    fn touch(&mut self, key: &BufKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }

    /// 插入缓存并在超上限时淘汰最旧条目。
    fn insert(&mut self, key: BufKey, buf: Vec<f32>) {
        if self.cache.contains_key(&key) {
            self.touch(&key);
            return;
        }
        self.cache.insert(key.clone(), buf);
        self.order.push(key);
        while self.order.len() > CACHE_CAP {
            let oldest = self.order.remove(0);
            self.cache.remove(&oldest);
        }
    }

    /// 求值单节点为一个 RGB f32 平面缓冲（`iw*ih*3`）。
    ///
    /// 锥外输入（理论上不存在——锥是祖先闭包）按无输入处理，绝不 panic。
    #[allow(clippy::too_many_arguments)]
    fn eval_node(
        &self,
        node: &prism_core::node::Node,
        inputs: &[(u8, NodeId)],
        cone: &BTreeSet<NodeId>,
        sigs: &BTreeMap<NodeId, prism_core::node::Hash256>,
        dims: &CacheDims,
        iw: u32,
        ih: u32,
        spp: u32,
    ) -> Vec<f32> {
        let n = (iw as usize) * (ih as usize);
        let mut out = vec![0.0f32; n * 3];

        // 取输入缓冲（已在拓扑序算完并入缓存）。
        let in_bufs: Vec<&Vec<f32>> = inputs
            .iter()
            .filter(|(_, from)| cone.contains(from))
            .filter_map(|(_, from)| {
                let key: BufKey = (*from, cache_key(&sigs[from], dims));
                self.cache.get(&key)
            })
            .collect();

        if in_bufs.is_empty() {
            // 叶子：无输入 → 由结构 + 参数生成程序化图案。
            leaf_pattern(&node.structure, &node.params, iw, ih, spp, &mut out);
            return out;
        }

        // 有输入：先按组合方式折叠多输入，再让本节点的结构调制。
        match node.composition {
            prism_core::node::Composition::Merge(op) => {
                // 逐个折叠：源与目标不能别名，故每步写入新缓冲。
                let mut acc: Vec<f32> = in_bufs[0].clone();
                for other in &in_bufs[1..] {
                    let mut next = acc.clone();
                    combine(op, &acc, other, &mut next);
                    acc = next;
                }
                out.copy_from_slice(&acc);
            }
            _ => {
                // Chain / Feed / Dual / Csg / Mis：预览一律按"取第 0 输入"处理
                // （简化点，见汇报"诚实边界"：这些变体在预览中不产生额外视觉效果）。
                out.copy_from_slice(in_bufs[0]);
            }
        }

        // 内部节点仍按其结构做一次轻量调制，使不同结构产生可区分的观感。
        // （`modulate` 读 `out` 写 `modulated`——不可变借用与可变借用不能重叠，
        // 故先克隆一份作只读源。）
        let mut modulated = out.clone();
        modulate(&out, &node.structure, &node.params, iw, ih, &mut modulated);
        modulated
    }
}

// ---------------------------------------------------------------------------
// 根节点选取与祖先锥
// ---------------------------------------------------------------------------

/// 选根：无下游（不是任何节点的上游）且在拓扑序中的节点；多个取 id 最大者。
fn pick_root(graph: &Graph, order: &[NodeId]) -> Option<NodeId> {
    let mut has_downstream: BTreeSet<NodeId> = BTreeSet::new();
    for &id in order {
        for (_, from) in graph.inputs_of(id) {
            has_downstream.insert(from);
        }
    }
    order.iter().copied().filter(|id| !has_downstream.contains(id)).max()
}

/// 祖先锥：root 及其全部上游传递闭包。
fn ancestor_cone(graph: &Graph, order: &[NodeId], root: NodeId) -> BTreeSet<NodeId> {
    let mut cone: BTreeSet<NodeId> = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !cone.insert(id) {
            continue;
        }
        for (_, from) in graph.inputs_of(id) {
            if !cone.contains(&from) {
                stack.push(from);
            }
        }
    }
    // 防御：确保只保留图中存在的节点（order 已保证，但显式过滤更稳）。
    cone.retain(|id| order.contains(id));
    cone
}

// ---------------------------------------------------------------------------
// 图案生成（纯整数散列 + 多项式插值，无 libm 超越函数）
// ---------------------------------------------------------------------------

/// 上取整除（避免 `div_ceil` 在旧工具链上的可用性差异）。
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// 32 位整数散列（finalizer 变体，雪崩充分）。
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// 整数格点散列 → `[0, 1)`。
fn lattice(x: i32, y: i32, seed: u32) -> f32 {
    let a = (x as u32).wrapping_mul(0x9e37_79b1);
    let b = (y as u32).wrapping_mul(0x85eb_ca77);
    (hash_u32(a ^ b ^ seed) >> 8) as f32 / 16_777_216.0
}

/// 平滑步进（smoothstep）：三次多项式，无超越函数。
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 线性插值。
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 二维值噪声（双线性 + smoothstep）。
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let fx = smoothstep(x - xi);
    let fy = smoothstep(y - yi);
    let (xi, yi) = (xi as i32, yi as i32);
    let n00 = lattice(xi, yi, seed);
    let n10 = lattice(xi + 1, yi, seed);
    let n01 = lattice(xi, yi + 1, seed);
    let n11 = lattice(xi + 1, yi + 1, seed);
    lerp(lerp(n00, n10, fx), lerp(n01, n11, fx), fy)
}

/// 分形叠加噪声（`octaves` 层，每层频率 ×2、振幅 ×0.5）。
fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for o in 0..octaves.clamp(1, 8) {
        sum += value_noise(x * freq, y * freq, seed ^ o.wrapping_mul(0x9e37_79b1)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// 读取浮点参数（F32 直接取；U32 视作 f32；缺省 `default`）。
fn pget(params: &Params, name: &str, default: f32) -> f32 {
    match params.get(name) {
        Some(prism_core::node::ParamValue::F32(v)) => *v,
        Some(prism_core::node::ParamValue::U32(v)) => *v as f32,
        _ => default,
    }
}

/// 图案族：由结构序号映射到 8 个图案族之一。
///
/// 这是**风格化映射**，不是引擎的物理核算子——预览只需"不同结构看起来不同"
/// 且确定性可复现。已在汇报"诚实边界"中登记。
fn pattern_family(structure: &StructureId) -> u8 {
    match structure.ordinal() {
        Some(1) => 0,  // ScalarField → 场
        Some(2) => 1,  // Convolution → 平滑
        Some(3) => 2,  // ConstraintFlow → 条纹
        Some(4) => 3,  // DeterministicSampling → 抖散
        Some(5) => 4,  // Reparameterization → 旋涡
        Some(11) => 5, // Diffusion → 径向
        Some(12) => 6, // EntityRepresentation → 网格
        _ => 7,        // 其余官方段与用户段 → 默认场
    }
}

/// 由结构 + 参数生成整幅叶子图案。
#[allow(clippy::too_many_arguments)]
fn leaf_pattern(
    structure: &StructureId,
    params: &Params,
    iw: u32,
    ih: u32,
    spp: u32,
    out: &mut [f32],
) {
    let family = pattern_family(structure);
    let scale = pget(params, "scale", 0.5);
    let freq = 2.0 + (scale * 8.0).floor().clamp(0.0, 24.0);
    let octaves = pget(params, "octaves", 4.0).clamp(1.0, 8.0) as u32;
    let seed = family as u32 ^ 0x5eed;

    for py in 0..ih {
        for px in 0..iw {
            // spp > 1 时在像素内做确定性分层抖动采样并平均（抗锯齿）。
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            for s in 0..spp {
                let jx = jitter(px, py, s, 0);
                let jy = jitter(px, py, s, 1);
                let u = (px as f32 + jx) / iw as f32;
                let v = (py as f32 + jy) / ih as f32;
                let (cr, cg, cb) = pattern_color(family, u, v, freq, octaves, seed);
                r += cr;
                g += cg;
                b += cb;
            }
            let inv = 1.0 / spp as f32;
            let idx = ((py * iw + px) as usize) * 3;
            out[idx] = r * inv;
            out[idx + 1] = g * inv;
            out[idx + 2] = b * inv;
        }
    }
}

/// 像素内确定性抖动（分层网格，无随机数状态）。
fn jitter(px: u32, py: u32, s: u32, axis: u32) -> f32 {
    let grid = 4u32; // 4x4 分层
    let gx = s % grid;
    let gy = s / grid;
    let h = hash_u32((px << 8) ^ (py << 20) ^ (s << 4) ^ axis.wrapping_mul(0x9e37_79b9));
    let base_x = (gx as f32 + 0.5) / grid as f32;
    let base_y = (gy as f32 + 0.5) / grid as f32;
    let r = (h >> 8) as f32 / 16_777_216.0 * 0.5 / grid as f32;
    let base = if axis == 0 { base_x } else { base_y };
    (base + r).clamp(0.0, 0.999_999)
}

/// 单个采样点的图案颜色（按族分派）。
fn pattern_color(family: u8, u: f32, v: f32, freq: f32, octaves: u32, seed: u32) -> (f32, f32, f32) {
    let x = u * freq;
    let y = v * freq;
    match family {
        0 => {
            // 场：fbm 直接作亮度，冷暖双色渐变。
            let n = fbm(x, y, octaves, seed);
            (n * 0.9, n * 0.95, 1.0 - n * 0.7)
        }
        1 => {
            // 平滑：低频 fbm，强对比。
            let n = fbm(x * 0.4, y * 0.4, octaves.min(3), seed);
            let s = smoothstep(n);
            (s, s * 0.8, 1.0 - s * 0.5)
        }
        2 => {
            // 条纹：格点坐标的分段多项式波。
            let s = ((x.floor() * 0.5 + y.floor() * 0.5) % 2.0) as f32;
            let t = fbm(x * 0.25, y * 0.25, 2, seed);
            (0.3 + s * 0.5, 0.4 + t * 0.4, 0.6 + (1.0 - s) * 0.3)
        }
        3 => {
            // 抖散：高频值噪声，颗粒感。
            let n = value_noise(x * 4.0, y * 4.0, seed);
            (n, 1.0 - n, n * 0.5 + 0.25)
        }
        4 => {
            // 旋涡：到中心的距离 + 角度的多项式近似（用 atan-free 折减）。
            let dx = u - 0.5;
            let dy = v - 0.5;
            let r2 = dx * dx + dy * dy;
            let ang = (dx * 3.0 + dy * 5.0) - (dx * dx * 3.0 - dy * dy * 2.0);
            let t = (ang * freq * 0.3).rem_euclid(1.0);
            (0.2 + t * 0.7, 0.3 + (1.0 - t) * 0.5, 0.5 + r2 * 1.5)
        }
        5 => {
            // 径向：以中心为原点的高斯近似的多项式衰减。
            let dx = u - 0.5;
            let dy = v - 0.5;
            let d2 = dx * dx + dy * dy;
            let g = 1.0 / (1.0 + d2 * 12.0);
            (g, g * 0.85, g * 0.7)
        }
        6 => {
            // 网格：格线 + 单元噪声。
            let gx = x.floor();
            let gy = y.floor();
            let line = if (x - gx) < 0.08 || (y - gy) < 0.08 { 0.15 } else { 0.0 };
            let n = value_noise(gx * 0.7, gy * 0.7, seed);
            (n * 0.6 + line, n * 0.7 + line, n * 0.5 + 0.3 + line)
        }
        _ => {
            // 默认场：斜向渐变 + 轻噪声。
            let n = fbm(x * 0.6, y * 0.6, octaves.min(4), seed);
            let d = (u + v) * 0.5;
            (0.3 + d * 0.5, 0.4 + n * 0.4, 0.6 + (1.0 - d) * 0.3)
        }
    }
}

/// 内部节点按结构的轻量调制（在已有输入结果上叠加结构特征）。
fn modulate(
    buf: &[f32],
    structure: &StructureId,
    params: &Params,
    iw: u32,
    ih: u32,
    out: &mut [f32],
) {
    let family = pattern_family(structure);
    let strength = pget(params, "strength", 0.7).clamp(0.0, 1.0);
    let seed = family as u32 ^ 0xc0de;
    let n = (iw as usize) * (ih as usize);
    for py in 0..ih {
        for px in 0..iw {
            let u = px as f32 / iw as f32;
            let v = py as f32 / ih as f32;
            let m = fbm(u * 4.0, v * 4.0, 3, seed) - 0.5;
            let idx = (py * iw + px) as usize * 3;
            for c in 0..3 {
                let base = buf[idx + c];
                // 以结构决定的固定增益做对比度拉伸（不同结构 → 不同观感）。
                let gain = 0.85 + 0.3 * (family as f32 / 7.0);
                let tweaked = (base - 0.5) * gain + 0.5 + m * strength * 0.25;
                out[idx + c] = tweaked.clamp(0.0, 1.0);
            }
        }
    }
    debug_assert_eq!(buf.len(), n * 3);
}

/// 按 MergeOp 合并两个 RGB 平面（`out` 可与 `a` 别名，故逐元素先算后写）。
fn combine(op: MergeOp, a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let (x, y) = (a[i], b[i]);
        let v = match op {
            MergeOp::Over => y + x * (1.0 - y),
            MergeOp::Plus => x + y,
            MergeOp::Screen => 1.0 - (1.0 - x) * (1.0 - y),
            MergeOp::Multiply => x * y,
            // Mask：预览按 50/50 混合近似（无真实掩码信号，简化点见"诚实边界"）。
            MergeOp::Mask => (x + y) * 0.5,
        };
        out[i] = v.clamp(0.0, 1.0);
    }
}

/// 根缓冲 → RGBA8 输出（内部尺寸与输出尺寸不同时最近邻放大）。
///
/// 色调映射：`c / (1 + c)`（Reinhard）后 `sqrt` 近似 gamma 2.2——两者都是
/// 基本运算，无 libm 依赖，跨平台逐位一致。
fn to_rgba(buf: &[f32], iw: u32, ih: u32, width: u32, height: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    for y in 0..height {
        // 最近邻：源坐标 = floor(dst * src / dst_size)，块状放大（Draft 的可视特征）。
        let sy = if ih >= height { y } else { ((y as u64 * ih as u64) / height as u64) as u32 };
        for x in 0..width {
            let sx = if iw >= width { x } else { ((x as u64 * iw as u64) / width as u64) as u32 };
            let si = ((sy.min(ih - 1) * iw + sx.min(iw - 1)) as usize) * 3;
            let di = ((y * width + x) as usize) * 4;
            for c in 0..3 {
                let lin = buf.get(si + c).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let mapped = lin / (1.0 + lin);
                // sqrt 近似 gamma 编码（0..1 → 0..255）。
                rgba[di + c] = (mapped.sqrt() * 255.0 + 0.5) as u8;
            }
            rgba[di + 3] = 255; // 不透明
        }
    }
    rgba
}

// ---------------------------------------------------------------------------
// 单元测试（零 GPU 依赖，native 直接跑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::state::{build_reference_graph, EditorState};
    use prism_core::node::{Composition, Node, ParamValue};

    /// 无通道定义（参考图无绑定）。
    fn defs() -> ChannelDefs {
        ChannelDefs::new()
    }

    /// 单节点图（叶子）构造。
    fn single_leaf(structure: StructureId, v: f32) -> Graph {
        let mut g = Graph::new();
        let mut n = Node::new(1, structure, Composition::Feed);
        n.params.set("scale", ParamValue::F32(v));
        g.add_node(n).expect("单节点图构造应成功");
        g
    }

    #[test]
    fn render_is_byte_identical_across_calls() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let a = r.render(&g, &defs(), PreviewQuality::Final, 64, 48, 1, 0).unwrap();
        let b = r.render(&g, &defs(), PreviewQuality::Final, 64, 48, 1, 0).unwrap();
        assert_eq!(a.rgba, b.rgba, "同图同档位同帧 → 像素逐字节一致（确定性铁律）");
        assert_eq!(a.rgba.len(), 64 * 48 * 4);
    }

    #[test]
    fn second_render_hits_cache_fully() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let first = r.render(&g, &defs(), PreviewQuality::Final, 64, 48, 1, 0).unwrap();
        assert!(!first.stats.recomputed.is_empty());
        let second = r.render(&g, &defs(), PreviewQuality::Final, 64, 48, 1, 0).unwrap();
        assert!(
            second.stats.recomputed.is_empty(),
            "第二次同帧应全部命中缓存，零重算；实际重算 {:?}",
            second.stats.recomputed
        );
        assert_eq!(second.stats.reused.len(), 4, "参考图 4 节点全部复用");
    }

    #[test]
    fn draft_uses_lower_spp_and_smaller_internal_resolution() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let draft = r.render(&g, &defs(), PreviewQuality::Draft, 64, 64, 1, 0).unwrap();
        let final_ = r.render(&g, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        assert!(draft.stats.spp < final_.stats.spp, "Draft spp 必须低于 Final");
        assert!(draft.stats.internal_width < final_.stats.internal_width);
        assert_eq!(draft.stats.internal_width, 64 / DRAFT_DOWNSCALE);
        // 输出尺寸仍等于请求尺寸（只是内部低分辨率）。
        assert_eq!((draft.width, draft.height), (64, 64));
        assert_eq!(draft.rgba.len(), 64 * 64 * 4);
    }

    #[test]
    fn draft_and_final_differ_visually() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let draft = r.render(&g, &defs(), PreviewQuality::Draft, 64, 64, 1, 0).unwrap();
        let final_ = r.render(&g, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        assert_ne!(draft.rgba, final_.rgba, "不同档位应产生不同像素（采样数生效）");
    }

    #[test]
    fn leaf_edit_recomputes_only_ancestor_chain() {
        // 增量核心：改叶 1 → 只重算 {1,3,4}，兄弟叶 2 命中缓存。
        let mut state = EditorState::with_reference_graph();
        let mut r = PreviewRenderer::new();
        // 先渲染建立缓存。
        let g0 = state.model().to_graph().unwrap();
        let _ = r.render(&g0, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();

        let dirty = state.apply_edit(1, "scale", ParamValue::F32(0.9)).unwrap();
        assert_eq!(dirty.to_vec(), vec![1, 3, 4], "受影响子图 = 叶 1 到根的祖先链");

        let g1 = state.model().to_graph().unwrap();
        let second = r.render(&g1, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        assert_eq!(
            second.stats.recomputed,
            vec![1, 3, 4],
            "重算集合必须恰好等于受影响子图（兄弟叶 2 不重算）"
        );
        assert_eq!(second.stats.reused, vec![2], "兄弟叶 2 必须命中缓存复用");
    }

    #[test]
    fn undo_restores_cache_hits_fully() {
        // 撤销后签名逐位还原 → 全部命中缓存、零重算，且像素回到撤销前。
        let mut state = EditorState::with_reference_graph();
        let mut r = PreviewRenderer::new();
        let g0 = state.model().to_graph().unwrap();
        let before = r.render(&g0, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();

        state.apply_edit(1, "scale", ParamValue::F32(0.9)).unwrap();
        let g1 = state.model().to_graph().unwrap();
        let after = r.render(&g1, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        assert_ne!(before.rgba, after.rgba, "编辑后像素应变化");

        state.undo().unwrap();
        let g2 = state.model().to_graph().unwrap();
        let restored = r.render(&g2, &defs(), PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        assert_eq!(
            restored.rgba, before.rgba,
            "撤销后像素必须逐字节回到编辑前（确定性）"
        );
        assert!(
            restored.stats.recomputed.is_empty(),
            "撤销后签名还原 → 全部命中缓存，零重算；实际 {:?}",
            restored.stats.recomputed
        );
    }

    #[test]
    fn frame_dimension_invalidates_cache() {
        // D-28：帧号进缓存键，跨帧不得误复用。
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let f0 = r.render(&g, &defs(), PreviewQuality::Final, 32, 32, 1, 0).unwrap();
        assert!(!f0.stats.recomputed.is_empty());
        let f1 = r.render(&g, &defs(), PreviewQuality::Final, 32, 32, 1, 1).unwrap();
        assert_eq!(
            f1.stats.recomputed.len(),
            4,
            "帧号变化 → 缓存键变化 → 全部重算（D-28 帧维可区分）"
        );
    }

    #[test]
    fn different_structures_look_different() {
        // 不同结构 → 不同图案族 → 像素不同（编辑器需能"看得出选中了谁"）。
        let mut r = PreviewRenderer::new();
        let a = r
            .render(
                &single_leaf(StructureId::ScalarField, 0.5),
                &defs(),
                PreviewQuality::Final,
                32,
                32,
                1,
                0,
            )
            .unwrap();
        let b = r
            .render(
                &single_leaf(StructureId::EntityRepresentation, 0.5),
                &defs(),
                PreviewQuality::Final,
                32,
                32,
                1,
                0,
            )
            .unwrap();
        assert_ne!(a.rgba, b.rgba, "不同结构的预览应可区分");
    }

    #[test]
    fn bad_size_is_structured_error_not_panic() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        for (w, h) in [(0u32, 32u32), (32, 0), (MAX_DIM + 1, 32), (32, MAX_DIM + 1)] {
            let e = r.render(&g, &defs(), PreviewQuality::Draft, w, h, 1, 0).unwrap_err();
            match e {
                PreviewError::BadSize { width, height, .. } => {
                    assert_eq!((width, height), (w, h));
                }
                other => panic!("应为 BadSize，实际 {other:?}"),
            }
            assert!(e.to_string().contains("建议："), "必含建议段（C19）");
        }
    }

    #[test]
    fn cyclic_graph_is_structured_error_not_hang() {
        // 3 节点环 1→2→3→1（core 的 connect 只禁自环，环靠 topo_order 拦）。
        let mut g = Graph::new();
        for id in [1u64, 2, 3] {
            g.add_node(Node::new(id, StructureId::ScalarField, Composition::Chain)).unwrap();
        }
        g.connect(1, 2, 0).unwrap();
        g.connect(2, 3, 0).unwrap();
        g.connect(3, 1, 0).unwrap();
        let mut r = PreviewRenderer::new();
        let e = r.render(&g, &defs(), PreviewQuality::Draft, 32, 32, 1, 0).unwrap_err();
        assert!(matches!(e, PreviewError::Cyclic { .. }), "环图应结构化报错，实际 {e:?}");
        assert!(e.to_string().contains("建议："), "必含建议段（C19）");
    }

    #[test]
    fn empty_graph_reports_no_root() {
        let g = Graph::new();
        let mut r = PreviewRenderer::new();
        let e = r.render(&g, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap_err();
        assert!(matches!(e, PreviewError::NoRoot { .. }), "空图应报 NoRoot，实际 {e:?}");
    }

    #[test]
    fn cache_is_bounded() {
        let mut r = PreviewRenderer::new();
        assert_eq!(r.cache_len(), 0);
        let g = build_reference_graph();
        // 跑超过 CACHE_CAP 的不同帧，逼出淘汰。
        for frame in 0..(CACHE_CAP as u32 + 8) {
            r.render(&g, &defs(), PreviewQuality::Final, 32, 32, 1, u64::from(frame)).unwrap();
        }
        assert!(r.cache_len() <= CACHE_CAP, "缓存必须有界，实际 {}", r.cache_len());
    }

    /// 构造两叶 + Merge 的图（`op` 指定归约算子）。
    fn merge_graph(op: MergeOp) -> Graph {
        let mut g = Graph::new();
        let mut a = Node::new(1, StructureId::ScalarField, Composition::Feed);
        a.params.set("scale", ParamValue::F32(0.7));
        let mut b = Node::new(2, StructureId::Diffusion, Composition::Feed);
        b.params.set("scale", ParamValue::F32(0.3));
        let mut m = Node::new(3, StructureId::Diffusion, Composition::Merge(op));
        m.params.set("strength", ParamValue::F32(0.5));
        g.add_node(a).unwrap();
        g.add_node(b).unwrap();
        g.add_node(m).unwrap();
        g.connect(1, 3, 0).unwrap();
        g.connect(2, 3, 1).unwrap();
        g
    }

    #[test]
    fn merge_operations_change_output() {
        // Merge 算子决定归约语义 → 预览必须体现（Over / Multiply / Plus 互不相同）。
        let mut r = PreviewRenderer::new();
        let over = r
            .render(&merge_graph(MergeOp::Over), &defs(), PreviewQuality::Final, 32, 32, 1, 0)
            .unwrap()
            .rgba;
        let mul = r
            .render(&merge_graph(MergeOp::Multiply), &defs(), PreviewQuality::Final, 32, 32, 1, 0)
            .unwrap()
            .rgba;
        let plus = r
            .render(&merge_graph(MergeOp::Plus), &defs(), PreviewQuality::Final, 32, 32, 1, 0)
            .unwrap()
            .rgba;
        assert_ne!(over, mul, "Over vs Multiply → 像素不同");
        assert_ne!(mul, plus, "Multiply vs Plus → 像素不同");
        assert_ne!(over, plus, "Over vs Plus → 像素不同");
    }
}
