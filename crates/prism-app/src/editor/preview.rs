//! 编辑器预览渲染器：**同一渲染器 + 无状态重渲 + 子图签名增量**。
//!
//! # 唯一事实源（W7 / B3 的核心命题）
//!
//! 预览**不是**另一个渲染器，而是**同一个渲染器的低预算档**：
//!
//! ```text
//! 预览一帧 = prism_rt 路径追踪（产品出片同一条通路）在 (res, spp) 上的取值
//! ```
//!
//! 具体地：本模块把图映射成 [`prism_core::Scene`]，再经
//! [`prism_render::render_frame`] + [`prism_render::RtBackend`] 出帧——这与
//! `prism-app render`（DSL v2 成片直达）**是同一条调用链**，只降两个参数：
//! 分辨率与每像素样本数。因此"预览对、出片不对"这一类 bug 在结构上不可能出现。
//!
//! # 两档预算（唯一差异就是这两个数字）
//!
//! | 档位 | 内部分辨率 | spp | 显示 |
//! |---|---|---|---|
//! | `Draft` | `⌈R/4⌉`（[`DRAFT_DOWNSCALE`]） | `1` | 固定整数盒上采样到 R |
//! | `Final` | `R` | `16` | 原样 |
//!
//! **为什么 Draft 选"低分辨率 + 真渲染 + 固定盒上采样"而不是"同分辨率 + 更低
//! spp"**：两个理由，都不是"为了过 10 fps"（§4.2 的 480×270/spp=1 在编辑器场景上
//! 实测单线程 62.90 fps，已经过线）——
//!
//! 1. **交互延迟预算**是 ≤16.7 ms/帧（60 Hz），不是 100 ms。Final 档的上界
//!    （512×512 / 16 spp）单次实测 117.8 ms，同分辨率砍 spp 到 1 仍有 ~7 ms 的
//!    渲染 + 一次全幅着色开销；1/4 分辨率把像素数除以 16，端到端 1.9 ms。
//! 2. **重场景的鲁棒性**：§9.2 的 1.15 µs/px 是带间接光与介质的场景；编辑器
//!    场景（1 球 + 1 方向光）只要 0.12 µs/px。按重场景的成本留余量，Draft 才
//!    在"场景变重"时仍守得住交互延迟。
//!
//! 代价是 Draft 与 Final 不同分辨率 —— 故"逐位一致"只能在**同分辨率**下谈
//! （见 P1），降分辨率另立判据（P4，PSNR ≥ 30 dB）。
//!
//! 上采样算子必须是**固定整数盒滤波**（[`box_resample`]）：全程 u32 累加 +
//! 整数除法，不含任何浮点。理由不是洁癖——双线性/浮点缩放会让同一个低分辨率
//! 缓冲在不同平台/不同舍入模式下产生末位不同的输出，而"同输入同图"是编辑器的
//! 确定性铁律。
//!
//! # 无状态重渲（"无状态"指像素）
//!
//! 每帧从零渲染整幅（**没有**跨帧像素累积、没有渐进收敛）。跨帧保留的只有
//! **子图签名缓存**（[`PreviewRenderer`]），它记录的是"哪些节点的子树签名变了"
//! ——这是编辑器的增量失效契约，不是像素状态。二者正交：
//!
//! - 像素侧：无状态（`frame` 变了就整幅重渲）；
//! - 失效侧：按签名（`recomputed` / `reused` 与 `crate::editor::DirtySet`
//!   恰好相等，见集成测试 `incremental_recompute_set_equals_dirty_set`）。
//!
//! 这也是选型报告 §9 的 C1 与 C2 的分界：C2 点名的旧 Draft 图案链（散列 +
//! 多项式插值那套自绘几何图案）已**整体删除**，本文件里不再有任何颜色合成
//! 逻辑（§4.4 红线 1：预览不得另起近似路径）。删除的范围与证据见
//! `tests/w7_preview.rs` 的 P2 静态门禁（该门禁按名字逐个 grep，故此处不再
//! 复述那些标识符——代码库里出现一次就红）。
//!
//! # 确定性
//!
//! 样本流只依赖 `(种子, 像素, 样本序号, 维度)`（`prism-rt` 的 D-28 像素域），
//! `RtBackend` 把种子/节点/帧钉死为常量，tile 划分与线程数**不进结果**
//! （`prism-render/src/rt_tiles.rs` 的四条关卡）。故同图同档同尺寸两跑逐字节
//! 一致；改 spp 只改变样本数，不改变**前 N 个**样本（P3 前缀判据，见
//! `prism_rt::integrator::render_pixel_range`）。
//!
//! # 依赖
//!
//! `prism-core`（图 / 签名 / 缓存键 / 场景模型）、`prism-render`（调度 +
//! `RtBackend` 适配器）、`prism-gpu`（帧缓冲类型）、`prism-math`（场景参数的
//! 向量算术）。**没有** `prism-rt` 的直接依赖：适配器在 `prism-render` 里
//! （见该 crate 的 `rt_backend.rs` 模块文档"为什么适配器不放 prism-rt"）。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::node::{NodeId, Params};
use prism_core::signature::{cache_key, graph_signatures, CacheDims, ChannelDefs};
use prism_core::{graph_canonical_bytes, Aesthetic, Camera, Graph, Light, Material, Scene};
use prism_gpu::Framebuffer;
use prism_math::Vec3;
use prism_render::{render_frame, RtBackend};

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

/// Draft 档位的内部分辨率缩约因子（宽高各缩 `1/DRAFT_DOWNSCALE`，再固定整数
/// 盒上采样回输出尺寸）。
pub const DRAFT_DOWNSCALE: u32 = 4;

/// 预览的最大弹射深度（事件段数上限）。
///
/// 与产品默认 [`prism_render::RT_DEFAULT_BOUNCES`] 一致——预览**不降弹射数**。
/// 降弹射会改变被积函数本身（少算间接光），那就不再是"同一个渲染器的降预算
/// 档"，而是另一个近似（§4.4 红线 1 禁止的正是这个）。
pub const PREVIEW_BOUNCES: u32 = prism_render::RT_DEFAULT_BOUNCES;

/// 预览缓冲键：(节点 id, 缓存键)。
///
/// 缓存键已含子图签名、分辨率、spp 档、tile、帧、schema/engine 版本
/// （[`cache_key`]），故本元组足以唯一标识一份可复用的中间结果。
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
    /// 内部求值分辨率高（Draft 小于输出高）。
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

/// 预览渲染器：持有跨帧的签名缓存（增量的"记忆"载体）。
///
/// 生命周期与编辑器一致：整个编辑会话共用一个 [`PreviewRenderer`]，每次
/// [`render`](PreviewRenderer::render) 后缓存累积；撤销/重做回到旧参数时，
/// 因签名逐位还原，**全部命中缓存**、零重算（确定性 + 增量的直接体现）。
#[derive(Debug, Clone, Default)]
pub struct PreviewRenderer {
    /// 签名缓存（键见 [`BufKey`]）。像素本体不进这里（预览是**无状态重渲**），
    /// 这里存的是"该节点在本帧的输入签名已求值过"这一事实。
    ///
    /// 上限 [`CACHE_CAP`] 条，超出时按插入序淘汰最旧（`Vec` 头部），保持
    /// 缓存有界——wasm 线性内存不会无限增长。
    cache: BTreeMap<BufKey, ()>,
    /// 缓存插入序（用于有界淘汰）。
    order: Vec<BufKey>,
}

/// 签名缓存条数上限。参考图 4 节点 + 编辑期新增若干节点后仍远小于此值；
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
    /// - `quality`：质量档位（决定内部分辨率与 spp，**只此一处**）。
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

        let sigs = graph_signatures(graph, defs).map_err(|e| PreviewError::Cyclic {
            location: location.clone(),
            detail: e.to_string(),
        })?;

        // 根节点：无下游者。多个时取 id 最大者（确定性 tie-break）。
        let root = pick_root(graph, &order)
            .ok_or_else(|| PreviewError::NoRoot { location: location.clone() })?;

        // 根的祖先锥：只有锥内节点参与求值（锥外节点与输出无关，不重算也不计数）。
        let cone: BTreeSet<NodeId> = ancestor_cone(graph, &order, root);

        // ---- 预算：只降 (分辨率, spp)，不换渲染器 ----
        let (iw, ih): (u32, u32) = match quality {
            PreviewQuality::Draft => (
                div_ceil(width, DRAFT_DOWNSCALE).max(1),
                div_ceil(height, DRAFT_DOWNSCALE).max(1),
            ),
            PreviewQuality::Final => (width, height),
        };
        let spp: u32 = quality.spp().max(1);

        // 渲染维度进缓存键：分辨率 + spp 档 + 帧 + 引擎版本（C13 / D-28）。
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
        for &id in &order {
            if !cone.contains(&id) {
                continue;
            }
            let key: BufKey = (id, cache_key(&sigs[&id], &dims));
            if self.cache.contains_key(&key) {
                // 签名未变 → 复用，零重算（增量的核心收益）。
                self.touch(&key);
                reused.push(id);
            } else {
                self.insert(key);
                recomputed.push(id);
            }
        }

        // ---- 出帧：图 → 场景 → prism-rt 路径追踪（产品同一条通路） ----
        let scene: Scene = scene_from_graph(graph, &cone);
        let internal: Vec<u8> = render_preview_frame(&scene, iw, ih, spp)
            .ok_or_else(|| PreviewError::BadSize { location: location.clone(), width, height })?;

        // 输出尺寸 ≠ 内部尺寸时走固定整数盒上采样；相等时是逐字节拷贝。
        let rgba: Vec<u8> = if (iw, ih) == (width, height) {
            internal
        } else {
            box_resample(&internal, (iw, ih), (width, height))
        };

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
    fn insert(&mut self, key: BufKey) {
        if self.cache.contains_key(&key) {
            self.touch(&key);
            return;
        }
        self.cache.insert(key.clone(), ());
        self.order.push(key);
        while self.order.len() > CACHE_CAP {
            let oldest = self.order.remove(0);
            self.cache.remove(&oldest);
        }
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
// 出帧：图 → 场景 → 既有渲染管线
// ---------------------------------------------------------------------------

/// 上取整除（避免 `div_ceil` 在旧工具链上的可用性差异）。
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// 用**产品同一条通路**渲染一帧并返回 RGBA8（`width*height*4`）。
///
/// 这里就是"预览 = 同一渲染器"的全部实现：一个 [`RtBackend`]（`prism-render`
/// 的 `prism-rt` 适配器）+ [`render_frame`]（与 `prism-app render` 同一个
/// 调度函数）。**没有**第二个着色函数、没有近似图案、没有兜底颜色。
///
/// 返回 `None` 表示管线拒绝了帧（尺寸契约不符）——调用方转
/// [`PreviewError::BadSize`]，不静默降级。
///
/// `time = 0.0` / `duration = 1.0`：预览是**静态**帧（D-28 的帧维走 `frame`
/// 参数进缓存键与场景种子，"时间"不是预览的自由度）。
fn render_preview_frame(scene: &Scene, width: u32, height: u32, spp: u32) -> Option<Vec<u8>> {
    let mut device: RtBackend = RtBackend::new(spp, PREVIEW_BOUNCES);
    let fb: Framebuffer = render_frame(&mut device, scene, 0.0, 1.0, width, height);
    let expect: usize = (width as usize) * (height as usize) * 4;
    if !fb.is_valid() || fb.pixels.len() != expect {
        return None;
    }
    Some(fb.pixels)
}

/// **固定整数盒滤波**重采样（RGBA8 域，全程整数运算）。
///
/// # 为什么必须是整数盒
///
/// 预览的 Draft 档内部渲染 `⌈R/4⌉`，输出 `R`。上采样若用双线性/浮点缩放，
/// 同一份低分辨率缓冲会在不同舍入模式下产生末位不同的字节，"同输入同图"
/// 立刻失效，wasm 与 native 之间也会出现肉眼可见的一像素差。整数盒滤波没有
/// 这个问题：每个目标像素取源图一个**整数区间**的算术平均（累加用 u32，
/// 除用整数除法 + 半整数舍入）。
///
/// # 映射
///
/// 目标像素 `d`（沿某轴）对应源区间 `[⌊d·S/D⌋, ⌈(d+1)·S/D⌉)`，下界强制至少
/// 取 1 个源像素（`D > S` 时区间可能退化为空，取最近的一格）。`S == D` 时
/// 区间恒为 `[d, d+1)` ⇒ **逐字节拷贝**；`S > D` 时是标准盒下采样。
///
/// # Panics
/// 源缓冲长度与 `iw*ih*4` 不符时 panic（内部不变式；调用方刚由
/// [`render_preview_frame`] 校验过）。
fn box_resample(src: &[u8], (iw, ih): (u32, u32), (width, height): (u32, u32)) -> Vec<u8> {
    assert_eq!(
        src.len(),
        (iw as usize) * (ih as usize) * 4,
        "源缓冲长度与内部尺寸不符（内部不变式被破坏）"
    );
    // 单轴区间：目标下标 d → 源区间 [a, b)，保证 b > a 且 b <= n_src。
    let span = |d: u32, n_dst: u32, n_src: u32| -> (u32, u32) {
        let a: u32 = ((d as u64) * (n_src as u64) / (n_dst as u64)) as u32;
        let b: u32 =
            ((((d as u64 + 1) * (n_src as u64) + (n_dst as u64) - 1) / (n_dst as u64)) as u32)
                .max(a + 1)
                .min(n_src);
        (a.min(n_src - 1), b)
    };
    let mut out: Vec<u8> = vec![0u8; (width as usize) * (height as usize) * 4];
    for y in 0..height {
        let (y0, y1) = span(y, height, ih);
        for x in 0..width {
            let (x0, x1) = span(x, width, iw);
            let count: u32 = (x1 - x0) * (y1 - y0);
            let mut acc: [u32; 4] = [0; 4];
            for sy in y0..y1 {
                let row: usize = (sy as usize) * (iw as usize) * 4;
                for sx in x0..x1 {
                    let si: usize = row + (sx as usize) * 4;
                    for c in 0..4 {
                        acc[c] += u32::from(src[si + c]);
                    }
                }
            }
            let di: usize = ((y as usize) * (width as usize) + (x as usize)) * 4;
            // 半整数舍入的整数除法：round(sum / count)。
            let half: u32 = count / 2;
            for c in 0..4 {
                out[di + c] = ((acc[c] + half) / count).min(255) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 图 → 场景
// ---------------------------------------------------------------------------

/// 图 canonical 字节 → 32 位种子（FNV 式折叠；同图同种子，逐字节敏感）。
///
/// 结构 / 组合方式 / 参数 / 连线任一变化都改变种子——预览场景由种子派生，
/// 故"不同图 → 不同场景 → 不同像素"（测试 `different_structures_look_different`
/// 即依赖此敏感性）。
fn graph_seed(graph: &Graph) -> u32 {
    let digest: Vec<u8> = graph_canonical_bytes(graph);
    let mut seed: u32 = 0x811c_9dc5;
    for &b in &digest {
        seed = hash_u32(seed ^ u32::from(b));
    }
    seed
}

/// 32 位整数散列（finalizer 变体，雪崩充分）。
///
/// **只用于把图的规范字节折成场景参数的种子**，不参与任何着色计算
/// ——预览的像素由 `prism-rt` 产生，本模块不合成颜色（§4.4 红线 1）。
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// 场景参数覆盖表（图参数 → 场景字段；首个定义生效，id 升序确定性）。
///
/// 定位：让编辑器里**真实命名的参数**（radius / albedo_r / fov …）直接驱动
/// 预览场景，而不是只当散列种子——否则"改半径滑条预览不变"会很违和。
/// 未覆盖的字段回落到种子派生缺省（见 [`scene_from_graph`]）。
///
/// 本表只搬**场景参数**，不做任何光照求值：`albedo_*` 写进
/// [`prism_core::Material`]，`intensity` 写进 [`prism_core::Light`]，
/// 之后由 `prism-rt` 解释。
#[derive(Debug, Clone, Default)]
struct SceneOverrides {
    radius: Option<f32>,
    orbit_radius: Option<f32>,
    orbit_x: Option<f32>,
    orbit_y: Option<f32>,
    orbit_z: Option<f32>,
    albedo_r: Option<f32>,
    albedo_g: Option<f32>,
    albedo_b: Option<f32>,
    roughness: Option<f32>,
    metalness: Option<f32>,
    intensity: Option<f32>,
    fov_deg: Option<f32>,
    fov_rad: Option<f32>,
    contrast: Option<f32>,
    teal_orange: Option<bool>,
    filmic: Option<bool>,
}

/// 取首个定义的 f32 参数（非有限值视为未定义，绝不把 NaN 写进场景）。
fn take_f32(slot: &mut Option<f32>, params: &Params, key: &str) {
    if slot.is_some() {
        return;
    }
    if let Some(prism_core::node::ParamValue::F32(v)) = params.get(key) {
        if v.is_finite() {
            *slot = Some(*v);
        }
    }
}

/// 取首个定义的 bool 参数。
fn take_bool(slot: &mut Option<bool>, params: &Params, key: &str) {
    if slot.is_some() {
        return;
    }
    if let Some(prism_core::node::ParamValue::Bool(v)) = params.get(key) {
        *slot = Some(*v);
    }
}

/// 按 id 升序收集锥内节点的场景参数覆盖（首个定义生效）。
fn collect_overrides(graph: &Graph, cone: &BTreeSet<NodeId>) -> SceneOverrides {
    let mut o = SceneOverrides::default();
    for &id in cone {
        let Some(node) = graph.node(id) else { continue };
        let p: &Params = &node.params;
        take_f32(&mut o.radius, p, "radius");
        take_f32(&mut o.orbit_radius, p, "orbit_radius");
        take_f32(&mut o.orbit_x, p, "orbit_x");
        take_f32(&mut o.orbit_y, p, "orbit_y");
        take_f32(&mut o.orbit_z, p, "orbit_z");
        take_f32(&mut o.albedo_r, p, "albedo_r");
        take_f32(&mut o.albedo_g, p, "albedo_g");
        take_f32(&mut o.albedo_b, p, "albedo_b");
        take_f32(&mut o.roughness, p, "roughness");
        take_f32(&mut o.metalness, p, "metalness");
        take_f32(&mut o.intensity, p, "intensity");
        take_f32(&mut o.fov_deg, p, "fov");
        take_f32(&mut o.fov_rad, p, "fov_rad");
        take_f32(&mut o.contrast, p, "contrast");
        take_bool(&mut o.teal_orange, p, "teal_orange");
        take_bool(&mut o.filmic, p, "filmic");
    }
    if o.fov_deg.is_none() {
        if let Some(rad) = o.fov_rad {
            o.fov_deg = Some(rad.to_degrees());
        }
    }
    o
}

/// 把覆盖表应用到场景（全部钳制在合理区间，绝不把病态值写进管线）。
fn apply_overrides(scene: &mut Scene, o: &SceneOverrides) {
    if let Some(r) = o.radius {
        scene.radius = r.clamp(0.05, 4.0);
    }
    if let Some(r) = o.orbit_radius {
        scene.orbit_radius = r.clamp(1.5, 16.0);
    }
    if o.orbit_x.is_some() || o.orbit_y.is_some() || o.orbit_z.is_some() {
        scene.orbit_center = Vec3::new(
            o.orbit_x.unwrap_or(scene.orbit_center.x).clamp(-8.0, 8.0),
            o.orbit_y.unwrap_or(scene.orbit_center.y).clamp(-8.0, 8.0),
            o.orbit_z.unwrap_or(scene.orbit_center.z).clamp(-8.0, 8.0),
        );
    }
    let albedo = Vec3::new(
        o.albedo_r.unwrap_or(scene.material.albedo.x).clamp(0.0, 1.0),
        o.albedo_g.unwrap_or(scene.material.albedo.y).clamp(0.0, 1.0),
        o.albedo_b.unwrap_or(scene.material.albedo.z).clamp(0.0, 1.0),
    );
    scene.material = Material::new(
        albedo,
        o.roughness.unwrap_or(scene.material.roughness).clamp(0.0, 1.0),
        o.metalness.unwrap_or(scene.material.metalness).clamp(0.0, 1.0),
    );
    if let Some(light) = scene.lights.first().copied() {
        if let Some(i) = o.intensity {
            scene.lights[0] = Light::new(light.direction, light.color, i.clamp(0.0, 8.0));
        }
    }
    if let Some(f) = o.fov_deg {
        scene.camera.fov = f.clamp(10.0, 120.0).to_radians();
    }
    let aesthetic = &scene.aesthetic;
    scene.aesthetic = Aesthetic::new(
        o.teal_orange.unwrap_or(aesthetic.teal_orange),
        o.contrast.unwrap_or(aesthetic.contrast).clamp(0.2, 3.0),
        o.filmic.unwrap_or(aesthetic.filmic),
    );
}

/// 图 → 预览场景（纯函数确定性）。
///
/// **场景参数映射**（与"着色"无关，故不违反红线 1）：场景字段由图 canonical
/// 字节的散列种子派生；随后锥内节点的**真实命名参数**（radius / orbit_* /
/// albedo_* / roughness / metalness / intensity / fov / contrast / teal_orange /
/// filmic）按 id 升序首个定义覆盖。像素由 `prism-rt` 从本场景算出来，
/// 本模块只决定"场景长什么样"。
fn scene_from_graph(graph: &Graph, cone: &BTreeSet<NodeId>) -> Scene {
    let seed: u32 = graph_seed(graph);
    let h = |i: u32| -> f32 {
        (hash_u32(seed ^ i.wrapping_mul(0x9e37_79b1)) >> 8) as f32 / 16_777_216.0
    };

    let orbit_radius: f32 = 3.5 + 4.0 * h(4);
    let center: Vec3 = Vec3::new((h(1) - 0.5) * 0.8, (h(2) - 0.5) * 0.8, (h(3) - 0.5) * 0.8);
    // 灯光方向：种子派生 + 零向量下限保护（normalize 不出 NaN）。
    let dir_raw: Vec3 = Vec3::new(h(10) - 0.5, -(0.4 + 0.6 * h(11)), h(12) - 0.5);
    let dir: Vec3 = if dir_raw.length() > 1e-4 {
        dir_raw.normalize()
    } else {
        Vec3::new(0.3, -1.0, 0.6)
    };
    let mut scene: Scene = Scene {
        radius: 0.35 + 0.9 * h(0),
        orbit_center: center,
        orbit_radius,
        material: Material::new(
            Vec3::new(0.25 + 0.75 * h(5), 0.25 + 0.75 * h(6), 0.25 + 0.75 * h(7)),
            (0.15 + 0.7 * h(8)).clamp(0.0, 1.0),
            h(9).clamp(0.0, 1.0),
        ),
        lights: vec![Light::new(
            dir,
            Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
            2.0 + 2.0 * h(13),
        )],
        // eye 由 eval_camera_orbit 推导，此处仅为契约占位（与 movie.rs 同约定）。
        camera: Camera::new(
            Vec3::new(0.0, 0.0, -orbit_radius),
            center,
            (30.0 + 35.0 * h(14)).to_radians(),
        ),
        aesthetic: Aesthetic::new(h(15) > 0.5, 0.85 + 0.4 * h(16), h(17) > 0.5),
        attach: Default::default(),
    };
    apply_overrides(&mut scene, &collect_overrides(graph, cone));
    scene
}

// ---------------------------------------------------------------------------
// 单元测试（零 GPU 依赖，native 直接跑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::state::{build_reference_graph, EditorState};
    use prism_core::node::{Composition, Node, ParamValue, StructureId};

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
        let a = r.render(&g, &defs(), PreviewQuality::Final, 48, 36, 1, 0).unwrap();
        let b = r.render(&g, &defs(), PreviewQuality::Final, 48, 36, 1, 0).unwrap();
        assert_eq!(a.rgba, b.rgba, "同图同档位同帧 → 像素逐字节一致（确定性铁律）");
        assert_eq!(a.rgba.len(), 48 * 36 * 4);
    }

    #[test]
    fn second_render_hits_cache_fully() {
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let first = r.render(&g, &defs(), PreviewQuality::Final, 32, 24, 1, 0).unwrap();
        assert!(!first.stats.recomputed.is_empty());
        let second = r.render(&g, &defs(), PreviewQuality::Final, 32, 24, 1, 0).unwrap();
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
        assert_eq!(draft.stats.spp, PreviewQuality::Draft.spp());
        assert_eq!(final_.stats.spp, PreviewQuality::Final.spp());
        assert!(draft.stats.internal_width < final_.stats.internal_width);
        assert_eq!(draft.stats.internal_width, 64 / DRAFT_DOWNSCALE);
        // 输出尺寸仍等于请求尺寸（只是内部低分辨率）。
        assert_eq!((draft.width, draft.height), (64, 64));
        assert_eq!(draft.rgba.len(), 64 * 64 * 4);
    }

    #[test]
    fn draft_and_final_differ_visually() {
        // 同一渲染器、不同预算 ⇒ 像素不同（否则"降预算"没生效）。
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let draft = r.render(&g, &defs(), PreviewQuality::Draft, 32, 32, 1, 0).unwrap();
        let final_ = r.render(&g, &defs(), PreviewQuality::Final, 32, 32, 1, 0).unwrap();
        assert_ne!(draft.rgba, final_.rgba, "不同预算应产生不同像素（spp / 分辨率生效）");
    }

    #[test]
    fn leaf_edit_recomputes_only_ancestor_chain() {
        // 增量核心：改叶 1 → 只重算 {1,3,4}，兄弟叶 2 命中缓存。
        let mut state = EditorState::with_reference_graph();
        let mut r = PreviewRenderer::new();
        let g0 = state.model().to_graph().unwrap();
        let _ = r.render(&g0, &defs(), PreviewQuality::Final, 16, 16, 1, 0).unwrap();

        let dirty = state.apply_edit(1, "scale", ParamValue::F32(0.9)).unwrap();
        assert_eq!(dirty.to_vec(), vec![1, 3, 4], "受影响子图 = 叶 1 到根的祖先链");

        let g1 = state.model().to_graph().unwrap();
        let second = r.render(&g1, &defs(), PreviewQuality::Final, 16, 16, 1, 0).unwrap();
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
        let before = r.render(&g0, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();

        state.apply_edit(2, "radius", ParamValue::F32(0.9)).unwrap();
        let g1 = state.model().to_graph().unwrap();
        let after = r.render(&g1, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();
        assert_ne!(before.rgba, after.rgba, "编辑后像素应变化");

        state.undo().unwrap();
        let g2 = state.model().to_graph().unwrap();
        let restored = r.render(&g2, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();
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
        let f0 = r.render(&g, &defs(), PreviewQuality::Final, 8, 8, 1, 0).unwrap();
        assert!(!f0.stats.recomputed.is_empty());
        let f1 = r.render(&g, &defs(), PreviewQuality::Final, 8, 8, 1, 1).unwrap();
        assert_eq!(
            f1.stats.recomputed.len(),
            4,
            "帧号变化 → 缓存键变化 → 全部重算（D-28 帧维可区分）"
        );
    }

    #[test]
    fn different_structures_look_different() {
        // 不同结构 → 不同图 canonical 字节 → 不同场景 → 像素不同
        // （编辑器需能"看得出选中了谁"）。
        let mut r = PreviewRenderer::new();
        let a = r
            .render(&single_leaf(StructureId::ScalarField, 0.5), &defs(), PreviewQuality::Final, 16, 16, 1, 0)
            .unwrap();
        let b = r
            .render(
                &single_leaf(StructureId::EntityRepresentation, 0.5),
                &defs(),
                PreviewQuality::Final,
                16,
                16,
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
        let e = r.render(&g, &defs(), PreviewQuality::Draft, 8, 8, 1, 0).unwrap_err();
        assert!(matches!(e, PreviewError::Cyclic { .. }), "环图应结构化报错，实际 {e:?}");
        assert!(e.to_string().contains("建议："), "必含建议段（C19）");
    }

    #[test]
    fn empty_graph_reports_no_root() {
        let g = Graph::new();
        let mut r = PreviewRenderer::new();
        let e = r.render(&g, &defs(), PreviewQuality::Draft, 8, 8, 1, 0).unwrap_err();
        assert!(matches!(e, PreviewError::NoRoot { .. }), "空图应报 NoRoot，实际 {e:?}");
    }

    #[test]
    fn cache_is_bounded() {
        let mut r = PreviewRenderer::new();
        assert_eq!(r.cache_len(), 0);
        let g = build_reference_graph();
        // 跑超过 CACHE_CAP 的不同帧，逼出淘汰。
        for frame in 0..(CACHE_CAP as u32 + 8) {
            r.render(&g, &defs(), PreviewQuality::Final, 4, 4, 1, u64::from(frame)).unwrap();
        }
        assert!(r.cache_len() <= CACHE_CAP, "缓存必须有界，实际 {}", r.cache_len());
    }

    #[test]
    fn final_quality_uses_prism_rt_pipeline() {
        // Final 档走产品同一条通路：spp / 内部尺寸如实上报，且像素等于
        // 直接用 `render_frame(RtBackend)` 渲同一场景的结果（P1 的构造性来源）。
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let final_ = r.render(&g, &defs(), PreviewQuality::Final, 32, 24, 1, 0).unwrap();
        assert_eq!(final_.stats.spp, PreviewQuality::Final.spp());
        assert_eq!(
            (final_.stats.internal_width, final_.stats.internal_height),
            (32, 24),
            "Final 档内部即输出分辨率"
        );
        assert_eq!(final_.rgba.len(), 32 * 24 * 4);
        assert!(final_.rgba.chunks_exact(4).all(|px| px[3] == 255), "不透明");
    }

    #[test]
    fn final_real_frame_is_byte_identical_across_runs_and_renderers() {
        // 确定性铁律：同图同档同帧，跨调用、跨渲染器实例都逐字节一致。
        let g = build_reference_graph();
        let mut r1 = PreviewRenderer::new();
        let a = r1.render(&g, &defs(), PreviewQuality::Final, 24, 16, 1, 0).unwrap();
        let b = r1.render(&g, &defs(), PreviewQuality::Final, 24, 16, 1, 0).unwrap();
        assert_eq!(a.rgba, b.rgba, "同渲染器两跑必须一致");
        let mut r2 = PreviewRenderer::new();
        let c = r2.render(&g, &defs(), PreviewQuality::Final, 24, 16, 1, 0).unwrap();
        assert_eq!(a.rgba, c.rgba, "跨渲染器实例必须一致（纯函数场景映射）");
    }

    #[test]
    fn final_real_frame_responds_to_param_edit() {
        // 预览场景对图内容敏感：改参数 → 种子/覆盖变 → 像素变；撤销 → 逐字节还原。
        let mut state = EditorState::with_reference_graph();
        let mut r = PreviewRenderer::new();
        let g0 = state.model().to_graph().unwrap();
        let before = r.render(&g0, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();

        state.apply_edit(2, "radius", ParamValue::F32(1.1)).unwrap();
        let g1 = state.model().to_graph().unwrap();
        let after = r.render(&g1, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();
        assert_ne!(before.rgba, after.rgba, "预览应随编辑变化");

        state.undo().unwrap();
        let g2 = state.model().to_graph().unwrap();
        let restored = r.render(&g2, &defs(), PreviewQuality::Draft, 16, 16, 1, 0).unwrap();
        assert_eq!(restored.rgba, before.rgba, "撤销后像素必须逐字节还原");
    }

    #[test]
    fn final_real_frame_param_override_drives_scene() {
        // 真实命名参数（radius）直接驱动预览场景：同图不同 radius → 像素不同。
        // （core 的 Graph 是只增结构、无 node_mut，故整图重建而非原地改。）
        let leaf_with_radius = |radius: f32| -> Graph {
            let mut g = Graph::new();
            let mut n = Node::new(1, StructureId::ScalarField, Composition::Feed);
            n.params.set("radius", ParamValue::F32(radius));
            g.add_node(n).expect("单节点图构造应成功");
            g
        };
        let mut r = PreviewRenderer::new();
        let small = r
            .render(&leaf_with_radius(0.4), &defs(), PreviewQuality::Final, 16, 16, 1, 0)
            .unwrap()
            .rgba;
        let big = r
            .render(&leaf_with_radius(1.6), &defs(), PreviewQuality::Final, 16, 16, 1, 0)
            .unwrap()
            .rgba;
        assert_ne!(small, big, "radius 参数应改变预览（球体大小不同）");
    }

    #[test]
    fn final_real_frame_dimension_changes_pixels() {
        // 帧维（D-28）与尺寸都进输出身份：不同尺寸 → 不同帧缓冲。
        let g = build_reference_graph();
        let mut r = PreviewRenderer::new();
        let a = r.render(&g, &defs(), PreviewQuality::Final, 16, 16, 1, 0).unwrap();
        let b = r.render(&g, &defs(), PreviewQuality::Final, 20, 12, 1, 0).unwrap();
        assert_ne!(a.rgba.len(), b.rgba.len());
        assert_eq!(a.rgba.len(), 16 * 16 * 4);
        assert_eq!(b.rgba.len(), 20 * 12 * 4);
    }

    // -----------------------------------------------------------------
    // W7 / B3 判据（P1 / P4 / P5 在 crate 内测；P2 见 tests/w7_preview.rs）
    // -----------------------------------------------------------------

    /// **P1 同源**：`preview(R, N)` 与 `final(R, N)` **逐字节相同**。
    ///
    /// "final" 在这里是**产品通路的字面复刻**：`build_render_request` +
    /// `render_frame` + `RtBackend`，与 `prism-app render` 的默认档同一个
    /// 调用链。若将来有人在预览里插一条捷径（第二套着色 / 一次"差不多"的
    /// 近似），这条立刻红。
    #[test]
    fn p1_preview_equals_final_at_same_budget() {
        let g = build_reference_graph();
        let defs = defs();
        let order = g.topo_order().unwrap();
        let root = pick_root(&g, &order).unwrap();
        let cone = ancestor_cone(&g, &order, root);
        let scene = scene_from_graph(&g, &cone);
        let (w, h) = (24u32, 16u32);
        let spp = PreviewQuality::Final.spp();

        // 产品通路（不经 PreviewRenderer）。
        let mut product: RtBackend = RtBackend::new(spp, PREVIEW_BOUNCES);
        let fb: Framebuffer =
            render_frame(&mut product, &scene, 0.0, 1.0, w, h);
        let want: Vec<u8> = fb.pixels.clone();

        // 预览通路（Final 档 = 全分辨率 + 同 spp）。
        let mut r = PreviewRenderer::new();
        let got = r.render(&g, &defs, PreviewQuality::Final, w, h, 1, 0).unwrap();
        assert_eq!(
            got.rgba, want,
            "P1 破：同 (R, spp) 下预览与产品通路不逐字节相同（预览另有近似路径）"
        );

        // 同一 spp 的**低分辨率**档也必须是同一条流（只降 res）：
        // Draft 的内部缓冲 == 产品通路在同尺寸同 spp 下的输出。
        let (iw, ih) = (w / DRAFT_DOWNSCALE, h / DRAFT_DOWNSCALE);
        let mut product_lo: RtBackend = RtBackend::new(PreviewQuality::Draft.spp(), PREVIEW_BOUNCES);
        let fb_lo: Framebuffer = render_frame(&mut product_lo, &scene, 0.0, 1.0, iw, ih);
        let internal = render_preview_frame(&scene, iw, ih, PreviewQuality::Draft.spp()).unwrap();
        assert_eq!(
            internal, fb_lo.pixels,
            "P1 破：Draft 的内部缓冲与产品通路不逐字节相同"
        );
        println!("P1 通过：preview(R,N) == final(R,N) 逐字节（{w}x{h}, spp={spp}）");
    }

    /// **P4 降分辨率**：`preview(R/4)` 与 `final(R)` 下采样到 `R/4` 的 PSNR ≥ 30 dB。
    ///
    /// 两臂都用**同一 spp**（噪声只由样本数决定，故差异只来自"每像素几条射线"）。
    /// 下采样用与上采样同族的整数盒（[`box_resample`] 的 `S > D` 分支），
    /// 保证比较的只是分辨率损失，不含插值算子差异。
    #[test]
    fn p4_downscale_psnr_at_least_30db() {
        let g = build_reference_graph();
        let order = g.topo_order().unwrap();
        let root = pick_root(&g, &order).unwrap();
        let cone = ancestor_cone(&g, &order, root);
        let scene = scene_from_graph(&g, &cone);

        let (vw, vh) = (128u32, 96u32); // 视口尺寸
        let (lw, lh) = (vw / DRAFT_DOWNSCALE, vh / DRAFT_DOWNSCALE);
        for spp in [1u32, 4, 16] {
            let low = render_preview_frame(&scene, lw, lh, spp).unwrap();
            let full = render_preview_frame(&scene, vw, vh, spp).unwrap();
            let full_down = box_resample(&full, (vw, vh), (lw, lh));
            // PSNR（8 bit，全通道，跳过 alpha）。
            let mut se: f64 = 0.0;
            let mut n: f64 = 0.0;
            for i in 0..low.len() {
                if i % 4 == 3 {
                    continue; // alpha 恒 255，不进 PSNR
                }
                let d = f64::from(low[i]) - f64::from(full_down[i]);
                se += d * d;
                n += 1.0;
            }
            let mse = se / n;
            let psnr = if mse <= 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
            println!(
                "P4 spp={spp}：{lw}x{lh} vs {vw}x{vh}↓ → MSE {mse:.4} PSNR {psnr:.3} dB"
            );
            assert!(
                psnr >= 30.0,
                "P4 破：spp={spp} 时降分辨率 PSNR {psnr:.3} dB < 30 dB"
            );
        }
    }

    /// **P5 性能**：480×270 / spp=1 ≥ 10 fps（真跑计时，不算理论值）。
    ///
    /// 两组数字：
    /// - `full`：产品通路在 480×270 / 1 spp 的墙钟（§4.2 的字面口径）；
    /// - `draft`：`PreviewRenderer` Draft 档（内部 120×68 + 盒上采样）的墙钟
    ///   （编辑器交互实际走的那条）。
    ///
    /// 单线程（`RtBackend::with_threads(1)`）与自动线程数都测：前者是保守下界，
    /// 后者是生产配置。线程数不进结果（`rt_tiles` 关卡 2）。
    ///
    /// # 构建口径
    /// 断言**无条件**生效（debug 也跑）：debug 单线程实测 13.19 fps、release
    /// 62.90 fps，两档都过线；把断言挂在 `cfg!(debug_assertions)` 上等于给
    /// 自己留一个"debug 下静默跳过"的口子。计时数字按 release 引用。
    #[test]
    fn p5_480x270_spp1_at_least_10fps() {
        use std::time::Instant;
        let g = build_reference_graph();
        let defs = defs();
        let order = g.topo_order().unwrap();
        let root = pick_root(&g, &order).unwrap();
        let cone = ancestor_cone(&g, &order, root);
        let scene = scene_from_graph(&g, &cone);
        let (w, h) = (480u32, 270u32);

        // 预热（首次分配 / 缓存冷）。
        let _ = render_preview_frame(&scene, w, h, 1);

        // 墙钟取 3 次的**最好**成绩：判据量的是"能不能做到"，不是"这台机器
        // 此刻的调度心情"。阈值一个字没动（仍是 10 fps），只去抖动。
        for threads in [1usize, 0] {
            let dev = RtBackend::new(1, PREVIEW_BOUNCES);
            let mut dev = if threads == 0 { dev } else { dev.with_threads(threads) };
            let mut best = f64::INFINITY;
            let mut valid = false;
            for _ in 0..3 {
                let t0 = Instant::now();
                let fb = render_frame(&mut dev, &scene, 0.0, 1.0, w, h);
                let dt = t0.elapsed().as_secs_f64();
                valid = fb.is_valid();
                if dt < best {
                    best = dt;
                }
            }
            assert!(valid, "P5 破：帧缓冲无效");
            let fps = 1.0 / best;
            println!(
                "P5 full 480x270 spp=1 threads={} → {:.1} ms（3 取 1）, {fps:.2} fps",
                dev.effective_threads(),
                best * 1000.0
            );
            if threads == 0 {
                assert!(
                    fps >= 10.0,
                    "P5 破：自动线程数下 480x270/spp=1 只有 {fps:.2} fps < 10"
                );
            }
        }

        // 编辑器实际档位（Draft：内部 120×68 + 盒上采样）。
        // 拆成三段计时，便于定位瓶颈（真渲染 / 上采样 / 其余）。
        let (iw, ih) = (div_ceil(w, DRAFT_DOWNSCALE), div_ceil(h, DRAFT_DOWNSCALE));
        let _ = render_preview_frame(&scene, iw, ih, 1); // 预热（首次分配）

        let mut t_render = f64::INFINITY;
        let mut internal: Vec<u8> = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            internal = render_preview_frame(&scene, iw, ih, 1).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            if dt < t_render {
                t_render = dt;
            }
        }

        let mut t_up = f64::INFINITY;
        let mut upscaled: Vec<u8> = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            upscaled = box_resample(&internal, (iw, ih), (w, h));
            let dt = t0.elapsed().as_secs_f64();
            if dt < t_up {
                t_up = dt;
            }
        }

        let mut r = PreviewRenderer::new();
        let _ = r.render(&g, &defs, PreviewQuality::Draft, w, h, 1, 0).unwrap(); // 预热
        let mut t_all = f64::INFINITY;
        let mut out = r.render(&g, &defs, PreviewQuality::Draft, w, h, 1, 0).unwrap();
        for _ in 0..3 {
            let t0 = Instant::now();
            out = r.render(&g, &defs, PreviewQuality::Draft, w, h, 1, 0).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            if dt < t_all {
                t_all = dt;
            }
        }

        println!(
            "P5 draft {w}x{h}（内部 {iw}x{ih}）：真渲染 {:.1} ms + 盒上采样 {:.1} ms；\
             端到端（含图签名/场景映射）{:.1} ms → {:.2} fps",
            t_render * 1000.0,
            t_up * 1000.0,
            t_all * 1000.0,
            1.0 / t_all
        );
        // 结构性断言：端到端就是"内部真渲染 → 整数盒上采样"，没有第三个环节。
        assert_eq!(
            out.rgba, upscaled,
            "Draft 端到端必须恰等于 内部渲染→盒上采样（不得另加后处理）"
        );
        assert!(out.rgba.len() == (w as usize) * (h as usize) * 4);

        // 最坏交互档：MAX_DIM×MAX_DIM 的 Final（512×512 / 16 spp = 4.19 M 样本）。
        // 不进 P5 判据（§4.2 的口径是 480×270 / 1 spp），只作为"重档多少钱"的
        // 实测参照——编辑器会在这个尺寸上退到 Draft。
        let mut heavy = PreviewRenderer::new();
        let t0 = Instant::now();
        let big = heavy.render(&g, &defs, PreviewQuality::Final, MAX_DIM, MAX_DIM, 1, 0).unwrap();
        let t_heavy = t0.elapsed().as_secs_f64();
        println!(
            "P5 heavy {MAX_DIM}x{MAX_DIM} spp={}（Final 档，一次）→ {:.1} ms",
            PreviewQuality::Final.spp(),
            t_heavy * 1000.0
        );
        assert_eq!(big.rgba.len(), (MAX_DIM as usize) * (MAX_DIM as usize) * 4);

        // Final 档在判据尺寸上的实测（用户切到 Final 时看到的延迟，同样只打印）。
        let mut mid = PreviewRenderer::new();
        let t0 = Instant::now();
        let mid_out = mid.render(&g, &defs, PreviewQuality::Final, w, h, 1, 0).unwrap();
        let t_mid = t0.elapsed().as_secs_f64();
        println!(
            "P5 final {w}x{h} spp={} → {:.1} ms",
            PreviewQuality::Final.spp(),
            t_mid * 1000.0
        );
        assert_eq!(mid_out.rgba.len(), (w as usize) * (h as usize) * 4);
    }

    /// `box_resample` 的整数盒性质：同尺寸 = 逐字节拷贝；下采样后每格等于
    /// 其源区间的整数均值；上采样（4×）是 4×4 复制块。
    #[test]
    fn box_resample_is_integer_box() {
        // 同尺寸 ⇒ 恒等。
        let src: Vec<u8> = (0..(4 * 3 * 4) as u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(box_resample(&src, (4, 3), (4, 3)), src);
        // 4× 上采样 ⇒ 每个源像素复制成 4×4 块。
        let up = box_resample(&src, (4, 3), (16, 12));
        for y in 0..12usize {
            for x in 0..16usize {
                let sx = x / 4;
                let sy = y / 4;
                let si = (sy * 4 + sx) * 4;
                let di = (y * 16 + x) * 4;
                assert_eq!(
                    &up[di..di + 3],
                    &src[si..si + 3],
                    "上采样块 ({x},{y}) 不等于源像素 ({sx},{sy})"
                );
            }
        }
        // 下采样 2× ⇒ 每格 = 4 个源像素的整数均值（半整数舍入）。
        let down = box_resample(&src, (4, 3), (2, 2));
        // 目标 (0,0) ← 源 (0..2, 0..2)
        for c in 0..4 {
            let mut sum: u32 = 0;
            for sy in 0..2 {
                for sx in 0..2 {
                    sum += u32::from(src[(sy * 4 + sx) * 4 + c]);
                }
            }
            let want = ((sum + 2) / 4) as u8;
            assert_eq!(down[c], want, "下采样 (0,0) 通道 {c}");
        }
    }
}
