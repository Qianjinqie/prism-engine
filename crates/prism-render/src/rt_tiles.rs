//! 单帧 tile 化并行渲染（W0-C L3）。
//!
//! # 为什么这个模块能"并行且确定"
//!
//! `prism-rt` 的每像素每样本是**独立纯函数**：坐标 = `Sobol(i, dim) ⊕ 打散掩码`，
//! 其中 `i` 与掩码都由 `hash(种子, 像素, 样本, 维度)` 派生，无全局 RNG 状态
//! （`prism-rt/src/lib.rs` "确定性铁律"）。于是：
//!
//! - 像素之间**没有数据依赖**，谁先算谁后算都不影响任何一个像素的值；
//! - 每像素内部的归约（[`prism_rt::accum::mean_samples`] 的固定形状树）**不在
//!   本模块**，本模块只做"把已经算好的像素搬到帧缓冲的指定位置"——纯字节拷贝，
//!   不产生任何浮点合并。
//!
//! 这两条合起来给出结论：**线程数不进结果**。这不是"测出来一致"，是构造上
//! 一致；测试（`rt_tiles` 的四条关卡）只是把这条构造性结论钉死。
//!
//! # 三条硬纪律（D-20 的准入条件）
//!
//! 1. **tile 尺寸与核数无关**：恒为 [`RT_TILE_W`]×[`RT_TILE_H`]（32×32）。
//!    核数只决定"谁来算"，不决定"怎么切"。若改成 `height / threads`，同一像素
//!    会因核数不同落进不同 tile，一旦将来引入任何 tile 级缓存就立刻破确定性。
//! 2. **每个 tile 完整算完再合并**：tile 产物是独立 `Framebuffer`，不与主帧缓冲
//!    共享可变状态（无锁、无原子、无 `unsafe` 切片切分）。
//! 3. **合并顺序固定**：按 tile 在行主序网格中的索引**升序**逐字节拷贝。
//!    不按"谁先算完谁先写"——那会让写出顺序取决于调度器。
//!
//! # 归约为什么一定在 tile 内
//!
//! 每像素的 spp 均值在 [`prism_rt::integrator::render_pixel`] 内部完成，输入只有
//! 该像素自己的样本。tile 边界与像素边界重合，故**不存在跨 tile 归约**。
//! 这也是"1 线程 vs 16 线程逐字节一致"能成立的根因：没有任何一个浮点加法的
//! 求和顺序依赖线程划分。
//!
//! # 线程数
//!
//! [`RtBackend::threads`] 为 `0` 时取 [`auto_threads`]（`available_parallelism`，
//! 尊重 cgroup 限额）。测试与 A/B 用显式值。**线程数只影响墙钟，不影响输出**，
//! 故不进入任何缓存键或指纹。

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::thread;

// ===========================================================================
// 重建阈值的按档位配置（不加环境变量也能分档）
// ===========================================================================
//
// 这三个阈值决定"一个 4×4 单元算平坦（插值）还是算边缘（逐像素真实路径）"，
// 是画质与速度之间唯一的旋钮。任务书第七节要求**档位差异来自采样与重建
// 策略**，所以它必须能按大模式分档，而不是只有离线环境变量一条路。
//
// 存储用 `AtomicU32` 存 `f32::to_bits`，`0` 表示"未设置、走函数缺省"。选择
// 原子而非线程局部，是因为第二遍是跨线程并行的，所有 worker 必须看到同一组
// 阈值；写入只发生在帧边界（`UniversalReconScope::new`），读是热路径上的
// `Relaxed::load`，与已有的 `universal_shadow_mode` 同一个范式。
static RECON_FLAT_TOL: AtomicU32 = AtomicU32::new(0);
static RECON_SPREAD_TOL: AtomicU32 = AtomicU32::new(0);
static RECON_SPREAD2_TOL: AtomicU32 = AtomicU32::new(0);

// ===========================================================================
// 真实计算量计账（如实报数，不是性能开关）
// ===========================================================================
//
// 稀疏网格 + 门控重建让每帧**真正跑路径积分的像素数**远小于 `宽 × 高`，
// 所以对外报「本帧真实新增样本」时必须用它乘 `spp`。拿 `宽 × 高 × spp` 冒充
// 就是任务书第零节五明令禁止的「伪造 / 摊分测量值」——数字看着没降，实际
// 算的少了，那是假账。这两个计数器就是这份账本：每一处真实发出
// `render_pixel_range_universal` 或单条主射线的地方都记账，没有任何地方
// 按「看起来应该采了多少」反推。
//
// 写入只在帧末做一次（一次 `store`），读跨帧，`Relaxed` 足够。
static FRAME_SAMPLED_PIXELS: AtomicU64 = AtomicU64::new(0);
static FRAME_EXTRA_RAYS: AtomicU64 = AtomicU64::new(0);

/// 上一帧真实求值的像素条目数与附加主射线数：`(像素条目, 附加射线)`。
///
/// 语义：`像素条目 × spp = 真实样本数`；附加射线是纹理分离校正那条单主射线，
/// 不按 `spp` 计，单独报出来，绝不并进样本数里凑。
pub fn last_frame_sample_accounting() -> (u64, u64) {
    (
        FRAME_SAMPLED_PIXELS.load(Ordering::Relaxed),
        FRAME_EXTRA_RAYS.load(Ordering::Relaxed),
    )
}

/// 记一笔「整帧全部像素都真实求值」的账（单遍直渲路径）。
#[inline]
fn account_full_frame(pixels: u64) {
    FRAME_SAMPLED_PIXELS.store(pixels, Ordering::Relaxed);
    FRAME_EXTRA_RAYS.store(0, Ordering::Relaxed);
}

/// 读取一档重建阈值：未设置时用函数缺省。
#[inline]
fn recon_tol(cell: &AtomicU32, fallback: f32) -> f32 {
    let bits: u32 = cell.load(Ordering::Relaxed);
    if bits == 0 {
        fallback
    } else {
        f32::from_bits(bits)
    }
}

/// 重建阈值的帧级作用域；析构恢复旧值，不泄漏到 `一致` 档或独立 golden。
pub struct UniversalReconScope {
    prev: [u32; 3],
}

impl UniversalReconScope {
    /// 在当前帧打开一组重建阈值（二阶差分 / 四角极差 / 九样本极差）。
    pub fn new(flat_tol: f32, spread_tol: f32, spread2_tol: f32) -> Self {
        UniversalReconScope {
            prev: [
                RECON_FLAT_TOL.swap(flat_tol.to_bits(), Ordering::Relaxed),
                RECON_SPREAD_TOL.swap(spread_tol.to_bits(), Ordering::Relaxed),
                RECON_SPREAD2_TOL.swap(spread2_tol.to_bits(), Ordering::Relaxed),
            ],
        }
    }
}

impl Drop for UniversalReconScope {
    fn drop(&mut self) {
        RECON_FLAT_TOL.store(self.prev[0], Ordering::Relaxed);
        RECON_SPREAD_TOL.store(self.prev[1], Ordering::Relaxed);
        RECON_SPREAD2_TOL.store(self.prev[2], Ordering::Relaxed);
    }
}

use prism_core::EvalCtx;
use prism_gpu::{AestheticReq, Framebuffer};
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    render_pixel, render_pixel_range_universal, universal_key_compatible, universal_key_is_miss,
    universal_primary_key, pixel_base_color, PrimaryHitScope, RenderParams, Scene,
    UniversalAccel, UniversalIntersectGuard,
};

use crate::tile::{Tile, TileGrid};

/// tile 宽（像素）。**固定值，不得随核数变化**（D-20 准入条件）。
pub const RT_TILE_W: u32 = 32;

/// tile 高（像素）。**固定值，不得随核数变化**（D-20 准入条件）。
pub const RT_TILE_H: u32 = 32;

/// 默认线程数：`available_parallelism`（已尊重 cgroup / 亲和性限额），至少 1。
pub fn auto_threads() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}

/// 本帧的 tile 网格（**只依赖帧尺寸**，与线程数无关）。
pub fn grid(width: u32, height: u32) -> TileGrid {
    TileGrid::new(width, height, RT_TILE_W, RT_TILE_H)
}

/// 渲染一个 tile 到独立的 tile 局部帧缓冲（左上角为原点）。
///
/// 像素一律用**全局坐标**喂 [`render_pixel`]（D-28 像素域）——tile 划分不得
/// 影响采样流，这是"tile 化与整幅逐位一致"的前提。
fn render_tile(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    tile: Tile,
) -> Framebuffer {
    let mut fb: Framebuffer = Framebuffer::new(tile.w, tile.h);
    for ly in 0..tile.h {
        for lx in 0..tile.w {
            let gx: u32 = tile.x + lx;
            let gy: u32 = tile.y + ly;
            let rgba: [f32; 4] = render_pixel(scene, cam, ctx, (gx, gy), resolution, params);
            let linear: prism_math::Vec3 = prism_math::Vec3::new(rgba[0], rgba[1], rgba[2]);
            // 与 `CpuRefBackend` 逐式相同的后处理链（唯一实现见 prism-gpu）。
            let mapped: prism_math::Vec3 = prism_gpu::apply_aesthetic(linear, aesthetic);
            fb.set_pixel(lx, ly, mapped);
        }
    }
    fb
}

/// 把 tile 产物按**行主序 tile 索引升序**拷进主帧缓冲。
///
/// 纯字节搬运：不读不写任何浮点，故合并顺序对结果无影响——但顺序仍固定，
/// 以便"合并"本身也是可复现的一步（不依赖调度器完成顺序）。
fn blit_tile(dst: &mut Framebuffer, tile: Tile, src: &Framebuffer) {
    debug_assert_eq!((src.width, src.height), (tile.w, tile.h));
    let dst_stride: usize = dst.width as usize * 4;
    let src_stride: usize = tile.w as usize * 4;
    for ly in 0..tile.h as usize {
        let d0: usize = (tile.y as usize + ly) * dst_stride + tile.x as usize * 4;
        let s0: usize = ly * src_stride;
        dst.pixels[d0..d0 + src_stride].copy_from_slice(&src.pixels[s0..s0 + src_stride]);
    }
}

/// 渲染整帧（tile 化 + `threads` 路并行）。
///
/// `threads <= 1` 时退化为**逐 tile 顺序**求值——注意这**不是**"另一种算法"：
/// tile 划分、像素坐标、归约位置全同，只是没有并发。故 1 线程与 N 线程
/// 的输出必然逐字节相同（关卡见 `rt_tiles` 的测试）。
///
/// # Panics
///
/// 不 panic：尺寸非法时返回 `Framebuffer::new` 给出的拒绝缓冲（由调用方判
/// `is_valid`）。`tile` 索引越界属程序错误，由 [`TileGrid::tile`] 的断言暴露。
pub fn render_frame(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
) -> Framebuffer {
    let (width, height) = resolution;
    let mut frame: Framebuffer = Framebuffer::new(width, height);
    if !frame.is_valid() {
        return frame;
    }
    // 单遍直渲：每个像素都真实跑了 `spp` 个样本，样本条目 = 宽 × 高。
    account_full_frame((width as u64).saturating_mul(height as u64));
    let g: TileGrid = grid(width, height);
    let tiles: Vec<Tile> = g.tiles().collect();
    let workers: usize = threads.clamp(1, tiles.len().max(1));

    if workers == 1 {
        for &t in &tiles {
            let fb: Framebuffer = render_tile(scene, cam, ctx, resolution, params, aesthetic, t);
            blit_tile(&mut frame, t, &fb);
        }
        return frame;
    }

    // 结果槽：(tile 索引, 产物)。每 worker 一个私有 Vec（无共享可变状态），
    // scope 结束后统一按索引排序再合并——**完成顺序不参与结果**。
    let mut collected: Vec<(usize, Framebuffer)> = Vec::with_capacity(tiles.len());
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let tiles: &[Tile] = &tiles;
            handles.push(s.spawn(move || {
                let mut part: Vec<(usize, Framebuffer)> = Vec::new();
                let mut idx: usize = w;
                while idx < tiles.len() {
                    let t: Tile = tiles[idx];
                    part.push((
                        idx,
                        render_tile(scene, cam, ctx, resolution, params, aesthetic, t),
                    ));
                    idx += workers;
                }
                part
            }));
        }
        for h in handles {
            match h.join() {
                Ok(part) => collected.extend(part),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
    });

    // 固定序合并：按 tile 索引升序，与 worker / 完成顺序无关。
    collected.sort_by_key(|(idx, _)| *idx);
    for (idx, fb) in &collected {
        blit_tile(&mut frame, tiles[*idx], fb);
    }
    frame
}

// ---------------------------------------------------------------------------
// V3.1 T03：f32 线性帧旁路（同一 trace_sample / 同一 mean_samples_fill）
// ---------------------------------------------------------------------------

/// 渲染一个 tile 的 **f32 线性**像素（不 tone-map、不量化）。
///
/// 与 [`render_tile`] 的唯一差别是末尾不调用 `apply_aesthetic` /
/// `Framebuffer::set_pixel`，而是把 [`render_pixel_range`] 的 `[f32;4]`
/// 原样返回。全局像素坐标喂入、tile 边界与像素边界重合 ⇒ 与整幅逐位一致。
fn render_tile_linear(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    tile: Tile,
    s_begin: u32,
    s_end: u32,
) -> Vec<[f32; 4]> {
    use prism_rt::integrator::render_pixel_range;

    let mut out: Vec<[f32; 4]> = Vec::with_capacity((tile.w * tile.h) as usize);
    for ly in 0..tile.h {
        for lx in 0..tile.w {
            let gx: u32 = tile.x + lx;
            let gy: u32 = tile.y + ly;
            out.push(render_pixel_range(
                scene,
                cam,
                ctx,
                (gx, gy),
                resolution,
                params,
                s_begin,
                s_end,
            ));
        }
    }
    out
}

/// 把 tile 的线性产物按**行主序 tile 索引升序**拷进整幅线性帧。
fn blit_linear(dst: &mut [[f32; 4]], frame_w: u32, tile: Tile, src: &[[f32; 4]]) {
    for ly in 0..tile.h as usize {
        let d0: usize = (tile.y as usize + ly) * (frame_w as usize) + tile.x as usize;
        let s0: usize = ly * (tile.w as usize);
        dst[d0..d0 + tile.w as usize].copy_from_slice(&src[s0..s0 + tile.w as usize]);
    }
}

/// 渲染整帧 f32 线性像素（行主序 `w×h`），样本区间 `[s_begin, s_end)`。
///
/// # 同源保证（V3.1 时域累积的构造性基础）
/// `render_pixel_range` 的第 `s` 个样本只由 `(种子, 像素, s, 维度)` 决定，
/// **与 `params.spp` 无关**（W7/B3 前缀语义）。故 W 个窗口 pass 依次取
/// `[0,4) [4,8) [8,12) [12,16)` 覆盖的 16 个样本，与一次 `spp=16`
/// 直渲的 16 个样本**逐位相同**；`mean_samples_fill` 固定形状树保证
/// 每个 pass 内部归约也同源。差异仅来自窗口间 `running_mean` 的 f64
/// 结合序，量级远小于 8-bit 判据预算。
///
/// # 确定性（D-20 三条纪律原样适用）
/// 32×32 固定 tile、每 tile 完整算完再合并、按 tile 索引升序写入；
/// 线程数只影响墙钟，不进结果。
pub fn render_frame_linear(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    s_begin: u32,
    s_end: u32,
    threads: usize,
) -> Vec<[f32; 4]> {
    let (width, height) = resolution;
    let n: usize = (width as usize).saturating_mul(height as usize);
    if n == 0 || width > prism_gpu::MAX_FRAMEBUFFER_DIM || height > prism_gpu::MAX_FRAMEBUFFER_DIM {
        return Vec::new();
    }
    if s_end <= s_begin {
        return vec![[0.0; 4]; n];
    }
    let g: TileGrid = grid(width, height);
    let tiles: Vec<Tile> = g.tiles().collect();
    let workers: usize = threads.clamp(1, tiles.len().max(1));
    let mut frame: Vec<[f32; 4]> = vec![[0.0; 4]; n];

    if workers == 1 {
        for &t in &tiles {
            let part = render_tile_linear(scene, cam, ctx, resolution, params, t, s_begin, s_end);
            blit_linear(&mut frame, width, t, &part);
        }
        return frame;
    }

    let mut collected: Vec<(usize, Vec<[f32; 4]>)> = Vec::with_capacity(tiles.len());
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let tiles: &[Tile] = &tiles;
            handles.push(s.spawn(move || {
                let mut part: Vec<(usize, Vec<[f32; 4]>)> = Vec::new();
                let mut idx: usize = w;
                while idx < tiles.len() {
                    let t: Tile = tiles[idx];
                    part.push((
                        idx,
                        render_tile_linear(
                            scene, cam, ctx, resolution, params, t, s_begin, s_end,
                        ),
                    ));
                    idx += workers;
                }
                part
            }));
        }
        for h in handles {
            match h.join() {
                Ok(part) => collected.extend(part),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
    });
    collected.sort_by_key(|(idx, _)| *idx);
    for (idx, part) in &collected {
        blit_linear(&mut frame, width, tiles[*idx], part);
    }
    frame
}

// ---------------------------------------------------------------------------
// V3.1 独立通用路径：按档位启用，不按场景内容分流
// ---------------------------------------------------------------------------

/// 渲染一个 tile 到独立 RGBA8 帧缓冲；像素仍用全局坐标喂独立通用路径。
fn render_tile_universal(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    accel: &UniversalAccel,
    tile: Tile,
) -> Framebuffer {
    let mut fb: Framebuffer = Framebuffer::new(tile.w, tile.h);
    for ly in 0..tile.h {
        for lx in 0..tile.w {
            let gx: u32 = tile.x + lx;
            let gy: u32 = tile.y + ly;
            let rgba: [f32; 4] = render_pixel_range_universal(
                scene,
                cam,
                ctx,
                (gx, gy),
                resolution,
                params,
                accel,
                0,
                params.spp,
            );
            let linear: prism_math::Vec3 =
                prism_math::Vec3::new(rgba[0], rgba[1], rgba[2]);
            let mapped: prism_math::Vec3 = prism_gpu::apply_aesthetic(linear, aesthetic);
            fb.set_pixel(lx, ly, mapped);
        }
    }
    fb
}

/// 渲染整帧独立通用路径：先为任意场景构造只读加速视图，再按固定 tile 并行。
///
/// 守卫按 worker 线程挂接，作用域只覆盖本函数；`一致` 档与独立 golden 不经过
/// 这里，底层 `Scene::intersect` / `Scene::any_hit` 保持冻结老路径。
pub fn render_frame_universal(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
) -> Framebuffer {
    let accel: UniversalAccel = UniversalAccel::build(scene);
    render_frame_universal_with_accel(
        scene, cam, ctx, resolution, params, aesthetic, threads, &accel,
    )
}

/// 使用外部缓存的只读加速视图渲染整帧；`一致` 档与独立 golden 不经过这里。
pub fn render_frame_universal_with_accel(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
    accel: &UniversalAccel,
) -> Framebuffer {
    let (width, height) = resolution;
    let mut frame: Framebuffer = Framebuffer::new(width, height);
    if !frame.is_valid() {
        return frame;
    }
    // 单遍直渲：每个像素都真实跑了 `spp` 个样本，样本条目 = 宽 × 高。
    account_full_frame((width as u64).saturating_mul(height as u64));
    let g: TileGrid = grid(width, height);
    let tiles: Vec<Tile> = g.tiles().collect();
    let workers: usize = threads.clamp(1, tiles.len().max(1));

    if workers == 1 {
        let guard = UniversalIntersectGuard::new(scene, accel);
        for &t in &tiles {
            let fb: Framebuffer = render_tile_universal(
                scene, cam, ctx, resolution, params, aesthetic, accel, t,
            );
            blit_tile(&mut frame, t, &fb);
        }
        drop(guard);
        return frame;
    }

    let mut collected: Vec<(usize, Framebuffer)> = Vec::with_capacity(tiles.len());
    let accel_ref: &UniversalAccel = accel;
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let tiles: &[Tile] = &tiles;
            handles.push(s.spawn(move || {
                let guard = UniversalIntersectGuard::new(scene, accel_ref);
                let mut part: Vec<(usize, Framebuffer)> = Vec::new();
                let mut idx: usize = w;
                while idx < tiles.len() {
                    let t: Tile = tiles[idx];
                    part.push((
                        idx,
                        render_tile_universal(
                            scene, cam, ctx, resolution, params, aesthetic, accel_ref, t,
                        ),
                    ));
                    idx += workers;
                }
                drop(guard);
                part
            }));
        }
        for h in handles {
            match h.join() {
                Ok(part) => collected.extend(part),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
    });

    collected.sort_by_key(|(idx, _)| *idx);
    for (idx, fb) in &collected {
        blit_tile(&mut frame, tiles[*idx], fb);
    }
    frame
}

/// 独立通用路径的整帧直渲入口。
///
/// 与固定 tile 版本逐像素同式，但不在每帧再为每个 32×32 tile 分配临时
/// `Framebuffer`、收集 `Vec` 与按索引排序；只按行区间把一整幅线性颜色写进
/// 一份数组，再逐式 tone-map。`一致` 档与独立 golden 不经过这里。
pub fn render_frame_universal_direct(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
) -> Framebuffer {
    let accel: UniversalAccel = UniversalAccel::build(scene);
    render_frame_universal_direct_with_accel(
        scene,
        cam,
        ctx,
        resolution,
        params,
        aesthetic,
        threads,
        &accel,
    )
}

/// 使用外部缓存的只读加速视图做整帧直渲；供 `V31RenderDevice` 按当帧几何
/// 指纹复用加速视图时调用，避免每帧重建结构与固定 tile 的中间帧缓冲。
pub fn render_frame_universal_direct_with_accel(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
    accel: &UniversalAccel,
) -> Framebuffer {
    let (width, height) = resolution;
    let mut frame: Framebuffer = Framebuffer::new(width, height);
    if !frame.is_valid() {
        return frame;
    }
    let wu: usize = width as usize;
    let hu: usize = height as usize;
    let pixels: usize = wu.saturating_mul(hu);
    if pixels == 0 {
        return frame;
    }
    // 单遍直渲：全部像素真实求值。
    account_full_frame(pixels as u64);
    let accel_ref: &UniversalAccel = accel;
    let mut linear: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; pixels];

    let worker_cap: usize = threads.clamp(1, hu.max(1));
    let rows_per: usize = ((hu + worker_cap - 1) / worker_cap).max(1);
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut y0: usize = 0;
    while y0 < hu {
        let y1: usize = (y0 + rows_per).min(hu);
        ranges.push((y0, y1));
        y0 = y1;
    }
    {
        let mut rest: &mut [[f32; 3]] = &mut linear[..];
        let mut parts: Vec<(usize, usize, &mut [[f32; 3]])> =
            Vec::with_capacity(ranges.len());
        for &(a, b) in &ranges {
            let len: usize = (b - a) * wu;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            parts.push((a, b, head));
        }
        thread::scope(|scope| {
            for (a, b, part) in parts {
                spawn_worker(scope, move || {
                    let guard = UniversalIntersectGuard::new(scene, accel_ref);
                    for ly in 0..(b - a) {
                        let gy: u32 = (a + ly) as u32;
                        for gx in 0..width {
                            let rgba: [f32; 4] = render_pixel_range_universal(
                                scene,
                                cam,
                                ctx,
                                (gx, gy),
                                resolution,
                                params,
                                accel_ref,
                                0,
                                params.spp,
                            );
                            part[ly * wu + gx as usize] = [rgba[0], rgba[1], rgba[2]];
                        }
                    }
                    drop(guard);
                });
            }
        });
    }
    for gy in 0..height {
        for gx in 0..width {
            let c: [f32; 3] = linear[gy as usize * wu + gx as usize];
            let v: prism_math::Vec3 = prism_math::Vec3::new(c[0], c[1], c[2]);
            let mapped: prism_math::Vec3 = prism_gpu::apply_aesthetic(v, aesthetic);
            frame.set_pixel(gx, gy, mapped);
        }
    }
    frame
}

/// 生成稀疏网格节点坐标；末节点固定落在最后一列 / 最后一行。
fn universal_grid_axis(len: usize, stride: usize) -> Vec<usize> {
    let step: usize = stride.max(1);
    let mut nodes: Vec<usize> = Vec::new();
    let mut x: usize = 0;
    while x < len {
        nodes.push(x);
        x = x.saturating_add(step);
    }
    let last: usize = len.saturating_sub(1);
    if nodes.last().copied() != Some(last) {
        nodes.push(last);
    }
    if nodes.len() == 1 {
        nodes.push(last);
    }
    nodes
}

/// 逐像素映射到所属网格单元；单元数恒为节点数减一。
fn universal_cell_map(len: usize, nodes: &[usize]) -> Vec<usize> {
    let cells: usize = nodes.len().saturating_sub(1).max(1);
    let mut map: Vec<usize> = vec![0; len];
    let mut cell: usize = 0;
    let last_cell: usize = cells - 1;
    for (x, dst) in map.iter_mut().enumerate() {
        while cell < last_cell && x >= nodes[cell + 1] {
            cell += 1;
        }
        *dst = cell;
    }
    map
}

/// 以固定小栈在 `scope` 内起一个渲染工作线程。
///
/// 默认线程栈是 2 MiB。独立通用路径每帧要起 **三** 批工作线程（稀疏网格
/// 取样、二级细化、逐像素重建），每批按可用并行度切，320×180 上合计 12 次
/// `mmap` + `munmap` 与首次触碰缺页，实测占掉整帧墙钟的三分之一（约 3 ms）：
/// 这是小分辨率档位独有的固定开销，也是 320×180 迟迟压不进门槛的根因。
/// BVH 遍历已迭代化、路径积分层次浅，128 KiB 栈绰绰有余。
fn spawn_worker<'scope, 'env, F>(scope: &'scope thread::Scope<'scope, 'env>, f: F)
where
    F: FnOnce() + Send + 'scope,
{
    thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn_scoped(scope, f)
        .expect("创建渲染工作线程失败");
}

/// 独立通用路径的「稀疏真实样本 + 几何一致重建」整帧入口。
///
/// 每一帧都按当前 `SceneRenderRequest` 派生出的场景与相机重新工作：只在
/// 固定网格节点上求当前帧主射线几何键并走完整路径追踪；角点几何键一致
/// 的平坦单元用四个本帧真实样本重建，角点键不一致的边缘单元按空间象限
/// 选最近的本帧真实样本。没有第二条完整路径，没有历史帧、没有锁存请求、
/// 没有图像域平滑：它复用的是同一帧同一几何表面上的辐射样本，不是把噪点
/// 糊掉。
pub fn render_frame_universal_reconstructed(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    aesthetic: &AestheticReq,
    threads: usize,
    stride: u32,
    accel: &UniversalAccel,
) -> Framebuffer {
    let (width, height) = resolution;
    if width == 0 || height == 0 {
        return Framebuffer::empty();
    }
    if stride <= 1 {
        return render_frame_universal_direct_with_accel(
            scene, cam, ctx, resolution, params, aesthetic, threads, accel,
        );
    }

    let wu: usize = width as usize;
    let hu: usize = height as usize;
    let pixels: usize = wu.saturating_mul(hu);
    if pixels == 0 {
        return Framebuffer::empty();
    }
    let stride_us: usize = stride.clamp(2, 8) as usize;
    let accel_ref: &UniversalAccel = accel;
    let setup_started: Option<std::time::Instant> = if std::env::var("PRISM_V31_TILE_PROFILE")
        .ok()
        .as_deref()
        == Some("1")
    {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let xs: Vec<usize> = universal_grid_axis(wu, stride_us);
    let ys: Vec<usize> = universal_grid_axis(hu, stride_us);
    let gw: usize = xs.len();
    let gh: usize = ys.len();
    let node_count: usize = gw.saturating_mul(gh);
    if node_count == 0 {
        return Framebuffer::empty();
    }
    let x_cells: Vec<usize> = universal_cell_map(wu, &xs);
    let y_cells: Vec<usize> = universal_cell_map(hu, &ys);
    let xs_ref: &[usize] = &xs[..];
    let ys_ref: &[usize] = &ys[..];

    // 网格节点的颜色样本数：**与逐像素兜底同预算**。
    //
    // 早前这里写过 `params.spp.min(8)`，理由是"节点的产出只用于低方差区域的
    // 双线性与二级细化，8 个样本已够"。那同样是把域内片源的收敛性当成通用
    // 前提：节点的真实样本是重建**唯一**的信息源，它一旦比逐像素少，整个重建
    // 的噪声上限就被节点样本钳住——平坦区插值出来的是"8 spp 噪声的平滑版"，
    // 与 16 spp 参考在噪声主导区必然偏离，且这种偏离**不能**靠提高逐像素
    // 预算补回来。域内两片源（球面明暗 + 渐变背景）方差极小，掩盖了这一点。
    // 现在节点与兜底同用 `params.spp`：稀疏网格省的是**样本个数**（网格点数
    // 远少于像素数），不是**每点样本数**。
    let spp_node: u32 = params.spp.max(1);
    let mut keys: Vec<u64> = vec![0; node_count];
    let mut samples: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; node_count];
    // 纹理分离所需的节点 albedo 与「该节点首命中是否有贴图」标志。
    //
    // 节点除了带 albedo 的辐亮度，还要记下自己那一点的 albedo —— 第二遍才
    // 能在像素处用 `albedo_px / albedo_interp` 把被插值抹平的高频纹理还原。
    // `mapped` 为 0 的节点不参与校正（无贴图、或射线未命中）。
    let mut node_albedo: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; node_count];
    let mut node_mapped: Vec<u8> = vec![0; node_count];
    // **不做场景级门控**（2026-11-14 裁定三）：贴图分离是重建里的一个
    // 校正项，它的适用性由**每个单元自己的数据**决定（四角 `mapped` 标志），
    // 而不是由"这个场景有没有贴图"决定。任何按场景内容切换路线的判定都被
    // 禁止——那正是上一版被打回的场景特判。因此这里对每个节点都取一次
    // albedo（单条主射线，不按 `spp` 计），无贴图场景取回 `mapped = 0`，
    // 第二遍的校正分支自然整段跳过，输出与"没有这个功能"逐字节相同。
    // ---- 真实计算量账本 ----
    //
    // 第一遍：每个网格节点跑 1 次 `spp_node` 样本的路径积分；`tex_split` 打开
    // 时再各补 1 条主射线取 albedo（单条射线，不按 `spp` 计）。
    // 第二遍：平坦单元 0 次，二级细化单元 5 次，其余单元逐像素——由 worker
    // 各自局部计数后归并，不在热循环里打原子。
    let mut sampled_px: u64 = node_count as u64;
    let extra_rays: u64 = node_count as u64;
    // 第二遍由多个 worker 并行，各自局部累加，收尾各做一次原子加。
    // 声明在帧作用域，供第二遍结束后的账本收口读取。
    let traced_px: AtomicU64 = AtomicU64::new(0);
    let traced_alb: AtomicU64 = AtomicU64::new(0);
    // 帧缓冲直接作为第二遍的写入目标：不再先铺一层 `colors` 线性缓冲再遍历
    // 一遍做 tone-map。省下的是一次 691 KB（320×180）/ 11 MB（720p）的分配、
    // 一次全帧写读往返、和一次 4 线程 spawn。色调映射的算术与逐像素顺序
    // 完全不变，输出逐字节相同。
    let mut frame: Framebuffer = Framebuffer::new(width, height);
    if !frame.is_valid() {
        return frame;
    }
    let setup_us: u64 = setup_started
        .map(|t| t.elapsed().as_micros() as u64)
        .unwrap_or(0);
    let profile_enabled: bool =
        std::env::var("PRISM_V31_TILE_PROFILE").ok().as_deref() == Some("1");

    // 第一遍：只在稀疏网格节点上求当前帧几何键并做真实路径求值。节点
    // 0 号样本首交点由帧级作用域缓存给紧随其后的路径积分消费。
    let first_started: Option<std::time::Instant> = if profile_enabled {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let primary_scope = PrimaryHitScope::new();
    {
        let worker_cap: usize = threads.clamp(1, gh.max(1));
        let rows_per: usize = ((gh + worker_cap - 1) / worker_cap).max(1);
        let mut key_rest: &mut [u64] = &mut keys[..];
        let mut sample_rest: &mut [[f32; 3]] = &mut samples[..];
        let mut albedo_rest: &mut [[f32; 3]] = &mut node_albedo[..];
        let mut mapped_rest: &mut [u8] = &mut node_mapped[..];
        let mut parts: Vec<(
            usize,
            usize,
            &mut [u64],
            &mut [[f32; 3]],
            &mut [[f32; 3]],
            &mut [u8],
        )> = Vec::new();
        let mut iy0: usize = 0;
        while iy0 < gh {
            let iy1: usize = (iy0 + rows_per).min(gh);
            let len: usize = (iy1 - iy0) * gw;
            let (kh, kt) = key_rest.split_at_mut(len);
            key_rest = kt;
            let (sh, st) = sample_rest.split_at_mut(len);
            sample_rest = st;
            let (ah, at) = albedo_rest.split_at_mut(len);
            albedo_rest = at;
            let (mh, mt) = mapped_rest.split_at_mut(len);
            mapped_rest = mt;
            parts.push((iy0, iy1, kh, sh, ah, mh));
            iy0 = iy1;
        }
        thread::scope(|scope| {
            for (a, b, key_part, sample_part, albedo_part, mapped_part) in parts {
                spawn_worker(scope, move || {
                    let guard = UniversalIntersectGuard::new(scene, accel_ref);
                    for iy in a..b {
                        let gy: u32 = ys_ref[iy] as u32;
                        for ix in 0..gw {
                            let gx: u32 = xs_ref[ix] as u32;
                            let li: usize = (iy - a) * gw + ix;
                            let pixel: (u32, u32) = (gx, gy);
                            let key: u64 = universal_primary_key(
                                scene, cam, ctx, pixel, resolution, accel_ref,
                            );
                            let rgba: [f32; 4] = render_pixel_range_universal(
                                scene,
                                cam,
                                ctx,
                                pixel,
                                resolution,
                                params,
                                accel_ref,
                                0,
                                spp_node,
                            );
                            let (alb, mapped): ([f32; 3], bool) =
                                pixel_base_color(scene, cam, ctx, pixel, resolution);
                            key_part[li] = key;
                            sample_part[li] = [rgba[0], rgba[1], rgba[2]];
                            albedo_part[li] = alb;
                            mapped_part[li] = u8::from(mapped);
                        }
                    }
                    drop(guard);
                });
            }
        });
    }
    drop(primary_scope);
    let first_us: u64 = first_started
        .map(|t| t.elapsed().as_micros() as u64)
        .unwrap_or(0);
    let second_started: Option<std::time::Instant> = if profile_enabled {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // 第二遍：按单元角点的「几何键 + 辐射一致性」预分平坦 / 边缘两类。
    // 几何键兼容只保证四角落同一表面，程序纹理 / 高光 / 检查格线会让四角
    // 颜色高频变化。只有几何键兼容**且**四角真实样本辐射彼此接近才判平坦、
    // 用双线性插值省时；否则归边缘、逐像素真实路径求值。这是第零节「自适应
    // 采样：平坦区域少采、复杂区域多采」的数据驱动实现，判据是当帧真实样本
    // 辐射值，对任意场景一致生效，不按场景内容分流、不是图像域平滑。
    let flat_tol: f32 = std::env::var("PRISM_V31_UNIV_FLAT_TOL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or_else(|| recon_tol(&RECON_FLAT_TOL, 0.05));
    // 「同容器 + 辐射一阶连续」档的四角极差上限（线性辐射域）。缺省值按
    // 球面明暗在 4 像素块内的典型落差标定：真轮廓（球 / 背景交界）的极差
    // 量级是 1.0 以上，远在阈值之外。
    let spread_tol: f32 = std::env::var("PRISM_V31_UNIV_SPREAD_TOL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or_else(|| recon_tol(&RECON_SPREAD_TOL, 0.25));
    {
        let cell_w: usize = gw.saturating_sub(1).max(1);
        let cell_h: usize = gh.saturating_sub(1).max(1);
        let mut flat_cells: Vec<u8> = vec![0; cell_w.saturating_mul(cell_h)];
        let mut stat_geom: usize = 0;
        let mut stat_smooth: usize = 0;
        for cy in 0..cell_h {
            for cx in 0..cell_w {
                let i00: usize = cy * gw + cx;
                let i10: usize = i00 + 1;
                let i01: usize = i00 + gw;
                let i11: usize = i01 + 1;
                let k00: u64 = keys[i00];
                let k10: u64 = keys[i10];
                let k01: u64 = keys[i01];
                let k11: u64 = keys[i11];
                let geom_flat: bool = universal_key_compatible(k00, k10)
                    && universal_key_compatible(k00, k01)
                    && universal_key_compatible(k00, k11)
                    && universal_key_compatible(k10, k01)
                    && universal_key_compatible(k10, k11)
                    && universal_key_compatible(k01, k11);
                // 双线性可表示性：四角样本的**二阶交叉差分**
                // `s00 + s11 − s10 − s01` 逐通道取最大绝对值。若它接近零，单元
                // 内辐射在双线性函数空间里，四角插值就是精确重建（球面明暗这类
                // 一阶渐变任意陡都仍是零二阶差），应判平坦、插值铺开。若它很大，
                // 说明存在鞍点 / 高频（纹理格线、轮廓跳变、遮挡交界），必须逐
                // 像素走真实路径。
                //
                // 相比"四角极差"，二阶差分对一阶渐变不敏感——球面明暗在 4 像素
                // 块内的亮度差可以远超阈值，但插值误差仍是二阶小量，用极差判据
                // 会把整个球面误判成"边缘"，让稀疏网格退化成逐像素，白白丢掉
                // 加速。判据只用当帧真实样本辐射，对任意场景一致生效，不按场景
                // 内容分流，也不是图像域平滑（不读邻域以外、不读历史帧）。
                let s00: [f32; 3] = samples[i00];
                let s10: [f32; 3] = samples[i10];
                let s01: [f32; 3] = samples[i01];
                let s11: [f32; 3] = samples[i11];
                let mut curv: f32 = 0.0;
                for c in 0..3 {
                    let d2: f32 = (s00[c] + s11[c]) - (s10[c] + s01[c]);
                    curv = curv.max(d2.abs());
                }
                let radiant_flat: bool = curv <= flat_tol;
                // 「同表面 + 一阶连续」的第二档判据。
                //
                // 几何键兼容是个很严的量：球面在球缘处 4 像素间距就能跨掉好几个
                // 深度档，于是球缘一整圈单元被踢进"边缘"，每个像素都重跑完整
                // 路径。实测 v31_motion 320×180 上这类单元占 17% 像素、独吃次遍
                // 7 ms（spans 0.25 ms → 7 ms）。但它们的**辐射**是连续的（同一
                // 个球面、同一个材质、没有遮挡跳变），四角线性插值在这个尺度上
                // 足够准。
                //
                // 故补一条：只要四角落在**同一个几何容器**里、且四角辐射极差在
                // 阈值内，就允许插值。容器不同（球 vs 背景、球 vs 挡板）一律
                // 逐像素——那才是真轮廓，糊了就是判据① 的满量程离群。
                let same_shape: bool = (k00 >> 40) == (k10 >> 40)
                    && (k00 >> 40) == (k01 >> 40)
                    && (k00 >> 40) == (k11 >> 40);
                let mut spread: f32 = 0.0;
                for c in 0..3 {
                    let lo: f32 = s00[c].min(s10[c]).min(s01[c]).min(s11[c]);
                    let hi: f32 = s00[c].max(s10[c]).max(s01[c]).max(s11[c]);
                    spread = spread.max(hi - lo);
                }
                // 只在**几何跨档**时用「同容器 + 辐射连续」这条兜底：几何键
                // 全兼容但二阶差分超阈值的单元，是块内真有高频（窄镜面高光、
                // 程序纹理格线），必须逐像素——不能让兜底条款把它覆盖成平坦。
                let smooth_same: bool = !geom_flat && same_shape && spread <= spread_tol;
                flat_cells[cy * cell_w + cx] =
                    u8::from((geom_flat && radiant_flat) || smooth_same);
                if geom_flat {
                    stat_geom += 1;
                }
                if smooth_same {
                    stat_smooth += 1;
                }
            }
        }
        // 自动降级：重建只在**真的能省时**启用。
        //
        // 稀疏网格的收益 = 平坦单元占比。平坦单元插值、边缘单元逐像素，故
        // 理论成本 ≈ `1/stride² + 非平坦占比`。当非平坦占比接近 1（程序纹理 /
        // 位移网格 / 细几何 / 体积介质这类高频内容），这个式子恒 > 1，也就是
        // **重建比直接逐像素还慢**：白付一遍 1/stride² 的节点样本，而且因为
        // 阈值把噪声当平坦区插值，画质还更差。
        //
        // 实测（`640×360`，快速档，窄判据）：`j_geom`（位移网格 + 棋盘贴图）
        // 重建遍 735.5 ms，比同场景直接逐像素的 `一致` 档 520.8 ms 慢 1.41×。
        // 那不是"优化不够"，是拿额外一遍换零收益——任务书不允许把这种回退
        // 当交付，故这里按**当帧真实样本**算出平坦率，低于门槛就直接走单遍
        // 逐像素（与 golden 同流同值，画质只会更好）。
        //
        // 判据只用当帧自己的单元分类结果，不读场景名 / 不读历史帧 / 不按
        // 片源分流；同一套阈值对所有场景一致生效。
        let total_cells: usize = flat_cells.len().max(1);
        let nflat: usize = flat_cells.iter().filter(|&&v| v != 0).count();
        let flat_ratio: f64 = nflat as f64 / total_cells as f64;
        // 二级细化：每个非平坦单元补 5 个真实样本（中心 + 四边中点）。
        sampled_px = sampled_px
            .saturating_add((flat_cells.len().saturating_sub(nflat) as u64).saturating_mul(5));
        // 25% 是收益拐点：此时成本 ≈ 1/stride² + 75%，对步长 6 仍是 1.03×，
        // 再低就纯亏。取 25% 让"刚好没收益"的场景直接退到单遍。
        if flat_ratio < 0.25 {
            return render_frame_universal_direct_with_accel(
                scene, cam, ctx, resolution, params, aesthetic, threads, accel,
            );
        }
        // 重建诊断（离线标定用，`PRISM_V31_RECON_STATS=1` 打开）：
        // 输出三条比例——几何键全兼容、同容器兜底、最终判平坦。若「最终平坦」
        // 显著低于前两者之和，说明判据过严；若「几何键全兼容」本身很低，
        // 说明容差挡在高曲率几何上，需要动的是键容差而不是辐射判据。
        if std::env::var("PRISM_V31_RECON_STATS").ok().as_deref() == Some("1") {
            eprintln!(
                "[recon] 单元 {total_cells} / 几何兼容 {stat_geom} ({:.1}%) / 同容器兜底 {stat_smooth} ({:.1}%) / 最终平坦 {nflat} ({:.1}%)",
                100.0 * stat_geom as f64 / total_cells as f64,
                100.0 * stat_smooth as f64 / total_cells as f64,
                100.0 * flat_ratio
            );
        }
        // 注：这里曾试过「对平坦单元补 5 个 1 spp 探针（中心 + 四边中点），
        // 探针值与双线性预测比对不上就撤回平坦标记」。它确实有效——窄判据下
        // `k_rough` 从 31.83 dB 升到 58.73 dB、`p99` 0、`max` 102——但代价是
        // 每单元 5 次单像素渲染，640×360 上 +5.4 ms（`w2d` 快速档 26.5 → 31.9 ms，
        // 直接掉出 30.2 的门槛），而它对**速度**没有任何贡献，只在"已经放弃
        // 加速、改为逐像素"的配置下提升画质。收益与成本错配，故未启用。
        // 若将来要恢复，必须把探针点数降到 1~2 或改成"仅对兜底单元探测"。
        let mut x_frac: Vec<f32> = vec![0.0; wu];
        for gx in 0..wu {
            let cx: usize = x_cells[gx];
            let nx0: usize = xs[cx];
            let nx1: usize = xs[cx + 1];
            let denom: f32 = (nx1.saturating_sub(nx0)).max(1) as f32;
            x_frac[gx] = ((gx.saturating_sub(nx0)) as f32 / denom).clamp(0.0, 1.0);
        }
        let mut y_frac: Vec<f32> = vec![0.0; hu];
        for gy in 0..hu {
            let cy: usize = y_cells[gy];
            let ny0: usize = ys[cy];
            let ny1: usize = ys[cy + 1];
            let denom: f32 = (ny1.saturating_sub(ny0)).max(1) as f32;
            y_frac[gy] = ((gy.saturating_sub(ny0)) as f32 / denom).clamp(0.0, 1.0);
        }

        // 边缘单元逐像素真实求值的开关（离线 A/B 标定，非场景内容分流）。
        // 缺省打开：几何键不一致的边缘单元不糊、逐像素真实路径求值。
        let edge_exact: bool =
            std::env::var("PRISM_V31_UNIV_EDGE_EXACT").ok().as_deref() != Some("0");
        // 二级细化：非平坦单元再采「中心 + 四条边中点」5 个真实样本，把该单元
        // 细成 4 个 2×2 子块，各自双线性。这是第零节「自适应采样：平坦区域
        // 少采、复杂区域多采」的第二级——平坦单元 4 个角点就够，只有几何跨档
        // 或辐射陡变的单元才付这 5 份钱（= 逐像素成本的 5/16），而 2×2 子块的
        // 插值误差是 4×4 的约 1/4 量级。9 个样本的极差若仍超过 `spread2_tol`，
        // 说明这是真轮廓 / 真高频，退回逐像素，绝不糊。
        let spread2_tol: f32 = std::env::var("PRISM_V31_UNIV_SPREAD2_TOL")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or_else(|| recon_tol(&RECON_SPREAD2_TOL, 0.35));
        let mut refine: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; cell_w.saturating_mul(cell_h) * 5];
        {
            let fine_stride: usize = cell_w * 5;
            let mut r_rest: &mut [[f32; 3]] = &mut refine[..];
            let mut r_parts: Vec<(usize, usize, &mut [[f32; 3]])> = Vec::new();
            let fine_workers: usize = threads.clamp(1, cell_h.max(1));
            let fine_rows: usize = ((cell_h + fine_workers - 1) / fine_workers).max(1);
            let mut ry0: usize = 0;
            while ry0 < cell_h {
                let ry1: usize = (ry0 + fine_rows).min(cell_h);
                let (head, tail) = r_rest.split_at_mut((ry1 - ry0) * fine_stride);
                r_rest = tail;
                r_parts.push((ry0, ry1, head));
                ry0 = ry1;
            }
            let xs_ref2: &[usize] = &xs[..];
            let ys_ref2: &[usize] = &ys[..];
            let flat_ref2: &[u8] = &flat_cells[..];
            thread::scope(|scope| {
                for (a, b, part) in r_parts {
                    spawn_worker(scope, move || {
                        let _guard = UniversalIntersectGuard::new(scene, accel_ref);
                        for cy in a..b {
                            for cx in 0..cell_w {
                                if flat_ref2[cy * cell_w + cx] != 0 {
                                    continue;
                                }
                                let x0: usize = xs_ref2[cx];
                                let x1: usize = xs_ref2[cx + 1];
                                let y0: usize = ys_ref2[cy];
                                let y1: usize = ys_ref2[cy + 1];
                                let xm: usize = x0 + (x1 - x0) / 2;
                                let ym: usize = y0 + (y1 - y0) / 2;
                                let pts: [(usize, usize); 5] = [
                                    (xm, ym),
                                    (xm, y0),
                                    (xm, y1),
                                    (x0, ym),
                                    (x1, ym),
                                ];
                                let base: usize = (cy - a) * fine_stride + cx * 5;
                                for (k, (px, py)) in pts.iter().enumerate() {
                                    let rgba: [f32; 4] = render_pixel_range_universal(
                                        scene,
                                        cam,
                                        ctx,
                                        (*px as u32, *py as u32),
                                        resolution,
                                        params,
                                        accel_ref,
                                        0,
                                        params.spp,
                                    );
                                    part[base + k] = [rgba[0], rgba[1], rgba[2]];
                                }
                            }
                        }
                    });
                }
            });
        }
        let refine_ref: &[[f32; 3]] = &refine[..];
        let samples_ref: &[[f32; 3]] = &samples[..];
        let node_albedo_ref: &[[f32; 3]] = &node_albedo[..];
        let node_mapped_ref: &[u8] = &node_mapped[..];
        let keys_ref: &[u64] = &keys[..];
        let flat_ref: &[u8] = &flat_cells[..];
        let x_cells_ref: &[usize] = &x_cells[..];
        let y_cells_ref: &[usize] = &y_cells[..];
        let x_frac_ref: &[f32] = &x_frac[..];
        let y_frac_ref: &[f32] = &y_frac[..];
        let worker_cap: usize = threads.clamp(1, hu.max(1));
        let rows_per: usize = ((hu + worker_cap - 1) / worker_cap).max(1);
        let aesthetic_ref: &AestheticReq = aesthetic;
        let mut px_rest: &mut [u8] = &mut frame.pixels[..];
        let mut parts: Vec<(usize, usize, &mut [u8])> = Vec::new();
        // 第二遍的真实求值记账：worker 局部累加，收尾各做一次原子加。
        let traced_px_ref: &AtomicU64 = &traced_px;
        let traced_alb_ref: &AtomicU64 = &traced_alb;
        let mut y0: usize = 0;
        while y0 < hu {
            let y1: usize = (y0 + rows_per).min(hu);
            let len: usize = (y1 - y0) * wu * 4;
            let (head, tail) = px_rest.split_at_mut(len);
            px_rest = tail;
            parts.push((y0, y1, head));
            y0 = y1;
        }
        thread::scope(|scope| {
            for (a, b, dst_part) in parts {
                spawn_worker(scope, move || {
                    // 边缘单元逐像素真实求值复用本帧通用 BVH 求交视图；守卫在
                    // worker 线程作用域内挂接、析构恢复，不跨帧、不跨路径泄漏。
                    let _guard = UniversalIntersectGuard::new(scene, accel_ref);
                    let mut local_px: u64 = 0;
                    let mut local_alb: u64 = 0;
                    for ly in 0..(b - a) {
                        let gy: usize = a + ly;
                        let cy: usize = y_cells_ref[gy];
                        let fy: f32 = y_frac_ref[gy];
                        let row_base: usize = cy * gw;
                        for gx in 0..wu {
                            let cx: usize = x_cells_ref[gx];
                            let fx: f32 = x_frac_ref[gx];
                            let i00: usize = row_base + cx;
                            let i10: usize = i00 + 1;
                            let i01: usize = i00 + gw;
                            let i11: usize = i01 + 1;
                            let c00: [f32; 3] = samples_ref[i00];
                            let c10: [f32; 3] = samples_ref[i10];
                            let c01: [f32; 3] = samples_ref[i01];
                            let c11: [f32; 3] = samples_ref[i11];
                            let flat: bool = flat_ref[cy * cell_w + cx] != 0;
                            // 四角是否落在同一个几何容器里。`false` 就是真轮廓
                            // （球 / 背景、球 / 挡板），只有这类单元才需要逐像素
                            // 真实路径；同容器内的辐射陡变走二级细化插值。
                            let m00: bool = universal_key_is_miss(keys_ref[i00]);
                            let m10: bool = universal_key_is_miss(keys_ref[i10]);
                            let m01: bool = universal_key_is_miss(keys_ref[i01]);
                            let m11: bool = universal_key_is_miss(keys_ref[i11]);
                            // 逃逸族与命中族必须分开：`universal_shape_tag` 的
                            // 编号与逃逸键的容器段可能撞值，只看 shape 段会把
                            // "球 / 背景" 当成同容器，轮廓立刻被插值抹平。
                            let same_shape: bool = if m00 || m10 || m01 || m11 {
                                m00 && m10 && m01 && m11
                            } else {
                                (keys_ref[i00] >> 40) == (keys_ref[i10] >> 40)
                                    && (keys_ref[i00] >> 40) == (keys_ref[i01] >> 40)
                                    && (keys_ref[i00] >> 40) == (keys_ref[i11] >> 40)
                            };
                            let color: [f32; 3] = if flat {
                                let w00: f32 = (1.0 - fx) * (1.0 - fy);
                                let w10: f32 = fx * (1.0 - fy);
                                let w01: f32 = (1.0 - fx) * fy;
                                let w11: f32 = fx * fy;
                                let mut col: [f32; 3] = [
                                    c00[0] * w00 + c10[0] * w10 + c01[0] * w01 + c11[0] * w11,
                                    c00[1] * w00 + c10[1] * w10 + c01[1] * w01 + c11[1] * w11,
                                    c00[2] * w00 + c10[2] * w10 + c01[2] * w01 + c11[2] * w11,
                                ];
                                // ---- 纹理分离校正 ----
                                //
                                // 四角都挂了贴图时，上面的双线性把 **albedo 也一起
                                // 平均了**，棋盘 / 条纹这类高频图案会被抹平——那是
                                // 第零节五禁止的"糊一下"，判据① 会直接爆到满量程。
                                //
                                // 漫反射项里 albedo 与光照是可分离的：`L = albedo ⊙ E/π`。
                                // 于是插值结果≈`albedo_interp ⊙ E/π`，乘上
                                // `albedo_px / albedo_interp` 就把像素自己的高频 albedo
                                // 还原回去，而 E 仍是"两个节点算 16 spp 换来"的平滑量。
                                // 代价只是一条主射线取 albedo，不是逐像素跑满 spp。
                                //
                                // `mapped == false` 的单元完全不进这段（无贴图场景零开销、
                                // 输出逐字节不变），故对域内两片源没有任何数值影响。
                                if node_mapped_ref[i00] != 0
                                    && node_mapped_ref[i10] != 0
                                    && node_mapped_ref[i01] != 0
                                    && node_mapped_ref[i11] != 0
                                {
                                    let (alb_px, ok) = pixel_base_color(
                                        scene,
                                        cam,
                                        ctx,
                                        (gx as u32, gy as u32),
                                        resolution,
                                    );
                                    local_alb = local_alb.saturating_add(1);
                                    if ok {
                                        let a00: [f32; 3] = node_albedo_ref[i00];
                                        let a10: [f32; 3] = node_albedo_ref[i10];
                                        let a01: [f32; 3] = node_albedo_ref[i01];
                                        let a11: [f32; 3] = node_albedo_ref[i11];
                                        for c in 0..3 {
                                            let a_int: f32 = a00[c] * w00
                                                + a10[c] * w10
                                                + a01[c] * w01
                                                + a11[c] * w11;
                                            // 分母过小（albedo 接近全黑）时比值无意义，
                                            // 保持插值原值，避免除零放大成满量程离群。
                                            if a_int > 1.0e-4 {
                                                col[c] *= alb_px[c] / a_int;
                                            }
                                        }
                                    }
                                }
                                col
                            } else if edge_exact && same_shape {
                                // 同容器、辐射陡变：用二级细化的 2×2 子块双线性
                                // （本帧 9 个真实样本，无近似、不读历史帧）。
                                // 9 样本极差仍超宽松上限才退回逐像素。
                                let r: usize = (cy * cell_w + cx) * 5;
                                let sc: [f32; 3] = refine_ref[r];
                                let st: [f32; 3] = refine_ref[r + 1];
                                let sb: [f32; 3] = refine_ref[r + 2];
                                let sl: [f32; 3] = refine_ref[r + 3];
                                let sr: [f32; 3] = refine_ref[r + 4];
                                let mut spread9: f32 = 0.0;
                                let pts9: [[f32; 3]; 9] =
                                    [c00, c10, c01, c11, sc, st, sb, sl, sr];
                                for c in 0..3 {
                                    let mut lo: f32 = pts9[0][c];
                                    let mut hi: f32 = pts9[0][c];
                                    for p in &pts9 {
                                        lo = lo.min(p[c]);
                                        hi = hi.max(p[c]);
                                    }
                                    spread9 = spread9.max(hi - lo);
                                }
                                if spread9 <= spread2_tol {
                                    let (a00, a10, a01, a11, fx2, fy2): (
                                        [f32; 3],
                                        [f32; 3],
                                        [f32; 3],
                                        [f32; 3],
                                        f32,
                                        f32,
                                    ) = if fx < 0.5 {
                                        if fy < 0.5 {
                                            (c00, st, sl, sc, fx * 2.0, fy * 2.0)
                                        } else {
                                            (sl, sc, c01, sb, fx * 2.0, fy * 2.0 - 1.0)
                                        }
                                    } else if fy < 0.5 {
                                        (st, c10, sc, sr, fx * 2.0 - 1.0, fy * 2.0)
                                    } else {
                                        (sc, sr, sb, c11, fx * 2.0 - 1.0, fy * 2.0 - 1.0)
                                    };
                                    let w00: f32 = (1.0 - fx2) * (1.0 - fy2);
                                    let w10: f32 = fx2 * (1.0 - fy2);
                                    let w01: f32 = (1.0 - fx2) * fy2;
                                    let w11: f32 = fx2 * fy2;
                                    [
                                        a00[0] * w00 + a10[0] * w10 + a01[0] * w01 + a11[0] * w11,
                                        a00[1] * w00 + a10[1] * w10 + a01[1] * w01 + a11[1] * w11,
                                        a00[2] * w00 + a10[2] * w10 + a01[2] * w01 + a11[2] * w11,
                                    ]
                                } else {
                                    // 轮廓 / 材质接缝 / 遮挡边界：逐像素走当前帧
                                    // 确定性真实路径样本，与独立通用路径同口径，
                                    // 不取最近角点、不糊、不读历史帧。
                                    local_px = local_px.saturating_add(1);
                                    let rgba: [f32; 4] = render_pixel_range_universal(
                                        scene,
                                        cam,
                                        ctx,
                                        (gx as u32, gy as u32),
                                        resolution,
                                        params,
                                        accel_ref,
                                        0,
                                        params.spp,
                                    );
                                    [rgba[0], rgba[1], rgba[2]]
                                }
                            } else if edge_exact {
                                // 跨容器（球 / 背景、球 / 挡板、毛发 / 空气）：
                                // 真轮廓，必须逐像素走当前帧真实路径样本。
                                local_px = local_px.saturating_add(1);
                                let rgba: [f32; 4] = render_pixel_range_universal(
                                    scene,
                                    cam,
                                    ctx,
                                    (gx as u32, gy as u32),
                                    resolution,
                                    params,
                                    accel_ref,
                                    0,
                                    params.spp,
                                );
                                [rgba[0], rgba[1], rgba[2]]
                            } else if fx < 0.5 {
                                if fy < 0.5 {
                                    c00
                                } else {
                                    c01
                                }
                            } else if fy < 0.5 {
                                c10
                            } else {
                                c11
                            };
                            // 就地色调映射 + 写字节：与 `Framebuffer::set_pixel`
                            // 逐字节同式（同一个 `apply_aesthetic`、同一个
                            // `(v.clamp(0,1)*255).round() as u8`、alpha 恒 255），
                            // 只是省掉合法坐标上的边界检查与一次中间缓冲往返。
                            let mapped: prism_math::Vec3 = prism_gpu::apply_aesthetic(
                                prism_math::Vec3::new(color[0], color[1], color[2]),
                                aesthetic_ref,
                            );
                            let base: usize = ly * wu * 4 + gx * 4;
                            dst_part[base] = (mapped.x.clamp(0.0, 1.0) * 255.0).round() as u8;
                            dst_part[base + 1] =
                                (mapped.y.clamp(0.0, 1.0) * 255.0).round() as u8;
                            dst_part[base + 2] =
                                (mapped.z.clamp(0.0, 1.0) * 255.0).round() as u8;
                            dst_part[base + 3] = 255;
                        }
                    }
                    // 收尾各一次原子加：热循环里只碰线程局部变量。
                    traced_px_ref.fetch_add(local_px, Ordering::Relaxed);
                    traced_alb_ref.fetch_add(local_alb, Ordering::Relaxed);
                });
            }
        });
    }
    let second_us: u64 = second_started
        .map(|t| t.elapsed().as_micros() as u64)
        .unwrap_or(0);
    // 账本收口：节点 + 二级细化 + 逐像素兜底 = 本帧真实求值的像素条目数。
    // 除以 `宽 × 高` 就是「平均每像素真实跑了几次路径积分」，乘以 `spp` 是
    // 真实样本数。这一段是报数的唯一来源，不读任何"应该采多少"的先验。
    FRAME_SAMPLED_PIXELS.store(
        sampled_px.saturating_add(traced_px.load(Ordering::Relaxed)),
        Ordering::Relaxed,
    );
    FRAME_EXTRA_RAYS.store(
        extra_rays.saturating_add(traced_alb.load(Ordering::Relaxed)),
        Ordering::Relaxed,
    );
    // 色调映射已并入第二遍的工作线程（就地写字节），不再有独立的串行尾巴。
    let tone_us: u64 = 0;
    if profile_enabled {
        eprintln!(
            "[v31-tile-profile] 准备 {:.3} ms / 首遍 {:.3} ms / 次遍 {:.3} ms / 色调映射 {:.3} ms / 合计 {:.3} ms",
            setup_us as f64 / 1000.0,
            first_us as f64 / 1000.0,
            second_us as f64 / 1000.0,
            tone_us as f64 / 1000.0,
            (setup_us + first_us + second_us + tone_us) as f64 / 1000.0,
        );
    }
    frame
}

// ---------------------------------------------------------------------------
// 测试（W0-C L3 关卡：tile 化必须对确定性恒等）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gpu::apply_aesthetic;
    use prism_math::Vec3;
    use prism_rt::bsdf::Bsdf;
    use prism_rt::integrator::{Light, MaterialKind, Sphere};

    /// 测试场景：单球 + 单方向光（与 `rt_backend` 的 `red_sphere_request` 同构）。
    fn tiny_scene() -> (Scene, Camera, EvalCtx, RenderParams, AestheticReq) {
        let mut scene: Scene = Scene::default();
        scene.spheres.push((
            Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
            MaterialKind::Pbr(Bsdf::new(
                Vec3::new(0.8, 0.2, 0.2),
                0.05,
                0.35,
                1.5,
                0.0,
                0.0,
                Vec3::splat(0.3),
            )),
        ));
        scene.lights.push(Light::Directional {
            direction: Vec3::new(0.0, 0.0, -1.0),
            radiance: Vec3::splat(3.0),
            cos_theta_max: 1.0,
        });
        let cam: Camera = Camera::new(
            Vec3::new(0.0, 0.0, -4.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            0.0,
            3.0,
            0.0,
            0.5,
        );
        let res: (u32, u32) = (61, 37); // 刻意取非 tile 整数倍：边缘 tile 收窄
        let _ = res;
        (
            scene,
            cam,
            EvalCtx::new(0, 0, 0),
            RenderParams::new(3, 4),
            AestheticReq { teal_orange: true, contrast: 1.12, filmic: true },
        )
    }

    /// **关卡 1**：tile 划分只依赖帧尺寸——32×32，且覆盖整帧无重叠。
    ///
    /// 这条钉的是 D-20 的第一条准入条件：tile 尺寸**不得**随核数变化。
    /// `grid()` 的签名里根本没有线程数，所以"不会随核数变"是类型层面的，
    /// 这里再把数值与覆盖性钉一遍。
    #[test]
    fn tile_grid_is_fixed_and_covers_frame() {
        let g: TileGrid = grid(1920, 1080);
        assert_eq!((g.tile_w, g.tile_h), (RT_TILE_W, RT_TILE_H));
        assert_eq!((RT_TILE_W, RT_TILE_H), (32, 32), "tile 尺寸是被钉死的常量");
        assert_eq!((g.cols(), g.rows()), (60, 34));

        // 覆盖性：每个像素恰好被一个 tile 覆盖一次（用计数板验证，不靠眼算）。
        let (w, h): (u32, u32) = (g.width, g.height);
        let mut seen: Vec<u8> = vec![0; (w * h) as usize];
        for t in g.tiles() {
            for y in t.y..t.y + t.h {
                for x in t.x..t.x + t.w {
                    let i: usize = (y * w + x) as usize;
                    seen[i] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "存在漏覆盖或重复覆盖的像素");
    }

    /// **关卡 2**：1 / 4 / 16 线程输出**逐字节一致**。
    #[test]
    fn thread_counts_are_byte_identical() {
        let (scene, cam, ctx, params, aes) = tiny_scene();
        let res: (u32, u32) = (61, 37);
        let reference: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, 1);
        assert!(reference.is_valid());
        for threads in [2usize, 4, 5, 16] {
            let got: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, threads);
            assert_eq!(
                got.pixels, reference.pixels,
                "{threads} 线程与 1 线程不逐字节一致（D-20 破了）"
            );
        }
    }

    /// **关卡 3**：同核数连跑两次逐字节一致（防"完成顺序进了结果"）。
    #[test]
    fn same_thread_count_twice_is_byte_identical() {
        let (scene, cam, ctx, params, aes) = tiny_scene();
        let res: (u32, u32) = (61, 37);
        let a: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, 4);
        let b: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, 4);
        assert_eq!(a.pixels, b.pixels, "同核数两跑不逐字节一致");
    }

    /// **关卡 4（最强）**：tile 化输出 == **改之前的逐像素循环**输出。
    ///
    /// 这里把 W0-C 之前的实现（两层 `for` + `render_pixel` + `set_pixel`）
    /// **原样写在测试里**当参考实现，再与 tile 版比字节。参考实现不依赖任何
    /// 被测代码路径，所以它挡的是"tile 化顺手改了渲染语义"这一类回归。
    #[test]
    fn tiled_matches_pre_tiling_reference_loop() {
        let (scene, cam, ctx, params, aes) = tiny_scene();
        let (w, h): (u32, u32) = (61, 37);

        // —— 改之前的实现（逐字保留循环形状）——
        let mut reference: Framebuffer = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let rgba: [f32; 4] = render_pixel(&scene, &cam, &ctx, (x, y), (w, h), &params);
                let linear: Vec3 = Vec3::new(rgba[0], rgba[1], rgba[2]);
                let mapped: Vec3 = apply_aesthetic(linear, &aes);
                reference.set_pixel(x, y, mapped);
            }
        }

        for threads in [1usize, 4, 16] {
            let got: Framebuffer = render_frame(&scene, &cam, &ctx, (w, h), &params, &aes, threads);
            assert_eq!(
                got.pixels, reference.pixels,
                "{threads} 线程输出与 tile 化之前的单线程参考实现不逐字节一致"
            );
        }
    }

    /// 线程数不会改变 tile 网格（间接：`grid` 只吃帧尺寸）。
    #[test]
    fn worker_count_is_clamped_not_used_for_partitioning() {
        let (scene, cam, ctx, params, aes) = tiny_scene();
        let res: (u32, u32) = (8, 8); // 1 个 tile
        // 请求 64 线程但只有 1 个 tile：不得 panic、不得越界。
        let a: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, 64);
        let b: Framebuffer = render_frame(&scene, &cam, &ctx, res, &params, &aes, 1);
        assert!(a.is_valid());
        assert_eq!(a.pixels, b.pixels);
    }

    /// 尺寸被拒时返回拒绝缓冲（不 panic、不空转）。
    #[test]
    fn oversize_frame_is_refused() {
        let (scene, cam, ctx, params, aes) = tiny_scene();
        let bad: u32 = prism_gpu::MAX_FRAMEBUFFER_DIM + 1;
        let fb: Framebuffer = render_frame(&scene, &cam, &ctx, (bad, 4), &params, &aes, 4);
        assert!(!fb.is_valid(), "超限尺寸必须返回拒绝缓冲");
    }

    /// 后端默认线程数取 `available_parallelism`（`0` 是哨兵，不是"零线程"）。
    #[test]
    fn zero_threads_means_auto() {
        let backend: crate::RtBackend = crate::RtBackend::default();
        assert_eq!(backend.threads, 0, "默认必须是哨兵值 0");
        assert_eq!(backend.effective_threads(), auto_threads());
        assert!(backend.effective_threads() >= 1);
        assert_eq!(crate::RtBackend::default().with_threads(3).effective_threads(), 3);
    }
}
