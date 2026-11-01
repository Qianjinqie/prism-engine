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

use std::thread;

use prism_core::EvalCtx;
use prism_gpu::{AestheticReq, Framebuffer};
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, RenderParams, Scene};

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
