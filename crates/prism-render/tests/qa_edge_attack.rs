//! QA 边界反例攻击（prism-render 侧）。
//!
//! 判定方式：`render_frame_full` 与 `render_frame_tiled` 的输出**逐位比较**
//! （`Vec<u16>` 位模式直接相等——强于 MD5 指纹，MD5 仅用于报告可读性）。
//! 效果链用两个 BoxBlur3（总 halo = 2，覆盖多效果半径累加路径）。

use prism_core::SampleIndex;
use prism_render::{BoxBlur3, PixelSource, TileGrid, TileOrchestrator};

/// 确定性伪随机像素源：整数 xorshift 派生 f32（零浮点库依赖），值域含
/// >1.0 与极小值，制造 f16 舍入与非正规机会。帧图案由 `si.frame` 混入
/// （随 TileOrchestrator.frame 变化），同时**就地验证 D-28**：
/// SampleIndex.pixel 必须等于全局像素坐标。
struct PatternSource;

impl PixelSource for PatternSource {
    fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4] {
        // 关卡 4 的直接验证：tile 划分不得改变 SampleIndex 的全局坐标域。
        assert_eq!(si.pixel, (x, y), "SampleIndex.pixel 必须是全局坐标（D-28）");
        let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ si.frame.wrapping_mul(31)
            ^ si.node_id.wrapping_mul(0x27D4_EB2F_1656_67C5);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        let ch = |k: u64| ((h >> k) & 0xFF) as f32 / 255.0;
        [
            ch(40) * 1.7,        // 超 1.0：触发 f16 正规舍入
            ch(32) * 3.0e-8,     // 非正规域（< 2⁻²⁴）
            ch(24) / 3.0,        // 无限小数：f16 舍入敏感
            ch(16),
        ]
    }
}

/// 判定：整幅 vs tile 化输出位模式逐位相等。
fn assert_full_tiled_identical(grid: TileGrid, frame: u64, label: &str) {
    let full_orch = TileOrchestrator { frame, ..TileOrchestrator::new(grid) };
    let tiled_orch = TileOrchestrator { frame, ..TileOrchestrator::new(grid) };
    let source = PatternSource;
    let blur_a = BoxBlur3;
    let blur_b = BoxBlur3;
    let full = full_orch.render_frame_full(&source, &[&blur_a, &blur_b]);
    let tiled = tiled_orch.render_frame_tiled(&source, &[&blur_a, &blur_b]);
    assert_eq!(
        full.pixels, tiled.pixels,
        "{label}：整幅与 tile 化输出位模式不一致（full {} px, tiled {} px）",
        full.pixels.len(),
        tiled.pixels.len()
    );
    assert_eq!((full.width, full.height), (grid.width, grid.height), "{label}：输出尺寸");
}

/// C-⑤：1×1 单像素帧——tile 化的最小退化情形（halo 窗口完全饱和钳制）。
#[test]
fn qa_tile_single_pixel_frame() {
    for (tw, th) in [(1_u32, 1_u32), (4, 4), (64, 64)] {
        let grid = TileGrid::new(1, 1, tw, th);
        assert_full_tiled_identical(grid, 7, &format!("1×1 帧 tile {tw}×{th}"));
    }
}

/// C-⑤：1×1 格——每 tile 恰一个像素（16×9 帧 → 144 tile），halo 窗口被
/// 帧界 saturating 钳到 [0,0,3,3] 的极端路径全覆盖。
#[test]
fn qa_tile_grid_one_by_one_cells() {
    let grid = TileGrid::new(16, 9, 1, 1);
    assert_eq!(grid.cols() * grid.rows(), 144);
    assert_full_tiled_identical(grid, 3, "16×9 帧 1×1 格");
}

/// C-⑤：超大格——tile 尺寸远超帧尺寸（单 tile 覆盖全帧，边缘收窄路径）。
#[test]
fn qa_tile_oversized_cell() {
    for (w, h) in [(33_u32, 17_u32), (1, 17), (33, 1)] {
        let grid = TileGrid::new(w, h, 4096, 4096);
        assert_full_tiled_identical(grid, 1, &format!("超大格 {w}×{h}"));
    }
}

/// C-⑤：超扁 / 超高帧（一行/一列 tile 的极端纵横比）。
#[test]
fn qa_tile_extreme_aspect_ratios() {
    let grid_flat = TileGrid::new(64, 2, 8, 1);
    assert_full_tiled_identical(grid_flat, 5, "超扁 64×2（tile 8×1）");
    let grid_tall = TileGrid::new(2, 64, 1, 8);
    assert_full_tiled_identical(grid_tall, 5, "超高 2×64（tile 1×8）");
}

/// C-⑤：非整除边缘 tile 收窄 + 无效果（halo=0 的裸源路径）。
#[test]
fn qa_tile_non_divisible_edges_and_no_effects() {
    let grid = TileGrid::counts(33, 17, 8, 6);
    let source = PatternSource;
    let orch = TileOrchestrator::new(grid);
    let full = orch.render_frame_full(&source, &[]);
    let tiled = orch.render_frame_tiled(&source, &[]);
    assert_eq!(full.pixels, tiled.pixels, "非整除边缘 + 零效果链：逐位一致");
}

/// C-⑤：同帧重渲确定性（同 grid 同 frame 两次 tiled 渲染逐位一致）。
#[test]
fn qa_tile_render_determinism_same_inputs() {
    let grid = TileGrid::counts(40, 24, 16, 9);
    let a = TileOrchestrator::new(grid).render_frame_tiled(&PatternSource, &[&BoxBlur3]);
    let b = TileOrchestrator::new(grid).render_frame_tiled(&PatternSource, &[&BoxBlur3]);
    assert_eq!(a.pixels, b.pixels, "同输入两次渲染应逐位一致（纯函数契约）");
}
