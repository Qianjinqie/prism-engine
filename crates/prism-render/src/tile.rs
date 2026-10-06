//! Tile 网格与 halo 协议（A5）。
//!
//! 语义权威：《强大版路线图》§4 A5 行（TileGrid + 全帧效果 halo 协议，含
//! 最小 halo 测试效果：3×3 box blur）与 §2.1（tile 渲染：帧 → tile 网格，
//! 每 tile 独立调度求值，内存峰值与输出分辨率解耦）；《架构总设计》v2.4
//! §4.3（tile 无关性三规则：① per-tile 效果原地求值 ② 大半径全帧效果走
//! halo 扩展协议 ③ 跨 tile 与整幅渲染逐字节一致）；《算法白皮书》§26
//! （tile 级是天然并行安全区——**禁止跨 tile 浮点归约**，本模块全部求值
//! 均为像素级局部运算，无任何跨 tile 合并）。
//!
//! # halo 协议
//! 全帧效果（blur/bloom/DOF 等需要邻域像素的效果）声明邻域半径
//! [`TileEffect::halo_radius`]；编排器（orchestrator.rs）对每个 tile 按半径
//! 外扩采样窗口（钳制到帧边界，不环绕）、效果在窗口上求值、编排器只保留
//! tile 核心区回写——保证跨 tile 结果与整幅渲染**逐字节一致**（验收关卡
//! 1/2）。效果若申报半径不足，tile 边缘将发生窗口钳制读，与整幅产生可检出
//! 的边界差异——这是协议的反向验证面。

use prism_io::fbuf::Fbuf16F;

/// tile：帧内的一个矩形求值单元（**全局像素坐标域**）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// tile 左上角的全局像素 x。
    pub x: u32,
    /// tile 左上角的全局像素 y。
    pub y: u32,
    /// tile 宽（像素；边缘 tile 自动收窄）。
    pub w: u32,
    /// tile 高（像素；边缘 tile 自动收窄）。
    pub h: u32,
}

impl Tile {
    /// 外扩 `radius` 像素的采样窗口（钳制到帧边界——帧外无数据，不环绕）。
    pub fn expanded(&self, radius: u32, frame_w: u32, frame_h: u32) -> Tile {
        let x: u32 = self.x.saturating_sub(radius);
        let y: u32 = self.y.saturating_sub(radius);
        let right: u32 = self
            .x
            .saturating_add(self.w)
            .saturating_add(radius)
            .min(frame_w);
        let bottom: u32 = self
            .y
            .saturating_add(self.h)
            .saturating_add(radius)
            .min(frame_h);
        Tile { x, y, w: right - x, h: bottom - y }
    }
}

/// tile 网格：帧 → tile 划分（tile 尺寸可参数化，路线图 §2.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    /// 帧宽（像素）。
    pub width: u32,
    /// 帧高（像素）。
    pub height: u32,
    /// tile 名义宽（像素）。
    pub tile_w: u32,
    /// tile 名义高（像素）。
    pub tile_h: u32,
}

impl TileGrid {
    /// 指定 tile 尺寸构造。
    ///
    /// # Panics
    /// tile 尺寸为 0 时 panic（防呆：无意义的划分，宪章 C21）。
    pub fn new(width: u32, height: u32, tile_w: u32, tile_h: u32) -> Self {
        assert!(tile_w > 0 && tile_h > 0, "tile 尺寸必须为正");
        TileGrid { width, height, tile_w, tile_h }
    }

    /// 按列/行数上取整均分构造（帧尺寸恰被整除时，列/行数精确等于
    /// `cols`/`rows`——验收关卡 1 的 16×9 网格用此构造，如 192×108 →
    /// 12×12 tile × 16 列 × 9 行 = 144 tile）。
    ///
    /// # Panics
    /// 列/行数为 0 时 panic。
    pub fn counts(width: u32, height: u32, cols: u32, rows: u32) -> Self {
        assert!(cols > 0 && rows > 0, "网格列/行数必须为正");
        Self::new(width, height, width.div_ceil(cols), height.div_ceil(rows))
    }

    /// 列数。
    pub fn cols(&self) -> u32 {
        self.width.div_ceil(self.tile_w)
    }

    /// 行数。
    pub fn rows(&self) -> u32 {
        self.height.div_ceil(self.tile_h)
    }

    /// 取第 (col, row) 个 tile；边缘 tile 自动收窄（帧尺寸不必整除）。
    ///
    /// # Panics
    /// (col, row) 越出网格时 panic（调用方为内部编排器，编程错误即暴露）。
    pub fn tile(&self, col: u32, row: u32) -> Tile {
        let x: u32 = col * self.tile_w;
        let y: u32 = row * self.tile_h;
        assert!(x < self.width && y < self.height, "tile 索引越出网格");
        Tile {
            x,
            y,
            w: self.tile_w.min(self.width - x),
            h: self.tile_h.min(self.height - y),
        }
    }

    /// 全部 tile，**行主序固定遍历序**（拼接回写的确定性序；格序特例，
    /// 白皮书 §26 工程四律之"消序用排序"）。
    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        let cols: u32 = self.cols();
        let rows: u32 = self.rows();
        (0..rows).flat_map(move |r| (0..cols).map(move |c| self.tile(c, r)))
    }
}

/// 全帧效果的 tile 化求值协议（架构 §4.3 tile 无关性规则②；白皮书 §26：
/// 像素级局部运算，无跨 tile 归约）。
///
/// 效果 = 「输入窗口 → 同尺寸输出窗口」的逐像素纯函数：输出像素 p 只读
/// 输入窗口内 `p ± halo_radius()` 的邻域，窗口边缘越界读按边缘钳制复制。
/// 钳制读数与整幅求值一致的充要条件：被采样的输出像素距窗口边缘
/// ≥ `halo_radius()`，或窗口边缘恰为帧边界——编排器按申报半径扩采保证
/// 前者，帧边界处两者天然重合（tile 边界与整幅逐字节一致的正确性条件）。
pub trait TileEffect {
    /// 邻域半径（像素）。0 = per-tile 原地效果（架构 §4.3 规则①）。
    fn halo_radius(&self) -> u32;

    /// 对窗口求值：输入/输出同尺寸（RGBA16F 位模式缓冲；f32 运算域，
    /// 写回时量化 f16）。
    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F;
}

/// 3×3 box blur——最小 halo 测试效果（路线图 §4 A5 行点名，供 Phase 1
/// 验收 2 的 halo 用例；`halo_radius = 1`）。
///
/// 9 tap 按行主序（左上 → 右下）固定顺序求和、÷ 9——像素内固定序归约
/// （白皮书 §26），与 tile 划分无关；f32 域运算、写回量化 f16（架构 §4.3
/// 格锚：半浮点存储、运算位模式精确往返）。
pub struct BoxBlur3;

impl TileEffect for BoxBlur3 {
    fn halo_radius(&self) -> u32 {
        1
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                // 9 tap 固定顺序求和（行主序），坐标按窗口边缘钳制。
                let mut acc = [0.0f32; 4];
                for dy in 0..3u32 {
                    for dx in 0..3u32 {
                        let sx: u32 = (x + dx).saturating_sub(1).min(window.width - 1);
                        let sy: u32 = (y + dy).saturating_sub(1).min(window.height - 1);
                        let px: [f32; 4] = window.get(sx, sy);
                        for (a, p) in acc.iter_mut().zip(px.iter()) {
                            *a += *p;
                        }
                    }
                }
                out.set(x, y, [acc[0] / 9.0, acc[1] / 9.0, acc[2] / 9.0, acc[3] / 9.0]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试：网格几何 / halo 窗口 / 最小 halo 测试效果
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_counts_16x9_on_divisible_frame() {
        // 验收关卡 1 的划分：192×108 ÷ 16×9 → 12×12 tile，恰 144 块。
        let grid = TileGrid::counts(192, 108, 16, 9);
        assert_eq!((grid.tile_w, grid.tile_h), (12, 12));
        assert_eq!((grid.cols(), grid.rows()), (16, 9));
        assert_eq!(grid.tiles().count(), 144);
        assert_eq!(grid.tile(0, 0), Tile { x: 0, y: 0, w: 12, h: 12 });
        assert_eq!(grid.tile(15, 8), Tile { x: 180, y: 96, w: 12, h: 12 });
    }

    #[test]
    fn grid_edge_tiles_narrow_on_indivisible_frame() {
        // 帧尺寸不整除：tile(100×70, 16×9) → 7 列 8 行，末列/末行收窄。
        let grid = TileGrid::new(100, 70, 16, 9);
        assert_eq!((grid.cols(), grid.rows()), (7, 8));
        assert_eq!(grid.tile(6, 7), Tile { x: 96, y: 63, w: 4, h: 7 });
        // 全 tile 无缝覆盖整帧（面积和 = 帧面积）。
        let area: u64 = grid.tiles().map(|t| t.w as u64 * t.h as u64).sum();
        assert_eq!(area, 100 * 70);
    }

    #[test]
    fn tile_expanded_clamps_to_frame() {
        let frame = (192, 108);
        // 内部 tile：四向外扩 1。
        assert_eq!(
            Tile { x: 12, y: 12, w: 12, h: 12 }.expanded(1, frame.0, frame.1),
            Tile { x: 11, y: 11, w: 14, h: 14 }
        );
        // 角 tile：左上钳到 0。
        assert_eq!(
            Tile { x: 0, y: 0, w: 12, h: 12 }.expanded(3, frame.0, frame.1),
            Tile { x: 0, y: 0, w: 15, h: 15 }
        );
        // 右下角 tile：右/下钳到帧边界。
        assert_eq!(
            Tile { x: 180, y: 96, w: 12, h: 12 }.expanded(4, frame.0, frame.1),
            Tile { x: 176, y: 92, w: 16, h: 16 }
        );
        // 半径 0 = 原窗口。
        assert_eq!(
            Tile { x: 5, y: 6, w: 7, h: 8 }.expanded(0, frame.0, frame.1),
            Tile { x: 5, y: 6, w: 7, h: 8 }
        );
    }

    #[test]
    fn box_blur3_uniform_field_is_fixed_point() {
        // 常量场的 box blur 是不动点（能量守恒的平凡特例）。
        let mut win = Fbuf16F::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let out = BoxBlur3.eval_window(&win);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(x, y), [0.5, 0.5, 0.5, 1.0]);
            }
        }
    }

    /// 测试期望的 f16 量化（实现写回时量化 f16，期望值须过同一域）。
    fn q(v: f32) -> f32 {
        prism_io::f16_to_f32(prism_io::f32_to_f16(v))
    }

    #[test]
    fn box_blur3_delta_impulse_spreads_by_ninth() {
        // δ 冲激：中心单点亮 → 3×3 邻域各得 1/9，界外为零。
        let mut win = Fbuf16F::new(5, 5);
        win.set(2, 2, [1.0, 0.0, 0.0, 0.0]);
        let out = BoxBlur3.eval_window(&win);

        // 精确值：1/9 = 0.1111…，写回经 f16 量化（固定序归约 → 可精确断言）。
        let ninth: [f32; 4] = out.get(2, 2);
        assert_eq!(ninth[0], q(1.0 / 9.0), "中心像素 = 9 tap 之和 ÷ 9");
        assert_eq!(out.get(3, 2)[0], q(1.0 / 9.0), "水平邻点");
        assert_eq!(out.get(1, 1)[0], q(1.0 / 9.0), "对角邻点");
        assert_eq!(out.get(0, 0)[0], 0.0, "冲激不在 (0,0) 的 3×3 邻域内");
        assert_eq!(out.get(4, 4)[0], 0.0, "冲激不在 (4,4) 的 3×3 邻域内");
        // G/B/A 通道未受冲激污染（A 通道全零输入 → 全零输出）。
        assert_eq!(ninth[1], 0.0);
        assert_eq!(ninth[3], 0.0);
    }

    #[test]
    fn box_blur3_clamps_at_window_edge() {
        // 窗口边缘钳制复制：5×1 单行窗口——sy 恒钳为 0（每行 tap 纵向重复
        // 3 次），x=0 时 xs = {0,0,1}（横向也钳制）。
        let mut win = Fbuf16F::new(5, 1);
        for x in 0..5u32 {
            win.set(x, 0, [x as f32, 0.0, 0.0, 0.0]);
        }
        let out = BoxBlur3.eval_window(&win);
        // x=0：每行 taps (0,0,1) 和 = 1，三行合计 3 → 3/9。
        assert_eq!(out.get(0, 0)[0], q(3.0 / 9.0));
        // x=2：每行 taps (1,2,3) 和 = 6，三行 18 → 18/9 = 2。
        assert_eq!(out.get(2, 0)[0], q(18.0 / 9.0));
        // x=4：每行 taps (3,4,4) 和 = 11，三行 33 → 33/9。
        assert_eq!(out.get(4, 0)[0], q(33.0 / 9.0));
    }

    #[test]
    fn box_blur3_is_pure() {
        // 纯函数契约：同输入两次求值逐字节一致（无隐藏状态）。
        let mut win = Fbuf16F::new(6, 4);
        let mut state: u64 = 0xB0B;
        for y in 0..4u32 {
            for x in 0..6u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v: f32 = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let a = BoxBlur3.eval_window(&win);
        let b = BoxBlur3.eval_window(&win);
        assert_eq!(a, b, "同输入必同输出（求值纯函数）");
    }
}
