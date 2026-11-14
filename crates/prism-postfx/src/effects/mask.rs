//! 遮罩 / ROTO 族（C1）：形状遮罩 / alpha 遮罩 / 深度遮罩 / 通道遮罩 +
//! 羽化 / 反相 / 布尔组合。
//!
//! 语义权威：遮罩族：`PathMask`（矢量蒙版，复用
//! prism-raster）、`LumaKey`/`ChromaKey`（亮度/色度抠像）、Roto（手绘路径 +
//! 羽化 + 参数随时间 = Expr 驱动的动态遮罩）；（"PathMask：
//! 矢量蒙版复用光栅（羽化 = 覆盖率梯度展宽）；Roto 动态 = 路径参数接属性
//! 通道"；"LumaKey/ChromaKey：在 YCbCr/HSV 特征域做阈值 + 软过渡（不是 RGB
//! 硬键），spill suppression 内建"）。
//!
//! # 设计
//!
//! 全部遮罩统一产出**单通道掩版**（`Vec<f32>` ∈ [0,1] 逐像素），再经
//! [`MaskOp`] 布尔 / 一元算子组合成最终遮版，最后 [`apply_mask`] 把遮版逐像素
//! 乘进 RGBA。遮版运算全部逐像素单遍、固定表达式序——层序/组合序决定论。
//! 作为 [`TileEffect`] 时 halo = 0（纯逐像素）。

use prism_gpu::path_raster::{PathFillRule as GpuFillRule, RasterPoint};
use prism_io::fbuf::Fbuf16F;
use prism_raster::geom::Point;
use prism_raster::scanline::{rasterize_into, FillRule};
use prism_raster::Coverage;
use prism_render::tile::TileEffect;
use std::cell::RefCell;

/// 2D 矢量路径遮罩的 **GPU 光栅能力开关：默认点亮（`true`）**。
///
/// 与 `qa_curve_parity.rs::VULKAN_CURVE_PATH = true` 同形态：这是常亮态，不是
/// 「登记态 / 默认关闭 / 待真机」。`true` ⇒ [`PathMask::rasterize`] 与逐帧出片
/// 路径 [`PathMaskEffect::eval_window`] 先走 `prism_gpu::rasterize_path_gpu`
/// （Vulkan compute / WebGPU WGSL 真 dispatch）；无设备或单帧派发失败返回
/// `None`，**透明回退** `prism-raster` 的 f64 几何权威扫描线。两端输出经
/// `tests/qa_path_raster_gpu_parity.rs` 的 P2 判据证明等价（≤2e-6、u8 一致）。
/// 本常量有测试断言恒为 `true`，杜绝被翻成关闭态。
pub const PATH_MASK_GPU_PATH: bool = true;

/// `prism_raster::geom::Point` → `prism_gpu::path_raster::RasterPoint`（POD 桥接）。
///
/// 两个结构体都是 `{ x: f32, y: f32 }`，显式逐字段转换，保持 `prism-gpu` 不反向
/// 依赖 `prism-raster`（依赖 DAG：`postfx → {gpu, raster}`）。
#[inline]
fn to_raster_points(points: &[Point]) -> Vec<RasterPoint> {
    points.iter().map(|p| RasterPoint::new(p.x, p.y)).collect()
}

/// 逐帧遮罩复用池：CPU 兜底覆盖率缓冲 + feather 展宽 scratch。
///
/// [`PathMaskEffect::eval_window`]（真实逐帧出片路径）复用这两块缓冲：遮版数据
/// 不逃逸（`apply_mask_values` 直接读池内切片），池容量跨帧保留，消除每帧重新
/// 分配全分辨率 f32 输出并触发缺页（星 512@4K 旧实现约 43% 墙钟耗在此）。
struct MaskPool {
    cov: Coverage,
    scratch: Vec<f32>,
}

thread_local! {
    static MASK_POOL: RefCell<MaskPool> = const {
        RefCell::new(MaskPool {
            cov: Coverage { width: 0, height: 0, data: Vec::new() },
            scratch: Vec::new(),
        })
    };
}

/// 遮版：单通道逐像素覆盖（[0,1]，行主序）。
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 逐像素覆盖。
    pub data: Vec<f32>,
}

impl Mask {
    /// 构造全零遮版。
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0.0; (width as usize) * (height as usize)] }
    }

    /// 读一个像素（越界返回 0）。
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// 写一个像素。
    ///
    /// # Panics
    /// 坐标越界时 panic。
    pub fn set(&mut self, x: u32, y: u32, v: f32) {
        assert!(x < self.width && y < self.height, "遮版坐标越界 ({x},{y})");
        self.data[(y as usize) * (self.width as usize) + (x as usize)] = v;
    }
}

/// 遮罩布尔 / 一元算子（布尔组合）。
///
/// 全部逐像素运算：`Union`/`Intersect`/`Subtract` 为集合算子（min/max/饱和减），
/// `Invert` 反相，`Add`/`Multiply` 软组合。两操作数尺寸必须一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// 并集（max）。
    Union,
    /// 交集（min）。
    Intersect,
    /// 差集 a − b（饱和减）。
    Subtract,
    /// 和 a + b（饱和加）。
    Add,
    /// 积 a·b。
    Multiply,
    /// 反相 1 − a（一元算子，忽略右操作数）。
    Invert,
}

/// 按算子组合两个遮版。
///
/// # Panics
/// 尺寸不一致时 panic。
pub fn combine(op: MaskOp, a: &Mask, b: &Mask) -> Mask {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "遮罩布尔组合要求尺寸一致"
    );
    let mut out = Mask::new(a.width, a.height);
    for i in 0..a.data.len() {
        let va = a.data[i];
        let vb = b.data[i];
        out.data[i] = match op {
            MaskOp::Union => va.max(vb),
            MaskOp::Intersect => va.min(vb),
            MaskOp::Subtract => (va - vb).max(0.0),
            MaskOp::Add => (va + vb).min(1.0),
            MaskOp::Multiply => va * vb,
            MaskOp::Invert => 1.0 - va,
        };
    }
    out
}

/// 羽化：对遮版做半径 `r` 的 box 展宽（覆盖率梯度展宽）。
///
/// 逐像素取 `(2r+1)²` 邻域均值，边缘钳制复制——与 blur 同约定（tile 无关性：
/// halo = r）。固定序归约（行主序 tap）。
pub fn feather(mask: &Mask, r: u32) -> Mask {
    if r == 0 {
        return mask.clone();
    }
    let mut data = mask.data.clone();
    let mut scratch: Vec<f32> = Vec::new();
    feather_inplace(&mut data, mask.width, mask.height, r, &mut scratch);
    Mask { width: mask.width, height: mask.height, data }
}

/// 就地羽化：`data` 读、`scratch` 写，末了两缓冲交换（复用容量）。
///
/// 算术与循环序与旧 [`feather`] 逐句一致（`mask.get` 的边界钳制等价
/// `data` 行主序索引；边缘钳制已把 `sx/sy` 收进 `[0,w-1]/[0,h-1]`），故结果
/// **逐字节一致**；`scratch` 作为复用 scratch 供逐帧路径消除每帧分配。
fn feather_inplace(data: &mut Vec<f32>, w: u32, h: u32, r: u32, scratch: &mut Vec<f32>) {
    if r == 0 || w == 0 || h == 0 {
        return;
    }
    let n = (w as usize) * (h as usize);
    scratch.clear();
    scratch.resize(n, 0.0);
    let taps = ((2 * r + 1) as f32) * ((2 * r + 1) as f32);
    let wu = w as usize;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for dy in 0..=(2 * r) {
                let sy = (y + dy).saturating_sub(r).min(h - 1);
                for dx in 0..=(2 * r) {
                    let sx = (x + dx).saturating_sub(r).min(w - 1);
                    acc += data[(sy as usize) * wu + (sx as usize)];
                }
            }
            scratch[(y as usize) * wu + (x as usize)] = acc / taps;
        }
    }
    core::mem::swap(data, scratch);
}

/// 把遮版乘进 RGBA 缓冲（逐像素；halo = 0）。
pub fn apply_mask(fb: &Fbuf16F, mask: &Mask) -> Fbuf16F {
    apply_mask_values(fb, mask.width, mask.height, &mask.data)
}

/// 逐像素把 `data`（行主序，`mw×mh`）乘进 RGBA；与 [`apply_mask`] 逐句同式。
///
/// 单独暴露「读切片」形式，使逐帧路径可直接消费复用池里的遮版切片，
/// 无需克隆出拥有所有权的 `Mask`（那是复用池失效的根源）。
fn apply_mask_values(fb: &Fbuf16F, mw: u32, mh: u32, data: &[f32]) -> Fbuf16F {
    let mut out = Fbuf16F::new(fb.width, fb.height);
    for y in 0..fb.height {
        for x in 0..fb.width {
            let c = fb.get(x, y);
            let m = if x < mw && y < mh {
                data.get((y as usize) * (mw as usize) + (x as usize)).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            out.set(x, y, [c[0] * m, c[1] * m, c[2] * m, c[3] * m]);
        }
    }
    out
}

/// 形状遮罩（PathMask）：矢量多边形 → 覆盖遮版（复用 prism-raster 扫描线）。
///
/// 参数 `points` 为设备空间闭合折线顶点（y 向下）；`feather_radius` 羽化半径。
#[derive(Debug, Clone, PartialEq)]
pub struct PathMask {
    /// 设备空间顶点（闭合折线）。
    pub points: Vec<Point>,
    /// 羽化半径（像素；0 = 无羽化）。
    pub feather_radius: u32,
    /// 是否反相。
    pub invert: bool,
}

impl PathMask {
    /// 以顶点构造。
    pub fn new(points: Vec<Point>) -> Self {
        Self { points, feather_radius: 0, invert: false }
    }

    /// 光栅化为遮版（**GPU 优先、CPU 兜底**）。
    ///
    /// 点亮 [`PATH_MASK_GPU_PATH`] 时先走 `prism_gpu::rasterize_path_gpu`（真机
    /// dispatch → fence → readback）；返回 `None` 时透明回退 `prism-raster` 的
    /// f64 几何权威扫描线。两种出帧与 f64 标尺的等价性由
    /// `tests/qa_path_raster_gpu_parity.rs` 的 P2 判据钉死。羽化 / 反相语义不变。
    pub fn rasterize(&self, width: u32, height: u32) -> Mask {
        let mut cov = Coverage::new(0, 0);
        let mut scratch: Vec<f32> = Vec::new();
        self.rasterize_core(width, height, &mut cov, &mut scratch);
        Mask { width: cov.width, height: cov.height, data: cov.data }
    }

    /// 核心：把覆盖率写进 `out`（GPU 优先、CPU 兜底），再就地羽化 / 反相。
    ///
    /// `out` / `scratch` 由调用方提供，便于逐帧路径用 [`MASK_POOL`] 复用容量。
    fn rasterize_core(
        &self,
        width: u32,
        height: u32,
        out: &mut Coverage,
        scratch: &mut Vec<f32>,
    ) {
        // GPU 优先：能力开关常亮；无设备/单帧失败返回 None，透明回退。
        let gpu = if PATH_MASK_GPU_PATH {
            let contours = vec![to_raster_points(&self.points)];
            prism_gpu::rasterize_path_gpu(
                &contours,
                GpuFillRule::Winding,
                width,
                height,
            )
        } else {
            None
        };
        if let Some(g) = gpu {
            out.width = g.width;
            out.height = g.height;
            out.data = g.data; // 直接接管 GPU readback，零拷贝
        } else {
            // CPU 兜底：f64 几何权威扫描线；rasterize_into 复用 out.data 已驻留页。
            let contours: Vec<Vec<Point>> = vec![self.points.clone()];
            rasterize_into(&contours, FillRule::Winding, width, height, out);
        }
        if self.feather_radius > 0 {
            feather_inplace(&mut out.data, out.width, out.height, self.feather_radius, scratch);
        }
        if self.invert {
            for v in out.data.iter_mut() {
                *v = 1.0 - *v;
            }
        }
    }

    /// 逐帧出片：复用池内缓冲（不逃逸），返回乘好遮版的 RGBA。
    fn eval_window_pooled(&self, window: &Fbuf16F) -> Fbuf16F {
        MASK_POOL.with(|cell| {
            // 主线程正常复用；重入（同线程嵌套）退回本地临时池，结果不变。
            if let Ok(mut pool) = cell.try_borrow_mut() {
                let MaskPool { cov, scratch } = &mut *pool;
                self.rasterize_core(window.width, window.height, cov, scratch);
                apply_mask_values(window, cov.width, cov.height, &cov.data)
            } else {
                let mut tmp = MaskPool {
                    cov: Coverage::new(0, 0),
                    scratch: Vec::new(),
                };
                self.rasterize_core(window.width, window.height, &mut tmp.cov, &mut tmp.scratch);
                apply_mask_values(window, tmp.cov.width, tmp.cov.height, &tmp.cov.data)
            }
        })
    }
}

/// 亮度抠像（LumaKey）：按 Rec.709 亮度软阈值（特征域软过渡）。
///
/// 亮度 `L = 0.2126R + 0.7152G + 0.0722B`（BT.709 常量，固定）；
/// 阈值带 `[low, high]` 内线性软过渡（不是硬键）：`L < low → 0`、
/// `L > high → 1`、带内 `(L − low)/(high − low)`。`invert` 反相。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaKey {
    /// 阈值下限（遮版从 0 起）。
    pub low: f32,
    /// 阈值上限（遮版到 1）。
    pub high: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for LumaKey {
    fn default() -> Self {
        Self { low: 0.1, high: 0.5, invert: false }
    }
}

impl LumaKey {
    /// 从 RGBA 缓冲产出遮版。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let mut m = Mask::new(fb.width, fb.height);
        let span = (self.high - self.low).max(1.0e-6);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                let mut v = prism_math::clamp((l - self.low) / span, 0.0, 1.0);
                if self.invert {
                    v = 1.0 - v;
                }
                m.set(x, y, v);
            }
        }
        m
    }
}

/// 色度抠像（ChromaKey）：YCbCr 色度距离软键 + green spill suppression。
///
/// 键色 `key`（RGB）；逐像素转 YCbCr（BT.601 固定系数），按**色度距离**
/// 软阈值（`tolerance` 内 0、`softness` 带内线性过渡）——仅色度参与，
/// 亮度不敏感（特征域而非 RGB 硬键）。`spill` ∈ [0,1] 为
/// 溢出抑制强度：把像素中超出键绿色通道的绿色部分按强度削平。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaKey {
    /// 键色（RGB，通常绿幕/蓝幕）。
    pub key: [f32; 3],
    /// 色度距离软阈值下限（≤ 此距离完全键出）。
    pub tolerance: f32,
    /// 软过渡带宽（tolerance … tolerance+softness 线性过渡到 1）。
    pub softness: f32,
    /// 溢出抑制强度（0 = 关，1 = 全）。
    pub spill: f32,
}

impl Default for ChromaKey {
    fn default() -> Self {
        Self { key: [0.0, 1.0, 0.0], tolerance: 0.2, softness: 0.3, spill: 1.0 }
    }
}

/// RGB → BT.601 YCbCr（固定系数，禁 libm）。
fn rgb_to_ycbcr(c: [f32; 3]) -> (f32, f32, f32) {
    let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    let cb = (c[2] - y) * 0.564;
    let cr = (c[0] - y) * 0.713;
    (y, cb, cr)
}

impl ChromaKey {
    /// 从 RGBA 缓冲产出遮版（并另返回去溢出的颜色处理在 [`Self::process`]）。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let (_, kb, kr) = rgb_to_ycbcr(self.key);
        let mut m = Mask::new(fb.width, fb.height);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let (_, cb, cr) = rgb_to_ycbcr([c[0], c[1], c[2]]);
                let d = ((cb - kb) * (cb - kb) + (cr - kr) * (cr - kr)).sqrt();
                // 软过渡：d ≤ tolerance → 0（键出）；d ≥ tolerance+softness → 1（保留）。
                let hi = (self.tolerance + self.softness).max(self.tolerance + 1.0e-6);
                let v = prism_math::clamp((d - self.tolerance) / (hi - self.tolerance), 0.0, 1.0);
                m.set(x, y, v);
            }
        }
        m
    }

    /// 溢出抑制 + 遮版应用：键绿色通道过量的部分按强度削平，再乘遮版。
    ///
    /// spill suppression（内建）：对键色主通道（取键色最大分量
    /// 所在通道）做 `c = min(c, max(other two channels) + (1−spill)·excess)`
    /// ——绿色边缘过量绿被削到两邻通道水平。
    pub fn process(&self, fb: &Fbuf16F) -> Fbuf16F {
        let m = self.mask(fb);
        // 键色主通道（0=R 1=G 2=B）。
        let ch = if self.key[0] >= self.key[1] && self.key[0] >= self.key[2] {
            0
        } else if self.key[1] >= self.key[2] {
            1
        } else {
            2
        };
        let others: [usize; 2] = match ch {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let mut out = Fbuf16F::new(fb.width, fb.height);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let mut despi = c;
                let neighbor_max = c[others[0]].max(c[others[1]]);
                if c[ch] > neighbor_max {
                    let excess = c[ch] - neighbor_max;
                    despi[ch] = neighbor_max + excess * (1.0 - prism_math::clamp(self.spill, 0.0, 1.0));
                }
                let alpha = m.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        despi[0] * alpha,
                        despi[1] * alpha,
                        despi[2] * alpha,
                        c[3] * alpha,
                    ],
                );
            }
        }
        out
    }
}

/// 通道遮罩（ChannelMask）：取 RGBA / 深度某通道作遮版（软过渡可选）。
///
/// `channel` ∈ {0,1,2,3}（R/G/B/A）；`low`/`high` 软阈值带同 LumaKey。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMask {
    /// 通道索引（0=R 1=G 2=B 3=A）。
    pub channel: u8,
    /// 阈值下限。
    pub low: f32,
    /// 阈值上限。
    pub high: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for ChannelMask {
    fn default() -> Self {
        Self { channel: 3, low: 0.5, high: 0.5, invert: false }
    }
}

impl ChannelMask {
    /// 从 RGBA 缓冲产出遮版。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let ch = (self.channel.min(3)) as usize;
        let mut m = Mask::new(fb.width, fb.height);
        let span = (self.high - self.low).max(1.0e-6);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let mut v = prism_math::clamp((c[ch] - self.low) / span, 0.0, 1.0);
                if self.invert {
                    v = 1.0 - v;
                }
                m.set(x, y, v);
            }
        }
        m
    }
}

/// 遮罩效果（TileEffect 适配）：`PathMask` 光栅 + 羽化/反相（halo = 0，
/// 光栅在窗口局部坐标——整幅求值即规范语义，同 postfx 位置键控约定）。
#[derive(Debug, Clone, PartialEq)]
pub struct PathMaskEffect {
    /// 路径遮罩。
    pub mask: PathMask,
}

impl TileEffect for PathMaskEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        // 逐帧全分辨率遮罩：走 GPU 优先 / CPU 兜底核心，并复用池内缓冲，
        // 消除每帧重新分配输出缓冲的页错误。
        self.mask.eval_window_pooled(window)
    }
}

/// LumaKey 效果（TileEffect 适配，halo = 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaKeyEffect {
    /// 抠像参数。
    pub key: LumaKey,
}

impl TileEffect for LumaKeyEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let m = self.key.mask(window);
        apply_mask(window, &m)
    }
}

/// ChromaKey 效果（TileEffect 适配，halo = 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaKeyEffect {
    /// 抠像参数。
    pub key: ChromaKey,
}

impl TileEffect for ChromaKeyEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        self.key.process(window)
    }
}

/// 深度遮罩（DepthMask）：按深度 AOV 软阈值产出遮版（深度图喂色彩/遮罩，
/// "按距离调色"的地基）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthMask {
    /// 近阈值（AOV 小 = 近）。
    pub near: f32,
    /// 远阈值。
    pub far: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for DepthMask {
    fn default() -> Self {
        Self { near: 0.0, far: 1.0, invert: false }
    }
}

impl DepthMask {
    /// 从深度 AOV（逐像素 [0,1]）产出遮版。
    pub fn mask(&self, aov: &[f32], width: u32, height: u32) -> Mask {
        let mut m = Mask::new(width, height);
        let span = (self.far - self.near).max(1.0e-6);
        for (i, &d) in aov.iter().enumerate() {
            let mut v = prism_math::clamp((d - self.near) / span, 0.0, 1.0);
            if self.invert {
                v = 1.0 - v;
            }
            m.data[i] = v;
        }
        m
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, c);
            }
        }
        fb
    }

    fn mask_uniform(w: u32, h: u32, v: f32) -> Mask {
        let mut m = Mask::new(w, h);
        for i in 0..m.data.len() {
            m.data[i] = v;
        }
        m
    }

    // ---- 布尔运算断言 ----

    #[test]
    fn mask_boolean_union_intersect_subtract() {
        let a = mask_uniform(1, 1, 0.3);
        let b = mask_uniform(1, 1, 0.7);
        assert_eq!(combine(MaskOp::Union, &a, &b).get(0, 0), 0.7, "并 = max");
        assert_eq!(combine(MaskOp::Intersect, &a, &b).get(0, 0), 0.3, "交 = min");
        let sub = combine(MaskOp::Subtract, &a, &b).get(0, 0);
        assert_eq!(sub, 0.0, "差 a−b（饱和减，0.3−0.7<0 → 0）");
        let sub2 = combine(MaskOp::Subtract, &b, &a).get(0, 0);
        assert!((sub2 - 0.4).abs() < 1e-6, "差 b−a = 0.4");
    }

    #[test]
    fn mask_boolean_add_multiply_invert() {
        let a = mask_uniform(1, 1, 0.3);
        let b = mask_uniform(1, 1, 0.7);
        assert!((combine(MaskOp::Add, &a, &b).get(0, 0) - 1.0).abs() < 1e-6, "和饱和到 1");
        assert!((combine(MaskOp::Multiply, &a, &b).get(0, 0) - 0.21).abs() < 1e-6, "积");
        assert!((combine(MaskOp::Invert, &a, &a).get(0, 0) - 0.7).abs() < 1e-6, "反相 1−a");
    }

    #[test]
    fn mask_boolean_is_deterministic_and_pure() {
        let a = mask_uniform(4, 3, 0.4);
        let b = mask_uniform(4, 3, 0.6);
        let x = combine(MaskOp::Union, &a, &b);
        let y = combine(MaskOp::Union, &a, &b);
        assert_eq!(x.data, y.data, "同输入同输出（纯函数）");
    }

    // ---- 形状遮罩 ----

    #[test]
    fn path_mask_covers_polygon_interior() {
        // 10×10 帧内一个 4..8 的正方形 → 内部覆盖 1、外部 0。
        let pts = vec![
            Point::new(4.0, 4.0),
            Point::new(8.0, 4.0),
            Point::new(8.0, 8.0),
            Point::new(4.0, 8.0),
        ];
        let mask = PathMask::new(pts).rasterize(10, 10);
        // 中心在内部。
        assert!(mask.get(6, 6) > 0.9, "多边形内部应覆盖：{}", mask.get(6, 6));
        // 角落在外。
        assert!(mask.get(0, 0) < 0.1, "多边形外部应空：{}", mask.get(0, 0));
    }

    #[test]
    fn path_mask_invert_flips_coverage() {
        let pts = vec![
            Point::new(4.0, 4.0),
            Point::new(8.0, 4.0),
            Point::new(8.0, 8.0),
            Point::new(4.0, 8.0),
        ];
        let mut pm = PathMask::new(pts);
        pm.invert = true;
        let mask = pm.rasterize(10, 10);
        assert!(mask.get(6, 6) < 0.1, "反相后内部应空：{}", mask.get(6, 6));
        assert!(mask.get(0, 0) > 0.9, "反相后外部应满：{}", mask.get(0, 0));
    }

    #[test]
    fn feather_widens_edge_gradient() {
        // 硬边遮版（左半 1 右半 0）经羽化后边界带出现中间值。
        let mut m = Mask::new(9, 1);
        for x in 0..9 {
            m.set(x, 0, if x < 4 { 1.0 } else { 0.0 });
        }
        let f = feather(&m, 2);
        let edge = f.get(4, 0);
        assert!(edge > 0.0 && edge < 1.0, "羽化后边界应为中间值：{edge}");
        // 远处不受影响。
        assert_eq!(f.get(0, 0), 1.0, "远离边界处保持 1");
        assert_eq!(f.get(8, 0), 0.0, "远离边界处保持 0");
    }

    // ---- 亮度 / 色度抠像 ----

    #[test]
    fn luma_key_soft_threshold() {
        // 全灰 0.5，阈值带 [0.4, 0.6] → 覆盖 0.5。
        let fb = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let k = LumaKey { low: 0.4, high: 0.6, invert: false };
        let m = k.mask(&fb);
        assert!((m.get(0, 0) - 0.5).abs() < 1e-5, "软过渡中点：{}", m.get(0, 0));
        // 亮部 → 1，暗部 → 0。
        let bright = solid(1, 1, [1.0, 1.0, 1.0, 1.0]);
        assert!((k.mask(&bright).get(0, 0) - 1.0).abs() < 1e-5);
        let dark = solid(1, 1, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.mask(&dark).get(0, 0), 0.0);
    }

    #[test]
    fn chroma_key_removes_green_bgc_keeps_red() {
        let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.2, softness: 0.3, spill: 1.0 };
        // 纯绿 → 键出（遮版 0）。
        let green = solid(1, 1, [0.0, 1.0, 0.0, 1.0]);
        assert!(key.mask(&green).get(0, 0) < 0.05, "纯绿应被键出");
        // 纯红 → 保留（遮版 1）。
        let red = solid(1, 1, [1.0, 0.0, 0.0, 1.0]);
        assert!(key.mask(&red).get(0, 0) > 0.95, "纯红应保留");
    }

    #[test]
    fn chroma_key_spill_suppression_reduces_excess_green() {
        // 近绿但偏亮（R=0.4 G=0.8 B=0.4）→ spill 抑制后绿通道被削。
        let fb = solid(1, 1, [0.4, 0.8, 0.4, 1.0]);
        let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 1.0 };
        let out = key.process(&fb);
        let c = out.get(0, 0);
        // 溢出抑制：绿被削到邻通道水平（勿超 red 太多）。
        assert!(c[1] <= 0.4 + 1e-5, "溢出绿应被抑制到邻通道水平：{c:?}");
        // 无 spill 时绿保留。
        let key0 = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 0.0 };
        let out0 = key0.process(&fb);
        assert!(out0.get(0, 0)[1] > c[1], "spill=0 时绿应更大：{c:?} vs {:?}", out0.get(0, 0));
    }

    // ---- 通道 / 深度遮罩 ----

    #[test]
    fn channel_mask_reads_selected_channel() {
        let fb = solid(1, 1, [0.2, 0.4, 0.6, 0.8]);
        let cm = ChannelMask { channel: 2, low: 0.0, high: 1.0, invert: false };
        // f16 存储 → 比较容差 1e-3。
        assert!((cm.mask(&fb).get(0, 0) - 0.6).abs() < 1e-3, "通道 2（B）= 0.6");
        let ca = ChannelMask { channel: 3, low: 0.0, high: 1.0, invert: false };
        assert!((ca.mask(&fb).get(0, 0) - 0.8).abs() < 1e-3, "通道 3（A）= 0.8");
    }

    #[test]
    fn depth_mask_maps_distance_to_coverage() {
        let aov = vec![0.0, 0.5, 1.0];
        let dm = DepthMask { near: 0.0, far: 1.0, invert: false };
        let m = dm.mask(&aov, 3, 1);
        assert_eq!(m.get(0, 0), 0.0);
        assert!((m.get(1, 0) - 0.5).abs() < 1e-6);
        assert_eq!(m.get(2, 0), 1.0);
        // 反相。
        let dmi = DepthMask { near: 0.0, far: 1.0, invert: true };
        assert_eq!(dmi.mask(&aov, 3, 1).get(0, 0), 1.0);
    }

    // ---- 效果适配 ----

    #[test]
    fn path_mask_effect_has_zero_halo_and_applies() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(3.0, 3.0),
            Point::new(0.0, 3.0),
        ];
        let eff = PathMaskEffect { mask: PathMask::new(pts) };
        assert_eq!(eff.halo_radius(), 0, "遮罩效果 halo = 0");
        let fb = solid(3, 3, [1.0, 1.0, 1.0, 1.0]);
        let out = eff.eval_window(&fb);
        // 全在多边形内 → 全保留。
        assert!(out.get(1, 1)[0] > 0.9, "内部应保留：{:?}", out.get(1, 1));
    }

    #[test]
    fn apply_mask_multiplies_rgba() {
        let fb = solid(2, 1, [1.0, 0.5, 0.25, 1.0]);
        let m = mask_uniform(2, 1, 0.5);
        let out = apply_mask(&fb, &m);
        let c = out.get(0, 0);
        assert!((c[0] - 0.5).abs() < 1e-3 && (c[3] - 0.5).abs() < 1e-3, "遮版乘进 RGBA：{c:?}");
    }

    // ---- GPU 优先 / CPU 兜底真接线 + 复用池 ----

    fn poly(w: f32, h: f32) -> Vec<Point> {
        vec![
            Point::new(1.5, 1.5),
            Point::new(w - 2.5, 2.5),
            Point::new(w - 1.5, h - 2.5),
            Point::new(2.5, h - 1.5),
        ]
    }

    #[test]
    fn path_mask_gpu_path_is_lit_by_default() {
        assert!(
            PATH_MASK_GPU_PATH,
            "2D 矢量路径遮罩的 GPU 能力开关必须恒为点亮态 true（不是登记态/默认关闭）"
        );
    }

    /// `feather_inplace` 必须与旧 `feather` 逐字节一致（复用池不改变结果）。
    #[test]
    fn feather_inplace_is_byte_identical_to_feather() {
        let (w, h) = (11u32, 9u32);
        let mut m = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if x >= 3 && x <= 7 { 1.0 } else { 0.0 };
                m.set(x, y, v * (0.5 + 0.5 * ((x + y) as f32 / 20.0)));
            }
        }
        for r in 1..=3u32 {
            let expected = feather(&m, r);
            let mut data = m.data.clone();
            let mut scratch: Vec<f32> = Vec::new();
            feather_inplace(&mut data, w, h, r, &mut scratch);
            assert_eq!(data.len(), expected.data.len(), "r={r}");
            for (i, (a, b)) in expected.data.iter().zip(data.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "feather_inplace r={r} @{i} 逐字节一致");
            }
        }
    }

    /// 逐帧路径（复用池 `eval_window_pooled`）必须与「rasterize + apply_mask」
    /// 的规范路径逐字节一致：GPU/CPU 两端、羽化/反相组合、反复调用都成立。
    #[test]
    fn pooled_eval_window_matches_canonical_path_bytewise() {
        let cases: Vec<(Vec<Point>, u32, bool, u32, u32)> = vec![
            (poly(20.0, 16.0), 0, false, 20, 16),
            (poly(20.0, 16.0), 2, false, 20, 16),
            (poly(20.0, 16.0), 1, true, 20, 16),
            (vec![
                Point::new(2.0, 2.0), Point::new(15.0, 3.0), Point::new(9.0, 13.0),
            ], 0, false, 17, 15),
        ];
        for (pts, feather_radius, invert, w, h) in cases {
            let pm = PathMask { points: pts, feather_radius, invert };
            let eff = PathMaskEffect { mask: pm.clone() };
            let fb = solid(w, h, [0.8, 0.6, 0.4, 1.0]);
            let canonical = apply_mask(&fb, &pm.rasterize(w, h));
            // 反复调用 eval_window，验证复用池跨帧不留残影、结果稳定。
            for _ in 0..3 {
                let pooled = eff.eval_window(&fb);
                for y in 0..h {
                    for x in 0..w {
                        let a = canonical.get(x, y);
                        let b = pooled.get(x, y);
                        for c in 0..4 {
                            assert_eq!(
                                a[c].to_bits(), b[c].to_bits(),
                                "复用池路径与规范路径逐字节一致 @({x},{y})[{c}]"
                            );
                        }
                    }
                }
            }
        }
    }

    /// 有设备时逐帧路径确实走 GPU：`rasterize_path_gpu` 返回 Some，且遮罩结果
    /// 与 GPU 帧一致（反假绿：证明不是「CPU 兜底蒙混」）。无设备时自门控 skip。
    #[test]
    fn pooled_path_uses_gpu_when_device_present() {
        if !prism_gpu::vulkan::VulkanBackend::probe().device_created {
            println!("[mask-gpu] 无 Vulkan 设备：按既有范式自门控 skip");
            return;
        }
        let pts = poly(24.0, 18.0);
        let (w, h) = (24u32, 18u32);
        let contours = vec![to_raster_points(&pts)];
        let gpu = prism_gpu::rasterize_path_gpu(&contours, GpuFillRule::Winding, w, h)
            .expect("有设备时 GPU 2D 路径入口必须返回 Some");
        assert_eq!(gpu.data.len(), (w * h) as usize);

        // 无羽化/反相时，rasterize 结果应逐字节等于 GPU 帧。
        let pm = PathMask { points: pts, feather_radius: 0, invert: false };
        let mask = pm.rasterize(w, h);
        assert_eq!(mask.data.len(), gpu.data.len());
        for (i, (a, b)) in mask.data.iter().zip(gpu.data.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "PathMask::rasterize 应等于 GPU 帧 @{i}");
        }
        println!("[mask-gpu] 有设备：PathMask 逐帧路径走 GPU，{w}x{h} 逐字节一致");
    }
}
