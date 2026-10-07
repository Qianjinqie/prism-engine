//! 点精灵渲染（B8；对接 [`prism_render::tile`] tile 协议；粒子→像素覆盖，RGBA16F）。
//!
//! # 数学结构（D-22）
//!
//! 点精灵渲染 = 屏幕空间核 **加性 splat**：每粒子在其像素投影处落一个
//! 径向衰减核（`w(r) = 1 − r²`，紧支撑圆盘——比高斯更省且确定），多粒子在
//! 同一像素处 **over 合成**（前到后按 `spawn_index` 升序，确定性序）。
//!
//! # 对接 tile 协议（规则②）
//!
//! 粒子是**局部效果**但半径可超单像素，故每粒子贡献的 splat 半径 = 申报
//! [`PointSpriteEffect::halo_radius`]；编排器按半径外扩窗口，本效果在窗口上
//! 求值、只保留核心区回写——保证跨 tile 与整幅**逐字节一致**（
//! 像素级局部运算，无跨 tile 归约；本实现逐 tile 独立，粒子集合按窗口裁剪，
//! 无跨 tile 合并）。
//!
//! # 透明语义
//!
//! 输出 = 输入窗口上叠加粒子（加色 splat，alpha 累加至 1）。空粒子集时输出
//! 与输入逐位相同（恒等，halo 协议的空集恒等性）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use crate::emitter::ParticleView;

/// 点精灵渲染参数（像素域）。
#[derive(Debug, Clone, PartialEq)]
pub struct PointSpriteEffect {
    /// 帧宽（像素）。
    pub frame_w: u32,
    /// 帧高（像素）。
    pub frame_h: u32,
    /// 世界→像素缩放（每世界单位像素数）。
    pub scale: f32,
    /// 世界原点（对应像素 (0,0)）。
    pub origin_px: [f32; 2],
    /// 每世界单位尺寸 → 像素半径的系数。
    pub size_scale: f32,
    /// 全局强度系数（乘到颜色）。
    pub intensity: f32,
    /// splat 最大半径（像素；用于 halo 申报与求值裁剪）。
    pub max_radius: u32,
    /// 已投影的粒子（世界坐标位置 + 像素半径 + 颜色）。
    pub sprites: Vec<Sprite>,
}

/// 一个已投影的点精灵（世界位置、像素半径、颜色）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sprite {
    /// 世界坐标位置。
    pub world: [f32; 3],
    /// 像素半径（≥ 0）。
    pub radius_px: f32,
    /// 颜色（RGBA，线性）。
    pub color: [f32; 4],
}

impl PointSpriteEffect {
    /// 从粒子投影构造（世界坐标 → 像素；尺寸 = 粒子尺寸 × `size_scale`）。
    pub fn from_views(views: &[ParticleView], scale: f32, size_scale: f32, intensity: f32) -> Self {
        Self::from_views_origin(views, scale, size_scale, intensity, [0.0, 0.0], 0, 0)
    }

    /// 带画面尺寸与世界原点的构造（用于 halo 半径估算）。
    pub fn from_views_origin(
        views: &[ParticleView],
        scale: f32,
        size_scale: f32,
        intensity: f32,
        origin_px: [f32; 2],
        frame_w: u32,
        frame_h: u32,
    ) -> Self {
        let mut sprites = Vec::with_capacity(views.len());
        let mut max_radius: u32 = 1;
        for v in views {
            let r = (v.size * size_scale).max(0.0);
            let rp = r.ceil() as u32;
            if rp > max_radius {
                max_radius = rp;
            }
            sprites.push(Sprite {
                world: [v.position.x, v.position.y, v.position.z],
                radius_px: r,
                color: v.color,
            });
        }
        Self {
            frame_w,
            frame_h,
            scale,
            origin_px,
            size_scale,
            intensity,
            max_radius,
            sprites,
        }
    }

    /// 世界坐标 → 像素坐标（x = origin_px.x + world.x·scale；y 轴向下翻转）。
    #[inline]
    fn project(&self, world: [f32; 3]) -> [f32; 2] {
        [
            self.origin_px[0] + world[0] * self.scale,
            self.origin_px[1] - world[1] * self.scale,
        ]
    }
}

impl TileEffect for PointSpriteEffect {
    /// 邻域半径 = 最大 splat 半径（像素）——编排器按此外扩窗口。
    fn halo_radius(&self) -> u32 {
        self.max_radius
    }

    /// 对窗口求值：输入窗口逐像素叠加粒子 splat（RGBA16F；f32 运算，f16 量化）。
    ///
    /// **坐标约定**：窗口坐标为窗口局部坐标域（`window.get(0,0)` = 窗口左上）。
    /// 调用方构造效果时把 `origin_px` 设为**窗口左上在全局像素域的位置**，
    /// 于是 `project()` 直接给出窗口局部坐标（编排器对每个 tile 用该 tile 的
    /// 全局左上重建效果——与 halo 协议的外扩窗口一致）。粒子按窗口包围盒裁剪，
    /// 窗口外粒子不影响结果（局部性；无跨 tile 归约）。
    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let (w, h) = (window.width, window.height);
        let mut out = window.clone();
        if self.sprites.is_empty() {
            return out; // 空集恒等（halo 协议空集恒等性）。
        }
        for sp in &self.sprites {
            let gp = self.project(sp.world);
            let px = gp[0];
            let py = gp[1];
            let r = sp.radius_px;
            if r <= 0.0 {
                continue;
            }
            let r2 = r * r;
            let x0: i64 = (px - r).floor() as i64;
            let x1: i64 = (px + r).ceil() as i64;
            let y0: i64 = (py - r).floor() as i64;
            let y1: i64 = (py + r).ceil() as i64;
            for yy in y0..=y1 {
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                for xx in x0..=x1 {
                    if xx < 0 || xx >= w as i64 {
                        continue;
                    }
                    let dx = xx as f32 - px;
                    let dy = yy as f32 - py;
                    let d2 = dx * dx + dy * dy;
                    if d2 > r2 {
                        continue;
                    }
                    // 核 w(r) = 1 − (d/r)²（紧支撑，平滑到边缘 0；无超越函数）。
                    let t = 1.0 - d2 / r2;
                    let wgt = t * t * self.intensity; // 平方软化（平滑衰减）。
                    let base = out.get(xx as u32, yy as u32);
                    let cr = sp.color[0] * wgt;
                    let cg = sp.color[1] * wgt;
                    let cb = sp.color[2] * wgt;
                    let ca = sp.color[3] * wgt;
                    // over 合成（加色 splat；alpha 累加钳 1）。
                    let na = ca + base[3] * (1.0 - ca);
                    let nr = if na > 0.0 { cr + base[0] * (1.0 - ca) } else { 0.0 };
                    let ng = if na > 0.0 { cg + base[1] * (1.0 - ca) } else { 0.0 };
                    let nb = if na > 0.0 { cb + base[2] * (1.0 - ca) } else { 0.0 };
                    out.set(xx as u32, yy as u32, [nr, ng, nb, na.min(1.0)]);
                }
            }
        }
        out
    }
}
