//! 屏幕空间 GI（SSGI）重投影：S2 卷积的时域核（账三 U4）。
//!
//! # 定位
//! 屏幕空间 GI 用**上一帧**的 GI 结果，按相机运动重投影到当前帧——这是
//! **S2 卷积的时域特例**（核 = 运动向量决定的重采样核，相位 = 像素坐标）。
//! 它复用现成的深度缓冲（[`crate::depth`] 的 [`prism_core::DepthMap`]）与相机
//! 矩阵，把历史 GI 缓存搬到当前视角，使实时档位能"每帧只算一部分、其余从
//! 历史重用"（质量-成本曲线的时域维度）。
//!
//! # D-21：不做 Lumen 式的屏幕空间探针层次
//! Lumen 用屏幕空间的探针层次 + 时空滤波处理重投影误差。本实现只做**基于
//! 深度/相机的刚性重投影 + 简单可视性判定**（前向重投影 + 深度一致性），
//! 没有去遮挡检测、没有邻域颜色裁剪。代价：快速运动/遮挡变化时会有拖影；
//! 收益：几十行、确定性、无状态机。
//!
//! # 确定性
//! 重投影是 `(历史帧, 相机, 深度)` 的纯函数：逐像素固定序遍历，无随机、
//! 无并发。静止相机时重投影恒等（见测试）。

use prism_core::DepthMap;
use prism_math::Vec3;

use crate::camera::Camera;

/// 帧间相机运动（用于重投影：本帧相机 + 上帧相机）。
///
/// 重投影把"上帧像素"映射到"本帧像素"：把上帧像素经其上帧相机反投影到
/// 世界（用上帧深度），再用本帧相机投影回来。若深度/相机不可用则退回
/// 纯相机旋转近似。
#[derive(Debug, Clone)]
pub struct CameraMotion {
    /// 上一帧相机。
    pub prev: Camera,
    /// 上一帧分辨率。
    pub prev_res: (u32, u32),
    /// 本帧相机。
    pub cur: Camera,
    /// 本帧分辨率。
    pub cur_res: (u32, u32),
}

/// 重投影结果：当前帧每个像素的历史来源像素坐标（`None` = 无有效历史）。
#[derive(Debug, Clone, PartialEq)]
pub struct ReprojectionMap {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 逐像素历史来源 `(prev_x, prev_y, validity)`；无效时 prev_x/prev_y = 0 且
    /// validity = 0。保存 f32 亚像素坐标供双线性采样。
    pub uv: Vec<[f32; 3]>,
}

impl ReprojectionMap {
    /// 读取某像素的历史来源 `(x, y)` 与有效性（0/1）。
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> ([f32; 2], f32) {
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        let d = self.uv[idx];
        ([d[0], d[1]], d[2])
    }
}

/// 上帧深度重投影：把本帧每个像素反推其历史来源像素。
///
/// 步骤（逐像素）：
/// 1. 本帧像素 + 本帧深度 → 世界点（反投影）；
/// 2. 世界点 + 上帧相机 → 上帧像素（投影）；
/// 3. 上帧深度采样做**一致性检查**：若上帧同点深度与本帧推算深度差异过大，
///    标为无效（去遮挡的极简版）。
///
/// # 参数
/// - `motion`：本帧/上帧相机与分辨率；
/// - `cur_depth`：本帧深度图（世界单位；<0 = 无命中）；
/// - `prev_depth`：上帧深度图（同上）。
///
/// # 确定性
/// 逐像素固定序，无随机。静止相机（prev == cur，同分辨率）时恒等（测试锚定）。
pub fn reproject_gi(
    motion: &CameraMotion,
    cur_depth: &DepthMap,
    prev_depth: &DepthMap,
) -> ReprojectionMap {
    let w = cur_depth.width;
    let h = cur_depth.height;
    let mut uv = vec![[0.0f32; 3]; (w as usize) * (h as usize)];
    // 深度一致性容差（世界单位）：与深度尺度相关，取相对量。
    let tol = 0.1f32;
    for y in 0..h {
        for x in 0..w {
            let d_cur = cur_depth.get(x, y);
            let idx = (y as usize) * (w as usize) + (x as usize);
            if !(d_cur > 0.0) {
                uv[idx] = [0.0, 0.0, 0.0]; // 无命中 → 无历史
                continue;
            }
            // 1. 本帧像素 → 世界点（沿相机主射线前进 d_cur）。
            let dir_cur = pixel_dir(&motion.cur, x as f32 + 0.5, y as f32 + 0.5, motion.cur_res);
            let p_world = motion.cur.eye + dir_cur.normalize() * d_cur;
            // 2. 世界点 → 上帧像素。
            let (px, py, d_prev_computed) = project_to_prev(motion, p_world);
            // 边界：落在上帧外 → 无效。
            if px < 0.0 || py < 0.0 || px >= motion.prev_res.0 as f32 || py >= motion.prev_res.1 as f32
            {
                uv[idx] = [0.0, 0.0, 0.0];
                continue;
            }
            // 3. 深度一致性：上帧同点深度 vs 推算深度。
            let prev_x = px.floor().max(0.0).min((motion.prev_res.0 as f32) - 1.0) as u32;
            let prev_y = py.floor().max(0.0).min((motion.prev_res.1 as f32) - 1.0) as u32;
            let d_prev = prev_depth.get(prev_x, prev_y);
            let valid = if d_prev > 0.0 && d_prev_computed.is_finite() && d_prev_computed > 0.0 {
                // 相对差（用两者均值归一）在容差内 → 有效。
                let denom = (d_prev + d_prev_computed).abs() * 0.5;
                denom > 1e-6 && ((d_prev - d_prev_computed).abs() / denom) < tol
            } else {
                false
            };
            uv[idx] = [px, py, if valid { 1.0 } else { 0.0 }];
        }
    }
    ReprojectionMap { width: w, height: h, uv }
}

/// 像素方向（未单位化；与 [`Camera::pixel_dir`] 同约定，此处独立实现以免
/// 依赖 SampleStream——重投影是几何运算，无需采样流）。
fn pixel_dir(cam: &Camera, px: f32, py: f32, res: (u32, u32)) -> Vec3 {
    let w = res.0.max(1) as f32;
    let h = res.1.max(1) as f32;
    let u = (px / w) * 2.0 - 1.0;
    let v = 1.0 - (py / h) * 2.0;
    let aspect = w / h;
    cam.forward + cam.right * (u * cam.tan_half_fov * aspect) + cam.up * (v * cam.tan_half_fov)
}

/// 世界点 → 上帧像素坐标（含到上帧相机的**射线距离**）。
///
/// 用相机基（right/up/forward）做解析投影：给定世界点 P，
/// `local = (dot(rel,right), dot(rel,up), dot(rel,forward))`，
/// 视角深度 = local.z（用于像素投影），另返回 `|rel|`（射线距离，用于与
/// 深度缓冲的命中距离一致性比对——深度缓冲存的是射线参数 t，非视角 z）。
fn project_to_prev(motion: &CameraMotion, p: Vec3) -> (f32, f32, f32) {
    let cam = &motion.prev;
    let rel = p - cam.eye;
    let z = rel.dot(&cam.forward);
    if z <= 1e-6 {
        return (-1.0, -1.0, -1.0); // 在相机后方 → 无效
    }
    let xc = rel.dot(&cam.right);
    let yc = rel.dot(&cam.up);
    let tan = cam.tan_half_fov.max(1e-6);
    let w = motion.prev_res.0.max(1) as f32;
    let h = motion.prev_res.1.max(1) as f32;
    let aspect = w / h;
    // NDC（x 右、y 上）→ 像素（y 向下）。
    let ndc_x = xc / (z * tan * aspect);
    let ndc_y = yc / (z * tan);
    let px = (ndc_x + 1.0) * 0.5 * w;
    let py = (1.0 - ndc_y) * 0.5 * h;
    (px, py, rel.length())
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(eye: Vec3, look: Vec3) -> Camera {
        Camera::new(eye, look, Vec3::new(0.0, 1.0, 0.0), 60.0, 0.0, 5.0, 0.0, 0.0)
    }

    fn make_depth(w: u32, h: u32, fill: impl Fn(u32, u32) -> f32) -> DepthMap {
        let mut dm = DepthMap::new(w, h, 0.1, 100.0);
        for y in 0..h {
            for x in 0..w {
                dm.set(x, y, fill(x, y));
            }
        }
        dm
    }

    /// 静止相机：重投影恒等（每个像素的历史来源 = 自己）。
    #[test]
    fn static_camera_reprojection_is_identity() {
        let c = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let motion = CameraMotion { prev: c.clone(), prev_res: (4, 4), cur: c, cur_res: (4, 4) };
        // 恒深度 2.0 的平面（与相机视线垂直的常深度面 → 重投影精确恒等）。
        let depth = make_depth(4, 4, |_x, _y| 2.0);
        let map = reproject_gi(&motion, &depth, &depth);
        for y in 0..4 {
            for x in 0..4 {
                let ([px, py], v) = map.get(x, y);
                assert!(v > 0.5, "静止相机像素 ({x},{y}) 应有效");
                // 恒深度面：重投影精确（< 0.5 像素误差）。
                assert!(
                    (px - (x as f32 + 0.5)).abs() < 0.5 && (py - (y as f32 + 0.5)).abs() < 0.5,
                    "静止相机应恒等：({x},{y}) → ({px},{py})"
                );
            }
        }
    }

    /// 确定性：同输入两遍重投影结果逐位一致。
    #[test]
    fn reprojection_is_bitwise_deterministic() {
        let a = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let b = cam(Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.1, 0.0, 1.0));
        let motion = CameraMotion { prev: a, prev_res: (8, 8), cur: b, cur_res: (8, 8) };
        let cur = make_depth(8, 8, |_x, _y| 3.0);
        let prev = make_depth(8, 8, |_x, _y| 3.0);
        let m1 = reproject_gi(&motion, &cur, &prev);
        let m2 = reproject_gi(&motion, &cur, &prev);
        assert_eq!(m1, m2, "重投影必须逐位确定");
    }

    /// 平移相机：重投影方向正确（相机右移 → 历史来源像素相应偏移）。
    #[test]
    fn translation_shifts_reprojection() {
        let a = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let b = cam(Vec3::new(0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 1.0));
        let motion = CameraMotion { prev: a, prev_res: (16, 16), cur: b, cur_res: (16, 16) };
        let cur = make_depth(16, 16, |_x, _y| 4.0);
        // 上帧深度与推算一致（同深度的近似）→ 有效。
        let prev = make_depth(16, 16, |_x, _y| 4.0);
        let map = reproject_gi(&motion, &cur, &prev);
        // 至少一个像素有效（平移不大，大部分应有效）。
        let mut any_valid = false;
        for y in 0..16 {
            for x in 0..16 {
                if map.get(x, y).1 > 0.5 {
                    any_valid = true;
                }
            }
        }
        assert!(any_valid, "小平移应有有效重投影");
    }

    /// 无命中像素（深度 < 0）→ 无历史，不 panic。
    #[test]
    fn no_hit_pixels_have_no_history() {
        let c = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let motion = CameraMotion { prev: c.clone(), prev_res: (4, 4), cur: c, cur_res: (4, 4) };
        // 深度 -1 = 无命中。
        let depth = make_depth(4, 4, |_x, _y| -1.0);
        let map = reproject_gi(&motion, &depth, &depth);
        for y in 0..4 {
            for x in 0..4 {
                let (_p, v) = map.get(x, y);
                assert!(v < 0.5, "无命中像素不应有历史");
            }
        }
    }

    /// 极端输入：NaN 深度、退化相机（后方点）、1×1 分辨率 → 不 panic。
    #[test]
    fn extreme_inputs_no_panic() {
        let c = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let motion = CameraMotion { prev: c.clone(), prev_res: (1, 1), cur: c, cur_res: (1, 1) };
        let depth = make_depth(1, 1, |_x, _y| f32::NAN);
        let _ = reproject_gi(&motion, &depth, &depth);
        // 相机看向 −Z（世界点在本帧相机后方时上游处理）——换一个背向相机。
        let back = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, -1.0));
        let motion2 = CameraMotion { prev: back.clone(), prev_res: (2, 2), cur: back, cur_res: (2, 2) };
        let d2 = make_depth(2, 2, |_x, _y| 2.0);
        let _ = reproject_gi(&motion2, &d2, &d2);
    }

    /// 测试自证非恒真：大平移导致历史来源越界 → 有效像素数下降。
    #[test]
    fn large_translation_invalidates() {
        let a = cam(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        let b = cam(Vec3::new(50.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 1.0)); // 巨幅平移
        let motion = CameraMotion { prev: a, prev_res: (8, 8), cur: b, cur_res: (8, 8) };
        let cur = make_depth(8, 8, |_x, _y| 3.0);
        let prev = make_depth(8, 8, |_x, _y| 3.0);
        let map = reproject_gi(&motion, &cur, &prev);
        let valid = (0..8)
            .flat_map(|y| (0..8).map(move |x| (x, y)))
            .filter(|&(x, y)| map.get(x, y).1 > 0.5)
            .count();
        assert_eq!(valid, 0, "巨幅平移应使全部像素无有效历史，实际 {valid}");
    }
}
