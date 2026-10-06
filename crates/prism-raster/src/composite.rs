//! 覆盖率 → RGBA16F 的合成落地：路径填充与线段绘制。
//!
//! 合成语义为 **直通 alpha 的 src-over**（与 prism-io `Fbuf16F` 的非预乘
//! 存储一致）：`a_out = a_s + a_d·(1−a_s)`；f32 计算、f16 RNE 写入，
//! 逐字节确定。

use crate::geom::{Point, Transform};
use crate::path::PathBuilder;
use crate::scanline::{rasterize, FillRule};
use crate::Coverage;
use prism_io::Fbuf16F;

/// 直通 alpha 的 src-over 单像素合成。
///
/// 输入/输出均为非预乘 RGBA（f32 线性域）；`a_dst = 0` 时结果为源色。
#[inline]
pub fn blend_pixel_over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let a_s = src[3];
    let a_d = dst[3];
    let keep = 1.0 - a_s;
    let a_out = a_s + a_d * keep;
    if a_out == 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / a_out;
    [
        (src[0] * a_s + dst[0] * a_d * keep) * inv,
        (src[1] * a_s + dst[1] * a_d * keep) * inv,
        (src[2] * a_s + dst[2] * a_d * keep) * inv,
        a_out,
    ]
}

/// 把覆盖率缓冲作为源 alpha（颜色恒定）合成进帧缓冲。
fn composite_coverage(fbuf: &mut Fbuf16F, cov: &Coverage, color: [f32; 4]) {
    for y in 0..cov.height.min(fbuf.height) {
        for x in 0..cov.width.min(fbuf.width) {
            let c = cov.get(x, y);
            if c <= 0.0 {
                continue;
            }
            let src = [color[0], color[1], color[2], color[3] * c];
            let dst = fbuf.get(x, y);
            fbuf.set(x, y, blend_pixel_over(src, dst));
        }
    }
}

/// 路径填充：平铺 → 扫描线光栅 → src-over 合成进帧缓冲。
///
/// 变换把路径坐标映到设备像素；路径自动闭合。
pub fn fill_path(
    fbuf: &mut Fbuf16F,
    path: &PathBuilder,
    transform: &Transform,
    color: [f32; 4],
    rule: FillRule,
) {
    let contours = path.flatten(transform);
    let cov = rasterize(&contours, rule, fbuf.width, fbuf.height);
    composite_coverage(fbuf, &cov, color);
}

/// 线段绘制：以线宽构造矩形四边形后走同一填充管线（平头端帽）。
///
/// 零长线段不产出任何像素（方向未定义，避免静默伪影）。
pub fn draw_line(fbuf: &mut Fbuf16F, a: Point, b: Point, width: f32, color: [f32; 4]) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 || width <= 0.0 {
        return;
    }
    let half = width * 0.5;
    let inv_len = half / sqrt_f32(len_sq);
    let nx = -dy * inv_len;
    let ny = dx * inv_len;
    let mut pb = PathBuilder::new();
    pb.move_to(Point::new(a.x + nx, a.y + ny));
    pb.line_to(Point::new(b.x + nx, b.y + ny));
    pb.line_to(Point::new(b.x - nx, b.y - ny));
    pb.line_to(Point::new(a.x - nx, a.y - ny));
    pb.close();
    fill_path(fbuf, &pb, &Transform::IDENTITY, color, FillRule::Winding);
}

/// IEEE 754 精确平方根（单条硬件指令语义，正确舍入——铁律允许的 `√`，
/// 非 libm 超越函数）。
#[inline]
fn sqrt_f32(v: f32) -> f32 {
    f32::sqrt(v)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Transform;

    #[test]
    fn blend_over_straight_alpha_math() {
        // 不透明源覆盖一切。
        let out = blend_pixel_over([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0]);
        // 半透明源叠不透明底：alpha = 0.5，颜色为混合。
        let out = blend_pixel_over([1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 1.0]);
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[1] - 0.5).abs() < 1e-6);
        // 叠在透明底上：直通源色。
        let out = blend_pixel_over([0.25, 0.5, 0.75, 0.5], [0.0; 4]);
        assert_eq!(out, [0.25, 0.5, 0.75, 0.5]);
        // 双透明：结果透明。
        assert_eq!(blend_pixel_over([1.0, 1.0, 1.0, 0.0], [0.0; 4]), [0.0; 4]);
    }

    #[test]
    fn fill_path_paints_square_into_fbuf() {
        let mut fbuf = Fbuf16F::new(4, 4);
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(1.5, 1.5));
        pb.line_to(Point::new(3.0, 1.5));
        pb.line_to(Point::new(3.0, 3.0));
        pb.line_to(Point::new(1.5, 3.0));
        pb.close();
        fill_path(&mut fbuf, &pb, &Transform::IDENTITY, [1.0, 0.0, 0.0, 1.0], FillRule::Winding);
        // 像素 (2,2) 全覆盖 → 红色 alpha=1（f16 精确）。
        assert_eq!(fbuf.get(2, 2), [1.0, 0.0, 0.0, 1.0]);
        // 边界像素半覆盖：alpha = 0.5（f16 可精确表示）。
        let [_, _, _, a] = fbuf.get(1, 2);
        assert_eq!(a, 0.5, "边界列半覆盖");
        // 视口外像素不动。
        assert_eq!(fbuf.get(0, 0), [0.0; 4]);
    }

    #[test]
    fn draw_line_horizontal_band() {
        let mut fbuf = Fbuf16F::new(8, 5);
        draw_line(
            &mut fbuf,
            Point::new(1.0, 2.5),
            Point::new(7.0, 2.5),
            1.0,
            [0.0, 0.0, 1.0, 1.0],
        );
        // 水平线 y∈[2,3)：行 2 全覆盖。
        for x in 1..7u32 {
            let [_, _, _, a] = fbuf.get(x, 2);
            assert_eq!(a, 1.0, "线身全覆盖 x={x}");
        }
        let [_, _, _, a] = fbuf.get(0, 2);
        assert_eq!(a, 0.0, "起点之外无线");
    }

    #[test]
    fn zero_length_line_is_noop() {
        let mut fbuf = Fbuf16F::new(2, 2);
        draw_line(&mut fbuf, Point::new(1.0, 1.0), Point::new(1.0, 1.0), 3.0, [1.0; 4]);
        assert!(fbuf.pixels.iter().all(|&p| p == 0), "零长线段不得产出像素");
    }
}
