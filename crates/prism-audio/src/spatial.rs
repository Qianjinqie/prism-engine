//! 空间音频：3D 声源 → 声像 / 距离衰减，Stage3D 联动接口。
//!
//! 把场景中的 3D 声源位置（Stage3D 世界坐标）映射为立体声输出：**方向**决定
//! 声像（左右）、**距离**决定衰减。为不经第三方向量库、不与 `prism-anim` 循环
//! 依赖，此处定义最小向量 [`Vec3`]（`prism-anim` 的 Stage3D 可通过
//! [`SpatialSource::from_world_pos`] 适配）。
//!
//! 距离衰减模型（[`AttenuationModel`]）：
//! - Reference / Inverse（反比 `ref/（ref + rolloff·(d−ref)）`）
//! - Linear（`ref` 内 1、`max` 外 0，中间线性）
//! - None。
//!
//! 声像由声源相对听者的**方位角**在听者朝向坐标系下的水平分量决定（前左→左）。

use crate::fx::panner::constant_power_gains;

/// 3D 向量（最小实现，避免第三方依赖）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// X 分量。
    pub x: f32,
    /// Y 分量。
    pub y: f32,
    /// Z 分量。
    pub z: f32,
}

impl Vec3 {
    /// 构造。
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// 逐分量减法。
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    /// 点积。
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// 模长。
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// 归一化（零向量返回零向量）。
    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l < 1e-9 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(self.x / l, self.y / l, self.z / l)
        }
    }
}

/// 距离衰减模型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttenuationModel {
    /// 无衰减。
    None,
    /// 反比：`ref / (ref + rolloff·(d − ref))`（d > ref 时），d ≤ ref 时为 1。
    Inverse,
    /// 线性：`(max − d)/(max − ref)`（夹在 [0,1]）。
    Linear,
}

/// 距离衰减参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attenuation {
    /// 模型。
    pub model: AttenuationModel,
    /// 参考距离（此距离内增益 = 1）。
    pub ref_dist: f32,
    /// 最大距离（Linear 模型用；超出增益 0）。
    pub max_dist: f32,
    /// 滚降系数（Inverse 模型用）。
    pub rolloff: f32,
}

impl Default for Attenuation {
    fn default() -> Self {
        Self { model: AttenuationModel::Inverse, ref_dist: 1.0, max_dist: 50.0, rolloff: 1.0 }
    }
}

impl Attenuation {
    /// 距离 `d` 处的增益 ∈ [0, 1]（纯函数）。
    pub fn gain(&self, d: f32) -> f32 {
        let d = d.max(0.0);
        let r = self.ref_dist.max(1e-4);
        match self.model {
            AttenuationModel::None => 1.0,
            AttenuationModel::Inverse => {
                if d <= r {
                    1.0
                } else {
                    (r / (r + self.rolloff.max(0.0) * (d - r))).clamp(0.0, 1.0)
                }
            }
            AttenuationModel::Linear => {
                let mx = self.max_dist.max(r + 1e-4);
                if d <= r {
                    1.0
                } else if d >= mx {
                    0.0
                } else {
                    (mx - d) / (mx - r)
                }
            }
        }
    }
}

/// 听者姿态（世界坐标 + 朝向）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Listener {
    /// 位置。
    pub position: Vec3,
    /// 前向单位向量。
    pub forward: Vec3,
    /// 右向单位向量。
    pub right: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            forward: Vec3::new(0.0, 0.0, 1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
        }
    }
}

/// 3D 声源：位置 + 衰减参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialSource {
    /// 世界坐标位置（Stage3D 适配入参）。
    pub position: Vec3,
    /// 距离衰减。
    pub attenuation: Attenuation,
}

impl SpatialSource {
    /// 由 Stage3D 世界坐标构造（联动接口：`prism-anim` 的 Stage3D 传入 x/y/z）。
    pub fn from_world_pos(x: f32, y: f32, z: f32, attenuation: Attenuation) -> Self {
        Self { position: Vec3::new(x, y, z), attenuation }
    }

    /// 相对听者的距离。
    pub fn distance(&self, listener: &Listener) -> f32 {
        self.position.sub(listener.position).length()
    }

    /// 距离衰减增益。
    pub fn distance_gain(&self, listener: &Listener) -> f32 {
        self.attenuation.gain(self.distance(listener))
    }

    /// 声像 ∈ [−1, 1]：声源方向在听者右向轴上的投影（归一化方向）。
    pub fn pan(&self, listener: &Listener) -> f32 {
        let dir = self.position.sub(listener.position).normalized();
        let right = listener.right.normalized();
        dir.dot(right).clamp(-1.0, 1.0)
    }

    /// 综合空间增益对 `(gL, gR)`：恒功率声像 × 距离衰减。
    pub fn stereo_gains(&self, listener: &Listener) -> (f32, f32) {
        let (gl, gr) = constant_power_gains(self.pan(listener));
        let d = self.distance_gain(listener);
        (gl * d, gr * d)
    }
}

/// 把单声道源按空间位置混入立体声缓冲（就地叠加；源样本 × 左右增益）。
pub fn mix_spatial(
    input: &[f32],
    sample_rate: u32,
    source: &SpatialSource,
    listener: &Listener,
) -> crate::buffer::PcmBuffer {
    let (gl, gr) = source.stereo_gains(listener);
    let mut data = Vec::with_capacity(input.len() * 2);
    for &x in input {
        data.push(x * gl);
        data.push(x * gr);
    }
    crate::buffer::PcmBuffer::from_interleaved(sample_rate, 2, data).expect("形状必然合法")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 距离衰减：参考距离内 1，超出按反比下降，单调递减。
    #[test]
    fn inverse_attenuation_monotonic() {
        let att = Attenuation { model: AttenuationModel::Inverse, ref_dist: 1.0, max_dist: 100.0, rolloff: 1.0 };
        assert!((att.gain(0.5) - 1.0).abs() < 1e-6);
        assert!((att.gain(1.0) - 1.0).abs() < 1e-6);
        let g2 = att.gain(2.0);
        let g10 = att.gain(10.0);
        assert!(g2 < 1.0 && g10 < g2, "应单调递减：g2={g2} g10={g10}");
        // ref=1, rolloff=1, d=2 → 1/(1+1) = 0.5。
        assert!((g2 - 0.5).abs() < 1e-5);
    }

    /// 线性衰减：ref 内 1、max 外 0，中点≈0.5。
    #[test]
    fn linear_attenuation_endpoints() {
        let att = Attenuation { model: AttenuationModel::Linear, ref_dist: 0.0, max_dist: 10.0, rolloff: 1.0 };
        assert!((att.gain(0.0) - 1.0).abs() < 1e-6);
        assert!((att.gain(10.0)).abs() < 1e-6);
        assert!((att.gain(5.0) - 0.5).abs() < 1e-5);
    }

    /// 声像：听者右侧声源 → 右声道大；左前方 → 左声道大。
    #[test]
    fn pan_follows_direction() {
        let listener = Listener::default();
        let right = SpatialSource::from_world_pos(5.0, 0.0, 0.0, Attenuation::default());
        assert!(right.pan(&listener) > 0.9, "正右方声像应 ≈ +1，实际 {}", right.pan(&listener));
        let left = SpatialSource::from_world_pos(-5.0, 0.0, 0.0, Attenuation::default());
        assert!(left.pan(&listener) < -0.9);
        let front = SpatialSource::from_world_pos(0.0, 0.0, 5.0, Attenuation::default());
        assert!(front.pan(&listener).abs() < 0.01, "正前方居中");
    }

    /// 立体声增益：远景声源总能量低于近景（距离衰减生效）。
    #[test]
    fn distance_reduces_stereo_energy() {
        let listener = Listener::default();
        let near = SpatialSource::from_world_pos(0.0, 0.0, 1.0, Attenuation::default());
        let far = SpatialSource::from_world_pos(0.0, 0.0, 30.0, Attenuation::default());
        let (nl, nr) = near.stereo_gains(&listener);
        let (fl, fr) = far.stereo_gains(&listener);
        assert!(fl * fl + fr * fr < nl * nl + nr * nr, "远景总能量应更低");
    }

    /// mix_spatial：声道输出 = 源 × 左右增益；确定性。
    #[test]
    fn mix_spatial_orders_channels() {
        let listener = Listener::default();
        let src = SpatialSource::from_world_pos(10.0, 0.0, 0.0, Attenuation::default());
        let out = mix_spatial(&[1.0], 8_000, &src, &listener);
        let l = out.sample(0, 0).unwrap();
        let r = out.sample(0, 1).unwrap();
        assert!(r > l, "右方声源右声道应更大：L={l} R={r}");
        let again = mix_spatial(&[1.0], 8_000, &src, &listener);
        assert_eq!(out.as_slice(), again.as_slice());
    }
}
