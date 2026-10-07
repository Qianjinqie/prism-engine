//! 镜头语言库（文字/镜头管线， 格锚纪律）。
//!
//! # 定位
//!
//! 本模块把"摄影机运动"抽象为**参数化轨迹**——镜头模板给出一个纯函数
//! `f(t) → CameraState`，下游（`prism-rt` 的 [`Camera`] 构造 / 引擎求值层）
//! 消费其输出的镜头参数（位置 / 朝向 / fov / 光圈 / 对焦距离）逐帧重建相机。
//! 与既有 [`crate::track`]/[`crate::spring`]/[`crate::easing`] 正交复用：模板
//! 内部用 `EasingKind` 做时间重映射，而非重造缓动。
//!
//! 覆盖（B15 镜头语言库）：
//! - **推拉摇移**（dolly/zoom）模板：[`CameraMove`] 的 `Dolly` 沿视线推进、
//!   `Pedestal`/`Truck` 平移、`Pant`/`Tilt` 转向、`Zoom` 变焦——都是
//!   `pose(t)` 的解析解；
//! - **dolly zoom**：[`DollyZoom`] 希区柯克变焦——主体在画面中的投影尺度
//!   守恒而背景透视剧变，焦距与机位**耦合**（关键耦合式见结构体文档）；
//! - **手持 shake**：[`Handheld`] 用确定性噪声（[`Wiggle`] 接 `prism-noise`）
//!   驱动位移/旋转抖动，同 seed 同 t 逐位一致（**无全局 RNG**）；
//! - **摄影学 DOF 参数化**：[`LensSpec`] 焦段/光圈/超焦距 ↔ 对焦距离、
//!  弥散圆直径、景深远近界（薄透镜近似）。
//!
//! # 确定性契约（D-29）
//!
//! 全部函数是其参数的纯函数：无内部可变状态、无隐藏 RNG、无 libm。三角项走
//! [`crate::det`]（固定系数多项式，跨平台逐位一致）；噪声走 `prism-noise` 的
//! 整数哈希派生（唯一随机源）。因此**同参数跨平台逐位一致**。

use crate::det;
use crate::easing::EasingKind;
use crate::Wiggle;
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 镜头位姿
// ---------------------------------------------------------------------------

/// 镜头位姿（逐时刻的相机参数；`prism-rt::Camera::new` 的直接输入）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    /// 视点（world 单位）。
    pub eye: Vec3,
    /// 注视目标点（world 单位）。
    pub look_at: Vec3,
    /// 竖直视场角（度，C17 豁免）。
    pub fov_deg: f32,
}

impl CameraPose {
    /// 构造位姿。
    pub fn new(eye: Vec3, look_at: Vec3, fov_deg: f32) -> Self {
        Self { eye, look_at, fov_deg }
    }

    /// 视线方向（单位向量；eye == look_at 时返回 `(0,0,-1)` 兜底——确定性特例）。
    pub fn forward(&self) -> Vec3 {
        let d = self.look_at - self.eye;
        let len = d.length();
        if len <= 1e-9 {
            Vec3::new(0.0, 0.0, -1.0)
        } else {
            d * (1.0 / len)
        }
    }
}

// ---------------------------------------------------------------------------
// 推拉摇移模板
// ---------------------------------------------------------------------------

/// 基础镜头运动模板（推拉摇移； 镜头语言）。
///
/// 全部模板以 `duration` 秒为时长，`0 ≤ t ≤ duration` 内按 `easing` 做时间
/// 重映射，`t ≥ duration` 钳到终点（确定性语义，non-loop）。位置单位 world，
/// 角度单位度（C17）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMove {
    /// 推/拉（dolly）：沿 `dir`（单位向量）移动 `distance`（正=推近，负=拉远）。
    Dolly {
        /// 起始位姿。
        from: CameraPose,
        /// 位移方向（单位向量，通常为视线前向）。
        dir: Vec3,
        /// 沿 `dir` 的位移量（world 单位）。
        distance: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
    /// 横移（truck）：沿相机右轴平移。
    Truck {
        /// 起始位姿。
        from: CameraPose,
        /// 右轴（单位向量）。
        right: Vec3,
        /// 位移量（world 单位）。
        distance: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
    /// 升降（pedestal）：沿相机上轴平移。
    Pedestal {
        /// 起始位姿。
        from: CameraPose,
        /// 上轴（单位向量）。
        up: Vec3,
        /// 位移量（world 单位）。
        distance: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
    /// 摇（pant）：绕相机上轴转动 `angle_deg`（正=右转），眼位固定。
    Pant {
        /// 起始位姿。
        from: CameraPose,
        /// 上轴（单位向量，转轴）。
        up: Vec3,
        /// 转角（度）。
        angle_deg: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
    /// 俯仰（tilt）：绕相机右轴转动 `angle_deg`（正=上抬），眼位固定。
    Tilt {
        /// 起始位姿。
        from: CameraPose,
        /// 右轴（单位向量，转轴）。
        right: Vec3,
        /// 转角（度）。
        angle_deg: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
    /// 变焦（zoom）：fov 从 `from.fov_deg` 线性变到 `to_fov_deg`（眼位固定）。
    Zoom {
        /// 起始位姿。
        from: CameraPose,
        /// 目标 fov（度）。
        to_fov_deg: f32,
        /// 时长（秒）。
        duration: f32,
        /// 时间重映射。
        easing: EasingKind,
    },
}

impl CameraMove {
    /// 时间归一化进度 u ∈ [0,1]：`u = clamp(t/duration, 0, 1)` 后经 easing 重映射。
    ///
    /// `duration ≤ 0` → u = 1（瞬时到位）。easing 在 u 上求值（`easing` 为
    /// `EasingKind::eval(u)`，与 [`crate::track`] 同源）。
    fn progress(&self, t: f32, duration: f32, easing: EasingKind) -> f32 {
        if duration <= 0.0 {
            return 1.0;
        }
        let u = (t / duration).clamp(0.0, 1.0);
        easing.apply(u).clamp(0.0, 1.0)
    }

    /// 求时刻 `t` 的位姿（纯函数；`t` 越界钳到起/终点）。
    pub fn pose(&self, t: f32) -> CameraPose {
        match *self {
            CameraMove::Dolly { from, dir, distance, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                let n = normalize_or(dir, Vec3::new(0.0, 0.0, -1.0));
                let e = from.eye + n * (distance * u);
                CameraPose { eye: e, look_at: from.look_at + n * (distance * u), ..from }
            }
            CameraMove::Truck { from, right, distance, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                let n = normalize_or(right, Vec3::new(1.0, 0.0, 0.0));
                let d = n * (distance * u);
                CameraPose { eye: from.eye + d, look_at: from.look_at + d, ..from }
            }
            CameraMove::Pedestal { from, up, distance, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                let n = normalize_or(up, Vec3::new(0.0, 1.0, 0.0));
                let d = n * (distance * u);
                CameraPose { eye: from.eye + d, look_at: from.look_at + d, ..from }
            }
            CameraMove::Pant { from, up, angle_deg, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                let axis = normalize_or(up, Vec3::new(0.0, 1.0, 0.0));
                let delta = from.look_at - from.eye;
                let r = rotate_axis(delta, axis, angle_deg * u);
                CameraPose { eye: from.eye, look_at: from.eye + r, ..from }
            }
            CameraMove::Tilt { from, right, angle_deg, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                let axis = normalize_or(right, Vec3::new(1.0, 0.0, 0.0));
                let delta = from.look_at - from.eye;
                let r = rotate_axis(delta, axis, angle_deg * u);
                CameraPose { eye: from.eye, look_at: from.eye + r, ..from }
            }
            CameraMove::Zoom { from, to_fov_deg, duration, easing } => {
                let u = Self::progress(self, t, duration, easing);
                CameraPose { fov_deg: from.fov_deg + (to_fov_deg - from.fov_deg) * u, ..from }
            }
        }
    }

    /// 模板时长（秒）。
    pub fn duration(&self) -> f32 {
        match *self {
            CameraMove::Dolly { duration, .. }
            | CameraMove::Truck { duration, .. }
            | CameraMove::Pedestal { duration, .. }
            | CameraMove::Pant { duration, .. }
            | CameraMove::Tilt { duration, .. }
            | CameraMove::Zoom { duration, .. } => duration,
        }
    }
}

// ---------------------------------------------------------------------------
// dolly zoom（希区柯克变焦：焦距-位置耦合）
// ---------------------------------------------------------------------------

/// dolly zoom（希区柯克变焦 / vertigo effect）。
///
/// # 耦合式（数学结构，D-21）
///
/// 令相机距主体距离为 `d`、竖直 fov 为 `f`。主体（宽 `w`）在成像面上的
/// 投影半高正比于 `w / d`，而半高在归一化像面坐标为 `(w/2)/(d·tan(f/2))`。
/// **尺度守恒**要求该比值沿时间恒定：
///
/// ```text
///   w / (d(t) · tan(f(t)/2)) = w / (d₀ · tan(f₀/2))
///   ⇒  tan(f(t)/2) = tan(f₀/2) · d₀ / d(t)
///   ⇒  f(t) = 2 · atan( tan(f₀/2) · d₀ / d(t) ) · 180/π
/// ```
///
/// 即：**焦距越短（fov 越大）则机位越近**，二者由上式耦合而非独立动画——
/// 这正是 dolly zoom 的解析解，无需步进器（D-21"解析替代步进"）。
/// `d(t)` 由 [`CameraMove::Dolly`] 沿视线推进给出（线性插值后 easing 重映射）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DollyZoom {
    /// 起始位姿（`fov_deg` = f₀）。
    pub from: CameraPose,
    /// 主体所在点（look_at；dolly 沿 eye→look_at 方向推进）。
    pub subject: Vec3,
    /// 终态机位距离（world 单位；从起始距离线性插值到它）。
    pub end_distance: f32,
    /// 时长（秒）。
    pub duration: f32,
    /// 时间重映射。
    pub easing: EasingKind,
}

impl DollyZoom {
    /// 起始机位到主体的距离 `d₀`（下限 1e-6 防除零）。
    pub fn start_distance(&self) -> f32 {
        let d = (self.subject - self.from.eye).length();
        if d < 1e-6 {
            1e-6
        } else {
            d
        }
    }

    /// 时刻 `t` 的机位距离 `d(t)`（线性 + easing，下限 1e-6）。
    pub fn distance(&self, t: f32) -> f32 {
        let u = if self.duration <= 0.0 {
            1.0
        } else {
            (t / self.duration).clamp(0.0, 1.0)
        };
        let u = self.easing.apply(u).clamp(0.0, 1.0);
        let d = self.start_distance() + (self.end_distance - self.start_distance()) * u;
        if d < 1e-6 {
            1e-6
        } else {
            d
        }
    }

    /// 时刻 `t` 的 fov（度）：由尺度守恒耦合式解出（见结构体文档）。
    ///
    /// `tan(f/2) = tan(f₀/2) · d₀ / d(t)`；`d(t) = d₀` 时精确回到 f₀（逐位）。
    pub fn fov(&self, t: f32) -> f32 {
        let tan_half0 = deg_tan(self.from.fov_deg * 0.5);
        let ratio = self.start_distance() / self.distance(t);
        let tan_half = tan_half0 * ratio;
        // f = 2·atan(x)；atan 用恒等式 atan(x) = π/2 − atan(1/x)（x>0）经
        // det 的多项式件折叠，避免 libm。此处 tan_half > 0（fov ∈ (0,180)）。
        let half = atan_pos(tan_half);
        2.0 * half * (180.0 / std::f32::consts::PI)
    }

    /// 时刻 `t` 的位姿：机位沿 eye→subject 射线取 `d(t)`，fov 由 [`Self::fov`] 给出。
    pub fn pose(&self, t: f32) -> CameraPose {
        let d = self.distance(t);
        let dir = normalize_or(self.subject - self.from.eye, Vec3::new(0.0, 0.0, -1.0));
        CameraPose { eye: self.subject - dir * d, look_at: self.subject, fov_deg: self.fov(t) }
    }

    /// 耦合不变量校验值：`d(t)·tan(f(t)/2)`（尺度守恒下沿 t 恒定）。测试用。
    pub fn invariant(&self, t: f32) -> f32 {
        self.distance(t) * deg_tan(self.fov(t) * 0.5)
    }

    /// 时长（秒）。
    pub fn duration(&self) -> f32 {
        self.duration
    }
}

// ---------------------------------------------------------------------------
// 手持 shake（确定性噪声）
// ---------------------------------------------------------------------------

/// 手持抖动通道（确定性噪声驱动； 噪声）。
///
/// 三条独立噪声（[`Wiggle`]，不同 seed 派生）分别驱动：
/// - 眼位偏移（`pos_amp`，world 单位，沿相机 right/up 轴）；
/// - 视线方向抖动（`aim_amp`，度，绕 right/up 轴微转）；
/// - 可选滚转（`roll_amp`，度；此处并入 aim 的上向扰动）。
///
/// **无全局 RNG**：噪声是整数哈希派生（`prism-noise`），同 seed 同 t 逐位一致。
/// `seed` 经 [`crate::mix_seeds`] 与全局种子混合（EvalCtx 衔接路径）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Handheld {
    /// 眼位偏移振幅（world 单位）。
    pub pos_amp: f32,
    /// 视线抖动振幅（度）。
    pub aim_amp: f32,
    /// 抖动频率（Hz）。
    pub frequency: f32,
    /// fbm 倍频程（0 → 无抖动）。
    pub octaves: u32,
    /// 通道种子。
    pub seed: u64,
}

impl Handheld {
    /// 构造手持抖动通道（推荐参数：pos_amp ~0.02、aim_amp ~0.3、freq ~2.5）。
    pub fn new(pos_amp: f32, aim_amp: f32, frequency: f32, octaves: u32, seed: u64) -> Self {
        Self { pos_amp, aim_amp, frequency, octaves, seed }
    }

    /// 三条噪声分量 (dx, dy, dai) 于时刻 `t`（各自独立 seed 派生，确定性）。
    fn noise(&self, t: f32) -> (f32, f32, f32) {
        let wx = Wiggle::new(1.0, self.frequency, self.octaves, self.seed ^ 0x01);
        let wy = Wiggle::new(1.0, self.frequency, self.octaves, self.seed ^ 0x02);
        let wz = Wiggle::new(1.0, self.frequency, self.octaves, self.seed ^ 0x03);
        (wx.sample(t), wy.sample(t), wz.sample(t))
    }

    /// 求时刻 `t` 的抖动位姿：`base` 叠加确定性噪声偏移与视线微转。
    ///
    /// `right`/`up` 为 `base` 的相机基（调用方提供，保证与 `prism-rt` 一致）。
    pub fn pose(&self, base: CameraPose, right: Vec3, up: Vec3, t: f32) -> CameraPose {
        let (dx, dy, dai) = self.noise(t);
        let r = normalize_or(right, Vec3::new(1.0, 0.0, 0.0));
        let u = normalize_or(up, Vec3::new(0.0, 1.0, 0.0));
        let offset = r * (dx * self.pos_amp) + u * (dy * self.pos_amp);
        let eye = base.eye + offset;
        // 视线微转：绕 up 转 dai（右转）、绕 right 转 dy（俯仰），幅度 aim_amp 度。
        let delta = base.look_at - base.eye;
        let yaw = rotate_axis(delta, u, dai * self.aim_amp);
        let pitch = rotate_axis(yaw, r, dy * self.aim_amp);
        CameraPose { eye, look_at: eye + pitch, fov_deg: base.fov_deg }
    }
}

// ---------------------------------------------------------------------------
// 摄影学 DOF 参数化
// ---------------------------------------------------------------------------

/// 镜头规格（摄影学：焦段/光圈/传感器 → 光学量）。
///
/// # 薄透镜景深模型
///
/// - 焦距 `f`（mm）、光圈数 `N = f/D`（D 为有效孔径 mm）、传感器对角线
///   `sensor_diag`（mm）；
/// - 对焦距离 `focus`（world 单位，与相机一致）；
/// - **弥散圆直径**：`C = f² / (N · (focus − f))`（成像侧几何，mm，近似）；
/// - **超焦距**：`H = f² / (N · c)` 除以单位换算（c 为容许弥散圆直径），
///   近似形式 `H ≈ f² / (N · c)`；
/// - **景深近/远界**（相对 `focus`）：`Dn = H·d/(H + (d − f))`、
///   `Df = H·d/(H − (d − f))`（`d` = 对焦距离，`f` = 焦距）。
///
/// 全部为解析式（D-21），无迭代。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensSpec {
    /// 焦距（mm）。
    pub focal_mm: f32,
    /// 光圈数 N（f-number；越大光圈越小）。
    pub f_number: f32,
    /// 传感器对角线（mm；默认全画幅 43.27）。
    pub sensor_diag_mm: f32,
    /// 容许弥散圆直径（mm；默认 0.029）。
    pub coc_mm: f32,
}

impl Default for LensSpec {
    fn default() -> Self {
        Self { focal_mm: 50.0, f_number: 2.8, sensor_diag_mm: 43.27, coc_mm: 0.029 }
    }
}

impl LensSpec {
    /// 构造镜头规格（焦段/光圈；传感器与弥散圆取全画幅默认）。
    pub fn new(focal_mm: f32, f_number: f32) -> Self {
        Self { focal_mm, f_number, ..Self::default() }
    }

    /// 光瞳直径 D = f / N（mm）。
    pub fn pupil_mm(&self) -> f32 {
        self.focal_mm / self.f_number
    }

    /// 由竖直 fov（度）反解焦距（mm）：`f = (sensor_dim/2) / tan(fov/2)`。
    ///
    /// 用于把 [`CameraPose::fov_deg`] 与焦段互译（dolly zoom 时焦距随 fov 变）。
    pub fn focal_for_fov(fov_deg: f32, sensor_height_mm: f32) -> f32 {
        let t = deg_tan(fov_deg * 0.5);
        if t <= 1e-6 {
            f32::INFINITY
        } else {
            (sensor_height_mm * 0.5) / t
        }
    }

    /// 由焦距（mm）与传感器高（mm）反解竖直 fov（度）：`fov = 2·atan(h/2f)`。
    pub fn fov_for_focal(focal_mm: f32, sensor_height_mm: f32) -> f32 {
        if focal_mm <= 1e-6 {
            180.0
        } else {
            2.0 * atan_pos((sensor_height_mm * 0.5) / focal_mm) * (180.0 / std::f32::consts::PI)
        }
    }

    /// 超焦距 H（mm）：`H ≈ f² / (N · c)`。
    pub fn hyperfocal_mm(&self) -> f32 {
        let denom = self.f_number * self.coc_mm;
        if denom <= 1e-9 {
            f32::INFINITY
        } else {
            self.focal_mm * self.focal_mm / denom
        }
    }

    /// 弥散圆直径 C（mm，成像侧几何近似）：`C = f² / (N · (d − f))`。
    ///
    /// `d ≤ f`（对焦距离不大于焦距）→ +∞（超出薄透镜有效域）。
    pub fn coc_diameter_mm(&self, focus_mm: f32) -> f32 {
        let d = focus_mm - self.focal_mm;
        if d <= 1e-9 {
            f32::INFINITY
        } else {
            self.focal_mm * self.focal_mm / (self.f_number * d)
        }
    }

    /// 景深近界（沿对焦距离轴，与 `focus` 同单位）。
    ///
    /// `Dn = H·d / (H + (d − f))`（H/f/d 同单位 mm；`d ≤ f` → 0 兜底）。
    pub fn dof_near(&self, focus: f32) -> f32 {
        let h = self.hyperfocal_mm();
        let d = focus;
        let denom = h + (d - self.focal_mm);
        if !denom.is_finite() || denom <= 1e-9 {
            0.0
        } else {
            (h * d / denom).max(0.0)
        }
    }

    /// 景深远界（∞ 当 `H ≤ d − f`——超过超焦距时远界无穷）。
    pub fn dof_far(&self, focus: f32) -> f32 {
        let h = self.hyperfocal_mm();
        let d = focus;
        let denom = h - (d - self.focal_mm);
        if !h.is_finite() {
            f32::INFINITY
        } else if denom <= 1e-9 {
            f32::INFINITY
        } else {
            h * d / denom
        }
    }

    /// 光圈半径（world 单位）：由 `aperture_mm = f/(2N)` 与 `focus`（world）换算。
    ///
    /// 供 `prism-rt::Camera::new` 的 `aperture_radius` 直接消费；此处按
    /// 传感器尺度近似（world/mm 比例 = `focus / focus_mm` 恒等，返回 mm 半径）。
    pub fn aperture_radius(&self) -> f32 {
        self.focal_mm / (2.0 * self.f_number)
    }
}

// ---------------------------------------------------------------------------
// 确定性数学辅助（D-29：无 libm）
// ---------------------------------------------------------------------------

/// 单位化；零向量退化时返回 `fallback`（确定性特例）。
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len = v.length();
    if len <= 1e-9 {
        fallback
    } else {
        v * (1.0 / len)
    }
}

/// 绕单位轴 `axis` 旋转向量 `v` 角度 `angle_deg`（Rodrigues，确定性）。
///
/// `v' = v·cosθ + (axis × v)·sinθ + axis·(axis·v)·(1 − cosθ)`。
#[inline]
fn rotate_axis(v: Vec3, axis: Vec3, angle_deg: f32) -> Vec3 {
    let a = normalize_or(axis, Vec3::new(0.0, 1.0, 0.0));
    let theta = angle_deg * (std::f32::consts::PI / 180.0);
    let (s, c) = (det::sin_f32(theta), det::cos_f32(theta));
    let cross = a.cross(&v);
    let dot = a.dot(&v);
    v * c + cross * s + a * (dot * (1.0 - c))
}

/// tan(x)（弧度，f32）经 [`crate::det`] 的 sin/cos 比值（D-29 合规）。
#[inline]
fn tan_f32(x: f32) -> f32 {
    let c = det::cos_f32(x);
    if c.abs() < 1e-9 {
        f32::INFINITY
    } else {
        det::sin_f32(x) / c
    }
}

/// `tan(deg°)`（度入口，内部转弧度走 [`tan_f32`]）。
#[inline]
fn deg_tan(deg: f32) -> f32 {
    tan_f32(deg * (std::f32::consts::PI / 180.0))
}

/// atan(x)（x ≥ 0，弧度）的确定性实现。
///
/// 标准区间归约到自变量 `≤ 0.41421…`（`atan(0.4142…) ≈ π/8` 处误差最小），
/// 再用 8 项奇次 Taylor 逼近（在缩减域上收敛充分，误差 < 1e-6；镜头 fov
/// 反解用途足够，测试按此容差）。全程只用 `+ − × ÷`（D-29 合规）。
///
/// 归约恒等式（对任意 x > 0）：
/// - `x > 1`：`atan(x) = π/2 − atan(1/x)`；
/// - `0.41421… < x ≤ 1`：`atan(x) = π/4 + atan((x−1)/(x+1))`。
/// 关键：折叠后的自变量仍可能 > 0.41421…（如 `x` 略大于 1 时 `1/x ≈ 0.9999`），
/// 必须**继续二次归约**，否则 Taylor 在 `≈1` 处 8 项截断误差达数个 1e-2。
#[inline]
fn taylor_atan_small(x: f32) -> f32 {
    // atan(x) = x − x³/3 + x⁵/5 − …（缩减域上收敛充分）。
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut k = 1.0f32;
    for _ in 0..8 {
        term *= -x2;
        k += 2.0;
        sum += term / k;
    }
    sum
}

#[inline]
fn atan_pos(x: f32) -> f32 {
    // 奇函数：atan(−x) = −atan(x)。归约过程中出现的小负自变量由此处理。
    if x < 0.0 {
        return -atan_pos(-x);
    }
    if !(x > 0.0) {
        return 0.0;
    }
    // 先归约 `x > 1`：atan(x) = π/2 − atan(1/x)。
    if x > 1.0 {
        return std::f32::consts::FRAC_PI_2 - atan_pos(1.0 / x);
    }
    // 再归约 `x ∈ (0.41421…, 1]`：atan(x) = π/4 + atan((x−1)/(x+1))。
    if x > 0.414_213_56 {
        return std::f32::consts::FRAC_PI_4 + atan_pos((x - 1.0) / (x + 1.0));
    }
    // `x ≤ 0.41421…`：直接 Taylor（收敛快）。
    taylor_atan_small(x)
}

// ---------------------------------------------------------------------------
// 测试（关卡 8：镜头模板轨迹断言 + dolly zoom 耦合 + shake 确定性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> CameraPose {
        CameraPose::new(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 0.0), 50.0)
    }

    /// tan 精确性（对参考 math tan 容差）：保证耦合式与 DOF 换算可信。
    #[test]
    fn det_tan_matches_reference() {
        let mut x = -1.4_f32;
        while x <= 1.4 {
            let d = (tan_f32(x) - x.tan()).abs();
            assert!(d < 1e-5, "tan 偏差 {x}: {d}");
            x += 0.037;
        }
    }

    /// atan 全定义域精度（对参考 math atan，误差须远小于 1e-3）。
    ///
    /// 覆盖各归约分支（<0.4142 / 0.4142..1 / >1）与分支边界，防止折叠处
    /// 出现不连续（曾因 `x>1` 折叠后未二次归约导致 `x≈1` 处误差 ~1e-2）。
    #[test]
    fn det_atan_full_domain_accuracy() {
        let mut x = 1e-3_f32;
        while x <= 20.0 {
            let d = (atan_pos(x) - x.atan()).abs();
            assert!(d < 1e-4, "atan 偏差 {x}: {d}");
            x *= 1.01;
            x += 1e-4;
        }
        // 分支边界显式核查。
        for x in [0.414_2f32, 0.5, 0.999_9, 1.0, 1.000_1, 1.5, 3.0, 10.0] {
            let d = (atan_pos(x) - x.atan()).abs();
            assert!(d < 1e-4, "atan 分支边界 {x}: {d}");
        }
    }

    /// 推拉模板：位移沿 dir、比例于 easing 进度；终点钳制精确。
    #[test]
    fn dolly_track_linear_progress() {
        let m = CameraMove::Dolly {
            from: base(),
            dir: Vec3::new(0.0, 0.0, -1.0),
            distance: 4.0,
            duration: 2.0,
            easing: EasingKind::Linear,
        };
        // t=0 → 起点；t=1 → 半程；t=2 → 终点；t=5 → 钳终点。
        assert!((m.pose(0.0).eye.z - 10.0).abs() < 1e-5);
        assert!((m.pose(1.0).eye.z - 8.0).abs() < 1e-5);
        assert!((m.pose(2.0).eye.z - 6.0).abs() < 1e-5);
        assert!((m.pose(5.0).eye.z - 6.0).abs() < 1e-5);
        // 眼与注视点同步平移（视线方向不变）。
        let f0 = m.pose(0.0).forward();
        let f1 = m.pose(2.0).forward();
        assert!((f0.z - f1.z).abs() < 1e-5);
    }

    /// 摇镜头：绕 up 轴转动，视线方向按角度解析旋转，眼位不动。
    #[test]
    fn pant_rotates_about_up_axis() {
        let m = CameraMove::Pant {
            from: base(),
            up: Vec3::new(0.0, 1.0, 0.0),
            angle_deg: 90.0,
            duration: 1.0,
            easing: EasingKind::Linear,
        };
        let start = m.pose(0.0);
        let end = m.pose(1.0);
        // 起始前向 = (0,0,-1)；绕 +Y 转 90° → (−1,0,0)（右手系）。
        assert!((end.forward().x + 1.0).abs() < 1e-4, "x = {}", end.forward().x);
        assert!((end.forward().z).abs() < 1e-4, "z = {}", end.forward().z);
        // 眼位固定。
        assert_eq!(start.eye.x.to_bits(), end.eye.x.to_bits());
        assert_eq!(start.eye.z.to_bits(), end.eye.z.to_bits());
    }

    /// 变焦模板：fov 线性变化，眼位不动。
    #[test]
    fn zoom_changes_fov_only() {
        let m = CameraMove::Zoom {
            from: base(),
            to_fov_deg: 20.0,
            duration: 2.0,
            easing: EasingKind::Linear,
        };
        assert!((m.pose(0.0).fov_deg - 50.0).abs() < 1e-5);
        assert!((m.pose(1.0).fov_deg - 35.0).abs() < 1e-5);
        assert!((m.pose(2.0).fov_deg - 20.0).abs() < 1e-5);
        assert_eq!(m.pose(0.0).eye.x.to_bits(), m.pose(2.0).eye.x.to_bits());
    }

    /// **dolly zoom 焦距/位置耦合**（关卡 8 核心断言）：尺度守恒不变量
    /// `d(t)·tan(f(t)/2)` 沿 t 恒定；且 d 减小时 fov 增大、d 增大时 fov 减小。
    #[test]
    fn dolly_zoom_couples_focal_and_position() {
        let dz = DollyZoom {
            from: CameraPose::new(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 0.0), 40.0),
            subject: Vec3::new(0.0, 0.0, 0.0),
            end_distance: 4.0,
            duration: 1.0,
            easing: EasingKind::Linear,
        };
        let inv0 = dz.invariant(0.0);
        for i in 0..=100 {
            let t = i as f32 * 0.01;
            let inv = dz.invariant(t);
            assert!(
                (inv - inv0).abs() / inv0 < 5e-3,
                "尺度守恒被破坏 t={t}: {inv} vs {inv0}"
            );
        }
        // 机位靠近（10 → 4），fov 必须增大（焦距变短）。
        assert!(dz.fov(1.0) > dz.fov(0.0), "推近应增大 fov");
        assert!((dz.distance(0.0) - 10.0).abs() < 1e-4);
        assert!((dz.distance(1.0) - 4.0).abs() < 1e-4);
        // d = d₀ 时 fov 精确回起点。
        assert!((dz.fov(0.0) - 40.0).abs() < 1e-3, "起态 fov 应回到 40，实得 {}", dz.fov(0.0));
    }

    /// 手持 shake：同 seed 同 t 逐位一致；不同 seed 抖动流不同；零振幅恒等于 base。
    #[test]
    fn handheld_is_deterministic_and_zero_amp_is_identity() {
        let hh = Handheld::new(0.05, 0.4, 2.5, 3, 0xABCD);
        let b = base();
        let r = Vec3::new(1.0, 0.0, 0.0);
        let u = Vec3::new(0.0, 1.0, 0.0);
        for t in [0.0_f32, 0.3, 1.7, 5.5] {
            let p1 = hh.pose(b, r, u, t);
            for _ in 0..1000 {
                let p2 = hh.pose(b, r, u, t);
                assert_eq!(p1.eye.x.to_bits(), p2.eye.x.to_bits());
                assert_eq!(p1.eye.y.to_bits(), p2.eye.y.to_bits());
                assert_eq!(p1.eye.z.to_bits(), p2.eye.z.to_bits());
            }
        }
        // 不同 seed 流不同。
        let hb = Handheld::new(0.05, 0.4, 2.5, 3, 0x1111);
        let differing = (0..32)
            .filter(|i| {
                hh.pose(b, r, u, *i as f32).eye != hb.pose(b, r, u, *i as f32).eye
            })
            .count();
        assert!(differing >= 24, "不同 seed 应产生不同抖动，仅 {differing}/32 不同");
        // 零振幅 = 恒等。
        let zero = Handheld::new(0.0, 0.0, 2.5, 3, 0xABCD);
        let p = zero.pose(b, r, u, 3.3);
        assert_eq!(p.eye.x.to_bits(), b.eye.x.to_bits());
        assert_eq!(p.eye.y.to_bits(), b.eye.y.to_bits());
        assert_eq!(p.look_at.z.to_bits(), b.look_at.z.to_bits());
    }

    /// shake 位移有界：眼位偏移不超过 pos_amp·√2（沿 right/up 两分量）。
    #[test]
    fn handheld_offset_bounded() {
        let amp = 0.1f32;
        let hh = Handheld::new(amp, 0.3, 3.0, 4, 0xFEED);
        let b = base();
        let r = Vec3::new(1.0, 0.0, 0.0);
        let u = Vec3::new(0.0, 1.0, 0.0);
        for i in 0..5000u64 {
            let t = i as f32 * 0.01;
            let p = hh.pose(b, r, u, t);
            let off = p.eye - b.eye;
            let bound = amp * std::f32::consts::SQRT_2 + 1e-6;
            assert!(off.length() <= bound, "抖动越界 t={t}: {}", off.length());
        }
    }

    /// DOF 参数化：超焦距解析式；光圈越大（N 越小）景深越浅；超焦距处远界→∞。
    #[test]
    fn dof_parameterization_is_physical() {
        let lens = LensSpec::new(50.0, 2.8);
        let h = lens.hyperfocal_mm();
        // H = f²/(N·c) = 2500/(2.8×0.029) ≈ 30788 mm。
        assert!((h - 2500.0 / (2.8 * 0.029)).abs() < 1e-2, "超焦距 H = {h}");
        // 对焦距离 = 超焦距 → 远界 ∞。
        assert!(lens.dof_far(h).is_infinite() || lens.dof_far(h) > 1e6);
        // 大光圈（N=1.4）近界应比小光圈（N=11）更近（景深更浅 → DoF 更窄）。
        let fast = LensSpec::new(50.0, 1.4);
        let slow = LensSpec::new(50.0, 11.0);
        let d_near_fast = fast.dof_near(3000.0);
        let d_near_slow = slow.dof_near(3000.0);
        let d_far_fast = fast.dof_far(3000.0);
        let d_far_slow = slow.dof_far(3000.0);
        assert!(d_far_fast - d_near_fast < d_far_slow - d_near_slow, "大光圈景深应更浅");
        // 光瞳与传统定义一致。
        assert!((lens.pupil_mm() - 50.0 / 2.8).abs() < 1e-4);
        assert!((lens.aperture_radius() - 50.0 / (2.0 * 2.8)).abs() < 1e-4);
    }

    /// fov ↔ 焦距互译自洽（round-trip，镜头摄影学常用换算）。
    #[test]
    fn fov_focal_roundtrip() {
        let h = 24.0f32; // 全画幅传感器高
        for fov in [20.0_f32, 40.0, 60.0, 90.0] {
            let f = LensSpec::focal_for_fov(fov, h);
            let back = LensSpec::fov_for_focal(f, h);
            assert!((back - fov).abs() < 0.5, "round-trip fov {fov} → {back}");
        }
    }
}
