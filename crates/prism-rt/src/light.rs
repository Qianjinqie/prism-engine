//! 灯光族：方向光 / 球光 / 矩形面光 / 环境光 + NEE 采样与 MIS 权重。
//!
//! # 灯光语义与 MIS 契约（既有语义保持）
//! - [`Light::Directional`]：锥形方向光（软太阳），**非几何**——BSDF 采样路径
//!   永远打不中它，NEE 是唯一估计器 → NEE 权重恒 1（既有修正，必须遵守；
//!   引入伪 MIS 会系统性欠估）。
//! - [`Light::Sphere`]：均匀发光球，**场景几何**（Scene.spheres 以 Emissive
//!   材质注册）——BSDF 采样可命中 → NEE 与 BSDF 两策略 power heuristic 配对。
//! - [`Light::Rect`]：矩形面光（双面发光），**非几何**（不进 Scene 求交列表）
//!   ——BSDF 路径不可命中 → NEE 权重恒 1。契约：若未来矩形进入求交列表，
//!   必须同步改为 power heuristic，否则双重计数。
//! - [`Light::Ambient`]：均匀环境光，仅作为**逃逸背景**贡献（BSDF 采样路径
//!   miss 时由积分器经 [`ambient_background`] 全额取得）；NEE 不采样
//!   （[`sample_nee`] 返回 None）→ 无重复计数，权重语义恒 1。
//!
//! # 确定性（D-29）
//! 采样只用 ±×÷√ 与 [`crate::sampler`] 的 det 三角件（`cone_dir`/`cos_sin`），
//! 无 libm 超越函数、无全局 RNG；u1/u2 由样本流哈希派生，任意像素任意 pass
//! 可独立重算。

use std::f32::consts::PI;

use prism_math::Vec3;

use crate::sampler::cone_dir;

// ---------------------------------------------------------------------------
// 灯光定义
// ---------------------------------------------------------------------------

/// 场景灯光（`Directional`/`Sphere` 字段逐字保留，B2 增补 `Rect`/`Ambient`）。
#[derive(Debug, Clone, Copy)]
pub enum Light {
    /// 方向光：`direction` = 指向光的方向（单位化），锥半角由 `cos_theta_max`
    /// 给定，辐亮度常数（无距离衰减）。非几何——BSDF/相位抽样不可命中。
    Directional {
        /// 指向光的方向（单位向量）。
        direction: Vec3,
        /// 入射辐亮度。
        radiance: Vec3,
        /// 锥半角余弦（1.0 = 纯 δ 方向）。
        cos_theta_max: f32,
    },
    /// 球光源：均匀辐亮度发光球（场景几何，参与 MIS）。
    Sphere {
        /// 球心。
        center: Vec3,
        /// 半径（> 0）。
        radius: f32,
        /// 出射辐亮度。
        radiance: Vec3,
    },
    /// 矩形面光：中心 + 两正交边向量定义平行四边形，双面发光，非几何。
    Rect {
        /// 矩形中心。
        center: Vec3,
        /// 半宽边向量（中心到边中点的向量，即半边）。
        edge_u: Vec3,
        /// 另一半边向量。
        edge_v: Vec3,
        /// 出射辐亮度（双面同值）。
        radiance: Vec3,
    },
    /// 均匀环境光：方向无关的逃逸背景辐亮度。
    Ambient {
        /// 背景辐亮度。
        radiance: Vec3,
    },
    /// HDR 环境贴图（等距柱状 + 立体角 2D CDF；A2 / W2）。
    ///
    /// # W1-A 定桩：只有句柄
    ///
    /// 本变体在 W0-A 只落地**字段**（`u32` 句柄，保持 `Light: Copy`），
    /// 采样（CDF 二分 + 边际条件反演 + MIS 融合）全是 **W2** 的事。
    /// 当前它**不参与任何计算**：`sample_nee` 返回 `None`（NEE 不直接采它，
    /// 走 CDF 才是 W2 的路子），`ambient_background` 只认 [`Light::Ambient`]
    /// ⇒ 场景里出现本变体，本段的输出与"没有这盏灯"逐位相同。
    ///
    /// # 与 [`Light::Ambient`] **互斥**
    ///
    /// 两者都描述"逃逸射线看到的背景"。同时存在 ⇒ 逃逸分支会双计
    /// （`ambient_background` 给常数、环境图给采样值），而双计的表现是
    /// "背景偏亮"这种不会报错、只会被对拍抓到的 bug。故
    /// [`crate::integrator::Scene::validate_layout`] 把它做成**构造期断言**。
    Environment {
        /// 环境贴图句柄（索引进 `Scene::textures`；`0` = 无 —— 与
        /// [`crate::texture::NO_TEXTURE`] 同一约定）。
        map: u32,
    },
}

/// `Light` 必须仍是 `Copy`（选型报告 §2.2 / §3.1 第 4 条：`u32` 句柄而非
/// `Arc<EnvMap>`，就是为了不动这个 derive）。
///
/// 静态证：`CopyOnly<T: Copy>` 的常量若 `Light` 不满足 `Copy` 就编不过。
const _: () = {
    /// 只接受 `Copy` 类型的零尺寸标记。
    struct CopyOnly<T: Copy>(core::marker::PhantomData<T>);
    let _: CopyOnly<Light> = CopyOnly(core::marker::PhantomData::<Light>);
};

/// `Light` 的尺寸不得因 `Environment` 变胖：`Rect`（4 × `Vec3` = 48 B）仍是
/// 最大变体，判别式 1 B → 对齐后 52 B。`Environment { map }` 只占 4 B，
/// 落在 `Rect` 的payload 之内 ⇒ 尺寸不变。
const _: () = assert!(core::mem::size_of::<Light>() == 52);
/// 对齐 4 B（全 f32/u32）。
const _: () = assert!(core::mem::align_of::<Light>() == 4);

/// NEE 采样结果：对着色点 `p` 采样灯光得到的一条入射辐射样本。
#[derive(Debug, Clone, Copy)]
pub struct LightSample {
    /// 指向光采样点的方向（单位向量）。
    pub wi: Vec3,
    /// 遮挡射线的最大行进距离（方向光 = 无穷；球光 = 到球心；矩形 = 到面点）。
    pub dist: f32,
    /// 入射辐亮度（含距离/角度衰减的换算在调用方按 pdf 与几何项处理）。
    pub radiance: Vec3,
    /// 对入射方向 wi 的立体角 pdf（δ 退化时记 1.0 占位，积分器按 delta 处理）。
    pub pdf_w: f32,
    /// δ 灯（纯方向光）——积分器不给 BSDF 策略配对。
    pub delta: bool,
}

// ---------------------------------------------------------------------------
// MIS 权重
// ---------------------------------------------------------------------------

/// MIS 幂次启发式（Veach 1995 power heuristic，β=2）：w = a²/(a²+b²)。
/// B1 在 integrator 私有定义，B2 上收至此供灯光/积分器共用。
#[inline]
pub fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let den = a2 + b * b;
    if den == 0.0 {
        return 0.0; // 双零防御（正常路径不发生）
    }
    a2 / den
}

/// NEE 样本的 MIS 权重（按灯光种类的契约见模块文档）：
/// 球光（几何发光体）= power heuristic(pdf_light, pdf_bsdf)；
/// 环境**贴图**（`Environment`，BSDF 逃逸射线可命中）= 与球光同臂；
/// 方向光 / 矩形光 / 常数环境光 `Ambient`（非几何，BSDF 路径不可命中）= 恒 1。
#[inline]
pub fn nee_mis_weight(light: &Light, pdf_light: f32, pdf_bsdf: f32) -> f32 {
    match light {
        // 球光（几何发光体）与环境**贴图**（BSDF 逃逸射线可命中）同臂：
        // 两策略 power heuristic 配对。W2 起本臂已可达：
        // `integrator::nee_env_surface` / `nee_env_volume` 拿着 Environment 的
        // CDF 样本走到这里，与逃逸分支的 `1 − power_heuristic(...)` 逐位互补；
        // 忘接本臂的表现是背景偏亮 ~2×（权重 1 而非 0.5），V1 白炉 MIS
        // 互补断言就是拦这个的。
        //
        // V3.1 P1-2：合并两个逐字相同的 arm（去重复分支），数值零变化。
        Light::Sphere { .. } | Light::Environment { .. } => {
            power_heuristic(pdf_light, pdf_bsdf)
        }
        // 非几何灯：BSDF 采样路径取不到该灯的贡献，NEE 全额、无配对。
        _ => 1.0,
    }
}

/// 逃逸背景辐亮度：灯光列表中的环境光（至多一个生效）的辐射；无则黑。
pub fn ambient_background(lights: &[Light]) -> Vec3 {
    for l in lights {
        if let Light::Ambient { radiance } = l {
            return *radiance;
        }
    }
    Vec3::splat(0.0)
}

// ---------------------------------------------------------------------------
// NEE 采样（纯函数：p、u1、u2 完全决定结果）
// ---------------------------------------------------------------------------

/// 从着色点 `p` 对灯光做 NEE 采样（直接光照策略）。
/// `u1`/`u2` 为样本流坐标（hash 派生）。
/// 返回 None：点在球光内 / 矩形掠射（pdf 发散）/ 环境光（无 NEE 语义）。
pub fn sample_nee(light: &Light, p: Vec3, u1: f32, u2: f32) -> Option<LightSample> {
    match light {
        Light::Directional { direction, radiance, cos_theta_max } => {
            // 锥内均匀采样（同款）；pdf = 1/(2π(1−cos_max))。
            // cos_max = 1（δ）时 cone_dir 退化为正对轴，pdf 记 1.0 占位。
            // V3.1 P1-2：重复的 `*cos_theta_max >= 1.0` 预算一次复用
            // （纯比较，不改变任何浮点运算的操作数或顺序）。
            let is_delta: bool = *cos_theta_max >= 1.0;
            // 通用等价改造：δ 方向光的 cone_dir(w, 1, u, v) 恒等于 w
            // （cos_t = 1 − u·0 = 1、sin_t = √(1−1) = 0 ⇒ onb(w) 与
            // cos_sin(2πv) 的结果全部被乘零）。原每样本一次的 onb + 确定性
            // 三角 + 两次标量乘是纯死计算。直接取灯方向，逐位等价，对任意
            // 含 δ 方向光的场景（任意几何/材质/其它灯）生效——这是逐灯类型
            // 的通用分派，不是按场景内容判定的整帧分流。
            let wi = if is_delta {
                *direction
            } else {
                cone_dir(*direction, *cos_theta_max, u1, u2)
            };
            let pdf_w = if is_delta {
                1.0
            } else {
                1.0 / (2.0 * PI * (1.0 - cos_theta_max))
            };
            Some(LightSample {
                wi,
                dist: f32::INFINITY,
                radiance: *radiance,
                pdf_w,
                delta: is_delta,
            })
        }
        Light::Sphere { center, radius, radiance } => {
            // 着色点对球的立体角锥采样：cos_max = √(1−r²/d²)，
            // pdf_w = 1/(2π(1−cos_max)) = 1/Ω。点在球内不采样。
            let to_c = *center - p;
            let d_c = to_c.length();
            if d_c <= *radius {
                return None;
            }
            let cos_max = (1.0 - radius * radius / (d_c * d_c)).sqrt();
            let wi = cone_dir(to_c * (1.0 / d_c), cos_max, u1, u2);
            let pdf_w = 1.0 / (2.0 * PI * (1.0 - cos_max));
            Some(LightSample {
                wi,
                dist: d_c,
                radiance: *radiance,
                pdf_w,
                delta: false,
            })
        }
        Light::Rect { center, edge_u, edge_v, radiance } => {
            // 面上均匀采样 q = c + (u1−½)e_u + (u2−½)e_v，
            // 面积 pdf pdf_A = 1/|e_u×e_v|，换算立体角 pdf_w = pdf_A·d²/|cosθ_l|。
            let cr = edge_u.cross(edge_v);
            let norm2 = cr.dot(&cr);
            if norm2 <= 1e-12 {
                return None; // 退化矩形
            }
            let area = norm2.sqrt();
            let q = *center + *edge_u * (u1 - 0.5) + *edge_v * (u2 - 0.5);
            let to_q = q - p;
            let d2 = to_q.dot(&to_q);
            if d2 <= 1e-12 {
                return None; // 着色点贴在面上
            }
            let d = d2.sqrt();
            let wi = to_q * (1.0 / d);
            let n_l = cr * (1.0 / area);
            let cos_l = wi.dot(&n_l).abs(); // 双面发光
            if cos_l < 1e-6 {
                return None; // 掠射：pdf_w 发散，样本放弃（贡献按 0 计）
            }
            let pdf_w = (1.0 / area) * d2 / cos_l;
            Some(LightSample {
                wi,
                dist: d,
                radiance: *radiance,
                pdf_w,
                delta: false,
            })
        }
        Light::Ambient { .. } => None, // 环境光只走逃逸背景路径，不 NEE
        // W1-A 定桩：环境**贴图**也不走 NEE。它与 Ambient 的区别不在"能否
        // NEE"，而在"背景是否有方向结构"：W2 会在逃逸分支用 CDF 做**方向**
        // 重要性采样（那是 BSDF 侧的对偶策略，不是 NEE）。故这里恒 None。
        Light::Environment { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// 测试（NEE 权重契约 / pdf 换算恒等 / 锥采样几何锚点）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// V3.1 P1-2：合并 MIS arm + 预算 `is_delta` 后，与冻结的旧实现逐位一致。
    #[test]
    fn v31_structural_refactor_is_bitwise_identical() {
        fn mis_ref(light: &Light, a: f32, b: f32) -> f32 {
            match light {
                Light::Sphere { .. } => power_heuristic(a, b),
                Light::Environment { .. } => power_heuristic(a, b),
                _ => 1.0,
            }
        }
        fn dir_ref(
            direction: Vec3,
            radiance: Vec3,
            cos_max: f32,
            u1: f32,
            u2: f32,
        ) -> LightSample {
            let wi = cone_dir(direction, cos_max, u1, u2);
            let pdf_w = if cos_max >= 1.0 {
                1.0
            } else {
                1.0 / (2.0 * PI * (1.0 - cos_max))
            };
            LightSample {
                wi,
                dist: f32::INFINITY,
                radiance,
                pdf_w,
                delta: cos_max >= 1.0,
            }
        }
        let dir: Vec3 = Vec3::new(0.3, -0.9, 0.32);
        let rad: Vec3 = Vec3::new(4.0, 3.0, 2.0);
        let kinds: Vec<Light> = vec![
            Light::Directional { direction: dir, radiance: rad, cos_theta_max: 0.9 },
            Light::Directional { direction: dir, radiance: rad, cos_theta_max: 1.0 },
            Light::Sphere { center: Vec3::new(0.0, 0.0, 5.0), radius: 1.0, radiance: rad },
            Light::Rect {
                center: Vec3::new(0.0, 5.0, 0.0),
                edge_u: Vec3::new(2.0, 0.0, 0.0),
                edge_v: Vec3::new(0.0, 0.0, 2.0),
                radiance: rad,
            },
            Light::Ambient { radiance: rad },
            Light::Environment { map: 1 },
        ];

        // ① MIS 权重：合并 arm 后逐位一致。
        for l in &kinds {
            for (a, b) in [(0.0f32, 0.0f32), (0.2, 0.6), (1.5, 0.3), (0.9, 0.9)] {
                assert_eq!(
                    nee_mis_weight(l, a, b).to_bits(),
                    mis_ref(l, a, b).to_bits(),
                    "MIS 权重不等：{l:?}"
                );
            }
        }

        // ② Directional 分支：预算 is_delta 后逐位一致（含 δ 灯与锥灯边界）。
        let cos_maxes: [f32; 5] = [0.0, 0.5, 0.9, 0.999_999, 1.0];
        let mut checked: u32 = 0;
        for &cm in &cos_maxes {
            for k in 0..16usize {
                let u1: f32 = (k as f32 + 0.5) / 16.0;
                let u2: f32 = ((k * 7 % 16) as f32 + 0.5) / 16.0;
                let light = Light::Directional {
                    direction: dir,
                    radiance: rad,
                    cos_theta_max: cm,
                };
                let got: LightSample = sample_nee(
                    &light,
                    Vec3::new(0.0, 0.0, 0.0),
                    u1,
                    u2,
                )
                .expect("方向光 NEE 必有样本");
                let want: LightSample = dir_ref(dir, rad, cm, u1, u2);
                assert_eq!(got.wi.x.to_bits(), want.wi.x.to_bits(), "wi.x cm={cm} k={k}");
                assert_eq!(got.wi.y.to_bits(), want.wi.y.to_bits(), "wi.y cm={cm} k={k}");
                assert_eq!(got.wi.z.to_bits(), want.wi.z.to_bits(), "wi.z cm={cm} k={k}");
                assert_eq!(got.dist.to_bits(), want.dist.to_bits(), "dist cm={cm} k={k}");
                assert_eq!(got.pdf_w.to_bits(), want.pdf_w.to_bits(), "pdf_w cm={cm} k={k}");
                assert_eq!(got.delta, want.delta, "delta cm={cm} k={k}");
                checked += 1;
            }
        }
        assert!(checked >= 80, "对拍覆盖不足：{checked}");
    }

    /// 确定性样本流坐标（网格扫掠，测试内不引 Sobol 依赖）。
    fn grid_u(k: usize, n: usize) -> f32 {
        ((k % n) as f32 + 0.5) / n as f32
    }

    #[test]
    fn nee_mis_weights_by_kind() {
        // 方向光 / 矩形 / 环境 → 权重恒 1（逐位；非几何灯无 MIS 配对）。
        let dir = Light::Directional {
            direction: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::splat(1.0),
            cos_theta_max: 0.9,
        };
        assert_eq!(nee_mis_weight(&dir, 0.5, 0.3).to_bits(), 1.0f32.to_bits());
        let rect = Light::Rect {
            center: Vec3::new(0.0, 5.0, 0.0),
            edge_u: Vec3::new(2.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 2.0),
            radiance: Vec3::splat(3.0),
        };
        assert_eq!(nee_mis_weight(&rect, 0.5, 0.3).to_bits(), 1.0f32.to_bits());
        let amb = Light::Ambient { radiance: Vec3::splat(0.5) };
        assert_eq!(nee_mis_weight(&amb, 0.5, 0.3).to_bits(), 1.0f32.to_bits());
        // 球光 → power heuristic：对称、a=b → 0.5、解析值锚点。
        let sph = Light::Sphere {
            center: Vec3::new(0.0, 0.0, 5.0),
            radius: 1.0,
            radiance: Vec3::splat(2.0),
        };
        let w = nee_mis_weight(&sph, 0.2, 0.6);
        let w_rev = nee_mis_weight(&sph, 0.6, 0.2);
        // power heuristic 配对性质：互补（w_light + w_bsdf = 1）。
        assert!((w + w_rev - 1.0).abs() < 1e-7, "互补性破缺：{w} + {w_rev}");
        let want = 0.2 * 0.2 / (0.2 * 0.2 + 0.6 * 0.6);
        assert!((w - want).abs() < 1e-7, "w={w} 解析 {want}");
        let w_eq = nee_mis_weight(&sph, 0.4, 0.4);
        assert!((w_eq - 0.5).abs() < 1e-7, "等 pdf 权重应为 0.5：{w_eq}");
    }

    #[test]
    fn rect_pdf_conversion_identity() {
        // 面积↔立体角换算恒等：pdf_w·cos_l/d² == pdf_A（逐点，无 MC）。
        let rect = Light::Rect {
            center: Vec3::new(0.0, 6.0, 0.0),
            edge_u: Vec3::new(10.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 4.0),
            radiance: Vec3::splat(1.0),
        };
        let p = Vec3::new(1.2, 0.0, -0.7);
        let n_l = rect_edge_normal(&rect);
        let pdf_a = 1.0 / 40.0; // |10×4| = 40
        let mut checked = 0u32;
        for k in 0..256usize {
            let u1 = grid_u(k, 16);
            let u2 = grid_u(k + 7, 16);
            let s = match sample_nee(&rect, p, u1, u2) {
                Some(s) => s,
                None => continue, // 掠射样本允许被放弃
            };
            let cos_l = s.wi.dot(&n_l).abs();
            let lhs = s.pdf_w * cos_l / (s.dist * s.dist);
            assert!(
                (lhs - pdf_a).abs() / pdf_a < 1e-5,
                "换算恒等破缺：pdf_w·cos/d²={lhs} vs pdf_A={pdf_a}"
            );
            assert!(s.pdf_w > 0.0);
            checked += 1;
        }
        assert!(checked > 200, "有效样本过少：{checked}/256");
    }

    /// 提取矩形法线（测试辅助；与 sample_nee 内部同式）。
    fn rect_edge_normal(l: &Light) -> Vec3 {
        match l {
            Light::Rect { edge_u, edge_v, .. } => {
                let cr = edge_u.cross(edge_v);
                cr * (1.0 / cr.length())
            }
            _ => panic!("仅矩形"),
        }
    }

    #[test]
    fn sphere_solid_angle_anchor_and_hit() {
        // 立体角锚点：d=4, r=1 → cos_max=√15/4，pdf_w = 1/(2π(1−cos_max))。
        let sph = Light::Sphere {
            center: Vec3::new(0.0, 0.0, 4.0),
            radius: 1.0,
            radiance: Vec3::splat(2.0),
        };
        let p = Vec3::new(0.0, 0.0, 0.0);
        let cos_max = (15.0f32).sqrt() / 4.0;
        let pdf_want = 1.0 / (2.0 * PI * (1.0 - cos_max));
        let s = sample_nee(&sph, p, 0.3, 0.7).expect("球外点必须可采样");
        assert!((s.pdf_w - pdf_want).abs() / pdf_want < 1e-6, "pdf {} vs {pdf_want}", s.pdf_w);
        assert!((s.dist - 4.0).abs() < 1e-6);
        // 锥内方向必须 100% 命中球（ray-sphere 判别式 ≥ 0），采样 2000 个。
        let oc = p - Vec3::new(0.0, 0.0, 4.0);
        let mut hits = 0u32;
        let n = 2000u32;
        for k in 0..n {
            let u1 = grid_u(k as usize, 64);
            let u2 = grid_u(k as usize * 3 + 1, 64);
            let s = sample_nee(&sph, p, u1, u2).expect("锥采样不应失败");
            let b = oc.dot(&s.wi);
            let disc = b * b - (oc.dot(&oc) - 1.0);
            assert!(disc >= -1e-4, "锥内方向脱靶：disc={disc}");
            if disc >= 0.0 {
                hits += 1;
            }
        }
        assert_eq!(hits, n, "锥内方向应全部命中球");
        // 球内点不采样。
        let inside = sample_nee(&sph, Vec3::new(0.0, 0.0, 4.0), 0.5, 0.5);
        assert!(inside.is_none(), "球内点 sample_nee 应返回 None");
    }

    #[test]
    fn directional_cone_bounds_and_delta() {
        // 锥约束：wi·direction ≥ cos_theta_max（容差内），pdf 锚点。
        let dir_l = Light::Directional {
            direction: Vec3::new(0.3, 0.8, 0.5).normalize(),
            radiance: Vec3::splat(1.0),
            cos_theta_max: 0.9,
        };
        let pdf_want = 1.0 / (2.0 * PI * 0.1);
        let mut s0 = None;
        for k in 0..500usize {
            let s = sample_nee(&dir_l, Vec3::splat(0.0), grid_u(k, 32), grid_u(k + 11, 32))
                .expect("方向光总可采样");
            assert!(s.wi.dot(&Vec3::new(0.3, 0.8, 0.5).normalize()) >= 0.9 - 1e-5);
            assert!(
                (s.pdf_w - pdf_want).abs() / pdf_want < 1e-6,
                "pdf {} vs {pdf_want}",
                s.pdf_w
            );
            assert!(s.dist.is_infinite(), "方向光距离应为无穷");
            assert!(!s.delta);
            if s0.is_none() {
                s0 = Some(s);
            }
        }
        // δ 退化：cos_theta_max = 1 → wi 正对轴、delta = true、pdf 占位 1。
        let delta_l = Light::Directional {
            direction: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::splat(1.0),
            cos_theta_max: 1.0,
        };
        let s = sample_nee(&delta_l, Vec3::splat(0.0), 0.13, 0.77).unwrap();
        assert_eq!(s.wi.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(s.wi.z.to_bits(), 0.0f32.to_bits());
        assert!(s.delta, "cos_theta_max=1 应判定为 δ");
        assert_eq!(s.pdf_w.to_bits(), 1.0f32.to_bits(), "δ 光 pdf 占位 1.0");
    }

    #[test]
    fn ambient_escape_semantics() {
        // 环境光：NEE 不采样；逃逸背景取首个 Ambient 的辐射；无则黑。
        let amb = Light::Ambient { radiance: Vec3::new(0.3, 0.5, 0.7) };
        assert!(sample_nee(&amb, Vec3::splat(0.0), 0.5, 0.5).is_none());
        assert_eq!(ambient_background(&[]).x.to_bits(), 0.0f32.to_bits());
        let dir = Light::Directional {
            direction: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::splat(1.0),
            cos_theta_max: 0.95,
        };
        let lights = [dir, amb];
        let bg = ambient_background(&lights);
        assert_eq!(bg.x.to_bits(), 0.3f32.to_bits());
        assert_eq!(bg.z.to_bits(), 0.7f32.to_bits());
    }
}
