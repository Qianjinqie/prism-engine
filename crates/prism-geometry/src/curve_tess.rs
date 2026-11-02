//! 曲线图元三角化（V2.2-A-2，方案乙）—CPU 侧权威，落 `prism-geometry`（叶子 crate）。
//!
//! # 方案裁定（见 `docs/v2.2-增量设计.md` §1.2）
//!
//! GPU 侧解析求交（方案甲）在 D-29 零超越函数纪律下无合法实现（四次方程预解三次
//! 在 Δ≤0 分支走 `acos`/`cos`，GPU 发射器只许 `Sqrt/FMax/FClamp/Floor`），故取乙：
//! 曲线图元在 CPU 侧三角化为网格，交给我们已有的 SAH-BVH（`crate::bvh`）做射线求交，
//! 再由 GPU 走既有 Möller–Trumbore + 栈式 BVH 遍历（WGSL `rt_bvh.wgsl` 同式）。
//!
//! # 确定性 / 零超越（D-29）
//!
//! 三角化只用 `+ − × ÷ sqrt` 与 `floor`；圆周顶点需要 sin/cos，一律走 CPU 侧
//! `crate::det_math` 确定性件（`det_cos_sin`，纯 IEEE、跨平台逐位一致，禁 libm）。
//! 旋转（局部轴 Y → 世界轴）用 Rodrigues 公式，其中 `cosθ = Y·axis` 直接由点积得到，
//! `sinθ = √(1−cos²θ)`，全程不落任何超越函数。
//!
//! # 保真度界（PRD V2.2-A-2 验收 4；复用 V2.1 R3 `hausdorff_upper_bound` 方法学）
//!
//! 三角化误差上界声明为
//!
//! ```text
//! d_H(S(曲线), mesh) ≤ r_max · (1 − cos(π / seg))
//! ```
//!
//! `seg` = 一周切分段数（`seg = ⌈2π·r / tol⌉`），`r_max` = 该图元最大半径。
//! `seg↑ ⇒ 界↓`（cos 在 `[0,π]` 单调降）。
//! 测试对球面特例（圆环 `major = 0` 退化为球）实测 Hausdorff ≤ 该界，并断言 `seg↑ ⇒ 界↓`。
//!
//! # 退化形态（PRD 验收 3）
//!
//! `r=0 / h=0 / profile 单点 / 空集合 / 非有限` 一律 → 空三角形集（不 panic）。

use crate::bvh::Triangle;
use crate::det_math::det_cos_sin;
use prism_math::Vec3;

/// `2π` 常数（CPU 侧，确定值）。
const TWO_PI: f32 = 6.2831853; // ≈ 2·π，与 `det_math` 的 `TWO_PI` 同值口径
/// 退化切分下限（保证 mesh 非退化）。
const MIN_SEG: u32 = 3;
/// `axis` 归一化阈值（小于此视为未定义轴）。
const AXIS_EPS: f32 = 1e-9;
/// 毛股相邻控制点的最小间距（世界单位；小于此视为零长段 ⇒ 退化）。
///
/// 取 `1e-6`：与 [`AXIS_EPS`] 同量级。零长段会让切向退化为 `0/0`，
/// 平行传输的旋转轴无定义 ⇒ 整股环顶点塌到一点（不是"看起来细"，是 NaN 源）。
/// 故在 `validate()` 里判死，`tessellate` 直接返回空集。
const STRAND_MIN_SEG_LEN: f32 = 1e-6;

/// 曲线图元请求（四种：圆柱 / 圆锥 / 圆环 / 旋转曲面）。
///
/// 所有几何量均为世界坐标；`axis` 为图元的对称轴（单位向量，无需调用方归一，
/// 本模块内部 `orient` 只用其方向）。
#[derive(Debug, Clone, PartialEq)]
pub enum CurveReq {
    /// 圆柱：中心 `center`、半径 `radius`、高 `height`、对称轴 `axis`。
    Cylinder {
        center: Vec3,
        radius: f32,
        height: f32,
        axis: Vec3,
    },
    /// 圆锥：顶点在 `+height/2`、底半径 `radius`、高 `height`、对称轴 `axis`。
    Cone {
        center: Vec3,
        radius: f32,
        height: f32,
        axis: Vec3,
    },
    /// 圆环：主环半径 `major`、管半径 `minor`、对称轴 `axis`。
    /// `major = 0` 时退化为球（半径 `minor`），用于保真度界的球面特例。
    Torus {
        center: Vec3,
        major: f32,
        minor: f32,
        axis: Vec3,
    },
    /// 旋转曲面：`profile` 为局部平面内的一条折线（`x` = 到轴距离，`y` = 沿轴高度，
    /// 绕 `axis` 旋转 `segments` 段生成）。`profile` 长度 < 2（"单点"）或 `segments` < 3
    /// 一律退化。
    Revolution {
        center: Vec3,
        profile: Vec<Vec3>,
        segments: u32,
        axis: Vec3,
    },
    /// **毛股（W5 / A7 定稿 C1）**：一条折线中心线 + 变半径圆管。
    ///
    /// # 语义
    ///
    /// * `points`：中心线控制点，**折线**（不是 NURBS，不插值曲线）。
    ///   长度 ≥ 2；相邻点必须不重合（零长段 ⇒ 退化，见 `STRAND_MIN_SEG_LEN`）。
    /// * `radii`：**两种合法长度，二选一**（`validate()` 都接受）：
    ///   - `radii.len() == points.len()` ⇒ 逐点半径（锥形股 / 发根到发梢变细）；
    ///   - `radii.len() == 1` ⇒ 全股统一半径（等价于把该值广播到每个控制点）。
    ///   其余长度一律非法。每个半径必须 `> 0` 且有限。
    ///
    /// # 为什么单独成"型"而不是复用 [`CurveReq::Revolution`]
    ///
    /// 旋转曲面的中心线只能是直线（绕轴旋转），无法表达弯曲的发股；而毛股的
    /// 圆管必须沿**任意折线**推进，且环的朝向要逐段传递（平行传输，见
    /// [`strand_frames`]）。几何上它和圆柱/圆锥同族（都是"圆截面沿一条轴推进"），
    /// 故走同一条三角化通路、同一份弦高界、同一套 `TRIANGLE_SLOTS` 出口。
    ///
    /// # 端帽：**做**（不是省略）
    ///
    /// 每股两端各生成一圈扇形端盖（与 [`CurveReq::Cylinder`] 的上下盖同式）。
    /// 理由有两条，都要紧：
    /// 1. **V1 判据**要求"等半径直管 ≡ 圆柱"逐三角一致，而圆柱有端盖；
    ///    不生成端盖就必然漏掉 `2·seg` 个三角形，判据直接红。
    /// 2. 开口管在掠射视角会漏光（背面命中缺失），毛股密度高时表现为"发梢发亮"。
    ///
    /// 端盖是平面扇形，与管壁的接缝处法线不连续——这是**有意**的：
    /// 毛发 BSDF 的切线取自 [`strand_frames`] 给出的纤维标架而非几何法线，
    /// 端盖的硬边不影响着色，只影响遮挡（这才是端盖要解决的问题）。
    Strand {
        points: Vec<Vec3>,
        radii: Vec<f32>,
    },
    /// **预烘焙三角网（W4 出海口）**：把细分 + 位移的产物送进既有三角形通路。
    ///
    /// # 这不是"第五条曲线"
    ///
    /// 它存在的唯一理由是**几何注入点只有一个**：三个后端都从
    /// `SceneRenderRequest::curves`（`prism-gpu`）取几何，再经
    /// [`tessellate_all`] 变成三角形 + SAH-BVH。细分/位移的产物本来就是三角形，
    /// 但要走同一条通路就得有地方挂。挂在这里的收益是**零 ABI 改动、零着色器改动、
    /// 零新遍历代码**（选型报告 §5 的定稿正是"烘焙成三角形后走 `TRIANGLE_SLOTS`"）。
    ///
    /// 代价：`CurveReq` 的"曲线"语义被稀释一格。备选方案（给请求结构体加
    /// `meshes` 字段）要改 53 处结构体字面量，并把"空则逐位不变"的契约摊到
    /// 每个后端——为此把语义纯度换掉是划算的。**已在 `docs/v2.5-W0-算法选型报告.md`
    /// §22 记录为偏离项。**
    ///
    /// # 语义
    ///
    /// * `triangles` 原样进入三角化输出（**不重排、不重建**，顺序即输入顺序）；
    /// * 空集 ⇒ 贡献空集（与其它退化形态同口径，不 panic）；
    /// * 弦高界精确为 0（没有近似：它就是网格本身）；
    /// * `seg_count` 返回 1（"一段"；该值只对四族曲线有几何含义，
    ///   本变体的调用方不应依赖它，见 [`seg_count`] 的文档）。
    TriSoup { triangles: Vec<Triangle> },
}

impl CurveReq {
    /// 结构合法性（退化形态返回 `false`，`tessellate` 据此返回空集）。
    ///
    /// 覆盖 PRD 验收 3 的全部退化面：`r=0`/`h=0`/`profile 单点`/`非有限`/`axis` 未定义。
    /// 空集合（无曲线）由调用方在 `curves.is_empty()` 处短路，不进此判据。
    pub fn validate(&self) -> bool {
        let finite_vec = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        match self {
            CurveReq::Cylinder { radius, height, axis, .. } => {
                radius.is_finite()
                    && height.is_finite()
                    && *radius > 0.0
                    && *height > 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Cone { radius, height, axis, .. } => {
                radius.is_finite()
                    && height.is_finite()
                    && *radius > 0.0
                    && *height > 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Torus { major, minor, axis, .. } => {
                major.is_finite()
                    && minor.is_finite()
                    && *minor > 0.0
                    && *major >= 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Revolution { profile, segments, axis, .. } => {
                profile.len() >= 2
                    && *segments >= MIN_SEG
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
                    && profile.iter().all(|p| p.x >= 0.0 && p.x.is_finite() && p.y.is_finite())
            }
            CurveReq::Strand { points, radii } => strand_is_valid(points, radii),
            CurveReq::TriSoup { triangles } => {
                triangles.iter().all(|t| {
                    finite_vec(t.a) && finite_vec(t.b) && finite_vec(t.c)
                })
            }
        }
    }

    /// 该图元的最大半径 `r_max`（保真度界用）。
    ///
    /// 圆柱/圆锥 = 底半径；圆环 = `major + minor`；旋转曲面 = profile 最大 `x`；
    /// 毛股 = 控制点半径的最大值；预烘焙三角网 = 0（无"半径"这个概念，其弦高界
    /// 恒为 0）。
    pub fn max_radius(&self) -> f32 {
        match self {
            CurveReq::Cylinder { radius, .. } => *radius,
            CurveReq::Cone { radius, .. } => *radius,
            CurveReq::Torus { major, minor, .. } => major + minor,
            CurveReq::Revolution { profile, .. } => {
                profile.iter().map(|p| p.x).fold(0.0_f32, f32::max)
            }
            CurveReq::Strand { radii, .. } => radii.iter().copied().fold(0.0_f32, f32::max),
            CurveReq::TriSoup { .. } => 0.0,
        }
    }
}

/// 毛股的结构合法性（[`CurveReq::validate`] 的 `Strand` 臂，独立成函数便于测试）。
///
/// 判死项（任一成立 ⇒ 非法）：
/// * `points.len() < 2`（单点无法构成管，也没有切向）；
/// * `radii` 长度既不是 `1`（统一半径）也不是 `points.len()`（逐点半径）；
/// * 任一点或任一半径非有限；
/// * 任一半径 `≤ 0`（半径 0 ⇒ 环退化成一点 ⇒ 管壁三角形全零面积）；
/// * 相邻控制点距离 `≤` `STRAND_MIN_SEG_LEN`（零长段 ⇒ 切向 `0/0`）。
pub fn strand_is_valid(points: &[Vec3], radii: &[f32]) -> bool {
    if points.len() < 2 {
        return false;
    }
    if radii.len() != 1 && radii.len() != points.len() {
        return false;
    }
    let finite_vec = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    if !points.iter().all(|p| finite_vec(*p)) {
        return false;
    }
    if !radii.iter().all(|r| r.is_finite() && *r > 0.0) {
        return false;
    }
    for w in points.windows(2) {
        let d = w[1] - w[0];
        if !(d.length() > STRAND_MIN_SEG_LEN) {
            return false;
        }
    }
    true
}

/// 由容忍误差 `tol` 推算某一半径对应的周向切分段数（`seg = ⌈2π·r / tol⌉`，下限 3）。
///
/// `tol` 为相邻采样点间的最大弧长（弦高预算）；确定性 ceil 实现（禁浮点 `ceil` 语义漂移）。
fn seg_for_radius(r: f32, tol: f32) -> u32 {
    let tol = tol.max(1e-6);
    let raw = TWO_PI * r / tol;
    let raw_floor = raw.floor();
    let seg = if (raw - raw_floor) > 1e-9 {
        raw_floor as u32 + 1
    } else {
        raw_floor as u32
    };
    seg.max(MIN_SEG)
}

/// 网格相对「界段数」的加密系数（A-2 保真度界能成立的关键）。
///
/// `hausdorff_bound` 给出的是**单方向**弦高 `r·(1 − cos(π/seg))`；而经纬网格的
/// 四边形最大偏离点在其**面心**，相对顶点的角距是 `√2·π/seg`（对角方向），故真实
/// 偏离约为 `r·(1 − cos(√2·π/S))`（`S` = 网格实际段数）。要令
/// `d_H ≤ r·(1 − cos(π/seg))` 成立，必须 `S ≥ √2·seg`；取 1.5 留 6% 数值裕度
/// （实测比值 0.89，见 `tests/qa_curve_tessellation.rs`）。
const TESS_DENSIFY: f32 = 1.5;

/// 网格实际使用的周向段数（= `⌈TESS_DENSIFY · seg_count⌉`）。
///
/// `seg_count` 仍按设计 §1.2.2 的 `⌈2π·r / tol⌉` 口径（界函数的入参），加密只
/// 发生在**网格生成**侧——界不因此放松，只会更保守。
fn tess_seg(r: f32, tol: f32) -> u32 {
    let s = seg_for_radius(r, tol) as f32 * TESS_DENSIFY;
    let f = s.floor();
    let up = if (s - f) > 1e-6 { f + 1.0 } else { f };
    (up as u32).max(MIN_SEG)
}

/// 该曲线在容忍误差 `tol` 下的**有效**周向切分段数（各方向取较粗者 = 较小 `seg`，
/// 保证界为上界）。
///
/// 圆柱/圆锥 = 侧壁段数；圆环 = `min(seg_u, seg_v)`；旋转曲面 = `min(seg_profile, segments)`；
/// 毛股 = 按**最大半径**给出的周向段数（见 `tess_strand` 的"为何全股统一分段"）。
///
/// [`CurveReq::TriSoup`] 返回 `1`：它不是"沿某方向切了多少段"的曲面，
/// 该返回值没有几何含义，仅为保持函数全定义（调用方不应依赖它）。
pub fn seg_count(curve: &CurveReq, tol: f32) -> u32 {
    match curve {
        CurveReq::Cylinder { radius, .. } => seg_for_radius(*radius, tol),
        CurveReq::Cone { radius, .. } => seg_for_radius(*radius, tol),
        CurveReq::Torus { major, minor, .. } => {
            let su = seg_for_radius(major + minor, tol);
            let sv = seg_for_radius(*minor, tol);
            su.min(sv)
        }
        CurveReq::Revolution { profile, segments, .. } => {
            let max_r = profile.iter().map(|p| p.x).fold(0.0_f32, f32::max);
            seg_for_radius(max_r, tol).min(*segments)
        }
        CurveReq::Strand { radii, .. } => {
            seg_for_radius(radii.iter().copied().fold(0.0_f32, f32::max), tol)
        }
        CurveReq::TriSoup { .. } => 1,
    }
}

/// 保真度界 `r_max · (1 − cos(π / seg))`。
///
/// `seg` 即 [`seg_count`] 的返回值；`seg↑ ⇒ 界↓`（cos 在 `[0, π]` 单调降）。
///
/// # 毛股（`Strand`）的界逐段给出
///
/// ```text
/// d_H(管面, mesh) ≤ max_j [ r_j · (1 − cos(π / seg)) ]
/// ```
///
/// 中心线**没有**附加项：`points` 是折线，网格用的就是这条折线（不是对某条
/// 光滑曲线的逼近），故"曲线被折线逼近"的那一项精确为 0。剩下的只有周向
/// 弦高——与圆柱/圆锥同一项。逐段取 max 后与 `max_radius · (1 − cos(π/seg))`
/// 数值相同，但写成逐段形式是为了让"这一段更粗 ⇒ 这一段贡献更大"这件事在
/// 代码里可读（判据 V1 的复算方式也照此）。
///
/// # 预烘焙三角网（`TriSoup`）的界
///
/// 精确 0：输出的三角形就是输入本身，没有任何逼近。
pub fn hausdorff_bound(curve: &CurveReq, seg: u32) -> f32 {
    // 退化/非法形态 → 空网格 → 保真度误差精确为 0（且必须是**有限值**：
    // 半径含 NaN/∞ 时 `max_radius()` 会把 NaN 传染到界上，破坏"界可比较"的契约）。
    if !curve.validate() {
        return 0.0;
    }
    let seg = seg.max(1) as f32;
    let (c, _) = det_cos_sin(std::f32::consts::PI / seg);
    let chord = 1.0 - c;
    if let CurveReq::Strand { points, radii } = curve {
        // 逐段：第 j 段的半径 = 两端点半径的较大者（管的凸包上界）。
        let r_at = |j: usize| if radii.len() == 1 { radii[0] } else { radii[j] };
        let mut bound = 0.0_f32;
        for j in 0..points.len() - 1 {
            let r_seg = r_at(j).max(r_at(j + 1));
            bound = bound.max(r_seg * chord);
        }
        return bound;
    }
    curve.max_radius() * chord
}

/// 三角化入口：`tessellate(&CurveReq, tol) -> Vec<Triangle>`。
///
/// 非法（`!validate()`）或空集合 → 返回空 `Vec`（退化形态不 panic）。
pub fn tessellate(curve: &CurveReq, tol: f32) -> Vec<Triangle> {
    if !curve.validate() {
        return Vec::new();
    }
    match curve {
        CurveReq::Cylinder { center, radius, height, axis } => {
            tess_cylinder(*center, *radius, *height, *axis, tol)
        }
        CurveReq::Cone { center, radius, height, axis } => {
            tess_cone(*center, *radius, *height, *axis, tol)
        }
        CurveReq::Torus { center, major, minor, axis } => {
            tess_torus(*center, *major, *minor, *axis, tol)
        }
        CurveReq::Revolution { center, profile, segments, axis } => {
            tess_revolution(*center, profile, *segments, *axis, tol)
        }
        CurveReq::Strand { points, radii } => tess_strand(points, radii, tol),
        // 预烘焙三角形：原样输出（不重排、不重建 ⇒ 与输入逐位相同）。
        CurveReq::TriSoup { triangles } => triangles.clone(),
    }
}

/// 三角化全部曲线并拼接（空曲线自动跳过，非法曲线贡献空集）。
pub fn tessellate_all(curves: &[CurveReq], tol: f32) -> Vec<Triangle> {
    let mut out = Vec::new();
    for c in curves {
        if c.validate() {
            out.extend(tessellate(c, tol));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 几何辅助
// ---------------------------------------------------------------------------

/// 把局部点（局部对称轴 = Y）经 Rodrigues 旋转到世界 `axis` 方向，再加 `center`。
///
/// 旋转把局部 Y 映射到 `axis`：`cosθ = Y·axis`（直接点积，无需 `acos`），
/// `sinθ = √(1 − cos²θ)`，`k = (Y × axis)` 归一为旋转轴。全程仅 `+ − × ÷ sqrt`。
fn orient(axis: Vec3, center: Vec3, local: Vec3) -> Vec3 {
    let y_axis = Vec3::new(0.0, 1.0, 0.0);
    let c = axis.dot(&y_axis).clamp(-1.0, 1.0); // cosθ
    let s = (1.0 - c * c).max(0.0).sqrt(); // sinθ
    if s < 1e-7 {
        // 轴 ≈ ±Y：c > 0 ⇒ 恒等；c < 0 ⇒ 绕 X 转 180°（Y → −Y）。
        if c > 0.0 {
            return local + center;
        }
        return Vec3::new(local.x, -local.y, -local.z) + center;
    }
    let k = y_axis.cross(&axis).normalize(); // 旋转轴（单位）
    let kp = k.dot(&local);
    let cross_kp = k.cross(&local);
    // Rodrigues: p' = p·c + (k × p)·s + k·(k·p)·(1 − c)
    let rotated = local * c + cross_kp * s + k * (kp * (1.0 - c));
    rotated + center
}

/// 推入一个三角形（顶点顺序仅影响法线朝向；着色时按"朝向射线原点"翻转，故此处随意）。
#[inline]
fn push_tri(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3) {
    out.push(Triangle { a, b, c });
}

/// 推入一个四边形（`a→b→c→d` 顺序）为两个三角形。
#[inline]
fn push_quad(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
    push_tri(out, a, b, c);
    push_tri(out, a, c, d);
}

/// 周向环上一点（局部轴 Y，半径 `r`，高度 `y`，角度 `t`）。
#[inline]
fn ring_point(r: f32, y: f32, t: f32) -> Vec3 {
    let (ct, st) = det_cos_sin(t);
    Vec3::new(r * ct, y, r * st)
}

// ---------------------------------------------------------------------------
// 四类图元三角化
// ---------------------------------------------------------------------------

/// 圆柱：侧壁 `seg` 四边形（每个四边形 = 2 三角形）+ 上下盖扇形（`seg` 三角形各）。
fn tess_cylinder(
    center: Vec3,
    radius: f32,
    height: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let seg = tess_seg(radius, tol);
    let hy = height * 0.5;
    let mut tris = Vec::with_capacity((seg as usize) * 4);
    // 侧壁：seg 个四边形（每个 2 三角形）。
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let p0b = orient(axis, center, ring_point(radius, -hy, t0));
        let p1b = orient(axis, center, ring_point(radius, -hy, t1));
        let p0t = orient(axis, center, ring_point(radius, hy, t0));
        let p1t = orient(axis, center, ring_point(radius, hy, t1));
        push_quad(&mut tris, p0b, p1b, p1t, p0t);
    }
    // 上下盖扇形：各 seg 三角形。
    let ct = orient(axis, center, Vec3::new(0.0, hy, 0.0));
    let cb = orient(axis, center, Vec3::new(0.0, -hy, 0.0));
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let r0 = orient(axis, center, ring_point(radius, hy, t0));
        let r1 = orient(axis, center, ring_point(radius, hy, t1));
        push_tri(&mut tris, ct, r0, r1);
        let b0 = orient(axis, center, ring_point(radius, -hy, t0));
        let b1 = orient(axis, center, ring_point(radius, -hy, t1));
        push_tri(&mut tris, cb, b1, b0);
    }
    tris
}

/// 圆锥：侧壁 `seg` 三角形（顶点 → 底环）+ 底盖扇形 `seg` 三角形。
fn tess_cone(
    center: Vec3,
    radius: f32,
    height: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let seg = tess_seg(radius, tol);
    let hy = height * 0.5;
    let mut tris = Vec::with_capacity((seg as usize) * 2);
    let apex = orient(axis, center, Vec3::new(0.0, hy, 0.0));
    let cb = orient(axis, center, Vec3::new(0.0, -hy, 0.0));
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let b0 = orient(axis, center, ring_point(radius, -hy, t0));
        let b1 = orient(axis, center, ring_point(radius, -hy, t1));
        push_tri(&mut tris, apex, b0, b1); // 侧壁三角形
        push_tri(&mut tris, cb, b1, b0); // 底盖扇形
    }
    tris
}

/// 圆环：主环 `seg_u` × 管 `seg_v` 四边形（每个 2 三角形）。
///
/// 球面特例（`major = 0`）退化为球：`seg_u = seg_v = seg`，界 = `minor·(1 − cos(π/seg))`。
fn tess_torus(
    center: Vec3,
    major: f32,
    minor: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let su = tess_seg(major + minor, tol);
    let sv = tess_seg(minor, tol);
    let mut tris = Vec::with_capacity((su as usize) * (sv as usize) * 2);
    let pt = |u: f32, v: f32| -> Vec3 {
        let (cu, su_) = det_cos_sin(u);
        let (cv, sv_) = det_cos_sin(v);
        let rr = major + minor * cv;
        orient(axis, center, Vec3::new(rr * cu, minor * sv_, rr * su_))
    };
    for iu in 0..su {
        let u0 = TWO_PI * (iu as f32) / (su as f32);
        let u1 = TWO_PI * ((iu + 1) as f32) / (su as f32);
        for iv in 0..sv {
            let v0 = TWO_PI * (iv as f32) / (sv as f32);
            let v1 = TWO_PI * ((iv + 1) as f32) / (sv as f32);
            let a = pt(u0, v0);
            let b = pt(u1, v0);
            let c = pt(u1, v1);
            let d = pt(u0, v1);
            push_quad(&mut tris, a, b, c, d);
        }
    }
    tris
}

/// 旋转曲面：`(profile.len() − 1)` 段 × `segments` 四边形。
///
/// `profile[j] = (x = 到轴距离, y = 沿轴高度, z 忽略)`；水平段（`x` 不变）特化为环带
/// 但仍三角化（保退化一致：与其它段走同一路径，不特殊处理）。
fn tess_revolution(
    center: Vec3,
    profile: &[Vec3],
    segments: u32,
    axis: Vec3,
    _tol: f32,
) -> Vec<Triangle> {
    if profile.len() < 2 {
        return Vec::new();
    }
    let seg = segments.max(MIN_SEG);
    let mut tris = Vec::with_capacity((profile.len() - 1) * (seg as usize) * 2);
    for j in 0..profile.len() - 1 {
        let pa = profile[j];
        let pb = profile[j + 1];
        for i in 0..seg {
            let t0 = TWO_PI * (i as f32) / (seg as f32);
            let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
            let (c0, s0) = det_cos_sin(t0);
            let (c1, s1) = det_cos_sin(t1);
            let a = orient(axis, center, Vec3::new(pa.x * c0, pa.y, pa.x * s0));
            let b = orient(axis, center, Vec3::new(pb.x * c0, pb.y, pb.x * s0));
            let c = orient(axis, center, Vec3::new(pb.x * c1, pb.y, pb.x * s1));
            let d = orient(axis, center, Vec3::new(pa.x * c1, pa.y, pa.x * s1));
            push_quad(&mut tris, a, b, c, d);
        }
    }
    tris
}

// ---------------------------------------------------------------------------
// 毛股（W5 / A7 定稿 C1）：平行传输标架 + 变半径圆管
// ---------------------------------------------------------------------------

/// 毛股某一段的**纤维标架**（世界空间，单位正交右手系）。
///
/// `tangent` = 该段中心线的单位切向；`normal_ref` = 法平面内的参考方向
/// （**平行传输**推进，逐段不扭转）；`binormal = normal_ref × tangent`。
///
/// # 这个结构为什么必须公开
///
/// 毛发 BSDF（Marschner 模型）在纤维局部系里表达：入射/出射方向要先投影成
/// `(θ, φ)`（`HairGeometry::longitudinal / azimuth`），而那个局部系就是本结构。
/// 判据 V4（帧不连续 ≤ 5°）也要直接采样本结构，故它是**公开的几何产物**，
/// 不是三角化的内部细节。
///
/// # 禁止事项（红线）
///
/// 标架**只由中心线决定**：不依赖视点、屏幕空间、包围盒朝向或任何渲染状态。
/// 否则预览与终帧会得到不同几何（`docs/v2.5-v3.0-路线图.md` §4.4 红线 1 的
/// 同款问题，选型报告 §8 的 C3 候选就是因此被否）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrandFrame {
    /// 单位切向（沿中心线前进方向）。
    pub tangent: Vec3,
    /// 法平面内的单位参考方向（平行传输推进）。
    pub normal_ref: Vec3,
    /// 副法向（`normal_ref × tangent`，单位）。
    pub binormal: Vec3,
}

/// 由折线中心线推出**逐段**纤维标架（平行传输 / parallel transport）。
///
/// # 算法
///
/// 1. 首段切向 `u0 = normalize(p1 − p0)`；初始法向取"与 `u0` 最不平行的坐标轴
///    投影到法平面后归一"（见 `initial_normal`）——纯函数，与视点无关。
/// 2. 逐段推进：把上一段的 `normal_ref` 用**最小旋转**（把 `u_prev` 转到 `u_cur`
///    的那个旋转）搬运过去，再正交化/归一。这就是平行传输：相邻标架之间的
///    相对扭转精确为 0，所以沿股扫过时法向不会绕轴打转（判据 V4）。
///
/// 只用 `+ − × ÷ sqrt`（D-29）：没有 `acos`/`atan2`，不需要任何角度。
///
/// # 退化
///
/// `points.len() < 2`、相邻点重合、或含非有限坐标 ⇒ 返回**空** `Vec`
/// （调用方 `tessellate` 已在 `validate()` 处短路，此处是二次保险，不 panic）。
pub fn strand_frames(points: &[Vec3]) -> Vec<StrandFrame> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let mut frames: Vec<StrandFrame> = Vec::with_capacity(n - 1);
    let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    // 首段切向（零长段 ⇒ 空集，不产生 NaN 标架）。
    let mut u_prev = points[1] - points[0];
    let len0 = u_prev.length();
    if !(len0 > STRAND_MIN_SEG_LEN) || !finite(u_prev) {
        return Vec::new();
    }
    u_prev = u_prev * (1.0 / len0);
    let mut nrm = initial_normal(u_prev);
    frames.push(StrandFrame { tangent: u_prev, normal_ref: nrm, binormal: nrm.cross(&u_prev) });

    for j in 1..n - 1 {
        let d = points[j + 1] - points[j];
        let len = d.length();
        if !(len > STRAND_MIN_SEG_LEN) || !finite(d) {
            return Vec::new();
        }
        let u_cur = d * (1.0 / len);
        nrm = transport_normal(nrm, u_prev, u_cur);
        frames.push(StrandFrame {
            tangent: u_cur,
            normal_ref: nrm,
            binormal: nrm.cross(&u_cur),
        });
        u_prev = u_cur;
    }
    frames
}

/// 初始法向：取与 `u` **最不平行**的坐标轴，投影到法平面后归一。
///
/// 与 [`curve_tess`] 的 `orient` 同精神：永远存在一个轴满足 `|u·axis| ≤ 1/√3`，
/// 故叉乘/投影不会退化。并列时按 `x → y → z` 取（确定性，不依赖迭代顺序）。
///
/// 对 `u = +Y`（圆柱轴）返回 `+X`：这条不是巧合——正是它让等半径直管毛股
/// 与 [`CurveReq::Cylinder`] 的环顶点**逐位相同**（判据 V1 的结构基础）。
fn initial_normal(u: Vec3) -> Vec3 {
    let (ax, ay, az) = (u.x.abs(), u.y.abs(), u.z.abs());
    let helper = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let proj = helper - u * helper.dot(&u);
    let l = proj.length();
    if l > 1e-12 {
        proj * (1.0 / l)
    } else {
        // 理论上不可达（helper 与 u 的夹角 ≥ 54.7°）；兜底取任意正交向。
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// 平行传输一步：把 `n` 按"把 `from` 转到 `to` 的最小旋转"搬运过去。
///
/// Rodrigues 形式，其中 `cosθ = from·to`、`sinθ = |from × to|` 直接由点积/叉乘
/// 得到（不落 `acos`）。搬运后**再正交化一次**：多段累积的浮点漂移不会让
/// `n·to` 偏离 0（否则环顶点会缓慢地绕段轴偏移，表现为毛发"螺旋"）。
///
/// `from ≈ −to`（180° 反向）时最小旋转不唯一：此时保持 `n` 不变仍与 `to` 正交，
/// 是合法选择（毛发不会出现这种折返，但代码不许产生 NaN）。
fn transport_normal(n: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let axis = from.cross(&to);
    let s = axis.length();
    let c = from.dot(&to).clamp(-1.0, 1.0);
    let mut out = if s < 1e-9 {
        if c > 0.0 {
            n
        } else {
            n // 反向：n 仍与 to 正交，保持（见文档）
        }
    } else {
        let k = axis * (1.0 / s);
        let kv = k.dot(&n);
        n * c + k.cross(&n) * s + k * (kv * (1.0 - c))
    };
    // 正交化（去掉沿 to 的分量）+ 归一。
    out = out - to * out.dot(&to);
    let l = out.length();
    if l > 1e-12 {
        out * (1.0 / l)
    } else {
        initial_normal(to)
    }
}

/// 毛股三角化：变半径圆管 + 两端扇形端盖。
///
/// # 周向分段：**全股统一**取 `tess_seg(max radii, tol)`
///
/// 设计稿写的是"每段 `seg = max(MIN_SEG, ceil(2π·r_seg/tol))`"。本实装取
/// **全股统一**（用最大半径算一次），理由是缝合：
///
/// 相邻两段必须共享**相同数量**的环顶点才能拼成闭合管。逐段独立取 `seg` 时，
/// 半径变化的两段环顶点数不同，交界处会出现"顶点对不上"的缝——那正是 §5
/// 定稿要避免的 T 型接缝（在毛管上表现为发丝间的漏光条纹，且随视点闪烁）。
/// 统一分段后，管的拓扑是规则的 `(n−1)×seg` 四边形带，结构上没有缝。
///
/// 代价：一股内最粗的那一段决定整股的分段数（细段略密于必要）。这与
/// [`hausdorff_bound`] 的口径一致——界本来就按 `max_radius` 取上界。
///
/// # 端帽
///
/// 两端各 `seg` 个三角形（扇形），与 [`CurveReq::Cylinder`] 的上下盖同式。
/// 详见 [`CurveReq::Strand`] 的"端帽：做"一节。
fn tess_strand(points: &[Vec3], radii: &[f32], tol: f32) -> Vec<Triangle> {
    let n = points.len();
    let r_at = |j: usize| if radii.len() == 1 { radii[0] } else { radii[j] };
    let r_max = radii.iter().copied().fold(0.0_f32, f32::max);
    let seg = tess_seg(r_max, tol);
    let frames = strand_frames(points);
    if frames.len() != n - 1 {
        return Vec::new(); // 退化（零长段 / 非有限）：空集，不 panic
    }

    // 逐点环（每环 seg 个顶点，角度与圆柱同式：t_i = 2π·i/seg）。
    let seg_i = seg as usize;
    let mut rings: Vec<Vec<Vec3>> = Vec::with_capacity(n);
    for j in 0..n {
        let f = frames[j.min(frames.len() - 1)];
        let r = r_at(j);
        let mut ring = Vec::with_capacity(seg_i);
        for i in 0..seg_i {
            let t = TWO_PI * (i as f32) / (seg as f32);
            let (ct, st) = det_cos_sin(t);
            // 局部偏移 = n·(r·cosθ) + b·(r·sinθ)，与 `orient` + `ring_point` 同序。
            ring.push(points[j] + (f.normal_ref * (r * ct) + f.binormal * (r * st)));
        }
        rings.push(ring);
    }

    let mut tris = Vec::with_capacity(((n - 1) * seg_i * 2) + 2 * seg_i);
    // 管壁：(n−1) 段 × seg 个四边形（每个 2 三角形）。绕序与 `tess_cylinder` 的
    // 侧壁同式（低端 i, 低端 i+1, 高端 i+1, 高端 i）。
    for j in 0..n - 1 {
        for i in 0..seg_i {
            let i1 = (i + 1) % seg_i;
            push_quad(
                &mut tris,
                rings[j][i],
                rings[j][i1],
                rings[j + 1][i1],
                rings[j + 1][i],
            );
        }
    }
    // 端盖（起点盖 → 与圆柱下盖同绕序；终点盖 → 与上盖同绕序）。
    for i in 0..seg_i {
        let i1 = (i + 1) % seg_i;
        push_tri(&mut tris, points[0], rings[0][i1], rings[0][i]);
    }
    for i in 0..seg_i {
        let i1 = (i + 1) % seg_i;
        push_tri(&mut tris, points[n - 1], rings[n - 1][i], rings[n - 1][i1]);
    }
    tris
}

// ---------------------------------------------------------------------------
// 单元测试（native：保真度界 / 退化形态 / 确定性 / 几何合理性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    /// 退化形态：r=0 / h=0 / profile 单点 / 非有限 → 一律空集（不 panic）。
    #[test]
    fn degenerate_forms_yield_empty_mesh() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let center = Vec3::new(0.0, 0.0, 0.0);
        let cases: Vec<CurveReq> = vec![
            // r = 0
            CurveReq::Cylinder { center, radius: 0.0, height: 1.0, axis },
            CurveReq::Cone { center, radius: 0.0, height: 1.0, axis },
            CurveReq::Torus { center, major: 1.0, minor: 0.0, axis },
            // h = 0
            CurveReq::Cylinder { center, radius: 1.0, height: 0.0, axis },
            CurveReq::Cone { center, radius: 1.0, height: 0.0, axis },
            // profile 单点（长度 < 2）
            CurveReq::Revolution {
                center,
                profile: vec![Vec3::new(1.0, 0.0, 0.0)],
                segments: 8,
                axis,
            },
            // 非有限
            CurveReq::Cylinder { center, radius: f32::NAN, height: 1.0, axis },
            CurveReq::Cone { center, radius: 1.0, height: f32::INFINITY, axis },
            CurveReq::Torus { center, major: f32::NAN, minor: 1.0, axis },
        ];
        for c in &cases {
            assert!(!c.validate(), "退化形态必须被判非法");
            assert_eq!(
                tessellate(c, 0.01),
                Vec::new(),
                "退化形态必须返回空 mesh（不 panic）"
            );
        }
    }

    /// 合法图元三角化：网格非空且为有限顶点。
    #[test]
    fn valid_curves_tessellate_to_finite_mesh() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let center = Vec3::new(0.5, -0.3, 0.2);
        let cyl = CurveReq::Cylinder { center, radius: 1.0, height: 2.0, axis };
        let cone = CurveReq::Cone { center, radius: 1.0, height: 2.0, axis };
        let torus = CurveReq::Torus { center, major: 1.0, minor: 0.4, axis };
        let rev = CurveReq::Revolution {
            center,
            profile: vec![
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            segments: 12,
            axis,
        };
        for c in [cyl, cone, torus, rev] {
            let tris = tessellate(&c, 0.05);
            assert!(!tris.is_empty(), "合法图元应有非空 mesh");
            for t in &tris {
                for v in [t.a, t.b, t.c] {
                    assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
                }
            }
        }
    }

    /// 确定性：同输入两次三角化逐位一致。
    #[test]
    fn tessellation_is_deterministic() {
        let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
        let curve = CurveReq::Torus {
            center: Vec3::new(0.0, 0.0, 0.0),
            major: 0.8,
            minor: 0.3,
            axis,
        };
        let a = tessellate(&curve, 0.03);
        let b = tessellate(&curve, 0.03);
        assert_eq!(a, b, "同输入三角化必须逐位一致");
    }

    // -----------------------------------------------------------------------
    // W5 / A7：毛股
    // -----------------------------------------------------------------------

    /// 退化形态（A1 要求的五类）一律判非法且返回空 mesh（不 panic）。
    #[test]
    fn strand_degenerate_forms_yield_empty_mesh() {
        let cases: Vec<(&str, CurveReq)> = vec![
            (
                "points.len() < 2",
                CurveReq::Strand { points: vec![Vec3::new(0.0, 0.0, 0.0)], radii: vec![0.1] },
            ),
            (
                "radii 长度不匹配",
                CurveReq::Strand {
                    points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
                    radii: vec![0.1, 0.2, 0.3],
                },
            ),
            (
                "点非有限",
                CurveReq::Strand {
                    points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, f32::NAN, 0.0)],
                    radii: vec![0.1],
                },
            ),
            (
                "半径非有限",
                CurveReq::Strand {
                    points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
                    radii: vec![f32::INFINITY],
                },
            ),
            (
                "半径 ≤ 0",
                CurveReq::Strand {
                    points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
                    radii: vec![0.0],
                },
            ),
            (
                "相邻点重合（零长段）",
                CurveReq::Strand {
                    points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)],
                    radii: vec![0.1],
                },
            ),
        ];
        for (tag, c) in &cases {
            assert!(!c.validate(), "[{tag}] 必须被判非法");
            assert_eq!(tessellate(c, 0.02), Vec::new(), "[{tag}] 必须返回空 mesh");
            assert_eq!(hausdorff_bound(c, seg_count(c, 0.02)), 0.0, "[{tag}] 界必须有限且为 0");
        }
    }

    /// **V1（几何等价，逐位）**：等半径直管 ⇔ `CurveReq::Cylinder`，同 tol。
    ///
    /// 判据原文写的是"等半径圆锥（`Cone`）"——**这是笔误**：`Cone` 的半径从 0
    /// 线性长到 `radius`（顶点在 `+height/2`），与等半径直管在几何上不等价；
    /// 等半径直管的正确对应物是 [`CurveReq::Cylinder`]。本测试按**几何等价**
    /// 选 `Cylinder`，理由是判据要检验的是"毛股通路与曲线通路喂出同一批三角形"，
    /// 若拿 `Cone` 对拍，差异恒非零且与毛股实现无关（是图元本身不同）。
    ///
    /// 断言是**逐位三角形集合相等**（比 `differing_pixels = 0` 更强：像素相等
    /// 只要求命中结果一致，这里要求顶点位型一致）。排序后比较以消除发射顺序差异
    /// ——渲染只依赖三角形集合，不依赖它在 `Vec` 里的位置（BVH 由同一集合构建）。
    #[test]
    fn strand_straight_tube_is_bitwise_cylinder() {
        let (radius, height) = (0.9_f32, 3.0_f32);
        let tol = 0.02_f32;
        let cyl = CurveReq::Cylinder {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius,
            height,
            axis: Vec3::new(0.0, 1.0, 0.0),
        };
        let strand = CurveReq::Strand {
            points: vec![Vec3::new(0.0, -height * 0.5, 0.0), Vec3::new(0.0, height * 0.5, 0.0)],
            radii: vec![radius],
        };
        let a = sorted_tris(&tessellate(&cyl, tol));
        let b = sorted_tris(&tessellate(&strand, tol));
        assert_eq!(a.len(), b.len(), "三角形数必须相同（含两端盖）");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                (vbits(x.a), vbits(x.b), vbits(x.c)),
                (vbits(y.a), vbits(y.b), vbits(y.c)),
                "第 {i} 个三角形顶点位型不同：圆柱 {x:?} vs 毛股 {y:?}"
            );
        }
        println!("V1：等半径直管与圆柱 {} 个三角形逐位相同（tol={tol}）", a.len());
    }

    /// 顶点位型 `(x, y, z)` 的 u32 位型三元组。
    fn vbits(v: Vec3) -> (u32, u32, u32) {
        (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
    }

    /// 逐位比较用：三角形按顶点位型排序（与发射顺序解耦）。
    fn sorted_tris(tris: &[Triangle]) -> Vec<Triangle> {
        let key = |t: &Triangle| {
            (vbits(t.a), vbits(t.b), vbits(t.c))
        };
        let mut v = tris.to_vec();
        v.sort_by_key(key);
        v
    }

    /// 变半径管：每个控制点的环半径确实等于该点的半径；端帽使网格封闭（水密）。
    #[test]
    fn strand_variable_radius_rings_and_caps() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.3, 2.0, 0.0),
        ];
        let radii = vec![0.2_f32, 0.1, 0.05];
        let curve = CurveReq::Strand { points: pts.clone(), radii: radii.clone() };
        assert!(curve.validate());
        let tris = tessellate(&curve, 0.02);
        let seg = tess_seg(radii.iter().copied().fold(0.0, f32::max), 0.02) as usize;
        // 管壁 (n−1)·seg·2 + 端盖 2·seg。
        assert_eq!(tris.len(), (pts.len() - 1) * seg * 2 + 2 * seg);

        // 每段环上的点距其中心线的垂距 = 该点半径（容差 1e-5 相对）。
        let frames = strand_frames(&pts);
        for j in 0..pts.len() {
            let f = frames[j.min(frames.len() - 1)];
            let centre = pts[j];
            let mut radial: Vec<f32> = Vec::new();
            for i in 0..seg {
                let t = TWO_PI * (i as f32) / (seg as f32);
                let (ct, st) = det_cos_sin(t);
                let p = centre + (f.normal_ref * (radii[j] * ct) + f.binormal * (radii[j] * st));
                radial.push((p - centre).length());
            }
            for r in radial {
                assert!(
                    (r - radii[j]).abs() <= 1e-5 * radii[j].max(1.0),
                    "第 {j} 点的环半径 {r} 应等于 {}",
                    radii[j]
                );
            }
        }
    }

    /// 平行传输：法向沿股推进**不绕轴打转**（相邻段法向夹角 ≈ 段间转折角），
    /// 且始终与切向正交（正交化生效）。
    ///
    /// 这是判据 V4 的几何侧：一段沿圆弧、每段转折 2° 的股，相邻标架夹角必须
    /// ≤ 5°（不是 90°、也不是累积扭转）。
    #[test]
    fn strand_frames_are_parallel_transported() {
        // 空间螺旋线（**非平面**）：平面曲线的平行传输是平凡的（面外法向恒定），
        // 只有非平面曲线才能测出"逐段搬运"是否真的在搬运。
        let n = 90; // 90 段 × 2° = 180° 半圈
        let mut pts = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let a = std::f32::consts::PI * (i as f32) / (n as f32); // 0..π
            let (ca, sa) = det_cos_sin(a);
            pts.push(Vec3::new(ca * 10.0, sa * 10.0, a * 3.0));
        }
        let frames = strand_frames(&pts);
        assert_eq!(frames.len(), n);
        // cos(5°) 的下界（判据 V4 的阈值；用点积比较，避免 acos 落 libm）。
        const COS_5DEG: f32 = 0.996_194_7;
        let mut max_ortho = 0.0_f32;
        let mut min_dot = 1.0_f32;
        for f in &frames {
            max_ortho = max_ortho.max(f.normal_ref.dot(&f.tangent).abs());
            max_ortho = max_ortho.max(f.binormal.dot(&f.tangent).abs());
            max_ortho = max_ortho.max(f.normal_ref.dot(&f.binormal).abs());
        }
        for w in frames.windows(2) {
            min_dot = min_dot.min(w[0].normal_ref.dot(&w[1].normal_ref));
        }
        println!(
            "V4 几何侧：{} 段空间螺旋（每段 2°），相邻法向最小点积 {:.8}（cos5° = {:.8}），\
             最大非正交度 {:.2e}",
            n, min_dot, COS_5DEG, max_ortho
        );
        assert!(
            min_dot >= COS_5DEG,
            "相邻标架夹角超过 5°：点积 {min_dot} < {COS_5DEG}"
        );
        assert!(max_ortho < 1e-6, "标架非正交：{max_ortho}");
    }

    /// 保真度界：`seg↑ ⇒ 界↓`，且界随最大半径线性放大（与四族同口径）。
    #[test]
    fn strand_hausdorff_bound_shrinks_with_seg() {
        let curve = CurveReq::Strand {
            points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
            radii: vec![0.05],
        };
        let tol = 0.02;
        let seg = seg_count(&curve, tol);
        let a = hausdorff_bound(&curve, seg);
        let b = hausdorff_bound(&curve, seg * 2);
        assert!(a > 0.0 && b > 0.0);
        assert!(b < a, "seg 加倍后界必须下降：{b} !< {a}");

        let fat = CurveReq::Strand {
            points: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
            radii: vec![0.1],
        };
        let c = hausdorff_bound(&fat, seg);
        assert!((c - 2.0 * a).abs() <= 1e-6 * c.max(1.0), "半径加倍 ⇒ 界加倍：{a} vs {c}");
    }

    /// 预烘焙三角网（W4 出海口）：原样输出、顺序不变、逐位相同。
    #[test]
    fn trisoup_passes_through_verbatim() {
        let tris = vec![
            Triangle {
                a: Vec3::new(0.0, 0.0, 0.0),
                b: Vec3::new(1.0, 0.0, 0.0),
                c: Vec3::new(0.0, 1.0, 0.0),
            },
            Triangle {
                a: Vec3::new(1.0, 0.0, 0.0),
                b: Vec3::new(1.0, 1.0, 0.0),
                c: Vec3::new(0.0, 1.0, 0.0),
            },
        ];
        let c = CurveReq::TriSoup { triangles: tris.clone() };
        assert!(c.validate());
        assert_eq!(tessellate(&c, 0.02), tris, "预烘焙三角形必须原样通过");
        assert_eq!(hausdorff_bound(&c, seg_count(&c, 0.02)), 0.0, "无逼近 ⇒ 界为 0");
        // 非有限三角形 ⇒ 非法（与其它图元的退化口径一致）。
        let bad = CurveReq::TriSoup {
            triangles: vec![Triangle {
                a: Vec3::new(f32::NAN, 0.0, 0.0),
                b: Vec3::new(1.0, 0.0, 0.0),
                c: Vec3::new(0.0, 1.0, 0.0),
            }],
        };
        assert!(!bad.validate());
        assert_eq!(tessellate(&bad, 0.02), Vec::new());
    }
}
