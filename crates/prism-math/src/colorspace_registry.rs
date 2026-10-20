//! T4 色彩空间批量注册表：白点 + 原色 + 传递函数三元组的结构化定义与互转。
//!
//! 语义权威：的分层约定（"ACES 全流程有了，几十种色彩空间/ICC 未铺"）、
//! D-13（调校结果以 golden-file 固化）、D-16（Mini Code）、
//! D-29（确定性数学件，禁 libm）、宪章 C17/C18（全链路确定性 + 错误三段式）。
//!
//! # 与既有色彩模块的关系（共存，不改动）
//! - [`crate::colorspace`]（P0 预留）持有**引擎工作域**用的两块硬编码常量矩阵
//!   （`SRGB_TO_ACESCG` / `ACESCG_TO_SRGB`）与 [`crate::colorspace::ColorSpace`]
//!   枚举；本模块**不修改**该文件一字，只在其之上补"广度"：把任意
//!   （白点, 原色, 传递函数）三元组变成可注册、可查询、可互转的注册表。
//! - **两者口径不同，且是有意的**：本注册表按各规范的**名义白点**注册
//!   （ACES 三兄弟是 D60，其余 SDR 空间是 D65/D50/DCI），因此
//!   `srgb`(D65) → `acescg`(D60) 会**自动走 Bradford 适配**，装配出的矩阵
//!   与 ACES 官方公布的 sRGB→ACEScg 矩阵在 **1e-5** 内一致（见集成测试
//!   `registry_acescg_matches_official_aces_matrix`）——这同时反向验证了
//!   本模块的白点 + 原色 + Bradford 三件套推导正确。
//! - 既有 [`crate::colorspace::SRGB_TO_ACESCG`] 与本注册表装配值相差
//!   **~3.2e-2**（逐元素最大）。既有常量是**拟合矩阵**（行和 = 1 但并非
//!   由原色推导），本注册表给的是**由原色推导的精确值**。二者**不是同一口径**，
//!   本次不改既有文件（边界禁改），差异如实记录在「诚实边界」说明中，
//!   留待裁决是否统一。
//!
//! # ⚠️⚠️⚠️ 警告：既有拟合常量与本注册表**不是同一口径**，绝不可混用 ⚠️⚠️⚠️
//!
//! [`crate::colorspace::SRGB_TO_ACESCG`]（语义明确别名
//! [`crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL`]）是 **Stephen Hill 拟合
//! 矩阵**：行和精确为 1（D65 白点守恒），但它**不是**由 AP1 原色 + Bradford
//! 推导出来的。本注册表 `by_name("acescg")` 给的是**原色推导值**。
//!
//! | 口径 | 常量 / API | 行和 | 用途 | 逐元素最大差 |
//! |---|---|---|---|---|
//! | **拟合** | `colorspace::SRGB_TO_ACESCG`（= `..._FITTED_HILL`） | 精确 1 | 匹配 ACES 官方 IDT、**兼容既有渲染结果** | — |
//! | **推导** | `by_name("acescg").xyz_to_rgb_matrix()` | ≈ 1（Bradford 后） | 色彩学正确、白点 + 原色自洽 | **~3.2e-2** |
//!
//! - **别把推导值塞进期望 ACES 官方 IDT 的管线**（会引入可见色偏）；
//! - **也别用拟合值做色彩科学计算**（白点/原色不自洽）；
//! - **要统一的话，唯一正确方向是「改管线去适配既有结果」而非「偷换常量」**——
//!   替换会静默改变所有既有渲染输出。
//!
//! 差异的**逐元素数值**由 [`legacy_vs_derived_delta`] 报告（不只是一个
//! max 标量），并由测试
//! `legacy_constant_is_a_fitted_matrix_not_a_primaries_derivation` 锁定。
//! **T4 裁决：旧常量原样保留、数值一字不改**（本次只加名字与报告，不换值）。
//!
//! # 数学件（全部自写，零第三方依赖）
//! - [`invert3`]：3×3 求逆用**余子式展开 + 行列式**（伴随矩阵法），
//!   完全展开写死（不做高斯消元——无回溯、无除零抖动、数值可审计）；
//!   `|det| < 1e-12` 判奇异并返回结构化错误。
//! - [`bradford_adaptation`]：Bradford 色度适配矩阵
//!   `M_A = B⁻¹ · diag(B·W_dst / B·W_src) · B`，`B` 取 Bradford 锥体响应
//!   常数，`B⁻¹` 用**同一个 [`invert3`]** 求（不另写公式）。
//! - [`det_atan2`]：LCh 色相角需要 `atan2`，而 D-29 三角件只提供
//!   `sin/cos/tan`。本模块**自带**一份 `atan2`（三次半角归约 + 15 次
//!   Maclaurin），只用 `± × ÷` 与精确 `sqrt`（IEEE 精确运算），
//!   跨平台逐位一致。**已知重复**：将来 D-29 三角件可把它上收到
//!   [`crate::det_trig`]；本次不越界改那个文件。
//! - 幂函数一律走 [`crate::det_pow::det_pow`]（D-29），不用 `powf`
//!   （`powf` 落平台 libm，跨平台尾位不一致）。
//!
//! # 精度口径（诚实声明）
//! - **矩阵代数在 `f64`**：派生、Bradford 适配、求逆全用 `f64`（IEEE 精确
//!   运算，跨平台逐位一致），最后一步才收窄到调用点需要的精度。
//! - **传递函数在 `f32` 求值**：复用 D-29 `det_pow`（f32 进出），
//!   相对误差 ~1e-7。因此 sRGB 曲线往返误差量级 ~1e-7，不是 1e-12。
//! - 公开 API 用 `f64`；需要与既有 `f32` 世界对接时用
//!   [`ColorSpaceDef::rgb_to_xyz_mat3f32`] / [`ColorSpaceDef::to_xyz_vec3`]
//!   两个显式降精度适配器（收窄点唯一、可审计）。
//!
//! # 诚实边界（勿高估）
//! - **不是 ICC 引擎**：无 ICC 解析、无 LUT（shaper/LUT8/LUT16/tobAB）、
//!   无 PCS 往返、无渲染意图（intent）处理、无 profile 互操作。
//!   本模块是**解析式色彩空间**注册表（矩阵 + 曲线），不是 ICC 档案系统。
//! - **色度坐标取自公开规范常数**，精度以规范给出的位数（4~6 位小数）为准；
//!   不确定位数的地方（如 ProPhoto 的 B 原色 `y = 0.000105`）已就地注释说明。
//!   本模块**不**声称达到 colord/ACES 官方参考实现的逐位一致。
//! - **无 gamut 映射 / 无色域裁剪剪裁逻辑**：`from_xyz` 对色域外 XYZ 返回
//!   原始外插值（含负分量），**不**做 clamp。调用方要裁剪请自行处理。
//! - **无 HDR 扩展域标定**：PQ/HLG 在 [`crate::pq`]/[`crate::hlg`] 里，
//!   本注册表**不**把它们注册为条目（它们的"原色 + 白点"语义与 SDR
//!   三元组不同，硬塞进来会失真）。
//! - `ProPhoto RGB` 的 ROMM 曲线是**纯 1.8 幂近似**，ROMM 的趾段（toe）
//!   未建模；`ACEScc` 的负段断点常数系转录自 ACES/SMPTE 规范，未经
//!   官方测试向量逐位校验（见「诚实边界」与集成测试中的往返自洽断言）。
//! - **LCh 的三角函数精度受限**：[`lch_to_lab`] 走 D-29 `cos_sin`（f32 级，
//!   误差 ~1e-7）以保确定性，故 LCh↔Lab 往返误差量级 ~1e-5 而非 1e-15。
//! - **既有 `colorspace` 常量与本注册表不同口径**（见上）。本注册表的矩阵
//!   是「原色推导值」，既有常量是「拟合值」，~3.2e-2 差异属预期，
//!   不应被当作本模块的精度缺陷。

use std::fmt;

use crate::det_pow::det_pow;
use crate::Vec3;

// ===========================================================================
//  3×3 线性代数（自写；行主序 `m[row][col]`）
// ===========================================================================

/// 3×3 颜色矩阵（**行主序**：`m[r][c]` 为第 r 行第 c 列）。
///
/// 与 [`crate::colorspace::Mat3`]（`f32`）同布局但精度更高：本模块的
/// 矩阵代数全在 `f64` 完成，只在交付给既有 `f32` 世界时收窄。
pub type Mat3d = [[f64; 3]; 3];

/// 3×3 单位矩阵。
pub const IDENTITY3: Mat3d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// 3×3 矩阵乘法（行主序；固定求和序 `a[r][0]*b[0][c] + a[r][1]*b[1][c] + a[r][2]*b[2][c]`）。
#[inline]
pub fn mat3_mul(a: &Mat3d, b: &Mat3d) -> Mat3d {
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

/// 用 3×3 矩阵变换三维列向量（固定求和序）。
#[inline]
pub fn mat3_apply(m: &Mat3d, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 3×3 行列式（首行展开，固定求和序）。
#[inline]
pub fn mat3_determinant(m: &Mat3d) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 3×3 矩阵求逆（**自写**：余子式/伴随矩阵法，判定式 = [`mat3_determinant`]）。
///
/// # 错误
/// `|det| < 1e-12` 或 `det` 非有限 → [`ColorSpaceError::SingularMatrix`]。
/// 绝不返回"半成品"逆阵（宪章 C20：缺定义即明确报错，绝不静默降级）。
///
/// 数值说明：伴随矩阵法与高斯消元在本尺寸下误差同量级（相对 ~1e-16·κ），
/// 但它**无行交换、无主元选择、无除零抖动**，逐项可审计——这是选它的理由。
pub fn invert3(m: &Mat3d) -> Result<Mat3d, ColorSpaceError> {
    let det = mat3_determinant(m);
    if !det.is_finite() {
        return Err(ColorSpaceError::singular(
            "invert3",
            format!("行列式非有限（det = {det}）：矩阵含 NaN/Inf"),
        ));
    }
    if det.abs() < 1e-12 {
        return Err(ColorSpaceError::singular(
            "invert3",
            format!("矩阵奇异（|det| = {det:e} < 1e-12），无稳定逆阵"),
        ));
    }
    let inv = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

// ===========================================================================
//  确定性 atan2（D-29 三角件的 atan2 缺口，本模块自带）
// ===========================================================================

/// `[0, 1]` 上的 `atan`：三次半角归约 `t ← t / (1 + √(1 + t²))` 后
/// `atan(t) = 8 · atan(t₃)`，再用 8 项 Maclaurin（Horner 排布）。
///
/// 三次半角后 `t ≤ tan(π/32) ≈ 0.0985`，截断误差 `~t¹⁷/17 < 1e-18`。
/// 只含 `± × ÷` 与 `sqrt`（IEEE 精确）→ 跨平台逐位一致。
#[inline]
fn det_atan_unit(t: f64) -> f64 {
    let mut u = t;
    for _ in 0..3 {
        u = u / (1.0 + (1.0 + u * u).sqrt());
    }
    let u2 = u * u;
    let poly = 1.0
        + u2 * (-1.0 / 3.0
            + u2 * (1.0 / 5.0
                + u2 * (-1.0 / 7.0
                    + u2 * (1.0 / 9.0
                        + u2 * (-1.0 / 11.0
                            + u2 * (1.0 / 13.0 + u2 * (-1.0 / 15.0)))))));
    8.0 * u * poly
}

/// 确定性 `atan2(y, x)`，结果落在 `(-π, π]`（度）。
///
/// 象限归约 + [`det_atan_unit`]；`x = y = 0` 返回 `0.0`（不 panic，C18）。
/// **重复声明**：`crate::det_trig` 无 `atan`，本函数为本模块私有实现，
/// 未来 D-29 三角件应把它上收到 `det_trig`（本次不改那个文件）。
#[inline]
pub fn det_atan2(y: f64, x: f64) -> f64 {
    const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;
    const PI: f64 = std::f64::consts::PI;
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return if y > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }
    let ax = x.abs();
    let ay = y.abs();
    // `ax >= ay` 时 atan(|y|/|x|) ∈ [0, π/4]；否则用 π/2 + atan(|x|/|y|)。
    // `ax >= ay` 时角 ∈ [0, π/4]；否则用补角 `π/2 - atan(ax/ay)` ∈ (π/4, π/2]。
    let angle = if ax >= ay {
        det_atan_unit(ay / ax)
    } else {
        FRAC_PI_2 - det_atan_unit(ax / ay)
    };
    let mut r = if x > 0.0 { angle } else { PI - angle };
    if y < 0.0 {
        r = -r;
    }
    r
}

// ===========================================================================
//  白点
// ===========================================================================

/// 标准白点（CIE 1931 2° 观察者 xy 坐标）。
///
/// 取值出处：白点 xy 为各自规范原文；`D65` 亦见 ITU-R BT.709 / BT.2020 /
/// SMPTE 240M，`D50` 见 ISO 12647-1 与 ICC  PCS，`D60` 见 SMPTE ST 2065-1
/// （ACES 1.x 名义白点），`Dci` 见 SMPTE RP 431-2（DCI-P3 白点），
/// `A`/`C` 见 CIE 1931 标准照明体定义（`C` 与 SMPTE-C 同坐标）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhitePoint {
    /// D65：x = 0.3127, y = 0.3290（ITU-R BT.709 / BT.2020 / sRGB）。
    D65,
    /// D50：x = 0.34567, y = 0.35850（ISO 12647-1 印刷 / ICC PCS）。
    D50,
    /// D60：x = 0.32168, y = 0.33767（SMPTE ST 2065-1，ACES 1.x 名义白点）。
    D60,
    /// DCI 白点：x = 0.314, y = 0.351（SMPTE RP 431-2，DCI-P3）。
    Dci,
    /// 标准照明体 A（2° 观察者，2856 K 充气钨丝灯）。
    A,
    /// 标准照明体 C（2° 观察者，模拟昼光；与 SMPTE-C 同坐标）。
    C,
    /// 等能白点 E：x = 1/3, y = 1/3（CIE 1931 定义的理论白）。
    E,
}

impl WhitePoint {
    /// 全部白点（固定顺序，与枚举声明序一致）。
    pub const ALL: [WhitePoint; 7] = [
        WhitePoint::D65,
        WhitePoint::D50,
        WhitePoint::D60,
        WhitePoint::Dci,
        WhitePoint::A,
        WhitePoint::C,
        WhitePoint::E,
    ];

    /// 白点色度坐标 `xy`。
    #[inline]
    pub const fn chromaticity(self) -> (f64, f64) {
        match self {
            WhitePoint::D65 => (0.3127, 0.3290),
            WhitePoint::D50 => (0.34567, 0.35850),
            WhitePoint::D60 => (0.32168, 0.33767),
            WhitePoint::Dci => (0.3140, 0.3510),
            WhitePoint::A => (0.44757, 0.40745),
            WhitePoint::C => (0.31006, 0.31616),
            WhitePoint::E => (1.0 / 3.0, 1.0 / 3.0),
        }
    }

    /// 白点的 CIE XYZ 三刺激值（**Y = 1** 归一，按 `X = x/y, Z = (1-x-y)/y`）。
    ///
    /// 归一约定：本模块的 XYZ 域一律 `Yn = 1`（CIE 惯例），
    /// 故 `to_xyz([1,1,1])` 精确等于本函数返回值。
    #[inline]
    pub fn xyz(self) -> [f64; 3] {
        let (x, y) = self.chromaticity();
        // 非常量函数（浮点运算非 const）：白点在运行期求 XYZ。
        [(x / y), 1.0, ((1.0 - x - y) / y)]
    }
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            WhitePoint::D65 => "D65",
            WhitePoint::D50 => "D50",
            WhitePoint::D60 => "D60",
            WhitePoint::Dci => "DCI",
            WhitePoint::A => "A",
            WhitePoint::C => "C",
            WhitePoint::E => "E",
        }
    }

    /// 由色度坐标反查已注册白点（容差 1e-4）。
    ///
    /// 命中返回对应枚举；未命中返回 [`ColorSpaceError::UnknownWhitePoint`]
    /// （**不**静默近似到最近白点——宪章 C20）。
    pub fn from_chromaticity(x: f64, y: f64) -> Result<WhitePoint, ColorSpaceError> {
        if let Err(e) = validate_white_chromaticity(x, y) {
            return Err(e);
        }
        for wp in WhitePoint::ALL {
            let (wx, wy) = wp.chromaticity();
            if (wx - x).abs() < 1e-4 && (wy - y).abs() < 1e-4 {
                return Ok(wp);
            }
        }
        Err(ColorSpaceError::unknown_white_point(format!(
            "色度坐标 ({x}, {y}) 不匹配任何已注册白点（容差 1e-4）"
        )))
    }
}

/// 校验**白点**色度坐标：有限、`x > 0`、`y > 0`、`x + y < 1`。
///
/// 错误 → [`ColorSpaceError::InvalidChromaticity`]。
#[inline]
pub fn validate_white_chromaticity(x: f64, y: f64) -> Result<(), ColorSpaceError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(ColorSpaceError::invalid_chromaticity(format!(
            "白点坐标非有限（x = {x}, y = {y}）"
        )));
    }
    if x <= 0.0 || y <= 0.0 || x + y >= 1.0 {
        return Err(ColorSpaceError::invalid_chromaticity(format!(
            "白点坐标非法（x = {x}, y = {y}）：需 x > 0、y > 0 且 x + y < 1"
        )));
    }
    Ok(())
}

/// 校验**原色**色度坐标：有限、`y != 0`、`x + y <= 1`。
///
/// 刻意比白点宽松：ACES AP0 的 G/B 原色是**虚原色**（`x = 0`、
/// `y = -0.077`），在光谱轨迹之外，是 ACES 规范**有意**的产物，
/// 因此允许 `y < 0` 与 `x + y == 1`（`Z = 0`）。数学上仍可逆。
#[inline]
pub fn validate_primaries_chromaticity(x: f64, y: f64) -> Result<(), ColorSpaceError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(ColorSpaceError::invalid_chromaticity(format!(
            "原色坐标非有限（x = {x}, y = {y}）"
        )));
    }
    if y == 0.0 {
        return Err(ColorSpaceError::invalid_chromaticity(format!(
            "原色 y 坐标为 0（x = {x}）：X = x/y 与 Z = (1-x-y)/y 无定义"
        )));
    }
    if x + y > 1.0 {
        return Err(ColorSpaceError::invalid_chromaticity(format!(
            "原色坐标非法（x = {x}, y = {y}）：需 x + y <= 1（否则 Z < 0）"
        )));
    }
    Ok(())
}

// ===========================================================================
//  传递函数
// ===========================================================================

/// 传递函数（OETF/EOTF 的解析式形态）。
///
/// **本注册表的传递函数是「解码方向」（encoded → linear）为准**：
/// [`TransferFunction::decode`] 把色彩空间的存储值解为线性值，
/// [`TransferFunction::encode`] 反之。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransferFunction {
    /// 恒等（线性工作空间）。
    Linear,
    /// sRGB 曲线（IEC 61966-2-1：阈值 0.04045 / 0.0031308，指数 2.4）。
    Srgb,
    /// 纯幂 `sign(x)·|x|^(1/e)`（解码）/ `x^e`（编码）。
    ///
    /// 用于 Adobe RGB（`e = 563/256`）、Gamma 2.2 灰度、ProPhoto（`e = 1.8`）、
    /// Rec.601 PAL（`e = 2.8`）等。**只接受非负输入**（负底非整数幂无实数解，
    /// 引擎色彩域恒非负 → 返回结构化错误而非 NaN）。
    PureGamma {
        /// 编码端指数（`e > 0`）。
        exponent: f64,
    },
    /// ACEScc（AP1 对数编码；正段常数 `a = 0.0245786`、`c = 1.2047494`，
    /// 指数 2.4，与 SMPTE ST 2065-4 / ACES 1.x 一致）。
    ///
    /// # 分段定义（本实现口径）
    /// - 正段 `x >= 0`：`y = (x + a) / c`（**已对齐公开值**：
    ///   `ACEScc(1.0) = 0.8504496`、`ACEScc(0.18) = 0.1698101`）；
    /// - 负段 `x < 0`：`y = (a / c) · ((x + a) / a)^2.4`。
    ///
    /// 负段写成**归一化**形式（而非朴素的 `((x + a) / c)^2.4`）不是为了好看：
    /// 朴素形式在 `x = 0` 处**不连续**（左极限 `≈8.8e-5` vs 右极限
    /// `≈0.0204`），因而**不可逆**。归一化形式在断点连续、单调、且
    /// `encode ∘ decode = id` 精确成立（见 `acescc_roundtrip_and_continuity`）。
    ///
    /// # 诚实边界
    /// 负段的**归一化因子** `(a / c)` 是本实现自选的（目标是连续 + 可逆），
    /// 未经官方测试向量逐位校验——手头无规范原文可核。**正段与指数 2.4
    /// 是对齐公开值的**，负段形状（2.4 次幂）亦同。如需与某具体 ACES
    /// 实现逐位一致，须补该实现的官方测试向量后再校准本段。
    Acescc,
    /// BT.2020 SDR OETF（ITU-R BT.2020-2 §3.4 / Rec. BT.2020 Table 3）。
    ///
    /// # 规范定义（分段）
    /// ```text
    /// OETF (E → E')：E' = α·E^0.45 − (α−1)   当 E ≥ β
    ///                E' = 4.5·E               当 E < β
    /// EOTF (E' → E)：E  = ((E' + α−1)/α)^(1/0.45)  当 E' ≥ 4.5β
    ///                E  = E'/4.5                     当 E' < 4.5β
    /// ```
    /// 常数（**逐位取规范值**，不可由其他常数推导——推导会引入尾位差）：
    /// `α = 1.09929682680944`、`β = 0.018053968510807`。
    ///
    /// # `E = β` 处的连续性（本实现最易出错处）
    /// 两段在 `β` 处**按规范恒等连续**：BT.2020 的 `α` 与 `β` 取值使得
    /// `α·β^0.45 − (α−1) = 4.5·β` 精确成立。实测 f64 缺口
    /// **2.39e-15**（远低于 1e-9 阈值），由测试
    /// `bt2020_oetf_is_continuous_at_beta` 锁定。
    ///
    /// # 幂运算走 D-29
    /// `x^0.45` / `x^(1/0.45)` 一律经 [`crate::det_pow::det_pow`]
    /// （f32 进出，相对误差 ~1e-7），**不用 `powf`**（落平台 libm，
    /// 跨平台尾位不一致，违反 C17）。
    Bt2020,
}

/// sRGB 解码断点（IEC 61966-2-1）。
const SRGB_DECODE_BREAK: f64 = 0.04045;
/// sRGB 编码断点（IEC 61966-2-1）。
const SRGB_ENCODE_BREAK: f64 = 0.0031308;
/// sRGB 线性段斜率（IEC 61966-2-1）。
const SRGB_SLOPE: f64 = 12.92;
/// sRGB 幂指数（IEC 61966-2-1）。
const SRGB_EXPONENT: f64 = 2.4;
/// ACEScc 正段常数 `a`。
const ACESCC_A: f64 = 0.024_578_6;
/// ACEScc 负段常数 `c`。
const ACESCC_C: f64 = 1.204_749_4;
/// BT.2020 OETF 指数（ITU-R BT.2020-2 §4.1）。
const BT2020_OETF_EXPONENT: f64 = 0.45;
/// BT.2020 OETF 线性段斜率（`4.5`，ITU-R BT.2020-2 §4.1）。
const BT2020_LINEAR_SLOPE: f64 = 4.5;
/// BT.2020 OETF 常数 `α`（ITU-R BT.2020-2 §4.1，**逐位取规范值**）。
///
/// **不可由 `β` 反推**：规范给的 `α` 是拟合值，反推会引入 ~1e-5 级偏差，
/// 并连带把 `E = β` 处的连续性缺口从 2.4e-15 放大到 ~1e-5（见
/// [`TransferFunction::Bt2020`] 的文档与 `bt2020_oetf_is_continuous_at_beta`）。
pub const BT2020_ALPHA: f64 = 1.099_296_826_809_44;
/// BT.2020 OETF 常数 `β`（ITU-R BT.2020-2 §4.1，**逐位取规范值**）。
pub const BT2020_BETA: f64 = 0.018_053_968_510_807;
/// BT.2020 EOTF 断点 `4.5·β`（= 0.0812428582986315）。
pub const BT2020_EOTF_BREAK: f64 = 4.5 * BT2020_BETA;

impl TransferFunction {
    /// 解码：存储值 → 线性值。
    ///
    /// # 错误
    /// - [`ColorSpaceError::NonFiniteValue`]：输入含 NaN/Inf。
    /// - [`ColorSpaceError::DomainError`]：[`TransferFunction::PureGamma`] 遇负输入。
    pub fn decode(self, v: f64) -> Result<f64, ColorSpaceError> {
        if !v.is_finite() {
            return Err(ColorSpaceError::non_finite(format!(
                "TransferFunction::decode 输入非有限：{v}"
            )));
        }
        Ok(match self {
            TransferFunction::Linear => v,
            TransferFunction::Srgb => {
                if v <= SRGB_DECODE_BREAK {
                    v / SRGB_SLOPE
                } else {
                    // EOTF 指数是 2.4（不是 1/2.4——那是 OETF 的指数）。
                    det_pow(((v + 0.055) / 1.055) as f32, SRGB_EXPONENT as f32) as f64
                }
            }
            TransferFunction::PureGamma { exponent } => {
                if v < 0.0 {
                    return Err(ColorSpaceError::domain(format!(
                        "纯幂传递函数不接受负输入（exponent = {exponent}，v = {v}）"
                    )));
                }
                det_pow(v as f32, (1.0 / exponent) as f32) as f64
            }
            TransferFunction::Acescc => {
                // 断点在 `y = a / c`（等价于 `x = 0`），**不是** `y = 0`。
                if v >= ACESCC_A / ACESCC_C {
                    v * ACESCC_C - ACESCC_A
                } else {
                    if v < 0.0 {
                        return Err(ColorSpaceError::domain(format!(
                            "ACEScc 编码域为非负（收到 {v}）：正向编码恒 >= 0，负编码值无定义"
                        )));
                    }
                    let t = (v * ACESCC_C) / ACESCC_A; // ∈ [0, 1)
                    ACESCC_A * det_pow(t as f32, (1.0 / SRGB_EXPONENT) as f32) as f64 - ACESCC_A
                }
            }
            TransferFunction::Bt2020 => {
                // E' → E（BT.2020 EOTF，ITU-R BT.2020-2 §4.1）。断点是
                // `4.5β`（**不是** `β`）——β 是 OETF 侧的**线性域**输入阈值，
                // EOTF 侧的**编码域**阈值是线性段在 `E' = 4.5β` 处的取值。
                if v < BT2020_EOTF_BREAK {
                    v / BT2020_LINEAR_SLOPE
                } else {
                    let t = (v + (BT2020_ALPHA - 1.0)) / BT2020_ALPHA;
                    det_pow(t as f32, (1.0 / BT2020_OETF_EXPONENT) as f32) as f64
                }
            }
        })
    }

    /// 编码：线性值 → 存储值（[`TransferFunction::decode`] 的逆）。
    ///
    /// 负线性值在 sRGB 分支按线性段延伸（保留负号，HDR 数据友好）；
    /// 纯幂与 ACEScc 负段对过深负值返回结构化错误。
    pub fn encode(self, v: f64) -> Result<f64, ColorSpaceError> {
        if !v.is_finite() {
            return Err(ColorSpaceError::non_finite(format!(
                "TransferFunction::encode 输入非有限：{v}"
            )));
        }
        Ok(match self {
            TransferFunction::Linear => v,
            TransferFunction::Srgb => {
                if v <= SRGB_ENCODE_BREAK {
                    v * SRGB_SLOPE
                } else {
                    // OETF 指数是 1/2.4（EOTF 才是 2.4）。
                    det_pow(v as f32, (1.0 / SRGB_EXPONENT) as f32) as f64 * 1.055 - 0.055
                }
            }
            TransferFunction::PureGamma { exponent } => {
                if v < 0.0 {
                    return Err(ColorSpaceError::domain(format!(
                        "纯幂传递函数不接受负输入（exponent = {exponent}，v = {v}）"
                    )));
                }
                det_pow(v as f32, exponent as f32) as f64
            }
            TransferFunction::Acescc => {
                if v >= 0.0 {
                    (v + ACESCC_A) / ACESCC_C
                } else {
                    // 负段：`y = (a/c) · ((x + a) / a)^2.4`。
                    // `x = -a`（最负可表示点）时 `x + a = 0` → `y = 0`，这是
                    // **连续端点**而非错误，且正是 `encode(decode(0))` 的往返点。
                    let t = (v + ACESCC_A) / ACESCC_A; // ∈ [0, 1)
                    if t < 0.0 {
                        let lowest = -ACESCC_A;
                        return Err(ColorSpaceError::domain(format!(
                            "ACEScc 编码输入过小（v = {v}）：v + a < 0，无实数解（最负可表示点为 v = {lowest}）"
                        )));
                    }
                    (ACESCC_A / ACESCC_C)
                        * det_pow(t as f32, SRGB_EXPONENT as f32) as f64
                }
            }
            TransferFunction::Bt2020 => {
                // E → E'（BT.2020 OETF，ITU-R BT.2020-2 §4.1）。断点 `β`。
                // 负线性值按线性段延伸（保留负号，口径与上面 sRGB 分支一致：
                // HDR / 超范围数据友好，且 `encode ∘ decode = id` 在负半轴
                // 仍精确成立）。
                if v < BT2020_BETA {
                    BT2020_LINEAR_SLOPE * v
                } else {
                    let t = det_pow(v as f32, BT2020_OETF_EXPONENT as f32) as f64;
                    BT2020_ALPHA * t - (BT2020_ALPHA - 1.0)
                }
            }
        })
    }
}

// ===========================================================================
//  色彩空间定义
// ===========================================================================

/// 色度坐标（CIE 1931 `xy`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticity {
    /// 色度 x。
    pub x: f64,
    /// 色度 y。
    pub y: f64,
}

impl Chromaticity {
    /// 构造色度坐标。
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// 该色度对应的 CIE XYZ（`Y = 1` 归一）。
    ///
    /// # 错误
    /// `y == 0` 或坐标非有限 → [`ColorSpaceError::InvalidChromaticity`]。
    #[inline]
    pub fn to_xyz(self) -> Result<[f64; 3], ColorSpaceError> {
        validate_primaries_chromaticity(self.x, self.y)?;
        Ok([self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y])
    }
}

/// 原色三元组（红 / 绿 / 蓝的 CIE 1931 `xy`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primaries {
    /// 红原色。
    pub red: Chromaticity,
    /// 绿原色。
    pub green: Chromaticity,
    /// 蓝原色。
    pub blue: Chromaticity,
}

impl Primaries {
    /// 构造原色三元组。
    #[inline]
    pub const fn new(red: Chromaticity, green: Chromaticity, blue: Chromaticity) -> Self {
        Self { red, green, blue }
    }

    /// 灰度原色：三通道共用同一色度（`[W, W, W]`）。
    #[inline]
    pub const fn gray(wp: WhitePoint) -> Self {
        let (x, y) = wp.chromaticity();
        Self {
            red: Chromaticity { x, y },
            green: Chromaticity { x, y },
            blue: Chromaticity { x, y },
        }
    }
}

/// 由白点 + 原色推导 RGB → XYZ 矩阵（**Bradford 无关的纯数学推导**）。
///
/// 推导（标准 CIE 做法）：
/// 1. 原色 → `M = [[xr/yr, xg/yg, xb/yb], [1,1,1], [zr/yr, zg/yg, zb/yb]]`；
/// 2. 解 `M · Ws = W_white`，得每通道的白点归一因子 `Ws`（使 `(1,1,1)` → 白点）；
/// 3. `RGB → XYZ = M · diag(Ws)`。
///
/// # 错误
/// - 白点/原色坐标非法 → [`ColorSpaceError::InvalidChromaticity`]；
/// - `M` 奇异（三原色退化，如三者全同）→ [`ColorSpaceError::SingularMatrix`]。
pub fn build_rgb_to_xyz(primaries: &Primaries, white: WhitePoint) -> Result<Mat3d, ColorSpaceError> {
    let (wx, wy) = white.chromaticity();
    validate_white_chromaticity(wx, wy)?;
    let m = [
        [primaries.red.x / primaries.red.y, primaries.green.x / primaries.green.y, primaries.blue.x / primaries.blue.y],
        [1.0, 1.0, 1.0],
        [
            (1.0 - primaries.red.x - primaries.red.y) / primaries.red.y,
            (1.0 - primaries.green.x - primaries.green.y) / primaries.green.y,
            (1.0 - primaries.blue.x - primaries.blue.y) / primaries.blue.y,
        ],
    ];
    // 解 m · ws = W_white（ws 是 RGB→XYZ 的列归一因子）。
    let w_white = [(wx / wy), 1.0, ((1.0 - wx - wy) / wy)];
    let ws = mat3_apply(&invert3(&m)?, w_white);
    let mut out = [[0.0f64; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            out[r][c] = m[r][c] * ws[c];
        }
    }
    Ok(out)
}

/// 色彩空间的数据表示形态（决定矩阵路径是否可用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Representation {
    /// RGB 三通道：由白点 + 原色 + 传递函数完整决定。
    Rgb,
    /// 单通道灰度：三通道共用白点，矩阵为 `diag(Xw, 1, Zw)`。
    Gray,
    /// CIE XYZ 本身：RGB↔XYZ 为单位矩阵，传递函数为恒等。
    Xyz,
    /// CIE Lab（球面坐标的直角形式）：**不走 RGB 矩阵路径**，
    /// 用 [`ColorSpaceDef::xyz_to_lab`] / [`ColorSpaceDef::lab_to_xyz`]。
    CieLab,
    /// CIE LCh（Lab 的柱面形式）：用 [`ColorSpaceDef::xyz_to_lch`] /
    /// [`ColorSpaceDef::lch_to_xyz`]。
    CieLch,
}

/// 一个色彩空间的完整定义：**白点 + 原色 + 传递函数**三元组。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorSpaceDef {
    /// 规范名（小写 ASCII，`-`/空格归一为 `_`；查表键）。
    pub name: &'static str,
    /// 人类可读名（含规范代号）。
    pub display_name: &'static str,
    /// 数值出处（规范名 + 表/条），供审计追溯。
    pub citation: &'static str,
    /// 白点（CIE 1931 `xy`）。
    pub white_point: WhitePoint,
    /// 原色（CIE 1931 `xy`）。
    pub primaries: Primaries,
    /// 数据表示形态。
    pub representation: Representation,
    /// 传递函数。
    pub transfer: TransferFunction,
}

impl ColorSpaceDef {
    /// RGB（灰度）→ CIE XYZ 矩阵（`Y = 1` 归一）。
    ///
    /// - [`Representation::Rgb`]：由白点 + 原色推导（见 [`build_rgb_to_xyz`]）；
    /// - [`Representation::Gray`]：对角阵 `diag(Xw, 1, Zw)`——灰度空间无色度，
    ///   只有中性轴，故不设交叉项（若强行用三同原色推导，矩阵会奇异）；
    /// - [`Representation::Xyz`]：单位矩阵；
    /// - [`Representation::CieLab`] / [`Representation::CieLch`]：结构化错误
    ///   （这两者不是 RGB 空间，走 Lab/LCh 专用转换）。
    pub fn rgb_to_xyz_matrix(&self) -> Result<Mat3d, ColorSpaceError> {
        match self.representation {
            Representation::Rgb => build_rgb_to_xyz(&self.primaries, self.white_point),
            Representation::Gray => {
                let [x, _, z] = self.white_point.xyz();
                Ok([[x, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z]])
            }
            Representation::Xyz => Ok(IDENTITY3),
            Representation::CieLab | Representation::CieLch => {
                Err(ColorSpaceError::unsupported_representation(self))
            }
        }
    }

    /// CIE XYZ → RGB（灰度）矩阵（[`ColorSpaceDef::rgb_to_xyz_matrix`] 的逆）。
    pub fn xyz_to_rgb_matrix(&self) -> Result<Mat3d, ColorSpaceError> {
        invert3(&self.rgb_to_xyz_matrix()?)
    }

    /// RGB → 线性 RGB（只过传递函数，不换基）。
    pub fn decode(&self, rgb: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        Ok([
            self.transfer.decode(rgb[0])?,
            self.transfer.decode(rgb[1])?,
            self.transfer.decode(rgb[2])?,
        ])
    }

    /// 线性 RGB → RGB（只过传递函数）。
    pub fn encode(&self, linear: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        Ok([
            self.transfer.encode(linear[0])?,
            self.transfer.encode(linear[1])?,
            self.transfer.encode(linear[2])?,
        ])
    }

    /// 本色彩空间 RGB → CIE XYZ（先解码再换基；**不做白点适配**）。
    ///
    /// 输出 XYZ 相对于**本空间自己的白点**（`Yn = 1`），故
    /// `to_xyz([1,1,1])` 精确等于 [`WhitePoint::xyz`]。
    /// 要适配到别的白点用 [`adapt_white`] 或 [`convert`]。
    pub fn to_xyz(&self, rgb: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        if rgb.iter().any(|v| !v.is_finite()) {
            return Err(ColorSpaceError::non_finite(format!(
                "{}.to_xyz 输入非有限：{rgb:?}",
                self.name
            )));
        }
        match self.representation {
            Representation::CieLab | Representation::CieLch => {
                Err(ColorSpaceError::unsupported_representation(self))
            }
            _ => Ok(mat3_apply(&self.rgb_to_xyz_matrix()?, self.decode(rgb)?)),
        }
    }

    /// CIE XYZ → 本色彩空间 RGB（**不做白点适配**；色域外不裁剪）。
    pub fn from_xyz(&self, xyz: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        if xyz.iter().any(|v| !v.is_finite()) {
            return Err(ColorSpaceError::non_finite(format!(
                "{}.from_xyz 输入非有限：{xyz:?}",
                self.name
            )));
        }
        match self.representation {
            Representation::CieLab | Representation::CieLch => {
                Err(ColorSpaceError::unsupported_representation(self))
            }
            _ => {
                let rgb = mat3_apply(&self.xyz_to_rgb_matrix()?, xyz);
                self.encode(rgb)
            }
        }
    }

    /// 本空间 RGB → CIE XYZ，并**适配到 `dst_white`**（Bradford，源 = 本空间白点）。
    pub fn to_xyz_adapted(
        &self,
        rgb: [f64; 3],
        dst_white: WhitePoint,
    ) -> Result<[f64; 3], ColorSpaceError> {
        let xyz = self.to_xyz(rgb)?;
        adapt_white(xyz, self.white_point, dst_white)
    }

    /// CIE XYZ（本空间白点）→ 本空间 RGB。
    pub fn to_lab(&self, rgb: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        let xyz = self.to_xyz(rgb)?;
        xyz_to_lab(xyz, self.white_point)
    }

    /// CIE Lab（以本空间白点为基准）→ 本空间 RGB。
    pub fn from_lab(&self, lab: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        self.from_xyz(lab_to_xyz(lab, self.white_point)?)
    }

    /// 本空间 RGB → CIE LCh（以本空间白点为基准）。
    pub fn to_lch(&self, rgb: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        let lab = self.to_lab(rgb)?;
        Ok(lab_to_lch(lab))
    }

    /// CIE LCh（以本空间白点为基准）→ 本空间 RGB。
    pub fn from_lch(&self, lch: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        self.from_lab(lch_to_lab(lch))
    }

    /// XYZ → Lab（以本空间白点为基准；不要求本空间是 RGB 空间）。
    #[inline]
    pub fn xyz_to_lab(&self, xyz: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        xyz_to_lab(xyz, self.white_point)
    }

    /// Lab → XYZ（以本空间白点为基准）。
    #[inline]
    pub fn lab_to_xyz(&self, lab: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        lab_to_xyz(lab, self.white_point)
    }

    /// XYZ → LCh（以本空间白点为基准）。
    #[inline]
    pub fn xyz_to_lch(&self, xyz: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        let lab = xyz_to_lab(xyz, self.white_point)?;
        Ok(lab_to_lch(lab))
    }

    /// LCh → XYZ（以本空间白点为基准）。
    #[inline]
    pub fn lch_to_xyz(&self, lch: [f64; 3]) -> Result<[f64; 3], ColorSpaceError> {
        lab_to_xyz(lch_to_lab(lch), self.white_point)
    }

    /// 灰度辅助：单通道亮度 → CIE XYZ（沿白点中性轴）。
    #[inline]
    pub fn gray_to_xyz(&self, g: f64) -> Result<[f64; 3], ColorSpaceError> {
        let g = self.transfer.decode(g)?;
        let w = self.white_point.xyz();
        Ok([w[0] * g, g, w[2] * g])
    }

    /// 灰度辅助：CIE XYZ → 单通道亮度（取 `Y` 通道，即 CIE 亮度）。
    #[inline]
    pub fn xyz_to_gray(&self, xyz: [f64; 3]) -> Result<f64, ColorSpaceError> {
        self.transfer.encode(xyz[1])
    }

    /// 降精度适配器：把 RGB→XYZ 矩阵收窄成既有 `f32` 的
    /// [`crate::colorspace::Mat3`]（收窄点唯一，便于审计精度损失）。
    pub fn rgb_to_xyz_mat3f32(&self) -> Result<crate::colorspace::Mat3, ColorSpaceError> {
        let m = self.rgb_to_xyz_matrix()?;
        let mut out = [[0.0f32; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = m[r][c] as f32;
            }
        }
        Ok(out)
    }

    /// 降精度适配器：`f32` 的 [`Vec3`] → XYZ（`f64`）。
    pub fn to_xyz_vec3(&self, rgb: Vec3) -> Result<Vec3, ColorSpaceError> {
        let xyz = self.to_xyz([rgb.x as f64, rgb.y as f64, rgb.z as f64])?;
        Ok(Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32))
    }
}

// ===========================================================================
//  Bradford 色度适配（自写；B 与 B⁻¹ 均由 invert3 走同一条路径）
// ===========================================================================

/// Bradford 锥体响应矩阵 `B`（Lam 1985 的 Bradford 变换，业界标准常数）。
///
/// 取值出处：Lam, K. M. (1985) "Metamerism and colour constancy", PhD thesis,
/// University of Bradford；亦见 ACES 官方 `Bradford.html` 参考实现与
/// ICC （自适应色度匹配的标准做法）。
pub const BRADFORD: Mat3d = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Bradford 源白点 → 目标白点色度适配矩阵
/// `M = B⁻¹ · diag(B·W_dst / B·W_src) · B`。
///
/// `src == dst` 时直接返回单位矩阵（**零数值抖动**，往返恒等的前提）。
///
/// # 错误
/// 锥体响应坐标为 0（白点退化）或矩阵奇异 → 结构化错误。
pub fn bradford_adaptation(
    src: WhitePoint,
    dst: WhitePoint,
) -> Result<Mat3d, ColorSpaceError> {
    if src == dst {
        return Ok(IDENTITY3);
    }
    let b_inv = invert3(&BRADFORD)?;
    let cone_src = mat3_apply(&BRADFORD, src.xyz());
    let cone_dst = mat3_apply(&BRADFORD, dst.xyz());
    for i in 0..3 {
        if !cone_src[i].is_finite() || !cone_dst[i].is_finite() || cone_src[i] == 0.0 {
            return Err(ColorSpaceError::invalid_chromaticity(format!(
                "Bradford 锥体响应退化（第 {i} 分量：src = {}, dst = {}）",
                cone_src[i], cone_dst[i]
            )));
        }
    }
    let mut diag = [[0.0f64; 3]; 3];
    for i in 0..3 {
        diag[i][i] = cone_dst[i] / cone_src[i];
    }
    Ok(mat3_mul(&b_inv, &mat3_mul(&diag, &BRADFORD)))
}

/// 把 CIE XYZ 从 `src` 白点**适配**到 `dst` 白点（Bradford）。
///
/// `src == dst` 时返回输入原值（不引入任何浮点运算）。
pub fn adapt_white(
    xyz: [f64; 3],
    src: WhitePoint,
    dst: WhitePoint,
) -> Result<[f64; 3], ColorSpaceError> {
    if xyz.iter().any(|v| !v.is_finite()) {
        return Err(ColorSpaceError::non_finite(format!(
            "adapt_white 输入非有限：{xyz:?}"
        )));
    }
    if src == dst {
        return Ok(xyz);
    }
    Ok(mat3_apply(&bradford_adaptation(src, dst)?, xyz))
}

/// 色彩空间互转（自动决定是否需要 Bradford 适配）。
///
/// 路径：`rgb --解码--> 线性 --本空间矩阵--> XYZ --Bradford--> 目标白点 XYZ
/// --目标空间逆矩阵--> 线性 --编码--> 目标 rgb`。
///
/// 同白点时**不经过** Bradford（零额外误差）；跨白点时必经 Bradford。
pub fn convert(
    rgb: [f64; 3],
    from: &ColorSpaceDef,
    to: &ColorSpaceDef,
) -> Result<[f64; 3], ColorSpaceError> {
    let xyz = from.to_xyz(rgb)?;
    let adapted = adapt_white(xyz, from.white_point, to.white_point)?;
    to.from_xyz(adapted)
        .map_err(|e| ColorSpaceError::new(e.code, format!("convert {} → {}", from.name, to.name), e.problem, e.fix))
}

// ===========================================================================
//  CIE XYZ ↔ Lab ↔ LCh（CIE 1976 L*a*b*，D65 为默认基准）
// ===========================================================================

/// CIE 标准 `ε`：`6/29` 的立方（Lab 线性段阈值）。
const LAB_EPS: f64 = 6.0 / 29.0;
/// CIE 标准 `κ`：`24389/27`（Lab 线性段斜率因子）。
const LAB_KAPPA: f64 = 24_389.0 / 27.0;

/// Lab 相对白点的 `f(t)`（CIE 1976 定义，线性段斜率 `1/(3δ²)`）。
#[inline]
fn lab_f(t: f64) -> Result<f64, ColorSpaceError> {
    if t < 0.0 {
        return Err(ColorSpaceError::domain(format!(
            "Lab 转换要求 XYZ 分量非负（收到 {t}）：CIE 1976 定义域为 t >= 0"
        )));
    }
    if t > LAB_EPS * LAB_EPS * LAB_EPS {
        Ok(det_pow(t as f32, 1.0 / 3.0) as f64)
    } else {
        Ok(LAB_KAPPA * t / 27.0 + 4.0 / 29.0)
    }
}

/// `f(t)` 的逆（CIE 1976 定义）。
#[inline]
fn lab_f_inv(f: f64) -> f64 {
    if f > LAB_EPS {
        det_pow(f as f32, 3.0) as f64
    } else {
        27.0 * (f - 4.0 / 29.0) / LAB_KAPPA
    }
}

/// CIE XYZ → CIE Lab（`wp` 为参考白点，默认 D65 由调用方传入）。
///
/// `L* ∈ [0,100]`、`a*`/`b*` 相对白点（CIE 1976）。
pub fn xyz_to_lab(xyz: [f64; 3], wp: WhitePoint) -> Result<[f64; 3], ColorSpaceError> {
    if xyz.iter().any(|v| !v.is_finite()) {
        return Err(ColorSpaceError::non_finite(format!(
            "xyz_to_lab 输入非有限：{xyz:?}"
        )));
    }
    let w = wp.xyz();
    let fx = lab_f(xyz[0] / w[0])?;
    let fy = lab_f(xyz[1] / w[1])?;
    let fz = lab_f(xyz[2] / w[2])?;
    Ok([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)])
}

/// CIE Lab → CIE XYZ（`wp` 为参考白点）。
///
/// # 错误
/// 逆变换 `f⁻¹` 在**线性段**可产出负分量（当 `f < 4/29`，即 Lab 落在
/// CIE 1976 的**无定义区**——此时对应的 XYZ 无物理意义）。本函数**明确
/// 报错**（CS1005），**绝不**返回带负分量的 XYZ 让下游静默拿到坏色
/// （宪章 C20：缺定义即明确报错，绝不静默降级）。
pub fn lab_to_xyz(lab: [f64; 3], wp: WhitePoint) -> Result<[f64; 3], ColorSpaceError> {
    if lab.iter().any(|v| !v.is_finite()) {
        return Err(ColorSpaceError::non_finite(format!(
            "lab_to_xyz 输入非有限：{lab:?}"
        )));
    }
    let w = wp.xyz();
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    let xyz = [
        w[0] * lab_f_inv(fx),
        w[1] * lab_f_inv(fy),
        w[2] * lab_f_inv(fz),
    ];
    for (i, v) in xyz.iter().enumerate() {
        if *v < 0.0 {
            return Err(ColorSpaceError::domain(format!(
                "Lab {lab:?} 落在 CIE 1976 无定义区：逆变换得 X/Y/Z 第 {i} 分量 = {v}（负）"
            )));
        }
    }
    Ok(xyz)
}

/// CIE Lab → CIE LCh（`C = √(a²+b²)`、`h = atan2(b, a)`，角度制）。
///
/// 中性色（`a = b = 0`）的色相约定为 `0°`（CIE 惯例：消色差无色相）。
pub fn lab_to_lch(lab: [f64; 3]) -> [f64; 3] {
    let (a, b) = (lab[1], lab[2]);
    let c = det_pow_int_f64(a * a + b * b, 0.5);
    if a == 0.0 && b == 0.0 {
        return [lab[0], 0.0, 0.0];
    }
    // 弧度 → 度：纯乘法（IEEE 精确）。
    let mut h = det_atan2(b, a) * (180.0 / std::f64::consts::PI);
    if h < 0.0 {
        h += 360.0;
    }
    if h >= 360.0 {
        h = 0.0;
    }
    [lab[0], c, h]
}

/// CIE LCh → CIE Lab（`h` 为角度制）。
///
/// 三角函数走 [`crate::det_trig::cos_sin`]（D-29 自研件）而**非** `f64::cos`
/// / `f64::sin`——后者落平台 libm，跨平台尾位不一致，违反 C17 确定性铁律。
/// 代价：色度角精度 ~1e-7（f32 级），对本用途（色域描述）绰绰有余。
pub fn lch_to_lab(lch: [f64; 3]) -> [f64; 3] {
    if !lch[1].is_finite() || !lch[2].is_finite() {
        return [lch[0], 0.0, 0.0];
    }
    // 角度 → 弧度：纯乘法（IEEE 精确），不调 libm。
    let rad = lch[2] * (std::f64::consts::PI / 180.0);
    let (c, sn) = crate::det_trig::cos_sin(rad as f32);
    [lch[0], lch[1] * c as f64, lch[1] * sn as f64]
}

/// `√x`（`x >= 0`）：`det_pow` 走 f32 会引入相对误差 ~1e-7，
/// 这里为 CIE 定义式（平方根在 IEEE 精确）直接用 `f64::sqrt`。
#[inline]
fn det_pow_int_f64(x: f64, p: f64) -> f64 {
    if p == 0.5 {
        return x.sqrt();
    }
    if x < 0.0 {
        return f64::NAN;
    }
    det_pow(x as f32, p as f32) as f64
}

// ===========================================================================
//  注册表（固定顺序 = 名字升序；查表顺序确定，禁 HashMap）
// ===========================================================================

/// 注册表条目总数。
pub const COLOR_SPACE_COUNT: usize = 20;

/// 全部注册色彩空间。
///
/// **数组已按 `name` 严格升序排好**（Rust `&str` 的 `Ord` 是逐字节序），
/// 这既让 [`by_name`] 能二分查找，也让枚举输出天然确定——不依赖任何
/// 哈希容器的迭代顺序（宪章 C17）。
///
/// 白点口径统一为各自规范的**名义白点**，唯一例外是 ACES 两个空间
/// （见模块头「与既有色彩模块的关系」与「诚实边界」）。
pub static COLOR_SPACES: [ColorSpaceDef; COLOR_SPACE_COUNT] = [
    // 1. ACES 2065-1（AP0）：存档/交换线性空间。白点按项目既有常量矩阵
    //    口径锚在 D65（ACES 1.x 名义为 D60，见模块头诚实边界）。
    ColorSpaceDef {
        name: "aces2065",
        display_name: "ACES 2065-1 (AP0)",
        citation: "SMPTE ST 2065-1:2012 / ACES 1.x Appendix A（AP0 色彩学，白点 D60）",
        white_point: WhitePoint::D60,
        primaries: Primaries::new(
            Chromaticity::new(0.734_699, 0.265_301),
            // AP0 的 G/B 是「虚原色」（在光谱轨迹外），y < 0 为规范有意设计。
            Chromaticity::new(0.0, 1.0),
            Chromaticity::new(0.000_100, -0.077_000),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
    // 2. ACEScc：AP1 的对数编码（SMPTE ST 2065-4）。
    ColorSpaceDef {
        name: "acescc",
        display_name: "ACEScc (AP1 log)",
        citation: "SMPTE ST 2065-4:2012 / ACES 1.x（AP1 + 对数编码，白点 D60）",
        white_point: WhitePoint::D60,
        primaries: Primaries::new(
            Chromaticity::new(0.713, 0.293),
            Chromaticity::new(0.165, 0.830),
            Chromaticity::new(0.128, 0.044),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Acescc,
    },
    // 3. ACEScg（AP1）：线性绘画工作空间。
    ColorSpaceDef {
        name: "acescg",
        display_name: "ACEScg (AP1)",
        citation: "SMPTE ST 2065-3 / ACES 1.x Appendix A（AP1 色彩学，白点 D60）",
        white_point: WhitePoint::D60,
        primaries: Primaries::new(
            Chromaticity::new(0.713, 0.293),
            Chromaticity::new(0.165, 0.830),
            Chromaticity::new(0.128, 0.044),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
    // 4. Adobe RGB (1998)：γ = 563/256 ≈ 2.19921875（Adobe）。
    ColorSpaceDef {
        name: "adobergb1998",
        display_name: "Adobe RGB (1998)",
        citation: "Adobe RGB (1998) 白皮书（gamma = 563/256 ≈ 2.19921875）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.6400, 0.3300),
            Chromaticity::new(0.2100, 0.7100),
            Chromaticity::new(0.1500, 0.0600),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::PureGamma { exponent: 563.0 / 256.0 },
    },
    // 5. Adobe Wide Gamut：γ = 2.2，白点 D50。
    ColorSpaceDef {
        name: "adobergw",
        display_name: "Adobe Wide Gamut",
        citation: "Adobe Wide Gamut 白皮书（gamma ≈ 2.2，白点 D50）",
        white_point: WhitePoint::D50,
        primaries: Primaries::new(
            Chromaticity::new(0.7347, 0.2653),
            Chromaticity::new(0.1152, 0.8264),
            Chromaticity::new(0.1566, 0.0177),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::PureGamma { exponent: 2.2 },
    },
    // 6. CIE L*a*b*：默认 D65 基准。
    ColorSpaceDef {
        name: "cielab",
        display_name: "CIE L*a*b* (D65)",
        citation: "CIE 1976 L*a*b*（CIE 15:2004 Table 6），参考白点 D65",
        white_point: WhitePoint::D65,
        primaries: Primaries::gray(WhitePoint::D65),
        representation: Representation::CieLab,
        transfer: TransferFunction::Linear,
    },
    // 7. CIE LCh(ab)：Lab 的柱面形式。
    ColorSpaceDef {
        name: "cielch",
        display_name: "CIE LCh(ab) (D65)",
        citation: "CIE 15:2004（LCh(ab) 柱面坐标），参考白点 D65",
        white_point: WhitePoint::D65,
        primaries: Primaries::gray(WhitePoint::D65),
        representation: Representation::CieLch,
        transfer: TransferFunction::Linear,
    },
    // 8. CIE XYZ 本身（CIE 1931，Y = 1 归一）。
    ColorSpaceDef {
        name: "ciexyz",
        display_name: "CIE XYZ (D65)",
        citation: "CIE 1931 2° 标准观察者 XYZ（Y = 1 归一）",
        white_point: WhitePoint::D65,
        primaries: Primaries::gray(WhitePoint::D65),
        representation: Representation::Xyz,
        transfer: TransferFunction::Linear,
    },
    // 9. DCI-P3：白点 DCI（SMPTE RP 431-2），线性（影院工作域）。
    ColorSpaceDef {
        name: "dcip3",
        display_name: "DCI-P3 (DCI white)",
        citation: "SMPTE RP 431-2:2020（DCI-P3 色度 + DCI 白点 0.314/0.351）",
        white_point: WhitePoint::Dci,
        primaries: Primaries::new(
            Chromaticity::new(0.680, 0.320),
            Chromaticity::new(0.265, 0.690),
            Chromaticity::new(0.150, 0.060),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
    // 10. Display P3：sRGB 曲线 + P3-D65 色度。
    ColorSpaceDef {
        name: "displayp3",
        display_name: "Display P3 (sRGB curve)",
        citation: "SMPTE EG 432-1 / Apple Display P3（P3-D65 色度 + sRGB 传输函数）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.680, 0.320),
            Chromaticity::new(0.265, 0.690),
            Chromaticity::new(0.150, 0.060),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Srgb,
    },
    // 11. Gamma 2.2 灰度：Rec.709 亮度权重 + 纯 2.2 幂。
    ColorSpaceDef {
        name: "gammagray22",
        display_name: "Gamma 2.2 Gray",
        citation: "ITU-R BT.709 亮度权重（Kr = 0.2126, Kb = 0.0722）+ 纯 γ 2.2",
        white_point: WhitePoint::D65,
        primaries: Primaries::gray(WhitePoint::D65),
        representation: Representation::Gray,
        transfer: TransferFunction::PureGamma { exponent: 2.2 },
    },
    // 12. 线性 Rec.709 工作空间（与 `srgblinear` 数值等同，命名不同，
    //     对应 DCC 里 "linear Rec.709" 与 "linear sRGB" 两种叫法）。
    ColorSpaceDef {
        name: "lineargb709",
        display_name: "Linear Rec.709 (camera working)",
        citation: "ITU-R BT.709 原色 + D65，线性（0 = 黑，1 = 白）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.640, 0.330),
            Chromaticity::new(0.300, 0.600),
            Chromaticity::new(0.150, 0.060),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
    // 13. 线性灰度：Rec.709 亮度权重的纯线性形式。
    ColorSpaceDef {
        name: "lineargray",
        display_name: "Linear Gray",
        citation: "ITU-R BT.709 亮度权重 + 线性传递函数（无 gamma）",
        white_point: WhitePoint::D65,
        primaries: Primaries::gray(WhitePoint::D65),
        representation: Representation::Gray,
        transfer: TransferFunction::Linear,
    },
    // 14. ProPhoto RGB（ROMM）：D50 白点，γ = 1.8（趾段未建模，近似）。
    ColorSpaceDef {
        name: "prophoto",
        display_name: "ProPhoto RGB (ROMM, D50)",
        citation: "ANSI/I3A IT10.7666 (ROMM RGB) —— γ 1.8 纯幂近似，趾段未建模",
        white_point: WhitePoint::D50,
        primaries: Primaries::new(
            Chromaticity::new(0.734_699, 0.265_301),
            Chromaticity::new(0.159_597, 0.840_403),
            // ROMM 的蓝原色 y 极小（0.000105），常见文献亦记 0.0001。
            Chromaticity::new(0.036_598, 0.000_105),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::PureGamma { exponent: 1.8 },
    },
    // 15. Rec.2020 / BT.2020：广色域 UHD 视频。
    // **传递函数 = BT.2020 SDR OETF**（本次新增，原为 `Linear`——那是**错的**：
    // Rec.2020 的规范传递函数就是 §4.1 那条分段曲线，标成线性等于把
    // 「显示域编码值」当「线性光」用，会让所有 BT.2020 资产偏亮 ~2.4×。
    // 改动只影响本条目自身的 `decode`/`encode`；矩阵与原色一字未动。
    ColorSpaceDef {
        name: "rec2020",
        display_name: "Rec.2020 / BT.2020",
        citation: "ITU-R BT.2020-2（Rec.2020 色度 + D65 + §4.1 SDR OETF）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.708, 0.292),
            Chromaticity::new(0.170, 0.797),
            Chromaticity::new(0.131, 0.046),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Bt2020,
    },
    // 16. Rec.601 NTSC：SMPTE-C 白点（0.31006/0.31616）+ γ 2.2。
    ColorSpaceDef {
        name: "rec601ntsc",
        display_name: "Rec.601 NTSC (SMPTE-C)",
        citation: "SMPTE 170M / ITU-R BT.601（NTSC：白点 0.310/0.316，γ ≈ 2.2）",
        white_point: WhitePoint::C,
        primaries: Primaries::new(
            Chromaticity::new(0.630, 0.340),
            Chromaticity::new(0.310, 0.595),
            Chromaticity::new(0.155, 0.070),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::PureGamma { exponent: 2.2 },
    },
    // 17. Rec.601 PAL：EBU 601（Kr = 0.299, Kb = 0.114 → 0.640/0.330、
    //     0.290/0.600、0.150/0.060）白点 D65 + γ 2.8。
    ColorSpaceDef {
        name: "rec601pal",
        display_name: "Rec.601 PAL (gamma 2.8)",
        citation: "EBU Tech 3285 / ITU-R BT.601（PAL：Kr = 0.299, Kb = 0.114, γ 2.8）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.640, 0.330),
            Chromaticity::new(0.290, 0.600),
            Chromaticity::new(0.150, 0.060),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::PureGamma { exponent: 2.8 },
    },
    // 18. Rec.709 / BT.709：HDTV 基准。
    ColorSpaceDef {
        name: "rec709",
        display_name: "Rec.709 / BT.709",
        citation: "ITU-R BT.709-6（Rec.709 色度 + D65），此处配线性传递函数",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.640, 0.330),
            Chromaticity::new(0.300, 0.600),
            Chromaticity::new(0.150, 0.060),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
    // 19. sRGB（IEC 61966-2-1）：sRGB 曲线 + BT.709 色度 + D65。
    ColorSpaceDef {
        name: "srgb",
        display_name: "sRGB (IEC 61966-2-1)",
        citation: "IEC 61966-2-1:1999（色度取自原色定义；传输函数为 sRGB 曲线）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.6400, 0.3300),
            Chromaticity::new(0.3000, 0.6000),
            Chromaticity::new(0.1500, 0.0600),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Srgb,
    },
    // 20. 线性 sRGB：BT.709 色度 + D65 + 线性（数值上等同 `lineargb709`）。
    ColorSpaceDef {
        name: "srgblinear",
        display_name: "Linear sRGB",
        citation: "IEC 61966-2-1 色度 + D65，线性传递函数（与 `lineargb709` 数值等同）",
        white_point: WhitePoint::D65,
        primaries: Primaries::new(
            Chromaticity::new(0.6400, 0.3300),
            Chromaticity::new(0.3000, 0.6000),
            Chromaticity::new(0.1500, 0.0600),
        ),
        representation: Representation::Rgb,
        transfer: TransferFunction::Linear,
    },
];

/// 查表键归一（**分隔符无关**）：转小写并**丢弃** `-` / `_` / 空白。
///
/// 于是 `Display P3` / `display-p3` / `DISPLAY_P3` / `display_p3` 全部归一为
/// `displayp3`，与注册名一致；`AdobeRGB-1998` 归一为 `adobergb1998`。
///
/// 纯 ASCII 逐字节映射，**无 Unicode 归一化**（引擎内资产名约定 ASCII，
/// 免去引入表格化数据的复杂度——D-16）。
#[inline]
pub fn normalize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch == '-' || ch == '_' || ch.is_whitespace() {
            continue;
        }
        for lower in ch.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

/// 全部注册色彩空间（**已按 `name` 升序**，顺序确定）。
#[inline]
pub fn definitions() -> &'static [ColorSpaceDef] {
    &COLOR_SPACES
}

/// 全部注册色彩空间的名字（升序）。
#[inline]
pub fn names() -> Vec<&'static str> {
    COLOR_SPACES.iter().map(|d| d.name).collect()
}

/// 按名查（大小写、`-`/空格宽容；`name` 走 [`normalize_name`] 归一）。
///
/// 未知名 → [`ColorSpaceError::UnknownColorSpace`]，错误文本附**相近名建议**
/// （C19 必含建议段）。
pub fn by_name(name: &str) -> Result<&'static ColorSpaceDef, ColorSpaceError> {
    let key = normalize_name(name);
    // 线性扫描 + 精确匹配：表只有 20 项，二分的收益不抵可读性损失；
    // 顺序遍历保证**首个精确匹配**语义与表序一致（确定）。
    for d in COLOR_SPACES.iter() {
        if d.name == key {
            return Ok(d);
        }
    }
    Err(ColorSpaceError::unknown_color_space(
        format!("未注册色彩空间 {key:?}"),
        nearest_hint(&key),
    ))
}

/// 按白点查（升序）。
pub fn by_white_point(wp: WhitePoint) -> Vec<&'static ColorSpaceDef> {
    COLOR_SPACES.iter().filter(|d| d.white_point == wp).collect()
}

/// 拟合常量与原色推导值的**逐元素差异报告**（T4 裁决的可审计证据）。
///
/// 报告两个口径的 sRGB 线性 → ACEScg 矩阵之差：
/// - **拟合**：[`crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL`]
///   （= 旧名 [`crate::colorspace::SRGB_TO_ACESCG`]，Stephen Hill 拟合件）；
/// - **推导**：`by_name("acescg").xyz_to_rgb_matrix()` ∘
///   `bradford_adaptation(D65 → D60)` ∘ `by_name("srgblinear").rgb_to_xyz_matrix()`
///   （本注册表的白点 + 原色 + Bradford 三件套）。
///
/// # 为什么要有这个 API
/// 差异**不只是一个标量**：逐元素报告才能定位「差在哪、往哪差」，
/// 便于判断某个具体渲染偏差是否由口径混用引起。测试
/// `legacy_constant_is_a_fitted_matrix_not_a_primaries_derivation`
/// 锁定 `max_abs_diff ∈ (1e-2, 5e-2)`，防止后人误以为其中之一算错。
///
/// # 确定性（C17）
/// 求和序、矩阵乘法序、Bradford 装配序全部固定；同输入逐次调用返回
/// **逐位相同**的 `f64`（无 HashMap / 无浮点排序参与）。
///
/// # 错误
/// 矩阵奇异 / Bradford 退化 → [`ColorSpaceError::SingularMatrix`]。
pub fn legacy_vs_derived_delta() -> Result<LegacyVsDerivedDelta, ColorSpaceError> {
    let acescg = by_name("acescg")?;
    let srgb_linear = by_name("srgblinear")?;
    let derived = mat3_mul(
        &acescg.xyz_to_rgb_matrix()?,
        &mat3_mul(
            &bradford_adaptation(WhitePoint::D65, WhitePoint::D60)?,
            &srgb_linear.rgb_to_xyz_matrix()?,
        ),
    );
    // 拟合常量收窄到 f64（f32 → f64 是**精确**转换，不引入误差）。
    let fitted: Mat3d = [
        [
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[0][0] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[0][1] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[0][2] as f64,
        ],
        [
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[1][0] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[1][1] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[1][2] as f64,
        ],
        [
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[2][0] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[2][1] as f64,
            crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[2][2] as f64,
        ],
    ];
    let mut elements = [[0.0f64; 3]; 3];
    let mut max_abs_diff = 0.0f64;
    let mut max_abs_diff_at = (0usize, 0usize);
    for r in 0..3 {
        for c in 0..3 {
            let d = derived[r][c] - fitted[r][c];
            elements[r][c] = d;
            let a = d.abs();
            // 严格大于 ⇒ 首个最大值胜出（确定性 tie-break，按行优先序）。
            if a > max_abs_diff {
                max_abs_diff = a;
                max_abs_diff_at = (r, c);
            }
        }
    }
    let fitted_row_sums = [
        fitted[0][0] + fitted[0][1] + fitted[0][2],
        fitted[1][0] + fitted[1][1] + fitted[1][2],
        fitted[2][0] + fitted[2][1] + fitted[2][2],
    ];
    let derived_row_sums = [
        derived[0][0] + derived[0][1] + derived[0][2],
        derived[1][0] + derived[1][1] + derived[1][2],
        derived[2][0] + derived[2][1] + derived[2][2],
    ];
    Ok(LegacyVsDerivedDelta {
        fitted,
        derived,
        elements,
        max_abs_diff,
        max_abs_diff_at,
        fitted_row_sums,
        derived_row_sums,
    })
}

/// 拟合常量 vs 原色推导值的差异报告（[`legacy_vs_derived_delta`] 的返回体）。
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyVsDerivedDelta {
    /// 拟合矩阵（Hill 拟合件，f32 常量收窄；行和精确 1）。
    pub fitted: Mat3d,
    /// 原色推导矩阵（本注册表白点 + 原色 + Bradford 装配）。
    pub derived: Mat3d,
    /// 逐元素差 `derived[r][c] - fitted[r][c]`。
    pub elements: Mat3d,
    /// 逐元素绝对差的最大值（实测 ~3.2e-2）。
    pub max_abs_diff: f64,
    /// 最大差所在位置（行, 列）；并列时取行优先序首个。
    pub max_abs_diff_at: (usize, usize),
    /// 拟合矩阵的行和（三行都应为 1，验证 D65 白点守恒）。
    pub fitted_row_sums: [f64; 3],
    /// 推导矩阵的行和（Bradford 适配后接近 1，偏离量即白点适配代价）。
    pub derived_row_sums: [f64; 3],
}

impl core::fmt::Display for LegacyVsDerivedDelta {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "T4 口径差异报告（sRGB 线性 → ACEScg）：max |Δ| = {:.6e} @ [{}][{}]",
            self.max_abs_diff, self.max_abs_diff_at.0, self.max_abs_diff_at.1
        )?;
        writeln!(f, "  逐元素差（derived − fitted）：")?;
        for r in 0..3 {
            writeln!(
                f,
                "    [{:+9.6e}, {:+9.6e}, {:+9.6e}]",
                self.elements[r][0], self.elements[r][1], self.elements[r][2]
            )?;
        }
        writeln!(
            f,
            "  行和  fitted = [{:.7}, {:.7}, {:.7}]",
            self.fitted_row_sums[0], self.fitted_row_sums[1], self.fitted_row_sums[2]
        )?;
        write!(
            f,
            "        derived = [{:.7}, {:.7}, {:.7}]",
            self.derived_row_sums[0], self.derived_row_sums[1], self.derived_row_sums[2]
        )
    }
}

/// 按表示形态查（升序）。
pub fn by_representation(rep: Representation) -> Vec<&'static ColorSpaceDef> {
    COLOR_SPACES.iter().filter(|d| d.representation == rep).collect()
}

/// 传函形态查（升序）：带 sRGB 曲线 / 线性 / 纯幂 / ACEScc 的空间。
pub fn by_transfer(tf: TransferFunction) -> Vec<&'static ColorSpaceDef> {
    COLOR_SPACES.iter().filter(|d| d.transfer == tf).collect()
}

/// 相近名建议：取"共有前缀最长 + 编辑距离最小"的候选（确定 tie-break：
/// 长度差小者优先，再按字典序）。
fn nearest_hint(key: &str) -> String {
    let mut best: Option<(usize, usize, &'static str)> = None;
    for d in COLOR_SPACES.iter() {
        let common = common_prefix_len(key, d.name);
        let dist = edit_distance(key, d.name);
        let candidate = (usize::MAX - common, dist, d.name);
        best = match best {
            None => Some(candidate),
            Some(cur) => {
                if (candidate.0, candidate.1, candidate.2) < (cur.0, cur.1, cur.2) {
                    Some(candidate)
                } else {
                    Some(cur)
                }
            }
        };
    }
    match best {
        Some((_, _, name)) => format!("已注册 {COLOR_SPACE_COUNT} 个色彩空间，最接近的候选：{name:?}"),
        None => "注册表为空".to_string(),
    }
}

/// 共有前缀长度。
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0usize;
    for (x, y) in a.chars().zip(b.chars()) {
        if x != y {
            break;
        }
        n += 1;
    }
    n
}

/// Levenshtein 编辑距离（自写，滚动行 + `min` 三路；表规模 20×len，无性能问题）。
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let del = prev[j + 1] + 1;
            let ins = cur[j] + 1;
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

// ===========================================================================
//  错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）
// ===========================================================================

/// 色彩空间注册表错误。
///
/// 错误码取 **CS 段**（color space 族），与既有 `P3xxx`（图）、
/// `PD1xxx`（分发）、`PF2xxx`（缓存清单）前缀互不重叠。
#[derive(Debug, Clone, PartialEq)]
pub struct ColorSpaceError {
    /// 错误码（CS1xxx）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 色彩空间名。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for ColorSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for ColorSpaceError {}

impl ColorSpaceError {
    /// 构造错误（四字段全填，C19 建议段不得为空）。
    #[inline]
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }

    /// CS1001：未知名。
    pub fn unknown_color_space(problem: String, hint: String) -> Self {
        Self::new(
            "CS1001",
            "colorspace_registry::by_name",
            problem,
            format!("改用已注册名（{} 个，见 names()）；{hint}", COLOR_SPACE_COUNT),
        )
    }

    /// CS1002：非法色度 / 白点坐标。
    pub fn invalid_chromaticity(problem: String) -> Self {
        Self::new(
            "CS1002",
            "colorspace_registry 色度坐标校验",
            problem,
            "白点需 x > 0、y > 0 且 x + y < 1；原色需 y != 0 且 x + y <= 1（ACES AP0 的虚原色除外）"
                .to_string(),
        )
    }

    /// CS1003：矩阵奇异。
    pub fn singular(location: impl Into<String>, problem: String) -> Self {
        Self::new(
            "CS1003",
            location,
            problem,
            "检查三原色是否退化（如三者重合）；改用真实规范原色，或改走 Gray 表示形态"
                .to_string(),
        )
    }

    /// CS1004：非有限输入。
    pub fn non_finite(problem: String) -> Self {
        Self::new(
            "CS1004",
            "colorspace_registry 输入校验",
            problem,
            "先在上游剔除 NaN/Inf（如除零结果），再送入色彩转换；本模块不静默传播 NaN"
                .to_string(),
        )
    }

    /// CS1005：传递函数定义域外。
    pub fn domain(problem: String) -> Self {
        Self::new(
            "CS1005",
            "colorspace_registry 传递函数",
            problem,
            "纯幂 / ACEScc 负段只接受非负输入；如需负值扩展请改用 sRGB 曲线（线性段可延伸）或先做符号分离"
                .to_string(),
        )
    }

    /// CS1006：表示形态不适用（如对 Lab 空间调用 RGB 矩阵路径）。
    pub fn unsupported_representation(def: &ColorSpaceDef) -> Self {
        Self::new(
            "CS1006",
            format!("colorspace_registry::{}.rgb_to_xyz_matrix", def.name),
            format!(
                "表示形态 {:?} 不是 RGB 空间：RGB↔XYZ 矩阵路径对它没有定义",
                def.representation
            ),
            "改用该形态的专用入口：Lab → xyz_to_lab / lab_to_xyz，LCh → xyz_to_lch / lch_to_xyz"
                .to_string(),
        )
    }

    /// CS1007：白点未注册。
    pub fn unknown_white_point(problem: String) -> Self {
        Self::new(
            "CS1007",
            "colorspace_registry::WhitePoint::from_chromaticity",
            problem,
            "已注册白点：D65 / D50 / D60 / DCI / A / C / E；自定义白点请用 build_rgb_to_xyz 直接传坐标"
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert3_recovers_identity_product() {
        let m: Mat3d = [[0.4, 0.2, 0.1], [0.05, 0.9, 0.3], [0.2, 0.1, 0.7]];
        let inv = invert3(&m).unwrap_or_else(|e| panic!("{e}"));
        let prod = mat3_mul(&m, &inv);
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((prod[r][c] - want).abs() < 1e-12, "M·M⁻¹[{r}][{c}] = {}", prod[r][c]);
            }
        }
    }

    #[test]
    fn invert3_rejects_singular_without_panic() {
        let singular: Mat3d = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]];
        let e = invert3(&singular).expect_err("奇异矩阵必须结构化报错");
        assert_eq!(e.code, "CS1003");
        assert!(!e.fix.is_empty(), "建议段不得为空");
    }

    #[test]
    fn invert3_rejects_nan_matrix() {
        let bad: Mat3d = [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let e = invert3(&bad).expect_err("含 NaN 的矩阵必须报错");
        assert_eq!(e.code, "CS1003");
    }

    #[test]
    fn bradford_is_identity_for_same_white_point() {
        assert_eq!(bradford_adaptation(WhitePoint::D65, WhitePoint::D65).unwrap(), IDENTITY3);
    }

    #[test]
    fn bradford_maps_white_to_white() {
        let m = bradford_adaptation(WhitePoint::D65, WhitePoint::D50)
            .unwrap_or_else(|e| panic!("{e}"));
        let got = mat3_apply(&m, WhitePoint::D65.xyz());
        let want = WhitePoint::D50.xyz();
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "Bradford 应把源白点送到目标白点：{got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn bradford_d65_to_d50_matches_published_matrix() {
        // 已发表的 Bradford D65→D50 适配矩阵（1e-3 容差，f32/f64 常数转录差异）。
        let published: Mat3d = [
            [1.047_929_8, 0.022_946_8, -0.050_192_2],
            [0.029_627_8, 0.990_434_5, -0.017_073_8],
            [-0.009_243_0, 0.015_055_2, 0.751_874_3],
        ];
        let m = bradford_adaptation(WhitePoint::D65, WhitePoint::D50)
            .unwrap_or_else(|e| panic!("{e}"));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m[r][c] - published[r][c]).abs() < 1e-3,
                    "Bradford[{r}][{c}] = {}（发表值 {}）",
                    m[r][c],
                    published[r][c]
                );
            }
        }
    }

    #[test]
    fn det_atan2_matches_known_angles() {
        // 30°/45°/60°/120° 与轴向：det_trig 无 atan，此处锁自有实现。
        let pi = std::f64::consts::PI;
        for (deg, want) in [
            (30.0f64, pi / 6.0),
            (45.0, pi / 4.0),
            (60.0, pi / 3.0),
            (120.0, 2.0 * pi / 3.0),
        ] {
            let rad = deg.to_radians();
            let got = det_atan2(rad.sin(), rad.cos());
            assert!((got - want).abs() < 1e-6, "atan2({deg}°) = {got}，期望 {want}");
        }
        assert_eq!(det_atan2(0.0, 0.0), 0.0, "原点不 panic");
        assert!((det_atan2(1.0, 0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((det_atan2(-1.0, 0.0) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(det_atan2(1.0, -1.0) > 0.0, "第二象限为正");
    }

    #[test]
    fn registry_is_sorted_and_unique() {
        let n = names();
        for w in n.windows(2) {
            assert!(w[0] < w[1], "注册表必须严格升序：{:?} 之后是 {:?}", w[0], w[1]);
        }
        assert_eq!(n.len(), COLOR_SPACE_COUNT);
    }

    #[test]
    fn by_name_is_deterministic_and_tolerant() {
        for _ in 0..3 {
            let a = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
            let b = by_name("  SRGB ").unwrap_or_else(|e| panic!("{e}"));
            let c = by_name("display-p3").unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a.name, "srgb");
            assert_eq!(b.name, "srgb");
            assert_eq!(c.name, "displayp3");
        }
    }

    #[test]
    fn unknown_name_is_structured_error() {
        let e = by_name("rec2100").expect_err("未知名必须报错");
        assert_eq!(e.code, "CS1001");
        assert!(!e.fix.is_empty());
    }

    #[test]
    fn srgb_transfer_function_known_values() {
        let tf = TransferFunction::Srgb;
        // 0.5 存储值 → 约 0.2140 线性（sRGB 规范算例）。
        let lin = tf.decode(0.5).unwrap_or_else(|e| panic!("{e}"));
        assert!((lin - 0.214_041).abs() < 1e-5, "sRGB 解码 0.5 → {lin}");
        // 线性段：0.04 / 12.92。
        let lin2 = tf.decode(0.04).unwrap_or_else(|e| panic!("{e}"));
        assert!((lin2 - 0.04 / 12.92).abs() < 1e-9);
        // 往返。
        for i in 0..=100u32 {
            let v = i as f64 / 100.0;
            let back = tf.encode(tf.decode(v).unwrap_or_else(|e| panic!("{e}"))).unwrap_or_else(|e| panic!("{e}"));
            assert!((back - v).abs() < 1e-6, "sRGB 往返 v = {v} → {back}");
        }
    }

    #[test]
    fn pure_gamma_rejects_negative_input() {
        let e = TransferFunction::PureGamma { exponent: 2.2 }
            .decode(-0.5)
            .expect_err("负底幂必须结构化报错");
        assert_eq!(e.code, "CS1005");
    }

    #[test]
    fn acescc_matches_published_known_values() {
        // ACEScc 的公开参考值（正段口径）。
        let tf = TransferFunction::Acescc;
        let one = tf.encode(1.0).unwrap_or_else(|e| panic!("{e}"));
        assert!((one - 0.850_449_6).abs() < 1e-6, "ACEScc(1.0) = {one}，公开值 0.8504496");
        let mid = tf.encode(0.18).unwrap_or_else(|e| panic!("{e}"));
        assert!((mid - 0.169_810_1).abs() < 1e-6, "ACEScc(0.18) = {mid}，公开值 0.1698101");
        // 0 是断点：编码值恰为 a/c。
        let zero = tf.encode(0.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (zero - ACESCC_A / ACESCC_C).abs() < 1e-12,
            "ACEScc 断点编码值应为 a/c：{zero}"
        );
    }

    #[test]
    fn acescc_roundtrip_is_exactly_invertible_and_monotone() {
        let tf = TransferFunction::Acescc;
        // 精确可逆：全定义域扫描（负段端点 → 正段大值）。
        let mut v = -ACESCC_A;
        while v < 4.0 {
            let e = tf.encode(v).unwrap_or_else(|e| panic!("{e}"));
            let d = tf.decode(e).unwrap_or_else(|e| panic!("{e}"));
            assert!(
                (d - v).abs() < 1e-6,
                "ACEScc 必须可逆：x = {v} → y = {e} → x = {d}"
            );
            v += 0.000_7;
        }
        // 断点连续：x → 0⁻ 与 0⁺ 的编码值应几乎相等。
        let left = tf.encode(-1e-12).unwrap_or_else(|e| panic!("{e}"));
        let right = tf.encode(0.0).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (left - right).abs() < 1e-9,
            "ACEScc 正负段应在 x = 0 连续：{left} vs {right}"
        );
        // 单调递增（编码域不得折叠）。
        let mut prev = f64::NEG_INFINITY;
        let mut w = -ACESCC_A;
        while w < 4.0 {
            let e = tf.encode(w).unwrap_or_else(|e| panic!("{e}"));
            assert!(e >= prev, "ACEScc 编码必须单调：w = {w} → {e} < {prev}");
            prev = e;
            w += 0.000_7;
        }
    }

    #[test]
    fn acescc_rejects_out_of_domain_input() {
        let tf = TransferFunction::Acescc;
        // 编码域负值 → 无定义。
        assert_eq!(tf.decode(-0.5).expect_err("负编码值必须报错").code, "CS1005");
        // 线性值低于最负可表示点 → 无实数解。
        let e = tf.encode(-1.0).expect_err("低于 -a 必须报错");
        assert_eq!(e.code, "CS1005");
    }

    // ── BT.2020 SDR OETF（ITU-R BT.2020-2 §4.1）──────────────────────
    // 方向口径：`encode` = OETF（线性 E → 编码 E'），`decode` = EOTF
    // （编码 E' → 线性 E），与本注册表其余传递函数同一约定。

    /// 测试内 libm 参考 OETF（仅用于对拍，生产路径走 D-29 `det_pow`）。
    fn bt2020_ref_oetf(e: f64) -> f64 {
        if e >= BT2020_BETA {
            BT2020_ALPHA * e.powf(0.45) - (BT2020_ALPHA - 1.0)
        } else {
            4.5 * e
        }
    }

    /// 测试内 libm 参考 EOTF。
    fn bt2020_ref_eotf(ep: f64) -> f64 {
        if ep >= BT2020_EOTF_BREAK {
            ((ep + (BT2020_ALPHA - 1.0)) / BT2020_ALPHA).powf(1.0 / 0.45)
        } else {
            ep / 4.5
        }
    }

    #[test]
    fn bt2020_oetf_and_eotf_roundtrip_consistently() {
        let tf = TransferFunction::Bt2020;
        // 判据：往返容差 1e-6。实测最坏 ~3.3e-16（f64 解析式），
        // 经 D-29 f32 det_pow 后放大到 ~1e-7 量级，仍远低于阈值。
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f64;
        for i in 0..=10_000u32 {
            let e = i as f64 / 10_000.0;
            let enc = tf.encode(e).unwrap_or_else(|err| panic!("{err}"));
            let back = tf.decode(enc).unwrap_or_else(|err| panic!("{err}"));
            let d = (back - e).abs();
            if d > worst {
                worst = d;
                worst_at = e;
            }
        }
        assert!(
            worst < 1e-6,
            "BT.2020 正向往返最坏误差 {worst:e} @ E={worst_at}，应 < 1e-6"
        );
        // 反向（编码域 → 线性 → 编码）同样必须可逆。
        let mut worst2 = 0.0f64;
        for i in 0..=10_000u32 {
            let ep = i as f64 / 10_000.0;
            let lin = tf.decode(ep).unwrap_or_else(|err| panic!("{err}"));
            let back = tf.encode(lin).unwrap_or_else(|err| panic!("{err}"));
            worst2 = worst2.max((back - ep).abs());
        }
        assert!(worst2 < 1e-6, "BT.2020 反向往返最坏误差 {worst2:e}，应 < 1e-6");
    }

    #[test]
    fn bt2020_oetf_is_continuous_at_beta() {
        let tf = TransferFunction::Bt2020;
        // **E = β 处的连续性**（硬要求：两侧极限差 < 1e-9）。
        // 规范取值的 `α`/`β` 使两段**恒等连续**：`α·β^0.45 − (α−1) = 4.5·β`。
        // 实测 f64 缺口 2.39e-15 —— 那是「规范值只有 12 位有效数字」带来的
        // 残差，不是实现误差。阈值取 1e-9 留足余量。
        let h = 1e-11;
        let left = bt2020_ref_oetf(BT2020_BETA - h);
        let right = bt2020_ref_oetf(BT2020_BETA + h);
        let gap = (left - right).abs();
        assert!(
            gap < 1e-9,
            "参考实现在 E=β 处不连续：左极限 {left} vs 右极限 {right}（差 {gap:e}）"
        );
        // 生产路径的两侧极限（经 D-29 det_pow，f32 进出 ⇒ 阈值放宽到 1e-6）。
        let l_got = tf.encode(BT2020_BETA - 1e-9).unwrap_or_else(|e| panic!("{e}"));
        let r_got = tf.encode(BT2020_BETA + 1e-9).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (l_got - r_got).abs() < 1e-6,
            "BT.2020 OETF 在 E=β 处应连续：{l_got} vs {r_got}"
        );
        // 断点两侧的**分支归属**也确认：β 本身走幂段（`>=` 而非 `>`）。
        // 注意生产路径经 D-29 `det_pow`（f32 进出），故只能按容差比对，
        // 不能 `assert_eq` —— 实测偏差 1.6e-8 正是 f32 舍入口径。
        let at_beta = tf.encode(BT2020_BETA).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (at_beta - bt2020_ref_oetf(BT2020_BETA)).abs() < 1e-6,
            "β 本身应走幂段：{at_beta} vs 参考 {}",
            bt2020_ref_oetf(BT2020_BETA)
        );
        // 幂段值 ≈ 线性段值 4.5β（连续性的数值体现）。
        assert!(
            (bt2020_ref_oetf(BT2020_BETA) - 4.5 * BT2020_BETA).abs() < 1e-9,
            "β 处的幂段值应等于线性段值 4.5β"
        );
        // 生产路径在 β 处同样贴近 4.5β。
        assert!(
            (at_beta - 4.5 * BT2020_BETA).abs() < 1e-6,
            "β 处 OETF 值应 ≈ 4.5β：{at_beta}"
        );
    }

    #[test]
    fn bt2020_eotf_is_continuous_at_four_point_five_beta() {
        let tf = TransferFunction::Bt2020;
        // EOTF 侧的断点是 **4.5β**（编码域），不是 β —— 这是最容易写错的一处。
        let h = 1e-9;
        let left = bt2020_ref_eotf(BT2020_EOTF_BREAK - h);
        let right = bt2020_ref_eotf(BT2020_EOTF_BREAK + h);
        assert!(
            (left - right).abs() < 1e-8,
            "EOTF 在 4.5β={} 处应连续：{left} vs {right}",
            BT2020_EOTF_BREAK
        );
        // 断点值恰为 β。
        assert!(
            (left - BT2020_BETA).abs() < 1e-9,
            "4.5β 处的 EOTF 值应等于 β：{left}"
        );
        // 生产路径：两侧极限接近。
        let l_got = tf.decode(BT2020_EOTF_BREAK - 1e-7).unwrap_or_else(|e| panic!("{e}"));
        let r_got = tf.decode(BT2020_EOTF_BREAK + 1e-7).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (l_got - r_got).abs() < 1e-6,
            "BT.2020 EOTF 在断点处应连续：{l_got} vs {r_got}"
        );
    }

    #[test]
    fn bt2020_endpoints_are_exact_zero_and_one() {
        let tf = TransferFunction::Bt2020;
        // 端点（硬要求）。
        assert_eq!(tf.encode(0.0).unwrap_or_else(|e| panic!("{e}")), 0.0, "E=0 → 0");
        assert!((tf.encode(1.0).unwrap_or_else(|e| panic!("{e}")) - 1.0).abs() < 1e-5);
        assert_eq!(tf.decode(0.0).unwrap_or_else(|e| panic!("{e}")), 0.0, "E'=0 → 0");
        assert!((tf.decode(1.0).unwrap_or_else(|e| panic!("{e}")) - 1.0).abs() < 1e-5);
        // 线性段精确性：E = β/2 恰在 4.5 斜率上（无浮点幂介入）。
        let half = BT2020_BETA / 2.0;
        let got = tf.encode(half).unwrap_or_else(|e| panic!("{e}"));
        assert!((got - 4.5 * half).abs() < 1e-12, "线性段应精确：{got}");
        // 单调递增（两个分支都得单调，合起来也单调）。
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=20_000u32 {
            let e = i as f64 / 20_000.0;
            let v = tf.encode(e).unwrap_or_else(|err| panic!("{err}"));
            assert!(v >= prev, "OETF 非单调 @{e}: {v} < {prev}");
            prev = v;
        }
        prev = f64::NEG_INFINITY;
        for i in 0..=20_000u32 {
            let ep = i as f64 / 20_000.0;
            let v = tf.decode(ep).unwrap_or_else(|err| panic!("{err}"));
            assert!(v >= prev, "EOTF 非单调 @{ep}: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn bt2020_differs_from_srgb_at_the_same_value_with_measured_gap() {
        let bt = TransferFunction::Bt2020;
        let srgb = TransferFunction::Srgb;
        // 要求：断言 BT.2020 ≠ sRGB 并给出实测差。
        // BT.2020 的 OETF 在整个 [0,1) 上**远高于** sRGB 的 EOTF ——
        // 因为两者口径本就不同（BT.2020 OETF = 线性→编码；
        // sRGB EOTF = 编码→线性）。为可比，这里统一比**同一函数方向**：
        // 两者都是「编码值 → 线性值」的解码方向。
        for &ep in &[0.04f64, 0.18, 0.5] {
            let b = bt.decode(ep).unwrap_or_else(|e| panic!("{e}"));
            let s = srgb.decode(ep).unwrap_or_else(|e| panic!("{e}"));
            let d = (b - s).abs();
            assert!(
                d > 1e-3,
                "BT.2020 与 sRGB 在 E'={ep} 处应显著不同，实测仅差 {d:e}（{b} vs {s}）"
            );
        }
        // 实测差值锚点（f64 参考，供回归监控）。
        // E'=0.18：BT.2020 → 0.0476068，sRGB → 0.0272118，差 0.0203950。
        let b18 = bt.decode(0.18).unwrap_or_else(|e| panic!("{e}"));
        let s18 = srgb.decode(0.18).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (b18 - s18 - 0.020_395_0).abs() < 1e-4,
            "E'=0.18 处实测差 {}，参考 0.0203950",
            b18 - s18
        );
        // 编码方向（线性 → 编码）同样：BT.2020 OETF(0.18) = 0.4088481，
        // sRGB OETF(0.18) = 1.055·0.18^(1/2.4) − 0.055 ≈ 0.4614。
        let bo = bt.encode(0.18).unwrap_or_else(|e| panic!("{e}"));
        let so = srgb.encode(0.18).unwrap_or_else(|e| panic!("{e}"));
        assert!((bo - 0.408_848_1).abs() < 1e-4, "BT.2020 OETF(0.18) = {bo}");
        assert!(
            (bo - so).abs() > 0.01,
            "编码方向亦应不同：BT.2020 {bo} vs sRGB {so}"
        );
    }

    #[test]
    fn bt2020_matches_the_libm_reference_within_d29_precision() {
        // 对拍 libm 参考（D-29 det_pow 的 f32 进出误差 ~1e-7）。
        let tf = TransferFunction::Bt2020;
        let mut worst_enc = 0.0f64;
        let mut worst_dec = 0.0f64;
        for i in 0..=2_000u32 {
            let e = i as f64 / 2_000.0;
            let got = tf.encode(e).unwrap_or_else(|err| panic!("{err}"));
            worst_enc = worst_enc.max((got - bt2020_ref_oetf(e)).abs());
            let ep = i as f64 / 2_000.0;
            let gotd = tf.decode(ep).unwrap_or_else(|err| panic!("{err}"));
            worst_dec = worst_dec.max((gotd - bt2020_ref_eotf(ep)).abs());
        }
        assert!(worst_enc < 1e-6, "OETF 对拍偏差 {worst_enc:e}，应 < 1e-6");
        assert!(worst_dec < 1e-6, "EOTF 对拍偏差 {worst_dec:e}，应 < 1e-6");
    }

    #[test]
    fn rec2020_def_is_wired_to_the_bt2020_transfer_function() {
        // registry 集成：`rec2020` 必须携带 BT.2020 SDR OETF。
        let def = by_name("rec2020").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            def.transfer,
            TransferFunction::Bt2020,
            "rec2020 的传递函数必须是 Bt2020（此前误标 Linear）"
        );
        // 通过 def 的公开接口转换，验证接线真的通。
        let lin = def.decode([0.18, 0.18, 0.18]).unwrap_or_else(|e| panic!("{e}"));
        for v in lin.iter() {
            assert!(
                (*v - 0.047_606_8).abs() < 1e-4,
                "rec2020 解码 0.18 应 ≈ 0.0476068，实测 {v}"
            );
        }
        // by_transfer 分组查询能查到它。
        assert!(
            by_transfer(TransferFunction::Bt2020)
                .iter()
                .any(|d| d.name == "rec2020"),
            "by_transfer(Bt2020) 应包含 rec2020"
        );
    }

    #[test]
    fn rec2020_convert_is_same_source_as_the_hand_assembled_matrix() {
        // 判据：`convert` 与手工装配矩阵同源（1e-9）。
        // 同白点（D65 → D65）⇒ 无 Bradford，路径最短、最容易定位差异。
        let rec = by_name("rec2020").unwrap_or_else(|e| panic!("{e}"));
        let lin = by_name("lineargb709").unwrap_or_else(|e| panic!("{e}"));
        // 手工装配：解码 → rec2020 矩阵 → 编码。
        // ⚠️ `convert` 的路径是「解码 → 矩阵 → **编码**」，三步缺一不可：
        // `rec2020` 的传递函数是 BT.2020 OETF（**非** Linear），故矩阵
        // 前后各有一层曲线。若只装配矩阵而漏掉编码，结果会差一个 OETF。
        let m = mat3_mul(
            &rec.xyz_to_rgb_matrix().unwrap_or_else(|e| panic!("{e}")),
            &lin.rgb_to_xyz_matrix().unwrap_or_else(|e| panic!("{e}")),
        );
        let v = [0.2f64, 0.5, 0.8];
        let got = convert(v, lin, rec).unwrap_or_else(|e| panic!("{e}"));
        let direct = rec.encode(mat3_apply(&m, lin.decode(v).unwrap_or_else(|e| panic!("{e}"))))
            .unwrap_or_else(|e| panic!("{e}"));
        for i in 0..3 {
            assert!(
                (got[i] - direct[i]).abs() < 1e-9,
                "convert 与装配矩阵应同源 [{i}]：{got:?} vs {direct:?}"
            );
        }
        // 顺带确认「漏掉编码」确实会不同（证明上面那层 encode 不是摆设）。
        let no_encode = mat3_apply(&m, lin.decode(v).unwrap_or_else(|e| panic!("{e}")));
        assert!(
            (got[0] - no_encode[0]).abs() > 1e-3,
            "漏掉编码层应得到不同结果（got={} vs no_encode={}）",
            got[0],
            no_encode[0]
        );
    }

    #[test]
    fn rec2020_preserves_black_and_white_through_the_transfer_function() {
        // 判据：黑白点保持。
        let def = by_name("rec2020").unwrap_or_else(|e| panic!("{e}"));
        let black = def.to_xyz([0.0, 0.0, 0.0]).unwrap_or_else(|e| panic!("{e}"));
        for v in black.iter() {
            assert!(v.abs() < 1e-12, "[0,0,0] → 黑，实测 {black:?}");
        }
        // 白：OETF(1) = 1 ⇒ 解码回 1 ⇒ 行和守恒 ⇒ 精确等于 D65 白点。
        let white = def.to_xyz([1.0, 1.0, 1.0]).unwrap_or_else(|e| panic!("{e}"));
        let d65 = WhitePoint::D65.xyz();
        for (g, w) in white.iter().zip(d65.iter()) {
            assert!(
                (g - w).abs() < 1e-5,
                "[1,1,1] → D65 白点，实测 {g} vs {w}"
            );
        }
        // 端到端：[0,0,0] → rec2020 编码 → 解码 → 仍是黑。
        let enc = def.encode([0.0, 0.0, 0.0]).unwrap_or_else(|e| panic!("{e}"));
        let back = def.decode(enc).unwrap_or_else(|e| panic!("{e}"));
        for v in back.iter() {
            assert!(v.abs() < 1e-12, "黑白往返后应仍为黑：{back:?}");
        }
    }

    #[test]
    fn srgb_to_xyz_matches_known_values() {
        // sRGB 原色矩阵的公认数值：白 → (0.9505, 1.0, 1.089)，
        // 红 → (0.4124, 0.2127, 0.0193)。
        let srgb = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
        let white = srgb.to_xyz([1.0, 1.0, 1.0]).unwrap_or_else(|e| panic!("{e}"));
        for (got, want) in white.iter().zip(WhitePoint::D65.xyz().iter()) {
            assert!((got - want).abs() < 1e-6, "白点 XYZ：{got} vs {want}");
        }
        let lin = by_name("srgblinear").unwrap_or_else(|e| panic!("{e}"));
        let red = lin.to_xyz([1.0, 0.0, 0.0]).unwrap_or_else(|e| panic!("{e}"));
        for (got, want) in red.iter().zip([0.412_390_8, 0.212_639_0, 0.019_330_8].iter()) {
            assert!((got - want).abs() < 1e-6, "sRGB 线性红原色 XYZ：{got} vs {want}");
        }
    }

    #[test]
    fn lab_lch_roundtrip_within_tolerance() {
        // 只取**色域内**的 Lab（CIE 1976 的 Lab→XYZ 要求 XYZ 分量非负，
        // 色域外 Lab 属于无定义区，另见 out_of_gamut_lab_is_rejected）。
        for lab in [[0.0, 0.0, 0.0], [50.0, 20.0, -30.0], [100.0, 0.0, 0.0], [37.0, -5.0, 40.0]] {
            let xyz = lab_to_xyz(lab, WhitePoint::D65).unwrap_or_else(|e| panic!("{e}"));
            let back = xyz_to_lab(xyz, WhitePoint::D65).unwrap_or_else(|e| panic!("{e}"));
            for i in 0..3 {
                assert!((back[i] - lab[i]).abs() < 1e-3, "Lab 往返 [{i}]：{} vs {}", back[i], lab[i]);
            }
            let lch = lab_to_lch(back);
            let back2 = lch_to_lab(lch);
            for i in 0..3 {
                assert!((back2[i] - back[i]).abs() < 1e-3, "LCh 往返 [{i}]：{} vs {}", back2[i], back[i]);
            }
            assert!((0.0..360.0).contains(&lch[2]), "色相应归一到 [0,360)：{}", lch[2]);
        }
    }

    #[test]
    fn lab_lch_roundtrip_handles_negative_l_without_panic() {
        // L* 略低于 0（Y 为极小正数时的允许外插区），不得 panic。
        let xyz = [1e-9, 1e-9, 1e-9];
        let lab = xyz_to_lab(xyz, WhitePoint::D65).unwrap_or_else(|e| panic!("{e}"));
        let back = lab_to_xyz(lab, WhitePoint::D65).unwrap_or_else(|e| panic!("{e}"));
        assert!((back[1] - 1e-9).abs() < 1e-11);
    }

    #[test]
    fn out_of_gamut_lab_is_rejected_not_silently_clipped() {
        // 强烈色偏（如 a = 80）对应的 XYZ 含负分量 → CIE 1976 无定义。
        // 本模块**明确报错**，绝不静默裁剪（宪章 C20）。
        // b* = 130 时 Z 分量落到负区（实测 X/Y/Z = 0.175 / 0.184 / -0.007）。
        let e = lab_to_xyz([50.0, 0.0, 130.0], WhitePoint::D65).expect_err("色域外 Lab 必须报错");
        assert_eq!(e.code, "CS1005");
    }

    #[test]
    fn xyz_lab_rejects_negative_components() {
        let e = xyz_to_lab([-0.1, 0.5, 0.5], WhitePoint::D65).expect_err("负分量必须报错");
        assert_eq!(e.code, "CS1005");
    }

    #[test]
    fn lab_space_rejects_rgb_matrix_path() {
        let lab = by_name("cielab").unwrap_or_else(|e| panic!("{e}"));
        let e = lab.to_xyz([50.0, 10.0, 10.0]).expect_err("Lab 空间不走 RGB 矩阵");
        assert_eq!(e.code, "CS1006");
        assert!(e.fix.contains("xyz_to_lab"));
    }

    #[test]
    fn gray_space_neutral_axis() {
        let g = by_name("lineargray").unwrap_or_else(|e| panic!("{e}"));
        let xyz = g.to_xyz([0.5, 0.5, 0.5]).unwrap_or_else(|e| panic!("{e}"));
        let w = WhitePoint::D65.xyz();
        assert!((xyz[1] - 0.5).abs() < 1e-9, "灰度 0.5 的 Y 应为 0.5：{}", xyz[1]);
        assert!((xyz[0] / w[0] - 0.5).abs() < 1e-9);
        let back = g.xyz_to_gray(xyz).unwrap_or_else(|e| panic!("{e}"));
        assert!((back - 0.5).abs() < 1e-9);
    }

    #[test]
    fn white_point_from_chromaticity_and_its_rejection() {
        assert_eq!(
            WhitePoint::from_chromaticity(0.3127, 0.3290).unwrap_or_else(|e| panic!("{e}")),
            WhitePoint::D65
        );
        // 未注册的合法坐标 → 结构化错误（不近似）。
        let e = WhitePoint::from_chromaticity(0.300, 0.300).expect_err("未注册白点必须报错");
        assert_eq!(e.code, "CS1007");
        // 非法坐标 → 结构化错误。
        let e2 = WhitePoint::from_chromaticity(0.9, 0.9).expect_err("x+y>1 必须报错");
        assert_eq!(e2.code, "CS1002");
    }

    #[test]
    fn build_rgb_to_xyz_rejects_degenerate_primaries() {
        let wp = WhitePoint::D65;
        // 三原色全部重合 → M 奇异。
        let p = Primaries::new(Chromaticity::new(0.64, 0.33), Chromaticity::new(0.64, 0.33), Chromaticity::new(0.64, 0.33));
        let e = build_rgb_to_xyz(&p, wp).expect_err("退化原色必须报错");
        assert_eq!(e.code, "CS1003");
    }

    #[test]
    fn validate_primaries_allows_aces_ap0_imaginary_primaries() {
        // AP0 的 G（x = 0, y = 1）与 B（x = 0.0001, y = -0.077）在光谱轨迹外，
        // 但必须被接受（否则 ACES 2065-1 无法注册）。
        validate_primaries_chromaticity(0.0, 1.0).expect("AP0 绿原色应合法");
        validate_primaries_chromaticity(0.0001, -0.077).expect("AP0 蓝原色应合法");
        assert!(validate_primaries_chromaticity(0.5, 0.0).is_err(), "y = 0 必须拒绝");
    }

    #[test]
    fn query_by_white_point_is_sorted_and_stable() {
        let first = by_white_point(WhitePoint::D65);
        assert!(!first.is_empty());
        for w in first.windows(2) {
            assert!(w[0].name < w[1].name, "按白点查询也必须升序");
        }
        let second = by_white_point(WhitePoint::D65);
        assert_eq!(
            first.iter().map(|d| d.name).collect::<Vec<_>>(),
            second.iter().map(|d| d.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_space_roundtrips_and_preserves_white() {
        for def in definitions() {
            // 白点守恒（RGB/Gray 形态）：线性 1.0 → 本空间白点（Y = 1 归一）。
            // 注意：**log 编码空间（ACEScc）的「白」在编码域不是 1.0**
            //（白对应 ACEScc(1.0) = 0.8504496），故统一用「先编码再解码」
            // 的样本，保证测试的是色彩学而非编码数字。
            if matches!(def.representation, Representation::Rgb | Representation::Gray) {
                let one = def.encode([1.0, 1.0, 1.0]).unwrap_or_else(|e| panic!("{}: {e}", def.name));
                let white = def.to_xyz(one).unwrap_or_else(|e| panic!("{}: {e}", def.name));
                for (got, want) in white.iter().zip(def.white_point.xyz().iter()) {
                    assert!(
                        (got - want).abs() < 1e-6,
                        "{} 白点守恒失败：{got} vs {want}",
                        def.name
                    );
                }
            }
            // XYZ 往返（Lab/LCh 空间除外：它们不走 RGB 路径）。
            if matches!(def.representation, Representation::CieLab | Representation::CieLch) {
                continue;
            }
            // 用该空间基内的一个确定样本点做往返（不取 [1,1,1]，
            // 避免 sRGB 曲线的 1.0 边界与纯幂的饱和点掩盖问题）。
            let sample = match def.representation {
                Representation::Xyz => [0.2, 0.3, 0.4],
                _ => [0.25, 0.5, 0.75],
            };
            if def.transfer == TransferFunction::Acescc {
                // ACEScc 的基内是「编码值」域，样本取解码后落回自身的点。
                let lin = [0.2, 0.35, 0.6];
                let enc = def.encode(lin).unwrap_or_else(|e| panic!("{}: {e}", def.name));
                let back = def
                    .from_xyz(def.to_xyz(enc).unwrap_or_else(|e| panic!("{}: {e}", def.name)))
                    .unwrap_or_else(|e| panic!("{}: {e}", def.name));
                for i in 0..3 {
                    assert!((back[i] - enc[i]).abs() < 1e-3, "{} ACEScc 往返 [{i}]：{} vs {}", def.name, back[i], enc[i]);
                }
                continue;
            }
            let xyz = def.to_xyz(sample).unwrap_or_else(|e| panic!("{}: {e}", def.name));
            let back = def.from_xyz(xyz).unwrap_or_else(|e| panic!("{}: {e}", def.name));
            for i in 0..3 {
                assert!(
                    (back[i] - sample[i]).abs() < 1e-4,
                    "{} 往返 [{i}]：{} vs {}",
                    def.name,
                    back[i],
                    sample[i]
                );
            }
        }
    }

    #[test]
    fn same_white_convert_is_matrix_only_and_roundtrips() {
        let a = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
        let b = by_name("rec2020").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.white_point, b.white_point, "本用例要求同白点");
        let c = [0.2f64, 0.5, 0.9];
        let there = convert(c, a, b).unwrap_or_else(|e| panic!("{e}"));
        let back = convert(there, b, a).unwrap_or_else(|e| panic!("{e}"));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "同白点往返 [{i}]：{} vs {}", back[i], c[i]);
        }
    }

    #[test]
    fn cross_white_convert_invokes_bradford_and_roundtrips() {
        let a = by_name("srgb").unwrap_or_else(|e| panic!("{e}")); // D65
        let b = by_name("prophoto").unwrap_or_else(|e| panic!("{e}")); // D50
        assert_ne!(a.white_point, b.white_point, "本用例要求跨白点");
        let c = [0.3f64, 0.4, 0.6];
        let there = convert(c, a, b).unwrap_or_else(|e| panic!("{e}"));
        let back = convert(there, b, a).unwrap_or_else(|e| panic!("{e}"));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-3, "跨白点往返 [{i}]：{} vs {}", back[i], c[i]);
        }
    }

    #[test]
    fn d50_white_reaches_known_xyz_via_bradford() {
        // 已知值：sRGB 纯白 (1,1,1) 适配到 D50 后约 (0.9642, 1.0, 0.8249)。
        let srgb = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
        let got = srgb
            .to_xyz_adapted([1.0, 1.0, 1.0], WhitePoint::D50)
            .unwrap_or_else(|e| panic!("{e}"));
        let want = WhitePoint::D50.xyz();
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-6, "D65→D50 白点适配 [{i}]：{got:?} vs {want:?}");
        }
        assert!((got[0] - 0.964_22).abs() < 1e-4);
        assert!((got[2] - 0.825_21).abs() < 1e-4);
    }

    #[test]
    fn registry_acescg_matches_official_aces_matrix() {
        // ACES 官方文档公布的 sRGB→ACEScg 矩阵（D65 sRGB → D60 AP1，
        // 含 Bradford 适配）。本注册表由「白点 + 原色 + Bradford」独立推导，
        // 应收敛到官方值 ~1e-5（官方值只印 4 位小数）。
        let official: Mat3d = [
            [0.613_097, 0.339_523, 0.047_379],
            [0.070_194, 0.916_354, 0.013_452],
            [0.020_616, 0.109_570, 0.869_815],
        ];
        let got = convert(
            [0.2f64, 0.5, 0.8],
            &by_name("srgblinear").unwrap_or_else(|e| panic!("{e}")),
            &by_name("acescg").unwrap_or_else(|e| panic!("{e}")),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let m = mat3_mul(
            &by_name("acescg")
                .unwrap_or_else(|e| panic!("{e}"))
                .xyz_to_rgb_matrix()
                .unwrap_or_else(|e| panic!("{e}")),
            &bradford_adaptation(WhitePoint::D65, WhitePoint::D60).unwrap_or_else(|e| panic!("{e}")),
        );
        let m = mat3_mul(
            &m,
            &by_name("srgblinear")
                .unwrap_or_else(|e| panic!("{e}"))
                .rgb_to_xyz_matrix()
                .unwrap_or_else(|e| panic!("{e}")),
        );
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m[r][c] - official[r][c]).abs() < 1e-4,
                    "注册表推导的 sRGB→ACEScg[{r}][{c}] = {}，ACES 官方 = {}",
                    m[r][c],
                    official[r][c]
                );
            }
        }
        // 抽查向量：装配矩阵作用于 sRGB 线性值应等于 convert() 的结果。
        let v = [0.2f64, 0.5, 0.8];
        let direct = super::mat3_apply(&m, v);
        for i in 0..3 {
            assert!(
                (got[i] - direct[i]).abs() < 1e-9,
                "convert 与装配矩阵应同源 [{i}]：{got:?} vs {direct:?}"
            );
        }
    }

    #[test]
    fn legacy_constant_is_a_fitted_matrix_not_a_primaries_derivation() {
        // ⚠️ 见**文件头「⚠️⚠️⚠️ 警告」块**（第 24~44 行）：既有
        // `colorspace::SRGB_TO_ACESCG` 是**拟合矩阵**（行和 = 1，但并非
        // 由 AP1 原色推导），与本注册表的推导值口径不同，两者**不可混用**。
        // 本测试**锁定这个已知差异**，防止后人误以为其中之一算错了。
        let registry = mat3_mul(
            &by_name("acescg")
                .unwrap_or_else(|e| panic!("{e}"))
                .xyz_to_rgb_matrix()
                .unwrap_or_else(|e| panic!("{e}")),
            &mat3_mul(
                &bradford_adaptation(WhitePoint::D65, WhitePoint::D60)
                    .unwrap_or_else(|e| panic!("{e}")),
                &by_name("srgblinear")
                    .unwrap_or_else(|e| panic!("{e}"))
                    .rgb_to_xyz_matrix()
                    .unwrap_or_else(|e| panic!("{e}")),
            ),
        );
        let legacy = crate::colorspace::SRGB_TO_ACESCG;
        let mut worst = 0.0f64;
        for r in 0..3 {
            for c in 0..3 {
                worst = worst.max((registry[r][c] - legacy[r][c] as f64).abs());
            }
        }
        // 差异显著（~3e-2 量级），但两者都把白点送到白点（行和 = 1）。
        assert!(
            worst > 1e-2 && worst < 5e-2,
            "与既有拟合常量的差异应落在 ~3e-2 量级，实际 {worst}"
        );
        for r in 0..3 {
            let row_sum: f32 = legacy[r][0] + legacy[r][1] + legacy[r][2];
            assert!((row_sum - 1.0).abs() < 1e-4, "既有常量行和仍应为 1：{row_sum}");
        }
    }

    #[test]
    fn srgb_to_acescg_fitted_hill_alias_is_bit_identical_to_the_legacy_name() {
        // T4 裁决落地：旧名 `SRGB_TO_ACESCG` **原样保留**（数值一字不改），
        // 新名 `SRGB_TO_ACESCG_FITTED_HILL` 是**同一份字面量的别名**。
        // 本测试用 `to_bits()` 逐位锁定等价性 —— 这是「没偷换数值」的
        // 可执行证据：任何一方被改动都会红。
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    crate::colorspace::SRGB_TO_ACESCG[r][c].to_bits(),
                    crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[r][c].to_bits(),
                    "[{r}][{c}] 两个名字必须逐位相同（拟合矩阵不得被偷换成推导值）"
                );
            }
        }
        // 旧常量仍是**已发布**的 ACES 官方 IDT 数值（锚点，防静默漂移）。
        assert_eq!(crate::colorspace::SRGB_TO_ACESCG[0][0], 0.597_190_2);
        assert_eq!(crate::colorspace::SRGB_TO_ACESCG[1][1], 0.908_340_4);
        assert_eq!(crate::colorspace::SRGB_TO_ACESCG[2][2], 0.837_769_2);
        // 拟合口径的「白点守恒」特征：行和精确为 1。
        for r in 0..3 {
            let s: f32 = crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[r][0]
                + crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[r][1]
                + crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL[r][2];
            assert!((s - 1.0).abs() < 1e-6, "拟合矩阵行和应精确为 1：{s}");
        }
    }

    #[test]
    fn legacy_vs_derived_delta_api_reports_every_element_and_the_max() {
        // T4 裁决落地的**可审计证据 API**：`legacy_vs_derived_delta()`
        // 返回逐元素差 + 最大差 + 最大差位置 + 两侧行和。
        let rep = legacy_vs_derived_delta().unwrap_or_else(|e| panic!("{e}"));
        // 最大差落在 ~3.2e-2 量级（与既有测试锁定同一区间）。
        assert!(
            rep.max_abs_diff > 1e-2 && rep.max_abs_diff < 5e-2,
            "max_abs_diff 应落在 ~3.2e-2，实测 {}",
            rep.max_abs_diff
        );
        // 逐元素差必须**逐个**等于 `derived − fitted`（不能只对 max）。
        for r in 0..3 {
            for c in 0..3 {
                let want = rep.derived[r][c] - rep.fitted[r][c];
                assert_eq!(
                    rep.elements[r][c].to_bits(),
                    want.to_bits(),
                    "elements[{r}][{c}] 必须是 derived − fitted"
                );
            }
        }
        // 最大差位置自洽：`elements` 在该位置的绝对值 == `max_abs_diff`。
        let (mr, mc) = rep.max_abs_diff_at;
        assert_eq!(
            rep.elements[mr][mc].abs().to_bits(),
            rep.max_abs_diff.to_bits(),
            "max_abs_diff_at 指向的位置必须真的是最大差"
        );
        // 拟合侧行和精确为 1；推导侧因 Bradford 适配略有偏离。
        for r in 0..3 {
            assert!(
                (rep.fitted_row_sums[r] - 1.0).abs() < 1e-6,
                "拟合矩阵行和应精确为 1：{}",
                rep.fitted_row_sums[r]
            );
        }
        // Display 可用且含关键数字（供 CLI / 调试直接打印）。
        let text = rep.to_string();
        assert!(text.contains("T4"), "Display 应含标题：{text}");
        assert!(text.contains("fitted"), "Display 应含拟合行和：{text}");
        assert!(text.contains("derived"), "Display 应含推导行和：{text}");
    }

    #[test]
    fn legacy_vs_derived_delta_is_bitwise_deterministic_across_calls() {
        // C17：同输入逐次调用必须逐位相同（无 HashMap / 无浮点排序参与）。
        let a = legacy_vs_derived_delta().unwrap_or_else(|e| panic!("{e}"));
        let b = legacy_vs_derived_delta().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "差异报告两次调用必须完全相同");
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    a.elements[r][c].to_bits(),
                    b.elements[r][c].to_bits(),
                    "elements[{r}][{c}] 必须逐位确定"
                );
            }
        }
        assert_eq!(a.max_abs_diff.to_bits(), b.max_abs_diff.to_bits());
        assert_eq!(a.max_abs_diff_at, b.max_abs_diff_at);
    }

    #[test]
    fn f32_adapters_work_and_are_explicit_narrowing() {
        let srgb = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
        let m = srgb.rgb_to_xyz_mat3f32().unwrap_or_else(|e| panic!("{e}"));
        assert!((m[0][0] - 0.412_4 as f32).abs() < 1e-3);
        let v = srgb
            .to_xyz_vec3(crate::Vec3::new(1.0, 1.0, 1.0))
            .unwrap_or_else(|e| panic!("{e}"));
        let w = WhitePoint::D65.xyz();
        assert!((v.x as f64 - w[0]).abs() < 1e-6);
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        let srgb = by_name("srgb").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(srgb.to_xyz([f64::NAN, 0.0, 0.0]).expect_err("NaN 必须报错").code, "CS1004");
        assert_eq!(srgb.from_xyz([0.0, f64::INFINITY, 0.0]).expect_err("Inf 必须报错").code, "CS1004");
        assert_eq!(adapt_white([f64::NAN, 0.0, 0.0], WhitePoint::D65, WhitePoint::D50)
            .expect_err("NaN 必须报错")
            .code, "CS1004");
    }

    #[test]
    fn aliases_are_numerically_identical() {
        let a = by_name("srgblinear").unwrap_or_else(|e| panic!("{e}"));
        let b = by_name("lineargb709").unwrap_or_else(|e| panic!("{e}"));
        let c = by_name("rec709").unwrap_or_else(|e| panic!("{e}"));
        let s = [0.2f64, 0.4, 0.6];
        let xa = a.to_xyz(s).unwrap_or_else(|e| panic!("{e}"));
        let xb = b.to_xyz(s).unwrap_or_else(|e| panic!("{e}"));
        let xc = c.to_xyz(s).unwrap_or_else(|e| panic!("{e}"));
        for i in 0..3 {
            assert!((xa[i] - xb[i]).abs() < 1e-12, "srgblinear vs lineargb709 [{i}]");
            assert!((xa[i] - xc[i]).abs() < 1e-12, "srgblinear vs rec709 [{i}]");
        }
    }

    #[test]
    fn transfer_and_representation_queries_are_sorted() {
        for group in [
            by_representation(Representation::Rgb),
            by_representation(Representation::Gray),
            by_representation(Representation::Xyz),
            by_transfer(TransferFunction::Srgb),
            by_transfer(TransferFunction::Linear),
            by_transfer(TransferFunction::Acescc),
        ] {
            for w in group.windows(2) {
                assert!(w[0].name < w[1].name, "分组查询也必须升序");
            }
        }
        assert!(by_representation(Representation::CieLab).iter().any(|d| d.name == "cielab"));
        assert!(by_transfer(TransferFunction::Acescc).iter().any(|d| d.name == "acescc"));
    }

    #[test]
    fn lch_definition_roundtrips_through_its_own_api() {
        let lch = by_name("cielch").unwrap_or_else(|e| panic!("{e}"));
        let xyz = lch
            .lch_to_xyz([50.0, 40.0, 120.0])
            .unwrap_or_else(|e| panic!("{e}"));
        let back = lch.xyz_to_lch(xyz).unwrap_or_else(|e| panic!("{e}"));
        assert!((back[0] - 50.0).abs() < 1e-2, "L* 往返：{}", back[0]);
        assert!((back[1] - 40.0).abs() < 1e-2, "C 往返：{}", back[1]);
        assert!((back[2] - 120.0).abs() < 1e-2, "h 往返：{}", back[2]);
    }

    #[test]
    fn name_normalization_is_separator_insensitive() {
        assert_eq!(normalize_name("Display P3"), "displayp3");
        assert_eq!(normalize_name("display-p3"), "displayp3");
        assert_eq!(normalize_name("DISPLAY_P3"), "displayp3");
        assert_eq!(normalize_name("AdobeRGB-1998"), "adobergb1998");
        assert_eq!(normalize_name("  REC709 "), "rec709");
    }
}
