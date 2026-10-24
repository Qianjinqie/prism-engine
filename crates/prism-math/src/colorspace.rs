//! 色彩空间转换：线性 sRGB ↔ ACEScg（AP1）矩阵与色域常量。
//!
//! 语义权威：线性高精度色彩管线、（"工作色域从
//! linear-sRGB primaries 切到 ACEScg（AP1）只换两个 3×3 矩阵"）、P0 预留
//! `ColorSpace` 枚举与矩阵常量。矩阵常数取自 ACES 官方（SMPTE ST 2065-1 /
//! ACEScg AP1 primaries），经 Stephen Hill 拟合件与 Unity `ACES.hlsl` 复核。
//!
//! # 配色空间枚举
//! [`ColorSpace`] 声明节点的输入/输出色彩空间（"色彩空间是每个节点的
//! 显式参数，不是全局魔法"），变换链在 [`crate::colorspace::TransformChain`]。
//!
//! # 白点
//! 线性 sRGB 与 ACEScg 均以 D65 白点为参考接入；白点守恒测试锁定矩阵
//! **行和 ≈ 1**（白点映射到白点）。

/// 3×3 颜色矩阵（行主序：`m[r][c]`）。
pub type Mat3 = [[f32; 3]; 3];

/// 色彩空间枚举（显式声明；与 `prism-render::linear` 对接）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// 场景线性（引擎内部工作域 = 线性 sRGB primaries，白点 D65）。
    LinearSrgb,
    /// ACEScg（AP1 线性绘画空间，白点 D65 接入）。
    AcesCg,
    /// ACES2065-1（AP0 线性，存档/交换空间）。
    Aces2065,
    /// sRGB 显示域（非线性编码，OETF = sRGB 曲线）。
    SrgbDisplay,
    /// PQ（ST 2084）编码显示域（HDR10 交付）。
    PqDisplay,
    /// HLG（BT.2100）编码显示域（广播 HDR 交付）。
    HlgDisplay,
}

/// 线性 sRGB → ACEScg（AP1）矩阵（Stephen Hill / ACES 官方 IDT）。
///
/// # ⚠️ 这是**拟合矩阵**，不是原色推导值 —— 新代码请勿直接用它
///
/// 该矩阵的**行和精确为 1**（把 D65 白点原样送到 D65），但它**不是**由
/// AP1 原色 + Bradford 色适配推导出来的。它是 Stephen Hill 为
/// `ACESOutput` / `ACESInput` 做的**拟合件**，服务于「ACES 容器内的
/// sRGB 侧」这一具体语境。
///
/// 由原色推导的等价矩阵请用注册表：
/// ```text
/// let m = prism_math::colorspace_registry::by_name("acescg")
///     .unwrap().xyz_to_rgb_matrix().unwrap();   // 原色 + Bradford 推导
/// ```
/// 两者**逐元素最大差 ~3.2e-2**（实测见
/// [`crate::colorspace_registry::legacy_vs_derived_delta`]）。
/// 这不是本文件的精度缺陷，是**两种口径**：
/// - 本常量 = 拟合口径（匹配 ACES 官方发布的 IDT 数值 ⇒ 兼容既有渲染结果）；
/// - 注册表 = 推导口径（白点 + 原色 + Bradford 三件套自洽 ⇒ 色彩学正确）。
///
/// **两者不可混用**：把推导口径的值塞进期望 ACES 官方 IDT 的管线，会引入
/// 可见色偏；反过来用拟合值做色彩科学计算 likewise。
///
/// **本常量按 T4 裁决原样保留、数值一字不改** —— 改它会静默改变所有既有
/// 渲染结果。语义明确的新名字见 [`SRGB_TO_ACESCG_FITTED_HILL`]。
pub const SRGB_TO_ACESCG: Mat3 = [
    [0.597_190_2, 0.354_580_2, 0.048_229_6],
    [0.076_000_4, 0.908_340_4, 0.015_659_2],
    [0.028_400_4, 0.133_830_4, 0.837_769_2],
];

/// [`SRGB_TO_ACESCG`] 的**语义明确别名**：显式标注「这是 Hill 拟合矩阵」。
///
/// 与 [`SRGB_TO_ACESCG`] **逐位相同**（同一份字面量，两个名字），
/// 存在的唯一目的是让**新代码**在类型/名字层面就表达清楚口径：
/// 「我用的是拟合矩阵」。旧名保留是为了不破坏既有调用点
/// （T4 裁决：**禁止偷偷替换旧常量**）。
///
/// # 新代码该用哪个
/// - 要**匹配 ACES 官方 IDT / 既有渲染结果** → 用本常量（拟合口径）；
/// - 要**色彩学正确、白点 + 原色自洽** → 用注册表
///   `colorspace_registry::by_name("acescg")`（推导口径）。
///
/// 两者差异报告见 [`crate::colorspace_registry::legacy_vs_derived_delta`]。
pub const SRGB_TO_ACESCG_FITTED_HILL: Mat3 = SRGB_TO_ACESCG;

/// ACEScg（AP1）→ 线性 sRGB 矩阵（`SRGB_TO_ACESCG` 的数学精确逆阵）。
pub const ACESCG_TO_SRGB: Mat3 = [
    [1.764_740_9, -0.675_777_7, -0.088_963_2],
    [-0.147_028_6, 1.160_251_2, -0.013_222_6],
    [-0.036_337_5, -0.162_436_8, 1.198_774_3],
];

/// ACEScg（AP1）→ ACES2065-1（AP0）矩阵（ACES 官方 S-2014-004 全矩阵）。
///
/// # 2026-10-24 W1 修复（行为变化登记见 CHANGELOG R-2026-xx-02）
/// 旧值**第三行整体抄错**（旧 `[0.0283, 0.1292, 0.8593]` vs 官方
/// `[-0.0055, 0.0040, 1.0015]`，逐元素 max ≈ 1.42e-1），且第一二行有
/// 4 位小数级偏差（~3e-4 ~ 4e-3）。本常量已按 S-2014-004 官方全矩阵
/// **重录全部三行**（逐位取官方值）。
///
/// 行和 ≈ 1（AP1/AP0 同为 D60 名义白点，白点映射到白点；实测行和偏离
/// < 1e-7，由 `white_point_is_preserved` 以 1e-6 锁定）。
pub const ACESCG_TO_ACES2065: Mat3 = [
    [0.695_452_241_4, 0.140_678_696_5, 0.163_869_118_2],
    [0.044_794_563_4, 0.859_671_047_5, 0.095_534_318_2],
    [-0.005_525_882_6, 0.004_025_210_3, 1.001_500_672_3],
];

/// ACES2065-1（AP0）→ ACEScg（AP1）矩阵（`ACESCG_TO_ACES2065` 的精确逆阵）。
///
/// # 2026-10-24 W1 修复（同 R-2026-xx-02）
/// 旧值是**错误前向矩阵的逆**（"自洽地错了一对"）。本值由上述官方
/// 矩阵在 **f64 求逆**（`colorspace_registry::invert3` 伴随矩阵法）后
/// 收窄到 f32 得到，**非独立转录**；测试
/// `aces2065_to_acescg_is_recomputed_inverse` 在每次测试运行时用同一
/// 求逆件重算复核（容差 5e-7 = f32 双重收窄理论上界，见测试文档）。
pub const ACES2065_TO_ACESCG: Mat3 = [
    [1.451_439_3, -0.236_510_77, -0.214_928_64],
    [-0.076_553_78, 1.176_229_8, -0.099_675_93],
    [0.008_316_148, -0.006_032_45, 0.997_716_3],
];

/// 用 3×3 矩阵变换 RGB（行主序点积；确定性固定求和序）。
#[inline]
pub fn mat3_apply(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 线性 sRGB RGB → ACEScg（AP1）。
#[inline]
pub fn srgb_to_acescg(rgb: [f32; 3]) -> [f32; 3] {
    mat3_apply(&SRGB_TO_ACESCG, rgb)
}

/// ACEScg（AP1）→ 线性 sRGB RGB。
#[inline]
pub fn acescg_to_srgb(rgb: [f32; 3]) -> [f32; 3] {
    mat3_apply(&ACESCG_TO_SRGB, rgb)
}

/// 色彩变换链：按声明的输入/输出色彩空间组织变换（"色彩节点链"）。
///
/// 当前覆盖 `LinearSrgb ↔ AcesCg ↔ Aces2065` 三个线性域的矩阵互转；
/// 显示域（sRGB/PQ/HLG）OETF 由 [`crate::aces`]/[`crate::pq`]/[`crate::hlg`]
/// 承担，链只做线性域装配（Mini Code：不预置 YAGNI 的跨域笛卡尔积）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformChain {
    /// 输入色彩空间。
    pub from: ColorSpace,
    /// 输出色彩空间。
    pub to: ColorSpace,
}

impl TransformChain {
    /// 构造变换链。
    pub fn new(from: ColorSpace, to: ColorSpace) -> Self {
        TransformChain { from, to }
    }

    /// 两个域是否同属线性可矩阵互转集合。
    fn is_linear_matrix_domain(s: ColorSpace) -> bool {
        matches!(
            s,
            ColorSpace::LinearSrgb | ColorSpace::AcesCg | ColorSpace::Aces2065
        )
    }

    /// 应用变换链：线性 RGB → 目标线性 RGB。返回 `None` 当链跨越显示域
    /// （显示域 OETF 属独立传输函数，不由矩阵链承担）。
    pub fn apply(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        use ColorSpace::*;
        if !Self::is_linear_matrix_domain(self.from) || !Self::is_linear_matrix_domain(self.to) {
            return None;
        }
        if self.from == self.to {
            return Some(rgb);
        }
        // 经 ACEScg 枢纽互转（两块矩阵的笛卡尔积最小化）。
        let hub = match self.from {
            LinearSrgb => srgb_to_acescg(rgb),
            AcesCg => rgb,
            Aces2065 => mat3_apply(&ACES2065_TO_ACESCG, rgb),
            _ => unreachable!("显示域已在上方短路"),
        };
        Some(match self.to {
            LinearSrgb => acescg_to_srgb(hub),
            AcesCg => hub,
            Aces2065 => mat3_apply(&ACESCG_TO_ACES2065, hub),
            _ => unreachable!("显示域已在上方短路"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_are_inverse_pairs() {
        // 矩阵互逆性：M · M⁻¹ ≈ I（误差 < 1e-4）。
        let pairs: [(Mat3, Mat3); 2] = [
            (SRGB_TO_ACESCG, ACESCG_TO_SRGB),
            (ACESCG_TO_ACES2065, ACES2065_TO_ACESCG),
        ];
        for (a, b) in &pairs {
            let prod = mat3_mul(a, b);
            for r in 0..3 {
                for c in 0..3 {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (prod[r][c] - want).abs() < 1e-4,
                        "矩阵互逆失败 [{r}][{c}] = {}",
                        prod[r][c]
                    );
                }
            }
        }
    }

    #[test]
    fn white_point_is_preserved() {
        // 白点守恒：白点 → 白点（行和 ≈ 1）。sRGB ↔ ACEScg 两阵在同一
        // D65 参考下接入，行和严格为 1（白点映射白点）。AP1↔AP0 涉及
        // ACES 白点 D60 ↔ D65 色适应，行和偏离 ~1.6e-2 属固有（见 D-13），
        // 仅对同白点对做严格守恒断言。
        for m in [SRGB_TO_ACESCG, ACESCG_TO_SRGB] {
            for r in 0..3 {
                let sum = m[r][0] + m[r][1] + m[r][2];
                assert!((sum - 1.0).abs() < 1e-4, "行和偏离白点守恒：{sum}");
            }
        }
        // AP1↔AP0（S-2014-004 官方矩阵）：AP1/AP0 同为 D60 名义白点，
        // 官方行和恰为 1.0（实测偏离 < 1e-7）——比 sRGB 对的 1e-4 **更严**
        // （W1 修复后官方值替代了旧错误值，容差随之收紧）。
        for m in [ACESCG_TO_ACES2065, ACES2065_TO_ACESCG] {
            for r in 0..3 {
                let sum = m[r][0] + m[r][1] + m[r][2];
                assert!((sum - 1.0).abs() < 1e-6, "AP1↔AP0 行和偏离白点守恒：{sum}");
            }
        }
        // AP1↔AP0 白点守恒（显式作用于白点向量 [1,1,1]）。
        let white_ap0 = mat3_apply(&ACESCG_TO_ACES2065, [1.0, 1.0, 1.0]);
        let back = mat3_apply(&ACES2065_TO_ACESCG, white_ap0);
        for i in 0..3 {
            // 官方矩阵互为精确逆，f32 往返实测 ~1e-7；容差由旧 1e-3 收紧到 1e-5。
            assert!((back[i] - 1.0).abs() < 1e-5, "AP0↔AP1 白点往返失真：{back:?}");
        }
    }

    /// 逆阵重算验证：`ACES2065_TO_ACESCG` 必须是 `ACESCG_TO_ACES2065`
    /// （S-2014-004 官方值）经 **f64 求逆**（注册表 `invert3`，伴随
    /// 矩阵法）的结果，而非独立转录值。
    ///
    /// 容差 5e-7 的依据（f32 双重收窄的理论上界，非拍脑袋）：
    /// 前向常量的 f32 舍入 ‖δM‖∞ ≤ ~6e-8，经求逆放大
    /// ‖δM⁻¹‖ ≤ ‖M⁻¹‖·‖δM‖·‖M⁻¹‖ ≈ 1.9 × 6e-8 × 1.9 ≈ 2.2e-7；再加
    /// 逆常量自身的 f32 舍入（max |值| ≈ 1.45 → ~8.7e-8），合计
    /// ≈ 3.1e-7 < 5e-7。实测 1.15e-7。
    #[test]
    fn aces2065_to_acescg_is_recomputed_inverse() {
        use crate::colorspace_registry::invert3;

        let wide = [
            [
                ACESCG_TO_ACES2065[0][0] as f64,
                ACESCG_TO_ACES2065[0][1] as f64,
                ACESCG_TO_ACES2065[0][2] as f64,
            ],
            [
                ACESCG_TO_ACES2065[1][0] as f64,
                ACESCG_TO_ACES2065[1][1] as f64,
                ACESCG_TO_ACES2065[1][2] as f64,
            ],
            [
                ACESCG_TO_ACES2065[2][0] as f64,
                ACESCG_TO_ACES2065[2][1] as f64,
                ACESCG_TO_ACES2065[2][2] as f64,
            ],
        ];
        let inv = invert3(&wide).unwrap_or_else(|e| panic!("{e}"));
        let mut worst = 0.0f64;
        for r in 0..3 {
            for c in 0..3 {
                worst = worst.max((inv[r][c] - ACES2065_TO_ACESCG[r][c] as f64).abs());
            }
        }
        assert!(
            worst < 5e-7,
            "ACES2065_TO_ACESCG 不是官方前向矩阵的 f64 精确逆（实测 {worst:e}，f32 双重收窄理论上界 ~3.1e-7）"
        );
    }

    /// 新旧轨一致性对拍：旧轨直接矩阵（官方 S-2014-004 值）vs 新轨
    /// `colorspace_registry` 由 AP1/AP0 原色推导的转换结果。
    ///
    /// AP1/AP0 在注册表中同为 D60 名义白点 ⟺ `convert` 不走 Bradford
    /// 适配项，复合即单一原色换基矩阵——与官方直接矩阵比对。
    /// 样本含白点 [1,1,1]、三原色与固定伪随机值（确定性，禁真 RNG，C17）。
    /// 预期 < 1e-5；**达不到即失败并如实报出实测值，不调容差**。
    #[test]
    fn aces_ap1_ap0_matches_registry_derivation() {
        use crate::colorspace_registry::{by_name, convert};

        let acescg = by_name("acescg").unwrap_or_else(|e| panic!("{e}"));
        let aces2065 = by_name("aces2065").unwrap_or_else(|e| panic!("{e}"));
        let samples: [[f32; 3]; 8] = [
            [1.0, 1.0, 1.0], // 白点
            [1.0, 0.0, 0.0], // AP1 红原色
            [0.0, 1.0, 0.0], // AP1 绿原色
            [0.0, 0.0, 1.0], // AP1 蓝原色
            [0.18, 0.42, 0.91],
            [0.63, 0.11, 0.27],
            [0.50, 0.50, 0.50],
            [1.25, 0.06, 0.44], // 超范围值（线性域允许）
        ];
        let mut worst_fwd = 0.0f64;
        let mut worst_inv = 0.0f64;
        for s in &samples {
            let sd = [s[0] as f64, s[1] as f64, s[2] as f64];
            // 正向 AP1 → AP0。
            let direct = mat3_apply(&ACESCG_TO_ACES2065, *s);
            let derived = convert(sd, acescg, aces2065).unwrap_or_else(|e| panic!("{e}"));
            for i in 0..3 {
                worst_fwd = worst_fwd.max((direct[i] as f64 - derived[i]).abs());
            }
            // 逆向 AP0 → AP1。
            let direct_inv = mat3_apply(&ACES2065_TO_ACESCG, *s);
            let derived_inv = convert(sd, aces2065, acescg).unwrap_or_else(|e| panic!("{e}"));
            for i in 0..3 {
                worst_inv = worst_inv.max((direct_inv[i] as f64 - derived_inv[i]).abs());
            }
        }
        println!(
            "新旧轨 AP1↔AP0 一致性：fwd worst = {worst_fwd:e}，inv worst = {worst_inv:e}（容差 1e-5）"
        );
        assert!(
            worst_fwd < 1e-5,
            "旧轨 ACESCG_TO_ACES2065 vs 注册表推导超差（实测 {worst_fwd:e}，预期 < 1e-5）"
        );
        assert!(
            worst_inv < 1e-5,
            "旧轨 ACES2065_TO_ACESCG vs 注册表推导超差（实测 {worst_inv:e}，预期 < 1e-5）"
        );
    }

    #[test]
    fn transform_roundtrip_via_hub() {
        // 线性 sRGB → ACEScg → 线性 sRGB 往返。
        let c = [0.2f32, 0.5, 0.8];
        let chain = TransformChain::new(ColorSpace::LinearSrgb, ColorSpace::AcesCg);
        let aces = chain.apply(c).unwrap();
        let back = TransformChain::new(ColorSpace::AcesCg, ColorSpace::LinearSrgb)
            .apply(aces)
            .unwrap();
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "往返失真 {back:?}");
        }
    }

    #[test]
    fn chain_rejects_display_domain() {
        let chain = TransformChain::new(ColorSpace::LinearSrgb, ColorSpace::PqDisplay);
        assert!(chain.apply([0.1, 0.2, 0.3]).is_none(), "显示域链矩阵应拒绝");
    }

    #[test]
    fn identity_chain_is_exact() {
        let chain = TransformChain::new(ColorSpace::AcesCg, ColorSpace::AcesCg);
        let c = [0.33f32, 0.44, 0.55];
        assert_eq!(chain.apply(c), Some(c));
    }
}

/// 3×3 矩阵乘法（行主序，确定性固定求和序）。公用以供链式矩阵装配复用。
pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}
