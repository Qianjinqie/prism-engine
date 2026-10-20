//! Sobol 低差异序列 + 打散采样器（的 B1 实装——"采样器是灵魂"）。
//!
//! # 语义（逐条对应）
//! 哈希只负责**选点与打散**两件事：
//! ① `hash(种子, 像素)` 定数字位移 c，Sobol 索引 = `样本序号 ⊕ c`（样本域经
//!    ⊕ 进入派生式）——Sobol 映射线性：`Sobol(i⊕c) = Sobol(i) ⊕ Sobol(c)`，
//!    样本序号集 {0..N} 平移后 (t,m,s)-net 性质逐点保持（N = 2ᵐ 时严格成立）。
//!    **禁止把 hash 值直接当索引**：那是对索引空间的任意置换 = 从 2³² 网格
//!    iid 抽点，O(1/N) 退化为 O(1/√N)；
//! ② `hash(种子, 像素, 维度)` 定打散掩码（xor 滚动打散——与随机常数异或
//!    保持 net 性质）；
//! 坐标 = `Sobol(i, 维度) ⊕ 打散掩码`。**禁止按维度独立取均匀随机数**
//! （否则向量均匀性整体消失，收敛从 O(1/N) 退化为 IID 的 O(1/√N)）。
//! 维度区分路径内用途，弹射深度编码进维度序号（预算表见 integrator）。
//!
//! # 确定性
//! 每个像素的每个样本是独立纯函数：无全局 RNG 状态，任意像素任意 pass 可
//! 独立重算（超越点）。整数 hash 走 [`prism_noise::pcg`]（格 Zᵏ）；
//! 浮点只做 Sobol 逆映射与格点间换算（格锚表）。
//!
//! # 方向数表（数据资产，D-12 审查锚点；数据资产清单 #1）
//! Sobol 方向数采用 **Joe–Kuo 官方表**（BSD 三条款许可，**非公有领域**）：
//! S. Joe & F. Y. Kuo, "Constructing Sobol sequences with better
//! two-dimensional projections" (SIAM J. Sci. Comput. 30(5), 2008)，
//! 表文件 `new-joe-kuo-6.21201`，下载锚点
//! <https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201>
//! （当前取维度 2–64，维内最高次数 s ≤ 9）。维度 1 不在表中（平凡：
//! 所有 m_i = 1，即 van der Corput 序列，按定义构造）。
//! 超出 64 维的请求走 PCG 纯函数流兜底（lib.rs 模块文档"超表兜底"）。
//!
//! BSD 许可义务（原文锚点 <https://web.maths.unsw.edu.au/~fkuo/sobol/licence>，
//! 本注释即履行二进制再分发之声明义务）：
//! Copyright (c) 2008, Frances Y. Kuo and Stephen Joe. All rights reserved.
//! ① 再分发（源码/二进制）须保留上述版权声明与许可条件；② 数据按 "AS IS"
//! 提供，无任何明示/默示担保（Merchantability/Fitness 均不承诺）；
//! ③ 不得以版权人及 UNSW/Waikato 名义推广衍生产品；
//! ④ 学术致谢（非强制）：Joe & Kuo, ACM TOMS 29:49-57 (2003)。
//!
//! # 模块内容
//! - Sobol 核心：方向数构造（Bratley–Fox 递推）+ [`sobol_u32`]。
//! - 纯函数采样入口：[`sample_1d`] / [`sample_2d`] / [`aux_1d`]（面向
//!  `EvalCtx`/`SampleIndex`， 唯一随机入口）。
//! - [`SampleStream`]：单像素单样本的维度化采样句柄（integrator 侧便利层）。
//! - 分布映射工具族：disk / 余弦半球 / 锥方向 / 正交基（相机、介质、
//!   积分器共享，零重复实现；全部只用 ±×÷√ 与确定性三角件）。
//! - 确定性三角件 [`cos_sin`] / [`tan`]：区间逼近多项式 + 固定系数
//!   （D-29 纪律：禁 libm；sqrt 为 IEEE 精确舍入运算不在禁列）。

use std::sync::OnceLock;

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;
use prism_noise::pcg;

// ---------------------------------------------------------------------------
// Joe–Kuo 方向数表（维度 2..=64；来源锚点见模块文档）
// ---------------------------------------------------------------------------
// 每行：(s, a, m_1..m_9)。s = 原始多项式次数；a = 中间系数编码
// （a 的第 j−1 位 = a_j，多项式 x^s + a_{s−1}x^{s−1} + … + a_1 x + 1）；
// m_i 为初始方向数（m_i < 2^i，m_1 恒为 1）。m 数组按 s 截断使用。

/// 表覆盖的维度总数（维度 0 = van der Corput + 表内 2..=64）。
pub const SOBOL_DIMS: u16 = 64;

/// 维度 1..=63 的方向数初值（即表中维度 2..=64；下标 d−1 对应维度 d）。
#[rustfmt::skip]
const JOE_KUO: [(u8, u8, [u32; 9]); 63] = [
    (1, 0,  [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2, 1,  [1, 3, 0, 0, 0, 0, 0, 0, 0]),
    (3, 1,  [1, 3, 1, 0, 0, 0, 0, 0, 0]),
    (3, 2,  [1, 1, 1, 0, 0, 0, 0, 0, 0]),
    (4, 1,  [1, 1, 3, 3, 0, 0, 0, 0, 0]),
    (4, 4,  [1, 3, 5, 13, 0, 0, 0, 0, 0]),
    (5, 2,  [1, 1, 5, 5, 17, 0, 0, 0, 0]),
    (5, 4,  [1, 1, 5, 5, 5, 0, 0, 0, 0]),
    (5, 7,  [1, 1, 7, 11, 19, 0, 0, 0, 0]),
    (5, 11, [1, 1, 5, 1, 1, 0, 0, 0, 0]),
    (5, 13, [1, 1, 1, 3, 11, 0, 0, 0, 0]),
    (5, 14, [1, 3, 5, 5, 31, 0, 0, 0, 0]),
    (6, 1,  [1, 3, 3, 9, 7, 49, 0, 0, 0]),
    (6, 13, [1, 1, 1, 15, 21, 21, 0, 0, 0]),
    (6, 16, [1, 3, 1, 13, 27, 49, 0, 0, 0]),
    (6, 19, [1, 1, 1, 15, 7, 5, 0, 0, 0]),
    (6, 22, [1, 3, 1, 15, 13, 25, 0, 0, 0]),
    (6, 25, [1, 1, 5, 5, 19, 61, 0, 0, 0]),
    (7, 1,  [1, 3, 7, 11, 23, 15, 103, 0, 0]),
    (7, 4,  [1, 3, 7, 13, 13, 15, 69, 0, 0]),
    (7, 7,  [1, 1, 3, 13, 7, 35, 63, 0, 0]),
    (7, 8,  [1, 3, 5, 9, 1, 25, 53, 0, 0]),
    (7, 14, [1, 3, 1, 13, 9, 35, 107, 0, 0]),
    (7, 19, [1, 3, 1, 5, 27, 61, 31, 0, 0]),
    (7, 21, [1, 1, 5, 11, 19, 41, 61, 0, 0]),
    (7, 28, [1, 3, 5, 3, 3, 13, 69, 0, 0]),
    (7, 31, [1, 1, 7, 13, 1, 19, 1, 0, 0]),
    (7, 32, [1, 3, 7, 5, 13, 19, 59, 0, 0]),
    (7, 37, [1, 1, 3, 9, 25, 29, 41, 0, 0]),
    (7, 41, [1, 3, 5, 13, 23, 1, 55, 0, 0]),
    (7, 42, [1, 3, 7, 3, 13, 59, 17, 0, 0]),
    (7, 50, [1, 3, 1, 3, 5, 53, 69, 0, 0]),
    (7, 55, [1, 1, 5, 5, 23, 33, 13, 0, 0]),
    (7, 56, [1, 1, 7, 7, 1, 61, 123, 0, 0]),
    (7, 59, [1, 1, 7, 9, 13, 61, 49, 0, 0]),
    (7, 62, [1, 3, 3, 5, 3, 55, 33, 0, 0]),
    (8, 14, [1, 3, 1, 15, 31, 13, 49, 245, 0]),
    (8, 21, [1, 3, 5, 15, 31, 59, 63, 97, 0]),
    (8, 22, [1, 3, 1, 11, 11, 11, 77, 249, 0]),
    (8, 38, [1, 3, 1, 11, 27, 43, 71, 9, 0]),
    (8, 47, [1, 1, 7, 15, 21, 11, 81, 45, 0]),
    (8, 49, [1, 3, 7, 3, 25, 31, 65, 79, 0]),
    (8, 50, [1, 3, 1, 1, 19, 11, 3, 205, 0]),
    (8, 52, [1, 1, 5, 9, 19, 21, 29, 157, 0]),
    (8, 56, [1, 3, 7, 11, 1, 33, 89, 185, 0]),
    (8, 67, [1, 3, 3, 3, 15, 9, 79, 71, 0]),
    (8, 70, [1, 3, 7, 11, 15, 39, 119, 27, 0]),
    (8, 84, [1, 1, 3, 1, 11, 31, 97, 225, 0]),
    (8, 97, [1, 1, 1, 3, 23, 43, 57, 177, 0]),
    (8, 103, [1, 3, 7, 7, 17, 17, 37, 71, 0]),
    (8, 115, [1, 3, 1, 5, 27, 63, 123, 213, 0]),
    (8, 122, [1, 1, 3, 5, 11, 43, 53, 133, 0]),
    (9, 8,  [1, 3, 5, 5, 29, 17, 47, 173, 479]),
    (9, 13, [1, 3, 3, 11, 3, 1, 109, 9, 69]),
    (9, 16, [1, 1, 1, 5, 17, 39, 23, 5, 343]),
    (9, 22, [1, 3, 1, 5, 25, 15, 31, 103, 499]),
    (9, 25, [1, 1, 1, 11, 11, 17, 63, 105, 183]),
    (9, 44, [1, 1, 5, 11, 9, 29, 97, 231, 363]),
    (9, 47, [1, 1, 5, 15, 19, 45, 41, 7, 383]),
    (9, 52, [1, 3, 7, 7, 31, 19, 83, 137, 221]),
    (9, 55, [1, 1, 1, 3, 23, 15, 111, 223, 83]),
    (9, 59, [1, 1, 5, 13, 31, 15, 55, 25, 161]),
    (9, 62, [1, 1, 3, 13, 25, 47, 39, 87, 257]),
];

/// 构造 64×32 方向数矩阵（懒初始化一次）。
///
/// 维度 0：V_i = 2^(32−i)（van der Corput，按定义）。
/// 维度 d ≥ 1（表中第 d−1 行）：V_i = m_i·2^(32−i)（i ≤ s）；i > s 时按
/// Bratley–Fox 递推 V_i = V_{i−s} ⊕ (V_{i−s} >> s) ⊕ Σ_{a_j=1} V_{i−j}。
fn direction_table() -> &'static [[u32; 32]; SOBOL_DIMS as usize] {
    static TABLE: OnceLock<[[u32; 32]; SOBOL_DIMS as usize]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0u32; 32]; SOBOL_DIMS as usize];
        for k in 0..32usize {
            t[0][k] = 1u32 << (31 - k);
        }
        for d in 1..SOBOL_DIMS as usize {
            let (s, a, m) = JOE_KUO[d - 1];
            let s = s as usize;
            for i in 1..=s {
                t[d][i - 1] = m[i - 1] << (32 - i);
            }
            for i in (s + 1)..=32usize {
                let prev = t[d][i - 1 - s];
                let mut v = prev ^ (prev >> s);
                // a 的第 j−1 位 = a_j：a_j = 1 时异或 V_{i−j}（Burkardt i4_sobol 同式）。
                let mut bits = a as u32;
                for j in 1..s {
                    if bits & 1 == 1 {
                        v ^= t[d][i - 1 - j];
                    }
                    bits >>= 1;
                }
                t[d][i - 1] = v;
            }
        }
        t
    })
}

/// Sobol 坐标（32 位定点）：`Sobol(index, dim) = ⊕_{k: index 第 k 位=1} V_{k+1}`。
/// 纯整数异或——格 Zᵏ 运算，确定性是定理而非承诺。
#[inline]
pub fn sobol_u32(dim: u16, index: u32) -> u32 {
    debug_assert!(
        (dim as usize) < SOBOL_DIMS as usize,
        "sobol_u32 维度越界（超表兜底应由 dim_value 处理）"
    );
    let row = &direction_table()[dim as usize];
    let mut v = 0u32;
    let mut bits = index;
    let mut k = 0usize;
    while bits != 0 {
        if bits & 1 == 1 {
            v ^= row[k];
        }
        bits >>= 1;
        k += 1;
    }
    v
}

// ---------------------------------------------------------------------------
// 样本流派生（hash(全局种子, 节点id, 帧, pixel, 样本[, 维度])）
// ---------------------------------------------------------------------------

/// Sobol 索引盐（派生 ① 的 hash 参数，与掩码盐分离防混叠）。
const IDX_SALT: u64 = 0x243F_6A88_85A3_08D3;
/// 打散掩码盐（派生 ② 的 hash 参数）。
const MASK_SALT: u64 = 0x1319_8A2E_0370_7344;
/// 游程辅助流盐（aux_1d 的 hash 参数，与上两者分离防混叠）。
const AUX_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// 像素基种子：hash(全局种子, 像素, 帧, 节点)——**不含样本序号**（
///  ①②：选点位移与打散掩码都以"像素整条序列"为单位，同一像素的样本
/// 序列共享同一 c 与掩码，net 性质才逐点保持）。
#[inline]
fn base_seed(ctx: &EvalCtx, pixel: (u32, u32)) -> u64 {
    let pack = ((pixel.0 as u64) << 32) | pixel.1 as u64;
    pcg::hash4(ctx.global_seed, pack, ctx.frame, ctx.node_id, 0)
}

/// 游程辅助流种子：hash(全局种子, 像素, 样本, 帧, 节点)（含样本序号——
/// 变长游程序列按样本独立，不参与 net 向量结构）。
#[inline]
fn aux_seed(ctx: &EvalCtx, si: &SampleIndex) -> u64 {
    let pack = ((si.pixel.0 as u64) << 32) | si.pixel.1 as u64;
    pcg::hash4(ctx.global_seed, pack, si.sample as u64, si.frame, si.node_id)
}

/// ① Sobol 索引 = 样本序号 ⊕ hash(种子, 像素)（随机数字平移，见模块文档）。
#[inline]
fn sobol_index(base: u64, sample: u32) -> u32 {
    sample ^ (pcg::hash1(base, IDX_SALT) as u32)
}

/// 单维坐标值（含打散与超表兜底）：`Sobol(i, dim) ⊕ mask(seed, pixel, dim)`。
#[inline]
fn dim_value(seed: u64, dim: u16, index: u32) -> f32 {
    if dim >= SOBOL_DIMS {
        // 超表兜底（lib.rs 模块文档声明）：PCG 纯函数流，确定性不变；
        // 低差异 net 性质不再保证（仅超出 64 维的深弹射路径会走到）。
        return pcg::hash2_f32(seed, dim as u64, index as u64);
    }
    // ② hash(种子, 像素, 维度) → 打散掩码。
    let mask = pcg::hash2(seed, MASK_SALT, dim as u64) as u32;
    let v = sobol_u32(dim, index) ^ mask;
    // 高 24 位 → [0,1)：24 位恰为 f32 尾数精度，乘 2^-24 是纯指数运算，
    // 换算逐位可复现（与 prism-noise::pcg::unit_f32 同一技巧）。
    ((v >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// 一维 Sobol 采样（纯函数）：坐标 = Sobol(i, dim) ⊕ 打散掩码，f32 ∈ [0,1)。
pub fn sample_1d(ctx: &EvalCtx, si: &SampleIndex) -> f32 {
    let base = base_seed(ctx, si.pixel);
    dim_value(base, si.dimension, sobol_index(base, si.sample))
}

/// 二维 Sobol 采样（纯函数）：维度 (d, d+1) 共享同一索引 i（
/// 向量坐标成对出自同一 net 点——**禁止按维度独立取随机数**）。
pub fn sample_2d(ctx: &EvalCtx, si: &SampleIndex) -> (f32, f32) {
    let base = base_seed(ctx, si.pixel);
    let index = sobol_index(base, si.sample);
    (
        dim_value(base, si.dimension, index),
        dim_value(base, si.dimension + 1, index),
    )
}

/// 游程辅助流（介质 delta tracking 步进等变长随机序列）：PCG 纯函数流，
/// 以 (维度槽位, 步序号) 区分。此类抽取不参与 (t,m,s)-net 向量结构
/// （自由程不是低差异量），hash 派生与"选点与打散"分工一致。
#[inline]
pub fn aux_1d(ctx: &EvalCtx, si: &SampleIndex, step: u32) -> f32 {
    pcg::hash3_f32(aux_seed(ctx, si), si.dimension as u64, step as u64, AUX_SALT)
}

// ---------------------------------------------------------------------------
// SampleStream：单像素单样本的维度化采样句柄（integrator/camera 便利层）
// ---------------------------------------------------------------------------

/// 单像素单样本采样流：`(frame, pixel, sample, node_id, global_seed)` 的
/// 纯函数包。同流同输出；不同像素/样本/维度互不串流（D-28 像素域）。
#[derive(Debug, Clone, Copy)]
pub struct SampleStream<'a> {
    ctx: &'a EvalCtx,
    pixel: (u32, u32),
    sample: u32,
}

impl<'a> SampleStream<'a> {
    /// 以求值上下文 + 全局像素坐标 + 样本序号构造流句柄。
    pub fn new(ctx: &'a EvalCtx, pixel: (u32, u32), sample: u32) -> Self {
        SampleStream { ctx, pixel, sample }
    }

    /// 本流的像素坐标（相机投射取亚像素抖动基准）。
    #[inline]
    pub fn pixel(&self) -> (u32, u32) {
        self.pixel
    }

    /// 构造指定维度的 [`SampleIndex`]（frame/node_id 取自 ctx，D-28）。
    #[inline]
    fn si(&self, dimension: u16) -> SampleIndex {
        SampleIndex {
            frame: self.ctx.frame,
            pixel: self.pixel,
            sample: self.sample,
            dimension,
            node_id: self.ctx.node_id,
        }
    }

    /// 一维 Sobol 抽取。
    #[inline]
    pub fn sample_1d(&self, dimension: u16) -> f32 {
        sample_1d(self.ctx, &self.si(dimension))
    }

    /// 二维 Sobol 抽取（维度 d 与 d+1）。
    #[inline]
    pub fn sample_2d(&self, dimension: u16) -> (f32, f32) {
        sample_2d(self.ctx, &self.si(dimension))
    }

    /// 游程辅助抽取（变长序列：介质步进）。
    #[inline]
    pub fn aux_1d(&self, dimension: u16, step: u32) -> f32 {
        aux_1d(self.ctx, &self.si(dimension), step)
    }
}

// ---------------------------------------------------------------------------
// 确定性三角件（D-29 简化件：区间逼近多项式 + 固定系数，禁 libm）
// ---------------------------------------------------------------------------
// 范围归约：x mod 2π（floor 为精确舍入运算，非超越函数）→ 象限对称 →
// [0, π/4] Maclaurin 级数（Bernoulli 系数为公开数学常量）。
// π/4 处截断误差：sin ~2.5e-8、cos ~2.5e-8，覆盖引擎全部使用场景。

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// `[0, π/4]` 上的 sin（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
}

/// `[0, π/4]` 上的 cos（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_cos(x: f32) -> f32 {
    let x2 = x * x;
    1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
}

/// 任意角 `(cos x, sin x)`：mod 2π 归约（精确 floor）→ 象限对称 →
/// `[0, π/4]` 级数。跨平台逐位一致（只含 ±×÷ 与精确 floor/round）。
pub fn cos_sin(x: f32) -> (f32, f32) {
    const TWO_PI: f32 = 2.0 * PI;
    // 归约到 [0, 2π)；引擎输入量级小（相机角/相位角），floor 精度充分。
    let mut r = x - TWO_PI * (x / TWO_PI).floor();
    // 象限号：q = floor(r / (π/2)) ∈ {0,1,2,3}。
    let q = (r * (2.0 / PI)) as i32 & 3;
    r -= q as f32 * FRAC_PI_2; // r ∈ [0, π/2]
    // 半区归约到 [0, π/4]。
    let (c, s) = if r > FRAC_PI_4 {
        let rr = FRAC_PI_2 - r;
        (quarter_sin(rr), quarter_cos(rr))
    } else {
        (quarter_cos(r), quarter_sin(r))
    };
    match q {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// 确定性 tan（sin/cos 同一归约路径；当前唯一调用点 = 相机焦距标度
/// tan(fov/2)，fov ∈ (0°,180°) 开区间——宪章 C17/C18）。按任务说明
/// "自研 tan 从简"落实现并在此声明：精度 ~1e-6 量级，足够相机几何。
pub fn tan(x: f32) -> f32 {
    let (c, s) = cos_sin(x);
    s / c
}

// ---------------------------------------------------------------------------
// 分布映射工具族（相机 / 介质 / 积分器共享，零重复实现）
// ---------------------------------------------------------------------------

/// 方形 → 圆盘同心映射（Shirley–Chiu，面积保持、双射）。
/// 输出均匀分布于单位圆盘：E[r²] = 1/2。
pub fn concentric_disk(u: f32, v: f32) -> (f32, f32) {
    let ox = 2.0 * u - 1.0;
    let oy = 2.0 * v - 1.0;
    if ox == 0.0 && oy == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if ox.abs() > oy.abs() {
        (ox, FRAC_PI_4 * (oy / ox))
    } else {
        (oy, FRAC_PI_2 - FRAC_PI_4 * (ox / oy))
    };
    let (c, s) = cos_sin(theta);
    (r * c, r * s)
}

/// 余弦加权半球采样（局部坐标系，+z 为法线）：pdf = cosθ/π。
/// Malley 法：圆盘同心映射 + z = √(1−x²−y²)，零三角。
pub fn cosine_hemisphere(u: f32, v: f32) -> Vec3 {
    let (x, y) = concentric_disk(u, v);
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    Vec3::new(x, y, z)
}

/// 由单位向量 n 构造正交基 (t, b)（叉积法，格锚纪律：禁浮点三角）。
pub fn onb(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = up.cross(&n).normalize();
    let b = n.cross(&t);
    (t, b)
}

/// 锥内均匀方向采样：绕单位轴 w、半角余弦 `cos_theta_max` 的球冠。
/// pdf = 1/(2π(1−cos_theta_max))。u 定极角，v 定方位角。
pub fn cone_dir(w: Vec3, cos_theta_max: f32, u: f32, v: f32) -> Vec3 {
    let cos_t = 1.0 - u * (1.0 - cos_theta_max); // ∈ [cos_max, 1]
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let (t, b) = onb(w);
    let (c, s) = cos_sin(2.0 * PI * v);
    w * cos_t + t * (sin_t * c) + b * (sin_t * s)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(seed: u64) -> EvalCtx {
        EvalCtx::new(seed, 7, 3)
    }

    fn si(pixel: (u32, u32), sample: u32, dimension: u16) -> SampleIndex {
        SampleIndex { frame: 3, pixel, sample, dimension, node_id: 7 }
    }

    // ---- Sobol 核心锚点：维度 0 = van der Corput；前 2D 点为经典模式 ----

    #[test]
    fn sobol_dim0_is_van_der_corput() {
        // 按定义构造（非记忆数据）：dim0 第 i 点 = i 的逆位二进制小数。
        assert_eq!(sobol_u32(0, 1), 1 << 31);
        assert_eq!(sobol_u32(0, 2), 1 << 30);
        assert_eq!(sobol_u32(0, 3), (1 << 31) | (1 << 30));
        let f = |v: u32| ((v >> 8) as f32) * (1.0 / 16_777_216.0);
        assert_eq!(f(sobol_u32(0, 1)), 0.5);
        assert_eq!(f(sobol_u32(0, 2)), 0.25);
        assert_eq!(f(sobol_u32(0, 3)), 0.75);
    }

    #[test]
    fn sobol_first_2d_points_match_classic_pattern() {
        // 经典 Sobol 2D 开头：(0,0), (0.5,0.5), (0.25,0.75), (0.75,0.25)。
        // 该模式可由方向数递推纯数学推出（V_1 = 2^31, V_{i} = V_{i−1} ⊕ (V_{i−1}>>1)）。
        let f = |d: u16, i: u32| ((sobol_u32(d, i) >> 8) as f32) * (1.0 / 16_777_216.0);
        assert_eq!((f(0, 1), f(1, 1)), (0.5, 0.5));
        assert_eq!((f(0, 2), f(1, 2)), (0.25, 0.75));
        assert_eq!((f(0, 3), f(1, 3)), (0.75, 0.25));
    }

    // ---- 低差异 smoke：O(1/N) 单调改善（Owen 打散后的引擎路径）----

    /// 测试函数 f(x,y) = x·y·(1−x)·(1−y)，精确积分 = 1/36。
    fn test_fn(x: f32, y: f32) -> f32 {
        x * y * (1.0 - x) * (1.0 - y)
    }

    #[test]
    fn scrambled_sobol_integration_error_decreases_monotonically() {
        let c = ctx(42);
        let exact = 1.0f32 / 36.0;
        let mut errors = Vec::new();
        for &n in &[256u32, 1024, 4096] {
            let mut sum = 0.0f64;
            for k in 0..n {
                let s = si((0, 0), k, 0);
                let (u, v) = sample_2d(&c, &s);
                // 以 base.dimension 为 2D 对（dim 0,1）；估算用 f64 累加避免
                // 测试累加自身误差淹没收敛信号。
                sum += test_fn(u, v) as f64;
            }
            let mean = (sum / n as f64) as f32;
            let err = (mean - exact).abs();
            errors.push(err);
            println!("N={n}: estimate={mean:.6} exact={exact:.6} err={err:.3e}");
        }
        // O(1/N) 单调改善：误差随 N 下降（4096 点应显著优于 256 点）。
        assert!(errors[2] < errors[0], "N=4096 误差必须小于 N=256：{:?}", errors);
        assert!(errors[1] < errors[0], "N=1024 误差必须小于 N=256：{:?}", errors);
        assert!(errors[2] < 2.0e-4, "N=4096 绝对误差应进入 1e-4 量级：{:?}", errors);
    }

    #[test]
    fn sampler_is_pure_and_pixel_decorrelated() {
        let c = ctx(42);
        // 纯函数：同参数两遍逐位一致。
        let s = si((17, 23), 5, 4);
        let a = sample_2d(&c, &s);
        let b = sample_2d(&c, &s);
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        // 不同像素不同流（D-28 防网格伪影）。
        let mut same_pixel = 0;
        for px in 1..64u32 {
            let v = sample_2d(&c, &si((px, px), 5, 4));
            if v.0.to_bits() == a.0.to_bits() {
                same_pixel += 1;
            }
        }
        assert!(same_pixel <= 1, "不同像素的流必须去相关");
        // 范围 [0,1)。
        for k in 0..512u32 {
            let (u, v) = sample_2d(&c, &si((3, 9), k, 11));
            assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
        }
    }

    // ---- 确定性三角件锚点（固定系数多项式 vs 公开数学常量）----

    #[test]
    fn deterministic_trig_matches_known_constants() {
        // 数学常量锚点（非表数据）：sin/cos/tan 在若干标准角。
        let cases: [(f32, (f64, f64)); 5] = [
            (1.0, (0.540302306, 0.841470985)),        // cos(1), sin(1)
            (2.0, (-0.416146837, 0.909297427)),       // cos(2), sin(2)
            (std::f32::consts::FRAC_PI_6, (0.866_025_4, 0.5)), // π/6
            (std::f32::consts::FRAC_PI_3, (0.5, 0.866_025_4)), // π/3
            (5.0, (0.283662185, -0.958924275)),       // cos(5), sin(5)
        ];
        for (x, (ec, es)) in cases {
            let (c, s) = cos_sin(x);
            assert!((c as f64 - ec).abs() < 2.0e-5, "cos({x}) = {c} vs {ec}");
            assert!((s as f64 - es).abs() < 2.0e-5, "sin({x}) = {s} vs {es}");
        }
        // tan 锚点。
        assert!((tan(1.0) - 1.55740772).abs() < 2.0e-5);
        assert!((tan(std::f32::consts::FRAC_PI_4) - 1.0).abs() < 1.0e-5);
        assert!((tan(std::f32::consts::FRAC_PI_3) - 1.7320508).abs() < 2.0e-5);
        // 确定性：两遍逐位一致。
        assert_eq!(tan(1.3).to_bits(), tan(1.3).to_bits());
        assert_eq!(cos_sin(0.7).0.to_bits(), cos_sin(0.7).0.to_bits());
    }

    // ---- 分布映射性质 ----

    #[test]
    fn concentric_disk_is_uniform_in_unit_disk() {
        let c = ctx(7);
        let n = 20_000u32;
        let (mut sum_r2, mut outside) = (0.0f64, 0u32);
        for k in 0..n {
            let (u, v) = sample_2d(&c, &si((1, 2), k, 20));
            let (x, y) = concentric_disk(u, v);
            let r2 = (x * x + y * y) as f64;
            if r2 > 1.0 + 1e-6 {
                outside += 1;
            }
            sum_r2 += r2;
        }
        assert_eq!(outside, 0, "圆盘采样不得越出单位圆");
        let mean_r2 = sum_r2 / n as f64;
        assert!(
            (mean_r2 - 0.5).abs() < 0.02,
            "均匀圆盘应有 E[r²] = 0.5，实测 {mean_r2}"
        );
    }

    #[test]
    fn cosine_hemisphere_statistics() {
        let c = ctx(9);
        let n = 20_000u32;
        let (mut sum_z, mut bad) = (0.0f64, 0u32);
        for k in 0..n {
            let (u, v) = sample_2d(&c, &si((2, 3), k, 20));
            let d = cosine_hemisphere(u, v);
            if (d.length() - 1.0).abs() > 1e-4 || d.z < 0.0 {
                bad += 1;
            }
            sum_z += d.z as f64;
        }
        assert_eq!(bad, 0, "半球方向应单位化且 z ≥ 0");
        // 余弦加权：E[cosθ] = 2/3。
        let mean_z = sum_z / n as f64;
        assert!((mean_z - 2.0 / 3.0).abs() < 0.02, "E[cosθ] 应为 2/3，实测 {mean_z}");
    }

    #[test]
    fn cone_dir_statistics() {
        let c = ctx(11);
        let w = Vec3::new(0.3, -0.5, 0.8).normalize();
        let cos_max: f32 = 0.75;
        let n = 20_000u32;
        let (mut sum_dot, mut bad) = (0.0f64, 0u32);
        for k in 0..n {
            let (u, v) = sample_2d(&c, &si((4, 5), k, 20));
            let d = cone_dir(w, cos_max, u, v);
            let dot = d.dot(&w) as f64;
            if (d.length() - 1.0).abs() > 1e-4 || dot < cos_max as f64 - 1e-4 {
                bad += 1;
            }
            sum_dot += dot;
        }
        assert_eq!(bad, 0, "锥内方向应单位化且 dot ≥ cos_max");
        // 均匀锥：E[cosθ] = (1 + cos_max)/2。
        let mean_dot = sum_dot / n as f64;
        let expected = (1.0 + cos_max as f64) / 2.0;
        assert!((mean_dot - expected).abs() < 0.02, "E[cosθ] 应为 {expected}，实测 {mean_dot}");
    }

    #[test]
    fn onb_is_orthonormal_for_typical_normals() {
        for n in [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.577, 0.577, 0.577).normalize(),
            Vec3::new(0.1, -0.9, 0.4).normalize(),
        ] {
            let (t, b) = onb(n);
            assert!((t.length() - 1.0).abs() < 1e-5);
            assert!((b.length() - 1.0).abs() < 1e-5);
            assert!(t.dot(&n).abs() < 1e-5, "t ⟂ n");
            assert!(b.dot(&n).abs() < 1e-5, "b ⟂ n");
            assert!(t.dot(&b).abs() < 1e-5, "t ⟂ b");
        }
    }
}
