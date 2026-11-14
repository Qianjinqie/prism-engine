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

/// 光谱波长维（T08 切片）：取未占用的高位维度，避开相机段（0–4）与
/// 弹射段（基 5 + 每弹 6 槽，max_bounces=8 时最高 ≈ 52）。取 60 落在
/// Sobol 表内（< 64）且不与任何既有维度冲突——单条路径每样本仅抽一次，
/// 作为 hero wavelength Monte Carlo 的分层坐标（PRD Q3①）。
pub const DIM_LAMBDA: u16 = 60;

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
///
/// 迭代只走 `index` 的**置位位**（`trailing_zeros` + `bits &= bits-1`），
/// 而不是逐位右移 32 次。W0-C L5 前这里是 `while bits != 0 { if bits & 1 … }`，
/// 平均走 31 次迭代；置位位平均 16 次。**结果逐位不变**：GF(2) 上的异或满足
/// 交换律与结合律，求和顺序不影响值（这正是"整数件可以随便换写法、浮点件
/// 不行"的分界）。
#[inline]
pub fn sobol_u32(dim: u16, index: u32) -> u32 {
    debug_assert!(
        (dim as usize) < SOBOL_DIMS as usize,
        "sobol_u32 维度越界（超表兜底应由 dim_value 处理）"
    );
    let row = &direction_table()[dim as usize];
    let mut v = 0u32;
    let mut bits = index;
    while bits != 0 {
        let k = bits.trailing_zeros() as usize;
        v ^= row[k];
        bits &= bits - 1;
    }
    v
}

/// 一对 Sobol 坐标（维度 `dim` 与 `dim + 1`，共享同一 `index`）。
///
/// [`sample_2d`] 的专用件：两个维度走**同一趟**置位位循环，省掉一次位扫描。
/// **结果与两次 [`sobol_u32`] 逐位相同**（同上：异或的顺序无关性）。
#[inline]
fn sobol_pair_u32(dim: u16, index: u32) -> (u32, u32) {
    let table = direction_table();
    let r0 = &table[dim as usize];
    let r1 = &table[dim as usize + 1];
    let mut v0 = 0u32;
    let mut v1 = 0u32;
    let mut bits = index;
    while bits != 0 {
        let k = bits.trailing_zeros() as usize;
        v0 ^= r0[k];
        v1 ^= r1[k];
        bits &= bits - 1;
    }
    (v0, v1)
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

/// 打散掩码的哈希前缀：`hash2(base, MASK_SALT, ·)` 的前半段只依赖 `base`，
/// 故**每像素只算一次**（`hash2_prefix` 是 `hash2` 的同一个表达式的前半段，
/// 拆开不改变任何一位）。
#[inline]
fn mask_prefix(base: u64) -> u64 {
    pcg::hash2_prefix(base, MASK_SALT)
}

/// 单维坐标值（含打散与超表兜底）：`Sobol(i, dim) ⊕ mask(seed, pixel, dim)`。
#[inline]
fn dim_value(seed: u64, dim: u16, index: u32) -> f32 {
    dim_value_p(mask_prefix(seed), seed, dim, index)
}

/// [`dim_value`] 的"前缀已备好"版本（`prefix = mask_prefix(base)`）。
#[inline]
fn dim_value_p(prefix: u64, seed: u64, dim: u16, index: u32) -> f32 {
    if dim >= SOBOL_DIMS {
        // 超表兜底（lib.rs 模块文档声明）：PCG 纯函数流，确定性不变；
        // 低差异 net 性质不再保证（仅超出 64 维的深弹射路径会走到）。
        return pcg::hash2_f32(seed, dim as u64, index as u64);
    }
    // ② hash(种子, 像素, 维度) → 打散掩码。
    let mask = pcg::hash2_finish(prefix, dim as u64) as u32;
    let v = sobol_u32(dim, index) ^ mask;
    to_unit(v)
}

/// 一对坐标值（维度 `dim` / `dim + 1`，共享 `index`）：与两次 [`dim_value`]
/// 逐位相同，只是共用一趟位扫描。
#[inline]
fn dim_pair(seed: u64, dim: u16, index: u32) -> (f32, f32) {
    dim_pair_p(mask_prefix(seed), dim, index)
}

/// [`dim_pair`] 的"前缀已备好"版本。调用方保证 `dim + 1 < SOBOL_DIMS`。
#[inline]
fn dim_pair_p(prefix: u64, dim: u16, index: u32) -> (f32, f32) {
    let mask0 = pcg::hash2_finish(prefix, dim as u64) as u32;
    let mask1 = pcg::hash2_finish(prefix, dim as u64 + 1) as u32;
    let (v0, v1) = sobol_pair_u32(dim, index);
    (to_unit(v0 ^ mask0), to_unit(v1 ^ mask1))
}

/// 32 位定点 Sobol 值 → `[0, 1)` 的 f32。
///
/// 取高 24 位乘 2^-24：24 位恰为 f32 尾数精度，乘 2^-24 是纯指数运算，
/// 换算逐位可复现（与 `prism-noise::pcg::unit_f32` 同一技巧）。**乘的不是
/// 除法**——除法在 2 的幂上是精确的，乘法更是，两者位相同，此处取乘法是为了
/// 让编译器合成单条 `scalef` 而不是 `divss`。
#[inline]
fn to_unit(v: u32) -> f32 {
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
    if si.dimension + 1 >= SOBOL_DIMS {
        // 超表兜底：两维各自独立走 PCG 流（与改前逐位相同）。
        return (
            dim_value(base, si.dimension, index),
            dim_value(base, si.dimension + 1, index),
        );
    }
    dim_pair(base, si.dimension, index)
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
///
/// # 为什么把派生量存成字段（W0-C L5）
///
/// `base = hash4(种子, 像素, 帧, 节点)` 与 `index = 样本 ⊕ hash1(base)` 都是
/// **纯函数**，且在整个流的生命周期内不变。改前每次 `sample_2d` 都重算这两
/// 项——一条样本平均调 5.26 次 `sample_2d`，等于把同一个 `hash4`（4 次
/// `pcg::permute`）算 5 遍。构造期算一次、字段存住，**取值逐位不变**（纯函数 +
/// 相同输入 ⇒ 相同输出），于是这条优化对确定性是恒等变换，不是"测出来一致"。
#[derive(Debug, Clone, Copy)]
pub struct SampleStream<'a> {
    ctx: &'a EvalCtx,
    pixel: (u32, u32),
    sample: u32,
    /// `base_seed(ctx, pixel)`：像素级常量。
    base: u64,
    /// `sobol_index(base, sample)`：样本级常量（全流共享同一 Sobol 索引）。
    index: u32,
    /// `hash2_prefix(base, MASK_SALT)`：打散掩码的哈希前缀（像素级常量）。
    prefix: u64,
    /// 像素级维度打散掩码表 `hash2_finish(prefix, d)`（d ∈ [0, SOBOL_DIMS)）。
    ///
    /// `Some` = [`PixelStreamBase::stream`] 通用渲染主路径：每像素只预算一次
    /// 64 个掩码，`sample_1d` / `sample_2d` 直接查表，省掉每次抽取的 1–2 次
    /// `pcg::permute`。`None` = `SampleStream::new` 的单流入口（相机 / 波前 /
    /// 深度 / 单样本 / 测试），保留 `dim_value_p` / `dim_pair_p` 现算，逐字节不变。
    /// 两条路径同式逐位一致：`masks[d]` 恒等于 `hash2_finish(prefix, d) as u32`
    /// （判据 `pixel_stream_base_bitwise_tests`）。不判定场景内容，对任意
    /// 几何 / 灯族 / 材质 / spp 组合生效。
    masks: Option<&'a [u32; SOBOL_DIMS as usize]>,
}

impl<'a> SampleStream<'a> {
    /// 以求值上下文 + 全局像素坐标 + 样本序号构造流句柄。
    #[inline]
    pub fn new(ctx: &'a EvalCtx, pixel: (u32, u32), sample: u32) -> Self {
        let base: u64 = base_seed(ctx, pixel);
        let index: u32 = sobol_index(base, sample);
        let prefix: u64 = mask_prefix(base);
        SampleStream { ctx, pixel, sample, base, index, prefix, masks: None }
    }

    /// 本流的像素坐标（相机投射取亚像素抖动基准）。
    #[inline]
    pub fn pixel(&self) -> (u32, u32) {
        self.pixel
    }

    /// 本流的样本序号；供独立通用路径消费同帧首交点缓存时校验身份。
    #[inline]
    pub fn sample_index(&self) -> u32 {
        self.sample
    }

    /// 构造指定维度的 [`SampleIndex`]（frame/node_id 取自 ctx，D-28）。
    ///
    /// 只给 `aux_1d` 这类**不共享**像素级派生量的通道用；`sample_1d` /
    /// `sample_2d` 直接吃缓存的 `base` / `index`，不走这里。
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

    /// 一维 Sobol 抽取（与自由函数 [`sample_1d`] 逐位相同）。
    ///
    /// 通用渲染主路径（`masks = Some`）把每维打散掩码 `hash2_finish(prefix, dim)`
    /// 提升为「每像素预算一次」的查表，省掉每次抽取的一次 `pcg::permute`；
    /// 表内维度逐位等价于 `dim_value_p`，超表兜底仍走同式 `hash2_f32`。
    #[inline]
    pub fn sample_1d(&self, dimension: u16) -> f32 {
        match self.masks {
            Some(m) => {
                if dimension >= SOBOL_DIMS {
                    return pcg::hash2_f32(self.base, dimension as u64, self.index as u64);
                }
                to_unit(sobol_u32(dimension, self.index) ^ m[dimension as usize])
            }
            None => dim_value_p(self.prefix, self.base, dimension, self.index),
        }
    }

    /// 二维 Sobol 抽取（维度 d 与 d+1；与自由函数 [`sample_2d`] 逐位相同）。
    ///
    /// 表内维度对走「每像素预算的 64 项掩码表」+ 一趟位扫描，与
    /// `dim_pair_p` 逐位相同（同一 `hash2_finish(prefix, ·)` 结果）；
    /// 超表兜底一律保留原 `dim_value_p` 路径，不查表。
    #[inline]
    pub fn sample_2d(&self, dimension: u16) -> (f32, f32) {
        if dimension + 1 >= SOBOL_DIMS {
            return (
                dim_value_p(self.prefix, self.base, dimension, self.index),
                dim_value_p(self.prefix, self.base, dimension + 1, self.index),
            );
        }
        match self.masks {
            Some(m) => {
                let (v0, v1) = sobol_pair_u32(dimension, self.index);
                (
                    to_unit(v0 ^ m[dimension as usize]),
                    to_unit(v1 ^ m[dimension as usize + 1]),
                )
            }
            None => dim_pair_p(self.prefix, dimension, self.index),
        }
    }

    /// 游程辅助抽取（变长序列：介质步进）。
    #[inline]
    pub fn aux_1d(&self, dimension: u16, step: u32) -> f32 {
        aux_1d(self.ctx, &self.si(dimension), step)
    }

    /// 波长维采样（T08 切片）：返回 [0,1) 均匀量，由调用方经
    /// [`crate::spectral::hero_wavelength`] 映射为 hero λ（退化模式被钳为
    /// 固定中值）。每样本仅抽一次，确定性继承自 Sobol 流。
    #[inline]
    pub fn sample_lambda(&self) -> f32 {
        self.sample_1d(DIM_LAMBDA)
    }
}

/// 像素级采样常量（V3.1 等价化简快速内核的「样本流常量提升」）。
///
/// `SampleStream::new(ctx, pixel, sample)` 的派生量里，`base = base_seed(ctx,pixel)`、
/// `index_xor = hash1(base, IDX_SALT)`、`prefix = mask_prefix(base)`，以及
/// `dim_pair_p` 的两个打散掩码 `mask0/mask1`（只依赖 `prefix` 与固定维度对）都是
/// **像素级常量**——同一像素的 16 个槽位把它们重复计算 16 次。本结构每像素
/// 构造一次，把每样本 8 次 `permute` 降为 0。
///
/// # 逐位一致性（两层恒等变换，均为 GF(2)/整数律，非近似）
/// 设 `c = index_xor`，样本 `s` 的 Sobol 索引 `i = s ^ c`。原
/// `dim_pair_p(prefix, dim, i)` 返回
/// `to_unit(sobol(dim, i) ^ mask0)`、`to_unit(sobol(dim+1, i) ^ mask1)`。
/// ① 打散掩码与 `base`/`prefix` 只依赖像素，提到构造期算一次。
/// ② Sobol 方向数按 `bits(i)` 异或 ⇒ `sobol(dim, a^b) = sobol(dim,a) ^
///    sobol(dim,b)`（GF(2) 线性）。故 `sobol(dim, i) = sobol(dim, c) ^
///    sobol(dim, s)`，其中 `sobol(dim, c)` 是像素级常量并入 `c0/c1`，而
///    `sobol(dim, s)` 对 `s < 16` 至多 4 个置位（原 `i` 是随机 32 位，平均
///    16 个置位）⇒ 逐样本位扫描从 ~16 次降到 ≤4 次。
/// 于是 `sample_2d(s) = (to_unit(c0 ^ sobol(dim,s)), to_unit(c1 ^
///    sobol(dim+1,s)))`，与原式逐位相同（`pixel_sample_base_is_bitwise_identical`
/// 覆盖表内维度 / 相机像素维 / 超表边界逐位对拍）。
#[derive(Debug, Clone, Copy)]
pub struct PixelSampleBase {
    /// `sobol(dim, c) ^ mask0`（像素级常量）。
    c0: u32,
    /// `sobol(dim+1, c) ^ mask1`（像素级常量）。
    c1: u32,
    dim: u16,
    /// 超表兜底用的像素级派生量（`dim + 1 >= SOBOL_DIMS` 时才用）。
    base: u64,
    prefix: u64,
    index_xor: u32,
}

impl PixelSampleBase {
    /// 以 `(ctx, pixel, dim)` 预算像素级常量；`dim` 固定为该像素的采样维度对。
    #[inline]
    pub fn new(ctx: &EvalCtx, pixel: (u32, u32), dim: u16) -> Self {
        let base: u64 = base_seed(ctx, pixel);
        let index_xor: u32 = pcg::hash1(base, IDX_SALT) as u32;
        let prefix: u64 = mask_prefix(base);
        let (c0, c1): (u32, u32) = if dim + 1 < SOBOL_DIMS {
            let (sc0, sc1): (u32, u32) = sobol_pair_u32(dim, index_xor);
            let mask0: u32 = pcg::hash2_finish(prefix, dim as u64) as u32;
            let mask1: u32 = pcg::hash2_finish(prefix, dim as u64 + 1) as u32;
            (sc0 ^ mask0, sc1 ^ mask1)
        } else {
            (0, 0)
        };
        PixelSampleBase { c0, c1, dim, base, prefix, index_xor }
    }

    /// 样本 `sample` 在该维度对的二维抖动，与 `SampleStream::sample_2d` 逐位一致。
    #[inline]
    pub fn sample_2d(&self, sample: u32) -> (f32, f32) {
        if self.dim + 1 >= SOBOL_DIMS {
            // 超表兜底：与 SampleStream::sample_2d 的两次 dim_value_p 同式。
            let index: u32 = sample ^ self.index_xor;
            return (
                dim_value_p(self.prefix, self.base, self.dim, index),
                dim_value_p(self.prefix, self.base, self.dim + 1, index),
            );
        }
        let (s0, s1): (u32, u32) = sobol_pair_u32(self.dim, sample);
        (to_unit(self.c0 ^ s0), to_unit(self.c1 ^ s1))
    }

    /// 用调用方预算的 `(sobol(dim,s), sobol(dim+1,s))` 定点坐标取抖动。
    /// 该表只依赖 `dim/sample`，与像素无关，可整帧共享一次预算，逐样本连
    /// `sobol_pair_u32` 位扫描都省掉。仅 `dim + 1 < SOBOL_DIMS` 有效。
    #[inline]
    pub fn jitter_for_sobol(&self, s0: u32, s1: u32) -> (f32, f32) {
        debug_assert!(self.dim + 1 < SOBOL_DIMS);
        (to_unit(self.c0 ^ s0), to_unit(self.c1 ^ s1))
    }
}

/// 像素级样本流常量（V3.1 通用路径「跨样本常量提升」）。
///
/// `SampleStream::new(ctx, pixel, sample)` 的三个派生量里，`base_seed(ctx,pixel)`、
/// `mask_prefix(base)` 与 `sobol_index` 的 `hash1(base, IDX_SALT)` 都只依赖
/// `(ctx, pixel)`，**与样本序号无关**。`render_pixel_range` 对同一像素的 N 个
/// 样本把它们重算 N 次（每样本 3 次哈希、共 7 次 `pcg::permute`）。本结构每
/// 像素构造一次，`stream` 逐样本只做 `index = sample ^ index_xor`，把这些哈希
/// 从 O(N) 降到 O(1)。
///
/// 在此基础上再预算**全 64 维打散掩码表** `masks[d] = hash2_finish(prefix, d)`
/// （同样只依赖像素）。原 `sample_1d` / `sample_2d` 每次抽取都现算 1–2 次
/// `pcg::permute`；一条路径 16 spp 会调用数十次，改查表后把这些 permute 从
/// O(N·维度) 降到「每像素 64 次」。表内维度逐位等价于现算（同一个纯函数、相同
/// 输入），超表兜底仍保留原式 ⇒ 任意几何 / 灯族 / 材质 / 贴图 / 介质 / spp
/// 组合输出逐字节不变。
///
/// # 逐位一致性（纯函数 + GF(2) 整数律，非近似）
/// `sobol_index(base, sample) = sample ^ (hash1(base, IDX_SALT) as u32)`
/// （见 [`sobol_index`]），故 `index = sample ^ self.index_xor` 与
/// `SampleStream::new` 的 `index` 逐位相等；`base` / `prefix` 是同一纯函数的
/// 相同输入下的输出。`masks[d]` 与 `dim_value_p` / `dim_pair_p` 里的
/// `hash2_finish(prefix, ·)` 逐位相等。判据见
/// `pixel_stream_base_bitwise_tests::stream_matches_new_bitwise`（对拍
/// `masks = Some` 与 `masks = None` 两条路径）。
///
/// 这条改造落在 `prism-rt` 通用渲染入口，**不判定场景内容**：任意场景都吃到
/// 同一份收益；`一致` 档哈希锚因此不变。
#[derive(Debug, Clone, Copy)]
pub struct PixelStreamBase {
    base: u64,
    prefix: u64,
    index_xor: u32,
    /// 每像素预算的 64 维打散掩码（表内维度；超表走 `hash2_f32` 兜底）。
    masks: [u32; SOBOL_DIMS as usize],
}

impl PixelStreamBase {
    /// 以 `(ctx, pixel)` 预算像素级常量（每像素一次）。
    #[inline]
    pub fn new(ctx: &EvalCtx, pixel: (u32, u32)) -> Self {
        let base: u64 = base_seed(ctx, pixel);
        let prefix: u64 = mask_prefix(base);
        let mut masks: [u32; SOBOL_DIMS as usize] = [0u32; SOBOL_DIMS as usize];
        for d in 0..SOBOL_DIMS as usize {
            masks[d] = pcg::hash2_finish(prefix, d as u64) as u32;
        }
        PixelStreamBase {
            prefix,
            index_xor: pcg::hash1(base, IDX_SALT) as u32,
            base,
            masks,
        }
    }

    /// 样本 `sample` 的流句柄，与 `SampleStream::new(ctx, pixel, sample)`
    /// **逐位一致**（标量字段逐个相同；掩码表是同一纯函数的预算值，非近似）。
    #[inline]
    pub fn stream<'a>(
        &'a self,
        ctx: &'a EvalCtx,
        pixel: (u32, u32),
        sample: u32,
    ) -> SampleStream<'a> {
        SampleStream {
            ctx,
            pixel,
            sample,
            base: self.base,
            index: sample ^ self.index_xor,
            prefix: self.prefix,
            masks: Some(&self.masks),
        }
    }
}

#[cfg(test)]
mod pixel_stream_base_bitwise_tests {
    use super::*;

    #[test]
    fn stream_matches_new_bitwise() {
        let ctx = EvalCtx::new(0x9E37_79B9, 3, 7);
        for &(px, py) in &[(0u32, 0u32), (1, 2), (17, 991), (4095, 2160)] {
            let pb = PixelStreamBase::new(&ctx, (px, py));
            for sample in 0u32..64 {
                let a = SampleStream::new(&ctx, (px, py), sample);
                let b = pb.stream(&ctx, (px, py), sample);
                assert_eq!(a.base, b.base, "base @{px},{py} s{sample}");
                assert_eq!(a.index, b.index, "index @{px},{py} s{sample}");
                assert_eq!(a.prefix, b.prefix, "prefix @{px},{py} s{sample}");
                assert_eq!(a.sample, b.sample);
                assert_eq!(a.pixel, b.pixel);
                for dim in [0u16, 2, 4, 7, 100, 1000] {
                    assert_eq!(
                        a.sample_1d(dim).to_bits(),
                        b.sample_1d(dim).to_bits(),
                        "1d dim{dim} @{px},{py} s{sample}"
                    );
                    let (u0, v0) = a.sample_2d(dim);
                    let (u1, v1) = b.sample_2d(dim);
                    assert_eq!(u0.to_bits(), u1.to_bits(), "2d.0 dim{dim} s{sample}");
                    assert_eq!(v0.to_bits(), v1.to_bits(), "2d.1 dim{dim} s{sample}");
                    assert_eq!(
                        a.aux_1d(dim, sample).to_bits(),
                        b.aux_1d(dim, sample).to_bits(),
                        "aux dim{dim} s{sample}"
                    );
                }
            }
        }
    }

    /// 全维度 + 超表边界逐位对拍：`masks = Some`（每像素预算查表）与
    /// `masks = None`（现算 `dim_value_p` / `dim_pair_p`）在整个 64 维表域、
    /// 超表兜底、以及 `sample_2d` 的 `dim + 1 == SOBOL_DIMS` 边界上逐位相同。
    /// 维度选择只由弹射深度决定、与几何 / 灯族 / 材质 / 贴图无关，钉死全部
    /// 维度即钉死「任意场景」这条通用路径的确定性。
    #[test]
    fn masks_table_matches_recompute_all_dims_bitwise() {
        let ctx = EvalCtx::new(0xD1B5_4A32_D192_ED03, 11, 5);
        let pixels: [(u32, u32); 4] = [(0, 0), (3, 71), (640, 360), (3839, 2159)];
        let mut dims_1d: Vec<u16> = (0..=SOBOL_DIMS).collect();
        dims_1d.extend_from_slice(&[65, 100, 1000, 32_767, 65_535]);
        let mut dims_2d: Vec<u16> = (0..SOBOL_DIMS).collect();
        dims_2d.extend_from_slice(&[63, 64, 100, 1000]);
        for &(px, py) in &pixels {
            let pb = PixelStreamBase::new(&ctx, (px, py));
            for sample in [0u32, 1, 7, 15, 16, 63, 1024, 0xFFFF_FFFF] {
                let a = SampleStream::new(&ctx, (px, py), sample);
                let b = pb.stream(&ctx, (px, py), sample);
                assert_eq!(a.base, b.base, "base px({px},{py}) s{sample}");
                assert_eq!(a.index, b.index, "index px({px},{py}) s{sample}");
                assert_eq!(a.prefix, b.prefix, "prefix px({px},{py}) s{sample}");
                for &dim in &dims_1d {
                    assert_eq!(
                        a.sample_1d(dim).to_bits(),
                        b.sample_1d(dim).to_bits(),
                        "1d dim{dim} px({px},{py}) s{sample}"
                    );
                }
                for &dim in &dims_2d {
                    let (u0, v0) = a.sample_2d(dim);
                    let (u1, v1) = b.sample_2d(dim);
                    assert_eq!(
                        u0.to_bits(), u1.to_bits(),
                        "2d.0 dim{dim} px({px},{py}) s{sample}"
                    );
                    assert_eq!(
                        v0.to_bits(), v1.to_bits(),
                        "2d.1 dim{dim} px({px},{py}) s{sample}"
                    );
                }
                assert_eq!(
                    a.aux_1d(7, sample).to_bits(),
                    b.aux_1d(7, sample).to_bits(),
                    "aux px({px},{py}) s{sample}"
                );
            }
        }
    }
}

#[cfg(test)]
mod pixel_sample_base_bitwise_tests {
    use super::*;

    fn test_ctx() -> EvalCtx {
        EvalCtx::new(0x9E37_79B9_7F4A_7C15, 7, 3)
    }

    /// Sobol 索引的 GF(2) 线性：`Sobol(a^b) = Sobol(a) ^ Sobol(b)`。
    /// 这是 `PixelSampleBase` 把 `Sobol(s^idx)` 拆成 `Sobol(s)^Sobol(idx)`
    /// 的代数依据，逐位锁定。
    #[test]
    fn sobol_index_is_gf2_linear() {
        for dim in [0u16, 3, 10, 30, 62, 63] {
            for a in 0..64u32 {
                for b in [0u32, 1, 7, 0x9E37_79B9, 0xFFFF_FFFF] {
                    assert_eq!(
                        sobol_u32(dim, a ^ b),
                        sobol_u32(dim, a) ^ sobol_u32(dim, b),
                        "GF(2) 线性在 dim{dim} a{a} b{b} 不成立"
                    );
                }
            }
        }
    }

    /// `PixelSampleBase::sample_2d(s)` 与 `SampleStream::new(...).sample_2d(dim)`
    /// 逐位一致：覆盖表内维度、相机像素维 `DIM_PIXEL`、超表边界 `dim = 63`。
    /// 同时验证 `jitter_for_sobol`（整帧预算表路径）与 `sample_2d` 逐位一致。
    #[test]
    fn pixel_sample_base_is_bitwise_identical() {
        let c = test_ctx();
        for dim in [0u16, 3, 10, 30, 62, 63] {
            for &(px, py) in &[(0u32, 0u32), (17, 23), (123, 456), (4095, 2047)] {
                let base = PixelSampleBase::new(&c, (px, py), dim);
                for s in 0..16u32 {
                    let stream = SampleStream::new(&c, (px, py), s);
                    let (a, b) = stream.sample_2d(dim);
                    let (x, y) = base.sample_2d(s);
                    assert_eq!(a.to_bits(), x.to_bits(), "dim{dim} px({px},{py}) s{s} u");
                    assert_eq!(b.to_bits(), y.to_bits(), "dim{dim} px({px},{py}) s{s} v");
                    if dim + 1 < SOBOL_DIMS {
                        let (jx, jy) = base.jitter_for_sobol(
                            sobol_u32(dim, s),
                            sobol_u32(dim + 1, s),
                        );
                        assert_eq!(x.to_bits(), jx.to_bits(), "jitter u dim{dim} s{s}");
                        assert_eq!(y.to_bits(), jy.to_bits(), "jitter v dim{dim} s{s}");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 确定性三角件（D-29 简化件：区间逼近多项式 + 固定系数，禁 libm）
// ---------------------------------------------------------------------------
// 范围归约：x mod 2π（floor 为精确舍入运算，非超越函数）→ 象限对称 →
// [0, π/4] Maclaurin 级数（Bernoulli 系数为公开数学常量）。
// π/4 处截断误差：sin ~2.5e-8、cos ~2.5e-8，覆盖引擎全部使用场景。

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// ---------------------------------------------------------------------------
// V3.1 P1-1：构建期 const 查表（等价变换，**禁数值近似**）
//
// 与 `prism_math::det_trig` 同款纪律：系数是编译期常量表达式，与优化前的
// 字面量逐位相同；Horner 结合序逐句不变 ⇒ 输出与冻结参考实现逐位一致
// （判据：`v31_det_trig_bitwise_tests`）。这不是插值表，不改任何一项
// 浮点运算的操作数或顺序，故纯 CPU 单帧哈希锚不变。
// ---------------------------------------------------------------------------
const SIN_C: [f32; 5] = [
    1.0,
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5040.0,
    1.0 / 362_880.0,
];
const COS_C: [f32; 5] = [
    1.0,
    -0.5,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40_320.0,
];

/// `[0, π/4]` 上的 sin（构建期 const 系数 + Horner，逐位不变）。
///
/// 热路径改走 [`quarter_sin_cos`]；本函数保留为冻结对拍对象。
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn quarter_sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (SIN_C[0] + x2 * (SIN_C[1] + x2 * (SIN_C[2] + x2 * (SIN_C[3] + x2 * SIN_C[4]))))
}

/// `[0, π/4]` 上的 cos（构建期 const 系数 + Horner，逐位不变）。
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn quarter_cos(x: f32) -> f32 {
    let x2 = x * x;
    COS_C[0] + x2 * (COS_C[1] + x2 * (COS_C[2] + x2 * (COS_C[3] + x2 * COS_C[4])))
}

/// `[0, π/4]` 上一次性算 `(cos x, sin x)`：只做一次 `x*x`。
///
/// V3.1 P1-1 等价变换：旧路径两个函数各算一次 `x2 = x*x`（同一纯运算、
/// 逐位相同），这里复用同一个 `x2`；两条 Horner 的操作数与结合序逐句不变
/// ⇒ 两个分量都逐位一致，每样本省一次 f32 乘法。
#[inline]
fn quarter_sin_cos(x: f32) -> (f32, f32) {
    let x2 = x * x;
    let s = x * (SIN_C[0] + x2 * (SIN_C[1] + x2 * (SIN_C[2] + x2 * (SIN_C[3] + x2 * SIN_C[4]))));
    let c = COS_C[0] + x2 * (COS_C[1] + x2 * (COS_C[2] + x2 * (COS_C[3] + x2 * COS_C[4])));
    (c, s)
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
        // quarter_sin_cos(rr) = (cos rr, sin rr)；半区互换回 (sin rr, cos rr)。
        let (cr, sr) = quarter_sin_cos(rr);
        (sr, cr)
    } else {
        quarter_sin_cos(r)
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
mod v31_det_trig_bitwise_tests {
    use super::*;

    /// 优化前实现的冻结逐字拷贝（字面量系数 + 分别调用两个 quarter 函数）。
    fn quarter_sin_ref(x: f32) -> f32 {
        let x2 = x * x;
        x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
    }

    fn quarter_cos_ref(x: f32) -> f32 {
        let x2 = x * x;
        1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
    }

    fn cos_sin_ref(x: f32) -> (f32, f32) {
        const TWO_PI: f32 = 2.0 * PI;
        let mut r = x - TWO_PI * (x / TWO_PI).floor();
        let q = (r * (2.0 / PI)) as i32 & 3;
        r -= q as f32 * FRAC_PI_2;
        let (c, s) = if r > FRAC_PI_4 {
            let rr = FRAC_PI_2 - r;
            (quarter_sin_ref(rr), quarter_cos_ref(rr))
        } else {
            (quarter_cos_ref(r), quarter_sin_ref(r))
        };
        match q {
            0 => (c, s),
            1 => (-s, c),
            2 => (-c, -s),
            _ => (s, -c),
        }
    }

    /// const 表 + 复用 `x2` 后，与冻结参考在整圆周 4096 点上逐位一致。
    #[test]
    fn cos_sin_is_bitwise_identical_to_frozen_reference() {
        for k in 0..4096 {
            let x: f32 = -4.0 * PI + 8.0 * PI * k as f32 / 4096.0;
            let (c, s) = cos_sin(x);
            let (cr, sr) = cos_sin_ref(x);
            assert_eq!(c.to_bits(), cr.to_bits(), "cos({x})");
            assert_eq!(s.to_bits(), sr.to_bits(), "sin({x})");
        }
        for k in 0..=512 {
            let t: f32 = FRAC_PI_4 * k as f32 / 512.0;
            assert_eq!(quarter_sin(t).to_bits(), quarter_sin_ref(t).to_bits());
            assert_eq!(quarter_cos(t).to_bits(), quarter_cos_ref(t).to_bits());
        }
    }

    /// 单一真源纪律：`prism-rt::sampler::cos_sin` 与
    /// `prism_math::det_trig::cos_sin` 两份实现必须逐位相同，防漂移。
    #[test]
    fn sampler_and_math_det_trig_are_bitwise_identical() {
        for k in 0..4096 {
            let x: f32 = -4.0 * PI + 8.0 * PI * k as f32 / 4096.0;
            let (c, s) = cos_sin(x);
            let (mc, ms) = prism_math::det_trig::cos_sin(x);
            assert_eq!(c.to_bits(), mc.to_bits(), "cos({x}) 跨 crate 漂移");
            assert_eq!(s.to_bits(), ms.to_bits(), "sin({x}) 跨 crate 漂移");
        }
    }
}

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
