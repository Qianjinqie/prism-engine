//! 三合一降噪（B4）：à-trous 空域 + 确定性时域累积 + variance 自适应 spp
//! 预算 + [`DenoiseBackend`] 后端挂点。
//!
//! 语义权威：《算法白皮书》§5（降噪三件套：空间 à-trous、时域累积、
//! 自适应 spp——预算分配必须确定性，白皮书 §26 固定序归约）；《强大版
//! 路线图》§4 B4 行（"三合一降噪……DenoiseBackend trait 预留 + UV warp"）；
//! 《架构总设计》v2.4 §4.3（halo 协议）。
//!
//! # 确定性纪律（关卡 3 / 4）
//! - **零超越函数（D-29）**：边缘停止用**有理核** `w = 1/(1 + d²/σ²)`
//!   （纯 +−×÷）而非标准 exp 核——白皮书只要求"边缘停止函数用色差"，
//!   形式自定，有理核对同输入更稳定可复现；
//! - à-trous 权重核 = B3 样条 `[1,4,6,4,1]/16` 外积（÷256 归一）；
//! - 时域累积 = **递推增量均值** `acc ← acc + (cur − acc)/(n+1)`（f32
//!   固定表达式）；累积状态 = `(acc, samples)`，**seek 只能从帧 0 重放**
//!   ——f16 量化 + 固定序 ⇒ 任意 seek 逐位可复现（关卡 4）；
//! - spp 预算 = largest-remainder 分配（总数精确、tie 索引序确定）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// à-trous 5×5 权重核：B3 样条 `[1,4,6,4,1]`，外积后 ÷256 归一。
const KERNEL: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

// ---------------------------------------------------------------------------
// AtrousSpatial —— 空域（TileEffect，halo 型：可平移不变 ⇒ 关卡 1 可验）
// ---------------------------------------------------------------------------

/// à-trous 空域降噪：`iterations` 趟步进核迭代，第 i 趟步距 `2^i`。
///
/// halo = Σ_{i=0}^{k−1} 2·2^i = `2·(2^k − 1)`（每趟 5 tap 步距 `2^i` 的
/// 支撑半径为 `2·2^i`）——足额申报 ⇒ tile 化与整幅逐位一致（关卡 1）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtrousSpatial {
    /// 迭代趟数（0 = 恒等直通）。
    pub iterations: u32,
    /// 色差边缘停止带宽 σ（有理核 `1/(1+d²/σ²)`；0 按最小正值处理防 NaN）。
    pub sigma_color: f32,
}

impl Default for AtrousSpatial {
    fn default() -> Self {
        AtrousSpatial { iterations: 3, sigma_color: 0.1 }
    }
}

impl AtrousSpatial {
    /// 趟步距序列：`2^i`，i = 0..iterations（白皮书 §5）。
    fn step_of(iter: u32) -> u32 {
        1 << iter
    }
}

impl TileEffect for AtrousSpatial {
    fn halo_radius(&self) -> u32 {
        if self.iterations == 0 {
            0
        } else {
            2 * ((1u32 << self.iterations) - 1)
        }
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut buf = window.clone();
        for iter in 0..self.iterations {
            let sigma2 = (self.sigma_color * self.sigma_color).max(1e-12);
            buf = atrous_pass(&buf, Self::step_of(iter), sigma2);
        }
        buf
    }
}

/// 单趟 à-trous：5×5 步进核 + 色差有理边缘停止；RGB 色差驱动权重，
/// alpha 随同权重平均。邻域坐标按窗口边缘钳制复制（blur 同约定）。
///
/// 权重和下界 = 中心 tap `36/256`（d²=0 ⇒ 有理核 = 1）> 0 ⇒ 除法安全。
fn atrous_pass(src: &Fbuf16F, step: u32, sigma2: f32) -> Fbuf16F {
    let mut out = Fbuf16F::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let c = src.get(x, y);
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            // 行主序双循环（白皮书 §26 固定序归约）。
            for l in 0..5u32 {
                let sy: u32 = (y as i64 + (l as i64 - 2) * step as i64)
                    .clamp(0, src.height as i64 - 1) as u32;
                for k in 0..5u32 {
                    let sx: u32 = (x as i64 + (k as i64 - 2) * step as i64)
                        .clamp(0, src.width as i64 - 1) as u32;
                    let q = src.get(sx, sy);
                    // RGB 色差平方（直接平方和，无需 sqrt）。
                    let d2 = (q[0] - c[0]) * (q[0] - c[0])
                        + (q[1] - c[1]) * (q[1] - c[1])
                        + (q[2] - c[2]) * (q[2] - c[2]);
                    let w = KERNEL[k as usize] * KERNEL[l as usize] / 256.0 / (1.0 + d2 / sigma2);
                    for i in 0..4 {
                        acc[i] += w * q[i];
                    }
                    wsum += w;
                }
            }
            let inv = 1.0 / wsum;
            out.set(x, y, [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// TemporalAccumulator —— 时域（跨帧状态显式携带：seek = 重放）
// ---------------------------------------------------------------------------

/// 确定性时域累积器（递推增量均值）。
///
/// 累积状态 = `(acc 缓冲, samples)` 两件显式数据——**无隐藏状态**。
/// seek 到任意帧 n 的唯一路径是从帧 0 逐帧重放到 n：每步都是 f32 固定
/// 表达式 + f16 量化 ⇒ 重放与首次渲染逐位一致（关卡 4 判定）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TemporalAccumulator {
    /// 当前累积均值所含帧数 n（`acc = 前 n 帧的递推均值`）。
    pub samples: u32,
}

impl TemporalAccumulator {
    /// 递推一步：`acc' = acc + (cur − acc)/(n+1)`（f32 固定式，逐通道）。
    ///
    /// # Panics
    /// 两缓冲尺寸不一致时 panic（调用方编程错误，宪章 C21）。
    pub fn accumulate(&self, acc: &Fbuf16F, cur: &Fbuf16F) -> Fbuf16F {
        assert_eq!(
            (acc.width, acc.height),
            (cur.width, cur.height),
            "时域累积要求 acc 与 cur 同尺寸"
        );
        let n1 = (self.samples + 1) as f32;
        let mut out = Fbuf16F::new(acc.width, acc.height);
        for y in 0..acc.height {
            for x in 0..acc.width {
                let a = acc.get(x, y);
                let c = cur.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        a[0] + (c[0] - a[0]) / n1,
                        a[1] + (c[1] - a[1]) / n1,
                        a[2] + (c[2] - a[2]) / n1,
                        a[3] + (c[3] - a[3]) / n1,
                    ],
                );
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// variance 自适应 spp 预算（largest-remainder，确定性）
// ---------------------------------------------------------------------------

/// 将总样本预算 `total` 按权重 `weights` 分配到各区域（variance 自适应）。
///
/// largest-remainder 法：`ideal_i = total·w_i/W`；`base_i = floor(ideal_i)`；
/// 余量 `total − Σbase` 按小数部分**降序**（并列按**索引升序**）逐项 +1。
/// 总和恰为 `total`，tie-break 固定 ⇒ 同输入同分配（白皮书 §26 确定性）。
///
/// # Panics
/// 权重含 NaN/负数、或 `total > 2²⁴` 时 panic（后者防 f32 舍入破坏
/// "余量 < n" 的数学前提，宪章 C21 防呆）。
pub fn allocate_spp(weights: &[f32], total: u32) -> Vec<u32> {
    assert!(
        weights.iter().all(|w| w.is_finite() && *w >= 0.0),
        "spp 权重必须非负有限"
    );
    assert!(total <= 1 << 24, "spp 预算过大：f32 舍入会破坏精确分配");
    let mut out = vec![0u32; weights.len()];
    if total == 0 || weights.is_empty() {
        return out;
    }
    let wsum: f32 = weights.iter().sum();
    if !(wsum > 0.0) {
        return out; // 全零权重 → 全 0（确定性，不虚构预算）
    }
    let mut rem: Vec<(f32, usize)> = Vec::with_capacity(weights.len());
    let mut allocated: u64 = 0;
    for (i, &w) in weights.iter().enumerate() {
        let ideal = w / wsum * total as f32;
        let base = ideal.floor();
        out[i] = base as u32;
        allocated += out[i] as u64;
        rem.push((ideal - base, i));
    }
    // Σfrac < n（数学）；assert 上界保证 f32 舍入不越出该前提。
    let leftover = (total as u64) - allocated.min(total as u64);
    rem.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    for &(_, i) in rem.iter().take(leftover as usize) {
        out[i] += 1;
    }
    debug_assert_eq!(out.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
    out
}

// ---------------------------------------------------------------------------
// DenoiseBackend —— 后端挂点（只定义协议，不实现后端；后续波次接入）
// ---------------------------------------------------------------------------

/// 降噪后端协议（B4 预留挂点）。
///
/// 实现契约（宪章 D 系 + 白皮书 §26 + 架构 §4.3）：
/// - **纯函数**：同输入同输出，无隐藏状态、无私开 RNG（随机性唯一入口
///   prism-noise 整数 hash）；
/// - **f16 存储 / f32 运算域**：写回量化 f16，位模式精确往返；
/// - `spp` 为该区域的 variance 自适应预算（[`allocate_spp`] 产出）——后端
///   可按其调节降噪强度，但不得破坏确定性；
/// - 零第三方依赖（与整个 prism 引擎一致）。
///
/// W4 交付 CPU 参考路径（[`AtrousSpatial`] + [`TemporalAccumulator`]）；
/// GPU / AI 后端由后续波次实现本 trait 接入。
pub trait DenoiseBackend {
    /// 后端名（诊断 / 注册用）。
    fn name(&self) -> &'static str;

    /// 对输入执行降噪；`spp` 为该区域的自适应样本预算。
    fn denoise(&self, input: &Fbuf16F, spp: u32) -> Fbuf16F;
}

// ---------------------------------------------------------------------------
// 测试：关卡 3（方差下降 + 边缘保持）/ 关卡 4（收敛 + seek 重放逐位）/ 预算
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_noise::pcg;

    /// 确定性噪声场：`hash2(seed, x, y) → [0,1)` 映射到 ±amp。
    fn noise_at(seed: u64, x: u32, y: u32, amp: f32) -> f32 {
        (pcg::hash2_f32(seed, x as u64, y as u64) - 0.5) * amp
    }

    fn fill(base: f32, amp: f32, seed: u64, w: u32, h: u32, rgbs: fn(f32) -> [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let n = base + noise_at(seed, x, y, amp);
                fb.set(x, y, rgbs(n));
            }
        }
        fb
    }

    /// R 通道对 center 的偏差平方均值（方差）。
    fn variance_r(fb: &Fbuf16F, center: f32) -> f32 {
        let n = (fb.width * fb.height) as f32;
        let mut s = 0.0;
        for y in 0..fb.height {
            for x in 0..fb.width {
                let d = fb.get(x, y)[0] - center;
                s += d * d;
            }
        }
        s / n
    }

    // ---- 关卡 3a：平坦噪声区方差必须下降 ----

    #[test]
    fn checkpoint3_atrous_reduces_variance_on_flat_noise() {
        let input = fill(0.5, 0.4, 7, 16, 16, |v| [v, v, v, 1.0]);
        let effect = AtrousSpatial { iterations: 1, sigma_color: 0.5 };
        // halo 申报：1 趟 → 2·(2¹−1) = 2。
        assert_eq!(effect.halo_radius(), 2);

        let out = effect.eval_window(&input);
        let var_in = variance_r(&input, 0.5);
        let var_out = variance_r(&out, 0.5);
        println!("checkpoint3 平坦区方差：输入 {var_in:.6} → à-trous {var_out:.6}");
        assert!(var_out < var_in, "à-trous 必须降低平坦噪声方差（{var_out} ≥ {var_in}）");
        // 能量守恒（均值保持）：|mean − 0.5| 应远小于噪声幅度。
        let mut mean = 0.0;
        for y in 0..out.height {
            for x in 0..out.width {
                mean += out.get(x, y)[0];
            }
        }
        mean /= (out.width * out.height) as f32;
        assert!((mean - 0.5).abs() < 0.02, "降噪必须近似保持均值（mean = {mean}）");
    }

    // ---- 关卡 3b：强边缘必须保持（边缘停止生效）----

    #[test]
    fn checkpoint3_atrous_preserves_strong_edge() {
        // 左半 0（带微噪）| 右半 1（带微噪）的竖直陡边，x=8 为边。
        let mut input = Fbuf16F::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let base = if x >= 8 { 1.0 } else { 0.0 };
                let v = base + noise_at(11, x, y, 0.1);
                input.set(x, y, [v, v, v, 1.0]);
            }
        }
        // 小 σ：跨边色差 d² ≈ 1 ⇒ 有理核 1/(1+1/σ²) ≈ 0.0025 ≪ 同侧权重
        // ⇒ 跨边混合被边缘停止抑制。
        let effect = AtrousSpatial { iterations: 2, sigma_color: 0.05 };
        let out = effect.eval_window(&input);

        let mut left_mean = 0.0;
        let mut right_mean = 0.0;
        for y in 0..16u32 {
            for dx in 0..2u32 {
                left_mean += out.get(7 - dx, y)[0];
                right_mean += out.get(8 + dx, y)[0];
            }
        }
        left_mean /= 32.0;
        right_mean /= 32.0;
        println!("checkpoint3 边缘两侧均值：左 {left_mean:.4} / 右 {right_mean:.4}");
        assert!(left_mean < 0.06, "边缘左侧必须保持暗侧（{left_mean} ≥ 0.06）");
        assert!(right_mean > 0.94, "边缘右侧必须保持亮侧（{right_mean} ≤ 0.94）");
        // 同时：同侧内部仍被平滑（方差下降在有噪声的同侧成立）。
        let mut side = Fbuf16F::new(8, 16);
        for y in 0..16u32 {
            for x in 0..8u32 {
                side.set(x, y, input.get(x, y));
            }
        }
        let var_side_in = variance_r(&side, 0.0);
        let mut side_out = Fbuf16F::new(8, 16);
        for y in 0..16u32 {
            for x in 0..8u32 {
                side_out.set(x, y, out.get(x, y));
            }
        }
        let var_side_out = variance_r(&side_out, 0.0);
        assert!(
            var_side_out < var_side_in,
            "同侧平坦区仍应被平滑（{var_side_out} ≥ {var_side_in}）"
        );
    }

    // ---- 关卡 4a：时域累积收敛方向 ----

    #[test]
    fn checkpoint4_temporal_accumulation_converges_toward_truth() {
        let w = 8u32;
        let h = 8u32;
        let truth = 0.5;
        let acc = Fbuf16F::new(w, h); // 从 0 开始
        let mut state = TemporalAccumulator { samples: 0 };

        let mut cur = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = truth + noise_at(23, x, y, 0.6);
                cur.set(x, y, [v, v, v, 1.0]);
            }
        }
        let var_frame = variance_r(&cur, truth);

        let mut acc = acc;
        for _ in 0..64 {
            // 每帧换噪声（帧号进 hash：seed 混帧号——用 hash3 三键派生）。
            for y in 0..h {
                for x in 0..w {
                    let v = truth
                        + (pcg::hash3_f32(23, x as u64, y as u64, state.samples as u64 + 1)
                            - 0.5)
                            * 0.6;
                    cur.set(x, y, [v, v, v, 1.0]);
                }
            }
            acc = state.accumulate(&acc, &cur);
            state.samples += 1;
        }
        let var_acc = variance_r(&acc, truth);
        println!(
            "checkpoint4 时域方差：单帧 {var_frame:.6} → 累积 64 帧 {var_acc:.6}"
        );
        assert!(
            var_acc < var_frame / 4.0,
            "64 帧累积方差必须显著低于单帧（{var_acc} vs {var_frame}）"
        );
    }

    // ---- 关卡 4b：seek 重放逐位可复现 + 状态依赖 ----

    #[test]
    fn checkpoint4_temporal_seek_replay_is_bit_reproducible() {
        let w = 8u32;
        let h = 8u32;
        let frame = |f: u64| -> Fbuf16F {
            let mut fb = Fbuf16F::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let v = (pcg::hash3_f32(99, x as u64, y as u64, f) - 0.5) * 0.6 + 0.5;
                    fb.set(x, y, [v, v, v, 1.0]);
                }
            }
            fb
        };

        // 路径 A：从帧 0 重放到帧 4。
        let mut a = Fbuf16F::new(w, h);
        let mut st = TemporalAccumulator { samples: 0 };
        for f in 0..5u64 {
            a = st.accumulate(&a, &frame(f));
            st.samples += 1;
        }
        // 路径 B：同序列完全重放（seek 语义）→ 逐位一致。
        let mut b = Fbuf16F::new(w, h);
        let mut st2 = TemporalAccumulator { samples: 0 };
        for f in 0..5u64 {
            b = st2.accumulate(&b, &frame(f));
            st2.samples += 1;
        }
        assert_eq!(
            a.pixels, b.pixels,
            "seek 重放必须与首次渲染逐位一致（f16 位模式全等）"
        );
        // 状态依赖：少累积一帧（重放到帧 3）→ 结果不同（大概率，位级
        // 碰撞概率 ~0）——证明累积确在携带历史而非幂等映射。
        let mut c = Fbuf16F::new(w, h);
        let mut st3 = TemporalAccumulator { samples: 0 };
        for f in 0..4u64 {
            c = st3.accumulate(&c, &frame(f));
            st3.samples += 1;
        }
        assert_ne!(a.pixels, c.pixels, "不同累积深度必须产生不同结果");
    }

    // ---- spp 预算：largest-remainder 精确 + tie-break 确定 ----

    #[test]
    fn allocate_spp_largest_remainder_is_exact_and_deterministic() {
        // 均分 4 区 × 总 10：ideal 2.5 → base [2,2,2,2]，余 2 按 tie 索引序
        // 给前两区。
        assert_eq!(allocate_spp(&[1.0, 1.0, 1.0, 1.0], 10), vec![3, 3, 2, 2]);
        // 加权 0.7/0.2/0.1 × 10 → [7,2,1]。
        assert_eq!(allocate_spp(&[0.7, 0.2, 0.1], 10), vec![7, 2, 1]);
        // 全零权重 → 全 0；空表 → 空；total 0 → 全 0。
        assert_eq!(allocate_spp(&[0.0, 0.0], 5), vec![0, 0]);
        assert_eq!(allocate_spp(&[], 5), Vec::<u32>::new());
        assert_eq!(allocate_spp(&[3.0, 1.0], 0), vec![0, 0]);
        // 总和恒等于 total（不变量）。
        let out = allocate_spp(&[0.13, 0.27, 0.41, 0.19], 1000);
        assert_eq!(out.iter().sum::<u32>(), 1000);
        // 同输入同分配（确定性）。
        let again = allocate_spp(&[0.13, 0.27, 0.41, 0.19], 1000);
        assert_eq!(out, again);
    }
}
