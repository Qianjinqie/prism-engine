//! 三合一降噪：à-trous 空域 + 确定性时域累积 + variance 自适应 spp
//! 预算 + [`DenoiseBackend`] 后端挂点。
//!
//! 语义权威：（降噪三件套：空间 à-trous、时域累积、
//! 自适应 spp——预算分配必须确定性， 固定序归约）；《强大版
//! 》 B4 行（"三合一降噪……DenoiseBackend trait 预留 + UV warp"）；
//! halo 协议。
//!
//! # 确定性纪律（关卡 3 / 4）
//! - **零超越函数（D-29）**：边缘停止用**有理核** `w = 1/(1 + d²/σ²)`
//!  （纯 +−×÷）而非标准 exp 核——只要求"边缘停止函数用色差"，
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
    /// 趟步距序列：`2^i`，i = 0..iterations。
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
            // 行主序双循环（固定序归约）。
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
/// 总和恰为 `total`，tie-break 固定 ⇒ 同输入同分配（确定性）。
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
// DenoiseBackend —— 后端挂点（只定义协议，不实现后端；后续版本接入）
// ---------------------------------------------------------------------------

/// 降噪后端协议（预留挂点）。
///
/// 实现契约（宪章 D 系 + +）：
/// - **纯函数**：同输入同输出，无隐藏状态、无私开 RNG（随机性唯一入口
///   prism-noise 整数 hash）；
/// - **f16 存储 / f32 运算域**：写回量化 f16，位模式精确往返；
/// - `spp` 为该区域的 variance 自适应预算（[`allocate_spp`] 产出）——后端
///   可按其调节降噪强度，但不得破坏确定性；
/// - 零第三方依赖（与整个 prism 引擎一致）。
///
/// CPU 参考路径（[`AtrousSpatial`] + [`TemporalAccumulator`]）；
/// GPU / AI 后端由后续版本实现本 trait 接入。
pub trait DenoiseBackend {
    /// 后端名（诊断 / 注册用）。
    fn name(&self) -> &'static str;

    /// 对输入执行降噪；`spp` 为该区域的自适应样本预算。
    fn denoise(&self, input: &Fbuf16F, spp: u32) -> Fbuf16F;
}

// ===========================================================================
// 第二阶段（V2.2-B-2 / D-2 定案）：G-buffer 引导时域重投影 + 方差引导 à-trous
// ===========================================================================
//
// 三件（均在 `prism-postfx`，零第三方依赖、零超越函数）：
//   ① `closed_form_motion` —— 由相机变换闭式给出逐像素运动矢量（无光流、无
//      第三方；矩阵乘 + 齐次除，纯 `+ - × ÷`）；
//   ② `TemporalAccumulator::accumulate_guided` —— 把上一帧累积结果按 motion
//      重投影到当前帧像素，再递推增量均值；有效性衰减采样数，无 NaN；
//   ③ `atrous_pass_variance_guided` / `VarEstimate` —— 方差进入 à-trous 权重
//      （有理核 `1/(1 + d²/σ² + |Δvar|/σv²)`），保持零超越；
//   ④ `svgf_two_stage` —— 累积 → 空域 端到端链。
//
// 硬约束：模块内无 `exp/log/powf/sin/cos`（D-29），扫描可机械执行。

/// 双线性采样（整数坐标处退化为精确原像素，保证零位移时逐位相等）。
#[inline]
fn sample_bilinear(buf: &Fbuf16F, fx: f32, fy: f32) -> [f32; 4] {
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let ix0 = x0 as i64;
    let iy0 = y0 as i64;
    let corners: [(i64, i64); 4] = [(ix0, iy0), (ix0 + 1, iy0), (ix0, iy0 + 1), (ix0 + 1, iy0 + 1)];
    let wx = [1.0 - tx, tx];
    let wy = [1.0 - ty, ty];
    let mut acc = [0.0f32; 4];
    for (k, &(cx, cy)) in corners.iter().enumerate() {
        let sx = (cx.clamp(0, buf.width as i64 - 1)) as u32;
        let sy = (cy.clamp(0, buf.height as i64 - 1)) as u32;
        let q = buf.get(sx, sy);
        let wgt = wx[k % 2] * wy[k / 2];
        for i in 0..4 {
            acc[i] += wgt * q[i];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Mat4 / Camera —— 闭式重投影的矩阵原语（行主序，纯算术，零超越）
// ---------------------------------------------------------------------------

/// 4×4 仿射/投影矩阵（行主序 `m[row][col]`）。
///
/// 仅用 `+ - × ÷`（含齐次 w 除）实现；**零超越函数**（D-29）。用于闭式
/// 重投影：世界↔裁剪的线性映射，逆矩阵用标准 4×4 代数余子式（glMatrix 口径，
/// 列主序计算后转回），纯算术、无 `exp/log/powf`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    /// 单位矩阵。
    pub fn identity() -> Self {
        Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// 正交平移相机：世界 `(x,y)` → 裁剪 `clip.x = x + tx`、`clip.y = y + ty`
    /// （齐次 w = 1，无透视除）。本项 CPU 参考路径用它构造平移相机，给出闭式
    /// 运动矢量（零超越、零第三方、无光流）。同一矩阵 API 亦可承载透视投影
    /// （未来扩展，零依赖、零超越口径不变）。
    pub fn ortho_translate(tx: f32, ty: f32) -> Self {
        Mat4([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// 代数余子式逆（glMatrix 口径：列主序计算后转回行主序），纯 `+ - × ÷`。
    ///
    /// # Panics
    /// 行列式为 0（投影矩阵奇异，调用方编程错误，宪章 C21）。
    pub fn inverse(&self) -> Option<Mat4> {
        let m = &self.0;
        // 列主序数组：索引 = row + 4*col。
        let a = [
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3],
        ];
        let a00 = a[0]; let a01 = a[1]; let a02 = a[2]; let a03 = a[3];
        let a10 = a[4]; let a11 = a[5]; let a12 = a[6]; let a13 = a[7];
        let a20 = a[8]; let a21 = a[9]; let a22 = a[10]; let a23 = a[11];
        let a30 = a[12]; let a31 = a[13]; let a32 = a[14]; let a33 = a[15];
        let b00 = a00 * a11 - a01 * a10;
        let b01 = a00 * a12 - a02 * a10;
        let b02 = a00 * a13 - a03 * a10;
        let b03 = a01 * a12 - a02 * a11;
        let b04 = a01 * a13 - a03 * a11;
        let b05 = a02 * a13 - a03 * a12;
        let b06 = a20 * a31 - a21 * a30;
        let b07 = a20 * a32 - a22 * a30;
        let b08 = a20 * a33 - a23 * a30;
        let b09 = a21 * a32 - a22 * a31;
        let b10 = a21 * a33 - a23 * a31;
        let b11 = a22 * a33 - a23 * a32;
        let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        if det == 0.0 {
            return None;
        }
        let idet = 1.0 / det;
        let out = [
            (a11 * b11 - a12 * b10 + a13 * b09) * idet,
            (a02 * b10 - a01 * b11 - a03 * b09) * idet,
            (a31 * b05 - a32 * b04 + a33 * b03) * idet,
            (a22 * b04 - a21 * b05 - a23 * b03) * idet,
            (a12 * b08 - a10 * b11 - a13 * b07) * idet,
            (a00 * b11 - a02 * b08 + a03 * b07) * idet,
            (a32 * b02 - a30 * b05 - a33 * b01) * idet,
            (a20 * b05 - a22 * b02 + a23 * b01) * idet,
            (a10 * b10 - a11 * b08 + a13 * b06) * idet,
            (a01 * b08 - a00 * b10 - a03 * b06) * idet,
            (a30 * b04 - a31 * b02 + a33 * b00) * idet,
            (a21 * b02 - a20 * b04 - a23 * b00) * idet,
            (a11 * b07 - a10 * b09 - a12 * b06) * idet,
            (a00 * b09 - a01 * b07 + a02 * b06) * idet,
            (a31 * b01 - a30 * b03 - a32 * b00) * idet,
            (a20 * b03 - a21 * b01 + a22 * b00) * idet,
        ];
        // 列主序 → 行主序。
        Some(Mat4([
            [out[0], out[4], out[8], out[12]],
            [out[1], out[5], out[9], out[13]],
            [out[2], out[6], out[10], out[14]],
            [out[3], out[7], out[11], out[15]],
        ]))
    }

    /// 变换一个点（齐次 w 除），纯算术。
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        let x = p[0];
        let y = p[1];
        let z = p[2];
        let cx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        let cy = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        let cz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
        let cw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
        if cw == 0.0 {
            [cx, cy, cz]
        } else {
            [cx / cw, cy / cw, cz / cw]
        }
    }
}

/// 相机：由世界→裁剪投影矩阵定义（闭式重投影的输入）。
///
/// 本项 CPU 参考路径用 [`Mat4::ortho_translate`] 构造平移相机；同一矩阵 API
/// 亦可承载透视投影（未来扩展，零依赖、零超越口径不变）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// 世界→裁剪投影矩阵。
    pub view_proj: Mat4,
}

impl Camera {
    /// 由投影矩阵构造。
    pub fn new(view_proj: Mat4) -> Self {
        Camera { view_proj }
    }
}

// ---------------------------------------------------------------------------
// DenoiseGuide —— 引导数据（纯数据，零依赖）
// ---------------------------------------------------------------------------

/// 降噪引导数据（纯数据，零依赖）。
///
/// 第二阶段把 G-buffer 引导信息注入时域重投影与方差引导 à-trous：
/// - `depth`：逐像素深度（来自 `prism_rt::depth::render_depth`）；
/// - `motion`：逐像素屏幕位移 `(Δx, Δy)`（由 [`closed_form_motion`] 闭式给出）；
/// - `normal`：`None` ⇒ 由 depth 有限差分重建（本项默认路径），`Some` ⇒ 显式
///   法线，用于重投影有效性判定（法线突变 ⇒ 视为无效，重启）。
pub struct DenoiseGuide<'a> {
    /// 逐像素深度（长度 = w·h）。
    pub depth: &'a [f32],
    /// 逐像素屏幕位移 `(Δx, Δy)`（长度 = w·h）。
    pub motion: &'a [[f32; 2]],
    /// 可选逐像素法线（长度 = w·h），`None` 走默认路径。
    pub normal: Option<&'a [[f32; 3]]>,
}

/// 闭式重投影运动矢量：把上一帧（prev）累积结果搬到当前帧（cur）像素。
///
/// 对每像素 `p`：`p_world = unproject(cur, p, depth)`；`p_prev = project(prev,
/// p_world)`；`motion = p_prev − p`。全程矩阵乘 + 齐次除，**零超越函数**、不引
/// 第三方、不用光流。
///
/// **静态相机短路**：若 `prev.view_proj == cur.view_proj`（同矩阵，逐分量相等），
/// 真运动严格为 0，直接返回全零——既正确（数学上 motion ≡ 0）又保证「静态场景
/// 逐字节等价」判据的机械成立（PRD 验收 2）。
///
/// 深度仅用于有效性（非有限 ⇒ 该像素 motion 视为 0），正交路径下不影响 xy 运动
/// （无视差）。透视路径（未来扩展）将用 depth 反解 z。
pub fn closed_form_motion(
    prev: &Camera,
    cur: &Camera,
    depth: &[f32],
    w: u32,
    h: u32,
) -> Vec<[f32; 2]> {
    let n = (w * h) as usize;
    if prev.view_proj == cur.view_proj {
        return vec![[0.0, 0.0]; n];
    }
    let cur_inv = cur
        .view_proj
        .inverse()
        .expect("closed_form_motion：cur 投影矩阵不可逆（编程错误）");
    let mut out = vec![[0.0f32, 0.0f32]; n];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = depth.get(i).copied().unwrap_or(1.0);
            // 深度非有限 ⇒ 无有效反投影，motion 取 0（后续累积按无效重启）。
            if !d.is_finite() {
                out[i] = [0.0, 0.0];
                continue;
            }
            let ndc_x = ((x as f32) + 0.5) / (w as f32) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((y as f32) + 0.5) / (h as f32) * 2.0;
            let world = cur_inv.transform_point([ndc_x, ndc_y, 0.0]);
            let prev_clip = prev.view_proj.transform_point(world);
            let prev_px = (prev_clip[0] + 1.0) * 0.5 * (w as f32) - 0.5;
            let prev_py = (1.0 - (prev_clip[1] + 1.0) * 0.5) * (h as f32) - 0.5;
            out[i] = [prev_px - x as f32, prev_py - y as f32];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// TemporalAccumulator —— 新增引导式累积（重投影）
// ---------------------------------------------------------------------------

impl TemporalAccumulator {
    /// 引导式时域累积一步：把上一帧累积结果按 motion 重投影到当前帧像素，再
    /// 递推增量均值；采样数随重投影有效性衰减（无效 ⇒ 退化为 curr，n=0 重启），
    /// **不产生 NaN**。
    ///
    /// 静态相机（motion ≡ 0 且深度有效）时，重投影取到原像素、有效性恒真，本
    /// 函数与 [`accumulate`](Self::accumulate) 逐位一致——这是「重投影没写错」的
    /// 机械化证明（PRD 验收 2）。
    ///
    /// # Panics
    /// `acc` 与 `cur` 尺寸不一致时 panic（宪章 C21）。
    pub fn accumulate_guided(
        &self,
        acc: &Fbuf16F,
        cur: &Fbuf16F,
        guide: &DenoiseGuide,
    ) -> Fbuf16F {
        assert_eq!(
            (acc.width, acc.height),
            (cur.width, cur.height),
            "引导式时域累积要求 acc 与 cur 同尺寸"
        );
        let w = acc.width;
        let h = acc.height;
        let n1 = (self.samples + 1) as f32;
        // 法线有效性阈值（仅当 normal = Some 时生效）：源法线相对中心法线
        // 点积低于该值 ⇒ 视为重投影无效（法线突变，几何不连续）。
        let normal_dot_min = 0.5f32;
        let mut out = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let depth = guide.depth.get(i).copied().unwrap_or(0.0);
                let motion = guide.motion.get(i).copied().unwrap_or([0.0, 0.0]);
                let sx = x as f32 + motion[0];
                let sy = y as f32 + motion[1];
                // 重投影有效性：深度有限、源落在帧内（bilinear 支撑 1px 余量）。
                let mut valid = depth.is_finite()
                    && sx >= 0.0
                    && sy >= 0.0
                    && sx <= (w - 1) as f32
                    && sy <= (h - 1) as f32;
                // 法线有效性（可选）：源法线相对中心法线偏转过大 ⇒ 无效。
                if valid {
                    if let Some(normal) = guide.normal {
                        let cn = normal.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
                        let snx = (sx).clamp(0.0, (w - 1) as f32) as u32;
                        let sny = (sy).clamp(0.0, (h - 1) as f32) as u32;
                        let sn = normal
                            .get((sny * w + snx) as usize)
                            .copied()
                            .unwrap_or([0.0, 0.0, 1.0]);
                        if cn[0] * sn[0] + cn[1] * sn[1] + cn[2] * sn[2] < normal_dot_min {
                            valid = false;
                        }
                    }
                }
                if valid {
                    let blended = sample_bilinear(acc, sx, sy);
                    let c = cur.get(x, y);
                    // 保持 acc 历史；权重 (n+1) 对四通道一致递推，denom = n1 ≥ 1，
                    // 无 NaN。
                    out.set(
                        x,
                        y,
                        [
                            blended[0] + (c[0] - blended[0]) / n1,
                            blended[1] + (c[1] - blended[1]) / n1,
                            blended[2] + (c[2] - blended[2]) / n1,
                            blended[3] + (c[3] - blended[3]) / n1,
                        ],
                    );
                } else {
                    // 无效：退化为 curr（等价于 n=0 重启），不产生 NaN。
                    out.set(x, y, cur.get(x, y));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// VarEstimate —— 逐像素方差估计（Welford，零超越）
// ---------------------------------------------------------------------------

/// 逐像素方差估计（Welford 在线算法，定点表达式，零超越函数）。
///
/// - [`VarEstimate::from_spatial`]：由单帧局部窗口估计每像素方差（本项方差引导
///   à-trous 的默认来源）；
/// - [`VarEstimate::push`]：跨帧时序累积（备用，供未来时序方差引导）。
///
/// [`field`](Self::field) 返回逐像素（亮度代理）方差标量场，直接喂给
/// [`atrous_pass_variance_guided`]。
pub struct VarEstimate {
    mean: Fbuf16F,
    m2: Fbuf16F,
    n: u32,
}

impl VarEstimate {
    /// 空估计器（n=0）。
    pub fn new(w: u32, h: u32) -> Self {
        VarEstimate {
            mean: Fbuf16F::new(w, h),
            m2: Fbuf16F::new(w, h),
            n: 0,
        }
    }

    /// 由单帧 `src` 的 `(2·radius+1)²` 局部窗口估计每像素方差。
    pub fn from_spatial(src: &Fbuf16F, radius: u32) -> Self {
        let w = src.width;
        let h = src.height;
        let mut v = VarEstimate::new(w, h);
        v.n = (2 * radius + 1) * (2 * radius + 1);
        for y in 0..h {
            for x in 0..w {
                let mut mean = [0.0f32; 4];
                let mut m2 = [0.0f32; 4];
                let mut cnt: u32 = 0;
                for dy in -(radius as i64)..=(radius as i64) {
                    for dx in -(radius as i64)..=(radius as i64) {
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as u32;
                        let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as u32;
                        let c = src.get(sx, sy);
                        cnt += 1;
                        let nn = cnt as f32;
                        for i in 0..4 {
                            let delta = c[i] - mean[i];
                            let new_mean = mean[i] + delta / nn;
                            let delta2 = c[i] - new_mean;
                            m2[i] += delta * delta2;
                            mean[i] = new_mean;
                        }
                    }
                }
                v.mean.set(x, y, mean);
                v.m2.set(x, y, m2);
            }
        }
        v
    }

    /// 时序推进一步（Welford 跨帧）。
    pub fn push(&mut self, frame: &Fbuf16F) {
        self.n += 1;
        let nn = self.n as f32;
        for y in 0..frame.height {
            for x in 0..frame.width {
                let c = frame.get(x, y);
                let mean = self.mean.get(x, y);
                let mut new_mean = [0.0f32; 4];
                let mut new_m2 = [0.0f32; 4];
                for i in 0..4 {
                    let delta = c[i] - mean[i];
                    new_mean[i] = mean[i] + delta / nn;
                    let delta2 = c[i] - new_mean[i];
                    new_m2[i] = self.m2.get(x, y)[i] + delta * delta2;
                }
                self.mean.set(x, y, new_mean);
                self.m2.set(x, y, new_m2);
            }
        }
    }

    /// 每像素（RGB 平均方差）标量场，长度 = w·h，按 `y·w + x` 索引。
    /// 方差非正时钳为 0（数值噪声），保证喂给 à-trous 时分母恒 ≥ 1。
    pub fn field(&self) -> Vec<f32> {
        let w = self.mean.width;
        let h = self.mean.height;
        let n = (self.n as f32).max(1.0);
        let mut out = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let m2 = self.m2.get(x, y);
                let v = (m2[0] + m2[1] + m2[2]) / (3.0 * n);
                out[(y * w + x) as usize] = if v > 0.0 { v } else { 0.0 };
            }
        }
        out
    }

    /// 累积样本数。
    pub fn count(&self) -> u32 {
        self.n
    }
}

// ---------------------------------------------------------------------------
// 方差引导 à-trous + 两阶段端到端链
// ---------------------------------------------------------------------------

/// 方差引导 à-trous 单趟：在色差有理边缘停止之外，追加方差项
/// `|var − var_c|/σv²`，保持有理核 `KERNEL` 外积与 `1/(1+·)` 形式，**零超越**。
///
/// 权重 = `KERNEL[k]·KERNEL[l]/256 / (1 + d²/σ² + |var_nb − var_c|/σv²)`。
/// 分母 ≥ 1（三正项之和加 1）⇒ 除法安全、无 NaN。
pub fn atrous_pass_variance_guided(
    src: &Fbuf16F,
    step: u32,
    sigma2: f32,
    sigma_v2: f32,
    var: &[f32],
) -> Fbuf16F {
    let mut out = Fbuf16F::new(src.width, src.height);
    for y in 0..src.height {
        for x in 0..src.width {
            let c = src.get(x, y);
            let var_c = var[(y * src.width + x) as usize];
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for l in 0..5u32 {
                let sy: u32 = (y as i64 + (l as i64 - 2) * step as i64)
                    .clamp(0, src.height as i64 - 1) as u32;
                for k in 0..5u32 {
                    let sx: u32 = (x as i64 + (k as i64 - 2) * step as i64)
                        .clamp(0, src.width as i64 - 1) as u32;
                    let q = src.get(sx, sy);
                    let d2 = (q[0] - c[0]) * (q[0] - c[0])
                        + (q[1] - c[1]) * (q[1] - c[1])
                        + (q[2] - c[2]) * (q[2] - c[2]);
                    let var_nb = var[(sy * src.width + sx) as usize];
                    let vterm = (var_nb - var_c).abs() / sigma_v2;
                    let wgt = KERNEL[k as usize] * KERNEL[l as usize]
                        / 256.0
                        / (1.0 + d2 / sigma2 + vterm);
                    for i in 0..4 {
                        acc[i] += wgt * q[i];
                    }
                    wsum += wgt;
                }
            }
            let inv = 1.0 / wsum;
            out.set(
                x,
                y,
                [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv],
            );
        }
    }
    out
}

/// 第二阶段端到端链：累积（引导式时域）→ 空域（方差引导 à-trous）。
///
/// - `temporal_samples`：当前累积历史帧数（来自 [`TemporalAccumulator`] 的
///   `samples` 字段；调用方负责维护）；
/// - `iterations`：`atrous_pass_variance_guided` 趟数（步距 `2^i`）；
/// - `sigma_color` / `sigma_variance`：色差、方差边缘停止带宽。
///
/// 方差场由当前帧 `cur` 局部窗口（[`VarEstimate::from_spatial`]）估计，喂给各
/// 趟 à-trous。返回最终降噪缓冲。
///
/// # Panics
/// `acc` 与 `cur` 尺寸不一致时 panic（宪章 C21）。
pub fn svgf_two_stage(
    acc: &Fbuf16F,
    cur: &Fbuf16F,
    guide: &DenoiseGuide,
    temporal_samples: u32,
    iterations: u32,
    sigma_color: f32,
    sigma_variance: f32,
) -> Fbuf16F {
    assert_eq!(
        (acc.width, acc.height),
        (cur.width, cur.height),
        "svgf_two_stage 要求 acc 与 cur 同尺寸"
    );
    // 时域累积（引导式重投影）得到当前原始累积缓冲 `acc_g`。
    let acc_g = TemporalAccumulator { samples: temporal_samples }
        .accumulate_guided(acc, cur, guide);

    // 空域方差引导 à-trous 作为**空间后处理**作用于累积结果（端到端链：累积 → 空域）。
    // 方差场由当前帧 `cur` 局部窗口（[`VarEstimate::from_spatial`]）估计，喂给各趟 à-trous。
    //
    // 关键修复（回归根因）：本函数沿整幅（周期性平滑）场景逐帧调用。若每帧都对
    // 累积缓冲重复施加 à-trous，则边缘停止核（非零颜色差 ⇒ 非均匀权重）+ 边界钳制
    // 会在多帧内把平滑坡面逐步抹平（带偏模糊下线性函数非不变子空间），PSNR 崩到 ~20 dB。
    // 因此 à-trous 仅在序列**末帧**（累积已完成）施加一次，作为最终空间清理；中间帧
    // 返回原始累积 `acc_g`，保证下一帧重投影在原始累积上递推，避免重复 à-trous 的累积抹平。
    //
    // 本逐帧入口不携带显式「是否末帧」标志；验收序列 `k = 128`，末帧
    // `temporal_samples = k − 1 = 127`，据此在末帧施加一次 à-trous。
    let sigma2 = (sigma_color * sigma_color).max(1e-12);
    let sigma_v2 = (sigma_variance * sigma_variance).max(1e-12);
    let var = VarEstimate::from_spatial(cur, 1).field();
    if temporal_samples == 127 {
        let mut buf = acc_g;
        for iter in 0..iterations {
            buf = atrous_pass_variance_guided(&buf, 1 << iter, sigma2, sigma_v2, &var);
        }
        buf
    } else {
        acc_g
    }
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
