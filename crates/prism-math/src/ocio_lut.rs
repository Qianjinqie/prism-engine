//! V2.5-B1：OCIO config → 定点 1D/3D LUT 生成与求值（零 libm、零第三方依赖）。
//!
//! # 数据形态
//! - [`Lut1D`] / [`Lut3D`] 的条目是 **Q24 定点**（[`Q`] = 2²⁴），`i32` 存储；
//!   逐像素求值全程 `i64` / `i128` 整数运算（三线性 / 线性插值），**不碰浮点**。
//! - 生成期的超越函数（ExponentTransform / LogTransform）只走本仓自研
//!   [`crate::det_pow::det_ln`] / [`crate::det_pow::det_exp2`]，
//!   **不调用平台 libm**；生成结果立即量化成 Q24 定点，之后永不再走浮点。
//! - `.cube`（3D）/ `.spi1d`（1D）LUT 解析自带，尺寸不符结构化报错
//!   [`OcioError::lut_size`]（`OC1006`），不 panic、不静默截断。
//!
//! # 装配口径（写死并落档，避免"配置加了但没接"）
//! [`OcioLut::build`] 把 [`OcioConfig::view_chain`] 展平后：
//! 1. 所有 `FileTransform(.spi1d)` 烘成 **前置 1D shaper**（多个则按声明序串接）；
//!    没有则生成恒等 1D LUT（仍真生成一张 1D 表）；
//! 2. 其余 transform（Matrix / Exponent / Log / Range / Group / `.cube` FileTransform）
//!    按声明序在 cube 网格点上求值，烘成 **3D LUT**；
//! 3. 求值顺序恒为 `shaper → cube`，与 OCIO 的 shaper+cube 标准组合一致。
//! 链为空 ⇒ 返回 `None`，产品路径**完全不进 LUT 分支**（V1 逐字节兼容）。
//!
//! # 确定性
//! 网格点按 `r + size*(g + size*b)` 固定顺序生成（红最快，与 `.cube` 同口径）；
//! 所有插值用整数乘加（不依赖目标平台浮点收缩）；同输入两跑逐字节一致。

use crate::det_pow::{det_exp2, det_ln};
use crate::ocio::{OcioError, OcioResult, Transform};

/// 定点标度 Q24（1.0 ≡ 16 777 216）。
pub const Q: i64 = 1 << 24;
/// Q24 的 `f64` 镜像（只用于生成期量化，不参与逐像素求值）。
const QF: f64 = Q as f64;
/// 默认 3D LUT 边长（17 是 OCIO/Cube 常用档）。
pub const DEFAULT_CUBE_SIZE: usize = 17;
/// 默认 1D shaper 长度。
pub const DEFAULT_SHAPER_SIZE: usize = 1024;
/// 单个 Q24 条目的合法范围（允许变换输出暂时越界，求值末端再钳）。
const Q_CLAMP: i64 = Q * 64;

#[inline]
fn clamp_q(v: i64) -> i64 {
    if v < -Q_CLAMP {
        -Q_CLAMP
    } else if v > Q_CLAMP {
        Q_CLAMP
    } else {
        v
    }
}

#[inline]
fn quantize(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    let q: f64 = (x * QF).round();
    clamp_q(q as i64) as i32
}

#[inline]
fn dequantize(v: i32) -> f64 {
    v as f64 / QF
}

/// 1D LUT（Q24，RGB 三通道）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lut1D {
    /// 采样点数（≥2）。
    pub size: usize,
    /// `size × 3` 个 Q24 值。
    pub data: Vec<[i32; 3]>,
}

impl Lut1D {
    /// 生成恒等 1D LUT（Q24 精确）。
    pub fn identity(size: usize) -> Self {
        let size: usize = size.max(2);
        let mut data: Vec<[i32; 3]> = Vec::with_capacity(size);
        for i in 0..size {
            let v: i64 = (i as i64 * Q) / (size as i64 - 1);
            let q: i32 = v as i32;
            data.push([q, q, q]);
        }
        Self { size, data }
    }

    /// Q24 输入 → Q24 输出（线性插值，全整数）。
    pub fn eval_q(&self, x_q: i64) -> [i64; 3] {
        let x: i64 = clamp_q(x_q).max(0).min(Q);
        let scaled: i64 = x * (self.size as i64 - 1);
        let i0: usize = (scaled / Q) as usize;
        let i1: usize = (i0 + 1).min(self.size - 1);
        let frac: i64 = scaled - i0 as i64 * Q; // ∈ [0, Q)
        let mut out: [i64; 3] = [0; 3];
        let a: [i32; 3] = self.data[i0];
        let b: [i32; 3] = self.data[i1];
        for c in 0..3 {
            let av: i64 = a[c] as i64;
            let bv: i64 = b[c] as i64;
            out[c] = av + ((bv - av) * frac) / Q;
        }
        out
    }
}

/// 3D LUT（Q24，红最快：`idx = r + size*(g + size*b)`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lut3D {
    /// 边长（≥2）。
    pub size: usize,
    /// `size³ × 3` 个 Q24 值。
    pub data: Vec<[i32; 3]>,
}

impl Lut3D {
    /// 生成恒等 3D LUT。
    pub fn identity(size: usize) -> Self {
        let size: usize = size.max(2);
        let mut data: Vec<[i32; 3]> = Vec::with_capacity(size * size * size);
        let qf = |i: usize| -> i64 { (i as i64 * Q) / (size as i64 - 1) };
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push([qf(r) as i32, qf(g) as i32, qf(b) as i32]);
                }
            }
        }
        Self { size, data }
    }

    #[inline]
    fn at(&self, r: usize, g: usize, b: usize) -> [i32; 3] {
        self.data[r + self.size * (g + self.size * b)]
    }

    /// Q24 三分量 → Q24 输出（三线性插值，全 `i64`）。
    pub fn eval_q(&self, rgb: [i64; 3]) -> [i64; 3] {
        let n: i64 = self.size as i64 - 1;
        let mut idx: [usize; 3] = [0; 3];
        let mut frac: [i64; 3] = [0; 3];
        for c in 0..3 {
            let x: i64 = clamp_q(rgb[c]).max(0).min(Q);
            let scaled: i64 = x * n;
            idx[c] = (scaled / Q) as usize;
            frac[c] = scaled - idx[c] as i64 * Q;
        }
        let (r0, g0, b0): (usize, usize, usize) = (idx[0], idx[1], idx[2]);
        let (r1, g1, b1): (usize, usize, usize) = (
            (r0 + 1).min(self.size - 1),
            (g0 + 1).min(self.size - 1),
            (b0 + 1).min(self.size - 1),
        );
        let (fr, fg, fb): (i64, i64, i64) = (frac[0], frac[1], frac[2]);
        let c000 = self.at(r0, g0, b0);
        let c100 = self.at(r1, g0, b0);
        let c010 = self.at(r0, g1, b0);
        let c110 = self.at(r1, g1, b0);
        let c001 = self.at(r0, g0, b1);
        let c101 = self.at(r1, g0, b1);
        let c011 = self.at(r0, g1, b1);
        let c111 = self.at(r1, g1, b1);
        let mut out: [i64; 3] = [0; 3];
        for ch in 0..3 {
            // 先在 r 方向插值，再 g，再 b；除法顺序固定 ⇒ 逐位确定。
            let x00: i64 = c000[ch] as i64 + ((c100[ch] as i64 - c000[ch] as i64) * fr) / Q;
            let x10: i64 = c010[ch] as i64 + ((c110[ch] as i64 - c010[ch] as i64) * fr) / Q;
            let x01: i64 = c001[ch] as i64 + ((c101[ch] as i64 - c001[ch] as i64) * fr) / Q;
            let x11: i64 = c011[ch] as i64 + ((c111[ch] as i64 - c011[ch] as i64) * fr) / Q;
            let y0: i64 = x00 + ((x10 - x00) * fg) / Q;
            let y1: i64 = x01 + ((x11 - x01) * fg) / Q;
            out[ch] = y0 + ((y1 - y0) * fb) / Q;
        }
        out
    }
}

/// 完整 OCIO LUT 管线（1D shaper + 3D cube，全 Q24）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcioLut {
    /// 前置 1D shaper。
    pub shaper: Lut1D,
    /// 3D cube。
    pub cube: Lut3D,
    /// 展平后的 transform 条数（诊断 / golden 对拍）。
    pub transform_count: usize,
}

impl OcioLut {
    /// 恒等管线（测试与空 config 的对拍基线）。
    pub fn identity(cube_size: usize, shaper_size: usize) -> Self {
        Self {
            shaper: Lut1D::identity(shaper_size),
            cube: Lut3D::identity(cube_size),
            transform_count: 0,
        }
    }

    /// 由 config 的 view transform 链装配管线。
    ///
    /// `provider(src)` 返回 `FileTransform` 指向的 LUT 文件字节；产品路径按
    /// config.ocio 所在目录读盘，测试用内存字节，二者共用同一份解析代码。
    /// 链为空 ⇒ `Ok(None)`（产品路径不进 LUT 分支，V1 逐字节兼容）。
    pub fn build(
        chain: &[Transform],
        provider: &mut dyn FnMut(&str) -> OcioResult<Vec<u8>>,
        cube_size: usize,
        shaper_size: usize,
    ) -> OcioResult<Option<Self>> {
        let flat: Vec<Transform> = crate::ocio::OcioConfig::flatten_chain(chain);
        if flat.is_empty() {
            return Ok(None);
        }
        let mut shapers: Vec<Lut1D> = Vec::new();
        let mut cubes: Vec<Lut3D> = Vec::new();
        for t in &flat {
            if let Transform::File { src, .. } = t {
                let bytes: Vec<u8> = provider(src)?;
                let text: String = String::from_utf8(bytes).map_err(|_| {
                    OcioError::lut_source(1, format!("`{src}` 不是 UTF-8 文本"))
                })?;
                if src.ends_with(".cube") {
                    cubes.push(parse_cube_lut(&text, src)?);
                } else {
                    shapers.push(parse_spi1d_lut(&text, src)?);
                }
            }
        }
        // 1D shaper：多个按声明序串接（全整数重采样）。
        let shaper: Lut1D = if shapers.is_empty() {
            Lut1D::identity(shaper_size)
        } else {
            compose_shapers(&shapers, shaper_size)
        };
        // 3D cube：链中非 1D-File 的 transform 按声明序在网格点求值。
        let analytic: Vec<Transform> = flat
            .iter()
            .filter(|t| !matches!(t, Transform::File { src, .. } if src.ends_with(".spi1d")))
            .cloned()
            .collect();
        let cube: Lut3D = bake_cube(&analytic, &cubes, cube_size);
        Ok(Some(Self { shaper, cube, transform_count: flat.len() }))
    }

    /// 直接对 transform 链求值（生成期参考实现；V3 用它做 golden 对拍）。
    pub fn eval_chain_direct(chain: &[Transform], rgb: [f64; 3]) -> [f64; 3] {
        let mut v: [f64; 3] = rgb;
        for t in crate::ocio::OcioConfig::flatten_chain(chain).iter() {
            match t {
                Transform::Matrix { matrix, offset } => {
                    let x: [f64; 3] = v;
                    for row in 0..3 {
                        v[row] = matrix[row * 4] * x[0]
                            + matrix[row * 4 + 1] * x[1]
                            + matrix[row * 4 + 2] * x[2]
                            + matrix[row * 4 + 3]
                            + offset[row];
                    }
                }
                Transform::Exponent { value } => {
                    for c in 0..3 {
                        v[c] = det_pow_f64(v[c].max(0.0), value[c]);
                    }
                }
                Transform::Log { base, direction } => {
                    let ln_base: f64 = det_ln(*base);
                    for c in 0..3 {
                        let x: f64 = v[c] * (base - 1.0) + 1.0;
                        let y: f64 = if x > 0.0 { det_ln(x) / ln_base } else { 0.0 };
                        v[c] = if direction == "inverse" {
                            (det_pow_f64(*base, y) - 1.0) / (base - 1.0)
                        } else {
                            y
                        };
                    }
                }
                Transform::Range { min_in, max_in, min_out, max_out } => {
                    let scale: f64 = (max_out - min_out) / (max_in - min_in);
                    for c in 0..3 {
                        v[c] = (v[c] - min_in) * scale + min_out;
                    }
                }
                Transform::Group(_) => unreachable!("flatten_chain 已展开 Group"),
                Transform::File { .. } => {
                    // 直接求值参考实现不含外部 LUT（外部 LUT 由 bake 时的查表覆盖）。
                }
            }
        }
        v
    }

    /// 管线求值（Q24 → Q24，全整数）。
    pub fn apply_q(&self, rgb: [i64; 3]) -> [i64; 3] {
        let s0: [i64; 3] = self.shaper.eval_q(rgb[0]);
        let s1: [i64; 3] = self.shaper.eval_q(rgb[1]);
        let s2: [i64; 3] = self.shaper.eval_q(rgb[2]);
        let shaped: [i64; 3] = [s0[0], s1[1], s2[2]];
        self.cube.eval_q(shaped)
    }

    /// f32 入口（供现有色彩管理 [`crate`] 消费；量化只在边界发生一次）。
    pub fn apply_f32(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut q: [i64; 3] = [0; 3];
        for c in 0..3 {
            q[c] = (rgb[c] as f64 * QF).round() as i64;
        }
        let out: [i64; 3] = self.apply_q(q);
        let mut r: [f32; 3] = [0.0; 3];
        for c in 0..3 {
            r[c] = (out[c] as f64 / QF) as f32;
        }
        r
    }

    /// RGBA8 原地变换（产品出片路径；纯整数，不碰 alpha）。
    pub fn apply_rgba8(&self, pixels: &mut [u8]) {
        for px in pixels.chunks_exact_mut(4) {
            let q: [i64; 3] = [
                px[0] as i64 * Q / 255,
                px[1] as i64 * Q / 255,
                px[2] as i64 * Q / 255,
            ];
            let out: [i64; 3] = self.apply_q(q);
            for c in 0..3 {
                let v: i64 = (out[c] * 255 + Q / 2) / Q;
                px[c] = v.max(0).min(255) as u8;
            }
        }
    }

    /// 逐字段确定性摘要（V5 双跑逐字节断言用；FNV-1a，零依赖）。
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(self.transform_count as u64);
        mix(self.shaper.size as u64);
        mix(self.cube.size as u64);
        for e in &self.shaper.data {
            for c in 0..3 {
                mix(e[c] as u32 as u64);
            }
        }
        for e in &self.cube.data {
            for c in 0..3 {
                mix(e[c] as u32 as u64);
            }
        }
        h
    }
}

/// 对一批 RGBA8 帧应用 LUT（`prism-app render --ocio` 的产品路径）。
pub fn apply_lut_rgba8_frames(frames: &mut [Vec<u8>], lut: &OcioLut) {
    for frame in frames.iter_mut() {
        lut.apply_rgba8(frame);
    }
}

/// 自研确定性 `x^p`（`det_exp2(p·det_ln(x))`，零 libm）。
fn det_pow_f64(x: f64, p: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_x: f64 = det_ln(x);
    // det_exp2 只接受可归约范围内的参数；OCIO gamma 常见 [0.2, 3]。
    det_exp2(p * ln_x / std::f64::consts::LN_2)
}

/// 多个 1D LUT 串接成一张（全整数重采样，`shaper_size` 点）。
fn compose_shapers(shapers: &[Lut1D], size: usize) -> Lut1D {
    let size: usize = size.max(2);
    let mut data: Vec<[i32; 3]> = Vec::with_capacity(size);
    for i in 0..size {
        let mut v: [i64; 3] = {
            let q: i64 = (i as i64 * Q) / (size as i64 - 1);
            [q, q, q]
        };
        for lut in shapers {
            let s0: [i64; 3] = lut.eval_q(v[0]);
            let s1: [i64; 3] = lut.eval_q(v[1]);
            let s2: [i64; 3] = lut.eval_q(v[2]);
            v = [s0[0], s1[1], s2[2]];
        }
        data.push([clamp_q(v[0]) as i32, clamp_q(v[1]) as i32, clamp_q(v[2]) as i32]);
    }
    Lut1D { size, data }
}

/// 把解析 transform 链（含外部 3D LUT）在 `size³` 网格点求值，烘成 3D LUT。
fn bake_cube(chain: &[Transform], cubes: &[Lut3D], size: usize) -> Lut3D {
    let size: usize = size.max(2);
    let mut data: Vec<[i32; 3]> = Vec::with_capacity(size * size * size);
    let mut cube_idx: usize = 0;
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let q: [i64; 3] = [
                    r as i64 * Q / (size as i64 - 1),
                    g as i64 * Q / (size as i64 - 1),
                    b as i64 * Q / (size as i64 - 1),
                ];
                let v: [i64; 3] = eval_chain_q(chain, cubes, &mut cube_idx, q);
                data.push([
                    clamp_q(v[0]) as i32,
                    clamp_q(v[1]) as i32,
                    clamp_q(v[2]) as i32,
                ]);
            }
        }
    }
    Lut3D { size, data }
}

/// 生成期链求值（Q24 整数；超越件走自研 det_*，结果立即量化回 Q24）。
fn eval_chain_q(chain: &[Transform], cubes: &[Lut3D], cube_idx: &mut usize, mut v: [i64; 3]) -> [i64; 3] {
    for t in chain {
        match t {
            Transform::Matrix { matrix, offset } => {
                let m: [i64; 16] = {
                    let mut a: [i64; 16] = [0; 16];
                    for i in 0..16 {
                        a[i] = (matrix[i] * QF).round() as i64;
                    }
                    a
                };
                let o: [i64; 4] = {
                    let mut a: [i64; 4] = [0; 4];
                    for i in 0..4 {
                        a[i] = (offset[i] * QF).round() as i64;
                    }
                    a
                };
                let x: [i64; 3] = v;
                for row in 0..3 {
                    let acc: i128 = m[row * 4] as i128 * x[0] as i128
                        + m[row * 4 + 1] as i128 * x[1] as i128
                        + m[row * 4 + 2] as i128 * x[2] as i128
                        + m[row * 4 + 3] as i128 * Q as i128
                        + o[row] as i128 * Q as i128;
                    v[row] = clamp_q((acc / Q as i128) as i64);
                }
            }
            Transform::Exponent { value } => {
                for c in 0..3 {
                    let x: f64 = dequantize(clamp_q(v[c]) as i32);
                    v[c] = (det_pow_f64(x.max(0.0), value[c]) * QF).round() as i64;
                    v[c] = clamp_q(v[c]);
                }
            }
            Transform::Log { base, direction } => {
                let ln_base: f64 = det_ln(*base);
                for c in 0..3 {
                    let x: f64 = dequantize(clamp_q(v[c]) as i32);
                    let a: f64 = x * (base - 1.0) + 1.0;
                    let y: f64 = if a > 0.0 { det_ln(a) / ln_base } else { 0.0 };
                    let out: f64 = if direction == "inverse" {
                        (det_pow_f64(*base, y) - 1.0) / (base - 1.0)
                    } else {
                        y
                    };
                    v[c] = clamp_q((out * QF).round() as i64);
                }
            }
            Transform::Range { min_in, max_in, min_out, max_out } => {
                for c in 0..3 {
                    let x: f64 = dequantize(clamp_q(v[c]) as i32);
                    let y: f64 = (x - min_in) * (max_out - min_out) / (max_in - min_in) + min_out;
                    v[c] = clamp_q((y * QF).round() as i64);
                }
            }
            Transform::File { src, .. } => {
                if src.ends_with(".cube") {
                    if let Some(lut) = cubes.get(*cube_idx) {
                        v = lut.eval_q(v);
                    }
                    *cube_idx += 1;
                }
            }
            Transform::Group(_) => {} // flatten 后不会出现；防御性空实现，不 panic。
        }
    }
    v
}

/// 解析 `.cube`（3D LUT）。
///
/// 逐行扫 `LUT_3D_SIZE n`、`TITLE`、`DOMAIN_*` 与数据三元组；
/// 数据行数必须**恰好** n³，否则 `OC1006`（判据 V4「错 LUT 尺寸」负控）。
pub fn parse_cube_lut(text: &str, src: &str) -> OcioResult<Lut3D> {
    let mut declared: Option<usize> = None;
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no: usize = i + 1;
        let line: &str = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let upper: String = line.to_ascii_uppercase();
        if upper.starts_with("LUT_3D_SIZE") {
            let n: usize = line
                .split_whitespace()
                .last()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| {
                    OcioError::lut_size(line_no, format!("`{src}` 的 LUT_3D_SIZE 不是正整数"))
                })?;
            if n < 2 || n > 256 {
                return Err(OcioError::lut_size(
                    line_no,
                    format!("`{src}` 的 LUT_3D_SIZE={n} 越界（允许 2..=256）"),
                ));
            }
            declared = Some(n);
            continue;
        }
        if upper.starts_with("TITLE")
            || upper.starts_with("DOMAIN_MIN")
            || upper.starts_with("DOMAIN_MAX")
            || upper.starts_with("LUT_1D_SIZE")
            || upper.starts_with("LUT_1D_INPUT_RANGE")
        {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(OcioError::lut_size(
                line_no,
                format!("`{src}` 第 {line_no} 行应有 3 个浮点分量，实际 {} 个", parts.len()),
            ));
        }
        let mut triple: [f64; 3] = [0.0; 3];
        for (c, p) in parts.iter().enumerate() {
            triple[c] = p.parse::<f64>().map_err(|_| {
                OcioError::number(line_no, format!("`{src}` 的 `{p}` 不是合法浮点数"))
            })?;
            if !triple[c].is_finite() {
                return Err(OcioError::number(
                    line_no,
                    format!("`{src}` 的 `{p}` 非有限数"),
                ));
            }
        }
        rows.push(triple);
    }
    let n: usize = declared.ok_or_else(|| {
        OcioError::lut_size(1, format!("`{src}` 缺 `LUT_3D_SIZE n` 声明"))
    })?;
    if rows.len() != n * n * n {
        return Err(OcioError::lut_size(
            1,
            format!(
                "`{src}` 声明 LUT_3D_SIZE {n}（应 {expected} 行），实际 {actual} 行",
                expected = n * n * n,
                actual = rows.len()
            ),
        ));
    }
    Ok(Lut3D {
        size: n,
        data: rows
            .iter()
            .map(|t| [quantize(t[0]), quantize(t[1]), quantize(t[2])])
            .collect(),
    })
}

/// 解析 `.spi1d`（1D shaper LUT）。
///
/// 支持 1 通道（复制到 RGB）与 3 通道两种数据行；`Length` 与数据行数
/// 必须一致，否则 `OC1006`。
pub fn parse_spi1d_lut(text: &str, src: &str) -> OcioResult<Lut1D> {
    let mut declared: Option<usize> = None;
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no: usize = i + 1;
        let line: &str = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let upper: String = line.to_ascii_uppercase();
        if upper.starts_with("LENGTH") {
            declared = line
                .split_whitespace()
                .last()
                .and_then(|s| s.parse::<usize>().ok());
            continue;
        }
        if upper.starts_with("VERSION")
            || upper.starts_with("FROM")
            || upper.starts_with("TO")
            || upper.starts_with("COMPONENTS")
            || upper.starts_with("BITS_PER_SAMPLE")
        {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() || parts.len() > 3 {
            return Err(OcioError::lut_size(
                line_no,
                format!("`{src}` 第 {line_no} 行数据列数 {} 不合法（1 或 3）", parts.len()),
            ));
        }
        let mut triple: [f64; 3] = [0.0; 3];
        for (c, p) in parts.iter().enumerate() {
            let v: f64 = p.parse::<f64>().map_err(|_| {
                OcioError::number(line_no, format!("`{src}` 的 `{p}` 不是合法浮点数"))
            })?;
            if !v.is_finite() {
                return Err(OcioError::number(line_no, format!("`{src}` 的 `{p}` 非有限数")));
            }
            triple[c] = v;
        }
        if parts.len() == 1 {
            triple[1] = triple[0];
            triple[2] = triple[0];
        }
        rows.push(triple);
    }
    let n: usize = declared.ok_or_else(|| {
        OcioError::lut_size(1, format!("`{src}` 缺 `Length` 声明"))
    })?;
    if n < 2 || rows.len() != n {
        return Err(OcioError::lut_size(
            1,
            format!("`{src}` 声明 Length {n}，实际数据 {actual} 行", actual = rows.len()),
        ));
    }
    Ok(Lut1D {
        size: n,
        data: rows
            .iter()
            .map(|t| [quantize(t[0]), quantize(t[1]), quantize(t[2])])
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_lut_is_exact_at_grid_points() {
        let lut = OcioLut::identity(17, 64);
        for i in 0..17 {
            let q: i64 = i as i64 * Q / 16;
            let out: [i64; 3] = lut.apply_q([q, q, q]);
            for c in 0..3 {
                assert!((out[c] - q).abs() <= 1, "out={out:?} q={q}");
            }
        }
    }

    #[test]
    fn cube_lut_wrong_size_is_structured_error() {
        let bad = "LUT_3D_SIZE 2\n0 0 0\n1 1 1\n";
        let e = parse_cube_lut(bad, "bad.cube").unwrap_err();
        assert_eq!(e.code, "OC1006");
        assert!(e.problem.contains("8"));
    }
}
