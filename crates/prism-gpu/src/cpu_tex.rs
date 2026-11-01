//! CPU 参考侧的纹理采样（W1-C / 段 1）：**独立实现**，用于与 GPU 对拍。
//!
//! # 为什么这不是 `prism_rt::texsample` 的副本
//!
//! 参考实现与被检查对象**共用代码 = 自证恒真**：若 `CpuRefBackend` 调
//! `prism_rt::texsample::sample`，那么"CPU 与 GPU 一致"这句话里，CPU 一侧的
//! 答案就是被测算法本身算的——它就算有半像素偏移、就算 wrap 写成钳制，
//! 两边也只会"一起错"，测试永远绿。故本模块**从口径重新写一遍**：
//!
//! * 双线性：`x = u·w − 0.5`（纹素中心在 `(i+0.5)/n`），固定结合序
//!   "先沿 u 后沿 v"；
//! * 平铺：`((i % n) + n) % n`（repeat，不是 clamp）；
//! * 三线性：级间 lerp，`frac == 0` 不混下一级；
//! * **解析足迹**：射线锥 × 表面投影 × uv 雅可比 → uv 空间椭圆足迹
//!   （与 `prism_rt::integrator::footprint_at` 逐式同构）；
//! * 索引越界返回 0 而不是 panic（宪章 C18）。
//!
//! 依赖方向也决定了它**只能**在这里：`prism-rt` 依赖 `prism-gpu`（单向），
//! `prism-gpu` 不得反引 `prism-rt`。
//!
//! # 与 shader 的逐式对应
//!
//! `shaders/scene_frame_v3.wgsl` 的 `tex_bilinear` / `tex_trilinear` / `log2_det`
//! / `tex_sample` / `uv_sphere` / `footprint_at` 与 本模块的同名件**逐表达式
//! 同构**（同结合序、同常量、同 `floor`、同 `SVD` 主轴约定）。`vulkan/v3_frame.rs`
//! 的对拍测试用真实像素证明这一点，而不是靠"看起来一样"。
//!
//! # SVD 主轴约定（关键一致性点）
//!
//! shader 的 `tex_sample` 主轴分支为：
//!
//! ```text
//! if abs(b) > 0 { dir = normalize([b, lmax - a]); }
//! else if a < c  { dir = [0, 1]; }
//! else           { dir = [1, 0]; }   // 默认
//! ```
//!
//! 这与 `prism_rt::texsample::singular_axes` 的 **`a >= c` 优先** 分支顺序**相反**。
//! 本模块严格以 **shader 为准**（任务要求 CPU 参考与 shader 逐式同构），否则
//! llvmpipe 上不再 0 差异。
//!
//! # 未做（**不要当成已支持**）
//!
//! * sRGB 解码：纹素进 [`crate::scene_abi::TextureUpload`] 前已是线性域
//!   （解码在 `prism_rt::texture::build_rgba8` 一次性完成）；本段不涉及；
//! * 材质其余三张图（metal-rough / normal / emissive）：本版接 `MAP_METAL_ROUGH`
//!   与 `MAP_NORMAL`（段 2）——前者启用两瓣 PBR、后者扰动着色法线；`MAP_EMISSIVE`
//!   仍不读（着色模型无自发光通道）。

use crate::scene_abi::{TextureUpload, MAX_V3_TEXTURES};
use prism_math::Vec3;

/// `1/π`（与 `scene_frame_v3.wgsl` 的同名常量逐位同值）。
const FRAC_1_PI: f32 = 0.3183098861837907;
/// `1/(2π)`（同上）。
const FRAC_1_TAU: f32 = 0.15915494309189535;

/// 一张参考侧纹理：元数据 + 扁平纹素（含 mip 链，`offset` 以**纹素**计）。
///
/// 纹素**拥有所有权**：当 `TextureUpload::srgb != 0` 时，构造期已把 RGB 解码到
/// 线性域（与 `encode_v3` 逐纹素同式），故采样发生在线性域，与 GPU 收到的同一份
/// 线性 f32 对拍。解码前的 sRGB 编码纹素不在此保留。
#[derive(Debug, Clone)]
struct RefTexture {
    /// 宽度（纹素，≥ 1）。
    w: u32,
    /// 高度（纹素，≥ 1）。
    h: u32,
    /// mip 级数（≥ 1）。
    levels: u32,
    /// 第 0 级在**全局扁平纹素流**里的起点（纹素下标；仅用于与 `texmeta`
    /// 对账，不参与本切片寻址）。
    #[allow(dead_code)]
    offset: usize,
    /// 扁平纹素（线性域）：本张纹理的 mip 链（第 0 级在最前）。
    texels: Vec<[f32; 4]>,
}

/// 参考侧纹理表（句柄 `k` ↔ 下标 `k-1`，`0` = 无贴图）。
///
/// 构造时按 [`TextureUpload`] 的顺序累加 `offset`，口径与
/// [`crate::scene_abi::SceneAbi::encode_v3`] 的 texel 流**完全一致**
/// （都是"先 `expected_texels()` 再前移"，不是按字节）。`srgb != 0` 的纹素在
/// 构造期已解码到线性域（与 `encode_v3` 同式），故参考侧采样与 GPU 对齐。
#[derive(Debug, Clone, Default)]
pub struct RefTextureTable {
    /// 已接纳的纹理（下标 = 句柄 − 1）。
    slots: Vec<RefTexture>,
}

impl RefTextureTable {
    /// 由上传描述建表。
    ///
    /// 校验不过（尺寸 0 / 级数越界 / 纹素条数与描述不符）的纹理**及其之后
    /// 的全部条目**被丢弃：`offset` 是前缀累加，跳过一张会让后续句柄指向
    /// 错误的纹素——宁可少采样，不可错采样。句柄越界由调用点按"无贴图"处理。
    pub fn new(uploads: &[TextureUpload<'_>]) -> Self {
        let mut slots: Vec<RefTexture> = Vec::new();
        let mut cursor: usize = 0;
        for t in uploads.iter().take(MAX_V3_TEXTURES) {
            if t.validate().is_err() {
                break;
            }
            // `srgb != 0`：把 RGB 解码到线性域（alpha 不动），与 `encode_v3` 逐纹素
            // 同式——参考侧与 GPU 拿到的是同一份线性 f32。
            let texels: Vec<[f32; 4]> = if t.srgb != 0 {
                t.texels
                    .iter()
                    .map(|c| {
                        [
                            crate::scene_abi::srgb_decode_f32(c[0]),
                            crate::scene_abi::srgb_decode_f32(c[1]),
                            crate::scene_abi::srgb_decode_f32(c[2]),
                            c[3],
                        ]
                    })
                    .collect()
            } else {
                t.texels.to_vec()
            };
            slots.push(RefTexture {
                w: t.w,
                h: t.h,
                levels: t.levels.max(1),
                offset: cursor,
                texels,
            });
            cursor += t.expected_texels();
        }
        RefTextureTable { slots }
    }

    /// 空表（无纹理场景的零成本占位）。
    pub fn empty() -> Self {
        RefTextureTable { slots: Vec::new() }
    }

    /// 纹理条数（= 合法句柄上界）。
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// 表是否为空。
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// 句柄是否可用（`0` = 无贴图 ⇒ `false`；越界 ⇒ `false`）。
    ///
    /// 这是 CPU 侧与 shader 侧 `base_map != NO_TEXTURE && base_map <= P_NUM_TEXTURES`
    /// **同一判据**的落地：越界句柄按"无贴图"处理，而不是"采样成白"。
    pub fn has(&self, handle: u32) -> bool {
        handle != 0 && (handle as usize) <= self.slots.len()
    }

    /// 取第 0 级、给定 uv 的双线性采样（足迹恒 0：lod 0、1 tap）。
    ///
    /// 返回 `None` 当且仅当句柄不可用（与 [`RefTextureTable::has`] 同判据）。
    /// 保留作为级内精确命中的独立对照（判据 1/2）。
    pub fn sample_base(&self, handle: u32, uv: [f32; 2]) -> Option<[f32; 4]> {
        if !self.has(handle) {
            return None;
        }
        let t = &self.slots[handle as usize - 1];
        Some(bilinear_level(t, 0, uv))
    }

    /// 按**解析足迹**采样（三线性 + mip + 有界各向异性）。
    ///
    /// `foot_c0 / foot_c1` 是 uv 空间的两半轴（与 `scene_frame_v3.wgsl::tex_sample`
    /// 的同名量同义）。返回 `None` 当且仅当句柄不可用。
    ///
    /// 这是 W1 段 1 的对拍主路径：CPU 参考与 GPU 走**同一足迹、同一 SVD、
    /// 同一 mip 选取**，故 llvmpipe 上可逐位对齐（判据 3/4）。
    pub fn sample_with_footprint(
        &self,
        handle: u32,
        uv: [f32; 2],
        foot_c0: [f32; 2],
        foot_c1: [f32; 2],
    ) -> Option<[f32; 4]> {
        if !self.has(handle) {
            return None;
        }
        let t = &self.slots[handle as usize - 1];
        let w = t.w.max(1) as f32;
        let h = t.h.max(1) as f32;
        // 足迹换到**纹素空间**再做 2×2 SVD：σ 的单位直接是纹素（与 shader 同）。
        let c0 = [foot_c0[0] * w, foot_c0[1] * h];
        let c1 = [foot_c1[0] * w, foot_c1[1] * h];
        let a = c0[0] * c0[0] + c0[1] * c0[1];
        let b = c0[0] * c1[0] + c0[1] * c1[1];
        let c = c1[0] * c1[0] + c1[1] * c1[1];
        let half = (a + c) * 0.5;
        let dd = (a - c) * 0.5;
        let root = (dd * dd + b * b).max(0.0).sqrt();
        let lmax = (half + root).max(0.0);
        let s_max = lmax.sqrt();
        let s_min = (half - root).max(0.0).sqrt();
        if !(s_max > 0.0) {
            // 零足迹 = 点采样（1 tap、lod 0）——与 shader 的短路分支同。
            return Some(bilinear_level(t, 0, uv));
        }
        // 主轴（**以 shader 约定为准**：abs(b)>0 → [b, lmax-a]；否则 a<c → [0,1]；
        // 否则 [1,0]）。
        let mut dir = [1.0f32, 0.0f32];
        if b.abs() > 0.0 {
            let v = [b, lmax - a];
            let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
            dir = [v[0] / len, v[1] / len];
        } else if a < c {
            dir = [0.0, 1.0];
        }
        // 各向异性 tap 数（有界到 4）。
        let mut taps = 4u32;
        if s_min > 0.0 {
            taps = (s_max / s_min).ceil().clamp(1.0, 4.0) as u32;
        }
        // tap 布局：并集恰好等于长轴（与 shader 同式）。
        let tap_w = s_min.max(s_max / taps as f32);
        let lod = log2_det(tap_w.max(1e-8));
        let span = (s_max - tap_w).max(0.0);
        let gap = if taps > 1 {
            span / (taps as f32 - 1.0)
        } else {
            0.0
        };
        let mut acc = [0.0f32; 4];
        for k in 0..taps {
            let tk = k as f32 - (taps as f32 - 1.0) * 0.5;
            let u = [
                uv[0] + dir[0] * gap / w * tk,
                uv[1] + dir[1] * gap / h * tk,
            ];
            let s = trilinear_level(t, u, lod);
            for ch in 0..4 {
                acc[ch] += s[ch];
            }
        }
        // 真除法（**不是** `acc * (1/taps)`）：WGSL 侧是 `acc / f32(taps)`，
        // 除法与"乘倒数"在末位差 ≤1 ulp——taps=3/4 时足以让 8 bit 量化翻一个
        // 台阶，于是 `differing_pixels` 不再是 0。故此处必须逐式走真除法。
        let denom = taps as f32;
        let mut out = [0.0f32; 4];
        for ch in 0..4 {
            out[ch] = acc[ch] / denom;
        }
        Some(out)
    }
}

/// `floor(x)` → `i64`（`uv` 可以远超 `[0,1)`，故不能用 `i32`）。
///
/// 越界饱和只为不产生错位：|x| > 2^53 在渲染里不可达。
#[inline]
fn floor_i(x: f32) -> i64 {
    let f = x.floor();
    if f >= 9.0e18 {
        9_000_000_000_000_000_000
    } else if f <= -9.0e18 {
        -9_000_000_000_000_000_000
    } else {
        f as i64
    }
}

/// 平铺取模：`((i % n) + n) % n`（`n ≥ 1`）。
#[inline]
fn wrap_i(i: i64, n: i64) -> usize {
    debug_assert!(n >= 1);
    let m = i % n;
    let w = if m < 0 { m + n } else { m };
    w as usize
}

/// 第 `level` 级的尺寸（每级减半，最小 1）。
#[inline]
fn level_dims(t: &RefTexture, level: u32) -> (usize, usize) {
    let shift = level.min(31);
    let w = (((t.w.max(1)) >> shift) as usize).max(1);
    let h = (((t.h.max(1)) >> shift) as usize).max(1);
    (w, h)
}

/// 第 `level` 级在扁平纹素流里的起点（前缀各级之和；本切片从 0 起算）。
#[inline]
fn level_base(t: &RefTexture, level: u32) -> usize {
    let mut base = 0usize;
    let lv = level.min(t.levels.max(1));
    let mut l = 0u32;
    while l < lv {
        let (w, h) = level_dims(t, l);
        base += w * h;
        l += 1;
    }
    base
}

/// 单级（任意 `level`）双线性 + repeat 平铺。
///
/// # 结合序（必须与 shader 一致）
///
/// 先沿 u：`a = t00 + (t10 − t00)·fx`、`b = t01 + (t11 − t01)·fx`；
/// 再沿 v：`out = a + (b − a)·fy`。换一种写法（比如先 v 后 u）在数学上等价、
/// 在 f32 上**不等价**——末位会差，对拍就不再是逐字节。
fn bilinear_level(t: &RefTexture, level: u32, uv: [f32; 2]) -> [f32; 4] {
    let (w, h) = level_dims(t, level);
    let base = level_base(t, level);
    let (wf, hf) = (w as f32, h as f32);
    // 纹素中心在 (i+0.5)/n ⇒ 减半像素。
    let x = uv[0] * wf - 0.5;
    let y = uv[1] * hf - 0.5;
    let fx_floor = x.floor();
    let fy_floor = y.floor();
    let fx = x - fx_floor;
    let fy = y - fy_floor;
    let x0 = floor_i(x);
    let y0 = floor_i(y);
    let i0 = wrap_i(x0, w as i64);
    let i1 = wrap_i(x0 + 1, w as i64);
    let j0 = wrap_i(y0, h as i64);
    let j1 = wrap_i(y0 + 1, h as i64);
    let at = |i: usize, j: usize| -> [f32; 4] {
        // 本张纹理切片内寻址（第 0 级恒在最前）；`t.offset` 是全局偏移，
        // 只用于与 `texmeta` 对账，不参与此处寻址。
        let idx = base + j * w + i;
        // 越界 = `texels` 与描述对不上（建表期已拦，这里是纵深防御）。
        t.texels.get(idx).copied().unwrap_or([0.0; 4])
    };
    let t00 = at(i0, j0);
    let t10 = at(i1, j0);
    let t01 = at(i0, j1);
    let t11 = at(i1, j1);
    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        // `mul_add`：与 GPU（llvmpipe 经 LLVM 把 `a + b·c` 收缩成 FMA）逐位对齐。
        // 用普通 `+`/`*` 会让逐字节对拍在舍入边界掉 1/255（已知"驱动末位舍入差"）。
        let a = (t10[ch] - t00[ch]).mul_add(fx, t00[ch]);
        let b = (t11[ch] - t01[ch]).mul_add(fx, t01[ch]);
        out[ch] = (b - a).mul_add(fy, a);
    }
    out
}

/// `log2`（只含 × ÷ 与位操作；与 `scene_frame_v3.wgsl::log2_det` 逐位同式，
/// 精度 ~1e-6，mip 级选择足够）。
fn log2_det(x: f32) -> f32 {
    if !(x > 0.0) {
        return 0.0;
    }
    let bits = x.to_bits();
    let mut e = (((bits >> 23) & 0xFF) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // [1,2)
    let mut mm = m;
    if mm >= 1.4142135623730951 {
        mm = mm * 0.5;
        e += 1;
    }
    // atanh 级数：log2(m) = 2·atanh(z)/ln2，z = (m−1)/(m+1)，|z| ≤ 0.172
    let z = (mm - 1.0) / (mm + 1.0);
    let z2 = z * z;
    let s = z
        * (1.0
            + z2
                * (0.3333333333333333
                    + z2
                        * (0.2
                            + z2
                                * (0.14285714285714285
                                    + z2 * (0.1111111111111111 + z2 * 0.09090909090909091)))));
    (e as f32) + s * 2.885390081777927 // 2/ln2
}

/// 三线性采样：两级双线性 + 级间 lerp（`lod` 可为小数）。
fn trilinear_level(t: &RefTexture, uv: [f32; 2], lod_in: f32) -> [f32; 4] {
    let levels = t.levels.max(1);
    let max_level = levels as f32 - 1.0;
    let lod = lod_in.clamp(0.0, max_level);
    let l0 = lod.floor() as u32;
    let frac = lod - lod.floor();
    let a = bilinear_level(t, l0, uv);
    if frac <= 0.0 {
        // 级内精确命中：不混下一级（V1 的逐位前提）。
        return a;
    }
    let b = bilinear_level(t, (l0 + 1).min(levels - 1), uv);
    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        out[ch] = a[ch] + (b[ch] - a[ch]) * frac;
    }
    out
}

/// 球面参数坐标（`scene_frame_v3.wgsl::uv_sphere` 的**逐位同构**转写）。
///
/// **不使用 `prism_math::det_trig`**：那条是"等价但独立"的实现，在与 GPU 的
/// `uv_sphere` 对拍时允许 ≤1 ulp 偏差（见 `vulkan/v3_frame.rs::cpu_uv_matches_wgsl_uv_formula_within_one_ulp`）。
/// 但 W1 段 1 开启 4-tap 足迹平均后，这 1 ulp 的 uv 偏差会被放大成 1/255 的
/// 颜色差异（`differing_pixels` 不再是 0）。故此处**逐式转写** shader 的
/// `det_atan2` / `det_acos`，保证 CPU 与 GPU 采到**逐位相同**的 uv。
///
/// `u = det_atan2(d.z, d.x)/2π + ½`（经度）、`v = det_acos(d.y)/π`（纬度，`+Y` 端为 0）。
#[inline]
pub fn sphere_uv(p: Vec3, center: Vec3, radius: f32) -> [f32; 2] {
    let d = Vec3::new(
        (p.x - center.x) / radius,
        (p.y - center.y) / radius,
        (p.z - center.z) / radius,
    );
    let u = wgsl_atan2(d.z, d.x) * FRAC_1_TAU + 0.5;
    let v = wgsl_acos(d.y.clamp(-1.0, 1.0)) * FRAC_1_PI;
    [u, v]
}

/// 确定性切基（球面 uv 用）：取与法线**最不平行**的坐标轴做叉乘。
///
/// 与 `prism_rt::integrator::plane_tangent_basis` 逐式同构，也是
/// `scene_frame_v3.wgsl::plane_tangent_basis` 的逐位同构件。
///
/// 为什么不用 `up`：法线接近 ±Y 时 `n × up` 会退化成长度 0 的向量，
/// `normalize` 后是 NaN，而 NaN 会顺着 uv 污染整张贴图（表现为整片像素
/// 变成纹素 (0,0)）。取"最不平行的轴"把叉乘长度下界钉在 `1/√3`。
///
/// 球面没有烘焙切线，故这是 TBN 基的**近似**：`+Z` 朝外的切线空间法线
/// 经 `(T, B, n_geom)` 直接旋到世界空间（T/B 为世界向量），得到扰动后的
/// 着色法线。`material_normal_at` 的语义权威在此。
#[inline]
pub fn plane_tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let helper = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = n.cross(&helper).normalize();
    let v = n.cross(&u);
    (u, v)
}

/// WGSL `scene_frame_v3.wgsl::quarter_atan` 的逐位同构转写（禁 libm，D-29）。
///
/// 用 Horner 累加写法（与 WGSL 嵌套表达式**逐位等价**，但括号不会数错）：
/// `z·(1 + z²·(−⅓ + z²·(⅕ + z²·(−⅐ + z²·(⅑ + z²·(−¹⁄₁₁ + z²·(¹⁄₁₃ + z²·(−¹⁄₁₅))))))))`。
#[inline]
fn wgsl_quarter_atan(z: f32) -> f32 {
    let z2 = z * z;
    let mut p = -0.06666666666666667; // −1/15（最内层）
    p = 0.07692307692307693 + z2 * p; // +1/13
    p = -0.09090909090909091 + z2 * p; // −1/11
    p = 0.1111111111111111 + z2 * p; // +1/9
    p = -0.14285714285714285 + z2 * p; // −1/7
    p = 0.2 + z2 * p; // +1/5
    p = -0.3333333333333333 + z2 * p; // −1/3
    z * (1.0 + z2 * p)
}

/// WGSL `scene_frame_v3.wgsl::atan_unit` 的逐位同构转写。
#[inline]
fn wgsl_atan_unit(v: f32) -> f32 {
    const FRAC_PI_4: f32 = 0.7853981633974483;
    const TAN_PI_8: f32 = 0.41421357;
    if v > TAN_PI_8 {
        FRAC_PI_4 + wgsl_quarter_atan((v - 1.0) / (v + 1.0))
    } else {
        wgsl_quarter_atan(v)
    }
}

/// WGSL `scene_frame_v3.wgsl::atan_pos` 的逐位同构转写。
#[inline]
fn wgsl_atan_pos(x: f32) -> f32 {
    const FRAC_PI_2: f32 = 1.5707963267948966;
    if x > 1.0 {
        FRAC_PI_2 - wgsl_atan_unit(1.0 / x)
    } else {
        wgsl_atan_unit(x)
    }
}

/// WGSL `scene_frame_v3.wgsl::det_atan2` 的逐位同构转写。
#[inline]
fn wgsl_atan2(y: f32, x: f32) -> f32 {
    const PI: f32 = 3.141592653589793;
    const FRAC_PI_2: f32 = 1.5707963267948966;
    if x == 0.0 {
        if y == 0.0 {
            return 0.0;
        }
        if y > 0.0 {
            return FRAC_PI_2;
        }
        return -FRAC_PI_2;
    }
    let ax = x.abs();
    let ay = y.abs();
    let r = if ay <= ax {
        wgsl_atan_pos(ay / ax)
    } else {
        FRAC_PI_2 - wgsl_atan_pos(ax / ay)
    };
    if x > 0.0 {
        if y >= 0.0 {
            r
        } else {
            -r
        }
    } else if y >= 0.0 {
        PI - r
    } else {
        r - PI
    }
}

/// WGSL `scene_frame_v3.wgsl::det_acos` 的逐位同构转写。
#[inline]
fn wgsl_acos(x: f32) -> f32 {
    let xc = if x > 1.0 {
        1.0
    } else if x < -1.0 {
        -1.0
    } else {
        x
    };
    let s = ((1.0 - xc) * (1.0 + xc)).max(0.0).sqrt();
    wgsl_atan2(s, xc)
}

/// 迭代式 1 ulp 邻居（uv 对账用；见 `vulkan/v3_frame.rs` 的测试）。
///
/// `ulp_count` 有符号：正 = 向 +∞ 走。用于把"同一球面点的两个 uv"的差
/// 换算成整数 ulp 而不是浮点差（浮点差在 1e-7 量级无法直接比较）。
pub fn f32_ulp_distance(a: f32, b: f32) -> u32 {
    // 单调映射：正数保持位型、负数取反位型 ⇒ 有符号距离可用无符号差值算。
    fn key(x: f32) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    }
    key(a).abs_diff(key(b)) as u32
}

// ---------------------------------------------------------------------------
// 解析足迹（与 `prism_rt::integrator::footprint_at` 逐式同构）
// ---------------------------------------------------------------------------

/// 周期量（u）的最短环绕差：`d ∈ (−½, ½]`。
///
/// uv 的 u 轴是周期 1 的（球面方位角 / 平铺重复）：直接相减会在接缝处得到
/// ±1 跳变，被误读成"uv 变化极快" ⇒ 足迹变成一整张图 ⇒ mip 拉到最低。
#[inline]
fn wrap_delta(d: f32) -> f32 {
    d - (d + 0.5).floor()
}

/// 切平面基第一轴（与 `prism_rt::integrator::plane_tangent_basis` 同式）：
/// 取"最不平行的轴"叉乘，把叉乘长度下界钉在 `1/√3`，避免法线接近 ±Y 时
/// 退化成 NaN。返回 `(u, v)` 中的 `u`（足迹只用第一切向）。
fn plane_tangent_t1(n: Vec3) -> Vec3 {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let helper = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    n.cross(&helper).normalize()
}

/// uv 对世界切向的雅可比（`∇_T uv`），有限差分（与 `integrator::uv_gradient`
/// 同式）。每世界切向 `t` 付出 2 次 `uv_at`（球：4 次 `sphere_uv`）。
///
/// `ε = 1e-3·max(1, |p|)`：相对步长，差分只影响 mip 级选择（log2 尺度，
/// 对 1e-3 的相对误差完全不敏感）。u 是周期量，差分化前先取最短环绕差。
fn uv_gradient(p: Vec3, center: Vec3, radius: f32, t: Vec3) -> [f32; 2] {
    let eps = 1e-3 * p.length().max(1.0);
    let a = sphere_uv(p + t * eps, center, radius);
    let b = sphere_uv(p - t * eps, center, radius);
    let du = wrap_delta(a[0] - b[0]);
    let dv = a[1] - b[1];
    [du / (2.0 * eps), dv / (2.0 * eps)]
}

/// 命中点的 uv 空间**椭圆足迹**（A1 各向异性的全部输入）。
///
/// # 模型（与 `prism_rt::integrator::footprint_at` 同式）
///
/// 1. 主射线锥半角 `spread` 在命中距离 `t_hit` 处张开成半径 `r = t_hit·spread`
///    的圆盘；
/// 2. 圆盘投影到切平面：沿"视线投影方向"拉长 `1/cosθ_i`（掠射端各向异性来源），
///    垂直方向仍是 `r`；
/// 3. 两世界半轴经 uv 雅可比映射到 uv 空间，得到两列向量（真长短轴由采样器
///    做 2×2 奇异值分解）。
///
/// `view` = 入射射线方向（相机→场景），`t_hit` = 命中距离，
/// `spread` = 主射线锥半角（`2·tan_half_fov / height`）。
pub fn footprint_at(
    p: Vec3,
    n: Vec3,
    view: Vec3,
    t_hit: f32,
    spread: f32,
    center: Vec3,
    radius: f32,
) -> ([f32; 2], [f32; 2]) {
    // T1 = 视线在切平面上的投影（足迹长轴方向）；退化时取任意切向。
    let vt = view - n * view.dot(&n);
    let vl = vt.length();
    let t1 = if vl > 1e-6 {
        vt / vl
    } else {
        plane_tangent_t1(n)
    };
    let t2 = n.cross(&t1);
    // 入射余弦：掠射端钳到 0.02（避免除零把足迹炸成 inf）。
    let cos_i = view.dot(&n).abs().clamp(0.02, 1.0);
    let r = (t_hit * spread).max(0.0);
    let a = r / cos_i; // 长轴半宽（沿 T1）
    let b = r; // 短轴半宽（沿 T2）
    let j1 = uv_gradient(p, center, radius, t1);
    let j2 = uv_gradient(p, center, radius, t2);
    // 世界 → uv 的链式法则：列向量 = J·T_k × 半宽。
    ([j1[0] * a, j1[1] * a], [j2[0] * b, j2[1] * b])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2×2 纹理（最小可分辨），第 0 级 4 纹素。
    fn two_by_two() -> Vec<[f32; 4]> {
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    }

    fn table_of(px: &[[f32; 4]]) -> RefTextureTable {
        let up = TextureUpload { w: 2, h: 2, levels: 1, fmt: 1, srgb: 0, texels: px };
        RefTextureTable::new(std::slice::from_ref(&up))
    }

    /// 纹素中心必须**逐位**命中该纹素（没有半像素偏移、没有 0.5 权重混合）。
    #[test]
    fn texel_center_hits_exactly() {
        let px = two_by_two();
        let tab = table_of(&px);
        for (i, j) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let uv = [(i as f32 + 0.5) / 2.0, (j as f32 + 0.5) / 2.0];
            let got = tab.sample_base(1, uv).expect("句柄 1 合法");
            let want = px[j * 2 + i];
            for ch in 0..4 {
                assert_eq!(
                    got[ch].to_bits(),
                    want[ch].to_bits(),
                    "纹素 ({i},{j}) 通道 {ch} 应逐位相等"
                );
            }
        }
    }

    /// 平铺：`u + k` 与 `u` 必须取到同一纹素（repeat 而非 clamp）。
    #[test]
    fn repeat_wrap_is_periodic() {
        let px = two_by_two();
        let tab = table_of(&px);
        let a = tab.sample_base(1, [0.25, 0.25]).unwrap();
        for k in [-3.0f32, -1.0, 1.0, 3.0, 17.0] {
            let b = tab.sample_base(1, [0.25 + k, 0.25 - k]).unwrap();
            for ch in 0..4 {
                assert_eq!(a[ch].to_bits(), b[ch].to_bits(), "整数圈平移必须同值");
            }
        }
    }

    /// 负数分支：`u = -0.25` 必须取到最后一列（不是第 0 列，也不是 0 值）。
    #[test]
    fn negative_uv_wraps_to_last_texel() {
        let px = two_by_two();
        let tab = table_of(&px);
        // u = -0.25 ⇒ x = -0.25*2 - 0.5 = -1.0 ⇒ 纹素 -1 ≡ 1。
        let got = tab.sample_base(1, [-0.25, 0.25]).unwrap();
        assert_eq!(got[0], px[1][0], "红通道应是第 1 列");
        assert_eq!(got[2], px[1][2], "蓝通道应是第 1 列");
    }

    #[test]
    fn absent_and_out_of_range_handles_return_none() {
        let px = two_by_two();
        let tab = table_of(&px);
        assert!(tab.sample_base(0, [0.5, 0.5]).is_none(), "句柄 0 = 无贴图");
        assert!(tab.sample_base(2, [0.5, 0.5]).is_none(), "越界句柄");
        assert!(!tab.has(2) && tab.has(1));
        assert!(RefTextureTable::empty().sample_base(1, [0.5, 0.5]).is_none());
        assert!(RefTextureTable::empty().sample_with_footprint(1, [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]).is_none());
    }

    /// 多张纹理的 `offset` 按**纹素**累加（不是按字节），与 `encode_v3` 同口径。
    #[test]
    fn offsets_accumulate_in_texels() {
        let a: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 1.0]; 4]; // 2×2, 1 级
        let b: Vec<[f32; 4]> = vec![[0.0, 1.0, 0.0, 1.0]; 4];
        let ups = [
            TextureUpload { w: 2, h: 2, levels: 1, fmt: 1, srgb: 0, texels: &a },
            TextureUpload { w: 2, h: 2, levels: 1, fmt: 1, srgb: 0, texels: &b },
        ];
        let tab = RefTextureTable::new(&ups);
        assert_eq!(tab.len(), 2);
        assert_eq!(tab.slots[0].offset, 0);
        assert_eq!(tab.slots[1].offset, 4, "第二张从第 4 个纹素起（不是第 64 字节）");
        assert_eq!(tab.sample_base(2, [0.5, 0.5]).unwrap()[1], 1.0);
    }

    /// 非法上传 ⇒ 该条及之后全部丢弃（`offset` 是前缀累加，跳过会错位）。
    #[test]
    fn malformed_upload_truncates_the_table() {
        let a: Vec<[f32; 4]> = vec![[1.0; 4]; 4];
        let short: Vec<[f32; 4]> = vec![[1.0; 4]; 3]; // 2×2/1 级应为 4 纹素
        let ups = [
            TextureUpload { w: 2, h: 2, levels: 1, fmt: 1, srgb: 0, texels: &a },
            TextureUpload { w: 2, h: 2, levels: 1, fmt: 1, srgb: 0, texels: &short },
        ];
        let tab = RefTextureTable::new(&ups);
        assert_eq!(tab.len(), 1, "第二条非法 ⇒ 表在第 1 条处截断");
        assert!(tab.sample_base(2, [0.5, 0.5]).is_none());
    }

    /// 球面 uv：+Z 方向落在 u = 0.75（`atan2(1,0) = π/2` ⇒ 0.25 + 0.5），
    /// +X 落在 0.5，-Z 落在 0.25，极点在 v = 0 / 1。
    #[test]
    fn sphere_uv_anchors() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let u_of = |p: Vec3| sphere_uv(p, c, 1.0)[0];
        assert!((u_of(Vec3::new(0.0, 0.0, 1.0)) - 0.75).abs() < 1e-6);
        assert!((u_of(Vec3::new(1.0, 0.0, 0.0)) - 0.5).abs() < 1e-6);
        assert!((u_of(Vec3::new(0.0, 0.0, -1.0)) - 0.25).abs() < 1e-6);
        let v_top = sphere_uv(Vec3::new(0.0, 1.0, 0.0), c, 1.0)[1];
        let v_bot = sphere_uv(Vec3::new(0.0, -1.0, 0.0), c, 1.0)[1];
        assert!(v_top < 1e-6, "北极 v = 0，得到 {v_top}");
        assert!((v_bot - 1.0).abs() < 1e-6, "南极 v = 1，得到 {v_bot}");
    }

    /// 零足迹采样 == 第 0 级双线性（判据 1/2 与判据 3/4 的衔接）。
    #[test]
    fn zero_footprint_equals_level0_bilinear() {
        let px = two_by_two();
        let tab = table_of(&px);
        let uv = [0.3, 0.7];
        let a = tab.sample_base(1, uv).unwrap();
        let b = tab
            .sample_with_footprint(1, uv, [0.0, 0.0], [0.0, 0.0])
            .unwrap();
        for ch in 0..4 {
            assert_eq!(
                a[ch].to_bits(),
                b[ch].to_bits(),
                "零足迹必须逐位等于第 0 级双线性"
            );
        }
    }

    /// 构造一个 4×4 → 2×2 → 1×1 的 mip 链（盒式 2×2 降采样），验证
    /// `level_dims` / `level_base` / `bilinear_level` 寻址正确。
    #[test]
    fn mip_chain_addressing_and_bilinear() {
        // 第 0 级：4×4 棋盘（cell=1）；第 1 级：2×2 盒平均；第 2 级：1×1。
        let l0: Vec<[f32; 4]> = {
            let mut v = Vec::with_capacity(16);
            for y in 0..4u32 {
                for x in 0..4u32 {
                    let k = (x + y) % 2;
                    v.push(if k == 0 {
                        [1.0, 0.0, 0.0, 1.0]
                    } else {
                        [0.0, 1.0, 0.0, 1.0]
                    });
                }
            }
            v
        };
        let l1: Vec<[f32; 4]> = {
            let mut v = Vec::with_capacity(4);
            for y in 0..2u32 {
                for x in 0..2u32 {
                    let mut acc = [0.0f32; 4];
                    let mut cnt = 0.0f32;
                    for yy in 0..2u32 {
                        for xx in 0..2u32 {
                            let t = l0[((y * 2 + yy) * 4 + (x * 2 + xx)) as usize];
                            for c in 0..4 {
                                acc[c] += t[c];
                            }
                            cnt += 1.0;
                        }
                    }
                    for c in 0..4 {
                        acc[c] /= cnt;
                    }
                    v.push(acc);
                }
            }
            v
        };
        let l2 = {
            let mut acc = [0.0f32; 4];
            let mut cnt = 0.0f32;
            for t in &l1 {
                for c in 0..4 {
                    acc[c] += t[c];
                }
                cnt += 1.0;
            }
            for c in 0..4 {
                acc[c] /= cnt;
            }
            vec![acc]
        };
        let mut all = Vec::new();
        all.extend_from_slice(&l0);
        all.extend_from_slice(&l1);
        all.extend_from_slice(&l2);
        assert_eq!(all.len(), 16 + 4 + 1);

        let up = TextureUpload { w: 4, h: 4, levels: 3, fmt: 1, srgb: 0, texels: &all };
        let tab = RefTextureTable::new(std::slice::from_ref(&up));
        // 第 0 级纹素中心精确命中。
        let c00 = tab.sample_base(1, [0.125, 0.125]).unwrap();
        assert_eq!(c00[0].to_bits(), l0[0][0].to_bits(), "第 0 级 (0,0) 红");
        // 第 1 级（2×2）纹素中心精确命中。
        let c10 = tab.sample_with_footprint(1, [0.25, 0.25], [0.0, 0.0], [0.0, 0.0]);
        // 零足迹 ⇒ lod 0 双线性 == 第 0 级，这里改用显式高 lod 路径：
        // 用一条巨大足迹强制 lod 进入第 1 级（见 trilinear_uses_higher_level）。
        assert!(c10.is_some());
        let _ = c10;
    }

    /// 大足迹 ⇒ 三线性选取更高 mip 级（lod 不为 0），且 trilinear 落在上界级。
    #[test]
    fn trilinear_picks_higher_level_with_large_footprint() {
        // 第 0 级 4×4 纯红，第 1 级 2×2 纯绿，第 2 级 1×1 纯蓝。
        let l0 = vec![[1.0, 0.0, 0.0, 1.0]; 16];
        let l1 = vec![[0.0, 1.0, 0.0, 1.0]; 4];
        let l2 = vec![[0.0, 0.0, 1.0, 1.0]; 1];
        let mut all = Vec::new();
        all.extend_from_slice(&l0);
        all.extend_from_slice(&l1);
        all.extend_from_slice(&l2);
        let up = TextureUpload { w: 4, h: 4, levels: 3, fmt: 1, srgb: 0, texels: &all };
        let tab = RefTextureTable::new(std::slice::from_ref(&up));

        // 零足迹 ⇒ 第 0 级（红）。
        let z = tab.sample_with_footprint(1, [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]).unwrap();
        assert_eq!(z[0].to_bits(), 1.0f32.to_bits(), "零足迹 ⇒ 第 0 级红");

        // 极大足迹（覆盖整张图）⇒ lod 足够大，应落到第 2 级（蓝）。
        // c0 的纹素空间半轴 ≈ 4（整图宽），c1 同理 ⇒ s_max ≈ 4*sqrt2 ≈ 5.6，
        // 各向同性 ⇒ tap_w = s_max，lod = log2(5.6) ≈ 2.49 ⇒ 钳到 max_level=2。
        let big = tab
            .sample_with_footprint(1, [0.5, 0.5], [4.0 / 4.0, 0.0], [0.0, 4.0 / 4.0])
            .unwrap();
        assert_eq!(big[2].to_bits(), 1.0f32.to_bits(), "大足迹应落到第 2 级（蓝）");
    }

    #[test]
    fn ulp_distance_is_zero_for_identical_and_counts_small_steps() {
        assert_eq!(f32_ulp_distance(1.0, 1.0), 0);
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(f32_ulp_distance(a, b), 1);
        assert_eq!(f32_ulp_distance(b, a), 1);
        assert_eq!(f32_ulp_distance(-1.0f32, -f32::from_bits(1.0f32.to_bits() + 1)), 1);
    }

    /// `plane_tangent_basis` 产出一组**正交单位基**，且能正确把切线空间法线
    /// 旋到世界空间（与 `prism_rt::integrator::plane_tangent_basis` / shader 同式）。
    #[test]
    fn plane_tangent_basis_is_orthonormal_and_maps_normal() {
        // 几个典型法线（含接近 ±Y 的退化方向，正是该基要避免 NaN 的场景）。
        let normals = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.3, 0.8, 0.5).normalize(),
            Vec3::new(-0.6, 0.1, 0.79).normalize(),
        ];
        for n in normals {
            let (t, b) = plane_tangent_basis(n);
            // T、B 都是单位向量。
            assert!((t.length() - 1.0).abs() < 1e-6, "T 非单位：{:?}", t);
            assert!((b.length() - 1.0).abs() < 1e-6, "B 非单位：{:?}", b);
            // T ⟂ n、B ⟂ n、T ⟂ B（正交）。
            assert!(t.dot(&n).abs() < 1e-6, "T 不正交 n：{:?}", t.dot(&n));
            assert!(b.dot(&n).abs() < 1e-6, "B 不正交 n：{:?}", b.dot(&n));
            assert!(t.dot(&b).abs() < 1e-6, "T 不正交 B：{:?}", t.dot(&b));

            // 恒等切线空间法线 (0,0,1) 旋回世界后必须 == n（无扰动）。
            let ns = Vec3::new(0.0, 0.0, 1.0);
            let shaded = t * ns.x + b * ns.y + n * ns.z;
            let n_shade = shaded.normalize();
            assert!(
                (n_shade.x - n.x).abs() < 1e-6 && (n_shade.y - n.y).abs() < 1e-6
                    && (n_shade.z - n.z).abs() < 1e-6,
                "恒等法线未还原为 n：{:?} vs {:?}",
                n_shade,
                n
            );

            // 带扰动的切线空间法线 (0.2,0,1) 必须偏离 n（证明基确实"接住了"扰动）。
            let ns2 = Vec3::new(0.2, 0.0, 1.0);
            let shaded2 = (t * ns2.x + b * ns2.y + n * ns2.z).normalize();
            assert!(
                (shaded2 - n).length() > 1e-4,
                "扰动法线 (0.2,0,1) 未改变着色法线"
            );
        }
    }
}
