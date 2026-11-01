//! CPU 参考侧的纹理采样（W1-C 段 3）：**独立实现**，用于与 GPU 对拍。
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
//! * 足迹恒 **0** ⇒ `lod = 0`、`tap = 1`（mip 链只用到第 0 级）；
//! * 索引越界返回 0 而不是 panic（宪章 C18）。
//!
//! 依赖方向也决定了它**只能**在这里：`prism-rt` 依赖 `prism-gpu`（单向），
//! `prism-gpu` 不得反引 `prism-rt`。
//!
//! # 与 shader 的逐式对应
//!
//! `shaders/scene_frame_v3.wgsl` 的 `tex_bilinear` / `wrap_i` / `uv_sphere` 与
//! 本模块的同名件**逐表达式同构**（同结合序、同常量、同 `floor`）。
//! `vulkan/v3_frame.rs` 的对拍测试用真实像素证明这一点，而不是靠"看起来一样"。
//!
//! # 未做（**不要当成已支持**）
//!
//! * 三线性 / 各向异性：本版足迹恒 0，只做双线性；
//! * sRGB 解码：纹素进 [`crate::scene_abi::TextureUpload`] 前已是线性域
//!   （解码在 `prism_rt::texture::build_rgba8` 一次性完成）；
//! * 材质其余三张图（metal-rough / normal / emissive）：本版不读（着色模型
//!   里没有落点，见 `scene_frame_v3.wgsl` 入口处的说明）。

use crate::scene_abi::{TextureUpload, MAX_V3_TEXTURES};
use prism_math::Vec3;

/// `1/π`（与 `scene_frame_v3.wgsl` 的同名常量逐位同值）。
const FRAC_1_PI: f32 = 0.3183098861837907;
/// `1/(2π)`（同上）。
const FRAC_1_TAU: f32 = 0.15915494309189535;

/// 一张参考侧纹理：元数据 + 扁平纹素（含 mip 链，`offset` 以**纹素**计）。
#[derive(Debug, Clone, Copy)]
struct RefTexture<'a> {
    /// 宽度（纹素，≥ 1）。
    w: u32,
    /// 高度（纹素，≥ 1）。
    h: u32,
    /// 第 0 级在扁平纹素流里的起点（纹素下标）。
    offset: usize,
    /// 扁平纹素（线性域）。
    texels: &'a [[f32; 4]],
}

/// 参考侧纹理表（句柄 `k` ↔ 下标 `k-1`，`0` = 无贴图）。
///
/// 构造时按 [`TextureUpload`] 的顺序累加 `offset`，口径与
/// [`crate::scene_abi::SceneAbi::encode_v3`] 的 texel 流**完全一致**
/// （都是"先 `expected_texels()` 再前移"，不是按字节）。
#[derive(Debug, Clone, Default)]
pub struct RefTextureTable<'a> {
    /// 已接纳的纹理（下标 = 句柄 − 1）。
    slots: Vec<RefTexture<'a>>,
}

impl<'a> RefTextureTable<'a> {
    /// 由上传描述建表。
    ///
    /// 校验不过（尺寸 0 / 级数越界 / 纹素条数与描述不符）的纹理**及其之后
    /// 的全部条目**被丢弃：`offset` 是前缀累加，跳过一张会让后续句柄指向
    /// 错误的纹素——宁可少采样，不可错采样。句柄越界由调用点按"无贴图"处理。
    pub fn new(uploads: &[TextureUpload<'a>]) -> Self {
        let mut slots: Vec<RefTexture<'a>> = Vec::new();
        let mut cursor: usize = 0;
        for t in uploads.iter().take(MAX_V3_TEXTURES) {
            if t.validate().is_err() {
                break;
            }
            slots.push(RefTexture {
                w: t.w,
                h: t.h,
                offset: cursor,
                texels: t.texels,
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
    pub fn sample_base(&self, handle: u32, uv: [f32; 2]) -> Option<[f32; 4]> {
        if !self.has(handle) {
            return None;
        }
        let t = &self.slots[handle as usize - 1];
        Some(bilinear_level0(t, uv))
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

/// 单级（第 0 级）双线性 + repeat 平铺。
///
/// # 结合序（必须与 shader 一致）
///
/// 先沿 u：`a = t00 + (t10 − t00)·fx`、`b = t01 + (t11 − t01)·fx`；
/// 再沿 v：`out = a + (b − a)·fy`。换一种写法（比如先 v 后 u）在数学上等价、
/// 在 f32 上**不等价**——末位会差，对拍就不再是逐字节。
fn bilinear_level0(t: &RefTexture<'_>, uv: [f32; 2]) -> [f32; 4] {
    let w = t.w.max(1) as i64;
    let h = t.h.max(1) as i64;
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
    let i0 = wrap_i(x0, w);
    let i1 = wrap_i(x0 + 1, w);
    let j0 = wrap_i(y0, h);
    let j1 = wrap_i(y0 + 1, h);
    let at = |i: usize, j: usize| -> [f32; 4] {
        // 索引**相对本张纹理的切片**：第 0 级就是切片开头（mip 链按级序递增，
        // 第 0 级恒在最前）。`t.offset` 是"这条纹理在扁平静态流的全局起点"，
        // 只用于与 `texmeta` 的 `offset` 字段对账，不参与切片寻址——切片本身
        // 已经是分开的，再加全局偏移就会读到下一张图的像素。
        let idx = j * t.w.max(1) as usize + i;
        // 越界 = `texels` 与描述对不上（建表期已拦，这里是纵深防御）。
        t.texels.get(idx).copied().unwrap_or([0.0; 4])
    };
    let t00 = at(i0, j0);
    let t10 = at(i1, j0);
    let t01 = at(i0, j1);
    let t11 = at(i1, j1);
    let mut out = [0.0f32; 4];
    for ch in 0..4 {
        let a = t00[ch] + (t10[ch] - t00[ch]) * fx;
        let b = t01[ch] + (t11[ch] - t01[ch]) * fx;
        out[ch] = a + (b - a) * fy;
    }
    out
}

/// 球面参数坐标（`prism_rt::integrator::Scene::uv_at` 的 Sphere 臂逐式移植，
/// 也是 `scene_frame_v3.wgsl::uv_sphere` 的逐式同构件）。
///
/// `u = atan2(d.z, d.x)/2π + ½`（经度）、`v = acos(d.y)/π`（纬度，`+Y` 端为 0）。
/// 反三角走 [`prism_math::det_trig`]（禁 libm，D-29），与 shader 侧同一套
/// 系数与归约——这是"CPU 与 GPU 采到同一张贴图的同一个位置"的前提。
#[inline]
pub fn sphere_uv(p: Vec3, center: Vec3, radius: f32) -> [f32; 2] {
    let d = Vec3::new(
        (p.x - center.x) / radius,
        (p.y - center.y) / radius,
        (p.z - center.z) / radius,
    );
    let u = prism_math::det_trig::atan2(d.z, d.x) * FRAC_1_TAU + 0.5;
    let v = prism_math::det_trig::acos(d.y.clamp(-1.0, 1.0)) * FRAC_1_PI;
    [u, v]
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

    fn table_of(px: &[[f32; 4]]) -> RefTextureTable<'_> {
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

    #[test]
    fn ulp_distance_is_zero_for_identical_and_counts_small_steps() {
        assert_eq!(f32_ulp_distance(1.0, 1.0), 0);
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(f32_ulp_distance(a, b), 1);
        assert_eq!(f32_ulp_distance(b, a), 1);
        assert_eq!(f32_ulp_distance(-1.0f32, -f32::from_bits(1.0f32.to_bits() + 1)), 1);
    }
}
