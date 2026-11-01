//! 位图纹理引用（W1-A 定桩；采样/mip/sRGB 解码**不在本段**）。
//!
//! # 本模块只有布局
//!
//! 路线图 W1 的 A1 位图纹理管线（mip 链 + 三线性 + 解析射线微分足迹 +
//! sRGB 解码 LUT）是 **W1-B** 的事。本模块在 W0-A 只落地**结构定桩**：
//! [`TextureRef`] 的字段、槽宽、句柄语义与编译期断言。**一行采样代码都没有**，
//! 也**没有** 256 项 sRGB LUT（那是 W1-B 的产物）。
//!
//! 定桩的唯一目的：让 W1-B（纹理）与 W3（实例化）能在同一份
//! `Scene` 布局上并行开工，而不必各自再改一次结构（路线图 §4.2「W0-A 不产出
//! 任何用户可见能力」）。
//!
//! # 类型纪律（V2.2 `vec3<f32>` 事故的规避写法）
//!
//! 本结构**只含** `u32` / `[u32; N]`。三元组一律用 `[u32; 3]` 或
//! `array<vec4<u32>>` 承载，**禁** `vec3<f32>` —— 真机驱动对 `vec3` 的
//! `size` 处理与 WGSL 规范不一致（规范 12，实现更大），而
//! `array<vec3<f32>>` 的 stride 直接由该 size 决定，整个结构体的后续字段
//! 偏移会连锁错位。门禁见 [`crate::layout`]。
//!
//! # 32 B / 条的理由
//!
//! 语义字段 6 个（`w/h/levels/offset/fmt/srgb` = 24 B），但 GPU 侧
//! `texmeta` 绑定是 `array<vec4<u32>>`：WGSL 要求 `array<E>` 的 stride 是
//! `E` 对齐（16 B）的整数倍，24 B **不满足**。故补 2 个 `reserved` 字凑
//! 32 B（= 2 × `vec4<u32>`）。`reserved` 不承载语义、恒 0，只保证 stride。
//! 这是一处对选型报告 §11 字面描述的**补全**（§11 未写 `reserved`，但其
//! 「32 B/条」与 `array<vec4<f32>>` 两条同时成立只有加 pad 才可能）。

/// 单条纹理引用的字节数（GPU `array<vec4<u32>>` 的 stride）。
pub const TEXTURE_REF_BYTES: usize = 32;

/// 单条纹理引用的 u32 字数（= 2 个 `vec4<u32>`）。
pub const TEXTURE_REF_WORDS: u32 = 8;

/// 单个纹素的字节数（`array<vec4<f32>>` 里的一个元素）。
///
/// 纹理存储**恒以 `vec4<f32>` 为单元**（哪怕源是 RGBA8）：这样
/// `offset` 的口径与 GPU 侧寻址单位无条件一致，且天然满足 16 B 对齐。
pub const TEXEL_BYTES: usize = 16;

/// 句柄 0 的语义：**无贴图**。
///
/// [`crate::bsdf::Bsdf::maps`] 与 [`crate::light::Light::Environment`]`{ map }`
/// 共用此约定。0 是"无"，不是"第 0 张贴图"——贴图句柄从 1 起。
/// （`Scene::textures` 的下标口径由 W1-B 定义；本段只钉 0 的语义，
/// 因为 `Default` / 零初始化的语义正确性依赖它。）
pub const NO_TEXTURE: u32 = 0;

/// 句柄 → [`Scene::textures`](crate::integrator::Scene::textures) 下标。
///
/// **句柄从 1 起**：句柄 `k ≥ 1` 映射到 `textures[k-1]`；句柄 `0` 是
/// [`NO_TEXTURE`]（无），返回 `None`。这条映射在此单点定义，
/// [`crate::integrator::Scene::validate_layout`] 据此判越界，W1-B 的编码器
/// 据此查表——两处不能各写一份（0-based / 1-based 混用是那种"看起来能跑、
/// 只是少了一张贴图"的错）。
pub const fn handle_index(handle: u32) -> Option<usize> {
    if handle == NO_TEXTURE {
        None
    } else {
        Some((handle - 1) as usize)
    }
}

/// 句柄在当前纹理条数下是否可用（`0` 恒可用；`k` 需 `k <= n_textures`）。
pub const fn handle_in_range(handle: u32, n_textures: u32) -> bool {
    handle == NO_TEXTURE || handle <= n_textures
}

/// 像素格式：无（占位；合法引用不得取此值）。
pub const TEX_FMT_NONE: u32 = 0;
/// 像素格式：RGBA8 线性（sRGB 位由 [`TextureRef::srgb`] 另标）。
pub const TEX_FMT_RGBA8: u32 = 1;
/// 像素格式：RGBA16F 线性（HDR 环境图 / A2）。
pub const TEX_FMT_RGBA16F: u32 = 2;

/// [`TextureRef::srgb`]：数据已是线性域，**不做**解码。
pub const SRGB_LINEAR: u32 = 0;
/// [`TextureRef::srgb`]：数据是 sRGB 编码域，采样后须解码（W1-B 的 LUT）。
pub const SRGB_ENCODED: u32 = 1;

/// 位图纹理引用（`Scene::textures` 的元素；**只描述，不含像素**）。
///
/// 像素本体在 GPU 侧是一块扁平的 `array<vec4<f32>>`；本结构只给出该纹理
/// 在这块扁平存储里的位置与形状。三者（CPU 侧索引 / GPU 侧寻址 / 断言）
/// 由 `Scene::textures` 单点派生，禁止各后端各算一遍。
///
/// # 字段
///
/// 字段名与选型报告 §11 逐字一致（`w`/`h` 刻意不写成 `width`/`height`，
/// 便于与文档对账）。
/// # `repr(C)` 是**承重**的，不是装饰
///
/// 首版草稿漏了它，于是 `size_of == 32` 照样绿，而 `offset_of!(…, w) == 0`
/// 直接炸——Rust 对非 `repr(C)` 结构体**可以重排字段**。对本结构体来说，
/// 重排意味着 `h` 跑到 `w` 前面：单测的尺寸全过，真机的 texmeta 却按
/// WGSL 声明的顺序读，读出来是错的形状。故此处钉 `repr(C)`，
/// 与三个后端的声明顺序逐格对齐。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRef {
    /// 宽度（纹素）。
    pub w: u32,
    /// 高度（纹素）。
    pub h: u32,
    /// mip 级数（1 = 无 mip；`W1-B` 生成的链为 `1 + floor(log2(max(w,h)))`）。
    pub levels: u32,
    /// 在扁平纹理存储中的**起始纹素下标**（单位 texel，非字节；每 texel
    /// [`TEXEL_BYTES`] 字节）。
    pub offset: u32,
    /// 像素格式（[`TEX_FMT_RGBA8`] / [`TEX_FMT_RGBA16F`]）。
    pub fmt: u32,
    /// 传输曲线标记（[`SRGB_LINEAR`] / [`SRGB_ENCODED`]）。
    pub srgb: u32,
    /// 保留字（恒 0）：只为把步长撑到 `array<vec4<u32>>` 要求的 16 B 倍数。
    /// **不承载语义**；改它等于改 GPU ABI。
    pub reserved: [u32; 2],
}

impl TextureRef {
    /// 构造一条引用（`reserved` 恒 0）。
    pub const fn new(w: u32, h: u32, levels: u32, offset: u32, fmt: u32, srgb: u32) -> Self {
        TextureRef { w, h, levels, offset, fmt, srgb, reserved: [0, 0] }
    }

    /// 该纹理占用的纹素数（含 mip 链；等比 1/4 递减，末级 1 纹素）。
    ///
    /// `levels == 0` 按 1 级计（防御：不让 0 级退化成空纹理）。
    pub const fn texel_count(&self) -> usize {
        let levels = if self.levels == 0 { 1 } else { self.levels };
        let mut total: usize = 0;
        let mut lv: u32 = 0;
        let mut w = if self.w == 0 { 1 } else { self.w as usize };
        let mut h = if self.h == 0 { 1 } else { self.h as usize };
        while lv < levels {
            total += w * h;
            w = if w > 1 { w / 2 } else { 1 };
            h = if h > 1 { h / 2 } else { 1 };
            lv += 1;
        }
        total
    }

    /// 该纹理（含 mip 链）占用的字节数。
    pub const fn byte_len(&self) -> usize {
        self.texel_count() * TEXEL_BYTES
    }

    /// 该纹理若作为**等距柱状环境图**使用时的 2D CDF 字节数。
    ///
    /// 恒 `(W+1)·H·4`：条件 CDF 逐行 `W+1` 个 u32（前缀和比输入多一格），
    /// 边际 CDF `H` 个 u32 另外存放（4 KiB @ 1024 行）。
    /// 与选型报告 §3.1 第 2 条的 2048×1024 → 8,392,704 B 同式。
    pub const fn cdf_len_bytes(&self) -> usize {
        cdf_len_bytes(self.w, self.h)
    }
}

/// 2D CDF 的字节数：`(W+1)·H·4`。
///
/// 这是 A2（HDR 环境光照）CDF 缓冲的**唯一**长度口径；W0-A 先把它钉成
/// 编译期函数，W2 直接调用，禁止各处手写 `(w+1)*h*4`。
pub const fn cdf_len_bytes(w: u32, h: u32) -> usize {
    ((w as usize) + 1) * (h as usize) * 4
}

/// 运行时断言：给定缓冲长度必须是 `(W+1)·H·4`。
///
/// 编译期已能验的部分见本模块尾部的 `const _` 断言（公式 + 规格算例）；
/// 本函数是给**持有真实缓冲**的调用方（W2 的 `cdf` 绑定）用的运行时版本：
/// 只有拿到 `len_bytes` 才能判"这条 CDF 与它的 `TextureRef` 是否对得上"。
/// 没有 CDF 时（本段）不调用即可。
///
/// # Panics
///
/// 长度不符时 panic（与 `Bsdf::new` 的参数域断言同风格：错误参数是编程
/// 错误，不是用户输入）。
pub fn assert_cdf_len(w: u32, h: u32, len_bytes: usize) {
    let want = cdf_len_bytes(w, h);
    assert_eq!(
        len_bytes, want,
        "CDF 长度必须为 (W+1)·H·4 = ({w}+1)·{h}·4 = {want} 字节，收到 {len_bytes}；\
         偏差一格会让整条前缀和错位（行内二分找到的是邻居的台阶）"
    );
}

// ---------------------------------------------------------------------------
// W1-B：sRGB 传递曲线 —— 编译期 256 项 LUT（1 KiB）
// ---------------------------------------------------------------------------
//
// 口径（选型报告 §2.2）：解码走 `TransferCurve::Srgb`（IEC 61966-2-1 原曲线，
// 即 `colorspace_registry::TransferFunction::Srgb` 的 EOTF 方向），
// **编译期**生成 256 项 f32 表。
//
// 为什么是**常量表**而不是运行时算：
//   1. 三后端必须读到**同一份 f32 位型**——表是 `const`，它在 CPU 侧被
//      `include` 进 `.rodata`，在 GPU 侧由 `scene_abi` 的编码器位拷贝上传；
//      "各自按公式算"会因 `pow(2.4)` 的实现差异在末位分叉（§4.3 的最保守写法）。
//   2. 表值由 `TransferFunction::Srgb.decode` **生成**（不是手抄），
//      由 `srgb_lut_matches_registry_curve` 逐项对拍钉死——公式改动会让测试红，
//      而不是让表悄悄过期。
//
// 1 KiB = 256 × 4 B。表里的值就是 `decode(i/255)` 的 f32 位型。

/// sRGB 解码 LUT：`SRGB_DECODE_LUT[i] = srgb_eotf(i / 255)`（线性域 f32）。
///
/// 下标 = 8 位码值（0–255），值 = 线性辐亮度。表值由
/// [`prism_math::colorspace_registry::TransferFunction::Srgb`] 生成，
/// 见 `srgb_lut_matches_registry_curve`。
pub const SRGB_DECODE_LUT: [f32; 256] = [
    f32::from_bits(0x00000000), f32::from_bits(0x399f22b4), f32::from_bits(0x3a1f22b4), f32::from_bits(0x3a6eb40e),
    f32::from_bits(0x3a9f22b4), f32::from_bits(0x3ac6eb61), f32::from_bits(0x3aeeb40e), f32::from_bits(0x3b0b3e5d),
    f32::from_bits(0x3b1f22b4), f32::from_bits(0x3b33070a), f32::from_bits(0x3b46eb61), f32::from_bits(0x3b5b518a),
    f32::from_bits(0x3b70f18a), f32::from_bits(0x3b83e1c5), f32::from_bits(0x3b8fe614), f32::from_bits(0x3b9c87fb),
    f32::from_bits(0x3ba9c9b3), f32::from_bits(0x3bb7ad6d), f32::from_bits(0x3bc63547), f32::from_bits(0x3bd5635d),
    f32::from_bits(0x3be539bd), f32::from_bits(0x3bf5ba6b), f32::from_bits(0x3c0373b5), f32::from_bits(0x3c0c6152),
    f32::from_bits(0x3c15a703), f32::from_bits(0x3c1f45bc), f32::from_bits(0x3c293e68), f32::from_bits(0x3c3391f4),
    f32::from_bits(0x3c3e4149), f32::from_bits(0x3c494d43), f32::from_bits(0x3c54b6c7), f32::from_bits(0x3c607eb1),
    f32::from_bits(0x3c6ca5dc), f32::from_bits(0x3c792d1f), f32::from_bits(0x3c830aa7), f32::from_bits(0x3c89af9f),
    f32::from_bits(0x3c9085db), f32::from_bits(0x3c978dc5), f32::from_bits(0x3c9ec7c0), f32::from_bits(0x3ca63431),
    f32::from_bits(0x3cadd37b), f32::from_bits(0x3cb5a5ff), f32::from_bits(0x3cbdac20), f32::from_bits(0x3cc5e639),
    f32::from_bits(0x3cce54ab), f32::from_bits(0x3cd6f7d3), f32::from_bits(0x3cdfd00e), f32::from_bits(0x3ce8ddb7),
    f32::from_bits(0x3cf22129), f32::from_bits(0x3cfb9ac1), f32::from_bits(0x3d02a569), f32::from_bits(0x3d0798dc),
    f32::from_bits(0x3d0ca7e4), f32::from_bits(0x3d11d2ae), f32::from_bits(0x3d171963), f32::from_bits(0x3d1c7c2e),
    f32::from_bits(0x3d21fb39), f32::from_bits(0x3d2796af), f32::from_bits(0x3d2d4eb8), f32::from_bits(0x3d332380),
    f32::from_bits(0x3d39152b), f32::from_bits(0x3d3f23e3), f32::from_bits(0x3d454fd1), f32::from_bits(0x3d4b991c),
    f32::from_bits(0x3d51ffeb), f32::from_bits(0x3d588466), f32::from_bits(0x3d5f26b4), f32::from_bits(0x3d65e6fb),
    f32::from_bits(0x3d6cc561), f32::from_bits(0x3d73c20c), f32::from_bits(0x3d7add21), f32::from_bits(0x3d810b64),
    f32::from_bits(0x3d84b791), f32::from_bits(0x3d88732e), f32::from_bits(0x3d8c3e48), f32::from_bits(0x3d9018f4),
    f32::from_bits(0x3d940343), f32::from_bits(0x3d97fd48), f32::from_bits(0x3d9c0714), f32::from_bits(0x3da020b9),
    f32::from_bits(0x3da44a49), f32::from_bits(0x3da883d5), f32::from_bits(0x3daccd6e), f32::from_bits(0x3db12726),
    f32::from_bits(0x3db5910d), f32::from_bits(0x3dba0b36), f32::from_bits(0x3dbe95b3), f32::from_bits(0x3dc3308f),
    f32::from_bits(0x3dc7dbe0), f32::from_bits(0x3dcc97b4), f32::from_bits(0x3dd1641c), f32::from_bits(0x3dd6412a),
    f32::from_bits(0x3ddb2eed), f32::from_bits(0x3de02d75), f32::from_bits(0x3de53cd3), f32::from_bits(0x3dea5d16),
    f32::from_bits(0x3def8e4f), f32::from_bits(0x3df4d08e), f32::from_bits(0x3dfa23e2), f32::from_bits(0x3dff885b),
    f32::from_bits(0x3e027f06), f32::from_bits(0x3e05427f), f32::from_bits(0x3e080ea2), f32::from_bits(0x3e0ae376),
    f32::from_bits(0x3e0dc104), f32::from_bits(0x3e10a752), f32::from_bits(0x3e139669), f32::from_bits(0x3e168e50),
    f32::from_bits(0x3e198f0e), f32::from_bits(0x3e1c98ab), f32::from_bits(0x3e1fab2f), f32::from_bits(0x3e22c6a0),
    f32::from_bits(0x3e25eb06), f32::from_bits(0x3e29186a), f32::from_bits(0x3e2c4ed0), f32::from_bits(0x3e2f8e41),
    f32::from_bits(0x3e32d6c4), f32::from_bits(0x3e362861), f32::from_bits(0x3e39831e), f32::from_bits(0x3e3ce703),
    f32::from_bits(0x3e405416), f32::from_bits(0x3e43ca5f), f32::from_bits(0x3e4749e4), f32::from_bits(0x3e4ad2ae),
    f32::from_bits(0x3e4e64c2), f32::from_bits(0x3e520027), f32::from_bits(0x3e55a4e6), f32::from_bits(0x3e595303),
    f32::from_bits(0x3e5d0a87), f32::from_bits(0x3e60cb78), f32::from_bits(0x3e6495dc), f32::from_bits(0x3e6869bb),
    f32::from_bits(0x3e6c471c), f32::from_bits(0x3e702e04), f32::from_bits(0x3e741e7f), f32::from_bits(0x3e78188c),
    f32::from_bits(0x3e7c1c34), f32::from_bits(0x3e8014bf), f32::from_bits(0x3e822039), f32::from_bits(0x3e84308b),
    f32::from_bits(0x3e8645b8), f32::from_bits(0x3e885fc3), f32::from_bits(0x3e8a7eb0), f32::from_bits(0x3e8ca281),
    f32::from_bits(0x3e8ecb3a), f32::from_bits(0x3e90f8df), f32::from_bits(0x3e932b72), f32::from_bits(0x3e9562f6),
    f32::from_bits(0x3e979f6f), f32::from_bits(0x3e99e0e0), f32::from_bits(0x3e9c274b), f32::from_bits(0x3e9e72b5),
    f32::from_bits(0x3ea0c320), f32::from_bits(0x3ea3188f), f32::from_bits(0x3ea57306), f32::from_bits(0x3ea7d287),
    f32::from_bits(0x3eaa3715), f32::from_bits(0x3eaca0b4), f32::from_bits(0x3eaf0f67), f32::from_bits(0x3eb18330),
    f32::from_bits(0x3eb3fc13), f32::from_bits(0x3eb67a15), f32::from_bits(0x3eb8fd34), f32::from_bits(0x3ebb8576),
    f32::from_bits(0x3ebe12de), f32::from_bits(0x3ec0a56e), f32::from_bits(0x3ec33d2a), f32::from_bits(0x3ec5da14),
    f32::from_bits(0x3ec87c30), f32::from_bits(0x3ecb2380), f32::from_bits(0x3ecdd008), f32::from_bits(0x3ed081ca),
    f32::from_bits(0x3ed338c9), f32::from_bits(0x3ed5f508), f32::from_bits(0x3ed8b68a), f32::from_bits(0x3edb7d51),
    f32::from_bits(0x3ede4962), f32::from_bits(0x3ee11abe), f32::from_bits(0x3ee3f168), f32::from_bits(0x3ee6cd64),
    f32::from_bits(0x3ee9aeb3), f32::from_bits(0x3eec955a), f32::from_bits(0x3eef815a), f32::from_bits(0x3ef272b7),
    f32::from_bits(0x3ef56973), f32::from_bits(0x3ef86591), f32::from_bits(0x3efb6714), f32::from_bits(0x3efe6dfe),
    f32::from_bits(0x3f00bd2b), f32::from_bits(0x3f02460c), f32::from_bits(0x3f03d1a5), f32::from_bits(0x3f055ff8),
    f32::from_bits(0x3f06f105), f32::from_bits(0x3f0884ce), f32::from_bits(0x3f0a1b54), f32::from_bits(0x3f0bb499),
    f32::from_bits(0x3f0d509f), f32::from_bits(0x3f0eef65), f32::from_bits(0x3f1090ef), f32::from_bits(0x3f12353d),
    f32::from_bits(0x3f13dc50), f32::from_bits(0x3f15862a), f32::from_bits(0x3f1732cc), f32::from_bits(0x3f18e237),
    f32::from_bits(0x3f1a946d), f32::from_bits(0x3f1c4970), f32::from_bits(0x3f1e013f), f32::from_bits(0x3f1fbbde),
    f32::from_bits(0x3f21794c), f32::from_bits(0x3f23398c), f32::from_bits(0x3f24fc9e), f32::from_bits(0x3f26c284),
    f32::from_bits(0x3f288b40), f32::from_bits(0x3f2a56d1), f32::from_bits(0x3f2c253b), f32::from_bits(0x3f2df680),
    f32::from_bits(0x3f2fca9c), f32::from_bits(0x3f31a195), f32::from_bits(0x3f337b6a), f32::from_bits(0x3f35581e),
    f32::from_bits(0x3f3737b1), f32::from_bits(0x3f391a24), f32::from_bits(0x3f3aff7a), f32::from_bits(0x3f3ce7b3),
    f32::from_bits(0x3f3ed2d0), f32::from_bits(0x3f40c0d2), f32::from_bits(0x3f42b1bc), f32::from_bits(0x3f44a58e),
    f32::from_bits(0x3f469c49), f32::from_bits(0x3f4895ef), f32::from_bits(0x3f4a9280), f32::from_bits(0x3f4c91ff),
    f32::from_bits(0x3f4e946c), f32::from_bits(0x3f5099c8), f32::from_bits(0x3f52a216), f32::from_bits(0x3f54ad55),
    f32::from_bits(0x3f56bb87), f32::from_bits(0x3f58ccae), f32::from_bits(0x3f5ae0cb), f32::from_bits(0x3f5cf7de),
    f32::from_bits(0x3f5f11e9), f32::from_bits(0x3f612eee), f32::from_bits(0x3f634eef), f32::from_bits(0x3f6571e9),
    f32::from_bits(0x3f6797e1), f32::from_bits(0x3f69c0d6), f32::from_bits(0x3f6beccb), f32::from_bits(0x3f6e1bbf),
    f32::from_bits(0x3f704db6), f32::from_bits(0x3f7282af), f32::from_bits(0x3f74baac), f32::from_bits(0x3f76f5ae),
    f32::from_bits(0x3f7933b6), f32::from_bits(0x3f7b74c6), f32::from_bits(0x3f7db8de), f32::from_bits(0x3f800000),
];

/// sRGB 解码（8 位码值 → 线性 f32），走 [`SRGB_DECODE_LUT`]。
#[inline]
pub fn srgb_decode_u8(code: u8) -> f32 {
    SRGB_DECODE_LUT[code as usize]
}

/// sRGB 编码（线性 f32 → 8 位码值；向下取到最近的码值）。
///
/// # 往返恒等（V4 的一条）
///
/// 表严格单调递增（`srgb_lut_is_strictly_increasing` 钉住），故
/// `srgb_encode_u8(SRGB_DECODE_LUT[i]) == i` **逐位**成立：二分找到的
/// "最后一个 `<= x` 的下标"就是 `i` 本身。
///
/// 越界输入按端点钳制（负 → 0，≥1 → 255）。
#[inline]
pub fn srgb_encode_u8(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0; // 含 NaN / 负
    }
    if x >= SRGB_DECODE_LUT[255] {
        return 255;
    }
    // 二分：最后一个 LUT[i] <= x（表严格递增 ⇒ 结果唯一）。
    let mut lo: usize = 0;
    let mut hi: usize = 255;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if SRGB_DECODE_LUT[mid] <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo as u8
}

/// sRGB 编码（线性 f32 → `[0,1]` 连续编码值；表内线性插值）。
///
/// 与 [`srgb_encode_u8`] 的区别：不做 8 位量化，输出是连续的编码域值，
/// 量化误差 ≤ **0.5 LSB**（1 LSB = 1/255）。这正是 V4 第二条判据的口径。
/// 表值输入（`x == SRGB_DECODE_LUT[i]`）时插值系数为 0，返回 `i/255`
/// 的**精确** f32 值 ⇒ `encode_f32(decode_u8(i))` 与 `i as f32 / 255.0` 逐位相等。
#[inline]
pub fn srgb_encode_f32(x: f32) -> f32 {
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= SRGB_DECODE_LUT[255] {
        return 1.0;
    }
    let i = srgb_encode_u8(x) as usize;
    if i >= 255 {
        return 1.0;
    }
    let lo = SRGB_DECODE_LUT[i];
    let hi = SRGB_DECODE_LUT[i + 1];
    let frac = if hi > lo { (x - lo) / (hi - lo) } else { 0.0 };
    (i as f32 + frac.clamp(0.0, 1.0)) / 255.0
}

// ---------------------------------------------------------------------------
// W1-B：纹理构建（CPU 侧像素 → 扁平 `array<vec4<f32>>` + mip 链）
// ---------------------------------------------------------------------------
//
// 定稿偏离记录（选型报告 §2 的"LUT 上传"一句）：**解码在 build 期做**。
// `Scene::texels` 恒为**线性域 f32**；`TextureRef::srgb` 只是源数据出处标记，
// 采样器不读它。理由见 `docs/v2.5-W0-算法选型报告.md` §15。
//
// 这条选择让"三后端读到同一份 f32 位型"从"约定"变成"结构"：
// 上传的就是线性纹素本身，GPU 侧不需要再实现一遍 LUT 查表与量化。

/// 2×2 盒式降采样（mip 生成；固定求和顺序 ⇒ 确定性）。
///
/// 逐通道 `((a+b) + (c+d)) · 0.25`：括号固定，加法可交换，与向量化无关。
fn downsample_2x2(src: &[[f32; 4]], w: usize, h: usize) -> (Vec<[f32; 4]>, usize, usize) {
    let nw = if w > 1 { w / 2 } else { 1 };
    let nh = if h > 1 { h / 2 } else { 1 };
    let mut out = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            // 源像素：2×2 块（边界处重复取最后一个，保证任意尺寸都有定义）。
            let x0 = (x * 2).min(w - 1);
            let x1 = (x * 2 + 1).min(w - 1);
            let y0 = (y * 2).min(h - 1);
            let y1 = (y * 2 + 1).min(h - 1);
            let a = src[y0 * w + x0];
            let b = src[y0 * w + x1];
            let c = src[y1 * w + x0];
            let d = src[y1 * w + x1];
            let mut px = [0.0f32; 4];
            for ch in 0..4 {
                px[ch] = ((a[ch] + b[ch]) + (c[ch] + d[ch])) * 0.25;
            }
            out.push(px);
        }
    }
    (out, nw, nh)
}

/// mip 级数：`1 + floor(log2(max(w,h)))`（末级 1×1）。
pub fn mip_levels(w: u32, h: u32) -> u32 {
    let m = w.max(h).max(1);
    let mut lv = 1u32;
    let mut d = m;
    while d > 1 {
        d /= 2;
        lv += 1;
    }
    lv
}

/// 由 8 位 RGBA 源构建纹理：解码（可选 sRGB）→ 线性 f32 → mip 链。
///
/// 返回 `(TextureRef, 纹素流)`。`offset` 由调用方给出（`Scene::texels`
/// 的当前长度），纹素流按 mip 级顺序**紧接**排布。
///
/// # Panics
///
/// `pixels.len() != w*h*4` 时 panic——形状不符是编程错误，且越界读会让
/// 采样器读到下一张纹理的像素（比 panic 难查得多）。
pub fn build_rgba8(
    w: u32,
    h: u32,
    srgb: bool,
    offset: u32,
    pixels: &[u8],
) -> (TextureRef, Vec<[f32; 4]>) {
    assert!(
        w > 0 && h > 0,
        "纹理尺寸必须 > 0，收到 {w}×{h}；0 尺寸纹理没有可采样的纹素"
    );
    let n = (w as usize) * (h as usize);
    assert_eq!(
        pixels.len(),
        n * 4,
        "RGBA8 像素数必须等于 w·h·4 = {n}·4，收到 {}；短一寸就是越界读",
        pixels.len()
    );
    let curve = if srgb { SRGB_ENCODED } else { SRGB_LINEAR };
    let mut level0 = Vec::with_capacity(n);
    for i in 0..n {
        let mut px = [0.0f32; 4];
        for ch in 0..3 {
            let code = pixels[i * 4 + ch];
            px[ch] = if srgb { srgb_decode_u8(code) } else { code as f32 / 255.0 };
        }
        px[3] = pixels[i * 4 + 3] as f32 / 255.0; // alpha 恒线性（不以 sRGB 编码）
        level0.push(px);
    }
    let levels = mip_levels(w, h);
    let mut texels = level0;
    let (mut cw, mut ch_) = (w as usize, h as usize);
    while cw > 1 || ch_ > 1 {
        // 只对**当前末级**降采样（`texels` 前段是更粗的兄弟级）。
        let start = texels.len() - cw * ch_;
        let (next, nw, nh) = downsample_2x2(&texels[start..], cw, ch_);
        texels.extend_from_slice(&next);
        cw = nw;
        ch_ = nh;
    }
    (
        TextureRef::new(w, h, levels, offset, TEX_FMT_RGBA8, curve),
        texels,
    )
}

/// 由线性 f32 源构建纹理（RGBA16F / HDR 环境图 / 程序化贴图路径）。
///
/// `pixels` 逐纹素 `[r,g,b,a]`，已是线性域（`srgb == false` 才合法）。
/// 与 [`build_rgba8`] 同为"level0 + 盒式 mip 链"的紧接排布。
///
/// # Panics
///
/// 尺寸为 0、长度不符、或 `srgb == true`（线性源标 sRGB 会让解码二次生效）
/// 时 panic。
pub fn build_linear_f32(
    w: u32,
    h: u32,
    srgb: bool,
    offset: u32,
    pixels: &[[f32; 4]],
) -> (TextureRef, Vec<[f32; 4]>) {
    assert!(w > 0 && h > 0, "纹理尺寸必须 > 0，收到 {w}×{h}");
    assert_eq!(
        pixels.len(),
        (w as usize) * (h as usize),
        "线性源纹素数必须等于 w·h"
    );
    assert!(
        !srgb,
        "线性 f32 源不得标 SRGB_ENCODED：解码会对已是线性的值再做一次，结果偏暗"
    );
    let levels = mip_levels(w, h);
    let mut texels = pixels.to_vec();
    let (mut cw, mut ch_) = (w as usize, h as usize);
    while cw > 1 || ch_ > 1 {
        let start = texels.len() - cw * ch_;
        let (next, nw, nh) = downsample_2x2(&texels[start..], cw, ch_);
        texels.extend_from_slice(&next);
        cw = nw;
        ch_ = nh;
    }
    (
        TextureRef::new(w, h, levels, offset, TEX_FMT_RGBA16F, SRGB_LINEAR),
        texels,
    )
}

// ---------------------------------------------------------------------------
// 编译期断言（写错就编不过 —— 这是 W0-A 的全部产出之一）
// ---------------------------------------------------------------------------


/// 步长：32 B（`array<vec4<u32>>`，两格）。
const _: () = assert!(core::mem::size_of::<TextureRef>() == TEXTURE_REF_BYTES);
/// 对齐：4 B（全 `u32`，无浮点、无指针 ⇒ 跨 target 恒定）。
const _: () = assert!(core::mem::align_of::<TextureRef>() == 4);
/// 字数：8（= [`TEXTURE_REF_BYTES`] / 4）。
const _: () = assert!(TEXTURE_REF_BYTES == TEXTURE_REF_WORDS as usize * 4);
/// 逐字段偏移 —— 槽位错一格，GPU 侧读到的就是下一个字段。
const _: () = assert!(core::mem::offset_of!(TextureRef, w) == 0);
const _: () = assert!(core::mem::offset_of!(TextureRef, h) == 4);
const _: () = assert!(core::mem::offset_of!(TextureRef, levels) == 8);
const _: () = assert!(core::mem::offset_of!(TextureRef, offset) == 12);
const _: () = assert!(core::mem::offset_of!(TextureRef, fmt) == 16);
const _: () = assert!(core::mem::offset_of!(TextureRef, srgb) == 20);
const _: () = assert!(core::mem::offset_of!(TextureRef, reserved) == 24);
/// CDF 长度口径：2048×1024 → 8,392,704 B（选型报告 §3.1 第 2 条）。
const _: () = assert!(cdf_len_bytes(2048, 1024) == 8_392_704);
/// 退化：1×1 的 CDF 是 `(1+1)·1·4` = 8 B（不是 4 B —— 多的一格是前缀和末位）。
const _: () = assert!(cdf_len_bytes(1, 1) == 8);
/// 与「每行多一格」的构造一致：`H` 行各 `W+1` 个 u32。
const _: () = assert!(cdf_len_bytes(3, 5) == 4 * 5 * 4);
/// 句柄 0 = 无（不映射任何下标）；1 → 0 号元素。
const _: () = assert!(handle_index(NO_TEXTURE).is_none());
const _: () = assert!(matches!(handle_index(1), Some(0)));
/// 范围判定：0 恒可；`k` 需 ≤ 条数。
const _: () = assert!(handle_in_range(NO_TEXTURE, 0));
const _: () = assert!(!handle_in_range(1, 0));
const _: () = assert!(handle_in_range(1, 1));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_count_follows_mip_chain() {
        // 4×4 全链：16 + 4 + 1 = 21 纹素（1×1 后不再减半）。
        let t = TextureRef::new(4, 4, 3, 0, TEX_FMT_RGBA8, SRGB_LINEAR);
        assert_eq!(t.texel_count(), 21);
        assert_eq!(t.byte_len(), 21 * TEXEL_BYTES);
        // 无 mip：就是 w·h。
        let t1 = TextureRef::new(7, 3, 1, 0, TEX_FMT_RGBA8, SRGB_LINEAR);
        assert_eq!(t1.texel_count(), 21);
        // levels = 0 按 1 级计（防御）。
        let t0 = TextureRef::new(7, 3, 0, 0, TEX_FMT_RGBA8, SRGB_LINEAR);
        assert_eq!(t0.texel_count(), 21);
        // 2 的幂链：8×8 / 4 级 = 64+16+4+1 = 85。
        let t2 = TextureRef::new(8, 8, 4, 0, TEX_FMT_RGBA16F, SRGB_LINEAR);
        assert_eq!(t2.texel_count(), 85);
    }

    #[test]
    fn cdf_len_matches_spec_for_environment_map() {
        let env = TextureRef::new(2048, 1024, 1, 0, TEX_FMT_RGBA16F, SRGB_LINEAR);
        assert_eq!(env.cdf_len_bytes(), 8_392_704);
        // 运行时断言：长度对得上 ⇒ 不 panic。
        assert_cdf_len(2048, 1024, 8_392_704);
    }

    #[test]
    #[should_panic(expected = "CDF 长度必须为")]
    fn cdf_len_assertion_rejects_off_by_one_row() {
        // 少一行（1023 行）：整条前缀和会错位 —— 断言必须拦住。
        assert_cdf_len(2048, 1024, cdf_len_bytes(2048, 1023));
    }

    #[test]
    fn no_texture_is_zero_and_refs_are_32_bytes() {
        assert_eq!(NO_TEXTURE, 0);
        assert_eq!(core::mem::size_of::<TextureRef>(), 32);
        // reserved 恒 0：`Default` 语义正确的前提（0 = 无贴图，不是垃圾）。
        let d = TextureRef::new(0, 0, 0, 0, TEX_FMT_NONE, SRGB_LINEAR);
        assert_eq!(d.reserved, [0u32, 0u32]);
    }

    // -----------------------------------------------------------------
    // W1-B：sRGB LUT（V4 的两条判据）
    // -----------------------------------------------------------------

    /// **V4-a：256 值 decode→encode 恒等（逐位）**。
    ///
    /// 两条口径都测：8 位码值往返（`encode_u8`）与连续域往返
    /// （`encode_f32` 返回 `i/255` 的精确 f32 位型）。
    #[test]
    fn srgb_roundtrip_is_bitwise_identity_for_all_256_codes() {
        for i in 0..256usize {
            let lin = SRGB_DECODE_LUT[i];
            assert_eq!(
                srgb_encode_u8(lin),
                i as u8,
                "码值 {i}：decode→encode 不恒等（得到 {}）",
                srgb_encode_u8(lin)
            );
            let back = srgb_encode_f32(lin);
            let want = (i as f32) / 255.0;
            assert_eq!(
                back.to_bits(),
                want.to_bits(),
                "码值 {i}：连续域往返 {} ≠ {want}（位型不等）",
                back
            );
        }
        // 端点行为（越界钳制，不 panic 不 NaN）。
        assert_eq!(srgb_encode_u8(-1.0), 0);
        assert_eq!(srgb_encode_u8(f32::NAN), 0);
        println!("V4：256/256 码值 decode→encode 逐位恒等（LUT 1024 B）");
        assert_eq!(srgb_encode_u8(2.0), 255);
        assert_eq!(srgb_encode_f32(-1.0), 0.0);
        assert_eq!(srgb_encode_f32(2.0), 1.0);
    }

    /// **V4-b：任意 f32 编码误差 ≤ 0.5 LSB**（1 LSB = 1/255）。
    #[test]
    fn srgb_encode_error_is_within_half_lsb() {
        // 线性域密集扫掠（含分段断点附近 0.0031308 两侧）。
        let mut worst = 0.0f32;
        for k in 0..=20000 {
            let x = (k as f32) / 20000.0;
            let got = srgb_encode_f32(x);
            // 参考：把 got 换回码值空间后，与最近码值的中点在编码域的偏差。
            // 判据形式：|get - 真值| ≤ 0.5/255，真值用 registry 曲线算。
            let want = prism_math::colorspace_registry::TransferFunction::Srgb
                .encode(x as f64)
                .expect("sRGB 编码对 [0,1] 输入无定义域错误") as f32;
            let err = (got - want).abs();
            worst = worst.max(err);
            assert!(
                err <= 0.5 / 255.0 + 1e-7,
                "x={x}：编码值 {got} vs 参考 {want}，误差 {err} > 0.5 LSB"
            );
        }
        assert!(worst <= 0.5 / 255.0 + 1e-7, "最坏编码误差 {worst}");
    }

    /// 表值必须由 registry 的 sRGB 曲线**生成**（防公式改了而表过期）。
    #[test]
    fn srgb_lut_matches_registry_curve() {
        let tf = prism_math::colorspace_registry::TransferFunction::Srgb;
        for i in 0..256usize {
            let want = tf.decode(i as f64 / 255.0).expect("sRGB 解码无域错误") as f32;
            assert_eq!(
                SRGB_DECODE_LUT[i].to_bits(),
                want.to_bits(),
                "LUT[{i}] 与 TransferFunction::Srgb 的 decode 不一致：{} vs {want}",
                SRGB_DECODE_LUT[i]
            );
        }
        // 端点与断点值（IEC 61966-2-1 的公开算例）。
        assert_eq!(SRGB_DECODE_LUT[0], 0.0);
        assert_eq!(SRGB_DECODE_LUT[255], 1.0);
    }

    /// 表严格单调递增（`encode_u8` 的二分与往返恒等的前提）。
    #[test]
    fn srgb_lut_is_strictly_increasing() {
        for i in 1..256usize {
            assert!(
                SRGB_DECODE_LUT[i] > SRGB_DECODE_LUT[i - 1],
                "LUT 在 {i} 处非严格递增：{} ≤ {}",
                SRGB_DECODE_LUT[i],
                SRGB_DECODE_LUT[i - 1]
            );
        }
        // 1 KiB：256 × 4 B。
        assert_eq!(core::mem::size_of_val(&SRGB_DECODE_LUT), 1024);
    }

    // -----------------------------------------------------------------
    // W1-B：构建器（mip 链 / 布局）
    // -----------------------------------------------------------------

    #[test]
    fn build_rgba8_produces_full_mip_chain_with_expected_length() {
        // 4×4 全链：16 + 4 + 1 = 21 纹素。
        let px: Vec<u8> = (0..16 * 4).map(|i| (i % 256) as u8).collect();
        let (r, texels) = build_rgba8(4, 4, false, 0, &px);
        assert_eq!(r.w, 4);
        assert_eq!(r.h, 4);
        assert_eq!(r.levels, 3, "4×4 的 mip 级数 = 1 + floor(log2 4) = 3");
        assert_eq!(texels.len(), 21);
        assert_eq!(texels.len(), r.texel_count(), "与 TextureRef 的口径一致");
        // 末级 1×1 = 全图平均（盒式链的构造性结论）：4×4 的均值的均值。
        let sum: f32 = texels[..16].iter().map(|p| p[0]).sum();
        let mean = sum / 16.0;
        assert!((texels[20][0] - mean).abs() < 1e-4, "末级应 ≈ 全图均值");
        // 非 2 的幂：5×3 → 级数 1+floor(log2 5) = 3；尺寸 5×3 → 2×1 → 1×1。
        let px2: Vec<u8> = vec![128; 5 * 3 * 4];
        let (r2, t2) = build_rgba8(5, 3, false, 0, &px2);
        assert_eq!(r2.levels, 3);
        assert_eq!(t2.len(), 15 + 2 + 1);
        assert_eq!(t2.len(), r2.texel_count());
    }

    #[test]
    fn build_rgba8_decodes_srgb_and_keeps_alpha_linear() {
        // 码值 128（中间灰）：sRGB 解码后 ≈ 0.21586，线性直存则是 0.50196。
        let px = [128u8, 128, 128, 200];
        let (r_srgb, t_srgb) = build_rgba8(1, 1, true, 0, &px);
        let (r_lin, t_lin) = build_rgba8(1, 1, false, 0, &px);
        assert_eq!(r_srgb.srgb, SRGB_ENCODED);
        assert_eq!(r_lin.srgb, SRGB_LINEAR);
        assert!((t_srgb[0][0] - 0.2158).abs() < 1e-3, "sRGB 128 解码 ≈ 0.2159");
        assert!((t_lin[0][0] - 128.0 / 255.0).abs() < 1e-6);
        // alpha 不走 sRGB（两条路径同值）。
        assert_eq!(t_srgb[0][3].to_bits(), t_lin[0][3].to_bits());
    }

    #[test]
    #[should_panic(expected = "RGBA8 像素数必须等于")]
    fn build_rgba8_rejects_short_pixel_buffer() {
        let px = [0u8; 4 * 3]; // 3 像素，但声明 2×2
        let _ = build_rgba8(2, 2, false, 0, &px);
    }

    #[test]
    fn build_linear_f32_marks_hdr_format_and_rejects_srgb_flag() {
        let px = vec![[4.0f32, 2.0, 1.0, 1.0]; 4];
        let (r, t) = build_linear_f32(2, 2, false, 7, &px);
        assert_eq!(r.fmt, TEX_FMT_RGBA16F);
        assert_eq!(r.offset, 7, "offset 原样传递（Scene::texels 的起点）");
        assert_eq!(t.len(), 4 + 1, "2×2 → 1×1");
        assert_eq!(t[4][0], 4.0, "HDR 值不被钳制");
    }

    #[test]
    #[should_panic(expected = "线性 f32 源不得标 SRGB_ENCODED")]
    fn build_linear_f32_rejects_srgb_flag() {
        let px = vec![[0.5f32, 0.5, 0.5, 1.0]; 4];
        let _ = build_linear_f32(2, 2, true, 0, &px);
    }

    #[test]
    fn mip_levels_matches_spec_formula() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(2, 2), 2);
        assert_eq!(mip_levels(7, 3), 3); // 1 + floor(log2 7) = 1 + 2 = 3 级
        assert_eq!(mip_levels(2048, 1024), 12); // 1 + 11
    }
}
