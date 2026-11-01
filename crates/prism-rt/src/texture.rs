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
}
