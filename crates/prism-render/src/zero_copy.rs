//! 零拷贝身份判据（T09 / V2.2-D-2 判据 ③）。
//!
//! 判据是「指针相等」（`as_ptr()`），而非「看起来没拷」。防「声明了但没做」：
//! 只有显式登记在 [`ZERO_COPY_WHITELIST`] 的路径才被认可为零拷贝；未登记路径
//! 不得声称零拷贝。[`verify_zero_copy`] 同时校验「已登记」与「输入/输出缓冲
//! 指针相等」。

use prism_gpu::Framebuffer;

/// 已声明为零拷贝的路径白名单（零 crate，编译期固定集合）。
///
/// 任何想自称零拷贝的路径必须在此登记；否则 [`verify_zero_copy`] 直接拒绝。
/// 当前登记项均为「输入缓冲 = 输出缓冲（同一分配，指针不变）」的真零拷贝。
pub const ZERO_COPY_WHITELIST: &[&str] = &[
    "tile.passthrough_identity",
    "frame.buffer_reborrow",
];

/// 路径是否登记为零拷贝。
pub fn is_zero_copy_path(path: &str) -> bool {
    ZERO_COPY_WHITELIST.contains(&path)
}

/// 校验零拷贝身份（判据 ③）：
/// 1. `path` 必须登记在 [`ZERO_COPY_WHITELIST`]（否则即「未声明路径声称零拷贝」，
///    违反映计 ③ 的「防声明了但没做」）；
/// 2. 输入/输出缓冲指针必须相等（否则即「声明了但做了拷贝」）。
///
/// # Panics
/// 任一条件不满足即 panic（测试硬断言）；panic 文案含「未登记」/「声明零拷贝但」。
pub fn verify_zero_copy(path: &str, input: &[u8], output: &[u8]) {
    assert!(
        is_zero_copy_path(path),
        "路径 '{path}' 未登记为零拷贝，不得声称零拷贝（防声明了但没做）"
    );
    assert_eq!(
        input.as_ptr(),
        output.as_ptr(),
        "路径 '{path}' 声明零拷贝但输入/输出缓冲指针不等（发生了拷贝）"
    );
}

/// 零拷贝透明直通（帧级）：返回输入缓冲本身（同一分配，指针不变）。
///
/// 登记于 [`ZERO_COPY_WHITELIST`] 的 `frame.buffer_reborrow`；仅当确实不分配、
/// 不拷贝、仅转交所有权/借用时使用。
pub fn buffer_reborrow(fb: &Framebuffer) -> &Framebuffer {
    fb
}
