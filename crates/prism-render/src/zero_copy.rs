//! 零拷贝身份判据（T09 / V2.2-D-2 判据 ③；V3.1 收尾扩展到出片主链）。
//!
//! 判据是「指针相等」（`as_ptr()`），而非「看起来没拷」。防「声明了但没做」：
//! 只有显式登记在 [`ZERO_COPY_WHITELIST`] 的路径才被认可为零拷贝；未登记路径
//! 不得声称零拷贝。[`verify_zero_copy`] 同时校验「已登记」与「输入/输出缓冲
//! 指针相等」。
//!
//! # 两类登记路径（V3.1 收尾 C）
//! - **identity**：输入缓冲 = 输出缓冲（同一分配，指针不变），如 tile 直通、
//!   frame reborrow、mux 借用视图；
//! - **reuse**：跨帧复用同一分配（归还帧池后再取出的缓冲指针与归还前相等），
//!   如 `movie.frame_pool_reuse`。两者都由同一「指针相等」判据钉死。

use prism_gpu::Framebuffer;

/// 已声明为零拷贝 / 零复用分配的路径白名单（零 crate，编译期固定集合）。
///
/// 任何想自称零拷贝的路径必须在此登记；否则 [`verify_zero_copy`] 直接拒绝。
pub const ZERO_COPY_WHITELIST: &[&str] = &[
    "tile.passthrough_identity",
    "frame.buffer_reborrow",
    // V3.1 出片主链：Framebuffer.pixels → FrameView.rgba8（CLI 借用视图）。
    "mux.frame_view_borrow",
    // V3.1 出片主链：帧池归还缓冲 → 下一帧取出缓冲（同分配跨帧复用）。
    "movie.frame_pool_reuse",
];

/// 路径是否登记为零拷贝。
pub fn is_zero_copy_path(path: &str) -> bool {
    ZERO_COPY_WHITELIST.contains(&path)
}

/// 校验零拷贝 / 零复用分配身份（判据 ③）：
/// 1. `path` 必须登记在 [`ZERO_COPY_WHITELIST`]（否则即「未声明路径声称零拷贝」，
///    违反映计 ③ 的「防声明了但没做」）；
/// 2. 输入/输出缓冲指针必须相等（否则即「声明了但做了拷贝 / 新分配」）。
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

/// 出片主链借用链判据：`source`（`Framebuffer.pixels`）→ `view`（`FrameView.rgba8`）
/// 必须是指针相等的同一分配。语义化封装，供 `cmd_render` 主链调用。
pub fn verify_borrow_chain(source: &[u8], view: &[u8]) {
    verify_zero_copy("mux.frame_view_borrow", source, view);
}

/// 帧池复用判据：归还前一帧缓冲与下一帧取出缓冲必须指针相等。
pub fn verify_pool_reuse(released: &[u8], acquired: &[u8]) {
    verify_zero_copy("movie.frame_pool_reuse", released, acquired);
}

/// 零拷贝透明直通（帧级）：返回输入缓冲本身（同一分配，指针不变）。
///
/// 登记于 [`ZERO_COPY_WHITELIST`] 的 `frame.buffer_reborrow`；仅当确实不分配、
/// 不拷贝、仅转交所有权/借用时使用。
pub fn buffer_reborrow(fb: &Framebuffer) -> &Framebuffer {
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_chain_paths_are_whitelisted() {
        assert!(is_zero_copy_path("mux.frame_view_borrow"));
        assert!(is_zero_copy_path("movie.frame_pool_reuse"));
        assert!(!is_zero_copy_path("made.up.path"));
    }

    #[test]
    fn borrow_chain_is_pointer_checked() {
        let fb: Framebuffer = Framebuffer::new(4, 4);
        verify_borrow_chain(&fb.pixels, &fb.pixels);
        verify_pool_reuse(&fb.pixels, &fb.pixels);
    }

    #[test]
    #[should_panic(expected = "声明零拷贝但")]
    fn borrow_chain_rejects_a_real_copy() {
        let a: Vec<u8> = vec![0; 16];
        let b: Vec<u8> = vec![0; 16];
        verify_borrow_chain(&a, &b);
    }
}
