//! `prism-bytes` —— **全引擎唯一的 DEFLATE / zlib 解码真源**（L0 叶子层）。
//!
//! # 职责
//!
//! 本 crate 只做一件事：**字节流 ⇄ 字节流**的解码，具体是
//!
//! - [`inflate`] —— RFC 1951 DEFLATE 裸码流（stored / fixed Huffman / dynamic Huffman）；
//! - [`zlib`] —— RFC 1950 zlib 容器（2 字节头 + DEFLATE + 4 字节 Adler-32 尾）
//!   与 Adler-32 校验和。
//!
//! 它**不**做图像/音频/视频的任何领域解释：不解 PNG 滤波、不解 GIF LZW、
//! 不解 EXR 通道布局、不解 APNG 帧合成。那些留在各自领域 crate。
//!
//! # 为什么需要它（这笔债的来历）
//!
//! 项目里曾有**两份手写 DEFLATE inflate**：
//!
//! 1. `prism-io/src/exr.rs` 内的一份（U7）；
//! 2. `prism-media/src/inflate.rs` 内的一份（B11）。
//!
//! 第二份的注释写着「因循环依赖无法复用，自研第二份 inflate，形成技术债」。
//! 分层是 `io ∈ L4 → core ∈ L2`，低层不得依赖高层，两份实现都被卡在 L4 附近，
//! 谁也够不着谁。**双份实现 = 双份 bug 面**，而它已经产生过事故：EXR 侧的
//! zlib 容器口径（裸 DEFLATE vs zlib 流）错3 次才由测试逮住。
//!
//! 下沉到**所有消费者之下**的 L0 后，`prism-io`(L4) 与 `prism-media`(L4) 都向下
//! 依赖本 crate，反向依赖不存在，循环自然消失。
//!
//! # 为什么是叶子
//!
//! 本 crate **不依赖本仓任何其他 crate**（连 `prism-math` 也不用）。
//! 理由：它处在依赖图的**最底部**，任何一个 `use` 都会成为全仓所有 crate 的
//! 传递依赖，把 L0 拖进高层依赖网。DEFLATE 解码只需要 `std`（位运算 +
//! `Vec`），引入任何数学库都是负债而非资产。
//!
//! # 铁律
//!
//! - **零第三方 crate**：只用 `std`；
//! - **确定性**：位流固定低位→高位，Huffman 固定按码长→符号升序，
//!   同输入逐字节一致，跨平台一致；
//! - **永不 panic**：所有下标/长度都经 `get()` 与 `min()` 兜底，畸形输入
//!   （截断、越界码长、保留块类型）返回三段式 [`BytesError`]。
//!
//! # 两处刻意不共用同一条代码路径的地方
//!
//! 1. **本 crate（读取侧）与 `exr_write::deflate_zlib`（写入侧）**：方向相反，
//!    不存在"共用"的余地；更要紧的是 EXR 读取侧测试**刻意不复用**写入侧实现
//!    来造负载——那会让读/写两侧共用同一份实现，两边同错时互相掩盖，
//!    读取侧测试就再也逮不住写入侧的 bug。这个顾虑是对的，详见 [`zlib`] 模块文档。
//! 2. **EXR 测试构造器 `zlib_stream()` 与本 crate 的 [`zlib`]**：前者造流、
//!    后者解流，本来就是一对被测关系，不应合并。
//!
//! 换言之：**结构上收敛到一处（解码只有本 crate），验证上保持互相独立**。
//!
//! # 诚实边界
//!
//! - 只做**解压**，**没有 deflate 压缩器**。写入侧压缩仍分散在各领域 crate
//!   （`exr_write::deflate_zlib` 目前只有 stored 块，无压缩比）。
//! - **只支持 zlib 容器**：不支持 gzip（RFC 1952）、zstd、brotli、LZMA；
//!   不支持 FDICT（预置字典），遇到即结构化拒绝。
//! - [`zlib::decompress`] 不比对 Adler-32 **值**（只校验尾部存在），
//!   需要校验和用 [`zlib::decompress_verified`]。
//! - 不支持多 gzip 成员流、不支持尾部拼接的其他压缩格式。

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod inflate;
pub mod zlib;

pub use inflate::inflate_raw;
pub use zlib::{adler32, decompress, decompress_verified, parse_header, ZlibHeader};

use std::fmt;

/// 字节层错误（三段式：在哪/ 什么事 / 怎么改）。
///
/// **错误码由调用方分配**：本 crate 是叶子，不知道自己被EXR 还是 APNG 使用，
/// 故只给出三段式内容，调用方把它映射成自己领域的错误码
/// （`prism-io` → `EXR2001`，`prism-media` → `INF1001`）。这样既保住各领域
/// 既有的错误码分段（不破坏既有断言），又让实现只有一份。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytesError {
    /// 三段式一【在哪】：模块 / 函数。
    pub location: &'static str,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl BytesError {
    /// 构造一个三段式字节层错误。
    pub fn new(
        location: &'static str,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { location, problem: problem.into(), fix: fix.into() }
    }
}

impl fmt::Display for BytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}：{}（修复：{}）", self.location, self.problem, self.fix)
    }
}

impl std::error::Error for BytesError {}

/// 字节层 `Result` 别名。
pub type BytesResult<T> = Result<T, BytesError>;