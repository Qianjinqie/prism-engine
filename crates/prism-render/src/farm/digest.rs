//! 零依赖 MD5（RFC 1321）——主节点 manifest 校验与"并行 == 串行"证据。
//!
//! # 为什么需要它
//!
//! 路线图 §2.6 P2 设计要求"主节点校验 md5 合帧"：worker 交回帧文件，主节点
//! 逐帧校验 md5 后再合帧。要独立实现（本仓库**零第三方依赖**铁律：不引
//! `md5` / `digest` / `crypto` 任何 crate），只用 Rust 标准库。
//!
//! # 纯整数，无浮点、无平台 API
//!
//! MD5 全部运算都是 `u32` 的加法 / 位运算 / 循环移位。移位用
//! `rotate_left`（`u32::rotate_left` 是纯整数、标准库、跨平台一致），
//! 加法用 `wrapping_add`（RFC 1321 定义的就是 mod 2^32 的加法，不是"防溢出
//! 截断"而是**规格本身**）。因此 `md5_hex` 是输入字节的**纯函数**：
//! 同输入两遍逐字符相同，与机器、线程数、进程数无关。
//!
//! 这正是"并行 md5 == 串行 md5"能被严格断言的前提：**digest 本身也是纯
//! 函数**，所以两条路径的 digest 字符串可以直接 `assert_eq!` 比对。
//!
//! # 与 `orchestrator` 测试助手的差异（诚实边界）
//!
//! `crate::orchestrator` 的 `#[cfg(test)] mod md5` 是早期为 tile 编排器写的
//! 测试助手。本模块是**生产代码**（农场 manifest 校验要用），两者算法同源
//! 但存在目的不同，故不合并——合并会动到 orchestrator 的既有测试（回归风险
//! 违反"既有测试零回归"）。已请主理人裁决是否后续统一。

/// MD5 分组字节数（RFC 1321 §3：512 位 = 64 字节）。
const BLOCK_BYTES: usize = 64;

/// RFC 1321 §3.4 的每轮左移量表（16 项 × 4 轮）。
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// RFC 1321 §3.4 的常量表 `K[i] = floor(2^32 × |sin(i + 1)|)`。
///
/// 硬编码整数（**不调用 `libm::sin`**）：调用 `f64::sin` 会引入 libm 依赖，
/// 且其结果在不同平台/版本上不保证逐位一致——而本模块的整个意义就是逐位
/// 确定的 digest。RFC 1321 附录给出这 64 个常量，此处照抄。
const K: [u32; 64] = [
    0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee, //
    0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501, //
    0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be, //
    0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821, //
    0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa, //
    0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8, //
    0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed, //
    0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a, //
    0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c, //
    0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70, //
    0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05, //
    0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665, //
    0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039, //
    0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1, //
    0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1, //
    0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
];

/// 计算 MD5 并返回小写十六进制串（32 字符）。
///
/// **纯函数**：同输入两遍逐字符相同。零第三方依赖、零浮点、零平台 API。
///
/// # 例子
/// ```
/// use prism_render::farm::md5_hex;
/// assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
/// assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
/// ```
pub fn md5_hex(data: &[u8]) -> String {
    let digest: [u32; 4] = md5_words(data);
    let mut out: String = String::with_capacity(32);
    for word in digest {
        // RFC 1321 §3.4：摘要输出是 A/B/C/D 四个字的**小端字节序列**。
        // 注意不能直接 `format!("{word:08x}")`——那打印的是字的大端十六进制，
        // 会得到逐字节颠倒的结果（曾经踩过：`"abc"` 会得到
        // `98500190b04fd23c7d3f96d6727fe128` 而非
        // `900150983cd24fb0d6963f7d28e17f72`）。必须按小端逐字节输出。
        for byte in word.to_le_bytes() {
            out.push(HEX_DIGITS[usize::from(byte >> 4)]);
            out.push(HEX_DIGITS[usize::from(byte & 0x0f)]);
        }
    }
    out
}

/// 小写十六进制数字表（避免 `format!` 引入临时 String，也避免 `as` 强转）。
const HEX_DIGITS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', //
    '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// MD5 核心：返回 4 个 `u32` 字（`a, b, c, d`）。
///
/// 按 RFC 1321 §3.3/§3.4：填充 → 分组 → 64 步四轮。所有加减都用
/// `wrapping_*`（这是 mod 2^32 的**规格语义**，不是溢出防护）。
fn md5_words(data: &[u8]) -> [u32; 4] {
    // ---- 初始状态（RFC 1321 §3.3 规定的 A/B/C/D 初值） ----
    let mut a0: u32 = 0x6745_2301;
    let mut b0: u32 = 0xefcd_ab89;
    let mut c0: u32 = 0x98ba_dcfe;
    let mut d0: u32 = 0x1032_5476;

    // ---- 填充（§3.1） ----
    // 追加 0x80，补零到长度 ≡ 56 (mod 64)，再追加 8 字节小端比特长度。
    // 比特长度用 u64 累加（RFC 定义就是 mod 2^64 截断，不是溢出防护）。
    let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
    let mut padded: Vec<u8> = Vec::with_capacity(data.len() + 2 * BLOCK_BYTES);
    padded.extend_from_slice(data);
    padded.push(0x80);
    // 补零到 (len + 8) 是 64 的倍数。补零数 = 目标长度 - 当前长度。
    while (padded.len() + 8) % BLOCK_BYTES != 0 {
        padded.push(0u8);
    }
    for byte in bit_len.to_le_bytes() {
        padded.push(byte);
    }

    // ---- 分组处理（§3.4） ----
    let mut offset: usize = 0;
    while offset + BLOCK_BYTES <= padded.len() {
        // 分组内 16 个小端 u32 字（RFC 1321 §3.4 的 `X[0..16]`）。
        let mut x: [u32; 16] = [0u32; 16];
        for (i, word) in x.iter_mut().enumerate() {
            let base: usize = offset + i * 4;
            // 循环条件保证 base+4 <= padded.len()，故四个下标全部在界内；
            // 仍用 get 兜底（C22：禁越界索引），缺失按 0 处理。
            let b0b: u32 = byte_at(&padded, base).into();
            let b1b: u32 = byte_at(&padded, base + 1).into();
            let b2b: u32 = byte_at(&padded, base + 2).into();
            let b3b: u32 = byte_at(&padded, base + 3).into();
            *word = b0b | (b1b << 8) | (b2b << 16) | (b3b << 24);
        }

        let (mut a, mut b, mut c, mut d): (u32, u32, u32, u32) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g): (u32, usize) = md5_round_step(i, b, c, d);
            let tmp: u32 = d;
            d = c;
            c = b;
            // b = b + leftrotate(a + f + K[i] + X[g], s[i]) —— 全部 mod 2^32。
            b = b.wrapping_add(
                a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(x[g]).rotate_left(SHIFTS[i]),
            );
            a = tmp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
        offset = offset.wrapping_add(BLOCK_BYTES);
    }
    [a0, b0, c0, d0]
}

/// 第 `i` 步的非线性函数 `F` 与字下标 `g`（RFC 1321 §3.4 表格）。
fn md5_round_step(i: usize, b: u32, c: u32, d: u32) -> (u32, usize) {
    match i / 16 {
        0 => ((b & c) | (!b & d), i),
        1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
        2 => (b ^ c ^ d, (3 * i + 5) % 16),
        // 第 4 轮：i ∈ [48, 64) ⇒ (7 * i) % 16 ∈ [0, 16)，不会越界。
        _ => (c ^ (b | !d), (7 * i) % 16),
    }
}

/// 读取 `data[i]`，越界返回 0（padding 语义；禁越界索引）。
fn byte_at(data: &[u8], i: usize) -> u8 {
    data.get(i).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性载荷派生：splitmix64（纯整数，禁 `rand` crate）。
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z: u64 = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[test]
    fn md5_matches_rfc1321_suite() {
        // RFC 1321 §附录 A.5 的官方向量。
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5_hex(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(md5_hex(b"abcdefghijklmnopqrstuvwxyz"), "c3fcd3d76192e4007dfb496cca67e13b");
        assert_eq!(
            md5_hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
        assert_eq!(
            md5_hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn md5_is_pure_function_of_bytes_across_lengths() {
        // 覆盖分组边界前 / 边界上 / 边界后（55/56/57/63/64/65）与非 2 的幂。
        for n in [0usize, 1, 3, 7, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000] {
            let mut state: u64 = 0x5eed_0000_u64.wrapping_add(n as u64);
            let data: Vec<u8> = (0..n).map(|_| (splitmix64(&mut state) & 0xff) as u8).collect();
            let a: String = md5_hex(&data);
            let b: String = md5_hex(&data);
            assert_eq!(a, b, "n={n}：md5 必须是输入的纯函数");
            assert_eq!(a.len(), 32, "n={n}：十六进制串必须 32 字符");
            assert!(a.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
        }
    }

    #[test]
    fn md5_differs_when_any_single_bit_differs() {
        // 单比特敏感性：不同载荷的 digest 必须不同（否则无法做 manifest 校验）。
        let base: Vec<u8> = (0..64u8).collect();
        let base_digest: String = md5_hex(&base);
        for i in 0..base.len() {
            let mut other: Vec<u8> = base.clone();
            other[i] ^= 0x01;
            assert_ne!(base_digest, md5_hex(&other), "字节 {i} 翻转后 digest 必须改变");
        }
    }
}
