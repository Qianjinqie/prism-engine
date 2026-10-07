//! SHA-256（FIPS 180-4）纯手写实现。
//!
//! 语义权威：S10 行（prism-io 宿主 SHA-256）、
//! 签名/键取 256 位哈希、宪章 C13（缓存键含 schema/engine 版本，
//! 本 crate 是缓存 manifest 侧的哈希原语）、D-29（确定性纪律：整数运算，
//! 不依赖 libm——本实现只用整数移位/旋转/加法，跨平台逐位一致）。
//!
//! # 约束
//! - 零第三方依赖、零 `unsafe`、不依赖 libm：全部 `u32` 整数运算；
//! - 输出 256 位摘要，碰撞概率 ~10⁻⁵⁹（10⁹ 条目生日界），引擎以无碰撞为
//!  公理；
//! - 测试向量取 FIPS 180-4 官方向量：空串、"abc"、56 字节双块边界串、
//!   'a'×1 000 000 长输入（含长度字段与多块推进路径）。

/// SHA-256 轮常量 K[0..64]：前 64 个素数的立方根小数部分前 32 位（FIPS 180-4 §4.2.3）。
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// 处理一个 512 位消息块：消息调度 + 64 轮压缩（FIPS 180-4 §6.2.2）。
fn compress(h: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64, "压缩函数输入必须是完整 64 字节块");

    // 消息调度：前 16 字大端直读；其余由 σ0/σ1 扩展。
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        *word = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h7] = *h;
    for i in 0..64 {
        // Σ1(e) 与 ch(e,f,g)。
        let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h7
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        // Σ0(a) 与 maj(a,b,c)。
        let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = big_s0.wrapping_add(maj);

        h7 = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    let rounds_out = [a, b, c, d, e, f, g, h7];
    for (slot, value) in h.iter_mut().zip(rounds_out.iter()) {
        *slot = slot.wrapping_add(*value);
    }
}

/// 计算 SHA-256 摘要（FIPS 180-4）：输入任意长度字节序列，输出 256 位。
///
/// 填充规则：追加 `0x80`、补零至距块尾 8 字节、追加 64 位大端比特长度
/// （模 2⁶⁴，与 FIPS 一致）。
pub fn sha256(data: &[u8]) -> [u8; 32] {
    // 初值 H[0..8]：前 8 个素数的平方根小数部分前 32 位（FIPS 180-4 §5.3.3）。
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    // 组装填充后的完整消息（原始长度 ≤ 2 字节边界内安全：u64 比特长度）。
    let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
    let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        compress(&mut h, block);
    }

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

// ---------------------------------------------------------------------------
// 测试：FIPS 180-4 官方向量
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 把摘要格式化为十六进制（与 FIPS 文档记法对拍用）。
    fn hex(digest: &[u8; 32]) -> String {
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn fips_vector_empty_string() {
        // FIPS 180-4：SHA256("") —— 单个纯填充块路径。
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn fips_vector_abc() {
        // FIPS 180-4：SHA256("abc") —— 3 字节单块路径。
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn fips_vector_two_block_boundary() {
        // FIPS 180-4：56 字节消息 —— 填充恰好溢出到第二个块（双块边界）。
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(msg.len(), 56);
        assert_eq!(
            hex(&sha256(msg)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn fips_vector_one_million_a() {
        // FIPS 180-4：'a'×1 000 000 —— 多块推进 + 64 位长度字段路径。
        let msg = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&sha256(&msg)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn digest_is_deterministic_and_input_sensitive() {
        // 同输入同摘要（确定性）。
        assert_eq!(sha256(b"prism"), sha256(b"prism"));
        // 异输入异摘要（smoke）。
        assert_ne!(sha256(b"prism"), sha256(b"prisn"));
        // 摘要长度恒为 256 位。
        assert_eq!(sha256(&[]).len(), 32);
    }
}
