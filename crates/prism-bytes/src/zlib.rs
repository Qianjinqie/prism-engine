//! RFC 1950 zlib 容器：`2 字节头 + DEFLATE + 4 字节 Adler-32`。
//!
//! DEFLATE 裸码流解码在 [`crate::inflate`]。
//!
//! # 与写入侧「实现互相独立」的理由（重要，勿合并）
//!
//! 本模块是**读取侧**。`prism_io::exr_write::deflate_zlib` 是**写入侧**的 zlib
//! 压缩器。二者**刻意不共用同一条代码路径**，理由是 U7 的一条教训：
//!
//! > EXR 侧的 zlib 容器口径（裸 DEFLATE vs zlib 流）错 3 次才由测试逮住。
//!
//! 如果读取侧测试用写入侧的实现来造负载，那么**读/写两侧会同时犯同一个错，
//! 且互相掩盖**——测试永远绿，错误却已写进文件。所以读取侧的测试负载继续
//! 自建独立编码器（见本 crate 文档「两处刻意不共用同一条代码路径的地方」）。
//!
//! 换言之：**验证链上的两端必须独立**，这是有意的设计冗余，不是待清理的重复。
//! 本 crate 只做「结构上收敛到一处」：解码实现唯一。
//!
//! # 为什么不做 Adler-32 强制校验
//!
//! [`decompress`] 只校验尾部**存在**，不校验校验和值——这是刻意保留的既有行为
//! （两侧旧实现都不校验）。需要完整性校验的调用方（PNG 的 IDAT 等自带 CRC 的
//! 格式，或任何怕损坏的读入路径）用 [`decompress_verified`]。

use crate::inflate::inflate_raw_prefix;
use crate::{BytesError, BytesResult};

/// 构造三段式错误。
fn err(location: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> BytesError {
    BytesError::new(location, problem, fix)
}

/// zlib 头固定长度（CMF + FLG）。
const HEADER_LEN: usize = 2;
/// zlib 尾部 Adler-32 长度。
const TRAILER_LEN: usize = 4;

/// zlib 容器头解析结果（`CMF` / `FLG` 两字节）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZlibHeader {
    /// 压缩方法（低 4 位 `CM`）：必须为 `8`（DEFLATE）。
    pub cm: u8,
    /// 窗口尺寸指示（高 4 位 `CINFO`）：必须 `≤ 7`（32K 窗口）。
    pub cinfo: u8,
    /// 压缩等级提示（`FLEVEL`，第 6..7 位）：仅提示，**不参与**正确性。
    pub flevel: u8,
    /// 是否声明 FDICT 预置字典：标准 zlib 流必须为 `false`。
    pub fdict: bool,
}

/// Adler-32 校验和（RFC 1950 §9）。
///
/// 已知向量：`adler32(b"")` = 1，`adler32(b"a")` = 0x0062_0062，
/// `adler32(b"abc")` = 0x024D_0127。
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// 解析并校验 zlib 容器头（`CMF` / `FLG`）。
///
/// 校验项（RFC 1950 §2.2）：
/// 1. 至少 2 字节；
/// 2. `(CMF << 8 | FLG) % 31 == 0`（FCHECK）；
/// 3. `CM == 8`（DEFLATE）；
/// 4. `CINFO ≤ 7`（窗口 ≤ 32K）；
/// 5. `FDICT == 0`（不支持预置字典）。
///
/// **校验顺序**：先 FCHECK 再 CM 位域，与旧 `prism-media` 侧一致（旧
/// `prism-io` 侧顺序相反）。二者对**同一输入的可解性判定完全一致**
/// （FCHECK 失败 ⟺ 非标准头），只是 `problem` 文案不同，故对拍不受影响。
pub fn parse_header(data: &[u8]) -> BytesResult<ZlibHeader> {
    const LOC: &str = "zlib::parse_header";
    let head = data.get(..HEADER_LEN).ok_or_else(|| {
        err(
            LOC,
            format!("zlib 流短于 2 字节头（实际 {} 字节）", data.len()),
            "压缩块被截断；确认文件完整",
        )
    })?;
    let cmf = head[0];
    let flg = head[1];
    if !(((cmf as u16) << 8) | flg as u16).is_multiple_of(31) {
        return Err(err(LOC, "zlib 头 FCHECK 校验失败", "zlib 头损坏；确认文件完整"));
    }
    let cm = cmf & 0x0f;
    if cm != 8 {
        return Err(err(
            LOC,
            format!("zlib CMF = {cmf:#04x}（低 4 位须为 8 = deflate）"),
            "该块不是 zlib 流；确认压缩类型标注正确且文件未被改写",
        ));
    }
    let cinfo = cmf >> 4;
    if cinfo > 7 {
        return Err(err(
            LOC,
            format!("zlib 窗口尺寸超过 32K（CINFO = {cinfo}，须 ≤ 7）"),
            "该流不是标准 zlib 容器；确认数据完整",
        ));
    }
    let fdict = flg & 0x20 != 0;
    if fdict {
        return Err(err(
            LOC,
            "zlib 流声明了 FDICT（预置字典）",
            "本实现不支持预置字典；该数据不是标准压缩流",
        ));
    }
    Ok(ZlibHeader { cm, cinfo, flevel: (flg >> 6) & 0x03, fdict })
}

/// zlib 解压：头 + 裸 DEFLATE + 尾部存在性检查（**不**校验 Adler-32 值）。
///
/// # 与旧 `prism-io::exr::zlib_decompress` 的行为差异（有意为之，属修 bug）
///
/// 旧实现在解出后有一句
///
/// ```text
/// if data.len() < 2 + out.len() { return Err("zlib 流截断") }
/// ```
///
/// 它把**压缩后总长**与**解压后长度**相比——只要压缩率 > 0（即真实世界里一切
/// 正常压缩的数据），`data.len()` 必然小于 `2 + out.len()`，于是**任何真正
/// 压缩过的 zlib 流都会被误判为「截断」**。
///
/// 本实现用 [`crate::inflate::inflate_raw_prefix`] 回传的**实际消费量**定位
/// 尾部，判定与 RFC 1950 §2.2 一致。
pub fn decompress(data: &[u8]) -> BytesResult<Vec<u8>> {
    const LOC: &str = "zlib::decompress";
    parse_header(data)?;
    let body = data.get(HEADER_LEN..).unwrap_or(&[]);
    let (out, consumed) = inflate_raw_prefix(body)?;
    // 尾部 4 字节 Adler-32 的**存在性**检查（不比较值，与既有行为一致）。
    let end = HEADER_LEN + consumed;
    if data.len() < end + TRAILER_LEN {
        return Err(err(
            LOC,
            format!(
                "zlib 流截断：DEFLATE 结束于偏移 {end}，其后需再 {TRAILER_LEN} 字节校验和（总长 {}）",
                data.len()
            ),
            "压缩块被截断；确认文件完整",
        ));
    }
    Ok(out)
}

/// zlib 解压（**宽容口径**）：只校验头，**不**要求尾部 Adler-32 存在。
///
/// # 为什么需要这个入口（迁移不变性要求，不是多余的第三个函数）
///
/// 迁移前项目里有**两种**zlib 尾部口径：
///
/// | 旧实现 | 尾部口径 |
/// |---|---|
/// | `prism_io::exr::zlib_decompress` | 要求尾部存在（但判据写错，见 [`decompress`]） |
/// | `prism_media::inflate::zlib_decompress` | **完全不看尾部**，校验完 2 字节头就 `inflate_raw(&data[2..])` |
///
/// media 侧的宽容是**其既有 API 契约的一部分**：仓库里两个 PNG 测试构造器
/// （`prism-media/tests/b11_integration.rs` 与 `src/apng.rs` 的 `zlib_stored`）
/// 造的就是**不带 Adler-32 尾**的 zlib 流，收紧尾部要求会让它们当场回归。
///
/// 「迁移后行为必须与迁移前一致，不一致是回归不是改进」——所以这里保留宽容
/// 入口，而不是顺手把 media侧也收紧。**收紧口径应当是独立的一次行为变更**，
/// 由调用方（PNG/EXR 写入侧补齐尾部）先就位，再单独提出。
///
/// 换言之：本 crate 提供**两种口径**是迁移的忠实还原，不是重复实现——
/// 两者共用同一条 DEFLATE 解码路径（[`crate::inflate`]），差别只在尾部策略。
pub fn decompress_lenient(data: &[u8]) -> BytesResult<Vec<u8>> {
    const LOC: &str = "zlib::decompress_lenient";
    parse_header(data)?;
    let body = data.get(HEADER_LEN..).unwrap_or(&[]);
    crate::inflate::inflate_raw_prefix(body).map(|(out, _)| out).map_err(|e| {
        // location 归到本函数名下，保持调用方按 location 聚合时语义完整。
        BytesError::new(LOC, e.problem, e.fix)
    })
}

/// zlib 解压 + **强制校验**尾部 Adler-32。
///
/// 与 [`decompress`] 的唯一区别是会比对校验和值。用于自带完整性保护
/// （如 PNG `IDAT` 外层还有 CRC）的格式，或任何需要防损坏的读入路径。
pub fn decompress_verified(data: &[u8]) -> BytesResult<Vec<u8>> {
    const LOC: &str = "zlib::decompress_verified";
    let out = decompress(data)?;
    // 尾部定位：DEFLATE 码流消费到 `end`，其后 4 字节即 Adler-32。
    // 允许尾部之后有多余字节（本实现与旧两侧都容忍），故从末尾取 4 字节。
    let tail = data
        .get(data.len().saturating_sub(TRAILER_LEN)..)
        .filter(|t| t.len() == TRAILER_LEN)
        .ok_or_else(|| err(LOC, "zlib 流缺少 Adler-32 尾", "压缩块被截断；确认文件完整"))?;
    let got = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let want = adler32(&out);
    if got != want {
        return Err(err(
            LOC,
            format!("Adler-32 校验失败：尾部 {got:#010x}，实算 {want:#010x}"),
            "压缩数据损坏；确认文件完整且未被改写",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 自建 zlib stored 流（`CMF`/`FLG` + stored 块 + Adler-32）。
    ///
    /// **刻意不复用任何写入侧实现**（见模块文档「与写入侧实现互相独立」）。
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&adler32(payload).to_be_bytes());
        out
    }

    /// 手工构造的 **fixed Huffman** zlib 流：把 259 个 `A` 压到 ~4 字节。
    ///
    /// 用于钉住「真正压缩过的流必须能解」——旧 `prism-io` 侧正是在这里
    /// 把压缩流误判为截断。
    fn zlib_fixed_rle259() -> Vec<u8> {
        let mut cur: u32 = 0b011; // BFINAL=1, BTYPE=01（fixed）
        let mut nbit: u32 = 3;
        let mut out: Vec<u8> = Vec::new();
        fn put(code: u32, n: u32, cur: &mut u32, nbit: &mut u32, out: &mut Vec<u8>) {
            for i in (0..n).rev() {
                if (code >> i) & 1 == 1 {
                    *cur |= 1 << *nbit;
                }
                *nbit += 1;
                if *nbit == 8 {
                    out.push(*cur as u8);
                    *cur = 0;
                    *nbit = 0;
                }
            }
        }
        put(0x30 + 65, 8, &mut cur, &mut nbit, &mut out); // 字面 'A'
        put(0xc0 + 5, 8, &mut cur, &mut nbit, &mut out); // 长度码 285（基础长度 258）
        put(0, 5, &mut cur, &mut nbit, &mut out); // 距离码 0（距离 1）
        put(0, 7, &mut cur, &mut nbit, &mut out); // 256 块结束
        if nbit > 0 {
            out.push(cur as u8);
        }
        let payload = vec![b'A'; 259];
        let mut z = vec![0x78, 0x01];
        z.extend_from_slice(&out);
        z.extend_from_slice(&adler32(&payload).to_be_bytes());
        z
    }

    /// **不带Adler-32 尾**的 zlib 流（只有 `CMF`/`FLG` + 裸 DEFLATE）。
    ///
    /// 这正是 `prism-media` 侧两个 PNG 测试构造器
    /// （`tests/b11_integration.rs` 与 `src/apng.rs`）造的东西，
    /// 也是旧 `prism-media::inflate::zlib_decompress` 一直在宽容接收的输入。
    fn zlib_stored_no_trailer(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    // -- 1. Adler-32 已知向量 ---------------------------------------------

    #[test]
    fn adler32_known_vectors() {
        // RFC 1950 §9 的参考值。
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"abc"), 0x024D_0127);
        // Wikipedia 条目「Adler-32」给出的示例。
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn adler32_order_sensitive() {
        // Adler-32 **不**与字节排列无关（这是它与 CRC 的关键差别）。
        assert_ne!(adler32(b"ab"), adler32(b"ba"));
    }

    #[test]
    fn adler32_empty_and_long_input() {
        // 空输入：a=1, b=0 → 1。
        assert_eq!(adler32(b""), 1);
        // 长输入不得溢出（逐字节取模，天然安全）。
        let long: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let _ = adler32(&long);
    }

    // -- 2. zlib 头解析 ---------------------------------------------------

    #[test]
    fn header_typical_levels_parse() {
        // 0x78 0x01 / 0x5E / 0x9C / 0xDA 四种 FLEVEL 都应通过。
        for &flg in &[0x01u8, 0x5E, 0x9C, 0xDA] {
            let h = parse_header(&[0x78, flg, 0x00]).expect("标准 zlib 头应通过");
            assert_eq!(h.cm, 8, "CM 必须是 8");
            assert_eq!(h.cinfo, 7, "0x78 的 CINFO = 7");
            assert!(!h.fdict, "FDICT 应为 false");
        }
    }

    #[test]
    fn header_too_short_rejected() {
        for n in 0..HEADER_LEN {
            let e = parse_header(&vec![0u8; n]).expect_err("短于 2 字节必须拒绝");
            assert_eq!(e.location, "zlib::parse_header");
        }
    }

    #[test]
    fn header_bad_fcheck_rejected() {
        // 0x78 0x02：(0x78<<8|0x02) % 31 ≠ 0。
        let e = parse_header(&[0x78, 0x02]).expect_err("FCHECK 失败必须拒绝");
        assert!(e.problem.contains("FCHECK"), "问题段须点名 FCHECK：{}", e.problem);
    }

    #[test]
    fn header_cm_not_8_rejected() {
        // CM=7，需 FCHECK 通过才能走到这一步。
        let cmf = 0x77u8;
        let mut flg = 0u8;
        while !(((cmf as u16) << 8) | flg as u16).is_multiple_of(31) {
            flg += 1;
        }
        let e = parse_header(&[cmf, flg]).expect_err("CM != 8 必须拒绝");
        assert!(e.problem.contains("CMF"), "问题段须点名 CMF：{}", e.problem);
    }

    #[test]
    fn header_cinfo_gt_7_rejected() {
        // CINFO=8（窗口 64K）超出 DEFLATE 上限；CMF = 0x88。
        let cmf = 0x88u8;
        let mut flg = 0u8;
        while !(((cmf as u16) << 8) | flg as u16).is_multiple_of(31) {
            flg += 1;
        }
        let e = parse_header(&[cmf, flg]).expect_err("CINFO > 7 必须拒绝");
        assert!(e.problem.contains("窗口"), "问题段须点名窗口：{}", e.problem);
    }

    #[test]
    fn header_fdict_rejected() {
        // FDICT = 位 5。CMF=0x78, FLG 含 0x20 且 FCHECK 通过。
        let cmf = 0x78u8;
        let mut flg = 0x20u8;
        while !(((cmf as u16) << 8) | flg as u16).is_multiple_of(31) {
            flg += 1;
        }
        let e = parse_header(&[cmf, flg]).expect_err("FDICT 必须拒绝");
        assert!(e.problem.contains("FDICT"), "问题段须点名 FDICT：{}", e.problem);
    }

    #[test]
    fn header_error_carries_actionable_fix() {
        // C18 三段式：【怎么改】不得为空。
        for bad in [vec![0x78u8], vec![0x78, 0x02]] {
            let e = parse_header(&bad).expect_err("应拒绝");
            assert!(!e.fix.is_empty(), "三段式三【怎么改】不得为空：{e}");
        }
    }

    // -- 3. 往返 ----------------------------------------------------------

    #[test]
    fn roundtrip_stored_small_and_empty() {
        for payload in [Vec::new(), b"hi".to_vec(), (0..255u8).collect::<Vec<_>>()] {
            let z = zlib_stored(&payload);
            assert_eq!(decompress(&z).expect("zlib 往返"), payload);
            assert_eq!(decompress_verified(&z).expect("zlib 往返+校验"), payload);
        }
    }

    #[test]
    fn roundtrip_multi_block_70000_bytes() {
        // 70000 > 65535，stored 块流必须自动分块。
        let payload: Vec<u8> = (0..70000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut deflate = Vec::new();
        let mut off = 0usize;
        loop {
            let n = (payload.len() - off).min(65535);
            let last = off + n >= payload.len();
            deflate.push(if last { 0x01 } else { 0x00 });
            deflate.extend_from_slice(&(n as u16).to_le_bytes());
            deflate.extend_from_slice(&(!(n as u16)).to_le_bytes());
            deflate.extend_from_slice(&payload[off..off + n]);
            off += n;
            if last {
                break;
            }
        }
        let mut z = vec![0x78, 0x01];
        z.extend_from_slice(&deflate);
        z.extend_from_slice(&adler32(&payload).to_be_bytes());
        assert_eq!(decompress(&z).expect("多块 zlib 往返"), payload);
    }

    #[test]
    fn decompress_accepts_trailing_garbage() {
        // 迁移前两侧都容忍 DEFLATE 码流之后的多余字节，保持不变。
        let payload = b"hello".to_vec();
        let mut z = zlib_stored(&payload);
        z.extend_from_slice(b"junkjunk");
        assert_eq!(decompress(&z).expect("多余尾字节应容忍"), payload);
    }

    #[test]
    fn decompress_rejects_missing_trailer() {
        // 去掉尾部 4 字节 → 截断。
        let mut z = zlib_stored(b"hello");
        z.truncate(z.len() - TRAILER_LEN);
        assert!(decompress(&z).is_err(), "缺尾部校验和必须拒绝");
    }

    #[test]
    fn decompress_accepts_truly_compressed_stream() {
        // **迁移回归守卫**：旧 `prism-io::exr::zlib_decompress` 用
        // `data.len() < 2 + out.len()` 判截断，会把「压缩后比原文短」的流
        // 全部误判为截断，故旧实现根本解不开这条流。
        let z = zlib_fixed_rle259();
        let payload = vec![b'A'; 259];
        // 前置断言：这条流确实「压缩过」（总长远小于 2 + 259）。
        assert!(
            z.len() < HEADER_LEN + payload.len(),
            "该流必须是压缩过的（迁移前会被误拒）：{} vs {}",
            z.len(),
            HEADER_LEN + payload.len()
        );
        assert_eq!(decompress(&z).expect("真正压缩过的 zlib 流必须能解"), payload);
        assert_eq!(decompress_verified(&z).expect("真正压缩过的 zlib 流必须能解+校验"), payload);
    }

    #[test]
    fn decompress_header_only_is_rejected() {
        assert!(decompress(&[0x78, 0x01]).is_err(), "只有头的流必须拒绝");
    }

    // -- 4. 校验和路径 ---------------------------------------------------

    #[test]
    fn verified_detects_adler_mismatch() {
        let payload = b"hi".to_vec();
        let mut z = zlib_stored(&payload);
        let n = z.len();
        z[n - 1] ^= 0xff;
        // 不校验的入口照旧接受（保持迁移前行为）。
        assert_eq!(decompress(&z).expect("decompress 不校校验和"), payload);
        // 校验入口必须抓到。
        let e = decompress_verified(&z).expect_err("校验入口须抓到 Adler-32 不符");
        assert!(e.problem.contains("Adler-32"), "问题段须点名 Adler-32：{}", e.problem);
    }

    #[test]
    fn verified_rejects_short_stream() {
        assert!(decompress_verified(&[0x78, 0x01, 0x01, 0x00, 0x00, 0xff, 0xff]).is_err());
    }

    #[test]
    fn zlib_container_errors_are_located_in_zlib_module() {
        // 三段式一【在哪】必须落在 zlib 模块（便于按域聚合日志）。
        let e = decompress(&[0x78]).expect_err("短头");
        assert_eq!(e.location, "zlib::parse_header");
    }

    #[test]
    fn invalid_deflate_body_inside_zlib_is_reported() {
        // 合法 zlib 头 + 非法块类型 3（BFINAL=1, BTYPE=11）。
        let mut s = vec![0x78, 0x9C];
        s.push(0b0000_0111);
        s.extend_from_slice(&[0, 0, 0, 0]);
        assert!(decompress(&s).is_err(), "保留块类型应拒绝");
    }

    #[test]
    fn stored_len_nlen_mismatch_inside_zlib_rejected() {
        // zlib 头 + stored 块 LEN/NLEN 不互补。
        let mut s = vec![0x78, 0x9C];
        s.push(0x01);
        s.extend_from_slice(&4u16.to_le_bytes());
        s.extend_from_slice(&9999u16.to_le_bytes());
        s.extend_from_slice(&[1, 2, 3, 4]);
        assert!(decompress(&s).is_err(), "LEN/NLEN 不互补应拒绝");
    }

    // -- 9. 宽容口径（`decompress_lenient`）= 迁移不变性守卫 ---------------
    #[test]
    fn lenient_accepts_stream_without_trailer() {
        // **迁移不变性**：旧 `prism-media::inflate::zlib_decompress` 接收
        // 不带 Adler-32 尾的流（仓库里两个 PNG 测试构造器就是这么造的）。
        // 迁移后必须仍接收，否则那些测试当场回归。
        let payload = b"PNG raw scanline data";
        assert_eq!(
            decompress_lenient(&zlib_stored_no_trailer(payload)).expect("宽容入口须接收无尾流"),
            payload
        );
    }

    #[test]
    fn lenient_also_accepts_stream_with_trailer() {
        // 有尾也必须能解（宽容 = 不检查，不是不接受）。
        let payload = b"payload";
        assert_eq!(decompress_lenient(&zlib_stored(payload)).expect("有尾流"), payload);
    }

    #[test]
    fn lenient_still_validates_header() {
        // 宽容只放宽**尾部**，头部五项校验一项都不能松。
        assert!(decompress_lenient(&[0x78]).is_err(), "短头须拒绝");
        assert!(decompress_lenient(&[0x78, 0x00]).is_err(), "坏 FCHECK 须拒绝");
        assert!(decompress_lenient(&[0x77, 0x8B]).is_err(), "CM!=8 须拒绝");
        assert!(decompress_lenient(&[0x78, 0xBB]).is_err(), "FDICT 须拒绝");
    }

    #[test]
    fn lenient_reports_body_errors_under_its_own_location() {
        // 合法头 + 保留块类型 3 → 报错，且【在哪】落在 lenient 名下。
        let mut s = vec![0x78, 0x01];
        s.push(0b0000_0111);
        let e = decompress_lenient(&s).expect_err("保留块类型应拒绝");
        assert_eq!(e.location, "zlib::decompress_lenient");
    }

    #[test]
    fn strict_and_lenient_differ_exactly_on_trailer_presence() {
        // 两种口径的唯一差别：无尾流 strict 拒、lenient 收；有尾流两者同结果。
        let payload = b"x";
        let bare = zlib_stored_no_trailer(payload);
        let full = zlib_stored(payload);
        assert!(decompress(&bare).is_err(), "strict 必须要求尾部");
        assert!(decompress_lenient(&bare).is_ok(), "lenient 必须接收");
        assert_eq!(decompress(&full).expect("strict"), decompress_lenient(&full).expect("lenient"));
    }

    // -- 10. 迁移修掉的缺陷（回归守卫）-----------------------------------

    /// **旧 `prism_io::exr::zlib_decompress` 的截断误判**。
    ///
    /// 旧判据是 `data.len() < 2 + out.len()`——拿**压缩后总长**比**解压后长度**，
    /// 于是任何压缩率 > 0 的流（真实世界里一切正常压缩数据）都被误判为「截断」。
    /// 本测试钉住修复：`zlib_fixed_rle259` 必须能解。
    #[test]
    fn migration_true_compressed_stream_is_not_mistaken_for_truncation() {
        let payload = vec![b'A'; 259];
        let z = zlib_fixed_rle259();
        // 前置：这条流确实被压缩过 → 旧判据必然误杀。
        assert!(
            z.len() < 2 + payload.len(),
            "该流必须被压缩过（{} < {}），否则本测试无意义",
            z.len(),
            2 + payload.len()
        );
        assert_eq!(decompress(&z).expect("压缩流不得被误判为截断"), payload);
        assert_eq!(decompress_lenient(&z).expect("宽容口径同样能解"), payload);
    }

    /// **旧 `prism_io::exr::zlib_decompress` 在 stored 流上的边界**。
    ///
    /// 旧判据 `data.len() < 2 + out.len()` 对**未压缩**流恰好成立
    /// （`2 + 5 + len + 4 >= 2 + len`），所以旧实现在 stored 路径上"碰巧能用"，
    /// 正是这一点掩盖了它在压缩流上的致命 bug。本测试同时钉住两条路径。
    #[test]
    fn migration_stored_stream_still_decodes_under_strict_gate() {
        let payload = b"stored payload, 29 bytes exactly!!";
        let z = zlib_stored(payload);
        assert_eq!(decompress(&z).expect("stored 流必须能解"), payload);
    }
}
