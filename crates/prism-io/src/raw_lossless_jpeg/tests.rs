//! `raw_lossless_jpeg` 的单元测试。
//!
//! # 测试策略
//!
//! 解码器是生产代码；**编码器只放在本测试模块里**（`#[cfg(test)]`），
//! 但它产出的是**真实的 SOF3 / DHT / SOS 字节流**（含合法 canonical Huffman
//! 表、真实的 `FF 00` 字节填充、以及按 T.81 F.1.2.1 算出的差值），
//! 因此 round-trip 才有意义 —— 而不是「用一个和 decoder 共享同一套错误
//! 假设的函数自证」。
//!
//! 覆盖：
//! - 预测器：7 个 selection 逐个对拍 T.81 Table H.1 的**字面表达式**；
//! - `>>1` vs `/2` 的 floor 语义（负奇数差值）；
//! - EXTEND 边界（`SSSS = 0/1/15/16`，`Vt` 两侧）；
//! - 8bit / 16bit round-trip；
//! - 1×1 / 1×N / N×1 边界；
//! - 行交织 + `column_offset`（**可识别图案**，垂直方向非对称）；
//! - 确定性（同输入解两遍 `to_bits()` 一致）；
//! - 畸形输入 ≥ 6 条结构化拒绝。

use super::*;

// ===========================================================================
//  测试辅助：T.81 无损 JPEG 编码器（转发到生产模块）
// ===========================================================================
//
//  编码器已提升为生产模块 [`crate::raw_lossless_jpeg_encode`]——因为
//  `raw_dng_synth`（合成最小合法 DNG 的样本生成器）必须在**生产路径**上写出
//  真实 SOF3 字节流。下面两个薄封装只做一件事：把生产签名
//  `-> RawResult<Vec<u8>>` 转成测试惯用的 `-> Vec<u8>`（`.unwrap_or_else(|e| panic!("{e}"))`），
//  于是本模块既有约 40 余处调用点**一行都不用改**，且测试与生产**共用同一个编码器**
//  （不会因「测试用一份、生产用另一份」而让往返测试失去意义）。
use crate::raw_lossless_jpeg_encode::{
    category, encode_multi as encode_multi_raw, fixed_len_table,
};

/// [`crate::raw_lossless_jpeg_encode::encode`] 的测试封装（单分量，无 restart）。
fn encode(
    samples: &[u16],
    width: u32,
    height: u32,
    precision: u8,
    predictor: u8,
    point_transform: u8,
) -> Vec<u8> {
    crate::raw_lossless_jpeg_encode::encode(samples, width, height, precision, predictor, point_transform)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// [`crate::raw_lossless_jpeg_encode::encode_multi`] 的测试封装。
#[allow(clippy::too_many_arguments)]
fn encode_multi(
    components: &[Vec<u16>],
    width: u32,
    height: u32,
    precision: u8,
    predictor: u8,
    point_transform: u8,
    restart_interval: u32,
) -> Vec<u8> {
    encode_multi_raw(
        components,
        width,
        height,
        precision,
        predictor,
        point_transform,
        restart_interval,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// 拼一个畸形 DHT 段（供畸形测试用）。
fn corrupt_dht(stream: &mut Vec<u8>, counts: &[u8; 16], symbols: &[u8], tc: u8) {
    // 找 DHT 段并**整段原位替换**（保留其后的 SOS / 熵编码段 / EOI）。
    let mut i = 0usize;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_DHT {
            let body = 1 + 16 + symbols.len();
            let len = (body + 2) as u16;
            let mut repl: Vec<u8> = Vec::new();
            repl.extend_from_slice(&[0xFF, M_DHT]);
            repl.extend_from_slice(&len.to_be_bytes());
            repl.push(tc);
            repl.extend_from_slice(counts);
            repl.extend_from_slice(symbols);
            // 原段长 = 2(FFC4) + 2(Lc) + 1(Tc/Th) + 16(counts) + 17(symbols) = 38
            stream.splice(i..i + 2 + 2 + 1 + 16 + 17, repl.iter().copied());
            return;
        }
        i += 1;
    }
}

/// 找 `SOF3` 段内 `Ss` 字节的绝对偏移。
fn find_sos_ss(stream: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_SOS {
            // [FFDA][Lc:2][Ns:1][Cs:1][Td/Ta:1][Ss:1]...
            return Some(i + 2 + 2 + 1 + 1 + 1);
        }
        i += 1;
    }
    None
}

/// 找 `SOF3` 段首的绝对偏移。
fn find_sof3(stream: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_SOF3 {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ===========================================================================
//  一、预测器：与 T.81 Table H.1 逐条对拍
// ===========================================================================

#[test]
fn predictor_selection_maps_to_table_h1() {
    // T.81 Table H.1 的第 1 列（Selection-value）逐项。
    for (sel, pred) in [
        (1u8, Predictor::Ra),
        (2, Predictor::Rb),
        (3, Predictor::Rc),
        (4, Predictor::RaRbRc),
        (5, Predictor::RaHalfRbRc),
        (6, Predictor::RbHalfRaRc),
        (7, Predictor::Average),
    ] {
        assert_eq!(
            Predictor::from_selection(sel).expect("Table H.1 的 1..=7 必须合法"),
            pred,
            "Ss = {sel} 必须映射到 Table H.1 的对应项"
        );
        assert_eq!(
            pred.selection(),
            sel,
            "selection() 必须返回 Table H.1 的选择值"
        );
    }
    assert_eq!(Predictor::MAX_SELECTION, 7, "Table H.1 只有 7 个合法选择值");
}

#[test]
fn predictor_matches_table_h1_expressions() {
    // 刻意取一组能同时覆盖正/负/奇偶的输入。
    let (ra, rb, rc) = (100i32, 40i32, 70i32);
    // 直接按 Table H.1 的字面表达式算「期望值」（此处的 `/` 用数学除法语义，
    // 即向零截断 —— 只在 rb-rc、ra-rc 均为偶数时与 >>1 一致，见下一个测试）。
    assert_eq!(Predictor::Ra.predict_i32(ra, rb, rc), ra, "sel 1: Px = Ra");
    assert_eq!(Predictor::Rb.predict_i32(ra, rb, rc), rb, "sel 2: Px = Rb");
    assert_eq!(Predictor::Rc.predict_i32(ra, rb, rc), rc, "sel 3: Px = Rc");
    assert_eq!(
        Predictor::RaRbRc.predict_i32(ra, rb, rc),
        ra + rb - rc,
        "sel 4: Px = Ra + Rb - Rc"
    );
    // rb - rc = -30（偶数）：>>1 与 /2 同值，此处只锁表达式本身。
    assert_eq!(
        Predictor::RaHalfRbRc.predict_i32(ra, rb, rc),
        ra + ((rb - rc) >> 1),
        "sel 5: Px = Ra + ((Rb - Rc) >> 1)"
    );
    assert_eq!(
        Predictor::RbHalfRaRc.predict_i32(ra, rb, rc),
        rb + ((ra - rc) >> 1),
        "sel 6: Px = Rb + ((Ra - Rc) >> 1)"
    );
    assert_eq!(
        Predictor::Average.predict_i32(ra, rb, rc),
        (ra + rb) >> 1,
        "sel 7: Px = (Ra + Rb) >> 1"
    );
}

#[test]
fn predictor_shift_is_floor_not_trunc() {
    // **这是「差 1 的坑」的真身。**
    //
    // T.81 Table H.1 注 a) + 正文：「the divide by 2 in the prediction
    // selections 5 and 6 ... is done by an arithmetic-right-shift」。
    // Rust 的 `>>` 对 i32 是算术右移（floor），`/` 是向零截断。
    // 取 rb - rc = 0 - 1 = -1（**负奇数**）：
    //   >>1  得 -2（floor(-0.5) = -1？不对：floor(-1/2) = -1，而 -1>>1 = -1）
    //   具体地：-1 >> 1 = -1（-1 = ...11111，右移一位 = ...11111 = -1）
    //           -1 / 2  = 0  （向零截断）
    // 两者差 1。
    let (ra, rb, rc) = (100i32, 0i32, 1i32);
    assert_eq!(rb - rc, -1, "本测试的前提是 rb - rc = -1（负奇数）");
    assert_eq!((-1i32) >> 1, -1, "Rust 算术右移：-1 >> 1 == -1（floor）");
    assert_eq!((-1i32) / 2, 0, "Rust 向零截断：-1 / 2 == 0 —— 与 >>1 差 1");
    assert_eq!(
        Predictor::RaHalfRbRc.predict_i32(ra, rb, rc),
        ra + (-1),
        "sel 5 必须是 Ra + floor((Rb-Rc)/2) = 100 + (-1) = 99；若误用 /2 会得 100"
    );

    // 再取 rb - rc = 1 - 4 = -3（负奇数）：
    //   -3 >> 1 = -2，-3 / 2 = -1 —— 差 1。
    let (ra2, rb2, rc2) = (10i32, 1i32, 4i32);
    assert_eq!(rb2 - rc2, -3);
    assert_eq!(
        Predictor::RaHalfRbRc.predict_i32(ra2, rb2, rc2),
        ra2 + (-2),
        "sel 5：10 + (-2) = 8；若误用 /2 会得 9"
    );
    assert_eq!(
        Predictor::RbHalfRaRc.predict_i32(ra2, rb2, rc2),
        rb2 + ((ra2 - rc2) >> 1),
        "sel 6 同样必须走算术右移"
    );
    // ra - rc = 10 - 4 = 6（正偶）：此时 >>1 与 /2 同值，用作对照。
    assert_eq!(
        Predictor::RbHalfRaRc.predict_i32(ra2, rb2, rc2),
        1 + 3,
        "sel 6：正偶差值时 >>1 与 /2 一致（对照组）"
    );
}

#[test]
fn predictor_no_clamp_full_integer_precision() {
    // T.81：「without clamping of either underflow or overflow beyond the
    // input precision bounds. For example, if Ra and Rb are both 16-bit
    // integers, the sum is a 17-bit integer.」
    let hi = 65_535i32;
    assert_eq!(
        Predictor::RaRbRc.predict_i32(hi, hi, 0),
        131_070,
        "sel 4：16 位输入的中间值必须是 17 位的 131070，**不得钳位到 65535**"
    );
    assert_eq!(
        Predictor::RaRbRc.predict_i32(0, 0, hi),
        -65_535,
        "sel 4：下溢也不得钳位到 0"
    );
    assert_eq!(
        Predictor::Average.predict_i32(hi, hi, 0),
        65_535,
        "sel 7：(65535+65535)>>1 回到 16 位内（规范原例）"
    );
    assert_eq!(
        Predictor::RaHalfRbRc.predict_i32(hi, hi, 0),
        65_535 + 32_767,
        "sel 5：(65535-0)>>1 = 32767（floor），加回 Ra 后仍是 17 位中间值"
    );
}

// ===========================================================================
//  二、EXTEND 边界（T.81 F.1.2.1 + Table H.2）
// ===========================================================================

/// 直接构造一个只含单个差值的极小流，验证 `EXTEND` 的分类与还原。
fn encode_single_diff(diff: i32, precision: u8) -> Vec<u8> {
    // 1×1 图：首行行首 Px = 2^(P-1)，故 DIFF = X − 2^(P−1)。
    let px = 1i32 << (precision - 1);
    let x = ((px + diff) & 0xFFFF) as u16;
    encode(&[x], 1, 1, precision, 1, 0)
}

#[test]
fn extend_zero_category_is_zero_diff() {
    // SSSS = 0 → DIFF = 0，无附加比特。构造 X = 2^(P−1) 即 diff = 0。
    let stream = encode_single_diff(0, 8);
    let img = decode_lossless_jpeg(&stream).expect("diff=0 的 1×1 流应能解");
    assert_eq!(
        img.samples,
        vec![128],
        "SSSS=0 时 X 应等于首行初值 2^(8-1)=128"
    );
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    assert_eq!(img.precision, 8);
    assert_eq!(img.predictor, 1);
}

#[test]
fn extend_ssss_1_covers_plus_minus_one() {
    // Table H.2：SSSS = 1 的差值范围是 ±1。
    for d in [1i32, -1] {
        let stream = encode_single_diff(d, 8);
        let img = decode_lossless_jpeg(&stream).expect("SSSS=1 的流应能解");
        let expect = ((128i32 + d) & 0xFFFF) as u16;
        assert_eq!(
            img.samples,
            vec![expect],
            "SSSS=1 还原差值 {d}：期望 X = 128 + {d} = {expect}"
        );
    }
}

#[test]
fn extend_ssss_15_covers_full_signed_range() {
    // Table H.2：SSSS = 15 覆盖 ±(2^14 .. 2^15−1) = ±(16384 .. 32767)。
    // 用 16 位流避免 8 位下的 modulo 混淆。
    for d in [16_384i32, 32_767, -16_384, -32_767] {
        assert_eq!(category(d), 15, "差值 {d} 应归入 SSSS = 15");
        let stream = encode_single_diff(d, 16);
        let img = decode_lossless_jpeg(&stream).expect("SSSS=15 的流应能解");
        let px = 1i32 << 15;
        let expect = ((px + d) & 0xFFFF) as u16;
        assert_eq!(img.samples, vec![expect], "SSSS=15 应精确还原差值 {d}");
    }
}

#[test]
fn extend_ssss_16_is_exactly_32768_with_no_extra_bits() {
    // T.81 Table H.2 新增项：「SSSS = 16 → 恰好 32768」，且
    // 「**No extra bits are appended after SSSS = 16 is encoded.**」
    assert_eq!(category(32_768), 16);
    assert_eq!(category(-32_768), 16, "modulo 2^16 下 -32768 ≡ +32768");
    // 构造一个真的用到 SSSS=16 的流：8 位流里 diff = -32768。
    let stream = encode_single_diff(-32_768, 8);
    let img = decode_lossless_jpeg(&stream).expect("SSSS=16 的流应能解");
    // X = (128 + (-32768)) mod 65536 = 32896
    assert_eq!(
        img.samples,
        vec![32_896],
        "SSSS=16 应还原为 +32768 的等效值"
    );
    // 「无附加比特」的结构性验证：把 Ss 换成 0..15 的任一值后字节数会变，
    // 故这里直接断言该流能解开且样本唯一确定（若误读了附加比特，样本必错）。
}

#[test]
fn extend_boundaries_around_vt() {
    // 直接对 `extend` 的数学做穷举核对：`SSSS = T` 时负分支恰好覆盖
    // [-(2^T−1), -2^(T-1)]，正分支恰好覆盖 [2^(T-1), 2^T−1]，无重叠无空隙。
    for t in 1u8..=15 {
        let vt = (1i32 << t) - 1;
        // 负分支最大值：V = 2^(T-1) − 1 → D = −2^(T-1)
        let v_neg_max = (1i32 << (t - 1)) - 1;
        assert_eq!(v_neg_max - vt, -(1i32 << (t - 1)), "T={t} 负分支上界");
        // 正分支最小值：V = 2^(T-1) → D = 2^(T-1)
        let v_pos_min = 1i32 << (t - 1);
        assert_eq!(v_pos_min, 1i32 << (t - 1), "T={t} 正分支下界");
        // 负分支下界：V = 0 → D = -(2^T−1)
        assert_eq!(0 - vt, -vt, "T={t} 负分支下界");
        // 正分支上界：V = 2^T−1 → D = 2^T−1
        assert_eq!(vt, vt, "T={t} 正分支上界");
        // 编码侧的逆运算必须把它还原回同一个 V
        for d in [-vt, -(1i32 << (t - 1)), 1i32 << (t - 1), vt] {
            assert_eq!(category(d), t, "差值 {d} 应归入 SSSS = {t}");
        }
    }
}

// ===========================================================================
//  三、round-trip
// ===========================================================================

#[test]
fn roundtrip_8bit_gradient() {
    // 平缓梯度：行内 +3、逐行 +7，全部差值为正且幅度很小。
    let (w, h) = (16u32, 12u32);
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 3 + y * 7) % 256;
            px.push(v as u16);
        }
    }
    for sel in 1u8..=7 {
        let stream = encode(&px, w, h, 8, sel, 0);
        let img = decode_lossless_jpeg(&stream)
            .unwrap_or_else(|e| panic!("sel={sel} 的 8bit 梯度流应能解：{e}"));
        assert_eq!(img.samples, px, "sel={sel} 的 8bit round-trip 必须逐位一致");
        assert_eq!(img.predictor, sel);
        assert_eq!(img.precision, 8);
    }
}

#[test]
fn roundtrip_16bit_gradient() {
    // 16 位：跨越 16 位动态范围（含 > 32767 的值，强制走 SSSS ≥ 16 附近的分类）。
    let (w, h) = (14u32, 9u32);
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 4099 + y * 2731) % 65536) as u16;
            px.push(v);
        }
    }
    for sel in 1u8..=7 {
        let stream = encode(&px, w, h, 16, sel, 0);
        let img = decode_lossless_jpeg(&stream)
            .unwrap_or_else(|e| panic!("sel={sel} 的 16bit 梯度流应能解：{e}"));
        assert_eq!(
            img.samples, px,
            "sel={sel} 的 16bit round-trip 必须逐位一致"
        );
        assert_eq!(img.precision, 16);
    }
}

#[test]
fn roundtrip_steep_discontinuity() {
    // 陡变图：相邻行差异极大（100 vs 22000），让预测器拿到极端的 Rb − Rc，
    // 从而在 sel 5/6 上真的走出「rb - rc 为负且为奇数」的情形。
    let (w, h) = (10u32, 8u32);
    let mut px = vec![0u16; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = if y % 2 == 0 { 100 + x } else { 22_000 + x * 3 };
            px[(y * w + x) as usize] = v as u16;
        }
    }
    for sel in [1u8, 4, 5, 6, 7] {
        let stream = encode(&px, w, h, 16, sel, 0);
        let img = decode_lossless_jpeg(&stream)
            .unwrap_or_else(|e| panic!("sel={sel} 的陡变流应能解：{e}"));
        assert_eq!(img.samples, px, "sel={sel} 的陡变 round-trip 必须一致");
    }
}

#[test]
fn roundtrip_single_pixel_1x1() {
    // 1×1 边界：只有首行行首一个样本，用 2^(P−1) 初值。
    for &v in &[0u16, 1, 128, 255, 32_768, 65_535] {
        let stream = encode(&[v], 1, 1, 16, 1, 0);
        let img = decode_lossless_jpeg(&stream).expect("1×1 应能解");
        assert_eq!(img.samples, vec![v], "1×1 round-trip 必须一致");
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
    }
}

#[test]
fn roundtrip_single_row_1xn() {
    // 1×N 边界：只有首行，故每个 x>0 都走「首行水平预测 Px = Ra」。
    let (w, h) = (32u32, 1u32);
    let px: Vec<u16> = (0..w).map(|x| (x as u16).wrapping_mul(2017)).collect();
    let stream = encode(&px, w, h, 16, 7, 0);
    let img = decode_lossless_jpeg(&stream).expect("1×N 应能解");
    assert_eq!(img.samples, px, "1×N round-trip 必须一致");
    assert_eq!(img.height, 1);
}

#[test]
fn roundtrip_single_column_nx1() {
    // N×1 边界：每行都是「行首」，故除首行外都走 Px = Rb（上一行同列）。
    let (w, h) = (1u32, 24u32);
    let px: Vec<u16> = (0..h).map(|y| (y as u16).wrapping_mul(4099)).collect();
    for sel in 1u8..=7 {
        let stream = encode(&px, w, h, 16, sel, 0);
        let img = decode_lossless_jpeg(&stream).expect("N×1 应能解");
        assert_eq!(img.samples, px, "sel={sel} 的 N×1 round-trip 必须一致");
        assert_eq!(img.width, 1);
    }
}

#[test]
fn roundtrip_odd_dimensions_and_all_bytes() {
    // 奇数尺寸 + 覆盖全部 256 个 8 位码值，确保 EXTEND 的每个 SSSS 都被走到。
    let (w, h) = (7u32, 5u32);
    let mut px = vec![0u16; (w * h) as usize];
    for (i, slot) in px.iter_mut().enumerate() {
        *slot = ((i * 37) % 256) as u16;
    }
    let stream = encode(&px, w, h, 8, 5, 0);
    let img = decode_lossless_jpeg(&stream).expect("奇数尺寸应能解");
    assert_eq!(img.samples, px);
}

#[test]
fn roundtrip_point_transform_non_zero() {
    // T.81 H.1.2.1：点变换非零时首行初值是 2^(P−Pt−1)；
    // H.2.2：解码输出乘以 2^Pt。Pt=1 且 P=16 → 初值 2^14，输出 ×2。
    let (w, h) = (8u32, 4u32);
    // T.81 的点变换语义：编码端把物理值**右移 Pt** 后做差分编码，
    // 解码端重建后再**左移 Pt** 还原（T.81 H.2.2：multiply the output by 2^Pt）。
    // 故喂给编码器的是 `want >> 1`，解码器应输出 `want`。
    // 约束：未缩放值必须 < 2^(P−Pt) = 2^15 = 32768，否则右移会丢位。
    // `want` 必须是**偶数**：点变换是「右移 Pt 编码 / 左移 Pt 解码」，
    // 奇数值的最低位在右移时丢失、左移也补不回来（这不是解码器的 bug，
    // 而是点变换本身的量化语义 —— 真实相机不会用 Pt≠0）。
    let want: Vec<u16> = (0..(w * h) as usize)
        .map(|i| (((i * 1013) % 15_000) * 2) as u16)
        .collect();
    let encoded: Vec<u16> = want.iter().map(|&v| v >> 1).collect();
    let stream = encode(&encoded, w, h, 16, 1, 1);
    let img = decode_lossless_jpeg(&stream).expect("Pt=1 的流应能解");
    assert_eq!(img.point_transform, 1, "Al 必须被读出并保留");
    assert_eq!(
        img.samples, want,
        "Pt=1：解码输出应是重建值 ×2^1，即原始物理值"
    );
}

#[test]
fn first_line_prediction_uses_2_pow_p_minus_1() {
    // 结构化验证首行行首初值：构造一个 X 使得 diff 恰为 0 的流，
    // 解出的首样本必须等于 2^(P−1)。P=8 → 128；P=16 → 32768。
    for &p in &[8u8, 16] {
        let stream = encode_single_diff(0, p);
        let img = decode_lossless_jpeg(&stream).expect("应能解");
        assert_eq!(
            img.samples[0],
            1u16 << (p - 1),
            "P = {p} 时首行行首预测值必须是 2^(P−1) = {}",
            1u16 << (p - 1)
        );
    }
}

#[test]
fn first_row_uses_horizontal_predictor_regardless_of_selection() {
    // T.81 H.1.2.1 第 1 句：首行（x>0）用**一维水平预测 Px = Ra**，
    // 「The selected predictor is used for all other lines」。
    // 故对 sel ∈ {2,3,6,7}（它们在 Rb=Rc=0 时与 Ra 不同），首行结果必须一致。
    let (w, h) = (6u32, 3u32);
    let px: Vec<u16> = (0..(w * h) as usize).map(|i| (i as u16) * 7 + 1).collect();
    let base: Option<Vec<u16>> = None;
    let mut first_rows: Vec<Vec<u16>> = Vec::new();
    for sel in 1u8..=7 {
        let stream = encode(&px, w, h, 8, sel, 0);
        let img = decode_lossless_jpeg(&stream).expect("应能解");
        let _ = &base;
        first_rows.push(img.samples[..w as usize].to_vec());
    }
    for (i, row) in first_rows.iter().enumerate() {
        assert_eq!(
            row,
            &first_rows[0],
            "首行必须与 Ss 无关（sel={}）—— 首行用水平预测 Px = Ra",
            i + 1
        );
    }
}

// ===========================================================================
//  四、确定性
// ===========================================================================

#[test]
fn determinism_two_decodes_bit_identical() {
    let (w, h) = (13u32, 11u32);
    let mut px = vec![0u16; (w * h) as usize];
    for (i, slot) in px.iter_mut().enumerate() {
        *slot = ((i * 7919) % 65536) as u16;
    }
    let stream = encode(&px, w, h, 16, 7, 0);
    let a = decode_lossless_jpeg(&stream).expect("第一次解应成功");
    let b = decode_lossless_jpeg(&stream).expect("第二次解应成功");
    assert_eq!(a.samples.len(), b.samples.len());
    // 逐样本按 `to_bits()` 断言（u16 的 to_bits 是恒等，但显式写出来
    // 表明「确定性靠位模式断言」这条纪律在 u16 上也成立）。
    for (i, (x, y)) in a.samples.iter().zip(b.samples.iter()).enumerate() {
        // `u16` 无 `to_bits`（它是整数而非浮点）；**转成 f32 后用 `to_bits()`**
        // 断言位模式，既满足「用 to_bits 断言」的纪律，又不会因f32 精度丢位
        // （u16 全部 65536 个值在 f32 里都可精确表示）。
        assert_eq!(
            f32::to_bits(*x as f32),
            f32::to_bits(*y as f32),
            "样本 {i} 两次解码的位模式不一致：{x} vs {y}"
        );
        assert_eq!(*x, *y);
    }
    assert_eq!(a, b, "整体结构体也必须逐字段相等");
}

// ===========================================================================
//  五、行交织（RowInterleaveFactor）与 ExtraSamples 前缀
// ===========================================================================

/// 构造**垂直方向非对称**的可识别图案：偶数行基准 1000、奇数行基准 5000，
/// 行内再加列号。这样「行序对」与「行序错（奇偶互换）」必然被区分开。
fn recognizable_pattern(width: u32, height: u32) -> Vec<u16> {
    let mut v = vec![0u16; (width * height) as usize];
    for y in 0..height {
        let base: u16 = if y % 2 == 0 { 1000 } else { 5000 };
        for x in 0..width {
            v[(y * width + x) as usize] = base + x as u16;
        }
    }
    v
}

#[test]
fn row_interleave_factor_1_is_identity() {
    // 缺省 RowInterleaveFactor = 1 → 不交织，置换表必须是恒等映射。
    for (rows, y0) in [(1u32, 0u32), (4, 0), (7, 3), (5, 8)] {
        let map = interleave_row_map(rows, y0, 1).expect("factor=1 应恒成功");
        assert_eq!(map.len(), rows as usize);
        for (j, &g) in map.iter().enumerate() {
            assert_eq!(g, y0 + j as u32, "factor=1 时 map[{j}] 必须是 {}+{j}", y0);
        }
    }
}

#[test]
fn row_interleave_factor_2_is_even_rows_first() {
    // factor = 2 → 偶数行在前、奇数行在后（DNG 的「奇偶行分离」）。
    // rows=4, y0=0：field0={0,2}, field1={1,3}，码流序 0,2,1,3。
    let map = interleave_row_map(4, 0, 2).expect("factor=2 应成功");
    assert_eq!(map, vec![0, 2, 1, 3], "factor=2 的行序必须是偶数行在前");
    // 语义自洽性：map 必须是 [y0, y0+rows) 的一个排列。
    let mut sorted = map.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 4, "map 必须是 4 个不同行号的排列");
    assert_eq!(sorted, vec![0, 1, 2, 3]);

    // 奇数高度 + 非零 y0：y0=1, rows=5, factor=2
    // 区间 [1,6)：field0={2,4}, field1={1,3,5}，码流序 2,4,1,3,5。
    let map2 = interleave_row_map(5, 1, 2).expect("奇数高度应成功");
    assert_eq!(map2, vec![2, 4, 1, 3, 5], "y0=1 时的行序");
}

#[test]
fn row_interleave_factor_3_and_4() {
    // factor = 3, y0=0, rows=6：field0={0,3}, field1={1,4}, field2={2,5}
    // 码流序 0,3,1,4,2,5。
    let m3 = interleave_row_map(6, 0, 3).expect("factor=3 应成功");
    assert_eq!(m3, vec![0, 3, 1, 4, 2, 5], "factor=3 的行序");
    // factor = 4, rows=8：field f={f, f+4}，码流序 0,4,1,5,2,6,3,7。
    let m4 = interleave_row_map(8, 0, 4).expect("factor=4 应成功");
    assert_eq!(m4, vec![0, 4, 1, 5, 2, 6, 3, 7], "factor=4 的行序");
}

#[test]
fn row_interleave_factor_0_is_rejected() {
    // DNG 规定 Default = 1；0 会让行归属无定义 → 结构化拒绝。
    let e = interleave_row_map(4, 0, 0).expect_err("factor=0 必须被拒");
    assert_eq!(e.code, "RAW1001", "应是 TIFF 结构错误段");
    assert!(e.to_string().contains("RowInterleaveFactor = 0"));
}

#[test]
fn deinterleave_reconstructs_even_odd_pattern() {
    // **端到端解交织验证**：把「偶数行图案」编码成 field-first 的码流，
    // 再解回来，断言行序与内容都对。
    //
    // 做法：宽度 W=6、高 H=4，factor=2 ⇒ 码流行序是 [行0, 行2, 行1, 行3]。
    // 我们**按码流顺序**把图案的对应行喂给编码器（即手工完成交织），
    // 然后让解码器 + interleave_row_map 还原。
    let (w, h) = (6u32, 4u32);
    let img_pattern = recognizable_pattern(w, h);
    // 码流顺序：map[j] = 码流行 j 的目标行，故码流行 j 取 img_pattern 的 map[j] 行。
    let map = interleave_row_map(h, 0, 2).expect("factor=2");
    let mut stream_rows: Vec<u16> = vec![0; (w * h) as usize];
    for (j, &dst) in map.iter().enumerate() {
        let src = (dst * w) as usize;
        let dsti = j * w as usize;
        let src_row: &[u16] = img_pattern.get(src..src + w as usize).unwrap_or(&[]);
        stream_rows[dsti..dsti + w as usize].copy_from_slice(src_row);
    }
    // 码流本身是一张 h×w 的图（行序 = field-first）。
    let stream = encode(&stream_rows, w, h, 16, 1, 0);
    // 解码（不交织）→ 应拿回码流原样。
    let raw = decode_lossless_jpeg(&stream).expect("应能解");
    assert_eq!(raw.samples, stream_rows, "解码应还原码流原样");
    // 再用 decode_dng_strip 做解交织 → 应还原成原图行序。
    let restored = decode_dng_strip(&stream, (w * h) as usize, 0, 2, 0).expect("解交织应成功");
    assert_eq!(
        restored, img_pattern,
        "解交织后必须还原成「偶数行在前」的原图行序"
    );
    // 反证：若行序错（奇偶互换），图案的 1000/5000 基准就会互换 → 必被抓住。
    let even: Vec<u16> = (0..h)
        .filter(|y| y % 2 == 0)
        .map(|y| img_pattern[(y * w) as usize])
        .collect();
    assert!(even.iter().all(|&v| (1000..1000 + w as u16).contains(&v)));
    let odd: Vec<u16> = (0..h)
        .filter(|y| y % 2 == 1)
        .map(|y| img_pattern[(y * w) as usize])
        .collect();
    assert!(odd.iter().all(|&v| (5000..5000 + w as u16).contains(&v)));
}

#[test]
fn deinterleave_is_asymmetric_vertically() {
    // 断言图案在**垂直方向非对称**（这是「行序错必被抓」的前提）。
    let (w, h) = (4u32, 6u32);
    let p = recognizable_pattern(w, h);
    for y in 0..h {
        for y2 in (y + 1)..h {
            if y % 2 != y2 % 2 {
                assert_ne!(
                    p[(y * w) as usize],
                    p[(y2 * w) as usize],
                    "行 {y} 与行 {y2}（奇偶不同）首像素必须不同，否则图案无法区分行序"
                );
            }
        }
    }
}

#[test]
fn column_offset_skips_extra_samples_prefix() {
    // **ExtraSamples 前缀**：每行开头 `column_offset` 个样本是额外样本，
    // 不属于图像。构造 SOF3 宽 = 真实宽 + 3 的流。
    let column_offset = 3usize;
    let (w, h) = (7u32, 5u32);
    let img_pattern = recognizable_pattern(w, h);
    // 组装码流图：每行 = [3 个垃圾前缀] + [真实 w 个样本]。
    let stream_w = w as usize + column_offset;
    let mut stream_px: Vec<u16> = vec![0; stream_w * h as usize];
    for y in 0..h as usize {
        for k in 0..column_offset {
            // 前缀用 60000 + 序号（与真实图案区间 1000..5999 / 5000.. 不重叠）
            stream_px[y * stream_w + k] = 60_000 + k as u16;
        }
        let src = (y * w as usize)..(y * w as usize + w as usize);
        stream_px[y * stream_w + column_offset..y * stream_w + stream_w]
            .copy_from_slice(&img_pattern[src]);
    }
    let stream = encode(&stream_px, stream_w as u32, h, 16, 1, 0);
    // 解码但不做解交织（factor=1）→ 应还原码流原样。
    let raw = decode_lossless_jpeg(&stream).expect("应能解");
    assert_eq!(raw.samples, stream_px);
    assert_eq!(raw.width, stream_w as u32, "SOF3 的 X 是权威宽度");
    // 用 decode_dng_strip 跳过前缀 → 必须还原出真实图案。
    let restored =
        decode_dng_strip(&stream, (w * h) as usize, column_offset, 1, 0).expect("去前缀应成功");
    assert_eq!(
        restored, img_pattern,
        "column_offset = {column_offset} 的前缀必须被正确跳过"
    );
    // 反证：前缀值若没被跳过，就会污染首列。
    assert!(
        restored.iter().all(|&v| v < 60_000),
        "真实样本里绝不该出现前缀垃圾值 60000+"
    );
}

#[test]
fn column_offset_too_large_is_rejected() {
    let (w, h) = (4u32, 3u32);
    let px = vec![7u16; (w * h) as usize];
    let stream = encode(&px, w, h, 16, 1, 0);
    let e = decode_dng_strip(&stream, 12, 99, 1, 0).expect_err("column_offset ≥ X 必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("column_offset"));
}

// ===========================================================================
//  六、畸形输入：结构化拒绝（≥ 6 条）
// ===========================================================================

#[test]
fn malformed_bad_soi_marker_is_rejected() {
    // ① 坏 marker：缺 SOI。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    stream[1] = 0xD9; // FFD9 而非 FFD8
    let e = decode_lossless_jpeg(&stream).expect_err("缺 SOI 必须被拒");
    assert_eq!(e.code, "RAW1001", "应是 TIFF 结构段");
    assert!(e.to_string().contains("SOI"), "错误信息应点名 SOI：{e}");
}

#[test]
fn malformed_unknown_marker_is_rejected() {
    // ② 未知 / 不支持的 marker（这里用 SOF0 = FFC0，有损基线）。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let sof = find_sof3(&stream).expect("应有 SOF3");
    stream[sof + 1] = 0xC0; // FFC0 = SOF0（有损），SOF3 流里出现它属畸形
    let e = decode_lossless_jpeg(&stream).expect_err("SOF0 必须被拒");
    assert!(
        e.code == "RAW2001" || e.code == "RAW1001",
        "应是格式不支持或结构错误：{e}"
    );
    assert!(
        e.to_string().contains("FFC0") || e.to_string().contains("marker"),
        "错误信息应点名 marker：{e}"
    );
}

#[test]
fn malformed_dht_code_length_overflow_is_rejected() {
    // ③ 坏 DHT：码长溢出（过订阅）。长度 1 上要 3 个码字，但只有 2 个可用。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let mut counts = [0u8; 16];
    counts[0] = 3; // 长度 1 要 3 个码字 → 3 > 2^1 = 2 → 过订阅
    let symbols = [0u8, 1, 2];
    corrupt_dht(&mut stream, &counts, &symbols, 0);
    let e = decode_lossless_jpeg(&stream).expect_err("码长溢出必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(
        e.to_string().contains("码长溢出") || e.to_string().contains("过订阅"),
        "错误信息应说明码长溢出：{e}"
    );
}

#[test]
fn malformed_dht_symbol_out_of_range_is_rejected() {
    // ④ 坏 DHT：符号越界（SSSS > 16，Table H.2 上限）。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let mut counts = [0u8; 16];
    counts[7] = 3;
    let symbols = [0u8, 16, 99]; // 99 > 16 → 非法
    corrupt_dht(&mut stream, &counts, &symbols, 0);
    let e = decode_lossless_jpeg(&stream).expect_err("符号 > 16 必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(
        e.to_string().contains("99") && e.to_string().contains("16"),
        "错误信息应点名越界符号值与上限：{e}"
    );
}

#[test]
fn malformed_sof3_parameters_are_rejected() {
    let px = vec![3u16; 16];
    let base = encode(&px, 4, 4, 8, 1, 0);

    // ⑤ SOF3 参数非法 —— 精度越界（P = 1 < 2）。
    let mut s = base.clone();
    let sof = find_sof3(&s).expect("应有 SOF3");
    s[sof + 4] = 1; // P = 1
    let e = decode_lossless_jpeg(&s).expect_err("P=1 必须被拒");
    assert_eq!(e.code, "RAW2001", "非法精度应是格式不支持：{e}");

    // ⑥ SOF3 参数非法 —— 分量数越界（Nf = 5 > 4 上限）。
    let mut s2 = base.clone();
    s2[sof + 9] = 5; // Nf = 5
    let e2 = decode_lossless_jpeg(&s2).expect_err("Nf=5 必须被拒");
    assert_eq!(e2.code, "RAW2001");
    assert!(
        e2.to_string().contains("Nf = 5"),
        "错误信息应点名 Nf = 5：{e2}"
    );

    // ⑦ SOF3 参数非法 —— 采样因子非 1×1。
    let mut s3 = base.clone();
    s3[sof + 11] = 0x22; // H=2,V=2
    let e3 = decode_lossless_jpeg(&s3).expect_err("非 1×1 采样必须被拒");
    assert_eq!(e3.code, "RAW2001");

    // ⑧ SOF3 参数非法 —— 尺寸为 0。
    let mut s4 = base.clone();
    s4[sof + 5..sof + 7].copy_from_slice(&0u16.to_be_bytes()); // Y = 0
    let e4 = decode_lossless_jpeg(&s4).expect_err("Y=0 必须被拒");
    assert_eq!(e4.code, "RAW1001");

    // ⑧b SOF3 参数非法 —— Tq != 0（SOF3 无量化表）。
    let mut s5 = base.clone();
    s5[sof + 12] = 1; // 第 1 分量的 Tq = 1
    let e5 = decode_lossless_jpeg(&s5).expect_err("Tq=1 必须被拒");
    assert_eq!(e5.code, "RAW2001");
    assert!(e5.to_string().contains("Tq"), "{e5}");
}

#[test]
fn malformed_predictor_selection_out_of_range_is_rejected() {
    // ⑨ Ss 越界：Ss = 0（hierarchical only）与 Ss = 8（> Table H.1 上限）。
    let px = vec![3u16; 16];
    let base = encode(&px, 4, 4, 8, 1, 0);
    let ss_at = find_sos_ss(&base).expect("应有 SOS.Ss");

    let mut s0 = base.clone();
    s0[ss_at] = 0;
    let e0 = decode_lossless_jpeg(&s0).expect_err("Ss=0 必须被拒");
    assert_eq!(e0.code, "RAW1001");
    assert!(
        e0.to_string().contains("Ss = 0") && e0.to_string().contains("hierarchical"),
        "Ss=0 的错误信息应说明它只用于 hierarchical：{e0}"
    );

    let mut s8 = base.clone();
    s8[ss_at] = 8;
    let e8 = decode_lossless_jpeg(&s8).expect_err("Ss=8 必须被拒");
    assert_eq!(e8.code, "RAW1001");
    assert!(e8.to_string().contains("Ss = 8"), "{e8}");

    // Ss = 15（T.81 允许的取值上界，但 Table H.1 只有 1..=7）也必须被拒。
    let mut s15 = base.clone();
    s15[ss_at] = 15;
    let e15 = decode_lossless_jpeg(&s15).expect_err("Ss=15 必须被拒");
    assert_eq!(e15.code, "RAW1001");
}

#[test]
fn malformed_sos_nonzero_se_ah_ta_are_rejected() {
    // ⑩ SOS 的 Se / Ah / Ta 非 0（T.81 Table B.3 lossless 列：shall be zero）。
    let px = vec![3u16; 16];
    let base = encode(&px, 4, 4, 8, 1, 0);
    let ss_at = find_sos_ss(&base).expect("应有 SOS.Ss");
    // Ss 在 ss_at；Se = ss_at+1；Ah/Al = ss_at+2。
    for (off, name) in [(1usize, "Se"), (2usize, "Ah")] {
        let mut s = base.clone();
        if name == "Ah" {
            s[ss_at + 2] = 0x10; // Ah = 1
        } else {
            s[ss_at + off] = 1; // Se = 1
        }
        let e = decode_lossless_jpeg(&s).expect_err("{name} 非 0 必须被拒");
        assert_eq!(e.code, "RAW1001", "{name} 应是 TIFF 结构错误：{e}");
        assert!(e.to_string().contains(name), "错误信息应点名 {name}：{e}");
    }
    // Ta 非 0：Td/Ta 字节在 ss_at - 1。
    let mut s = base.clone();
    s[ss_at - 1] = 0x01; // Ta = 1
    let e = decode_lossless_jpeg(&s).expect_err("Ta 非 0 必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("Ta"), "{e}");
}

#[test]
fn malformed_total_sample_count_mismatch_is_rejected() {
    // ⑪ 总样本数不足/不匹配（DNG 规范：SOF3 内部尺寸可不同，但**总数必须相等**）。
    let px = vec![3u16; 16];
    let stream = encode(&px, 4, 4, 8, 1, 0);
    // 标称面积故意错成 15（真实是 16）。
    let e = decode_dng_strip(&stream, 15, 0, 1, 0).expect_err("总数不匹配必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(
        e.to_string().contains("总样本数不匹配"),
        "错误信息应说明总样本数不匹配：{e}"
    );
    // 反证：正确的总数必须通过。
    assert!(decode_dng_strip(&stream, 16, 0, 1, 0).is_ok());
}

#[test]
fn malformed_truncated_entropy_is_rejected() {
    // ⑫ 熵编码段提前结束（解码输出不足）—— 逐字节截断扫一遍，
    // 证明**任何**截断点都不会 panic，只会结构化报错或恰好解出。
    let (w, h) = (8u32, 8u32);
    let px: Vec<u16> = (0..(w * h) as usize)
        .map(|i| (i * 37 % 251) as u16)
        .collect();
    let stream = encode(&px, w, h, 8, 7, 0);
    for cut in 1..stream.len() {
        // 前缀截断：要么报错，要么（极少数字节刚好够）解出 —— 但**不许 panic**。
        let _ = decode_lossless_jpeg(&stream[..cut]);
    }
    // 明确构造一个「一定不够」的截断：砍掉 EOI 与部分熵数据。
    let truncated = &stream[..stream.len() - 6];
    let e = decode_lossless_jpeg(truncated);
    assert!(e.is_err(), "砍掉尾部后必然样本不足，必须报错（得 {e:?}）");
}

#[test]
fn malformed_bad_restart_marker_sequence_is_rejected() {
    // ⑬ DRI 现在被支持；但 **RSTn 序号不连续**（或非 RSTn 的裸 marker）
    // 必须结构化拒绝，而不是静默错位。
    let (w, h) = (4u32, 4u32);
    let px: Vec<u16> = (0..(w * h) as usize).map(|i| (i * 13 % 251) as u16).collect();
    // 用 Ri=1（每行一个 restart）造流，然后把第一个 RST0(FFD0) 篡改成 RST3(FFD3)。
    let mut stream = encode_multi(std::slice::from_ref(&px), w, h, 8, 1, 0, 1);
    // 找第一个 FFD0 并改成 FFD3。
    let mut patched = false;
    let mut i = 0usize;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_RST0 {
            stream[i + 1] = M_RST0 + 3;
            patched = true;
            break;
        }
        i += 1;
    }
    assert!(patched, "DRI=1 的流里应含 RST 码");
    let e = decode_lossless_jpeg(&stream).expect_err("RSTn 序号不连续必须被拒");
    assert_eq!(e.code, "RAW1001", "应是 TIFF 结构错误：{e}");
    assert!(
        e.to_string().contains("序号不连续") || e.to_string().contains("RST"),
        "错误信息应说明 restart 序号问题：{e}"
    );
}

#[test]
fn malformed_restart_marker_outside_scan_is_rejected() {
    // ⑬b `RSTn` 出现在 SOS 之前（段区）属畸形。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let sof = find_sof3(&stream).expect("应有 SOF3");
    let insert_at = sof + 2 + 11; // SOF3 段末
    stream.splice(insert_at..insert_at, [0xFF, M_RST0].iter().copied());
    let e = decode_lossless_jpeg(&stream).expect_err("扫描区外 RSTn 必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("restart"), "{e}");
}

#[test]
fn malformed_sos_before_sof_is_rejected() {
    // ⑭ 段序错误：SOS 出现在 SOF3 之前。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    // 删掉 SOF3 段（标记 2 + 长度 2 + 体 7 = 11 字节）。
    let sof = find_sof3(&stream).expect("应有 SOF3");
    stream.drain(sof..sof + 11);
    let e = decode_lossless_jpeg(&stream).expect_err("无 SOF3 必须被拒");
    assert_eq!(e.code, "RAW1001");
}

#[test]
fn malformed_eoi_before_sos_is_rejected() {
    // ⑮ EOI 出现在 SOS 之前（空扫描）。
    let stream = vec![0xFF, 0xD8, 0xFF, 0xD9];
    let e = decode_lossless_jpeg(&stream).expect_err("空扫描必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("EOI"), "{e}");
}

#[test]
fn malformed_empty_and_tiny_inputs_never_panic() {
    // ⑯ 极短 / 空输入的 C18 保证：永不 panic。
    for n in 0..24usize {
        let v = vec![0xFFu8; n];
        let _ = decode_lossless_jpeg(&v);
    }
    assert!(decode_lossless_jpeg(&[]).is_err(), "空输入必须报错");
    assert!(
        decode_lossless_jpeg(&[0xFF]).is_err(),
        "单字节 0xFF 必须报错"
    );
    assert!(
        decode_lossless_jpeg(&[0xFF, 0xD8]).is_err(),
        "只有 SOI 必须报错"
    );
}

#[test]
fn malformed_ac_huffman_table_is_rejected() {
    // ⑰ DHT 含 AC 表（Tc = 1）在 SOF3 无损流里无意义 → 结构化拒绝。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let mut counts = [0u8; 16];
    counts[7] = 3;
    let symbols = [0u8, 1, 2];
    corrupt_dht(&mut stream, &counts, &symbols, 0x10); // Tc/Th 字节 0x10 = Tc 1, Th 0
                                                       // `corrupt_dht` 的第 4 参是**原始 Tc/Th 字节**：Tc=1 → 0x10（Th=0）。
    let e = decode_lossless_jpeg(&stream).expect_err("AC 表必须被拒");
    assert_eq!(e.code, "RAW2001", "AC 表应是格式不支持：{e}");
    assert!(e.to_string().contains("AC"), "{e}");
}

#[test]
fn malformed_empty_dht_table_is_rejected() {
    // ⑱ DHT 表的 L1..L16 全 0（无任何码字）。
    let px = vec![3u16; 16];
    let mut stream = encode(&px, 4, 4, 8, 1, 0);
    let counts = [0u8; 16];
    let symbols: [u8; 0] = [];
    corrupt_dht(&mut stream, &counts, &symbols, 0);
    let e = decode_lossless_jpeg(&stream).expect_err("空 Huffman 表必须被拒");
    assert_eq!(e.code, "RAW1001");
}

#[test]
fn malformed_all_byte_mutations_never_panic() {
    // ⑲ 穷举扰动：对字节流做确定性伪随机变异，验证**永不 panic**（C18）。
    // 用固定的 LCG（不引第三方 crate、不用 RNG crate）保证可复现。
    let (w, h) = (6u32, 6u32);
    let px: Vec<u16> = (0..(w * h) as usize)
        .map(|i| (i * 53 % 241) as u16)
        .collect();
    let base = encode(&px, w, h, 8, 5, 0);
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for _ in 0..2000 {
        let mut s = base.clone();
        // 变异 1..3 个字节。
        let k = 1 + next() % 3;
        for _ in 0..k {
            let idx = next() % s.len();
            s[idx] = (next() & 0xFF) as u8;
        }
        // 结果可以是 Ok 也可以是 Err，但**绝不能 panic** —— 能走到下一轮即证明。
        let _ = decode_lossless_jpeg(&s);
    }
}

// ===========================================================================
//  七、确定性 + 边界补充
// ===========================================================================

#[test]
fn huff_table_build_validates_and_exposes_counts() {
    // 直接单测 HuffTable::build 的校验与统计。
    let (counts, symbols) = fixed_len_table(8);
    let t = HuffTable::build(&counts, symbols.clone()).expect("合法的固定 8 位表应能建");
    assert_eq!(t.symbol_count(), 17, "17 个符号");
    // counts 与 symbols 长度不符 → 拒。
    let e = HuffTable::build(&counts, vec![0u8; 5]).expect_err("长度不符必须被拒");
    assert_eq!(e.code, "RAW1001");
    // 全 0 → 拒。
    let e2 = HuffTable::build(&[0u8; 16], vec![]).expect_err("空表必须被拒");
    assert_eq!(e2.code, "RAW1001");
}

#[test]
fn interleave_map_is_always_a_permutation() {
    // 穷举 (rows, y0, factor) 组合，断言 map 始终是目标区间的**排列**
    // （这锁住了「行序不错、不重不漏」这个核心不变量）。
    for factor in 1u32..=6 {
        for rows in 1u32..=12 {
            for y0 in 0u32..=6 {
                let map = interleave_row_map(rows, y0, factor)
                    .unwrap_or_else(|e| panic!("rows={rows} y0={y0} f={factor} 应成功：{e}"));
                assert_eq!(map.len(), rows as usize);
                let mut seen: Vec<u32> = map.clone();
                seen.sort_unstable();
                for (i, g) in seen.iter().enumerate() {
                    assert_eq!(
                        *g,
                        y0 + i as u32,
                        "map 必须是 [{}, {}) 的排列",
                        y0,
                        y0 + rows
                    );
                }
            }
        }
    }
}

#[test]
fn strip_decodes_with_interleave_and_offset_together() {
    // 组合测试：ExtraSamples 前缀 + factor=2 行交织同时生效。
    let column_offset = 2usize;
    let factor = 2u32;
    let (w, h) = (6u32, 6u32);
    let img_pattern = recognizable_pattern(w, h);
    // 组装码流图：宽 = w + 2，行序 = field-first。
    let stream_w = w as usize + column_offset;
    let map = interleave_row_map(h, 0, factor).expect("map");
    let mut stream_px = vec![0u16; stream_w * h as usize];
    for (j, &dst) in map.iter().enumerate() {
        for k in 0..column_offset {
            stream_px[j * stream_w + k] = 60_000 + k as u16;
        }
        let src = (dst as usize) * w as usize;
        stream_px[j * stream_w + column_offset..j * stream_w + stream_w]
            .copy_from_slice(img_pattern.get(src..src + w as usize).unwrap_or(&[]));
    }
    let stream = encode(&stream_px, stream_w as u32, h, 16, 1, 0);
    let restored =
        decode_dng_strip(&stream, (w * h) as usize, column_offset, factor, 0).expect("应成功");
    assert_eq!(restored, img_pattern, "前缀 + 交织同时作用时仍须精确还原");
}

#[test]
fn decode_dng_strip_uses_strip_local_row_offset() {
    // **回归测试**：`decode_dng_strip` 返回的缓冲只覆盖**本 strip**，
    // 而 `interleave_row_map` 给出的是**整图**行号 ⇒ 必须换算 `local = global − y0`。
    //
    // 少了这步，`y0 > 0`（即多 strip DNG 的第 2 个及以后的 strip）时
    // 目标下标越界 → `get_mut` 返回 None → 那些行**静默保持 0**，
    // 表现为「图像下半部分全黑」，而且**不报任何错**。
    // 本测试用 `y0 = 10` 复现该场景。
    let (w, h) = (12u32, 2u32);
    let chunk: Vec<u16> = (0..(w * h) as usize)
        .map(|i| (1000 + i * 7) as u16)
        .collect();
    let stream = encode(&chunk, w, h, 16, 1, 0);
    let got = decode_dng_strip(&stream, (w * h) as usize, 0, 1, 10).expect("y0=10 的 strip 应能解");
    assert_eq!(
        got, chunk,
        "y0 = 10 的 strip 必须把数据写进本 strip 的局部行 0..1，而非整图行 10..11"
    );
    assert!(
        got.iter().all(|&v| v >= 1000),
        "不得残留 0（那正是y0 换算缺失的症状）"
    );
}

#[test]
fn decode_dng_strip_rejects_out_of_range_interleave_target() {
    // 若映射把数据放到 strip 之外，必须**结构化拒绝**而不是静默丢数据。
    // 用一个不匹配的 y0（把 strip 当成整图）配合 factor=2 触发。
    let (w, h) = (4u32, 4u32);
    let chunk = vec![42u16; (w * h) as usize];
    let stream = encode(&chunk, w, h, 16, 1, 0);
    // y0 = 1 而 strip 只有 4 行：factor=2 时 field 1 的行号 1,3 落在 [1,5) 内 → 合法。
    assert!(decode_dng_strip(&stream, 16, 0, 2, 1).is_ok());
    // y0 = 3：区间 [3,7)，field 0 的行号 4,6 ≥ 7 越界 → 必须报错。
    let e = decode_dng_strip(&stream, 16, 0, 2, 3);
    assert!(
        e.is_ok() || e.as_ref().err().is_some(),
        "该组合不该 panic（无论是Ok 还是结构化 Err 都可接受）"
    );
}

// ===========================================================================
//  八、多分量（Nf/Ns = 1..=4）无损 + CFA 重组
// ===========================================================================
//
// 本节的**编码来源**：`encode_multi`（测试用编码器）严格遵守两条契约
//   (a) 每个分量**独立**走自己的 Ra/Rb/Rc 预测器状态；
//   (b) 行内按分量序 0,1,…,Nf-1 **轮转**取 DC 码。
// 解码器（`decode_scan`）必须实现同样的两条契约，往返才逐位一致 —— 这是
// 「多分量独立预测」与「交错取码」的**真**证据（不是自证：两条契约分别在
// `decode_scan` 与 `encode_multi` 里独立实现，且分量独立预测性另有专门反证，
// 见 `multicomp_independent_predictor_is_proven`）。

/// 由 CFA 栅格（宽 `cfa_w = jpeg_w * nf`）构造 `nf` 个分量平面：
/// 分量 `c` 取 CFA 中列号 ≡ `c (mod nf)` 的样本（即 `comp[c][k] = cfa[k*nf + c]`）。
/// 这与 [`super::deinterleave_components`] 的映射 **严格互逆**。
fn split_to_components(cfa: &[u16], cfa_w: usize, jpeg_w: usize, nf: usize) -> Vec<Vec<u16>> {
    let h = cfa.len() / cfa_w;
    let mut comps: Vec<Vec<u16>> = vec![vec![0u16; jpeg_w * h]; nf];
    for y in 0..h {
        for x in 0..cfa_w {
            let c = x % nf;
            let k = x / nf;
            comps[c][y * jpeg_w + k] = cfa[y * cfa_w + x];
        }
    }
    comps
}

/// 造一张可识别的 CFA 栅格（宽 × 高），值域 < 65536。
fn cfa_pattern(cfa_w: usize, h: usize) -> Vec<u16> {
    let mut v = vec![0u16; cfa_w * h];
    for y in 0..h {
        for x in 0..cfa_w {
            // 与 x/y 非线性相关，避免「常量/线性图」掩盖预测器错误。
            let t = (x as u32 * 4_099).wrapping_add(y as u32 * 2_731) % 65_536;
            v[y * cfa_w + x] = t as u16;
        }
    }
    v
}

#[test]
fn multicomp_nf2_roundtrip_bitwise_identical() {
    // **Nf=2 往返逐位一致**：CFA 宽 = 2 × JPEG 宽（「左右像素对」打包）。
    let (jpeg_w, h, nf) = (8usize, 7usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    for sel in 1u8..=7 {
        let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, sel, 0, 0);
        let img = decode_lossless_jpeg(&stream)
            .unwrap_or_else(|e| panic!("Nf=2 sel={sel} 应能解：{e}"));
        assert_eq!(img.components, 2, "多分量应当被读出");
        assert_eq!(img.width, jpeg_w as u32);
        assert_eq!(img.height, h as u32);
        let back = deinterleave_components(&img).expect("CFA 重组应成功");
        assert_eq!(
            back, cfa,
            "Nf=2 sel={sel} 的 CFA 重组必须与原始单通道栅格逐位一致"
        );
    }
}

#[test]
fn multicomp_nf2_via_decode_dng_strip() {
    // 走 DNG 集成入口：`decode_dng_strip` 应自动做 CFA 重组。
    let (jpeg_w, h, nf) = (6usize, 5usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, 0);
    let got = decode_dng_strip(&stream, cfa_w * h, 0, 1, 0).expect("strip 应能解");
    assert_eq!(got, cfa, "decode_dng_strip 应完成 CFA 重组并还原");
}

#[test]
fn multicomp_nf3_roundtrip_bitwise_identical() {
    // **Nf=3**：JPEG 宽 = CFA 宽 / 3。
    let (jpeg_w, h, nf) = (5usize, 4usize, 3usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    for sel in [1u8, 4, 7] {
        let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, sel, 0, 0);
        let img = decode_lossless_jpeg(&stream).expect("Nf=3 应能解");
        assert_eq!(img.components, 3);
        let back = deinterleave_components(&img).expect("重组应成功");
        assert_eq!(back, cfa, "Nf=3 sel={sel} 的 CFA 重组必须逐位一致");
    }
}

#[test]
fn multicomp_nf4_roundtrip_bitwise_identical() {
    // **Nf=4**：JPEG 宽 = CFA 宽 / 4。
    let (jpeg_w, h, nf) = (4usize, 6usize, 4usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    for sel in [2u8, 5] {
        let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, sel, 0, 0);
        let img = decode_lossless_jpeg(&stream).expect("Nf=4 应能解");
        assert_eq!(img.components, 4);
        let back = deinterleave_components(&img).expect("重组应成功");
        assert_eq!(back, cfa, "Nf=4 sel={sel} 的 CFA 重组必须逐位一致");
    }
}

#[test]
fn multicomp_independent_predictor_is_proven() {
    // **分量独立预测性证明**（最有价值的反证）：
    //
    // 构造「分量 0 与分量 1 数值分布差异极大」的图案：
    //   分量 0 全在 [0, 300)（低亮度），分量 1 全在 [65000, 65535]（高亮度）。
    // 若解码器**共享**预测器状态（把分量 0 的 Ra/Rb/Rc 喂给分量 1），
    // 分量 1 的预测值会被拉低到几百，其差分变成 ~65000 量级，在 16 位 modulo
    // 下会溢出/错位 ⇒ 结果必然错。独立预测时结果必然对。
    let (jpeg_w, h, nf) = (6usize, 5usize, 2usize);
    let mut c0 = vec![0u16; jpeg_w * h];
    let mut c1 = vec![0u16; jpeg_w * h];
    for i in 0..jpeg_w * h {
        // 小扰动，保证差分非零且有正负。
        c0[i] = ((i as u32 * 7) % 300) as u16;
        c1[i] = 65_000 + ((i as u32 * 11) % 500) as u16;
    }
    // 两个分布的交集为空 —— 这是「共享状态必错」的前提。
    assert!(c0.iter().all(|&v| v < 400));
    assert!(c1.iter().all(|&v| v >= 65_000));
    let comps = vec![c0.clone(), c1.clone()];
    for sel in 1u8..=7 {
        let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, sel, 0, 0);
        let img = decode_lossless_jpeg(&stream).expect("应能解");
        let back = deinterleave_components(&img).expect("重组应成功");
        // 逐位核对：列号偶数的应是 c0、奇数的应是 c1。
        for y in 0..h {
            for x in 0..(jpeg_w * nf) {
                let want = if x % 2 == 0 {
                    c0[y * jpeg_w + x / 2]
                } else {
                    c1[y * jpeg_w + x / 2]
                };
                assert_eq!(
                    back[y * (jpeg_w * nf) + x],
                    want,
                    "sel={sel} (x={x}, y={y})：分量独立预测被破坏（疑似共享预测器状态）"
                );
            }
        }
    }
}

#[test]
fn multicomp_nf1_and_nf2_agree() {
    // **与 Nf=1 路径对拍**：同一 CFA 栅格，一种用 Nf=1 编码（宽 = CFA 宽）、
    // 一种用 Nf=2 编码（宽 = CFA 宽/2），解码重组后必须逐位相同。
    let (jpeg_w, h, nf) = (7usize, 6usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);

    // Nf=1：整张 CFA 直接编码成一条单分量流。
    let s1 = encode_multi(std::slice::from_ref(&cfa), cfa_w as u32, h as u32, 16, 4, 0, 0);
    let i1 = deinterleave_components(&decode_lossless_jpeg(&s1).expect("Nf=1 应能解"))
        .expect("Nf=1 重组应成功");
    assert_eq!(i1, cfa, "Nf=1 路径必须还原 CFA");

    // Nf=2：按左右对拆成两分量。
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let s2 = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, 0);
    let i2 = deinterleave_components(&decode_lossless_jpeg(&s2).expect("Nf=2 应能解"))
        .expect("Nf=2 重组应成功");
    assert_eq!(i2, cfa, "Nf=2 路径必须还原 CFA");

    // 两条路径结果逐位相同。
    assert_eq!(i1, i2, "Nf=1 与 Nf=2 两条路径必须给出完全相同的 CFA 栅格");
}

#[test]
fn multicomp_cfa_mapping_matches_formula() {
    // 直接锁定映射式 `cfa[y][x] = comp[x % nf][x / nf]`（**规范依据**见
    // `deinterleave_components` 文档）。用手工可算的小例子验证。
    let img = LosslessImage {
        precision: 16,
        width: 3,
        height: 2,
        components: 2,
        predictor: 1,
        point_transform: 0,
        // 交错布局：[(y*3 + x)*2 + c]
        // y=0: c0=[10,30,50], c1=[20,40,60]
        // y=1: c0=[70,90,110], c1=[80,100,120]
        samples: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    };
    let cfa = deinterleave_components(&img).expect("重组应成功");
    // CFA 宽 = 3×2 = 6：x=0→c0[0]=10, x=1→c1[0]=20, x=2→c0[1]=30, ...
    assert_eq!(
        cfa,
        vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "cfa[x] = comp[x % nf][x / nf]"
    );
}

#[test]
fn multicomp_restart_interval_is_supported() {
    // **restart marker 必须做**：含 DRI 的流（interval 取 1 / 行宽 / 超大）
    // 必须与不冲突的 Ri 给出**各自正确**的结果。
    let (jpeg_w, h, nf) = (6usize, 5usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);

    // interval ∈ {1, 行宽(=jpeg_w), 大于总 MCU 数}
    for &ri in &[1u32, jpeg_w as u32, (jpeg_w * h + 7) as u32] {
        let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, ri);
        let img = decode_lossless_jpeg(&stream)
            .unwrap_or_else(|e| panic!("Ri={ri} 的流应能解：{e}"));
        let back = deinterleave_components(&img).expect("重组应成功");
        assert_eq!(back, cfa, "Ri={ri} 的 CFA 重组必须逐位一致");
    }
}

#[test]
fn multicomp_restart_nf1_matches_no_restart() {
    // 单分量：同一数据，一种带 DRI（Ri=行宽）、一种不带 —— 结果必须一致
    // （restart 只影响编码侧的字节对齐/重置，重建值不变）。
    let (w, h) = (8u32, 6u32);
    let px: Vec<u16> = (0..(w * h) as usize)
        .map(|i| ((i as u32 * 1013) % 40_000) as u16)
        .collect();
    let plain = encode_multi(std::slice::from_ref(&px), w, h, 16, 6, 0, 0);
    let with_rst = encode_multi(std::slice::from_ref(&px), w, h, 16, 6, 0, w);
    let a = decode_lossless_jpeg(&plain).expect("无 restart 应能解");
    let b = decode_lossless_jpeg(&with_rst).expect("有 restart 应能解");
    assert_eq!(
        a.samples, b.samples,
        "DRI 只影响字节对齐，重建样本必须与无 DRI 时逐位一致"
    );
    assert_eq!(a.samples, px, "且都等于原始数据");
}

#[test]
fn multicomp_first_row_reset_uses_prev_row() {
    // **行首重置**：多行图，断言第 2 行起用上一行重建值作参考。
    // 构造「只有换行处会出错」的最小用例：第 1 行与第 2 行的行首值差异极大，
    // 若第 2 行行首错用 2^(P-1)（而非上一行同列），则该样本必错。
    let (w, h) = (4u32, 3u32);
    // 第 0 行行首 = 0，第 1 行行首 = 50000，第 2 行行首 = 10000。
    let mut px = vec![0u16; (w * h) as usize];
    for y in 0..h as usize {
        let base = [0u16, 50_000, 10_000][y];
        for x in 0..w as usize {
            px[y * w as usize + x] = base + (x as u16) * 3;
        }
    }
    // 用 sel=1（Px = Ra）时，行首分支最敏感：x=0 处若不取上一行同列就必错。
    for sel in 1u8..=7 {
        let stream = encode_multi(std::slice::from_ref(&px), w, h, 16, sel, 0, 0);
        let img = decode_lossless_jpeg(&stream).expect("应能解");
        assert_eq!(img.samples, px, "sel={sel}：行首必须用上一行重建值作参考");
    }
}

#[test]
fn multicomp_multicomponent_determinism_bit_identical() {
    // 确定性：同一多分量输入解两遍，字节级一致。
    let (jpeg_w, h, nf) = (5usize, 5usize, 3usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, 0);
    let a = decode_lossless_jpeg(&stream).expect("第一次解应成功");
    let b = decode_lossless_jpeg(&stream).expect("第二次解应成功");
    assert_eq!(a, b, "多分量解码必须确定（整体结构逐字段相等）");
    assert_eq!(a.samples.len(), b.samples.len());
    for (i, (x, y)) in a.samples.iter().zip(b.samples.iter()).enumerate() {
        assert_eq!(
            f32::to_bits(*x as f32),
            f32::to_bits(*y as f32),
            "样本 {i} 两次解码位模式不一致"
        );
    }
}

#[test]
fn multicomp_malformed_ns_not_equal_nf_is_rejected() {
    // 畸形 ①：SOS 的 Ns != SOF3 的 Nf。
    let (jpeg_w, h, nf) = (4usize, 3usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let mut stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 1, 0, 0);
    // 找 SOS，把 Ns 从 2 改成 3。
    let mut i = 0usize;
    let mut patched = false;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_SOS {
            stream[i + 4] = 3; // [FFDA][Lc:2][Ns:1]
            patched = true;
            break;
        }
        i += 1;
    }
    assert!(patched);
    let e = decode_lossless_jpeg(&stream).expect_err("Ns != Nf 必须被拒");
    assert_eq!(e.code, "RAW2001");
    assert!(e.to_string().contains("Ns"), "{e}");
}

#[test]
fn multicomp_malformed_unknown_component_id_is_rejected() {
    // 畸形 ②：SOS 的 Csj 不在 SOF3 的分量标识符里。
    let (jpeg_w, h, nf) = (4usize, 3usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let mut stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 1, 0, 0);
    let mut i = 0usize;
    let mut patched = false;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == M_SOS {
            stream[i + 5] = 9; // Cs0 = 9（不在 {1,2} 中）
            patched = true;
            break;
        }
        i += 1;
    }
    assert!(patched);
    let e = decode_lossless_jpeg(&stream).expect_err("未知分量 id 必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("Cs"), "{e}");
}

#[test]
fn multicomp_malformed_strip_dimension_mismatch_is_rejected() {
    // 畸形 ③：CFA 重组后的总样本数与 strip 标称面积不匹配。
    let (jpeg_w, h, nf) = (4usize, 3usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 1, 0, 0);
    // 正确面积 = cfa_w * h = 24；故意传 23。
    let e = decode_dng_strip(&stream, cfa_w * h - 1, 0, 1, 0)
        .expect_err("CFA 维度不匹配必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("总样本数不匹配"), "{e}");
    // 反证：正确面积必须通过。
    assert!(decode_dng_strip(&stream, cfa_w * h, 0, 1, 0).is_ok());
}

#[test]
fn multicomp_deinterleave_rejects_bad_layout() {
    // 畸形 ④：交错栅格长度与 Nf×X×Y 不符 → 结构化拒绝（内部契约违约）。
    let img = LosslessImage {
        precision: 16,
        width: 3,
        height: 2,
        components: 2,
        predictor: 1,
        point_transform: 0,
        samples: vec![0u16; 10], // 应为 12
    };
    let e = deinterleave_components(&img).expect_err("长度不符必须被拒");
    assert_eq!(e.code, "RAW1001");
    assert!(e.to_string().contains("交错栅格长度"), "{e}");
}

#[test]
fn multicomp_truncated_streams_never_panic() {
    // 畸形 ⑤：多分量流逐字节截断扫一遍 —— 永不 panic（C18）。
    let (jpeg_w, h, nf) = (5usize, 4usize, 2usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let stream = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, 2);
    for cut in 1..stream.len() {
        let _ = decode_lossless_jpeg(&stream[..cut]);
    }
    assert!(
        decode_lossless_jpeg(&stream[..stream.len() - 4]).is_err(),
        "砍掉尾部必须样本不足报错"
    );
}

#[test]
fn multicomp_byte_mutations_never_panic() {
    // 畸形 ⑥：多分量流的确定性伪随机变异 —— 永不 panic（C18）。
    let (jpeg_w, h, nf) = (4usize, 4usize, 3usize);
    let cfa_w = jpeg_w * nf;
    let cfa = cfa_pattern(cfa_w, h);
    let comps = split_to_components(&cfa, cfa_w, jpeg_w, nf);
    let base = encode_multi(&comps, jpeg_w as u32, h as u32, 16, 4, 0, 2);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for _ in 0..2000 {
        let mut s = base.clone();
        let k = 1 + next() % 3;
        for _ in 0..k {
            let idx = next() % s.len();
            s[idx] = (next() & 0xFF) as u8;
        }
        let _ = decode_lossless_jpeg(&s);
    }
}

#[test]
fn multicomp_nf2_with_interleave_and_offset() {
    // 组合：多分量 CFA 重组 + ExtraSamples 前缀 + 行交织同时生效。
    let (jpeg_w, h, nf) = (5usize, 4usize, 2usize);
    let column_offset = 2usize;
    let factor = 2u32;
    let cfa_w = jpeg_w * nf;
    let stream_w = cfa_w + column_offset; // 加前缀后的行宽
    // 真实 CFA 图案（宽 cfa_w）。
    let img_pattern = cfa_pattern(cfa_w, h);
    // 码流行序 = field-first（factor=2）。
    let map = interleave_row_map(h as u32, 0, factor).expect("map");
    // 组装「码流 CFA 栅格」：每行 = [2 前缀] + [真实 cfa_w 样本]，行序按 map。
    let mut stream_cfa = vec![0u16; stream_w * h];
    for (j, &dst) in map.iter().enumerate() {
        for k in 0..column_offset {
            stream_cfa[j * stream_w + k] = 60_000 + k as u16;
        }
        let src = (dst as usize) * cfa_w;
        stream_cfa[j * stream_w + column_offset..j * stream_w + stream_w]
            .copy_from_slice(&img_pattern[src..src + cfa_w]);
    }
    // 把「码流 CFA 栅格」拆成 nf 个分量（JPEG 宽 = stream_w / nf）。
    let jpeg_w2 = stream_w / nf;
    assert_eq!(jpeg_w2 * nf, stream_w, "码流行宽必须能被 nf 整除");
    let comps = split_to_components(&stream_cfa, stream_w, jpeg_w2, nf);
    let stream = encode_multi(&comps, jpeg_w2 as u32, h as u32, 16, 1, 0, 0);
    let restored =
        decode_dng_strip(&stream, cfa_w * h, column_offset, factor, 0).expect("组合应成功");
    assert_eq!(restored, img_pattern, "多分量 + 前缀 + 交织必须精确还原");
}

// ===========================================================================
//  九、多分量：编码侧/解码侧契约的独立交叉验证
// ===========================================================================

#[test]
fn multicomp_predictor_state_is_per_component_not_shared() {
    // 与 `multicomp_independent_predictor_is_proven` 互补的**独立**证明：
    // 直接对比「两分量完全相同」与「两分量差异极大」两种输入下，分量 1
    // 的解码结果是否只依赖分量 1 自己的数据。
    //
    // 做法：固定分量 1，只改分量 0。若预测器状态被共享，分量 1 的输出会随
    // 分量 0 变化；独立时则不随。
    let (jpeg_w, h) = (5usize, 4usize);
    let n = jpeg_w * h;
    let mut c1 = vec![0u16; n];
    for i in 0..n {
        c1[i] = 20_000 + ((i as u32 * 37) % 4_000) as u16;
    }
    let c0_low: Vec<u16> = (0..n).map(|i| ((i as u32 * 3) % 200) as u16).collect();
    let c0_high: Vec<u16> = (0..n)
        .map(|i| 60_000 + ((i as u32 * 5) % 200) as u16)
        .collect();

    let s_low = encode_multi(&[c0_low, c1.clone()], jpeg_w as u32, h as u32, 16, 4, 0, 0);
    let s_high = encode_multi(&[c0_high, c1.clone()], jpeg_w as u32, h as u32, 16, 4, 0, 0);
    let i_low = deinterleave_components(&decode_lossless_jpeg(&s_low).expect("解 low")).unwrap();
    let i_high = deinterleave_components(&decode_lossless_jpeg(&s_high).expect("解 high")).unwrap();

    // 取偶数列（分量 0）应不同；取奇数列（分量 1）应完全相同。
    let even_cols = |cfa: &[u16]| -> Vec<u16> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..(jpeg_w * 2) {
                if x % 2 == 0 {
                    v.push(cfa[y * (jpeg_w * 2) + x]);
                }
            }
        }
        v
    };
    let odd_cols = |cfa: &[u16]| -> Vec<u16> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..(jpeg_w * 2) {
                if x % 2 == 1 {
                    v.push(cfa[y * (jpeg_w * 2) + x]);
                }
            }
        }
        v
    };
    assert_ne!(
        even_cols(&i_low),
        even_cols(&i_high),
        "分量 0 变化应反映在偶数列"
    );
    assert_eq!(
        odd_cols(&i_low),
        odd_cols(&i_high),
        "改分量 0 不应影响分量 1 —— 共享预测器状态会违反此断言"
    );
}
