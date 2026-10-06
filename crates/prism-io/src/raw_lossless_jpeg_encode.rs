//! ITU-T T.81（JPEG 1.0）**无损模式编码器** —— DNG `Compression = 7` 的**写入**侧。
//!
//! # 为什么需要它
//!
//! [`crate::raw_dng_synth`]（合成最小合法 DNG 的样本生成器）必须能**从零写出**
//! 一条真实的 SOF3 字节流，否则它只能生成 `Compression = 1` 的样本，端到端链路
//! 的无损分支就无法在没有真实相机样本时也被验证。
//!
//! # 与解码器的关系：**不共享假设**
//!
//! 本模块与 [`crate::raw_lossless_jpeg`] 的解码器是**同一份T.81 规范的两个独立
//! 实现**，往返测试（round-trip）才有意义 —— 若二者共享同一套错误假设，
//! 「编码 → 解码」恒等只能证明自洽，不能证明符合规范。
//!
//! # 权威依据（与解码器逐条对应）
//!
//! - **B.2.1** —— `SOI`/`EOI`/`RSTn` marker；
//! - **B.2.2** —— `SOF3`（无损 Huffman）帧头：精度 `P`、高 `Y`、宽 `X`、分量数
//!   `Nf`，每分量 `(C_i, H_i/V_i, T_q_i)`；
//! - **B.2.3** —— `SOS` 扫描头：`Ss` 在无损模式下**就是预测器选择值**
//!   （见 [`Predictor`]与 T.81 Table H.1），`Se = 0`，`Ah = 0 | Al = Pt`；
//! - **B.2.4.1** —— `DRI`（`Ri` = restart interval）；
//! - **B.2.4.2 + Annex C** —— `DHT` 的 canonical Huffman 码表构造；
//! - **B.2.4.2 / B.2.4.1** —— 熵编码后的 **`FF 00` 字节填充**与「丢弃填充位」
//!   的字节对齐规则；
//! - **F.1.2.1 + Table H.2** —— 差值的 `SSSS` 分类与 `EXTEND` 附加比特。
//!
//! # 确定性
//!
//! 码表用**固定长度**（`SSSS = 0..=16` 全 8 位，Kraft 和 = 17/256 ≤ 1，合法）
//! 而非最优 Huffman —— 合法、确定性强、且使期望字节流可手算复核。
//! 熵编码的扫描顺序（行内按分量轮转、每分量独立预测状态、restart 处重置）
//! 全部固定，无迭代器 zip、无 `HashMap`，**同输入逐字节可复现**。
//!
//! # C18（永不 panic）
//!
//! 全部入参都做**防御性校验**并返回 [`RawResult`]：预测器选择值经
//! [`Predictor::from_selection`] 校验、精度/尺寸/分量数越界即结构化拒绝。
//! **没有任何 `unwrap` / `expect` / `panic!`**，故本模块可在生产路径上调用。

use crate::raw::{unsupported_err, RawResult};
use crate::raw_lossless_jpeg::{Predictor, M_DHT, M_DRI, M_EOI, M_RST0, M_SOF3, M_SOI, M_SOS};

/// 编码器支持的**最大分量数** `Nf`（DNG 的 CFA 重打包最多用 4）。
pub const MAX_COMPONENTS: usize = 4;

/// 编码器支持的**最大帧宽/高**（T.81 的 `X`/`Y` 是 `u16`，故硬上限 65535）。
pub const MAX_DIMENSION: u32 = 65535;

/// SOF3 的最大精度 `P`（T.81 B.2.2 允许 2..=16；本模块额外接受 1 以便测试
/// `BitsPerSample = 1` 的合成样本，解码器同样接受）。
pub const MAX_PRECISION: u8 = 16;

// ===========================================================================
//  一、Huffman 码表构造（T.81 B.2.4.2 + Annex C）
// ===========================================================================

/// 由符号频次构造一张**合法 canonical Huffman 表**（T.81 Annex C）。
///
/// 用**固定长度**（全 8 位）而非最优 Huffman：Kraft 和 `= 17/256 ≤ 1`，
/// **合法**（JPEG 允许 DC 表不满），且让期望字节流可手算。
///
/// # 参数
///
/// - `nbits` —— 每个符号的码长（`clamp(1, 16)`；17 个符号全用该长度）。
///
/// # 返回
///
/// `(L1..L16 码长计数, 符号表)` —— 恰好是 `DHT` 段的负载布局。
pub fn fixed_len_table(nbits: u8) -> ([u8; 16], Vec<u8>) {
    let mut counts = [0u8; 16];
    let l = (nbits as usize).clamp(1, 16);
    counts[l - 1] = 17; // SSSS = 0..=16 共 17 个符号
    let symbols: Vec<u8> = (0..=16u8).collect();
    (counts, symbols)
}

/// canonical 码生成器（T.81 Annex C 的 `CODE` 递推）。
#[derive(Debug, Clone)]
struct CodeWriter {
    codes: [u32; 17],
    lens: [u8; 17],
}

impl CodeWriter {
    /// 由 `L1..L16` 与符号表生成 canonical 码。
    ///
    /// 递推（T.81 Annex C.2）：`CODE(1) = 0`；处理完长度 `l` 的全部符号后
    /// `CODE(l+1) = (CODE(l) + 计数(l)) << 1`。
    fn new(counts: &[u8; 16], symbols: &[u8]) -> Self {
        let mut codes = [0u32; 17];
        let mut lens = [0u8; 17];
        let mut code: u32 = 0;
        let mut k = 0usize;
        for l in 1..=16usize {
            let n = counts[l - 1] as u32;
            if n > 0 {
                for j in 0..n {
                    let sym = symbols.get(k + j as usize).copied().unwrap_or(0);
                    codes[sym as usize] = code + j;
                    lens[sym as usize] = l as u8;
                }
                code += n;
                k += n as usize;
            }
            code <<= 1;
        }
        Self { codes, lens }
    }
}

// ===========================================================================
//  二、位写入器（T.81 B.2.4.2 的 `FF 00` 填充 + B.2.4.1 的字节对齐）
// ===========================================================================

/// 位写入器（MSB first + `FF 00` 字节填充）。
#[derive(Debug, Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    /// 写 `n` 位的 `v`（MSB first）。`n <= 16`。
    fn put(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = (v >> i) & 1;
            self.acc = (self.acc << 1) | bit;
            self.n += 1;
            if self.n == 8 {
                let b = (self.acc & 0xFF) as u8;
                self.out.push(b);
                // 字节填充：0xFF 后必须跟一个 0x00（T.81 B.2.4.2）。
                if b == 0xFF {
                    self.out.push(0x00);
                }
                self.acc = 0;
                self.n = 0;
            }
        }
    }

    /// 字节对齐：把不足一字节的填充位补 **1** 并落盘。
    ///
    /// T.81 B.2.4.1 要求 restart marker 前「丢弃不足一字节的填充位」，故必须
    /// 先对齐；补 **1** 而非 0 是 T.81 B.2.4.2 的规定（也避免全 0流被误判为
    /// 段填充）。
    fn align(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1u32 << pad) - 1, pad);
        }
    }

    /// 收尾：不足一字节的填充位补 **1**。
    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

// ===========================================================================
//  三、差值 → SSSS / 附加比特（T.81 Table H.2 与 F.1.2.1 的逆映射）
// ===========================================================================

/// 差值的 `SSSS` 分类（T.81 Table H.2 的逆映射）。
///
/// - `0` → 0；
/// - `±32768` → **16**（Table H.2 新增项，恰好 32768）；
/// - 其余按 `|diff|` 的**有效位数**定 `SSSS ∈ 1..=15`。
pub fn category(diff: i32) -> u8 {
    if diff == 0 {
        return 0;
    }
    // modulo 2^16 下 +32768 与 -32768 同余；Table H.2 用 32768。
    if diff == 32_768 || diff == -32_768 {
        return 16;
    }
    let a = diff.unsigned_abs();
    // a ∈ [1, 32767]：SSSS = a 的有效位数 = ceil(log2(a+1))
    let mut s = 0u8;
    let mut v = a;
    while v > 0 {
        s += 1;
        v >>= 1;
    }
    s
}

/// 差值 → 附加比特（`EXTEND` 的逆，T.81 F.1.2.1）。
///
/// 对 `SSSS = T > 0`：`V = diff`（正）或 `V = diff + Vt`（负），其中
/// `Vt = 2^T − 1`。这与解码侧 `if v < 2^(T-1) { v - vt } else { v }` 严格互逆。
pub fn diff_bits(diff: i32, ssss: u8) -> u32 {
    if ssss == 0 || ssss == 16 {
        return 0; // 无附加比特
    }
    let vt = (1i32 << ssss) - 1;
    if diff < 0 {
        (diff + vt) as u32 & ((1u32 << ssss) - 1)
    } else {
        diff as u32 & ((1u32 << ssss) - 1)
    }
}

// ===========================================================================
//  四、编码器主体
// ===========================================================================

/// 校验并归一化编码入参（C18：所有可能 panic 的算术都在这里挡住）。
fn validate(
    width: u32,
    height: u32,
    precision: u8,
    predictor: u8,
    point_transform: u8,
    components: usize,
) -> RawResult<()> {
    const LOC: &str = "raw_lossless_jpeg_encode::validate";
    if width == 0 || height == 0 {
        return Err(unsupported_err(
            LOC,
            format!("帧尺寸 {width}×{height} 含 0 维"),
            "T.81 的 SOF3 宽高须均 ≥ 1；确认调用方的 strip/tile 尺寸非空",
        ));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(unsupported_err(
            LOC,
            format!("帧尺寸 {width}×{height} 超出 SOF3 的 u16 上限 {MAX_DIMENSION}"),
            "T.81 B.2.2 的 X/Y 是 u16；超大图请分块（tile）编码",
        ));
    }
    if precision == 0 || precision > MAX_PRECISION {
        return Err(unsupported_err(
            LOC,
            format!("精度 P = {precision}，本编码器只接受 1..={MAX_PRECISION}"),
            "DNG 的 BitsPerSample 常见为 8/10/12/14/16；32 位浮点 RAW 请先转 16 位整数 DNG",
        ));
    }
    if point_transform >= precision {
        return Err(unsupported_err(
            LOC,
            format!("点变换 Al = {point_transform} ≥ 精度 P = {precision}"),
            "T.81 B.2.3 的 Al 是「左移位数」，必须小于 P；确认调用方未把Al 与 P 搞反",
        ));
    }
    if components == 0 || components > MAX_COMPONENTS {
        return Err(unsupported_err(
            LOC,
            format!("分量数 Nf = {components}，本编码器只接受 1..={MAX_COMPONENTS}"),
            "DNG 的 CFA 重打包最多 4 分量（2×2 Bayer）；更多分量需专用管线",
        ));
    }
    // 预测器合法性由`Predictor::from_selection` 判定（它对 0 与 >7 都报错）。
    let _ = Predictor::from_selection(predictor)?;
    Ok(())
}

/// **T.81 无损JPEG 编码器**（单分量特例，无 restart）。
///
/// 产出的段序严格按 T.81 B.2：`SOI` → `SOF3` → `DHT` → [`DRI`] → `SOS` →
/// 熵编码段（含 `FF 00` 填充）→ `EOI`。
///
/// 本函数是 [`encode_multi`] 在 `Nf = 1` 且无 restart 时的特例，保留它使
/// 单分量调用点的意图更直白。
///
/// # 参数
///
/// - `samples` —— 单分量平面，长度须恰为 `width * height`（行主序）；
/// - `width` / `height` —— `SOF3` 的 `X` / `Y`；
/// - `precision` —— `SOF3.P`（样本位深）；
/// - `predictor` —— 预测器选择值 `Ss`（1..=7，T.81 Table H.1）；
/// - `point_transform` —— `Al`（点变换 / 左移位数，0..P）。
///
/// # 错误
///
/// 参数越界（尺寸/精度/分量数/预测器）→ [`RawError`](../raw/struct.RawError.html)
/// （`RAW2xxx`）。**不 panic**。
pub fn encode(
    samples: &[u16],
    width: u32,
    height: u32,
    precision: u8,
    predictor: u8,
    point_transform: u8,
) -> RawResult<Vec<u8>> {
    encode_multi(
        &[samples.to_vec()],
        width,
        height,
        precision,
        predictor,
        point_transform,
        0,
    )
}

/// **T.81 无损 JPEG 编码器**（多分量 + 可选 restart interval）。
///
/// # 参数
///
/// - `components` —— `Nf` 个分量平面，每个长度须为 `width * height`，行主序；
/// - `width` / `height` —— `SOF3` 的 `X` / `Y`（**每分量**的宽高）；
/// - `precision` —— `SOF3.P`；
/// - `predictor` —— 预测器选择值 `Ss`（1..=7）；
/// - `point_transform` —— `Al`（0..P）；
/// - `restart_interval` —— `DRI` 的 `Ri`（0 = 不写 DRI、无 restart）。
///
/// # 熵编码（与解码器 `decode_scan` 互为逆）
///
/// - 每个分量**独立**走 T.81 H.1.2.1 的预测器状态（`Ra`/`Rb`/`Rc` 用该分量
///   自己的上/左/左上重建值）；
/// - 行内按分量序 `0,1,…,Nf-1` **轮转**取DC 码——即 T.81 B.2.3 的非交错 MCU
///   语义（`H_i = V_i = 1` ⇒ 一个 MCU = 一行内的 `Nf` 个样本）。
///
/// # restart interval
///
/// `Ri != 0` 时，每 `Ri` 个 MCU 就：丢弃填充位（字节对齐）→ 写 `RSTn`
/// （`n` 按 0..7 循环）→ 重置所有分量的预测器状态（回到「首行行首」`2^(P-1)`）。
///
/// # 与 CFA 重打包的配合
///
/// DNG 1.6 明确允许 SOF3 的 `X`/`Y`/`Nf` 与 strip/tile 的标称不同，只要**总样本数**
/// 相等（见 [`crate::raw_lossless_jpeg::decode_dng_strip`]）。本编码器因此直接暴露
/// `Nf`：调用方把 CFA 的一行按「列号 ≡ c (mod Nf)」拆成 `Nf` 个分量即可。
pub fn encode_multi(
    components: &[Vec<u16>],
    width: u32,
    height: u32,
    precision: u8,
    predictor: u8,
    point_transform: u8,
    restart_interval: u32,
) -> RawResult<Vec<u8>> {
    const LOC: &str = "raw_lossless_jpeg_encode::encode_multi";
    let nf = components.len();
    validate(width, height, precision, predictor, point_transform, nf)?;

    let w = width as usize;
    let h = height as usize;
    let want = w.checked_mul(h).ok_or_else(|| {
        unsupported_err(
            LOC,
            format!("{w} × {h} 样本数溢出"),
            "帧尺寸过大；请分块编码",
        )
    })?;
    for (c, plane) in components.iter().enumerate() {
        if plane.len() != want {
            return Err(unsupported_err(
                LOC,
                format!("分量 {c} 长度 {} ≠ 宽×高 = {want}", plane.len()),
                "每个分量平面都必须恰好是 width*height 个样本（行主序）",
            ));
        }
    }

    let pred_fn = Predictor::from_selection(predictor)?;
    let (counts, symbols) = fixed_len_table(8);
    let cw = CodeWriter::new(&counts, &symbols);
    let first_pred: i32 = 1i32 << (precision - 1 - point_transform);

    // 每个分量的上一行/ 当前行缓冲（编码侧与解码侧对称：独立预测）。
    let mut prev: Vec<Vec<u16>> = vec![vec![0u16; w]; nf];
    let mut cur: Vec<Vec<u16>> = vec![vec![0u16; w]; nf];
    let mut rows_since_reset: usize = 0;

    // 逐样本、逐分量「按需写码」：restart 需要**在interval 边界对齐字节**，
    // 故在扫描过程中按 (y, x) 顺序、每个样本位置内按分量序写。
    let mut bw = BitWriter::new();
    let mut rst_index: u8 = 0;
    for y in 0..h {
        let mut row_had_reset = false;
        for x in 0..w {
            let mcu_index = (y as u64) * (w as u64) + x as u64;
            let interval_head = restart_interval != 0
                && mcu_index != 0
                && mcu_index.is_multiple_of(restart_interval as u64);
            if interval_head {
                // 字节对齐：写足填充 1 位到整字节边界（T.81 B.2.4.1）。
                bw.align();
                // 写 RSTn（n 按 0..7 循环）。
                bw.out.push(0xFF);
                bw.out.push(M_RST0 + rst_index);
                rst_index = (rst_index + 1) & 7;
                for c in 0..nf {
                    prev[c].fill(0);
                    cur[c].fill(0);
                }
                rows_since_reset = 0;
                row_had_reset = true;
            }
            let fresh = rows_since_reset == 0;
            for c in 0..nf {
                let plane = components.get(c).map(|v| v.as_slice()).unwrap_or(&[]);
                let v = plane.get(y * w + x).copied().unwrap_or(0) as i32;
                let px: i32 = if x == 0 {
                    if fresh {
                        first_pred
                    } else {
                        prev[c][0] as i32
                    }
                } else if fresh {
                    cur[c][x - 1] as i32
                } else {
                    pred_fn.predict_i32(
                        cur[c][x - 1] as i32,
                        prev[c][x] as i32,
                        prev[c][x - 1] as i32,
                    )
                };
                let d = ((v - px) & 0xFFFF) as i32;
                let d = if d >= 32_768 { d - 65_536 } else { d };
                let ssss = category(d);
                let l = cw.lens[ssss as usize] as u32;
                bw.put(cw.codes[ssss as usize], l);
                let extra = if ssss == 16 { 0 } else { ssss as u32 };
                if extra > 0 {
                    bw.put(diff_bits(d, ssss), extra);
                }
                if let Some(s) = cur.get_mut(c).and_then(|r| r.get_mut(x)) {
                    *s = ((px + d) & 0xFFFF) as u16;
                }
            }
        }
        for c in 0..nf {
            std::mem::swap(&mut prev[c], &mut cur[c]);
        }
        if !row_had_reset {
            rows_since_reset = rows_since_reset.saturating_add(1);
        }
    }
    let entropy = bw.finish();

    // ---- 段 ----
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&[0xFF, M_SOI]);

    // SOF3（T.81 B.2.2）：P, Y, X, Nf, (Ci, HV=0x11, Tq=0)*Nf
    out.extend_from_slice(&[0xFF, M_SOF3]);
    out.extend_from_slice(&((6 + 3 * nf + 2) as u16).to_be_bytes());
    out.push(precision);
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(nf as u8);
    for c in 0..nf {
        out.extend_from_slice(&[(c + 1) as u8, 0x11, 0]);
    }

    // DHT（T.81 B.2.4.2）：Tc=0/Th=0, L1..L16, 符号
    out.extend_from_slice(&[0xFF, M_DHT]);
    let dht_body = 1 + 16 + symbols.len();
    out.extend_from_slice(&((dht_body + 2) as u16).to_be_bytes());
    out.push(0x00); // Tc=0（DC）, Th=0
    out.extend_from_slice(&counts);
    out.extend_from_slice(&symbols);

    // 可选 DRI（T.81 B.2.4.1）：FF DD 00 04 Ri(2B)
    if restart_interval != 0 {
        out.extend_from_slice(&[0xFF, M_DRI, 0x00, 0x04]);
        out.extend_from_slice(&(restart_interval as u16).to_be_bytes());
    }

    // SOS（T.81 B.2.3）：Ns, (Csj, Td/Ta=0)*Ns, Ss, Se=0, Ah/Al=0|Pt
    out.extend_from_slice(&[0xFF, M_SOS]);
    out.extend_from_slice(&((1 + 2 * nf + 3 + 2) as u16).to_be_bytes());
    out.push(nf as u8);
    for c in 0..nf {
        out.extend_from_slice(&[(c + 1) as u8, 0x00]); // Cs = c+1, Td=0|Ta=0
    }
    out.push(predictor); // Ss = 预测器选择值（Table H.1）
    out.push(0); // Se = 0
    out.push(point_transform & 0x0F); // Ah=0 | Al=Pt

    out.extend_from_slice(&entropy);
    out.extend_from_slice(&[0xFF, M_EOI]);
    Ok(out)
}
