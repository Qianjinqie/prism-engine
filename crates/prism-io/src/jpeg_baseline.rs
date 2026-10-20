//! 基线 JPEG（sequential DCT / Huffman / 8-bit）解码器。
//!
//! 算法规格：定点 IDCT（i64 主路径 + 超阈 i128 兜底）。要点：
//!
//! * **零第三方 crate**：容器解析、Huffman 解码、IDCT、上采样、色彩变换全部手写；
//! * **确定性**：IDCT 全程整数（Stage1 i64 + Stage2 阈值内 i64 / 超阈 i128），
//!   色彩变换用 2^16 定点乘加，**不出现 `f64::cos` / `log2` / `exp` 等超越函数**，
//!   同输入两跑逐位相同；
//! * **诚实边界**：只支持基线（SOF0/SOF1，8-bit precision、单扫描、Huffman），
//!   渐进式 / 算术编码 / 12-bit / CMYK 一律返回明确错误，绝不静默降级。
//!
//! 中心定理（§2.1①）：把 DCT 正交性与 ¼·c(u)c(v) 权重折叠进
//! [`IDCT_C_TABLE`]（2^14·K）与 [`IDCT_W_TABLE`]（2^16·c·c）两张整数表后，
//! 反变换可写成两阶段纯整数累加，末位标度 `2^-46` 恰好回收 2^16·2^14·2^14·¼ 的
//! 全部标度。

// ===========================================================================
//  ① 精确常量表（§2.1①，exp23 独立复算）
// ===========================================================================

/// 余弦基表：`C[a][b] = round(2^14 · cos((2b+1)·a·π/16))`，扁平索引 `a*8+b`。
///
/// 与编码器 `prism-mux/src/jpeg.rs` 的 `DCT_COS` **同源同值**（编解码共用一份资产）。
pub const IDCT_C_TABLE: [i32; 64] = [
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, //
    16069, 13623, 9102, 3196, -3196, -9102, -13623, -16069, //
    15137, 6270, -6270, -15137, -15137, -6270, 6270, 15137, //
    13623, -3196, -16069, -9102, 9102, 16069, 3196, -13623, //
    11585, -11585, -11585, 11585, 11585, -11585, -11585, 11585, //
    9102, -16069, 3196, 13623, -13623, -3196, 16069, -9102, //
    6270, -15137, 15137, -6270, -6270, 15137, -15137, 6270, //
    3196, -9102, 13623, -16069, 16069, -13623, 9102, -3196, //
];

/// 归一化权重表：`W[u][v] = round(2^16 · c(u)·c(v)) ∈ {65536, 46341, 32768}`。
///
/// 扁平索引 `v*8+u`（与 `D[i] = F[i]·Q[i]` 的自然序 `i = v*8+u` 对齐），
/// 表对称因而布局不影响结果。
pub const IDCT_W_TABLE: [i32; 64] = [
    32768, 46341, 46341, 46341, 46341, 46341, 46341, 46341, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
    46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, //
];

/// Stage2 走 i64 快路径的安全阈值：`2^29 − 1 = 536870911`（B1 复算）。
///
/// 安全条件：`64·T·2^28 ≤ i64::MAX` ⇒ `T ≤ 5.36870912e8 = 2^29 − 2^-34`，
/// 向下取整即 `2^29 − 1`；而 `2^29` 本身使 `64·2^29·2^28 = 2^63 = i64::MAX + 1`
/// **恰超一个单位**，故阈值必须是 `2^29 − 1` 而不能上取整。
pub const IDCT_STAGE2_I64_THRESHOLD: i64 = (1 << 29) - 1;

/// const assert ①：阈值不得超过 `2^29 − 1`（否则 i64 路径可能回绕）。
const _: () = assert!(IDCT_STAGE2_I64_THRESHOLD <= (1 << 29) - 1);

/// const assert ②：W 表值域恰为 `{32768, 46341, 65536}`（错一个数即 PSNR 崩塌）。
const _: () = {
    let mut i: usize = 0;
    while i < 64 {
        let w: i32 = IDCT_W_TABLE[i];
        assert!(
            w == 32768 || w == 46341 || w == 65536,
            "IDCT_W_TABLE 值域越界（只允许 32768/46341/65536）"
        );
        i += 1;
    }
};

/// ZigZag 位置 → 自然序下标（`natural[ZIGZAG[k]] = zigzag[k]`）。
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// ===========================================================================
//  ② 定点 IDCT（§2.1②）
// ===========================================================================

/// 定点 8×8 反离散余弦变换（8-bit 输出路径）。
///
/// # 参数
/// * `out`  — 输出 64 个样本（行主序 `y*8+x`，已含 +128 电平偏移并 clamp 到 `[0,255]`）。
/// * `coef` — 反 ZigZag 后的**量化系数** `F`（自然序，`i = v*8+u` 持有 `F[u][v]`）。
/// * `q`    — 量化表（自然序，值域 `1..=65535`）。
///
/// # 算法（逐块判定走哪条 Stage2 路径）
/// 1. `D[i] = coef[i]·q[i]`，同时求 `max_dw = max|D[i]·W[i]|`；
/// 2. `max_dw ≤ IDCT_STAGE2_I64_THRESHOLD` → Stage2 走 i64 快路径，否则 i128；
/// 3. Stage1（列）：`S1[u][y] = Σ_v D[u][v]·W[u][v]·C[v][y]`（i64）；
/// 4. Stage2（行）：`S2[y][x] = Σ_u S1[u][y]·C[u][x]`；
/// 5. 末位：`s = ((S2 + 2^45) >> 46) + 128`，clamp 到 `[0,255]`。
pub fn idct_8x8_fixed(out: &mut [i32; 64], coef: &[i32; 64], q: &[i32; 64]) {
    let (d, max_dw) = dequantize(coef, q);
    let s1: [i64; 64] = stage1_columns(&d);
    if max_dw <= IDCT_STAGE2_I64_THRESHOLD {
        stage2_rows_i64(out, &s1);
    } else {
        stage2_rows_i128(out, &s1);
    }
}

/// 强制走 i128 路径的定点 IDCT（供 A1 / B-M1 双路径对拍使用）。
///
/// 语义与 [`idct_8x8_fixed`] 完全一致，唯一区别是不做阈值分流——任何输入都走
/// 宽路径。测试据此断言「阈值内的块两条路径逐位相同」。
pub fn idct_8x8_fixed_wide(out: &mut [i32; 64], coef: &[i32; 64], q: &[i32; 64]) {
    let (d, _max_dw) = dequantize(coef, q);
    let s1: [i64; 64] = stage1_columns(&d);
    stage2_rows_i128(out, &s1);
}

/// 反量化并求 `max|D·W|`（逐块阈值判定的唯一依据）。
#[inline]
fn dequantize(coef: &[i32; 64], q: &[i32; 64]) -> ([i64; 64], i64) {
    let mut d = [0i64; 64];
    let mut max_dw: i64 = 0;
    for i in 0..64 {
        // D = F·Q（i64；F ≤ 2^11 量级、Q ≤ 2^16 → 远不溢出）。
        let di: i64 = (coef[i] as i64) * (q[i] as i64);
        d[i] = di;
        let dw: i64 = (di * (IDCT_W_TABLE[i] as i64)).abs();
        if dw > max_dw {
            max_dw = dw;
        }
    }
    (d, max_dw)
}

/// Stage1（列变换）：`S1[u][y] = Σ_v D[u][v]·W[u][v]·C[v][y]`，i64 累加。
///
/// 界（§2.1①）：`|S1| ≤ 8·max|D·W|·2^14`，16-bit 极端块 ≤ 1.152341e18，
/// i64 余量 8.004×，因此**Stage1 恒可 i64**（无需分流）。
#[inline]
fn stage1_columns(d: &[i64; 64]) -> [i64; 64] {
    let mut s1 = [0i64; 64];
    for u in 0..8 {
        for y in 0..8 {
            let mut acc: i64 = 0;
            for v in 0..8 {
                let i: usize = v * 8 + u;
                acc += d[i] * (IDCT_W_TABLE[i] as i64) * (IDCT_C_TABLE[v * 8 + y] as i64);
            }
            s1[u * 8 + y] = acc;
        }
    }
    s1
}

/// Stage2（行变换）i64 快路径：仅当 `max|D·W| ≤ 2^29−1` 时调用。
///
/// 界：`|S2| ≤ 64·T·2^28 = 2^63 − 2^34 ≤ i64::MAX`（余量 `2^34 − 1`）。
#[inline]
fn stage2_rows_i64(out: &mut [i32; 64], s1: &[i64; 64]) {
    const HALF: i64 = 1i64 << 45;
    for y in 0..8 {
        for x in 0..8 {
            let mut acc: i64 = 0;
            for u in 0..8 {
                acc += s1[u * 8 + y] * (IDCT_C_TABLE[u * 8 + x] as i64);
            }
            // round-half-up 后加电平偏移 128（与 §2.1 金参考 golden_u8_from_exact 同构）。
            let s: i64 = ((acc + HALF) >> 46) + 128;
            out[y * 8 + x] = s.clamp(0, 255) as i32;
        }
    }
}

/// Stage2（行变换）i128 宽路径：阈值超界块（含对抗块）走这里。
///
/// 与 i64 路径**数学同构**，只是累加器更宽；对阈值内的块两者必然逐位一致。
#[inline]
fn stage2_rows_i128(out: &mut [i32; 64], s1: &[i64; 64]) {
    const HALF: i128 = 1i128 << 45;
    for y in 0..8 {
        for x in 0..8 {
            let mut acc: i128 = 0;
            for u in 0..8 {
                acc += (s1[u * 8 + y] as i128) * (IDCT_C_TABLE[u * 8 + x] as i128);
            }
            let s: i128 = ((acc + HALF) >> 46) + 128;
            let clamped: i128 = if s < 0 {
                0
            } else if s > 255 {
                255
            } else {
                s
            };
            out[y * 8 + x] = clamped as i32;
        }
    }
}

// ===========================================================================
//  ③ Huffman 解码（规范 §F.2.2.3 mincode/maxcode/valptr 法）
// ===========================================================================

/// Huffman 解码表（`mincode` / `maxcode` / `valptr` 三数组 + 码值表）。
#[derive(Debug, Clone)]
pub struct HuffDec {
    /// 各码长下的最小码字（下标 = 码长 1..=16）。
    pub mincode: [i32; 17],
    /// 各码长下的最大码字（`-1` 表示该码长无码字）。
    pub maxcode: [i32; 17],
    /// 各码长首个码值在 `huffval` 中的下标。
    pub valptr: [i32; 17],
    /// 码值表（BITS/VALS 的 VALS 拼接）。
    pub huffval: Vec<u8>,
}

impl HuffDec {
    /// 由 DHT 段的 `BITS[16]` + `VALS[]` 构建解码表。
    pub fn build(bits: &[u8; 16], vals: &[u8]) -> HuffDec {
        let mut mincode = [0i32; 17];
        let mut maxcode = [-1i32; 17];
        let mut valptr = [0i32; 17];
        let mut code: i32 = 0;
        let mut k: usize = 0;
        for l in 1..=16 {
            if bits[l - 1] == 0 {
                maxcode[l] = -1;
            } else {
                valptr[l] = k as i32;
                mincode[l] = code;
                code += bits[l - 1] as i32;
                maxcode[l] = code - 1;
                k += bits[l - 1] as usize;
            }
            code <<= 1;
        }
        let mut huffval: Vec<u8> = vals.to_vec();
        if huffval.len() < k {
            // 截断/损坏的 DHT：补齐零值，后续解码会因码字不匹配而报错（不静默）。
            huffval.resize(k, 0);
        }
        HuffDec { mincode, maxcode, valptr, huffval }
    }
}

/// JPEG 熵流位读取器（MSB 优先，处理 `0xFF 0x00` 字节填充与 `0xFF xx` 标记）。
///
/// 遇标记或流尾时以全 1 填充（等价 T.81 的熵段尾部 1 填充），并记录标记位置，
/// 供重启动（RSTn）重同步使用。
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
    padding: bool,
    /// 已探测到的标记 `0xFF` 所在下标（`None` 表示尚未遇到标记）。
    pending_marker: Option<usize>,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader { data, pos: 0, buf: 0, cnt: 0, padding: false, pending_marker: None }
    }

    /// 把缓冲区补到 ≥ 57 位（保证一次 `get_bits(16)` 安全）。
    #[inline]
    fn fill(&mut self) {
        while self.cnt <= 56 {
            if self.padding || self.pos >= self.data.len() {
                self.buf = (self.buf << 8) | 0xFF;
                self.cnt += 8;
                self.padding = true;
                continue;
            }
            let b: u8 = self.data[self.pos];
            self.pos += 1;
            if b == 0xFF {
                if self.pos < self.data.len() && self.data[self.pos] == 0x00 {
                    // 字节填充：0xFF 是真实数据字节，吃掉其后的 0x00。
                    self.pos += 1;
                } else {
                    // 标记：记录位置并以 1 填充（熵段在此终止）。
                    self.pending_marker = Some(self.pos - 1);
                    self.padding = true;
                    self.buf = (self.buf << 8) | 0xFF;
                    self.cnt += 8;
                    continue;
                }
            }
            self.buf = (self.buf << 8) | (b as u64);
            self.cnt += 8;
        }
    }

    #[inline]
    fn get_bit(&mut self) -> u32 {
        self.fill();
        self.cnt -= 1;
        ((self.buf >> self.cnt) & 1) as u32
    }

    #[inline]
    fn get_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.fill();
        self.cnt -= n;
        ((self.buf >> self.cnt) & ((1u64 << n) - 1)) as u32
    }

    /// 丢弃缓冲区内未用完的位（字节对齐）。
    #[inline]
    fn align(&mut self) {
        self.buf = 0;
        self.cnt = 0;
    }

    /// 跳过重启动标记 `RSTn`：字节对齐后重同步到下一个 `FFD0..FFD7` 之后。
    fn skip_restart(&mut self) {
        self.align();
        self.padding = false;
        let mut p: usize = match self.pending_marker.take() {
            Some(p) => p,
            None => self.pos,
        };
        while p + 1 < self.data.len() {
            if self.data[p] == 0xFF {
                let m: u8 = self.data[p + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos = p + 2;
                    return;
                }
                if m != 0x00 && m != 0xFF {
                    // 遇到非 RST 标记（多为 EOI）：重同步结束，剩余块以填充位解码。
                    self.pos = p;
                    return;
                }
            }
            p += 1;
        }
        self.pos = self.data.len();
    }
}

/// Huffman 解码一个符号（逐位累积码字，按码长查 mincode/maxcode）。
#[inline]
fn huff_decode(r: &mut BitReader, h: &HuffDec) -> Result<u8, String> {
    let mut code: i32 = 0;
    for l in 1..=16 {
        code = (code << 1) | (r.get_bit() as i32);
        if h.maxcode[l] >= 0 && code >= h.mincode[l] && code <= h.maxcode[l] {
            let idx: usize = (h.valptr[l] + (code - h.mincode[l])) as usize;
            return match h.huffval.get(idx) {
                Some(&v) => Ok(v),
                None => Err(format!(
                    "Huffman 解码越界（码长 {}，码值下标 {} 超出 {} 项）",
                    l,
                    idx,
                    h.huffval.len()
                )),
            };
        }
    }
    Err("Huffman 解码未匹配到码字（熵流损坏）".to_string())
}

/// 读取 `s` 位幅值并按 JPEG 规范（§F.2.2.1）还原带符号值。
#[inline]
fn receive_extend(r: &mut BitReader, s: u32) -> i32 {
    if s == 0 {
        return 0;
    }
    let v: i32 = r.get_bits(s) as i32;
    if v < (1i32 << (s - 1)) {
        v - ((1i32 << s) - 1)
    } else {
        v
    }
}

// ===========================================================================
//  ④ 容器解析（SOI / DQT / SOF0 / DHT / DRI / SOS / EOI）
// ===========================================================================

/// 单个分量（平面）的几何与样本存储。
#[derive(Debug, Clone)]
struct Component {
    /// 分量 ID（SOF 与 SOS 中的 `Ci`）。
    id: u8,
    /// 水平采样因子 `Hi`（1..=4）。
    hs: usize,
    /// 垂直采样因子 `Vi`（1..=4）。
    vs: usize,
    /// 量化表 ID `Tqi`。
    tq: usize,
    /// 平面宽（样本数）`ceil(width·Hi/Hmax)`。
    cw: usize,
    /// 平面高（样本数）`ceil(height·Vi/Vmax)`。
    ch: usize,
    /// 平面块列数 `ceil(cw/8)`。
    bw: usize,
    /// 平面块行数 `ceil(ch/8)`。
    bh: usize,
    /// 平面样本（8-bit，尚未上采样）。
    plane: Vec<u8>,
}

/// 解码后的 JPEG 图像（RGB8 行主序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegImage {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// 分量个数（1 = 灰度，3 = YCbCr）。
    pub components: usize,
    /// RGB8 像素（`width*height*3`，行主序）。
    pub rgb: Vec<u8>,
}

impl JpegImage {
    /// 展开为 RGBA8（alpha 恒 255）。
    pub fn to_rgba8(&self) -> Vec<u8> {
        let n: usize = (self.width as usize) * (self.height as usize);
        let mut out: Vec<u8> = vec![255u8; n * 4];
        for i in 0..n {
            out[i * 4] = self.rgb[i * 3];
            out[i * 4 + 1] = self.rgb[i * 3 + 1];
            out[i * 4 + 2] = self.rgb[i * 3 + 2];
            out[i * 4 + 3] = 255;
        }
        out
    }
}

/// 解码一个基线 JPEG 字节流，返回 RGB8 图像。
///
/// # 错误
/// 返回 `Err(String)`：三段式以外的简短中文说明（含损坏位置/原因），
/// **不静默降级**：不支持的编码（渐进式、算术编码、12-bit、CMYK）明确报错。
pub fn decode_baseline(data: &[u8]) -> Result<JpegImage, String> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err("不是 JPEG 字节流（缺少 SOI 标记 FFD8）".to_string());
    }

    // ---- 头部状态 ----
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut comps: Vec<Component> = Vec::new();
    let mut quant: [[i32; 64]; 4] = [[0i32; 64]; 4];
    let mut quant_set: [bool; 4] = [false; 4];
    let mut huff_dc: [Option<HuffDec>; 4] = [None, None, None, None];
    let mut huff_ac: [Option<HuffDec>; 4] = [None, None, None, None];
    let mut restart_interval: usize = 0;
    let mut scan_map: Vec<(usize, usize, usize)> = Vec::new(); // (分量下标, DC 表, AC 表)
    let mut entropy_start: usize = 0;
    let mut saw_sof: bool = false;

    let mut pos: usize = 2usize;
    'hdr: while pos + 1 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker: u8 = data[pos + 1];
        pos += 2;
        // 填充字节 FF FF ... 与无长度标记。
        if marker == 0xFF {
            pos -= 1;
            continue;
        }
        match marker {
            0xD9 => break 'hdr, // EOI
            0x01 => continue,   // TEM（无长度段）
            0xD0..=0xD7 => continue, // 游离 RSTn：忽略
            0xDA => {
                if !saw_sof {
                    return Err("SOS 出现在 SOF 之前（JPEG 结构损坏）".to_string());
                }
                if pos + 1 >= data.len() {
                    return Err("SOS 段被截断".to_string());
                }
                let len: usize = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + len > data.len() {
                    return Err("SOS 段越界（JPEG 损坏）".to_string());
                }
                let ns: usize = data[pos + 2] as usize;
                let mut p: usize = pos + 3;
                for _ in 0..ns {
                    if p + 1 >= pos + len {
                        return Err("SOS 分量描述被截断".to_string());
                    }
                    let cid: u8 = data[p];
                    let ta: u8 = data[p + 1];
                    let cidx: usize = comps
                        .iter()
                        .position(|c| c.id == cid)
                        .ok_or_else(|| format!("SOS 引用了未定义的扫描分量 ID={}", cid))?;
                    scan_map.push((cidx, (ta >> 4) as usize, (ta & 0x0F) as usize));
                    p += 2;
                }
                entropy_start = pos + len;
                break 'hdr;
            }
            _ => {
                if pos + 1 >= data.len() {
                    return Err(format!("标记 FF{:02X} 的长度字段被截断", marker));
                }
                let len: usize = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if len < 2 || pos + len > data.len() {
                    return Err(format!("标记 FF{:02X} 段越界（JPEG 损坏）", marker));
                }
                let seg: &[u8] = &data[pos + 2..pos + len];
                match marker {
                    // DQT
                    0xDB => {
                        let mut p: usize = 0;
                        while p < seg.len() {
                            let pq_tq: u8 = seg[p];
                            p += 1;
                            let pq: usize = (pq_tq >> 4) as usize;
                            let tq: usize = (pq_tq & 0x0F) as usize;
                            if tq > 3 {
                                return Err(format!("DQT 表 ID {} 越界（仅 0..3）", tq));
                            }
                            if pq == 0 {
                                if p + 64 > seg.len() {
                                    return Err("DQT（8-bit）数据被截断".to_string());
                                }
                                for k in 0..64 {
                                    // DQT 以 ZigZag 序存储 → 还原自然序。
                                    quant[tq][ZIGZAG[k]] = seg[p + k] as i32;
                                }
                                p += 64;
                            } else {
                                if p + 128 > seg.len() {
                                    return Err("DQT（16-bit）数据被截断".to_string());
                                }
                                for k in 0..64 {
                                    let v: i32 =
                                        u16::from_be_bytes([seg[p + k * 2], seg[p + k * 2 + 1]])
                                            as i32;
                                    quant[tq][ZIGZAG[k]] = v;
                                }
                                p += 128;
                            }
                            quant_set[tq] = true;
                        }
                    }
                    // SOF0（基线）/ SOF1（扩展顺序，解码同构）
                    0xC0 | 0xC1 => {
                        if seg.len() < 6 {
                            return Err("SOF 段被截断".to_string());
                        }
                        let precision: u8 = seg[0];
                        height = u16::from_be_bytes([seg[1], seg[2]]) as u32;
                        width = u16::from_be_bytes([seg[3], seg[4]]) as u32;
                        let nf: usize = seg[5] as usize;
                        if precision != 8 {
                            return Err(format!(
                                "JPEG 精度 {} bit 不在支持范围（L2：仅 8-bit 基线）",
                                precision
                            ));
                        }
                        if nf == 0 || nf > 4 {
                            return Err(format!("SOF 分量数 {} 非法（1..=4）", nf));
                        }
                        if seg.len() < 6 + nf * 3 {
                            return Err("SOF 分量描述被截断".to_string());
                        }
                        let mut p: usize = 6;
                        for _ in 0..nf {
                            let id: u8 = seg[p];
                            let hv: u8 = seg[p + 1];
                            let tq: u8 = seg[p + 2];
                            comps.push(Component {
                                id,
                                hs: (hv >> 4) as usize,
                                vs: (hv & 0x0F) as usize,
                                tq: tq as usize,
                                cw: 0,
                                ch: 0,
                                bw: 0,
                                bh: 0,
                                plane: Vec::new(),
                            });
                            p += 3;
                        }
                        if nf == 4 {
                            return Err(
                                "4 分量（CMYK/YCCK）JPEG 不在支持范围（L2）".to_string()
                            );
                        }
                        saw_sof = true;
                    }
                    // DHT
                    0xC4 => {
                        let mut p: usize = 0;
                        while p < seg.len() {
                            let tc_th: u8 = seg[p];
                            p += 1;
                            let tc: usize = (tc_th >> 4) as usize;
                            let th: usize = (tc_th & 0x0F) as usize;
                            if th > 3 {
                                return Err(format!("DHT 表 ID {} 越界（仅 0..3）", th));
                            }
                            if p + 16 > seg.len() {
                                return Err("DHT BITS 被截断".to_string());
                            }
                            let mut bits = [0u8; 16];
                            let mut total: usize = 0;
                            for i in 0..16 {
                                bits[i] = seg[p + i];
                                total += bits[i] as usize;
                            }
                            p += 16;
                            if p + total > seg.len() {
                                return Err("DHT VALS 被截断".to_string());
                            }
                            let vals: Vec<u8> = seg[p..p + total].to_vec();
                            p += total;
                            let tbl: HuffDec = HuffDec::build(&bits, &vals);
                            if tc == 0 {
                                huff_dc[th] = Some(tbl);
                            } else {
                                huff_ac[th] = Some(tbl);
                            }
                        }
                    }
                    // DRI（重启动间隔）
                    0xDD => {
                        if seg.len() >= 2 {
                            restart_interval = u16::from_be_bytes([seg[0], seg[1]]) as usize;
                        }
                    }
                    // 渐进式 / 算术编码：明确拒绝
                    0xC2 | 0xC6 | 0xCA | 0xCE => {
                        return Err(format!(
                            "JPEG 标记 FF{:02X}（渐进式/算术编码）不在支持范围\
                             （L2：仅基线顺序 DCT）",
                            marker
                        ));
                    }
                    // 其他 SOFn（无损、层次）
                    0xC3 | 0xC5 | 0xC7 | 0xC9 | 0xCB | 0xCD | 0xCF => {
                        return Err(format!(
                            "JPEG 标记 FF{:02X}（无损/层次编码）不在支持范围（L2）",
                            marker
                        ));
                    }
                    // APPn / COM / 其他：跳过
                    _ => {}
                }
                pos += len;
            }
        }
    }

    if !saw_sof {
        return Err("JPEG 缺少 SOF0 段（无法确定尺寸）".to_string());
    }
    if width == 0 || height == 0 {
        return Err("JPEG SOF0 尺寸为 0（非法图像）".to_string());
    }
    if scan_map.is_empty() {
        return Err("JPEG 缺少 SOS 扫描段（无可解码数据）".to_string());
    }
    if entropy_start >= data.len() {
        return Err("JPEG 熵编码数据为空（文件被截断）".to_string());
    }

    // ---- 分量几何 ----
    let hmax: usize = comps.iter().map(|c| c.hs).max().unwrap_or(1).max(1);
    let vmax: usize = comps.iter().map(|c| c.vs).max().unwrap_or(1).max(1);
    for c in comps.iter_mut() {
        if c.hs == 0 || c.vs == 0 {
            return Err("SOF 采样因子为 0（非法）".to_string());
        }
        if c.hs > 4 || c.vs > 4 {
            return Err(format!("SOF 采样因子 {}x{} 越界（最大 4x4）", c.hs, c.vs));
        }
        if c.tq > 3 || !quant_set[c.tq] {
            return Err(format!("分量引用了未定义的量化表 ID={}", c.tq));
        }
        c.cw = (width as usize * c.hs + hmax - 1) / hmax;
        c.ch = (height as usize * c.vs + vmax - 1) / vmax;
        c.bw = (c.cw + 7) / 8;
        c.bh = (c.ch + 7) / 8;
        c.plane = vec![0u8; c.cw * c.ch];
    }

    // ---- 熵解码 ----
    let mcus_x: usize = (width as usize + 8 * hmax - 1) / (8 * hmax);
    let mcus_y: usize = (height as usize + 8 * vmax - 1) / (8 * vmax);
    let mut pred_dc: Vec<i32> = vec![0i32; comps.len()];
    let mut reader: BitReader = BitReader::new(&data[entropy_start..]);
    let mut since_restart: usize = 0;

    // 扫描内每个分量的 (DC 表, AC 表) 与在 scan_map 中的位置。
    let scan_tables: Vec<(usize, usize, usize)> = scan_map.clone();

    if scan_tables.len() == 1 {
        // 非交错扫描：单个分量按自身块网格遍历。
        let (ci, dc_t, ac_t) = scan_tables[0];
        for by in 0..comps[ci].bh {
            for bx in 0..comps[ci].bw {
                if restart_interval > 0 && since_restart == restart_interval {
                    reader.skip_restart();
                    for p in pred_dc.iter_mut() {
                        *p = 0;
                    }
                    since_restart = 0;
                }
                decode_one_block(
                    &mut reader,
                    &huff_dc,
                    &huff_ac,
                    &mut comps,
                    &quant,
                    ci,
                    dc_t,
                    ac_t,
                    &mut pred_dc[ci],
                    bx,
                    by,
                )?;
                since_restart += 1;
            }
        }
    } else {
        // 交错扫描：按 MCU 遍历（每 MCU 覆盖各分量 Hi×Vi 个块）。
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if restart_interval > 0 && since_restart == restart_interval {
                    reader.skip_restart();
                    for p in pred_dc.iter_mut() {
                        *p = 0;
                    }
                    since_restart = 0;
                }
                for &(ci, dc_t, ac_t) in scan_tables.iter() {
                    let hs: usize = comps[ci].hs;
                    let vs: usize = comps[ci].vs;
                    for j in 0..vs {
                        for i in 0..hs {
                            let bx: usize = mx * hs + i;
                            let by: usize = my * vs + j;
                            decode_one_block(
                                &mut reader,
                                &huff_dc,
                                &huff_ac,
                                &mut comps,
                                &quant,
                                ci,
                                dc_t,
                                ac_t,
                                &mut pred_dc[ci],
                                bx,
                                by,
                            )?;
                        }
                    }
                }
                since_restart += 1;
            }
        }
    }

    // ---- 上采样 + 色彩变换 ----
    let w: usize = width as usize;
    let h: usize = height as usize;
    let mut full: Vec<Vec<u8>> = Vec::with_capacity(comps.len());
    for c in comps.iter() {
        full.push(upsample(&c.plane, c.cw, c.ch, w, h));
    }

    let mut rgb: Vec<u8> = vec![0u8; w * h * 3];
    if comps.len() == 1 {
        // 灰度：三通道复本。
        for i in 0..w * h {
            let v: u8 = full[0][i];
            rgb[i * 3] = v;
            rgb[i * 3 + 1] = v;
            rgb[i * 3 + 2] = v;
        }
    } else if comps.len() == 3 {
        for i in 0..w * h {
            let (r, g, b) = ycbcr_to_rgb(full[0][i], full[1][i], full[2][i]);
            rgb[i * 3] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
        }
    } else {
        return Err(format!(
            "{} 分量 JPEG 不在支持范围（L2：仅灰度与三通道）",
            comps.len()
        ));
    }

    Ok(JpegImage { width, height, components: comps.len(), rgb })
}

/// 解码一个 8×8 块并写入所在分量平面（越界部分裁掉，保证尺寸非 8 倍数时也正确）。
#[allow(clippy::too_many_arguments)]
fn decode_one_block(
    reader: &mut BitReader,
    huff_dc: &[Option<HuffDec>; 4],
    huff_ac: &[Option<HuffDec>; 4],
    comps: &mut [Component],
    quant: &[[i32; 64]; 4],
    ci: usize,
    dc_t: usize,
    ac_t: usize,
    pred_dc: &mut i32,
    bx: usize,
    by: usize,
) -> Result<(), String> {
    let hdc: &HuffDec = match huff_dc.get(dc_t).and_then(|o| o.as_ref()) {
        Some(t) => t,
        None => return Err(format!("缺少 DC Huffman 表 ID={}", dc_t)),
    };
    let hac: &HuffDec = match huff_ac.get(ac_t).and_then(|o| o.as_ref()) {
        Some(t) => t,
        None => return Err(format!("缺少 AC Huffman 表 ID={}", ac_t)),
    };

    // ---- 系数（ZigZag 序）----
    let mut zz = [0i32; 64];
    let s_dc: u32 = huff_decode(reader, hdc)? as u32;
    let dc: i32 = *pred_dc + receive_extend(reader, s_dc);
    *pred_dc = dc;
    zz[0] = dc;
    let mut k: usize = 1;
    while k < 64 {
        let rs: u8 = huff_decode(reader, hac)?;
        let run: usize = (rs >> 4) as usize;
        let size: u32 = (rs & 0x0F) as u32;
        if size == 0 {
            if run == 0 {
                break; // EOB
            }
            k += 16; // ZRL：跳过 16 个零
            continue;
        }
        k += run;
        if k >= 64 {
            break;
        }
        zz[k] = receive_extend(reader, size);
        k += 1;
    }

    // ---- ZigZag → 自然序 ----
    let mut coef = [0i32; 64];
    for i in 0..64 {
        coef[ZIGZAG[i]] = zz[i];
    }

    // ---- 定点 IDCT ----
    let mut out = [0i32; 64];
    let tq: usize = comps[ci].tq;
    let qtab: &[i32; 64] = &quant[tq];
    idct_8x8_fixed(&mut out, &coef, qtab);

    // ---- 写回平面（按块坐标裁剪）----
    let c: &mut Component = &mut comps[ci];
    let base_x: usize = bx * 8;
    let base_y: usize = by * 8;
    if base_y >= c.ch {
        return Ok(()); // 该块完全落在平面外（尺寸裁剪块）
    }
    for yy in 0..8 {
        let py: usize = base_y + yy;
        if py >= c.ch {
            break;
        }
        for xx in 0..8 {
            let px: usize = base_x + xx;
            if px >= c.cw {
                break;
            }
            let v: i32 = out[yy * 8 + xx];
            c.plane[py * c.cw + px] = v.clamp(0, 255) as u8;
        }
    }
    Ok(())
}

// ===========================================================================
//  ⑤ 上采样与色彩变换
// ===========================================================================

/// 分量平面 → 全分辨率平面。
///
/// * 比例 1:1（4:4:4）→ 直通（必要时裁剪）；
/// * 2× 放大 → **三角形滤波**（libjpeg "fancy upsampling" 同构的整数权重
///   `(3·s[i] + s[i±1] + 2) >> 2`），与主流解码器逐像素贴近，互操 PSNR 更高；
/// * 其他整数比例 → 最近邻（`x·sw/dw`），实现最简、结果确定。
fn upsample(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == 0 || sh == 0 {
        return vec![128u8; dw * dh];
    }
    // 水平：sw → dw。
    let mid: Vec<u8> = if sw == dw {
        src.to_vec()
    } else if dw == sw * 2 {
        let mut out = vec![0u8; dw * sh];
        for y in 0..sh {
            let row: &[u8] = &src[y * sw..y * sw + sw];
            let dst: &mut [u8] = &mut out[y * dw..y * dw + dw];
            for i in 0..sw {
                let left: u32 = row[if i == 0 { 0 } else { i - 1 }] as u32;
                let cur: u32 = row[i] as u32;
                let right: u32 = row[if i + 1 >= sw { sw - 1 } else { i + 1 }] as u32;
                dst[2 * i] = ((3 * cur + left + 2) >> 2) as u8;
                dst[2 * i + 1] = ((3 * cur + right + 2) >> 2) as u8;
            }
        }
        out
    } else {
        let mut out = vec![0u8; dw * sh];
        for y in 0..sh {
            for x in 0..dw {
                let sx: usize = (x * sw / dw).min(sw - 1);
                out[y * dw + x] = src[y * sw + sx];
            }
        }
        out
    };

    // 垂直：sh → dh。
    if sh == dh {
        return mid;
    }
    if dh == sh * 2 {
        let mut out = vec![0u8; dw * dh];
        for y in 0..sh {
            let cur_row: &[u8] = &mid[y * dw..y * dw + dw];
            let up: &[u8] = &mid[if y == 0 { 0 } else { y - 1 } * dw..][..dw];
            let down: &[u8] = &mid[if y + 1 >= sh { sh - 1 } else { y + 1 } * dw..][..dw];
            for x in 0..dw {
                let c: u32 = cur_row[x] as u32;
                out[(2 * y) * dw + x] = ((3 * c + up[x] as u32 + 2) >> 2) as u8;
                out[(2 * y + 1) * dw + x] = ((3 * c + down[x] as u32 + 2) >> 2) as u8;
            }
        }
        return out;
    }
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy: usize = (y * sh / dh).min(sh - 1);
        out[y * dw..y * dw + dw].copy_from_slice(&mid[sy * dw..sy * dw + dw]);
    }
    out
}

/// YCbCr → RGB（2^16 定点乘加，无超越函数）。
///
/// `R = Y + 1.402·(Cr−128)`、`G = Y − 0.344136·(Cb−128) − 0.714136·(Cr−128)`、
/// `B = Y + 1.772·(Cb−128)`；系数按 `round(65536·k)` 定点化。
#[inline]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    const K_R: i32 = 91881; // 1.402 · 2^16
    const K_G_CB: i32 = 22554; // 0.344136 · 2^16
    const K_G_CR: i32 = 46802; // 0.714136 · 2^16
    const K_B: i32 = 116130; // 1.772 · 2^16

    let yv: i32 = y as i32;
    let cbv: i32 = cb as i32 - 128;
    let crv: i32 = cr as i32 - 128;
    let r: i32 = yv + ((K_R * crv) >> 16);
    let g: i32 = yv - ((K_G_CB * cbv + K_G_CR * crv) >> 16);
    let b: i32 = yv + ((K_B * cbv) >> 16);
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
}

/// 解码并直接返回 `(width, height, RGBA8)`（门面/上层便捷入口）。
pub fn decode_baseline_rgba(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let img: JpegImage = decode_baseline(data)?;
    Ok((img.width, img.height, img.to_rgba8()))
}

// ===========================================================================
//  ⑥ 单元测试（模块内：常量与 IDCT 内核）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 全零系数块：输出恒 128（电平偏移中点）。
    #[test]
    fn idct_zero_block_yields_128() {
        let coef = [0i32; 64];
        let q = [1i32; 64];
        let mut out = [0i32; 64];
        idct_8x8_fixed(&mut out, &coef, &q);
        assert!(out.iter().all(|&v| v == 128), "全零块必须恒为 128，实际 {:?}", out);
    }

    /// 阈值两侧各一块：8191·2^16 走 i64（阈值内），8192·2^16 走 i128（超界一单位）。
    #[test]
    fn idct_threshold_boundary_dispatch() {
        // D[0]·W[0] = (coef·q)·65536? W[0] = 32768（u=v=0），故取 coef·q 使 D·W 恰达边界。
        // 目标 max|D·W| = 8191·2^16 与 8192·2^16：W[0]=32768=2^15 ⇒ D = 2·8191 与 2·8192。
        let mut below = [0i32; 64];
        let mut above = [0i32; 64];
        below[0] = 2 * 8191;
        above[0] = 2 * 8192;
        let q = [1i32; 64];

        let (_, max_below) = dequantize(&below, &q);
        let (_, max_above) = dequantize(&above, &q);
        assert_eq!(max_below, 8191 * (1 << 16), "阈值内块 max|D·W| 应为 8191·2^16");
        assert_eq!(max_above, 8192 * (1 << 16), "超界块 max|D·W| 应为 8192·2^16");
        assert!(max_below <= IDCT_STAGE2_I64_THRESHOLD, "8191·2^16 应落入 i64 快路径");
        assert!(max_above > IDCT_STAGE2_I64_THRESHOLD, "8192·2^16 必须走 i128");

        // 两条路径对同一输入必须逐位一致（阈值处不得出现语义差）。
        let mut a = [0i32; 64];
        let mut b = [0i32; 64];
        idct_8x8_fixed(&mut a, &above, &q);
        idct_8x8_fixed_wide(&mut b, &above, &q);
        assert_eq!(a, b, "超阈块 i64/i128 路径输出必须一致");
    }

    /// W 表值域与 const assert 一致性（编译期已断言，此处再核一份运行期）。
    #[test]
    fn w_table_domain() {
        for &w in IDCT_W_TABLE.iter() {
            assert!(w == 32768 || w == 46341 || w == 65536);
        }
        assert_eq!(IDCT_W_TABLE[0], 32768, "W[0][0] = 2^16·c(0)·c(0) = 32768");
        assert_eq!(IDCT_W_TABLE[9], 65536, "W[1][1] = 2^16");
        assert_eq!(IDCT_W_TABLE[1], 46341, "W[0][1] = 2^16/√2");
    }

    /// 色彩变换：中性灰（Cb=Cr=128）必须原样通过。
    #[test]
    fn ycbcr_neutral_is_identity() {
        for y in [0u8, 1, 64, 128, 200, 255] {
            assert_eq!(ycbcr_to_rgb(y, 128, 128), (y, y, y));
        }
    }

    /// 上采样：2× 三角形滤波端点行为（首像素 = 原首像素附近）。
    #[test]
    fn upsample_2x_endpoints() {
        let src = [10u8, 20, 30, 40];
        let up = upsample(&src, 4, 1, 8, 2);
        assert_eq!(up.len(), 16);
        // 水平 2×：out[0] = (3·10 + 10 + 2)>>2 = 10。
        assert_eq!(up[0], 10);
    }
}
