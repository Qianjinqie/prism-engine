//! ITU-T T.81（JPEG 1.0）**无损模式**解码器 —— DNG `Compression = 7` 的压缩层。
//!
//! # 权威依据
//!
//! 本模块的每一条规则都逐字对照 **ITU-T T.81 (1992E)** 原文：
//!
//! - **Annex H, H.1.2.1「Prediction」+ Table H.1** —— 7 个固定预测器；
//! - **H.1.2.2 + Table H.2 + F.1.2.1 EXTEND** —— 差值的 Huffman 编码；
//! - **B.2.3 + Table B.3（lossless 列）** —— 扫描头 `Ss/Se/Ah/Al` 语义；
//! - **B.2.4.2 / Table B.5 + Annex C** —— `DHT` 的 canonical Huffman 构造；
//! - **B.2.4.2 `Ss`** —— 预测器选择值（**无损模式下 `Ss` 就是预测器**）。
//!
//! # ⚠️ 与任务书伪码的差异（必须记录在案）
//!
//! 任务书第 39–49 行给了一个「标准七点（PNG/JPEG lossless 混合）优先级」的
//! 4 分支梯度比较伪码：
//!
//! ```text
//! 1. if |A + B - C| 较小 → P = A + B - C
//! 2. else if |B - C| 较小 → P = B - C
//! 3. else if |C - D| 较小 → P = C - D
//! 4. else → P = A
//! ```
//!
//! **这个算法在 T.81 里根本不存在。** 它属于 JPEG-LS / HEIF / FLAC 那一族
//! （「MEDIAN 预测 + 局部梯度逃逸」）的思路，**不是** lossless JPEG
//! （H.81 Annex H / process 14）。T.81 Annex H 里**一个 `abs()` 都没有**，
//! 更没有 4 分支优先级比较：T.81 用的是**扫描头 `Ss` 里选定的 7 个固定预测器
//! 之一**，每个样本**不做任何梯度比较**。
//!
//! 任务书第 50 行提到的「T.81 用 `MEDIAN`/`ABS` 宏的模运算，直接写 `abs()`
//! 会与规范差 1」在 T.81 Annex H 中**同样查无此宏**。
//!
//! 本模块实现的是 **T.81 原文 Table H.1**，并在 [`Predictor::predict_i32`]
//! 的注释里把每一行表达式与 Table H.1 逐条对齐。
//!
//! # 那个「差 1 的坑」的真身：除以 2 的取整方向
//!
//! T.81 原文（Table H.1 注 a 与紧接其后的正文）**明文**写了：
//!
//! > For simplicity of implementation, the divide by 2 in the prediction
//! > selections 5 and 6 of Table H.1 is done by an arithmetic-right-shift of
//! > the integer values.
//!
//! 即 selections 5/6 的「÷2」是**算术右移（floor）**，**不是** C 的 `/2`
//! （向零截断）。在 Rust 里 `(x >> 1)` 与 `(x / 2)` 只在 `x >= 0` 时相等；
//! `x < 0` 且为奇数时**差 1**。例如 `x = -3`：`>>1` 得 `-2`，`/2` 得 `-1`。
//! 因为 `Rb - Rc` 与 `Ra - Rc` **可以为负**，这个坑是真实存在的。
//! 测试 [`predictor_shift_is_floor_not_trunc`] 用 `Ra=100, Rb=0, Rc=1`
//! （`Rb - Rc = -1`，负奇数）把它锁死。
//!
//! 另一个「差 1」的候选 —— `EXTEND` 的分支条件 —— 见 [`extend`] 的注释。
//!
//! # 全程整数精度、不钳位
//!
//! > Each prediction is calculated with full integer arithmetic precision, and
//! > without clamping of either underflow or overflow beyond the input
//! > precision bounds. For example, if Ra and Rb are both 16-bit integers, the
//! > sum is a 17-bit integer.
//!
//! 所以 `Px = Ra + Rb - Rc` 在 16 位下可能是 **17 位中间值**（最大 131070）。
//! 本模块用 `i32` 承载 `Px`，**全程不钳位**，只在最后一步按 **modulo 2^16**
//! 收口（见 [`decode_scan`]）。
//!
//! # 多分量（`Nf`/`Ns` = 1..=4）与 CFA 重组 —— 权威依据
//!
//! DNG 的 CFA 图常把 Bayer 阵列**重打包**成多分量的无损 JPEG，让预测器
//! 「跨同色像素」工作。本模块据此支持 1..=4 个分量，并按下述**三条相互独立
//! 且彼此吻合**的依据做 CFA 重组：
//!
//! 1. **DNG 1.6 规范 `Compression`（=7）标签描述**：
//!
//!    > For lossless JPEG, the internal width/length/components in the JPEG
//!    > stream are **not required to match** the strip or tile's
//!    > width/length/components. Only the total sample counts need to match.
//!    > **It is common for CFA images to be encoded with a different width,
//!    > length or component count to allow the JPEG compression predictors to
//!    > work across like colors.**
//!
//! 2. **Adobe DNG SDK / `dcraw` 的实际打包口径**（业界一致描述）：
//!
//!    > Adobe's DNG converter packs a Bayer mosaic into a lossless-JPEG image
//!    > whose component count equals the number of CFA columns in the tile
//!    > width divisor (commonly 2): the JPEG is `cfa_width/2` wide with 2
//!    > components, and the two components are the **left/right sample of
//!    > every horizontal pixel pair**.
//!
//! 3. **ITU-T T.81 B.2.3 的非交错扫描语义**：当 `Ns == Nf` 且每个分量的采样
//!    因子为 1×1 时，一行内**按分量顺序 `0,1,…,Ns-1` 轮转**取 DC 码，每个分量
//!    在一行内恰好取 `ceil(X / Ns)` 个样本（T.81 B.2.3 + A.2.3 的 MCU 定义；
//!    在本例中 `H_i = V_i = 1` ⇒ 一个 MCU = 一行内的 `Ns` 个样本）。
//!
//! 三条依据共同锁定的**唯一**重组式是：
//!
//! ```text
//! cfa_row[y][x] = component[x % Ns][x / Ns]      （同一行内）
//! ```
//!
//! 即分量 `c`（`0..Ns`）承载**列号 ≡ c (mod Ns)** 的那些样本；相邻同色像素
//! 相距 `Ns` 列（Bayer 下 R/G/B/G 的相邻同色相距 2 列 ⇒ `Ns = 2` 是「左右对」）。
//! 这正是 [`deinterleave_components`] 实现的映射。

#![allow(clippy::needless_range_loop)]

use crate::raw::{tiff_err, unsupported_err, RawResult};

// ===========================================================================
//  一、marker 常量（T.81 B.2 表 B.1）
// ===========================================================================

/// `SOI`（Start Of Image，T.81 B.2.1）。
pub(crate) const M_SOI: u8 = 0xD8;
/// `EOI`（End Of Image）。
pub(crate) const M_EOI: u8 = 0xD9;
/// `SOF3`（Start Of Frame，**无损** Huffman 模式；T.81 B.2.2）。
pub(crate) const M_SOF3: u8 = 0xC3;
/// `DHT`（Define Huffman Table，T.81 B.2.4.2）。
pub(crate) const M_DHT: u8 = 0xC4;
/// `SOS`（Start Of Scan，T.81 B.2.3）。
pub(crate) const M_SOS: u8 = 0xDA;
/// `DRI`（Define Restart Interval，T.81 B.2.4.1）。
pub(crate) const M_DRI: u8 = 0xDD;
/// `RST0`（Restart marker `FFD0`，T.81 B.2.1 / Table B.1）。
pub(crate) const M_RST0: u8 = 0xD0;
/// `RST7`（Restart marker `FFD7`）。
const M_RST7: u8 = 0xD7;
/// `COM`（Comment，T.81 B.2.5）—— 有长度字段，可跳过。
const M_COM: u8 = 0xFE;
/// `APPn` 段首（`0xE0..=0xEF`，有长度字段，可跳过）。
const M_APP0: u8 = 0xE0;

/// 解码样本数上限（**C18 防御**）：1 << 28 = 268,435,456 样本 ≈ 100 MP。
///
/// T.81 的 `X`/`Y` 各为 `u16`（最大 65535），理论上限 ~4.29e9 样本 = 8 GB，
/// 一次 `vec![0u16; n]` 会直接 OOM abort（违反 C18「永不 panic」）。真实相机
/// 最大约 100 MP（Sony A9 III 约 61 MP、DNG 全域 100 MP 级），1 << 28 有
/// 2.7 倍余量，故在此结构化拒绝。
const MAX_SAMPLES: usize = 1 << 28;

// ===========================================================================
//  二、预测器 —— T.81 Table H.1（逐条对照）
// ===========================================================================

/// T.81 **Table H.1「Predictors for lossless coding」** 的 7 个选择值。
///
/// > The allowed predictors, one of which is selected in the scan header, are
/// > listed in Table H.1.
/// >
/// > | Selection-value | Prediction |
/// > |---|---|
/// > | 0 | No prediction (See Annex J) |
/// > | 1 | Px = Ra |
/// > | 2 | Px = Rb |
/// > | 3 | Px = Rc |
/// > | 4 | Px = Ra + Rb – Rc |
/// > | 5 | Px = Ra + ((Rb – Rc)/2) a) |
/// > | 6 | Px = Rb + ((Ra – Rc)/2) a) |
/// > | 7 | Px = (Ra + Rb)/2 |
/// > a) Shift right arithmetic operation
///
/// > Selection-value 0 shall only be used for differential coding in the
/// > hierarchical mode of operation. Selections 1, 2 and 3 are one-dimensional
/// > predictors and selections 4, 5, 6, and 7 are two-dimensional predictors.
///
/// **本枚举刻意不含 selection 0**：T.81 规定它「shall only be used for
/// differential coding in the hierarchical mode」，而 DNG 用的是**顺序无损
/// 模式**（SOF3），故 `Ss == 0` 一律**结构化拒绝**（见 [`Predictor::from_selection`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    /// `Selection-value 1`：`Px = Ra`（一维，左）。
    Ra,
    /// `Selection-value 2`：`Px = Rb`（一维，上）。
    Rb,
    /// `Selection-value 3`：`Px = Rc`（一维，左上）。
    Rc,
    /// `Selection-value 4`：`Px = Ra + Rb – Rc`（二维）。
    RaRbRc,
    /// `Selection-value 5`：`Px = Ra + ((Rb – Rc)/2)`（二维，**算术右移**）。
    RaHalfRbRc,
    /// `Selection-value 6`：`Px = Rb + ((Ra – Rc)/2)`（二维，**算术右移**）。
    RbHalfRaRc,
    /// `Selection-value 7`：`Px = (Ra + Rb)/2`（二维）。
    Average,
}

impl Predictor {
    /// T.81 Table H.1 的最大合法 selection-value。
    pub const MAX_SELECTION: u8 = 7;

    /// 由扫描头 `Ss`（T.81 B.2.3「Table B.3 lossless 列」）构造预测器。
    ///
    /// T.81 B.2.3 对 `Ss` 在无损模式下的原文：
    ///
    /// > Ss: … **In the lossless mode of operations this parameter is used to
    /// > select the predictor.**
    ///
    /// 即 **`Ss` 就是预测器选择值（1..=7），必须读并使用**。
    ///
    /// # 拒绝
    ///
    /// - `Ss == 0`：T.81 Table H.1 注「shall only be used for differential
    ///   coding in the hierarchical mode」——顺序 SOF3 无损里**非法**；
    /// - `Ss > 7`：超出 Table H.1 的 7 个合法选择值。
    pub fn from_selection(sel: u8) -> RawResult<Self> {
        const LOC: &str = "raw_lossless_jpeg::Predictor::from_selection";
        match sel {
            1 => Ok(Self::Ra),
            2 => Ok(Self::Rb),
            3 => Ok(Self::Rc),
            4 => Ok(Self::RaRbRc),
            5 => Ok(Self::RaHalfRbRc),
            6 => Ok(Self::RbHalfRaRc),
            7 => Ok(Self::Average),
            0 => Err(tiff_err(
                LOC,
                "扫描头 Ss = 0（No prediction）",
                "T.81 Table H.1 规定 selection 0 仅用于 hierarchical 模式的差分编码；\
                 DNG 的 Compression=7 是顺序无损（SOF3），Ss 必须在 1..=7",
            )),
            other => Err(tiff_err(
                LOC,
                format!("扫描头 Ss = {other}，超出 T.81 Table H.1 的 0..=7"),
                "DNG lossless JPEG 的 Ss（预测器选择）必须是 1..=7",
            )),
        }
    }

    /// 本预测器对应的 `Selection-value`（Table H.1 的第 1 列）。
    pub fn selection(self) -> u8 {
        match self {
            Self::Ra => 1,
            Self::Rb => 2,
            Self::Rc => 3,
            Self::RaRbRc => 4,
            Self::RaHalfRbRc => 5,
            Self::RbHalfRaRc => 6,
            Self::Average => 7,
        }
    }

    /// T.81 Table H.1 的预测式，**全程 i32、不钳位**。
    ///
    /// `ra` / `rb` / `rc` 分别是**已重建**的左 / 上 / 左上样本
    /// （T.81 H.1.2.1：「Ra, Rb, and Rc [are] the reconstructed samples
    /// immediately to the left, immediately above, and diagonally to the left
    /// of the current sample」）。
    ///
    /// # 与 Table H.1 的逐条对照
    ///
    /// | selection | Table H.1 原文 | 本函数表达式 |
    /// |---|---|---|
    /// | 1 | `Px = Ra` | `ra` |
    /// | 2 | `Px = Rb` | `rb` |
    /// | 3 | `Px = Rc` | `rc` |
    /// | 4 | `Px = Ra + Rb – Rc` | `ra + rb - rc` |
    /// | 5 | `Px = Ra + ((Rb – Rc)/2)` a) | `ra + ((rb - rc) >> 1)` |
    /// | 6 | `Px = Rb + ((Ra – Rc)/2)` a) | `rb + ((ra - rc) >> 1)` |
    /// | 7 | `Px = (Ra + Rb)/2` | `(ra + rb) >> 1` |
    ///
    /// a) T.81 Table H.1 注：**"Shift right arithmetic operation"**。
    ///
    /// # 为什么是 `>> 1` 而不是 `/ 2`（D-31 纪律）
    ///
    /// T.81 在 Table H.1 之后**明文**规定：
    ///
    /// > For simplicity of implementation, the divide by 2 in the prediction
    /// > selections 5 and 6 of Table H.1 is done by an **arithmetic-right-shift**
    /// > of the integer values.
    ///
    /// Rust 的 `i32 >> 1` 是**算术右移**（补符号位，即 floor 除）；
    /// `i32 / 2` 是**向零截断**。二者在 `x >= 0` 时相等，在 `x < 0` 且为奇数时
    /// **差 1**：`x = -3` 时 `>>1` 得 `-2`，`/2` 得 `-1`。
    /// `Rb - Rc` 与 `Ra - Rc` 可以为负 ⇒ **这个坑真实存在**。
    /// 本函数一律用 `>> 1`，绝不用 `/ 2`。
    ///
    /// selection 7 的 `(Ra + Rb)` 恒 `>= 0`（两者都是已重建的 `u16` 样本），
    /// 故 `>> 1` 与 `/ 2` 等价；仍写 `>> 1` 以**贴合规范原文的「shift right」
    /// 措辞**，并使 7 个选择值的实现风格统一。
    ///
    /// # 不钳位（T.81 原文）
    ///
    /// > Each prediction is calculated with full integer arithmetic precision,
    /// > and without clamping of either underflow or overflow beyond the input
    /// > precision bounds. For example, if Ra and Rb are both 16-bit integers,
    /// > the sum is a 17-bit integer.
    ///
    /// 故 `Px` 用 `i32` 承载：`Px = Ra + Rb – Rc` 在 16 位输入下最大 131070、
    /// 最小 -65535，远在 `i32` 范围内，**不做任何钳位**；modulo 2^16 只在
    /// [`decode_scan`] 的重建步收口。
    #[inline]
    pub fn predict_i32(self, ra: i32, rb: i32, rc: i32) -> i32 {
        match self {
            // selection 1: Px = Ra
            Self::Ra => ra,
            // selection 2: Px = Rb
            Self::Rb => rb,
            // selection 3: Px = Rc
            Self::Rc => rc,
            // selection 4: Px = Ra + Rb - Rc
            Self::RaRbRc => ra + rb - rc,
            // selection 5: Px = Ra + ((Rb - Rc) / 2)，a) shift right arithmetic operation
            Self::RaHalfRbRc => ra + ((rb - rc) >> 1),
            // selection 6: Px = Rb + ((Ra - Rc) / 2)，a) shift right arithmetic operation
            Self::RbHalfRaRc => rb + ((ra - rc) >> 1),
            // selection 7: Px = (Ra + Rb) / 2
            Self::Average => (ra + rb) >> 1,
        }
    }
}

// ===========================================================================
//  三、差值的 EXTEND —— T.81 F.1.2.1 + Table H.2
// ===========================================================================

/// T.81 Table H.2 中 `SSSS` 的**最大值**（比常规 DC 表的 11 多一格）。
///
/// > The table for DC coding contained in Tables F.1 and F.6 is extended by one
/// > additional entry. **No extra bits are appended after SSSS = 16 is encoded.**
pub const MAX_SSSS: u8 = 16;

/// T.81 F.1.2.1 `EXTEND`：把 `SSSS` 个附加比特还原成有符号差值。
///
/// # 算法（T.81 F.1.2.1 / Figure F.12 的 `EXTEND`）
///
/// 1. `RECEIVE(SSSS)`：从熵编码段取 `SSSS` 个比特，**MSB first**；
///    **`SSSS == 0` 时 `DIFF` 直接为 0**（无附加比特）。
/// 2. `SSSS == 16`（Table H.2 新增项）：**恰好 32768，无附加比特**。
/// 3. 其余 `1..=15`：令 `Vt = 2^SSSS - 1`；若 `V < 2^(SSSS-1)`（即最高位
///    为 0，表示负数），则 `DIFF = V - Vt`；否则 `DIFF = V`。
///
/// # 与 Table H.2 的取值范围逐条对照
///
/// Table H.2（lossless DC 表）在 `SSSS = 1..15` 时的差值范围是
/// `±(2^(SSSS−1) .. 2^SSSS − 1)`：
///
/// | `SSSS` | `Vt = 2^SSSS−1` | 负分支 `V ∈ [0, 2^(SSSS−1))` → `V − Vt` | 正分支 `V ≥ 2^(SSSS−1)` |
/// |---|---|---|---|
/// | 1 | 1 | `V=0` → `-1` | `V=1` → `+1` |
/// | 2 | 3 | `V=0,1` → `-3,-2` | `V=2,3` → `+2,+3` |
/// | 3 | 7 | `V=0..3` → `-7,-6,-5,-4` | `V=4..7` → `+4..+7` |
/// | 15 | 32767 | `V=0..16383` → `-32767..-16384` | `V=16384..32767` → `+16384..+32767` |
/// | 16 | — | **特殊：恰好 32768，无附加比特** | — |
///
/// 可见负分支恰好覆盖 `[-(2^SSSS−1), -2^(SSSS−1)]`、正分支恰好覆盖
/// `[2^(SSSS−1), 2^SSSS−1]`，两分支**无重叠、无空隙**，与 Table H.2 的
/// 「差值范围」栏逐格吻合。
///
/// # ⚠️ 关于「`V + Vt + 1`」那个坑
///
/// T.81 F.1.2.1 的 `EXTEND` 伪码在不同转写里常被写成
/// 「`Vt = 2^T − 1`；若 `V < Vt` 则 `V = V + Vt + 1`」。**这个写法是错的**
/// （`T = 1, V = 0` 会得到 `0 + 1 + 1 = 2`，而 Table H.2 明确 `SSSS = 1`
/// 的范围是 `±1`，不含 `+2`）。正确的算术是 `V − Vt`，等价于
/// `V + ((-1) << T) + 1`（因为 `-(2^T−1) = -2^T + 1`）。
/// 本函数用 `v - vt` 这一唯一与 Table H.2 逐格自洽、且与 libjpeg 的
/// `HUFF_EXTEND` 逐位一致的写法。
///
/// 注：modulo 2^16 下 `+32768 ≡ -32768`，故 `SSSS = 16` 取 `+32768` 与取
/// `-32768` 对重建结果**完全等价**；本函数按任务书与 Table H.2 取 `+32768`。
#[inline]
fn extend(br: &mut BitReader<'_>, ssss: u8) -> RawResult<i32> {
    if ssss == 0 {
        // SSSS == 0 → DIFF = 0，无附加比特。
        return Ok(0);
    }
    if ssss == MAX_SSSS {
        // Table H.2 新增项：SSSS = 16 → 恰好 32768，**不追加任何比特**。
        return Ok(32_768);
    }
    if ssss > MAX_SSSS {
        return Err(unsupported_err(
            "raw_lossless_jpeg::extend",
            format!("SSSS = {ssss}，超出 T.81 Table H.2 的上限 16"),
            "DHT 里出现 > 16 的 DC 符号属于非法表；请确认 DHT 未被错位解析",
        ));
    }
    // RECEIVE(SSSS)：取 SSSS 个比特，MSB first。
    let v = br.take(ssss)? as i32;
    // Vt = 2^T - 1
    let vt = (1i32 << ssss) - 1;
    // 分支条件：最高位为 0（即 V < 2^(T-1)）表示负数。
    if v < (1i32 << (ssss - 1)) {
        Ok(v - vt)
    } else {
        Ok(v)
    }
}

// ===========================================================================
//  四、位读取器 —— T.81 B.2.4.2 的字节填充（byte stuffing）
// ===========================================================================

/// 熵编码段的位读取器，处理 T.81 B.2.4.2 的 **`FF 00` 字节填充**。
///
/// T.81 B.2.4.2 原文（`SOF` 之后的熵编码数据）：
///
/// > In the entropy-coded segment, any occurrence of the marker prefix `FF`
/// > shall be followed by a stuffed zero bit, i.e. `FF 00`.
///
/// 于是在熵编码段里：
/// - `FF 00` → **数据字节 `FF`**（须跳过填充的 `00`）；
/// - `FF xx`（`xx != 00`）→ **标记**，熵编码段结束；
/// - 裸 `FF` 后面跟 `00` 之外的任何值 → 同样是标记。
struct BitReader<'a> {
    /// 整个 JPEG 字节流（熵编码段从 `data[pos]` 开始）。
    data: &'a [u8],
    /// 下一待读字节的偏移。
    pos: usize,
    /// 位缓冲（已读入但未消耗的比特，左对齐无关，统一按 `nbits` 取低段）。
    buf: u32,
    /// 缓冲里剩余的比特数。
    nbits: u32,
    /// 已读比特总数（诊断用）。
    consumed: u64,
    /// 熵编码段结束的原因（供错误信息区分「截断」与「遇到标记」）。
    end: EndReason,
}

/// 熵编码段终止原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndReason {
    /// 仍在读。
    Open,
    /// 遇到 `FF xx`（`xx != 00`）标记。
    Marker(u8),
    /// 数据用尽（被截断）。
    Truncated,
}

impl<'a> BitReader<'a> {
    /// 从 `pos` 处开始构造。
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            buf: 0,
            nbits: 0,
            consumed: 0,
            end: EndReason::Open,
        }
    }

    /// 取一个**数据字节**（解开 `FF 00` 填充）。
    ///
    /// 返回 `None` 表示熵编码段已结束（遇标记或数据用尽），
    /// 原因记录在 [`BitReader::end`]。
    fn data_byte(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        if b != 0xFF {
            return Some(b);
        }
        match self.data.get(self.pos).copied() {
            // FF 00 → 数据字节 FF（跳过填充的 00）。
            Some(0x00) => {
                self.pos += 1;
                Some(0xFF)
            }
            // FF xx（xx != 00）→ 标记，段结束。
            Some(m) => {
                self.pos -= 1; // 把标记留���解析器
                self.end = EndReason::Marker(m);
                None
            }
            // 裸 FF 收尾 → 截断。
            None => {
                self.end = EndReason::Truncated;
                None
            }
        }
    }

    /// 取 1 个比特（0 / 1）。
    fn bit(&mut self) -> RawResult<u32> {
        if self.nbits == 0 {
            let Some(b) = self.data_byte() else {
                return Err(self.end_error());
            };
            self.buf = b as u32;
            self.nbits = 8;
        }
        self.nbits -= 1;
        self.consumed += 1;
        Ok((self.buf >> self.nbits) & 1)
    }

    /// 取 `n` 个比特（`n <= 16`），**MSB first**（T.81 B.2.4.2 的比特序）。
    #[inline]
    fn take(&mut self, n: u8) -> RawResult<u32> {
        debug_assert!(n <= 16, "take 只支持 ≤ 16 比特");
        let mut v: u32 = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    /// 熵编码段提前结束时的结构化错误。
    fn end_error(&self) -> crate::raw::RawError {
        const LOC: &str = "raw_lossless_jpeg::BitReader";
        match self.end {
            EndReason::Marker(m) => tiff_err(
                LOC,
                format!(
                    "熵编码段在偏移 {}（已读 {} 比特）遇到标记 FF{m:02X} 而提前结束",
                    self.pos, self.consumed
                ),
                "DNG 的每个 strip/tile 都是完整 JPEG 流；此字节流可能被截断或不是无损流",
            ),
            EndReason::Truncated => tiff_err(
                LOC,
                format!(
                    "熵编码段在偏移 {}（已读 {} 比特）数据用尽",
                    self.pos, self.consumed
                ),
                "strip/tile 字节数不足；确认 DNG 未被截断",
            ),
            EndReason::Open => tiff_err(
                LOC,
                "熵编码段读取状态不一致",
                "这是内部状态错误，请提交 issue",
            ),
        }
    }

    /// 对齐到下一个字节边界并消费一个 `RSTn` marker（T.81 B.2.4.1 / H.1.2.1）。
    ///
    /// # 依据
    ///
    /// T.81 B.2.4.1（`DRI`）与 H.1.2.1 规定：在每个 restart interval 的边界，
    /// 编码器会**丢弃不足一字节的填充位**，写出一个 `RSTn`（`FFD0..FFD7`，
    /// `n` 按 0..7 **循环**递增），并**重置**所有预测器状态。解码器必须：
    ///
    /// 1. **丢弃**位缓冲中剩余的比特（不足一字节的部分）；
    /// 2. 从当前字节边界**消费** `FF Dn` 两个字节；
    /// 3. 校验 `n` 与**期望**的序号一致（`expected` 由调用方的 RST 计数器给出，
    ///    按 0..7 循环）—— 序号不连续说明字节流错位，**必须结构化拒绝**
    ///    （否则会静默错位，比报错更坏）。
    ///
    /// `expected_high` 是该 restart marker 的序号（0..=7）。
    fn restart(&mut self, expected: u8) -> RawResult<()> {
        const LOC: &str = "raw_lossless_jpeg::BitReader::restart";
        // 1) 丢弃填充位 + 清缓冲，回到未读入新字节的状态。
        self.buf = 0;
        self.nbits = 0;
        // 2) 期望位置处是 `FF Dn`。允许 `FF` 填充字节（T.81 B.2.1）。
        loop {
            let b = *self
                .data
                .get(self.pos)
                .ok_or_else(|| self.truncated_restart(expected))?;
            self.pos += 1;
            if b == 0xFF {
                let m = *self
                    .data
                    .get(self.pos)
                    .ok_or_else(|| self.truncated_restart(expected))?;
                self.pos += 1;
                if m == 0xFF {
                    // 连续 FF 填充，继续。
                    continue;
                }
                if !(M_RST0..=M_RST7).contains(&m) {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "restart 点处期望 RSTn（FFD0..FFD7），实得 FF{m:02X}\
                             （序号期望 RST{expected}）"
                        ),
                        "T.81 B.2.4.1：restart interval 边界必须紧跟一个 RSTn；\
                         确认 DRI 的 Ri 与码流自洽、或熵数据未被截断",
                    ));
                }
                let got = m - M_RST0;
                if got != expected {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "restart marker 序号不连续：期望 RST{expected}（FFD{:X}），\
                             实得 RST{got}（FFD{:X}）",
                            M_RST0 + expected,
                            M_RST0 + got
                        ),
                        "T.81 B.2.1：RSTn 的 n 按 0..7 循环递增；序号不连续说明\
                         之前某个 restart interval 的样本数算错了，继续解会静默错位",
                    ));
                }
                return Ok(());
            }
            return Err(tiff_err(
                LOC,
                format!(
                    "restart 点处期望 marker 前缀 FF，实得 0x{b:02X}（序号期望 RST{expected}）"
                ),
                "T.81 B.2.4.1：restart interval 边界必须紧跟一个 RSTn；\
                 确认 DRI 的 Ri 与码流自洽",
            ));
        }
    }

    /// restart 点数据不足时的结构化错误。
    fn truncated_restart(&self, expected: u8) -> crate::raw::RawError {
        tiff_err(
            "raw_lossless_jpeg::BitReader::restart",
            format!(
                "读取 restart marker RST{expected} 时数据用尽（偏移 {}，总长 {}）",
                self.pos,
                self.data.len()
            ),
            "strip/tile 字节数不足或 DRI 与码流不自洽；确认 DNG 未被截断",
        )
    }
}

// ===========================================================================
//  五、canonical Huffman —— T.81 B.2.4.2 / Table B.5 + Annex C
// ===========================================================================

/// canonical Huffman 解码表（T.81 B.2.4.2 `DHT` + Annex C 构造）。
///
/// 与 `exr*.rs` 里那份 PIZ 用的表**结构相似但数据完全独立**：本表来自
/// JPEG 的 `DHT` 段（`counts[16]` + `symbols[]`），语义、码长上限（16）、
/// 校验规则都不同，**不与 EXR 耦合**。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffTable {
    /// 长度 `l` 是否有码字（不能用 `max_code[l] == 0` 当「无」的哨兵：
    /// 「只有一个码字且其值为 0」是合法表，会与哨兵撞车）。
    has: [bool; 17],
    /// 各码长的最小码值（索引 1..=16，索引 0 不用）。
    min_code: [u32; 17],
    /// 各码长的最大码值。
    max_code: [u32; 17],
    /// 各码长在 `symbols` 中的起始下标。
    val_ptr: [usize; 17],
    /// 码字到符号的映射（`DHT` 里的 `Th` 顺序）。
    symbols: Vec<u8>,
}

impl HuffTable {
    /// 由 `DHT` 段的 `counts[1..=16]` 与 `symbols[]` 构造。
    ///
    /// # 校验（T.81 B.2.4.2 + Annex C）
    ///
    /// - 符号总数 `= sum(L1..L16)` 且 `≤ 256`（`u8` 空间的自然上界）；
    /// - **码长溢出（过订阅）**：canonical 码分配到长度 `l` 时，
    ///   `code + L_l` 不得大于 `2^l`；
    /// - **符号越界**：无损 DC 表的符号就是 `SSSS`，T.81 Table H.2 规定
    ///   `SSSS ∈ 0..=16`，故**任何 > 16 的符号结构化拒绝**。
    pub fn build(counts: &[u8; 16], symbols: Vec<u8>) -> RawResult<Self> {
        const LOC: &str = "raw_lossless_jpeg::HuffTable::build";
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total == 0 {
            return Err(tiff_err(
                LOC,
                "DHT 表的 L1..L16 全为 0（无任何码字）",
                "确认 DHT 段未被截断或错位；空 Huffman 表无法解码",
            ));
        }
        if total > 256 {
            return Err(tiff_err(
                LOC,
                format!("DHT 表声明 {total} 个码字，超过 256 上界"),
                "确认 DHT 段长度与 counts 一致",
            ));
        }
        if symbols.len() != total {
            return Err(tiff_err(
                LOC,
                format!(
                    "DHT 表 counts 声明 {total} 个码字，但只读到 {} 个符号",
                    symbols.len()
                ),
                "确认 DHT 段长度字段正确",
            ));
        }
        for (s, &sym) in symbols.iter().enumerate() {
            if sym > MAX_SSSS {
                return Err(tiff_err(
                    LOC,
                    format!("DHT 表第 {s} 个符号 = {sym}，超出 T.81 Table H.2 的 SSSS 上限 16"),
                    "无损 DC 表的符号即 SSSS，只能是 0..=16；确认这是无损 SOF3 的 DC 表",
                ));
            }
        }

        // Annex C：由 counts 派生 canonical 码的 min/max/val_ptr。
        let mut has = [false; 17];
        let mut min_code = [0u32; 17];
        let mut max_code = [0u32; 17];
        let mut val_ptr = [0usize; 17];
        let mut code: u32 = 0;
        let mut k: usize = 0;
        for l in 1..=16usize {
            let n = counts[l - 1] as u32;
            if n > 0 {
                // 码长溢出（过订阅）检测：长度 l 的码字必须落在 [0, 2^l)。
                if code + n > (1u32 << l) {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "DHT 表码长溢出（过订阅）：长度 {l} 处需 {n} 个码字，\
                             但该长度只有 {} 个可用码字",
                            1u32 << l
                        ),
                        "确认 DHT 的 L1..L16 合法（Kraft 和 ≤ 1）",
                    ));
                }
                has[l] = true;
                min_code[l] = code;
                max_code[l] = code + n - 1;
                val_ptr[l] = k;
                code += n;
                k += n as usize;
            }
            code <<= 1;
        }

        Ok(Self {
            has,
            min_code,
            max_code,
            val_ptr,
            symbols,
        })
    }

    /// 解一个码字（T.81 B.2.4.2 / Annex C 的 `HUFFDECODE`）。
    ///
    /// 逐比特累积 `code`，与 `max_code[l]` 比较；命中后由
    /// `symbols[val_ptr[l] + (code - min_code[l])]` 取符号。
    fn decode(&self, br: &mut BitReader<'_>) -> RawResult<u8> {
        const LOC: &str = "raw_lossless_jpeg::HuffTable::decode";
        let mut code: u32 = 0;
        for l in 1..=16usize {
            code = (code << 1) | br.bit()?;
            if self.has[l] && code >= self.min_code[l] && code <= self.max_code[l] {
                let idx = self.val_ptr[l] + (code - self.min_code[l]) as usize;
                return self.symbols.get(idx).copied().ok_or_else(|| {
                    tiff_err(
                        LOC,
                        format!("码字 {code}（长 {l}）映射到越界符号下标 {idx}"),
                        "这是内部索引错误，请提交 issue",
                    )
                });
            }
        }
        Err(tiff_err(
            LOC,
            "16 位内未命中任何码字（比特流与 DHT 不匹配）",
            "确认 DHT 与熵编码段属于同一条无损 JPEG 流",
        ))
    }

    /// 本表的码字数（`sum(L1..L16)`）。
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }
}

// ===========================================================================
//  六、段解析 —— T.81 B.2
// ===========================================================================

/// 段解析游标（`SOI` 之后的主循环）。
struct SegReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SegReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    #[inline]
    fn u8(&mut self) -> RawResult<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| self.truncated("1 字节标记/字段"))?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn u16(&mut self) -> RawResult<u16> {
        let hi = self.u8()? as u16;
        let lo = self.u8()? as u16;
        Ok((hi << 8) | lo)
    }

    fn truncated(&self, what: &str) -> crate::raw::RawError {
        tiff_err(
            "raw_lossless_jpeg::SegReader",
            format!(
                "在偏移 {} 读取 {what} 时数据用尽（总长 {}）",
                self.pos,
                self.data.len()
            ),
            "strip/tile 字节数不足或流被截断；确认 DNG 文件完整",
        )
    }

    /// 读一个 marker（T.81 B.2.1 的 `FF xx`），跳过 `FF` 填充字节。
    ///
    /// `FF 00` 在熵编码段**之外**是非法的（那里不允许字节填充），
    /// 结构化拒绝。
    fn marker(&mut self) -> RawResult<u8> {
        const LOC: &str = "raw_lossless_jpeg::SegReader::marker";
        let start = self.pos;
        let b = self.u8()?;
        if b != 0xFF {
            return Err(tiff_err(
                LOC,
                format!("偏移 {start} 处期望 marker 前缀 0xFF，实得 0x{b:02X}"),
                "确认字节流是 JPEG（应以 FF D8 开头）",
            ));
        }
        loop {
            let m = self.u8()?;
            match m {
                // 0xFF 填充：连续的 FF 合法（T.81 B.2.1 允许 fill byte）。
                0xFF => continue,
                // 0x00：熵编码段外不允许字节填充。
                0x00 => {
                    return Err(tiff_err(
                        LOC,
                        format!("偏移 {start} 处出现 FF 00（字节填充）——段外不允许"),
                        "确认段长度字段正确、未把熵编码段当成普通段解析",
                    ))
                }
                _ => return Ok(m),
            }
        }
    }

    /// 读一个带长度字段的段头，返回**段体长度**（= `Lc - 2`）。
    fn segment_body_len(&mut self, marker_name: &str) -> RawResult<usize> {
        const LOC: &str = "raw_lossless_jpeg::SegReader::segment_body_len";
        let lc = self.u16()?;
        if lc < 2 {
            return Err(tiff_err(
                LOC,
                format!("{marker_name} 段长度 Lc = {lc} < 2（最小值）"),
                "确认段长度字段正确",
            ));
        }
        Ok((lc - 2) as usize)
    }

    /// 跳过 `n` 字节段体。
    fn skip(&mut self, n: usize) -> RawResult<()> {
        self.pos = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.truncated("段体（偏移溢出）"))?;
        if self.pos > self.data.len() {
            return Err(self.truncated("段体"));
        }
        Ok(())
    }
}

/// `SOF3` 段解析结果（T.81 B.2.2）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Sof3 {
    /// `P`：输入精度（bit/sample）。T.81 B.2.2 规定 `2 ≤ P ≤ 16`。
    precision: u8,
    /// `Y`：行数。
    height: u32,
    /// `X`：列数。
    width: u32,
    /// 每个分量的 `C_i`（组件标识符，T.81 B.2.2 的 `C1..Cn`）。
    ///
    /// T.81 B.2.3 规定 `SOS` 的 `Csj` 必须与 `SOF` 的 `Ci` **相匹配**
    /// （「each `Csj` shall be the same as the component identifier `Ci` used
    /// in the frame header」），故必须保留下来做交叉校验。
    comp_ids: Vec<u8>,
}

/// `SOS` 段解析结果（T.81 B.2.3 + Table B.3 的 lossless 列）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Sos {
    /// `Ns`：扫描内的分量数。
    ns: u8,
    /// 扫描分量列表，每项的 `(Csj, Tdj)`（`Tdj` = DC 表选择值 0..=3）。
    comps: Vec<(u8, u8)>,
    /// `Ss`：**预测器选择值**（1..=7；Table H.1）。
    ss: u8,
    /// `Al`：点变换参数 `Pt`（0..=15）。
    al: u8,
}

// ===========================================================================
//  七、解码结果
// ===========================================================================

/// 一条无损 JPEG（SOF3）流的解码结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LosslessImage {
    /// `P`：输入精度（2..=16）。
    pub precision: u8,
    /// `X`：`SOF3` 声明的列数（**权威**，可能与 strip 标称宽不同）。
    ///
    /// **注意**：多分量流下这是 **JPEG 内部的每分量列数**（DNG 的 CFA 重打包
    /// 常令 `X = cfa_width / Nf`），不是 CFA 栅格宽度。
    pub width: u32,
    /// `Y`：`SOF3` 声明的行数。
    pub height: u32,
    /// `Nf`：分量数（1..=4）。
    pub components: u8,
    /// `Ss`：预测器选择值（1..=7）。
    pub predictor: u8,
    /// `Al`：点变换参数 `Pt`（0..=15）。
    pub point_transform: u8,
    /// 重建后的样本，行主序，长度 `= width * height`。
    ///
    /// 每个元素是 `modulo 2^16` 的重建结果（T.81 H.1.2.1：
    /// 「The difference between the prediction value and the input is
    /// calculated modulo 2^16」）。`point_transform != 0` 时已按
    /// T.81 H.2.2 乘过 `2^Pt`。
    pub samples: Vec<u16>,
}

// ===========================================================================
//  八、顶层解码
// ===========================================================================

/// 解一条 DNG 内的无损 JPEG（SOF3）字节流。
///
/// 支持的 marker：`SOI` / `SOF3` / `DHT` / `DRI` / `SOS` / `EOI`，以及可跳过的
/// `COM` / `APPn`。其余 marker 一律**结构化拒绝**
/// （含所有 `SOF0/1/2/4..7/9..15` 与 `DQT`/`DAC` —— 它们在 SOF3 里无意义）。
///
/// # 多分量（`Nf` = 1..=4）
///
/// 支持 1..=4 个分量（每分量采样因子须 `1×1`、`Tq = 0`）。分量的**独立预测**
/// 与**行内交错取码**见 [`decode_scan`]；CFA 重组的映射依据见模块头文档。
///
/// # restart interval（`DRI` ≠ 0）
///
/// `Ri != 0` 时，每解出 `Ri` 个 MCU（本例 `H_i = V_i = 1` ⇒ 一个 MCU =
/// 一行内 `Ns` 个样本）就按 T.81 B.2.4.1 / H.1.2.1 重置**所有**分量的预测器
/// 状态、丢弃位缓冲并对齐到下一字节边界，然后消费一个 `RSTn` marker。
///
/// # 拒绝清单（T.81 依据）
///
/// - `SOF3` 的 `P` 不在 2..=16；
/// - `SOF3` 的 `X` 或 `Y` 为 0；
/// - `SOF3` 的 `Nf` 不在 1..=4；
/// - `SOF3` 的采样因子非 `1×1`、或 `Tq != 0`（SOF3 无量化表）；
/// - `DHT` 码长溢出（过订阅）/ 符号 > 16 / 码字数与 `counts` 不符；
/// - `SOS` 的 `Ns` 不在 1..=4 或 `Ns != Nf`、组件 id 与 `SOF3` 不符；
/// - `SOS` 的 `Se != 0`、`Ah != 0`、`Ta != 0`（T.81 Table B.3 lossless 列
///   均规定「shall be set to zero」）；
/// - `SOS` 的 `Ss` 不在 1..=7（含 `Ss == 0`，见 [`Predictor::from_selection`]）；
/// - `SOS` 的 `Al >= P`（点变换后 `2^(P−Pt−1)` 无定义）；
/// - `SOS` 的 `Td` 指向不存在的 DC 表；
/// - `DRI` 后的 `RSTn` 序列号不连续 / 熵编码段在重启点前结束；
/// - 熵编码段在重建完全部样本前结束（截断 / 坏码字）。
pub fn decode_lossless_jpeg(bytes: &[u8]) -> RawResult<LosslessImage> {
    const LOC: &str = "raw_lossless_jpeg::decode_lossless_jpeg";
    let mut r = SegReader::new(bytes, 0);

    // ---- SOI（T.81 B.2.1）----
    let m = r.marker()?;
    if m != M_SOI {
        return Err(tiff_err(
            LOC,
            format!("流首 marker = FF{m:02X}，期望 SOI（FFD8）"),
            "确认这是 JPEG 字节流（DNG 的 strip/tile 载荷以 FFD8 开头）",
        ));
    }

    let mut sof: Option<Sof3> = None;
    let mut tables: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut sos: Option<Sos> = None;
    // `DRI`：restart interval（缺省 0 = 无 restart）。T.81 B.2.4.1。
    let mut restart_interval: u32 = 0;

    loop {
        let m = r.marker()?;
        match m {
            M_SOF3 => {
                if sof.is_some() {
                    return Err(tiff_err(
                        LOC,
                        "出现第二个 SOF3 段",
                        "一条无损流只能有一个 SOF；确认字节流未拼接错",
                    ));
                }
                sof = Some(read_sof3(&mut r)?);
            }
            M_DHT => read_dht(&mut r, &mut tables)?,
            M_DRI => {
                // T.81 B.2.4.1：段长 `Lc` 必为 4（2 字节长度 + 2 字节 `Ri`），
                // 先消费长度字段，再读 `Ri`（两个 restart marker 之间的 MCU 数）。
                // `Ri = 0` 表示**不定义** restart interval（等价于无）。二者都合法。
                let body = r.segment_body_len("DRI")?;
                if body != 2 {
                    return Err(tiff_err(
                        LOC,
                        format!("DRI 段体长 {body} != 2（T.81 B.2.4.1 规定 Lc = 4）"),
                        "确认 DRI 段长度字段正确",
                    ));
                }
                restart_interval = r.u16()? as u32;
            }
            M_COM => {
                let n = r.segment_body_len("COM")?;
                r.skip(n)?;
            }
            // APP0..APP15：有长度字段，跳过。
            _ if (M_APP0..=0xEF).contains(&m) => {
                let n = r.segment_body_len("APPn")?;
                r.skip(n)?;
            }
            M_SOS => {
                if sos.is_some() {
                    return Err(tiff_err(
                        LOC,
                        "出现第二个 SOS 段",
                        "一条无损流只能有一个 SOS；确认字节流未拼接错",
                    ));
                }
                let sof_v = sof.clone().ok_or_else(|| {
                    tiff_err(
                        LOC,
                        "SOS 出现在 SOF3 之前",
                        "T.81 B.2 的段序要求 SOF 先于 SOS；确认字节流完整",
                    )
                })?;
                sos = Some(read_sos(&mut r, &sof_v)?);
                break;
            }
            // 熵编码段之外出现 `RSTn` 属畸形（T.81 B.2.1：RSTn 只在熵编码段内）。
            _ if (M_RST0..=M_RST7).contains(&m) => {
                return Err(tiff_err(
                    LOC,
                    format!("SOS 之前出现 restart marker FFD{m:02X}"),
                    "T.81 B.2.1：RSTn 只能出现在熵编码段内部；\
                     确认 DRI/SOS 段序或长度字段正确",
                ))
            }
            M_EOI => {
                return Err(tiff_err(
                    LOC,
                    "遇到 EOI 但此前没有 SOS（空扫描）",
                    "确认 strip/tile 载荷是完整的无损 JPEG 流",
                ))
            }
            other => {
                return Err(unsupported_err(
                    LOC,
                    format!(
                        "marker FF{other:02X} 不被支持（SOF3 只接受 \
                         SOI/SOF3/DHT/DRI/SOS/EOI/COM/APPn）"
                    ),
                    "DNG Compression=7 必须是 lossless JPEG（SOF3）；\
                     若是算术编码（SOF9..15）或有损 SOF0/1/2，本模块不解码",
                ))
            }
        }
    }

    let sof =
        sof.ok_or_else(|| tiff_err(LOC, "流中没有 SOF3 段", "确认这是 lossless JPEG（SOF3）流"))?;
    let sos = sos.ok_or_else(|| tiff_err(LOC, "流中没有 SOS 段", "确认 strip/tile 载荷完整"))?;

    // ---- 每个扫描分量各查一张 DC Huffman 表（T.81 B.2.3：每个 Csj 带自己的 Tdj）----
    let mut scan_tables: Vec<&HuffTable> = Vec::with_capacity(sos.comps.len());
    for (cs, td) in sos.comps.iter() {
        let t = tables
            .get(*td as usize)
            .and_then(|t: &Option<HuffTable>| t.as_ref())
            .ok_or_else(|| {
                tiff_err(
                    LOC,
                    format!("SOS 分量 Cs = {cs} 的 Td = {td} 指向未定义的 DC Huffman 表"),
                    "确认 DHT 段存在且其 Th 覆盖所有扫描分量使用的 DC 表号",
                )
            })?;
        scan_tables.push(t);
    }

    let total = (sof.width as usize)
        .checked_mul(sof.height as usize)
        .ok_or_else(|| {
            tiff_err(
                LOC,
                format!("SOF3 的 {}×{} 样本数溢出 usize", sof.width, sof.height),
                "文件损坏；在 64 位平台重试",
            )
        })?;
    if total == 0 {
        return Err(tiff_err(
            LOC,
            format!("SOF3 声明 {}×{} = 0 个样本", sof.width, sof.height),
            "确认图像尺寸非零",
        ));
    }
    if total > MAX_SAMPLES {
        return Err(unsupported_err(
            LOC,
            format!("SOF3 声明 {total} 个样本，超过本模块上限 {MAX_SAMPLES}"),
            "图像过大；请分区处理，或确认 SOF3 的 X/Y 未被错位读取",
        ));
    }

    let mut samples = decode_scan(
        &scan_tables,
        sos.ss,
        sos.al,
        &sof,
        bytes,
        r.pos,
        total,
        restart_interval,
    )?;
    // T.81 H.2.2：if the point transform parameter is non-zero, multiply the
    // output of the lossless decoder by 2^Pt.
    if sos.al > 0 {
        for v in samples.iter_mut() {
            *v = ((*v as u32) << sos.al) as u16;
        }
    }

    Ok(LosslessImage {
        precision: sof.precision,
        width: sof.width,
        height: sof.height,
        components: sos.ns,
        predictor: sos.ss,
        point_transform: sos.al,
        samples,
    })
}

/// `SOF3` 允许的最大分量数（T.81 B.2.2 规定 `Nf ∈ 1..=255`；DNG 的 CFA
/// 重打包实际只用到 1..=4，见模块头文档的依据 1/2。更大的值本模块结构化拒绝）。
const MAX_COMPONENTS: u8 = 4;

/// 解析 `SOF3`（T.81 B.2.2）。
///
/// `Nf` 允许 **1..=4**（DNG 的 CFA 重打包常用 2）。每个分量的采样因子必须
/// `1×1`（T.81 B.2.2 的 `H_i`/`V_i`；DNG 的无损 JPEG 不用子采样），`Tq`
/// 必须为 0（SOF3 无量化表）。
fn read_sof3(r: &mut SegReader<'_>) -> RawResult<Sof3> {
    const LOC: &str = "raw_lossless_jpeg::read_sof3";
    let body = r.segment_body_len("SOF3")?;
    if body < 6 {
        return Err(tiff_err(
            LOC,
            format!("SOF3 段体只有 {body} 字节，至少需要 6"),
            "确认段长度正确",
        ));
    }
    let precision = r.u8()?;
    let height = r.u16()? as u32;
    let width = r.u16()? as u32;
    let nf = r.u8()?;

    if !(2..=16).contains(&precision) {
        return Err(unsupported_err(
            LOC,
            format!("SOF3 的 P = {precision}，T.81 B.2.2 规定 2 ≤ P ≤ 16"),
            "确认这是 8/12/14/16 位的无损 JPEG；arithmetic-coded 或 1 位流不在支持范围",
        ));
    }
    if width == 0 || height == 0 {
        return Err(tiff_err(
            LOC,
            format!("SOF3 尺寸非法：X = {width}、Y = {height}"),
            "确认图像尺寸非零",
        ));
    }
    if nf == 0 || nf > MAX_COMPONENTS {
        return Err(unsupported_err(
            LOC,
            format!("SOF3 的 Nf = {nf}，本模块只支持 1..={MAX_COMPONENTS}"),
            "DNG 的 CFA 无损 JPEG 通常为 1（灰度预测）或 2（左右像素对，\
             让预测器跨同色工作）；Nf > 4 的流请确认不是被错位读取的 SOF",
        ));
    }
    let expected_body = 6usize + 3 * nf as usize;
    if body != expected_body {
        return Err(tiff_err(
            LOC,
            format!("SOF3 段体长 {body} != 预期的 {expected_body}（6 + 3×Nf）"),
            "确认段长度 Lc 与 Nf 一致",
        ));
    }
    let mut comp_ids: Vec<u8> = Vec::with_capacity(nf as usize);
    for i in 0..nf {
        let comp_id = r.u8()?;
        let hv = r.u8()?;
        let tq = r.u8()?;
        if hv != 0x11 {
            return Err(unsupported_err(
                LOC,
                format!("SOF3 第 {i} 个分量（C = {comp_id}）采样因子 = 0x{hv:02X}，不是 1×1"),
                "DNG 的 CFA 无损 JPEG 各分量都是 1×1 采样（无子采样）；\
                 出现 2×1 等子采样说明这不是 DNG CFA 无损流",
            ));
        }
        if tq != 0 {
            return Err(unsupported_err(
                LOC,
                format!("SOF3 第 {i} 个分量（C = {comp_id}）的 Tq = {tq} != 0"),
                "无损 SOF3 **没有量化表**（T.81 Table B.3 lossless 列：Tq no meaning）；\
                 Tq != 0 说明这不是 SOF3",
            ));
        }
        comp_ids.push(comp_id);
    }
    Ok(Sof3 {
        precision,
        height,
        width,
        comp_ids,
    })
}

/// 解析 `DHT`（T.81 B.2.4.2 + Table B.5）。
fn read_dht(r: &mut SegReader<'_>, tables: &mut [Option<HuffTable>; 4]) -> RawResult<()> {
    const LOC: &str = "raw_lossless_jpeg::read_dht";
    let body = r.segment_body_len("DHT")?;
    let end = r
        .pos
        .checked_add(body)
        .ok_or_else(|| r.truncated("DHT 段体（偏移溢出）"))?;
    if end > r.data.len() {
        return Err(r.truncated("DHT 段体"));
    }

    while r.pos < end {
        let tc_th = r.u8()?;
        let tc = tc_th >> 4;
        let th = tc_th & 0x0F;
        if th >= 4 {
            return Err(tiff_err(
                LOC,
                format!("DHT 的 Th = {th}，超出 Table B.5 的 0..=3"),
                "确认 DHT 段未被错位解析",
            ));
        }
        let mut counts = [0u8; 16];
        for c in counts.iter_mut() {
            *c = r.u8()?;
        }
        // 读 16 个计数字节可能已越过段末（畸形长度）——必须先夹住，
        // 否则下面的 `end - r.pos` 会**下溢 panic**（违反 C18）。
        // 这个分支正是 `malformed_all_byte_mutations_never_panic` 抓到的。
        if r.pos > end {
            return Err(tiff_err(
                LOC,
                format!("DHT 表的 L1..L16 越过段末（偏移 {} > 段末 {end}）", r.pos),
                "确认 DHT 段长度 Lc 正确（至少 2 + 1 + 16 = 19 字节）",
            ));
        }
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total > 256 {
            return Err(tiff_err(
                LOC,
                format!("DHT 表声明 {total} 个码字，超过 256 上界"),
                "确认 DHT 段长度与 counts 一致",
            ));
        }
        // 段体剩余长度不足以容纳 total 个符号 → 结构化拒绝，不越界读。
        let remaining = end - r.pos;
        if total > remaining {
            return Err(tiff_err(
                LOC,
                format!("DHT 表声明 {total} 个符号，但段体只剩 {remaining} 字节"),
                "确认 DHT 段长度 Lc 正确",
            ));
        }
        let mut symbols = Vec::with_capacity(total);
        for _ in 0..total {
            symbols.push(r.u8()?);
        }
        if tc == 0 {
            // Tc = 0：DC 表 —— 无损模式下承载 SSSS 符号。
            let t = HuffTable::build(&counts, symbols)?;
            if let Some(slot) = tables.get_mut(th as usize) {
                *slot = Some(t);
            }
        }
        // Tc = 1（AC 表）在 SOF3 无损模式里无意义（T.81 Table B.3 lossless 列：
        // Ta no meaning），**结构化拒绝**而不是静默忽略 —— 静默忽略会让
        // 「本来是有损/算术流」的文件被当成无损流读出垃圾。
        else {
            return Err(unsupported_err(
                LOC,
                format!("DHT 含 AC 表（Tc = {tc}）"),
                "SOF3 无损流只使用 DC 表（Tc = 0）；出现 AC 表说明段序/流类型不符",
            ));
        }
    }
    if r.pos != end {
        return Err(tiff_err(
            LOC,
            format!("DHT 段体解析后偏移 {} != 段末 {end}", r.pos),
            "确认 DHT 段长度正确",
        ));
    }
    Ok(())
}

/// 解析 `SOS`（T.81 B.2.3 + **Table B.3 的 lossless 列**）。
///
/// Table B.3 lossless 列的逐条原文：
///
/// > Ss: … **In the lossless mode of operations this parameter is used to
/// > select the predictor.**
/// > Se: … In the lossless mode of operations this parameter has no meaning.
/// > **It shall be set to zero.**
/// > Ah: … In the lossless mode of operations this parameter has no meaning.
/// > **It shall be set to zero.**
/// > Al: … In the lossless mode of operations, this parameter specifies the
/// > point transform, `Pt`.
///
/// # `Ns` 与 `Nf`（T.81 B.2.3）
///
/// 无损模式为**非交错顺序扫描**，每个分量各有一张 DC 表：
///
/// > The scan header specifies the number of image components (`Ns`) in the
/// > scan … Each `Csj` shall be the same as the component identifier `Ci`
/// > used in the frame header.
///
/// 本模块要求 `1 <= Ns <= 4` 且 **`Ns == Nf`**（即扫描覆盖帧的**全部**分量，
/// 与 DNG CFA 重打包的「交错扫描」一致；T.81 B.2.3 表 B.3 的 `Csj` 逐项校验）。
fn read_sos(r: &mut SegReader<'_>, sof: &Sof3) -> RawResult<Sos> {
    const LOC: &str = "raw_lossless_jpeg::read_sos";
    let body = r.segment_body_len("SOS")?;
    if body < 3 {
        return Err(tiff_err(
            LOC,
            format!("SOS 段体只有 {body} 字节，至少需要 3"),
            "确认段长度正确",
        ));
    }
    let ns = r.u8()?;
    if ns == 0 || ns > MAX_COMPONENTS {
        return Err(unsupported_err(
            LOC,
            format!("SOS 的 Ns = {ns}，本模块只支持 1..={MAX_COMPONENTS}"),
            "无损扫描的 Ns 应在 1..=4；确认 SOS 段未被错位解析",
        ));
    }
    let nf = sof.comp_ids.len() as u8;
    if ns != nf {
        return Err(unsupported_err(
            LOC,
            format!("SOS 的 Ns = {ns} != SOF3 的 Nf = {nf}"),
            "本模块只支持覆盖**全部**帧分量的交错扫描（T.81 B.2.3 的 \
             non-interleaved 单分量扫描需另建多扫描管线）；\
             若这是多扫描流，请改存为交错扫描",
        ));
    }
    let expected_body = 1usize + 2 * ns as usize + 3;
    if body != expected_body {
        return Err(tiff_err(
            LOC,
            format!("SOS 段体长 {body} != 预期的 {expected_body}（1 + 2×Ns + 3）"),
            "确认段长度 Lc 与 Ns 一致",
        ));
    }
    let mut comps: Vec<(u8, u8)> = Vec::with_capacity(ns as usize);
    for j in 0..ns {
        let cs = r.u8()?;
        let td_ta = r.u8()?;
        let td = td_ta >> 4;
        let ta = td_ta & 0x0F;
        if td >= 4 {
            return Err(tiff_err(
                LOC,
                format!("SOS 第 {j} 个分量（Cs = {cs}）的 Td = {td}，超出 Table B.5 的 0..=3"),
                "确认 Td 指向存在的 DC 表",
            ));
        }
        if ta != 0 {
            return Err(tiff_err(
                LOC,
                format!("SOS 第 {j} 个分量（Cs = {cs}）的 Ta = {ta} != 0"),
                "T.81 Table B.3 lossless 列：Ta 在无损模式下无意义，**shall be set to zero**",
            ));
        }
        // T.81 B.2.3：`Csj` 必须与帧头的分量标识符一致。
        if !sof.comp_ids.contains(&cs) {
            return Err(tiff_err(
                LOC,
                format!(
                    "SOS 第 {j} 个分量标识符 Cs = {cs} 不在 SOF3 的分量标识符 {:?} 中",
                    sof.comp_ids
                ),
                "T.81 B.2.3 要求 `Csj` 与帧头的 `Ci` 一致；确认 SOS/SOF3 未被错位解析",
            ));
        }
        comps.push((cs, td));
    }
    let ss = r.u8()?;
    let se = r.u8()?;
    let ah_al = r.u8()?;
    let ah = ah_al >> 4;
    let al = ah_al & 0x0F;

    if se != 0 {
        return Err(tiff_err(
            LOC,
            format!("SOS 的 Se = {se} != 0"),
            "T.81 Table B.3 lossless 列：Se 在无损模式下无意义，**shall be set to zero**",
        ));
    }
    if ah != 0 {
        return Err(tiff_err(
            LOC,
            format!("SOS 的 Ah = {ah} != 0"),
            "T.81 Table B.3 lossless 列：Ah 在无损模式下无意义，**shall be set to zero**",
        ));
    }
    // 预测器：Ss 越界检查（0 与 > 7 都拒）。
    let _ = Predictor::from_selection(ss)?;
    if al >= sof.precision {
        return Err(unsupported_err(
            LOC,
            format!("SOS 的 Al（点变换 Pt）= {al} >= P = {}", sof.precision),
            "T.81 H.1.2.1：点变换非零时首样本预测值为 2^(P−Pt−1)，需 P−Pt−1 ≥ 0；\
             DNG 实际总为 Pt = 0",
        ));
    }
    Ok(Sos { ns, comps, ss, al })
}

// ===========================================================================
//  九、熵解码与预测重建 —— T.81 H.1.2.1 / H.1.2.2
// ===========================================================================

/// 熵解码 + 预测重建（T.81 H.1.2.1「Prediction」+ H.1.2.2）。
///
/// # 输出布局（**交错样本栅格**）
///
/// 返回的 `out` 长度 `= Nf * width * height`，行主序，**每个样本位置按
/// 分量顺序 0,1,…,Nf-1 连续存放**：`out[(y * width + x) * Nf + c]` =
/// 第 `c` 个分量在 `(x, y)` 的重建值。多分量流重组成单通道 CFA 栅格由
/// [`deinterleave_components`] 按模块头文档的三条依据完成。
///
/// # 多分量：每分量独立预测（关键）
///
/// T.81 H.1.2.1 的 `Ra` / `Rb` / `Rc` 是「**当前分量**的左 / 上 / 左上
/// 重建样本」，故每个分量各有**独立的** `prev` / `cur` 行缓冲与预测器状态；
/// **绝不共享**（否则分量 0 的邻域会污染分量 1 的预测值）。行内取码顺序是
/// **固定分量序 `0,1,…,Nf-1` 轮转**（T.81 B.2.3 的交错 MCU 定义；本例各分量
/// `H_i = V_i = 1` ⇒ 一个 MCU = 一行内 `Nf` 个样本）。
///
/// # 行首预测值的状态机（T.81 H.1.2.1 逐条）
///
/// > The one-dimensional horizontal predictor (prediction sample Ra) is used
/// > for the first line of samples at the start of the scan and at the
/// > beginning of each restart interval. The selected predictor is used for all
/// > other lines. The sample from the line above (prediction sample Rb) is used
/// > at the start of each line, except for the first line.
/// >
/// > **At the beginning of the first line and at the beginning of each restart
/// > interval the prediction value of 2^(P−1) is used**, where P is the input
/// > precision.
///
/// 这四句的**唯一自洽读法**是把它们看成对图像的**无重叠划分**：
///
/// | 位置 | 预测值 | 依据 |
/// |---|---|---|
/// | `(x=0, y=0)`（首行行首） | `2^(P−1)` | 第 4 句（明文覆盖第 1 句） |
/// | `(x=0, y>0)`（其余行行首） | `Rb`（上一行同列） | 第 3 句 |
/// | `(x>0, y=0)`（首行非行首） | `Ra`（水平预测） | 第 1 句（"for the first line of samples"） |
/// | `(x>0, y>0)` | **扫描头选定的预测器** | 第 2 句（"for all other lines"） |
///
/// 第 1 句的 "The one-dimensional horizontal predictor … is used **for the
/// first line of samples**" 管辖**整条首行**（`x > 0` 部分），第 2 句
/// "The selected predictor is used **for all other lines**" 管辖 `y >= 1`
/// （`x == 0` 的位置被第 3 句单独规定）。这样四句恰好拼满全图，**不留空洞**。
///
/// 这四个分支对**每个分量各自成立**：`y` 与 `x` 用的都是该分量内的行/列坐标
/// （各分量 `H_i = V_i = 1`，尺寸相同）。restart interval 开始（含每个 interval
/// 的第一行行首）同样回到 `2^(P−1)`，即「行首」分支被 restart 边界再次触发
/// —— 见 [`decode_scan`] 的 `is_first_line_of_interval` 处理。
///
/// # restart interval（T.81 B.2.4.1）
///
/// `restart_interval != 0` 时，每解出 `Ri` 个 MCU（= `Ri` × 一行内 `Nf` 个
/// 样本，因各分量 `H_i = V_i = 1`）就丢弃填充位、消费一个 `RSTn`、并把
/// **所有**分量的预测器状态重置为「首行行首」。`RSTn` 的序号按 0..7 循环，
/// 由调用方计数并校验（[`BitReader::restart`]）。
///
/// # modulo 2^16（T.81 H.1.2.1 末句）
///
/// > The difference between the prediction value and the input is calculated
/// > modulo 2^16. In the decoder the difference is decoded and added,
/// > modulo 2^16, to the prediction.
///
/// 故重建是 `x = (Px + DIFF) mod 2^16`，**按 2^16 而非 2^P 取模**
/// （8 位时结果一样，但代码按规范字面写）。
#[allow(clippy::too_many_arguments)]
fn decode_scan(
    tables: &[&HuffTable],
    ss: u8,
    al: u8,
    sof: &Sof3,
    bytes: &[u8],
    start: usize,
    total: usize,
    restart_interval: u32,
) -> RawResult<Vec<u16>> {
    let predictor = Predictor::from_selection(ss)?;
    // 首行行首预测值 2^(P−1)；点变换非零时为 2^(P−Pt−1)（T.81 H.1.2.1）。
    let first_pred: i32 = 1i32 << (sof.precision - 1 - al);

    let width = sof.width as usize;
    let height = sof.height as usize;
    let nf = sof.comp_ids.len();
    // 每个分量独立的上一行 / 当前行缓冲（**绝不共享**，见函数文档）。
    let mut prev: Vec<Vec<u16>> = vec![vec![0u16; width]; nf];
    let mut cur: Vec<Vec<u16>> = vec![vec![0u16; width]; nf];

    let mut br = BitReader::new(bytes, start);
    let mut out: Vec<u16> = vec![0u16; total.saturating_mul(nf)];
    // 下一个期望的 `RSTn` 序号（T.81 B.2.1：按 0..7 **循环**递增）。
    let mut rst_index: u8 = 0;
    // 自上一个 restart 点以来，已经解出的**行数**。restart 点必然落在某个
    // 样本位置（本例 MCU = 一行内一个列位置），该位置所在行即「新 interval 的
    // 首行」——于是该行 `x > 0` 的部分走水平预测、`x == 0` 走 `2^(P−1)`
    // （T.81 H.1.2.1：「at the beginning of each restart interval the prediction
    // value of 2^(P−1) is used」+「horizontal predictor used for the first line
    // of samples at the start of the scan and at the beginning of each restart
    // interval」）。
    //
    // `rows_since_reset` 从 0 起：== 0 表示「本行是 interval 的首行」。
    // 注意 interval 边界可能**落在一行中间**，此时该行的前半属于上一个
    // interval、后半属于新 interval —— 但由于「新 interval 首行」的判定是
    // 逐样本的（见 `fresh`），这里按整行粒度推进 `rows_since_reset` 即可：
    // 边界之后的行 `rows_since_reset` 自然 ≥ 1（≠ 0），不再误判为首行。
    let mut rows_since_reset: usize = 0;

    // 伪码（T.81 Annex H 的 lossless 解码流程）：
    //   for each MCU (本例一个 MCU = 一行内 nf 个样本)
    //     if restart_interval != 0 and 已解 MCU 数 % Ri == 0 and 非首个 MCU:
    //        重新对齐字节、消费 RSTn、重置所有分量预测器
    //     for c in 0..nf:  按固定分量序轮转
    //       diff = HUFFDECODE(tables[c]) + EXTEND
    //       px   = 该分量在 (x, y) 处的行首/首行/一般预测
    //       out[(y*width+x)*nf + c] = (px + diff) mod 2^16
    // 逐行、逐列、逐分量展开（这里的「MCU」= 一行内的一个列位置的 nf 个样本）。
    for y in 0..height {
        // 本行是否出现过 restart 边界（出现则 `rows_since_reset` 已被置 0，
        // 行末不再自增，使真正的「新 interval 首行」保持在 0）。
        let mut row_had_reset = false;
        for x in 0..width {
            // ---- restart interval 检查（T.81 B.2.4.1）----
            // 本例一个 MCU = 一行内 nf 个样本 ⇒ 一行含 width 个 MCU，
            // 全局 MCU 序号 = y * width + x。
            let interval_head = is_interval_head(y, x, width, restart_interval);
            if interval_head {
                // 到达 restart 边界：丢弃填充位 + 消费 RSTn（序号循环校验）。
                br.restart(rst_index)?;
                rst_index = (rst_index + 1) & 7;
                // 重置**所有**分量的预测器状态（回到「首行行首」）：
                // 行缓冲清零，等价于「上一行不存在 + 本行左邻不存在」。
                for c in 0..nf {
                    prev[c].fill(0);
                    cur[c].fill(0);
                }
                // 本行即新 interval 的首行（清零后从 `y == 0` 语义重新开始）。
                rows_since_reset = 0;
                row_had_reset = true;
            }

            // 「首行」语义：整图首行，或新 interval 的首行。T.81 H.1.2.1 的
            // 第 1 / 4 句把二者并列，故 `x > 0` 走水平预测、`x == 0` 走 2^(P−1)。
            let fresh = rows_since_reset == 0;

            for c in 0..nf {
                let table = tables.get(c).copied().unwrap_or_else(|| tables[0]);
                // ---- 差值解码（T.81 H.1.2.2 + Table H.2）----
                let ssss = table.decode(&mut br)?;
                let diff = extend(&mut br, ssss)?;

                // ---- 预测（T.81 H.1.2.1），每分量独立取自己的邻域 ----
                let px: i32 = if x == 0 {
                    if fresh {
                        // 首行行首 / restart interval 首样本：2^(P−1)。
                        first_pred
                    } else {
                        // 其余行行首：Rb = 该分量上一行同列。
                        prev[c][0] as i32
                    }
                } else if fresh {
                    // 首行非行首：一维水平预测 Px = Ra（该分量本行左邻）。
                    cur[c][x - 1] as i32
                } else {
                    // 其余：扫描头选定的预测器（T.81 Table H.1），用**该分量**的邻域。
                    predictor.predict_i32(
                        cur[c][x - 1] as i32,
                        prev[c][x] as i32,
                        prev[c][x - 1] as i32,
                    )
                };

                // ---- 重建：x = (Px + DIFF) mod 2^16 ----
                // `& 0xFFFF` 对 i32 是**补码低位**，等价于数学上的 mod 2^16
                // （包括 `Px + DIFF < 0` 的情形），与 T.81 的 "modulo 2^16" 一致。
                let v = ((px + diff) & 0xFFFF) as u16;
                if let Some(slot) = cur.get_mut(c).and_then(|r| r.get_mut(x)) {
                    *slot = v;
                }
                if let Some(slot) = out.get_mut((y * width + x) * nf + c) {
                    *slot = v;
                }
            }
        }
        // 行末把每个分量的 `cur` 滚成下一行的 `prev`（零拷贝交换）。
        for c in 0..nf {
            std::mem::swap(&mut prev[c], &mut cur[c]);
        }
        // 下一行的 `cur` 从 x=0 起被逐个覆写，无需清零。
        // 推进「自上次 reset 的行数」：本行含 restart 边界则保持 0，否则 +1。
        if !row_had_reset {
            rows_since_reset = rows_since_reset.saturating_add(1);
        }
    }
    Ok(out)
}

/// 判断 `(x, y)` 是否是某个 restart interval 的**首个样本位置**
/// （用于把该位置当作「首行行首」处理，走 `2^(P−1)` 分支）。
///
/// 本例一个 MCU = 一行内 `Nf` 个样本，故一行含 `width` 个 MCU；
/// 全局 MCU 序号 = `y * width + x`。interval 首位置即 `MCU 序号 % Ri == 0`
/// 且 `MCU 序号 != 0`。
#[inline]
fn is_interval_head(y: usize, x: usize, width: usize, restart_interval: u32) -> bool {
    if restart_interval == 0 {
        return false;
    }
    let mcu_index = (y as u32).saturating_mul(width as u32) + x as u32;
    mcu_index != 0 && mcu_index.is_multiple_of(restart_interval)
}

// ===========================================================================
//  十、DNG 行交织（RowInterleaveFactor，tag 50975 / 0xC71F）
// ===========================================================================

/// 计算 DNG `RowInterleaveFactor` 的**行置换表**。
///
/// DNG 1.6 规范 `RowInterleaveFactor`（Tag 50975 / `C71F.H`）原文：
///
/// > Type SHORT or LONG / Count 1 / Default **1** / Usage Raw IFD
/// > This tag specifies that rows of the image are stored in interleaved
/// > order. The value of the tag specifies the number of interleaved fields.
/// > The use of a non-default value for this tag requires setting the
/// > DNGBackwardVersion tag to at least 1.2.0.0.
///
/// 语义（本模块的实现）：`factor = N` 表示图像行被分成 `N` 个 field，
/// **field `f` 存放所有 `row % N == f` 的行**，**field 内行序保持**；
/// 码流中的排列是 `field 0` 的全部行、`field 1` 的全部行、……
///
/// - `factor = 1`（**默认**）→ 不交织，置换表是恒等映射；
/// - `factor = 2` → **偶数行在前、奇数行在后**（即「奇偶行分离」）。
///
/// # 参数
///
/// - `rows`：本 strip/tile 覆盖的**行数**；
/// - `y0`：本 strip/tile 首行在**整图**中的行号（多 strip 时行号不连续，
///   必须由调用方给出，才能正确判定每个码流行属于哪个 field）；
/// - `factor`：`RowInterleaveFactor`。
///
/// # 返回
///
/// 长度 `rows` 的置换表 `map`：`out_row[map[j]]` 应写入**码流第 `j` 行**。
///
/// # 例子（`rows = 4, y0 = 0, factor = 2`）
///
/// field 0 = 行 `{0, 2}`，field 1 = 行 `{1, 3}`；码流顺序 `0,2,1,3`
/// ⇒ `map = [0, 2, 1, 3]`，即 `out[0]=码流0`、`out[2]=码流1`、
/// `out[1]=码流2`、`out[3]=码流3`。
pub fn interleave_row_map(rows: u32, y0: u32, factor: u32) -> RawResult<Vec<u32>> {
    const LOC: &str = "raw_lossless_jpeg::interleave_row_map";
    if factor == 0 {
        return Err(tiff_err(
            LOC,
            "RowInterleaveFactor = 0",
            "DNG 1.6 规定该 tag 的 Default = 1；0 会让行归属无定义",
        ));
    }
    let rows = rows as usize;
    let f = factor as usize;
    let y0s = y0 as usize;

    // 每个 field 在本 strip 区间 [y0, y0+rows) 内的行数，以及码流里的起始偏移
    // （码流排列 = field 0 的全部行、field 1 的全部行、……）。
    let mut counts = vec![0usize; f];
    for j in 0..rows {
        let g = (y0s + j) % f;
        if let Some(c) = counts.get_mut(g) {
            *c += 1;
        }
    }
    let mut offsets = vec![0usize; f];
    let mut acc = 0usize;
    for g in 0..f {
        offsets[g] = acc;
        acc = acc.saturating_add(counts.get(g).copied().unwrap_or(0));
    }

    // 码流行 j → 它落在哪个 field 的区间 → 该 field 内的第 k 行
    // → 该 field 的第 k 个全局行号（field g 的行号 ≡ g (mod f)，且 ≥ y0）。
    let mut within = vec![0usize; f];
    let mut map: Vec<u32> = Vec::with_capacity(rows);
    for j in 0..rows {
        // field 区间是 [offsets[g], offsets[g]+counts[g])，按 g 升序首个命中者。
        let mut field = f.saturating_sub(1);
        for g in 0..f {
            let start = offsets.get(g).copied().unwrap_or(0);
            let n = counts.get(g).copied().unwrap_or(0);
            if j < start.saturating_add(n) {
                field = g;
                break;
            }
        }
        let start = offsets.get(field).copied().unwrap_or(0);
        let k = j - start;
        if let Some(c) = within.get_mut(field) {
            *c = k + 1;
        }
        // field `field` 的第 k 个全局行：从 y0 起第一个 ≡ field (mod f) 的行，
        // 再加 k*f。因为 y0 + ((field - y0%f + f) % f) 是首个候选。
        let lead = (field + f - (y0s % f)) % f;
        let global = y0s + lead + k * f;
        if global < y0s || (global - y0s) >= rows {
            return Err(tiff_err(
                LOC,
                format!(
                    "行交织映射越界：码流行 {j} → 全局行 {global}（区间 [{y0}, {}]）",
                    y0s + rows
                ),
                "确认 RowInterleaveFactor 与 strip 行范围自洽",
            ));
        }
        map.push(global as u32);
    }
    Ok(map)
}

// ===========================================================================
//  十一、DNG strip/tile 集成（SOF3 尺寸为准 + CFA 重组 + ExtraSamples 前缀 + 行交织）
// ===========================================================================

/// 把多分量的交错样本栅格**重组**成单通道 CFA 栅格（DNG 特有）。
///
/// # 映射式（**有规范/实现依据，非「我觉得」**）
///
/// ```text
/// cfa[y][x] = comp[x % nf][x / nf]          对每个 cfa 行 y
/// ```
///
/// 即：分量 `c`（`0..nf`）承载**列号 ≡ c (mod nf)** 的那些样本。三条**相互独立
/// 且彼此吻合**的依据（详见模块头文档）：
///
/// 1. **DNG 1.6 规范 `Compression = 7` 描述**：SOF3 内部的「宽/高/分量数」不必
///    与 strip 标称一致，**CFA 图常故意用不同分量数让预测器跨同色工作**；
/// 2. **Adobe DNG Converter / `dcraw` 的打包口径**：JPEG 宽 = `cfa_width / Nf`
///    （常见 `Nf = 2`），**两个分量是每个水平像素对的左/右样本** —— 即分量 0 是
///    偶列、分量 1 是奇列，正是本式的 `nf = 2` 特例；
/// 3. **ITU-T T.81 B.2.3 的非交错扫描**：各分量 `H_i = V_i = 1` 时，一行内按
///    分量序 `0,1,…,Nf-1` 轮转取码，分量 `c` 在行内的第 `k` 个样本位于 CFA 列
///    `k * Nf + c`。
///
/// # 维度契约
///
/// - 输入 `img` 是 [`decode_lossless_jpeg`] 的结果：`samples` 长度
///   `= Nf * width * height`，布局 `[(y * width + x) * Nf + c]`；
/// - 输出长度 `= width * height`（**CFA 栅格**）。
///
/// 若 `img.samples.len() != Nf * width * height` → 结构化拒绝（内部契约违约）。
pub fn deinterleave_components(img: &LosslessImage) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw_lossless_jpeg::deinterleave_components";
    let nf = img.components as usize;
    let width = img.width as usize;
    let height = img.height as usize;
    if nf == 0 {
        return Err(tiff_err(
            LOC,
            "分量数 Nf = 0，无法做 CFA 重组",
            "确认 SOF3 的 Nf 在 1..=4；这是内部契约违约，请提交 issue",
        ));
    }
    let expected = nf
        .checked_mul(width)
        .and_then(|v| v.checked_mul(height))
        .ok_or_else(|| {
            tiff_err(
                LOC,
                format!("{nf} × {width} × {height} 样本数溢出"),
                "SOF3 尺寸过大；确认 X/Y/Nf 未被错位读取",
            )
        })?;
    if img.samples.len() != expected {
        return Err(tiff_err(
            LOC,
            format!(
                "交错栅格长度 {} != Nf×X×Y = {nf}×{width}×{height} = {expected}",
                img.samples.len()
            ),
            "这是内部契约违约（decode_lossless_jpeg 的输出布局）；请提交 issue",
        ));
    }

    // CFA 栅格宽度 = Nf × SOF3.X（DNG 的 CFA 实际宽度）。
    // 映射：`cfa[y][x] = comp[x % nf][x / nf]`，
    // 即 `cfa[(y*cfa_w) + x] = samples[(y*width + x/nf)*nf + x%nf]`。
    let cfa_w = nf * width;
    let mut cfa: Vec<u16> = vec![0u16; cfa_w.saturating_mul(height)];
    for y in 0..height {
        for x in 0..cfa_w {
            let k = x / nf; // 该 CFA 列落在分量内的第 k 个样本
            let c = x % nf; // 属于第 c 个分量
            let src = (y * width + k) * nf + c;
            if let (Some(&v), Some(slot)) = (img.samples.get(src), cfa.get_mut(y * cfa_w + x)) {
                *slot = v;
            }
        }
    }
    Ok(cfa)
}

/// 解一条 DNG strip/tile 的无损 JPEG 载荷，返回**已按 CFA 重组 + 解交织**的
/// 单通道样本序列。
///
/// # 为什么要以 `SOF3` 的宽高为准
///
/// DNG 1.6 `Compression` 标签描述里的原文：
///
/// > For lossless JPEG, **the internal width/length/components in the JPEG
/// > stream are not required to match the strip or tile's
/// > width/length/components. Only the total sample counts need to match.**
/// > It is common for CFA images to be encoded with a different width, length
/// > or component count to allow the JPEG compression predictors to work
/// > across like colors.
///
/// 故本函数**用 `SOF3` 的 `X`/`Y`/`Nf` 重建栅格**，再按「**总样本数 == strip
/// 面积**」校验。**遇到总样本数不匹配必须结构化拒绝**，绝不静默截断 / 补零。
///
/// # 多分量 CFA 重组（`Nf > 1`）
///
/// 多分量流先按 [`deinterleave_components`] 把 `Nf` 个分量重组成宽
/// `X * Nf` 的单通道 CFA 栅格（映射式与依据见该函数与模块头文档），再走
/// 下方的「去前缀 + 解交织」。单分量（`Nf == 1`）时重组是恒等映射。
///
/// # 参数
///
/// - `bytes` —— strip/tile 的完整字节流（以 `FFD8` 开头）；
/// - `expected_samples` —— strip/tile 标称面积（`width * rows`），用于校验；
/// - `column_offset` —— **每行开头要丢弃的样本数**（`ExtraSamples` 前缀），
///   **缺省 0**。DNG 规范没有为它定义标签（它由「SOF3 内部宽 ≠ strip 标称宽」
///   这条规则隐含表达），故做成显式入参；
/// - `row_interleave_factor` —— `RowInterleaveFactor`（缺省 1 = 不交织）；
/// - `y0` —— 本 strip/tile 首行在**整图**中的行号（多 strip 时行号不连续）。
///
/// # 返回
///
/// 长度 `== expected_samples` 的样本序列，行主序，**已按 `RowInterleaveFactor`
/// 还原为整图行序**。
pub fn decode_dng_strip(
    bytes: &[u8],
    expected_samples: usize,
    column_offset: usize,
    row_interleave_factor: u32,
    y0: u32,
) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw_lossless_jpeg::decode_dng_strip";
    let img = decode_lossless_jpeg(bytes)?;
    // ---- 0) 多分量 → 单通道 CFA 栅格重组（Nf == 1 时恒等）----
    let cfa = deinterleave_components(&img)?;
    // 重组后每行宽度 = Nf × SOF3.X（DNG 的 CFA 实际宽度）。
    let stream_width = (img.components as usize)
        .checked_mul(img.width as usize)
        .ok_or_else(|| {
            tiff_err(
                LOC,
                format!("Nf = {} × X = {} 溢出", img.components, img.width),
                "SOF3 尺寸过大；确认 X/Nf 未被错位读取",
            )
        })?;
    let stream_height = img.height as usize;

    // ---- 1) 去掉每行开头的 ExtraSamples 前缀 ----
    let usable = stream_width.checked_sub(column_offset).ok_or_else(|| {
        tiff_err(
            LOC,
            format!(
                "column_offset = {column_offset} ≥ CFA 行宽 {stream_width}，\
                 整行都是额外样本"
            ),
            "确认 column_offset 未被错位设置；DNG 规范下该值应为 0 或很小",
        )
    })?;
    if usable == 0 {
        return Err(tiff_err(
            LOC,
            format!("去掉 {column_offset} 个前缀样本后每行剩 0 个样本"),
            "确认 column_offset 小于 CFA 行宽",
        ));
    }

    // ---- 2) 总样本数校验（DNG 规范：只要求总数相等）----
    let total = usable.checked_mul(stream_height).ok_or_else(|| {
        tiff_err(
            LOC,
            format!("{usable} × {stream_height} 样本数溢出"),
            "确认 SOF3 的 X/Y/Nf 未被错位读取",
        )
    })?;
    if total != expected_samples {
        return Err(tiff_err(
            LOC,
            format!(
                "总样本数不匹配：CFA {}×{} 去掉 {column_offset} 样本/行 = {total}，\
                 但 strip 标称面积 = {expected_samples}",
                stream_width, stream_height
            ),
            "DNG 1.6 规定 SOF3 内部宽/高/分量数可与 strip 标称不同，但**总样本数必须相等**；\
             请确认 Compression/尺寸标签与载荷自洽",
        ));
    }

    // ---- 3) 行交织还原（RowInterleaveFactor，缺省 1 = 恒等）----
    //
    // **注意**：`map` 给出的是**整图**行号（`interleave_row_map` 的契约），
    // 而返回的 `out` 只覆盖**本 strip**（长度 `== expected_samples`）。
    // 故必须先把整图行号换算成 strip 内行号：`local = global − y0`。
    // 直接拿 `dst_row * usable` 当下标会在 `y0 > 0`（多 strip）时越界/错位，
    // 表现为「后面的 strip 全是 0」——**静默丢数据**，故这里必须显式减y0。
    let map = interleave_row_map(stream_height as u32, y0, row_interleave_factor)?;
    let mut out = vec![0u16; expected_samples];
    for (j, &dst_row) in map.iter().enumerate() {
        let local = (dst_row as usize).saturating_sub(y0 as usize);
        if local >= stream_height {
            return Err(tiff_err(
                LOC,
                format!(
                    "行交织映射越界：目标行 {dst_row} 落在 strip 区间 \
                     [{y0}, {}) 之外",
                    y0 as usize + stream_height
                ),
                "确认传入的 y0 是本strip 首行的全局行号，且 RowInterleaveFactor 自洽",
            ));
        }
        // **源行的步长是 `stream_width`（CFA 实际行宽），不是 `usable`** ——
        // 少了这步，column_offset > 0 时每行都会从错误的位置取样，
        // 结果是「前缀没被跳过」而非「整图错位」（更隐蔽）。
        let src = j * stream_width + column_offset;
        let dst = local * usable;
        if let (Some(s), Some(d)) = (cfa.get(src..src + usable), out.get_mut(dst..dst + usable)) {
            d.copy_from_slice(s);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

/// 把**生产编码器**暴露给同 crate 的其它测试模块（`raw.rs` 的 DNG 端到端测试、
/// `raw_dng_synth` 的合成样本测试都需要它造真实的 `Compression = 7` 载荷）。
///
/// 编码器已从本模块的测试代码里**提升为生产模块**
/// [`crate::raw_lossless_jpeg_encode`]（合成最小合法 DNG 的样本生成器必须能在
/// 生产路径上写出 SOF3 字节流），此处只做转发以保持既有调用点不变。
///
/// >转发时把 `RawResult<Vec<u8>>` **降为 `Vec<u8>`**（测试惯用法：
/// > `unwrap_or_else(|e| panic!("{e}"))`），故既有 `raw.rs` 测试里
/// > `tests_support::encode(...)` 的调用点无需改动。
#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) fn encode(
        samples: &[u16],
        width: u32,
        height: u32,
        precision: u8,
        predictor: u8,
        point_transform: u8,
    ) -> Vec<u8> {
        crate::raw_lossless_jpeg_encode::encode(
            samples,
            width,
            height,
            precision,
            predictor,
            point_transform,
        )
        .unwrap_or_else(|e| panic!("{e}"))
    }
}
