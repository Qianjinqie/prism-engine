//! DNG 解码支持矩阵自证（prism-engine V2.2 · T02「测 1」· 验收 ④ · 设计 §7.3 · Q10）
//!
//! # 目的
//!
//! 在**没有真实相机样本**的前提下，用 `raw_dng_synth` 的合成能力从零造出一个
//! 最小合法 DNG，再走**完整解码管线**（`diagnose_dng_bytes`：`parse →
//! read_raw_samples → develop → 统计`）并额外独立调用一次 `read_raw_samples`，
//! 对每个 `(Compression × 布局)` 组合断言：
//!
//! 1. 合成成功 —— `synthesize_dng` 不返回 `RawError`；
//! 2. 解码成功 —— `diagnose_dng_bytes` 不返回 `RawError`；
//! 3. 输出非空 —— 解码出的 raw 样本数 = `width × height`（> 0），且诊断里
//!    `develop` 已实际运行（去马赛克三通道均值有限）。
//!
//! 三点一起构成「已支持 Compression 1 / 7 / 8 / 32946」的自动化证据。
//!
//! # 矩阵维度
//!
//! | 维度 | 取值 |
//! |---|---|
//! | 维度一 · Compression | `1`（未压缩）/ `7`（无损 JPEG）/ `8`（Deflate）/ `32946`（Deflate 等价厂商码，DNG 1.4 认可） |
//! | 维度二 · 布局 | `strip` / `tile` |
//!
//! 共 8 个格子（`32946` 在接口支持时纳入；`synthesize_dng` 与读端均将 `8` 与
//! `32946` 视为等价并支持，故纳入）。全绿即代表主路径自证成立。
//!
//! # 诚实边界（Q10）
//!
//! 真实机型 golden 属于**待验登记**档，不在本测试范围：本测试用合成 DNG 证明
//! 「读端主路径能跑通 1/7/8/32946 的 strip 与 tile 布局」，但不声称「某真实相机
//! 的 `.dng` 已逐位核对」——那需要真实样本，而沙箱内 `find *.dng` 零命中。
//!
//! # 纪律
//!
//! 本文件全部 `#[test]`、**禁止 `#[ignore]`**；自包含、可重复、不依赖任何外部文件。

use prism_io::raw_dng_synth::{diagnose_dng_bytes, synthesize_dng, DngLayout, DngSampleSpec};
use prism_io::DngFile;

/// 参与矩阵的全部 Compression 取值（维度一）。
///
/// `32946` 是 Deflate 的厂商私有等价码（`0x80B2`），DNG 1.4 继承 TIFF 规范
/// 「压缩算法与 8 完全相同，应视为过时但建议识别并读取」。合成器（`raw_dng_synth`
/// 的 `synthesize_dng` 匹配 `8 | 32946`）与读端（`raw::DngFile::read_raw_samples`
/// 的分派）均已支持，故纳入矩阵。
const COMPRESSIONS: [(u16, &str); 4] = [
    (1, "未压缩"),
    (7, "无损JPEG"),
    (8, "Deflate"),
    (32946, "Deflate(32946)"),
];

/// 参与矩阵的全部布局（维度二）。
///
/// `strip` 用 `rows_per_strip = 8`（图高 12 ⇒ 2 个 strip，覆盖多 strip 拼接）；
/// `tile` 用 `8×6`（图幅 16×12 ⇒ 2×2 个 tile，覆盖多 tile 网格）。
const LAYOUTS: [DngLayout; 2] = [
    DngLayout::Strip { rows_per_strip: 8 },
    DngLayout::Tile {
        tile_width: 8,
        tile_length: 6,
    },
];

/// 合成 + 解码单一矩阵格子的核心自证。
///
/// # 参数
/// * `compression` —— 维度一取值（`1` / `7` / `8` / `32946`）。
/// * `layout` —— 维度二取值（`strip` / `tile`）。
///
/// # 返回
/// `(raw_samples_len, diag_width, diag_height, is_tile)` —— 供上层断言与口径对账。
fn prove_cell(compression: u16, layout: DngLayout) -> (usize, u32, u32, bool) {
    const W: u32 = 16;
    const H: u32 = 12;

    // ---- 合成：从零造一个最小合法 DNG（无需任何真实相机样本）----
    let spec = DngSampleSpec {
        width: W,
        height: H,
        bits: 16,
        cfa: [0, 1, 1, 2],
        compression,
        layout,
        ..DngSampleSpec::default()
    };
    let bytes = synthesize_dng(&spec).unwrap_or_else(|e| {
        panic!("合成失败 [compression={compression}, layout={layout:?}]: {e}")
    });

    // ---- 解码：完整管线（parse → raw → develop → 统计）----
    let diag = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| {
        panic!("解码失败 [compression={compression}, layout={layout:?}]: {e}")
    });

    // ---- 独立再读一次 raw samples：确认读端**确实解出非空像素** ----
    let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("DngFile::parse 失败: {e}"));
    let mosaic = dng.read_raw_samples().unwrap_or_else(|e| {
        panic!("read_raw_samples 失败 [compression={compression}, layout={layout:?}]: {e}")
    });

    // ---- 不变量 ①：口径对账（Compression / 尺寸须与合成一致）----
    assert_eq!(
        diag.compression, compression,
        "诊断报出的 Compression 须与合成时一致（compression={compression}）"
    );
    assert_eq!(
        (diag.width, diag.height),
        (W, H),
        "诊断尺寸须与合成时一致（compression={compression}）"
    );

    // ---- 不变量 ②：输出非空（解码样本数 = 宽×高，且 > 0）----
    assert!(
        !mosaic.is_empty(),
        "解码样本不得为空（compression={compression}, layout={layout:?}）"
    );
    assert_eq!(
        mosaic.len(),
        (W * H) as usize,
        "解码样本数须 = 宽×高（非空且有界，compression={compression}）"
    );

    // ---- 不变量 ③：develop 已实际运行（去马赛克三通道均值有限，非 NaN/±∞）----
    assert!(
        diag.demosaic_mean_rgb.iter().all(|v| v.is_finite()),
        "去马赛克均值须有限（develop 已运行，compression={compression}）"
    );

    // ---- 不变量 ④：布局口径对账（strip 合成→strip 诊断，tile 同理）----
    let is_tile = matches!(layout, DngLayout::Tile { .. });
    if is_tile {
        assert!(
            diag.layout.contains("tile"),
            "tile 合成须被诊断为 tile 布局，实得：{}",
            diag.layout
        );
    } else {
        assert!(
            diag.layout.contains("strip"),
            "strip 合成须被诊断为 strip 布局，实得：{}",
            diag.layout
        );
    }

    (mosaic.len(), diag.width, diag.height, is_tile)
}

/// 生成单格子的 `#[test]`（命名规则 `dng_{layout}_{compression}`）。
///
/// 每个格子独立成测，失败即精确定位到 `(Compression, 布局)`，且**不使用
/// `#[ignore]`**——全绿才是矩阵自证成立。
macro_rules! matrix_test {
    ($name:ident, $compression:expr, $layout:expr) => {
        #[test]
        fn $name() {
            let (n, w, h, _) = prove_cell($compression, $layout);
            assert_eq!(n, (w * h) as usize, "格子须解出非空且有界像素");
        }
    };
}

// ---- 维度二 = strip × 维度一 {1,7,8,32946} ----
matrix_test!(dng_strip_compression_1, 1, DngLayout::Strip { rows_per_strip: 8 });
matrix_test!(dng_strip_compression_7, 7, DngLayout::Strip { rows_per_strip: 8 });
matrix_test!(dng_strip_compression_8, 8, DngLayout::Strip { rows_per_strip: 8 });
matrix_test!(
    dng_strip_compression_32946,
    32946,
    DngLayout::Strip { rows_per_strip: 8 }
);

// ---- 维度二 = tile × 维度一 {1,7,8,32946} ----
matrix_test!(
    dng_tile_compression_1,
    1,
    DngLayout::Tile {
        tile_width: 8,
        tile_length: 6
    }
);
matrix_test!(
    dng_tile_compression_7,
    7,
    DngLayout::Tile {
        tile_width: 8,
        tile_length: 6
    }
);
matrix_test!(
    dng_tile_compression_8,
    8,
    DngLayout::Tile {
        tile_width: 8,
        tile_length: 6
    }
);
matrix_test!(
    dng_tile_compression_32946,
    32946,
    DngLayout::Tile {
        tile_width: 8,
        tile_length: 6
    }
);

// ===========================================================================
//  合并矩阵测试（验收 ④ 主证）：一次性跑完所有格子并打印矩阵
// ===========================================================================

#[test]
fn dng_support_matrix_full_cross_all_green() {
    // 打印矩阵表头，便于「交叉矩阵全绿」的人工核对（cargo test -- --nocapture）。
    println!("DNG 解码支持矩阵（合成 → 完整解码管线，输出非空 = raw=宽×高）");
    println!("                  │ strip │ tile ");
    println!("------------------┼───────┼──────");

    let mut all_ok = true;
    for (code, name) in COMPRESSIONS {
        let mut row = format!("Comp={code:>5} {name:<10}│");
        for layout in LAYOUTS {
            // 用 catch_unwind 让矩阵「全部格子都跑完、全部显式标注」，
            // 即使某格 panic 也不阻断其余格子的结果呈现。
            let r = std::panic::catch_unwind(|| prove_cell(code, layout));
            match r {
                Ok((n, w, h, _)) => {
                    let ok = n == (w * h) as usize;
                    row.push_str(&format!(" {:<5} │", if ok { "OK" } else { "BAD" }));
                    if !ok {
                        all_ok = false;
                    }
                }
                Err(_) => {
                    row.push_str(" FAIL │");
                    all_ok = false;
                }
            }
        }
        println!("{row}");
    }
    println!("------------------┼───────┼──────");
    assert!(all_ok, "DNG 支持矩阵存在非绿格子，见上表");
}
