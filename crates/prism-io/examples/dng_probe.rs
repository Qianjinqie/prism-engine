//! **DNG 端到端探针**：给定 `.dng` 路径 → 走完整管线 → 打印关键诊断。
//!
//! 这是路C 的「样本驱动端到端测试入口」。用户未来丢一个真实相机 `.dng` 进来，
//! **不需要再改任何代码**，直接跑：
//!
//! ```bash
//! # 诊断一个真实 DNG
//! cargo run -p prism-io --example dng_probe -- /path/to/real.dng
//!
//! # 先自造一个最小合法 DNG，再诊断它（链路自检，不需要真实样本）
//! cargo run -p prism-io --example dng_probe -- --synth /tmp/mini.dng --diagnose
//!
//! # 无损 JPEG + tile + 12 位 + Nf=2 CFA 重打包
//! cargo run -p prism-io --example dng_probe -- --synth /tmp/t.dng \
//!     --bits 12 --compression 7 --tile 16x16 --components 2
//!
//! # Deflate(ZIP) + Horizontal 预测器 + SubIFD 预览图（W12 路E）
//! cargo run -p prism-io --example dng_probe -- --synth /tmp/d.dng \
//!     --compression 8 --deflate-predictor 2 --preview 160x120 --diagnose
//! ```
//!
//! 退出码：`0` 成功，`2` 结构化错误（错误码/位置/成因/建议已打印），`1` 参数错误。
//!
//! C18：本文件属工具层，同样**不 panic**——所有失败都走结构化打印 + 退出码。

use std::process::ExitCode;

use prism_io::raw::{ByteOrder, FillOrder, RawError};
use prism_io::{
    diagnose_dng, synthesize_dng, DngLayout, DngSampleSpec, PreviewSpec,
};

const USAGE: &str = "\
用法：
  dng_probe <file.dng>                       诊断一个 .dng 文件
  dng_probe --synth <out.dng> [选项] [--diagnose]
                                              合成一个最小合法 DNG（可选立即诊断）

选项：
  --width N            图宽（默认 32）
  --height N           图高（默认 16）
  --bits N             位深 2..=16（默认 16）
  --compression N      1=未压缩 / 7=无损JPEG / 8或32946=Deflate(ZIP)（默认 1）
  --cfa R G B G        CFA 平铺颜色（默认 RGGB = 0 1 1 2）
  --rows-per-strip N   strip 行数（默认 = 图高；与 --tile 互斥）
  --tile WxH           改用 tile 布局（与 --rows-per-strip 互斥）
  --components N       无损 JPEG 的 Nf（1..=4，默认 1）
  --predictor N        无损 JPEG 的 Ss 预测器（1..=7，默认 1）
  --deflate-predictor N Deflate 的 Predictor 标签（1/2/34892/34893，默认 1）
  --preview WxH        写一张 SubIFD 预览图（默认 160x120 像素、8 位、
                       无损 JPEG、RGB）；也可用 --preview-compression 指定其压缩方式
  --big-endian         写 MM（大端）容器
  --lsb-first          Compression=1 时用 LSB-first 位序
  --row-interleave N   写 RowInterleaveFactor=N
  --preview-compression N  预览图压缩方式（1/7/8，默认 7）
";

/// 参数解析 / 运行失败。
enum Fail {
    /// 用法错误（退出码 1）。
    Usage(String),
    /// 库返回的结构化错误（退出码 2）。
    Raw(RawError),
}

impl From<RawError> for Fail {
    fn from(e: RawError) -> Self {
        Fail::Raw(e)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail::Usage(m)) => {
            eprintln!("[DNG 探针] 参数错误：{m}");
            eprint!("{USAGE}");
            ExitCode::from(1)
        }
        Err(Fail::Raw(e)) => {
            eprintln!("[DNG 探针] 失败");
            eprintln!("  错误码 : {}", e.code);
            eprintln!("  位置   : {}", e.location);
            eprintln!("  成因   : {}", e.problem);
            eprintln!("  建议   : {}", e.fix);
            ExitCode::from(2)
        }
    }
}

fn parse_u32(s: &str, name: &str) -> Result<u32, Fail> {
    s.parse::<u32>()
        .map_err(|_| Fail::Usage(format!("{name} 的取值 {s:?} 不是合法非负整数")))
}

/// 参数游标。
struct Args<'a> {
    items: &'a [String],
    idx: usize,
}

impl<'a> Args<'a> {
    fn value(&mut self, name: &str) -> Result<&'a str, Fail> {
        self.idx += 1;
        self.items
            .get(self.idx)
            .map(String::as_str)
            .ok_or_else(|| Fail::Usage(format!("{name} 缺少取值")))
    }

    fn u32(&mut self, name: &str) -> Result<u32, Fail> {
        let raw = self.value(name)?;
        parse_u32(raw, name)
    }
}

fn run(argv: &[String]) -> Result<(), Fail> {
    if argv.is_empty() {
        return Err(Fail::Usage("缺少参数".to_string()));
    }

    // ---- 模式一：直接诊断既有文件 ----------------------------------------
    if !argv[0].starts_with("--") {
        if argv.len() > 1 {
            return Err(Fail::Usage(format!("诊断模式不接受额外参数：{argv:?}")));
        }
        let path = argv[0].as_str();
        println!("=== DNG 端到端诊断：{path} ===");
        let d = diagnose_dng(path)?;
        print!("{}", d.report());
        return Ok(());
    }

    // ---- 模式二：合成 ----------------------------------------------------
    let mut out: Option<&str> = None;
    let mut spec = DngSampleSpec::new(32, 16, 16, [0, 1, 1, 2]);
    let mut tile: Option<DngLayout> = None;
    let mut rows: Option<u32> = None;
    let mut do_diagnose = false;
    let mut big_endian = false;
    let mut lsb_first = false;
    let mut preview: Option<(u32, u32)> = None;
    let mut preview_compression: u16 = 7;

    let mut p = Args { items: argv, idx: 0 };
    while p.idx < argv.len() {
        let flag = argv[p.idx].as_str();
        match flag {
            "--synth" => out = Some(p.value("--synth")?),
            "--diagnose" => do_diagnose = true,
            "--width" => spec.width = p.u32("--width")?,
            "--height" => spec.height = p.u32("--height")?,
            "--bits" => spec.bits = p.u32("--bits")? as u16,
            "--compression" => spec.compression = p.u32("--compression")? as u16,
            "--components" => spec.components = p.u32("--components")? as u16,
            "--predictor" => spec.predictor = p.u32("--predictor")? as u8,
            "--row-interleave" => spec.row_interleave_factor = Some(p.u32("--row-interleave")?),
            "--cfa" => {
                let mut cfa = [0u8; 4];
                for slot in cfa.iter_mut() {
                    *slot = p.u32("--cfa")? as u8;
                }
                spec.cfa = cfa;
            }
            "--rows-per-strip" => rows = Some(p.u32("--rows-per-strip")?),
            "--tile" => {
                let v = p.value("--tile")?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .ok_or_else(|| Fail::Usage(format!("--tile 需要 WxH 形式，得到 {v}")))?;
                tile = Some(DngLayout::Tile {
                    tile_width: parse_u32(w, "--tile")?,
                    tile_length: parse_u32(h, "--tile")?,
                });
            }
            "--deflate-predictor" => {
                spec.deflate_predictor = p.u32("--deflate-predictor")? as u16
            }
            "--preview-compression" => {
                preview_compression = p.u32("--preview-compression")? as u16
            }
            "--preview" => {
                let v = p.value("--preview")?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .ok_or_else(|| Fail::Usage(format!("--preview 需要 WxH 形式，得到 {v}")))?;
                preview = Some((parse_u32(w, "--preview")?, parse_u32(h, "--preview")?));
            }
            "--big-endian" => big_endian = true,
            "--lsb-first" => lsb_first = true,
            other => return Err(Fail::Usage(format!("未知参数 {other}"))),
        }
        p.idx += 1;
    }

    let out = out.ok_or_else(|| Fail::Usage("--synth 需要一个输出路径".to_string()))?;
    if tile.is_some() && rows.is_some() {
        return Err(Fail::Usage(
            "--tile 与 --rows-per-strip 是两种互斥的载荷布局，只能给一个".to_string(),
        ));
    }
    spec.layout = match (tile, rows) {
        (Some(t), _) => t,
        (None, Some(r)) => DngLayout::Strip { rows_per_strip: r },
        (None, None) => DngLayout::Strip { rows_per_strip: spec.height },
    };
    if big_endian {
        spec.order = ByteOrder::Big;
    }
    if lsb_first {
        spec.fill_order = FillOrder::LsbFirst;
    }
    if let Some((w, h)) = preview {
        spec.preview = Some(PreviewSpec {
            width: w,
            height: h,
            compression: preview_compression,
            ..PreviewSpec::default()
        });
    }

    let bytes = synthesize_dng(&spec)?;
    std::fs::write(out, &bytes).map_err(|e| {
        RawError::new(
            "RAW1001",
            "dng_probe::write",
            format!("写出 {out} 失败：{e}"),
            "确认目标目录存在且可写",
        )
    })?;
    println!("已合成 {out}（{} 字节）", bytes.len());
    println!(
        "规格: {}×{} {}bit Compression={} Nf={} {:?}",
        spec.width, spec.height, spec.bits, spec.compression, spec.components, spec.layout
    );
    if spec.compression == 8 || spec.compression == 32946 {
        println!("Deflate 预测器: Predictor={}", spec.deflate_predictor);
    }
    if let Some(pv) = spec.preview {
        println!(
            "SubIFD 预览: {}×{} {}bit Compression={} PI={} spp={}",
            pv.width, pv.height, pv.bits, pv.compression, pv.photometric, pv.samples_per_pixel
        );
    }

    if do_diagnose {
        println!("=== 合成样本的端到端诊断 ===");
        let d = diagnose_dng(out)?;
        print!("{}", d.report());
    }
    Ok(())
}