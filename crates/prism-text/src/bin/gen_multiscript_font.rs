//! 多语种字体资产生成器（纯 Rust，零第三方 crate）。
//!
//! 运行：`cargo run -p prism-text --bin gen_multiscript_font`
//!
//! 产出 `crates/prism-text/assets/prism_multiscript.ttf`，并打印其
//! 字节数、SHA-256、表目录与关键统计（码位覆盖、GSUB 特征/查找数）。
//!
//! # 确定性
//!
//! 生成是纯函数：同一源码两次运行产出**逐字节一致**的文件（SHA-256 稳定）。
//! 因此 `tests/multiscript_asset.rs` 可以把 SHA-256 作为回归锚点。
//!
//! # 为什么不是构建脚本（build.rs）
//!
//! 本仓约定：资产是**入库的确定性产物**，不是每次构建现算——这样 QA、审计、
//! 版本对比都面对同一份字节。生成器是独立可执行，人工/CI 在需要时重跑。

use std::fs;
use std::path::PathBuf;

fn main() {
    let data = prism_text::multiscript_font::build_font();

    // 输出路径：crate 根 / assets / prism_multiscript.ttf。
    // CARGO_MANIFEST_DIR 在 bin 目标下指向 crate 根（crates/prism-text）。
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest_dir.join("assets");
    fs::create_dir_all(&out_dir).expect("创建 assets 目录");
    let out_path = out_dir.join("prism_multiscript.ttf");
    fs::write(&out_path, &data).expect("写入字体资产");

    let sha = prism_io::sha256(&data);
    let sha_hex = sha.iter().map(|b| format!("{b:02x}")).collect::<String>();

    println!("== prism_multiscript.ttf ==");
    println!("路径 : {}", out_path.display());
    println!("字节 : {} B", data.len());
    println!("sha256 : {sha_hex}");

    // 用本仓引擎自检生成物。
    let ot = prism_text::opentype::OtFont::parse(&data).expect("生成物应可解析");
    println!("sfnt : 0x{:08x}", ot.sfnt_version());
    println!(
        "表   : {}",
        ot.directory()
            .tags()
            .iter()
            .map(|t| String::from_utf8_lossy(t).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let gsub = ot.gsub().expect("GSUB 表应存在");
    let t = prism_text::gsub::LayoutTable::parse(gsub).expect("GSUB 应可解析");
    println!(
        "GSUB : {} B / {} 特征 / {} 查找",
        gsub.len(),
        t.features().len(),
        t.lookups().len()
    );
    let f = ot.glyphs().expect("ttf::Font");
    // 码位覆盖计数（阿拉伯 U+0600–U+06FF / 天城文 U+0900–U+097F / 拉丁 A–Z）。
    let count = |lo: u32, hi: u32| -> usize {
        (lo..=hi)
            .filter(|&cp| char::from_u32(cp).and_then(|c| f.glyph_index(c)).is_some())
            .count()
    };
    println!("阿拉伯码位 : {}", count(0x0600, 0x06FF));
    println!("天城文码位 : {}", count(0x0900, 0x097F));
    println!("拉丁A-Z码位 : {}", count(0x0041, 0x005A));
    println!("拉丁a-z码位 : {}", count(0x0061, 0x007A));
    println!("数字0-9码位 : {}", count(0x0030, 0x0039));
    println!("字形总数   : {}", f.num_glyphs());
    // off-curve（二次贝塞尔控制点）字形数：本路技术核心的存在性自检。
    let with_off = (0..f.num_glyphs().min(u32::from(u16::MAX)) as u16)
        .filter_map(|g| f.outline(g).ok())
        .filter(|o| o.contours.iter().flatten().any(|p| !p.on_curve))
        .count();
    println!("含贝塞尔曲线字形 : {with_off}");
}
