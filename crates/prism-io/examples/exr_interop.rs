//! DWA 互通对拍 · Rust 侧 harness（与 `oexr_interop.cpp` 通过**文件字节**交互）。
//!
//! # 定位：验证工具，不是项目依赖
//!
//! 官方 OpenEXR 3.1.5 只出现在 [`oexr_interop.cpp`]（`tests/interop/` 下的
//! **纯源码**）里，**不进任何 `Cargo.toml`、没有 `build.rs`、不链接 C++**。
//! 本文件与它之间只有 `.exr` / `.txt` 两种文件字节，没有共享 ABI、没有共享内存。
//!
//! # 为什么要有 `src` 这一步（反假绿）
//!
//! 对拍的前提是**两侧喂进去的是同一批源数据**。若两侧各自跑一遍
//! `gen_pattern`，`exp()` 差 1 ULP 就会让测出的"压缩误差"里混入源数据差异——
//! 那是 harness 的噪声，不是压缩器的行为。故由 C++ 侧 `oexr_interop src` 导出
//! 源数据的 f32 位模式，本 harness 的 `gen` **直接读那份文件**，从构造上
//! 消除"两侧源数据不同"这一类噪声。
//!
//! # 子命令（与 C++ 侧一一对应）
//!
//! ```text
//! exr_interop gen    <out.exr> <w> <h> <spec> <src.txt>   我们写（DWAA/DWAB）
//! exr_interop dump   <in.exr> <out.txt>                    我们读 → DWA-DUMP 文本
//! exr_interop read   <in.exr>                              我们读（不导出），报 OK/错
//! exr_interop srccmp <in.exr> <src.txt>                    我们解码 vs 源数据（有损幅度）
//! exr_interop cmpdump <a.txt> <b.txt>                      两份 dump 文本互比
//! exr_interop acinfo <in.exr>                              打印首块 DWA 块头
//! exr_interop acdump <in.exr> <out.txt>                     我们的 Huffman 解码 → u16 文本
//! ```
//!
//! `spec` 形如 `dwaa-half-scan-3-noise`，字段与 C++ 侧 `parse_spec` 完全同构。
//!
//! # 用法
//!
//! ```bash
//! cargo run --release -p prism-io --example exr_interop -- gen out.exr 64 48 dwaa-half-scan-3-noise src.txt
//! ```

use prism_io::exr::{self, ChannelPlane, Compression, PixelType};
use prism_io::exr_dwa;
use prism_io::exr_write;

use std::collections::BTreeMap;
use std::fs;
use std::process;

// ===========================================================================
//  一、DWA-DUMP 文本格式（与 C++ 侧 cmd_dump / cmd_src 同构）
// ===========================================================================

/// 一份 dump：尺寸 + 逐通道 f32 数据（顺序 = 通道名字典序）。
#[derive(Debug, Clone)]
struct Dump {
    w: usize,
    h: usize,
    /// 通道名 → f32 平面（`BTreeMap` 保证字典序 = 官方 `ChannelList` 序）。
    planes: BTreeMap<String, Vec<f32>>,
}

/// 解析 DWA-DUMP 1 文本。
///
/// 格式（每行以 `\n` 结束，行内单空格分隔）：
/// ```text
/// DWA-DUMP 1
/// W <w>
/// H <h>
/// NC <n>
/// C <index> <name> <ptype> <count>
/// V <v0> <v1> ...          8 位十六进制 = f32 的 to_bits()，行主序
/// ```
fn parse_dump(text: &str) -> Result<Dump, String> {
    let mut w: Option<usize> = None;
    let mut h: Option<usize> = None;
    let mut planes: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    // 当前正在接收 V 数据的通道名（C 行设置，V 行消费）。
    let mut cur: Option<String> = None;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next() {
            Some("W") => {
                let tok = it.next().ok_or_else(|| format!("W 行缺值: {line}"))?;
                w = Some(tok.parse::<usize>().map_err(|e| format!("W 行解析失败: {e}"))?);
            }
            Some("H") => {
                let tok = it.next().ok_or_else(|| format!("H 行缺值: {line}"))?;
                h = Some(tok.parse::<usize>().map_err(|e| format!("H 行解析失败: {e}"))?);
            }
            Some("NC") | Some("DWA-DUMP") => continue,
            // `C <index> <name> <ptype> <count>`：登记一个新通道并预分配容量。
            Some("C") => {
                let _idx = it.next().ok_or_else(|| format!("C 行缺 index: {line}"))?;
                let name = it
                    .next()
                    .ok_or_else(|| format!("C 行缺通道名: {line}"))?
                    .to_string();
                let count: usize = it
                    .nth(1)
                    .ok_or_else(|| format!("C 行缺 count: {line}"))?
                    .parse()
                    .map_err(|e| format!("C 行 count 解析失败: {e}"))?;
                planes.insert(name.clone(), Vec::with_capacity(count));
                cur = Some(name);
            }
            Some("V") => {
                let name = cur
                    .clone()
                    .ok_or_else(|| "V 行出现在任何 C 行之前".to_string())?;
                let plane = planes.get_mut(&name).expect("C 行已登记该通道");
                for tok in it {
                    let bits = u32::from_str_radix(tok, 16)
                        .map_err(|e| format!("V 行位模式 {tok:?} 解析失败: {e}"))?;
                    plane.push(f32::from_bits(bits));
                }
            }
            Some(other) => return Err(format!("未知行首 {other:?}: {line}")),
            None => continue,
        }
    }
    Ok(Dump {
        w: w.ok_or("缺 W 行")?,
        h: h.ok_or("缺 H 行")?,
        planes,
    })
}

/// 序列化为 DWA-DUMP 1 文本（每行 16 个值，与 C++ 侧格式逐字节对齐）。
fn format_dump(d: &Dump, types: &BTreeMap<String, PixelType>) -> String {
    let mut out = String::new();
    out.push_str("DWA-DUMP 1\n");
    out.push_str(&format!("W {}\n", d.w));
    out.push_str(&format!("H {}\n", d.h));
    out.push_str(&format!("NC {}\n", d.planes.len()));
    for (i, (name, plane)) in d.planes.iter().enumerate() {
        let tn = match types.get(name).copied().unwrap_or(PixelType::Half) {
            PixelType::Half => "HALF",
            PixelType::Float => "FLOAT",
            PixelType::Uint => "UINT",
        };
        out.push_str(&format!("C {} {} {} {}\n", i, name, tn, plane.len()));
        for (k, v) in plane.iter().enumerate() {
            if k % 16 == 0 {
                out.push_str("V ");
            } else {
                out.push(' ');
            }
            out.push_str(&format!("{:08x}", v.to_bits()));
            if k % 16 == 15 || k + 1 == plane.len() {
                out.push('\n');
            }
        }
    }
    out
}

// ===========================================================================
//  二、spec 解析（与 C++ 侧 `parse_spec` 同构）
// ===========================================================================

/// 对拍矩阵的一格。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Spec {
    compression: Compression,
    pixel_type: PixelType,
    tiled: bool,
    nchan: usize,
    /// 形态名（`gradient` / `noise` / `const` / `c<value>`），仅用于报错定位——
    /// 源数据一律从 C++ 侧 `src` 导出的文件读入，本字段不参与数据生成。
    pattern: String,
}

/// 通道名：1=Y、3=RGB、4=RGBA（官方 DWA 默认规则表按这些后缀分类）。
fn chan_name(nchan: usize, index: usize) -> &'static str {
    match (nchan, index) {
        (1, 0) => "Y",
        (3, 0) => "R",
        (3, 1) => "G",
        (3, 2) => "B",
        (4, 0) => "R",
        (4, 1) => "G",
        (4, 2) => "B",
        (4, 3) => "A",
        _ => "Y",
    }
}

fn parse_spec(s: &str) -> Result<Spec, String> {
    let f: Vec<&str> = s.split('-').collect();
    if f.len() != 5 {
        return Err(format!("spec 须 5 段（comp-ptype-layout-nchan-pattern），收到 {s:?}"));
    }
    let compression = match f[0] {
        "dwaa" => Compression::Dwaa,
        "dwab" => Compression::Dwab,
        other => return Err(format!("未知压缩 {other:?}")),
    };
    let pixel_type = match f[1] {
        "half" => PixelType::Half,
        "float" => PixelType::Float,
        other => return Err(format!("未知像素类型 {other:?}")),
    };
    let tiled = match f[2] {
        "scan" => false,
        "tile" => true,
        other => return Err(format!("未知布局 {other:?}")),
    };
    let nchan: usize = f[3].parse().map_err(|_| format!("通道数解析失败 {:?}", f[3]))?;
    if !(1..=4).contains(&nchan) {
        return Err(format!("通道数须 ∈ 1..=4，收到 {nchan}"));
    }
    Ok(Spec {
        compression,
        pixel_type,
        tiled,
        nchan,
        pattern: f[4].to_string(),
    })
}

// ===========================================================================
//  三、我们写：gen
// ===========================================================================

/// 由 C++ 侧导出的源数据构造 `ChannelPlane` 列表（按 spec 的通道名/类型）。
fn planes_from_src(src: &Dump, sp: &Spec) -> Vec<ChannelPlane> {
    (0..sp.nchan)
        .map(|c| {
            let name = chan_name(sp.nchan, c);
            let values = src.planes.get(name).cloned().unwrap_or_else(|| {
                panic!("源数据缺通道 {name}（spec={}）", sp.pattern)
            });
            ChannelPlane {
                name: name.to_string(),
                pixel_type: sp.pixel_type,
                width: src.w as u32,
                height: src.h as u32,
                values,
            }
        })
        .collect()
}

fn cmd_gen(args: &[String]) -> Result<(), String> {
    // gen <out.exr> <w> <h> <spec> <src.txt>
    if args.len() < 5 {
        return Err("用法: gen <out.exr> <w> <h> <spec> <src.txt>".into());
    }
    let out = &args[0];
    let w: u32 = args[1].parse().map_err(|_| "宽解析失败".to_string())?;
    let h: u32 = args[2].parse().map_err(|_| "高解析失败".to_string())?;
    let sp = parse_spec(&args[3])?;
    let src = parse_dump(&fs::read_to_string(&args[4]).map_err(|e| format!("读源数据: {e}"))?)?;

    let planes = planes_from_src(&src, &sp);
    let chans: Vec<(&str, PixelType)> = planes
        .iter()
        .map(|p| (p.name.as_str(), p.pixel_type))
        .collect();
    // tiled 与 scanline 走**同一个**头部构造 + 同一个压缩参数，
    // 只切换布局入口。tile 尺寸固定 32×32 —— 与 C++ harness 的
    // `hdr.setTileDescription(TileDescription(32, 32, ONE_LEVEL))` 一致，
    // 否则两侧的块边界不同，方向 C 的「同一源数据」前提会被破坏。
    const TILE: u32 = 32;
    let bytes = if sp.tiled {
        let header = exr_write::header_for_tiled(w, h, &chans, sp.compression, TILE, TILE);
        exr_write::write_tiled_image_with(&header, &planes, sp.compression)
    } else {
        let header = exr_write::header_for(w, h, &chans, sp.compression);
        exr_write::write_image(&header, &planes, sp.compression)
    }
    .map_err(|e| format!("DWA 编码失败: {e:?}"))?;
    fs::write(out, &bytes).map_err(|e| format!("写 {out}: {e}"))?;
    println!(
        "GEN ok {} {w}x{h} spec={} layout={} bytes={}",
        out,
        args[3],
        if sp.tiled { "tile" } else { "scan" },
        bytes.len()
    );
    Ok(())
}

// ===========================================================================
//  四、我们读：dump / read
// ===========================================================================

/// 读一个 `.exr`（扫描线或分块皆可）成 [`Dump`]。
fn read_exr(path: &str) -> Result<(Dump, BTreeMap<String, PixelType>), String> {
    let data = fs::read(path).map_err(|e| format!("读 {path}: {e}"))?;
    let header = exr::read_header(&data).map_err(|e| format!("解析头 {path}: {e:?}"))?;
    let img = if header.tiled {
        exr::read_all_tiles(&data, &header)
    } else {
        exr::read_image(&data)
    }
    .map_err(|e| format!("解码 {path}: {e:?}"))?;

    let mut planes: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut types: BTreeMap<String, PixelType> = BTreeMap::new();
    for p in &img.planes {
        planes.insert(p.name.clone(), p.values.clone());
        types.insert(p.name.clone(), p.pixel_type);
    }
    let w = header.data_window.width() as usize;
    let h = header.data_window.height() as usize;
    Ok((Dump { w, h, planes }, types))
}

fn cmd_dump(args: &[String]) -> Result<(), String> {
    // dump <in.exr> <out.txt>
    if args.len() < 2 {
        return Err("用法: dump <in.exr> <out.txt>".into());
    }
    let (d, types) = read_exr(&args[0])?;
    fs::write(&args[1], format_dump(&d, &types)).map_err(|e| format!("写 {}: {e}", args[1]))?;
    println!(
        "DUMP ok {} -> {} ({}x{} nc={})",
        args[0],
        args[1],
        d.w,
        d.h,
        d.planes.len()
    );
    Ok(())
}

fn cmd_read(args: &[String]) -> Result<(), String> {
    // read <in.exr>
    if args.is_empty() {
        return Err("用法: read <in.exr>".into());
    }
    let (d, _) = read_exr(&args[0])?;
    let sum: f64 = d
        .planes
        .values()
        .flat_map(|p| p.iter())
        .map(|v| *v as f64)
        .sum();
    println!(
        "READ_OK 1 {}x{} nc={} checksum={}",
        d.w,
        d.h,
        d.planes.len(),
        sum
    );
    Ok(())
}

// ===========================================================================
//  五、比误差：srccmp / cmpdump
// ===========================================================================

/// 比两份 dump，输出与 C++ 侧同构的机器可读指标。
///
/// 相对误差口径（**两侧必须一致**，否则数字不可比）：
/// ```text
/// rel = |a - b| / max(|b|, 1e-6)
/// ```
/// `b` 为基准（基准 = 源数据或官方解码结果）。
fn compare(tag: &str, decoded: &Dump, base: &Dump) -> Result<(), String> {
    if decoded.w != base.w || decoded.h != base.h {
        return Err(format!(
            "尺寸不一致：decoded {}x{} vs base {}x{}",
            decoded.w, decoded.h, base.w, base.h
        ));
    }
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut bitdiff: u64 = 0;
    let mut total: u64 = 0;
    let mut max_at = (-1i64, -1i64);
    for (name, pd) in &decoded.planes {
        let pb = base
            .planes
            .get(name)
            .ok_or_else(|| format!("基准缺通道 {name}"))?;
        let n = pd.len().min(pb.len());
        for i in 0..n {
            let a = pd[i] as f64;
            let b = pb[i] as f64;
            let ad = (a - b).abs();
            let den = if b.abs() > 1e-6 { b.abs() } else { 1e-6 };
            let rd = ad / den;
            if pd[i].to_bits() != pb[i].to_bits() {
                bitdiff += 1;
            }
            total += 1;
            if ad > max_abs {
                max_abs = ad;
            }
            if rd > max_rel {
                max_rel = rd;
                let idx = i as i64;
                max_at = (name_index(base, name), idx);
            }
        }
    }
    println!("{tag}_OK 1");
    println!("DIM {} {} {}", decoded.w, decoded.h, decoded.planes.len());
    println!("MAX_ABS {max_abs:.9e}");
    println!("MAX_REL {max_rel:.9e}");
    println!("MAX_AT {} {}", max_at.0, max_at.1);
    println!("BITDIFF {bitdiff}");
    println!("N {total}");
    Ok(())
}

/// 通道在 `BTreeMap` 中的序号（仅用于报错定位）。
fn name_index(d: &Dump, name: &str) -> i64 {
    d.planes
        .keys()
        .position(|k| k == name)
        .map(|p| p as i64)
        .unwrap_or(-1)
}

fn cmd_srccmp(args: &[String]) -> Result<(), String> {
    // srccmp <in.exr> <src.txt>
    if args.len() < 2 {
        return Err("用法: srccmp <in.exr> <src.txt>".into());
    }
    let (decoded, _) = read_exr(&args[0])?;
    let base = parse_dump(&fs::read_to_string(&args[1]).map_err(|e| format!("读源数据: {e}"))?)?;
    compare("SRCCMP", &decoded, &base)
}

fn cmd_cmpdump(args: &[String]) -> Result<(), String> {
    // cmpdump <a.txt> <b.txt>  —— a 为被测，b 为基准
    if args.len() < 2 {
        return Err("用法: cmpdump <a.txt> <b.txt>".into());
    }
    let a = parse_dump(&fs::read_to_string(&args[0]).map_err(|e| format!("读 {}: {e}", args[0]))?)?;
    let b = parse_dump(&fs::read_to_string(&args[1]).map_err(|e| format!("读 {}: {e}", args[1]))?)?;
    compare("CMPDUMP", &a, &b)
}

// ===========================================================================
//  六、acdump：定位"AC 区 Huffman 解码"这一层的问题
// ===========================================================================
//
// 为什么需要这一层：对拍失败时必须能区分三处
//   (a) 我们的 `huf_uncompress` 解错      → 改 `exr_dwa::huf_uncompress`
//   (b) `un_rle_ac` / `assemble` 解错      → 改 `exr_dwa`
//   (c) 官方文件自洽、是我们的期望错了     → 改 harness
// 没有这一层就只能靠猜——那正是本项目吃过亏的地方。
// 官方侧对应物是 `hufdump.cpp`（`Imf::hufUncompress`）。

/// 取文件中第一个块的负载字节（扫描线块布局）。
fn first_block_payload(data: &[u8]) -> Result<&[u8], String> {
    let header = exr::read_header(data).map_err(|e| format!("解析头: {e:?}"))?;
    let block0 = *header
        .chunk_offsets
        .first()
        .ok_or("该文件无块偏移表（不是扫描线 flat EXR？）")? as usize;
    let size = i32::from_le_bytes([
        data[block0 + 4],
        data[block0 + 5],
        data[block0 + 6],
        data[block0 + 7],
    ]) as usize;
    data.get(block0 + 8..block0 + 8 + size)
        .ok_or_else(|| "块负载越界".to_string())
}

/// 打印分块文件首个 tile 的 DWA 块头（定位 tiled 路径问题用）。
fn cmd_tileinfo(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("用法: tileinfo <in.exr>".into());
    }
    let data = fs::read(&args[0]).map_err(|e| format!("读 {}: {e}", args[0]))?;
    let header = exr::read_header(&data).map_err(|e| format!("解析头: {e:?}"))?;
    println!("TILED {}", header.tiled);
    println!("TILE {:?}", header.tile);
    println!("BLOCKS {}", header.block_count());
    println!("OFFSETS {}", header.chunk_offsets.len());
    println!("FIRST_OFFSET {}", header.chunk_offsets.first().copied().unwrap_or(0));
    Ok(())
}

fn cmd_dcdump(args: &[String]) -> Result<(), String> {
    // dcdump <in.exr> <out.txt>：解第一块的 DC 区（zlib →原生 u16 小端）。
    if args.len() < 2 {
        return Err("用法: dcdump <in.exr> <out.txt>".into());
    }
    let data = fs::read(&args[0]).map_err(|e| format!("读 {}: {e}", args[0]))?;
    let payload = first_block_payload(&data)?;
    let (s, r) = exr_dwa::block_regions(payload).map_err(|e| format!("block_regions: {e:?}"))?;
    let (off, len) = r.dc;
    let dc_buf = payload
        .get(off..off + len)
        .ok_or("DC 区越界")?;
    let mut out = String::new();
    let count = s.dc_uncompressed_count as usize;
    if s.dc_compressed_size == 0 {
        out.push_str("DC_OK 1\nN 0\n");
    } else {
        // 必须走 `exr_dwa` 的 `zip_decompress` 同一条路径（预测器 + 交织），
        // 不能自己裸 inflate —— 否则测的就不是解码器真实用的那条路了。
        let raw = exr_dwa::zip_decompress(dc_buf, count * 2).map_err(|e| format!("{e:?}"))?;
        out.push_str("DC_OK 1\n");
        out.push_str(&format!("N {count}\n"));
        for (i, c) in raw.chunks_exact(2).enumerate() {
            if i % 16 == 0 {
                out.push_str("V ");
            } else {
                out.push(' ');
            }
            out.push_str(&format!(
                "{:04x}",
                u16::from_le_bytes([c[0], c[1]])
            ));
            if i % 16 == 15 || i + 1 == count {
                out.push('\n');
            }
        }
    }
    fs::write(&args[1], &out).map_err(|e| format!("写 {}: {e}", args[1]))?;
    println!("DCDUMP ok {} -> {} ({count} 项)", args[0], args[1]);
    Ok(())
}

/// 找出文件中第一个 DWA 块负载的字节区间，并打印其块头 + AC 区定位。
fn cmd_acinfo(args: &[String]) -> Result<(), String> {
    // acinfo <in.exr>
    if args.is_empty() {
        return Err("用法: acinfo <in.exr>".into());
    }
    let data = fs::read(&args[0]).map_err(|e| format!("读 {}: {e}", args[0]))?;
    let header = exr::read_header(&data).map_err(|e| format!("解析头: {e:?}"))?;
    let block0 = *header
        .chunk_offsets
        .first()
        .ok_or("该文件无块偏移表（不是扫描线/分块 flat EXR？）")? as usize;
    // 扫描线块 = [i32 y][i32 size][payload]
    let payload_off = block0 + 8;
    let payload_size = i32::from_le_bytes([
        data[block0 + 4],
        data[block0 + 5],
        data[block0 + 6],
        data[block0 + 7],
    ]) as usize;
    let payload = data
        .get(payload_off..payload_off + payload_size)
        .ok_or("块负载越界")?;
    let (s, r) = exr_dwa::block_regions(payload).map_err(|e| format!("block_regions: {e:?}"))?;
    println!("BLOCK0 {block0} {payload_size}");
    println!("VERSION {}", s.version);
    println!(
        "UNKNOWN {} {}",
        s.unknown_uncompressed_size, s.unknown_compressed_size
    );
    println!("AC {} {}", s.ac_compressed_size, s.ac_uncompressed_count);
    println!("DC {} {}", s.dc_compressed_size, s.dc_uncompressed_count);
    println!(
        "RLE {} {} {}",
        s.rle_compressed_size, s.rle_uncompressed_size, s.rle_raw_size
    );
    println!("AC_COMPRESSION {}", s.ac_compression);
    println!("RULES_LEN {}", r.rules_len);
    println!("AC_OFF {}", r.ac.0);
    println!("AC_LEN {}", r.ac.1);
    Ok(())
}

fn cmd_acdump(args: &[String]) -> Result<(), String> {
    // acdump <in.exr> <out.txt>：用我们的 huf_uncompress 解第一块的 AC 区。
    if args.len() < 2 {
        return Err("用法: acdump <in.exr> <out.txt>".into());
    }
    let data = fs::read(&args[0]).map_err(|e| format!("读 {}: {e}", args[0]))?;
    let header = exr::read_header(&data).map_err(|e| format!("解析头: {e:?}"))?;
    let block0 = *header
        .chunk_offsets
        .first()
        .ok_or("该文件无块偏移表")? as usize;
    let payload_off = block0 + 8;
    let payload_size = i32::from_le_bytes([
        data[block0 + 4],
        data[block0 + 5],
        data[block0 + 6],
        data[block0 + 7],
    ]) as usize;
    let payload = &data[payload_off..payload_off + payload_size];
    let (s, r) = exr_dwa::block_regions(payload).map_err(|e| format!("block_regions: {e:?}"))?;
    let (ac_off, ac_len) = r.ac;
    let ac_buf = payload
        .get(ac_off..ac_off + ac_len)
        .ok_or("AC 区越界")?;
    let ac = exr_dwa::huf_uncompress(ac_buf, s.ac_uncompressed_count as usize)
        .map_err(|e| format!("huf_uncompress: {e:?}"))?;
    let mut out = String::new();
    out.push_str("HUF_OK 1\n");
    out.push_str(&format!("N {}\n", ac.len()));
    for (i, v) in ac.iter().enumerate() {
        if i % 16 == 0 {
            out.push_str("V ");
        } else {
            out.push(' ');
        }
        out.push_str(&format!("{v:04x}"));
        if i % 16 == 15 || i + 1 == ac.len() {
            out.push('\n');
        }
    }
    fs::write(&args[1], &out).map_err(|e| format!("写 {}: {e}", args[1]))?;
    println!(
        "ACDUMP ok {} -> {} ({} 项)",
        args[0],
        args[1],
        ac.len()
    );
    Ok(())
}

// ===========================================================================
//  main
// ===========================================================================

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!(
            "usage: exr_interop <gen|dump|read|srccmp|cmpdump|acinfo|acdump> ...\n  \
             gen    <out.exr> <w> <h> <spec> <src.txt>\n  \
             dump   <in.exr> <out.txt>\n  \
             read   <in.exr>\n  \
             srccmp <in.exr> <src.txt>\n  \
             cmpdump <a.txt> <b.txt>\n  \
             acinfo <in.exr>\n  \
             acdump <in.exr> <out.txt>"
        );
        process::exit(2);
    }
    let cmd = argv[0].as_str();
    let rest = &argv[1..];
    let result = match cmd {
        "gen" => cmd_gen(rest),
        "dump" => cmd_dump(rest),
        "read" => cmd_read(rest),
        "srccmp" => cmd_srccmp(rest),
        "cmpdump" => cmd_cmpdump(rest),
        "acinfo" => cmd_acinfo(rest),
        "dcdump" => cmd_dcdump(rest),
        "tileinfo" => cmd_tileinfo(rest),
        "acdump" => cmd_acdump(rest),
        other => Err(format!("未知子命令 {other:?}")),
    };
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("FAIL {e}");
            process::exit(1);
        }
    }
}
