//! V2.5-B2 / 矩阵 G 补充（USD）：真实 USD 写出的 `.usdc` / `.usdz` 全判据。
//!
//! golden 夹具由 `tests/usdc_golden_gen.py` 用 **Pixar OpenUSD（usd-core）**
//! 生成：`cube.usdc` 是 `Sdf.Layer.Export` 的真二进制 Crate（0.8.0），
//! `cube.usdz` 是 `UsdUtils.CreateNewUsdzPackage` 的真实 USDZ 包。
//! 本仓解析器只读这些字节，不与写端共用实现 ⇒ 不构成「两端同错」（§1.4）。
//!
//! | # | 判据 | 本文件 |
//! |---|---|---|
//! | V1 | `.usdc` vs `.usda` 逐字段语义对拍 + vs 真实 USD 字段 golden | `v1_*` |
//! | V2 | USDZ 解包后与裸 `.usdc` 逐位（stored + deflate 两种） | `v2_*` |
//! | V3 | 双跑逐字节 | `v3_*` |
//! | V4 | ≥2 负控（截断 / 错 TOC / 错压缩段 / 坏 zip） | `v4_*` |
//! | V5 | 产品路径非空断言 + 静态门禁 | `v5_*` |
//! | V6 | 既有 USDA 解析测试行为不回退 | `v6_*` |

use std::fs;

use prism_io::usd::{self, AttrValue, Layer, ParsedUsda, PrimKind};
use prism_io::usdc;
use prism_io::usdz;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/usdc");

fn fixture(name: &str) -> Vec<u8> {
    fs::read(format!("{FIX}/{name}"))
        .unwrap_or_else(|e| panic!("读不到 golden 夹具 {FIX}/{name}：{e}"))
}

fn source(path: &str) -> String {
    fs::read_to_string(format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("静态门禁读不到 {path}：{e}"))
}

fn fmt_num(v: f64) -> String {
    if v.fract().abs() < 1e-12 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// AttrValue → 与 `cube.usdc.fields.txt`（pxr 读回真值）同口径的稳定文本。
fn fmt_value(v: &AttrValue) -> String {
    match v {
        AttrValue::None => "None".to_string(),
        AttrValue::Number(n) => fmt_num(*n),
        AttrValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        AttrValue::Str(s) | AttrValue::Asset(s) | AttrValue::Path(s) => s.clone(),
        AttrValue::Numbers(ns) => format!(
            "[{}]",
            ns.iter().map(|n| fmt_num(*n)).collect::<Vec<_>>().join(", ")
        ),
        AttrValue::Rows(rows) | AttrValue::Matrix(rows) => format!(
            "[{}]",
            rows.iter()
                .map(|r| format!(
                    "[{}]",
                    r.iter().map(|n| fmt_num(*n)).collect::<Vec<_>>().join(", ")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        AttrValue::Strings(v) | AttrValue::Paths(v) => format!("[{}]", v.join(", ")),
        AttrValue::EmptyArray => "[]".to_string(),
    }
}

fn kind_suffix(k: PrimKind) -> &'static str {
    match k {
        PrimKind::Def => "Def",
        PrimKind::Over => "Over",
        PrimKind::Class => "Class",
    }
}

/// Layer → 逐字段 canonical dump（顺序与 golden 完全一致：层元数据按键排序、
/// 每 prim 先属性/关系再子 prim，属性/关系按名排序）。
fn canonical(layer: &Layer) -> Vec<String> {
    fn walk(p: &usd::Prim, out: &mut Vec<String>) {
        out.push(format!(
            "PRIM {} specifier=Sdf.Specifier{} typeName={}",
            p.path,
            kind_suffix(p.kind),
            p.type_name
        ));
        let mut attrs = p.attributes.clone();
        attrs.sort_by(|a, b| a.name.cmp(&b.name));
        for a in attrs {
            out.push(format!(
                "ATTR {} type={} custom={} uniform={} default={}",
                a.name,
                a.type_name,
                a.custom,
                a.uniform,
                fmt_value(&a.value)
            ));
        }
        let mut rels = p.relationships.clone();
        rels.sort_by(|a, b| a.name.cmp(&b.name));
        for r in rels {
            out.push(format!("REL {} targets=[{}]", r.name, r.targets.join(", ")));
        }
        for c in &p.children {
            walk(c, out);
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut md: Vec<(String, String)> = layer.metadata.clone();
    md.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, v) in md {
        out.push(format!("LAYER {k} = {v}"));
    }
    for p in &layer.prims {
        walk(p, &mut out);
    }
    out
}

fn usda_layer() -> Layer {
    let text: String = String::from_utf8(fixture("cube.usda")).unwrap();
    usd::parse_str(&text).expect("源 USDA 应可解析").layer
}

fn usdc_layer() -> Layer {
    usd::parse_bytes(&fixture("cube.usdc"))
        .expect("真实 USD 写出的 .usdc 必须真解析成功")
        .layer
}

// ---------------------------------------------------------------------------
// V1：.usdc ↔ .usda 逐字段语义对拍 + 真实 USD golden
// ---------------------------------------------------------------------------

#[test]
fn v1_usdc_matches_usda_field_by_field() {
    let a: Layer = usda_layer();
    let b: Layer = usdc_layer();
    assert_eq!(a, b, "两种容器的 Layer 必须逐字段相等（prim/属性/变换/绑定/元数据）");
    // 可渲染场景签名（sha256）逐位相等：prim 层级 + 网格 + 变换 + 材质绑定全覆盖。
    let sa = usd::representation_signature(&usd::build_scene(&a).unwrap());
    let sb = usd::representation_signature(&usd::build_scene(&b).unwrap());
    assert_eq!(sa, sb, "representation_signature 必须逐位相等");
}

#[test]
fn v1_usdc_matches_pxr_golden_dump() {
    let golden: String = String::from_utf8(fixture("cube.usdc.fields.txt")).unwrap();
    let want: Vec<String> = golden.lines().map(str::to_string).collect();
    let got: Vec<String> = canonical(&usdc_layer());
    assert_eq!(got, want, "解析结果必须与真实 USD 读回的逐字段 golden 一致");
    // 数字证据：3 LAYER + 3 PRIM + 5 ATTR + 1 REL = 12 行。
    assert_eq!(want.len(), 12);
    let attrs: usize = want.iter().filter(|l| l.starts_with("ATTR ")).count();
    let rels: usize = want.iter().filter(|l| l.starts_with("REL ")).count();
    let prims: usize = want.iter().filter(|l| l.starts_with("PRIM ")).count();
    assert_eq!((attrs, rels, prims), (5, 1, 3));
    println!("V1 golden：{prims} prim / {attrs} 属性 / {rels} 关系逐字段相等");
}

#[test]
fn v1_crate_inspection_reports_real_structure() {
    let info: usdc::CrateInfo = usdc::inspect(&fixture("cube.usdc")).unwrap();
    assert_eq!(info.version, "0.8.0");
    assert_eq!(info.sections.len(), 6, "TOKENS/STRINGS/FIELDS/FIELDSETS/PATHS/SPECS");
    assert_eq!(info.token_count, 30);
    assert_eq!(info.field_count, 24);
    assert_eq!(info.spec_count, 10);
    let names: Vec<&str> = info.sections.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["TOKENS", "STRINGS", "FIELDS", "FIELDSETS", "PATHS", "SPECS"]
    );
}

// ---------------------------------------------------------------------------
// V2：USDZ 解包后与裸 .usdc 逐位（stored / deflate 两路）
// ---------------------------------------------------------------------------

#[test]
fn v2_usdz_root_is_bitwise_identical_to_bare_usdc() {
    let bare: Vec<u8> = fixture("cube.usdc");
    for name in ["cube.usdz", "cube_deflate.usdz"] {
        let (root, payload, names) = usdz::extract_root(&fixture(name))
            .unwrap_or_else(|e| panic!("{name} 解包失败：{e}"));
        assert_eq!(root, "cube.usdc", "{name} 根层名");
        assert_eq!(payload, bare, "{name} 解包后的根层字节必须与裸 .usdc 逐位相同");
        assert_eq!(names.first().map(String::as_str), Some("cube.usdc"));
    }
}

#[test]
fn v2_usdz_parses_to_same_semantics_as_bare_usdc() {
    let bare: Layer = usdc_layer();
    let sig_bare = usd::representation_signature(&usd::build_scene(&bare).unwrap());
    for name in ["cube.usdz", "cube_deflate.usdz"] {
        let parsed: ParsedUsda = usd::parse_bytes(&fixture(name))
            .unwrap_or_else(|e| panic!("{name} 必须真解析成功：{e}"));
        assert_eq!(parsed.layer, bare, "{name} 解析出的 Layer 必须逐字段相等");
        let sig = usd::representation_signature(&usd::build_scene(&parsed.layer).unwrap());
        assert_eq!(sig, sig_bare, "{name} 场景签名必须逐位相等");
    }
}

// ---------------------------------------------------------------------------
// V3：双跑逐字节
// ---------------------------------------------------------------------------

#[test]
fn v3_double_run_is_byte_identical() {
    let bytes: Vec<u8> = fixture("cube.usdc");
    let a: Vec<String> = canonical(&usd::parse_bytes(&bytes).unwrap().layer);
    let b: Vec<String> = canonical(&usd::parse_bytes(&bytes).unwrap().layer);
    assert_eq!(a, b, "同一 .usdc 两跑逐字段一致");
    let ia = usdc::inspect(&bytes).unwrap();
    let ib = usdc::inspect(&bytes).unwrap();
    assert_eq!(ia, ib);
    // USDZ 两跑
    let z: Vec<u8> = fixture("cube.usdz");
    let za = canonical(&usd::parse_bytes(&z).unwrap().layer);
    let zb = canonical(&usd::parse_bytes(&z).unwrap().layer);
    assert_eq!(za, zb);
    assert_eq!(za, a, "USDZ 与裸 usdc 的 canonical dump 逐字节一致");
}

// ---------------------------------------------------------------------------
// V4：≥2 负控（截断 / 错 TOC / 错压缩段 / 坏 zip）→ 结构化报错
// ---------------------------------------------------------------------------

fn assert_structured(err: usd::UsdError, expected_codes: &[&str]) {
    assert!(err.code.starts_with("PU1"), "错误码应是 PU1xxx：{err}");
    assert!(!err.location.is_empty() && !err.problem.is_empty() && !err.fix.is_empty());
    assert!(
        expected_codes.contains(&err.code),
        "期望 {expected_codes:?}，实际 {}：{err}",
        err.code
    );
}

#[test]
fn v4_truncated_crate_is_rejected() {
    let mut b: Vec<u8> = fixture("cube.usdc");
    b.truncate(200); // 保留 bootstrap + 部分段，砍掉 TOC
    let e = usd::parse_bytes(&b).expect_err("截断的 crate 必须报错");
    assert_structured(e, &["PU1011", "PU1012"]);
}

#[test]
fn v4_bad_toc_is_rejected() {
    let mut b: Vec<u8> = fixture("cube.usdc");
    // bootstrap 的 tocOffset 在字节 16..24：改成远超文件长度的值。
    b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let e = usd::parse_bytes(&b).expect_err("错 TOC 必须报错");
    assert_structured(e, &["PU1012"]);
}

#[test]
fn v4_bad_compressed_section_is_rejected() {
    let mut b: Vec<u8> = fixture("cube.usdc");
    // TOKENS 段起始 364：破坏压缩块的 token / match offset（第 372 字节起）。
    b[372] = 0xff;
    b[373] = 0xff;
    b[374] = 0xff;
    b[375] = 0xff;
    let e = usd::parse_bytes(&b).expect_err("错压缩段必须报错");
    assert_structured(e, &["PU1013", "PU1011", "PU1012", "PU1014"]);
}

#[test]
fn v4_bad_zip_is_rejected() {
    let mut b: Vec<u8> = fixture("cube.usdz");
    b.truncate(60); // 砍掉 EOCD 与中央目录
    let e = usd::parse_bytes(&b).expect_err("坏 zip 必须报错");
    assert_structured(e, &["PU1016"]);
}

#[test]
fn v4_magic_only_blob_keeps_pu1001() {
    // 既有判据原文不变：`PXR-USDC` + 64 个 0（长度 < 88 B bootstrap）
    // 仍是真非法输入，保留 PU1001 结构化拒绝。
    let mut bytes = b"PXR-USDC".to_vec();
    bytes.extend_from_slice(&[0u8; 64]);
    let e = usd::parse_bytes(&bytes).expect_err("真非法 crate 输入必须拒绝");
    assert_eq!(e.code, "PU1001");
    assert!(e.fix.contains("usdcat"));
}

// ---------------------------------------------------------------------------
// V5：产品路径非空断言 + 静态门禁
// ---------------------------------------------------------------------------

#[test]
fn v5_product_asset_api_reads_all_three_containers() {
    // 统一产品入口 `parse_asset` 按扩展名 + magic 分派；三种容器都非空。
    let dir = std::env::temp_dir().join("prism_v25_b2_product");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for name in ["cube.usda", "cube.usdc", "cube.usdz"] {
        let p = dir.join(name);
        fs::write(&p, fixture(name)).unwrap();
        let parsed: ParsedUsda = usd::parse_asset(p.to_str().unwrap())
            .unwrap_or_else(|e| panic!("产品入口读 {name} 失败：{e}"));
        assert!(!parsed.layer.prims.is_empty(), "{name} 解析结果非空");
        let scene = usd::build_scene(&parsed.layer).unwrap();
        assert_eq!(scene.mesh_count(), 1, "{name} 应有 1 个可渲染网格");
        assert_eq!(scene.render_triangle_count(), 12, "{name} 应三角化成 12 个三角形");
    }
}

#[test]
fn v5_static_gate_product_path_is_wired() {
    let main = source("crates/prism-app/src/main.rs");
    assert!(main.contains("\"usd\" => Some(cmd_usd::run(rest))"), "CLI 必须分发 usd 子命令");
    assert!(main.contains("prism-app usd <FILE>"), "用法文本必须列 usd 子命令");
    let cmd = source("crates/prism-app/src/cmd_usd.rs");
    assert!(cmd.contains("prism_io::usd::parse_asset"), "产品命令必须调统一入口");
    assert!(cmd.contains("build_scene"), "产品命令必须构建可渲染场景");
    assert!(cmd.contains(".usdc") && cmd.contains(".usdz"), "产品命令必须显式支持二进制容器");
    let parse = source("crates/prism-io/src/usd.rs");
    assert!(parse.contains("crate::usdc::parse_crate"), "parse_bytes 必须分派到 usdc");
    assert!(parse.contains("crate::usdz::parse_usdz"), "parse_bytes 必须分派到 usdz");
    assert!(parse.contains("pub fn parse_asset"), "统一资产入口必须存在");
    let lz4 = source("crates/prism-io/src/usdz.rs");
    assert!(lz4.contains("prism_bytes::inflate_raw"), "USDZ 必须复用 prism-bytes 的 inflate");
}

// ---------------------------------------------------------------------------
// V6：既有 USDA 解析行为不回退
// ---------------------------------------------------------------------------

#[test]
fn v6_usda_parser_behavior_is_unchanged() {
    let text: String = String::from_utf8(fixture("cube.usda")).unwrap();
    let p: ParsedUsda = usd::parse_str(&text).unwrap();
    assert_eq!(p.layer.version, "1");
    assert_eq!(p.layer.prims.len(), 1);
    assert_eq!(p.layer.prims[0].children.len(), 2);
    // 既有错误码口径不回退。
    let e = usd::parse_str("#usda 1.0\ndef Mesh \"m\" {").unwrap_err();
    assert_eq!(e.code, "PU1002");
    let e = usd::parse_bytes(&[0xff, 0xfe, 0x00]).unwrap_err();
    assert_eq!(e.code, "PU1009");
    // 文本里以 magic 开头 → PU1001 预检仍在。
    let e = usd::parse_str("PXR-USDCxxxx").unwrap_err();
    assert_eq!(e.code, "PU1001");
    // 未知扩展名 → 结构化拒绝（新增门面，不影响旧调用点）。
    let e = usd::parse_asset("/tmp/definitely.notusd").unwrap_err();
    assert_eq!(e.code, "PU1018");
}

#[test]
fn v1_path_table_reconstructs_ten_sdf_paths() {
    // 压缩路径树（jumps 方案）重建出的 10 条 SdfPath，逐条与真实 USD 的
    // path 表对拍（prim 层级 + prim property path 全覆盖）。
    let got: Vec<String> = usdc::path_table(&fixture("cube.usdc")).unwrap();
    assert_eq!(
        got,
        vec![
            "/",                              // 0: absolute root
            "/root",                          // 1
            "/root/cube",                     // 2
            "/root/mat",                      // 3
            "/root/cube.faceVertexCounts",    // 4
            "/root/cube.faceVertexIndices",   // 5
            "/root/cube.material:binding",    // 6
            "/root/cube.points",              // 7
            "/root.xformOp:translate",        // 8
            "/root.xformOpOrder",             // 9
        ]
    );
    assert!(usdc::crate_diag(&fixture("cube.usdc")).unwrap().contains("paths=["));
}
