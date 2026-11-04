//! V2.5-B4 / 矩阵 G 补充（MaterialX）：覆盖度判据表 + 逐节点 BSDF golden。
//!
//! | # | 判据 | 本文件 |
//! |---|---|---|
//! | V1 | 覆盖度表给具体数字（21 种 / 17 标量槽 / 别名数 / 编译通过数） | `v1_*` |
//! | V2 | 每个已支持节点一条 golden 对拍（节点图 → BSDF 参数 ≤1e-6） | `v2_*` |
//! | V3 | ≥2 负控（未知节点 / 循环图 / 类型错配） | `v3_*` |
//! | V4 | 双跑逐字节 | `v4_*` |
//! | V5 | 产品路径接线断言 + 静态门禁 | `v5_*` |

use std::fs;

use prism_io::materialx::{
    self, compile_to_bsdf, coverage_report, coverage_table, BsdfParams, MtlxError, NodeKind,
};

/// 覆盖度判据表的 21 种节点（nodedef 原文，逐个编译）。
const NODEDEFS: &[&str] = &[
    "standard_surface",
    "bsdf",
    "diffuse",
    "glossy",
    "mix",
    "add",
    "multiply",
    "ramp",
    "constant",
    "position",
    "normal",
    "image",
    "dot",
    "burley_diffuse",
    "oren_nayar_diffuse",
    "conductor",
    "dielectric",
    "sheen",
    "subsurface",
    "transmission",
    "thinfilm",
];

/// 每个节点都挂同一组常量输入（golden 数字由此独立给出）。
const INPUTS: &str = r#"
          <input name="base_color" type="color3" value="0.2, 0.4, 0.6"/>
          <input name="color" type="color3" value="0.3, 0.5, 0.7"/>
          <input name="roughness" type="float" value="0.3"/>
          <input name="metallic" type="float" value="0.7"/>
          <input name="specular" type="float" value="0.4"/>
          <input name="transmission" type="float" value="0.2"/>
          <input name="ior" type="float" value="1.45"/>
          <input name="emission" type="color3" value="0.1, 0.2, 0.3"/>
          <input name="coat" type="float" value="0.5"/>
          <input name="sheen" type="float" value="0.6"/>
          <input name="subsurface" type="float" value="0.25"/>
          <input name="diffuse_roughness" type="float" value="0.35"/>
          <input name="opacity" type="float" value="0.9"/>
          <input name="weight" type="float" value="0.8"/>
          <input name="thickness" type="float" value="400"/>
          <input name="fg" type="color3" value="0.1, 0.1, 0.1"/>
          <input name="bg" type="color3" value="0.9, 0.9, 0.9"/>
          <input name="in1" type="color3" value="0.25, 0.5, 0.75"/>
          <input name="in2" type="color3" value="0.5, 0.25, 0.125"/>
          <input name="value" type="color3" value="0.6, 0.7, 0.8"/>
          <input name="colors" type="color3[]" value="0.11, 0.22, 0.33, 0.44, 0.55, 0.66"/>
          <output name="out"/>
"#;

fn golden_doc() -> String {
    let mut s = String::from(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<materialx version="1.38">
  <materyialgraph>
    <material name="m">
      <nodegraph>
"#,
    );
    for (i, nd) in NODEDEFS.iter().enumerate() {
        s.push_str(&format!(
            "        <node name=\"n{i:02}\" type=\"{nd}\">\n{INPUTS}        </node>\n"
        ));
    }
    s.push_str(
        r#"        <output name="surface" type="material">
          <connectfrom node="n00" output="out"/>
        </output>
      </nodegraph>
    </material>
  </materyialgraph>
</materialx>
"#,
    );
    s
}

/// 与生产映射**独立**写死的 golden（数值来自 INPUTS 字面量，不调用被测函数）。
fn golden_params(kind: NodeKind) -> BsdfParams {
    let mut p = BsdfParams::default();
    let col: [f32; 3] = [0.2, 0.4, 0.6];
    match kind {
        NodeKind::StandardSurface => {
            p.base_color = col;
            p.metallic = 0.7;
            p.roughness = 0.3;
            p.specular = 0.4;
            p.transmission = 0.2;
            p.ior = 1.45;
            p.emission = [0.1, 0.2, 0.3];
            p.coat = 0.5;
            p.sheen = 0.6;
            p.subsurface = 0.25;
            p.diffuse_roughness = 0.35;
            p.opacity = 0.9;
        }
        NodeKind::Bsdf => {
            p.base_color = col;
            p.metallic = 0.7;
            p.roughness = 0.3;
            p.specular = 0.4;
            p.opacity = 0.9;
        }
        NodeKind::Diffuse | NodeKind::BurleyDiffuse | NodeKind::OrenNayarDiffuse => {
            p.base_color = col;
            p.diffuse_roughness = 0.3;
            p.opacity = 0.9;
            p.metallic = 0.0;
            p.specular = 0.0;
        }
        NodeKind::Glossy | NodeKind::Dielectric => {
            p.base_color = col;
            p.roughness = 0.3;
            p.specular = 0.4;
            p.ior = 1.45;
        }
        NodeKind::Mix => {
            p.base_color = [0.74, 0.74, 0.74]; // 0.1 + (0.9 - 0.1) * 0.8
            p.opacity = 0.8;
        }
        NodeKind::Add => p.base_color = [0.75, 0.75, 0.875],
        NodeKind::Multiply => {
            p.base_color = [0.125, 0.125, 0.09375];
            p.roughness = 0.8;
        }
        NodeKind::Ramp => p.base_color = [0.11, 0.22, 0.33],
        NodeKind::Constant => p.base_color = [0.6, 0.7, 0.8],
        NodeKind::Position => p.base_color = [0.5, 0.5, 0.5],
        NodeKind::Normal => p.base_color = [0.5, 0.5, 1.0],
        NodeKind::Image => {
            p.base_color = [1.0, 1.0, 1.0];
            p.opacity = 0.9;
        }
        NodeKind::Dot => p.roughness = 0.25,
        NodeKind::Conductor => {
            p.base_color = col;
            p.roughness = 0.3;
            p.specular = 0.4;
            p.ior = 1.45;
            p.metallic = 0.7;
        }
        NodeKind::Transmission => {
            p.base_color = col;
            p.roughness = 0.3;
            p.specular = 0.4;
            p.ior = 1.45;
            p.transmission = 0.8;
        }
        NodeKind::Sheen => {
            p.base_color = col;
            p.sheen = 0.8;
            p.roughness = 0.3;
        }
        NodeKind::Subsurface => {
            p.base_color = col;
            p.subsurface = 0.8;
            p.diffuse_roughness = 0.3;
            p.ior = 1.45;
        }
        NodeKind::ThinFilm => {
            p.thinfilm_thickness = 400.0;
            p.ior = 1.45;
            p.specular = 0.4;
        }
    }
    p
}

fn assert_params_close(got: BsdfParams, want: BsdfParams, label: &str) {
    let mut worst: f32 = 0.0;
    let pairs: [(f32, f32); 17] = [
        (got.base_color[0], want.base_color[0]),
        (got.base_color[1], want.base_color[1]),
        (got.base_color[2], want.base_color[2]),
        (got.metallic, want.metallic),
        (got.roughness, want.roughness),
        (got.specular, want.specular),
        (got.transmission, want.transmission),
        (got.ior, want.ior),
        (got.emission[0], want.emission[0]),
        (got.emission[1], want.emission[1]),
        (got.emission[2], want.emission[2]),
        (got.coat, want.coat),
        (got.sheen, want.sheen),
        (got.subsurface, want.subsurface),
        (got.diffuse_roughness, want.diffuse_roughness),
        (got.thinfilm_thickness, want.thinfilm_thickness),
        (got.opacity, want.opacity),
    ];
    for (a, b) in pairs {
        worst = worst.max((a - b).abs());
    }
    assert!(worst <= 1e-6, "{label} BSDF 参数偏差 {worst} > 1e-6：got={got:?} want={want:?}");
}

fn source(path: &str) -> String {
    fs::read_to_string(format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("静态门禁读不到 {path}：{e}"))
}

// ---------------------------------------------------------------------------
// V1：覆盖度表给具体数字
// ---------------------------------------------------------------------------

#[test]
fn v1_coverage_table_has_concrete_numbers() {
    let table = coverage_table();
    assert_eq!(table.len(), 21, "支持的节点类型总数 = 21（13 既有 + 8 补齐）");
    let aliases: usize = table.iter().map(|c| c.aliases.len()).sum();
    assert_eq!(aliases, 45, "别名总数（逐条钉死）");
    let mapped: usize = table.iter().map(|c| c.bsdf_params.len()).sum();
    assert!(mapped >= 21, "每个节点至少写 1 个 BSDF 参数，总映射条数 {mapped}");
    for (i, c) in table.iter().enumerate() {
        assert_eq!(c.structure, c.kind.structure(), "第 {i} 行结构位与 NodeKind 不一致");
        assert!(!c.aliases.is_empty(), "第 {i} 行别名不能为空");
        assert!(!c.bsdf_params.is_empty(), "第 {i} 行必须有节点→BSDF 参数映射");
        // 表内清单不发散：不允许 include / nodedef / function 之类。
        assert!(!c.aliases.iter().any(|a| a.contains("include") || a.contains("function")));
    }
    let pack = materialx::compile_str(&golden_doc()).unwrap();
    let report = coverage_report(&pack);
    assert_eq!(report.node_kinds, 21);
    assert_eq!(report.aliases, 45);
    assert_eq!(report.bsdf_param_slots, 17);
    assert_eq!(report.compiled_nodes, 21, "编译通过 21 个节点");
    assert!(report.unsupported_behavior.contains("MX2004"));
    println!(
        "V1 覆盖度：{} 种节点 / {} 别名 / {} BSDF 标量槽 / 编译通过 {} 节点 / 映射 {} 条",
        report.node_kinds, aliases, report.bsdf_param_slots, report.compiled_nodes, mapped
    );
}

// ---------------------------------------------------------------------------
// V2：每个已支持节点一条 golden 对拍
// ---------------------------------------------------------------------------

#[test]
fn v2_every_supported_node_has_a_bsdf_golden() {
    let pack = materialx::compile_str(&golden_doc()).unwrap();
    assert_eq!(pack.len(), 1);
    let graph = &pack.material("m").unwrap().graph;
    assert_eq!(graph.node_count(), 21);
    let bsdf = compile_to_bsdf(&pack).unwrap();
    assert_eq!(bsdf.len(), 21, "每个节点一条 BSDF 参数");
    for (i, nd) in NODEDEFS.iter().enumerate() {
        let name: String = format!("n{i:02}");
        let node = graph.nodes.get(&name).unwrap_or_else(|| panic!("缺节点 {name}"));
        let kind: NodeKind = NodeKind::from_mtlx_type(nd).unwrap_or_else(|| panic!("{nd} 未注册"));
        assert_eq!(node.structure, kind.structure(), "{nd} 结构位");
        let got: BsdfParams = materialx::node_to_bsdf(node).unwrap();
        assert_params_close(got, golden_params(kind), &format!("{nd}（{name}）"));
        // compile_to_bsdf 的同一条也对拍（按名升序）。
        let entry = bsdf.iter().find(|(k, _)| k == &format!("m/{name}")).unwrap();
        assert_eq!(entry.1, got, "{name} compile_to_bsdf 与 node_to_bsdf 逐位一致");
    }
    println!("V2：21 个节点 × 17 个标量槽 golden 全部 ≤1e-6");
}

// ---------------------------------------------------------------------------
// V3：≥2 负控
// ---------------------------------------------------------------------------

#[test]
fn v3_unknown_node_is_structured_error() {
    let src = golden_doc().replace("type=\"dot\"", "type=\"volumetric_scatter\"");
    let e: MtlxError = materialx::compile_str(&src).unwrap_err();
    assert_eq!(e.code, "MX2004");
    assert!(e.problem.contains("volumetric_scatter"));
    assert!(!e.fix.is_empty());
}

#[test]
fn v3_cyclic_graph_is_structured_error() {
    let src = r#"<materialx version="1.0"><materyialgraph><material name="m">
<nodegraph>
  <node name="a" type="add"><input name="in1" type="float" interconnect="b.out"/><output name="out"/></node>
  <node name="b" type="add"><input name="in1" type="float" interconnect="a.out"/><output name="out"/></node>
</nodegraph></material></materyialgraph></materialx>"#;
    let e = materialx::compile_str(src).unwrap_err();
    assert_eq!(e.code, "MX2007");
}

#[test]
fn v3_type_mismatch_is_structured_error() {
    let src = golden_doc().replace(
        r#"name="base_color" type="color3" value="0.2, 0.4, 0.6""#,
        r#"name="base_color" type="color3" value="0.2, 0.4""#,
    );
    let e = materialx::compile_str(&src).unwrap_err();
    assert_eq!(e.code, "MX2005");
}

// ---------------------------------------------------------------------------
// V4：双跑逐字节
// ---------------------------------------------------------------------------

#[test]
fn v4_double_run_is_byte_identical() {
    let doc = golden_doc();
    let a = materialx::compile_str(&doc).unwrap();
    let b = materialx::compile_str(&doc).unwrap();
    assert_eq!(a, b, "编译两跑逐字段一致");
    let pa = compile_to_bsdf(&a).unwrap();
    let pb = compile_to_bsdf(&b).unwrap();
    assert_eq!(pa, pb, "BSDF 参数两跑逐位一致");
    let sa: Vec<u8> = format!("{pa:?}").into_bytes();
    let sb: Vec<u8> = format!("{pb:?}").into_bytes();
    assert_eq!(sa, sb, "序列化文本两跑逐字节一致");
    let sig: u64 = pa.iter().fold(0xcbf2_9ce4_8422_2325u64, |mut h, (_, p)| {
        h ^= p.signature();
        h.wrapping_mul(0x0000_0100_0000_01b3)
    });
    let sig2: u64 = pb.iter().fold(0xcbf2_9ce4_8422_2325u64, |mut h, (_, p)| {
        h ^= p.signature();
        h.wrapping_mul(0x0000_0100_0000_01b3)
    });
    assert_eq!(sig, sig2);
    println!("V4 双跑：21 条参数，签名 {sig:#018x} 逐字节一致");
}

// ---------------------------------------------------------------------------
// V5：产品路径接线断言 + 静态门禁
// ---------------------------------------------------------------------------

#[test]
fn v5_product_path_compiles_file_and_maps_bsdf() {
    let dir = std::env::temp_dir().join("prism_v25_b4_product");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let p = dir.join("golden.mtlx");
    fs::write(&p, golden_doc()).unwrap();
    let pack = materialx::compile_file(p.to_str().unwrap()).unwrap();
    assert_eq!(pack.len(), 1);
    let report = coverage_report(&pack);
    assert_eq!(report.compiled_nodes, 21);
    let bsdf = compile_to_bsdf(&pack).unwrap();
    assert_eq!(bsdf.len(), 21, "产品入口产出的 BSDF 映射非空");
}

#[test]
fn v5_static_gate_product_path_is_wired() {
    let main = source("crates/prism-app/src/main.rs");
    assert!(main.contains("\"mtlx\" => Some(cmd_mtlx::run(rest))"), "CLI 必须分发 mtlx 子命令");
    assert!(main.contains("prism-app mtlx <FILE>"), "用法文本必须列 mtlx 子命令");
    let cmd = source("crates/prism-app/src/cmd_mtlx.rs");
    assert!(cmd.contains("materialx::compile_mtlx_report"), "产品命令必须调库侧统一出口");
    assert!(!cmd.to_lowercase().contains("bsdf"), "应用层必须薄转发、零散射词汇（既有 w7 门禁）");
    let mtlx = source("crates/prism-io/src/materialx.rs");
    assert!(mtlx.contains("pub const NODE_COVERAGE"), "覆盖度表必须是库的一等公民");
    assert!(mtlx.contains("pub fn compile_to_bsdf"), "参数映射必须导出给产品路径");
    assert!(mtlx.contains("pub fn compile_mtlx_report"), "统一出口必须公开");
    assert!(mtlx.contains("compile_file(path)"), "统一出口必须真调编译入口");
    assert!(mtlx.contains("coverage_report(&pack)"), "统一出口必须真消费覆盖度报告");
    assert!(mtlx.contains("compile_to_bsdf(&pack)"), "统一出口必须真消费参数映射");
}
