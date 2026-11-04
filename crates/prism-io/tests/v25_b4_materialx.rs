//! V2.5-B4 / 矩阵 G 补充（MaterialX）：节点覆盖度判据全表。
//!
//! 判据表（**数字**）：
//!
//! | 项 | 数字 |
//! |---|---|
//! | 支持节点种类 | 21 |
//! | nodedef 别名 | 45 |
//! | `BsdfParams` 命名域 | 13 |
//! | `BsdfParams` 标量槽（`signature()` 混入库） | 17 |
//! | 本轮补齐的常见标准节点 | 8 |
//! | 每节点 golden 对拍 | 21 条（逐字段 + 逐位签名） |
//! | 别名归一化对拍 | 45 条 |
//! | 负控 | 6 类（MX2001/2004/2005/2006/2007） |
//! | 产品路径接线断言 | 库侧静态门禁 + `prism-app` CLI 实跑 |
//!
//! golden 值的来源是**独立推导**：`NodeKind` → `BsdfParams` 映射按
//! [`prism_io::materialx::NODE_COVERAGE`] 的字段清单逐项算出，写死在
//! [`GOLDEN`] 里；测试不能调用被测 `signature()` 来「自己对自己」。
//! 每种节点单独编译一份 `.mtlx`（不是一张大图里挑一行），避免节点间
//! 输入串味造成假绿。

use std::fs;

use prism_io::materialx::{
    self, BsdfParams, CoverageReport, MtlxError, NodeCoverage, NodeKind, NODE_COVERAGE,
};

/// 单节点 `.mtlx` 模板：一次给全 20 个输入端口，让每个节点映射分支
/// 都拿到确定值（未使用的端口自然被忽略），并连到材质输出。
fn single_node_mtlx(ty: &str) -> String {
    format!(
        r#"<materialx version="1.0"><materyialgraph><material name="m"><nodegraph>
<node name="n" type="{ty}">
 <input name="base_color" type="color3" value="0.2, 0.4, 0.6"/>
 <input name="roughness" type="float" value="0.3"/>
 <input name="metallic" type="float" value="0.7"/>
 <input name="specular" type="float" value="0.8"/>
 <input name="transmission" type="float" value="0.2"/>
 <input name="ior" type="float" value="1.45"/>
 <input name="emission" type="color3" value="0.1, 0.2, 0.3"/>
 <input name="coat" type="float" value="0.6"/>
 <input name="sheen" type="float" value="0.35"/>
 <input name="subsurface" type="float" value="0.25"/>
 <input name="diffuse_roughness" type="float" value="0.15"/>
 <input name="opacity" type="float" value="0.9"/>
 <input name="weight" type="float" value="0.25"/>
 <input name="thickness" type="float" value="400"/>
 <input name="value" type="color3" value="0.1, 0.2, 0.3"/>
 <input name="fg" type="color3" value="0.9, 0.1, 0.1"/>
 <input name="bg" type="color3" value="0.1, 0.1, 0.9"/>
 <input name="in1" type="color3" value="0.5, 0.5, 0.5"/>
 <input name="in2" type="color3" value="0.5, 0.5, 0.5"/>
 <input name="default" type="color3" value="0.7, 0.6, 0.5"/>
 <output name="out"/>
</node>
<output name="surface" type="material"><connectfrom node="n" output="out"/></output>
</nodegraph></material></materyialgraph></materialx>"#
    )
}

/// 逐字段稳定文本（golden 口径：不用 `signature()`，直接列全部 17 个标量槽）。
fn fields_line(p: &BsdfParams) -> String {
    format!(
        "base={:?} met={} rough={} spec={} trans={} ior={} emi={:?} coat={} sheen={} sss={} dr={} thin={} op={}",
        p.base_color,
        p.metallic,
        p.roughness,
        p.specular,
        p.transmission,
        p.ior,
        p.emission,
        p.coat,
        p.sheen,
        p.subsurface,
        p.diffuse_roughness,
        p.thinfilm_thickness,
        p.opacity,
    )
}

/// 21 条 golden：结构位名 → 逐字段文本 → 逐位签名。
///
/// 前 13 项是 V2.5 之前既有节点，后 8 项是 B4 按覆盖度表补齐的
/// Burley / Oren–Nayar / Conductor / Dielectric / Sheen / Subsurface /
/// Transmission / ThinFilm。签名由字段位型经 FNV-1a 独立算出，落死在表内。
const GOLDEN: &[(&str, &str, u64)] = &[
    (
        "standard_surface",
        "base=[0.2, 0.4, 0.6] met=0.7 rough=0.3 spec=0.8 trans=0.2 ior=1.45 emi=[0.1, 0.2, 0.3] coat=0.6 sheen=0.35 sss=0.25 dr=0.15 thin=0 op=0.9",
        0xafaca1229138703d,
    ),
    (
        "bsdf",
        "base=[0.2, 0.4, 0.6] met=0.7 rough=0.3 spec=0.8 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=0.9",
        0x45973d3642dad4d1,
    ),
    (
        "diffuse",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.5 spec=0 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0.3 thin=0 op=0.9",
        0xd207aa98e0d2f723,
    ),
    (
        "glossy",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.3 spec=0.8 trans=0 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x3265ea8c772abe82,
    ),
    (
        "mix",
        "base=[0.7, 0.1, 0.29999998] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=0.25",
        0xeeaf9556d944ac2c,
    ),
    (
        "add",
        "base=[1.0, 1.0, 1.0] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x2588091f27b7979f,
    ),
    (
        "multiply",
        "base=[0.25, 0.25, 0.25] met=0 rough=0.25 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0xdcf1eebddf37979f,
    ),
    (
        "ramp",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x9ceb9644edf7d517,
    ),
    (
        "constant",
        "base=[0.1, 0.2, 0.3] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x2db6a9878677d517,
    ),
    (
        "position",
        "base=[0.5, 0.5, 0.5] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x4d8231b8b837979f,
    ),
    (
        "normal",
        "base=[0.5, 0.5, 1.0] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0xb766c11e0eb7979f,
    ),
    (
        "image",
        "base=[0.7, 0.6, 0.5] met=0 rough=0.5 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=0.9",
        0x99c7a6aa812ac58a,
    ),
    (
        "dot",
        "base=[0.8, 0.8, 0.8] met=0 rough=0.25 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x944738662ed8a4f6,
    ),
    (
        "burley_diffuse",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.5 spec=0 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0.3 thin=0 op=0.9",
        0xd207aa98e0d2f723,
    ),
    (
        "oren_nayar_diffuse",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.5 spec=0 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0.3 thin=0 op=0.9",
        0xd207aa98e0d2f723,
    ),
    (
        "conductor",
        "base=[0.2, 0.4, 0.6] met=0.7 rough=0.3 spec=0.8 trans=0 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0xc7f2e321d94bf92d,
    ),
    (
        "dielectric",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.3 spec=0.8 trans=0 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x3265ea8c772abe82,
    ),
    (
        "sheen",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.3 spec=0.5 trans=0 ior=1.5 emi=[0.0, 0.0, 0.0] coat=0 sheen=0.25 sss=0 dr=0 thin=0 op=1",
        0xfeeafe9c1256c2d5,
    ),
    (
        "subsurface",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.5 spec=0.5 trans=0 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0.25 dr=0.3 thin=0 op=1",
        0xd7fd8cd41ad0295f,
    ),
    (
        "transmission",
        "base=[0.2, 0.4, 0.6] met=0 rough=0.3 spec=0.8 trans=0.25 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=0 op=1",
        0x55ad1f8a8aaabe82,
    ),
    (
        "thinfilm",
        "base=[0.8, 0.8, 0.8] met=0 rough=0.5 spec=0.8 trans=0 ior=1.45 emi=[0.0, 0.0, 0.0] coat=0 sheen=0 sss=0 dr=0 thin=400 op=1",
        0xa8fc5f89a0ed0651,
    ),
];

/// 编译一个单节点 `.mtlx` 并取出该节点的 `BsdfParams` + 结构位名。
fn compile_one(ty: &str) -> (String, BsdfParams) {
    let pack = materialx::compile_str(&single_node_mtlx(ty))
        .unwrap_or_else(|e| panic!("节点 {ty} 应编译成功：{e}"));
    let graph = &pack.material("m").unwrap_or_else(|| panic!("材质 m 缺失")).graph;
    let structure = graph.nodes["n"].structure.clone();
    let params = materialx::node_to_bsdf(&graph.nodes["n"])
        .unwrap_or_else(|e| panic!("节点 {ty} 应映射成功：{e}"));
    (structure, params)
}

/// 读源码（静态门禁）。
fn source(path: &str) -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    fs::read_to_string(format!("{root}/../../{path}"))
        .unwrap_or_else(|e| panic!("静态门禁读不到 {path}：{e}"))
}

// ---------------------------------------------------------------------------
// V1：覆盖度判据表（带数字）+ 表内自洽
// ---------------------------------------------------------------------------

#[test]
fn v1_coverage_table_numbers_are_exact() {
    let table: &[NodeCoverage] = materialx::coverage_table();
    assert_eq!(table.len(), 21, "支持节点种类必须是 21");
    assert_eq!(GOLDEN.len(), 21, "每种节点一条 golden 对拍");
    assert_eq!(
        table.iter().map(|c| c.aliases.len()).sum::<usize>(),
        45,
        "nodedef 别名总数必须是 45"
    );
    assert_eq!(table, NODE_COVERAGE, "公开表与公开常量同一份");

    // 13 个命名域 / 17 个标量槽（3+1+1+1+1+1+3+1+1+1+1+1+1）。
    let d = BsdfParams::default();
    let slots = [
        d.base_color[0], d.base_color[1], d.base_color[2],
        d.metallic, d.roughness, d.specular, d.transmission, d.ior,
        d.emission[0], d.emission[1], d.emission[2],
        d.coat, d.sheen, d.subsurface, d.diffuse_roughness,
        d.thinfilm_thickness, d.opacity,
    ];
    assert_eq!(slots.len(), 17, "BsdfParams 标量槽 17 个");

    // 本轮补齐的 8 个常见标准节点必须全部在表内（按结构位名精确断言）。
    let added = [
        "burley_diffuse", "oren_nayar_diffuse", "conductor", "dielectric",
        "sheen", "subsurface", "transmission", "thinfilm",
    ];
    for s in added {
        assert!(table.iter().any(|c| c.structure == s), "补齐节点 {s} 必须在表内");
    }
    assert_eq!(added.len(), 8);

    // 表内字段清单：13 个合法域名，每个节点非空、无越界域名。
    const ALL_FIELDS: &[&str] = &[
        "base_color", "metallic", "roughness", "specular", "transmission", "ior",
        "emission", "coat", "sheen", "subsurface", "diffuse_roughness",
        "thinfilm_thickness", "opacity",
    ];
    assert_eq!(ALL_FIELDS.len(), 13, "BsdfParams 命名域 13 个");
    for c in table {
        assert!(!c.bsdf_params.is_empty(), "{} 必须声明映射字段", c.structure);
        for f in c.bsdf_params {
            assert!(ALL_FIELDS.contains(f), "{} 声明了未知字段 {f}", c.structure);
        }
        assert_eq!(c.kind.structure(), c.structure, "{} 结构位名与 NodeKind 一致", c.structure);
    }

    // 报告数字：21 节点全通过 / 17 标量槽 / 未支持节点明确 MX2004。
    let pack = materialx::compile_str(&single_node_mtlx("standard_surface")).unwrap();
    let rep: CoverageReport = materialx::coverage_report(&pack);
    assert_eq!(rep.node_kinds, 21);
    assert_eq!(rep.aliases, 45);
    assert_eq!(rep.bsdf_param_slots, 17);
    assert_eq!(rep.compiled_nodes, 1);
    assert!(rep.unsupported_behavior.contains("MX2004"));
    println!(
        "V1 覆盖度：{} 种节点 / {} 别名 / {} 命名域 / {} 标量槽 / 补齐 {} 个",
        rep.node_kinds, rep.aliases, ALL_FIELDS.len(), rep.bsdf_param_slots, added.len()
    );
}

#[test]
fn v1_every_alias_resolves_to_its_table_row() {
    let table = materialx::coverage_table();
    let mut checked = 0usize;
    for c in table {
        for alias in c.aliases {
            let kind = NodeKind::from_mtlx_type(alias)
                .unwrap_or_else(|| panic!("别名 {alias} 必须解析成 NodeKind"));
            assert_eq!(kind, c.kind, "别名 {alias} 应归到 {}", c.structure);
            // 带命名空间前缀 / 带 `_surfaceshader` 尾缀的真实 nodedef 写法。
            for wrapped in [
                format!("ND__{alias}"),
                format!("ND_{alias}"),
                format!("mx:{alias}"),
                format!("{alias}_surfaceshader"),
            ] {
                if let Some(k) = NodeKind::from_mtlx_type(&wrapped) {
                    assert_eq!(k, c.kind, "写法 {wrapped} 应归到 {}", c.structure);
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 45, "45 个别名逐条归一化对拍");
}

// ---------------------------------------------------------------------------
// V2：每个已支持节点一条 golden 对拍（21 条）+ 全部别名（45 条）
// ---------------------------------------------------------------------------

#[test]
fn v2_every_supported_node_has_one_golden_pair() {
    let table = materialx::coverage_table();
    assert_eq!(table.len(), GOLDEN.len());
    for (c, (want_structure, want_fields, want_sig)) in table.iter().zip(GOLDEN.iter()) {
        let (structure, params) = compile_one(c.aliases[0]);
        assert_eq!(&structure, want_structure, "节点 {} 结构位名", c.structure);
        assert_eq!(&fields_line(&params), want_fields, "节点 {} 逐字段 golden", c.structure);
        assert_eq!(params.signature(), *want_sig, "节点 {} 逐位签名 golden", c.structure);
        // 表内声明的映射字段必须真的被写进 BsdfParams（不是空声明）：
        // 与默认值对拍，声明的字段至少有一个偏离默认（每个节点都改动了 base_color）。
        let d = BsdfParams::default();
        assert!(
            params.signature() != d.signature(),
            "节点 {} 的映射必须真的写入字段（不能整包等于默认值）",
            c.structure
        );
    }
    println!("V2 golden：{} 种节点逐字段 + 逐位签名全对拍", GOLDEN.len());
}

#[test]
fn v2_all_45_aliases_compile_to_their_golden_signature() {
    let table = materialx::coverage_table();
    let mut checked = 0usize;
    for (c, (_, want_fields, want_sig)) in table.iter().zip(GOLDEN.iter()) {
        for alias in c.aliases {
            let (structure, params) = compile_one(alias);
            assert_eq!(structure, c.structure, "别名 {alias} 结构位名");
            assert_eq!(&fields_line(&params), want_fields, "别名 {alias} 逐字段 golden");
            assert_eq!(params.signature(), *want_sig, "别名 {alias} 逐位签名 golden");
            checked += 1;
        }
    }
    assert_eq!(checked, 45, "45 个别名逐条 golden 对拍");
    println!("V2 别名：{checked} 条全部落在各自节点的 golden 上");
}

// ---------------------------------------------------------------------------
// V3：双跑逐字节（编译 + 映射 + 21 节点整包）
// ---------------------------------------------------------------------------

/// 21 节点整包（一次编译覆盖全表）。
fn all_nodes_mtlx() -> String {
    let mut nodes = String::new();
    for (i, c) in NODE_COVERAGE.iter().enumerate() {
        let body = single_node_mtlx(c.aliases[0]);
        // 把单节点模板里的 `<node name="n" ...>` 段抽出来并改名，避免重名。
        let start = body.find("<node ").unwrap();
        let end = body.find("</node>").unwrap() + "</node>".len();
        let mut node = body[start..end].to_string();
        node = node.replacen("name=\"n\"", &format!("name=\"n{i:02}\""), 1);
        nodes.push_str(&node);
    }
    format!(
        r#"<materialx version="1.0"><materyialgraph><material name="m"><nodegraph>{nodes}<output name="surface" type="material"><connectfrom node="n00" output="out"/></output></nodegraph></material></materyialgraph></materialx>"#
    )
}

/// 整包 → 逐字节序列化（节点名 + 17 个标量位型），用于双跑逐字节对拍。
fn pack_bytes() -> Vec<u8> {
    let text = all_nodes_mtlx();
    let pack = materialx::compile_str(&text).unwrap();
    let pairs = materialx::compile_to_bsdf(&pack).unwrap();
    let mut out = Vec::new();
    for (name, p) in &pairs {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        for bits in [
            p.base_color[0].to_bits(), p.base_color[1].to_bits(), p.base_color[2].to_bits(),
            p.metallic.to_bits(), p.roughness.to_bits(), p.specular.to_bits(),
            p.transmission.to_bits(), p.ior.to_bits(),
            p.emission[0].to_bits(), p.emission[1].to_bits(), p.emission[2].to_bits(),
            p.coat.to_bits(), p.sheen.to_bits(), p.subsurface.to_bits(),
            p.diffuse_roughness.to_bits(), p.thinfilm_thickness.to_bits(), p.opacity.to_bits(),
        ] {
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

#[test]
fn v3_double_run_is_byte_identical() {
    let a = pack_bytes();
    let b = pack_bytes();
    assert_eq!(a, b, "双跑逐字节必须完全一致");
    // 每节点 = 名字字节 + 1 个 0 分隔 + 17 个 f32 位型（各 4 B）。
    let names_len: usize = (0..21).map(|i| format!("m/n{i:02}").len() + 1).sum();
    assert_eq!(a.len(), names_len + 21 * 17 * 4, "21 节点序列化长度精确对拍");
    // 独立编译两遍的 MaterialPack 结构本身也逐字段相等。
    let text = all_nodes_mtlx();
    let p1 = materialx::compile_str(&text).unwrap();
    let p2 = materialx::compile_str(&text).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(materialx::coverage_report(&p1).compiled_nodes, 21);
    assert_eq!(materialx::compile_to_bsdf(&p1).unwrap().len(), 21);
    println!("V3 双跑：{} 字节逐位一致，21 节点编译通过", a.len());
}

// ---------------------------------------------------------------------------
// V4：≥2 负控（本表 6 类），全部结构化三段式
// ---------------------------------------------------------------------------

fn wrap(body: &str) -> String {
    format!(
        r#"<materialx version="1.0"><materyialgraph><material name="m"><nodegraph>{body}<output name="surface" type="material"><connectfrom node="n" output="out"/></output></nodegraph></material></materyialgraph></materialx>"#
    )
}

fn assert_structured(err: MtlxError, want_code: &str) {
    assert_eq!(err.code, want_code, "错误码：{err}");
    assert!(!err.location.is_empty(), "三段式【在哪】不得为空");
    assert!(!err.problem.is_empty(), "三段式【什么事】不得为空");
    assert!(!err.fix.is_empty(), "三段式【怎么改】不得为空");
}

#[test]
fn v4_six_negative_controls_are_structured_errors() {
    let cases: &[(&str, &str, String)] = &[
        (
            "MX2004", "未知节点",
            wrap(r#"<node name="n" type="quantum_bsdf"><output name="out"/></node>"#),
        ),
        (
            "MX2005", "类型错配",
            wrap(r#"<node name="n" type="standard_surface"><input name="base_color" type="color3" value="0.1, 0.2"/><output name="out"/></node>"#),
        ),
        (
            "MX2006", "悬空连线",
            wrap(r#"<node name="n" type="standard_surface"><input name="base_color" type="color3" interconnect="missing.out"/><output name="out"/></node>"#),
        ),
        (
            "MX2007", "循环图",
            wrap(r#"<node name="n" type="mix"><input name="fg" type="color3" interconnect="m.out"/><output name="out"/></node><node name="m" type="mix"><input name="fg" type="color3" interconnect="n.out"/><output name="out"/></node>"#),
        ),
        (
            "MX2001", "XML 语法",
            "<materialx><oops></materialx>".to_string(),
        ),
        (
            "MX2005", "分量非数值",
            wrap(r#"<node name="n" type="standard_surface"><parameter name="x" type="boolean" value="maybe"/><output name="out"/></node>"#),
        ),
    ];
    for (code, what, src) in cases {
        let err = materialx::compile_str(src).expect_err(&format!("{what} 必须报错"));
        assert_structured(err, code);
    }
    println!("V4 负控：{} 类结构化错误，0 静默降级", cases.len());
}

// ---------------------------------------------------------------------------
// V5：产品路径接线断言（库侧静态门禁）
// ---------------------------------------------------------------------------

#[test]
fn v5_static_gate_product_path_is_wired() {
    let main = source("crates/prism-app/src/main.rs");
    assert!(main.contains("\"mtlx\" => Some(cmd_mtlx::run(rest))"), "CLI 必须分发 mtlx 子命令");
    assert!(main.contains("prism-app mtlx <FILE>"), "用法文本必须列 mtlx 子命令");
    let cmd = source("crates/prism-app/src/cmd_mtlx.rs");
    assert!(cmd.contains("materialx::compile_mtlx_report"), "产品命令必须调库侧统一出口");
    assert!(cmd.contains("21 种节点"), "产品命令必须打印判据表数字");
    // 既有 w7_preview::p2_no_shading_math_in_app_crate 禁止应用 crate 出现散射
    // 模型词汇；本判据同步钉死「应用层薄转发、零该词」（覆盖只增不减）。
    assert!(!cmd.to_lowercase().contains("bsdf"), "应用层必须保持薄转发、零散射词汇");
    let lib = source("crates/prism-io/src/lib.rs");
    assert!(lib.contains("pub mod materialx;"), "库必须公开 materialx 模块");
    let mx = source("crates/prism-io/src/materialx.rs");
    assert!(mx.contains("pub const NODE_COVERAGE"), "覆盖度表必须公开");
    assert!(mx.contains("pub fn compile_to_bsdf"), "映射必须公开");
    assert!(mx.contains("pub fn compile_mtlx_report"), "库侧统一出口必须公开");
    assert!(mx.contains("compile_file(path)"), "统一出口必须真调编译入口");
    assert!(mx.contains("coverage_report(&pack)"), "统一出口必须真消费覆盖度报告");
    assert!(mx.contains("compile_to_bsdf(&pack)"), "统一出口必须真消费参数映射");
    println!("V5 静态门禁：CLI / 薄转发 / 库 / 表 / 映射 / 统一出口内联三函数 六处接线齐备");
}

// ---------------------------------------------------------------------------
// V6：既有 MaterialX 行为不回退
// ---------------------------------------------------------------------------

#[test]
fn v6_existing_materialx_behavior_is_unchanged() {
    let minimal = r#"<materialx version="1.0"><materyialgraph><material name="mat_red"><nodegraph>
<node name="bsdf1" type="standard_surface"><input name="base_color" type="color3" value="0.8, 0.1, 0.05"/><output name="out"/></node>
<output name="surface" type="material"><connectfrom node="bsdf1" output="out"/></output>
</nodegraph></material></materyialgraph></materialx>"#;
    let a = materialx::compile_str(minimal).unwrap();
    let b = materialx::compile_str(minimal).unwrap();
    assert_eq!(a, b, "既有最小图仍确定性");
    assert_eq!(a.len(), 1);
    assert_eq!(a.material("mat_red").unwrap().graph.node_count(), 1);
    // 既有错误码口径不回退。
    let e = materialx::compile_str("<materialx><oops></materialx>").unwrap_err();
    assert_eq!(e.code, "MX2001");
    let e = materialx::compile_str(r#"<materialx version="1.0"><materyialgraph></materyialgraph></materialx>"#).unwrap_err();
    assert_eq!(e.code, "MX2003");
    // 补齐的 8 个节点在旧版本会报 MX2004，现在必须全部编译成功。
    for s in [
        "burley_diffuse", "oren_nayar_diffuse", "conductor", "dielectric",
        "sheen", "subsurface", "transmission", "thinfilm",
    ] {
        assert!(materialx::compile_str(&single_node_mtlx(s)).is_ok(), "{s} 不再拒绝");
    }
    println!("V6：既有行为不回退，8 个补齐节点由 MX2004 转为编译通过");
}
