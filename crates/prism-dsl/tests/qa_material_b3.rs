//! V2.2-B-3（T05）DSL 材质块兼容 —— 新字段解析 + 行列号诊断 + 老场景输出逐位不变。
//!
//! 验收 4：
//! - `anisotropy` / `anisotropy_rotation` 进入 `build_material` 的 P1002 白名单；
//! - 未知字段仍报 P1002，且带**行号诊断**（`err_at(f.line, …)`）；
//! - 老场景（仅 3 字段）输出逐位不变：既有 5 个参数键（albedo_r/g/b、roughness、
//!   metalness）的数值 / 键序与历史一致；新键仅在显式声明时插入。

use prism_core::node::{ParamValue, StructureId};
use prism_dsl::{compile_v2, CompileV2Output};

/// 最小可编译 v2 外壳（场景含一个 stage3d + material）。
fn shell(material_body: &str) -> String {
    format!(
        "// prism-dsl 1\n\
         movie \"m\" {{ duration 1s fps 12 resolution 320x180 format png }}\n\
         scene \"s\" {{ duration 1s stage3d {{\n\
             material m1 {{\n\
         {material_body}\n\
             }}\n\
         }} }}\n"
    )
}

/// 取出场景里的 material 节点（RadiativeTransfer）。
fn material_node(out: &CompileV2Output) -> &prism_core::node::Node {
    out.graph
        .nodes()
        .find(|n| n.structure == StructureId::RadiativeTransfer)
        .expect("编译产物应含 material（RadiativeTransfer）节点")
}

/// 读出 F32 参数（非 F32 即 panic）。
fn f32_param(node: &prism_core::node::Node, key: &str) -> f32 {
    match node.params.get(key) {
        Some(ParamValue::F32(v)) => *v,
        other => panic!("参数 {key} 应为 F32，实得 {other:?}"),
    }
}

/// 验收 4-①：新字段 `anisotropy` / `anisotropy_rotation` 可解析、投影为节点参数。
#[test]
fn new_anisotropy_fields_parse_and_projected() {
    // 每个字段独占一行，便于核对行号诊断（第 8 / 9 行）。
    let src = shell(
        "        albedo #c0392b\n\
         roughness 0.3\n\
         metalness 0.05\n\
         anisotropy 0.6\n\
         anisotropy_rotation 0.5\n",
    );
    let out = compile_v2(&src).expect("含 anisotropy 的材质应可编译");
    let node = material_node(&out);
    // 老键不变。
    assert_eq!(f32_param(node, "roughness"), 0.3);
    assert_eq!(f32_param(node, "metalness"), 0.05);
    // 新键已投影。
    assert_eq!(f32_param(node, "anisotropy"), 0.6);
    assert_eq!(f32_param(node, "anisotropy_rotation"), 0.5);
    // 反照率键按 0..1 归一化（C17）。
    assert!((f32_param(node, "albedo_r") - 192.0 / 255.0).abs() < 1e-6);
}

/// 验收 4-②：未知字段仍被 P1002 拒绝，且带行号诊断（断言错误行 = 字段所在行）。
#[test]
fn unknown_field_still_rejected_with_line() {
    // `bogusfield` 在第 8 行。
    let src = shell(
        "        albedo #c0392b\n\
         roughness 0.3\n\
         metalness 0.05\n\
         bogusfield 0.5\n",
    );
    let err = compile_v2(&src).expect_err("未知字段必须报 P1002");
    assert_eq!(err.line, 8, "P1002 应指向未知字段所在行（第 8 行）");
    assert!(
        err.message.contains("P1002"),
        "错误码应为 P1002，实得：{}",
        err.message
    );
    assert!(
        err.message.contains("anisotropy"),
        "错误建议段应列出允许字段（含 anisotropy）：{}",
        err.message
    );
}

/// 验收 4-③：老场景（仅 3 字段）输出逐位不变 —— 既有 5 键数值与历史一致，
/// 且不出现 anisotropy 键（新键仅在显式声明时插入）。
#[test]
fn legacy_three_field_material_unchanged() {
    // 单行写法（与既有黄金示例 `w2d_showcase.prism` 同构）。
    let src = shell("albedo #c0392b roughness 0.3 metalness 0.05");
    let out = compile_v2(&src).expect("老 3 字段材质应可编译");
    let node = material_node(&out);
    // 既有 5 键数值与历史一致（确认本次改动未触碰既有计算）。
    assert_eq!(f32_param(node, "albedo_r"), 192.0 / 255.0);
    assert_eq!(f32_param(node, "albedo_g"), 57.0 / 255.0);
    assert_eq!(f32_param(node, "albedo_b"), 43.0 / 255.0);
    assert_eq!(f32_param(node, "roughness"), 0.3);
    assert_eq!(f32_param(node, "metalness"), 0.05);
    // 老场景不得出现 anisotropy 键 ⇒ 序列化逐字节不变（Params 为 BTreeMap）。
    assert!(
        node.params.get("anisotropy").is_none(),
        "老场景不应有 anisotropy 键（输出逐位不变）"
    );
    assert!(
        node.params.get("anisotropy_rotation").is_none(),
        "老场景不应有 anisotropy_rotation 键"
    );
}

/// 验收 4-④：P1002 白名单消息列出全部允许字段（albedo / roughness / metalness /
/// anisotropy / anisotropy_rotation）。
#[test]
fn p1002_message_lists_all_allowed_fields() {
    let src = shell(
        "        albedo #c0392b\n\
         roughness 0.3\n\
         totallyunknown 1\n",
    );
    let err = compile_v2(&src).expect_err("未知字段必须报 P1002");
    let msg = &err.message;
    for allow in ["albedo", "roughness", "metalness", "anisotropy", "anisotropy_rotation"] {
        assert!(msg.contains(allow), "P1002 建议段应列出 '{allow}'：{msg}");
    }
}
