//! USD `.usda`（ASCII USD）**导出**：引擎场景 → 标准文本，[`crate::usd`]
//! 解析器的规范逆。
//!
//! 设计依据：《表示层反向转换设计》§2 最小子集 / §4 R2 三级判据 / §5 R3
//! 逐类有损表 / §6 不做清单；宪章 C18（错误三段式）、C20（降级 / 丢弃逐条
//! 记账，绝不静默）、C24 ③（UTF-8 + LF）、C17（确定性）。
//!
//! # 两个入口，一个序列化器
//! - [`export_layer`]：`Layer` → canonical `.usda` 文本。这是**核心**（解析器
//!   的逆），导入资产再导出走这里——导入时未知 prim / 属性的原文已保留在
//!   [`crate::usd::Layer`] 里，直导不丢；
//! - [`export_scene`]：`Scene` + 材质表 + 层元数据 → 扁平装配成 `Layer` →
//!   同一个序列化器。引擎自产内容走这里。
//!
//! 两个入口共用内部 `serialize`，不设第二套写出门。
//!
//! # 规范形（canonical form）
//! 文本只由数据结构决定（无时间戳、无哈希序、无环境信息）：
//! - 层头 `#usda <version>`，元数据块按 [`crate::usd::Layer::metadata`] 的源顺序；
//! - 每个 prim 一个 `{ }` 体；属性按 `Vec` 源顺序在前，关系紧随，子 prim 殿后；
//! - 属性行 `[uniform ] [custom ]<type> <name> = <value>[ ( interpolation = "…" )]`；
//! - 数字一律十进制最短往返形式（`f64` 用 `Display`；解析器 `parse::<f64>()`
//!   逐位还原，见 `f64_text_roundtrip_is_bitwise`）；
//! - 数组 `[a, b]`、元组 `(a, b, c)` 由**声明类型**决定（`int[]` vs `double3`）；
//! - UTF-8 + LF，文件以 `\n` 结尾。
//!
//! 不变量（R2 L2 字节不动点）：
//! `export_layer(&parse(export_layer(L)).layer) == export_layer(L)`。
//! 不与**手写** `.usda` 比字节（手写源只断言 L1 语义，见设计 §4 表格）。
//!
//! # 引擎场景的装配形状（[`export_scene`]）
//! 一个根 `Xform`（`defaultPrim`）+ 每个网格一个 `Mesh` prim + 每个**被绑定
//! 引用**的材质一个 `Material` prim（内含一个 `Shader`）。几何**已是世界
//! 坐标**：`Scene` 在导入时就把 Xform 链烘焙进顶点，导出不写 `xformOp`
//! （写了会二次变换）。面**保 polygon**：`faceVertexCounts` 原样写，不三角化。
//!
//! # 诚实边界（与导入侧同界，见设计 §1 / §6）
//! - **不导出** `.usdc`/`.usdz`、`subLayers`、timeSamples/动画、
//!   variantSet——被丢弃项逐条进 [`ExportReport`]（`dropped_layer_metadata`
//!   / `dropped_time_samples` / `dropped_variant_sets`）；
//! - **组合弧（`references` / `payload`）不重建**：导入器把它们建模成普通
//!   字符串属性（`type_name` 记的是 `prepend`/`append` 修饰词，值是
//!   `Asset`+`Path` 拼成的 `Str`），故本模块只能按原属性写回成**带引号的
//!   字符串**（`prepend references = "./x.usda@</P>"`）。文本对**本仓库解析器**
//!   是 L2 不动点，但对真正的 USD 消费方已不是组合弧（`@…@` 资产语法丢了）。
//!   该信息缺口在导入侧建模（`usd::parse_attribute`），不在导出口，故
//!   **不**进 [`ExportReport`]；修复需导入侧建模改动，见设计遗留项。
//! - **非 ASCII 文本**照常写出（文件本身是合法 UTF-8），但本仓库解析器的
//!   字符串体按字节推进（`usd::Lexer::read_string_body`），回读会破坏多字节
//!   字符，故 L2 判据语料限 ASCII；命中时 [`ExportReport::non_ascii_text`] 记账；
//! - **材质是 UsdPreviewSurface 属性级直射**：解析器不求值 Shader 输入
//!   （设计 D-1 未决），故属性级往返成立、语义级待接线；`sheen*` 无对应
//!   输入，进 [`ExportReport::dropped_sheen`]；
//! - **f32 材质参数的文本口径**：`float`/`color3f` 在 USD 里就是 f32，故按
//!   f32 最短文本写（`0.1f32` → `0.1`），回读成 f32 逐位无损；极端值若
//!   f64 文本会更短（双重舍入），回退到精确 f64 文本以保证 L2——两条路径都
//!   是确定的，选择条件是纯文本比较，不含随机性。

use crate::usd::{
    AttrValue, Attribute, Layer, Mat4, Prim, PrimKind, Relationship, Scene, UsdError,
};
use std::collections::{BTreeMap, BTreeSet};

// ===========================================================================
//  常量与错误码
// ===========================================================================

/// 缩进单位（4 空格；LF 换行，C24 ③）。
const INDENT: &str = "    ";

/// 导出器 schema 版本。写进文本的注释行——导入器丢弃 ⟹ 不参与往返签名；
/// 交付 manifest / 人读可辨（设计 §7 C13，同 V11「版本写入所有产物」精神）。
pub const USDA_EXPORTER_SCHEMA_VERSION: u32 = 1;

/// 导出错误码：输入**不可表示**（非有限浮点 / 非法标识符 / 缺材质参数 /
/// 拓扑不自洽）。接在解析器 PU1001–PU1012 之后，同属 PU 段（C18）。
const E_BAD_INPUT: &str = "PU1013";

// ===========================================================================
//  导出报告（C20：降级 / 丢弃逐条记账）
// ===========================================================================

/// 导出报告：凡是「没按原样写出去」的东西都在这里留名，**永不静默**。
///
/// 所有字段在返回前统一升序排序 + 去重（确定性契约）；空报告 = 全程无损
/// （相对本模块声明的子集）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExportReport {
    /// 被丢弃的 `timeSamples`（`<prim 路径>@<属性名>`）。
    pub dropped_time_samples: Vec<String>,
    /// 被丢弃的 variantSet（`<prim 路径>:<集名>`）。
    pub dropped_variant_sets: Vec<String>,
    /// 被丢弃的层元数据键（当前只有 `subLayers`——数组语义无法用标量文本重建）。
    pub dropped_layer_metadata: Vec<String>,
    /// sheen 无处落地的材质（UsdPreviewSurface 无 sheenColor/sheenRoughness 输入）。
    pub dropped_sheen: Vec<String>,
    /// 插值标签被改写的属性（长度 ≠ 顶点数，只能标 `vertex` 才能保住数值）。
    pub relabeled_interpolation: Vec<String>,
    /// 世界变换已烘焙进顶点的网格（`world_transform` 非单位；导出不写 xform）。
    pub baked_transforms: Vec<String>,
    /// 因非法字符 / 重名被改写的 prim（`(源标识, 导出路径)`）。
    pub renamed_prims: Vec<(String, String)>,
    /// 材质表里从未被任何 `material:binding` 引用的 key。
    pub unused_materials: Vec<String>,
    /// 含非 ASCII 文本的位置（导出文本合法；本仓库解析器回读会破坏多字节字符）。
    pub non_ascii_text: Vec<String>,
}

impl ExportReport {
    /// 报告是否为空（= 子集内全程无损）。
    pub fn is_empty(&self) -> bool {
        self.dropped_time_samples.is_empty()
            && self.dropped_variant_sets.is_empty()
            && self.dropped_layer_metadata.is_empty()
            && self.dropped_sheen.is_empty()
            && self.relabeled_interpolation.is_empty()
            && self.baked_transforms.is_empty()
            && self.renamed_prims.is_empty()
            && self.unused_materials.is_empty()
            && self.non_ascii_text.is_empty()
    }

    /// 合并另一份报告（入口 [`export_scene`] 把适配层与序列化层的记账合流）。
    fn absorb(&mut self, o: ExportReport) {
        self.dropped_time_samples.extend(o.dropped_time_samples);
        self.dropped_variant_sets.extend(o.dropped_variant_sets);
        self.dropped_layer_metadata.extend(o.dropped_layer_metadata);
        self.dropped_sheen.extend(o.dropped_sheen);
        self.relabeled_interpolation
            .extend(o.relabeled_interpolation);
        self.baked_transforms.extend(o.baked_transforms);
        self.renamed_prims.extend(o.renamed_prims);
        self.unused_materials.extend(o.unused_materials);
        self.non_ascii_text.extend(o.non_ascii_text);
    }

    /// 排序 + 去重（集合语义字段；`renamed_prims` 按 (源, 目标) 排序）。
    fn normalize(&mut self) {
        let all: [&mut Vec<String>; 8] = [
            &mut self.dropped_time_samples,
            &mut self.dropped_variant_sets,
            &mut self.dropped_layer_metadata,
            &mut self.dropped_sheen,
            &mut self.relabeled_interpolation,
            &mut self.baked_transforms,
            &mut self.unused_materials,
            &mut self.non_ascii_text,
        ];
        for v in all {
            v.sort();
            v.dedup();
        }
        self.renamed_prims.sort();
        self.renamed_prims.dedup();
    }
}

// ===========================================================================
//  材质（PbrMaterial 的中性镜像）
// ===========================================================================

/// 材质参数——`prism-geometry::PbrMaterial` 的**字段级中性镜像**。
///
/// # 为什么在 io 侧另立结构
/// `prism-io` 不依赖 `prism-geometry`（依赖方向是 geometry → io，反向成环），
/// 与 [`crate::usd::RenderMesh`] 在 io 侧出中性数据是同一取舍（设计 D-4）。
/// 调用方按字段名直射，零计算：
///
/// | 本结构 | UsdPreviewSurface |
/// |---|---|
/// | `base_color[0..3]` | `inputs:diffuseColor` |
/// | `base_color[3]` | `inputs:opacity` |
/// | `metalness` | `inputs:metallic` |
/// | `roughness` | `inputs:roughness` |
/// | `ior` | `inputs:ior` |
/// | `clearcoat` | `inputs:clearcoat` |
/// | `clearcoat_roughness` | `inputs:clearcoatRoughness` |
/// | `alpha_cutoff` | `inputs:opacityThreshold`（`None` = 不写） |
/// | `sheen_color` / `sheen_roughness` | **无对应输入**（进导出报告） |
///
/// 默认值与 glTF / `PbrMaterial::default()` 同口径（金属度 1、粗糙度 1、
/// IOR 1.5、不透明白）。
#[derive(Debug, Clone, PartialEq)]
pub struct UsdMaterial {
    /// 线性 RGBA 底色（RGB → diffuseColor，A → opacity）。
    pub base_color: [f32; 4],
    /// 金属度。
    pub metalness: f32,
    /// 粗糙度。
    pub roughness: f32,
    /// 折射率（默认 1.5）。
    pub ior: f32,
    /// sheen 颜色（无对应输入，非零时进报告）。
    pub sheen_color: [f32; 3],
    /// sheen 粗糙度（无对应输入，非零时进报告）。
    pub sheen_roughness: f32,
    /// 清漆强度。
    pub clearcoat: f32,
    /// 清漆粗糙度。
    pub clearcoat_roughness: f32,
    /// 镂空阈值（`None` = 不写 `opacityThreshold`）。
    pub alpha_cutoff: Option<f32>,
}

impl Default for UsdMaterial {
    fn default() -> Self {
        UsdMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metalness: 1.0,
            roughness: 1.0,
            ior: 1.5,
            sheen_color: [0.0, 0.0, 0.0],
            sheen_roughness: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            alpha_cutoff: None,
        }
    }
}

// ===========================================================================
//  层元数据
// ===========================================================================

/// 导出层元数据（设计 §2「场景组织」三件：defaultPrim / upAxis / metersPerUnit）。
#[derive(Debug, Clone, PartialEq)]
pub struct ExportMeta {
    /// 根 prim 名，同时写进 `defaultPrim`（必须是合法 prim 标识符）。
    pub default_prim: String,
    /// `upAxis`：USD 只允许 `"Y"` / `"Z"`。
    pub up_axis: String,
    /// `metersPerUnit`：必须有限且 > 0。
    pub meters_per_unit: f64,
}

impl Default for ExportMeta {
    fn default() -> Self {
        ExportMeta {
            default_prim: "root".to_string(),
            up_axis: "Y".to_string(),
            meters_per_unit: 1.0,
        }
    }
}

// ===========================================================================
//  公开入口
// ===========================================================================

/// 层 → canonical `.usda` 文本 + 导出报告（解析器的规范逆）。
///
/// 属性 / prim 顺序按 `Layer` 的 `Vec` 源顺序保留（R1：不重排）；被丢弃的
/// `timeSamples` / variantSet / `subLayers` 逐条记账（C20）。
///
/// # 错误
/// `PU1013`：数值非有限、名字不是裸标识符、字符串含不可转义控制字符、
/// 路径目标含 `<>`、prim 路径与树位置不一致——这些都会写出**语法错误或
/// 语义漂移**的文本，故结构化拒绝而非"尽力而为"。
pub fn export_layer(layer: &Layer) -> Result<(String, ExportReport), UsdError> {
    let mut report = ExportReport::default();
    let text = serialize(layer, &mut report)?;
    report.normalize();
    Ok((text, report))
}

/// 引擎场景 + 材质表 + 层元数据 → canonical `.usda` 文本 + 导出报告。
///
/// 装配形状见模块文档：根 `Xform` + 每网格一个 `Mesh` + 每材质一个
/// `Material`（内含 `Shader`）。**面保 polygon**，不做三角化；几何按世界
/// 坐标写（`Scene` 已烘焙），不写 `xformOp`。
///
/// `materials` 的 key 必须与 `Mesh.material` **逐字相同**（即导入时
/// `material:binding` 的目标路径，或引擎侧材质标识）；装配顺序按 key 升序，
/// 因此同一输入两次导出字节一致。层版本恒写 `1.0`（`Scene.version` 只是
/// 导入来源的记账，不参与导出）。
///
/// # 错误
/// - `PU1013`：绑定目标缺材质参数（写出即悬空，回读必撞 PU1005）→ 一次
///   列全所有缺口，不写半份文件；
/// - `PU1013`：拓扑不自洽（counts 之和 ≠ indices 长度、count < 3、索引越界）、
///   非有限数、非法标识符、层元数据非法。
pub fn export_scene(
    scene: &Scene,
    materials: &BTreeMap<String, UsdMaterial>,
    meta: &ExportMeta,
) -> Result<(String, ExportReport), UsdError> {
    validate_meta(meta)?;
    let mut report = ExportReport::default();

    // ── 0. 绑定目标必须有参数（设计 §1 事实 1：写 rel 必须同时写目标 prim）──
    let mut missing: Vec<String> = Vec::new();
    for m in &scene.meshes {
        if let Some(k) = &m.material {
            if !materials.contains_key(k) && !missing.contains(k) {
                missing.push(k.clone());
            }
        }
    }
    if !missing.is_empty() {
        missing.sort();
        return Err(UsdError::new(
            E_BAD_INPUT,
            "usd_export::export_scene",
            format!(
                "{} 个网格的 `material:binding` 目标没有材质参数（会写出悬空关系，\
                 回读必撞 PU1005）：{missing:?}",
                missing.len()
            ),
            "为每个绑定目标在材质表里补 `UsdMaterial`（key 与 `Mesh.material` 逐字相同）；\
             若资产刚由 `.usda` 导入，改用 `export_layer` 直导（材质语义求值见设计 D-1）",
        ));
    }

    // ── 1. 命名：网格先占位，材质随后（同名加 `_2`），全 ASCII 标识符 ──
    // 只有**被绑定引用**的材质才进文件：材质表是绑定行李，不是资产清单。
    let bound: BTreeSet<String> = scene
        .meshes
        .iter()
        .filter_map(|m| m.material.clone())
        .collect();
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut mesh_names: Vec<String> = Vec::with_capacity(scene.meshes.len());
    for (i, m) in scene.meshes.iter().enumerate() {
        let leaf = leaf_of(&m.path);
        let (name, changed) = unique_name(leaf, &format!("mesh_{i}"), &mut used);
        if changed {
            report
                .renamed_prims
                .push((m.path.clone(), format!("/{}/{name}", meta.default_prim)));
        }
        mesh_names.push(name);
    }
    let mut mat_paths: BTreeMap<String, String> = BTreeMap::new();
    for (n_bound, key) in materials.keys().filter(|k| bound.contains(*k)).enumerate() {
        let leaf = leaf_of(key);
        let (name, changed) = unique_name(leaf, &format!("material_{n_bound}"), &mut used);
        let path = format!("/{}/{name}", meta.default_prim);
        if changed {
            report.renamed_prims.push((key.clone(), path.clone()));
        }
        mat_paths.insert(key.clone(), path);
    }

    // ── 2. 网格 prim（属性名升序 = 规范顺序；rel 目标回填见步骤 3）──
    let mut mesh_prims: Vec<Prim> = Vec::with_capacity(scene.meshes.len());
    for (i, m) in scene.meshes.iter().enumerate() {
        check_topology(m)?;
        let name = mesh_names[i].clone();
        let path = format!("/{}/{}", meta.default_prim, name);
        let mut p = new_prim(PrimKind::Def, "Mesh", name, path.clone());
        p.attributes.push(new_attr(
            "faceVertexCounts",
            "int[]",
            AttrValue::Numbers(m.face_counts.iter().map(|c| f64::from(*c)).collect()),
        ));
        p.attributes.push(new_attr(
            "faceVertexIndices",
            "int[]",
            AttrValue::Numbers(m.face_indices.iter().map(|c| f64::from(*c)).collect()),
        ));
        if !m.normals.is_empty() {
            let mut a = new_attr(
                "normals",
                "normal3f[]",
                AttrValue::Rows(m.normals.iter().map(|n| n.to_vec()).collect()),
            );
            a.interpolation = Some("vertex".to_string());
            p.attributes.push(a);
            if m.normals.len() != m.points.len() {
                report.relabeled_interpolation.push(format!(
                    "{}（normals {} ≠ points {}）",
                    path,
                    m.normals.len(),
                    m.points.len()
                ));
            }
        }
        p.attributes.push(new_attr(
            "points",
            "point3f[]",
            AttrValue::Rows(m.points.iter().map(|q| q.to_vec()).collect()),
        ));
        if !m.uvs.is_empty() {
            let mut a = new_attr(
                "primvars:st",
                "texCoord2f[]",
                AttrValue::Rows(m.uvs.iter().map(|q| q.to_vec()).collect()),
            );
            a.interpolation = Some("vertex".to_string());
            p.attributes.push(a);
            if m.uvs.len() != m.points.len() {
                report.relabeled_interpolation.push(format!(
                    "{}（primvars:st {} ≠ points {}）",
                    path,
                    m.uvs.len(),
                    m.points.len()
                ));
            }
        }
        if m.world_transform != Mat4::identity() {
            report.baked_transforms.push(path.clone());
        }
        mesh_prims.push(p);
    }
    // rel 目标回填（命名阶段已完成，路径表就绪）。
    for (i, m) in scene.meshes.iter().enumerate() {
        if let Some(k) = &m.material {
            mesh_prims[i].relationships.push(Relationship {
                name: "material:binding".to_string(),
                targets: vec![mat_paths[k].clone()],
            });
        }
    }

    // ── 3. 材质 prim（Material + Shader；参数 f32 最短文本）──
    let mut mat_prims: Vec<Prim> = Vec::with_capacity(mat_paths.len());
    for key in materials.keys().filter(|k| bound.contains(*k)) {
        let mat = &materials[key];
        let path = mat_paths[key].clone();
        let name = path.rsplit('/').next().unwrap_or("material").to_string();
        let loc = format!("材质 {key}");
        if mat.sheen_color != [0.0, 0.0, 0.0] || mat.sheen_roughness != 0.0 {
            report.dropped_sheen.push(key.clone());
        }
        let shader_path = format!("{path}/shader");
        let mut shader = new_prim(
            PrimKind::Def,
            "Shader",
            "shader".to_string(),
            shader_path.clone(),
        );
        shader.attributes.push(uniform_attr(
            "info:id",
            "token",
            AttrValue::Str("UsdPreviewSurface".to_string()),
        ));
        shader.attributes.push(new_attr(
            "inputs:diffuseColor",
            "color3f",
            AttrValue::Numbers(vec![
                f32_text_f64(mat.base_color[0], &loc)?,
                f32_text_f64(mat.base_color[1], &loc)?,
                f32_text_f64(mat.base_color[2], &loc)?,
            ]),
        ));
        shader
            .attributes
            .push(attr_f32("inputs:opacity", mat.base_color[3], &loc)?);
        shader
            .attributes
            .push(attr_f32("inputs:metallic", mat.metalness, &loc)?);
        shader
            .attributes
            .push(attr_f32("inputs:roughness", mat.roughness, &loc)?);
        shader
            .attributes
            .push(attr_f32("inputs:ior", mat.ior, &loc)?);
        shader
            .attributes
            .push(attr_f32("inputs:clearcoat", mat.clearcoat, &loc)?);
        shader.attributes.push(attr_f32(
            "inputs:clearcoatRoughness",
            mat.clearcoat_roughness,
            &loc,
        )?);
        if let Some(c) = mat.alpha_cutoff {
            shader
                .attributes
                .push(attr_f32("inputs:opacityThreshold", c, &loc)?);
        }

        let mut mp = new_prim(PrimKind::Def, "Material", name, path);
        mp.attributes.push(new_attr(
            "outputs:surface.connect",
            "token",
            AttrValue::Path(format!("{shader_path}.outputs:surface")),
        ));
        mp.children.push(shader);
        mat_prims.push(mp);
    }

    // ── 4. 扁平装配：根 Xform + 网格（场景序）+ 材质（key 升序）──
    let mut root = new_prim(
        PrimKind::Def,
        "Xform",
        meta.default_prim.clone(),
        format!("/{}", meta.default_prim),
    );
    root.children.extend(mesh_prims);
    root.children.extend(mat_prims);

    // 材质表里没被任何网格引用的条目（不写进文件，免得给资产塞垃圾 prim）。
    for key in materials.keys() {
        if !bound.contains(key) {
            report.unused_materials.push(key.clone());
        }
    }

    let layer = Layer {
        version: "1.0".to_string(),
        metadata: vec![
            ("defaultPrim".to_string(), meta.default_prim.clone()),
            ("upAxis".to_string(), meta.up_axis.clone()),
            (
                "metersPerUnit".to_string(),
                format!("{}", meta.meters_per_unit),
            ),
        ],
        prims: vec![root],
        // 派生字段（解析时建表用）；序列化器不读它。
        all_paths: Vec::new(),
    };

    let (text, inner) = export_layer(&layer)?;
    report.absorb(inner);
    report.normalize();
    Ok((text, report))
}

// ===========================================================================
//  装配辅助
// ===========================================================================

/// 路径的最后一节（`/a/b` → `b`；无 `/` 时取全文；结尾 `/` → `None`）。
fn leaf_of(path: &str) -> Option<&str> {
    let leaf = match path.rsplit_once('/') {
        Some((_, l)) => l,
        None => path,
    };
    if leaf.is_empty() {
        None
    } else {
        Some(leaf)
    }
}

/// 把任意串压成 USD 标识符（`[A-Za-z_][A-Za-z0-9_]*`）；非法字符 → `_`。
fn sanitize_ident(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// 取一个在 `used` 里未占用的名字；返回 `(名字, 是否与源名/源叶子不同)`。
///
/// 判据是**叶子**（`/mats/blue` 的叶子是 `blue`，不是整条 key）——否则每个
/// 带层级的材质 key 都会被误记为"改名"，报告失去信息量。
fn unique_name(leaf: Option<&str>, fallback: &str, used: &mut BTreeSet<String>) -> (String, bool) {
    let base = match leaf {
        Some(l) => {
            let s = sanitize_ident(l);
            if s.is_empty() {
                fallback.to_string()
            } else {
                s
            }
        }
        None => fallback.to_string(),
    };
    let mut name = base.clone();
    let mut n = 2u32;
    while used.contains(&name) {
        name = format!("{base}_{n}");
        n += 1;
    }
    let changed = leaf != Some(name.as_str());
    used.insert(name.clone());
    (name, changed)
}

/// 层元数据校验。
fn validate_meta(meta: &ExportMeta) -> Result<(), UsdError> {
    let loc = "usd_export::export_scene（层元数据）";
    let name = &meta.default_prim;
    let ok = matches!(name.chars().next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!(
                "`defaultPrim` {name:?} 不是合法 prim 标识符（USD 只允许 [A-Za-z_][A-Za-z0-9_]*）"
            ),
            "传一个纯 ASCII 标识符（如 \"root\" / \"Scene\"）；它同时是根 prim 名与层元数据",
        ));
    }
    if meta.up_axis != "Y" && meta.up_axis != "Z" {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("`upAxis` = {:?}，USD 只允许 \"Y\" 或 \"Z\"", meta.up_axis),
            "改成 \"Y\"（Y-up，USD 默认）或 \"Z\"（Z-up，DCC 常见）",
        ));
    }
    if !meta.meters_per_unit.is_finite() || meta.meters_per_unit <= 0.0 {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("`metersPerUnit` = {}，必须是有限正数", meta.meters_per_unit),
            "传 1（米）或 0.01（厘米）等有限正数；负数/0/NaN 会让下游尺度求值失真",
        ));
    }
    Ok(())
}

/// 网格拓扑自洽性（与 `usd::build_mesh` 的判据同口径：写出去必须能读回来）。
fn check_topology(m: &crate::usd::Mesh) -> Result<(), UsdError> {
    let loc = format!("usd_export::export_scene（网格 {}）", m.path);
    if !m.face_counts.is_empty() {
        let sum: u64 = m.face_counts.iter().map(|c| u64::from(*c)).sum();
        if sum as usize != m.face_indices.len() {
            return Err(UsdError::new(
                E_BAD_INPUT,
                loc.clone(),
                format!(
                    "`faceVertexCounts` 之和 = {sum}，但 `faceVertexIndices` 长度 = {}",
                    m.face_indices.len()
                ),
                "修正 counts/indices 其一（回读时同一判据会报 PU1004；导出不写半份资产）",
            ));
        }
        if let Some(c) = m.face_counts.iter().find(|c| **c < 3) {
            return Err(UsdError::new(
                E_BAD_INPUT,
                loc,
                format!("面顶点数 {c} < 3：USD 面至少 3 个顶点"),
                "删掉退化面或把它拆成合法多边形；回读时同一判据会报 PU1004",
            ));
        }
    }
    let n = m.points.len();
    if m.face_indices.iter().any(|i| *i as usize >= n) {
        let bad = m
            .face_indices
            .iter()
            .find(|i| **i as usize >= n)
            .copied()
            .unwrap_or(0);
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("面索引 {bad} 越界（顶点数 = {n}）"),
            "修正索引表；负索引 / 越界索引回读时会报 PU1004",
        ));
    }
    Ok(())
}

/// 新 prim（空体）。
fn new_prim(kind: PrimKind, type_name: &str, name: String, path: String) -> Prim {
    Prim {
        kind,
        type_name: type_name.to_string(),
        name,
        path,
        attributes: Vec::new(),
        relationships: Vec::new(),
        children: Vec::new(),
        xform_op_order: Vec::new(),
        references: Vec::new(),
        payloads: Vec::new(),
        variant_sets: Vec::new(),
    }
}

/// 新属性（无 qualified / 无 interpolation）。
fn new_attr(name: &str, type_name: &str, value: AttrValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        type_name: type_name.to_string(),
        value,
        uniform: false,
        custom: false,
        interpolation: None,
        has_time_samples: false,
    }
}

/// `uniform` 属性（`info:id` 用）。
fn uniform_attr(name: &str, type_name: &str, value: AttrValue) -> Attribute {
    let mut a = new_attr(name, type_name, value);
    a.uniform = true;
    a
}

/// `float` 属性；值走 f32 最短文本的口径（见 [`f32_text_f64`]）。
fn attr_f32(name: &str, v: f32, loc: &str) -> Result<Attribute, UsdError> {
    Ok(new_attr(
        name,
        "float",
        AttrValue::Number(f32_text_f64(v, loc)?),
    ))
}

/// f32 → 文本 → f64（**文本**是唯一权威，保证 L2 字节不动点）。
///
/// 常规路径：f32 最短十进制（`0.1f32` → `"0.1"`，回读成 f32 逐位无损）。
/// 极端值若该十进制的最短 **f64** 文本不同（双重舍入），回退到
/// `v as f64` 的精确文本——两种结果都由输入唯一决定，无随机性。
fn f32_text_f64(v: f32, loc: &str) -> Result<f64, UsdError> {
    if !v.is_finite() {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("材质参数 {v} 不是有限 f32，`.usda` 文本无法表示"),
            "把该参数改成有限数（NaN/inf 会被回读成标识符，静默写出去等于坏资产）",
        ));
    }
    let s = format!("{v}");
    let d: f64 = s.parse().map_err(|_| {
        UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("f32 最短文本 {s:?} 无法回读为 f64（不应发生：Display 契约）"),
            "这是导出器的内部不变量被破坏；请把该值与场景一起报告为缺陷",
        )
    })?;
    if format!("{d}") == s {
        Ok(d)
    } else {
        Ok(f64::from(v))
    }
}

// ===========================================================================
//  规范序列化器（解析器的逆）
// ===========================================================================

/// 文本写出门（全模块唯一）。
struct Writer<'a> {
    out: String,
    report: &'a mut ExportReport,
}

impl<'a> Writer<'a> {
    /// 写一行（缩进 + 内容 + LF）。
    fn line(&mut self, depth: usize, s: &str) {
        for _ in 0..depth {
            self.out.push_str(INDENT);
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// 非 ASCII 记账（导出正确，回读有损）。
    fn note_non_ascii(&mut self, s: &str, loc: &str) {
        if !s.is_ascii() {
            self.report.non_ascii_text.push(loc.to_string());
        }
    }

    /// 整层。
    fn layer(&mut self, layer: &Layer) -> Result<(), UsdError> {
        let version = canonical_version(&layer.version)?;
        self.out.push_str("#usda ");
        self.out.push_str(&version);
        self.out.push('\n');

        // 元数据块：先过滤（subLayers 不导），全空则不写块——
        // 否则回读得到空 metadata，再导出就少一块，L2 立刻破。
        let mut meta: Vec<(String, String)> = Vec::new();
        for (k, v) in &layer.metadata {
            let loc = format!("层元数据 {k}");
            if k == "subLayers" {
                self.report.dropped_layer_metadata.push(k.clone());
                continue;
            }
            check_ident(k, &loc, "层元数据键")?;
            self.note_non_ascii(v, &loc);
            meta.push((k.clone(), meta_value(v, &loc)?));
        }
        if !meta.is_empty() {
            self.out.push_str("(\n");
            for (k, v) in &meta {
                self.out.push_str(INDENT);
                self.out.push_str(k);
                self.out.push_str(" = ");
                self.out.push_str(v);
                self.out.push('\n');
            }
            self.out.push_str(")\n");
        }

        self.out.push('\n');
        self.out.push_str(&format!(
            "# prism-engine usda exporter schema {USDA_EXPORTER_SCHEMA_VERSION}\n"
        ));

        for p in &layer.prims {
            self.out.push('\n');
            self.prim(p, 0, "")?;
        }
        Ok(())
    }

    /// 单个 prim（含子树）。
    fn prim(&mut self, p: &Prim, depth: usize, parent: &str) -> Result<(), UsdError> {
        let loc = format!("prim {}", p.path);
        if p.name.is_empty() || p.name.contains('/') {
            return Err(UsdError::new(
                E_BAD_INPUT,
                loc,
                format!("prim 名 {:?} 为空或含 `/`（会让路径与树位置错位）", p.name),
                "prim 名取路径最后一节且不得含 `/`；改名后重试",
            ));
        }
        let expect = format!("{parent}/{}", p.name);
        if p.path != expect {
            return Err(UsdError::new(
                E_BAD_INPUT,
                loc,
                format!(
                    "记录的路径 {:?} 与树位置不一致（按父路径应为 {expect:?}）——写出去会改变关系目标",
                    p.path
                ),
                "以树位置为准修正 `Prim.path`，或修正 `Prim.name`；否则 `material:binding` 会指向旧路径",
            ));
        }
        self.note_non_ascii(&p.name, &loc);
        if !p.variant_sets.is_empty() {
            for v in &p.variant_sets {
                self.report
                    .dropped_variant_sets
                    .push(format!("{}:{v}", p.path));
            }
        }

        let mut head = String::new();
        head.push_str(p.kind.keyword());
        head.push(' ');
        if !p.type_name.is_empty() {
            check_type(&p.type_name, &loc)?;
            head.push_str(&p.type_name);
            head.push(' ');
        }
        head.push_str(&quote(&p.name, &loc)?);
        self.line(depth, &head);
        self.line(depth, "{");
        for a in &p.attributes {
            self.attribute(a, depth + 1, &p.path)?;
        }
        for r in &p.relationships {
            self.relationship(r, depth + 1, &p.path)?;
        }
        let has_props = !p.attributes.is_empty() || !p.relationships.is_empty();
        for (i, c) in p.children.iter().enumerate() {
            if has_props || i > 0 {
                self.out.push('\n');
            }
            self.prim(c, depth + 1, &p.path)?;
        }
        self.line(depth, "}");
        Ok(())
    }

    /// 属性行。
    fn attribute(&mut self, a: &Attribute, depth: usize, prim_path: &str) -> Result<(), UsdError> {
        let loc = format!("prim {prim_path} 的属性 {}", a.name);
        check_ident(&a.name, &loc, "属性名")?;
        check_type(&a.type_name, &loc)?;
        if a.has_time_samples {
            self.report
                .dropped_time_samples
                .push(format!("{prim_path}@{}", a.name));
        }
        let mut s = String::new();
        if a.uniform {
            s.push_str("uniform ");
        }
        if a.custom {
            s.push_str("custom ");
        }
        s.push_str(&a.type_name);
        s.push(' ');
        s.push_str(&a.name);
        if let Some(v) = emit_value(&a.type_name, &a.value, &loc)? {
            s.push_str(" = ");
            s.push_str(&v);
            match &a.value {
                AttrValue::Str(t) | AttrValue::Asset(t) | AttrValue::Path(t) => {
                    self.note_non_ascii(t, &loc);
                }
                AttrValue::Strings(ts) => {
                    for t in ts {
                        self.note_non_ascii(t, &loc);
                    }
                }
                AttrValue::Paths(ps) => {
                    for t in ps {
                        self.note_non_ascii(t, &loc);
                    }
                }
                _ => {}
            }
        }
        if let Some(i) = &a.interpolation {
            s.push_str(" ( interpolation = ");
            s.push_str(&quote(i, &loc)?);
            s.push_str(" )");
            self.note_non_ascii(i, &loc);
        }
        self.line(depth, &s);
        Ok(())
    }

    /// 关系行。
    fn relationship(
        &mut self,
        r: &Relationship,
        depth: usize,
        prim_path: &str,
    ) -> Result<(), UsdError> {
        let loc = format!("prim {prim_path} 的关系 {}", r.name);
        check_ident(&r.name, &loc, "关系名")?;
        let text = match r.targets.len() {
            0 => "[]".to_string(),
            1 => format!("<{}>", check_target(&r.targets[0], &loc)?),
            _ => {
                let mut parts = Vec::with_capacity(r.targets.len());
                for t in &r.targets {
                    parts.push(format!("<{}>", check_target(t, &loc)?));
                }
                format!("[{}]", parts.join(", "))
            }
        };
        for t in &r.targets {
            self.note_non_ascii(t, &loc);
        }
        self.line(depth, &format!("rel {} = {}", r.name, text));
        Ok(())
    }
}

/// 序列化入口（唯一写出门）。
fn serialize(layer: &Layer, report: &mut ExportReport) -> Result<String, UsdError> {
    let mut w = Writer {
        out: String::new(),
        report,
    };
    w.layer(layer)?;
    Ok(w.out)
}

// ===========================================================================
//  值的规范文本（按声明类型决定数组 / 元组形态）
// ===========================================================================

/// 属性值 → 文本。`AttrValue::None` 返回 `None`（只写声明，不写 `= …`）。
///
/// # 形态规则（回读必须落到同一个 `AttrValue`，否则 L2 破）
/// - 数组类型（`…[]`）：`[a, b]` / `[(…), (…)]`；
/// - 非数组类型：单值裸写，多值元组 `(…)`（`double3` / `matrix4d`）；
/// - `Rows` 的元素**恒带括号**（哪怕行宽 1）——否则回读会塌成 `Numbers`；
/// - 空行拒绝（回读会补 0，属于静默改值）。
fn emit_value(type_name: &str, v: &AttrValue, loc: &str) -> Result<Option<String>, UsdError> {
    let is_array = type_name.ends_with("[]");
    let s = match v {
        AttrValue::None => return Ok(None),
        AttrValue::Number(n) => num(*n, loc)?,
        AttrValue::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        AttrValue::Str(s) => quote(s, loc)?,
        AttrValue::Asset(s) => {
            if s.contains('@') || s.contains('\n') || s.contains('\r') {
                return Err(UsdError::new(
                    E_BAD_INPUT,
                    loc,
                    format!("资产路径 {s:?} 含 `@` 或换行，`@…@` 定界会被截断"),
                    "资产路径里不能出现 `@`；改名或转义后再导出",
                ));
            }
            format!("@{s}@")
        }
        AttrValue::Path(s) => format!("<{}>", check_target(s, loc)?),
        AttrValue::Numbers(ns) => {
            let mut items = Vec::with_capacity(ns.len());
            for n in ns {
                items.push(num(*n, loc)?);
            }
            if is_array {
                format!("[{}]", items.join(", "))
            } else if items.len() == 1 {
                items[0].clone()
            } else if items.is_empty() {
                // 空元组：回读是 `Numbers([])`，与写入前同形。
                "()".to_string()
            } else {
                format!("({})", items.join(", "))
            }
        }
        AttrValue::Strings(ss) => {
            let mut items = Vec::with_capacity(ss.len());
            for s in ss {
                items.push(quote(s, loc)?);
            }
            format!("[{}]", items.join(", "))
        }
        AttrValue::Paths(ps) => {
            let mut items = Vec::with_capacity(ps.len());
            for p in ps {
                items.push(format!("<{}>", check_target(p, loc)?));
            }
            format!("[{}]", items.join(", "))
        }
        AttrValue::Rows(rows) => {
            let mut items = Vec::with_capacity(rows.len());
            for r in rows {
                if r.is_empty() {
                    return Err(UsdError::new(
                        E_BAD_INPUT,
                        loc,
                        "行数组里出现空行：`.usda` 无对应写法，回读会补 0（静默改值）",
                        "删掉该行或补足分量；空行通常来自上游构造错误",
                    ));
                }
                let mut cols = Vec::with_capacity(r.len());
                for n in r {
                    cols.push(num(*n, loc)?);
                }
                // 恒带括号：行宽 1 也不省略，否则回读塌成 Numbers。
                items.push(format!("({})", cols.join(", ")));
            }
            if is_array {
                format!("[{}]", items.join(", "))
            } else {
                format!("({})", items.join(", "))
            }
        }
        AttrValue::Matrix(rows) => {
            let mut items = Vec::with_capacity(rows.len());
            for r in rows {
                if r.is_empty() {
                    return Err(UsdError::new(
                        E_BAD_INPUT,
                        loc,
                        "矩阵出现空行：回读会补 0（静默改值）",
                        "补足矩阵每行分量后重试",
                    ));
                }
                let mut cols = Vec::with_capacity(r.len());
                for n in r {
                    cols.push(num(*n, loc)?);
                }
                items.push(format!("({})", cols.join(", ")));
            }
            format!("({})", items.join(", "))
        }
        AttrValue::EmptyArray => "[]".to_string(),
    };
    Ok(Some(s))
}

/// 有限 f64 → 十进制最短往返文本（`Display` 契约；见模块文档）。
fn num(v: f64, loc: &str) -> Result<String, UsdError> {
    if !v.is_finite() {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("数值 {v} 不是有限 f64，`.usda` 文本无法表示（NaN/inf 会被回读成标识符）"),
            "把该值改成有限数；导出器不静默写 NaN（回读时那是一串错误几何）",
        ));
    }
    Ok(format!("{v}"))
}

/// 双引号字符串（只转义词法器认的 5 种；控制字符结构化拒绝）。
fn quote(s: &str, loc: &str) -> Result<String, UsdError> {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                return Err(UsdError::new(
                    E_BAD_INPUT,
                    loc,
                    format!(
                        "字符串含控制字符 U+{:04X}：`.usda` 双引号串只认 \\n \\t \\r 三种转义",
                        c as u32
                    ),
                    "导出前去掉该控制字符或改写为可打印文本；未知转义会被词法器原样保留，\
                     反而改变字符串内容",
                ));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(out)
}

/// 场景路径目标（`<…>` 内不得再有尖括号 / 换行）。
fn check_target<'a>(s: &'a str, loc: &str) -> Result<&'a str, UsdError> {
    if s.contains('<') || s.contains('>') || s.contains('\n') || s.contains('\r') {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("关系/路径目标 {s:?} 含 `<` `>` 或换行，`<…>` 定界会被截断"),
            "路径里不能出现尖括号；改写目标路径后重试",
        ));
    }
    Ok(s)
}

// ===========================================================================
//  标识符 / 元数据 / 版本
// ===========================================================================

/// 裸标识符校验（词法器会按 `[A-Za-z0-9_.:+-]` 切，首字符不能是数字）。
fn check_ident(s: &str, loc: &str, what: &str) -> Result<(), UsdError> {
    let ok_first = matches!(s.chars().next(), Some(c) if c.is_ascii_alphabetic() || c == '_');
    let ok_rest = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-' | '+'));
    if !ok_first || !ok_rest {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!(
                "{what} {s:?} 不是 `.usda` 的裸标识符（首字符须为字母或 `_`，其余限 \
                 [A-Za-z0-9_.:+-]）；写出去会被词法器切成别的 token"
            ),
            "改成合法标识符；含空格/中文/引号的名字应改走字符串或先做 ASCII 规范化",
        ));
    }
    Ok(())
}

/// 类型名校验（`point3f[]` 的 `[]` 是类型的一部分）。
fn check_type(t: &str, loc: &str) -> Result<(), UsdError> {
    let base = t.strip_suffix("[]").unwrap_or(t);
    check_ident(base, loc, "属性类型")
}

/// 层元数据值的规范文本：像数字就**规范化后**裸写（`1e5` → `100000`，
/// 否则回读再导出会变一次，L2 破），否则加引号（回读同值）。
fn meta_value(v: &str, loc: &str) -> Result<String, UsdError> {
    match bare_number(v) {
        Some(f) => num(f, loc),
        None => quote(v, loc),
    }
}

/// 可裸写的数字文本（能解析为有限 f64）→ 数值；否则 `None`。
fn bare_number(s: &str) -> Option<f64> {
    let numeric = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'));
    if !numeric {
        return None;
    }
    s.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// 层版本规范化：`1` / `1.0` 一律写 `1.0`（解析器把版本读成 `Num`，
/// `format!("{v}")` 会给 `1`；不规范化则首轮 `1.0`、次轮 `1`，L2 立刻破）。
fn canonical_version(v: &str) -> Result<String, UsdError> {
    let loc = "usd_export::export_layer（层版本）";
    let f: f64 = v.trim().parse().map_err(|_| {
        UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("层版本 {v:?} 不是数字，`.usda` 头要求 `#usda <版本号>`"),
            "用 \"1.0\"（当前唯一支持的 USD 文本版本）",
        )
    })?;
    if !f.is_finite() {
        return Err(UsdError::new(
            E_BAD_INPUT,
            loc,
            format!("层版本 {v:?} 不是有限数"),
            "用 \"1.0\"",
        ));
    }
    Ok(if f == 1.0 {
        "1.0".to_string()
    } else {
        format!("{f}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usd::{build_scene, parse_str, Mesh};

    // ── 语料（引擎自产：扁平网格 + 绑定 + 单位世界变换）──────────────────

    /// 四边形（保 polygon 用）+ 法线 + UV + 材质绑定。
    fn quad_mesh() -> Mesh {
        Mesh {
            path: "/geo/quad".to_string(),
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            face_counts: vec![4],
            face_indices: vec![0, 1, 2, 3],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            material: Some("/mats/blue".to_string()),
            world_transform: Mat4::identity(),
        }
    }

    /// 三角形，无法线/UV/材质（可选块全缺的最小网格）。
    fn tri_mesh() -> Mesh {
        Mesh {
            path: "/geo/tri".to_string(),
            points: vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
            face_counts: vec![3],
            face_indices: vec![0, 1, 2],
            normals: Vec::new(),
            uvs: Vec::new(),
            material: None,
            world_transform: Mat4::identity(),
        }
    }

    fn engine_scene() -> Scene {
        Scene {
            version: "1.0".to_string(),
            meshes: vec![quad_mesh(), tri_mesh()],
            non_mesh_prims: 0,
        }
    }

    fn blue() -> UsdMaterial {
        UsdMaterial {
            base_color: [0.1, 0.2, 0.3, 1.0],
            metalness: 0.0,
            roughness: 0.25,
            ior: 1.5,
            sheen_color: [0.0, 0.0, 0.0],
            sheen_roughness: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            alpha_cutoff: None,
        }
    }

    /// `/mats/blue` 被引用；`/mats/green` 未被引用（测 unused 记账）。
    fn materials() -> BTreeMap<String, UsdMaterial> {
        let mut m = BTreeMap::new();
        m.insert("/mats/blue".to_string(), blue());
        m.insert("/mats/green".to_string(), UsdMaterial::default());
        m
    }

    fn export_ok(scene: &Scene, mats: &BTreeMap<String, UsdMaterial>) -> (String, ExportReport) {
        export_scene(scene, mats, &ExportMeta::default()).expect("导出应成功")
    }

    fn layer_with(prims: Vec<Prim>) -> Layer {
        Layer {
            version: "1.0".to_string(),
            metadata: Vec::new(),
            prims,
            all_paths: Vec::new(),
        }
    }

    // ── R2：三级判据 ─────────────────────────────────────────────────

    #[test]
    fn export_is_byte_level_deterministic() {
        let (a, ra) = export_ok(&engine_scene(), &materials());
        let (b, rb) = export_ok(&engine_scene(), &materials());
        assert_eq!(a, b, "同场景两遍导出必须字节一致");
        assert_eq!(ra, rb);
        assert!(!a.contains('\r'), "C24 ③：只许 LF，不得出现 CR");
        assert!(a.ends_with('\n'), "文件必须以 LF 结尾");
        assert!(a.starts_with("#usda 1.0\n"), "层头必须规范为 `#usda 1.0`");
        assert!(
            std::str::from_utf8(a.as_bytes()).is_ok(),
            "输出必须是合法 UTF-8"
        );
    }

    /// **R2 L2（核心验收）**：`export(parse(export(S))) == export(S)`。
    #[test]
    fn export_import_export_is_byte_fixed_point() {
        let (t1, r1) = export_ok(&engine_scene(), &materials());
        assert_eq!(
            r1.unused_materials,
            vec!["/mats/green".to_string()],
            "未绑定的材质应只记账、不进文件"
        );

        let parsed = parse_str(&t1).expect("导出的文本必须能被现有解析器读回");
        assert!(
            parsed.report.skipped_time_samples.is_empty(),
            "导出器不写 timeSamples，回读也不该有跳过项"
        );
        let (t2, r2) = export_layer(&parsed.layer).expect("再导出应成功");
        assert_eq!(
            t1, t2,
            "R2 L2（字节不动点）不成立：规范形的再导出必须逐字节相同"
        );
        assert!(r2.is_empty(), "规范形的再导出不该出现新的降级：{r2:?}");
    }

    /// **R2 L1（语义）**：几何 / 法线 / UV / 绑定逐项相等。
    #[test]
    fn l1_scene_semantics_survive_roundtrip() {
        let src = engine_scene();
        let (t1, _) = export_ok(&src, &materials());
        let layer = parse_str(&t1).expect("回读").layer;
        let out = build_scene(&layer).expect("导出层应可构建场景");

        assert_eq!(out.meshes.len(), src.meshes.len());
        for (a, b) in src.meshes.iter().zip(out.meshes.iter()) {
            assert_eq!(a.points, b.points, "points 必须逐项相等：{}", a.path);
            assert_eq!(a.face_counts, b.face_counts, "面数（polygon）必须保持");
            assert_eq!(a.face_indices, b.face_indices, "面索引必须保持");
            assert_eq!(a.normals, b.normals, "法线必须保持");
            assert_eq!(a.uvs, b.uvs, "UV 必须保持");
        }
        // 绑定：源 key 的叶子 → 根下 Material prim；回读后目标非悬空。
        assert_eq!(out.meshes[0].material.as_deref(), Some("/root/blue"));
        assert_eq!(out.meshes[1].material, None);
        assert!(
            layer.all_paths.iter().any(|p| p == "/root/blue"),
            "绑定目标必须真实存在（否则回读即 PU1005）"
        );

        // 材质参数是属性级直射（解析器不求值，D-1）：读属性即验证映射。
        let mat = layer.prims[0]
            .children
            .iter()
            .find(|p| p.type_name == "Material")
            .expect("Material prim 必须存在");
        let shader = &mat.children[0];
        assert_eq!(
            shader.value("info:id"),
            Some(&AttrValue::Str("UsdPreviewSurface".to_string()))
        );
        assert_eq!(
            shader.value("inputs:diffuseColor").map(|v| v.as_numbers()),
            Some(vec![0.1, 0.2, 0.3])
        );
        assert_eq!(
            shader.value("inputs:opacity").and_then(|v| v.as_number()),
            Some(1.0)
        );
        assert_eq!(
            shader.value("inputs:roughness").and_then(|v| v.as_number()),
            Some(0.25)
        );
        // 世界变换：引擎侧单位阵 → 回读仍单位阵（导出没写 xformOp，不二次变换）。
        assert_eq!(out.meshes[0].world_transform, Mat4::identity());
    }

    /// **R2 L3（结构）**：路径集合 / 父子关系 / 关系目标 / 属性名序。
    #[test]
    fn l3_structure_is_flat_and_closed() {
        let (t1, _) = export_ok(&engine_scene(), &materials());
        let layer = parse_str(&t1).expect("回读").layer;
        assert_eq!(layer.prims.len(), 1, "单根层");
        let root = &layer.prims[0];
        assert_eq!(root.type_name, "Xform");
        assert_eq!(root.path, "/root");

        let paths: Vec<&str> = root.walk().iter().map(|p| p.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "/root",
                "/root/quad",
                "/root/tri",
                "/root/blue",
                "/root/blue/shader"
            ],
            "网格按场景序、材质按 key 升序；未绑定材质不写"
        );
        assert_eq!(
            root.children[0]
                .attributes
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "faceVertexCounts",
                "faceVertexIndices",
                "normals",
                "points",
                "primvars:st"
            ],
            "属性名序 = 规范顺序（faceVertexCounts/…/points/primvars:st）"
        );
        let targets: BTreeSet<String> = root
            .children
            .iter()
            .flat_map(|p| p.relationships.iter().flat_map(|r| r.targets.clone()))
            .collect();
        assert_eq!(
            targets,
            BTreeSet::from(["/root/blue".to_string()]),
            "关系目标集合必须封闭在层内"
        );
        // 材质子树：Material → Shader。
        let mat = &root.children[2];
        assert_eq!(mat.type_name, "Material");
        assert_eq!(mat.children.len(), 1);
        assert_eq!(mat.children[0].type_name, "Shader");
    }

    /// §4 钉死的浮点文本契约：`Display` → `parse::<f64>()` 逐位还原。
    #[test]
    fn f64_text_roundtrip_is_bitwise() {
        let corpus: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1e-3,
            1.0 / 3.0,
            1.234_567_890_123_456_7,
            1e16,
            1e-16,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            f64::from_bits(1),
            f64::from_bits(0x7fef_ffff_ffff_ffff),
        ];
        let mut prim = new_prim(PrimKind::Def, "Scope", "s".to_string(), "/s".to_string());
        prim.attributes.push(new_attr(
            "vals",
            "double[]",
            AttrValue::Numbers(corpus.clone()),
        ));
        let (text, rep) = export_layer(&layer_with(vec![prim])).expect("导出");
        assert!(rep.is_empty(), "{rep:?}");

        let parsed = parse_str(&text).expect("回读");
        let got = parsed.layer.prims[0]
            .value("vals")
            .expect("vals 必须回来")
            .as_numbers();
        assert_eq!(got.len(), corpus.len());
        for (a, b) in corpus.iter().zip(got.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "浮点文本往返丢位：{a:?} → {b:?}（文本 = {:?}）",
                format!("{a}")
            );
        }
        // 再导出不动点（文本口径本身也是幂等的）。
        let (t2, _) = export_layer(&parsed.layer).expect("再导出");
        assert_eq!(text, t2);
    }

    // ── 子集覆盖 ─────────────────────────────────────────────────────

    #[test]
    fn polygons_are_preserved_not_triangulated() {
        let (text, _) = export_ok(&engine_scene(), &materials());
        assert!(
            text.contains("int[] faceVertexCounts = [4]"),
            "四边形必须原样保 polygon：\n{text}"
        );
        let layer = parse_str(&text).expect("回读").layer;
        let scene = build_scene(&layer).expect("构建");
        assert_eq!(scene.meshes[0].face_counts, vec![4]);
        // 渲染侧才三角化：quad(4) + tri(3) → 2 + 1。
        assert_eq!(scene.meshes[0].triangulated().len(), 2);
        assert_eq!(scene.triangle_count(), 3);
    }

    #[test]
    fn material_inputs_are_field_mapped() {
        let mut mats = BTreeMap::new();
        mats.insert(
            "/mats/blue".to_string(),
            UsdMaterial {
                base_color: [0.05, 0.15, 0.25, 0.5],
                metalness: 0.75,
                roughness: 0.125,
                ior: 1.33,
                sheen_color: [0.0; 3],
                sheen_roughness: 0.0,
                clearcoat: 0.25,
                clearcoat_roughness: 0.375,
                alpha_cutoff: Some(0.25),
            },
        );
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![quad_mesh()],
            non_mesh_prims: 0,
        };
        let (text, rep) = export_scene(&scene, &mats, &ExportMeta::default()).expect("导出");
        assert!(rep.is_empty(), "{rep:?}");

        let layer = parse_str(&text).expect("回读").layer;
        let mat = &layer.prims[0].children[1];
        assert_eq!(mat.path, "/root/blue");
        assert_eq!(
            mat.value("outputs:surface.connect")
                .and_then(|v| v.as_str()),
            Some("/root/blue/shader.outputs:surface"),
            "材质输出必须连到自己的 Shader"
        );
        let s = &mat.children[0];
        assert_eq!(
            s.value("inputs:diffuseColor").map(|v| v.as_numbers()),
            Some(vec![0.05, 0.15, 0.25])
        );
        assert_eq!(
            s.value("inputs:opacity").and_then(|v| v.as_number()),
            Some(0.5)
        );
        assert_eq!(
            s.value("inputs:metallic").and_then(|v| v.as_number()),
            Some(0.75)
        );
        assert_eq!(
            s.value("inputs:ior").and_then(|v| v.as_number()),
            Some(1.33)
        );
        assert_eq!(
            s.value("inputs:clearcoat").and_then(|v| v.as_number()),
            Some(0.25)
        );
        assert_eq!(
            s.value("inputs:clearcoatRoughness")
                .and_then(|v| v.as_number()),
            Some(0.375)
        );
        assert_eq!(
            s.value("inputs:opacityThreshold")
                .and_then(|v| v.as_number()),
            Some(0.25)
        );
        // f32 语义：`0.05` 写出去再按 f32 读回，逐位不变。
        let back: f32 = "0.05".parse().expect("f32 文本");
        assert_eq!(back.to_bits(), 0.05f32.to_bits());
    }

    #[test]
    fn layer_metadata_and_root_prim_are_canonical() {
        let meta = ExportMeta {
            default_prim: "Scene".to_string(),
            up_axis: "Z".to_string(),
            meters_per_unit: 0.01,
        };
        let (text, rep) = export_scene(&engine_scene(), &materials(), &meta).expect("导出");
        assert!(
            rep.renamed_prims.is_empty() && rep.baked_transforms.is_empty(),
            "改根名不该产生改名/烘焙记账：{rep:?}"
        );

        let layer = parse_str(&text).expect("回读").layer;
        assert_eq!(
            layer.metadata,
            vec![
                ("defaultPrim".to_string(), "Scene".to_string()),
                ("upAxis".to_string(), "Z".to_string()),
                ("metersPerUnit".to_string(), "0.01".to_string()),
            ]
        );
        assert_eq!(layer.prims[0].name, "Scene");
        assert_eq!(layer.prims[0].path, "/Scene");
        assert_eq!(
            build_scene(&layer).expect("构建").meshes[0]
                .material
                .as_deref(),
            Some("/Scene/blue")
        );

        let (t2, _) = export_layer(&layer).expect("再导出");
        assert_eq!(text, t2, "改根名/轴/单位后 L2 仍须成立");
    }

    #[test]
    fn empty_scene_exports_a_valid_layer() {
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: Vec::new(),
            non_mesh_prims: 0,
        };
        let (text, rep) = export_scene(&scene, &BTreeMap::new(), &ExportMeta::default())
            .expect("空场景也应产出合法层");
        assert!(rep.is_empty(), "{rep:?}");
        let layer = parse_str(&text).expect("回读").layer;
        assert_eq!(layer.prims.len(), 1);
        assert!(layer.prims[0].children.is_empty());
        assert_eq!(build_scene(&layer).expect("构建").meshes.len(), 0);
        let (t2, _) = export_layer(&layer).expect("再导出");
        assert_eq!(text, t2);
    }

    // ── C20：降级 / 丢弃逐条记账 ─────────────────────────────────────

    #[test]
    fn sheen_is_reported_and_never_silently_dropped() {
        let mut mats = BTreeMap::new();
        let mut m = blue();
        m.sheen_color = [0.1, 0.2, 0.3];
        m.sheen_roughness = 0.4;
        mats.insert("/mats/blue".to_string(), m);
        let (text, rep) =
            export_scene(&engine_scene(), &mats, &ExportMeta::default()).expect("导出");
        assert_eq!(rep.dropped_sheen, vec!["/mats/blue".to_string()]);
        assert!(!text.contains("sheen"), "UsdPreviewSurface 无 sheen 输入");
    }

    #[test]
    fn non_vertex_attribute_lengths_are_reported() {
        let mut mesh = quad_mesh();
        // 源是 faceVarying：UV 已被导入器按面顶点展开（长度 = indices 长度）。
        mesh.uvs = vec![[0.0, 0.0]; 8];
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![mesh.clone()],
            non_mesh_prims: 0,
        };
        let (text, rep) = export_scene(&scene, &materials(), &ExportMeta::default()).expect("导出");
        assert_eq!(rep.relabeled_interpolation.len(), 1);
        assert!(rep.relabeled_interpolation[0].contains("primvars:st"));
        let layer = parse_str(&text).expect("回读").layer;
        let out = build_scene(&layer).expect("构建");
        assert_eq!(
            out.meshes[0].uvs, mesh.uvs,
            "数值必须逐个保住（丢的只是标签）"
        );
    }

    #[test]
    fn time_samples_and_variants_are_reported_not_written() {
        let src = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)] (
        timeSamples = {
            0: [(0,0,0), (1,0,0), (0,1,0)]
            1: [(0,0,1), (1,0,1), (0,1,1)]
        }
    )
    variantSet "look" = {
        "a" { def Material "x" {} }
    }
}
"#;
        let parsed = parse_str(src).expect("源可解析");
        let (text, rep) = export_layer(&parsed.layer).expect("导出");
        assert_eq!(rep.dropped_time_samples, vec!["/m@points".to_string()]);
        assert_eq!(rep.dropped_variant_sets, vec!["/m:look".to_string()]);
        assert!(!text.contains("timeSamples"), "动画第一版不导");
        assert!(!text.contains("variantSet"), "变体集第一版不导");
        // 默认帧仍是可渲染的（与导入侧同界，PU1007 同族）。
        assert_eq!(
            build_scene(&parsed.layer).expect("构建").meshes[0]
                .points
                .len(),
            3
        );
        let (t2, r2) = export_layer(&parse_str(&text).expect("回读").layer).expect("再导出");
        assert_eq!(text, t2);
        assert!(r2.is_empty(), "丢弃项只记一次，二轮不该复现：{r2:?}");
    }

    #[test]
    fn sublayers_are_dropped_and_reported() {
        let src = r#"#usda 1.0
(
    subLayers = [
        @./base.usda@
    ]
    upAxis = "Z"
)
def Xform "root"
{
}
"#;
        let parsed = parse_str(src).expect("源可解析");
        let (text, rep) = export_layer(&parsed.layer).expect("导出");
        assert_eq!(rep.dropped_layer_metadata, vec!["subLayers".to_string()]);
        assert!(!text.contains("subLayers"), "跨文件导出第一版不做");
        assert!(text.contains("upAxis = \"Z\""), "同块其它元数据必须保留");
        // 空 metadata 块不写（否则回读后二轮少一块，L2 破）。
        let p2 = parse_str(&text).expect("回读");
        assert_eq!(
            p2.layer.metadata,
            vec![("upAxis".to_string(), "Z".to_string())]
        );
        let (t2, _) = export_layer(&p2.layer).expect("再导出");
        assert_eq!(text, t2);
    }

    #[test]
    fn name_collisions_are_resolved_and_reported() {
        let mut scene = engine_scene();
        scene.meshes[0].path = "/a/x".to_string();
        scene.meshes[1].path = "/b/x".to_string();
        scene.meshes[0].material = Some("/m/x".to_string());
        scene.meshes[1].material = None;
        let mut mats = BTreeMap::new();
        mats.insert("/m/x".to_string(), blue());

        let (text, rep) = export_scene(&scene, &mats, &ExportMeta::default()).expect("导出");
        assert_eq!(
            rep.renamed_prims,
            vec![
                ("/b/x".to_string(), "/root/x_2".to_string()),
                ("/m/x".to_string(), "/root/x_3".to_string()),
            ],
            "重名必须确定性地加后缀并记账"
        );
        let layer = parse_str(&text).expect("回读").layer;
        let mut paths: Vec<&str> = layer.prims[0]
            .children
            .iter()
            .map(|p| p.path.as_str())
            .collect();
        paths.sort();
        assert_eq!(paths, vec!["/root/x", "/root/x_2", "/root/x_3"]);
        let scene2 = build_scene(&layer).expect("构建");
        assert_eq!(scene2.meshes[0].material.as_deref(), Some("/root/x_3"));
    }

    #[test]
    fn unused_materials_are_reported_and_not_written() {
        let (text, rep) = export_ok(&engine_scene(), &materials());
        assert_eq!(rep.unused_materials, vec!["/mats/green".to_string()]);
        assert!(!text.contains("green"), "未绑定的材质不塞进资产");
    }

    #[test]
    fn baked_world_transform_is_reported() {
        let mut mesh = tri_mesh();
        mesh.world_transform = Mat4 {
            m: [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![mesh],
            non_mesh_prims: 0,
        };
        let (text, rep) =
            export_scene(&scene, &BTreeMap::new(), &ExportMeta::default()).expect("导出");
        assert_eq!(rep.baked_transforms, vec!["/root/tri".to_string()]);
        assert!(!text.contains("xformOp"), "几何已世界化，不写 xform");
        let out = build_scene(&parse_str(&text).expect("回读").layer).expect("构建");
        assert_eq!(out.meshes[0].world_transform, Mat4::identity());
    }

    #[test]
    fn non_ascii_text_is_reported_but_written_verbatim() {
        let mut prim = new_prim(PrimKind::Def, "Scope", "s".to_string(), "/s".to_string());
        let mut a = new_attr("doc", "string", AttrValue::Str("材质".to_string()));
        a.custom = true;
        prim.attributes.push(a);
        let (text, rep) = export_layer(&layer_with(vec![prim])).expect("导出");
        assert_eq!(rep.non_ascii_text, vec!["prim /s 的属性 doc".to_string()]);
        assert!(text.contains("材质"), "导出写的是合法 UTF-8 原文");
    }

    /// 导入资产直导：未知属性 / over / Xform 链原样回写（R1：不重排不解释）。
    #[test]
    fn imported_layer_re_export_preserves_geometry_and_unknown_attrs() {
        let src = r#"#usda 1.0
(
    defaultPrim = "root"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "root"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
        custom string user:note = "a \"quoted\" \\ path\nnext"
        bool user:flag = true
        int[] user:empty = []
        double3 user:vec = (1.5, -2.5, 1e-3)
        asset user:tex = @./tex/a.png@
    }
    over "quad"
    {
        float custom:density = 2.5
    }
}
"#;
        let parsed = parse_str(src).expect("源可解析");
        let (text, rep) = export_layer(&parsed.layer).expect("导出");
        assert!(rep.is_empty(), "{rep:?}");

        let layer2 = parse_str(&text).expect("回读").layer;
        let s1 = build_scene(&parsed.layer).expect("源构建");
        let s2 = build_scene(&layer2).expect("回读构建");
        assert_eq!(s1.meshes.len(), s2.meshes.len());
        assert_eq!(
            s1.meshes[0].points, s2.meshes[0].points,
            "Xform 必须原样保留"
        );

        let mesh = &layer2.prims[0].children[0];
        assert_eq!(
            mesh.value("user:note").and_then(|v| v.as_str()),
            Some("a \"quoted\" \\ path\nnext"),
            "转义必须逐字还原"
        );
        assert_eq!(mesh.value("user:flag"), Some(&AttrValue::Bool(true)));
        assert_eq!(mesh.value("user:empty"), Some(&AttrValue::EmptyArray));
        assert_eq!(
            mesh.value("user:vec").map(|v| v.as_numbers()),
            Some(vec![1.5, -2.5, 0.001])
        );
        assert_eq!(
            mesh.value("user:tex").and_then(|v| v.as_str()),
            Some("./tex/a.png")
        );
        // `over` 与 `xformOpOrder` 保留（R1：over 叠加序不因导出而变）。
        let over = &layer2.prims[0].children[1];
        assert_eq!(over.kind, PrimKind::Over);
        assert_eq!(
            layer2.prims[0]
                .attribute("xformOpOrder")
                .map(|a| a.interpolation.clone()),
            Some(None)
        );
        assert_eq!(
            layer2.prims[0].xform_op_order,
            vec!["xformOp:translate".to_string()]
        );

        let (t2, _) = export_layer(&layer2).expect("再导出");
        assert_eq!(text, t2, "导入资产直导也必须字节不动点");
    }

    #[test]
    fn schema_comment_is_readable_and_outside_the_signature() {
        let (text, _) = export_ok(&engine_scene(), &materials());
        assert!(
            text.contains(&format!(
                "# prism-engine usda exporter schema {USDA_EXPORTER_SCHEMA_VERSION}"
            )),
            "导出器 schema 版本必须人读可见（V11 同精神）"
        );
        let layer = parse_str(&text).expect("回读").layer;
        assert_eq!(
            layer.metadata.len(),
            3,
            "注释行不进元数据（不参与往返签名）"
        );
        assert!(layer.metadata.iter().all(|(k, _)| k != "prism-engine"));
    }

    // ── C18：失败结构化、不静默 ─────────────────────────────────────

    #[test]
    fn non_finite_values_are_structured_errors() {
        let mut mesh = tri_mesh();
        mesh.points[0][0] = f64::NAN;
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![mesh],
            non_mesh_prims: 0,
        };
        let e = export_scene(&scene, &BTreeMap::new(), &ExportMeta::default())
            .expect_err("NaN 必须结构化拒绝");
        assert_eq!(e.code, "PU1013");
        assert!(
            e.location.contains("points"),
            "三段式【在哪】：{}",
            e.location
        );
        assert!(
            e.problem.contains("有限"),
            "三段式【什么事】：{}",
            e.problem
        );
        assert!(!e.fix.is_empty(), "三段式【怎么改】不得为空");
    }

    #[test]
    fn missing_material_params_is_a_structured_error() {
        let e = export_scene(&engine_scene(), &BTreeMap::new(), &ExportMeta::default())
            .expect_err("缺材质参数必须拒绝（否则写出悬空 rel）");
        assert_eq!(e.code, "PU1013");
        assert!(e.problem.contains("/mats/blue"), "{}", e.problem);
        assert!(
            e.fix.contains("export_layer"),
            "建议段应指出导入资产走直导：{}",
            e.fix
        );
    }

    #[test]
    fn invalid_inputs_are_structured_errors() {
        // prim 路径与树位置不一致 → 写出去会改关系目标。
        let prim = new_prim(
            PrimKind::Def,
            "Scope",
            "a".to_string(),
            "/wrong".to_string(),
        );
        let e = export_layer(&layer_with(vec![prim])).expect_err("路径错位必须拒绝");
        assert_eq!(e.code, "PU1013");
        assert!(e.problem.contains("/wrong"), "{}", e.problem);

        // 非法裸标识符（属性名含空格）。
        let mut prim = new_prim(PrimKind::Def, "Scope", "s".to_string(), "/s".to_string());
        let mut a = new_attr("bad name", "int", AttrValue::Number(1.0));
        a.custom = true;
        prim.attributes.push(a);
        let e = export_layer(&layer_with(vec![prim])).expect_err("非法属性名必须拒绝");
        assert_eq!(e.code, "PU1013");

        // 层元数据非法。
        let bad = ExportMeta {
            up_axis: "X".to_string(),
            ..ExportMeta::default()
        };
        let e = export_scene(&engine_scene(), &materials(), &bad).expect_err("upAxis=X 非法");
        assert_eq!(e.code, "PU1013");

        let bad = ExportMeta {
            default_prim: "根".to_string(),
            ..ExportMeta::default()
        };
        let e = export_scene(&engine_scene(), &materials(), &bad).expect_err("非 ASCII 根名非法");
        assert_eq!(e.code, "PU1013");

        // 非有限材质参数。
        let mut mats = materials();
        mats.get_mut("/mats/blue").expect("blue").roughness = f32::INFINITY;
        let e = export_scene(&engine_scene(), &mats, &ExportMeta::default())
            .expect_err("inf 材质参数必须拒绝");
        assert_eq!(e.code, "PU1013");
        assert!(e.location.contains("材质"), "{}", e.location);
    }

    #[test]
    fn inconsistent_topology_is_a_structured_error() {
        let mut mesh = quad_mesh();
        mesh.face_indices.push(9); // counts 之和 ≠ indices 长度 + 越界
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![mesh],
            non_mesh_prims: 0,
        };
        let e = export_scene(&scene, &materials(), &ExportMeta::default())
            .expect_err("拓扑不自洽必须拒绝（不写半份资产）");
        assert_eq!(e.code, "PU1013");
        assert!(e.problem.contains("faceVertexCounts"), "{}", e.problem);
    }

    /// 层元数据 / 版本号的规范化：数字裸写且**重排成最短形式**（否则
    /// `1e5` 首轮原样、次轮 `100000`，L2 破）。
    #[test]
    fn metadata_and_version_are_canonicalized() {
        let layer = Layer {
            version: "1".to_string(),
            metadata: vec![
                ("metersPerUnit".to_string(), "1e5".to_string()),
                ("custom:name".to_string(), "a b".to_string()),
            ],
            prims: Vec::new(),
            all_paths: Vec::new(),
        };
        let (text, rep) = export_layer(&layer).expect("导出");
        assert!(rep.is_empty(), "{rep:?}");
        assert!(
            text.starts_with("#usda 1.0\n"),
            "版本必须规范为 1.0：\n{text}"
        );
        assert!(text.contains("metersPerUnit = 100000"), "\n{text}");
        assert!(text.contains("custom:name = \"a b\""), "\n{text}");

        let p2 = parse_str(&text).expect("回读");
        assert_eq!(
            p2.layer.metadata,
            vec![
                ("metersPerUnit".to_string(), "100000".to_string()),
                ("custom:name".to_string(), "a b".to_string()),
            ]
        );
        let (t2, _) = export_layer(&p2.layer).expect("再导出");
        assert_eq!(text, t2);
    }

    /// 多顶层 prim + 无类型 / 空体的 `over`：布局与 L2 一并钉死。
    #[test]
    fn multiple_top_level_prims_roundtrip() {
        let mut a = new_prim(PrimKind::Def, "Scope", "a".to_string(), "/a".to_string());
        a.attributes
            .push(new_attr("x", "int", AttrValue::Number(1.0)));
        let b = new_prim(PrimKind::Over, "", "b".to_string(), "/b".to_string());
        let (text, rep) = export_layer(&layer_with(vec![a, b])).expect("导出");
        assert!(rep.is_empty(), "{rep:?}");
        let p2 = parse_str(&text).expect("回读");
        assert_eq!(p2.layer.prims.len(), 2);
        assert_eq!(p2.layer.prims[1].kind, PrimKind::Over);
        assert!(p2.layer.prims[1].type_name.is_empty());
        let (t2, _) = export_layer(&p2.layer).expect("再导出");
        assert_eq!(text, t2);
    }

    // ── 规范形冻结（golden）：布局改动会显式破这个测试 ────────────────

    #[test]
    fn canonical_text_is_frozen() {
        let scene = Scene {
            version: "1.0".to_string(),
            meshes: vec![Mesh {
                path: "/geo/tri".to_string(),
                points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                face_counts: vec![3],
                face_indices: vec![0, 1, 2],
                normals: Vec::new(),
                uvs: Vec::new(),
                material: Some("/mat/red".to_string()),
                world_transform: Mat4::identity(),
            }],
            non_mesh_prims: 0,
        };
        let mut mats = BTreeMap::new();
        mats.insert(
            "/mat/red".to_string(),
            UsdMaterial {
                base_color: [1.0, 0.0, 0.0, 1.0],
                metalness: 0.0,
                roughness: 0.5,
                ior: 1.5,
                sheen_color: [0.0; 3],
                sheen_roughness: 0.0,
                clearcoat: 0.0,
                clearcoat_roughness: 0.0,
                alpha_cutoff: None,
            },
        );
        let (text, _) = export_scene(&scene, &mats, &ExportMeta::default()).expect("导出");
        let expect = r#"#usda 1.0
(
    defaultPrim = "root"
    upAxis = "Y"
    metersPerUnit = 1
)

# prism-engine usda exporter schema 1

def Xform "root"
{
    def Mesh "tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        rel material:binding = </root/red>
    }

    def Material "red"
    {
        token outputs:surface.connect = </root/red/shader.outputs:surface>

        def Shader "shader"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0, 0)
            float inputs:opacity = 1
            float inputs:metallic = 0
            float inputs:roughness = 0.5
            float inputs:ior = 1.5
            float inputs:clearcoat = 0
            float inputs:clearcoatRoughness = 0
        }
    }
}
"#;
        assert_eq!(text, expect, "规范形布局必须冻结（设计 §7 C15 纪律）");
    }
}
