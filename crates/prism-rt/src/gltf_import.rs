//! glTF / 通用实例化场景 → [`Scene`] 的桥（W3 / A3）。
//!
//! # 为什么单独一个模块
//!
//! `prism-geometry` 只负责"把文件读成中性数据"（`GltfScene`：共享几何 +
//! 实例表），`prism-rt` 负责"把中性数据装进渲染场景"。两边不互相依赖
//! （`prism-geometry` 不依赖 `prism-rt`，反向依赖会成环），桥只能落在
//! `prism-rt` 这一侧。
//!
//! # 它在做什么（一句话）
//!
//! **同一份几何只注册一次**（`Scene::instance_geoms`），每个节点一条实例记录
//! （`Scene::instances`），几何内存因此与节点数无关。
//!
//! ```
//! // 3 个节点引用同一个 mesh：几何 1 份 + 实例 3 条（不是 3 份顶点）。
//! let mut scene = prism_rt::integrator::Scene::default();
//! // let (g, n) = scene.add_gltf_scene(&gltf_scene, |m| lambert_from(m))?;
//! ```

use prism_geometry::gltf::{GltfMesh, GltfScene};
use prism_math::Vec3;

use crate::instance::{Instance, INSTANCE_MAT_FROM_GEOM};
use crate::integrator::{MaterialKind, Scene};

/// glTF 场景的**默认**材质映射（近似：`baseColorFactor` → Lambert 反照率）。
///
/// # 这是近似，不是 PBR 导入
///
/// glTF 的 `metallicFactor` / `roughnessFactor` / clearcoat / sheen 到
/// `Bsdf` 的映射属材质管线（A1/材质波次），不归 W3。这里只给一个**能出图**
/// 的兜底，避免桥接层逼调用方先实现完整材质映射。需要精确材质的调用方把
/// 自己的闭包传给 [`Scene::add_gltf_scene`]。
///
/// `baseColorFactor` 的 alpha 被丢弃（本引擎的 Lambert 不透明）。
pub fn gltf_default_material(m: &GltfMesh) -> MaterialKind {
    match &m.material {
        Some(p) => MaterialKind::Lambert {
            albedo: Vec3::new(p.base_color[0], p.base_color[1], p.base_color[2]),
        },
        None => MaterialKind::Lambert { albedo: Vec3::splat(0.72) },
    }
}

/// 桥接统计（调用方用来打日志 / 断言"几何与实例解耦"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfImportStats {
    /// 注册的共享几何条数（= 资产里 `(mesh, primitive)` 的去重数）。
    pub geometries: usize,
    /// 实例条数（= 资产里引用 mesh 的节点数 × primitive 数）。
    pub instances: usize,
    /// 几何三角形总数（与 `instances` **无关**）。
    pub triangles: usize,
    /// 若走展平，会多出多少个几何副本（= `instances - geometries`）。
    pub saved_copies: usize,
}

impl Scene {
    /// 把 [`GltfScene`] 装进场景：共享几何 + 实例记录（**一次** TLAS 重建）。
    ///
    /// * 几何：逐个 `GltfMesh` → `Scene::instance_geoms`（BLAS 建一次）；
    /// * 实例：逐个 `GltfInstance` → `Instance::from_affine`（逆矩阵在此算一次）；
    /// * 材质：由 `material_of` 决定，`mat_id` 用哨兵（取自共享几何）。
    ///
    /// # 错误
    ///
    /// 实例的仿射退化（行列式为 0 / 非有限）⇒ `Err`：这种实例的逆不存在，
    /// 静默丢弃会让"少画一个节点"变成最难二分的一类 bug。
    pub fn add_gltf_scene(
        &mut self,
        gltf: &GltfScene,
        material_of: impl Fn(&GltfMesh) -> MaterialKind,
    ) -> Result<GltfImportStats, String> {
        let mut geom_ids: Vec<u32> = Vec::with_capacity(gltf.meshes.len());
        let mut triangles = 0usize;
        for (i, m) in gltf.meshes.iter().enumerate() {
            let positions: Vec<Vec3> = m.positions.clone();
            let indices: Vec<u32> = m.indices.clone();
            if indices.len() % 3 != 0 {
                return Err(format!(
                    "gltf.meshes[{i}]：索引数 {} 不是 3 的倍数（三角流被破坏）",
                    indices.len()
                ));
            }
            triangles += indices.len() / 3;
            geom_ids.push(self.add_instance_geom(positions, indices, material_of(m)));
        }
        let mut insts: Vec<Instance> = Vec::with_capacity(gltf.instances.len());
        for (k, gi) in gltf.instances.iter().enumerate() {
            let Some(&gid) = geom_ids.get(gi.mesh as usize) else {
                return Err(format!(
                    "gltf.instances[{k}]（{}）：mesh 句柄 {} 越界（共享几何 {} 条）",
                    gi.path,
                    gi.mesh,
                    geom_ids.len()
                ));
            };
            let Some(inst) = Instance::from_affine(gi.affine, gid, INSTANCE_MAT_FROM_GEOM) else {
                return Err(format!(
                    "gltf.instances[{k}]（{}）的变换不可逆（行列式为 0 或含非有限分量）：\
                     affine = {:?}。实例化要求可逆仿射——请在导出端修掉零缩放",
                    gi.path, gi.affine
                ));
            };
            insts.push(inst);
        }
        let n_inst = insts.len();
        self.add_instances(insts);
        Ok(GltfImportStats {
            geometries: geom_ids.len(),
            instances: n_inst,
            triangles,
            saved_copies: n_inst.saturating_sub(geom_ids.len()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小的 glTF 文档生成器：一个立方体 mesh、N 个节点各引用它。
    fn cube_gltf(nodes: usize) -> String {
        let mut node_list = String::new();
        for i in 0..nodes {
            if i > 0 {
                node_list.push(',');
            }
            node_list.push_str(&format!(
                r#"{{"mesh":0,"translation":[{}.0,0.0,0.0]}}"#,
                i
            ));
        }
        let scene_nodes: Vec<String> = (0..nodes).map(|i| i.to_string()).collect();
        format!(
            r#"{{"asset":{{"version":"2.0"}},
                 "scenes":[{{"nodes":[{}]}}],"scene":0,
                 "nodes":[{}],
                 "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],
                 "accessors":[
                   {{"bufferView":0,"componentType":5126,"count":8,"type":"VEC3"}},
                   {{"bufferView":1,"componentType":5123,"count":36,"type":"SCALAR"}}
                 ],
                 "bufferViews":[
                   {{"buffer":0,"byteOffset":0,"byteLength":96}},
                   {{"buffer":0,"byteOffset":96,"byteLength":72}}
                 ],
                 "buffers":[{{"byteLength":168}}]}}"#,
            scene_nodes.join(","),
            node_list,
        )
    }

    /// 拼 bin：8 个 f32×3 顶点 + 36 个 u16 索引。
    fn cube_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        for &z in &[-1.0f32, 1.0] {
            for &y in &[-1.0f32, 1.0] {
                for &x in &[-1.0f32, 1.0] {
                    for c in [x, y, z] {
                        bin.extend_from_slice(&c.to_le_bytes());
                    }
                }
            }
        }
        let idx: [u16; 36] = [
            0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4,
            1, 5, 7, 1, 7, 3,
        ];
        for i in idx {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin
    }

    /// glTF 导入**保实例**：3 个节点引用同一个 mesh ⇒ 几何 1 份 + 实例 3 条。
    #[test]
    fn gltf_import_preserves_instances() {
        let json = cube_gltf(3);
        let bin = cube_bin();
        let (gs, warn) = prism_geometry::gltf::parse_gltf_scene(&json, &bin).expect("应解析成功");
        assert_eq!(gs.meshes.len(), 1, "同一 mesh 应只保留一份几何");
        assert_eq!(gs.instances.len(), 3, "3 个节点应产出 3 条实例");
        assert_eq!(gs.saved_mesh_copies(), 2, "展平会多出 2 份副本");
        for (k, inst) in gs.instances.iter().enumerate() {
            assert_eq!(inst.mesh, 0);
            assert_eq!(inst.affine[3], k as f32, "平移 x 应为节点序号");
        }
        assert!(warn.is_empty(), "最小资产不该有 warning：{warn:?}");

        // 装进 Scene：几何 1 份、实例 3 条、三角形总数与实例数无关。
        let mut scene = Scene::default();
        let stats = scene
            .add_gltf_scene(&gs, gltf_default_material)
            .expect("桥接应成功");
        assert_eq!(stats.geometries, 1);
        assert_eq!(stats.instances, 3);
        assert_eq!(stats.triangles, 12);
        assert_eq!(stats.saved_copies, 2);
        assert_eq!(scene.instance_geom_count(), 1);
        assert_eq!(scene.instance_count(), 3);
        assert_eq!(scene.instance_geom_triangle_count(), 12);
        assert_eq!(scene.triangle_count(), 0, "原型不得进 meshes");
        assert_eq!(
            scene.instance_bvh.as_ref().map(|b| b.prim_count()),
            Some(3),
            "TLAS 叶子数 = 实例数"
        );
        scene.assert_layout();
        // 三个节点分别在 x = 0/1/2：从 -z 侧打一条射线必须命中 x=0 那个。
        let h = scene
            .intersect(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0e30,
                None,
            )
            .expect("应命中第一个立方体");
        assert!((h.t - 4.0).abs() < 1e-5, "t = {}（应 ≈ 4，即 z=−1 面）", h.t);
    }

    /// 展平兼容入口与"保实例"入口的几何是同一份数据（逐位）。
    #[test]
    fn gltf_flatten_matches_pre_w3_bitwise() {
        let json = cube_gltf(3);
        let bin = cube_bin();
        let (gs, _) = prism_geometry::gltf::parse_gltf_scene(&json, &bin).unwrap();
        let flat_via_wrapper = prism_geometry::gltf::parse_gltf(&json, &bin).unwrap().0;
        let flat = gs.flatten();
        assert_eq!(flat.len(), flat_via_wrapper.len());
        assert_eq!(flat.len(), 3, "展平后每个节点一份");
        for (a, b) in flat.iter().zip(flat_via_wrapper.iter()) {
            assert_eq!(a.positions.len(), b.positions.len());
            for (p, q) in a.positions.iter().zip(b.positions.iter()) {
                assert_eq!(p.x.to_bits(), q.x.to_bits(), "展平顶点不逐位");
                assert_eq!(p.y.to_bits(), q.y.to_bits());
                assert_eq!(p.z.to_bits(), q.z.to_bits());
            }
            assert_eq!(a.indices, b.indices);
        }
        // 第 k 份的顶点整体平移 +k（证明变换真的烘进去了）。
        for (k, m) in flat.iter().enumerate() {
            let p = m.positions[0].x;
            let q = gs.meshes[0].positions[0].x;
            assert!((p - q - k as f32).abs() < 1e-6, "第 {k} 份的 x 平移不对：{p} vs {q}");
        }
    }

    /// 退化实例（零缩放）必须报错而不是静默丢弃。
    #[test]
    fn gltf_degenerate_instance_is_rejected() {
        let json = cube_gltf(1);
        let bin = cube_bin();
        let (mut gs, _) = prism_geometry::gltf::parse_gltf_scene(&json, &bin).unwrap();
        gs.instances[0].affine[0] = 0.0; // 压扁 x
        gs.instances[0].affine[1] = 0.0;
        gs.instances[0].affine[2] = 0.0;
        let mut scene = Scene::default();
        let err = scene
            .add_gltf_scene(&gs, gltf_default_material)
            .expect_err("零缩放必须被拒");
        assert!(err.contains("不可逆"), "实际：{err}");
    }
}
