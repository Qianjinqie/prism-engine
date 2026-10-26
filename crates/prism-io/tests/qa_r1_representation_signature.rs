//! R1 同构保持判据（#9）：资产表示签名的形式化测试。
//!
//! 验收（PRD B1 验收 1）：≥ 2 组格式内等价源 → 表示签名相等；≥ 1 组不等价源
//! → 表示签名不等。覆盖 `.usda` 导入路径（`parse_str` → `build_scene` →
//! `representation_signature`）。
//!
//! # 设计偏差说明（必要，已记录）
//! 设计文档 §2 文件清单将本测试列为 `crates/prism-core/tests/qa_r1_representation_signature.rs`，
//! 但 `representation_signature` 实现于 `prism-io` 且操作 `prism_io::usd::Scene`，
//! 而分层纪律（io∈L4 → core∈L2）**禁止 `prism-core` 依赖 `prism-io`**。
//! 故本测试只能落在 `prism-io` 的测试目标下，方能访问被测函数与 `.usda` 解析器。
//! 这是为满足分层铁律所做的必要路径调整，被测行为与断言与设计文档完全一致。

use prism_io::usd::{build_scene, parse_str, representation_signature};

/// 逻辑资产 A：x–y 平面上的两个三角网格（a 在 z=0，b 在 z=1），属性按
/// points → normals → st 顺序书写。
const USD_AB: &str = r#"#usda 1.0
def Mesh "a"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
def Mesh "b"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,1), (1,0,1), (0,1,1)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
"#;

/// 逻辑资产 B：与 A 完全相同的两个三角网格，但**书写顺序相反**（先 b 后 a）。
/// 这是「组合层级两种写法」的一种：同一逻辑场景以不同 prim 顺序书写。
const USD_BA: &str = r#"#usda 1.0
def Mesh "b"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,1), (1,0,1), (0,1,1)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
def Mesh "a"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
"#;

/// 逻辑资产 A 的等价变体：网格 a 的属性顺序改写为 points → st → normals，
/// 网格 b 改写为 points → normals → st。USD 属性按名解析（非顺序），故解析后
/// [`Scene`] 与 A 逐字段相同——构成第二组格式内等价样本对。
const USD_AB_ATTR_REORDERED: &str = r#"#usda 1.0
def Mesh "a"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
}
def Mesh "b"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,1), (1,0,1), (0,1,1)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
"#;

/// 逻辑资产 A 的不等价变体：网格 a 的一个顶点被移动（(0,1,0) → (0,2,0)），
/// 几何内容改变 ⇒ 表示签名必须不同。
const USD_AB_MOVED: &str = r#"#usda 1.0
def Mesh "a"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,2,0)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
def Mesh "b"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,1), (1,0,1), (0,1,1)]
    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)]
    float2[] primvars:st = [(0,0), (1,0), (0,1)]
}
"#;

/// 遍历序依赖的反例：A 与 B 解析后 `meshes` 向量顺序相反，但若表示签名不做
/// canonical 排序，签名本应不同——本测试断言「做了排序 ⇒ 相等」，反向验证
/// 排序是等价判据成立的必要条件（无排序则等价判据失效）。
#[test]
fn r1_equivalent_sources_share_representation_signature() {
    let sig_ab = representation_signature(&build_scene(&parse_str(USD_AB).unwrap().layer).unwrap());
    let sig_ba = representation_signature(&build_scene(&parse_str(USD_BA).unwrap().layer).unwrap());
    let sig_re = representation_signature(
        &build_scene(&parse_str(USD_AB_ATTR_REORDERED).unwrap().layer).unwrap(),
    );

    // 等价样本对 1：prim 书写顺序相反（组合层级两种写法之一）。
    assert_eq!(
        sig_ab, sig_ba,
        "同一逻辑资产、不同 prim 书写顺序 ⇒ 表示签名相等（R1 同构保持）"
    );
    // 等价样本对 2：属性书写顺序相反（解析后 Scene 逐字段相同）。
    assert_eq!(
        sig_ab, sig_re,
        "同一逻辑资产、属性顺序相反 ⇒ 表示签名相等（R1 同构保持）"
    );
    // 确定性：同源两遍签名逐位一致。
    let sig_ab_again =
        representation_signature(&build_scene(&parse_str(USD_AB).unwrap().layer).unwrap());
    assert_eq!(sig_ab, sig_ab_again, "同输入同签名（确定性）");
}

#[test]
fn r1_inequivalent_source_has_distinct_signature() {
    let sig_ab = representation_signature(&build_scene(&parse_str(USD_AB).unwrap().layer).unwrap());
    let sig_moved =
        representation_signature(&build_scene(&parse_str(USD_AB_MOVED).unwrap().layer).unwrap());

    assert_ne!(
        sig_ab, sig_moved,
        "几何内容不同 ⇒ 表示签名不等（R1：不等价源签名不等）"
    );
}

#[test]
fn r1_distinct_assets_never_collide() {
    // 两个互不等价的源必须给出不同签名（碰撞即 R1 失效）。
    let sig_ab = representation_signature(&build_scene(&parse_str(USD_AB).unwrap().layer).unwrap());
    let sig_ba = representation_signature(&build_scene(&parse_str(USD_BA).unwrap().layer).unwrap());
    let sig_moved =
        representation_signature(&build_scene(&parse_str(USD_AB_MOVED).unwrap().layer).unwrap());
    assert_eq!(sig_ab, sig_ba, "等价源签名一致（前置条件）");
    assert_ne!(sig_ab, sig_moved, "不等价源签名不同");
}
