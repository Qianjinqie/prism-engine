//! T0 的 DSL 侧判据：五个附加能力块能声明、旧语法不受影响。
//!
//! # 判据
//!
//! | # | 判据 |
//! |---|---|
//! | D1 | `examples/t0_media_mesh.prism` 能编译，且 `attach` 六项各有内容（纹理 1 / 网格 1 / 实例原型 1 / 实例 2 / 介质 1 / 毛发 8） |
//! | D2 | **向后兼容**：不含这五个块的旧文件（`examples/spin_red_ball.prism` 与 `w2d_showcase.prism`）编译后 `attach` 全空 |
//! | D3 | 逐个块的最小写法都能单独编译（`texture` / `mesh` / `media` / `hair` / `instance`） |
//! | D4 | 非法输入报三段式错误（不是 panic）：反向盒、未声明纹理、未声明网格 |
//!
//! # 为什么这些断言在 DSL 侧而不在 render 侧
//!
//! render 侧的 P1/P2/P3 管的是"翻译层有没有把它们搬过去"；本文件管的是
//! "语言能不能表达"。两件事分开测，缺一条都会出现"字段加了但 DSL 写不出来"
//! 或"DSL 能写但渲染忽略"的半截状态。

use prism_dsl::compile_v2;

/// 仓库根目录下的示例文件（`CARGO_MANIFEST_DIR` = `crates/prism-dsl`）。
fn read_example(name: &str) -> String {
    let p: std::path::PathBuf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("读不到示例 {}：{e}", p.display()))
}

/// 编译一段 DSL 源码并返回第一个场景的附加能力。
fn attach_of(src: &str) -> prism_core::SceneAttach {
    let out = compile_v2(src).unwrap_or_else(|e| panic!("编译失败：{e}"));
    out.scenes
        .first()
        .expect("至少一个场景")
        .attach
        .clone()
}

/// 最小文档骨架（只含 v2 必需块）。
fn wrap(body: &str) -> String {
    format!(
        "// prism-dsl 1\nmovie \"t\" {{ duration 1s fps 4 resolution 32x24 }}\nscene \"s\" {{ duration 1s\n{body}\n}}\n"
    )
}

/// **D1：示例文件能编译，六项能力都真的进了 `attach`。**
#[test]
fn d1_example_declares_all_capabilities() {
    let src: String = read_example("t0_media_mesh.prism");
    let a: prism_core::SceneAttach = attach_of(&src);
    assert_eq!(a.textures.len(), 1, "纹理块数量");
    assert_eq!(a.textures[0].width, 8, "纹理边长");
    assert_eq!(a.meshes.len(), 1, "网格块数量");
    assert_eq!(a.meshes[0].tri_count(), 2, "平面网格是 2 个三角形");
    assert!(
        a.meshes[0].displacement.is_some(),
        "示例的位移槽必须挂上（这是 G-10 运行期通路的 DSL 入口）"
    );
    assert_eq!(a.meshes[0].displacement.unwrap().amplitude, 0.12);
    assert_eq!(a.instance_geoms.len(), 1, "两份实例共享一个原型");
    assert_eq!(a.instances.len(), 2, "实例条数");
    assert_eq!(a.instances[0].geom_id, 0);
    assert_eq!(a.media.len(), 1, "介质块数量");
    assert!((a.media[0].sigma_t - 0.35).abs() < 1e-6);
    assert_eq!(a.media[0].priority, 1);
    assert_eq!(a.strands.len(), 8, "毛发股数（count 8）");
    assert_eq!(a.strands[0].points.len(), 5, "segments 4 ⇒ 5 个点");
    println!(
        "[t0-d1] textures={} meshes={} geoms={} instances={} media={} strands={}",
        a.textures.len(),
        a.meshes.len(),
        a.instance_geoms.len(),
        a.instances.len(),
        a.media.len(),
        a.strands.len()
    );
}

/// **D2：不含附加能力块的旧文件 ⇒ `attach` 全空（逐字节同帧的结构前提）。**
///
/// 两个旧文件走**两条不同的编译入口**，两条都要验：
/// - `spin_red_ball.prism` 是 v1 块语法 ⇒ [`prism_dsl::compile`]；
/// - `w2d_showcase.prism` 是 v2 全块语法 ⇒ [`prism_dsl::compile_v2`]。
#[test]
fn d2_legacy_files_have_empty_attach() {
    // v1
    let src: String = read_example("spin_red_ball.prism");
    let v1 = prism_dsl::compile(&src)
        .unwrap_or_else(|e| panic!("v1 旧示例必须照常解析，却失败：{e}"));
    assert!(
        v1.scene.attach.is_empty(),
        "v1 旧示例的 Scene::attach 必须全空（v1 文档模型没有附加能力块）"
    );

    // v2
    let src: String = read_example("w2d_showcase.prism");
    let out = compile_v2(&src)
        .unwrap_or_else(|e| panic!("v2 旧示例必须照常解析，却失败：{e}"));
    assert!(!out.scenes.is_empty(), "至少要有一个场景");
    for s in &out.scenes {
        assert!(
            s.attach.is_empty(),
            "旧示例 w2d_showcase 的场景 '{}' 不该有附加能力（会破坏逐字节同帧）",
            s.name
        );
    }
}

/// **D3：五个块各自的最小写法都能编译。**
#[test]
fn d3_each_block_parses_standalone() {
    let cases: [(&str, &str); 5] = [
        ("texture", "texture t { kind gradient size 4 }"),
        ("mesh", "mesh m { kind box size 1 }"),
        ("media", "media f { min (-1,-1,-1) max (1,1,1) }"),
        ("hair", "hair h { count 2 from (0,0,0) dir (0,1,0) length 0.5 radius 0.01 }"),
        ("mesh+instance", "mesh m { kind plane size 1 }\ninstance i { geom \"m\" }"),
    ];
    for (name, body) in cases {
        let src: String = wrap(body);
        let a: prism_core::SceneAttach = attach_of(&src);
        assert!(!a.is_empty(), "{name} 块必须产出附加能力");
    }
    // 无名块也能解析（名缺省同种类）。
    let a: prism_core::SceneAttach = attach_of(&wrap("texture { kind checker size 2 }"));
    assert_eq!(a.textures.len(), 1);
}

/// **D4：非法输入报三段式错误（P10xx / P30xx），不 panic。**
#[test]
fn d4_invalid_blocks_report_p_codes() {
    // 反向盒：min > max
    let err = compile_v2(&wrap("media f { min (2,2,2) max (-2,-2,-2) }")).unwrap_err();
    assert!(err.to_string().contains("P1011"), "反向盒应报 P1011，实际：{err}");

    // 未声明的纹理名
    let err = compile_v2(&wrap(
        "mesh m { kind plane displacement { texture \"nope\" amplitude 0.1 } }",
    ))
    .unwrap_err();
    assert!(err.to_string().contains("P3006"), "未声明纹理应报 P3006，实际：{err}");

    // 未声明的网格名
    let err = compile_v2(&wrap("instance i { geom \"nope\" }")).unwrap_err();
    assert!(err.to_string().contains("P3006"), "未声明网格应报 P3006，实际：{err}");

    // 缺必填字段
    let err = compile_v2(&wrap("media f { sigma_t 0.5 }")).unwrap_err();
    assert!(err.to_string().contains("P1003"), "缺 min/max 应报 P1003，实际：{err}");
}

/// **D5：未知条目的报错文案已包含五个新块名**（旧文件里的拼写错误仍能定位）。
#[test]
fn d5_unknown_item_message_lists_new_kinds() {
    let err = compile_v2(&wrap("nonsense { x 1 }")).unwrap_err();
    let msg: String = err.to_string();
    for k in ["media", "mesh", "texture", "hair", "instance"] {
        assert!(msg.contains(k), "未知条目提示里应列出 '{k}'，实际：{msg}");
    }
}

/// **D6（T1 / G-9）：`mesh` 块能声明折痕边与角点，且真的进了 `MeshDesc`。**
///
/// 折痕是"调用方无法表达"的那类缺口——管线里有算法、有判据，但语言里没有
/// 承载它的位置。D6 就是这一层的存在性证明：写得出、解析得出、带得到
/// `prism_core::MeshDesc`。至于它有没有真的改变渲染，由 render 侧的
/// P4 判据（`t0_product_path.rs`）负责。
#[test]
fn d6_mesh_block_declares_crease_and_corner() {
    let src: String = wrap(
        "mesh m { kind box size 2\n  crease (0, 1, 1.0)\n  crease (1, 3, 0.5)\n  corner 0\n}",
    );
    let a: prism_core::SceneAttach = attach_of(&src);
    assert_eq!(a.meshes.len(), 1);
    let m = &a.meshes[0];
    assert_eq!(m.crease.len(), 2, "两条折痕边");
    assert_eq!(m.crease[0], (0, 1, 1.0), "第一条折痕原样带过来");
    assert_eq!(m.crease[1], (1, 3, 0.5), "分数权重不丢精度");
    assert_eq!(m.corners, vec![0], "角点标记");
    assert!(m.has_marks(), "has_marks 必须能看出有标记");

    // 不写 crease / corner 的网格：全空，且 `has_marks()` 为假。
    let a: prism_core::SceneAttach = attach_of(&wrap("mesh m { kind plane size 1 }"));
    assert!(a.meshes[0].crease.is_empty());
    assert!(a.meshes[0].corners.is_empty());
    assert!(!a.meshes[0].has_marks());

    // 非法写法报三段式错误，不 panic。
    let err = compile_v2(&wrap("mesh m { kind plane crease 3 }")).unwrap_err();
    assert!(err.to_string().contains("P2107"), "crease 需要三元组，实际：{err}");
    let err = compile_v2(&wrap("mesh m { kind plane corner (1,2,3) }")).unwrap_err();
    assert!(err.to_string().contains("P2107"), "corner 需要下标，实际：{err}");
}
