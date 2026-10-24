//! W1-A3 验收测试：渲染侧消费 DSL v2 三种新节点。
//!
//! 覆盖验收四条：
//! 1. 渲染管线能消费三种新节点类型，不 panic、不静默错（消费不到显式报错）；
//! 2. post 链按声明序串行（f∘g∘h，Chain 语义）；
//! 3. 未知 / 缺失节点类型有稳妥降级路径（不崩溃、显式处理）；
//! 4. 消费行为确定：同图同参两遍一致。
//!
//! 用例分两组：DSL 端到端（`compile_v2` 真实产物）+ 手工图夹具（错误 /
//! 退化路径，codegen 产不出的形态）。

use prism_core::node::{Composition, Node, ParamValue, StructureId};
use prism_core::{Graph, Light, Material, NodeId};
use prism_dsl::{
    compile_v2, AudioInfo, CompileV2Output, OutputFormat, PostInfo, SceneV2,
};
use prism_gpu::{CpuRefBackend, Framebuffer};
use prism_math::Vec3;
use prism_render::nodes::{
    consume_scene, PostEffectKind, PostStep, SceneConsumption,
};

use prism_core::{Aesthetic, Camera, Scene};

// ---------------------------------------------------------------------------
// DSL 夹具
// ---------------------------------------------------------------------------

/// 全三族节点一战之 DSL（场景内声明序 = id 分配序）：
/// camera=1, sdf=2, material=3, light=4, post: bloom=5 grain=6 vignette=7
/// chromatic=8, audio=9。
const SRC_ALL: &str = concat!(
    "// prism-dsl 1\n",
    "movie \"w1\" { duration 2s fps 12 resolution 64x36 }\n",
    "scene \"s\" {\n",
    "  duration 2s\n",
    "  stage3d {\n",
    "    camera main { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    sdf ball { kind sphere radius 1 }\n",
    "    material red { albedo #c0392b roughness 0.3 metalness 0.05 }\n",
    "    light key { direction(1,-1,-0.5) color #fff4e6 intensity 3 }\n",
    "  }\n",
    "  post { bloom 0.3 grain 0.1 vignette 0.2 chromatic 0.15 }\n",
    "  audio { music { bpm 120 gain 0.8 duck 12 } }\n",
    "}\n",
);

/// 双场景各带一盏灯（归属判别用）。
const SRC_TWO_SCENES: &str = concat!(
    "// prism-dsl 1\n",
    "movie \"w1b\" { duration 4s fps 12 resolution 64x36 }\n",
    "scene \"a\" {\n",
    "  duration 2s\n",
    "  stage3d {\n",
    "    camera c1 { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    light l1 { direction(1,0,0) color #ff0000 intensity 2 }\n",
    "  }\n",
    "}\n",
    "scene \"b\" {\n",
    "  duration 2s\n",
    "  stage3d {\n",
    "    camera c2 { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    light l2 { direction(0,1,0) color #00ff00 intensity 5 }\n",
    "  }\n",
    "}\n",
);

/// 单效果 vignette（串行语义的手动参照）。
fn src_post(body: &str) -> String {
    format!(
        concat!(
            "// prism-dsl 1\n",
            "movie \"w1c\" {{ duration 2s fps 12 resolution 64x36 }}\n",
            "scene \"s\" {{\n",
            "  duration 2s\n",
            "  stage3d {{ camera main {{ orbit {{ center 0,0,0 radius 8 }} fov 45deg }} }}\n",
            "  post {{ {} }}\n",
            "}}\n",
        ),
        body
    )
}

/// 非均匀测试帧（渐变 + 亮块），供效果应用类断言使用。
fn test_fb(w: u32, h: u32) -> Framebuffer {
    let mut fb = Framebuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            fb.pixels[i] = (x * 255 / w.max(1)) as u8;
            fb.pixels[i + 1] = (y * 255 / h.max(1)) as u8;
            fb.pixels[i + 2] = ((x + y) * 255 / (w + h).max(1)) as u8;
            fb.pixels[i + 3] = 255;
        }
    }
    fb
}

// ---------------------------------------------------------------------------
// 手工图夹具（错误 / 退化路径）
// ---------------------------------------------------------------------------

/// 最小 SceneV2（无任何新节点）。
fn bare_scene(name: &str) -> SceneV2 {
    SceneV2 {
        name: name.to_string(),
        start: 0.0,
        duration: 1.0,
        node_ids: Vec::new(),
        camera: None,
        layers: Vec::new(),
        sim: None,
        post: PostInfo::default(),
        audio: None,
        notes: Vec::new(),
    }
}

/// 以场景列表 + 图组装编译产物（其余字段取合法占位）。
fn base_compile(scenes: Vec<SceneV2>, graph: Graph) -> CompileV2Output {
    CompileV2Output {
        title: "fixture".to_string(),
        width: 32,
        height: 18,
        fps: 8,
        duration: 1.0,
        format: OutputFormat::Frames,
        scenes,
        transitions: Vec::new(),
        graph,
        channels: Vec::new(),
        packs: Vec::new(),
        theme: None,
        fonts: Vec::new(),
        formulas: Vec::new(),
    }
}

/// post 效果节点（结构 / 组合 / 参数可定制——构造退化形态用）。
fn post_node(id: NodeId, structure: StructureId, composition: Composition, field: &str, value: f32) -> Node {
    let mut node = Node::new(id, structure, composition);
    node.params.set(field, ParamValue::F32(value));
    node
}

/// 带 post 摘要 + 节点清单的场景。
fn scene_with_post(name: &str, params: &[(&str, f32)], node_ids: Vec<NodeId>) -> SceneV2 {
    let mut scene = bare_scene(name);
    scene.post = PostInfo {
        params: params.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
        node_ids,
    };
    scene
}

// ---------------------------------------------------------------------------
// 验收 1：三种新节点一战消费（不 panic、不静默错）
// ---------------------------------------------------------------------------

#[test]
fn consume_scene_consumes_audio_post_and_light_end_to_end() {
    let compile: CompileV2Output = compile_v2(SRC_ALL).expect("全三族 DSL 应编译成功");
    let got: SceneConsumption = consume_scene(&compile, 0).expect("三种新节点都应可消费");

    // ---- audio → S6 谱解 ----
    assert_eq!(got.audio.len(), 1, "一个 audio 块 → 一个音乐床节点");
    let bed = &got.audio[0];
    assert_eq!(bed.node_id, 9, "audio 节点 id（场景内分配序末位）");
    assert_eq!(bed.bpm, 120.0);
    assert_eq!(bed.gain, 0.8);
    assert_eq!(bed.duck, 12.0);

    // ---- post → S2/S4 效果链（声明序 = 链序）----
    let steps: &[PostStep] = got.post.steps();
    let kinds: Vec<PostEffectKind> = steps.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            PostEffectKind::Bloom,
            PostEffectKind::Grain,
            PostEffectKind::Vignette,
            PostEffectKind::Chromatic,
        ],
        "链序必须等于 DSL 声明序"
    );
    let strengths: Vec<f32> = steps.iter().map(|s| s.strength).collect();
    assert_eq!(strengths, vec![0.3, 0.1, 0.2, 0.15]);
    let ids: Vec<NodeId> = steps.iter().map(|s| s.node_id).collect();
    assert_eq!(ids, vec![5, 6, 7, 8], "效果节点 id 按声明序分配");
    assert_eq!(got.post.len(), 4);
    assert!(!got.post.is_empty());

    // ---- light v2 → S8 源项 ----
    assert_eq!(got.lights.len(), 1, "一盏灯 → 一个源项（material 不被误判）");
    let light = &got.lights[0];
    assert_eq!(light.node_id, 4, "light 节点 id（stage3d 内 camera/sdf/material 之后）");
    assert_eq!(light.direction, Vec3::new(1.0, -1.0, -0.5), "direction 逐分量透传不归一化");
    assert_eq!(
        light.color,
        Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
        "color 按 0..1 归一化（C17）"
    );
    assert_eq!(light.intensity, 3.0, "intensity 原值透传");

    // ---- 折算原生光源（管线可直用）----
    let core: Vec<Light> = got.core_lights();
    assert_eq!(core, vec![Light::new(Vec3::new(1.0, -1.0, -0.5), Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0), 3.0)]);
}

#[test]
fn consume_scene_without_new_blocks_is_empty_not_error() {
    // 未声明三种新块的场景：消费为空（不是错误）——无块 ≠ 缺块。
    let minimal: &str = concat!(
        "// prism-dsl 1\n",
        "movie \"m\" { duration 1s fps 12 resolution 32x18 }\n",
        "scene \"s\" { duration 1s stage3d { camera c { orbit { center 0,0,0 radius 8 } fov 45deg } } }\n",
    );
    let compile: CompileV2Output = compile_v2(minimal).expect("最小文档应编译");
    let got: SceneConsumption = consume_scene(&compile, 0).expect("无新块也应可消费");
    assert!(got.audio.is_empty());
    assert!(got.post.is_empty());
    assert!(got.lights.is_empty());
}

// ---------------------------------------------------------------------------
// 验收 2：post 链按声明序串行（f∘g∘h，Chain 语义）
// ---------------------------------------------------------------------------

#[test]
fn post_chain_applies_serially_in_declared_order() {
    // 串行语义的行为坐实：[f, f] 应用一次 ≡ [f] 应用两次（逐位相等）。
    // 选 vignette（径向乘算、与节点 id 无关）——唯一变量是复合次序。
    let twice: CompileV2Output = compile_v2(&src_post("vignette 0.5 vignette 0.5")).expect("应编译");
    let once: CompileV2Output = compile_v2(&src_post("vignette 0.5")).expect("应编译");
    assert_eq!(twice.scenes[0].post.node_ids.len(), 2, "重复字段 → 两个节点（codegen 逐字段建点）");
    assert_eq!(once.scenes[0].post.node_ids.len(), 1);

    let chain_twice = consume_scene(&twice, 0).expect("应可消费").post;
    let chain_once = consume_scene(&once, 0).expect("应可消费").post;

    let mut a = test_fb(48, 27);
    let mut b = test_fb(48, 27);
    chain_twice.apply(&mut a, 0);
    chain_once.apply(&mut b, 0);
    chain_once.apply(&mut b, 0);
    assert_eq!(a.pixels, b.pixels, "[f,f] 一趟必须逐位等于 [f] 两趟（f∘g∘h）");

    // 非交换性：两种效果换序 ⇒ 结果不同（若为并行 / 加和则换序无害）。
    let vg: CompileV2Output = compile_v2(&src_post("vignette 0.5 chromatic 0.5")).expect("应编译");
    let gv: CompileV2Output = compile_v2(&src_post("chromatic 0.5 vignette 0.5")).expect("应编译");
    let chain_vg = consume_scene(&vg, 0).expect("应可消费").post;
    let chain_gv = consume_scene(&gv, 0).expect("应可消费").post;
    let mut x = test_fb(48, 27);
    let mut y = test_fb(48, 27);
    chain_vg.apply(&mut x, 0);
    chain_gv.apply(&mut y, 0);
    assert_ne!(x.pixels, y.pixels, "效果非交换：换序必须产生不同画面（串行有序）");
    // 链首输入隐式挂场景合成输出：链首图内无入边（约定落地的事实检查）。
    assert!(
        vg.graph.inputs_of(chain_vg.steps()[0].node_id).is_empty(),
        "链首输入隐式挂场景合成输出——图内不应有入边"
    );
}

// ---------------------------------------------------------------------------
// 验收 3：未知 / 缺失节点类型的稳妥降级（不崩溃、显式处理）
// ---------------------------------------------------------------------------

#[test]
fn missing_registered_nodes_error_explicitly() {
    // post 登记悬空：node_ids 里的 id 不在图中 → R5031（不 panic、不静默跳）。
    let scene = scene_with_post("s", &[("bloom", 0.3)], vec![9]);
    let compile = base_compile(vec![scene], Graph::new());
    let err = consume_scene(&compile, 0).unwrap_err();
    assert_eq!(err.code, "R5031");
    assert!(err.location.contains("post"), "定位应指向 post 节点：{}", err.location);

    // audio 登记悬空 → 同一显式错误族。
    let mut scene = bare_scene("s");
    scene.audio = Some(AudioInfo {
        music: [("bpm".to_string(), 120.0)].into_iter().collect(),
        node_ids: vec![9],
    });
    let compile = base_compile(vec![scene], Graph::new());
    let err = consume_scene(&compile, 0).unwrap_err();
    assert_eq!(err.code, "R5031");
    assert!(err.location.contains("audio"), "定位应指向 audio 节点：{}", err.location);
}

#[test]
fn malformed_post_nodes_error_explicitly() {
    // (a) 未知结构（sim 的 S3 节点混进 post 清单）。
    let mut graph = Graph::new();
    graph.add_node(post_node(9, StructureId::ConstraintFlow, Composition::Chain, "bloom", 0.3)).unwrap();
    let compile = base_compile(vec![scene_with_post("s", &[("bloom", 0.3)], vec![9])], graph);
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5034");

    // (b) 效果参数不可识别（白名单外的字段名）。
    let mut graph = Graph::new();
    graph.add_node(post_node(9, StructureId::Convolution, Composition::Chain, "warp", 0.3)).unwrap();
    let compile = base_compile(vec![scene_with_post("s", &[], vec![9])], graph);
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5033");

    // (c) 组合方式不是 Chain（串行链约定被破坏）。
    let mut graph = Graph::new();
    graph.add_node(post_node(9, StructureId::Convolution, Composition::Feed, "bloom", 0.3)).unwrap();
    let compile = base_compile(vec![scene_with_post("s", &[("bloom", 0.3)], vec![9])], graph);
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5035");

    // (d) 链边断链（声明序在 10 之前但图内无 9→10 连线）。
    let mut graph = Graph::new();
    graph.add_node(post_node(9, StructureId::Convolution, Composition::Chain, "bloom", 0.3)).unwrap();
    graph.add_node(post_node(10, StructureId::DeterministicSampling, Composition::Chain, "grain", 0.1)).unwrap();
    let compile = base_compile(
        vec![scene_with_post("s", &[("bloom", 0.3), ("grain", 0.1)], vec![9, 10])],
        graph,
    );
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5037");

    // (e) 强度非有限（NaN 钳不了 → 显式拒绝）。
    let mut graph = Graph::new();
    graph.add_node(post_node(9, StructureId::Convolution, Composition::Chain, "bloom", f32::NAN)).unwrap();
    let compile = base_compile(vec![scene_with_post("s", &[("bloom", 0.3)], vec![9])], graph);
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5036");

    // (f) 摘要声明了效果但节点清单缺失（防静默不渲染）。
    let compile = base_compile(vec![scene_with_post("s", &[("bloom", 0.3)], vec![])], Graph::new());
    assert_eq!(consume_scene(&compile, 0).unwrap_err().code, "R5038");
}

#[test]
fn malformed_audio_node_errors_explicitly() {
    // audio 节点结构不符（S2 卷积冒充 S6 谱解）。
    let mut graph = Graph::new();
    let mut node = Node::new(9, StructureId::Convolution, Composition::Feed);
    node.params.set("bpm", ParamValue::F32(120.0));
    node.params.set("gain", ParamValue::F32(0.8));
    node.params.set("duck", ParamValue::F32(12.0));
    graph.add_node(node).unwrap();
    let mut scene = bare_scene("s");
    scene.audio = Some(AudioInfo::default());
    scene.audio.as_mut().expect("已置").node_ids.push(9);
    let compile = base_compile(vec![scene], graph);
    let err = consume_scene(&compile, 0).unwrap_err();
    assert_eq!(err.code, "R5032");
}

#[test]
fn scene_index_out_of_range_errors_explicitly() {
    let compile = base_compile(vec![bare_scene("s")], Graph::new());
    let err = consume_scene(&compile, 5).unwrap_err();
    assert_eq!(err.code, "R5030");
}

#[test]
fn unsupported_nodes_outside_new_families_are_ignored_not_fatal() {
    // node_ids 里的暂不支持节点（sim/particles/compose）归 movie 侧报错；
    // 本消费层只取三种新族，其余原样忽略——不崩溃、不误伤。
    let mut graph = Graph::new();
    let mut sim = Node::new(1, StructureId::ConstraintFlow, Composition::Feed);
    sim.params.set("kind", ParamValue::Str("cloth".to_string()));
    graph.add_node(sim).unwrap();
    let mut light = Node::new(2, StructureId::RadiativeTransfer, Composition::Feed);
    light.params.set("dir_x", ParamValue::F32(0.0));
    light.params.set("dir_y", ParamValue::F32(-1.0));
    light.params.set("dir_z", ParamValue::F32(0.0));
    light.params.set("color_r", ParamValue::F32(1.0));
    light.params.set("color_g", ParamValue::F32(1.0));
    light.params.set("color_b", ParamValue::F32(1.0));
    light.params.set("intensity", ParamValue::F32(1.0));
    graph.add_node(light).unwrap();
    // material 节点（S8 + albedo_r 签名）不得被误判为光源。
    let mut mat = Node::new(3, StructureId::RadiativeTransfer, Composition::Feed);
    mat.params.set("albedo_r", ParamValue::F32(0.9));
    mat.params.set("roughness", ParamValue::F32(0.4));
    mat.params.set("metalness", ParamValue::F32(0.1));
    graph.add_node(mat).unwrap();
    let mut scene = bare_scene("s");
    scene.node_ids = vec![1, 3]; // sim + material 进 render 消费面清单
    let compile = base_compile(vec![scene], graph);
    let got: SceneConsumption = consume_scene(&compile, 0).expect("本层应稳妥降级（不报错）");
    assert!(got.audio.is_empty());
    assert!(got.post.is_empty());
    assert_eq!(got.lights.len(), 1, "只有带 dir_x 的 S8 节点是光源");
    assert_eq!(got.lights[0].node_id, 2);
    assert_eq!(got.lights[0].direction, Vec3::new(0.0, -1.0, 0.0));
    assert_eq!(got.lights[0].intensity, 1.0);
}

// ---------------------------------------------------------------------------
// 验收 4：消费行为确定（同图同参两遍一致）
// ---------------------------------------------------------------------------

#[test]
fn consumption_is_deterministic_across_compilations_and_applies() {
    // 同一 DSL 两遍编译 → 消费结果逐字段相等。
    let c1: CompileV2Output = compile_v2(SRC_ALL).expect("应编译");
    let c2: CompileV2Output = compile_v2(SRC_ALL).expect("应编译");
    let a: SceneConsumption = consume_scene(&c1, 0).expect("应可消费");
    let b: SceneConsumption = consume_scene(&c2, 0).expect("应可消费");
    assert_eq!(a, b, "同图同参两遍编译的消费必须一致");

    // 同一链两遍应用（含 grain 帧键）→ 帧缓冲逐位一致。
    let mut f1 = test_fb(48, 27);
    let mut f2 = test_fb(48, 27);
    a.post.apply(&mut f1, 5);
    a.post.apply(&mut f2, 5);
    assert_eq!(f1.pixels, f2.pixels, "同帧两遍应用必须逐位一致");

    // 帧号进 grain 哈希：不同帧不同图案（帧间去相关的确定性坐实）。
    let mut f3 = test_fb(48, 27);
    a.post.apply(&mut f3, 6);
    assert_ne!(f1.pixels, f3.pixels, "grain 必须帧间去相关");
}

#[test]
fn light_terms_are_attributed_per_scene() {
    let compile: CompileV2Output = compile_v2(SRC_TWO_SCENES).expect("应编译");
    let a: SceneConsumption = consume_scene(&compile, 0).expect("场景 a 应可消费");
    let b: SceneConsumption = consume_scene(&compile, 1).expect("场景 b 应可消费");
    // 场景 a：camera=1, light=2；场景 b：camera=3, light=4。
    assert_eq!(a.lights.len(), 1);
    assert_eq!(a.lights[0].node_id, 2);
    assert_eq!(a.lights[0].direction, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(a.lights[0].color, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(a.lights[0].intensity, 2.0);
    assert_eq!(b.lights.len(), 1);
    assert_eq!(b.lights[0].node_id, 4);
    assert_eq!(b.lights[0].direction, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(b.lights[0].color, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(b.lights[0].intensity, 5.0);
}

#[test]
fn consumed_post_chain_applies_to_rendered_frame_deterministically() {
    // 管线级：真实渲染一帧 → 应用消费到的 post 链 → 确定且非恒等。
    let compile: CompileV2Output = compile_v2(SRC_ALL).expect("应编译");
    let got: SceneConsumption = consume_scene(&compile, 0).expect("应可消费");
    let scene: Scene = Scene {
        radius: 1.0,
        orbit_center: Vec3::new(0.0, 0.0, 0.0),
        orbit_radius: 4.0,
        material: Material::new(Vec3::new(0.8, 0.2, 0.2), 0.4, 0.1),
        lights: vec![],
        camera: Camera::new(Vec3::new(0.0, 0.0, -4.0), Vec3::new(0.0, 0.0, 0.0), 60.0_f32.to_radians()),
        aesthetic: Aesthetic::new(false, 1.0, false),
    };
    let mut device = CpuRefBackend;
    let base: Framebuffer = prism_render::render_frame(&mut device, &scene, 0.0, 1.0, 32, 18);
    let mut x = base.clone();
    let mut y = base.clone();
    got.post.apply(&mut x, 0);
    got.post.apply(&mut y, 0);
    assert_eq!(x.pixels, y.pixels, "管线级两遍应用必须逐位一致");
    assert_ne!(x.pixels, base.pixels, "非空链（强度非 0）必须改变画面");
}
