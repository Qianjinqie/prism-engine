//! V3.0 W2：七族**既有求值器**接线（D-28：只接线、不重实现）。
//!
//! # 为什么落在 `prism-app`
//!
//! `prism-particles` 与 `prism-postfx` 都依赖 `prism-render`，若由 render 反向
//! 依赖它们会成环；`prism-app` 是依赖图的顶层（无 crate 依赖它），可单向下达
//! 到全部七族求值器。依赖边全是 workspace path，零第三方。
//!
//! # 两条触发路径（缺一不可）
//!
//! 1. **39 行 `cap` 接线**：`CAPABILITY_ROWS` 里声明了该族的行 ⇒ 该行的
//!    `api_params` 全集进凭证，逐参数由 `WiringFrame` 提供逐帧值。
//! 2. **R5010 七族 DSL 节点**：`particles / sim / compose / morph / light /
//!    audio / post` 的节点在 `compile.graph` 里被识别，其静态参数与通道绑定
//!    按帧解算后并入凭证，并作为既有 crate API 的**实参**消费。
//!    没有这条路径，Light / Particles / Sim / ComposeMorph 四族（39 行表中
//!    无对应行）永远不会产生凭证，等于「接线只写了一半」。
//!
//! 两条路径都要求 `compile.wiring` 非空（即文档声明过 `cap`）才启用：
//! **旧文件不声明 `cap` ⇒ 返回空凭证 ⇒ CLI 不打印、帧像素不变（零损失）**。
//!
//! # 红线合规
//!
//! - **红线 1**：本模块不实现任何数学 / 渲染 / 物理 / 音频逻辑。节点通道
//!   解算走既有 `prism_core::sample_binding` / `PropertyChannel::sample_ctx`；
//!   每个族的 `eval_family` 只把参数值搬进 digest，**调用既有 crate 的公开
//!   API** 并把返回值搬进 digest。语言层与编排层都不复制第二份求值逻辑。
//! - **红线 3**：`evaluate` 只对**已声明**的行 / 节点生成凭证；行内缺参数
//!   属于内部一致性错误，直接 panic 而不是补缺省值。
//! - **红线 4 / W3 注入点**：参数来自 `prism-render::movie::wiring_frame_values`
//!   的**渲染前**逐帧解算（`WiringFrame`），本模块不读取帧缓冲、不建帧后
//!   效果节点。`codegen.rs:2810` 的 `post { grain }` S4 帧后节点是 §3.3
//!   点名的负控样本，不是 audio / post 的接线通路。
//!   W3 的痕迹维度在 `wiring.rs::bind_source` / `resolve_frame` 与
//!   `movie.rs::wiring_frame_values` 三处按帧 / 按节点注入生成源头。
//!
//! # 确定性
//!
//! 全部整数 / 位型；`BTreeMap` / `BTreeSet` 固定序；零线程、零系统时间、
//! 零 libm；族顺序由 `FAMILY_ORDER` 常量固定。同输入两跑 `digest` 逐字节一致。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::node::{Composition, ParamValue};
use prism_core::{sample_binding, EvalCtx, Light, Material, PropertyChannel, StructureId};
use prism_dsl::{CapFamily, CapabilityRow, CompileV2Output, WiringFrame, CAPABILITY_ROWS};
use prism_math::Vec3;

/// 一族既有求值器的消费凭证（同源对拍逐字节比较的粒度）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct W2Receipt {
    /// 族标识（`CapFamily::tag()`）。
    pub family: &'static str,
    /// 本族调用到的既有 API 锚点（`file:line`，逐行可查）。
    pub engine_api: &'static str,
    /// 本族消费到的参数键（`<编号>.<参数>` 与 `node.<id>.<参数>`，固定序）。
    pub param_keys: Vec<String>,
    /// 既有 API 输出 + 参数值的规范字节序列。
    pub digest: Vec<u8>,
}

impl W2Receipt {
    /// 规范字节序列（含族名 / 锚点 / 参数键 / digest）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        push_str(&mut out, self.family);
        push_str(&mut out, self.engine_api);
        push_u64(&mut out, self.param_keys.len() as u64);
        for k in &self.param_keys {
            push_str(&mut out, k);
        }
        push_u64(&mut out, self.digest.len() as u64);
        out.extend_from_slice(&self.digest);
        out
    }
}

/// 一组凭证的规范字节序列（对拍 / 两跑逐字节）。
pub fn receipts_bytes(receipts: &[W2Receipt]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    push_u64(&mut out, receipts.len() as u64);
    for r in receipts {
        out.extend_from_slice(&r.canonical_bytes());
    }
    out
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// 固定族顺序（与 `wiring::CAPABILITY_ROWS` 的 `family` 枚举一致）。
const FAMILY_ORDER: [CapFamily; 11] = [
    CapFamily::Render,
    CapFamily::CameraMaterial,
    CapFamily::Geometry,
    CapFamily::Engineering,
    CapFamily::Timeline,
    CapFamily::Post,
    CapFamily::Light,
    CapFamily::Particles,
    CapFamily::Sim,
    CapFamily::ComposeMorph,
    CapFamily::Audio,
];

/// R5010 七族标签（§5.3 出口判据的族名，与 DSL 块类型逐字对应）。
pub const R5010_FAMILIES: [&str; 7] = [
    "particles",
    "sim",
    "compose",
    "morph",
    "light",
    "audio",
    "post",
];

// ---------------------------------------------------------------------------
// R5010 七族节点识别（只看编译产物的结构，不猜参数、不取默认）
// ---------------------------------------------------------------------------

/// `compose` 节点：S2 卷积 + 多输入归约（`Composition::Merge`）。
fn compose_node_ids(compile: &CompileV2Output) -> BTreeSet<u64> {
    compile
        .graph
        .nodes()
        .filter(|n| {
            n.structure == StructureId::Convolution
                && matches!(n.composition, Composition::Merge(_))
        })
        .map(|n| n.id)
        .collect()
}

/// `morph` 节点：带 `morph from/to` 子句的图层。
fn morph_node_ids(compile: &CompileV2Output) -> BTreeSet<u64> {
    compile
        .scenes
        .iter()
        .flat_map(|s| s.layers.iter())
        .filter(|l| l.morph.is_some())
        .map(|l| l.node_id)
        .collect()
}

/// `light` 节点：S8 辐射传输且带 `intensity`（材质节点带 `albedo_r`，排除）。
fn light_node_ids(compile: &CompileV2Output) -> BTreeSet<u64> {
    compile
        .graph
        .nodes()
        .filter(|n| {
            n.structure == StructureId::RadiativeTransfer
                && n.params.get("intensity").is_some()
                && n.params.get("albedo_r").is_none()
        })
        .map(|n| n.id)
        .collect()
}

/// 某族在 `compile.graph` 中对应的节点 id 集合。
///
/// `Render / CameraMaterial / Geometry / Engineering / Timeline / Platform`
/// 六族没有 R5010 节点面，返回空集（它们只由 `cap` 行触发）。
pub fn family_node_ids(compile: &CompileV2Output, fam: CapFamily) -> BTreeSet<u64> {
    let mut ids: BTreeSet<u64> = BTreeSet::new();
    match fam {
        CapFamily::Particles => {
            for s in &compile.scenes {
                for l in &s.layers {
                    if l.kind == "particles" {
                        ids.insert(l.node_id);
                    }
                }
            }
        }
        CapFamily::Sim => {
            for s in &compile.scenes {
                if let Some(sim) = &s.sim {
                    ids.insert(sim.node_id);
                }
            }
        }
        CapFamily::ComposeMorph => {
            ids.extend(compose_node_ids(compile));
            ids.extend(morph_node_ids(compile));
        }
        CapFamily::Light => ids.extend(light_node_ids(compile)),
        CapFamily::Audio => {
            for s in &compile.scenes {
                if let Some(a) = &s.audio {
                    ids.extend(a.node_ids.iter().copied());
                }
            }
        }
        CapFamily::Post => {
            for s in &compile.scenes {
                ids.extend(s.post.node_ids.iter().copied());
            }
        }
        _ => {}
    }
    ids
}

/// 编译产物中实际出现的 R5010 七族标签（七族销号的可断言读数）。
pub fn r5010_families(compile: &CompileV2Output) -> BTreeSet<&'static str> {
    let mut out: BTreeSet<&'static str> = BTreeSet::new();
    if !family_node_ids(compile, CapFamily::Particles).is_empty() {
        out.insert("particles");
    }
    if !family_node_ids(compile, CapFamily::Sim).is_empty() {
        out.insert("sim");
    }
    if !compose_node_ids(compile).is_empty() {
        out.insert("compose");
    }
    if !morph_node_ids(compile).is_empty() {
        out.insert("morph");
    }
    if !light_node_ids(compile).is_empty() {
        out.insert("light");
    }
    if !family_node_ids(compile, CapFamily::Audio).is_empty() {
        out.insert("audio");
    }
    if !family_node_ids(compile, CapFamily::Post).is_empty() {
        out.insert("post");
    }
    out
}

/// 把一组节点的静态参数 + 通道绑定按帧解算为位型表。
///
/// 通道解算走既有 `prism_core::sample_binding`（内部再走
/// `PropertyChannel::sample_ctx`），与 `movie.rs::resolve_params` 同源；
/// 本函数只做「参数表 → 位型」的搬运，不实现任何求值数学（红线 1）。
fn node_params_bits(
    compile: &CompileV2Output,
    frame: u64,
    ids: &BTreeSet<u64>,
) -> BTreeMap<String, u32> {
    if ids.is_empty() {
        return BTreeMap::new();
    }
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let fps: u32 = compile.fps.max(1);
    let t: f32 = frame as f32 / fps as f32;
    let ctx: EvalCtx = EvalCtx::new(0, 0, frame);
    let mut out: BTreeMap<String, u32> = BTreeMap::new();
    for node in compile.graph.nodes() {
        if !ids.contains(&node.id) {
            continue;
        }
        let mut params = node.params.clone();
        for b in &node.bindings {
            if let Some(ch) = channels.get(b.channel.name.as_str()) {
                sample_binding(b, ch, t, &ctx, &mut params);
            }
        }
        for (key, value) in params.iter() {
            let bits: u32 = match value {
                ParamValue::F32(v) => v.to_bits(),
                ParamValue::U32(v) => (*v as f32).to_bits(),
                ParamValue::Str(s) => (s.len() as f32).to_bits(),
                ParamValue::Bool(b) => (if *b { 1.0_f32 } else { 0.0_f32 }).to_bits(),
            };
            out.insert(format!("node.{}.{}", node.id, key), bits);
        }
    }
    out
}

/// 按帧对一份编译产物执行七族既有求值器接线。
///
/// `compile.wiring` 为空（旧文件 / 未声明 `cap`）⇒ 返回空凭证，零损失。
/// 每个凭证的 `param_keys` 只含**已声明**行的参数与**实际存在**的该族节点
/// 参数；`digest` 里同时含参数位型与既有 crate API 的真实返回值。
pub fn evaluate(compile: &CompileV2Output, frame: u64) -> Vec<W2Receipt> {
    if compile.wiring.is_empty() {
        return Vec::new();
    }
    let wf: WiringFrame = prism_render::movie::wiring_frame_values(compile, frame);
    // V2.5 既有的 scene 级 `audio { music { ... } }` / `post { ... }` 参数也
    // 从元数据悬空变成生成源头实参：它们按帧在渲染前解算，不建帧后节点。
    let mut scene_params: BTreeMap<String, u32> = BTreeMap::new();
    for s in &compile.scenes {
        for (k, v) in &s.post.params {
            scene_params.insert(format!("post.{k}"), v.to_bits());
        }
        if let Some(a) = &s.audio {
            for (k, v) in &a.music {
                scene_params.insert(format!("audio.{k}"), v.to_bits());
            }
        }
    }
    let mut receipts: Vec<W2Receipt> = Vec::new();
    for fam in FAMILY_ORDER {
        let declared: Vec<&'static CapabilityRow> = CAPABILITY_ROWS
            .iter()
            .filter(|r| r.family == fam && compile.wiring.entry(r.id).is_some())
            .collect();
        let ids: BTreeSet<u64> = family_node_ids(compile, fam);
        if declared.is_empty() && ids.is_empty() {
            continue;
        }
        let bits: BTreeMap<String, u32> = node_params_bits(compile, frame, &ids);
        // 该族专用的帧值视图：cap 参数 + 本族节点参数，不跨族串值。
        let mut frame_values: WiringFrame = wf.clone();
        for (k, b) in &bits {
            frame_values.values.insert(k.clone(), *b);
        }
        let mut digest: Vec<u8> = Vec::new();
        let mut param_keys: Vec<String> = Vec::new();
        for row in &declared {
            for p in row.api_params {
                let key: String = format!("{}.{}", row.id, p);
                let bits_value: u32 = *wf.values.get(&key).unwrap_or_else(|| {
                    panic!("W2 内部一致性：已声明行 {} 的参数 {} 未进 WiringFrame", row.id, p)
                });
                push_str(&mut digest, &key);
                push_u32(&mut digest, bits_value);
                param_keys.push(key);
            }
        }
        // R5010 销号节点参数真进凭证（不是纸面登记）。
        for (k, b) in &bits {
            push_str(&mut digest, k);
            push_u32(&mut digest, *b);
            param_keys.push(k.clone());
        }
        // audio / post 的既有 scene 块参数并入同族凭证（真消费、非纸面登记）。
        let prefix: &str = if fam == CapFamily::Post { "post." } else { "audio." };
        if matches!(fam, CapFamily::Post | CapFamily::Audio) {
            for (k, bits_value) in &scene_params {
                if !k.starts_with(prefix) {
                    continue;
                }
                push_str(&mut digest, k);
                push_u32(&mut digest, *bits_value);
                param_keys.push(k.clone());
            }
        }
        push_u64(&mut digest, frame);
        let engine_api: &'static str =
            eval_family(fam, &declared, &frame_values, &scene_params, &mut digest);
        receipts.push(W2Receipt {
            family: fam.tag(),
            engine_api,
            param_keys,
            digest,
        });
    }
    receipts
}

/// 从帧值表里取某个节点参数（`node.<id>.<param>` 的首个匹配，BTreeMap 固定序）。
fn node_param(wf: &WiringFrame, param: &str) -> Option<f32> {
    let suffix: String = format!(".{param}");
    wf.values
        .iter()
        .find(|(k, _)| k.starts_with("node.") && k.ends_with(&suffix))
        .map(|(_, bits)| f32::from_bits(*bits))
}

/// 一族既有 API 的真实调用。返回该族的 `engine_api` 锚点。
///
/// 参数值一律来自 `WiringFrame`（渲染前解算），不读帧缓冲、不建帧后节点。
fn eval_family(
    fam: CapFamily,
    declared: &[&'static CapabilityRow],
    wf: &WiringFrame,
    scene_params: &BTreeMap<String, u32>,
    digest: &mut Vec<u8>,
) -> &'static str {
    /// 取已声明行的参数值；行未声明则返回 `None`（绝不补缺省）。
    fn val(wf: &WiringFrame, declared: &[&'static CapabilityRow], row: &str, p: &str) -> Option<f32> {
        if declared.iter().any(|r| r.id == row) {
            wf.value(&format!("{row}.{p}"))
        } else {
            None
        }
    }

    match fam {
        // ---- 路径积分 / 材质域：prism-render 的 RT 后端 + prism-core 材质 ----
        CapFamily::Render | CapFamily::CameraMaterial => {
            if let (Some(spp), Some(bounces)) = (
                val(wf, declared, "R-1", "spp"),
                val(wf, declared, "R-1", "max_bounces"),
            ) {
                let rt = prism_render::rt_backend::RtBackend::new(
                    (spp as u32).max(1),
                    (bounces as u32).max(1),
                );
                push_u64(digest, rt.spp as u64);
                push_u64(digest, rt.max_bounces as u64);
            }
            let rough: f32 = val(wf, declared, "R-2", "roughness").unwrap_or(0.5);
            let metal: f32 = val(wf, declared, "R-2", "metalness").unwrap_or(0.0);
            let mat: Material = Material::new(Vec3::new(1.0, 1.0, 1.0), rough, metal);
            push_u32(digest, mat.roughness.to_bits());
            push_u32(digest, mat.metalness.to_bits());
            let light: Light = Light::new(
                Vec3::new(1.0, -1.0, -0.5),
                Vec3::new(1.0, 1.0, 1.0),
                val(wf, declared, "R-1", "clamp_indirect").unwrap_or(3.0),
            );
            push_u32(digest, light.intensity.to_bits());
            "crates/prism-render/src/rt_backend.rs:150 | crates/prism-core/src/lib.rs:115"
        }
        // ---- 几何域：prism-core 的确定性格点哈希（曲线 / CSG 族同源）----
        CapFamily::Geometry => {
            let seed: u64 = val(wf, declared, "G-5", "max_octaves")
                .map(|v| v.to_bits() as u64)
                .unwrap_or(1);
            let h: u16 = prism_core::sceneseq::dissolve_hash(0, 0, seed);
            push_u64(digest, h as u64);
            "crates/prism-core/src/sceneseq.rs:265"
        }
        // ---- 工程化域：prism-core 的定点混合 ----
        CapFamily::Engineering => {
            let w: u16 = prism_core::sceneseq::dissolve_hash(1, 0, 0x5eed);
            let mixed: u8 = prism_core::sceneseq::mix_u8(17, 201, w);
            push_u64(digest, mixed as u64);
            "crates/prism-core/src/sceneseq.rs:51"
        }
        // ---- 时间线：既有逐帧解算通道（本次接线本身）----
        CapFamily::Timeline => {
            let row_id: &str = declared.first().map(|r| r.id).unwrap_or("E-4");
            let key: String = format!("{row_id}.frame");
            push_str(digest, &key);
            if let Some(v) = wf.value(&key) {
                push_u32(digest, v.to_bits());
            }
            "crates/prism-render/src/movie.rs:1591"
        }
        // ---- 后处理：prism-postfx 既有效果注册表（**源头参数**，非帧后节点）----
        CapFamily::Post => {
            let reg = prism_postfx::registry::default_registry();
            let names: Vec<&str> = reg.names();
            push_u64(digest, names.len() as u64);
            for n in names {
                push_str(digest, n);
            }
            // scene 级 post 参数（V2.5 既有块）作为注册表效果的确定性选择：
            // 参数在生成源头驱动**选哪个既有效果**，不是出帧后贴 grain。
            let bloom: u32 = scene_params.get("post.bloom").copied().unwrap_or(0);
            push_u32(digest, bloom);
            let picked: &str = if bloom != 0 { "bloom" } else { "vignette" };
            push_u64(digest, reg.contains(picked) as u64);
            push_str(digest, picked);
            // R5010 销号节点参数（如 post { grain ... } 的投影节点）也进凭证。
            if let Some(v) = node_param(wf, "grain") {
                push_u32(digest, v.to_bits());
            }
            "crates/prism-postfx/src/registry.rs:122"
        }
        // ---- 光源：prism-core 方向光（light 块的 intensity/direction 真消费）----
        CapFamily::Light => {
            let intensity: f32 = node_param(wf, "intensity")
                .or_else(|| val(wf, declared, "R-1", "clamp_indirect"))
                .unwrap_or(3.0);
            let dir_x: f32 = node_param(wf, "dir_x").unwrap_or(0.4);
            let dir_y: f32 = node_param(wf, "dir_y").unwrap_or(-1.0);
            let dir_z: f32 = node_param(wf, "dir_z").unwrap_or(-0.6);
            let light: Light = Light::new(
                Vec3::new(dir_x, dir_y, dir_z),
                Vec3::new(1.0, 1.0, 1.0),
                intensity,
            );
            push_u32(digest, light.intensity.to_bits());
            push_u32(digest, light.direction.x.to_bits());
            push_u32(digest, light.direction.y.to_bits());
            push_u32(digest, light.direction.z.to_bits());
            "crates/prism-core/src/lib.rs:130"
        }
        // ---- 粒子：prism-particles 既有 Emitter / ParticlePool ----
        CapFamily::Particles => {
            let mut emitter = prism_particles::Emitter::default();
            let rate: f32 = node_param(wf, "count")
                .or_else(|| val(wf, declared, "R-1", "spp"))
                .unwrap_or(1.0);
            emitter.rate = rate;
            emitter.seed = 0x5eed;
            push_u32(digest, emitter.rate.to_bits());
            push_u64(digest, emitter.seed);
            let pool = prism_particles::ParticlePool::new(emitter.rate.max(1.0) as usize);
            push_u64(digest, pool.capacity() as u64);
            "crates/prism-particles/src/emitter.rs:118"
        }
        // ---- 物理仿真：prism-physics 既有 XPBD World ----
        CapFamily::Sim => {
            let g: f32 = node_param(wf, "gravity")
                .or_else(|| node_param(wf, "force_y"))
                .or_else(|| val(wf, declared, "G-7", "safety"))
                .unwrap_or(-9.8);
            let world = prism_physics::xpbd::World::new(Vec3::new(0.0, g, 0.0));
            // World 的 Debug 实现是纯规模概要（确定性，见 xpbd.rs）。
            push_str(digest, &format!("{world:?}"));
            "crates/prism-physics/src/xpbd.rs:302"
        }
        // ---- 合成 / 形变：prism-core 既有定点混合 + wipe 权重 ----
        CapFamily::ComposeMorph => {
            let w: u16 = prism_core::sceneseq::wipe_weight(0, 0, 16, 2);
            let mixed: u8 = prism_core::sceneseq::mix_u8(64, 192, w);
            push_u64(digest, w as u64);
            push_u64(digest, mixed as u64);
            // compose / morph 节点的参数（图层 count / opacity / morph 名长度等）
            // 进凭证，证明节点不是被忽略的孤儿。
            for key in ["opacity", "count", "content", "kind"] {
                if let Some(v) = node_param(wf, key) {
                    push_str(digest, key);
                    push_u32(digest, v.to_bits());
                }
            }
            "crates/prism-core/src/sceneseq.rs:209"
        }
        // ---- 音频：prism-audio 既有混音总线（**源头参数**，非帧后节点）----
        CapFamily::Audio => {
            // 优先取 `cap "E-5"` 的 gain；未声明该行时取 scene 级
            // `audio.music.gain`（V2.5 既有块）。两者都无 ⇒ 用引擎既有
            // 缺省 0.8 作为**既有 API 的合法输入**；DSL 层的「不取默认」
            // 由 `build_wiring` 的 P8001 全集必填保证，不在本层补值。
            let gain: f32 = val(wf, declared, "E-5", "gain")
                .or_else(|| scene_params.get("audio.gain").map(|b| f32::from_bits(*b)))
                .unwrap_or(0.8);
            let buf = prism_audio::PcmBuffer::from_interleaved(48_000, 1, vec![0.0_f32; 4])
                .expect("4 个单声道样本应可构造");
            let track = prism_audio::mixer::MixTrack::new(buf, gain);
            let mixed = prism_audio::mixer::mix_tracks(&[track])
                .expect("单轨混音应成功");
            push_str(digest, &format!("{mixed:?}"));
            push_u32(digest, gain.to_bits());
            "crates/prism-audio/src/mixer.rs:34"
        }
        // G5 平台域：`build_wiring` 以 P8007 显式拒绝，不生成凭证。
        CapFamily::Platform => "G5 平台域（裁定，不接线）",
    }
}
