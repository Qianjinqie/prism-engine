//! W1-A2 验收测试：电影时间线（movie.rs）帧循环接通物理模拟
//! （ConstraintFlow，S3）。
//!
//! 覆盖任务书验收四条：
//! 1. 时间线驱动 sim 帧循环能渲染（不再走 R5010 拒绝）；
//! 2. 确定性：同参数两遍渲染字节级一致；
//! 3. seek 等价：跳帧后重算与顺序推进结果一致（对拍）；
//! 4. 签名敏感：改 SolveMode 参数（substeps / iterations / residual_eps）
//!    ⇒ 帧签名 / 缓存键变化（同参数永远同画面）。
//!
//! 全部用例走 DSL 端到端（`compile_v2` 真实产物）——渲染侧消费路径与
//! 影片生产路径同源。

use prism_core::node::StructureId;
use prism_core::signature::{cache_key, CacheDims};
use prism_dsl::compile_v2;
use prism_gpu::{CpuRefBackend, Framebuffer};
// 经 `pub mod movie` 公开面访问（不扩 crate 根导出——白名单只含 movie.rs
// 与 tests/ 新文件；模块内 pub 项即集成面）。
use prism_render::movie::{
    render_movie, render_movie_frame, sim_frame_signature, Deliverable, SimTimeline,
};

// ---------------------------------------------------------------------------
// DSL 夹具
// ---------------------------------------------------------------------------

/// 带 sim 的场景（DSL 端到端真实产物）：stage3d 相机 + 材质 + sim cloth。
/// fps=4 × duration=1s → 4 帧；sim 场景内帧号 = 影片帧号（start=0）。
const SRC_SIM: &str = concat!(
    "// prism-dsl 1\n",
    "movie \"simfilm\" { duration 1s fps 4 resolution 64x36 }\n",
    "scene \"cloth\" {\n",
    "  duration 1s\n",
    "  stage3d {\n",
    "    camera main { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    material m { albedo #c0392b roughness 0.4 }\n",
    "  }\n",
    "  sim cloth { grid 8 stiffness 0.9 damping 0.02 force wind { strength 2.5 direction(1,0,0) } }\n",
    "}\n",
);

/// 无 sim 的对照场景（同相机 / 材质 / 规格）。
const SRC_NO_SIM: &str = concat!(
    "// prism-dsl 1\n",
    "movie \"nosim\" { duration 1s fps 4 resolution 64x36 }\n",
    "scene \"bare\" {\n",
    "  duration 1s\n",
    "  stage3d {\n",
    "    camera main { orbit { center 0,0,0 radius 8 } fov 45deg }\n",
    "    material m { albedo #c0392b roughness 0.4 }\n",
    "  }\n",
    "}\n",
);

/// sim 编译产物（含节点摘要断言）。
fn sim_compile() -> prism_dsl::CompileV2Output {
    let compile: prism_dsl::CompileV2Output = compile_v2(SRC_SIM).expect("sim DSL 应编译成功");
    // sim 摘要登记且节点落入场景 node_ids（帧循环接线的入口）。
    let sim = compile.scenes[0].sim.as_ref().expect("sim 应登记到场景摘要");
    assert_eq!(sim.kind, "cloth", "sim 种类应落盘");
    assert!(
        compile.scenes[0].node_ids.contains(&sim.node_id),
        "sim 节点应进入场景 node_ids"
    );
    let node = compile.graph.node(sim.node_id).expect("sim 节点应在图中");
    assert_eq!(
        node.structure,
        StructureId::ConstraintFlow,
        "sim 块应投影为 S3 约束梯度流节点"
    );
    compile
}

/// sim 节点 id（帧签名查询用）。
fn sim_node_id(compile: &prism_dsl::CompileV2Output) -> u64 {
    compile.scenes[0].sim.as_ref().expect("sim 摘要").node_id
}

/// 暖橙粒子点像素计数（render 侧叠加色，逐字节判据）。
fn sim_dot_pixels(fb: &Framebuffer) -> usize {
    fb.pixels
        .chunks_exact(4)
        .filter(|px| px[0] == 255 && px[1] == 176 && px[2] == 66)
        .count()
}

/// 确定性摘要（帧序列拼接后 MD5）。
fn digest(d: &Deliverable) -> String {
    let mut buf: Vec<u8> = Vec::new();
    for fb in &d.frames {
        buf.extend_from_slice(&fb.pixels);
    }
    prism_render::farm::md5_hex(&buf)
}

// ---------------------------------------------------------------------------
// 验收 1：时间线驱动 sim 帧循环能渲染
// ---------------------------------------------------------------------------

#[test]
fn timeline_driven_sim_frame_loop_renders() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let mut device = CpuRefBackend;
    let d: Deliverable = render_movie(&compile, &mut device).expect("sim 时间线应可渲染（R5010 已移除）");

    // fps=4 × 1s → 4 帧；64x36。
    assert_eq!(d.frame_count(), 4, "fps=4 × 1s 应得 4 帧");
    assert_eq!(d.width, 64);
    assert_eq!(d.height, 36);
    for (i, fb) in d.frames.iter().enumerate() {
        assert_eq!(fb.pixels.len(), 64 * 36 * 4, "第 {i} 帧长度应恰为 w·h·4");
    }

    // 粒子叠加：暖橙圆点存在（帧循环推进后的粒子投影）。
    let dots: usize = d.frames.iter().map(sim_dot_pixels).sum();
    assert!(dots > 0, "sim 粒子应叠加为暖橙圆点（观测到 {dots} 个点像素）");

    // 对照：无 sim 场景无暖橙点、且画面与带 sim 不同（叠加真的发生了）。
    let control: prism_dsl::CompileV2Output = compile_v2(SRC_NO_SIM).expect("对照 DSL 应编译成功");
    let c: Deliverable = render_movie(&control, &mut device).expect("对照应可渲染");
    assert_eq!(c.frames.iter().map(sim_dot_pixels).sum::<usize>(), 0, "无 sim 不应出现暖橙点");
    assert_ne!(d.frames[0].pixels, c.frames[0].pixels, "sim 叠加应改变画面");

    // 帧循环推进：外力推动布料，首末帧不同。
    assert_ne!(d.frames[0].pixels, d.frames[3].pixels, "sim 帧循环应推进粒子");
}

// ---------------------------------------------------------------------------
// 验收 2：确定性（同参数两遍渲染字节级一致）
// ---------------------------------------------------------------------------

#[test]
fn same_params_render_bitwise_identical() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let mut device = CpuRefBackend;
    let a: Deliverable = render_movie(&compile, &mut device).expect("第一遍应可渲染");
    let b: Deliverable = render_movie(&compile, &mut device).expect("第二遍应可渲染");
    for (i, (x, y)) in a.frames.iter().zip(b.frames.iter()).enumerate() {
        assert_eq!(x.pixels, y.pixels, "第 {i} 帧两跑必须逐字节一致");
    }
    assert_eq!(digest(&a), digest(&b), "帧序列整体摘要必须一致");
}

// ---------------------------------------------------------------------------
// 验收 3：seek 等价（跳帧重算 ≡ 顺序推进）
// ---------------------------------------------------------------------------

#[test]
fn seek_jump_equals_sequential_advance() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let mut device = CpuRefBackend;
    let n: u64 = 4; // frame_count = fps × duration

    // A：顺序推进 0..=n-2（快照库随帧累积）。
    let mut sims_a: SimTimeline = SimTimeline::new();
    let mut sequential: Vec<Framebuffer> = Vec::new();
    for i in 0..n - 1 {
        let fb: Framebuffer =
            render_movie_frame(&compile, &mut device, i, &mut sims_a).expect("顺序帧应可渲染");
        sequential.push(fb);
    }
    // A 顺序推进到末帧。
    let last_seq: Framebuffer =
        render_movie_frame(&compile, &mut device, n - 1, &mut sims_a).expect("顺序末帧");
    sequential.push(last_seq);

    // B：注入 A 预热 0..n-2 的快照库，跳帧 n-1（前驱快照命中 + 步进一帧）。
    let mut sims_b: SimTimeline = SimTimeline::new();
    *sims_b.snapshots_mut() = sims_a.snapshots().clone();
    let jumped: Framebuffer =
        render_movie_frame(&compile, &mut device, n - 1, &mut sims_b).expect("跳帧应可渲染");
    assert_eq!(
        jumped.pixels, sequential[(n - 1) as usize].pixels,
        "快照装配跳帧 ≡ 顺序推进（逐字节）"
    );

    // 全片入口与逐帧入口同语义（seek 对拍的基线）。
    let whole: Deliverable = render_movie(&compile, &mut device).expect("全片应可渲染");
    let mut sims_c: SimTimeline = SimTimeline::new();
    for (i, fb) in whole.frames.iter().enumerate() {
        let one: Framebuffer =
            render_movie_frame(&compile, &mut device, i as u64, &mut sims_c).expect("逐帧");
        assert_eq!(one.pixels, fb.pixels, "第 {i} 帧：逐帧入口 ≡ 全片入口");
    }
    for (i, fb) in whole.frames.iter().enumerate() {
        assert_eq!(fb.pixels, sequential[i].pixels, "第 {i} 帧：全片 ≡ 顺序推进");
    }
}

#[test]
fn cold_seek_without_predecessor_snapshot_errors_p3012() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let mut device = CpuRefBackend;
    let mut sims: SimTimeline = SimTimeline::new();
    let err: prism_render::RenderError = render_movie_frame(&compile, &mut device, 2, &mut sims)
        .expect_err("冷 seek（帧 2 无前驱快照）必须结构化报错");
    assert_eq!(err.code, "P3012", "冷 seek 报 P3012，绝不静默 fallback 到初态");
    assert!(err.to_string().contains("建议"), "三段式必含建议段");
}

// ---------------------------------------------------------------------------
// 验收 4：签名敏感（模式元数据进帧签名 / 缓存键）
// ---------------------------------------------------------------------------

#[test]
fn mode_metadata_changes_frame_signature_and_cache_key() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let id: u64 = sim_node_id(&compile);

    // 同参数两调同签名（确定性铁律）。
    let base = sim_frame_signature(&compile, id, 4, 8, 1e-4).expect("基线签名");
    assert_eq!(
        base,
        sim_frame_signature(&compile, id, 4, 8, 1e-4).expect("复算签名"),
        "同参数必须同签名"
    );

    // 模式三元组任一变化 ⇒ 帧签名变化（(ε, n) 进签名）。
    let changed: [(u32, u32, f32); 3] = [(5, 8, 1e-4), (4, 9, 1e-4), (4, 8, 1e-3)];
    for (substeps, iterations, eps) in changed {
        let sig = sim_frame_signature(&compile, id, substeps, iterations, eps).expect("变体签名");
        assert_ne!(
            base, sig,
            "模式 ({substeps}, {iterations}, {eps}) 必须改变帧签名"
        );
        // 签名变 ⇒ 缓存键变（同维度）⇒ 不误复用（S3 迭代层缓存键 (ε, n) 纪律）。
        let dims = CacheDims {
            width: 64,
            height: 36,
            spp_tier: 2,
            tile: (0, 0),
            engine_version: 1,
            frame: 0,
        };
        assert_ne!(
            cache_key(&base, &dims),
            cache_key(&sig, &dims),
            "签名变化必须传导到缓存键"
        );
    }

    // 模式参数域非法 → R5018（防呆，无静默兜底）。
    let err: prism_render::RenderError =
        sim_frame_signature(&compile, id, 0, 8, 1e-4).expect_err("0 子步必须报错");
    assert_eq!(err.code, "R5018");
}

// ---------------------------------------------------------------------------
// W3 遗留：逐轮（帧内）发散判定进报告面——不得误报
// ---------------------------------------------------------------------------

/// 逐轮发散信号接通后，健康布料（XPBD 收缩投影）在整条时间线上都不得
/// 被误判为发散。这是本次接线的主要回归风险：把一个**新增**信号 OR 进
/// 公开判据 [`SimTimeline::diverged`]，最容易出的错不是漏报而是误报，
/// 那会让上游把正常影片当病态解算处理。
#[test]
fn per_iteration_divergence_signal_stays_quiet_for_healthy_cloth() {
    let compile: prism_dsl::CompileV2Output = sim_compile();
    let mut device = CpuRefBackend;
    let mut sims: SimTimeline = SimTimeline::new();
    let whole: Deliverable = render_movie(&compile, &mut device).expect("全片应可渲染");
    for i in 0..whole.frame_count() {
        render_movie_frame(&compile, &mut device, i as u64, &mut sims).expect("逐帧推进");
        assert!(
            !sims.diverged(),
            "第 {i} 帧：健康布料（刚度 0.9 / 阻尼 0.02）不得误报逐轮发散"
        );
    }
    // 渲染字节不受该信号影响（只观测、不改写迭代控制流）。
    let again: Deliverable = render_movie(&compile, &mut CpuRefBackend).expect("二次全片");
    for (i, (a, b)) in whole.frames.iter().zip(again.frames.iter()).enumerate() {
        assert_eq!(a.pixels, b.pixels, "第 {i} 帧：发散观测不得改变任何像素");
    }
}
