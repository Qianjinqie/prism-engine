//! S7 片尾 3D 光追序列：场景组装 + 相机轨道 + 行分带并行预渲染 + 后处理链
//! （架构 §2.6 / PRD §4-S7「片尾 flex·3D 光追收尾」）。
//!
//! 本模块是全片唯一的重计算段：310 帧 @960×540 spp48 路径追踪，产出
//! RGBA8 帧序列，由主入口经 [`prism_author::Track::Sequence`]（src_fps=30）
//! 逐帧对应到 30fps 成片。三条硬约束在此落地：
//!
//! 1. **确定性**（D-28/D-29）：像素级 [`prism_rt::integrator::render_pixel`]
//!    为纯函数，行分带 `std::thread::scope` 并行不改变任一像素的采样流
//!    ⇒ 并行与串行逐位一致（首帧双跑校验 [`VERIFY_PARALLEL`]，失败 panic）。
//! 2. **后处理显式化**：`Fbuf16F::to_framebuffer` 仅 clamp01×255 量化、
//!    无色调映射（架构 §2.6 明令禁止直接用于 HDR，否则高光死白）⇒
//!    Reinhard `x/(1+x)` + 精确 sRGB OETF 必须在本模块内显式完成；
//!    v3-B 追加显示域 bloom（与 2D 主链同域同参，见 [`post_process`]）。
//! 3. **预算门禁**（§8-R2 / 裁定②）：第 0 帧整帧计时 × 帧数外推 > 45min
//!    时按预授权限降级（先 640×360，仍超再 spp4），决策打印到运行日志。
//! 4. **玻璃切平面 Create 入场**（PRD §4-S7「4.0–5.0【Create】玻璃切平面
//!    淡入 0.5s」）：30fps 源帧 120 = 4.0s，此帧之前场景**不含**切平面；
//!    帧 120..135 为淡入段（0.5s = 15 源帧），两套场景各渲一帧后按
//!    `k=(f-120+1)/15` 在显示域（RGBA8）交叉混合；帧 ≥135 恒全平面。
//!    帧号换算与分派逻辑见 [`glass_gate`]。
//!
//! # 相对规格字面的偏差（均已确认无意图损失）
//! - **`EvalCtx::new` 实参序**：prism-core `node.rs` 实比为
//!   `new(global_seed, node_id, frame)`；规格字面
//!   `EvalCtx::new(frame_idx, NODE_3D, SEED_3D)` 会把帧号写进种子槽、
//!   常量写进帧槽，违背架构 §2.6「帧号入采样流 ⇒ 轨道运动有时间维噪声」
//!   的意图。故按意图落实为 `EvalCtx::new(SEED_3D, NODE_3D, frame_idx)`。
//! - **相机 `height` 语义**：架构 §2.6 未指定相对/绝对，按 PRD「高度
//!   2.2→3.0」的字面读法取**世界绝对高度** `eye.z = height`（视点始终
//!   位于内容区 z∈[0,1.44] 与面光 z=3.4 之间的经典四分之三俯视带）。
//! - **`prism-author` dev-dependencies 增补** `prism-core` / `prism-math`：
//!   `EvalCtx` / `Vec3` 无转导导出，example 必须直名（路径依赖，零第三方）。
//! - **v3 画质提升（spp 8→48 / bounces 3→6）**：S7 画质项。
//!   integrator.rs 既有对拍同值为 spp8/bounces3（cornell 参考渲染
//!   `RenderParams::new(4096, 3)` 及多数单测同款）——那是对拍基准不是
//!   画质上限；bounces 6 使玻璃透射（ior 1.5）+ 金属二次弹射有完整
//!   光路。预算探测（[`plan_budget`]）按实际耗时外推，> 30min 仍走
//!   §8-R2 降级（先 640×360，仍超再 spp4），纪律不变。

use std::thread;
use std::time::Instant;

use prism_core::EvalCtx;
use prism_io::fbuf::Fbuf16F;
use prism_math::Vec3;
use prism_postfx::denoise::AtrousSpatial;
use prism_postfx::effects::Bloom;
use prism_render::tile::TileEffect;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, MaterialKind, Paraboloid, Plane, RenderParams, Scene};
use prism_rt::light::Light;

// ---------------------------------------------------------------------------
// 模块常量（确定性契约的一部分：同常量 ⇒ 同输出）
// ---------------------------------------------------------------------------

/// 采样流节点 id（本模块自定义常量；入 `EvalCtx.node_id`）。
pub const NODE_3D: u64 = 0x5337_0007;
/// 采样流全局种子（本模块自定义常量；入 `EvalCtx.global_seed`）。
pub const SEED_3D: u64 = 0x5C17_3D17_2024;

/// 序列输出分辨率宽（架构 §2.6：960×540）。
pub const S7_WIDTH: u32 = 960;
/// 序列输出分辨率高。
pub const S7_HEIGHT: u32 = 540;
/// 序列源帧率（310 帧铺满 10.333s @30fps；与画布同帧率逐帧对应）。
pub const SRC_FPS: u32 = 30;
/// 每像素样本数（v3-C：架构 §2.6 原值 8 ⇒ 48，焦散/玻璃细节信噪比 ×6）。
pub const SPP: u32 = 48;
/// 最大弹射深度（v3-C：3 ⇒ 6，玻璃透射 + 金属二次弹射完整光路）。
pub const MAX_BOUNCES: u32 = 6;
/// Reinhard 曝光试值（面光辐亮度 14 量级：高光 ≈0.93 不死白、暗部保层次）。
pub const EXPOSURE: f32 = 1.0;
/// 竖直视场角（度；架构 §2.6 / PRD §4-S7）。
pub const FOV_DEG: f32 = 40.0;
/// 并行线程上限（架构 §2.6：`available_parallelism()` 上限 32）。
pub const MAX_THREADS: usize = 32;
/// 首帧并行 vs 串行逐位校验开关（架构 §2.6；失败即 panic）。
pub const VERIFY_PARALLEL: bool = true;
/// 渐黑起始帧（240/30 = 8.0s；PRD「8.0–8.3 淡出至黑」由预渲染帧承担）。
const FADE_START_FRAME: u32 = 240;
/// 全黑帧（264/30 = 8.8s；264..309 纯黑，落版文字由 2D 轨叠加在黑底上）。
const FADE_END_FRAME: u32 = 264;
/// 玻璃切平面 Create 入场起始帧（PRD §4-S7「4.0–5.0【Create】玻璃切平面
/// 淡入 0.5s」：源帧 120 = 120/30 = 4.0s，此帧之前切平面不可见）。
const GLASS_START_FRAME: u32 = 120;
/// 玻璃切平面淡入时长（源帧数：0.5s × 30fps = 15 帧）。
const GLASS_FADE_FRAMES: u32 = 15;
/// 淡入完成（全平面）起始帧 = [`GLASS_START_FRAME`] + [`GLASS_FADE_FRAMES`]
/// = 135（135/30 = 4.5s；此帧起切平面完全可见）。
const GLASS_FULL_FRAME: u32 = GLASS_START_FRAME + GLASS_FADE_FRAMES;
/// 预算上限（秒 = 45 分钟 = 2700s；裁定② / 架构 §8-R2；v3 升级 S7 源帧
/// 率 15→30fps 使源帧数 155→310，渲染总量同比翻倍，故上限同步放宽到
/// 45 分钟以保住 §8-R2 降级纪律不变）。
const BUDGET_SEC: f32 = 45.0 * 60.0;
/// à-trous 空域降噪迭代趟数（架构 §2.6）。
const DENOISE_ITERATIONS: u32 = 3;
/// à-trous 色差边缘停止带宽 σ（架构 §2.6）。
const DENOISE_SIGMA_COLOR: f32 = 0.1;

/// 相机轨道关键帧（时刻 s, azimuth°, radius, height）；8.0s 后定格
/// （架构 §2.6 / PRD §4-S7「相机拉远升高后展示全貌」）。
const CAMERA_KEYS: [(f32, f32, f32, f32); 4] = [
    (0.0, 30.0, 4.2, 2.2),
    (4.0, 42.0, 4.2, 2.2),
    (6.5, 55.0, 3.8, 2.2),
    (8.0, 55.0, 4.6, 3.0),
];

/// 相机注视目标（架构 §2.6：碗心偏上，兼顾抛物面与切平面）。
const CAMERA_TARGET_XY: (f32, f32) = (0.0, 0.35);
/// 相机注视目标高度（世界 z）。
const CAMERA_TARGET_Z: f32 = 0.5;

// ---------------------------------------------------------------------------
// 场景组装（架构 §2.6）
// ---------------------------------------------------------------------------

/// 组装 S7 场景（z 向上）：金属抛物面（焦散碗）+ 无限地面 [+ 玻璃切平面]
/// + 矩形面光 + 环境光。纯函数：同输入同输出，无全局状态。
///
/// `with_glass` 控制玻璃切平面是否在场（Create 入场分镜：4.0s 前不在场，
/// 帧号换算见 [`glass_gate`]）；其余元素（碗 / 地面 / 灯光）与开关无关，
/// 两套场景仅差这一个平面 ⇒ 淡入段可对两套产物直接做显示域交叉混合。
///
/// 场景叙事（PRD §4-S7）：S5 的曲线 `z=x²` 旋转成面 `z=x²+y²`（抛物面
/// 碗），S4 的切线推广为过切点 `P=(0.8,0.8,1.28)` 的玻璃切平面；面光经
/// 金属碗 spec 瓣聚焦 + 玻璃透射在地面投出焦散光池。
pub fn build_s7_scene(with_glass: bool) -> Scene {
    // 金属抛物面（焦散碗）：z = x² + y²，r ∈ [0, 1.2] ⇒ z ∈ [0, 1.44]。
    let bowl = Paraboloid { k: 1.0, r_max: 1.2 };
    let bowl_material = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.95, 0.93, 0.88), // base_color：暖白金属
        0.9,                         // metalness：光泽金属
        0.15,                        // roughness：偏镜面（聚焦焦散）
        1.5,                         // ior
        0.0,                         // transmission：不透
        0.0,                         // subsurface
        Vec3::splat(0.3),            // subsurface_radius（透传项，须 > 0）
    ));

    // 无限地面（ Lambert 深蓝灰）：焦散光池自然铺开，无盒缘穿帮。
    let ground = Plane::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        None,
    );
    let ground_material = MaterialKind::Lambert {
        albedo: Vec3::new(0.18, 0.20, 0.24),
    };

    // 玻璃切平面：过切点 P=(0.8,0.8,1.28)，法线 (-1.6,-1.6,1)（即平面
    // z = 1.6x + 1.6y − 1.28），边长 1.7 的有限切片（half_extent 0.85）。
    // 按开关入场：Create 分镜 4.0s（源帧 120）前不在场（见 glass_gate）。
    let mut planes = vec![(ground, ground_material)];
    if with_glass {
        let glass_material = MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.95, 0.97, 1.0), // base_color：极淡冷白（透射着色）
            0.0,                        // metalness：电介质
            0.02,                       // roughness：近镜面（清晰折射）
            1.5,                        // ior：玻璃
            0.9,                        // transmission：玻璃
            0.0,                        // subsurface
            Vec3::splat(0.3),
        ));
        planes.push((
            Plane::new(
                Vec3::new(0.8, 0.8, 1.28),
                Vec3::new(-1.6, -1.6, 1.0),
                Some(0.85),
            ),
            glass_material,
        ));
    }

    Scene {
        spheres: vec![],
        boxes: vec![],
        paraboloids: vec![(bowl, bowl_material)],
        planes,
        meshes: Vec::new(),
        lights: vec![
            // 矩形面光：自上方倾斜照射（非几何 ⇒ NEE 权重恒 1，light.rs 契约）。
            Light::Rect {
                center: Vec3::new(0.8, 0.8, 3.4),
                edge_u: Vec3::new(1.5, 0.0, 0.0),
                edge_v: Vec3::new(0.3, 1.3, -0.6),
                radiance: Vec3::new(14.0, 12.0, 9.0),
            },
            // 环境光保底（仅逃逸背景贡献，无重复计数）。
            Light::Ambient {
                radiance: Vec3::new(0.05, 0.06, 0.09),
            },
        ],
        medium: None,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 相机轨道（纯函数 camera_at）
// ---------------------------------------------------------------------------

/// Smoothstep 插值（纯函数；`±×` only，无超越函数）。
fn smoothstep(s: f32) -> f32 {
    let x = s.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// 时刻 `t`（秒）的相机（纯函数）：由关键帧（azimuth°, radius, height）
/// 段内 smoothstep 插值解算视点，注视目标 `(0, 0.35, 0.5)`，world_up
/// `(0,0,1)`，竖直 fov [`FOV_DEG`]，针孔光圈（无景深，保确定性）。
///
/// `t <= 0` 取首关键帧；`t >= 8.0` 取末关键帧并定格至序列尾
/// （10.333s）。height 按世界绝对高度解算 `eye.z = height`（见模块头偏差）。
pub fn camera_at(t: f32) -> Camera {
    let target = Vec3::new(CAMERA_TARGET_XY.0, CAMERA_TARGET_XY.1, CAMERA_TARGET_Z);

    let (azimuth_deg, radius, height) = if t <= CAMERA_KEYS[0].0 {
        let k = CAMERA_KEYS[0];
        (k.1, k.2, k.3)
    } else if t >= CAMERA_KEYS[CAMERA_KEYS.len() - 1].0 {
        let k = CAMERA_KEYS[CAMERA_KEYS.len() - 1];
        (k.1, k.2, k.3)
    } else {
        // 定位所在段：推进 i 直至 t 落入 (KEYS[i].0, KEYS[i+1].0]。
        let mut i = 0usize;
        while i + 1 < CAMERA_KEYS.len() && t > CAMERA_KEYS[i + 1].0 {
            i += 1;
        }
        let (t0, az0, r0, h0) = CAMERA_KEYS[i];
        let (t1, az1, r1, h1) = CAMERA_KEYS[i + 1];
        let s = smoothstep((t - t0) / (t1 - t0));
        (
            az0 + (az1 - az0) * s,
            r0 + (r1 - r0) * s,
            h0 + (h1 - h0) * s,
        )
    };

    // 轨道解算：目标点 xy 平面内按 azimuth/radius 布点，z 取绝对高度。
    let azimuth = azimuth_deg.to_radians();
    let eye = Vec3::new(
        target.x + radius * azimuth.cos(),
        target.y + radius * azimuth.sin(),
        height,
    );
    let focus = (eye - target).length();

    Camera::new(
        eye,
        target,
        Vec3::new(0.0, 0.0, 1.0), // world_up：z 向上
        FOV_DEG,
        0.0,             // aperture_radius：针孔
        focus,           // focus_distance（针孔下不参与，但须 > 0）
        0.0,             // shutter_open
        1.0 / SRC_FPS as f32, // shutter_close
    )
}

// ---------------------------------------------------------------------------
// 预渲染（行分带并行 + 首帧校验 + 预算门禁）
// ---------------------------------------------------------------------------

/// 预渲染产物：帧序列（RGBA8 行主序，每帧 `width*height*4` 字节）+ 实际
/// 分辨率（触发 §8-R2 降级时与 [`S7_WIDTH`]/[`S7_HEIGHT`] 不同）。
#[derive(Debug, Clone)]
pub struct SequenceFrames {
    /// 帧宽（像素）。
    pub width: u32,
    /// 帧高（像素）。
    pub height: u32,
    /// RGBA8 帧序列。
    pub frames: Vec<Vec<u8>>,
}

/// 渲染计划（预算探测 + §8-R2 降级后的实际参数）。
struct RenderPlan {
    width: u32,
    height: u32,
    spp: u32,
    estimated_sec: f32,
}

/// 预渲染 `n` 帧 S7 序列（架构 §2.6）：
/// 1. 预算探测（两个代表帧整帧计时取慢者 × n 外推；超 45min 按 §8-R2 降级）；
/// 2. 逐帧经 [`render_sequence_frame`] 出图（含玻璃切平面 Create 入场调度：
///    4.0s 前无切平面、帧 120..135 显示域淡入、帧 ≥135 全平面）；
/// 3. 首帧并行 vs 串行双跑逐位校验（[`VERIFY_PARALLEL`]，失败 panic）。
pub fn prerender_frames(n: usize) -> SequenceFrames {
    // 两套场景仅差玻璃切平面（Create 入场开关），其余元素同源。
    let scene_bare = build_s7_scene(false);
    let scene_glass = build_s7_scene(true);
    let plan = plan_budget(&scene_bare, &scene_glass, n);
    println!(
        "[s7] 预渲染计划：{}×{}，spp={}，max_bounces={}；代表帧外推全序列 {:.1} 分钟（预算 45 分钟）",
        plan.width,
        plan.height,
        plan.spp,
        MAX_BOUNCES,
        plan.estimated_sec / 60.0
    );

    let params = RenderParams::new(plan.spp, MAX_BOUNCES);
    let res = (plan.width, plan.height);
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(n);
    for f in 0..n as u32 {
        if f == 0 && VERIFY_PARALLEL {
            // 首帧双跑：并行结果必须与串行结果逐位一致（D-28 像素域纯函数契约）。
            // 帧 0 处于 Create 入场前（无切平面场景），与 render_sequence_frame
            // 内部对帧 0 的调用同参同场景，校验等价于生产路径首帧校验。
            let cam = camera_at(0.0);
            let ctx = EvalCtx::new(SEED_3D, NODE_3D, 0);
            let parallel = render_hdr(&scene_bare, &cam, &ctx, res, &params, true);
            let serial = render_hdr(&scene_bare, &cam, &ctx, res, &params, false);
            assert!(
                bits_eq(&parallel, &serial),
                "VERIFY_PARALLEL 失败：首帧并行与串行渲染不逐位一致（render_pixel 纯函数契约被破坏）"
            );
            println!(
                "[s7] VERIFY_PARALLEL：首帧 {}×{} spp{} 并行 vs 串行逐位一致 ✓",
                res.0, res.1, plan.spp
            );
        }
        frames.push(render_sequence_frame(f, &scene_bare, &scene_glass, res, &params));
        if f % 15 == 0 {
            println!("[s7] 帧 {f}/{n} 完成");
        }
    }
    SequenceFrames {
        width: res.0,
        height: res.1,
        frames,
    }
}

/// 预算探测（§8-R2）：对两个代表帧——帧 0（无切平面，Create 入场前）与
/// 帧 [`GLASS_FULL_FRAME`]（全切平面，含透射负载）——整帧计时取**较慢者**
/// × n 外推；> 45min 依次降级 640×360 → spp4（裁定②预授），每次重新探测
/// 外推，全部决策打印到运行日志。
///
/// 取慢者而非均值：切平面透射是全程最重负载，均值口径会因透射仅覆盖
/// 190/310 帧而低估约 15%，保守口径保证 45min 门禁不被低估。
fn plan_budget(scene_bare: &Scene, scene_glass: &Scene, n: usize) -> RenderPlan {
    let probe = |scene: &Scene, frame: u32, width: u32, height: u32, spp: u32| -> f32 {
        let cam = camera_at(frame as f32 / SRC_FPS as f32);
        let ctx = EvalCtx::new(SEED_3D, NODE_3D, frame as u64);
        let params = RenderParams::new(spp, MAX_BOUNCES);
        let started = Instant::now();
        let hdr = render_hdr(scene, &cam, &ctx, (width, height), &params, true);
        let _frame = post_process(&hdr, (width, height), frame);
        started.elapsed().as_secs_f32()
    };

    let mut plan = RenderPlan {
        width: S7_WIDTH,
        height: S7_HEIGHT,
        spp: SPP,
        estimated_sec: 0.0,
    };

    // 三级预算：原画质 → 降级① 640×360 → 降级② spp4；每级双帧探测取慢者。
    for level in 0..3u8 {
        let dt_bare = probe(scene_bare, 0, plan.width, plan.height, plan.spp);
        let dt_glass = probe(scene_glass, GLASS_FULL_FRAME, plan.width, plan.height, plan.spp);
        let dt = dt_bare.max(dt_glass);
        plan.estimated_sec = dt * n as f32;
        println!(
            "[s7] 预算探测（级 {level}，{w}×{h} spp{spp}）：无平面帧 0 {dt_bare:.2}s / 全平面帧 {gf} {dt_glass:.2}s ⇒ 取慢者外推 {n} 帧 {est:.1}s（{min:.1} min）",
            level = level,
            w = plan.width,
            h = plan.height,
            spp = plan.spp,
            gf = GLASS_FULL_FRAME,
            dt_bare = dt_bare,
            dt_glass = dt_glass,
            n = n,
            est = plan.estimated_sec,
            min = plan.estimated_sec / 60.0,
        );
        if plan.estimated_sec <= BUDGET_SEC {
            break;
        }
        match level {
            0 => {
                println!(
                    "[s7] R2 降级①：{}×{} ⇒ 640×360（0.44× 像素）",
                    plan.width, plan.height
                );
                plan.width = 640;
                plan.height = 360;
            }
            1 => {
                println!("[s7] R2 降级②：spp {} ⇒ 4（0.5× 样本）", plan.spp);
                plan.spp = 4;
            }
            _ => {
                // 级 2 仍超预算：保持最低画质执行（30min 门禁失守，风险已打印）。
                println!("[s7] R2 降级用尽仍超预算：保持 {}×{} spp{} 执行", plan.width, plan.height, plan.spp);
                break;
            }
        }
    }
    plan
}

// ---------------------------------------------------------------------------
// 单帧装配（玻璃切平面 Create 入场调度）
// ---------------------------------------------------------------------------

/// 玻璃切平面入场调度（纯函数）：帧号 ⇒ 该帧切平面的可见状态。
///
/// 帧号换算（30fps 源帧 ↔ 片上时刻）：源帧 120 = 120/30 = 4.0s，即 PRD
/// §4-S7「4.0–5.0【Create】玻璃切平面淡入 0.5s」的起点；淡入 0.5s ≈
/// 0.5 × 30 = 15 源帧 ⇒ 帧 120..135 为淡入段，`k=(f-120+1)/15`（帧 120 已
/// 1/15 可见、帧 134 全可见）；帧 ≥135 恒全平面。
#[derive(Debug, Clone, Copy, PartialEq)]
enum GlassGate {
    /// 切平面未入场（帧 < 120）：仅无平面场景参与渲染。
    Bare,
    /// 淡入中（120 ≤ 帧 < 135）：两套场景各渲一帧，显示域按 `k` 交叉混合。
    Fading { k: f32 },
    /// 全平面（帧 ≥ 135）：仅含平面场景参与渲染。
    Full,
}

/// 帧 `f` 的切平面入场调度（纯函数；[`render_sequence_frame`] 的唯一分派依据）。
fn glass_gate(f: u32) -> GlassGate {
    if f < GLASS_START_FRAME {
        GlassGate::Bare
    } else if f < GLASS_FULL_FRAME {
        GlassGate::Fading {
            k: (f - GLASS_START_FRAME + 1) as f32 / GLASS_FADE_FRAMES as f32,
        }
    } else {
        GlassGate::Full
    }
}

/// 渲染单帧序列帧（含 Create 入场调度；逐位确定性：同帧号 ⇒ 同字节）。
///
/// - [`GlassGate::Bare`] / [`GlassGate::Full`]：单场景渲染 + 后处理；
/// - [`GlassGate::Fading`]：两套场景各渲一帧（同一 `EvalCtx` ⇒ 采样流仅
///   场景不同），各自后处理后在**显示域**（RGBA8）按 `k` 交叉混合——淡入
///   是感知域操作，显示域混合即标准交叉淡化，且避免在 HDR 域合成改变
///   降噪口径（降噪在各自后处理链内完成）。
fn render_sequence_frame(
    f: u32,
    scene_bare: &Scene,
    scene_glass: &Scene,
    res: (u32, u32),
    params: &RenderParams,
) -> Vec<u8> {
    let cam = camera_at(f as f32 / SRC_FPS as f32);
    let ctx = EvalCtx::new(SEED_3D, NODE_3D, f as u64);
    let render = |scene: &Scene| render_hdr(scene, &cam, &ctx, res, params, true);
    match glass_gate(f) {
        GlassGate::Bare => post_process(&render(scene_bare), res, f),
        GlassGate::Fading { k } => {
            let bare = post_process(&render(scene_bare), res, f);
            let glass = post_process(&render(scene_glass), res, f);
            blend_rgba8(&bare, &glass, k)
        }
        GlassGate::Full => post_process(&render(scene_glass), res, f),
    }
}

/// 显示域交叉混合两帧 RGBA8（逐通道 `round` + `clamp`；纯函数）。
///
/// `k=0` 逐位返回 `a`（`1.0×a + 0.0×b` 在 f32 精确）、`k=1` 逐位返回 `b`
/// （`0.0×a + 1.0×b` 精确）；alpha 通道两帧均 255 ⇒ 混合后仍 255
/// （`round` 吸收至多 1ulp 误差）。
fn blend_rgba8(a: &[u8], b: &[u8], k: f32) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "blend_rgba8：两帧字节数不一致");
    let mut out: Vec<u8> = vec![0u8; a.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let v = a[i] as f32 * (1.0 - k) + b[i] as f32 * k;
        *o = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 渲染一帧线性 HDR（`[f32;4]` 行主序）。`parallel = true` 时按行分带
/// `std::thread::scope` 并行（上限 [`MAX_THREADS`]）；`false` 为串行。
///
/// 像素级 [`render_pixel`] 为纯函数（采样流由 `EvalCtx` + 像素坐标 +
/// 样本序号哈希派生，无跨像素状态）⇒ 两种模式输出逐位一致。
fn render_hdr(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    res: (u32, u32),
    params: &RenderParams,
    parallel: bool,
) -> Vec<[f32; 4]> {
    let (w, h) = res;
    let mut buf: Vec<[f32; 4]> = vec![[0.0; 4]; (w as usize) * (h as usize)];

    let threads = if parallel {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, MAX_THREADS)
    } else {
        1
    };

    if threads <= 1 || h < 2 {
        render_band(scene, cam, ctx, res, params, &mut buf, 0);
        return buf;
    }

    // 行分带：把整幅按连续行切为 `threads` 段（末段可更短），各段经
    // `split_at_mut` 获得不相交可变切片 ⇒ 无锁、无数据竞争。
    let rows_per_band = (h as usize).div_ceil(threads) as u32;
    thread::scope(|s| {
        let mut rest: &mut [[f32; 4]] = &mut buf;
        let mut y0: u32 = 0;
        while y0 < h {
            let rows = rows_per_band.min(h - y0);
            let (band, tail) = rest.split_at_mut((rows * w) as usize);
            rest = tail;
            s.spawn(move || render_band(scene, cam, ctx, res, params, band, y0));
            y0 += rows;
        }
    });
    buf
}

/// 渲染一个行带（`band` 为第 `y0` 行起的连续像素，行主序）。
fn render_band(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    res: (u32, u32),
    params: &RenderParams,
    band: &mut [[f32; 4]],
    y0: u32,
) {
    let w = res.0;
    for (i, px) in band.iter_mut().enumerate() {
        let x = (i as u32) % w;
        let y = y0 + (i as u32) / w;
        *px = render_pixel(scene, cam, ctx, (x, y), res, params);
    }
}

/// 逐位比较两个 HDR 帧（`to_bits` 口径：区分 ±0.0，NaN 视为不一致）。
fn bits_eq(a: &[[f32; 4]], b: &[[f32; 4]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(p, q)| (0..4).all(|c| p[c].to_bits() == q[c].to_bits()))
}

// ---------------------------------------------------------------------------
// 后处理链（架构 §2.6：tonemap + OETF 必须显式）
// ---------------------------------------------------------------------------

/// 后处理：线性 HDR → [`Fbuf16F`] → à-trous 空域降噪 → Reinhard 色调映射
/// → 精确 sRGB OETF → **显示域 bloom**（v3-B）→ u8 RGBA8；并按帧号施加
/// 淡黑尾（落版由 2D 轨叠加）。
///
/// 禁直通 `Fbuf16F::to_framebuffer`（仅 clamp01×255、无 tonemap ⇒ HDR 死白）。
///
/// v3-B 的 bloom 落在**显示域**（OETF 之后、fade 之前）：与 2D 主链
/// （`film_postfx::post_chain`）同域同参。S7 是**预渲染成品位图**
/// （Sequence 轨注入，届时线性 HDR 已不存在），其泛光只能在内含链里
/// 自含；主入口据此按 `film_postfx::s7_body_start_frame()`（= 2405）
/// 将 S7 体帧排除在 2D 主链之外——两侧各一次 bloom，全片高光泛光
/// 剂量一致且不双重叠加。阈值/强度/半径复用 `film_postfx` 常量
/// （全片唯一真值）。fade 在 bloom 之后：渐黑是时序效应，压暗不应
/// 再产生新泛光。
fn post_process(hdr: &[[f32; 4]], res: (u32, u32), frame_idx: u32) -> Vec<u8> {
    let (w, h) = res;

    // 线性 HDR → f16 量化缓冲（写入即量化；降噪在量化域进行）。
    let mut fbuf = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            fbuf.set(x, y, hdr[(y as usize) * (w as usize) + x as usize]);
        }
    }

    // à-trous 空域降噪（3 趟、σ=0.1；有理边缘停止核，确定性）。
    let denoised = AtrousSpatial {
        iterations: DENOISE_ITERATIONS,
        sigma_color: DENOISE_SIGMA_COLOR,
    }
    .eval_window(&fbuf);

    // Reinhard + sRGB OETF → 显示域缓冲（fade 之前的中间态）。
    let mut display = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let px = denoised.get(x, y);
            let mut enc = [0.0f32; 4];
            for c in 0..3 {
                let v = (px[c] * EXPOSURE).max(0.0);
                enc[c] = srgb_oETF(v / (1.0 + v)); // Reinhard → OETF
            }
            enc[3] = 1.0; // alpha 不透明（bloom 亮部提取对 A 同样生效）
            display.set(x, y, enc);
        }
    }

    // 显示域 bloom（v3-B：S7 补镜头泛光；与 2D 主链同域同参的纯函数）。
    let bloomed = Bloom {
        threshold: crate::film_postfx::BLOOM_THRESHOLD,
        radius: crate::film_postfx::BLOOM_RADIUS,
        intensity: crate::film_postfx::BLOOM_INTENSITY,
    }
    .eval_window(&display);

    // 量化 + 淡黑尾（fade 在 bloom 之后：压暗不产生新泛光）。
    let fade = fade_factor(frame_idx);
    let mut out: Vec<u8> = vec![0u8; (w as usize) * (h as usize) * 4];
    let mut idx = 0usize;
    for y in 0..h {
        for x in 0..w {
            let px = bloomed.get(x, y);
            for c in 0..3 {
                out[idx] = (px[c] * 255.0 * fade).round().clamp(0.0, 255.0) as u8;
                idx += 1;
            }
            out[idx] = 255; // alpha 不透明
            idx += 1;
        }
    }
    out
}

/// 精确 sRGB OETF（IEC 61966-2-1）：线性 → 显示编码，分段连续。
/// 命名保留规范术语「OETF」全大写（架构 §2.6 同款写法）。
#[allow(non_snake_case)]
fn srgb_oETF(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// 淡黑尾因子：帧号 ≥ [`FADE_START_FRAME`] 起线性渐黑，[`FADE_END_FRAME`]
/// 全黑，之后纯黑（落版文字自行淡入淡出，叠在黑底 Sequence 上）。
fn fade_factor(frame_idx: u32) -> f32 {
    let span = (FADE_END_FRAME - FADE_START_FRAME) as f32;
    ((FADE_END_FRAME as f32 - frame_idx as f32) / span).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// 测试（纯函数级；不触发任何光追渲染）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 相机关键帧字面值（架构 §2.6）：t=0/4/6.5/8.0 及定格段取值。
    #[test]
    fn camera_at_matches_keyframes() {
        let c0 = camera_at(0.0);
        let az = 30.0f32.to_radians();
        assert!((c0.eye.x - 4.2 * az.cos()).abs() < 1e-4, "t=0 视点 x");
        assert!((c0.eye.y - (0.35 + 4.2 * az.sin())).abs() < 1e-4, "t=0 视点 y");
        assert!((c0.eye.z - 2.2).abs() < 1e-4, "t=0 视点 z（绝对高度）");
        assert!(c0.aperture_radius == 0.0, "针孔光圈");

        // t=8.0 及定格至序列尾（10.333s）：末关键帧 (55°, 4.6, 3.0)。
        let c8 = camera_at(8.0);
        let c_end = camera_at(310.0 / SRC_FPS as f32);
        let az55 = 55.0f32.to_radians();
        for c in [&c8, &c_end] {
            assert!((c.eye.x - 4.6 * az55.cos()).abs() < 1e-4);
            assert!((c.eye.y - (0.35 + 4.6 * az55.sin())).abs() < 1e-4);
            assert!((c.eye.z - 3.0).abs() < 1e-4);
        }
        for (a, b) in [
            (c8.eye.x, c_end.eye.x),
            (c8.eye.y, c_end.eye.y),
            (c8.eye.z, c_end.eye.z),
        ] {
            assert_eq!(a.to_bits(), b.to_bits(), "8.0s 后定格");
        }
    }

    /// 段内 smoothstep：方位角单调不减、中点落在两端之间、焦点距有限。
    #[test]
    fn camera_at_interpolates_smoothly() {
        let angle = |t: f32| {
            let e = camera_at(t).eye;
            (e.y - 0.35).atan2(e.x)
        };
        let a0 = angle(0.0);
        let a_mid = angle(5.0); // 段 (4, 6.5] 内
        let a_end = angle(8.0);
        assert!(a_mid > a0 && a_mid < a_end, "方位角 30°→55° 单调推进");
        // 焦点距（视点到目标）落在关键帧包络内。
        for t in [0.0, 2.0, 5.0, 7.0, 8.0, 10.0] {
            let d = (camera_at(t).eye - Vec3::new(0.0, 0.35, 0.5)).length();
            assert!(d.is_finite() && d > 4.0 && d < 6.0, "焦点距越界：{d}");
        }
        // smoothstep 端点与中点。
        assert!((smoothstep(0.0)).abs() < 1e-6);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);
    }

    /// 场景结构与材质契约（架构 §2.6 字面值；含 Create 入场前后两变体）。
    #[test]
    fn scene_structure_matches_spec() {
        let scene = build_s7_scene(true);
        assert!(scene.spheres.is_empty() && scene.boxes.is_empty());
        assert!(scene.medium.is_none());
        assert_eq!(scene.paraboloids.len(), 1);
        assert_eq!(scene.planes.len(), 2);
        assert_eq!(scene.lights.len(), 2);

        let (bowl, mat) = &scene.paraboloids[0];
        assert!((bowl.k - 1.0).abs() < 1e-6 && (bowl.r_max - 1.2).abs() < 1e-6);
        match mat {
            MaterialKind::Pbr(b) => {
                assert!((b.metalness - 0.9).abs() < 1e-6);
                assert!((b.roughness - 0.15).abs() < 1e-6);
                assert!((b.transmission).abs() < 1e-6);
            }
            _ => panic!("抛物面必须为 Pbr"),
        }

        // 平面 0 = 无限地面（Lambert 深蓝灰）。
        assert!(scene.planes[0].0.half_extent.is_none());
        match &scene.planes[0].1 {
            MaterialKind::Lambert { albedo } => {
                assert!((albedo.x - 0.18).abs() < 1e-6);
                assert!((albedo.y - 0.20).abs() < 1e-6);
                assert!((albedo.z - 0.24).abs() < 1e-6);
            }
            _ => panic!("地面必须为 Lambert"),
        }
        // 平面 1 = 玻璃切平面（过切点、法线已单位化、半宽 0.85）。
        assert_eq!(scene.planes[1].0.half_extent, Some(0.85));
        let n = scene.planes[1].0.normal;
        let expected = Vec3::new(-1.6, -1.6, 1.0).normalize();
        assert!((n - expected).length() < 1e-5, "法线须单位化到 (-1.6,-1.6,1) 方向");
        match &scene.planes[1].1 {
            MaterialKind::Pbr(b) => {
                assert!((b.transmission - 0.9).abs() < 1e-6);
                assert!((b.roughness - 0.02).abs() < 1e-6);
                assert!((b.ior - 1.5).abs() < 1e-6);
            }
            _ => panic!("切平面必须为 Pbr 玻璃"),
        }

        // 灯光：Rect + Ambient。
        match scene.lights[0] {
            Light::Rect { center, radiance, .. } => {
                assert!((center - Vec3::new(0.8, 0.8, 3.4)).length() < 1e-6);
                assert!((radiance - Vec3::new(14.0, 12.0, 9.0)).length() < 1e-6);
            }
            _ => panic!("首灯必须为 Rect"),
        }
        match scene.lights[1] {
            Light::Ambient { radiance } => {
                assert!((radiance - Vec3::new(0.05, 0.06, 0.09)).length() < 1e-6);
            }
            _ => panic!("次灯必须为 Ambient"),
        }

        // 无切平面变体（Create 入场前，帧 < 60）：仅无限地面，其余同源。
        let bare = build_s7_scene(false);
        assert!(bare.spheres.is_empty() && bare.boxes.is_empty());
        assert!(bare.medium.is_none());
        assert_eq!(bare.planes.len(), 1, "入场前不得有切平面");
        assert_eq!(bare.paraboloids.len(), 1);
        assert_eq!(bare.lights.len(), 2);
        assert!(bare.planes[0].0.half_extent.is_none(), "唯一平面须为无限地面");
        assert_eq!(scene.planes.len(), bare.planes.len() + 1, "两套场景仅差切平面");
    }

    /// 淡黑曲线：240 起线性、264 全黑、之后纯黑、之前恒 1（30fps 帧号）。
    #[test]
    fn fade_curve_is_linear_to_black() {
        assert!((fade_factor(0) - 1.0).abs() < 1e-6);
        assert!((fade_factor(119) - 1.0).abs() < 1e-6);
        assert!((fade_factor(239) - 1.0).abs() < 1e-6);
        assert!((fade_factor(252) - 0.5).abs() < 1e-6);
        assert!((fade_factor(263) - 1.0 / 24.0).abs() < 1e-6);
        assert!(fade_factor(264) == 0.0);
        assert!(fade_factor(309) == 0.0);
    }

    /// sRGB OETF：端点钉死 + 单调 + 分段连续性。
    #[test]
    fn srgb_oetf_endpoints_and_monotone() {
        assert!(srgb_oETF(0.0) == 0.0);
        assert!((srgb_oETF(1.0) - 1.0).abs() < 1e-6);
        let mut prev = -1.0f32;
        let mut x = 0.0f32;
        while x <= 1.0 {
            let y = srgb_oETF(x);
            assert!(y >= prev, "OETF 必须单调不减");
            assert!((0.0..=1.0).contains(&y));
            prev = y;
            x += 0.01;
        }
        // 分段点 0.0031308 处两枝近似连续（差 < 1e-3）。
        let lo = 12.92 * 0.0031308;
        let hi = 1.055 * 0.0031308f32.powf(1.0 / 2.4) - 0.055;
        assert!((lo - hi).abs() < 1e-3, "分段连续：{lo} vs {hi}");
    }

    /// 逐位比较：±0.0 位模式不同 ⇒ 不一致（校验口径严于 `==`）。
    #[test]
    fn bits_eq_is_bitwise() {
        let a = [
            [0.0f32, 1.0, -0.0, f32::MIN_POSITIVE],
            [1.0, 2.0, 3.0, 4.0],
        ];
        let b = a;
        assert!(bits_eq(&a, &b));
        let c = [
            [-0.0f32, 1.0, -0.0, f32::MIN_POSITIVE],
            [1.0, 2.0, 3.0, 4.0],
        ];
        assert!(!bits_eq(&a, &c), "+0.0 与 -0.0 位模式不同");
        assert!(!bits_eq(&a, &a[..1]), "长度不同必须不一致");
    }

    /// 玻璃切平面 Create 入场调度（PRD §4-S7「4.0–5.0【Create】玻璃切平面
    /// 淡入 0.5s」）：帧号换算 + 三段分派 + k 公式 + 单调性。
    #[test]
    fn glass_gate_follows_create_beat() {
        // 帧号换算：源帧 120 = 4.0s（30fps）；淡入 0.5s = 15 源帧（±1 源帧容差）。
        assert!((GLASS_START_FRAME as f32 / SRC_FPS as f32 - 4.0).abs() < 1e-6);
        assert!((GLASS_FADE_FRAMES as f32 / SRC_FPS as f32 - 0.5).abs() <= 1.0 / SRC_FPS as f32);
        // 三段分派：<120 不入场；120..135 淡入；≥135 全平面。
        assert_eq!(glass_gate(0), GlassGate::Bare);
        assert_eq!(glass_gate(GLASS_START_FRAME - 1), GlassGate::Bare);
        assert_eq!(
            glass_gate(GLASS_START_FRAME),
            GlassGate::Fading { k: 1.0 / GLASS_FADE_FRAMES as f32 }
        );
        assert_eq!(
            glass_gate(GLASS_START_FRAME + 4),
            GlassGate::Fading { k: 5.0 / GLASS_FADE_FRAMES as f32 }
        );
        assert_eq!(glass_gate(GLASS_FULL_FRAME - 1), GlassGate::Fading { k: 1.0 });
        assert_eq!(glass_gate(GLASS_FULL_FRAME), GlassGate::Full);
        assert_eq!(glass_gate(154), GlassGate::Full);
        // 淡入段 k 严格递增（帧 120 → 1/15，帧 134 → 1）。
        let mut prev = 0.0f32;
        for f in GLASS_START_FRAME..GLASS_FULL_FRAME {
            match glass_gate(f) {
                GlassGate::Fading { k } => {
                    assert!(k > prev, "淡入系数必须严格递增（帧 {f}）");
                    prev = k;
                }
                other => panic!("淡入段内出现非 Fading 调度：{other:?}"),
            }
        }
        assert!((prev - 1.0).abs() < 1e-6, "淡入末帧系数须为 1");
    }

    /// 显示域混合：端点逐位保端 + 中点取中 + alpha 恒 255。
    #[test]
    fn blend_rgba8_endpoints_and_midpoint() {        let a: Vec<u8> = (0..16u8).collect();
        let b: Vec<u8> = (200..216u8).collect();
        // k=0 / k=1 逐位保端（0×b 与 1×b 在 f32 精确）。
        assert_eq!(blend_rgba8(&a, &b, 0.0), a, "k=0 须逐位返回 a");
        assert_eq!(blend_rgba8(&a, &b, 1.0), b, "k=1 须逐位返回 b");
        // k=0.5 取两端均值（小整数半加和在 f32 精确）。
        let mid = blend_rgba8(&a, &b, 0.5);
        for i in 0..a.len() {
            let expect = ((a[i] as f32 + b[i] as f32) / 2.0).round() as u8;
            assert_eq!(mid[i], expect, "中点混合通道 {i}");
        }
        // alpha 通道（4 字节组末位）两帧均 255 ⇒ 混合后恒 255。
        let mut fa = vec![0u8; 8];
        let mut fb = vec![0u8; 8];
        for i in (3..8).step_by(4) {
            fa[i] = 255;
            fb[i] = 255;
        }
        let m = blend_rgba8(&fa, &fb, 0.37);
        for i in (3..8).step_by(4) {
            assert_eq!(m[i], 255, "alpha 通道须保持不透明");
        }
    }

    /// [任务 4] 30fps 玻璃门时间换算：源帧率翻倍 15→30，Create 入场时刻仍须
    /// 对齐 PRD 4.0s（源帧 120）与 0.5s 淡入（15 源帧），与全局帧表一致。
    #[test]
    fn glass_door_timing_at_30fps() {
        assert_eq!(SRC_FPS, 30, "S7 源帧率须为 30fps（规格 15→30）");
        // 入场起点 4.0s ⇒ 源帧 120（120/30 = 4.0）。
        assert!((GLASS_START_FRAME as f32 / SRC_FPS as f32 - 4.0).abs() < 1e-6);
        // 淡入 0.5s ⇒ 15 源帧（0.5 × 30 = 15）。
        assert_eq!(GLASS_FADE_FRAMES, 15);
        assert!((GLASS_FADE_FRAMES as f32 / SRC_FPS as f32 - 0.5).abs() < 1e-6);
        // 全平面起点 135/30 = 4.5s。
        assert!((GLASS_FULL_FRAME as f32 / SRC_FPS as f32 - 135.0 / 30.0).abs() < 1e-6);
    }

    /// [任务 4] 总帧数契约：序列尾帧（309）须落在相机 8.0s 定格段，且帧索引
    /// 严格单调（玻璃门 < 渐黑 < 尾帧），与全局 310 帧（30fps × 10.333s）吻合。
    #[test]
    fn s7_frame_count_contract() {
        assert_eq!(SRC_FPS, 30);
        let last = 309u32; // 310 帧序列尾帧（下标 0..309）。
        // 尾帧时刻 309/30 ≈ 10.3s ≥ 8.0s ⇒ 处于相机定格段，与末关键帧逐位一致。
        let c_last = camera_at(last as f32 / SRC_FPS as f32);
        let c_end = camera_at(8.0);
        // 尾帧须处于 8.0s 定格段（镜头不再运动）⇒ 视点逐分量位级一致。
        assert_eq!(c_last.eye.x.to_bits(), c_end.eye.x.to_bits(), "尾帧视点 x 须定格");
        assert_eq!(c_last.eye.y.to_bits(), c_end.eye.y.to_bits(), "尾帧视点 y 须定格");
        assert_eq!(c_last.eye.z.to_bits(), c_end.eye.z.to_bits(), "尾帧视点 z 须定格");
        // 帧索引单调性（玻璃门入场 → 渐黑 → 序列尾），任一越界都将破坏分派。
        assert!(GLASS_FULL_FRAME < FADE_START_FRAME);
        assert!(FADE_START_FRAME < FADE_END_FRAME);
        assert!(FADE_END_FRAME <= last + 1);
    }

    /// [任务 4] 相机插值连续性（30fps）：相邻源帧相机位移连续无跳变，且定格段
    /// （t ≥ 8.0s）恒等于末关键帧 (55°, 4.6, 3.0)，与源帧率翻倍无关。
    #[test]
    fn camera_interpolation_continuous_at_30fps() {
        let mut prev_eye = camera_at(0.0).eye;
        for i in 1..=309u32 {
            let t = i as f32 / SRC_FPS as f32;
            let eye = camera_at(t).eye;
            let d = (eye - prev_eye).length();
            assert!(d < 0.08, "源帧 {i} 相机跳变过大：{d}");
            prev_eye = eye;
        }
        // 定格段恒等于末关键帧（55°, 4.6, 3.0）。
        let az55 = 55.0f32.to_radians();
        for t in [8.0f32, 9.0, 10.333] {
            let c = camera_at(t);
            assert!((c.eye.x - 4.6 * az55.cos()).abs() < 1e-4, "定格段视点 x（t={t}）");
            assert!((c.eye.z - 3.0).abs() < 1e-4, "定格段视点 z（t={t}）");
        }
    }
}

// ---------------------------------------------------------------------------
// v3-B/C 测试（post_process 显示域 bloom；不触发光追渲染）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod v3_tests {
    use super::*;

    /// 微型 HDR 帧：中心一像素极亮（HDR 50），其余全黑。
    fn tiny_hdr(w: u32, h: u32) -> Vec<[f32; 4]> {
        let mut buf = vec![[0.0f32; 4]; (w * h) as usize];
        buf[((h / 2) * w + w / 2) as usize] = [50.0, 50.0, 50.0, 1.0];
        buf
    }

    /// post_process 确定性（同输入逐字节一致）与 alpha 恒 255。
    #[test]
    fn post_process_is_deterministic_and_opaque() {
        let (w, h) = (24u32, 24u32);
        let hdr = tiny_hdr(w, h);
        let a = post_process(&hdr, (w, h), 0);
        let b = post_process(&hdr, (w, h), 0);
        assert_eq!(a, b, "同 HDR 输入必须逐字节一致（纯函数）");
        assert_eq!(a.len(), (w * h * 4) as usize);
        for i in (3..a.len()).step_by(4) {
            assert_eq!(a[i], 255, "alpha 恒 255");
        }
    }

    /// 显示域 bloom：亮斑邻域获得辉光（边缘停止降噪不越过大跳变 ⇒
    /// 邻域增亮只能来自 bloom）。
    #[test]
    fn post_process_bloom_glows_around_bright_spot() {
        let (w, h) = (24u32, 24u32);
        let hdr = tiny_hdr(w, h);
        let out = post_process(&hdr, (w, h), 0);
        let at = |x: u32, y: u32| out[((y * w + x) * 4) as usize];
        let (cx, cy) = (w / 2, h / 2);
        // 中心亮斑本身（Reinhard 后近白）。
        assert!(at(cx, cy) > 200, "亮斑应近白：{}", at(cx, cy));
        // 邻域 4px：输入为 0、边缘停止降噪不扩散 ⇒ 有墨即 bloom 辉光。
        assert!(at(cx + 4, cy) > 0, "bloom 辉光必须到达 +4px 邻域：{}", at(cx + 4, cy));
        assert!(at(cx - 4, cy) > 0, "辉光须左右对称：{}", at(cx - 4, cy));
        // 支撑边界外（radius 4 之外 8px）：box blur 平台外应显著弱于邻域。
        let near = at(cx + 4, cy) as i32;
        let far = at(cx + 9, cy) as i32;
        assert!(far < near, "辉光随距离衰减（{far} < {near}）");
    }

    /// fade 与 bloom 的次序：fade=0（帧 ≥ FADE_END_FRAME）时全黑（落版
    /// 叠黑底），fade=1 时 bloom 辉光存在——压暗不产生新泛光。
    #[test]
    fn fade_to_black_suppresses_bloom_output() {
        let (w, h) = (16u32, 16u32);
        let hdr = tiny_hdr(w, h);
        let lit = post_process(&hdr, (w, h), 0);
        let dark = post_process(&hdr, (w, h), FADE_END_FRAME);
        assert!(dark.iter().all(|&b| b == 0 || b == 255), "渐黑帧须全黑（仅 alpha 255）");
        assert!(lit.iter().any(|&b| b != 0 && b != 255), "未渐黑帧应有中间调");
    }

    /// v3-C 常量：spp 48 / bounces 6（规格画质项；对拍基准同值的历史
    /// 已在模块头偏差记录备案）。
    #[test]
    fn v3_quality_constants() {
        assert_eq!(SPP, 48, "spp 8 → 48");
        assert_eq!(MAX_BOUNCES, 6, "max_bounces 3 → 6");
        // 预算门禁常量保持（降级逻辑不因画质提升而失守；S7 源帧率 15→30
        // 使渲染总量同比翻倍，故上限由 30min（1800）放宽到 45min（2700））。
        assert!((BUDGET_SEC - 2700.0).abs() < 1e-6);
    }
}
