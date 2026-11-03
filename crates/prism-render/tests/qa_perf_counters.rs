//! T09 测 1：内存/带宽剖析 + 帧池 + 零拷贝身份判据（V2.2-D-2 / PE-3）。
//!
//! 覆盖 PRD 验收 1–5：
//! 1. 计数器确定性（两次运行四项计数完全相等，硬断言）
//! 2. 帧池中性（复用 vs 非复用输出逐字节一致）
//! 3. 零拷贝身份判据（`as_ptr` 相等 + 未登记路径不得声称）
//! 4. RSS 报表可产出（非门禁）
//! 5. G1–G9（本次仅触 `prism-render`，零 crate、无 `#[link]`、Cargo.lock 不变）
//!
//! 设计 §6.2 / Line 516：`#[global_allocator]` **只在专用测试二进制**安装，生产零改动。
//!
//! 序列化的必要性：本二进制的计数分配器与计数静态是「进程级」共享的。
//! `cargo test` 默认并行跑各 `#[test]`，并行测试会各自分配、污染彼此的计数器窗口，
//! 导致「两次运行」的增量不等。故用 [`SERIAL`] 互斥锁把本文件所有测试串行化，
//! 保证 `counters_deterministic` 的两个测量窗口内只有它自己在分配。

use std::sync::Mutex;

use prism_core::{Aesthetic, Camera, Light, Material, Scene};
use prism_gpu::{CpuRefBackend, Framebuffer};
use prism_io::fbuf::Fbuf16F;
use prism_math::Vec3;
use prism_render::frame_pool::{self, FramePool, PoolStats};
use prism_render::measure::{self, Counters};
use prism_render::zero_copy;
use prism_render::{render_video, tile};

/// 仅本测试二进制安装计数分配器（生产零改动，见设计 Line 516）。
#[global_allocator]
static COUNTING_ALLOC: measure::CountingAllocator<std::alloc::System> =
    measure::CountingAllocator::new(std::alloc::System);

/// 测试间串行化互斥锁（进程级）。取锁后即使发生（预期的）panic 也以
/// `into_inner` 恢复，避免后续测试 `lock().unwrap()` 因中毒而连锁失败。
static SERIAL: Mutex<()> = Mutex::new(());

/// 取得串行化守卫；容忍中毒（供 `should_panic` 测试复用）。
fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const W: u32 = 32;
const H: u32 = 18;
const FPS: u32 = 4;
const DUR: f32 = 2.0;
const FRAMES: usize = (FPS as f32 * DUR).ceil() as usize; // 8

/// 固定场景：轨道红球 + 斜主光（与 lib.rs `orbit_scene` 同源，保证确定性）。
fn fixed_scene() -> Scene {
    Scene {
        radius: 1.0,
        orbit_center: Vec3::new(0.0, 0.0, 0.0),
        orbit_radius: 3.0,
        material: Material::new(Vec3::new(0.8, 0.2, 0.2), 0.4, 0.1),
        lights: vec![Light::new(
            Vec3::new(0.3, -1.0, 0.6),
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
        )],
        camera: Camera::new(
            Vec3::new(0.0, 0.0, -3.0),
            Vec3::new(0.0, 0.0, 0.0),
            60.0_f32.to_radians(),
        ),
        aesthetic: Aesthetic::new(false, 1.0, false),
            attach: Default::default(),
        }
}

/// 池化路径场景：每帧 acquire→render→copy→release，返回逐帧像素（供判据 ②）。
/// 内部自清空池 + 重置统计，保证两次调用的起点一致（确定性）。
fn run_pooled_scenario() -> Vec<Vec<u8>> {
    FramePool::clear();
    frame_pool::reset_pool_stats();
    let scene = fixed_scene();
    let mut device = CpuRefBackend;
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(FRAMES);
    for i in 0..FRAMES {
        let t = i as f32 / FPS as f32;
        let mut fb = frame_pool::render_frame_pooled(&mut device, &scene, t, DUR, W, H);
        out.push(fb.pixels.clone()); // 捕获逐字节结果
        FramePool::release(fb); // 归还池，供下一帧复用（产生命中）
    }
    out
}

/// 非池化基线：直接 render_video。
fn run_baseline_scenario() -> Vec<Vec<u8>> {
    let scene = fixed_scene();
    let mut device = CpuRefBackend;
    let frames = render_video(&mut device, &scene, W, H, FPS, DUR);
    frames.into_iter().map(|fb| fb.pixels).collect()
}

// ---------------------------------------------------------------------------
// 验收 1：计数器确定性
// ---------------------------------------------------------------------------

#[test]
fn counters_deterministic_across_two_runs() {
    let _guard = serial_guard(); // 串行化：测量窗口内只有本测试在分配

    // 预热：触发任何惰性初始化（thread_local 池、HashMap、Backend 内部），不计入测量。
    let _ = run_pooled_scenario();

    // 第一次测量
    measure::reset_counters();
    let base1 = measure::snapshot();
    let _o1 = run_pooled_scenario();
    let d1: Counters = measure::snapshot().delta_since(&base1);
    let p1: PoolStats = frame_pool::pool_stats();

    // run_pooled_scenario 内部已 clear + reset，第二次起点与第一次一致。
    // 第二次测量
    measure::reset_counters();
    let base2 = measure::snapshot();
    let _o2 = run_pooled_scenario();
    let d2: Counters = measure::snapshot().delta_since(&base2);
    let p2: PoolStats = frame_pool::pool_stats();

    // 判据 ①：四项计数完全相等（硬断言，不打印）。
    assert_eq!(d1.alloc_count, d2.alloc_count, "alloc_count 两次运行不等");
    assert_eq!(d1.alloc_bytes, d2.alloc_bytes, "alloc_bytes 两次运行不等");
    assert_eq!(d1.copy_count, d2.copy_count, "copy_count 两次运行不等");
    assert_eq!(p1.hits, p2.hits, "pool_hits 两次运行不等");
    assert_eq!(p1.misses, p2.misses, "pool_misses 两次运行不等");

    // 整体快照相等（额外保险，含释放计数）。
    assert_eq!(d1, d2, "两次运行计数快照不完全相等");

    // 正确性副断言：池确实被复用（命中>0），且每帧恰好一次显式拷贝。
    assert!(p1.hits > 0, "帧池应发生复用命中");
    assert_eq!(d1.copy_count, FRAMES as u64, "每帧应恰好一次显式拷贝");
    assert_eq!(p1.misses, 1, "同尺寸 8 帧应仅首次未命中");
    assert_eq!(p1.hits, (FRAMES - 1) as u64, "其余帧应命中复用");
}

// ---------------------------------------------------------------------------
// 验收 2：帧池中性（复用 vs 非复用输出逐字节一致）
// ---------------------------------------------------------------------------

#[test]
fn pool_reuse_is_byte_identical_to_fresh() {
    let _guard = serial_guard();
    let baseline = run_baseline_scenario();
    let pooled = run_pooled_scenario();
    assert_eq!(baseline.len(), pooled.len(), "帧数应一致");
    for (i, (a, b)) in baseline.iter().zip(pooled.iter()).enumerate() {
        assert_eq!(a, b, "第 {i} 帧：池化路径与非池化路径像素逐字节不一致");
    }
}

// ---------------------------------------------------------------------------
// 验收 3：零拷贝身份判据（as_ptr 相等 + 未登记不得声称）
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_tile_passthrough_identity() {
    let _guard = serial_guard();
    let win = Fbuf16F::new(4, 4);
    let out = tile::passthrough_identity(&win);
    // 输入/输出缓冲指针必须相等（零拷贝身份）。
    assert_eq!(
        win.pixels.as_ptr() as *const u8,
        out.pixels.as_ptr() as *const u8,
        "passthrough_identity 必须零拷贝（指针相等）"
    );
    // 经白名单校验（防「声明了但没做」）。
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(win.pixels.as_ptr() as *const u8, win.pixels.len() * 2) };
    let out_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(out.pixels.as_ptr() as *const u8, out.pixels.len() * 2)
    };
    zero_copy::verify_zero_copy("tile.passthrough_identity", in_bytes, out_bytes);
}

#[test]
fn zero_copy_frame_buffer_reborrow() {
    let _guard = serial_guard();
    let fb = Framebuffer::new(16, 9);
    let out = zero_copy::buffer_reborrow(&fb);
    assert_eq!(
        fb.pixels.as_ptr(),
        out.pixels.as_ptr(),
        "frame reborrow 必须零拷贝（指针相等）"
    );
    zero_copy::verify_zero_copy("frame.buffer_reborrow", &fb.pixels, &out.pixels);
}

#[test]
fn undeclared_path_must_not_be_whitelisted() {
    let _guard = serial_guard();
    // 未声明路径不得进白名单（防「声明了但没做」）。
    assert!(!zero_copy::is_zero_copy_path("made.up.path"));
    // box_blur3 是拷贝路径（分配新缓冲），不得登记为零拷贝。
    assert!(!zero_copy::is_zero_copy_path("tile.box_blur3"));
    // 已声明路径必须在白名单。
    assert!(zero_copy::is_zero_copy_path("tile.passthrough_identity"));
    assert!(zero_copy::is_zero_copy_path("frame.buffer_reborrow"));
}

#[test]
#[should_panic(expected = "未登记")]
fn undeclared_zero_copy_is_rejected() {
    let _guard = serial_guard();
    let fb = Framebuffer::new(4, 4);
    let out = zero_copy::buffer_reborrow(&fb);
    // 用未登记路径调用 verify → 必须 panic（任何路径想自称零拷贝必须先登记）。
    zero_copy::verify_zero_copy("made.up.path", &fb.pixels, &out.pixels);
}

#[test]
#[should_panic(expected = "声明零拷贝但")]
fn registered_path_with_copy_is_caught() {
    let _guard = serial_guard();
    // 已登记路径若实际发生拷贝（指针不等），身份判据必须抓出（防「声明了但没做」）。
    let input: Vec<u8> = vec![0; 16];
    let output: Vec<u8> = vec![0; 16]; // 内容相同、分配不同
    zero_copy::verify_zero_copy("tile.passthrough_identity", &input, &output);
}

// ---------------------------------------------------------------------------
// 验收 4：RSS 报表可产出（非门禁，绝对值只作报表）
// ---------------------------------------------------------------------------

#[test]
fn rss_report_is_produced() {
    let _guard = serial_guard();
    let scene = fixed_scene();
    let mut device = CpuRefBackend;
    let _frames = render_video(&mut device, &scene, W, H, FPS, DUR); // 跑固定场景
    let report = measure::rss_report("fixed_scene_32x18", W, H);
    // 报表结构断言（不判绝对值）：场景名/尺寸正确，峰值字段可读。
    assert_eq!(report.scenario, "fixed_scene_32x18");
    assert_eq!(report.width, W);
    assert_eq!(report.height, H);
    let _ = report.peak_rss_kib; // 字段存在且可读取（Linux 为 Some，其他平台 None）
}
