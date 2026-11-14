//! **P2 判据：GPU 侧 2D 路径光栅与 `prism-raster` f64 几何权威的数学等价性**。
//!
//! 这是 `crates/prism-gpu/tests/vulkan_path_raster_gpu.rs` 文件头「P2（数学等价，
//! f64 精确面积标尺）」承诺的那条断言的落地文件。P1（GPU 输出与同一份 f32 片段
//! 缓冲上的 `rasterize_cpu_prepared` 逐字节一致）已在 prism-gpu 侧真机闭环里
//! 断言；本文件补上另一半——**GPU 真的算对了面积**，而不是「跟一个自己写的 f32
//! 参考自洽」。
//!
//! # 为什么需要 P2，以及为什么它不能靠放宽阈值过关
//!
//! `prism-raster::scanline::rasterize` 是引擎里既有的 CPU 扫描线权威：几何插值、
//! 交点、面积原函数全在 **f64** 里算，只在写回覆盖率缓冲时舍入一次到 f32，其
//! 精度由 `crates/prism-raster/src/scanline_diff.rs` 用 341 组 f64 精确有理积分
//! 标尺钉在 `2e-6` 内。SPIR-V 1.0 与 WGSL 都没有 f64——GPU kernel 只能 f32。
//!
//! 若把全局 x 坐标直接以 f32 传给 kernel，1080p/4K 坐标的 ulp 是 `1e-4` 量级，
//! 覆盖率会偏离精确面积到 `1e-4`，**「GPU 与 f64 标尺逐字节一致」数学上不可达**。
//! 本实现不靠调阈值，而是改数据编码：片段的 x 拆成「精确整数基准 + 小量小数」
//! （`X = col_base + f`，`base = col_base + 2²⁰` 是 f32 精确整数），kernel 里
//! `u = (base − 2²⁰ − px) + f`；括号内是整数差（f32 精确），只有 `u ∈ (−1,2)`
//! 的敏感区影响 `pos_area`，此时 `ulp(u) ≤ 2.4e-7`；每像素最多 4 条穿越边贡献
//! ⇒ `|Δcov| < 1e-6 < 2e-6`。**这就是 `TOL_EXACT = 2e-6` 的推导来源**，不是
//! 实测凑出来的数字。
//!
//! # 判据（动手前写死，绝不事后改尺子）
//!
//! 对下面 `cases()` 的每一个用例，把 GPU 优先入口 `prism_gpu::path_raster::
//! rasterize`（有设备时走 GPU，无设备时透明回退 f32 参考）的输出与
//! `prism_raster::scanline::rasterize`（f64 几何权威）逐像素比：
//!
//! 1. `max_abs_delta ≤ 2e-6`（每像素覆盖率最大绝对差，f64 口径）；
//! 2. u8 量化视图 `differing_pixels = 0` ∧ `max_channel_delta = 0`（视觉等价）；
//! 3. 面积总量相对误差 `≤ 1e-6`（Σcoverage 守恒）；
//! 4. 覆盖率值域 `∈ [−2e-6, 1+2e-6]`、有限。
//!
//! 另外三条辅助判据：
//! * **f32 参考单独对拍**：`rasterize_cpu` 与权威同样满足 1–3，把「数学等价」
//!   与「GPU 是否真执行」解耦；
//! * **GPU 帧显式对拍**：有设备时 `rasterize_path_gpu` 返回 `Some`，其输出与
//!   权威同样满足 1–3（证明真机 dispatch 出来的数也算对了面积）；
//! * **负控（反假绿）**：故意把一份正确的覆盖率缓冲的一个像素扰动 `0.1`，
//!   `compare()` 必须检出 `differing_pixels > 0` 且 `max_abs_delta > TOL_EXACT`
//!   ——证明本文件的判据有判别力，不是恒真断言。
//!
//! # 数学等价性（不是近似、不是超采样）
//!
//! 覆盖率是区域 `{winding ≠ 0}`（或 even-odd 奇区）与像素矩形 `P=[px,px+1)×
//! [py,py+1)` 的**精确交面积**。两份实现是同一个面积积分的两种写法：
//!
//! * `prism-raster`：按活跃跨越把每行切成若干区间 `[xl(y), xr(y))`，对**两个
//!   边界列**用 `S(X,t) = ∫(X(y)−t)⁺dy` 解析求交宽，对内部列用一次 O(1) 区间加
//!   `+= h`（Green/射线法里的「区域积分」）；
//! * `prism-gpu`：把指示函数写成带符号跳变权重的望远镜和
//!   `1_R(x,y) = Σ_k delta_k · 1[x < X_k(y)]`，对像素列积分
//!   `∫1[x<X]dx = clamp(X−px,0,1) = u⁺ − (u−1)⁺`，`∫(c+kt)⁺dt` 的闭式就是
//!   `pos_area`（三角/梯形，纯 `+ − × ÷` 与比较）。
//!
//! 两者都是对同一测度的精确积分，逐点恒等；差别**只**来自 f32 舍入（GPU/kernel
//! 全程 f32；权威在最后写回时舍入一次）。故 `≤ 2e-6` 是「同一精确量的两次 f32
//! 求值之差」的上界，而非把近似放宽到能过。
//!
//! # 纪律
//!
//! 为变绿而调 `TOL_EXACT` / `AREA_REL_TOL` / 放宽 `differing_pixels` = **改判据**，
//! 不是修实现。失败退路是修 kernel 或修预处理编码，本文件阈值一个字都不动。

use prism_gpu::path_raster::{
    rasterize as rasterize_engine, rasterize_cpu as rasterize_f32_ref, CoverageMask as GpuCoverage,
    PathFillRule, RasterPoint,
};
use prism_gpu::rasterize_path_gpu;
use prism_raster::geom::Point;
use prism_raster::scanline::{rasterize as rasterize_authority, FillRule};

/// P2 的 f64 标尺阈值（推导见文件头；与 prism-gpu 侧 `TOL_EXACT` 同源）。
const TOL_EXACT: f64 = 2e-6;
/// 面积总量相对误差阈值。
const AREA_REL_TOL: f64 = 1e-6;

/// `PathFillRule` → `prism_raster::FillRule`。两个枚举一一对应。
fn to_rule(r: PathFillRule) -> FillRule {
    match r {
        PathFillRule::Winding => FillRule::Winding,
        PathFillRule::EvenOdd => FillRule::EvenOdd,
    }
}

/// POD 顶点桥接：`prism_gpu::path_raster::RasterPoint` → `prism_raster::geom::Point`。
///
/// 两个结构体都是 `{ x: f32, y: f32 }`；这里显式逐字段拷贝，是为了让「GPU 侧
/// 刻意不复用 raster 的类型」这一依赖方向论证在测试里也一眼可见，且没有引入
/// 任何跨 crate 的反向依赖。
fn to_points(contours: &[Vec<RasterPoint>]) -> Vec<Vec<Point>> {
    contours
        .iter()
        .map(|c| c.iter().map(|p| Point::new(p.x, p.y)).collect())
        .collect()
}

/// 一份对拍统计（f64 口径 + u8 量化视图 + 面积守恒）。
#[derive(Debug)]
struct Parity {
    max_abs_delta: f64,
    differing_pixels: usize,
    max_channel_delta: u8,
    differing_f32: usize,
    area_engine: f64,
    area_authority: f64,
    area_rel: f64,
    first: Option<(usize, f32, f32)>,
}

/// 逐像素比较「GPU 侧输出」与「f64 几何权威」。
///
/// 两侧用**同一个** `to_u8` 量化函数（覆盖率 ×255 四舍五入、钳 `[0,1]`），
/// 因此 `differing_pixels` 只反映 f32 覆盖率数值本身的差异，不含量化口径差。
fn compare(engine: &[f32], authority: &[f32]) -> Parity {
    assert_eq!(engine.len(), authority.len(), "两份覆盖率缓冲尺寸必须一致");
    let eu = GpuCoverage { width: 0, height: 0, data: engine.to_vec() }.to_u8();
    let au = GpuCoverage { width: 0, height: 0, data: authority.to_vec() }.to_u8();
    let mut p = Parity {
        max_abs_delta: 0.0,
        differing_pixels: 0,
        max_channel_delta: 0,
        differing_f32: 0,
        area_engine: 0.0,
        area_authority: 0.0,
        area_rel: 0.0,
        first: None,
    };
    for i in 0..engine.len() {
        let g = engine[i] as f64;
        let a = authority[i] as f64;
        let d = (g - a).abs();
        if d > p.max_abs_delta {
            p.max_abs_delta = d;
        }
        p.area_engine += g;
        p.area_authority += a;
        if engine[i].to_bits() != authority[i].to_bits() {
            p.differing_f32 += 1;
            if p.first.is_none() {
                p.first = Some((i, engine[i], authority[i]));
            }
        }
        let cd = eu[i].max(au[i]) - eu[i].min(au[i]);
        if cd > 0 {
            p.differing_pixels += 1;
        }
        if cd > p.max_channel_delta {
            p.max_channel_delta = cd;
        }
    }
    let den = p.area_authority.abs().max(1.0);
    p.area_rel = (p.area_engine - p.area_authority).abs() / den;
    p
}

/// 断言一组 P2 数值判据全绿，并打印一行 `[parity]` 供交付摘录。
fn assert_parity(tag: &str, name: &str, w: u32, h: u32, p: &Parity) {
    println!(
        "[parity] engine={tag} case={name} size={w}x{h} max_abs_delta={:.3e} \
         differing_pixels={} max_channel_delta={} differing_f32={} area_rel={:.3e} \
         (area {:.3} vs {:.3})",
        p.max_abs_delta, p.differing_pixels, p.max_channel_delta, p.differing_f32,
        p.area_rel, p.area_engine, p.area_authority
    );
    assert!(
        p.max_abs_delta <= TOL_EXACT,
        "[{tag}/{name}] P2 f64 标尺：max_abs_delta={:.3e} 超过 {TOL_EXACT:e}，首个 f32 差异 {:?}",
        p.max_abs_delta, p.first
    );
    assert_eq!(
        p.differing_pixels, 0,
        "[{tag}/{name}] P2 u8 量化：differing_pixels={}（首个 f32 差异 {:?}）",
        p.differing_pixels, p.first
    );
    assert_eq!(
        p.max_channel_delta, 0,
        "[{tag}/{name}] P2 u8 量化：max_channel_delta={}",
        p.max_channel_delta
    );
    assert!(
        p.area_rel <= AREA_REL_TOL,
        "[{tag}/{name}] P2 面积守恒：area_rel={:.3e} 超过 {AREA_REL_TOL:e}",
        p.area_rel
    );
}

/// 值域门禁：覆盖率有限、且 ∈ [−2e-6, 1+2e-6]。
fn assert_range(name: &str, data: &[f32]) {
    for (i, &v) in data.iter().enumerate() {
        assert!(
            v.is_finite() && v >= -TOL_EXACT as f32 && v <= 1.0 + TOL_EXACT as f32,
            "[{name}] 覆盖率越界 @{i}: {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// 形状构造（仅测试数据；trig 只出现在形状生成，不进光栅算子/kernel）
// ---------------------------------------------------------------------------

fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<RasterPoint> {
    vec![
        RasterPoint::new(x0, y0),
        RasterPoint::new(x1, y0),
        RasterPoint::new(x1, y1),
        RasterPoint::new(x0, y1),
    ]
}

fn star(n: usize, cx: f32, cy: f32, r: f32) -> Vec<RasterPoint> {
    (0..n)
        .map(|i| {
            let t = (i as f32) * (core::f32::consts::TAU / n as f32);
            let ri = if i % 2 == 0 { r } else { r * 0.42 };
            RasterPoint::new(cx + ri * t.cos(), cy + ri * t.sin())
        })
        .collect()
}

/// 判据覆盖的用例集：轴对齐半像素边界 / 斜边 / 非方阵 / 孔洞 / winding vs
/// even-odd / 多穿越星形 / 大星压力 / 45° 半平面 / 空路径。
fn cases() -> Vec<(&'static str, Vec<Vec<RasterPoint>>, PathFillRule, u32, u32)> {
    let mut outer = square(2.0, 2.0, 58.0, 44.0);
    outer.reverse();
    let inner = square(12.0, 10.0, 40.0, 34.0);
    vec![
        ("square_64x48", vec![square(1.5, 1.5, 40.5, 30.5)], PathFillRule::Winding, 64, 48),
        ("triangle_37x23", vec![vec![
            RasterPoint::new(2.3, 1.7), RasterPoint::new(31.7, 6.1), RasterPoint::new(8.9, 20.3),
        ]], PathFillRule::Winding, 37, 23),
        ("tailgroups_100x70", vec![square(3.25, 2.75, 91.5, 63.25)], PathFillRule::Winding, 100, 70),
        ("ring_hole_64x48", vec![outer, inner], PathFillRule::Winding, 64, 48),
        ("overlap_winding_48x32", vec![square(1.0, 1.0, 30.0, 24.0), square(12.0, 8.0, 44.0, 30.0)], PathFillRule::Winding, 48, 32),
        ("overlap_evenodd_48x32", vec![square(1.0, 1.0, 30.0, 24.0), square(12.0, 8.0, 44.0, 30.0)], PathFillRule::EvenOdd, 48, 32),
        ("star64_64x48", vec![star(64, 32.0, 24.0, 21.0)], PathFillRule::Winding, 64, 48),
        ("star96_96x72", vec![star(96, 48.0, 36.0, 33.0)], PathFillRule::Winding, 96, 72),
        // 45° 斜边 x+y=40：斜穿像素的覆盖率恰为 0.5，专门压测「斜边精确半覆盖」
        // 与 f64 权威的一致性（不是只有轴对齐半像素边界）。
        ("diagonal_halfplane_40x40", vec![vec![
            RasterPoint::new(0.0, 0.0), RasterPoint::new(40.0, 0.0), RasterPoint::new(0.0, 40.0),
        ]], PathFillRule::Winding, 40, 40),
        ("empty_16x12", Vec::new(), PathFillRule::Winding, 16, 12),
    ]
}

/// 一份用例的 f64 权威输出（供多个测试复用）。
fn authority_of(contours: &[Vec<RasterPoint>], rule: PathFillRule, w: u32, h: u32) -> Vec<f32> {
    let pts = to_points(contours);
    rasterize_authority(&pts, to_rule(rule), w, h).data
}

// ---------------------------------------------------------------------------
// P2 主判据
// ---------------------------------------------------------------------------

/// **P2 主判据**：GPU 优先引擎入口 `rasterize` 的输出与 f64 几何权威逐像素等价。
///
/// 有 Vulkan 设备时 `rasterize` 内部走真机 GPU dispatch（见 prism-gpu 侧
/// `engine_entry_is_byte_identical_and_uses_gpu_when_available`）；无设备时透明
/// 回退 f32 参考。两种情况下结果都必须落在同一把 f64 标尺的 `2e-6` 内——
/// 这正是「能力提升对上层透明、结果不变」的量化表述。
#[test]
fn engine_entry_matches_f64_authority() {
    for (name, contours, rule, w, h) in cases() {
        let authority = authority_of(&contours, rule, w, h);
        let engine = rasterize_engine(&contours, rule, w, h);
        assert_eq!(engine.width, w, "[{name}] 引擎输出宽");
        assert_eq!(engine.height, h, "[{name}] 引擎输出高");
        assert_eq!(engine.data.len(), authority.len(), "[{name}] 缓冲长度");
        assert_range(name, &engine.data);
        let p = compare(&engine.data, &authority);
        assert_parity("rasterize", name, w, h, &p);
    }
}

/// **数学等价解耦**：f32 参考 `rasterize_cpu`（不经 GPU）与权威同样满足 1–3。
/// 这条把「公式写对了」与「GPU 有没有真跑」分开：即便某台机器没有设备，
/// f32 算子本身也必须与 f64 标尺等价。
#[test]
fn f32_reference_matches_f64_authority() {
    for (name, contours, rule, w, h) in cases() {
        let authority = authority_of(&contours, rule, w, h);
        let reference = rasterize_f32_ref(&contours, rule, w, h);
        assert_range(name, &reference.data);
        let p = compare(&reference.data, &authority);
        assert_parity("rasterize_cpu", name, w, h, &p);
    }
}

/// **GPU 帧显式对拍**：有设备时 `rasterize_path_gpu` 必须返回 `Some`，且真机
/// dispatch → fence → readback 出来的覆盖率与 f64 权威同样满足 1–3。
/// 这条把 P2 直接钉在「设备上算出来的数」上，而不是仅凭引擎入口的透明回退。
#[test]
fn gpu_frame_matches_f64_authority_when_device_present() {
    let device = prism_gpu::vulkan::VulkanBackend::probe().device_created;
    if !device {
        println!("[gpu-parity] 本机无 Vulkan 设备：按既有范式自门控 skip（不 fail、不假装跑过）");
        return;
    }
    let mut checked = 0usize;
    for (name, contours, rule, w, h) in cases() {
        if contours.is_empty() {
            continue; // 空路径不产生 GPU 帧（资源守卫），由引擎入口用例覆盖
        }
        let authority = authority_of(&contours, rule, w, h);
        let gpu = rasterize_path_gpu(&contours, rule, w, h)
            .unwrap_or_else(|| panic!("[{name}] 有设备时 GPU 入口必须返回 Some（不得静默回退）"));
        assert_eq!(gpu.data.len(), authority.len(), "[{name}] GPU 帧长度");
        assert_range(name, &gpu.data);
        let p = compare(&gpu.data, &authority);
        assert_parity("rasterize_path_gpu", name, w, h, &p);
        checked += 1;
    }
    println!("[gpu-parity] device present：{checked} 个用例真机帧全部与 f64 权威等价");
    assert!(checked >= 8, "有设备时至少对拍 8 个非空用例，实际 {checked}");
}

/// **确定性**：引擎入口同输入两跑逐字节一致，且两遍都与 f64 权威等价。
#[test]
fn engine_entry_is_deterministic_both_runs_match_authority() {
    let contours = vec![star(48, 40.0, 30.0, 27.0)];
    let (w, h) = (80, 60);
    let rule = PathFillRule::Winding;
    let a = rasterize_engine(&contours, rule, w, h);
    let b = rasterize_engine(&contours, rule, w, h);
    assert_eq!(a.data.len(), b.data.len());
    for (i, (x, y)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "两遍引擎输出必须逐位一致 @{i}");
    }
    let authority = authority_of(&contours, rule, w, h);
    let p = compare(&a.data, &authority);
    assert_parity("rasterize(determinism)", "star48_80x60", w, h, &p);
    println!("[determinism] 80x60 两遍逐字节一致，且均与 f64 权威等价");
}

/// **负控（反假绿）**：证明 `compare()` 与本文件判据确有判别力。
///
/// 取一份正确对拍通过的覆盖率，把其中一个非零像素扰动 `+0.1`（远超 2e-6），
/// 断言 `differing_pixels > 0` ∧ `max_abs_delta > TOL_EXACT`。若这条也「绿」，
/// 说明判据恒真、前面的 P2 全是假绿——那正是本测试要抓的形态。
#[test]
fn negative_control_perturbed_coverage_fails_parity() {
    let contours = vec![square(1.5, 1.5, 40.5, 30.5)];
    let (w, h) = (64, 48);
    let authority = authority_of(&contours, PathFillRule::Winding, w, h);
    let good = rasterize_f32_ref(&contours, PathFillRule::Winding, w, h);

    // 先确认「未扰动」确实通过 P2（否则负控无意义）。
    let baseline = compare(&good.data, &authority);
    assert!(
        baseline.max_abs_delta <= TOL_EXACT && baseline.differing_pixels == 0,
        "负控前提：未扰动的对拍必须先通过 P2，实得 {baseline:?}"
    );

    // 找一个被覆盖的像素，强制改成 0（原本 ≥ 0.5，|Δ| ≥ 0.5 ≫ 2e-6）。
    // 注意：不能对已是 1.0 的内部像素做 `(+0.1).min(1.0)`——那是空操作扰动。
    let idx = good.data.iter().position(|&v| v > 0.5).expect("用例必须有非零覆盖像素");
    let mut bad = good.data.clone();
    bad[idx] = 0.0;
    assert_ne!(bad[idx], good.data[idx], "负控扰动必须真的改变该像素值");
    let p = compare(&bad, &authority);
    println!(
        "[negative-control] 单像素置 0：differing_pixels={} max_abs_delta={:.3e} ⇒ 判据 {}",
        p.differing_pixels, p.max_abs_delta,
        if p.differing_pixels > 0 && p.max_abs_delta > TOL_EXACT {
            "FAIL（符合预期）"
        } else {
            "PASS（假绿！）"
        }
    );
    assert!(
        p.differing_pixels > 0 && p.max_abs_delta > TOL_EXACT,
        "扰动覆盖率后 P2 判据必须 FAIL；若也绿，说明判据恒真"
    );
}
