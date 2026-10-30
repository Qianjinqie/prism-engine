//! QA 裁决探针 · P1 GPU 修复验证（prism-gpu）。
//!
//! 目标：独立验证「尺寸乘法溢出 + 巨额分配 abort」修复真有效，
//! 且 debug / release 双端行为一致（宪章 C18）。
//!
//! 覆盖：
//! - G1 `Framebuffer::new` 在极端尺寸下不 panic / 不 abort；
//! - G2 拒绝语义：pixels 空 + 尺寸原样回显 + `is_valid() == false`；
//! - G3 非超限输入下 `pixels.len() == w*h*4` 严格成立（a08 不变量）；
//! - G4 `MAX_FRAMEBUFFER_BYTES` 边界精确性（`==` 接受 / `+1` 拒绝）；
//! - G5 `set_pixel` 越界坐标（含 `u32::MAX`）不 panic 且被丢弃；
//! - G6 `CpuRefBackend::render_scene` 在 `u32::MAX × u32::MAX` 下秒回（不挂死）。

use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, Framebuffer, GpuDevice, LightReq, SceneRenderRequest,
    SphereReq, MAX_FRAMEBUFFER_BYTES, MAX_FRAMEBUFFER_DIM,
};
use prism_math::Vec3;
use std::time::Instant;

/// 构造一个最小可用的渲染请求（只关心 width/height，故其余字段填合理默认）。
fn scene_req(width: u32, height: u32) -> SceneRenderRequest {
    SceneRenderRequest {
        width,
        height,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.0, 8.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 45.0,
            aspect: width as f32 / height.max(1) as f32,
        },
        lights: vec![LightReq {
            direction: Vec3::new(1.0, -1.0, -0.5),
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 3.0,
        }],
        spheres: vec![SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            albedo: Vec3::new(0.75, 0.22, 0.17),
            roughness: 0.3,
            metalness: 0.05,
        }],
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
    curves: Vec::new(),
    }
}

/// 所有必须「不 panic」的尺寸组合（含溢出源与边界值）。
const EXTREME: &[(u32, u32)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (65535, 65535),
    (65536, 65536),
    (65536, 65536), // 16 GiB > 1 GiB 上限
    (u32::MAX, u32::MAX),
    (u32::MAX, u32::MAX - 1),
    (u32::MAX - 1, u32::MAX),
    (1 << 31, 1 << 31),
    (65536, 1),
    (1, 65536),
    (65537, 1),
    (1, 65537),
];

#[test]
fn g1_new_extreme_dims_never_panics() {
    for &(w, h) in EXTREME {
        let r = std::panic::catch_unwind(|| Framebuffer::new(w, h));
        assert!(r.is_ok(), "Framebuffer::new({w}, {h}) 导致 panic");
    }
}

#[test]
fn g2_refusal_semantics_pixels_empty_dims_echoed() {
    for &(w, h) in EXTREME {
        let fb = Framebuffer::new(w, h);
        // 尺寸必须原样回显（不得回写成 0 或钳制值）——调用方要能看出「被拒绝了什么」。
        assert_eq!(fb.width, w, "width 未原样回显");
        assert_eq!(fb.height, h, "height 未原样回显");
        if !fb.is_valid() {
            assert!(fb.pixels.is_empty(), "({w},{h}) 被拒但 pixels 非空");
        }
    }
    // 明确点名的三例
    for &(w, h) in &[(u32::MAX, u32::MAX), (1 << 31, 1 << 31), (65536, 65536)] {
        let fb = Framebuffer::new(w, h);
        assert!(fb.pixels.is_empty(), "({w},{h}) 应拒绝：pixels 须为空");
        assert!(!fb.is_valid(), "({w},{h}) 应拒绝：is_valid() 须为 false");
    }
}

#[test]
fn g3_accepted_dims_pixel_len_invariant_strict() {
    // a08 不变量：成功路径下 pixels.len() == w*h*4，任何修复都不得破坏。
    for &(w, h) in &[(1u32, 1u32), (64, 64), (24, 16), (255, 3), (1, 4096), (4096, 1)] {
        let fb = Framebuffer::new(w, h);
        assert!(fb.is_valid(), "({w},{h}) 应为有效缓冲");
        assert_eq!(
            fb.pixels.len(),
            w as usize * h as usize * 4,
            "({w},{h}) 像素长度不满足 w*h*4"
        );
        assert_eq!(fb.pixels.iter().filter(|&&b| b == 0).count(), fb.pixels.len(), "新缓冲须全零");
    }
}

#[test]
fn g4_max_bytes_boundary_is_exact() {
    // 上限是「字节」而非「边长」：找出一组恰好等于上限的尺寸。
    // MAX = 1 GiB = 1073741824；w*h*4 == MAX ⇒ w*h == 268435456 == 2^28。
    let side: u32 = 1 << 14; // 16384；16384^2 * 4 = 2^28 * 4 = 2^30 = 1 GiB 恰好
    assert_eq!(side as usize * side as usize * 4, MAX_FRAMEBUFFER_BYTES, "边界构造前提失效");

    let at = Framebuffer::new(side, side);
    assert!(at.is_valid(), "恰好 1 GiB 应被接受（边界含端点）");
    assert_eq!(at.pixels.len(), MAX_FRAMEBUFFER_BYTES, "1 GiB 边界像素长度不符");

    // +1 字节必然拒绝：把高度方向多要 1 像素不行，改用 +1 行（4 字节）已超；
    // 故用「字节数 > MAX」的最小可达组合验证拒绝侧。
    let over_w = 1u32 << 14;
    let over = Framebuffer::new(over_w, side + 1); // (2^14)*(2^14+1)*4 > 1 GiB
    assert!(!over.is_valid(), "超过 1 GiB 必须拒绝");
    assert!(over.pixels.is_empty(), "超过 1 GiB 必须 pixels 为空");
    assert_eq!(over.width, over_w, "拒绝时尺寸仍须原样回显");
    assert_eq!(over.height, side + 1, "拒绝时尺寸仍须原样回显");
}

#[test]
fn g4b_dim_ceiling_boundary_is_exact() {
    // MAX_FRAMEBUFFER_DIM 边界：单轴 == DIM 且另一轴为 0 ⇒ 0 字节，接受。
    let at = Framebuffer::new(MAX_FRAMEBUFFER_DIM, 0);
    assert!(at.pixels.is_empty(), "0 高度 ⇒ 0 字节");
    assert_eq!(at.width, MAX_FRAMEBUFFER_DIM, "0 高度时 width 应原样回显");

    let over = Framebuffer::new(MAX_FRAMEBUFFER_DIM + 1, 0);
    assert!(over.pixels.is_empty(), "超单轴上限须拒绝");
    assert_eq!(over.width, MAX_FRAMEBUFFER_DIM + 1, "超限不回写尺寸");
    assert!(!over.is_valid(), "超单轴上限 ⇒ is_valid()==false");

    // **判别性用例**：单轴超限但总字节数仍在 1 GiB 之内（65537×1×4 = 256 KiB）。
    // 只有单轴上限守卫能拒绝它——若该守卫被移除，本例会「合法」分配出
    // 262148 字节的缓冲从而转红。这使探针对该守卫敏感（变异自证）。
    let sneaky = Framebuffer::new(MAX_FRAMEBUFFER_DIM + 1, 1);
    assert!(
        !sneaky.is_valid() && sneaky.pixels.is_empty(),
        "单轴超限（{0}×1，仅 256 KiB）必须被单轴上限守卫拒绝",
        MAX_FRAMEBUFFER_DIM + 1
    );
    assert_eq!(sneaky.width, MAX_FRAMEBUFFER_DIM + 1, "拒绝时尺寸原样回显");
    assert_eq!(sneaky.height, 1, "拒绝时尺寸原样回显");
}

#[test]
fn g5_set_pixel_out_of_range_never_panics() {
    let mut fb = Framebuffer::new(64, 64);
    assert!(fb.is_valid());
    let before = fb.pixels.clone();
    // 含溢出源坐标：u32::MAX / 2^31 / 越界组合。
    for &(x, y) in &[
        (0u32, 0u32),
        (63, 63),
        (63, 64),   // y 越界（行内，须丢弃）
        (0, 64),    // y 越界（须丢弃）
        (u32::MAX, u32::MAX), // 索引乘法溢出源
        (u32::MAX, 0),
        (0, u32::MAX),
        (1 << 31, 1 << 31), // y*width 溢出源
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fb.set_pixel(x, y, Vec3::new(1.0, 0.0, 0.0));
        }));
        assert!(r.is_ok(), "set_pixel({x},{y}) 导致 panic");
    }
    // 越界写入必须被丢弃：只有 (0,0) 与 (63,63) 两次**合法**写入生效。
    //
    // 注意：`x >= width` 的坐标当前会被回绕到下一行（见 g8 已知缺陷），
    // 故此处只用**行内**越界（y 越界）与**溢出源**坐标（u32::MAX / 2^31）来
    // 验证「丢弃」语义，这两类坐标经 `checked_mul/checked_add` 后必然越界。
    assert_eq!(
        fb.get_pixel(63, 63),
        Some([255, 0, 0, 255]),
        "合法角点 (63,63) 写入应生效"
    );
    assert_eq!(fb.get_pixel(0, 0), Some([255, 0, 0, 255]), "(0,0) 写入应生效");
    let changed = fb.pixels.iter().enumerate().filter(|(i, &v)| before[*i] != v).count();
    // 两次合法写入 × (R + A) 各改 1 字节 = 4 字节；被丢弃的坐标一个字节都不许改。
    assert_eq!(changed, 4, "越界坐标必须被丢弃，实际有 {changed} 字节被改动");

    // 被拒缓冲上的写入也必须是惰性的（不 panic）。
    let mut refused = Framebuffer::new(u32::MAX, u32::MAX);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        refused.set_pixel(0, 0, Vec3::new(1.0, 1.0, 1.0));
    }))
    .is_ok());
    assert!(refused.pixels.is_empty(), "被拒缓冲不得接受任何写入");
}

#[test]
fn g6_cpu_ref_render_scene_extreme_returns_promptly() {
    // 修前：`u32::MAX × u32::MAX` 会在空 pixels 上空转挂死（QA 实测 >300s）。
    // 本测试自带硬超时断言：超过 10 s 视为挂死。
    let scene = scene_req(u32::MAX, u32::MAX);
    let t0 = Instant::now();
    let mut dev = CpuRefBackend;
    let fb = dev.render_scene(&scene);
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs() < 10,
        "CpuRefBackend::render_scene(u32::MAX, u32::MAX) 耗时 {elapsed:?}，疑似挂死"
    );
    assert!(!fb.is_valid(), "u32::MAX×u32::MAX 应被拒 ⇒ is_valid()==false");
    assert!(fb.pixels.is_empty(), "u32::MAX×u32::MAX 应被拒 ⇒ pixels 为空");
    assert_eq!(fb.width, u32::MAX, "拒绝时 width 原样回显");
    assert_eq!(fb.height, u32::MAX, "拒绝时 height 原样回显");
    println!("G6 render_scene(u32::MAX,u32::MAX) 耗时 = {elapsed:?}");

    // 正常尺寸仍须秒回且有效（确认提前返回没有误伤正常路径）。
    let t1 = Instant::now();
    let ok = dev.render_scene(&scene_req(64, 64));
    let e1 = t1.elapsed();
    assert!(ok.is_valid(), "64×64 应有效");
    assert_eq!(ok.pixels.len(), 64 * 64 * 4);
    println!("G6 render_scene(64,64) 耗时 = {e1:?}");
}

#[test]
fn g7_trait_defaults_present_and_non_panicking() {
    // `is_available()` / `backend_name()` 必须带默认实现（零破坏），
    // 且对未实现后端返回可用性 false 而不是崩。
    struct Minimal;
    impl GpuDevice for Minimal {
        fn render_scene(&mut self, _req: &prism_gpu::SceneRenderRequest) -> Framebuffer {
            Framebuffer::empty()
        }
    }
    let m = Minimal;
    // 默认实现：可用（后端未声明不可用即为可用），名字为稳定占位符。
    assert!(m.is_available(), "默认 is_available() 应为 true（后端须显式声明不可用）");
    assert_eq!(m.backend_name(), "unknown", "默认 backend_name() 应返回稳定占位符");
    // CpuRefBackend：可用 + 有实名。
    let c = CpuRefBackend;
    assert!(c.is_available(), "CpuRefBackend 恒可用");
    assert_eq!(c.backend_name(), "cpu-ref");
    // VulkanBackend：**不得 panic**；可用性 = compute 自检 ∧ 渲染自检（V2.1 Q4）。
    // 注：W3 起 `is_available()` 的口径从「恒 false（渲染未接）」改为
    // 「mul2 探针真跑通且读回与 CPU 参考逐位一致才 true」；V2.1 T-A1 起再叠加
    // 「场景渲染自检真跑通」。本机有 lavapipe 时该值为 true，无设备时为 false；
    // 两种环境都由下面的等式/蕴含断言刻画。
    let mut v = prism_gpu::vulkan::VulkanBackend::new();
    assert_eq!(
        v.is_available(),
        v.compute_available() && v.render_available(),
        "is_available() 必须严格等同 compute + render 双自检结果"
    );
    assert_eq!(v.backend_name(), "vulkan-compute");
    if v.device_name().is_none() {
        assert!(!v.is_available(), "无设备 ⇒ 必须诚实报不可用");
        assert!(!v.render_available(), "无设备 ⇒ 渲染自检必须为假");
    } else {
        assert!(
            v.is_available(),
            "探测到了设备 ⇒ 两条自检闭环必须跑通（跑不通即真 bug，不许静默降级）"
        );
    }
    let t0 = Instant::now();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.render_scene(&scene_req(64, 64))));
    let e = t0.elapsed();
    assert!(r.is_ok(), "VulkanBackend::render_scene 不得 panic（修前为 unimplemented!()）");
    let fb = r.expect("不应 panic");
    // V2.1 T-A1：渲染路径实装后，帧有效性必须与渲染自检**严格一致**——
    // 自检通过 ⇒ 出 `64*64*4` 有效帧；自检不通过/无设备 ⇒ 空缓冲（不伪造出图）。
    if v.render_available() {
        assert!(fb.is_valid(), "渲染自检通过 ⇒ render_scene 必须出图");
        assert_eq!(fb.pixels.len(), 64 * 64 * 4);
        assert_eq!((fb.width, fb.height), (64, 64));
    } else {
        assert!(
            !fb.is_valid(),
            "渲染路径不可用 ⇒ 须返回空帧（不得伪造出图）"
        );
        assert!(fb.pixels.is_empty() && fb.width == 0 && fb.height == 0);
    }
    println!("G7 VulkanBackend::render_scene 未 panic，耗时 = {e:?}");
}

/// **行宽守卫正式断言**（原 `#[ignore]` 已知缺陷取证，缺陷已修复）。
///
/// 契约：`set_pixel` / `get_pixel` 对**行宽**越界（`x >= width` 或 `y >= height`）
/// 必须**丢弃**，而不是像修复前那样把 `x == width` 回绕写到下一行第 0 列。
///
/// 断言强度（相对原取证版的三点加强）：
/// 1. **整块缓冲逐字节不变**——不是"第 1 行第 0 列没变"，而是全 16384 字节比对，
///    因此任何位置的静默写错都会被抓到；
/// 2. **25 组穷举**：`x, y ∈ {0, 63, 64, 65, u32::MAX}`，其中 (0,0) 与 (63,63)
///    为合法坐标（断言"**不得被守卫误杀**"）共 4 组，其余 21 组必须整块不变；
/// 3. **`get_pixel` 同型断言**：越界坐标须返回 `None`，且读不出任何像素值。
#[test]
fn g8_out_of_range_coords_dropped_never_row_wrap() {
    const COORDS: [u32; 5] = [0, 63, 64, 65, u32::MAX];
    const SIDE: u32 = 64;
    let red = Vec3::new(1.0, 0.0, 0.0);

    // ── 判别性前提（防"恒真"）──────────────────────────────────────────
    // 修复前的写入判据只有"全局缓冲区末尾检查"（`last >= pixels.len()`）。
    // 故须证明：本探针中最关键的越界用例 `x == width` 若无行宽守卫，其目标
    // 偏移**落在缓冲区内部**，尾检查救不了它——即该用例确实只由行宽守卫兜住。
    let would_be = (0usize + SIDE as usize) * 4; // set_pixel(64, 0) 的旧目标 = 像素 (0,1)
    assert!(
        would_be < SIDE as usize * SIDE as usize * 4,
        "判别性前提失效：(64,0) 的旧目标偏移 {would_be} 竟在缓冲外，尾检查即可拦下，本探针不敏感"
    );

    let mut rows: Vec<String> = Vec::with_capacity(25);
    let mut n_legal = 0usize;
    let mut n_dropped = 0usize;

    for &y in COORDS.iter() {
        for &x in COORDS.iter() {
            // 每组都用**全新且有效**的缓冲：排除"被拒空缓冲恒真"的伪通过
            // （此前教训：删掉 checked_mul 后探针仍"存活"）。
            let mut fb = Framebuffer::new(SIDE, SIDE);
            assert!(
                fb.is_valid() && fb.pixels.len() == (SIDE * SIDE * 4) as usize,
                "探针前提失效：需有效 {SIDE}×{SIDE} 缓冲（len={}）",
                fb.pixels.len()
            );
            let before = fb.pixels.clone();

            let legal = x < SIDE && y < SIDE;
            fb.set_pixel(x, y, red);
            let changed = fb.pixels.iter().zip(before.iter()).filter(|(a, b)| a != b).count();

            if legal {
                // 合法坐标必须照常写入 —— 新守卫的主要风险是"误杀合法坐标"。
                assert!(changed > 0, "合法坐标 ({x},{y}) 的写入被守卫误杀（changed={changed}）");
                assert_eq!(
                    fb.get_pixel(x, y),
                    Some([255, 0, 0, 255]),
                    "合法坐标 ({x},{y}) 写入后读不回"
                );
                n_legal += 1;
                rows.push(format!("({x:>10},{y:>10})  合法   写入生效 changed={changed}"));
            } else {
                // 越界：整块缓冲逐字节不变（覆盖"任何位置被写"）。
                assert_eq!(
                    changed, 0,
                    "越界坐标 ({x},{y}) 改动了 {changed} 个字节，应整块不变（回绕写错像素）"
                );
                assert_eq!(
                    fb.pixels, before,
                    "越界坐标 ({x},{y}) 之后整块缓冲非逐字节一致 —— 存在静默写入"
                );
                // get_pixel 同型：必须 None，且读不出任何像素值。
                let got = fb.get_pixel(x, y);
                assert!(got.is_none(), "get_pixel({x},{y}) 越界却读出 {got:?}，应为 None");
                assert_ne!(
                    got,
                    Some([255, 0, 0, 255]),
                    "get_pixel({x},{y}) 回绕读回了刚写入的红色"
                );
                // 强化：越界读**不得**命中回绕目标像素 (x-width, y+1) 的值。
                if x >= SIDE && y < SIDE {
                    assert_ne!(
                        fb.get_pixel(x - SIDE, y + 1),
                        Some([255, 0, 0, 255]),
                        "越界写 ({x},{y}) 污染了回绕目标 ({},{})",
                        x - SIDE,
                        y + 1
                    );
                }
                n_dropped += 1;
                rows.push(format!("({x:>10},{y:>10})  越界   整块不变 changed=0 get=None"));
            }
        }
    }

    assert_eq!(n_legal + n_dropped, 25, "穷举组数不对");
    // 合法组 = x∈{0,63} × y∈{0,63} = 4 组（(0,0)(0,63)(63,0)(63,63)）。
    assert_eq!(n_legal, 4, "合法坐标组数应为 4（x,y 各取 0 与 63）");
    assert_eq!(n_dropped, 21, "越界坐标组数应为 21");
    println!("G8 25 组穷举结果（Framebuffer 64×64，红色写入）：\n{}", rows.join("\n"));
    println!("G8 合计：合法 {n_legal} 组全部生效，越界 {n_dropped} 组全部整块不变");
}

/// **边界精确性 + 合法坐标未被误杀**（新守卫的主要风险面）。
#[test]
fn g9_legal_coords_survive_and_boundary_is_exact() {
    const SIDE: u32 = 64;
    // 1) 四个合法角点 + 中心：写入/读回必须全部正确。
    let mut fb = Framebuffer::new(SIDE, SIDE);
    let probes: [(u32, u32, [u8; 4]); 5] = [
        (0, 0, [255, 0, 0, 255]),
        (SIDE - 1, 0, [0, 255, 0, 255]),
        (0, SIDE - 1, [0, 0, 255, 255]),
        (SIDE - 1, SIDE - 1, [255, 255, 255, 255]),
        (32, 32, [255, 0, 255, 255]),
    ];
    for &(x, y, rgb) in &probes {
        let c = Vec3::new(rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0);
        fb.set_pixel(x, y, c);
    }
    for &(x, y, rgb) in &probes {
        assert_eq!(fb.get_pixel(x, y), Some(rgb), "合法坐标 ({x},{y}) 读回不符");
    }
    // 未被写过的合法像素仍为全零 —— 证明守卫没有越界改写邻居。
    // 注意：新缓冲是**全零**初始化，alpha 也为 0；只有 set_pixel 才写 255。
    assert_eq!(fb.get_pixel(1, 0), Some([0, 0, 0, 0]), "相邻合法像素 (1,0) 被污染");

    // 2) 边界精确性：x == width-1（最后合法列）能写能读；x == width（越界）必丢弃。
    let mut edge = Framebuffer::new(SIDE, SIDE);
    edge.set_pixel(SIDE - 1, 10, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(
        edge.get_pixel(SIDE - 1, 10),
        Some([255, 0, 0, 255]),
        "x == width-1 是合法最后列，必须能写能读"
    );
    let snap = edge.pixels.clone();
    edge.set_pixel(SIDE, 10, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(
        edge.get_pixel(SIDE, 10),
        None,
        "x == width 越界，get_pixel 须返回 None"
    );
    assert_eq!(edge.pixels, snap, "x == width 的写入必须被整块丢弃（不得回绕到下一行）");
    // 同型：y == height 越界
    edge.set_pixel(10, SIDE, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(edge.pixels, snap, "y == height 的写入必须被丢弃");
    assert_eq!(edge.get_pixel(10, SIDE), None, "y == height 越界须返回 None");
    println!("G9 合法坐标 5/5 正确读写；x=width-1 可写可读，x=width / y=height 全部丢弃");
}

/// **拒绝语义未被新守卫改动** + `CpuRefBackend::render_scene` 极端请求秒回。
#[test]
fn g10_refusal_semantics_and_render_latency_unchanged() {
    // 超限尺寸：is_valid()==false 且 pixels 为空（新守卫不得改变拒绝路径）。
    for &(w, h) in &[(u32::MAX, u32::MAX), (1 << 31, 1 << 31), (65536, 65536), (65_536 + 1, 1)] {
        let mut fb = Framebuffer::new(w, h);
        assert!(!fb.is_valid(), "({w},{h}) 超限应is_valid()==false");
        assert!(fb.pixels.is_empty(), "({w},{h}) 超限应 pixels 为空");
        // 拒绝缓冲上的写/读必须是惰性 no-op，不 panic。
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fb.set_pixel(0, 0, Vec3::new(1.0, 1.0, 1.0));
        }))
        .is_ok());
        assert_eq!(fb.get_pixel(0, 0), None, "被拒缓冲 get_pixel 须为 None");
        assert!(fb.pixels.is_empty(), "被拒缓冲不得接受任何写入");
    }
    // 极端请求秒回（修前 >300 s 挂死；上轮实测 ~160 ns）。
    for &(w, h) in &[(u32::MAX, u32::MAX), (1 << 31, 1 << 31)] {
        let t0 = Instant::now();
        let mut dev = CpuRefBackend;
        let out = dev.render_scene(&scene_req(w, h));
        let e = t0.elapsed();
        assert!(e.as_secs() < 10, "render_scene({w},{h}) 耗时 {e:?}，疑似挂死");
        assert!(!out.is_valid() && out.pixels.is_empty(), "({w},{h}) 应被拒");
        println!("G10 render_scene({w},{h}) 耗时 = {e:?}");
    }
    let t1 = Instant::now();
    let ok = CpuRefBackend.render_scene(&scene_req(64, 64));
    let e1 = t1.elapsed();
    assert!(ok.is_valid() && ok.pixels.len() == 64 * 64 * 4, "64×64 正常路径不得被守卫误伤");
    println!("G10 render_scene(64,64) 耗时 = {e1:?}（正常路径未回退）");
}
