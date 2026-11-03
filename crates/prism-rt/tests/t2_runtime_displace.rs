//! T2 / G-10 判据：**运行期位移材质槽**（`细分 → 位移`，不落盘中间网格）。
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V1 | 幅值 0 / `-0.0` / 无贴图句柄 ⇒ 与"没有位移槽"逐位相同 | 逐位（构造性短路） |
//! | V2 | 运行期结果 vs 烘焙期（`subdivide → displace_positions`） | ≤ 1e-6（实测逐位 0） |
//! | V3 | 三后端 `differing_pixels` | 软件 ICD 上 = 0 |
//! | V4 | 共享边无裂缝 | 内部边两侧同索引（缝隙恒 0）+ 反例证明判据有牙 |
//!
//! # 这一条与 W4/G-9 的分界
//!
//! W4 验的是"细分 + 位移这件事算得对"（Hausdorff、uv 缝、法线口径），G-9 验的是
//! "细分支持折痕"。T2 验的是**产品出口**：`MeshDesc.displacement` 里的槽在渲染
//! 时按需触发细分与位移，且
//!
//! * `amplitude == 0` 时**连细分都不做**（不是"细分后加 0"）；
//! * 运行期与烘焙期是**同一份数学**（不是两套实现碰巧接近）；
//! * 产物进 GPU 的三个后端逐位一致。
//!
//! # 为什么 V1 用"构造性短路"而不是"结果相等"
//!
//! `p + n·0` 在 `p = -0.0` 时会得到 `+0.0`（`-0.0 + 0.0 = +0.0`），位型变了。
//! 只要"先细分再位移"这条路径被执行过，就存在改位型的机会。故本模块的口径是
//! **返回 `None`、调用方保留原网格**——那才是逐位相同的充分条件。

use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, CurveReq, Framebuffer, GpuDevice, LightReq,
    SceneRenderRequest, SphereReq,
};
use prism_math::Vec3;
use prism_rt::displace::{
    bake_triangles, displace_positions, tessellate_displace, DisplaceParams, RuntimeDisplace,
};
use prism_rt::integrator::Scene;
use prism_rt::texsample::TexTable;

// ---------------------------------------------------------------------------
// 素材
// ---------------------------------------------------------------------------

/// 16×16 灰度高度图（**纯整数**构造：`0` 或 `255`，亮度恰为 `0` / `1`）。
///
/// 用二值图而不是渐变：V2-b 要把"常值高度 ⇒ 位移量恰为 `amplitude`"写成解析
/// 断言，图里一旦有中间值，解析值就得靠采样器插值约定去凑，判据会变得不可读。
fn height_map(scene: &mut Scene, on: bool) -> u32 {
    let mut px: Vec<[f32; 4]> = Vec::with_capacity(16 * 16);
    let v: f32 = if on { 1.0 } else { 0.0 };
    for _ in 0..(16 * 16) {
        px.push([v, v, v, 1.0]);
    }
    scene.add_texture_linear(16, 16, &px)
}

/// 立方体三角网格（8 顶点 12 三角，带顶点域 uv；**闭合**，无 uv 缝）。
fn cube_mesh() -> (Vec<Vec3>, Vec<u32>, Vec<[f32; 2]>) {
    let h: f32 = 0.8;
    let mut pos: Vec<Vec3> = Vec::with_capacity(8);
    for &z in &[-h, h] {
        for &y in &[-h, h] {
            for &x in &[-h, h] {
                pos.push(Vec3::new(x, y, z));
            }
        }
    }
    let idx: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2, // z = −h
        4, 6, 7, 4, 7, 5, // z = +h
        0, 4, 5, 0, 5, 1, // y = −h
        2, 3, 7, 2, 7, 6, // y = +h
        1, 5, 7, 1, 7, 3, // x = +h
        0, 2, 6, 0, 6, 4, // x = −h
    ];
    let mut uv: Vec<[f32; 2]> = Vec::with_capacity(8);
    for &z in &[0.0f32, 1.0] {
        for &y in &[0.0f32, 1.0] {
            for &x in &[0.0f32, 1.0] {
                uv.push([x, 0.5 * y + 0.5 * z]);
            }
        }
    }
    (pos, idx, uv)
}

/// 一个正对相机、**倾斜 45°** 的四边形（V2-b 的解析载体）。
///
/// 倾斜的理由与 T0 的 `plane_mesh` 同：正对相机的平面沿法线位移只改深度，
/// 画面不变；倾斜后位移同时改轮廓与明暗。
fn tilted_quad() -> (Vec<Vec3>, Vec<u32>, Vec<[f32; 2]>) {
    let n = std::f32::consts::FRAC_1_SQRT_2;
    let u = Vec3::new(1.0, 0.0, 0.0);
    let v = Vec3::new(0.0, n, n);
    let pos = vec![u * -1.0 - v, u - v, u + v, v - u];
    let idx = vec![0u32, 2, 1, 0, 3, 2];
    let uv = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    (pos, idx, uv)
}

/// 帧差统计（与 `qa_curve_parity` 同口径）。
struct Parity {
    total_pixels: usize,
    max_channel_delta: u32,
    differing_pixels: usize,
    first: Option<(usize, [u8; 4], [u8; 4])>,
}

fn parity(a: &Framebuffer, b: &Framebuffer) -> Parity {
    assert_eq!((a.width, a.height), (b.width, b.height), "帧尺寸必须一致");
    let mut p = Parity {
        total_pixels: (a.width * a.height) as usize,
        max_channel_delta: 0,
        differing_pixels: 0,
        first: None,
    };
    for i in 0..p.total_pixels {
        let pa = &a.pixels[i * 4..i * 4 + 4];
        let pb = &b.pixels[i * 4..i * 4 + 4];
        let mut d = 0u32;
        for c in 0..4 {
            d = d.max((pa[c] as i32 - pb[c] as i32).unsigned_abs());
        }
        if d > 0 {
            p.differing_pixels += 1;
            if p.first.is_none() {
                p.first = Some((i, [pa[0], pa[1], pa[2], pa[3]], [pb[0], pb[1], pb[2], pb[3]]));
            }
        }
        p.max_channel_delta = p.max_channel_delta.max(d);
    }
    p
}

fn is_software_icd(device: &str) -> bool {
    let d = device.to_ascii_lowercase();
    d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
}

fn gpu_backend() -> Option<prism_gpu::vulkan::VulkanBackend> {
    let be = prism_gpu::vulkan::VulkanBackend::new();
    if !be.render_available() {
        println!("无可用设备：0 差异判据跳过（实机项 = 永不进门禁）");
        return None;
    }
    Some(be)
}

/// 只装一个 `TriSoup` + 一盏灯的最小请求（把几何差异隔离出来）。
fn trisoup_request(w: u32, h: u32, tris: Vec<prism_geometry::bvh::Triangle>) -> SceneRenderRequest {
    SceneRenderRequest {
        width: w,
        height: h,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.0, -2.5),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![LightReq {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }],
        spheres: Vec::<SphereReq>::new(),
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        curves: vec![CurveReq::TriSoup { triangles: tris }],
        ..Default::default()
    }
}

/// 背景判据（与 `w4_w5_gpu_parity` 同阈值）。
fn is_background(px: [u8; 4]) -> bool {
    px[0] <= 12 && px[1] <= 15 && px[2] <= 22
}

// ---------------------------------------------------------------------------
// V1：幅值 0 ⇒ 构造性短路
// ---------------------------------------------------------------------------

/// **V1：`amplitude == 0` / `-0.0` / 无贴图句柄 ⇒ `tessellate_displace` 返回 `None`。**
///
/// `None` 的语义是"调用方保留原网格"：连细分都不做，因此不会出现
/// "细分后位移 0" 那种改位型的机会（`-0.0 + 0.0 = +0.0`）。
/// 这是**结构性**判据，不是数值判据——没有"误差多大"的余地。
#[test]
fn v1_zero_amplitude_short_circuits_constructively() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = cube_mesh();

    let cases: [(&str, RuntimeDisplace); 5] = [
        ("幅值 +0.0", RuntimeDisplace { texture: handle, amplitude: 0.0, scale: 1.0, levels: 3 }),
        ("幅值 -0.0", RuntimeDisplace { texture: handle, amplitude: -0.0, scale: 1.0, levels: 3 }),
        (
            "无贴图句柄",
            RuntimeDisplace { texture: 0, amplitude: 1.0, scale: 1.0, levels: 3 },
        ),
        (
            "幅值 0 且无句柄",
            RuntimeDisplace { texture: 0, amplitude: 0.0, scale: 1.0, levels: 3 },
        ),
        // 倍数也是 0（`0 * x == 0`，含 `-0.0 * x`）——不因缩放而"活过来"。
        (
            "幅值 0 且缩放 4",
            RuntimeDisplace { texture: handle, amplitude: 0.0, scale: 4.0, levels: 3 },
        ),
    ];
    for (name, slot) in cases {
        assert!(slot.is_identity(), "[{name}] 必须判为恒等");
        let got = tessellate_displace(&pos, &idx, &uv, slot, table, None);
        assert!(got.is_none(), "[{name}] 恒等槽必须返回 None（构造性短路），实际产出了网格");
    }

    // 对照：非零幅值必须真的产出网格（否则上面五条都是空转）。
    let live = RuntimeDisplace { texture: handle, amplitude: 0.12, scale: 1.0, levels: 2 };
    let got = tessellate_displace(&pos, &idx, &uv, live, table, None).expect("非零幅值必须产出网格");
    assert_eq!(got.indices.len() / 3, 12 * 4 * 4, "k=2 ⇒ 12·4² = 192 三角");
    println!("V1：五条恒等槽全部短路；对照（amp=0.12,k=2）产出 {} 三角", got.indices.len() / 3);
}

/// **V1-b：产品路径上，幅值 0 的槽与"没有槽"渲染出的帧逐字节相同。**
///
/// V1 的结构性判据（返回 `None`）说明"没有中间网格"；这条说明产品出口的**像素**
/// 也没变。两条一起才排掉"`None` 之后调用方又做了一次恒等变换"。
#[test]
fn v1_zero_amplitude_frame_is_bitwise_identical_on_product_path() {
    use prism_render::rt_backend::RtBackend;
    use prism_render::build_render_request;
    use prism_gpu::SceneRenderRequest;

    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = tilted_quad();
    let disp = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: handle, amplitude: 0.0, scale: 1.0, levels: 2 },
        table,
        None,
    );
    assert!(disp.is_none(), "幅值 0 必须短路");

    // 两条产品路径：带「幅值 0 的槽」与「完全没有槽」。
    let build = |with_slot: bool| -> prism_core::Scene {
        let mut s = prism_core::Scene::default();
        s.radius = 1.0;
        s.orbit_radius = 3.4;
        let mut m = prism_core::MeshDesc::with_uv(pos.clone(), idx.clone(), uv.clone());
        if with_slot {
            m.displacement = Some(prism_core::DisplacementSlot::new(handle, 0.0, 1.0, 2));
        }
        s.attach.textures.push(prism_core::TextureDesc::new(
            16,
            16,
            false,
            {
                let mut px: Vec<u8> = Vec::with_capacity(16 * 16 * 4);
                for _ in 0..(16 * 16) {
                    px.extend_from_slice(&[255, 255, 255, 255]);
                }
                px
            },
        ));
        s.attach.meshes.push(m);
        s
    };

    let render = |s: &prism_core::Scene| -> Framebuffer {
        let req: SceneRenderRequest = build_render_request(s, 0.0, 10.0, 32, 24);
        let mut dev: RtBackend = RtBackend::new(4, 2).with_threads(1);
        dev.render_scene(&req)
    };
    let a = render(&build(false));
    let b = render(&build(true));
    assert!(a.is_valid() && b.is_valid());
    let p = parity(&a, &b);
    println!(
        "V1-b：无槽 vs 幅值 0 的槽 ⇒ pixels={} max_channel_delta={} differing_pixels={}",
        p.total_pixels, p.max_channel_delta, p.differing_pixels
    );
    assert_eq!(p.max_channel_delta, 0, "幅值 0 的槽必须与无槽逐字节相同");
    assert_eq!(p.differing_pixels, 0);
}

// ---------------------------------------------------------------------------
// V2：运行期 vs 烘焙期
// ---------------------------------------------------------------------------

/// **V2：运行期 `tessellate_displace` 与烘焙期组合逐位相同。**
///
/// "烘焙期"= 调用方自己 `subdivide(k)` 再 `displace_positions(..)`（W4 的写法）。
/// 判据原文是 ≤1e-6；这里实测是 **0**（两者是同一份数学——运行期入口只负责
/// 把两段拼起来，不前也不后加任何运算）。逐位比 ≤1e-6 强，故判据只收紧不放宽。
#[test]
fn v2_runtime_matches_baked_bitwise() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = cube_mesh();

    for &(amp, scale, k) in &[(0.12f32, 1.0f32, 2u32), (0.05, 3.0, 3), (0.3, 0.5, 1)] {
        let slot = RuntimeDisplace { texture: handle, amplitude: amp, scale, levels: k };
        let run = tessellate_displace(&pos, &idx, &uv, slot, table, None)
            .expect("非零幅值必须产出网格");

        // ---- 烘焙期：照 W4 的写法手拼一遍 ----
        let sub = prism_geometry::subdiv::subdivide(&pos, &idx, &uv, k);
        let scaled: Vec<[f32; 2]> = sub
            .uv
            .iter()
            .map(|t| [t[0] * scale, t[1] * scale])
            .collect();
        let baked = displace_positions(
            &sub.positions,
            &sub.normals,
            &scaled,
            table,
            handle,
            DisplaceParams::new(amp),
        );

        assert_eq!(run.indices, sub.indices, "拓扑必须同源");
        assert_eq!(run.uv.len(), sub.uv.len(), "uv 长度必须同源");
        assert_eq!(run.positions.len(), baked.len());
        let mut worst = 0.0f32;
        for (a, b) in run.positions.iter().zip(baked.iter()) {
            worst = worst.max((*a - *b).length());
        }
        println!("V2：amp={amp} scale={scale} k={k} ⇒ 顶点数 {}，最大差 {worst:e}", run.positions.len());
        assert_eq!(worst, 0.0, "运行期与烘焙期必须逐位相同（实测最大差 {worst:e}）");
    }
}

/// **V2-b：常值高度图 ⇒ 每个顶点**恰好**沿法线移动 `amplitude`（解析值 ≤1e-6）。**
///
/// 这条不靠"两条路径互相印证"（那可能同错同对）：位移阶段是
/// `p' = p + n̂·(amp·亮度)`，常值图亮度恰为 `1.0`（`0.2126+0.7152+0.0722`，
/// 三个系数之和是精确的 1），`n̂` 由 `vertex_normals` 归一 ⇒ `|p' − p| = amp`
/// 逐顶点成立。基准 `p` 用**同一 k 的 `subdivide` 产物**（极限位置），
/// 因为细分本身就会移动顶点——把它算进位移量里量的是别的东西。
#[test]
fn v2_constant_height_displaces_by_exactly_the_amplitude() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true); // 全 1.0 ⇒ 亮度 1.0
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = tilted_quad();
    let amp: f32 = 0.25;
    let k: u32 = 2;
    let base = prism_geometry::subdiv::subdivide(&pos, &idx, &uv, k);
    let got = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: handle, amplitude: amp, scale: 1.0, levels: k },
        table,
        None,
    )
    .expect("非零幅值");
    assert_eq!(got.positions.len(), base.positions.len());

    let mut worst = 0.0f32;
    for (p1, p0) in got.positions.iter().zip(base.positions.iter()) {
        let d = (*p1 - *p0).length();
        worst = worst.max((d - amp).abs());
    }
    println!(
        "V2-b：常值高度 ⇒ {} 个顶点的位移量 vs amp={amp}，最大偏差 {worst:e}",
        got.positions.len()
    );
    assert!(worst <= 1e-6, "常值高度必须位移 amp，实测偏差 {worst:e}");

    // 反向：全 0 高度图 ⇒ 位移 0（不是恒等短路，因为幅值非 0，仍会细分）。
    let dark = height_map(&mut scene, false);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let zero = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: dark, amplitude: amp, scale: 1.0, levels: k },
        table,
        None,
    )
    .expect("幅值非 0 ⇒ 仍要细分");
    let mut worst0 = 0.0f32;
    for (p1, p0) in zero.positions.iter().zip(base.positions.iter()) {
        worst0 = worst0.max((*p1 - *p0).length());
    }
    println!("V2-b 反向：全黑高度图 ⇒ 最大位移 {worst0:e}（应为 0）");
    assert!(worst0 <= 1e-6, "全黑高度图必须位移 0，实测最大 {worst0:e}");
}

/// **级别上界：`levels` 在运行期入口就被钳到 `MAX_LEVELS`，不靠下游兜底。**
#[test]
fn v2_levels_are_clamped_at_the_runtime_entry() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    // 只有 2 个三角形的平面：4^8 × 2 = 131072 三角（可控）。
    let (pos, idx, uv) = tilted_quad();
    let cap = prism_geometry::subdiv::MAX_LEVELS;
    let got = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: handle, amplitude: 0.02, scale: 1.0, levels: cap + 997 },
        table,
        None,
    )
    .expect("非零幅值");
    assert_eq!(
        got.indices.len() / 3,
        2 * 4usize.pow(cap),
        "级别必须钳到 MAX_LEVELS = {cap}"
    );
    println!("V2-levels：请求 {} 级 ⇒ 实际 {cap} 级，{} 三角", cap + 997, got.indices.len() / 3);
}

// ---------------------------------------------------------------------------
// V3：三后端 0 差异
// ---------------------------------------------------------------------------

/// **V3：运行期位移的产物进 GPU ⇒ 软件 ICD 上 `differing_pixels = 0`。**
///
/// 三个后端里两路在这里实测（CPU 参考 + Vulkan/llvmpipe）；WebGPU 腿在本机
/// 无浏览器环境，**不冒充已验**（如实登记在报告里）。CPU rt 的第三路已在
/// T0 的 P2（`differing_pixels = 0`）里与产品路径对过。
#[test]
fn v3_runtime_displaced_geometry_matches_on_gpu_backends() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = cube_mesh();
    let got = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: handle, amplitude: 0.12, scale: 2.0, levels: 2 },
        table,
        None,
    )
    .expect("非零幅值");
    let tris = bake_triangles(&got.positions, &got.indices);
    assert_eq!(tris.len(), got.indices.len() / 3);
    assert!(tris.len() >= 100, "至少要有 100 个三角形，实测 {}", tris.len());

    let req = trisoup_request(32, 24, tris);
    let mut cpu = CpuRefBackend;
    let cpu_fb = cpu.render_scene(&req);
    assert!(cpu_fb.is_valid(), "CPU 参考必须出有效帧");
    // 非平凡：画面被几何覆盖（否则"两边都空"也能拿 0 差异）。
    let lit = (0..(32 * 24))
        .filter(|i| !is_background(cpu_fb.get_pixel(i % 32, i / 32).unwrap()))
        .count();
    assert!(lit >= 60, "位移后的立方体应覆盖画面，实测 {lit} 个非背景像素");

    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    let gpu_fb = backend.render_scene(&req);
    assert!(gpu_fb.is_valid(), "GPU 必须产出有效帧");
    let p = parity(&cpu_fb, &gpu_fb);
    println!(
        "V3：{} 三角，{lit} 个非背景像素；CPU vs GPU differing_pixels={} max_channel_delta={} \
         software_icd={} first_diff={:?}",
        got.indices.len() / 3,
        p.differing_pixels,
        p.max_channel_delta,
        software,
        p.first
    );
    if software {
        assert_eq!(p.differing_pixels, 0, "软件 ICD 上必须 0 差异（首个差异 {:?}）", p.first);
        assert_eq!(p.max_channel_delta, 0, "软件 ICD 上必须逐字节一致");
    } else {
        assert!(
            p.max_channel_delta <= 1 && p.differing_pixels * 200 <= p.total_pixels,
            "真机档位不达标：max={} diff={}",
            p.max_channel_delta,
            p.differing_pixels
        );
    }
}

// ---------------------------------------------------------------------------
// V4：共享边无裂缝
// ---------------------------------------------------------------------------

/// 沿"每条内部边"量裂缝：内部边两侧的三角形必须引用**同一对顶点下标**。
///
/// 返回 `(内部边数, 不一致边数)`。索引一致 ⇒ 两侧的端点位置是同一份数据
/// ⇒ 几何缝隙恒为 0（不是"小于容差"）。若实现改成逐三角形独立细分（soup），
/// 两侧会各自生成边点 ⇒ 下标不同 ⇒ 计数 > 0。
fn interior_edge_cracks(indices: &[u32]) -> (usize, usize) {
    use std::collections::BTreeMap;
    let mut count: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
    for c in indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            let e = count.entry(key).or_insert((0, 0));
            if a < b {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
    let mut interior = 0usize;
    let mut bad = 0usize;
    for (_, (fwd, rev)) in count {
        if fwd + rev != 2 {
            continue; // 边界边（1 次）或非流形（>2）：不算"内部边"
        }
        interior += 1;
        // 一致绕序 ⇒ 一个方向出现 1 次、反向 1 次。若两侧各自生成独立边点，
        // 这里会变成两条各出现 1 次的**不同**边 ⇒ 上面 `count` 里根本没有它们，
        // 表现为"内部边数明显偏少 + 顶点数偏多"。故这条同时校验两件事。
        if fwd != 1 || rev != 1 {
            bad += 1;
        }
    }
    (interior, bad)
}

/// **V4：运行期细分 + 位移后，共享边上一条裂缝都没有。**
///
/// 三条断言：
/// 1. 索引流里没有"同位置但不同下标"的顶点 —— 顶点数是理论值
///    （`V + E` 与 `V + E + F` 的细分公式精确相等），多了就是裂开了；
/// 2. 每条内部边两侧绕序相反（`fwd == 1 && rev == 1`）⇒ 两侧引用同一对下标
///    ⇒ 缝隙**恒为 0**；
/// 3. 反例：把三角形拆成 soup（每个三角形独立顶点）再量一次，内部边数塌成 0
///    —— 证明这套量法真的能看见裂缝，不是恒真的空断言。
#[test]
fn v4_shared_edges_have_no_cracks() {
    let mut scene = Scene::default();
    let handle = height_map(&mut scene, true);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (pos, idx, uv) = cube_mesh();
    // 立方体：V=8、E=18、F=12。k=1 ⇒ V' = 8+18 = 26、E' = 3·F'/2 = 3·48/2 = 72、F' = 48。
    let k: u32 = 2;
    let got = tessellate_displace(
        &pos,
        &idx,
        &uv,
        RuntimeDisplace { texture: handle, amplitude: 0.1, scale: 1.0, levels: k },
        table,
        None,
    )
    .expect("非零幅值");

    let f = 12usize * 4usize.pow(k);
    // 闭合三角网格：E = 3F/2；V = V₀ + Σ E_{i-1}。用欧拉公式反推校验。
    let mut v_want = 8usize;
    let mut f_cur = 12usize;
    let mut e_cur = 18usize;
    for _ in 0..k {
        v_want += e_cur;
        f_cur *= 4;
        e_cur *= 4;
    }
    assert_eq!(f_cur, f);
    assert_eq!(got.positions.len(), v_want, "顶点数必须等于细分公式（多出来 = 裂开了）");
    assert_eq!(got.indices.len() / 3, f);

    let (interior, bad) = interior_edge_cracks(&got.indices);
    let e_want = 3 * f / 2;
    println!(
        "V4：k={k} ⇒ V={} E(理论)={e_want} F={f}，内部边 {interior}，绕序不一致 {bad}",
        got.positions.len()
    );
    assert_eq!(interior, e_want, "内部边数必须等于 3F/2（闭合网格）");
    assert_eq!(bad, 0, "内部边两侧必须引用同一对顶点（否则就是裂缝）");

    // 反例：拆 soup（每三角形独立顶点）⇒ 内部边塌成 0，判据能看见"裂缝"。
    let mut soup: Vec<u32> = Vec::with_capacity(got.indices.len());
    for i in 0..got.indices.len() {
        soup.push(i as u32);
    }
    let (soup_interior, _) = interior_edge_cracks(&soup);
    println!("V4 反例：soup 化之后内部边 = {soup_interior}（应为 0）");
    assert_eq!(soup_interior, 0, "soup 化后不存在共享边 —— 这套量法确实能看见裂缝");
}
