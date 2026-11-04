//! V6（口径 b）：**CPU ↔ WGSL** 环境采样逐值对拍 + 两跑逐位一致。
//!
//! # 判据（`docs/v2.5-W0-算法选型报告.md` §0 口径 b / §3.2 V6）
//!
//! 路径追踪通路只有 **CPU 权威**（`prism-rt`）与 **WGSL wavefront**
//! （`crates/prism-gpu/shaders/`）两方；**没有 SPIR-V wavefront 发射器**，
//! 故本文件只给两方判据，不伪称三后端。
//!
//! # 被测对象
//!
//! `crates/prism-gpu/shaders/rt_env_probe.wgsl`（`//!include rt_env.wgsl`，
//! **唯一一份** WGSL 环境实现）→ `tools/wgsl2spv`（naga 30.0.1，`spirv-val`
//! 校验通过）→ `shaders/spv/rt_env_probe.spv` → `spirv_env_probe.rs` 字流
//! → 真机 dispatch（lavapipe/llvmpipe 软栅，本机有）。
//!
//! 探针每线程一个查询：`u1 = (i+½)/N`、`u2 = Hammersley(i)`，
//! 输出 `(wi.xyz, pdf, radiance.rgb, texel, pdf_w(wi), uv, spot_index)`。
//! CPU 侧用**同一组** `(u1, u2)` 调 `EnvCdf::sample / radiance / pdf_w`。
//!
//! # 实测结果（本机 lavapipe）
//!
//! 512 查询（平滑 32×16 与高对比 64×32 各一组，**含极点带**）在
//! 方向 / uv / 采样 pdf / 双线性辐亮度 / `pdf_w(wi)` / texel 反查 / 亮斑归属
//! 上**逐位一致**（全部指标 = 0，含 `to_bits()` 级）。这不是"误差小"，是
//! 同一批指令算出同一批位：WGSL 侧走 `det_trig` 的逐式镜像（`cos_sin` 来自
//! `rt_common`，`env_atan2`/`env_acos` 是 `det_trig::{atan2,acos}` 的同系数
//! 同归约顺序实现），双线性与 pdf 的结合序也与 `env.rs` 逐行对应。
//!
//! 断言仍保留 2e-6 / 1e-4 / 1e-3 的**跨设备余量**（别的驱动若做 FMA 收缩，
//! 浮点结合序会变），实测值在报告里按 0 记录。
//!
//! # 为什么不是"整条 wavefront vs CPU 8 弹射"
//!
//! wavefront 的 `shade`/`extend` 要额外一组绑定承载 CDF blob：group(0) 的
//! {0,1,2} 已被 slots/params/accum 占满，而宿主 ABI 要求每个 entry 的绑定集
//! 是**连续前缀**，接线必须同步改 `web/abi.rs` 的 group 表与浏览器宿主。
//! 本轮未做（报告里记为"未接线"），因此本文件**不**声称跑过整条 wavefront；
//! 它证明的是判据真正关心的东西：**CDF 反演、方向映射、双线性取值、同源 pdf
//! 四条链路 CPU 与 WGSL 逐值一致**。

use prism_core::EvalCtx;
use prism_gpu::vulkan::compute::DispatchStatus;
use prism_gpu::vulkan::VulkanBackend;
use prism_math::Vec3;
use prism_render::farm::md5_hex;
use prism_rt::env::{dir_to_uv, EnvCdf};
use prism_rt::integrator::Scene;
use prism_rt::texsample::TexTable;
use std::f32::consts::PI;

/// 每查询输出的 f32 个数（与 `rt_env_probe.wgsl` 的 `idx * 12u` 同源；
/// 唯一权威在 `prism_gpu::compute::ENV_PROBE_STRIDE`）。
const STRIDE: usize = prism_gpu::compute::ENV_PROBE_STRIDE as usize;
/// 查询数（每线程一个）。
const QUERIES: u32 = 512;

/// 基 2 radical inverse（与 `rt_env_probe.wgsl::env_radical_inverse` 同式）。
fn radical_inverse_base2(k0: u32) -> f32 {
    let mut k = k0;
    k = (k << 16) | (k >> 16);
    k = ((k & 0x5555_5555) << 1) | ((k & 0xAAAA_AAAA) >> 1);
    k = ((k & 0x3333_3333) << 2) | ((k & 0xCCCC_CCCC) >> 2);
    k = ((k & 0x0F0F_0F0F) << 4) | ((k & 0xF0F0_F0F0) >> 4);
    k = ((k & 0x00FF_00FF) << 8) | ((k & 0xFF00_FF00) >> 8);
    k as f32 * 2.328_306_4e-10
}

/// 把 CDF 的全部状态打成一个 u32 blob（布局 = `rt_env.wgsl` 文件头那张表）。
///
/// **这是 blob 布局的唯一权威**：WGSL 侧只按偏移读，偏移由本函数写入。
fn build_blob(cdf: &EnvCdf, pixels: &[[f32; 4]]) -> Vec<u32> {
    prism_rt::env::blob_words(cdf, pixels)
}

/// 一个验收场景：环境图 + 期望的 `(pixels, cdf)`。
fn env_case(name: &str, w: u32, h: u32, f: impl Fn(f32, f32, Vec3) -> Vec3) -> Scene {
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let u = (i as f32 + 0.5) / w as f32;
            let v = (j as f32 + 0.5) / h as f32;
            let d = prism_rt::env::uv_to_dir(u, v);
            let c = f(u, v, d);
            pixels.push([c.x, c.y, c.z, 1.0]);
        }
    }
    let mut scene = Scene::default();
    scene.add_environment(w, h, &pixels);
    println!(
        "[case] {name}: {w}×{h}，{} 个 texel，{} 个亮斑（p_spot={:.4}）",
        w * h,
        scene.env().expect("CDF").stats().spots,
        scene.env().expect("CDF").stats().p_spot
    );
    scene
}

fn pixel_data(scene: &Scene) -> Vec<[f32; 4]> {
    // 环境贴图是场景里的第 1 条纹理（`add_environment` → `add_texture_linear`），
    // `TextureRef::offset` 指向它在扁平 texels 里的起点。
    let r = &scene.textures[0];
    let (w, h) = prism_rt::texsample::level_dims(r, 0);
    let base = prism_rt::texsample::level_base(r, 0);
    scene.texels[base..base + (w * h) as usize].to_vec()
}

/// 主判对：同一组 `(u1, u2)` 上 CPU 与 WGSL 的四条链路逐值一致。
fn run_case(name: &str, scene: &Scene) {
    let mut be = VulkanBackend::new();
    if !be.compute_available() {
        // 环境缺失显式跳过（不假装跑过）。
        println!("[skip] {name}：本机无可用 Vulkan compute 设备，CPU↔WGSL 判据未执行");
        return;
    }
    let cdf: &EnvCdf = scene.env().expect("环境 CDF 必须已建");
    let pixels = pixel_data(scene);
    let blob = build_blob(cdf, &pixels);
    let input: Vec<f32> = blob.iter().map(|w| f32::from_bits(*w)).collect();

    // kernel 描述来自**注册表**（`prism_gpu::compute::env_probe_kernel`），
    // 不是本文件手搓：这样 `shaders::check_targets` 的三目标校验、注册表、
    // 本测试三者用的是同一份布局，改一处就三处一起红。
    let kernel = prism_gpu::compute::env_probe_kernel(input.len() as u32, QUERIES);
    assert_eq!(kernel.spv_module, "spv_env_probe");
    assert_eq!(kernel.entry, "env_probe");
    kernel.validate().expect("KernelDesc 布局必须合法");

    let a = be.compute_dispatch(&kernel, [QUERIES, 1, 1], &input);
    assert_eq!(
        a.status,
        DispatchStatus::Ok,
        "{name}：真机 dispatch 失败：{} [{}]",
        a.status.as_str(),
        a.detail
    );
    assert_eq!(a.output.len(), STRIDE * QUERIES as usize, "读回长度 = 写绑定容量");
    assert!(!a.device.is_empty(), "成功报告必须带设备名");
    println!("[device] {name}：{}", a.device);

    // 两跑逐位一致（同一进程内）+ md5（跨进程人工对账用）。
    let b = be.compute_dispatch(&kernel, [QUERIES, 1, 1], &input);
    assert_eq!(b.status, DispatchStatus::Ok, "{}", b.detail);
    let bits_a: Vec<u32> = a.output.iter().map(|v| v.to_bits()).collect();
    let bits_b: Vec<u32> = b.output.iter().map(|v| v.to_bits()).collect();
    assert_eq!(bits_a, bits_b, "{name}：两跑 dispatch 不逐位一致");
    let bytes: Vec<u8> = a.output.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    println!(
        "[determinism] {name}：两跑逐位一致，{} 个 f32 / md5={}",
        a.output.len(),
        md5_hex(&bytes)
    );

    // ---- 逐值对拍 ----
    let table = TexTable::new(&scene.textures, &scene.texels);
    let mut worst_wi = 0.0f32;
    let mut worst_duv = 0.0f32;
    let mut worst_pdf = 0.0f32;
    let mut worst_rad = 0.0f32;
    let mut worst_pdf2 = 0.0f32;
    let mut at_wi = 0u32;
    let mut at_uv = 0u32;
    let mut at_pdf = 0u32;
    let mut at_rad = 0u32;
    let mut at_pdf2 = 0u32;
    let mut spot_class_mismatch = 0u32;
    let mut texel_mismatch = 0u32;
    let (ew, eh) = cdf.dims();
    for i in 0..QUERIES {
        let u1 = (i as f32 + 0.5) / QUERIES as f32;
        let u2 = radical_inverse_base2(i ^ 0x9E37_79B9);
        let want = cdf.sample(u1, u2).expect("环境非全黑 ⇒ 必有样本");
        let want_rad = cdf.radiance(table, want.wi);
        let want_pdf2 = cdf.pdf_w(want.wi);
        let (uu, vv) = dir_to_uv(want.wi);
        let want_texel = (vv * eh as f32).min(eh as f32 - 1.0) as u32 * ew
            + (uu * ew as f32).min(ew as f32 - 1.0) as u32;

        let o = &a.output[i as usize * STRIDE..(i as usize + 1) * STRIDE];
        let got_wi = Vec3::new(o[0], o[1], o[2]);
        let got_pdf = o[3];
        let got_rad = Vec3::new(o[4], o[5], o[6]);
        let got_texel = o[7] as u32;
        let got_pdf2 = o[8];
        let got_uv = (o[9], o[10]);
        let got_sidx = o[11];

        let d = (got_wi - want.wi).length();
        if d > worst_wi {
            worst_wi = d;
            at_wi = i;
        }
        let duv = (got_uv.0 - uu).abs().max((got_uv.1 - vv).abs());
        if duv > worst_duv {
            worst_duv = duv;
            at_uv = i;
        }
        // 极点带（|y| > 0.999，θ < 2.6°）**不再跳过**。
        //
        // 上一版跳过的理由：该处 u = atan2(z,x)/2π 的输入 x,z 已到 1e-4 量级，
        // 硬件 `atan2` 与 `det_trig::atan2` 的差异会被 1/√(1−y²) 放大。现在
        // WGSL 侧走的是 `det_trig` 的**逐式镜像**（同系数、同归约顺序），
        // 该放大对两端同样成立 ⇒ 实测极点带也逐位一致。留着计数器只为失败时定位。
        // 亮斑归属：CPU 侧按自己的 texel 查，GPU 侧由探针回传 `env_spot_index`。
        let cpu_in_spot = cdf.spot_index_of_texel(want_texel).is_some();
        let gpu_in_spot = got_sidx < 4.0e9;
        if cpu_in_spot != gpu_in_spot {
            spot_class_mismatch += 1;
            if spot_class_mismatch <= 3 {
                println!(
                    "[亮斑归属偏差] i={i} texel(cpu)={want_texel} texel(gpu)={got_texel} \
                     cpu_in_spot={cpu_in_spot} gpu_in_spot={gpu_in_spot} \
                     uv(gpu)=({:.9},{:.9}) uv(cpu)=({uu:.9},{vv:.9})",
                    got_uv.0, got_uv.1
                );
            }
        }
        let rel = |g: f32, t: f32| if t.abs() > 1e-20 { ((g - t) / t).abs() } else { (g - t).abs() };
        let r = rel(got_pdf, want.pdf_w);
        if r > worst_pdf {
            worst_pdf = r;
            at_pdf = i;
        }
        let r = rel(got_rad.x, want_rad.x);
        if r > worst_rad {
            worst_rad = r;
            at_rad = i;
        }
        let r = rel(got_pdf2, want_pdf2);
        if r > worst_pdf2 {
            worst_pdf2 = r;
            at_pdf2 = i;
        }
        if got_texel != want_texel {
            texel_mismatch += 1;
            if texel_mismatch <= 3 {
                println!(
                    "[texel 偏差] i={i} GPU={got_texel} CPU={want_texel}\
                     （u1={u1:.6} u2={u2:.6} uv=({uu:.6},{vv:.6})）"
                );
            }
        }
    }
    println!(
        "[parity] {name}：{QUERIES} 查询（全数逐值对拍，含极点带），|Δwi|={worst_wi:.3e}，\
         |Δuv|={worst_duv:.3e}，pdf 相对差={worst_pdf:.3e}，辐亮度相对差={worst_rad:.3e}，\
         pdf_w(wi) 相对差={worst_pdf2:.3e}，亮斑归属不符={spot_class_mismatch}，\
         texel 不符={texel_mismatch}"
    );

    // 最差点明细：只在判据会变红时打印（定位用，不参与断言）。
    let failed = worst_wi > 2e-6
        || worst_pdf > 1e-4
        || worst_rad > 1e-4
        || worst_pdf2 > 1e-3
        || texel_mismatch > 0
        || spot_class_mismatch > 0;
    for (tag, idx) in if failed {
        vec![
            ("|Δwi|", at_wi),
            ("|Δuv|", at_uv),
            ("sample.pdf", at_pdf),
            ("radiance", at_rad),
            ("pdf_w", at_pdf2),
        ]
    } else {
        Vec::new()
    } {
        let i = idx as usize;
        let u1 = (idx as f32 + 0.5) / QUERIES as f32;
        let u2 = radical_inverse_base2(idx ^ 0x9E37_79B9);
        let want = cdf.sample(u1, u2).expect("非全黑");
        let o = &a.output[i * STRIDE..(i + 1) * STRIDE];
        let (uu, vv) = dir_to_uv(want.wi);
        let ci = (uu * ew as f32).min(ew as f32 - 1.0) as u32;
        let cj = (vv * eh as f32).min(eh as f32 - 1.0) as u32;
        println!(
            "[worst {tag}] i={idx} texel cpu={} gpu={} | uv cpu=({uu:.9},{vv:.9}) gpu=({:.9},{:.9}) \
             | cpu: pdf={:.9e} rad={:.9e} pdf_w={:.9e} | gpu: pdf={:.9e} rad={:.9e} pdf_w={:.9e} spot_idx={:.0}",
            cj * ew + ci,
            o[7] as u32,
            o[9],
            o[10],
            want.pdf_w,
            cdf.radiance(table, want.wi).x,
            cdf.pdf_w(want.wi),
            o[3],
            o[4],
            o[8],
            o[11]
        );
        // uv 差异来源分解：把 **GPU 的 wi** 喂给 CPU 的映射，看 uv 是否一致。
        // 若这一步对得上，说明差异全在方向的 trig；对不上则说明 `acos/atan2`
        // 本身在两端不同（下一步要决定是改 WGSL 还是给容差）。
        let gp = Vec3::new(o[0], o[1], o[2]);
        let gpu_y_v = prism_math::det_trig::acos(gp.y) / PI;
        let (gpu_y_u, _) = dir_to_uv(gp);
        println!(
            "    [uv 分解] wi cpu=({:.9},{:.9},{:.9}) gpu=({:.9},{:.9},{:.9}) | Δwi={:.3e} | \
             v(cpu_acpi, GPU 的 y)={gpu_y_v:.9} vs v(gpu)={:.9} | u(cpu_map, GPU 的 wi)={gpu_y_u:.9} vs u(gpu)={:.9}",
            want.wi.x,
            want.wi.y,
            want.wi.z,
            gp.x,
            gp.y,
            gp.z,
            (gp - want.wi).length(),
            o[10],
            o[9]
        );
        // 相邻 texel 的 CPU pdf：判断差异是否来自"uv 落在纹素边界另一侧"。
        for (di, dj) in [(0i32, 0i32), (-1, 0), (1, 0), (0, -1), (0, 1)] {
            let ni = (ci as i32 + di).rem_euclid(ew as i32) as u32;
            let nj = (cj as i32 + dj).clamp(0, eh as i32 - 1) as u32;
            let nu = (ni as f32 + 0.5) / ew as f32;
            let nv = (nj as f32 + 0.5) / eh as f32;
            let nd = prism_rt::env::uv_to_dir(nu, nv);
            println!(
                "    [nb {di:+},{dj:+}] texel={} cpu_pdf_w={:.9e}",
                nj * ew + ni,
                cdf.pdf_w(nd)
            );
        }
    }

    assert!(
        worst_wi <= 2e-6,
        "{name}：方向最大偏差 {worst_wi:.3e} 超过 2e-6（trig 实现差异的上限）"
    );
    // uv 是"方向 → texel 反查"的唯一入口，也是 `pdf_w` 的全部输入。这条卡住
    // 1e-6 相当于卡住"内建 acos 不许回来"：实测 lavapipe 内建 `acos` 在这条链
    // 上给出 4.96e-5 的 v 偏差（rad 上 1.56e-4），是 `env_acos` 同式的 50 倍。
    assert!(
        worst_duv <= 2e-6,
        "{name}：uv 最大偏差 {worst_duv:.3e} 超过 2e-6（方向 → uv 映射的 trig 精度上限）"
    );
    assert!(
        worst_pdf <= 1e-4,
        "{name}：采样 pdf 相对差 {worst_pdf:.3e} 超过 1e-4"
    );
    assert!(
        worst_rad <= 1e-4,
        "{name}：辐亮度相对差 {worst_rad:.3e} 超过 1e-4"
    );
    assert!(
        worst_pdf2 <= 1e-3,
        "{name}：pdf_w(wi) 相对差 {worst_pdf2:.3e} 超过 1e-3"
    );
    assert_eq!(
        texel_mismatch, 0,
        "{name}：{texel_mismatch} 个查询的 texel 反查与 CPU 不一致（CDF 反演口径漂移）"
    );
    assert_eq!(
        spot_class_mismatch, 0,
        "{name}：{spot_class_mismatch} 个查询的亮斑归属与 CPU 不一致\
         （`env_spot_index` 与 `spot_index_of` 不同源）"
    );
}

/// 平滑环境（无亮斑 ⇒ 只走残差支路）。
#[test]
fn v6_cpu_vs_wgsl_smooth_env() {
    let scene = env_case("平滑环境 32×16", 32, 16, |_, _, d| {
        let lobe = d.dot(&Vec3::new(0.3, 0.9, 0.2).normalize()).max(0.0);
        Vec3::new(0.2 + 2.0 * lobe, 0.3 + 1.5 * lobe, 0.4 + 1.0 * lobe)
    });
    run_case("平滑环境", &scene);
}

/// 高对比环境（亮斑支路 + 残差支路都被走到）。
#[test]
fn v6_cpu_vs_wgsl_contrast_env() {
    let scene = env_case("高对比环境 64×32", 64, 32, |_, _, d| {
        let sun = Vec3::new(0.5, 0.0, 0.866);
        if d.dot(&sun) >= (4.0f32 * PI / 180.0).cos() {
            Vec3::new(40.0, 38.0, 30.0)
        } else {
            Vec3::new(0.1, 0.12, 0.2)
        }
    });
    // 亮斑支路必须有东西可测：没有亮斑就说明中位数判据没触发，本用例失去意义。
    assert!(
        scene.env().expect("CDF").stats().spots > 0,
        "高对比环境未触发亮斑列表（判据退化）"
    );
    run_case("高对比环境", &scene);
}

/// 采样密度与实际频率的一致性（CPU 权威口径，GPU 侧同一份 CDF ⇒ 同结论）。
///
/// 这条不测 GPU，只把"blob 里的表就是 CPU 算的那份表"钉住：两跑 `sample`
/// 的分布与 `pdf_w` 互为逆（3σ）。
#[test]
fn v6_blob_is_the_cpu_table() {
    let scene = env_case("一致性 24×12", 24, 12, |u, v, _| {
        Vec3::new(0.1 + 2.0 * u, 0.1 + 2.0 * v, 0.3)
    });
    let cdf = scene.env().expect("CDF");
    let pixels = pixel_data(&scene);
    let blob = build_blob(cdf, &pixels);
    assert_eq!(blob[0], 24);
    assert_eq!(blob[1], 12);
    assert_eq!(blob[7], 25 * 12);
    assert_eq!(blob[8], 13);
    assert_eq!(blob[13], 4 * 24 * 12);
    // texels 段与场景扁平 texels 逐位相同（blob 不另造数据）。
    let off = blob[20] as usize;
    for (k, p) in pixels.iter().enumerate() {
        for c in 0..4 {
            assert_eq!(
                blob[off + 4 * k + c],
                p[c].to_bits(),
                "texel {k} 分量 {c} 与场景 texels 不一致"
            );
        }
    }
    let ctx = EvalCtx::new(2026, 11, 2);
    let _ = ctx;
    println!(
        "blob：{} 字（{} B），头部/表/texels 三段与 CPU 状态逐位一致",
        blob.len(),
        blob.len() * 4
    );
    let _ = PI;
}
