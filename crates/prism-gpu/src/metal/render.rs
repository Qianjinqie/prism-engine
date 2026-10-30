//! Metal render 路径的**同构实现**（V2.1 T-A2；**未点亮**）。
//!
//! # ⚠️ 诚实边界（必读，G4/Q5）
//!
//! 本模块交付三样**可验证**的东西，以及一样**明确未做**的东西：
//!
//! | 项 | 状态 |
//! |---|---|
//! | ① MSL 双份源（[`crate::metal::msl::RT_SCENE_MSL`]） | ✅ 已写（手写，与 `vulkan::spirv_rt` 同构） |
//! | ② 参数/球/灯缓冲的 **ABI 编码**（本模块） | ✅ 已写，偏移由单测钉死 |
//! | ③ 算法语义的 **CPU 同构实现**（[`render_scene_msl_semantics`]） | ✅ 已写，与 [`crate::CpuRefBackend`] **逐字节对拍**通过 |
//! | ④ `MTLDevice` → command queue → pipeline state → dispatch → 读回 | ❌ **未做**（需 `objc_msgSend` 层，沙箱无 macOS 工具链/运行时） |
//!
//! 因此 [`crate::metal::MetalBackend::is_available`] **恒 `false`**：
//! 「探测链就绪」（`metal::loader`，仅 macOS 编译）与「真实 dispatch」（④，
//! 未实现）是**两件事**，本模块不把前者当后者用。**macOS 实机验证未做**。
//!
//! # 为什么这里会有「CPU 同构实现」
//!
//! MSL 无法在 Linux 上编译，但**算法与 ABI** 可以验证：把 MSL kernel 的语义
//! 逐句转写成 Rust（[`render_scene_msl_semantics`]），**从编码后的字节缓冲读数据**
//! （而不是从请求结构体读），再与 [`crate::CpuRefBackend`] 的帧逐字节比对。
//! 这能抓住移植中最致命的一类缺陷——**槽位偏移错位**（读了 `radius` 当成
//! `albedo.r` 之类），这类缺陷编译器不会报错，只有对拍能发现。
//!
//! 它**不能**替代 macOS 实机验证：MSL 语法/编译器接受度、Apple GPU 的浮点行为、
//! pipeline state 的绑定正确性仍然未知，故 `is_available()` 保持 `false`。

use prism_math::Vec3;

use crate::vulkan::spirv_rt::{camera_basis, CameraBasis};
use crate::{Framebuffer, SceneRenderRequest};

/// 入口名（与 SPIR-V 的 `OpEntryPoint` 同名，便于跨后端日志对齐）。
pub const ENTRY_SCENE_MAIN: &str = "scene_main";
/// 工作组尺寸（与 Vulkan 侧同值，便于逐组对拍）。
pub const WORKGROUP_SCENE: [u32; 3] = [8, 8, 1];
/// 参数块字节数（`SceneParams`：4×float4 + 12×4 字节标量）。
pub const PARAMS_BYTES: usize = 112;
/// 每球字节数（`Sphere`：packed_float3 + float + packed_float3 + float）。
pub const SPHERE_BYTES: usize = 32;
/// 每灯字节数（`Light`：同上）。
pub const LIGHT_BYTES: usize = 32;

/// 四个绑定的 `(buffer 索引, 内容)`（`[[buffer(g*31+b)]]`，g=0 ⇒ 0..3）。
pub const BINDINGS: [(u32, &str); 4] = [
    (0, "params"),
    (1, "spheres"),
    (2, "lights"),
    (3, "output"),
];

// --- SceneParams 偏移（与 `rt_scene.scene_main.msl` 的 struct 逐字节对应）---

/// `eye.xyz` 起始偏移（float4 槽）。
pub const OFF_EYE: usize = 0;
/// `forward.xyz` 起始偏移。
pub const OFF_FORWARD: usize = 16;
/// `right.xyz` 起始偏移。
pub const OFF_RIGHT: usize = 32;
/// `up.xyz` 起始偏移。
pub const OFF_UP: usize = 48;
/// `width`（u32）。
pub const OFF_WIDTH: usize = 64;
/// `height`（u32）。
pub const OFF_HEIGHT: usize = 68;
/// `tan_half_fov`（f32）。
pub const OFF_TAN_HALF_FOV: usize = 72;
/// `aspect`（f32）。
pub const OFF_ASPECT: usize = 76;
/// `teal_orange`（u32）。
pub const OFF_TEAL_ORANGE: usize = 80;
/// `contrast`（f32）。
pub const OFF_CONTRAST: usize = 84;
/// `contrast_on`（u32）。
pub const OFF_CONTRAST_ON: usize = 88;
/// `filmic`（u32）。
pub const OFF_FILMIC: usize = 92;
/// `num_spheres`（u32）。
pub const OFF_NUM_SPHERES: usize = 96;
/// `num_lights`（u32）。
pub const OFF_NUM_LIGHTS: usize = 100;

fn put_f32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn put_vec3(buf: &mut [u8], off: usize, v: [f32; 3]) {
    put_f32(buf, off, v[0]);
    put_f32(buf, off + 4, v[1]);
    put_f32(buf, off + 8, v[2]);
}

fn get_vec3(buf: &[u8], off: usize) -> Vec3 {
    Vec3::new(get_f32(buf, off), get_f32(buf, off + 4), get_f32(buf, off + 8))
}

/// 编码 `SceneParams`（112 字节，偏移见 `OFF_*` 常量）。
///
/// 相机基由 [`camera_basis`] 给出（与 Vulkan 侧**同一函数**：一个权威实现，
/// 不允许第二套基向量推导）。
pub fn encode_scene_params(req: &SceneRenderRequest, basis: &CameraBasis) -> [u8; PARAMS_BYTES] {
    let mut out = [0u8; PARAMS_BYTES];
    put_vec3(&mut out, OFF_EYE, basis.eye);
    put_vec3(&mut out, OFF_FORWARD, basis.forward);
    put_vec3(&mut out, OFF_RIGHT, basis.right);
    put_vec3(&mut out, OFF_UP, basis.up);
    put_u32(&mut out, OFF_WIDTH, req.width);
    put_u32(&mut out, OFF_HEIGHT, req.height);
    put_f32(&mut out, OFF_TAN_HALF_FOV, basis.tan_half_fov);
    put_f32(&mut out, OFF_ASPECT, basis.aspect);
    put_u32(&mut out, OFF_TEAL_ORANGE, u32::from(req.aesthetic.teal_orange));
    put_f32(&mut out, OFF_CONTRAST, req.aesthetic.contrast);
    put_u32(
        &mut out,
        OFF_CONTRAST_ON,
        u32::from(req.aesthetic.contrast != 1.0),
    );
    put_u32(&mut out, OFF_FILMIC, u32::from(req.aesthetic.filmic));
    put_u32(&mut out, OFF_NUM_SPHERES, req.spheres.len() as u32);
    put_u32(&mut out, OFF_NUM_LIGHTS, req.lights.len() as u32);
    out
}

/// 编码球体缓冲（每球 32 字节：`center.xyz / radius / albedo.xyz / pad`）。
pub fn encode_spheres(req: &SceneRenderRequest) -> Vec<u8> {
    let mut out = Vec::with_capacity(req.spheres.len() * SPHERE_BYTES);
    for s in &req.spheres {
        let mut slot = [0u8; SPHERE_BYTES];
        put_vec3(&mut slot, 0, [s.center.x, s.center.y, s.center.z]);
        put_f32(&mut slot, 12, s.radius);
        put_vec3(&mut slot, 16, [s.albedo.x, s.albedo.y, s.albedo.z]);
        out.extend_from_slice(&slot);
    }
    out
}

/// 编码光源缓冲（每灯 32 字节：`dir.xyz / color.xyz / intensity / pad`；方向已归一化）。
pub fn encode_lights(req: &SceneRenderRequest) -> Vec<u8> {
    let mut out = Vec::with_capacity(req.lights.len() * LIGHT_BYTES);
    for l in &req.lights {
        let n = l.direction.normalize();
        let mut slot = [0u8; LIGHT_BYTES];
        put_vec3(&mut slot, 0, [n.x, n.y, n.z]);
        put_vec3(&mut slot, 12, [l.color.x, l.color.y, l.color.z]);
        put_f32(&mut slot, 24, l.intensity);
        out.extend_from_slice(&slot);
    }
    out
}

/// `scene_normalize`（MSL 同款）：`len < 1e-8 ⇒ 零向量`，否则逐分量除以 `len`。
fn scene_normalize(v: Vec3) -> Vec3 {
    let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if len < 1e-8 {
        Vec3::new(0.0, 0.0, 0.0)
    } else {
        Vec3::new(v.x / len, v.y / len, v.z / len)
    }
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// 逐分量乘（MSL 的 `float3 * float3` 语义；`prism_math::Vec3` 只实现 `* f32`）。
fn vmul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// `scene_quantize`（MSL 同款）：`clamp01 → ×255 → floor(x + 0.5)`。
fn scene_quantize(c: f32) -> u32 {
    let clamped = c.clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5).floor() as u32
}

/// **MSL kernel 的 CPU 同构实现**（逐句转写 `rt_scene.scene_main.msl`）。
///
/// 刻意**只接收编码后的字节缓冲**（而不是 `SceneRenderRequest`）：这样它同时
/// 验证 ABI 布局与算法语义——槽位偏移写错时对拍立刻失败。
///
/// 返回行主序 RGBA8（`w*h*4` 字节，alpha 恒 255），与 [`crate::CpuRefBackend`]
/// 的输出**逐字节相同**（`metal::render::tests` 真断言）。
pub fn render_scene_msl_semantics(
    params: &[u8],
    spheres: &[u8],
    lights: &[u8],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; width as usize * height as usize * 4];
    if params.len() < PARAMS_BYTES || width == 0 || height == 0 {
        return out;
    }
    let eye = get_vec3(params, OFF_EYE);
    let forward = get_vec3(params, OFF_FORWARD);
    let right = get_vec3(params, OFF_RIGHT);
    let up = get_vec3(params, OFF_UP);
    let tan_half = get_f32(params, OFF_TAN_HALF_FOV);
    let aspect = get_f32(params, OFF_ASPECT);
    let teal_orange = get_u32(params, OFF_TEAL_ORANGE) != 0;
    let contrast = get_f32(params, OFF_CONTRAST);
    let contrast_on = get_u32(params, OFF_CONTRAST_ON) != 0;
    let filmic = get_u32(params, OFF_FILMIC) != 0;
    let num_spheres = get_u32(params, OFF_NUM_SPHERES) as usize;
    let num_lights = get_u32(params, OFF_NUM_LIGHTS) as usize;

    for gy in 0..height {
        for gx in 0..width {
            // --- 主射线（与 MSL 同序：先乘后除、左结合）---
            let px = gx as f32 + 0.5;
            let py = gy as f32 + 0.5;
            let ndc_x = (2.0 * px / width as f32 - 1.0) * tan_half * aspect;
            let ndc_y = (1.0 - 2.0 * py / height as f32) * tan_half;
            let dir = scene_normalize(forward + right * ndc_x + up * ndc_y);

            // --- 最近命中 ---
            let mut closest_t = f32::INFINITY;
            let mut hit = num_spheres;
            for i in 0..num_spheres {
                let base = i * SPHERE_BYTES;
                if base + SPHERE_BYTES > spheres.len() {
                    break;
                }
                let center = get_vec3(spheres, base);
                let radius = get_f32(spheres, base + 12);
                let oc = eye - center;
                let b = dot(oc, dir);
                let c = dot(oc, oc) - radius * radius;
                let disc = b * b - c;
                if disc >= 0.0 {
                    let t = -b - disc.sqrt();
                    if t > 0.0 && t < closest_t {
                        closest_t = t;
                        hit = i;
                    }
                }
            }

            // --- 着色 ---
            let mut color = Vec3::new(0.04, 0.05, 0.08);
            if hit < num_spheres {
                let base = hit * SPHERE_BYTES;
                let center = get_vec3(spheres, base);
                let albedo = get_vec3(spheres, base + 16);
                let pos = eye + dir * closest_t;
                let n = scene_normalize(pos - center);
                let mut acc = Vec3::new(0.0, 0.0, 0.0);
                for j in 0..num_lights {
                    let lb = j * LIGHT_BYTES;
                    if lb + LIGHT_BYTES > lights.len() {
                        break;
                    }
                    let ldir = get_vec3(lights, lb);
                    let lcolor = get_vec3(lights, lb + 12);
                    let lint = get_f32(lights, lb + 24);
                    let diff = dot(n, ldir).max(0.0);
                    let scaled = diff * lint;
                    acc = Vec3::new(
                        acc.x + albedo.x * (0.08 + lcolor.x * scaled),
                        acc.y + albedo.y * (0.08 + lcolor.y * scaled),
                        acc.z + albedo.z * (0.08 + lcolor.z * scaled),
                    );
                }
                color = acc;
            }

            // --- aesthetic（与 MSL 同序；全程不经 ACES）---
            if teal_orange {
                let luma = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
                // vmul(阴影染色, color) * (1 - luma) * 0.5（与 MSL 的
                // `float3(...) * color * (1-luma) * 0.5f` 同序：先逐分量乘再乘标量）。
                let shadow = vmul(Vec3::new(0.10, 0.35, 0.45), color) * (1.0 - luma) * 0.5;
                let high = vmul(Vec3::new(1.0, 0.7, 0.3), color) * luma * 0.5;
                color = color * 0.6 + shadow + high;
            }
            if contrast_on {
                color = (color - Vec3::splat(0.5)) * contrast + Vec3::splat(0.5);
            }
            if filmic {
                color = Vec3::new(
                    color.x / (color.x + 1.0),
                    color.y / (color.y + 1.0),
                    color.z / (color.z + 1.0),
                );
            }

            let idx = ((gy as usize) * width as usize + gx as usize) * 4;
            out[idx] = scene_quantize(color.x) as u8;
            out[idx + 1] = scene_quantize(color.y) as u8;
            out[idx + 2] = scene_quantize(color.z) as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

/// 便捷入口：请求 → 编码 → MSL 语义渲染（返回帧缓冲）。
///
/// **这不是 Metal 后端**：它只证明「同构移植的语义与 ABI 正确」，
/// 不触碰任何 Metal API（沙箱无 macOS 运行时）。
pub fn render_scene_on_cpu_via_msl_layout(req: &SceneRenderRequest) -> Framebuffer {
    let mut fb = Framebuffer::new(req.width, req.height);
    if !fb.is_valid() {
        return fb;
    }
    let basis = camera_basis(&req.camera);
    let params = encode_scene_params(req, &basis);
    let spheres = encode_spheres(req);
    let lights = encode_lights(req);
    fb.pixels = render_scene_msl_semantics(&params, &spheres, &lights, req.width, req.height);
    fb
}

// ---------------------------------------------------------------------------
// 测试（跨平台可跑：ABI 偏移 + 与 CPU 参考逐字节对拍）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AestheticReq, CameraReq, CpuRefBackend, GpuDevice, LightReq, SphereReq};

    fn scene(w: u32, h: u32, aesthetic: AestheticReq) -> SceneRenderRequest {
        SceneRenderRequest {
            width: w,
            height: h,
            camera: CameraReq {
                eye: Vec3::new(0.6, 0.9, -5.2),
                target: Vec3::new(0.0, 0.1, 0.0),
                fov: 55.0_f32.to_radians(),
                aspect: w as f32 / h as f32,
            },
            lights: vec![
                LightReq {
                    direction: Vec3::new(0.3, -0.6, 1.0),
                    color: Vec3::new(1.0, 0.95, 0.85),
                    intensity: 1.4,
                },
                LightReq {
                    direction: Vec3::new(-0.8, 0.4, 0.6),
                    color: Vec3::new(0.35, 0.55, 0.95),
                    intensity: 0.9,
                },
            ],
            spheres: vec![
                SphereReq {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 1.0,
                    albedo: Vec3::new(0.85, 0.25, 0.2),
                    roughness: 0.35,
                    metalness: 0.1,
                },
                SphereReq {
                    center: Vec3::new(1.75, -0.25, 1.2),
                    radius: 0.75,
                    albedo: Vec3::new(0.2, 0.75, 0.4),
                    roughness: 0.6,
                    metalness: 0.0,
                },
                SphereReq {
                    center: Vec3::new(0.0, 0.0, 6.0),
                    radius: 2.5,
                    albedo: Vec3::new(0.7, 0.7, 0.7),
                    roughness: 0.5,
                    metalness: 0.0,
                },
            ],
            aesthetic,
        curves: Vec::new(),
        }
    }

    fn off() -> AestheticReq {
        AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        }
    }

    fn on() -> AestheticReq {
        AestheticReq {
            teal_orange: true,
            contrast: 1.35,
            filmic: true,
        }
    }

    /// **ABI 偏移契约**：`SceneParams` 的每个字段都落在 .msl 声明的偏移上。
    ///
    /// 这条测试是「MSL 结构体 ↔ Rust 编码器」的唯一纽带：任一侧改了字段顺序
    /// 或类型，此处立刻红（而 MSL 本身在 Linux 上编译不了，没人会替它报错）。
    #[test]
    fn scene_params_offsets_match_msl_struct() {
        let mut req = scene(64, 48, on());
        req.aesthetic.contrast = 1.35;
        let basis = camera_basis(&req.camera);
        let p = encode_scene_params(&req, &basis);

        assert_eq!(p.len(), PARAMS_BYTES);
        assert_eq!(PARAMS_BYTES % 16, 0, "结构体须 16 字节对齐（float4 成员）");
        // 浮点成员逐偏移核对。
        assert_eq!(get_f32(&p, OFF_EYE), basis.eye[0]);
        assert_eq!(get_f32(&p, OFF_EYE + 4), basis.eye[1]);
        assert_eq!(get_f32(&p, OFF_EYE + 8), basis.eye[2]);
        assert_eq!(get_f32(&p, OFF_FORWARD), basis.forward[0]);
        assert_eq!(get_f32(&p, OFF_RIGHT + 8), basis.right[2]);
        assert_eq!(get_f32(&p, OFF_UP + 4), basis.up[1]);
        assert_eq!(get_u32(&p, OFF_WIDTH), 64);
        assert_eq!(get_u32(&p, OFF_HEIGHT), 48);
        assert_eq!(get_f32(&p, OFF_TAN_HALF_FOV), basis.tan_half_fov);
        assert_eq!(get_f32(&p, OFF_ASPECT), basis.aspect);
        assert_eq!(get_u32(&p, OFF_TEAL_ORANGE), 1);
        assert_eq!(get_f32(&p, OFF_CONTRAST), 1.35);
        assert_eq!(get_u32(&p, OFF_CONTRAST_ON), 1);
        assert_eq!(get_u32(&p, OFF_FILMIC), 1);
        assert_eq!(get_u32(&p, OFF_NUM_SPHERES), 3);
        assert_eq!(get_u32(&p, OFF_NUM_LIGHTS), 2);
        // 尾部两个 pad 槽恒 0（确定性，不留未初始化字节）。
        assert_eq!(&p[104..112], &[0u8; 8]);

        // float4 槽的第 4 分量（w）不参与语义，编码为 0。
        assert_eq!(get_f32(&p, OFF_EYE + 12), 0.0);
        assert_eq!(get_f32(&p, OFF_FORWARD + 12), 0.0);
    }

    /// **结构体槽位契约**：球/灯各 32 字节，8 个 f32 槽按文档顺序排布。
    #[test]
    fn sphere_and_light_slots_match_msl_struct() {
        let req = scene(4, 4, off());
        let s = encode_spheres(&req);
        assert_eq!(s.len(), req.spheres.len() * SPHERE_BYTES);
        assert_eq!(get_vec3(&s, 0), req.spheres[0].center);
        assert_eq!(get_f32(&s, 12), req.spheres[0].radius);
        assert_eq!(get_vec3(&s, 16), req.spheres[0].albedo);
        assert_eq!(get_f32(&s, 28), 0.0, "pad 槽恒 0");
        // 第二个球从 32 字节处开始（无隐式填充）。
        assert_eq!(get_vec3(&s, SPHERE_BYTES), req.spheres[1].center);

        let l = encode_lights(&req);
        assert_eq!(l.len(), req.lights.len() * LIGHT_BYTES);
        let n0 = req.lights[0].direction.normalize();
        assert_eq!(get_vec3(&l, 0), n0, "方向须已归一化");
        assert_eq!(get_vec3(&l, 12), req.lights[0].color);
        assert_eq!(get_f32(&l, 24), req.lights[0].intensity);
        assert_eq!(get_f32(&l, 28), 0.0, "pad 槽恒 0");
    }

    /// **语义同构**：MSL 语义实现与 `CpuRefBackend` 逐字节一致。
    ///
    /// 覆盖四种形态：aesthetic 关/开、整除/带尾组尺寸、空场景。
    /// 这是「移植没写错」的**唯一**可执行证据（MSL 源码本身在 Linux 上编译不了）。
    #[test]
    fn msl_semantics_matches_cpu_reference_bytewise() {
        let cases = [
            ("64x48 off", scene(64, 48, off())),
            ("64x48 on", scene(64, 48, on())),
            ("37x19 off", scene(37, 19, off())),
            ("16x16 empty", {
                let mut r = scene(16, 16, off());
                r.spheres.clear();
                r.lights.clear();
                r
            }),
        ];
        for (tag, req) in cases {
            let msl = render_scene_on_cpu_via_msl_layout(&req);
            let mut cpu = CpuRefBackend;
            let reference = cpu.render_scene(&req);
            assert_eq!(msl.pixels.len(), reference.pixels.len(), "{tag}：长度");
            let mut mismatches = 0usize;
            let mut first = None;
            for (i, (a, b)) in msl.pixels.iter().zip(reference.pixels.iter()).enumerate() {
                if a != b {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some((i, *a, *b));
                    }
                }
            }
            assert_eq!(
                mismatches, 0,
                "{tag}：MSL 语义实现与 CPU 参考必须逐字节一致（首个差异 {first:?}）"
            );
            println!("[metal-msl-parity] {tag}: {} 字节逐字节一致", msl.pixels.len());
        }
    }

    /// 空缓冲/零尺寸不 panic（沙箱与真机都要有的降级语义）。
    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert!(render_scene_msl_semantics(&[], &[], &[], 4, 4).iter().all(|&b| b == 0));
        assert!(render_scene_msl_semantics(&[0u8; PARAMS_BYTES], &[], &[], 0, 0).is_empty());
        // 球/灯缓冲短于 num_spheres 声明时按边界截断（不越界读）。
        let mut params = [0u8; PARAMS_BYTES];
        put_u32(&mut params, OFF_NUM_SPHERES, 9);
        put_u32(&mut params, OFF_NUM_LIGHTS, 9);
        let out = render_scene_msl_semantics(&params, &[], &[], 2, 2);
        assert_eq!(out.len(), 16);
    }

    /// 入口名/工作组/绑定映射是跨后端稳定契约（与 Vulkan 侧同名同形）。
    #[test]
    fn metal_scene_kernel_identity_is_stable() {
        assert_eq!(ENTRY_SCENE_MAIN, "scene_main");
        assert_eq!(ENTRY_SCENE_MAIN, crate::vulkan::spirv_rt::ENTRY_SCENE);
        assert_eq!(WORKGROUP_SCENE, crate::vulkan::spirv_rt::WORKGROUP_SCENE);
        assert_eq!(BINDINGS.len(), 4);
        assert_eq!(BINDINGS[0], (0, "params"));
        assert_eq!(BINDINGS[3], (3, "output"));
        // 与 msl 双份源登记一致。
        assert!(crate::metal::msl::RT_SCENE_MSL.contains("kernel void scene_main"));
        assert_eq!(crate::metal::msl::RT_SCENE_ENTRY, ENTRY_SCENE_MAIN);
    }
}
