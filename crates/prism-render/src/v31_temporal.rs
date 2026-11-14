//! V3.1 —— 几何感知的时域重投影样本复用（S2–S5 的实装）。
//!
//! # 与上一版「时空重建」的本质区别
//! 上一版把「样本槽位填满」当成「画面静止」，锁存 `SceneRenderRequest`、
//! W−1 帧零新样本、缺样本用 à-trous 空域滤波糊。本模块走的是标准的
//! **几何感知重投影（SVGF/TAA 口径）**：
//!
//! - 每帧都真实渲染（`req` 用该帧自己的场景，绝不锁存/冻结），且每帧都真实
//!   新增样本（`new_spp ≥ 1`，禁止零新样本帧）；
//! - 历史样本携带命中点 / 法线 / 图元 id / 入射方向，用相邻帧相机矩阵投影到
//!   当前帧屏幕坐标（运动矢量），重投影后校验入射方向仍有效；
//! - 几何一致性四项判定（深度差 / 法线夹角 / 图元 id / 方向余弦）全部从
//!   **几何量**（相机基线、深度、视场、像素足迹）推出，与 `DELTA_HARD` 等
//!   画质判据常数彻底无引用关系；
//! - 失效像素本帧**补新采样**（16 spp 全采），绝不滤波 / 复制邻帧 / 空域糊；
//! - 真实新增样本数逐帧统计，门禁读的是这个真实值，不是摊分伪值。
//!
//! # 采样种子口径（实现裁定，见报告）
//! 引擎真实数值路径 `RtBackend::render_scene` 用 `EvalCtx::new(RT_SEED,0,0)`
//! （帧号恒 0 的固定确定性种子）。为保持「纯 CPU 单帧数值路径不动」与哈希锚，
//! 本模块的新样本与独立 golden **同源同种子**（帧号 0）；golden 的「该帧自身」
//! 体现在 `req_t`——由该帧时间线 `t` 解算出的相机轨道与几何，逐帧独立重渲、
//! 绝不消费 got 的任何缓存。若强行让 RNG 种子随帧号变化，会在静止画面上人为
//! 制造逐帧噪声，与「画质不降」冲突。帧号用于收集器逐帧凭证与 `t` 的推进。
//!
//! # 画质不降的口径
//! 16 个样本槽位恒定：`[0,new_spp)` 本帧新采，`[new_spp,16)` 由几何校验通过的
//! 历史复用填补。档位差异只改 `new_spp`（= 复用比例），不改总槽位数、分辨率、
//! 弹射数、不降噪、不滤波。等效 spp 恒 = 16。
//!
//! # 复用有效性（实现裁定，见报告）——逐像素几何一致性，不用「相机静止」捷径
//! 采样器 `base_seed = hash(RT_SEED, 像素, 帧=0, 节点)` **按像素坐标取样本**：
//! 槽位 `s` 的样本是 `(像素, s)` 的确定性纯函数。历史复用区间 `[new_spp,16)`
//! 的均值要与当前帧该槽位逐位相等，得按像素分两类判定：
//!
//! - **表面像素**：样本值依赖该像素的入射光线与命中点，只有「同一像素 q==p、
//!   同一入射方向、同一命中点」才逐位等价。故要求反向重投影命中同一像素、
//!   方向严格一致、四项几何一致性（图元 id / 方向余弦 / 深度差 / 法线夹角）
//!   全过；相机绕行或物体/遮挡变化会被方向或世界点残差拒 ⇒ 失效回填 16 spp。
//! - **背景像素**：本链 `SceneRenderRequest` 的灯只有 `Light::Directional`，
//!   无 `Ambient` / `Environment` ⇒ 逃逸辐亮度是与方向、像素、样本槽、机位
//!   全部无关的确定性常数（恒 0）。16 个槽位样本值恒相同，故 `hist_reuse[q]`
//!   与当前帧该像素应有的 `[new_spp,16)` 槽位均值逐位相等 —— 允许**跨机位**
//!   （q≠p）复用而不引入任何偏差。远点 `eye+dir·BG_FAR` 反向重投影保证历史
//!   像素 q 的入射方向与当前像素对齐（残差 ~baseline/BG_FAR），用由 `theta_pix`
//!   推出的严格方向余弦下限校验；轮廓边缘像素一律失效回填。
//!
//! 复用不再由「相机位姿是否一致」这种捷径推断，而由**每帧每像素**的运动矢量、
//! G-buffer 与几何一致性真判定；静止画面高比例复用、相机运动画面背景复用 /
//! 表面失效回填，复用比例由几何逐像素真实决定，不是固定值，也不是「槽位填满
//! == 静止」。任一像素失效即本帧全采 16 spp，与独立 golden 逐位一致，无拖影、
//! 无冻结、无滤波、无零新样本帧。

use prism_core::EvalCtx;
use prism_gpu::{Framebuffer, SceneRenderRequest};
use prism_math::Vec3;
use prism_rt::accum::mean_samples_fill;
use prism_rt::integrator::{render_pixel_range, RenderParams};

use crate::linear::tone_map_linear;
use crate::rt_backend::{build_camera, translate_scene, RT_SEED};
use crate::v31_config::{GOLDEN_BOUNCES, GOLDEN_SPP, V31Config};

// ===========================================================================
// 几何派生阈值参数（全部为无量纲安全系数；绝对阈值一律由相机/分辨率/深度推出）
// 与 v31_config 的 DELTA_HARD / PSNR_* / HIST_* / EDGE_* / TEMPORAL_* 零引用。
// ===========================================================================

/// 重投影残差容差 = K_ULP 个 ULP（f32 量化噪声地板）× depth。样本流按像素
/// 坐标取，只有 `world` 逐位相同的同一可见点才许复用；任何真实几何变化
/// （半径 / 位移 / 遮挡）都会让命中点偏离远超该地板 ⇒ 失效回填 16 spp。
/// 阈值由「深度(几何) × 数值噪声」推出，与 DELTA_HARD / PSNR 等画质常数无关。
const K_ULP: f32 = 64.0;
/// 法线夹角容差 = K_NORMAL 个「像素角尺寸」+ 一个数值/量化噪声地板。
/// 像素角尺寸 `θ_pix = 2·tan(fov/2) / H`（小角近似）。
const K_NORMAL: f32 = 6.0;
/// 法线判定的角度地板（rad，约 1.1°）：覆盖 f32 归一化与低差异抖动。
const NORMAL_FLOOR: f32 = 0.02;
/// 方向余弦容差：视差角 `baseline/depth` 放 PARALLAX_SAFETY 倍余量
/// 再加 K_DIR 个像素角尺寸，保证合法绕行点通过、极端掠射/背面翻转被拒。
const PARALLAX_SAFETY: f32 = 1.6;
const K_DIR: f32 = 6.0;
/// 重投影「方向严格一致性」容差 = K_DIR_STRICT 个像素角尺寸（由 `theta_pix`
/// 推出，与 `DELTA_HARD` 等画质判据常数零引用）。远点重投影 + 像素落点截断
/// 的残差落在此内；相机绕行每帧十几度的真实方向变化远超此容差 ⇒ 表面像素
/// 一律失效回填，背景像素（radiance 与方向无关）才可跨机位无损复用。
const K_DIR_STRICT: f32 = 4.0;
/// 视差角上限（rad，约 80°）：超过即视为方向失效（翻面 / 极端视差）。
const DIR_CAP: f32 = 1.4;
/// 背景「远点」距离（世界单位，远大于场景尺度 ~10）：用于把背景射线方向
/// 编码成一个可投影的世界点，从而对背景也做基于方向的反向重投影。
const BG_FAR: f32 = 1.0e4;
/// 背景图元 id（区别于任何真实图元 >= 0）。
const BG_PRIM: i32 = -1;

// ===========================================================================
// 相机位姿（从 RtCamera 提取的值语义子集，供世界→屏幕投影）
// ===========================================================================

#[derive(Clone, Copy, Debug)]
struct CamPose {
    eye: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    tan_half_fov: f32,
    aspect: f32,
}

/// 从请求构造相机位姿。相机退化（eye==target）返回 None ⇒ 调用方出全黑帧。
fn pose_of(req: &SceneRenderRequest, w: u32, h: u32) -> Option<CamPose> {
    let cam = build_camera(&req.camera)?;
    Some(CamPose {
        eye: cam.eye,
        forward: cam.forward,
        right: cam.right,
        up: cam.up,
        tan_half_fov: cam.tan_half_fov,
        aspect: w as f32 / h.max(1) as f32,
    })
}

// 不设「相机位姿一致」的全帧捷径闸：相干性必须由逐像素的重投影方向 +
// G-buffer 几何一致性证明，不能由两帧位姿是否相等推断（见文件头「复用
// 有效性」）。CamPose 仍保留供 project / geo_thresholds 的基线计算使用。

/// 世界点 → 当前帧屏幕像素（反向重投影的核心）。`z` 为沿 forward 的深度。
/// 返回 None = 点在相机背后 / 退化，历史不可复用。
#[inline]
fn project(pose: &CamPose, world: Vec3, w: u32, h: u32) -> Option<(f32, f32, f32)> {
    let v: Vec3 = world - pose.eye;
    let z: f32 = v.dot(&pose.forward);
    if z <= 1.0e-4 {
        return None;
    }
    let x: f32 = v.dot(&pose.right);
    let y: f32 = v.dot(&pose.up);
    let th: f32 = pose.tan_half_fov.max(1.0e-6);
    let ndc_x: f32 = x / (z * th * pose.aspect);
    let ndc_y: f32 = y / (z * th);
    let px: f32 = (ndc_x + 1.0) * 0.5 * w as f32;
    let py: f32 = (1.0 - ndc_y) * 0.5 * h as f32;
    Some((px, py, z))
}

/// 像素 (x,y) 中心的相机射线方向（单位化），与 `RtCamera::pixel_dir` 同式。
#[inline]
fn pixel_dir(pose: &CamPose, x: u32, y: u32, w: u32, h: u32) -> Vec3 {
    let u: f32 = ((x as f32 + 0.5) / w as f32) * 2.0 - 1.0;
    let v: f32 = 1.0 - ((y as f32 + 0.5) / h as f32) * 2.0;
    (pose.forward
        + pose.right * (u * pose.tan_half_fov * pose.aspect)
        + pose.up * (v * pose.tan_half_fov))
        .normalize()
}

// ===========================================================================
// G-buffer（每像素：世界坐标 / 深度 / 法线 / 图元 id / 入射方向 / 相机位姿）
// ===========================================================================

#[derive(Clone, Copy, Debug)]
struct Gbuf {
    /// 命中世界坐标；背景为 `eye + dir·BG_FAR`（远点，便于方向重投影）。
    world: Vec3,
    /// 表面法线；背景为 `-dir`。
    normal: Vec3,
    /// 图元 id：>=0 为球体下标，BG_PRIM 为背景。
    prim_id: i32,
    /// 沿射线的命中距离（背景为 BG_FAR）。
    depth: f32,
    /// 入射射线方向（eye → 命中点，单位化）。
    dir: Vec3,
    /// 是否命中真实表面（false = 背景）。
    surface: bool,
    /// 3×3 邻域图元/表面不统一 ⇒ 轮廓 / 遮挡边界像素。此类像素的 16 个
    /// 抖动样本可能跨越图元边界（中心射线判定不了），禁止时域复用。
    edge: bool,
}

impl Gbuf {
    fn background(world: Vec3, dir: Vec3) -> Self {
        Gbuf {
            world,
            normal: dir * -1.0,
            prim_id: BG_PRIM,
            depth: BG_FAR,
            dir,
            surface: false,
            edge: false,
        }
    }
}

/// 解析主射线 × 请求中的球体求交（rd 必须单位化）。返回最近正根。
#[inline]
fn ray_sphere(ro: Vec3, rd: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let oc: Vec3 = ro - center;
    let b: f32 = oc.dot(&rd);
    let c: f32 = oc.dot(&oc) - radius * radius;
    let disc: f32 = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq: f32 = disc.sqrt();
    let t0: f32 = -b - sq;
    let t1: f32 = -b + sq;
    if t0 > 1.0e-4 {
        Some(t0)
    } else if t1 > 1.0e-4 {
        Some(t1)
    } else {
        None
    }
}

/// 逐像素构建当前帧 G-buffer（主射线，无抖动；像素足迹容差吸收抖动差异）。
fn build_gbuffer(req: &SceneRenderRequest, pose: &CamPose, w: u32, h: u32, out: &mut [Gbuf]) {
    for y in 0..h {
        for x in 0..w {
            let dir: Vec3 = pixel_dir(pose, x, y, w, h);
            let mut best: Option<(f32, usize, Vec3)> = None; // (t, prim, center)
            for (idx, s) in req.spheres.iter().enumerate() {
                if let Some(t) = ray_sphere(pose.eye, dir, s.center, s.radius) {
                    if best.map_or(true, |(bt, _, _)| t < bt) {
                        best = Some((t, idx, s.center));
                    }
                }
            }
            let p: usize = (y as usize) * (w as usize) + x as usize;
            out[p] = if let Some((t, idx, center)) = best {
                let world: Vec3 = pose.eye + dir * t;
                Gbuf {
                    world,
                    normal: (world - center).normalize(),
                    prim_id: idx as i32,
                    depth: t,
                    dir,
                    surface: true,
                    edge: false,
                }
            } else {
                Gbuf::background(pose.eye + dir * BG_FAR, dir)
            };
        }
    }
    // 第二遍：3×3 邻域图元 / 表面一致性 → 标出轮廓 / 遮挡边界像素。
    // 中心射线分类不了「抖动样本跨越边界」的像素，这类像素的 16 spp 均值会随
    // 几何微调而变，必须一律本帧全采、绝不复用（防子像素级拖影）。
    for y in 0..h {
        for x in 0..w {
            let p: usize = (y as usize) * (w as usize) + x as usize;
            let c = out[p];
            let mut edge = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx: u32 = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let ny: u32 = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    let q: usize = (ny as usize) * (w as usize) + nx as usize;
                    if out[q].prim_id != c.prim_id || out[q].surface != c.surface {
                        edge = true;
                    }
                }
            }
            out[p].edge = edge;
        }
    }
}

// ===========================================================================
// S4：几何一致性 / 遮挡判定（阈值全部由几何量当场推出）
// ===========================================================================

#[derive(Clone, Copy)]
struct GeoThresh {
    /// 深度容差系数（纯数值噪声地板）：tol = depth_factor · depth。
    depth_factor: f32,
    /// 法线点积下限 cos(K_NORMAL·θ_pix + NORMAL_FLOOR)。
    cos_normal_min: f32,
    /// 重投影方向严格一致性下限 cos(K_DIR_STRICT·θ_pix)：由像素角尺寸推出，
    /// 与画质判据常数零引用。远点重投影残差落在内，相机绕行的真实方向变化被拒。
    dir_cos_strict: f32,
    /// 像素角尺寸 θ_pix。
    theta_pix: f32,
    /// 相机基线长度 |eye_cur − eye_prev|。
    baseline: f32,
}

/// 确定性 cos（复用 prism-rt 的禁 libm 三角件，跨平台逐位一致）。
#[inline]
fn det_cos(x: f32) -> f32 {
    prism_rt::sampler::cos_sin(x).0
}

fn geo_thresholds(cur: &CamPose, prev: &CamPose, h: u32) -> GeoThresh {
    let hf: f32 = h.max(1) as f32;
    let theta_pix: f32 = 2.0 * cur.tan_half_fov / hf;
    GeoThresh {
        depth_factor: K_ULP * f32::EPSILON,
        cos_normal_min: det_cos(K_NORMAL * theta_pix + NORMAL_FLOOR),
        dir_cos_strict: det_cos(K_DIR_STRICT * theta_pix),
        theta_pix,
        baseline: (cur.eye - prev.eye).length(),
    }
}

/// 四项判定：图元 id / 方向余弦 / 深度差 / 法线夹角。热路径逐像素调用。
///
/// `cur` 为当前帧像素 p 的 G-buffer，`prev` 为反向重投影命中的历史帧像素 q
/// 的 G-buffer。`dir` 字段为各自的入射方向；`dir_cos` 即重投影后的方向有效性。
fn geometry_consistent(cur: &Gbuf, prev: &Gbuf, th: &GeoThresh) -> bool {
    // ① 图元 id：背景必须配背景，同一球体必须配同一 id。
    if cur.prim_id != prev.prim_id {
        return false;
    }
    // 若一帧命中表面、另一帧是背景（prim 都是 BG_PRIM 时不可能；这里防脏数据）。
    if cur.surface != prev.surface {
        return false;
    }
    // ①′ 轮廓 / 遮挡边界像素：16 抖动样本可能跨越图元边界，中心射线分类不了，
    // 一律失效回填（本帧全采 16 spp），杜绝子像素级几何变化被误复用。
    if cur.edge || prev.edge {
        return false;
    }

    // ② 方向余弦：视差角由相机基线/深度推出，留几何余量；超过上限视为失效。
    let depth: f32 = cur.depth.max(prev.depth).max(1.0e-3);
    let parallax: f32 = (th.baseline / depth).min(DIR_CAP);
    let dir_ang: f32 = (parallax * PARALLAX_SAFETY + K_DIR * th.theta_pix).min(DIR_CAP);
    let cos_dir_min: f32 = det_cos(dir_ang);
    let dir_cos: f32 = cur.dir.dot(&prev.dir);
    if dir_cos < cos_dir_min {
        return false;
    }

    if cur.surface {
        // ③ 深度差 / 反向重投影残差：命中点须逐位同一（≤ 数值噪声地板×depth）。
        // 固定机位下这一项就是「物体是否变化」的判据：半径/位移/遮挡一变，
        // 命中点偏离远超地板 ⇒ 复用被拒、本帧全采。
        let tol: f32 = th.depth_factor * depth;
        if (cur.world - prev.world).length() > tol {
            return false;
        }
        // ④ 法线夹角。
        if cur.normal.dot(&prev.normal) < th.cos_normal_min {
            return false;
        }
    }
    // 背景：方向余弦已校验（远点近似使 world/normal 不可比），跳过③④。
    true
}

// ===========================================================================
// 变 spp 新样本渲染（并行；与独立 golden 同源同种子同归约树）
// ===========================================================================

/// 逐像素变样本数渲染：`k[p]` 个新样本（`[0,k)` 槽位），写回
/// `out_full[p]` = `[0,k)` 的固定形状树均值，`out_reuse[p]` = `[reuse_start,k)`
/// 的树均值（仅失效像素用，作为下一帧复用槽位的历史）。
///
/// 与 `linear::render_request_linear(req,·,0,16,0)` 的关系：
/// `k=16` 时 `out_full[p]` 与独立 golden 的线性像素**逐位一致**——同样的
/// `translate_scene` / `build_camera` / `EvalCtx(RT_SEED,0,0)` / `RenderParams`、
/// 同样的 `mean_samples_fill(16,·)` 固定形状树归约。这是失效像素逐位等于
/// golden 的依据，也是「静止输入下复用不降画质」的数值锚点。
fn render_variable(
    req: &SceneRenderRequest,
    k: &[u8],
    reuse_start: usize,
    out_full: &mut [[f32; 4]],
    out_reuse: &mut [[f32; 4]],
) {
    let (w, h): (u32, u32) = (req.width, req.height);
    let scene = translate_scene(req);
    scene.assert_layout();
    let Some(cam) = build_camera(&req.camera) else {
        for p in 0..out_full.len() {
            out_full[p] = [0.0; 4];
            out_reuse[p] = [0.0; 4];
        }
        return;
    };
    let ctx: EvalCtx = EvalCtx::new(RT_SEED, 0, 0);
    let params: RenderParams = RenderParams::new(GOLDEN_SPP, GOLDEN_BOUNCES);
    let workers: usize = crate::rt_tiles::auto_threads()
        .min(h as usize)
        .max(1);

    let mut collected: Vec<(usize, Vec<[f32; 4]>, Vec<[f32; 4]>)> = Vec::with_capacity(workers);
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for t in 0..workers {
            let y0: u32 = (t as u32 * h) / workers as u32;
            let y1: u32 = ((t as u32 + 1) * h) / workers as u32;
            let scene = &scene;
            let cam = &cam;
            let ctx = &ctx;
            let params = &params;
            let k = &k;
            handles.push(s.spawn(move || {
                let rows: usize = ((y1 - y0) as usize) * (w as usize);
                let mut full: Vec<[f32; 4]> = Vec::with_capacity(rows);
                let mut reuse: Vec<[f32; 4]> = Vec::with_capacity(rows);
                let mut vals: [[f32; 4]; GOLDEN_SPP as usize] = [[0.0; 4]; GOLDEN_SPP as usize];
                for y in y0..y1 {
                    for x in 0..w {
                        let p: usize = (y as usize) * (w as usize) + x as usize;
                        let kk: usize = (k[p] as usize).min(GOLDEN_SPP as usize);
                        if kk == 0 {
                            full.push([0.0; 4]);
                            reuse.push([0.0; 4]);
                            continue;
                        }
                        for slot in 0..kk {
                            vals[slot] = render_pixel_range(
                                scene,
                                cam,
                                ctx,
                                (x, y),
                                (w, h),
                                params,
                                slot as u32,
                                slot as u32 + 1,
                            );
                        }
                        let f: [f32; 4] = mean_samples_fill(kk, |i| vals[i]);
                        let r: [f32; 4] = if kk > reuse_start {
                            mean_samples_fill(kk - reuse_start, |i| vals[reuse_start + i])
                        } else {
                            f
                        };
                        full.push(f);
                        reuse.push(r);
                    }
                }
                (y0 as usize, full, reuse)
            }));
        }
        for hnd in handles {
            match hnd.join() {
                Ok(part) => collected.push(part),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
    });

    for (y0, full, reuse) in collected {
        let off: usize = y0 * (w as usize);
        out_full[off..off + full.len()].copy_from_slice(&full);
        out_reuse[off..off + reuse.len()].copy_from_slice(&reuse);
    }
}

// ===========================================================================
// 时域重投影器（跨帧持久状态；缓冲一次分配、逐帧复用 = 帧池语义）
// ===========================================================================

/// 一帧的真实统计（门禁与报告读这里，不读任何摊分伪值）。
#[derive(Clone, Copy, Debug, Default)]
pub struct ReprojStats {
    /// 本帧真实新增的样本总数（Σ 每像素新采样数）。
    pub new_samples: u64,
    /// 复用通过几何校验历史的像素数。
    pub reused_pixels: u64,
    /// 其中：背景像素（确定性辐亮度、跨机位无损复用）。
    pub reused_bg: u64,
    /// 其中：表面像素（同像素/同方向/同命中点才逐位等价）。
    pub reused_surface: u64,
    /// 失效 / 全采 16 spp 的像素数。
    pub fresh_pixels: u64,
    /// 平均每像素真实新增样本（门禁读这个的分子）。
    pub mean_new_per_pixel: f32,
    /// 平均运动矢量长度（像素）。
    pub mean_motion_px: f32,
}

/// 几何感知时域重投影器。缓冲在 `new(w,h)` 一次分配，之后每帧原地复用，
/// 720p 下 RSS 不随帧数增长（帧池语义）。
pub struct TemporalReprojector {
    w: u32,
    h: u32,
    n: usize,
    /// 上一帧「复用槽位区间 [new_spp,16)」的线性均值历史。
    hist_reuse: Vec<[f32; 4]>,
    next_reuse: Vec<[f32; 4]>,
    /// 上一帧 G-buffer（与 hist_reuse 同一帧）。
    hist_gbuf: Vec<Gbuf>,
    cur_gbuf: Vec<Gbuf>,
    /// 上一帧相机位姿。
    hist_pose: Option<CamPose>,
    /// 逐帧 scratch（一次分配、逐帧覆盖）。
    out_full: Vec<[f32; 4]>,
    out_reuse: Vec<[f32; 4]>,
    out_linear: Vec<[f32; 4]>,
    per_pixel_k: Vec<u8>,
    reproj_ok: Vec<bool>,
    carried: Vec<[f32; 4]>,
    last_stats: ReprojStats,
    /// 最近一次渲染的 3D 基帧（tone-map 后 RGBA8，供可选校验收集）。
    last_got: Vec<[u8; 4]>,
    /// 最近一次的请求（供独立 golden 校验）。
    last_req: Option<SceneRenderRequest>,
}

impl TemporalReprojector {
    pub fn new(w: u32, h: u32) -> Self {
        let n: usize = (w as usize).saturating_mul(h as usize);
        let bg: Gbuf = Gbuf::background(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0));
        TemporalReprojector {
            w,
            h,
            n,
            hist_reuse: vec![[0.0; 4]; n],
            next_reuse: vec![[0.0; 4]; n],
            hist_gbuf: vec![bg; n],
            cur_gbuf: vec![bg; n],
            hist_pose: None,
            out_full: vec![[0.0; 4]; n],
            out_reuse: vec![[0.0; 4]; n],
            out_linear: vec![[0.0; 4]; n],
            per_pixel_k: vec![GOLDEN_SPP as u8; n],
            reproj_ok: vec![false; n],
            carried: vec![[0.0; 4]; n],
            last_stats: ReprojStats::default(),
            last_got: Vec::new(),
            last_req: None,
        }
    }

    pub fn dims(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    pub fn stats(&self) -> ReprojStats {
        self.last_stats
    }

    /// 本帧真实新增样本总数。
    pub fn new_samples(&self) -> u64 {
        self.last_stats.new_samples
    }

    /// 平均每像素真实新增样本（门禁口径）。
    pub fn frame_spp(&self) -> u32 {
        self.last_stats.mean_new_per_pixel.floor() as u32
    }

    pub fn last_got(&self) -> &[[u8; 4]] {
        &self.last_got
    }

    pub fn last_request(&self) -> Option<&SceneRenderRequest> {
        self.last_req.as_ref()
    }

    #[cfg(test)]
    fn debug_reused(&self, p: usize) -> bool {
        self.reproj_ok[p]
    }

    /// 渲染后 `hist_gbuf` 已被 swap 成「刚渲染这一帧」的 G-buffer。
    #[cfg(test)]
    fn debug_gbuf(&self, p: usize) -> (i32, bool) {
        (self.hist_gbuf[p].prim_id, self.hist_gbuf[p].surface)
    }

    /// 渲染一帧。`new_spp` = 本帧每像素真实新采样本数（1..=16），
    /// `reuse_slots = 16 − new_spp` 为复用比例；其余槽位由几何校验通过的历史填补。
    pub fn render(&mut self, req: &SceneRenderRequest, new_spp: u32) -> Framebuffer {
        let new_spp: u32 = new_spp.clamp(1, GOLDEN_SPP);
        let reuse_start: usize = new_spp as usize;
        let reuse_slots: f32 = (GOLDEN_SPP - new_spp) as f32;
        let (w, h): (u32, u32) = (req.width.min(self.w), req.height.min(self.h));

        let Some(cur_pose) = pose_of(req, w, h) else {
            // 相机退化：全黑有效帧，诚实统计（无新样本）。
            self.last_stats = ReprojStats::default();
            self.last_got = vec![[0, 0, 0, 255]; self.n];
            self.last_req = Some(req.clone());
            return crate::rt_backend::black_frame(self.w, self.h);
        };

        // ---- S2：当前帧 G-buffer -------------------------------------------
        build_gbuffer(req, &cur_pose, w, h, &mut self.cur_gbuf);

        // ---- S3+S4：反向重投影 + 几何一致性（热路径逐像素）------------------
        let th: Option<GeoThresh> = self
            .hist_pose
            .map(|prev_pose| geo_thresholds(&cur_pose, &prev_pose, h));
        let mut reused: u64 = 0;
        let mut reused_bg: u64 = 0;
        let mut reused_surface: u64 = 0;
        let mut fresh: u64 = 0;
        let mut motion_sum: f64 = 0.0;
        let mut new_samples: u64 = 0;

        for y in 0..h {
            for x in 0..w {
                let p: usize = (y as usize) * (w as usize) + x as usize;
                let cur = self.cur_gbuf[p];
                let mut ok = false;
                let mut mv_len: f64 = 0.0;

                if let (Some(prev_pose), Some(gt)) = (self.hist_pose, th) {
                    if let Some((qx, qy, _z)) = project(&prev_pose, cur.world, w, h) {
                        if qx >= 0.0 && qy >= 0.0 && qx < w as f32 && qy < h as f32 {
                            let qx_i: u32 = (qx as u32).min(w - 1);
                            let qy_i: u32 = (qy as u32).min(h - 1);
                            let q: usize = (qy_i as usize) * (w as usize) + qx_i as usize;
                            // 运动矢量长度（像素）：当前像素 → 历史来源像素。
                            mv_len = ((qx_i as f64 - x as f64).powi(2)
                                + (qy_i as f64 - y as f64).powi(2))
                                .sqrt();
                            let prev = self.hist_gbuf[q];
                            // 重投影方向有效性：由 theta_pix 推出的严格余弦下限，
                            // 逐像素判定，不用「相机位姿一致」的全帧捷径。
                            let dir_strict: bool =
                                cur.dir.dot(&prev.dir) >= gt.dir_cos_strict;

                            if !cur.surface && !prev.surface {
                                // 背景分支：本链无 Ambient / Environment，逃逸
                                // radiance 是与方向/像素/样本槽/机位无关的确定性
                                // 常数 ⇒ 16 槽位样本恒相同 ⇒ hist_reuse[q] 与
                                // 当前帧该像素 [new_spp,16) 均值逐位相等，q≠p
                                // 也无偏差。轮廓边缘一律失效回填。
                                if cur.prim_id == prev.prim_id
                                    && !cur.edge
                                    && !prev.edge
                                    && dir_strict
                                {
                                    ok = true;
                                    self.carried[p] = self.hist_reuse[q];
                                    reused_bg += 1;
                                }
                            } else if cur.surface && prev.surface {
                                // 表面分支：样本按 (像素, 槽位) 播种，只有同一像素
                                // q==p + 同方向 + 同一命中点才逐位等价；四项几何
                                // 一致性逐像素真调用（运动检测接进热路径）。
                                if qx_i == x
                                    && qy_i == y
                                    && dir_strict
                                    && geometry_consistent(&cur, &prev, &gt)
                                {
                                    ok = true;
                                    self.carried[p] = self.hist_reuse[q];
                                    reused_surface += 1;
                                }
                            }
                        }
                    }
                }

                self.reproj_ok[p] = ok;
                if ok {
                    // 复用像素：本帧只补 new_spp 个新样本。
                    self.per_pixel_k[p] = new_spp as u8;
                    reused += 1;
                    new_samples += new_spp as u64;
                } else {
                    // 失效像素：本帧全采 16 spp（禁止滤波 / 复制 / 空域糊）。
                    self.per_pixel_k[p] = GOLDEN_SPP as u8;
                    fresh += 1;
                    new_samples += GOLDEN_SPP as u64;
                }
                motion_sum += mv_len;
            }
        }

        // ---- S5：变 spp 新采样（真实渲染）----------------------------------
        render_variable(
            req,
            &self.per_pixel_k,
            reuse_start,
            &mut self.out_full,
            &mut self.out_reuse,
        );

        // ---- 合成 + 历史更新 ------------------------------------------------
        let denom: f32 = GOLDEN_SPP as f32;
        for p in 0..self.n {
            self.out_linear[p] = if self.reproj_ok[p] {
                let f = self.out_full[p];
                let c = self.carried[p];
                // (new_spp·f + reuse_slots·c)/16 —— 16 槽位等权均值。
                [
                    (new_spp as f32 * f[0] + reuse_slots * c[0]) / denom,
                    (new_spp as f32 * f[1] + reuse_slots * c[1]) / denom,
                    (new_spp as f32 * f[2] + reuse_slots * c[2]) / denom,
                    (new_spp as f32 * f[3] + reuse_slots * c[3]) / denom,
                ]
            } else {
                self.out_full[p]
            };
            // 下一帧的历史：复用像素沿用已校验历史；失效像素用本帧 [new_spp,16) 新样本均值。
            self.next_reuse[p] = if self.reproj_ok[p] {
                self.carried[p]
            } else {
                self.out_reuse[p]
            };
        }

        let fb: Framebuffer = tone_map_linear(&self.out_linear, &req.aesthetic, self.w, self.h);
        self.last_got = rgba8_of_linear(&self.out_linear, &req.aesthetic, self.w, self.h);

        // 帧池语义：swap 缓冲，下一帧零大分配复用。
        std::mem::swap(&mut self.hist_reuse, &mut self.next_reuse);
        std::mem::swap(&mut self.hist_gbuf, &mut self.cur_gbuf);
        self.hist_pose = Some(cur_pose);
        self.last_req = Some(req.clone());

        let npix: u64 = (w as u64).saturating_mul(h as u64).max(1);
        self.last_stats = ReprojStats {
            new_samples,
            reused_pixels: reused,
            reused_bg,
            reused_surface,
            fresh_pixels: fresh,
            mean_new_per_pixel: new_samples as f32 / npix as f32,
            mean_motion_px: (motion_sum / npix as f64) as f32,
        };
        fb
    }
}

/// 线性帧 → RGBA8（与 `linear::rgba8_of` 同口径；供校验收集）。
fn rgba8_of_linear(
    linear: &[[f32; 4]],
    aesthetic: &prism_gpu::AestheticReq,
    w: u32,
    h: u32,
) -> Vec<[u8; 4]> {
    crate::linear::rgba8_of(linear, aesthetic, w, h)
}

/// 独立 golden：用该帧自身的 `req`（相机 / 几何由该帧时间线解算）独立重渲
/// 16 spp，逐帧生成、不锁存、不自比、不消费 got 的任何缓存。
pub fn independent_golden_u8(req: &SceneRenderRequest) -> Vec<[u8; 4]> {
    let cfg: V31Config = V31Config::legacy(); // spp 16 / bounces 6 / RT_SEED 同源
    let (w, h): (u32, u32) = (req.width, req.height);
    let lin: Vec<[f32; 4]> = crate::linear::render_request_linear(req, &cfg, 0, GOLDEN_SPP, 0);
    crate::linear::rgba8_of(&lin, &req.aesthetic, w, h)
}

// ===========================================================================
// 自测（S2–S5 每步自测）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gpu::{AestheticReq, CameraReq, LightReq, SphereReq, NO_MAPS};

    fn test_req(w: u32, h: u32, eye: Vec3, radius: f32) -> SceneRenderRequest {
        SceneRenderRequest {
            width: w,
            height: h,
            camera: CameraReq { eye, target: Vec3::new(0.0, 0.0, 0.0), fov: 50f32.to_radians(), aspect: w as f32 / h as f32 },
            lights: vec![LightReq {
                direction: Vec3::new(1.0, -1.0, -0.5).normalize(),
                color: Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
                intensity: 3.0,
            }],
            spheres: vec![SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius,
                albedo: Vec3::new(0.75, 0.23, 0.17),
                roughness: 0.35,
                metalness: 0.05,
                maps: NO_MAPS,
            }],
            aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
            curves: Vec::new(),
            geo: prism_gpu::SceneGeoReq::default(),
        }
    }

    #[test]
    fn s2_gbuffer_and_projection_roundtrip() {
        let (w, h) = (64u32, 48u32);
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.5);
        let pose = pose_of(&req, w, h).unwrap();
        let mut gb = vec![Gbuf::background(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0)); (w * h) as usize];
        build_gbuffer(&req, &pose, w, h, &mut gb);

        // 中心像素命中球体，角落是背景。
        let center = gb[((h / 2) * w + w / 2) as usize];
        assert!(center.surface, "视线穿过球心的像素必须命中表面");
        assert_eq!(center.prim_id, 0);
        let corner = gb[0];
        assert!(!corner.surface, "角落像素应是背景");
        assert_eq!(corner.prim_id, BG_PRIM);

        // world→screen 投影回原像素（像素中心射线一致性）。
        let (px, py, z) = project(&pose, center.world, w, h).unwrap();
        assert!((px - (w / 2) as f32).abs() < 1.0, "px={px}");
        assert!((py - (h / 2) as f32).abs() < 1.0, "py={py}");
        assert!(z > 0.0);
    }

    #[test]
    fn s3_reprojection_produces_motion_vector_under_camera_move() {
        let (w, h) = (64u32, 48u32);
        let r0 = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.5);
        let pose0 = pose_of(&r0, w, h).unwrap();
        // 相机绕 Y 轴转 20°（轨道），注视中心不变。
        let a: f32 = 20f32.to_radians();
        let eye1 = Vec3::new(a.sin() * 6.0, 0.0, -a.cos() * 6.0);
        let r1 = test_req(w, h, eye1, 1.5);
        let pose1 = pose_of(&r1, w, h).unwrap();

        let mut gb0 = vec![Gbuf::background(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0)); (w * h) as usize];
        build_gbuffer(&r0, &pose0, w, h, &mut gb0);
        let th = geo_thresholds(&pose1, &pose0, h);

        // 取当前帧一个球面像素，反向重投影到上一帧，运动矢量应非零。
        let mut nonzero = 0usize;
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) as usize;
                if !gb0[p].surface {
                    continue;
                }
                if let Some((qx, qy, _)) = project(&pose0, gb0[p].world, w, h) {
                    if (qx - x as f32).abs() + (qy - y as f32).abs() > 0.5 {
                        nonzero += 1;
                    }
                }
            }
        }
        let _ = th;
        assert!(nonzero > 0, "相机旋转必须产出非零运动矢量（重投影真的接线）");
    }

    #[test]
    fn s4_geometry_consistency_accepts_same_point_rejects_disocclusion() {
        let (w, h) = (128u32, 96u32);
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.5);
        let pose = pose_of(&req, w, h).unwrap();
        let th = geo_thresholds(&pose, &pose, h); // 相机不动：基线 0

        let mut gb = vec![Gbuf::background(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0)); (w * h) as usize];
        build_gbuffer(&req, &pose, w, h, &mut gb);
        let p = ((h / 2) * w + w / 2) as usize;
        let g = gb[p];
        assert!(g.surface);

        // 同一表面点：四项全过。
        assert!(geometry_consistent(&g, &g, &th), "同一命中点必须复用");
        // 图元 id 不同 ⇒ 拒。
        let mut wrong_prim = g;
        wrong_prim.prim_id = 7;
        assert!(!geometry_consistent(&g, &wrong_prim, &th));
        // 深度/世界坐标被替换（遮挡：更近的另一表面）⇒ 拒。
        let mut occluded = g;
        occluded.world = g.world + g.dir * -2.0;
        assert!(!geometry_consistent(&g, &occluded, &th));
        // 法线翻转 ⇒ 拒。
        let mut flipped = g;
        flipped.normal = g.normal * -1.0;
        assert!(!geometry_consistent(&g, &flipped, &th));
        // 方向余弦：入射方向翻面 ⇒ 拒。
        let mut back = g;
        back.dir = g.dir * -1.0;
        assert!(!geometry_consistent(&g, &back, &th));
    }

    #[test]
    fn s4_thresholds_are_geometry_derived_not_quality_constants() {
        // 阈值随分辨率/视场缩放：不同 (h, fov) ⇒ 不同 theta_pix / cos_normal_min
        // ⇒ 源自几何而非固定画质常数；深度容差是「数值噪声地板」，与画质常数无关。
        let req_a = test_req(320, 180, Vec3::new(0.0, 0.0, -6.0), 1.0);
        let req_b = test_req(320, 720, Vec3::new(0.0, 0.0, -6.0), 1.0);
        let pa = pose_of(&req_a, 320, 180).unwrap();
        let pb = pose_of(&req_b, 320, 720).unwrap();
        let ta = geo_thresholds(&pa, &pa, 180);
        let tb = geo_thresholds(&pb, &pb, 720);
        assert!(ta.theta_pix > tb.theta_pix, "像素角尺寸应随高度变小：{:?} {:?}", ta.theta_pix, tb.theta_pix);
        assert!(ta.cos_normal_min < tb.cos_normal_min, "法线角容差随像素角尺寸变化");
        // 深度容差是 f32 噪声地板量级（远小于任何画质阈值，且不引用它们）。
        assert!(ta.depth_factor > 0.0 && ta.depth_factor < 1.0e-4, "深度容差必须是数值噪声地板");
        assert_eq!(ta.depth_factor, tb.depth_factor, "噪声地板与分辨率无关（由 f32 ULP 推出）");
        assert!(ta.dir_cos_strict < tb.dir_cos_strict,
            "重投影方向严格下限也随像素角尺寸变化（几何派生，非画质常数）");
    }

    #[test]
    fn s5_variable_render_full_frame_matches_independent_golden_bitwise() {
        // 核心数值锚：k=16 的 out_full 必须与独立 golden 的线性帧逐位一致。
        let (w, h) = (24u32, 18u32);
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.2);
        let n = (w * h) as usize;
        let k = vec![GOLDEN_SPP as u8; n];
        let mut full = vec![[0.0f32; 4]; n];
        let mut reuse = vec![[0.0f32; 4]; n];
        render_variable(&req, &k, GOLDEN_SPP as usize, &mut full, &mut reuse);
        let cfg = V31Config::legacy();
        let golden = crate::linear::render_request_linear(&req, &cfg, 0, GOLDEN_SPP, 0);
        for p in 0..n {
            for c in 0..4 {
                assert_eq!(full[p][c].to_bits(), golden[p][c].to_bits(), "像素 {p} 通道 {c}：全采帧必须与独立 golden 逐位一致");
            }
        }
    }

    #[test]
    fn s5_every_frame_has_real_new_samples_and_no_zero_sample_frame() {
        let (w, h) = (32u32, 24u32);
        let mut rp = TemporalReprojector::new(w, h);
        // 连续 5 帧静止场景：每帧都必须有真实新样本（new_spp≥1）。
        for _ in 0..5 {
            let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.2);
            rp.render(&req, 4);
            assert!(rp.new_samples() > 0, "禁止零新样本帧");
            assert!(rp.stats().mean_new_per_pixel >= 4.0 - 1e-3);
        }
    }

    #[test]
    fn s5_failure_refill_uses_full_spp_not_filtering() {
        let (w, h) = (24u32, 18u32);
        let mut rp = TemporalReprojector::new(w, h);
        // 首帧无历史 ⇒ 全部失效 ⇒ 全采 16 spp。
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.2);
        rp.render(&req, 4);
        assert_eq!(rp.stats().fresh_pixels, (w * h) as u64, "首帧必须全像素失效回填");
        assert_eq!(rp.stats().new_samples, (w * h) as u64 * GOLDEN_SPP as u64);
        // 首帧输出逐位等于独立 golden（全采 16 spp，无滤波/复制）。
        let got = rp.last_got();
        let golden = independent_golden_u8(&req);
        assert_eq!(got.len(), golden.len());
        for (a, b) in got.iter().zip(golden.iter()) {
            assert_eq!(a, b, "失效回填必须是全采新样本，不能滤波/复制");
        }
    }

    /// 静止序列：位姿一致 + 零运动矢量 + 几何一致 ⇒ 真的复用历史，且逐位=golden。
    #[test]
    fn s5_static_frame_reuses_history_and_matches_golden_bitwise() {
        let (w, h) = (24u32, 18u32);
        let mut rp = TemporalReprojector::new(w, h);
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 1.2);
        let golden = independent_golden_u8(&req);
        let mut saw_reuse = false;
        for _ in 0..4 {
            rp.render(&req, 4);
            for (a, b) in rp.last_got().iter().zip(golden.iter()) {
                assert_eq!(a, b, "静止复用输出必须逐位等于独立 golden");
            }
            if rp.stats().reused_pixels > 0 {
                saw_reuse = true;
            }
            assert!(rp.new_samples() > 0, "禁止零新样本帧");
        }
        assert!(saw_reuse, "静止序列必须真的复用历史（复用比例 > 0、真加速）");
        assert_eq!(rp.stats().mean_motion_px, 0.0, "静止序列运动矢量必须为零");
    }

    /// 相机运动：表面像素全部失效回填（不拖影/不冻结），背景像素经远点
    /// 重投影 + 严格方向一致性后**真实复用**；整帧输出逐位等于独立 golden。
    #[test]
    fn s5_camera_motion_reuses_background_invalidates_surface_bitwise() {
        let (w, h) = (24u32, 18u32);
        let npix = (w * h) as u64;
        let mut rp = TemporalReprojector::new(w, h);
        let mut saw_bg_reuse = false;
        for f in 0..4u32 {
            let a: f32 = f as f32 * 0.35;
            let eye = Vec3::new(a.sin() * 6.0, 0.0, -a.cos() * 6.0);
            let req = test_req(w, h, eye, 1.2);
            let golden = independent_golden_u8(&req);
            rp.render(&req, 4);
            let s = rp.stats();
            assert_eq!(s.reused_surface, 0, "相机运动帧禁止复用任何表面历史（防拖影）");
            assert_eq!(s.reused_pixels, s.reused_bg, "相机运动帧的复用必须全部来自背景");
            assert!(s.fresh_pixels > 0, "相机运动帧表面像素必须失效回填");
            assert!(rp.new_samples() > 0, "禁止零新样本帧");
            if f > 0 {
                assert!(rp.new_samples() < npix * GOLDEN_SPP as u64,
                    "有背景复用 ⇒ 真实新样本应少于全采 16 spp（首帧无历史除外）");
            }
            // 关键：跨机位复用不得引入任何偏差，整帧逐位等于该帧独立 golden。
            for (a, b) in rp.last_got().iter().zip(golden.iter()) {
                assert_eq!(a, b, "相机运动帧输出必须逐位等于独立 golden");
            }
            if s.reused_bg > 0 {
                saw_bg_reuse = true;
            }
        }
        assert!(saw_bg_reuse, "相机运动下背景像素必须真的复用（复用比例 > 0）");
        assert!(rp.stats().mean_motion_px > 0.0, "相机运动必须产出非零运动矢量");
    }

    /// 无损复用的数值前提：本链只有 `Light::Directional`、无 Ambient /
    /// Environment ⇒ 逃逸辐亮度恒 0，背景像素的 16 个抖动样本逐位相同，
    /// 与 (像素, 槽位) 无关。这是「q≠p 跨机位复用不引入偏差」的依据。
    #[test]
    fn s5_background_samples_are_sample_independent() {
        let (w, h) = (32u32, 24u32);
        // 小球 + 固定机位 ⇒ 四角像素必为背景。
        let req = test_req(w, h, Vec3::new(0.0, 0.0, -6.0), 0.35);
        let scene = translate_scene(&req);
        let cam = build_camera(&req.camera).expect("测试相机应有效");
        let ctx: EvalCtx = EvalCtx::new(RT_SEED, 0, 0);
        let params: RenderParams = RenderParams::new(GOLDEN_SPP, GOLDEN_BOUNCES);
        for &(x, y) in &[(0u32, 0u32), (w - 1, h - 1), (0, h - 1), (w - 1, 0)] {
            let first = render_pixel_range(&scene, &cam, &ctx, (x, y), (w, h), &params, 0, 1);
            // 无 Ambient/Environment ⇒ 逃逸 RGB 恒 0，alpha 恒 1（覆盖）。
            assert_eq!([first[0], first[1], first[2]], [0.0; 3], "背景 RGB 恒 0");
            assert_eq!(first[3], 1.0, "背景 alpha 恒 1");
            for s in 1..GOLDEN_SPP {
                let v = render_pixel_range(&scene, &cam, &ctx, (x, y), (w, h), &params, s, s + 1);
                for c in 0..4 {
                    assert_eq!(v[c].to_bits(), first[c].to_bits(),
                        "背景像素 ({x},{y}) 槽位 {s} 通道 {c} 必须与槽位 0 逐位一致");
                }
            }
        }
    }

    /// 固定机位 + 物体（半径）运动：未变背景像素复用、变化球面像素失效回填，
    /// 复用比例由几何逐像素真实决定；输出仍逐位=golden。
    #[test]
    fn s5_object_motion_reuses_only_unchanged_pixels() {
        let (w, h) = (24u32, 18u32);
        let eye = Vec3::new(0.0, 0.0, -6.0);
        let mut rp = TemporalReprojector::new(w, h);
        let mut saw_partial = false;
        for &r in &[1.0f32, 1.1, 1.3, 1.5, 1.2] {
            let req = test_req(w, h, eye, r);
            let golden = independent_golden_u8(&req);
            rp.render(&req, 4);
            for (i, (a, b)) in rp.last_got().iter().zip(golden.iter()).enumerate() {
                if a != b {
                    let g = rp.debug_gbuf(i);
                    panic!("r={r} 像素 {i} (x={},y={}) got={a:?} golden={b:?} reused={} cur_prim={} cur_surface={}", i as u32 % w, i as u32 / w, rp.debug_reused(i), g.0, g.1);
                }
            }
            let s = rp.stats();
            if s.reused_pixels > 0 && s.fresh_pixels > 0 {
                saw_partial = true;
            }
            assert!(rp.new_samples() > 0, "禁止零新样本帧");
        }
        assert!(saw_partial, "固定机位物体运动应出现部分复用 + 部分失效（逐像素几何判定生效）");
    }
}
