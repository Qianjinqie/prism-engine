//! `RtBackend` —— 把 [`GpuDevice`] 的渲染请求接到 `prism-rt` 完整路径追踪。
//!
//! 这是 **D-19 双轨**的默认档：`prism-app render` 默认走本后端，
//! `--fast` 仍走 [`prism_gpu::CpuRefBackend`]（单弹射 Lambert 射线投射）。
//!
//! # 为什么适配器放本 crate（而不是 `prism-rt`）
//!
//! 1. `render_movie` 的签名是 `&mut dyn GpuDevice`（`movie.rs:248`）——做成
//!    `GpuDevice` 实现即可让 `render_movie` **一行不改**，风险最小；
//! 2. `prism-rt` 是**求解器**（`Scene` / `Camera` / `RenderParams`），不该认识
//!    `prism-render` 的场景 DTO（`SceneRenderRequest`）；把 DTO 翻译放在调度层
//!    （本 crate）保持单向依赖：`prism-render → prism-rt` 已存在（曲线路径），
//!    无需新增任何 Cargo 边；
//! 3. `prism-rt` 的 dev-dependency 里有 `prism-render`（tile 关卡对拍），
//!    把适配器写进 `prism-rt` 会让生产代码与 dev 依赖形成人眼难辨的双向边。
//!
//! # 翻译边界（W0-B 最小可用集）
//!
//! | 请求字段 | 处理 |
//! |---|---|
//! | `width` / `height` | 逐字透传（`Framebuffer` 尺寸契约先判，拒绝即返回） |
//! | `camera.eye/target/fov` | → [`prism_rt::camera::Camera`]（针孔、零长快门、fov 转度） |
//! | `camera.aspect` | **不消费**：rt 相机从 `resolution` 自身推导宽高比（`pixel_dir`）。`build_render_request` 恒传 `w/h`，两者一致；此处记录为已知口径差 |
//! | `lights[].direction/color/intensity` | → `Light::Directional { radiance: color × intensity × π, cos_theta_max: 1.0 }`（δ 灯，与 `CpuRefBackend` 的方向光同语义：**无软阴影**；π 是"辐照度式强度"↔"辐亮度式强度"的换算，见 [`RT_INTENSITY_TO_RADIANCE`]） |
//! | `spheres[]` | → `(Sphere, MaterialKind::Pbr(Bsdf))`，`albedo/metalness/roughness` 逐字段映射（与 glTF PbrMaterial 同构） |
//! | `aesthetic` | 与 `CpuRefBackend` **逐式相同**的后处理链（teal-orange → contrast → filmic → clamp01），保证两档唯一差异是着色模型 |
//! | `curves[]` | **拒绝**（返回拒绝缓冲）。理由见 [`RtBackend::render_scene`] 文档 |
//!
//! # 与 `--fast` 的已知、**有意**差异
//!
//! `CpuRefBackend` 在着色里硬编码了 `ambient = Vec3::splat(0.08)`（`prism-gpu/src/lib.rs:539`）
//! ——那不是场景里的光，是兜底渲染器自己编的补光。本后端**不伪造**它：
//!
//! - 逃逸射线看到的是黑（`ambient_background` 在无 `Light::Ambient` 时返回 0）；
//! - 因此背景是纯黑、背光面趋于黑，而 `--fast` 是「深蓝背景 + 0.08 平光」。
//!
//! 这不是接线缺陷：HDR 环境光（`Light::Environment` + IBL）是路线图 **W2** 的 A2 项。
//! 若此处先塞一个常数 ambient，W2 接真环境光时会与它双计。故 W0-B 保留"场景确实
//! 没有环境光"这一事实。
//!
//! # 确定性
//!
//! 样本流只依赖 `(全局种子, 节点 id, 帧号, 像素, 样本序号)`（`integrator.rs:38-40`）。
//! 本后端把前三者**全部钉死为 0**（与 `render_curve_frame` 的 `EvalCtx::new(0, 0, frame)`
//! 同口径），于是：
//!
//! - 同一请求任意次调用**逐字节相同**（跨进程、跨机器，只要 f32 语义一致）；
//! - 噪声图案在屏幕空间**静止**（不随帧闪烁）——对逐帧独立渲染的影片是优点；
//! - 全 crate 无 `thread` / `Mutex` / 哈希表迭代序依赖（`prism-rt` 单线程纯函数）。
//!
//! # 性能口径
//!
//! 逐像素调 [`prism_rt::integrator::render_pixel`]（语义权威 + 与选型报告的
//! 1.15 µs/px/sample 基线同源）。该函数每像素分配两处 `Vec`（样本缓冲 +
//! `mean_samples` 的 `to_vec`，`integrator.rs:2119` / `accum.rs:100`），
//! 在 1080p 上约 400 万次小分配——已知开销，W1 之后按需消掉，本波不动。

use prism_core::EvalCtx;
use prism_gpu::{CameraReq, Framebuffer, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera as RtCamera;
use prism_rt::displace::{tessellate_displace, RuntimeDisplace};
use prism_rt::hair_scatter::{HairGeometry, HairScatter};
use prism_rt::integrator::{
    Aabb as RtAabb, Light, MaterialKind, MediumRegion, RenderParams, Scene as RtScene, Sphere,
};
use prism_rt::instance::Instance;
use prism_rt::media::{HomogeneousMedium, Medium};
use prism_rt::texsample::TexTable;

use crate::rt_tiles::auto_threads;


/// 默认每像素样本数（CLI 默认档）。
pub const RT_DEFAULT_SPP: u32 = 16;

/// 默认最大弹射深度（事件段数上限，相机段不计）。
///
/// 选型报告实测「4 弹射 183 ms vs 6 弹射 191 ms」——RR 让路径早死，
/// 弹射数不是成本主项，故取 6 而不取 4。
pub const RT_DEFAULT_BOUNCES: u32 = 6;

/// 全局种子 / 节点 id / 帧号的固定值（三项全钉死 ⇒ 跨进程逐位可复现）。
pub const RT_SEED: u64 = 0;

/// 方向光的锥半角余弦。
///
/// `1.0` = 纯 δ 方向光，与 `CpuRefBackend` 的 `dot(n, l)` 同语义（硬边界、无软阴影）。
/// 刻意**不**取 0.9 之类的有限锥：那会把「场景里没有的软光源」编进来，
/// 与 `CpuRefBackend` 的对照就失去归因意义（软阴影属 W2 的 IBL/面光范畴）。
pub const RT_DIRECTIONAL_COS_MAX: f32 = 1.0;

/// 灯强度 → 辐亮度的曝光标度（`π`）。
///
/// **这不是拟调参数，是两种"强度"定义之间的换算**：
///
/// - `CpuRefBackend` 的着色是 `albedo × (ambient + color·intensity·(n·l))`
///   —— 没有 BRDF 归一，`color·intensity` 实际是**漫射辐照度式**的量；
/// - `prism-rt` 的 Lambert BRDF 是 `albedo/π`，`Light::Directional.radiance`
///   是**辐亮度式**的量，贡献 = `albedo/π × (n·l) × radiance`。
///
/// 两者在 `n·l = 1` 处相等 ⇒ `radiance = color·intensity·π`。
/// 只有这样，同一份 `.prism` 的两档才是"同曝光、不同着色模型"——
/// 否则"默认档与 `--fast` 不同"里会混进一个纯亮度差，无法归因。
pub const RT_INTENSITY_TO_RADIANCE: f32 = std::f32::consts::PI;

/// 电介质折射率（请求没有 ior 字段；1.5 是玻璃，也是 Disney 缺省电介质口径）。
pub const RT_DIELECTRIC_IOR: f32 = 1.5;

/// 次表面平均自由程占位（请求无此字段；`Bsdf` 要求逐分量为正，故给 0.3）。
pub const RT_SSS_RADIUS: f32 = 0.3;

/// 相机 `forward` 与 world-up 的共线判定阈值（`RtCamera::new` 内部断言阈值为 1e-6）。
const CAMERA_UP_EPS: f32 = 1e-4;

/// 相机 `eye → target` 退化为零向量的判定阈值。
const CAMERA_BASIS_EPS: f32 = 1e-6;

/// `prism-rt` 路径追踪后端（D-19 默认档）。
///
/// 纯 CPU、无 GPU、无外部依赖；[`GpuDevice::is_available`] 恒 `true`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtBackend {
    /// 每像素样本数（≥ 1）。
    pub spp: u32,
    /// 最大弹射深度（≥ 1）。
    pub max_bounces: u32,
    /// 求值线程数（`0` = [`auto_threads`]；其余钳到 ≥ 1）。
    ///
    /// **不进结果**：每像素是独立纯函数，线程数只决定谁在算（见
    /// [`crate::rt_tiles`] 模块文档与 `rt_tiles` 的四条确定性关卡）。
    pub threads: usize,
}

impl Default for RtBackend {
    fn default() -> Self {
        RtBackend {
            spp: RT_DEFAULT_SPP,
            max_bounces: RT_DEFAULT_BOUNCES,
            threads: 0,
        }
    }
}

impl RtBackend {
    /// 以 `(spp, max_bounces)` 构造；两者均钳到 ≥ 1（`0` 样本无定义）。
    ///
    /// 线程数取 [`auto_threads`]（`available_parallelism`）。
    pub fn new(spp: u32, max_bounces: u32) -> Self {
        RtBackend {
            spp: spp.max(1),
            max_bounces: max_bounces.max(1),
            threads: 0,
        }
    }

    /// 指定线程数（`0` = 自动；`1` = 单线程参考路径）。
    ///
    /// 存在的唯一理由是**验证**：1 / 4 / 16 线程必须逐字节相同，否则这条
    /// 并行路径不成立（D-20）。生产调用不设它即取自动值。
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// 实际生效的线程数（`0` → [`auto_threads`]，其余钳到 ≥ 1）。
    pub fn effective_threads(&self) -> usize {
        if self.threads == 0 {
            auto_threads()
        } else {
            self.threads.max(1)
        }
    }

    /// 单帧渲染的样本总量口径（诊断 / 日志用）。
    pub fn samples_per_pixel(&self) -> u32 {
        self.spp
    }
}

impl GpuDevice for RtBackend {
    /// 渲染一帧：`SceneRenderRequest` → `prism-rt` 场景 → 逐像素路径追踪 → RGBA8。
    ///
    /// # 三种非渲染情形（一律不 panic、不静默降级）
    ///
    /// 1. **尺寸被拒** → 透传那个**无效缓冲**（`is_valid() == false`，与
    ///    `CpuRefBackend` 同处理；调用方按契约回退）；
    /// 2. **相机退化** → 全黑**有效**帧（`eye == target`，如 DSL `orbit { radius 0 }`）。
    ///    rt 相机对零基线是 `assert`（宪章 C18 禁止），故前置拦截。
    ///    注意 `CpuRefBackend` 在此情形输出"纯背景色"（NaN 方向无命中，不留 NaN），
    ///    两者都代表"没有画面"；本后端选全黑，不伪造背景色；
    /// 3. **请求含曲线** → 返回 [`Framebuffer::empty`]（拒绝服务）。两条理由——
    ///    (a) 产品路径上不可能出现：`render_scene_frame` 对有曲线的场景直接走
    ///    `render_curve_frame`（已是 prism-rt），`build_render_request` 恒填空集；
    ///    (b) `CurveReq` 带任意 `axis`，而 rt 的圆柱/圆锥/圆环/旋转面**只有 +Y 轴**
    ///    实例，无旋转变换 ⇒ 非 +Y 轴向的曲线无法忠实翻译。
    ///    与其静默丢几何（少画东西还宣称成功），不如明确拒绝。
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer {
        // 只用来判「尺寸是否被拒」；真正的帧由 `rt_tiles::render_frame` 产出。
        let fb: Framebuffer = Framebuffer::new(req.width, req.height);
        // 尺寸被拒 ⇒ 没有像素可写。继续按 req.width/height 循环会在超大请求上
        // 空转（u32::MAX² 迭代），立即返回。
        if !fb.is_valid() {
            return fb;
        }
        // 曲线：无法忠实翻译 ⇒ 拒绝（见 trait 方法文档第 3 条）。
        if !req.curves.is_empty() {
            return Framebuffer::empty();
        }
        // 相机退化 ⇒ 全黑有效帧（见 trait 方法文档第 2 条）。
        let Some(cam) = build_camera(&req.camera) else {
            return black_frame(req.width, req.height);
        };

        let scene: RtScene = translate_scene(req);
        // W1-A 定桩断言：W1-B/W2/W3/W6 起场景会带 textures / instances /
        // media，各自的句柄与互斥约束在这里一次性验掉（每帧一次，纯读，
        // 不进 per-sample 路径）。既有场景全为空 ⇒ 恒通过，行为零变化。
        scene.assert_layout();
        let params: RenderParams = RenderParams::new(self.spp, self.max_bounces);
        // 样本流三要素全钉死（RT_SEED / 0 / 0）⇒ 跨进程逐位可复现。
        let ctx: EvalCtx = EvalCtx::new(RT_SEED, 0, 0);
        let resolution: (u32, u32) = (req.width, req.height);

        // tile 化并行求值：固定 32×32 tile、每 tile 独立算完后按索引升序合并。
        // 线程数只影响墙钟，不进结果（见 `crate::rt_tiles` 模块文档）。
        crate::rt_tiles::render_frame(
            &scene,
            &cam,
            &ctx,
            resolution,
            &params,
            &req.aesthetic,
            self.effective_threads(),
        )
    }

    /// 纯软件路径追踪：无外部依赖 ⇒ 恒可用。
    fn is_available(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "rt-path-tracing"
    }
}

// ---------------------------------------------------------------------------
// 请求 → prism-rt 场景翻译
// ---------------------------------------------------------------------------

/// 把请求翻译成 `prism-rt` 场景（球 / 灯 / **以及 T0 的六项附加能力**）。
///
/// # 六条非空翻译分支（判据 P3 查的就是它们在场）
///
/// | # | 能力 | 请求字段 | rt 落点 |
/// |---|---|---|---|
/// | 1 | 介质区域 | `geo.media` | `Scene::media`（`MediumRegion` + `priority`） |
/// | 2 | 纹理表 | `geo.textures` | `Scene::textures` + `Scene::texels`（`build_rgba8`） |
/// | 3 | 三角网格 | `geo.meshes` | `Scene::add_mesh`（+ 运行期位移） |
/// | 4 | 实例原型 | `geo.instance_geoms` | `Scene::instance_geoms` |
/// | 5 | 实例 | `geo.instances` | `Scene::instances`（`Instance::from_affine`）+ `rebuild_instance_bvh` |
/// | 6 | 毛发股 | `geo.strands` | 每股一个 `Mesh` + `MaterialKind::Hair`（每股自己的标架） |
///
/// # 顺序不是随意的
///
/// 纹理必须在网格**之前**建好：位移槽要在采样器里查纹素。故本函数分两段——
/// 先把纹理构造成**局部** `refs`/`texels`（不借 `scene`），用它建 `TexTable`
/// 烘焙所有网格，最后再把纹理搬进 `scene`（避开"借用 scene 又要写 scene"）。
///
/// # 空 ⇒ 逐字节同帧
///
/// `geo.is_empty()` 时六段循环一次都不执行，`scene` 与加字段前逐字段相同；
/// 末尾 `if !refs.is_empty()` 的搬运也不发生。判据 P1（md5 对账）。
///
/// # 非法输入
///
/// 沿用既有风格：**显式跳过**（`continue`），绝不 panic、绝不塞垃圾进 rt
/// （rt 侧的 `Bsdf::new` / `HomogeneousMedium::new` / `RtCamera::new` 都带断言）。
fn translate_scene(req: &SceneRenderRequest) -> RtScene {
    let mut scene: RtScene = RtScene::default();
    for s in &req.spheres {
        if !(s.radius.is_finite() && s.radius > 0.0) {
            // 半径非正 / 非有限：rt 球求交对它永不命中，但显式跳过更省时，
            // 也与 DSL 的 `radius 0`（"软光追对 r=0 无命中"）语义一致。
            continue;
        }
        if !is_finite_vec(s.center) {
            continue;
        }
        let material: MaterialKind =
            MaterialKind::Pbr(Bsdf::new(
                clamp01(s.albedo),
                s.metalness.clamp(0.0, 1.0),
                s.roughness.clamp(0.0, 1.0),
                RT_DIELECTRIC_IOR,
                0.0,
                0.0,
                Vec3::splat(RT_SSS_RADIUS),
            ));
        scene.spheres.push((
            Sphere { center: s.center, radius: s.radius },
            material,
        ));
    }
    for l in &req.lights {
        let dir_len: f32 = l.direction.length();
        if !(dir_len.is_finite() && dir_len > 0.0) || !is_finite_vec(l.color) {
            continue;
        }
        if !(l.intensity.is_finite() && l.intensity > 0.0) {
            // 非正 / 非有限强度：物理上无意义（会产出负辐亮度再被 clamp 掉），
            // 显式跳过而不是把它当"0 强度灯"塞进 NEE 循环。
            continue;
        }
        scene.lights.push(Light::Directional {
            direction: l.direction * (1.0 / dir_len),
            radiance: l.color * (l.intensity * RT_INTENSITY_TO_RADIANCE),
            cos_theta_max: RT_DIRECTIONAL_COS_MAX,
        });
    }

    // ---- 分支 1/6：纹理表（必须在网格之前：位移槽要采样它）-----------------
    //
    // 逐张按 `build_rgba8`（**同一份** mip 链 / sRGB LUT 实现）构造，
    // offset 用运行游标——与 `Scene::add_texture_rgba8` 的排布逐字节等价。
    let mut refs: Vec<prism_rt::texture::TextureRef> = Vec::new();
    let mut texels: Vec<[f32; 4]> = Vec::new();
    for t in &req.geo.textures {
        let need: u64 = (t.width as u64) * (t.height as u64) * 4;
        if t.width == 0 || t.height == 0 || need != t.rgba8.len() as u64 {
            // 尺寸与像素流不自洽：显式跳过（不 panic、不产生越界读）。
            continue;
        }
        let offset: u32 = texels.len() as u32;
        let (r, px) = prism_rt::texture::build_rgba8(t.width, t.height, t.srgb, offset, &t.rgba8);
        refs.push(r);
        texels.extend_from_slice(&px);
    }

    // ---- 分支 2/6：介质区域（A6）------------------------------------------
    for m in &req.geo.media {
        if !is_finite_vec(m.bounds_min) || !is_finite_vec(m.bounds_max) {
            continue;
        }
        if m.bounds_min.x > m.bounds_max.x
            || m.bounds_min.y > m.bounds_max.y
            || m.bounds_min.z > m.bounds_max.z
        {
            continue; // 空/反向盒：rt 侧没有"反向区间"语义，跳过
        }
        if !(m.sigma_t.is_finite() && m.sigma_t > 0.0) {
            continue; // `HomogeneousMedium::new` 对 σt ≤ 0 是断言
        }
        if !(m.albedo.is_finite() && (0.0..=1.0).contains(&m.albedo)) {
            continue;
        }
        if !(m.g.is_finite() && m.g > -1.0 && m.g < 1.0) {
            continue;
        }
        let bounds: RtAabb = RtAabb::new(m.bounds_min, m.bounds_max);
        let medium = Medium::Homogeneous(HomogeneousMedium::new(m.sigma_t, m.albedo, m.g));
        scene.media.push(MediumRegion::new(bounds, medium).with_priority(m.priority));
    }

    // 位移要用纹理：先把视图建好（只读借用局部 `refs`/`texels`）。
    let tex_view: TexTable<'_> = TexTable::new(&refs, &texels);

    // ---- 分支 3/6：三角网格（+ G-10 运行期位移）---------------------------
    for m in &req.geo.meshes {
        if m.indices.len() < 3 {
            continue;
        }
        if !m
            .positions
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            continue;
        }
        // G-9 折痕/角点标记：空标记 ⇒ `None`（与加标记前逐位相同）。
        let marks: Option<prism_geometry::subdiv::SubdivMarks> =
            if m.crease.is_empty() && m.corners.is_empty() {
                None
            } else {
                let mut sm = prism_geometry::subdiv::SubdivMarks::new();
                for &(a, b, w) in &m.crease {
                    sm = sm.with_crease(a, b, w);
                }
                for &v in &m.corners {
                    sm = sm.with_corner(v);
                }
                Some(sm)
            };
        // 运行期位移：`amplitude == 0` / 无贴图句柄 ⇒ `tessellate_displace`
        // 返回 `None` ⇒ 走**原网格**（判据 V1：与没有位移槽逐位相同）。
        let displaced = m.displacement.and_then(|d| {
            tessellate_displace(
                &m.positions,
                &m.indices,
                &m.uv,
                RuntimeDisplace {
                    texture: d.texture,
                    amplitude: d.amplitude,
                    scale: d.scale,
                    levels: d.levels,
                },
                tex_view,
                marks.as_ref(),
            )
        });
        let material: MaterialKind = MaterialKind::Pbr(Bsdf::new(
            default_mesh_albedo(),
            0.0,
            0.6,
            RT_DIELECTRIC_IOR,
            0.0,
            0.0,
            Vec3::splat(RT_SSS_RADIUS),
        ));
        match displaced {
            Some(d) => {
                let mut mesh = prism_rt::integrator::Mesh::new(d.positions, d.indices);
                mesh.uv = d.uv;
                scene.meshes.push((mesh, material));
            }
            None => {
                let mut mesh =
                    prism_rt::integrator::Mesh::new(m.positions.clone(), m.indices.clone());
                // uv 长度不符 ⇒ 按"无 uv"处理（`Mesh::uv` 的契约），不 panic。
                if m.uv.len() == m.positions.len() {
                    mesh.uv = m.uv.clone();
                }
                scene.meshes.push((mesh, material));
            }
        }
    }

    // ---- 分支 4/6：实例原型（共享几何；**不**直接进场景）-------------------
    for m in &req.geo.instance_geoms {
        if m.indices.len() < 3 {
            continue;
        }
        if !m
            .positions
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            continue;
        }
        let material: MaterialKind = MaterialKind::Pbr(Bsdf::new(
            default_mesh_albedo(),
            0.0,
            0.6,
            RT_DIELECTRIC_IOR,
            0.0,
            0.0,
            Vec3::splat(RT_SSS_RADIUS),
        ));
        let mut mesh = prism_rt::integrator::Mesh::new(m.positions.clone(), m.indices.clone());
        if m.uv.len() == m.positions.len() {
            mesh.uv = m.uv.clone();
        }
        scene.instance_geoms.push((mesh, material));
    }

    // ---- 分支 5/6：实例（世界仿射 + 逆由 rt 侧算一次）----------------------
    for inst in &req.geo.instances {
        if !inst.affine.iter().all(|v| v.is_finite()) {
            continue;
        }
        // `geom_id` 越界 ⇒ 该实例的 TLAS 叶子是空盒（永不命中）；rt 侧的
        // `validate_layout` 会把这种"实例指向不存在的几何"当错误报出来，
        // 故这里**跳过**而不是塞进去变成一个静默空叶子。
        if inst.geom_id as usize >= scene.instance_geoms.len() {
            continue;
        }
        if let Some(instance) = Instance::from_affine(inst.affine, inst.geom_id, inst.mat_id) {
            scene.instances.push(instance);
        }
    }
    if !scene.instances.is_empty() {
        scene.rebuild_instance_bvh();
    }

    // ---- 分支 6/6：毛发股（**每股一个 Mesh + 该股自己的标架**）-------------
    //
    // T3 的修法：`MaterialKind::Hair` 是一材质一标架，于是"同 Mesh 内多股异向"
    // 只有一股精确。这里**按股切 Mesh**——每股一个 `HairGeometry`，标架取该股
    // 的首段（`strand_frames` 的平行传输标架），于是异向股束里每一股都精确。
    // `hair_scatter.rs` 与 `surface_brdf` 一行不改（命中路径仍是一个 Mesh 一个
    // 材质）。理由与备选方案对比见 `prism_core::attach::StrandDesc` 的文档。
    for st in &req.geo.strands {
        if st.points.len() < 2 {
            continue;
        }
        if !st
            .points
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            continue;
        }
        let radii: Vec<f32> = if st.radii.len() == st.points.len() {
            st.radii.clone()
        } else {
            vec![DEFAULT_STRAND_RADIUS; st.points.len()]
        };
        if !radii.iter().all(|r| r.is_finite() && *r > 0.0) {
            continue;
        }
        let curve = prism_geometry::curve_tess::CurveReq::Strand {
            points: st.points.clone(),
            radii,
        };
        let tris = prism_geometry::curve_tess::tessellate(&curve, STRAND_TOLERANCE);
        if tris.is_empty() {
            continue;
        }
        // 该股自己的标架：平行传输的首段（切向 + 法平面参考）。
        let frames = prism_geometry::curve_tess::strand_frames(&st.points);
        let geometry: HairGeometry = match frames.first() {
            Some(f) => HairGeometry::new(f.tangent, f.normal_ref),
            None => continue,
        };
        // 三瓣参数：先走 `HairScatter::new`（参数域断言在此生效），再覆盖权重。
        if !(st.params.beta_m > 0.0 && st.params.beta_m <= 1.0)
            || !(st.params.k_azimuth >= 0.0)
            || !(st.params.eta >= 1.0)
            || !(st.params.sigma_a >= 0.0)
            || !(st.params.radius > 0.0)
        {
            continue;
        }
        let mut scatter = HairScatter::new(
            st.params.beta_m,
            st.params.k_azimuth,
            st.params.eta,
            st.params.sigma_a,
            st.params.radius,
        );
        scatter.w_r = st.params.w_r;
        scatter.w_tt = st.params.w_tt;
        scatter.w_trt = st.params.w_trt;
        scatter.phi_tt_shift = st.params.phi_tt_shift;
        scene.meshes.push((
            prism_rt::integrator::Mesh::from_triangles(tris),
            MaterialKind::Hair { scatter, geometry },
        ));
    }

    // 纹理最后搬进场景（避免与上面的只读借用冲突）。
    if !refs.is_empty() {
        scene.textures = refs;
        scene.texels = texels;
    }
    scene
}

/// 场景网格的固定反照率（请求侧的 `MeshReq` 不带材质；与 `CURVE_ALBEDO` 同口径
/// 的中性灰，保证"网格进得来、颜色可对拍"）。
fn default_mesh_albedo() -> Vec3 {
    Vec3::new(0.72, 0.74, 0.78)
}

/// 毛股三角化的弦高容差（与 `curve_tess` 的既有缺省同量级）。
const STRAND_TOLERANCE: f32 = 0.02;

/// 毛股半径缺省（请求没给逐点半径时）。
const DEFAULT_STRAND_RADIUS: f32 = 0.02;

/// 请求相机 → `prism-rt` 透视相机（针孔 / 零长快门 / fov 转度）。
///
/// 返回 `None` 表示相机几何退化（`eye == target`），调用方按"无画面"处理。
fn build_camera(cam: &CameraReq) -> Option<RtCamera> {
    if !is_finite_vec(cam.eye) || !is_finite_vec(cam.target) {
        return None;
    }
    let to_target: Vec3 = cam.target - cam.eye;
    let distance: f32 = to_target.length();
    if !(distance.is_finite() && distance > CAMERA_BASIS_EPS) {
        return None;
    }
    let forward: Vec3 = to_target * (1.0 / distance);
    // world-up：优先 +Y；与 forward 共线时退到 +Z（`RtCamera::new` 对共线是 assert，
    // 而 GpuDevice 契约禁止 panic）。
    let world_up_y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    let world_up: Vec3 = if forward.cross(&world_up_y).length() > CAMERA_UP_EPS {
        world_up_y
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    // fov：请求是弧度（运行时契约），rt 相机吃度（C17 行业豁免）；
    // 非有限回落 45°，并钳到 (0,180) 开区间内（`RtCamera::new` 断言）。
    let fov_deg: f32 = if cam.fov.is_finite() {
        cam.fov.to_degrees().clamp(1.0, 179.0)
    } else {
        45.0
    };
    Some(RtCamera::new(
        cam.eye,
        cam.target,
        world_up,
        fov_deg,
        0.0,                  // 光圈半径 = 0 ⇒ 针孔（请求无 DOF 字段）
        distance.max(1e-3),   // 对焦距离 = 机距（针孔下不参与成像，仅满足正性约束）
        0.0,                  // 快门开
        0.0,                  // 快门闭（零长快门 ⇒ 无运动模糊，请求无此字段）
    ))
}

/// 全黑**有效**帧（相机退化时的兜底：不透明确 0，alpha 由 `set_pixel` 置 255）。
fn black_frame(width: u32, height: u32) -> Framebuffer {
    let mut fb: Framebuffer = Framebuffer::new(width, height);
    if !fb.is_valid() {
        return fb;
    }
    for y in 0..height {
        for x in 0..width {
            fb.set_pixel(x, y, Vec3::splat(0.0));
        }
    }
    fb
}

// ---------------------------------------------------------------------------
// 私有数学助手（`prism-math` 的锁定契约里没有逐分量 clamp）
//
// 后处理链不在这里：它是 `prism_gpu::apply_aesthetic`——两档渲染器共用同一份实现，
// 避免"两档差异"里混进后期差异（见该函数文档）。
// ---------------------------------------------------------------------------

/// 逐分量钳到 `0.0..=1.0`。
#[inline]
fn clamp01(c: Vec3) -> Vec3 {
    Vec3::new(
        c.x.clamp(0.0, 1.0),
        c.y.clamp(0.0, 1.0),
        c.z.clamp(0.0, 1.0),
    )
}

/// 三个分量是否都有限。
#[inline]
fn is_finite_vec(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

// ---------------------------------------------------------------------------
// 单元测试（`cargo test -p prism-render --lib rt_backend`）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gpu::{apply_aesthetic, AestheticReq, CameraReq, LightReq, SphereReq};

    /// 与 `prism-gpu` 的 `red_sphere_request` 同构的红球请求（关后期）。
    fn red_sphere_request(w: u32, h: u32) -> SceneRenderRequest {
        SceneRenderRequest {
            width: w,
            height: h,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: w as f32 / h as f32,
            },
            lights: vec![LightReq {
                direction: Vec3::new(0.0, 0.0, -1.0),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 3.0,
            }],
            spheres: vec![SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                albedo: Vec3::new(0.8, 0.2, 0.2),
                roughness: 0.35,
                metalness: 0.05,
                maps: prism_gpu::NO_MAPS,
            }],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
            curves: Vec::new(),
    ..Default::default()
}
    }

    /// 两跑逐字节一致（确定性铁律：同输入必然同位模式）。
    #[test]
    fn two_runs_are_byte_identical() {
        let req: SceneRenderRequest = red_sphere_request(24, 14);
        let mut dev: RtBackend = RtBackend::new(4, 3);
        let a: Framebuffer = dev.render_scene(&req);
        let b: Framebuffer = dev.render_scene(&req);
        assert!(a.is_valid() && b.is_valid(), "两跑都必须是有效帧");
        assert_eq!(a.pixels.len(), 24 * 14 * 4);
        assert_eq!(a.pixels, b.pixels, "同请求两跑必须逐字节相同");
    }

    /// 尺寸契约：越过上限的请求被拒（返回空缓冲，不 panic）。
    #[test]
    fn oversize_request_is_refused() {
        let mut req: SceneRenderRequest = red_sphere_request(8, 8);
        req.width = prism_gpu::MAX_FRAMEBUFFER_DIM + 1;
        let mut dev: RtBackend = RtBackend::default();
        let fb: Framebuffer = dev.render_scene(&req);
        assert!(!fb.is_valid(), "超限尺寸必须返回拒绝缓冲");
    }

    /// 曲线请求被明确拒绝（不静默丢几何）。
    #[test]
    fn curves_are_refused_not_dropped() {
        let mut req: SceneRenderRequest = red_sphere_request(8, 8);
        req.curves.push(prism_gpu::CurveReq::Cylinder {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.4,
            height: 1.0,
            axis: Vec3::new(0.0, 1.0, 0.0),
        });
        let mut dev: RtBackend = RtBackend::default();
        assert!(
            !dev.render_scene(&req).is_valid(),
            "含曲线的请求必须拒绝（rt 曲线图元只有 +Y 轴，无法忠实翻译任意 axis）"
        );
    }

    /// 相机退化（eye == target）：不 panic，返回全黑有效帧。
    #[test]
    fn degenerate_camera_is_black_not_panic() {
        let mut req: SceneRenderRequest = red_sphere_request(6, 4);
        req.camera.eye = Vec3::new(0.0, 0.0, 0.0);
        req.camera.target = Vec3::new(0.0, 0.0, 0.0);
        let mut dev: RtBackend = RtBackend::default();
        let fb: Framebuffer = dev.render_scene(&req);
        assert!(fb.is_valid(), "退化相机不得返回拒绝缓冲（下游按有效帧合成）");
        assert!(fb.pixels.iter().step_by(4).all(|&r| r == 0), "退化相机应全黑");
    }

    /// 相机与 +Y 共线（俯视）：world-up 自动退到 +Z，不触发 rt 相机断言。
    #[test]
    fn top_down_camera_does_not_panic() {
        let mut req: SceneRenderRequest = red_sphere_request(6, 4);
        req.camera.eye = Vec3::new(0.0, 4.0, 0.0);
        req.camera.target = Vec3::new(0.0, 0.0, 0.0);
        let mut dev: RtBackend = RtBackend::default();
        assert!(dev.render_scene(&req).is_valid());
    }

    /// 默认档与 `--fast` 必须肉眼可见不同（这里是机器判据：差异像素占比 > 30%，
    /// 且默认档存在"比 fast 更亮"的像素——GGX 高光，Lambert 不可能有）。
    #[test]
    fn default_path_differs_from_cpu_reference() {
        let req: SceneRenderRequest = red_sphere_request(40, 24);
        let slow: Framebuffer = RtBackend::new(8, 4).render_scene(&req);
        let fast: Framebuffer = prism_gpu::CpuRefBackend.render_scene(&req);
        assert_eq!(slow.pixels.len(), fast.pixels.len());
        let mut differing: usize = 0;
        let mut brighter: usize = 0;
        for (s, f) in slow.pixels.chunks_exact(4).zip(fast.pixels.chunks_exact(4)) {
            if s != f {
                differing += 1;
            }
            let ls: u32 = s[0] as u32 + s[1] as u32 + s[2] as u32;
            let lf: u32 = f[0] as u32 + f[1] as u32 + f[2] as u32;
            if ls > lf + 6 {
                brighter += 1;
            }
        }
        let total: usize = (40 * 24) as usize;
        assert!(
            differing * 100 / total > 30,
            "差异像素占比应 > 30%，实际 {differing}/{total}"
        );
        assert!(
            brighter > 0,
            "默认档必须存在比 --fast 更亮的像素（GGX 高光；Lambert 出不来）"
        );
    }

    /// 阴影射线确实在工作：光从 +Y 来，上球遮挡下球 ⇒ 部分像素
    /// `--fast` 判为受光、默认档判为阴影。
    #[test]
    fn shadow_rays_actually_run() {
        let req: SceneRenderRequest = SceneRenderRequest {
            width: 64,
            height: 64,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -3.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 40.0_f32.to_radians(),
                aspect: 1.0,
            },
            lights: vec![LightReq {
                direction: Vec3::new(0.0, 1.0, 0.0),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 3.0,
            }],
            spheres: vec![
                // 遮光球（靠上，位于光源与受影球之间）。
                SphereReq {
                    center: Vec3::new(0.0, 0.42, 0.0),
                    radius: 0.3,
                    albedo: Vec3::splat(0.8),
                    roughness: 0.6,
                    metalness: 0.0,
                    maps: prism_gpu::NO_MAPS,
                },
                // 受影球（靠下）。
                SphereReq {
                    center: Vec3::new(0.0, -0.42, 0.0),
                    radius: 0.42,
                    albedo: Vec3::splat(0.8),
                    roughness: 0.6,
                    metalness: 0.0,
                    maps: prism_gpu::NO_MAPS,
                },
            ],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
            curves: Vec::new(),
    ..Default::default()
};
        let slow: Framebuffer = RtBackend::new(12, 4).render_scene(&req);
        let fast: Framebuffer = prism_gpu::CpuRefBackend.render_scene(&req);
        // fast 只看 N·L（无阴影射线）；slow 有阴影。统计"fast 亮、slow 黑"的像素。
        let mut shadowed: usize = 0;
        let mut lit_fast: usize = 0;
        for (s, f) in slow.pixels.chunks_exact(4).zip(fast.pixels.chunks_exact(4)) {
            let ls: f32 = (s[0] as f32 + s[1] as f32 + s[2] as f32) / (3.0 * 255.0);
            let lf: f32 = (f[0] as f32 + f[1] as f32 + f[2] as f32) / (3.0 * 255.0);
            if lf > 0.25 {
                lit_fast += 1;
                if ls < 0.05 {
                    shadowed += 1;
                }
            }
        }
        assert!(lit_fast > 100, "受影球在 fast 档应有大量受光像素，实际 {lit_fast}");
        assert!(
            shadowed >= 30,
            "应至少有 30 个像素被判为阴影（fast 亮 / 默认档黑），实际 {shadowed}"
        );
    }

    /// 后处理链对齐：`prism_gpu::apply_aesthetic` 与"逐式手算"逐位相等。
    ///
    /// 这条钉住的是**两档共用同一份后处理**这一事实：`RtBackend` 不另写后期，
    /// 所以"默认档与 `--fast` 的差异"只可能来自着色模型。
    #[test]
    fn aesthetic_chain_matches_literal_reference() {
        let a: AestheticReq = AestheticReq { teal_orange: true, contrast: 1.2, filmic: true };
        let c = Vec3::new(0.6, 0.3, 0.2);
        let got: Vec3 = apply_aesthetic(c, &a);

        // 参照：不走任何 helper，逐分量按文档顺序手算。
        let luma: f32 = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
        let mut want = [
            c.x * 0.6 + (0.10 * c.x) * (1.0 - luma) * 0.5 + (1.0 * c.x) * luma * 0.5,
            c.y * 0.6 + (0.35 * c.y) * (1.0 - luma) * 0.5 + (0.7 * c.y) * luma * 0.5,
            c.z * 0.6 + (0.45 * c.z) * (1.0 - luma) * 0.5 + (0.3 * c.z) * luma * 0.5,
        ];
        for v in want.iter_mut() {
            *v = (*v - 0.5) * 1.2 + 0.5;
            *v = *v / (*v + 1.0);
            *v = v.clamp(0.0, 1.0);
        }
        assert_eq!(got.x.to_bits(), want[0].to_bits());
        assert_eq!(got.y.to_bits(), want[1].to_bits());
        assert_eq!(got.z.to_bits(), want[2].to_bits());

        // 关掉全部后期 ⇒ 恒等（只 clamp）。
        let off: AestheticReq =
            AestheticReq { teal_orange: false, contrast: 1.0, filmic: false };
        let id: Vec3 = apply_aesthetic(Vec3::new(0.25, 0.5, 2.0), &off);
        assert_eq!(id.x, 0.25);
        assert_eq!(id.y, 0.5);
        assert_eq!(id.z, 1.0);
    }

    /// 参数钳制：越界的 metalness / roughness 不得 panic（`Bsdf::new` 有断言）。
    #[test]
    fn out_of_range_material_params_do_not_panic() {
        let mut req: SceneRenderRequest = red_sphere_request(8, 6);
        req.spheres[0].metalness = 7.5;
        req.spheres[0].roughness = -3.0;
        req.spheres[0].albedo = Vec3::new(2.0, -1.0, 0.5);
        let mut dev: RtBackend = RtBackend::new(2, 2);
        assert!(dev.render_scene(&req).is_valid());
    }

    /// 非正 / 非有限强度与非有限方向的灯被跳过（不产生 NaN 辐亮度）。
    #[test]
    fn degenerate_lights_are_skipped() {
        let mut req: SceneRenderRequest = red_sphere_request(8, 6);
        req.lights = vec![
            LightReq {
                direction: Vec3::new(0.0, 0.0, 0.0),
                color: Vec3::splat(1.0),
                intensity: 1.0,
            },
            LightReq {
                direction: Vec3::new(0.0, 0.0, -1.0),
                color: Vec3::splat(1.0),
                intensity: -2.0,
            },
            LightReq {
                direction: Vec3::new(f32::NAN, 0.0, -1.0),
                color: Vec3::splat(1.0),
                intensity: 1.0,
            },
        ];
        let scene: RtScene = translate_scene(&req);
        assert!(scene.lights.is_empty(), "退化灯必须全部跳过");
    }

    /// `backend_name` 与 `is_available` 的口径。
    #[test]
    fn backend_identity() {
        let dev: RtBackend = RtBackend::default();
        assert_eq!(dev.backend_name(), "rt-path-tracing");
        assert!(dev.is_available());
        assert_eq!(dev.samples_per_pixel(), RT_DEFAULT_SPP);
        // 构造期钳制。
        assert_eq!(RtBackend::new(0, 0).spp, 1);
        assert_eq!(RtBackend::new(0, 0).max_bounces, 1);
    }
}
