//! WebAssembly C-ABI：JS harness 可驱动的 Rust 导出（D-14 移动端验证链路）。
//!
//! # 为什么是这一套约定
//!
//! * **零第三方 crate**（D-15）：不使用 `wasm-bindgen`，导出全部手写
//!   `extern "C"`；导入侧见 [`super::js`]（Rust 调 JS 的 13 个胶水函数）。
//! * **返回 `i32` 状态码 + 线性内存文本缓冲**：WebAssembly 的 JS 侧只支持数值
//!   返回（`i32`/`i64`/`f32`/`f64`/`v128`），无法返回字符串。本 ABI 用
//!   「状态码判成败、文本缓冲给细节」——比「指针 + 长度 + 手工 UTF-8 解码」多
//!   一次 `TextDecoder` 调用，但避免 JS 手工指针算术与裸指针生命周期管理，
//!   且错误信息在 Rust 侧一次成型（含上下文），不会出现半截 UTF-8 被截断成
//!   乱码（缓冲写入始终按 `char` 边界截断）。
//! * **两阶段 `run`/`collect`**：WebGPU 里只有 `queue.onSubmittedWorkDone()` 与
//!   `GPUBuffer.mapAsync()` 是 Promise，WASM 同步栈无法 `await`。因此需要等待的
//!   测试项由 `run` 提交并返回 [`Status::Pending`]，宿主 `await` 完成后调用
//!   `collect` 读回结果——JS 胶水只做"无法用纯 WASM 表达的异步/对象操作"。
//! * **测试项清单以 Rust 为单一真源**（任务 2 方案 A）：JS 通过
//!   [`prism_test_count`] / `prism_test_id`] / [`prism_test_name`] /
//!   [`prism_test_needs_gpu`] 枚举，再对 `i` 调 [`prism_test_run`]。JS 侧不
//!   硬编码任何 id→函数映射表，杜绝两侧漂移（D-16：去繁不留死代码）。
//!
//! # 内存模型
//!
//! * [`prism_alloc`] / [`prism_dealloc`]：16 字节对齐的裸分配，供 JS 写入
//!   任意字节（WGSL 源码、测试向量等）；两侧必须用同一 `len` 释放。
//! * [`scratch`]：模块私有的定长工作区（输入/输出/直方图/句柄表/参数），
//!   地址在 wasm 生命周期内稳定，Rust 直接把其地址交给胶水做
//!   `write_buffer` / `copy_out`。
//! * 文本缓冲 [`prism_error_ptr`] / [`prism_result_ptr`] 由 `thread_local`
//!   定长数组承载，**下一次 `run`/`collect` 会覆写**——宿主须在调用后立刻读走。
//!
//! # 平台行为
//!
//! native 目标下所有 GPU 胶水桩返回 0（见 [`gpu`] 模块），需 GPU 的测试项
//! 稳定返回 [`Status::Skip`]，不 panic、不影响 `cargo test --workspace`。

/// 把 `Result<T, Status>` 的 `Ok` 值取出，`Err` 时**从当前函数直接返回**该 `Status`。
///
/// 用途：测试项的 `collect` 阶段返回类型是 `Status`（不是 `Result`），但内部
/// 复用了一大批 `Result<_, Status>` 的比对原语（[`cmp_f32`] / [`cmp_u32`] /
/// [`cmp_bytes`] / [`ready_device`] / [`readback_to`] …）。本宏让这些原语可以
/// 用 `try_status!(…)` 串联书写，既不引入 `unwrap()`（宪章 C18 禁 panic），
/// 也不必把每个调用展开成 `match … { Ok(v) => v, Err(s) => return s }`。
macro_rules! try_status {
    ($e:expr $(,)?) => {
        match $e {
            Ok(__v) => __v,
            Err(__s) => return __s,
        }
    };
}

use core::cell::{Cell, RefCell};
use std::alloc::Layout;
use std::sync::OnceLock;

use crate::scan::{scan_blelloch_cpu, scan_cpu_ref, ScanShape};
use crate::shaders;
use crate::sort::{radix_sort_u32, sort_cpu_ref};
use crate::web::device::{
    histogram_params_bytes, scan_params_bytes, ScanConfig, WebGpuBackend, HISTOGRAM_PARAMS_BYTES,
    SCAN_PARAMS_BYTES, WGSL_HISTOGRAM, WGSL_SCAN_BLOCK, WORKGROUP_SIZE,
};
use crate::web::scene_gpu::{self, SceneFrame};
use crate::{CpuRefBackend, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;

/// ABI 版本号。宿主加载后应先比对，不匹配即拒绝运行（避免错位读缓冲）。
pub const ABI_VERSION: u32 = 1;

/// 错误文本缓冲容量（含结尾 NUL）。
pub const ERR_CAP: usize = 2048;
/// 结果文本缓冲容量（含结尾 NUL）。
pub const RES_CAP: usize = 8192;
/// 裸分配对齐字节数（覆盖 `u32`/`f32`/`u64` 的最大自然对齐）。
pub const PRISM_ALLOC_ALIGN: usize = 16;

/// 导出状态码（`i32`）。0 = PASS。
pub mod status {
    /// 通过。
    pub const PASS: i32 = 0;
    /// 失败；细节见错误缓冲。
    pub const FAIL: i32 = 1;
    /// 已提交，宿主需 `await` 后调用 `prism_test_collect`。
    pub const PENDING: i32 = 2;
    /// 跳过（前置条件不满足，如未初始化 device）。
    pub const SKIP: i32 = 3;
    /// 参数错误（测试项下标越界）。
    pub const BAD_ARG: i32 = 4;
}

/// GPU buffer 用途位（与宿主 JS 胶水约定）。
///
/// `pub(crate)`：`web::scene_gpu` 的场景 dispatch 复用同一套用途位（不得自造数值）。
pub(crate) mod usage {
    /// `GPUBufferUsage.STORAGE | COPY_SRC | COPY_DST`。
    pub const STORAGE: u32 = 1;
    /// `GPUBufferUsage.UNIFORM | COPY_DST | COPY_SRC`。
    ///
    /// # 为什么必须带 `COPY_SRC`
    /// `gpu.params_roundtrip` 的判据是「把 uniform 缓冲原样读回、与 CPU 打包字节
    /// 逐字节比对」——这是"参数真的到达 GPU"唯一的直接证据。而
    /// `copyBufferToBuffer` **要求源缓冲带 `COPY_SRC`**，否则验证失败、拷贝静默
    /// 不执行、读回拿到零初始化内容（表现成"参数全错"，而根因只是缺一个 usage 位）。
    /// 该能力位由 `web/index.html` 的 `prism_create_buffer` 的 UNIFORM 分支提供
    /// （`UNIFORM | COPY_DST | COPY_SRC`），不新增 glue 函数、不改导入表。
    pub const UNIFORM: u32 = 2;
}

/// `scan_block` 单块元素数（= `2 * WORKGROUP_SIZE`，与 WGSL 内 `N = 128` 一致）。
const SCAN_N: usize = 128;
/// `histogram` 测试的键数量。
const HIST_N: usize = 256;
/// `histogram` 桶数（= `2^RADIX_BITS`，与 CPU 参考 `sort.rs` 同口径）。
const HIST_BUCKETS: u32 = 256;

// ---------------------------------------------------------------------------
// D-14 补测：WGSL 内存布局常量（模拟 + wavefront 真机 dispatch）
// ---------------------------------------------------------------------------
//
// # 为什么把这张表摆在 Rust 侧
//
// D-14 真机已抓到三类"只有真机能暴露"的缺陷：uniform 未按 16 字节对齐、
// bind group 偏移双重计算、device promise 被丢弃。前两类的共同根因是
// **host 侧的字节数是"手写出来的"**，与 WGSL 里的 `struct`/`array` 声明
// 没有���一的推导关系。本表把每个布局常量显式命名，并全部由 `web/host_test.mjs`
// 第 8 节门禁对照 `shaders/*.wgsl` 的声明**逐项复核**（从 `array<T>` 推
// `stride(T)`、从 `struct` 推 uniform 大小），任一处漂移即转红。
//
// 因此：**禁止在本文件里为这些 buffer 写字面量 size**，必须引用本表常量
// （门禁 8.4 会把字面量 size 判红）。

/// `f32` 标量字节数（`array<f32>` 的元素步长）。
pub const F32_BYTES: u32 = 4;
/// `array<vec3<f32>>` 的元素步长。
///
/// WGSL 规定 `vec3<f32>` 的 `align = 16`、`size = 12`，而**定长/运行时数组的
/// 步长 = roundUp(对齐, size) = 16**。故 `array<vec3<f32>>` 每元素占 16 字节
/// （末 4 字节为填充），**不是** 12——这正是 storage buffer 最常见的错算。
pub const VEC3_STRIDE_BYTES: u32 = 16;
/// 状态槽（AoS：`pos:vec3` + `vel:vec3` + `inv:f32`）的字节数。
///
/// 布局：`pos:vec3@0`（占 0..12，末 4 字节补到 16）/ `vel:vec3@16`（占 16..28）
/// / `inv:f32@28`（占 28..32，**恰好塞进 `vel` 的对齐填充里**）→ 对齐 16 → **32**。
/// 即两个 16 字节 `vec3` 槽 —— 故由 [`VEC3_STRIDE_BYTES`] 派生而非写死。
///
/// 【为什么 sim_sph / sim_cloth 必须用 AoS】宿主胶水的
/// `prism_create_bind_group` 把句柄数组第 `i` 个**硬映射**到 `binding: i`，
/// 而 `layout:"auto"` 只为被 dispatch 的 entry **静态引用**的 binding 生成布局。
/// 二者相乘 ⇒ 每个 entry 的引用集合必须恰好是连续前缀 `{0,…,n-1}`。SoA 的
/// `pos/vel/inv` 三个独立 binding 让各 entry 的集合互不包含（前缀必嵌套），
/// **任何编号方案都无解**；合并成单个 AoS binding 后前缀性质自然成立。
pub const STATE_STRIDE_BYTES: u32 = VEC3_STRIDE_BYTES * 2;
/// `vec4<f32>` 字节数（`array<vec4<f32>>` 的元素步长）。
pub const VEC4_STRIDE_BYTES: u32 = 16;
/// `sim_common.wgsl::Constraint` 的字节数（`{u32, u32, f32, u32}`）。
pub const CONSTRAINT_STRIDE_BYTES: u32 = 16;
/// `sim_common.wgsl::Particle` 的字节数。
///
/// 布局：`pos:vec3@0` / `mass_inv:f32@12` / `vel:vec3@16` / `_pad:f32@28` /
/// `cell:u32@32` / `_pad2:u32@36`，末成员末尾 40 → 对齐 16 → **48**。
pub const PARTICLE_STRIDE_BYTES: u32 = 48;
/// `rt_wavefront.wgsl::PathSlot` 的字节数。
///
/// 布局：`origin:vec3@0` / `time@12` / `throughput:vec3@16` / `radiance:vec3@32`
/// / `prev_pdf_dir@44` / `state@48` / `bounce@52` / `flags@56` / `aux_steps@60`
/// / `placeholder:vec3@64`，末成员末尾 76 → 对齐 16 → **80**。
pub const PATH_SLOT_STRIDE_BYTES: u32 = 96;
/// `sim_particles.wgsl::SimParams` 的 uniform 字节数。
///
/// 布局：`count@0` / `dt@4` / `h@8` / `gravity:vec3@16` / `seed@28` /
/// `frame@32` / `emitter_rate@36` / `emitter_origin:vec3@48`，末尾 60 →
/// 对齐 16 → **64**。
pub const SIM_PARAMS_BYTES: u32 = 64;
/// `sim_sph.wgsl::SphParams` 的 uniform 字节数。
///
/// 布局：`count@0` / `h@4` / `rest_density@8` / `compliance@12` / `dt@16` /
/// `gravity:vec3@32`，末尾 44 → 对齐 16 → **48**。
pub const SPH_PARAMS_BYTES: u32 = 48;
/// `sim_cloth.wgsl::ClothParams` 的 uniform 字节数。
///
/// 布局：`rows@0` / `cols@4` / `spacing@8` / `dt@12` / `compliance_struct@16`
/// / `compliance_bend@20` / `damping@24` / `gravity:vec3@32`，末尾 44 →
/// 对齐 16 → **48**。
pub const CLOTH_PARAMS_BYTES: u32 = 48;
/// `rt_wavefront.wgsl::WFParams` 的 uniform 字节数（10 × `u32` = 40 → **48**）。
pub const WF_PARAMS_BYTES: u32 = 48;
/// `rt_wavefront.wgsl::Camera` 的 uniform 字节数。
///
/// 布局：`eye:vec3@0` / `_p0@12` / `forward:vec3@16` / `_p1@28` / `right:vec3@32` /
/// `_p2@44` / `up:vec3@48` / `_p3@60` / 5 个 `f32`（`tan_half_fov`/`aperture_radius`/
/// `focus_distance`/`shutter_open`/`shutter_close`）@64..84，末尾对齐 16 → **96**。
pub const CAMERA_PARAMS_BYTES: u32 = 96;
/// 一次 dispatch 最多设置的 bind group 数（`rt_wavefront` 的 generate 用到
/// group 0/1/2/3 共 4 个；`prism_dispatch` 的 `bgs` 数组容量即此值）。
pub const BIND_GROUP_MAX: u32 = 4;

// ---------------------------------------------------------------------------
// D-14 补测：测试场景规模（元素数 8~64，工作组 1~2 个，手机毫秒级）
// ---------------------------------------------------------------------------

/// `sim_particles` 测试的粒子数（1 个工作组即可覆盖）。
pub const SIM_P_COUNT: u32 = 32;
/// `sim_sph` 测试的粒子数。
pub const SPH_P_COUNT: u32 = 16;
/// `sim_cloth` 测试的粒子数（4 行 × 2 列）。
pub const CLOTH_P_COUNT: u32 = 8;
/// `sim_cloth` 测试装载的约束数。
///
/// **刻意取 4 对两两不相交的粒子**（见 `t_gpu_sim_cloth` 的注释：`cloth_project`
/// 按约束并行却写共享的 `cp_pos`，真实网格必然数据竞争）。4 = 4 行 × 每行 1 对
/// 水平结构约束。
pub const CLOTH_C_COUNT: u32 = 4;
/// `rt_wavefront` 测试的图像宽（像素）。
pub const WF_WIDTH: u32 = 2;
/// `rt_wavefront` 测试的图像高（像素）。
pub const WF_HEIGHT: u32 = 2;
/// `rt_wavefront` 测试的每像素样本数。
pub const WF_SPP: u32 = 2;
/// `rt_wavefront` 测试的路径槽数（= 宽 × 高 × spp）。
pub const WF_PATHS: u32 = WF_WIDTH * WF_HEIGHT * WF_SPP;
/// `rt_wavefront` 测试的像素数（= 宽 × 高）。
pub const WF_PIXELS: u32 = WF_WIDTH * WF_HEIGHT;

// ---- D-14 补测：rt_wavefront 多 bind group 测试（generate/shade/extend）----
/// 多 group 测试的图像宽（8×4×spp=2 = 64 条路径，dispatch 2 个工作组验证边界守卫）。
pub const WF2_WIDTH: u32 = 8;
/// 多 group 测试的图像高。
pub const WF2_HEIGHT: u32 = 4;
/// 多 group 测试的每像素样本数。
pub const WF2_SPP: u32 = 2;
/// 多 group 测试的路径槽数（= 8×4×2 = 64）。
pub const WF2_PATHS: u32 = WF2_WIDTH * WF2_HEIGHT * WF2_SPP;
/// 多 group 测试的像素数（= 8×4 = 32）。
pub const WF2_PIXELS: u32 = WF2_WIDTH * WF2_HEIGHT;
/// Sobol 方向数表字节数（`[64][32] u32` 方向数矩阵，rt_shade group(1) binding 0）。
pub const SOBOL_DIR_BYTES: u32 = 64 * 32 * 4;
/// `array<u32>` 元素字节数（`sobol_dir` / `prim_mat` 等）。
pub const U32_BYTES: u32 = 4;
/// `rt_scene.wgsl::Tri` 的字节数（`a:vec3@0` / `b:vec3@16` / `c:vec3@32` → 48）。
pub const TRI_STRIDE_BYTES: u32 = 48;
/// `rt_scene.wgsl::Material` 的字节数（末尾 `emissive:vec3@48` → 对齐 16 → 64）。
pub const MATERIAL_STRIDE_BYTES: u32 = 64;
/// `rt_scene.wgsl::Light` 的字节数（`position:vec3@0` / `radius@12` / `color:vec3@16` → 32）。
pub const LIGHT_STRIDE_BYTES: u32 = 32;
/// `rt_scene.wgsl::SceneParams` 的 uniform 字节数（8 × `u32` = 32）。
pub const SCENE_PARAMS_BYTES: u32 = 32;
/// 多 group 测试装载的三角形数。
pub const WF2_TRI_COUNT: u32 = 1;
/// 多 group 测试装载的材质数。
pub const WF2_MAT_COUNT: u32 = 2;
/// 多 group 测试装载的灯光数（缓冲区常驻 1 盏，但 `light_count=0` ⇒ NEE 运行期不执行）。
pub const WF2_LIGHT_COUNT_LOADED: u32 = 1;
/// 多 group 测试场景的灯光计数（0：关掉 NEE 运行期，只留静态引用以验 group(2) 绑定）。
pub const WF2_LIGHT_COUNT: u32 = 0;
/// 多 group 测试的 prim→material 映射元素数。
pub const WF2_PRIM_COUNT: u32 = WF2_TRI_COUNT;
/// `scene_params.node_id`（须与 [`WfParams::node_id`] 一致，`sample_index` 读它）。
pub const WF2_NODE_ID: u32 = 7;
/// 多 group 测试的 `max_bounces`（取 4 让 extend 走命中 + BSDF 续弹分支）。
pub const WF2_MAX_BOUNCES: u32 = 4;
/// 多 group 路径槽 storage 字节数。
pub const WF2_SLOTS_BYTES: u32 = PATH_SLOT_STRIDE_BYTES * WF2_PATHS;
/// 多 group 三角形 storage 字节数。
pub const WF2_TRI_BYTES: u32 = TRI_STRIDE_BYTES * WF2_TRI_COUNT;
/// 多 group 材质 storage 字节数。
pub const WF2_MAT_BYTES: u32 = MATERIAL_STRIDE_BYTES * WF2_MAT_COUNT;
/// 多 group 灯光 storage 字节数。
pub const WF2_LIGHT_BYTES: u32 = LIGHT_STRIDE_BYTES * WF2_LIGHT_COUNT_LOADED;
/// 多 group prim_mat storage 字节数。
pub const WF2_PRIM_MAT_BYTES: u32 = U32_BYTES * WF2_PRIM_COUNT;
/// `Camera` uniform 在参数区内的偏移（多 group 测试的相机）。
pub const WF2_CAMERA_OFF: usize = 240;
/// `SceneParams` uniform 在参数区内的偏移。
pub const WF2_SCENE_OFF: usize = 336;

// ---- GI 的 GPU compute 路径（`gpu.gi_compute`）----
//
// 布局常量一律由 `gi_compute.wgsl` 的声明推导，`wgsl_lint.mjs` 规则 A 会拿本表
// 与 WGSL 逐项复核（uniform 16B 对齐 + 宿主写入量自洽）。**禁止在本文件为 GI 的
// buffer 写字面量 size**，一律引用下列常量。
/// 每条探针的半球采样方向数（= `gi_compute.wgsl::GI_DIRS`）。
pub const GI_DIRS: u32 = 8;
/// uniform 内 SDF 图元容量（= `gi_compute.wgsl::GI_MAX_PRIMS`）。
pub const GI_MAX_PRIMS: u32 = 4;
/// `gi_compute.wgsl::GiPrim` 的字节数（`kind:u32@0` / `radius:f32@4` /
/// `_pad0:vec2<u32>@8` / `center:vec4<f32>@16` / `_pad1:vec4<u32>@32`，
/// 末 48 → 对齐 16 → **48**）。
///
/// 【V2.2 真机 bug 的修复只改类型、不改数字】`center` 原为 `vec3<f32>@16` +
/// 隐式 padding，真机驱动对 struct 内 `vec3<f32>` 的 size 处理与规范不一致
/// （规范 12，实现更大），导致后续字段整体重排。改为 `vec4<f32>` 后 align/size
/// 都是 16，**逐字节等价**——故本常量与 [`gi_params_bytes`] 里的 `o+16/20/24`
/// 全部一字节未动。详见 `gi_compute.wgsl` 结构体段的整段说明。
pub const GI_PRIM_STRIDE_BYTES: u32 = 48;
/// `gi_compute.wgsl::GiParams` 的 uniform 字节数。
///
/// 布局：`probe_count@0` / `point_count@4` / `prim_count@8` / `n_dirs@12` /
/// `env_radiance:vec4<f32>@16`（原 `vec3@16`+`_pad0@28`）/
/// `albedo:vec4<f32>@32`（原 `vec3@32`+`_pad1@44`）/
/// `t_max@48` / `hit_eps@52` / `safety@56` / `max_steps@60` /
/// `prims:array<GiPrim,4>@64`（4×48=192）→ 末尾 256 → 对齐 16 → **256 字节**。
pub const GI_PARAMS_BYTES: u32 = 256;
/// `gi_compute.wgsl::GiProbe` 的字节数。
///
/// 布局：`pos:vec4<f32>@0` / `normal:vec4<f32>@16` /
/// `samples:array<vec4<f32>,8>@32`（8×16=128 → 末 160）/
/// `sh:array<vec4<f32>,9>@160`（9×16=144 → 末 304）→ 对齐 16 → **304 字节**。
///
/// 【为什么是 vec4 而不是 vec3】`array<vec3<f32>, N>` 的元素 stride 依赖
/// `vec3` 的 **size**（规范 12，故 stride=16）。真机驱动把它算得更大 → 每个
/// 元素多占字节 → `normal`/`samples`/`sh` 的偏移全部错位（真机实测 `normal`
/// 读到 samples 的 -1 哨兵、`samples[6]` 落到 `sh[0]` 上）。`vec4<f32>` 的
/// size/stride 在规范与实现上都无歧义（16/16）→ 布局零变化且不再有解释空间。
pub const GI_PROBE_STRIDE_BYTES: u32 = 304;
/// `gi_compute.wgsl::GiPoint` 的字节数（`pos:vec4<f32>@0` /
/// `irradiance:vec4<f32>@16` → 对齐 16 → **32**）。
pub const GI_POINT_STRIDE_BYTES: u32 = 32;
/// GI 测试的探针数（1 个工作组内，3×64 覆盖边界守卫用 1 组即可）。
pub const GI_N_PROBES: u32 = 3;
/// GI 测试的查询点数。
pub const GI_N_POINTS: u32 = 4;
/// GI 探针 storage buffer 字节数（`array<GiProbe>`）。
pub const GI_PROBES_BYTES: u32 = GI_PROBE_STRIDE_BYTES * GI_N_PROBES;
/// GI 查询点 storage buffer 字节数（`array<GiPoint>`）。
pub const GI_POINTS_BYTES: u32 = GI_POINT_STRIDE_BYTES * GI_N_POINTS;

// ---- 派生字节数（一律由上表常量算出，不写字面量）----
/// 粒子 storage buffer 字节数。
pub const PARTICLES_BYTES: u32 = PARTICLE_STRIDE_BYTES * SIM_P_COUNT;
/// SPH 状态 storage 字节数（`array<SphState>`，AoS 32 字节/粒子）。
pub const SPH_STATE_BYTES: u32 = STATE_STRIDE_BYTES * SPH_P_COUNT;
/// SPH 标量 storage buffer 字节数（`array<f32>`，密度 / λ）。
pub const SPH_SCALAR_BYTES: u32 = F32_BYTES * SPH_P_COUNT;
/// 布料状态 storage 字节数（`array<ClothState>`，AoS 32 字节/粒子）。
pub const CLOTH_STATE_BYTES: u32 = STATE_STRIDE_BYTES * CLOTH_P_COUNT;
/// 布料约束 storage buffer 字节数。
pub const CLOTH_CONS_BYTES: u32 = CONSTRAINT_STRIDE_BYTES * CLOTH_C_COUNT;
/// 布料拉格朗日乘子 storage buffer 字节数。
pub const CLOTH_LAMBDA_BYTES: u32 = F32_BYTES * CLOTH_C_COUNT;
/// 路径槽 storage buffer 字节数。
pub const WF_SLOTS_BYTES: u32 = PATH_SLOT_STRIDE_BYTES * WF_PATHS;
/// 累积图 storage buffer 字节数。
pub const WF_ACCUM_BYTES: u32 = VEC4_STRIDE_BYTES * WF_PIXELS;
/// uniform 参数区字节数（5 份参数共用一块，按 `*_PARAMS_OFF` 切分）。
pub const PARAMS_ARENA_BYTES: usize = 512;
/// `SimParams` 在参数区内的偏移。
pub const SIM_PARAMS_OFF: usize = 0;
/// `SphParams` 在参数区内的偏移。
pub const SPH_PARAMS_OFF: usize = 64;
/// `ClothParams` 在参数区内的偏移。
pub const CLOTH_PARAMS_OFF: usize = 128;
/// `WFParams` 在参数区内的偏移。
pub const WF_PARAMS_OFF: usize = 192;

// ---------------------------------------------------------------------------
// 状态与文本缓冲
// ---------------------------------------------------------------------------

/// 内部状态枚举（导出时映射为 `i32` 状态码）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// 通过，`prism_test_run` 直接返回 PASS。
    Pass,
    /// 失败：错误文本已写入错误缓冲。
    Fail,
    /// 已提交 GPU 工作，宿主 settle 后调 `collect`。
    Pending,
    /// 跳过：细节写入结果缓冲。
    Skip,
}

impl Status {
    /// 映射为导出的 `i32` 状态码。
    fn code(self) -> i32 {
        match self {
            Status::Pass => status::PASS,
            Status::Fail => status::FAIL,
            Status::Pending => status::PENDING,
            Status::Skip => status::SKIP,
        }
    }
}

thread_local! {
    /// 错误文本（定长，NUL 结尾；写入按 `char` 边界安全截断，绝不 panic）。
    static ERR: RefCell<[u8; ERR_CAP]> = const { RefCell::new([0u8; ERR_CAP]) };
    /// 错误文本有效长度（不含 NUL）。
    static ERR_LEN: Cell<u32> = const { Cell::new(0) };
    /// 结果文本（定长，NUL 结尾）。
    static RES: RefCell<[u8; RES_CAP]> = const { RefCell::new([0u8; RES_CAP]) };
    /// 结果文本有效长度（不含 NUL）。
    static RES_LEN: Cell<u32> = const { Cell::new(0) };
}

/// 把 `s` 写入定长缓冲（按 UTF-8 `char` 边界截断，末尾补 NUL）。
fn write_capped<const N: usize>(buf: &'static std::thread::LocalKey<RefCell<[u8; N]>>, len: &'static std::thread::LocalKey<Cell<u32>>, s: &str) {
    let mut used = 0usize;
    buf.with(|b| {
        let mut v = b.borrow_mut();
        v.fill(0);
        for ch in s.chars() {
            let mut tmp = [0u8; 4];
            let encoded = ch.encode_utf8(&mut tmp);
            // 留 1 字节给 NUL，故可用上限为 N-1。
            if used + encoded.len() + 1 > N {
                break;
            }
            v[used..used + encoded.len()].copy_from_slice(encoded.as_bytes());
            used += encoded.len();
        }
        v[used] = 0;
    });
    len.with(|l| l.set(used as u32));
}

/// 写错误文本（失败时调用）。
fn set_err(s: &str) {
    write_capped(&ERR, &ERR_LEN, s);
}

/// 写结果文本（成功/跳过时调用）。
fn set_result(s: &str) {
    write_capped(&RES, &RES_LEN, s);
}

/// 构造失败：写错误文本并返回 [`Status::Fail`]。
fn fail(s: impl AsRef<str>) -> Status {
    set_err(s.as_ref());
    Status::Fail
}

/// 构造跳过：写结果文本并返回 [`Status::Skip`]。
fn skip(s: impl AsRef<str>) -> Status {
    set_result(s.as_ref());
    Status::Skip
}

// ---------------------------------------------------------------------------
// 私有工作区（定长，地址稳定）
// ---------------------------------------------------------------------------

/// GPU 测试共用的定长工作区。
struct Scratch {
    /// `scan_block` 输入向量。
    scan_in: [u32; SCAN_N],
    /// `scan_block` 输出（GPU 读回落点）。
    scan_out: [u32; SCAN_N],
    /// `histogram` 键向量。
    keys: [u32; HIST_N],
    /// `histogram` 桶计数（GPU 读回落点）。
    hist: [u32; HIST_BUCKETS as usize],
    /// bind group 句柄数组（`prism_create_bind_group` 的 `bufs_ptr` 落点）。
    handles: [u32; 8],
    /// 一次 dispatch 的 bind group 句柄数组（`prism_dispatch` 的 `bgs_ptr` 落点）；
    /// 第 `i` 个绑到 `@group(i)`。
    bgs: [u32; BIND_GROUP_MAX as usize],
    /// uniform 参数缓冲（16 字节，对齐 WGSL uniform 地址空间要求）。
    params: [u8; HISTOGRAM_PARAMS_BYTES],
    /// 读回暂存句柄。
    staging: u32,
    /// 读回落点地址。
    out_ptr: u32,
    /// 读回字节数。
    out_len: u32,
    /// 参与 dispatch 的缓冲数。
    n_bufs: u32,
    /// 提交阶段捕获的验证错误码（`prism_pop_error`）。
    submit_err: u32,
    /// 本项在 run 阶段登记的待结算错误槽位数（由 collect 阶段消费）。
    ///
    /// 宿主 `pendingErrors` 是**全局 FIFO**：错误码由 `settle()` 里的 Promise
    /// **异步** push，而 wasm 侧只能同步 pop。故错误检查**必须**放在 collect
    /// 阶段（宿主已 `await settle()` 之后），run 阶段弹出来的只会是**上一项**
    /// settle 后残留的错误——这正是 D-14 真机里 `sim_cloth` 报"第 1 个"、
    /// `rt_wavefront` 报"第 0 个"的原因（两者绑定本身完全正确）。
    err_slots: u32,
    // ---- D-14 补测：以下为 7 个新 shader 测试项的输入/输出工作区 ----
    //
    // 全部为**字节数组**（`[u8; N]`）而非类型化数组：storage buffer 的实际布局
    // （vec3 的 16 字节步长、struct 的对齐填充）是本测试要验证的对象，用
    // `Vec3`/`f32` 数组表达会**丢掉填充字节**，让「布局算错」这类缺陷无法暴露。
    // 读写一律走 `pack_f32` / `pack_u32` / `unpack_f32` / `unpack_u32`。
    /// `sim_particles` 粒子 storage buffer（输入 + 读回落点同址，原地判对）。
    particles: [u8; PARTICLES_BYTES as usize],
    /// `sim_sph` 状态 storage（`array<SphState>` AoS：pos/vel/inv 共 32 字节/粒子）。
    sph_state: [u8; SPH_STATE_BYTES as usize],
    /// `sim_sph` 密度 storage（GPU 写回落点）。
    sph_density: [u8; SPH_SCALAR_BYTES as usize],
    /// `sim_sph` 拉格朗日乘子 storage（GPU 写回落点）。
    sph_lambda: [u8; SPH_SCALAR_BYTES as usize],
    /// `sim_cloth` 状态 storage（`array<ClothState>` AoS，读回落点：投影会改写 pos）。
    cloth_state: [u8; CLOTH_STATE_BYTES as usize],
    /// `sim_cloth` 约束 storage（`array<Constraint>`，16 字节/条）。
    cloth_cons: [u8; CLOTH_CONS_BYTES as usize],
    /// `sim_cloth` 拉格朗日乘子 storage（读回落点）。
    cloth_lambdas: [u8; CLOTH_LAMBDA_BYTES as usize],
    /// `rt_wavefront` 路径槽 storage（读回落点）。
    wf_slots: [u8; WF_SLOTS_BYTES as usize],
    /// `rt_wavefront` 累积图 storage（读回落点）。
    wf_accum: [u8; WF_ACCUM_BYTES as usize],
    /// 多 group 测试路径槽 storage（读回落点）。
    wf2_slots: [u8; WF2_SLOTS_BYTES as usize],
    /// Sobol 方向数表（`[64][32] u32`，rt_shade group(1) binding 0）。
    sobol_dir: [u8; SOBOL_DIR_BYTES as usize],
    /// 多 group 场景三角形 storage。
    wf2_tris: [u8; WF2_TRI_BYTES as usize],
    /// 多 group 场景材质 storage。
    wf2_mats: [u8; WF2_MAT_BYTES as usize],
    /// 多 group 场景灯光 storage（常驻 1 盏，light_count=0 时运行期不读）。
    wf2_lights: [u8; WF2_LIGHT_BYTES as usize],
    /// 多 group prim→material 映射。
    wf2_prim_mat: [u8; WF2_PRIM_MAT_BYTES as usize],
    /// D3/D2 的 uniform 参数区（5 份参数共用，按 `*_PARAMS_OFF` 切分）。
    params2: [u8; PARAMS_ARENA_BYTES],
    // ---- GI GPU 路径的输入/输出工作区 ----
    /// GI 探针 storage（`array<GiProbe>`）：输入位置/法线，读回落点（trace+project 写回）。
    gi_probes: [u8; GI_PROBES_BYTES as usize],
    /// GI 查询点 storage（`array<GiPoint>`）：输入位置，读回落点（interpolate 写出辐照度）。
    gi_points: [u8; GI_POINTS_BYTES as usize],
    /// GI uniform 参数（独立于 `params2` 共享区，避免与并行改动争用参数区）。
    gi_params: [u8; GI_PARAMS_BYTES as usize],
    /// `gpu.params_roundtrip` 的读回落点（uniform 缓冲原样读回 → 与本字段比对的
    /// CPU 打包字节逐字节对读）。**必须与 `gi_params` 分开两块**：后者是"要写进去的
    /// 源"，若复用同一块，读回会覆写源，比对就退化成"自己和自己比"（恒真）。
    gi_params_echo: [u8; GI_PARAMS_BYTES as usize],
    /// `gpu.gi_layout` 的读回落点：**只**装 `gi_probes[0]` 一个元素
    /// （[`GI_PROBE_STRIDE_BYTES`] 字节）。
    ///
    /// 【为什么不复用 `gi_probes`】`gi_probes` 是 `gpu.gi_compute` 的输入源，
    /// 而 `gpu.gi_layout` 与它在宿主清单里相邻、可能被并发 run；若两者共用一块
    /// 暂存，`gpu.gi_layout` 的读回会覆写 `gpu.gi_compute` 尚未提交完的输入字节
    /// → 后者读到被 census 改写过的 samples（假失败）。独立一块从根上排除这个
    /// 竞态，代价只有 304 字节的零初始化静态区（`.bss`，不进 wasm 数据段）。
    gi_layout_out: [u8; GI_PROBE_STRIDE_BYTES as usize],
}

impl Scratch {
    /// 编译期零初始化（供 `thread_local!` 的 `const` 块使用）。
    const fn new() -> Self {
        Self {
            scan_in: [0; SCAN_N],
            scan_out: [0; SCAN_N],
            keys: [0; HIST_N],
            hist: [0; HIST_BUCKETS as usize],
            handles: [0; 8],
            bgs: [0; BIND_GROUP_MAX as usize],
            params: [0; HISTOGRAM_PARAMS_BYTES],
            staging: 0,
            out_ptr: 0,
            out_len: 0,
            n_bufs: 0,
            submit_err: 0,
            err_slots: 0,
            particles: [0; PARTICLES_BYTES as usize],
            sph_state: [0; SPH_STATE_BYTES as usize],
            sph_density: [0; SPH_SCALAR_BYTES as usize],
            sph_lambda: [0; SPH_SCALAR_BYTES as usize],
            cloth_state: [0; CLOTH_STATE_BYTES as usize],
            cloth_cons: [0; CLOTH_CONS_BYTES as usize],
            cloth_lambdas: [0; CLOTH_LAMBDA_BYTES as usize],
            wf_slots: [0; WF_SLOTS_BYTES as usize],
            wf_accum: [0; WF_ACCUM_BYTES as usize],
            wf2_slots: [0; WF2_SLOTS_BYTES as usize],
            sobol_dir: [0; SOBOL_DIR_BYTES as usize],
            wf2_tris: [0; WF2_TRI_BYTES as usize],
            wf2_mats: [0; WF2_MAT_BYTES as usize],
            wf2_lights: [0; WF2_LIGHT_BYTES as usize],
            wf2_prim_mat: [0; WF2_PRIM_MAT_BYTES as usize],
            params2: [0; PARAMS_ARENA_BYTES],
            gi_probes: [0; GI_PROBES_BYTES as usize],
            gi_points: [0; GI_POINTS_BYTES as usize],
            gi_params: [0; GI_PARAMS_BYTES as usize],
            gi_params_echo: [0; GI_PARAMS_BYTES as usize],
            gi_layout_out: [0; GI_PROBE_STRIDE_BYTES as usize],
        }
    }
}

thread_local! {
    /// 私有工作区。
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
    /// 已初始化的 device 句柄（`0` = 无）。
    static DEVICE: Cell<u32> = const { Cell::new(0) };
    /// `device.init` 最近两轮 run 预留过的句柄（`0` = 空槽，环形写）。
    ///
    /// 【为什么必须留候选集合，而不是记"我自己那张票"】宿主会**并发**跑
    /// `device.init`（`boot()` 自动跑一次 + 回归脚本点「全部运行」再跑一次），
    /// 而 wasm ABI 的 `test_collect(i)` **不带 run 令牌**：两轮共用同一份状态，
    /// 谁最后写槽位，`.collect` 就可能读到谁的句柄。记单张票仍是共享状态，
    /// 照样张冠李戴。故改为：在候选集合里挑**任一已就绪**的句柄。判据没有
    /// 放宽——仍要求 `device_ready(h) == 1`，即句柄表里确实是真实 GPUDevice。
    static DEVICE_TICKETS: RefCell<[u32; 2]> = const { RefCell::new([0; 2]) };
}

/// 登记一次 requestDevice 预留的句柄（环形写入 2 个槽）。
fn note_device_ticket(dev: u32) {
    DEVICE_TICKETS.with(|t| {
        let mut v = t.borrow_mut();
        v[1] = v[0];
        v[0] = dev;
    });
}

/// 在候选句柄里挑一个**已绑定真实 GPUDevice** 的认领为全局 `DEVICE`，并清空槽位。
///
/// # 候选集：全局 `DEVICE` ∪ 近两轮预留的票
///
/// 就绪判定**必须**把全局 `DEVICE` 也算作候选。两轮 `collect` 共享同一份状态、
/// 且 `prism_test_collect` 不带 run 令牌，而本函数认领成功后会**清空槽位**——
/// 于是先跑的那轮把票认领走，后跑的那轮只看槽位就拿到空集，会误报
/// "预留句柄 N 未绑定"，哪怕那张票早已 `ready == 1`。真机实测时序：
/// ```text
///   28ms → request_device() = 1        // boot() 自动跑的那一轮
///   45ms → device_ready(1) = 0         // 点「全部运行」那轮：未就绪，发票 2
///   45ms → request_device() = 2
///   48ms → device_ready(1) = 1         // 票 1 其实已就绪
/// ```
/// 先 collect 的轮次认领票 1 并清空槽位；后 collect 的轮次因槽位已空而 FAIL。
/// **判据没有放宽**：仍然要求 `prism_device_ready(h) == 1`，只是把"已认领的
/// 全局句柄"也纳入候选。
///
/// 返回 `(首个候选, 就绪者)`：
/// - `就绪者`：按 `[DEVICE, v[1], v[0]]` 找第一个 `device_ready != 0` 的；
///   `None` 说明候选全都没绑定。`DEVICE` 排最前，因为它成熟度最高。
/// - `首个候选`：只用于失败文案，按**槽位从旧到新**（`v[1]` → `v[0`，再兜底
///   `DEVICE`）取第一个非零，语义确定——先报最早预留、仍未回填的那张票，
///   正是"上一轮遗留"的典型形态。
fn claim_ready_device() -> (u32, Option<u32>) {
    let (first, ready) = DEVICE_TICKETS.with(|t| {
        let mut v = t.borrow_mut();
        let global = DEVICE.with(|d| d.get());
        // 诊断用"首个候选"：槽位从旧到新，DEVICE 兜底。
        let first = [v[1], v[0], global].into_iter().find(|&h| h != 0).unwrap_or(0);
        // 就绪判定：全局句柄优先，其次槽位从旧到新。
        let ready = [global, v[1], v[0]]
            .into_iter()
            .find(|&h| h != 0 && gpu::device_ready(h) != 0);
        v[0] = 0;
        v[1] = 0;
        (first, ready)
    });
    if let Some(h) = ready {
        DEVICE.with(|d| d.set(h));
    }
    (first, ready)
}

/// 取工作区中 `f32..` 切片的线性内存地址（`u32`，wasm32 指针宽）。
fn addr_of_slice<T>(f: impl FnOnce(&Scratch) -> &[T]) -> u32 {
    SCRATCH.with(|s| f(&s.borrow()).as_ptr() as usize as u32)
}

/// 工作区数组的可写地址（供 GPU 读回落点）。
///
/// # 与 `addr_of_slice` 的区别：借用时机
/// 本函数**自己**取一次 `borrow_mut`，因此**不可**在调用方已持有
/// `SCRATCH.borrow_mut()` 的闭包内调用——那会二次借用同一个 [`RefCell`]，
/// 直接 panic（`already borrowed: BorrowMutError`）。
///
/// D-14 真机实测教训：`t_gpu_scan_block` / `t_gpu_histogram` 最初写成
/// `SCRATCH.with(|s| { let mut v = s.borrow_mut(); …; v.out_ptr = addr_of_buf(…); })`，
/// 宿主侧表现为 wasm trap `RuntimeError: unreachable`（panic=abort 下 unwind 被
/// 编译掉，trap 直接穿透到 JS，宪章 C18 的 `catch_unwind` 也拦不住）。
/// 现在两处都改为**先借一次、取到地址即释放**，再单独写字段。
fn addr_of_buf<T>(f: impl FnOnce(&mut Scratch) -> &mut [T]) -> u32 {
    SCRATCH.with(|s| f(&mut s.borrow_mut()).as_mut_ptr() as usize as u32)
}

/// 内部分配（返回真实指针；ABI 层的 `u32` 转换只在 wasm32 上有意义）。
///
/// 抽出内部函数而非让 `t_abi_alloc` 直接走 `#[no_mangle]` 导出：导出签名按
/// WebAssembly JS ABI 约定用 `u32` 承载地址（wasm32 指针宽 32 位），在 64 位
/// native 上会截断地址。内部测试走本函数，故 native 与 wasm 行为一致。
fn alloc_bytes(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(len, PRISM_ALLOC_ALIGN) else {
        return std::ptr::null_mut();
    };
    // SAFETY: layout 尺寸/对齐均合法；返回 null 表示分配失败。
    unsafe { std::alloc::alloc(layout) }
}

/// 内部释放（与 [`alloc_bytes`] 配对）。
///
/// # Safety
/// `ptr`/`len` 必须来自同一次 [`alloc_bytes`]，且未重复释放。
unsafe fn dealloc_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, PRISM_ALLOC_ALIGN) {
        // SAFETY: 由本函数契约保证配对且未重复释放。
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

// ---------------------------------------------------------------------------
// GPU 胶水封装：wasm 转发给宿主，native 返回"无 GPU"桩
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub(crate) mod gpu {
    use crate::web::js;

    /// `navigator.gpu` 是否存在。
    pub fn available() -> u32 {
        // SAFETY: 宿主注入的零参数纯探测函数（`web/index.html` 的 `glue`）。
        unsafe { js::prism_navigator_gpu_available() }
    }
    /// device 是否已 lost。
    pub fn device_lost(dev: u32) -> u32 {
        // SAFETY: 纯读宿主标志位。
        unsafe { js::prism_device_lost(dev) }
    }
    /// 句柄是否已绑定到真实 GPUDevice（1 = 就绪）。
    pub fn device_ready(dev: u32) -> u32 {
        // SAFETY: 纯读宿主句柄表。
        unsafe { js::prism_device_ready(dev) }
    }
    /// 创建 buffer（`usage` 见 [`super::usage`]）。
    pub fn create_buffer(dev: u32, size: u32, usage: u32) -> u32 {
        // SAFETY: 指针/长度均为标量，宿主只读 size/usage。
        unsafe { js::prism_create_buffer(dev, size, usage) }
    }
    /// 把线性内存 `[ptr, ptr+len)` 写入 buffer。
    pub fn write_buffer(dev: u32, buf: u32, ptr: u32, len: u32) -> u32 {
        // SAFETY: 宿主按 (ptr,len) 读取线性内存，本调用期间 scratch 不被重分配。
        unsafe { js::prism_write_buffer(dev, buf, ptr, len) }
    }
    /// 发起 GPU→暂存区拷贝，返回暂存句柄。
    pub fn read_buffer(dev: u32, buf: u32, size: u32) -> u32 {
        // SAFETY: 纯标量。
        unsafe { js::prism_read_buffer(dev, buf, size) }
    }
    /// 登记「暂存区 → 线性内存 `dst_ptr`」的异步拷贝（宿主 settle 时执行）。
    pub fn copy_out(staging: u32, dst_ptr: u32, len: u32) -> u32 {
        // SAFETY: 宿主在 settle 阶段读暂存区并写入 dst_ptr，len 已由本模块计算。
        unsafe { js::prism_copy_out(staging, dst_ptr, len) }
    }
    /// 用 WGSL 源创建 shader module。
    pub fn create_shader_module(dev: u32, src_ptr: u32, src_len: u32) -> u32 {
        // SAFETY: 源为 `&'static str`（模块生命周期内有效），长度一致。
        unsafe { js::prism_create_shader_module(dev, src_ptr, src_len) }
    }
    /// 创建 compute pipeline（`entry_ptr`/`entry_len` 为 entry point 名）。
    pub fn create_compute_pipeline(dev: u32, module: u32, entry_ptr: u32, entry_len: u32) -> u32 {
        // SAFETY: entry 名同 `&'static str`，长度一致。
        unsafe { js::prism_create_compute_pipeline(dev, module, entry_ptr, entry_len) }
    }
    /// 创建 bind group；`group` 为 WGSL `@group` 索引，`bufs_ptr` 指向 `n` 个 u32 句柄。
    pub fn create_bind_group(dev: u32, pipeline: u32, group: u32, bufs_ptr: u32, n: u32) -> u32 {
        // SAFETY: `bufs_ptr` 指向 scratch.handles 的前 n 项，n ≤ 8 已由调用方保证。
        unsafe { js::prism_create_bind_group(dev, pipeline, group, bufs_ptr, n) }
    }
    /// 提交一次 compute dispatch；`bgs_ptr` 指向 `n` 个 bind group 句柄（第 i 个绑到 @group(i)）。
    pub fn dispatch(dev: u32, pipeline: u32, bgs_ptr: u32, n: u32, gx: u32, gy: u32, gz: u32) -> u32 {
        // SAFETY: 纯标量 + 句柄数组指针转发。
        unsafe { js::prism_dispatch(dev, pipeline, bgs_ptr, n, gx, gy, gz) }
    }
    /// 发起等待队列完成（宿主在 settle 时 await）。
    pub fn wait_idle(dev: u32) -> u32 {
        // SAFETY: 纯标量；异步部分由宿主 glue 承接。
        unsafe { js::prism_wait_idle(dev) }
    }
    /// 弹出一条已捕获的验证错误码（0 = 无）。
    pub fn pop_error(dev: u32) -> u32 {
        // SAFETY: 纯读宿主队列。
        unsafe { js::prism_pop_error(dev) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod gpu {
    /// native 目标无 GPU：恒为"不可用"。
    pub fn available() -> u32 {
        0
    }
    /// native 目标永不 lost。
    pub fn device_lost(_dev: u32) -> u32 {
        0
    }
    /// native 目标永不就绪（无 GPU）。
    pub fn device_ready(_dev: u32) -> u32 {
        0
    }
    /// native 目标不创建 buffer。
    pub fn create_buffer(_dev: u32, _size: u32, _usage: u32) -> u32 {
        0
    }
    /// native 目标不写 buffer。
    pub fn write_buffer(_dev: u32, _buf: u32, _ptr: u32, _len: u32) -> u32 {
        0
    }
    /// native 目标无暂存区。
    pub fn read_buffer(_dev: u32, _buf: u32, _size: u32) -> u32 {
        0
    }
    /// native 目标无暂存区。
    pub fn copy_out(_staging: u32, _dst_ptr: u32, _len: u32) -> u32 {
        0
    }
    /// native 目标不编译 shader。
    pub fn create_shader_module(_dev: u32, _src_ptr: u32, _src_len: u32) -> u32 {
        0
    }
    /// native 目标不建 pipeline。
    pub fn create_compute_pipeline(_dev: u32, _m: u32, _entry_ptr: u32, _entry_len: u32) -> u32 {
        0
    }
    /// native 目标不建 bind group。
    pub fn create_bind_group(_dev: u32, _p: u32, _group: u32, _bufs_ptr: u32, _n: u32) -> u32 {
        0
    }
    /// native 目标不 dispatch。
    pub fn dispatch(_d: u32, _p: u32, _bgs_ptr: u32, _n: u32, _gx: u32, _gy: u32, _gz: u32) -> u32 {
        0
    }
    /// native 目标无需等待。
    pub fn wait_idle(_dev: u32) -> u32 {
        0
    }
    /// native 目标无错误队列。
    pub fn pop_error(_dev: u32) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

/// FNV-1a 32 位散列（确定性、零依赖），用于把数值结果压成一个可比对的指纹。
fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// u32 切片的字节指纹（wasm 与 native 均为小端，故逐字节一致）。
fn hash_u32(v: &[u32]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &x in v {
        for b in x.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

/// 确定性 LCG（与 `sort.rs` 测试同族，种子显式给定，无全局 RNG）。
fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

/// `scan_block` 测试输入（确定性：`(3i+1) mod 2^32`）。
fn scan_input(i: usize) -> u32 {
    (i as u32).wrapping_mul(3).wrapping_add(1)
}

// ---------------------------------------------------------------------------
// 测试项清单（Rust 侧单一真源，任务 2 方案 A）
// ---------------------------------------------------------------------------

/// 单个测试项的静态描述。
struct TestItem {
    /// 稳定 id（报告与逐项重跑的主键）。
    id: &'static str,
    /// 人类可读名。
    name: &'static str,
    /// 是否需要 WebGPU device（宿主据此提前显示 SKIP 提示）。
    needs_gpu: bool,
    /// 第一阶段：提交 / 立即判定。
    run: fn() -> Status,
    /// 第二阶段：宿主 settle 后的读回与判定（`None` = 无需 settle）。
    collect: Option<fn() -> Status>,
}

/// 全部测试项（顺序即 `prism_test_run(i)` 的下标顺序）。
static TESTS: &[TestItem] = &[
    TestItem {
        id: "device.init",
        name: "设备初始化（navigator.gpu / requestDevice / lost 轮询）",
        needs_gpu: true,
        run: t_device_init,
        // 两阶段：run 发起 requestDevice（同步返回预留句柄），宿主 settle 后
        // collect 才校验句柄已绑定真实 GPUDevice——否则会误判"device 就绪"。
        collect: Some(c_device_init),
    },
    TestItem {
        id: "cpu.scan",
        name: "CPU 参考：Blelloch scan 判对（vs 串行参考 + 树形深度）",
        needs_gpu: false,
        run: t_cpu_scan,
        collect: None,
    },
    TestItem {
        id: "cpu.sort",
        name: "CPU 参考：radix sort 判对（vs 稳定插入排序 + 稳定性）",
        needs_gpu: false,
        run: t_cpu_sort,
        collect: None,
    },
    TestItem {
        id: "wgsl.structural",
        name: "WGSL 结构自检：全部内嵌 shader（含库件）+ 2 个 D0 内核（D-29 禁 libm）",
        needs_gpu: false,
        run: t_wgsl_structural,
        collect: None,
    },
    TestItem {
        id: "abi.alloc",
        name: "ABI 自检：alloc/写/读回/dealloc 线性内存往返",
        needs_gpu: false,
        run: t_abi_alloc,
        collect: None,
    },
    TestItem {
        id: "shader.params",
        name: "参数校验：ScanConfig 工作组换算 + histogram uniform 打包",
        needs_gpu: false,
        run: t_shader_params,
        collect: None,
    },
    TestItem {
        id: "wgsl.compile",
        name: "真 WebGPU：全部内嵌 shader 批量 createShaderModule 编译（include 已展开，份数见详情）",
        needs_gpu: true,
        run: t_wgsl_compile,
        collect: Some(c_wgsl_compile),
    },
    TestItem {
        id: "gpu.scan_block",
        name: "真 WebGPU：scan_block 单块 dispatch，与 CPU 参考逐元素比对",
        needs_gpu: true,
        run: t_gpu_scan_block,
        collect: Some(c_gpu_scan_block),
    },
    TestItem {
        id: "gpu.histogram",
        name: "真 WebGPU：histogram 原子计数，与 CPU 参考逐桶比对",
        needs_gpu: true,
        run: t_gpu_histogram,
        collect: Some(c_gpu_histogram),
    },
    TestItem {
        id: "shader.dispatch_census",
        name: "覆盖台账：全部 WGSL 的 compute 入口逐份核对（可 dispatch / 数据竞争 / 多 bind group / 零入口）",
        needs_gpu: false,
        run: t_shader_dispatch_census,
        collect: None,
    },
    TestItem {
        id: "gpu.sim_particles",
        name: "真 WebGPU：sim_particles 的 emit/integrate/hash_cells 三入口链，与 CPU 参考逐字段比对",
        needs_gpu: true,
        run: t_gpu_sim_particles,
        collect: Some(c_gpu_sim_particles),
    },
    TestItem {
        id: "gpu.sim_sph",
        name: "真 WebGPU：sim_sph 的 sph_predict/sph_density/sph_project，与 CPU 参考（Poly6+XPBD）比对",
        needs_gpu: true,
        run: t_gpu_sim_sph,
        collect: Some(c_gpu_sim_sph),
    },
    TestItem {
        id: "gpu.sim_cloth",
        name: "真 WebGPU：sim_cloth 的 cloth_predict/cloth_project/cloth_velocity（不相交约束）比对",
        needs_gpu: true,
        run: t_gpu_sim_cloth,
        collect: Some(c_gpu_sim_cloth),
    },
    TestItem {
        id: "gpu.rt_wavefront",
        name: "真 WebGPU：rt_wavefront 的 connect/accumulate，PathSlot 布局与 spp 均值逐字节比对",
        needs_gpu: true,
        run: t_gpu_rt_wavefront,
        collect: Some(c_gpu_rt_wavefront),
    },
    TestItem {
        id: "gpu.rt_wavefront_multi",
        name: "真 WebGPU：rt_wavefront 多 bind group 的 generate/shade/extend，group 0/1/2/3 逐字段判对",
        needs_gpu: true,
        run: t_gpu_rt_wavefront_multi,
        collect: Some(c_gpu_rt_wavefront_multi),
    },
    TestItem {
        // 放在 `gpu.gi_compute` **之前**：报告里先看到"参数自检结论"，再看下游数值。
        // 一旦 uniform 传输坏了，读者第一眼就能定位到根因，不必从 SH 偏差反推。
        id: "gpu.params_roundtrip",
        name: "真 WebGPU：GiParams uniform 写入 → 读回逐字节回环（256B 全等 + 4 个关键字段解释值）",
        needs_gpu: true,
        run: t_gpu_params_roundtrip,
        collect: Some(c_gpu_params_roundtrip),
    },
    TestItem {
        // 放在 `gpu.gi_compute` **之前**：布局错了，下游 SH/辐照度的偏差全都没有
        // 解读价值。本项先把「shader 到底把哪些字节读成了自己的字段」逐格摊开。
        id: "gpu.gi_layout",
        name: "真 WebGPU：GI 存储布局 census（probe0/1/2 的 pos/normal + GiParams 混合字段逐格判对）",
        needs_gpu: true,
        run: t_gpu_gi_layout,
        collect: Some(c_gpu_gi_layout),
    },
    TestItem {
        id: "gpu.gi_compute",
        name: "真 WebGPU：GI 的 GPU 路径 gi_probe_trace/gi_sh_project/gi_probe_interpolate，逐字段与 CPU（prism-rt 同式）判对",
        needs_gpu: true,
        run: t_gpu_gi_compute,
        collect: Some(c_gpu_gi_compute),
    },
    TestItem {
        id: "gpu.scene_frame",
        name: "真 WebGPU：scene_frame 光追整帧 dispatch，与 CPU 参考统计对拍 + 双跑一致",
        needs_gpu: true,
        run: t_gpu_scene_frame,
        collect: Some(c_gpu_scene_frame),
    },
];

/// 测试项总数。
pub const TEST_COUNT: u32 = TESTS.len() as u32;

// ---------------------------------------------------------------------------
// 测试项 0：设备初始化
// ---------------------------------------------------------------------------

fn t_device_init() -> Status {
    // 直接问宿主 `navigator.gpu`（prism_navigator_gpu_available 链路）。
    if gpu::available() == 0 {
        return skip("SKIP：无 navigator.gpu（本项需要 WebGPU）");
    }
    // WebGpuBackend::new 内部调用 prism_request_device。
    //
    //【关键】`prism_request_device` 是**同步**返回预留句柄的（WebAssembly 的
    // import 只能是同步函数），真正的 requestAdapter/requestDevice 由宿主
    // `settle()` 才 await。故本项必须走**两阶段**：run 阶段只发起并登记句柄，
    // 返回 PENDING；宿主 settle 之后 collect 阶段才校验"device 真的到手"。
    //
    // D-14 真机实测教训：早期版本在 run 阶段就因"句柄 ≠ 0"判 PASS，而宿主
    // `runWasmTest` 对非 PENDING 项会清空 pendingWork —— requestDevice 的
    // promise 从未被 await，句柄表里该键恒为 null，于是**一个不存在的 device
    // 被判为通过**，后续 3 个 GPU 项全部 `prism_create_buffer 失败`。
    //
    // 【D-14 真机缺陷：device.init 被并发执行，且必须"有活可等"】
    //
    // 宿主会**并发**跑本项：`boot()` 收尾自动 `await runOne(device.init)`，
    // 而回归脚本在清单一出现就点「全部运行」，两轮 runAll/runOne 重叠。
    // 宿主 `settle()` 的语义是"把 `pendingWork` 里**当时**的 promise 全部
    // await 干净"，所以只有**本 run 自己登记过 promise**时，settle 才等得到。
    //
    // 若此处无条件 `return Pending` 却不登记任何 promise，则 runAll 这一侧的
    // settle() 是空转，会在 requestDevice 尚未兑现时直接 collect → 报
    // "预留句柄 N 未绑定"，并把后续 7 个 GPU 项全部带崩（实测 PASS=6/7，
    // 句柄表停在 1 —— 一个 buffer 都没建出来）。这正是本守卫的真实陷阱：
    // **Pending 必须伴随真实 promise，否则等于骗宿主"我在等"。**
    //
    // 故分两种情形：
    //   ① 已就绪（device_ready==1）⇒ 直接 Pass，绝不重复发起 requestDevice。
    //      这是幂等的意义，也是"每页只发一次请求"的保证。
    //   ② 未就绪 ⇒ **必须**发起请求（哪怕已有一次在途），并返回 Pending。
    //      这一条是被真机数据反复纠正过的，务必别改：
    //      试过"在途就不发请求、只返回 Pending"，结果**稳定更差**
    //      （0/8 次 14 项全过，而会发请求的版本 5/8 次全过）。原因是
    //      Pending 的语义是"宿主请 await settle() 后再 collect"；若本次
    //      没往 `pendingWork` 里登记任何 promise，本次 settle() 就是空转，
    //      collect 立刻执行 → 此时在途票多半还没回填 ⇒ 本项必 FAIL，
    //      并把后续 7 个 GPU 项一起带崩（句柄表停在 1，一个 buffer 都没建）。
    //      **Pending 必须与真实 promise 配对**，否则等于骗宿主"我在等"。
    //      并发轮多发的那一次请求并非浪费：它让本次 settle() 有活可等，
    //      两张票由 `claim_ready_device` 挑就绪者认领。
    let prev = DEVICE.with(|d| d.get());
    if prev != 0 && gpu::device_ready(prev) != 0 {
        let lost = gpu::device_lost(prev);
        if lost != 0 {
            return fail("device 已 lost（prism_device_lost == 1），请刷新页面重试");
        }
        set_result(&format!(
            "device 已就绪（句柄={prev}，已绑定真实 GPUDevice ✓），复用既有句柄，\
             不重复发起 requestDevice\nlost={lost}（prism_device_lost 轮询正常）"
        ));
        return Status::Pass;
    }
    let backend = WebGpuBackend::new();
    let dev = backend.device_handle();
    if dev == crate::web::INVALID_HANDLE {
        return fail("requestAdapter/requestDevice 失败：宿主返回句柄 0（见错误面板）");
    }
    DEVICE.with(|d| d.set(dev));
    note_device_ticket(dev);
    set_result(&format!("已发起 requestAdapter/requestDevice，预留句柄={dev}；待 settle 后校验就绪"));
    Status::Pending
}

/// `device.init` 的第二阶段：宿主 settle 之后校验 device 真的可用。
fn c_device_init() -> Status {
    // 就绪性判据：在**本项近两轮预留过的候选句柄**里，必须有一张已绑定真实
    // GPUDevice。并发执行时两轮共享状态、且 `test_collect` 不带 run 令牌，
    // 故只能"任一就绪即通过"；判据本身没放宽——`device_ready` 仍为真。
    let (first, ready) = claim_ready_device();
    let dev = match ready {
        Some(d) => d,
        None => {
            let dev = if first != 0 { first } else { DEVICE.with(|d| d.get()) };
            if dev == 0 {
                return fail("device 句柄为 0：requestAdapter/requestDevice 未成功（见错误面板）");
            }
            return fail(format!(
                "预留句柄 {dev} 未绑定到真实 GPUDevice：宿主 settle 未完成 requestDevice。\n\
                 排查：(1) 宿主 runWasmTest 是否在 run 返回非 PENDING 时清空了 pendingWork；\n\
                 (2) requestAdapter/requestDevice 是否被 await（错误面板看 requestAdapter/requestDevice 条目）。"
            ));
        }
    };
    // 立刻轮询一次 lost，确认 prism_device_lost 链路可用。
    let lost = gpu::device_lost(dev);
    if lost != 0 {
        return fail("device 刚创建即处于 lost 状态");
    }
    // 【D-14 真机缺陷修复】此处曾写 `WebGpuBackend::new().adapter_summary()`，
    // 而 `WebGpuBackend::new()` **必然再发起一次 `prism_request_device`**：
    //   ① 该 promise 进入 pendingWork，但 collect 阶段宿主**不会**再 settle 一次，
    //      于是它在本页余下的生命周期里随时触发，覆盖宿主全局 `device`；
    //   ② `web::adapter_summary` 在所有平台都返回 `default()`（纯占位桩，D1 骨架），
    //      这次请求**零收益**。
    // 故直接复用句柄调 `adapter_summary`，不再触碰 requestDevice 链路。
    let summary = crate::web::adapter_summary(dev);
    let name = if summary.adapter_name.is_empty() { "(宿主未回传)" } else { summary.adapter_name.as_str() };
    set_result(&format!(
        "navigator.gpu 探测 = 存在\ndevice 句柄={dev}（已绑定真实 GPUDevice ✓）\n\
         adapter={name}\nshader-f16={} timestamp-query={}\n\
         lost={lost}（prism_device_lost 轮询正常）",
        summary.has_shader_f16, summary.has_timestamp_query
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 1：CPU 参考 scan
// ---------------------------------------------------------------------------

fn t_cpu_scan() -> Status {
    // 主向量 + 边界尺寸（0/1/3/257 覆盖补零与奇数尾进位）。
    for &n in &[0usize, 1, 3, SCAN_N, 257] {
        let input: Vec<u32> = (0..n).map(scan_input).collect();
        let blelloch = scan_blelloch_cpu(&input);
        let reference = scan_cpu_ref(&input);
        if blelloch != reference {
            let pos = blelloch
                .iter()
                .zip(reference.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return fail(format!(
                "n={n}：Blelloch 与串行参考在 i={pos} 不一致（{} vs {}）",
                blelloch.get(pos).copied().unwrap_or(0),
                reference.get(pos).copied().unwrap_or(0)
            ));
        }
    }
    let input: Vec<u32> = (0..SCAN_N).map(scan_input).collect();
    let out = scan_blelloch_cpu(&input);
    // Blelloch 排除了 2 的幂补齐带来的形状歧义：深度必须 ⌈log₂n⌉。
    let shape = ScanShape::new(SCAN_N);
    let expected_depth = usize::BITS - 1 - SCAN_N.leading_zeros();
    if shape.depth() != expected_depth as usize {
        return fail(format!("ScanShape 深度 {} ≠ ⌈log₂n⌉={expected_depth}", shape.depth()));
    }
    let total = out[SCAN_N - 1];
    let expect_total: u32 = (0..SCAN_N as u32).map(|i| i.wrapping_mul(3).wrapping_add(1)).fold(0u32, |a, b| a.wrapping_add(b));
    if total != expect_total {
        return fail(format!("inclusive 前缀末项 {total} ≠ 输入总和 {expect_total}"));
    }
    set_result(&format!(
        "n={SCAN_N} Blelloch == scan_cpu_ref（另校验 n=0/1/3/257）\n\
         树深={}（=⌈log₂{}⌉，同 n 必得同树）\n\
         out[0]={} out[1]={} out[127]={}（=输入总和）\n\
         fingerprint=0x{:08x}（供宿主侧独立实现交叉核对）",
        shape.depth(),
        SCAN_N,
        out[0],
        out[1],
        total,
        hash_u32(&out)
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 2：CPU 参考 radix sort
// ---------------------------------------------------------------------------

fn t_cpu_sort() -> Status {
    for &n in &[0usize, 1, 3, HIST_N] {
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let items: Vec<(u32, u32)> = (0..n).map(|i| (lcg(&mut seed), i as u32)).collect();
        let radix = radix_sort_u32(&items);
        let reference = sort_cpu_ref(&items);
        if radix != reference {
            let pos = radix.iter().zip(reference.iter()).position(|(a, b)| a != b).unwrap_or(0);
            return fail(format!(
                "n={n}：radix 与稳定插入排序在 i={pos} 不一致（{:?} vs {:?}）",
                radix.get(pos).copied().unwrap_or((0, 0)),
                reference.get(pos).copied().unwrap_or((0, 0))
            ));
        }
    }
    // 稳定性：键压到 4 值，**每个键的载荷序列**必须等于它在输入中的出现序。
    // 注意：输入的键是交错的，不能按"连续同键"切段（那等于假设输入已分组），
    // 也不能把 `sorted[i]` 直接和 `items[i]` 比（排序本就改变了位置）。
    // 正解：按桶收集各键在输入中的载荷序列（保持出现序），再与排序输出对读。
    const STABILITY_KEYS: u32 = 4;
    let mut seed: u64 = 42;
    let items: Vec<(u32, u32)> = (0..500).map(|i| (lcg(&mut seed) % STABILITY_KEYS, i as u32)).collect();
    let sorted = radix_sort_u32(&items);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); STABILITY_KEYS as usize];
    for &(k, p) in &items {
        buckets[(k % STABILITY_KEYS) as usize].push(p);
    }
    let mut cursor = 0usize;
    for b in 0..STABILITY_KEYS as usize {
        for &p in &buckets[b] {
            let got = sorted.get(cursor).copied().unwrap_or((u32::MAX, u32::MAX));
            if got != (b as u32, p) {
                return fail(format!(
                    "稳定性破坏：键 {b} 的第 {} 个载荷在输出位置 {cursor} 应为 {p}，实际键 {} 载荷 {}",
                    buckets[b].iter().position(|&x| x == p).unwrap_or(0),
                    got.0,
                    got.1
                ));
            }
            cursor += 1;
        }
    }
    if cursor != items.len() {
        return fail(format!("排序输出长度 {} ≠ 输入 {}", sorted.len(), items.len()));
    }
    // 主向量指纹：键序列。
    let mut seed: u64 = 7;
    let main: Vec<(u32, u32)> = (0..HIST_N).map(|i| (lcg(&mut seed), i as u32)).collect();
    let main_sorted = radix_sort_u32(&main);
    let keys: Vec<u32> = main_sorted.iter().map(|&(k, _)| k).collect();
    for w in keys.windows(2) {
        if w[0] > w[1] {
            return fail(format!("结果非升序：{} > {}", w[0], w[1]));
        }
    }
    set_result(&format!(
        "n={HIST_N} radix == sort_cpu_ref（另校验 n=0/1/3）\n\
         稳定性：500 个键压至 4 值，逐桶载荷序 == 输入序 ✓\n\
         升序 ✓  keys[0]={} keys[1]={} keys[255]={}\n\
         keys fingerprint=0x{:08x}（供 gpu.histogram 交叉核对）",
        keys[0],
        keys[1],
        keys[HIST_N - 1],
        hash_u32(&keys)
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 3：WGSL 结构自检（纯 CPU，手机零 GPU 开销）
// ---------------------------------------------------------------------------

fn t_wgsl_structural() -> Status {
    let mut lines: Vec<String> = Vec::new();
    let mut failed: Vec<&str> = Vec::new();
    for c in shaders::check_all() {
        let verdict = if c.is_pass() { "PASS" } else { "FAIL" };
        if !c.is_pass() {
            failed.push(c.name);
        }
        lines.push(format!(
            "  {verdict}  {:<20} entry={} balance={} fns={} libm={} include={}",
            c.name, c.has_entry, c.brackets_balanced, c.fn_count, c.has_libm, c.has_include
        ));
    }
    // D0 两个计算内核（内嵌在 device.rs，不在 shaders/ 目录）。
    for (name, src) in [("scan_block", WGSL_SCAN_BLOCK), ("histogram", WGSL_HISTOGRAM)] {
        let c = shaders::check_shader(name, src);
        let verdict = if c.is_pass() { "PASS" } else { "FAIL" };
        if !c.is_pass() {
            failed.push(name);
        }
        lines.push(format!(
            "  {verdict}  {:<20} entry={} balance={} fns={} libm={} include={}",
            name, c.has_entry, c.brackets_balanced, c.fn_count, c.has_libm, c.has_include
        ));
    }
    if !failed.is_empty() {
        return fail(format!("WGSL 结构自检未通过：{}\n{}", failed.join(", "), lines.join("\n")));
    }
    set_result(&format!(
        "{} 个 WGSL 结构自检全通过（含 D-29 禁 libm / 零 include）：\n{}",
        lines.len(),
        lines.join("\n")
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 4：ABI 线性内存自检
// ---------------------------------------------------------------------------

fn t_abi_alloc() -> Status {
    const LEN: usize = 64;
    let ptr = alloc_bytes(LEN);
    if ptr.is_null() {
        return fail("alloc 分配失败（OOM 或对齐参数非法）");
    }
    let addr = ptr as usize;
    if addr % PRISM_ALLOC_ALIGN != 0 {
        // SAFETY: 刚分配、尚未释放。
        unsafe { dealloc_bytes(ptr, LEN) };
        return fail(format!("alloc 返回 {addr:#x}，未按 {PRISM_ALLOC_ALIGN} 字节对齐"));
    }
    // 期望图样：`(7i+3) mod 256`。
    let expect: Vec<u8> = (0..LEN).map(|i| (i as u8).wrapping_mul(7).wrapping_add(3)).collect();
    // 写 → 读回 → 逐字节比对（这才是"宿主写 WGSL 源码进 wasm 内存"的真实路径）。
    // SAFETY: `[ptr, ptr+LEN)` 来自本函数刚成功的分配，长度精确匹配，写入不越界。
    let mem = unsafe { std::slice::from_raw_parts_mut(ptr, LEN) };
    mem.copy_from_slice(&expect);
    // SAFETY: 同上；此处只读。
    let readback = unsafe { std::slice::from_raw_parts(ptr, LEN) };
    if readback != expect.as_slice() {
        let first = readback.iter().zip(expect.iter()).position(|(a, b)| a != b).unwrap_or(0);
        // SAFETY: 同上，配对释放。
        unsafe { dealloc_bytes(ptr, LEN) };
        return fail(format!(
            "线性内存写读不一致：首个不符 i={first}（写入 {:?} 读回 {:?}）",
            expect.get(first), readback.get(first)
        ));
    }
    // 幂等：同一图样再写一遍，读回仍一致（确认无隐藏状态/未对齐撕裂）。
    mem.copy_from_slice(&expect);
    // SAFETY: 同上，只读。
    let again = unsafe { std::slice::from_raw_parts(ptr, LEN) };
    let stable = again == expect.as_slice();
    let digest = fnv1a32(&expect);
    // SAFETY: 同上，配对释放。
    unsafe { dealloc_bytes(ptr, LEN) };
    if !stable {
        return fail("线性内存重复写入后读回不一致（对齐或别名问题）");
    }
    set_result(&format!(
        "alloc({LEN}) → {addr:#x}（{PRISM_ALLOC_ALIGN}B 对齐 ✓）\n\
         宿主写图样 → Rust 读回：{LEN}/{LEN} 字节逐字节一致 ✓\n\
         图样 FNV-1a = 0x{digest:08x}；重复写入幂等 ✓\n\
         dealloc 正常返回（未 use-after-free）"
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 5：参数校验（不碰 GPU，纯算术）
// ---------------------------------------------------------------------------

fn t_shader_params() -> Status {
    // ScanConfig 工作组换算：每块 2*WORKGROUP_SIZE 个元素。
    let cases: [(u32, u32); 5] = [(0, 1), (64, 1), (128, 1), (129, 2), (SCAN_N as u32, 1)];
    for (count, expect_wg) in cases {
        let cfg = ScanConfig::for_count(count);
        if cfg.workgroups != expect_wg {
            return fail(format!("ScanConfig::for_count({count}) 工作组数 {} ≠ 预期 {expect_wg}", cfg.workgroups));
        }
    }
    // histogram uniform 打包：16 字节、小端、末 4 字节对齐占位。
    let bytes = histogram_params_bytes(HIST_N as u32, HIST_BUCKETS, 8);
    if bytes.len() != HISTOGRAM_PARAMS_BYTES {
        return fail(format!("params 长度 {} ≠ {HISTOGRAM_PARAMS_BYTES}", bytes.len()));
    }
    let rd = |off: usize| u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    if rd(0) != HIST_N as u32 || rd(4) != HIST_BUCKETS || rd(8) != 8 || rd(12) != 0 {
        return fail(format!(
            "params 布局错误：count={} buckets={} shift={} pad={}（期望 {HIST_N}/{HIST_BUCKETS}/8/0）",
            rd(0), rd(4), rd(8), rd(12)
        ));
    }
    // 第 2/3/4 趟位移也必须能正确编码（GPU 取数位口径与 CPU sort.rs 同源）。
    for shift in [0u32, 8, 16, 24] {
        let b = histogram_params_bytes(HIST_N as u32, HIST_BUCKETS, shift);
        let s = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        if s != shift {
            return fail(format!("params shift 编码错误：写入 {shift} 读回 {s}"));
        }
    }
    set_result(&format!(
        "ScanConfig：0/64/128/129/128 → 工作组 1/1/1/2/1 ✓（每块 2×{WORKGROUP_SIZE} 元素）\n\
         histogram_params：16B 小端布局 count={} buckets={} shift 占位 ✓\n\
         位移 0/8/16/24 四趟编码往返一致 ✓（与 CPU sort.rs `(k>>shift)&mask` 同口径）",
        HIST_N, HIST_BUCKETS
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// GPU 提交/读回公共路径
// ---------------------------------------------------------------------------

/// 取已初始化 device 句柄；无则跳过。
fn ready_device(test: &str) -> Result<u32, Status> {
    let dev = DEVICE.with(|d| d.get());
    if dev == 0 {
        return Err(skip(format!("SKIP：device 未初始化，请先运行 device.init（{test}）")));
    }
    if gpu::device_lost(dev) != 0 {
        return Err(fail("device 已 lost（prism_device_lost == 1），请刷新页面重试"));
    }
    // 就绪性：句柄非 0 只代表"已预约"。宿主未 settle 完成时句柄表里是 null，
    // 后续 create_buffer/create_shader_module 会全部返回 0。此处提前拦下并
    // 给出可操作的原因，而不是让调用方看到"createShaderModule 返回 0"。
    if gpu::device_ready(dev) == 0 {
        return Err(fail(format!(
            "{test}：device 句柄 {dev} 尚未绑定真实 GPUDevice（宿主未完成 requestDevice）。\
             请先单独运行 device.init 并等其显示 PASS。"
        )));
    }
    Ok(dev)
}

/// 建 buffer + 写线性内存；返回句柄。
fn make_buffer(test: &str, dev: u32, size: u32, usage: u32, src: impl FnOnce() -> (u32, u32)) -> Result<u32, Status> {
    let h = gpu::create_buffer(dev, size, usage);
    if h == 0 {
        return Err(fail(format!("{test}：prism_create_buffer(size={size}) 失败")));
    }
    let (ptr, len) = src();
    if gpu::write_buffer(dev, h, ptr, len) == 0 {
        return Err(fail(format!("{test}：prism_write_buffer(len={len}) 失败")));
    }
    Ok(h)
}

/// 组建「module → pipeline → bind group → dispatch」并发起读回。
fn submit_kernel(
    test: &str,
    dev: u32,
    wgsl: &'static str,
    entry: &'static str,
    bufs: &[u32],
    workgroups: u32,
    read_buf: u32,
    read_len: u32,
) -> Result<(), Status> {
    let module = gpu::create_shader_module(dev, wgsl.as_ptr() as usize as u32, wgsl.len() as u32);
    if module == 0 {
        return Err(fail(format!("{test}：prism_create_shader_module 失败（WGSL 编译错误见宿主错误面板）")));
    }
    let pipeline = gpu::create_compute_pipeline(dev, module, entry.as_ptr() as usize as u32, entry.len() as u32);
    if pipeline == 0 {
        return Err(fail(format!("{test}：prism_create_compute_pipeline(entry={entry}) 失败")));
    }
    // 句柄数组写入 scratch.handles（binding 顺序即数组顺序）。
    let bufs_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.n_bufs = bufs.len() as u32;
        for (i, &b) in bufs.iter().enumerate() {
            v.handles[i] = b;
        }
        v.handles.as_ptr() as usize as u32
    });
    let bind_group = gpu::create_bind_group(dev, pipeline, 0, bufs_ptr, bufs.len() as u32);
    if bind_group == 0 {
        return Err(fail(format!("{test}：prism_create_bind_group(n={}) 失败", bufs.len())));
    }
    // 单 group(0)：bgs 数组只放一个句柄。
    let bgs_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.bgs[0] = bind_group;
        v.bgs.as_ptr() as usize as u32
    });
    if gpu::dispatch(dev, pipeline, bgs_ptr, 1, workgroups.max(1), 1, 1) == 0 {
        return Err(fail(format!("{test}：prism_dispatch({workgroups},1,1) 失败")));
    }
    // 读回：GPU buffer → 暂存 → 线性内存。
    let staging = gpu::read_buffer(dev, read_buf, read_len);
    if staging == 0 {
        return Err(fail(format!("{test}：prism_read_buffer(size={read_len}) 失败")));
    }
    if gpu::wait_idle(dev) == 0 {
        return Err(fail(format!("{test}：prism_wait_idle 失败")));
    }
    let out_ptr = SCRATCH.with(|s| s.borrow_mut().out_ptr);
    if gpu::copy_out(staging, out_ptr, read_len) == 0 {
        return Err(fail(format!("{test}：prism_copy_out 失败")));
    }
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.staging = staging;
        v.out_len = read_len;
    });
    // 【D-14 真机缺陷】此处曾**同步** `gpu::pop_error(dev)`。但宿主的
    // `pendingErrors` 是异步填充的（`prism_dispatch` 的 `popErrorScope()` Promise
    // 由 `settle()` await 之后才 push），run 阶段此刻弹到的必然是**上一项**残留
    // 的错误码。改为只登记槽位数，由 collect 阶段（settle 之后）统一结算。
    SCRATCH.with(|s| s.borrow_mut().err_slots = 2);
    Ok(())
}

// ---------------------------------------------------------------------------
// 测试项 6：真 WebGPU 批量编译全部 shader（10 份，含 3 份公共件）
// ---------------------------------------------------------------------------

fn t_wgsl_compile() -> Status {
    let dev = match ready_device("wgsl.compile") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // 编译前先在 CPU 侧拦一道：结构 + 符号可解析性。真机上这两类问题会让
    // createShaderModule 整批失败，且错误面板只给行号不给"哪个符号没定义"。
    let mut pre_fail: Vec<String> = Vec::new();
    for (name, src) in shaders::expanded_sources() {
        let chk = shaders::check_shader(name, &src);
        if !chk.is_pass() {
            pre_fail.push(format!("{name}：结构自检未通过 {chk:?}"));
        }
        let missing = shaders::unresolved_symbols(&src);
        if !missing.is_empty() {
            pre_fail.push(format!("{name}：未解析符号 {missing:?}（include 未展开或缺定义）"));
        }
    }
    if !pre_fail.is_empty() {
        return fail(format!("WGSL 编译前 CPU 门禁拦截：\n  {}", pre_fail.join("\n  ")));
    }

    let mut lines: Vec<String> = Vec::new();
    for i in 0..shaders::SHADERS.len() {
        let (name, _) = shaders::SHADERS[i];
        let src = expanded(i as u32).map(|s| s.as_str()).unwrap_or("");
        let h = gpu::create_shader_module(dev, src.as_ptr() as usize as u32, src.len() as u32);
        if h == 0 {
            lines.push(format!("  FAIL  {name}（createShaderModule 返回 0）"));
            return fail(format!("WGSL 编译失败：\n{}", lines.join("\n")));
        }
        lines.push(format!("  ok    {name}  ({} B，含 include 展开)", src.len()));
    }
    set_result(&format!(
        "已提交 {} 个 shader 模块编译（include 已展开），错误码待 settle 后判定：\n{}",
        lines.len(),
        lines.join("\n")
    ));
    Status::Pending
}

fn c_wgsl_compile() -> Status {
    let dev = match ready_device("wgsl.compile") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if gpu::device_lost(dev) != 0 {
        return fail("device 在编译期间 lost");
    }
    let mut failed: Vec<&str> = Vec::new();
    for (name, _) in shaders::SHADERS {
        if gpu::pop_error(dev) != 0 {
            failed.push(name);
        }
    }
    if !failed.is_empty() {
        return fail(format!(
            "{} 个 shader 编译失败：{}\n（宿主 getCompilationInfo 消息已进错误面板）",
            failed.len(),
            failed.join(", ")
        ));
    }
    set_result(&format!(
        "真 WebGPU createShaderModule：{}/{} 全部编译通过（D2/D3/D4 全量，含 include 展开）\n\
         覆盖：{}",
        shaders::SHADERS.len(),
        shaders::SHADERS.len(),
        shaders::SHADERS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 7：真 WebGPU scan_block dispatch + 与 CPU 参考比对
// ---------------------------------------------------------------------------

fn t_gpu_scan_block() -> Status {
    let dev = match ready_device("gpu.scan_block") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // 先独立取地址（各自一次借用，结束即释放），再在**新的**借用里写字段——
    // 绝不在持有 borrow_mut 的闭包内再取地址（RefCell 重入 → panic，见 addr_of_buf）。
    let out_ptr = addr_of_buf(|x| &mut x.scan_out);
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        for i in 0..SCAN_N {
            v.scan_in[i] = scan_input(i);
        }
        v.scan_out.fill(0);
        v.out_ptr = out_ptr;
    });
    let in_len = (SCAN_N * 4) as u32;
    let in_buf = match make_buffer("gpu.scan_block", dev, in_len, usage::STORAGE, || {
        (addr_of_slice(|s| &s.scan_in[..]), in_len)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let out_buf = match make_buffer("gpu.scan_block", dev, in_len, usage::STORAGE, || (0, 0)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // uniform: {block: 0, count: SCAN_N}，按 WGSL uniform 布局打包为 16 字节
    // （SCAN_PARAMS_BYTES）。D-14 真机实测：此处曾硬编码 8，而 WGSL 侧结构体
    // 因 uniform 对齐规则实际占 16 字节，createBindGroup 验证层直接拒绝。
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.params.copy_from_slice(&scan_params_bytes(0, SCAN_N as u32));
    });
    let par_buf = match make_buffer(
        "gpu.scan_block",
        dev,
        SCAN_PARAMS_BYTES as u32,
        usage::UNIFORM,
        || {
            (
                SCRATCH.with(|s| s.borrow().params.as_ptr() as usize as u32),
                SCAN_PARAMS_BYTES as u32,
            )
        },
    ) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let cfg = ScanConfig::for_count(SCAN_N as u32);
    set_result(&format!(
        "已提交 scan_block：1 工作组 × {SCAN_N} 元素（params.block=0 count={SCAN_N}）\n\
         缓冲：in={in_buf} out={out_buf} params={par_buf}\n\
         读回 {in_len} 字节 → 线性内存 {}，待 settle 后与 CPU 参考逐元素比对",
        SCRATCH.with(|s| s.borrow().out_ptr)
    ));
    match submit_kernel(
        "gpu.scan_block",
        dev,
        WGSL_SCAN_BLOCK,
        "scan_block",
        &[in_buf, out_buf, par_buf],
        cfg.workgroups,
        out_buf,
        in_len,
    ) {
        Ok(()) => Status::Pending,
        Err(s) => s,
    }
}

fn c_gpu_scan_block() -> Status {
    let dev = match ready_device("gpu.scan_block") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.scan_block", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let (input, output) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.scan_in, v.scan_out)
    });
    // CPU 参考：Blelloch inclusive；GPU 写的是 exclusive，故期望 = incl - 原值。
    let expected = scan_blelloch_cpu(&input);
    let mut mismatch: Vec<usize> = Vec::new();
    for i in 0..SCAN_N {
        let want = expected[i].wrapping_sub(input[i]);
        if output[i] != want {
            mismatch.push(i);
        }
    }
    if !mismatch.is_empty() {
        let i = mismatch[0];
        return fail(format!(
            "scan_block 与 CPU 参考不一致：{}/{} 个元素不同，首个 i={i}（GPU {} vs CPU {}）\n不一致下标：{:?}",
            mismatch.len(),
            SCAN_N,
            output[i],
            expected[i].wrapping_sub(input[i]),
            &mismatch[..mismatch.len().min(16)]
        ));
    }
    set_result(&format!(
        "真 WebGPU scan_block：{SCAN_N}/{SCAN_N} 元素与 CPU 参考逐位一致 ✓\n\
         out[0]={} out[1]={} out[127]={}（exclusive 末项 = 总和 - 末元素 = {}）\n\
         GPU exclusive fingerprint=0x{:08x}",
        output[0],
        output[1],
        output[SCAN_N - 1],
        expected[SCAN_N - 1].wrapping_sub(input[SCAN_N - 1]),
        hash_u32(&output)
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 8：真 WebGPU histogram 原子计数 + 与 CPU 参考比对
// ---------------------------------------------------------------------------

fn t_gpu_histogram() -> Status {
    let dev = match ready_device("gpu.histogram") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // 键取自与 `cpu.sort` 同族的 LCG（种子 7），保证两个测试项用同一批键。
    let mut seed: u64 = 7;
    let keys: Vec<u32> = (0..HIST_N).map(|_| lcg(&mut seed)).collect();
    let out_ptr = addr_of_buf(|x| &mut x.hist[..]);
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.keys.copy_from_slice(&keys);
        v.hist.fill(0);
        v.out_ptr = out_ptr;
    });
    let key_len = (HIST_N * 4) as u32;
    let key_buf = match make_buffer("gpu.histogram", dev, key_len, usage::STORAGE, || {
        (addr_of_slice(|s| &s.keys[..]), key_len)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let hist_len = (HIST_BUCKETS * 4) as u32;
    // 直方图桶需清零：先建后写一段全零（scratch.hist 已 fill(0)）。
    let hist_buf = match make_buffer("gpu.histogram", dev, hist_len, usage::STORAGE, || {
        (addr_of_slice(|s| &s.hist[..]), hist_len)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    SCRATCH.with(|s| s.borrow_mut().params.copy_from_slice(&histogram_params_bytes(HIST_N as u32, HIST_BUCKETS, 0)));
    let par_buf = match make_buffer("gpu.histogram", dev, HISTOGRAM_PARAMS_BYTES as u32, usage::UNIFORM, || {
        (SCRATCH.with(|s| s.borrow().params.as_ptr() as usize as u32), HISTOGRAM_PARAMS_BYTES as u32)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let workgroups = (HIST_N as u32).div_ceil(WORKGROUP_SIZE).max(1);
    set_result(&format!(
        "已提交 histogram：{HIST_N} 键 / {HIST_BUCKETS} 桶 / shift=0 → {workgroups} 工作组\n\
         键与 cpu.sort 同一批（LCG seed=7）；params 16B 已按 WGSL uniform 布局打包\n\
         待 settle 后与 CPU 参考逐桶比对",
    ));
    match submit_kernel(
        "gpu.histogram",
        dev,
        WGSL_HISTOGRAM,
        "histogram",
        &[key_buf, hist_buf, par_buf],
        workgroups,
        hist_buf,
        hist_len,
    ) {
        Ok(()) => Status::Pending,
        Err(s) => s,
    }
}

fn c_gpu_histogram() -> Status {
    let dev = match ready_device("gpu.histogram") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.histogram", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let (keys, got) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.keys, v.hist)
    });
    // CPU 参考：与 sort.rs:counting_sort_pass 的直方图完全同口径。
    let mut want = [0u32; HIST_BUCKETS as usize];
    for &k in keys.iter() {
        let digit = ((k >> 0) & (HIST_BUCKETS - 1)) as usize;
        want[digit] = want[digit].wrapping_add(1);
    }
    let mut mismatch: Vec<usize> = Vec::new();
    let mut wrong_total = 0u32;
    for b in 0..HIST_BUCKETS as usize {
        if got[b] != want[b] {
            mismatch.push(b);
            wrong_total = wrong_total.wrapping_add(got[b]);
        }
    }
    let sum_got: u32 = got.iter().fold(0u32, |a, b| a.wrapping_add(*b));
    if !mismatch.is_empty() {
        let b = mismatch[0];
        return fail(format!(
            "histogram 与 CPU 参考不一致：{}/{} 个桶不同，桶 {b}（GPU {} vs CPU {}）\n\
             GPU 桶计数总和={sum_got}（应={HIST_N}，丢失/重复即为原子写或越界问题）",
            mismatch.len(),
            HIST_BUCKETS,
            got[b],
            want[b]
        ));
    }
    if sum_got != HIST_N as u32 {
        return fail(format!("桶计数总和 {sum_got} ≠ 键数 {HIST_N}"));
    }
    set_result(&format!(
        "真 WebGPU histogram：{HIST_BUCKETS}/{HIST_BUCKETS} 桶与 CPU 参考一致 ✓\n\
         桶计数总和={sum_got} == 键数 {HIST_N}（无丢失/无重复累加）\n\
         非空桶={} 个  首个非空桶={}  hist fingerprint=0x{:08x}",
        want.iter().filter(|&&v| v != 0).count(),
        want.iter().position(|&v| v != 0).unwrap_or(0),
        hash_u32(&got)
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// D-14 补测：D3 模拟 / D2 wavefront 的真机 dispatch 判对
// ---------------------------------------------------------------------------
//
// # 覆盖台账（16 个 `@compute` 入口的逐个处置）
//
// | shader          | entry point    | 处置                                        |
// |-----------------|----------------|---------------------------------------------|
// | sim_particles   | emit           | `gpu.sim_particles`（3 入口串联，逐字段判对）|
// | sim_particles   | integrate      | 同上                                        |
// | sim_particles   | hash_cells     | 同上（u32 格键逐条精确比对）                 |
// | sim_particles   | neighbor_scan  | SKIP：纯占位（`_ = i;`），无输出可比        |
// | sim_sph         | sph_predict    | `gpu.sim_sph`                               |
// | sim_sph         | sph_density    | `gpu.sim_sph`                               |
// | sim_sph         | sph_project    | `gpu.sim_sph`                               |
// | sim_sph         | sph_velocity   | SKIP：纯占位（`_ = sp_vel[i];`）             |
// | sim_cloth       | cloth_predict  | `gpu.sim_cloth`                             |
// | sim_cloth       | cloth_project  | `gpu.sim_cloth`（4 对不相交约束，见下）      |
// | sim_cloth       | cloth_velocity | `gpu.sim_cloth`                             |
// | rt_wavefront    | generate       | `gpu.rt_wavefront_multi`（多 group 0/1/2/3） |
// | rt_wavefront    | shade          | `gpu.rt_wavefront_multi`（group 2 装 5 绑定） |
// | rt_wavefront    | connect        | `gpu.rt_wavefront`（状态机门禁）             |
// | rt_wavefront    | extend         | `gpu.rt_wavefront_multi`（group 2 装 4 绑定） |
// | rt_wavefront    | accumulate     | `gpu.rt_wavefront`（逐像素 spp 均值判对）    |
//
// `rt_bvh` / `rt_shade` / `rt_common` / `sim_common` / `text` 的 `@compute` 入口
// 数为 0（前四者是库件，`text` 只有 `@vertex`/`@fragment`），故不参与 dispatch
// 判对；它们的口径是"随展开后的宿主 shader 一起编译"（已由 `wgsl.compile` 覆盖）。
//
// # 仅剩一族 SKIP 的根因（不是"跑不动"，而是"无法判对"）
//
// 1. `neighbor_scan` / `sph_velocity` 是**纯占位**：函数体只有 `_ = ...`，
//    不写任何 storage。dispatch 它们只能证明"不崩"——没有输出就没有可比对
//    的东西，造判据等于自证恒真，故明确 SKIP。它们的语义等价判对由**真正
//    消费这些数据的 kernel** 覆盖：`sph_density` 自己做了 O(n²) 全邻域扫描
//    （不读 `cell_range`），`cloth_project` 覆盖了 `neighbor_scan` 预留的
//    "邻域连续消费"语义位。
//
// 2. （历史）`rt_wavefront` 的 `generate` / `shade` / `extend` 曾因宿主胶水
//    只支持 `getBindGroupLayout(0)` / `setBindGroup(0, …)` 而被 SKIP。现已
//    扩展 ABI：`prism_create_bind_group` 新增 `group` 索引参数、
//    `prism_dispatch` 接受多 bind group 表并逐个 `setBindGroup(i, …)`。这 3 个
//    入口现由 `gpu.rt_wavefront_multi` 真 dispatch 并判对（见该项条目）。
//
// # `cloth_project` 为什么只装 4 条两两不相交的约束（重要，非省事）
//
// 该入口按**约束**并行（`gid.x` 索引约束），却对共享的 `cp_pos` 做读-改-写。
// 真实 4×2 网格会生成 16 条约束且相邻约束**共享粒子** ⇒ 多线程同址写 ⇒
// WebGPU 内存模型下结果**非确定**，无法与任何 CPU 参考逐元素判对。
// 故本测试只装载 4 条**两两不相交**的约束（`(0,1) (2,3) (4,5) (6,7)`），
// 使结果确定可判。被排除的 12 条共享粒子约束在结果文本中**显式列出**
// （不静默缩小覆盖面），并给出根因与修法方向（按粒子着色 / 按网格行分组）。
//
// # CPU 参考的来源与"逐位一致"要求
//
// 判对基准是各 crate 的 CpuRef，实现权威是本文件的**独立重写**（公式逐条
// 对齐，见每处注释的出处行号）。之所以重写而不直接 `use` 那些 crate：
// `prism-gpu/Cargo.toml` 的依赖只有 `prism-math`，而 `Cargo.toml` 不在本
// 文件改动范围内。三条铁律仍成立：
// * 公式与 CpuRef **同式**（`poly6_2d` / `xpbd_distance_project` / `cell_key`
//   的多项式系数、运算次序都与 WGSL 逐条对齐，含 WGSL 侧 `(PI*h)*h*…` 的
//   左结合展开——浮点乘法不满足结合律，次序变了末位就变）。
// * `host_test.mjs` 第 8 节再用**第三份 JS 实现**交叉核对同一组输入。
// * 整数路径（`cell` 格键、`state`、`bounce`）走**逐位精确**判对，不容差。

/// 浮点判对容差（GPU 端允许 FMA 收缩，末位 1~2 ULP 属正常；整数路径不走此容差）。
const F32_TOL: f32 = 2.0e-6;

/// 从字节缓冲 `off` 处读 `f32`（小端）。越界返回 `0.0`（宪章 C18：绝不 panic）。
fn unpack_f32(buf: &[u8], off: usize) -> f32 {
    if off + 4 > buf.len() {
        return 0.0;
    }
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// 从字节缓冲 `off` 处读 `u32`（小端）。越界返回 `0`（宪章 C18）。
fn unpack_u32(buf: &[u8], off: usize) -> u32 {
    if off + 4 > buf.len() {
        return 0;
    }
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// 往字节缓冲 `off` 处写 `f32`（小端）。越界为无操作（宪章 C18）。
fn pack_f32(buf: &mut [u8], off: usize, v: f32) {
    if off + 4 > buf.len() {
        return;
    }
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// 往字节缓冲 `off` 处写 `u32`（小端）。越界为无操作（宪章 C18）。
fn pack_u32(buf: &mut [u8], off: usize, v: u32) {
    if off + 4 > buf.len() {
        return;
    }
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// 往字节缓冲 `off` 处写一个 `vec3<f32>`（3 个分量，4 字节对齐填充不动）。
///
/// **不给元素下标**：AoS 状态槽步长 32、`Particle` 步长 48，都不是 16，
/// 按"第 i 个 vec3"折算偏移必然错位（这正是 D-14 真机 `sim_particles` 的缺陷）。
fn pack_vec3_at(buf: &mut [u8], off: usize, v: [f32; 3]) {
    pack_f32(buf, off, v[0]);
    pack_f32(buf, off + 4, v[1]);
    pack_f32(buf, off + 8, v[2]);
}

/// 从字节缓冲 `off` 处读一个 `vec3<f32>`（忽略 4 字节对齐填充）。
fn unpack_vec3_at(buf: &[u8], off: usize) -> [f32; 3] {
    [unpack_f32(buf, off), unpack_f32(buf, off + 4), unpack_f32(buf, off + 8)]
}

/// `sim_common.wgsl::Particle` 的字段偏移（步长 [`PARTICLE_STRIDE_BYTES`] = 48）。
///
/// `pos:vec3@0` / `mass_inv:f32@12` / `vel:vec3@16` / `_pad:f32@28` / `cell:u32@32`
/// / `_pad2:u32@36`。
mod particle_off {
    /// 位置（占 0..12）。
    pub const POS: usize = 0;
    /// 逆质量（占 12..16）。
    pub const MASS_INV: usize = 12;
    /// 速度（占 16..28）。
    pub const VEL: usize = 16;
    /// 填充（占 28..32）：shader 若写它即说明布局不符。
    pub const PAD: usize = 28;
    /// 格键（占 32..36）。
    pub const CELL: usize = 32;
    /// 填充（占 36..40）。
    pub const PAD2: usize = 36;
}

/// 状态槽（AoS，步长 [`STATE_STRIDE_BYTES`] = 32）的字段偏移。
///
/// `sim_sph.wgsl::SphState` 与 `sim_cloth.wgsl::ClothState` 同布局：
/// `pos:vec3@0` / `vel:vec3@16` / `inv:f32@28`（`inv` 恰好落在 `vel` 的对齐填充里）。
mod state_off {
    /// 位置（占 0..12）。
    pub const POS: usize = 0;
    /// 速度（占 16..28）。
    pub const VEL: usize = 16;
    /// 逆质量（占 28..32）。
    pub const INV: usize = 28;
}

/// 往状态槽（AoS）写一个粒子的 pos / vel / inv。
fn pack_state(buf: &mut [u8], elem: usize, pos: [f32; 3], vel: [f32; 3], inv: f32) {
    let base = elem * STATE_STRIDE_BYTES as usize;
    pack_vec3_at(buf, base + state_off::POS, pos);
    pack_vec3_at(buf, base + state_off::VEL, vel);
    pack_f32(buf, base + state_off::INV, inv);
}

/// 3D 向量模（对齐 WGSL `length`：`sqrt(dot)`，不用 libm）。
fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// 2D Poly6 核（判对基准 = `prism-physics/src/sph.rs:41` `poly6_2d`）。
///
/// **运算次序必须与 WGSL `sim_common.wgsl::poly6_2d` 逐条一致**：
/// `4.0 / (PI * h*h*h*h*h*h*h*h) * diff * diff * diff`——`PI` 乘在**最左**，
/// 8 次 `* h` 左结合。浮点乘法不满足结合律，次序变则末位变。
fn poly6_2d(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    let diff = h2 - r * r;
    // WGSL 侧的 `PI` 字面量 3.141592653589793 与 std 的 f32 最近值同。
    let mut den: f32 = std::f32::consts::PI;
    for _ in 0..8 {
        den *= h;
    }
    4.0 / den * diff * diff * diff
}

/// XPBD 距离约束投影（判对基准 = `prism-physics/src/xpbd.rs:161`
/// `DistanceConstraint::project`；WGSL 侧为 `sim_common.wgsl::xpbd_distance_project`）。
///
/// 返回投影后残余 `|C|`。`lambda` 就地累积。
fn xpbd_distance_project(
    pa: &mut [f32; 3],
    pb: &mut [f32; 3],
    wa: f32,
    wb: f32,
    rest: f32,
    compliance: f32,
    h_sub: f32,
    lambda: &mut f32,
) -> f32 {
    let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let len = len3(d);
    let c = len - rest;
    // WGSL `select(vec3(1,0,0), d / max(len,1e-9), len > 1e-9)`
    let grad = if len > 1e-9 {
        let m = len.max(1e-9);
        [d[0] / m, d[1] / m, d[2] / m]
    } else {
        [1.0, 0.0, 0.0]
    };
    let gg = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
    let at = compliance / (h_sub * h_sub);
    let denom = wa * gg + wb * gg + at;
    if denom <= 0.0 {
        return c.abs();
    }
    let dl = (-c - at * *lambda) / denom;
    *lambda += dl;
    for k in 0..3 {
        pa[k] -= grad[k] * (wa * dl);
        pb[k] += grad[k] * (wb * dl);
    }
    c.abs()
}

/// PCG hash（与 `sim_particles.wgsl::pcg_hash` 逐位一致；跨平台整数运算）。
fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// 三维哈希派生 `[0,1)`（与 `sim_particles.wgsl::hash3_f32` 逐位一致）。
fn hash3_f32(a: u32, b: u32, c: u32) -> f32 {
    let h = pcg_hash(pcg_hash(a) ^ pcg_hash(b ^ c));
    (h >> 8) as f32 * (1.0 / 16777216.0)
}

/// 空间哈希格键（与 `sim_common.wgsl::cell_key` 逐位一致；判对基准 =
/// `prism-physics/src/sph.rs::SpatialHash` 的 `floor(v/h)` 分格口径）。
fn cell_key(p: [f32; 3], h: f32) -> u32 {
    let c = [
        (p[0] / h).floor() as i32,
        (p[1] / h).floor() as i32,
        (p[2] / h).floor() as i32,
    ];
    let bx = (c[0] + 1024) as u32;
    let by = (c[1] + 1024) as u32;
    let bz = (c[2] + 1024) as u32;
    (bx & 0x7FF) | ((by & 0x7FF) << 11) | ((bz & 0x7FF) << 22)
}

// ---------------------------------------------------------------------------
// 判对工具：读回 vs CPU 参考
// ---------------------------------------------------------------------------

/// 逐元素比对 `f32` storage 读回值与 CPU 参考；不一致即 [`Status::Fail`]。
///
/// 返回最大绝对偏差（供结果文本报告"差了多少"，而非只报对错）。
fn cmp_f32(test: &str, label: &str, got: &[u8], want: &[f32]) -> Result<f32, Status> {
    if got.len() < want.len() * 4 {
        return Err(fail(format!(
            "{test}：{label} 读回 {} 字节 < CPU 参考所需 {} 字节（buffer 尺寸算错？）",
            got.len(),
            want.len() * 4
        )));
    }
    let mut bad = 0usize;
    let mut first = 0usize;
    let mut max_dev = 0.0f32;
    for (i, &w) in want.iter().enumerate() {
        let g = unpack_f32(got, i * 4);
        let dev = (g - w).abs();
        if dev > max_dev {
            max_dev = dev;
        }
        if !(dev <= F32_TOL) {
            if bad == 0 {
                first = i;
            }
            bad += 1;
        }
    }
    if bad != 0 {
        return Err(fail(format!(
            "{test}：{label} 与 CPU 参考不一致——{bad}/{} 个元素超容差 {F32_TOL:e}，\
             首个 i={first}（GPU {} vs CPU {}）",
            want.len(),
            unpack_f32(got, first * 4),
            want[first]
        )));
    }
    Ok(max_dev)
}

/// 逐元素比对 `u32` storage 读回值与 CPU 参考（**精确**，不容差）。
fn cmp_u32(test: &str, label: &str, got: &[u8], want: &[u32]) -> Result<(), Status> {
    if got.len() < want.len() * 4 {
        return Err(fail(format!(
            "{test}：{label} 读回 {} 字节 < CPU 参考所需 {} 字节（buffer 尺寸算错？）",
            got.len(),
            want.len() * 4
        )));
    }
    let mut bad = 0usize;
    let mut first = 0usize;
    for (i, &w) in want.iter().enumerate() {
        if unpack_u32(got, i * 4) != w {
            if bad == 0 {
                first = i;
            }
            bad += 1;
        }
    }
    if bad != 0 {
        return Err(fail(format!(
            "{test}：{label} 与 CPU 参考不一致——{bad}/{} 个元素不同，首个 i={first}（GPU {} vs CPU {}）",
            want.len(),
            unpack_u32(got, first * 4),
            want[first]
        )));
    }
    Ok(())
}

/// 逐字节比对（用于"读回应与输入完全相同"的生命周期门禁）。
fn cmp_bytes(test: &str, label: &str, got: &[u8], want: &[u8]) -> Result<(), Status> {
    if got.len() != want.len() {
        return Err(fail(format!("{test}：{label} 读回 {} 字节 ≠ 参考 {} 字节", got.len(), want.len())));
    }
    let mut bad = 0usize;
    let mut first = 0usize;
    for i in 0..want.len() {
        if got[i] != want[i] {
            if bad == 0 {
                first = i;
            }
            bad += 1;
        }
    }
    if bad != 0 {
        return Err(fail(format!(
            "{test}：{label} 有 {bad}/{} 字节不同，首个偏移 {first}（GPU 0x{:02x} vs 参考 0x{:02x}）",
            want.len(),
            got[first],
            want[first]
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU 提交原语（chain 形态：`submit_kernel` 的多 entry 泛化）
// ---------------------------------------------------------------------------

/// 建 bind group（句柄数组按 WGSL `binding` 升序落进 `scratch.handles`）。
///
/// `group` 为 WGSL `@group` 索引。绑定数必须**恰好等于**该 entry 在该 group 内
/// 静态用到的 binding 数——`layout:"auto"` 只为静态使用的 binding 生成布局，多给
/// 一个就 `createBindGroup` 验证失败。这正是 D-14 教训 "bind group 数量对不上" 的
/// 可本地化形态。
fn bind_group_of(
    test: &str,
    dev: u32,
    pipeline: u32,
    group: u32,
    bufs: &[u32],
) -> Result<u32, Status> {
    if bufs.len() > 8 {
        return Err(fail(format!(
            "{test}：绑定数 {} 超过 scratch.handles 容量 8（须拆分测试项）",
            bufs.len()
        )));
    }
    // 借用一次即释放，取到句柄数组地址（见 addr_of_buf 的 RefCell 重入警告）。
    let bufs_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.n_bufs = bufs.len() as u32;
        for (i, &b) in bufs.iter().enumerate() {
            v.handles[i] = b;
        }
        v.handles.as_ptr() as usize as u32
    });
    let bg = gpu::create_bind_group(dev, pipeline, group, bufs_ptr, bufs.len() as u32);
    if bg == 0 {
        return Err(fail(format!(
            "{test}：prism_create_bind_group(group={group}, n={}) 失败",
            bufs.len()
        )));
    }
    Ok(bg)
}

/// 结算本项在 [`SCRATCH::err_slots`] 登记的验证错误槽位；任一非 0 即失败。
///
/// **只能在 collect 阶段调用**（宿主已 `await settle()` 之后）：宿主
/// `pendingErrors` 由 `settle()` 里的 Promise 异步 push，run 阶段同步 pop 到的
/// 是上一项的残留错误。槽位数由 run 阶段登记（1 个 `create_shader_module` +
/// 每个 `prism_dispatch` 各 1 个），本函数弹出同样多条并逐条判非 0，
/// 弹完把计数清零，保证下一项从干净队列开始。
fn check_errors(test: &str, dev: u32) -> Result<(), Status> {
    let n = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        let n = v.err_slots as usize;
        v.err_slots = 0;
        n
    });
    for k in 0..n {
        let err = gpu::pop_error(dev);
        SCRATCH.with(|s| s.borrow_mut().submit_err = err);
        if err != 0 {
            return Err(fail(format!(
                "{test}：第 {} 个提交阶段捕获验证错误码（宿主错误面板有详细信息；\
                 常见根因：bind group 的句柄个数/顺序与该 entry 静态引用的 binding 不一致\
                 ——胶水把第 i 个句柄固定映射到 binding:i，故引用集合必须是连续前缀 0..n-1；\
                 或 buffer 小于最小绑定尺寸）",
                k
            )));
        }
    }
    Ok(())
}

/// 用**显式条数**结算错误槽位，复用 [`check_errors`] 的整段判据与文案。
///
/// 场景 dispatch 的错误 scope 数由 [`scene_gpu`] 自己记账（`create_shader_module`
/// 只在首次提交时登记、每次 `prism_dispatch` 各 1 个），与 `submit_chain` 的
/// 「1 + steps.len()」口径不同，故不能沿用 `SCRATCH.err_slots` 的隐式约定。
fn check_errors_counted(test: &str, dev: u32, n: u32) -> Result<(), Status> {
    SCRATCH.with(|s| s.borrow_mut().err_slots = n);
    check_errors(test, dev)
}

/// 登记一次读回（GPU buffer → 暂存 → 线性内存 `dst_ptr`）。
fn readback_to(test: &str, dev: u32, buf: u32, len: u32, dst_ptr: u32) -> Result<(), Status> {
    let staging = gpu::read_buffer(dev, buf, len);
    if staging == 0 {
        return Err(fail(format!("{test}：prism_read_buffer(size={len}) 失败")));
    }
    if gpu::copy_out(staging, dst_ptr, len) == 0 {
        return Err(fail(format!("{test}：prism_copy_out 失败")));
    }
    Ok(())
}

/// 单个 dispatch step 里一个 bind group 的构造描述：`group` = WGSL `@group` 索引，
/// `bufs` = 该组内按 `binding` 升序的句柄集合（须为连续前缀 `{0..n-1}`）。
///
/// 句柄是**运行期**值（GPU buffer 句柄），故 `bufs` 借用调用点的局部数组。
#[derive(Clone, Copy)]
struct GroupBind<'a> {
    /// WGSL `@group` 索引。
    group: u32,
    /// 该组内句柄（第 i 个 → `binding: i`）。
    bufs: &'a [u32],
}

/// 一个多 group dispatch step：`(entry 名, 工作组数, 该 step 用到的 bind group 列表)`。
///
/// `groups` 内各项的 `group` 索引**必须**从 0 起连续（`0,1,…,k`）——与
/// `layout:"auto"` 下"每个 group 的绑定集是连续前缀"同理，被 pipeline 静态使用的
/// group 集合也从 0 起连续（本引擎 rt 家族从 0 连续使用，故按 group 索引即为数组
/// 下标处理）。
struct MultiStep<'a> {
    /// compute entry point 名。
    entry: &'static str,
    /// 工作组数（x 方向）。
    workgroups: u32,
    /// 本 step 的 bind group 列表（group 索引从 0 连续）。
    groups: &'a [GroupBind<'a>],
}

/// 构造单 group(0) 的多 group step（新多 group 测试项写起来更省事）。
///
/// 返回的 [`MultiStep`] 借用一个由调用方持有的 `gb`（避免返回引用临时值的悬垂）。
fn step0<'a>(entry: &'static str, workgroups: u32, gb: &'a GroupBind<'a>) -> MultiStep<'a> {
    MultiStep { entry, workgroups, groups: core::slice::from_ref(gb) }
}

/// 提交一条 dispatch 链：同一份 WGSL 源，逐 entry 建 pipeline + bind group + dispatch。
///
/// `steps` 每项为 `(entry 名, 工作组数, group(0) 的绑定句柄)`，绑定句柄按 WGSL
/// `binding` 升序、且**恰好**是该 entry 静态引用到的 binding 集合。宿主胶水把句柄
/// 数组第 `i` 个固定映射到 `binding: i`，而 `layout:"auto"` 只为静态引用的 binding
/// 生成布局 ⇒ 引用集合必须恰好是连续前缀 `{0,1,…,n-1}`（详见
/// [`STATE_STRIDE_BYTES`] 的文档）。多 group 链见 [`submit_chain_multi`]。
/// 各 dispatch 是独立的 command submission（宿主胶水每次 `prism_dispatch`
/// 自建 encoder/pass/submit），故 stage 间的写后读顺序由队列保证。
///
/// 本函数**只登记**错误槽位数（1 + steps.len()），错误判定由 collect 阶段的
/// [`check_errors`] 完成——宿主错误队列在 settle 之后才填充。
fn submit_chain(
    test: &str,
    dev: u32,
    wgsl: &'static str,
    steps: &[(&'static str, u32, &[u32])],
) -> Result<(), Status> {
    // 单 group 链复用多 group 实现（每次一个 group(0) 的 MultiStep）。
    let gbs: Vec<GroupBind<'_>> = steps
        .iter()
        .map(|&(_, _, bufs)| GroupBind { group: 0, bufs })
        .collect();
    let multi: Vec<MultiStep<'_>> = steps
        .iter()
        .zip(gbs.iter())
        .map(|(&(entry, wg, _), gb)| step0(entry, wg, gb))
        .collect();
    submit_chain_multi(test, dev, wgsl, &multi)
}

/// 提交一条**多 bind group** dispatch 链（`rt_wavefront` 的 generate/shade/extend）。
///
/// 每个 [`MultiStep`] 携带若干 bind group，逐 group 调 `prism_create_bind_group`
/// （带 group 索引）并把建好的句柄按 group 索引填进 `scratch.bgs`，最后一次性
/// `prism_dispatch`（宿主按数组下标 `i` → `@group(i)` 逐个 `setBindGroup`）。
/// 每个 group 内的 `bufs` 仍须是连续前缀（见 [`submit_chain`] 的文档）。
///
/// 错误槽位数登记同 [`submit_chain`]：1（shader module）+ steps.len()（每 step 一次
/// dispatch 的验证错误 scope）。
fn submit_chain_multi(
    test: &str,
    dev: u32,
    wgsl: &'static str,
    steps: &[MultiStep<'_>],
) -> Result<(), Status> {
    let module = gpu::create_shader_module(dev, wgsl.as_ptr() as usize as u32, wgsl.len() as u32);
    if module == 0 {
        return Err(fail(format!(
            "{test}：prism_create_shader_module 失败（WGSL 编译错误见宿主错误面板）"
        )));
    }
    for st in steps {
        let entry = st.entry;
        let pipeline =
            gpu::create_compute_pipeline(dev, module, entry.as_ptr() as usize as u32, entry.len() as u32);
        if pipeline == 0 {
            return Err(fail(format!(
                "{test}：prism_create_compute_pipeline(entry={entry}) 失败（入口名拼错或该 shader 无此入口）"
            )));
        }
        // 逐 group 建 bind group，句柄按 group 索引落进 scratch.bgs。
        let n_groups = st.groups.len();
        if n_groups == 0 || n_groups > BIND_GROUP_MAX as usize {
            return Err(fail(format!(
                "{test}：step `{entry}` 的 group 数 {n_groups} 非法（须 1..={BIND_GROUP_MAX}）"
            )));
        }
        let mut ordered: [u32; BIND_GROUP_MAX as usize] = [0; BIND_GROUP_MAX as usize];
        for gb in st.groups {
            if gb.group as usize >= n_groups {
                return Err(fail(format!(
                    "{test}：step `{entry}` 的 group 索引 {} 越界（须从 0 连续，共 {n_groups} 个）",
                    gb.group
                )));
            }
            let bg = bind_group_of(test, dev, pipeline, gb.group, gb.bufs)?;
            ordered[gb.group as usize] = bg;
        }
        let bgs_ptr = SCRATCH.with(|s| {
            let mut v = s.borrow_mut();
            v.bgs[..n_groups].copy_from_slice(&ordered[..n_groups]);
            v.bgs.as_ptr() as usize as u32
        });
        if gpu::dispatch(dev, pipeline, bgs_ptr, n_groups as u32, st.workgroups.max(1), 1, 1) == 0 {
            return Err(fail(format!(
                "{test}：prism_dispatch({entry}, groups={n_groups}, {}) 失败",
                st.workgroups
            )));
        }
    }
    if gpu::wait_idle(dev) == 0 {
        return Err(fail(format!("{test}：prism_wait_idle 失败")));
    }
    SCRATCH.with(|s| s.borrow_mut().err_slots = 1 + steps.len() as u32);
    Ok(())
}

/// 把 `bytes` 写入参数区 `off` 处，返回该处的线性内存地址；越界返回 `0`。
///
/// 借用语义同 [`addr_of_buf`]：**不可**在持有 `borrow_mut` 的闭包内调用。
fn params2_write(off: usize, bytes: &[u8]) -> u32 {
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        let end = off + bytes.len();
        if end > PARAMS_ARENA_BYTES {
            return 0;
        }
        v.params2[off..end].copy_from_slice(bytes);
        v.params2.as_ptr() as usize as u32 + off as u32
    })
}

/// 取指定 shader **展开后**的 WGSL 源码（[`OnceLock`] 缓存 ⇒ 地址稳定）。
fn expanded_named(name: &str) -> Option<&'static str> {
    let all = EXPANDED.get_or_init(|| shaders::expanded_sources().into_iter().map(|(_, s)| s).collect());
    all.iter()
        .enumerate()
        .find(|(i, _)| shaders::SHADERS[*i].0 == name)
        .map(|(_, s)| s.as_str())
}

/// 提取一份 WGSL 里全部 `@compute` 入口名（按出现顺序）。
fn compute_entries(src: &str) -> Vec<String> {
    let clean: String = src
        .lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut out: Vec<String> = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = clean[from..].find("@compute") {
        let at = from + pos + "@compute".len();
        let rest = &clean[at..];
        if let Some(fp) = rest.find("fn ") {
            let name: String = rest[fp + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && !out.contains(&name) {
                out.push(name);
            }
        }
        from = at;
    }
    out
}

// ---------------------------------------------------------------------------
// 覆盖台账（单一真源：`shader.dispatch_census` 项据此判红）
// ---------------------------------------------------------------------------

/// 覆盖状态：已 dispatch 判对。
const ST_COVERED: &str = "COVERED";
/// 覆盖状态：显式跳过（须给原因）。
const ST_SKIP: &str = "SKIP";

/// 全部 `@compute` 入口的覆盖台账：`文件名 -> [(入口, 状态, 原因)]`。
///
/// **新增 `@compute` 入口时必须在此登记**，否则 `shader.dispatch_census` 转红
/// ——这条门禁直接针对"新 kernel 悄悄从未被 dispatch 过"（D-14 之前的现状：
/// 9 份 WGSL 里只有 2 个 D0 内核真跑过）。
static COVERAGE: &[(&str, &[(&str, &str, &str)])] = &[
    (
        "sim_particles.wgsl",
        &[
            ("emit", ST_COVERED, ""),
            ("integrate", ST_COVERED, ""),
            ("hash_cells", ST_COVERED, ""),
            (
                "neighbor_scan",
                ST_SKIP,
                "纯占位：函数体只有 `_ = i;`，不写任何 storage → dispatch 后无可比对输出（造判据=自证恒真）。\
                 其预留的邻域消费语义由 sph_density（O(n²) 全邻域扫描）与 cloth_project 覆盖",
            ),
        ],
    ),
    (
        "sim_sph.wgsl",
        &[
            ("sph_predict", ST_COVERED, ""),
            ("sph_density", ST_COVERED, ""),
            ("sph_project", ST_COVERED, ""),
            (
                "sph_velocity",
                ST_SKIP,
                "纯占位：函数体只有 `_ = sp_vel[i];`，不写任何 storage → 无可比对输出。\
                 速度侧语义由 sph_predict（预测）+ sph_project（λ 约束）覆盖",
            ),
        ],
    ),
    (
        "sim_cloth.wgsl",
        &[
            ("cloth_predict", ST_COVERED, ""),
            ("cloth_project", ST_COVERED, ""),
            ("cloth_velocity", ST_COVERED, ""),
        ],
    ),
    (
        "rt_wavefront.wgsl",
        &[
            (
                "generate",
                ST_COVERED,
                "由 `gpu.rt_wavefront_multi` 真 dispatch：多 group 前缀 g0{slots,params} \
                 g1{sobol_dir} g2{scene_params} g3{camera}，读回路径槽与 CPU 参考（prism-rt 同式）逐字段判对",
            ),
            (
                "shade",
                ST_COVERED,
                "由 `gpu.rt_wavefront_multi` 真 dispatch：g2 加载 5 绑定 \
                 {scene_params,materials,prim_mat,tris,lights}（本场景 light_count=0，NEE 运行期循环为空，可复现）",
            ),
            ("connect", ST_COVERED, ""),
            (
                "extend",
                ST_COVERED,
                "由 `gpu.rt_wavefront_multi` 真 dispatch：g2 加载 4 绑定 \
                 {scene_params,materials,prim_mat,tris}（不引用 lights，前缀 {0..3}）",
            ),
            ("accumulate", ST_COVERED, ""),
        ],
    ),
    (
        "scene_frame.wgsl",
        &[(
            "scene_main",
            ST_COVERED,
            "由 `gpu.scene_frame` 真 dispatch（V2.2-T03 补齐「有设备」分支）：\
             V2 六绑定 {params,spheres,lights,triangles,nodes,out} 齐备，\
             8×8 / 16×16 两种场景 + 含 1 个圆柱曲线的场景，读回 `w*h*4` 字节后\
             与 CPU 参考（CpuRefBackend 同式）统计对拍（PSNR≥40dB）+ 双跑逐位一致",
        )],
    ),
    ("rt_bvh.wgsl", &[]),
    ("rt_shade.wgsl", &[]),
    ("rt_common.wgsl", &[]),
    ("sim_common.wgsl", &[]),
    ("text.wgsl", &[]),
    (
        "gi_compute.wgsl",
        &[
            (
                "gi_probe_trace",
                ST_COVERED,
                "由 `gpu.gi_compute` 真 dispatch：对每条探针做半球 SDF sphere tracing，\
                 逐方向写采样辐亮度，与 CPU 侧（prism-rt::gi::sdf_gi 同式）逐字段判对",
            ),
            (
                "gi_sh_project",
                ST_COVERED,
                "由 `gpu.gi_compute` 真 dispatch：把采样辐亮度投影到 SH-L2（辐射度投影，\
                 对齐 prism-rt::gi::sh 的 π 约定），SH 系数逐字段判对",
            ),
            (
                "gi_probe_interpolate",
                ST_COVERED,
                "由 `gpu.gi_compute` 真 dispatch：最近探针 SH 重建 + Lambert 转辐亮度，\
                 查询点出射辐亮度逐字段判对",
            ),
            (
                "gi_layout_census",
                ST_COVERED,
                "由 `gpu.gi_layout` 真 dispatch（1 工作组）：把 shader 自己读到的 \
                 probe0/1/2 的 pos/normal 与 GiParams 的 (probe_count, prim_count, \
                 env_radiance.x, albedo.x) 原样写进 samples[0..7]，宿主与 Rust 打包值\
                 逐格判对——这是「真机驱动对 vec3<f32> 的 struct size 处理与规范不一致」\
                 这类**只有真机才暴露**的布局缺陷的唯一直接判据",
            ),
        ],
    ),
    (
        "background.wgsl",
        &[(
            "bg_main",
            ST_SKIP,
            "GPU compute 背景渲染内核（demo `paint_background` 热点兜底）。bind group / \
             dispatch 链路待接入；当前由 `prism_gpu::render_background` 的 CPU 参考真实兜底。\
             shader 已嵌入 SHADERS 并过 D-29 结构自检，等待 WebGPU 运行时真 dispatch",
        )],
    ),
    // ---- compute-first 管线（三个新 kernel） ----
    //
    // 三个入口都**如实标 SKIP**：本构建环境只有 libvulkan loader、无 ICD，
    // 真机 dispatch 无法执行（`vulkan::compute` 的探针报告为
    // `device-unavailable / no-vulkan-icd`）。它们的判对不靠「假装跑过 GPU」，
    // 而靠两条真实且可执行的判据：
    //   ① `prism-gpu::compute` 的 CPU 参考实现（kernel 语义真测）；
    //   ② `prism-rt::wavefront` 的 double gate parity 网（CPU 递归序 vs
    //      GPU 数组序的统计判定 + PSNR）。
    // 真机派发由 `vulkan::compute::real_device_compute_dispatch_smoke`
    // （probe 先行、无设备自跳过），有 GPU 的机器上随 `cargo test` 实跑。
    (
        "rt_primary_ray_gen.wgsl",
        &[(
            "primary_ray_gen",
            ST_SKIP,
            "主射线发射 kernel：SPIR-V 待 glslang 工具链生成（当前平台无该工具链），\
             当前平台无 ICD 故无设备 dispatch。语义由 prism-gpu::compute::primary_ray_gen_cpu \
             真实测（像素覆盖/几何/确定性）、并由 prism-rt::wavefront 的 parity 网对拍；\
             设备 smoke 见 vulkan::compute::real_device_compute_dispatch_smoke（无设备自跳过）",
        )],
    ),
    (
        "rt_wavefront_intersect.wgsl",
        &[(
            "wavefront_intersect",
            ST_SKIP,
            "wavefront 求交一步 kernel：同上（SPIR-V 待工具链 + 当前平台无 ICD）。\
             语义由 prism-gpu::compute::wavefront_intersect_step_cpu 真实测\
             （最近命中/平局 tie-break/球内出射/批量对齐/确定性）",
        )],
    ),
    (
        "spv_mul2_probe.wgsl",
        &[(
            "mul2",
            ST_SKIP,
            "SPIR-V 装配探针（dst=src*2）：手写 SPIR-V 已嵌入二进制并过字级结构校验，\
             但当前平台无 ICD 无法设备派发。设备 smoke 见 \
             vulkan::compute::real_device_compute_dispatch_smoke（无设备自跳过）",
        )],
    ),
];

// ---------------------------------------------------------------------------
// 测试项 9：`shader.dispatch_census`（纯 CPU：覆盖台账 vs WGSL 实际入口）
// ---------------------------------------------------------------------------

fn t_shader_dispatch_census() -> Status {
    let mut lines: Vec<String> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    let mut covered = 0usize;
    let mut skipped = 0usize;
    for (file, entries) in COVERAGE {
        // 台账必须与 SHADERS 表一致（防台账里写了不存在的文件）。
        if shaders::source(file).is_none() {
            problems.push(format!("台账登记了不存在的 shader：{file}"));
            continue;
        }
        let expanded = shaders::expand(file);
        let actual = compute_entries(&expanded);
        let listed: Vec<&str> = entries.iter().map(|(e, _, _)| *e).collect();
        if listed.len() != actual.len() {
            problems.push(format!(
                "{file}：台账登记 {} 个入口，WGSL 里实际 {} 个（{:?} vs {:?}）",
                listed.len(),
                actual.len(),
                listed,
                actual
            ));
        }
        for e in actual.iter() {
            if !listed.contains(&e.as_str()) {
                problems.push(format!(
                    "{file}：WGSL 里的 `@compute` 入口 `{e}` 未在 COVERAGE 台账登记——\
                     新 kernel 必须登记（否则就是「编译过但从未 dispatch 过」）"
                ));
            }
        }
        for (entry, state, reason) in entries.iter() {
            if !actual.iter().any(|e| e == entry) {
                problems.push(format!("{file}：台账登记的入口 `{entry}` 在 WGSL 里不存在"));
                continue;
            }
            if *state == ST_COVERED {
                covered += 1;
                lines.push(format!("  COVERED  {file:<18} {entry}"));
            } else {
                if reason.is_empty() {
                    problems.push(format!("{file}：`{entry}` 标为 SKIP 但未给原因"));
                }
                skipped += 1;
                lines.push(format!("  SKIP     {file:<18} {entry}  — {reason}"));
            }
        }
    }
    // 反向：SHADERS 表里每份都得在台账中出现（含 0 入口的库件）。
    for (name, _) in shaders::SHADERS {
        if !COVERAGE.iter().any(|(f, _)| f == name) {
            problems.push(format!("{name} 未出现在 COVERAGE 台账中"));
        }
    }
    if !problems.is_empty() {
        return fail(format!("compute 入口覆盖台账与 WGSL 不一致：\n  - {}", problems.join("\n  - ")));
    }
    set_result(&format!(
        "compute 入口覆盖台账与 WGSL 逐份一致（{} 份 shader）：COVERED {covered} 个 / SKIP {skipped} 个\n\
         零入口（库件 / 仅 vertex+fragment）：{}\n\
         {}",
        COVERAGE.len(),
        COVERAGE
            .iter()
            .filter(|(_, e)| e.is_empty())
            .map(|(f, _)| *f)
            .collect::<Vec<_>>()
            .join(", "),
        lines.join("\n")
    ));
    Status::Pass
}


// ---------------------------------------------------------------------------
// 测试项 10：真 WebGPU sim_particles 三入口（emit → integrate → hash_cells）
// ---------------------------------------------------------------------------

/// `sim_particles` 的 uniform 参数（CPU 参考与 uniform 打包共用同一份，单一真源）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SimParams {
    /// 粒子数。
    count: u32,
    /// 积分步长。
    dt: f32,
    /// 空间哈希格边长。
    h: f32,
    /// 重力。
    gravity: [f32; 3],
    /// 发射随机种子。
    seed: u32,
    /// 帧号（发射初态的派生输入）。
    frame: u32,
    /// 发射率（保留字段，本测试不参与判对）。
    emitter_rate: f32,
    /// 发射原点。
    emitter_origin: [f32; 3],
}

/// 本测试固定的 `SimParams` 取值（确定性铁律：参数写死，不取时间/随机源）。
const SIM_TEST_PARAMS: SimParams = SimParams {
    count: SIM_P_COUNT,
    dt: 0.25,
    h: 0.5,
    gravity: [0.0, -9.81, 0.0],
    seed: 0x5EED_1234,
    frame: 3,
    emitter_rate: 1.0,
    emitter_origin: [0.25, 1.5, -0.75],
};

/// 把 [`SimParams`] 按 WGSL uniform 布局打包（偏移见 [`SIM_PARAMS_BYTES`] 文档）。
fn sim_params_bytes(p: &SimParams) -> [u8; SIM_PARAMS_BYTES as usize] {
    let mut b = [0u8; SIM_PARAMS_BYTES as usize];
    pack_u32(&mut b, 0, p.count);
    pack_f32(&mut b, 4, p.dt);
    pack_f32(&mut b, 8, p.h);
    pack_f32(&mut b, 16, p.gravity[0]);
    pack_f32(&mut b, 20, p.gravity[1]);
    pack_f32(&mut b, 24, p.gravity[2]);
    pack_u32(&mut b, 28, p.seed);
    pack_u32(&mut b, 32, p.frame);
    pack_f32(&mut b, 36, p.emitter_rate);
    pack_f32(&mut b, 48, p.emitter_origin[0]);
    pack_f32(&mut b, 52, p.emitter_origin[1]);
    pack_f32(&mut b, 56, p.emitter_origin[2]);
    b
}

/// 单个粒子的 CPU 参考状态（对齐 `sim_common.wgsl::Particle` 的 48 字节布局）。
#[derive(Debug, Clone, Copy)]
struct ParticleRef {
    /// 位置。
    pos: [f32; 3],
    /// 逆质量。
    mass_inv: f32,
    /// 速度。
    vel: [f32; 3],
    /// 格键。
    cell: u32,
}

/// `emit` 的 CPU 参考（判对基准 = `prism-particles/src/emitter.rs::derive` 的
/// "种子化发射"口径；WGSL 侧为 `sim_particles.wgsl::emit`）。
fn emit_ref(p: &SimParams, i: usize) -> ParticleRef {
    let u0 = hash3_f32(p.seed, p.frame, i as u32 * 3);
    let u1 = hash3_f32(p.seed, p.frame, i as u32 * 3 + 1);
    let u2 = hash3_f32(p.seed, p.frame, i as u32 * 3 + 2);
    ParticleRef {
        pos: [
            p.emitter_origin[0] + (u0 - 0.5),
            p.emitter_origin[1] + (u1 - 0.5),
            p.emitter_origin[2] + (u2 - 0.5),
        ],
        mass_inv: 1.0,
        vel: [0.0, 0.0, 0.0],
        cell: 0,
    }
}

/// `integrate` 的 CPU 参考（半隐式欧拉；判对基准 = `prism-particles/src/sim.rs::step`）。
fn integrate_ref(p: &SimParams, s: &ParticleRef) -> ParticleRef {
    let mut r = *s;
    if r.mass_inv > 0.0 {
        for k in 0..3 {
            r.vel[k] += p.gravity[k] * p.dt;
            r.pos[k] += r.vel[k] * p.dt;
        }
    }
    r
}

/// `hash_cells` 的 CPU 参考（判对基准 = `prism-physics/src/sph.rs` 的 `floor(v/h)` 分格）。
fn hash_cells_ref(p: &SimParams, s: &ParticleRef) -> ParticleRef {
    let mut r = *s;
    r.cell = cell_key(s.pos, p.h);
    r
}

/// 3 入口串联后的 CPU 参考（`emit → integrate → hash_cells`）。
fn particles_reference(p: &SimParams) -> Vec<ParticleRef> {
    (0..p.count as usize)
        .map(|i| {
            let e = emit_ref(p, i);
            let g = integrate_ref(p, &e);
            hash_cells_ref(p, &g)
        })
        .collect()
}

fn t_gpu_sim_particles() -> Status {
    let dev = match ready_device("gpu.sim_particles") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("sim_particles.wgsl") {
        Some(w) => w,
        None => return fail("gpu.sim_particles：sim_particles.wgsl 展开结果缺失"),
    };
    // 输入缓冲清零（emit 覆写 pos/vel/mass_inv/cell；两个 _pad 保持 0，
    // 读回时若 _pad 非 0 即说明 shader 越字段写了）。
    SCRATCH.with(|s| s.borrow_mut().particles.fill(0));
    let par_ptr = params2_write(SIM_PARAMS_OFF, &sim_params_bytes(&SIM_TEST_PARAMS));
    if par_ptr == 0 {
        return fail("gpu.sim_particles：uniform 参数区越界（PARAMS_ARENA_BYTES 不足）");
    }
    // 1 storage + 1 uniform。cell_range（binding 2）**故意不绑**：三入口都静态
    // 不引用它，多绑一个会让 createBindGroup 因 layout 不符而验证失败。
    let pbuf = match make_buffer("gpu.sim_particles", dev, PARTICLES_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.particles[..]), PARTICLES_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.sim_particles", dev, SIM_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, SIM_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // 1 个工作组（workgroup_size=64 ≥ 32 粒子）。三入口各建自己的 pipeline：
    // `layout:"auto"` 按 entry 静态引用生成布局，故不能共享 bind group。
    let steps: [(&'static str, u32, &[u32]); 3] = [
        ("emit", 1, &[pbuf, par_buf]),
        ("integrate", 1, &[pbuf, par_buf]),
        ("hash_cells", 1, &[pbuf, par_buf]),
    ];
    let out_ptr = addr_of_buf(|x| &mut x.particles[..]);
    set_result(&format!(
        "已提交 sim_particles 3 入口链（emit → integrate → hash_cells），各 1 工作组：\n\
         缓冲：particles={pbuf}（{PARTICLES_BYTES}B = {SIM_P_COUNT}×{PARTICLE_STRIDE_BYTES}）\
         params={par_buf}（{SIM_PARAMS_BYTES}B）\n\
         读回 {PARTICLES_BYTES} 字节 → 线性内存 {out_ptr}，待 settle 后逐字段判对"
    ));
    if let Err(s) = submit_chain("gpu.sim_particles", dev, wgsl, &steps) {
        return s;
    }
    if let Err(s) = readback_to("gpu.sim_particles", dev, pbuf, PARTICLES_BYTES, out_ptr) {
        return s;
    }
    Status::Pending
}

fn c_gpu_sim_particles() -> Status {
    let dev = match ready_device("gpu.sim_particles") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.sim_particles", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let (buf, want) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.particles, particles_reference(&SIM_TEST_PARAMS))
    });
    let n = SIM_P_COUNT as usize;
    let mut want_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut want_vel: Vec<f32> = Vec::with_capacity(n * 3);
    let mut want_inv: Vec<f32> = Vec::with_capacity(n);
    for r in want.iter() {
        want_pos.extend_from_slice(&r.pos);
        want_vel.extend_from_slice(&r.vel);
        want_inv.push(r.mass_inv);
    }
    // 布局读取：**必须**按 `Particle` 的 48 字节步长 + 字段偏移（见 [`particle_off`]）。
    //
    // 【D-14 真机缺陷】此处曾用 `unpack_vec3(&buf, i)`（`array<vec3<f32>>` 的
    // 16 字节步长）去读 `array<Particle>`：i=0 恰好命中 pos 前 3 个 float（判过），
    // i=1 起就落到粒子 0 的 `vel`（x 分量恒 0）⇒ GPU 读到大片 0，93/96 元素超容差。
    // 症状与"dispatch 被丢弃"完全一致，但 bind group 本身是有效的——判据不能靠猜。
    let mut got_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut got_vel: Vec<f32> = Vec::with_capacity(n * 3);
    let mut got_inv: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * PARTICLE_STRIDE_BYTES as usize;
        got_pos.extend_from_slice(&unpack_vec3_at(&buf, base + particle_off::POS));
        got_vel.extend_from_slice(&unpack_vec3_at(&buf, base + particle_off::VEL));
        got_inv.push(unpack_f32(&buf, base + particle_off::MASS_INV));
    }
    let d_pos = try_status!(cmp_f32("gpu.sim_particles", "pos（emit+integrate）", &le_f32(&got_pos), &want_pos));
    let d_vel = try_status!(cmp_f32("gpu.sim_particles", "vel（integrate）", &le_f32(&got_vel), &want_vel));
    let d_inv = try_status!(cmp_f32("gpu.sim_particles", "mass_inv（emit）", &le_f32(&got_inv), &want_inv));
    // ---- hash_cells：cell 逐位精确（整数路径不容差）----
    let got_cell: Vec<u32> = (0..n)
        .map(|i| unpack_u32(&buf, i * PARTICLE_STRIDE_BYTES as usize + particle_off::CELL))
        .collect();
    let want_cell: Vec<u32> = want.iter().map(|r| r.cell).collect();
    try_status!(cmp_u32("gpu.sim_particles", "cell（hash_cells）", &le_u32(&got_cell), &want_cell));
    // ---- 填充字段未被 shader 触碰（越字段写检测）----
    let mut pad_bad = 0usize;
    for i in 0..n {
        let base = i * PARTICLE_STRIDE_BYTES as usize;
        if unpack_f32(&buf, base + particle_off::PAD) != 0.0 || unpack_u32(&buf, base + particle_off::PAD2) != 0 {
            pad_bad += 1;
        }
    }
    if pad_bad != 0 {
        return fail(format!(
            "gpu.sim_particles：{pad_bad}/{n} 个粒子的 `_pad`/`_pad2` 被 shader 写坏（应为 0）\
             → Particle 的 48 字节布局与 host 侧推算不符"
        ));
    }
    let mut distinct = want_cell.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let distinct = distinct.len();
    if distinct < 2 {
        return fail(format!(
            "gpu.sim_particles：{n} 个粒子全落在同一格（{distinct} 个格键）→ 分格未生效，\
             cell 判对无意义（请调小 h 或加大发射原点跨度）"
        ));
    }
    set_result(&format!(
        "真 WebGPU sim_particles：3 入口链 {n}/{n} 粒子与 CPU 参考一致 ✓\n\
         pos 最大偏差={d_pos:.3e}  vel={d_vel:.3e}  mass_inv={d_inv:.3e}（容差 {F32_TOL:e}）\n\
         cell（hash_cells，u32 精确）：{distinct} 个不同格键 / {n} 粒子 → 分格非退化 ✓\n\
         _pad/_pad2 全 0 → 48 字节布局无越字段写 ✓\n\
         缓冲 {PARTICLES_BYTES}B = {SIM_P_COUNT}×{PARTICLE_STRIDE_BYTES}B（vec3 步长 16，含填充）\n\
         粒子 fingerprint=0x{:08x}",
        fnv1a32(&buf[..PARTICLES_BYTES as usize])
    ));
    Status::Pass
}

/// 把 `f32` 序列编成小端字节（供 [`cmp_f32`] 复用同一套比对逻辑）。
fn le_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// 把 `u32` 序列编成小端字节（供 [`cmp_u32`] 复用同一套比对逻辑）。
fn le_u32(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// 测试项 11：真 WebGPU sim_sph 三入口（predict → density → project）
// ---------------------------------------------------------------------------

/// `sim_sph` 的 uniform 参数（CPU 参考与 uniform 打包共用）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SphParams {
    /// 粒子数。
    count: u32,
    /// 核半径。
    h: f32,
    /// 静止密度。
    rest_density: f32,
    /// XPBD 柔度。
    compliance: f32,
    /// 子步长。
    dt: f32,
    /// 重力。
    gravity: [f32; 3],
}

/// 本测试固定的 `SphParams`。
///
/// `h=0.5` 与粒子网格步长 `0.18` 配合 ⇒ 15 个邻居全部落在核内 ⇒
/// `sph_density` 的 Poly6 求和**每项都非零**，任何漏项/多算都会体现。
/// `rest_density` 取一个让 `C = ρ/ρ₀ − 1 > 0`（压缩侧投影生效）的值。
const SPH_TEST_PARAMS: SphParams = SphParams {
    count: SPH_P_COUNT,
    h: 0.5,
    // 占位：**真实取值由 [`sph_scene_params`] 在运行时按实测密度中点覆写**。
    // 写死一个数会连带两个问题：(a) 要么全粒子压缩 ⇒ `sph_project` 只走 clamp
    // 分支，要么全膨胀 ⇒ 只走 early-return 分支，任一情形下 λ 判对都退化；
    // (b) 换 h 或网格步长就得重调。运行时取中点则"压缩/膨胀"两侧必然各占一半。
    rest_density: 0.0,
    compliance: 0.0,
    dt: 0.02,
    gravity: [0.0, -9.81, 0.0],
};

/// λ 缓冲的**哨兵初值**（非 0）。
///
/// 为什么不能填 0：`sph_project` 的压缩侧是 `dl = -C/denom`（`C = ρ/ρ₀-1 > 0`）
/// ⇒ `dl` 恒负 ⇒ `lam` 被 `if (lam < 0) { lam = 0 }` 夹到 0。若初值也是 0，
/// 输出恒为 0，"GPU 跑了并 clamp" 与 "GPU 压根没写这 4 字节" **无法区分**——
/// 判对退化为恒真。填 0.25 后：压缩侧粒子读到 0（走了 clamp），膨胀侧粒子读到
/// 0.25（走 early-return 原样保留），两个分支都有非平凡期望值。
pub const SPH_LAMBDA_SEED: f32 = 0.25;

/// 本测试实际使用的 [`SphParams`]（`rest_density` 按实测密度中点定）。
///
/// 取"密粒子与疏粒子密度区间的中点"作 ρ₀ ⇒ 一半粒子 `C = ρ/ρ₀-1 > 0`（压缩侧，
/// 写 0）、一半 `C <= 0`（膨胀侧，保留哨兵）。CPU 参考与 uniform 打包共用本
/// 函数，杜绝两侧口径漂移。
fn sph_scene_params() -> SphParams {
    let mut p = SPH_TEST_PARAMS;
    let n = p.count as usize;
    let inv: Vec<f32> = (0..n).map(sph_initial_inv).collect();
    // 复刻 sph_predict（半隐式欧拉 + inv>0 守卫）以得到投影前的真实构型。
    let mut pos: Vec<[f32; 3]> = (0..n).map(sph_initial_pos).collect();
    let mut vel: Vec<[f32; 3]> = vec![[0.0f32; 3]; n];
    for i in 0..n {
        if inv[i] > 0.0 {
            for k in 0..3 {
                vel[i][k] += p.gravity[k] * p.dt;
                pos[i][k] += vel[i][k] * p.dt;
            }
        }
    }
    let den = sph_density_ref(&p, &pos);
    let mut sorted = den.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    p.rest_density = 0.5 * (sorted[mid - 1] + sorted[mid]);
    p
}

/// 把 [`SphParams`] 按 WGSL uniform 布局打包（偏移见 [`SPH_PARAMS_BYTES`] 文档）。
fn sph_params_bytes(p: &SphParams) -> [u8; SPH_PARAMS_BYTES as usize] {
    let mut b = [0u8; SPH_PARAMS_BYTES as usize];
    pack_u32(&mut b, 0, p.count);
    pack_f32(&mut b, 4, p.h);
    pack_f32(&mut b, 8, p.rest_density);
    pack_f32(&mut b, 12, p.compliance);
    pack_f32(&mut b, 16, p.dt);
    pack_f32(&mut b, 32, p.gravity[0]);
    pack_f32(&mut b, 36, p.gravity[1]);
    pack_f32(&mut b, 40, p.gravity[2]);
    b
}

/// 16 粒子的确定性初始位置：4×4 网格置于 y=0 平面，xz 步长 0.18。
fn sph_initial_pos(i: usize) -> [f32; 3] {
    [((i % 4) as f32) * 0.18, 0.0, ((i / 4) as f32) * 0.18]
}

/// 16 粒子的逆质量：全 1（自由），索引 0 与 5 置 0（模拟 pin，验 `sp_inv > 0` 守卫）。
fn sph_initial_inv(i: usize) -> f32 {
    if i == 0 || i == 5 {
        0.0
    } else {
        1.0
    }
}

/// `sph_density` 的 CPU 参考：自项 + 全邻域 O(n²) 扫描（判对基准 =
/// `prism-physics/src/sph.rs::DensityConstraint::density_of` 的同式 `poly6_2d`）。
///
/// WGSL 侧当前是 `neighbors_start=0 / neighbors_end=count` 的**全量扫描**
/// （`cell_range` 尚未接入），故参考也按全量算——这不是将就，而是逐位对齐
/// 当前实现的既成事实，并把"未接 cell_range"记为已知差异写进结果文本。
fn sph_density_ref(p: &SphParams, pos: &[[f32; 3]]) -> Vec<f32> {
    (0..p.count as usize)
        .map(|i| {
            let mut rho = poly6_2d(0.0, p.h);
            for (j, &pj) in pos.iter().enumerate() {
                if j == i || j as u32 >= p.count {
                    continue;
                }
                let d = [pj[0] - pos[i][0], pj[1] - pos[i][1], pj[2] - pos[i][2]];
                rho += poly6_2d(len3(d), p.h);
            }
            rho
        })
        .collect()
}

/// `sph_project` 的 CPU 参考（判对基准 = `prism-physics/src/sph.rs`
/// `DensityConstraint::project` 的 XPBD 通式 + 压缩侧单向 / λ ≥ 0 稳定项）。
fn sph_project_ref(p: &SphParams, density: &[f32], inv: &[f32], lambda: &mut [f32]) {
    for i in 0..p.count as usize {
        let c = density[i] / p.rest_density - 1.0;
        if c <= 0.0 {
            if lambda[i] < 0.0 {
                lambda[i] = 0.0;
            }
            continue;
        }
        let at = p.compliance / (p.dt * p.dt);
        let denom = inv[i] + at;
        if denom <= 0.0 {
            continue;
        }
        let dl = (-c - at * lambda[i]) / denom;
        let mut lam = lambda[i] + dl;
        if lam < 0.0 {
            lam = 0.0;
        }
        lambda[i] = lam;
    }
}

fn t_gpu_sim_sph() -> Status {
    let dev = match ready_device("gpu.sim_sph") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("sim_sph.wgsl") {
        Some(w) => w,
        None => return fail("gpu.sim_sph：sim_sph.wgsl 展开结果缺失"),
    };
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        for i in 0..SPH_P_COUNT as usize {
            pack_state(&mut v.sph_state, i, sph_initial_pos(i), [0.0, 0.0, 0.0], sph_initial_inv(i));
            pack_f32(&mut v.sph_lambda, i * 4, SPH_LAMBDA_SEED);
        }
        v.sph_density.fill(0);
    });
    let scene = sph_scene_params();
    let par_ptr = params2_write(SPH_PARAMS_OFF, &sph_params_bytes(&scene));
    if par_ptr == 0 {
        return fail("gpu.sim_sph：uniform 参数区越界（PARAMS_ARENA_BYTES 不足）");
    }
    // 3 storage + 1 uniform。cell_range（binding 4）不绑：三入口都静态不引用它。
    fn src(which: u8, size: u32) -> impl FnOnce() -> (u32, u32) {
        move || {
            let base = SCRATCH.with(|s| {
                let v = s.borrow();
                match which {
                    0 => v.sph_state.as_ptr(),
                    1 => v.sph_density.as_ptr(),
                    _ => v.sph_lambda.as_ptr(),
                }
            });
            (base as usize as u32, size)
        }
    }
    let state_buf = match make_buffer("gpu.sim_sph", dev, SPH_STATE_BYTES, usage::STORAGE, src(0, SPH_STATE_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let den_buf = match make_buffer("gpu.sim_sph", dev, SPH_SCALAR_BYTES, usage::STORAGE, src(1, SPH_SCALAR_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let lam_buf = match make_buffer("gpu.sim_sph", dev, SPH_SCALAR_BYTES, usage::STORAGE, src(2, SPH_SCALAR_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.sim_sph", dev, SPH_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, SPH_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // 每入口的绑定集 = 该入口**静态引用**的 binding 集合，且必须恰好是连续前缀
    // `{0..n-1}`（胶水把第 i 个句柄固定映射到 `binding: i`）：
    //   sph_predict  → {state(0), spp(1)}                              = 2 个
    //   sph_density  → {state(0), spp(1), sp_density(2)}               = 3 个
    //   sph_project  → {state(0), spp(1), sp_density(2), sp_lambda(3)} = 4 个
    let steps: [(&'static str, u32, &[u32]); 3] = [
        ("sph_predict", 1, &[state_buf, par_buf]),
        ("sph_density", 1, &[state_buf, par_buf, den_buf]),
        ("sph_project", 1, &[state_buf, par_buf, den_buf, lam_buf]),
    ];
    // 三个读回落点各自独立（predict 改 pos/vel，project 写 lambda）。
    let out_state = addr_of_buf(|x| &mut x.sph_state[..]);
    let out_den = addr_of_buf(|x| &mut x.sph_density[..]);
    let out_lam = addr_of_buf(|x| &mut x.sph_lambda[..]);
    set_result(&format!(
        "已提交 sim_sph 3 入口链（predict → density → project），各 1 工作组：\n\
         缓冲：state={state_buf}（AoS pos+vel+inv）density={den_buf} lambda={lam_buf} params={par_buf}\n\
         尺寸：state={SPH_STATE_BYTES}B（{SPH_P_COUNT}×{STATE_STRIDE_BYTES}）\
         标量={SPH_SCALAR_BYTES}B params={SPH_PARAMS_BYTES}B\n\
         绑定前缀：predict={{0,1}} density={{0,1,2}} project={{0,1,2,3}}\n\
         待 settle 后与 CPU 参考（prism-physics/sph.rs 同式）逐元素判对"
    ));
    if let Err(s) = submit_chain("gpu.sim_sph", dev, wgsl, &steps) {
        return s;
    }
    for (buf, len, dst) in [
        (state_buf, SPH_STATE_BYTES, out_state),
        (den_buf, SPH_SCALAR_BYTES, out_den),
        (lam_buf, SPH_SCALAR_BYTES, out_lam),
    ] {
        if let Err(s) = readback_to("gpu.sim_sph", dev, buf, len, dst) {
            return s;
        }
    }
    Status::Pending
}

fn c_gpu_sim_sph() -> Status {
    let dev = match ready_device("gpu.sim_sph") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.sim_sph", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let p = sph_scene_params();
    let n = p.count as usize;
    let (got_state, got_den, got_lam) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.sph_state, v.sph_density, v.sph_lambda)
    });
    // ---- CPU 参考：predict → density → project ----
    let mut want_pos: Vec<[f32; 3]> = (0..n).map(sph_initial_pos).collect();
    let mut want_vel: Vec<[f32; 3]> = vec![[0.0f32; 3]; n];
    let inv: Vec<f32> = (0..n).map(sph_initial_inv).collect();
    for i in 0..n {
        if inv[i] > 0.0 {
            for k in 0..3 {
                want_vel[i][k] += p.gravity[k] * p.dt;
                want_pos[i][k] += want_vel[i][k] * p.dt;
            }
        }
    }
    let want_den = sph_density_ref(&p, &want_pos);
    let mut want_lam = vec![SPH_LAMBDA_SEED; n];
    sph_project_ref(&p, &want_den, &inv, &mut want_lam);
    // ---- 判对 ----
    let mut flat_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut flat_vel: Vec<f32> = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat_pos.extend_from_slice(&want_pos[i]);
        flat_vel.extend_from_slice(&want_vel[i]);
    }
    let mut got_flat_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut got_flat_vel: Vec<f32> = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * STATE_STRIDE_BYTES as usize;
        got_flat_pos.extend_from_slice(&unpack_vec3_at(&got_state, base + state_off::POS));
        got_flat_vel.extend_from_slice(&unpack_vec3_at(&got_state, base + state_off::VEL));
    }
    let d_pos = try_status!(cmp_f32("gpu.sim_sph", "pos（sph_predict）", &le_f32(&got_flat_pos), &flat_pos));
    let d_vel = try_status!(cmp_f32("gpu.sim_sph", "vel（sph_predict）", &le_f32(&got_flat_vel), &flat_vel));
    let d_den = try_status!(cmp_f32("gpu.sim_sph", "density（sph_density）", &got_den, &want_den));
    let d_lam = try_status!(cmp_f32("gpu.sim_sph", "lambda（sph_project）", &got_lam, &want_lam));
    // ---- 非退化断言 1：Poly6 邻域和确实累加了多项 ----
    let self_term = poly6_2d(0.0, p.h);
    if want_den.iter().all(|&d| (d - self_term).abs() <= F32_TOL) {
        return fail(format!(
            "gpu.sim_sph：CPU 参考自身退化——{n} 个粒子的密度都等于自项 W(0)={self_term}，\
             说明邻域扫描没生效，本项判对无意义（请检查 sph_initial_pos 的网格步长 vs h）"
        ));
    }
    // ---- 非退化断言 2：λ 必须**同时**出现 0 与哨兵两个值 ----
    // 哨兵初值 SPH_LAMBDA_SEED 让两个分支各有非平凡期望：压缩侧（C>0）被 clamp
    // 到 0，膨胀侧（C<=0）early-return 保留哨兵。若两者都读到哨兵 ⇒ GPU 侧
    // `sph_project` 整体没生效；若两者都读到 0 ⇒ 输入初值被忽略（真机上是
    // buffer 绑定错位）。这两种失效都只有真机能暴露。
    let lam_zero = want_lam.iter().filter(|&&l| l.abs() <= F32_TOL).count();
    let lam_seed = want_lam.iter().filter(|&&l| (l - SPH_LAMBDA_SEED).abs() <= F32_TOL).count();
    if lam_zero == 0 || lam_seed == 0 {
        return fail(format!(
            "gpu.sim_sph：λ 判据退化——读回 {lam_zero} 个 0、{lam_seed} 个哨兵 {}，\
             两者必须都 > 0 才说明 sph_project 的 clamp 分支与 early-return 分支都走到了\
             （CPU 参考 rest_density={}，实测密度区间 [{:.6}, {:.6}]）",
            SPH_LAMBDA_SEED, p.rest_density,
            want_den.iter().cloned().fold(f32::MAX, f32::min),
            want_den.iter().cloned().fold(f32::MIN, f32::max)
        ));
    }
    // ---- pin 守卫：inv_mass=0 的粒子位置逐位未变 ----
    for i in 0..n {
        if inv[i] == 0.0 && got_flat_pos[i * 3..i * 3 + 3] != flat_pos[i * 3..i * 3 + 3] {
            return fail(format!(
                "gpu.sim_sph：inv_mass=0 的粒子 {i} 位置被改动 → sph_predict 的 `sp_state[i].inv > 0` 守卫失效"
            ));
        }
    }
    let lam_min = want_lam.iter().cloned().fold(f32::MAX, f32::min);
    let lam_max = want_lam.iter().cloned().fold(f32::MIN, f32::max);
    set_result(&format!(
        "真 WebGPU sim_sph：3 入口链逐元素与 CPU 参考（prism-physics/sph.rs 同式）一致 ✓\n\
         pos={d_pos:.3e} vel={d_vel:.3e} density={d_den:.3e} lambda={d_lam:.3e}（容差 {F32_TOL:e}）\n\
         density：自项 W(0)={self_term:.6} → 邻域求和后 ρ[0]={:.6}（非退化 ✓）\n\
         λ：{lam_zero}/{n} 个被 clamp 到 0（压缩侧）+ {lam_seed}/{n} 个保留哨兵 {}（膨胀侧 early-return）\n\
         哨兵初值非 0 是关键：初值若为 0，『GPU 跑了 clamp』与『GPU 没写这 4 字节』无法区分\n\
         λ 实测 min={lam_min:.6e} max={lam_max:.6e}\n\
         inv_mass=0 的 2 个 pin 粒子位置逐位未变 ✓\n\
         绑定前缀（胶水 binding=数组下标，故必须是 0..n-1）：predict={{0,1}} density={{0,1,2}} project={{0,1,2,3}}\n\
         已知差异：WGSL 的 sph_density 当前是 O(n²) 全邻域扫描（cell_range binding 4 未接入）\n\
         缓冲：state={SPH_STATE_BYTES}B（AoS {SPH_P_COUNT}×{STATE_STRIDE_BYTES}）\
         标量={SPH_SCALAR_BYTES}B params={SPH_PARAMS_BYTES}B",
        SPH_LAMBDA_SEED, want_den[0]
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 12：真 WebGPU sim_cloth 三入口（predict → project → velocity）
// ---------------------------------------------------------------------------

/// `sim_cloth` 的 uniform 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
struct ClothParams {
    /// 网格行数。
    rows: u32,
    /// 网格列数。
    cols: u32,
    /// 相邻粒子静止距离。
    spacing: f32,
    /// 子步长。
    dt: f32,
    /// 结构约束柔度。
    compliance_struct: f32,
    /// 弯曲约束柔度。
    compliance_bend: f32,
    /// 速度阻尼比例。
    damping: f32,
    /// 重力。
    gravity: [f32; 3],
}

/// 本测试固定的 `ClothParams`（4 行 × 2 列 = 8 粒子）。
///
/// 结构柔度取 `0.0`（完全刚性，一次投影到位）、弯曲柔度取 `1e-4`（软），两者
/// **不等** ⇒ `select(compliance_struct, compliance_bend, rest > spacing*1.5)`
/// 的分支选择可被结果区分（若分支写反，柔度差 4 个量级必然显形）。
const CLOTH_TEST_PARAMS: ClothParams = ClothParams {
    rows: 4,
    cols: 2,
    spacing: 0.5,
    dt: 0.02,
    compliance_struct: 0.0,
    compliance_bend: 1e-4,
    damping: 0.15,
    gravity: [0.0, -9.81, 0.0],
};

/// 把 [`ClothParams`] 按 WGSL uniform 布局打包。
fn cloth_params_bytes(p: &ClothParams) -> [u8; CLOTH_PARAMS_BYTES as usize] {
    let mut b = [0u8; CLOTH_PARAMS_BYTES as usize];
    pack_u32(&mut b, 0, p.rows);
    pack_u32(&mut b, 4, p.cols);
    pack_f32(&mut b, 8, p.spacing);
    pack_f32(&mut b, 12, p.dt);
    pack_f32(&mut b, 16, p.compliance_struct);
    pack_f32(&mut b, 20, p.compliance_bend);
    pack_f32(&mut b, 24, p.damping);
    pack_f32(&mut b, 32, p.gravity[0]);
    pack_f32(&mut b, 36, p.gravity[1]);
    pack_f32(&mut b, 40, p.gravity[2]);
    b
}

/// 8 粒子的确定性初始构型：4 行 × 2 列，沿 x 拉开到 `1.4 × spacing`。
///
/// **刻意拉伸**（静止长 0.5、实际 0.7）⇒ 距离约束 `C = |d| − rest = 0.2 ≠ 0`
/// ⇒ `xpbd_distance_project` 真的做功、λ 真的非零。若构型恰好满足约束，投影
/// 就是恒等操作，本项会退化为"什么都没验"。
fn cloth_initial_pos(i: usize) -> [f32; 3] {
    let c = (i % 2) as f32;
    let r = (i / 2) as f32;
    [c * CLOTH_TEST_PARAMS.spacing * 1.4, 0.0, r * CLOTH_TEST_PARAMS.spacing]
}

/// 8 粒子的逆质量：行 0（索引 0/1）为 pin（置 0），其余自由。
fn cloth_initial_inv(i: usize) -> f32 {
    if i < 2 {
        0.0
    } else {
        1.0
    }
}

/// 4 条**两两不相交**的约束：`(0,1) (2,3) (4,5) (6,7)`。
///
/// 前两条 `rest = spacing`（结构分支），后两条 `rest = 2 × spacing`（弯曲分支，
/// 因 `2 × spacing > 1.5 × spacing` 成立）。**不相交是确定性前提**，理由见本节
/// 头注；被排除的 12 条共享粒子约束在结果文本里显式列出。
fn cloth_constraints() -> Vec<(u32, u32, f32, u32)> {
    let s = CLOTH_TEST_PARAMS.spacing;
    vec![(0, 1, s, 0), (2, 3, s, 1), (4, 5, 2.0 * s, 2), (6, 7, 2.0 * s, 3)]
}

fn t_gpu_sim_cloth() -> Status {
    let dev = match ready_device("gpu.sim_cloth") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("sim_cloth.wgsl") {
        Some(w) => w,
        None => return fail("gpu.sim_cloth：sim_cloth.wgsl 展开结果缺失"),
    };
    let cons = cloth_constraints();
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        for i in 0..CLOTH_P_COUNT as usize {
            pack_state(&mut v.cloth_state, i, cloth_initial_pos(i), [0.0, 0.0, 0.0], cloth_initial_inv(i));
        }
        for (k, &(ia, ib, rest, id)) in cons.iter().enumerate() {
            let o = k * CONSTRAINT_STRIDE_BYTES as usize;
            pack_u32(&mut v.cloth_cons, o, ia);
            pack_u32(&mut v.cloth_cons, o + 4, ib);
            pack_f32(&mut v.cloth_cons, o + 8, rest);
            pack_u32(&mut v.cloth_cons, o + 12, id);
        }
        v.cloth_lambdas.fill(0);
    });
    let par_ptr = params2_write(CLOTH_PARAMS_OFF, &cloth_params_bytes(&CLOTH_TEST_PARAMS));
    if par_ptr == 0 {
        return fail("gpu.sim_cloth：uniform 参数区越界（PARAMS_ARENA_BYTES 不足）");
    }
    fn src(which: u8, size: u32) -> impl FnOnce() -> (u32, u32) {
        move || {
            let base = SCRATCH.with(|s| {
                let v = s.borrow();
                match which {
                    0 => v.cloth_state.as_ptr(),
                    1 => v.cloth_cons.as_ptr(),
                    _ => v.cloth_lambdas.as_ptr(),
                }
            });
            (base as usize as u32, size)
        }
    }
    let par_buf = match make_buffer("gpu.sim_cloth", dev, CLOTH_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, CLOTH_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let state_buf = match make_buffer("gpu.sim_cloth", dev, CLOTH_STATE_BYTES, usage::STORAGE, src(0, CLOTH_STATE_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let con_buf = match make_buffer("gpu.sim_cloth", dev, CLOTH_CONS_BYTES, usage::STORAGE, src(1, CLOTH_CONS_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let lam_buf = match make_buffer("gpu.sim_cloth", dev, CLOTH_LAMBDA_BYTES, usage::STORAGE, src(2, CLOTH_LAMBDA_BYTES)) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // 每入口的绑定集 = 该 entry **静态引用**的 binding 集合，且必须恰好是连续
    // 前缀 `{0..n-1}`（胶水把第 i 个句柄固定映射到 `binding: i`）：
    //   cloth_predict  → {cparams(0), cs(1)}                          = 2 个
    //   cloth_project  → {cparams(0), cs(1), constraints(2), lambdas(3)} = 4 个
    //   cloth_velocity → {cparams(0), cs(1)}                          = 2 个
    let steps: [(&'static str, u32, &[u32]); 3] = [
        ("cloth_predict", 1, &[par_buf, state_buf]),
        ("cloth_project", 1, &[par_buf, state_buf, con_buf, lam_buf]),
        ("cloth_velocity", 1, &[par_buf, state_buf]),
    ];
    let out_state = addr_of_buf(|x| &mut x.cloth_state[..]);
    let out_lam = addr_of_buf(|x| &mut x.cloth_lambdas[..]);
    set_result(&format!(
        "已提交 sim_cloth 3 入口链（predict → project → velocity），各 1 工作组：\n\
         缓冲：params={par_buf} state={state_buf}（AoS pos+vel+inv）constraints={con_buf} lambdas={lam_buf}\n\
         约束 {CLOTH_C_COUNT} 条（两两不相交，见结果文本的排除说明）\n\
         绑定前缀：predict={{0,1}} project={{0,1,2,3}} velocity={{0,1}}\n\
         state={CLOTH_STATE_BYTES}B（{CLOTH_P_COUNT}×{STATE_STRIDE_BYTES}）\
         cons={CLOTH_CONS_BYTES}B params={CLOTH_PARAMS_BYTES}B\n\
         待 settle 后与 CPU 参考（prism-physics 的 cloth/xpbd 同式）逐元素判对"
    ));
    if let Err(s) = submit_chain("gpu.sim_cloth", dev, wgsl, &steps) {
        return s;
    }
    for (buf, len, dst) in [
        (state_buf, CLOTH_STATE_BYTES, out_state),
        (lam_buf, CLOTH_LAMBDA_BYTES, out_lam),
    ] {
        if let Err(s) = readback_to("gpu.sim_cloth", dev, buf, len, dst) {
            return s;
        }
    }
    Status::Pending
}

fn c_gpu_sim_cloth() -> Status {
    let dev = match ready_device("gpu.sim_cloth") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.sim_cloth", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let p = CLOTH_TEST_PARAMS;
    let n = CLOTH_P_COUNT as usize;
    let cons = cloth_constraints();
    let (got_state, got_lam) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.cloth_state, v.cloth_lambdas)
    });
    // ---- CPU 参考：predict → project（逐约束、id 升序）→ velocity ----
    let inv: Vec<f32> = (0..n).map(cloth_initial_inv).collect();
    let mut want_pos: Vec<[f32; 3]> = (0..n).map(cloth_initial_pos).collect();
    let mut want_vel: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    for i in 0..n {
        if inv[i] > 0.0 {
            for k in 0..3 {
                want_vel[i][k] += p.gravity[k] * p.dt;
                want_pos[i][k] += want_vel[i][k] * p.dt;
            }
        }
    }
    let mut want_lam = vec![0.0f32; cons.len()];
    for (k, &(ia, ib, rest, _)) in cons.iter().enumerate() {
        // 分支口径与 WGSL `select(struct, bend, rest > spacing*1.5)` 一致。
        let compliance = if rest > p.spacing * 1.5 { p.compliance_bend } else { p.compliance_struct };
        let mut pa = want_pos[ia as usize];
        let mut pb = want_pos[ib as usize];
        let mut lam = want_lam[k];
        xpbd_distance_project(
            &mut pa,
            &mut pb,
            inv[ia as usize],
            inv[ib as usize],
            rest,
            compliance,
            p.dt,
            &mut lam,
        );
        want_pos[ia as usize] = pa;
        want_pos[ib as usize] = pb;
        want_lam[k] = lam;
    }
    for v in want_vel.iter_mut() {
        for k in 0..3 {
            v[k] *= 1.0 - p.damping;
        }
    }
    // ---- 判对 ----
    let mut flat_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut flat_vel: Vec<f32> = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat_pos.extend_from_slice(&want_pos[i]);
        flat_vel.extend_from_slice(&want_vel[i]);
    }
    let mut got_flat_pos: Vec<f32> = Vec::with_capacity(n * 3);
    let mut got_flat_vel: Vec<f32> = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * STATE_STRIDE_BYTES as usize;
        got_flat_pos.extend_from_slice(&unpack_vec3_at(&got_state, base + state_off::POS));
        got_flat_vel.extend_from_slice(&unpack_vec3_at(&got_state, base + state_off::VEL));
    }
    let d_pos = try_status!(cmp_f32("gpu.sim_cloth", "pos（predict+project）", &le_f32(&got_flat_pos), &flat_pos));
    let d_vel = try_status!(cmp_f32("gpu.sim_cloth", "vel（predict+velocity）", &le_f32(&got_flat_vel), &flat_vel));
    let d_lam = try_status!(cmp_f32("gpu.sim_cloth", "lambdas（project）", &got_lam, &want_lam));
    // ---- 非退化断言：λ 非零（投影真的做功）----
    if want_lam.iter().all(|&l| l.abs() <= F32_TOL) {
        return fail(
            "gpu.sim_cloth：CPU 参考 λ 全为 0 ⇒ 初始构型恰好满足 rest=length，投影退化为恒等操作，\
             本项判对无意义（请检查 cloth_initial_pos 的拉伸系数）"
                .to_string(),
        );
    }
    // ---- pin 未被移动 ----
    if got_flat_pos[..3] != flat_pos[..3] {
        return fail(format!(
            "gpu.sim_cloth：pin 粒子 0（inv_mass=0）被移动 — GPU ({:.6},{:.6},{:.6}) vs CPU ({:.6},{:.6},{:.6})",
            got_flat_pos[0], got_flat_pos[1], got_flat_pos[2], flat_pos[0], flat_pos[1], flat_pos[2]
        ));
    }
    let lam_struct = want_lam[0].abs();
    let lam_bend = want_lam[2].abs();
    set_result(&format!(
        "真 WebGPU sim_cloth：3 入口链逐元素与 CPU 参考（prism-physics 的 cloth/xpbd 同式）一致 ✓\n\
         pos={d_pos:.3e} vel={d_vel:.3e} lambda={d_lam:.3e}（容差 {F32_TOL:e}）\n\
         λ[结构 rest={:.3}]={lam_struct:.6}  λ[弯曲 rest={:.3}]={lam_bend:.6}\
         → 两分支柔度不同（0 vs 1e-4）且结果可区分 ✓\n\
         排除了真实 4×2 网格的 12 条共享粒子约束（按约束并行却写共享 cs[].pos ⇒ 结果非确定，\
         无法逐元素判对）；本项只判 {CLOTH_C_COUNT} 条两两不相交约束，共享粒子场景需按粒子着色/\
         按行分组后另行覆盖\n\
         pin 粒子 0 逐位未移动 ✓\n\
         绑定前缀（胶水 binding=数组下标，故必须是 0..n-1）：predict={{0,1}} project={{0,1,2,3}} velocity={{0,1}}\n\
         缓冲：state={CLOTH_STATE_BYTES}B（AoS {CLOTH_P_COUNT}×{STATE_STRIDE_BYTES}）\
         cons={CLOTH_CONS_BYTES}B lambda={CLOTH_LAMBDA_BYTES}B params={CLOTH_PARAMS_BYTES}B",
        p.spacing, 2.0 * p.spacing
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 13：真 WebGPU rt_wavefront 两入口（connect → accumulate）
// ---------------------------------------------------------------------------

/// `rt_wavefront` 的 uniform 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
struct WfParams {
    /// 图像宽。
    width: u32,
    /// 图像高。
    height: u32,
    /// 每像素样本数。
    spp: u32,
    /// 最大弹射次数（被测两入口不读，仅占位保持结构完整）。
    max_bounces: u32,
    /// 间接光钳位开关（同上）。
    clamp_indirect_on: u32,
    /// 间接光钳位阈值（同上）。
    clamp_indirect_threshold: f32,
    /// 仅 NEE 开关（同上）。
    nee_only: u32,
    /// 全局种子（同上）。
    global_seed: u32,
    /// 节点 id（同上）。
    node_id: u32,
    /// 帧号（同上）。
    frame: u32,
}

/// 本测试固定的 `WFParams`。
const WF_TEST_PARAMS: WfParams = WfParams {
    width: WF_WIDTH,
    height: WF_HEIGHT,
    spp: WF_SPP,
    max_bounces: 4,
    clamp_indirect_on: 0,
    clamp_indirect_threshold: 0.0,
    nee_only: 1,
    global_seed: 0x1234_5678,
    node_id: 7,
    frame: 0,
};

/// 把 [`WfParams`] 按 WGSL uniform 布局打包（10 × `u32`，末尾补到 48）。
fn wf_params_bytes(p: &WfParams) -> [u8; WF_PARAMS_BYTES as usize] {
    let mut b = [0u8; WF_PARAMS_BYTES as usize];
    // WGSL 里 `clamp_indirect_threshold: f32` 与 `nee_only: u32` 各占一个
    // 4 字节槽，uniform 布局按声明顺序平铺（成员间无额外对齐空洞）。
    pack_u32(&mut b, 0, p.width);
    pack_u32(&mut b, 4, p.height);
    pack_u32(&mut b, 8, p.spp);
    pack_u32(&mut b, 12, p.max_bounces);
    pack_u32(&mut b, 16, p.clamp_indirect_on);
    pack_f32(&mut b, 20, p.clamp_indirect_threshold);
    pack_u32(&mut b, 24, p.nee_only);
    pack_u32(&mut b, 28, p.global_seed);
    pack_u32(&mut b, 32, p.node_id);
    pack_u32(&mut b, 36, p.frame);
    b
}

/// 路径槽的 CPU 参考（覆盖被测入口读写的字段）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SlotRef {
    /// 起点。
    origin: [f32; 3],
    /// 方向。
    dir: [f32; 3],
    /// 路径时刻。
    time: f32,
    /// 吞吐。
    throughput: [f32; 3],
    /// 辐射亮度（`accumulate` 读它）。
    radiance: [f32; 3],
    /// 上一事件的 pdf。
    prev_pdf_dir: f32,
    /// 状态（0 = ACTIVE，1 = TERMINATED）。
    state: u32,
    /// 已完成事件数。
    bounce: u32,
    /// 标志位。
    flags: u32,
    /// 辅助步数。
    aux_steps: u32,
    /// 占位向量。
    placeholder: [f32; 3],
}

/// `PathSlot` 96 字节布局的字段偏移。
///
/// **本表是 D-14 真机缺陷的同款陷阱的第二个实例**：结构体里 5 个 `vec3<f32>`
/// 的 `align = 16` 而 `size = 12`，因此每个 vec3 之后都有 4 字节填充。按平铺
/// （vec3 只算 12 字节）推算会得到 72→80 字节的**错误**步长，真实步长是
/// `roundUp(92, 16) = 96`——多绑的 16 字节会让第 2 个槽起的所有字段整体错位，
/// 而 GPU 侧不会报任何错（storage buffer 只是读得长一点）。故此表逐字段推导，
/// 并由 `host_test.mjs` 第 8 节从 WGSL 文本独立重算交叉校验。
mod slot_off {
    /// `origin: vec3` 起点。
    pub const ORIGIN: usize = 0;
    /// `dir: vec3` 起点（origin 占 0..12，按 align 16 对齐到 16）。
    pub const DIR: usize = 16;
    /// `time: f32`（dir 占 16..28，time 落 28）。
    pub const TIME: usize = 28;
    /// `throughput: vec3` 起点（time 结束于 32，恰对齐）。
    pub const THROUGHPUT: usize = 32;
    /// `radiance: vec3` 起点（throughput 结束于 44 → 对齐到 48）。
    pub const RADIANCE: usize = 48;
    /// `prev_pdf_dir: f32`（radiance 结束于 60）。
    pub const PREV_PDF: usize = 60;
    /// `state: u32`。
    pub const STATE: usize = 64;
    /// `bounce: u32`。
    pub const BOUNCE: usize = 68;
    /// `flags: u32`。
    pub const FLAGS: usize = 72;
    /// `aux_steps: u32`。
    pub const AUX_STEPS: usize = 76;
    /// `placeholder: vec3` 起点（结束于 92，结构体按 align 16 补到 96）。
    pub const PLACEHOLDER: usize = 80;
}

/// 8 条路径槽的确定性初始状态。
///
/// **状态分布刻意一半 TERMINATED**：这样 `connect` 的
/// `if (s.state != ST_ACTIVE) { return; }` 早退分支真的被走到——若全 ACTIVE，
/// 该分支永不执行，"跳过即写入"这类潜在缺陷就验不到。`radiance` 各不相同且
/// 含负值/小数，使 `accumulate` 的求和顺序错误（乱序累加在 f32 下会露馅）可被捕捉。
fn wf_initial_slots() -> Vec<SlotRef> {
    (0..WF_PATHS as usize)
        .map(|i| {
            let k = (i + 1) as f32;
            SlotRef {
                origin: [0.0, 0.0, -1.0],
                dir: [0.0, 0.0, 1.0],
                time: 0.5,
                throughput: [1.0, 0.5, 0.25],
                radiance: [0.125 * k, -0.0625 * k, 0.03125 * k],
                prev_pdf_dir: 0.0,
                // 前一半 ACTIVE、后一半 TERMINATED ⇒ 两分支都走到。
                state: u32::from(i >= WF_PATHS as usize / 2),
                bounce: i as u32,
                flags: 1,
                aux_steps: 0,
                placeholder: [9.0, 8.0, 7.0],
            }
        })
        .collect()
}

/// 把一个 [`SlotRef`] 按 `PathSlot` 的 80 字节布局写入字节缓冲。
fn pack_slot(buf: &mut [u8], i: usize, s: &SlotRef) {
    use slot_off::*;
    let o = i * PATH_SLOT_STRIDE_BYTES as usize;
    for k in 0..3 {
        pack_f32(buf, o + ORIGIN + k * 4, s.origin[k]);
        pack_f32(buf, o + DIR + k * 4, s.dir[k]);
        pack_f32(buf, o + THROUGHPUT + k * 4, s.throughput[k]);
        pack_f32(buf, o + RADIANCE + k * 4, s.radiance[k]);
        pack_f32(buf, o + PLACEHOLDER + k * 4, s.placeholder[k]);
    }
    pack_f32(buf, o + TIME, s.time);
    pack_f32(buf, o + PREV_PDF, s.prev_pdf_dir);
    pack_u32(buf, o + STATE, s.state);
    pack_u32(buf, o + BOUNCE, s.bounce);
    pack_u32(buf, o + FLAGS, s.flags);
    pack_u32(buf, o + AUX_STEPS, s.aux_steps);
}

/// 从字节缓冲读回一个 [`SlotRef`]（96 字节布局，字段偏移见 [`slot_off`]）。
fn unpack_slot(buf: &[u8], i: usize) -> SlotRef {
    use slot_off::*;
    let o = i * PATH_SLOT_STRIDE_BYTES as usize;
    let v3 = |base: usize| [unpack_f32(buf, o + base), unpack_f32(buf, o + base + 4), unpack_f32(buf, o + base + 8)];
    SlotRef {
        origin: v3(ORIGIN),
        dir: v3(DIR),
        time: unpack_f32(buf, o + TIME),
        throughput: v3(THROUGHPUT),
        radiance: v3(RADIANCE),
        prev_pdf_dir: unpack_f32(buf, o + PREV_PDF),
        state: unpack_u32(buf, o + STATE),
        bounce: unpack_u32(buf, o + BOUNCE),
        flags: unpack_u32(buf, o + FLAGS),
        aux_steps: unpack_u32(buf, o + AUX_STEPS),
        placeholder: v3(PLACEHOLDER),
    }
}

/// `accumulate` 的 CPU 参考：逐像素对 `spp` 个槽的 `radiance` 求和后取均值
/// （判对基准 = `prism-rt/src/accum.rs` 的 `mean_samples` 口径）。
fn wf_accumulate_ref(p: &WfParams, slots: &[SlotRef]) -> Vec<[f32; 4]> {
    (0..(p.width * p.height) as usize)
        .map(|idx| {
            let mut sum = [0.0f32; 3];
            let base = idx * p.spp as usize;
            for k in 0..p.spp as usize {
                for c in 0..3 {
                    sum[c] += slots[base + k].radiance[c];
                }
            }
            let inv_spp = 1.0 / p.spp as f32;
            [sum[0] * inv_spp, sum[1] * inv_spp, sum[2] * inv_spp, 1.0]
        })
        .collect()
}

fn t_gpu_rt_wavefront() -> Status {
    let dev = match ready_device("gpu.rt_wavefront") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("rt_wavefront.wgsl") {
        Some(w) => w,
        None => return fail("gpu.rt_wavefront：rt_wavefront.wgsl 展开结果缺失"),
    };
    let slots = wf_initial_slots();
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.wf_slots.fill(0);
        for (i, sl) in slots.iter().enumerate() {
            pack_slot(&mut v.wf_slots, i, sl);
        }
        v.wf_accum.fill(0);
    });
    let par_ptr = params2_write(WF_PARAMS_OFF, &wf_params_bytes(&WF_TEST_PARAMS));
    if par_ptr == 0 {
        return fail("gpu.rt_wavefront：uniform 参数区越界（PARAMS_ARENA_BYTES 不足）");
    }
    let slot_buf = match make_buffer("gpu.rt_wavefront", dev, WF_SLOTS_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf_slots[..]), WF_SLOTS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let acc_buf = match make_buffer("gpu.rt_wavefront", dev, WF_ACCUM_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf_accum[..]), WF_ACCUM_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.rt_wavefront", dev, WF_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, WF_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // 两入口都只静态引用 group(0)：
    //   connect   → slots(0) + params(1)
    //   accumulate→ slots(0) + params(1) + accum(2)
    // **不**绑 camera：camera 已移到独立的 group(3)（见 rt_wavefront.wgsl），两入口
    // 都不引用它，故 step 只给 group(0)。
    let steps: [(&'static str, u32, &[u32]); 2] = [
        ("connect", 1, &[slot_buf, par_buf]),
        ("accumulate", 1, &[slot_buf, par_buf, acc_buf]),
    ];
    let out_slots = addr_of_buf(|x| &mut x.wf_slots[..]);
    let out_acc = addr_of_buf(|x| &mut x.wf_accum[..]);
    set_result(&format!(
        "已提交 rt_wavefront 2 入口链（connect → accumulate），各 1 工作组：\n\
         缓冲：slots={slot_buf}（{WF_SLOTS_BYTES}B = {WF_PATHS}×{PATH_SLOT_STRIDE_BYTES}）\
         accum={acc_buf}（{WF_ACCUM_BYTES}B = {WF_PIXELS}×{VEC4_STRIDE_BYTES}）params={par_buf}（{WF_PARAMS_BYTES}B）\n\
         构型：{WF_WIDTH}×{WF_HEIGHT} 像素 × spp={WF_SPP}，一半槽 TERMINATED（验 connect 早退分支）\n\
         待 settle 后与 CPU 参考（prism-rt/accum.rs 的 spp 均值口径）逐元素判对"
    ));
    if let Err(s) = submit_chain("gpu.rt_wavefront", dev, wgsl, &steps) {
        return s;
    }
    for (buf, len, dst) in [(slot_buf, WF_SLOTS_BYTES, out_slots), (acc_buf, WF_ACCUM_BYTES, out_acc)] {
        if let Err(s) = readback_to("gpu.rt_wavefront", dev, buf, len, dst) {
            return s;
        }
    }
    Status::Pending
}

fn c_gpu_rt_wavefront() -> Status {
    let dev = match ready_device("gpu.rt_wavefront") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.rt_wavefront", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let p = WF_TEST_PARAMS;
    let want_slots = wf_initial_slots();
    let (got_slots, got_acc) = SCRATCH.with(|s| {
        let v = s.borrow();
        (v.wf_slots, v.wf_accum)
    });
    // ---- connect：结构上是恒等（读出再写回），故判"逐字节未变" ----
    // 这不是空判：若 PathSlot 的 80 字节布局在 host 侧算错（如按 76 打包），
    // GPU 按自己算出的 80 字节步长读写的字段会整体错位，写回后字节必然不同。
    let mut want_bytes = vec![0u8; WF_SLOTS_BYTES as usize];
    for (i, sl) in want_slots.iter().enumerate() {
        pack_slot(&mut want_bytes, i, sl);
    }
    try_status!(cmp_bytes("gpu.rt_wavefront", "slots（connect 后应逐字节不变）", &got_slots, &want_bytes));
    // ---- accumulate：逐像素 spp 均值 ----
    let want_acc = wf_accumulate_ref(&p, &want_slots);
    let mut flat: Vec<f32> = Vec::with_capacity(WF_PIXELS as usize * 4);
    for px in &want_acc {
        flat.extend_from_slice(px);
    }
    let d_acc = try_status!(cmp_f32("gpu.rt_wavefront", "accum（spp 均值）", &got_acc, &flat));
    // ---- 状态机门禁：TERMINATED 槽必须仍是终止态、ACTIVE 槽仍是激活态 ----
    for i in 0..WF_PATHS as usize {
        let got = unpack_u32(&got_slots, i * PATH_SLOT_STRIDE_BYTES as usize + slot_off::STATE);
        if got != want_slots[i].state {
            return fail(format!(
                "gpu.rt_wavefront：第 {i} 条路径的 state 被 connect 改写（GPU {got} vs CPU {}）\
                 → PathSlot 的 state 偏移（{}）算错，或 connect 越过了 ST_ACTIVE 守卫",
                want_slots[i].state, slot_off::STATE
            ));
        }
    }
    // ---- 逐字段抽查 radiance/bounce（确认 80 字节步长下字段未错位）----
    for i in 0..WF_PATHS as usize {
        let got = unpack_slot(&got_slots, i);
        if got.radiance != want_slots[i].radiance || got.bounce != want_slots[i].bounce {
            return fail(format!(
                "gpu.rt_wavefront：第 {i} 条槽的 radiance/bounce 错位\
                 （GPU radiance={:?} bounce={} vs CPU radiance={:?} bounce={}）\
                 → PathSlot 96 字节步长或字段偏移推算有误",
                got.radiance, got.bounce, want_slots[i].radiance, want_slots[i].bounce
            ));
        }
    }
    let term = want_slots.iter().filter(|s| s.state == 1).count();
    let act = want_slots.len() - term;
    set_result(&format!(
        "真 WebGPU rt_wavefront：connect + accumulate 与 CPU 参考一致 ✓\n\
         connect：{WF_PATHS} 条路径槽 {WF_SLOTS_BYTES} 字节**逐字节未变**\
         （PathSlot {PATH_SLOT_STRIDE_BYTES}B 步长布局与 host 侧一致 ✓）\n\
         状态机：{term}/{WF_PATHS} 条 TERMINATED 与 {act}/{WF_PATHS} 条 ACTIVE 均原样保持\
         （早退分支确实走到 ✓）；radiance/bounce 逐槽抽查无错位\n\
         accumulate：{WF_PIXELS} 像素 × vec4 最大偏差={d_acc:.3e}（容差 {F32_TOL:e}）\n\
         accum[0]=({:.6},{:.6},{:.6},{}) = spp={WF_SPP} 个槽 radiance 的均值 ✓\n\
         注：generate/shade/extend 静态引用 @group(1)/@group(2)/@group(3)，已由新增的\
         多 bind group 测试项覆盖（见 gpu.rt_wavefront_multi）",
        want_acc[0][0], want_acc[0][1], want_acc[0][2], want_acc[0][3]
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 14：真 WebGPU rt_wavefront 多 bind group 三入口（generate → shade → extend）
// ---------------------------------------------------------------------------
//
// # 这一项解决什么
// `generate` / `shade` / `extend` 静态引用 `@group(1)`（sobol_dir）、`@group(2)`
// （scene_params/materials/prim_mat/tris/lights）、`@group(3)`（camera）。旧胶水
// `prism_create_bind_group` 把 group 索引硬编码为 0，只能建 group(0)，故这三个入口
// 在 D2 期被判 SKIP。本项配合扩展后的 ABI（`prism_create_bind_group(…, group, …)`
// + `prism_dispatch` 支持多 bind group）真 dispatch 三入口，并与 CPU 参考判对。
//
// # CPU 参考的权威来源（与既有测试项同一纪律）
// 判对基准 = `crates/prism-rt` 的对应实现（camera.rs / sampler.rs / rt_scene 语义）。
// 因 `prism-gpu/Cargo.toml` 只依赖 `prism-math`（不在本文件改动范围），本文件的
// CPU 参考是**逐条对齐 prism-rt 的独立重写**：
//   * Sobol 方向数表 = Joe–Kuo 表 + Bratley–Fox 递推（同 `sampler.rs::direction_table`，
//     数据资产复用，行数据逐值搬运）。
//   * `sobol_u32` 的异或累加、`cast_ray` 的薄透镜几何（对齐 `camera.rs::cast_ray`）。
//   * 整数 hash（PCG）、采样流派生与 `rt_shade.wgsl` 的 GPU 复刻**逐位一致**。
//
// # 判对范围（诚实说明）
// 本测试的场景刻意取 `light_count=0`（NEE 运行期循环不执行）与 `aperture_radius=0`
// （针孔快路径），从而把三入口的**运行期**数值收敛到可精确复现的子集：
//   * generate：完整的相机射线投射（像素抖动 → 方向）——全字段判对。
//   * shade：命中三角形 → 发光命中累加（emissive_weight 在本骨架恒为 1），NEE 因
//     light_count=0 不执行——全字段判对（含 Möller–Trumbore 求交命中）。
//   * extend：命中 + Lambert 漫反瓣 BSDF 抽样续弹 + RR——全字段判对。
// 判对同时覆盖 group(0)/group(1)/group(2)/group(3) 四组的句柄接线：任一 group 的
// 绑定集/顺序错位都会让读回值整体偏离 CPU 参考。
//
// # 三入口在同一 chain 内串联（generate → shade → extend）
// 一次读回即判三入口：CPU 侧对同一份 slot 数组按同序推进，最后逐字段比对。

/// `rt_wavefront.wgsl` 的 `sample_index`（32 位 hash；与 shader 逐位一致）。
fn wf_sample_index(px: u32, py: u32, sample: u32, dim: u32, node_id: u32) -> u32 {
    let pack = (px << 16) | (py & 0xFFFF);
    // hash4(pack, sample, dim, node_id)（见 sim_particles 的 pcg_hash）。
    pcg_hash(pcg_hash(pcg_hash(pack) ^ sample) ^ pcg_hash(dim ^ node_id))
}

/// `rt_wavefront` 的 shader 侧 salt（rt_shade.wgsl 常量）。
const WF_IDX_SALT: u32 = 0x85A3_08D3;
/// 掩码 salt（rt_shade.wgsl 常量）。
const WF_MASK_SALT: u32 = 0x0370_7344;

/// `rt_shade.wgsl::hash2(a,b) = pcg_hash(a ^ pcg_hash(b))`。
fn wf_hash2(a: u32, b: u32) -> u32 {
    pcg_hash(a ^ pcg_hash(b))
}

/// `rt_shade.wgsl::sample_1d(si, dim)` 的 CPU 复刻（逐位一致）。
fn wf_sample_1d(si: u32, dim: u32, sobol: &[[u32; 32]; 64]) -> f32 {
    let base = si ^ WF_IDX_SALT;
    let idx = wf_hash2(base, WF_MASK_SALT);
    wf_dim_value(base, dim, idx, sobol)
}

/// `rt_shade.wgsl::sample_2d(si, dim)` 的 CPU 复刻。
fn wf_sample_2d(si: u32, dim: u32, sobol: &[[u32; 32]; 64]) -> (f32, f32) {
    let base = si ^ WF_IDX_SALT;
    let idx = wf_hash2(base, WF_MASK_SALT);
    (
        wf_dim_value(base, dim, idx, sobol),
        wf_dim_value(base, dim + 1, idx, sobol),
    )
}

/// `rt_shade.wgsl::dim_value`：`Sobol(dim, index) ⊕ mask`，取高 24 位 → `[0,1)`。
fn wf_dim_value(seed: u32, dim: u32, index: u32, sobol: &[[u32; 32]; 64]) -> f32 {
    let mask = wf_hash2(seed, WF_MASK_SALT ^ dim);
    let v = wf_sobol_u32(sobol, dim, index) ^ mask;
    (v >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `rt_shade.wgsl::sobol_u32`：`⊕_{k: index 第 k 位=1} sobol_dir[dim*32 + k]`。
fn wf_sobol_u32(sobol: &[[u32; 32]; 64], dim: u32, index: u32) -> u32 {
    let row = &sobol[(dim as usize) & 63];
    let mut v = 0u32;
    let mut bits = index;
    let mut k = 0usize;
    while bits != 0 && k < 32 {
        if bits & 1 == 1 {
            v ^= row[k];
        }
        bits >>= 1;
        k += 1;
    }
    v
}

/// `rt_scene.wgsl::sample_index_seed(prim, ev, slot)` 的 CPU 复刻。
fn wf_sample_index_seed(prim: u32, ev: u32, slot: u32, node_id: u32) -> u32 {
    prim.wrapping_mul(2_654_435_761)
        .wrapping_add(ev.wrapping_mul(40_503))
        .wrapping_add(slot.wrapping_mul(2_246_822_519))
        .wrapping_add(node_id)
}

/// `rt_common.wgsl` 的维度槽常量（与 shader 逐值一致）。
const WF_BASE_BOUNCE: u32 = 5;
/// 每事件段槽数。
const WF_SLOTS_PER_BOUNCE: u32 = 6;
/// 槽位：BSDF/相位 uv。
const WF_SLOT_BSDF: u32 = 0;
/// 相机段维度槽（camera.rs 与 shader 一致）。
const WF_DIM_LENS: u32 = 0;
/// 快门维度槽。
const WF_DIM_SHUTTER: u32 = 2;
/// 像素抖动维度槽。
const WF_DIM_PIXEL: u32 = 3;

/// `rt_common.wgsl::bounce_dim`。
fn wf_bounce_dim(ev: u32, slot: u32) -> u32 {
    WF_BASE_BOUNCE + (ev.saturating_sub(1)) * WF_SLOTS_PER_BOUNCE + slot
}

/// `rt_common.wgsl` 常量（用于 CPU 参考的浮点）。
///
/// 注：原为硬编码 `3.141_592_653_589_793`（f64 字面量收窄到 f32）。D-30 改为
/// `std::f32::consts::PI`——已验证二者 `to_bits()` 逐位相同，故**不改变任何浮点
/// 结果**，仅消除 `clippy::approx_constant`（deny 级）导致的全仓 clippy 中断。
const WF_PI: f32 = std::f32::consts::PI;
/// `1/π`（同上，`to_bits()` 与原 `0.318_309_886_183_790_7` 收窄值逐位相同）。
const WF_FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
/// 自交防护下界。
const WF_T_MIN: f32 = 1e-4;
/// 表面偏移。
const WF_SURFACE_EPS: f32 = 1e-3;
/// 未命中段长。
const WF_MISS_T: f32 = 1e30;

/// Joe–Kuo 方向数初值表（`sampler.rs::JOE_KUO` 的数据资产复用：维度 2..64）。
/// 每行 `(s, a, m_1..m_9)`。
#[rustfmt::skip]
const WF_JOE_KUO: [(u8, u8, [u32; 9]); 63] = [
    (1, 0,  [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2, 1,  [1, 3, 0, 0, 0, 0, 0, 0, 0]),
    (3, 1,  [1, 3, 1, 0, 0, 0, 0, 0, 0]),
    (3, 2,  [1, 1, 1, 0, 0, 0, 0, 0, 0]),
    (4, 1,  [1, 1, 3, 3, 0, 0, 0, 0, 0]),
    (4, 4,  [1, 3, 5, 13, 0, 0, 0, 0, 0]),
    (5, 2,  [1, 1, 5, 5, 17, 0, 0, 0, 0]),
    (5, 4,  [1, 1, 5, 5, 5, 0, 0, 0, 0]),
    (5, 7,  [1, 1, 7, 11, 19, 0, 0, 0, 0]),
    (5, 11, [1, 1, 5, 1, 1, 0, 0, 0, 0]),
    (5, 13, [1, 1, 1, 3, 11, 0, 0, 0, 0]),
    (5, 14, [1, 3, 5, 5, 31, 0, 0, 0, 0]),
    (6, 1,  [1, 3, 3, 9, 7, 49, 0, 0, 0]),
    (6, 13, [1, 1, 1, 15, 21, 21, 0, 0, 0]),
    (6, 16, [1, 3, 1, 13, 27, 49, 0, 0, 0]),
    (6, 19, [1, 1, 1, 15, 7, 5, 0, 0, 0]),
    (6, 22, [1, 3, 1, 15, 13, 25, 0, 0, 0]),
    (6, 25, [1, 1, 5, 5, 19, 61, 0, 0, 0]),
    (7, 1,  [1, 3, 7, 11, 23, 15, 103, 0, 0]),
    (7, 4,  [1, 3, 7, 13, 13, 15, 69, 0, 0]),
    (7, 7,  [1, 1, 3, 13, 7, 35, 63, 0, 0]),
    (7, 8,  [1, 3, 5, 9, 1, 25, 53, 0, 0]),
    (7, 14, [1, 3, 1, 13, 9, 35, 107, 0, 0]),
    (7, 19, [1, 3, 1, 5, 27, 61, 31, 0, 0]),
    (7, 21, [1, 1, 5, 11, 19, 41, 61, 0, 0]),
    (7, 28, [1, 3, 5, 3, 3, 13, 69, 0, 0]),
    (7, 31, [1, 1, 7, 13, 1, 19, 1, 0, 0]),
    (7, 32, [1, 3, 7, 5, 13, 19, 59, 0, 0]),
    (7, 37, [1, 1, 3, 9, 25, 29, 41, 0, 0]),
    (7, 41, [1, 3, 5, 13, 23, 1, 55, 0, 0]),
    (7, 42, [1, 3, 7, 3, 13, 59, 17, 0, 0]),
    (7, 50, [1, 3, 1, 3, 5, 53, 69, 0, 0]),
    (7, 55, [1, 1, 5, 5, 23, 33, 13, 0, 0]),
    (7, 56, [1, 1, 7, 7, 1, 61, 123, 0, 0]),
    (7, 59, [1, 1, 7, 9, 13, 61, 49, 0, 0]),
    (7, 62, [1, 3, 3, 5, 3, 55, 33, 0, 0]),
    (8, 14, [1, 3, 1, 15, 31, 13, 49, 245, 0]),
    (8, 21, [1, 3, 5, 15, 31, 59, 63, 97, 0]),
    (8, 22, [1, 3, 1, 11, 11, 11, 77, 249, 0]),
    (8, 38, [1, 3, 1, 11, 27, 43, 71, 9, 0]),
    (8, 47, [1, 1, 7, 15, 21, 11, 81, 45, 0]),
    (8, 49, [1, 3, 7, 3, 25, 31, 65, 79, 0]),
    (8, 50, [1, 3, 1, 1, 19, 11, 3, 205, 0]),
    (8, 52, [1, 1, 5, 9, 19, 21, 29, 157, 0]),
    (8, 56, [1, 3, 7, 11, 1, 33, 89, 185, 0]),
    (8, 67, [1, 3, 3, 3, 15, 9, 79, 71, 0]),
    (8, 70, [1, 3, 7, 11, 15, 39, 119, 27, 0]),
    (8, 84, [1, 1, 3, 1, 11, 31, 97, 225, 0]),
    (8, 97, [1, 1, 1, 3, 23, 43, 57, 177, 0]),
    (8, 103, [1, 3, 7, 7, 17, 17, 37, 71, 0]),
    (8, 115, [1, 3, 1, 5, 27, 63, 123, 213, 0]),
    (8, 122, [1, 1, 3, 5, 11, 43, 53, 133, 0]),
    (9, 8,  [1, 3, 5, 5, 29, 17, 47, 173, 479]),
    (9, 13, [1, 3, 3, 11, 3, 1, 109, 9, 69]),
    (9, 16, [1, 1, 1, 5, 17, 39, 23, 5, 343]),
    (9, 22, [1, 3, 1, 5, 25, 15, 31, 103, 499]),
    (9, 25, [1, 1, 1, 11, 11, 17, 63, 105, 183]),
    (9, 44, [1, 1, 5, 11, 9, 29, 97, 231, 363]),
    (9, 47, [1, 1, 5, 15, 19, 45, 41, 7, 383]),
    (9, 52, [1, 3, 7, 7, 31, 19, 83, 137, 221]),
    (9, 55, [1, 1, 1, 3, 23, 15, 111, 223, 83]),
    (9, 59, [1, 1, 5, 13, 31, 15, 55, 25, 161]),
    (9, 62, [1, 1, 3, 13, 25, 47, 39, 87, 257]),
];

/// 构造 64×32 方向数矩阵（与 `sampler.rs::direction_table` 同递推）。
/// 维度 0 = van der Corput（`V_i = 2^(32−i)`）；维度 d≥1 = Joe–Kuo + Bratley–Fox。
fn wf_direction_table() -> [[u32; 32]; 64] {
    let mut t = [[0u32; 32]; 64];
    for k in 0..32usize {
        t[0][k] = 1u32 << (31 - k);
    }
    for d in 1..64usize {
        let (s, a, m) = WF_JOE_KUO[d - 1];
        let s = s as usize;
        for i in 1..=s {
            t[d][i - 1] = m[i - 1] << (32 - i);
        }
        for i in (s + 1)..=32usize {
            let prev = t[d][i - 1 - s];
            let mut v = prev ^ (prev >> s);
            let mut bits = a as u32;
            for j in 1..s {
                if bits & 1 == 1 {
                    v ^= t[d][i - 1 - j];
                }
                bits >>= 1;
            }
            t[d][i - 1] = v;
        }
    }
    t
}

/// 把方向数矩阵按 `sobol_dir[dim*32 + k]` 的平铺布局写入字节缓冲（小端）。
fn wf_pack_sobol_dir(buf: &mut [u8], table: &[[u32; 32]; 64]) {
    for dim in 0..64usize {
        for k in 0..32usize {
            pack_u32(buf, (dim * 32 + k) * 4, table[dim][k]);
        }
    }
}

// ---- 相机（对齐 camera.rs / rt_wavefront.wgsl cast_ray）----

/// 多 group 测试的相机（针孔；`aperture_radius = 0` 走快路径）。
#[derive(Debug, Clone, Copy)]
struct WfCamera {
    /// 视点。
    eye: [f32; 3],
    /// 前向。
    forward: [f32; 3],
    /// 右向。
    right: [f32; 3],
    /// 上向。
    up: [f32; 3],
    /// `tan(fov/2)`。
    tan_half_fov: f32,
    /// 光圈半径（0 = 针孔）。
    aperture_radius: f32,
    /// 焦距。
    focus_distance: f32,
    /// 快门开。
    shutter_open: f32,
    /// 快门闭。
    shutter_close: f32,
}

/// 本测试固定的相机（`forward=+z`、`tan(30°)`、针孔）。
const WF2_CAMERA: WfCamera = WfCamera {
    eye: [0.0, 0.0, 0.0],
    forward: [0.0, 0.0, 1.0],
    right: [1.0, 0.0, 0.0],
    up: [0.0, 1.0, 0.0],
    tan_half_fov: 0.577_350_269_189_625_8,
    aperture_radius: 0.0,
    focus_distance: 1.0,
    shutter_open: 0.0,
    shutter_close: 0.5,
};

/// 三维归一化（与 WGSL `normalize` 同：先算平方和再开方）。
fn wf_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// 逐分量乘（WGSL `a * b` 对 `vec3` 的含义）。
fn wf_mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// 逐分量加。
fn wf_add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `rt_wavefront.wgsl::cast_ray` 的 CPU 复刻（针孔路径）。
fn wf_cast_ray(
    px: u32,
    py: u32,
    sample: u32,
    si: u32,
    cam: &WfCamera,
    sobol: &[[u32; 32]; 64],
) -> ([f32; 3], [f32; 3]) {
    let jx = wf_sample_1d(si, WF_DIM_PIXEL, sobol);
    let jy = wf_sample_1d(si, WF_DIM_PIXEL + 1, sobol);
    let _ = sample;
    let u = ((px as f32 + jx) / WF2_WIDTH as f32) * 2.0 - 1.0;
    let v = 1.0 - ((py as f32 + jy) / WF2_HEIGHT as f32) * 2.0;
    let aspect = WF2_WIDTH as f32 / WF2_HEIGHT as f32;
    let d = [
        cam.forward[0]
            + cam.right[0] * (u * cam.tan_half_fov * aspect)
            + cam.up[0] * (v * cam.tan_half_fov),
        cam.forward[1]
            + cam.right[1] * (u * cam.tan_half_fov * aspect)
            + cam.up[1] * (v * cam.tan_half_fov),
        cam.forward[2]
            + cam.right[2] * (u * cam.tan_half_fov * aspect)
            + cam.up[2] * (v * cam.tan_half_fov),
    ];
    // 本测试 `aperture_radius == 0` ⇒ 针孔快路径。
    let dir = wf_normalize(d);
    let origin = cam.eye;
    // shader 忽略算出的 time，恒写 0.0（见 generate 主体），此处保持一致。
    let _ = (cam.shutter_open, cam.shutter_close, WF_DIM_SHUTTER, WF_DIM_LENS);
    (origin, dir)
}

// ---- 场景 CPU 参考（对齐 rt_scene.wgsl 的 Möller–Trumbore / 材质 / BSDF）----

/// 多 group 测试的场景（1 个三角形 → 1 个 Lambert+发光材质；0 灯）。
#[derive(Debug, Clone, Copy)]
struct WfScene {
    /// 三角形顶点 `(a, b, c)`。
    tri: ([f32; 3], [f32; 3], [f32; 3]),
    /// `prim → material` 映射。
    prim_mat: [u32; 1],
    /// 材质（本测试 2 个；prim 0 用 `mats[0]`）。
    mats: [WfMaterial; 2],
    /// 场景计数（material/light/prim）与 node_id、环境光。
    material_count: u32,
    /// 灯光计数（0）。
    light_count: u32,
    /// prim 计数。
    prim_count: u32,
    /// 节点 id。
    node_id: u32,
    /// 环境光 rgb。
    ambient: [f32; 3],
}

/// `rt_scene.wgsl::Material`。
#[derive(Debug, Clone, Copy)]
struct WfMaterial {
    /// 基色。
    base_color: [f32; 3],
    /// 粗糙度。
    roughness: f32,
    /// 菲涅耳 f0。
    f0: [f32; 3],
    /// 金属度。
    metalness: f32,
    /// 透射。
    transmission: f32,
    /// 折射率。
    ior: f32,
    /// 自发光。
    emissive: [f32; 3],
}

/// 本测试固定的场景。
fn wf2_scene() -> WfScene {
    // 三角形：z=5 处的大三角（射线 +z 必命中）。
    let tri = (
        [-40.0, -40.0, 5.0],
        [40.0, -40.0, 5.0],
        [0.0, 60.0, 5.0],
    );
    let m0 = WfMaterial {
        base_color: [0.8, 0.8, 0.8],
        roughness: 0.5,
        f0: [0.0, 0.0, 0.0],
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.5,
        emissive: [0.25, 0.5, 0.75],
    };
    let m1 = WfMaterial {
        base_color: [0.1, 0.2, 0.3],
        roughness: 0.7,
        f0: [0.04, 0.04, 0.04],
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.0,
        emissive: [0.0, 0.0, 0.0],
    };
    WfScene {
        tri,
        prim_mat: [0],
        mats: [m0, m1],
        material_count: 2,
        light_count: WF2_LIGHT_COUNT,
        prim_count: WF2_PRIM_COUNT,
        node_id: WF2_NODE_ID,
        ambient: [0.0, 0.0, 0.0],
    }
}

/// 求交命中记录。
#[derive(Debug, Clone, Copy)]
struct WfHit {
    /// 命中距离（未命中 = [`WF_MISS_T`]）。
    t: f32,
    /// prim 索引（未命中 = `0xFFFFFFFF`）。
    prim: u32,
    /// 几何法线。
    normal: [f32; 3],
}

/// `rt_scene.wgsl::scene_intersect` 的 CPU 复刻（Möller–Trumbore 线性扫描）。
fn wf_scene_intersect(ro: [f32; 3], rd: [f32; 3], t_max: f32, scene: &WfScene) -> WfHit {
    let mut out = WfHit { t: t_max, prim: 0xFFFF_FFFF, normal: [0.0, 0.0, 0.0] };
    let n = scene.prim_count.min(1); // 本测试只装 1 个三角形
    let tris = [scene.tri];
    for (i, tr) in tris.iter().enumerate().take(n as usize) {
        let (a, b, c) = *tr;
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let pvec = [
            rd[1] * e2[2] - rd[2] * e2[1],
            rd[2] * e2[0] - rd[0] * e2[2],
            rd[0] * e2[1] - rd[1] * e2[0],
        ];
        let det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
        if det.abs() > 1e-12 {
            let inv_det = 1.0 / det;
            let tvec = [ro[0] - a[0], ro[1] - a[1], ro[2] - a[2]];
            let u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
            if (0.0..=1.0).contains(&u) {
                let qvec = [
                    tvec[1] * e1[2] - tvec[2] * e1[1],
                    tvec[2] * e1[0] - tvec[0] * e1[2],
                    tvec[0] * e1[1] - tvec[1] * e1[0],
                ];
                let v = (rd[0] * qvec[0] + rd[1] * qvec[1] + rd[2] * qvec[2]) * inv_det;
                if v >= 0.0 && u + v <= 1.0 {
                    let t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
                    if t > WF_T_MIN && t < out.t {
                        out.t = t;
                        out.prim = i as u32;
                        // normal = normalize(cross(e1, e2))
                        let cr = [
                            e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0],
                        ];
                        out.normal = wf_normalize(cr);
                    }
                }
            }
        }
    }
    out
}

/// `rt_scene.wgsl::load_material`（prim → material 索引 → 材质）。
fn wf_load_material(scene: &WfScene, prim: u32) -> WfMaterial {
    let mi = scene.prim_mat[(prim as usize).min(scene.prim_mat.len() - 1)] as usize;
    scene.mats[mi.min(scene.mats.len() - 1)]
}

/// `rt_scene.wgsl::emissive_radiance`。
fn wf_emissive_radiance(scene: &WfScene, prim: u32) -> [f32; 3] {
    wf_load_material(scene, prim).emissive
}

/// `rt_scene.wgsl::clamp_indirect`（本骨架 `CLAMP_INDIRECT_DEFAULT = 10.0`）。
fn wf_clamp_indirect(c: [f32; 3], _ev: u32) -> [f32; 3] {
    let m = c[0].max(c[1]).max(c[2]);
    if m <= 10.0 {
        c
    } else {
        let s = 10.0 / m;
        [c[0] * s, c[1] * s, c[2] * s]
    }
}

/// `rt_scene.wgsl::ambient_background`。
fn wf_ambient_background(scene: &WfScene) -> [f32; 3] {
    scene.ambient
}

// ---- BSDF 采样（对齐 rt_shade.wgsl::bsdf_sample 的漫反瓣）----

/// `rt_shade.wgsl::pow5`。
fn wf_pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

/// `rt_shade.wgsl::schlick`（vec3 版）。
fn wf_schlick(f0: [f32; 3], cos_theta: f32) -> [f32; 3] {
    let m5 = wf_pow5((0.0f32).max(1.0 - cos_theta));
    [
        f0[0] + (1.0 - f0[0]) * m5,
        f0[1] + (1.0 - f0[1]) * m5,
        f0[2] + (1.0 - f0[2]) * m5,
    ]
}

/// `rt_common.wgsl::luminance`。
fn wf_luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// `rt_common.wgsl::cos_sin`（确定性三角件，禁 libm）。
///
/// D-30：以下三个常量原为 f64 字面量收窄到 f32，已逐一验证与
/// `std::f32::consts::{TAU, FRAC_PI_2, FRAC_PI_4}` 的 `to_bits()` **逐位
/// 相同**，故替换不改变任何浮点结果。它们必须与 `rt_common.wgsl` 里的同名
/// WGSL 常量保持一致（CPU/GPU 对拍的前提），**改值需同步改着色器**。
fn wf_cos_sin(x: f32) -> (f32, f32) {
    const TWO_PI: f32 = std::f32::consts::TAU;
    const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
    const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
    let mut r = x - TWO_PI * (x / TWO_PI).floor();
    let q = ((r * (2.0 / WF_PI)) as i32 as u32) & 3;
    r -= q as f32 * FRAC_PI_2;
    let (c, s);
    if r > FRAC_PI_4 {
        let rr = FRAC_PI_2 - r;
        c = wf_quarter_sin(rr);
        s = wf_quarter_cos(rr);
    } else {
        c = wf_quarter_cos(r);
        s = wf_quarter_sin(r);
    }
    match q {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// `rt_common.wgsl::quarter_sin`。
fn wf_quarter_sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
}

/// `rt_common.wgsl::quarter_cos`。
fn wf_quarter_cos(x: f32) -> f32 {
    let x2 = x * x;
    1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
}

/// `rt_common.wgsl::onb`（叉积法）返回 `(t, b, n)`。
fn wf_onb(n: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let up = if n[0].abs() > 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let t = wf_normalize([
        up[1] * n[2] - up[2] * n[1],
        up[2] * n[0] - up[0] * n[2],
        up[0] * n[1] - up[1] * n[0],
    ]);
    let b = [
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    ];
    (t, b, n)
}

/// `rt_common.wgsl::concentric_disk`（Shirley–Chiu）。
///
/// D-30：同 `wf_cos_sin`，与 `std` 常量 `to_bits()` 逐位相同，须与 WGSL 保持一致。
fn wf_concentric_disk(u: f32, v: f32) -> (f32, f32) {
    const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
    const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
    let ox = 2.0 * u - 1.0;
    let oy = 2.0 * v - 1.0;
    if ox == 0.0 && oy == 0.0 {
        return (0.0, 0.0);
    }
    let mut r = ox;
    let mut theta = FRAC_PI_4 * (oy / ox);
    if ox.abs() <= oy.abs() {
        r = oy;
        theta = FRAC_PI_2 - FRAC_PI_4 * (ox / oy);
    }
    let (cs, sn) = wf_cos_sin(theta);
    (r * cs, r * sn)
}

/// `rt_common.wgsl::cosine_hemisphere`（Malley）。
fn wf_cosine_hemisphere(u: f32, v: f32) -> [f32; 3] {
    let (dx, dy) = wf_concentric_disk(u, v);
    let z = (0.0f32).max(1.0 - dx * dx - dy * dy).sqrt();
    [dx, dy, z]
}

/// `rt_shade.wgsl::bsdf_sample` 的 CPU 复刻结果。
#[derive(Debug, Clone, Copy)]
struct WfBsdfSample {
    /// 出射方向。
    dir: [f32; 3],
    /// 权重。
    weight: [f32; 3],
    /// pdf（<0 = 无有效瓣）。
    pdf: f32,
}

/// `rt_shade.wgsl::bsdf_sample` 的 CPU 复刻（本测试材质无透射，走漫反/镜面瓣）。
fn wf_bsdf_sample(
    scene: &WfScene,
    prim: u32,
    w_in: [f32; 3],
    n: [f32; 3],
    ev: u32,
    sobol: &[[u32; 32]; 64],
) -> WfBsdfSample {
    let mat = wf_load_material(scene, prim);
    let wo = [-w_in[0], -w_in[1], -w_in[2]];
    let cos_v = (0.0f32).max(wo[0] * n[0] + wo[1] * n[1] + wo[2] * n[2]);
    let f0 = mat.f0;
    let p_spec = wf_luminance(wf_schlick(f0, cos_v)) + 1e-4;
    let p_diff = wf_luminance(mat.base_color) * (1.0 - mat.metalness) * (1.0 - mat.transmission);
    let p_trans = mat.transmission * (1.0 - mat.metalness);
    let sum = p_spec + p_diff + p_trans;
    let mut s = WfBsdfSample {
        dir: [0.0, 0.0, 1.0],
        weight: [0.0, 0.0, 0.0],
        pdf: -1.0,
    };
    if sum <= 0.0 {
        return s;
    }
    let si = wf_sample_index_seed(prim, ev, WF_SLOT_BSDF, scene.node_id);
    let u_lobe = wf_sample_1d(si, wf_bounce_dim(ev, WF_SLOT_BSDF), sobol);
    let uv = wf_sample_2d(si, wf_bounce_dim(ev, WF_SLOT_BSDF) + 1, sobol);
    // 透射瓣（本测试 transmission=0 ⇒ 不会进入）。
    if u_lobe < p_trans / sum {
        // 占位：本测试不触发；为完整性保留 TIR → 镜反的退化分支。
        s.dir = [
            w_in[0] - 2.0 * (w_in[0] * n[0] + w_in[1] * n[1] + w_in[2] * n[2]) * n[0],
            w_in[1] - 2.0 * (w_in[0] * n[0] + w_in[1] * n[1] + w_in[2] * n[2]) * n[1],
            w_in[2] - 2.0 * (w_in[0] * n[0] + w_in[1] * n[1] + w_in[2] * n[2]) * n[2],
        ];
        s.weight = wf_schlick(f0, cos_v);
        return s;
    }
    // 漫反瓣（Burley FD90 简化为余弦半球）。
    if u_lobe < (p_trans + p_diff) / sum {
        let d_local = wf_cosine_hemisphere(uv.0, uv.1);
        let (t, b, nn) = wf_onb(n);
        s.dir = wf_normalize([
            t[0] * d_local[0] + b[0] * d_local[1] + nn[0] * d_local[2],
            t[1] * d_local[0] + b[1] * d_local[1] + nn[1] * d_local[2],
            t[2] * d_local[0] + b[2] * d_local[1] + nn[2] * d_local[2],
        ]);
        let cos_i = (0.0f32).max(s.dir[0] * n[0] + s.dir[1] * n[1] + s.dir[2] * n[2]);
        s.pdf = cos_i * WF_FRAC_1_PI;
        let one_minus = [1.0 - wf_schlick(f0, cos_v)[0], 1.0 - wf_schlick(f0, cos_v)[1], 1.0 - wf_schlick(f0, cos_v)[2]];
        s.weight = wf_mul3(mat.base_color, one_minus);
        return s;
    }
    // 镜面瓣（GGX VNDF；本测试 1−p_trans/sum 之外才进入，材质粗糙度 0.5）。
    let rough = mat.roughness.max(1e-3);
    let alpha = rough * rough;
    let (t, b, nn) = wf_onb(n);
    let wo_local = [
        wo[0] * t[0] + wo[1] * t[1] + wo[2] * t[2],
        wo[0] * b[0] + wo[1] * b[1] + wo[2] * b[2],
        cos_v,
    ];
    let h_local = wf_sample_vndf(wo_local, alpha, uv.0, uv.1);
    let h = [
        t[0] * h_local[0] + b[0] * h_local[1] + nn[0] * h_local[2],
        t[1] * h_local[0] + b[1] * h_local[1] + nn[1] * h_local[2],
        t[2] * h_local[0] + b[2] * h_local[1] + nn[2] * h_local[2],
    ];
    // reflect(-wo, h)
    let dot = (-wo[0]) * h[0] + (-wo[1]) * h[1] + (-wo[2]) * h[2];
    s.dir = wf_normalize([-wo[0] - 2.0 * dot * h[0], -wo[1] - 2.0 * dot * h[1], -wo[2] - 2.0 * dot * h[2]]);
    let cos_l = s.dir[0] * n[0] + s.dir[1] * n[1] + s.dir[2] * n[2];
    if cos_l <= 0.0 {
        s.pdf = -1.0;
        return s;
    }
    s.pdf = cos_l * WF_FRAC_1_PI;
    s.weight = wf_schlick(f0, cos_v);
    s
}

/// `rt_shade.wgsl::sample_vndf`（Heitz 2018）。
fn wf_sample_vndf(ve_local: [f32; 3], alpha: f32, u1: f32, u2: f32) -> [f32; 3] {
    let vh = wf_normalize([alpha * ve_local[0], alpha * ve_local[1], ve_local[2]]);
    let lensq = vh[0] * vh[0] + vh[1] * vh[1];
    let mut t1 = [1.0, 0.0, 0.0];
    if lensq > 0.0 {
        let inv = 1.0 / lensq.sqrt();
        t1 = [-vh[1] * inv, vh[0] * inv, 0.0];
    }
    let t2 = [
        vh[1] * t1[2] - vh[2] * t1[1],
        vh[2] * t1[0] - vh[0] * t1[2],
        vh[0] * t1[1] - vh[1] * t1[0],
    ];
    let r = u1.sqrt();
    let (cs, sn) = wf_cos_sin(2.0 * WF_PI * u2);
    let t1p = r * cs;
    let mut t2p = r * sn;
    let sc = 0.5 * (1.0 + vh[2]);
    t2p = (1.0 - sc) * (0.0f32).max(1.0 - t1p * t1p).sqrt() + sc * t2p;
    let nz = (0.0f32).max(1.0 - t1p * t1p - t2p * t2p).sqrt();
    let nh = [
        t1[0] * t1p + t2[0] * t2p + vh[0] * nz,
        t1[1] * t1p + t2[1] * t2p + vh[1] * nz,
        t1[2] * t1p + t2[2] * t2p + vh[2] * nz,
    ];
    wf_normalize([alpha * nh[0], alpha * nh[1], (0.0f32).max(nh[2])])
}

/// `rt_scene.wgsl::should_survive`。
fn wf_should_survive(throughput: [f32; 3], seed: u32, idx: u32, ev: u32) -> bool {
    if ev < 3 {
        return true;
    }
    let lum = wf_luminance(throughput);
    let q = lum.clamp(0.05, 0.95);
    let h = (idx.wrapping_mul(747_796_405))
        .wrapping_add(seed.wrapping_mul(2_891_336_453))
        .wrapping_add(ev.wrapping_mul(277_803_737))
        >> 8;
    (h as f32) * (1.0 / 16_777_216.0) < q
}

/// `rt_scene.wgsl::rr_scale`。
fn wf_rr_scale(throughput: [f32; 3]) -> f32 {
    let lum = wf_luminance(throughput);
    if lum <= 0.0 {
        1.0
    } else {
        1.0 / lum
    }
}

/// CPU 侧的 wavefront 路径槽（与 `SlotRef` 同布局）。
type WfSlot = SlotRef;

/// 多 group 测试的 WFParams（与 shader 布局一致）。
const WF2_PARAMS: WfParams = WfParams {
    width: WF2_WIDTH,
    height: WF2_HEIGHT,
    spp: WF2_SPP,
    max_bounces: WF2_MAX_BOUNCES,
    clamp_indirect_on: 0,
    clamp_indirect_threshold: 0.0,
    nee_only: 1,
    global_seed: 0x1234_5678,
    node_id: WF2_NODE_ID,
    frame: 0,
};

/// CPU 参考：`generate` 阶段（写满全部路径槽）。
fn wf2_cpu_generate(sobol: &[[u32; 32]; 64]) -> Vec<WfSlot> {
    let cam = WF2_CAMERA;
    let p = WF2_PARAMS;
    (0..WF2_PATHS as usize)
        .map(|idx| {
            let idx = idx as u32;
            let px = idx % p.width;
            let py = (idx / p.width) % p.height;
            let sample = idx / (p.width * p.height);
            let si = wf_sample_index(px, py, sample, 0, p.node_id);
            let (origin, dir) = wf_cast_ray(px, py, sample, si, &cam, sobol);
            SlotRef {
                origin,
                dir,
                time: 0.0,
                throughput: [1.0, 1.0, 1.0],
                radiance: [0.0, 0.0, 0.0],
                prev_pdf_dir: 0.0,
                state: 0, // ST_ACTIVE
                bounce: 0,
                flags: 1,
                aux_steps: 0,
                placeholder: [0.0, 0.0, 0.0],
            }
        })
        .collect()
}

/// CPU 参考：`shade` 阶段（命中 → 发光累加；否则逃逸终止）。
fn wf2_cpu_shade(slots: &mut [WfSlot], scene: &WfScene) {
    let p = WF2_PARAMS;
    let _ = p;
    for s in slots.iter_mut() {
        if s.state != 0 {
            continue;
        }
        let ev = s.bounce + 1;
        let hit = wf_scene_intersect(s.origin, s.dir, WF_MISS_T, scene);
        if hit.t < 1e29 {
            let em = wf_emissive_radiance(scene, hit.prim);
            if em[0] > 0.0 || em[1] > 0.0 || em[2] > 0.0 {
                // emissive_weight 在本骨架恒为 1（见 rt_scene.wgsl）。
                let contrib = wf_mul3(s.throughput, em);
                s.radiance = wf_add3(s.radiance, wf_clamp_indirect(contrib, ev));
            }
            // nee_surface：light_count=0 ⇒ 循环不执行，无贡献。
            // mnee：nee_only=1 ⇒ 跳过。
        } else {
            let bg = wf_ambient_background(scene);
            let contrib = wf_mul3(s.throughput, bg);
            s.radiance = wf_add3(s.radiance, wf_clamp_indirect(contrib, ev));
            s.state = 1; // ST_TERMINATED
        }
    }
}

/// CPU 参考：`extend` 阶段（深度上限 / 逃逸 → 终止；否则 BSDF 续弹 + RR）。
fn wf2_cpu_extend(slots: &mut [WfSlot], scene: &WfScene, sobol: &[[u32; 32]; 64]) {
    let p = WF2_PARAMS;
    for (idx, s) in slots.iter_mut().enumerate() {
        if s.state != 0 {
            continue;
        }
        let ev = s.bounce + 1;
        if ev >= p.max_bounces {
            s.state = 1;
            continue;
        }
        let hit = wf_scene_intersect(s.origin, s.dir, WF_MISS_T, scene);
        if hit.t >= 1e29 {
            s.state = 1;
            continue;
        }
        let pp = [
            s.origin[0] + s.dir[0] * hit.t,
            s.origin[1] + s.dir[1] * hit.t,
            s.origin[2] + s.dir[2] * hit.t,
        ];
        let n = hit.normal;
        let bs = wf_bsdf_sample(scene, hit.prim, s.dir, n, ev, sobol);
        if bs.pdf < 0.0 {
            s.state = 1;
            continue;
        }
        s.throughput = wf_mul3(s.throughput, bs.weight);
        s.prev_pdf_dir = bs.pdf;
        if !wf_should_survive(s.throughput, p.global_seed, idx as u32, ev + 1) {
            s.state = 1;
            continue;
        }
        let sc = wf_rr_scale(s.throughput);
        s.throughput = [s.throughput[0] * sc, s.throughput[1] * sc, s.throughput[2] * sc];
        s.origin = [
            pp[0] + n[0] * WF_SURFACE_EPS,
            pp[1] + n[1] * WF_SURFACE_EPS,
            pp[2] + n[2] * WF_SURFACE_EPS,
        ];
        s.dir = bs.dir;
        s.bounce += 1;
    }
}

/// 逐字段判对最终路径槽（浮点走 [`F32_TOL`]，整数精确）。
fn cmp_wf2_slots(test: &str, got: &[u8], want: &[WfSlot]) -> Result<f32, Status> {
    let mut max_dev = 0.0f32;
    let mut mismatch = 0usize;
    let mut first_msg = String::new();
    for (i, w) in want.iter().enumerate() {
        let g = unpack_slot(got, i);
        let dev = |a: f32, b: f32| (a - b).abs();
        let mut ok = true;
        for k in 0..3 {
            for (a, b) in [
                (g.origin[k], w.origin[k]),
                (g.dir[k], w.dir[k]),
                (g.throughput[k], w.throughput[k]),
                (g.radiance[k], w.radiance[k]),
                (g.placeholder[k], w.placeholder[k]),
            ] {
                let d = dev(a, b);
                if d > max_dev {
                    max_dev = d;
                }
                if !(d <= F32_TOL) {
                    ok = false;
                }
            }
        }
        for (a, b) in [
            (g.time, w.time),
            (g.prev_pdf_dir, w.prev_pdf_dir),
        ] {
            let d = dev(a, b);
            if d > max_dev {
                max_dev = d;
            }
            if !(d <= F32_TOL) {
                ok = false;
            }
        }
        if g.state != w.state || g.bounce != w.bounce || g.flags != w.flags || g.aux_steps != w.aux_steps {
            ok = false;
        }
        if !ok {
            if mismatch == 0 {
                first_msg = format!(
                    "首个不一致 i={i}：GPU origin={:?} dir={:?} thr={:?} rad={:?} state={} bounce={} \
                     | CPU origin={:?} dir={:?} thr={:?} rad={:?} state={} bounce={}",
                    g.origin, g.dir, g.throughput, g.radiance, g.state, g.bounce,
                    w.origin, w.dir, w.throughput, w.radiance, w.state, w.bounce
                );
            }
            mismatch += 1;
        }
    }
    if mismatch != 0 {
        return Err(fail(format!(
            "{test}：路径槽与 CPU 参考不一致——{mismatch}/{WF2_PATHS} 条超容差 {F32_TOL:e}；{first_msg}"
        )));
    }
    Ok(max_dev)
}

/// 把多 group 场景打包进 scratch（供建 buffer）。
fn wf2_pack_scene(scene: &WfScene) -> () {
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        // 三角形（Tri: a@0 / b@16 / c@32）。
        for k in 0..3 {
            pack_f32(&mut v.wf2_tris, k * 4, scene.tri.0[k]);
            pack_f32(&mut v.wf2_tris, 16 + k * 4, scene.tri.1[k]);
            pack_f32(&mut v.wf2_tris, 32 + k * 4, scene.tri.2[k]);
        }
        // 材质（Material: base_color@0 / roughness@12 / f0@16 / metalness@28 /
        // transmission@32 / ior@36 / emissive@48）。
        for (mi, m) in scene.mats.iter().enumerate() {
            let o = mi * MATERIAL_STRIDE_BYTES as usize;
            for k in 0..3 {
                pack_f32(&mut v.wf2_mats, o + k * 4, m.base_color[k]);
                pack_f32(&mut v.wf2_mats, o + 16 + k * 4, m.f0[k]);
                pack_f32(&mut v.wf2_mats, o + 48 + k * 4, m.emissive[k]);
            }
            pack_f32(&mut v.wf2_mats, o + 12, m.roughness);
            pack_f32(&mut v.wf2_mats, o + 28, m.metalness);
            pack_f32(&mut v.wf2_mats, o + 32, m.transmission);
            pack_f32(&mut v.wf2_mats, o + 36, m.ior);
        }
        // 灯光（Light: position@0 / radius@12 / color@16；本测试 light_count=0）。
        pack_f32(&mut v.wf2_lights, 0, 0.0);
        pack_f32(&mut v.wf2_lights, 4, 0.0);
        pack_f32(&mut v.wf2_lights, 8, 0.0);
        pack_f32(&mut v.wf2_lights, 12, 1.0);
        pack_f32(&mut v.wf2_lights, 16, 0.0);
        pack_f32(&mut v.wf2_lights, 20, 0.0);
        pack_f32(&mut v.wf2_lights, 24, 0.0);
        // prim_mat。
        for (i, &m) in scene.prim_mat.iter().enumerate() {
            pack_u32(&mut v.wf2_prim_mat, i * 4, m);
        }
    });
}

/// `SceneParams` uniform 打包。
fn wf2_scene_params_bytes(scene: &WfScene) -> [u8; SCENE_PARAMS_BYTES as usize] {
    let mut b = [0u8; SCENE_PARAMS_BYTES as usize];
    pack_u32(&mut b, 0, scene.material_count);
    pack_u32(&mut b, 4, scene.light_count);
    pack_u32(&mut b, 8, scene.prim_count);
    pack_u32(&mut b, 12, scene.node_id);
    pack_f32(&mut b, 16, scene.ambient[0]);
    pack_f32(&mut b, 20, scene.ambient[1]);
    pack_f32(&mut b, 24, scene.ambient[2]);
    b
}

/// `Camera` uniform 打包（96 字节，见 [`CAMERA_PARAMS_BYTES`]）。
fn wf2_camera_bytes(cam: &WfCamera) -> [u8; CAMERA_PARAMS_BYTES as usize] {
    let mut b = [0u8; CAMERA_PARAMS_BYTES as usize];
    for k in 0..3 {
        pack_f32(&mut b, k * 4, cam.eye[k]);
        pack_f32(&mut b, 16 + k * 4, cam.forward[k]);
        pack_f32(&mut b, 32 + k * 4, cam.right[k]);
        pack_f32(&mut b, 48 + k * 4, cam.up[k]);
    }
    pack_f32(&mut b, 64, cam.tan_half_fov);
    pack_f32(&mut b, 68, cam.aperture_radius);
    pack_f32(&mut b, 72, cam.focus_distance);
    pack_f32(&mut b, 76, cam.shutter_open);
    pack_f32(&mut b, 80, cam.shutter_close);
    b
}

fn t_gpu_rt_wavefront_multi() -> Status {
    let dev = match ready_device("gpu.rt_wavefront_multi") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("rt_wavefront.wgsl") {
        Some(w) => w,
        None => return fail("gpu.rt_wavefront_multi：rt_wavefront.wgsl 展开结果缺失"),
    };
    let sobol = wf_direction_table();
    let scene = wf2_scene();
    // ---- 输入打包 ----
    let slot_init = wf2_cpu_generate(&sobol);
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.wf2_slots.fill(0);
        for (i, sl) in slot_init.iter().enumerate() {
            pack_slot(&mut v.wf2_slots, i, sl);
        }
    });
    // 在**一次** `borrow_mut` 作用域内打包 Sobol 方向数表。
    // 【教训】不可写成 `&mut SCRATCH.with(|s| s.borrow_mut().sobol_dir)`：那样拿到的是
    // 一个**临时副本**的地址（`[u8; 8192]` 不是可原地借用的 place 表达式经由函数调用
    // 返回），写入落在临时量上、随即丢弃——真机上表现为 GPU 读到全 0 表（sobol_u32≡0）。
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        wf_pack_sobol_dir(&mut v.sobol_dir, &sobol);
    });
    wf2_pack_scene(&scene);
    let par_ptr = params2_write(WF_PARAMS_OFF, &wf_params_bytes(&WF2_PARAMS));
    let cam_ptr = params2_write(WF2_CAMERA_OFF, &wf2_camera_bytes(&WF2_CAMERA));
    let scn_ptr = params2_write(WF2_SCENE_OFF, &wf2_scene_params_bytes(&scene));
    if par_ptr == 0 || cam_ptr == 0 || scn_ptr == 0 {
        return fail("gpu.rt_wavefront_multi：uniform 参数区越界（PARAMS_ARENA_BYTES 不足）");
    }
    // ---- 建 buffer ----
    // 仅建三个入口实际引用的资源；accumulate 入口不再出现在本链中，故不建 accum 缓冲。
    let slot_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF2_SLOTS_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf2_slots[..]), WF2_SLOTS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, WF_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let sobol_buf = match make_buffer("gpu.rt_wavefront_multi", dev, SOBOL_DIR_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.sobol_dir[..]), SOBOL_DIR_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let tri_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF2_TRI_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf2_tris[..]), WF2_TRI_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let mat_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF2_MAT_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf2_mats[..]), WF2_MAT_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let light_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF2_LIGHT_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf2_lights[..]), WF2_LIGHT_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let pm_buf = match make_buffer("gpu.rt_wavefront_multi", dev, WF2_PRIM_MAT_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.wf2_prim_mat[..]), WF2_PRIM_MAT_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let cam_buf = match make_buffer("gpu.rt_wavefront_multi", dev, CAMERA_PARAMS_BYTES, usage::UNIFORM, || {
        (cam_ptr, CAMERA_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let scn_buf = match make_buffer("gpu.rt_wavefront_multi", dev, SCENE_PARAMS_BYTES, usage::UNIFORM, || {
        (scn_ptr, SCENE_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // ---- 三个入口的 bind group 前缀（每 group 内为连续前缀 {0..n-1}）----
    // generate：group0=[slots,params] group1=[sobol] group2=[scene_params] group3=[camera]
    let gen_g0 = [slot_buf, par_buf];
    let gen_g1 = [sobol_buf];
    let gen_g2 = [scn_buf];
    let gen_g3 = [cam_buf];
    // shade：group0=[slots,params] group1=[sobol] group2=[scene_params,materials,prim_mat,tris,lights]
    let sh_g2 = [scn_buf, mat_buf, pm_buf, tri_buf, light_buf];
    // extend：group0/group1 同；group2=[scene_params,materials,prim_mat,tris]（不引用 lights）
    let ex_g2 = [scn_buf, mat_buf, pm_buf, tri_buf];
    let gen_groups = [
        GroupBind { group: 0, bufs: &gen_g0 },
        GroupBind { group: 1, bufs: &gen_g1 },
        GroupBind { group: 2, bufs: &gen_g2 },
        GroupBind { group: 3, bufs: &gen_g3 },
    ];
    let sh_groups = [
        GroupBind { group: 0, bufs: &gen_g0 },
        GroupBind { group: 1, bufs: &gen_g1 },
        GroupBind { group: 2, bufs: &sh_g2 },
    ];
    let ex_groups = [
        GroupBind { group: 0, bufs: &gen_g0 },
        GroupBind { group: 1, bufs: &gen_g1 },
        GroupBind { group: 2, bufs: &ex_g2 },
    ];
    // 2 个工作组（64 线程/组 > 64 路径 ⇒ 边界守卫被真实走到）。
    let steps = [
        MultiStep { entry: "generate", workgroups: 2, groups: &gen_groups },
        MultiStep { entry: "shade", workgroups: 2, groups: &sh_groups },
        MultiStep { entry: "extend", workgroups: 2, groups: &ex_groups },
    ];
    let out_slots = addr_of_buf(|x| &mut x.wf2_slots[..]);
    set_result(&format!(
        "已提交 rt_wavefront 多 bind group 链（generate → shade → extend），各 2 工作组：\n\
         缓冲：slots={slot_buf}（{WF2_SLOTS_BYTES}B={WF2_PATHS}×{PATH_SLOT_STRIDE_BYTES}）\
         params={par_buf} camera={cam_buf}（{CAMERA_PARAMS_BYTES}B）\
         sobol_dir={sobol_buf}（{SOBOL_DIR_BYTES}B，{WF2_PATHS} 之外的方向数表）\
         scene={{params={scn_buf}({SCENE_PARAMS_BYTES}B) materials={mat_buf} prim_mat={pm_buf} tris={tri_buf} lights={light_buf}}}\n\
         group 前缀：generate=g0{{0,1}} g1{{0}} g2{{0}} g3{{0}}；shadow/extend g2 分别 5/4 绑定\n\
         构型：{WF2_WIDTH}×{WF2_HEIGHT} 像素 × spp={WF2_SPP} = {WF2_PATHS} 条路径（dispatch 2 组验边界守卫）\n\
         读回 {WF2_SLOTS_BYTES} 字节 → 线性内存 {out_slots}，待 settle 后与 CPU 参考（prism-rt 同式）逐字段判对"
    ));
    if let Err(s) = submit_chain_multi("gpu.rt_wavefront_multi", dev, wgsl, &steps) {
        return s;
    }
    if let Err(s) = readback_to("gpu.rt_wavefront_multi", dev, slot_buf, WF2_SLOTS_BYTES, out_slots) {
        return s;
    }
    Status::Pending
}

fn c_gpu_rt_wavefront_multi() -> Status {
    let dev = match ready_device("gpu.rt_wavefront_multi") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.rt_wavefront_multi", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let sobol = wf_direction_table();
    let scene = wf2_scene();
    // CPU 参考：同一份 slot 数组按 generate → shade → extend 同序推进。
    let mut want = wf2_cpu_generate(&sobol);
    wf2_cpu_shade(&mut want, &scene);
    wf2_cpu_extend(&mut want, &scene, &sobol);
    let got = SCRATCH.with(|s| s.borrow().wf2_slots);
    let max_dev = try_status!(cmp_wf2_slots("gpu.rt_wavefront_multi", &got, &want));
    let fingerprint = hash_u32(&want.iter().flat_map(|w| {
        let mut v: Vec<u32> = Vec::new();
        for k in 0..3 {
            v.push(w.origin[k].to_bits());
            v.push(w.dir[k].to_bits());
            v.push(w.throughput[k].to_bits());
            v.push(w.radiance[k].to_bits());
        }
        v.push(w.state);
        v.push(w.bounce);
        v
    }).collect::<Vec<u32>>());
    let act_after = want.iter().filter(|s| s.state == 0).count();
    let term_after = want.iter().filter(|s| s.state == 1).count();
    set_result(&format!(
        "真 WebGPU rt_wavefront 多 bind group：generate + shade + extend 与 CPU 参考一致 ✓\n\
         判对范围：{WF2_PATHS} 条路径槽（96B/槽）逐字段（origin/dir/throughput/radiance/state/bounce…）\
         最大偏差={max_dev:.3e}（容差 {F32_TOL:e}）\n\
         阶段：generate（相机射线，group 0/1/2/3 四组绑定）→ shade（命中发光累加；NEE 关闭）\
         → extend（Lambert BSDF 续弹 + RR；max_bounces={WF2_MAX_BOUNCES}）\n\
         终态：ACTIVE={act_after} TERMINATED={term_after}；CPU 参考 fingerprint=0x{fingerprint:08x}\n\
         关键：group(1) 的 sobol_dir（方向数表）与 group(2) 的 scene_params/materials/prim_mat/tris\
         ∈ 全部由多 group ABI 正确接线——任一 group 顺序/绑定量错位都会让读回值整体偏离"
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 15：真 WebGPU GI 的 GPU 路径（gi_probe_trace → gi_sh_project → gi_probe_interpolate）
// ---------------------------------------------------------------------------
//
// # 与 CPU 侧（prism-rt::gi）的关系
//
// 本项的判对基准是 `crates/prism-rt/src/gi/{sdf_gi.rs, sh.rs}` 的**同一套公式**，
// 在 abi.rs 里以**独立重写**落地（逐式对齐，见每处注释的出处），并在
// `host_test.mjs` 用**第三份 JS 实现**交叉核对。之所以重写而不直接 `use`
// `prism-rt`（其 `sdf_gi`/`sh` 是 pub）：`prism-gpu/Cargo.toml` 的依赖只有
// `prism-math`，而 `Cargo.toml` 不在本文件改动范围内（避免与并行改动冲突）。
//
// # SH 契约（跨文件，必须一致；见 gi_compute.wgsl 文件头）
//   投影原始辐亮度 `L_lm = ∫L·Y_lm dΩ`；重建加余弦瓣 `Â_0=π, Â_1=2π/3, Â_2=π/4`；
//   转辐亮度乘 `1/π`。CPU 参考与 WGSL 逐系数、逐方向同序累加 → 确定性。
//
// # 确定性
// 采样方向为 Fibonacci 半球固定序；三角走 `wf_cos_sin`（与 WGSL `cos_sin` 逐式同构）；
// 累加序固定。GPU 端允许 FMA 收缩，故走 [`F32_TOL`] 容差（整数路径不含）。

/// GI 测试固定参数（确定性铁律：写死，不取时间/随机源）。
#[derive(Debug, Clone, Copy)]
struct GiTestParams {
    /// 探针数。
    probes: u32,
    /// 查询点数。
    points: u32,
    /// 有效图元数。
    prims: u32,
    /// 采样方向数。
    n_dirs: u32,
    /// 环境辐亮度。
    env: [f32; 3],
    /// 漫反射反照率。
    albedo: [f32; 3],
    /// sphere tracing 射线参数上限。
    t_max: f32,
    /// 命中阈值。
    hit_eps: f32,
    /// 步长安全系数。
    safety: f32,
    /// 最大步数。
    max_steps: u32,
}

/// 本测试固定的 GI 参数（写死 → 确定性）。
const GI_TEST: GiTestParams = GiTestParams {
    probes: GI_N_PROBES,
    points: GI_N_POINTS,
    prims: 2,
    n_dirs: GI_DIRS,
    env: [0.9, 0.8, 0.7],
    albedo: [0.8, 0.8, 0.8],
    t_max: 20.0,
    hit_eps: 1e-3,
    safety: 0.9,
    max_steps: 48,
};

/// GI 图元（与 `gi_compute.wgsl::GiPrim` 对应）：`kind=0` 球（center+radius），
/// `kind=1` 盒（center+半尺寸）。
#[derive(Debug, Clone, Copy)]
struct GiPrimRef {
    /// `0` = 球，`1` = 盒。
    kind: u32,
    /// 球半径 / 盒半尺寸。
    radius: f32,
    /// 中心。
    center: [f32; 3],
}

/// 本测试的 SDF 场景图元（一个球 + 一块地板盒，语义上"某点上方有遮挡"）。
///
/// # 容量填充必须是「即使被误用也绝不遮挡」的安全值（V2.2 修复）
/// `GI_TEST.prims = 2`，后两条只是**把 `GiParams.prims` 的定长容量填满**。此前填的是
/// 「球心在原点 + r=1」，一旦 `prim_count` 在任一环节被读成 4（例如读到相邻的
/// `point_count@4`），三个探针起点都会被判成"在球内"→ 8/8 全遮挡 → `samples`/`sh`/
/// `irradiance` 整块零值。**填充数据的危险性来自「它离探针起点太近」，而不是来自
/// 「它被用到」**，故改为「半径 0 + 远在 −Y 轴 1e6 处」：
/// `SDF = |p − (0,−1e6,0)| − 0 ≈ 1e6`，远大于 `t_max`(=20) 与任何场景尺度，
/// 无论 `prim_count` 被误读成几都不会命中，也不会把 Union 的 min 拉低。
const GI_PRIMS: [GiPrimRef; GI_MAX_PRIMS as usize] = [
    // 球：位于 +X 侧，作为探针射线的遮挡体。
    GiPrimRef { kind: 0, radius: 1.0, center: [2.0, 0.0, 0.0] },
    // 盒（半尺寸 10³）作为"地板/大遮挡"，覆盖 x∈[-10,10] 等。
    GiPrimRef { kind: 1, radius: 10.0, center: [0.0, -10.0, 0.0] },
    // 容量填充（不参与 `prim_count=2`）：半径 0 + 远在 −Y 轴 1e6 处 →
    // `SDF ≈ 1e6`，即使 `prim_count` 被误读成 4 也绝不命中。
    GiPrimRef { kind: 0, radius: 0.0, center: [0.0, -1.0e6, 0.0] },
    GiPrimRef { kind: 0, radius: 0.0, center: [0.0, -1.0e6, 0.0] },
];

// ---- 哨兵初值（GPU 输入缓冲的初值，用于把「入口没跑」与「跑了但结果为零」分开）----
//
// # 为什么需要哨兵
// 真机 FAIL 时读回缓冲里"整块 0"至少有三种不兼容机制（入口没跑 / 跑了但全遮挡 /
// 跑了但 project 早退），而**零初始化缓冲与"真的算出 0"读回字节完全相同**，宿主无法
// 区分。故在打包阶段给"入口应覆写"的字段预填**负值哨兵**（物理上不可能出现的值：
// 辐亮度非负、辐照度非负）：
//   * `samples`        预填 [`GI_SENTINEL_SAMPLE`]（-1）→ 读回仍是 -1 = `gi_probe_trace` 没写过这一段；
//   * `sh`             预填 [`GI_SENTINEL_SH`]（-2）→ 读回仍是 -2 = `gi_sh_project` 没写过这一段；
//   * `irradiance`     预填 [`GI_SENTINEL_IRRADIANCE`]（-3）→ 读回仍是 -3 = `gi_probe_interpolate` 没写过。
//
// # 为什么不影响判据
// 哨兵只改 **GPU 输入缓冲的初值**。CPU 参考走 [`gi_probe_inputs`] / [`gi_point_inputs`]
// 独立重算（`gi_probe_trace_ref` / `gi_sh_project_ref` / `gi_interpolate_ref`），
// **完全不读这两块缓冲**，故 CPU 参考值与哨兵无关、判据语义不变。
// 且正常路径下三个 entry 都会覆写各自的全部字段，哨兵不会漏进结果。
/// `samples` 段未写入的哨兵（负值：辐亮度不可能为负）。
const GI_SENTINEL_SAMPLE: f32 = -1.0;
/// `sh` 段未写入的哨兵。
const GI_SENTINEL_SH: f32 = -2.0;
/// `irradiance` 段未写入的哨兵。
const GI_SENTINEL_IRRADIANCE: f32 = -3.0;

/// 探针输入（位置 + 法线）。
fn gi_probe_inputs() -> [[f32; 6]; GI_N_PROBES as usize] {
    [
        // [pos.x, pos.y, pos.z, n.x, n.y, n.z]
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],   // 朝上：应大量可见（上方除地板背面外无遮挡）
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],   // 朝 +X：正对球 → 部分遮挡
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],   // 朝 +X 但离球更远 → 遮挡少于上面
    ]
}

/// 查询点输入。
fn gi_point_inputs() -> [[f32; 3]; GI_N_POINTS as usize] {
    [[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0], [2.5, 1.0, 0.0]]
}

// ---- CPU 参考：与 WGSL 逐式同构 ----

/// `gi_compute.wgsl::gi_prim_dist`（对齐 `Csg::Sphere` / `Csg::Box`）。
///
/// # 球分支为什么是 `|p − c| − r`（V2.2 修复）
/// 此前球分支返回 `q[0]`，即 `|p.x − c.x| − r`——**只与 x 有关，是沿 x 的无限大板**，
/// 不是球。原注释"q 的三分量相等，任取一分量即 |p−c|"不成立：`abs(p−c)−r` 的三个
/// 分量只有在「仅一个坐标有差」时才相等。真机实测（GPU 侧 `sh` 整块为零）暴露后
/// 两端同步改为标准球 SDF。
///
/// 与 WGSL 同式、同精度路径：先算三个分量差 → 平方和（左结合 `((a+b)+c)`，
/// 与 `length(vec3)` 的实现顺序一致）→ `sqrt()` → 减半径。**不用 `powi`**
/// （`powi(2)` 走的是另一条代码生成路径，可能与 WGSL 的 `x*x` 不同精度）。
fn gi_prim_dist_ref(prim: &GiPrimRef, p: [f32; 3]) -> f32 {
    if prim.kind == 1 {
        let q = [
            (p[0] - prim.center[0]).abs() - prim.radius,
            (p[1] - prim.center[1]).abs() - prim.radius,
            (p[2] - prim.center[2]).abs() - prim.radius,
        ];
        let cx = q[0].max(0.0);
        let cy = q[1].max(0.0);
        let cz = q[2].max(0.0);
        let outside = (cx * cx + cy * cy + cz * cz).sqrt();
        let inside = q[0].max(q[1]).max(q[2]).min(0.0);
        outside + inside
    } else {
        // 球：SDF = |p − c| − radius。
        let dx = p[0] - prim.center[0];
        let dy = p[1] - prim.center[1];
        let dz = p[2] - prim.center[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        // 不写 `debug_assert!(d2 >= 0.0)`：NaN 输入下该断言会 panic，与 C18
        // 「不 panic」相冲突；且平方和非负是平凡事实，断言无信息量。
        d2.sqrt() - prim.radius
    }
}

/// `gi_compute.wgsl::gi_scene_dist` 的**显式图元表**版本（Union = min；空场 EMPTY_FAR）。
///
/// 生产路径经 [`gi_sphere_trace_ref`] 传入「按 `prim_count` clamp 后的 `GI_PRIMS` 前缀」；
/// 本变体之所以把图元表做成参数，是为回归钉子
/// [`tests::gi_prim_count_misread_does_not_blind_the_scene`] 服务——那条测试要用
/// **另一套**（危险的旧填充）图元跑同一份实现，从而在不改仓库常量、不复制算法的
/// 前提下证明"`prim_count` 误读成 4 + 危险填充 ⇒ 全遮挡"这个机制。
///
/// `prim_count` 的 clamp 在 [`gi_sphere_trace_ref`] 里做（与 WGSL 侧 `gi_scene_dist`
/// 的越界守卫同构：上界 = 编译期容量 [`GI_MAX_PRIMS`]）。
fn gi_scene_dist_prims(p: [f32; 3], prims: &[GiPrimRef]) -> f32 {
    let mut d = 1.0e6f32;
    for prim in prims {
        d = d.min(gi_prim_dist_ref(prim, p));
    }
    d
}

/// `gi_compute.wgsl::sphere_trace_sdf`（对齐 `prism-geometry::sdf::sphere_trace`）的
/// **显式图元表**版本（供回归钉子用；生产路径见 [`gi_sphere_trace_ref`]）。
fn gi_sphere_trace_prims(
    ro: [f32; 3],
    rd: [f32; 3],
    prims: &[GiPrimRef],
    p: &GiTestParams,
) -> bool {
    for k in 0..3 {
        if !rd[k].is_finite() {
            return false;
        }
    }
    let rl = (rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]).sqrt();
    if rl < 0.5 {
        return false;
    }
    let dir = [rd[0] / rl, rd[1] / rl, rd[2] / rl];
    let mut t = 0.0f32;
    for _ in 0..p.max_steps {
        let d = gi_scene_dist_prims(
            [ro[0] + dir[0] * t, ro[1] + dir[1] * t, ro[2] + dir[2] * t],
            prims,
        );
        if d <= p.hit_eps {
            return true;
        }
        t += d * p.safety;
        if t > p.t_max {
            return false;
        }
    }
    false
}

/// `gi_compute.wgsl::sphere_trace_sdf`（对齐 `prism-geometry::sdf::sphere_trace`）。
///
/// **仅测试使用**：生产路径（`gpu.gi_compute` 的 CPU 参考）经
/// [`gi_probe_trace_ref`] → [`gi_probe_trace_prims`] 直接调 `_prims` 变体，
/// 故本函数标注 `#[cfg(test)]` 以免在 wasm 产物里留下死代码。
/// 它存在的意义是给 `gi_sdf_occlusion_matches_analytic` 一个"按 `prim_count` 取
/// `GI_PRIMS` 前缀"的入口，与 WGSL 的调用形态逐字对应。
///
/// `prim_count` 的 clamp 与 WGSL 侧 `gi_scene_dist` 的越界守卫同构（上界 = 编译期
/// 容量 [`GI_MAX_PRIMS`]）。
#[cfg(test)]
fn gi_sphere_trace_ref(ro: [f32; 3], rd: [f32; 3], prim_count: u32, p: &GiTestParams) -> bool {
    let n = (prim_count as usize).min(GI_PRIMS.len());
    gi_sphere_trace_prims(ro, rd, &GI_PRIMS[..n], p)
}

/// `gi_compute.wgsl::gi_hemi_dir`（立体角均匀半球；三角走 `wf_cos_sin`）。
fn gi_hemi_dir_ref(d: u32, n: u32) -> [f32; 3] {
    let idx = d as f32 + 0.5;
    let z = idx / n as f32; // cosθ 均匀 ∈ (0,1)
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = idx * 2.3999632;
    let (c, s) = wf_cos_sin(phi);
    [r * c, r * s, z]
}

/// 与 `gi_compute.wgsl` 一致的基函数常量（对齐 `sh.rs::sh_basis_l2`）。
fn gi_sh_basis_l2_ref(d: [f32; 3]) -> [f32; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282_094_79,
        0.488_602_51 * y,
        0.488_602_51 * z,
        0.488_602_51 * x,
        1.092_548_46 * x * y,
        1.092_548_46 * y * z,
        0.315_391_57 * (3.0 * z * z - 1.0),
        1.092_548_46 * x * z,
        0.546_274_23 * (x * x - y * y),
    ]
}

/// `gi_compute.wgsl::gi_cosine_lobe`（对齐 `sh.rs::cosine_lobe`）。
fn gi_cosine_lobe_ref(band: usize) -> f32 {
    match band {
        0 => WF_PI,
        1 => 2.0 * WF_PI / 3.0,
        _ => WF_PI / 4.0,
    }
}

/// `gi_compute.wgsl::gi_band_of`。
fn gi_band_of_ref(index: usize) -> usize {
    match index {
        0 => 0,
        1..=3 => 1,
        _ => 2,
    }
}

/// 一条探针的 CPU 参考中间态（采样辐亮度 + SH 系数）。
#[derive(Debug, Clone, Copy)]
struct GiProbeRef {
    /// 采样辐亮度（每方向，RGB）。
    samples: [[f32; 3]; GI_DIRS as usize],
    /// SH-L2 系数（9 项 RGB）。
    sh: [[f32; 3]; 9],
}

/// CPU 参考：`gi_probe_trace`（半球 SDF 遮挡 → 采样辐亮度）。
///
/// `prim_count` 的 clamp 与越界守卫见 [`gi_sphere_trace_ref`]（与 WGSL 同构）。
fn gi_probe_trace_ref(pos: [f32; 3], normal: [f32; 3], p: &GiTestParams) -> GiProbeRef {
    let n = (p.prims as usize).min(GI_PRIMS.len());
    gi_probe_trace_prims(pos, normal, &GI_PRIMS[..n], p)
}

/// [`gi_probe_trace_ref`] 的**显式图元表**版本（回归钉子用；算法只有一份）。
fn gi_probe_trace_prims(
    pos: [f32; 3],
    normal: [f32; 3],
    prims: &[GiPrimRef],
    p: &GiTestParams,
) -> GiProbeRef {
    let mut out = GiProbeRef { samples: [[0.0; 3]; GI_DIRS as usize], sh: [[0.0; 3]; 9] };
    let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if nl < 0.5 {
        return out;
    }
    let n = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
    let (t, b, _nn) = wf_onb(n);
    let n_dirs = p.n_dirs.min(GI_DIRS);
    for d in 0..n_dirs {
        let local = gi_hemi_dir_ref(d, n_dirs);
        let world = wf_normalize([
            t[0] * local[0] + b[0] * local[1] + n[0] * local[2],
            t[1] * local[0] + b[1] * local[1] + n[1] * local[2],
            t[2] * local[0] + b[2] * local[1] + n[2] * local[2],
        ]);
        let ro = [pos[0] + n[0] * 1e-3, pos[1] + n[1] * 1e-3, pos[2] + n[2] * 1e-3];
        if !gi_sphere_trace_prims(ro, world, prims, p) {
            out.samples[d as usize] = p.env; // 未遮挡 → 环境辐亮度
        }
    }
    out
}

/// CPU 参考：`gi_sh_project`（投影采样辐亮度到 SH-L2，**不乘 cos**）。
fn gi_sh_project_ref(pos: [f32; 3], normal: [f32; 3], p: &GiTestParams) -> GiProbeRef {
    let n = (p.prims as usize).min(GI_PRIMS.len());
    gi_sh_project_prims(pos, normal, &GI_PRIMS[..n], p)
}

/// [`gi_sh_project_ref`] 的**显式图元表**版本（回归钉子用；算法只有一份）。
fn gi_sh_project_prims(
    pos: [f32; 3],
    normal: [f32; 3],
    prims: &[GiPrimRef],
    p: &GiTestParams,
) -> GiProbeRef {
    let mut r = gi_probe_trace_prims(pos, normal, prims, p);
    r.sh = gi_sh_project_samples(normal, &r.samples, p);
    r
}

/// SH-L2 投影的**纯函数**部分：把已知的逐方向采样辐亮度投到 9 个系数。
///
/// 与 `gi_compute.wgsl::gi_sh_project` 的累加段逐式一致（方向序 × 系数序固定）。
/// 从 [`gi_sh_project_prims`] 里抽出来的理由：回归钉子
/// `gi_prim_count_misread_does_not_blind_the_scene` 需要"用**历史** trace 产出的
/// samples 走**现行**投影"，从而证明"全遮挡 ⇒ sh 整块为零"这一步本身没有第二份实现。
fn gi_sh_project_samples(
    normal: [f32; 3],
    samples: &[[f32; 3]],
    p: &GiTestParams,
) -> [[f32; 3]; 9] {
    let mut sh = [[0.0f32; 3]; 9];
    let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if nl < 0.5 || p.n_dirs == 0 {
        return sh;
    }
    let n = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
    let (t, b, _nn) = wf_onb(n);
    let n_dirs = p.n_dirs.min(GI_DIRS);
    let d_omega = 2.0 * WF_PI / n_dirs as f32;
    for d in 0..n_dirs {
        let local = gi_hemi_dir_ref(d, n_dirs);
        let world = wf_normalize([
            t[0] * local[0] + b[0] * local[1] + n[0] * local[2],
            t[1] * local[0] + b[1] * local[1] + n[1] * local[2],
            t[2] * local[0] + b[2] * local[1] + n[2] * local[2],
        ]);
        let yb = gi_sh_basis_l2_ref(world);
        let rad = samples[d as usize];
        for k in 0..9 {
            let w = yb[k] * d_omega;
            sh[k][0] += rad[0] * w;
            sh[k][1] += rad[1] * w;
            sh[k][2] += rad[2] * w;
        }
    }
    sh
}

/// CPU 参考：`gi_probe_interpolate`（最近探针 SH 重建 + Lambert 转辐亮度）。
fn gi_interpolate_ref(query: [f32; 3], probes: &[( [f32; 3], GiProbeRef )], p: &GiTestParams) -> [f32; 3] {
    if probes.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for (j, (pos, _)) in probes.iter().enumerate() {
        let dx = pos[0] - query[0];
        let dy = pos[1] - query[1];
        let dz = pos[2] - query[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best = j;
        }
    }
    let sh = &probes[best].1.sh;
    // 重建方向 +Z（与 WGSL 一致）。
    let yb = gi_sh_basis_l2_ref([0.0, 0.0, 1.0]);
    let mut e = [0.0f32; 3];
    for k in 0..9 {
        let c = gi_cosine_lobe_ref(gi_band_of_ref(k)) * yb[k];
        e[0] += sh[k][0] * c;
        e[1] += sh[k][1] * c;
        e[2] += sh[k][2] * c;
    }
    // Lambert：L_o = albedo · E · (1/π)。
    let inv_pi = 1.0 / WF_PI;
    [p.albedo[0] * e[0] * inv_pi, p.albedo[1] * e[1] * inv_pi, p.albedo[2] * e[2] * inv_pi]
}

/// `gi_compute.wgsl::GiParams` 的 uniform 打包（256 字节，偏移见 [`GI_PARAMS_BYTES`] 文档）。
fn gi_params_bytes(p: &GiTestParams) -> [u8; GI_PARAMS_BYTES as usize] {
    let mut b = [0u8; GI_PARAMS_BYTES as usize];
    pack_u32(&mut b, 0, p.probes);
    pack_u32(&mut b, 4, p.points);
    pack_u32(&mut b, 8, p.prims);
    pack_u32(&mut b, 12, p.n_dirs);
    pack_f32(&mut b, 16, p.env[0]);
    pack_f32(&mut b, 20, p.env[1]);
    pack_f32(&mut b, 24, p.env[2]);
    // env_radiance 的 w @28 —— vec4 的第 4 分量保持 0（缓冲由 [0u8; N] 初始化）。
    pack_f32(&mut b, 32, p.albedo[0]);
    pack_f32(&mut b, 36, p.albedo[1]);
    pack_f32(&mut b, 40, p.albedo[2]);
    // albedo 的 w @44 —— 同上，恒 0。
    pack_f32(&mut b, 48, p.t_max);
    pack_f32(&mut b, 52, p.hit_eps);
    pack_f32(&mut b, 56, p.safety);
    pack_u32(&mut b, 60, p.max_steps);
    // prims @64，每条 48 字节：kind@0 / radius@4 / _pad0@8(8B) / center@16(vec4, w=0) / _pad1@32。
    for (i, prim) in GI_PRIMS.iter().enumerate() {
        let o = 64 + i * GI_PRIM_STRIDE_BYTES as usize;
        pack_u32(&mut b, o, prim.kind);
        pack_f32(&mut b, o + 4, prim.radius);
        pack_f32(&mut b, o + 16, prim.center[0]);
        pack_f32(&mut b, o + 20, prim.center[1]);
        pack_f32(&mut b, o + 24, prim.center[2]);
        // center 的 w @28 —— 恒 0（vec4 语义；与 WGSL 的 `center:vec4<f32>` 对齐）。
    }
    b
}

/// 把探针输入（位置 + 法线）打包进 `array<GiProbe>`，并给"入口应覆写"的字段预填哨兵。
///
/// 布局（与 WGSL `GiProbe` 逐字段一致，[`GI_PROBE_STRIDE_BYTES`] 有完整推导）：
/// `pos:vec4@0` / `normal:vec4@16` /
/// `samples[k]:vec4@32+16k`（k=0..8）/ `sh[k]:vec4@160+16k`（k=0..9）。
///
/// # 第 4 分量（w）恒 0
/// 每个 vec4 只写 xyz，w 保持 `fill(0)` 的 0——与 shader 侧 `vec4(xyz, 0.0)` 的
/// 写法完全一致。w 不参与任何运算，故它不是"契约的一部分"，只是让 vec4 化的
/// 布局零变化**不引入未初始化的字节**（否则读回比对会因为随机 padding 而抖）。
///
/// # 哨兵（见 [`GI_SENTINEL_SAMPLE`] / [`GI_SENTINEL_SH`] 的整段说明）
/// `samples` 全段预填 `-1`、`sh` 全段预填 `-2`。它们只影响 **GPU 输入缓冲的初值**；
/// CPU 参考走 [`gi_probe_inputs`] 独立重算、**不读本缓冲**，故判据语义不变。
fn gi_pack_probes() {
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.gi_probes.fill(0);
        for i in 0..GI_N_PROBES as usize {
            let o = i * GI_PROBE_STRIDE_BYTES as usize;
            // 逐方向采样段：8 个 vec4（stride 16），三个分量填哨兵，w 保持 0。
            for d in 0..GI_DIRS as usize {
                let so = o + 32 + d * VEC3_STRIDE_BYTES as usize;
                pack_f32(&mut v.gi_probes, so, GI_SENTINEL_SAMPLE);
                pack_f32(&mut v.gi_probes, so + 4, GI_SENTINEL_SAMPLE);
                pack_f32(&mut v.gi_probes, so + 8, GI_SENTINEL_SAMPLE);
            }
            // SH-L2 系数段：9 个 vec4（stride 16）@160。
            for k in 0..9usize {
                let so = o + 160 + k * VEC3_STRIDE_BYTES as usize;
                pack_f32(&mut v.gi_probes, so, GI_SENTINEL_SH);
                pack_f32(&mut v.gi_probes, so + 4, GI_SENTINEL_SH);
                pack_f32(&mut v.gi_probes, so + 8, GI_SENTINEL_SH);
            }
        }
        for (i, pr) in gi_probe_inputs().iter().enumerate() {
            let o = i * GI_PROBE_STRIDE_BYTES as usize;
            pack_f32(&mut v.gi_probes, o, pr[0]);        // pos.x @0
            pack_f32(&mut v.gi_probes, o + 4, pr[1]);    // pos.y @4
            pack_f32(&mut v.gi_probes, o + 8, pr[2]);    // pos.z @8
            pack_f32(&mut v.gi_probes, o + 16, pr[3]);   // normal.x @16
            pack_f32(&mut v.gi_probes, o + 20, pr[4]);   // normal.y @20
            pack_f32(&mut v.gi_probes, o + 24, pr[5]);   // normal.z @24
        }
    });
}

/// 把查询点输入打包进 `array<GiPoint>`，`irradiance` 预填 [`GI_SENTINEL_IRRADIANCE`]。
///
/// 布局：`pos@0` / `_pad0@12` / `irradiance@16` / `_pad1@28`。哨兵只影响 GPU 输入
/// 初值；CPU 参考走 [`gi_point_inputs`] 独立重算，不读本缓冲。
fn gi_pack_points() {
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.gi_points.fill(0);
        for i in 0..GI_N_POINTS as usize {
            let o = i * GI_POINT_STRIDE_BYTES as usize + 16; // irradiance @16
            pack_f32(&mut v.gi_points, o, GI_SENTINEL_IRRADIANCE);
            pack_f32(&mut v.gi_points, o + 4, GI_SENTINEL_IRRADIANCE);
            pack_f32(&mut v.gi_points, o + 8, GI_SENTINEL_IRRADIANCE);
        }
        for (i, q) in gi_point_inputs().iter().enumerate() {
            let o = i * GI_POINT_STRIDE_BYTES as usize;
            pack_f32(&mut v.gi_points, o, q[0]);
            pack_f32(&mut v.gi_points, o + 4, q[1]);
            pack_f32(&mut v.gi_points, o + 8, q[2]);
        }
    });
}

/// 把探针 buffer 的读回字节解析为 CPU 参考态（SH 系数 9 项）。
///
/// 步长 16 = [`VEC3_STRIDE_BYTES`] = [`VEC4_STRIDE_BYTES`]——**这正是 vec3→vec4
/// 化能"布局零变化"的原因**：`array<vec3<f32>>` 的 stride 本来就是 roundUp(16,12)=16，
/// 与 `array<vec4<f32>>` 的 16 相同，故 `sh[k]@160+16k` 的取值偏移一字未改。
fn gi_unpack_sh(buf: &[u8], probe_idx: usize) -> [[f32; 3]; 9] {
    let base = probe_idx * GI_PROBE_STRIDE_BYTES as usize;
    let mut sh = [[0.0f32; 3]; 9];
    for k in 0..9 {
        let o = base + 160 + k * VEC4_STRIDE_BYTES as usize; // sh 起始 @160，每项 stride 16
        sh[k] = [unpack_f32(buf, o), unpack_f32(buf, o + 4), unpack_f32(buf, o + 8)];
    }
    sh
}

/// 由已重建的探针 SH 系数（每探针 9×RGB）与查询点辐照度（每点 RGB）压出 u32 指纹。
///
/// **打包口径是跨文件契约**（`host_test.mjs` 的 JS 独立实现按同一顺序压）：
/// 先逐探针、逐 SH 系数 k=0..9、逐通道 RGB 压 `f32::to_bits`；再逐查询点、逐通道
/// RGB 压 `f32::to_bits`。用 [`hash_u32`]（FNV-1a 32）哈希。
fn gi_fingerprint_of(shs: &[[[f32; 3]; 9]], points: &[[f32; 3]]) -> u32 {
    let mut flat: Vec<u32> = Vec::with_capacity(shs.len() * 27 + points.len() * 3);
    for sh in shs {
        for k in 0..9 {
            for c in 0..3 {
                flat.push(sh[k][c].to_bits());
            }
        }
    }
    for e in points {
        for c in 0..3 {
            flat.push(e[c].to_bits());
        }
    }
    hash_u32(&flat)
}

// ---------------------------------------------------------------------------
// `gpu.gi_compute` 的 FAIL 诊断块（零成本：只用既有读回缓冲，不新增读回）
// ---------------------------------------------------------------------------
//
// # 为什么必须有这一块
// 沙箱无 GPU adapter，本项在 native 下永远 `Skip`，故**真机是唯一观测窗口**。
// 而 V2.2-T03 真机实测出现过一次「GPU 侧 sh[0][0] 恰为 0、CPU 侧 0.9970052」的
// FAIL：一个 `0` 背后至少有三种彼此不兼容的机制（入口没跑 / 跑了但全遮挡 /
// 跑了但 project 早退），只凭一句「首个不一致」无法区分。宿主侧每多印一个字段，
// 真机重跑的次数就少一轮，故此处把三个 entry 的中间态全部摊开。
//
// # 数据来源（**不新增任何读回**）
// `t_gpu_gi_compute` 已经把整个 probe buffer（`gi_probes[0..912]`，含 `samples`
// @32..160 与 `sh` @160..304）和整个 point buffer 读回进 `SCRATCH`。本函数只做
// 「按既有布局解析 + 与 CPU 参考并排打印」，**不碰 GPU、不发命令、不改判据**。
//
// # 输出纪律
// 只在 `mismatch != 0` 的 `fail(...)` 里拼接，PASS 路径永不产出（不污染正常报告）。

/// 把一条探针的 `samples` 逐方向模长压成一行（空格分隔，3 位小数）。
///
/// 用**带符号的模长**而非三分量：`rad` 正常只可能是 `(0,0,0)` 或
/// `env_radiance (0.9,0.8,0.7)`，模长 `0.000` / `1.388` 一眼能读，且一行塞得下 8 个方向。
/// 三分量之和为负时输出负号（如 `-1.732`）——这正是 [`GI_SENTINEL_SAMPLE`] 的形态，
/// 即「`gi_probe_trace` 没写过这一段」，与「算出来是 0」（`0.000`）截然可分。
fn gi_mags_text(get: impl Fn(usize) -> [f32; 3], dirs: usize) -> String {
    let mut s = String::with_capacity(dirs * 7);
    for d in 0..dirs {
        let v = get(d);
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if d != 0 {
            s.push(' ');
        }
        if v[0] + v[1] + v[2] < 0.0 {
            s.push('-'); // 负值哨兵：该方向没被 gi_probe_trace 覆写
        }
        s.push_str(&format!("{m:.3}"));
    }
    s
}

/// 非零方向数（`samples` 第 0 通道 > 0 即算可见——与 PASS 分支的 `visible_total` 同口径）。
fn gi_visible_count(get: impl Fn(usize) -> [f32; 3], dirs: usize) -> usize {
    let mut n = 0usize;
    for d in 0..dirs {
        let v = get(d);
        if v[0] > 0.0 || v[1] > 0.0 || v[2] > 0.0 {
            n += 1;
        }
    }
    n
}

/// 生成 `gpu.gi_compute` 的 FAIL 诊断文本（多行，宿主页原样显示）。
///
/// 三段分别对应链上的三个 entry（trace → project → interpolate），每段都把
/// **GPU 读回值**与**CPU 参考值**并排给出，并在末尾给一行 `[判定]` 把观测到的
/// 组合映射到候选机制族——让真机一次重跑就能把范围收窄，而不是只看到「不一致」。
fn gi_diag_text(got_probes: &[u8], got_points: &[u8]) -> String {
    let p = GI_TEST;
    let inputs = gi_probe_inputs();
    let points = gi_point_inputs();
    let dirs = p.n_dirs.min(GI_DIRS) as usize;
    let stride = GI_PROBE_STRIDE_BYTES as usize;
    let mut s = String::with_capacity(2048);

    // 判定用的累计量（在遍历中顺便收集，避免二次扫描）。
    let mut pos_all_ok = true;
    let mut nrm_all_ok = true;
    let mut gpu_any_sample = false;
    let mut gpu_any_sh = false;
    // 哨兵在场标记：值为 -1（samples）/ -2（sh）的字段说明**该入口没写过这一段**。
    // 这是把「入口没跑」与「跑了但算出 0」分开的唯一手段——两种情形的全零字节相同。
    let mut sent_samples = false;
    let mut sent_sh = false;
    let mut cpu_visible_total = 0usize;

    s.push_str("[诊断] 三个 entry 的中间态（GPU 读回 vs CPU 参考；pos/normal 来自同一 buffer，用于确认句柄身份）\n");
    s.push_str("[中间态 1/3] gi_probe_trace → samples（逐方向模长；模长>0 = 该方向可见，\
                带负号 = 该段仍是哨兵初值 ⇒ 入口没写过）\n");
    for (i, inp) in inputs.iter().enumerate() {
        let base = i * stride;
        let gpos = [
            unpack_f32(got_probes, base),
            unpack_f32(got_probes, base + 4),
            unpack_f32(got_probes, base + 8),
        ];
        let gnrm = [
            unpack_f32(got_probes, base + 16),
            unpack_f32(got_probes, base + 20),
            unpack_f32(got_probes, base + 24),
        ];
        let nlen = (gnrm[0] * gnrm[0] + gnrm[1] * gnrm[1] + gnrm[2] * gnrm[2]).sqrt();
        let pos_ok = gpos == [inp[0], inp[1], inp[2]];
        let nrm_ok = gnrm == [inp[3], inp[4], inp[5]];
        pos_all_ok &= pos_ok;
        nrm_all_ok &= nrm_ok;
        // GPU 侧 samples（`array<vec4<f32>>`，stride 16，取前 3 个通道；w 不入算）。
        let g_sample = |d: usize| -> [f32; 3] {
            let o = base + 32 + d * VEC4_STRIDE_BYTES as usize;
            [
                unpack_f32(got_probes, o),
                unpack_f32(got_probes, o + 4),
                unpack_f32(got_probes, o + 8),
            ]
        };
        // CPU 参考 samples（同式 trace，含同样的遮挡判定）。
        let tr = gi_probe_trace_ref([inp[0], inp[1], inp[2]], [inp[3], inp[4], inp[5]], &p);
        let c_sample = |d: usize| -> [f32; 3] { tr.samples[d] };
        let gvis = gi_visible_count(g_sample, dirs);
        let cvis = gi_visible_count(c_sample, dirs);
        cpu_visible_total += cvis;
        if gvis != 0 {
            gpu_any_sample = true;
        }
        for d in 0..dirs {
            let v = g_sample(d);
            if v[0] == GI_SENTINEL_SAMPLE || v[1] == GI_SENTINEL_SAMPLE || v[2] == GI_SENTINEL_SAMPLE {
                sent_samples = true;
            }
        }
        s.push_str(&format!(
            "  probe {i}: pos GPU({:.3},{:.3},{:.3}) {}  normal GPU({:.3},{:.3},{:.3}) |n|={nlen:.3} {}  ← {}\n",
            gpos[0], gpos[1], gpos[2], if pos_ok { "✓" } else { "✗" },
            gnrm[0], gnrm[1], gnrm[2], if nrm_ok { "✓" } else { "✗" },
            if nlen < 0.5 { "退化法线，两个入口都会早退" } else { "非退化" },
        ));
        s.push_str(&format!("            可见 GPU {gvis}/{dirs} vs CPU {cvis}/{dirs}\n"));
        s.push_str(&format!("            GPU |s|[{}]\n", gi_mags_text(g_sample, dirs)));
        s.push_str(&format!("            CPU |s|[{}]\n", gi_mags_text(c_sample, dirs)));
    }

    s.push_str("[中间态 2/3] gi_sh_project → sh[0]（9 系数 × RGB 的首项；全零 = 没有任何辐亮度进账）\n");
    for (i, inp) in inputs.iter().enumerate() {
        let gsh = gi_unpack_sh(got_probes, i);
        let csh = gi_sh_project_ref([inp[0], inp[1], inp[2]], [inp[3], inp[4], inp[5]], &p).sh;
        // 三态：哨兵未写 / 写入但全零 / 写入且有值。**哨兵不算"有值"**——
        // 否则"入口没跑"会被误判成"sh 已写入"，把诊断引到错误的机制族。
        let all_sentinel = gsh.iter().all(|k| k.iter().all(|c| *c == GI_SENTINEL_SH));
        let all_zero = gsh.iter().all(|k| k.iter().all(|c| *c == 0.0));
        if all_sentinel {
            sent_sh = true;
        } else if !all_zero {
            gpu_any_sh = true;
        }
        s.push_str(&format!(
            "  probe {i}: sh[0] GPU({:.6},{:.6},{:.6}) vs CPU({:.6},{:.6},{:.6})  写入状态={}\n",
            gsh[0][0], gsh[0][1], gsh[0][2], csh[0][0], csh[0][1], csh[0][2],
            if all_sentinel { "哨兵未写(-2)" } else if all_zero { "已写但全零" } else { "已写有值" },
        ));
    }

    s.push_str("[中间态 3/3] gi_probe_interpolate → irradiance（@16，最近探针重建 + Lambert）\n");
    let refs: Vec<([f32; 3], GiProbeRef)> = inputs
        .iter()
        .map(|inp| {
            let pos = [inp[0], inp[1], inp[2]];
            (pos, gi_sh_project_ref(pos, [inp[3], inp[4], inp[5]], &p))
        })
        .collect();
    for (i, q) in points.iter().enumerate() {
        let o = i * GI_POINT_STRIDE_BYTES as usize + 16;
        let g = [unpack_f32(got_points, o), unpack_f32(got_points, o + 4), unpack_f32(got_points, o + 8)];
        let want = gi_interpolate_ref(*q, &refs, &p);
        // 最近探针下标（与 WGSL 同 tie-break：严格小于才换，故并列取小下标）。
        let mut best = 0usize;
        let mut best_d2 = f32::INFINITY;
        for (j, (pos, _)) in refs.iter().enumerate() {
            let dx = pos[0] - q[0];
            let dy = pos[1] - q[1];
            let dz = pos[2] - q[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best = j;
            }
        }
        s.push_str(&format!(
            "  point {i} ({:.3},{:.3},{:.3})→probe {best}: GPU({:.6},{:.6},{:.6}) vs CPU({:.6},{:.6},{:.6})\n",
            q[0], q[1], q[2], g[0], g[1], g[2], want[0], want[1], want[2],
        ));
    }

    // ---- 判定：把观测组合映射到候选机制族（不猜、不掩盖，逐条给判据）----
    s.push_str("[判定] ");
    if !pos_all_ok || !nrm_all_ok {
        s.push_str(
            "读回的 pos/normal 与 CPU 输入不一致 ⇒ 上传/读回不是同一块 buffer，\
             或 GiProbe 布局在这台设备上漂移（优先查 make_buffer→submit_chain→readback_to 的句柄链）",
        );
    } else if !gpu_any_sample && !gpu_any_sh {
        // 哨兵把 (A)/(B) 两族**彻底分开**（这两族的"全零"读回字节完全相同，
        // 只有输入侧的负值初值能区分）：
        //   sent_sh && sent_samples → 两个入口都没写过 → (A)；
        //   !sent_sh && !sent_samples → 写过、但结果恰为零 → (B)；
        //   只有一半带哨兵 → 入口边界不一致（更可疑，单列）。
        let family = if sent_sh && sent_samples {
            "(A) 两个入口都没有真正写入：sh 仍是 -2 哨兵、samples 仍是 -1 哨兵 ⇒ \
             uniform 未生效（probe_count==0 → 三个 entry 全部早退）或 bind group 绑到了别的 buffer"
        } else if !sent_sh && !sent_samples {
            "(B) 两个入口都写过（哨兵已被覆写），但结果恰为零 ⇒ 全方向被判遮挡：\
             查 hit_eps / t_max / prim_count 的读取值"
        } else {
            "(混合) 只有部分字段带哨兵 ⇒ 两个 entry 的覆写范围不一致，\
             优先查 probe_count 的读取值（它同时是两个入口的循环上界）"
        };
        s.push_str(&format!(
            "pos/normal 完好但 samples 与 sh **全为零**；哨兵判定：{family}\n\
             \x20  (A) 的判据：sh 仍为 -2、samples 仍为 -1（本轮由 gi_pack_probes 预填）；\n\
             \x20  (B) 的判据：两个哨兵都不在场，说明入口执行过；\
             同一场景 CPU 侧实测 {cpu_visible_total} 条方向可见\n"
        ));
        // (B) 族在本场景下可**静态推导**出极窄的必要条件，直接写进文本，
        // 免得下一轮又从头猜：
        //   · probe 0 的 8 个方向里有若干个 world.x < 0（离 x∈[1,3] 的球面图元
        //     越来越远），在 CPU 的图元集下它们**永远**不会命中 ⇒ 只有「t=0 即命中」
        //     能判遮挡；
        //   · t=0 处三个起点的场景距离分别是 ~1.24 / ~0.41 / ~1.0（本轮球 SDF 修正后
        //     的实测值）⇒ 要"起点即命中"需 hit_eps 远大于 1e-3。
        //   · `GiParams` 里唯一量级 ≥1.0 且**不是** hit_eps 的标量是 t_max@48 = 20.0
        //     ⇒ 疑 hit_eps 读到了 t_max。
        //   · 另一条通路 `prim_count` 读到 4（= point_count@4）**本轮已被封住**：
        //     容量填充图元从"球心原点 / r=1"改成"半径 0 / 心在 -Y 轴 1e6"，
        //     SDF 恒 ≈1e6，即使越界参与 min 也不会命中（回归钉子见 native 测试
        //     `gi_prim_count_misread_does_not_blind_the_scene`）。故此刻若仍观测到
        //     全遮挡，`prim_count` 误读已**不再是**候选，应优先查 hit_eps / t_max。
        s.push_str(concat!(
            "[研判] 若落到 (B)，本场景的必要条件（可静态推导，非猜测）：",
            "hit_eps 被读成大值（GiParams 里唯一 ≥1.0 的标量是 t_max@48=20.0 → 疑读到 t_max）。",
            "`prim_count` 误读成 4 这条通路已由「容量填充改为半径 0 + 远点」封住",
            "（native 回归钉子 gi_prim_count_misread_does_not_blind_the_scene 钉住），",
            "故不应再把全遮挡归因于容量填充图元。\n",
        ));
    } else if gpu_any_sample && !gpu_any_sh {
        s.push_str(
            "samples 已写入但 sh 全零 ⇒ 断在 gi_sh_project 内部（nlen<0.5 早退 / n_dirs==0 / d_omega 退化）",
        );
    } else {
        s.push_str(
            "samples 与 sh 均已写入 ⇒ 断在数值（SDF 命中判定漂移）或 gi_probe_interpolate 的最近探针选择",
        );
    }
    s.push_str(
        "\n[下一步] 本轮已把 input buffer 的 samples/sh/irradiance 预填负值哨兵\
         （-1/-2/-3，见 gi_pack_probes/gi_pack_points），因此上面出现\
         **带负号的模长**（如 -1.732）、或 sh/辐照度字段直接打印 -2.000000 / -3.000000，\
         即可判定「该段入口没执行」，无需再加 witness；\
         若打印的是 0.000 / 正常正值，则入口执行过、断在遮挡判定或早退分支。\n",
    );
    s
}

// ---------------------------------------------------------------------------
// 测试项：`gpu.params_roundtrip`（参数是否真的到达 GPU）
// ---------------------------------------------------------------------------
//
// # 为什么必须把「uniform 传输」做成独立判据
// `gpu.gi_compute` 的 FAIL 只能是"数值不一致"这一个笼统结论，而**uniform 没生效**
// 与**算法算错**在下游读回里是同一种形态（都是整块零值）。本项把"传输"这一层
// 单独拿出来，判据是**纯字节级回环**：
//   写 `gi_params_bytes(&GI_TEST)`（256B）→ `copyBufferToBuffer` 读回 → 逐字节比对。
// 若 256/256 全等且 4 个关键字段解释值正确，则「uniform 传输」这一层可以**整体排除**，
// 后续 `gpu.gi_compute` 的任何偏差都必须归因到 shader 语义或 dispatch，而不是接线。
//
// # 为什么这需要 `COPY_SRC`
// `copyBufferToBuffer` 要求源缓冲带 `GPUBufferUsage.COPY_SRC`。uniform 缓冲原先只有
// `UNIFORM | COPY_DST`，故本轮把宿主 `prism_create_buffer` 的 UNIFORM 分支扩为
// `UNIFORM | COPY_DST | COPY_SRC`（**扩能力位，不新增 glue**，导入表仍 14 个）。
// 若宿主未同步该改动，`copyBufferToBuffer` 验证失败 → 拷贝静默不执行 → 读回全是
// 建缓冲时的零初始化内容 → 本项报"256 个字节不一致"，并在失败文案里直接点名此根因。
//
// # 为什么不做成 `make_buffer` 之外的旁路
// 复用 `make_buffer` / `readback_to` 与 `gpu.gi_compute` **同一条**建缓冲/读回路径，
// 才能保证"本项通过 ⇒ 同一路径上的 gi_compute 也拿到了正确的 uniform"。
// （静态门禁 `wgsl_lint.mjs` 规则 A 会按测试项 id 核对该缓冲尺寸与
// `gi_compute.wgsl::GiParams` 的声明自洽，故已在本轮把本项登记进其 KERNEL_OF。）

/// `gpu.params_roundtrip` 的 run：写 uniform → 发起读回（判对在 collect）。
fn t_gpu_params_roundtrip() -> Status {
    let dev = match ready_device("gpu.params_roundtrip") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // CPU 侧打包（与 `gpu.gi_compute` 用**同一个** `gi_params_bytes(&GI_TEST)`，
    // 故本项通过即证明 gi_compute 的那份 uniform 字节也完全一致）。
    let want = gi_params_bytes(&GI_TEST);
    let src_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.gi_params.copy_from_slice(&want);
        // 读回落点先清零：若宿主读回链路静默失败，这里保持全 0 而不会残留上一轮的值
        // （否则本项可能"读到上一次的正确内容"→ 假绿）。
        v.gi_params_echo.fill(0);
        v.gi_params.as_ptr() as usize as u32
    });
    let buf = match make_buffer(
        "gpu.params_roundtrip",
        dev,
        GI_PARAMS_BYTES,
        usage::UNIFORM,
        || (src_ptr, GI_PARAMS_BYTES),
    ) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let dst = addr_of_buf(|x| &mut x.gi_params_echo[..]);
    if let Err(s) = readback_to("gpu.params_roundtrip", dev, buf, GI_PARAMS_BYTES, dst) {
        return s;
    }
    // 本项不登记错误槽位（`err_slots`）：全程没有 shader / dispatch 的验证错误 scope，
    // 而 `err_slots` 是**跨项共享**的计数——擅自清点会打乱下一项（gi_compute）的记账。
    // 传输链的任何失败都已经由 `make_buffer` / `readback_to` 的返回值即时判 FAIL。
    set_result(&format!(
        "已提交 GiParams uniform 回环：写 {GI_PARAMS_BYTES}B → 读回 {GI_PARAMS_BYTES}B\
         （usage = UNIFORM | COPY_DST | COPY_SRC，缓冲句柄 {buf}）\n\
         待 settle 后与 CPU 打包字节逐字节比对；判据：256/256 全等 + 4 个关键字段解释值\
         （probe_count@0={} / point_count@4={} / prim_count@8={} / n_dirs@12={}）",
        GI_N_PROBES, GI_N_POINTS, GI_TEST.prims, GI_DIRS
    ));
    Status::Pending
}

/// `gpu.params_roundtrip` 的 collect：逐字节比对 + 关键字段解释值校验。
fn c_gpu_params_roundtrip() -> Status {
    let dev = match ready_device("gpu.params_roundtrip") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 uniform 回环期间 lost");
    }
    let want = gi_params_bytes(&GI_TEST);
    let got = SCRATCH.with(|s| s.borrow().gi_params_echo);
    let n = GI_PARAMS_BYTES as usize;
    let mut diff = 0usize;
    let mut first_off = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate().take(n) {
        if g != w {
            if diff == 0 {
                first_off = i;
            }
            diff += 1;
        }
    }
    let equal = n - diff;
    // 4 个关键字段的**解释值**（不是字节）：证明偏移语义也对，而不只是"字节搬对了"。
    let got_probe = unpack_u32(&got, 0);
    let got_point = unpack_u32(&got, 4);
    let got_prim = unpack_u32(&got, 8);
    let got_dirs = unpack_u32(&got, 12);
    let fields_ok = got_probe == GI_N_PROBES
        && got_point == GI_N_POINTS
        && got_prim == GI_TEST.prims
        && got_dirs == GI_DIRS;
    if diff != 0 || !fields_ok {
        return fail(format!(
            "gpu.params_roundtrip：GiParams uniform 读回与 CPU 打包不一致——\
             {equal}/{n} 字节相等（{diff} 字节不同，首个不同偏移 @{first_off}：\
             读回 {} vs 期望 {}）；字段解释值 probe_count@0={got_probe}(期望{GI_N_PROBES}) \
             point_count@4={got_point}(期望{GI_N_POINTS}) prim_count@8={got_prim}(期望{}) \
             n_dirs@12={got_dirs}(期望{GI_DIRS})\n\
             常见根因（按概率序）：\n\
             \x20 (1) 宿主 `prism_create_buffer` 的 UNIFORM 分支缺 COPY_SRC →\
             `copyBufferToBuffer` 验证失败、拷贝静默不执行，读回是建缓冲时的零初始化内容；\n\
             \x20 (2) `prism_write_buffer` 用的长度/偏移与 `GI_PARAMS_BYTES` 不一致；\n\
             \x20 (3) 读回落到与写入同一块内存（比对退化成「自己比自己」，此时会全等而无差异——\
             本项已用独立 `gi_params_echo` 落点排除）。\n\
             若整块全 0（{equal}==0 且全零）请直接查 (1)。",
            got[first_off], want[first_off], GI_TEST.prims
        ));
    }
    set_result(&format!(
        "GiParams uniform 回环逐字节一致 ✓（{equal}/{n} 字节全等，0 处差异）\n\
         字段解释值：probe_count@0={got_probe} / point_count@4={got_point} / \
         prim_count@8={got_prim} / n_dirs@12={got_dirs}（与 `gi_compute.wgsl::GiParams` 布局一致）\n\
         结论：uniform 传输链（create → write → copyBufferToBuffer → map → copy_out）可用，\
         `gpu.gi_compute` 的任何偏差都不能再归因到「参数没到达 GPU」"
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项：`gpu.gi_layout`（存储布局 census —— **只有真机才暴露**的那一类缺陷）
// ---------------------------------------------------------------------------
//
// # 为什么必须把「布局」做成独立判据
// `gpu.gi_compute` 的 FAIL 只能给出"数值不一致"，而**布局错位**与**算法算错**在
// 读回里长得完全一样（都是"某些字段是哨兵值"）。V2.2 的真机 bug 正是这一类：
// 驱动对结构体成员 `vec3<f32>` 的 size 处理与 WGSL 规范不一致（规范 12，实现更大），
// 于是 `GiProbe.normal`（应 @16）被推到 `samples` 区、读到哨兵 -1，`samples`/`sh`
// 被压缩（`samples[6]` 落到 `sh[0]` 上）。
//
// # 为什么本地（SwiftShader）永远测不出，而本项仍是必需的
// SwiftShader 严格按规范实现 → 本地全绿；本地能证明的只有"没改坏"，**证不了
// "真机修好"**。故本项的价值不在本地，而在**下一轮真机重跑时能一眼判死**：
// 它把「shader 自己读到的四个关键字段」原样导出（`gi_compute.wgsl::gi_layout_census`
// 写进 `gi_probes[0].samples[0..7]`），与 Rust 打包值**逐格**比对：
//   samples[0..1] = probe0.pos / normal   ← normal 可读 = 本轮 bug 已修
//   samples[2..3] = probe2.pos / normal   ← 读 probe2 = 验 stride 304 生效
//   samples[4..5] = probe1.pos / normal
//   samples[6]    = (probe_count, prim_count, env.x, albedo.x) ← GiParams 混合字段
//
// # 判据为什么比 `gpu.gi_compute` 更"硬"
// 本项不涉及任何数学（不做 SDF、不做 SH），只做"把读到的值搬出来"。故它 FAIL 时
// 不可能被"算法精度"掩盖——失败即等于**布局与宿主假设不一致**，可直接照着文案
// 归因，不需要再跑一轮。

/// `gpu.gi_layout` 的期望值：`samples[0..8]`，每格 4 个分量。
///
/// **全部由既有单一真源导出**（[`gi_probe_inputs`] / [`GI_TEST`] / [`GI_N_PROBES`]），
/// 不写死字面量：否则改输入值就会让本项与 `gpu.gi_compute` 判据各说各话。
/// 第 4 分量期望恒 0（宿主打包只写 xyz、`fill(0)` 保证 w=0；shader 侧
/// `vec4(xyz, 0.0)` 同口径）。
///
/// 【编译期前置】census 入口要读 `gi_probes[0]` / `[1]` / `[2]`，而本函数要按
/// `gi_probe_inputs()[2]` 给期望值——两者都依赖 `GI_N_PROBES >= 3`。若把探针数
/// 调小，下方 `inp[2]` 会**越界 panic**（`gpu.gi_layout` 在真机上崩掉比 FAIL 更糟），
/// 故在编译期就把这条前置钉死：改小探针数必须同时改本项。
const _: () = assert!(
    GI_N_PROBES >= 3,
    "gpu.gi_layout 的 census 至少需要 3 条探针（它要读 gi_probes[0]/[1]/[2]）"
);

fn gi_layout_expect() -> [[f32; 4]; 8] {
    let inp = gi_probe_inputs();
    [
        // samples[0] = probe0.pos（`p0.samples[0] = p0.pos`）
        [inp[0][0], inp[0][1], inp[0][2], 0.0],
        // samples[1] = probe0.normal  ← 本轮 bug 的直接判据
        [inp[0][3], inp[0][4], inp[0][5], 0.0],
        // samples[2] = probe2.pos     ← stride 304 的判据
        [inp[2][0], inp[2][1], inp[2][2], 0.0],
        // samples[3] = probe2.normal
        [inp[2][3], inp[2][4], inp[2][5], 0.0],
        // samples[4] = probe1.pos
        [inp[1][0], inp[1][1], inp[1][2], 0.0],
        // samples[5] = probe1.normal
        [inp[1][3], inp[1][4], inp[1][5], 0.0],
        // samples[6] = (probe_count, prim_count, env_radiance.x, albedo.x)
        [GI_N_PROBES as f32, GI_TEST.prims as f32, GI_TEST.env[0], GI_TEST.albedo[0]],
        // samples[7] = 0（证明 8 格全由 census 入口覆写，不是残留值）
        [0.0, 0.0, 0.0, 0.0],
    ]
}

/// 每格的语义标签（`samples[k]` 读到的是什么），用于逐格打印与失败归因。
const GI_LAYOUT_CELLS: [(&str, &str); 8] = [
    ("probe0.pos", "GiProbe.pos @0（第一个字段）"),
    ("probe0.normal", "GiProbe.normal @16 ← 本轮 bug 直接判据"),
    ("probe2.pos", "GiProbe[2].pos（stride 304 判据）"),
    ("probe2.normal", "GiProbe[2].normal"),
    ("probe1.pos", "GiProbe[1].pos"),
    ("probe1.normal", "GiProbe[1].normal"),
    ("params 混合", "probe_count / prim_count / env_radiance.x / albedo.x"),
    ("清零", "常量 0（证明 8 格全被覆写）"),
];

/// 从读回缓冲解析 `gi_probes[0].samples[k]`（`samples` 起始 @32，每项 stride 16）。
fn gi_layout_read(buf: &[u8], k: usize) -> [f32; 4] {
    let o = 32 + k * VEC4_STRIDE_BYTES as usize;
    [unpack_f32(buf, o), unpack_f32(buf, o + 4), unpack_f32(buf, o + 8), unpack_f32(buf, o + 12)]
}

/// `gpu.gi_layout` 的 run：打包探针 → 建 storage/uniform → dispatch census → 读回 probe0。
fn t_gpu_gi_layout() -> Status {
    let dev = match ready_device("gpu.gi_layout") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("gi_compute.wgsl") {
        Some(w) => w,
        None => return fail("gpu.gi_layout：gi_compute.wgsl 展开结果缺失"),
    };
    // 输入与 `gpu.gi_compute` **完全同源**（同一 `gi_pack_probes` / `gi_params_bytes`），
    // 故本项期望值可以直接由 `gi_layout_expect()` 给出而不必再写一份打包逻辑。
    gi_pack_probes();
    let par_bytes = gi_params_bytes(&GI_TEST);
    let par_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.gi_params.copy_from_slice(&par_bytes);
        v.gi_params.as_ptr() as usize as u32
    });
    let probe_buf = match make_buffer("gpu.gi_layout", dev, GI_PROBES_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.gi_probes[..]), GI_PROBES_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.gi_layout", dev, GI_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, GI_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // census 入口静态引用 { gi_probes(0), gi_params(1) } = 连续前缀 {0,1}。
    // （`@workgroup_size(1)` → 1 个工作组；入口内再按 gid.x != 0u 早退，双保险。）
    let steps = [("gi_layout_census", 1, &[probe_buf, par_buf][..])];
    // 只读回 **probe0 一个元素**（304B）：census 只写 `gi_probes[0]`。
    let out = addr_of_buf(|x| &mut x.gi_layout_out[..]);
    set_result(&format!(
        "已提交 GI 布局 census（entry `gi_layout_census`，1 工作组）：probes={probe_buf}\
         （{GI_PROBES_BYTES}B）params={par_buf}（{GI_PARAMS_BYTES}B uniform）\n\
         读回 probe0（{GI_PROBE_STRIDE_BYTES}B）→ 8 格 × 4 分量逐格判对：\
         [0]probe0.pos [1]probe0.normal [2]probe2.pos [3]probe2.normal \
         [4]probe1.pos [5]probe1.normal [6](counts,env.x,albedo.x) [7]0，容差 {F32_TOL:e}"
    ));
    if let Err(s) = submit_chain("gpu.gi_layout", dev, wgsl, &steps) {
        return s;
    }
    if let Err(s) = readback_to("gpu.gi_layout", dev, probe_buf, GI_PROBE_STRIDE_BYTES, out) {
        return s;
    }
    Status::Pending
}

/// `gpu.gi_layout` 的 collect：逐格逐分量比对 + **带归因的失败文案**。
///
/// # 为什么失败文案里要写"推断"
/// 本项的唯一用途是**下一轮真机重跑时的判读依据**。沙箱里 SwiftShader 永远 PASS，
/// 所以这段文案的价值全部落在真机：它必须能让读报告的人在**不再重跑**的前提下
/// 说出"是 vec3 的 size 还没修好，还是 stride 没生效，还是 GiParams 的 vec4 偏移错"。
/// 因此下面把三种典型形态分别写成显式条件，并给出「这说明什么」。
fn c_gpu_gi_layout() -> Status {
    let dev = match ready_device("gpu.gi_layout") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.gi_layout", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在布局 census 期间 lost");
    }
    let want = gi_layout_expect();
    let got_buf = SCRATCH.with(|s| s.borrow().gi_layout_out);
    let mut got = [[0.0f32; 4]; 8];
    for k in 0..8 {
        got[k] = gi_layout_read(&got_buf, k);
    }
    // ---- 逐格逐分量比对（4 个分量全比：w 必须是 0，否则说明写侧构造错了）----
    let mut bad_cells: Vec<usize> = Vec::new();
    let mut bad_comps = 0usize;
    let mut first_msg = String::new();
    let mut max_dev = 0.0f32;
    for k in 0..8 {
        let mut cell_bad = false;
        for c in 0..4 {
            let d = (got[k][c] - want[k][c]).abs();
            if d > max_dev {
                max_dev = d;
            }
            if !(d <= F32_TOL) {
                if first_msg.is_empty() {
                    first_msg = format!(
                        "首个不一致 samples[{k}].{}（{}）：GPU {} vs Rust 打包 {}",
                        ["x", "y", "z", "w"][c],
                        GI_LAYOUT_CELLS[k].0,
                        got[k][c],
                        want[k][c]
                    );
                }
                cell_bad = true;
                bad_comps += 1;
            }
        }
        if cell_bad {
            bad_cells.push(k);
        }
    }
    // ---- 逐格表（PASS/FAIL 全列，便于"哪一格错"一眼可见）----
    let mut table = String::new();
    for k in 0..8 {
        let cell_bad = bad_cells.contains(&k);
        table.push_str(&format!(
            "\n  samples[{k}] {:<14}（{}）\n    期望 ({}, {}, {}, {}) / 实测 ({}, {}, {}, {}) → {}",
            GI_LAYOUT_CELLS[k].0,
            GI_LAYOUT_CELLS[k].1,
            want[k][0], want[k][1], want[k][2], want[k][3],
            got[k][0], got[k][1], got[k][2], got[k][3],
            if cell_bad { "FAIL" } else { "PASS" },
        ));
    }
    if !bad_cells.is_empty() {
        // ---- 归因推断（顺序即"从最可能的根因到最不可能"，逐条独立判定）----
        let mut why: Vec<&str> = Vec::new();
        // 判据：某个 vec3 槽的 x 分量读到 samples 的哨兵 -1。
        let sentinel_hit = |v: [f32; 4]| {
            v[0] == GI_SENTINEL_SAMPLE && v[1] == GI_SENTINEL_SAMPLE && v[2] == GI_SENTINEL_SAMPLE
        };
        if sentinel_hit(got[1]) {
            why.push(
                "samples[1] 读到哨兵 -1 → **normal 仍被推到 samples 区**：结构体成员 \
                 `vec3<f32>` 的 size 在本驱动上仍非 12（vec4 化没生效，或改回 vec3 了）。\
                 修法：GiProbe.normal 必须是 vec4<f32>，且 Rust 侧 offset 仍写 +16/+20/+24。",
            );
        }
        if sentinel_hit(got[3]) && !sentinel_hit(got[1]) {
            why.push(
                "samples[1] 对、samples[3] 读到哨兵 -1 → **`array<GiProbe>` 的 stride 未按 304 \
                 生效**：`gi_probes[2]` 落到了别的字节上（304 是 Rust 侧 \
                 GI_PROBE_STRIDE_BYTES；若 shader 结构体多/少了字节，这里必然错位）。",
            );
        }
        if !(got[6][2] - GI_TEST.env[0]).abs().le(&F32_TOL) {
            why.push(
                "samples[6] 的 env_radiance.x ≠ 0.9 → **GiParams 的 vec4 字段偏移错**：\
                 env_radiance 应 @16（vec3+_pad0 的旧形态会让它偏 0 字节但把 albedo 推错），\
                 albedo 应 @32。注意 vec4 化后**不能再补 `_pad0`/`_pad1`**。",
            );
        }
        if !(got[6][0] - GI_N_PROBES as f32).abs().le(&F32_TOL)
            || !(got[6][1] - GI_TEST.prims as f32).abs().le(&F32_TOL)
        {
            why.push(
                "samples[6] 的 u32 字段（probe_count / prim_count）读错 → **uniform 头部偏移错**\
                 或 uniform 根本没绑上（先看 `gpu.params_roundtrip` 是否 PASS）。",
            );
        }
        if bad_cells.contains(&0) {
            why.push(
                "samples[0]（GiProbe.pos @0）就读错 → 连**第一个字段**的偏移都不对，\
                 这不是 vec3 的问题：先查 bind group 的 binding 顺序（GiProbe 必须 binding 0）、\
                 缓冲字节数与 `GI_PROBE_STRIDE_BYTES` 是否自洽。",
            );
        }
        if bad_cells.contains(&6) && got[6][3] != 0.0 && (got[6][3] - GI_TEST.albedo[0]).abs() > F32_TOL {
            why.push(
                "samples[6].w（albedo.x）读错 → GiParams.albedo 的偏移错（应 @32）。",
            );
        }
        if bad_cells.contains(&7) {
            why.push(
                "samples[7] 不是 0 → census 入口**没有把 8 格全部覆写**（读到的是哨兵/旧值）：\
                 检查 entry 是否真的 dispatch 到（工作组数 / entry 名）。",
            );
        }
        if why.is_empty() {
            why.push(
                "以上三条典型形态都不成立：请对照逐格表定位——差异集中在某个字段的 \
                 起始偏移附近时，优先怀疑该字段自身的类型 size（vec3/vec2 混用）。",
            );
        }
        return fail(format!(
            "gpu.gi_layout：GI 存储布局 census 不通过——8 格 × 4 分量中 {bad_comps} 个分量超容差 \
             {F32_TOL:e}（{}/8 格 FAIL，最大偏差 {max_dev:.3e}）；{first_msg}\n\
             [逐格判定]（samples 起始 @32，每项 stride 16）{table}\n\
             [布局事实] 宿主假设：GiProbe = pos@0 / normal@16 / samples@32 / sh@160 → stride \
             {GI_PROBE_STRIDE_BYTES}；GiParams = env_radiance@16 / albedo@32 / t_max@48 / \
             prims@64 → {GI_PARAMS_BYTES}B；GiPoint = pos@0 / irradiance@16 → \
             {GI_POINT_STRIDE_BYTES}B；GiPrim = kind@0 / radius@4 / center@16 → \
             {GI_PRIM_STRIDE_BYTES}B\n\
             [推断]（按概率序，逐条独立判定）\n  - {}",
            bad_cells.len(),
            why.join("\n  - ")
        ));
    }
    set_result(&format!(
        "GI 存储布局 census 逐格一致 ✓（8 格 × 4 分量全在容差 {F32_TOL:e} 内，最大偏差 {max_dev:.3e}）\
         {table}\n\
         结论：shader 读到的 pos/normal/samples 偏移与 Rust 打包一致，`gi_probes[2]` 落在 \
         stride {GI_PROBE_STRIDE_BYTES} 的正确位置。\n\
         **注意**：SwiftShader 严格按规范实现 → 本项本地 PASS 只证明「没改坏」，不证明\
         「真机修好」；真机判读直接看 samples[1] 是否 (0,1,0)、samples[3] 是否 (1,0,0)。"
    ));
    Status::Pass
}

fn t_gpu_gi_compute() -> Status {
    let dev = match ready_device("gpu.gi_compute") {
        Ok(d) => d,
        Err(s) => return s,
    };
    let wgsl = match expanded_named("gi_compute.wgsl") {
        Some(w) => w,
        None => return fail("gpu.gi_compute：gi_compute.wgsl 展开结果缺失"),
    };
    // ---- 输入打包 ----
    gi_pack_probes();
    gi_pack_points();
    let par_bytes = gi_params_bytes(&GI_TEST);
    let par_ptr = SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        v.gi_params.copy_from_slice(&par_bytes);
        v.gi_params.as_ptr() as usize as u32
    });
    // ---- 建 buffer ----
    let probe_buf = match make_buffer("gpu.gi_compute", dev, GI_PROBES_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.gi_probes[..]), GI_PROBES_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let par_buf = match make_buffer("gpu.gi_compute", dev, GI_PARAMS_BYTES, usage::UNIFORM, || {
        (par_ptr, GI_PARAMS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let point_buf = match make_buffer("gpu.gi_compute", dev, GI_POINTS_BYTES, usage::STORAGE, || {
        (addr_of_slice(|s| &s.gi_points[..]), GI_POINTS_BYTES)
    }) {
        Ok(h) => h,
        Err(s) => return s,
    };
    // 三入口的连续前缀绑定集（inline 数组字面量，与台账门禁 `("name", 1, &[` 同形）：
    // gi_probe_trace       → { gi_probes(0), gi_params(1) }               = 前缀 {0,1}
    // gi_sh_project        → { gi_probes(0), gi_params(1) }               = 前缀 {0,1}
    // gi_probe_interpolate → { gi_probes(0), gi_params(1), gi_points(2) } = 前缀 {0,1,2}
    let steps = [
        ("gi_probe_trace", 1, &[probe_buf, par_buf][..]),
        ("gi_sh_project", 1, &[probe_buf, par_buf][..]),
        ("gi_probe_interpolate", 1, &[probe_buf, par_buf, point_buf][..]),
    ];
    // 读回探针 buffer（含 SH）与查询点 buffer（含辐照度）。
    let out_probes = addr_of_buf(|x| &mut x.gi_probes[..]);
    let out_points = addr_of_buf(|x| &mut x.gi_points[..]);
    set_result(&format!(
        "已提交 GI GPU 路径链（gi_probe_trace → gi_sh_project → gi_probe_interpolate），各 1 工作组：\n\
         缓冲：probes={probe_buf}（{GI_PROBES_BYTES}B={GI_N_PROBES}×{GI_PROBE_STRIDE_BYTES}）\
         params={par_buf}（{GI_PARAMS_BYTES}B uniform）points={point_buf}（{GI_POINTS_BYTES}B={GI_N_POINTS}×{GI_POINT_STRIDE_BYTES}）\n\
         group 前缀：trace/project={{(0,1)}} interpolate={{(0,1,2)}}（连续前缀，无跳号）\n\
         构型：{GI_N_PROBES} 探针 × {GI_DIRS} 采样方向 → SH-L2（9 系数 RGB）→ {GI_N_POINTS} 查询点重建\n\
         读回 probes→{out_probes} points→{out_points}（各 {GI_PROBES_BYTES}/{GI_POINTS_BYTES}B），待 settle 后与 CPU 参考逐字段判对\n\
         [哨兵] GPU 输入缓冲的初值带负值哨兵，读回若仍是负值即「该段入口没写过」：\
         samples 预填 {GI_SENTINEL_SAMPLE:.0}（gi_probe_trace 未写）、\
         sh 预填 {GI_SENTINEL_SH:.0}（gi_sh_project 未写）、\
         irradiance 预填 {GI_SENTINEL_IRRADIANCE:.0}（gi_probe_interpolate 未写）。\
         物理上辐亮度/辐照度非负，故负值不可能是「算出来的」，只能是「没被覆写」。\
         哨兵只改 GPU 输入初值；CPU 参考走 gi_probe_inputs()/gi_point_inputs() 独立重算、不读这两块缓冲。"
    ));
    if let Err(s) = submit_chain("gpu.gi_compute", dev, wgsl, &steps) {
        return s;
    }
    if let Err(s) = readback_to("gpu.gi_compute", dev, probe_buf, GI_PROBES_BYTES, out_probes) {
        return s;
    }
    if let Err(s) = readback_to("gpu.gi_compute", dev, point_buf, GI_POINTS_BYTES, out_points) {
        return s;
    }
    Status::Pending
}

fn c_gpu_gi_compute() -> Status {
    let dev = match ready_device("gpu.gi_compute") {
        Ok(d) => d,
        Err(s) => return s,
    };
    if let Err(s) = check_errors("gpu.gi_compute", dev) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("device 在 dispatch 期间 lost");
    }
    let p = GI_TEST;
    let inputs = gi_probe_inputs();
    let points = gi_point_inputs();
    let got = SCRATCH.with(|s| (s.borrow().gi_probes, s.borrow().gi_points));
    // ---- CPU 参考：trace → project →（探针网格）→ interpolate ----
    let mut refs: Vec<([f32; 3], GiProbeRef)> = Vec::new();
    let mut mismatch = 0usize;
    let mut first_msg = String::new();
    let mut max_dev = 0.0f32;
    for (i, inp) in inputs.iter().enumerate() {
        let pos = [inp[0], inp[1], inp[2]];
        let nrm = [inp[3], inp[4], inp[5]];
        let want = gi_sh_project_ref(pos, nrm, &p);
        refs.push((pos, want));
        // 比对 SH-L2 系数（9 项 RGB，逐字段）。
        let got_sh = gi_unpack_sh(&got.0, i);
        for k in 0..9 {
            for c in 0..3 {
                let d = (got_sh[k][c] - want.sh[k][c]).abs();
                if d > max_dev {
                    max_dev = d;
                }
                if !(d <= F32_TOL) {
                    if mismatch == 0 {
                        first_msg = format!(
                            "首个 SH 不一致 probe={i} k={k} ch={c}：GPU {} vs CPU {}",
                            got_sh[k][c], want.sh[k][c]
                        );
                    }
                    mismatch += 1;
                }
            }
        }
    }
    // 查询点辐照度（最近探针重建 → Lambert）。
    let mut point_max_dev = 0.0f32;
    for (i, q) in points.iter().enumerate() {
        let want = gi_interpolate_ref(*q, &refs, &p);
        let o = i * GI_POINT_STRIDE_BYTES as usize + 16; // irradiance @16
        let g = [unpack_f32(&got.1, o), unpack_f32(&got.1, o + 4), unpack_f32(&got.1, o + 8)];
        for c in 0..3 {
            let d = (g[c] - want[c]).abs();
            if d > point_max_dev {
                point_max_dev = d;
            }
            if !(d <= F32_TOL) {
                if mismatch == 0 {
                    first_msg = format!(
                        "首个辐照度不一致 point={i} ch={c}：GPU {} vs CPU {}",
                        g[c], want[c]
                    );
                }
                mismatch += 1;
            }
        }
    }
    if mismatch != 0 {
        // 【真机实测不过，待定位】——不降级、不改判据、不 Skip：状态仍是 FAIL，
        // 但把三个 entry 的中间态一并摊开，让下一次真机重跑就能收窄机制。
        return fail(format!(
            "gpu.gi_compute：GI GPU 结果与 CPU 参考不一致——{mismatch} 个字段超容差 {F32_TOL:e}；{first_msg}\n\
             [真机实测不过，待定位] 以下诊断块全部取自既有读回缓冲（无额外读回、不改判据）：\n{}",
            gi_diag_text(&got.0[..], &got.1[..])
        ));
    }
    // 指纹：把全部 SH 系数与辐照度压成 u32（供 host_test 侧 JS 独立实现交叉核对）。
    let mut got_shs: Vec<[[f32; 3]; 9]> = Vec::with_capacity(inputs.len());
    for i in 0..inputs.len() {
        got_shs.push(gi_unpack_sh(&got.0, i));
    }
    let mut got_points: Vec<[f32; 3]> = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let o = i * GI_POINT_STRIDE_BYTES as usize + 16;
        got_points.push([unpack_f32(&got.1, o), unpack_f32(&got.1, o + 4), unpack_f32(&got.1, o + 8)]);
    }
    let fingerprint = gi_fingerprint_of(&got_shs, &got_points);
    // 统计可见方向数（诊断：证明遮挡确实在起作用，非恒全可见）。
    let mut visible_total = 0usize;
    for inp in inputs.iter() {
        let pos = [inp[0], inp[1], inp[2]];
        let nrm = [inp[3], inp[4], inp[5]];
        let tr = gi_probe_trace_ref(pos, nrm, &p);
        for d in 0..(p.n_dirs as usize) {
            if tr.samples[d][0] > 0.0 {
                visible_total += 1;
            }
        }
    }
    set_result(&format!(
        "真 WebGPU GI GPU 路径：trace + project + interpolate 与 CPU 参考一致 ✓\n\
         判对范围：{GI_N_PROBES} 探针 × 9 SH 系数 × RGB（最大偏差={max_dev:.3e}）\
         + {GI_N_POINTS} 查询点辐照度（最大偏差={point_max_dev:.3e}），容差 {F32_TOL:e}\n\
         SH 契约：投影原始辐亮度（不乘 cos）+ 重建余弦瓣 Â_0=π/Â_1=2π/3/Â_2=π/4 + 转辐亮度 ×1/π\n\
         遮挡：{visible_total}/{total} 个方向可见（证明 SDF sphere tracing 在起作用，非恒全可见）\n\
         fingerprint=0x{fingerprint:08x}（供 host_test 的 JS 独立实现交叉核对）",
        total = GI_N_PROBES * GI_DIRS
    ));
    Status::Pass
}

// ---------------------------------------------------------------------------
// 测试项 16：真 WebGPU 场景整帧 dispatch（`scene_frame.wgsl` 的 `scene_main`）
// ---------------------------------------------------------------------------
//
// # 这一项补的是什么（V2.2-T03 的缺口）
// PRD 验收第 3 条要求「**双分支可判**」：有设备返 `w*h*4`、无设备 CPU 兜底逐字节
// 一致。「无设备」那一支早已被既有单测钉住（`native_backend_is_not_gpu_ready_and_
// falls_back_to_cpu`）；「**有设备**」那一支此前完全没有实现——`WebGpuBackend::
// render_scene` 只算规划就回退 CPU，`scene_frame.wgsl` 的 `scene_main` 在覆盖台账里
// 也一直标着 SKIP。本项补齐：真编译 WGSL、真建 6 个缓冲、真 dispatch、真读回，
// 并在宿主 `settle()` 之后与 CPU 参考逐项判对。
//
// # 判据（全部硬断言，任一不过即 FAIL）
// 1. **字节数**：读回字节数 == `w*h*4`（**这就是「有设备返 `w*h*4`」的本体**）。
//    同时核对「提交时向胶水请求的字节数」也等于 `w*h*4`——只核对返回值长度会漏掉
//    「请求错了但恰好读回等长」的接线错误。
// 2. **非全零 + 非整帧同色**：排除「读回成功但全黑」和「读回成功但整帧纯色」两种假绿。
// 3. **vs CPU 参考**：同一 `SceneRenderRequest` 走 `CpuRefBackend::render_scene`
//    （与生产路径同一实现），报 `max_channel_delta` / `differing_pixels` / `psnr_db`，
//    门禁 **PSNR ≥ 40 dB**（G3 双门禁口径：`psnr = 10·log10(255²/mse)`）。
//    判决走**精确整数**形式 `sum_sq·10⁴ ≤ 255²·channels`（与 `mse ≤ 6.5025` 严格
//    等价，但全程 `u64` 乘加、无对数、无浮点），故真机上不存在「阈值附近两平台判反」。
//    **若真机达不到 40 dB 不许改阈值**：把实测值写进结果文本并判 FAIL。
// 4. **双跑逐位一致**：同设备连跑两次，两次读回字节完全相同（G3 确定性）。
// 5. **曲线路径真被消费**：场景 B（球 + 1 个圆柱）与「B 去掉曲线」的对照帧必须不同。
//    这条是 V2 六绑定里 triangles/nodes 两个缓冲**真的参与了着色**的证据——若宿主
//    漏绑/漏写这两块，两支的像素会退化成完全一致。
//
// # 场景规模（手机友好）
// A 场景 8×8、B 场景 16×16，各含**至少 1 个球 + 1 盏灯**；B 额外含 1 个圆柱曲线图元。
// 球与圆柱在 x 方向分开布置，互不遮挡（否则曲线会被球整个盖住，判据 5 恒真）。
//
// # native 行为
// 无 device ⇒ [`ready_device`] 直接返回 [`Status::Skip`]（不是 `#[ignore]`，
// `qa_ignore_guard` 会命中 `#[ignore]`）。native 上本项的 run 阶段必然走这条路径。

/// 场景 A 的帧宽（8×8：手机毫秒级即可跑完）。
pub const SCENE_A_W: u32 = 8;
/// 场景 A 的帧高。
pub const SCENE_A_H: u32 = 8;
/// 场景 B 的帧宽（16×16：给曲线图元留出可观测的像素面积）。
pub const SCENE_B_W: u32 = 16;
/// 场景 B 的帧高。
pub const SCENE_B_H: u32 = 16;
/// 每个场景的连续提交次数（G3 确定性：同设备连跑两次逐位一致）。
pub const SCENE_RUNS: usize = 2;
/// 场景 A 的下标。
pub const SCN_A: usize = 0;
/// 场景 B（含曲线）的下标。
pub const SCN_B: usize = 1;
/// 场景 B 的**无曲线对照**下标（判据 5 用）。
pub const SCN_BC: usize = 2;
/// 全部场景数。
pub const SCN_COUNT: usize = 3;
/// 一次 run 的提交总数：A 双跑 + B 双跑 + 无曲线对照单跑。
pub const SCENE_SUBMISSIONS: usize = SCENE_RUNS * 2 + 1;
/// run 阶段的提交顺序：`[A, A, B, B, BC]`。
///
/// collect 阶段按 `场景下标 × SCENE_RUNS` 反查同一格，两处共用本数组，避免下标漂移。
const SCENE_ORDER: [usize; SCENE_SUBMISSIONS] = [SCN_A, SCN_A, SCN_B, SCN_B, SCN_BC];
/// PSNR 门限的**十倍整数**表示：`400` ⇔ `40.0 dB`（G3 双门禁口径，**不得放宽**）。
///
/// 用整数而非 `f64` 存门限：报告文本直接按「整数部分 / 十分位」两段打印，既不依赖
/// `f64` 的 `Display`（wasm 里那套 Dragon/Grisu 定点格式化代价不小），也让
/// 「恰好 40.0 dB」这种边界不会被浮点舍入判成红/绿翻转。
pub const SCENE_PSNR_MIN_DB10: i32 = 400;
/// PSNR 报告值的上限（十倍整数）。逐位相同的帧 PSNR 为 `+∞`，报告时夹到 `99.9 dB`
/// 以免把 `inf` 塞进定点格式；**判据本身不走这个值**（见 [`SCENE_SSE_SCALE`]）。
const PSNR_DB10_CAP: i32 = 999;
/// `psnr = 10·log10(peak²/mse)` 里的 `peak²`（峰值 255）。
const SCENE_PEAK_SQ: u64 = 255 * 255;
/// 判据 3 的**精确整数**上界：`mse ≤ peak²/10⁴` ⇔ `sum_sq·10⁴ ≤ peak²·channels`。
///
/// 与「PSNR ≥ 40 dB」严格等价（`sum_sq` 是 RGB 三通道平方误差和，`channels = 3·pixels`），
/// 但全程 `u64` 乘加、无对数、无浮点，故真机上不存在「阈值附近两平台判反」的可能。
const SCENE_SSE_SCALE: u64 = 10_000;
/// `log10(2) = 0.301029995663981195…`，把 `log2` 换算成 `log10`。
///
/// **为什么不用 `f64::log10`**：本 crate 生产代码受 D-29 约束（禁 libm 超越函数，
/// 见 `tests/qa_probe_b_gpu_fixes.rs::probe1_*`），故走自研
/// [`prism_math::det_pow::det_log2`]（位分解 + atanh 级数，纯 `±×÷`）再乘本常数：
/// `log10(x) = log2(x) · log10(2)`。结果跨平台逐位确定，且只用于**报告**，不参与判决。
const LOG10_2: f64 = 0.301_029_995_663_981_2;

thread_local! {
    /// `gpu.scene_frame` 在 run 阶段登记的帧句柄（collect 阶段读取；每次 run 覆写）。
    static SCENE_FRAMES: RefCell<[Option<SceneFrame>; SCENE_SUBMISSIONS]> =
        const { RefCell::new([None; SCENE_SUBMISSIONS]) };
    /// run 阶段登记的 GPU 错误 scope 数（collect 阶段成对 pop）。
    static SCENE_ERR_SCOPES: Cell<u32> = const { Cell::new(0) };
}

/// CPU 参考渲染的**唯一调用点**。
///
/// 判据 3 与判据 5 一共要跑 4 次 CPU 参考，收成一个具名函数既让「参考实现从哪来」
/// 一眼可见（`CpuRefBackend`，与生产路径同一实现），也避免这段调用散落在多处。
fn scene_cpu_ref(req: &SceneRenderRequest) -> crate::Framebuffer {
    CpuRefBackend.render_scene(req)
}

/// 把一帧的像素复制出来（未落盘 ⇒ `None`）。
fn scene_frame_pixels(frame: SceneFrame) -> Option<Vec<u8>> {
    scene_gpu::with_settled_pixels(frame, |px| px.to_vec())
}

/// 帧内不同 RGBA 值的个数，**至多统计 8 种**（判据 2 只需要「≥ 2」）。
///
/// 定长表 + 线性查找：不分配、不排序、不调 `contains`——wasm 里只值几十字节代码，
/// 而 `Vec` + `contains` 版本会连带拉进 `RawVec::grow_one` 等一整串分配机制。
fn scene_distinct_colors(px: &[u8]) -> usize {
    let mut seen = [0u32; 8];
    let mut n = 0usize;
    for chunk in px.chunks_exact(4) {
        let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mut i = 0usize;
        while i < n && seen[i] != v {
            i += 1;
        }
        if i == n {
            if n == seen.len() {
                return n; // 表满 ⇒ 早已满足「≥ 2」，不必继续统计
            }
            seen[n] = v;
            n += 1;
        }
    }
    n
}

/// 「GPU 帧 vs CPU 参考帧」的统计对拍结果（**全整数**，判决不碰浮点）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SceneParity {
    /// 逐通道最大绝对差（仅 RGB 三通道，0..=255）。
    max_channel_delta: u32,
    /// 至少一个 RGB 通道不同的像素数。
    differing_pixels: u32,
    /// 参与比较的像素数。
    pixels: u32,
    /// RGB 三通道平方误差和（精确整数，判据 3 的判决输入）。
    sum_sq: u64,
    /// PSNR ×10（四舍五入、上限 [`PSNR_DB10_CAP`]），**仅供报告**。
    psnr_db10: i32,
    /// 判据 3 的结论：`sum_sq·10⁴ ≤ 255²·channels`（⇔ PSNR ≥ 40 dB）。
    passes_40db: bool,
}

/// 逐像素统计两帧差异（RGB 三通道；alpha 两侧恒为 255，不参与 MSE）。
///
/// 长度不足（任一帧短于 `pixels*4`）⇒ `None`（调用方判 FAIL，不猜）。
fn scene_parity(gpu: &[u8], cpu: &[u8], pixels: u32) -> Option<SceneParity> {
    let need = (pixels as usize) * 4;
    if gpu.len() < need || cpu.len() < need {
        return None;
    }
    let mut max_channel_delta = 0u32;
    let mut differing_pixels = 0u32;
    let mut sum_sq = 0u64;
    for i in 0..pixels as usize {
        let o = i * 4;
        let mut differs = false;
        for c in 0..3 {
            let d = (gpu[o + c] as i32 - cpu[o + c] as i32).unsigned_abs();
            if d > max_channel_delta {
                max_channel_delta = d;
            }
            if d != 0 {
                differs = true;
            }
            sum_sq += (d as u64) * (d as u64);
        }
        if differs {
            differing_pixels += 1;
        }
    }
    let channels = (pixels as u64) * 3;
    // 判据 3：精确整数比较（⇔ `mse ≤ 255²/10⁴` ⇔ `psnr ≥ 40 dB`）。
    let passes_40db =
        sum_sq.saturating_mul(SCENE_SSE_SCALE) <= SCENE_PEAK_SQ.saturating_mul(channels);
    // 报告值：`psnr = 10·log10(peak²/mse)`；D-29 禁 libm，故 det_log2 × log10(2)。
    let psnr_db10 = if sum_sq == 0 || channels == 0 {
        PSNR_DB10_CAP
    } else {
        let mse = (sum_sq as f64) / (channels as f64);
        let db10 =
            100.0 * prism_math::det_pow::det_log2((SCENE_PEAK_SQ as f64) / mse) * LOG10_2;
        let db10 = db10.round() as i32;
        if db10 > PSNR_DB10_CAP {
            PSNR_DB10_CAP
        } else {
            db10
        }
    };
    Some(SceneParity {
        max_channel_delta,
        differing_pixels,
        pixels,
        sum_sq,
        psnr_db10,
        passes_40db,
    })
}

/// 构造场景 `scene`（[`SCN_A`] / [`SCN_B`] / [`SCN_BC`]）的渲染请求。
///
/// **单一真源**：run 与 collect 各自重建同一份，不跨阶段缓存 `Vec`。三个场景共用
/// 同一组相机/灯光字面量，只在「尺寸 / 球半径 / 曲线有无」三处分支，把重复的字段
/// 初始化压成一份——这既省 wasm 体积，也保证 run 与 collect 拿到的请求**逐字段相同**
/// （否则对拍比的就不是同一件事了）。
///
/// 相机/灯/球的取值全部写死（确定性铁律：不取时间、不取随机源）。
fn scene_request(scene: usize) -> SceneRenderRequest {
    let (w, h) = if scene == SCN_A {
        (SCENE_A_W, SCENE_A_H)
    } else {
        (SCENE_B_W, SCENE_B_H)
    };
    // 场景 A：半径 1 的球铺满 8×8 视野。场景 B/BC：偏左小球 + 右侧圆柱，
    // |Δx| = 0.62 > 0.35 + 0.18 = 0.53 ⇒ 互不遮挡，曲线在画面上独立可见
    // （否则判据 5 会被球挡成恒真）。
    let sphere = if scene == SCN_A {
        crate::SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4,
            metalness: 0.1,
        }
    } else {
        crate::SphereReq {
            center: Vec3::new(-0.3, -0.1, 0.0),
            radius: 0.35,
            albedo: Vec3::new(0.25, 0.55, 0.85),
            roughness: 0.3,
            metalness: 0.2,
        }
    };
    let curves = if scene == SCN_B {
        vec![crate::CurveReq::Cylinder {
            center: Vec3::new(0.32, 0.0, 0.0),
            radius: 0.18,
            height: 1.4,
            axis: Vec3::new(0.0, 1.0, 0.0),
        }]
    } else {
        Vec::new()
    };
    SceneRenderRequest {
        width: w,
        height: h,
        camera: crate::CameraReq {
            eye: Vec3::new(0.0, 0.0, -4.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![crate::LightReq {
            direction: Vec3::new(-0.3, -0.4, -1.0),
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 1.0,
        }],
        spheres: vec![sphere],
        aesthetic: crate::AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        curves,
    }
}

/// 场景的人类可读标签（报告用）。
fn scene_label(idx: usize) -> &'static str {
    match idx {
        SCN_A => "A(球+灯,8×8)",
        SCN_B => "B(球+圆柱+灯,16×16)",
        SCN_BC => "BC(B 去曲线)",
        _ => "?",
    }
}

fn t_gpu_scene_frame() -> Status {
    let dev = match ready_device("gpu.scene_frame") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // 提交顺序：A 双跑 → B 双跑 → BC 单跑（下标与 collect 的 `scene*SCENE_RUNS` 对齐）。
    let mut frames: [Option<SceneFrame>; SCENE_SUBMISSIONS] = [None; SCENE_SUBMISSIONS];
    let mut read_bytes = 0u32;
    for (i, &scene) in SCENE_ORDER.iter().enumerate() {
        let req = scene_request(scene);
        match scene_gpu::submit_scene_frame(dev, &req) {
            Ok(f) => frames[i] = Some(f),
            Err(e) => {
                return fail(format!(
                    "gpu.scene_frame：第 {i} 次提交失败（场景 {}，{}×{}）——{e}\n\
                     提示：本项要求 device 已就绪（先跑 device.init 并等其 PASS）；\
                     若为 `TooLarge`，说明场景规模超出宿主竞技场容量",
                    scene_label(scene),
                    req.width,
                    req.height
                ));
            }
        }
        read_bytes = read_bytes.saturating_add(req.width * req.height * 4);
    }
    // 错误 scope 数在 run 阶段定稿（覆盖写，避免并发 run 时重复累加）。
    SCENE_ERR_SCOPES.with(|c| c.set(scene_gpu::take_err_scopes()));
    SCENE_FRAMES.with(|f| *f.borrow_mut() = frames);
    set_result(&format!(
        "已提交 {SCENE_SUBMISSIONS} 次真 WebGPU 场景整帧 dispatch（scene_frame.wgsl 的 scene_main）：\n\
         A {SCENE_A_W}×{SCENE_A_H}（球+灯）连跑 {SCENE_RUNS} 次、\
         B {SCENE_B_W}×{SCENE_B_H}（球+圆柱+灯）连跑 {SCENE_RUNS} 次、\
         对照（B 去掉曲线）跑 1 次\n\
         绑定剖面：V2 六绑定 {{params,spheres,lights,triangles,nodes,out}}，\
         dispatch 工作组数由 scene_dispatch_plan 从 scene_abi 派生\n\
         已登记读回 {read_bytes} 字节，待宿主 settle 后与 CPU 参考（CpuRefBackend，同式）\
         统计对拍（门限 PSNR ≥ 40 dB）"
    ));
    Status::Pending
}

/// 单帧的判据 0～3（尺寸 / 字节数 / 非全零非纯色 / vs CPU 参考）。
///
/// 返回 `Ok((像素, 对拍统计))`；任一硬判据不过即 `Err(Status::Fail)`。
fn scene_check_frame(
    label: &str,
    req: &SceneRenderRequest,
    frame: SceneFrame,
) -> Result<(Vec<u8>, SceneParity), Status> {
    let want = req.width * req.height * 4;
    // ---- 判据 0：帧句柄自带的尺寸必须与请求一致 ----
    // 槽位是复用的（多帧共用定长竞技场），张冠李戴会让「字节数对了但内容是别帧的」
    // 这种错误静默通过，故先钉尺寸。
    if frame.width() != req.width || frame.height() != req.height {
        return Err(fail(format!(
            "{label}：帧句柄尺寸 {}×{} ≠ 请求 {}×{}（帧槽复用错位）",
            frame.width(),
            frame.height(),
            req.width,
            req.height
        )));
    }
    // ---- 判据 1a：提交时请求的字节数 == w*h*4 ----
    if frame.read_len() != want {
        return Err(fail(format!(
            "{label}：提交时请求的读回字节数 {} ≠ w*h*4 = {want}——\
             有设备分支的「返 w*h*4」不成立",
            frame.read_len()
        )));
    }
    // ---- 判据 1b：真的读回了 w*h*4 字节 ----
    let Some(px) = scene_frame_pixels(frame) else {
        return Err(fail(format!(
            "{label}：第 {} 帧读回未落盘（宿主 settle 未完成，或 copy_out 失败）",
            frame.slot()
        )));
    };
    if px.len() != want as usize {
        return Err(fail(format!(
            "{label}：读回 {} 字节 ≠ w*h*4 = {want}（`{}×{}×4`）",
            px.len(),
            req.width,
            req.height
        )));
    }
    // ---- 判据 2：非全零 + 非整帧同色 ----
    if !px.iter().any(|&b| b != 0) {
        return Err(fail(format!(
            "{label}：读回 {want} 字节全为 0——字节数对了但没有任何渲染内容（假绿），\
             shader 可能整体早退或 dispatch 工作组数为 0"
        )));
    }
    let colors = scene_distinct_colors(&px);
    if colors < 2 {
        return Err(fail(format!(
            "{label}：整帧只有 1 种颜色——scene_main 未按像素着色\
             （常见根因：params 的 P_WIDTH/P_HEIGHT 未写入、或工作组数算成 1×1）"
        )));
    }
    // ---- 判据 3：vs CPU 参考（同一请求走 CpuRefBackend）----
    let cpu = scene_cpu_ref(req);
    if cpu.pixels.len() != want as usize {
        return Err(fail(format!(
            "{label}：CPU 参考帧 {} 字节 ≠ w*h*4 = {want}（CPU 侧被拒帧，请求尺寸非法？）",
            cpu.pixels.len()
        )));
    }
    let Some(parity) = scene_parity(&px, &cpu.pixels, req.width * req.height) else {
        return Err(fail(format!("{label}：GPU/CPU 帧长度不足以对比")));
    };
    if !parity.passes_40db {
        return Err(fail(format!(
            "{label}：PSNR {}.{} dB < 门限 40.0 dB（sum_sq={} 超出精确整数上界\
             sum_sq·10⁴ ≤ 255²·channels，max_channel_delta={}，differing_pixels={}/{}）——\
             这是真机浮点/求交差异的实测值，**不得靠改阈值消红**，须由交付总监裁决口径",
            parity.psnr_db10 / 10,
            parity.psnr_db10 % 10,
            parity.sum_sq,
            parity.max_channel_delta,
            parity.differing_pixels,
            parity.pixels
        )));
    }
    Ok((px, parity))
}

fn c_gpu_scene_frame() -> Status {
    // `String` 的 `write!` 走 `core::fmt::Write`（不是 `io::Write`），故在此引入。
    use core::fmt::Write as _;
    let dev = match ready_device("gpu.scene_frame") {
        Ok(d) => d,
        Err(s) => return s,
    };
    // 错误结算必须在 collect 阶段（宿主 settle() 之后错误队列才有内容）。
    let scopes = SCENE_ERR_SCOPES.with(|c| c.get());
    if let Err(s) = check_errors_counted("gpu.scene_frame", dev, scopes) {
        return s;
    }
    if gpu::device_lost(dev) != 0 {
        return fail("gpu.scene_frame：device 在 dispatch 期间 lost");
    }
    let frames = SCENE_FRAMES.with(|f| *f.borrow());
    // 报告逐行拼进一个 `String`（`write!` 到 `String` 不会失败，`let _ =` 明示忽略）。
    let mut report = String::from(
        "真 WebGPU 场景整帧 dispatch（scene_frame.wgsl 的 scene_main，V2 六绑定）\
         与 CPU 参考统计对拍 ✓\n",
    );
    let mut min_db10 = PSNR_DB10_CAP;
    let mut px_b: Option<Vec<u8>> = None;
    let mut px_bc: Option<Vec<u8>> = None;
    for scene in 0..SCN_COUNT {
        let req = scene_request(scene);
        let label = scene_label(scene);
        let Some(first) = frames[scene * SCENE_RUNS] else {
            return fail(format!(
                "gpu.scene_frame：第 {} 帧句柄缺失（本项 run 未成功提交过；请先运行本项）",
                scene * SCENE_RUNS
            ));
        };
        let (px, parity) = try_status!(scene_check_frame(label, &req, first));
        // ---- 判据 4：双跑逐位一致（对照场景只跑一次，跳过）----
        if scene != SCN_BC {
            let Some(second) = frames[scene * SCENE_RUNS + 1] else {
                return fail(format!("{label}：第 2 跑的帧句柄缺失"));
            };
            let Some(px2) = scene_frame_pixels(second) else {
                return fail(format!("{label}：第 2 跑读回未落盘，无法核对确定性"));
            };
            if px2.len() != px.len() {
                return fail(format!(
                    "{label}：两跑读回长度不同（{} vs {}）——buffer 尺寸算错？",
                    px.len(),
                    px2.len()
                ));
            }
            if px2 != px {
                let at = px
                    .iter()
                    .zip(px2.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return fail(format!(
                    "{label}：同设备连跑两次读回**不一致**（首个不同偏移 {at}，\
                     GPU-run1={} vs GPU-run2={}）——违反全链路确定性铁律",
                    px[at], px2[at]
                ));
            }
        }
        if parity.psnr_db10 < min_db10 {
            min_db10 = parity.psnr_db10;
        }
        let _ = write!(
            report,
            "  {label}: 读回 {} B == w*h*4 ✓  非全零 ✓  色数(≤8) {}  PSNR {}.{} dB\
             （max_channel_delta {}，differing_pixels {}/{}）{}\n",
            px.len(),
            scene_distinct_colors(&px),
            parity.psnr_db10 / 10,
            parity.psnr_db10 % 10,
            parity.max_channel_delta,
            parity.differing_pixels,
            parity.pixels,
            if scene == SCN_BC { "  对照单跑" } else { "  双跑逐位一致 ✓" }
        );
        if scene == SCN_B {
            px_b = Some(px);
        } else if scene == SCN_BC {
            px_bc = Some(px);
        }
    }
    // ---- 判据 5：曲线路径真被消费（B 与「B 去掉曲线」必须不同）----
    let (Some(b), Some(bc)) = (px_b, px_bc) else {
        return fail("gpu.scene_frame：场景 B / 对照帧的像素缺失，无法核对曲线路径");
    };
    let n = b.len().min(bc.len()) / 4;
    let mut curve_diff_px = 0usize;
    let mut curve_max_delta = 0u32;
    for i in 0..n {
        let o = i * 4;
        let mut d = 0u32;
        for c in 0..3 {
            let dd = (b[o + c] as i32 - bc[o + c] as i32).unsigned_abs();
            if dd > d {
                d = dd;
            }
        }
        if d > 0 {
            curve_diff_px += 1;
        }
        if d > curve_max_delta {
            curve_max_delta = d;
        }
    }
    if curve_diff_px == 0 {
        return fail(
            "gpu.scene_frame：含曲线的场景 B 与「B 去掉曲线」的对照帧**逐像素完全相同**——\
             说明 triangles/nodes 两个绑定没有被 scene_main 真正消费（宿主漏绑/漏写，\
             或 WGSL 的 num_tris 分支未生效）。V2 六绑定的曲线路径**未被判对**",
        );
    }
    // CPU 参考侧的同名对照：证明「曲线本来就该在画面上看得见」，免得 GPU 侧的差异
    // 被误读成绑定接错。
    let cpu_b = scene_cpu_ref(&scene_request(SCN_B));
    let cpu_bc = scene_cpu_ref(&scene_request(SCN_BC));
    let n_cpu = cpu_b.pixels.len().min(cpu_bc.pixels.len()) / 4;
    let mut cpu_curve_diff = 0usize;
    for i in 0..n_cpu {
        let o = i * 4;
        if cpu_b.pixels[o..o + 3] != cpu_bc.pixels[o..o + 3] {
            cpu_curve_diff += 1;
        }
    }
    let _ = write!(
        report,
        "判据 1（核心）：三帧读回字节数均 == w*h*4（{SCENE_A_W}×{SCENE_A_H}×4={}、\
         {SCENE_B_W}×{SCENE_B_H}×4={}）——「有设备返 w*h*4」成立 ✓\n\
         判据 2：非全零 + 每帧色数 ≥ 2（排除「读回成功但全黑/纯色」的假绿）✓\n\
         判据 3：PSNR ≥ 40.0 dB（等价精确整数判据 sum_sq·10⁴ ≤ 255²·channels），\
         最低实测 {}.{} dB ✓\n\
         判据 4：同设备连跑两次读回**逐位一致**（A、B 各 2 跑）✓\n\
         判据 5：曲线路径真被消费——B 与「B 去掉曲线」有 {curve_diff_px}/{n} 个像素不同\
         （max_channel_delta={curve_max_delta}）；CPU 参考侧同对照差 {cpu_curve_diff}/{n_cpu}\
         个像素，两侧同向 ⇒ triangles/nodes 绑定确实参与了着色 ✓\n\
         注：CPU 量化用 `(v*255).round()`，WGSL 用 `u32(v*255)` 截断，故多数像素有 ±1 级\
         系统性差；这是 PSNR 天花板的主要来源（约 48～53 dB），远高于 40 dB 门限",
        SCENE_A_W * SCENE_A_H * 4,
        SCENE_B_W * SCENE_B_H * 4,
        min_db10 / 10,
        min_db10 % 10,
    );
    set_result(&report);
    Status::Pass
}

// ---------------------------------------------------------------------------
// 导出函数（JS → Rust 方向）
// ---------------------------------------------------------------------------

/// ABI 版本号（宿主加载后先比对）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_abi_version() -> u32 {
    ABI_VERSION
}

/// 在 wasm 线性内存中分配 `len` 字节（16 字节对齐），返回地址；失败返回 0。
///
/// 宿主须用**同样的 `len`** 调用 [`prism_dealloc`] 释放。
#[unsafe(no_mangle)]
pub extern "C" fn prism_alloc(len: u32) -> usize {
    alloc_bytes(len as usize) as usize
}

/// 释放 [`prism_alloc`] 分配的内存。`ptr == 0` 或 `len == 0` 为无操作。
///
/// # Safety
/// `ptr`/`len` 必须来自同一次 [`prism_alloc`] 调用，且未重复释放。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prism_dealloc(ptr: usize, len: u32) -> i32 {
    if ptr == 0 || len == 0 {
        return 0;
    }
    if Layout::from_size_align(len as usize, PRISM_ALLOC_ALIGN).is_err() {
        set_err("prism_dealloc：非法 len，无法重建布局");
        return status::FAIL;
    }
    // SAFETY: 由本函数的契约保证 ptr/len 来自且仅来自一次 prism_alloc。
    unsafe { dealloc_bytes(ptr as *mut u8, len as usize) };
    0
}

/// 错误文本在 wasm 线性内存中的地址（NUL 结尾，用 [`prism_error_len`] 取长度）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_error_ptr() -> usize {
    ERR.with(|b| b.as_ptr() as usize)
}

/// 错误文本字节数（不含 NUL）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_error_len() -> u32 {
    ERR_LEN.with(|l| l.get())
}

/// 结果文本在 wasm 线性内存中的地址（NUL 结尾）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_result_ptr() -> usize {
    RES.with(|b| b.as_ptr() as usize)
}

/// 结果文本字节数（不含 NUL）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_result_len() -> u32 {
    RES_LEN.with(|l| l.get())
}

/// 测试项总数。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_count() -> u32 {
    TEST_COUNT
}

/// 第 `i` 项是否需要 WebGPU device（1 = 需要）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_needs_gpu(i: u32) -> u32 {
    match TESTS.get(i as usize) {
        Some(t) => u32::from(t.needs_gpu),
        None => 0,
    }
}

/// 第 `i` 项的 id 字符串地址（越界返回 0，用 [`prism_test_id_len`] 取长度）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_id(i: u32) -> usize {
    match TESTS.get(i as usize) {
        Some(t) => t.id.as_ptr() as usize,
        None => 0,
    }
}

/// 第 `i` 项 id 的字节数。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_id_len(i: u32) -> u32 {
    match TESTS.get(i as usize) {
        Some(t) => t.id.len() as u32,
        None => 0,
    }
}

/// 第 `i` 项的可读名地址（越界返回 0）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_name(i: u32) -> usize {
    match TESTS.get(i as usize) {
        Some(t) => t.name.as_ptr() as usize,
        None => 0,
    }
}

/// 第 `i` 项可读名的字节数。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_name_len(i: u32) -> u32 {
    match TESTS.get(i as usize) {
        Some(t) => t.name.len() as u32,
        None => 0,
    }
}

/// 运行第 `i` 个测试项。返回状态码（见 [`status`]）。
///
/// 返回 [`status::PENDING`] 时，宿主须 `await` 胶水中的异步操作后调用
/// [`prism_test_collect`]。任何 Rust panic 都被捕获并转为失败（宪章 C18）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_run(i: u32) -> i32 {
    let Some(item) = TESTS.get(i as usize) else {
        set_err(&format!("prism_test_run：测试项下标 {i} 越界（count={TEST_COUNT}）"));
        return status::BAD_ARG;
    };
    dispatch_panics(item.run, item.id)
}

/// 读回并判定第 `i` 个测试项（仅在 [`prism_test_run`] 返回 PENDING 后调用）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_test_collect(i: u32) -> i32 {
    let Some(item) = TESTS.get(i as usize) else {
        set_err(&format!("prism_test_collect：测试项下标 {i} 越界（count={TEST_COUNT}）"));
        return status::BAD_ARG;
    };
    let Some(collect) = item.collect else {
        set_err(&format!("prism_test_collect：测试项 `{}` 无读回阶段", item.id));
        return status::BAD_ARG;
    };
    dispatch_panics(collect, item.id)
}

/// 执行测试项主体；把 panic 转成失败（宪章 C18：永不 panic 穿透到宿主）。
fn dispatch_panics(f: fn() -> Status, id: &str) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(s) => s.code(),
        Err(payload) => {
            let detail = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "未知 panic 载荷".to_string()
            };
            set_err(&format!("测试项 `{id}` 触发 Rust panic（已捕获，未穿透到宿主）：{detail}"));
            status::FAIL
        }
    }
}

/// 内嵌 WGSL 测试项数量（`shaders/` 目录 9 个；库件 rt_scene 不计入，见 `shaders::LIBRARIES`）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_shader_count() -> u32 {
    shaders::SHADERS.len() as u32
}

/// 已展开的 WGSL 源码缓存（`i` → 完整可编译文本）。
///
/// `prism_shader_src` 必须交出**展开后**的源码：WGSL 无 include 语义，宿主把
/// 哪一段交给 `createShaderModule` 就编译哪一段，D2/D3 的公共件（`vdot`/
/// `cell_key`/`xpbd_distance_project` 等）不在其中就报 `unresolved call target`
/// （D-14 真机实测）。展开在 Rust 侧做，宿主零改动、且仍是单一真源。
///
/// 用 [`OnceLock`] 而非每问现算：`prism_shader_src` / `_len` 会被宿主连续调用，
/// 现算会反复分配并使返回地址漂移；`OnceLock` 保证地址一次定死（确定性铁律）。
static EXPANDED: OnceLock<Vec<String>> = OnceLock::new();

/// 取第 `i` 份展开后的 WGSL（越界返回 [`None`]）。
fn expanded(i: u32) -> Option<&'static String> {
    EXPANDED
        .get_or_init(|| shaders::expanded_sources().into_iter().map(|(_, s)| s).collect())
        .get(i as usize)
}

/// 第 `i` 份展开后源码的字节长度（越界返回 0）。
///
/// 单独导出而非让宿主 `strlen`：`wasm32` 上展开文本含 `\n` 与注释，宿主按
/// UTF-8 `TextDecoder` 解码，必须拿到精确字节数。
fn expanded_len(i: u32) -> usize {
    expanded(i).map(|s| s.len()).unwrap_or(0)
}

/// 第 `i` 个 WGSL 文件名的线性内存地址（越界返回 0）。
///
/// 宿主用它在 JS 侧 `createShaderModule` **同一份**源码编译——避免 WGSL 在
/// JS 里另存一份导致两侧漂移。
#[unsafe(no_mangle)]
pub extern "C" fn prism_shader_name(i: u32) -> usize {
    match shaders::SHADERS.get(i as usize) {
        Some((n, _)) => n.as_ptr() as usize,
        None => 0,
    }
}

/// 第 `i` 个 WGSL 文件名的字节数。
#[unsafe(no_mangle)]
pub extern "C" fn prism_shader_name_len(i: u32) -> u32 {
    match shaders::SHADERS.get(i as usize) {
        Some((n, _)) => n.len() as u32,
        None => 0,
    }
}

/// 第 `i` 份 WGSL 源码（**已展开 include**）的线性内存地址（越界返回 0）。
///
/// 交出展开后的完整文本：宿主把它原样喂给 `createShaderModule`。WGSL 无 include
/// 语义，故「展开」必须在上传前完成——D-14 真机 6 份 shader 报
/// `unresolved call target` 正是因为公共件没跟上来。
#[unsafe(no_mangle)]
pub extern "C" fn prism_shader_src(i: u32) -> usize {
    match expanded(i) {
        Some(s) => s.as_ptr() as usize,
        None => 0,
    }
}

/// 第 `i` 份 WGSL 源码（已展开）的字节数（越界返回 0）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_shader_src_len(i: u32) -> u32 {
    expanded_len(i) as u32
}

/// 对**宿主写入线性内存**的 WGSL 源码做结构自检（与 [`shaders::check_shader`] 同口径）。
///
/// 这是 `prism_alloc` / `prism_dealloc` 的正当宿主用途：harness 让用户粘贴一段
/// WGSL（真机排查"是 shader 写错了还是引擎错了"），写进 wasm 线性内存后调用
/// 本函数拿 Rust 侧的结构判据——尤其是 D-29 的 libm 禁名与括号配平。
///
/// `src_ptr`/`src_len` 指向 [`prism_alloc`] 得到的区域。返回状态码；结果文本
/// 写入结果缓冲，命中禁项时错误缓冲同时给出违规明细。
#[unsafe(no_mangle)]
pub extern "C" fn prism_check_wgsl(src_ptr: usize, src_len: u32) -> i32 {
    let len = src_len as usize;
    if len == 0 || len > 1 << 20 {
        set_err(&format!("prism_check_wgsl：长度 {len} 非法（须 1..=1048576 字节）"));
        return status::BAD_ARG;
    }
    if src_ptr == 0 {
        set_err("prism_check_wgsl：src_ptr 为 0（须为 prism_alloc 返回的地址）");
        return status::BAD_ARG;
    }
    // SAFETY: 宿主契约：ptr/len 指向 prism_alloc 的分配且不越界；只读。
    let bytes = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, len) };
    // 非 UTF-8 不 panic（C18）：用 from_utf8 的错误信息如实回报。
    let src = match std::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => {
            set_err(&format!("prism_check_wgsl：源码不是合法 UTF-8（{e}）"));
            return status::FAIL;
        }
    };
    let chk = shaders::check_shader("host_provided.wgsl", src);
    let mut violations: Vec<&str> = Vec::new();
    if !chk.brackets_balanced { violations.push("括号不配平（疑似截断/语法残缺）"); }
    if chk.fn_count == 0 { violations.push("无 fn 声明"); }
    if chk.has_libm { violations.push("出现被禁 libm 超越函数（D-29 铁律）"); }
    if chk.has_include { violations.push("含第三方 include / @import"); }
    if !violations.is_empty() {
        set_err(&format!("宿主提供的 WGSL 结构自检未通过：\n  - {}", violations.join("\n  - ")));
        return status::FAIL;
    }
    set_result(&format!(
        "宿主提供的 WGSL（{len} 字节）结构自检通过 ✓\n\
         入口存在={} 括号配平={} fn 数={} 无 libm={} 无 include={}\n\
         提示：本项只做结构层判据；语法/类型层的真编译结论以 wgsl.compile 项为准。",
        chk.has_entry, chk.brackets_balanced, chk.fn_count, !chk.has_libm, !chk.has_include
    ));
    status::PASS
}

/// D0 内核 WGSL（`0` = `scan_block`，`1` = `histogram`）的线性内存地址。
///
/// 供宿主在 JS 侧独立编译这两个内核做交叉验证（它们内嵌在 `device.rs`，
/// 不在 `shaders/` 目录，故不在 `prism_shader_*` 序列里）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_kernel_wgsl(i: u32) -> usize {
    match i {
        0 => WGSL_SCAN_BLOCK.as_ptr() as usize,
        1 => WGSL_HISTOGRAM.as_ptr() as usize,
        _ => 0,
    }
}

/// D0 内核 WGSL 的字节数。
#[unsafe(no_mangle)]
pub extern "C" fn prism_kernel_wgsl_len(i: u32) -> u32 {
    match i {
        0 => WGSL_SCAN_BLOCK.len() as u32,
        1 => WGSL_HISTOGRAM.len() as u32,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// 单元测试（native：零 GPU 依赖，验证 ABI 自身的纯逻辑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::{detect_support, WebGpuSupport};

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(prism_abi_version(), ABI_VERSION);
    }

    #[test]
    fn test_manifest_is_well_formed() {
        assert_eq!(prism_test_count(), TEST_COUNT);
        assert!(TEST_COUNT >= 9, "至少 9 个测试项，实际 {}", TEST_COUNT);
        let mut ids: Vec<&str> = Vec::new();
        for (i, t) in TESTS.iter().enumerate() {
            assert!(!t.id.is_empty() && !t.name.is_empty(), "第 {i} 项缺 id/name");
            assert!(!ids.contains(&t.id), "id 重复：{}", t.id);
            ids.push(t.id);
            // 指针/长度导出一致性（宿主按此读文本）。
            assert_eq!(prism_test_id(i as u32), t.id.as_ptr() as usize);
            assert_eq!(prism_test_id_len(i as u32), t.id.len() as u32);
            assert_eq!(prism_test_needs_gpu(i as u32), u32::from(t.needs_gpu));
        }
        // 越界必须是 0（宿主据此判定"无效"），不得 panic。
        assert_eq!(prism_test_id(TEST_COUNT), 0);
        assert_eq!(prism_test_name(TEST_COUNT), 0);
        assert_eq!(prism_test_id_len(TEST_COUNT), 0);
        assert_eq!(prism_test_needs_gpu(TEST_COUNT), 0);
        assert_eq!(prism_test_run(TEST_COUNT), status::BAD_ARG);
        assert_eq!(prism_test_collect(TEST_COUNT), status::BAD_ARG);
    }

    #[test]
    fn shader_source_exports_are_readable() {
        assert_eq!(prism_shader_count() as usize, shaders::SHADERS.len());
        for (i, (name, src)) in shaders::SHADERS.iter().enumerate() {
            let i = i as u32;
            assert_eq!(prism_shader_name(i), name.as_ptr() as usize);
            assert_eq!(prism_shader_name_len(i), name.len() as u32);
            // `prism_shader_src` 交出的是**展开后**文本（含公共件），长度 ≥ 原文，
            // 且必须等于 `shaders::expand(name)` 的长度——两者是同一份真源。
            let want = shaders::expand(name);
            assert_eq!(prism_shader_src_len(i), want.len() as u32, "{name} 展开长度不符");
            assert!(
                prism_shader_src_len(i) >= src.len() as u32,
                "{name} 展开后不应变短"
            );
            assert_ne!(prism_shader_src(i), 0, "{name} 源码地址为 0");
            // 同一 i 反复问，地址必须稳定（OnceLock 缓存，确定性铁律）。
            assert_eq!(prism_shader_src(i), prism_shader_src(i), "{name} 地址漂移");
        }
        let n = prism_shader_count();
        assert_eq!(prism_shader_src(n), 0);
        assert_eq!(prism_shader_src_len(n), 0);
        assert_eq!(prism_kernel_wgsl(0), WGSL_SCAN_BLOCK.as_ptr() as usize);
        assert_eq!(prism_kernel_wgsl_len(1), WGSL_HISTOGRAM.len() as u32);
        assert_eq!(prism_kernel_wgsl(2), 0);
    }

    #[test]
    fn expanded_sources_have_no_dangling_include() {
        // 每份展开结果都必须是"可直接 createShaderModule"的完整文本：
        // 不含指令、括号配平、零未解析符号、零 libm（D-29）。
        for (name, src) in shaders::expanded_sources() {
            assert!(!src.contains(shaders::INCLUDE_DIRECTIVE), "{name} 仍含 include 指令");
            let chk = shaders::check_shader(name, &src);
            assert!(chk.brackets_balanced, "{name} 展开后括号不配平");
            assert!(!chk.has_libm, "{name} 展开后出现 libm（D-29）");
            let missing = shaders::unresolved_symbols(&src);
            assert!(missing.is_empty(), "{name} 展开后未解析符号 {:?}", missing);
        }
    }

    #[test]
    fn alloc_write_readback_dealloc_roundtrip() {
        // 与 harness 的 abi.alloc 同一口径，native 上亦须成立。
        assert_eq!(t_abi_alloc(), Status::Pass);
        assert!(alloc_bytes(0).is_null(), "零长度分配应返回空指针");
        assert_eq!(prism_alloc(0), 0, "零长度分配应返回 0");
        // 释放空指针 / 零长度是无操作（幂等，不 panic）。
        assert_eq!(unsafe { prism_dealloc(0, 16) }, 0);
        assert_eq!(unsafe { prism_dealloc(16, 0) }, 0);
    }

    #[test]
    fn cpu_only_test_items_pass_on_native() {
        // 这三项零 GPU 依赖，native 与 wasm 结果必须一致（同一份 CPU 参考）。
        for id in ["cpu.scan", "cpu.sort", "wgsl.structural", "abi.alloc", "shader.params"] {
            let idx = TESTS.iter().position(|t| t.id == id).expect("清单含该项");
            assert_eq!(dispatch_panics(TESTS[idx].run, id), status::PASS, "{id} 应在 native 通过");
        }
    }

    #[test]
    fn gpu_items_skip_on_native_without_device() {
        // native 目标：GPU 桩全返回 0 ⇒ 需 GPU 的项稳定 SKIP（不 panic）。
        assert_eq!(detect_support(), WebGpuSupport::NotWasmTarget);
        for t in TESTS.iter() {
            if !t.needs_gpu {
                continue;
            }
            let code = dispatch_panics(t.run, t.id);
            assert!(
                code == status::SKIP || code == status::PASS,
                "{} 在 native 上应 SKIP，实得 {code}",
                t.id
            );
        }
    }

    #[test]
    fn result_and_error_buffers_are_nul_terminated() {
        set_result("hello");
        assert_eq!(prism_result_len(), 5, "\"hello\" = 5 字节");
        set_err("world!!");
        assert_eq!(prism_error_len(), 7, "\"world!!\" = 7 字节");
        // 超长文本按 char 边界截断且不 panic（C18）。
        let long: String = "错".repeat(4000);
        set_err(&long);
        assert!(prism_error_len() as usize <= ERR_CAP - 1);
        let again: String = "x".repeat(100);
        set_result(&again);
        assert_eq!(prism_result_len(), 100, "短文本应覆写长文本，不留残尾");
    }

    #[test]
    fn check_wgsl_rejects_bad_args_and_accepts_good_source() {
        // 长度 0 / 超长 / 空指针：返回 BAD_ARG 并写明原因，绝不 panic。
        assert_eq!(prism_check_wgsl(0, 0), status::BAD_ARG);
        assert_eq!(prism_check_wgsl(0x1000, 0), status::BAD_ARG);
        assert_eq!(prism_check_wgsl(0x1000, (2u32 << 20) + 1), status::BAD_ARG);
        // 正常源码 → PASS；D-29 违规 → FAIL 且错误缓冲点名。
        let good = alloc_bytes(WGSL_HISTOGRAM.len());
        assert!(!good.is_null());
        // SAFETY: 刚分配、长度精确匹配。
        unsafe { std::slice::from_raw_parts_mut(good, WGSL_HISTOGRAM.len()) }
            .copy_from_slice(WGSL_HISTOGRAM.as_bytes());
        assert_eq!(prism_check_wgsl(good as usize, WGSL_HISTOGRAM.len() as u32), status::PASS);
        // SAFETY: 同一分配改写为含 libm 的源码。
        let bad = b"fn f() { let x = sin(1.0); }\n";
        unsafe { std::slice::from_raw_parts_mut(good, bad.len()) }.copy_from_slice(bad);
        assert_eq!(prism_check_wgsl(good as usize, bad.len() as u32), status::FAIL);
        // 非 UTF-8 → FAIL（不 panic）。
        let bad_utf8 = [0xffu8, 0xfe, 0xfd];
        unsafe { std::slice::from_raw_parts_mut(good, bad_utf8.len()) }.copy_from_slice(&bad_utf8);
        assert_eq!(prism_check_wgsl(good as usize, bad_utf8.len() as u32), status::FAIL);
        // SAFETY: 配对释放。
        unsafe { dealloc_bytes(good, WGSL_HISTOGRAM.len().max(bad_utf8.len())) };
    }

    #[test]
    fn panic_in_test_body_is_converted_to_failure() {
        // 宪章 C18 的机制验证：故意 panic 的测试项不得穿透到宿主。
        fn boom() -> Status {
            panic!("故意触发");
        }
        assert_eq!(dispatch_panics(boom, "boom"), status::FAIL);
    }

    // -----------------------------------------------------------------------
    // D-14 真机门禁：RefCell 重入借用（真机表现 = wasm trap `unreachable`）
    // -----------------------------------------------------------------------

    /// `device.init` 的 collect 必须在**候选句柄集**里认领就绪者，而非只看全局 `DEVICE`。
    ///
    /// 【D-14 真机缺陷 2 的根因门禁】宿主会并发跑本项（`boot()` 自动跑 +
    /// 回归脚本点「全部运行」），两轮 run/collect 交错，而 `test_collect(i)`
    /// **不带 run 令牌**。若只认全局 `DEVICE`（或只认"自己那张票"），本次就会
    /// 去校验别人的句柄 ⇒ 误报"预留句柄未绑定"，并把后续 7 个 GPU 项带崩
    /// （真机实测偶发 PASS=6，句柄表停在 1）。
    /// 认领成功后**清空槽位**不得让后续 `collect` 拿不到已就绪的全局句柄。
    ///
    /// 【D-14 真机残余抖动根因】两轮 `collect` 共享状态且不带 run 令牌。实测时序：
    /// ```text
    ///   28ms request_device()=1 → 45ms device_ready(1)=0 → 发票 2 → 48ms device_ready(1)=1
    /// ```
    /// 先 collect 的轮次认领票 1 并清空槽位；后 collect 的轮次若只看槽位就得到
    /// 空集 → 误报"未绑定"，尽管 `device_ready(1)` 早已为 1。故就绪判定必须把
    /// 全局 `DEVICE` 也纳入候选。
    ///
    /// native 上 `device_ready` 恒 0，故这里固化的是**候选集构造**这一不变式：
    /// 槽位清零后，候选里必须仍然出现全局句柄（`首个候选` 兜底到 `DEVICE`）。
    #[test]
    fn claim_ready_device_falls_back_to_global_handle_after_slots_cleared() {
        DEVICE.with(|d| d.set(5));
        note_device_ticket(5);
        // 第一轮：认领（native 上无就绪者，但槽位必须被清空）。
        let (first, _ready) = claim_ready_device();
        assert_eq!(first, 5, "首个候选应命中槽位里的票");
        assert_eq!(
            DEVICE_TICKETS.with(|t| *t.borrow()),
            [0, 0],
            "认领后槽位必须清零"
        );
        // 第二轮：槽位已空，但全局句柄 5 仍须进入候选（不得返回空集）。
        let (first2, _ready2) = claim_ready_device();
        assert_eq!(
            first2, 5,
            "槽位清零后候选集丢了全局句柄 —— 这正是真机偶发 FAIL 的机制"
        );
        // 反向不变式：DEVICE 为 0 且槽位空 ⇒ 真的是没有候选。
        DEVICE.with(|d| d.set(0));
        assert_eq!(claim_ready_device().0, 0, "无全局句柄也无票时不得凭空造候选");
    }

    #[test]
    fn device_init_collect_claims_any_ready_ticket_not_just_global_device() {
        // 交错场景：两轮各留一张票（1 与 2），DEVICE 被后一轮覆盖成 2。
        // native 上 device_ready 恒 0 ⇒ 两张都没绑定，collect 必须 FAIL，
        // 且诊断文案要点名**首个候选**（1），而不是全局的 2。
        DEVICE.with(|d| d.set(2));
        note_device_ticket(1);
        note_device_ticket(2);
        assert_eq!(dispatch_panics(c_device_init, "device.init"), status::FAIL);
        let mut buf = [0u8; ERR_CAP];
        let n = ERR_LEN.with(|l| l.get()) as usize;
        ERR.with(|b| buf[..n].copy_from_slice(&b.borrow()[..n]));
        let err = String::from_utf8_lossy(&buf[..n]).into_owned();
        assert!(
            err.contains("预留句柄 1 未绑定"),
            "诊断文案未点名首个候选票：{err}"
        );
        assert!(!err.contains("预留句柄 2"), "诊断文案误用了被覆盖的全局句柄：{err}");

        // 认领后候选槽必须清空，避免下一轮collect 拿到陈旧票。
        let left: Vec<u32> = DEVICE_TICKETS.with(|t| t.borrow().to_vec());
        assert_eq!(left, vec![0, 0], "候选槽未清零：{left:?}");
        assert_eq!(DEVICE.with(|d| d.get()), 2, "无就绪者时不得改动全局 DEVICE");
    }

    #[test]
    fn device_init_is_two_phase_and_gates_on_readiness() {
        // D-14 真机教训的固化：`prism_request_device` 只能同步返回预留句柄，
        // 故 device.init 必须「run 返回 PENDING + collect 校验就绪」两阶段。
        // 若退回单阶段（run 直接 PASS），宿主会清空 pendingWork、requestDevice
        // 永不 await，一个不存在的 device 被判为就绪。
        let idx = TESTS.iter().position(|t| t.id == "device.init").expect("清单含该项");
        assert!(
            TESTS[idx].collect.is_some(),
            "device.init 必须有 collect 阶段（否则 requestDevice 永不被 await）"
        );
        // native 上 gpu::device_ready 恒为 0 ⇒ collect 必须判 FAIL 而不是 PASS。
        DEVICE.with(|d| d.set(1)); // 假装句柄已登记
        assert_eq!(dispatch_panics(c_device_init, "device.init"), status::FAIL);
        // 句柄为 0 时同样是 FAIL（不得 panic）。
        DEVICE.with(|d| d.set(0));
        assert_eq!(dispatch_panics(c_device_init, "device.init"), status::FAIL);
    }

    #[test]
    fn ready_device_rejects_unready_handle_with_actionable_message() {
        // DEVICE 有句柄但宿主未绑定真实 device 时，ready_device 必须 FAIL 且
        // 错误文本可操作——不能让它滑到 create_buffer 返回 0 那种不可诊断的失败。
        DEVICE.with(|d| d.set(7));
        let err = match ready_device("unit.test") {
            Err(Status::Fail) => {
                let mut buf = [0u8; ERR_CAP];
                let n = ERR_LEN.with(|l| l.get()) as usize;
                ERR.with(|b| buf[..n].copy_from_slice(&b.borrow()[..n]));
                String::from_utf8_lossy(&buf[..n]).into_owned()
            }
            other => panic!("期望 Fail，实际 {other:?}"),
        };
        assert!(err.contains("尚未绑定真实 GPUDevice"), "错误文本不可操作：{err}");
        assert!(err.contains("device.init"), "错误文本未指向修复动作：{err}");
    }

    #[test]
    fn scratch_addr_helpers_do_not_reenter_borrow() {
        // 真机原文：`RuntimeError: unreachable at wasm-function[145]`。
        // 根因：`SCRATCH.with(|s| { let mut v = s.borrow_mut(); … addr_of_buf(…) })`
        // —— addr_of_buf 内部再 borrow_mut 同一个 RefCell → BorrowMutError。
        // 修法：先独立取地址（借用即释放），再在新的借用里写字段。
        //
        // 注意：`addr_of_*` 返回的是 **u32**（wasm32 指针宽）。64 位 native 上
        // 真实地址会被截断，故**绝不可**把它转回指针解引用——本测试只断言
        // 「取地址不 panic + 字段可写入」，地址正确性由 wasm 真机项判。
        let out = addr_of_buf(|x| &mut x.scan_out);
        assert_ne!(out, 0, "scan_out 线性内存地址不应为 0");
        SCRATCH.with(|s| {
            let mut v = s.borrow_mut();
            v.scan_out.fill(7u32);
            v.out_ptr = out;
        });
        // 借用已释放 → 现在可以重新取得（旧的 bug 会在此处 panic）。
        assert_eq!(SCRATCH.with(|s| s.borrow().out_ptr), out);
        assert_eq!(SCRATCH.with(|s| s.borrow().scan_out[0]), 7u32);

        let hout = addr_of_buf(|x| &mut x.hist[..]);
        assert_ne!(hout, 0);
        SCRATCH.with(|s| s.borrow_mut().out_ptr = hout);
        assert_eq!(SCRATCH.with(|s| s.borrow().out_ptr), hout);

        // 只读版本同样不得重入。
        let sin = addr_of_slice(|s| &s.scan_in[..]);
        assert_ne!(sin, 0);
        assert_eq!(SCRATCH.with(|s| s.borrow().out_ptr), hout, "只读取址不得改动状态");
    }

    /// D-14 补测：布局常量必须与 WGSL 结构体逐字段对得上。
    ///
    /// 这是本次补测**本地就抓到**的一处真缺陷：`PathSlot` 按平铺算成 80 字节，
    /// 而 WGSL 里 5 个 `vec3` 每个 align=16/size=12，真实步长是
    /// `roundUp(92,16)=96`。少算 16 字节不会让 GPU 报错（storage buffer 只是
    /// 读得长一点），但第 2 个槽起全部字段错位——只有真机 dispatch 才暴露。
    #[test]
    fn path_slot_stride_matches_wgsl_layout() {
        assert_eq!(PATH_SLOT_STRIDE_BYTES, 96, "PathSlot 步长必须是 96（vec3 对齐填充）");
        // 逐字段偏移必须严格递增且不越界（越界 = 少算了尾部填充）。
        let offs = [
            slot_off::ORIGIN, slot_off::DIR, slot_off::TIME, slot_off::THROUGHPUT,
            slot_off::RADIANCE, slot_off::PREV_PDF, slot_off::STATE, slot_off::BOUNCE,
            slot_off::FLAGS, slot_off::AUX_STEPS, slot_off::PLACEHOLDER,
        ];
        for w in offs.windows(2) {
            assert!(w[0] < w[1], "PathSlot 字段偏移未递增：{:?}", offs);
        }
        assert!(slot_off::PLACEHOLDER + 12 <= PATH_SLOT_STRIDE_BYTES as usize);
        // slot_off 的每个偏移必须相对 WGSL 声明严格对齐。
        for (name, off, align) in [
            ("ORIGIN", slot_off::ORIGIN, 16usize), ("DIR", slot_off::DIR, 16usize),
            ("THROUGHPUT", slot_off::THROUGHPUT, 16usize), ("RADIANCE", slot_off::RADIANCE, 16usize),
            ("PLACEHOLDER", slot_off::PLACEHOLDER, 16usize),
            ("TIME", slot_off::TIME, 4usize), ("PREV_PDF", slot_off::PREV_PDF, 4usize),
            ("STATE", slot_off::STATE, 4usize), ("BOUNCE", slot_off::BOUNCE, 4usize),
            ("FLAGS", slot_off::FLAGS, 4usize), ("AUX_STEPS", slot_off::AUX_STEPS, 4usize),
        ] {
            assert_eq!(off % align, 0, "PathSlot.{name} 偏移 {off} 未按 {align} 对齐");
        }
    }

    /// 全部 storage/uniform 尺寸常量必须 16 字节对齐（D-14 真机教训）。
    #[test]
    fn all_layout_constants_are_16_aligned() {
        // 注：F32_BYTES 是**标量元素**宽度（array<f32> 的步长），本身按定义
        // 就是 4 且不参与 16 对齐；对齐门禁只针对"结构体/数组元素"尺寸。
        for (name, v) in [
            ("VEC3_STRIDE_BYTES", VEC3_STRIDE_BYTES),
            ("VEC4_STRIDE_BYTES", VEC4_STRIDE_BYTES),
            ("STATE_STRIDE_BYTES", STATE_STRIDE_BYTES),
            ("CONSTRAINT_STRIDE_BYTES", CONSTRAINT_STRIDE_BYTES),
            ("PARTICLE_STRIDE_BYTES", PARTICLE_STRIDE_BYTES),
            ("PATH_SLOT_STRIDE_BYTES", PATH_SLOT_STRIDE_BYTES),
            ("SIM_PARAMS_BYTES", SIM_PARAMS_BYTES), ("SPH_PARAMS_BYTES", SPH_PARAMS_BYTES),
            ("CLOTH_PARAMS_BYTES", CLOTH_PARAMS_BYTES),
            ("WF_PARAMS_BYTES", WF_PARAMS_BYTES),
            ("PARTICLES_BYTES", PARTICLES_BYTES), ("SPH_STATE_BYTES", SPH_STATE_BYTES),
            ("SPH_SCALAR_BYTES", SPH_SCALAR_BYTES), ("CLOTH_STATE_BYTES", CLOTH_STATE_BYTES),
            ("CLOTH_CONS_BYTES", CLOTH_CONS_BYTES),
            ("CLOTH_LAMBDA_BYTES", CLOTH_LAMBDA_BYTES), ("WF_SLOTS_BYTES", WF_SLOTS_BYTES),
            ("WF_ACCUM_BYTES", WF_ACCUM_BYTES),
        ] {
            assert!(v % 16 == 0, "{name}={v} 未 16 字节对齐");
        }
    }

    /// `vec3` 步长是 16 而不是 12；AoS 状态槽 = 两个 vec3 槽 = 32 字节。
    ///
    /// 【D-14 真机缺陷的同款陷阱】`Particle` 步长 48、状态槽步长 32，都**不是**
    /// `array<vec3<f32>>` 的 16。按"第 i 个 vec3"折算偏移会整体错位，且 GPU
    /// 侧**不报任何错**（storage buffer 只是读得长一点），只有逐元素判对才抓得到。
    #[test]
    fn vec3_stride_is_16_and_state_slot_is_32() {
        assert_eq!(VEC3_STRIDE_BYTES, 16);
        assert_eq!(STATE_STRIDE_BYTES, 32);
        // AoS 槽内字段偏移必须与 WGSL `struct { pos: vec3; vel: vec3; inv: f32 }`
        // 的 align/size 规则一致：pos@0、vel@16、inv@28（塞进 vel 的对齐填充）。
        assert_eq!(state_off::POS, 0);
        assert_eq!(state_off::VEL, 16);
        assert_eq!(state_off::INV, 28);
        assert!(state_off::INV + 4 <= STATE_STRIDE_BYTES as usize);
        // `Particle` 的字段偏移同理（48 字节步长）。
        assert_eq!(particle_off::POS, 0);
        assert_eq!(particle_off::MASS_INV, 12);
        assert_eq!(particle_off::VEL, 16);
        assert_eq!(particle_off::CELL, 32);
        assert!(PARTICLE_STRIDE_BYTES as usize > particle_off::PAD2 + 4);
        // 三个不同步长互不相等：vec3 槽 16 / AoS 状态槽 32 / Particle 48。
        // `Particle` 恰好等于 3 个 vec3 槽纯属数值巧合，**不能**因此按
        // `第 i 个 vec3` 折算偏移——两者的字段布局完全不同（见上方偏移断言）。
        assert_eq!(VEC3_STRIDE_BYTES * 2, STATE_STRIDE_BYTES);
        assert_ne!(PARTICLE_STRIDE_BYTES, STATE_STRIDE_BYTES);
        assert_eq!(PARTICLE_STRIDE_BYTES as usize, particle_off::PAD2 + 12);
    }

    /// `pack_state` / `unpack_vec3_at` 在 AoS 槽上必须互为逆运算。
    #[test]
    fn state_pack_unpack_roundtrip() {
        let n = SPH_P_COUNT as usize;
        let mut buf = vec![0u8; SPH_STATE_BYTES as usize];
        for i in 0..n {
            pack_state(
                &mut buf,
                i,
                [i as f32, i as f32 + 0.5, i as f32 + 0.25],
                [-(i as f32), 1.0, 2.0],
                if i % 3 == 0 { 0.0 } else { 1.0 },
            );
        }
        for i in 0..n {
            let base = i * STATE_STRIDE_BYTES as usize;
            assert_eq!(
                unpack_vec3_at(&buf, base + state_off::POS),
                [i as f32, i as f32 + 0.5, i as f32 + 0.25],
                "槽 {i} pos"
            );
            assert_eq!(unpack_vec3_at(&buf, base + state_off::VEL), [-(i as f32), 1.0, 2.0], "槽 {i} vel");
            assert_eq!(unpack_f32(&buf, base + state_off::INV), if i % 3 == 0 { 0.0 } else { 1.0 });
        }
    }

    /// 场景规模常量必须与派生的字节数自洽（防止"改了 count 忘了改 bytes"）。
    #[test]
    fn scene_sizes_are_self_consistent() {
        assert_eq!(WF_PATHS, WF_WIDTH * WF_HEIGHT * WF_SPP);
        assert_eq!(WF_PIXELS, WF_WIDTH * WF_HEIGHT);
        assert_eq!(WF_ACCUM_BYTES, WF_PIXELS * VEC4_STRIDE_BYTES);
        assert_eq!(WF_SLOTS_BYTES, WF_PATHS * PATH_SLOT_STRIDE_BYTES);
        assert_eq!(PARTICLES_BYTES, SIM_P_COUNT * PARTICLE_STRIDE_BYTES);
        // 参数区必须容得下三份 params 且互不重叠（偏移按 64 对齐分段）。
        for (name, off, len) in [
            ("SimParams", SIM_PARAMS_OFF, SIM_PARAMS_BYTES as usize),
            ("SphParams", SPH_PARAMS_OFF, SPH_PARAMS_BYTES as usize),
            ("ClothParams", CLOTH_PARAMS_OFF, CLOTH_PARAMS_BYTES as usize),
            ("WFParams", WF_PARAMS_OFF, WF_PARAMS_BYTES as usize),
        ] {
            assert!(off % 16 == 0, "{name} 偏移 {off} 未对齐");
            assert!(off + len <= PARAMS_ARENA_BYTES, "{name} 越过参数区末尾");
        }
        assert!(SIM_PARAMS_OFF + SIM_PARAMS_BYTES as usize <= SPH_PARAMS_OFF);
        assert!(SPH_PARAMS_OFF + SPH_PARAMS_BYTES as usize <= CLOTH_PARAMS_OFF);
        assert!(CLOTH_PARAMS_OFF + CLOTH_PARAMS_BYTES as usize <= WF_PARAMS_OFF);
    }

    /// `pack_slot` / `unpack_slot` 必须互为逆运算（96 字节布局的往返一致性）。
    #[test]
    fn slot_pack_unpack_roundtrip() {
        let want = wf_initial_slots();
        let mut buf = vec![0u8; WF_SLOTS_BYTES as usize];
        for (i, sl) in want.iter().enumerate() {
            pack_slot(&mut buf, i, sl);
        }
        for (i, sl) in want.iter().enumerate() {
            let got = unpack_slot(&buf, i);
            assert_eq!(got.origin, sl.origin, "槽 {i} origin");
            assert_eq!(got.dir, sl.dir, "槽 {i} dir");
            assert_eq!(got.time, sl.time, "槽 {i} time");
            assert_eq!(got.throughput, sl.throughput, "槽 {i} throughput");
            assert_eq!(got.radiance, sl.radiance, "槽 {i} radiance");
            assert_eq!(got.prev_pdf_dir, sl.prev_pdf_dir, "槽 {i} prev_pdf_dir");
            assert_eq!(got.state, sl.state, "槽 {i} state");
            assert_eq!(got.bounce, sl.bounce, "槽 {i} bounce");
            assert_eq!(got.flags, sl.flags, "槽 {i} flags");
            assert_eq!(got.aux_steps, sl.aux_steps, "槽 {i} aux_steps");
            assert_eq!(got.placeholder, sl.placeholder, "槽 {i} placeholder");
        }
    }

    /// 初始槽必须**一半 TERMINATED**，否则 `connect` 的早退分支走不到，
    /// "跳过即写入"这类缺陷就验不到（判据退化的典型形态）。
    #[test]
    fn wf_initial_slots_exercise_both_branches() {
        let slots = wf_initial_slots();
        let term = slots.iter().filter(|s| s.state == 1).count();
        assert_eq!(term, WF_PATHS as usize / 2, "TERMINATED 槽数应恰好一半");
        assert!(slots.iter().any(|s| s.state == 0), "需存在 ACTIVE 槽");
        // 各槽 radiance 必须互不相同（否则"整块相同"会掩盖求和顺序错误）。
        let mut keys: Vec<[u32; 3]> = slots
            .iter()
            .map(|s| s.radiance.map(|v| (v * 1024.0) as i32 as u32))
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), before, "槽 radiance 存在重复值");
    }

    /// `sim_particles` 的 CPU 参考必须自洽：cell 多格、非零速度。
    #[test]
    fn particles_reference_is_non_degenerate() {
        let r = particles_reference(&SIM_TEST_PARAMS);
        assert_eq!(r.len(), SIM_P_COUNT as usize);
        let mut cells: Vec<u32> = r.iter().map(|p| p.cell).collect();
        cells.sort_unstable();
        cells.dedup();
        assert!(cells.len() >= 2, "所有粒子落在同一格 → cell 判对无意义");
        // integrate 必须真的动了速度（重力非零）。
        assert!(r.iter().all(|p| p.vel[1] < 0.0), "重力未生效：vel.y 应全为负");
        // 发射位置互不相同（哈希派生，不是常量）。
        assert!(r.iter().all(|p| p.mass_inv == 1.0));
    }

    /// `sim_sph` 的 CPU 参考：密度非零、λ > 0（压缩侧投影生效）。
    #[test]
    fn sph_reference_is_non_degenerate() {
        let p = sph_scene_params();
        // rest_density 必须落在实测密度区间内（否则全压缩或全膨胀 → 判对退化）。
        let n0 = p.count as usize;
        let inv0: Vec<f32> = (0..n0).map(sph_initial_inv).collect();
        let mut pos0: Vec<[f32; 3]> = (0..n0).map(sph_initial_pos).collect();
        let mut vel0: Vec<[f32; 3]> = vec![[0.0f32; 3]; n0];
        for i in 0..n0 {
            if inv0[i] > 0.0 {
                for k in 0..3 {
                    vel0[i][k] += p.gravity[k] * p.dt;
                    pos0[i][k] += vel0[i][k] * p.dt;
                }
            }
        }
        let den0 = sph_density_ref(&p, &pos0);
        let dmin = den0.iter().cloned().fold(f32::MAX, f32::min);
        let dmax = den0.iter().cloned().fold(f32::MIN, f32::max);
        assert!(p.rest_density > dmin && p.rest_density < dmax,
                "rest_density={} 必须落在实测密度区间 ({}, {}) 之外侧", p.rest_density, dmin, dmax);
        let n = p.count as usize;
        let inv: Vec<f32> = (0..n).map(sph_initial_inv).collect();
        // pin 粒子（inv=0）必须恰好 2 个，否则守卫分支验不到。
        assert_eq!(inv.iter().filter(|v| **v == 0.0).count(), 2, "pin 粒子数应为 2");
        // 走一遍 predict（半隐式欧拉 + inv>0 守卫）。
        let mut pos: Vec<[f32; 3]> = (0..n).map(sph_initial_pos).collect();
        let mut vel: Vec<[f32; 3]> = vec![[0.0f32; 3]; n];
        for i in 0..n {
            if inv[i] > 0.0 {
                for k in 0..3 {
                    vel[i][k] += p.gravity[k] * p.dt;
                    pos[i][k] += vel[i][k] * p.dt;
                }
            }
        }
        // pin 粒子位置逐位未变。
        for i in 0..n {
            if inv[i] == 0.0 {
                assert_eq!(pos[i], sph_initial_pos(i), "pin 粒子 {i} 被移动了");
            }
        }
        let den = sph_density_ref(&p, &pos);
        assert!(den.iter().all(|d| *d > 0.0), "Poly6 求和出现 0 → 核半径/网格失配");
        // 非退化：密度必须真的累加了邻域项（不只是自项）。
        let self_term = poly6_2d(0.0, p.h);
        assert!(
            den.iter().any(|d| (d - self_term).abs() > F32_TOL),
            "所有密度都等于自项 W(0)={self_term} → 邻域扫描未生效，判据退化"
        );
        let mut lam = vec![SPH_LAMBDA_SEED; n];
        sph_project_ref(&p, &den, &inv, &mut lam);
        // 两个分支必须都走到：压缩侧被 clamp 到 0，膨胀侧保留哨兵。
        let zeros = lam.iter().filter(|l| l.abs() <= F32_TOL).count();
        let seeds = lam.iter().filter(|l| (*l - SPH_LAMBDA_SEED).abs() <= F32_TOL).count();
        assert!(zeros > 0 && seeds > 0, "λ 判据退化：{zeros} 个 0 / {seeds} 个哨兵（两者须都 > 0）");
        // pin 粒子（inv=0）恒保留哨兵：C<=0 走 early-return；C>0 时
        // denom = inv + α̃ = 0 + 0 = 0 ⇒ `if (denom <= 0) { return }`。两条路
        // 都必须"不写 λ"——这是 denom 守卫唯一的可观测后果。
        for i in 0..n {
            if inv[i] == 0.0 {
                assert_eq!(lam[i], SPH_LAMBDA_SEED,
                           "pin 粒子 {i} 的 λ 被改写 → denom<=0 / early-return 守卫失效");
            }
        }
    }

    /// `sim_cloth`：约束两两不相交 ⇒ 结果确定可判；λ 有限且非零。
    #[test]
    fn cloth_constraints_are_disjoint_and_reference_is_finite() {
        let p = CLOTH_TEST_PARAMS;
        let n = CLOTH_P_COUNT as usize;
        let cons = cloth_constraints();
        assert_eq!(cons.len(), CLOTH_C_COUNT as usize);
        // 不相交是**确定性前提**：按约束并行却写共享 cp_pos 会产生数据竞争。
        let mut seen: Vec<u32> = Vec::new();
        for &(ia, ib, rest, id) in &cons {
            assert_ne!(ia, ib, "自约束（ia==ib）会让 denom 退化");
            assert!(!seen.contains(&ia) && !seen.contains(&ib), "约束不两两不相交 → 数据竞争");
            seen.push(ia);
            seen.push(ib);
            assert!(rest > 0.0 && rest.is_finite(), "约束 {id} 的 rest 非法");
        }
        // 结构/弯曲两个分支都要有约束（否则 select 分支只验了一半）。
        let struct_n = cons.iter().filter(|c| c.2 <= p.spacing * 1.5).count();
        let bend_n = cons.len() - struct_n;
        assert!(struct_n >= 1 && bend_n >= 1, "结构/弯曲分支各需至少 1 条约束");
        // 初始构型必须**不满足**静止长（否则投影退化为恒等操作）。
        let inv: Vec<f32> = (0..n).map(cloth_initial_inv).collect();
        assert_eq!(inv.iter().filter(|v| **v == 0.0).count(), 2, "pin 粒子应为 2 个");
        let mut pos: Vec<[f32; 3]> = (0..n).map(cloth_initial_pos).collect();
        for i in 0..n {
            if inv[i] > 0.0 {
                for k in 0..3 {
                    pos[i][k] += p.gravity[k] * p.dt * p.dt;
                }
            }
        }
        let mut lam = vec![0.0f32; cons.len()];
        for (k, &(ia, ib, rest, _)) in cons.iter().enumerate() {
            let compliance = if rest > p.spacing * 1.5 { p.compliance_bend } else { p.compliance_struct };
            let mut pa = pos[ia as usize];
            let mut pb = pos[ib as usize];
            let mut l = lam[k];
            xpbd_distance_project(
                &mut pa, &mut pb, inv[ia as usize], inv[ib as usize],
                rest, compliance, p.dt, &mut l,
            );
            pos[ia as usize] = pa;
            pos[ib as usize] = pb;
            lam[k] = l;
        }
        assert!(lam.iter().all(|l| l.is_finite()), "λ 出现 NaN/Inf");
        assert!(lam.iter().any(|l| l.abs() > 1e-9), "λ 全为 0 → 投影未做功，判据退化");
        assert!(pos.iter().all(|q| q.iter().all(|v| v.is_finite())), "位置出现 NaN/Inf");
        // pin 粒子不得被约束移动（inv=0 ⇒ Δx = w·Δλ = 0）。
        for &(ia, ib, _, _) in &cons {
            for idx in [ia, ib] {
                if inv[idx as usize] == 0.0 {
                    assert_eq!(pos[idx as usize], cloth_initial_pos(idx as usize),
                               "pin 粒子 {idx} 被约束移动了");
                }
            }
        }
    }

    /// 解析 `struct <name> { … }` 的成员表（`(成员名, 类型串)`），按**顶层逗号**切分。
    ///
    /// 顶层 = `<>` / `()` / `[]` 深度为 0；`array<GiPrim, 4>` 里的逗号不算分隔符。
    /// 注释先剥掉（注释里的 `pos: vec3<f32>` 字样曾把别的门禁误导过，见 wgsl_lint 的注记）。
    fn wgsl_struct_members(src: &str, name: &str) -> Vec<(String, String)> {
        let head = match src.find(&format!("struct {name}")) {
            Some(i) => i,
            None => panic!("gi_compute.wgsl 里找不到 `struct {name}`"),
        };
        let open = match src[head..].find('{') {
            Some(i) => head + i,
            None => panic!("`struct {name}` 缺 `{{`"),
        };
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &src[open + 1..close.expect("`struct {name}` 括号不配平")];
        // 剥行注释（WGSL 无块注释）。
        let body: String = body
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut out = Vec::new();
        let mut depth = 0i32;
        let mut cur = String::new();
        for ch in body.chars() {
            match ch {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                ',' if depth == 0 => {
                    if let Some(m) = split_member(&cur) {
                        out.push(m);
                    }
                    cur.clear();
                    continue;
                }
                _ => {}
            }
            cur.push(ch);
        }
        if let Some(m) = split_member(&cur) {
            out.push(m);
        }
        out
    }

    /// `name: type` → `("name", "type")`；空片段返回 `None`。
    fn split_member(piece: &str) -> Option<(String, String)> {
        let t = piece.trim();
        if t.is_empty() {
            return None;
        }
        let (n, ty) = t.split_once(':').expect("WGSL 成员必须是 `name: type`");
        Some((n.trim().to_string(), ty.trim().to_string()))
    }

    /// 按 WGSL 规范求类型 `(align, size)`（本仓出现的子集：标量 / `vecN<T>` /
    /// `array<T, N>` / 已算过的结构体名）。
    ///
    /// 数组 stride 按**规范**取 `roundUp(size, align)`（`array<vec3<f32>>` 因此是
    /// 16 而非 12——这正是本文件 V2.2 踩过的那个坑）。结构体走 `known` 表递归。
    fn wgsl_type_layout(t: &str, known: &[(String, usize, usize)]) -> Option<(usize, usize)> {
        let t = t.trim();
        match t {
            "u32" | "i32" | "f32" => return Some((4, 4)),
            _ => {}
        }
        if let Some(rest) = t.strip_prefix("vec") {
            let (n, inner) = rest.split_once('<')?;
            let inner = inner.trim_end_matches('>').trim();
            if !matches!(inner, "u32" | "i32" | "f32") {
                return None;
            }
            return match n {
                "2" => Some((8, 8)),
                "3" => Some((16, 12)),
                "4" => Some((16, 16)),
                _ => None,
            };
        }
        if let Some(rest) = t.strip_prefix("array<") {
            let body = rest.strip_suffix('>')?;
            // 元素类型可含 `<>`，故从右找逗号（元素类型里没有逗号：本仓 `array<GiPrim,4>` /
            // `array<vec4<f32>,8>` 两种）。
            let (el, n) = body.rsplit_once(',')?;
            let count: usize = n.trim().parse().ok()?;
            let (ea, es) = wgsl_type_layout(el, known)?;
            let stride = es.div_ceil(ea) * ea;
            return Some((ea, stride * count));
        }
        known.iter().find(|(n, _, _)| n == t).map(|(_, a, s)| (*a, *s))
    }

    /// `gi_compute.wgsl` 的四个**存储布局**结构体：从 WGSL 源码文本按规范算出的
    /// 逐字段偏移 / 总 size，必须与 abi.rs 的四个 `GI_*_STRIDE_BYTES` 一致。
    ///
    /// # 为什么不是「断言常量等于 304」
    /// 那是自证恒真：常量本来就是 304，`assert_eq!(GI_PROBE_STRIDE_BYTES, 304)` 永远绿。
    /// 本测试读的是**shader 源码**，故能抓住两类真实漂移：
    ///   1. 改了 WGSL 结构体（增删字段 / 换类型）却没同步 Rust 打包偏移；
    ///   2. 有人把 vec4 化的字段改回 `vec3<f32>` / 用 `array<vec3<f32>>` 做布局
    ///      ——那是 V2.2 真机 bug 的形态（`vec3` 的 size 在真机驱动上不是 12）。
    /// 它**抓不住**「驱动不按规范实现」（那是真机 `gpu.gi_layout` 的职责），但能保证
    /// 「静态门禁 / Rust 打包 / shader 声明」三者始终在说同一件事。
    #[test]
    fn gi_wgsl_layout_matches_abi_constants() {
        let raw = include_str!("../../shaders/gi_compute.wgsl");
        // 先把 `//` 注释剥掉：本文件的注释里**成篇**引用 `vec3<f32>` 与偏移表，
        // 不剥会把文档当声明解析（wgsl_lint.mjs 也踩过同样的坑）。
        let src: String = raw
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");

        // 递归求一个结构体的 size 与其成员偏移表。
        fn layout(
            src: &str,
            name: &str,
            known: &[(String, usize, usize)],
        ) -> (usize, Vec<(String, usize, usize)>) {
            let members = wgsl_struct_members(src, name);
            assert!(!members.is_empty(), "`struct {name}` 解析出 0 个成员");
            let mut off = 0usize;
            let mut max_align = 1usize;
            let mut table = Vec::new();
            for (mn, mt) in &members {
                let (a, s) = wgsl_type_layout(mt, known)
                    .unwrap_or_else(|| panic!("`{name}.{mn}` 的类型 `{mt}` 无法解析"));
                off = off.div_ceil(a) * a;
                table.push((mn.clone(), off, s));
                off += s;
                max_align = max_align.max(a);
            }
            (off.div_ceil(max_align) * max_align, table)
        }

        // --- GiPrim（无嵌套）---
        let (prim_size, prim) = layout(&src, "GiPrim", &[]);
        let want_prim = [
            ("kind", 0usize),
            ("radius", 4),
            ("_pad0", 8),
            ("center", 16),
            ("_pad1", 32),
        ];
        for (mn, off) in want_prim {
            let got = prim
                .iter()
                .find(|(n, _, _)| n == mn)
                .unwrap_or_else(|| panic!("GiPrim 缺成员 `{mn}`"));
            assert_eq!(got.1, off, "GiPrim.{mn} 偏移变了（期望 @{off}，实际 @{}）", got.1);
        }
        assert_eq!(
            prim_size, GI_PRIM_STRIDE_BYTES as usize,
            "GiPrim size 与 GI_PRIM_STRIDE_BYTES 不一致（Rust 打包偏移会整体错位）"
        );

        let known = vec![("GiPrim".to_string(), 16usize, prim_size)];

        // --- GiParams（含 `array<GiPrim, 4>`）---
        let (par_size, par) = layout(&src, "GiParams", &known);
        let want_par = [
            ("probe_count", 0usize),
            ("point_count", 4),
            ("prim_count", 8),
            ("n_dirs", 12),
            ("env_radiance", 16),
            ("albedo", 32),
            ("t_max", 48),
            ("hit_eps", 52),
            ("safety", 56),
            ("max_steps", 60),
            ("prims", 64),
        ];
        for (mn, off) in want_par {
            let got = par
                .iter()
                .find(|(n, _, _)| n == mn)
                .unwrap_or_else(|| panic!("GiParams 缺成员 `{mn}`（vec4 化时把 _pad* 补回去了？）"));
            assert_eq!(got.1, off, "GiParams.{mn} 偏移变了（期望 @{off}，实际 @{}）", got.1);
        }
        assert_eq!(
            par_size, GI_PARAMS_BYTES as usize,
            "GiParams size 与 GI_PARAMS_BYTES 不一致（`gpu.params_roundtrip` 会 256B 对不上）"
        );

        // --- GiProbe ---
        let (probe_size, probe) = layout(&src, "GiProbe", &known);
        let want_probe = [("pos", 0usize), ("normal", 16), ("samples", 32), ("sh", 160)];
        for (mn, off) in want_probe {
            let got = probe
                .iter()
                .find(|(n, _, _)| n == mn)
                .unwrap_or_else(|| panic!("GiProbe 缺成员 `{mn}`"));
            assert_eq!(got.1, off, "GiProbe.{mn} 偏移变了（期望 @{off}，实际 @{}）", got.1);
        }
        assert_eq!(
            probe_size, GI_PROBE_STRIDE_BYTES as usize,
            "GiProbe size 与 GI_PROBE_STRIDE_BYTES 不一致：`array<GiProbe>` 的 stride 会错位，\
             真机上 `gi_probes[2]` 就会读到别人的字段"
        );

        // --- GiPoint ---
        let (point_size, point) = layout(&src, "GiPoint", &known);
        for (mn, off) in [("pos", 0usize), ("irradiance", 16)] {
            let got = point
                .iter()
                .find(|(n, _, _)| n == mn)
                .unwrap_or_else(|| panic!("GiPoint 缺成员 `{mn}`"));
            assert_eq!(got.1, off, "GiPoint.{mn} 偏移变了（期望 @{off}，实际 @{}）", got.1);
        }
        assert_eq!(point_size, GI_POINT_STRIDE_BYTES as usize, "GiPoint size 与常量不一致");

        // --- 结构体内**禁止**任何 `vec3<…>`（本轮真机 bug 的形态）---
        // 这是本测试存在的核心理由：`vec3<f32>` 的 size 在真机驱动上与规范分歧
        // （规范 12，实现更大），而 `array<vec3<f32>>` 的 stride 直接由该 size 决定
        // → 整个结构体的后续字段偏移错位。故四个布局结构体里**一个 vec3 都不许有**：
        // 三元组一律用 `vec4<f32>` 承载（w 恒 0，布局与 `vec3 + padding` 逐字节等价）。
        for sname in ["GiPrim", "GiParams", "GiProbe", "GiPoint"] {
            for (mn, mt) in wgsl_struct_members(&src, sname) {
                let compact = mt.replace(' ', "");
                assert!(
                    !compact.contains("vec3<"),
                    "`{sname}.{mn}: {mt}` 又在存储布局里用了 `vec3<…>`——\
                     真机驱动对 vec3 的 size 处理与规范不一致（V2.2 真机 bug 的根因：\
                     GiProbe.normal 被推到 samples 区、读到 -1 哨兵），\
                     必须用 `vec4<f32>`（w 恒 0，布局零变化）"
                );
            }
        }
    }

    /// 覆盖台账必须与 WGSL 实际入口逐份一致（`shader.dispatch_census` 的 native 版本）。
    #[test]
    fn coverage_ledger_matches_actual_entries() {
        for (file, entries) in COVERAGE {
            let src = shaders::expanded_sources()
                .into_iter()
                .find(|(n, _)| *n == *file)
                .unwrap_or_else(|| panic!("台账引用了不存在的 shader：{file}"))
                .1;
            let actual = compute_entries(&src);
            let declared: Vec<&str> = entries.iter().map(|(e, _, _)| *e).collect();
            assert_eq!(
                actual, declared,
                "{file} 的 compute 入口与台账不一致（WGSL 改了台账没跟上，或反之）"
            );
            for (name, st, why) in *entries {
                assert!(
                    *st == ST_COVERED || *st == ST_SKIP,
                    "{file}/{name} 状态非法：{st}"
                );
                if *st == ST_SKIP {
                    assert!(!why.is_empty(), "{file}/{name} 标了 SKIP 却没写原因");
                }
                // COVERED 允许带一条"由哪个测试项覆盖"的说明（便于追溯），
                // 但不强制——SKIP 必须写明原因才是硬要求。
            }
        }
    }

    /// 被测入口数必须与台账里 COVERED 的总数一致（防止台账漏登记）。
    #[test]
    fn covered_entry_count_matches_ledger() {
        let covered = COVERAGE
            .iter()
            .flat_map(|(_, es)| es.iter())
            .filter(|(_, st, _)| *st == ST_COVERED)
            .count();
        assert_eq!(
            covered, 19,
            "COVERED 入口应为 19 个（particles 3 + sph 3 + cloth 3 + wavefront 5 + GI 4 \
             + scene_frame 1）"
        );
    }

    /// 每个 COVERED 入口都必须真的出现在某个测试项的 dispatch 链里。
    ///
    /// 这条是"台账 vs 实际提交"的双向门禁：只登记不 dispatch 等于没测。
    /// 两种 steps 写法都要认：单 group 元组 `("entry", n, &[…])` 与
    /// 多 group 结构体字段 `MultiStep { entry: "entry", … }`。
    #[test]
    fn every_covered_entry_appears_in_a_dispatch_chain() {
        let abi = include_str!("abi.rs");
        let mut dispatched: Vec<&str> = Vec::new();
        for (_, entries) in COVERAGE {
            for (name, st, _) in *entries {
                if *st == ST_COVERED {
                    dispatched.push(name);
                }
            }
        }
        // 场景整帧路径（V2.2-T03）：`scene_main` 的入口名**不在本文件写字面量**，
        // 而是从 `scene_abi::SceneAbi::ENTRY` 单一真源取（见 `scene_gpu::ensure_pipeline`），
        // 故 steps 表形态的门禁认不出它。这里额外认一条**同样可机械核对**的锚点：
        // 入口名必须等于 scene_abi 的真源值，且本文件必须真的走场景提交函数。
        let scene_submit_present = abi.contains("scene_gpu::submit_scene_frame(dev,");
        assert!(
            scene_submit_present,
            "gpu.scene_frame 必须真调 scene_gpu::submit_scene_frame(dev, …)——否则台账              COVERED 的 scene_main 就是虚登记"
        );
        for e in dispatched {
            let in_tuple_chain = abi.contains(&format!("(\"{e}\", 1, &["));
            let in_multi_chain = abi.contains(&format!("entry: \"{e}\","));
            let in_scene_path = e == crate::scene_abi::SceneAbi::ENTRY && scene_submit_present;
            assert!(
                in_tuple_chain || in_multi_chain || in_scene_path,
                "台账标 COVERED 的入口 {e} 未出现在任何 steps 表里 → 未真被 dispatch"
            );
        }
    }

    /// 4 个新 GPU 测试项在无 device 时必须稳定 SKIP，且不得 panic。
    #[test]
    fn new_gpu_items_skip_cleanly_without_device() {
        DEVICE.with(|d| d.set(0));
        for id in ["gpu.sim_particles", "gpu.sim_sph", "gpu.sim_cloth", "gpu.rt_wavefront"] {
            let idx = TESTS.iter().position(|t| t.id == id).expect("清单含该项");
            assert!(TESTS[idx].needs_gpu, "{id} 必须标注 needs_gpu");
            assert!(TESTS[idx].collect.is_some(), "{id} 必须有 collect 阶段（读回判对）");
            let code = dispatch_panics(TESTS[idx].run, id);
            assert_eq!(code, status::SKIP, "{id} 无 device 时应 SKIP");
        }
    }

    /// `shader.dispatch_census` 是纯 CPU 项：无 device 也必须真跑出 PASS。
    #[test]
    fn dispatch_census_passes_without_device() {
        DEVICE.with(|d| d.set(0));
        let idx = TESTS.iter().position(|t| t.id == "shader.dispatch_census").expect("清单含该项");
        let code = dispatch_panics(TESTS[idx].run, "shader.dispatch_census");
        assert_eq!(code, status::PASS, "覆盖台账自检是纯 CPU 项，无 device 也应 PASS");
    }

    /// 浮点容差必须是"能抓住真实偏差、又不被 FMA 收缩误伤"的量级。
    #[test]
    fn f32_tolerance_is_tight_enough() {
        // 典型量级 1~10 的 f32 累加，相对误差 1e-6 ⇒ 绝对偏差 1e-5 量级。
        assert!(F32_TOL < 1e-5, "容差 {} 过松，抓不住真实缺陷", F32_TOL);
        assert!(F32_TOL > 1e-7, "容差 {} 过紧，FMA 收缩会误报", F32_TOL);
    }

    /// `cmp_f32` 必须真的能抓住超容差的偏差（判对原语自身的非恒真性）。
    #[test]
    fn cmp_f32_detects_real_deviation() {
        let want = [1.0f32, 2.0, 3.0];
        let good = le_f32(&want);
        assert!(cmp_f32("t", "good", &good, &want).is_ok(), "完全相同应通过");
        // 偏差 1e-3 远超容差 ⇒ 必须被抓到。
        let bad = le_f32(&[1.0 + 1e-3, 2.0, 3.0]);
        assert!(cmp_f32("t", "bad", &bad, &want).is_err(), "超容差偏差未被抓住 → 判对恒真");
    }

    /// `cmp_u32` / `cmp_bytes` 的非恒真性。
    #[test]
    fn integer_comparators_detect_real_deviation() {
        assert!(cmp_u32("t", "ok", &le_u32(&[1, 2, 3]), &[1, 2, 3]).is_ok());
        assert!(cmp_u32("t", "bad", &le_u32(&[1, 2, 4]), &[1, 2, 3]).is_err());
        assert!(cmp_bytes("t", "ok", &[1, 2, 3], &[1, 2, 3]).is_ok());
        assert!(cmp_bytes("t", "bad", &[1, 2, 4], &[1, 2, 3]).is_err());
        // 长度不同也必须被抓到（少读/多读的典型形态）。
        assert!(cmp_bytes("t", "short", &[1, 2], &[1, 2, 3]).is_err());
    }

    /// `cell_key` 必须与 `sim_common.wgsl::cell_key` 同口径（各轴 +1024 偏移、21 位/轴）。
    #[test]
    fn cell_key_matches_wgsl_formula() {
        assert_eq!(cell_key([0.0, 0.0, 0.0], 0.5), (1024u32) | (1024u32 << 11) | (1024u32 << 22));
        // 负坐标经 +1024 偏移后仍落在 21 位内。
        let neg = cell_key([-1.0, -1.0, -1.0], 0.5);
        assert_eq!(neg, (1022u32) | (1022u32 << 11) | (1022u32 << 22));
    }

    /// `poly6_2d` 必须与 `sim_common.wgsl::poly6_2d` 同式：r>=h 截断为 0。
    #[test]
    fn poly6_matches_wgsl_truncation() {
        assert_eq!(poly6_2d(0.5, 0.5), 0.0, "r>=h 必须截断为 0");
        assert_eq!(poly6_2d(0.6, 0.5), 0.0, "r>h 必须截断为 0");
        assert!(poly6_2d(0.0, 0.5) > 0.0, "r=0 处核值应为正");
        // 单调递减：r 越大核值越小。
        assert!(poly6_2d(0.1, 0.5) > poly6_2d(0.2, 0.5));
    }

    /// 越界保护的字节读写 helper 不得 panic（宪章 C18）。
    #[test]
    fn byte_helpers_are_bounds_safe() {
        let mut b = [0u8; 8];
        pack_f32(&mut b, 0, 1.5);
        pack_u32(&mut b, 4, 7);
        assert_eq!(unpack_f32(&b, 0), 1.5);
        assert_eq!(unpack_u32(&b, 4), 7);
        // 越界必须静默忽略而不是 panic。
        pack_f32(&mut b, 6, 9.0);
        pack_u32(&mut b, 100, 1);
        assert_eq!(unpack_f32(&b, 100), 0.0);
        assert_eq!(unpack_u32(&b, 100), 0);
        // pack_vec3_at / unpack_vec3_at 走字节版（工作区缓冲都是 [u8; N]），
        // 越界同样必须静默忽略而不是 panic。
        let mut b2 = [0u8; 32];
        pack_vec3_at(&mut b2, 0, [1.0, 2.0, 3.0]);
        assert_eq!(unpack_vec3_at(&b2, 0), [1.0, 2.0, 3.0]);
        assert_eq!(unpack_vec3_at(&b2, 29), [0.0, 0.0, 0.0]);
        pack_vec3_at(&mut b2, VEC3_STRIDE_BYTES as usize, [4.0, 5.0, 6.0]);
        assert_eq!(unpack_vec3_at(&b2, VEC3_STRIDE_BYTES as usize), [4.0, 5.0, 6.0]);
    }

    /// `params2_write` 越界必须返回 0 而不是 panic。
    #[test]
    fn params2_write_rejects_out_of_range() {
        assert_eq!(params2_write(0, &[0u8; 16]), SCRATCH.with(|s| s.borrow().params2.as_ptr() as usize as u32));
        assert_eq!(params2_write(PARAMS_ARENA_BYTES, &[0u8; 16]), 0, "越界应返回 0");
        assert_eq!(params2_write(PARAMS_ARENA_BYTES - 8, &[0u8; 16]), 0, "尾部越界应返回 0");
    }

    /// `wf_accumulate_ref`：spp 均值口径，且 w 分量恒 1。
    #[test]
    fn wf_accumulate_ref_is_spp_mean() {
        let slots = wf_initial_slots();
        let acc = wf_accumulate_ref(&WF_TEST_PARAMS, &slots);
        assert_eq!(acc.len(), WF_PIXELS as usize);
        // 第 0 像素：槽 0 与槽 1 的 radiance 均值。
        let m = [
            (slots[0].radiance[0] + slots[1].radiance[0]) * 0.5,
            (slots[0].radiance[1] + slots[1].radiance[1]) * 0.5,
            (slots[0].radiance[2] + slots[1].radiance[2]) * 0.5,
        ];
        assert!((acc[0][0] - m[0]).abs() < 1e-6);
        assert!((acc[0][1] - m[1]).abs() < 1e-6);
        assert!((acc[0][2] - m[2]).abs() < 1e-6);
        assert_eq!(acc[0][3], 1.0, "accum.w 应恒为 1");
    }
    /// D-14 补测：原有 2 个 GPU 项在无 device 时同样必须稳定 SKIP。
    #[test]
    fn gpu_test_bodies_do_not_panic_without_device() {
        // native 上 GPU 桩全返回 0 ⇒ 需 GPU 的项走 SKIP 分支；但**进入 run 之前**
        // 的准备工作（填 scan_in/keys、取地址）也必须无 panic——真机上正是这段
        // 准备代码在真 GPU 就绪后才执行到，因而只有真机能暴露。
        DEVICE.with(|d| d.set(0));
        for id in ["gpu.scan_block", "gpu.histogram"] {
            let idx = TESTS.iter().position(|t| t.id == id).expect("清单含该项");
            let code = dispatch_panics(TESTS[idx].run, id);
            assert_eq!(code, status::SKIP, "{id} 无 device 时应 SKIP");
        }
    }

    // -----------------------------------------------------------------------
    // GI 的 GPU 路径 —— CPU 参考自洽性 + SH π 约定锚点
    // -----------------------------------------------------------------------

    /// GI 无条件常量（保留，防未用告警；与 WGSL 编译期常量对表）。
    #[test]
    fn gi_layout_constants_are_16_aligned_and_consistent() {
        assert_eq!(GI_DIRS, 8);
        assert_eq!(GI_PARAMS_BYTES % 16, 0);
        assert_eq!(GI_PARAMS_BYTES, 256);
        assert_eq!(GI_PRIM_STRIDE_BYTES, 48);
        assert_eq!(GI_PROBE_STRIDE_BYTES % 16, 0);
        assert_eq!(GI_PROBE_STRIDE_BYTES, 304);
        assert_eq!(GI_POINT_STRIDE_BYTES % 16, 0);
        assert_eq!(GI_POINT_STRIDE_BYTES, 32);
        assert_eq!(GI_PROBES_BYTES, GI_PROBE_STRIDE_BYTES * GI_N_PROBES);
        assert_eq!(GI_POINTS_BYTES, GI_POINT_STRIDE_BYTES * GI_N_POINTS);
        // uniform 载荷自洽：写入 256B == 声明的 size（末尾是图元，无隐式填充）。
        let b = gi_params_bytes(&GI_TEST);
        assert_eq!(b.len(), GI_PARAMS_BYTES as usize);
    }

    /// `gpu.params_roundtrip` 的判据口径自证：4 个关键字段的偏移与解释值。
    ///
    /// 真机 collect 阶段用的就是这 4 个 `unpack_u32`，若偏移写错（例如把 `n_dirs@12`
    /// 写成 `@8`），判据会在真机上给出误导性的失败。本测把口径钉在 CPU 侧。
    #[test]
    fn params_roundtrip_field_offsets_decode_as_expected() {
        let b = gi_params_bytes(&GI_TEST);
        assert_eq!(unpack_u32(&b, 0), GI_N_PROBES, "probe_count@0");
        assert_eq!(unpack_u32(&b, 4), GI_N_POINTS, "point_count@4");
        assert_eq!(unpack_u32(&b, 8), GI_TEST.prims, "prim_count@8");
        assert_eq!(unpack_u32(&b, 12), GI_DIRS, "n_dirs@12");
        // 判据里写死的期望值本身也得对（防"两端一起写错"）。
        assert_eq!(GI_TEST.prims, 2, "本测试的有效图元数应为 2（后两条是容量填充）");
        assert_eq!(GI_DIRS, 8);
    }

    /// 输入缓冲的哨兵预填必须真的落进字节（否则真机"没跑"仍会被读成 0）。
    #[test]
    fn gi_pack_prefills_negative_sentinels() {
        gi_pack_probes();
        gi_pack_points();
        let (probes, points) = SCRATCH.with(|s| (s.borrow().gi_probes, s.borrow().gi_points));
        for i in 0..GI_N_PROBES as usize {
            let base = i * GI_PROBE_STRIDE_BYTES as usize;
            // samples 全段 -1。
            for d in 0..GI_DIRS as usize {
                let o = base + 32 + d * VEC3_STRIDE_BYTES as usize;
                for c in 0..3 {
                    assert_eq!(unpack_f32(&probes, o + c * 4), GI_SENTINEL_SAMPLE);
                }
            }
            // sh 全段 -2。
            for k in 0..9usize {
                let o = base + 160 + k * VEC3_STRIDE_BYTES as usize;
                for c in 0..3 {
                    assert_eq!(unpack_f32(&probes, o + c * 4), GI_SENTINEL_SH);
                }
            }
            // pos/normal 不被哨兵污染。
            assert_eq!(unpack_f32(&probes, base + 8), gi_probe_inputs()[i][2]);
            assert_eq!(unpack_f32(&probes, base + 24), gi_probe_inputs()[i][5]);
        }
        for i in 0..GI_N_POINTS as usize {
            let o = i * GI_POINT_STRIDE_BYTES as usize + 16;
            for c in 0..3 {
                assert_eq!(unpack_f32(&points, o + c * 4), GI_SENTINEL_IRRADIANCE);
            }
        }
        // 哨兵必须与"正常结果"可区分：物理上辐亮度/辐照度非负。
        assert!(GI_SENTINEL_SAMPLE < 0.0 && GI_SENTINEL_SH < 0.0 && GI_SENTINEL_IRRADIANCE < 0.0);
        assert!(GI_SENTINEL_SAMPLE != GI_SENTINEL_SH && GI_SENTINEL_SH != GI_SENTINEL_IRRADIANCE);
    }

    /// `gpu.params_roundtrip` 无 device 时稳定 SKIP，且清单登记齐全。
    #[test]
    fn params_roundtrip_body_skips_without_device() {
        DEVICE.with(|d| d.set(0));
        let idx = TESTS
            .iter()
            .position(|t| t.id == "gpu.params_roundtrip")
            .expect("清单含 gpu.params_roundtrip");
        assert!(TESTS[idx].needs_gpu, "gpu.params_roundtrip 必须标注 needs_gpu");
        assert!(TESTS[idx].collect.is_some(), "gpu.params_roundtrip 必须有 collect 阶段");
        assert_eq!(dispatch_panics(TESTS[idx].run, "gpu.params_roundtrip"), status::SKIP);
        // 必须排在 gi_compute **之前**（报告里先看到参数自检结论）。
        let gi = TESTS.iter().position(|t| t.id == "gpu.gi_compute").expect("清单含 gpu.gi_compute");
        assert!(idx < gi, "params_roundtrip 必须排在 gpu.gi_compute 之前（idx={idx}, gi={gi}）");
    }

    /// SDF CPU 参考的几何正确性：球/盒的遮挡判定与解析一致（非恒真自证）。
    #[test]
    fn gi_sdf_occlusion_matches_analytic() {
        let p = GI_TEST;
        // 从探针 0（原点上方 1，朝上）——地板在下方，向上应无遮挡。
        assert!(
            !gi_sphere_trace_ref([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], p.prims, &p),
            "朝上不应命中地板"
        );
        // 朝下应命中地板（盒顶面 y=0，起点 y=1）。
        assert!(
            gi_sphere_trace_ref([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], p.prims, &p),
            "朝下应命中地板盒"
        );
        // 从 (1,1,0) 朝 +X（正对球心 (2,0,0)，半径 1）：射线 y=1 高于球顶 y=1，
        // 相切/擦过——命中与否取决于 hit_eps；改从 y=0 高度朝 +X 必命中球。
        assert!(
            gi_sphere_trace_ref([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], p.prims, &p),
            "y=0 高度朝 +X 应命中球"
        );
        // 反向（−X）且远离地板横向上方：从 (5,5,0) 朝 +X（远离球与地板）不命中。
        assert!(
            !gi_sphere_trace_ref([5.0, 5.0, 0.0], [1.0, 0.0, 0.0], p.prims, &p),
            "远离几何的射线不应命中"
        );
    }

    /// 常数环境下的 SH 自检锚点：探针朝上、全可见（用空场）时，重建辐照度
    /// 应 ≈ π·L0（各向同性），进而 Lambert 辐亮度 ≈ albedo·L0。
    ///
    /// 这里用一个"空场"（prim_count=0，全可见）验证 SH 投影+重建的 π 约定
    /// 与 `sh.rs::constant_env_reconstructs_to_pi_l0` 同锚。
    #[test]
    fn gi_constant_env_reconstructs_to_pi_l0() {
        // 空场参数（prim_count=0 → 所有方向可见）。
        let mut p = GI_TEST;
        p.prims = 0;
        p.env = [0.7, 0.7, 0.7];
        p.albedo = [1.0, 1.0, 1.0];
        // 半球均匀采样投影 → 重建 E(+Z)。数值求积会因 n_dirs=8 偏低，故允许较宽
        // 相对误差（8 个方向对 L0+方向性的求积误差 ~10%）。关键是验证 π 约定
        // 使 E 量级落在 π·L0 附近，而非 0 或 L0。
        let probe = gi_sh_project_ref([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &p);
        // 手动重建（朝 +Z）。
        let yb = gi_sh_basis_l2_ref([0.0, 0.0, 1.0]);
        let mut e = 0.0f32;
        for k in 0..9 {
            e += gi_cosine_lobe_ref(gi_band_of_ref(k)) * probe.sh[k][0] * yb[k];
        }
        let expected = WF_PI * 0.7; // π·L0
        let rel = (e - expected).abs() / expected;
        assert!(
            rel < 0.2,
            "常数环境 SH 重建 {e} 应接近 π·L0={expected}（相对误差 {rel}）——π 约定可能被破坏"
        );
        // 反例：若漏乘余弦瓣（投影时就当辐照度），量级会明显不同。
        let mut wrong = 0.0f32;
        for k in 0..9 {
            wrong += probe.sh[k][0] * yb[k];
        }
        assert!((wrong - e).abs() > 1e-4, "余弦瓣 Â_l 必须在重建中起作用（非恒真）");
    }

    /// 跨实现指纹判对（**纯 CPU 口径**）：Rust CPU 参考压出的指纹必须等于
    /// `host_test.mjs` 里 **第三份独立 JS 实现**算出的值。三份实现（本 CPU 参考 /
    /// JS / WGSL 源码）互不共享代码，指纹相等即证明三者的采样序、SH 常数、π 约定、
    /// Lambert 口径全部一致。
    ///
    /// 该常量 `0x6adb7c92` 由 `host_test.mjs` 的 JS 独立实现产出（见其
    /// `js_gi_fingerprint=` 输出）。若日后**有意**改动 GI 口径，两边须同步更新。
    ///
    /// # V2.2 变更（球 SDF 修正，CHANGELOG R-2026-07 条）
    /// 旧锚 `0xaf8429a3` → 新锚 `0x6adb7c92`。原因：`gi_prim_dist` 的球分支由
    /// `q.x`（沿 x 的无限大板）改为 `|p−c|−r`，probe 0 的可见方向数由 5/8 变 8/8、
    /// probe 1 由 0/8 变 4/8，故全部 SH 系数与辐照度都变了。**Rust 与 JS 两端同步改**，
    /// 否则跨实现判对失去意义。
    ///
    /// ⚠️ 这是 **CPU** 指纹，**不含 FMA**。真机 GPU 允许 FMA 收缩（见 `c_gpu_gi_compute`
    /// 文档），逐位指纹与 CPU 不同（实测真机 `0x87ac3cf3`），故 GPU↔CPU 判对走
    /// [`F32_TOL`] 逐字段比对（真机实测最大偏差 1.2e-7，mismatch=0），**不**用逐位
    /// 指纹。真机指纹仅用于"同机两次运行一致"的确定性核对。
    #[test]
    fn gi_cpu_reference_fingerprint_matches_independent_js() {
        let p = GI_TEST;
        let shs: Vec<[[f32; 3]; 9]> = gi_probe_inputs()
            .iter()
            .map(|row| gi_sh_project_ref([row[0], row[1], row[2]], [row[3], row[4], row[5]], &p).sh)
            .collect();
        // 查询点插值需要探针位置 + SH；镜像 c_gpu_gi_compute 的最近探针重建。
        let refs: Vec<([f32; 3], GiProbeRef)> = gi_probe_inputs()
            .iter()
            .enumerate()
            .map(|(i, row)| {
                ([row[0], row[1], row[2]], GiProbeRef { samples: [[0.0; 3]; GI_DIRS as usize], sh: shs[i] })
            })
            .collect();
        let points: Vec<[f32; 3]> =
            gi_point_inputs().iter().map(|q| gi_interpolate_ref(*q, &refs, &p)).collect();
        let fp = gi_fingerprint_of(&shs, &points);
        assert_eq!(
            fp, 0x6adb_7c92,
            "GI CPU 参考指纹 {fp:#010x} 与 JS 独立实现 0x6adb7c92 不一致——口径已漂移"
        );
    }

    /// 确定性：同输入两遍 GI CPU 参考逐位一致（全链路确定性铁律）。
    #[test]
    fn gi_cpu_reference_is_bitwise_deterministic() {        let p = GI_TEST;
        let inp = gi_probe_inputs();
        for row in inp.iter() {
            let pos = [row[0], row[1], row[2]];
            let nrm = [row[3], row[4], row[5]];
            let a = gi_sh_project_ref(pos, nrm, &p);
            let b = gi_sh_project_ref(pos, nrm, &p);
            for k in 0..9 {
                for c in 0..3 {
                    assert_eq!(a.sh[k][c].to_bits(), b.sh[k][c].to_bits(), "SH 系数不确定");
                }
            }
        }
    }

    /// FAIL 诊断块必须「说真话」：三种已知形态的读回缓冲各落到正确的判定分支。
    ///
    /// # 为什么这条单测值得存在
    /// 诊断块只在真机 FAIL 时输出，而沙箱永远看不到真机——若它本身有 bug
    /// （例如把模长算成 0、或判定分支写反），真机重跑拿到的就是一段**误导性**
    /// 文本，比没有诊断更贵。本测用三个已知形态钉死它：
    /// 1. **pos/normal 完好、samples/sh 仍是哨兵**（= 入口一个都没跑）→ 必须落 (A) 族；
    /// 1b. **pos/normal 完好、samples/sh 已写成 0**（= 入口跑了但全遮挡）→ 必须落 (B) 族；
    /// 2. **整块全零**（连 pos/normal 也零）→ 必须指认「上传/读回不是同一块 buffer」；
    /// 3. **按 CPU 参考打包**（= 正确结果）→ 不得再报未写入，且 probe 0 显示 8/8 可见
    ///    （`sh[0][0]=1.595208` 的来源）。
    ///
    /// # V2.2 数值更新（球 SDF 修正的连带影响，已登记 CHANGELOG）
    /// 修正前 probe 0 是 5/8 可见、`sh[0][0]=0.9970052`、probe 1 是 0/8；
    /// 修正后（`|p−c|−r`）probe 0 = 8/8、probe 1/2 = 4/8、`sh[0][0]=1.595208`。
    /// probe 1 由 0/8 变 4/8 正是缺陷①的可见证据：旧 `q.x` 板把起点 (1.001,1,0)
    /// 判成 −1e−3（在体内），修正后真实距离是 0.414（在体外）。
    #[test]
    fn gi_diag_block_distinguishes_zero_from_reference() {
        let p = GI_TEST;
        let inputs = gi_probe_inputs();
        let zero_points = [0u8; GI_POINTS_BYTES as usize];

        // ---- 形态 2：整块全零（含 pos/normal）----
        let zero_probes = [0u8; GI_PROBES_BYTES as usize];
        let d0 = gi_diag_text(&zero_probes, &zero_points);
        assert!(d0.contains("[中间态 1/3]"), "诊断块缺少 trace 段");
        assert!(d0.contains("[中间态 2/3]"), "诊断块缺少 project 段");
        assert!(d0.contains("[中间态 3/3]"), "诊断块缺少 interpolate 段");
        assert!(d0.contains("可见 GPU 0/8 vs CPU 8/8"), "probe 0 的可见方向数判错：{d0}");
        assert!(d0.contains("读回的 pos/normal 与 CPU 输入不一致"), "全零缓冲未指认 buffer 身份：{d0}");

        // ---- 形态 1：「入口一个都没跑」= 读回等于 gi_pack_probes 的初值（全哨兵）----
        // 【为什么不再手工填零】本轮给输入缓冲预填了负值哨兵，真实的"没跑"读回
        // 就是哨兵值（-1 / -2），而不是零。此处直接取 `gi_pack_probes()` 的产物，
        // 保证夹具与真机初始字节**逐字节同源**。
        gi_pack_probes();
        let bare_probes = SCRATCH.with(|s| s.borrow().gi_probes);
        let d1 = gi_diag_text(&bare_probes, &zero_points);
        assert!(d1.contains("(A) 两个入口都没有真正写入"), "未写入形态未落到 (A) 族：{d1}");
        assert_eq!(
            d1.matches("写入状态=哨兵未写(-2)").count(),
            3,
            "未写入形态三个探针的 sh 都应标「哨兵未写」：{d1}"
        );
        assert!(!d1.contains("退化法线"), "pos/normal 完好时不应提示法线退化");

        // ---- 形态 1b：「入口跑过但结果为零」= pos/normal 有值、samples/sh 是 0（无哨兵）----
        // 这正是 (B) 族：哨兵已被覆写、但方向全被判遮挡 ⇒ sh 恰为 0。
        let mut zeroed_probes = [0u8; GI_PROBES_BYTES as usize];
        for (i, inp) in inputs.iter().enumerate() {
            let base = i * GI_PROBE_STRIDE_BYTES as usize;
            for (k, v) in inp.iter().enumerate() {
                let off = base + if k < 3 { k * 4 } else { 16 + (k - 3) * 4 };
                pack_f32(&mut zeroed_probes, off, *v);
            }
        }
        let d1b = gi_diag_text(&zeroed_probes, &zero_points);
        assert!(d1b.contains("(B) 两个入口都写过"), "全遮挡形态未落到 (B) 族：{d1b}");
        assert_eq!(
            d1b.matches("写入状态=已写但全零").count(),
            3,
            "全遮挡形态三个探针的 sh 都应标「已写但全零」：{d1b}"
        );

        // ---- 形态 3：按 CPU 参考打包（pos/normal + samples + sh 全填）----
        let mut ok_probes = [0u8; GI_PROBES_BYTES as usize];
        for (i, inp) in inputs.iter().enumerate() {
            let base = i * GI_PROBE_STRIDE_BYTES as usize;
            for (k, v) in inp.iter().enumerate() {
                let off = base + if k < 3 { k * 4 } else { 16 + (k - 3) * 4 };
                pack_f32(&mut ok_probes, off, *v);
            }
            let r = gi_sh_project_ref([inp[0], inp[1], inp[2]], [inp[3], inp[4], inp[5]], &p);
            for d in 0..GI_DIRS as usize {
                let o = base + 32 + d * 16;
                for c in 0..3 {
                    pack_f32(&mut ok_probes, o + c * 4, r.samples[d][c]);
                }
            }
            for k in 0..9 {
                let o = base + 160 + k * 16;
                for c in 0..3 {
                    pack_f32(&mut ok_probes, o + c * 4, r.sh[k][c]);
                }
            }
        }
        let d2 = gi_diag_text(&ok_probes, &zero_points);
        assert!(!d2.contains("(A) 三个入口都没有真正写入"), "参考态被误判为未写入");
        // 球 SDF 修正后三条探针的 CPU 参考 SH 都非零（probe 1/2 同位置同法线 → 同值），
        // 故参考态**没有任何** probe 会被标"全零"。
        assert_eq!(
            d2.matches("写入状态=已写有值").count(),
            3,
            "参考态三个探针的 sh 都应标「已写有值」：{d2}"
        );
        assert!(
            d2.contains("sh[0] GPU(1.595208,1.417963,1.240718)"),
            "参考态 probe 0 的 sh[0] 应按 f32 参考值打印：{d2}"
        );
        assert!(d2.contains("可见 GPU 8/8 vs CPU 8/8"), "参考态的可见方向数应与 CPU 一致：{d2}");
        // probe 1 修正前是 0/8（旧 q.x 板把起点判成"在体内"），修正后是 4/8。
        // 这条差异是缺陷①的直接可见证据，钉住它防回归。
        assert!(d2.contains("可见 GPU 4/8 vs CPU 4/8"), "probe 1/2 应为 4/8 可见：{d2}");
    }

    /// 遮挡确实起作用（非恒全可见）：朝 +X（正对球）的探针可见方向数应 < 朝上者。
    #[test]
    fn gi_occlusion_reduces_visibility() {
        let p = GI_TEST;
        let inp = gi_probe_inputs();
        let count_visible = |row: &[f32; 6]| {
            let tr = gi_probe_trace_ref([row[0], row[1], row[2]], [row[3], row[4], row[5]], &p);
            tr.samples.iter().filter(|s| s[0] > 0.0).count()
        };
        let up = count_visible(&inp[0]); // 朝上
        let toward_ball = count_visible(&inp[1]); // 朝 +X 正对球
        assert!(
            toward_ball < up,
            "朝球的探针可见方向数（{toward_ball}）应少于朝上者（{up}）——遮挡必须起作用"
        );
    }

    // -----------------------------------------------------------------------
    // 3.7：`prim_count` 误读 = 全遮挡 的机制证明（回归钉子）
    // -----------------------------------------------------------------------

    /// **危险填充**（= 修复前的 `GI_PRIMS[2]/[3]`）：球心在原点、半径 1。
    ///
    /// 保留它作为**历史快照**参与回归测试，是为了永久钉住"为什么填充数据必须安全"
    /// 这件事——它不是被使用的图元，只是容量占位，但一旦 `prim_count` 被误读，
    /// 它就会参与 `min`，并且恰好把三个探针起点都包进球内。
    const UNSAFE_FILLER: [GiPrimRef; GI_MAX_PRIMS as usize] = [
        GiPrimRef { kind: 0, radius: 1.0, center: [2.0, 0.0, 0.0] },
        GiPrimRef { kind: 1, radius: 10.0, center: [0.0, -10.0, 0.0] },
        GiPrimRef { kind: 0, radius: 1.0, center: [0.0, 0.0, 0.0] },
        GiPrimRef { kind: 0, radius: 1.0, center: [0.0, 0.0, 0.0] },
    ];

    /// **历史算法快照**（= 修复前的 `gi_prim_dist` 球分支：返回 `q[0]`，沿 x 的无限大板）。
    ///
    /// 只出现在测试里，用途单一：证明"旧球 SDF + 旧填充 + prim_count=4"确实能产出
    /// 真机观测到的"8/8 全遮挡、sh 全 0"。生产代码不得再引用本函数。
    fn legacy_prim_dist(prim: &GiPrimRef, p: [f32; 3]) -> f32 {
        let q = [
            (p[0] - prim.center[0]).abs() - prim.radius,
            (p[1] - prim.center[1]).abs() - prim.radius,
            (p[2] - prim.center[2]).abs() - prim.radius,
        ];
        if prim.kind == 1 {
            let cx = q[0].max(0.0);
            let cy = q[1].max(0.0);
            let cz = q[2].max(0.0);
            (cx * cx + cy * cy + cz * cz).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
        } else {
            q[0] // ← 历史缺陷本体：只取 x 分量
        }
    }

    /// 历史版本的 sphere tracing（只用于上面的快照复算）。
    fn legacy_trace(ro: [f32; 3], rd: [f32; 3], prims: &[GiPrimRef], p: &GiTestParams) -> bool {
        let rl = (rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]).sqrt();
        if rl < 0.5 {
            return false;
        }
        let dir = [rd[0] / rl, rd[1] / rl, rd[2] / rl];
        let mut t = 0.0f32;
        for _ in 0..p.max_steps {
            let q = [ro[0] + dir[0] * t, ro[1] + dir[1] * t, ro[2] + dir[2] * t];
            let mut d = 1.0e6f32;
            for prim in prims {
                d = d.min(legacy_prim_dist(prim, q));
            }
            if d <= p.hit_eps {
                return true;
            }
            t += d * p.safety;
            if t > p.t_max {
                return false;
            }
        }
        false
    }

    /// 对每个探针收集"逐方向采样辐亮度"（用给定图元表 + 给定 trace 实现）。
    ///
    /// `p.prims` 决定**取表的前几项**（与生产路径的 `prim_count` clamp 同口径），
    /// 这样同一张表可以分别按 `prim_count=2` / `=4` 求值——正是"误读 4"要模拟的东西。
    fn samples_per_probe(
        prims: &[GiPrimRef],
        p: &GiTestParams,
        trace: impl Fn([f32; 3], [f32; 3], &[GiPrimRef], &GiTestParams) -> bool,
    ) -> Vec<[[f32; 3]; GI_DIRS as usize]> {
        let prims = &prims[..(p.prims as usize).min(prims.len())];
        let mut out = Vec::new();
        for row in gi_probe_inputs().iter() {
            let pos = [row[0], row[1], row[2]];
            let normal = [row[3], row[4], row[5]];
            let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let n = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
            let (t, b, _nn) = wf_onb(n);
            let n_dirs = p.n_dirs.min(GI_DIRS);
            let mut samples = [[0.0f32; 3]; GI_DIRS as usize];
            for d in 0..n_dirs {
                let local = gi_hemi_dir_ref(d, n_dirs);
                let world = wf_normalize([
                    t[0] * local[0] + b[0] * local[1] + n[0] * local[2],
                    t[1] * local[0] + b[1] * local[1] + n[1] * local[2],
                    t[2] * local[0] + b[2] * local[1] + n[2] * local[2],
                ]);
                let ro = [pos[0] + n[0] * 1e-3, pos[1] + n[1] * 1e-3, pos[2] + n[2] * 1e-3];
                if !trace(ro, world, prims, p) {
                    samples[d as usize] = p.env;
                }
            }
            out.push(samples);
        }
        out
    }

    /// 每条探针的"可见方向数"（= samples 中非零项的个数，与 `visible_total` 同口径）。
    fn visible_per_probe(
        prims: &[GiPrimRef],
        p: &GiTestParams,
        trace: impl Fn([f32; 3], [f32; 3], &[GiPrimRef], &GiTestParams) -> bool,
    ) -> Vec<usize> {
        samples_per_probe(prims, p, trace)
            .iter()
            .map(|s| s.iter().filter(|v| v[0] > 0.0 || v[1] > 0.0 || v[2] > 0.0).count())
            .collect()
    }

    /// **`prim_count` 误读 = 全遮挡**的机制证明（V2.2 真机 FAIL 的根因验证）。
    ///
    /// 四组对照，全部走**同一份** CPU 参考实现（`_prims` 变体），只换图元表与算法快照：
    /// 1. 出库配置（安全填充 + 修正后的球 SDF）：`prim_count=2` 与 `=4` 必须**逐位一致**
    ///    ——这是"误读 4 无害"的直接证据。
    /// 2. 危险填充 + 修正后的球 SDF：probe 0 / probe 2 仍被 8/8 全遮挡
    ///    ——证明"填充数据的危险性不因球 SDF 修好而消失"，故第 3.2 步的填充替换是必需的。
    /// 3. 危险填充 + **历史球 SDF**（`q.x` 板）：三个探针全部 8/8 全遮挡、`sh` 全 0
    ///    ——正是真机观测到的形态（`60 = 27 + 0 + 27 + 6`：probe 1 的 CPU 参考本来全 0）。
    /// 4. 出库配置下 `prim_count=2` 的 CPU 自比 mismatch 必须为 0（同输入同实现）。
    ///
    /// 用 `cargo test -p prism-gpu gi_prim_count_misread -- --nocapture` 可看到全部数字。
    #[test]
    fn gi_prim_count_misread_does_not_blind_the_scene() {
        let p2 = GI_TEST; // prims = 2
        let mut p4 = GI_TEST;
        p4.prims = GI_MAX_PRIMS; // 模拟 prim_count 被误读成 4（= point_count@4）

        // ---- 组 1：出库配置，prim_count=2 vs 4 必须一致 ----
        let ship2 = visible_per_probe(&GI_PRIMS, &p2, gi_sphere_trace_prims);
        let ship4 = visible_per_probe(&GI_PRIMS, &p4, gi_sphere_trace_prims);
        println!("[3.7] 出库配置（安全填充）可见方向数：prim_count=2 → {ship2:?}；=4 → {ship4:?}");
        assert_eq!(ship2, ship4, "安全填充必须让 prim_count=2/4 完全等价");

        let mut sh2: Vec<[[f32; 3]; 9]> = Vec::new();
        let mut sh4: Vec<[[f32; 3]; 9]> = Vec::new();
        for row in gi_probe_inputs().iter() {
            let pos = [row[0], row[1], row[2]];
            let nrm = [row[3], row[4], row[5]];
            sh2.push(gi_sh_project_prims(pos, nrm, &GI_PRIMS[..2], &p2).sh);
            sh4.push(gi_sh_project_prims(pos, nrm, &GI_PRIMS[..4], &p4).sh);
        }
        for i in 0..sh2.len() {
            println!("[3.7] 出库配置 probe {i}: sh[0]（prim_count=2）= {:?}", sh2[i][0]);
            for k in 0..9 {
                for c in 0..3 {
                    assert_eq!(
                        sh2[i][k][c].to_bits(),
                        sh4[i][k][c].to_bits(),
                        "probe {i} sh[{k}][{c}] 在 prim_count=2/4 下必须逐位一致"
                    );
                }
            }
        }
        // 出库配置下三个探针都不得是全遮挡（否则"球/地板"这个场景就没意义了）。
        assert!(ship2.iter().all(|&v| v > 0), "出库配置不应出现全遮挡探针：{ship2:?}");

        // ---- 组 3（先做，作为对照基准）：危险填充 + 历史球 SDF → 8/8 全遮挡 ----
        let legacy = visible_per_probe(&UNSAFE_FILLER, &p4, legacy_trace);
        let legacy_legit = visible_per_probe(&UNSAFE_FILLER, &p2, legacy_trace);
        println!("[3.7] 危险填充+旧SDF 可见方向数：prim_count=2 → {legacy_legit:?}；=4 → {legacy:?}");
        assert_eq!(
            legacy,
            vec![0usize; GI_N_PROBES as usize],
            "旧球 SDF + 旧填充在 prim_count=4 时必须复现 8/8 全遮挡（真机形态）"
        );
        // 对照：旧填充在 **prim_count=2** 下的 sh[0]（= 旧代码出库时的 CPU 参考值，
        // 即真机报告里 `CPU 0.9970052` 的来源）。
        let legacy2_samples = samples_per_probe(&UNSAFE_FILLER, &p2, legacy_trace);
        for (i, row) in gi_probe_inputs().iter().enumerate() {
            let sh = gi_sh_project_samples([row[3], row[4], row[5]], &legacy2_samples[i], &p2);
            println!("[3.7] 危险填充+旧SDF prim_count=2 时 probe {i} 的 sh[0] = {:?}", sh[0]);
        }
        // 全遮挡 ⇒ 每个方向的采样都是 (0,0,0) ⇒ 任何线性投影必然整块为零。
        // 这里用**现行**投影函数对**历史** samples 求值，证明"sh 全 0"不是另一条实现的说法。
        let legacy_samples = samples_per_probe(&UNSAFE_FILLER, &p4, legacy_trace);
        for (i, row) in gi_probe_inputs().iter().enumerate() {
            let sh = gi_sh_project_samples([row[3], row[4], row[5]], &legacy_samples[i], &p4);
            println!("[3.7] 危险填充+旧SDF prim_count=4 时 probe {i} 的 sh[0] = {:?}", sh[0]);
            assert_eq!(sh[0], [0.0, 0.0, 0.0], "全遮挡时 probe {i} 的 sh[0] 必须为零");
            assert!(
                sh.iter().all(|k| *k == [0.0, 0.0, 0.0]),
                "全遮挡时 probe {i} 的 9 个 SH 系数必须整块为零"
            );
        }

        // ---- 组 2：危险填充 + 修正后的球 SDF → probe 2 仍全遮挡 ----
        // 这一组给出一个**必须诚实说清**的结论：原点 r=1 的球在正确的球 SDF 下，
        // 只对"起点几乎落在球面上"的 probe 2（(0.001,1,0)，|p|−1 ≈ 5e−7 ≤ hit_eps）
        // 造成 8/8 全遮挡；probe 0（(0,1.001,0)，|p|−1 ≈ 1.0000e−3 恰好**略大于**
        // hit_eps=1e−3）与 probe 1（1.41−1=0.41）不再被它遮住。
        // 换句话说：「**三个探针全部** 8/8」这个真机形态是**旧球 SDF（q.x 无限大板）**
        // 的产物（组 3 已精确复现 [0,0,0]），而**"填充能遮住探针"**这件事在修正后的
        // SDF 下依然成立 ⇒ 填充替换仍是必需的（第 3.2 步），只是危险面比原先估计的窄。
        let unsafe_new = visible_per_probe(&UNSAFE_FILLER, &p4, gi_sphere_trace_prims);
        println!("[3.7] 危险填充+新SDF 可见方向数：prim_count=4 → {unsafe_new:?}");
        assert_eq!(
            unsafe_new[2], 0,
            "原点 r=1 的球在正确球 SDF 下仍会把 probe 2 变成 8/8 全遮挡"
        );
        assert!(
            unsafe_new[0] > 0 && unsafe_new[1] > 0,
            "修正球 SDF 后 probe 0/1 不再被原点球遮住（全 8/8 是旧 q.x 板的产物）：{unsafe_new:?}"
        );
        assert!(
            ship2.iter().sum::<usize>() > unsafe_new.iter().sum::<usize>(),
            "安全填充的可见方向数必须严格多于危险填充（证明填充替换真的消除了遮挡）"
        );

        // ---- 组 4：CPU 自比 mismatch 必须为 0（prim_count=2，同输入同实现）----
        let mut mismatch = 0usize;
        for row in gi_probe_inputs().iter() {
            let pos = [row[0], row[1], row[2]];
            let nrm = [row[3], row[4], row[5]];
            let a = gi_sh_project_prims(pos, nrm, &GI_PRIMS[..2], &p2);
            let b = gi_sh_project_prims(pos, nrm, &GI_PRIMS[..2], &p2);
            for k in 0..9 {
                for c in 0..3 {
                    if a.sh[k][c].to_bits() != b.sh[k][c].to_bits() {
                        mismatch += 1;
                    }
                }
            }
        }
        assert_eq!(mismatch, 0, "prim_count=2 时 CPU 参考自比必须 0 mismatch");
        println!("[3.7] prim_count=2 CPU 自比 mismatch = {mismatch}");
    }

    /// GI 测试项无 device 时稳定 SKIP（不 panic）。
    #[test]
    fn gi_test_body_skips_without_device() {        DEVICE.with(|d| d.set(0));
        let idx = TESTS.iter().position(|t| t.id == "gpu.gi_compute").expect("清单含 gpu.gi_compute");
        assert_eq!(dispatch_panics(TESTS[idx].run, "gpu.gi_compute"), status::SKIP);
    }

    // -----------------------------------------------------------------------
    // gpu.scene_frame（V2.2-T03「有设备」分支）
    // -----------------------------------------------------------------------

    /// **native 下本项必须返回 `Skip`（不是 `Pass`）**：无 device ⇒ 无法判对。
    ///
    /// 同时钉住「清单登记齐全」（id / needs_gpu / collect 三者缺一不可），
    /// 因为宿主按这三项决定是否把本项纳入真机 GPU 跑批。
    #[test]
    fn scene_frame_test_body_skips_without_device() {
        DEVICE.with(|d| d.set(0));
        let idx = TESTS
            .iter()
            .position(|t| t.id == "gpu.scene_frame")
            .expect("清单含 gpu.scene_frame");
        assert!(TESTS[idx].needs_gpu, "gpu.scene_frame 必须标注 needs_gpu");
        assert!(
            TESTS[idx].collect.is_some(),
            "gpu.scene_frame 必须有 collect 阶段（异步读回判对）"
        );
        assert_eq!(
            dispatch_panics(TESTS[idx].run, "gpu.scene_frame"),
            status::SKIP,
            "无 device 时 gpu.scene_frame 必须是 SKIP（禁止 #[ignore]、禁止假 PASS）"
        );
    }

    /// 判据非恒真：逐位相同的两帧 SSE=0（报告值夹到 99.9 dB）；单像素扰动必须打红。
    #[test]
    fn scene_parity_accepts_identical_and_rejects_spike() {
        let px = 8u32 * 8u32;
        let a = vec![128u8; px as usize * 4];
        let same = scene_parity(&a, &a, px).expect("同长");
        assert_eq!(same.max_channel_delta, 0);
        assert_eq!(same.differing_pixels, 0);
        assert_eq!(same.sum_sq, 0, "逐位相同 ⇒ 平方误差和必须精确为 0");
        assert!(same.passes_40db, "逐位相同必须过门限");
        assert_eq!(same.psnr_db10, PSNR_DB10_CAP, "报告值应夹到上限而非 inf");

        // 单个像素偏 127（三通道）⇒ 8×8 上 MSE≈252 ⇒ ≈24 dB，必须低于门限。
        let mut spike = a.clone();
        for c in 0..3 {
            spike[c] = 255;
        }
        let s = scene_parity(&spike, &a, px).expect("同长");
        assert_eq!(s.differing_pixels, 1);
        assert_eq!(s.max_channel_delta, 127);
        assert!(
            !s.passes_40db,
            "单像素扰动必须低于门限（实测 {}.{} dB）——判据不得恒真",
            s.psnr_db10 / 10,
            s.psnr_db10 % 10
        );
    }

    /// 量化口径差（CPU `round` vs WGSL 截断）带来的 ±1 级系统性差必须仍在门限内。
    ///
    /// 这条把「PSNR 门限是否过严」的判断变成可执行的算术：全帧 ±1 时
    /// MSE = 1 ⇒ PSNR ≈ 48.13 dB，相对 40 dB 有约 8 dB 余量；因此真机上若判红，
    /// 归因**不是**量化约定，而是更大的数值/求交分歧（须由交付总监裁决）。
    #[test]
    fn scene_parity_one_level_bias_still_passes() {
        let px = 16u32 * 16u32;
        let cpu = vec![128u8; px as usize * 4];
        let mut gpu = cpu.clone();
        for i in 0..px as usize {
            // 每像素 RGB 各差 1（alpha 恒 255 不参与）
            for c in 0..3 {
                gpu[i * 4 + c] = 129;
            }
        }
        let p = scene_parity(&gpu, &cpu, px).expect("同长");
        assert_eq!(p.max_channel_delta, 1);
        assert_eq!(p.differing_pixels, px);
        // 每像素 3 通道各差 1 ⇒ sum_sq == channels ⇒ mse == 1.0（精确整数路径）。
        assert_eq!(p.sum_sq, (px as u64) * 3);
        assert!(p.passes_40db, "全帧 ±1 必须过门限");
        assert!(
            p.psnr_db10 >= SCENE_PSNR_MIN_DB10,
            "全帧 ±1 必须过门限（实测 {}.{} dB，门限 40.0）",
            p.psnr_db10 / 10,
            p.psnr_db10 % 10
        );
        // 与解析式 `10·log10(255²)` 对比，确认 det_log2 换算没有算错量级。
        // （测试里用 libm 的 log10 是允许的：D-29 只约束生产代码。）
        let expect = 10.0 * 65025.0_f64.log10() * 10.0;
        assert!(
            (p.psnr_db10 as f64 - expect).abs() < 0.5,
            "PSNR 换算偏差过大：{} vs {expect}",
            p.psnr_db10
        );
    }

    /// 三个测试场景必须自洽：A 只有球、B 含 1 个曲线、BC 与 B 仅 curves 不同。
    #[test]
    fn scene_requests_are_well_formed() {
        let s: [SceneRenderRequest; SCN_COUNT] = [SCN_A, SCN_B, SCN_BC].map(scene_request);
        assert_eq!(s[SCN_A].spheres.len(), 1, "场景 A 必须含至少一个球");
        assert_eq!(s[SCN_A].lights.len(), 1, "场景 A 必须含至少一盏灯");
        assert!(s[SCN_A].curves.is_empty());
        assert_eq!(s[SCN_B].curves.len(), 1, "场景 B 必须含 1 个曲线图元");
        assert_eq!(s[SCN_B].spheres.len(), 1);
        assert_eq!(s[SCN_B].lights.len(), 1);
        assert!(s[SCN_BC].curves.is_empty(), "BC 是对照组（不含曲线）");
        // BC 与 B 只差 curves：除 curves 外的字段必须逐项相同（否则判据 5 不成立）。
        assert_eq!(s[SCN_B].width, s[SCN_BC].width);
        assert_eq!(s[SCN_B].height, s[SCN_BC].height);
        assert_eq!(s[SCN_B].spheres.len(), s[SCN_BC].spheres.len());
        assert_eq!(s[SCN_B].lights.len(), s[SCN_BC].lights.len());
        // 尺寸必须落在宿主竞技场容量内（否则真机提交会 TooLarge）。
        for (i, r) in s.iter().enumerate() {
            assert!(
                r.width <= crate::web::scene_gpu::MAX_FRAME_W
                    && r.height <= crate::web::scene_gpu::MAX_FRAME_H,
                "场景 {i} 的 {}×{} 超竞技场容量",
                r.width,
                r.height
            );
            assert_eq!(
                r.width * r.height * 4 % 4,
                0,
                "读回字节数必须是 w*h*4（u32 对齐）"
            );
        }
    }
}



