//! [`WebGpuBackend`]：在 WebGPU 上承载 D0 的 scan/sort 内核（WGSL compute）。
//!
//! 设计要点（骨架，无 WGSL 运行时）：
//! * WGSL 源码内嵌为 `&'static str`，与 vulkan/metal 后端共享同一份 shader
//!   逻辑源。
//! * 编排逻辑（工作组规模、绑定顺序、dispatch 维度）全部在 Rust 侧计算，
//!   JS 胶水仅做对象转发。
//! * 在非 wasm 目标上，`render_scene` 走 CPU 兜底以保证可编译、可测试；
//!   在 wasm 目标上，若 device 获取失败亦回退 CPU，绝不 panic（harness 要求）。

use crate::scene_abi::{BindingAccess, SceneAbi};
use crate::shaders;
use crate::web::{self, GpuHandle, WebGpuSupport, INVALID_HANDLE};
use crate::{Framebuffer, GpuDevice, SceneRenderRequest};

/// 每工作组线程数（WGSL `@workgroup_size(64)`），用于把元素数换算成
/// dispatch 的工作组数。
pub const WORKGROUP_SIZE: u32 = 64;

/// D0 scan（Blelloch 上扫）单块的配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanConfig {
    /// 参与扫描的元素总数。
    pub count: u32,
    /// 工作组数（= ceil(count / (2*WORKGROUP_SIZE))，单块就地扫描）。
    pub workgroups: u32,
}

impl ScanConfig {
    /// 由元素个数推导出 dispatch 配置。
    pub fn for_count(count: u32) -> Self {
        let per_block = 2 * WORKGROUP_SIZE;
        let workgroups = count.div_ceil(per_block).max(1);
        ScanConfig { count, workgroups }
    }
}

/// WebGPU 后端：把 D0 scan/sort 用 WGSL compute shader 形式承载。
///
/// `device` 为宿主 JS 侧登记的 GPUDevice 句柄；`support` 记录初始化时的
/// 能力探测结果。native 目标下 `support == NotWasmTarget`，一律走 CPU 兜底。
pub struct WebGpuBackend {
    device: GpuHandle,
    support: WebGpuSupport,
    cpu_fallback: crate::CpuRefBackend,
}

impl WebGpuBackend {
    /// 创建一个后端实例：探测 WebGPU、请求 device（失败则为无效句柄）。
    ///
    /// 该构造函数永不 panic——失败时后端自动降级为 CPU 渲染，供 harness
    /// 继续运行其它测试项（要求）。
    pub fn new() -> Self {
        let support = web::detect_support();
        let device = if support == WebGpuSupport::Available {
            // SAFETY: 宿主注入的探测+请求函数；零参数、无副作用申请。
            #[cfg(target_arch = "wasm32")]
            {
                unsafe { web::js::prism_request_device() }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                INVALID_HANDLE
            }
        } else {
            INVALID_HANDLE
        };
        WebGpuBackend {
            device,
            support,
            cpu_fallback: crate::CpuRefBackend,
        }
    }

    /// 当前后端是否已拿到可用的 WebGPU device。
    pub fn is_gpu_ready(&self) -> bool {
        self.support == WebGpuSupport::Available && self.device != INVALID_HANDLE
    }

    /// device 句柄（无效时为 [`INVALID_HANDLE`]）。
    pub fn device_handle(&self) -> GpuHandle {
        self.device
    }

    /// 能力探测结果。
    pub fn support(&self) -> WebGpuSupport {
        self.support
    }

    /// 能力探测摘要（供 harness 报告）。
    pub fn adapter_summary(&self) -> web::AdapterSummary {
        web::adapter_summary(self.device)
    }

    /// device 是否已 lost（native 恒为 false）。
    pub fn device_lost(&self) -> bool {
        if self.device == INVALID_HANDLE {
            return false;
        }
        #[cfg(target_arch = "wasm32")]
        {
            // SAFETY: 宿主注入的查询函数，纯读。
            unsafe { web::js::prism_device_lost(self.device) != 0 }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// 顶层编排：把 D0 的两大内核（前缀和 scan、键值 sort 的计数阶段）
    /// 翻译成 WGSL dispatch 计划。返回按执行顺序排列的内核列表。
    ///
    /// 说明：骨架阶段只产出**dispatch 计划 + shader 源**，真实提交留待
    /// 具备 WGSL 运行时的真机（D-14 移动端验证路径）。
    pub fn plan_scan(&self, cfg: ScanConfig) -> Vec<KernelDispatch> {
        vec![KernelDispatch {
            name: "scan_block",
            entry: "scan_block",
            wgsl: WGSL_SCAN_BLOCK,
            workgroups: (cfg.workgroups, 1, 1),
            buffer_bindings: 2, // input(read) + output(read_write)
        }]
    }

    /// 计算 sort 计数直方图内核的 dispatch 计划。`buckets` 决定直方图容量
    /// （调用方据以分配 hist buffer；dispatch 规模只与 `count` 有关）。
    ///
    /// `shift` 为 LSD 基数排序当前趟的数位位移（0 / 8 / 16 / 24），与 CPU
    /// 参考 [`crate::sort`] 的 `(k >> shift) & mask` 同一口径；host 端用
    /// [`histogram_params_bytes`] 把它写进 uniform 缓冲。
    pub fn plan_sort_histogram(&self, count: u32, buckets: u32, shift: u32) -> Vec<KernelDispatch> {
        // 立即按 WGSL 布局打包一次：dispatch 前即校验 buckets 为 2 的幂，
        // 避免把非法参数带到 bind group 上传阶段才暴露。
        let _ = histogram_params_bytes(count, buckets, shift);
        vec![KernelDispatch {
            name: "histogram",
            entry: "histogram",
            wgsl: WGSL_HISTOGRAM,
            workgroups: (count.div_ceil(WORKGROUP_SIZE).max(1), 1, 1),
            buffer_bindings: 3, // keys(read) + hist(atomic) + params(uniform)
        }]
    }
}

impl Default for WebGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// 一个 WGSL compute 内核的提交描述（零 GPU 依赖的纯数据）。
pub struct KernelDispatch {
    /// 内核名（报告/调试用）。
    pub name: &'static str,
    /// WGSL `@compute` entry point 名。
    pub entry: &'static str,
    /// 内嵌 WGSL 源码。
    pub wgsl: &'static str,
    /// dispatch 的工作组三维规模 `(gx, gy, gz)`。
    pub workgroups: (u32, u32, u32),
    /// storage buffer 绑定数量（决定 bind group 布局）。
    pub buffer_bindings: u32,
}

impl GpuDevice for WebGpuBackend {
    /// 渲染一帧。若 WebGPU 未就绪则降级到 CPU 参考后端。
    ///
    /// 骨架阶段 GPU 路径尚未提交真实 compute（待 WGSL 运行时），因此仅做
    /// 能力判定；判定失败时走 `CpuRefBackend`，保证永不 panic、永有输出。
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer {
        // 真实 GPU 提交路径（需 WGSL 运行时 + 真机）尚未打通：
        // 这里保持 CPU 兜底，符合 D1 骨架"先出 shader 源 + 绑定骨架"的约定。
        self.cpu_fallback.render_scene(req)
    }
}

// ---------------------------------------------------------------------------
// WGSL 内核源码（内嵌；D0 scan/sort 的 GPU 承载）
// ---------------------------------------------------------------------------

/// D0 前缀和（单块 Blelloch 上扫）的 WGSL 源。
///
/// 工作组大小 [`WORKGROUP_SIZE`]，一次处理 `2*WORKGROUP_SIZE` 个元素；
/// 使用共享内存两轮交换避免 bank conflict（经典实现）。
///
/// 必须是 `static` 而非 `const`：`const &str` 会在每个引用点各物化一份副本，
/// 使 `abi::prism_kernel_wgsl(i)` 交给宿主的地址与本模块其它引用点的地址不同
/// （宿主按该地址读到的仍是自己那份内容，逻辑正确但白白多占一份内存，且
/// 任何"两侧地址应相等"的断言都会失败）。
pub static WGSL_SCAN_BLOCK: &str = r#"
// scan_block: single-block exclusive prefix sum (Blelloch up-sweep + down-sweep).
// bindings: 0 => input (read), 1 => output (read_write), 2 => params (uniform)
@group(0) @binding(0) var<storage, read>       input:  array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform>            params: Params;

// params 布局（uniform 地址空间要求结构体大小按 16 字节对齐）。
//
// 【为何是两个 _pad 而不是只补一个】规范上 `struct Params { block: u32, count: u32 }`
// 在 uniform 地址空间里的 size **本来就是 16**（align = roundUp(16, max(4,4)) = 16，
// size = roundUp(16, 8) = 16）——D-14 真机失败的根因**不是** WGSL 只写了 8 字节，
// 而是宿主侧把 uniform 缓冲建成了 8 字节（见 abi.rs 该处注释）。
//
// 但只靠"size 向上取整到 16"这条隐式规则会留下隐患：结构体字面写8 字节、
// 实际占 16 字节，人读代码时按"成员数 × 4"算出 8，与宿主写入长度静默错位。
// 显式补满两个 u32 后，`成员数 × 4 == size == SCAN_PARAMS_BYTES == 16` 四者相等、
// 无隐式尾部填充，两端各说各话的可能性被消除。
// 门禁 `wgsl_lint.mjs` 规则 A3 正是断言这一点（payload == size == 宿主字节数）。
struct Params { block: u32, count: u32, _pad0: u32, _pad1: u32 };

const WG: u32 = 64u;
const N: u32 = 128u;   // = 2 * WG，单块元素数
var<workgroup> temp: array<u32, 128>;

@compute @workgroup_size(64)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let base = params.block * (2u * WG) + tid * 2u;

    // ---- load ----
    // 守卫口径与 up/down-sweep 不同：此处要防的是**尾块越界读**（按元素总数
    // params.count 判定），而非 Blelloch 每轮的活跃线程数；越界槽填 0 参与
    // 归约，保证 exclusive scan 的高位为零填充语义。
    if (base < params.count) {
        temp[tid * 2u] = input[base];
    } else {
        temp[tid * 2u] = 0u;
    }
    if (base + 1u < params.count) {
        temp[tid * 2u + 1u] = input[base + 1u];
    } else {
        temp[tid * 2u + 1u] = 0u;
    }
    // load 的store 与up-sweep 的首次读之间必须同步，否则工作组内读到未初始化
    // 的 temp（数据竞争 ⇒ 结果不确定，违反全链路确定性铁律）。
    workgroupBarrier();

    // up-sweep (reduce)：offset = 1,2,...,64
    var offset: u32 = 1u;
    loop {
        if (offset >= N) { break; }
        let idx = (tid + 1u) * (offset * 2u) - 1u;
        // Blelloch 线程活跃守卫：本轮仅前 N/(2*offset) 个线程参与。
        // offset=1 → 64 线程全活跃；offset≥2 → 逐轮减半。
        if (tid < N / (offset * 2u)) {
            temp[idx] = temp[idx] + temp[idx - offset];
        }
        // 屏障置于守卫**之外**、循环体内：offset 是各线程一致的uniform 量，
        // 故此处控制流仍是 uniform，满足 WGSL 验证层对 workgroupBarrier 的要求。
        workgroupBarrier();
        offset = offset * 2u;
    }

    // clear the last element, then down-sweep
    // 各线程写同一常量 0，无竞争；随后屏障隔开 up/down-sweep 两个阶段。
    temp[N - 1u] = 0u;
    workgroupBarrier();

    // down-sweep (distribute)：offset = 64,32,...,1
    offset = 64u;
    loop {
        if (offset == 0u) { break; }
        let idx = (tid + 1u) * (offset * 2u) - 1u;
        // 与 up-sweep 同一活跃守卫：offset=64 → 仅 tid 0 参与根节点交换。
        if (tid < N / (offset * 2u)) {
            let left = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] = temp[idx] + left;
        }
        workgroupBarrier();
        offset = offset / 2u;
    }

    // write exclusive prefix sums（写回同样按 count 守卫，防尾块越界写）
    if (base < params.count) {
        output[base] = temp[tid * 2u];
    }
    if (base + 1u < params.count) {
        output[base + 1u] = temp[tid * 2u + 1u];
    }
}
"#;

/// D0 sort 计数直方图的 WGSL 源（原子累加）。
///
/// 同 [`WGSL_SCAN_BLOCK`]：用 `static` 而非 `const`，保证全程序唯一实例。
pub static WGSL_HISTOGRAM: &str = r#"
// histogram: bucket keys into an atomic histogram (counting sort phase 1).
// bindings: 0 => keys (read), 1 => hist (atomic<u32>), 2 => params (uniform)
@group(0) @binding(0) var<storage, read>       keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>            params: Params;

// params 布局（uniform 地址空间要求结构体大小 16 字节对齐，故 3×u32 后补
// 1×u32 padding = 16 字节；host 端写入长度必须与此处一致）。
struct Params { count: u32, buckets: u32, shift: u32, _pad: u32 };

@compute @workgroup_size(64)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    // 取数位口径必须与 CPU 参考 sort.rs:31 完全一致：
    //   (k >> shift) & (RADIX_BUCKETS - 1)
    // buckets 恒为 2 的幂，故 mask == buckets - 1；`%` 无法表达 shift>0 的
    // 第 2/3/4 趟数位，会使 GPU 与 CpuRef 排序结果发散。
    let bucket = (keys[i] >> params.shift) & (params.buckets - 1u);
    atomicAdd(&hist[bucket], 1u);
}
"#;

/// 光追场景整帧 kernel 的 WGSL 源（V2.2-A-1 / T03 `scene_frame.wgsl`）。
///
/// 与 `crates/prism-gpu/shaders/scene_frame.wgsl` 是**同一份**源（经
/// `include_str!` 引入，避免与 `SHADERS` 表里的登记副本漂移）。入口 `scene_main`、
/// 工作组 `[8,8,1]`、4 绑定连续前缀 `{0,1,2,3}` = params/spheres/lights/out，
/// 全部以 `crate::scene_abi` 为唯一真源（WGSL 内的 `const` 字面量只是其镜像）。
pub static WGSL_SCENE_FRAME: &str = include_str!("../../shaders/scene_frame.wgsl");

/// [`WGSL_SCAN_BLOCK`] 的 `Params` 结构在 WGSL uniform 地址空间中的字节长度。
///
/// WGSL 规定 uniform 地址空间里的结构体，其**对齐**取 `max(16, max(成员对齐))`，
/// **大小**再向上取整到该对齐（见 WGSL 规范 §"Address space layout constraints"）。
/// 故 `{block, count}` 两个 `u32` 即使只写到第 8 字节，结构体大小仍是 16 字节。
/// host 端建 uniform buffer 时长度必须 ≥ 本常量，否则 `createBindGroup` 的
/// 验证层报 `buffer size ... is less than the minimum binding size`（D-14 真机实测）。
pub const SCAN_PARAMS_BYTES: usize = 16;

/// 把 [`WGSL_SCAN_BLOCK`] 的 uniform 参数按 WGSL 布局打包成小端字节缓冲。
///
/// `block` 为块序号（单块就地扫描时恒为 0），`count` 为参与扫描的元素总数
/// （尾块据此做越界守卫，见 shader 内load 段）。语义与 shader 内
/// `params.block` / `params.count` 逐位一致。
pub fn scan_params_bytes(block: u32, count: u32) -> [u8; SCAN_PARAMS_BYTES] {
    let mut buf = [0u8; SCAN_PARAMS_BYTES];
    buf[0..4].copy_from_slice(&block.to_le_bytes());
    buf[4..8].copy_from_slice(&count.to_le_bytes());
    // [8,16) 是 WGSL 侧 `_pad0` / `_pad1` 两个显式占位成员，必须写 0：
    // 门禁 A3 断言"成员数 × 4 == SCAN_PARAMS_BYTES"，改这里必须同步改 WGSL 声明。
    buf[8..12].copy_from_slice(&0u32.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes());
    buf
}

/// [`WGSL_HISTOGRAM`] 的 `Params` 结构在 WGSL uniform 地址空间中的字节长度。
///
/// WGSL 要求 uniform 结构体成员按 16 字节对齐，故 `{count, buckets, shift}`
/// 三个 `u32` 之后补一个 `u32` 占位，总长16 字节。host 端写缓冲长度必须等于
/// 此常量，否则 bind group 上传的数据与 shader 解释不一致（结果级错误）。
pub const HISTOGRAM_PARAMS_BYTES: usize = 16;

/// 把 [`WGSL_HISTOGRAM`] 的 uniform 参数按 WGSL 布局打包成小端字节缓冲。
///
/// `shift` 为 LSD基数排序当前趟的位移（0 / 8 / 16 / 24），语义与 CPU 参考
/// [`crate::sort`] 的 `(k >> shift) & mask` 完全一致；`buckets` 须为 2 的幂。
///
/// # Panics
/// `buckets` 非 2 的幂时 panic——该值由 [`crate::sort`] 的基数表推导，
/// 不接受调用方传入非法值。
pub fn histogram_params_bytes(count: u32, buckets: u32, shift: u32) -> [u8; HISTOGRAM_PARAMS_BYTES] {
    assert!(buckets.is_power_of_two(), "histogram buckets 必须是 2 的幂（基数排序约定），得到 {buckets}");
    let mut buf = [0u8; HISTOGRAM_PARAMS_BYTES];
    buf[0..4].copy_from_slice(&count.to_le_bytes());
    buf[4..8].copy_from_slice(&buckets.to_le_bytes());
    buf[8..12].copy_from_slice(&shift.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes()); // 16 字节对齐占位
    buf
}

// ---------------------------------------------------------------------------
// 单元测试（native：不触碰 GPU，仅验证计划/配置逻辑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    #[test]
    fn scan_config_rounds_up_to_workgroups() {
        // 64 元素 => 每块 128 => 但 count 只 64 => 1 块；除不尽向上取整。
        assert_eq!(ScanConfig::for_count(64).workgroups, 1);
        assert_eq!(ScanConfig::for_count(0).workgroups, 1, "空输入至少 1 块");
        assert_eq!(ScanConfig::for_count(128).workgroups, 1);
        assert_eq!(ScanConfig::for_count(129).workgroups, 2);
    }

    #[test]
    fn plan_scan_produces_one_dispatch_with_expected_entry() {
        let backend = WebGpuBackend::new();
        let cfg = ScanConfig::for_count(256);
        let plan = backend.plan_scan(cfg);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].entry, "scan_block");
        assert_eq!(plan[0].workgroups, (cfg.workgroups, 1, 1));
        assert!(plan[0].wgsl.contains("@compute"), "WGSL 应含 compute 入口");
    }

    #[test]
    fn plan_sort_histogram_workgroups_match_count() {
        let backend = WebGpuBackend::new();
        let plan = backend.plan_sort_histogram(200, 16, 0);
        assert_eq!(plan.len(), 1);
        // ceil(200/64) = 4 个工作组。
        assert_eq!(plan[0].workgroups, (4, 1, 1));
        assert_eq!(plan[0].entry, "histogram");
    }

    #[test]
    fn native_backend_is_not_gpu_ready_and_falls_back_to_cpu() {
        // native 目标：探测为 NotWasmTarget，后端必然未就绪。
        let mut backend = WebGpuBackend::new();
        assert!(!backend.is_gpu_ready());
        assert_eq!(backend.support(), WebGpuSupport::NotWasmTarget);
        assert!(!backend.device_lost());

        // render_scene 应安全降级到 CPU，并产出与 CpuRefBackend 一致的结果。
        let req = SceneRenderRequest {
            width: 16,
            height: 9,
            camera: crate::CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 16.0 / 9.0,
            },
            lights: vec![crate::LightReq {
                direction: Vec3::new(0.0, 0.0, -1.0),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 1.0,
            }],
            spheres: vec![crate::SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                albedo: Vec3::new(0.8, 0.2, 0.2),
                roughness: 0.4,
                metalness: 0.1,
            }],
            aesthetic: crate::AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        };
        let gpu_fb = backend.render_scene(&req);
        let cpu_fb = crate::CpuRefBackend.render_scene(&req);
        assert_eq!(gpu_fb.width, cpu_fb.width);
        assert_eq!(gpu_fb.pixels, cpu_fb.pixels, "降级路径应与 CPU 参考逐字节一致");
    }

    #[test]
    fn webgpu_wgsl_sources_pass_structure_check() {
        // 复用 D2/D3/D4 的结构自检：入口存在、括号配平、无 libm 超越函数。
        for (name, src) in [("scan_block", WGSL_SCAN_BLOCK), ("histogram", WGSL_HISTOGRAM)] {
            let chk = crate::shaders::check_shader(name, src);
            assert!(chk.is_pass(), "WGSL `{}` 结构自检未通过: {:?}", name, chk);
        }
    }

    #[test]
    fn uniform_params_bytes_are_sixteen_byte_aligned() {
        // WGSL uniform 地址空间：结构体 align = roundUp(16, max(成员 align))，
        // size = roundUp(align, 末成员末尾)。两个结构体都只写 u32/标量成员，
        // 故 size 必为 16 的倍数——这是 createBindGroup 的硬性验证条件。
        for (name, bytes) in [
            ("scan_block", SCAN_PARAMS_BYTES),
            ("histogram", HISTOGRAM_PARAMS_BYTES),
        ] {
            assert_eq!(bytes % 16, 0, "{name} 的 params 常量 {bytes} 未按 16 字节对齐");
            assert!(bytes >= 16, "{name} 的 params 常量 {bytes} 小于 WGSL 最小结构体尺寸 16");
        }
    }

    #[test]
    fn scan_params_bytes_layout_matches_wgsl_declaration() {
        // WGSL 侧 `struct Params { block: u32, count: u32, _pad0: u32, _pad1: u32 }`：
        // 偏移 0/4/8/12，末成员占 [12,16)，payload 16 == size 16（无隐式填充）。
        let b = scan_params_bytes(0, 128);
        let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        assert_eq!(b.len(), SCAN_PARAMS_BYTES);
        assert_eq!(rd(0), 0, "block 应在偏移 0");
        assert_eq!(rd(4), 128, "count 应在偏移 4");
        assert_eq!(rd(8), 0, "_pad0 占位应为 0");
        assert_eq!(rd(12), 0, "_pad1 占位应为 0");
        // block/count 为任意 u32 都应无损往返（不做溢出假设）。
        let b2 = scan_params_bytes(u32::MAX, 0);
        let rd2 = |o: usize| u32::from_le_bytes([b2[o], b2[o + 1], b2[o + 2], b2[o + 3]]);
        assert_eq!(rd2(0), u32::MAX);
        assert_eq!(rd2(4), 0);
    }

    #[test]
    fn scan_wgsl_declares_padding_members() {
        // 防回归：结构体一旦漏掉显式 _pad，size 仍会由 align 向上取整到 16（规范上
        // 不算错），但"成员数 × 4"与实际字节数就静默错位了——这正是 D-14 真机把
        // 宿主缓冲建成 8 字节的思维盲区。故显式断言两个占位成员都在。
        // 改这里必须同步 wgsl_lint.mjs 规则 A3（payload == size == 宿主字节数）。
        assert!(
            WGSL_SCAN_BLOCK.contains("struct Params { block: u32, count: u32, _pad0: u32, _pad1: u32 }"),
            "scan_block 的 Params 必须声明 _pad0/_pad1 占位，与 SCAN_PARAMS_BYTES=16 对齐"
        );
        assert!(
            WGSL_HISTOGRAM.contains("struct Params { count: u32, buckets: u32, shift: u32, _pad: u32 }"),
            "histogram 的 Params 必须声明 _pad 占位"
        );
    }

    #[test]
    fn uniform_structs_have_no_implicit_tail_padding() {
        // WGSL 侧两个 Params 的"有效载荷"必须已写满 16 字节：末成员偏移 + 大小 == 16。
        // 手工核算（与 wgsl_lint.mjs 规则 A3 同一判据，Rust 侧留一份可独立运行的副本）：
        //   scan_block: block@0 count@4 _pad0@8 _pad1@12 → 末成员末偏移 = 16 ✓
        //   histogram:  count@0 buckets@4 shift@8 _pad@12 → 末成员末偏移 = 16 ✓
        // 二者成员数均为 4，4 × 4B = 16B == 各自 *_PARAMS_BYTES。
        //
        // 【必须先剥注释】结构体的说明注释里**本身**就引用了
        // `struct Params { block: u32, count: u32 }`（解释历史缺陷的那段）。
        // 直接 `split_once("struct Params {")` 会命中注释里那句，把 2 个成员
        // 当成真声明 → 本测试自己就先误报了。注释剥离与 wgsl_lint.mjs 同理。
        for (name, wgsl, bytes) in [
            ("scan_block", WGSL_SCAN_BLOCK, SCAN_PARAMS_BYTES),
            ("histogram", WGSL_HISTOGRAM, HISTOGRAM_PARAMS_BYTES),
        ] {
            let code: String = wgsl
                .lines()
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n");
            let members = code
                .split_once("struct Params {")
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(body, _)| body)
                .unwrap_or("")
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .count();
            assert_eq!(
                members * 4,
                bytes,
                "{name}: 成员数 {members} × 4B != {bytes}B（WGSL 声明与宿主常量脱钩）"
            );
        }
    }
}
