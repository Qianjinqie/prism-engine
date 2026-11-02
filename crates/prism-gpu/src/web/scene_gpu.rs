//! 场景整帧的**真 WebGPU dispatch**（V2.2-T03 / A-1「有设备」分支的唯一实现）。
//!
//! # 这一层补的是什么
//!
//! V2.2-T03 的 PRD 验收第 3 条要求「**双分支可判**」：有设备返 `w*h*4` 字节、
//! 无设备 CPU 兜底且逐字节一致。此前 [`super::device::WebGpuBackend::render_scene`]
//! 只算出 [`scene_dispatch_plan`](super::device::scene_dispatch_plan)（入口 / 工作组 /
//! 绑定 / 工作组数）就**直接**回退 CPU，从未真正提交过 compute —— 「有设备」那一支
//! 是空的。本模块把那一支补齐：用既有的 14 个 JS 胶水函数，按
//! [`crate::scene_abi`] 的单一真源建 6 个缓冲、建 bind group、真 dispatch、
//! 真读回。
//!
//! # 一次提交做什么（全部走既有 14 个胶水，**不新增任何导入**）
//!
//! 1. `prism_create_shader_module` —— 源为
//!    [`WGSL_SCENE_FRAME`](super::device::WGSL_SCENE_FRAME)（`include_str!` 内嵌，
//!    与 `shaders/scene_frame.wgsl` 是同一份，不复制文本）。
//! 2. `prism_create_compute_pipeline` —— 入口 [`SceneAbi::ENTRY`]。工作组尺寸由
//!    WGSL 的 `@workgroup_size` 声明，与 [`SceneAbi::WORKGROUP`] 同源锚定；宿主侧
//!    只需按同源值算出**工作组个数**（[`scene_dispatch_plan`] 的 `groups`）。
//! 3. 6 个 buffer，顺序**严格**等于 [`SceneAbi::bindings`]：
//!    `params / spheres / lights / triangles / nodes / out`。尺寸一律由
//!    `scene_abi` 的槽位数派生（`PARAMS_WORDS_V2` / `SPHERE_SLOTS` /
//!    `LIGHT_SLOTS` / `TRIANGLE_SLOTS` / `NODE_SLOTS`），**本文件不写字面量**。
//! 4. `prism_create_bind_group`（6 entries，第 `i` 个句柄 → `binding: i`）。
//! 5. `prism_dispatch`（工作组数取 `scene_dispatch_plan(req).groups`）。
//! 6. `prism_wait_idle` → `prism_read_buffer` → `prism_copy_out` 读回 `w*h*4`。
//!
//! 任一步返回 0 ⇒ 转 [`SceneDispatchError`]（带步骤名 + 绑定号 + 胶水返回码），
//! **绝不 panic**。
//!
//! # 为什么读回是「登记」而不是「立即拿到」（诚实说明）
//!
//! WebGPU 没有阻塞式读回：`GPUBuffer.mapAsync` 是 Promise，而 WASM 导出/导入
//! 只能是同步函数。既有胶水因此把 `prism_copy_out` 设计成**登记**——真正
//! `mapAsync` + 写回线性内存由宿主 `settle()` 执行。所以：
//!
//! * [`submit_scene_frame`] 只保证「已提交 + 读回已登记」，像素在宿主 `settle()`
//!   之后才落进本模块的定长竞技场；
//! * [`with_settled_pixels`] 只在**已落盘**时交出切片；
//! * [`dispatch_scene_frame`] 是给 [`WebGpuBackend::render_scene`]（同步签名）用的
//!   门面：它**提交本帧**并返回**上一帧已落盘**的像素（同尺寸），首次调用必然
//!   无可返回像素 ⇒ 调用方按 CPU 兜底。这条「一帧延迟」是浏览器读回语义决定的，
//!   不是实现偷懒；逐位/统计一致性由 ABI 测试项 `gpu.scene_frame` 的 collect 阶段
//!   在宿主 `settle()` 之后判对（那才是**有设备**分支的真实证据）。
//!
//! # 落盘判定为什么是可靠的（不是启发式）
//!
//! 像素区在登记读回前被填满哨兵 [`CANARY`]；宿主 `settle()` 里用
//! `Uint8Array.set` **一次性**覆盖整段。故：
//!
//! * 未落盘 ⇒ 整段仍是哨兵 ⇒ 判「未就绪」（**正确**）；
//! * 已落盘 ⇒ 整段是真像素；若真帧恰好每字节都等于哨兵，则被判「未就绪」
//!   （**保守误判**，方向安全：只会多走一次 CPU 兜底，绝不会把未落盘的槽当真帧）。
//!
//! 也就是说误判**只可能朝保守方向**发生，这是本判据可以硬用的原因。
//!
//! # 已知边界
//!
//! * 竞技场是定长数组（地址必须稳定——宿主在 `settle()` 时才写入，`Vec` 会被
//!   后续调用搬走），故帧尺寸/实体数有上限（见 `MAX_*`）。超限如实返回
//!   [`SceneDispatchError::TooLarge`]，调用方回退 CPU。
//! * 14 个胶水里没有 `destroyBuffer`，故每次提交创建的 buffer 由浏览器 GC 回收，
//!   本模块**缓存** shader module + pipeline（最贵的那一步），避免每帧重编译 WGSL。

use core::cell::RefCell;

use crate::scene_abi::{
    SceneAbi, SceneAbiProfile, EnvCdf, InstanceReq, TextureUpload, INSTANCE_SLOTS, LIGHT_SLOTS,
    MAX_V3_TEXTURES, NODE_SLOTS, PARAMS_WORDS_V2, PARAMS_WORDS_V3, SPHERE_SLOTS, SPHERE_SLOTS_V3,
    TRIANGLE_SLOTS,
};
use crate::web::abi::gpu;
use crate::web::abi::usage;
use crate::web::device::{
    scene_dispatch_plan, scene_dispatch_plan_v3, WGSL_SCENE_FRAME, WGSL_SCENE_FRAME_V3,
};
use crate::web::GpuHandle;
use crate::SceneRenderRequest;

// ---------------------------------------------------------------------------
// 容量与常量（全部由 scene_abi 派生，本模块不写布局字面量）
// ---------------------------------------------------------------------------

/// 单帧最大宽度（像素）。竞技场定长（地址必须稳定，见模块文档），故有上限；
/// 超限如实返回 [`SceneDispatchError::TooLarge`]，调用方（`render_scene`）回退 CPU。
///
/// 取 32 是**手机内存友好**的工程取舍：`gpu.scene_frame` 的证据场景只要 8×8 / 16×16，
/// 而每帧像素区按 MAX 尺寸定长分配（10 槽 × 4 KiB = 40 KiB 常驻），比 64×64 省 120 KiB
/// 常驻内存。注意：这块是 `.bss`，**不占 wasm 下载体积**（实测把竞技场从 238 KiB 缩到
/// 40 KiB，产物字节数只差 9 B）。
pub const MAX_FRAME_W: u32 = 32;
/// 单帧最大高度（像素）。
pub const MAX_FRAME_H: u32 = 32;
/// 单帧最大像素字节数（RGBA8 = `w*h*4`）。
pub const MAX_FRAME_PIXELS: usize = (MAX_FRAME_W as usize) * (MAX_FRAME_H as usize) * 4;
/// 场景球体数上限（`gpu.scene_frame` 的测试场景远小于此）。
pub const MAX_SPHERES: usize = 8;
/// 场景光源数上限。
pub const MAX_LIGHTS: usize = 4;
/// 曲线三角化后的三角形数上限。
pub const MAX_TRIS: usize = 512;
/// 扁平 BVH 节点数上限。
pub const MAX_NODES: usize = 1024;
/// 帧槽位数。
///
/// 一次 `gpu.scene_frame` run 提交 5 帧（2 场景 × 双跑 + 1 个无曲线对照），这里留
/// **两轮**的余量：宿主可能并发跑两轮 run（`boot()` 自动跑一次 + 用户点「全部运行」
/// 再跑一次），而 wasm 侧无法区分「哪一轮的 collect」。留 10 格后两轮各自独立占槽，
/// 互不覆写；槽位耗尽的极端情况下如实报 [`SceneDispatchError::NoFreeFrameSlot`]。
pub const FRAME_SLOTS: usize = 10;

/// WebGPU 拒绝 `size == 0` 的 buffer，故空输入（无曲线 / 无灯）也要给最小合法长度。
///
/// `array<u32>` 的最小可绑定尺寸是 1 个元素 = 4 字节。
pub const MIN_BUF_BYTES: u32 = 4;

/// 像素区「尚未落盘」的哨兵字节（详见模块文档的可靠性论证）。
pub const CANARY: u8 = 0xA5;

/// 参数块字节数（V2 剖面 = `PARAMS_WORDS_V2` 个字）。
const PARAMS_BYTES: usize = PARAMS_WORDS_V2 * 4;
/// 球体缓冲竞技场字节数。
const SPHERE_BYTES: usize = SPHERE_SLOTS as usize * 4 * MAX_SPHERES;
/// 光源缓冲竞技场字节数。
const LIGHT_BYTES: usize = LIGHT_SLOTS as usize * 4 * MAX_LIGHTS;
/// 三角形缓冲竞技场字节数。
const TRI_BYTES: usize = TRIANGLE_SLOTS as usize * 4 * MAX_TRIS;
/// 节点缓冲竞技场字节数。
const NODE_BYTES: usize = NODE_SLOTS as usize * 4 * MAX_NODES;
/// `prism_create_bind_group` 的句柄数组容量（V3 剖面 9 个，留余量）。
const HANDLE_CAP: usize = 16;
/// `prism_dispatch` 的 bind group 数组容量（本路径恒为 1 个 group）。
const BG_CAP: usize = 1;

/// V3 参数块字节数（V3 剖面 = `PARAMS_WORDS_V3` 个字，比 V2 多 2 字）。
const V3_PARAMS_BYTES: usize = PARAMS_WORDS_V3 * 4;
/// V3 球体缓冲竞技场字节数（每球 [`SPHERE_SLOTS_V3`] 槽 = 48 B）。
const V3_SPHERE_BYTES: usize = SPHERE_SLOTS_V3 as usize * 4 * MAX_SPHERES;
/// V3 纹素流竞技场字节数（`array<vec4<f32>>`，16 B/纹素）。
///
/// 覆盖台账与对拍测试用到的纹理规模（单/多张 8×8～64×64、含 mip 链），
/// 留 256 KiB 余量；`.bss` 零初始化，不占 wasm 下载体积。
const TEXEL_ARENA_BYTES: usize = 256 * 1024;
/// V3 纹理元数据竞技场字节数（`array<vec4<u32>>`，32 B/条）。
const TEXMETA_ARENA_BYTES: usize = 8 * 1024;
/// V3 环境 CDF 竞技场字节数（`array<u32>`，`(W+1)·H·4` B）。
const CDF_ARENA_BYTES: usize = 256 * 1024;
/// V3 实例记录数上限（每实例 [`INSTANCE_SLOTS`] 槽 = 112 B）。
///
/// 取 256：`.bss` 里 28 KiB，与 [`MAX_TRIS`] / [`MAX_NODES`] 同量级；浏览器
/// 侧证据场景（[`MAX_SPHERES`] 个球 / 一条曲线的实例副本）远小于此。
/// **"1 万实例 BLAS 只建一次"的判据不在本模块**——那条判据量的是 CPU 侧
/// `prism_gpu::instance::build_instance_set` 的构建次数与字节数，不经 WebGPU
/// 竞技场（见 `vulkan/v3_frame.rs` 的实例测试）。超本上限如实报
/// [`SceneDispatchError::TooLarge`]，不回退、不静默截断。
pub const MAX_INSTANCES: usize = 256;
/// V3 实例 TLAS 节点数上限（每节点 [`NODE_SLOTS`] 槽 = 48 B）。
///
/// TLAS 的最坏节点数是 `2n-1`（`n` 个叶子全在最后一层），故取
/// `2 * MAX_INSTANCES` 即可覆盖任何形状的实例树。
pub const MAX_INSTANCE_NODES: usize = 2 * MAX_INSTANCES;
/// V3 实例缓冲竞技场字节数（`array<u32>`，28 槽/条）。
const INSTANCE_ARENA_BYTES: usize = INSTANCE_SLOTS as usize * 4 * MAX_INSTANCES;
/// V3 实例 TLAS 节点竞技场字节数（`array<u32>`，12 槽/条 = 48 B）。
const INSTANCE_NODE_ARENA_BYTES: usize = NODE_SLOTS as usize * 4 * MAX_INSTANCE_NODES;

// ---------------------------------------------------------------------------
// 错误
// ---------------------------------------------------------------------------

/// 一次场景 dispatch 的失败原因（**永不 panic**：所有失败都走这里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneDispatchError {
    /// 非 wasm 目标：没有浏览器胶水（native 的 `gpu::*` 全是返回 0 的桩）。
    NotWasmTarget,
    /// 请求尺寸为 0：`w*h*4 == 0` 的 buffer 在 WebGPU 里非法，且没有任何像素可判。
    EmptyFrame,
    /// 超出竞技场容量（`need` = 实际需要，`cap` = 竞技场上限）。
    TooLarge {
        /// 超限对象名（`"width"` / `"height"` / `"spheres"` / `"triangles"` …）。
        what: &'static str,
        /// 实际需要量。
        need: u32,
        /// 竞技场容量。
        cap: u32,
    },
    /// 某一步胶水返回 0。
    Step {
        /// 步骤名（`"create_shader_module"` / `"create_buffer"` / `"dispatch"` …）。
        step: &'static str,
        /// 出错绑定号（无绑定语义的步骤为 [`NO_BINDING`]）。
        binding: u32,
        /// 胶水原样返回码（失败约定为 0；保留原值便于诊断）。
        code: u32,
    },
    /// 所有帧槽都还在等宿主落盘（既没有空槽，也没有可回收的已落盘槽）。
    NoFreeFrameSlot,
    /// 本帧已提交，但读回尚未落盘（宿主 `settle()` 之后才可用）。
    NotSettledYet,
}

/// [`SceneDispatchError::Step`] 中「无绑定语义」的占位值。
pub const NO_BINDING: u32 = u32::MAX;

impl SceneDispatchError {
    /// 构造「某步胶水返回 0」的错误。
    const fn step(step: &'static str, binding: u32) -> Self {
        SceneDispatchError::Step { step, binding, code: 0 }
    }
}

impl core::fmt::Display for SceneDispatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SceneDispatchError::NotWasmTarget => {
                write!(f, "非 wasm 目标：场景 dispatch 需要浏览器 WebGPU 胶水")
            }
            SceneDispatchError::EmptyFrame => {
                write!(f, "请求尺寸为 0（w*h*4 == 0 的 buffer 在 WebGPU 里非法）")
            }
            SceneDispatchError::TooLarge { what, need, cap } => {
                write!(f, "{what} 超竞技场容量：需要 {need}，上限 {cap}")
            }
            SceneDispatchError::Step { step, binding, code } => {
                if *binding == NO_BINDING {
                    write!(f, "{step} 失败（胶水返回 {code}）")
                } else {
                    write!(f, "{step} 失败于 binding({binding})（胶水返回 {code}）")
                }
            }
            SceneDispatchError::NoFreeFrameSlot => {
                write!(f, "所有帧槽都在等宿主 settle（无空槽可回收）")
            }
            SceneDispatchError::NotSettledYet => {
                write!(f, "读回尚未落盘（宿主 settle() 之后才可读）")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 帧句柄与竞技场
// ---------------------------------------------------------------------------

/// 一次 GPU 提交的帧句柄（**不含像素本体**：像素在竞技场里，地址稳定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneFrame {
    /// 竞技场槽位号。
    slot: usize,
    /// 帧宽（像素）。
    width: u32,
    /// 帧高（像素）。
    height: u32,
    /// 提交时向 `prism_read_buffer` / `prism_copy_out` 请求的读回字节数。
    ///
    /// 独立记录一次是**判据需要**：PRD 验收 3 的「有设备返 `w*h*4`」必须能
    /// 核对「我们请求的字节数」与「`w*h*4`」一致，而不是只核对返回值长度。
    read_len: u32,
}

impl SceneFrame {
    /// 槽位号（诊断用）。
    pub fn slot(&self) -> usize {
        self.slot
    }
    /// 帧宽。
    pub fn width(&self) -> u32 {
        self.width
    }
    /// 帧高。
    pub fn height(&self) -> u32 {
        self.height
    }
    /// 本帧向胶水请求的读回字节数（应为 `w*h*4`）。
    pub fn read_len(&self) -> u32 {
        self.read_len
    }
}

/// 帧槽的归属者：`gpu.scene_frame`（测试项）与 `render_scene`（设备门面）互不抢占。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// 由 ABI 测试项提交（collect 阶段读取，**不消费**）。
    Test,
    /// 由 `WebGpuBackend::render_scene` 提交（`dispatch_scene_frame` **消费**式取用）。
    Render,
}

/// 帧槽元数据（不含 16 KB 像素区，故可 `Copy`，便于 `const` 初始化）。
#[derive(Debug, Clone, Copy)]
struct FrameMeta {
    /// 槽内是否持有一帧未消费的提交。
    used: bool,
    /// 归属者。
    owner: Owner,
    /// 帧宽。
    width: u32,
    /// 帧高。
    height: u32,
    /// 请求的读回字节数。
    read_len: u32,
}

impl FrameMeta {
    /// 空槽（`const` 可用）。
    const EMPTY: FrameMeta = FrameMeta {
        used: false,
        owner: Owner::Test,
        width: 0,
        height: 0,
        read_len: 0,
    };
}

/// 定长竞技场：所有交给胶水的地址都指向这里（wasm 生命周期内**地址稳定**）。
///
/// # 为什么大块是 `Box<[u8]>` 而不是 `[u8; N]`
///
/// 这些段落加起来 ~660 KiB（纹素 256 KiB + CDF 256 KiB + BH 节点 48 KiB + 像素
/// 40 KiB + 两块实例 52 KiB + …）。写成**内联数组**时，`SceneArena::new()` 的
/// 返回值会先在调用栈上构造再 move 进 TLS 槽——debug 下不折叠，光这一下就要
/// 两倍 660 KiB，加上测试线程默认 2 MiB 栈，加两块实例缓冲就直接 `SIGABRT`
/// （实测：`web::scene_gpu::tests::allocation_refuses_to_overwrite_pending_slots`
/// 栈溢出）。`Box<[u8]>` 把内容放堆上，move `SceneArena` 只搬指针，栈占用恒定。
///
/// # 地址稳定性
///
/// `Box<[u8]>` 的内存在**首次分配后**不迁移（本结构只会整体 move，不 realloc），
/// 故 [`addr_of`] 取到的地址在 wasm 生命周期内恒定——这与胶水的使用方式相符
/// （胶水读该地址时数据已写完，中间不发生任何再分配）。
struct SceneArena {
    /// 参数块（`PARAMS_WORDS_V2 × 4` 字节，binding 0）。
    params: Box<[u8]>,
    /// 球体缓冲（binding 1）。
    spheres: Box<[u8]>,
    /// 光源缓冲（binding 2）。
    lights: Box<[u8]>,
    /// 三角形缓冲（binding 3）。
    tris: Box<[u8]>,
    /// 扁平 BVH 节点缓冲（binding 4）。
    nodes: Box<[u8]>,
    /// V3 参数块（`PARAMS_WORDS_V3 × 4` 字节，比 V2 多 2 字）。
    ///
    /// 与 `params` 分开：V2 的 `EncodedScene.params` 类型是 `[u8; PARAMS_BYTES]`，
    /// 直接复用 `params` 会因长度不符编译失败；V3 的 `encode_v3` 产出更长参数块，
    /// 故独占一段，互不覆写（V2/V3 不在同一测试里并发跑）。
    params_v3: Box<[u8]>,
    /// V3 球体缓冲（每球 48 B，binding 1）。
    spheres_v3: Box<[u8]>,
    /// V3 纹素流（`array<vec4<f32>>`，binding 5）。
    texels: Box<[u8]>,
    /// V3 纹理元数据（`array<vec4<u32>>`，32 B/条，binding 6）。
    texmeta: Box<[u8]>,
    /// V3 环境 CDF（`array<u32>`，binding 7）。
    cdf: Box<[u8]>,
    /// V3 实例记录（`array<u32>`，28 槽 = 112 B/条，binding 8）。
    instances: Box<[u8]>,
    /// V3 实例 TLAS 节点（`array<u32>`，12 槽 = 48 B/条，binding 9）。
    instance_nodes: Box<[u8]>,
    /// `prism_create_bind_group` 的句柄数组落点。
    handles: [u32; HANDLE_CAP],
    /// `prism_dispatch` 的 bind group 句柄数组落点。
    bgs: [u32; BG_CAP],
    /// 各帧像素区（宿主 `settle()` 时写入）。
    pixels: [[u8; MAX_FRAME_PIXELS]; FRAME_SLOTS],
    /// 各帧元数据。
    meta: [FrameMeta; FRAME_SLOTS],
}

impl SceneArena {
    /// 零初始化（大块走堆：见结构体文档的栈占用论证）。
    ///
    /// 不是 `const fn`：`vec![0u8; N].into_boxed_slice()` 是运行期分配。
    /// `thread_local!` 因此用惰性初始化（首次 `ARENA.with` 时才分配），
    /// 对 wasm 产物体积无影响（堆分配，不进 `.bss`）。
    fn new() -> Self {
        SceneArena {
            params: vec![0u8; PARAMS_BYTES].into_boxed_slice(),
            spheres: vec![0u8; SPHERE_BYTES].into_boxed_slice(),
            lights: vec![0u8; LIGHT_BYTES].into_boxed_slice(),
            tris: vec![0u8; TRI_BYTES].into_boxed_slice(),
            nodes: vec![0u8; NODE_BYTES].into_boxed_slice(),
            params_v3: vec![0u8; V3_PARAMS_BYTES].into_boxed_slice(),
            spheres_v3: vec![0u8; V3_SPHERE_BYTES].into_boxed_slice(),
            texels: vec![0u8; TEXEL_ARENA_BYTES].into_boxed_slice(),
            texmeta: vec![0u8; TEXMETA_ARENA_BYTES].into_boxed_slice(),
            cdf: vec![0u8; CDF_ARENA_BYTES].into_boxed_slice(),
            instances: vec![0u8; INSTANCE_ARENA_BYTES].into_boxed_slice(),
            instance_nodes: vec![0u8; INSTANCE_NODE_ARENA_BYTES].into_boxed_slice(),
            handles: [0u32; HANDLE_CAP],
            bgs: [0u32; BG_CAP],
            pixels: [[0u8; MAX_FRAME_PIXELS]; FRAME_SLOTS],
            meta: [FrameMeta::EMPTY; FRAME_SLOTS],
        }
    }
}

/// pipeline 缓存：(device, pipeline) 二元组。
///
/// WGSL 编译 + pipeline 创建是一次提交里最贵的一步，且 `layout:"auto"` 的
/// pipeline 与具体 buffer 尺寸无关（布局由 shader 的绑定声明决定），故按 device
/// 缓存可安全复用。换 device（刷新页面/重建后端）即重编译。
///
/// 不缓存 shader module 句柄：宿主把每个 GPU 对象登记在 `Map<u32, GPUObject>` 里，
/// 句柄一旦登记即终身有效（14 个胶水里没有 destroy），故 module 不会因为本函数
/// 不再持有句柄而失效——它只被 pipeline 引用。
#[derive(Debug, Clone, Copy)]
struct PipelineCache {
    /// 所属 device 句柄。
    device: GpuHandle,
    /// compute pipeline 句柄（`0` = 未建）。
    pipeline: GpuHandle,
}

impl PipelineCache {
    /// 空缓存。
    const EMPTY: PipelineCache = PipelineCache { device: 0, pipeline: 0 };
}

thread_local! {
    /// 场景 dispatch 的定长竞技场（惰性初始化：大块在堆上，见 [`SceneArena`]）。
    static ARENA: RefCell<SceneArena> = RefCell::new(SceneArena::new());
    /// pipeline 缓存（省掉每帧的 WGSL 编译）。
    static PIPE: RefCell<PipelineCache> = const { RefCell::new(PipelineCache::EMPTY) };
    /// V3 十一绑定 kernel 的 pipeline 缓存（与 [`PIPE`] 独立：源/绑定布局不同，
    /// 共用缓存会让 V2 pipeline 被错用于 V3 派发）。
    static PIPE_V3: RefCell<PipelineCache> = const { RefCell::new(PipelineCache::EMPTY) };
    /// 本线程待结算的 GPU 错误 scope 数（由 run/collect 阶段成对消费）。
    static ERR_SCOPES: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// 像素区是否仍是哨兵（⇒ 宿主尚未落盘）。
fn is_canary(px: &[u8]) -> bool {
    !px.is_empty() && px.iter().all(|&b| b == CANARY)
}

/// 取切片在线性内存里的地址（wasm32 指针宽 32 位；native 上仅供断言/日志用）。
fn addr_of(bytes: &[u8]) -> u32 {
    bytes.as_ptr() as usize as u32
}

/// 由实际字节数算 buffer 尺寸（空输入给 [`MIN_BUF_BYTES`]，绝不建 0 长 buffer）。
fn buf_size(len: usize) -> u32 {
    if len == 0 {
        MIN_BUF_BYTES
    } else {
        len as u32
    }
}

/// 登记一个待结算的 GPU 错误 scope（`create_shader_module` / `prism_dispatch` 各 1 个）。
fn note_err_scope() {
    ERR_SCOPES.with(|c| c.set(c.get().saturating_add(1)));
}

/// 取走本线程累计的错误 scope 数（由 ABI 测试项的 collect 阶段消费）。
///
/// 宿主 `prism_pop_error` 是**全局 FIFO**，只能同步 pop；而错误码由 `settle()`
/// 里的 Promise 异步 push，故判定必须在 collect 阶段做（见 `abi::check_errors`
/// 的长注释）。本函数让场景路径也能复用同一套「登记 → collect 结算」纪律。
pub(crate) fn take_err_scopes() -> u32 {
    ERR_SCOPES.with(|c| {
        let n = c.get();
        c.set(0);
        n
    })
}

// ---------------------------------------------------------------------------
// pipeline（缓存）
// ---------------------------------------------------------------------------

/// 确保拿到该 device 上可用的 compute pipeline（首次提交时真编译 WGSL）。
fn ensure_pipeline(device: GpuHandle) -> Result<GpuHandle, SceneDispatchError> {
    let cached = PIPE.with(|p| {
        let c = *p.borrow();
        if c.device == device && c.pipeline != 0 {
            Some(c.pipeline)
        } else {
            None
        }
    });
    if let Some(pipeline) = cached {
        return Ok(pipeline);
    }
    let src = WGSL_SCENE_FRAME;
    let module = gpu::create_shader_module(device, addr_of(src.as_bytes()), src.len() as u32);
    if module == 0 {
        return Err(SceneDispatchError::step("create_shader_module", NO_BINDING));
    }
    // 宿主在 create_shader_module 里 push 了一个 `getCompilationInfo()` 的 promise，
    // 编译错误会经 pendingErrors 回传 ⇒ 此处登记 1 个待结算槽位。
    note_err_scope();
    let entry = SceneAbi::ENTRY;
    let pipeline = gpu::create_compute_pipeline(
        device,
        module,
        addr_of(entry.as_bytes()),
        entry.len() as u32,
    );
    if pipeline == 0 {
        return Err(SceneDispatchError::step("create_compute_pipeline", NO_BINDING));
    }
    PIPE.with(|p| {
        *p.borrow_mut() = PipelineCache { device, pipeline };
    });
    Ok(pipeline)
}

/// V3 十一绑定 kernel 的 pipeline 缓存（与 [`ensure_pipeline`] 独立，因为两者源不同、
/// 绑定布局不同，共用缓存会导致 V2 的 6 绑定 pipeline 被错用于 V3 的 11 绑定派发）。
fn ensure_pipeline_v3(device: GpuHandle) -> Result<GpuHandle, SceneDispatchError> {
    let cached = PIPE_V3.with(|p| {
        let c = *p.borrow();
        if c.device == device && c.pipeline != 0 {
            Some(c.pipeline)
        } else {
            None
        }
    });
    if let Some(pipeline) = cached {
        return Ok(pipeline);
    }
    let src = WGSL_SCENE_FRAME_V3;
    let module = gpu::create_shader_module(device, addr_of(src.as_bytes()), src.len() as u32);
    if module == 0 {
        return Err(SceneDispatchError::step("create_shader_module", NO_BINDING));
    }
    // 宿主在 create_shader_module 里 push 了一个 `getCompilationInfo()` 的 promise，
    // 编译错误会经 pendingErrors 回传 ⇒ 此处登记 1 个待结算槽位。
    note_err_scope();
    let entry = SceneAbi::ENTRY;
    let pipeline = gpu::create_compute_pipeline(
        device,
        module,
        addr_of(entry.as_bytes()),
        entry.len() as u32,
    );
    if pipeline == 0 {
        return Err(SceneDispatchError::step("create_compute_pipeline", NO_BINDING));
    }
    PIPE_V3.with(|p| {
        *p.borrow_mut() = PipelineCache { device, pipeline };
    });
    Ok(pipeline)
}

// ---------------------------------------------------------------------------
// 帧槽分配
// ---------------------------------------------------------------------------

/// 分配一个帧槽：优先空槽，其次复用**已落盘**的槽（其拷贝早已完成，覆写安全）。
///
/// 关键纪律：**绝不覆写还在等宿主写入的槽**——否则上一帧的拷贝会晚一步落进
/// 新帧的像素区，把「未落盘」误判成「已落盘」（这正是本模块唯一要防的假绿）。
fn alloc_frame_slot(
    owner: Owner,
    width: u32,
    height: u32,
    read_len: u32,
) -> Result<usize, SceneDispatchError> {
    ARENA.with(|a| {
        let mut a = a.borrow_mut();
        let pick = (0..FRAME_SLOTS).find(|&i| !a.meta[i].used).or_else(|| {
            (0..FRAME_SLOTS).find(|&i| {
                let n = a.meta[i].read_len as usize;
                !is_canary(&a.pixels[i][..n])
            })
        });
        let Some(i) = pick else {
            return Err(SceneDispatchError::NoFreeFrameSlot);
        };
        a.pixels[i] = [CANARY; MAX_FRAME_PIXELS];
        a.meta[i] = FrameMeta { used: true, owner, width, height, read_len };
        Ok(i)
    })
}

/// 归还一个**提交失败**的帧槽（该槽的读回从未登记，永远不会落盘）。
///
/// 只清 `used`，像素区不必重置——下次 `alloc_frame_slot` 会把整段重填哨兵。
fn release_frame_slot(slot: usize) {
    ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.meta[slot] = FrameMeta::EMPTY;
    });
}

/// 竞技场的各部分地址（一次性取出，随后释放借用再调胶水）。
#[derive(Debug, Clone, Copy)]
struct ArenaAddrs {
    params: u32,
    spheres: u32,
    lights: u32,
    tris: u32,
    nodes: u32,
    pixels: u32,
}

// ---------------------------------------------------------------------------
// 提交
// ---------------------------------------------------------------------------

/// 一帧的 5 块输入缓冲（V2 六绑定去掉 `out`）。
///
/// 只编码**一次**（`SceneAbi::encode_v2` 会做曲线三角化 + BVH 构建，代价不低），
/// 然后交给 [`emit_frame`] 建 buffer。
struct EncodedScene {
    /// 参数块（`PARAMS_WORDS_V2 × 4` 字节，binding 0）。
    params: [u8; PARAMS_BYTES],
    /// 球体缓冲（binding 1）。
    spheres: Vec<u8>,
    /// 光源缓冲（binding 2）。
    lights: Vec<u8>,
    /// 三角形缓冲（binding 3）。
    tris: Vec<u8>,
    /// 扁平 BVH 节点缓冲（binding 4）。
    nodes: Vec<u8>,
}

impl EncodedScene {
    /// 按 `scene_abi` 的单一真源编码一帧的全部输入缓冲。
    fn encode(req: &SceneRenderRequest) -> Self {
        let (params, spheres, lights, tris, nodes) = SceneAbi::encode_v2(req);
        EncodedScene { params, spheres, lights, tris, nodes }
    }
}

/// 真 WebGPU 提交一帧场景渲染（读回登记在该帧槽的像素区）。
///
/// 返回的 [`SceneFrame`] 在宿主 `settle()` 之后可经 [`with_settled_pixels`] 读取。
/// 设备未就绪 / 规模超限 / 任一步胶水失败 ⇒ `Err`，**不 panic**。
pub(crate) fn submit_scene_frame(
    device: GpuHandle,
    req: &SceneRenderRequest,
) -> Result<SceneFrame, SceneDispatchError> {
    submit_frame(device, req, Owner::Test)
}

/// [`submit_scene_frame`] 的实现体（`owner` 决定该帧归哪个消费者取用）。
fn submit_frame(
    device: GpuHandle,
    req: &SceneRenderRequest,
    owner: Owner,
) -> Result<SceneFrame, SceneDispatchError> {
    if !cfg!(target_arch = "wasm32") {
        // native：`gpu::*` 全是返回 0 的桩，如实报「本目标无 GPU 路径」。
        // 用 `cfg!` 而非 `#[cfg]` 是为了让本函数在 native 下**仍然完整类型检查**
        // （否则 wasm 侧的接线错误只能在构建 wasm 时才发现）。
        return Err(SceneDispatchError::NotWasmTarget);
    }
    // ---- 0. 规模守卫（先判再写竞技场，避免半途覆写留下不一致状态）----
    if req.width == 0 || req.height == 0 {
        return Err(SceneDispatchError::EmptyFrame);
    }
    if req.width > MAX_FRAME_W {
        return Err(SceneDispatchError::TooLarge {
            what: "width",
            need: req.width,
            cap: MAX_FRAME_W,
        });
    }
    if req.height > MAX_FRAME_H {
        return Err(SceneDispatchError::TooLarge {
            what: "height",
            need: req.height,
            cap: MAX_FRAME_H,
        });
    }
    if req.spheres.len() > MAX_SPHERES {
        return Err(SceneDispatchError::TooLarge {
            what: "spheres",
            need: req.spheres.len() as u32,
            cap: MAX_SPHERES as u32,
        });
    }
    if req.lights.len() > MAX_LIGHTS {
        return Err(SceneDispatchError::TooLarge {
            what: "lights",
            need: req.lights.len() as u32,
            cap: MAX_LIGHTS as u32,
        });
    }
    // 尺寸守卫已把 w/h 限在 32 以内，故 w*h*4 ≤ 4096，不会溢出。
    let out_bytes = req.width * req.height * 4;
    // ---- 1. 编码（单一真源：`SceneAbi::encode_v2`，本模块不写参数布局）----
    let enc = EncodedScene::encode(req);
    if enc.tris.len() > TRI_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "triangles",
            need: enc.tris.len() as u32,
            cap: TRI_BYTES as u32,
        });
    }
    if enc.nodes.len() > NODE_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "nodes",
            need: enc.nodes.len() as u32,
            cap: NODE_BYTES as u32,
        });
    }
    // dispatch 规划（入口 / 工作组 / 工作组数 / 绑定数）——单一真源派生。
    let plan = scene_dispatch_plan(req);
    debug_assert_eq!(plan.entry, SceneAbi::ENTRY);
    debug_assert_eq!(plan.bindings, SceneAbi::bindings().len());
    debug_assert!(plan.wgsl_present, "scene_frame.wgsl 未注册进 SHADERS 表");
    // ---- 2. 占槽（像素区填哨兵）----
    let slot = alloc_frame_slot(owner, req.width, req.height, out_bytes)?;
    // ---- 3~8：建 buffer / bind group / dispatch / 登记读回 ----
    match emit_frame(device, slot, out_bytes, plan.groups, &enc) {
        Ok(()) => Ok(SceneFrame {
            slot,
            width: req.width,
            height: req.height,
            read_len: out_bytes,
        }),
        Err(e) => {
            // 任一步失败 ⇒ 该槽永不落盘，必须立刻归还，否则竞技场被慢性耗光
            // （每失败一次永久占 1 格，10 次失败后所有提交都 `NoFreeFrameSlot`）。
            release_frame_slot(slot);
            Err(e)
        }
    }
}

/// 第 3~8 步：填竞技场 → 建 6 个 buffer → bind group → dispatch → 登记读回。
///
/// 与 [`submit_frame`] 拆开的原因：失败时由调用方**归还帧槽**
/// （见 [`release_frame_slot`]），这样「提交失败」不会把定长竞技场慢性耗光。
fn emit_frame(
    device: GpuHandle,
    slot: usize,
    out_bytes: u32,
    groups: [u32; 3],
    enc: &EncodedScene,
) -> Result<(), SceneDispatchError> {
    // ---- 3. 填竞技场并一次取齐地址 ----
    let addrs = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.params.copy_from_slice(&enc.params);
        a.spheres[..enc.spheres.len()].copy_from_slice(&enc.spheres);
        a.lights[..enc.lights.len()].copy_from_slice(&enc.lights);
        a.tris[..enc.tris.len()].copy_from_slice(&enc.tris);
        a.nodes[..enc.nodes.len()].copy_from_slice(&enc.nodes);
        ArenaAddrs {
            params: addr_of(&a.params),
            spheres: addr_of(&a.spheres),
            lights: addr_of(&a.lights),
            tris: addr_of(&a.tris),
            nodes: addr_of(&a.nodes),
            pixels: addr_of(&a.pixels[slot]),
        }
    });
    // ---- 4. pipeline（缓存）+ 5 个输入 buffer + 输出 buffer ----
    let pipeline = ensure_pipeline(device)?;
    // 顺序**严格**等于 `SceneAbi::bindings()`：0 params / 1 spheres / 2 lights /
    // 3 triangles / 4 nodes / 5 out。前 5 个的源数据与长度逐项对应。
    let inputs: [(u32, usize); 5] = [
        (addrs.params, enc.params.len()),
        (addrs.spheres, enc.spheres.len()),
        (addrs.lights, enc.lights.len()),
        (addrs.tris, enc.tris.len()),
        (addrs.nodes, enc.nodes.len()),
    ];
    let mut bufs = [0u32; HANDLE_CAP];
    for (i, &(ptr, len)) in inputs.iter().enumerate() {
        let size = buf_size(len);
        let h = gpu::create_buffer(device, size, usage::STORAGE);
        if h == 0 {
            return Err(SceneDispatchError::step("create_buffer", i as u32));
        }
        // 空输入不写（WebGPU 的 buffer 创建即零初始化，且 shader 只在计数 > 0 时读取）。
        if len > 0 && gpu::write_buffer(device, h, ptr, len as u32) == 0 {
            return Err(SceneDispatchError::step("write_buffer", i as u32));
        }
        bufs[i] = h;
    }
    let out_idx = inputs.len() as u32; // = 5，= `SceneAbiProfile::V2::out_binding()`
    debug_assert_eq!(out_idx, SceneAbi::PROFILE.out_binding());
    let out = gpu::create_buffer(device, out_bytes, usage::STORAGE);
    if out == 0 {
        return Err(SceneDispatchError::step("create_buffer", out_idx));
    }
    bufs[out_idx as usize] = out;
    // ---- 6. bind group（6 entries，第 i 个句柄 → binding: i）----
    let n_bound = SceneAbi::bindings().len();
    let bufs_ptr = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.handles[..n_bound].copy_from_slice(&bufs[..n_bound]);
        // 句柄数组的线性内存地址：胶水按 `n × 4` 字节、每 4 字节一个 u32 读取。
        a.handles.as_ptr() as usize as u32
    });
    // 全部 6 个绑定都在 `@group(0)`（见 scene_frame.wgsl 的 `@group(0) @binding(k)`）。
    let bind_group = gpu::create_bind_group(device, pipeline, 0, bufs_ptr, n_bound as u32);
    if bind_group == 0 {
        return Err(SceneDispatchError::step("create_bind_group", NO_BINDING));
    }
    // ---- 7. dispatch（工作组数取 scene_dispatch_plan 的 groups）----
    let bgs_ptr = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.bgs[0] = bind_group;
        a.bgs.as_ptr() as usize as u32
    });
    if gpu::dispatch(device, pipeline, bgs_ptr, 1, groups[0], groups[1], groups[2]) == 0 {
        return Err(SceneDispatchError::step("dispatch", NO_BINDING));
    }
    note_err_scope(); // 宿主每次 dispatch 前 push 一个 validation error scope
    if gpu::wait_idle(device) == 0 {
        return Err(SceneDispatchError::step("wait_idle", NO_BINDING));
    }
    // ---- 8. 读回：out → 暂存 → 竞技场像素区（真正落盘发生在宿主 settle()）----
    let staging = gpu::read_buffer(device, out, out_bytes);
    if staging == 0 {
        return Err(SceneDispatchError::step("read_buffer", out_idx));
    }
    if gpu::copy_out(staging, addrs.pixels, out_bytes) == 0 {
        return Err(SceneDispatchError::step("copy_out", out_idx));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 读回访问
// ---------------------------------------------------------------------------

/// 在**已落盘**时把像素切片交给 `f`；未落盘 / 已被消费 ⇒ `None`。
///
/// 只取不可变借用，故可安全嵌套（比较两帧时用得上）。
pub(crate) fn with_settled_pixels<R>(frame: SceneFrame, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
    ARENA.with(|a| {
        let a = a.borrow();
        let meta = a.meta[frame.slot];
        let n = frame.read_len as usize;
        if !meta.used || n == 0 || n > MAX_FRAME_PIXELS {
            return None;
        }
        let px = &a.pixels[frame.slot][..n];
        if is_canary(px) {
            return None;
        }
        Some(f(px))
    })
}

/// 帧的像素是否已落盘（宿主 `settle()` 之后为 `true`）。
///
/// 目前只被单元测试用来钉住「未提交的槽不得交出像素」这条语义；保留独立函数
/// 而不是让测试直接调 [`with_settled_pixels`]，是为了让意图可读。
#[cfg(test)]
pub(crate) fn is_settled(frame: SceneFrame) -> bool {
    with_settled_pixels(frame, |_| ()).is_some()
}

/// 消费式取用**上一帧已落盘**的同尺寸像素（`render_scene` 专用）。
///
/// 只取 `Owner::Render` 的帧，且取走后标记为已消费 ⇒ 同一帧不会被返回两次。
fn take_settled_render_frame(width: u32, height: u32) -> Option<Vec<u8>> {
    ARENA.with(|a| {
        let mut a = a.borrow_mut();
        let idx = (0..FRAME_SLOTS).find(|&i| {
            let m = a.meta[i];
            if !m.used || m.owner != Owner::Render || m.width != width || m.height != height {
                return false;
            }
            let n = m.read_len as usize;
            n != 0 && n <= MAX_FRAME_PIXELS && !is_canary(&a.pixels[i][..n])
        })?;
        let n = a.meta[idx].read_len as usize;
        let px = a.pixels[idx][..n].to_vec();
        a.meta[idx].used = false;
        Some(px)
    })
}

/// 真 WebGPU 执行一帧场景渲染：**提交本帧**，并返回**上一帧已落盘**的 RGBA8 像素
/// （`w*h*4` 字节）。
///
/// # 为什么返回的是「上一帧」
///
/// WebGPU 无阻塞读回（见模块文档）。同步签名 `render_scene` 拿不到刚提交那一帧的
/// 像素，因此本函数：① 提交本帧（真 GPU 路径必须先真的被执行）；② 交出上一帧
/// 已落盘、且**尺寸相同**的像素。首次调用必然 [`SceneDispatchError::NotSettledYet`]
/// ⇒ 调用方按 CPU 参考兜底，帧率与输出长度都不受影响。
///
/// 设备未就绪 / 规模超限 / 任一步胶水失败 ⇒ `Err`（**不 panic**，调用方决定降级）。
pub(crate) fn dispatch_scene_frame(
    device: GpuHandle,
    req: &SceneRenderRequest,
) -> Result<Vec<u8>, SceneDispatchError> {
    // 先取旧帧（消费），再提交本帧：顺序不能反——否则本次提交可能复用到该槽，
    // 把「即将返回的旧帧」当场覆写成哨兵。
    let previous = take_settled_render_frame(req.width, req.height);
    submit_frame(device, req, Owner::Render)?;
    previous.ok_or(SceneDispatchError::NotSettledYet)
}

// ---------------------------------------------------------------------------
// V3 剖面真 WebGPU dispatch（W1-C 九绑定 → W3 十一绑定：再 + instances/instance_nodes）
// ---------------------------------------------------------------------------

/// V3 一帧的 10 块输入缓冲（11 绑定去掉 `out`）。
///
/// 只编码**一次**（`SceneAbi::encode_v3*` 会做曲线三角化 + BVH 构建 + 纹理展平，
/// 代价不低），然后交给 [`emit_frame_v3`] 建 buffer。
struct EncodedSceneV3 {
    /// 参数块（`PARAMS_WORDS_V3 × 4` 字节，binding 0）。
    params: [u8; V3_PARAMS_BYTES],
    /// 球体缓冲（binding 1，每球 48 B）。
    spheres: Vec<u8>,
    /// 光源缓冲（binding 2）。
    lights: Vec<u8>,
    /// 三角形缓冲（binding 3）。
    tris: Vec<u8>,
    /// 扁平 BVH 节点缓冲（binding 4）。
    nodes: Vec<u8>,
    /// 纹素流（`array<vec4<f32>>`，binding 5）。
    textures: Vec<u8>,
    /// 纹理元数据（`array<vec4<u32>>`，32 B/条，binding 6）。
    texmeta: Vec<u8>,
    /// 环境 CDF（`array<u32>`，binding 7）。
    cdf: Vec<u8>,
    /// 实例记录（`array<u32>`，28 槽 = 112 B/条，binding 8）。
    instances: Vec<u8>,
    /// 实例 TLAS 节点（`array<u32>`，12 槽 = 48 B/条，binding 9）。
    instance_nodes: Vec<u8>,
}

impl EncodedSceneV3 {
    /// 按 `scene_abi` 的单一真源编码一帧的全部输入缓冲。
    ///
    /// `instances` 为空 ⇒ 与 W2 收口时的 9 绑定路径**逐字节相同**（实例两块
    /// 缓冲为空、两计数为 0，WGSL 的实例遍历段整段跳过）。
    ///
    /// 实例缓冲的字节布局由 [`crate::instance`] 独家决定（112 B/条，逆矩阵在
    /// CPU 侧算一次），本模块**不重算、不重排**，只做搬运——与 Vulkan 宿主
    /// （`vulkan/v3_frame.rs::dispatch_v3_instanced`）读的是同一份编码。
    ///
    /// `encode_v3_instanced` 失败（尺寸非法 / 级数越界 / 纹素条数不符 / CDF
    /// 字节数不符 / 纹理太多 / 实例几何非共享）⇒ 转为
    /// [`SceneDispatchError::Step`]，不 panic。
    fn encode_instanced(
        req: &SceneRenderRequest,
        textures: &[TextureUpload<'_>],
        env: Option<EnvCdf<'_>>,
        instances: &[InstanceReq],
    ) -> Result<Self, SceneDispatchError> {
        let b = match SceneAbi::encode_v3_instanced(req, textures, env, instances) {
            Ok(b) => b,
            // 编码侧错误（尺寸非法 / 级数越界 / 纹素条数不符 / CDF 字节数不符 /
            // 纹理太多 / 实例几何非共享）在真机上表现为"驱动读到垃圾"，故 CPU 侧
            // 拦死；转成 [`SceneDispatchError::Step`] 让调用方走统一失败路径。
            Err(_) => {
                return Err(SceneDispatchError::Step {
                    step: "encode_v3",
                    binding: NO_BINDING,
                    code: 0,
                })
            }
        };
        Ok(EncodedSceneV3 {
            params: b.params,
            spheres: b.spheres,
            lights: b.lights,
            tris: b.triangles,
            nodes: b.nodes,
            textures: b.textures,
            texmeta: b.texmeta,
            cdf: b.cdf,
            instances: b.instances,
            instance_nodes: b.instance_nodes,
        })
    }
}

/// 竞技场的各部分地址（V3：10 输入缓冲 + 像素区，一次性取出，随后释放借用再调胶水）。
#[derive(Debug, Clone, Copy)]
struct ArenaAddrsV3 {
    params_v3: u32,
    spheres_v3: u32,
    lights: u32,
    tris: u32,
    nodes: u32,
    texels: u32,
    texmeta: u32,
    cdf: u32,
    instances: u32,
    instance_nodes: u32,
    pixels: u32,
}

/// 真 WebGPU 提交一帧 V3 场景渲染（11 绑定，无实例），读回登记在该帧槽的像素区。
///
/// 返回的 [`SceneFrame`] 在宿主 `settle()` 之后可经 [`with_settled_pixels`] 读取。
/// 设备未就绪 / 规模超限 / 任一步胶水失败 ⇒ `Err`，**不 panic**。
pub(crate) fn submit_scene_frame_v3(
    device: GpuHandle,
    req: &SceneRenderRequest,
    textures: &[TextureUpload<'_>],
    env: Option<EnvCdf<'_>>,
) -> Result<SceneFrame, SceneDispatchError> {
    submit_scene_frame_v3_instanced(device, req, textures, env, &[])
}

/// 同 [`submit_scene_frame_v3`]，但带 **W3 实例表**（`instances` 为空 ⇒ 与旧路径
/// 逐字节相同）。
///
/// 这是 Part B 在 WebGPU 侧的落点：实例缓冲按 [`crate::scene_abi::V3_BINDINGS`]
/// 的 binding 8/9 建 buffer 并派发，与 Vulkan 宿主走同一份 `SceneAbi` 编码。
pub(crate) fn submit_scene_frame_v3_instanced(
    device: GpuHandle,
    req: &SceneRenderRequest,
    textures: &[TextureUpload<'_>],
    env: Option<EnvCdf<'_>>,
    instances: &[InstanceReq],
) -> Result<SceneFrame, SceneDispatchError> {
    submit_frame_v3(device, req, textures, env, instances, Owner::Test)
}

/// [`submit_scene_frame_v3`] 的实现体（`owner` 决定该帧归哪个消费者取用）。
fn submit_frame_v3(
    device: GpuHandle,
    req: &SceneRenderRequest,
    textures: &[TextureUpload<'_>],
    env: Option<EnvCdf<'_>>,
    instances: &[InstanceReq],
    owner: Owner,
) -> Result<SceneFrame, SceneDispatchError> {
    if !cfg!(target_arch = "wasm32") {
        // native：`gpu::*` 全是返回 0 的桩，如实报「本目标无 GPU 路径」。
        return Err(SceneDispatchError::NotWasmTarget);
    }
    // ---- 0. 规模守卫（先判再写竞技场，避免半途覆写留下不一致状态）----
    if req.width == 0 || req.height == 0 {
        return Err(SceneDispatchError::EmptyFrame);
    }
    if req.width > MAX_FRAME_W {
        return Err(SceneDispatchError::TooLarge {
            what: "width",
            need: req.width,
            cap: MAX_FRAME_W,
        });
    }
    if req.height > MAX_FRAME_H {
        return Err(SceneDispatchError::TooLarge {
            what: "height",
            need: req.height,
            cap: MAX_FRAME_H,
        });
    }
    if req.spheres.len() > MAX_SPHERES {
        return Err(SceneDispatchError::TooLarge {
            what: "spheres",
            need: req.spheres.len() as u32,
            cap: MAX_SPHERES as u32,
        });
    }
    if req.lights.len() > MAX_LIGHTS {
        return Err(SceneDispatchError::TooLarge {
            what: "lights",
            need: req.lights.len() as u32,
            cap: MAX_LIGHTS as u32,
        });
    }
    if textures.len() > MAX_V3_TEXTURES {
        return Err(SceneDispatchError::TooLarge {
            what: "textures",
            need: textures.len() as u32,
            cap: MAX_V3_TEXTURES as u32,
        });
    }
    if instances.len() > MAX_INSTANCES {
        return Err(SceneDispatchError::TooLarge {
            what: "instances",
            need: instances.len() as u32,
            cap: MAX_INSTANCES as u32,
        });
    }
    // 尺寸守卫已把 w/h 限在 32 以内，故 w*h*4 ≤ 4096，不会溢出。
    let out_bytes = req.width * req.height * 4;
    // ---- 1. 编码（单一真源：SceneAbi::encode_v3_with_instances）----
    let enc = EncodedSceneV3::encode_instanced(req, textures, env, instances)?;
    if enc.tris.len() > TRI_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "triangles",
            need: enc.tris.len() as u32,
            cap: TRI_BYTES as u32,
        });
    }
    if enc.nodes.len() > NODE_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "nodes",
            need: enc.nodes.len() as u32,
            cap: NODE_BYTES as u32,
        });
    }
    if enc.textures.len() > TEXEL_ARENA_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "textures",
            need: enc.textures.len() as u32,
            cap: TEXEL_ARENA_BYTES as u32,
        });
    }
    if enc.texmeta.len() > TEXMETA_ARENA_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "texmeta",
            need: enc.texmeta.len() as u32,
            cap: TEXMETA_ARENA_BYTES as u32,
        });
    }
    if enc.cdf.len() > CDF_ARENA_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "cdf",
            need: enc.cdf.len() as u32,
            cap: CDF_ARENA_BYTES as u32,
        });
    }
    if enc.instances.len() > INSTANCE_ARENA_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "instances",
            need: enc.instances.len() as u32,
            cap: INSTANCE_ARENA_BYTES as u32,
        });
    }
    if enc.instance_nodes.len() > INSTANCE_NODE_ARENA_BYTES {
        return Err(SceneDispatchError::TooLarge {
            what: "instance_nodes",
            need: enc.instance_nodes.len() as u32,
            cap: INSTANCE_NODE_ARENA_BYTES as u32,
        });
    }
    // dispatch 规划（入口 / 工作组 / 工作组数 / 绑定数）——单一真源派生。
    let plan = scene_dispatch_plan_v3(req);
    debug_assert_eq!(plan.entry, SceneAbi::ENTRY);
    debug_assert_eq!(plan.bindings, SceneAbi::bindings_for(SceneAbiProfile::V3).len());
    debug_assert!(plan.wgsl_present, "scene_frame_v3.wgsl 未注册进 SHADERS 表");
    // ---- 2. 占槽（像素区填哨兵）----
    let slot = alloc_frame_slot(owner, req.width, req.height, out_bytes)?;
    // ---- 3~8：建 buffer / bind group / dispatch / 登记读回 ----
    match emit_frame_v3(device, slot, out_bytes, plan.groups, &enc) {
        Ok(()) => Ok(SceneFrame {
            slot,
            width: req.width,
            height: req.height,
            read_len: out_bytes,
        }),
        Err(e) => {
            // 任一步失败 ⇒ 该槽永不落盘，必须立刻归还，否则竞技场被慢性耗光。
            release_frame_slot(slot);
            Err(e)
        }
    }
}

/// 第 3~8 步（V3）：填竞技场 → 建 11 个 buffer → bind group → dispatch → 登记读回。
///
/// 与 [`emit_frame`] 拆开的原因相同：失败时由调用方**归还帧槽**。
fn emit_frame_v3(
    device: GpuHandle,
    slot: usize,
    out_bytes: u32,
    groups: [u32; 3],
    enc: &EncodedSceneV3,
) -> Result<(), SceneDispatchError> {
    // ---- 3. 填竞技场并一次取齐地址 ----
    let addrs = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.params_v3.copy_from_slice(&enc.params);
        a.spheres_v3[..enc.spheres.len()].copy_from_slice(&enc.spheres);
        a.lights[..enc.lights.len()].copy_from_slice(&enc.lights);
        a.tris[..enc.tris.len()].copy_from_slice(&enc.tris);
        a.nodes[..enc.nodes.len()].copy_from_slice(&enc.nodes);
        a.texels[..enc.textures.len()].copy_from_slice(&enc.textures);
        a.texmeta[..enc.texmeta.len()].copy_from_slice(&enc.texmeta);
        a.cdf[..enc.cdf.len()].copy_from_slice(&enc.cdf);
        a.instances[..enc.instances.len()].copy_from_slice(&enc.instances);
        a.instance_nodes[..enc.instance_nodes.len()].copy_from_slice(&enc.instance_nodes);
        ArenaAddrsV3 {
            params_v3: addr_of(&a.params_v3),
            spheres_v3: addr_of(&a.spheres_v3),
            lights: addr_of(&a.lights),
            tris: addr_of(&a.tris),
            nodes: addr_of(&a.nodes),
            texels: addr_of(&a.texels),
            texmeta: addr_of(&a.texmeta),
            cdf: addr_of(&a.cdf),
            instances: addr_of(&a.instances),
            instance_nodes: addr_of(&a.instance_nodes),
            pixels: addr_of(&a.pixels[slot]),
        }
    });
    // ---- 4. pipeline（独立缓存）+ 10 个输入 buffer + 输出 buffer ----
    let pipeline = ensure_pipeline_v3(device)?;
    // 顺序**严格**等于 `V3_BINDINGS`：0 params_v3 / 1 spheres_v3 / 2 lights /
    // 3 triangles / 4 nodes / 5 textures / 6 texmeta / 7 cdf /
    // 8 instances / 9 instance_nodes / 10 out。
    // 前 10 个的源数据与长度逐项对应。
    let inputs: [(u32, usize); 10] = [
        (addrs.params_v3, enc.params.len()),
        (addrs.spheres_v3, enc.spheres.len()),
        (addrs.lights, enc.lights.len()),
        (addrs.tris, enc.tris.len()),
        (addrs.nodes, enc.nodes.len()),
        (addrs.texels, enc.textures.len()),
        (addrs.texmeta, enc.texmeta.len()),
        (addrs.cdf, enc.cdf.len()),
        (addrs.instances, enc.instances.len()),
        (addrs.instance_nodes, enc.instance_nodes.len()),
    ];
    let mut bufs = [0u32; HANDLE_CAP];
    for (i, &(ptr, len)) in inputs.iter().enumerate() {
        let size = buf_size(len);
        let h = gpu::create_buffer(device, size, usage::STORAGE);
        if h == 0 {
            return Err(SceneDispatchError::step("create_buffer", i as u32));
        }
        // 空输入不写（WebGPU 的 buffer 创建即零初始化，且 shader 只在计数 > 0 时读取）。
        if len > 0 && gpu::write_buffer(device, h, ptr, len as u32) == 0 {
            return Err(SceneDispatchError::step("write_buffer", i as u32));
        }
        bufs[i] = h;
    }
    let out_idx = inputs.len() as u32; // = 10，= `SceneAbiProfile::V3::out_binding()`
    debug_assert_eq!(out_idx, SceneAbiProfile::V3.out_binding());
    let out = gpu::create_buffer(device, out_bytes, usage::STORAGE);
    if out == 0 {
        return Err(SceneDispatchError::step("create_buffer", out_idx));
    }
    bufs[out_idx as usize] = out;
    // ---- 6. bind group（11 entries，第 i 个句柄 → binding: i）----
    let n_bound = SceneAbi::bindings_for(SceneAbiProfile::V3).len();
    let bufs_ptr = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.handles[..n_bound].copy_from_slice(&bufs[..n_bound]);
        // 句柄数组的线性内存地址：胶水按 `n × 4` 字节、每 4 字节一个 u32 读取。
        a.handles.as_ptr() as usize as u32
    });
    // 全部 11 个绑定都在 `@group(0)`（见 scene_frame_v3.wgsl 的 `@group(0) @binding(k)`）。
    let bind_group = gpu::create_bind_group(device, pipeline, 0, bufs_ptr, n_bound as u32);
    if bind_group == 0 {
        return Err(SceneDispatchError::step("create_bind_group", NO_BINDING));
    }
    // ---- 7. dispatch（工作组数取 scene_dispatch_plan_v3 的 groups）----
    let bgs_ptr = ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.bgs[0] = bind_group;
        a.bgs.as_ptr() as usize as u32
    });
    if gpu::dispatch(device, pipeline, bgs_ptr, 1, groups[0], groups[1], groups[2]) == 0 {
        return Err(SceneDispatchError::step("dispatch", NO_BINDING));
    }
    note_err_scope(); // 宿主每次 dispatch 前 push 一个 validation error scope
    if gpu::wait_idle(device) == 0 {
        return Err(SceneDispatchError::step("wait_idle", NO_BINDING));
    }
    // ---- 8. 读回：out → 暂存 → 竞技场像素区（真正落盘发生在宿主 settle()）----
    let staging = gpu::read_buffer(device, out, out_bytes);
    if staging == 0 {
        return Err(SceneDispatchError::step("read_buffer", out_idx));
    }
    if gpu::copy_out(staging, addrs.pixels, out_bytes) == 0 {
        return Err(SceneDispatchError::step("copy_out", out_idx));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 单元测试（native：不碰 GPU，只验竞技场状态机与尺寸派生）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuRefBackend;
    use crate::GpuDevice;
    use prism_math::Vec3;

    /// 复位竞技场与 pipeline 缓存（测试间隔离；`thread_local` 在同测试线程内共享）。
    fn reset() {
        ARENA.with(|a| *a.borrow_mut() = SceneArena::new());
        PIPE.with(|p| *p.borrow_mut() = PipelineCache::EMPTY);
        let _ = take_err_scopes();
    }

    fn small_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 8,
            height: 8,
            camera: crate::CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 1.0,
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
                maps: crate::NO_MAPS,
            }],
            aesthetic: crate::AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
            curves: Vec::new(),
        }
    }

    /// 缓冲区尺寸派生：空输入给最小合法长度（WebGPU 拒绝 size == 0）。
    #[test]
    fn buf_size_never_zero() {
        assert_eq!(buf_size(0), MIN_BUF_BYTES);
        assert_eq!(buf_size(4), 4);
        assert_eq!(buf_size(96), 96);
        assert!(MIN_BUF_BYTES >= 4, "array<u32> 的最小可绑定尺寸是 4 字节");
    }

    /// 「球 + 圆柱」场景（与 `abi::gpu.scene_frame` 的场景 B 同几何）：
    /// 球半径 0.35 置于 (-0.3,-0.1,0)，圆柱（半径 0.18）置于 x=0.32 ——
    /// 二者在 x 上相距 0.62 > 0.35+0.18，互不遮挡，曲线在画面上独立可见。
    fn curve_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 16,
            height: 16,
            spheres: vec![crate::SphereReq {
                center: Vec3::new(-0.3, -0.1, 0.0),
                radius: 0.35,
                albedo: Vec3::new(0.25, 0.55, 0.85),
                roughness: 0.3,
                metalness: 0.2,
                maps: crate::NO_MAPS,
            }],
            curves: vec![crate::CurveReq::Cylinder {
                center: Vec3::new(0.32, 0.0, 0.0),
                radius: 0.18,
                height: 1.4,
                axis: Vec3::new(0.0, 1.0, 0.0),
            }],
            ..small_request()
        }
    }

    /// 竞技场容量必须装得下「1 球 + 1 灯 + 1 个圆柱」的测试场景。
    #[test]
    fn arena_holds_curve_scene_buffers() {
        let req = curve_request();
        let (_p, spheres, lights, tris, nodes) = SceneAbi::encode_v2(&req);
        assert!(spheres.len() <= SPHERE_BYTES);
        assert!(lights.len() <= LIGHT_BYTES);
        assert!(!tris.is_empty(), "圆柱必须三角化出三角形");
        assert!(!nodes.is_empty(), "非空三角形集必须建出 BVH 节点");
        assert!(
            tris.len() <= TRI_BYTES,
            "三角形缓冲 {} 字节超过竞技场 {TRI_BYTES}（调 MAX_TRIS 或换更小的曲线）",
            tris.len()
        );
        assert!(
            nodes.len() <= NODE_BYTES,
            "节点缓冲 {} 字节超过竞技场 {NODE_BYTES}（调 MAX_NODES）",
            nodes.len()
        );
    }

    /// native 下提交必然 `NotWasmTarget`（不 panic、不碰胶水）。
    #[test]
    fn submit_on_native_reports_not_wasm_target() {
        reset();
        let err = submit_scene_frame(0, &small_request()).unwrap_err();
        assert_eq!(err, SceneDispatchError::NotWasmTarget);
        // 错误文案必须带可操作信息（真机错误面板直接展示）。
        assert!(format!("{err}").contains("wasm"));
    }

    /// native 下门面也必须如实失败（`render_scene` 据此回退 CPU）。
    #[test]
    fn dispatch_scene_frame_on_native_never_panics() {
        reset();
        assert!(dispatch_scene_frame(0, &small_request()).is_err());
    }

    /// V3 十一绑定 dispatch 在 native 下同样如实报 `NotWasmTarget`（不 panic、不伪绿），
    /// 且无纹理集时也必须走同一分支。
    #[test]
    fn submit_scene_frame_v3_on_native_reports_not_wasm_target() {
        reset();
        let err = submit_scene_frame_v3(0, &small_request(), &[], None).unwrap_err();
        assert_eq!(err, SceneDispatchError::NotWasmTarget);
        assert!(format!("{err}").contains("wasm"));
        // 带一张纹理也必须同样如实失败（纹理路径不改变「native 无 GPU」的判定）。
        let tex = TextureUpload {
            w: 2,
            h: 2,
            levels: 1,
            fmt: 1,
            srgb: 0,
            texels: &[[0.5f32; 4]; 4],
        };
        let err2 =
            submit_scene_frame_v3(0, &small_request(), &[tex], None).unwrap_err();
        assert_eq!(err2, SceneDispatchError::NotWasmTarget);
    }

    /// 哨兵判定：只有「整段都是哨兵」才算未落盘，且空切片不算。
    #[test]
    fn canary_detection_is_conservative() {
        assert!(!is_canary(&[]), "空切片不得被判为未落盘");
        let mut px = [CANARY; 8];
        assert!(is_canary(&px));
        px[7] = 0;
        assert!(!is_canary(&px), "有任一字节被覆盖即视为已落盘");
    }

    /// 未提交的槽不得交出像素（防「读到 bss 零初始化区」的假绿）。
    #[test]
    fn untouched_slots_yield_no_pixels() {
        reset();
        let frame = SceneFrame { slot: 0, width: 8, height: 8, read_len: 8 * 8 * 4 };
        assert!(with_settled_pixels(frame, |_| ()).is_none());
        assert!(!is_settled(frame));
    }

    /// 落盘后能读到像素，且 `take_settled_render_frame` 只给一次（消费语义）。
    #[test]
    fn settled_render_frame_is_consumed_once() {
        reset();
        let slot = alloc_frame_slot(Owner::Render, 4, 4, 4 * 4 * 4).expect("有空槽");
        // 模拟宿主 settle() 的整段覆盖：写入非哨兵像素。
        ARENA.with(|a| {
            let mut a = a.borrow_mut();
            for (i, b) in a.pixels[slot][..4 * 4 * 4].iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        });
        let frame = SceneFrame { slot, width: 4, height: 4, read_len: 4 * 4 * 4 };
        assert!(is_settled(frame));
        let got = with_settled_pixels(frame, |px| px.to_vec()).expect("已落盘");
        assert_eq!(got.len(), 4 * 4 * 4, "读回字节数必须恰为 w*h*4");
        let first = take_settled_render_frame(4, 4).expect("应取到一帧");
        assert_eq!(first, got);
        assert!(take_settled_render_frame(4, 4).is_none(), "同一帧只能被消费一次");
    }

    /// 尺寸不匹配的旧帧不得被门面返回（否则会把 A 尺寸的像素当成 B 尺寸的帧）。
    #[test]
    fn render_frame_requires_matching_dimensions() {
        reset();
        let slot = alloc_frame_slot(Owner::Render, 4, 4, 4 * 4 * 4).expect("有空槽");
        ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.pixels[slot][..4 * 4 * 4].fill(1);
        });
        assert!(take_settled_render_frame(8, 8).is_none(), "尺寸不匹配必须拒绝");
        assert!(take_settled_render_frame(4, 4).is_some(), "同尺寸才可取用");
    }

    /// 全部槽都在等落盘时，分配必须失败而不是覆写（唯一的假绿防线）。
    #[test]
    fn allocation_refuses_to_overwrite_pending_slots() {
        reset();
        for _ in 0..FRAME_SLOTS {
            alloc_frame_slot(Owner::Test, 4, 4, 4 * 4 * 4).expect("先占满");
        }
        let err = alloc_frame_slot(Owner::Test, 4, 4, 4 * 4 * 4).unwrap_err();
        assert_eq!(err, SceneDispatchError::NoFreeFrameSlot);
        // 一旦某槽落盘（可回收），分配又能成功。
        ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.pixels[0][..16].fill(0);
        });
        assert!(alloc_frame_slot(Owner::Test, 4, 4, 4 * 4 * 4).is_ok());
    }

    /// 错误 scope 计数是「登记-取走」成对的（collect 阶段据此 pop 同样多条）。
    #[test]
    fn err_scopes_are_taken_atomically() {
        reset();
        note_err_scope();
        note_err_scope();
        assert_eq!(take_err_scopes(), 2);
        assert_eq!(take_err_scopes(), 0, "取走后必须归零");
    }

    /// 场景 A/B 的 CPU 参考可跑、非全零、且含曲线者与不含者确实不同。
    ///
    /// 这条在 native 上就把「曲线路径必须影响像素」的可判性钉死：真机上 GPU 的
    /// 同一对照（GPU 有曲线 vs GPU 无曲线）才有意义。
    #[test]
    fn cpu_reference_scenes_are_nonempty_and_curve_changes_pixels() {
        let mut without_curve = curve_request();
        without_curve.curves.clear();
        let b = CpuRefBackend.render_scene(&curve_request());
        let bc = CpuRefBackend.render_scene(&without_curve);
        let a = CpuRefBackend.render_scene(&small_request());
        assert_eq!(a.pixels.len(), 8 * 8 * 4);
        assert_eq!(b.pixels.len(), 16 * 16 * 4);
        assert!(a.pixels.iter().any(|&x| x != 0), "CPU 参考帧不得全零");
        assert_ne!(
            b.pixels, bc.pixels,
            "含曲线场景的 CPU 参考必须与「去掉曲线」不同（否则判据 5 恒真）"
        );
        let diff = b.pixels.chunks_exact(4).zip(bc.pixels.chunks_exact(4)).filter(|(x, y)| x != y).count();
        assert!(diff >= 2, "曲线至少要影响 2 个像素（实测 {diff}），否则判据 5 太脆弱");
    }
}
