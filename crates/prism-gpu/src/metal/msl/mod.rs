//! MSL「双份源」组织（后续 定案：Metal 侧手写，非发射器）。
//!
//! # 为什么 Metal 侧手写、Vulkan 侧走发射器
//!
//! 明定：*Vulkan 侧 SPIR-V 由自研发射器覆盖本引擎 WGSL 子集；
//! Metal 侧 MSL 手写双份源*。取舍理由：
//!
//! * SPIR-V 是**二进制 IR**，有完整公开规范，WGSL→SPIR-V 的映射可由我们自己的
//!   发射器确定化生成（也符合"确定性 md5 复现"）。
//! * MSL 是**高级 C++ 方言**，没有稳定的文本→文本规范；自动生成器会引入对
//!   Apple 编译器的隐式假设，风险高于手写。且 Metal 真机路径是**次要路径**
//!   （WebGPU 才是移动端验证主路径），手写双份源的维护成本可接受。
//!
//! # 目录与命名约定
//!
//! * 目录：本文件所在 `src/metal/msl/`，与 WGSL 源 `shaders/` **平行**。
//! * 文件名：`<wgsl 名去 .wgsl>.<entry>.msl`。
//!   * 例：WGSL `shaders/sim_cloth.wgsl` 的入口 `cloth_predict`
//!     → `src/metal/msl/sim_cloth.cloth_predict.msl`。
//! * 每个 MSL 文件**头注必须**写明：① 对应 WGSL 文件与入口；② 判对基准
//!   （对应的 CPU 参考实现路径）；③ 绑定映射表。
//!
//! # 绑定映射（WGSL → MSL）
//!
//! | WGSL | MSL |
//! |---|---|
//! | `@group(g) @binding(b) var<uniform> …` | `[[buffer(g*31+b)]]` |
//! | `@group(g) @binding(b) var<storage> …` | `[[buffer(g*31+b)]]` |
//! | `@builtin(global_invocation_id) gid` | `[[thread_position_in_grid]]` |
//! | `@workgroup_size(n)` | `[[max_total_threads_per_threadgroup(n)]]` 或 dispatch 时指定 |
//!
//! 编号 `g*31+b` 与 WebGPU 后端的 `@group` 多级绑定语义对齐（此处简化为单级
//! buffer 命名空间）；真实实装时应由后端统一计算，避免散落魔数。
//!
//! # 落地范围（D-16：不建空文件充数）
//!
//! 已落地三份**真实**样本，均为手写且各有判对基准：
//!
//! 1. [`CLOTH_PREDICT_MSL`] —— `sim_cloth.wgsl::cloth_predict`（规格示范）；
//! 2. [`PRIMARY_RAY_GEN_MSL`] —— `rt_primary_ray_gen.wgsl::primary_ray_gen`；
//! 3. [`RT_SCENE_MSL`] —— 光追场景整帧 kernel（V2.1 T-A2），
//!    **以 `include_str!` 从 `.msl` 文件编译期读入**（源文件即权威）。
//!
//! 其余 kernel 待后续按同一约定逐个手写；**不**预先创建几十个空 `.msl` 占位。

// ---------------------------------------------------------------------------
// 首样例：sim_cloth.wgsl::cloth_predict 的手写 MSL 对应
// ---------------------------------------------------------------------------

/// 首个手写 MSL 对应的 WGSL 入口名。
pub const FIRST_SAMPLE_ENTRY: &str = "cloth_predict";

/// 对应的 WGSL 源文件名（用于交叉核对，防止约定漂移）。
pub const FIRST_SAMPLE_WGSL_FILE: &str = "sim_cloth.wgsl";

/// `sim_cloth.wgsl::cloth_predict` 的手写 MSL 双份源。
///
/// 语义与 WGSL 完全一致（半隐式欧拉预测；`inv > 0` 才积分），供 Metal 真机
/// 路径上传。判对基准：`crates/prism-physics/src/cloth.rs` 的预测段。
///
/// 说明：MSL 的 `float3` 与 WGSL `vec3<f32>` 都是 16 字节对齐的向量类型；
/// `ClothState` 采用与 WGSL 相同的 AoS 布局（32 字节：pos@0 / vel@16 / inv@28），
/// 以保证跨后端逐位一致。
pub const CLOTH_PREDICT_MSL: &str = r#"// sim_cloth.wgsl::cloth_predict 的 MSL 双份源（手写）。
//
// 对应 WGSL：shaders/sim_cloth.wgsl  入口：cloth_predict
// 判对基准：crates/prism-physics/src/cloth.rs（预测段）
// 绑定映射：cparams→buffer(0)，cs→buffer(1)（即 g=0，b=0/1）。
//
// 布局对齐：ClothParams 与 WGSL 一致（rows/cols/spacing/dt/... 共 48 字节，
// vec3 gravity 后补 _pad 保证 16 字节对齐）；ClothState 为 32 字节 AoS，
// 与 WGSL ClothState 一致，从而 CPU/WebGPU/Metal 三路数值可逐位判对。

#include <metal_stdlib>
using namespace metal;

struct ClothParams {
    uint  rows;
    uint  cols;
    float spacing;
    float dt;
    float compliance_struct;
    float compliance_bend;
    float damping;
    // 对齐 WGSL：vec3<f32> gravity 占 12 字节 + 4 字节 _pad = 16 字节槽。
    float3 gravity;
    uint  pad;
};

struct ClothState {
    float3 pos;   // @0..12
    float3 vel;   // @16..28
    float  inv;   // @28..32
};

// 与 WGSL 相同的入口语义：越界直接返回；仅对非固定粒子（inv>0）积分。
kernel void cloth_predict(
    uint gid                              [[thread_position_in_grid]],
    constant ClothParams& cparams         [[buffer(0)]],
    device ClothState*    cs              [[buffer(1)]])
{
    if (gid >= cparams.rows * cparams.cols) { return; }
    ClothState s = cs[gid];
    if (s.inv > 0.0f) {
        s.vel = s.vel + cparams.gravity * cparams.dt;
        s.pos = s.pos + s.vel * cparams.dt;
        cs[gid] = s;
    }
}
"#;

// ---------------------------------------------------------------------------
// 单元测试：约定自证（纯文本，跨平台可跑）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 首样例必须真实对应一个既有的 WGSL 入口（防约定漂移）。
    ///
    /// 在 Linux 与 macOS 上均成立：该断言检查的是**源码文本约定**，与平台无关。
    #[test]
    fn first_sample_matches_wgsl_entry() {
        let wgsl = crate::shaders::SHADERS
            .iter()
            .find(|(name, _)| *name == FIRST_SAMPLE_WGSL_FILE)
            .map(|(_, src)| *src)
            .expect("sim_cloth.wgsl 必须在 shaders::SHADERS 中登记");
        assert!(
            wgsl.contains("fn cloth_predict"),
            "WGSL 侧应存在 cloth_predict 入口，否则双份源约定失配"
        );
        assert!(
            CLOTH_PREDICT_MSL.contains("kernel void cloth_predict"),
            "MSL 侧应声明同名 kernel"
        );
    }

    /// MSL 头注必须包含三项必备元信息（对应 WGSL / 判对基准 / 绑定映射）。
    #[test]
    fn msl_head_comment_declares_provenance() {
        assert!(CLOTH_PREDICT_MSL.contains("shaders/sim_cloth.wgsl"));
        assert!(CLOTH_PREDICT_MSL.contains("crates/prism-physics/src/cloth.rs"));
        assert!(CLOTH_PREDICT_MSL.contains("[[buffer(0)]]"));
        assert!(CLOTH_PREDICT_MSL.contains("[[buffer(1)]]"));
    }

    /// 语义一致性抽查：MSL 与 WGSL 一样用 `inv > 0` 门控积分。
    ///
    /// **非恒真自证**：若手写 MSL 误改成无条件积分，本断言会失败——它钉死的是
    /// 一条真实语义契约，而非永真式。
    #[test]
    fn msl_preserves_inv_gate_and_euler_step() {
        assert!(CLOTH_PREDICT_MSL.contains("s.inv > 0.0f"));
        assert!(CLOTH_PREDICT_MSL.contains("cparams.gravity * cparams.dt"));
        assert!(CLOTH_PREDICT_MSL.contains("s.vel * cparams.dt"));
    }
}

// ---------------------------------------------------------------------------
// compute-first 管线的第二个手写 MSL 双份源（主射线发射）
// ---------------------------------------------------------------------------

/// `rt_primary_ray_gen.wgsl::primary_ray_gen` 的手写 MSL 双份源。
///
/// 语义与 WGSL / CPU 参考逐条对应（像素中心 + 半像素、NDC 推导、归一化），
/// 供 Metal 真机路径上传；与 WGSL、SPIR-V 的绑定号/工作组尺寸由
/// `crate::shaders` 的三目标宿主表交叉锚定。
pub const PRIMARY_RAY_GEN_MSL: &str = r#"// rt_primary_ray_gen.wgsl::primary_ray_gen 的 MSL 双份源（手写）。
//
// 对应 WGSL：shaders/rt_primary_ray_gen.wgsl  入口：primary_ray_gen
// 判对基准：crates/prism-gpu/src/compute/mod.rs::primary_ray_gen_cpu
// 绑定映射：cparams→buffer(0)，rays_out→buffer(1)（g=0，b=0/1 的简化命名空间）
// 布局：CameraParams 为 4 个 float4 槽 + 4 个标量槽（与 WGSL b0/b1 同序），
//       RayOut 为 32 字节 AoS（origin.xyz + pad / dir.xyz + pad），与 WGSL b2
//       的「每像素 2 个 vec4」一致。

#include <metal_stdlib>
using namespace metal;

struct CameraParams {
    float4 eye;        // xyz = eye, w 未用
    float4 forward;    // xyz = 前向单位向量
    float4 right;      // xyz = 右向单位向量
    float4 up;         // xyz = 上向单位向量
    float  tan_half_fov;
    float  aspect;
    uint   width;
    uint   height;
};

struct RayOut {
    float4 origin;     // xyz = 射线起点
    float4 dir;        // xyz = 单位方向
};

kernel void primary_ray_gen(
    constant CameraParams& cam [[buffer(0)]],
    device   RayOut*       out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = cam.width * cam.height;
    // 尾组越界判断（与 DispatchShape::dispatch_cpu 语义同构）。
    if (gid >= total) {
        return;
    }
    uint x = gid % cam.width;
    uint y = gid / cam.width;
    float px = float(x) + 0.5f;
    float py = float(y) + 0.5f;
    float ndc_x = (2.0f * (px / float(cam.width)) - 1.0f) * cam.tan_half_fov * cam.aspect;
    float ndc_y = (1.0f - 2.0f * (py / float(cam.height))) * cam.tan_half_fov;
    float3 dir = normalize(cam.forward.xyz + cam.right.xyz * ndc_x + cam.up.xyz * ndc_y);
    out[gid].origin = float4(cam.eye.xyz, 0.0f);
    out[gid].dir = float4(dir, 0.0f);
}
"#;

#[cfg(test)]
mod v2g_tests {
    use super::*;
    /// 第二份 MSL 双份源在场且可查（三目标宿主表的 MSL 侧依赖它）。
    #[test]
    fn primary_ray_gen_msl_is_registered_and_well_formed() {
        assert!(!PRIMARY_RAY_GEN_MSL.trim().is_empty(), "MSL 源不得为空");
        assert!(
            PRIMARY_RAY_GEN_MSL.contains("kernel void primary_ray_gen"),
            "MSL 须声明 kernel void primary_ray_gen 入口"
        );
        // 头注三项元信息（同约定）。
        assert!(PRIMARY_RAY_GEN_MSL.contains("shaders/rt_primary_ray_gen.wgsl"));
        assert!(PRIMARY_RAY_GEN_MSL.contains("compute/mod.rs::primary_ray_gen_cpu"));
        assert!(PRIMARY_RAY_GEN_MSL.contains("[[buffer(0)]]"));
        assert!(PRIMARY_RAY_GEN_MSL.contains("[[buffer(1)]]"));
        // 语义锚点：像素中心 + NDC + 归一化 + 越界裁剪。
        for needle in ["+ 0.5f", "2.0f * (px", "normalize(", "gid >= total"] {
            assert!(
                PRIMARY_RAY_GEN_MSL.contains(needle),
                "MSL 缺关键语义：{needle}"
            );
        }
        // 禁 libm 超越函数调用（D-29 口径：调用形式 `sin(` 等）。
        for banned in ["sin(", "cos(", "tan(", "exp(", "pow(", "log("] {
            assert!(
                !PRIMARY_RAY_GEN_MSL.contains(banned),
                "MSL 出现被禁超越函数调用：{banned}"
            );
        }
        // 括号配平（防截断）。
        let opens = PRIMARY_RAY_GEN_MSL.matches('{').count();
        let closes = PRIMARY_RAY_GEN_MSL.matches('}').count();
        assert_eq!(opens, closes, "MSL 括号不配平（疑似被截断）");
    }
}

// ---------------------------------------------------------------------------
// 第三份手写 MSL 双份源（V2.1 T-A2）：光追场景整帧 kernel
// ---------------------------------------------------------------------------

/// `rt_scene.scene_main.msl` 的入口名。
pub const RT_SCENE_ENTRY: &str = "scene_main";

/// `rt_scene.scene_main.msl` 的文件名（与 [`RT_SCENE_MSL`] 的 `include_str!` 同源）。
pub const RT_SCENE_MSL_FILE: &str = "rt_scene.scene_main.msl";

/// 光追场景 kernel（`vulkan::spirv_rt` 的 MSL 同构移植）的双份源。
///
/// **源文件即权威**：这里用 `include_str!` 把
/// `crates/prism-gpu/src/metal/msl/rt_scene.scene_main.msl` 原样编进二进制——
/// 「双份源」不再是一份 Rust 字符串常量 + 一份没人看过的 .msl，而是**同一份
/// 文本**：改动必须落在 .msl 文件里，编译期即校验文件存在且为 UTF-8。
///
/// 归入「双份源」约定（`<wgsl 去扩展>.<entry>.msl`）：`rt_scene` + `scene_main`。
///
/// # 状态（G4/Q5 口径）
///
/// **未实机验证**：Linux 沙箱无 macOS 工具链与 Metal 运行时，本文件从未被
/// Apple 编译器读过，也从未在 `MTLDevice` 上 dispatch 过。
/// [`crate::metal::MetalBackend::is_available`] 因此**恒 `false`**——
/// 「有源码」≠「能编译」≠「能出图」，三者不得混为一谈。
pub const RT_SCENE_MSL: &str = include_str!("rt_scene.scene_main.msl");

#[cfg(test)]
mod v2a2_tests {
    use super::*;

    /// 第三份双份源在场、入口与绑定映射符合约定（跨平台文本检查）。
    #[test]
    fn rt_scene_msl_is_registered_and_well_formed() {
        assert!(!RT_SCENE_MSL.trim().is_empty(), "MSL 源不得为空");
        assert!(
            RT_SCENE_MSL.contains("kernel void scene_main"),
            "MSL 须声明 kernel void scene_main 入口"
        );
        assert_eq!(RT_SCENE_ENTRY, "scene_main");
        assert_eq!(RT_SCENE_MSL_FILE, "rt_scene.scene_main.msl");
        // 头注三项元信息：对应 SPIR-V / 判对基准 / 绑定映射。
        assert!(RT_SCENE_MSL.contains("vulkan/spirv_rt.rs"));
        assert!(RT_SCENE_MSL.contains("CpuRefBackend::render_scene"));
        for b in ["[[buffer(0)]]", "[[buffer(1)]]", "[[buffer(2)]]", "[[buffer(3)]]"] {
            assert!(RT_SCENE_MSL.contains(b), "MSL 缺绑定声明：{b}");
        }
        assert!(RT_SCENE_MSL.contains("[[thread_position_in_grid]]"));
        // G4：头注必须显式留名「未实机验证」。
        assert!(
            RT_SCENE_MSL.contains("未实机验证"),
            "MSL 头注必须写明未实机验证（G4/Q5）"
        );
    }

    /// 数值纪律：禁 `fma`（会改末位）/ 禁超越函数（D-29）/ 括号配平。
    ///
    /// 扫描前先剥掉 `//` 行注释——头注里**提到**这些被禁符号是合规的
    /// （正是为了让后来者知道它们为什么被禁），真正的约束在代码本体上。
    #[test]
    fn rt_scene_msl_respects_numeric_discipline() {
        let code: String = RT_SCENE_MSL
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "fma(", "fast::", "sin(", "cos(", "tan(", "exp(", "pow(", "log(", "rsqrt(",
        ] {
            assert!(
                !code.contains(banned),
                "MSL 代码体出现被禁调用：{banned}（fma 改末位；超越函数违 D-29）"
            );
        }
        // 归一化必须走带退化守卫的本地函数，不用内建 normalize(。
        assert!(
            code.contains("scene_normalize"),
            "MSL 须使用带退化守卫的 scene_normalize"
        );
        assert_eq!(
            code.matches("normalize(").count(),
            code.matches("scene_normalize(").count(),
            "MSL 不得调用无退化守卫的内建 normalize(（只允许 scene_normalize）"
        );
        let opens = RT_SCENE_MSL.matches('{').count();
        let closes = RT_SCENE_MSL.matches('}').count();
        assert_eq!(opens, closes, "MSL 括号不配平（疑似被截断）");
        // 两层循环必须在场（球 / 灯）。
        assert_eq!(code.matches("for (uint ").count(), 2, "应有球/灯两层循环");
        // 量化口径：floor(x + 0.5)，与 CPU 的 round 等价。
        assert!(code.contains("floor(scaled + 0.5f)"));
        // 打包字节序：r | g<<8 | b<<16 | 255<<24。
        assert!(code.contains("0xFF000000u | (b << 16) | (g << 8) | r"));
    }

    /// 与 Rust 侧编码器声明的槽位布局一致（`packed_float3` 是 32 字节槽的前提）。
    #[test]
    fn rt_scene_msl_slot_layout_matches_encoder() {
        assert!(
            RT_SCENE_MSL.contains("packed_float3 center"),
            "球心必须是 packed_float3（12 字节），否则 32 字节槽装不下"
        );
        assert!(RT_SCENE_MSL.contains("packed_float3 albedo"));
        assert!(RT_SCENE_MSL.contains("packed_float3 dir"));
        assert!(RT_SCENE_MSL.contains("packed_float3 color"));
        for decl in ["float  tan_half_fov", "float  aspect", "uint   num_spheres", "uint   num_lights"]
        {
            assert!(RT_SCENE_MSL.contains(decl), "SceneParams 缺字段：{decl}");
        }
    }
}
