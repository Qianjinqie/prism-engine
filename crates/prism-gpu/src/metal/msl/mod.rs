//! MSL「双份源」组织（Phase 4 定案：Metal 侧手写，非发射器）。
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
//! 本次**只落地一个真实样本**——[`CLOTH_PREDICT_MSL`]——作为规格示范。其余
//! kernel 待后续按同一约定逐个手写；**不**预先创建几十个空 `.msl` 占位。

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
// V2.0-G：compute-first 管线的第二个手写 MSL 双份源（主射线发射）
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
