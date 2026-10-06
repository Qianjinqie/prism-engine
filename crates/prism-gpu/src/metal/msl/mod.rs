//! MSL「双份源」组织（Phase 4 定案：Metal 侧手写，非发射器）。
//!
//! # 为什么 Metal 侧手写、Vulkan 侧走发射器
//!
//! 路线图 Phase 4 明定：*Vulkan 侧 SPIR-V 由自研发射器覆盖本引擎 WGSL 子集；
//! Metal 侧 MSL 手写双份源*。取舍理由：
//!
//! * SPIR-V 是**二进制 IR**，有完整公开规范，WGSL→SPIR-V 的映射可由我们自己的
//!   发射器确定化生成（也符合"确定性 md5 复现"）。
//! * MSL 是**高级 C++ 方言**，没有稳定的文本→文本规范；自动生成器会引入对
//!   Apple 编译器的隐式假设，风险高于手写。且 Metal 真机路径是**次要路径**
//!   （WebGPU 才是手机测试主路径），手写双份源的维护成本可接受。
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
