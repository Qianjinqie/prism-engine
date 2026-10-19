// spv_mul2_probe.wgsl —— SPIR-V 装配探针 kernel 的 WGSL 宿主源。
//
// 与手写 SPIR-V（crates/prism-gpu/src/vulkan/spirv.rs 的 `spv_mul2_probe`）
// **语义完全一致**：dst[i] = src[i] * 2.0。二者由测试交叉锚定（LocalSize /
// 入口名 / 绑定号 / 语义），使「手写字节码」与「WGSL 权威源」不会漂移。
//
// 布局权威：compute::KernelDesc("spv_mul2_probe") —— b0 读 src、b1 写 dst。

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn mul2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&dst);
  if (gid.x >= n) {
    return;
  }
  dst[gid.x] = src[gid.x] * 2.0;
}
