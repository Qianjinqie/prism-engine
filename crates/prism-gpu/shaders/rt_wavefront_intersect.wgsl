// rt_wavefront_intersect.wgsl —— compute-first 管线的
// **wavefront 求交一步** kernel（射线批 → 命中批）。
//
// 语义权威：crates/prism-gpu/src/compute/mod.rs::wavefront_intersect_step_cpu
// 布局权威：compute::KernelDesc("rt_wavefront_intersect") ——
//   b0 (read)  : 射线批 [origin, dir]（每条 2 个 vec4）
//   b1 (read)  : 球体批 [center.xyz, radius]（每个 1 个 vec4）
//   b2 (write) : 命中批 [t, prim, hit_flag, 0]（每条 1 个 vec4）
// 判对基准：CPU 参考实现（t 最近优先；平局按下标升序 = 确定性 tie-break）
//
// 禁 libm（D-29）：只用 dot / sqrt。

@group(0) @binding(0) var<storage, read> ray_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sphere_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hit_buf: array<vec4<f32>>;

// 与 CPU 参考同值的最近命中下界。
const T_MIN: f32 = 1e-4;

// 单条射线对单球的求交；未命中返回 -1.0（与 CPU 的 Option::None 对应）。
fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = ro - c;
  let b = dot(oc, rd);
  let cc = dot(oc, oc) - r * r;
  let disc = b * b - cc;
  if (disc < 0.0) {
    return -1.0;
  }
  let sq = sqrt(disc);
  let t0 = -b - sq;
  let t1 = -b + sq;
  if (t1 < T_MIN) {
    return -1.0;
  }
  if (t0 >= T_MIN) {
    return t0;
  }
  return t1;
}

@compute @workgroup_size(64, 1, 1)
fn wavefront_intersect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&hit_buf);
  if (gid.x >= n) {
    return;
  }
  let slot = gid.x * 2u;
  let ro = ray_buf[slot].xyz;
  let rd = ray_buf[slot + 1u].xyz;
  var best_t = 1.0e30;
  var best_prim = -1;
  let m = arrayLength(&sphere_buf);
  for (var i = 0u; i < m; i = i + 1u) {
    let s = sphere_buf[i];
    let t = intersect_sphere(ro, rd, s.xyz, s.w);
    // 严格小于 ⇒ 平局保留下标更小者（与 CPU tie-break 同规则）。
    if (t >= 0.0 && t < best_t) {
      best_t = t;
      best_prim = i32(i);
    }
  }
  let flag = select(0.0, 1.0, best_prim >= 0);
  hit_buf[gid.x] = vec4<f32>(best_t, f32(best_prim), flag, 0.0);
}
