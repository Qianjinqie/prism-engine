// rt_primary_ray_gen.wgsl —— V2.0-G compute-first 管线的**主射线发射** kernel。
//
// 语义权威：crates/prism-gpu/src/compute/mod.rs::primary_ray_gen_cpu
//           （CPU 参考实现，同序 dispatch，逐位一致）
// 布局权威：compute::KernelDesc("rt_primary_ray_gen") ——
//   b0 (read)  : 相机基 [eye, forward, right, up]（各占一个 vec4 槽）
//   b1 (read)  : 标量 [tan_half_fov, aspect, width, height]
//   b2 (write) : 输出射线 [origin, dir]（每像素 2 个 vec4）
// 判对基准：CPU 参考实现（parity 网见 prism-rt::wavefront 的 double gate）
//
// 说明：本文件自包含（不依赖任何包含指令），便于真机单独编译；
// 禁 libm（D-29）——只用 dot / normalize / 四则运算。

@group(0) @binding(0) var<storage, read> cam_basis: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cam_scalar: array<f32>;
@group(0) @binding(2) var<storage, read_write> rays_out: array<vec4<f32>>;

// 行主序线性下标 = 输出槽起点（每像素 2 个 vec4）。
fn ray_slot(x: u32, y: u32, width: u32) -> u32 {
  return (y * width + x) * 2u;
}

@compute @workgroup_size(64, 1, 1)
fn primary_ray_gen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(cam_scalar[2]);
  let height = u32(cam_scalar[3]);
  // 尾组越界判断（与 DispatchShape::dispatch_cpu 的「越界调用照样执行」同构）。
  if (gid.x >= width || gid.y >= height) {
    return;
  }
  let tan_half = cam_scalar[0];
  let aspect = cam_scalar[1];
  let eye = cam_basis[0].xyz;
  let forward = cam_basis[1].xyz;
  let right = cam_basis[2].xyz;
  let up = cam_basis[3].xyz;
  // 像素中心 + 半像素偏移。
  let px = f32(gid.x) + 0.5;
  let py = f32(gid.y) + 0.5;
  let ndc_x = (2.0 * (px / f32(width)) - 1.0) * tan_half * aspect;
  let ndc_y = (1.0 - 2.0 * (py / f32(height))) * tan_half;
  let dir = normalize(forward + right * ndc_x + up * ndc_y);
  let i = ray_slot(gid.x, gid.y, width);
  rays_out[i] = vec4<f32>(eye, 0.0);
  rays_out[i + 1u] = vec4<f32>(dir, 0.0);
}
