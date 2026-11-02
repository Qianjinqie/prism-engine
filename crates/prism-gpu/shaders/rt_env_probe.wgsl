// rt_env_probe.wgsl —— A2 / W2：环境采样语义的**可判对探针**（V6 口径 b）。
//
// 同一份 `rt_env.wgsl`（`//!include` 展开，无第二份实现）在这里被包成
// "每线程一个查询"的最小 kernel：输入 = CDF blob（binding 0，`rt_env.wgsl`
// 声明），输出 = 每查询 12 个 f32（binding 1）。
//
// 为什么是探针而不是整条 wavefront：wavefront 的 `shade`/`extend` 要额外
// 一组绑定来承载 blob（group(0) 的 {0,1,2} 已被 slots/params/accum 占满，
// 且宿主 ABI 要求每个 entry 的绑定集是**连续前缀**），那需要同步改
// `web/abi.rs` 的 group 表与浏览器宿主——本轮未做（见报告"未接线"）。
// 探针覆盖的是**判据真正关心的东西**：CDF 反演 → 方向 → 双线性辐亮度 →
// 同源 pdf，四条链路逐值与 CPU `env.rs` 对拍。

//!include rt_env.wgsl

@group(0) @binding(1) var<storage, read_write> env_out: array<f32>;

/// 基 2 radical inverse（与 CPU 侧测试同式；无 RNG）。
fn env_radical_inverse(k0: u32) -> f32 {
  var k = k0;
  k = (k << 16u) | (k >> 16u);
  k = ((k & 0x55555555u) << 1u) | ((k & 0xAAAAAAAAu) >> 1u);
  k = ((k & 0x33333333u) << 2u) | ((k & 0xCCCCCCCCu) >> 2u);
  k = ((k & 0x0F0F0F0Fu) << 4u) | ((k & 0xF0F0F0F0u) >> 4u);
  k = ((k & 0x00FF00FFu) << 8u) | ((k & 0xFF00FF00u) >> 8u);
  return f32(k) * 2.3283064e-10;
}

@compute @workgroup_size(64)
fn env_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cap = arrayLength(&env_out) / 12u;
  let idx = gid.x;
  if (idx >= cap) {
    return;
  }
  // 查询对与 CPU 侧逐位同式：u1 = (i+0.5)/N（分层）、u2 = Hammersley 第二维。
  let u1 = (f32(idx) + 0.5) / f32(cap);
  let u2 = env_radical_inverse(idx ^ 0x9E3779B9u);
  let s = env_sample(u1, u2);
  let r = env_radiance(s.wi);
  let p = env_pdf(s.wi);
  let b = idx * 12u;
  env_out[b + 0u] = s.wi.x;
  env_out[b + 1u] = s.wi.y;
  env_out[b + 2u] = s.wi.z;
  env_out[b + 3u] = s.pdf;
  env_out[b + 4u] = r.x;
  env_out[b + 5u] = r.y;
  env_out[b + 6u] = r.z;
  env_out[b + 7u] = f32(s.texel);
  env_out[b + 8u] = p;
  // 诊断三元组：WGSL 自己的 uv、以及 `env_pdf` 内部归到的 texel 的亮斑下标
  // （未命中为 `ENV_INF_TEXEL`，f32 上 > 4.29e9）。用于区分"pdf 公式漂移"
  // 与"纹素边界两侧分类不同"这两种完全不同的原因。
  let uv = env_dir_to_uv(s.wi);
  env_out[b + 9u] = uv.x;
  env_out[b + 10u] = uv.y;
  var sidx = ENV_INF_TEXEL;
  let ew = env_w();
  let eh = env_h();
  if (ew > 0u && eh > 0u) {
    let pi = min(u32(uv.x * f32(ew)), ew - 1u);
    let pj = min(u32(uv.y * f32(eh)), eh - 1u);
    sidx = env_spot_index(pj * ew + pi);
  }
  env_out[b + 11u] = f32(sidx);
}
