// scene_frame.wgsl — 光追场景整帧 kernel（V2.2-A-1 / T03 + A-2 曲线扩绑定）。
//
// 第 4 份手写场景源（WGSL 文本态），与 `vulkan/spirv_rt`（SPIR-V 发射器）、
// `metal`（本仓不提供独立后端）共享同一套参数块与绑定剖面
// （真源见 `crate::scene_abi`）。本文件只负责"语义逐句对应"
// `lib.rs` 的 `CpuRefBackend::render_scene`（相机基 → 球循环最近命中 →
// A-2 曲线 BVH 遍历最近命中 → 灯循环漫反射 → teal/contrast/filmic → 量化打包）。
//
// 设计约束（与 prism-gpu 既有门禁一致）：
// - 禁超越函数：不使用任何"逐 GPU 实现精度漂移"的内建（无 sin/cos/exp/pow/log），
//   相机焦距标度 `tan(fov/2)` 由 CPU 预算成 `P_TAN_HALF_FOV` 位型传入（D-29）。
//   曲线段同样只用 `+ − × ÷ sqrt floor`（Möller–Trumbore + slab 盒交）。
// - 不写 `fma()`（WGSL 无保证的融合乘加，破坏跨端逐位一致性）。
// - 浮点一律位型：参数块与实体缓冲里的 f32 以 `to_bits()` 存入，shader 侧
//   `bitcast<f32>` 还原；不存在第二套十进制/文本往返口径。
// - 绑定连续前缀 **V2 = `{0..5}`** = params / spheres / lights / triangles / nodes / out，
//   **输出恒为最后一个绑定**（满足 `web/mod.rs` 的"每个 group 的绑定须从 0 起连续"
//   硬约束；V1 剖面 `{0..3}` 仍由 `scene_abi::SceneAbiProfile::V1` 描述，供既有
//   Vulkan dispatch 编码使用，不回归）。

// ---- 参数块 / 槽位（与 `crate::scene_abi` 逐字一致；qa 测试钉偏移） ----
const PARAMS_WORDS_: u32 = 24u;   // 必须等于 scene_abi::PARAMS_WORDS_V2
const SPHERE_STRIDE: u32 = 8u;    // 每球 f32 槽数，必须等于 scene_abi::SPHERE_SLOTS
const LIGHT_STRIDE: u32 = 8u;     // 每灯 f32 槽数，必须等于 scene_abi::LIGHT_SLOTS
const TRI_STRIDE: u32 = 12u;      // 每三角形 f32 槽数，必须等于 scene_abi::TRIANGLE_SLOTS
const NODE_STRIDE: u32 = 12u;     // 每 BVH 节点 f32 槽数，必须等于 scene_abi::NODE_SLOTS

// 参数块各字偏移（u32 槽位）。
const P_WIDTH: u32 = 0u;
const P_HEIGHT: u32 = 1u;
const P_EYE: u32 = 2u;
const P_FORWARD: u32 = 5u;
const P_RIGHT: u32 = 8u;
const P_UP: u32 = 11u;
const P_TAN_HALF_FOV: u32 = 14u;
const P_ASPECT: u32 = 15u;
const P_TEAL_ORANGE: u32 = 16u;
const P_CONTRAST: u32 = 17u;
const P_CONTRAST_ON: u32 = 18u;        // contrast != 1.0 的预计算开关（CPU 侧判定）
const P_FILMIC: u32 = 19u;
const P_NUM_SPHERES: u32 = 20u;
const P_NUM_LIGHTS: u32 = 21u;
const P_NUM_TRIS: u32 = 22u;
const P_NUM_NODES: u32 = 23u;

// 曲线命中面的反照率（与 `crate::CURVE_ALBEDO` 同值；A-2 切片曲面材质参数化留 V2.3）。
const CURVE_ALBEDO: vec3<f32> = vec3<f32>(0.72, 0.74, 0.78);
// 曲线求交的 t_min（与 `crate::CURVE_T_MIN` 同值）。
const CURVE_T_MIN: f32 = 1e-4;

// ---- 绑定（V2 连续前缀 0..5；输出恒为最后一个） ----
@group(0) @binding(0) var<storage, read>       params: array<u32>;
@group(0) @binding(1) var<storage, read>       spheres: array<u32>;
@group(0) @binding(2) var<storage, read>       lights: array<u32>;
@group(0) @binding(3) var<storage, read>       triangles: array<u32>;
@group(0) @binding(4) var<storage, read>       nodes: array<u32>;
@group(0) @binding(5) var<storage, read_write> outbuf: array<u32>;

// 把颜色各分量钳到 [0,1]。
fn clamp01(c: vec3<f32>) -> vec3<f32> {
  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// 分量相乘（albedo * 光照项）。
fn vmul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(a.x * b.x, a.y * b.y, a.z * b.z);
}

// 单位化（与 `prism_math::Vec3::normalize` 同口径：长度 ≤ 1e-8 退化到零向量）。
fn vnormalize(v: vec3<f32>) -> vec3<f32> {
  let l = sqrt(dot(v, v));
  if (l > 1e-8) {
    return v / l;
  }
  return vec3<f32>(0.0, 0.0, 0.0);
}

// RGBA8 little-endian 打包：内存字节序 [r, g, b, a]（与 `background.wgsl` 同式）。
fn pack_rgba(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x, 0.0, 1.0) * 255.0);
  let g = u32(clamp(c.g, 0.0, 1.0) * 255.0);
  let b = u32(clamp(c.b, 0.0, 1.0) * 255.0);
  let a = 255u;
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

// ---- 扁平缓冲取数（槽位 → 向量） ----

fn node_bmin(i: u32) -> vec3<f32> {
  let b = i * NODE_STRIDE;
  return vec3<f32>(bitcast<f32>(nodes[b]), bitcast<f32>(nodes[b + 1u]), bitcast<f32>(nodes[b + 2u]));
}

fn node_bmax(i: u32) -> vec3<f32> {
  let b = i * NODE_STRIDE + 4u;
  return vec3<f32>(bitcast<f32>(nodes[b]), bitcast<f32>(nodes[b + 1u]), bitcast<f32>(nodes[b + 2u]));
}

fn tri_vertex(slot: u32, base: u32) -> vec3<f32> {
  let b = slot * TRI_STRIDE + base;
  return vec3<f32>(
    bitcast<f32>(triangles[b]),
    bitcast<f32>(triangles[b + 1u]),
    bitcast<f32>(triangles[b + 2u]));
}

// slab 盒交（exit 侧 +1e-7 外扩消除边界漏交；与 `rt_bvh.wgsl` / `bvh.rs::Aabb::ray` 同式）。
// 返回 (enter, exit)；空区间返回哨兵 (1e30, -1e30)。
fn aabb_ray(bmin: vec3<f32>, bmax: vec3<f32>, ro: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  let t1 = (bmin - ro) * inv;
  let t2 = (bmax - ro) * inv;
  let lo = min(t1, t2);
  let hi = max(t1, t2);
  let enter = max(max(lo.x, max(lo.y, lo.z)), t_min);
  let exitv = min(min(hi.x, min(hi.y, hi.z)), t_max);
  if (enter <= exitv + 1e-7) {
    return vec2<f32>(enter, exitv);
  }
  return vec2<f32>(1e30, -1e30);
}

// Möller–Trumbore（禁 fma；与 `rt_bvh.wgsl::tri_ray_hit` / `bvh.rs::tri_ray_hit` 同式）。
// 命中返回 t，未命中返回 -1.0。
fn tri_ray_hit(slot: u32, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
  let va = tri_vertex(slot, 0u);
  let vb = tri_vertex(slot, 4u);
  let vc = tri_vertex(slot, 8u);
  let e1 = vb - va;
  let e2 = vc - va;
  let pvec = cross(rd, e2);
  let det = dot(e1, pvec);
  if (abs(det) < 1e-12) {
    return -1.0;
  }
  let inv_det = 1.0 / det;
  let tvec = ro - va;
  let u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return -1.0;
  }
  let qvec = cross(tvec, e1);
  let v = dot(rd, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return -1.0;
  }
  let hit = dot(e2, qvec) * inv_det;
  if (hit >= t_min && hit <= t_max) {
    return hit;
  }
  return -1.0;
}

// 三角形几何法线（朝向来光侧；与 CPU 参考 `normal_unnormalized().normalize()` +
// `if dot(n, dir) > 0 { n = -n }` 同口径）。
fn tri_normal(slot: u32, rd: vec3<f32>) -> vec3<f32> {
  let va = tri_vertex(slot, 0u);
  let vb = tri_vertex(slot, 4u);
  let vc = tri_vertex(slot, 8u);
  var n = vnormalize(cross(vb - va, vc - va));
  if (dot(n, rd) > 0.0) {
    n = n * -1.0;
  }
  return n;
}

// 栈式 BVH 遍历（最近命中）：返回 (t, tri_slot)；未命中 tri_slot = 0xFFFFFFFF。
// 语义逐句移植 `rt_bvh.wgsl::bvh_intersect`：固定深度显式栈（64）、近子先处理、
// 进入 t 相等先左（确定性 tie-break）。缓冲是扁平 `array<u32>`（位型），故取数
// 经 `node_bmin`/`node_bmax` 解码（与 `BvhNode` 结构体版数值等价）。
fn bvh_intersect(node_count: u32, tri_count: u32,
                 ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  var best_t = t_max;
  var best_tri = 0xFFFFFFFFu;
  if (node_count == 0u || tri_count == 0u) {
    return vec2<f32>(best_t, f32(best_tri));
  }
  let inv = vec3<f32>(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
  var stack: array<u32, 64>;
  var tlo: array<f32, 64>;
  stack[0] = 0u;
  tlo[0] = t_min;
  var sp: i32 = 1;
  loop {
    if (sp <= 0) {
      break;
    }
    sp = sp - 1;
    let id = stack[sp];
    let t_lo = tlo[sp];
    if (t_lo > best_t) {
      continue;
    }
    let nb = id * NODE_STRIDE;
    let count = nodes[nb + 9u];
    if (count > 0u) {
      let start = nodes[nb + 8u];
      var i: u32 = 0u;
      loop {
        if (i >= count) {
          break;
        }
        let slot = start + i;
        let ht = tri_ray_hit(slot, ro, rd, t_min, best_t);
        if (ht >= 0.0 && ht < best_t) {
          best_t = ht;
          best_tri = slot;
        }
        i = i + 1u;
      }
    } else {
      let l = id + 1u;
      let r = nodes[nb + 10u];
      let hl = aabb_ray(node_bmin(l), node_bmax(l), ro, inv, t_min, t_max);
      let hr = aabb_ray(node_bmin(r), node_bmax(r), ro, inv, t_min, t_max);
      let ok_l = hl.x <= hl.y;
      let ok_r = hr.x <= hr.y;
      // 近子后入栈（先处理）；进入 t 相等先左（确定性 tie-break）。
      if (ok_l && ok_r) {
        if (hl.x <= hr.x) {
          stack[sp] = r;
          tlo[sp] = min(hr.x, hr.y);
          sp = sp + 1;
          stack[sp] = l;
          tlo[sp] = min(hl.x, hl.y);
          sp = sp + 1;
        } else {
          stack[sp] = l;
          tlo[sp] = min(hl.x, hl.y);
          sp = sp + 1;
          stack[sp] = r;
          tlo[sp] = min(hr.x, hr.y);
          sp = sp + 1;
        }
      } else if (ok_l) {
        stack[sp] = l;
        tlo[sp] = min(hl.x, hl.y);
        sp = sp + 1;
      } else if (ok_r) {
        stack[sp] = r;
        tlo[sp] = min(hr.x, hr.y);
        sp = sp + 1;
      }
    }
  }
  return vec2<f32>(best_t, f32(best_tri));
}

@compute @workgroup_size(8, 8, 1)
fn scene_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params[P_WIDTH];
  let h = params[P_HEIGHT];
  // 尾组守卫：越界 invocation 直接返回（与 CPU 的 `for py in 0..h / for px in 0..w` 同口径）。
  if (gid.x >= w || gid.y >= h) {
    return;
  }
  let idx = gid.y * w + gid.x;

  // ---- 相机基（CPU 已算好，逐字段取出；与 `camera_basis` 同布局） ----
  let eye = vec3<f32>(
    bitcast<f32>(params[P_EYE]),
    bitcast<f32>(params[P_EYE + 1u]),
    bitcast<f32>(params[P_EYE + 2u]));
  let forward = vec3<f32>(
    bitcast<f32>(params[P_FORWARD]),
    bitcast<f32>(params[P_FORWARD + 1u]),
    bitcast<f32>(params[P_FORWARD + 2u]));
  let right = vec3<f32>(
    bitcast<f32>(params[P_RIGHT]),
    bitcast<f32>(params[P_RIGHT + 1u]),
    bitcast<f32>(params[P_RIGHT + 2u]));
  let up = vec3<f32>(
    bitcast<f32>(params[P_UP]),
    bitcast<f32>(params[P_UP + 1u]),
    bitcast<f32>(params[P_UP + 2u]));
  let tan_f = bitcast<f32>(params[P_TAN_HALF_FOV]);
  let aspect = bitcast<f32>(params[P_ASPECT]);
  let teal = params[P_TEAL_ORANGE];
  let contrast = bitcast<f32>(params[P_CONTRAST]);
  let contrast_on = params[P_CONTRAST_ON];
  let filmic = params[P_FILMIC];
  let num_spheres = params[P_NUM_SPHERES];
  let num_lights = params[P_NUM_LIGHTS];
  let num_tris = params[P_NUM_TRIS];
  let num_nodes = params[P_NUM_NODES];

  // ---- 每像素射线（与 CPU 参考逐式对应） ----
  let nx = (2.0 * (f32(gid.x) + 0.5) / f32(w) - 1.0) * tan_f * aspect;
  let ny = (1.0 - 2.0 * (f32(gid.y) + 0.5) / f32(h)) * tan_f;
  let dir = normalize(forward + right * nx + up * ny);
  let origin = eye;

  // ---- 球循环：最近命中（t = -b - sqrt(disc)） ----
  var closest_t: f32 = 0.0;
  var hit_idx: i32 = -1;                     // -1 = 未命中
  var j: u32 = 0u;
  loop {
    if (j >= num_spheres) {
      break;
    }
    let base = j * SPHERE_STRIDE;
    let center = vec3<f32>(
      bitcast<f32>(spheres[base]),
      bitcast<f32>(spheres[base + 1u]),
      bitcast<f32>(spheres[base + 2u]));
    let radius = bitcast<f32>(spheres[base + 3u]);
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc >= 0.0) {
      let t = -b - sqrt(disc);
      if (t > 0.0 && (hit_idx < 0 || t < closest_t)) {
        closest_t = t;
        hit_idx = i32(j);
      }
    }
    j = j + 1u;
  }

  // ---- A-2 曲线：栈式 BVH 遍历（覆盖球命中若更近；曲线为空 ⇒ 整段跳过） ----
  var curve_slot: i32 = -1;
  if (num_tris > 0u && num_nodes > 0u) {
    let t_max_curve = select(1e30, closest_t, hit_idx >= 0);
    let hitc = bvh_intersect(num_nodes, num_tris, origin, dir, CURVE_T_MIN, t_max_curve);
    if (hitc.y != f32(0xFFFFFFFFu)) {
      closest_t = hitc.x;
      curve_slot = i32(hitc.y);
      hit_idx = -1;
    }
  }

  // ---- 着色 ----
  var color = vec3<f32>(0.04, 0.05, 0.08);   // 背景色（与 CPU 参考同值）
  let ambient = vec3<f32>(0.08, 0.08, 0.08);
  if (hit_idx >= 0) {
    let base = u32(hit_idx) * SPHERE_STRIDE; // 命中球的真实下标
    let center = vec3<f32>(
      bitcast<f32>(spheres[base]),
      bitcast<f32>(spheres[base + 1u]),
      bitcast<f32>(spheres[base + 2u]));
    let p = origin + dir * closest_t;
    let n = normalize(p - center);
    let albedo = vec3<f32>(
      bitcast<f32>(spheres[base + 4u]),
      bitcast<f32>(spheres[base + 5u]),
      bitcast<f32>(spheres[base + 6u]));
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var l: u32 = 0u;
    loop {
      if (l >= num_lights) {
        break;
      }
      let lbase = l * LIGHT_STRIDE;
      // 方向已在 CPU 侧归一化（`encode_lights`），此处直接取用。
      let ldir = vec3<f32>(
        bitcast<f32>(lights[lbase]),
        bitcast<f32>(lights[lbase + 1u]),
        bitcast<f32>(lights[lbase + 2u]));
      let lcolor = vec3<f32>(
        bitcast<f32>(lights[lbase + 3u]),
        bitcast<f32>(lights[lbase + 4u]),
        bitcast<f32>(lights[lbase + 5u]));
      let lint = bitcast<f32>(lights[lbase + 6u]);
      let diff = max(dot(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + vmul(albedo, term);
      l = l + 1u;
    }
    color = acc;
  } else if (curve_slot >= 0) {
    // 曲线命中：几何法线 + 固定中性反照率（与 CPU 参考逐式相同）。
    let n = tri_normal(u32(curve_slot), dir);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var l: u32 = 0u;
    loop {
      if (l >= num_lights) {
        break;
      }
      let lbase = l * LIGHT_STRIDE;
      let ldir = vec3<f32>(
        bitcast<f32>(lights[lbase]),
        bitcast<f32>(lights[lbase + 1u]),
        bitcast<f32>(lights[lbase + 2u]));
      let lcolor = vec3<f32>(
        bitcast<f32>(lights[lbase + 3u]),
        bitcast<f32>(lights[lbase + 4u]),
        bitcast<f32>(lights[lbase + 5u]));
      let lint = bitcast<f32>(lights[lbase + 6u]);
      let diff = max(dot(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + vmul(CURVE_ALBEDO, term);
      l = l + 1u;
    }
    color = acc;
  }

  // ---- 后处理（与 CPU 参考同式，开关由 CPU 侧预算成 u32 标志） ----
  if (teal != 0u) {
    let luma = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
    let shadow_tint = vec3<f32>(0.10, 0.35, 0.45);
    let high_tint = vec3<f32>(1.0, 0.7, 0.3);
    let shadow_comp = vmul(shadow_tint, color) * (1.0 - luma) * 0.5;
    let high_comp = vmul(high_tint, color) * luma * 0.5;
    color = color * 0.6 + shadow_comp + high_comp;
  }

  if (contrast_on != 0u) {
    color = (color - vec3<f32>(0.5, 0.5, 0.5)) * contrast + vec3<f32>(0.5, 0.5, 0.5);
  }

  if (filmic != 0u) {
    // 简易 Reinhard 色调映射（与 CPU 参考同式）。
    color = color / (color + vec3<f32>(1.0, 1.0, 1.0));
  }

  color = clamp01(color);
  outbuf[idx] = pack_rgba(color);
}
