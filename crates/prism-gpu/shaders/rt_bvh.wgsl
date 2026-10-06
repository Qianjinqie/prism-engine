// prism-rt BVH 遍历 kernel（D2）——对应 prism-geometry/bvh.rs 的 ray_intersect。
//
// # 与 CpuRef 对齐
// - 扁平节点数组（左子恒 = id+1，内部节点 right_child 异于 0），同 bvh.rs。
// - 显式栈遍历（深度上界 64），近子先处理，命中以 (t, tri) 字典序 tie-break。
// - slab 法盒交（inv 分量；exit 侧 +1e-7 外扩消除边界漏交）同 bvh.rs::Aabb::ray。
// - Möller–Trumbore 三角形命中（禁 fma）同 bvh.rs::tri_ray_hit。

//!include rt_common.wgsl
struct BvhNode {
  bmin: vec3<f32>,
  bmax: vec3<f32>,
  start: u32,        // 叶子：图元区间起点；内部：0
  count: u32,        // 叶子：图元数；内部：0
  right_child: u32,  // 内部：右子索引；叶子：0
  _pad: u32,
}

struct Tri {
  a: vec3<f32>,
  _p0: f32,
  b: vec3<f32>,
  _p1: f32,
  c: vec3<f32>,
  _p2: f32,
}

// slab 盒交（与 bvh.rs::Aabb::ray 同式；exit 侧 +1e-7）。
fn aabb_ray(bmin: vec3<f32>, bmax: vec3<f32>, ro: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  let t1 = (bmin - ro) * inv;
  let t2 = (bmax - ro) * inv;
  let lo = min(t1, t2);
  let hi = max(t1, t2);
  let enter = max(max(lo.x, max(lo.y, lo.z)), t_min);
  let exit = min(min(hi.x, min(hi.y, hi.z)), t_max);
  if (enter <= exit + 1e-7) { return vec2<f32>(enter, exit); }
  return vec2<f32>(1e30, -1e30); // 空区间哨兵
}

// Möller–Trumbore（禁 fma；与 bvh.rs::tri_ray_hit 同式）。
fn tri_ray_hit(tri: Tri, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
  let e1 = tri.b - tri.a;
  let e2 = tri.c - tri.a;
  let pvec = cross(rd, e2);
  let det = vdot(e1, pvec);
  if (abs(det) < 1e-12) { return -1.0; }
  let inv_det = 1.0 / det;
  let tvec = ro - tri.a;
  let u = vdot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) { return -1.0; }
  let qvec = cross(tvec, e1);
  let v = vdot(rd, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) { return -1.0; }
  let hit = vdot(e2, qvec) * inv_det;
  if (hit >= t_min && hit <= t_max) { return hit; }
  return -1.0;
}

// 最近命中：返回 (t, tri_index)；未命中 tri_index = 0xFFFFFFFF。
// WGSL 形参不能是 runtime-sized 类型，storage 数组须以
// `ptr<storage, array<T>, read>` 传入。动态索引写法（真机逐一实测确认）：
//   `p + i`        → 非法（storage 指针无 operator+ 重载）
//   `*(p + i32(i))`→ 非法（同上）
//   `(*p)[i]`      → 合法（先解引用成 array 引用再索引）
//   绑到局部 let   → 非法（runtime-sized 数组不可作函数作用域变量）
fn bvh_intersect(nodes: ptr<storage, array<BvhNode>, read>, tris: ptr<storage, array<Tri>, read>, node_count: u32,
                 ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  var best_t = t_max;
  var best_tri = 0xFFFFFFFFu;
  if (node_count == 0u) { return vec2<f32>(best_t, f32(best_tri)); }
  let inv = vec3<f32>(1.0/rd.x, 1.0/rd.y, 1.0/rd.z);
  // 固定深度显式栈（bvh.rs 契约：64 层必够）。
  var stack: array<u32, 64>;
  var tlo: array<f32, 64>;
  var sp: i32 = 0;
  stack[0] = 0u;
  tlo[0] = t_min;
  sp = 1;
  while (sp > 0) {
    sp = sp - 1;
    let id = stack[sp];
    let t_lo = tlo[sp];
    if (t_lo > best_t) { continue; }
    let node = (*nodes)[id];
    if (node.count > 0u) {
      var i = 0u;
      while (i < node.count) {
        let slot = node.start + i;
        let ht = tri_ray_hit((*tris)[slot], ro, rd, t_min, best_t);
        if (ht >= 0.0 && ht < best_t) {
          best_t = ht;
          best_tri = slot;
        }
        i = i + 1u;
      }
    } else {
      let l = id + 1u;
      let r = node.right_child;
      let hl = aabb_ray((*nodes)[l].bmin, (*nodes)[l].bmax, ro, inv, t_min, t_max);
      let hr = aabb_ray((*nodes)[r].bmin, (*nodes)[r].bmax, ro, inv, t_min, t_max);
      let ok_l = hl.x <= hl.y;
      let ok_r = hr.x <= hr.y;
      // 近子后入栈（先处理）；进入 t 相等先左（确定性 tie-break）。
      if (ok_l && ok_r) {
        if (hl.x <= hr.x) {
          stack[sp] = r; tlo[sp] = min(hr.x, hr.y); sp = sp + 1;
          stack[sp] = l; tlo[sp] = min(hl.x, hl.y); sp = sp + 1;
        } else {
          stack[sp] = l; tlo[sp] = min(hl.x, hl.y); sp = sp + 1;
          stack[sp] = r; tlo[sp] = min(hr.x, hr.y); sp = sp + 1;
        }
      } else if (ok_l) {
        stack[sp] = l; tlo[sp] = min(hl.x, hl.y); sp = sp + 1;
      } else if (ok_r) {
        stack[sp] = r; tlo[sp] = min(hr.x, hr.y); sp = sp + 1;
      }
    }
  }
  return vec2<f32>(best_t, f32(best_tri));
}
