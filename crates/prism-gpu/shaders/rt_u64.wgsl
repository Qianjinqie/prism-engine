// prism-rt wavefront 路径模式 —— 64 位整数哈希仿真（PCG RXS-M-XS）+ 采样流。
//
// # 为什么需要本文件（T-B R-1 逐位对拍的根基）
//
// CPU 侧的采样器（`prism-rt/src/sampler.rs` + `prism-noise/src/pcg.rs`）全部
// 建立在 **64 位整数哈希**上：`hash4(种子, 像素, 帧, 节点)` 定像素基种子、
// `hash1(base, IDX_SALT)` 定 Sobol 索引、`hash2_finish(prefix, dim)` 定打散
// 掩码、`hash3(aux_seed, dim, step, AUX_SALT)` 定游程辅助流。WGSL 没有 64 位
// 整数，故这里用 **u32 对 (lo, hi) 仿真 u64**——乘法走 16 位 limb 进位、加法
// 走进位、移位走高低拼接。全部是整数运算，仿真与真 64 位**逐位相同**（不是
// 近似）：这是 CPU↔GPU 同一随机数流的唯一途径，逐位对拍由此才有定义。
//
// # 与 CPU 实现的对应（判对基准 = prism-noise/src/pcg.rs）
//
// | WGSL               | CPU                          |
// |--------------------|------------------------------|
// | `pcg_permute`      | `pcg::permute`               |
// | `pcg_hash1`        | `pcg::hash1`                 |
// | `pcg_hash2_prefix` | `pcg::hash2_prefix`          |
// | `pcg_hash2_finish` | `pcg::hash2_finish`          |
// | `pcg_hash4`        | `pcg::hash4`                 |
// | `pcg_hash3`        | `pcg::hash3`                 |
//
// 常量逐字取自 pcg.rs（高低 32 位拆分由脚本生成，注释附原值）。

const PCG_MULT_LO: u32 = 0x4c957f2du; // 6364136223846793005 = 0x5851F42D4C957F2D
const PCG_MULT_HI: u32 = 0x5851f42du;
const PCG_INC_LO: u32 = 0xf767814fu; // 1442695040888963407 = 0x14057B7EF767814F
const PCG_INC_HI: u32 = 0x14057b7eu;
const PCG_MULT2_LO: u32 = 0x108f21b9u; // 12605985483714929081 = 0xAEF17502108F21B9
const PCG_MULT2_HI: u32 = 0xaef17502u;
const SALT1_LO: u32 = 0x7f4a7c15u; // 0x9E3779B97F4A7C15
const SALT1_HI: u32 = 0x9e3779b9u;
const SALT2_LO: u32 = 0x27d4eb4fu; // 0xC2B2AE3D27D4EB4F
const SALT2_HI: u32 = 0xc2b2ae3du;
const SALT3_LO: u32 = 0x9e3779f9u; // 0x165667B19E3779F9
const SALT3_HI: u32 = 0x165667b1u;
const SALT4_LO: u32 = 0x165667c5u; // 0x27D4EB2F165667C5
const SALT4_HI: u32 = 0x27d4eb2fu;
const IDX_SALT_LO: u32 = 0x85a308d3u; // 0x243F6A8885A308D3
const IDX_SALT_HI: u32 = 0x243f6a88u;
const MASK_SALT_LO: u32 = 0x03707344u; // 0x13198A2E03707344
const MASK_SALT_HI: u32 = 0x13198a2eu;
const AUX_SALT_LO: u32 = 0x7f4a7c15u; // 0x9E3779B97F4A7C15（与 SALT1 同值）
const AUX_SALT_HI: u32 = 0x9e3779b9u;

// ---- u64 基础件（(lo, hi) u32 对）--------------------------------------

/// 64 位加法：(al,ah) + (bl,bh)，返回 (lo, hi)。
fn u64_add(al: u32, ah: u32, bl: u32, bh: u32) -> vec2<u32> {
  let lo = al + bl; // 溢出回绕 = mod 2^32，正确
  let carry = select(0u, 1u, lo < al);
  return vec2<u32>(lo, ah + bh + carry);
}

/// 64 位乘法：(al,ah) × (bl,bh) mod 2^64，返回 (lo, hi)。
///
/// 16 位 limb schoolbook：每个 16×16 部分积 < 2^32（u32 精确），
/// 逐个加进 limb 数组并即时进位归一化（每个 limb 加完后 < 2^16），
/// 故任何中间量都不溢出。mod 2^64 = 只保留 limb 0..3。
fn u64_mul(al: u32, ah: u32, bl: u32, bh: u32) -> vec2<u32> {
  var a: array<u32, 4>;
  var b: array<u32, 4>;
  a[0] = al & 0xFFFFu;
  a[1] = al >> 16u;
  a[2] = ah & 0xFFFFu;
  a[3] = ah >> 16u;
  b[0] = bl & 0xFFFFu;
  b[1] = bl >> 16u;
  b[2] = bh & 0xFFFFu;
  b[3] = bh >> 16u;
  var c: array<u32, 5>;
  c[0] = 0u;
  c[1] = 0u;
  c[2] = 0u;
  c[3] = 0u;
  c[4] = 0u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    for (var j = 0u; i + j < 4u; j = j + 1u) {
      let p = a[i] * b[j]; // < 2^32
      let plo = p & 0xFFFFu;
      let phi = p >> 16u;
      // 低 16 位：plo 落在 limb k=i+j，溢出进位**加到下一 limb**（k 递增后
      // 再取值——同 limb 复加会把 carry 双算进本位，这是初版的逐位缺陷）。
      var k = i + j;
      c[k] = c[k] + plo;
      var carry = c[k] >> 16u;
      c[k] = c[k] & 0xFFFFu;
      loop {
        if (carry == 0u || k >= 4u) { break; } // limb ≥4 = mod 2^64 丢弃域
        k = k + 1u;
        let s = c[k] + carry;
        c[k] = s & 0xFFFFu;
        carry = s >> 16u;
      }
      // 高 16 位（落在下一 limb）
      if (phi != 0u) {
        var k2 = i + j + 1u;
        c[k2] = c[k2] + phi;
        var carry2 = c[k2] >> 16u;
        c[k2] = c[k2] & 0xFFFFu;
        loop {
          if (carry2 == 0u || k2 >= 4u) { break; }
          k2 = k2 + 1u;
          let s2 = c[k2] + carry2;
          c[k2] = s2 & 0xFFFFu;
          carry2 = s2 >> 16u;
        }
      }
    }
  }
  return vec2<u32>(c[0] | (c[1] << 16u), c[2] | (c[3] << 16u));
}

/// 64 位逻辑右移（s < 64；WGSL 动态移位量合法）。
fn u64_shr(vl: u32, vh: u32, s: u32) -> vec2<u32> {
  if (s == 0u) { return vec2<u32>(vl, vh); }
  if (s < 32u) {
    return vec2<u32>((vl >> s) | (vh << (32u - s)), vh >> s);
  }
  return vec2<u32>(vh >> (s - 32u), 0u);
}

/// 64 位异或。
fn u64_xor(al: u32, ah: u32, bl: u32, bh: u32) -> vec2<u32> {
  return vec2<u32>(al ^ bl, ah ^ bh);
}

// ---- PCG RXS-M-XS 输出置换（与 pcg::permute 逐位同构）-------------------
//
// state = v·MULT + INC；shift = (state >> 59) + 5；xorshifted = state >> shift；
// word = (xorshifted ^ state)·MULT2；return (word >> 43) ^ word。
fn pcg_permute(vl: u32, vh: u32) -> vec2<u32> {
  let m = u64_mul(vl, vh, PCG_MULT_LO, PCG_MULT_HI);
  let st = u64_add(m.x, m.y, PCG_INC_LO, PCG_INC_HI);
  // (state >> 59)：64 位右移 59 = 高 32 位右移 27（剩 5 位）。
  let shamt = (st.y >> 27u) + 5u;
  let xs = u64_shr(st.x, st.y, shamt);
  let wv = u64_mul(xs.x ^ st.x, xs.y ^ st.y, PCG_MULT2_LO, PCG_MULT2_HI);
  let r = u64_shr(wv.x, wv.y, 43u);
  return vec2<u32>(r.x ^ wv.x, r.y ^ wv.y);
}

/// `pcg::hash1(seed, x)`：permute(seed ^ (x + SALT1))。
fn pcg_hash1(sl: u32, sh: u32, xl: u32, xh: u32) -> vec2<u32> {
  let xs = u64_add(xl, xh, SALT1_LO, SALT1_HI);
  let v = u64_xor(sl, sh, xs.x, xs.y);
  return pcg_permute(v.x, v.y);
}

/// `pcg::hash2_prefix(seed, x)`：permute(seed ^ (x + SALT1))（与 hash1 同式）。
fn pcg_hash2_prefix(sl: u32, sh: u32, xl: u32, xh: u32) -> vec2<u32> {
  return pcg_hash1(sl, sh, xl, xh);
}

/// `pcg::hash2_finish(prefix, y)`：permute(prefix ^ (y + SALT2))。
fn pcg_hash2_finish(pl: u32, ph: u32, yl: u32, yh: u32) -> vec2<u32> {
  let ys = u64_add(yl, yh, SALT2_LO, SALT2_HI);
  let v = u64_xor(pl, ph, ys.x, ys.y);
  return pcg_permute(v.x, v.y);
}

/// `pcg::hash4(seed, x, y, z, w)`：四段折叠（每段加对应槽位盐）。
fn pcg_hash4(sl: u32, sh: u32, xl: u32, xh: u32, yl: u32, yh: u32, zl: u32, zh: u32, wl: u32, wh: u32) -> vec2<u32> {
  let s1 = u64_add(xl, xh, SALT1_LO, SALT1_HI);
  let v1 = u64_xor(sl, sh, s1.x, s1.y);
  let p1 = pcg_permute(v1.x, v1.y);
  let s2 = u64_add(yl, yh, SALT2_LO, SALT2_HI);
  let v2 = u64_xor(p1.x, p1.y, s2.x, s2.y);
  let p2 = pcg_permute(v2.x, v2.y);
  let s3 = u64_add(zl, zh, SALT3_LO, SALT3_HI);
  let v3 = u64_xor(p2.x, p2.y, s3.x, s3.y);
  let p3 = pcg_permute(v3.x, v3.y);
  let s4 = u64_add(wl, wh, SALT4_LO, SALT4_HI);
  let v4 = u64_xor(p3.x, p3.y, s4.x, s4.y);
  return pcg_permute(v4.x, v4.y);
}

/// `pcg::hash3(seed, x, y, z)`：hash4 的前三段。
fn pcg_hash3(sl: u32, sh: u32, xl: u32, xh: u32, yl: u32, yh: u32, zl: u32, zh: u32) -> vec2<u32> {
  let s1 = u64_add(xl, xh, SALT1_LO, SALT1_HI);
  let v1 = u64_xor(sl, sh, s1.x, s1.y);
  let p1 = pcg_permute(v1.x, v1.y);
  let s2 = u64_add(yl, yh, SALT2_LO, SALT2_HI);
  let v2 = u64_xor(p1.x, p1.y, s2.x, s2.y);
  let p2 = pcg_permute(v2.x, v2.y);
  let s3 = u64_add(zl, zh, SALT3_LO, SALT3_HI);
  let v3 = u64_xor(p2.x, p2.y, s3.x, s3.y);
  return pcg_permute(v3.x, v3.y);
}

/// `pcg::unit_f32`：64 位哈希高 24 位 → [0,1)（`(bits >> 40) as f32 * 2^-24`）。
fn pcg_unit_f32(vl: u32, vh: u32) -> f32 {
  // bits >> 40 = (hi >> 8)（高 32 位取低 24 位）。
  let top = vh >> 8u;
  return f32(top) * (1.0 / 16777216.0);
}
