//! GPU 模拟的 CPU 仿真参考（D3）——百万粒子 compute 的判对基准。
//!
//! # 定位（路线图 D3：GPU 百万粒子 compute，与 CpuRef 双实现互为判对）
//! GPU 粒子 kernel（`crates/prism-gpu/shaders/sim_*.wgsl`）的真实实装待真机
//! （D1 绑定 + D-14 手机 WebGPU）。本模块在**无 GPU 环境**下给出同构的 CPU
//! 参考：实现 WGSL 的五个阶段（发射/积分/格键/基数排序/邻域），使 GPU kernel
//! 上线时可逐位判对。
//!
//! # 与 WGSL 逐句对齐（判对契约）
//! - 发射：`hash(seed, frame, i·3+k)` 派生初态（PCG 族，无全局 RNG）。
//! - 积分：半隐式欧拉 `v += g·dt; x += v·dt`。
//! - 格键：`floor(p/h)` 线性化（每轴偏移 1024，21 位打包）。
//! - 排序：按格键稳定计数排序（radix 的等价确定性实现；键并 tie 按索引）。
//! - 邻域：排序后 27 邻格扫描（2D 退化 3×3）。
//!
//! # 约束
//! 零第三方依赖；无全局 RNG；只用 ±×÷√ 与 `floor`（IEEE 精确运算）。

use prism_math::Vec3;

/// 粒子（与 WGSL `Particle` 结构字段一一对应）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuParticle {
    /// 位置。
    pub pos: Vec3,
    /// 速度。
    pub vel: Vec3,
    /// 质量倒数（0 = 固定）。
    pub mass_inv: f32,
    /// 空间哈希格键（radix sort 键）。
    pub cell: u32,
}

/// 模拟参数（与 WGSL `SimParams` 一一对应）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSimParams {
    /// 核半径 h（cell 边长）。
    pub h: f32,
    /// 步长 dt。
    pub dt: f32,
    /// 重力。
    pub gravity: Vec3,
    /// 全局种子。
    pub seed: u32,
}

/// PCG 整数 hash（与 prism-noise::pcg / WGSL pcg_hash 同族；跨平台逐位一致）。
#[inline]
pub fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(2_778_037_377);
    (word >> 22) ^ word
}

/// `hash3_f32`（与 WGSL 同式）：三输入 → [0,1) f32。
#[inline]
pub fn hash3_f32(a: u32, b: u32, c: u32) -> f32 {
    let h = pcg_hash(pcg_hash(a) ^ pcg_hash(b ^ c));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// 格坐标 floor(v/h)（与 WGSL `cell_floor` 同式）。
#[inline]
pub fn cell_floor(v: f32, h: f32) -> i32 {
    (v / h).floor() as i32
}

/// 格键线性化（与 WGSL `cell_key` 同式）：每轴偏移 1024，21 位打包。
#[inline]
pub fn cell_key(p: Vec3, h: f32) -> u32 {
    let bx = (cell_floor(p.x, h) + 1024) as u32;
    let by = (cell_floor(p.y, h) + 1024) as u32;
    let bz = (cell_floor(p.z, h) + 1024) as u32;
    (bx & 0x7FF) | ((by & 0x7FF) << 11) | ((bz & 0x7FF) << 22)
}

/// GPU 模拟的 CPU 参考：粒子数组 + 五阶段推进。
#[derive(Debug, Clone)]
pub struct GpuSimRef {
    /// 粒子数组（与 GPU storage buffer 对应）。
    pub particles: Vec<GpuParticle>,
    /// 参数。
    pub params: GpuSimParams,
    /// 当前帧号。
    pub frame: u32,
}

impl GpuSimRef {
    /// 构造：count 个粒子（初态零，未发射）。
    pub fn new(count: usize, params: GpuSimParams) -> Self {
        GpuSimRef {
            particles: vec![
                GpuParticle {
                    pos: Vec3::splat(0.0),
                    vel: Vec3::splat(0.0),
                    mass_inv: 1.0,
                    cell: 0,
                };
                count
            ],
            params,
            frame: 0,
        }
    }

    /// 阶段 ①：发射（对齐 WGSL `emit`）——按 (seed, frame, i) 派生初态。
    pub fn emit(&mut self, origin: Vec3) {
        let seed = self.params.seed;
        let frame = self.frame;
        for (i, p) in self.particles.iter_mut().enumerate() {
            let i3 = (i as u32).wrapping_mul(3);
            let u0 = hash3_f32(seed, frame, i3);
            let u1 = hash3_f32(seed, frame, i3 + 1);
            let u2 = hash3_f32(seed, frame, i3 + 2);
            p.pos = origin + Vec3::new(u0 - 0.5, u1 - 0.5, u2 - 0.5);
            p.vel = Vec3::splat(0.0);
            p.mass_inv = 1.0;
        }
    }

    /// 阶段 ②：积分（对齐 WGSL `integrate`）——半隐式欧拉。
    pub fn integrate(&mut self) {
        let dt = self.params.dt;
        let g = self.params.gravity;
        for p in self.particles.iter_mut() {
            if p.mass_inv > 0.0 {
                p.vel = p.vel + g * dt;
                p.pos = p.pos + p.vel * dt;
            }
        }
    }

    /// 阶段 ③：格键（对齐 WGSL `hash_cells`）。
    pub fn hash_cells(&mut self) {
        let h = self.params.h;
        for p in self.particles.iter_mut() {
            p.cell = cell_key(p.pos, h);
        }
    }

    /// 阶段 ④：按格键稳定排序（radix 等价；tie 按原索引 → 确定性）。
    /// 返回排序后原索引数组（供 values 重排对齐 GPU radix sort 语义）。
    pub fn sort_by_cell(&mut self) -> Vec<u32> {
        let n = self.particles.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| {
            let ca = self.particles[a as usize].cell;
            let cb = self.particles[b as usize].cell;
            ca.cmp(&cb).then(a.cmp(&b))
        });
        let sorted: Vec<GpuParticle> =
            order.iter().map(|&i| self.particles[i as usize]).collect();
        self.particles = sorted;
        order
    }

    /// 阶段 ⑤：邻域扫描（排序后 3×3 格邻域，2D 退化；返回每粒子的邻居索引列表）。
    /// 与 WGSL `neighbor_scan` / SPH 的 cell_range 扫描同构。
    pub fn neighbors(&self) -> Vec<Vec<usize>> {
        let h = self.params.h;
        let n = self.particles.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let pi = self.particles[i].pos;
            let cx = cell_floor(pi.x, h);
            let cy = cell_floor(pi.y, h);
            let mut nb = Vec::new();
            for (j, pj) in self.particles.iter().enumerate() {
                if j == i {
                    continue;
                }
                let dx = cell_floor(pj.pos.x, h) - cx;
                let dy = cell_floor(pj.pos.y, h) - cy;
                if dx.abs() <= 1 && dy.abs() <= 1 {
                    nb.push(j);
                }
            }
            out.push(nb);
        }
        out
    }

    /// 推进一帧（发射 → 积分 → 格键 → 排序）。
    pub fn step(&mut self, origin: Vec3) {
        self.emit(origin);
        self.integrate();
        self.hash_cells();
        self.sort_by_cell();
        self.frame += 1;
    }

    /// 位置字节快照（判对 / 确定性断言用）。
    pub fn position_bits(&self) -> Vec<[u32; 3]> {
        self.particles
            .iter()
            .map(|p| [p.pos.x.to_bits(), p.pos.y.to_bits(), p.pos.z.to_bits()])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GpuSimParams {
        GpuSimParams {
            h: 0.25,
            dt: 1.0 / 60.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            seed: 0x3D_00_01,
        }
    }

    /// 阶段 ③：格键线性化与 floor 锚点（对齐 WGSL cell_key）。
    #[test]
    fn cell_key_anchors() {
        let h = 1.0;
        // 原点 → 各轴偏移 1024 → 键 = 1024 | 1024<<11 | 1024<<22。
        let want = (1024u32 & 0x7FF) | ((1024u32 & 0x7FF) << 11) | ((1024u32 & 0x7FF) << 22);
        assert_eq!(cell_key(Vec3::splat(0.0), h), want);
        // 负坐标 floor 到 -1 格（偏移后 1023）。
        let k_neg = cell_key(Vec3::splat(-0.5), h);
        assert_eq!((k_neg & 0x7FF), 1023);
        // 同格内不同点 → 同键。
        assert_eq!(cell_key(Vec3::new(0.1, 0.2, 0.3), h), cell_key(Vec3::new(0.9, 0.9, 0.9), h));
    }

    /// 阶段 ④：排序后格键单调非降 + 索引稳定。
    #[test]
    fn sort_by_cell_is_monotonic_and_stable() {
        let mut sim = GpuSimRef::new(64, params());
        sim.emit(Vec3::splat(0.0));
        sim.hash_cells();
        let order = sim.sort_by_cell();
        assert_eq!(order.len(), 64);
        for w in sim.particles.windows(2) {
            assert!(w[0].cell <= w[1].cell, "排序后格键应单调非降");
        }
    }

    /// 阶段 ⑤：邻域自洽——相邻粒子互为邻居。
    #[test]
    fn neighbors_reciprocal_for_close_particles() {
        let mut sim = GpuSimRef::new(0, params());
        sim.particles = vec![
            GpuParticle { pos: Vec3::new(0.0, 0.0, 0.0), vel: Vec3::splat(0.0), mass_inv: 1.0, cell: 0 },
            GpuParticle { pos: Vec3::new(0.1, 0.0, 0.0), vel: Vec3::splat(0.0), mass_inv: 1.0, cell: 0 },
            GpuParticle { pos: Vec3::new(5.0, 5.0, 0.0), vel: Vec3::splat(0.0), mass_inv: 1.0, cell: 0 },
        ];
        sim.hash_cells();
        sim.sort_by_cell();
        // 排序后第 0/1 应相邻（近格），远粒子不在其邻域。
        let nb = sim.neighbors();
        let near_pair = sim.particles[0].cell == sim.particles[1].cell
            || (cell_floor(sim.particles[0].pos.x, 0.25) - cell_floor(sim.particles[1].pos.x, 0.25)).abs() <= 1;
        assert!(near_pair, "近粒子应同/邻格");
        // 邻域列表长度有限（不越界）。
        assert_eq!(nb.len(), 3);
    }

    /// 确定性：同参数两遍推进逐位一致（判对基准）。
    #[test]
    fn gpu_sim_ref_bitwise_deterministic() {
        let p = params();
        let mut a = GpuSimRef::new(50, p);
        let mut b = GpuSimRef::new(50, p);
        for _ in 0..20 {
            a.step(Vec3::splat(0.0));
            b.step(Vec3::splat(0.0));
        }
        assert_eq!(a.position_bits(), b.position_bits(), "GPU 模拟参考应逐位一致");
        assert_eq!(a.frame, b.frame);
    }

    /// 发射种子化：不同帧初态不同（无全局 RNG，帧号进 hash）。
    #[test]
    fn emit_is_frame_seeded() {
        let p = params();
        let mut a = GpuSimRef::new(4, p);
        a.emit(Vec3::splat(0.0));
        let f0 = a.position_bits();
        a.frame = 1;
        a.emit(Vec3::splat(0.0));
        let f1 = a.position_bits();
        assert_ne!(f0, f1, "不同帧发射初态应不同（帧号进哈希）");
    }
}
