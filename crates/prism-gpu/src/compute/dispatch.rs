//! Dispatch 形状（设备无关）：workgroup 尺寸、组数展开、CPU 侧等价执行。
//!
//! # 语义权威
//!
//! *"GPU 用途全景"段* 的 ①compute dispatch 形态；D-15（自研抽象层）、
//! D-21（从数学推导开始设计，先定语义再落 backend）。
//!
//! # 为什么要设备无关的 dispatch 形状
//!
//! Vulkan / WebGPU / Metal 三家的 dispatch 语义相同（`(gx,gy,gz)` 个组，
//! 每组 `local_size` 个调用，`global_invocation_id = group_id * local_size +
//! local_id`），差别只在 API 名字。把这条**纯数学**抽出来之后：
//!
//! * CPU 可逐字跑（[`DispatchShape::dispatch_cpu`] 生成与 GPU 完全相同的
//!   invocation 序）；kernel 语义的正确性因此能在无 GPU 环境真测；
//! * 真机 backend 只需把 [`DispatchShape::groups`] 三个数喂给 `vkCmdDispatch`
//!   （见 `crate::vulkan::compute`），不存在第二套形状口径。
//!
//! # 尾组浪费（tail waste）
//!
//! extent 不整除 workgroup 时，最后一组里有调用落在 extent 外。GPU 上这些
//! 调用**照样执行**（必须由 kernel 自己用边界判断丢弃），因此
//! [`DispatchShape::coverage`] 如实报出「有效 / 浪费 / 占用率」，供效率账
//! 与 workgroup 尺寸选型使用——不把浪费藏起来。

/// 工作组尺寸（对应 WGSL `@workgroup_size(x, y, z)`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// x 维调用数。
    pub x: u32,
    /// y 维调用数。
    pub y: u32,
    /// z 维调用数。
    pub z: u32,
}

impl WorkgroupSize {
    /// 构造工作组尺寸；零值非法（由 [`WorkgroupSize::is_valid`] 判定）。
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        WorkgroupSize { x, y, z }
    }

    /// 单组调用总数（x·y·z；溢出按饱和处理，不 panic）。
    pub fn invocations(&self) -> u32 {
        self.x.saturating_mul(self.y).saturating_mul(self.z)
    }

    /// 三维皆非零（Vulkan 要求 local size 每维 ≥ 1）。
    pub fn is_valid(&self) -> bool {
        self.x > 0 && self.y > 0 && self.z > 0
    }

    /// 三元组形式（喂 `vkCmdDispatch` 之外的诊断打印用）。
    pub fn as_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

impl Default for WorkgroupSize {
    /// 默认 64×1×1（RT kernel 的常用一维形状：一个 subgroup 两条 warp）。
    fn default() -> Self {
        WorkgroupSize::new(64, 1, 1)
    }
}

/// 一次 dispatch 的完整形状：问题域 extent + 工作组尺寸。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchShape {
    /// 问题域三维规模（对 RT 主射线发射：`[w, h, 1]`）。
    pub extent: [u32; 3],
    /// 工作组尺寸。
    pub workgroup: WorkgroupSize,
}

impl DispatchShape {
    /// 以 extent + workgroup 构造（不做合法性校验，见 [`DispatchShape::is_valid`]）。
    pub const fn new(extent: [u32; 3], workgroup: WorkgroupSize) -> Self {
        DispatchShape { extent, workgroup }
    }

    /// 一维 extent 的便捷构造（`[n, 1, 1]`）。
    pub const fn linear(n: u32, workgroup: WorkgroupSize) -> Self {
        DispatchShape::new([n, 1, 1], workgroup)
    }

    /// extent 每维非零且 workgroup 合法。
    pub fn is_valid(&self) -> bool {
        self.extent.iter().all(|&e| e > 0) && self.workgroup.is_valid()
    }

    /// 工作组个数（逐维向上取整：`ceil(extent / workgroup)`）。
    ///
    /// 这正是 `vkCmdDispatch(gx, gy, gz)` 的三个实参。
    pub fn groups(&self) -> [u32; 3] {
        let wg = self.workgroup;
        let div_ceil = |e: u32, w: u32| -> u32 {
            if w == 0 {
                0
            } else {
                e / w + if e % w == 0 { 0 } else { 1 }
            }
        };
        [
            div_ceil(self.extent[0], wg.x),
            div_ceil(self.extent[1], wg.y),
            div_ceil(self.extent[2], wg.z),
        ]
    }

    /// 组总数（64 位，避免大 extent 溢出）。
    pub fn total_groups(&self) -> u64 {
        let g = self.groups();
        g[0] as u64 * g[1] as u64 * g[2] as u64
    }

    /// 调用总数（含落在 extent 外的「尾组浪费」调用）。
    pub fn total_invocations(&self) -> u64 {
        self.total_groups() * self.workgroup.invocations() as u64
    }

    /// 有效调用数（= extent 体积）。
    pub fn useful_invocations(&self) -> u64 {
        self.extent[0] as u64 * self.extent[1] as u64 * self.extent[2] as u64
    }

    /// 覆盖率诊断：有效 / 浪费 / 占用率（尾组浪费如实报出，不隐藏）。
    pub fn coverage(&self) -> Coverage {
        let total = self.total_invocations();
        let useful = self.useful_invocations().min(total);
        let wasted = total - useful;
        let occupancy = if total == 0 {
            0.0
        } else {
            useful as f32 / total as f32
        };
        Coverage {
            useful_invocations: useful,
            wasted_invocations: wasted,
            occupancy,
        }
    }

    /// 按 GPU 的 invocation 序逐字执行 `f`（**CPU 等价执行器**）。
    ///
    /// 遍历序与 GPU 的 `global_invocation_id` 递增序一致：x 最快、z 最慢。
    /// 这是「CPU 模拟 GPU 数组序」的权威实现——parity 门槛（`prism-rt`
    /// 的 wavefront harness）与 kernel 语义测试都建立在它之上。
    ///
    /// 落在 extent 外的调用**照样回调**（GPU 上确实会执行），由 `f` 自己
    /// 按 extent 判边界丢弃——与真机 kernel 的 `if (gid >= n) return;` 同构。
    pub fn dispatch_cpu<F: FnMut([u32; 3])>(&self, mut f: F) {
        let g = self.groups();
        let wg = self.workgroup;
        for gz in 0..g[2] {
            for gy in 0..g[1] {
                for gx in 0..g[0] {
                    for lz in 0..wg.z {
                        for ly in 0..wg.y {
                            for lx in 0..wg.x {
                                f([gx * wg.x + lx, gy * wg.y + ly, gz * wg.z + lz]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// 仅遍历 extent 内的有效调用（测试/参考实现的便捷入口）。
    ///
    /// 序与 [`DispatchShape::dispatch_cpu`] 相同（只是跳过越界者）。
    pub fn dispatch_cpu_clamped<F: FnMut([u32; 3])>(&self, mut f: F) {
        let extent = self.extent;
        self.dispatch_cpu(|id| {
            if id[0] < extent[0] && id[1] < extent[1] && id[2] < extent[2] {
                f(id);
            }
        });
    }
}

/// 尾组浪费诊断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coverage {
    /// 有效调用数。
    pub useful_invocations: u64,
    /// 落在 extent 外、被 kernel 边界判断丢弃的调用数。
    pub wasted_invocations: u64,
    /// 占用率（有效 / 总调用，0..=1）。
    pub occupancy: f32,
}

/// 三维坐标 → 行主序线性下标。
pub fn index_3d_to_linear(p: [u32; 3], extent: [u32; 3]) -> u32 {
    (p[2] * extent[1] + p[1]) * extent[0] + p[0]
}

/// 行主序线性下标 → 三维坐标（[`index_3d_to_linear`] 的逆）。
pub fn linear_to_index_3d(i: u32, extent: [u32; 3]) -> [u32; 3] {
    let plane = extent[0] * extent[1];
    let z = if plane == 0 { 0 } else { i / plane };
    let rem = if plane == 0 { 0 } else { i % plane };
    let y = if extent[0] == 0 { 0 } else { rem / extent[0] };
    let x = if extent[0] == 0 { 0 } else { rem % extent[0] };
    [x, y, z]
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_invocations_and_validity() {
        let wg = WorkgroupSize::new(8, 4, 2);
        assert_eq!(wg.invocations(), 64);
        assert!(wg.is_valid());
        assert!(!WorkgroupSize::new(0, 1, 1).is_valid(), "零维非法");
        assert!(!WorkgroupSize::new(4, 0, 4).is_valid());
        assert_eq!(WorkgroupSize::default().invocations(), 64);
    }

    #[test]
    fn groups_round_up_per_dimension() {
        // 1920×1080 / 64×1 → 30×1080 组（5000-2）
        let s = DispatchShape::new([1920, 1080, 1], WorkgroupSize::new(64, 1, 1));
        assert_eq!(s.groups(), [30, 1080, 1]);
        assert_eq!(s.total_groups(), 30 * 1080);
        assert_eq!(s.useful_invocations(), 1920 * 1080);
        // 整除情况：无尾组浪费。
        let exact = DispatchShape::new([128, 64, 1], WorkgroupSize::new(8, 8, 1));
        assert_eq!(exact.groups(), [16, 8, 1]);
        assert_eq!(exact.coverage().wasted_invocations, 0);
        assert!((exact.coverage().occupancy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tail_groups_waste_is_reported_not_hidden() {
        // extent 100 / wg 8 → 13 组 = 104 调用，浪费 4。
        let s = DispatchShape::linear(100, WorkgroupSize::new(8, 1, 1));
        assert_eq!(s.groups(), [13, 1, 1]);
        assert_eq!(s.total_invocations(), 104);
        let cov = s.coverage();
        assert_eq!(cov.useful_invocations, 100);
        assert_eq!(cov.wasted_invocations, 4, "尾组浪费必须如实报出");
        assert!(cov.occupancy < 1.0 && cov.occupancy > 0.96);
    }

    #[test]
    fn dispatch_cpu_visits_every_invocation_exactly_once_in_gpu_order() {
        let s = DispatchShape::new([10, 3, 1], WorkgroupSize::new(4, 2, 1));
        let groups = s.groups();
        assert_eq!(groups, [3, 2, 1]);
        let mut seen: Vec<[u32; 3]> = Vec::new();
        s.dispatch_cpu(|id| seen.push(id));
        // 调用总数 = 组数 × 组内调用数（含越界调用）。
        assert_eq!(seen.len() as u64, s.total_invocations());
        // 每个 invocation id 唯一（GPU 语义：一个 id 一次）。
        let mut uniq = seen.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), seen.len(), "invocation id 必须唯一");
        // 序：x 最快。首 8 个是 (0,0,0)..(3,0,0),(0,1,0)..(3,1,0)。
        assert_eq!(
            &seen[..8],
            &[
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0]
            ]
        );
        // x 维第二个组：gx=1 → 4..7。
        assert_eq!(seen[8], [4, 0, 0]);
        // 越界调用确实被回调（GPU 上它们照样执行）。
        assert!(seen.iter().any(|id| id[0] >= 10), "越界调用应出现在回调里");
    }

    #[test]
    fn dispatch_cpu_clamped_covers_extent_exactly_once() {
        for extent in [[7u32, 5u32, 1u32], [1, 1, 1], [33, 2, 1], [64, 64, 1]] {
            let s = DispatchShape::new(extent, WorkgroupSize::new(8, 4, 1));
            let mut count = 0u64;
            s.dispatch_cpu_clamped(|_| count += 1);
            assert_eq!(
                count,
                extent[0] as u64 * extent[1] as u64 * extent[2] as u64,
                "extent {extent:?}：有效调用数应等于体积"
            );
        }
    }

    #[test]
    fn linear_index_roundtrip() {
        let extent = [13u32, 7u32, 3u32];
        for z in 0..extent[2] {
            for y in 0..extent[1] {
                for x in 0..extent[0] {
                    let i = index_3d_to_linear([x, y, z], extent);
                    assert_eq!(linear_to_index_3d(i, extent), [x, y, z]);
                }
            }
        }
    }

    #[test]
    fn dispatch_is_deterministic() {
        let s = DispatchShape::new([9, 4, 1], WorkgroupSize::new(4, 4, 1));
        let mut a = Vec::new();
        let mut b = Vec::new();
        s.dispatch_cpu(|id| a.push(id));
        s.dispatch_cpu(|id| b.push(id));
        assert_eq!(a, b, "同形状两遍 dispatch 序应完全一致");
    }
}
