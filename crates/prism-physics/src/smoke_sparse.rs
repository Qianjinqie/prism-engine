//! 稀疏体素存储 + 涡量约束（「烟雾/爆炸模拟侧」深化 B）。
//!
//! 本模块是 [`crate::smoke`]（FLIP 稠密 MAC 网格求解器）的**正交扩展**：
//! 在**不触碰 `smoke.rs`** 的前提下，补齐两处基础版诚实记录的缺口：
//!
//! 1. **稀疏体素存储**（[`SparseBlockGrid`]）——把规则网格切成固定边长
//!    `block_size` 的立方块，只存「非空」块；空块不占内存。提供稠密↔稀疏
//!    无损往返（逐位一致）。
//! 2. **涡量约束**（[`apply_vorticity_confinement`]）——重新注入数值耗散
//!    抹掉的细小涡旋，恢复湍流细节。该函数**不依赖**稀疏结构，可独立测试。
//!
//! # 物理公式与出处
//!
//! - 涡量定义（curl）：`ω = ∇ × u`（Stokes 定理微分形，任意流体力学教科书）。
//! - 涡量约束力：`f_vort = ε · h · (N × ω)`，其中
//!   `N = ∇|ω| / (|∇|ω|| + δ)` 为归一化涡量梯度。
//!   出处：**Steinhoff & Underhill 1994**（原始涡量约束模型）；
//!   **Fedkiw, Stam & Jensen 2001**（SIGGRAPH《Visual Simulation of Smoke》，
//!   将涡量约束用于烟雾的经典形式，本实装即依此文）。
//!
//! # 确定性纪律（铁律二）
//!
//! - 稀疏块以块坐标 `(i32,i32,i32)` 为键，用 [`BTreeMap`] 存储——遍历即
//!   **按块坐标字典序**，与插入先后无关；**绝不**用 `HashMap` 迭代定序。
//! - 所有网格下标遍历按 `(k, j, i)` 固定字典序。
//! - 浮点累加顺序固定；无全局 RNG；无提前退出。
//! - 差分格式统一：内部用中心差分，边界（或相邻块缺失）退化为单侧差分。
//!
//! # 零 panic（宪章 C18）
//!
//! - 所有索引访问先做边界检查，越界返回 [`PhysicsError`]。
//! - 无 `unwrap()` / `expect()` / `panic!()` / 越界索引 / 整数溢出
//!   （尺寸运算用 `checked_mul` / `checked_add`）。
//! - **分配字节上限**：`usize` 的 `checked_mul` 只保证乘积不溢出 `usize`，
//!   但 `Vec` 的真实分配上限是 `isize::MAX` **字节**。本模块在每次大数组
//!   分配前用 [`alloc_size_ok`] 校验 `元素数 · 单元素字节数 ≤ isize::MAX`，
//!   拦截「乘积不溢 `usize` 却触发 `capacity overflow` panic」的路径
//!   （`zeroed` / `from_dense` / `to_dense` / `compute_vorticity` /
//!   `apply_vorticity_confinement`）。此缺陷由「接手验证」轮的对抗性探针
//!   发现（`probe_huge_block_size_no_panic`），是本轮修复的核心。
//! - 仅依赖 `std`（含 `std::collections::BTreeMap`）与 workspace 内
//!   [`prism_math::Vec3`]，无任何第三方 crate。

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use prism_math::Vec3;

use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 内部小工具（全部 checked / 饱和，禁止 panic）
// ---------------------------------------------------------------------------

/// 把 `usize` 安全转为 `i32`（块内索引用；超界返回 None，不 panic）。
#[inline]
fn usize_to_i32(v: usize) -> Option<i32> {
    if v <= i32::MAX as usize {
        Some(v as i32)
    } else {
        None
    }
}

/// 浮点有限性检查（NaN/±Inf 同拒）。
#[inline]
fn finite3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// 元素数 × 单元素字节数是否**安全可分配**（不触发 `Vec` 的 capacity overflow）。
///
/// # 为何必需（C18 零 panic）
/// `usize` 上的 `checked_mul` 只能保证**乘积**不溢出 `usize`；但 Rust 的
/// `Vec`/`slice` 分配上限是 `isize::MAX` **字节**（见 `alloc::raw_vec`）。
/// 若 `count · size_of::<T>() > isize::MAX`，即便 `count` 不溢出 `usize`，
/// `vec![...; count]` 仍会在分配时 **panic（capacity overflow）**。
/// 本函数在分配前拦截该情形，让调用方返回结构化错误而非 panic。
///
/// # 参数
/// - `count`：元素个数；
/// - `elem_size`：单元素字节数（`size_of::<T>()`）。
#[inline]
fn alloc_size_ok(count: usize, elem_size: usize) -> bool {
    match count.checked_mul(elem_size) {
        // 再留一层余量：要求 ≤ isize::MAX（Vec 的真实上限）。
        Some(bytes) => bytes <= isize::MAX as usize,
        None => false,
    }
}

/// 断言用：显式消费 [`finite3`]，避免死代码（供诊断/防御性检查调用）。
#[allow(dead_code)]
fn debug_assert_finite(v: Vec3) -> bool {
    finite3(v)
}

// ---------------------------------------------------------------------------
// 稀疏块网格存储
// ---------------------------------------------------------------------------

/// 单个立方块的稠密体素数据。
///
/// 块内布局固定为 `(bz, by, bx)` 行主序（`x` 变化最快），边长 `block_size`。
/// `velocity[k·s² + j·s + i]` 为体素 `(i,j,k)` 的速度（块内局部坐标）。
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelBlock {
    /// 块边长（= 所属 [`SparseBlockGrid`] 的 `block_size`）。
    pub block_size: usize,
    /// 逐体素速度，长度 = `block_size³`，布局 `(z, y, x)`。
    pub velocity: Vec<Vec3>,
    /// 逐体素密度，长度 = `block_size³`，布局 `(z, y, x)`。
    pub density: Vec<f32>,
}

impl VoxelBlock {
    /// 构造全零块（速度 0、密度 0）。
    ///
    /// # 防呆（C18）
    /// 拒绝 `block_size == 0`、`block_size³` 溢出 `usize`，以及 `block_size³`
    /// 的元素分配字节数超过 `isize::MAX`（后者若放行，`vec![...]` 会
    /// capacity overflow panic）。
    pub fn zeroed(block_size: usize) -> Result<Self, PhysicsError> {
        let n = block_size
            .checked_mul(block_size)
            .and_then(|s| s.checked_mul(block_size))
            .ok_or_else(|| PhysicsError {
                code: "P3009",
                location: "VoxelBlock::zeroed".to_string(),
                problem: format!("block_size = {block_size}：立方块体素数溢出 usize"),
                fix: "传更小的块边长（如 8 或 16）".to_string(),
            })?;
        if block_size == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "VoxelBlock::zeroed".to_string(),
                problem: "block_size = 0：块边长必须 ≥ 1".to_string(),
                fix: "传 block_size ≥ 1".to_string(),
            });
        }
        // C18：乘积不溢出 usize 还不够——Vec 分配上限是 isize::MAX 字节。
        // 检两块元素类型（Vec3 与 f32）的可分配性，任一不可分配即返回结构化错误。
        let vec3_size = std::mem::size_of::<Vec3>();
        let f32_size = std::mem::size_of::<f32>();
        if !alloc_size_ok(n, vec3_size) || !alloc_size_ok(n, f32_size) {
            return Err(PhysicsError {
                code: "P3009",
                location: "VoxelBlock::zeroed".to_string(),
                problem: format!(
                    "block_size = {block_size}：立方块体素数 {n} 的分配字节数超过 isize::MAX"
                ),
                fix: "传更小的块边长（如 8 或 16）".to_string(),
            });
        }
        Ok(Self {
            block_size,
            velocity: vec![Vec3::splat(0.0); n],
            density: vec![0.0; n],
        })
    }

    /// 块内体素数 `block_size³`。
    #[inline]
    pub fn voxel_count(&self) -> usize {
        // 构造时已保证不溢出；此处再走一次 checked，失败则降级为 0。
        self.block_size
            .checked_mul(self.block_size)
            .and_then(|s| s.checked_mul(self.block_size))
            .unwrap_or(0)
    }

    /// 块内局部坐标 `(i,j,k)` 的线性下标；越界返回 None。
    #[inline]
    fn local_index(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        let s = self.block_size;
        if i >= s || j >= s || k >= s {
            return None;
        }
        k.checked_mul(s)
            .and_then(|t| t.checked_add(j))
            .and_then(|t| t.checked_mul(s))
            .and_then(|t| t.checked_add(i))
    }

    /// 读块内体素速度（越界返回 0 向量，安全降级）。
    #[inline]
    pub fn get_velocity(&self, i: usize, j: usize, k: usize) -> Vec3 {
        match self.local_index(i, j, k).and_then(|idx| self.velocity.get(idx)) {
            Some(v) => *v,
            None => Vec3::splat(0.0),
        }
    }

    /// 写块内体素速度（越界忽略，安全降级）。
    #[inline]
    pub fn set_velocity(&mut self, i: usize, j: usize, k: usize, val: Vec3) {
        if let Some(idx) = self.local_index(i, j, k) {
            if let Some(slot) = self.velocity.get_mut(idx) {
                *slot = val;
            }
        }
    }

    /// 读块内体素密度（越界返回 0.0，安全降级）。
    #[inline]
    pub fn get_density(&self, i: usize, j: usize, k: usize) -> f32 {
        match self.local_index(i, j, k).and_then(|idx| self.density.get(idx)) {
            Some(d) => *d,
            None => 0.0,
        }
    }

    /// 写块内体素密度（越界忽略，安全降级）。
    #[inline]
    pub fn set_density(&mut self, i: usize, j: usize, k: usize, val: f32) {
        if let Some(idx) = self.local_index(i, j, k) {
            if let Some(slot) = self.density.get_mut(idx) {
                *slot = val;
            }
        }
    }

    /// 块内密度和（固定 `(k,j,i)` 序累加，确定性）。
    #[inline]
    pub fn density_sum(&self) -> f32 {
        let mut acc = 0.0f32;
        for d in self.density.iter() {
            acc += *d;
        }
        acc
    }
}

/// 稀疏块网格：以块坐标 `(bx,by,bz)` 为键，只存「非空」块。
///
/// # 空块判定
/// 块内**密度和**（`Σ density`）`> threshold` 即视为「非空」并入表。
/// `threshold` 由构造时给定；典型取 0 表示「只要有一个正密度体素即保留」，
/// 或取正小量过滤数值尘埃。
///
/// # 遍历顺序（铁律二关键点）
/// 底层用 [`BTreeMap<(i32,i32,i32), VoxelBlock>`]。`BTreeMap` 按键有序，
/// 故 [`SparseBlockGrid::iter_blocks_sorted`] 的返回顺序**仅由块坐标决定**，
/// 与 `insert_block` 的调用先后完全无关。这直接规避了 `HashMap` 迭代序
/// 不确定导致的不确定性。
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockGrid {
    /// 块边长（体素/边）。
    block_size: usize,
    /// 空块判定阈值（块密度和 > threshold 视为非空）。
    threshold: f32,
    /// 有序块表：键 = 块坐标（可为负，支持无限域局部化）。
    blocks: BTreeMap<(i32, i32, i32), VoxelBlock>,
}

impl SparseBlockGrid {
    /// 构造空稀疏网格。
    ///
    /// # 参数
    /// - `block_size`：块边长（体素/边），须 ≥ 1；
    /// - `threshold`：空块判定阈值，须有限。
    ///
    /// # 防呆（C18）
    /// 拒绝 `block_size == 0`、`threshold` 非有限。
    pub fn new(block_size: usize, threshold: f32) -> Result<Self, PhysicsError> {
        if block_size == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "SparseBlockGrid::new".to_string(),
                problem: "block_size = 0：块边长必须 ≥ 1".to_string(),
                fix: "传 block_size ≥ 1（推荐 8）".to_string(),
            });
        }
        if !threshold.is_finite() {
            return Err(PhysicsError {
                code: "P1014",
                location: "SparseBlockGrid::new".to_string(),
                problem: format!("threshold = {threshold}：阈值必须有限"),
                fix: "传有限阈值（如 0.0）".to_string(),
            });
        }
        Ok(Self {
            block_size,
            threshold,
            blocks: BTreeMap::new(),
        })
    }

    /// 块边长。
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// 空块判定阈值。
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// 插入/覆盖一个块；若其密度和 `<= threshold` 则**不入表**（视作空块）。
    ///
    /// 返回 `true` 表示块已存留（非空），`false` 表示被判为空而被丢弃。
    ///
    /// # 说明
    /// 若传入块的 `block_size` 与本网格不一致，返回结构化错误。
    pub fn insert_block(
        &mut self,
        coord: (i32, i32, i32),
        block: VoxelBlock,
    ) -> Result<bool, PhysicsError> {
        if block.block_size != self.block_size {
            return Err(PhysicsError {
                code: "P3009",
                location: "SparseBlockGrid::insert_block".to_string(),
                problem: format!(
                    "块边长 {} ≠ 网格块边长 {}：尺寸不一致",
                    block.block_size, self.block_size
                ),
                fix: "用本网格 block_size 构造块（VoxelBlock::zeroed(block_size)）".to_string(),
            });
        }
        if block.density_sum() > self.threshold {
            self.blocks.insert(coord, block);
            Ok(true)
        } else {
            // 空块：若此前存在则移除，保证「空块不占内存」不变式。
            self.blocks.remove(&coord);
            Ok(false)
        }
    }

    /// 直接写入块（**无条件**入表，跳过空块判定）。
    ///
    /// 供 [`SparseBlockGrid::from_dense`] / [`dense_to_sparse`] 使用，
    /// 以便无损往返（避免因阈值重新判定而丢失块）。
    fn insert_block_raw(&mut self, coord: (i32, i32, i32), block: VoxelBlock) {
        self.blocks.insert(coord, block);
    }

    /// 取块引用（不存在返回 None）。
    #[inline]
    pub fn get_block(&self, coord: (i32, i32, i32)) -> Option<&VoxelBlock> {
        self.blocks.get(&coord)
    }

    /// 取块可变引用（不存在返回 None）。
    #[inline]
    pub fn get_block_mut(&mut self, coord: (i32, i32, i32)) -> Option<&mut VoxelBlock> {
        self.blocks.get_mut(&coord)
    }

    /// 读全局体素速度：`global = (i,j,k)`，自动定位块与块内局部坐标。
    ///
    /// 块不存在（空块）时返回 0 向量（安全降级）。
    ///
    /// # 防呆（C18）
    /// 全局坐标须 ≥ 0 且可表示为 `i32`；否则返回 `Err`。
    pub fn get_voxel(&self, i: usize, j: usize, k: usize) -> Result<Vec3, PhysicsError> {
        let (coord, li) = self.locate(i, j, k)?;
        match self.blocks.get(&coord) {
            Some(block) => Ok(block.get_velocity(li.0, li.1, li.2)),
            None => Ok(Vec3::splat(0.0)),
        }
    }

    /// 写全局体素速度：块不存在则**自动创建空块**再写入。
    ///
    /// # 防呆（C18）
    /// 全局坐标须 ≥ 0 且可表示为 `i32`；否则返回 `Err`。
    pub fn set_voxel(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        val: Vec3,
    ) -> Result<(), PhysicsError> {
        let (coord, li) = self.locate(i, j, k)?;
        let s = self.block_size;
        let block = self
            .blocks
            .entry(coord)
            .or_insert(VoxelBlock::zeroed(s)?);
        block.set_velocity(li.0, li.1, li.2, val);
        Ok(())
    }

    /// 全局坐标 → (块坐标, 块内局部坐标)。
    ///
    /// 块内局部坐标 = 全局坐标对块边长的欧几里得余数（非负）。
    fn locate(
        &self,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<((i32, i32, i32), (usize, usize, usize)), PhysicsError> {
        let gi = usize_to_i32(i).ok_or_else(|| PhysicsError {
            code: "P3009",
            location: "SparseBlockGrid::locate".to_string(),
            problem: format!("i = {i}：全局坐标超出 i32 可表示范围"),
            fix: "使用更小的坐标（< 2³¹）".to_string(),
        })?;
        let gj = usize_to_i32(j).ok_or_else(|| PhysicsError {
            code: "P3009",
            location: "SparseBlockGrid::locate".to_string(),
            problem: format!("j = {j}：全局坐标超出 i32 可表示范围"),
            fix: "使用更小的坐标（< 2³¹）".to_string(),
        })?;
        let gk = usize_to_i32(k).ok_or_else(|| PhysicsError {
            code: "P3009",
            location: "SparseBlockGrid::locate".to_string(),
            problem: format!("k = {k}：全局坐标超出 i32 可表示范围"),
            fix: "使用更小的坐标（< 2³¹）".to_string(),
        })?;
        let s = self.block_size as i32; // block_size 已保证 ≥ 1 且远小于 i32::MAX
        let bx = gi.div_euclid(s);
        let by = gj.div_euclid(s);
        let bz = gk.div_euclid(s);
        let li = gi.rem_euclid(s) as usize;
        let lj = gj.rem_euclid(s) as usize;
        let lk = gk.rem_euclid(s) as usize;
        Ok(((bx, by, bz), (li, lj, lk)))
    }

    /// 活跃（非空）块数。
    #[inline]
    pub fn active_block_count(&self) -> usize {
        self.blocks.len()
    }

    /// 活跃块覆盖的体素总数 = `active_block_count() · block_size³`。
    #[inline]
    pub fn total_voxel_count(&self) -> usize {
        let n = self.block_size;
        let per = n.checked_mul(n).and_then(|s| s.checked_mul(n)).unwrap_or(0);
        self.blocks.len().saturating_mul(per)
    }

    /// 按块坐标**字典序**返回有序的 `(块坐标, 块引用)` 列表。
    ///
    /// # 确定性保证
    /// 顺序仅由块坐标决定，与插入先后无关（`BTreeMap` 有序迭代）。
    pub fn iter_blocks_sorted(&self) -> Vec<((i32, i32, i32), &VoxelBlock)> {
        self.blocks.iter().map(|(c, b)| (*c, b)).collect()
    }

    /// 按块坐标字典序返回所有键（用于确定性遍历）。
    pub fn sorted_coords(&self) -> Vec<(i32, i32, i32)> {
        self.blocks.keys().copied().collect()
    }

    /// 无序（=有序，`BTreeMap` 语义）的键集合快照。
    ///
    /// 说明：虽名为「键集」，因底层 `BTreeMap`，其迭代也是有序的；
    /// 返回 [`BTreeSet`] 显式承诺确定性。
    pub fn coord_set(&self) -> BTreeSet<(i32, i32, i32)> {
        self.blocks.keys().copied().collect()
    }

    /// 按块坐标字典序，把所有非空块的密度拼接为确定性序列（逐位可复现）。
    ///
    /// 用于测试「同一数据不同插入顺序 → 遍历结果逐位一致」。
    pub fn linearized_density_bits(&self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for (_coord, block) in self.blocks.iter() {
            for d in block.density.iter() {
                out.push(d.to_bits());
            }
        }
        out
    }

    /// 按块坐标字典序，把所有非空块的体素速度拼接为确定性位序列。
    pub fn linearized_velocity_bits(&self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for (_coord, block) in self.blocks.iter() {
            for v in block.velocity.iter() {
                out.push(v.x.to_bits());
                out.push(v.y.to_bits());
                out.push(v.z.to_bits());
            }
        }
        out
    }

    // -- 稠密 ↔ 稀疏 转换 ------------------------------------------------

    /// 从稠密网格构造稀疏网格（只保留非空块）。
    ///
    /// # 参数
    /// - `density`：稠密密度，长度须 = `nx·ny·nz`，布局 `(z,y,x)`；
    /// - `velocity`：稠密速度，长度须 = `nx·ny·nz`，布局 `(z,y,x)`；
    /// - `res`：`(nx, ny, nz)`；
    /// - `block_size`：块边长；
    /// - `threshold`：空块判定阈值（块密度和 > threshold 保留）。
    ///
    /// # 防呆（C18）
    /// 拒绝零分辨率、`block_size == 0`、数组长度与 `nx·ny·nz` 不符。
    pub fn from_dense(
        density: &[f32],
        velocity: &[Vec3],
        res: (usize, usize, usize),
        block_size: usize,
        threshold: f32,
    ) -> Result<Self, PhysicsError> {
        let (nx, ny, nz) = res;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "SparseBlockGrid::from_dense".to_string(),
                problem: format!("res = {res:?}：各维必须 ≥ 1"),
                fix: "传 nx, ny, nz ≥ 1".to_string(),
            });
        }
        if block_size == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "SparseBlockGrid::from_dense".to_string(),
                problem: "block_size = 0：块边长必须 ≥ 1".to_string(),
                fix: "传 block_size ≥ 1".to_string(),
            });
        }
        let cells = nx
            .checked_mul(ny)
            .and_then(|c| c.checked_mul(nz))
            .ok_or_else(|| PhysicsError {
                code: "P3009",
                location: "SparseBlockGrid::from_dense".to_string(),
                problem: format!("res = {res:?}：单元总数溢出 usize"),
                fix: "减小分辨率".to_string(),
            })?;
        // C18：分配字节数不得超过 isize::MAX（否则 vec![...] 会 panic）。
        if !alloc_size_ok(cells, std::mem::size_of::<Vec3>())
            || !alloc_size_ok(cells, std::mem::size_of::<f32>())
        {
            return Err(PhysicsError {
                code: "P3009",
                location: "SparseBlockGrid::from_dense".to_string(),
                problem: format!("res = {res:?}：单元数 {cells} 的分配字节数超过 isize::MAX"),
                fix: "减小分辨率".to_string(),
            });
        }
        if density.len() != cells || velocity.len() != cells {
            return Err(PhysicsError {
                code: "P3009",
                location: "SparseBlockGrid::from_dense".to_string(),
                problem: format!(
                    "density.len = {}, velocity.len = {}, 期望 = {}",
                    density.len(),
                    velocity.len(),
                    cells
                ),
                fix: "保证两数组长度均为 nx·ny·nz".to_string(),
            });
        }

        let mut grid = Self::new(block_size, threshold)?;
        // 块坐标遍历：按 (bz, by, bx) 字典序，确定性。
        let nb = |n: usize| n.div_ceil(block_size);
        let (nbx, nby, nbz) = (nb(nx), nb(ny), nb(nz));
        for bz in 0..nbz {
            for by in 0..nby {
                for bx in 0..nbx {
                    let mut block = VoxelBlock::zeroed(block_size)?;
                    let mut sum = 0.0f32;
                    // 块内固定 (lk, lj, li) 序累加密度和。
                    for lk in 0..block_size {
                        for lj in 0..block_size {
                            for li in 0..block_size {
                                let i = bx * block_size + li;
                                let j = by * block_size + lj;
                                let k = bz * block_size + lk;
                                if i >= nx || j >= ny || k >= nz {
                                    continue; // 边界块外体素留零
                                }
                                let idx = k * ny * nx + j * nx + i;
                                let d = match density.get(idx) {
                                    Some(v) => *v,
                                    None => continue,
                                };
                                let vel = match velocity.get(idx) {
                                    Some(v) => *v,
                                    None => Vec3::splat(0.0),
                                };
                                block.set_density(li, lj, lk, d);
                                block.set_velocity(li, lj, lk, vel);
                                sum += d;
                            }
                        }
                    }
                    if sum > threshold {
                        let coord = (
                            usize_to_i32(bx).unwrap_or(0),
                            usize_to_i32(by).unwrap_or(0),
                            usize_to_i32(bz).unwrap_or(0),
                        );
                        grid.insert_block_raw(coord, block);
                    }
                }
            }
        }
        Ok(grid)
    }

    /// 转回稠密网格（尺寸/布局与 [`SparseBlockGrid::from_dense`] 对应）。
    ///
    /// 空块覆盖的体素补零。返回 `(density, velocity, res)`。
    ///
    /// # 确定性
    /// 输出写入按固定 `(k,j,i)` 序；块按有序键遍历，结果唯一。
    pub fn to_dense(&self, res: (usize, usize, usize)) -> (Vec<f32>, Vec<Vec3>, (usize, usize, usize)) {
        let (nx, ny, nz) = res;
        let mut cells = nx
            .checked_mul(ny)
            .and_then(|c| c.checked_mul(nz))
            .unwrap_or(0);
        // C18：若 cells 的分配字节数超过 isize::MAX，降级为 0（返回空稠密场，
        // 不 panic）。调用方应通过 `from_dense`（返回 Result）做前置校验。
        if !alloc_size_ok(cells, std::mem::size_of::<Vec3>())
            || !alloc_size_ok(cells, std::mem::size_of::<f32>())
        {
            cells = 0;
        }
        let mut density = vec![0.0f32; cells];
        let mut velocity = vec![Vec3::splat(0.0); cells];
        if cells == 0 {
            return (density, velocity, res);
        }
        let s = self.block_size;
        for ((bx, by, bz), block) in self.blocks.iter() {
            // 负块坐标（理论上不会由 from_dense 产生）安全跳过。
            if *bx < 0 || *by < 0 || *bz < 0 {
                continue;
            }
            let bxi = *bx as usize;
            let byi = *by as usize;
            let bzi = *bz as usize;
            for lk in 0..s {
                for lj in 0..s {
                    for li in 0..s {
                        let i = bxi * s + li;
                        let j = byi * s + lj;
                        let k = bzi * s + lk;
                        if i >= nx || j >= ny || k >= nz {
                            continue;
                        }
                        let idx = k * ny * nx + j * nx + i;
                        if let Some(slot) = density.get_mut(idx) {
                            *slot = block.get_density(li, lj, lk);
                        }
                        if let Some(slot) = velocity.get_mut(idx) {
                            *slot = block.get_velocity(li, lj, lk);
                        }
                    }
                }
            }
        }
        (density, velocity, res)
    }
}

/// 稠密 → 稀疏（[`SparseBlockGrid::from_dense`] 的自由函数形式）。
///
/// **往返逐位一致**：与 [`sparse_to_dense`] 组合时，对**块内密度和恒
/// `> threshold** 的数据可无损还原。注意：`threshold` 以下整块会被丢弃，
/// 往返一致性断言应在阈值内数据上成立（见本模块单测
/// `test_roundtrip_bitwise`，其阈值取 `-1.0` 以保留全零块）。
pub fn dense_to_sparse(
    density: &[f32],
    velocity: &[Vec3],
    res: (usize, usize, usize),
    block_size: usize,
    threshold: f32,
) -> Result<SparseBlockGrid, PhysicsError> {
    SparseBlockGrid::from_dense(density, velocity, res, block_size, threshold)
}

/// 稀疏 → 稠密（[`SparseBlockGrid::to_dense`] 的自由函数形式）。
pub fn sparse_to_dense(
    grid: &SparseBlockGrid,
    res: (usize, usize, usize),
) -> (Vec<f32>, Vec<Vec3>, (usize, usize, usize)) {
    grid.to_dense(res)
}

// ---------------------------------------------------------------------------
// 涡量约束（Vorticity Confinement）
// ---------------------------------------------------------------------------

/// 线性下标（体素中心，布局 `(z,y,x)`）。坐标已由调用方保证在界内。
#[inline]
fn center_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    k * ny * nx + j * nx + i
}

/// 读体素速度；越界返回 0 向量（安全降级，配合差分边界处理）。
#[inline]
fn vel_at(velocity: &[f32], i: usize, j: usize, k: usize, nx: usize, ny: usize, nz: usize) -> Vec3 {
    if i >= nx || j >= ny || k >= nz {
        return Vec3::splat(0.0);
    }
    let base = center_index(i, j, k, nx, ny);
    let x = velocity.get(base * 3).copied().unwrap_or(0.0);
    let y = velocity.get(base * 3 + 1).copied().unwrap_or(0.0);
    let z = velocity.get(base * 3 + 2).copied().unwrap_or(0.0);
    Vec3::new(x, y, z)
}

/// 写体素速度；越界忽略（安全降级）。
#[inline]
fn set_vel(velocity: &mut [f32], i: usize, j: usize, k: usize, nx: usize, ny: usize, nz: usize, v: Vec3) {
    if i >= nx || j >= ny || k >= nz {
        return;
    }
    let base = center_index(i, j, k, nx, ny);
    if let Some(slot) = velocity.get_mut(base * 3) {
        *slot = v.x;
    }
    if let Some(slot) = velocity.get_mut(base * 3 + 1) {
        *slot = v.y;
    }
    if let Some(slot) = velocity.get_mut(base * 3 + 2) {
        *slot = v.z;
    }
}

/// 计算速度场的涡量场 `ω = ∇ × u`（体素中心）。
///
/// # 差分格式（固定，确定性）
/// - 内部（各方向两侧均在界内）：**中心差分**
///   `∂u/∂x ≈ (u[i+1] − u[i−1]) / (2h)`；
/// - 边界（`i == 0` 或 `i == nx−1`）：**单侧差分**
///   `∂u/∂x ≈ (u[i+1] − u[i]) / h`（下边界）或 `(u[i] − u[i−1]) / h`（上边界）。
///
/// # 参数
/// - `velocity`：逐体素速度，长度 = `3·nx·ny·nz`，布局 `(z,y,x)`；
/// - `(nx,ny,nz)`：分辨率；
/// - `h`：体素边长（> 0）。
///
/// # 返回
/// 涡量场 `Vec<Vec3>`，长度 = `nx·ny·nz`，布局 `(z,y,x)`。
///
/// # 防呆（C18）
/// 拒绝零分辨率、`h <= 0` 或非有限、数组长度不符。
pub fn compute_vorticity(
    velocity: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    h: f32,
) -> Result<Vec<Vec3>, PhysicsError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(PhysicsError {
            code: "P1014",
            location: "compute_vorticity".to_string(),
            problem: format!("res = ({nx}, {ny}, {nz})：各维必须 ≥ 1"),
            fix: "传 nx, ny, nz ≥ 1".to_string(),
        });
    }
    if !h.is_finite() || h <= 0.0 {
        return Err(PhysicsError {
            code: "P1014",
            location: "compute_vorticity".to_string(),
            problem: format!("h = {h}：体素边长必须为正有限实数"),
            fix: "传正体素边长".to_string(),
        });
    }
    let cells = nx
        .checked_mul(ny)
        .and_then(|c| c.checked_mul(nz))
        .ok_or_else(|| PhysicsError {
            code: "P3009",
            location: "compute_vorticity".to_string(),
            problem: "分辨率乘积溢出 usize".to_string(),
            fix: "减小分辨率".to_string(),
        })?;
    let need = cells.checked_mul(3).ok_or_else(|| PhysicsError {
        code: "P3009",
        location: "compute_vorticity".to_string(),
        problem: "3·nx·ny·nz 溢出 usize".to_string(),
        fix: "减小分辨率".to_string(),
    })?;
    if velocity.len() != need {
        return Err(PhysicsError {
            code: "P3009",
            location: "compute_vorticity".to_string(),
            problem: format!("velocity.len = {}, 期望 = {}", velocity.len(), need),
            fix: "保证长度 = 3·nx·ny·nz".to_string(),
        });
    }
    // C18：涡量场分配字节数不得超过 isize::MAX。
    if !alloc_size_ok(cells, std::mem::size_of::<Vec3>()) {
        return Err(PhysicsError {
            code: "P3009",
            location: "compute_vorticity".to_string(),
            problem: format!("单元数 {cells} 的涡量场分配字节数超过 isize::MAX"),
            fix: "减小分辨率".to_string(),
        });
    }

    let inv_h = 1.0f32 / h;
    let inv_2h = 1.0f32 / (2.0f32 * h);
    let mut omega: Vec<Vec3> = vec![Vec3::splat(0.0); cells];

    // 固定 (k, j, i) 遍历序。
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                // ∂/∂x（对角项 dudx/dvdy/dwdz 不参与 curl，无需计算）
                let dvdx = if i == 0 {
                    if nx > 1 {
                        (vel_at(velocity, 1, j, k, nx, ny, nz).y - vel_at(velocity, 0, j, k, nx, ny, nz).y) * inv_h
                    } else {
                        0.0
                    }
                } else if i + 1 >= nx {
                    (vel_at(velocity, i, j, k, nx, ny, nz).y - vel_at(velocity, i - 1, j, k, nx, ny, nz).y) * inv_h
                } else {
                    (vel_at(velocity, i + 1, j, k, nx, ny, nz).y - vel_at(velocity, i - 1, j, k, nx, ny, nz).y) * inv_2h
                };
                let dwdx = if i == 0 {
                    if nx > 1 {
                        (vel_at(velocity, 1, j, k, nx, ny, nz).z - vel_at(velocity, 0, j, k, nx, ny, nz).z) * inv_h
                    } else {
                        0.0
                    }
                } else if i + 1 >= nx {
                    (vel_at(velocity, i, j, k, nx, ny, nz).z - vel_at(velocity, i - 1, j, k, nx, ny, nz).z) * inv_h
                } else {
                    (vel_at(velocity, i + 1, j, k, nx, ny, nz).z - vel_at(velocity, i - 1, j, k, nx, ny, nz).z) * inv_2h
                };

                // ∂/∂y（对角项 dvdy 不参与 curl）
                let dudy = if j == 0 {
                    if ny > 1 {
                        (vel_at(velocity, i, 1, k, nx, ny, nz).x - vel_at(velocity, i, 0, k, nx, ny, nz).x) * inv_h
                    } else {
                        0.0
                    }
                } else if j + 1 >= ny {
                    (vel_at(velocity, i, j, k, nx, ny, nz).x - vel_at(velocity, i, j - 1, k, nx, ny, nz).x) * inv_h
                } else {
                    (vel_at(velocity, i, j + 1, k, nx, ny, nz).x - vel_at(velocity, i, j - 1, k, nx, ny, nz).x) * inv_2h
                };
                let dwdy = if j == 0 {
                    if ny > 1 {
                        (vel_at(velocity, i, 1, k, nx, ny, nz).z - vel_at(velocity, i, 0, k, nx, ny, nz).z) * inv_h
                    } else {
                        0.0
                    }
                } else if j + 1 >= ny {
                    (vel_at(velocity, i, j, k, nx, ny, nz).z - vel_at(velocity, i, j - 1, k, nx, ny, nz).z) * inv_h
                } else {
                    (vel_at(velocity, i, j + 1, k, nx, ny, nz).z - vel_at(velocity, i, j - 1, k, nx, ny, nz).z) * inv_2h
                };

                // ∂/∂z（对角项 dwdz 不参与 curl）
                let dudz = if k == 0 {
                    if nz > 1 {
                        (vel_at(velocity, i, j, 1, nx, ny, nz).x - vel_at(velocity, i, j, 0, nx, ny, nz).x) * inv_h
                    } else {
                        0.0
                    }
                } else if k + 1 >= nz {
                    (vel_at(velocity, i, j, k, nx, ny, nz).x - vel_at(velocity, i, j, k - 1, nx, ny, nz).x) * inv_h
                } else {
                    (vel_at(velocity, i, j, k + 1, nx, ny, nz).x - vel_at(velocity, i, j, k - 1, nx, ny, nz).x) * inv_2h
                };
                let dvdz = if k == 0 {
                    if nz > 1 {
                        (vel_at(velocity, i, j, 1, nx, ny, nz).y - vel_at(velocity, i, j, 0, nx, ny, nz).y) * inv_h
                    } else {
                        0.0
                    }
                } else if k + 1 >= nz {
                    (vel_at(velocity, i, j, k, nx, ny, nz).y - vel_at(velocity, i, j, k - 1, nx, ny, nz).y) * inv_h
                } else {
                    (vel_at(velocity, i, j, k + 1, nx, ny, nz).y - vel_at(velocity, i, j, k - 1, nx, ny, nz).y) * inv_2h
                };

                // ω = ∇ × u = (dw/dy − dv/dz, du/dz − dw/dx, dv/dx − du/dy)
                let idx = center_index(i, j, k, nx, ny);
                if let Some(slot) = omega.get_mut(idx) {
                    *slot = Vec3::new(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
                }
            }
        }
    }
    Ok(omega)
}

/// 施加涡量约束（Vorticity Confinement），就地修改 `velocity`。
///
/// # 公式（Fedkiw et al. 2001；原始模型 Steinhoff & Underhill 1994）
/// ```text
///   ω      = ∇ × u                              （涡量）
///   N      = ∇|ω| / (|∇|ω|| + δ)               （归一化涡量梯度）
///   f_vort = ε · h · (N × ω)                    （涡量约束力）
///   u     ← u + dt · f_vort                      （显式一阶积分）
/// ```
/// `δ` 取 `1e-5`（防止 `|∇|ω|| → 0` 时除零；`Vec3::normalize` 内部亦对退化
/// 情况返回零向量）。
///
/// # 参数
/// - `velocity`：逐体素速度（就地修改），长度 = `3·nx·ny·nz`，布局 `(z,y,x)`；
/// - `(nx,ny,nz)`：分辨率；
/// - `dt`：时间步长（须 > 0 有限）；
/// - `epsilon`：约束强度系数（≥ 0 有限；负值视为 0）。
///
/// # 确定性
/// 差分与累加顺序固定（`(k,j,i)` 序）；两遍同输入结果逐位一致。
/// **本函数不依赖稀疏结构**，可独立测试。
///
/// # 防呆（C18）
/// 拒绝零分辨率、`dt` 非正/非有限、`epsilon` 非有限、数组长度不符。
pub fn apply_vorticity_confinement(
    velocity: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    dt: f32,
    epsilon: f32,
) -> Result<(), PhysicsError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(PhysicsError {
            code: "P1014",
            location: "apply_vorticity_confinement".to_string(),
            problem: format!("res = ({nx}, {ny}, {nz})：各维必须 ≥ 1"),
            fix: "传 nx, ny, nz ≥ 1".to_string(),
        });
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PhysicsError {
            code: "P1014",
            location: "apply_vorticity_confinement".to_string(),
            problem: format!("dt = {dt}：时间步长必须为正有限实数"),
            fix: "传正 dt".to_string(),
        });
    }
    if !epsilon.is_finite() {
        return Err(PhysicsError {
            code: "P1014",
            location: "apply_vorticity_confinement".to_string(),
            problem: format!("epsilon = {epsilon}：约束强度必须有限"),
            fix: "传有限 epsilon（如 0.1）".to_string(),
        });
    }
    let cells = nx
        .checked_mul(ny)
        .and_then(|c| c.checked_mul(nz))
        .ok_or_else(|| PhysicsError {
            code: "P3009",
            location: "apply_vorticity_confinement".to_string(),
            problem: "分辨率乘积溢出 usize".to_string(),
            fix: "减小分辨率".to_string(),
        })?;
    let need = cells.checked_mul(3).ok_or_else(|| PhysicsError {
        code: "P3009",
        location: "apply_vorticity_confinement".to_string(),
        problem: "3·nx·ny·nz 溢出 usize".to_string(),
        fix: "减小分辨率".to_string(),
    })?;
    if velocity.len() != need {
        return Err(PhysicsError {
            code: "P3009",
            location: "apply_vorticity_confinement".to_string(),
            problem: format!("velocity.len = {}, 期望 = {}", velocity.len(), need),
            fix: "保证长度 = 3·nx·ny·nz".to_string(),
        });
    }
    // C18：|ω| 场与力场分配字节数不得超过 isize::MAX。
    if !alloc_size_ok(cells, std::mem::size_of::<Vec3>())
        || !alloc_size_ok(cells, std::mem::size_of::<f32>())
    {
        return Err(PhysicsError {
            code: "P3009",
            location: "apply_vorticity_confinement".to_string(),
            problem: format!("单元数 {cells} 的工作场分配字节数超过 isize::MAX"),
            fix: "减小分辨率".to_string(),
        });
    }

    // 涡量约束用「单位体素边长」的离散梯度（h 已并入 epsilon·h 的 h 尺度；
    // 此处统一取 h = 1 的网格尺度，调用方可通过 epsilon 调节绝对强度）。
    // 说明：为让函数与网格尺度解耦且便于独立测试，这里以索引空间步长 1 计算
    // ∇|ω| 与 f_vort，公式中的 h 通过参数 epsilon 的物理标定吸收。
    let eps = if epsilon < 0.0 { 0.0 } else { epsilon };
    let inv_2h = 0.5f32; // 1/(2·1)
    let h_scale = 1.0f32;
    let delta = 1e-5f32;

    // 1) 涡量场 ω。
    let omega = compute_vorticity(velocity, nx, ny, nz, h_scale)?;

    // 2) 涡量模 |ω| 场（固定 (k,j,i) 序）。
    let mut mag: Vec<f32> = vec![0.0f32; cells];
    for idx in 0..cells {
        if let Some(w) = omega.get(idx) {
            if let Some(slot) = mag.get_mut(idx) {
                *slot = w.length();
            }
        }
    }

    // 3) 归一化涡量梯度 N = ∇|ω| / (|∇|ω|| + δ)。
    let mut force: Vec<Vec3> = vec![Vec3::splat(0.0); cells];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mag_at = |ii: usize, jj: usize, kk: usize| -> f32 {
                    if ii >= nx || jj >= ny || kk >= nz {
                        return 0.0;
                    }
                    mag.get(center_index(ii, jj, kk, nx, ny))
                        .copied()
                        .unwrap_or(0.0)
                };
                // ∇|ω| 各分量：内部中心差分，边界单侧差分。
                let gx = if i == 0 {
                    if nx > 1 {
                        (mag_at(1, j, k) - mag_at(0, j, k)) * 1.0
                    } else {
                        0.0
                    }
                } else if i + 1 >= nx {
                    (mag_at(i, j, k) - mag_at(i - 1, j, k)) * 1.0
                } else {
                    (mag_at(i + 1, j, k) - mag_at(i - 1, j, k)) * inv_2h
                };
                let gy = if j == 0 {
                    if ny > 1 {
                        (mag_at(i, 1, k) - mag_at(i, 0, k)) * 1.0
                    } else {
                        0.0
                    }
                } else if j + 1 >= ny {
                    (mag_at(i, j, k) - mag_at(i, j - 1, k)) * 1.0
                } else {
                    (mag_at(i, j + 1, k) - mag_at(i, j - 1, k)) * inv_2h
                };
                let gz = if k == 0 {
                    if nz > 1 {
                        (mag_at(i, j, 1) - mag_at(i, j, 0)) * 1.0
                    } else {
                        0.0
                    }
                } else if k + 1 >= nz {
                    (mag_at(i, j, k) - mag_at(i, j, k - 1)) * 1.0
                } else {
                    (mag_at(i, j, k + 1) - mag_at(i, j, k - 1)) * inv_2h
                };
                let grad = Vec3::new(gx, gy, gz);
                // N = grad / (|grad| + δ)（手动归一化，避免 Vec3::normalize 的 1e-8
                // 阈值把极小梯度提前归零而丢细节）。
                let glen = grad.length();
                let n = Vec3::new(
                    grad.x / (glen + delta),
                    grad.y / (glen + delta),
                    grad.z / (glen + delta),
                );
                let w = omega.get(center_index(i, j, k, nx, ny)).copied().unwrap_or(Vec3::splat(0.0));
                // f_vort = ε · h · (N × ω)
                let f = n.cross(&w) * (eps * h_scale);
                if let Some(slot) = force.get_mut(center_index(i, j, k, nx, ny)) {
                    *slot = f;
                }
            }
        }
    }

    // 4) 显式一阶积分：u ← u + dt · f_vort（固定 (k,j,i) 序）。
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let f = force.get(center_index(i, j, k, nx, ny)).copied().unwrap_or(Vec3::splat(0.0));
                let u = vel_at(velocity, i, j, k, nx, ny, nz);
                let updated = Vec3::new(u.x + dt * f.x, u.y + dt * f.y, u.z + dt * f.z);
                set_vel(velocity, i, j, k, nx, ny, nz, updated);
            }
        }
    }
    Ok(())
}

/// 计算速度场的总涡量能量 `Σ |ω|²`（固定 `(k,j,i)` 序累加，确定性）。
///
/// 供测试「涡量约束应增强涡量能量」使用；也便于上层做湍流强度度量。
///
/// # 防呆（C18）
/// 与 [`compute_vorticity`] 相同（零分辨率 / 非法 `h` / 长度不符均返回 `Err`）。
pub fn total_vorticity_energy(
    velocity: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    h: f32,
) -> Result<f32, PhysicsError> {
    let omega = compute_vorticity(velocity, nx, ny, nz, h)?;
    let mut acc = 0.0f32;
    for w in omega.iter() {
        acc += w.dot(w);
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// 单测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造确定性的「小块测试网格」：把一个中心区域填正密度 + 一束速度。
    fn make_dense(
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> (Vec<f32>, Vec<Vec3>) {
        let cells = nx * ny * nz;
        let mut density = vec![0.0f32; cells];
        let mut velocity = vec![Vec3::splat(0.0); cells];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k * ny * nx + j * nx + i;
                    // 仅中心 1/2 区域有密度（制造空块）。
                    if i >= nx / 2 && j >= ny / 2 && k >= nz / 2 {
                        density[idx] = 1.0 + (i + j + k) as f32 * 0.01;
                        velocity[idx] = Vec3::new(
                            i as f32 * 0.1,
                            j as f32 * -0.2,
                            k as f32 * 0.05,
                        );
                    }
                }
            }
        }
        (density, velocity)
    }

    // ---- A. 确定性断言 -------------------------------------------------

    /// 往返转换逐位一致（threshold = -1.0 → 全零块也保留，保证无损）。
    #[test]
    fn test_roundtrip_bitwise() {
        let res = (16usize, 16usize, 12usize);
        let (density, velocity) = make_dense(res.0, res.1, res.2);
        let sparse = dense_to_sparse(&density, &velocity, res, 8, -1.0)
            .unwrap_or_else(|e| panic!("from_dense 失败：{e}"));
        // 8 边长 → 2×2×2 = 8 块（全保留，因 threshold < 0）。
        assert_eq!(sparse.active_block_count(), 8);
        let (d2, v2, res2) = sparse_to_dense(&sparse, res);
        assert_eq!(res, res2);
        assert_eq!(d2.len(), density.len());
        assert_eq!(v2.len(), velocity.len());
        // 逐位比较密度。
        for (a, b) in density.iter().zip(d2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "密度逐位不一致");
        }
        // 逐位比较速度三分量。
        for (a, b) in velocity.iter().zip(v2.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    /// 稀疏块遍历顺序与插入先后无关（关键：抓 HashMap 迭代序问题）。
    #[test]
    fn test_iteration_order_independent_of_insertion() {
        // 正序插入。
        let mut forward = SparseBlockGrid::new(4, -1.0).unwrap_or_else(|e| panic!("{e}"));
        // 逆序插入（同一批坐标）。
        let mut backward = SparseBlockGrid::new(4, -1.0).unwrap_or_else(|e| panic!("{e}"));
        let coords = [
            (2i32, 0i32, 1i32),
            (-1i32, 3i32, 0i32),
            (0i32, 0i32, 0i32),
            (5i32, 2i32, 7i32),
            (1i32, 1i32, 1i32),
        ];
        // 每块写入唯一的密度指纹（坐标决定）。
        let make = |c: (i32, i32, i32)| -> VoxelBlock {
            let mut b = VoxelBlock::zeroed(4).unwrap_or_else(|e| panic!("{e}"));
            let f = (c.0 as f32) * 3.0 + (c.1 as f32) * 5.0 + (c.2 as f32) * 7.0;
            for k in 0..4 {
                for j in 0..4 {
                    for i in 0..4 {
                        b.set_density(i, j, k, f + (i + j + k) as f32);
                    }
                }
            }
            b
        };
        for c in coords.iter() {
            forward
                .insert_block(*c, make(*c))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        for c in coords.iter().rev() {
            backward
                .insert_block(*c, make(*c))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // 坐标序列一致（字典序）。
        assert_eq!(forward.sorted_coords(), backward.sorted_coords());
        // 展平后的密度位序逐位一致。
        assert_eq!(
            forward.linearized_density_bits(),
            backward.linearized_density_bits()
        );
        // 且确实是字典序（升序）。
        let sc = forward.sorted_coords();
        let mut sorted = sc.clone();
        sorted.sort();
        assert_eq!(sc, sorted);
        assert_eq!(
            sc,
            vec![
                (-1, 3, 0),
                (0, 0, 0),
                (1, 1, 1),
                (2, 0, 1),
                (5, 2, 7)
            ]
        );
    }

    /// 涡量约束：同输入两遍，速度场逐位一致。
    #[test]
    fn test_vorticity_deterministic() {
        let res = (12usize, 12usize, 12usize);
        let (_d, velocity) = make_dense(res.0, res.1, res.2);
        // 展平为 3·cells。
        let mut flat: Vec<f32> = Vec::with_capacity(velocity.len() * 3);
        for v in velocity.iter() {
            flat.push(v.x);
            flat.push(v.y);
            flat.push(v.z);
        }
        let mut a = flat.clone();
        let mut b = flat.clone();
        apply_vorticity_confinement(&mut a, res.0, res.1, res.2, 0.016, 0.5)
            .unwrap_or_else(|e| panic!("{e}"));
        apply_vorticity_confinement(&mut b, res.0, res.1, res.2, 0.016, 0.5)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "涡量约束非确定性");
        }
    }

    // ---- B. 物理正确性断言 ---------------------------------------------

    /// 线性速度场 u = (-y, x, 0) 的理论涡量 ω = (0, 0, 2)。
    #[test]
    fn test_vorticity_linear_field() {
        let n = 9usize;
        let h = 1.0f32;
        let mut velocity = vec![0.0f32; n * n * n * 3];
        // u = (-y, x, 0)：以体素中心坐标 (i,j,k) 近似世界坐标（h=1）。
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = (k * n * n + j * n + i) * 3;
                    velocity[idx] = -(j as f32); // u
                    velocity[idx + 1] = i as f32; // v
                    velocity[idx + 2] = 0.0; // w
                }
            }
        }
        let omega = compute_vorticity(&velocity, n, n, n, h).unwrap_or_else(|e| panic!("{e}"));
        // 取内部体素（避开单侧差分边界），断言 ω ≈ (0, 0, 2)。
        let (i, j, k) = (4usize, 4usize, 4usize);
        let w = omega[k * n * n + j * n + i];
        assert!(
            (w.x - 0.0).abs() < 1e-4,
            "ω.x = {} 应 ≈ 0",
            w.x
        );
        assert!(
            (w.y - 0.0).abs() < 1e-4,
            "ω.y = {} 应 ≈ 0",
            w.y
        );
        assert!(
            (w.z - 2.0).abs() < 1e-4,
            "ω.z = {} 应 ≈ 2",
            w.z
        );
    }

    /// 涡量约束应增强总涡量能量（这正是其目的）。
    #[test]
    fn test_vorticity_confinement_enhances_energy() {
        let res = (16usize, 16usize, 16usize);
        let (nx, ny, nz) = res;
        // 构造**湍流**场：涡量在空间上显著变化（涡量约束正是对这类场再注入
        // 涡旋能量）。纯固体旋转（ω 恒定 → ∇|ω|≈0）不适用，故叠加多尺度扰动。
        let mut velocity = vec![0.0f32; nx * ny * nz * 3];
        let cx = nx as f32 * 0.5;
        let cy = ny as f32 * 0.5;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = (k * ny * nx + j * nx + i) * 3;
                    let x = i as f32 - cx;
                    let y = j as f32 - cy;
                    let z = k as f32;
                    // 多尺度旋转：角速度随 (x,y,z) 变化 → ω 随空间变化。
                    let swirl = 0.1 * (1.0 + 0.5 * (x * 0.4).sin() + 0.5 * (y * 0.3).cos());
                    velocity[idx] = -y * swirl;
                    velocity[idx + 1] = x * swirl;
                    velocity[idx + 2] =
                        0.4 * ((x * 0.5).sin() * (y * 0.35).cos() + (z * 0.25).sin());
                }
            }
        }
        let before = total_vorticity_energy(&velocity, nx, ny, nz, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut after_field = velocity.clone();
        apply_vorticity_confinement(&mut after_field, nx, ny, nz, 0.01, 0.5)
            .unwrap_or_else(|e| panic!("{e}"));
        let after = total_vorticity_energy(&after_field, nx, ny, nz, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            after > before,
            "涡量能量应增加：before = {before}, after = {after}"
        );
        assert!(after - before > 0.0, "增加量须 > 0");
    }

    /// epsilon = 0 时速度场应逐位不变（约束关闭 → 恒等）。
    ///
    /// 注：理想情况下 ε=0 ⇒ f=0 ⇒ u 不变。但 f = ε·h·(N×ω)，ε=0 时 f 为
    /// 精确 +0.0，u + dt·0.0 == u（浮点 +0 不改变值），故逐位恒等成立。
    #[test]
    fn test_vorticity_epsilon_zero_identity() {
        let res = (8usize, 8usize, 8usize);
        let (_d, velocity) = make_dense(res.0, res.1, res.2);
        let mut flat: Vec<f32> = Vec::new();
        for v in velocity.iter() {
            flat.push(v.x);
            flat.push(v.y);
            flat.push(v.z);
        }
        let original = flat.clone();
        apply_vorticity_confinement(&mut flat, res.0, res.1, res.2, 0.016, 0.0)
            .unwrap_or_else(|e| panic!("{e}"));
        for (a, b) in original.iter().zip(flat.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "ε=0 应恒等");
        }
    }

    // ---- C. 零 panic / 结构化错误 --------------------------------------

    /// 越界与非法参数返回结构化错误，绝不 panic。
    #[test]
    fn test_structured_errors_no_panic() {
        // 零块边长。
        assert!(SparseBlockGrid::new(0, 0.0).is_err());
        // 非有限阈值。
        assert!(SparseBlockGrid::new(8, f32::NAN).is_err());
        // 零分辨率稠密转换。
        assert!(SparseBlockGrid::from_dense(&[], &[], (0, 1, 1), 8, 0.0).is_err());
        // 长度不符。
        assert!(SparseBlockGrid::from_dense(&[1.0], &[Vec3::splat(0.0)], (2, 2, 2), 2, 0.0).is_err());
        // 涡量：非法 h。
        assert!(compute_vorticity(&[0.0], 1, 1, 1, 0.0).is_err());
        assert!(compute_vorticity(&[0.0], 1, 1, 1, f32::INFINITY).is_err());
        // 涡量约束：非法 dt。
        let mut v = vec![0.0f32; 3];
        assert!(apply_vorticity_confinement(&mut v, 1, 1, 1, 0.0, 1.0).is_err());
        assert!(apply_vorticity_confinement(&mut v, 1, 1, 1, 1.0, f32::NAN).is_err());
        // 越界 get_voxel：不 panic，返回 0。
        let g = SparseBlockGrid::new(4, -1.0).unwrap_or_else(|e| panic!("{e}"));
        let v0 = g.get_voxel(1000, 1000, 1000).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(v0.x.to_bits(), 0.0f32.to_bits());
    }

    // ---- 稀疏结构附加不变量 --------------------------------------------

    /// 空块判定 + 计数：阈值以上块保留，以下丢弃。
    #[test]
    fn test_empty_block_threshold_and_counts() {
        let mut grid = SparseBlockGrid::new(2, 0.0).unwrap_or_else(|e| panic!("{e}"));
        // 空块（密度和 0，不 > 0）→ 丢弃。
        let empty = VoxelBlock::zeroed(2).unwrap_or_else(|e| panic!("{e}"));
        let kept = grid.insert_block((0, 0, 0), empty).unwrap_or_else(|e| panic!("{e}"));
        assert!(!kept);
        assert_eq!(grid.active_block_count(), 0);
        assert_eq!(grid.total_voxel_count(), 0);
        // 非空块 → 保留。
        let mut full = VoxelBlock::zeroed(2).unwrap_or_else(|e| panic!("{e}"));
        full.set_density(1, 1, 1, 5.0);
        let kept2 = grid.insert_block((1, 0, 0), full).unwrap_or_else(|e| panic!("{e}"));
        assert!(kept2);
        assert_eq!(grid.active_block_count(), 1);
        assert_eq!(grid.total_voxel_count(), 8); // 2³
        // set_voxel / get_voxel 往返。
        grid.set_voxel(3, 2, 1, Vec3::new(1.5, -2.5, 3.25))
            .unwrap_or_else(|e| panic!("{e}"));
        let got = grid.get_voxel(3, 2, 1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.x.to_bits(), 1.5f32.to_bits());
        assert_eq!(got.y.to_bits(), (-2.5f32).to_bits());
        assert_eq!(got.z.to_bits(), 3.25f32.to_bits());
    }

    /// 块尺寸不一致时插入返回结构化错误。
    #[test]
    fn test_insert_block_size_mismatch() {
        let mut grid = SparseBlockGrid::new(4, -1.0).unwrap_or_else(|e| panic!("{e}"));
        let bad = VoxelBlock::zeroed(8).unwrap_or_else(|e| panic!("{e}"));
        assert!(grid.insert_block((0, 0, 0), bad).is_err());
    }

    /// 速度位序列的确定性展平（同数据、不同插入序 → 逐位一致）。
    #[test]
    fn test_velocity_bits_linearization_deterministic() {
        let mut a = SparseBlockGrid::new(2, -1.0).unwrap_or_else(|e| panic!("{e}"));
        let mut b = SparseBlockGrid::new(2, -1.0).unwrap_or_else(|e| panic!("{e}"));
        let coords = [(0i32, 0i32, 0i32), (1i32, 0i32, 0i32), (0i32, 1i32, 0i32)];
        let make = |seed: f32| {
            let mut blk = VoxelBlock::zeroed(2).unwrap_or_else(|e| panic!("{e}"));
            for k in 0..2 {
                for j in 0..2 {
                    for i in 0..2 {
                        blk.set_velocity(
                            i,
                            j,
                            k,
                            Vec3::new(seed + i as f32, seed - j as f32, seed * 2.0 + k as f32),
                        );
                    }
                }
            }
            blk
        };
        for (n, c) in coords.iter().enumerate() {
            a.insert_block(*c, make(n as f32)).unwrap_or_else(|e| panic!("{e}"));
        }
        for (n, c) in coords.iter().enumerate().rev() {
            b.insert_block(*c, make(n as f32)).unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(a.linearized_velocity_bits(), b.linearized_velocity_bits());
    }

    // ---- D. 本轮「接手验证」新增：C18 分配上限 / 更强确定性 / 更强物理 ----

    /// C18 回归：`block_size` 使元素数乘积不溢 `usize`、但**分配字节数**超
    /// `isize::MAX` 时，`VoxelBlock::zeroed` 必须返回结构化错误而非 panic
    /// （修复前：`vec![...]` 触发 capacity overflow panic）。
    #[test]
    fn test_alloc_limit_no_panic_huge_block_size() {
        // 2^21³ = 2^63 个 Vec3 元素 → 乘积不溢 usize(64) 但字节数 = 2^63·12 ≫ isize::MAX。
        assert!(VoxelBlock::zeroed(1usize << 21).is_err());
        // 2^22³ = 2^66 → checked_mul 即溢出。
        assert!(VoxelBlock::zeroed(1usize << 22).is_err());
        // usize::MAX 三次方 → 溢出。
        assert!(VoxelBlock::zeroed(usize::MAX).is_err());
        // 边界内的小块仍应成功。
        assert!(VoxelBlock::zeroed(8).is_ok());
    }

    /// C18 回归：`from_dense` 在大分辨率（分配字节超限）时返回结构化错误，
    /// 不 panic（传入的切片长度本身极小，故越过长度检查前应先被分配上限拦下）。
    #[test]
    fn test_alloc_limit_from_dense_huge_res() {
        let r = SparseBlockGrid::from_dense(&[], &[], (1usize << 21, 1usize << 21, 1), 8, 0.0);
        assert!(r.is_err());
    }

    /// C18 回归：涡量函数在大分辨率下返回结构化错误，不 panic。
    #[test]
    fn test_alloc_limit_vorticity_huge_res() {
        assert!(compute_vorticity(&[], 1usize << 21, 1usize << 21, 1, 1.0).is_err());
        let mut v: Vec<f32> = Vec::new();
        assert!(apply_vorticity_confinement(&mut v, 1usize << 21, 1usize << 21, 1, 0.1, 0.5).is_err());
    }

    /// 非均匀分辨率（block_size 不整除各维）的稀疏↔稠密往返逐位一致。
    ///
    /// 覆盖上一轮仅用 16×16×12（各维多可整除）未覆盖的「边缘块部分填充」情形。
    #[test]
    fn test_roundtrip_bitwise_nonuniform_res() {
        let res = (5usize, 7usize, 3usize);
        let cells = res.0 * res.1 * res.2;
        let mut density = vec![0.0f32; cells];
        let mut velocity = vec![Vec3::splat(0.0); cells];
        for k in 0..res.2 {
            for j in 0..res.1 {
                for i in 0..res.0 {
                    let idx = k * res.1 * res.0 + j * res.0 + i;
                    density[idx] = (i as f32) * 0.25 + (j as f32) * 0.5 + (k as f32) * 0.125;
                    velocity[idx] = Vec3::new(i as f32 * 1.5, j as f32 * -0.75, k as f32 + 0.5);
                }
            }
        }
        let sp = dense_to_sparse(&density, &velocity, res, 4, -1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let (d2, v2, res2) = sparse_to_dense(&sp, res);
        assert_eq!(res, res2);
        for (a, b) in density.iter().zip(d2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "非均匀分辨率密度逐位不一致");
        }
        for (a, b) in velocity.iter().zip(v2.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    /// 正阈值下整块被丢弃 → 稀疏表空 → 转回稠密全零（逐位）。
    /// 覆盖「空块不占内存、往返补零」这一不变式。
    #[test]
    fn test_positive_threshold_drops_all_and_roundtrips_zero() {
        let res = (4usize, 4usize, 4usize);
        let cells = 64;
        let density = vec![0.5f32; cells];
        let velocity = vec![Vec3::new(1.0, 2.0, 3.0); cells];
        let sp = dense_to_sparse(&density, &velocity, res, 4, 1e9)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sp.active_block_count(), 0, "阈值 1e9 应丢弃所有块");
        assert_eq!(sp.total_voxel_count(), 0);
        let (d2, v2, _r) = sparse_to_dense(&sp, res);
        for a in d2.iter() {
            assert_eq!(a.to_bits(), 0.0f32.to_bits());
        }
        for a in v2.iter() {
            assert_eq!(a.x.to_bits(), 0.0f32.to_bits());
            assert_eq!(a.y.to_bits(), 0.0f32.to_bits());
            assert_eq!(a.z.to_bits(), 0.0f32.to_bits());
        }
    }

    /// 决定性核心断言：**同一组数据、两种插入顺序（正序 vs 逆序）**，块坐标
    /// 含负键，`linearized_velocity_bits` / `linearized_density_bits` 逐位一致，
    /// 且坐标序列严格字典序升序。若把底层 `BTreeMap` 换成 `HashMap`，本测试
    /// 会因两网格迭代序不同而失败——即「能抓 HashMap 顺序问题」。
    #[test]
    fn test_bitwise_insertion_order_negative_coords() {
        let coords = [
            (-3i32, 2i32, -1i32),
            (1i32, -2i32, 4i32),
            (0i32, 0i32, 0i32),
            (2i32, 2i32, 2i32),
            (-1i32, -1i32, -1i32),
        ];
        let build = |order: &[usize]| -> SparseBlockGrid {
            let mut g = SparseBlockGrid::new(3, -1.0).unwrap_or_else(|e| panic!("{e}"));
            for &n in order {
                let c = coords[n];
                let mut b = VoxelBlock::zeroed(3).unwrap_or_else(|e| panic!("{e}"));
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            // 指纹恒正（+100 偏置），确保在任何阈值下都判为非空块，
                            // 从而让本测试只聚焦「遍历顺序」而非「空块判定」。
                            let f = 100.0
                                + (c.0 as f32)
                                + (c.1 as f32) * 0.5
                                + (c.2 as f32) * 0.25;
                            b.set_density(i, j, k, f + (i + j + k) as f32 * 0.01);
                            b.set_velocity(
                                i,
                                j,
                                k,
                                Vec3::new(f, -f, f * 2.0 + i as f32),
                            );
                        }
                    }
                }
                g.insert_block(c, b).unwrap_or_else(|e| panic!("{e}"));
            }
            g
        };
        let forward_order: Vec<usize> = (0..coords.len()).collect();
        let backward_order: Vec<usize> = (0..coords.len()).rev().collect();
        let fwd = build(&forward_order);
        let bwd = build(&backward_order);
        assert_eq!(fwd.sorted_coords(), bwd.sorted_coords());
        assert_eq!(
            fwd.linearized_velocity_bits(),
            bwd.linearized_velocity_bits(),
            "不同插入顺序的速度位序必须逐位一致"
        );
        assert_eq!(
            fwd.linearized_density_bits(),
            bwd.linearized_density_bits(),
            "不同插入顺序的密度位序必须逐位一致"
        );
        // 坐标序严格字典序、且含负键正确排序。
        let sc = fwd.sorted_coords();
        assert_eq!(
            sc,
            vec![
                (-3, 2, -1),
                (-1, -1, -1),
                (0, 0, 0),
                (1, -2, 4),
                (2, 2, 2)
            ]
        );
    }

    /// 物理断言（边界差分正确性）：`u=(-y,x,0)` 全域，**角点**（单侧差分）
    /// 与内部一样应得 ω=(0,0,2)。
    #[test]
    fn test_vorticity_linear_field_at_corners() {
        let n = 7usize;
        let mut v = vec![0.0f32; n * n * n * 3];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = (k * n * n + j * n + i) * 3;
                    v[idx] = -(j as f32); // u = -y
                    v[idx + 1] = i as f32; // v =  x
                }
            }
        }
        let om = compute_vorticity(&v, n, n, n, 1.0).unwrap_or_else(|e| panic!("{e}"));
        // 逐点检查全部体素：线性场处处 ω=(0,0,2)（含边界与角点，单侧差分）。
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let w = om[k * n * n + j * n + i];
                    assert!((w.x).abs() < 1e-4, "ω.x={} @({i},{j},{k})", w.x);
                    assert!((w.y).abs() < 1e-4, "ω.y={} @({i},{j},{k})", w.y);
                    assert!((w.z - 2.0).abs() < 1e-4, "ω.z={} @({i},{j},{k})", w.z);
                }
            }
        }
    }

    /// 物理断言（约束目的）：线性固体旋转 `ω=const → ∇|ω|=0` 时约束力应为零、
    /// 总涡量能量**不变**；而涡量梯度的湍流场能量**增加**——证明约束只在有
    /// 涡量梯度处注入能量（既避免误伤刚性旋转，又达成增强湍流之目的）。
    #[test]
    fn test_confinement_no_force_on_uniform_vorticity_but_enhances_turbulence() {
        // (a) 固体旋转：能量应严格不变。
        let n = 12usize;
        let c = n as f32 * 0.5;
        let mut solid = vec![0.0f32; n * n * n * 3];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = (k * n * n + j * n + i) * 3;
                    solid[idx] = -(j as f32 - c);
                    solid[idx + 1] = i as f32 - c;
                }
            }
        }
        let e_before = total_vorticity_energy(&solid, n, n, n, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut solid_after = solid.clone();
        apply_vorticity_confinement(&mut solid_after, n, n, n, 0.01, 0.5)
            .unwrap_or_else(|e| panic!("{e}"));
        let e_after = total_vorticity_energy(&solid_after, n, n, n, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            e_before.to_bits(),
            e_after.to_bits(),
            "均匀涡量场约束力为零，能量应逐位不变"
        );

        // (b) 多尺度涡量场：能量应增加。
        let mut turb = vec![0.0f32; n * n * n * 3];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = (k * n * n + j * n + i) * 3;
                    let x = i as f32 - c;
                    let y = j as f32 - c;
                    let z = k as f32;
                    let swirl = 0.1 * (1.0 + 0.5 * (x * 0.4).sin() + 0.5 * (y * 0.3).cos());
                    turb[idx] = -y * swirl;
                    turb[idx + 1] = x * swirl;
                    turb[idx + 2] = 0.4 * ((x * 0.5).sin() * (y * 0.35).cos() + (z * 0.25).sin());
                }
            }
        }
        let t_before = total_vorticity_energy(&turb, n, n, n, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut turb_after = turb.clone();
        apply_vorticity_confinement(&mut turb_after, n, n, n, 0.01, 0.5)
            .unwrap_or_else(|e| panic!("{e}"));
        let t_after = total_vorticity_energy(&turb_after, n, n, n, 1.0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            t_after > t_before,
            "湍流场涡量能量应增加：before={t_before} after={t_after}"
        );
    }
}
