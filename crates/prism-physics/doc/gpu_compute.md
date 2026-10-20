# GPU Compute 设计文档（仅设计，不实装）

> **状态**：设计文档。GPU 路径**尚不实装**——标注列明
> "SPH 3D、GPU compute = GPU-only"，落位于 `crates/prism-gpu/shaders/sim_*.wgsl`。
>
> **目的**：把 CPU 先行（2D SPH / 布料 / 关节 / 破碎）的**数据布局与算法分相**
> 提前钉死，使 GPU 移植成为"同构映射"而非"重写"（D-18：以结构红利用更少代码
> 追平并超越）。本文与 `crates/prism-physics/src/*.rs` 一一对应。

---

## 1. 目标与定位

| 项 | CPU 先行（本批已落） | GPU compute（本文设计） |
|---|---|---|
| 规模 | 千–万粒子（交互调试） | 百万粒子（目标） |
| SPH | 2D 先行（`sph.rs`） | **3D**（结构同构，核常数换 3D 归一化） |
| 求解器 | XPBD 单栈（`xpbd.rs`） | **同一套 XPBD**，线程化粒子并行 |
| 确定性 | 逐位一致（CPU 序） | **同 CPU 逐位**（见 §5 归约纪律） |

**红线不变**：GPU 路径必须满足与 CPU 相同的确定性契约（seek 等价、
宪章 C24 平台中立）——GPU 不是"另一套物理"，是同一 XPBD 的并行执行后端。
**GPU-only 标注**：SPH 3D 与百万级 compute 仅 GPU 可用，CPU 侧不提供等价路径
（诚实边界，非降级项）。

---

## 2. 统一数据布局（SoA）

CPU 先行实现（可变粒子切片 `Particle`）在 GPU 侧映射为**结构体数组 → 数组结构
体（SoA）**四缓冲，与 `xpbd.rs::Particle` 字段一一对应：

```
x[] : vec3<f32>   // 当前预测位置（投影读写）
xp[]: vec3<f32>   // 子步起点（速度重建基准）
v[] : vec3<f32>   // 速度
w[] : f32         // 逆质量（0 = 固定）
```

约束同构映射为**拓扑数组 + 参数数组**（求解序 = 数组序 = id 升序，格锚不变）：

```
c_a[], c_b[] : u32        // 端点粒子下标（距离/铰链/滑槽）
c_rest[], c_comp[] : f32  // 静止量 / 柔度
c_kind[] : u32            // 约束类型标签（分发到内核）
```

SPH 额外两缓冲（每子步重建，见 §4）：`cell_start[]`/`cell_items[]`（空间哈希
的 CSR 形式，桶序按格点坐标排序 → 与 CPU 的有序 Vec 同序）。

---

## 3. XPBD 求解相（三 kernel 对应 `World::step`）

每子步三个 dispatch，与 CPU `World::step` 的 ①②③ 严格同相：

1. **predict**（1 线程/粒子）：`v += g·h; xp = x; x = x + v·h`（w=0 固定粒子
   跳过更新，`xp = x`）。
2. **project**（每迭代 1 dispatch）：按约束数组序并行——**这是唯一需谨慎的相**。
   XPBD 的 Gauss–Seidel 是串行投影（约束间有数据依赖）；GPU 上取
   **Jacobi（同时投影）+ 固定迭代数**。二者收敛到同一不动点（压缩映射），
   固定迭代数保证确定性；CPU 先行与 GPU 的**迭代语义差异**
   记入"已知差异"（§6）。
3. **velocity**（1 线程/粒子）：`v = (x − xp)/h`。

λ（对偶变量）每子步清零，与 CPU 一致；project 内的 λ 更新对 Jacobi 使用
"每迭代独立 λ 缓冲 + 双缓冲"，避免同迭代内读写竞争。

---

## 4. SPH 分相（对应 `sph.rs`，2D→3D 同构）

PBF 密度约束的 GPU 版本 = **两遍 gather/scatter**：

1. **density pass**（1 线程/粒子，gather）：邻域 = 3³ 桶（3D），
   `ρ_i = Σ_j W(|x_i − x_j|)` —— 只读，无竞争。
2. **constraint pass**（1 线程/粒子，scatter，**需原子加**）：`C_i = ρ_i/ρ₀ − 1`，
   投影 `Δx_j += w_j∇C·Δλ`。**原子加的顺序不确定性**是确定性的最大威胁。

**确定性对策（§5）**：scatter 不用浮点原子加，改**两遍分解**：
① 每粒子先把对邻居的贡献写到自己行的 CSR 槽（无竞争写入）；
② 固定形状归约（D-26）按"邻居下标升序"树归约到目标粒子——归约树形状 =
   邻居数的纯函数（`reduction_shape(n)`，prism-core 单点定义），与线程调度无关。

3D 核常数：Poly6 `315/(64πh⁹)`、Spiky 梯度 `−45/(πh⁶)`（2D 常数换 3D 归一化
因子，结构不变）。

---

## 5. 确定性并行纪律（对比 D-26）

| 相 | 竞争 | 对策 |
|---|---|---|
| predict | 无（逐粒子独立） | — |
| project | Jacobi 写入目标粒子 | 双缓冲 λ + 固定迭代数 |
| density | 无（gather） | — |
| constraint scatter | 多粒子写同一 `Δx` | **CSR 分解 + 固定形状树归约**（禁浮点原子加） |
| velocity | 无 | — |

**归约树形状 = n 的纯函数**（D-26）：GPU 的 warp/shared 归约必须
复现 `reduction_shape(n)` 的固定形状，使 GPU 结果与 CPU 串行参考**逐位一致**
（验收：并行 md5 ≡ 串行参考，Phase 1 验收第 6 条同款方法论）。

---

## 6. 已知差异与边界（诚实）

1. **Gauss–Seidel（CPU）vs Jacobi（GPU）**：同不动点、收敛路径不同 → 有限
   迭代下浮点结果**不逐位一致**。取舍：确定性以"同后端内逐位一致"为准（CPU
   与 GPU 各自可复现）；跨后端逐位一致仅在相机制对齐后成立——留待实现轮次
   用"固定形状归约 + 统一迭代语义"收口，**不在本文承诺**（C20：不静默假设）。
2. **3D SPH / 百万级**：GPU-only，CPU 侧无等价路径（标注，非缺陷）。
3. **空间哈希重建**：每子步重建，GPU 用计数排序（counting sort）分桶，桶序
   按格点坐标 → 与 CPU 有序 Vec 同序（§2）。

---

## 7. 落地清单（后续版本，非当前版本）

| 工件 | 内容 | 依赖 |
|---|---|---|
| `prism-gpu/shaders/sim_predict.wgsl` | 预测相 | D1 |
| `prism-gpu/shaders/sim_project.wgsl` | 投影相（Jacobi + 双缓冲 λ） | §3 |
| `prism-gpu/shaders/sim_hash.wgsl` | 空间哈希（counting sort） | §4 |
| `prism-gpu/shaders/sim_density.wgsl` | PBF 密度（gather + CSR 归约） | §4/§5 |
| `prism-gpu/src/sim.rs` | 主机侧装配 + 与 `prism-physics` SoA 互转 | B7/B8 |
| 验收 | 并行 md5 ≡ 串行参考逐位一致；百万粒子 60fps（桌面 GPU 真机） | D-26 |

> 本批交付的是 **CPU 先行实现的正确性锚**（`prism-physics`：关节/布料/SPH2D/
> 破碎/耦合 + 全模拟 seek 等价测试）——GPU 移植以此为判对基准（CpuRef 哲学）。
