# Changelog

本文件遵循兼容性宪章 C10/C15：PATCH 级 bug 修复若造成输出字节变化，
必须逐条进顶部固定栏目《渲染行为变化清单》；每条含
「受影响特性 / 变化原因 / 视觉影响描述 / 是否有开关」四要素。

---

## 渲染行为变化清单

### R-2026-xx-05：修复 Disney BSDF 多散射补偿形式错误（高饱和金属低粗糙度掠射角偏补色）

- **受影响特性**：`prism-rt` 的 Disney BSDF 多散射能量补偿——`crates/prism-rt/src/bsdf.rs`
  的 `Bsdf::eval` 补偿项 `ms`，以及 `energy_dir` / `energy_avg` 的出口。影响一切经
  BSDF 着色的路径：CPU 参考后端光追、Vulkan 光追 kernel 的材质求值，以及
  `metalness→1` 的高饱和金属工作流（金、铜）。
- **变化原因**：旧实现按通道回填
  `ms_ch = (1−E_o)(1−E_i)·f0_ch / (π·(1−E_avg))`，其中 `E` 取自
  `energy_dir(·, ·, f0_ch)`——**每个通道用自己的 F0 查表**。低 F0 通道的 `(1−E)`
  更大、`1−E_avg` 更小，二者相除把回填量在 `f0→0` 侧放大，补偿幅度不随 F0 单调。
  修复改为 Fdez-Agüera 2019 彩色导体形式：幅度取**白炉表**（F0=1）的
  `(1−E_o)(1−E_i)/(π(1−E_avg))`，色相由半球平均 Fresnel
  `F_avg = f0 + (1−f0)/21` 承担；`F_avg=1` 时逐字退化为白炉标准形。
  同时在 `energy_dir` / `energy_avg` 出口加 `clamp(0,1)` 作**防御**——当前 LUT 上
  该 clamp **零触发**（实测金/铜/银在 α∈[0.05,0.35]、μ∈[0.05,0.25] 全域 raw < 1），
  它不是本次修复的主因，主因是补偿形式置换。
- **视觉影响描述**：bug 修复导致**输出字节变化**，仅限彩色金属（尤其高饱和）在
  低粗糙度、掠射角下的多散射补偿项。实测金（f0 = 1.000/0.766/0.336）在
  α=0.05、cosθo=0.25、cosθi=0.25 时，补偿量旧 → 新为
  `[0.019914, 0.046192, 0.050627]` → `[0.019914, 0.012003, 0.002676]`：
  **R 通道逐位不变（比值 1.000）、G 通道旧/新 = 3.85×、B 通道 = 18.92×**；
  更极端的掠射（cosθo=0.1、cosθi=0.8）B 通道达 **49.83×**。即旧版把蓝绿通道
  补了 4–50 倍于正确值的能量，整球偏成补色（金球发青绿）。铜
  （0.955/0.638/0.538）同向，α=0.35、cosθo=0.25、cosθi=0.25 时三通道比
  1.19/3.35/4.52。白金属（f0=1）逐位不变，既有白炉断言与电介质上界断言不受影响。
- **是否有开关**：无。旧行为不保留——旧形式本身是缺陷，不是可选口径。
- **既有 golden 零位移的真相**：本修复未触碰任何 golden 文件，全量回归
  3929 passed / 0 failed。但这不是「无位移」，而是**测试盲区**：现有测试只覆盖
  白金属（f0=1）与彩色电介质（metal=0），从未覆盖高饱和彩色金属。钉子补齐
  归 V2.2-C-4。
- **分发包说明**：`prism-engine-v2.1-source.txt` 打包于本修复之前，**含此缺陷**；
  源码树修复按计划随 v2.2 分发，不单独发 v2.1.1。v2.2 分发说明须显式载明此行。
- **验收**：钉子待 V2.2-C-4 落地（金/铜补偿单调性 + 白炉退化逐位 + 旧形式差异锁定）。

### R-2026-xx-03：T4 结清——sRGB ↔ ACEScg（AP1）矩阵由 Hill 拟合件换为官方原色推导（V2.1 流 C）

- **受影响特性**：`prism-math` 矩阵常量 `colorspace::SRGB_TO_ACESCG` /
  `colorspace::ACESCG_TO_SRGB`（`crates/prism-math/src/colorspace.rs`），
  以及一切消费这两块矩阵的路径：`aces.rs` 的 ACES 风格化调色（`ACESInput`/
  `ACESOutput` 拟合链）、`colorspace_registry` 的 legacy/derived 对拍照、
  经 `TransformChain` 的 `Srgb ↔ AcesCg` 互转。
- **变化原因**：T4 债务结清（作者拍板，四件套）。旧主常量是 Stephen Hill
  拟合件，并非由官方 AP1 原色 + Bradford(D65→D60) 推导，实测偏差
  **3.2045e-2 @ [2][2]**（f32/f64 双口径）。本版按官方原色重录主常量
  （同名换值，先例 R-2026-xx-02），逆阵 `ACESCG_TO_SRGB` 由新主常量经
  f64 精确求逆后收窄到 f32；旧 Hill 值保留为
  `colorspace::SRGB_TO_ACESCG_FITTED_HILL` **显式可选口径**（数值一字未改，
  行和仍精确为 1）。
- **视觉影响描述**：bug 修复导致**输出字节变化**——凡经 sRGB↔AP1 的线性域
  互转结果改变。矩阵逐元素最大位移 3.2e-2 量级（`[2][2]` 0.8378→0.8698，
  `[0][0]` 0.5972→0.6131；逆阵 `[0][0]` 1.7647→1.7051）；ACES 指纹 27 字
  全量重录（非黑样本全部位移）。
- **是否有开关**：无运行时开关。旧口径以 `SRGB_TO_ACESCG_FITTED_HILL`
  常量保留（编译期显式引用），二者**不可混用**。
- **黄金值重录清单（G6 逐条对应）**：

  | # | 测试名 | 旧 → 新 | 性质 |
  |---|--------|---------|------|
  | 1 | `c2_color_golden::aces_golden_file_stable`（指纹常量 `ACES_FINGERPRINT`） | `ACES_FINGERPRINT` 27 字全量重录：18% 灰 `3DD8_403D/3A/3B`→`3DD8_402C/3D/44`；中灰 `3EBF_A557/A556/A556`→`3EBF_A558/A556/A554`；白 `3F1E_7E5A/59/5B`→`3F1E_7E63/59/54`；红 `3F41_0AF6`→`3F3F_786E`；绿 `3D6A_D79F/3F24_5E27`→`3D52_E8EA/3F23_F4F4`；蓝 `3F2B_7813`→`3F28_D13F`；超亮 `3F68_B522/20/21`→`3F68_B53A/21/12`；混合 `3D1A_CC7B/3E5B_0065/3F02_9C41`→`3D1B_8FA0/3E5A_E656/3F01_D3D0` | 真实输出位移 |
  | 2 | `qa_edge_attack_math::aces_fitted_vs_independent_reference` | 测试内独立参考矩阵常量同步重录为新口径 | 测试装置同步（无输出位移） |
  | 3 | `aces::tests::reference_aces_fitted` | 同上 | 测试装置同步（无输出位移） |
  | 4 | `colorspace::tests::legacy_srgb_acescg_matrices_debt_t4_is_named` | 结清锁更名 `..._settled_t4_is_named`：判定由「偏差必须 > 1e-2（债务存在）」翻转为「偏差 ≤ 1e-2（已收敛）+ 逆阵可重算 + Hill 旧值未改」 | 债务锁 → 结清锁 |
  | 5 | `colorspace_registry::tests::legacy_constant_is_a_fitted_matrix_not_a_primaries_derivation` | 取值对象由 `SRGB_TO_ACESCG` 改为 `SRGB_TO_ACESCG_FITTED_HILL`，仍锁 3.2e-2 差异 | 参照物随结清换位 |
  | 6 | `colorspace_registry::tests::srgb_to_acescg_fitted_hill_alias_is_bit_identical_to_the_legacy_name` | **删除**（别名同值语义随结清失效），由 `..._preserved_and_diverged_from_recut_main` 接替（锁旧值一字未改 + 二者已分流） | 断言语义翻转 |

验收：`cargo test -p prism-math --release` 全绿；`legacy_srgb_acescg_matrices_settled_t4_is_named`
每次运行重算复核（官方公布矩阵对齐 < 1e-6、主常量收敛 ≤ 1e-2、逆阵 5e-7）。

### R-2026-xx-04：修复 BakedSdf 查询的 brick 边缘插值缺陷（查询点被吸附到格点）

- **受影响特性**：`prism-geometry` 的 `BakedSdf::dist`（烘焙 SDF 的任意点查询，
  `crates/prism-geometry/src/mesh_sdf.rs`），影响一切消费烘焙 SDF 的几何/渲染
  路径（SDF 图元求值、mesh→SDF 距离查询）。
- **变化原因**：查询把 **brick 局部**坐标 clamp 到 `BRICK−2`、权重 clamp 到 1——
  单元落在 brick 末格（局部 `[7,8)`）时查询点被吸附到最近格点，插值不是在被查
  点上求值；brick 无 apron，跨 brick 单元完全无插值。1-Lipschitz 场下定位偏差
  直接变成距离误差，实测最坏 **|SDF| = 0.0693（h=0.05）@ 真实表面点**
  （误差恰等于 `|p−q|`，q 为被吸附后的求值点）。
- **视觉影响描述**：bug 修复导致**几何查询结果变化**——brick 边界邻域的 SDF
  值/等值面位置修正，最坏 6.9e-2（h=0.05）量级的定位误差消除（修复后同点
  1.4e-3）。既有 golden 无位移（本版全仓测试证实），但 SDF 精度行为已变化。
- **是否有开关**：无。
- **验收**：`mesh_sdf::tests::r3_hausdorff_bound_holds_and_tightens_with_resolution`
  （新增 `≤ √3/2·h` 更严断言 + 实测值表）+
  `mesh_sdf::tests::brick_edge_cells_resolve_global_lattice_not_clamped_local`
  （专挑 ≥2 轴同时落末格的 177 个表面点：旧实现最坏 0.06929246 超阈，新实现
  1.4464e-3，余量 ~30×）。

### R-2026-xx-02：修复 ACEScg（AP1）→ ACES2065-1（AP0）色彩矩阵（按官方 S-2014-004 重录 + 精确逆阵重算）

- **受影响特性**：`prism-math` 矩阵常量 `colorspace::ACESCG_TO_ACES2065` /
  `colorspace::ACES2065_TO_ACESCG`（`crates/prism-math/src/colorspace.rs`），
  以及一切经 `TransformChain` 的 `AcesCg ↔ Aces2065` 线性域互转
  （渲染侧可达 postfx color 相关路径，如 `prism-postfx` `ColorManage`
  经 `effects::color` 的 `apply_color` 消费这两块矩阵）。
- **变化原因**：旧轨常量与 ACES 官方 S-2014-004 矩阵**坐实不符**——
  第三行整体抄错（旧 `[0.0283, 0.1292, 0.8593]` vs 官方
  `[-0.0055, 0.0040, 1.0015]`，逐元素 max ≈ 1.42e-1），第一二行另有
  4 位小数级偏差（~3e-4 ~ 4e-3）；两常量互为精确逆，属「自洽地错了
  一对」。本修复按 S-2014-004 官方全矩阵**重录三行**（逐位取官方值），
  逆矩阵由官方矩阵经 f64 求逆（`colorspace_registry::invert3`）**精确
  重算**（非照抄旧值），并由测试 `aces2065_to_acescg_is_recomputed_inverse`
  每次运行重算复核。
- **视觉影响描述**：bug 修复导致**输出字节变化**——凡经 AP1↔AP0 转换的
  结果改变（旧第三行错误使 AP0 输出的绿/蓝分量偏差达 ~1e-1 量级）。
  修复后旧轨直接矩阵与新轨 `colorspace_registry`（原色推导）一致性实测：
  正向 worst ≈ 1.9e-6、逆向 worst ≈ 2.9e-6（预期 1e-5 内，测试
  `aces_ap1_ap0_matches_registry_derivation` 锁定，不调容差）。
  白点守恒：AP1/AP0 同为 D60 名义白点，官方矩阵行和恰为 1.0（实测偏离
  < 1e-7），白点仍映射白点（`white_point_is_preserved` 容差由 1e-4
  加严至 1e-6）。
- **是否有开关**：无开关。属 bug 修复（宪章 C10：PATCH 修 bug——本修复
  导致输出字节变化，故按 C15 逐条进本清单）。重渲兜底：引擎为静态
  二进制，旧版本可并存安装（C10 逃生门），可用旧版本重渲既有基线。

验收：`crates/prism-math/src/colorspace.rs` 单测（互逆对
`matrices_are_inverse_pairs` / 白点守恒加严 / 逆阵重算验证 / 新旧轨
一致性对拍）。

### R-2026-xx-01：`MergeOp::Over` 半透明层叠半透明层改用标准 Porter-Duff 预乘混合

- **受影响特性**：`prism-core` 合成算子 `MergeOp::Over`
  （`compose::blend_pixel` / `compose::merge` / `ComposeNode` 全部求值路径），
  以及渲染侧经 `nodes::ComposeMergeNode`、`nodes::DepthCompositeNode` 的
  over 合成。
- **变化原因**：修复 over 的两处偏离标准 Porter-Duff 的缺陷——缺 αa
  权重（下层色按 `1` 而非 `αa` 加权）与缺归一化（输出未除以合成 alpha）。
  修复按规格《值域层-预乘混合规格》定义 2.1–2.3 在预乘混合域
  （`C = α·c`）实施；对外 `LayerBuf` 仍为直通 RGBA，接口与存储格式不变。
- **视觉影响描述**：仅**半透明层叠半透明层**（`αa < 1` 且 `αb < 1`）的
  象限输出改变（修复前该象限结果非标准 Porter-Duff；嵌套越深误差复利
  越深，实测 legacy 两层即 ~2e-2）。以下象限**逐位不变**：
  上层不透明（`αb = 1` ⟹ 输出 = 上层，最常见用例）、双不透明
  （⟹ 上层）、下层不透明 + 上层半透明、任一层全透明（⟹ 另一层，
  幂二 alpha 下逐位）。其余算子（plus/screen/multiply/mask）公式零
  改动，黄金值不变。修复后 over 满足结合律（规格命题 2.4），误差结构
  从复利 O(nε) 退化为单层 O(ε)（注 2.4.1），覆盖类子树可走
  `eval_flat` 扁平化路径。
- **是否有开关**：无开关。属 bug 修复（宪章 C10：PATCH 修 bug 不改
  行为——本修复导致输出字节变化，故按 C15 逐条进本清单；影响面为
  「半透明叠半透明」单一象限，维持 PATCH 级）。重渲兜底：引擎为静态
  二进制，旧版本可并存安装（C10 逃生门），可用旧版本重渲既有基线。

验收：`crates/prism-core/tests/qa_over_premultiplied.rs`（规格 §5 六条
全量断言，含 f32 逐位结合不可达的如实记录）+
`crates/prism-core/src/compose.rs` 单测（新语义钉子 4 枚）。

## 能力扩容清单（非 C15：特性扩容——此前被拒场景现在可渲染）

### F-2026-xx-02：V2.1 能力扩容（Vulkan 光追实帧 / USD R1·R3 判据 / 签名闭包 S3 / 异质体积 / Metal 门控）

- **扩容内容**：
  1. **Vulkan 光追管线真实化**（#11）：`render_scene` 由「恒返回空帧、调用方
     回退 CPU」升级为 **lavapipe 实机真实出图**（自研发射器 `vulkan/spirv_rt.rs`
     光追 megakernel + `vulkan/scene.rs` dispatch 编排）。实测 64×48 / 113×47
     与 `CpuRefBackend` **max_channel_delta=0、differing_pixels=0（逐字节
     一致）**，GPU 双跑逐位一致；16×16 空场景全帧背景色 256/256 像素一致。
     `is_available()` 升级为 **compute ∧ render 双自检**（可用却出空帧的组合
     被测试排除）。Metal 同构移植（设备无关部分 + MSL kernel 源）+ 编译门控，
     `is_available()` 保持 false（**未实机验证、不点亮**，诚实留名）；WebGPU
     可行性结论见 `docs/v2.1-WebGPU可行性评估.md`。
  2. **USD 资产 R1/R3 判据**（#9）：新增「同构保持」（表示签名等价）与
     「保真度界」（Hausdorff `≤ √3/2·h`）的形式化与测试；R2 字节往返不动点
     不回归。
  3. **签名闭包 S3 收尾**（#8）：`cache_key_with_mode` —— 迭代节点键含
     `(ε,n)`（经 ChannelDef 0x5333 编码）、流节点键含 frame、S4 逐出中性；
     `mode=None` 时逐字节退化为旧 `cache_key`（零回归）。
  4. **异质体积**（#10）：`HeterogeneousMedium`（Woodcock delta tracking，
     σt 空间变化）+ 格心采样口径修正；σt 恒定场与均匀实现**逐位一致**
     （退化一致性）。**已接入渲染链路**（QA-F2 结清）：场景介质盒统一为
     `enum Medium { Homogeneous, Heterogeneous }`，integrator 与 wavefront
     双路径均可消费；均匀路径的采样流消耗与像素输出**逐字节不变**（单均匀
     分支），异质路径按需多抽一个接受均匀。无偏性口径如实分述：`shadow_transmittance_tracked`
     （调用方喂独立均匀）300k 样本内对 Beer 定律偏差 < 1%；生产入口
     `shadow_transmittance`（2-uniform 签名）的空碰撞派生链已由乘法链改为
     splitmix64 去相关（QA-F1：旧链实测 +1.08% 正偏），修复后由测试
     `shadow_transmittance_production_entry_unbiased_vs_beer` 钉在 2% 内。
- **为何非 C15**：全部为新增能力；既有可渲染工程输出逐字节不变（GPU 出帧与
  CPU 参考 0 差异；新增签名键只在带模式调用时生效）。R3 修复本身另有
  C15 条目（R-2026-xx-04）。
- **是否有开关**：无（新增能力，不改变既有路径行为）。
- **验收**：`prism-gpu/tests/qa_render_scene.rs`、`prism-io/tests/qa_r1_representation_signature.rs`、
  `prism-core/tests/qa_signature_s3_closure.rs`、`prism-render/tests/qa_incremental_s3.rs`、
  `prism-rt/tests/qa_heterogeneous_volume.rs` +
  `integrator::tests::heterogeneous_medium_render_path_is_reachable_and_matches_beer`
  （渲染路径可达性 + 整画面 Beer + 两遍逐位确定性）。

### F-2026-xx-01：movie 成片路径新增 sim（S3 约束梯度流，cloth）节点渲染支持

- **扩容内容**：`prism-render` movie 路径（`render_movie` /
  `render_movie_frame`，`crates/prism-render/src/movie.rs`）新增
  `StructureId::ConstraintFlow`（sim，当前种类仅 cloth）消费：自包含
  XPBD 求解器（半隐式欧拉 + 距离约束投影，数值纪律逐位对齐
  `prism-physics` 的 xpbd.rs）+ 帧循环驱动（`SimTimeline` 跨帧快照库）。
  此前含 sim 节点的场景在 `classify_node` 按 `R5010`「sim 暂不支持」
  显式拒绝；现在该类场景可渲染（粒子叠加为暖橙圆点）。`R5010` 对
  particles / compose / 未知结构的拒绝不变——仅 sim 一族解禁。
- **为何非 C15**：C15 只登记「bug 修复导致的输出字节变化」；本条是
  能力扩容——此前直接报错的场景现在成功出图，既有可渲染工程的输出
  逐字节不变（无字节变化可登记，故不进《渲染行为变化清单》。
- **确定性口径**：模式元数据（substeps / iterations / residual_eps）
  经通道定义（tag = 0x5333）哈希进帧签名——改迭代参数 ⇒ 签名变 ⇒
  快照/缓存不误复用；同参数两遍渲染逐字节一致；快照装配跳帧 ≡
  顺序推进；冷 seek（无前驱快照）按 `P3012` 结构化报错，绝不静默
  fallback 到初态。
- **是否有开关**：无开关（新增能力，不改变既有路径行为）。

验收：`crates/prism-render/tests/movie_sim_timeline.rs`（帧循环出图 /
确定性 / seek 等价 / 冷 seek P3012 / 模式元数据进签名与缓存键）+
`crates/prism-render/src/movie.rs` 单测。
