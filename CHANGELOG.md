# Changelog

本文件遵循兼容性宪章 C10/C15：PATCH 级 bug 修复若造成输出字节变化，
必须逐条进顶部固定栏目《渲染行为变化清单》；每条含
「受影响特性 / 变化原因 / 视觉影响描述 / 是否有开关」四要素。

---

## 渲染行为变化清单

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
