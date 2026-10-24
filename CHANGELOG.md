# Changelog

本文件遵循兼容性宪章 C10/C15：PATCH 级 bug 修复若造成输出字节变化，
必须逐条进顶部固定栏目《渲染行为变化清单》；每条含
「受影响特性 / 变化原因 / 视觉影响描述 / 是否有开关」四要素。

---

## 渲染行为变化清单

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
