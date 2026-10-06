# assets — 自研多语种字体资产（零第三方 crate）

## 资产

| 文件 | 字节 | sha256 |
| --- | --- | --- |
| `prism_multiscript.ttf` | 4224 B | `110e2be38595d39e03070ac9d90c02cf60e25fdd1b0f336ec8ab7f79404c3b34` |

字形总数 **55**；`GSUB` **298 B / 5 特征 / 5 查找**。

`sha256` 由 `tests/multiscript_asset.rs` 锁定（回归锚点）。重跑生成器后若
SHA 变化，测试会转红并要求同步更新断言——这是**故意**的：资产是入库的
确定性产物，任何字节漂移都必须被看见。

## 生成器

`src/multiscript_font.rs`（生成逻辑）+ `src/bin/gen_multiscript_font.rs`（落盘 CLI）。

```bash
cargo run -p prism-text --bin gen_multiscript_font
```

**纯 Rust 手写 SFNT 二进制，零第三方 crate**（不引 harfbuzz / rustybuzz /
ttf-parser / ab_glyph）。`Blob` 写入器与 `assemble_sfnt` / `coverage_fmt1` /
`build_lookup` / `assemble_layout_table` 等助手复用自 `opentype::synth_font`
（测试内的自造字体构造器），**不重复第二份 SFNT 组装逻辑**。

## 内容（诚实清单）

| 书写系统 | 覆盖 | 未覆盖 / 简化 |
| --- | --- | --- |
| 拉丁 | A–Z（U+0041–U+005A，26 个） | 无小写/数字/标点（本资产用途是引擎验证，不是通用字体） |
| 阿拉伯 | **U+0627 ALEF、U+0628 BEH、U+0633 SEEN、U+0644 LAM、U+0645 MEEM**（5 个基字）；BEH/LAM/SEEN/MEEM 各 `isol/fina/init/medi` 四态，ALEF 仅 `fina`（Right 连接型）；`lam-alef` 连字（U+FEFB/FEFC） | 其余阿拉伯字母未覆盖；四态是**几何折线**非排版美术形 |
| 天城文 | U+0915 KA、U+093F I-MATRA、U+094D VIRAMA + `abvm` MarkBase + `KA+VIRAMA` conjunct | 仅 KA 一个基字；reph/rakar/半形等未覆盖 |

字形轮廓是**尺寸两两不同的矩形折线**（真实二次贝塞尔 4 锚点），保证四态包围盒
互异、端到端栅格化后像素可辨——**本资产的价值在「引擎能在真实字体资产上跑通」，
不在字形美观**。

## GSUB / GPOS（真表，非空壳）

- `GSUB`：`isol` / `init` / `medi` / `fina` 四个 **SingleSubst**（各覆盖
  4 个阿拉伯字母）+ `liga` **LigatureSubst**（`LAM+ALEF→连字`、
  `KA+VIRAMA→conjunct`）。特征序 = 施加序 = `isol/init/medi/fina/liga`。
- `GPOS`：`abvm` → **MarkToBasePos**（i-matra 附着到 KA，锚点 (230,-640)）。

对比内嵌 `PrismSansSC-*.ttf` 的 **66 B GSUB 空壳（0 特征 0 查找）**，
本资产是**真表**（见 `tests/multiscript_asset.rs` 的对比断言）。

## 许可

本资产由本项目**原创生成**（无上游字体、无 OFL 约束）。与 `font_data/`
下 OFL 1.1 的 Noto 子集是**两套独立资产**：`font_data/` 提供中文覆盖，
`assets/` 提供阿拉伯/天城文/拉丁的多脚本 GSUB 覆盖。
