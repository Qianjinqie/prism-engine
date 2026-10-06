# font_data — 内嵌字体资产（OFL 1.1）

## 资产

| 文件 | 体积 | sha256 |
| --- | --- | --- |
| `PrismSansSC-Regular.ttf` | 1,948,392 B | `e971e63b5376e9179a114ef4ba679850b27bce468c368f1b7fd6b01c4f2d5ae0` |
| `PrismSansSC-Bold.ttf` | 1,919,124 B | `b9b6396d61365f191a59a9a54b65d9ff150c26c1e947d36b6dbc622c42a67010` |

合计 3.87MB（预算 ≤6MB；单档 1–3MB 达标）。

## 来源链（D-31 资产自研化总账）

1. 上游：`github.com/notofonts/noto-cjk` → `Sans/OTF/SimplifiedChinese/`
   （Noto Sans SC Regular/Bold，SIL OFL 1.1）；
2. 镜像探测顺序 `gh-proxy.com` → `ghfast.top` → jsdelivr（任务书 a/b/c），
   实际命中：**gh-proxy.com**（首次探测即通）；
3. 原始 OTF sha256：
   - Regular `faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9`（8,331,336 B）
   - Bold    `c6cb5a93abaa9edc8ee7463b7ebb7f42d618d40e6ed2f7a5371c97b0b64767c0`（8,543,168 B）
4. 构建期处理（Python + fontTools，脚本留存于构建环境，非运行时）：
   子集化（ASCII 95 + 中文标点 + GB2312 全集 6763 = 6888 码位）→
   CFF→glyf 转换（cu2qu）→ 改名 Prism Sans SC → post 3.0；
   详见 `MODIFICATIONS.md`。

## OFL 1.1 合规五条（任务书 §6.1 数据资产清单 #2）

1. **许可附带**：`LICENSE` 为 OFL 1.1 全文，随字体与 crate 分发；
2. **Reserved Font Name**：原名 `Noto` 为 RFN，衍生版改名 **Prism Sans SC**
   避开（nameID 1/4/6 同步改名；nameID 0/13/14 保留版权与许可指向）；
3. **不单独售卖**：字体仅作为 prism-text crate 之内嵌资产分发，不单独
   出售（OFL 本身允许收费捆绑，本项目亦不单独收费）；
4. **不将原名用于推广**：项目/产品宣传不使用 `Noto` 名义；
5. **同许可分发**：修改版以 OFL 1.1 原样分发（`MODIFICATIONS.md` 记录
   全部修改——子集化/格式转换/改名，属许可允许的修改范围）。

## 内嵌方式

`src/lib.rs` 以 `include_bytes!("../font_data/PrismSansSC-*.ttf")` 直接
内嵌字节流——**禁系统字体降级**，资产即代码，运行时零外部依赖、
零网络请求、零文件系统查找。

## 构建参数备查

- 字符集：ASCII 可打印（U+0020–U+007E）+ 中文常用标点
  （，。、；：？！“”‘’（）《》〈〉【】「」『』…—·～等）+ GB2312 全集
  汉字（`bytes([0xB0..0xF7, 0xA1..0xFE]).decode("gb2312")` 程序化生成）；
- cu2qu：`max_err=1.0`（upem=1000）、`reverse_direction=True`（otf2ttf
  标准配方）；
- maxp 版本 0.5 → 1.0（补 maxZones 等 1.0 版字段，maxPoints/maxContours
  遍历 glyf 实际统计）。
