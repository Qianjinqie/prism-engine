# MODIFICATIONS — Prism Sans SC（OFL 1.1 修改版字体）

本文件为 SIL Open Font License, Version 1.1（见同目录 `LICENSE`）第 1 条
要求的**修改说明**，同时记录资产来源与合规链。

## 原始字体

- **Noto Sans SC** Regular / Bold（SIL Open Font License, Version 1.1）
- 上游：github.com/notofonts/noto-cjk（Sans/OTF/SimplifiedChinese）
- 获取方式：`gh-proxy.com` GitHub 加速镜像直连（见 `README.md` 记录）
- 原始文件 sha256：
  - Regular `faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9`（8,331,336 B）
  - Bold    `c6cb5a93abaa9edc8ee7463b7ebb7f42d618d40e6ed2f7a5371c97b0b64767c0`（8,543,168 B）

## 修改内容（构建期完成；fontTools/cu2qu 仅构建期使用，运行时零依赖）

1. **子集化**（pyftsubset）：字符集 = ASCII 可打印（95 码位）+ 中文常用
   标点 + GB2312 全集汉字（6763 码位），合计 6888 码位；
2. **CFF → TrueType 轮廓转换**（cu2qu，三次→二次贝塞尔，
   `max_err=1.0`/upem=1000，`reverse_direction=True`）：原始 Noto Sans SC
   为 CFF（OTF），本引擎自研解析器只吃 `glyf` 二次轮廓；转换后补齐
   maxp 1.0 版字段（maxPoints/maxContours 按实际轮廓统计）；
3. **改名**：family `Noto Sans SC` → `Prism Sans SC`（OFL 的 RFN 条款
   ——保留字体原名 Noto 不得用于衍生版命名——故衍生版避开原名）；
   `nameID 0`（版权声明）、`nameID 13`（许可说明）、`nameID 14`
   （许可 URL）按 OFL 要求保留并指向本许可；
4. **post 表 format 3.0**（去除字形名与 hint，缩减体积）。

## 产物

- `PrismSansSC-Regular.ttf` 1,948,392 B
  sha256 `e971e63b5376e9179a114ef4ba679850b27bce468c368f1b7fd6b01c4f2d5ae0`
- `PrismSansSC-Bold.ttf` 1,919,124 B
  sha256 `b9b6396d61365f191a59a9a54b65d9ff150c26c1e947d36b6dbc622c42a67010`

## 分发许可

本修改版字体以 **SIL Open Font License, Version 1.1** 同许可分发；
`LICENSE` 文件为许可全文（未修改）。
