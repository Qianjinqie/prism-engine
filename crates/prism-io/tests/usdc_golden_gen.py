#!/usr/bin/env python3
"""V2.5-B2 golden 夹具生成器（**开发期工具，不是 workspace 依赖**）。

与 `jpeg_golden_gen.py` / `png_golden_gen.py` 同性质：只在开发机上跑一次，
把产物字节提交入库；`cargo test` 只读这些字节，不调用 Python、不联网。

权威来源：Pixar OpenUSD（`usd-core`，Python `pxr.Sdf` / `pxr.UsdUtils`）。
`cube.usdc` 与 `cube.usdz` 内的根层均由**真实 USD 写出器**生成，不是本仓
解析器自己写的字节——避免「两端同错」的自洽对拍（路线图 §1.4）。

运行：

    pip install usd-core
    python3 crates/prism-io/tests/usdc_golden_gen.py

产物（同目录 `fixtures/usdc/`）：

| 文件 | 是什么 | 判据用途 |
|---|---|---|
| `cube.usda` | 源 USDA 文本 | V1 语义对拍的文本侧 |
| `cube.usdc` | `Sdf.Layer.Export("*.usdc")` 真二进制 crate | V1/V2/V3/V4 主夹具 |
| `cube.usdz` | `UsdUtils.CreateNewUsdzPackage(cube.usdc)`，根层 stored | V2 USDZ ↔ 裸 usdc 逐位 |
| `cube_deflate.usdz` | 同一根层改为 zip method=8（deflate） | V2 deflate 解包路径（复用 `prism-bytes::inflate_raw`） |
| `cube.usdc.fields.txt` | 用 `pxr.Sdf` 读回的逐 spec / 逐 field 真值 | V1 golden 逐字段对拍 |

USDZ 规范：首个条目是根层且为 stored（method 0），本生成器由 `UsdUtils`
保证；deflate 变体是额外的 zip 路径覆盖，不作为 USDZ 规范合规声明。
"""

from __future__ import annotations

import os
import zipfile

from pxr import Sdf, UsdUtils

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "usdc")

USDA = r'''#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "root"
{
    double3 xformOp:translate = (0, 1, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "cube"
    {
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        rel material:binding = </root/mat>
    }
    def Material "mat"
    {
    }
}
'''


def fmt_value(value) -> str:
    """把 pxr 值压成稳定单行文本（golden 只比语义，不比 Python repr）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if hasattr(value, "__len__") and not isinstance(value, str):
        return "[" + ", ".join(fmt_value(v) for v in value) + "]"
    return str(value)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    usda = os.path.join(OUT, "cube.usda")
    usdc = os.path.join(OUT, "cube.usdc")
    usdz = os.path.join(OUT, "cube.usdz")
    deflate = os.path.join(OUT, "cube_deflate.usdz")
    fields = os.path.join(OUT, "cube.usdc.fields.txt")

    with open(usda, "w", encoding="utf-8") as fp:
        fp.write(USDA)

    layer = Sdf.Layer.FindOrOpen(usda)
    if layer is None or not layer.Export(usdc):
        raise RuntimeError("Sdf.Layer.Export(.usdc) 失败")
    if not UsdUtils.CreateNewUsdzPackage(usdc, usdz):
        raise RuntimeError("UsdUtils.CreateNewUsdzPackage 失败")

    raw = open(usdc, "rb").read()
    with zipfile.ZipFile(usdz, "r") as z:
        root = z.infolist()[0]
        if z.read(root.filename) != raw:
            raise RuntimeError("USDZ 根层与裸 .usdc 不逐位相同")
    with zipfile.ZipFile(deflate, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cube.usdc", raw)

    # 用真实 USD **读回** .usdc，逐 spec / 逐 field 落 golden。
    back = Sdf.Layer.FindOrOpen(usdc)
    lines: list[str] = []
    layer_obj = back.pseudoRoot
    for key in sorted(layer_obj.ListInfoKeys()):
        lines.append(f"LAYER {key} = {fmt_value(layer_obj.GetInfo(key))}")

    def walk(prim) -> None:
        path = str(prim.path)
        lines.append(f"PRIM {path} specifier={prim.specifier} typeName={prim.typeName}")
        for attr in prim.attributes:
            lines.append(
                f"ATTR {attr.name} type={attr.typeName} custom={fmt_value(attr.custom)} "
                f"uniform={fmt_value(attr.variability == Sdf.VariabilityUniform)} "
                f"default={fmt_value(attr.default)}"
            )
        for rel in prim.relationships:
            targets = list(rel.targetPathList.GetAddedOrExplicitItems())
            lines.append(f"REL {rel.name} targets={fmt_value([str(t) for t in targets])}")
        for child in prim.nameChildren:
            walk(child)

    for root_prim in back.rootPrims:
        walk(root_prim)
    with open(fields, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"generated: {usdc} ({len(raw)} B), {usdz}, {deflate}, {fields}")


if __name__ == "__main__":
    main()
