#!/usr/bin/env python3
"""PNG 黄金固件生成器（独立参考实现，与引擎实现零共用代码）。

独立性说明
-----------
本脚本刻意**不**调用任何引擎代码路径，也不借助 Pillow 做解码：

* 编码侧：仅用 ``zlib.compress`` 产真实 DEFLATE 码流 + 手工拼 chunk
  （长度/类型/数据/CRC32，CRC 用 ``zlib.crc32``）；
* 解码侧：内置独立参考解码器（zlib 解压 + 逐行反滤波），Pillow 仅作
  **交叉校验**（校验参考解码器自身，不作为黄金值来源）。

产物（写入 ``fixtures/png/``）
-----------------------------
* ``<name>.png``          固件文件（非交错、8/16 位、滤波 0..4 全覆盖）
* ``<name>.samples.bin``  参考解码器输出的**原始样本流**（反滤波之后、
                          未做任何色彩解释；16 位为大端、调色板为索引），
                          即 Rust 解码器 ``samples`` 字段的逐字节基准
* ``<name>.rgba.bin``     RGBA8 期望值（引擎 ``to_rgba8`` 的基准）：
                          8 位固件取 Pillow ``convert("RGBA")``；
                          16 位固件为「每通道取高字节」截断镜像
                          （已用 Pillow 对 RGB16/GA16/RGBA16 逐字节验证；
                          gray16 因 Pillow 的 ``I;16`` 直转 RGBA 产出垃圾值，
                          改为自算镜像，manifest 里如实标注）
* ``manifest.json``       全部固件的元数据 + 摘要（无内部词汇）

确定性：样本由 splitmix64（常量与 Rust 测试镜像完全一致）生成，
同一 seed 永远产出同一固件；重跑脚本只会覆盖出逐字节相同的文件。

用法::

    python3 crates/prism-io/tests/png_golden_gen.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import sys
import zlib

from PIL import Image

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "png")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# splitmix64 常量（与 Rust 侧测试镜像逐位一致，保证两侧输入同源）。
SPLIT_GAMMA = 0x9E3779B97F4A7C15
SPLIT_MIX1 = 0xBF58476D1CE4E5B9
SPLIT_MIX2 = 0x94D049BB133111EB
U64_MASK = (1 << 64) - 1

# 颜色类型 → 通道数（RFC 2083 表 11）。
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

COLOR_TYPE_NAMES = {
    0: "grayscale",
    2: "truecolor",
    3: "palette",
    4: "grayscale+alpha",
    6: "truecolor+alpha",
}


class SplitMix64:
    """splitmix64 确定性随机源（Rust 侧同款常量的 Python 镜像）。"""

    def __init__(self, seed: int) -> None:
        self.state = seed & U64_MASK

    def next_u64(self) -> int:
        self.state = (self.state + SPLIT_GAMMA) & U64_MASK
        z = self.state
        z = ((z ^ (z >> 30)) * SPLIT_MIX1) & U64_MASK
        z = ((z ^ (z >> 27)) * SPLIT_MIX2) & U64_MASK
        return (z ^ (z >> 31)) & U64_MASK

    def next_u8(self) -> int:
        return self.next_u64() & 0xFF


# ---------------------------------------------------------------------------
# 独立参考解码器（chunk 解析 + CRC 校验 + zlib 解压 + 逐行反滤波）
# ---------------------------------------------------------------------------


def paeth_predictor(a: int, b: int, c: int) -> int:
    """Paeth 预测子（RFC 2083 §6.6）：平局优先 a，其次 b，最后 c。"""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def unfilter_rows(raw: bytes, width: int, height: int, bpp: int) -> bytes:
    """把「每行 1 字节滤波类型 + 行数据」的码流反滤波为原始样本流。"""
    stride = width * bpp
    expected = (stride + 1) * height
    if len(raw) != expected:
        raise ValueError(f"解压后长度 {len(raw)} != 期望 {expected}")
    out = bytearray()
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        ftype = raw[pos]
        pos += 1
        row = bytearray(raw[pos:pos + stride])
        pos += stride
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            x = row[i]
            if ftype == 0:  # None
                v = x
            elif ftype == 1:  # Sub
                v = x + a
            elif ftype == 2:  # Up
                v = x + b
            elif ftype == 3:  # Average：向下取整
                v = x + ((a + b) >> 1)
            elif ftype == 4:  # Paeth
                v = x + paeth_predictor(a, b, c)
            else:
                raise ValueError(f"未知滤波类型 {ftype}")
            row[i] = v & 0xFF
        out += row
        prev = row
    return bytes(out)


def decode_png_reference(data: bytes) -> dict:
    """独立参考 PNG 解码器：返回 IHDR 字段 + 原始样本流 + PLTE/tRNS。

    校验项：签名、每个 chunk 的 CRC32、IHDR 必现且唯一、IDAT 非空、
    颜色类型与位深合法、非交错。**不**做色彩解释（解释留给引擎测试侧）。
    """
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("PNG 签名不符")
    pos = 8
    ihdr = None
    plte = None
    trns = None
    idat = bytearray()
    seen_iend = False
    chunk_log = []
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc_stored,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        crc_calc = zlib.crc32(ctype + body) & 0xFFFFFFFF
        if crc_stored != crc_calc:
            raise ValueError(f"chunk {ctype!r} CRC 不符：存 {crc_stored:#010x} != 算 {crc_calc:#010x}")
        pos += 12 + length
        chunk_log.append(ctype.decode("ascii"))
        if ctype == b"IHDR":
            if ihdr is not None:
                raise ValueError("IHDR 重复出现")
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"PLTE":
            plte = bytes(body)
        elif ctype == b"tRNS":
            trns = bytes(body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            seen_iend = True
            break
    if ihdr is None:
        raise ValueError("缺少 IHDR")
    if not seen_iend:
        raise ValueError("缺少 IEND")
    if not idat:
        raise ValueError("缺少 IDAT 图像数据")
    width, height, depth, ctype_id, comp, filt, interlace = ihdr
    if comp != 0 or filt != 0 or interlace != 0:
        raise ValueError(f"压缩/滤波/交错方法须为 0（实际 {comp}/{filt}/{interlace}）")
    if ctype_id not in CHANNELS:
        raise ValueError(f"颜色类型 {ctype_id} 非法")
    if depth not in (8, 16):
        raise ValueError(f"位深 {depth} 不在 8/16 之内")
    if width == 0 or height == 0:
        raise ValueError("宽或高为 0")
    bpp = CHANNELS[ctype_id] * depth // 8
    raw = zlib.decompress(bytes(idat))  # 标准 zlib：解压 + 校验 Adler-32
    samples = unfilter_rows(raw, width, height, bpp)
    return {
        "width": width,
        "height": height,
        "bit_depth": depth,
        "color_type": ctype_id,
        "channels": CHANNELS[ctype_id],
        "bpp": bpp,
        "samples": samples,
        "plte": plte,
        "trns": trns,
        "chunk_log": chunk_log,
    }


# ---------------------------------------------------------------------------
# 编码侧（自写：每行可选滤波 0..4 + zlib 真实压缩 + 手工 chunk）
# ---------------------------------------------------------------------------


def filter_row(raw_row: bytes, prev_row, bpp: int, ftype: int) -> bytes:
    """按 RFC 2083 §6 对单行做滤波（逆运算即反滤波）。"""
    n = len(raw_row)
    out = bytearray(n)
    for i in range(n):
        a = raw_row[i - bpp] if i >= bpp else 0
        b = prev_row[i] if prev_row is not None else 0
        c = prev_row[i - bpp] if (prev_row is not None and i >= bpp) else 0
        x = raw_row[i]
        if ftype == 0:
            v = x
        elif ftype == 1:
            v = x - a
        elif ftype == 2:
            v = x - b
        elif ftype == 3:
            v = x - ((a + b) >> 1)
        elif ftype == 4:
            v = x - paeth_predictor(a, b, c)
        else:
            raise ValueError(f"未知滤波类型 {ftype}")
        out[i] = v & 0xFF
    return bytes(out)


def make_chunk(ctype: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + ctype
        + payload
        + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
    )


def encode_png(
    width: int,
    height: int,
    depth: int,
    color_type: int,
    samples: bytes,
    filters,
    plte: bytes | None = None,
    trns: bytes | None = None,
    extra_chunks=(),
    idat_splits: int = 1,
) -> bytes:
    """把原始样本流编码为 PNG 字节流。

    ``filters`` 为长度 1（全程同一滤波）或长度 = height 的列表（逐行指定），
    用于覆盖全部 5 种滤波类型（Pillow 只产 None，故必须自编码）。
    """
    channels = CHANNELS[color_type]
    bpp = channels * depth // 8
    stride = width * bpp
    if len(samples) != stride * height:
        raise ValueError(f"样本长度 {len(samples)} != {stride * height}")
    stream = bytearray()
    prev = None
    for y in range(height):
        row = samples[y * stride:(y + 1) * stride]
        ftype = filters[y % len(filters)] if isinstance(filters, (list, tuple)) else filters
        stream.append(ftype)
        stream += filter_row(row, prev, bpp, ftype)
        prev = row
    compressed = zlib.compress(bytes(stream), 9)
    out = bytearray(PNG_SIGNATURE)
    out += make_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0))
    if plte is not None:
        out += make_chunk(b"PLTE", plte)
    if trns is not None:
        out += make_chunk(b"tRNS", trns)
    for ctype, payload in extra_chunks:
        out += make_chunk(ctype, payload)
    step = max(1, (len(compressed) + idat_splits - 1) // idat_splits)
    pos = 0
    while pos < len(compressed):
        out += make_chunk(b"IDAT", compressed[pos:pos + step])
        pos += step
    out += make_chunk(b"IEND", b"")
    return bytes(out)


# ---------------------------------------------------------------------------
# 样本生成（确定性：splitmix64 + 结构化梯度）
# ---------------------------------------------------------------------------


def make_samples(width: int, height: int, depth: int, color_type: int, seed: int) -> bytes:
    """生成确定性样本流：平滑梯度（让滤波真正起作用）+ 少量噪声。"""
    channels = CHANNELS[color_type]
    rng = SplitMix64(seed)
    out = bytearray()
    for y in range(height):
        for x in range(width):
            for ch in range(channels):
                gx = (x * 255) // max(1, width - 1)
                gy = (y * 255) // max(1, height - 1)
                base = (gx * 3 + gy * 5 + ch * 29) & 0xFF
                noise = rng.next_u8() >> 4  # 0..15 小幅扰动
                v = (base + noise) & 0xFF
                if depth == 8:
                    out.append(v)
                else:
                    lo = rng.next_u8()
                    out.append(v)
                    out.append(lo)
    return bytes(out)


def make_palette(seed: int, count: int = 16) -> tuple[bytes, bytes]:
    """确定性调色板（RGB 三元组）+ 部分条目透明的 tRNS。"""
    rng = SplitMix64(seed ^ 0x5A5A_5A5A)
    plte = bytearray()
    for _ in range(count):
        plte += bytes([rng.next_u8(), rng.next_u8(), rng.next_u8()])
    # 索引 0/3/7 全透明，其余不透明（Pillow 对 P→RGBA 亦应用该语义）。
    trns = bytearray(count)
    for i in (0, 3, 7):
        trns[i] = 0
    return bytes(plte), bytes(trns)


def make_indices(width: int, height: int, seed: int, count: int = 16) -> bytes:
    """确定性调色板索引流（只取前 count 个条目，保证有透明命中）。"""
    rng = SplitMix64(seed)
    return bytes(rng.next_u8() % count for _ in range(width * height))


# ---------------------------------------------------------------------------
# RGBA 期望值（8 位：Pillow；16 位：高位截断镜像）
# ---------------------------------------------------------------------------


def rgba_from_pillow(png_bytes: bytes) -> tuple[bytes, str]:
    im = Image.open(io.BytesIO(png_bytes))
    return im.convert("RGBA").tobytes(), im.mode


def rgba_mirror_16bit(samples: bytes, color_type: int, channels: int) -> bytes:
    """16 位样本 → RGBA8 镜像规则：每通道取高字节，无 alpha 补 255。"""
    out = bytearray()
    px = len(samples) // (channels * 2)
    for i in range(px):
        base = i * channels * 2
        if color_type == 0:  # 灰度
            g = samples[base]
            out += bytes([g, g, g, 255])
        elif color_type == 2:  # RGB
            out += bytes([samples[base], samples[base + 2], samples[base + 4], 255])
        elif color_type == 4:  # 灰度 + alpha
            out += bytes([samples[base], samples[base], samples[base], samples[base + 2]])
        elif color_type == 6:  # RGBA
            out += bytes(
                [
                    samples[base],
                    samples[base + 2],
                    samples[base + 4],
                    samples[base + 6],
                ]
            )
        else:
            raise ValueError(f"颜色类型 {color_type} 无 16 位镜像规则")
    return bytes(out)


# ---------------------------------------------------------------------------
# 固件清单定义
# ---------------------------------------------------------------------------

# (name, width, height, depth, color_type, filters, plte, trns, extra, idat_splits, seed)
#   filters: int = 全程同一滤波；list = 逐行循环；None = 全类型轮转
CYCLE_ALL = [0, 1, 2, 3, 4]

FIXTURES = [
    # -- 8 位：五种颜色类型 × 滤波覆盖 ------------------------------------
    dict(name="rgba8_f0", width=16, height=16, depth=8, color_type=6, filters=0, seed=0x4741_0001),
    dict(name="rgba8_mixed", width=9, height=11, depth=8, color_type=6, filters=CYCLE_ALL, seed=0x4741_0002),
    dict(name="rgb8_odd", width=13, height=7, depth=8, color_type=2, filters=CYCLE_ALL, seed=0x4741_0003),
    dict(name="gray8", width=17, height=5, depth=8, color_type=0, filters=CYCLE_ALL, seed=0x4741_0004),
    dict(name="gray_alpha8", width=8, height=8, depth=8, color_type=4, filters=CYCLE_ALL, seed=0x4741_0005),
    dict(name="palette8_trns", width=12, height=9, depth=8, color_type=3, filters=CYCLE_ALL,
         palette_seed=0x4741_0006, seed=0x4741_0006),
    dict(name="rgb8_trns", width=10, height=10, depth=8, color_type=2, filters=CYCLE_ALL,
         key_rgb=(0x40, 0x80, 0xC0), seed=0x4741_0007),
    dict(name="gray8_trns", width=6, height=6, depth=8, color_type=0, filters=CYCLE_ALL,
         key_gray=0x40, seed=0x4741_0008),
    # -- 16 位：四种颜色类型 ----------------------------------------------
    dict(name="rgba16", width=8, height=6, depth=16, color_type=6, filters=CYCLE_ALL, seed=0x4741_0009),
    dict(name="rgb16", width=7, height=5, depth=16, color_type=2, filters=CYCLE_ALL, seed=0x4741_000A),
    dict(name="gray16", width=9, height=9, depth=16, color_type=0, filters=CYCLE_ALL, seed=0x4741_000B),
    dict(name="gray_alpha16", width=5, height=5, depth=16, color_type=4, filters=CYCLE_ALL, seed=0x4741_000C),
    # -- 大图 + 结构变体 ----------------------------------------------------
    dict(name="rgb8_big", width=64, height=48, depth=8, color_type=2, filters=CYCLE_ALL, seed=0x4741_000D),
    dict(name="gray8_big", width=48, height=48, depth=8, color_type=0, filters=CYCLE_ALL, seed=0x4741_000E),
    dict(name="rgba8_ancillary", width=11, height=11, depth=8, color_type=6, filters=CYCLE_ALL,
         extra=[(b"gAMA", struct.pack(">I", 45455)), (b"pHYs", struct.pack(">IIB", 2835, 2835, 1)),
                (b"tEXt", b"Comment\x00deterministic fixture")], seed=0x4741_000F),
    dict(name="gray8_multi_idat", width=14, height=14, depth=8, color_type=0, filters=CYCLE_ALL,
         idat_splits=3, seed=0x4741_0010),
]


def build_fixture(spec: dict) -> dict:
    name = spec["name"]
    width, height, depth, color_type = spec["width"], spec["height"], spec["depth"], spec["color_type"]
    channels = CHANNELS[color_type]

    plte = None
    trns = None
    if color_type == 3:
        plte, trns = make_palette(spec["palette_seed"])
        samples = make_indices(width, height, spec["seed"])
    else:
        samples = make_samples(width, height, depth, color_type, spec["seed"])
    if "key_rgb" in spec:
        trns = struct.pack(">HHH", *spec["key_rgb"])
    if "key_gray" in spec:
        trns = struct.pack(">H", spec["key_gray"])

    filters = spec.get("filters", CYCLE_ALL)
    extra = spec.get("extra", [])
    idat_splits = spec.get("idat_splits", 1)
    png = encode_png(width, height, depth, color_type, samples, filters, plte, trns, extra, idat_splits)

    # 1) 参考解码器自校验：解出来的样本必须与编码输入逐字节一致。
    ref = decode_png_reference(png)
    assert ref["samples"] == samples, f"{name}: 参考解码往返不一致"
    assert ref["width"] == width and ref["height"] == height, f"{name}: 尺寸往返不一致"

    # 2) Pillow 交叉校验（校验参考解码器，非黄金值来源）。
    pillow_note = ""
    if depth == 8:
        im = Image.open(io.BytesIO(png))
        pillow_raw = im.convert("RGBA").tobytes()
        if color_type == 3:
            # 调色板：Pillow 索引流与参考样本一致。
            assert list(im.tobytes()) == list(samples), f"{name}: Pillow 索引流与参考样本不一致"
        else:
            assert list(im.tobytes()) == list(samples), f"{name}: Pillow 样本流与参考解码不一致"
        rgba = pillow_raw
        pillow_checked = True
    else:
        rgba = rgba_mirror_16bit(samples, color_type, channels)
        pillow_checked = False
        try:
            im = Image.open(io.BytesIO(png))
            if color_type == 0:
                # Pillow 对 gray16 给 I;16（小端 u16）：字节交换后比对样本。
                le = im.tobytes()
                be = bytearray()
                for i in range(0, len(le), 2):
                    be += bytes([le[i + 1], le[i]])
                assert bytes(be) == samples, f"{name}: Pillow I;16 样本（交换后）与参考不一致"
                pillow_note = "pillow I;16 小端样本交叉校验通过；RGBA 镜像自算（Pillow I;16 直转 RGBA 为垃圾值）"
            else:
                # RGB16/GA16/RGBA16：Pillow 降为 8 位即高位截断，与镜像规则一致。
                assert im.convert("RGBA").tobytes() == rgba, f"{name}: Pillow 16 位截断与镜像不一致"
                pillow_note = "pillow 16 位高位截断交叉校验通过"
                pillow_checked = True
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - 记录后按未校验处理
            pillow_note = f"Pillow 交叉校验跳过：{exc}"
            pillow_checked = False

    return {
        "spec": spec,
        "png": png,
        "samples": samples,
        "rgba": rgba,
        "plte": plte,
        "trns": trns,
        "chunk_log": ref["chunk_log"],
        "pillow_checked": pillow_checked,
        "pillow_note": pillow_note,
    }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    entries = []
    for spec in FIXTURES:
        fx = build_fixture(spec)
        name = spec["name"]
        png_path = os.path.join(FIXTURE_DIR, f"{name}.png")
        samples_path = os.path.join(FIXTURE_DIR, f"{name}.samples.bin")
        rgba_path = os.path.join(FIXTURE_DIR, f"{name}.rgba.bin")
        with open(png_path, "wb") as fh:
            fh.write(fx["png"])
        with open(samples_path, "wb") as fh:
            fh.write(fx["samples"])
        with open(rgba_path, "wb") as fh:
            fh.write(fx["rgba"])

        filters = spec.get("filters", CYCLE_ALL)
        if isinstance(filters, int):
            filter_desc = f"all-{filters}"
        else:
            filter_desc = "cycle-0-4"
        entry = {
            "name": name,
            "file": f"{name}.png",
            "samples_file": f"{name}.samples.bin",
            "rgba_file": f"{name}.rgba.bin",
            "width": spec["width"],
            "height": spec["height"],
            "bit_depth": spec["depth"],
            "color_type": spec["color_type"],
            "color_type_name": COLOR_TYPE_NAMES[spec["color_type"]],
            "channels": CHANNELS[spec["color_type"]],
            "bytes_per_pixel": CHANNELS[spec["color_type"]] * spec["depth"] // 8,
            "filters": filter_desc,
            "has_plte": fx["plte"] is not None,
            "has_trns": fx["trns"] is not None,
            "extra_chunks": [c[0].decode("ascii") for c in spec.get("extra", [])],
            "idat_chunks": spec.get("idat_splits", 1),
            "seed": spec["seed"],
            "samples_len": len(fx["samples"]),
            "samples_sha256": sha256_hex(fx["samples"]),
            "rgba_sha256": sha256_hex(fx["rgba"]),
            "png_sha256": sha256_hex(fx["png"]),
            "png_len": len(fx["png"]),
            "chunk_sequence": fx["chunk_log"],
            "pillow_cross_check": fx["pillow_checked"],
        }
        if fx["pillow_note"]:
            entry["pillow_note"] = fx["pillow_note"]
        if spec["depth"] == 16:
            entry["rgba_rule"] = "16-bit high-byte truncation mirror (alpha=255 when absent)"
        entries.append(entry)
        print(f"[ok] {name:20s} {spec['width']}x{spec['height']} d{spec['depth']} "
              f"ct{spec['color_type']} filters={filter_desc} png={len(fx['png'])}B "
              f"samples={len(fx['samples'])}B pillow={fx['pillow_checked']}")

    manifest = {
        "format": "png-golden-fixtures",
        "version": 1,
        "generator": "png_golden_gen.py",
        "signature_hex": PNG_SIGNATURE.hex(),
        "reference_decoder": "independent in-script decoder (zlib + manual unfiltering); Pillow used only as cross-check",
        "random_source": "splitmix64 (constants mirrored in Rust tests)",
        "count": len(entries),
        "fixtures": entries,
    }
    manifest_path = os.path.join(FIXTURE_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print(f"\n共 {len(entries)} 张固件 + manifest.json 写入 {FIXTURE_DIR}")
    print(f"manifest sha256: {sha256_hex(open(manifest_path, 'rb').read())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
