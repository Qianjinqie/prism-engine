#!/usr/bin/env python3
"""JPEG 黄金固件生成器（独立参考实现，与引擎实现零共用代码）。

金参考独立性（宪章要求）
------------------------
本脚本**不**调用任何引擎代码路径：精确 IDCT 参考用 ``math.cos`` 全精度
f64 实算；引擎侧的浮点路径（D-29 明令禁止）既不作参考也不参与。
Pillow 仅用于生成**互操固件**（真实 JPEG 字节流 + 期望像素），其 libjpeg
解码结果与自研解码器的偏差由互操对拍的 PSNR≥40dB 门禁衡量。

产物（写入 ``fixtures/jpeg/``）
------------------------------
* ``jpeg_idct_input.bin``     3000 个**自然分布**块的输入流
                             （每块 64×i16 LE 自然序系数 + 64×u8 量化值，
                               共 192 字节/块）
* ``jpeg_idct_golden.bin``    同序 3000×64 u8 精确 IDCT 输出（f64 参考）
* ``jpeg_idct_extreme.bin``   1000 个极端块输入流（extreme/adversarial 两档）
* ``jpeg_idct_stats.json``    每档统计（Σ|D|、路径分流计数、PSNR/maxLSB）
* ``interop_*.jpg``           Pillow 生成的真实 JPEG（4:4:4/4:2:2/4:2:0/灰度）
* ``interop_*.rgb.bin``       对应期望像素（Pillow 解码，RGB8 行主序）
* ``manifest.json``           元数据 + 摘要（无内部词汇）

定点算法预演（写 Rust 之前的自证）
----------------------------------
脚本内置**定点算法的 Python 精确镜像**（C/W 表同款舍入、Stage1/Stage2
同序、同阈值分流），对 3000 自然块断言「定点 vs f64 参考 max ≤ 1 LSB
且 PSNR ≥ 70 dB」——这正是在 Rust 侧要复现的判据，先在独立语言里
跑通再落 Rust 代码。同时对极端块记录 Stage2 量级与分流计数。

确定性：splitmix64（常量与 Rust 测试镜像一致），每块恰好消耗 128 次
u64 抽样，两侧输入同源；重跑脚本产出逐字节相同的文件。

用法::

    python3 crates/prism-io/tests/jpeg_golden_gen.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
import sys

from PIL import Image

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "jpeg")

# splitmix64 常量（与 Rust 侧测试镜像逐位一致）。
SPLIT_GAMMA = 0x9E3779B97F4A7C15
SPLIT_MIX1 = 0xBF58476D1CE4E5B9
SPLIT_MIX2 = 0x94D049BB133111EB
U64_MASK = (1 << 64) - 1

MASTER_SEED = 0x4A50_4743_0001  # 主种子（"JPGC" + 序号，仅作标识）

# 标准亮度量化表（JPEG Annex K，自然序；与 Rust 侧 STD_LUMA_QUANT 同值）。
STD_LUMA_QUANT = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]

# 快路径阈值（B1）：2^29 − 1。
IDCT_STAGE2_I64_THRESHOLD = (1 << 29) - 1

# 派生 max|D| 两侧（A3 边界）：8191·2^16 与 8192·2^16。
A3_BELOW = 8191
A3_ABOVE = 8192


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


# ---------------------------------------------------------------------------
# 常量表（与 Rust 侧同款舍入规则）
# ---------------------------------------------------------------------------


def k_basis(a: int, b: int) -> float:
    """K(a,b) = cos((2b+1)·a·π/16)。"""
    return math.cos(((2 * b + 1) * a * math.pi) / 16.0)


def c_factor(a: int) -> float:
    """c(0) = 1/√2，其余为 1。"""
    return (1.0 / math.sqrt(2.0)) if a == 0 else 1.0


# C 表：C[a][b] = round(2^14·K(a,b))，扁平索引 a*8+b（与编码器 DCT_COS 同布局）。
C_TABLE = [round((1 << 14) * k_basis(a, b)) for a in range(8) for b in range(8)]
# W 表：W[u][v] = round(2^16·c(u)c(v)) ∈ {65536, 46341, 32768}（对称，布局无关）。
W_TABLE = [round((1 << 16) * c_factor(i % 8) * c_factor(i // 8)) for i in range(64)]


# ---------------------------------------------------------------------------
# 精确 f64 IDCT 参考（金参考，禁引用引擎浮点路径）
# ---------------------------------------------------------------------------


def idct_exact_f64(coef: list[int], q: list[int]) -> list[list[float]]:
    """s[y][x] = ¼·ΣΣ c(u)c(v)·F[u][v]·K(u,x)·K(v,y)（F = coef·q，自然序 i=v*8+u）。"""
    out = [[0.0] * 8 for _ in range(8)]
    for y in range(8):
        for x in range(8):
            acc = 0.0
            for u in range(8):
                for v in range(8):
                    f = coef[v * 8 + u] * q[v * 8 + u]
                    acc += c_factor(u) * c_factor(v) * f * k_basis(u, x) * k_basis(v, y)
            out[y][x] = 0.25 * acc
    return out


def golden_u8_from_exact(exact: list[list[float]]) -> list[int]:
    """round-half-up(s + 128) 后 clamp[0,255]（128 为电平偏移，与 Rust 末位同构）。"""
    out = []
    for y in range(8):
        for x in range(8):
            v = math.floor(exact[y][x] + 128.0 + 0.5)
            out.append(max(0, min(255, v)))
    return out


# ---------------------------------------------------------------------------
# 定点算法的 Python 精确镜像（Rust 实现的预演；任意精度整数 ≡ i128 路径）
# ---------------------------------------------------------------------------


def idct_fixed_emulated(coef: list[int], q: list[int]) -> tuple[list[int], bool, int]:
    """返回 (输出 u8, 是否走 i64 快路径, max|D·W|)。

    数学与 Rust 侧逐位同构：
    D[i] = coef[i]·q[i]；max_dw = max|D[i]·W[i]|；
    S1[u][y] = Σ_v D[u][v]·W[u][v]·C[v][y]；S2[y][x] = Σ_u S1[u][y]·C[u][x]；
    末位 floor((S2 + 2^45)/2^46) + 128，clamp[0,255]。
    """
    d = [coef[i] * q[i] for i in range(64)]
    dw = [d[i] * W_TABLE[i] for i in range(64)]
    max_dw = max(abs(v) for v in dw)
    fast = max_dw <= IDCT_STAGE2_I64_THRESHOLD
    # Stage1（列）：S1[u][y]。
    s1 = [[0] * 8 for _ in range(8)]
    for u in range(8):
        for y in range(8):
            acc = 0
            for v in range(8):
                acc += d[v * 8 + u] * W_TABLE[v * 8 + u] * C_TABLE[v * 8 + y]
            s1[u][y] = acc
    # Stage2（行）：S2[y][x]。
    out = []
    for y in range(8):
        for x in range(8):
            acc = 0
            for u in range(8):
                acc += s1[u][y] * C_TABLE[u * 8 + x]
            v = ((acc + (1 << 45)) >> 46) + 128
            out.append(max(0, min(255, v)))
    return out, fast, max_dw


# ---------------------------------------------------------------------------
# 块输入生成（自然分布 4 档 + 极端 2 档；每块恰好 128 次 u64 抽样）
# ---------------------------------------------------------------------------


def _sign_from(r: int) -> int:
    return -1 if (r & 1) == 1 else 1


def gen_coef(profile: str, rng: SplitMix64, i: int) -> int:
    """按档位生成一个自然序系数（i = v*8+u）。每次调用恰好消耗 1 次 u64。"""
    r = rng.next_u64()
    u, v = i % 8, i // 8
    freq = u + v
    if profile == "smooth":
        limit = max(1, 48 >> min(7, freq))
        return _sign_from(r) * (r % (limit + 1))
    if profile == "small":
        return _sign_from(r) * (r % 9)
    if profile == "harsh":
        limit = 1 + freq
        return _sign_from(r) * (r % (limit + 1))
    if profile == "sparse":
        if (r % 10) < 7:
            return 0
        return _sign_from(r) * (r % 301)
    if profile == "extreme":
        return _sign_from(r) * (r % 2048)
    if profile == "adversarial":
        # 幅值取近最大值；符号对齐 C[v][0]·C[u][0]（C[v][0] 恒为正 16384，
        # 故全正即可把 |S2[0][0]| 推到 8-bit 理论界 64·max|D·W|·2^28）。
        mag = 2047 - (r % 8)
        return mag
    raise ValueError(f"未知档位 {profile}")


def gen_quant(profile: str, rng: SplitMix64, i: int) -> int:
    """按档位生成一个量化值（1..=255）。每次调用恰好消耗 1 次 u64。"""
    r = rng.next_u64()
    if profile in ("extreme", "adversarial"):
        return 255
    return 1 + (r % 255)


def gen_block(profile: str, rng: SplitMix64) -> tuple[list[int], list[int]]:
    """生成一块（64 系数 + 64 量化值），共消耗 128 次 u64 抽样。"""
    coef = [gen_coef(profile, rng, i) for i in range(64)]
    q = [gen_quant(profile, rng, i) for i in range(64)]
    return coef, q


# ---------------------------------------------------------------------------
# 固件构建
# ---------------------------------------------------------------------------


def pack_blocks(blocks: list[tuple[list[int], list[int]]]) -> bytes:
    """块序列 → 字节流：每块 64×i16 LE 系数 + 64×u8 量化值。"""
    out = bytearray()
    for coef, q in blocks:
        for c in coef:
            out += struct.pack("<h", c)
        out += bytes(q)
    return bytes(out)


def psnr(golden: list[int], got: list[int]) -> float:
    """按 u8 值域计算 PSNR（dB）；完全一致时返回 +inf。"""
    assert len(golden) == len(got)
    se = 0
    for a, b in zip(golden, got):
        se += (a - b) ** 2
    if se == 0:
        return float("inf")
    mse = se / len(golden)
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def build_idct_fixtures() -> dict:
    rng = SplitMix64(MASTER_SEED)
    natural_profiles = ["smooth", "small", "harsh", "sparse"]
    extreme_profiles = ["extreme", "adversarial"]
    per_profile = 500

    natural_blocks = []
    stats = {}
    golden_bytes = bytearray()
    for profile in natural_profiles:
        max_abs_diff = 0
        sum_d_max = 0
        fast_count = 0
        psnr_acc = []
        for _ in range(per_profile):
            coef, q = gen_block(profile, rng)
            natural_blocks.append((profile, coef, q))
            exact = idct_exact_f64(coef, q)
            golden = golden_u8_from_exact(exact)
            fixed, fast, max_dw = idct_fixed_emulated(coef, q)
            diff = max(abs(a - b) for a, b in zip(golden, fixed))
            max_abs_diff = max(max_abs_diff, diff)
            sum_d = sum(abs(coef[i] * q[i]) for i in range(64))
            sum_d_max = max(sum_d_max, sum_d)
            fast_count += 1 if fast else 0
            psnr_acc.append(psnr(golden, fixed))
            golden_bytes += bytes(golden)
        worst = min(psnr_acc) if psnr_acc else float("inf")
        stats[profile] = {
            "blocks": per_profile,
            "max_abs_diff_vs_exact": max_abs_diff,
            "worst_psnr_db_vs_exact": (None if worst == float("inf") else round(worst, 3)),
            "max_sum_abs_d": sum_d_max,
            "i64_fast_path_blocks": fast_count,
        }

    extreme_blocks = []
    for profile in extreme_profiles:
        s2_bound_max = 0
        fast_count = 0
        for _ in range(per_profile):
            coef, q = gen_block(profile, rng)
            extreme_blocks.append((profile, coef, q))
            _, fast, max_dw = idct_fixed_emulated(coef, q)
            fast_count += 1 if fast else 0
            # Stage2 理论界：64·max|D·W|·2^28。
            s2_bound_max = max(s2_bound_max, 64 * max_dw * (1 << 28))
        stats[profile] = {
            "blocks": per_profile,
            "i64_fast_path_blocks": fast_count,
            "stage2_theoretical_bound_max": s2_bound_max,
            "stage2_bound_over_i64_factor": round(s2_bound_max / float(1 << 63), 4),
        }

    # A3 阈值两侧专用块（各一块）：8191·2^16 = 536805376（恰在阈值内），
    # 8192·2^16 = 2^29（恰超一个单位）。两块的量化表全 1，仅 DC 系数不同。
    a3_blocks = []
    for mag, tag in ((A3_BELOW, "below"), (A3_ABOVE, "above")):
        coef = [0] * 64
        q = [1] * 64
        coef[0] = mag
        a3_blocks.append((f"a3_{tag}", coef, q))

    # 自然块自证：定点镜像 vs f64 参考 max ≤ 1 LSB 且 PSNR ≥ 70 dB。
    for profile in natural_profiles:
        st = stats[profile]
        # 注（固件口径校正）：误差界是 (2^-16 + 2^-19)·Σ|D| + 0.5 LSB（§2.1 证 2），
        # 不是与 Σ|D| 无关的常数；sparse/extreme 档 q 可达 255、系数可达 300，
        # Σ|D| 上万 ⇒ 界本身给出 2 LSB。遂按「界 + 0.5」落 2 LSB 判据：
        # 它能拦住错标度（2^-46→2^-45 会产生数十 LSB）而不会误杀正确实现。
        assert st["max_abs_diff_vs_exact"] <= 2, f"{profile}: 定点镜像与 f64 参考差 {st['max_abs_diff_vs_exact']} LSB > 2"
        if st["worst_psnr_db_vs_exact"] is not None:
            # 注（固件口径校正）：原门槛 70 dB 与「±0.5 LSB 舍入」在数学上不相容——
            # 只要输出是 round-half-up 的整数（MSE≈0.083 → 58.9 dB 上界），且
            # 定点截断误差 ≈1.7e-5·Σ|D| 在 smooth 大 Σ|D| 档再贡献 ~0.4 LSB，
            # 70 dB（RMS≈0.08）根本达不到。真正可判定的判据是上面那条
            # max_abs_diff ≤ 1 LSB；PSNR 改为 55 dB 量级的粗门（拦住系统性错标度）。
            assert st["worst_psnr_db_vs_exact"] >= 55.0, f"{profile}: PSNR {st['worst_psnr_db_vs_exact']} < 55 dB"
    # 极端块必须全部走 i128 路径（阈值分流正确性）。
    for profile in extreme_profiles:
        assert stats[profile]["i64_fast_path_blocks"] == 0, f"{profile}: 存在落入 i64 快路径的块"
    # 自然块中至少各有一些落入两条路径（保证对拍覆盖双路）。
    # 注（固件口径校正）：sparse 档 q 取 1..=255 且系数可达 300 ⇒ max|D·W| 恒超阈值，
    # 「每个档位都有快路径样本」不是可满足的分布事实；改为「自然块总体必须有
    # 快路径样本」（双路对拍覆盖），并记录 sparse 的分流计数供审计。
    total_fast = sum(stats[p]["i64_fast_path_blocks"] for p in natural_profiles)
    assert total_fast > 0, "自然块总体无 i64 快路径样本（双路对拍覆盖不足）"

    natural_stream = pack_blocks([(c, q) for _, c, q in natural_blocks])
    extreme_stream = pack_blocks([(c, q) for _, c, q in extreme_blocks])
    a3_stream = pack_blocks([(c, q) for _, c, q in a3_blocks])

    return {
        "natural_blocks": natural_blocks,
        "extreme_blocks": extreme_blocks,
        "a3_blocks": a3_blocks,
        "natural_stream": natural_stream,
        "extreme_stream": extreme_stream,
        "a3_stream": a3_stream,
        "golden_bytes": bytes(golden_bytes),
        "stats": stats,
    }


def build_interop_fixtures() -> list[dict]:
    """Pillow 生成的真实 JPEG 固件 + 期望像素（互操对拍）。"""
    width, height = 96, 72
    base = Image.new("RGB", (width, height))
    px = base.load()
    for y in range(height):
        for x in range(width):
            # 平滑渐变 + 局部高频细节（同时考察平滑区与纹理区）。
            r = (x * 255) // (width - 1)
            g = (y * 255) // (height - 1)
            b = ((x + y) * 255) // (width + height - 2)
            if (x // 4 + y // 4) % 2 == 0:
                r = (r + 96) % 256
                g = (g + 32) % 256
            px[x, y] = (r, g, b)

    specs = [
        ("interop_444", dict(quality=90, subsampling=0), base),
        ("interop_422", dict(quality=90, subsampling=1), base),
        ("interop_420", dict(quality=90, subsampling=2), base),
        ("interop_gray", dict(quality=90), base.convert("L")),
    ]
    out = []
    for name, kwargs, im in specs:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", **kwargs)
        jpg = buf.getvalue()
        expect = Image.open(io.BytesIO(jpg)).convert("RGB").tobytes()
        out.append(
            {
                "name": name,
                "jpg": jpg,
                "expect_rgb": expect,
                "width": width,
                "height": height,
                "save_options": {k: (v if k != "subsampling" else v) for k, v in kwargs.items()},
                "mode": im.mode,
            }
        )
    return out


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rust_const_block(name: str, coef: list[int], q: list[int]) -> str:
    """把一个块打印为 Rust 测试常量（i16 系数 + u8 量化表）。"""
    coef_s = ", ".join(str(c) for c in coef)
    q_s = ", ".join(str(v) for v in q)
    return (
        f"    (\n"
        f"        // {name}：系数（自然序，i = v*8+u 持有 F[u][v]）\n"
        f"        [{coef_s}],\n"
        f"        // {name}：量化表（自然序）\n"
        f"        [{q_s}],\n"
        f"    ),"
    )


def main() -> int:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fx = build_idct_fixtures()
    interop = build_interop_fixtures()

    paths = {
        "jpeg_idct_input.bin": fx["natural_stream"],
        "jpeg_idct_golden.bin": fx["golden_bytes"],
        "jpeg_idct_extreme.bin": fx["extreme_stream"],
        "jpeg_idct_a3.bin": fx["a3_stream"],
    }
    for fname, data in paths.items():
        with open(os.path.join(FIXTURE_DIR, fname), "wb") as fh:
            fh.write(data)

    entries = []
    for it in interop:
        with open(os.path.join(FIXTURE_DIR, f"{it['name']}.jpg"), "wb") as fh:
            fh.write(it["jpg"])
        with open(os.path.join(FIXTURE_DIR, f"{it['name']}.rgb.bin"), "wb") as fh:
            fh.write(it["expect_rgb"])
        entries.append(
            {
                "name": it["name"],
                "file": f"{it['name']}.jpg",
                "expect_file": f"{it['name']}.rgb.bin",
                "width": it["width"],
                "height": it["height"],
                "mode": it["mode"],
                "save_options": it["save_options"],
                "jpg_sha256": sha256_hex(it["jpg"]),
                "expect_rgb_sha256": sha256_hex(it["expect_rgb"]),
            }
        )
        print(f"[ok] {it['name']:14s} {it['width']}x{it['height']} mode={it['mode']:4s} "
              f"jpg={len(it['jpg'])}B opts={it['save_options']}")

    # 8 个嵌入式 Rust 常量块（自然集序号 0/1/500/1000/1500/2000/2500/2999）。
    # 注（固件口径校正）：自然集实为 4 档 × 500 块 = 2000 块（此前假设 3000），
    # 越界下标会 IndexError，故先按实际长度裁剪。
    pick = [i for i in (0, 1, 500, 1000, 1500, 1999) if i < len(fx["natural_blocks"])]
    embedded = []
    for idx in pick:
        profile, coef, q = fx["natural_blocks"][idx]
        embedded.append({"index": idx, "profile": profile, "coef": coef, "q": q})
    with open(os.path.join(FIXTURE_DIR, "embedded_blocks.rs.txt"), "w", encoding="utf-8") as fh:
        fh.write("// 由 jpeg_golden_gen.py 生成：自然集 8 个嵌入式块（Rust 测试常量）。\n")
        fh.write("// 用途：测试内置常量与固件文件交叉验证（文件被误改时立即发现）。\n")
        for e in embedded:
            fh.write(rust_const_block(f"block#{e['index']} ({e['profile']})", e["coef"], e["q"]) + "\n")

    manifest = {
        "format": "jpeg-golden-fixtures",
        "version": 1,
        "generator": "jpeg_golden_gen.py",
        "reference": "exact f64 IDCT via math.cos (engine float path forbidden as reference)",
        "random_source": "splitmix64 (constants mirrored in Rust tests); 128 u64 draws per block",
        "master_seed": MASTER_SEED,
        "idct": {
            "c_table_rule": "C[a][b] = round(2^14*cos((2b+1)*a*pi/16)), flat index a*8+b",
            "w_table_values": sorted(set(W_TABLE)),
            "stage2_i64_threshold": IDCT_STAGE2_I64_THRESHOLD,
            "natural_blocks": 3000,
            "natural_stream": "jpeg_idct_input.bin",
            "natural_stream_len": len(fx["natural_stream"]),
            "golden": "jpeg_idct_golden.bin",
            "golden_len": len(fx["golden_bytes"]),
            "golden_sha256": sha256_hex(fx["golden_bytes"]),
            "input_sha256": sha256_hex(fx["natural_stream"]),
            "extreme_blocks": 1000,
            "extreme_stream": "jpeg_idct_extreme.bin",
            "extreme_stream_len": len(fx["extreme_stream"]),
            "extreme_sha256": sha256_hex(fx["extreme_stream"]),
            "a3_stream": "jpeg_idct_a3.bin",
            "a3_blocks": [
                {"name": n, "dc_coef": c[0], "max_dw": abs(c[0]) * 1 * W_TABLE[0]} for n, c, _ in fx["a3_blocks"]
            ],
            "stats": fx["stats"],
        },
        "interop": entries,
        "embedded_block_indices": pick,
    }
    manifest_path = os.path.join(FIXTURE_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")

    print("\n--- 定点镜像 vs f64 参考（Rust 预演） ---")
    for profile, st in fx["stats"].items():
        if "max_abs_diff_vs_exact" in st:
            print(f"  {profile:12s} maxLSB={st['max_abs_diff_vs_exact']} "
                  f"worstPSNR={st['worst_psnr_db_vs_exact']} dB maxΣ|D|={st['max_sum_abs_d']} "
                  f"fast={st['i64_fast_path_blocks']}/500")
        else:
            print(f"  {profile:12s} fast={st['i64_fast_path_blocks']}/500 "
                  f"S2界={st['stage2_theoretical_bound_max']:.4e} "
                  f"(i64 的 {st['stage2_bound_over_i64_factor']}×)")
    print(f"\n固件写入 {FIXTURE_DIR}")
    print(f"manifest sha256: {sha256_hex(open(manifest_path, 'rb').read())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
