// ===========================================================================
// refdc.cpp — 官方 DC-only 解码路径的**权威参考实现**（互通对拍用）
//
// ===========================================================================
// 【它是什么】
// 一份**独立于本仓Rust 实现**的 DC-only 解码链路复刻。作用是：当我们的
// 解码结果与官方不一致时，用它当「第三方裁判」区分是哪一层错了——
// 若refdc 与官方一致而我们不一致，则错在本仓；若三者一致，则差异来自
// 更上游的比特流层（AC 区 / DC 区解压）。
//
// 【零污染纪律】
// 本文件是**源码**，不参与 Rust 构建：`crates/` 下没有 build.rs、没有
// C++ 链接、没有 `cc` crate。它只由 `interop_gate.mjs` / 手工诊断时
// 在 `/tmp` 里现场编译，Rust 侧仅通过 **文件字节 / 命令行参数** 与它交互。
//
// 【复刻链路（逐行给出官方 3.1.5 源码依据）】
//   ImfDwaCompressor.cpp:790   halfZigBlock[comp]._buffer[0] = *currDcComp[comp]++
//   ImfDwaCompressor.cpp:840   h.setBits(buffer[0]); _dctData[0] = (float)h
//   ImfDwaCompressorSimd.h:821 dctInverse8x8DcOnly:
//                                        val = data[0]*0.3535536*0.3535536
//   ImfDwaCompressor.cpp:949   ((half)_dctData[0]).bits()  -> 铺满 64系数
//   ImfDwaCompressor.cpp:991+  _toLinear[h] 逐像素
//
// 【为什么 0.353536e-01 这个字面量必须照抄】
// 官方用的是 `3.535536e-01f` 这个**截断到 7 位有效数字**的近似值，
// 而不是精确的 1/sqrt(8) = 0.35355339059327373。两者在 f32 下差 1 ULP，
// 而 `val` 随后要被 `(half)` 舍入 —— 在某些 DC 值上这个 1 ULP 会跨过
// half 的舍入中点，让最终像素差 1 个 half-ULP。故此处**逐字符照抄官方
// 字面量**，不做「数学上更准」的替换。本仓 Rust 侧的 `DC_ONLY_GAIN`
// 同理。
//
// 用法：
//   refdc <u16_hex> [<u16_hex> ...]
//   逐个给出 DC u16（原生 little-endian half 位模式）对应的官方像素值。
//   例：refdc 45d7  ->  官方把常量块 0.5 解成 0.50048828125
// ===========================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <Imath/half.h>

int
main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf (stderr,
                      "usage: refdc <u16_hex> [<u16_hex> ...]\n"
                      "  逐个给出 DC u16（LE half 位模式）对应的官方像素值\n");
        return 2;
    }

    // ---- 重建官方 dwaCompressorToLinear 表 ----
    // 严格复刻 dwaLookups.cpp:307-345 `generateToLinear()`。
    // 实测核对：本表与官方 `dwaLookups.h` 里预生成的 65536 项静态表
    // **逐位全等**（用官方头文件编译的对照程序比对：mismatches = 0）。
    static unsigned short toLinear[65536];
    toLinear[0]       = 0;
    for (int i = 1; i < 65536; ++i)
    {
        // NaN / Inf -> 0
        if ((i & 0x7c00) == 0x7c00)
        {
            toLinear[i] = 0;
            continue;
        }

        half  h;
        float sign    = 1;
        float logBase = std::pow (2.7182818, 2.2);

        h.setBits ((uint16_t) i);
        if ((float) h < 0) sign = -1;

        if (std::fabs ((float) h) <= 1.0)
            h = (half) (sign * std::pow ((float) std::fabs ((float) h), 2.2f));
        else
            h = (half) (sign *
                        std::pow (logBase, (float) (std::fabs ((float) h) - 1.0)));

        // Xdr::write<CharPtrIO>(char*, unsigned short) 在 3.x 是小端恒等。
        toLinear[i] = h.bits ();
    }

    for (int a = 1; a < argc; ++a)
    {
        unsigned int u = 0;
        if (std::sscanf (argv[a], "%x", &u) != 1)
        {
            std::fprintf (stderr, "refdc: 无法解析 DC 值 %s\n", argv[a]);
            return 2;
        }

        // currDcComp：指向 DC 区的原生 unsigned short（小端）。
        half dcHalf;
        dcHalf.setBits ((uint16_t) u);
        float dct0 = (float) dcHalf;

        // dctInverse8x8DcOnly（float 运算）—— 字面量照抄官方，不做数学修正。
        float val = dct0 * 3.535536e-01f * 3.535536e-01f;

        // blockIsConstant 路径：((half)val).bits()，再 _toLinear 查表。
        unsigned short pixBits = ((half) val).bits ();
        unsigned short outBits = toLinear[pixBits];

        half outH;
        outH.setBits (outBits);
        half rawH;
        rawH.setBits ((uint16_t) u);

        std::printf ("dc=0x%04x dcHalf=%-12g dct0*0.125=%-14g pixBits=0x%04x "
                     "lut=0x%04x out=%g\n",
                     u, (float) rawH, val, pixBits, outBits, (float) outH);
    }
    return 0;
}