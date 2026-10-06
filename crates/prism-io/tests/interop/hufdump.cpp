// ===========================================================================
// hufdump.cpp — 用官方 Imf::hufUncompress 解一段 AC 区字节，导出 u16 序列。
//
// 【定位：验证工具，不是项目依赖】
//   与 oexr_interop 同一性质：项目外 / 不链接进任何 Rust 产物 / 无 build.rs。
//   存在的理由：AC 区解码出问题时，必须能区分
//     (a) 我们的 huf_uncompress 解错→  改prism-io
//     (b) 我们的 un_rle_ac / assemble 解错  →  改 prism-io
//     (c) 官方文件本身自洽            →  改 harness
//   没有这个探针就得靠猜，那正是本项目吃过亏的地方。
//
// 编译：
//   g++ -std=c++14 hufdump.cpp -o hufdump \
//       -I/usr/include/OpenEXR -I/usr/include/Imath -lIex-3_1
//
// 用法：
//   hufdump <ac.bin> <n_raw>
//     ac.bin — AC 区原始字节（huf 头 + 码流）
//     n_raw  — 期望解出的 u16 项数（= DWA 块头的 acUncompressedCount）
// 输出（机器可读）：
//   HUF_OK 1|0
//   N <n_raw>
//   V <u16> <u16> ...   每行 16 项，4 位十六进制
// ===========================================================================

#include <ImfHuf.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

int
main (int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf (stderr,
                      "usage: hufdump <ac.bin> <n_raw>\n"
                      "  用官方 Imf::hufUncompress 解 AC 区字节，导出 u16 序列\n");
        return 2;
    }

    // 读入 AC 区字节。
    FILE* f = std::fopen (argv[1], "rb");
    if (!f)
    {
        std::perror ("fopen ac.bin");
        return 2;
    }
    std::vector<char> buf;
    char        tmp[65536];
    size_t      n;
    while ((n = std::fread (tmp, 1, sizeof (tmp), f)) > 0)
        buf.insert (buf.end (), tmp, tmp + n);
    std::fclose (f);

    const int n_raw = std::atoi (argv[2]);
    std::printf ("HUF_BYTES %zu\n", buf.size ());
    std::printf ("N %d\n", n_raw);

    // hufUncompress 需 nRaw 个 u16 的输出缓冲。
    std::vector<unsigned short> raw (static_cast<size_t> (n_raw > 0 ? n_raw : 1), 0xDEAD);

    try
    {
        // 注意 3.1 的签名返回 `int`（内部抛异常），不是 void。
        Imf_3_1::hufUncompress (
            reinterpret_cast<const char*> (buf.data ()),
            static_cast<int> (buf.size ()),
            reinterpret_cast<unsigned short*> (raw.data ()),
            n_raw);
    }
    catch (const std::exception& e)
    {
        std::printf ("HUF_OK 0\n");
        std::printf ("HUF_THROW %s\n", e.what ());
        return 1;
    }

    std::printf ("HUF_OK 1\n");
    for (int i = 0; i < n_raw; ++i)
    {
        if (i % 16 == 0) std::printf ("V ");
        else             std::printf (" ");
        std::printf ("%04x", raw[static_cast<size_t> (i)]);
        if (i % 16 == 15 || i + 1 == n_raw) std::printf ("\n");
    }
    return 0;
}