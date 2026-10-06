// ===========================================================================
// oexr_interop.cpp — 官方 OpenEXR 3.1.5 的 DWA 比特流对拍辅助程序。
//
// 【定位：验证工具，不是项目依赖】
//   本文件**不进 crates/**、不链接进任何 Rust 产物、不出现在任何 Cargo.toml、
//   没有 build.rs。Rust 侧（prism-io）只通过**文件字节**与本程序交互：
//     gen  <out.exr> <w> <h> <spec> <seed>   官方写 .exr
//     dump <in.exr> <out.txt>                官方读 .exr，导出像素位模式
//     cmp  <a.exr> <b.exr>                   官方解码两者并比误差
//     read <in.exr>                          官方读（不导出），报 OK/异常
//   全部走 stdio / 文件：不共享内存、不共享 ABI、不共享链接。
//
// 【为什么官方 OpenEXR 是"规范源"】
//   DWA 是有损压缩，其编码端的选择（规则表 / 量化口径 / Huffman 表）由官方
//   实现定义。本仓目标是"官方能读我们的、我们能读官方的"，故凡两侧不一致，
//   一律以官方为准（任务书 P1 第 1/2 条）。
//
// 编译（两个 -I 都要给：OpenEXR 3 头在 /usr/include/OpenEXR，Imath 在 /usr/include/Imath）：
//   g++ -std=c++14 oexr_interop.cpp -o oexr_interop \
//       -I/usr/include/OpenEXR -I/usr/include/Imath \
//       -lOpenEXR-3_1 -lOpenEXRCore -lImath-3_1 -lIex-3_1 -lIlmThread-3_1
//
// 【注意】OpenEXR 3 的 `Array2D` 拷贝构造已删除，本文件不写 `Array2D<T> b(a)`。
// ===========================================================================

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfStandardAttributes.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledOutputFile.h>
#include <ImfTileDescription.h>

#include <ImathBox.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace Imf;
using namespace Imath;

namespace {

// ---------------------------------------------------------------------------
// 数据形态：与 Rust 侧 `exr_interop.rs` 的 `gen_pattern` 逐位同构。
// 同一 LCG（a=1664525, c=1013904223）、同一浮点取整顺序 —— 两侧必须生成
// 完全相同的源数据，否则"误差"里会混入源数据差异。
// ---------------------------------------------------------------------------

/// 32 位 LCG（Numerical Recipes 参数）。
uint32_t lcg(uint32_t &s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

/// 生成 `w × h` 单通道源数据（f32）。
std::vector<float> gen_pattern(const std::string &kind, int w, int h, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    if (kind == "gradient") {
        // 平滑斜坡：DC 主导、AC 极少。
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                v[static_cast<size_t>(y) * w + x] =
                    static_cast<float>(x + y) / static_cast<float>(w + h);
    } else if (kind == "const") {
        for (size_t i = 0; i < v.size(); ++i) v[i] = 0.5f;
    } else if (kind.size() > 1 && kind[0] == 'c') {
        // `c<value>`：全常量块，值由形态串给出（如 c1 / c2 / c0.5）。
        // 用于把 DWA 的 DC 编码（字节序 + DCT 增益 + 量化）逐项反解出来。
        float val = static_cast<float>(std::atof(kind.c_str() + 1));
        for (size_t i = 0; i < v.size(); ++i) v[i] = val;
    } else if (kind == "noise") {
        // 逐像素伪随机 + 对数均匀分布：AC 全带宽、动态范围大，
        // 是最能暴露量化步长 / Huffman 表问题的形态。
        uint32_t s = seed;
        for (size_t i = 0; i < v.size(); ++i) {
            float u = static_cast<float>(lcg(s) >> 8) / static_cast<float>(1u << 24);
            v[i] = std::exp(u * 3.0f - 1.5f); // e^{-1.5} .. e^{1.5}
        }
    } else {
        std::fprintf(stderr, "unknown pattern '%s'\n", kind.c_str());
        std::exit(2);
    }
    return v;
}

// ---------------------------------------------------------------------------
// 规格串：`comp-ptype-layout-nchan-pattern`，如 `dwaa-half-scan-3-noise`
// ---------------------------------------------------------------------------

struct Spec {
    Compression comp = DWAA_COMPRESSION;
    PixelType pt = HALF;
    bool tiled = false;
    int nchan = 3;
    std::string pattern = "gradient";
};

bool parse_spec(const std::string &s, Spec &out) {
    std::vector<std::string> f;
    std::string cur;
    for (char c : s) {
        if (c == '-') { f.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    f.push_back(cur);
    if (f.size() != 5) return false;
    if (f[0] == "dwaa") out.comp = DWAA_COMPRESSION;
    else if (f[0] == "dwab") out.comp = DWAB_COMPRESSION;
    else if (f[0] == "none") out.comp = NO_COMPRESSION;
    else if (f[0] == "zip") out.comp = ZIP_COMPRESSION;
    else return false;
    if (f[1] == "half") out.pt = HALF;
    else if (f[1] == "float") out.pt = FLOAT;
    else return false;
    if (f[2] == "scan") out.tiled = false;
    else if (f[2] == "tile") out.tiled = true;
    else return false;
    out.nchan = std::atoi(f[3].c_str());
    if (out.nchan < 1 || out.nchan > 4) return false;
    out.pattern = f[4];
    return true;
}

/// 通道名：1=Y、3=RGB、4=RGBA（官方 DWA 默认规则表按这些后缀分类）。
const char *chan_name(int n, int i) {
    static const char *tbl[5][4] = {
        {nullptr, nullptr, nullptr, nullptr},
        {"Y", nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr},
        {"R", "G", "B", nullptr},
        {"R", "G", "B", "A"},
    };
    if (n < 1 || n > 4) return nullptr;
    return tbl[n][i];
}

// ---------------------------------------------------------------------------
// 统一读取：把任意 .exr（扫描线 / 分块）读成「逐通道 f32 平面」
// ---------------------------------------------------------------------------

struct Decoded {
    int w = 0;
    int h = 0;
    /// 通道（顺序 = 官方 ChannelList 的字典序，即 A < B < G < R < Y）。
    std::vector<std::string> names;
    std::vector<PixelType> types;
    std::vector<std::vector<float> > planes;
};

/// 读一个 .exr。扫描线走 `InputFile`，分块走 `TiledInputFile`（先试前者）。
///
/// 官方对 half 通道做 half→float 转换是**精确**的（half 是 float 的子集），
/// 故把一切都读成 f32 后按位比较是合法的：half 通道比的是「解码出的 f32 位模式」。
Decoded read_exr(const std::string &path) {
    Decoded d;
    // ---- 探测是否分块 ----
    bool is_tiled = false;
    {
        InputFile probe(path.c_str());
        is_tiled = probe.header().hasTileDescription();
    }

    if (!is_tiled) {
        InputFile file(path.c_str());
        const Header &hdr = file.header();
        const Box2i &dw = hdr.dataWindow();
        d.w = dw.max.x - dw.min.x + 1;
        d.h = dw.max.y - dw.min.y + 1;
        std::vector<std::vector<float> > storage;
        FrameBuffer fb;
        for (ChannelList::ConstIterator it = hdr.channels().begin();
             it != hdr.channels().end(); ++it) {
            const std::string nm = it.name();
            d.names.push_back(nm);
            d.types.push_back(it.channel().type);
            storage.push_back(std::vector<float>(static_cast<size_t>(d.w) * d.h, 0.0f));
        }
        size_t k = 0;
        for (ChannelList::ConstIterator it = hdr.channels().begin();
             it != hdr.channels().end(); ++it, ++k) {
            fb.insert(it.name(),
                      Slice(FLOAT,
                            reinterpret_cast<char *>(storage[k].data()),
                            sizeof(float),
                            sizeof(float) * static_cast<size_t>(d.w)));
        }
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);
        d.planes.swap(storage);
        return d;
    }

    TiledInputFile file(path.c_str());
    const Header &hdr = file.header();
    const Box2i &dw = hdr.dataWindow();
    d.w = dw.max.x - dw.min.x + 1;
    d.h = dw.max.y - dw.min.y + 1;
    std::vector<std::vector<float> > storage;
    FrameBuffer fb;
    for (ChannelList::ConstIterator it = hdr.channels().begin();
         it != hdr.channels().end(); ++it) {
        d.names.push_back(it.name());
        d.types.push_back(it.channel().type);
        storage.push_back(std::vector<float>(static_cast<size_t>(d.w) * d.h, 0.0f));
    }
    size_t k = 0;
    for (ChannelList::ConstIterator it = hdr.channels().begin();
         it != hdr.channels().end(); ++it, ++k) {
        fb.insert(it.name(),
                  Slice(FLOAT,
                        reinterpret_cast<char *>(storage[k].data()),
                        sizeof(float),
                        sizeof(float) * static_cast<size_t>(d.w)));
    }
    file.setFrameBuffer(fb);
    file.readTiles(0, file.numXTiles() - 1, 0, file.numYTiles() - 1);
    d.planes.swap(storage);
    return d;
}

// ---------------------------------------------------------------------------
// cmd_src：导出「源数据」（未经任何压缩）
// ---------------------------------------------------------------------------
//
// 为什么必须有这个命令（本轮加）：
//   对拍的前提是**两侧喂进去的是同一批源数据**。若 C++ 与 Rust 各自跑
//   `gen_pattern`，只要 `exp()` 的实现差 1 ULP，测出来的"压缩误差"里就会
//   混入源数据差异 —— 那是**假绿/假红**的根源，不是压缩器的行为。
//   故让 C++ 侧导出源数据的 f32 位模式，Rust 侧**直接读这份文件**，
//   从构造上消除"两侧源数据不同"这一类噪声。
//
// 输出格式与 `dump` 完全一致（DWA-DUMP 1），只是 ptype 一律 FLOAT：
// 源数据尚未量化成 half，统一按 f32 位模式存最不易出错。

int cmd_src(int argc, char **argv) {
    // oexr_interop src <out.txt> <w> <h> <spec> <seed>
    if (argc < 7) { std::fprintf(stderr, "src <out.txt> <w> <h> <spec> <seed>\n"); return 2; }
    const int w = std::atoi(argv[3]);
    const int h = std::atoi(argv[4]);
    Spec sp;
    if (!parse_spec(argv[5], sp)) { std::fprintf(stderr, "bad spec '%s'\n", argv[5]); return 2; }
    const uint32_t seed = static_cast<uint32_t>(std::atoll(argv[6]));

    std::vector<std::vector<float> > planes;
    for (int c = 0; c < sp.nchan; ++c)
        planes.push_back(gen_pattern(sp.pattern, w, h,
                                     seed + 0x9E3779B9u * static_cast<uint32_t>(c + 1)));

    FILE *fp = std::fopen(argv[2], "w");
    if (!fp) { std::perror("fopen"); return 2; }
    std::fprintf(fp, "DWA-DUMP 1\n");
    std::fprintf(fp, "W %d\n", w);
    std::fprintf(fp, "H %d\n", h);
    std::fprintf(fp, "NC %d\n", sp.nchan);
    for (int c = 0; c < sp.nchan; ++c) {
        std::fprintf(fp, "C %d %s FLOAT %d\n", c, chan_name(sp.nchan, c),
                     static_cast<int>(planes[c].size()));
        for (size_t i = 0; i < planes[c].size(); ++i) {
            float v = planes[c][i];
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            std::fprintf(fp, "%s%08x", (i % 16 == 0) ? "V " : " ", bits);
            if (i % 16 == 15 || i + 1 == planes[c].size()) std::fputc('\n', fp);
        }
    }
    std::fclose(fp);
    std::printf("SRC ok %s -> %s (%dx%d nc=%d)\n", argv[2], argv[5], w, h, sp.nchan);
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_srccmp：官方解码结果 vs 源数据（量化 DWA 的真实有损幅度）
// ---------------------------------------------------------------------------
//
// 输出与 `cmp` 同构，便于门禁脚本用同一套解析：
//   SRCCMP_OK 1
//   MAX_ABS / MAX_REL / MAX_AT / BITDIFF / N

int cmd_srccmp(int argc, char **argv) {
    // oexr_interop srccmp <in.exr> <src.txt>
    if (argc < 4) { std::fprintf(stderr, "srccmp <in.exr> <src.txt>\n"); return 2; }
    Decoded d = read_exr(argv[2]);

    // 极简 DWA-DUMP 解析（只取 ptype / name / V 行）。
    FILE *fp = std::fopen(argv[3], "r");
    if (!fp) { std::perror("fopen src"); return 2; }
    std::vector<std::string> snames;
    std::vector<std::vector<float> > splanes;
    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "C ", 2) == 0) {
            char nm[64] = {0};
            int idx = 0, cnt = 0;
            if (std::sscanf(line, "C %d %63s %*s %d", &idx, nm, &cnt) == 3) {
                snames.push_back(nm);
                splanes.push_back(std::vector<float>());
                (void)cnt;
            }
            splanes.back().reserve(static_cast<size_t>(cnt));
        } else if (line[0] == 'V' && !splanes.empty()) {
            char *p = line + 1;
            while (*p) {
                while (*p == ' ') ++p;
                if (!*p) break;
                char *end = nullptr;
                const unsigned long b = std::strtoul(p, &end, 16);
                if (end == p) break;
                float f;
                const uint32_t bits = static_cast<uint32_t>(b);
                std::memcpy(&f, &bits, 4);
                splanes.back().push_back(f);
                p = end;
            }
        }
    }
    std::fclose(fp);

    if (snames.size() != d.names.size()) {
        std::printf("SRCCMP_OK 0\n");
        std::printf("SHAPE_MISMATCH exr_nc=%d src_nc=%d\n",
                    static_cast<int>(d.names.size()), static_cast<int>(snames.size()));
        return 1;
    }

    double max_abs = 0.0, max_rel = 0.0;
    long long bitdiff = 0, total = 0;
    int max_chan = -1;
    long long max_idx = -1;
    for (size_t c = 0; c < d.names.size(); ++c) {
        const std::vector<float> &pd = d.planes[c];
        const std::vector<float> &ps = splanes[c];
        const size_t n = pd.size() < ps.size() ? pd.size() : ps.size();
        for (size_t i = 0; i < n; ++i) {
            const double x = static_cast<double>(pd[i]);
            const double y = static_cast<double>(ps[i]);
            const double ad = std::fabs(x - y);
            const double den = std::fabs(y) > 1e-6 ? std::fabs(y) : 1e-6;
            const double rd = ad / den;
            uint32_t bd, bs;
            std::memcpy(&bd, &pd[i], 4);
            std::memcpy(&bs, &ps[i], 4);
            if (bd != bs) bitdiff += 1;
            total += 1;
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) { max_rel = rd; max_chan = static_cast<int>(c); max_idx = static_cast<long long>(i); }
        }
    }
    std::printf("SRCCMP_OK 1\n");
    std::printf("DIM %d %d %d\n", d.w, d.h, static_cast<int>(d.names.size()));
    std::printf("MAX_ABS %.9g\n", max_abs);
    std::printf("MAX_REL %.9g\n", max_rel);
    std::printf("MAX_AT %d %lld\n", max_chan, max_idx);
    std::printf("BITDIFF %lld\n", bitdiff);
    std::printf("N %lld\n", total);
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_gen：官方写
// ---------------------------------------------------------------------------

int cmd_gen(int argc, char **argv) {
    // oexr_interop gen <out.exr> <w> <h> <spec> <seed>
    if (argc < 7) { std::fprintf(stderr, "gen <out.exr> <w> <h> <spec> <seed>\n"); return 2; }
    const std::string out = argv[2];
    const int w = std::atoi(argv[3]);
    const int h = std::atoi(argv[4]);
    Spec sp;
    if (!parse_spec(argv[5], sp)) { std::fprintf(stderr, "bad spec '%s'\n", argv[5]); return 2; }
    const uint32_t seed = static_cast<uint32_t>(std::atoll(argv[6]));

    // 每通道独立 seed（R/G/B 若同源会让 CSC 路径退化）。
    std::vector<std::vector<float> > fplanes;   // FLOAT 通道的缓冲
    std::vector<std::vector<uint16_t> > hplanes; // HALF 通道的缓冲
    for (int c = 0; c < sp.nchan; ++c) {
        std::vector<float> src =
            gen_pattern(sp.pattern, w, h, seed + 0x9E3779B9u * static_cast<uint32_t>(c + 1));
        if (sp.pt == FLOAT) {
            fplanes.push_back(src);
        } else {
            std::vector<uint16_t> hp(src.size());
            for (size_t i = 0; i < src.size(); ++i) hp[i] = half(src[i]).bits();
            hplanes.push_back(hp);
        }
    }

    // Header(w, h) 直接给出 0 基 dataWindow / displayWindow。
    Header hdr(w, h);
    hdr.compression() = sp.comp;
    for (int c = 0; c < sp.nchan; ++c)
        hdr.channels().insert(chan_name(sp.nchan, c), Channel(sp.pt));
    if (sp.tiled) hdr.setTileDescription(TileDescription(32, 32, ONE_LEVEL));

    FrameBuffer fb;
    for (int c = 0; c < sp.nchan; ++c) {
        const char *nm = chan_name(sp.nchan, c);
        if (sp.pt == FLOAT) {
            fb.insert(nm, Slice(FLOAT,
                                reinterpret_cast<char *>(fplanes[c].data()),
                                sizeof(float), sizeof(float) * static_cast<size_t>(w)));
        } else {
            fb.insert(nm, Slice(HALF,
                                reinterpret_cast<char *>(hplanes[c].data()),
                                sizeof(uint16_t), sizeof(uint16_t) * static_cast<size_t>(w)));
        }
    }

    if (sp.tiled) {
        TiledOutputFile file(out.c_str(), hdr);
        file.setFrameBuffer(fb);
        file.writeTiles(0, file.numXTiles() - 1, 0, file.numYTiles() - 1);
    } else {
        OutputFile file(out.c_str(), hdr);
        file.setFrameBuffer(fb);
        file.writePixels(static_cast<int>(hdr.dataWindow().max.y) -
                         static_cast<int>(hdr.dataWindow().min.y) + 1);
    }
    std::printf("GEN ok %s %dx%d spec=%s seed=%u\n", out.c_str(), w, h, argv[5], seed);
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_dump：官方读 → 文本位模式
// ---------------------------------------------------------------------------
//
// 格式（每行以 \n 结束，行内以单空格分隔）：
//   DWA-DUMP 1
//   W <w>
//   H <h>
//   NC <n>
//   C <index> <name> <ptype> <count>       ptype: HALF|FLOAT|UINT
//   V <v0> <v1> ... <v_{count-1}>           8 位十六进制 = f32 的 to_bits()
//   ...
// `V` 按行主序（y 外层、x 内层）。half 通道存的是「解码后 f32」的位模式 ——
// half→float 是精确转换，故与本仓 `ChannelPlane::values` 逐位可比。

int cmd_dump(int argc, char **argv) {
    // oexr_interop dump <in.exr> <out.txt>
    if (argc < 4) { std::fprintf(stderr, "dump <in.exr> <out.txt>\n"); return 2; }
    Decoded d = read_exr(argv[2]);
    FILE *fp = std::fopen(argv[3], "w");
    if (!fp) { std::perror("fopen"); return 2; }
    std::fprintf(fp, "DWA-DUMP 1\n");
    std::fprintf(fp, "W %d\n", d.w);
    std::fprintf(fp, "H %d\n", d.h);
    std::fprintf(fp, "NC %d\n", static_cast<int>(d.names.size()));
    for (size_t c = 0; c < d.names.size(); ++c) {
        const char *tn = d.types[c] == HALF ? "HALF" : (d.types[c] == FLOAT ? "FLOAT" : "UINT");
        std::fprintf(fp, "C %d %s %s %d\n", static_cast<int>(c), d.names[c].c_str(), tn,
                     static_cast<int>(d.planes[c].size()));
        const std::vector<float> &p = d.planes[c];
        for (size_t i = 0; i < p.size(); ++i) {
            float v = p[i];
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            std::fprintf(fp, "%s%08x", (i % 16 == 0) ? "V " : " ", bits);
            if (i % 16 == 15 || i + 1 == p.size()) std::fputc('\n', fp);
        }
        if (p.empty()) std::fprintf(fp, "V\n");
    }
    std::fclose(fp);
    std::printf("DUMP ok %s -> %s (%dx%d nc=%d)\n", argv[2], argv[3], d.w, d.h,
                static_cast<int>(d.names.size()));
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_cmp：官方解码两个文件并比误差
// ---------------------------------------------------------------------------
//
// 输出（机器可读，key=value 每行）：
//   CMP_OK 1|0
//   DIM <w> <h> <nc>
//   NAME <i> <name>
//   MAX_ABS <v>          最大绝对误差
//   MAX_REL <v>          最大相对误差（分母 = max(|b|, 1e-6)）
//   MAX_AT <chan> <idx>  最大相对误差所在位置（通道下标 / 像素线性下标）
//   BITDIFF <n>          f32 位模式不同的样本数
//   N <总比较样本数>
//
// 相对误差定义与本仓 exr_write 的 `DWA_TOL_REL` 口径一致：
//   rel = |a - b| / max(|b|, 1e-6)

int cmd_cmp(int argc, char **argv) {
    // oexr_interop cmp <a.exr> <b.exr>
    if (argc < 4) { std::fprintf(stderr, "cmp <a.exr> <b.exr>\n"); return 2; }
    Decoded a = read_exr(argv[2]);
    Decoded b = read_exr(argv[3]);
    bool shape_ok = (a.w == b.w && a.h == b.h && a.names.size() == b.names.size());
    for (size_t i = 0; shape_ok && i < a.names.size(); ++i)
        shape_ok = (a.names[i] == b.names[i]);
    if (!shape_ok) {
        std::printf("CMP_OK 0\n");
        std::printf("SHAPE_MISMATCH a=%dx%dx%d b=%dx%dx%d\n", a.w, a.h,
                    static_cast<int>(a.names.size()), b.w, b.h,
                    static_cast<int>(b.names.size()));
        return 1;
    }
    double max_abs = 0.0, max_rel = 0.0;
    long long bitdiff = 0, total = 0;
    int max_chan = -1;
    long long max_idx = -1;
    for (size_t c = 0; c < a.names.size(); ++c) {
        const std::vector<float> &pa = a.planes[c];
        const std::vector<float> &pb = b.planes[c];
        const size_t n = pa.size() < pb.size() ? pa.size() : pb.size();
        for (size_t i = 0; i < n; ++i) {
            const double x = static_cast<double>(pa[i]);
            const double y = static_cast<double>(pb[i]);
            const double ad = std::fabs(x - y);
            const double den = std::fabs(y) > 1e-6 ? std::fabs(y) : 1e-6;
            const double rd = ad / den;
            uint32_t ba, bb;
            std::memcpy(&ba, &pa[i], 4);
            std::memcpy(&bb, &pb[i], 4);
            if (ba != bb) bitdiff += 1;
            total += 1;
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) { max_rel = rd; max_chan = static_cast<int>(c); max_idx = static_cast<long long>(i); }
        }
    }
    std::printf("CMP_OK 1\n");
    std::printf("DIM %d %d %d\n", a.w, a.h, static_cast<int>(a.names.size()));
    for (size_t c = 0; c < a.names.size(); ++c)
        std::printf("NAME %d %s\n", static_cast<int>(c), a.names[c].c_str());
    std::printf("MAX_ABS %.9g\n", max_abs);
    std::printf("MAX_REL %.9g\n", max_rel);
    std::printf("MAX_AT %d %lld\n", max_chan, max_idx);
    std::printf("BITDIFF %lld\n", bitdiff);
    std::printf("N %lld\n", total);
    return 0;
}

// ---------------------------------------------------------------------------
// cmd_read：官方读但不导出（验证「不抛异常」）
// ---------------------------------------------------------------------------

int cmd_read(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "read <in.exr>\n"); return 2; }
    Decoded d = read_exr(argv[2]);
    double sum = 0.0;
    for (size_t c = 0; c < d.planes.size(); ++c)
        for (size_t i = 0; i < d.planes[c].size(); ++i)
            sum += static_cast<double>(d.planes[c][i]);
    std::printf("READ_OK 1 %dx%d nc=%d checksum=%.9g\n", d.w, d.h,
                static_cast<int>(d.names.size()), sum);
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: oexr_interop <gen|src|srccmp|dump|cmp|read> ...\n"
                     "  gen  <out.exr> <w> <h> <comp-{dwaa,dwab,none,zip}-{half,float}-{scan,tile}-<1|3|4>-<pattern>> <seed>\n"
                     "  src  <out.txt> <w> <h> <spec> <seed>   导出源数据（DWA-DUMP 1 格式）\n"
                     "  srccmp <in.exr> <src.txt>官方解码 vs 源数据（有损幅度）\n"
                     "  dump <in.exr> <out.txt>\n"
                     "  cmp  <a.exr> <b.exr>\n"
                     "  read <in.exr>\n");
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "gen") return cmd_gen(argc, argv);
        if (cmd == "src") return cmd_src(argc, argv);
        if (cmd == "srccmp") return cmd_srccmp(argc, argv);
        if (cmd == "dump") return cmd_dump(argc, argv);
        if (cmd == "cmp") return cmd_cmp(argc, argv);
        if (cmd == "read") return cmd_read(argc, argv);
        if (cmd == "version") {
            std::printf("OPENEXR_VERSION %d.%d.%d\n", OPENEXR_VERSION_MAJOR,
                        OPENEXR_VERSION_MINOR, OPENEXR_VERSION_PATCH);
            return 0;
        }
    } catch (const std::exception &e) {
        // 官方读我们的文件失败 → 这正是方向 B 要捕捉的信号，必须如实报出。
        std::printf("EXCEPTION %s\n", e.what());
        std::fprintf(stderr, "EXCEPTION %s\n", e.what());
        return 3;
    } catch (...) {
        std::printf("EXCEPTION unknown\n");
        std::fprintf(stderr, "EXCEPTION unknown\n");
        return 3;
    }
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    return 2;
}
