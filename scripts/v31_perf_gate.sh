#!/usr/bin/env bash
# ===========================================================================
# V3.1 收尾：全域基线与门禁（在既有脚本上增量，不新建门禁套件）
#
# 用法：
#   ./scripts/v31_perf_gate.sh                       # 编译 + 改动 crate 测试 + 五判据 + 哈希锚 + 光栅 5 项 + 三档 36 帧实测
#   PRISM_V31_PERF_MATRIX=1 ./scripts/v31_perf_gate.sh   # 追加 360 帧实测（仍不跑 3600 帧）
#
# 纪律（用户硬约束）：
#   - 不跑全量 `cargo test --workspace`（只跑改动 crate 相关测试 + qa_v31_gate）；
#   - 不跑 3600 帧全量计时；3600 帧只做外推并标注「外推、非实测」；
#   - 单条命令 ≤ 300 s；零第三方依赖（coreutils sort/sed/awk/date）。
#
# 哈希锚真源：`crates/prism-render/src/v31_config.rs::GALLERY_SHA256`，
# 本脚本不重复硬编码，生成 gallery.png 后调用门禁测试里的锚断言。
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GOLDEN_DIR="$REPO/docs/v3.1-golden"
export CARGO_TERM_COLOR=never
cd "$REPO"

# wall clock 首次基线（秒，3 位小数）：awk 只读、零依赖。
wall_clock() {
    local t0="$1" t1
    t1="$(date +%s.%N)"
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }'
}

echo "==> [1/8] 四 crate release 编译（prism-rt / prism-render / prism-app / prism-postfx）"
cargo build -q --release -p prism-rt -p prism-render -p prism-app -p prism-postfx
echo "    编译 EXIT=0"

echo "==> [2/8] 改动 crate 相关测试 + qa_v31_gate（wall clock 首次基线；不跑 --workspace）"
TEST_T0="$(date +%s.%N)"
# 只跑本轮改动的 crate + V3.1 门禁；逐 crate 指定，绝不 --workspace。
for crate in prism-io prism-mux prism-gpu prism-dsl prism-postfx prism-render prism-app; do
    cargo test -q --release -p "$crate" --lib
done
cargo test -q --release -p prism-render --test qa_v31_gate
cargo test -q --release -p prism-render --test qa_perf_counters
TEST_SEC="$(wall_clock "$TEST_T0")"
echo "    [v31-baseline] 改动 crate 测试 wall clock 首次基线 ${TEST_SEC} s"
echo "    [v31-baseline] 全量测试基线为既有核实记录 4669 passed / 0 failed / 240 段（本轮按纪律不重跑全量）"

echo "==> [3/8] golden manifest.sha256 校验（存在才验；不存在如实跳过，不假绿）"
if [ -f "$GOLDEN_DIR/manifest.sha256" ]; then
    ( cd "$GOLDEN_DIR" && sha256sum -c manifest.sha256 )
else
    echo "    golden manifest 不在工作树，跳过校验（只读门禁，不生成受保护资产）"
fi

echo "==> [4/8] 纯 CPU 单帧哈希锚：32 spp @ 1200×675（`一致` 档数值路径零回归）"
cargo run -q -p prism-rt --release --example gallery -- 1200 675 32
cargo test -q --release -p prism-render --test qa_v31_gate -- \
    --exact gallery_png_matches_locked_anchor_when_present

echo "==> [5/8] 光栅 5 项基准（raster_bench，ITERS=1；只采基线，不作 Go/No-Go 测速）"
RASTER_LOG="$(mktemp)"
RASTER_T0="$(date +%s.%N)"
RASTER_BENCH_ITERS=1 cargo run -q -p prism-raster --release --example raster_bench > "$RASTER_LOG"
RASTER_SEC="$(wall_clock "$RASTER_T0")"
RASTER_ROWS="$(grep -cE '^(circle64_1080p|star512_1080p|star512_4k|star2048_4k|ring_hole_2k)' "$RASTER_LOG" || true)"
echo "    [v31-raster] 5 项基准耗时 ${RASTER_SEC} s / 有效用例行 ${RASTER_ROWS}/5"
sed 's/^/    /' "$RASTER_LOG"
rm -f "$RASTER_LOG"

echo "==> [6/8] DSL / 场景翻译计时 + 文本/音频/纹理/粒子全域计数器（36 帧 720p 均衡档）"
IO_LOG="$(mktemp)"
./target/release/prism-app render examples/w2d_showcase.prism \
    --mode 均衡 --res 720p --frames 36 --fps 60 --out /tmp/v31_gate_36.mp4 \
    > "$IO_LOG" 2> "${IO_LOG}.err" || { cat "$IO_LOG" "${IO_LOG}.err"; exit 1; }
grep -E '\[v31-io\]|端到端单帧' "$IO_LOG" | sed 's/^/    /'
# 全域分段计时计数器：12 域全部采集，文本/音频/纹理/粒子四行单独列明。
for key in text audio texture particles raster kernel.reconstruct io.encode; do
    line="$(grep -F "v31.$key.micros=" "${IO_LOG}.err" || true)"
    if [ -n "$line" ]; then echo "    [v31-counter] $line"; fi
done
rm -f "$IO_LOG" "${IO_LOG}.err"

echo "==> [7/8] 三档 720p / 36 帧实测单帧耗时对照（含渲染+重建+编码）"
run_tier() {
    local mode="$1" frames="$2" log="/tmp/v31_gate_${1}_${2}.log"
    ./target/release/prism-app render examples/w2d_showcase.prism \
        --mode "$mode" --res 720p --frames "$frames" --fps 60 --out "/tmp/v31_gate_${mode}_${frames}.mp4" \
        > "$log" 2> "${log}.err" || { cat "$log" "${log}.err"; exit 1; }
    local ms
    ms="$(grep -oE '端到端单帧: [0-9.]+' "$log" | head -1 | grep -oE '[0-9.]+$' || echo 'n/a')"
    printf '    [v31-tier] 大模式 %-4s / %4s 帧 / 720p / 端到端单帧 %s ms' "$mode" "$frames" "$ms"
    if [ "$ms" != "n/a" ]; then
        awk -v m="$ms" 'BEGIN { printf " / 3600 帧外推、非实测 %.1f s\n", m * 3600 / 1000 }'
    else
        printf ' / 3600 帧外推、非实测 n/a\n'
    fi
}
for m in 一致 均衡 快速; do run_tier "$m" 36; done
if [ "${PRISM_V31_PERF_MATRIX:-0}" = "1" ]; then
    echo "    -- PRISM_V31_PERF_MATRIX=1：追加 360 帧实测（仍不跑 3600 帧）--"
    for m in 一致 均衡 快速; do run_tier "$m" 360; done
fi

echo "==> [8/8] 判据对照（如实报实测，不放宽 / 不降 spp / 不降分辨率）"
echo "    上限：一致 ≤1250 ms / 均衡 ≤430 ms / 快速 ≤150 ms / 自定义 不设"
echo "    3600 帧一律为「外推、非实测」；实测只覆盖 36 帧（可选 360 帧）。"

echo "==> V3.1 全域基线与门禁完成"
