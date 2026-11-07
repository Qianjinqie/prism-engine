#!/bin/sh
# ===========================================================================
# V3.0 W4 —— 统一顶层入口冷启动 + 单文件双档 + 零损失 可复现脚本。
#
# 用法（workspace 根目录）：
#   sh scripts/w4_cold.sh            # 全部：零损失 md5 / 空环境冷启动 / 双档两跑逐字节
#   sh scripts/w4_cold.sh cold       # 只跑空环境冷启动（无 ffmpeg / 无 GPU）
#   sh scripts/w4_cold.sh zero       # 只跑零损失凭证（3 个 .prism md5）
#   sh scripts/w4_cold.sh dual       # 只跑双档两跑逐字节
#
# 判据与门槛全部冻结在 docs/v3.0-路线图.md §7 与 docs/v3.0-W4-收官实录.md，
# 本脚本不接收可调参数（不事后调门槛、不换尺子、不软化阈值）。
# 零第三方依赖：只调 cargo / prism 二进制 / md5sum / cmp（系统基础工具）。
# ===========================================================================

set -eu
MODE="${1:-all}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/target/debug/prism"
FILMS="$ROOT/examples/films"
OUT="${TMPDIR:-/tmp}/prism-w4-cold-$$"
mkdir -p "$OUT"

# 先确保统一顶层入口二进制存在（不联网、只用 workspace 内源码）。
cargo build -q -p prism-app --bin prism

fail() { echo "FAIL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 零损失凭证：3 个既有 .prism 的 md5（与 W1/W2/W3 收口逐字节一致）
# ---------------------------------------------------------------------------
zero_loss() {
    echo "=== 零损失 3 个 .prism md5 ==="
    for name in spin_red_ball t0_media_mesh w2d_showcase; do
        M="$(md5sum "examples/$name.prism" | cut -d' ' -f1)"
        echo "  $name.prism  $M"
    done
    echo "  期望前缀: e75b03e4… / 46e29136… / 88f73550…"
    A="$(md5sum examples/spin_red_ball.prism | cut -c1-8)"
    B="$(md5sum examples/t0_media_mesh.prism | cut -c1-8)"
    C="$(md5sum examples/w2d_showcase.prism | cut -c1-8)"
    [ "$A" = "e75b03e4" ] || fail "spin_red_ball md5 漂移：$A"
    [ "$B" = "46e29136" ] || fail "t0_media_mesh md5 漂移：$B"
    [ "$C" = "88f73550" ] || fail "w2d_showcase md5 漂移：$C"
    echo "  3/3 逐字节一致：PASS"
}

# ---------------------------------------------------------------------------
# 空环境冷启动：PATH 清空（无 ffmpeg）、--fast（无 GPU）、内置 MJPEG（不重开 C2）
# ---------------------------------------------------------------------------
cold_start() {
    echo "=== 空环境冷启动（PATH=，无 ffmpeg / 无 GPU）==="
    env PATH="" "$BIN" render "$FILMS/analytic.prism" \
        --out "$OUT/cold.mp4" --codec mjpeg --fast > "$OUT/cold.log" 2>&1 || {
        cat "$OUT/cold.log"; fail "冷启动退出码非 0";
    }
    grep -q "帧数      : 180 帧" "$OUT/cold.log" || fail "帧数未对上 180"
    grep -q "32x24" "$OUT/cold.log" || fail "尺寸未对上 32x24"
    [ "$(od -An -tx1 -j4 -N4 "$OUT/cold.mp4" | tr -d ' \n')" = "66747970" ] || fail "MP4 ftyp 头缺失"
    echo "  退出码 0 / 180 帧 / 32x24 / MP4(ftyp) / mjpeg：PASS"
}

# ---------------------------------------------------------------------------
# 观感双档：analytic + crafted 各自一条命令出片、两跑逐字节
# ---------------------------------------------------------------------------
dual_finish() {
    echo "=== 观感双档两跑逐字节 ==="
    for name in analytic crafted; do
        env PATH="" "$BIN" render "$FILMS/$name.prism" \
            --out "$OUT/$name.1.mp4" --codec mjpeg --fast > /dev/null 2>&1 \
            || fail "$name 第 1 跑退出码非 0"
        env PATH="" "$BIN" render "$FILMS/$name.prism" \
            --out "$OUT/$name.2.mp4" --codec mjpeg --fast > /dev/null 2>&1 \
            || fail "$name 第 2 跑退出码非 0"
        cmp -s "$OUT/$name.1.mp4" "$OUT/$name.2.mp4" \
            || fail "$name 两跑逐字节不一致"
        SZ="$(wc -c < "$OUT/$name.1.mp4" | tr -d ' ')"
        echo "  $name: 两跑逐字节一致（$SZ B）：PASS"
    done
}

case "$MODE" in
    zero) zero_loss ;;
    cold) cold_start ;;
    dual) dual_finish ;;
    all)
        zero_loss
        cold_start
        dual_finish
        echo "W4 冷启动 / 单文件双档 / 零损失复现完成：PASS"
        ;;
    *) echo "用法: sh scripts/w4_cold.sh [all|cold|zero|dual]" >&2; exit 2 ;;
esac
