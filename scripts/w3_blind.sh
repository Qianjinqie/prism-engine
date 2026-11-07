#!/bin/sh
# V3.0 W3 盲测协议 —— 可复现脚本（§3.5 V3 / 主理人裁定）。
#
# 用法（workspace 根目录）：
#   sh scripts/w3_blind.sh             # 主判据 + 负控 + critic 侧证 + 两跑逐字节
#   sh scripts/w3_blind.sh report      # 只看主判据与 64 行逐对输出
#   sh scripts/w3_blind.sh control     # 反假绿负控
#   sh scripts/w3_blind.sh critic      # critic 五维画质侧证
#
# 判官是确定性可计算函数，样本池 / 种子 / 配对顺序 / 门槛全部冻结在
# crates/prism-app/src/w3_blind.rs 与 docs/v3.0-盲测协议.md，
# 本脚本不接收可调参数（不事后调门槛、不换尺子）。
# 零第三方依赖：只调 cargo + workspace 内 example。

set -eu

MODE="${1:-all}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 单次运行 ==="
if cargo run -q -p prism-app --example w3_blind -- "$MODE" > /tmp/prism_w3_blind_run1.txt; then
    RUN1_RC=0
else
    RUN1_RC=$?
fi
cat /tmp/prism_w3_blind_run1.txt

echo
echo "=== 两跑逐字节对拍（同命令再跑一次）==="
if cargo run -q -p prism-app --example w3_blind -- "$MODE" > /tmp/prism_w3_blind_run2.txt; then
    RUN2_RC=0
else
    RUN2_RC=$?
fi

if cmp -s /tmp/prism_w3_blind_run1.txt /tmp/prism_w3_blind_run2.txt; then
    echo "两跑输出逐字节一致：是"
else
    echo "两跑输出逐字节一致：否（FAIL）"
    exit 1
fi

if [ "$RUN1_RC" -ne 0 ] || [ "$RUN2_RC" -ne 0 ]; then
    echo "协议判定 FAIL（如实给数字，不改门槛）"
    exit 1
fi

echo "W3 盲测协议复现完成：PASS"
