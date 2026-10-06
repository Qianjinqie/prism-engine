#!/usr/bin/env bash
# ===========================================================================
# build_editor.sh — 产出 editor.html 用的 prism_editor.wasm（U13 交互式编辑器）。
#
# 与 build.sh（D-14 的 prism_gpu.wasm）**刻意分离**，理由有二：
#
#   1. **依赖方向**：编辑器的增量失效依赖 prism-core 的子图签名树，而 prism-gpu
#      只依赖 prism-math，反向依赖会成环（prism-gpu/src/scan.rs 模块头明确禁止）。
#      故 editor ABI 落在 prism-app（已依赖 prism-core），对 abi.rs 零改动——
#      也顺带避开与并行批次编辑 abi.rs 的冲突。
#   2. **互不干扰**：本脚本只写 prism_editor.wasm + 跑本脚本自带的门禁，
#      绝不碰 prism_gpu.wasm / index.html / realgpu_check.mjs（D-14 地基）。
#
# 产物路径：crates/prism-gpu/web/prism_editor.wasm
# 该文件与 editor.html 同目录，页面以相对路径 "prism_editor.wasm" 加载，
# 因此"发布为应用"时把 web/ 目录一起发布即可（单页 + 一个二进制）。
#
# 用法：
#   ./build_editor.sh              # release 体积优化版（推荐）
#   ./build_editor.sh --debug      # 带调试信息，便于排查
#   ./build_editor.sh --no-test    # 只编译，跳过门禁
#
# 验证链（--no-test 之外全跑，任一失败即构建失败）：
#   cargo build --lib → 拷贝产物 → gate_editor.mjs（导出齐全 / 零导入 /
#                      实例化自检 / 端到端编辑-撤销-渲染不变量）
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PROFILE="release"       # 与 build.sh 一致：默认出体积优化版
RUN_TEST=1

# 产物文件名：**唯一定义处**。editor.html 侧对应 WASM_URL 常量；两处若各写一份
# 字面量，改名就会只改一边、页面静默 404。
WASM_NAME="prism_editor.wasm"

for arg in "$@"; do
  case "$arg" in
    --debug)   PROFILE="dev" ;;
    --no-test) RUN_TEST=0 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

# cargo 的产物目录名规则：内置 profile `dev` 落在 **`debug/`**（不是 `dev/`）；
# `release` 落在 `release/`。这里显式推导，避免"编译成功但找不到文件"这种假失败。
# （踩过的坑：曾把 dev 映射到 `dev/`，--debug 必然找不到产物。）
profile_dir() {
  case "$1" in
    dev)     echo "debug" ;;
    release) echo "release" ;;
    *)       echo "$1" ;;
  esac
}

# 独立 target-dir：绝不污染 native 的 target/（build.sh 踩过这个坑）。
WASM_TARGET_DIR="$REPO/target/wasm32"
OUT_DIR="$WASM_TARGET_DIR/wasm32-unknown-unknown/$(profile_dir "$PROFILE")"
OUT_WASM="$HERE/$WASM_NAME"

echo "==> 编译 prism-app (lib/cdylib, target=wasm32-unknown-unknown, profile=$PROFILE)"
echo "    target-dir: $WASM_TARGET_DIR  （独立目录，不污染 native 构建）"
cd "$REPO"
# 注意必须带 --lib：binary 目标（main.rs）不引用 editor ABI，链接器会把导出全部
# 剥离，产出物里一个 `prism_editor_*` 符号都没有。
cargo build -p prism-app --lib \
  --target wasm32-unknown-unknown \
  --target-dir "$WASM_TARGET_DIR" \
  --profile "$PROFILE"

if [[ ! -f "$OUT_DIR/prism_app.wasm" ]]; then
  echo "!! cargo 未在预期目录产出 .wasm：$OUT_DIR/prism_app.wasm" >&2
  echo "   实际内容：" >&2
  ls -1 "$OUT_DIR/"*.wasm 2>/dev/null || echo "   （目录内无 .wasm）" >&2
  # 关键：产物缺失时**先删旧产物**再退出。否则 OUT_WASM 里留着上一次的陈旧
  # wasm，而门禁/页面仍会去加载它 —— 表现为"门禁全绿"却验的是旧二进制
  # （踩过的坑：--debug 失败后 release 门禁仍拿 release 旧产物报 48 项全过）。
  echo "   已删除陈旧产物 $OUT_WASM，避免下游误用。" >&2
  rm -f "$OUT_WASM"
  exit 1
fi
cp "$OUT_DIR/prism_app.wasm" "$OUT_WASM"
SIZE=$(wc -c < "$OUT_WASM")
echo "==> 产物：$OUT_WASM ($SIZE 字节)"

if [[ "$RUN_TEST" == "1" ]]; then
  if ! command -v node >/dev/null 2>&1; then
    echo "!! 未找到 node：跳过 wasm 门禁（仅编译通过）" >&2
    echo "   安装 node 后重跑本脚本可获得完整验证链。" >&2
  else
    echo "==> 门禁：gate_editor.mjs（导出齐全 / 零导入 / 端到端不变量）"
    node "$HERE/gate_editor.mjs" "$OUT_WASM"
  fi
fi

echo "==> 完成。打开 $HERE/editor.html 即可使用（需经 http(s) 提供，见文件头说明）。"
