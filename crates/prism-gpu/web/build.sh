#!/usr/bin/env bash
# ===========================================================================
# build.sh — 一条命令产出 harness 用的 .wasm（D-14 手机测试链路）。
#
# 关键点：wasm32 产物**必须**走独立 --target-dir。历史上踩过的坑：wasm32 的
# 编译产物混进 native 的 target/，导致后续 native 编译报
# `can't find crate prism_xxx`（两套 target 的元数据互相污染）。
#
# 产物路径：<repo>/crates/prism-gpu/web/prism_gpu.wasm
# 该文件与 index.html 同目录，harness 以相对路径 "prism_gpu.wasm" 加载，
# 因此"发布为应用"时整个 web/ 目录一起发布即可（单页 harness + 旁边一个
# 二进制，无其它运行时依赖）。
#
# 用法：
#   ./build.sh            # release 体积优化版（推荐，手机流量友好）
#   ./build.sh --debug    # 带调试信息，便于 wasm 侧排查
#   ./build.sh --no-test  # 只编译，跳过静态门禁与 node 验证
#
# 验证链（--no-test 之外全跑，任一失败即构建失败）：
#   cargo build → 拷贝产物 → wgsl_lint.mjs（静态门禁 + 非恒真自证）
#                → host_test.mjs（wasm 数值交叉判对）
# 静态门禁在最前：它秒级且零依赖，能在构建阶段就拦住 WGSL 对齐 / 胶水 async /
# 绑定偏移双重计算这类**真机才会暴露、但静态可查**的缺陷（D-14 教训）。
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PROFILE="wasm"          # 对应workspace 根 Cargo.toml 的 [profile.wasm]
RUN_TEST=1
# 产物文件名：**唯一定义处**。index.html 侧对应 WASM_URL 常量；两处若各写一份
# 字面量，改名就会只改一边、页面静默 404。门禁 wgsl_lint.mjs 规则 D4 断言
# 本文件（剥注释后）只出现这一次。
WASM_NAME="prism_gpu.wasm"

# cargo 的产物目录名规则：内置 profile `dev` 落在 `debug/` 目录；自定义
# profile `[profile.wasm]` 落在 `wasm/` 目录。这里显式推导，避免"编译成功但
# 找不到文件"这种假失败。
profile_dir() {
  case "$1" in
    dev)     echo "debug" ;;  # 内置 dev profile → debug/
    release) echo "release" ;;
    *)       echo "$1" ;;     # 自定义 profile → 同名目录
  esac
}

for arg in "$@"; do
  case "$arg" in
    --debug)   PROFILE="dev" ;;
    --no-test) RUN_TEST=0 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

# 独立 target-dir：绝不污染 native 的 target/。
WASM_TARGET_DIR="$REPO/target/wasm32"
OUT_DIR="$REPO/target/wasm32/wasm32-unknown-unknown/$(profile_dir "$PROFILE")"
OUT_WASM="$HERE/$WASM_NAME"

echo "==> 编译 prism-gpu (target=wasm32-unknown-unknown, profile=$PROFILE)"
echo "    target-dir: $WASM_TARGET_DIR  （独立目录，不污染 native 构建）"
cd "$REPO"
cargo build -p prism-gpu \
  --target wasm32-unknown-unknown \
  --target-dir "$WASM_TARGET_DIR" \
  --profile "$PROFILE"

if [[ ! -f "$OUT_DIR/$WASM_NAME" ]]; then
  echo "!! cargo 未在预期目录产出 .wasm：$OUT_DIR/$WASM_NAME" >&2
  echo "   实际内容：" >&2
  ls -1 "$REPO/target/wasm32/wasm32-unknown-unknown/" >&2 || true
  exit 1
fi
cp "$OUT_DIR/$WASM_NAME" "$OUT_WASM"
SIZE=$(wc -c < "$OUT_WASM")
echo "==> 产物：$OUT_WASM（$SIZE 字节）"

if [[ "$RUN_TEST" == "1" ]]; then
  if ! command -v node >/dev/null 2>&1; then
    echo "!! 未找到 node：跳过本地验证（安装 node 后跑 node $HERE/host_test.mjs）" >&2
  else
    # ---- 静态门禁先行（D-14 真机缺陷本地化）----
    # 顺序理由：门禁是纯静态检查（秒级、零依赖），必须在编译产物验证之前跑完，
    # 这样 WGSL 对齐 / 胶水 async / 偏移双重计算这类问题在构建阶段就红灯，
    # 不必等真机。门禁失败 ⇒ 构建失败（set -e 之外的显式 exit，见下）。
    echo "==> 静态门禁（wgsl_lint.mjs：WGSL 16 字节对齐 / 胶水同步 import / 绑定偏移）"
    if ! node "$HERE/wgsl_lint.mjs"; then
      echo "!! 静态门禁未通过 ⇒ 构建判定失败（这三类缺陷都是静态可查的，不该等真机）" >&2
      exit 1
    fi
    # 门禁自证：确认规则本身非恒真（故意违反的样例必须被抓到）。
    node "$HERE/wgsl_lint.mjs" --self-test

    echo "==> node 宿主验证（模块加载 / 内存读写 / CPU 参考交叉判对）"
    node "$HERE/host_test.mjs" "$OUT_WASM"
  fi
fi

echo "==> 完成。用浏览器打开 $HERE/index.html 即可在真机（含手机）上测。"
