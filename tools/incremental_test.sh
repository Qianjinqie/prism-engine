#!/usr/bin/env bash
# 增量回归：从 BASE 到 HEAD 的变更 crate 集合 + 反向依赖闭包，只跑受影响 crate 的测试。
# 用法：
#   tools/incremental_test.sh [BASE] [--dry-run] [-- extra cargo test 参数]
#   BASE 默认 HEAD~1；--dry-run 只列受影响的 crate 不执行。
# 退出码：0 = 通过或无受影响 crate；1 = 测试失败；2 = 需要全量（根级变更，脚本不代为决定）。
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BASE="HEAD~1"
DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) BASE="$1"; shift ;;
  esac
done

# 1. 变更文件分两类：crates/<name>/ 内的走剪枝；根级构建配置变更要求全量。
CHANGED_CRATES="$(git diff --name-only "$BASE" HEAD -- 'crates/*' | awk -F/ '{print $2}' | sort -u || true)"
ROOT_LEVEL="$(git diff --name-only "$BASE" HEAD -- 'Cargo.toml' 'Cargo.lock' 'rust-toolchain*' '.cargo/*' || true)"
if [ -n "$ROOT_LEVEL" ]; then
  echo "[incremental] 检出根级构建配置变更："
  echo "$ROOT_LEVEL" | sed 's/^/  /'
  echo "[incremental] 依赖图可能全变，请跑全量：cargo test --workspace"
  exit 2
fi
if [ -z "$CHANGED_CRATES" ]; then
  echo "[incremental] 无 crate 级变更（仅 docs/examples/工具链外围），无需跑测试"
  exit 0
fi

# 2. 反向依赖闭包：cargo metadata 拿 workspace 成员间依赖，迭代扩张到不动点。
AFFECTED="$(python3 - "$CHANGED_CRATES" <<'PY'
import json, subprocess, sys

changed = set()
# 变更 crate 列表由 bash 以**单个多行字符串**传入（"$CHANGED_CRATES"），
# 故按空白切分；同时兼容逐参数传入（argv 多元素）。
for arg in sys.argv[1:]:
    changed.update(part for part in arg.split() if part)
meta = json.loads(subprocess.run(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"],
    capture_output=True, text=True, check=True).stdout)

members = {p["name"] for p in meta["packages"]}
# dep name -> 依赖它的成员集合（仅 path 依赖的 workspace 内部边）
dependents: dict[str, set[str]] = {m: set() for m in members}
for p in meta["packages"]:
    for d in p.get("dependencies", []):
        # --no-deps 下依赖路径可见；仅统计能解析到成员名的内部边
        if d["name"] in members:
            dependents[d["name"]].add(p["name"])

affected = set(c for c in changed if c in members)
missing = set(c for c in changed if c not in members)
if missing:
    print("[incremental] 警告：变更路径不在 workspace 成员内: " + ", ".join(sorted(missing)), file=sys.stderr)

frontier = list(affected)
while frontier:
    cur = frontier.pop()
    for dep in dependents.get(cur, ()):
        if dep not in affected:
            affected.add(dep)
            frontier.append(dep)

print(" ".join(sorted(affected)))
PY
)"

if [ -z "$AFFECTED" ]; then
  echo "[incremental] 变更均不在 workspace 成员内，无需跑测试"
  exit 0
fi

PKG_ARGS=()
for c in $AFFECTED; do PKG_ARGS+=(-p "$c"); done

echo "[incremental] BASE=$BASE"
echo "[incremental] 变更 crate : $(echo $CHANGED_CRATES | tr ' ' ',')"
echo "[incremental] 影响闭包   : $(echo $AFFECTED | tr ' ' ',')"
if [ "$DRY" -eq 1 ]; then
  echo "[incremental] dry-run，未执行。将跑: cargo test ${PKG_ARGS[*]}"
  exit 0
fi

cargo test "${PKG_ARGS[@]}" "${EXTRA[@]}"
