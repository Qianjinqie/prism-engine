#!/usr/bin/env bash
# ===========================================================================
# backup.sh — 并发写入纪律的落地脚本（防线 2）
#
# 事故背景（2026-10-06）：`crates/prism-io/src/raw.rs` 在 17:31 与 17:58 两次被
# **无人运行的写入**注入 254 行，触发 E0592；来源未定位（疑为客户端网络中断后
# 重试重放写入）。本脚本把"改前备份、改后核对"这条人工纪律变成可执行命令，
# 让"文件是不是被我改坏的"从一个需要回忆的问题变成一条可查的记录。
#
# 用法：
#   ./tools/backup.sh backup<file>        备份 + 记录 stat/md5/行数到台账
#   ./tools/backup.sh verify  <file> <baseline>
#                                         用 md5 + diff 判定**精确还原**
#   ./tools/backup.sh status              列出全部台账记录
#   ./tools/backup.sh --help
#
# 判定「精确还原」的标准（两条都满足才算还原）：
#   ① md5 完全一致 —— 逐字节相同；
#   ② `diff` 无任何输出 —— 行级也一致（含行序、空白）。
# 只满足①就足够判定"逐字节相同"，但仍跑②：md5 相同而 diff 有输出是不可能的
# （md5 已覆盖全部字节），保留②是为了在 md5 因故不可用时仍有第二道判据，
# 并且让"到底差在哪一行"在失败时能直接打印出来。
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER_DIR="$REPO_ROOT/tools/backups"

usage() {
  # 只打印文件顶部的注释块（到第一行非 '#' 为止），不把代码也打出来。
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
  exit 0
}

# --- 工具函数 ---------------------------------------------------------------

die() { echo "!! $*" >&2; exit 1; }

# 打印 md5；没有 md5sum 就退回 openssl。两者都没有则明确报错（不静默跳过）。
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl md5 "$1" | awk '{print $NF}'
  else
    die "既无 md5sum 也无 openssl，无法计算 md5"
  fi
}

line_count() { wc -l < "$1" | tr -d ' '; }

stat_of() {
  # Linux 与 macOS 的 stat 参数不同，这里只取 mtime 字段。
  stat -c '%y' "$1" 2>/dev/null || stat -f '%Sm' "$1" 2>/dev/null || echo "unknown"
}

# 台账里每条记录一行：`ISO时间<TAB>动作<TAB>文件<TAB>md5<TAB>行数`
ledger_file() { echo "$LEDGER_DIR/ledger.tsv"; }

# --- 命令：backup -----------------------------------------------------------

cmd_backup() {
  local file="${1:-}"
  [ -n "$file" ] || die "用法：backup <file>"
  [ -f "$file" ] || die "文件不存在：$file"

  mkdir -p "$LEDGER_DIR"
  local rel="$file"
  case "$rel" in /*) rel="${rel#"$REPO_ROOT"/}" ;; esac

  # 快照文件名把路径分隔符换成 __，避免目录穿越。
  local snap="$LEDGER_DIR/$(echo "$rel" | tr '/' '__').bak"
  cp -p "$file" "$snap"

  local sum lines
  sum="$(md5_of "$file")"
  lines="$(line_count "$file")"

  #追加台账（不覆盖历史——台账是流水，不是快照）
  printf '%s\tbackup\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')" \
    "$rel" "$sum" "$lines" "$(stat_of "$file")" >> "$(ledger_file)"

  echo "✓ 已备份 $rel"
  echo "    快照：tools/backups/$(basename "$snap")"
  echo "    md5：$sum"
  echo "    行数：$lines"
  echo "    mtime：$(stat_of "$file")"
  echo "改完后再跑一次 backup（会覆盖快照并追加台账）；若怀疑非本人改动，用："
  echo "    ./tools/backup.sh verify $rel tools/backups/$(basename "$snap")"
}

# --- 命令：verify -----------------------------------------------------------

cmd_verify() {
  local file="${1:-}" base="${2:-}"
  [ -n "$file" ] && [ -n "$base" ] || die "用法：verify <file> <baseline>"
  [ -f "$file" ] || die "文件不存在：$file"
  [ -f "$base" ] || die "基线不存在：$base"

  local f_md5 b_md5 f_lines b_lines
  f_md5="$(md5_of "$file")"
  b_md5="$(md5_of "$base")"
  f_lines="$(line_count "$file")"
  b_lines="$(line_count "$base")"

  local red=0 green=0
  if [[ "$f_md5" == "$b_md5" ]]; then
    echo "✓ md5 一致：$f_md5"
    green=$((green + 1))
  else
    echo "✗ md5 不一致："
    echo "    当前：$f_md5（$f_lines 行）"
    echo "    基线：$b_md5（$b_lines 行）"
    red=$((red + 1))
  fi

  # 第二道判据：行级 diff。md5 已覆盖字节，理论上不会到这里才失败；
  # 保留它是为了失败时能直接指出"差在哪一行"。
  if diff -q "$file" "$base" >/dev/null 2>&1; then
    echo "✓ diff 无差异（逐行一致）"
    green=$((green + 1))
  else
    echo "✗ diff 有差异，行级对照（前 20 行）："
    diff "$file" "$base" | head -20 | sed 's/^/    /'
    red=$((red + 1))
  fi

  if [[ $red -eq 0 ]]; then
    echo "结论：精确还原（$green/$green 项判据通过）"
    return 0
  fi
  echo "结论：未精确还原（$red 项不符）" >&2
  return 1
}

# --- 命令：status -----------------------------------------------------------

cmd_status() {
  local lf; lf="$(ledger_file)"
  if [[ ! -f "$lf" ]]; then
    echo "（尚无台账记录：$lf）"
    return 0
  fi
  echo "备份台账（时间 / 动作 / 文件 / md5 / 行数 / mtime）："
  sort "$lf" | awk -F'\t' '{printf "  %s  %-6s  %-40s  %s  %s\n", $1, $2, $3, $4, $5}'
}

# --- 分发 -------------------------------------------------------------------

case "${1:-}" in
  backup) shift; cmd_backup "${1:-}" ;;
  verify) shift; cmd_verify "${1:-}" "${2:-}" ;;
  status) cmd_status ;;
  ""|--help|-h) usage ;;
  *) die "未知命令：$1（用 --help 看用法）" ;;
esac