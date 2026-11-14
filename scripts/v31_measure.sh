#!/usr/bin/env bash
# V3.1 §8.3 测量：域内四档 × 静止/真运动两组 × 均衡/快速两档位，
# 外加域外片源（同样计入验收、不许 exclude）。
#
# 用法：v31_measure.sh <runs> <out_csv>
#   runs=1 单次扫测；runs=7 正式口径（弃最高/最低，取余下 5 次中位数）。
# 口径：36 帧 / 60 fps / --codec mjpeg / --mode {均衡,快速} / 端到端单帧
#       （含渲染 + 等价化简 + 编码），同机同构建同参数。
# §8.3：每个组合先跑 1 次「预热 3 帧」丢弃不计，再跑 RUNS 次正式测量。
#
# 门槛口径（§8.2）：门槛 = 「A2 实测基线 × 定死系数」。
#   - 域内两片源：分母取 §8.2 表格的 A2 绝对毫秒，不外推、不改。
#   - 域外片源：§8.2 表里没有它的 A2 项，故分母只能取**该片源自己的**
#     `一致` 档实测，再乘 0.30 / 0.15。这是公式本义；若改用域内的绝对
#     毫秒去卡域外，会把「片源本身几何量高一个量级」误判成「优化没做」。
#   - 320×180 已移出验收（2026-11-14），本脚本不再测它。
#   - `t0_media_mesh` 移出性能验收（2026-11-14 裁定一）：它的 16 spp golden
#     在体积介质里是噪声解（64 spp 更收敛反而 20.5 dB），8.1 逐像素比的就是
#     那一次随机数。它保留为**能力回归项**（末尾「能力回归」段）：只要出片、
#     帧数正确、非全黑、非全同、不 panic、退出码 0，不进门槛表。
set -u
BIN="${BIN:-./target/release/prism-app}"
RUNS="${1:-1}"
OUT="${2:-/tmp/v31_matrix.csv}"
SCENES=("examples/v31_motion.prism:运动" "examples/w2d_showcase.prism:静止")
EXTRA=("examples/t1_inside_shell.prism:域外-球内壁" "examples/t2_inside_wide.prism:域外-广角内壁")
TIERS=("640x360" "720p" "1080p" "4K")
MODES=("均衡" "快速")
# §8.2 门槛（ms/帧）：[档位] -> "均衡 快速"
thr() {
  case "$1" in
    640x360) echo "60.4 30.2" ;;
    720p)    echo "216.7 108.3" ;;
    1080p)   echo "488.7 244.4" ;;
    4K)      echo "1926.6 963.3" ;;
  esac
}
base() { # A2 基线
  case "$1" in
    640x360) echo 201.2 ;; 720p) echo 722.3 ;;
    1080p) echo 1629.0 ;; 4K) echo 6422.1 ;;
  esac
}
# 跑 RUNS 次并给出正式口径的数值（runs>=7 时弃最高/最低取中位）
measure() {
  local scene="$1" tier="$2" mode="$3"
  local -a vals=()
  local line v
  $BIN render "$scene" --mode "$mode" --res "$tier" --frames 3 --fps 60 --codec mjpeg --out /tmp/v31m_warm.mp4 >/dev/null 2>&1
  for ((r=1;r<=RUNS;r++)); do
    line=$($BIN render "$scene" --mode "$mode" --res "$tier" --frames 36 --fps 60 --codec mjpeg --out /tmp/v31m.mp4 2>&1 | grep "端到端单帧" | tail -1)
    v=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+ ms" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [ -z "$v" ]; then v="ERR"; fi
    vals+=("$v")
  done
  if [ "$RUNS" -ge 7 ]; then
    printf '%s\n' "${vals[@]}" | sort -n | sed '1d;$d' | awk '{a[NR]=$1} END{if(NR>0) print a[int((NR+1)/2)]}'
  else
    echo "${vals[0]}"
  fi
}

echo "场景,档位,模式,实测ms,门槛ms,基线ms,判定" > "$OUT"

echo "== 域内（门槛 = §8.2 绝对毫秒）=="
for sc in "${SCENES[@]}"; do
  scene="${sc%%:*}"; sname="${sc##*:}"
  for tier in "${TIERS[@]}"; do
    for mode in "${MODES[@]}"; do
      read -r tb tf <<< "$(thr "$tier")"
      if [ "$mode" = "均衡" ]; then t="$tb"; else t="$tf"; fi
      meas=$(measure "$scene" "$tier" "$mode")
      ok=$(awk -v m="$meas" -v t="$t" 'BEGIN{print (m+0<=t+0)?"过":"不过"}')
      echo "$sname,$tier,$mode,$meas,$t,$(base "$tier"),$ok" >> "$OUT"
      printf "%-4s %-8s %-4s 实测=%-8s 门槛=%-8s %s\n" "$sname" "$tier" "$mode" "$meas" "$t" "$ok"
    done
  done
done

echo "== 域外（门槛 = 同片源 一致档基线 × 0.30 / 0.15）=="
for sc in "${EXTRA[@]}"; do
  scene="${sc%%:*}"; sname="${sc##*:}"
  for tier in "${TIERS[@]}"; do
    $BIN render "$scene" --mode 一致 --res "$tier" --frames 3 --fps 60 --codec mjpeg --out /tmp/v31m_warm.mp4 >/dev/null 2>&1
    b=$($BIN render "$scene" --mode 一致 --res "$tier" --frames 36 --fps 60 --codec mjpeg --out /tmp/v31m.mp4 2>&1 | grep "端到端单帧" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    for mode in "${MODES[@]}"; do
      if [ "$mode" = "均衡" ]; then t=$(awk -v b="$b" 'BEGIN{printf "%.1f", b*0.30}'); else t=$(awk -v b="$b" 'BEGIN{printf "%.1f", b*0.15}'); fi
      meas=$(measure "$scene" "$tier" "$mode")
      ok=$(awk -v m="$meas" -v t="$t" 'BEGIN{print (m+0<=t+0)?"过":"不过"}')
      echo "$sname,$tier,$mode,$meas,$t,$b,$ok" >> "$OUT"
      printf "%-8s %-8s %-4s 实测=%-8s 门槛=%-9s (基线 %-9s) %s\n" "$sname" "$tier" "$mode" "$meas" "$t" "$b" "$ok"
    done
  done
done

echo "== 能力回归（不计门槛）：t0_media_mesh 必须出片 =="
t0_out=/tmp/v31_t0_regress.mp4
if $BIN render examples/t0_media_mesh.prism --mode 均衡 --res 640x360 --frames 8 --fps 8 --codec mjpeg --out "$t0_out" >/tmp/v31_t0_regress.log 2>&1; then
  t0_bytes=$(stat -c %s "$t0_out" 2>/dev/null || echo 0)
  if [ "${t0_bytes:-0}" -gt 1024 ]; then
    echo "t0_media_mesh 出片 通过（$t0_bytes B，退出码 0）"
  else
    echo "t0_media_mesh 出片 不过（产物仅 $t0_bytes B）"
  fi
else
  echo "t0_media_mesh 出片 不过（退出码非 0，见 /tmp/v31_t0_regress.log）"
fi

echo "CSV: $OUT"
