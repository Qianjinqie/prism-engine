//! 扫描线光栅核心：一次全高 y 扫描 + 解析精确覆盖率。
//!
//! # 算法（一次扫描贯穿整张画布，而不是逐行带重来）
//!
//! 1. **边规范化**：每条轮廓边裁到画布 y 范围 `[0, height]`，表示为
//!    `x(y) = xlo + (y − lo)·k`（`lo < hi`，附 winding 符号 ±1）。水平边丢弃
//!    （横射线模型下不改变 winding）。
//! 2. **事件 = 段端点 ∪ 段间交点**（全局只算一次，x 区间剪枝）。**列折点不是
//!    事件**：区域 `[xl(y), xr(y))` 在一个事件区间内两端都是 y 的线性函数，它与
//!    像素列 `[c, c+1)` 的交宽随 y 分段线性，折点在列循环里解析求出即可——把
//!    每条横跨 3800 px 的边展开成 3800 个「列事件」是旧实现子带爆炸的根源。
//! 3. **跨越序增量维护**：活跃边按 x 排成数组；事件处只做「删除已结束 / 插入
//!    新入场」。定序点取**事件区间中点**而不是事件本身：交点处两条边的 x 恰好
//!    相等，按事件 y 排序会把次序钉死在末位舍入决定的一侧。
//! 4. **区域批量积分**：区域 `[xl(y), xr(y))` 在 `[p, q]` 上，
//!    - **内部列**（整列在整个区间被覆盖）贡献恒为 `q − p` → 一次 `+= h` 的
//!      向量化区间加，O(1) 个区间而不是逐列算 min/max；
//!    - **两个边界列**用 `S(X, t) = ∫ (X(y) − t)⁺ dy` 的解析原函数求精确重叠
//!      （三角/梯形，纯算术，无采样无量化）。
//!
//! 每行的代价是 O(活跃跨越数)，而不是 O(子带数 × 活跃跨越数)。交错轮廓
//! （winding=2）按集合测度计一次，不重复计数。
//!
//! # 正确性
//! 覆盖率是区域 `{winding ≠ 0}` 的**精确面积积分**：没有子采样/量化/MSAA 近似。
//! 满覆盖列贡献恰为 `q − p`（整行满覆盖时 `1.0` 逐位精确）；边界列走解析原函数。
//! 几何插值、交点、面积原函数都在 f64 里算（顶点仍是调用方给的 f32）：f32 的
//! 唯一一次舍入落在「写回覆盖率缓冲」那一步，所以对同一批 f32 顶点，本实现的
//! 误差约为 1 ulp，而逐行带 f32 旧实现约为 `ulp(坐标)` 量级（1080p 下 5e-5）。
//!
//! # 确定性
//! 只用 ±×÷、`floor`/`ceil` 与比较（禁 libm）；排序键 `(x, k, id)` 全序
//! （`f64::total_cmp`）；无隐藏状态、无随机源。同输入同字节输出；整数平移下
//! 逐位等价（所有运算在平移下是同一组操作）。
//!
//! # 复杂度
//! 预处理 `O(E log E + P)`（`P` = x 区间重叠的候选边对数），积分
//! `O(H·A + C)`（`E` 边数、`H` 行数、`A` 每行活跃跨越数、`C` 覆盖列总数）。
//! 旧实现是 `O(Σ_行带 子带数 × 活跃边²)`，子带数被列折点放大到每行数百个。

use crate::geom::Point;
use crate::Coverage;

/// 填充规则。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    /// 非零 winding（TrueType 字形标准规则）。
    Winding,
    /// 奇偶（交替）。
    EvenOdd,
}

/// 裁到画布 y 范围的边：`x(y) = xlo + (y − lo)·k`，`lo < hi`。
///
/// 几何量用 f64 存：f32 顶点在 1080p/4K 坐标上的 ulp 是 1e-4 量级，用 f32 做
/// 插值会把光栅化误差放大到同一量级；f64 把误差压到顶点量化以下（顶点本身
/// 仍是调用方给的 f32，语义不变）。
///
/// 热字段只留 `(xlo, lo, k, sign)`：`accumulate` 的内层循环每处理一个跨越就要
/// 读一遍，结构体大小直接决定它能否留在 L1（4K 星形有数百条活跃边）。
/// `hi`/`xmin`/`xmax` 只在扫描预处理阶段用，拆到 [`EdgeMeta`] 里，不进热路径。
#[derive(Clone, Copy)]
struct Edge {
    /// `y = lo` 处的 x。
    xlo: f64,
    /// 段顶 y（含）。
    lo: f64,
    /// 斜率 dx/dy。
    k: f64,
    /// winding 符号：+1 向下、−1 向上（整数，避免浮点前缀和累积）。
    sign: i32,
}

/// 边的冷字段：段底、x 极值（只用于事件生成与交点剪枝）。
#[derive(Clone, Copy)]
struct EdgeMeta {
    /// 段底 y（不含）。
    hi: f64,
    /// x 下界。
    xmin: f64,
    /// x 上界。
    xmax: f64,
}

impl Edge {
    /// 边在 `y` 处的 x。
    #[inline]
    fn x_at(&self, y: f64) -> f64 {
        self.xlo + (y - self.lo) * self.k
    }
}

/// 活跃跨越表的一项：**按值**存边的热字段（32 B），配 `id` 用于回查冷字段。
///
/// 跨越表是 `accumulate` 里顺序扫的数组；按值存让它变成顺序读（预取友好），
/// 不必每次顺着 `edges[idx]` 做一次随机间接寻址。
#[derive(Clone, Copy)]
struct Crossing {
    xlo: f64,
    lo: f64,
    k: f64,
    sign: i32,
    id: u32,
}

impl Crossing {
    #[inline]
    fn x_at(&self, y: f64) -> f64 {
        self.xlo + (y - self.lo) * self.k
    }
}

/// `∫ (u(y))⁺ dy` 的解析原函数展开：`u` 在区间上由 `u0` 线性变到 `u1`，`h = q − p`。
///
/// 两端都非负 → 梯形 `(u0+u1)·h/2`（这是满覆盖/精确断言的路径，逐位保持旧式）。
/// 一端为负 → 正部是三角形，面积 `pos²/(2·(pos − neg))`；而 `pos − neg = |u0 − u1|`
/// 在同一段直线上是**常数**，于是把 `fac = h/(2·|u0 − u1|)` 提到循环外，
/// 每列只剩 `fac·pos²`——没有除法，也几乎没有分支。
#[inline]
fn pos_area(u0: f64, u1: f64, h: f64, fac: f64) -> f64 {
    let trap = (u0 + u1) * 0.5 * h;
    let p = if u0 > u1 { u0 } else { u1 };
    let pp = if p > 0.0 { p } else { 0.0 };
    let tri = fac * pp * pp;
    if (if u0 < u1 { u0 } else { u1 }) >= 0.0 {
        trap
    } else {
        tri
    }
}

/// 一段直线上「三角形模式」的常数因子 `h/(2·|Δx|)`；`Δx = 0`（竖直边）时取 0，
/// 使退化情形自动得到 0 而不是 inf·0。
///
/// 生产路径现在预算 `half_h = 0.5·h` 后调用 [`tri_factor_with_half_h`]；本包装
/// 保留原三参数口径，供冻结参考与逐位对拍使用。非测试构建里只有对拍用得到，
/// 故显式关掉该目标的 dead_code 警告，不把「测试辅助」伪装成热路径依赖。
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
fn tri_factor(xp: f64, xq: f64, h: f64) -> f64 {
    tri_factor_with_half_h(xp, xq, 0.5 * h)
}

/// [`tri_factor`] 的「`half_h = 0.5·h` 已预算」内核。
///
/// 原式 `0.5 * h / |xp − xq|` 在 Rust 中本就是 `(0.5*h)/|...|`；这里只是把
/// 公共乘法提到一个事件区间之外。同一 `h` 下没有改变任何操作数或结合顺序，
/// 因而逐字节一致（见 `boundary_chain_is_byte_identical_to_frozen_accumulate`）。
#[inline]
fn tri_factor_with_half_h(xp: f64, xq: f64, half_h: f64) -> f64 {
    let den = xp - xq;
    if den == 0.0 {
        0.0
    } else {
        half_h / (if den < 0.0 { -den } else { den })
    }
}

/// 边界链单步：左翼 / 右翼 / 窄区三种形状共用同一套「2 列展开 + 尾巴」骨架。
///
/// 关键纪律：每步仍逐次 `-= 1.0`、逐次调用 `pos_area`、按原括号顺序更新
/// `row[c]`，不做 `u -= 4.0` 之类的合并。否则 f64 减法舍入路径会变，逐位
/// 判据必须能抓到；这里只省循环分支与重复代码，不动任何一项算术。
macro_rules! boundary_chain_step {
    (left, $row:ident, $c:ident, $h:expr, $u0:ident, $u1:ident, $fac:expr, $s_prev:ident) => {{
        $u0 -= 1.0;
        $u1 -= 1.0;
        let s_next = pos_area($u0, $u1, $h, $fac);
        $row[$c as usize] += ($h - $s_prev + s_next) as f32;
        $s_prev = s_next;
    }};
    (right, $row:ident, $c:ident, $h:expr, $u0:ident, $u1:ident, $fac:expr, $s_prev:ident) => {{
        $u0 -= 1.0;
        $u1 -= 1.0;
        let s_next = pos_area($u0, $u1, $h, $fac);
        $row[$c as usize] += ($s_prev - s_next) as f32;
        $s_prev = s_next;
    }};
    (narrow, $row:ident, $c:ident, $h:expr, $lu0:ident, $lu1:ident, $ru0:ident, $ru1:ident, $fac_l:expr, $fac_r:expr, $sl:ident, $sr:ident) => {{
        $lu0 -= 1.0;
        $lu1 -= 1.0;
        $ru0 -= 1.0;
        $ru1 -= 1.0;
        let sln = pos_area($lu0, $lu1, $h, $fac_l);
        let srn = pos_area($ru0, $ru1, $h, $fac_r);
        $row[$c as usize] += (($sr - srn) - ($sl - sln)) as f32;
        $sl = sln;
        $sr = srn;
    }};
}

/// 生成边界列链：左右翼一次处理 2 列；窄区双链保持逐列推进（双链寄存器
/// 压力更高，实测 2/2 列展开都变慢，合并到同一骨架才是收益点）。所有形态
/// 的 `c` 每次只加 1，与旧 `while c <= b` / `loop { …; c += 1 }` 完全同序。
macro_rules! boundary_chain {
    (left, $row:expr, $a:expr, $b:expr, $h:expr, $u0:ident, $u1:ident, $fac:expr, $s_prev:ident) => {{
        let row = $row;
        let a: i64 = $a;
        let b: i64 = $b;
        let h: f64 = $h;
        let fac: f64 = $fac;
        let mut c: i64 = a;
        while c + 1 <= b {
            boundary_chain_step!(left, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
            boundary_chain_step!(left, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
        }
        while c <= b {
            boundary_chain_step!(left, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
        }
    }};
    (right, $row:expr, $a:expr, $b:expr, $h:expr, $u0:ident, $u1:ident, $fac:expr, $s_prev:ident) => {{
        let row = $row;
        let a: i64 = $a;
        let b: i64 = $b;
        let h: f64 = $h;
        let fac: f64 = $fac;
        let mut c: i64 = a;
        while c + 1 <= b {
            boundary_chain_step!(right, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
            boundary_chain_step!(right, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
        }
        while c <= b {
            boundary_chain_step!(right, row, c, h, $u0, $u1, fac, $s_prev);
            c += 1;
        }
    }};
    (narrow, $row:expr, $a:expr, $b:expr, $h:expr, $lu0:ident, $lu1:ident, $ru0:ident, $ru1:ident, $fac_l:expr, $fac_r:expr, $sl:ident, $sr:ident) => {{
        let row = $row;
        let a: i64 = $a;
        let b: i64 = $b;
        let h: f64 = $h;
        let fac_l: f64 = $fac_l;
        let fac_r: f64 = $fac_r;
        let mut c: i64 = a;
        while c <= b {
            boundary_chain_step!(narrow, row, c, h, $lu0, $lu1, $ru0, $ru1, fac_l, fac_r, $sl, $sr);
            c += 1;
        }
    }};
}

/// 左翼边界列：`cov(c) = h − S(L,c) + S(L,c+1)`，2 列展开。
#[inline]
fn add_left_wing(
    row: &mut [f32],
    a: i64,
    b: i64,
    lp: f64,
    lq: f64,
    h: f64,
    half_h: f64,
) {
    let fac = tri_factor_with_half_h(lp, lq, half_h);
    let mut u0 = lp - a as f64;
    let mut u1 = lq - a as f64;
    let mut s_prev = pos_area(u0, u1, h, fac);
    boundary_chain!(left, row, a, b, h, u0, u1, fac, s_prev);
}

/// 右翼边界列：`cov(c) = S(R,c) − S(R,c+1)`，2 列展开。
#[inline]
fn add_right_wing(
    row: &mut [f32],
    a: i64,
    b: i64,
    rp: f64,
    rq: f64,
    h: f64,
    half_h: f64,
) {
    let fac = tri_factor_with_half_h(rp, rq, half_h);
    let mut u0 = rp - a as f64;
    let mut u1 = rq - a as f64;
    let mut s_prev = pos_area(u0, u1, h, fac);
    boundary_chain!(right, row, a, b, h, u0, u1, fac, s_prev);
}

/// 窄区 / 边界翼合并路径：同一「2 列展开」骨架一次推进左右两条原函数链。
#[inline]
fn add_narrow_columns(
    row: &mut [f32],
    a: i64,
    b: i64,
    lp: f64,
    lq: f64,
    rp: f64,
    rq: f64,
    h: f64,
    half_h: f64,
) {
    let fac_l = tri_factor_with_half_h(lp, lq, half_h);
    let fac_r = tri_factor_with_half_h(rp, rq, half_h);
    let mut lu0 = lp - a as f64;
    let mut lu1 = lq - a as f64;
    let mut ru0 = rp - a as f64;
    let mut ru1 = rq - a as f64;
    let mut sl = pos_area(lu0, lu1, h, fac_l);
    let mut sr = pos_area(ru0, ru1, h, fac_r);
    boundary_chain!(narrow, row, a, b, h, lu0, lu1, ru0, ru1, fac_l, fac_r, sl, sr);
}

/// `a < b` 的 NaN 安全写法：任一侧是 NaN 时返回 `false`。
///
/// 等价于 `a < b`（IEEE 语义下 NaN 参与的比较恒假），但写成函数是为了让
/// 「取反后跳过」的意图不必用 `!(a > b)` 表达——那种写法对偏序类型是坏味道
/// （clippy `neg_cmp_op_on_partial_ord`），而这里恰恰需要「NaN → 不进分支」。
#[inline]
fn lt(a: f64, b: f64) -> bool {
    a.partial_cmp(&b) == Some(core::cmp::Ordering::Less)
}

/// `a > b` 的 NaN 安全写法：任一侧是 NaN 时返回 `false`。
#[inline]
fn gt(a: f64, b: f64) -> bool {
    a.partial_cmp(&b) == Some(core::cmp::Ordering::Greater)
}

/// `row[a..=b] += v`（列下标钳到 `[0, last]`；空区间直接返回）。
///
/// 循环体是「常量加到整段」，编译器可向量化——这是内部列的 O(1) 大手笔。
#[inline]
fn add_span(row: &mut [f32], a: i64, b: i64, v: f32, last: i64) {
    let a = a.max(0);
    let b = b.min(last);
    if b < a {
        return;
    }
    for x in &mut row[a as usize..=(b as usize)] {
        *x += v;
    }
}

/// 把一个事件区间 `[p, q]` 的全部覆盖区域精确累加进 `row`（当前行的覆盖率行）。
///
/// 跨越序 `order` 在该区间内恒定；`row` 长度即画布宽。边界列已合并为统一的
/// 2 列展开链，且 `half_h` 在区间级预算一次；算术顺序与冻结参考逐句一致。
#[inline]
fn accumulate(order: &[Crossing], p: f64, q: f64, rule: FillRule, row: &mut [f32]) {
    let n = order.len();
    if n < 2 {
        return;
    }
    let h = q - p;
    if !gt(h, 0.0) {
        return; // h ≤ 0 或 NaN
    }
    let last = row.len() as i64 - 1;
    let hv = h as f32;
    let half_h = 0.5 * h;

    #[cfg(debug_assertions)]
    {
        // 区间内跨越序必须非降（区间内部无交点，交点都是事件）。
        let pm = (p + q) * 0.5;
        for w in order.windows(2) {
            let a = w[0].x_at(pm);
            let b = w[1].x_at(pm);
            assert!(a <= b, "跨越序非单调 @[{p},{q}] pm={pm}: {a} > {b}");
        }
    }

    let evenodd = rule == FillRule::EvenOdd;
    let mut winding: i32 = 0;
    // 左→右游走：跨越 k 与 k+1 之间区域左侧的 winding = 前 k+1 条跨越的符号和
    // （闭合轮廓保证总和为 0）。奇偶规则只看右侧跨越数的奇偶。
    // 下一个区域的左边界就是本区域的右边界，x 值直接接力（省一半插值）。
    let mut lp = order[0].x_at(p);
    let mut lq = order[0].x_at(q);
    let mut k = 0usize;
    while k + 1 < n {
        let rp = order[k + 1].x_at(p);
        let rq = order[k + 1].x_at(q);
        winding += order[k].sign;
        let inside = if evenodd { ((n - 1 - k) & 1) == 1 } else { winding != 0 };
        k += 1;
        if inside {
            // 区间内两端各自的 x 极值（线性 → 极值必在端点）。
            let llo = if lp < lq { lp } else { lq };
            let lhi = if lp < lq { lq } else { lp };
            let rlo = if rp < rq { rp } else { rq };
            let rhi = if rp < rq { rq } else { rp };
            if rhi > llo {
                // 整列被覆盖的列区间 [b0, b1]：
                //   左端要求 c ≥ max xl(y)，右端要求 c+1 ≤ min xr(y)。
                let b0 = lhi.ceil() as i64;
                let b1 = (rlo.floor() as i64).saturating_sub(1);
                if b1 >= b0 {
                    // 内部列：每列在整个区间内满覆盖 → 贡献恒为 h。
                    add_span(row, b0, b1, hv, last);

                    // 左翼列 [floor(llo), b0 − 1]：右侧被完全覆盖，
                    // cov(c) = h − S(L, c) + S(L, c+1)。与右翼 / 窄区共用展开链。
                    let fa = (llo.floor() as i64).max(0);
                    let fb = (b0 - 1).min(last);
                    if fb >= fa {
                        add_left_wing(row, fa, fb, lp, lq, h, half_h);
                    }

                    // 右翼列 [b1 + 1, floor(rhi)]：左侧被完全覆盖，
                    // cov(c) = S(R, c) − S(R, c+1)。与左翼 / 窄区共用展开链。
                    let ga = (b1 + 1).max(0);
                    let gb = (rhi.floor() as i64).min(last);
                    if gb >= ga {
                        add_right_wing(row, ga, gb, rp, rq, h, half_h);
                    }
                } else {
                    // 窄区（宽度不足一整列，没有内部列）：与边界翼合并为同一链，
                    //   cov(c) = [S(R,c) − S(R,c+1)] − [S(L,c) − S(L,c+1)]。
                    let a = (llo.min(rlo).floor() as i64).max(0);
                    let b = (rhi.max(lhi).floor() as i64).min(last);
                    if b >= a {
                        add_narrow_columns(row, a, b, lp, lq, rp, rq, h, half_h);
                    }
                }
            }
        }
        lp = rp;
        lq = rq;
    }
}

/// 跨越序比较：按「区间内某点 y 处 x」升序，并列时按斜率、再按 id。
#[inline]
fn order_less(a: &Crossing, b: &Crossing, y: f64) -> bool {
    let xa = a.x_at(y);
    let xb = b.x_at(y);
    match xa.total_cmp(&xb) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => match a.k.total_cmp(&b.k) {
            core::cmp::Ordering::Less => true,
            core::cmp::Ordering::Greater => false,
            core::cmp::Ordering::Equal => a.id < b.id,
        },
    }
}

/// 对近乎有序的活跃数组做插入排序（事件之间次序只可能局部变化）。
#[inline]
fn sort_active(order: &mut [Crossing], y: f64) {
    let mut i = 1usize;
    while i < order.len() {
        let v = order[i];
        let mut j = i;
        while j > 0 && order_less(&v, &order[j - 1], y) {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = v;
        i += 1;
    }
}

/// 当前事件区间 `[open_y, 下一个事件)` 的中点（没有下一个事件时用画布底边）。
///
/// 定序必须落在**区间内部**：区间内跨越序恒定，而端点处两条相交边的 x 相等，
/// 取哪一侧只剩末位舍入信息。
#[inline]
fn mid_of(open_y: f64, ev_idx: usize, events: &[f64], height: f64) -> f64 {
    let next = if ev_idx < events.len() { events[ev_idx] } else { height };
    (open_y + next) * 0.5
}

/// 事件排序（f64 全序）。小数组走比较排序；> [`RADIX_THRESHOLD`] 走**位模式
/// LSD 基数排序**（8 趟 × 8 bit），把 `O(n log n)` 的分支比较换成 `O(n)` 的
/// 计数分配。键是 `f64::to_bits` 的单调映射，排序结果与
/// `sort_unstable_by(f64::total_cmp)` **逐位相同**（含 ±0、±inf、NaN 全序）。
const RADIX_THRESHOLD: usize = 10_000;

/// 把 f64 位型映射成「数值升序 == `total_cmp` 升序」的 u64 键。
#[inline]
fn f64_ordered_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b & (1u64 << 63) != 0 {
        !b // 负数：整体取反（越负越小）
    } else {
        b ^ (1u64 << 63) // 正数：只翻符号位
    }
}

/// [`f64_ordered_key`] 的逆变换。
#[inline]
fn f64_from_ordered_key(k: u64) -> f64 {
    f64::from_bits(if k & (1u64 << 63) != 0 {
        k ^ (1u64 << 63)
    } else {
        !k
    })
}

/// 对事件数组按 f64 全序排序（结果等价于 `sort_unstable_by(total_cmp)`）。
fn sort_events(v: &mut Vec<f64>) {
    if v.len() <= RADIX_THRESHOLD {
        v.sort_unstable_by(f64::total_cmp);
        return;
    }
    // NaN 的 total_cmp 位序在某些实现上有 payload 细节；几何事件不含 NaN，
    // 但为「绝不改判据」起见，检测到 NaN 直接退回权威比较排序。
    if v.iter().any(|x| x.is_nan()) {
        v.sort_unstable_by(f64::total_cmp);
        return;
    }
    let n = v.len();
    let mut keys: Vec<u64> = v.iter().map(|&x| f64_ordered_key(x)).collect();
    let mut tmp: Vec<u64> = vec![0u64; n];
    let mut counts = [0u32; 256];
    for pass in 0..8u32 {
        counts.fill(0);
        for &k in keys.iter() {
            counts[((k >> (pass * 8)) & 0xff) as usize] += 1;
        }
        let mut sum = 0u32;
        for c in counts.iter_mut() {
            let cc = *c;
            *c = sum;
            sum += cc;
        }
        for &k in keys.iter() {
            let b = ((k >> (pass * 8)) & 0xff) as usize;
            tmp[counts[b] as usize] = k;
            counts[b] += 1;
        }
        core::mem::swap(&mut keys, &mut tmp);
    }
    // 8 趟（偶数）后结果回到 `keys`。
    for (dst, &k) in v.iter_mut().zip(keys.iter()) {
        *dst = f64_from_ordered_key(k);
    }
}

/// 光栅化闭合轮廓集合 → 每像素覆盖率。
///
/// 轮廓视为闭合环（首尾自动相连，无需重复末点）；坐标为设备像素，
/// 输出按 `width × height` 裁剪。等价于构造空缓冲后调用 [`rasterize_into`]。
pub fn rasterize(contours: &[Vec<Point>], rule: FillRule, width: u32, height: u32) -> Coverage {
    let mut cov = Coverage::new(width, height);
    rasterize_into(contours, rule, width, height, &mut cov);
    cov
}

/// 光栅化到调用方提供的 [`Coverage`]（**复用其 `data` 分配**）。
///
/// 只增不改 [`rasterize`] 的语义：算法主体逐句不变，区别仅在输出缓冲是复用
/// `out.data` 的已驻留页，而不是每帧 `vec![0.0; w*h]` 重新分配并触发大缓冲
/// 缺页（星 512@4K 的 67 MB 输出分配占旧实现约 43% 墙钟）。逐帧路径
/// （`prism-postfx` 的 PathMask 复用池）用本入口消除该分配；结果与
/// `rasterize` **逐字节一致**（同一 `clear+零填充` + 同一累加序）。
pub fn rasterize_into(
    contours: &[Vec<Point>],
    rule: FillRule,
    width: u32,
    height: u32,
    out: &mut Coverage,
) {
    // 复用容量：clear 后 resize(n, 0.0) 只在尺寸变化时扩容，命中已驻留页。
    out.width = width;
    out.height = height;
    let n = (width as usize).saturating_mul(height as usize);
    out.data.clear();
    out.data.resize(n, 0.0);
    let cov = out; // 算法主体保持原样（读写 cov.data）

    if width == 0 || height == 0 || contours.is_empty() {
        return;
    }
    let height_f = height as f64;

    // ---- 1. 轮廓边 → 裁到画布 y 范围的直线段 ----
    let mut count = 0usize;
    for c in contours {
        if c.len() >= 3 {
            count += c.len();
        }
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(count);
    let mut meta: Vec<EdgeMeta> = Vec::with_capacity(count);
    for c in contours {
        let n = c.len();
        if n < 3 {
            continue; // 少于 3 点不构成有面积轮廓
        }
        for i in 0..n {
            let a = c[i];
            let b = c[(i + 1) % n];
            if a.y == b.y {
                continue; // 水平边不改变 winding
            }
            let (ax, ay) = (a.x as f64, a.y as f64);
            let (bx, by) = (b.x as f64, b.y as f64);
            let dy = by - ay;
            let dx = bx - ax;
            let (mut lo, mut hi) = if ay < by { (ay, by) } else { (by, ay) };
            if lo < 0.0 {
                lo = 0.0;
            }
            if hi > height_f {
                hi = height_f;
            }
            if !lt(lo, hi) {
                continue; // 完全在画布上下之外（NaN 也走这里）
            }
            let k = dx / dy;
            let xlo = ax + ((lo - ay) / dy) * dx;
            let xhi = ax + ((hi - ay) / dy) * dx;
            let (xmin, xmax) = if xlo < xhi { (xlo, xhi) } else { (xhi, xlo) };
            let sign = if dy > 0.0 { 1 } else { -1 };
            edges.push(Edge { xlo, lo, k, sign });
            meta.push(EdgeMeta { hi, xmin, xmax });
        }
    }
    if edges.is_empty() {
        return;
    }
    let ne = edges.len();

    // ---- 2. 事件：段端点 + 段间交点 ----
    let mut events: Vec<f64> = Vec::with_capacity(ne * 2 + 64);
    for (e, m) in edges.iter().zip(meta.iter()) {
        events.push(e.lo);
        events.push(m.hi);
    }

    // 交点剪枝：按 xmin 排序后扫描，只对 x 区间重叠（交点 x 必在两者 x 区间内）
    // 且 y 区间重叠的段对求交。绝大多数边对在此被直接跳过。
    {
        let mut by_x: Vec<u32> = (0..ne as u32).collect();
        by_x.sort_unstable_by(|&i, &j| {
            let (a, b) = (&meta[i as usize], &meta[j as usize]);
            a.xmin.total_cmp(&b.xmin).then(i.cmp(&j))
        });
        for ii in 0..ne {
            let ea = edges[by_x[ii] as usize];
            let ma = meta[by_x[ii] as usize];
            for jj in (ii + 1)..ne {
                let eb = edges[by_x[jj] as usize];
                let mb = meta[by_x[jj] as usize];
                if mb.xmin > ma.xmax {
                    break; // 后面所有段的 x 区间都在 ea 右侧
                }
                let lo = if ea.lo > eb.lo { ea.lo } else { eb.lo };
                let hi = if ma.hi < mb.hi { ma.hi } else { mb.hi };
                if !lt(lo, hi) {
                    continue; // y 区间不重叠（NaN 也走这里）
                }
                let den = ea.k - eb.k;
                if den == 0.0 {
                    continue; // 平行 / 重合
                }
                // 在公共参考 y = lo 处求两段 x 之差，条件数优于用截距。
                let ystar = lo + (eb.x_at(lo) - ea.x_at(lo)) / den;
                if ystar >= lo && ystar <= hi {
                    events.push(ystar);
                }
            }
        }
    }
    sort_events(&mut events);
    events.dedup();

    // ---- 3. y 扫描：跨越序增量维护 + 每行区域积分 ----
    let mut by_lo: Vec<u32> = (0..ne as u32).collect();
    by_lo.sort_unstable_by(|&i, &j| {
        let (a, b) = (&edges[i as usize], &edges[j as usize]);
        a.lo.total_cmp(&b.lo).then(i.cmp(&j))
    });

    let make_crossing = |id: u32| {
        let e = edges[id as usize];
        Crossing { xlo: e.xlo, lo: e.lo, k: e.k, sign: e.sign, id }
    };
    let mut order: Vec<Crossing> = Vec::with_capacity(ne);
    let mut cur_lo = 0usize;
    let mut ev_idx = 0usize;
    let mut open_y = 0.0f64; // 当前「跨越序恒定区间」的起点

    // 起点：所有 lo ≤ 0 的边（裁到画布后 lo ∈ [0, height)）。
    while cur_lo < ne && edges[by_lo[cur_lo] as usize].lo <= open_y {
        order.push(make_crossing(by_lo[cur_lo]));
        cur_lo += 1;
    }
    sort_active(&mut order, mid_of(open_y, ev_idx, &events, height_f));

    let width_usize = width as usize;
    let height_i = height as i64;
    let mut row: i64 = 0;
    while row < height_i {
        // 活跃集为空且所有边都已入场 → 剩余行必然全空。
        if order.is_empty() && cur_lo >= ne {
            break;
        }
        // 活跃集为空但还有边没入场 → 直接跳到下一个事件所在行。
        if order.is_empty() && ev_idx < events.len() {
            let nr = events[ev_idx].floor() as i64;
            if nr > row {
                row = nr;
                if row >= height_i {
                    break;
                }
                continue;
            }
        }

        let y0 = row as f64;
        let y1 = y0 + 1.0;
        let base = row as usize * width_usize;
        let line = &mut cov.data[base..base + width_usize];

        // 本行内的事件：逐个「先结算、再改序」。
        while ev_idx < events.len() && events[ev_idx] < y1 {
            let ye = events[ev_idx];
            let p = if open_y > y0 { open_y } else { y0 };
            accumulate(&order, p, ye, rule, line);
            // 删：hi ≤ ye 的边在本事件之后不再覆盖任何面积。
            order.retain(|c| meta[c.id as usize].hi > ye);
            // 插：lo ≤ ye 的边。
            while cur_lo < ne && edges[by_lo[cur_lo] as usize].lo <= ye {
                order.push(make_crossing(by_lo[cur_lo]));
                cur_lo += 1;
            }
            open_y = ye;
            ev_idx += 1;
            sort_active(&mut order, mid_of(open_y, ev_idx, &events, height_f));
        }
        // 行尾结算到 y1。
        let p = if open_y > y0 { open_y } else { y0 };
        accumulate(&order, p, y1, rule, line);

        row += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
        vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ]
    }

    #[test]
    fn unit_square_exact_coverage() {
        // 方块 [1.5,3)×[1.5,3)：内部 1.0，边界列/行 0.5（半开语义）。
        let cov = rasterize(&[square(1.5, 1.5, 3.0, 3.0)], FillRule::Winding, 5, 5);
        assert_eq!(cov.get(2, 2), 1.0, "内部像素全覆盖");
        assert_eq!(cov.get(1, 2), 0.5, "左边界列恰半覆盖");
        assert_eq!(cov.get(2, 1), 0.5, "上边界行恰半覆盖");
        assert_eq!(cov.get(1, 1), 0.25, "角落 1/4 覆盖");
        assert_eq!(cov.get(0, 0), 0.0);
        assert_eq!(cov.get(3, 2), 0.0, "x∈[3,4) 属半开区间外");
    }

    #[test]
    fn diagonal_triangle_half_plane() {
        // 直角三角形 (0,0)(2,0)(0,2)：斜边 x+y=2。
        // 像素 (0,0)：{x+y<2} ⊇ [0,1)²（仅触角点）→ 全覆盖 1。
        // 像素 (1,0)/(0,1)：对角线切半 → 0.5；像素 (1,1) 在斜边外侧 → 0。
        let tri = vec![Point::new(0.0, 0.0), Point::new(2.0, 0.0), Point::new(0.0, 2.0)];
        let cov = rasterize(&[tri], FillRule::Winding, 4, 4);
        assert!((cov.get(0, 0) - 1.0).abs() < 1e-6, "实际 {}", cov.get(0, 0));
        assert!((cov.get(1, 0) - 0.5).abs() < 1e-6, "实际 {}", cov.get(1, 0));
        assert!((cov.get(0, 1) - 0.5).abs() < 1e-6, "实际 {}", cov.get(0, 1));
        assert_eq!(cov.get(1, 1), 0.0, "对角线上方为空");
        // 总面积 = 2（三角形面积），覆盖率总和守恒。
        let sum: f32 = cov.data.iter().sum();
        assert!((sum - 2.0).abs() < 1e-5, "面积守恒：{sum}");
    }

    #[test]
    fn winding_vs_evenodd_on_overlap() {
        // 两个同向重叠方块 [0,2)² 与 [1,3)²：
        // winding 规则 = 并集（7）；even-odd 规则重叠区为偶数 = 洞（6）。
        let contours = [square(0.0, 0.0, 2.0, 2.0), square(1.0, 1.0, 3.0, 3.0)];
        let w = rasterize(&contours, FillRule::Winding, 5, 5);
        let e = rasterize(&contours, FillRule::EvenOdd, 5, 5);
        let sum_w: f32 = w.data.iter().sum();
        let sum_e: f32 = e.data.iter().sum();
        assert!((sum_w - 7.0).abs() < 1e-4, "winding 并集面积 7，实际 {sum_w}");
        assert!((sum_e - 6.0).abs() < 1e-4, "even-odd 重叠区成洞：6，实际 {sum_e}");
        // 重叠像素：winding=2 → 覆盖 1（不重复计数）；even-odd → 洞。
        assert!((w.get(1, 1) - 1.0).abs() < 1e-6, "重叠区 = 1.0，实际 {}", w.get(1, 1));
        assert_eq!(e.get(1, 1), 0.0, "even-odd 重叠区为洞");
    }

    #[test]
    fn hole_ring_winding_excludes_counter() {
        // 外方 [0,4)² + 反向内方 [1,3)²：环形面积 16−4=12。
        let mut outer = square(0.0, 0.0, 4.0, 4.0);
        outer.reverse();
        let inner = square(1.0, 1.0, 3.0, 3.0);
        let w = rasterize(&[outer, inner], FillRule::Winding, 6, 6);
        let sum: f32 = w.data.iter().sum();
        assert!((sum - 12.0).abs() < 1e-4, "环形面积 12，实际 {sum}");
        assert_eq!(w.get(2, 2), 0.0, "孔内为空");
        assert!((w.get(0, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_bitwise_across_runs() {
        let tri = vec![
            Point::new(0.3, 0.7),
            Point::new(4.9, 1.3),
            Point::new(2.2, 5.8),
        ];
        let a = rasterize(core::slice::from_ref(&tri), FillRule::Winding, 8, 8);
        let b = rasterize(&[tri], FillRule::Winding, 8, 8);
        assert_eq!(a.data.len(), b.data.len());
        for (x, y) in a.data.iter().zip(b.data.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "两次光栅化必须逐位一致");
        }
    }

    #[test]
    fn out_of_bounds_geometry_clips_safely() {
        // 超出缓冲的大方块：覆盖率不越界、被视口裁剪。
        let cov = rasterize(&[square(-5.0, -5.0, 10.0, 10.0)], FillRule::Winding, 3, 3);
        for v in &cov.data {
            assert!((v - 1.0).abs() < 1e-6, "视口内应全覆盖，实际 {v}");
        }
    }

    /// `rasterize_into` 复用缓冲：结果必须与 `rasterize` **逐字节一致**，且反复
    /// 复用同一缓冲（不同尺寸 / 不同几何）不残留旧数据、不污染下一次。
    #[test]
    fn rasterize_into_is_byte_identical_and_reusable() {
        let cases: Vec<(Vec<Vec<Point>>, FillRule, u32, u32)> = vec![
            (vec![square(1.5, 1.5, 8.5, 6.5)], FillRule::Winding, 12, 9),
            (vec![vec![Point::new(0.3, 0.7), Point::new(9.1, 2.2), Point::new(3.4, 7.9)]],
             FillRule::Winding, 11, 9),
            (vec![square(0.0, 0.0, 4.0, 4.0), square(2.0, 2.0, 6.0, 6.0)],
             FillRule::EvenOdd, 8, 8),
            (vec![square(-5.0, -5.0, 20.0, 20.0)], FillRule::Winding, 6, 5),
            (Vec::new(), FillRule::Winding, 7, 6),
        ];
        let mut reused = Coverage::new(1, 1); // 起始尺寸故意不同，强制 resize 路径
        for (contours, rule, w, h) in &cases {
            let fresh = rasterize(contours, *rule, *w, *h);
            rasterize_into(contours, *rule, *w, *h, &mut reused);
            assert_eq!(reused.width, *w, "复用缓冲宽");
            assert_eq!(reused.height, *h, "复用缓冲高");
            assert_eq!(reused.data.len(), fresh.data.len(), "复用缓冲长度");
            for (i, (a, b)) in fresh.data.iter().zip(reused.data.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "rasterize_into 必须逐字节一致 @{i}");
            }
        }
        // 尺寸收缩后再调用：clear+resize 不留旧的大缓冲内容。
        rasterize_into(&[], FillRule::Winding, 2, 2, &mut reused);
        assert!(reused.data.iter().all(|&v| v == 0.0), "收缩后应全零");
    }

    /// 事件排序的 LSD 基数排序分支（> RADIX_THRESHOLD）必须与
    /// `sort_unstable_by(f64::total_cmp)` 逐位等价：覆盖 ±0、±inf、次正规、
    /// 负值、大数、重复键、随机值。
    #[test]
    fn radix_event_sort_matches_total_cmp_bitwise() {
        let mut v: Vec<f64> = Vec::new();
        // 特殊值。
        v.extend_from_slice(&[
            0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::MIN_POSITIVE,
            f64::from_bits(1), -f64::from_bits(1), 1.0, -1.0, 1e300, -1e300,
        ]);
        // 确定性伪随机填充到超过阈值（LCG，避免依赖随机源）。
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        while v.len() < RADIX_THRESHOLD * 2 + 137 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let bits = s ^ (s >> 33);
            v.push(f64::from_bits(bits));
        }
        // 补一批重复键，验证计数排序稳定性无关（全序下重复值顺序无关）。
        for _ in 0..500 {
            v.push(3.25);
            v.push(-7.5);
        }
        let mut expected = v.clone();
        expected.sort_unstable_by(f64::total_cmp);
        sort_events(&mut v);
        assert_eq!(v.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(v.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "基数排序与 total_cmp 逐位等价 @{i}");
        }
        // 小数组走比较排序分支也应等价。
        let mut small = vec![2.0, -1.0, 0.0, -0.0, 5.5, 1e-320];
        let mut small_ref = small.clone();
        small_ref.sort_unstable_by(f64::total_cmp);
        sort_events(&mut small);
        for (a, b) in small_ref.iter().zip(small.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "小数组分支等价");
        }
    }

    // =======================================================================
    // B.1 边界列 2 列展开 + 窄区合并的冻结逐字参考。
    //
    // 这是优化前 `accumulate` 的逐字拷贝：每次 `tri_factor(...)` 都重算
    // `0.5*h`，左翼 / 右翼 / 窄区各自一条 while/loop 链，没有 2 列展开，也没有
    // `half_h` 预算。生产路径与之逐字节对拍；任何把 `u -= 4.0`、调换括号、
    // 或改变列推进顺序的「优化」都会在这里红掉。
    // =======================================================================
    fn accumulate_frozen(
        order: &[Crossing],
        p: f64,
        q: f64,
        rule: FillRule,
        row: &mut [f32],
    ) {
        let n = order.len();
        if n < 2 {
            return;
        }
        let h = q - p;
        if !gt(h, 0.0) {
            return;
        }
        let last = row.len() as i64 - 1;
        let hv = h as f32;
        let evenodd = rule == FillRule::EvenOdd;
        let mut winding: i32 = 0;
        let mut lp = order[0].x_at(p);
        let mut lq = order[0].x_at(q);
        let mut k = 0usize;
        while k + 1 < n {
            let rp = order[k + 1].x_at(p);
            let rq = order[k + 1].x_at(q);
            winding += order[k].sign;
            let inside = if evenodd { ((n - 1 - k) & 1) == 1 } else { winding != 0 };
            k += 1;
            if inside {
                let llo = if lp < lq { lp } else { lq };
                let lhi = if lp < lq { lq } else { lp };
                let rlo = if rp < rq { rp } else { rq };
                let rhi = if rp < rq { rq } else { rp };
                if rhi > llo {
                    let b0 = lhi.ceil() as i64;
                    let b1 = (rlo.floor() as i64).saturating_sub(1);
                    if b1 >= b0 {
                        add_span(row, b0, b1, hv, last);
                        let fa = (llo.floor() as i64).max(0);
                        let fb = (b0 - 1).min(last);
                        if fb >= fa {
                            let fac = tri_factor(lp, lq, h);
                            let mut u0 = lp - fa as f64;
                            let mut u1 = lq - fa as f64;
                            let mut s_prev = pos_area(u0, u1, h, fac);
                            let mut c = fa;
                            while c <= fb {
                                u0 -= 1.0;
                                u1 -= 1.0;
                                let s_next = pos_area(u0, u1, h, fac);
                                row[c as usize] += (h - s_prev + s_next) as f32;
                                s_prev = s_next;
                                c += 1;
                            }
                        }
                        let ga = (b1 + 1).max(0);
                        let gb = (rhi.floor() as i64).min(last);
                        if gb >= ga {
                            let fac = tri_factor(rp, rq, h);
                            let mut u0 = rp - ga as f64;
                            let mut u1 = rq - ga as f64;
                            let mut s_prev = pos_area(u0, u1, h, fac);
                            let mut c = ga;
                            while c <= gb {
                                u0 -= 1.0;
                                u1 -= 1.0;
                                let s_next = pos_area(u0, u1, h, fac);
                                row[c as usize] += (s_prev - s_next) as f32;
                                s_prev = s_next;
                                c += 1;
                            }
                        }
                    } else {
                        let a = (llo.min(rlo).floor() as i64).max(0);
                        let b = (rhi.max(lhi).floor() as i64).min(last);
                        if b >= a {
                            let fac_l = tri_factor(lp, lq, h);
                            let fac_r = tri_factor(rp, rq, h);
                            let mut lu0 = lp - a as f64;
                            let mut lu1 = lq - a as f64;
                            let mut ru0 = rp - a as f64;
                            let mut ru1 = rq - a as f64;
                            let mut sl = pos_area(lu0, lu1, h, fac_l);
                            let mut sr = pos_area(ru0, ru1, h, fac_r);
                            let mut c = a;
                            loop {
                                lu0 -= 1.0;
                                lu1 -= 1.0;
                                ru0 -= 1.0;
                                ru1 -= 1.0;
                                let sln = pos_area(lu0, lu1, h, fac_l);
                                let srn = pos_area(ru0, ru1, h, fac_r);
                                row[c as usize] += ((sr - srn) - (sl - sln)) as f32;
                                if c == b {
                                    break;
                                }
                                sl = sln;
                                sr = srn;
                                c += 1;
                            }
                        }
                    }
                }
            }
            lp = rp;
            lq = rq;
        }
    }

    fn crossing(p: f64, q: f64, xp: f64, xq: f64, sign: i32, id: u32) -> Crossing {
        Crossing {
            xlo: xp,
            lo: p,
            k: (xq - xp) / (q - p),
            sign,
            id,
        }
    }

    fn pair(p: f64, q: f64, lp: f64, lq: f64, rp: f64, rq: f64) -> Vec<Crossing> {
        vec![
            crossing(p, q, lp, lq, 1, 0),
            crossing(p, q, rp, rq, -1, 1),
        ]
    }

    fn quad(p: f64, q: f64) -> Vec<Crossing> {
        vec![
            crossing(p, q, 2.2, 7.8, 1, 0),
            crossing(p, q, 40.2, 45.8, -1, 1),
            crossing(p, q, 60.2, 65.8, 1, 2),
            crossing(p, q, 90.2, 95.8, -1, 3),
        ]
    }

    fn assert_rows_bits(tag: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{tag}: 行宽不一致");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{tag}: 边界链逐字节不一致 @{i} got={a} want={b}");
        }
    }

    #[test]
    fn tri_factor_half_h_budget_is_bitwise_identical() {
        let hs = [0.0f64, 1e-3, 0.37, 0.5, 0.83, 1.0, 3.25];
        let xs = [-4.5f64, -0.25, 0.0, 0.125, 2.2, 40.2, 4096.0];
        let mut checks = 0u64;
        for &h in &hs {
            let half_h = 0.5 * h;
            for &xp in &xs {
                for &xq in &xs {
                    let got = tri_factor_with_half_h(xp, xq, half_h);
                    let want = tri_factor(xp, xq, h);
                    assert_eq!(got.to_bits(), want.to_bits(),
                        "tri_factor: xp={xp} xq={xq} h={h} 逐位不一致 got={got} want={want}");
                    checks += 1;
                }
            }
        }
        println!("[tri-factor] {checks} 组 half_h 预算逐位对拍全等");
    }

    /// B.1 主判据：2 列展开链 + 窄区合并 + `half_h` 预算与冻结老实现逐字节一致。
    ///
    /// 覆盖：1/2/6/10/17 列边界链（触发 2 列主体与所有尾长）、宽区内部列、
    /// 窄区双链、负坐标裁剪、多区域 winding/even-odd、非零点区间起点。
    #[test]
    fn boundary_chain_is_byte_identical_to_frozen_accumulate() {
        let cases: Vec<(&str, f64, f64, f64, f64, f64, f64, FillRule, usize)> = vec![
            ("wide_1col", 0.0, 0.91, 3.2, 3.8, 20.2, 20.9, FillRule::Winding, 64),
            ("wide_6col", 0.0, 0.83, 2.2, 7.8, 40.2, 45.8, FillRule::Winding, 64),
            ("wide_10col", 0.17, 0.88, 1.2, 10.8, 30.2, 39.8, FillRule::EvenOdd, 64),
            ("wide_negative_clip", 0.0, 0.73, -3.2, 2.8, 20.2, 25.8, FillRule::Winding, 64),
            ("narrow_2col", 0.0, 0.63, 10.2, 10.5, 10.9, 11.2, FillRule::Winding, 32),
            ("narrow_17col", 0.0, 0.77, 2.2, 10.7, 10.2, 18.7, FillRule::Winding, 32),
            ("narrow_evenodd", 0.11, 0.69, 4.2, 12.7, 12.2, 20.7, FillRule::EvenOdd, 32),
        ];
        let mut checks = 0u64;
        for (name, p, q, lp, lq, rp, rq, rule, width) in cases {
            let order = pair(p, q, lp, lq, rp, rq);
            let mut got = vec![0.0f32; width];
            let mut want = vec![0.0f32; width];
            accumulate(&order, p, q, rule, &mut got);
            accumulate_frozen(&order, p, q, rule, &mut want);
            assert_rows_bits(name, &got, &want);
            checks += width as u64;
        }
        // 多区域：4 条跨越形成两个覆盖区，验证 winding 累加 / even-odd 计数 /
        // 区域间接力 x 值与多次借用同一行的累加顺序都逐位一致。
        for &rule in &[FillRule::Winding, FillRule::EvenOdd] {
            for &(p, q) in &[(0.0f64, 0.71f64), (0.13f64, 0.94f64)] {
                let order = quad(p, q);
                let mut got = vec![0.0f32; 128];
                let mut want = vec![0.0f32; 128];
                accumulate(&order, p, q, rule, &mut got);
                accumulate_frozen(&order, p, q, rule, &mut want);
                assert_rows_bits("quad", &got, &want);
                checks += 128;
            }
        }
        println!("[boundary-chain] {checks} 个 f32 覆盖率逐位对拍全等（宽区 / 窄区 / 2 列展开 / 多区域 / 双规则）");
    }

    /// B.1 边界链微基准：同一组边界列的「冻结老链」vs「2 列展开合并链」。
    ///
    /// 只打印数字，不做速度断言（CI 机器负载不同不该让正确性判据变红）；
    /// 逐字节等价由上面的 `boundary_chain_is_byte_identical_to_frozen_accumulate`
    /// 硬判担保。两路都跑同样的 `black_box(row)` 与同一输入行，差额只来自
    /// `accumulate` 主体里的边界列实现。
    #[test]
    fn boundary_chain_kernel_microbench() {
        fn median(v: &mut [f64]) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            v[v.len() / 2]
        }
        type AccumulateFn = fn(&[Crossing], f64, f64, FillRule, &mut [f32]);
        fn bench(f: AccumulateFn, order: &[Crossing], p: f64, q: f64) -> f64 {
            let mut row = vec![0.0f32; 128];
            let mut samples = [0.0f64; 7];
            for s in &mut samples {
                let t0 = std::time::Instant::now();
                for _ in 0..20_000u32 {
                    // 间接调用：防止编译器把某一路特化 / 内联进基准循环，使
                    // 「冻结函数」与「生产函数」在同等调用约定下比较。
                    std::hint::black_box(f)(order, p, q, FillRule::Winding, &mut row);
                    std::hint::black_box(&row);
                }
                *s = t0.elapsed().as_secs_f64() * 1e3;
            }
            median(&mut samples)
        }
        let (p, q) = (0.0f64, 0.83f64);
        let wide = quad(p, q);
        let wide_before = bench(accumulate_frozen as AccumulateFn, &wide, p, q);
        let wide_after = bench(accumulate as AccumulateFn, &wide, p, q);

        // 17 列窄区双链：逐列共用骨架，边界列密集时才是本优化的对照形状；
        // 宽区只有 6 列链，展开收益会被内部列 add_span 稀释。
        let (np, nq) = (0.0f64, 0.77f64);
        let narrow = pair(np, nq, 2.2, 10.7, 10.2, 18.7);
        let narrow_before = bench(accumulate_frozen as AccumulateFn, &narrow, np, nq);
        let narrow_after = bench(accumulate as AccumulateFn, &narrow, np, nq);
        println!(
            "[boundary-bench] accumulate 20k×7：wide frozen={wide_before:.3} ms chain={wide_after:.3} ms speedup={:.2}x | narrow17 frozen={narrow_before:.3} ms chain={narrow_after:.3} ms speedup={:.2}x",
            wide_before / wide_after,
            narrow_before / narrow_after
        );
    }
}
