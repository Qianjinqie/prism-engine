//! QA 验收：over 预乘修复（规格《值域层-预乘混合规格》§5 六条，照方断言）。
//!
//! 覆盖：
//! - 验收 1 结合律：over 随机 20000 组 `(a∘b)∘c` vs `a∘(b∘c)`；
//! - 验收 2 兼容性：αb=1 ⟹ 输出 = b（逐位）；双不透明 ⟹ b；引擎记号下
//!   「下层不透明」与修复前 legacy 公式逐位相同；半透明叠半透明为**唯一**
//!   行为变化象限（显式钉住，对应 changelog）；
//! - 验收 3 误差结构：n ∈ {2,4,8,16,32} 恒定 alpha 栈，预乘版误差 ≤ 单层
//!   ε 量级且曲线平坦（f64 参考），legacy 复利对照（n=2 即 ≥1e-2）；
//! - 验收 4 零 alpha：双透明 ⟹ 透明黑，无 NaN；
//! - 验收 5 扁平化：覆盖类 `eval_flat` 与树形 `eval`（n=3 逐位；n=4/5
//!   不透明逐位、随机半透明 ULP 界）；着色类非结合锁定（命题 3.1）；
//! - 验收 6 plus/screen/multiply/mask 黄金值不变。
//!
//! # 如实记录（规格与现实之差，不静默放宽）
//!
//! - 规格 §5.1 要求「逐位 20000/20000」。实测：f32 下预乘域 over 的
//!   **逐位**结合不成立（20000 组中 15656 组存在 ULP 级差异，max abs
//!   diff = 1.788e-07 ≈ 2–4 ULP；纯 f64 预乘域亦有 15325/20000）——
//!   括号只影响最后求和的舍入（注 2.4.1），有限精度下结合律是 ULP 级
//!   而非逐位。本测试以 **max ≤ 1e-6**（≈ 8ε，ε = 2^-23 ≈ 1.19e-7）为
//!   判据，对照 legacy 同测的 2.295e-01 量级违犯——规格意图「误差结构
//!   单层化、可扁平化」由此满足。
//!   （实测数字为 2026-10-23 复跑记录：固定 seed splitmix64，同构建
//!   确定性复跑两次一致；换构建/平台可能有个别 ULP 抖动，量级稳定。）
//! - 规格命题 2.5(b) 称「下层不透明 + 上层半透明」输出改变。引擎层序
//!   约定（`blend_pixel(op, a=下层, b=上层)`）下重算 + 2000 组数值对拍：
//!   该象限新旧公式**逐位相同**（αa 权重与归一化均成平凡，实测差异
//!   组数 0/2000）。真实行为变化仅在「半透明叠半透明」（αa<1 且
//!   αb<1），见验收 2 的显式钉子。

use prism_core::blend_pixel;
use prism_core::compose::{merge, ComposeNode, LayerBuf};
use prism_core::node::MergeOp;

// ---------------------------------------------------------------------------
// 确定性 PRNG（零依赖内实现；splitmix64，跨平台逐位一致）
// ---------------------------------------------------------------------------

/// splitmix64：小状态、快、确定；仅用于本测试的确定抽样。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[0,1)` 的 f32（24 位精度）。
    fn f01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }

    /// `[0,1]` 的 f32（两端闭：alpha 需要 α=1 端点样本）。
    fn f01_closed(&mut self) -> f32 {
        // 0..=2^24 映射到 [0,1]；端点概率各 1/(2^24+1)，确定。
        (self.next_u64() % 0x0100_0001) as f32 / ((1u32 << 24) as f32)
    }

    /// 随机 RGBA（alpha ∈ [0,1]）。
    fn rgba(&mut self) -> [f32; 4] {
        [self.f01(), self.f01(), self.f01(), self.f01_closed()]
    }
}

/// 修复前 legacy over 公式（规格 §1 病灶，作对照实现内联于此）。
///
/// 引擎层序约定 a = 下层、b = 上层；缺 αa 权重与归一化。
fn legacy_over(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    out[3] = b[3] + a[3] * (1.0 - b[3]);
    for i in 0..3 {
        out[i] = a[i] * (1.0 - b[3]) + b[i] * b[3];
    }
    out
}

/// f64 预乘域 over 参考（引擎层序约定；α ≤ 0 ⟹ (0,0,0,0)）。
///
/// 用作「真值」：f64 下单括号化的舍入差异比 f32 小 2^29 倍，链式参考
/// 视为精确（规格注 2.4.1 的单层误差度量基准）。
fn over_f64_reference(a: [f32; 4], b: [f32; 4]) -> [f64; 4] {
    let a = [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64];
    let b = [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64];
    if a[3] <= 0.0 && b[3] <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let ca = [a[0] * a[3], a[1] * a[3], a[2] * a[3], a[3]];
    let cb = [b[0] * b[3], b[1] * b[3], b[2] * b[3], b[3]];
    let keep = 1.0 - cb[3];
    let ao = cb[3] + ca[3] * keep;
    if ao <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    [
        (cb[0] + ca[0] * keep) / ao,
        (cb[1] + ca[1] * keep) / ao,
        (cb[2] + ca[2] * keep) / ao,
        ao,
    ]
}

/// 四通道最大绝对差。
fn max_abs_diff(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..4).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)
}

/// f32 与 f64 参考的最大绝对差。
fn max_abs_diff_f64(a: [f32; 4], b: [f64; 4]) -> f64 {
    (0..4)
        .map(|i| (a[i] as f64 - b[i]).abs())
        .fold(0.0f64, f64::max)
}

/// 构造单像素层。
fn layer1(c: [f32; 4]) -> LayerBuf {
    let mut b = LayerBuf::new(1, 1);
    b.set(0, 0, c);
    b
}

// ---------------------------------------------------------------------------
// 验收 1：结合律（20000 组）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_1_over_is_associative_20000_groups() {
    // 规格 §5.1：`(a∘b)∘c = a∘(b∘c)`。判据 max ≤ 1e-6（≈ 8ε）。
    // 如实记录：f32 逐位结合不成立（15656/20000 有 ULP 差，max 1.788e-07），
    // 见模块头「如实记录」段；legacy 同测为 2.295e-01 量级。
    let mut rng = Rng::new(0x5EED_0000_0000_0001);
    let mut worst_new = 0.0f32;
    let mut worst_legacy = 0.0f32;
    for _ in 0..20_000 {
        let a = rng.rgba();
        let b = rng.rgba();
        let c = rng.rgba();
        let left = blend_pixel(MergeOp::Over, blend_pixel(MergeOp::Over, a, b), c);
        let right = blend_pixel(MergeOp::Over, a, blend_pixel(MergeOp::Over, b, c));
        worst_new = worst_new.max(max_abs_diff(left, right));
        let lleft = legacy_over(legacy_over(a, b), c);
        let lright = legacy_over(a, legacy_over(b, c));
        worst_legacy = worst_legacy.max(max_abs_diff(lleft, lright));
    }
    assert!(
        worst_new <= 1e-6,
        "预乘 over 结合律超差: max = {worst_new:e}（判据 1e-6）"
    );
    assert!(
        worst_legacy > 1e-2,
        "legacy 复利对照应显著违犯（否则对照失效）: max = {worst_legacy:e}"
    );
}

// ---------------------------------------------------------------------------
// 验收 2：兼容性（逐位钉子 + 行为变化象限显式钉住）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_2a_opaque_upper_returns_upper_bitwise() {
    // 规格命题 2.5(a)：αb=1 ⟹ 输出 = b（逐位），2000 组。
    let mut rng = Rng::new(0x5EED_0000_0000_0002);
    for _ in 0..2_000 {
        let a = rng.rgba();
        let b = [rng.f01(), rng.f01(), rng.f01(), 1.0];
        let out = blend_pixel(MergeOp::Over, a, b);
        assert_eq!(out, b, "αb=1 必须逐位输出上层: a={a:?} b={b:?}");
    }
}

#[test]
fn acceptance_2c_double_opaque_returns_upper_bitwise() {
    // 规格命题 2.5(c)：双不透明 ⟹ 输出 = b（逐位），2000 组。
    let mut rng = Rng::new(0x5EED_0000_0000_0003);
    for _ in 0..2_000 {
        let a = [rng.f01(), rng.f01(), rng.f01(), 1.0];
        let b = [rng.f01(), rng.f01(), rng.f01(), 1.0];
        let out = blend_pixel(MergeOp::Over, a, b);
        assert_eq!(out, b, "双不透明必须逐位输出上层");
    }
}

#[test]
fn acceptance_2b_opaque_lower_matches_legacy_bitwise() {
    // 规格命题 2.5(b) 在引擎记号下的如实修正：「下层不透明 + 上层半透明」
    // （αa=1, 0<αb<1）新旧公式**逐位相同**——αa 权重与归一化成平凡
    // （加法交换律 + f32 Sterbenz：αa=1 ⟹ αo = fl(αb + fl(1−αb)) = 1）。
    // 2000 组数值对拍确认；真实行为变化仅在半透明叠半透明（下一测试）。
    let mut rng = Rng::new(0x5EED_0000_0000_0004);
    for _ in 0..2_000 {
        let a = [rng.f01(), rng.f01(), rng.f01(), 1.0];
        let b = [rng.f01(), rng.f01(), rng.f01(), rng.f01()];
        let out = blend_pixel(MergeOp::Over, a, b);
        let legacy = legacy_over(a, b);
        assert_eq!(out, legacy, "αa=1 必须与 legacy 逐位相同: a={a:?} b={b:?}");
    }
}

#[test]
fn acceptance_2_semitransparent_over_semitransparent_changes() {
    // 唯一行为变化象限（changelog「渲染行为变化」条目）：
    // αa<1 且 αb<1 —— 修复前缺 αa 权重与归一化。显式钉住新值，防回退。
    // 0.5 红（下）上叠 0.5 蓝（上）：预乘合成 ⟹ (1/3, 0, 2/3, 0.75)。
    let a = layer1([1.0, 0.0, 0.0, 0.5]);
    let b = layer1([0.0, 0.0, 1.0, 0.5]);
    let out = merge(MergeOp::Over, &a, &b).get(0, 0);
    let legacy = legacy_over([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]);
    assert!(
        (out[0] - 1.0 / 3.0).abs() < 1e-6 && (out[2] - 2.0 / 3.0).abs() < 1e-6,
        "半叠半必须是标准 Porter-Duff: {out:?}"
    );
    assert!(
        (legacy[0] - 0.5).abs() < 1e-6,
        "legacy 对照值应为 0.5（非标准）: {legacy:?}"
    );
    assert!(
        max_abs_diff(out, legacy) > 1e-3,
        "该象限必须与 legacy 不同（否则修复未生效）: out={out:?} legacy={legacy:?}"
    );
}

// ---------------------------------------------------------------------------
// 验收 3：误差结构（n 层栈：平坦 vs 复利）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_3_error_is_flat_not_compound() {
    // 恒定 alpha=0.5 栈（每层颜色不同），与 f64 预乘参考对拍。
    // 新公式：各 n 误差 ≤ 1e-6（单层 ε 量级）且曲线平坦（n=2 ≈ 6.0e-8，
    // n=32 ≈ 4.5e-8，2026-10-23 实测）；
    // legacy 对照：单层（n=2）即 1.6e-1 量级（其绝对误差随 n 收敛——
    // 颜色通道不归一化导致输出被上层锁死——「复利增长」的描述不适用于
    // 本场景；真正的判据是 legacy 最浅栈误差比新公式最深栈大 7 个量级，
    // 即误差结构从「每层引入非标准项」退化为「仅末端舍入」）。
    // 对照失效（legacy 不大幅超差）则 panic。
    let alpha = 0.5f32;
    let mut rng = Rng::new(0x5EED_0000_0000_0005);
    let mut new_worst_by_n: Vec<f64> = Vec::new();
    let mut legacy_worst_by_n: Vec<f64> = Vec::new();
    for &n in &[2usize, 4, 8, 16, 32] {
        let mut new_worst = 0.0f64;
        let mut legacy_worst = 0.0f64;
        for _ in 0..200 {
            let colors: Vec<[f32; 4]> = (0..n)
                .map(|_| [rng.f01(), rng.f01(), rng.f01(), alpha])
                .collect();
            // f64 参考链（引擎层序：colors[0] 最下，colors[n-1] 最上）。
            let mut refv = over_f64_reference(colors[0], colors[1]);
            for c in &colors[2..] {
                refv = over_f64_reference(
                    [
                        refv[0] as f32,
                        refv[1] as f32,
                        refv[2] as f32,
                        refv[3] as f32,
                    ],
                    *c,
                );
            }
            // 新公式链。
            let mut acc = blend_pixel(MergeOp::Over, colors[0], colors[1]);
            for c in &colors[2..] {
                acc = blend_pixel(MergeOp::Over, acc, *c);
            }
            // legacy 链。
            let mut lacc = legacy_over(colors[0], colors[1]);
            for c in &colors[2..] {
                lacc = legacy_over(lacc, *c);
            }
            new_worst = new_worst.max(max_abs_diff_f64(acc, refv));
            legacy_worst = legacy_worst.max(max_abs_diff_f64(lacc, refv));
        }
        new_worst_by_n.push(new_worst);
        legacy_worst_by_n.push(legacy_worst);
    }
    for (i, &n) in [2usize, 4, 8, 16, 32].iter().enumerate() {
        assert!(
            new_worst_by_n[i] <= 1e-6,
            "n={n} 预乘版误差超单层 ε 界: {}",
            new_worst_by_n[i]
        );
    }
    // 平坦：n=32 不得显著超过 n=2（复利会随 n 线性增长）。
    assert!(
        new_worst_by_n[4] <= new_worst_by_n[0] + 1e-7,
        "误差曲线应平坦: n=2 {} vs n=32 {}",
        new_worst_by_n[0],
        new_worst_by_n[4]
    );
    // 单层对照：legacy n=2 即 ≥1e-2（单层即非标准，非末端舍入）。
    assert!(
        legacy_worst_by_n[0] >= 1e-2,
        "legacy n=2 复利对照应显著: {}",
        legacy_worst_by_n[0]
    );
    // 结构对照：预乘 n=32 仍远小于 legacy n=2（误差结构单层化的判据）。
    assert!(
        new_worst_by_n[4] < legacy_worst_by_n[0],
        "预乘版最深栈误差应小于 legacy 最浅栈: {} vs {}",
        new_worst_by_n[4],
        legacy_worst_by_n[0]
    );
}

// ---------------------------------------------------------------------------
// 验收 4：零 alpha 边界
// ---------------------------------------------------------------------------

#[test]
fn acceptance_4_zero_alpha_no_nan() {
    // 规格命题 4.1：双透明 ⟹ (0,0,0,0)，from_pre 不被 0 除。
    let mut rng = Rng::new(0x5EED_0000_0000_0006);
    for _ in 0..2_000 {
        let a = [rng.f01(), rng.f01(), rng.f01(), 0.0];
        let b = [rng.f01(), rng.f01(), rng.f01(), 0.0];
        let out = blend_pixel(MergeOp::Over, a, b);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0], "双透明必须输出透明黑");
        assert!(out.iter().all(|v| v.is_finite()), "不得有 NaN/Inf");
    }
    // 单侧零 alpha ⟹ 输出 = 另一侧（语义恒等；有限精度下为 ULP 级）。
    //
    // 如实记录：to_pre/from_pre 往返是 `b·α/α`，仅当 α 为 2 的幂时无
    // 舍入（逐位）；一般 α 下有 1–2 ULP 差（定义 2.2 的有限精度表现，
    // 规格命题 4.1 明言「无新增边界情形；无需特判码」——故不为此加
    // 特判分支）。下面分组断言：幂二 alpha 逐位，一般 alpha ≤ 1e-6。
    for i in 0..2_000u32 {
        let b = rng.rgba();
        let a = [rng.f01(), rng.f01(), rng.f01(), 0.0];
        let out = blend_pixel(MergeOp::Over, a, b);
        assert!(out.iter().all(|v| v.is_finite()), "αa=0 不得有 NaN/Inf");
        assert!(max_abs_diff(out, b) <= 1e-6, "αa=0 必须输出上层（ULP 界）");
        if i % 4 == 0 {
            // 幂二 alpha 子集（1/2, 1/4, ..., 1/256）：往返无舍入 ⟹ 逐位。
            let b_pow2 = [b[0], b[1], b[2], 0.5f32.powi((i % 8) as i32 + 1)];
            let out_pow2 = blend_pixel(MergeOp::Over, a, b_pow2);
            assert_eq!(out_pow2, b_pow2, "幂二 αb 下 αa=0 必须逐位输出上层");
        }
    }
}

// ---------------------------------------------------------------------------
// 验收 5：扁平化（覆盖类 flat ≡ tree；着色类锁定）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_5_over_flat_matches_tree_n3_bitwise() {
    // n=3：固定形状归约与左结合同形 ⟹ 逐位相同（随机半透明层）。
    let mut rng = Rng::new(0x5EED_0000_0000_0007);
    let layers: Vec<LayerBuf> = (0..3).map(|_| layer1(rng.rgba())).collect();
    let flat = ComposeNode::eval_flat(MergeOp::Over, &layers).unwrap();
    let tree = ComposeNode::from_flat(MergeOp::Over, layers)
        .unwrap()
        .eval();
    assert_eq!(flat, tree, "n=3 覆盖类扁平化必须逐位相同");
}

#[test]
fn acceptance_5_over_flat_matches_tree_n4_n5() {
    // n=4/5：归约形状与左链不同形（如 n=4 为 (a∘b)∘(c∘d)）。
    // 不透明层：over ≡ 取上层，任意括号化逐位相同；
    // 随机半透明层：ULP 界（max ≤ 1e-6，同验收 1 判据）。
    let mut rng = Rng::new(0x5EED_0000_0000_0008);
    for &n in &[4usize, 5] {
        let opaque: Vec<LayerBuf> = (0..n)
            .map(|_| layer1([rng.f01(), rng.f01(), rng.f01(), 1.0]))
            .collect();
        let flat = ComposeNode::eval_flat(MergeOp::Over, &opaque).unwrap();
        let tree = ComposeNode::from_flat(MergeOp::Over, opaque)
            .unwrap()
            .eval();
        assert_eq!(flat, tree, "n={n} 不透明层扁平化必须逐位相同");

        let semi: Vec<LayerBuf> = (0..n).map(|_| layer1(rng.rgba())).collect();
        let flat = ComposeNode::eval_flat(MergeOp::Over, &semi).unwrap();
        let tree = ComposeNode::from_flat(MergeOp::Over, semi).unwrap().eval();
        assert_eq!((flat.width, flat.height), (tree.width, tree.height));
        let worst = flat
            .rgba
            .iter()
            .zip(tree.rgba.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-6, "n={n} 半透明扁平化超 ULP 界: {worst:e}");
    }
}

#[test]
fn acceptance_5_shading_ops_are_bracket_locked() {
    // 命题 3.1（可证面）：着色类（screen/multiply/mask）锁层序 + 左结合，
    // **不可与覆盖类混布后重排——混排改变结果**。
    //
    // 如实记录（修正早期转述）：重算证明引擎直通公式下 screen / multiply /
    // mask 各自满足结合律（`s(x,y)=x+y−xy` 结合、乘积结合、
    // `mask(x,y)=(x.rgb·αy, αx·αy)` 结合——乘法结合律），对着色类单独
    // 扁平化在数值上并不改语义。锁序的真正依据是规格 §3：预乘域无标准
    // 定义（mask 强行预乘会把遮罩静默改成淡出）+ 命题 3.1 的混布纪律。
    // 故本测试断言两条**为真**的不变量：
    //   (i) 混布括号敏感：over(a, screen(b,c)) ≠ screen(over(a,b), c)；
    //   (ii) mask 层序敏感（不交换）：αa≠αb 时 mask(a,b) ≠ mask(b,a)。

    // (i) 混布：两类算子交错时括号化改变结果。
    let a = layer1([0.5, 0.5, 0.5, 0.5]);
    let b = layer1([0.5, 0.5, 0.5, 0.5]);
    let c = layer1([1.0, 0.0, 0.0, 0.5]);
    let screen_bc = merge(MergeOp::Screen, &b, &c);
    let over_ab = merge(MergeOp::Over, &a, &b);
    let l = merge(MergeOp::Over, &a, &screen_bc); // over(a, screen(b,c))
    let r = merge(MergeOp::Screen, &over_ab, &c); // screen(over(a,b), c)
    assert_ne!(
        l.get(0, 0),
        r.get(0, 0),
        "混布必须括号敏感（否则命题 3.1 形同虚设）: {:?} vs {:?}",
        l.get(0, 0),
        r.get(0, 0)
    );

    // (ii) mask 不交换：上层/下层的 alpha 扮演的角色不同（亮度乘 m vs 覆盖乘 m）。
    let ma = layer1([1.0, 1.0, 1.0, 0.5]);
    let mb = layer1([0.5, 0.5, 0.5, 1.0]);
    let m_ab = merge(MergeOp::Mask, &ma, &mb).get(0, 0);
    let m_ba = merge(MergeOp::Mask, &mb, &ma).get(0, 0);
    assert_ne!(m_ab, m_ba, "mask 必须层序敏感: {m_ab:?} vs {m_ba:?}");
    // 锚点值：mask(a,b) = (a.rgb·αb, αa·αb) = (1, 1, 1, 0.5)。
    assert!(
        (m_ab[0] - 1.0).abs() < 1e-6 && (m_ab[3] - 0.5).abs() < 1e-6,
        "{m_ab:?}"
    );
    // mask(b,a) = (b.rgb·αa, αb·αa) = (0.25, 0.25, 0.25, 0.5)。
    assert!(
        (m_ba[0] - 0.25).abs() < 1e-6 && (m_ba[3] - 0.5).abs() < 1e-6,
        "{m_ba:?}"
    );
}

// ---------------------------------------------------------------------------
// 验收 6：plus/screen/multiply/mask 行为不变（黄金值）
// ---------------------------------------------------------------------------

#[test]
fn acceptance_6_other_ops_golden_unchanged() {
    // 规格 §5.6：不碰公式，只归档语义。黄金值与修复前一致。
    let a = layer1([0.1, 0.2, 0.3, 0.4]);
    let b = layer1([0.5, 0.6, 0.7, 0.8]);
    // plus：四通道相加（alpha 可 > 1，HDR 设计）。
    let p = merge(MergeOp::Plus, &a, &b).get(0, 0);
    assert!(
        (p[0] - 0.6).abs() < 1e-6 && (p[3] - 1.2).abs() < 1e-6,
        "plus: {p:?}"
    );
    // screen / multiply：0.5·0.5 参考值。
    let a5 = layer1([0.5, 0.5, 0.5, 1.0]);
    let b5 = layer1([0.5, 0.5, 0.5, 1.0]);
    let s = merge(MergeOp::Screen, &a5, &b5).get(0, 0);
    assert!(
        (s[0] - 0.75).abs() < 1e-6 && (s[3] - 1.0).abs() < 1e-6,
        "screen: {s:?}"
    );
    let m = merge(MergeOp::Multiply, &a5, &b5).get(0, 0);
    assert!(
        (m[0] - 0.25).abs() < 1e-6 && (m[3] - 1.0).abs() < 1e-6,
        "multiply: {m:?}"
    );
    // mask：上层 alpha 当蒙版。
    let aw = layer1([1.0, 1.0, 1.0, 1.0]);
    let bm = layer1([9.0, 9.0, 9.0, 0.25]);
    let k = merge(MergeOp::Mask, &aw, &bm).get(0, 0);
    assert!(
        (k[0] - 0.25).abs() < 1e-6 && (k[3] - 0.25).abs() < 1e-6,
        "mask: {k:?}"
    );
    // 对照：这些算子在 2000 组随机输入下与 legacy 直通公式逐位相同
    // （它们的代码路径未被修复触碰——blend_pixel 只改 Over 臂）。
    let mut rng = Rng::new(0x5EED_0000_0000_0009);
    for _ in 0..2_000 {
        let x = rng.rgba();
        let y = rng.rgba();
        for op in [
            MergeOp::Plus,
            MergeOp::Screen,
            MergeOp::Multiply,
            MergeOp::Mask,
        ] {
            let out = blend_pixel(op, x, y);
            let legacy = match op {
                MergeOp::Plus => [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]],
                MergeOp::Screen => [
                    1.0 - (1.0 - x[0]) * (1.0 - y[0]),
                    1.0 - (1.0 - x[1]) * (1.0 - y[1]),
                    1.0 - (1.0 - x[2]) * (1.0 - y[2]),
                    x[3].max(y[3]),
                ],
                MergeOp::Multiply => [x[0] * y[0], x[1] * y[1], x[2] * y[2], x[3].max(y[3])],
                MergeOp::Mask => [x[0] * y[3], x[1] * y[3], x[2] * y[3], x[3] * y[3]],
                MergeOp::Over => unreachable!("over 单独处理"),
            };
            assert_eq!(out, legacy, "{op:?} 黄金公式必须逐位不变");
        }
    }
}
