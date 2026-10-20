//! 3D LUT：`.cube` 文本解析 + 三线性插值应用（清单第 12 项）。
//!
//! 语义权威：LUT 应用（3D LUT 三线性插值）；
//! 《兼容性宪章》行业对照表（达芬奇 / Resolve / Nuke 的 LUT 节点 → `lut3d`）
//! 与验收 V12（`.cube` 解析 / 应用数值期望）；（B4
//! 落位 prism-postfx）。
//!
//! # 确定性纪律（关卡 2 / 5）
//! - LUT 数据与插值全程 f32 运算域；写回经 [`Fbuf16F`] 量化 f16（
//!   格锚：半浮点存储、运算位模式精确往返）；
//! - 三线性为**固定序 8 角加权和**（像素内固定序归约）：
//!   `w000·c000 + w100·c100 + … + w111·c111`，求和顺序固定 ⇒ 同输入逐位一致；
//! - 输入落在格点时 frac = 0 ⇒ 权重精确为 (1, 0, …, 0) ⇒ **格点采样逐位
//!   恒等**（恒等 LUT + 格点输入 → 输出位模式不变，关卡 2 判定）；
//! - 零超越函数（D-29）：floor / clamp 均为 IEEE 精确运算，无 libm。

use std::fmt;

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

// ---------------------------------------------------------------------------
// 解析错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）
// ---------------------------------------------------------------------------

/// `.cube` 解析错误。错误码延续 PF 段（PF1003–PF1007；段位表单点在
/// prism-core lib.rs，本 crate 新增条目需先上报获批）。
///
/// `line = 0` 表示文件级问题（如缺少 `LUT_3D_SIZE` 声明，无行可指）。
#[derive(Debug, Clone, PartialEq)]
pub struct LutError {
    /// 错误码（PF1xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：出错行号（1 起；0 = 文件级）。
    pub line: usize,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for LutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] .cube 第 {} 行\n  什么事：{}\n  建议：{}",
            self.code, self.line, self.problem, self.fix
        )
    }
}

impl std::error::Error for LutError {}

fn lut_err(code: &'static str, line: usize, problem: String, fix: &str) -> LutError {
    LutError { code, line, problem, fix: fix.to_string() }
}

// ---------------------------------------------------------------------------
// Lut3D
// ---------------------------------------------------------------------------

/// 3D 颜色查找表：`N × N × N` 格点 + 输入域，三线性插值采样。
#[derive(Debug, Clone, PartialEq)]
pub struct Lut3D {
    /// 每维格点数 N（`.cube` 规范 2..=256）。
    pub size: u32,
    /// 输入域下界（`DOMAIN_MIN`，默认 [0,0,0]）。
    pub domain_min: [f32; 3],
    /// 输入域上界（`DOMAIN_MAX`，默认 [1,1,1]）。
    pub domain_max: [f32; 3],
    /// 格点数据，**红通道最快序**（`.cube` 规范）：
    /// `data[r + g·N + b·N²]`。
    pub data: Vec<[f32; 3]>,
}

impl Lut3D {
    /// 恒等 LUT：格点值 = 格点坐标（`data = [r, g, b] / (N−1)`，域 [0,1]）。
    ///
    /// 对恒等 LUT，三线性插值在整个域上数学恒等重构输入（线性函数属于
    /// 三线性基张成空间）；格点输入时逐位恒等——关卡 2 的判定载体。
    ///
    /// # Panics
    /// `size < 2` 时 panic（单格点表无插值意义，宪章 C21 防呆）。
    pub fn identity(size: u32) -> Lut3D {
        assert!(size >= 2, "LUT 每维至少 2 个格点");
        let n = size;
        let last = (n - 1) as f32;
        let mut data = Vec::with_capacity((n * n * n) as usize);
        // 红最快序（与 .cube 数据行序一致）。
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    data.push([r as f32 / last, g as f32 / last, b as f32 / last]);
                }
            }
        }
        Lut3D { size: n, domain_min: [0.0; 3], domain_max: [1.0; 3], data }
    }

    /// 解析 `.cube` 文本（Iridas / Adobe 规范子集）。
    ///
    /// 支持：`#` 注释、空行、`TITLE`（接受忽略）、`LUT_3D_INPUT_DEPTH`
    /// （接受忽略）、`LUT_3D_SIZE N`、`DOMAIN_MIN/MAX x y z`、数据行
    /// （`r g b` 浮点，红通道最快序）。
    pub fn parse_cube(text: &str) -> Result<Lut3D, LutError> {
        let mut size: Option<u32> = None;
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut data: Vec<[f32; 3]> = Vec::new();

        for (no, raw) in text.lines().enumerate() {
            let line_no = no + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                if size.is_some() {
                    return Err(lut_err(
                        "PF1004",
                        line_no,
                        "LUT_3D_SIZE 重复声明".to_string(),
                        "一个 .cube 文件只允许一条 LUT_3D_SIZE，删除多余声明",
                    ));
                }
                let n = parse_u32(rest, line_no)?;
                if !(2..=256).contains(&n) {
                    return Err(lut_err(
                        "PF1004",
                        line_no,
                        format!("LUT_3D_SIZE = {n} 越出合法范围 2..=256"),
                        "将每维格点数改到 2..=256（建议 17 / 33 / 65）",
                    ));
                }
                size = Some(n);
                continue;
            }
            // TITLE / LUT_3D_INPUT_DEPTH：元数据，接受并忽略（规范兼容）。
            if line.starts_with("TITLE") || line.starts_with("LUT_3D_INPUT_DEPTH") {
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
                domain_min = parse_rgb(rest, line_no)?;
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
                domain_max = parse_rgb(rest, line_no)?;
                continue;
            }
            // 其余非空行 = 数据行（r g b）。
            data.push(parse_rgb(line, line_no)?);
        }

        let n = size.ok_or_else(|| {
            lut_err(
                "PF1003",
                0,
                "缺少 LUT_3D_SIZE 声明".to_string(),
                "在数据行之前加入一行 `LUT_3D_SIZE N`（N ∈ 2..=256）",
            )
        })?;
        if data.len() != (n as usize).pow(3) {
            return Err(lut_err(
                "PF1005",
                text.lines().count(),
                format!("数据行数 {} ≠ N³ = {}", data.len(), (n as usize).pow(3)),
                format!("补齐或删减数据行至 N³ = {} 行（红通道最快序）", (n as usize).pow(3)).as_str(),
            ));
        }
        for k in 0..3 {
            if !(domain_max[k] > domain_min[k]) {
                return Err(lut_err(
                    "PF1007",
                    text.lines().count(),
                    format!("DOMAIN 退化：第 {k} 通道 max({}) ≤ min({})",
                        domain_max[k], domain_min[k]),
                    "修正 DOMAIN_MAX 使各分量严格大于 DOMAIN_MIN（退化域无采样意义）",
                ));
            }
        }
        Ok(Lut3D { size: n, domain_min, domain_max, data })
    }

    /// 三线性插值采样：输入先按域归一并钳制到 [0,1]，再映射到格点坐标
    /// `f = c·(N−1)`，8 角固定序加权和。
    ///
    /// 格点输入（`f` 为整数）时 frac = 0 ⇒ 权重 (1,0,…,0) ⇒ 逐位恒等。
    pub fn sample_trilinear(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let c = [r, g, b];
        let last = self.size - 1;
        let mut f = [0.0f32; 3];
        for k in 0..3 {
            let span = self.domain_max[k] - self.domain_min[k];
            let t = (c[k] - self.domain_min[k]) / span; // 解析器保证 span > 0
            f[k] = t.clamp(0.0, 1.0) * last as f32;
        }
        let mut i0 = [0u32; 3];
        let mut i1 = [0u32; 3];
        let mut t = [0.0f32; 3];
        for k in 0..3 {
            let fl = f[k].floor(); // IEEE 精确舍入（D-29）
            let base = fl as u32; // f ∈ [0, N−1] ⇒ base ∈ [0, N−1]
            i0[k] = base.min(last);
            i1[k] = (base + 1).min(last);
            t[k] = f[k] - fl;
        }
        let (tx, ty, tz) = (t[0], t[1], t[2]);
        let n = self.size;
        let g8 = |x: u32, y: u32, z: u32| -> [f32; 3] {
            self.data[(x + y * n + z * n * n) as usize]
        };
        let c000 = g8(i0[0], i0[1], i0[2]);
        let c100 = g8(i1[0], i0[1], i0[2]);
        let c010 = g8(i0[0], i1[1], i0[2]);
        let c110 = g8(i1[0], i1[1], i0[2]);
        let c001 = g8(i0[0], i0[1], i1[2]);
        let c101 = g8(i1[0], i0[1], i1[2]);
        let c011 = g8(i0[0], i1[1], i1[2]);
        let c111 = g8(i1[0], i1[1], i1[2]);

        let (mx, my) = (1.0 - tx, 1.0 - ty);
        let w000 = mx * my * (1.0 - tz);
        let w100 = tx * my * (1.0 - tz);
        let w010 = mx * ty * (1.0 - tz);
        let w110 = tx * ty * (1.0 - tz);
        let w001 = mx * my * tz;
        let w101 = tx * my * tz;
        let w011 = mx * ty * tz;
        let w111 = tx * ty * tz;

        let mut out = [0.0f32; 3];
        for k in 0..3 {
            out[k] = w000 * c000[k]
                + w100 * c100[k]
                + w010 * c010[k]
                + w110 * c110[k]
                + w001 * c001[k]
                + w101 * c101[k]
                + w011 * c011[k]
                + w111 * c111[k];
        }
        out
    }
}

fn parse_u32(rest: &str, line_no: usize) -> Result<u32, LutError> {
    let tok = rest
        .split_whitespace()
        .next()
        .ok_or_else(|| lut_err("PF1004", line_no, "LUT_3D_SIZE 缺少数值".to_string(), "写成 `LUT_3D_SIZE N`，N ∈ 2..=256"))?;
    tok.parse::<u32>().map_err(|_| {
        lut_err(
            "PF1004",
            line_no,
            format!("“{tok}”不是合法格点数"),
            "LUT_3D_SIZE 后接十进制整数 N ∈ 2..=256",
        )
    })
}

/// 解析三个浮点（数据行 / DOMAIN 行）；多余列与非法浮点均拒绝（PF1006）。
fn parse_rgb(rest: &str, line_no: usize) -> Result<[f32; 3], LutError> {
    let mut it = rest.split_whitespace();
    let mut v = [0.0f32; 3];
    for slot in v.iter_mut() {
        let tok = it.next().ok_or_else(|| {
            lut_err(
                "PF1006",
                line_no,
                "行内浮点不足 3 个（需要 r g b）".to_string(),
                "数据行 / DOMAIN 行必须恰含 3 个浮点数",
            )
        })?;
        let val: f32 = tok.parse().map_err(|_| {
            lut_err(
                "PF1006",
                line_no,
                format!("“{tok}”不是合法浮点"),
                "检查该行是否混入非数值令牌（未知指令行会被当作数据行拒绝）",
            )
        })?;
        if !val.is_finite() {
            return Err(lut_err(
                "PF1006",
                line_no,
                format!("数值 {val} 非有限（NaN/Inf）"),
                ".cube 数据必须是有限浮点数",
            ));
        }
        *slot = val;
    }
    if it.next().is_some() {
        return Err(lut_err(
            "PF1006",
            line_no,
            "行内浮点多于 3 个".to_string(),
            "数据行 / DOMAIN 行必须恰含 3 个浮点数",
        ));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// ApplyLut —— TileEffect（halo 0：逐像素点操作，零邻域）
// ---------------------------------------------------------------------------

/// LUT 应用效果：逐像素 RGB 过 3D LUT（三线性插值），alpha 直通。
pub struct ApplyLut {
    /// 已加载的查找表。
    pub lut: Lut3D,
}

impl ApplyLut {
    /// 以给定 LUT 构造。
    pub fn new(lut: Lut3D) -> Self {
        ApplyLut { lut }
    }
}

impl TileEffect for ApplyLut {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let px = window.get(x, y);
                let rgb = self.lut.sample_trilinear(px[0], px[1], px[2]);
                out.set(x, y, [rgb[0], rgb[1], rgb[2], px[3]]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试：恒等逐位（关卡 2 / V12）/ 三线性数值 / .cube 解析
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 格点输入图：17×17，R/G/B 覆盖全部 k/16 格点值（k/16 为 f16/f32
    /// 双域精确值——逐位恒等的构造前提）。
    fn grid_input() -> Fbuf16F {
        let mut fb = Fbuf16F::new(17, 17);
        for y in 0..17u32 {
            for x in 0..17u32 {
                let r = x as f32 / 16.0;
                let g = y as f32 / 16.0;
                let b = ((x + y) % 17) as f32 / 16.0;
                fb.set(x, y, [r, g, b, 0.5]);
            }
        }
        fb
    }

    #[test]
    fn checkpoint2_identity_lut_grid_inputs_bit_exact() {
        // 关卡 2：恒等 LUT + 格点输入 → 输出与输入位模式逐位一致。
        let lut = Lut3D::identity(17);
        let effect = ApplyLut::new(lut);
        assert_eq!(effect.halo_radius(), 0, "逐像素点操作必须申报 halo 0");

        let input = grid_input();
        let out = effect.eval_window(&input);
        assert_eq!(
            out.pixels, input.pixels,
            "恒等 LUT 对格点输入必须逐位一致（f16 位模式全等）"
        );
    }

    #[test]
    fn trilinear_midpoint_of_identity_lut_is_exact() {
        // N=2 恒等 LUT：立方体 8 角 ∈ {0,1}³；输入 (½,½,½) 的 8 权重全为
        // 1/8（2 的幂），加权和精确 = 0.5。
        let lut = Lut3D::identity(2);
        let out = lut.sample_trilinear(0.5, 0.5, 0.5);
        assert_eq!(out, [0.5, 0.5, 0.5], "三线性中点必须精确等于 0.5");
    }

    #[test]
    fn identity_lut_is_identity_up_to_fp_tolerance_off_grid() {
        // 恒等 LUT 数学上处处恒等（三线性精确重构线性函数）；
        // 非格点输入允许 f32 舍入级容差（固定序 ⇒ 可复现，只是不逐位）。
        let lut = Lut3D::identity(8);
        for &(r, g, b) in &[
            (0.1, 0.2, 0.3),
            (0.37, 0.61, 0.83),
            (0.999, 0.001, 0.5),
        ] {
            let out = lut.sample_trilinear(r, g, b);
            for k in 0..3 {
                let want = [r, g, b][k];
                assert!(
                    (out[k] - want).abs() < 1e-6,
                    "恒等 LUT 非格点采样偏差过大：{out:?} vs ({r},{g},{b})"
                );
            }
        }
    }

    #[test]
    fn parse_cube_minimal_red_fastest_order() {
        // 红通道最快序：第 2 数据行 = (r=1,g=0,b=0) 角 → data[1]。
        let text = "# 最小 .cube\n\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = Lut3D::parse_cube(text).expect("合法最小 .cube 必须解析成功");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.domain_min, [0.0; 3], "DOMAIN_MIN 默认 0");
        assert_eq!(lut.domain_max, [1.0; 3], "DOMAIN_MAX 默认 1");
        assert_eq!(lut.data[0], [0.0, 0.0, 0.0], "红最快序：首行 = (0,0,0) 角");
        assert_eq!(lut.data[1], [1.0, 0.0, 0.0], "第二行 = (1,0,0) 角（r 最快）");
        assert_eq!(lut.data[4], [0.0, 0.0, 1.0], "b·N² 项：第 5 行 = (0,0,1) 角");
    }

    #[test]
    fn parse_cube_domain_remapping_and_clamp() {
        let text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0.25 0.25 0.25\nDOMAIN_MAX 0.75 0.75 0.75\n\
                    0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = Lut3D::parse_cube(text).expect("带 DOMAIN 的 .cube 必须解析成功");
        // 域中心 → 归一 0.5 → 立方体中点。
        assert_eq!(lut.sample_trilinear(0.5, 0.5, 0.5), [0.5, 0.5, 0.5]);
        // 域下界 → 归一 0 → 角 (0,0,0)。
        assert_eq!(lut.sample_trilinear(0.25, 0.25, 0.25), [0.0, 0.0, 0.0]);
        // 越域输入钳制到 [0,1]：1.5 → 归一并钳到 1 → 角 (1,·,·)。
        let out = lut.sample_trilinear(1.5, 0.5, 0.5);
        assert_eq!(out, [1.0, 0.5, 0.5]);
    }

    #[test]
    fn parse_cube_rejects_malformed_files_with_structured_errors() {
        // 缺 SIZE → PF1003（文件级，line 0）。
        let e = Lut3D::parse_cube("0 0 0\n").expect_err("缺声明必须拒绝");
        assert_eq!((e.code, e.line), ("PF1003", 0));
        assert!(e.fix.contains("LUT_3D_SIZE"));

        // SIZE 越界 → PF1004（行 1）。
        let e = Lut3D::parse_cube("LUT_3D_SIZE 1\n").expect_err("N<2 必须拒绝");
        assert_eq!((e.code, e.line), ("PF1004", 1));

        // SIZE 重复 → PF1004。
        let e = Lut3D::parse_cube("LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n").expect_err("重复声明必须拒绝");
        assert_eq!((e.code, e.line), ("PF1004", 2));

        // 行数不足 → PF1005。
        let e = Lut3D::parse_cube("LUT_3D_SIZE 2\n0 0 0\n1 0 0\n").expect_err("行数不足必须拒绝");
        assert_eq!(e.code, "PF1005");

        // 非法浮点 → PF1006（行 2）。
        let e = Lut3D::parse_cube("LUT_3D_SIZE 2\nfoo 0 0\n").expect_err("坏浮点必须拒绝");
        assert_eq!((e.code, e.line), ("PF1006", 2));

        // 未知指令行被当数据行拒绝（列数不足）→ PF1006。
        let e = Lut3D::parse_cube("LUT_3D_SIZE 2\nWARP_DRIVE 1\n").expect_err("未知令牌必须拒绝");
        assert_eq!((e.code, e.line), ("PF1006", 2));

        // 域退化 → PF1007。
        let degenerate =
            "LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0.5 1 1\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let e = Lut3D::parse_cube(degenerate).expect_err("退化域必须拒绝");
        assert_eq!(e.code, "PF1007");
        assert!(e.fix.contains("DOMAIN_MAX"), "错误必须含建议段：{e}");
    }
}
