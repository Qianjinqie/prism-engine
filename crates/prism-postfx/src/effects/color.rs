//! color（C2：色彩管理节点——输入/输出色彩空间声明 + 变换链）。
//!
//! 语义权威：（"色彩空间是每个节点的显式参数，不是全局
//! 魔法"；"全部是色彩节点链（`grade`/`aces_output`/`lut3d`）"）、C2 行
//! （"ACEScg 工作空间切换；ACES 输出变换（拟合）；PQ/HLG OETF"）；
//! （管线顺序：线性域白平衡/曝光 → AP1→输出色域 3×3 →
//! ODT 拟合 → OETF）。
//!
//! # 单节点变换（[`ColorManage`]）
//! 一个节点声明 `input`/`output` 色彩空间（[`prism_math::colorspace::ColorSpace`]），
//! 装配完整变换链：矩阵互转（[`prism_math::colorspace::TransformChain`]）+
//! ACES 输出变换拟合（[`prism_math::aces::AcesOutput`]）+ OETF（sRGB/PQ/HLG）。
//! 节点链由多个 [`ColorManage`] 首尾相接构成（"色彩节点链"）。
//!
//! # 处理域
//! `Fbuf16F` 的 RGBA16F 缓冲（线性高精度管线）。节点对 RGB 三分量
//! 求值、alpha 直通；f32 运算、写回量化 f16（格锚）。
//! halo = 0（逐像素，可平移不变）。

use prism_io::fbuf::Fbuf16F;
use prism_math::aces::{AcesFit, AcesOutput};
use prism_math::colorspace::{ColorSpace, TransformChain};
use prism_math::hlg::hlg_oetf;
use prism_math::ocio_lut::OcioLut;
use prism_math::pq::pq_oetf;
use prism_render::tile::TileEffect;

/// 输出 OETF（传输函数）选择（"输出侧 OETF 可插拔"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Oetf {
    /// 无传输编码（保留线性值）。
    #[default]
    None,
    /// sRGB 显示 OETF（分段幂 1/2.4，BT.709 形式）。
    Srgb,
    /// PQ（SMPTE ST 2084）HDR 传输。
    Pq,
    /// HLG（ITU-R BT.2100）传输。
    Hlg,
}

/// 色彩管理节点：声明输入/输出色彩空间 + 可选 ACES 输出变换 + OETF。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorManage {
    /// 输入色彩空间（声明）。
    pub input: ColorSpace,
    /// 输出色彩空间（声明）。
    pub output: ColorSpace,
    /// 是否插入 ACES 输出变换（RRT+ODT 拟合）。
    pub aces: bool,
    /// ACES 档位（准确/风格）。
    pub aces_fit: AcesFit,
    /// 输出 OETF。
    pub oetf: Oetf,
}

impl Default for ColorManage {
    fn default() -> Self {
        ColorManage {
            input: ColorSpace::LinearSrgb,
            output: ColorSpace::LinearSrgb,
            aces: false,
            aces_fit: AcesFit::Fitted,
            oetf: Oetf::None,
        }
    }
}

/// sRGB 显示 OETF（线性 → 编码）：`x ≤ 0.0031308 → 12.92x`，否则
/// `1.055·x^{1/2.4} − 0.055`。幂经自研 `det_pow`（D-29：禁 libm）。
#[inline]
fn srgb_oetf(x: f32) -> f32 {
    let v = x.max(0.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * prism_math::det_pow::det_pow(v, 1.0 / 2.4) - 0.055
    }
}

impl ColorManage {
    /// 对单个像素应用整条变换链（供链装配/`eval_window` 复用的纯函数）。
    ///
    /// 顺序：①矩阵互转 `input → output`（经 ACEScg 枢纽）；
    /// ②若 `aces` 则插入 RRT+ODT 拟合（作用于线性 sRGB 域）；
    /// ③OETF 编码。alpha 直通。
    pub fn apply_color(&self, c: [f32; 4]) -> [f32; 4] {
        let mut rgb = [c[0], c[1], c[2]];

        // ① 矩阵互转：线性域之间经 ACEScg 枢纽；跨显示域时跳过（显示域
        //    由 OETF/ACES 承担，矩阵链 apply 返回 None）。
        let chain = TransformChain::new(self.input, self.output);
        if let Some(mapped) = chain.apply(rgb) {
            rgb = mapped;
        }

        // ② ACES 输出变换（RRT+ODT 拟合）：在显示线性 sRGB 域作用。
        if self.aces {
            rgb = AcesOutput::new(self.aces_fit).apply(rgb);
        }

        // ③ OETF 编码。
        match self.oetf {
            Oetf::None => {}
            Oetf::Srgb => {
                for v in rgb.iter_mut() {
                    *v = srgb_oetf(*v);
                }
            }
            Oetf::Pq => {
                for v in rgb.iter_mut() {
                    *v = pq_oetf(*v);
                }
            }
            Oetf::Hlg => {
                for v in rgb.iter_mut() {
                    *v = hlg_oetf(*v);
                }
            }
        }

        [rgb[0], rgb[1], rgb[2], c[3]]
    }

    /// V2.5-B1：带 OCIO config 的像素变换（**无 config ⇒ 逐字节同 [`Self::apply_color`]**）。
    ///
    /// 有 config 时：config 烘焙出的 Q24 定点 1D shaper + 3D cube 取代
    /// `ACES` 输出变换（避免双重 tone map），随后仍按本节点声明走
    /// `input → output` 矩阵与 OETF；alpha 直通。
    /// `None` 分支**直接调用** `apply_color`，一行新运算都不插入，故旧场景
    /// 逐字节不变（判据 V1）。
    pub fn apply_color_with_ocio(&self, c: [f32; 4], lut: Option<&OcioLut>) -> [f32; 4] {
        match lut {
            None => self.apply_color(c),
            Some(l) => {
                let mapped: [f32; 3] = l.apply_f32([c[0], c[1], c[2]]);
                let mut rgb: [f32; 3] = mapped;
                let chain = TransformChain::new(self.input, self.output);
                if let Some(m) = chain.apply(rgb) {
                    rgb = m;
                }
                rgb = self.encode_oetf(rgb);
                [rgb[0], rgb[1], rgb[2], c[3]]
            }
        }
    }

    /// OETF 编码（与 `apply_color` 第 ③ 段逐字相同；仅供 config 分支复用）。
    fn encode_oetf(&self, mut rgb: [f32; 3]) -> [f32; 3] {
        match self.oetf {
            Oetf::None => {}
            Oetf::Srgb => {
                for v in rgb.iter_mut() {
                    *v = srgb_oetf(*v);
                }
            }
            Oetf::Pq => {
                for v in rgb.iter_mut() {
                    *v = pq_oetf(*v);
                }
            }
            Oetf::Hlg => {
                for v in rgb.iter_mut() {
                    *v = hlg_oetf(*v);
                }
            }
        }
        rgb
    }
}

/// V2.5-B1：把 OCIO 管线接进既有 `TileEffect` 色彩节点链（同一 `Fbuf16F`
/// 求值器，不另起第二条色彩路径）。
#[derive(Debug, Clone, PartialEq)]
pub struct ColorManageOcio {
    /// 基础色彩节点（输入/输出空间 + OETF）。
    pub base: ColorManage,
    /// config.ocio 烘焙出的定点 LUT。
    pub lut: OcioLut,
}

impl TileEffect for ColorManageOcio {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                out.set(
                    x,
                    y,
                    self.base.apply_color_with_ocio(window.get(x, y), Some(&self.lut)),
                );
            }
        }
        out
    }
}

impl TileEffect for ColorManage {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                out.set(x, y, self.apply_color(window.get(x, y)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_identity_is_bit_exact() {
        // 全中性（同空间、无 ACES、无 OETF）节点必须逐位恒等。
        let mut win = Fbuf16F::new(8, 5);
        for y in 0..5u32 {
            for x in 0..8u32 {
                let v = (x * 13 + y * 7) as f32 / 80.0;
                win.set(x, y, [v, 1.0 - v, v * 0.25, 0.75]);
            }
        }
        let out = ColorManage::default().eval_window(&win);
        assert_eq!(out, win, "中性色彩管理节点必须逐位恒等");
    }

    #[test]
    fn aces_node_matches_engine_fn() {
        // 节点内 ACES 路径与 prism-math 引擎函数一致（Import Before Use）。
        let node = ColorManage { aces: true, ..ColorManage::default() };
        let c = [0.18f32, 0.4, 0.9, 1.0];
        let got = node.apply_color(c);
        let want = AcesOutput::new(AcesFit::Fitted).apply([0.18, 0.4, 0.9]);
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-6, "节点 ACES 与引擎函数不一致");
        }
        assert_eq!(got[3], 1.0, "alpha 直通");
    }

    #[test]
    fn pq_hlg_oetf_nodes_encode() {
        // PQ/HLG 编码节点：线性 0 → 0，线性 1 → 编码接近 1。
        let pq_node = ColorManage { oetf: Oetf::Pq, ..ColorManage::default() };
        assert!(pq_node.apply_color([0.0, 0.0, 0.0, 1.0])[0].abs() < 1e-6);
        assert!((pq_node.apply_color([1.0, 1.0, 1.0, 1.0])[0] - 1.0).abs() < 1e-4);

        let hlg_node = ColorManage { oetf: Oetf::Hlg, ..ColorManage::default() };
        assert!(hlg_node.apply_color([0.0, 0.0, 0.0, 1.0])[0].abs() < 1e-6);
        assert!((hlg_node.apply_color([1.0, 1.0, 1.0, 1.0])[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn srgb_oetf_anchors() {
        // sRGB OETF 锚点：0 → 0，1 → 1，线性 0.0031308 段界连续。
        assert!(srgb_oetf(0.0).abs() < 1e-9);
        assert!((srgb_oetf(1.0) - 1.0).abs() < 1e-5);
        let lo = 12.92 * 0.003_130_8;
        let hi = 1.055 * prism_math::det_pow::det_pow(0.003_130_8, 1.0 / 2.4) - 0.055;
        assert!((lo - hi).abs() < 1e-4, "OETF 段界不连续：{lo} vs {hi}");
    }

    #[test]
    fn color_chain_composes() {
        // 色彩节点链：两个节点首尾相接（sRGB → ACEScg → sRGB 经节点）。
        let to_aces = ColorManage {
            input: ColorSpace::LinearSrgb,
            output: ColorSpace::AcesCg,
            ..ColorManage::default()
        };
        let to_srgb = ColorManage {
            input: ColorSpace::AcesCg,
            output: ColorSpace::LinearSrgb,
            ..ColorManage::default()
        };
        let c = [0.2f32, 0.5, 0.8, 1.0];
        let mid = to_aces.apply_color(c);
        let back = to_srgb.apply_color(mid);
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "节点链往返失真：{back:?}");
        }
    }

    #[test]
    fn color_manage_is_pure() {
        let node = ColorManage {
            aces: true,
            oetf: Oetf::Srgb,
            ..ColorManage::default()
        };
        let mut win = Fbuf16F::new(6, 4);
        let mut state: u64 = 0xC0FE;
        for y in 0..4u32 {
            for x in 0..6u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        assert_eq!(node.eval_window(&win), node.eval_window(&win), "纯函数契约");
        assert_eq!(node.halo_radius(), 0);
    }
}
