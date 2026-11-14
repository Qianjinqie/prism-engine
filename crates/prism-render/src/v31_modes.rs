//! V3.1 收尾 —— 四个大模式 + 独立精细开关。
//!
//! # 大模式由用户亲自命名，名字与备注原样带上，不许改名
//! | 大模式 | 备注 | 语义 |
//! |---|---|---|
//! | `一致` | 一致但较慢 | 老路径，逐字节一致，哈希锚 `78366f3e…` 留在这档 |
//! | `均衡` | 适中 | 画质按误差上限，速度与画质居中 |
//! | `快速` | 较快 | 画质按误差上限，速度优先 |
//! | `自定义` | 自定义参数 | 每个开关单独可拨 |
//!
//! **大模式 = 一组预设值，不是把细项吞掉**：[`V31Knobs`] 的每个开关独立可拨；
//! 拨了任一细项，[`V31Knobs::resolve`] 就把结果标成 [`V31Mode::Custom`]。
//! `一致` 档直接落在 [`V31Config::legacy`] 的数值语义上（16 spp / 不累积 /
//! 不降噪 / 不重建），只保留 CLI 规格覆写，故纯 CPU 单帧哈希锚不变。

use crate::v31_config::{
    QualityTier, ResTier, V31Config, DEFAULT_FRAMES, DEFAULT_FPS, GOLDEN_SPP,
};

/// 四个大模式（字面名即 CLI `--mode` 的取值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum V31Mode {
    /// 一致 —— 备注「一致但较慢」。
    Consistent,
    /// 均衡 —— 备注「适中」。
    Balanced,
    /// 快速 —— 备注「较快」。
    Fast,
    /// 自定义 —— 备注「自定义参数」。
    Custom,
}

impl V31Mode {
    /// 用户命名的字面标签（报告 / CLI 帮助原样使用）。
    pub fn label(self) -> &'static str {
        match self {
            V31Mode::Consistent => "一致",
            V31Mode::Balanced => "均衡",
            V31Mode::Fast => "快速",
            V31Mode::Custom => "自定义",
        }
    }

    /// 用户给的备注，原样带上、不许改名。
    pub fn note(self) -> &'static str {
        match self {
            V31Mode::Consistent => "一致但较慢",
            V31Mode::Balanced => "适中",
            V31Mode::Fast => "较快",
            V31Mode::Custom => "自定义参数",
        }
    }

    /// 计数器键 / 目录名用的稳定 ASCII 标签。
    pub fn tag(self) -> &'static str {
        match self {
            V31Mode::Consistent => "consistent",
            V31Mode::Balanced => "balanced",
            V31Mode::Fast => "fast",
            V31Mode::Custom => "custom",
        }
    }

    /// CLI 解析：中文名原样 + 大小写不敏感 ASCII 别名。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "一致" => return Some(V31Mode::Consistent),
            "均衡" => return Some(V31Mode::Balanced),
            "快速" => return Some(V31Mode::Fast),
            "自定义" => return Some(V31Mode::Custom),
            _ => {}
        }
        match s.to_ascii_lowercase().as_str() {
            "consistent" | "exact" => Some(V31Mode::Consistent),
            "balanced" => Some(V31Mode::Balanced),
            "fast" => Some(V31Mode::Fast),
            "custom" => Some(V31Mode::Custom),
            _ => None,
        }
    }

    /// 单帧耗时上限（ms；`自定义` 不设 = `None`，用户自担）。
    pub fn frame_budget_ms(self) -> Option<u64> {
        match self {
            V31Mode::Consistent => Some(1250),
            V31Mode::Balanced => Some(430),
            V31Mode::Fast => Some(150),
            V31Mode::Custom => None,
        }
    }

    /// 一组预设值（**不是**把细项吞掉：返回的是可被 [`V31Knobs`] 逐项覆写的完整配置）。
    ///
    /// 预设只填**配置**：`spp_per_frame` 四档恒 16，`new_spp` 按档给出复用
    /// 比例（`一致` = 16 零复用、`均衡` = 8、`快速` = 4、`自定义` 由 `--spp`
    /// 细项覆写），不改分辨率 / 不降噪 / 不滤波。V3.1 出片链路里 `一致` 档
    /// 单帧直渲（哈希锚档），其余三档走通用路径的稀疏网格 + 门控重建，档位
    /// 差异落在重建步长与阈值上（见 `v31_device::universal_reconstruct_stride`
    /// 与 `universal_recon_tuning`）。
    pub fn preset(self, res: ResTier) -> Result<V31Config, String> {
        let mut cfg: V31Config = direct_single_frame(res);
        cfg.new_spp = match self {
            // 逐帧 16 spp 直渲、零复用：与老路径逐字节一致。
            V31Mode::Consistent => GOLDEN_SPP,
            // 静止 / 零运动区域复用 8 个槽位。
            V31Mode::Balanced => 8,
            // 复用 12 个槽位（复用比例最高，速度优先）。
            V31Mode::Fast => 4,
            // 从零复用底线出发，`--spp` 覆写复用比例。
            V31Mode::Custom => GOLDEN_SPP,
        };
        cfg.mode = self;
        // 有真 GPU 自动启用；「一致」档逐字节一致，不碰 GPU 桥。
        cfg.gpu_auto = self != V31Mode::Consistent;
        Ok(cfg)
    }
}

/// 该档的画质上限（§8.1 分档口径）：`均衡` = [`JudgeLimits::BALANCED`]、
/// `快速` = [`JudgeLimits::FAST`]。`自定义` 由用户细项自担，`一致` 是分母，
/// 两者都按最严口径 [`JudgeLimits::STRICT`] 报，仅诊断不设门槛。
pub fn judge_limits(mode: V31Mode) -> crate::v31_config::JudgeLimits {
    match mode {
        V31Mode::Balanced => crate::v31_config::JudgeLimits::BALANCED,
        V31Mode::Fast => crate::v31_config::JudgeLimits::FAST,
        V31Mode::Custom | V31Mode::Consistent => crate::v31_config::JudgeLimits::STRICT,
    }
}

/// 「单帧直渲」基线：`V31Config::legacy` 的数值语义 + 指定分辨率档。
fn direct_single_frame(res: ResTier) -> V31Config {
    let mut c: V31Config = V31Config::legacy();
    let (w, h) = res.dims();
    c.width = w;
    c.height = h;
    c.res = res;
    c.quality = QualityTier::Final;
    c.spp_per_frame = GOLDEN_SPP;
    c.window = 1;
    // `legacy()` 的 fps/frames 是「不覆写」哨兵（0）；V3.1 快捷路径必须给出
    // 可用的默认值，否则 `--mode 一致 --frames N` 无 `--fps` 会 R5001。
    c.fps = DEFAULT_FPS;
    c.frames = DEFAULT_FRAMES;
    c
}

/// 精细控制：每个开关单独可拨。`None` = 没拨（沿用大模式预设）。
///
/// 拨了任一细项 ⇒ 结果自动成为 `自定义`（备注「自定义参数」），
/// 但**不会**把其它细项吞掉：所有 `None` 项继续沿用所选大模式的预设值。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct V31Knobs {
    pub accumulate: Option<bool>,
    pub denoise: Option<bool>,
    pub adaptive: Option<bool>,
    pub gpu_prefer: Option<bool>,
    pub gpu_auto: Option<bool>,
    /// 可选误差校验（默认关，不进热路径）。
    pub verify: Option<bool>,
    pub spp_per_frame: Option<u32>,
    pub window: Option<u32>,
    pub bounces: Option<u32>,
    pub min_spp: Option<u32>,
}

impl V31Knobs {
    /// 是否拨动了任一**渲染细项**（会改变像素 / 采样 / 速度，故翻转成自定义）。
    ///
    /// `verify` 是正交的**可选离线校验开关**：它不改变任何像素或采样，也
    /// 不进出片热路径，故**不算细项** —— 这样用户可以对「均衡 / 快速 /
    /// 一致」任一命名档单独打开校验，档名与备注仍原样保留，不会被误标
    /// 成「自定义参数」。其余细项（累积 / 降噪 / 自适应 / GPU / 重建 /
    /// 采样）拨任一项仍按用户要求自动成为自定义。
    pub fn any_dialed(&self) -> bool {
        self.accumulate.is_some()
            || self.denoise.is_some()
            || self.adaptive.is_some()
            || self.gpu_prefer.is_some()
            || self.gpu_auto.is_some()
            || self.spp_per_frame.is_some()
            || self.window.is_some()
            || self.bounces.is_some()
            || self.min_spp.is_some()
    }

    /// 把细项应用到预设上，返回最终生效的大模式标签。
    ///
    /// `一致` 档即便拨了细项也保留「逐字节一致」底线：只允许拨
    /// `verify` / `gpu_*`（不改变像素数值），其余细项被拒绝（返回 Err）。
    pub fn resolve(self, base: V31Mode, res: ResTier) -> Result<(V31Config, V31Mode), String> {
        let mut cfg: V31Config = base.preset(res)?;
        let mut mode: V31Mode = base;

        // `verify` 正交应用：不触发 `一致` 的数值底线检查，也不翻转档名。
        if let Some(v) = self.verify {
            cfg.verify = v;
        }

        if self.any_dialed() {
            if base == V31Mode::Consistent
                && (self.accumulate.is_some()
                    || self.denoise.is_some()
                    || self.adaptive.is_some()
                    || self.spp_per_frame.is_some()
                    || self.window.is_some()
                    || self.bounces.is_some()
                    || self.min_spp.is_some())
            {
                return Err(
                    "「一致」档必须逐字节一致：不可拨累积/降噪/自适应/重建/采样细项；\
                     需要改这些请选「均衡」「快速」或「自定义」"
                        .to_string(),
                );
            }
            mode = V31Mode::Custom;
            // 从「自定义」档语义出发会覆盖预设，这里改为在预设上逐项覆写。
            cfg.mode = V31Mode::Custom;
        }

        if let Some(v) = self.accumulate {
            cfg.accumulate = v;
        }
        if let Some(v) = self.denoise {
            cfg.denoise = v;
        }
        if let Some(v) = self.adaptive {
            cfg.adaptive = v;
        }
        if let Some(v) = self.gpu_prefer {
            cfg.gpu_prefer = v;
        }
        if let Some(v) = self.gpu_auto {
            cfg.gpu_auto = v;
        }
        if let Some(v) = self.spp_per_frame {
            // `--spp` 是复用比例细项：改逐帧真实新采样本数 new_spp（1..=16）。
            // 总槽位 spp_per_frame 恒 16 不动 ⇒ 等效 spp 不降、画质不降。
            cfg.new_spp = v.clamp(1, GOLDEN_SPP);
        }
        if let Some(v) = self.window {
            cfg.window = v.max(1);
        }
        if let Some(v) = self.bounces {
            cfg.bounces = v.max(1);
        }
        if let Some(v) = self.min_spp {
            cfg.min_spp = v;
        }
        Ok((cfg, mode))
    }

    /// 稳定摘要（报告 / 计数器键后缀用）。
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        macro_rules! push_bool {
            ($name:expr, $field:expr) => {
                if let Some(v) = $field {
                    parts.push(format!("{}={}", $name, if v { 1 } else { 0 }));
                }
            };
        }
        macro_rules! push_num {
            ($name:expr, $field:expr) => {
                if let Some(v) = $field {
                    parts.push(format!("{}={}", $name, v));
                }
            };
        }
        push_bool!("accum", self.accumulate);
        push_bool!("denoise", self.denoise);
        push_bool!("adaptive", self.adaptive);
        push_bool!("gpu_prefer", self.gpu_prefer);
        push_bool!("gpu_auto", self.gpu_auto);
        push_bool!("verify", self.verify);
        push_num!("spp", self.spp_per_frame);
        push_num!("window", self.window);
        push_num!("bounces", self.bounces);
        push_num!("min_spp", self.min_spp);
        if parts.is_empty() {
            "预设（未拨细项）".to_string()
        } else {
            parts.join(" ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_notes_are_verbatim() {
        assert_eq!(V31Mode::Consistent.label(), "一致");
        assert_eq!(V31Mode::Consistent.note(), "一致但较慢");
        assert_eq!(V31Mode::Balanced.label(), "均衡");
        assert_eq!(V31Mode::Balanced.note(), "适中");
        assert_eq!(V31Mode::Fast.label(), "快速");
        assert_eq!(V31Mode::Fast.note(), "较快");
        assert_eq!(V31Mode::Custom.label(), "自定义");
        assert_eq!(V31Mode::Custom.note(), "自定义参数");
        assert_eq!(V31Mode::parse("均衡"), Some(V31Mode::Balanced));
        assert_eq!(V31Mode::parse("FAST"), Some(V31Mode::Fast));
        assert_eq!(V31Mode::parse("自定义"), Some(V31Mode::Custom));
    }

    #[test]
    fn consistent_preset_is_byte_identical_legacy_numerics() {
        let legacy = V31Config::legacy();
        let cfg = V31Mode::Consistent.preset(ResTier::R720p).unwrap();
        assert_eq!(cfg.spp_per_frame, legacy.spp_per_frame);
        assert_eq!(cfg.window, legacy.window);
        assert_eq!(cfg.bounces, legacy.bounces);
        assert_eq!(cfg.accumulate, false);
        assert_eq!(cfg.denoise, false);
        assert_eq!(cfg.effective_spp(), GOLDEN_SPP);
    }

    #[test]
    fn all_four_modes_available_with_16_total_spp() {
        // 四档全部可用；总槽位恒 16，差异只来自 new_spp（复用比例）。
        for (m, want_new) in [
            (V31Mode::Consistent, GOLDEN_SPP),
            (V31Mode::Balanced, 8),
            (V31Mode::Fast, 4),
            (V31Mode::Custom, GOLDEN_SPP),
        ] {
            let c = m.preset(ResTier::R720p).expect("四档全部可用");
            assert_eq!(c.spp_per_frame, GOLDEN_SPP, "{m:?} 总槽位恒 16");
            assert_eq!(c.effective_spp(), GOLDEN_SPP, "{m:?} 等效 spp 不降");
            assert_eq!(c.new_spp, want_new, "{m:?} 复用比例");
            assert_eq!(c.window, 1, "{m:?}");
            assert_eq!(c.width, 1280);
            assert_eq!(c.height, 720);
            assert_eq!(c.mode, m);
        }
        assert_eq!(V31Mode::Consistent.frame_budget_ms(), Some(1250));
        assert_eq!(V31Mode::Balanced.frame_budget_ms(), Some(430));
        assert_eq!(V31Mode::Fast.frame_budget_ms(), Some(150));
        assert_eq!(V31Mode::Custom.frame_budget_ms(), None);
    }

    #[test]
    fn dialing_any_knob_becomes_custom_without_swallowing_others() {
        let mut k = V31Knobs::default();
        k.denoise = Some(false);
        let (cfg, mode) = k.resolve(V31Mode::Custom, ResTier::R720p).unwrap();
        assert_eq!(mode, V31Mode::Custom);
        assert_eq!(cfg.mode, V31Mode::Custom);
        assert!(!cfg.denoise);
        // 未拨的细项沿用自定义基线（单帧直渲），不被吞掉。
        assert_eq!(cfg.spp_per_frame, GOLDEN_SPP);
        assert_eq!(cfg.window, 1);
        assert_eq!(cfg.effective_spp(), GOLDEN_SPP);
    }

    #[test]
    fn consistent_rejects_numeric_knobs_but_allows_verify() {
        let mut k = V31Knobs::default();
        k.spp_per_frame = Some(4);
        assert!(k.resolve(V31Mode::Consistent, ResTier::R720p).is_err());
        let mut v = V31Knobs::default();
        v.verify = Some(true);
        let (cfg, mode) = v.resolve(V31Mode::Consistent, ResTier::R720p).unwrap();
        assert!(cfg.verify);
        // verify 是正交离线校验开关：不改变档名（命名档可单独开校验）。
        assert_eq!(mode, V31Mode::Consistent);
        assert_eq!(cfg.mode, V31Mode::Consistent);
        assert_eq!(cfg.spp_per_frame, GOLDEN_SPP);
    }

    #[test]
    fn verify_alone_keeps_named_mode_for_all_four() {
        for base in [
            V31Mode::Consistent,
            V31Mode::Balanced,
            V31Mode::Fast,
            V31Mode::Custom,
        ] {
            let mut k = V31Knobs::default();
            k.verify = Some(true);
            let (cfg, mode) = k.resolve(base, ResTier::R720p).unwrap();
            assert!(cfg.verify);
            assert_eq!(mode, base, "{base:?}");
            assert_eq!(cfg.mode, base);
            assert_eq!(cfg.spp_per_frame, GOLDEN_SPP);
        }
    }

    #[test]
    fn spp_knob_controls_reuse_ratio_not_total_spp() {
        // --spp 改 new_spp（复用比例），总槽位 spp_per_frame 恒 16 不动。
        let mut k = V31Knobs::default();
        k.spp_per_frame = Some(6);
        let (cfg, mode) = k.resolve(V31Mode::Custom, ResTier::R720p).unwrap();
        assert_eq!(mode, V31Mode::Custom);
        assert_eq!(cfg.new_spp, 6);
        assert_eq!(cfg.spp_per_frame, GOLDEN_SPP);
        assert_eq!(cfg.effective_spp(), GOLDEN_SPP);
        // 越界钳制到 [1,16]，绝不降总采样。
        let mut k2 = V31Knobs::default();
        k2.spp_per_frame = Some(999);
        let (cfg2, _) = k2.resolve(V31Mode::Custom, ResTier::R720p).unwrap();
        assert_eq!(cfg2.new_spp, GOLDEN_SPP);
    }

    #[test]
    fn knobs_summary_is_deterministic() {
        let mut k = V31Knobs::default();
        assert_eq!(k.summary(), "预设（未拨细项）");
        k.spp_per_frame = Some(2);
        k.verify = Some(true);
        assert_eq!(k.summary(), "verify=1 spp=2");
    }
}
