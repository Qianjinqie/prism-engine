//! 效果注册表：名称 → 效果工厂。
//!
//! 语义权威：效果=节点，`post { chain [...] }` 任意
//! 排序任意组合——注册表是 chain 按名取效果的索引；
//! PostFX = S2/S11/S5 特解族，节点暴露形态 = 效果节点；注册表风格
//! 与 prism-core `NodeRegistry` 一致（HashMap 索引 + 结构化三段式错误，
//! 宪章 C18：在哪 / 什么事 / 怎么改；C19：必含建议段）。
//!
//! # 参数面（Mini Code 边界声明）
//!
//! 工厂按**默认参数**构造效果实例；参数覆写走各效果结构体的具名字段
//! （类型安全、零解析开销）。schema 化参数（DSL `post` 块 → 校验 → 覆写）
//! 随 A4 codegen 接入，本注册表只承担"按名取效果"。

use std::collections::HashMap;
use std::fmt;

use prism_render::tile::TileEffect;

use crate::denoise::AtrousSpatial;
use crate::effects::{
    Blur, Bloom, ChromaticAberration, ColorManage, Dissolve, Glitch, Grade, Grain, Halftone,
    Kuwahara, LayeredBloom, Lut3D, MandelbrotBackground, Pixelate, Plasma, Sharpen, Sketch, Toon,
    Vignette, Watercolor,
};
use crate::effects::{Halation, HalationAndBloom, FilmGrain};
use crate::effects::{ChromaKey, ChromaKeyEffect, LumaKey, LumaKeyEffect, PathMask, PathMaskEffect};
use crate::warp::Warp;

/// 效果工厂：构造一个默认参数的效果实例。
pub type EffectFactory = fn() -> Box<dyn TileEffect>;

/// 注册表错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）。
/// 错误码取 PF 段（postfx 命名空间；段位表单点在 prism-core lib.rs，PF 段
/// 为本 crate 新增段——增补条目见完工汇报缺口上报）。
#[derive(Debug, Clone, PartialEq)]
pub struct EffectError {
    /// 错误码（PF1xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：注册名。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for EffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for EffectError {}

/// 效果注册表：名称 → 工厂（官方预置效果与后续数据包效果汇入同一张表，
/// 对齐"双通道汇入同一注册表"的形态）。
#[derive(Debug, Default)]
pub struct EffectRegistry {
    factories: HashMap<String, EffectFactory>,
}

impl EffectRegistry {
    /// 构造空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个效果工厂；重名报错（`PF1001`）。
    pub fn register(&mut self, name: &str, factory: EffectFactory) -> Result<(), EffectError> {
        if self.factories.contains_key(name) {
            return Err(EffectError {
                code: "PF1001",
                location: format!("效果注册表 · {name}"),
                problem: "注册名重复".to_string(),
                fix: format!("换一个未占用的名称，或先 `unregister(\"{name}\")` 再注册"),
            });
        }
        self.factories.insert(name.to_string(), factory);
        Ok(())
    }

    /// 注销一个效果（数据包卸载路径）。
    pub fn unregister(&mut self, name: &str) -> bool {
        self.factories.remove(name).is_some()
    }

    /// 按名构造默认参数效果实例；未注册名报错（`PF1002`）。
    pub fn create(&self, name: &str) -> Result<Box<dyn TileEffect>, EffectError> {
        let factory = self.factories.get(name).ok_or_else(|| EffectError {
            code: "PF1002",
            location: format!("效果注册表 · {name}"),
            problem: "未注册的效果名".to_string(),
            fix: format!(
                "使用已注册名称之一（见 `names()`），或先 `register(\"{name}\", …)`"
            ),
        })?;
        Ok(factory())
    }

    /// 是否已注册。
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// 全部注册名，**字典序**（消序用排序， 工程四律）。
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.factories.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

/// 官方预置效果注册表（清单全集：12 效果 + denoise + warp）。
///
/// 名称取宪章 C4–C6 行业对照表词汇（`lut3d` = 宪章对照 #8
/// "LUT（达芬奇、Resolve、Nuke）→ lut3d 节点"；`ca` = "CA"）。
pub fn default_registry() -> EffectRegistry {
    let mut reg = EffectRegistry::new();
    // 逐条 register；官方预置名集无重复，重复即编程错误（unwrap 暴露）。
    reg.register("blur", || Box::new(Blur::default())).unwrap();
    reg.register("bloom", || Box::new(Bloom::default())).unwrap();
    reg.register("vignette", || Box::new(Vignette::default())).unwrap();
    reg.register("grain", || Box::new(Grain::default())).unwrap();
    reg.register("dissolve", || Box::new(Dissolve::default())).unwrap();
    reg.register("glitch", || Box::new(Glitch::default())).unwrap();
    reg.register("grade", || Box::new(Grade::default())).unwrap();
    reg.register("ca", || Box::new(ChromaticAberration::default())).unwrap();
    reg.register("sharpen", || Box::new(Sharpen::default())).unwrap();
    reg.register("plasma", || Box::new(Plasma::default())).unwrap();
    reg.register("mandelbrot_bg", || Box::new(MandelbrotBackground::default())).unwrap();
    reg.register("lut3d", || {
        Box::new(crate::effects::ApplyLut::new(Lut3D::identity(17)))
    })
    .unwrap();
    reg.register("denoise", || Box::new(AtrousSpatial::default())).unwrap();
    reg.register("warp", || Box::new(Warp::default())).unwrap();
    // ---- C1 遮罩 / ROTO 族（深度合成与遮罩族）----
    // PathMask：单位正方形默认路径（调用方覆写 points/羽化/反相）。
    reg.register("path_mask", || {
        Box::new(PathMaskEffect {
            mask: PathMask::new(vec![
                prism_raster::geom::Point::new(0.0, 0.0),
                prism_raster::geom::Point::new(1.0, 0.0),
                prism_raster::geom::Point::new(1.0, 1.0),
                prism_raster::geom::Point::new(0.0, 1.0),
            ]),
        })
    })
    .unwrap();
    reg.register("luma_key", || Box::new(LumaKeyEffect { key: LumaKey::default() })).unwrap();
    reg.register("chroma_key", || Box::new(ChromaKeyEffect { key: ChromaKey::default() })).unwrap();
    // ---- B14 画面风格引擎：胶片仿真包 + NPR 族 ----
    reg.register("halation", || Box::new(Halation::default())).unwrap();
    reg.register("layered_bloom", || Box::new(LayeredBloom::default())).unwrap();
    reg.register("halation_bloom", || Box::new(HalationAndBloom::default())).unwrap();
    reg.register("film_grain", || Box::new(FilmGrain::default())).unwrap();
    reg.register("toon", || Box::new(Toon::default())).unwrap();
    reg.register("kuwahara", || Box::new(Kuwahara::default())).unwrap();
    reg.register("halftone", || Box::new(Halftone::default())).unwrap();
    reg.register("watercolor", || Box::new(Watercolor::default())).unwrap();
    reg.register("pixelate", || Box::new(Pixelate::default())).unwrap();
    reg.register("sketch", || Box::new(Sketch::default())).unwrap();
    // ---- C2 色彩与 HDR：ACES 输出变换节点 + 色彩管理节点 ----
    // `aces_output`：线性 sRGB → ACES 输出变换（RRT+ODT 拟合准确档）→ sRGB。
    reg.register("aces_output", || {
        Box::new(ColorManage {
            aces: true,
            oetf: crate::effects::color::Oetf::Srgb,
            ..ColorManage::default()
        })
    })
    .unwrap();
    // `color`：声明输入/输出色彩空间 + 变换链（PQ/HLG 经 OETF 参数配置）。
    reg.register("color", || Box::new(ColorManage::default())).unwrap();
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_contains_b4_checklist() {
        let reg = default_registry();
        // 12 效果 + denoise + warp，一个不少。
        for name in [
            "blur", "bloom", "vignette", "grain", "dissolve", "glitch", "grade", "ca",
            "sharpen", "plasma", "mandelbrot_bg", "lut3d", "denoise", "warp",
        ] {
            assert!(reg.contains(name), "官方预置缺 {name}");
        }
        // ---- B14 画面风格引擎：胶片仿真包 + NPR 族 ----
        for name in [
            "halation", "layered_bloom", "halation_bloom", "film_grain",
            "toon", "kuwahara", "halftone", "watercolor", "pixelate", "sketch",
        ] {
            assert!(reg.contains(name), "B14 预置缺 {name}");
        }
        assert_eq!(reg.names().len(), 29);
        // ---- C2 色彩与 HDR ----
        for name in ["aces_output", "color"] {
            assert!(reg.contains(name), "C2 色彩节点缺 {name}");
        }
        // ---- C1 遮罩 / ROTO 族 ----
        for name in ["path_mask", "luma_key", "chroma_key"] {
            assert!(reg.contains(name), "C1 遮罩族缺 {name}");
        }
        // names() 字典序（确定性列举）。
        let names = reg.names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn create_builds_default_effects() {
        let reg = default_registry();
        for name in reg.names() {
            let effect = reg.create(name).expect("已注册名必须可构造");
            // 构造出的实例能报出自身 halo（TileEffect 接口贯通）。
            let _ = effect.halo_radius();
        }
    }

    #[test]
    fn duplicate_register_rejected_with_actionable_error() {
        let mut reg = EffectRegistry::new();
        reg.register("blur", || Box::new(Blur::default())).unwrap();
        let err = reg
            .register("blur", || Box::new(Blur::default()))
            .expect_err("重名必须拒绝");
        assert_eq!(err.code, "PF1001");
        // 三段式：建议段非空（宪章 C19 结构化要求）。
        assert!(err.fix.contains("unregister"), "错误必须含建议段：{err}");
        assert!(err.to_string().contains("什么事") && err.to_string().contains("建议"));
    }

    #[test]
    fn unknown_name_error_names_the_fix() {
        let reg = EffectRegistry::new();
        // dyn TileEffect 无 Debug ⇒ 不能用 expect_err，match 拆错误。
        let err = match reg.create("blur") {
            Err(e) => e,
            Ok(_) => panic!("空表查询必须报错"),
        };
        assert_eq!(err.code, "PF1002");
        assert!(err.fix.contains("register"), "错误必须含建议段：{err}");
    }

    #[test]
    fn unregister_removes_and_allows_reregister() {
        let mut reg = default_registry();
        assert!(reg.unregister("grain"));
        assert!(!reg.contains("grain"));
        reg.register("grain", || Box::new(Grain::default())).unwrap();
        assert!(reg.contains("grain"));
        assert!(!reg.unregister("grain_nonexistent"));
    }
}
