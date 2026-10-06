//! 纹理注册表（B14）：名称 → 纹理工厂。
//!
//! 语义权威：《架构总设计》v2.4 §2.2（纹理 = 程序化生成器特解，节点暴露
//! 形态 = 纹理节点）；注册表风格与 prism-postfx `EffectRegistry` 一致
//! （HashMap 索引 + 结构化三段式错误，宪章 C18/C19）。

use std::collections::HashMap;
use std::fmt;

use prism_render::tile::TileEffect;

use crate::gradient::Gradient;
use crate::noise::NoiseTexture;
use crate::paper::Paper;
use crate::pattern::Pattern;
use crate::voronoi::Voronoi;

/// 纹理工厂：构造一个默认参数的纹理实例。
pub type TextureFactory = fn() -> Box<dyn TileEffect>;

/// 注册表错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）。
/// 错误码取 PT 段（texture 命名空间）。
#[derive(Debug, Clone, PartialEq)]
pub struct TextureError {
    /// 错误码（PT1xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：注册名。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for TextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for TextureError {}

/// 纹理注册表：名称 → 工厂。
#[derive(Debug, Default)]
pub struct TextureRegistry {
    factories: HashMap<String, TextureFactory>,
}

impl TextureRegistry {
    /// 构造空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册纹理工厂；重名报错（`PT1001`）。
    pub fn register(&mut self, name: &str, factory: TextureFactory) -> Result<(), TextureError> {
        if self.factories.contains_key(name) {
            return Err(TextureError {
                code: "PT1001",
                location: format!("纹理注册表 · {name}"),
                problem: "注册名重复".to_string(),
                fix: format!("换一个未占用的名称，或先 `unregister(\"{name}\")` 再注册"),
            });
        }
        self.factories.insert(name.to_string(), factory);
        Ok(())
    }

    /// 注销一个纹理。
    pub fn unregister(&mut self, name: &str) -> bool {
        self.factories.remove(name).is_some()
    }

    /// 按名构造默认参数纹理实例；未注册名报错（`PT1002`）。
    pub fn create(&self, name: &str) -> Result<Box<dyn TileEffect>, TextureError> {
        let factory = self.factories.get(name).ok_or_else(|| TextureError {
            code: "PT1002",
            location: format!("纹理注册表 · {name}"),
            problem: "未注册的纹理名".to_string(),
            fix: format!("使用已注册名称之一（见 `names()`），或先 `register(\"{name}\", …)`"),
        })?;
        Ok(factory())
    }

    /// 是否已注册。
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// 全部注册名，**字典序**。
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.factories.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

/// 官方预置纹理注册表（B14 程序化纹理族全集）。
pub fn default_registry() -> TextureRegistry {
    let mut reg = TextureRegistry::new();
    reg.register("paper", || Box::new(Paper::default())).unwrap();
    reg.register("noise", || Box::new(NoiseTexture::default())).unwrap();
    reg.register("gradient", || Box::new(Gradient::two_color([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))).unwrap();
    reg.register("pattern", || Box::new(Pattern::default())).unwrap();
    reg.register("voronoi", || Box::new(Voronoi::default())).unwrap();
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_contains_b14_texture_family() {
        let reg = default_registry();
        for name in ["paper", "noise", "gradient", "pattern", "voronoi"] {
            assert!(reg.contains(name), "官方预置缺 {name}");
        }
        assert_eq!(reg.names().len(), 5);
        let names = reg.names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn create_builds_all_and_duplicate_rejected() {
        let reg = default_registry();
        for name in reg.names() {
            reg.create(name).expect("已注册名必须可构造");
        }
        let mut reg2 = TextureRegistry::new();
        reg2.register("paper", || Box::new(Paper::default())).unwrap();
        let err = reg2.register("paper", || Box::new(Paper::default())).expect_err("重名拒绝");
        assert_eq!(err.code, "PT1001");
        assert!(err.fix.contains("unregister"), "必须含建议段：{err}");
    }

    #[test]
    fn unknown_name_reports_fix() {
        let reg = TextureRegistry::new();
        let err = match reg.create("paper") {
            Err(e) => e,
            Ok(_) => panic!("空表必须报错"),
        };
        assert_eq!(err.code, "PT1002");
        assert!(err.fix.contains("register"));
    }
}
