//! 特解注册表：双通道汇入 + 三问校验 + `registry!` 静态汇入制。
//!
//! 语义权威：（registry! 与 alias.rs 双通道汇入同一
//! 注册表，运行时无差别）、D-27（静态汇入制：宏在定义所在 crate 内展开为
//! `__prism_collect(reg)` 式注册函数，可执行 crate 启动时静态汇入一次，禁
//! inventory / proc-macro）、（三问注册：问不齐注册表拒绝，三段式结构化
//! 报错——在哪 / 缺哪问 / 怎么补，宪章 C18/C19）。
//!
//! 双通道形态：官方特解（alias 通道，alias.rs 随 A4 落地）与自定义特解
//! （`registry!` 通道）都走本文件的注册 API，汇入同一张 [`NodeRegistry`]。

use std::collections::HashMap;
use std::fmt;

use crate::node::{
    DualityPair, KernelKind, LatticePlan, MathStructure, StructureId, StructureMeta,
};

/// 特解工厂：按需构造结构实例（求值器的"按签名复用"策略在求值层，第二批）。
pub type Factory = fn() -> Box<dyn MathStructure>;

/// 泛型特解的具体工厂（实例化后可强转为 [`Factory`] 函数指针）。
fn make_factory<T: MathStructure + Default + 'static>() -> Box<dyn MathStructure> {
    Box::new(T::default())
}

/// 注册条目：注册名 + 三问元数据快照 + 工厂。
#[derive(Debug)]
pub struct RegistryEntry {
    /// 注册名（`registry!` 通道 = 类型名；官方通道 = alias 名）。
    pub name: String,
    /// 三问元数据快照（汇入时校验通过的那份）。
    pub meta: StructureMeta,
    /// 特解工厂。
    pub factory: Factory,
}

/// 注册表错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）。
/// 错误码取 P3 段（图结构，宪章 C18 分段：P3xxx 图结构）。
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryError {
    /// 错误码（P3xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：结构 / 注册名。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实（缺哪问 / 哪类冲突）。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for RegistryError {}

/// 特解注册表：官方通道与 `registry!` 通道汇入的同一张表。
/// 运行时无差别——求值调度只认 [`StructureId`]，不认通道来源。
#[derive(Debug, Default)]
pub struct NodeRegistry {
    entries: HashMap<StructureId, RegistryEntry>,
    by_name: HashMap<String, StructureId>,
}

impl NodeRegistry {
    /// 构造空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个特解（官方通道入口：直接登记元数据与工厂）。
    ///
    /// 三问校验在此执行（D-27：校验落运行期注册汇入时）；编号越界、重复
    /// 注册按三段式结构化报错拒绝，永不静默（宪章 C20）。
    pub fn collect(
        &mut self,
        name: &str,
        meta: StructureMeta,
        factory: Factory,
    ) -> Result<(), RegistryError> {
        let location = format!("registry 汇入 `{}`（结构 {}）", name, meta.id);
        Self::check_meta(&location, &meta)?;

        if let Some(old) = self.entries.get(&meta.id) {
            return Err(RegistryError {
                code: "P3003",
                location,
                problem: format!(
                    "结构 {} 已被 `{}` 注册，重复注册（同 crate 同编号按冲突处理）",
                    meta.id, old.name
                ),
                fix: "换用未占用的编号；跨 crate 同编号无需处理（按 crate 名天然隔离）"
                    .to_string(),
            });
        }
        if self.by_name.contains_key(name) {
            return Err(RegistryError {
                code: "P3003",
                location,
                problem: format!("注册名 `{}` 已被占用", name),
                fix: "换一个未占用的注册名".to_string(),
            });
        }

        self.by_name.insert(name.to_string(), meta.id.clone());
        self.entries.insert(meta.id.clone(), RegistryEntry { name: name.to_string(), meta, factory });
        Ok(())
    }

    /// `registry!` 通道入口：实例化特解类型，核对声明 id 与 `meta().id` 一致后
    /// 走统一 [`collect`](Self::collect) 校验注册。
    pub fn collect_from<T: MathStructure + Default + 'static>(
        &mut self,
        name: &str,
        declared: StructureId,
    ) -> Result<(), RegistryError> {
        let instance = T::default();
        let meta = instance.meta().clone();
        if meta.id != declared {
            return Err(RegistryError {
                code: "P3002",
                location: format!("registry 汇入 `{}`", name),
                problem: format!(
                    "registry! 声明的结构（{}）与 meta().id（{}）不一致",
                    declared, meta.id
                ),
                fix: "两处写同一个 StructureId（推荐都用 StructureId::custom(n, env!(\"CARGO_CRATE_NAME\")))"
                    .to_string(),
            });
        }
        self.collect(name, meta, make_factory::<T>)
    }

    /// 三问校验 + 用户段编号边界（官方段 1–12，S13+ 起为自定义）。
    fn check_meta(location: &str, meta: &StructureMeta) -> Result<(), RegistryError> {
        if let StructureId::Custom { n, .. } = &meta.id {
            if *n <= StructureId::OFFICIAL_COUNT {
                return Err(RegistryError {
                    code: "P3004",
                    location: location.to_string(),
                    problem: format!(
                        "自定义结构编号 S{} 越界：官方段为 S1–S{}，用户段自 S13 起",
                        n,
                        StructureId::OFFICIAL_COUNT
                    ),
                    fix: format!(
                        "改用 ≥{} 的编号，如 StructureId::custom({}, env!(\"CARGO_CRATE_NAME\"))",
                        StructureId::OFFICIAL_COUNT + 1,
                        StructureId::OFFICIAL_COUNT + 1
                    ),
                });
            }
        }

        // 三问必答：任一问 Unspecified 即拒绝（三段式，宪章 C18/C19）。
        let mut missing: Vec<&str> = Vec::new();
        if meta.kernel == KernelKind::Unspecified {
            missing.push("问一 kernel（核是什么）");
        }
        if meta.duality == DualityPair::Unspecified {
            missing.push("问二 duality（配对哪一侧）");
        }
        if let LatticePlan::Unspecified = meta.lattice {
            missing.push("问三 lattice（格锚了吗）");
        }
        if !missing.is_empty() {
            return Err(RegistryError {
                code: "P3001",
                location: location.to_string(),
                problem: format!(
                    "三问校验未通过：{}未回答（Unspecified）",
                    missing.join("、")
                ),
                fix: "在 StructureMeta 填写三问：kernel = KernelKind::Convolution | Green | \
                      Transfer | Covariance | Reproducing | NoKernel；duality = DualityPair::\
                      BoundaryVolume | SampleMeasure | TimeFrequency | PrimalMultiplier | \
                      ContentDigest；lattice = LatticePlan::LatticeAnchored | \
                      LatticeAnchoredInterp { kernel }"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// 按结构标识解析一个特解实例（工厂产物；参数装配在求值层，第二批）。
    pub fn resolve(&self, id: &StructureId) -> Option<Box<dyn MathStructure>> {
        self.entries.get(id).map(|e| (e.factory)())
    }

    /// 按注册名解析（存档读档路径：图按名记录，读档时按当前注册表解析，D-23；
    /// 缺定义由调用方走结构化报错，绝不静默降级，宪章 C20）。
    pub fn resolve_by_name(&self, name: &str) -> Option<Box<dyn MathStructure>> {
        self.by_name.get(name).and_then(|id| self.resolve(id))
    }

    /// 查询某结构的元数据快照。
    pub fn meta_of(&self, id: &StructureId) -> Option<&StructureMeta> {
        self.entries.get(id).map(|e| &e.meta)
    }

    /// 某结构是否已注册。
    pub fn contains(&self, id: &StructureId) -> bool {
        self.entries.contains_key(id)
    }

    /// 已注册特解数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `registry!`（D-27 静态汇入制）：在定义所在 crate 内展开为
/// `__prism_collect(reg)` 式注册函数；可执行 crate（prism-app / 用户 bin）
/// 启动时静态汇入一次——无动态加载、无脚本层，静态链接即用。
///
/// 用法（贡献者通道，opt-in 内核词汇）：
///
/// ```
/// use prism_core::node::{
///     DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
///     SampleIndex, StructureId, StructureMeta,
/// };
/// use prism_core::{registry, NodeRegistry};
///
/// // 贡献者特解：任何满足 `MathStructure + Default + 'static` 的类型。
/// #[derive(Debug)]
/// struct RippleField {
///     meta: StructureMeta,
/// }
///
/// impl Default for RippleField {
///     fn default() -> Self {
///         Self {
///             meta: StructureMeta::new(
///                 StructureId::custom(13, "doc_demo_crate"),
///                 KernelKind::Covariance,
///                 DualityPair::SampleMeasure,
///                 LatticePlan::LatticeAnchored,
///                 OpCapabilities { invert: false, sample: false },
///             ),
///         }
///     }
/// }
///
/// impl MathStructure for RippleField {
///     fn meta(&self) -> &StructureMeta { &self.meta }
///     fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload { *x }
/// }
///
/// // 静态汇入：展开为 `__prism_collect(reg)`。
/// registry! {
///     structure RippleField => StructureId::custom(13, "doc_demo_crate");
/// }
///
/// // 汇入点：启动期调一次，三问齐备则注册成功。
/// // 注意：`__prism_collect` 内部对 `collect_from` 的Err 走**启动期 panic**
/// //（宪章 C20 绝不静默），故其返回类型是 `()` 而非 Result。
/// let mut reg = NodeRegistry::new();
/// __prism_collect(&mut reg);
/// assert_eq!(reg.len(), 1);
/// assert!(reg.resolve_by_name("RippleField").is_some());
/// ```
///
/// 汇入失败（三问不齐 / 编号越界 / id 不一致 / 重复注册）在启动期以三段式
/// 报错 panic——绝不静默（宪章 C20）；宏展开期仅查字段存在性（编译器类型
/// 检查 meta() 方法与字段），三问校验落运行期注册时（D-27）。
#[macro_export]
macro_rules! registry {
    ( $( structure $ty:ident => $sid:expr ; )+ ) => {
        #[doc(hidden)]
        pub fn __prism_collect(registry: &mut $crate::NodeRegistry) {
            $(
                if let Err(e) = registry.collect_from::<$ty>(stringify!($ty), $sid) {
                    panic!("registry! 静态汇入失败：{}", e);
                }
            )+
        }
    };
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{EvalCtx, Payload, SampleIndex};

    /// 测试特解（registry! 通道）：同 crate 用 registry! 声明，验证宏汇入路径。
    #[derive(Debug)]
    struct EchoSpec {
        meta: StructureMeta,
    }

    impl Default for EchoSpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::custom(42, env!("CARGO_CRATE_NAME")),
                    KernelKind::Covariance,
                    DualityPair::SampleMeasure,
                    LatticePlan::LatticeAnchored,
                    crate::node::OpCapabilities { invert: false, sample: true },
                ),
            }
        }
    }

    impl MathStructure for EchoSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Scalar(v) => Payload::Scalar(v * 2.0),
                other => *other,
            }
        }
        fn sample(&self, si: SampleIndex, _ctx: &EvalCtx) -> Payload {
            // 纯函数派生示意：同索引同值，无隐藏状态。
            Payload::Scalar(si.sample as f32 + si.pixel.0 as f32 * 0.001)
        }
    }

    registry! {
        structure EchoSpec => StructureId::custom(42, env!("CARGO_CRATE_NAME"));
    }

    /// 测试特解（官方通道形态）：直接工厂注册，模拟 alias.rs 落地后的官方汇入。
    #[derive(Debug)]
    struct OfficialSpec {
        meta: StructureMeta,
    }

    impl Default for OfficialSpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::PiecewisePolynomial,
                    KernelKind::Reproducing,
                    DualityPair::ContentDigest,
                    LatticePlan::LatticeAnchoredInterp { kernel: "quintic" },
                    crate::node::OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl MathStructure for OfficialSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            *x
        }
    }

    /// 故意缺一问的测试特解（问一 kernel 留 Unspecified）。
    #[derive(Debug)]
    struct MissingQuestion {
        meta: StructureMeta,
    }

    impl Default for MissingQuestion {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::custom(43, env!("CARGO_CRATE_NAME")),
                    KernelKind::Unspecified,
                    DualityPair::SampleMeasure,
                    LatticePlan::LatticeAnchored,
                    crate::node::OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl MathStructure for MissingQuestion {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            *x
        }
    }

    #[test]
    fn registry_macro_collects_into_same_registry() {
        // registry! 展开的 __prism_collect 是静态汇入入口（D-27）。
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 1);

        // 运行时无差别：按 id resolve 拿到实例，三问元数据与 apply/sample 可调。
        let id = StructureId::custom(42, env!("CARGO_CRATE_NAME"));
        let inst = reg.resolve(&id).expect("静态汇入后应可按 id resolve");
        assert_eq!(inst.meta().kernel, KernelKind::Covariance);
        assert_eq!(inst.meta().duality, DualityPair::SampleMeasure);

        let ctx = EvalCtx::new(7, 1, 0);
        assert_eq!(inst.apply(&Payload::Scalar(1.5), &ctx), Payload::Scalar(3.0));
        let si = SampleIndex { frame: 0, pixel: (3, 0), sample: 2, dimension: 0, node_id: 1 };
        assert_eq!(inst.sample(si, &ctx), Payload::Scalar(2.003));

        // 按名 resolve（D-23 读档路径支撑）：注册名 = 类型名。
        let by_name = reg.resolve_by_name("EchoSpec").expect("应可按名解析");
        assert_eq!(by_name.meta().id, id);
        assert!(reg.meta_of(&id).is_some());
        assert!(reg.contains(&id));
    }

    #[test]
    fn missing_question_is_rejected_with_three_part_error() {
        let mut reg = NodeRegistry::new();
        let err = reg
            .collect_from::<MissingQuestion>(
                "MissingQuestion",
                StructureId::custom(43, env!("CARGO_CRATE_NAME")),
            )
            .unwrap_err();

        // 错误码 + 三段式（C18）：在哪 / 缺哪问 / 怎么补（C19 建议段）。
        assert_eq!(err.code, "P3001");
        let text = err.to_string();
        assert!(text.starts_with("error[P3001] "), "首段应在哪：{}", text);
        assert!(text.contains("MissingQuestion"), "在哪段应含特解名：{}", text);
        assert!(text.contains("问一 kernel"), "什么事段应指出缺哪问：{}", text);
        assert!(text.contains("Unspecified"), "什么事段应含事实：{}", text);
        assert!(text.contains("建议："), "怎么补段必须有建议（C19）：{}", text);
        assert!(text.contains("KernelKind::Convolution"), "建议应给可粘贴写法：{}", text);

        // 拒绝后注册表保持干净（不静默降级，C20）。
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn dual_channels_merge_into_one_registry_runtime_indistinguishable() {
        let mut reg = NodeRegistry::new();

        // 官方通道：直接 collect（alias.rs 随 A4 落地后走同一 API）。
        reg.collect(
            "official.piecewise",
            OfficialSpec::default().meta,
            make_factory::<OfficialSpec>,
        )
        .expect("官方通道汇入应成功");

        // 自定义通道：registry! 静态汇入。
        __prism_collect(&mut reg);

        // 两条通道汇入同一张表：运行时无差别。
        assert_eq!(reg.len(), 2);
        let ctx = EvalCtx::new(0, 1, 0);
        let official = reg
            .resolve(&StructureId::PiecewisePolynomial)
            .expect("官方通道结构应可解析");
        assert_eq!(official.apply(&Payload::Scalar(1.0), &ctx), Payload::Scalar(1.0));
        let custom = reg
            .resolve(&StructureId::custom(42, env!("CARGO_CRATE_NAME")))
            .expect("registry! 通道结构应可解析");
        // EchoSpec 的特解语义 = ×2：同一 resolve 入口拿到的实例行为只由特解决定，
        // 与通道来源无关。
        assert_eq!(custom.apply(&Payload::Scalar(1.0), &ctx), Payload::Scalar(2.0));
    }

    #[test]
    fn declared_id_mismatch_is_rejected() {
        let mut reg = NodeRegistry::new();
        let err = reg
            .collect_from::<EchoSpec>("EchoSpec", StructureId::custom(99, "somewhere"))
            .unwrap_err();
        assert_eq!(err.code, "P3002");
        assert!(err.problem.contains("不一致"), "报错应指出 id 漂移：{}", err.problem);
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn custom_index_below_13_is_rejected() {
        let mut reg = NodeRegistry::new();
        let err = reg
            .collect(
                "bad",
                StructureMeta::new(
                    StructureId::custom(12, "c"),
                    KernelKind::Covariance,
                    DualityPair::SampleMeasure,
                    LatticePlan::LatticeAnchored,
                    crate::node::OpCapabilities { invert: false, sample: false },
                ),
                make_factory::<OfficialSpec>,
            )
            .unwrap_err();
        assert_eq!(err.code, "P3004");
        assert!(err.problem.contains("S13"), "报错应给出用户段边界：{}", err.problem);
    }

    #[test]
    fn duplicate_conflict_same_crate_and_isolation_across_crates() {
        let meta42 = |crate_name: &'static str| {
            StructureMeta::new(
                StructureId::custom(42, crate_name),
                KernelKind::Covariance,
                DualityPair::SampleMeasure,
                LatticePlan::LatticeAnchored,
                crate::node::OpCapabilities { invert: false, sample: false },
            )
        };

        // 同 crate 同编号重复注册 → 冲突拒绝（P3003）。
        let mut reg = NodeRegistry::new();
        reg.collect("first", meta42("prism-core"), make_factory::<OfficialSpec>)
            .expect("首次注册应成功");
        let err = reg
            .collect("second", meta42("prism-core"), make_factory::<OfficialSpec>)
            .unwrap_err();
        assert_eq!(err.code, "P3003");
        assert_eq!(reg.len(), 1);

        // 同名重复注册同样拒绝（注册名唯一）。
        let err = reg
            .collect("first", meta42("prism-core"), make_factory::<OfficialSpec>)
            .unwrap_err();
        assert_eq!(err.code, "P3003");

        // 跨 crate 同编号：按 crate 名天然隔离，互不冲突。
        let mut reg2 = NodeRegistry::new();
        reg2.collect("x", meta42("crate_a"), make_factory::<OfficialSpec>).unwrap();
        reg2.collect("y", meta42("crate_b"), make_factory::<OfficialSpec>).unwrap();
        assert_eq!(reg2.len(), 2);
    }

    #[test]
    fn resolve_by_name_missing_definition_returns_none() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        // 读档语义：缺定义 → None（调用方结构化报错），绝不静默编造（C20）。
        assert!(reg.resolve_by_name("NoSuchStructure").is_none());
    }
}
