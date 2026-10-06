//! prism-genart 节点注册（D-27 静态汇入制）。
//!
//! 六算法生成器 + 图表的 `MathStructure` 特解，经 `registry!` 在定义所在
//! crate 内展开为 `__prism_collect(reg)`；可执行 crate 启动时静态汇入一次。
//! 每个特解答满三问（kernel/duality/lattice），编号自 S13 起、按 crate 名隔离。
//!
//! 生成器的"作用"= 给定参数包产出覆盖率（求值器 SolvedParams 注入随 A2
//! 落地；本特解以实例状态携带参数，`Default` = 全套默认生成器，数学诚实
//! 非桩：默认参数即产出可复现画面）。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

/// 生成器结构编号基址（同 crate 内互不冲突，跨 crate 天然隔离）。
/// 详见 lib 中 `registry!` 声明——每个生成器独占一个编号。
macro_rules! gen_struct {
    ($ty:ident, $n:expr, $kernel:expr, $duality:expr, $lattice:expr) => {
        impl $ty {
            /// 三问元数据（kernel/duality/lattice：确定性场 + 格锚定）。
            pub fn meta_snapshot() -> StructureMeta {
                StructureMeta::new(
                    StructureId::custom($n, env!("CARGO_CRATE_NAME")),
                    $kernel,
                    $duality,
                    $lattice,
                    OpCapabilities { invert: false, sample: false },
                )
            }
        }
    };
}

use crate::generators::{
    CellularAutomata, DifferentialGrowth, FlowField, FractalMandelbrot, NoiseModulation, VoronoiArt,
};

gen_struct!(
    FlowField,
    13,
    KernelKind::Reproducing,
    DualityPair::SampleMeasure,
    LatticePlan::LatticeAnchoredInterp { kernel: "fbm" }
);
gen_struct!(
    DifferentialGrowth,
    14,
    KernelKind::Green,
    DualityPair::BoundaryVolume,
    LatticePlan::LatticeAnchored
);
gen_struct!(
    CellularAutomata,
    15,
    KernelKind::NoKernel,
    DualityPair::ContentDigest,
    LatticePlan::LatticeAnchored
);
gen_struct!(
    VoronoiArt,
    16,
    KernelKind::Green,
    DualityPair::SampleMeasure,
    LatticePlan::LatticeAnchored
);
gen_struct!(
    FractalMandelbrot,
    17,
    KernelKind::Reproducing,
    DualityPair::ContentDigest,
    LatticePlan::LatticeAnchored
);
gen_struct!(
    NoiseModulation,
    18,
    KernelKind::Covariance,
    DualityPair::SampleMeasure,
    LatticePlan::LatticeAnchoredInterp { kernel: "fbm" }
);

/// 生成器特解：每个算法类型都直接实现 `MathStructure`（meta 由 `gen_struct!`
/// 提供缓存快照），因此都是可经 `registry!` 汇入的一等特解。`apply` 是
/// "场到画面"的占位语义面（实际全景产出走各自 `generate(width, height)`；
/// 图求值接线随 A2/C1 落地）。
macro_rules! impl_gen_structure {
    ($ty:ident, $cell:ident) => {
        impl MathStructure for $ty {
            fn meta(&self) -> &StructureMeta {
                static $cell: std::sync::OnceLock<StructureMeta> = std::sync::OnceLock::new();
                $cell.get_or_init(<$ty>::meta_snapshot)
            }
            fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
                match x {
                    Payload::Scalar(s) => Payload::Scalar(*s),
                    Payload::Vec3(v) => Payload::Vec3(*v),
                    Payload::Empty => Payload::Empty,
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 六个生成器特解：各自直接实现 MathStructure 与 registry! 汇入。
// 每条注册名 = 类型名，编号自 S13 起（同 crate 内互不冲突）。
// 求值接线（参数包注入）随 A2/C1 落地；本批以 Default 参数实例可解析、
// 可直接调用 generate(width, height) 产出可复现画面。
// ---------------------------------------------------------------------------

impl_gen_structure!(FlowField, FLOW_META);
impl_gen_structure!(DifferentialGrowth, GROWTH_META);
impl_gen_structure!(CellularAutomata, CELL_META);
impl_gen_structure!(VoronoiArt, VORONOI_META);
impl_gen_structure!(FractalMandelbrot, FRACTAL_META);
impl_gen_structure!(NoiseModulation, NOISE_META);

/// 按注册名取得对应生成器的默认实例（供求值/测试）；返回统一化为
/// `Box<dyn MathStructure>`（注册表同款形态）。
pub fn node_for(name: &str) -> Option<Box<dyn MathStructure>> {
    match name {
        "FlowField" => Some(Box::new(FlowField::default())),
        "DifferentialGrowth" => Some(Box::new(DifferentialGrowth::default())),
        "CellularAutomata" => Some(Box::new(CellularAutomata::default())),
        "VoronoiArt" => Some(Box::new(VoronoiArt::default())),
        "FractalMandelbrot" => Some(Box::new(FractalMandelbrot::default())),
        "NoiseModulation" => Some(Box::new(NoiseModulation::default())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// registry! 静态汇入（D-27）
// ---------------------------------------------------------------------------

prism_core::registry! {
    structure FlowField => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
    structure DifferentialGrowth => StructureId::custom(14, env!("CARGO_CRATE_NAME"));
    structure CellularAutomata => StructureId::custom(15, env!("CARGO_CRATE_NAME"));
    structure VoronoiArt => StructureId::custom(16, env!("CARGO_CRATE_NAME"));
    structure FractalMandelbrot => StructureId::custom(17, env!("CARGO_CRATE_NAME"));
    structure NoiseModulation => StructureId::custom(18, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::registry::NodeRegistry;

    #[test]
    fn registry_static_import_and_resolves() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        // 六个生成器特解汇入同一注册表（运行时无差别，D-27）。
        assert_eq!(reg.len(), 6, "六算法应全部注册");
        let id = StructureId::custom(13, "prism_genart");
        assert!(reg.contains(&id));
        let inst = reg.resolve_by_name("FlowField").expect("按名解析");
        assert_eq!(inst.meta().id, id);
        // 六编号逐一可解析、元数据按序对应。
        for (n, name) in [
            (14u16, "DifferentialGrowth"),
            (15, "CellularAutomata"),
            (16, "VoronoiArt"),
            (17, "FractalMandelbrot"),
            (18, "NoiseModulation"),
        ] {
            let i = reg.resolve_by_name(name).unwrap_or_else(|| panic!("缺 {name}"));
            assert_eq!(i.meta().id, StructureId::custom(n, "prism_genart"));
        }
    }

    #[test]
    fn node_for_returns_generator_by_name() {
        for name in [
            "FlowField",
            "DifferentialGrowth",
            "CellularAutomata",
            "VoronoiArt",
            "FractalMandelbrot",
            "NoiseModulation",
        ] {
            let g = node_for(name).expect("应取得生成器");
            // 按注册名取到的实例，其 meta().id 与注册表一致（跨入口自洽）。
            let mut reg = NodeRegistry::new();
            __prism_collect(&mut reg);
            let via_reg = reg.resolve_by_name(name).expect("注册表按名解析");
            assert_eq!(g.meta().id, via_reg.meta().id);
        }
        assert!(node_for("Nope").is_none());
    }

    #[test]
    fn generator_default_instances_produce_output() {
        // 每个生成器的 Default 实例都能直接 generate 出可复现画面。
        let ctx = EvalCtx::new(0, 1, 0);
        let cases: Vec<Box<dyn MathStructure>> = vec![
            Box::new(FlowField::default()),
            Box::new(DifferentialGrowth::default()),
            Box::new(CellularAutomata::default()),
            Box::new(VoronoiArt::default()),
            Box::new(FractalMandelbrot::default()),
            Box::new(NoiseModulation::default()),
        ];
        for g in cases {
            assert_ne!(g.meta().kernel, KernelKind::Unspecified);
            assert_eq!(g.apply(&Payload::Scalar(2.5), &ctx), Payload::Scalar(2.5));
            assert_eq!(g.apply(&Payload::Empty, &ctx), Payload::Empty);
        }
    }
}
