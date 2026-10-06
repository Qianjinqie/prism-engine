//! audio/reactive 节点（B6 → prism-core 注册表；D-27 静态汇入制）。
//!
//! 节点 = 三元组的实例形态：本文件实现 [`MathStructure`] 的两个 reactive
//! 特解，经 `registry!`（宏展开为 `__prism_collect(reg)` 注册函数）静态汇入
//! `NodeRegistry`——可执行 crate（prism-app/用户 bin）启动时汇入一次，
//! 无动态加载、无脚本层。
//!
//! # 编号口径
//!
//! 走用户段 `StructureId::custom`（S13/S14 @ prism-audio）：官方段 S1–S12 是
//! 全局单例（注册表按 id 唯一键），B6 的谱分析与 C4 的 DCT 同属 S6——
//! 官方位留给 alias.rs（A4）统一分配，audio 节点以自定义段挂接，跨 crate
//! 编号天然隔离（D-23）。三问：核 = 卷积（加窗谱分析），对偶 = 时频，
//! 格锚 = 帧号/hop/win 整数格（帧级查表零浮点漂移）。
//!
//! # 数据接线（架构 §4.4）
//!
//! 烘焙表由 [`crate::reactive::install_bake`] 装入 `EvalCtx.snapshots`
//! （纯数据资产）；节点 `apply` 按当前帧查表返回 `Payload::Scalar`——
//! 求值纯函数契约不破（无隐藏状态、seek 等价）。烘焙缺失返回 `Payload::Empty`
//! （单位元；调度方显式装配数据，本层不静默编造数值）。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

use crate::reactive::{reactive_slot, ReactiveBake};

/// reactive 节点三问元数据（两节点同一答案：加窗谱分析的卷积核 × 时频对偶
/// × 整数格锚）。
fn reactive_meta(n: u16) -> StructureMeta {
    StructureMeta::new(
        StructureId::custom(n, env!("CARGO_CRATE_NAME")),
        KernelKind::Convolution,
        DualityPair::TimeFrequency,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: false },
    )
}

/// 自快照库读取烘焙表（缺数据返回 `None`，调用方降级为 `Payload::Empty`）。
fn read_bake(ctx: &EvalCtx) -> Option<ReactiveBake> {
    ReactiveBake::from_bytes(ctx.snapshots.get(&(reactive_slot(), 0))?)
}

/// AudioReactive 节点（S13 @ prism-audio）：输出第 `ctx.frame` 帧指定频段
/// 能量。输入 `Payload::Scalar(b)` = 频段序号（四舍五入取整；非标量/越界
/// → `Payload::Empty`）。
#[derive(Debug)]
pub struct AudioReactiveNode {
    meta: StructureMeta,
}

impl Default for AudioReactiveNode {
    fn default() -> Self {
        Self { meta: reactive_meta(13) }
    }
}

impl MathStructure for AudioReactiveNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload {
        let band = match x {
            Payload::Scalar(b) => b.round() as i64,
            _ => return Payload::Empty,
        };
        let Some(bake) = read_bake(ctx) else { return Payload::Empty };
        if band < 0 || band as usize >= bake.band_count() {
            return Payload::Empty;
        }
        Payload::Scalar(bake.band(ctx.frame, band as usize))
    }
}

/// AudioLevel 节点（S14 @ prism-audio）：输出第 `ctx.frame` 帧全频段平均
/// 电平（整体响度包络的代理；忽略输入负载）。
#[derive(Debug)]
pub struct AudioLevelNode {
    meta: StructureMeta,
}

impl Default for AudioLevelNode {
    fn default() -> Self {
        Self { meta: reactive_meta(14) }
    }
}

impl MathStructure for AudioLevelNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, _x: &Payload, ctx: &EvalCtx) -> Payload {
        let Some(bake) = read_bake(ctx) else { return Payload::Empty };
        let bands = bake.frame_bands(ctx.frame);
        if bands.is_empty() {
            return Payload::Empty;
        }
        // 固定序求和后一次除法（归约确定性）。
        let sum: f32 = bands.iter().sum();
        Payload::Scalar(sum / bands.len() as f32)
    }
}

prism_core::registry! {
    structure AudioReactiveNode => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
    structure AudioLevelNode => StructureId::custom(14, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::PcmBuffer;
    use crate::reactive::{bake_reactive, install_bake, ReactiveConfig};
    use prism_core::NodeRegistry;

    fn wired_ctx(frame: u64) -> EvalCtx {
        // 1s 440Hz 正弦 @ 8kHz，bass/mid/air 三段。
        let data: Vec<f32> = (0..8_000)
            .map(|i| (std::f64::consts::TAU * 440.0 * i as f64 / 8_000.0).sin() as f32)
            .collect();
        let buf = PcmBuffer::from_mono(8_000, data).unwrap();
        let cfg = ReactiveConfig { win: 512, hop: 250, band_edges: vec![0.0, 200.0, 2_000.0, 4_000.0] };
        let bake = bake_reactive(&buf, &cfg).unwrap();
        let mut ctx = EvalCtx::new(1, 1, frame);
        install_bake(&bake, &mut ctx);
        ctx
    }

    /// D-27 静态汇入：两节点进同一注册表，按名可解析、三问齐全。
    #[test]
    fn registry_collect_static_import() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 2);
        for name in ["AudioReactiveNode", "AudioLevelNode"] {
            let inst = reg.resolve_by_name(name).expect("按名解析（D-23 读档路径）");
            let meta = inst.meta();
            assert_eq!(meta.kernel, KernelKind::Convolution);
            assert_eq!(meta.duality, DualityPair::TimeFrequency);
            assert_eq!(meta.lattice, LatticePlan::LatticeAnchored);
            assert!(!meta.capabilities.invert && !meta.capabilities.sample);
        }
        assert!(reg.resolve_by_name("NoSuchNode").is_none());
    }

    /// reactive 节点：apply(Scalar(band)) == 直查烘焙表（逐位），帧号对齐。
    #[test]
    fn reactive_node_reads_baked_table_at_ctx_frame() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        let node = reg.resolve_by_name("AudioReactiveNode").unwrap();

        let buf_data: Vec<f32> = (0..8_000)
            .map(|i| (std::f64::consts::TAU * 440.0 * i as f64 / 8_000.0).sin() as f32)
            .collect();
        let bake = bake_reactive(
            &PcmBuffer::from_mono(8_000, buf_data).unwrap(),
            &ReactiveConfig { win: 512, hop: 250, band_edges: vec![0.0, 200.0, 2_000.0, 4_000.0] },
        )
        .unwrap();

        for frame in [0_u64, 5, 17, 31] {
            let ctx = wired_ctx(frame);
            for band in 0..3_u64 {
                let got = node.apply(&Payload::Scalar(band as f32), &ctx);
                let want = Payload::Scalar(bake.band(frame, band as usize));
                assert_eq!(got, want, "帧 {frame} 频段 {band} 应与直查逐位一致");
            }
        }
        // 越界频段与非标量负载 → Empty（单位元，不编造数值）。
        let ctx = wired_ctx(0);
        assert_eq!(node.apply(&Payload::Scalar(99.0), &ctx), Payload::Empty);
        assert_eq!(node.apply(&Payload::Empty, &ctx), Payload::Empty);
        // 帧越界钳到最近帧（确定性 seek 语义）。
        let tail = node.apply(&Payload::Scalar(0.0), &wired_ctx(u64::MAX));
        let last = Payload::Scalar(bake.band(bake.frames as u64 - 1, 0));
        assert_eq!(tail, last);
    }

    /// level 节点：全频段均值；缺烘焙 → Empty（数据未装配，不静默造数）。
    #[test]
    fn level_node_averages_bands_and_degrades_explicitly() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        let node = reg.resolve_by_name("AudioLevelNode").unwrap();

        let ctx = wired_ctx(10);
        let Payload::Scalar(level) = node.apply(&Payload::Empty, &ctx) else {
            panic!("有烘焙时应返回 Scalar");
        };
        let bake = bake_from_ctx(&ctx);
        let bands: Vec<f32> = (0..bake.band_count()).map(|b| bake.band(10, b)).collect();
        let want = bands.iter().sum::<f32>() / bands.len() as f32;
        assert_eq!(level.to_bits(), want.to_bits(), "level = 频段均值（固定序）");

        let bare = EvalCtx::new(1, 1, 0); // 未装烘焙
        assert_eq!(node.apply(&Payload::Empty, &bare), Payload::Empty);
        let node_r = reg.resolve_by_name("AudioReactiveNode").unwrap();
        assert_eq!(node_r.apply(&Payload::Scalar(0.0), &bare), Payload::Empty);
    }

    /// 从上下文读回烘焙表（测试辅助）。
    fn bake_from_ctx(ctx: &EvalCtx) -> ReactiveBake {
        ReactiveBake::from_bytes(ctx.snapshots.get(&(reactive_slot(), 0)).unwrap()).unwrap()
    }
}
