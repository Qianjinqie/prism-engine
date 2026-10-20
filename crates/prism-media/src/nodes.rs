//! embed 预置子图节点（→ prism-core 注册表；D-27 静态汇入制）。
//!
//! 素材本已存在，引擎只负责"读入 → 进图 → 被一切能力作用"。本文件实现
//! **图像嵌入节点**：以 [`PixelImage`] 为素材、按 `EvalCtx.frame` 索引帧（静态图
//! 单帧），输出该像素的 RGBA（`Payload::Vec3` 场景线性 RGB + alpha 经标量）。
//!
//! 素材数据经 [`EvalCtx.snapshots`] 的**纯数据槽位**装入（与 prism-audio 的
//! reactive 烘焙同模型）：`apply` 按当前帧/像素查表，求值纯函数契约不破
//! （无隐藏状态、seek 等价）；素材缺失返回 [`Payload::Empty`]（单位元，不静默
//! 编造数值）。
//!
//! 嵌入的图片**过 bloom/调色的能力不受本层限制**：节点只输出改到 channel 的
//! 场景线性值，后续 post 链（S2 卷积 bloom / 调色）照常作用——这正是"被引擎
//! 能力影响"的接入点（后续第 7 条）。
//!
//! # 编号口径
//!
//! 用户段 `StructureId::custom(13, "prism-media")`：官方段 S1–S12 全局单例，
//! 嵌入属 S5 重参数化（素材坐标→像素采样）与 S12 实体表示（外部资产入图）
//! 交汇，官方位预留 alias.rs，本节点以自定义段挂接，跨 crate 编号天然隔离。
//! 三问：核 = 无核服务（查表采样），对偶 = 边界↔体（帧序列↔单帧画布），
//! 格锚 = 帧号/像素整数格（帧级查表零浮点漂移）。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};
use prism_core::node::Hash256;
use prism_math::Vec3;

use crate::source::PixelImage;

/// 嵌入素材槽位（快照键的第一分量）；`EvalCtx.snapshots` 里存序列化后的帧。
fn embed_slot() -> Hash256 {
    // 固定域隔离常量：与 prism-audio 的 reactive 槽位区分（避免误取）。
    let mut h = [0u8; 32];
    h[0] = 0xE1;
    h[1] = 0x3B;
    h[2] = 0x11;
    h[3] = 0x01;
    h
}

/// 把嵌入素材（一帧或多帧 RGBA8）序列化装入快照库字节流。
///
/// 编码：`u32 帧数` + 每帧 `u32 w` + `u32 h` + `w*h*4` 字节 RGBA。定长前缀 +
/// 行主序像素——同素材必得同字节（确定性载体）。
pub fn encode_embed(frames: &[PixelImage]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + frames.len() * 8 + frames.iter().map(|f| f.pixels.len()).sum::<usize>());
    out.extend_from_slice(&(frames.len() as u32).to_be_bytes());
    for f in frames {
        out.extend_from_slice(&f.width.to_be_bytes());
        out.extend_from_slice(&f.height.to_be_bytes());
        out.extend_from_slice(&f.pixels);
    }
    out
}

/// 解码嵌入素材字节流（[`encode_embed`] 的逆）。
pub fn decode_embed(bytes: &[u8]) -> Option<Vec<PixelImage>> {
    if bytes.len() < 4 {
        return None;
    }
    let n = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut pos = 4usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        if pos + 8 > bytes.len() {
            return None;
        }
        let w = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        let h = u32::from_be_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]);
        pos += 8;
        let len = (w as usize) * (h as usize) * 4;
        if pos + len > bytes.len() {
            return None;
        }
        frames.push(PixelImage { width: w, height: h, pixels: bytes[pos..pos + len].to_vec() });
        pos += len;
    }
    Some(frames)
}

/// 把嵌入素材装入上下文快照库（帧 0 槽位；调用方在求值前装配一次）。
pub fn install_embed(ctx: &mut EvalCtx, frames: &[PixelImage]) {
    ctx.snapshots.insert((embed_slot(), 0), encode_embed(frames));
}

/// 图像嵌入节点（S13 @ prism-media）。
///
/// 输入 `Payload::Vec3(uv)`：`x,y ∈ [0,1]` 为归一化坐标（越界钳制）；
/// 输出 `Payload::Vec3(rgb)`（场景线性域，alpha 置于第三分量之外的约定由调用方
/// 决定——本层把 RGBA 的 RGB 作为场景线性值，8 位整数除以 255）。
/// 素材缺失 / 帧越界 → `Payload::Empty`。
#[derive(Debug)]
pub struct ImageEmbedNode {
    meta: StructureMeta,
}

impl Default for ImageEmbedNode {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(13, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::BoundaryVolume,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for ImageEmbedNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload {
        let (u, v): (f32, f32) = match x {
            Payload::Vec3(v3) => (v3.x, v3.y),
            _ => return Payload::Empty,
        };
        let frames = match ctx.snapshots.get(&(embed_slot(), 0)).and_then(|b| decode_embed(b)) {
            Some(f) => f,
            None => return Payload::Empty, // 素材缺失：单位元，不编造
        };
        // 帧索引：静态图恒 0，序列/动画按 EvalCtx.frame 循环取模（确定性）。
        let frame = &frames[(ctx.frame as usize) % frames.len()];
        if frame.width == 0 || frame.height == 0 {
            return Payload::Empty;
        }
        let px: u32 = ((u.clamp(0.0, 1.0) * frame.width as f32) as u32).min(frame.width - 1);
        let py: u32 = ((v.clamp(0.0, 1.0) * frame.height as f32) as u32).min(frame.height - 1);
        let [r, g, b, a] = frame.get(px, py);
        if a == 0 {
            return Payload::Empty; // 全透明像素：无贡献
        }
        // 8 位整数 → 场景线性（除以 255，IEEE 精确不引入额外舍入）。
        Payload::Vec3(Vec3 {
            x: r as f32 / 255.0,
            y: g as f32 / 255.0,
            z: b as f32 / 255.0,
        })
    }
}

/// 音频源嵌入节点（S14 @ prism-media）：按 `EvalCtx.frame` 输出该帧指定的
/// 音频源采样标量。输入 `Payload::Scalar(帧内偏移)`（整数帧偏移）；素材由
/// [`install_audio`] 装入。素材缺失 → `Payload::Empty`。
#[derive(Debug)]
pub struct AudioEmbedNode {
    meta: StructureMeta,
}

impl Default for AudioEmbedNode {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(14, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::SampleMeasure,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

/// 音频嵌入槽位（与图像槽位区分）。
fn audio_slot() -> Hash256 {
    let mut h = [0u8; 32];
    h[0] = 0xE1;
    h[1] = 0x3B;
    h[2] = 0x11;
    h[3] = 0x02;
    h
}

/// 装入音频源（单声道下混样本 + 采样率）。
pub fn install_audio(ctx: &mut EvalCtx, sample_rate: u32, mono: &[f32]) {
    let mut out = Vec::with_capacity(8 + mono.len() * 4);
    out.extend_from_slice(&sample_rate.to_be_bytes());
    out.extend_from_slice(&(mono.len() as u32).to_be_bytes());
    for &s in mono {
        out.extend_from_slice(&s.to_bits().to_be_bytes());
    }
    ctx.snapshots.insert((audio_slot(), 0), out);
}

impl MathStructure for AudioEmbedNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload {
        let offset = match x {
            Payload::Scalar(o) => o.round() as i64,
            _ => 0,
        };
        let bytes = match ctx.snapshots.get(&(audio_slot(), 0)) {
            Some(b) => b,
            None => return Payload::Empty,
        };
        if bytes.len() < 8 {
            return Payload::Empty;
        }
        let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let idx = ctx.frame as i64 + offset;
        if idx < 0 || idx as usize >= n {
            return Payload::Empty;
        }
        let p = 8 + (idx as usize) * 4;
        if p + 4 > bytes.len() {
            return Payload::Empty;
        }
        let bits = u32::from_be_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
        Payload::Scalar(f32::from_bits(bits))
    }
}

prism_core::registry! {
    structure ImageEmbedNode => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
    structure AudioEmbedNode => StructureId::custom(14, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> PixelImage {
        let mut img = PixelImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, rgba);
            }
        }
        img
    }

    #[test]
    fn embed_encode_decode_roundtrip() {
        let frames = vec![solid(2, 2, [10, 20, 30, 255]), solid(1, 1, [40, 50, 60, 255])];
        let bytes = encode_embed(&frames);
        let back = decode_embed(&bytes).expect("往返必成功");
        assert_eq!(back, frames);
    }

    #[test]
    fn image_embed_node_returns_scene_linear() {
        let img = solid(2, 2, [255, 128, 0, 255]);
        let mut ctx = EvalCtx::new(0, 1, 0);
        install_embed(&mut ctx, &[img]);

        let node = ImageEmbedNode::default();
        let out = node.apply(&Payload::Vec3(Vec3 { x: 0.5, y: 0.5, z: 0.0 }), &ctx);
        match out {
            Payload::Vec3(v) => {
                assert!((v.x - 1.0).abs() < 1e-6);
                assert!((v.y - 128.0 / 255.0).abs() < 1e-6);
                assert_eq!(v.z, 0.0);
            }
            other => panic!("期望 Vec3，得 {other:?}"),
        }
    }

    #[test]
    fn image_embed_missing_material_is_empty() {
        let ctx = EvalCtx::new(0, 1, 0);
        let node = ImageEmbedNode::default();
        assert_eq!(node.apply(&Payload::Vec3(Vec3 { x: 0.0, y: 0.0, z: 0.0 }), &ctx), Payload::Empty);
    }

    #[test]
    fn embedded_pixels_can_be_affected_by_downstream_gain() {
        // 证明嵌入值处于场景线性域、可被后续能力（如曝光/gain）线性作用：
        // 2× 增益后 ≥ 0.5 的像素必溢出到 >1（bloom 阈值可达）。
        let img = solid(1, 1, [200, 200, 200, 255]);
        let mut ctx = EvalCtx::new(0, 1, 0);
        install_embed(&mut ctx, &[img]);
        let node = ImageEmbedNode::default();
        let base = match node.apply(&Payload::Vec3(Vec3 { x: 0.0, y: 0.0, z: 0.0 }), &ctx) {
            Payload::Vec3(v) => v.x,
            _ => panic!(),
        };
        assert!(base * 2.0 > 1.0, "高亮像素 2× 增益应超 1.0（bloom 可拾取）");
    }

    #[test]
    fn registry_static_import_registers_embed_nodes() {
        use prism_core::NodeRegistry;
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        let cn = env!("CARGO_CRATE_NAME");
        assert!(reg.contains(&StructureId::custom(13, cn)));
        assert!(reg.contains(&StructureId::custom(14, cn)));
        let inst = reg.resolve_by_name("ImageEmbedNode").expect("按名解析");
        assert_eq!(inst.meta().kernel, KernelKind::NoKernel);
    }

    #[test]
    fn audio_embed_reads_frame_and_offset() {
        let mut ctx = EvalCtx::new(0, 1, 2);
        install_audio(&mut ctx, 8_000, &[0.1, 0.2, 0.3, 0.4]);
        let node = AudioEmbedNode::default();
        assert_eq!(node.apply(&Payload::Scalar(0.0), &ctx), Payload::Scalar(0.3));
        assert_eq!(node.apply(&Payload::Scalar(1.0), &ctx), Payload::Scalar(0.4));
        assert_eq!(node.apply(&Payload::Scalar(9.0), &ctx), Payload::Empty, "越界返回单位元");
    }
}
