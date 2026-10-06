//! 硬件编码器适配层（C7；路线图 C7 行 / §3.5 三层策略 ②）。
//!
//! **设计 + 骨架，真机启用**（Phase 4）：VideoToolbox（macOS）/ MediaFoundation
//! （Windows）/ NVENC（Linux + N 卡）/ VAAPI（Linux 通用）。本模块只定义**厂商无关
//! 的统一接口与能力声明**；实际编码调用厂商 API（超出零依赖 crate 范围，由
//! `ffi` 绑定层在真机启用）。
//!
//! # 边界（红线）
//!
//! **不自研 H.264/HEVC 编码器**（D-10）。硬件编码器是"原始厂商 API"边界内，
//! 本层只做适配骨架与降级编排：真机不可用时，编码回退到 [`crate::extprobe`] 的
//! 外部 ffmpeg 路径，再回退到内置 MJPEG。

use crate::extprobe::Codec;

/// 硬件编码器后端。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    /// macOS VideoToolbox。
    VideoToolbox,
    /// Windows MediaFoundation。
    MediaFoundation,
    /// NVIDIA NVENC。
    Nvenc,
    /// Linux VA-API。
    Vaapi,
}

impl HwBackend {
    /// 人类可读名。
    pub fn name(self) -> &'static str {
        match self {
            HwBackend::VideoToolbox => "VideoToolbox",
            HwBackend::MediaFoundation => "MediaFoundation",
            HwBackend::Nvenc => "NVENC",
            HwBackend::Vaapi => "VAAPI",
        }
    }

    /// 该后端在此 OS 上是否**可能**可用（编译期平台判定；真机能力需运行时探测）。
    pub fn platform_possible(self) -> bool {
        match self {
            HwBackend::VideoToolbox => cfg!(target_os = "macos"),
            HwBackend::MediaFoundation => cfg!(target_os = "windows"),
            HwBackend::Nvenc | HwBackend::Vaapi => cfg!(target_os = "linux"),
        }
    }

    /// 当前平台的可能后端清单（确定性序）。
    pub fn for_platform() -> Vec<HwBackend> {
        [HwBackend::VideoToolbox, HwBackend::MediaFoundation, HwBackend::Nvenc, HwBackend::Vaapi]
            .into_iter()
            .filter(|b| b.platform_possible())
            .collect()
    }
}

/// 硬件编码器能力（真机探测结果）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HwCapability {
    /// 后端。
    pub backend: HwBackend,
    /// 支持的编码族。
    pub codecs: &'static [Codec],
    /// 最大宽。
    pub max_width: u32,
    /// 最大高。
    pub max_height: u32,
}

/// 硬件编码器探测接口（真机由厂商 API 实现；沙箱返回空集）。
pub trait HwEncoders {
    /// 枚举当前机器可用的硬件编码器能力（确定性序）。
    fn enumerate(&self) -> Vec<HwCapability>;
}

/// 骨架探测：**无真机 FFI**，仅返回平台"可能"后端的能力声明（未验证）。
///
/// 真机启用时替换为厂商 API 查询；沙箱/无 FFI 环境返回空集，编码自动降级。
#[derive(Debug, Default, Clone, Copy)]
pub struct SkeletonProbe;

impl HwEncoders for SkeletonProbe {
    fn enumerate(&self) -> Vec<HwCapability> {
        // 无 FFI 绑定 → 保守返回空集（真机启用由 Phase 4 的绑定层实现）。
        Vec::new()
    }
}

/// 统一编码器请求（厂商无关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeRequest {
    /// 目标编码族。
    pub codec: Codec,
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 帧率。
    pub fps: u32,
    /// 目标码率（kbps；0 = 恒定质量）。
    pub bitrate_kbps: u32,
}

/// 编码器适配器接口（厂商无关）；真机实现由各后端提供。
///
/// 当前仅定义契约（骨架），沙箱无实现——`available()` 恒 `false`。
pub trait EncoderAdapter {
    /// 本适配器是否在当前机器可用（真机探测）。
    fn available(&self) -> bool;

    /// 底层后端。
    fn backend(&self) -> HwBackend;

    /// 是否支持该请求（编码族 + 尺寸）。
    fn supports(&self, req: &EncodeRequest) -> bool;

    /// 编码一帧（返回压缩字节；骨架/不可用时返回 `None`）。
    fn encode_frame(&mut self, rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>>;

    /// 冲刷（返回尾部压缩字节 + 统计）。
    fn flush(&mut self) -> Vec<u8>;
}

/// 从能力列表挑选满足请求的后端（确定性序：按枚举序取首个）。
pub fn select_encoder(caps: &[HwCapability], req: &EncodeRequest) -> Option<HwBackend> {
    caps.iter()
        .find(|c| {
            c.codecs.contains(&req.codec)
                && req.width <= c.max_width
                && req.height <= c.max_height
        })
        .map(|c| c.backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_backends_are_os_consistent() {
        let backends = HwBackend::for_platform();
        for b in &backends {
            assert!(b.platform_possible());
        }
        if cfg!(target_os = "linux") {
            assert!(backends.contains(&HwBackend::Nvenc));
            assert!(!backends.contains(&HwBackend::VideoToolbox));
        }
    }

    #[test]
    fn skeleton_probe_empty_then_degrades() {
        let caps = SkeletonProbe.enumerate();
        assert!(caps.is_empty(), "无 FFI 环境应返回空能力集");
        let req = EncodeRequest {
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            fps: 30,
            bitrate_kbps: 8000,
        };
        assert_eq!(select_encoder(&caps, &req), None, "无能力时应降级");
    }

    #[test]
    fn selection_respects_codec_and_size() {
        let caps = [HwCapability {
            backend: HwBackend::Nvenc,
            codecs: &[Codec::H264, Codec::Hevc],
            max_width: 3840,
            max_height: 2160,
        }];
        let ok = EncodeRequest { codec: Codec::H264, width: 1920, height: 1080, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &ok), Some(HwBackend::Nvenc));
        let too_big = EncodeRequest { codec: Codec::H264, width: 8000, height: 4320, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &too_big), None);
        let mjpeg = EncodeRequest { codec: Codec::Mjpeg, width: 640, height: 480, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &mjpeg), None, "硬件后端不支持 MJPEG");
    }

    #[test]
    fn backend_names() {
        assert_eq!(HwBackend::Nvenc.name(), "NVENC");
        assert_eq!(HwBackend::Vaapi.name(), "VAAPI");
    }
}
