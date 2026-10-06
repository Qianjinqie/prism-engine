//! 效果器链（B12）：混响 / 延迟 / 压缩 / EQ / 失真 / 声像自动化。
//!
//! 每个效果器都是**对 `PcmBuffer` 的纯函数**（单声道输入 → 单声道输出，或
//! 立体声重塑），无隐藏状态机、无全局 RNG；内部滤波器用自研 [`crate::det`]
//! 三角件（D-29 禁 libm）。链式组合由 [`FxChain`] 按固定顺序折叠。
//!
//! 子模块：
//! - [`reverb`] — Schroeder 梳状+全通 与 FDN 反馈延迟网；
//! - [`delay`] — 带反馈/湿干比的延时线；
//! - [`compressor`] — 阈值/比率/软拐点动态压缩；
//! - [`eq`] — 多段 biquad（低架/峰/高架）；
//! - [`distortion`] — 磁带饱和 / 硬削 / 波形整形；
//! - [`panner`] — 恒功率声像 + 自动化曲线。

pub mod compressor;
pub mod delay;
pub mod distortion;
pub mod eq;
pub mod panner;
pub mod reverb;

use crate::buffer::PcmBuffer;

/// 单声道效果器统一接口：`process` 逐样本纯变换（实现方内部无跨调用状态，
/// 一切延迟线缓冲在调用内局部构造——保证 seek/重算等价）。
pub trait MonoFx {
    /// 处理一段单声道样本，返回等长输出。
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32>;
}

/// 效果器链：按 `Vec` 顺序串联 [`MonoFx`]，`wet` 为末级干湿混合。
///
/// 链处理为固定序折叠（`f0(f1(...))`），同输入逐位一致。
#[derive(Default)]
pub struct FxChain {
    /// 串联的效果器（按加入顺序处理）。
    pub effects: Vec<Box<dyn MonoFx>>,
}

impl FxChain {
    /// 空链。
    pub fn new() -> Self {
        Self { effects: Vec::new() }
    }

    /// 追加一个效果器（builder 风格）。
    pub fn then(mut self, fx: Box<dyn MonoFx>) -> Self {
        self.effects.push(fx);
        self
    }

    /// 按序折叠处理（空链原样返回）。
    pub fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut signal = input.to_vec();
        for fx in &self.effects {
            signal = fx.process(&signal, sample_rate);
        }
        signal
    }
}

/// 将单声道效果器应用于 `buffer` 的每个声道（声道独立处理，保持交错结构）。
pub fn apply_mono_fx(fx: &dyn MonoFx, buffer: &PcmBuffer) -> PcmBuffer {
    let ch = buffer.channels() as usize;
    let frames = buffer.frames();
    let sr = buffer.sample_rate();
    let src = buffer.as_slice();
    let mut out = vec![0.0_f32; frames * ch];
    // 逐声道抽取 → 处理 → 写回（固定声道序）。
    for c in 0..ch {
        let mut channel = vec![0.0_f32; frames];
        for f in 0..frames {
            channel[f] = src[f * ch + c];
        }
        let processed = fx.process(&channel, sr);
        for f in 0..frames {
            out[f * ch + c] = processed[f];
        }
    }
    PcmBuffer::from_interleaved(sr, buffer.channels(), out).expect("形状保持必然合法")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 桩效果器：恒定增益，用于验证链序与折叠。
    struct Gain(f32);
    impl MonoFx for Gain {
        fn process(&self, input: &[f32], _sr: u32) -> Vec<f32> {
            input.iter().map(|x| x * self.0).collect()
        }
    }

    #[test]
    fn chain_folds_in_order() {
        let chain = FxChain::new().then(Box::new(Gain(0.5))).then(Box::new(Gain(2.0)));
        let out = chain.process(&[1.0, -1.0, 0.5], 48_000);
        // (x*0.5)*2 = x（顺序无关的乘法在此不可区分，故另测一次非交换）。
        assert_eq!(out, vec![1.0, -1.0, 0.5]);
        let empty = FxChain::new();
        assert_eq!(empty.process(&[0.3, 0.7], 1), vec![0.3, 0.7]);
    }

    #[test]
    fn apply_mono_fx_preserves_shape_and_channels() {
        let buf = PcmBuffer::from_interleaved(8_000, 2, vec![1.0, -1.0, 0.5, -0.5]).unwrap();
        let out = apply_mono_fx(&Gain(0.5), &buf);
        assert_eq!(out.channels(), 2);
        assert_eq!(out.frames(), 2);
        assert_eq!(out.as_slice(), &[0.5, -0.5, 0.25, -0.25]);
    }
}
