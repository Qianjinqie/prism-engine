//! AudioReactive 烘焙（B6； 第 4/5 条）。
//!
//! 音频 → FFT 频段能量 → 参数通道表：第 **N** 帧读样本窗 `[N·hop, N·hop+win)`
//! （补零过界），窗内容完全由 N 与缓冲决定——**帧级纯函数**，跳第 N 帧 ≡
//! 顺序推进（seek 等价， 音画同源 + 状态可重建）。
//!
//! 频段：`band_edges`（Hz，升序）把半谱切成 N−1 段（默认 bass/mid/air 三段
//! = 边缘 `[0, 200, 2000, Nyquist]`）；段值 = 段内功率和，按满幅正弦单 bin
//! 能量 `(win/4)²` 归一并钳到 [0, 1]（Hann 相干增益 0.5 → 单 bin 谱峰 win/4）。
//!
//! # EvalCtx 接线（状态在快照库，纯数据资产而非隐藏全局）
//!
//! [`install_bake`] 把烘焙表写入 `EvalCtx.snapshots` 的固定槽位
//! （[`reactive_slot`]，帧 0）；`nodes::AudioReactiveNode` / `AudioLevelNode`
//! 按帧读取——节点求值保持 `eval(ctx, t)` 无隐藏状态契约。烘焙表序列化
//! （[`ReactiveBake::to_bytes`]）格式版本化，供签名缓存与快照库承载。

use crate::buffer::PcmBuffer;
use crate::fft::{hann_window, Fft};
use crate::{AudioError, AudioResult};
use prism_core::node::{EvalCtx, Hash256};

/// 烘焙参数（win/hop 为格锚整数；hop = 每视频帧样本数，调用方按 `sr/fps`
/// 取整）。
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveConfig {
    /// 分析窗长（样本，2 的幂）。
    pub win: usize,
    /// 帧移（样本/帧）。
    pub hop: usize,
    /// 频段边缘（Hz，升序，≥2 个：首 0 末 Nyquist）。
    pub band_edges: Vec<f32>,
}

impl Default for ReactiveConfig {
    fn default() -> Self {
        Self { win: 1024, hop: 512, band_edges: vec![0.0, 200.0, 2000.0, 24_000.0] }
    }
}

/// AudioReactive 烘焙表：`table[frame·bands + b]` ∈ [0, 1]（行主序）。
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveBake {
    /// 源采样率。
    pub sample_rate: u32,
    /// 分析窗长。
    pub win: usize,
    /// 帧移。
    pub hop: usize,
    /// 频段边缘（Hz）。
    pub band_edges: Vec<f32>,
    /// 帧数（= ⌈样本数 / hop⌉）。
    pub frames: usize,
    /// 频段能量表（frames × bands，行主序）。
    table: Vec<f32>,
}

/// 烘焙 AudioReactive 通道（纯函数；同输入逐位一致）。
pub fn bake_reactive(buf: &PcmBuffer, cfg: &ReactiveConfig) -> AudioResult<ReactiveBake> {
    if cfg.win < 2 || !cfg.win.is_power_of_two() {
        return Err(AudioError::new(
            "AU1002",
            format!("reactive::bake_reactive(win={})", cfg.win),
            "窗长必须为 ≥2 的 2 的幂",
            "win 取 512/1024/2048",
        ));
    }
    if cfg.hop < 1 {
        return Err(AudioError::new(
            "AU1002",
            "reactive::bake_reactive(hop)",
            "帧移必须 ≥ 1 样本",
            "hop 取 sr/fps（如 44100/30 ≈ 1470）",
        ));
    }
    let nyquist = buf.sample_rate() as f32 / 2.0;
    if cfg.band_edges.len() < 2
        || cfg.band_edges.first() != Some(&0.0)
        || cfg.band_edges.last().map_or(true, |&e| e <= 0.0 || e > nyquist)
        || cfg.band_edges.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(AudioError::new(
            "AU1002",
            "reactive::bake_reactive(band_edges)",
            format!("频段边缘非法：{:?}（Nyquist = {nyquist} Hz）", cfg.band_edges),
            "边缘须升序、首元素 0、末元素 ≤ Nyquist（如 [0, 200, 2000, 22050]）",
        ));
    }

    let mono = buf.to_mono();
    let frames = mono.len().div_ceil(cfg.hop);
    let bands = cfg.band_edges.len() - 1;
    let fft = Fft::new(cfg.win)?;
    let hann = hann_window(cfg.win);
    // bin 边缘：ceil(ed·win/sr) 钳到 [1, win/2]（跳过 DC，覆盖全部正频 bin）。
    let bin_edges: Vec<usize> = cfg
        .band_edges
        .iter()
        .map(|&e| ((e / buf.sample_rate() as f32 * cfg.win as f32).ceil() as usize)
            .clamp(1, cfg.win / 2))
        .collect();
    let norm = (cfg.win as f32 / 4.0).powi(2);

    let mut table = vec![0.0_f32; frames * bands];
    for f in 0..frames {
        let start = f * cfg.hop;
        let windowed: Vec<f32> = (0..cfg.win)
            .map(|i| mono.get(start + i).copied().unwrap_or(0.0) * hann[i])
            .collect();
        let mag = fft.magnitudes(&windowed)?;
        for b in 0..bands {
            let (lo, hi) = (bin_edges[b], bin_edges[b + 1].max(bin_edges[b] + 1));
            let energy: f32 = mag[lo..hi.min(mag.len())].iter().map(|m| m * m).sum();
            table[f * bands + b] = (energy / norm).clamp(0.0, 1.0);
        }
    }

    Ok(ReactiveBake {
        sample_rate: buf.sample_rate(),
        win: cfg.win,
        hop: cfg.hop,
        band_edges: cfg.band_edges.clone(),
        frames,
        table,
    })
}

impl ReactiveBake {
    /// 频段数。
    pub fn band_count(&self) -> usize {
        self.band_edges.len() - 1
    }

    /// 第 `frame` 帧的频段行（帧号钳到 `[0, frames−1]`——越界取最近帧，
    /// 确定性；空表返回空切片）。
    pub fn frame_bands(&self, frame: u64) -> &[f32] {
        if self.frames == 0 {
            return &[];
        }
        let f = (frame.min(self.frames as u64 - 1)) as usize;
        &self.table[f * self.band_count()..(f + 1) * self.band_count()]
    }

    /// 第 `frame` 帧第 `band` 段能量；`band` 越界为 0（空表同）。
    pub fn band(&self, frame: u64, band: usize) -> f32 {
        self.frame_bands(frame).get(band).copied().unwrap_or(0.0)
    }

    /// 序列化为确定性字节（版本化格式，供快照库/签名缓存承载）。
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PRAB"); // Prism Reactive Audio Bake
        out.extend_from_slice(&1_u32.to_le_bytes()); // 格式版本
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.win as u32).to_le_bytes());
        out.extend_from_slice(&(self.hop as u32).to_le_bytes());
        out.extend_from_slice(&(self.frames as u32).to_le_bytes());
        out.extend_from_slice(&(self.band_edges.len() as u32).to_le_bytes());
        for &e in &self.band_edges {
            out.extend_from_slice(&e.to_bits().to_le_bytes());
        }
        for &v in &self.table {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }

    /// 反序列化（魔数/版本/维度校验失败返回 `None`——调用方走结构化报错，
    /// 绝不静默编造）。
    pub fn from_bytes(bytes: &[u8]) -> Option<ReactiveBake> {
        if bytes.len() < 28 || &bytes[0..4] != b"PRAB" {
            return None;
        }
        let u32le = |o: usize| {
            Some(u32::from_le_bytes(bytes.get(o..o + 4)?.try_into().ok()?))
        };
        if u32le(4)? != 1 {
            return None; // 未知版本：结构化拒绝，不猜格式
        }
        let sample_rate = u32le(8)?;
        let win = u32le(12)? as usize;
        let hop = u32le(16)? as usize;
        let frames = u32le(20)? as usize;
        let edge_count = u32le(24)? as usize;
        let mut pos = 28;
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let b = bytes.get(pos..pos + 4)?;
            edges.push(f32::from_bits(u32::from_le_bytes(b.try_into().ok()?)));
            pos += 4;
        }
        if edge_count < 2 || win < 2 || !win.is_power_of_two() || hop < 1 || sample_rate < 1 {
            return None;
        }
        let bands = edge_count - 1;
        let mut table = Vec::with_capacity((bytes.len() - pos) / 4);
        for c in bytes[pos..].chunks_exact(4) {
            let b = c.try_into().ok()?;
            table.push(f32::from_bits(u32::from_le_bytes(b)));
        }
        if table.len() != frames * bands {
            return None;
        }
        Some(ReactiveBake { sample_rate, win, hop, band_edges: edges, frames, table })
    }
}

/// 快照库槽位键（接线）：按 crate 标签经 PCG 派生的固定 256 位
/// 标识——一个求值上下文装载一份 reactive 烘焙（多烘焙路由属求值层轮次）。
pub fn reactive_slot() -> Hash256 {
    const TAG: u64 = 0x0A0D_10A0_00B6_0001; // "audio reactive" B6 标记
    let mut key = [0_u8; 32];
    for (i, slot) in key.chunks_exact_mut(8).enumerate() {
        slot.copy_from_slice(&prism_noise::pcg::hash2(TAG, i as u64, 32).to_be_bytes());
    }
    key
}

/// 把烘焙表装载进求值上下文（写入快照库固定槽位，帧 0；纯数据操作）。
pub fn install_bake(bake: &ReactiveBake, ctx: &mut EvalCtx) {
    ctx.snapshots.insert((reactive_slot(), 0), bake.to_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buffer() -> PcmBuffer {
        // 0.5s：样本 [1600, 3200) 为 1kHz 满幅正弦（落在 mid 段），其余静音
        // （帧长 800 → 帧 2/3 覆盖爆发段）。
        let (sr, n) = (8_000_u32, 4_000_usize);
        let mut v = vec![0.0_f32; n];
        for i in 1_600..3_200 {
            v[i] = (std::f64::consts::TAU * 1_000.0 * i as f64 / sr as f64).sin() as f32;
        }
        PcmBuffer::from_mono(sr, v).unwrap()
    }

    /// 测试频段边缘（8kHz SR：Nyquist 4k；bass <200、mid <2k、air ≤4k）。
    fn test_edges() -> Vec<f32> {
        vec![0.0, 200.0, 2_000.0, 4_000.0]
    }

    /// 窗语义：第 N 帧读 [N·hop, N·hop+win)——爆发段只提升覆盖它的帧。
    #[test]
    fn window_semantics_frame_n_reads_nth_window() {
        let buf = test_buffer();
        let cfg = ReactiveConfig { win: 512, hop: 800, band_edges: test_edges() };
        let bake = bake_reactive(&buf, &cfg).unwrap();
        assert_eq!(bake.frames, 5); // 4000/800 = 5 帧
        // 爆发在样本 [1600, 3200) = 帧 2/3 的窗覆盖。
        assert!(bake.band(2, 1) > 0.5, "帧 2 mid={}", bake.band(2, 1));
        assert!(bake.band(3, 1) > 0.5);
        assert!(bake.band(0, 1) < 0.01, "帧 0 静音 mid={}", bake.band(0, 1));
        assert!(bake.band(0, 0) < 0.01 && bake.band(4, 2) < 0.01);
        // 频段选择性：1kHz 只进 mid 段，bass/air 不响应。
        assert!(bake.band(2, 0) < 0.01 && bake.band(2, 2) < 0.01);
    }

    /// seek 等价（判据）：第 N 帧独立重算（把窗抽成独立缓冲、hop=win
    /// 单帧再烘焙）== 顺序烘焙的表行，逐位一致。
    #[test]
    fn frame_n_independent_recompute_equals_sequential() {
        let data: Vec<f32> = (0..9_000)
            .map(|i| {
                let t = i as f64 / 8_000.0;
                (0.6 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * 1_500.0 * t).sin()) as f32
            })
            .collect();
        let buf = PcmBuffer::from_mono(8_000, data).unwrap();
        let cfg = ReactiveConfig { win: 1_024, hop: 300, band_edges: vec![0.0, 500.0, 3_000.0, 4_000.0] };
        let sequential = bake_reactive(&buf, &cfg).unwrap();
        assert_eq!(sequential.frames, 30);

        let mono = buf.to_mono();
        for n in [0_usize, 1, 7, 15, 29] {
            // 独立重算：仅凭帧号 N 抽窗（不经顺序状态）。
            let start = n * cfg.hop;
            let mut window = vec![0.0_f32; cfg.win];
            for (i, w) in window.iter_mut().enumerate() {
                *w = mono.get(start + i).copied().unwrap_or(0.0);
            }
            let solo = bake_reactive(
                &PcmBuffer::from_mono(8_000, window).unwrap(),
                &ReactiveConfig { win: cfg.win, hop: cfg.win, band_edges: cfg.band_edges.clone() },
            )
            .unwrap();
            assert_eq!(solo.frames, 1);
            for b in 0..solo.band_count() {
                assert_eq!(
                    solo.band(0, b).to_bits(),
                    sequential.band(n as u64, b).to_bits(),
                    "帧 {n} 频段 {b}：独立重算 ≠ 顺序推进"
                );
            }
        }
    }

    /// 烘焙逐位确定；序列化往返逐字节；非法输入拒绝。
    #[test]
    fn determinism_bytes_roundtrip_and_validation() {
        let buf = test_buffer();
        let cfg = ReactiveConfig { win: 512, hop: 250, band_edges: test_edges() };
        let a = bake_reactive(&buf, &cfg).unwrap();
        let b = bake_reactive(&buf, &cfg).unwrap();
        assert_eq!(a.table, b.table, "同输入逐位一致");
        assert_eq!(a.to_bytes(), b.to_bytes());

        let bytes = a.to_bytes();
        let back = ReactiveBake::from_bytes(&bytes).expect("合法字节应可反序列化");
        assert_eq!(back, a, "序列化往返无损");

        assert!(ReactiveBake::from_bytes(b"XXXX rest").is_none(), "坏魔数");
        let mut bad_ver = bytes.clone();
        bad_ver[4] = 9;
        assert!(ReactiveBake::from_bytes(&bad_ver).is_none(), "未知版本");
        let truncated = &bytes[..bytes.len() - 8];
        assert!(ReactiveBake::from_bytes(truncated).is_none(), "截断");

        assert_eq!(
            bake_reactive(&buf, &ReactiveConfig { win: 1_000, ..Default::default() })
                .unwrap_err()
                .code,
            "AU1002"
        );
        assert_eq!(
            bake_reactive(&buf, &ReactiveConfig { hop: 0, ..Default::default() }).unwrap_err().code,
            "AU1002"
        );
        assert_eq!(
            bake_reactive(
                &buf,
                &ReactiveConfig { band_edges: vec![0.0, 3_000.0, 2_000.0, 4_000.0], ..Default::default() },
            )
            .unwrap_err()
            .code,
            "AU1002"
        );
    }

    /// 槽位派生稳定；install 后可从 EvalCtx 快照读回。
    #[test]
    fn eval_ctx_install_and_readback() {
        let buf = test_buffer();
        let bake = bake_reactive(&buf, &ReactiveConfig { win: 512, hop: 250, band_edges: test_edges() })
            .unwrap();
        assert_eq!(reactive_slot(), reactive_slot(), "槽位键派生确定");
        let mut ctx = EvalCtx::new(7, 1, 3);
        install_bake(&bake, &mut ctx);
        let raw = ctx.snapshots.get(&(reactive_slot(), 0)).expect("槽位应有数据");
        assert_eq!(ReactiveBake::from_bytes(raw).unwrap(), bake);
    }
}
