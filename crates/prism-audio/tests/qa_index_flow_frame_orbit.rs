//! 索引流 / 帧轨道一致性声明核对（音频侧，黑盒：仅经 `prism-audio` 公共
//! API，不触碰 crate 内部）。
//!
//! # 一致性声明
//!
//! 引擎内有三条同构的"帧轨道"，共享同一条 seek 等式（D-28：**跳第 N 帧 ≡
//! 顺序推进，逐位一致——状态 = 帧号/索引的纯函数，无隐藏累加器**）：
//!
//! 1. 物理 `SimSession` 帧轨道：顺序推进 + 快照续算（重放侧）；
//! 2. 粒子 `ParticleSystem` 索引轨道：`seek_with` 显式重放 + 快照续算；
//! 3. 音频两条轨道（本文件固化）：
//!    - **样本索引流** `sample_osc(wf, f, i, sr)`：相位 = fract(f·i/SR)，
//!      采样索引的纯函数——乱序随机访问 ≡ 顺序扫描（最强形式：无状态）；
//!    - **reactive 帧轨道** `ReactiveBake::frame_bands(frame)`：第 N 帧读
//!      样本窗 [N·hop, N·hop+win)（补零过界），窗内容完全由帧号与缓冲
//!      决定——帧号纯函数，乱序读帧 ≡ 顺序读帧。
//!
//! 核对结论：三条轨道在各自状态层级（无状态 / 快照 / 重放）满足同一条
//! seek 等式，声明与代码一致，无矛盾。
//!
//! # 覆盖检查项
//!
//! 1. 样本索引流：逆序 / 伪随机置换 / 跨段跳进跳出 == 顺序扫描（逐位）；
//! 2. reactive 帧轨道：乱序读帧 == 顺序读帧（逐位），`band` 与
//!    `frame_bands` 行视图一致，越界帧钳到最近帧（确定性）；
//! 3. 帧轨道快照往返：`to_bytes`/`from_bytes` 后乱序读帧仍逐位一致
//!    （快照承载的轨道与原地轨道同构——与粒子快照续算对拍同一范式）。

use prism_audio::buffer::PcmBuffer;
use prism_audio::reactive::{bake_reactive, ReactiveBake, ReactiveConfig};
use prism_audio::synth::{sample_osc, Waveform};

/// 确定性 LCG（零依赖；置换与跳步长不用系统随机源）。
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// 样本索引流乱序访问一致性：`sample_osc` 是索引的纯函数（相位 =
/// fract(f·i/SR)，无累加器），逆序、伪随机置换、跨段跳进跳出三条乱序
/// 路径都应逐位复现顺序扫描。
#[test]
fn sample_index_stream_unordered_access_matches_sequential() {
    let (f, sr) = (440.0_f32, 44_100_u32);
    let n = 20_000_usize;
    let mut rng = Lcg(0x0D1B_54A1_5EED_0001);
    for wf in [Waveform::Sine, Waveform::Square, Waveform::Saw, Waveform::Triangle] {
        let seq: Vec<f32> = (0..n).map(|i| sample_osc(wf, f, i, sr)).collect();

        // 路径 1：逆序扫描。
        for i in (0..n).rev() {
            assert_eq!(sample_osc(wf, f, i, sr).to_bits(), seq[i].to_bits(), "{wf:?} 逆序 @ {i}");
        }

        // 路径 2：伪随机置换（Fisher–Yates，LCG 驱动）。
        let mut order: Vec<usize> = (0..n).collect();
        for k in (1..n).rev() {
            let j = rng.below(k as u64 + 1) as usize;
            order.swap(k, j);
        }
        for &i in &order {
            assert_eq!(sample_osc(wf, f, i, sr).to_bits(), seq[i].to_bits(), "{wf:?} 置换 @ {i}");
        }

        // 路径 3：跨段跳进跳出——先探远处（应命中顺序流该位），顺序流本身
        // 不受跳探影响（若有隐藏状态则会污染后续样本）。
        let mut i = 0;
        while i < n {
            assert_eq!(sample_osc(wf, f, i, sr).to_bits(), seq[i].to_bits());
            let jump = 97 + rng.below(211) as usize; // 97..=307
            if let Some(&ahead) = seq.get(i + jump) {
                assert_eq!(
                    sample_osc(wf, f, i + jump, sr).to_bits(),
                    ahead.to_bits(),
                    "{wf:?} 跳探 @ {i}+{jump}"
                );
            }
            i += 1;
        }
    }
}

/// 测试缓冲：220Hz + 1500Hz 叠加（8kHz SR，9000 样本 = 30 帧 @ hop=300），
/// 两个频段都有非零响应（乱序读帧有实质内容可比）。
fn test_buffer() -> PcmBuffer {
    let data: Vec<f32> = (0..9_000)
        .map(|i| {
            let t = i as f64 / 8_000.0;
            (0.6 * (std::f64::consts::TAU * 220.0 * t).sin()
                + 0.4 * (std::f64::consts::TAU * 1_500.0 * t).sin()) as f32
        })
        .collect();
    PcmBuffer::from_mono(8_000, data).unwrap()
}

/// 测试配置：win=1024 / hop=300 → 30 帧，三段频段（bass/mid/air）。
fn test_config() -> ReactiveConfig {
    ReactiveConfig { win: 1_024, hop: 300, band_edges: vec![0.0, 500.0, 3_000.0, 4_000.0] }
}

/// reactive 帧轨道乱序访问一致性：`frame_bands` 是帧号的纯函数（第 N 帧
/// 读 [N·hop, N·hop+win)），乱序读帧 == 顺序读帧（逐位）；越界帧钳到
/// 最近帧（`frame ≥ frames` 取 `frames−1`，确定性 seek 语义）。
#[test]
fn reactive_frame_orbit_unordered_access_matches_sequential() {
    let bake = bake_reactive(&test_buffer(), &test_config()).unwrap();
    let (frames, bands) = (bake.frames, bake.band_count());
    assert_eq!(frames, 30);
    assert_eq!(bands, 3);
    // 轨道非平凡：频段确有响应（否则逐位一致是空断言）。
    assert!(bake.band(0, 1) > 0.01, "mid 段应有响应：{}", bake.band(0, 1));

    // 顺序轨道（行视图收集）。
    let sequential: Vec<Vec<f32>> =
        (0..frames as u64).map(|f| bake.frame_bands(f).to_vec()).collect();

    // 乱序：伪随机置换读帧，逐位一致。
    let mut rng = Lcg(0x5EED_F00D_0000_0002);
    let mut order: Vec<u64> = (0..frames as u64).collect();
    for k in (1..order.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        order.swap(k, j);
    }
    for &f in &order {
        let row = bake.frame_bands(f);
        assert_eq!(row.len(), bands);
        for (b, &v) in row.iter().enumerate() {
            assert_eq!(v.to_bits(), sequential[f as usize][b].to_bits(), "帧 {f} 段 {b}");
            // band(f, b) 与行视图同一存储，逐位一致。
            assert_eq!(bake.band(f, b).to_bits(), v.to_bits());
        }
    }

    // 越界钳制：frame ≥ frames 一律取最近帧（含 u64::MAX 极端值）。
    let last = bake.frame_bands(frames as u64 - 1);
    for &f in &[frames as u64, frames as u64 + 1, frames as u64 + 1_000, u64::MAX] {
        assert_eq!(bake.frame_bands(f), last, "越界帧 {f} 应钳到最近帧");
        for b in 0..bands {
            assert_eq!(bake.band(f, b), last[b]);
        }
    }
    // band 越界段为 0（空表同理），不与钳制语义混淆。
    assert_eq!(bake.band(0, bands), 0.0);
    assert_eq!(bake.band(u64::MAX, bands), 0.0);
}

/// 帧轨道快照往返：`to_bytes` → `from_bytes` 得到的轨道与原地轨道在乱序
/// 读帧下逐位一致——快照承载的帧轨道与原地轨道同构（与粒子系统
/// "快照续算 == 顺序推进"对拍同一范式）。
#[test]
fn reactive_frame_orbit_snapshot_roundtrip_matches_live_orbit() {
    let live = bake_reactive(&test_buffer(), &test_config()).unwrap();
    let bytes = live.to_bytes();
    let restored = ReactiveBake::from_bytes(&bytes).expect("合法字节应可反序列化");
    assert_eq!(restored.frames, live.frames);
    assert_eq!(restored.band_count(), live.band_count());

    let mut rng = Lcg(0xA110_5EED_0000_0003);
    let mut order: Vec<u64> = (0..live.frames as u64).collect();
    for k in (1..order.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        order.swap(k, j);
    }
    for &f in &order {
        let (a, b) = (live.frame_bands(f), restored.frame_bands(f));
        assert_eq!(a.len(), b.len());
        for (idx, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "帧 {f} 段 {idx}：快照轨道 ≠ 原地轨道");
        }
    }
}
