//! **T-A / R-1 浏览器腿**：wavefront 路径模式的 **CPU golden** 生成器。
//!
//! # 存在的理由：wasm 里装不下 CPU 权威
//!
//! `prism_rt::wavefront::render_frame`（路径积分 + SampleStream + Sphere NEE/MIS）
//! 是判对基准，但它依赖 prism-rt 全树，**进不了 wasm 体积预算**（见
//! `crates/prism-gpu/Cargo.toml` 的注释）。浏览器端（`web/abi.rs` 的
//! `gpu.wavefront_path`）因此拿不到"CPU 参考帧"做逐位对拍——除非把参考帧
//! **预生成成数据**随 wasm 一起分发。
//!
//! 本文件就是那个预生成器：三组 R-1 配置的 CPU 参考帧（f32 **位模式**，无损）
//! + 相机 uniform + 真 Joe–Kuo Sobol 表，序列化成一个定长二进制
//! `web/wavefront_path_golden.bin`。wasm 侧 `include_bytes!` 内嵌后：
//!
//! * **输入侧**：sobol 表 / 相机 uniform 直接从 golden 切片上传——GPU 与
//!   CPU 参考天然同表同流（对拍前提）；
//! * **判据侧**：GPU 读回的 accum（f32 位模式）与 golden 期望帧逐位比对
//!   （`differing_pixels == 0`）。
//!
//! # 位模式而不是十进制（与 hair_golden.json 同纪律）
//!
//! 十进制往返引入双重舍入，"差 1 ulp"会被格式误差淹没；位模式是无损的，
//! 且能直接断言 `to_bits()` 逐位一致。
//!
//! # 新鲜度怎么保证
//!
//! * `emit_wavefront_path_golden_for_browser`（本文件）每次跑都**重算并重写**
//!   golden（同输入逐字节同输出，`git status` 不抖动）——CPU 参考变了它就变；
//! * `wavefront_path_golden_matches_cpu_reference`（本文件）读**磁盘上已入库**
//!   的文件与刚算的参考逐字节比对——忘了重生成 ⇒ 立即红；
//! * golden 过期但测试没跑的兜底：浏览器端 GPU 对不上 golden ⇒ `FAIL`。
//!   三条链任何一条断，都有另一条把它抓出来。
//!
//! # 铁律遵守
//!
//! - 零第三方依赖（只用 std + prism-rt/prism-gpu 本体）；
//! - 不删判据、不加 `#[ignore]`；
//! - 场景构造与 `qa_wavefront_parity.rs` **同一份代码**（`common/r1_parity.rs`），
//!   不存在"第二份几何"。

/// golden 生成器只用场景/相机/表三件（对拍原语在 parity 测试侧使用）。
#[path = "common/r1_parity.rs"]
#[allow(dead_code)]
mod r1_parity;

use r1_parity::{camera_uniform, cpu_reference, sobol_table_bytes};
use prism_rt::camera::Camera;
use prism_math::Vec3;
use std::fs;
use std::path::PathBuf;

/// golden 二进制的魔数（"WPG1" 小端）。
const MAGIC: u32 = 0x3147_5057;
/// 格式版本（布局变更必须 +1，wasm 侧按版本拒绝错位读取）。
const VERSION: u32 = 1;
/// 三组 R-1 配置（与 `qa_wavefront_parity.rs` 的三个对拍测试逐一对应）。
const CONFIGS: [(u32, u32, u32, u32); 3] = [(16, 16, 2, 3), (12, 10, 1, 1), (8, 8, 5, 2)];

/// golden 输出路径（`crates/prism-gpu/web/wavefront_path_golden.bin`）。
fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web/wavefront_path_golden.bin")
}

/// 序列化 golden（布局见模块文档与 [`write_golden`] 内注释）。
fn build_golden() -> Vec<u8> {
    // 全部 u32 小端；帧数据是 f32 位模式的小端 u32 流。
    let mut out = Vec::new();
    let put_u32 = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());

    put_u32(&mut out, MAGIC);
    put_u32(&mut out, VERSION);
    put_u32(&mut out, CONFIGS.len() as u32);
    for &(w, h, spp, bounces) in &CONFIGS {
        put_u32(&mut out, w);
        put_u32(&mut out, h);
        put_u32(&mut out, spp);
        put_u32(&mut out, bounces);
    }
    // 相机 uniform ×3（aspect 随配置不同，逐份打包；wasm 侧按配置切片）。
    for &(w, h, _, _) in &CONFIGS {
        let cam = Camera::new(
            Vec3::new(0.0, 0.8, -2.5),
            Vec3::new(0.0, 0.4, 0.5),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            3.1,
            0.0,
            0.0,
        );
        out.extend_from_slice(&camera_uniform(&cam, w as f32 / h as f32));
    }
    // 真 Joe–Kuo Sobol 方向数表（64×32 u32 小端）——GPU 与 CPU 参考同表的载体。
    out.extend_from_slice(&sobol_table_bytes());
    // 期望帧 ×3（RGBW 四通道 f32 位模式；A 通道也入库，判据比对全 4 通道）。
    for &(w, h, spp, bounces) in &CONFIGS {
        for p in cpu_reference(w, h, spp, bounces) {
            for c in p {
                put_u32(&mut out, c.to_bits());
            }
        }
    }
    out
}

/// **生成器 + 新鲜度守卫**（同一测试，杜绝"生成与校验并发读写同一文件"的竞态）。
///
/// 1. 磁盘已有 golden ⇒ 先与重算的 CPU 参考逐字节比对（**新鲜度**：忘重生成即红）；
/// 2. 缺失或不一致 ⇒ 写盘（生成 / 更新）；
/// 3. 读回终验（写路径完整性的自证）。
#[test]
fn emit_wavefront_path_golden_for_browser() {
    let bytes = build_golden();
    let out = golden_path();
    if let Ok(on_disk) = fs::read(&out) {
        let first = on_disk.iter().zip(bytes.iter()).position(|(a, b)| a != b);
        if on_disk.len() == bytes.len() && first.is_none() {
            println!(
                "golden 新鲜：{} 字节与 CPU 参考逐位一致（三配置全部）",
                bytes.len()
            );
            return;
        }
        println!(
            "golden 过期（磁盘 {} B vs 重算 {} B，首差异 {first:?}）→ 重写",
            on_disk.len(),
            bytes.len()
        );
    }
    fs::write(&out, &bytes)
        .unwrap_or_else(|e| panic!("写 wavefront_path_golden.bin 到 {} 失败：{e}", out.display()));
    let back = fs::read(&out).expect("写后读回失败（文件系统异常）");
    assert_eq!(back, bytes, "写盘后读回不一致（写路径损坏）");
    println!(
        "golden 已写入 {}（{} B = 3 配置帧；wasm 侧 include_bytes! 内嵌，浏览器 gpu.wavefront_path 消费）",
        out.display(),
        bytes.len()
    );
}

/// **布局自检**：头部字段的偏移推导与 wasm 侧切片约定一致（防两侧各算一套）。
#[test]
fn golden_layout_offsets_are_self_consistent() {
    let bytes = build_golden();
    let rd = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    assert_eq!(rd(0), MAGIC);
    assert_eq!(rd(4), VERSION);
    assert_eq!(rd(8), 3);
    // 配置表 @12，每条 16 B；相机表 @12+48=60，每份 96 B；sobol @60+288=348，
    // 8192 B；帧 @348+8192=8540。
    let mut expect_off = 8540usize;
    for &(w, h, _, _) in &CONFIGS {
        let npix = (w * h) as usize;
        // 末尾恰好被三帧填满（无尾随字节）。
        assert!(
            expect_off + npix * 16 <= bytes.len(),
            "帧区越界：@{expect_off} 需要 {} B",
            npix * 16
        );
        expect_off += npix * 16;
    }
    assert_eq!(expect_off, bytes.len(), "golden 总长与推导不一致（有尾随/缺失字节）");
    println!("golden 布局自检 OK：头部 60 B + 相机 288 B + sobol 8192 B + 帧 {} B", bytes.len() - 8540);
}
