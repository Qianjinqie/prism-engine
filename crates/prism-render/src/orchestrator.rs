//! Tile 渲染编排器（A5）。
//!
//! 语义权威：《架构总设计》v2.4 §4.1（Orchestrator：eval(t, tile) → tile 产物
//! → 固定序合帧 → 交付）、§4.3（tile 无关性三规则）；《强大版路线图》§2.1/§2.6
//! （tile 级天然并行安全区；确定性优先——本批**逐 tile 顺序求值**，并行化属
//! 10/2 压测窗口；《算法白皮书》§26：顺序求值 = 并行拓扑的退化情形，二者为
//! 同一棵执行树的两种遍历，结果逐位一致）。
//!
//! # 确定性关键
//! tile 内像素一律用**全局像素坐标**喂 [`SampleIndex`]（D-28 像素域）——tile
//! 划分不得影响采样流（验收关卡 4）：同像素在整幅/tile 两模式下收到完全相同
//! 的 SampleIndex，源与效果的纯函数性保证拼接结果与整幅逐字节一致。
//!
//! # 归约纪律（白皮书 §26）
//! 编排全程无跨 tile 浮点归约：tile 产物经位模式拷贝（`blit_rect`）合帧，
//! 像素之间零浮点合并——tile 级是并行安全区。

use prism_core::SampleIndex;
use prism_io::fbuf::Fbuf16F;

use crate::tile::{Tile, TileEffect, TileGrid};

/// 逐像素源：全局像素坐标 + [`SampleIndex`] → RGBA（f32 场景线性域）。
///
/// 纯函数契约：同 `(x, y, si)` 必同输出；一切随机经 si 哈希派生（架构 §4.4，
/// 不得私开 RNG）。坐标是**全局像素域**——tile 化只改变求值的分批，不改变
/// 坐标与采样流。
pub trait PixelSource {
    /// 求值一个像素。`(x, y)` 为全局像素坐标；`si.pixel` 同为全局坐标（D-28）。
    fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4];
}

/// tile 化渲染编排器（架构 §4.1 Orchestrator 的 A5 骨架）。
pub struct TileOrchestrator {
    /// tile 网格划分。
    pub grid: TileGrid,
    /// 当前帧号（SampleIndex 格锚，格 Zᵏ）。
    pub frame: u64,
    /// 源节点 id（SampleIndex 格锚）。
    pub node_id: u64,
    /// 全局种子（样本流哈希的种子位；与 SampleIndex 一同构成派生键，
    /// 架构 §4.4：hash(全局种子, 节点id, 帧, pixel, 样本索引[, 维度])）。
    pub global_seed: u64,
}

impl TileOrchestrator {
    /// 以网格构造；帧号/节点/种子默认 0（可按需覆写）。
    pub fn new(grid: TileGrid) -> Self {
        TileOrchestrator { grid, frame: 0, node_id: 0, global_seed: 0 }
    }

    /// 构造一个像素的 SampleIndex——**全局像素坐标**（确定性关键）。
    fn sample_index(&self, x: u32, y: u32) -> SampleIndex {
        SampleIndex {
            frame: self.frame,
            pixel: (x, y),
            sample: 0,
            dimension: 0,
            node_id: self.node_id,
        }
    }

    /// 整幅直接求值（验收的"整幅"参考侧）：源整帧求值 → 效果链整帧求值。
    pub fn render_frame_full(
        &self,
        source: &dyn PixelSource,
        effects: &[&dyn TileEffect],
    ) -> Fbuf16F {
        let whole = Tile {
            x: 0,
            y: 0,
            w: self.grid.width,
            h: self.grid.height,
        };
        let mut buf = self.render_source_window(source, whole);
        for effect in effects {
            buf = effect.eval_window(&buf);
        }
        buf
    }

    /// tile 化求值：划分 → 逐 tile halo 扩采 → 源求值 → 效果链求值 → 裁核心
    /// → 固定序（行主序）拼接回写。
    ///
    /// 效果链的总 halo = 各效果申报半径之和：第 k 级效果的输出核心区距其
    /// 输入窗口边缘 ≥ 后续效果半径之和，逐级归纳保证核心区全部像素的邻域
    /// 读取与整幅求值一致（tile.rs 模块文档的协议正确性条件）。
    pub fn render_frame_tiled(
        &self,
        source: &dyn PixelSource,
        effects: &[&dyn TileEffect],
    ) -> Fbuf16F {
        let total_halo: u32 = effects.iter().map(|e| e.halo_radius()).sum();
        let mut out = Fbuf16F::new(self.grid.width, self.grid.height);
        for tile in self.grid.tiles() {
            let win: Tile = tile.expanded(total_halo, self.grid.width, self.grid.height);
            let mut buf = self.render_source_window(source, win);
            for effect in effects {
                buf = effect.eval_window(&buf);
            }
            // 裁核心区回写：窗口坐标系下核心区偏移 = tile 原点 − 窗口原点。
            out.blit_rect(
                tile.x,
                tile.y,
                &buf,
                tile.x - win.x,
                tile.y - win.y,
                tile.w,
                tile.h,
            );
        }
        out
    }

    /// 对一个窗口求值源：窗口内逐像素以**全局坐标**喂 SampleIndex。
    fn render_source_window(&self, source: &dyn PixelSource, win: Tile) -> Fbuf16F {
        let mut buf = Fbuf16F::new(win.w, win.h);
        for ly in 0..win.h {
            for lx in 0..win.w {
                let gx: u32 = win.x + lx;
                let gy: u32 = win.y + ly;
                buf.set(lx, ly, source.eval_pixel(gx, gy, self.sample_index(gx, gy)));
            }
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// 测试：验收关卡 1 / 2 / 4 + 确定性 + MD5 指纹助手
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::BoxBlur3;

    // ---- MD5（RFC 1321）测试助手：关卡判定要求的逐字节一致指纹 ----
    // 整数运算实现（零依赖、确定性）；正确性以 RFC 官方向量锚定（见下）。
    mod md5 {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
            20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
            16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        // T[i] = floor(2³² × |sin(i+1)|)（RFC 1321 表格常量，硬编码——禁 libm）。
        const K: [u32; 64] = [
            0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee, 0xf57c_0faf, 0x4787_c62a,
            0xa830_4613, 0xfd46_9501, 0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
            0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821, 0xf61e_2562, 0xc040_b340,
            0x265e_5a51, 0xe9b6_c7aa, 0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
            0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed, 0xa9e3_e905, 0xfcef_a3f8,
            0x676f_02d9, 0x8d2a_4c8a, 0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
            0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70, 0x289b_7ec6, 0xeaa1_27fa,
            0xd4ef_3085, 0x0488_1d05, 0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
            0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039, 0x655b_59c3, 0x8f0c_cc92,
            0xffef_f47d, 0x8584_5dd1, 0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
            0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
        ];

        fn round_f(i: usize, x: u32, y: u32, z: u32) -> u32 {
            match i / 16 {
                0 => (x & y) | (!x & z),
                1 => (x & z) | (y & !z),
                2 => x ^ y ^ z,
                _ => y ^ (x | !z),
            }
        }

        /// 摘要的十六进制串。
        pub fn hex(data: &[u8]) -> String {
            let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
            msg.extend_from_slice(data);
            msg.push(0x80);
            while msg.len() % 64 != 56 {
                msg.push(0);
            }
            msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());

            let (mut a0, mut b0, mut c0, mut d0) =
                (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
            for chunk in msg.chunks_exact(64) {
                let mut m = [0u32; 16];
                for (i, word) in m.iter_mut().enumerate() {
                    *word = u32::from_le_bytes([
                        chunk[4 * i],
                        chunk[4 * i + 1],
                        chunk[4 * i + 2],
                        chunk[4 * i + 3],
                    ]);
                }
                let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
                for i in 0..64 {
                    let g: usize = match i / 16 {
                        0 => i,
                        1 => (5 * i + 1) % 16,
                        2 => (3 * i + 5) % 16,
                        _ => (7 * i) % 16,
                    };
                    // RFC 1321：tmp = F(B,C,D) + A + K[i] + M[g]；
                    // B_new = B + (tmp <<< s[i])（b 不进预旋转和）。
                    let tmp: u32 = round_f(i, b, c, d)
                        .wrapping_add(a)
                        .wrapping_add(K[i])
                        .wrapping_add(m[g]);
                    a = d;
                    d = c;
                    c = b;
                    b = b.wrapping_add(tmp.rotate_left(S[i]));
                }
                a0 = a0.wrapping_add(a);
                b0 = b0.wrapping_add(b);
                c0 = c0.wrapping_add(c);
                d0 = d0.wrapping_add(d);
            }
            [a0, b0, c0, d0]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }
    }

    #[test]
    fn md5_rfc1321_official_vectors() {
        assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5::hex(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }

    // ---- 测试源：确定性逐像素函数（si 派生 + 全局坐标混频）----

    /// splitmix64 终结器（纯整数，确定性）。
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 确定性测试源：si 哈希派生 + 全局坐标平滑项混频——产生 f16 量化与
    /// blur 均有意义的非均匀场。
    struct HashSource {
        seed: u64,
    }

    impl PixelSource for HashSource {
        fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4] {
            let mut z: u64 = self.seed
                ^ si.frame.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (((si.pixel.0 as u64) << 32) | si.pixel.1 as u64)
                    .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ si.node_id.rotate_left(11)
                ^ ((si.sample as u64) | ((si.dimension as u64) << 32));
            let h: u64 = splitmix64(&mut z);
            let a: f32 = (h >> 40) as f32 / 16_777_216.0; // [0, 1)
            let b: f32 = ((h >> 16) & 0xFF_FFFF) as f32 / 16_777_216.0;
            let cx: f32 = x as f32 * 0.03125; // 坐标平滑项（blur 的输入成分）
            let cy: f32 = y as f32 * 0.03125;
            [
                (cx + a).fract(),
                (cy + b).fract(),
                ((cx + cy) * 0.5 + a * b).fract(),
                1.0 - a * 0.5,
            ]
        }
    }

    /// 缓冲指纹：RGBA16F 位模式序列化（小端字节）→ MD5。
    fn buffer_md5(fb: &Fbuf16F) -> String {
        let bytes: Vec<u8> = fb.pixels.iter().flat_map(|h| h.to_le_bytes()).collect();
        md5::hex(&bytes)
    }

    /// 逐像素统计差异（按 RGBA16F 位模式四元组比较）。
    fn count_differing_pixels(a: &Fbuf16F, b: &Fbuf16F) -> usize {
        assert_eq!((a.width, a.height), (b.width, b.height));
        a.pixels
            .chunks_exact(4)
            .zip(b.pixels.chunks_exact(4))
            .filter(|(p, q)| p != q)
            .count()
    }

    // ---- 验收关卡 1：16×9 tile = 整幅逐字节一致 ----

    #[test]
    fn checkpoint1_tiled_16x9_equals_full_frame_byte_identical() {
        // 192×108 ÷ 16 列 × 9 行 = 144 块 12×12 tile；halo 型效果（box blur，
        // 半径 1）下拼接必须与整幅逐字节一致。
        let grid = TileGrid::counts(192, 108, 16, 9);
        let orch = TileOrchestrator::new(grid);
        let effects: [&dyn TileEffect; 1] = [&BoxBlur3];

        let full = orch.render_frame_full(&HashSource { seed: orch.global_seed }, &effects);
        let tiled = orch.render_frame_tiled(&HashSource { seed: orch.global_seed }, &effects);

        let md5_full: String = buffer_md5(&full);
        let md5_tiled: String = buffer_md5(&tiled);
        println!("checkpoint1 full  md5 = {md5_full}");
        println!("checkpoint1 tiled md5 = {md5_tiled}");
        assert_eq!(md5_full, md5_tiled, "16×9 tile 拼接必须与整幅逐字节一致");
    }

    #[test]
    fn checkpoint1_indivisible_frame_and_multi_tile_sizes_also_identical() {
        // 加固：非整除帧（边缘 tile 收窄 + 帧边界钳制）与异形 tile 均成立。
        let source = HashSource { seed: 42 };
        for grid in [
            TileGrid::new(100, 70, 16, 9),  // 非整除：7×8 网格，末行末列收窄
            TileGrid::new(100, 70, 37, 23), // 素数尺寸 tile
            TileGrid::new(100, 70, 200, 90), // 单 tile 覆盖全帧
        ] {
            let orch = TileOrchestrator::new(grid);
            let effects: [&dyn TileEffect; 1] = [&BoxBlur3];
            let full = orch.render_frame_full(&source, &effects);
            let tiled = orch.render_frame_tiled(&source, &effects);
            assert_eq!(
                buffer_md5(&full),
                buffer_md5(&tiled),
                "网格 {}×{}（tile {}×{}）拼接必须逐字节一致",
                grid.cols(),
                grid.rows(),
                grid.tile_w,
                grid.tile_h
            );
        }
    }

    #[test]
    fn tiled_render_is_deterministic_across_runs() {
        // 同参数两遍 tile 化求值 → md5 一致（求值纯函数契约）。
        let grid = TileGrid::counts(96, 54, 16, 9);
        let orch = TileOrchestrator::new(grid);
        let effects: [&dyn TileEffect; 1] = [&BoxBlur3];
        let a = orch.render_frame_tiled(&HashSource { seed: 7 }, &effects);
        let b = orch.render_frame_tiled(&HashSource { seed: 7 }, &effects);
        assert_eq!(buffer_md5(&a), buffer_md5(&b));
    }

    // ---- 验收关卡 2：halo 不足可检出（协议反向验证）----

    #[test]
    fn checkpoint2_insufficient_halo_is_detected_as_boundary_difference() {
        // 协议违规模拟：效果申报 halo = 0，实际需要 1（误申报）——
        // 编排器按申报扩采 0 → tile 边缘发生窗口钳制读 → 与整幅产生边界差异。
        // （未申报的邻域需求不会被静默满足——差异可检出即协议生效。）
        struct MisdeclaredBlur;
        impl TileEffect for MisdeclaredBlur {
            fn halo_radius(&self) -> u32 {
                0 // 申报不足
            }
            fn eval_window(&self, w: &Fbuf16F) -> Fbuf16F {
                BoxBlur3.eval_window(w)
            }
        }

        let grid = TileGrid::counts(192, 108, 16, 9);
        let orch = TileOrchestrator::new(grid);
        let source = HashSource { seed: 1 };
        let full = orch.render_frame_full(&source, &[&BoxBlur3]);
        let tiled = orch.render_frame_tiled(&source, &[&MisdeclaredBlur]);

        let diffs: usize = count_differing_pixels(&full, &tiled);
        println!("checkpoint2 halo 不足差异像素数 = {diffs} / {}", grid.width * grid.height);
        assert!(diffs > 0, "halo 不足必须能检出边界差异（协议反向验证）");
        assert!(
            diffs < (grid.width as usize) * (grid.height as usize),
            "差异应仅出现在 tile 边界带（宽 1 像素），tile 中心区不受影响"
        );
    }

    // ---- 验收关卡 4：tile 划分不影响采样流 ----

    /// 记录型源：登记每次求值请求收到的 SampleIndex 全字段。
    #[derive(Default)]
    struct RecordingSource {
        log: std::cell::RefCell<Vec<(u32, u32, SampleIndex)>>,
    }

    impl PixelSource for RecordingSource {
        fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4] {
            self.log.borrow_mut().push((x, y, si));
            // 输出本身由 si 派生（同 si 同值），供逐字节比对联动验证。
            let mut z: u64 = (x as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (y as u64).rotate_left(17)
                ^ si.frame.rotate_left(3)
                ^ si.node_id.rotate_left(29);
            let h: u64 = splitmix64(&mut z);
            [((h >> 32) & 0xFF) as f32 / 255.0; 4]
        }
    }

    fn si_map_of(log: &[(u32, u32, SampleIndex)]) -> std::collections::HashMap<(u32, u32), SampleIndex> {
        log.iter().map(|(x, y, si)| ((*x, *y), *si)).collect()
    }

    #[test]
    fn checkpoint4_tile_partition_does_not_disturb_sample_stream() {
        let grid = TileGrid::counts(192, 108, 16, 9);
        let orch = TileOrchestrator::new(grid);

        // 整幅模式：全帧每像素恰好求值一次。
        let full_src = RecordingSource::default();
        let _ = orch.render_frame_full(&full_src, &[]);
        let full_log: Vec<(u32, u32, SampleIndex)> = full_src.log.borrow().clone();
        assert_eq!(full_log.len(), (192 * 108) as usize, "整幅模式每像素恰一次");
        let full_map: std::collections::HashMap<(u32, u32), SampleIndex> = si_map_of(&full_log);

        // tile 模式（无效果 → halo 0 → 每像素亦恰求值一次）：
        // 每像素收到的 SampleIndex 与整幅模式完全相同（全局像素域）。
        let tiled_src = RecordingSource::default();
        let _ = orch.render_frame_tiled(&tiled_src, &[]);
        let tiled_log: Vec<(u32, u32, SampleIndex)> = tiled_src.log.borrow().clone();
        assert_eq!(tiled_log.len(), (192 * 108) as usize, "无 halo 时 tile 模式每像素恰一次");
        for (x, y, si) in &tiled_log {
            assert_eq!(
                full_map.get(&(*x, *y)),
                Some(si),
                "像素 ({x},{y}) 的 SampleIndex 在两模式下必须一致"
            );
            assert_eq!(si.pixel, (*x, *y), "si.pixel 必须等于全局像素坐标（D-28）");
        }

        // 带 halo 效果时：窗口重叠使帧边缘像素被多次求值，但每次收到的
        // SampleIndex 仍与整幅模式一致（纯函数 → 同 si 同值，值不漂移）。
        let halo_src = RecordingSource::default();
        let _ = orch.render_frame_tiled(&halo_src, &[&BoxBlur3]);
        for (x, y, si) in halo_src.log.borrow().iter() {
            assert_eq!(
                full_map.get(&(*x, *y)),
                Some(si),
                "halo 扩采像素 ({x},{y}) 的 SampleIndex 必须与整幅模式一致"
            );
        }
    }
}
