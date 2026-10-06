//! RGBA16F 半精度帧缓冲（A5）。
//!
//! 语义权威：《强大版路线图》§4 A5 行（RGBA16F 缓冲）与 §2.2（内部帧缓冲统一
//! RGBA16F——f16 存储、f32 计算，内存省一半、精度足够，8bit 量化只发生在
//! 交付边界）；《架构总设计》v2.4 §4.3 格锚（RGBA16F 半浮点的确定性纪律——
//! 半浮点存储、运算位模式精确往返）。
//!
//! # 确定性纪律
//! - **存储域是格**：每通道为 16 位 IEEE 754 binary16 位模式（`u16`）——
//!   整数存储，同值同位模式，跨平台逐位一致（白皮书 §25.3）；
//! - **计算域是 f32**：读出（f16→f32 精确扩展，无舍入）→ 运算 → 写入
//!   （f32→f16 round-to-nearest-even）。对一切 f16 可表示值，往返位模式
//!   精确——无漂移累积通道；
//! - **手写 IEEE 754 half 转换**：纯位运算（移位/掩码/加法），零第三方、
//!   零 libm、零 unsafe（D-29 同款纪律：整数运算跨平台逐位一致）。

/// f32 → f16（IEEE 754 binary16，round-to-nearest-even）。
///
/// 返回 16 位半浮点位模式。溢出归无穷、下溢按 RNE 归半正规或零、
/// NaN 保留负载（负载仅存低 13 位可见部分，强制非零防 NaN→Inf）。
pub fn f32_to_f16(value: f32) -> u16 {
    let bits: u32 = value.to_bits();
    let sign: u16 = ((bits >> 16) & 0x8000) as u16;
    let exp32: i32 = ((bits >> 23) & 0xFF) as i32;
    let man32: u32 = bits & 0x007F_FFFF;

    // Inf / NaN：f16 指数全 1。NaN 保负载（右移 13 位后为 0 则强制置 1）。
    if exp32 == 0xFF {
        if man32 == 0 {
            return sign | 0x7C00;
        }
        let m: u16 = (man32 >> 13) as u16;
        return sign | 0x7C00 | if m == 0 { 0x0001 } else { m };
    }

    // f16 存储指数 = f32 无偏指数 + 15（f16 偏置）。
    let exp16: i32 = exp32 - 127 + 15;

    if exp16 >= 0x1F {
        // 溢出 → 无穷（RNE 下 |value| ≥ 65520 均归无穷）。
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            // |value| < 2⁻²⁵ → 舍入到零（ties-to-even 的零侧）。
            return sign;
        }
        // 半正规：带隐含位右移 `14 - exp16` 位，对移出位做 RNE 舍入。
        // 进位到 0x400 时自然落回正规表示（指数域 = 1、尾数 = 0），无需特判。
        let shift: u32 = (14 - exp16) as u32;
        let full: u32 = man32 | 0x0080_0000;
        let mut man16: u32 = full >> shift;
        let guard: u32 = (full >> (shift - 1)) & 1;
        let sticky: bool = (full & ((1u32 << (shift - 1)) - 1)) != 0;
        if guard == 1 && (sticky || (man16 & 1) == 1) {
            man16 += 1;
        }
        return sign | (man16 as u16);
    }

    // 正规：10 位尾数 RNE 舍入；尾数进位满（0x400 = 2¹⁰，10 位全 1 再 +1）
    // 时清尾数并令指数 +1。
    let mut exp16: i32 = exp16;
    let mut man16: u32 = man32 >> 13;
    let guard: u32 = (man32 >> 12) & 1;
    let sticky: bool = (man32 & 0x0000_0FFF) != 0;
    if guard == 1 && (sticky || (man16 & 1) == 1) {
        man16 += 1;
        if man16 == 0x0400 {
            man16 = 0;
            exp16 += 1;
            if exp16 >= 0x1F {
                return sign | 0x7C00; // 尾数进位溢出 → 无穷
            }
        }
    }
    sign | ((exp16 as u16) << 10) | (man16 as u16)
}

/// f16 → f32（精确扩展：一切 f16 值在 f32 中可精确表示，无舍入）。
pub fn f16_to_f32(half: u16) -> f32 {
    let sign: u32 = ((half & 0x8000) as u32) << 16;
    let exp: u32 = ((half >> 10) & 0x1F) as u32;
    let man: u32 = (half & 0x03FF) as u32;

    let bits: u32 = match exp {
        0 => {
            if man == 0 {
                sign // ±0
            } else {
                // 半正规精确归一：值 = man × 2⁻²⁴；最高有效位 p = 31 - lz，
                // 规格化 E = p - 24，f32 指数域 = E + 127；隐含位落在位 23，
                // 属 f32 指数域/隐含位，掩去后余下即尾数。
                let lz: u32 = man.leading_zeros(); // 22..=31
                let field: u32 = 134 - lz; // 103..=112
                sign | (field << 23) | ((man << (lz - 8)) & 0x007F_FFFF)
            }
        }
        0x1F => sign | 0x7F80_0000 | (man << 13), // Inf / NaN（负载原样左移）
        _ => sign | ((exp + 112) << 23) | (man << 13), // 正规：偏置 15 → 127
    };
    f32::from_bits(bits)
}

/// RGBA16F 帧缓冲：行主序，每像素 4 × `u16` 半浮点位模式。
///
/// `pixels.len() == width * height * 4`；存储即位模式（格上的量），
/// 拼接/拷贝全部按 `u16` 整数进行——逐字节一致的载体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fbuf16F {
    /// 帧宽（像素）。
    pub width: u32,
    /// 帧高（像素）。
    pub height: u32,
    /// 像素缓冲（RGBA 顺序，行主序，每通道 16 位半浮点位模式）。
    pub pixels: Vec<u16>,
}

impl Fbuf16F {
    /// 分配全零缓冲。
    pub fn new(width: u32, height: u32) -> Self {
        Fbuf16F {
            width,
            height,
            pixels: vec![0u16; (width as usize) * (height as usize) * 4],
        }
    }

    /// 写一个像素（f32 场景线性域 → f16 存储，写入即量化）。
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, rgba: [f32; 4]) {
        debug_assert!(x < self.width && y < self.height, "set 越界 ({x},{y})");
        let idx: usize = ((y as usize) * self.width as usize + x as usize) * 4;
        self.pixels[idx] = f32_to_f16(rgba[0]);
        self.pixels[idx + 1] = f32_to_f16(rgba[1]);
        self.pixels[idx + 2] = f32_to_f16(rgba[2]);
        self.pixels[idx + 3] = f32_to_f16(rgba[3]);
    }

    /// 读一个像素（f16 存储 → f32 计算域，精确扩展无舍入）。
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        debug_assert!(x < self.width && y < self.height, "get 越界 ({x},{y})");
        let idx: usize = ((y as usize) * self.width as usize + x as usize) * 4;
        [
            f16_to_f32(self.pixels[idx]),
            f16_to_f32(self.pixels[idx + 1]),
            f16_to_f32(self.pixels[idx + 2]),
            f16_to_f32(self.pixels[idx + 3]),
        ]
    }

    /// 把源缓冲的矩形区域（位模式原样）拷贝到本缓冲的 (dst_x, dst_y)。
    ///
    /// 拼接回写原语：纯 `u16` 整数拷贝，无任何浮点转换——逐字节一致。
    pub fn blit_rect(
        &mut self,
        dst_x: u32,
        dst_y: u32,
        src: &Fbuf16F,
        sx: u32,
        sy: u32,
        sw: u32,
        sh: u32,
    ) {
        debug_assert!(
            (sx as u64 + sw as u64) <= src.width as u64
                && (sy as u64 + sh as u64) <= src.height as u64,
            "blit_rect 源矩形越界"
        );
        debug_assert!(
            (dst_x as u64 + sw as u64) <= self.width as u64
                && (dst_y as u64 + sh as u64) <= self.height as u64,
            "blit_rect 目标矩形越界"
        );
        let count: usize = (sw as usize) * 4;
        for row in 0..sh {
            let src_row: usize = ((sy + row) as usize * src.width as usize + sx as usize) * 4;
            let dst_row: usize =
                ((dst_y + row) as usize * self.width as usize + dst_x as usize) * 4;
            self.pixels[dst_row..dst_row + count]
                .copy_from_slice(&src.pixels[src_row..src_row + count]);
        }
    }

    /// 交付边界：RGBA16F → RGBA8 [`prism_gpu::Framebuffer`]（§2.2：8bit 量化
    /// 只发生在这里）。量化约定与 `prism_gpu` 的 `channel_to_u8` 一致。
    pub fn to_framebuffer(&self) -> prism_gpu::Framebuffer {
        let mut fb = prism_gpu::Framebuffer::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let [r, g, b, _a] = self.get(x, y);
                let idx: usize = ((y as usize) * self.width as usize + x as usize) * 4;
                fb.pixels[idx] = channel_to_u8(r);
                fb.pixels[idx + 1] = channel_to_u8(g);
                fb.pixels[idx + 2] = channel_to_u8(b);
                fb.pixels[idx + 3] = 255;
            }
        }
        fb
    }
}

/// 交付边界量化（与 `prism_gpu::channel_to_u8` 同式：clamp01 × 255 四舍五入）。
#[inline]
fn channel_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---------------------------------------------------------------------------
// 测试：位模式精确往返（验收关卡 3）+ 已知向量 + 缓冲操作
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64 终结器（纯整数，测试用确定性随机源——无 RNG 状态污染）。
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn f16_known_bit_patterns() {
        // 已知向量：值 ↔ f16 位模式（IEEE 754 binary16 权威值）。
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(-2.5), 0xC100);
        assert_eq!(f32_to_f16(0.1), 0x2E66); // 0.1f16 = 0.0999755859375（1.6×2⁻⁴）
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000, "-0 的符号位必须保留");
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(f32_to_f16(f32::NAN), 0x7E00, "标准 quiet NaN 负载应保留");
        assert_eq!(
            f32_to_f16(f32::from_bits(0x7F80_0001)),
            0x7C01,
            "微负载 NaN 不得变 Inf"
        );
        // 极小 / 极大（f16 域边界）。
        assert_eq!(f32_to_f16(6.103_515_6e-5), 0x0400, "最小正规 2⁻¹⁴");
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001, "最小半正规 2⁻²⁴");
        assert_eq!(f32_to_f16(65504.0), 0x7BFF, "最大有限值");
        assert_eq!(
            f32_to_f16(65520.0),
            0x7C00,
            "65520 = 溢出中点，ties-to-even 归 Inf"
        );
        assert_eq!(f32_to_f16(65519.0), 0x7BFF, "中点之下仍归最大有限值");
        assert_eq!(f32_to_f16(2.5e-8), 0x0000, "低于 2⁻²⁵ 归零");
        assert_eq!(
            f32_to_f16(2.980_232_2e-8),
            0x0000,
            "恰为 2⁻²⁵（零与最小半正规的中点）：ties-to-even 归零"
        );
        assert_eq!(f32_to_f16(3.0e-8), 0x0001, "高于 2⁻²⁵ 归最小半正规");
    }

    #[test]
    fn f16_rne_mantissa_carry_bumps_exponent() {
        // 回归（W4-B4 发现）：正规路径尾数 RNE 进位（0x3FF → 0x400）必须
        // 清尾数并令指数 +1；原实现阈值误写 0x0800，exp16 为奇数时进位
        // 被位模式吞掉（如 0.4998… 错档为 0.25）。逐指数档验证：
        for e in 1u16..15 {
            let base = e << 10; // 指数档 e、尾数 0 的位模式
            let v = f16_to_f32(base | 0x03FF); // 该档尾数全 1 的精确 f32
                                               // 动点：精确可表示值往返不变。
            assert_eq!(f32_to_f16(v), base | 0x03FF, "档 {e}：精确值不得漂移");
            // v 上方恰半个 f16 ULP（guard=1、sticky=0、尾数奇）→ RNE 进位
            // 到下一档尾数 0。f16 ULP 在档 e = 2^(e−25)，半 ULP = 2^(e−26)，
            // 位构造纯 2 的幂（禁 libm）。
            let half_ulp = f32::from_bits(((e as i32 - 26 + 127) as u32) << 23);
            let up = v + half_ulp; // f32 尾数余量 13 位 ⇒ 加法精确
            assert_eq!(
                f32_to_f16(up),
                base + 0x0400,
                "档 {e}：RNE 进位必须升档（{v} + 半ULP）"
            );
        }
    }

    #[test]
    fn f16_roundtrip_special_values_bit_exact() {
        // 关卡 3（特判值）：0 / ±0 / 1 / 0.5 / 极小 / 极大 / ±Inf——
        // f32 → f16 → f32 位模式精确往返。
        let values: [f32; 11] = [
            0.0,
            -0.0,
            1.0,
            0.5,
            2.5,
            6.103_515_6e-5, // 最小正规
            5.960_464_5e-8, // 最小半正规（极小）
            65504.0,        // 最大有限（极大）
            -65504.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for v in values {
            let back: f32 = f16_to_f32(f32_to_f16(v));
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "值 {v} 的 f32→f16→f32 往返必须位模式精确"
            );
        }
        // NaN 往返：仍是 NaN 且负载保留。
        let nan_h: u16 = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(nan_h).is_nan());
        assert_eq!(nan_h & 0x03FF, 0x200, "quiet NaN 负载位应保留");
    }

    #[test]
    fn f16_roundtrip_random_1000_points() {
        // 关卡 3（随机 1000 点）：随机 f32（正规区间，含随机符号）——
        // ① f32→f16→f32 幂等（再量化位模式不变）；② 量化误差 ≤ 半 ulp
        // （正规 ≤ 2⁻¹¹ 相对，半正规 ≤ 2⁻²⁵ 绝对）。
        let mut state: u64 = 0xA5F1_00D5;
        for _ in 0..1000 {
            let rand: u64 = splitmix64(&mut state);
            let sign: u32 = ((rand >> 63) as u32) << 31;
            let exp_field: u32 = 103 + ((rand >> 52) & 0x1F) as u32; // 2⁻²⁴ .. 2⁸
            let man: u32 = (rand & 0x007F_FFFF) as u32;
            let x: f32 = f32::from_bits(sign | (exp_field << 23) | man);

            let h: u16 = f32_to_f16(x);
            let y: f32 = f16_to_f32(h);
            assert_eq!(f32_to_f16(y), h, "往返后再量化必须幂等（x = {x}）");

            let abs_x: f32 = x.abs();
            if abs_x < 6.103_515_6e-5 {
                assert!(
                    (y - x).abs() <= 2f32.powi(-25),
                    "半正规量化误差超界 (x = {x})"
                );
            } else {
                assert!(
                    (y - x).abs() <= 2f32.powi(-11) * abs_x,
                    "正规量化误差超半 ulp (x = {x})"
                );
            }
        }
    }

    #[test]
    fn f16_all_65536_patterns_expand_collapse_identity() {
        // 穷举加固：全部 65536 个 f16 位模式，f16→f32→f16 恒等
        // （精确扩展 + RNE 收敛的完备性证明，强于随机抽样）。
        for h in 0..=u16::MAX {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "位模式 {h:#06x} 往返破坏");
        }
    }

    #[test]
    fn fbuf_set_get_and_zero_init() {
        let mut fb = Fbuf16F::new(3, 2);
        assert_eq!(fb.pixels.len(), 3 * 2 * 4);
        assert!(fb.pixels.iter().all(|&p| p == 0));

        fb.set(1, 1, [0.25, 0.5, 65504.0, 1.0]);
        assert_eq!(fb.get(1, 1), [0.25, 0.5, 65504.0, 1.0]);
        // 未写像素保持零。
        assert_eq!(fb.get(0, 0), [0.0, 0.0, 0.0, 0.0]);
        // f16 量化在 set 处发生：位模式可读回。
        assert_eq!(fb.pixels[4 * 4 + 0], 0x3400); // 0.25f16
    }

    #[test]
    fn fbuf_blit_rect_is_bit_exact() {
        // 拼接回写原语：位模式原样拷贝（含 f16 不可精确表示的量化值）。
        let mut src = Fbuf16F::new(4, 3);
        for y in 0..3u32 {
            for x in 0..4u32 {
                let v: f32 = (x * 10 + y) as f32 / 7.0; // 故意产生 f16 舍入
                src.set(x, y, [v, v * 0.5, v * 0.25, 1.0]);
            }
        }
        let mut dst = Fbuf16F::new(6, 5);
        dst.blit_rect(1, 2, &src, 1, 1, 3, 2); // src 的 (1,1)-(3,2) → dst (1,2)

        for row in 0..2u32 {
            for col in 0..3u32 {
                let want: [u16; 4] = src.pixels[(((1 + row) * 4 + (1 + col)) as usize) * 4
                    ..(((1 + row) * 4 + (1 + col)) as usize) * 4 + 4]
                    .try_into()
                    .unwrap();
                let got: [u16; 4] = dst.pixels[(((2 + row) * 6 + (1 + col)) as usize) * 4
                    ..(((2 + row) * 6 + (1 + col)) as usize) * 4 + 4]
                    .try_into()
                    .unwrap();
                assert_eq!(got, want, "blit 必须按位模式原样拷贝");
            }
        }
        // 区域外不受影响。
        assert!(dst.pixels[..(2 * 6 + 1) * 4].iter().all(|&p| p == 0));
    }

    #[test]
    fn fbuf_to_framebuffer_is_delivery_boundary() {
        // 交付边界：f16 → RGBA8 量化（clamp01 × 255 round）只在这里发生。
        let mut fb = Fbuf16F::new(2, 1);
        fb.set(0, 0, [0.0, 0.5, 1.0, 1.0]);
        fb.set(1, 0, [2.0, -1.0, 0.2, 0.0]); // 超界分量应被钳制
        let out = fb.to_framebuffer();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 1);
        assert_eq!(&out.pixels[..8], &[0, 128, 255, 255, 255, 0, 51, 255]);
    }
}
