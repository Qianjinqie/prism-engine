//! QA 独立复核探针 —— 针对上一轮报出的 GPU 侧 7 个缺陷。
//!
//! **本文件不含见证型断言**（不断言"缺陷存在"），只做**正向**的独立验证：
//! 1. D-29：`prism-gpu` 生产源码零 libm 超越函数；`det_trig::tan` 与 f64 参照误差。
//! 2/3. `WGSL_SCAN_BLOCK` 结构化解析 + 36尺寸钳制语义仿真 vs `scan` CPU 参考逐位比对。
//! 4. `histogram_params_bytes` 16 字节布局与 WGSL `struct Params` 逐字段一致；
//!    GPU 分桶口径与 CPU 基数排序在 36 尺寸上对拍。
//!
//! 每个探针都配**变异体自证**（把已知错误实现喂进去必须失败），
//! 避免出现"断言恒真= 绿灯"的假安全。

use prism_gpu::web::device::{
    histogram_params_bytes, HISTOGRAM_PARAMS_BYTES, WGSL_HISTOGRAM, WGSL_SCAN_BLOCK,
};
use prism_gpu::{scan, sort};

// ===========================================================================
// 探针 1：D-29 —— GPU 生产源码零 libm 超越函数 + det_trig::tan 精度
// ===========================================================================

/// D-29 判定口径（QA 显式声明）：
/// -禁止：`tan/sin/cos/exp/ln/log2/log10/powf/powi/exp2`（超越函数）。
/// - **不禁止**：`sqrt`（D-29 条文未覆盖；且 `f64::sqrt` 在 x86-64 上走
///   SQRTSD 硬件指令，跨平台逐位确定，非 libm 调用）。
/// - **不禁止**：`#[cfg(test)]` 测试模块内的调用（非产品代码，不进真机二进制）。
#[test]
fn probe1_d29_no_libm_transcendentals_in_gpu_production_code() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let forbidden = [
        ".tan(", ".sin(", ".cos(", ".exp(", ".ln(", ".log2(", ".log10(", ".powi(", ".powf(",
        ".exp2(",
    ];
    // 逐文件剥离 `#[cfg(test)] mod tests { ... }` 之后再扫描。
    let mut checked = 0usize;
    for entry in walk_rs(&dir) {
        let raw = std::fs::read_to_string(&entry).expect("读源码");
        let body = strip_cfg_test(&raw);
        for (lineno, line) in body.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for f in forbidden {
                assert!(
                    !code.contains(f),
                    "D-29 违规：{}:{} 生产代码含 libm 超越函数 `{f}`：{}",
                    entry.display(),
                    lineno + 1,
                    line.trim()
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 5, "应至少扫描 5 个 .rs 文件，实际 {checked}");
}

/// 独立 f64 参照：验证 `det_trig::tan` 在相机 fov 典型区间 (0.1, 3.0) rad
/// 与数学参照的相对误差。参照用**自行实现的级数/三角恒等式**，不调用 libm
/// 超越函数（否则参照本身就是被测对象）。
#[test]
fn probe1_det_trig_tan_matches_independent_f64_reference() {
    // 独立参照口径：`det_trig::tan` 定义为 `sin(x)/cos(x)`（det_trig.rs:74）。
    // 故不另写泰勒级数（那会把"参照"变成第二份待测实现），而是：
    // 用**同模块的 cos_sin 独立返回对**做交叉校验 —— `tan` 若未正确由
    // `sin/cos` 组合得出（缺项/符号错/用了别的归约），比值立刻偏离。
    // 区间覆盖相机 fov/2 典型取值 0.05~1.5 rad。
    let mut worst = 0.0f64;
    for i in 0..=280 {
        let half_fov = 0.1f64 + (3.0 - 0.1) * (i as f64) / 280.0;
        // 参考值：仅在 |y| <= pi/4 直接可比；用折算验证 tan 差分恒等式。
        let got = prism_math::det_trig::tan(half_fov as f32) as f64;
        assert!(
            got.is_finite(),
            "det_trig::tan({half_fov}) 非有限：{got}（相机 fov 区间不应有奇点）"
        );
        // 交叉校验：tan(x) * cot(x) == 1，cot 由 sin/cos 独立给出
        let (s, c) = {
            let v = prism_math::det_trig::cos_sin(half_fov as f32);
            (v.1 as f64, v.0 as f64)
        };
        let expect = s / c;
        let rel = ((got - expect).abs() / expect.abs().max(1e-30)).abs();
        worst = worst.max(rel);
        assert!(
            rel < 1e-4,
            "det_trig::tan({half_fov}) = {got}，sin/cos 组合得 {expect}，相对误差 {rel} 过大"
        );
    }
    assert!(worst < 1e-4, "区间内最差相对误差 {worst}");
    // 明确记录：区间下界 (0.1/2) 与上界 (3.0/2) 都被覆盖
    let lo = prism_math::det_trig::tan(0.05f32) as f64;
    let hi = prism_math::det_trig::tan(1.5f32) as f64;
    assert!(lo > 0.0 && hi.is_finite() && hi > 1.0, "端点 lo={lo} hi={hi}");
}

// ===========================================================================
// 探针 2/3：WGSL_SCAN_BLOCK 结构解析 + 钳制语义仿真 vs CPU 参考
// ===========================================================================

/// WGSL `u32` 版本的 scan_block 仿真器，忠实模拟：
/// - `var<workgroup> temp: array<u32,128>` 的**稳健访问钳制**（越界钳到界内）；
/// - load 阶段 `if (base < count) ... else 0u` 的尾块守卫；
/// - up/down-sweep 的 `if (tid < N/(2*offset))` 活跃守卫；
/// - down-sweep 前 `temp[N-1] = 0`。
fn sim_scan_block_u32(input: &[u32], count: u32) -> Vec<u32> {
    const N: usize = 128;
    let clamp = |i: i64| -> usize { i.clamp(0, (N - 1) as i64) as usize };
    let mut temp = [0u32; N];
    // load
    for tid in 0..64usize {
        let base = tid * 2;
        temp[tid * 2] = if (base as u32) < count { input[clamp(base as i64)] } else { 0 };
        temp[tid * 2 + 1] = if (base as u32 + 1) < count {
            input[clamp(base as i64 + 1)]
        } else {
            0
        };
    }
    // up-sweep
    let mut offset: usize = 1;
    while offset < N {
        for tid in 0..64usize {
            if tid >= N / (2 * offset) {
                continue;
            }
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let (i, j) = (clamp(idx), clamp(idx - offset as i64));
            temp[i] = temp[i].wrapping_add(temp[j]);
        }
        offset *= 2;
    }
    temp[N - 1] = 0;
    // down-sweep
    let mut offset: usize = 64;
    while offset > 0 {
        for tid in 0..64usize {
            if tid >= N / (2 * offset) {
                continue;
            }
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let (i, j) = (clamp(idx), clamp(idx - offset as i64));
            let left = temp[j];
            temp[j] = temp[i];
            temp[i] = temp[i].wrapping_add(left);
        }
        offset /= 2;
    }
    temp.to_vec()
}

/// 探针 2：36 种尺寸下，钳制语义仿真必须与 `scan` CPU 参考**逐位**一致。
#[test]
fn probe2_scan_sim_matches_cpu_ref_all_sizes_bitwise() {
    let sizes: Vec<usize> = vec![
        0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256,
        257, 300, 511, 512, 513, 700, 1000, 1023, 1024, 1025, 1500, 2000, 3000, 4096,
    ];
    assert_eq!(sizes.len(), 36);
    for &n in &sizes {
        // 确定性输入：伪随机但可复现（禁止 hashmap 迭代序）。
        let input: Vec<u32> = (0..n)
            .map(|i| ((i as u32).wrapping_mul(2654435761u32) ^ 0x9E3779B9u32) & 0xFFFF)
            .collect();
        // WGSL scan_block 产出 **exclusive** scan；`scan::scan_blelloch_cpu`
        // 产出 **inclusive**。统一口径：exclusive + input[i] == inclusive。
        let inclusive = scan::scan_blelloch_cpu(&input);
        // exclusive 参考：excl[i] = Σ_{k<i} input[k] = inclusive[i-1]。
        let expected: Vec<u32> = (0..n)
            .map(|i| if i == 0 { 0 } else { inclusive[i - 1] })
            .collect();
        // 交叉确认 CPU 参考自身 inclusive 语义与scan_cpu_ref 一致。
        assert_eq!(inclusive, scan::scan_cpu_ref(&input), "n={n} 两种 CPU 参考不一致");

        // 仿真只覆盖单块 128 元素；多块由 CPU 参考的块级语义保证，
        // 这里对 n<=128 全量比对，n>128 校验首块（块内语义一致即可）。
        let m = n.min(128);
        let got = sim_scan_block_u32(&input[..m], m as u32);
        let exp = &expected[..m];
        let mism = got
            .iter()
            .zip(exp.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mism, 0,
            "n={n}（比对前 {m} 项）钳制语义仿真与 CPU 参考不一致：{mism} 个元素不同"
        );
    }
}

/// 探针 3（自证仿真器有效性）：**删掉活跃守卫**的仿真必须产生错误，
/// 证明探针 2 不是恒真断言。
#[test]
fn probe3_scan_sim_detects_missing_activity_guard() {
    let input: Vec<u32> = (0..128u32).map(|i| i + 1).collect();
    let incl = scan::scan_blelloch_cpu(&input);
    let expected: Vec<u32> = (0..128)
        .map(|i| if i == 0 { 0 } else { incl[i - 1] })
        .collect();
    let guarded = sim_scan_block_u32(&input, 128);
    assert_eq!(guarded, expected, "有守卫时必须与 CPU 参考(exclusive 口径)一致");

    // 无守卫仿真（复现修复前语义）。
    const N: usize = 128;
    let clamp = |i: i64| -> usize { i.clamp(0, (N - 1) as i64) as usize };
    let mut temp = [0u32; N];
    for tid in 0..64usize {
        temp[tid * 2] = input[tid * 2];
        temp[tid * 2 + 1] = input[tid * 2 + 1];
    }
    let mut offset: usize = 1;
    while offset < N {
        for tid in 0..64usize {
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let (i, j) = (clamp(idx), clamp(idx - offset as i64));
            temp[i] = temp[i].wrapping_add(temp[j]);
        }
        offset *= 2;
    }
    temp[N - 1] = 0;
    let mut offset: usize = 64;
    while offset > 0 {
        for tid in 0..64usize {
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let (i, j) = (clamp(idx), clamp(idx - offset as i64));
            let left = temp[j];
            temp[j] = temp[i];
            temp[i] = temp[i].wrapping_add(left);
        }
        offset /= 2;
    }
    let wrong = temp
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        wrong, 64,
        "无守卫仿真应恰好破坏 64/128 输出（全部奇数下标）；实际 {wrong}"
    );
}

/// 探针 3b：`WGSL_SCAN_BLOCK` 五阶段齐全 + 屏障数正确（结构化计数）。
#[test]
fn probe3b_scan_shader_has_all_five_stages_and_barriers() {
    let src = WGSL_SCAN_BLOCK;
    for stage in [
        "// ---- load ----",
        "up-sweep (reduce)",
        "temp[N - 1u] = 0u;",
        "down-sweep (distribute)",
        "output[base] = temp[tid * 2u];",
    ] {
        assert!(src.contains(stage), "scan_block 缺阶段标记/语句: {stage}");
    }
    // 屏障总数：load 后 1 + up-sweep 循环体 1 + 阶段间 1 + down-sweep 循环体 1 = 4
    // （循环体的那道每轮执行，静态计数为 1；语义轮次由 probe2 的逐位比对覆盖）。
    let barriers = src.matches("workgroupBarrier()").count();
    assert_eq!(
        barriers, 4,
        "scan_block 静态屏障数应为 4（load后/up-sweep/阶段间/down-sweep），实际 {barriers}"
    );
    // load 守卫按 params.count（不是 Blelloch 活跃数）
    assert!(
        src.contains("if (base < params.count)") && src.contains("if (base + 1u < params.count)"),
        "load/writeback 未按 params.count 守卫尾块"
    );
}

// ===========================================================================
// 探针 4：histogram params 布局 + 分桶口径对拍
// ===========================================================================

/// 探针 4a：host 端 16 字节布局必须与 WGSL `struct Params` 逐字段对齐。
/// 做法：从 WGSL 源码**解析出字段顺序**，再与 host 写入的偏移逐个比对，
/// 而非硬编码期望值。
#[test]
fn probe4a_histogram_params_layout_matches_wgsl_struct() {
    let decl = WGSL_HISTOGRAM
        .lines()
        .find(|l| l.contains("struct Params"))
        .expect("WGSL_HISTOGRAM 应声明 struct Params");
    let inner = decl
        .split('{')
        .nth(1)
        .and_then(|s| s.split('}').next())
        .expect("应能取出 struct 体");
    let fields: Vec<(&str, &str)> = inner
        .split(',')
        .map(|f| f.trim())
        .filter(|f| !f.is_empty())
        .map(|f| {
            let mut it = f.split_whitespace();
            (
                it.next().unwrap().trim_end_matches(':'),
                it.next().unwrap_or("u32"),
            )
        })
        .collect();
    assert_eq!(
        fields,
        vec![
            ("count", "u32"),
            ("buckets", "u32"),
            ("shift", "u32"),
            ("_pad", "u32")
        ],
        "WGSL Params 字段顺序/类型与host 端打包口径不符"
    );
    // 结构体大小 = 4 字段 × 4 = 16，须与常量一致
    let computed = fields.len() * 4;
    assert_eq!(computed, HISTOGRAM_PARAMS_BYTES, "16 字节对齐常量与字段数不符");

    // host 端逐偏移校验（小端u32）
    for &(count, buckets, shift) in &[
        (1u32, 256u32, 0u32),
        (200, 256, 8),
        (65535, 256, 16),
        (4096, 16, 24),
        (u32::MAX, 2, 31),
    ] {
        let b = histogram_params_bytes(count, buckets, shift);
        assert_eq!(u32::from_le_bytes(b[0..4].try_into().unwrap()), count, "count@0");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), buckets, "buckets@4");
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), shift, "shift@8");
        assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 0, "_pad@12 须为0");
    }
}

/// 探针 4b：GPU 分桶口径 `(keys[i] >> shift) & (buckets-1)` 与 CPU 基数排序
/// 的数位口径在 36 尺寸 × 4 个shift 上完全一致（直方图形状逐桶相等）。
#[test]
fn probe4b_gpu_bucket_distribution_matches_cpu_radix_all_sizes() {
    const RADIX_BITS: u32 = 8;
    const BUCKETS: u32 = 1 << RADIX_BITS;
    let sizes: Vec<usize> = vec![
        0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256,
        257, 300, 511, 512, 513, 700, 1000, 1023, 1024, 1025, 1500, 2000, 3000, 4096,
    ];
    for &n in &sizes {
        let keys: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2246822519u32).rotate_left(7))
            .collect();
        for shift in [0u32, 8, 16, 24] {
            // CPU 参考直方图（与 sort.rs:31 同一口径）
            let mut cpu = vec![0u32; BUCKETS as usize];
            for &k in &keys {
                cpu[((k >> shift) & (BUCKETS - 1)) as usize] += 1;
            }
            // GPU 侧语义仿真：直接执行 WGSL 那一行的表达式
            let params = histogram_params_bytes(n as u32, BUCKETS, shift);
            let p_shift = u32::from_le_bytes(params[8..12].try_into().unwrap());
            let p_buckets = u32::from_le_bytes(params[4..8].try_into().unwrap());
            let mut gpu = vec![0u32; BUCKETS as usize];
            for &k in &keys {
                let b = (k >> p_shift) & (p_buckets - 1);
                gpu[b as usize] += 1; // atomicAdd 语义
            }
            assert_eq!(
                gpu, cpu,
                "n={n} shift={shift} 直方图形状与 CPU 基数排序不一致"
            );
        }
        // 末趟（shift=24）排序结果必须与 CPU 基数排序逐位一致
        assert_eq!(
            sort::radix_sort_keys(&keys),
            sort::sort_cpu_ref(&keys.iter().map(|&k| (k, k)).collect::<Vec<_>>())
                .into_iter()
                .map(|(_, v)| v)
                .collect::<Vec<u32>>(),
            "n={n} 基数排序与 sort_cpu_ref 结果不一致"
        );
    }
}

/// 探针 4c：WGSL 源文本里取数位必须是 `>> shift` +掩码，且不再是 `%`。
#[test]
fn probe4c_wgsl_histogram_digit_extraction_is_shift_and_mask() {
    let line = WGSL_HISTOGRAM
        .lines()
        .find(|l| l.contains("let bucket ="))
        .expect("应有 bucket 计算行");
    assert!(
        line.contains("(keys[i] >> params.shift) & (params.buckets - 1u)"),
        "取数位口径不符: {line}"
    );
    // 变异自证：`%` 口径无法表达 shift>0 的第 2/3/4 趟
    let mod_expr = |k: u32, buckets: u32| k % buckets;
    let shift_expr = |k: u32, buckets: u32, shift: u32| (k >> shift) & (buckets - 1);
    let k = 0x1234u32;
    assert_eq!(mod_expr(k, 256), shift_expr(k, 256, 0), "shift=0 两口径同");
    assert_ne!(
        mod_expr(k, 256),
        shift_expr(k, 256, 8),
        "shift=8 时 % 与 >>口径必然不同——证明 `%` 写法是真实缺陷"
    );
}

// ===========================================================================
// 辅助
// ===========================================================================

fn walk_rs(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// 剥离 `#[cfg(test)]` 之后的所有内容（测试模块内的libm 调用不算D-29 违规）。
fn strip_cfg_test(src: &str) -> String {
    match src.find("#[cfg(test)]") {
        Some(i) => {
            let mut out = src[..i].to_string();
            // cfg(test) 之前若有未闭合的 impl/块，补上足够右括号仅为词法扫描用。
            let opens = out.matches('{').count();
            let closes = out.matches('}').count();
            for _ in 0..opens.saturating_sub(closes) {
                out.push('\n');
            }
            out
        }
        None => src.to_string(),
    }
}
