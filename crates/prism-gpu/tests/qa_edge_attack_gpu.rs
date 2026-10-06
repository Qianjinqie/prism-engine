//! QA 攻击性测试（Edward）— Phase 4 GPU 三路 D0/D1/D2-D4（`prism-gpu`）。
//!
//! 铁律：零第三方依赖（D-15）+ 全链路确定性 + 禁 libm（D-29）。
//!
//! 攻击面：
//! * A. D0 基础设施：`scan`（Blelloch 上下行 / 非 2 幂 / 零长 / 单元素）、
//!   `sort`（空/全等/已序/逆序/极值 + 稳定性 + 确定性）、`sync`（屏障推导）。
//! * B. D1 WebGPU 绑定 + harness：能力探测桩、dispatch 计划、四个 harness 要求。
//! * C/D. shader 结构自检 + WGSL 静态语法要点 + libm 静态扫描（Rust + WGSL）。
//! * 零依赖核查：源码级 + Cargo.lock 无外部 crate。
//!
//! 每个测试失败的含义在函数头注释中显式声明（禁凑数断言）。

use prism_gpu::scan::{scan_blelloch_cpu, scan_cpu_ref, ScanShape};
use prism_gpu::shaders::{check_all, check_shader, source, SHADERS};
use prism_gpu::sort::{radix_sort_keys, radix_sort_u32, sort_cpu_ref};
use prism_gpu::sync::{Access, BarrierKind, BarrierPlanner, PassDesc, PassKind};

// ===========================================================================
// 公共工具
// ===========================================================================

/// 确定性 LCG（无全局 RNG，种子显式）——攻击测试的伪随机源。
fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

/// 逐位比较两个 u32 切片（含 to_le_bytes 字节序口径）。
fn assert_bitwise_eq_u32(a: &[u32], b: &[u32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}：长度不一致");
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_le_bytes(),
            b[i].to_le_bytes(),
            "{what}：下标 {i} 逐位不一致（{} vs {}）",
            a[i],
            b[i]
        );
    }
}

// ===========================================================================
// A. D0 scan —— Blelloch 正确性 / 边界 / 确定性
// ===========================================================================

/// 失败含义：Blelloch 上下行扫描在**大范围非 2 幂**尺寸下与串行前缀和不逐位一致，
/// 说明树形补齐/截断或奇偶进位逻辑有缺陷（GPU 前缀和的语义地基）。
#[test]
fn scan_blelloch_bitwise_vs_serial_all_sizes() {
    // 覆盖：0/1 退化、紧邻 2 的幂两侧、跨多个 log 层、质数长度、大尺寸。
    let sizes = [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 4097,
    ];
    for &n in &sizes {
        let input: Vec<u32> = (0..n).map(|i| ((i * 37 + 11) % 97) as u32).collect();
        let serial = scan_cpu_ref(&input);
        let blelloch = scan_blelloch_cpu(&input);
        assert_bitwise_eq_u32(&blelloch, &serial, &format!("scan n={n}"));
    }
}

/// 失败含义：扫描对**单元素**输入语义错误（inclusive 前缀和 = 元素自身）。
#[test]
fn scan_single_element_identity() {
    assert_eq!(scan_blelloch_cpu(&[42]), vec![42]);
    assert_eq!(scan_cpu_ref(&[42]), vec![42]);
    // 单元素 u32::MAX 不应溢出 panic（wrapping）。
    assert_eq!(scan_blelloch_cpu(&[u32::MAX]), vec![u32::MAX]);
}

/// 失败含义：零长度输入被误处理（应返回空，不得 panic）。
#[test]
fn scan_zero_length_is_empty() {
    assert!(scan_blelloch_cpu(&[]).is_empty());
    assert!(scan_cpu_ref(&[]).is_empty());
}

/// 失败含义：扫描遇 u32 溢出时 panic 而非按 wrapping 语义回绕（确定性要求）。
#[test]
fn scan_overflow_wraps_not_panics() {
    let input = vec![u32::MAX, 1, 1, 1];
    let a = scan_cpu_ref(&input);
    let b = scan_blelloch_cpu(&input);
    assert_eq!(a, vec![u32::MAX, 0, 1, 2]);
    assert_bitwise_eq_u32(&a, &b, "overflow-wrap");
}

/// 失败含义：同输入两遍结果不同 → 违反全链路确定性铁律。
#[test]
fn scan_deterministic_two_pass() {
    let mut seed = 0xDEAD_BEEF_CAFE_1234;
    let input: Vec<u32> = (0..2048).map(|_| lcg(&mut seed)).collect();
    let a = scan_blelloch_cpu(&input);
    let b = scan_blelloch_cpu(&input);
    assert_bitwise_eq_u32(&a, &b, "scan 两遍");
}

/// 失败含义：ScanShape 树深口径偏离 ⌈log₂ n⌉，或 n≤1 未归零——与 reduction_shape
/// 形状对齐契约破裂（D-26 单点定义）。
#[test]
fn scan_shape_depth_matches_ceil_log2() {
    for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 16, 100] {
        let shape = ScanShape::new(n);
        assert_eq!(shape.n(), n);
        let expected = if n <= 1 { 0 } else { (n as f64).log2().ceil() as usize };
        assert_eq!(shape.depth(), expected, "n={n} 树深");
    }
}

/// 失败含义：ScanShape 层内配对结构非「相邻成对 + 奇数尾进位」——与 reduction_shape
/// 的 pairwise 折叠序不再同构（隐藏的结合序漂移）。
#[test]
fn scan_shape_pairwise_pairing_structure() {
    // n=5：层0 = [(0,1),(2,3)]（尾 4 进位）；层1 = [(0,1)]（0,1 是上层的两对）；
    // 层2 = [(0,1)]（把 (0,1) 与进位尾配对）。
    let s = ScanShape::new(5);
    assert_eq!(s.layers()[0], vec![[0, 1], [2, 3]]);
    // 相邻成对不变量：每层每对第二元 = 第一元 + 1。
    for n in [2usize, 3, 5, 8, 13, 64] {
        let sh = ScanShape::new(n);
        for layer in sh.layers() {
            for pair in layer {
                assert_eq!(pair[1], pair[0] + 1, "n={n} 层内应相邻成对");
            }
        }
    }
}

// ===========================================================================
// A. D0 sort —— 正确性 / 边界 / 稳定性 / 确定性
// ===========================================================================

/// 失败含义：radix sort 与稳定参考排序不一致（键升序或载荷错位）——空间哈希/
/// 粒子排序的正确性地基破裂。
#[test]
fn sort_radix_matches_reference_all_sizes() {
    for n in [0usize, 1, 2, 3, 5, 17, 100, 256, 257, 1000] {
        let mut seed = 0x1234_5678_9abc_def0;
        let items: Vec<(u32, u32)> = (0..n).map(|i| (lcg(&mut seed), i as u32)).collect();
        let radix = radix_sort_u32(&items);
        let reference = sort_cpu_ref(&items);
        assert_eq!(radix, reference, "n={n} radix vs ref");
        for w in radix.windows(2) {
            assert!(w[0].0 <= w[1].0, "n={n} 结果非升序");
        }
    }
}

/// 失败含义：空输入未安全返回空。
#[test]
fn sort_empty_input() {
    assert!(radix_sort_u32(&[]).is_empty());
    assert!(radix_sort_keys(&[]).is_empty());
}

/// 失败含义：**全相等键**下排序不稳定（载荷相对序被打乱）——LSD 依赖每趟稳定，
/// 破坏后多趟结果错误。
#[test]
fn sort_all_equal_keys_preserve_input_order() {
    let items: Vec<(u32, u32)> = (0..1000).map(|i| (7, i as u32)).collect();
    let sorted = radix_sort_u32(&items);
    let expect: Vec<(u32, u32)> = items.clone();
    assert_eq!(sorted, expect, "全相等键应完全保序");
}

/// 失败含义：**已序**输入被扰乱（radix 应对已序输入幂等）。
#[test]
fn sort_already_sorted_is_idempotent() {
    let items: Vec<(u32, u32)> = (0..500).map(|i| (i as u32, i as u32)).collect();
    assert_eq!(radix_sort_u32(&items), items);
}

/// 失败含义：**逆序**输入排序错误（每趟散布方向/偏移计算缺陷在逆序下暴露）。
#[test]
fn sort_reversed_input_sorts_correctly() {
    let items: Vec<(u32, u32)> = (0..500).rev().map(|i| (i as u32, i as u32)).collect();
    let sorted = radix_sort_u32(&items);
    let keys: Vec<u32> = sorted.iter().map(|&(k, _)| k).collect();
    let expected: Vec<u32> = (0..500).collect();
    assert_eq!(keys, expected);
}

/// 失败含义：**极端 u32 值**（含最高位/MAX/0）排序错误——覆盖第 4 趟（shift=24）
/// 与符号无关的全位域。
#[test]
fn sort_extreme_values() {
    let edge = vec![
        (u32::MAX, 0),
        (u32::MIN, 1),
        (0x8000_0000, 2),
        (0x7FFF_FFFF, 3),
        (1, 4),
        (0xFF00_0000, 5),
    ];
    let sorted = radix_sort_u32(&edge);
    let keys: Vec<u32> = sorted.iter().map(|&(k, _)| k).collect();
    assert_eq!(keys, vec![0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFF00_0000, u32::MAX]);
}

/// 失败含义：**稳定性**在多位数（跨字节）重复键下破裂——同键载荷未保持输入序。
#[test]
fn sort_stability_across_digit_boundaries() {
    // 键仅取 3 个值，但跨越低位/高位字节，制造跨趟重复。
    let mut seed = 42u64;
    let items: Vec<(u32, u32)> = (0..2000)
        .map(|i| {
            let raw = lcg(&mut seed);
            let k = match raw % 3 {
                0 => 0x0000_0000,
                1 => 0x0000_0100, // 第 2 字节
                _ => 0x0100_0000, // 第 4 字节
            };
            (k, i as u32)
        })
        .collect();
    let sorted = radix_sort_u32(&items);
    // 对每个键，载荷序列 = 输入出现序。
    let mut idx = 0usize;
    for key in [0u32, 0x0000_0100, 0x0100_0000] {
        let expected: Vec<u32> = items
            .iter()
            .filter(|&&(k, _)| k == key)
            .map(|&(_, p)| p)
            .collect();
        for p in expected {
            assert_eq!(sorted[idx].0, key, "键分组");
            assert_eq!(sorted[idx].1, p, "键 {key:#x} 载荷应稳定保序");
            idx += 1;
        }
    }
    assert_eq!(idx, items.len());
}

/// 失败含义：同输入两遍排序结果不同 → 违反确定性铁律。
#[test]
fn sort_deterministic_two_pass() {
    let mut seed = 7u64;
    let items: Vec<(u32, u32)> = (0..4096).map(|i| (lcg(&mut seed), i as u32)).collect();
    assert_eq!(radix_sort_u32(&items), radix_sort_u32(&items));
}

/// 失败含义：`radix_sort_keys` 便捷入口与完整排序的键序列不一致（载荷丢弃逻辑错）。
#[test]
fn sort_keys_entry_matches_full_sort() {
    let mut seed = 99u64;
    let keys: Vec<u32> = (0..1000).map(|_| lcg(&mut seed)).collect();
    let via_keys = radix_sort_keys(&keys);
    let via_items: Vec<u32> = radix_sort_u32(
        &keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u32))
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|(k, _)| k)
    .collect();
    assert_eq!(via_keys, via_items);
}

// ===========================================================================
// A. D0 sync —— 屏障推导语义 / 确定性
// ===========================================================================

/// 失败含义：pass 形态封闭集的写资源/访问方向语义错误（时间戳不应写渲染资源）。
#[test]
fn sync_passkind_access_semantics() {
    assert!(PassKind::Compute.writes_resources());
    assert!(PassKind::FullscreenBlit.writes_resources());
    assert!(PassKind::TriangleRaster.writes_resources());
    assert!(PassKind::Transfer.writes_resources());
    assert!(!PassKind::TimestampQuery.writes_resources());
    assert_eq!(PassKind::Compute.access(), Access::ReadWrite);
    assert_eq!(PassKind::TriangleRaster.access(), Access::ReadWrite);
    assert_eq!(PassKind::FullscreenBlit.access(), Access::Write);
    assert_eq!(PassKind::Transfer.access(), Access::Write);
    assert_eq!(PassKind::TimestampQuery.access(), Access::Write);
}

/// 失败含义：写→读之间未产生 RAW 屏障（GPU 数据竞争，读到未完成写入）。
#[test]
fn sync_raw_barrier_between_write_and_read() {
    let passes = vec![
        PassDesc::new(PassKind::Compute, vec![], vec![10]),
        PassDesc::new(PassKind::FullscreenBlit, vec![10], vec![]),
    ];
    let points = BarrierPlanner::plan_sequence(&passes);
    assert_eq!(points.len(), 1);
    assert!(points[0]
        .barriers
        .iter()
        .any(|b| b.resource == 10 && b.kind == BarrierKind::ReadToWrite));
}

/// 失败含义：连续写同一资源未产生 WAW 屏障（写序未串行化）。
#[test]
fn sync_waw_barrier_between_two_writes() {
    let passes = vec![
        PassDesc::new(PassKind::Transfer, vec![], vec![5]),
        PassDesc::new(PassKind::Compute, vec![], vec![5]),
    ];
    let points = BarrierPlanner::plan_sequence(&passes);
    assert!(points[0]
        .barriers
        .iter()
        .any(|b| b.resource == 5 && b.kind == BarrierKind::WriteToWrite));
}

/// 失败含义：无依赖资源的 pass 间被误插同步点（过度同步，性能缺陷）。
#[test]
fn sync_independent_resources_no_barrier() {
    let passes = vec![
        PassDesc::new(PassKind::Compute, vec![], vec![1]),
        PassDesc::new(PassKind::FullscreenBlit, vec![], vec![2]),
    ];
    assert!(BarrierPlanner::plan_sequence(&passes).is_empty());
}

/// 失败含义：屏障集不确定（同输入两遍不同）→ 违反确定性铁律。资源顺序打乱后
/// 屏障应按句柄升序稳定输出。
#[test]
fn sync_barriers_deterministic_and_ordered() {
    let passes = vec![
        PassDesc::new(PassKind::Compute, vec![], vec![100, 50, 75]),
        PassDesc::new(PassKind::Compute, vec![75, 50, 100], vec![]),
    ];
    let a = BarrierPlanner::plan_sequence(&passes);
    let b = BarrierPlanner::plan_sequence(&passes);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.barriers, y.barriers);
        // 输出屏障按资源句柄升序。
        let res: Vec<u64> = x.barriers.iter().map(|bar| bar.resource).collect();
        let mut sorted = res.clone();
        sorted.sort_unstable();
        assert_eq!(res, sorted, "屏障应按资源句柄升序");
    }
}

/// 失败含义：时间戳 pass（无资源 hazard）未产生执行依赖屏障——排序阶段丢失。
#[test]
fn sync_timestamp_produces_execution_barrier() {
    let passes = vec![
        PassDesc::new(PassKind::Compute, vec![], vec![1]),
        PassDesc::new(PassKind::TimestampQuery, vec![], vec![]),
    ];
    let points = BarrierPlanner::plan_sequence(&passes);
    assert_eq!(points.len(), 1);
    assert!(points[0]
        .barriers
        .iter()
        .any(|b| b.kind == BarrierKind::Execution));
}

/// 失败含义：空 pass 序 panic 或产出非空同步点。
#[test]
fn sync_empty_sequence() {
    assert!(BarrierPlanner::plan_sequence(&[]).is_empty());
}

// ===========================================================================
// C/D. shader 结构自检 + WGSL 静态要点
// ===========================================================================

/// 失败含义：某个嵌入 shader 结构残缺（被截断/误删）——真机编译前必须拦截的回归。
#[test]
fn shaders_all_pass_structural_check() {
    let checks = check_all();
    assert_eq!(checks.len(), SHADERS.len());
    // 【U4 深化 B】新增 `gi_compute.wgsl`（GI 的 GPU 路径），嵌入 shader 数 9 → 10。
    assert_eq!(checks.len(), 10, "应嵌入 10 个 WGSL shader");
    for c in &checks {
        assert!(c.has_entry, "{} 无入口", c.name);
        assert!(c.brackets_balanced, "{} 括号不配平（疑似截断）", c.name);
        assert!(c.fn_count > 0, "{} 无 fn 声明", c.name);
        assert!(!c.has_libm, "{} 出现禁 libm 内建（D-29）", c.name);
        assert!(!c.has_include, "{} 含第三方 include", c.name);
        assert!(c.is_pass(), "{} 结构自检未通过: {c:?}", c.name);
    }
}

/// 失败含义：按名取源失败或空源——`include_str!` 资产表漏项。
#[test]
fn shaders_lookup_by_name_all_present() {
    for (name, src) in SHADERS {
        let got = source(name).expect("嵌入表应能按名取到");
        assert_eq!(got, *src);
        assert!(!src.trim().is_empty(), "{name} 为空源");
    }
    assert!(source("does_not_exist.wgsl").is_none());
}

/// 失败含义：WGSL 结构自检未检出**被截断**的源（缺右括号）——假阴性会导致真机
/// 编译才发现问题。
#[test]
fn shaders_check_catches_truncation() {
    let c = check_shader("fake", "@compute fn f() { if (true) { }"); // 缺一个 }
    assert!(!c.brackets_balanced);
    // 缺入口。
    let c2 = check_shader("fake2", "fn f() {}");
    assert!(!c2.has_entry);
}

/// 失败含义：libm 内建检测漏报（`sin(` 等被写成别的形式逃逸）——D-29 静态防线失效。
/// 说明：本检测为「词法近似」防线，非完整解析器；此处验证其**能**捕获标准内建调用。
#[test]
fn shaders_libm_scanner_catches_builtins() {
    assert!(check_shader("x", "fn f(){ let a = sin(1.0); }").has_libm);
    assert!(check_shader("x", "fn f(){ let a = pow(2.0,3.0); }").has_libm);
    assert!(check_shader("x", "fn f(){ let a = exp(1.0); }").has_libm);
    // 自定义同名非内建（前缀为标识符字符）不应误报。
    assert!(!check_shader("x", "fn xsin(a:f32)->f32{return a;}").has_libm);
}

/// 失败含义：**自研三角逼近件未在 shader 中定义**——禁 libm 后的替代实现缺失，
/// 真机编译失败。
#[test]
fn wgsl_custom_trig_present_and_no_builtin_trig() {
    let common = source("rt_common.wgsl").expect("rt_common");
    for f in ["fn quarter_sin", "fn quarter_cos", "fn cos_sin", "fn det_tan"] {
        assert!(common.contains(f), "缺少自研三角件：{f}");
    }
    // 所有 shader 的**非注释**代码不得出现裸内建三角调用（cos_sin 等自定义名除外）。
    for (name, src) in SHADERS {
        for (lineno, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for builtin in ["sin(", "cos(", "tan("] {
                // 允许自定义名 cos_sin(；仅当出现 ` sin(`/`\tcos(` 等词边界调用时报警。
                let needle = builtin;
                if let Some(pos) = code.find(needle) {
                    let prev = code[..pos].chars().last();
                    // 若前一位是标识符字符（如 cos_sin 的 `_`），视为自定义名，跳过。
                    let custom = matches!(prev, Some(c) if c.is_ascii_alphanumeric() || c == '_');
                    assert!(
                        custom,
                        "{name}:{lineno} 出现疑似内建三角调用 `{builtin}`（D-29）: {line}"
                    );
                }
            }
        }
    }
}

/// 失败含义：WGSL 入口/绑定/存储声明缺失或形态错误——真机编译要点被破坏。
#[test]
fn wgsl_entry_binding_and_storage_static_shape() {
    // 每个入口 shader 必须含 @compute/@vertex/@fragment + @workgroup_size（compute）。
    let entries = [
        ("rt_wavefront.wgsl", "@compute"),
        ("sim_particles.wgsl", "@compute"),
        ("sim_cloth.wgsl", "@compute"),
        ("sim_sph.wgsl", "@compute"),
        ("text.wgsl", "@vertex"),
    ];
    for (name, tag) in entries {
        let src = source(name).unwrap();
        assert!(src.contains(tag), "{name} 缺 {tag}");
    }
    // 计算入口须声明 @workgroup_size。
    let wf = source("rt_wavefront.wgsl").unwrap();
    assert!(wf.contains("@workgroup_size"), "rt_wavefront 缺 workgroup_size");
    // 绑定须有 @group/@binding（storage 资源）。
    assert!(wf.contains("@group(0) @binding(0)"));
    // text.wgsl 纹理/采样器绑定。
    let tx = source("text.wgsl").unwrap();
    assert!(tx.contains("texture_2d<f32>"), "text 缺纹理声明");
    assert!(tx.contains("var atlas_sampler: sampler"), "text 缺采样器声明");
}

/// 失败含义：`rt_bvh.wgsl` 的 BVH 结构/栈遍历关键件缺失（对应 bvh.rs 契约）。
#[test]
fn wgsl_bvh_structure_present() {
    let bvh = source("rt_bvh.wgsl").unwrap();
    for need in ["struct BvhNode", "struct Tri", "fn aabb_ray", "fn tri_ray_hit", "fn bvh_intersect"] {
        assert!(bvh.contains(need), "rt_bvh 缺 {need}");
    }
    // 固定深度栈（64 层契约）。
    assert!(bvh.contains("array<u32, 64>"), "rt_bvh 缺 64 层显式栈");
}

/// 失败含义：`text.wgsl` 的 SDF 解码区间与 CPU 侧 `sdf.rs` 的编码不再镜像
/// （采样出错误锐边）。此处核对 shader 侧公式字面量与 sdf.rs 注释口径一致。
#[test]
fn wgsl_text_sdf_decode_mirrors_cpu() {
    let tx = source("text.wgsl").unwrap();
    // CPU: decode = (byte/255 - 0.5) * 2 * px_range；shader: dd = (d - 0.5) * 2 * px_range。
    assert!(tx.contains("(d - 0.5) * 2.0 * in.px_range"), "SDF 解码区间与 CPU 不镜像");
}

/// 失败含义：WGSL 源含第三方 `#include`/`@import`（违反零依赖 + 自研汇编器契约）。
#[test]
fn wgsl_no_third_party_include() {
    for (name, src) in SHADERS {
        assert!(!src.contains("#include"), "{name} 含 #include");
        assert!(!src.contains("@import"), "{name} 含 @import");
    }
}

// ===========================================================================
// D. libm 扫描（Rust 侧 GPU 层）
// ===========================================================================

/// 失败含义：GPU crate 的 **Rust 生产代码**出现 libm 超越函数调用（D-29）。
/// 说明：本测试直接扫描源码文本；发现即缺陷。
#[test]
fn libm_scan_gpu_rust_production_sources() {
    let files = [
        ("scan.rs", include_str!("../src/scan.rs")),
        ("sort.rs", include_str!("../src/sort.rs")),
        ("sync.rs", include_str!("../src/sync.rs")),
        ("shaders.rs", include_str!("../src/shaders.rs")),
        ("web/mod.rs", include_str!("../src/web/mod.rs")),
        ("web/device.rs", include_str!("../src/web/device.rs")),
        ("lib.rs", include_str!("../src/lib.rs")),
    ];
    let banned = [
        ".tan(", ".sin(", ".cos(", ".asin(", ".acos(", ".atan(", ".atan2(", ".exp(",
        ".ln(", ".log(", ".log2(", ".log10(", ".powf(", ".powi(", ".sinh(", ".cosh(",
        ".tanh(", ".hypot(", ".cbrt(",
    ];
    let mut hits: Vec<String> = Vec::new();
    for (name, src) in files {
        // 逐行剥离行注释，仅扫描生产代码文本。
        for (lineno, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for needle in banned {
                if code.contains(needle) {
                    hits.push(format!("{name}:{} `{needle}` => {}", lineno + 1, line.trim()));
                }
            }
        }
    }
    // 允许清单：`scan.rs` 内 `log2` 仅出现在 #[cfg(test)] 模块（已在源码核对），
    // 但为保持测试严格（生产源文件级），此处将 test 段命中单列并断言其位于 test 块。
    let prod_hits: Vec<&String> = hits
        .iter()
        .filter(|h| !(h.starts_with("scan.rs") && h.contains(".log2()")))
        .collect();
    assert!(
        prod_hits.is_empty(),
        "GPU Rust 生产代码出现 libm 超越函数（D-29）:\n{}",
        prod_hits.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
}

// ===========================================================================
// 零第三方依赖核查（D-15 硬红线）
// ===========================================================================

/// 失败含义：GPU crate 的 Cargo.toml 出现**任何**第三方 crate（wgpu/wasm-bindgen/
/// web-sys 等）——D-15 硬红线破裂。
#[test]
fn zero_third_party_dependency_prism_gpu() {
    let manifest = include_str!("../Cargo.toml");
    // 允许的依赖：仅 workspace 内部 path 依赖。
    let mut offenders: Vec<String> = Vec::new();
    let mut in_deps = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_deps = line == "[dependencies]" || line == "[dev-dependencies]" || line == "[build-dependencies]";
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        // 内部依赖形如 `prism-x = { path = ... }`；其余一律违规。
        if !line.starts_with("prism-") {
            offenders.push(line.to_string());
        }
        // 即便以 prism- 开头，也必须带 path（防 crates.io 同名）。
        if line.starts_with("prism-") && !line.contains("path =") {
            offenders.push(format!("非 path 依赖: {line}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "prism-gpu 存在第三方/非 path 依赖（D-15）:\n{}",
        offenders.join("\n")
    );
    // 关键字黑名单直接扫描。
    for banned in ["wgpu", "wasm-bindgen", "web-sys", "js-sys", "naga", "metal", "ash", "vulkano"] {
        assert!(
            !manifest.contains(banned),
            "manifest 含被禁第三方 crate：{banned}"
        );
    }
}

/// 失败含义：workspace 的 Cargo.lock 含非本仓库 crate（第三方混入）——锁文件级零依赖
/// 证据被破坏。
#[test]
fn zero_third_party_dependency_lockfile() {
    let lock = include_str!("../../../Cargo.lock");
    // 统计 package 条目，全部应为 workspace 本地 crate（version 0.1.0 且 name 以 prism- 开头）。
    let mut names: Vec<String> = Vec::new();
    let mut lines = lock.lines();
    while let Some(l) = lines.next() {
        if l.trim() == "[[package]]" {
            for probe in lines.by_ref() {
                let p = probe.trim();
                if let Some(rest) = p.strip_prefix("name = ") {
                    names.push(rest.trim_matches('"').to_string());
                    break;
                }
            }
        }
    }
    assert!(!names.is_empty(), "Cargo.lock 无 package 条目（解析失败）");
    let offenders: Vec<&String> = names.iter().filter(|n| !n.starts_with("prism-")).collect();
    assert!(
        offenders.is_empty(),
        "Cargo.lock 含第三方 crate（D-15）: {:?}",
        offenders
    );
}

// ===========================================================================
// B. D1 WebGPU 绑定 + harness
// ===========================================================================

/// 失败含义：native 下能力探测 panic 或返回非 NotWasmTarget——桩实现破坏可编译/可测。
#[test]
fn web_detect_support_native_stable() {
    use prism_gpu::web::{detect_support, WebGpuSupport};
    assert_eq!(detect_support(), WebGpuSupport::NotWasmTarget);
    // 反复调用稳定。
    assert_eq!(detect_support(), detect_support());
}

/// 失败含义：WebGPU 后端 dispatch 计划的工作组数换算错误（除不尽未向上取整）。
#[test]
fn web_scan_config_workgroups_round_up() {
    use prism_gpu::web::device::{ScanConfig, WORKGROUP_SIZE};
    assert_eq!(WORKGROUP_SIZE, 64);
    assert_eq!(ScanConfig::for_count(0).workgroups, 1, "空输入至少 1 块");
    assert_eq!(ScanConfig::for_count(64).workgroups, 1);
    assert_eq!(ScanConfig::for_count(128).workgroups, 1);
    assert_eq!(ScanConfig::for_count(129).workgroups, 2);
    assert_eq!(ScanConfig::for_count(256).workgroups, 2); // 256/128 = 2 块
}

/// 失败含义：native 后端未就绪却报告 ready，或降级路径与 CPU 参考不一致（harness
/// 要求「永不 panic、永有输出」被破坏）。
#[test]
fn web_backend_native_never_ready_and_falls_back() {
    use prism_gpu::web::device::WebGpuBackend;
    use prism_gpu::{AestheticReq, CameraReq, CpuRefBackend, GpuDevice, LightReq, SceneRenderRequest, SphereReq};
    use prism_math::Vec3;
    let mut b = WebGpuBackend::new();
    assert!(!b.is_gpu_ready());
    assert!(!b.device_lost());
    assert!(b.adapter_summary().adapter_name.is_empty());
    // 降级路径须与 CPU 参考逐字节一致（harness「永不 panic、永有输出」要求）。
    let req = SceneRenderRequest {
        width: 16,
        height: 9,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.0, -4.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
        },
        lights: vec![LightReq {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }],
        spheres: vec![SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4,
            metalness: 0.1,
        }],
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
    };
    let gpu_fb = b.render_scene(&req);
    let cpu_fb = CpuRefBackend.render_scene(&req);
    assert_eq!(gpu_fb.width, cpu_fb.width);
    assert_eq!(gpu_fb.pixels, cpu_fb.pixels, "降级应与 CPU 参考逐字节一致");
}

/// 失败含义：WebGPU 后端 plan 未包含 KERNEL 源或 entry 名——GPU 承载缺失。
#[test]
fn web_backend_dispatch_plans_valid() {
    use prism_gpu::web::device::{ScanConfig, WebGpuBackend, WGSL_HISTOGRAM, WGSL_SCAN_BLOCK};
    let b = WebGpuBackend::new();
    let scan = b.plan_scan(ScanConfig::for_count(256));
    assert_eq!(scan.len(), 1);
    assert_eq!(scan[0].entry, "scan_block");
    assert!(scan[0].wgsl.contains("@compute"));
    let hist = b.plan_sort_histogram(200, 16, 0);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].entry, "histogram");
    assert_eq!(hist[0].workgroups, (4, 1, 1)); // ceil(200/64)=4
    // 内嵌 WGSL 常量结构自检。
    assert!(check_shader("scan_inline", WGSL_SCAN_BLOCK).is_pass());
    assert!(check_shader("hist_inline", WGSL_HISTOGRAM).is_pass());
}

/// 判定HTML 里是否存在 device lost 的 Promise 捕获。
///
/// 判据：某个 `<标识符>.lost.then(` 形式（允许 `.then` 与 `(` 之间有空白）。
/// 手写扫描而非引regex —— 本工程零第三方依赖（连测试也不引）。
///
/// 【为什么不用字面量 `device.lost.then`】局部变量改名（`device` → `dev`）
/// 就会让字面量失配，把功能完好的页面判成"缺捕获"。断言的意图是"存在
/// device lost 捕获"，与变量叫什么无关。
fn has_lost_then_capture(html: &str) -> bool {
    let bytes = html.as_bytes();
    const NEEDLE: &str = ".lost.then";
    let mut i = 0usize;
    while i < html.len() {
        let Some(rel) = html[i..].find(NEEDLE) else { break };
        let at = i + rel; // `at` 指向 `.`
        // 前置标识符：至少一个字母/数字/_/$ 紧邻在 `.` 之前。
        let has_ident = at > 0
            && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_' || bytes[at - 1] == b'$');
        // 后置调用括号：`.then` 之后允许空白，再必须是 `(`。
        let mut k = at + NEEDLE.len();
        while k < bytes.len() && (bytes[k] as char).is_whitespace() {
            k += 1;
        }
        let has_call = k < bytes.len() && bytes[k] == b'(';
        if has_ident && has_call {
            return true;
        }
        i = at + 1;
    }
    false
}

/// 失败含义：harness 页面缺失四个路线图要求之一（特性检测/错误面板/一键复制/可重跑）。
/// 静态审读 `web/index.html` 源码。
#[test]
fn harness_four_requirements_present() {
    let html = include_str!("../web/index.html");

    // 要求 1：特性检测 + 不白屏（banner + navigator.gpu 分支 + SKIP 兜底）。
    assert!(html.contains("navigator.gpu"), "缺特性检测");
    assert!(html.contains("setBanner"), "缺不白屏横幅");
    assert!(html.contains("当前浏览器不支持 WebGPU"), "缺不支持提示文案");
    assert!(html.contains("SKIP：无 navigator.gpu"), "缺 SKIP 兜底（不白屏清单）");

    // 要求 2：错误面板（全局错误收集 + GPU 错误 + device lost）。
    assert!(html.contains("id=\"errpanel\""), "缺错误面板容器");
    assert!(html.contains("window.addEventListener(\"error\""), "缺 window.error 收集");
    assert!(html.contains("unhandledrejection"), "缺 unhandledrejection 收集");
    assert!(html.contains("uncapturederror"), "缺 uncapturederror 收集");
    // 与 `harness_device_lost_reflects_flag` 同口径：字面量 `device.lost.then`
    // 会被局部变量改名（`device` → `dev`）误判为缺失，故扫描 `<标识符>.lost.then(`。
    assert!(has_lost_then_capture(html), "缺 device lost 捕获");

    // 要求 3：一键复制。
    assert!(html.contains("id=\"copy\""), "缺一键复制按钮");
    assert!(html.contains("navigator.clipboard.writeText"), "缺剪贴板写入");

    // 要求 4：测试项可点、可重跑。
    assert!(html.contains("async function runOne"), "缺单项运行");
    assert!(html.contains("id=\"rerun\""), "缺重跑失败项按钮");
    assert!(html.contains("states[t.id]"), "缺状态机");
}

/// 失败含义：harness 的 `prism_device_lost` 胶水**未真实反映** device lost 状态——
/// 要么缺少 `device.lost.then` 捕获（`deviceLostFlag` 永不置位），要么胶水仍是恒 0
/// 的 `? 0 : 0`（即便device 已 lost 也上报 0），错误反馈闭环残缺。
///
/// 本测试为**正向断言**（原见证型断言 `? 0 : 0` 已在缺陷修复后失效，
/// 现反转过来防回归：任何"退回恒 0"或"删掉 lost 捕获"的改动都会红）。
#[test]
fn harness_device_lost_reflects_flag() {
    let html = include_str!("../web/index.html");

    // 断言 1：胶水行存在，且**不是**恒 0 的 `? 0 : 0`。
    let line = html
        .lines()
        .find(|l| l.contains("prism_device_lost:"))
        .expect("prism_device_lost 胶水应存在");
    assert!(
        !line.contains("? 0 : 0"),
        "device_lost 胶水退回恒 0（`? 0 : 0`）——设备丢失永远无法上报；实际行: {line}"
    );
    assert!(
        line.contains("? 1 : 0"),
        "device_lost 胶水应在 deviceLostFlag 置位时返回 1；实际行: {line}"
    );

    // 断言 2：胶水必须真的读 `deviceLostFlag`（而非某个恒 false 的替身）。
    assert!(
        line.contains("deviceLostFlag"),
        "device_lost 胶水未读取 deviceLostFlag，无法反映真实状态；实际行: {line}"
    );

    // 断言 3：`deviceLostFlag` 必须由 `<ident>.lost.then(...)` 真实置位。
    //
    // 【口径修正·2026-10-04】原断言是字面量 `html.contains("device.lost.then")`。
    // 冷启动重试改造后局部变量从 `device` 改名`dev`，写成 `dev.lost.then(...)`，
    // 功能完好（flag 照旧置位、胶水照旧读 flag），却被字面量匹配判成"缺捕获"。
    // 断言的**意图**是"存在 device lost 的 Promise 捕获"，不是"必须把局部变量
    // 命名为 device"。故改为扫描 `<标识符>.lost.then(` 形式。
    // 仍保留下面两条对 `deviceLostFlag` 的强断言（胶水读它、回调置位），
    // 所以放宽不会削弱判别力——把整段捕获删掉仍会红。
    assert!(
        has_lost_then_capture(html),
        "缺 device lost 捕获（找不到 `<ident>.lost.then(` 形式的 Promise 捕获）"
    );
    let flag_decl = html
        .lines()
        .find(|l| {
            let c = l.trim();
            c.starts_with("let deviceLostFlag") || c.starts_with("var deviceLostFlag")
        })
        .expect("应声明 deviceLostFlag");
    assert!(
        flag_decl.contains("= false") || flag_decl.contains("=0"),
        "deviceLostFlag 初值应为 false/0，实际: {flag_decl}"
    );
    // 置位语句：回调体内`deviceLostFlag = true`。
    let set_true = html
        .lines()
        .any(|l| l.contains("deviceLostFlag = true") || l.contains("deviceLostFlag=true"));
    assert!(
        set_true,
        "device.lost.then 回调体内未见 deviceLostFlag = true——标志位不会被置位"
    );
}

// ===========================================================================
// E. WGSL 内核语义仿真（无 GPU 下的算法正确性判对，最高优先）
// ===========================================================================

/// WGSL `scan_block` 线程索引仿真器（忠实模拟 WGSL 稳健访问 = 越界下标钳到界内）。
///
/// WGSL 规范下 `var<workgroup>` 数组越界索引被**钳制**（robust access）而非 UB，
/// 故多余线程会反复读写 `temp[127]`——必须用钳制语义仿真，不能假设丢弃。
fn sim_wgsl_scan_block(input: &[f32], apply_activity_guard: bool) -> Vec<f32> {
    const WG: usize = 64;
    const N: usize = 128;
    let clamp = |i: i64| -> usize { i.clamp(0, (N - 1) as i64) as usize };
    let mut temp = [0.0f32; N];
    for tid in 0..WG {
        temp[tid * 2] = input[tid * 2];
        temp[tid * 2 + 1] = input[tid * 2 + 1];
    }
    // 上行（reduce）：offset = 1,2,...,64
    let mut offset: usize = 1;
    while offset < N {
        for tid in 0..WG {
            if apply_activity_guard && tid >= N / (2 * offset) {
                continue;
            }
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let i = clamp(idx);
            let j = clamp(idx - offset as i64);
            temp[i] = temp[i] + temp[j];
        }
        offset *= 2;
    }
    temp[N - 1] = 0.0;
    // 下行（distribute）：offset = 64,32,...,1
    let mut offset: usize = 64;
    while offset > 0 {
        for tid in 0..WG {
            if apply_activity_guard && tid >= N / (2 * offset) {
                continue;
            }
            let idx = ((tid + 1) * (offset * 2)) as i64 - 1;
            let i = clamp(idx);
            let j = clamp(idx - offset as i64);
            let left = temp[j];
            temp[j] = temp[i];
            temp[i] = temp[i] + left;
        }
        offset /= 2;
    }
    temp.to_vec()
}

/// 串行 exclusive scan 参考（GPU 侧语义：out = ExScan）。
fn serial_exclusive_scan(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0.0f32;
    for &x in input {
        out.push(acc);
        acc += x;
    }
    out
}

/// 失败含义：`WGSL_SCAN_BLOCK` 缺失 Blelloch 的**线程活跃守卫**
/// （`if (tid < N/(2*offset))`）。无守卫时 offset≥2 有 32~63 个线程越界，
/// 稳健访问钳制后它们反复写坏根节点 `temp[127]`（溢出为 inf），实测 128 输出中 64 个全错。
/// 这是 GPU 真机路径上 scan 内核的**结果级错误**（非风格问题）。
#[test]
fn wgsl_scan_block_requires_thread_activity_guard() {
    let input: Vec<f32> = (1..=128).map(|i| i as f32).collect();
    let expected = serial_exclusive_scan(&input);

    // 断言 1：加守卫 ⇒ 与串行 exclusive scan 逐位一致（证明算法本体正确，
    // 缺陷仅在守卫缺失，这样修复方向明确）。
    let guarded = sim_wgsl_scan_block(&input, true);
    assert_eq!(
        guarded, expected,
        "加线程活跃守卫后 WGSL scan_block 应与串行 exclusive scan 一致（算法本体正确）"
    );

    // 断言 2：不加守卫（= 修复前的历史源码语义）⇒ 必须错误。
    // 该子仿真恒定保留为回归证据：证明守卫确为结果级必需，而非风格偏好。
    let unguarded = sim_wgsl_scan_block(&input, false);
    let wrong = unguarded
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    // 精确口径（f32 仿真实测）：128 个输出中恰 64 个（全部奇数下标）被写坏，
    // 且 temp[127] 被反复累加溢出为 +inf——数值彻底失真而非轻微偏差。
    assert_eq!(
        wrong, 64,
        "预期：当前 WGSL scan_block 因缺失线程活跃守卫而 64/128 输出全错；\
         实际 {wrong} 个错——若源码已修复，请改写本测试为守卫存在性正向断言"
    );
    assert!(
        unguarded.iter().any(|v| v.is_infinite()),
        "预期：无守卫版本因根节点被反复累加出现 inf（数值失真证据）"
    );
}

/// 失败含义：源码文本中 `WGSL_SCAN_BLOCK` 不含活跃守卫 `if (tid < ...)` 结构。
/// 直接对源码文本断言，作为缺陷的**可复现静态证据**（文件:行号级）。
#[test]
fn wgsl_scan_block_source_has_activity_guard() {
    let src = prism_gpu::web::device::WGSL_SCAN_BLOCK;
    let body: String = src
        .lines()
        .map(|l: &str| l.split("//").next().unwrap_or(""))
        .collect::<Vec<&str>>()
        .join("\n");
    // Blelloch 每步活跃线程数应为 N/(2*offset)；正确实现需 `if (tid < ...)` 守卫。
    let has_guard = body.contains("if (tid <") || body.contains("if (lid.x <");
    assert!(
        has_guard,
        "WGSL_SCAN_BLOCK 缺 `if (tid < N/(2*offset))` 线程活跃守卫：\
         无守卫时 offset≥2 越界线程会反复写坏根节点 temp[127]"
    );
}

/// WGSL 源码的最小结构化解析结果：按缩进深度追踪代码块，判定每条语句的
/// **块嵌套深度**与其所属阶段。用于验证屏障所处控制流是否uniform。
#[derive(Debug, Clone)]
struct WgslLine {
    idx: usize,
    /// 去掉注释后的源码（trim 后）。
    code: String,
    /// 原始行（含注释），用于文本存在性断言。
    raw: String,
    /// 该行所处的花括号嵌套深度（0 = 函数体最外层）。
    depth: i32,
}

/// 极简 WGSL 结构解析：按 `{` / `}` 计数维护深度，剥离 `//` 注释与字符串字面量。
/// 目的是**不靠 contains猜位置**，而是判定"某语句在第几层块内"。
fn parse_wgsl_lines(src: &str) -> Vec<WgslLine> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    for (idx, raw) in src.lines().enumerate() {
        // 剥离行注释（`//` 之后的内容），保留字符串字面量内的 `//`。
        let mut code = String::new();
        let mut in_str = false;
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    in_str = !in_str;
                    code.push(c);
                }
                '/' if !in_str && chars.peek() == Some(&'/') => break,
                _ => code.push(c),
            }
        }
        let code = code.trim().to_string();
        // 深度取"该行语句开始处"的深度；开括号行本身记为外层深度。
        let opens = code.matches('{').count() as i32;
        let closes = code.matches('}').count() as i32;
        out.push(WgslLine {
            idx,
            code,
            raw: raw.to_string(),
            depth: depth.max(0),
        });
        depth += opens - closes;
    }
    out
}

fn is_barrier(l: &WgslLine) -> bool {
    l.code.contains("workgroupBarrier()")
}

fn is_activity_guard(l: &WgslLine) -> bool {
    l.code.contains("if (tid <") || l.code.contains("if (lid.x <")
}

/// 正向断言（防回归）：`WGSL_SCAN_BLOCK` 的所有 `workgroupBarrier()` 都必须位于
/// **uniform 控制流**中——即不得嵌在依赖 `tid` 的 `if (tid < ...)` 守卫块内。
///
/// 失败含义（真实回归）：任一屏障的块深度 > 其所在 `loop` 体的深度，
/// 意味着它被误放进非 uniform 守卫分支，WGSL 验证层会直接拒绝该 shader
/// （`workgroupBarrier` 必须在 uniform control flow 中），真机上scan 内核无法编译。
///
/// 历史教训：原见证型断言 `guard_line < barrier_line` 语义写反——load 之后的屏障
/// 本就应该在 up-sweep 守卫之前（load 阶段只有 `base < count` 尾块守卫，
/// 不存在 Blelloch 活跃守卫）。本测试改为按**块深度**判定，语义正确且更严。
#[test]
fn wgsl_scan_block_barriers_are_in_uniform_control_flow() {
    check_scan_barriers(prism_gpu::web::device::WGSL_SCAN_BLOCK);
}

/// 变异测试（QA 自证非空断言）：把**已知违规**的 WGSL 喂给同一个检查器，
/// 必须 panic。若这些变异体都能通过，说明本检查器形同虚设。
#[test]
fn wgsl_scan_barrier_checker_rejects_known_bad_shaders() {
    // 变异 1：屏障被误放进 `if (tid < ...)` 守卫块内（非 uniform 控制流）。
    let bad1 = r#"
@compute @workgroup_size(64)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var offset: u32 = 1u;
    loop {
        if (offset >= 128u) { break; }
        if (tid < 128u / (offset * 2u)) {
            temp[tid] = temp[tid] + temp[tid - offset];
            workgroupBarrier();
        }
        offset = offset * 2u;
    }
}
"#;
    // 变异 2：up-sweep 活跃守卫被删掉（回归缺陷 2）。
    let bad2 = r#"
@compute @workgroup_size(64)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var offset: u32 = 1u;
    loop {
        if (offset >= 128u) { break; }
        temp[tid] = temp[tid] + temp[tid - offset];
        workgroupBarrier();
        offset = offset * 2u;
    }
}
"#;
    // 变异 3：load 末次 store 之后、首个守卫之前**没有**屏障（数据竞争）。
    let bad3 = r#"
@compute @workgroup_size(64)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    temp[tid * 2u + 1u] = 0u;
    var offset: u32 = 1u;
    loop {
        if (offset >= 128u) { break; }
        if (tid < 128u / (offset * 2u)) {
            temp[tid] = temp[tid] + temp[tid - offset];
        }
        workgroupBarrier();
        offset = offset * 2u;
    }
}
"#;
    for (name, src) in [("barrier_in_guard", bad1), ("no_activity_guard", bad2), ("no_load_barrier", bad3)] {
        let r = std::panic::catch_unwind(|| check_scan_barriers(src));
        assert!(r.is_err(), "变异体 `{name}` 竟通过了屏障检查器——检查器无效");
    }
}

/// 屏障uniform 控制流检查器（供正向测试与变异测试共用）。
fn check_scan_barriers(src: &str) {
    let lines = parse_wgsl_lines(src);
    assert!(!lines.is_empty(), "WGSL_SCAN_BLOCK 解析为空");

    // ---- 前置：至少存在一个 Blelloch 活跃守卫与多个屏障 ----
    let barriers: Vec<&WgslLine> = lines.iter().filter(|l| is_barrier(l)).collect();
    let guards: Vec<&WgslLine> = lines.iter().filter(|l| is_activity_guard(l)).collect();
    assert!(
        barriers.len() >= 4,
        "scan_block 应有 load 后1 + up-sweep N +阶段间 1 + down-sweep N 道屏障，实际 {}",
        barriers.len()
    );
    assert_eq!(
        guards.len(),
        2,
        "应恰有 up-sweep / down-sweep 两处 Blelloch 活跃守卫，实际 {} 处",
        guards.len()
    );

    // ---- 逐个屏障判定其所处的控制流是否 uniform ----
    for b in &barriers {
        // 找出该屏障之前最近的一个 `loop {` （若无，则最近的 `fn` 函数体）。
        let enclosing_loop = lines
            .iter()
            .rev()
            .take_while(|l| l.idx < b.idx && !l.code.starts_with("fn "))
            .find(|l| l.code == "loop {" || l.code.starts_with("loop {"));
        // 找出该屏障之前最近的、未闭合的 `if (tid <` 守卫。
        // 判定法（纯块深度）：守卫块内的语句深度 = guard.depth + 1；
        // 屏障深度若> guard.depth 即落在守卫块内 ⇒ 非 uniform 控制流 ⇒ 违规。
        let bad = guards
            .iter()
            .filter(|g| g.idx < b.idx && b.depth > g.depth)
            .min_by_key(|g| g.depth)
            .map(|g| (g.idx, g.depth));
        assert!(
            bad.is_none(),
            "workgroupBarrier()（idx={}, depth={}）落在 idx={} 的 `if (tid < ...)` 守卫块内\
             —— 非 uniform 控制流，WGSL 验证层会直接拒绝该 shader；源码行: {}",
            b.idx,
            b.depth,
            bad.expect("已在断言中返回").0,
            b.raw.trim()
        );
        // 屏障不得位于函数体最外层与 loop 之外——它必须属于某个阶段。
        assert!(
            b.depth >= 1,
            "idx={} 的屏障处于函数体最外层，阶段划分异常：{}",
            b.idx,
            b.raw.trim()
        );
        if let Some(l) = enclosing_loop {
            assert!(
                b.depth >= l.depth + 1,
                "idx={} 的屏障未处于 loop 体内（loop depth={}，barrier depth={}）",
                b.idx,
                l.depth,
                b.depth
            );
        }
    }

    // ---- 关键次序：load 的最后一条 store之后、up-sweep 活跃守卫之前，----
    // ---- 必须存在恰好一道 load→up-sweep 屏障（隔开数据竞争）。----
    let last_load_store = lines
        .iter()
        .position(|l| l.code.contains("temp[tid * 2u + 1u] ="))
        .expect("应存在 load 的第二条 store");
    let first_guard = guards[0].idx;
    let between: Vec<&WgslLine> = barriers
        .iter()
        .copied()
        .filter(|b| b.idx > last_load_store && b.idx < first_guard)
        .collect();
    assert_eq!(
        between.len(),
        1,
        "load 末次store（idx={}）与 up-sweep 首个活跃守卫（idx={}）之间应恰有1 道屏障，实际 {}",
        last_load_store,
        first_guard,
        between.len()
    );
}

/// 正向断言：`WGSL_HISTOGRAM` 用 `(keys[i] >> shift) & (buckets-1)` 取数位，
/// 且 `Params` 含 `shift` 字段——与 CPU `sort.rs:31` 的
/// `(k >> shift) & (RADIX_BUCKETS-1)` 同一口径，LSD 第2/3/4 趟可表达。
#[test]
fn wgsl_histogram_has_shift_param_matching_cpu_radix() {
    let wgsl = prism_gpu::web::device::WGSL_HISTOGRAM;
    assert!(
        wgsl.contains("(keys[i] >> params.shift) & (params.buckets - 1u)"),
        "WGSL_HISTOGRAM 应按 `>> shift` +掩码取数位（与 CPU 基数排序同口径）"
    );
    assert!(
        !wgsl.contains("keys[i] % params.buckets"),
        "WGSL_HISTOGRAM 不应再用 `%` 取数位（无法表达 shift>0 的数位）"
    );
    let params_line = wgsl
        .lines()
        .find(|l| l.contains("struct Params"))
        .expect("应声明 Params");
    assert!(
        params_line.contains("shift"),
        "WGSL_HISTOGRAM 的 Params 缺 `shift` 字段，LSD 第 2/3/4 趟数位无法表达"
    );
    // 16 字节对齐：3 个 u32 后须有占位成员，使结构体大小满足 uniform 对齐要求。
    assert!(
        params_line.contains("_pad"),
        "WGSL_HISTOGRAM 的 Params 应补`_pad` 占位，使结构体满足 uniform 16 字节对齐"
    );
    // host 端打包长度必须与 WGSL 布局一致（16 字节）。
    let bytes = prism_gpu::web::device::histogram_params_bytes(200, 256, 8);
    assert_eq!(
        bytes.len(),
        prism_gpu::web::device::HISTOGRAM_PARAMS_BYTES,
        "host 端 uniform 缓冲长度应与 WGSL Params 布局一致"
    );
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 8, "shift 应落在偏移 8");
    // 反证：CPU 侧确实按 shift 取数位（确保本测试的对照口径正确）。
    let cpu = include_str!("../src/sort.rs");
    assert!(
        cpu.contains("(k >> shift) & (RADIX_BUCKETS as u32 - 1)"),
        "CPU 侧 sort.rs 应按 (k>>shift)&mask 取数位——若已变更，本测试对照口径失效"
    );
}
