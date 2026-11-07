//! V3.0 W2 出口判据（DSL 侧）：39 行接线全覆盖、同源对拍、错误注入集、
//! L3 时序两帧值、两跑逐字节 / 换序反例、零损失 md5 对账。
//!
//! # 反假绿双证中的「运行时」一侧
//!
//! `coverage_gate.rs` 是静态门禁（遍历规格表 39 行）；本文件证明这些名字
//! **真的进求值链**：
//! 1. `cap "编号" { 参数全集 }` 经 LL(1) 前端 → `WiringPlan`，每行键集合
//!    与 `CAPABILITY_ROWS` 的 `api_params` 逐项相等；
//! 2. 「直接调用引擎 API 构造同一场景」一路用公开类型手工 new 出同一份
//!    `WiringPlan`，与 DSL 编译产物的 `canonical_bytes()` 逐字节相同
//!    （不是语言自己算一遍比自己：两路入口不同，值不经 DSL 前端搬运）；
//! 3. 删任意一行接线 / 拆任意一个字段映射 ⇒ 覆盖率下降或 P8001 判红；
//! 4. 六类核心项的 cap 参数声明成解析式通道，两帧取到不同值，且逐帧值
//!    与既有 `prism_anim::Expr::sample` 逐位相等；
//! 5. 同源码两跑逐字节、换序可检出、三个黄金 `.prism` md5 不变。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::PropertyChannel;
use prism_dsl::{
    compile_v2, example_cap_block, full_declaration_blocks, param_kind, range_rule,
    resolve_frame, total_param_count, CapFamily, ParamKind, WiringEntry,
    WiringFrame, WiringParam, WiringPlan, WiringValue, CAPABILITY_ROWS,
};

const FPS: u32 = 8;

/// 一份最小合法 v2 文档，把 `caps` 插入顶层。
fn doc(caps: &str) -> String {
    format!(
        "// prism-dsl 1\nmovie \"w2\" {{ duration 1s fps {FPS} resolution 64x36 }}\n{caps}\nscene \"s\" {{ duration 1s }}\n"
    )
}

/// 33 个非平台域行的参数全集总数（规格表分母 226 − 平台域 28）。
fn non_platform_param_count() -> usize {
    CAPABILITY_ROWS
        .iter()
        .filter(|r| r.family != CapFamily::Platform)
        .map(|r| r.api_params.len())
        .sum()
}

/// 「直接调用引擎 API 构造同一场景」：不经过 DSL 文本，用公开类型逐项 new
/// 出与 `example_cap_block` 语义相同的 `WiringPlan`。
fn direct_plan() -> WiringPlan {
    let mut entries: Vec<WiringEntry> = Vec::new();
    for row in CAPABILITY_ROWS {
        if row.family == CapFamily::Platform {
            continue;
        }
        let mut params: BTreeMap<String, WiringParam> = BTreeMap::new();
        for (i, p) in row.api_params.iter().enumerate() {
            let value: WiringValue = match param_kind(row, p) {
                ParamKind::Bool => WiringValue::Bool(false),
                ParamKind::Int => {
                    let n: i64 = range_rule(row.id, p)
                        .map(|r| (r.lo.ceil() as i64).max(1))
                        .unwrap_or(1);
                    WiringValue::Int(n)
                }
                ParamKind::Num => {
                    // 复现 DSL 路径：f64 中点 → 十进制字符串 → f32 位型。
                    let text: String = range_rule(row.id, p)
                        .map(|r| format!("{}", (r.lo + r.hi) / 2.0))
                        .unwrap_or_else(|| "0.5".to_string());
                    let n: f32 = text.parse().expect("样例数值应可解析为 f32");
                    WiringValue::Num(n.to_bits())
                }
                ParamKind::Text => WiringValue::Text(format!(
                    "w2-{}-{}",
                    row.id.to_lowercase(),
                    p.to_lowercase()
                )),
                ParamKind::Variant => WiringValue::Int(if i == 0 { 1 } else { 0 }),
            };
            params.insert((*p).to_string(), WiringParam::Const(value));
        }
        entries.push(WiringEntry {
            row_id: row.id.to_string(),
            line: 0,
            scene: None,
            params,
        });
    }
    WiringPlan { entries }
}

/// 编译一份完整声明文档（39 行中的 33 个非平台域行）。
fn full_compile() -> prism_dsl::CompileV2Output {
    compile_v2(&doc(&full_declaration_blocks()))
        .unwrap_or_else(|e| panic!("33 行完整声明应编译成功，实际：第 {} 行 {}", e.line, e.message))
}

/// 把某行的某个字段替换成解析式子块（L3 时序绑定）。
fn with_expr(row: &str, param: &str, expr: &str) -> String {
    let block: String = example_cap_block(row).expect("样例块应可生成");
    let mut out: String = String::new();
    let mut replaced: bool = false;
    for line in block.lines() {
        if line.trim_start().starts_with(&format!("{param} ")) {
            out.push_str(&format!("    {param} {{ expr \"{expr}\" }}\n"));
            replaced = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(replaced, "{row} 应含字段 {param}");
    out
}

/// 编译产物里的通道表（`resolve_frame` 的求值上下文）。
fn channel_map(
    out: &prism_dsl::CompileV2Output,
) -> BTreeMap<&str, &PropertyChannel> {
    out.channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect()
}

/// 断言一次编译失败，且带行号 + 指定 P8xxx 码 + 三段式建议段。
fn expect_red(src: &str, line: usize, code: &str) {
    match compile_v2(src) {
        Ok(_) => panic!("以下源码应当判红，却编译成功：\n{src}"),
        Err(e) => {
            assert_eq!(e.line, line, "错误行号不符：{e}");
            assert!(
                e.message.contains(&format!("[{code}]")),
                "应带 {code} 码：{e}"
            );
            assert!(
                e.message.contains(&format!("第 {line} 行")),
                "错误正文应带行号：{e}"
            );
            assert!(
                e.message.contains("\n  建议："),
                "三段式建议段缺失：{e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 1. 39 行接线全覆盖 / 参数全集
// ---------------------------------------------------------------------------

#[test]
fn w2_runtime_table_has_39_rows_and_226_params() {
    assert_eq!(CAPABILITY_ROWS.len(), 39, "39 行规格表不得增删");
    assert_eq!(total_param_count(), 226, "参数全集分母应随 E-8 扩容为 226");
    let ids: BTreeSet<&str> = CAPABILITY_ROWS.iter().map(|r| r.id).collect();
    assert_eq!(ids.len(), 39, "编号不得重复");
    assert_eq!(non_platform_param_count(), 198, "33 个非平台域行的参数总数");
}

#[test]
fn w2_full_declaration_covers_all_33_rows_with_exact_param_sets() {
    let out = full_compile();
    assert_eq!(out.wiring.entries.len(), 33, "39 行中 33 行非平台域全部接线");
    assert_eq!(
        out.wiring.param_count(),
        non_platform_param_count(),
        "每行参数数与 api_params 全集一致"
    );
    assert_eq!(out.wiring.wired_rows(), 33);
    for row in CAPABILITY_ROWS.iter().filter(|r| r.family != CapFamily::Platform) {
        let e = out
            .wiring
            .entry(row.id)
            .unwrap_or_else(|| panic!("{} 未进 WiringPlan", row.id));
        let dsl: BTreeSet<&str> = e.params.keys().map(|s| s.as_str()).collect();
        let api: BTreeSet<&str> = row.api_params.iter().copied().collect();
        assert_eq!(dsl, api, "{} 的 dsl_fields 与 api_params 差集非 0", row.id);
    }
}

// ---------------------------------------------------------------------------
// 2. 同源对拍：DSL 声明 vs 直接调用引擎 API 构造同一场景
// ---------------------------------------------------------------------------

#[test]
fn w2_dsl_plan_matches_direct_api_built_plan_byte_for_byte() {
    let dsl_plan: WiringPlan = full_compile().wiring;
    let api_plan: WiringPlan = direct_plan();
    assert_eq!(
        dsl_plan.canonical_bytes(),
        api_plan.canonical_bytes(),
        "DSL 前端与直接构造的同一场景接线产物必须逐字节相同"
    );
    assert_eq!(dsl_plan.entries.len(), api_plan.entries.len());
    // 直接构造一路也必须覆盖 198 个参数，不是空壳对拍。
    assert_eq!(api_plan.param_count(), 198);
}

#[test]
fn w2_resolve_frame_matches_property_channel_sample_bit_for_bit() {
    // 六类核心项各取一个 L3 行参数，声明成解析式通道。
    // 相机 / 图元族用 G-1（其 dsl_fields 含 stage3d.camera.orbit.*）。
    let cases: &[(&str, &str, &str)] = &[
        ("G-1", "revolutions", "t * 0.5 + 1"),
        ("R-2", "roughness", "t * 0.1 + 0.2"),
        ("R-1", "clamp_indirect", "t * 0.5 + 1"),
        ("G-2", "displacement", "t * 0.2 + 0.3"),
        ("E-4", "frame", "t * 2 + 1"),
        ("R-13", "n0", "t * 4 + 8"),
    ];
    let blocks: String = cases
        .iter()
        .map(|(row, param, expr)| with_expr(row, param, expr))
        .collect::<Vec<String>>()
        .join("\n");
    let out = compile_v2(&doc(&blocks))
        .unwrap_or_else(|e| panic!("L3 时序文档应编译成功：第 {} 行 {}", e.line, e.message));
    let channels = channel_map(&out);
    let f0: WiringFrame = resolve_frame(&out.wiring, &channels, 0.0, 0, 0, 0);
    let f4: WiringFrame = resolve_frame(&out.wiring, &channels, 4.0 / FPS as f32, 4, 0, 0);

    for (row, param, expr) in cases {
        let key: String = format!("{row}.{param}");
        let v0: f32 = f0.value(&key).unwrap_or_else(|| panic!("{key} 未在第 0 帧解算"));
        let v4: f32 = f4.value(&key).unwrap_or_else(|| panic!("{key} 未在第 4 帧解算"));
        assert_ne!(v0.to_bits(), v4.to_bits(), "{key} 两帧必须取到不同参数值");
        let analytic = prism_anim::Expr::parse(expr)
            .unwrap_or_else(|e| panic!("解析式 {expr} 应合法：{e:?}"));
        assert_eq!(
            v0.to_bits(),
            analytic.sample(0.0).to_bits(),
            "{key} 第 0 帧应与解析式对拍一致"
        );
        assert_eq!(
            v4.to_bits(),
            analytic.sample(4.0 / FPS as f32).to_bits(),
            "{key} 第 4 帧应与解析式对拍一致"
        );
        // 既有 PropertyChannel::sample 与解析式同值（统一通道的同源入口）。
        let ch: &PropertyChannel = out
            .channels
            .iter()
            .find(|c| c.name == format!("w2/{row}/0/{param}"))
            .map(|c| &c.channel)
            .unwrap_or_else(|| panic!("{key} 的时序通道未登记"));
        assert_eq!(v0.to_bits(), ch.sample(0.0).to_bits(), "{key} 通道采样应对拍");
    }
}

// ---------------------------------------------------------------------------
// 3. 错误注入集（红线 3：不静默降级、不取默认）
// ---------------------------------------------------------------------------

#[test]
fn w2_negative_controls_all_fail_with_line_code_and_three_part_message() {
    // P8001 漏必填（列出缺项，不补缺省）
    expect_red(
        &doc("cap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 }"),
        3,
        "P8001",
    );
    // P8002 未知字段
    expect_red(
        &doc("cap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 nee_only false bogus 1 }"),
        3,
        "P8002",
    );
    // P8003 类型错：布尔写数字 / 整数写小数 / 文本写数字
    expect_red(
        &doc("cap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 nee_only 1 }"),
        3,
        "P8003",
    );
    expect_red(
        &doc("cap \"R-1\" { spp 1.5 max_bounces 1 clamp_indirect 0.5 nee_only false }"),
        3,
        "P8003",
    );
    expect_red(
        &doc("cap \"R-3\" { center 1 radius 1 ior 1.5 }"),
        3,
        "P8003",
    );
    // P8004 超范围（不钳制、不取默认）
    expect_red(
        &doc("cap \"R-1\" { spp 99999 max_bounces 1 clamp_indirect 0.5 nee_only false }"),
        3,
        "P8004",
    );
    expect_red(
        &doc("cap \"R-2\" { base_color 0.5 metalness 1.5 roughness 0.5 ior 1.5 transmission 0 subsurface 0 subsurface_radius 0 alpha_t 0 alpha_b 0 rotation 0 maps \"m\" }"),
        3,
        "P8004",
    );
    // P8005 未知能力编号
    expect_red(&doc("cap \"X-9\" { a 1 }"), 3, "P8005");
    // P8006 重复声明（第二个块在第 4 行）
    let dup: &str = "// prism-dsl 1\nmovie \"x\" { duration 1s fps 8 }\ncap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 nee_only false }\ncap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 nee_only false }\nscene \"s\" { duration 1s }\n";
    expect_red(dup, 4, "P8006");
    // P8007 平台域（G5 正式裁定行）被声明
    expect_red(&doc("cap \"P-1\" { }"), 3, "P8007");
    // P8008 时序绑定非法：解析式非法字符 / 子块缺 expr / 值形态非法
    expect_red(
        &doc("cap \"R-1\" { spp { expr \"t & 2\" } max_bounces 1 clamp_indirect 0.5 nee_only false }"),
        3,
        "P8008",
    );
    expect_red(
        &doc("cap \"R-1\" { spp { bogus 1 } max_bounces 1 clamp_indirect 0.5 nee_only false }"),
        3,
        "P8008",
    );
    expect_red(
        &doc("cap \"R-1\" { spp 1 max_bounces 1 clamp_indirect 0.5 nee_only \"false\" }"),
        3,
        "P8003",
    );
    // P8009 枚举变体行未恰选中一个（选了两个）
    expect_red(
        &doc("cap \"G-3\" { Cylinder 1 Cone 1 Torus 0 Revolution 0 Strand 0 TriSoup 0 }"),
        3,
        "P8009",
    );
    // 未知节点 / 非法值在场景内 cap 块同样判红（不是只在顶层生效）
    let scene_cap: &str = "// prism-dsl 1\nmovie \"x\" { duration 1s fps 8 }\nscene \"s\" { duration 1s cap \"R-1\" { spp 1 } }\n";
    expect_red(scene_cap, 3, "P8001");
}

// ---------------------------------------------------------------------------
// 4. 删任意一行接线 / 拆任意一个字段映射 ⇒ 判红（运行时双证）
// ---------------------------------------------------------------------------

#[test]
fn w2_delete_any_row_lowers_runtime_coverage_for_all_33_rows() {
    let full: WiringPlan = full_compile().wiring;
    let baseline: usize = full.param_count();
    for row in CAPABILITY_ROWS.iter().filter(|r| r.family != CapFamily::Platform) {
        let blocks: String = CAPABILITY_ROWS
            .iter()
            .filter(|r| r.family != CapFamily::Platform && r.id != row.id)
            .filter_map(|r| example_cap_block(r.id).ok())
            .collect::<Vec<String>>()
            .join("\n\n");
        let reduced = compile_v2(&doc(&blocks))
            .unwrap_or_else(|e| panic!("删掉 {} 后其余行仍应可编译：{}", row.id, e));
        assert!(
            reduced.wiring.entry(row.id).is_none(),
            "{} 应确实被删掉",
            row.id
        );
        assert_eq!(reduced.wiring.entries.len(), 32, "删 {} 后应剩 32 行", row.id);
        assert_eq!(
            reduced.wiring.param_count(),
            baseline - row.api_params.len(),
            "删 {} 后参数覆盖率必须按全集下降",
            row.id
        );
        assert_ne!(
            reduced.wiring.canonical_bytes(),
            full.canonical_bytes(),
            "删 {} 后接线凭证必须变化（同源对拍可检出）",
            row.id
        );
    }
}

#[test]
fn w2_drop_any_single_field_mapping_is_red_for_all_33_rows() {
    for row in CAPABILITY_ROWS.iter().filter(|r| r.family != CapFamily::Platform) {
        for param in row.api_params {
            let block: String = example_cap_block(row.id).expect("样例块应可生成");
            let dropped: String = block
                .lines()
                .filter(|l| !l.trim_start().starts_with(&format!("{param} ")))
                .collect::<Vec<&str>>()
                .join("\n");
            match compile_v2(&doc(&dropped)) {
                Ok(_) => panic!("{} 删掉 {param} 后应判红，却编译成功", row.id),
                Err(e) => {
                    assert_eq!(e.line, 3, "{} 删 {param} 的报错行号", row.id);
                    assert!(
                        e.message.contains("[P8001]") || e.message.contains("[P8009]"),
                        "{} 删 {param} 应以漏必填 / 枚举非法判红，实际：{}",
                        row.id,
                        e.message
                    );
                    assert!(e.message.contains("\n  建议："), "三段式建议段缺失");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. 确定性：两跑逐字节 / 换序可检出 / 零损失 md5
// ---------------------------------------------------------------------------

#[test]
fn w2_same_source_two_runs_byte_identical_and_reorder_is_detected() {
    let a = full_compile();
    let b = full_compile();
    assert_eq!(a.wiring.canonical_bytes(), b.wiring.canonical_bytes());
    assert_eq!(
        format!("{:?}", a.wiring),
        format!("{:?}", b.wiring),
        "同源码两跑的接线表逐字节一致"
    );
    // 换序反例：把前两个 cap 块交换，声明序进 canonical_bytes ⇒ 可检出。
    let mut blocks: Vec<String> = CAPABILITY_ROWS
        .iter()
        .filter(|r| r.family != CapFamily::Platform)
        .filter_map(|r| example_cap_block(r.id).ok())
        .collect();
    blocks.swap(0, 1);
    let swapped = compile_v2(&doc(&blocks.join("\n\n")))
        .unwrap_or_else(|e| panic!("换序后仍应编译成功：{}", e));
    assert_ne!(
        swapped.wiring.canonical_bytes(),
        a.wiring.canonical_bytes(),
        "声明换序必须可检出（W1 判据不许退化）"
    );
    assert_eq!(swapped.wiring.param_count(), a.wiring.param_count());
}

#[test]
fn w2_zero_loss_three_golden_prism_are_unchanged_and_have_empty_wiring() {
    assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    // 三个黄金文件都无版本头：spin_red_ball 是 v1 语法，另两个是 v2 语法。
    // 入口按文件语法分派，与 W1 `v3_language.rs` 的零损失契约一致。
    let cases: &[(&[u8], &str, bool)] = &[
        (
            include_bytes!("../../../examples/spin_red_ball.prism"),
            "e75b03e4bc7fcc090561eb1163f8ad24",
            true,
        ),
        (
            include_bytes!("../../../examples/t0_media_mesh.prism"),
            "46e29136de352ea75341823f2b69bf18",
            false,
        ),
        (
            include_bytes!("../../../examples/w2d_showcase.prism"),
            "88f735505eab09c6763b644197eee51b",
            false,
        ),
    ];
    for (bytes, want, is_v1) in cases {
        assert_eq!(&md5::hex(bytes), want, "黄金 .prism 文件内容发生变化");
        let src: &str = std::str::from_utf8(bytes).expect("黄金文件应为 UTF-8");
        if *is_v1 {
            let out = prism_dsl::compile(src).expect("v1 黄金文件应编译成功");
            assert_eq!(out.width, 480, "v1 黄金文件分辨率零损失");
        } else {
            let out = compile_v2(src).expect("v2 黄金文件应编译成功");
            assert!(out.wiring.is_empty(), "旧文件不得产生接线声明");
            assert_eq!(out.wiring.canonical_bytes(), 0_u64.to_le_bytes().to_vec());
            // compile_auto 与 compile_v2 对无版本头 v2 文件逐字等价（零损失契约）。
            let auto = prism_dsl::compile_auto(src)
                .expect("无版本头 v2 文件应走 V2 分派")
                .into_v2_output();
            assert_eq!(
                format!("{:?}", out),
                format!("{:?}", auto),
                "compile_auto 与 compile_v2 产物必须逐字节等价"
            );
        }
    }
    // 三个旧文件不声明 cap ⇒ evaluate 返回空凭证（app 层断言，此处先给
    // compile 侧的等价读数：WiringPlan 为空且规范字节序列确定）。
    let empty = WiringPlan::default();
    assert!(empty.is_empty());
    assert_eq!(empty.canonical_bytes(), 0_u64.to_le_bytes().to_vec(), "空接线表的规范字节序列是 8 字节 0");
}

/// 精简 RFC 1321 MD5（纯整数、零第三方依赖），用于黄金文件逐字节对账。
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
        20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
        16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
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

    pub fn hex(data: &[u8]) -> String {
        let mut msg: Vec<u8> = data.to_vec();
        let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_le_bytes());
        let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        for chunk in msg.chunks_exact(64) {
            let mut m: [u32; 16] = [0; 16];
            for (i, slot) in m.iter_mut().enumerate() {
                *slot = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let [mut a, mut b, mut c, mut d] = state;
            for i in 0..64 {
                let (f, g): (u32, usize) = match i {
                    0..=15 => ((b & c) | (!b & d), i),
                    16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let sum: u32 = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
                let next: u32 = b.wrapping_add(sum.rotate_left(S[i]));
                a = d;
                d = c;
                c = b;
                b = next;
            }
            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in state {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{:02x}", byte));
            }
        }
        out
    }
}
