//! V3.0 W1 语言层验收：§4 A1 八类语法逐条正例 + 负控、确定性受限子集、
//! 两跑逐字节、换序反例、种子进产物哈希、同源对拍、零损失 md5 对账。
//!
//! 冻结依据 `docs/v3.0-语言规格.md`；判据见 `docs/v3.0-路线图.md` §5.3 W1。

use prism_dsl::{compile_auto, compile_v2, compile_v3, AutoCompile, CompileV3Output};

// ---------------------------------------------------------------------------
// 样例：一次覆盖 A1 的八类特性
// ---------------------------------------------------------------------------

const V3_OK: &str = r#"// prism-dsl 2
seed = 42

def ease(t):
    return t * t * (3 - 2 * t)

radius = 0.5 + ease(0.25)
items = [1, 2, 3, 4]
colors = {"albedo": 1, "roughness": 2}
total = 0
for i in 0..4:
    total = total + items[i]

if total > 5:
    tag = f"r={{radius}} t={{total}}"
else:
    tag = "small"

emit f"""
movie "gen" { duration 1s fps 8 resolution 160x120 format png }
scene "s" {
    duration 1s
    stage3d {
        sdf ball { kind sphere radius {{radius}} }
        material hero { albedo #c0392b roughness 0.3 metalness 0.05 }
    }
}
output { resolution 160x120 format png }
"""
"#;

/// 一段最小可用的 v2 声明（负控样例的 emit 尾部）。
const MIN_EMIT: &str = r#"
emit f"""
movie "g" { duration 1s fps 8 resolution 160x120 format png }
scene "s" { duration 1s stage3d { sdf ball { kind sphere radius 1 } } }
output { resolution 160x120 format png }
"""
"#;

fn wrap(body: &str) -> String {
    format!("// prism-dsl 2\nseed = 7\n{body}\n{MIN_EMIT}")
}

fn ok(src: &str) -> CompileV3Output {
    compile_v3(src).unwrap_or_else(|e| panic!("应编译成功，却在第 {} 行失败：{}", e.line, e.message))
}

fn expect_fail(src: &str, code: &str) -> prism_dsl::DslError {
    let err: prism_dsl::DslError = compile_v3(src)
        .err()
        .unwrap_or_else(|| panic!("负控必须 FAIL，却编译成功（期望 {code}）"));
    assert!(err.line > 0, "错误必须带行号：{}", err.message);
    assert!(
        err.message.contains(code),
        "错误码应含 {code}，实际：{}",
        err.message
    );
    assert!(
        err.message.contains("[P"),
        "消息必须带 [Pxxxx] 错误码：{}",
        err.message
    );
    assert!(
        err.message.contains("建议："),
        "消息必须有三段式的建议段：{}",
        err.message
    );
    assert!(
        err.message.contains("第"),
        "消息必须点明在哪：{}",
        err.message
    );
    err
}

// ---------------------------------------------------------------------------
// 1. A1 八类语法：一条正例全过
// ---------------------------------------------------------------------------

#[test]
fn a1_positive_all_eight_features() {
    let out: CompileV3Output = ok(V3_OK);
    // 缩进块 / 表达式 / 变量 / def / for / if / 列表 / 字典 / 插值 都真实参与了求值。
    assert_eq!(out.seed, 42);
    assert_eq!(out.base.title, "gen");
    assert_eq!(out.base.width, 160);
    assert_eq!(out.base.height, 120);
    assert_eq!(out.base.fps, 8);
    assert!(out.lowered_source.starts_with("// prism-dsl 1\n"));
    assert!(
        out.lowered_source.contains("radius 0.65625"),
        "插值应把 def/表达式结果降级为字面量：{}",
        out.lowered_source
    );
    assert_eq!(out.bindings.len() >= 5, true);
    assert!(!out.trace.is_empty(), "F-5 求值轨迹不能为空");
    assert_ne!(out.artifact_hash, 0);
    // 变量 / def / for / if / 列表 / 字典 / 插值 都在绑定表里留下痕迹。
    for name in ["radius", "items", "colors", "total", "tag"] {
        assert!(out.bindings.contains_key(name), "缺少绑定 {name}");
    }
    assert_eq!(
        out.bindings["total"].to_text(),
        "10",
        "for 按声明序展开 1+2+3+4"
    );
    assert_eq!(out.bindings["tag"].to_text(), "r=0.65625 t=10");
}

// ---------------------------------------------------------------------------
// 2. A1 八类语法：逐条负控（100% FAIL 且带行号 + 三段式）
// ---------------------------------------------------------------------------

#[test]
fn a1_negative_controls_all_fail_with_line_and_three_part_message() {
    // A1-1 缩进块：2 空格不是 4 的倍数 / tab 混用。
    expect_fail(&wrap("def f(x):\n  return x\n"), "P6001");
    expect_fail(&wrap("def f(x):\n\treturn x\n"), "P6002");
    // A1-2 表达式：括号不闭合 / 缺右操作数 / 运算符连写。
    expect_fail(&wrap("x = (1 + 2\n"), "P6004");
    expect_fail(&wrap("x = 1 +\n"), "P6003");
    expect_fail(&wrap("x = 1 + * 2\n"), "P6003");
    // A1-3 变量：未声明即用 / 自引用 / 赋值缺 `=`；重复声明 seed 见 §5 专项。
    expect_fail(&wrap("x = y + 1\n"), "P6102");
    expect_fail(&wrap("x = x + 1\n"), "P6102");
    expect_fail(&wrap("x 1\n"), "P6004");
    // A1-4 def：参数重名 / 缺 return / 直接递归 / 互递归。
    expect_fail(&wrap("def f(a, a):\n    return a\n"), "P6103");
    expect_fail(&wrap("def f(a):\n    x = a\n"), "P6104");
    expect_fail(&wrap("def f(x):\n    return f(x)\n"), "P6105");
    expect_fail(
        &wrap("def f(a):\n    return g(a)\ndef g(a):\n    return f(a)\n"),
        "P6105",
    );
    // A1-5 for / if：上界非常量 / 空区间 / `elif` 后缺分支 / 孤立 `else`。
    expect_fail(&wrap("for i in 0..n:\n    x = i\n"), "P6106");
    expect_fail(&wrap("for i in 4..4:\n    x = i\n"), "P6107");
    expect_fail(&wrap("if 1 > 0:\n    x = 1\nelif 2 > 1:\n"), "P6004");
    expect_fail(&wrap("else:\n    x = 1\n"), "P6108");
    // A1-6 字面量：尾逗号 / 重复键 / 下标越界 / 键不存在。
    expect_fail(&wrap("x = [1, 2,]\n"), "P6201");
    expect_fail(&wrap("x = {\"a\": 1, \"a\": 2}\n"), "P6202");
    expect_fail(&wrap("x = [1, 2][9]\n"), "P6203");
    expect_fail(&wrap("x = {\"a\": 1}[\"b\"]\n"), "P6203");
    // A1-7 插值：表达式非法（多余内容 / 空）/ 引用未声明变量 / `}}` 不闭合。
    expect_fail(&wrap("x = f\"a={{1 2}}\"\n"), "P6204");
    expect_fail(&wrap("x = f\"a={{}}\"\n"), "P6204");
    expect_fail(&wrap("x = f\"a={{missing}}\"\n"), "P6102");
    expect_fail(&wrap("x = f\"a={{b\"\n"), "P6205");
    // A1-8 行号：四类错误域各一个，断言见 `expect_fail` 内统一校验。
    expect_fail("// prism-dsl 2\nseed = 1\nx = @\n", "P6003");
}

// ---------------------------------------------------------------------------
// 3. 非法语义静态拒绝（§3.2）
// ---------------------------------------------------------------------------

#[test]
fn illegal_semantics_are_statically_rejected_with_line() {
    let cases: &[&str] = &[
        "t = now()",                    // 系统时间
        "r = http_get(\"http://x\")",    // 网络
        "write_file(\"a.txt\", \"x\")",  // 文件写
        "e = env(\"HOME\")",             // 环境变量
        "print(\"hi\")",                 // 通用 IO
        "p = open(\"a\")",               // 文件读
        "s = getenv(\"PATH\")",          // 环境变量
        "q = socket()",                 // 网络
        "d = date()",                   // 系统时间
        "z = exec(\"ls\")",              // 通用 IO
    ];
    for case in cases {
        expect_fail(&wrap(&format!("{case}\n")), "P6101");
    }
    // 无界递归（互递归）也静态拒绝。
    expect_fail(
        &wrap("def a(x):\n    return b(x)\ndef b(x):\n    return a(x)\n"),
        "P6105",
    );
    // `while` 不在语法清单内 ⇒ 明确拒绝，不留半截语法。
    expect_fail(&wrap("while x > 0:\n    x = 1\n"), "P6004");
}

// ---------------------------------------------------------------------------
// 3.1 版本头 / 求值 / 降级装配：三段式 + 行号折算
// ---------------------------------------------------------------------------

#[test]
fn p1024_p7002_p7003_negative_controls_carry_line_and_three_parts() {
    // P1024：未知版本 / 缺头 / 版本 1 头 / 版本 2 头未置首行，四类全判红。
    expect_fail("// prism-dsl 3\nseed = 1\n", "P1024");
    expect_fail("seed = 1\n", "P1024");
    expect_fail("// prism-dsl 1\nseed = 1\n", "P1024");
    expect_fail("seed = 1\n// prism-dsl 2\n", "P1024");

    // P7003：非法值（除零 / 类型错）在求值阶段判红，且带三段式。
    expect_fail(&wrap("x = 1 / 0\n"), "P7003");
    expect_fail(&wrap("x = 1 + \"a\"\n"), "P7003");

    // P7002：emit 展开出的 v2 文本编译失败，行号折算回 v3 的 emit 行（第 3 行起）。
    let err: prism_dsl::DslError = expect_fail(
        "// prism-dsl 2\nseed = 7\nemit \"not a v2 document\"\n",
        "P7002",
    );
    assert!(
        err.line >= 3,
        "P7002 必须把 v2 报错行折算回 v3 emit 行，实际第 {} 行",
        err.line
    );
}

#[test]
fn seed_is_first_class_and_never_defaulted() {
    expect_fail(
        "// prism-dsl 2\nemit f\"\"\"\nmovie \"g\" { duration 1s fps 8 resolution 160x120 format png }\n\"\"\"\n",
        "P6110",
    );
    expect_fail("// prism-dsl 2\nseed = 1\nseed = 2\n", "P6111");
    expect_fail("// prism-dsl 2\nseed = 1.5\n", "P6112");
    expect_fail("// prism-dsl 2\nseed = 99999999999999999999999\n", "P6113");
    expect_fail("// prism-dsl 2\nseed = 1\n", "P7001");
}

// ---------------------------------------------------------------------------
// 4. 确定性：两跑逐字节 / 换序反例 / 种子进哈希
// ---------------------------------------------------------------------------

#[test]
fn same_file_same_seed_two_runs_are_byte_identical() {
    let a: CompileV3Output = ok(V3_OK);
    let b: CompileV3Output = ok(V3_OK);
    assert_eq!(a.lowered_source.as_bytes(), b.lowered_source.as_bytes());
    assert_eq!(a.trace, b.trace);
    assert_eq!(a.artifact_hash, b.artifact_hash);
    assert_eq!(a.seed, b.seed);
    assert_eq!(format!("{:?}", a.base), format!("{:?}", b.base));
    assert_eq!(format!("{:?}", a.bindings), format!("{:?}", b.bindings));
}

#[test]
fn reordering_counterexamples_are_detectable() {
    let base: CompileV3Output = ok(V3_OK);
    // ① 语句换序：radius / items 两条赋值交换。
    let stmt_swapped: String = V3_OK
        .replace(
            "radius = 0.5 + ease(0.25)\nitems = [1, 2, 3, 4]",
            "items = [1, 2, 3, 4]\nradius = 0.5 + ease(0.25)",
        );
    assert_ne!(stmt_swapped, V3_OK, "反例必须真的改了源码");
    let s: CompileV3Output = ok(&stmt_swapped);
    assert_ne!(s.artifact_hash, base.artifact_hash, "语句换序必须可检出");
    assert_ne!(s.trace, base.trace, "求值轨迹顺序必须变化");

    // ② 容器声明序：列表元素倒序。
    let list_swapped: String = V3_OK.replace("items = [1, 2, 3, 4]", "items = [4, 3, 2, 1]");
    let l: CompileV3Output = ok(&list_swapped);
    assert_ne!(l.artifact_hash, base.artifact_hash, "容器换序必须可检出");
    assert_eq!(l.bindings["total"].to_text(), "10", "和相同仍应换序敏感");

    // ③ 二元操作数换序：加法交换操作数（数学值相同，求值顺序不同）。
    let op_swapped: String = V3_OK.replace(
        "radius = 0.5 + ease(0.25)",
        "radius = ease(0.25) + 0.5",
    );
    let o: CompileV3Output = ok(&op_swapped);
    assert_eq!(o.bindings["radius"].to_text(), base.bindings["radius"].to_text());
    assert_ne!(o.artifact_hash, base.artifact_hash, "二元求值换序必须可检出");
}

#[test]
fn seed_enters_artifact_hash() {
    let a: CompileV3Output = ok(V3_OK);
    let b: CompileV3Output = ok(&V3_OK.replace("seed = 42", "seed = 43"));
    assert_eq!(a.lowered_source, b.lowered_source, "场景未变，仅种子变");
    assert_ne!(a.artifact_hash, b.artifact_hash, "种子必须进产物哈希");

    // rand(tag) 的显式播种：同种子两跑逐字节，换种子结果变。
    let rand_src: String = r#"// prism-dsl 2
seed = 99
a = rand("grain")
emit f"""
movie "g" { duration 1s fps 8 resolution 160x120 format png }
scene "s" { duration 1s stage3d { sdf ball { kind sphere radius 1 } } }
output { resolution 160x120 format png }
"""
"#
    .to_string();
    let r1: CompileV3Output = ok(&rand_src);
    let r2: CompileV3Output = ok(&rand_src);
    assert_eq!(r1.bindings["a"].to_text(), r2.bindings["a"].to_text());
    let r3: CompileV3Output = ok(&rand_src.replace("seed = 99", "seed = 100"));
    assert_ne!(r1.bindings["a"].to_text(), r3.bindings["a"].to_text());
}

// ---------------------------------------------------------------------------
// 5. 同源对拍：语言产物 ≡ 直接调引擎 API
// ---------------------------------------------------------------------------

#[test]
fn lowered_source_equals_direct_engine_call() {
    let out: CompileV3Output = ok(V3_OK);
    let direct = compile_v2(&out.lowered_source)
        .unwrap_or_else(|e| panic!("展开文本应能被 compile_v2 独立装配：{}", e));
    assert_eq!(
        format!("{:?}", out.base),
        format!("{:?}", direct),
        "语言层不得成为第二事实源：产物必须与直接调 compile_v2 逐字节一致"
    );
}

#[test]
fn det_star_functions_are_reachable_and_deterministic() {
    let src: &str = r#"// prism-dsl 2
seed = 3
s = sqrt(4)
c = cos(0)
a = asin(0.5)
g = log10(1000)
p = pow(2, 10)
r = round(2.5)
m = max(1, 2)
emit f"""
movie "g" { duration 1s fps 8 resolution 160x120 format png }
scene "s" { duration 1s stage3d { sdf ball { kind sphere radius 1 } } }
output { resolution 160x120 format png }
"""
"#;
    let out: CompileV3Output = ok(src);
    assert_eq!(out.bindings["s"].to_text(), "2");
    assert_eq!(out.bindings["c"].to_text(), "1");
    assert_eq!(out.bindings["g"].to_text(), "3");
    assert_eq!(out.bindings["r"].to_text(), "3");
    assert_eq!(out.bindings["m"].to_text(), "2");
    let out2: CompileV3Output = ok(src);
    assert_eq!(out.artifact_hash, out2.artifact_hash);
    assert_eq!(
        out.bindings["a"].to_text(),
        out2.bindings["a"].to_text()
    );
}

// ---------------------------------------------------------------------------
// 6. 零损失：版本头、旧入口、三个 .prism 的 md5 对账
// ---------------------------------------------------------------------------

#[test]
fn version_heads_and_old_entry_are_lossless() {
    // 旧入口对 `// prism-dsl 2` 仍报 P1023（既有测试逐字保留、不删不改）。
    let old = compile_v2("// prism-dsl 2\nscene \"s\" {}");
    assert!(old.is_err());
    assert!(old.err().map(|e| e.message).unwrap_or_default().contains("P1023"));

    // 新分派：无头 / 版本 1 / 版本 2 各自走对路径。
    // 无头黄金文件属 DSL v1 语法，既有产品路径用 `compile()`（CLI 默认入口）。
    let no_head: &str = include_str!("../../../examples/spin_red_ball.prism");
    let v1_out = prism_dsl::compile(no_head).expect("v1 黄金示例必须能被 compile() 编译");
    assert_eq!(v1_out.width, 480);
    assert_eq!(v1_out.height, 270);

    let v1: &str = include_str!("../../../examples/w2d_showcase.prism");
    let t0: &str = include_str!("../../../examples/t0_media_mesh.prism");
    assert!(matches!(compile_auto(v1), Ok(AutoCompile::V2(_))));
    assert!(matches!(compile_auto(t0), Ok(AutoCompile::V2(_))));
    assert!(matches!(compile_auto(V3_OK), Ok(AutoCompile::V3(_))));

    // 分派对「无头 / 版本 1」与直接调 compile_v2 **完全等价**（成功或失败都一致，
    // 不静默降级、不改写行为）——这是 render 子命令侧的零损失契约。
    for src in [no_head, v1, t0] {
        let direct: Result<String, String> = compile_v2(src)
            .map(|o| format!("{:?}", o))
            .map_err(|e| e.message);
        let auto: Result<String, String> = compile_auto(src)
            .map(|o| format!("{:?}", o.into_v2_output()))
            .map_err(|e| e.message);
        assert_eq!(direct, auto, "无头 / 版本 1 文件的分派必须完全等价");
    }
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

#[test]
fn golden_prism_files_are_byte_identical_by_md5() {
    assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    let cases: &[(&[u8], &str)] = &[
        (
            include_bytes!("../../../examples/spin_red_ball.prism"),
            "e75b03e4bc7fcc090561eb1163f8ad24",
        ),
        (
            include_bytes!("../../../examples/t0_media_mesh.prism"),
            "46e29136de352ea75341823f2b69bf18",
        ),
        (
            include_bytes!("../../../examples/w2d_showcase.prism"),
            "88f735505eab09c6763b644197eee51b",
        ),
    ];
    for (bytes, want) in cases {
        assert_eq!(&md5::hex(bytes), want, "黄金 .prism 文件内容发生变化");
    }
}
