//! v1.5-C（ADR-18）DSL 往返测试：转场参数化（band / seed / dir）+ layer morph。
//!
//! 覆盖点：
//! 1. `kind` 白名单扩容后，引擎级四转场（fade / wipe / dissolve / push）与
//!    既有 v2 种类（crossfade / cut / black / white）**全部**可编译；
//! 2. 新参数 `band` / `seed` / `dir` 正确回填到产物 [`TransitionV2`]，
//!    缺省值（band 2 / seed 0 / dir None）可观察；
//! 3. `layer ... { morph from "A" to "B" ... }` 声明往返成
//!    [`LayerInfo::morph`]（路径引用逐字保留）；
//! 4. 非法参数显式拒绝（P2106 / P1011），不静默容错。

use prism_dsl::{compile_v2, CompileV2Output};

const HEADER: &str = "// prism-dsl 1\n";

/// 两场一过渡的最小文档（含 layer morph 声明）。
fn doc_two_scenes(transition_body: &str, morph: &str) -> String {
    format!(
        "{HEADER}movie \"m\" {{}}\n\
         scene \"a\" {{ layer text \"t\" {{ {morph} content \"hi\" x 10 y 20 }} }}\n\
         scene \"b\" {{ }}\n\
         transition \"x\" {{ from \"a\" to \"b\" duration 0.5s {transition_body} }}\n"
    )
}

/// 编译成功并返回产物（失败时带源码上下文 panic）。
fn compile_ok(src: &str) -> CompileV2Output {
    match compile_v2(src) {
        Ok(out) => out,
        Err(e) => panic!("以下源码应编译成功，却报错：{e}\n{src}"),
    }
}

/// wipe 参数带宽 + 方向往返。
#[test]
fn transition_wipe_band_dir_roundtrip() {
    let src: String = doc_two_scenes("kind wipe band 4 dir left", "");
    let out: CompileV2Output = compile_ok(&src);
    assert_eq!(out.transitions.len(), 1, "应产出一条过渡");
    let t = &out.transitions[0];
    assert_eq!(t.kind, "wipe", "kind 往返");
    assert_eq!(t.band, 4, "band 往返（px 整数）");
    assert_eq!(t.dir.as_deref(), Some("left"), "dir 往返");
    assert_eq!(t.seed, 0, "wipe 不涉及 seed ⇒ 缺省 0");
    assert!((t.duration - 0.5).abs() < 1e-6, "duration 往返");
}

/// dissolve 种子往返（确定性图案的输入）。
#[test]
fn transition_dissolve_seed_roundtrip() {
    let src: String = doc_two_scenes("kind dissolve seed 7", "");
    let out: CompileV2Output = compile_ok(&src);
    let t = &out.transitions[0];
    assert_eq!(t.kind, "dissolve");
    assert_eq!(t.seed, 7, "seed 往返");
    assert_eq!(t.band, 2, "band 缺省 2（core::sceneseq::WIPE_BAND 同值）");
    assert_eq!(t.dir, None, "dir 缺省 None");
}

/// push 方向 + fade 无参数：四转场白名单全部可用。
#[test]
fn transition_all_engine_kinds_accepted() {
    for kind in ["fade", "dissolve", "push", "wipe"] {
        let body: String = if kind == "push" {
            "kind push dir down".to_string()
        } else {
            format!("kind {kind}")
        };
        let src: String = doc_two_scenes(&body, "");
        let out: CompileV2Output = compile_ok(&src);
        assert_eq!(out.transitions[0].kind, kind, "引擎级转场 {kind} 应可编译");
    }
    // 既有 v2 种类零回归。
    for kind in ["crossfade", "cut", "black", "white"] {
        let src: String = doc_two_scenes(&format!("kind {kind}"), "");
        let out: CompileV2Output = compile_ok(&src);
        assert_eq!(out.transitions[0].kind, kind, "v2 种类 {kind} 零回归");
    }
}

/// 未知种类 / 未知方向 / 非法 band：显式拒绝。
#[test]
fn transition_invalid_params_rejected() {
    let bad_kind: String = doc_two_scenes("kind slide", "");
    let err = compile_v2(&bad_kind).expect_err("未知转场种类应报错");
    assert!(err.message.contains("P2106"), "未知种类应 P2106：{}", err.message);

    let bad_dir: String = doc_two_scenes("kind wipe dir diagonal", "");
    let err2 = compile_v2(&bad_dir).expect_err("未知方向应报错");
    assert!(err2.message.contains("P2106"), "未知方向应 P2106：{}", err2.message);

    let bad_band: String = doc_two_scenes("kind wipe band -3", "");
    let err3 = compile_v2(&bad_band).expect_err("负 band 应报错");
    assert!(err3.message.contains("P1011"), "负 band 应 P1011：{}", err3.message);

    let frac_band: String = doc_two_scenes("kind wipe band 2.5", "");
    let err4 = compile_v2(&frac_band).expect_err("非整数 band 应报错");
    assert!(err4.message.contains("P1011"), "非整数 band 应 P1011：{}", err4.message);
}

/// layer morph 声明往返（字符串路径与裸标识符两种写法）。
#[test]
fn layer_morph_roundtrip() {
    let src: String = doc_two_scenes("kind fade", "morph from \"path_a\" to \"path_b\"");
    let out: CompileV2Output = compile_ok(&src);
    let layer = &out.scenes[0].layers[0];
    assert_eq!(layer.kind, "text");
    assert_eq!(
        layer.morph,
        Some(("path_a".to_string(), "path_b".to_string())),
        "morph 路径引用往返"
    );

    // 裸标识符写法 + 无 morph 的图层保持 None。
    let src2: String = doc_two_scenes("kind fade", "morph from path_a to path_b");
    let out2: CompileV2Output = compile_ok(&src2);
    assert_eq!(
        out2.scenes[0].layers[0].morph,
        Some(("path_a".to_string(), "path_b".to_string())),
        "裸标识符路径引用往返"
    );
    let src3: String = doc_two_scenes("kind fade", "");
    let out3: CompileV2Output = compile_ok(&src3);
    assert_eq!(out3.scenes[0].layers[0].morph, None, "无 morph 声明 ⇒ None");
}

/// morph 声明缺 from / to 时应报错（不静默吞掉）。
#[test]
fn layer_morph_malformed_rejected() {
    let src: String = format!(
        "{HEADER}movie \"m\" {{}}\nscene \"a\" {{ layer text \"t\" {{ morph to \"b\" content \"hi\" }} }}\n"
    );
    let err = compile_v2(&src).expect_err("morph 缺 from 应报错");
    assert!(
        err.message.contains("P1001") || err.message.contains("P1003"),
        "morph 结构错误应 P1001/P1003：{}",
        err.message
    );
}
