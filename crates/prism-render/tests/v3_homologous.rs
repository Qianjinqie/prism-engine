//! V3.0 W1 同源对拍（反假绿 §1.3 第 2 条）与 CPU 两跑逐字节。
//!
//! 语言层不得成为第二事实源：`compile_v3` 降级出的 `CompileV2Output`，
//! 与「手写等价 v2 文件直接调引擎 API」的 `CompileV2Output` 必须出**逐字节同帧**。
//! 全部走既有 `render_movie` + `CpuRefBackend`，不新建渲染器、不新建 harness。

use prism_dsl::{compile_auto, compile_v2, compile_v3, AutoCompile, CompileV3Output};
use prism_gpu::CpuRefBackend;

/// 小画幅（32×24、1s、2fps = 2 帧），保证逐字节对拍快速且完全确定。
const V3_SMALL: &str = r#"// prism-dsl 2
seed = 11
r = 0.5 + 0.25
emit f"""
movie "g" { duration 1s fps 2 resolution 32x24 format png }
scene "s" {
    duration 1s
    stage3d {
        camera main { orbit { center 0,0,0 radius 4.5 } fov 55 }
        sdf ball { kind sphere radius {{r}} }
        material hero { albedo #c0392b roughness 0.4 metalness 0.1 }
        light key { direction(0.4,-1,-0.6) color #ffffff intensity 3.2 }
    }
}
output { resolution 32x24 format png }
"""
"#;

/// 与 `V3_SMALL` 等价的**手写 v2 文件**（直接调引擎 API 的一侧）。
const V2_EQUIVALENT: &str = r#"// prism-dsl 1
movie "g" { duration 1s fps 2 resolution 32x24 format png }
scene "s" {
    duration 1s
    stage3d {
        camera main { orbit { center 0,0,0 radius 4.5 } fov 55 }
        sdf ball { kind sphere radius 0.75 }
        material hero { albedo #c0392b roughness 0.4 metalness 0.1 }
        light key { direction(0.4,-1,-0.6) color #ffffff intensity 3.2 }
    }
}
output { resolution 32x24 format png }
"#;

/// 渲染产物的规范字节序列（标题 / 画幅 / 帧率 / 时长 / 每帧像素）。
fn render_bytes(compile: &prism_dsl::CompileV2Output) -> Vec<u8> {
    let mut device = CpuRefBackend;
    let deliverable = prism_render::render_movie(compile, &mut device)
        .expect("render_movie 应成功出片");
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(deliverable.title.as_bytes());
    bytes.extend_from_slice(&deliverable.width.to_le_bytes());
    bytes.extend_from_slice(&deliverable.height.to_le_bytes());
    bytes.extend_from_slice(&deliverable.fps.to_le_bytes());
    bytes.extend_from_slice(&deliverable.duration.to_le_bytes());
    bytes.extend_from_slice(&(deliverable.frames.len() as u64).to_le_bytes());
    for frame in &deliverable.frames {
        bytes.extend_from_slice(&(frame.width as u64).to_le_bytes());
        bytes.extend_from_slice(&(frame.height as u64).to_le_bytes());
        bytes.extend_from_slice(&frame.pixels);
    }
    bytes
}

#[test]
fn language_output_equals_direct_engine_call_byte_for_byte() {
    let v3: CompileV3Output = compile_v3(V3_SMALL)
        .unwrap_or_else(|e| panic!("v3 应编译成功，第 {} 行：{}", e.line, e.message));
    assert!(v3.lowered_source.contains("radius 0.75"));
    let via_language: Vec<u8> = render_bytes(&v3.base);
    let direct = compile_v2(V2_EQUIVALENT).expect("等价 v2 文件应能独立编译");
    let via_api: Vec<u8> = render_bytes(&direct);
    assert!(!via_language.is_empty(), "对拍产物不能为空");
    assert_eq!(
        via_language, via_api,
        "语言产物必须与直接调引擎 API 逐字节同帧（禁止语言层自己再算一遍）"
    );
    // 展开文本独立编译与 base 也逐字段一致。
    let lowered = compile_v2(&v3.lowered_source).expect("展开文本应可独立编译");
    assert_eq!(format!("{:?}", lowered), format!("{:?}", v3.base));
}

#[test]
fn same_file_same_seed_two_cpu_runs_are_byte_identical() {
    let a: CompileV3Output = compile_v3(V3_SMALL).expect("第一次编译");
    let b: CompileV3Output = compile_v3(V3_SMALL).expect("第二次编译");
    assert_eq!(a.artifact_hash, b.artifact_hash);
    assert_eq!(a.lowered_source.as_bytes(), b.lowered_source.as_bytes());
    assert_eq!(a.trace, b.trace);
    assert_eq!(render_bytes(&a.base), render_bytes(&b.base));
}

#[test]
fn compile_auto_dispatch_is_lossless_on_v2_files() {
    for src in [
        include_str!("../../../examples/t0_media_mesh.prism"),
        include_str!("../../../examples/w2d_showcase.prism"),
    ] {
        let direct = compile_v2(src).expect("版本 1 文件应能编译");
        let auto = match compile_auto(src).expect("分派应成功") {
            AutoCompile::V2(out) => out,
            AutoCompile::V3(_) => panic!("版本 1 文件不应走 v3 前端"),
        };
        assert_eq!(
            format!("{:?}", direct),
            format!("{:?}", auto),
            "compile_auto 对版本 1 文件必须与 compile_v2 完全等价"
        );
    }
}

#[test]
fn reordering_changes_artifact_hash_but_not_scene_bytes() {
    let base: CompileV3Output = compile_v3(V3_SMALL).expect("基线编译");
    let swapped: String = V3_SMALL.replace("r = 0.5 + 0.25", "r = 0.25 + 0.5");
    let reordered: CompileV3Output = compile_v3(&swapped).expect("换序文件应能编译");
    assert_eq!(base.lowered_source, reordered.lowered_source, "场景语义未变");
    assert_ne!(
        base.artifact_hash, reordered.artifact_hash,
        "换序必须被求值轨迹检出"
    );
    assert_eq!(
        render_bytes(&base.base),
        render_bytes(&reordered.base),
        "场景语义未变时帧字节应逐字节相同"
    );
}
