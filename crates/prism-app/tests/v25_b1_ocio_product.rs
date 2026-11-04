//! V2.5-B1 判据 V6（产品路径）：`prism-app render --ocio` / `Scene.attach.ocio_config`
//! 真能吃到 config，**非空断言**（不是只扫源码）。

use std::fs;
use std::path::PathBuf;

use prism_app::ocio_bridge::load_ocio_pipeline;

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("prism_v25_b1_{name}"));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).expect("临时目录应可创建");
    d
}

const CONFIG: &str = r#"
ocio_profile_version: 2
search_path: luts
strictparsing: false
roles:
  default: raw
displays:
  sRGB:
    - !<View> {name: Filmic, colorspace: Linear}
colorspaces:
  - !<ColorSpace>
    name: Linear
    from_scene_reference: !<GroupTransform>
      children:
        - !<RangeTransform> {min_in_value: 0, max_in_value: 1, min_out_value: 0, max_out_value: 1}
        - !<ExponentTransform> {value: [0.75, 0.75, 0.75, 1]}
        - !<FileTransform> {src: tone.spi1d}
  - !<ColorSpace>
    name: raw
    to_scene_reference: !<MatrixTransform> {matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
"#;

const TONE_SPI1D: &str = "Version 1\nFrom 0 1\nTo 0 1\nLength 3\nComponents 3\n0 0 0\n0.5 0.5 0.5\n1 1 1\n";

#[test]
fn v6_product_path_loads_config_and_applies_lut() {
    let d = temp_dir("product");
    let luts = d.join("luts");
    fs::create_dir_all(&luts).unwrap();
    fs::write(luts.join("tone.spi1d"), TONE_SPI1D).unwrap();
    let cfg = d.join("config.ocio");
    fs::write(&cfg, CONFIG).unwrap();

    let pipeline = load_ocio_pipeline(cfg.to_str().unwrap())
        .expect("合法 config 必须装配成功")
        .expect("非空 config 必须产出 LUT 管线");
    assert_eq!(pipeline.config.colorspaces.len(), 2);
    assert_eq!(pipeline.config.displays.len(), 1);
    assert!(pipeline.lut.transform_count >= 3, "transform 链非空");
    assert_eq!(pipeline.lut.cube.size, 17);
    assert!(pipeline.lut.shaper.size >= 3, "1D shaper 真生成");

    // 帧：LUT 必须真改变像素（防「接了个空操作」）。
    let frame: Vec<u8> = (0..4096u32).map(|i| ((i * 29 + 7) % 256) as u8).collect();
    let mut frames = vec![frame.clone()];
    pipeline.apply_frames(&mut frames);
    assert_ne!(frames[0], frame, "产品路径必须真应用 LUT（非空断言）");
    // 双跑逐字节（V5 的产品侧复验）。
    let mut again = vec![frame.clone()];
    pipeline.apply_frames(&mut again);
    assert_eq!(frames, again);
    // 诊断摘要非空且含数字。
    let summary = pipeline.summary();
    assert!(summary.contains("config.ocio"));
    assert!(summary.contains("cube 17"));
}

#[test]
fn v6_empty_config_is_noop_in_product_path() {
    let d = temp_dir("empty");
    let cfg = d.join("empty.ocio");
    fs::write(&cfg, "ocio_profile_version: 2\nsearch_path: .\n").unwrap();
    assert!(
        load_ocio_pipeline(cfg.to_str().unwrap()).unwrap().is_none(),
        "空 config 必须返回 None ⇒ 产品路径逐字节同既有 ACES"
    );
}

#[test]
fn v6_scene_field_type_is_present_and_default_none() {
    // `prism_core::Scene` 的字段带 Default（None），旧场景行为不变。
    let scene = prism_core::Scene::default();
    assert!(scene.attach.ocio_config.is_none());
    assert!(scene.attach.is_empty(), "ocio_config 不改变 rt 几何空判定");
}
