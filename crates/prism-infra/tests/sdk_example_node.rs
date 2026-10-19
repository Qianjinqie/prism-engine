//! EC-3 示例节点端到端：`examples/sdk-nodes/cinematic-basic`。
//!
//! 验收四步（缺一不可，对应"可编译 / 可加载 / 版本协商"三判据）：
//! 1. manifest 解析 + 静态校验 + 资产清单校验（契约完整）；
//! 2. 版本协商：当前 [`prism_infra::sdk::ENGINE_VERSION`] 满足包声明约束；
//! 3. `presets/*` 与 `subgraphs/*` 全部过 `prism_dsl::compile`（**可编译**）；
//! 4. 市场 add → list 往返（**可加载**），provides 能力名与文件一一对应。
//!
//! 失败代表：示例包本身腐烂（改坏了 DSL / manifest / 目录布局）而无人知晓。

use prism_infra::plugin::{Manifest, PackLibrary};
use prism_infra::sdk::{self, Compatibility};
use std::path::{Path, PathBuf};

/// 示例数据包目录（相对本测试 crate 的 workspace 根）。
const PACK_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/sdk-nodes/cinematic-basic");

/// 包内声明的能力名（与 manifest 的 provides 一致——文件与契约一一对应）。
const PROVIDES: [&str; 2] = ["grade-film", "hero-shot"];

/// 包内全部 DSL 文件（presets + subgraphs；端到端必须逐个可编译）。
const DSL_FILES: [&str; 2] = ["presets/grade-film.prism", "subgraphs/hero-shot.prism"];

#[test]
fn example_node_pack_compiles_and_loads_end_to_end() {
    let pack_root: &Path = Path::new(PACK_DIR);
    assert!(pack_root.is_dir(), "示例包目录必须存在：{}", pack_root.display());

    // ---- 1. manifest：解析 + 静态校验 + 资产清单 ----
    let manifest_text: String = std::fs::read_to_string(pack_root.join("manifest.toml"))
        .expect("manifest.toml 必须可读");
    let manifest: Manifest = Manifest::parse_toml(&manifest_text).expect("示例 manifest 必须可解析");
    manifest.validate_static().expect("静态字段校验必须通过");
    manifest.validate_assets(pack_root).expect("资产清单校验必须通过");
    assert_eq!(manifest.name, "cinematic-basic");
    assert_eq!(manifest.version, (0, 1, 0));
    assert_eq!(manifest.license, "MIT");
    assert_eq!(manifest.provides.len(), PROVIDES.len(), "provides 条数须与判据一致");
    for p in PROVIDES {
        assert!(manifest.provides.contains(&p.to_string()), "应提供 '{p}'");
    }
    assert_eq!(manifest.assets, vec!["assets/palette.cube".to_string()]);

    // ---- 2. 版本协商：与当前引擎接口版本兼容 ----
    match sdk::check_manifest(&manifest).expect("示例约束格式必须合法") {
        Compatibility::Compatible => {}
        other => panic!("示例包应与 {} 兼容，实际 {:?}", sdk::ENGINE_VERSION, other),
    }

    // ---- 3. 端到端可编译：每个 .prism 过 DSL 编译器 ----
    for rel in DSL_FILES {
        let path: PathBuf = pack_root.join(rel);
        let src: String =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("读 {rel} 失败：{e}"));
        prism_dsl::compile(&src).unwrap_or_else(|e| panic!("{rel} 应可编译（第 {} 行）：{}", e.line, e.message));
    }

    // ---- 4. 市场可加载：add → list 往返 ----
    let tmp: PathBuf =
        std::env::temp_dir().join(format!("prism-sdk-example-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let lib = PackLibrary::new(&tmp);
    let entry = lib.add(pack_root).expect("示例包必须可入库（版本协商 + 资产齐）");
    assert_eq!(entry.name, "cinematic-basic");
    assert_eq!(entry.version, (0, 1, 0));
    let listed = lib.list();
    assert_eq!(listed.len(), 1, "入库后 list 应恰一条");
    assert_eq!(listed[0].name, "cinematic-basic");
    assert!(lib.remove("cinematic-basic").is_ok(), "remove 往返应成功");
    assert!(lib.list().is_empty());
    let _ = std::fs::remove_dir_all(&tmp);
}

/// 反向闸门：把示例包的 `engine` 约束改成不可满足版本 ⇒ add 必须明确拒绝。
///
/// 与正向用例同一目录、同一文件，只改约束串——证明拒绝的是**版本协商**，
/// 而非别的（资产 / 名 / semver 都保持有效）。
#[test]
fn example_pack_with_future_engine_constraint_is_rejected() {
    let pack_root: &Path = Path::new(PACK_DIR);
    let manifest_text: String =
        std::fs::read_to_string(pack_root.join("manifest.toml")).expect("manifest 必须可读");
    // 就地替换约束窗口为不可满足版本（不碰磁盘文件：内存中改写后重解析）。
    let future: String = manifest_text.replace(
        "engine = \">=0.1, <0.2\"",
        "engine = \">=99.0, <99.1\"",
    );
    assert_ne!(future, manifest_text, "替换必须真的发生（防止 manifest 常量漂移后测试空转）");
    let manifest: Manifest = Manifest::parse_toml(&future).expect("改写后仍须可解析");
    match sdk::check_manifest(&manifest).expect("格式合法") {
        Compatibility::Incompatible(conflict) => {
            assert!(conflict.to_string().contains("P4001"));
            assert!(conflict.to_string().contains("建议"));
        }
        other => panic!("未来约束必须不兼容，实际 {other:?}"),
    }
}
