//! V3.0 W4 单文件样例编译探针：双档 `.prism` 经 `compile_film` 必须成功，
//! 且资产内容哈希锁定 / 命名通道降级 / 出片规格三项读数可断言。
//! 这是 `examples/films/` 样例进入 CI 的最小门禁（样例存在 ≠ 触达）。

use prism_dsl::compile_film;

const FILMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/films");

#[test]
fn both_films_compile_with_locked_assets_and_driven_channels() {
    for name in ["analytic", "crafted"] {
        let src = std::fs::read_to_string(format!("{FILMS}/{name}.prism")).unwrap();
        let fc = compile_film(&src).unwrap_or_else(|e| {
            panic!("{name}.prism 必须可编译，第 {} 行：{}", e.line, e.message)
        });
        // 3 类资产全内容哈希锁定。
        assert_eq!(fc.plan.assets.len(), 3, "{name}: EXR/字体/音频 3 类资产");
        // 命名通道全降级进既有 compile.channels。
        let drives = fc
            .compile
            .channels
            .iter()
            .filter(|c| c.name.starts_with("w4.drive."))
            .count();
        assert_eq!(drives, fc.plan.channels.len(), "{name}: 通道全降级");
        assert!(drives >= 2, "{name}: 至少 2 条命名时序通道");
        // 3 分钟形态低分辨率确定性档（K-4）：帧数/尺寸对得上。
        assert_eq!(fc.compile.duration, 180.0);
        assert_eq!(fc.compile.fps, 1);
        assert_eq!((fc.compile.width, fc.compile.height), (32, 24));
        assert_eq!(fc.compile.duration as u32 * fc.compile.fps, 180, "180 帧");
    }
}
