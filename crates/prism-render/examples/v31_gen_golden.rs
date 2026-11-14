//! V3.1 golden 资产生成器（真实可复跑，非桩）。
//!
//! ```bash
//! cargo run -q -p prism-render --release --example v31_gen_golden
//! ```
//!
//! 生成 `docs/v3.1-golden/320x180_preview/frame_%04d.png`（16 spp / 6 bounces /
//! 确定性 CPU / 无累积降噪自适应）并重写 `manifest.sha256`。T03 补齐全档位
//! 时只需构造更多 [`GoldenTier`] 并调用同一个 `GoldenStore::generate_tier`。

use prism_render::{GoldenStore, GoldenTier};

fn main() {
    let store: GoldenStore = GoldenStore::at_repo_root();
    let tier: GoldenTier = GoldenTier::minimal();

    println!("==> golden 根目录：{}", store.root().display());
    println!(
        "==> 生成档位 {}：{}×{}，{} 帧，16 spp / 6 bounces",
        tier.tag(),
        tier.res.dims().0,
        tier.res.dims().1,
        tier.frames
    );

    let summary = store
        .generate_tier(&tier)
        .unwrap_or_else(|e| panic!("生成 golden 失败：{e}"));
    for f in &summary.frames {
        println!(
            "    frame_{:04}.png  {} B  {}",
            f.frame, f.bytes, f.sha256_hex
        );
    }

    let entries = store
        .regenerate_manifest()
        .unwrap_or_else(|e| panic!("生成 manifest.sha256 失败：{e}"));
    println!("==> manifest.sha256 共 {} 条", entries.len());

    let verified = store
        .verify_manifest()
        .unwrap_or_else(|e| panic!("manifest 自校验失败：{e}"));
    assert_eq!(verified.len(), entries.len(), "manifest 写入与读回条数不一致");
    println!("==> manifest 自校验通过");
}
