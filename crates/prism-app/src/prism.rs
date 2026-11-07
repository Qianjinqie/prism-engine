//! `prism` —— V3.0 W4 统一顶层入口（作者契约 §5.3 W4 / §7 #9 字面兑现）。
//!
//! 一部片子 = 一个 `.prism` 文件，`prism render <file.prism>` 从冷启动到成片：
//! 编译（`compile_film`：单文件形态 + 资产内容哈希锁定 + 命名时序通道）→
//! `render_movie` 逐帧求值 → 内置 MP4(MJPEG) 容器落盘。无中间手工步骤、
//! 无外部依赖目录；无 ffmpeg / 无 GPU 时走内置 MJPEG 降级仍出片（不重开 C2）。
//!
//! # 与 `prism-app` 的关系
//! 出片实现只有一份：`cmd_render::run`（与 `prism-app render` 完全同源）。本文件
//! 只做 `render` 子命令的分发与薄封装，不重写出片逻辑、不引第三方依赖。
//! 独立源文件使 Cargo 不产生 "same main.rs in multiple bin targets" 歧义，
//! 且不把 CLI 挂进 lib、`cdylib`（wasm 编辑器产物）零影响。

#[path = "cmd_render.rs"]
mod cmd_render;

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let code: i32 = match raw.first().map(String::as_str) {
        // `prism render <file.prism> [选项...]`：与 `prism-app render` 同一实现。
        Some("render") => cmd_render::run(&raw[1..]),
        // 帮助 / 空参数：透传给同一用法说明。
        Some("-h") | Some("--help") | None => cmd_render::run(&raw),
        // 其余顶层子命令不属于统一出片入口：明确三段式报错，不静默吞掉。
        Some(other) => {
            eprintln!(
                "prism: 在哪：顶层入口 `prism`\n  什么事：未知子命令 '{other}'（统一出片入口只接受 `render`）\n  建议：用 `prism render <file.prism> [--out <FILE>] [--codec mjpeg] [--fast]`"
            );
            1
        }
    };
    std::process::exit(code);
}
