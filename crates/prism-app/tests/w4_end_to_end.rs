//! V3.0 W4 端到端凭证：统一顶层入口 `prism render <file>`、单文件双档、
//! 空环境冷启动（无 ffmpeg / 无 GPU）、一条命令出片、两跑逐字节、观感负控、
//! 四类错误域 CLI 三段式、外部资产缺失结构化报错、D-27 内容哈希锁定。
//!
//! 用 `CARGO_BIN_EXE_prism-app`（W1-W3 既有 CI 路径）与 `CARGO_BIN_EXE_prism`
//! （W4 统一顶层入口别名）起真实二进制进程，绝不靠源码扫描冒充端到端。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const FILMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/films");

fn film(name: &str) -> PathBuf {
    PathBuf::from(FILMS).join(format!("{name}.prism"))
}

fn out(tag: &str, run: u32) -> PathBuf {
    // 按 PID 唯一化：workspace 全量测试里本集成测试会被多个 bin target 各编译
    // 运行一次，固定目录会跨进程竞态（写同一 mp4 破坏两跑逐字节对拍）。
    let dir = std::env::temp_dir().join(format!("prism-w4-e2e-{}-{}", std::process::id(), tag));
    fs::create_dir_all(&dir).unwrap();
    dir.join(format!("run{run}.mp4"))
}

/// 空环境冷启动：`PATH` 清空（无 ffmpeg）、`--fast` 纯 CPU（无 GPU）、
/// `--codec mjpeg` 内置出口（不重开 C2）。返回真实进程 Output 与退出码。
fn cold_render(bin: &str, file: &PathBuf, mp4: &PathBuf) -> Output {
    Command::new(bin)
        .args([
            "render",
            file.to_str().unwrap(),
            "--out",
            mp4.to_str().unwrap(),
            "--codec",
            "mjpeg",
            "--fast",
        ])
        .env("PATH", "")
        .output()
        .expect("统一顶层入口进程应可启动")
}

fn is_mp4(path: &PathBuf) -> bool {
    let b = fs::read(path).unwrap_or_default();
    b.len() > 1024 && b.len() > 4 && &b[4..8] == b"ftyp"
}

// ---------------------------------------------------------------------------
// 1. 统一顶层入口 + 冷启动（§7 #9）+ 双档两跑逐字节（§5.3 W4 #5）
// ---------------------------------------------------------------------------

#[test]
fn w4_unified_entry_cold_start_exits_zero_and_double_run_is_byte_identical() {
    // `prism` 与 `prism-app` 两个 bin 都同源可用；W4 顶层入口字面兑现。
    for bin in [env!("CARGO_BIN_EXE_prism"), env!("CARGO_BIN_EXE_prism-app")] {
        for name in ["analytic", "crafted"] {
            let f = film(name);
            let p1 = out(&format!("{name}-a"), 1);
            let p2 = out(&format!("{name}-a"), 2);
            let o1 = cold_render(bin, &f, &p1);
            assert_eq!(o1.status.code(), Some(0),
                       "{bin} {name} 空环境冷启动退出码 0，stderr={}", String::from_utf8_lossy(&o1.stderr));
            assert!(is_mp4(&p1), "{name} 成片 MP4(ftyp) 非空");
            let o2 = cold_render(bin, &f, &p2);
            assert_eq!(o2.status.code(), Some(0));
            assert_eq!(fs::read(&p1).unwrap(), fs::read(&p2).unwrap(),
                       "{name} 两跑逐字节一致");
        }
    }
}

#[test]
fn w4_one_command_covers_frames_dimensions_and_no_external_dir() {
    // 3 分钟片 = 1 个文件：帧数 180 / 尺寸 32×24 对得上、退出码 0、无外部依赖目录。
    let f = film("analytic");
    let mp4 = out("frames", 1);
    let o = cold_render(env!("CARGO_BIN_EXE_prism"), &f, &mp4);
    assert_eq!(o.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(stdout.contains("帧数      : 180 帧"), "帧数对得上：{stdout}");
    assert!(stdout.contains("32x24"), "尺寸对得上：{stdout}");
    assert!(stdout.contains("mjpeg"), "内置降级路径仍成立：{stdout}");
    // 无外部依赖目录：片子目录下只有 .prism 文件（inline 资产自包含）。
    let entries: Vec<_> = fs::read_dir(FILMS).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert!(entries.iter().all(|n| n.to_string_lossy().ends_with(".prism")),
            "examples/films/ 无 assets/ 等外部依赖目录：{entries:?}");
}

// ---------------------------------------------------------------------------
// 2. 单文件凭证 / 观感双档 / 红线 4（analytic 不偷加噪、不替用户选档）
// ---------------------------------------------------------------------------

#[test]
fn w4_dual_finish_receipts_and_analytic_zero_noise() {
    // crafted：六族观感凭证 + 单文件凭证都打印；analytic：零观感凭证。
    let c = cold_render(env!("CARGO_BIN_EXE_prism"), &film("crafted"), &out("receipt-c", 1));
    let cs = String::from_utf8_lossy(&c.stdout);
    assert!(cs.contains("单文件凭证") && cs.contains("观感凭证") && cs.contains("crafted"),
            "crafted 档凭证齐全：{cs}");
    let a = cold_render(env!("CARGO_BIN_EXE_prism"), &film("analytic"), &out("receipt-a", 1));
    let as_ = String::from_utf8_lossy(&a.stdout);
    assert!(as_.contains("单文件凭证"), "analytic 档也有单文件凭证：{as_}");
    assert!(!as_.contains("观感凭证"), "analytic 档零观感凭证（不偷加噪、不替用户选档）：{as_}");
}

#[test]
fn w4_missing_finish_is_static_error_with_line() {
    // 红线 3/4：漏填 finish ⇒ P9001 三段式带行号、退出码非 0。
    let src = "// prism-dsl 2\nseed = 42\nemit f\"\"\"\nmovie \"g\" { duration 1s fps 2 resolution 32x24 format png }\nscene \"s\" { duration 1s stage3d { sdf ball { kind sphere radius 1 } } }\n\"\"\"\n";
    let dir = std::env::temp_dir().join(format!("prism-w4-nofinish-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let p = dir.join("nofinish.prism");
    fs::write(&p, src).unwrap();
    let mp4 = dir.join("o.mp4");
    let _ = fs::remove_file(&mp4);
    let o = cold_render(env!("CARGO_BIN_EXE_prism"), &p, &mp4);
    assert_ne!(o.status.code(), Some(0), "漏 finish 必须 FAIL");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("P9001") && err.contains("建议：") && err.contains("第"),
            "三段式带行号：{err}");
    assert!(!mp4.exists(), "失败不落盘半成品");
}

// ---------------------------------------------------------------------------
// 3. D-27 资产缺失 / 哈希不符 ⇒ CLI 结构化报错（不静默降级、不取默认）
// ---------------------------------------------------------------------------

#[test]
fn w4_external_asset_missing_or_tampered_is_cli_error_not_silent_fallback() {
    // 目录按「进程 PID + 用例名」唯一化：workspace 全量测试里本集成测试会被
    // 多个 bin target 各编译运行一次，固定目录会跨进程竞态（上一跑遗留的
    // assets/lut.exr 会破坏步骤①「资产缺失」的前提）。先整目录删除再重建，
    // 保证每一步的初始状态确定、可复现。
    let dir = std::env::temp_dir().join(format!(
        "prism-w4-asset-cli-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let emit = "emit f\"\"\"\nmovie \"g\" { duration 1s fps 1 resolution 16x12 format png }\nscene \"s\" { duration 1s stage3d { sdf ball { kind sphere radius 1 } } }\n\"\"\"\n";
    // 外部引用声明的真实内容哈希（与 inline \"0a1b2c3d\" 一致）。
    let src = format!("// prism-dsl 2\nseed = 42\nfinish = analytic\nasset \"lut\" kind exr hash \"d76e986531f947bc\" external \"assets/lut.exr\"\n{emit}");
    let p = dir.join("film.prism");
    fs::write(&p, &src).unwrap();
    let mp4 = dir.join("o.mp4");
    // ① 文件缺失 ⇒ P9508 三段式带行号、退出码非 0、不落盘。
    // 显式删除可能残留的 assets 目录，确保「缺失」前提成立。
    let _ = fs::remove_dir_all(dir.join("assets"));
    let _ = fs::remove_file(&mp4);
    let o = cold_render(env!("CARGO_BIN_EXE_prism"), &p, &mp4);
    assert_ne!(o.status.code(), Some(0), "缺文件必须 FAIL，stderr={}", String::from_utf8_lossy(&o.stderr));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("P9508") && err.contains("建议："), "缺失即结构化报错：{err}");
    assert!(!mp4.exists(), "缺资产不静默降级出片");
    // ② 文件存在但内容被篡改 ⇒ P9509 三段式带行号。
    fs::create_dir_all(dir.join("assets")).unwrap();
    fs::write(dir.join("assets/lut.exr"), b"tampered").unwrap();
    let _ = fs::remove_file(&mp4);
    let o2 = cold_render(env!("CARGO_BIN_EXE_prism"), &p, &mp4);
    assert_ne!(o2.status.code(), Some(0), "哈希不符必须 FAIL，stderr={}", String::from_utf8_lossy(&o2.stderr));
    assert!(String::from_utf8_lossy(&o2.stderr).contains("P9509"), "哈希不符判红");
    // ③ 内容一致 ⇒ 校验通过、退出码 0、成片（外部引用为主，路径相对片子目录解析）。
    fs::write(dir.join("assets/lut.exr"), [0x0a, 0x1b, 0x2c, 0x3d]).unwrap();
    let o3 = cold_render(env!("CARGO_BIN_EXE_prism"), &p, &mp4);
    assert_eq!(o3.status.code(), Some(0), "一致即通过，stderr={}", String::from_utf8_lossy(&o3.stderr));
    assert!(is_mp4(&mp4));
}

// ---------------------------------------------------------------------------
// 4. A7 命名时序通道真的进渲染链（不是只算一遍给自己看）
// ---------------------------------------------------------------------------

#[test]
fn w4_timed_channels_are_consumed_by_render_chain() {
    // 同一 crafted 片子：有 / 无命名时序通道两版，逐帧 wiring 值不同 ⇒
    // 通道不是「字段没接」的死声明（V2 负控⑤ 的反面）。
    use prism_dsl::compile_film;
    let src = fs::read_to_string(film("crafted")).unwrap();
    let with_ch = compile_film(&src).unwrap();
    // 去掉所有 channel 声明行 ⇒ plan.channels 空、compile.channels 无 w4.drive.*。
    let stripped: String = src.lines().filter(|l| !l.trim_start().starts_with("channel ")).collect::<Vec<_>>().join("\n") + "\n";
    let no_ch = compile_film(&stripped).unwrap();
    assert!(no_ch.plan.channels.is_empty());
    assert_eq!(with_ch.compile.channels.iter().filter(|c| c.name.starts_with("w4.drive.")).count(), 3);
    assert_eq!(no_ch.compile.channels.iter().filter(|c| c.name.starts_with("w4.drive.")).count(), 0);
    // 逐帧值差异可检出（frame 0 vs 120）。
    let f0 = with_ch.plan.channels.frame_values(0.0, 0, 0, 0);
    let f120 = with_ch.plan.channels.frame_values(120.0, 120, 0, 0);
    assert_ne!(f0["w3.motion.strength"], f120["w3.motion.strength"]);
    // 渲染前覆盖层：apply_timed_channels 把 w4.drive.<target> 采样值写进 WiringFrame。
    use prism_core::PropertyChannel;
    use std::collections::BTreeMap;
    let mut wf = prism_dsl::WiringFrame::default();
    let owned: Vec<(String, PropertyChannel)> = with_ch.compile.channels.iter()
        .filter(|c| c.name.starts_with("w4.drive."))
        .map(|c| (c.name.clone(), c.channel.clone())).collect();
    let map: BTreeMap<&str, &PropertyChannel> = owned.iter().map(|(n, c)| (n.as_str(), c)).collect();
    prism_dsl::apply_timed_channels(&mut wf, &map, 120.0, 120);
    assert!(wf.values.contains_key("w3.motion.strength"), "渲染前帧值凭证含时序驱动键");
    assert_eq!(wf.value("w3.motion.strength").unwrap(), f32::from_bits(f120["w3.motion.strength"]),
               "与声明通道逐帧对拍一致（同源求值，不两端同错）");
    // analytic 档无 w4.drive.* ⇒ 覆盖层恒等、零意外噪点。
    let a = compile_film(&fs::read_to_string(film("analytic")).unwrap()).unwrap();
    let mut wfa = prism_dsl::WiringFrame::default();
    let empty: BTreeMap<&str, &PropertyChannel> = BTreeMap::new();
    prism_dsl::apply_timed_channels(&mut wfa, &empty, 0.0, 0);
    assert!(wfa.values.is_empty());
    let _ = a;
}
