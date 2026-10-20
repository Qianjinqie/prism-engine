//! `prism-app preview` 子命令端到端测试。
//!
//! 用 `CARGO_BIN_EXE_prism-app` 起真实二进制进程，验证：
//! 1. **单帧模式**：退出码 0 + 输出 PNG 文件非空（PNG magic）；
//! 2. **服务模式**：HTTP 拉帧 200 + PNG magic + 非空；
//! 3. **scrub**：两个时间点拉到的帧不同（时间轴 scrub 语义）；
//! 4. **健壮性**：缺文件 / 语法错误 → 退出码 1（不静默）。
//!
//! 零第三方依赖：HTTP 客户端用 `std::net::TcpStream` 手写（服务端是我们
//! 自写的 HTTP/1.1，`Connection: close`，读到底即完整响应）。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// 最小合法 v2 示例（与 render_cli.rs 同源）：1s @ 8fps = 8 帧，96x54，
/// 相机环绕 + 文本淡入淡出 → 帧间必有差异（scrub 可观测）。
const MINIMAL_PRISM: &str = r#"// prism-dsl 1
movie "cli-min" { duration 1s fps 8 resolution 96x54 format png }
scene "s" {
  duration 1s
  stage3d {
    camera main { orbit { center 0,0,0 radius 6 } fov 45 }
    sdf ball { kind sphere radius 1.0 }
    material hero { albedo #e2b53f roughness 0.35 metalness 0.05 }
  }
  layer text title { content "CLI" x 48 y 24 size 14px color #ffffff opacity { key 0s 0 linear key 0.5s 1 linear key 1s 0 linear } }
}
"#;

/// 语法错误示例（未知顶层块 → 编译失败，带行号）。
const BAD_PRISM: &str = "this is not prism at all {{{\n";

/// 测试临时目录（按用例名区分，避免并行互踩）。
fn tmp_dir(case: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("prism_preview_test_{case}"));
    std::fs::create_dir_all(&dir).expect("临时目录创建应成功");
    dir
}

/// 写临时 .prism 文件并返回路径。
fn write_prism(case: &str, name: &str, source: &str) -> PathBuf {
    let path = tmp_dir(case).join(name);
    std::fs::write(&path, source).expect("临时 .prism 写入应成功");
    path
}

/// 找一个空闲端口（先绑后放；进程随后绑定同一端口）。
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("应能绑定临时端口");
    let port = listener.local_addr().expect("应能取到本地地址").port();
    drop(listener);
    port
}

/// 起 `prism-app preview <file> --addr 127.0.0.1:<port>` 子进程。
///
/// stderr **落盘**到 `server.stderr`（不丢）：全仓并行压测下服务若猝死
/// （OOM kill / abort），连接被拒时能拿到现场的 stderr 与退出状态——
/// 不静默原则对测试同样适用（否则一次负载抖动会伪装成"无法复现的玄学"）。
fn spawn_server(prism_path: &PathBuf, port: u16) -> (Child, PathBuf) {
    let stderr_path: PathBuf = tmp_dir("server").join("server.stderr");
    let stderr_file: std::fs::File =
        std::fs::File::create(&stderr_path).expect("stderr 日志文件应可创建");
    let child = Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args([
            "preview",
            prism_path.to_str().expect("路径应可转 str"),
            "--addr",
            &format!("127.0.0.1:{port}"),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(stderr_file))
        .spawn()
        .expect("预览服务进程应能启动");
    (child, stderr_path)
}

/// 等待服务可连（绑定在渲染之前，故应很快；给 30s 宽松上限）。
fn wait_connect(port: u16) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().expect("地址应可解析"),
            Duration::from_secs(2),
        ) {
            Ok(stream) => return stream,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("服务在 30s 内未就绪：{e}"),
        }
    }
}

/// 发一个 GET 请求，读回完整响应字节（服务端 Connection: close）。
///
/// 连接失败时**带诊断 panic**：子进程退出状态（若已死）+ stderr 尾部
/// （服务被 OOM kill / abort 时连接被拒，没有这些信息就是无头悬案）。
fn http_get(child: &mut Child, stderr_path: &Path, port: u16, target: &str) -> Vec<u8> {
    let mut stream = match TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().expect("地址应可解析"),
        Duration::from_secs(5),
    ) {
        Ok(stream) => stream,
        Err(e) => {
            let exit: Option<std::process::ExitStatus> = child.try_wait().ok().flatten();
            let stderr_tail: String = std::fs::read_to_string(stderr_path)
                .unwrap_or_default()
                .chars()
                .rev()
                .take(600)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            panic!(
                "应能连上预览服务（GET {target}）：{e}；子进程状态：{exit:?}；stderr 尾部：{stderr_tail}"
            );
        }
    };
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .expect("应能设置读超时");
    let request = format!("GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).expect("请求写入应成功");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).expect("响应读取应成功");
    buf
}

/// 从响应字节解析 (状态码, 正文)。
fn parse_response(raw: &[u8]) -> (u16, Vec<u8>) {
    let text = String::from_utf8_lossy(raw);
    let (head, _body) = text.split_once("\r\n\r\n").expect("响应应含头/体分隔");
    let status: u16 = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .expect("状态行应含状态码");
    // 正文按字节取（原始响应里正文在 \r\n\r\n 之后）。
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("响应应含分隔");
    (status, raw[split + 4..].to_vec())
}

/// PNG magic（8 字节）。
fn is_png(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[0..4] == [0x89, b'P', b'N', b'G']
}

#[test]
fn single_frame_mode_exits_zero_and_writes_nonempty_png() {
    // 验收 ④：单帧模式退出码 0 + 文件非空。
    let prism = write_prism("single", "min.prism", MINIMAL_PRISM);
    let out = tmp_dir("single").join("frame.png");
    let output = Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args([
            "preview",
            prism.to_str().expect("路径应可转 str"),
            "--frame",
            "0.5",
            "--out",
            out.to_str().expect("路径应可转 str"),
        ])
        .output()
        .expect("子进程应能执行");
    assert!(
        output.status.success(),
        "单帧模式应成功退出；stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).expect("输出文件应存在");
    assert!(!bytes.is_empty(), "输出文件不应为空");
    assert!(is_png(&bytes), "输出应为 PNG（magic 不符）：{:02X?}", &bytes[..bytes.len().min(8)]);
}

#[test]
fn server_serves_frame_200_png_and_scrub_differs() {
    // 验收 ②③：起服务 → HTTP 拉帧 200 + 非空；两个时间点的帧不同。
    let prism = write_prism("server", "min.prism", MINIMAL_PRISM);
    let port = free_port();
    let (child, stderr_path) = spawn_server(&prism, port);
    // 确保子进程最终被回收（即使断言失败）。
    let mut guard = ChildGuard(child);

    let _ = wait_connect(port);

    // /health：200（渲染中或就绪都可能，状态必须 200）。
    let (status, body) = parse_response(&http_get(guard.child_mut(), &stderr_path, port, "/health"));
    assert_eq!(status, 200, "/health 应 200");
    assert!(!body.is_empty(), "/health 正文不应为空");

    // /frame?t=0：200 + PNG magic + 非空（内部有界等待整片就绪）。
    let (status, frame0) =
        parse_response(&http_get(guard.child_mut(), &stderr_path, port, "/frame?t=0"));
    assert_eq!(status, 200, "/frame 应 200（就绪后）");
    assert!(!frame0.is_empty(), "帧字节不应为空");
    assert!(is_png(&frame0), "帧应为 PNG：{:02X?}", &frame0[..frame0.len().min(8)]);

    // scrub：t=0（文本不可见）vs t=0.5（文本满不透明 + 相机已环绕）→ 不同帧。
    let (status2, frame_mid) =
        parse_response(&http_get(guard.child_mut(), &stderr_path, port, "/frame?t=0.5"));
    assert_eq!(status2, 200, "/frame?t=0.5 应 200");
    assert!(is_png(&frame_mid), "t=0.5 帧应为 PNG");
    assert_ne!(frame0, frame_mid, "两个时间点的帧必须不同（scrub 语义）");

    // 超时长钳制到末帧：仍应 200 + PNG（不报错、不挂起）。
    let (status3, frame_last) =
        parse_response(&http_get(guard.child_mut(), &stderr_path, port, "/frame?t=99"));
    assert_eq!(status3, 200, "超时长请求应钳制后 200");
    assert!(is_png(&frame_last));

    // 未知路由：404。
    let (status4, _) = parse_response(&http_get(guard.child_mut(), &stderr_path, port, "/nope"));
    assert_eq!(status4, 404, "未知路由应 404");

    drop(guard); // 显式保留 guard 至测试结束
}

#[test]
fn missing_file_exits_nonzero() {
    // 缺文件：退出码非 0 + stderr 非空（不静默）。
    let output = Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args(["preview", "/nonexistent/demo.prism", "--frame", "0"])
        .output()
        .expect("子进程应能执行");
    assert!(!output.status.success(), "缺文件应失败退出");
    assert!(
        !output.stderr.is_empty(),
        "失败应有 stderr 说明（三段式）"
    );
}

#[test]
fn bad_syntax_exits_nonzero_with_line_info() {
    // 语法错误：退出码非 0 + stderr 含行号信息。
    let prism = write_prism("bad", "bad.prism", BAD_PRISM);
    let output = Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args(["preview", prism.to_str().expect("路径应可转 str"), "--frame", "0"])
        .output()
        .expect("子进程应能执行");
    assert!(!output.status.success(), "语法错误应失败退出");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains('行'), "应报告出错行号：{stderr}");
}

#[test]
fn help_flag_exits_zero() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args(["preview", "--help"])
        .output()
        .expect("子进程应能执行");
    assert!(output.status.success(), "--help 应退出码 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("/frame"), "帮助应说明路由：{stdout}");
}

/// 子进程 RAII 守卫：panic / 正常结束都 kill + wait，不遗留僵尸进程。
///
/// 按值持有 `Child`（而非借用）：测试中途要 `try_wait` 查退出状态做诊断，
/// 借用会让 `http_get(&mut child, …)` 与守卫的存活期打架。
struct ChildGuard(Child);

impl ChildGuard {
    /// 借出子进程（诊断用：连接失败时查它是否已死）。
    fn child_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
