//! W4/W5 内存判据共用的量测工具：**峰值 RSS** + **子进程隔离** + **同规模网格基准臂**。
//!
//! # 为什么每个 arm 要一个独立子进程
//!
//! `VmHWM`（peak resident set size）是**进程级**的单调高水位，而 `cargo test` 在同一个
//! 进程里并行跑同一二进制的所有测试。两个后果都会让"对比"失真：
//!
//! 1. **并行污染**：A 测试的峰值会算进 B 测试的读数（哪怕 B 一个字节没多分配）。
//! 2. **释放不还页**：glibc 在 `free` 之后通常不把页还给内核。同一进程里先后跑两个
//!    arm 时，后跑的那个会复用前者留下的页 ⇒ 它的"增量"**系统性偏小**。
//!
//! 这两条都偏向"通过"——正是最坏的一种偏差。故同口径比较一律：**每个 arm 一个全新
//! 子进程**（同一个可执行文件 ⇒ 同一套分配器、同一份启动基线），父进程只读回数字并比较。
//!
//! # 量测窗口（`measure`）
//!
//! 1. 读 `VmRSS` 作起点；
//! 2. 写 `/proc/self/clear_refs = 5`（Linux ≥ 4.0）把 `VmHWM` 重置到**当前** RSS——
//!    子进程本来就接近冷启动，这一步是防止测试框架的前置分配把基线抬高；
//! 3. 跑被测代码；
//! 4. 读 `VmHWM` 作峰值、`VmRSS` 作收尾驻留。
//!
//! **增量 = `peak_kb − start_kb`**。`VmHWM` 单调，故窗口内的"分配→释放→再分配"
//! 交替不影响读到真实峰值；被测对象在窗口末尾被 drop 掉也不影响（高水位不会被回收拉低）。
//! 这正是"构建期瞬态"要度量的东西。
//!
//! 非 Linux ⇒ 所有读数返回 `None`，调用方必须打印"未测"并**跳过**断言（不假装通过）。

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use prism_geometry::bvh::Triangle;
use prism_math::Vec3;

/// 子进程侧读取的 arm 名（父进程通过它指定"这一趟只跑哪个 arm"）。
pub const ARM_ENV: &str = "PRISM_QA_RSS_ARM";

/// 同一二进制内的内存敏感测试**串行化**（子进程隔离之外的第二道保险：
/// 父进程里的其它测试不该在别人量测时分配大块内存）。
static RSS_LOCK: Mutex<()> = Mutex::new(());

/// 取串行锁；毒化（前一个测试 panic）时照常继续，因为这里保护的是**读数质量**而非数据。
pub fn serialize() -> MutexGuard<'static, ()> {
    RSS_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 从 `/proc/self/status` 读一个 `kB` 字段（如 `VmRSS:` / `VmHWM:`）。
fn status_kb(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            // 形如 "VmHWM:\t  123456 kB"
            return rest.trim().trim_end_matches("kB").trim().parse::<u64>().ok();
        }
    }
    None
}

/// 当前驻留集（KB）；非 Linux ⇒ `None`。
pub fn rss_kb() -> Option<u64> {
    status_kb("VmRSS:")
}

/// 进程峰值驻留集（KB）；非 Linux ⇒ `None`。
pub fn peak_rss_kb() -> Option<u64> {
    status_kb("VmHWM:")
}

/// 把 `VmHWM` 重置到当前 RSS（`clear_refs = 5`，Linux ≥ 4.0）。
fn reset_peak() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// 一次量测窗口的三点读数（KB）。
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    /// 窗口起点（重置后的高水位基线）。
    pub start_kb: u64,
    /// 窗口内峰值（`VmHWM`）。
    pub peak_kb: u64,
    /// 收尾驻留（`VmRSS`）——只作参考，不参与判据。
    pub end_kb: u64,
    /// `clear_refs` 是否重置成功（失败时基线退化为进程原有高水位，读数偏保守）。
    pub reset_ok: bool,
}

impl Peak {
    /// 判据用的量：窗口内的峰值增量（KB）。
    pub fn delta_kb(&self) -> u64 {
        self.peak_kb.saturating_sub(self.start_kb)
    }
}

/// 在**当前进程**里量一段代码的峰值 RSS 增量。
///
/// 返回值是被测代码的返回值 + 量测结果（无 `/proc` ⇒ `None`）。
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Option<Peak>) {
    let start = rss_kb();
    let reset_ok = reset_peak();
    // 重置之后读 `VmHWM` 才是本窗口的基线；重置失败则退回起点 RSS（保守）。
    let base = peak_rss_kb().or(start);
    let value = f();
    let peak = match (base, peak_rss_kb(), rss_kb()) {
        (Some(b), Some(p), Some(e)) => {
            Some(Peak { start_kb: b, peak_kb: p.max(b), end_kb: e, reset_ok })
        }
        _ => None,
    };
    (value, peak)
}

/// 一个 arm 的量测结果（从子进程的 `@RSS` 行解析而来）。
#[derive(Debug, Clone)]
pub struct Arm {
    /// arm 名（`strand` / `mesh` / `subdiv:5` …）。
    pub arm: String,
    /// 该臂产出的三角形数（两臂必须相等，否则不是同规模）。
    pub tris: usize,
    /// 峰值量测；`None` ⇒ 本机量不了（必须打印"未测"）。
    pub peak: Option<Peak>,
    /// 人类可读的附加说明（产物字节、输入规模等）。
    pub note: String,
}

/// 子进程侧：打印一行可被 [`run_arm`] 解析的记录。
pub fn report_arm(arm: &str, tris: usize, peak: Option<Peak>, note: &str) {
    match peak {
        Some(p) => println!(
            "@RSS arm={arm} tris={tris} start_kb={} peak_kb={} end_kb={} reset_ok={} note={note}",
            p.start_kb,
            p.peak_kb,
            p.end_kb,
            u8::from(p.reset_ok)
        ),
        None => println!("@RSS arm={arm} tris={tris} unmeasured=1 note={note}"),
    }
}

/// 父进程侧：在**新进程**里只跑 `test_name` 这一个测试的指定 arm，读回 `@RSS` 行。
///
/// 子进程用 `--exact test_name --nocapture` 过滤（`--nocapture` 必需：否则 libtest
/// 会把 `@RSS` 行吞进捕获缓冲，父进程读不到）。
pub fn run_arm(test_name: &str, arm: &str) -> Arm {
    let exe = std::env::current_exe().expect("取不到测试可执行文件路径");
    let out = std::process::Command::new(&exe)
        .args(["--exact", test_name, "--nocapture"])
        .env(ARM_ENV, arm)
        .output()
        .unwrap_or_else(|e| panic!("拉起子进程 {exe:?}（arm={arm}）失败：{e}"));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "子进程 arm={arm} 退出码 {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status.code()
    );
    parse_arm(&stdout).unwrap_or_else(|| {
        panic!(
            "子进程 arm={arm} 没有打印 @RSS 行\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    })
}

/// 解析子进程输出里的 `@RSS` 行（`note=` 之后是自由文本）。
fn parse_arm(stdout: &str) -> Option<Arm> {
    let line = stdout.lines().find(|l| l.starts_with("@RSS "))?;
    let body = &line["@RSS ".len()..];
    let (head, note) = match body.split_once(" note=") {
        Some((h, n)) => (h, n.to_string()),
        None => (body, String::new()),
    };
    let mut kv: BTreeMap<&str, &str> = BTreeMap::new();
    for tok in head.split_whitespace() {
        let (k, v) = tok.split_once('=')?;
        kv.insert(k, v);
    }
    let arm = kv.get("arm")?.to_string();
    let tris: usize = kv.get("tris")?.parse().ok()?;
    let peak = if kv.contains_key("unmeasured") {
        None
    } else {
        Some(Peak {
            start_kb: kv.get("start_kb")?.parse().ok()?,
            peak_kb: kv.get("peak_kb")?.parse().ok()?,
            end_kb: kv.get("end_kb")?.parse().ok()?,
            reset_ok: *kv.get("reset_ok")? == "1",
        })
    };
    Some(Arm { arm, tris, peak, note })
}

/// 规则网格三角化：`cells_x × cells_z` 个单位 cell，每 cell 2 个三角形。
///
/// 这是"**同规模三角网**"的基准臂——结构上就是一张普通索引网格（与 glTF 导入的地形/
/// 平面同形），用来回答"同样多的三角形、走同一条 `Bvh::build`，普通网格付多少驻留"。
/// 平面网格对 SAH 无害（沿 X/Z 两轴可正常二分，非退化输入）。
pub fn grid_triangles(cells_x: usize, cells_z: usize) -> Vec<Triangle> {
    let mut out: Vec<Triangle> = Vec::with_capacity(cells_x * cells_z * 2);
    for j in 0..cells_z {
        for i in 0..cells_x {
            let (x0, z0) = (i as f32, j as f32);
            let (x1, z1) = (x0 + 1.0, z0 + 1.0);
            let p00 = Vec3::new(x0, 0.0, z0);
            let p10 = Vec3::new(x1, 0.0, z0);
            let p11 = Vec3::new(x1, 0.0, z1);
            let p01 = Vec3::new(x0, 0.0, z1);
            out.push(Triangle { a: p00, b: p10, c: p11 });
            out.push(Triangle { a: p00, b: p11, c: p01 });
        }
    }
    out
}
