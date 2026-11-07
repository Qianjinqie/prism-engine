//! V3.0 W3 **可复现盲测脚本**（§3.5 V3 / 主理人裁定）。
//!
//! # 用法
//!
//! ```text
//! cargo run -q -p prism-app --example w3_blind -- [report|control|critic|all]
//! ```
//!
//! - `report`（默认）：打印冻结协议 + **判官逐对输出（64 行）** + 胜率 / 样本数
//!   + 判定，并做两跑逐字节自查；
//! - `control`：反假绿负控（B 也用 `analytic`），证明判官不是无条件选 B；
//! - `critic`：全样本池 `prism-aesthetic::critic` 五维侧证（画质不退化，
//!   **不进胜率**）；
//! - `all`：三项全跑。
//!
//! # 反假绿 / 不事后调尺子
//!
//! 判官是确定性可计算函数（三条「解析性」指标），不是主观自评，也不是直接
//! 拿 critic 分数当胜率。样本池 / 种子 / 配对顺序 / 门槛全部冻结在
//! `prism-app/src/w3_blind.rs` 与 `docs/v3.0-盲测协议.md`，**跑测之前定桩**，
//! 本脚本不接收任何可调参数（没有 `--threshold`、没有 `--seed`），
//! 输出两跑逐字节一致。达不到门槛就如实打印 FAIL 并以退出码 1 终止，
//! 不改协议、不换尺子、不把 analytic 拉来凑数。
//!
//! # 零第三方 / 确定性
//!
//! 只依赖 workspace 内 crate 与 `std`；零 libm、零线程、零系统时间、
//! 零真随机。所有不规则走 `hash_mix` / `pcg::hash2_f32` 的种子化确定性噪声。

use std::process::ExitCode;

use prism_app::w3_blind::{
    critic_side_report, run_control, run_protocol, BLIND_FRAME_COUNT, BLIND_SAMPLE_COUNT,
    BLIND_STRENGTHS, BLIND_TAG, BLIND_THRESHOLD, BLIND_THRESHOLD_BASIS, CRITIC_REGRESSION_TOL,
};

fn frozen_header() -> String {
    let strengths: String = BLIND_STRENGTHS
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    format!(
        concat!(
            "=== V3.0 W3 成对盲测协议（冻结项，跑测之前定桩）===\n",
            "样本池 N            : {n} 对\n",
            "每对帧数            : {frames} 帧\n",
            "命名空间常量 BLIND_TAG: {tag:#018x}\n",
            "强度档（0..1000）   : [{strengths}]\n",
            "冻结门槛            : {threshold:.2}\n",
            "门槛依据            : {basis}\n",
            "配对顺序            : sample_params(index) 经 hash_mix 派生，判官不知标签\n",
            "判官                : 确定性可计算函数（二阶差分能量 / 局部方差谱离散度 / 帧间规律性）\n",
            "critic 五维         : 只作画质不退化侧证，容差 {tol:.2}，不进胜率\n",
        ),
        n = BLIND_SAMPLE_COUNT,
        frames = BLIND_FRAME_COUNT,
        tag = BLIND_TAG,
        strengths = strengths,
        threshold = BLIND_THRESHOLD,
        basis = BLIND_THRESHOLD_BASIS,
        tol = CRITIC_REGRESSION_TOL,
    )
}

/// `report`：主判据 + 逐对输出 + 两跑逐字节。
fn cmd_report() -> bool {
    print!("{}", frozen_header());
    let a = run_protocol();
    let b = run_protocol();
    let identical: bool = a.canonical_bytes() == b.canonical_bytes();
    println!("两跑逐字节一致    : {}", if identical { "是" } else { "否" });
    println!("胜率 / 样本数     : {:.4} / {}（B 胜 {}，平局 {}）", a.win_rate, a.total, a.b_wins, a.ties);
    println!("--- 判官逐对输出（{} 行）---", a.total);
    print!("{}", a.report_text());
    println!("--- 判定 ---");
    println!(
        "V3 盲测 : {}（胜率 {:.4} {} 冻结门槛 {:.2}；两跑逐字节 {}）",
        if a.passed && identical { "PASS" } else { "FAIL" },
        a.win_rate,
        if a.passed { "≥" } else { "<" },
        a.threshold,
        if identical { "一致" } else { "不一致" },
    );
    a.passed && identical
}

/// `control`：反假绿负控（B=A，判官应失去区分力）。
fn cmd_control() -> bool {
    let c = run_control();
    println!("=== 反假绿负控（B 也用 analytic，等价「字段没接 / 痕迹挪到帧后」）===");
    println!("胜率 / 样本数     : {:.4} / {}（B 胜 {}，平局 {}）", c.win_rate, c.total, c.b_wins, c.ties);
    println!("冻结门槛          : {:.2}", c.threshold);
    let no_power: bool = c.win_rate < c.threshold && c.ties > 0;
    println!("负控结论          : {}", if no_power { "判官无区分力（符合预期）" } else { "判官仍偏向 B（异常）" });
    no_power
}

/// `critic`：全池五维画质不退化侧证（不进胜率）。
fn cmd_critic() -> bool {
    let all = critic_side_report();
    let again = critic_side_report();
    let identical: bool = all
        .iter()
        .zip(again.iter())
        .all(|(a, b)| a.canonical_bytes() == b.canonical_bytes());
    let regressions: usize = all.iter().filter(|e| !e.no_regression).count();
    println!("=== critic 五维客观侧证（画质不退化，不进盲测胜率）===");
    println!("样本数            : {}", all.len());
    println!("两跑逐字节一致    : {}", if identical { "是" } else { "否" });
    println!("画质退化对数      : {regressions}");
    println!("--- 首对逐维输出 ---");
    if let Some(first) = all.first() {
        println!("{}", first.report_text());
    }
    let ok: bool = regressions == 0 && identical;
    println!("侧证结论          : {}", if ok { "画质不退化（PASS）" } else { "存在退化（FAIL）" });
    ok
}

fn main() -> ExitCode {
    let mode: String = std::env::args().nth(1).unwrap_or_else(|| "report".to_string());
    let ok: bool = match mode.as_str() {
        "report" => cmd_report(),
        "control" => cmd_control(),
        "critic" => cmd_critic(),
        "all" => {
            let a = cmd_report();
            println!();
            let b = cmd_control();
            println!();
            let c = cmd_critic();
            a && b && c
        }
        "-h" | "--help" => {
            println!("用法：cargo run -q -p prism-app --example w3_blind -- [report|control|critic|all]");
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("未知子命令 '{other}'（允许 report / control / critic / all）");
            return ExitCode::FAILURE;
        }
    };
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
