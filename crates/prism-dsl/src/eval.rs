//! V3.0 W1 确定性受限子集执行器。
//!
//! 冻结依据 `docs/v3.0-语言规格.md` §3（白名单）、§4（求值顺序 F-1~F-5）、
//! §5（种子方案 F-6~F-10）、§6（`det_*` 覆盖表）。
//!
//! - 数值域是 **Q32.32 定点**（`Fixed3`）：字面量解析与四则运算全为整数，
//!   零浮点舍入、零 libm；超越函数经 `prism_math::det_scalar` 的零 libm 实现，
//!   结果量化回定点。
//! - 唯一随机源 `prism_core::eval::hash_mix`，不新造随机源。
//! - 每个求值步骤产生 `EvalEvent`，按真实求值顺序进 `trace`；`trace` 与
//!   `lowered_source` 一起进 `artifact_hash`（F-8），换序必然可检出。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::eval::{hash_mix, hash_mix_bytes, hash_mix_unit_f64};
use prism_math::{colorspace_registry, det_scalar};

use crate::ast::v3::{
    BinOp3, Expr3, ForRange3, InterpPart3, Literal3, Program3, Stmt3, UnOp3,
};
use crate::{v3_err, DslError};

/// Q32.32 定标因子。
pub const SCALE: i64 = 1_i64 << 32;

/// `for` 展开步数静态上限（D-23：编译期可判定、必然终止）。
pub const MAX_EXPAND: usize = 4096;

/// 产物哈希的 schema 标签（换规格即换哈希，避免跨版本误判）。
const SCHEMA_TAG: u64 = 0x5633_5f57_315f_3031;

/// `rand(tag)` 的命名空间标签（与场景节点哈希分流）。
const RAND_NAMESPACE: u64 = 0x7261_6e64_5f76_335f;

/// 求值轨迹事件（F-5）。`tag` 是节点类型常量，顺序敏感由 `hash_mix` 链保证。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalEvent {
    pub line: usize,
    pub tag: u64,
    pub value_bits: u64,
}

// 节点类型标签（互不相同的常量）。
const TAG_LIT: u64 = 0x0101;
const TAG_VAR: u64 = 0x0102;
const TAG_UNARY: u64 = 0x0103;
const TAG_BINARY: u64 = 0x0104;
const TAG_CALL: u64 = 0x0105;
const TAG_INDEX: u64 = 0x0106;
const TAG_LIST: u64 = 0x0107;
const TAG_DICT: u64 = 0x0108;
const TAG_INTERP: u64 = 0x0109;
const TAG_RAND: u64 = 0x010a;

/// FNV-1a 64（字符串 / 字典键 → u64，纯整数）。
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Q32.32 定点数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed3(pub i64);

impl Fixed3 {
    pub const ZERO: Fixed3 = Fixed3(0);
    pub const ONE: Fixed3 = Fixed3(SCALE);

    /// 十进制字面量 → 定点（纯整数解析，不做任何浮点转换）。
    pub fn from_text(text: &str) -> Option<Self> {
        let (neg, body): (bool, &str) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let (int_s, frac_s): (&str, &str) = match body.split_once('.') {
            Some((a, b)) => (a, b),
            None => (body, ""),
        };
        if int_s.is_empty()
            || !int_s.bytes().all(|b| b.is_ascii_digit())
            || !frac_s.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let int_part: i128 = int_s.parse::<i128>().ok()?;
        let mut frac_num: i128 = 0;
        let mut denom: i128 = 1;
        // 最多吃 18 位小数，超出部分截断（口径固定，确定性）。
        for b in frac_s.bytes().take(18) {
            frac_num = frac_num.checked_mul(10)?.checked_add(i128::from(b - b'0'))?;
            denom = denom.checked_mul(10)?;
        }
        let unit: i128 = 1_i128 << 32;
        let scaled: i128 = int_part
            .checked_mul(unit)?
            .checked_add(frac_num.checked_mul(unit)? / denom)?;
        let value: i128 = if neg { -scaled } else { scaled };
        i64::try_from(value).ok().map(Fixed3)
    }

    /// f64 → 定点（`det_scalar::det_round` 是纯位操作，不调 libm）。
    pub fn from_f64(x: f64, line: usize) -> Result<Self, DslError> {
        if !x.is_finite() {
            return Err(v3_err(
                line,
                "P7003",
                "超越函数结果不是有限值（定义域越界）",
                "检查函数输入范围，例如 sqrt / ln 不接受负数",
            ));
        }
        let scaled: f64 = det_scalar::det_round(x * (SCALE as f64));
        if scaled > i64::MAX as f64 || scaled < i64::MIN as f64 {
            return Err(v3_err(
                line,
                "P7003",
                "求值结果超出 Q32.32 定点范围",
                "缩小中间量；定点整数部分上限约 ±21 亿",
            ));
        }
        Ok(Fixed3(scaled as i64))
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    /// 规范十进制文本（用于 `{{...}}` 插值进 v2 块文本；纯整数格式化）。
    pub fn to_text(self) -> String {
        let neg: bool = self.0 < 0;
        let abs: u128 = (self.0 as i128).unsigned_abs();
        let int_part: u128 = abs >> 32;
        let frac: u128 = abs & 0xffff_ffff;
        let mut body: String = if frac == 0 {
            format!("{}", int_part)
        } else {
            // frac < 2^32 ⇒ frac*1e8/2^32 < 1e8，恰为 8 位十进制小数。
            let mut digits: String = format!("{:08}", (frac * 100_000_000_u128) >> 32);
            while digits.ends_with('0') {
                digits.pop();
            }
            format!("{}.{}", int_part, digits)
        };
        if neg {
            body.insert(0, '-');
        }
        body
    }

    fn arithmetic(self, rhs: Self, op: BinOp3, line: usize) -> Result<Self, DslError> {
        let a: i128 = i128::from(self.0);
        let b: i128 = i128::from(rhs.0);
        let unit: i128 = i128::from(SCALE);
        let raw: Option<i128> = match op {
            BinOp3::Add => a.checked_add(b),
            BinOp3::Sub => a.checked_sub(b),
            BinOp3::Mul => a.checked_mul(b).map(|v| v / unit),
            BinOp3::Div => {
                if b == 0 {
                    return Err(v3_err(
                        line,
                        "P7003",
                        "除零：右侧操作数求值为 0",
                        "保证除数非 0；可先用 if 判断",
                    ));
                }
                a.checked_mul(unit).map(|v| v / b)
            }
            BinOp3::Mod => {
                if b == 0 {
                    return Err(v3_err(
                        line,
                        "P7003",
                        "取模的右侧操作数求值为 0",
                        "保证模数非 0；可先用 if 判断",
                    ));
                }
                Some(a % b)
            }
            _ => None,
        };
        match raw {
            Some(v) => match i64::try_from(v) {
                Ok(x) => Ok(Fixed3(x)),
                Err(_) => Err(v3_err(
                    line,
                    "P7003",
                    "定点算术溢出（结果超出 i64）",
                    "缩小中间量；Q32.32 整数部分上限约 ±21 亿",
                )),
            },
            None => Err(v3_err(
                line,
                "P7003",
                "定点算术溢出",
                "缩小中间量或拆分表达式",
            )),
        }
    }
}

/// 运行时值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value3 {
    Num(Fixed3),
    Bool(bool),
    Str(String),
    List(Vec<Value3>),
    Dict(Vec<(String, Value3)>),
}

impl Value3 {
    /// 规范位型（进 `EvalEvent.value_bits`；不同值尽量不同，且完全确定）。
    pub fn bits(&self) -> u64 {
        match self {
            Value3::Num(f) => f.0 as u64,
            Value3::Bool(true) => 0x9e37_79b9_7f4a_7c15,
            Value3::Bool(false) => 0x517c_c1b7_2722_0a95,
            Value3::Str(s) => fnv1a(s.as_bytes()),
            Value3::List(items) => items.iter().fold(0x6c69_7374_5f76_3301, |h, v| {
                hash_mix(h, v.bits())
            }),
            Value3::Dict(entries) => {
                entries.iter().fold(0x6469_6374_5f76_3301, |h, (k, v)| {
                    hash_mix(hash_mix(h, fnv1a(k.as_bytes())), v.bits())
                })
            }
        }
    }

    /// 插值输出文本（v2 块文本能直接消费）。
    pub fn to_text(&self) -> String {
        match self {
            Value3::Num(f) => f.to_text(),
            Value3::Bool(true) => "true".to_string(),
            Value3::Bool(false) => "false".to_string(),
            Value3::Str(s) => s.clone(),
            Value3::List(items) => {
                let parts: Vec<String> = items.iter().map(Value3::to_text).collect();
                format!("[{}]", parts.join(", "))
            }
            Value3::Dict(entries) => {
                let parts: Vec<String> = entries
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_text()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }

    fn as_num(&self, line: usize, what: &str) -> Result<Fixed3, DslError> {
        match self {
            Value3::Num(f) => Ok(*f),
            _ => Err(v3_err(
                line,
                "P7003",
                &format!("'{}' 需要数值参数，却得到 {}", what, self.to_text()),
                "传入数字字面量或求值为数字的变量",
            )),
        }
    }
}

/// 一次完整求值的产物。
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub seed: u64,
    pub bindings: BTreeMap<String, Value3>,
    pub trace: Vec<EvalEvent>,
    /// `emit` 拼接出的 v2 块文本（**不含**首行版本头）。
    pub lowered_source: String,
    /// `lowered_source` 每行对应的 v3 源文件行号（降级报错行号回折用）。
    pub lowered_lines: Vec<usize>,
    pub artifact_hash: u64,
}

// ---------------------------------------------------------------------------
// 内置函数白名单（§3.1 / §6）
// ---------------------------------------------------------------------------

/// 内置函数名 → 参数个数。
const BUILTINS: &[(&str, usize)] = &[
    ("abs", 1),
    ("sign", 1),
    ("min", 2),
    ("max", 2),
    ("clamp", 3),
    ("floor", 1),
    ("ceil", 1),
    ("trunc", 1),
    ("round", 1),
    ("sqrt", 1),
    ("pow", 2),
    ("exp", 1),
    ("ln", 1),
    ("log2", 1),
    ("log10", 1),
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("acos", 1),
    ("asin", 1),
    ("atan", 1),
    ("atan2", 2),
    ("lerp", 3),
    ("len", 1),
    ("int", 1),
    ("str", 1),
    ("rand", 1),
];

/// 非法语义黑名单（§3.2）——命中即给专门的 P6101 诊断，不落到「未定义函数」。
const FORBIDDEN: &[&str] = &[
    "now", "time", "clock", "date", "http_get", "fetch", "socket", "url", "write_file",
    "read_file", "open", "save", "env", "getenv", "environ", "print", "input", "stdin", "exec",
    "sleep", "thread", "spawn", "system",
];

fn builtin_arity(name: &str) -> Option<usize> {
    BUILTINS.iter().find(|(n, _)| *n == name).map(|(_, a)| *a)
}

fn is_forbidden(name: &str) -> bool {
    FORBIDDEN.contains(&name) || name == "while"
}

// ---------------------------------------------------------------------------
// 静态校验（先于任何求值，§3.2 全部带行号 + 三段式）
// ---------------------------------------------------------------------------

/// 静态校验入口：`validate` 通过后才允许求值。
pub fn validate(program: &Program3) -> Result<u64, DslError> {
    let mut seed: Option<u64> = None;
    let mut emits: usize = 0;
    for stmt in &program.stmts {
        match stmt {
            Stmt3::Seed { value, line } => {
                if seed.is_some() {
                    return Err(v3_err(
                        *line,
                        "P6111",
                        "seed 重复声明",
                        "一份文件只能写一条 seed；删掉多余的行",
                    ));
                }
                if value.contains('.') {
                    return Err(v3_err(
                        *line,
                        "P6112",
                        "seed 必须是非负整数字面量（不能带小数）",
                        "写成 seed = 20261106",
                    ));
                }
                let parsed: u64 = value.parse::<u64>().map_err(|_| {
                    v3_err(
                        *line,
                        "P6113",
                        &format!("seed '{}' 超出 u64 范围", value),
                        "写成 0 ..= 18446744073709551615 之间的十进制整数",
                    )
                })?;
                seed = Some(parsed);
            }
            Stmt3::Emit { .. } => emits += 1,
            _ => {}
        }
    }
    let seed: u64 = seed.ok_or_else(|| {
        v3_err(
            1,
            "P6110",
            "缺少一等字段 seed（种子不许取默认）",
            "在文件顶层写一条 seed = <非负整数>，例如 seed = 42",
        )
    })?;
    if emits == 0 {
        return Err(v3_err(
            program.line_count,
            "P7001",
            "文件里没有任何 emit 语句",
            "至少写一条 emit f\"\"\"...\"\"\" 把计算出的场景声明展开为 v2 块文本",
        ));
    }

    let mut defs: BTreeMap<String, (Vec<String>, Vec<Stmt3>)> = BTreeMap::new();
    collect_defs(&program.stmts, &mut defs)?;
    check_call_graph(&defs)?;

    let mut declared: BTreeSet<String> = BTreeSet::new();
    declared.insert("seed".to_string());
    validate_stmts(&program.stmts, &mut declared, false, &defs)?;
    Ok(seed)
}

fn collect_defs(
    stmts: &[Stmt3],
    defs: &mut BTreeMap<String, (Vec<String>, Vec<Stmt3>)>,
) -> Result<(), DslError> {
    for stmt in stmts {
        if let Stmt3::Def { name, params, body, line } = stmt {
            if defs.contains_key(name) || builtin_arity(name).is_some() {
                return Err(v3_err(
                    *line,
                    "P6103",
                    &format!("def '{}' 与已有函数 / 内置函数重名", name),
                    "换一个函数名；内置函数名不可覆盖",
                ));
            }
            defs.insert(name.clone(), (params.clone(), body.clone()));
            // def 体内也允许嵌套 def（仍受调用图与声明序约束）。
            collect_defs(body, defs)?;
        }
        if let Stmt3::For { body, .. } = stmt {
            collect_defs(body, defs)?;
        }
        if let Stmt3::If { branches, else_body, .. } = stmt {
            for (_, body) in branches {
                collect_defs(body, defs)?;
            }
            collect_defs(else_body, defs)?;
        }
    }
    Ok(())
}

/// 调用图无环检查（D-23：无界递归静态拒绝）。
fn check_call_graph(defs: &BTreeMap<String, (Vec<String>, Vec<Stmt3>)>) -> Result<(), DslError> {
    fn calls(stmts: &[Stmt3], out: &mut Vec<(String, usize)>) {
        for stmt in stmts {
            match stmt {
                Stmt3::Assign { expr, .. }
                | Stmt3::Return { expr, .. }
                | Stmt3::Emit { template: expr, .. }
                | Stmt3::Expr { expr, .. } => {
                    collect_calls(expr, out);
                }
                Stmt3::Def { body, .. } | Stmt3::For { body, .. } => calls(body, out),
                Stmt3::If { branches, else_body, .. } => {
                    for (cond, body) in branches {
                        collect_calls(cond, out);
                        calls(body, out);
                    }
                    calls(else_body, out);
                }
                Stmt3::Seed { .. } => {}
            }
        }
    }
    fn collect_calls(expr: &Expr3, out: &mut Vec<(String, usize)>) {
        match expr {
            Expr3::Call { name, args, line } => {
                out.push((name.clone(), *line));
                for a in args {
                    collect_calls(a, out);
                }
            }
            Expr3::Unary { rhs, .. } => collect_calls(rhs, out),
            Expr3::Binary { lhs, rhs, .. } | Expr3::Index { target: lhs, index: rhs, .. } => {
                collect_calls(lhs, out);
                collect_calls(rhs, out);
            }
            Expr3::List { items, .. } => {
                for i in items {
                    collect_calls(i, out);
                }
            }
            Expr3::Dict { entries, .. } => {
                for (_, v) in entries {
                    collect_calls(v, out);
                }
            }
            Expr3::Interp { parts, .. } => {
                for p in parts {
                    if let InterpPart3::Expr(e) = p {
                        collect_calls(e, out);
                    }
                }
            }
            Expr3::Lit { .. } | Expr3::Var { .. } => {}
        }
    }

    for (name, (_, body)) in defs {
        let mut edges: Vec<(String, usize)> = Vec::new();
        calls(body, &mut edges);
        for (callee, line) in edges {
            if !defs.contains_key(&callee) {
                continue;
            }
            let mut stack: Vec<String> = vec![callee.clone()];
            let mut seen: BTreeSet<String> = BTreeSet::new();
            while let Some(cur) = stack.pop() {
                if cur == *name {
                    return Err(v3_err(
                        line,
                        "P6105",
                        &format!("def '{}' 的调用链存在环（经 '{}'），不允许递归", name, callee),
                        "受限子集不做图灵完备；把循环改写成 for 的字面量区间展开",
                    ));
                }
                if !seen.insert(cur.clone()) {
                    continue;
                }
                if let Some((_, cbody)) = defs.get(&cur) {
                    let mut inner: Vec<(String, usize)> = Vec::new();
                    calls(cbody, &mut inner);
                    for (n, _) in inner {
                        if defs.contains_key(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_stmts(
    stmts: &[Stmt3],
    declared: &mut BTreeSet<String>,
    in_def: bool,
    defs: &BTreeMap<String, (Vec<String>, Vec<Stmt3>)>,
) -> Result<(), DslError> {
    for stmt in stmts {
        match stmt {
            Stmt3::Seed { .. } => {}
            Stmt3::Return { expr, line } => {
                if !in_def {
                    return Err(v3_err(
                        *line,
                        "P6104",
                        "return 只能出现在 def 函数体内",
                        "把 return 移进 def 块，或改用顶层变量赋值",
                    ));
                }
                validate_expr(expr, declared, defs)?;
            }
            Stmt3::Assign { name, expr, line } => {
                if name == "seed" {
                    return Err(v3_err(
                        *line,
                        "P6102",
                        "seed 是一等字段，不能被重新赋值",
                        "seed 只在顶层声明一次；需要派生值请换一个变量名",
                    ));
                }
                validate_expr(expr, declared, defs)?;
                declared.insert(name.clone());
            }
            Stmt3::Def { params, body, .. } => {
                let mut local: BTreeSet<String> = declared.clone();
                for p in params {
                    local.insert(p.clone());
                }
                let mut has_return: bool = false;
                for s in body {
                    if matches!(s, Stmt3::Return { .. }) {
                        has_return = true;
                    }
                }
                if !has_return {
                    return Err(v3_err(
                        stmt.line(),
                        "P6104",
                        "def 函数体缺少 return",
                        "在函数体末尾写一条 return <表达式>",
                    ));
                }
                validate_stmts(body, &mut local, true, defs)?;
            }
            Stmt3::For { var, range, body, line } => {
                let steps: usize = validate_range(range, *line)?;
                if steps == 0 {
                    return Err(v3_err(
                        *line,
                        "P6107",
                        "for 区间为空（上界不大于下界）",
                        "写成上界大于下界的区间，例如 0..4",
                    ));
                }
                let mut local: BTreeSet<String> = declared.clone();
                local.insert(var.clone());
                validate_stmts(body, &mut local, in_def, defs)?;
            }
            Stmt3::If { branches, else_body, .. } => {
                for (cond, body) in branches {
                    validate_expr(cond, declared, defs)?;
                    let mut local: BTreeSet<String> = declared.clone();
                    validate_stmts(body, &mut local, in_def, defs)?;
                }
                let mut local: BTreeSet<String> = declared.clone();
                validate_stmts(else_body, &mut local, in_def, defs)?;
            }
            Stmt3::Emit { template, .. } => validate_expr(template, declared, defs)?,
            Stmt3::Expr { expr, .. } => validate_expr(expr, declared, defs)?,
        }
    }
    Ok(())
}

/// `for` 上界编译期可判定（D-23 / F-3），返回展开步数。
fn validate_range(range: &ForRange3, line: usize) -> Result<usize, DslError> {
    match range {
        ForRange3::Range { lo, hi, .. } => {
            if lo.contains('.') || hi.contains('.') {
                return Err(v3_err(
                    line,
                    "P6106",
                    "for 区间上下界必须是整数字面量",
                    "写成 for i in 0..4: 这样的字面量区间",
                ));
            }
            let a: i64 = lo.parse::<i64>().map_err(|_| {
                v3_err(line, "P6106", "for 下界不是可判定的整数字面量", "写成 0..4 这样的字面量区间")
            })?;
            let b: i64 = hi.parse::<i64>().map_err(|_| {
                v3_err(line, "P6106", "for 上界不是可判定的整数字面量", "写成 0..4 这样的字面量区间")
            })?;
            if b <= a {
                return Err(v3_err(
                    line,
                    "P6107",
                    &format!("for 区间 {}..{} 上界不大于下界", lo, hi),
                    "让上界大于下界；空循环请删掉该块",
                ));
            }
            let steps: usize = (b - a) as usize;
            if steps > MAX_EXPAND {
                return Err(v3_err(
                    line,
                    "P6106",
                    &format!("for 展开 {} 步，超过静态上限 {}", steps, MAX_EXPAND),
                    "拆成多个小区间；受限子集要求展开步数编译期可判定",
                ));
            }
            Ok(steps)
        }
        ForRange3::List { items, .. } => {
            for item in items {
                if !matches!(item, Expr3::Lit { .. }) {
                    return Err(v3_err(
                        item.line(),
                        "P6106",
                        "for 的列表迭代源只允许字面量元素",
                        "写成 for x in [1, 2, 3]: 这样的字面量列表",
                    ));
                }
            }
            if items.len() > MAX_EXPAND {
                return Err(v3_err(
                    line,
                    "P6106",
                    "for 列表元素数超过静态上限",
                    "拆成多个 for 块",
                ));
            }
            Ok(items.len())
        }
    }
}

fn validate_expr(
    expr: &Expr3,
    declared: &BTreeSet<String>,
    defs: &BTreeMap<String, (Vec<String>, Vec<Stmt3>)>,
) -> Result<(), DslError> {
    match expr {
        Expr3::Lit { .. } => Ok(()),
        Expr3::Var { name, line } => {
            if declared.contains(name) {
                Ok(())
            } else {
                Err(v3_err(
                    *line,
                    "P6102",
                    &format!("引用了未声明的变量 '{}'", name),
                    "变量必须先赋值再使用；检查拼写与声明顺序",
                ))
            }
        }
        Expr3::Unary { rhs, .. } => validate_expr(rhs, declared, defs),
        Expr3::Binary { lhs, rhs, .. } | Expr3::Index { target: lhs, index: rhs, .. } => {
            validate_expr(lhs, declared, defs)?;
            validate_expr(rhs, declared, defs)
        }
        Expr3::Call { name, args, line } => {
            if is_forbidden(name) {
                return Err(v3_err(
                    *line,
                    "P6101",
                    &format!("'{}' 属于受限子集禁止的语义（时间 / 网络 / 文件 / 环境变量 / 通用 IO）", name),
                    "确定性语言不读外部状态；改用 seed + rand(tag) 或字面量",
                ));
            }
            let arity: usize = if let Some(a) = builtin_arity(name) {
                a
            } else if let Some((params, _)) = defs.get(name) {
                params.len()
            } else {
                return Err(v3_err(
                    *line,
                    "P6101",
                    &format!("未定义的函数 '{}'", name),
                    "只支持 §6 det_* 映射表与 len / int / str / rand；或先写 def 定义",
                ));
            };
            if args.len() != arity {
                return Err(v3_err(
                    *line,
                    "P6101",
                    &format!("'{}' 需要 {} 个参数，却给了 {} 个", name, arity, args.len()),
                    "按函数签名补齐参数；不取默认、不静默降级",
                ));
            }
            for a in args {
                validate_expr(a, declared, defs)?;
            }
            Ok(())
        }
        Expr3::List { items, .. } => {
            for i in items {
                validate_expr(i, declared, defs)?;
            }
            Ok(())
        }
        Expr3::Dict { entries, .. } => {
            for (_, v) in entries {
                validate_expr(v, declared, defs)?;
            }
            Ok(())
        }
        Expr3::Interp { parts, .. } => {
            for p in parts {
                if let InterpPart3::Expr(e) = p {
                    validate_expr(e, declared, defs)?;
                }
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// 求值（F-1 左→右深度优先；F-5 每步一条 EvalEvent）
// ---------------------------------------------------------------------------

enum Control3 {
    Normal,
    Return(Value3),
}

struct EvalCtx {
    seed: u64,
    ordinal: u64,
    vars: BTreeMap<String, Value3>,
    defs: BTreeMap<String, (Vec<String>, Vec<Stmt3>)>,
    trace: Vec<EvalEvent>,
    lowered: String,
    lowered_lines: Vec<usize>,
}

impl EvalCtx {
    fn push(&mut self, line: usize, tag: u64, value: &Value3) {
        self.ordinal += 1;
        self.trace.push(EvalEvent {
            line,
            tag,
            value_bits: value.bits(),
        });
    }
}

/// 执行整份程序（`validate` 必须先通过）。
pub fn eval_program(program: &Program3) -> Result<EvalOutcome, DslError> {
    let seed: u64 = validate(program)?;
    let mut defs: BTreeMap<String, (Vec<String>, Vec<Stmt3>)> = BTreeMap::new();
    collect_defs(&program.stmts, &mut defs)?;
    let mut ctx = EvalCtx {
        seed,
        ordinal: 0,
        vars: BTreeMap::new(),
        defs,
        trace: Vec::new(),
        lowered: String::new(),
        lowered_lines: Vec::new(),
    };
    let _: Control3 = exec_stmts(&program.stmts, &mut ctx)?;
    let mut h: u64 = hash_mix(seed, SCHEMA_TAG);
    h = hash_mix(h, ctx.lowered.len() as u64);
    h = hash_mix_bytes(h, ctx.lowered.as_bytes());
    for ev in &ctx.trace {
        h = hash_mix(h, ev.line as u64);
        h = hash_mix(h, ev.tag);
        h = hash_mix(h, ev.value_bits);
    }
    Ok(EvalOutcome {
        seed,
        bindings: ctx.vars,
        trace: ctx.trace,
        lowered_source: ctx.lowered,
        lowered_lines: ctx.lowered_lines,
        artifact_hash: h,
    })
}

fn exec_stmts(stmts: &[Stmt3], ctx: &mut EvalCtx) -> Result<Control3, DslError> {
    for stmt in stmts {
        match stmt {
            Stmt3::Seed { .. } => {}
            Stmt3::Assign { name, expr, line } => {
                let value: Value3 = eval_expr(expr, ctx)?;
                let _ = line;
                ctx.vars.insert(name.clone(), value);
            }
            Stmt3::Def { .. } => {}
            Stmt3::Return { expr, .. } => {
                let value: Value3 = eval_expr(expr, ctx)?;
                return Ok(Control3::Return(value));
            }
            Stmt3::For { var, range, body, line } => {
                let values: Vec<Value3> = expand_range(range, *line)?;
                for v in values {
                    ctx.vars.insert(var.clone(), v);
                    if let Control3::Return(x) = exec_stmts(body, ctx)? {
                        return Ok(Control3::Return(x));
                    }
                }
            }
            Stmt3::If { branches, else_body, .. } => {
                let mut taken = false;
                for (cond, body) in branches {
                    let c: Value3 = eval_expr(cond, ctx)?;
                    let truthy: bool = match c {
                        Value3::Bool(b) => b,
                        Value3::Num(f) => f.0 != 0,
                        _ => {
                            return Err(v3_err(
                                cond.line(),
                                "P7003",
                                "if 条件必须是布尔或数值",
                                "写成 x > 0 这样的比较表达式",
                            ));
                        }
                    };
                    if truthy {
                        taken = true;
                        if let Control3::Return(x) = exec_stmts(body, ctx)? {
                            return Ok(Control3::Return(x));
                        }
                        break;
                    }
                }
                if !taken {
                    if let Control3::Return(x) = exec_stmts(else_body, ctx)? {
                        return Ok(Control3::Return(x));
                    }
                }
            }
            Stmt3::Expr { expr, .. } => {
                let _: Value3 = eval_expr(expr, ctx)?;
            }
            Stmt3::Emit { template, line } => {
                let value: Value3 = eval_expr(template, ctx)?;
                let text: String = value.to_text();
                let start_line: usize = match template {
                    Expr3::Interp { start_line, .. } => *start_line,
                    _ => *line,
                };
                for (k, line_text) in text.split('\n').enumerate() {
                    ctx.lowered_lines.push(start_line + k);
                    ctx.lowered.push_str(line_text);
                    ctx.lowered.push('\n');
                }
            }
        }
    }
    Ok(Control3::Normal)
}

fn expand_range(range: &ForRange3, line: usize) -> Result<Vec<Value3>, DslError> {
    match range {
        ForRange3::Range { lo, hi, .. } => {
            let a: i64 = lo.parse::<i64>().map_err(|_| {
                v3_err(line, "P6106", "for 下界无法解析", "写成 0..4 这样的字面量区间")
            })?;
            let b: i64 = hi.parse::<i64>().map_err(|_| {
                v3_err(line, "P6106", "for 上界无法解析", "写成 0..4 这样的字面量区间")
            })?;
            let mut out: Vec<Value3> = Vec::with_capacity((b - a) as usize);
            for i in a..b {
                out.push(Value3::Num(Fixed3(i * SCALE)));
            }
            Ok(out)
        }
        ForRange3::List { items, .. } => {
            let mut out: Vec<Value3> = Vec::with_capacity(items.len());
            for item in items {
                out.push(match item {
                    Expr3::Lit { value: Literal3::Num(s), line } => Value3::Num(
                        Fixed3::from_text(s).ok_or_else(|| {
                            v3_err(*line, "P7003", "字面量无法转 Q32.32 定点", "缩小数字或去掉多余小数位")
                        })?,
                    ),
                    Expr3::Lit { value: Literal3::Str(s), .. } => Value3::Str(s.clone()),
                    Expr3::Lit { value: Literal3::Bool(b), .. } => Value3::Bool(*b),
                    _ => {
                        return Err(v3_err(
                            item.line(),
                            "P6106",
                            "for 列表只允许字面量",
                            "写成 [1, 2, 3] 这样的字面量列表",
                        ));
                    }
                });
            }
            Ok(out)
        }
    }
}

fn eval_expr(expr: &Expr3, ctx: &mut EvalCtx) -> Result<Value3, DslError> {
    let line: usize = expr.line();
    match expr {
        Expr3::Lit { value, .. } => {
            let v: Value3 = match value {
                Literal3::Num(s) => Value3::Num(Fixed3::from_text(s).ok_or_else(|| {
                    v3_err(line, "P7003", &format!("数字 '{}' 无法转 Q32.32 定点", s), "缩小数字或减少小数位")
                })?),
                Literal3::Str(s) => Value3::Str(s.clone()),
                Literal3::Bool(b) => Value3::Bool(*b),
            };
            ctx.push(line, TAG_LIT, &v);
            Ok(v)
        }
        Expr3::Var { name, .. } => {
            let v: Value3 = ctx.vars.get(name).cloned().ok_or_else(|| {
                v3_err(line, "P6102", &format!("变量 '{}' 尚未赋值", name), "变量必须先赋值再使用")
            })?;
            ctx.push(line, TAG_VAR, &v);
            Ok(v)
        }
        Expr3::Unary { op, rhs, .. } => {
            let r: Value3 = eval_expr(rhs, ctx)?;
            let v: Value3 = match op {
                UnOp3::Neg => Value3::Num(Fixed3(-r.as_num(line, "一元负号")?.0)),
                UnOp3::Not => {
                    let b: bool = match &r {
                        Value3::Bool(x) => !x,
                        Value3::Num(f) => f.0 == 0,
                        _ => {
                            return Err(v3_err(
                                line,
                                "P7003",
                                "not 只能作用于布尔或数值",
                                "写成 not (x > 0) 这样的表达式",
                            ));
                        }
                    };
                    Value3::Bool(b)
                }
            };
            ctx.push(line, TAG_UNARY, &v);
            Ok(v)
        }
        Expr3::Binary { op, lhs, rhs, .. } => {
            // F-1：先完整求左，再完整求右；and / or 也不短路（两侧都进轨迹）。
            let a: Value3 = eval_expr(lhs, ctx)?;
            let b: Value3 = eval_expr(rhs, ctx)?;
            let v: Value3 = eval_binary(*op, &a, &b, line)?;
            ctx.push(line, TAG_BINARY, &v);
            Ok(v)
        }
        Expr3::Call { name, args, .. } => {
            // F-1：实参严格按声明序求值。
            let mut vals: Vec<Value3> = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_expr(a, ctx)?);
            }
            let v: Value3 = if name == "rand" {
                call_rand(&vals, ctx, line)?
            } else if let Some((params, body)) = ctx.defs.get(name).cloned() {
                let saved: BTreeMap<String, Value3> = ctx.vars.clone();
                for (p, val) in params.iter().zip(vals.into_iter()) {
                    ctx.vars.insert(p.clone(), val);
                }
                let result: Value3 = match exec_stmts(&body, ctx)? {
                    Control3::Return(x) => x,
                    Control3::Normal => {
                        return Err(v3_err(line, "P6104", &format!("def '{}' 没有返回值", name), "在函数体末尾写 return <表达式>"));
                    }
                };
                ctx.vars = saved;
                result
            } else {
                call_builtin(name, &vals, line)?
            };
            ctx.push(line, if name == "rand" { TAG_RAND } else { TAG_CALL }, &v);
            Ok(v)
        }
        Expr3::Index { target, index, .. } => {
            let t: Value3 = eval_expr(target, ctx)?;
            let i: Value3 = eval_expr(index, ctx)?;
            let v: Value3 = index_value(&t, &i, line)?;
            ctx.push(line, TAG_INDEX, &v);
            Ok(v)
        }
        Expr3::List { items, .. } => {
            // F-2：按声明序逐项求值。
            let mut vals: Vec<Value3> = Vec::with_capacity(items.len());
            for item in items {
                vals.push(eval_expr(item, ctx)?);
            }
            let v: Value3 = Value3::List(vals);
            ctx.push(line, TAG_LIST, &v);
            Ok(v)
        }
        Expr3::Dict { entries, .. } => {
            // F-2：按声明序逐项求值，迭代序 = 插入序（不做键排序）。
            let mut vals: Vec<(String, Value3)> = Vec::with_capacity(entries.len());
            for (k, item) in entries {
                vals.push((k.clone(), eval_expr(item, ctx)?));
            }
            let v: Value3 = Value3::Dict(vals);
            ctx.push(line, TAG_DICT, &v);
            Ok(v)
        }
        Expr3::Interp { parts, .. } => {
            let mut text = String::new();
            for p in parts {
                match p {
                    InterpPart3::Text(t) => text.push_str(t),
                    InterpPart3::Expr(e) => {
                        let v: Value3 = eval_expr(e, ctx)?;
                        text.push_str(&v.to_text());
                    }
                }
            }
            let v: Value3 = Value3::Str(text);
            ctx.push(line, TAG_INTERP, &v);
            Ok(v)
        }
    }
}

fn eval_binary(op: BinOp3, a: &Value3, b: &Value3, line: usize) -> Result<Value3, DslError> {
    match op {
        BinOp3::Add | BinOp3::Sub | BinOp3::Mul | BinOp3::Div | BinOp3::Mod => {
            let x: Fixed3 = a.as_num(line, "算术运算")?;
            let y: Fixed3 = b.as_num(line, "算术运算")?;
            Ok(Value3::Num(x.arithmetic(y, op, line)?))
        }
        BinOp3::Eq => Ok(Value3::Bool(a == b)),
        BinOp3::Ne => Ok(Value3::Bool(a != b)),
        BinOp3::Lt | BinOp3::Le | BinOp3::Gt | BinOp3::Ge => {
            let x: Fixed3 = a.as_num(line, "比较运算")?;
            let y: Fixed3 = b.as_num(line, "比较运算")?;
            let r: bool = match op {
                BinOp3::Lt => x < y,
                BinOp3::Le => x <= y,
                BinOp3::Gt => x > y,
                BinOp3::Ge => x >= y,
                _ => unreachable!("比较分支已穷举"),
            };
            Ok(Value3::Bool(r))
        }
        BinOp3::And | BinOp3::Or => {
            let to_bool = |v: &Value3| -> Result<bool, DslError> {
                match v {
                    Value3::Bool(x) => Ok(*x),
                    Value3::Num(f) => Ok(f.0 != 0),
                    _ => Err(v3_err(
                        line,
                        "P7003",
                        "and / or 只接受布尔或数值",
                        "写成 (x > 0) and (y < 1) 这样的比较式",
                    )),
                }
            };
            let x: bool = to_bool(a)?;
            let y: bool = to_bool(b)?;
            Ok(Value3::Bool(if op == BinOp3::And { x && y } else { x || y }))
        }
    }
}

fn index_value(target: &Value3, index: &Value3, line: usize) -> Result<Value3, DslError> {
    match target {
        Value3::List(items) => {
            let n: Fixed3 = index.as_num(line, "列表下标")?;
            if n.0 % SCALE != 0 || n.0 < 0 {
                return Err(v3_err(
                    line,
                    "P6203",
                    &format!("列表下标 '{}' 不是非负整数", n.to_text()),
                    "下标写成 0、1、2 … 这样的整数",
                ));
            }
            let i: usize = (n.0 / SCALE) as usize;
            items.get(i).cloned().ok_or_else(|| {
                v3_err(
                    line,
                    "P6203",
                    &format!("列表下标 {} 越界（长度 {}）", i, items.len()),
                    "下标范围是 0 ..= 长度-1",
                )
            })
        }
        Value3::Dict(entries) => {
            let key: String = match index {
                Value3::Str(s) => s.clone(),
                _ => {
                    return Err(v3_err(
                        line,
                        "P6203",
                        "字典下标必须是字符串",
                        "写成 colors[\"albedo\"] 这样的形式",
                    ));
                }
            };
            entries
                .iter()
                .find(|(k, _)| k == &key)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    v3_err(
                        line,
                        "P6203",
                        &format!("字典里没有键 '{}'", key),
                        "检查键名拼写；字典不取默认、不静默返回空值",
                    )
                })
        }
        _ => Err(v3_err(
            line,
            "P6203",
            "只有列表和字典可以下标取值",
            "把下标用在列表 / 字典变量上",
        )),
    }
}

/// F-7：唯一随机源 `hash_mix(seed, tag, eval_ordinal)`，无隐式全局状态。
fn call_rand(args: &[Value3], ctx: &EvalCtx, line: usize) -> Result<Value3, DslError> {
    if args.len() != 1 {
        return Err(v3_err(
            line,
            "P6101",
            "rand 需要 1 个 tag 参数",
            "写成 rand(\"grain\") 这样的形式",
        ));
    }
    let mut h: u64 = hash_mix(ctx.seed, RAND_NAMESPACE);
    h = hash_mix(h, args[0].bits());
    h = hash_mix(h, ctx.ordinal);
    let unit: f64 = hash_mix_unit_f64(h);
    Ok(Value3::Num(Fixed3::from_f64(unit, line)?))
}

/// 内置函数分派（全部走 `prism_math` 的零 libm 实现）。
fn call_builtin(name: &str, args: &[Value3], line: usize) -> Result<Value3, DslError> {
    let arity: usize = builtin_arity(name).ok_or_else(|| {
        v3_err(
            line,
            "P6101",
            &format!("未定义的函数 '{}'", name),
            "只支持 §6 det_* 映射表与 len / int / str / rand",
        )
    })?;
    if args.len() != arity {
        return Err(v3_err(
            line,
            "P6101",
            &format!("'{}' 需要 {} 个参数，却给了 {} 个", name, arity, args.len()),
            "按函数签名补齐参数；不取默认、不静默降级",
        ));
    }
    let num = |i: usize| -> Result<f64, DslError> { Ok(args[i].as_num(line, name)?.to_f64()) };
    let out_f64: f64 = match name {
        "abs" => det_scalar::det_abs(num(0)?),
        "sign" => det_scalar::det_sign(num(0)?),
        "min" => det_scalar::det_min(num(0)?, num(1)?),
        "max" => det_scalar::det_max(num(0)?, num(1)?),
        "clamp" => det_scalar::det_clamp(num(0)?, num(1)?, num(2)?),
        "floor" => det_scalar::det_floor(num(0)?),
        "ceil" => det_scalar::det_ceil(num(0)?),
        "trunc" => det_scalar::det_trunc(num(0)?),
        "round" => det_scalar::det_round(num(0)?),
        "sqrt" => det_scalar::det_sqrt(num(0)?),
        "pow" => det_scalar::det_pow_f64(num(0)?, num(1)?),
        "exp" => det_scalar::det_exp_f64(num(0)?),
        "ln" => det_scalar::det_ln_f64(num(0)?),
        "log2" => det_scalar::det_log2_f64(num(0)?),
        "log10" => det_scalar::det_log10(num(0)?),
        "sin" => det_scalar::det_sin_f64(num(0)?),
        "cos" => det_scalar::det_cos_f64(num(0)?),
        "tan" => det_scalar::det_tan_f64(num(0)?),
        "acos" => det_scalar::det_acos_f64(num(0)?),
        "asin" => det_scalar::det_asin(num(0)?),
        "atan" => det_scalar::det_atan(num(0)?),
        "atan2" => colorspace_registry::det_atan2(num(0)?, num(1)?),
        "lerp" => det_scalar::det_lerp(num(0)?, num(1)?, num(2)?),
        "len" => {
            let n: usize = match &args[0] {
                Value3::Str(s) => s.chars().count(),
                Value3::List(v) => v.len(),
                Value3::Dict(v) => v.len(),
                _ => {
                    return Err(v3_err(
                        line,
                        "P7003",
                        "len 只接受字符串 / 列表 / 字典",
                        "传入容器值",
                    ));
                }
            };
            return Ok(Value3::Num(Fixed3(n as i64 * SCALE)));
        }
        "int" => {
            let f: Fixed3 = args[0].as_num(line, "int")?;
            return Ok(Value3::Num(Fixed3((f.0 / SCALE) * SCALE)));
        }
        "str" => return Ok(Value3::Str(args[0].to_text())),
        "rand" => unreachable!("rand 由 call_rand 单独处理"),
        _ => {
            return Err(v3_err(
                line,
                "P6101",
                &format!("内置函数 '{}' 没有实现", name),
                "检查 §6 det_* 覆盖表",
            ));
        }
    };
    Ok(Value3::Num(Fixed3::from_f64(out_f64, line)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_text_roundtrip_is_integer_only() {
        assert_eq!(Fixed3::from_text("2"), Some(Fixed3(2 * SCALE)));
        assert_eq!(Fixed3::from_text("0.5"), Some(Fixed3(SCALE / 2)));
        assert_eq!(Fixed3::from_text("-1.25"), Some(Fixed3(-5 * SCALE / 4)));
        assert_eq!(Fixed3::from_text("-.5"), None);
        assert_eq!(Fixed3::from_text("1.2.3"), None);
        assert_eq!(Fixed3(2 * SCALE).to_text(), "2");
        assert_eq!(Fixed3(SCALE / 2).to_text(), "0.5");
        assert_eq!(Fixed3(-5 * SCALE / 4).to_text(), "-1.25");
        assert_eq!(Fixed3(SCALE / 3).to_text(), "0.33333333");
    }

    #[test]
    fn fixed_arithmetic_is_pure_integer() {
        let a = Fixed3::from_text("1.5").unwrap();
        let b = Fixed3::from_text("0.25").unwrap();
        assert_eq!(a.arithmetic(b, BinOp3::Add, 1).unwrap().to_text(), "1.75");
        assert_eq!(a.arithmetic(b, BinOp3::Mul, 1).unwrap().to_text(), "0.375");
        assert_eq!(a.arithmetic(b, BinOp3::Div, 1).unwrap().to_text(), "6");
        assert!(a.arithmetic(Fixed3::ZERO, BinOp3::Div, 7).is_err());
    }

    #[test]
    fn value_bits_are_deterministic_and_order_sensitive() {
        let a = Value3::List(vec![
            Value3::Num(Fixed3::from_text("1").unwrap()),
            Value3::Num(Fixed3::from_text("2").unwrap()),
        ]);
        let b = Value3::List(vec![
            Value3::Num(Fixed3::from_text("2").unwrap()),
            Value3::Num(Fixed3::from_text("1").unwrap()),
        ]);
        assert_ne!(a.bits(), b.bits(), "换序必须可检出");
        assert_eq!(a.bits(), a.clone().bits());
    }
}
