//! 编辑命令与逆命令（命令模式）。
//!
//! 每条 [`EditCommand`] 是**可逆的原子编辑**：携带足以从「编辑后」还原到
//! 「编辑前」的旧值（undo），也足以从「编辑前」再前进到「编辑后」（redo）。
//! 逆命令（[`EditCommand::invert`]）通过交换 `old`/`new` 字段得到——因此
//! undo 与 redo 走**同一套** apply 逻辑，不存在两套代码漂移的风险。
//!
//! # 零第三方依赖
//! 命令的序列化为手写 JSON 字符串（[`EditCommand::to_json`]），供编辑器
//! 导出「编辑操作序列」，重放时逐字节可复现（确定性铁律）。

use std::fmt;

use prism_core::node::{Composition, ParamValue};

/// 编辑器命令/参数错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 建议段）。
///
/// 错误码取 **E4xxx** 段（app 层编辑器；与 core 的 P3xxx 图/签名段隔离）。
#[derive(Debug, Clone, PartialEq)]
pub enum EditError {
    /// 节点不存在（悬空引用）。
    UnknownNode {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：缺失的节点 id。
        node_id: u64,
    },
    /// 撤销栈为空（无可撤销的操作，结构化错误——**不 panic**，C18）。
    EmptyUndoStack {
        /// 三段式一【在哪】。
        location: String,
    },
    /// 重做栈为空（无可重做的操作）。
    EmptyRedoStack {
        /// 三段式一【在哪】。
        location: String,
    },
    /// 参数类型不匹配（编辑值与节点既有参数类型不同）。
    ParamTypeMismatch {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：参数名。
        param: String,
        /// 期望的类型名。
        expected: &'static str,
        /// 实际给出的类型名。
        got: &'static str,
    },
    /// 参数不存在（编辑了一个节点没有的参数，且未声明新增）。
    UnknownParam {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：参数名。
        param: String,
    },
    /// 组合方式变更的类型不匹配（如把非 Merge 节点改为 Merge）。
    CompositionMismatch {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：说明。
        detail: String,
    },
    /// 签名计算失败（图结构错误透传，如环）。
    Signature {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：底层签名错误文本。
        detail: String,
    },
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::UnknownNode { location, node_id } => write!(
                f,
                "error[E4001] {location}\n  什么事：节点 id {node_id} 不在编辑器中\n  建议：先通过 add_node 加入该节点，或改用存在的节点 id"
            ),
            EditError::EmptyUndoStack { location } => write!(
                f,
                "error[E4002] {location}\n  什么事：撤销栈为空，没有可撤销的编辑\n  建议：先 apply_edit 至少一次；或用 can_undo() 预判栈是否非空"
            ),
            EditError::EmptyRedoStack { location } => write!(
                f,
                "error[E4003] {location}\n  什么事：重做栈为空，没有可重做的编辑\n  建议：先 undo() 至少一次把命令压入重做栈；或用 can_redo() 预判"
            ),
            EditError::ParamTypeMismatch { location, param, expected, got } => write!(
                f,
                "error[E4004] {location}\n  什么事：参数 \"{param}\" 类型不匹配（期望 {expected}，实际 {got}）\n  建议：用与既有参数同类型的值编辑，或先删除该参数再以新类型写入"
            ),
            EditError::UnknownParam { location, param } => write!(
                f,
                "error[E4005] {location}\n  什么事：参数 \"{param}\" 在目标节点上不存在\n  建议：先 set_param 建立该参数（nodes 参数 schema 决定可用参数名）"
            ),
            EditError::CompositionMismatch { location, detail } => write!(
                f,
                "error[E4006] {location}\n  什么事：{detail}\n  建议：仅在同种组合方式变体间切换（如 Merge/Over → Merge/Plus）"
            ),
            EditError::Signature { location, detail } => write!(
                f,
                "error[E4007] {location}\n  什么事：签名计算失败：{detail}\n  建议：先修复图结构（如环），再重新应用编辑"
            ),
        }
    }
}

impl std::error::Error for EditError {}

/// 参数规格：描述一个可编辑参数（供编辑器 UI 生成滑条/数字控件）。
///
/// UI 侧据此渲染控件：`min`/`max`/`step` 给滑条，`ParamValue` 决定控件类型
/// （数值→滑条，布尔→开关，字符串→文本框）。
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    /// 参数名（[`params`](prism_core::node::Params) 的键）。
    pub name: String,
    /// 当前值。
    pub value: ParamValue,
    /// 滑条下界（仅对 `F32` 有意义；其余类型填 0.0）。
    pub min: f32,
    /// 滑条上界（仅对 `F32` 有意义）。
    pub max: f32,
    /// 步进（仅对 `F32` 有意义；0.0 表示连续）。
    pub step: f32,
}

impl ParamSpec {
    /// 以名称 + 当前值 + 滑条范围构造参数规格。
    pub fn new(name: impl Into<String>, value: ParamValue, min: f32, max: f32, step: f32) -> Self {
        Self { name: name.into(), value, min, max, step }
    }

    /// 该参数是否为数值型（UI 用滑条渲染）。
    pub fn is_numeric(&self) -> bool {
        matches!(self.value, ParamValue::F32(_) | ParamValue::U32(_))
    }
}

/// 一次**可逆**原子编辑。
///
/// - [`EditCommand::SetParam`]：把 `node` 的参数 `param` 从 `old` 改为 `new`；
/// - [`EditCommand::SetComposition`]：把 `node` 的组合方式从 `old` 改为 `new`；
/// - [`EditCommand::AddNode`] / [`EditCommand::RemoveNode`]：增删节点（redo 用
///   同一命令重放，undo 用逆命令重放）。
///
/// 逆命令通过 [`EditCommand::invert`] 获得：交换旧值/新值即可，**undo 与 redo
/// 共用同一 apply 逻辑**。
#[derive(Debug, Clone, PartialEq)]
pub enum EditCommand {
    /// 设置参数值（可逆：携带旧值）。
    SetParam {
        /// 目标节点 id。
        node: u64,
        /// 参数名。
        param: String,
        /// 编辑前的旧值。
        old: ParamValue,
        /// 编辑后的新值。
        new: ParamValue,
    },
    /// 设置组合方式（可逆：携带旧组合）。
    SetComposition {
        /// 目标节点 id。
        node: u64,
        /// 编辑前的旧组合方式。
        old: Composition,
        /// 编辑后的新组合方式。
        new: Composition,
    },
    /// 新增节点（可逆：逆命令为 RemoveNode，携带同一节点的 canonical 描述）。
    AddNode {
        /// 目标节点 id。
        node: u64,
        /// 组合方式（重建节点用）。
        composition: Composition,
    },
    /// 删除节点（可逆：逆命令为 AddNode）。
    RemoveNode {
        /// 目标节点 id。
        node: u64,
        /// 被删节点原组合方式（重建用）。
        composition: Composition,
    },
}

impl EditCommand {
    /// 目标节点 id（所有变体都指向一个节点）。
    pub fn node(&self) -> u64 {
        match self {
            EditCommand::SetParam { node, .. }
            | EditCommand::SetComposition { node, .. }
            | EditCommand::AddNode { node, .. }
            | EditCommand::RemoveNode { node, .. } => *node,
        }
    }

    /// 逆命令（交换旧值/新值）：undo 与 redo 走同一 apply 逻辑。
    pub fn invert(&self) -> EditCommand {
        match self {
            EditCommand::SetParam { node, param, old, new } => EditCommand::SetParam {
                node: *node,
                param: param.clone(),
                old: new.clone(),
                new: old.clone(),
            },
            EditCommand::SetComposition { node, old, new } => EditCommand::SetComposition {
                node: *node,
                old: new.clone(),
                new: old.clone(),
            },
            EditCommand::AddNode { node, composition } => EditCommand::RemoveNode {
                node: *node,
                composition: composition.clone(),
            },
            EditCommand::RemoveNode { node, composition } => EditCommand::AddNode {
                node: *node,
                composition: composition.clone(),
            },
        }
    }

    /// 人类可读的操作描述（供编辑器操作序列展示）。
    pub fn describe(&self) -> String {
        match self {
            EditCommand::SetParam { node, param, new, .. } => {
                format!("节点 {node} 参数 {param} = {}", param_value_text(new))
            }
            EditCommand::SetComposition { node, new, .. } => {
                format!("节点 {node} 组合 → {}", composition_text(new))
            }
            EditCommand::AddNode { node, .. } => format!("新增节点 {node}"),
            EditCommand::RemoveNode { node, .. } => format!("删除节点 {node}"),
        }
    }

    /// 序列化为手写 JSON（确定性：字段顺序固定，无第三方 serde）。
    ///
    /// 重放契约：`from_json(to_json(cmd)) == cmd`（round-trip，见测试）。
    pub fn to_json(&self) -> String {
        match self {
            EditCommand::SetParam { node, param, old, new } => format!(
                "{{\"kind\":\"set_param\",\"node\":{node},\"param\":{},\"old\":{},\"new\":{}}}",
                json_string(param),
                param_value_json(old),
                param_value_json(new)
            ),
            EditCommand::SetComposition { node, old, new } => format!(
                "{{\"kind\":\"set_composition\",\"node\":{node},\"old\":{},\"new\":{}}}",
                composition_json(old),
                composition_json(new)
            ),
            EditCommand::AddNode { node, composition } => format!(
                "{{\"kind\":\"add_node\",\"node\":{node},\"composition\":{}}}",
                composition_json(composition)
            ),
            EditCommand::RemoveNode { node, composition } => format!(
                "{{\"kind\":\"remove_node\",\"node\":{node},\"composition\":{}}}",
                composition_json(composition)
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// 手写 JSON 部件（零第三方依赖；确定性输出）
// ---------------------------------------------------------------------------

/// JSON 字符串字面量转义（仅处理必需的 `"`、`\` 与控制字符）。
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 参数值的 JSON 编码（带类型 tag，防止 `1` 与 `"1"` 歧义）。
pub fn param_value_json(v: &ParamValue) -> String {
    match v {
        ParamValue::F32(x) => format!("{{\"t\":\"f32\",\"v\":{}}}", json_f32(*x)),
        ParamValue::U32(x) => format!("{{\"t\":\"u32\",\"v\":{x}}}"),
        ParamValue::Bool(x) => format!("{{\"t\":\"bool\",\"v\":{}}}", if *x { "true" } else { "false" }),
        ParamValue::Str(x) => format!("{{\"t\":\"str\",\"v\":{}}}", json_string(x)),
    }
}

/// `f32` 的 JSON 数值文本（确定性：`{:.9}` 覆盖 f32 有效位；NaN/Inf → null）。
fn json_f32(x: f32) -> String {
    if x.is_finite() {
        format!("{x:.9}")
    } else {
        "null".to_string()
    }
}

/// 组合方式的 JSON 编码（tag + 负载）。
pub fn composition_json(c: &Composition) -> String {
    match c {
        Composition::Feed => "{\"kind\":\"feed\"}".to_string(),
        Composition::Chain => "{\"kind\":\"chain\"}".to_string(),
        Composition::Merge(op) => format!("{{\"kind\":\"merge\",\"op\":\"{}\"}}", merge_op_name(*op)),
        Composition::Csg { op, k } => format!(
            "{{\"kind\":\"csg\",\"op\":\"{}\",\"k\":{}}}",
            csg_op_name(*op),
            json_f32(*k)
        ),
        Composition::Mis(w) => format!("{{\"kind\":\"mis\",\"w\":{}}}", json_f32(*w)),
        Composition::Dual => "{\"kind\":\"dual\"}".to_string(),
    }
}

/// `MergeOp` 的稳定名（进 JSON，重放与展示共用）。
pub fn merge_op_name(op: prism_core::node::MergeOp) -> &'static str {
    use prism_core::node::MergeOp;
    match op {
        MergeOp::Over => "over",
        MergeOp::Plus => "plus",
        MergeOp::Screen => "screen",
        MergeOp::Multiply => "multiply",
        MergeOp::Mask => "mask",
    }
}

/// `CsgOp` 的稳定名。
pub fn csg_op_name(op: prism_core::node::CsgOp) -> &'static str {
    use prism_core::node::CsgOp;
    match op {
        CsgOp::Union => "union",
        CsgOp::Intersect => "intersect",
        CsgOp::Subtract => "subtract",
        CsgOp::SmoothUnion => "smooth_union",
        CsgOp::SmoothIntersect => "smooth_intersect",
        CsgOp::SmoothSubtract => "smooth_subtract",
    }
}

/// 组合方式的人类可读文本（操作描述用）。
fn composition_text(c: &Composition) -> String {
    match c {
        Composition::Feed => "Feed".to_string(),
        Composition::Chain => "Chain".to_string(),
        Composition::Merge(op) => format!("Merge({})", merge_op_name(*op)),
        Composition::Csg { op, k } => format!("Csg({}, k={k})", csg_op_name(*op)),
        Composition::Mis(w) => format!("Mis({w})"),
        Composition::Dual => "Dual".to_string(),
    }
}

/// 参数值的人类可读文本。
fn param_value_text(v: &ParamValue) -> String {
    match v {
        ParamValue::F32(x) => format!("{x}"),
        ParamValue::U32(x) => format!("{x}"),
        ParamValue::Bool(x) => format!("{x}"),
        ParamValue::Str(x) => x.clone(),
    }
}

/// 参数值的类型名（错误消息用）。
pub fn param_value_type_name(v: &ParamValue) -> &'static str {
    match v {
        ParamValue::F32(_) => "f32",
        ParamValue::U32(_) => "u32",
        ParamValue::Bool(_) => "bool",
        ParamValue::Str(_) => "str",
    }
}
