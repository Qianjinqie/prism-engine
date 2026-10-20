//! 编辑器状态机：图 + 选中节点 + 撤销/重做历史 + 增量失效。
//!
//! # 增量失效（核心机制）
//! 见模块头 [`super`] 的说明。要点：用签名树的「改一个节点只影响其到根祖先链」
//! 性质，把「受影响子图」精确算出——dirty 集合**恰好**是下游闭包，不多不少。
//!
//! # 为什么不直接改 `prism_core::Graph`
//! core 的 [`Graph`] 是**只增**结构（`add_node` / `connect`，无 `remove_node`
//! / `node_mut`），这是它的语义契约。编辑器需要「改参数 / 删节点」这些**非
//! 只增**操作。为**不侵入 core**（避免与其它轮次冲突），本模块持有自己的
//! 权威模型 [`EditorModel`]（节点表 + 边表），在需要签名 / canonical 字节时
//! **现建**一张 `prism_core::Graph` 交给签名树。core 的签名 API 原样复用。
//!
//! # 确定性
//! - 图 canonical 字节（[`graph_canonical_bytes`]）是图内容的单射指纹：编辑
//!   序列重放后重算该字节，与原路径逐字节一致（集成测试断言）；
//! - 命令的 JSON 序列化字段顺序固定；`to_json`/重放 round-trip 稳定；
//! - 所有遍历序用 `BTreeMap`/`BTreeSet`，无哈希随机序。

use std::collections::{BTreeMap, BTreeSet};

use prism_core::node::{Composition, Hash256, Node, NodeId, ParamValue, StructureId};
use prism_core::signature::{graph_signatures, ChannelDefs};
use prism_core::{graph_canonical_bytes, Graph};

use super::command::{param_value_type_name, EditCommand, EditError, ParamSpec};

/// 编辑器 schema 版本（进导出 JSON 头；版本变更 → 旧导出序列按版本处理）。
pub const EDITOR_SCHEMA_VERSION: u32 = 1;

/// 草稿档位采样数（低，实时预览优先延迟）。
pub const DRAFT_SPP: u32 = 1;
/// 成片档位采样数（高，质量优先）。
pub const FINAL_SPP: u32 = 16;

/// 预览质量档位（草稿 / 成片）。采样数经缓存键的 `spp_tier` 位进入缓存身份，
/// 档位切换使旧缓存自然失效（C13），无需手工清缓存。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewQuality {
    /// 草稿：低采样数，编辑期实时预览。
    Draft,
    /// 成片：高采样数，最终渲染。
    Final,
}

impl PreviewQuality {
    /// 每像素样本数（草稿 < 成片，硬性验收项）。
    pub fn spp(self) -> u32 {
        match self {
            PreviewQuality::Draft => DRAFT_SPP,
            PreviewQuality::Final => FINAL_SPP,
        }
    }

    /// 缓存键用的档号（Draft = 0，Final = 1）。
    pub fn spp_tier(self) -> u32 {
        match self {
            PreviewQuality::Draft => 0,
            PreviewQuality::Final => 1,
        }
    }

    /// 档位名（进导出 JSON）。
    pub fn name(self) -> &'static str {
        match self {
            PreviewQuality::Draft => "draft",
            PreviewQuality::Final => "final",
        }
    }
}

/// 增量失效产生的脏集合：**受影响子图** = 签名变化的节点 + 其全部下游。
///
/// 内部 `BTreeSet` 保证确定性遍历序与去重。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirtySet {
    nodes: BTreeSet<NodeId>,
    /// 是否发生**结构签名失效**（新图无法计算签名，如环）——此时全部节点视为脏。
    structural: bool,
}

impl DirtySet {
    /// 空脏集合（无任何节点需重算）。
    pub fn empty() -> Self {
        Self { nodes: BTreeSet::new(), structural: false }
    }

    /// 含全部给定节点的脏集合。
    pub fn from_nodes(nodes: impl IntoIterator<Item = NodeId>) -> Self {
        Self { nodes: nodes.into_iter().collect(), structural: false }
    }

    /// 结构失效脏集合（全部已知节点）。
    pub fn all(nodes: impl IntoIterator<Item = NodeId>) -> Self {
        Self { nodes: nodes.into_iter().collect(), structural: true }
    }

    /// 加入一个节点。
    pub fn insert(&mut self, id: NodeId) {
        self.nodes.insert(id);
    }

    /// 是否包含某节点。
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains(&id)
    }

    /// 脏节点个数。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否为空（无需重算任何子图）。
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 是否发生结构签名失效。
    pub fn is_structural(&self) -> bool {
        self.structural
    }

    /// 脏节点（id 升序）。
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().copied()
    }

    /// 脏节点（id 升序 Vec）。
    pub fn to_vec(&self) -> Vec<NodeId> {
        self.nodes.iter().copied().collect()
    }
}

/// 编辑器权威模型：节点表 + 边表（可增删改，超越 core 的只增契约）。
///
/// 与 [`Graph`] 的转换由 [`EditorModel::to_graph`] 完成——构造时按 id 升序
/// `add_node` / 按 (to, slot) 升序 `connect`，与 core 的 canonical 序一致。
#[derive(Debug, Clone, Default)]
pub struct EditorModel {
    /// 节点表（id → 节点）。
    nodes: BTreeMap<NodeId, Node>,
    /// 边表：(目标节点, 输入槽位) → 源节点。
    edges: BTreeMap<(NodeId, u8), NodeId>,
}

impl EditorModel {
    /// 由一张 core 图构造编辑器模型。
    pub fn from_graph(graph: &Graph) -> Self {
        let mut model = EditorModel::default();
        for node in graph.nodes() {
            model.nodes.insert(node.id, node.clone());
        }
        for node in graph.nodes() {
            for (slot, from) in graph.inputs_of(node.id) {
                model.edges.insert((node.id, slot), from);
            }
        }
        model
    }

    /// 转为 core 图（供签名树 / canonical 字节消费）。
    ///
    /// 若模型本身自洽（端点齐全），`add_node`/`connect` 不会失败；失败时返回
    /// `None`（由调用方转结构化错误，绝不 panic）。
    pub fn to_graph(&self) -> Option<Graph> {
        let mut g = Graph::new();
        for node in self.nodes.values() {
            g.add_node(node.clone()).ok()?;
        }
        for (&(to, slot), &from) in &self.edges {
            g.connect(from, to, slot).ok()?;
        }
        Some(g)
    }

    /// 节点引用。
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// 节点可变引用（编辑器内部改参数 / 组合方式用）。
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// 全部节点（id 升序）。
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// 节点是否存在。
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// 节点数。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 加入节点（id 冲突返回 false）。
    pub fn add_node(&mut self, node: Node) -> bool {
        if self.nodes.contains_key(&node.id) {
            return false;
        }
        self.nodes.insert(node.id, node);
        true
    }

    /// 删除节点（连带删除其入边与出边）。
    pub fn remove_node(&mut self, id: NodeId) -> Option<Node> {
        let removed = self.nodes.remove(&id)?;
        self.edges.retain(|&(to, _), &mut from| to != id && from != id);
        Some(removed)
    }

    /// 上游列表（(槽位, 源节点)，按槽位升序）。
    pub fn inputs_of(&self, to: NodeId) -> Vec<(u8, NodeId)> {
        self.edges
            .range((to, u8::MIN)..=(to, u8::MAX))
            .map(|(&(_, slot), &from)| (slot, from))
            .collect()
    }

    /// 下游列表（该节点的直接后继，id 升序去重）。
    pub fn downstream_of(&self, id: NodeId) -> Vec<NodeId> {
        let mut set: BTreeSet<NodeId> = BTreeSet::new();
        for (&(to, _), &from) in &self.edges {
            if from == id {
                set.insert(to);
            }
        }
        set.into_iter().collect()
    }
}

/// 编辑器状态：模型 + 选中节点 + 撤销/重做双栈 + 当前签名快照 + 质量档位。
///
/// 生命周期：
/// 1. [`EditorState::new`] 以一张图初始化（并缓存初始签名快照）；
/// 2. [`EditorState::apply_edit`] 应用编辑 → 返回本次的 [`DirtySet`]；
/// 3. [`EditorState::undo`] / [`EditorState::redo`] 前后移动历史。
#[derive(Debug, Clone)]
pub struct EditorState {
    /// 权威图模型。
    model: EditorModel,
    /// 通道定义表（签名计算所需；编辑器默认空表——参考图节点无通道绑定）。
    defs: ChannelDefs,
    /// 当前选中节点（UI 高亮用；`None` 表示未选中）。
    selected: Option<NodeId>,
    /// 参数 schema（节点 id → 该节点的可编辑参数规格），UI 控件生成用。
    schemas: BTreeMap<NodeId, Vec<ParamSpec>>,
    /// 撤销栈（栈顶为最后一次编辑；undo 弹出并逆应用）。
    undo_stack: Vec<EditCommand>,
    /// 重做栈（undo 压入；redo 弹出并正应用）。
    redo_stack: Vec<EditCommand>,
    /// 当前各节点签名快照（增量比对基线；每次编辑后刷新）。
    signatures: BTreeMap<NodeId, Hash256>,
    /// 当前质量档位。
    quality: PreviewQuality,
    /// 最近一次增量失效产生的脏集合（供 UI 查询）。
    last_dirty: DirtySet,
}

impl EditorState {
    /// 以一张 core 图构造编辑器状态（默认档位 Draft，无选中，历史为空）。
    ///
    /// 初始签名快照立即计算（作为后续增量比对的基线）。
    pub fn new(graph: Graph) -> Self {
        let model = EditorModel::from_graph(&graph);
        let defs = ChannelDefs::new();
        let signatures = graph_signatures(&graph, &defs).unwrap_or_default();
        let schemas = default_schemas(&model);
        Self {
            model,
            defs,
            selected: None,
            schemas,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            signatures,
            quality: PreviewQuality::Draft,
            last_dirty: DirtySet::empty(),
        }
    }

    /// 以默认参考图构造（编辑器 harness 的初始图；见 [`build_reference_graph`]）。
    pub fn with_reference_graph() -> Self {
        Self::new(build_reference_graph())
    }

    // -----------------------------------------------------------------
    // 只读访问器
    // -----------------------------------------------------------------

    /// 编辑器模型引用。
    pub fn model(&self) -> &EditorModel {
        &self.model
    }

    /// 当前选中节点。
    pub fn selected(&self) -> Option<NodeId> {
        self.selected
    }

    /// 当前质量档位。
    pub fn preview_quality(&self) -> PreviewQuality {
        self.quality
    }

    /// 最近一次编辑产生的脏集合（受影响子图）。
    pub fn dirty_subgraph(&self) -> &DirtySet {
        &self.last_dirty
    }

    /// 可撤销（撤销栈非空）。
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// 可重做（重做栈非空）。
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// 撤销栈深度。
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// 重做栈深度。
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// 参数 schema（节点 id → 参数规格列表；UI 控件生成用）。
    pub fn schemas(&self) -> &BTreeMap<NodeId, Vec<ParamSpec>> {
        &self.schemas
    }

    /// 某节点的参数规格列表。
    pub fn schema_of(&self, id: NodeId) -> Option<&[ParamSpec]> {
        self.schemas.get(&id).map(Vec::as_slice)
    }

    /// 图 canonical 字节（确定性指纹；编辑序列重放一致性断言用）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let g: Graph = self
            .model
            .to_graph()
            .expect("编辑器模型自洽：to_graph 不应失败");
        graph_canonical_bytes(&g)
    }

    /// 图当前签名表（节点 id → 子树签名）。
    pub fn signatures(&self) -> &BTreeMap<NodeId, Hash256> {
        &self.signatures
    }

    // -----------------------------------------------------------------
    // 选中（UI 交互）
    // -----------------------------------------------------------------

    /// 选中一个节点（不存在返回结构化错误，不 panic）。
    pub fn select(&mut self, id: NodeId) -> Result<(), EditError> {
        if !self.model.contains(id) {
            return Err(EditError::UnknownNode {
                location: format!("editor.select(节点 {id})"),
                node_id: id,
            });
        }
        self.selected = Some(id);
        Ok(())
    }

    /// 取消选中。
    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    // -----------------------------------------------------------------
    // 质量档位
    // -----------------------------------------------------------------

    /// 切换质量档位。
    pub fn set_preview_quality(&mut self, quality: PreviewQuality) {
        self.quality = quality;
    }

    // -----------------------------------------------------------------
    // 编辑（命令模式 + 增量失效）
    // -----------------------------------------------------------------

    /// 应用一次**参数编辑**：把 `node` 的参数 `param` 改为 `new`。
    ///
    /// 成功时：命令压入撤销栈、清空重做栈、参数写入模型、刷新签名并返回脏集合。
    /// 幂等编辑（新旧值逐字段相同）成功但脏集为**空**（签名不变 ⇒ 零重算）。
    ///
    /// # Errors
    /// 节点不存在 → [`EditError::UnknownNode`]；参数不存在 → [`EditError::UnknownParam`]；
    /// 类型不匹配 → [`EditError::ParamTypeMismatch`]。
    pub fn apply_edit(
        &mut self,
        node: NodeId,
        param: impl Into<String>,
        new: ParamValue,
    ) -> Result<DirtySet, EditError> {
        let param = param.into();
        let current: &Node = self.model.node(node).ok_or_else(|| EditError::UnknownNode {
            location: format!("editor.apply_edit(节点 {node}, 参数 {param})"),
            node_id: node,
        })?;
        let old: ParamValue = current.params.get(&param).cloned().ok_or_else(|| {
            EditError::UnknownParam {
                location: format!("editor.apply_edit(节点 {node}, 参数 {param})"),
                param: param.clone(),
            }
        })?;
        if std::mem::discriminant(&old) != std::mem::discriminant(&new) {
            return Err(EditError::ParamTypeMismatch {
                location: format!("editor.apply_edit(节点 {node}, 参数 {param})"),
                param: param.clone(),
                expected: param_value_type_name(&old),
                got: param_value_type_name(&new),
            });
        }
        let cmd = EditCommand::SetParam { node, param, old, new };
        self.commit(cmd)
    }

    /// 应用一次**组合方式编辑**（同种组合变体间切换，如 Merge/Over → Merge/Plus）。
    ///
    /// # Errors
    /// 节点不存在 → [`EditError::UnknownNode`]；新旧组合判别式不同 →
    /// [`EditError::CompositionMismatch`]（不允许跨变体改变节点拓扑语义）。
    pub fn apply_composition_edit(
        &mut self,
        node: NodeId,
        new: Composition,
    ) -> Result<DirtySet, EditError> {
        let current: &Node = self.model.node(node).ok_or_else(|| EditError::UnknownNode {
            location: format!("editor.apply_composition_edit(节点 {node})"),
            node_id: node,
        })?;
        let old: Composition = current.composition.clone();
        if std::mem::discriminant(&old) != std::mem::discriminant(&new) {
            return Err(EditError::CompositionMismatch {
                location: format!("editor.apply_composition_edit(节点 {node})"),
                detail: format!("组合方式变体不一致：旧 {old:?}，新 {new:?}"),
            });
        }
        let cmd = EditCommand::SetComposition { node, old, new };
        self.commit(cmd)
    }

    /// 撤销一次编辑。撤销栈为空 → 结构化错误（[`EditError::EmptyUndoStack`]，不 panic）。
    ///
    /// 实现：弹出栈顶命令，取其**逆命令**重放（逆命令交换旧值/新值），原命令压入重做栈。
    pub fn undo(&mut self) -> Result<DirtySet, EditError> {
        let cmd: EditCommand = self.undo_stack.pop().ok_or_else(|| EditError::EmptyUndoStack {
            location: "editor.undo()".to_string(),
        })?;
        let inverse: EditCommand = cmd.invert();
        let dirty: DirtySet = self.replay(&inverse)?;
        self.redo_stack.push(cmd);
        self.last_dirty = dirty.clone();
        Ok(dirty)
    }

    /// 重做一次编辑。重做栈为空 → 结构化错误（[`EditError::EmptyRedoStack`]，不 panic）。
    ///
    /// 实现：弹出重做栈顶命令，**正应用**（redo 与 undo 走同一 `replay` 逻辑），
    /// 原命令压回撤销栈。
    pub fn redo(&mut self) -> Result<DirtySet, EditError> {
        let cmd: EditCommand = self.redo_stack.pop().ok_or_else(|| EditError::EmptyRedoStack {
            location: "editor.redo()".to_string(),
        })?;
        let dirty: DirtySet = self.replay(&cmd)?;
        self.undo_stack.push(cmd);
        self.last_dirty = dirty.clone();
        Ok(dirty)
    }

    /// 提交一条新命令：压栈、正应用、清空重做栈。
    fn commit(&mut self, cmd: EditCommand) -> Result<DirtySet, EditError> {
        let dirty: DirtySet = self.replay(&cmd)?;
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        self.last_dirty = dirty.clone();
        Ok(dirty)
    }

    /// 重放一条命令（正应用：把 `new` 写入模型），并算出增量脏集合。
    ///
    /// **undo / redo / 首次应用共用此函数**——逆命令由 [`EditCommand::invert`]
    /// 产出后同样走这里，杜绝两套逻辑漂移。
    fn replay(&mut self, cmd: &EditCommand) -> Result<DirtySet, EditError> {
        match cmd {
            EditCommand::SetParam { node, param, new, .. } => {
                let target = self.model.node_mut(*node).ok_or_else(|| EditError::UnknownNode {
                    location: format!("editor.replay(节点 {node}, 参数 {param})"),
                    node_id: *node,
                })?;
                target.params.set(param.clone(), new.clone());
            }
            EditCommand::SetComposition { node, new, .. } => {
                let target = self.model.node_mut(*node).ok_or_else(|| EditError::UnknownNode {
                    location: format!("editor.replay(节点 {node}, 组合方式)"),
                    node_id: *node,
                })?;
                target.composition = new.clone();
            }
            EditCommand::AddNode { node, composition } => {
                let n = Node::new(*node, StructureId::ScalarField, composition.clone());
                if !self.model.add_node(n) {
                    return Err(EditError::UnknownNode {
                        location: format!("editor.replay(add_node 节点 {node})"),
                        node_id: *node,
                    });
                }
            }
            EditCommand::RemoveNode { node, .. } => {
                let _ = self.model.remove_node(*node);
            }
        }
        self.refresh_dirty()
    }

    /// 重算签名并算出增量脏集合：签名变化的节点 + 其全部下游闭包。
    ///
    /// 若新图无法计算签名（如引入环）→ 返回结构失效脏集（全部节点脏，保守安全）。
    /// 完成后同步刷新参数 schema（新增/删除节点 / 新参数）。
    fn refresh_dirty(&mut self) -> Result<DirtySet, EditError> {
        let graph: Graph = self.model.to_graph().ok_or_else(|| EditError::Signature {
            location: "editor.refresh_dirty（重建 core 图）".to_string(),
            detail: "编辑器模型存在悬空连线，无法构造 core 图".to_string(),
        })?;

        let new_sigs: BTreeMap<NodeId, Hash256> = match graph_signatures(&graph, &self.defs) {
            Ok(s) => s,
            Err(_) => {
                // 结构失效：全部节点视为脏（绝不漏算）。
                let all: Vec<NodeId> = self.model.nodes().map(|n| n.id).collect();
                self.signatures = BTreeMap::new();
                self.refresh_schemas();
                return Ok(DirtySet::all(all));
            }
        };

        // 签名变化的节点（含新增；删除在下一循环处理）。
        let mut changed: BTreeSet<NodeId> = BTreeSet::new();
        for (id, sig) in &new_sigs {
            match self.signatures.get(id) {
                Some(old) if old == sig => {}
                _ => {
                    changed.insert(*id);
                }
            }
        }
        // 旧快照存在、新图消失的节点 = 变化（其删除传播到下游）。
        for id in self.signatures.keys() {
            if !new_sigs.contains_key(id) {
                changed.insert(*id);
            }
        }

        // 受影响子图 = 变化节点 ∪ 其下游闭包。
        let mut dirty: BTreeSet<NodeId> = BTreeSet::new();
        let mut frontier: Vec<NodeId> = changed.iter().copied().collect();
        while let Some(id) = frontier.pop() {
            if !dirty.insert(id) {
                continue;
            }
            for down in self.model.downstream_of(id) {
                if !dirty.contains(&down) {
                    frontier.push(down);
                }
            }
        }

        self.signatures = new_sigs;
        self.refresh_schemas();
        Ok(DirtySet::from_nodes(dirty))
    }

    /// 依据当前模型重建参数 schema（UI 控件生成用）。
    fn refresh_schemas(&mut self) {
        self.schemas = default_schemas(&self.model);
    }

    // -----------------------------------------------------------------
    // 导出
    // -----------------------------------------------------------------

    /// 导出编辑操作序列为 JSON（确定性：字段顺序固定）。
    ///
    /// 导出**仅含成功应用的编辑**（撤销栈内容按从旧到新 = 时间正向），便于重放——
    /// 重放该序列得到的图与当前图 canonical 字节逐字节一致。
    pub fn export_edits_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\"schema\":");
        out.push_str(&EDITOR_SCHEMA_VERSION.to_string());
        out.push_str(",\"quality\":");
        out.push_str(&super::command::json_string(self.quality.name()));
        out.push_str(",\"draft_spp\":");
        out.push_str(&DRAFT_SPP.to_string());
        out.push_str(",\"final_spp\":");
        out.push_str(&FINAL_SPP.to_string());
        out.push_str(",\"edits\":[");
        for (i, cmd) in self.undo_stack.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&cmd.to_json());
        }
        out.push_str("]}");
        out
    }

    /// 导出当前图为 JSON 描述的节点树（UI 展示）。
    ///
    /// 节点按 id 升序；每个节点含 id / 结构 / 组合方式 / 参数 / 上游连线。
    pub fn export_graph_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\"nodes\":[");
        for (i, node) in self.model.nodes().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"id\":");
            out.push_str(&node.id.to_string());
            out.push_str(",\"structure\":");
            out.push_str(&super::command::json_string(&node.structure.to_string()));
            out.push_str(",\"composition\":");
            out.push_str(&super::command::composition_json(&node.composition));
            out.push_str(",\"params\":[");
            let specs: &[ParamSpec] = self.schemas.get(&node.id).map(Vec::as_slice).unwrap_or(&[]);
            for (j, spec) in specs.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str("{\"name\":");
                out.push_str(&super::command::json_string(&spec.name));
                out.push_str(",\"value\":");
                out.push_str(&super::command::param_value_json(&spec.value));
                out.push_str(",\"min\":");
                out.push_str(&format!("{}", spec.min));
                out.push_str(",\"max\":");
                out.push_str(&format!("{}", spec.max));
                out.push_str(",\"step\":");
                out.push_str(&format!("{}", spec.step));
                out.push('}');
            }
            out.push_str("],\"inputs\":[");
            for (k, (slot, from)) in self.model.inputs_of(node.id).iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{{\"slot\":{slot},\"from\":{from}}}"));
            }
            out.push_str("]}");
        }
        out.push_str("],\"selected\":");
        match self.selected {
            Some(id) => out.push_str(&id.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"dirty\":[");
        for (i, id) in self.dirty_subgraph().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&id.to_string());
        }
        out.push_str("]}");
        out
    }

    /// 导出当前选中节点的参数 schema 为 JSON（UI 生成控件用）。
    pub fn export_schema_json(&self, id: NodeId) -> String {
        let mut out = String::new();
        out.push_str("{\"node\":");
        out.push_str(&id.to_string());
        out.push_str(",\"params\":[");
        let specs: &[ParamSpec] = self.schema_of(id).unwrap_or(&[]);
        for (i, spec) in specs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":");
            out.push_str(&super::command::json_string(&spec.name));
            out.push_str(",\"value\":");
            out.push_str(&super::command::param_value_json(&spec.value));
            out.push_str(",\"min\":");
            out.push_str(&format!("{}", spec.min));
            out.push_str(",\"max\":");
            out.push_str(&format!("{}", spec.max));
            out.push_str(",\"step\":");
            out.push_str(&format!("{}", spec.step));
            out.push('}');
        }
        out.push_str("]}");
        out
    }

    /// 以一组命令**重放到一张新图**，返回重放后的 canonical 字节。
    ///
    /// 确定性断言：对同一初始图 + 同一命令序列，本函数输出与原编辑路径的
    /// [`canonical_bytes`] 逐字节一致。用于「编辑序列重放后图状态字节级一致」。
    pub fn replay_onto_fresh(
        initial: Graph,
        commands: &[EditCommand],
    ) -> Result<Vec<u8>, EditError> {
        let mut state = EditorState::new(initial);
        for cmd in commands {
            state.replay(cmd)?;
        }
        Ok(state.canonical_bytes())
    }
}

/// 导出错误（保留供未来扩展；当前为空类型，CLI 边界可回退到字符串）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    /// 导出前的签名计算失败。
    SignatureFailed,
}

/// 由模型构造默认参数 schema（UI 控件生成用）。
///
/// 范围与步进取保守默认（数值 [0,1] 步进 0.01），真实 schema 由节点注册表
/// （prism-core registry）在未来轮次接入。
fn default_schemas(model: &EditorModel) -> BTreeMap<NodeId, Vec<ParamSpec>> {
    let mut map = BTreeMap::new();
    for node in model.nodes() {
        map.insert(node.id, schemas_for_node(node));
    }
    map
}

/// 单节点的参数 schema：遍历其参数包，按类型生成规格。
fn schemas_for_node(node: &Node) -> Vec<ParamSpec> {
    let mut specs = Vec::new();
    for name in node.params.names() {
        let value: ParamValue = node
            .params
            .get(name)
            .cloned()
            .unwrap_or(ParamValue::F32(0.0));
        let (min, max, step): (f32, f32, f32) = match value {
            ParamValue::F32(_) => (0.0, 1.0, 0.01),
            ParamValue::U32(_) => (0.0, 64.0, 1.0),
            ParamValue::Bool(_) => (0.0, 1.0, 1.0),
            ParamValue::Str(_) => (0.0, 0.0, 0.0),
        };
        specs.push(ParamSpec::new(name.to_string(), value, min, max, step));
    }
    specs
}

/// 构造编辑器 harness 的**默认参考图**（D-14 地基风格的 4 节点链）。
///
/// 链形：
/// ```text
/// 1 叶(ScalarField)  ┐
///                    3 Merge(Plus) ── 4 Chain
/// 2 叶(Convolution)  ┘
/// ```
/// 改叶 1 参数 → 受影响子图恰为 {1, 3, 4}（兄弟叶 2 不受影响），
/// 是增量失效的教科书样例（见测试）。
pub fn build_reference_graph() -> Graph {
    use prism_core::node::MergeOp;
    let mut g = Graph::new();
    let mut n1 = Node::new(1, StructureId::ScalarField, Composition::Feed);
    n1.params.set("scale", ParamValue::F32(0.5));
    n1.params.set("octaves", ParamValue::U32(4));
    let mut n2 = Node::new(2, StructureId::Convolution, Composition::Feed);
    n2.params.set("radius", ParamValue::F32(0.3));
    let mut n3 = Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
    n3.params.set("strength", ParamValue::F32(0.7));
    let mut n4 = Node::new(4, StructureId::Convolution, Composition::Chain);
    n4.params.set("radius", ParamValue::F32(0.4));

    g.add_node(n1).expect("参考图节点 1 加入应成功");
    g.add_node(n2).expect("参考图节点 2 加入应成功");
    g.add_node(n3).expect("参考图节点 3 加入应成功");
    g.add_node(n4).expect("参考图节点 4 加入应成功");
    g.connect(1, 3, 0).expect("参考图连线 1→3 应成功");
    g.connect(2, 3, 1).expect("参考图连线 2→3 应成功");
    g.connect(3, 4, 0).expect("参考图连线 3→4 应成功");
    g
}
