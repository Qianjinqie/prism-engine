//! 同步原语与 Pass 类型封闭枚举（GPU 基础设施，D0）。
//!
//! 语义权威：屏障/同步原语封装、"GPU 用途全景"段
//! （抽象层需覆盖的**形态封闭**五类 pass + 同步）、D-15（自研抽象层）、D-16
//! （Mini Code：公开 API 面最小化）。
//!
//! 本模块定义引擎自定义的**有限 pass 类型**（非 wgpu 式任意兼容层）：
//! ①compute dispatch ②fullscreen/blit ③最小三角形光栅（SDF 文本/粒子 sprite）
//! ④纹理/buffer 上传与读回 ⑤时间戳查询；外加**屏障/同步**语义。
//!
//! 设备无关的**语义模型**（pass 图 + 依赖导出 + 屏障需求推导）在本模块以纯
//! 数据变换实现，可在 CPU 上真跑真测；真机的物理屏障（Vulkan pipeline
//! barrier / WebGPU memory barrier）由 backend 依本模型派生指令，标注待接入。

/// GPU pass 形态（封闭五类）。
///
/// 枚举封闭设计（D-15：引擎自定义有限形态），新增形态须同步抽象层设计文档。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    /// ① 计算派发（compute dispatch）：通用 kernel，网格尺寸由 backend 展开。
    Compute,
    /// ② 全屏 / blit pass：后处理链、分级、色调映射、拷贝。
    FullscreenBlit,
    /// ③ 最小三角形光栅管线：SDF 文本 / 粒子 sprite。
    TriangleRaster,
    /// ④ 纹理 / buffer 上传与读回（CPU↔GPU 数据搬运边界）。
    Transfer,
    /// ⑤ 时间戳查询：效率标注（效率账）。
    TimestampQuery,
}

impl PassKind {
    /// 该 pass 形态是否向后续 pass 暴露可读资源写入（决定屏障需求）。
    ///
    /// 时间戳查询只写查询缓冲、不影响渲染资源依赖；上传只写目标资源。
    pub fn writes_resources(self) -> bool {
        !matches!(self, PassKind::TimestampQuery)
    }

    /// 资源访问方向（读/写）——派生屏障时用于判定 hazard。
    pub fn access(self) -> Access {
        match self {
            PassKind::Compute | PassKind::TriangleRaster => Access::ReadWrite,
            PassKind::FullscreenBlit => Access::Write,
            PassKind::Transfer => Access::Write,
            PassKind::TimestampQuery => Access::Write,
        }
    }
}

/// 资源访问方向（派生屏障用的静态语义）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

/// 一次 pass 描述：形态 + 读写资源集合（资源以编排侧分配的句柄 id 标记）。
#[derive(Debug, Clone)]
pub struct PassDesc {
    /// pass 形态。
    pub kind: PassKind,
    /// 本 pass 读取的资源句柄。
    pub reads: Vec<u64>,
    /// 本 pass 写入的资源句柄。
    pub writes: Vec<u64>,
}

impl PassDesc {
    /// 构造 pass 描述。
    pub fn new(kind: PassKind, reads: Vec<u64>, writes: Vec<u64>) -> Self {
        Self { kind, reads, writes }
    }
}

/// 屏障类型（物理屏障的最小封闭集合）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierKind {
    /// 读→写：先读后写同一资源（RAW 前的 WAR 保护）。
    WriteToRead,
    /// 写→读：写后他处读（必须等待写入完成，RAW）。
    ReadToWrite,
    /// 写→写：同一资源两次写入串行化（WAW）。
    WriteToWrite,
    /// 执行依赖：仅排序阶段，无资源 hazard（如时间戳）。
    Execution,
}

/// 一条屏障需求：作用于资源 `resource`，语义为 [`BarrierKind`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Barrier {
    /// 目标资源句柄。
    pub resource: u64,
    /// 屏障类型。
    pub kind: BarrierKind,
}

/// 队列中的一次同步提交：在 `before_pass` 与 `after_pass` 之间插入屏障集。
#[derive(Debug, Clone)]
pub struct SyncPoint {
    /// 屏障前 pass 序号。
    pub before_pass: usize,
    /// 屏障后 pass 序号。
    pub after_pass: usize,
    /// 该点需插入的屏障。
    pub barriers: Vec<Barrier>,
}

/// 屏障推导器：由**线性 pass 序**推导资源 hazard 与所需屏障（纯函数、确定性）。
///
/// 记录每资源的最近写者与读者集合；相邻 pass 间若存在同一资源的方向冲突，
/// 产出对应屏障。输出屏障集与 pass 序一一对应，可逐点复现。
#[derive(Debug, Clone, Default)]
pub struct BarrierPlanner {
    /// 每资源最近写者的 pass 序号。
    last_writer: std::collections::BTreeMap<u64, usize>,
    /// 每资源自最近写后已读的标记（用于 WAW/WAR 判定）。
    read_since_write: std::collections::BTreeSet<u64>,
}

impl BarrierPlanner {
    /// 空规划器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 为第 `pass_idx` 次 pass 推导其**入口屏障**（相对前一 pass）。
    ///
    /// - 资源在本 pass 被写、且自上次写后已被读过 ⇒ WAR（WriteToRead）防假
    ///   依赖；若上次也是写 ⇒ WAW（WriteToWrite）真依赖。
    /// - 资源在本 pass 被读、且上次为写 ⇒ RAW（ReadToWrite）真依赖。
    ///
    /// 返回入口屏障列表（确定性顺序：按资源句柄升序）。同时更新内部状态。
    pub fn plan_pass(&mut self, pass_idx: usize, desc: &PassDesc) -> Vec<Barrier> {
        let mut barriers: Vec<Barrier> = Vec::new();

        // 本 pass 读 ⇒ 上次写需 RAW 屏障。
        for &r in &desc.reads {
            if self.last_writer.contains_key(&r) {
                barriers.push(Barrier { resource: r, kind: BarrierKind::ReadToWrite });
                self.read_since_write.insert(r);
            }
        }
        // 本 pass 写 ⇒ 处理与上次写的关系。
        for &w in &desc.writes {
            if let Some(&_prev) = self.last_writer.get(&w) {
                // 上次写后已被读：真依赖仍是写序（WAW），若被读过另需 WAR。
                if self.read_since_write.contains(&w) {
                    barriers.push(Barrier { resource: w, kind: BarrierKind::WriteToRead });
                }
                barriers.push(Barrier { resource: w, kind: BarrierKind::WriteToWrite });
            }
            self.last_writer.insert(w, pass_idx);
            self.read_since_write.remove(&w);
        }
        // 时间戳查询只排序阶段，无资源 hazard。
        if desc.kind == PassKind::TimestampQuery && barriers.is_empty() {
            barriers.push(Barrier { resource: u64::MAX, kind: BarrierKind::Execution });
        }
        // 去重 + 按资源句柄稳定排序（确定性）。
        barriers.sort_by(|a, b| a.resource.cmp(&b.resource).then((a.kind as u8).cmp(&(b.kind as u8))));
        barriers.dedup();
        barriers
    }

    /// 为整条 pass 序推导全部同步点（确定性）。
    pub fn plan_sequence(passes: &[PassDesc]) -> Vec<SyncPoint> {
        let mut planner = BarrierPlanner::new();
        let mut points: Vec<SyncPoint> = Vec::new();
        for (i, desc) in passes.iter().enumerate() {
            let barriers = planner.plan_pass(i, desc);
            if i > 0 && !barriers.is_empty() {
                points.push(SyncPoint {
                    before_pass: i - 1,
                    after_pass: i,
                    barriers,
                });
            }
        }
        points
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_kind_closed_set_semantics() {
        // 五类形态封闭；时间戳不写渲染资源。
        assert!(PassKind::Compute.writes_resources());
        assert!(!PassKind::TimestampQuery.writes_resources());
        assert_eq!(PassKind::Compute.access(), Access::ReadWrite);
        assert_eq!(PassKind::FullscreenBlit.access(), Access::Write);
    }

    #[test]
    fn barrier_planner_derives_raw_between_write_and_read() {
        let passes = vec![
            PassDesc::new(PassKind::Compute, vec![], vec![10]), // 写资源 10
            PassDesc::new(PassKind::FullscreenBlit, vec![10], vec![]), // 读资源 10
        ];
        let points = BarrierPlanner::plan_sequence(&passes);
        assert_eq!(points.len(), 1, "两次相关 pass 之间应产生一个同步点");
        assert_eq!(points[0].before_pass, 0);
        assert_eq!(points[0].after_pass, 1);
        assert!(
            points[0].barriers.iter().any(|b| b.resource == 10 && b.kind == BarrierKind::ReadToWrite),
            "写后读应产生 RAW 屏障：{:?}",
            points[0].barriers
        );
    }

    #[test]
    fn barrier_planner_handles_waw_and_war() {
        // 写→写（WAW）；随后写→读。
        let passes = vec![
            PassDesc::new(PassKind::Transfer, vec![], vec![5]),
            PassDesc::new(PassKind::Compute, vec![], vec![5]),
            PassDesc::new(PassKind::FullscreenBlit, vec![5], vec![]),
        ];
        let points = BarrierPlanner::plan_sequence(&passes);
        // pass1 相对 pass0：WAW。
        assert!(
            points[0].barriers.iter().any(|b| b.resource == 5 && b.kind == BarrierKind::WriteToWrite),
            "同资源连续写应产生 WAW：{:?}",
            points[0].barriers
        );
        // pass2 相对 pass1：RAW。
        assert!(
            points[1].barriers.iter().any(|b| b.resource == 5 && b.kind == BarrierKind::ReadToWrite),
            "写后读应产生 RAW：{:?}",
            points[1].barriers
        );
    }

    #[test]
    fn independent_resources_need_no_barrier() {
        let passes = vec![
            PassDesc::new(PassKind::Compute, vec![], vec![1]),
            PassDesc::new(PassKind::FullscreenBlit, vec![], vec![2]),
        ];
        let points = BarrierPlanner::plan_sequence(&passes);
        assert!(points.is_empty(), "无依赖资源不应产生同步点");
    }

    #[test]
    fn barrier_plan_is_deterministic() {
        let passes = vec![
            PassDesc::new(PassKind::Compute, vec![], vec![3, 1, 2]),
            PassDesc::new(PassKind::Compute, vec![1, 2, 3], vec![]),
        ];
        let a = BarrierPlanner::plan_sequence(&passes);
        let b = BarrierPlanner::plan_sequence(&passes);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.barriers, y.barriers, "屏障集应确定性（同输入两遍一致）");
        }
    }
}
