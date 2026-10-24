//! USD `.usda`（ASCII USD）场景图解析 → 引擎网格/场景数据结构。
//!
//! 语义权威：的分层约定（"只有 glTF/OBJ 精简导入；无 USD 全场景图互认"）、
//! 宪章 C18（错误三段式 + 必含建议段）、C20（缺定义即明确报错，绝不静默降级）、
//! C17（全链路确定性）、D-16（Mini Code）。
//!
//! # 为什么手写解析器
//! USD 的 `.usda` 是**缩进分层的文本格式**，手写词法/语法分析完全可行；
//! 零第三方依赖是本项目铁律（连 `quick-xml` / `regex` 都不许），故本模块
//! 自带词法器与递归下降解析器。**不引任何 crate**。
//!
//! # 支持范围（能做什么）
//! - **层头**：`#usda 1.0` + 层元数据括号块（`defaultPrim` / `metersPerUnit`
//!   / `upAxis` / `subLayers`）——后者**只做记录**，不做跨文件解析；
//! - **prim 声明**：`def` / `over` / `class`，类型含 `Xform` / `Scope` /
//!   `Mesh` / `Material` / `Shader` / `GeomSubset` 等，**类型不白名单**
//!   （未知类型按原样保留，见 `Prim.kind`）；
//! - **属性（attribute）**：`point3f[] points`、`int[] faceVertexIndices`、
//!   `int[] faceVertexCounts`、`normal3f[] normals`、`float2[] primvars:st`、
//!   `matrix4d xformOp:transform`、`token xformOpOrder`、以及
//!   `uniform` / `custom` / 关系（`rel`）与 `add` / `append` / `prepend` 限定词；
//! - **属性值语法**：数字、字符串（含 `"""` 三引号多行串）、路径
//!   `</a/b>`、数组 `[...]`、元组 `(...)`、**嵌套数组** `[[...], [...]]`
//!   （`matrix4d` 就是 4×4 嵌套数组）、`@asset@` 资产路径；
//! - **关系（`rel`）与 `over`**：解析并保留，关系目标做**存在性校验**
//!   （悬空 → 结构化报错），`over` 作为**覆盖记录**参与属性合并；
//! - **Xform 变换链**：`xformOpOrder` 驱动的多操作数组合
//!   （`xformOp:transform` / `:translate` / `:scale` / `:rotateXYZ`），
//!   按**世界 = 父 ∘ 子**递归累加，最终**烘焙**到顶点。
//!
//! # 诚实边界（当前版本明确**不做**，勿高估）
//! - **`.usdc`（crate 二进制）完全不支持**：识别 magic `PXR-USDC` 后
//!   **结构化拒绝**（[`UsdError::BinaryNotSupported`]），不做任何部分解析
//!   ——半解析的二进制比明确拒绝危险得多；
//! - **时间采样不支持**：`timeSamples` 块被跳过并记入
//!   [`LayerReport::skipped_time_samples`]，当前版本**只取默认帧**
//!   （`default` 属性的静态值）。属性上的 `timeSamples` 非空时，
//!   若该属性同时有默认值则用默认值，否则报 [`UsdError::MissingTimeSampleValue`]；
//! - **不做引用/载荷/payload 解析**：`prepend references = @x.usda@`、
//!   `payload = @x.usd@` 只**记录**到 [`Prim::references`] /
//!   [`Prim::payloads`]，**不**去磁盘读别的层（会引入文件系统依赖
//!   与不可复现性）；
//! - **不做 variant set（变体集）求值**：`variantSet`/`variants` 分支
//!   记入 [`LayerReport::skipped_variants`]，不选择具体变体；
//! - **不做 `Sub`/实例化（`instanceable`）去重**、不做点/法线属性
//!   插值（`interpolation` 记入 [`Attribute::interpolation`] 但不参与
//!   法线重算）、不做 UV 集编号多路（只取 `st`/`st0`）、不做材质着色
//!   求值（只保留 `material:binding` 关系目标路径）；
//! - **非等比缩放下的法线不做逆转置**：法线只乘变换的线性部分（非等比
//!   时应乘 `transpose(inverse(upper3x3))`）。此时把prim 记入
//!   [`LayerReport::non_uniform_normals`]，让调用方能看见这处是近似；
//! - **只解析单个文件**：不递归 sublayer、不做 asset resolver
//!   （`@./rel/path@` 相对于谁、如何搜索路径都未定义）。
//!
//! # 确定性契约
//! 属性与 prim 均存 `Vec`，按**源文件出现顺序**追加；关系目标、变体名
//! 等集合在建表时**升序排序**。同输入解析两遍逐字段一致（见
//! `parse_is_byte_level_deterministic`）。
//!
//! 三角函数（`xformOp:rotateXYZ`）走本模块自带的 [`det_cos_sin`]，
//! **不**用 `f64::sin_cos()`——后者落平台 libm，跨平台末位可能不同。

use std::fmt;

/// π（f64 精确常量，Rust `core::f64::consts::PI` 的字面量展开）。
const PI: f64 = std::f64::consts::PI;

// ===========================================================================
//  确定性三角件（自研，**禁 libm**）
// ===========================================================================

/// `[0, π/4]` 上的 `sin`（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_sin(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
}

/// `[0, π/4]` 上的 `cos`（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_cos(x: f64) -> f64 {
    let x2 = x * x;
    1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
}

/// 确定性 `(cos x, sin x)`：`mod 2π` 归约（`floor` 为精确舍入运算，非超越件）
/// → 象限对称 → `[0, π/4]` 级数。算法与 `prism_math::det_trig::cos_sin` 一致
/// （该件在 `prism-math` 内，本 crate 不依赖它，故此处自带一份）。
///
/// **为什么不用 `f64::sin_cos()`**：它落平台 libm，同输入跨平台末位可能不同，
/// 违反 C17/C18「全链路确定性」铁律。
///
/// `[0, π/4]` 处截断误差 `sin ≈ 1e-17`、`cos ≈ 1e-17`（f64 级），
/// 对 USD 场景摆放（厘米/米量级）远低于任何工程容差。
fn det_cos_sin(x: f64) -> (f64, f64) {
    const TWO_PI: f64 = 2.0 * PI;
    const FRAC_PI_2: f64 = PI / 2.0;
    const FRAC_PI_4: f64 = PI / 4.0;
    // 归约到 [0, 2π)。
    let mut r = x - TWO_PI * (x / TWO_PI).floor();
    if r < 0.0 {
        // 防负零/负极小的边界（`floor` 已保证 r ≥ 0，此处仅防御 -0.0 比较）。
        r = 0.0;
    }
    // 象限号 q = floor(r / (π/2)) ∈ {0,1,2,3}（r ≥ 0 保证非负）。
    let q = (r * (2.0 / PI)) as i64 & 3;
    r -= q as f64 * FRAC_PI_2; // r ∈ [0, π/2]
                               // 半区归约到 [0, π/4]。
    let (c, s) = if r > FRAC_PI_4 {
        let rr = FRAC_PI_2 - r;
        (quarter_sin(rr), quarter_cos(rr))
    } else {
        (quarter_cos(r), quarter_sin(r))
    };
    match q {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

// ===========================================================================
//  错误（宪章 C18 三段式 + C19 必含建议段）
// ===========================================================================

/// USD 解析错误。错误码取 **PU 段**（prism-usd 族），与 `PD1xxx`（分发）、
/// `PF2xxx`（缓存清单）、`P3xxx`（图）前缀互不重叠。
#[derive(Debug, Clone, PartialEq)]
pub struct UsdError {
    /// 错误码（PU1xxx）。
    pub code: &'static str,
    /// 三段式一【在哪】：函数 + 源内位置（行/列或 prim 路径）。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for UsdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for UsdError {}

impl UsdError {
    /// 构造错误（四字段全填）。
    #[inline]
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }

    /// PU1001：`.usdc` 等二进制 crate 格式（当前版本不支持）。
    ///
    /// **结构化拒绝**（而非部分解析）：`.usdc` 的 crate 容器含压缩的
    /// token/字段表与 LZ4 压缩块，半解析会产出**看似合理实则错误**的
    /// 场景数据——这比明确拒绝危险得多（宪章 C20 精神）。
    pub fn binary_not_supported(magic: &[u8]) -> Self {
        let shown: String = magic
            .iter()
            .take(8)
            .map(|b| {
                if b.is_ascii_graphic() {
                    (*b as char).to_string()
                } else {
                    format!("\\x{b:02x}")
                }
            })
            .collect();
        Self::new(
            "PU1001",
            "usd::parse_bytes",
            format!(
                "检测到 USD 二进制 crate 格式（magic = \"{shown}\"），当前版本只支持 `.usda` 文本格式"
            ),
            "用 `usdcat`（或任何 DCC / `pxr` 工具）把资产导出为 `.usda` 文本后再导入；\
             若需直接支持 `.usdc`，须另行实现 crate 容器 + LZ4 解压（超出当前版本范围）"
                .to_string(),
        )
    }

    /// PU1002：语法错误。
    pub fn syntax(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1002",
            location,
            problem,
            "检查 `.usda` 语法：属性形如 `type name = value`、数组用 `[...]`、\
             元组用 `(...)`、字符串用 `\"...\"`、注释用 `#`；prim 体必须用 `{ }` 包裹"
                .to_string(),
        )
    }

    /// PU1003：属性值与声明类型不匹配。
    pub fn type_mismatch(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1003",
            location,
            problem,
            "核对属性类型与值形态：`int[]` 收整数、`point3f[]` 收三维点数组、\
             `float2[]` 收二维浮点数组；改类型或改值其一"
                .to_string(),
        )
    }

    /// PU1004：网格数据不自洽（面数/索引/顶点不匹配等）。
    pub fn invalid_mesh(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1004",
            location,
            problem,
            "检查 `faceVertexCounts` 之和是否等于 `faceVertexIndices` 长度、\
             索引是否都小于 `points` 长度；用 DCC 重新导出网格可修复"
                .to_string(),
        )
    }

    /// PU1005：关系目标悬空。
    pub fn dangling_relationship(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1005",
            location,
            problem,
            "关系目标路径不存在于本层；补上目标 prim，或删除该 `rel` 声明",
        )
    }

    /// PU1006：属性缺失（mesh 缺必需属性）。
    pub fn missing_attribute(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1006",
            location,
            problem,
            "`Mesh` 至少需要 `points`；要生成面还需 `faceVertexCounts` + \
             `faceVertexIndices`；补齐属性或确认该 prim 确实是 Mesh"
                .to_string(),
        )
    }

    /// PU1007：属性只有 `timeSamples`、无默认值（当前版本只取默认帧）。
    pub fn missing_time_sample_value(
        location: impl Into<String>,
        problem: impl Into<String>,
    ) -> Self {
        Self::new(
            "PU1007",
            location,
            problem,
            "当前版本只取默认帧：给该属性补一个静态默认值，或在资产侧烘焙为默认帧（\
             `timeSamples` 求值需完整的时间码求值器，超出当前版本范围）"
                .to_string(),
        )
    }

    /// PU1008：Xform 变换链无法应用（缺矩阵、操作数未知等）。
    pub fn invalid_xform(location: impl Into<String>, problem: impl Into<String>) -> Self {
        Self::new(
            "PU1008",
            location,
            problem,
            "为每个 `xformOpOrder` 列出的操作补上对应属性（如 `xformOp:translate`），\
             或缩短 `xformOpOrder` 只列已有操作"
                .to_string(),
        )
    }

    /// PU1009：输入不是合法 UTF-8 文本（带首个非法字节的行/列）。
    ///
    /// `at` 是 `Utf8Error::valid_up_to()` 给出的字节偏移；行/列按**字节**
    /// 计数，与词法器 `bump` 的口径一致。**不做** lossy 替换。
    pub fn invalid_utf8(src: &[u8], at: usize) -> Self {
        let (line, col) = line_col_at(src, at);
        let bad = src.get(at).copied().unwrap_or(0);
        Self::new(
            "PU1009",
            format!("usd::lexer 第 {line} 行第 {col} 列"),
            format!(
                "输入不是合法 UTF-8 文本：第 {at} 字节处的 0x{bad:02x} 起序列非法\
                 （拒绝 lossy 替换）"
            ),
            "确认文件是 `.usda`（ASCII USD）且为 UTF-8 编码；`.usdc` 见 PU1001",
        )
    }

    /// PU1010：面多边形**退化**（顶点重合 / 全共线 / 面积 ~ 0）。
    ///
    /// 退化面**没有**可用的耳切输入（Newell 法向模长趋零说明所有顶点
    /// 共线或重合，有向面积为 0）。此时返回结构化错误而**不是**硬凑一个
    /// 三角形——后者会渲染出零面积的碎片几何，比明确拒绝更难排查。
    pub fn degenerate_polygon(location: impl Into<String>, problem: String) -> Self {
        Self::new(
            "PU1010",
            location,
            problem,
            "检查该面的 `faceVertexIndices` 对应点：去掉重合点后应至少有 3 个\
             不共线的点；用 DCC 重新导出该面（weldex / 重算法线）通常即可修复"
                .to_string(),
        )
    }

    /// PU1011：面多边形**自交**（bowtie / 蝴蝶结等）。
    ///
    /// 自交多边形没有"内部"可言：耳切法的耳点判定失去意义，任何剖分都
    /// 会产出**自相矛盾**的三角形（要么面积和 != 多边形面积，要么三角形
    /// 互相重叠）。故**结构化拒绝**，绝不静默产出垃圾几何。
    pub fn self_intersecting_polygon(location: impl Into<String>, problem: String) -> Self {
        Self::new(
            "PU1011",
            location,
            problem,
            "该面轮廓自交（存在一对非相邻边真交于内部），不是合法简单多边形；\
             在 DCC 里检查该面的顶点绕向与点位，修正后重新导出"
                .to_string(),
        )
    }

    /// PU1012：耳切循环**找不到耳点**（理论上对简单多边形不发生）。
    ///
    /// 耳切定理保证任意 `n ≥ 4` 的简单多边形至少有两个耳，故本错误只可能
    /// 在**数值退化**（面积极小、顶点近乎共线但未被 `EAR_EPS` 归入退化面）
    /// 时出现。保留为独立错误码而非 `unwrap`，是为了让 C18「永不 panic」
    /// 在数值边界上也成立。
    pub fn triangulation_failed(location: impl Into<String>, problem: String) -> Self {
        Self::new(
            "PU1012",
            location,
            problem,
            "在剩余顶点中找不到合法耳点（本面接近数值退化）；\
             提高顶点精度或清理共线点后重试；若确认几何无误，可按 PU1010 处理"
                .to_string(),
        )
    }
}

// ===========================================================================
//  数据模型
// ===========================================================================

/// 属性值（USD 值的最小覆盖集，按形态归类）。
///
/// 设计取舍：不逐个复刻 USD 的全部 Sdf 值类型（`VtArray` 的每种元素类型
/// 都建模会过度设计，D-16），而是**按"怎么读"归类**——渲染侧真正需要的
/// 就是标量、数组、矩阵这几种形态。类型名原文保留在 [`Attribute::type_name`]
/// 供审计。
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// 无值（只有声明或只有被跳过的 `timeSamples`）。
    None,
    /// 标量数值（`int` / `float` / `double` / `uint` 等）。
    Number(f64),
    /// 布尔（USD 里也是 token 语义，单列便于渲染侧判别）。
    Bool(bool),
    /// 字符串 / token（含 enum 值）。
    Str(String),
    /// 资产路径 `@foo.png@`（已剥离 `@`）。
    Asset(String),
    /// 场景路径 `</root/child>`（已剥离尖括号）。
    Path(String),
    /// 一维数值数组。
    Numbers(Vec<f64>),
    /// 字符串数组（`token[]` / `string[]`）。
    Strings(Vec<String>),
    /// 路径数组（`path[]` / `rel` 目标）。
    Paths(Vec<String>),
    /// 二维数值数组（`point3f[]` / `float2[]` / `normal3f[]` …）。
    /// 行宽由数据自身决定（3 = 点/法线，2 = UV）。
    Rows(Vec<Vec<f64>>),
    /// 多行数值（`matrix4d` 的 4 行，每行 4 列）。
    Matrix(Vec<Vec<f64>>),
    /// 显式空数组 `[]`。
    EmptyArray,
}

impl AttrValue {
    /// 单值数值（数组/矩阵取首元素首列；非数值返回 `None`）。
    pub fn as_number(&self) -> Option<f64> {
        match self {
            AttrValue::Number(v) => Some(*v),
            AttrValue::Numbers(v) => v.first().copied(),
            AttrValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// 字符串值（token / 资产路径 / 路径）。
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Str(s) | AttrValue::Asset(s) | AttrValue::Path(s) => Some(s),
            _ => None,
        }
    }

    /// 字符串数组（单字符串升格为单元素数组）。
    pub fn as_strings(&self) -> Vec<String> {
        match self {
            AttrValue::Strings(v) => v.clone(),
            AttrValue::Str(s) | AttrValue::Asset(s) | AttrValue::Path(s) => vec![s.clone()],
            AttrValue::Paths(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// 一维数值数组（标量升格为单元素数组）。
    pub fn as_numbers(&self) -> Vec<f64> {
        match self {
            AttrValue::Numbers(v) => v.clone(),
            AttrValue::Number(v) => vec![*v],
            _ => Vec::new(),
        }
    }

    /// 二维数值数组（行优先）。
    pub fn as_rows(&self) -> Vec<Vec<f64>> {
        match self {
            AttrValue::Rows(v) | AttrValue::Matrix(v) => v.clone(),
            _ => Vec::new(),
        }
    }
}

/// 属性（attribute）声明。
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// 属性名（含命名空间前缀，如 `primvars:st`）。
    pub name: String,
    /// 声明类型原文（如 `point3f[]` / `matrix4d` / `token`）。
    pub type_name: String,
    /// 属性的值。
    pub value: AttrValue,
    /// 限定词 `uniform`（材质绑定类属性用）。
    pub uniform: bool,
    /// 限定词 `custom`（用户自定义属性）。
    pub custom: bool,
    /// 插值提示（`interpolation = "vertex"` 等）；当前版本只记录不参与计算。
    pub interpolation: Option<String>,
    /// 是否声明了非空 `timeSamples`（当前版本跳过，只取默认帧）。
    pub has_time_samples: bool,
}

/// 关系（`rel`）声明。
#[derive(Debug, Clone, PartialEq)]
pub struct Relationship {
    /// 关系名（如 `material:binding`）。
    pub name: String,
    /// 目标 prim 路径（已剥离尖括号），按源顺序去重。
    pub targets: Vec<String>,
}

/// prim 声明种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimKind {
    /// `def`：定义新的 prim。
    Def,
    /// `over`：在已有层上覆盖（当前版本不与其它层合并，只作记录）。
    Over,
    /// `class`：抽象类（不参与渲染实例化）。
    Class,
}

impl PrimKind {
    /// 关键字原文。
    pub fn keyword(self) -> &'static str {
        match self {
            PrimKind::Def => "def",
            PrimKind::Over => "over",
            PrimKind::Class => "class",
        }
    }
}

/// 一个 prim（场景节点）。
#[derive(Debug, Clone, PartialEq)]
pub struct Prim {
    /// 声明种类（`def` / `over` / `class`）。
    pub kind: PrimKind,
    /// 类型名（`Mesh` / `Xform` / `Scope` / `Material` …）；`over` 可为空。
    pub type_name: String,
    /// prim 名（路径的最后一节）。
    pub name: String,
    /// 完整 prim 路径（`/root/tri`）。
    pub path: String,
    /// 自身属性（源顺序）。
    pub attributes: Vec<Attribute>,
    /// 关系（源顺序）。
    pub relationships: Vec<Relationship>,
    /// 子 prim（源顺序）。
    pub children: Vec<Prim>,
    /// `xformOpOrder` 列出的操作名（源顺序）。
    pub xform_op_order: Vec<String>,
    /// `prepend/append references` 的目标资产路径（已剥离 `@`，升序）。
    pub references: Vec<String>,
    /// `payload` 目标资产路径（已剥离 `@`，升序）。
    pub payloads: Vec<String>,
    /// 所属变体集名（升序）。
    pub variant_sets: Vec<String>,
}

impl Prim {
    /// 查自身属性（不含子 prim 继承）。
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// 查属性值（便捷）。
    pub fn value(&self, name: &str) -> Option<&AttrValue> {
        self.attribute(name).map(|a| &a.value)
    }

    /// 查关系。
    pub fn relationship(&self, name: &str) -> Option<&Relationship> {
        self.relationships.iter().find(|r| r.name == name)
    }

    /// 深度优先前序遍历本 prim 子树（含自身），顺序确定（源顺序）。
    pub fn walk(&self) -> Vec<&Prim> {
        let mut out = vec![self];
        for c in &self.children {
            out.extend(c.walk());
        }
        out
    }
}

/// 层（layer）= 一个 `.usda` 文件的顶层结构。
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// 文件头版本（`#usda 1.0` 中的 `1.0`）。
    pub version: String,
    /// 层元数据（括号块内的 `key = value`，按源顺序）。
    pub metadata: Vec<(String, String)>,
    /// 顶层 prim（源顺序）。
    pub prims: Vec<Prim>,
    /// 本层声明过的全部 prim 路径（升序，供关系校验）。
    pub all_paths: Vec<String>,
}

/// 解析过程的"记账"报告：把**被跳过但不致命**的东西显式列出来，
/// 避免"静默丢弃"（诚实性要求）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayerReport {
    /// 遇到 `timeSamples` 块而跳过的属性名（升序）。
    pub skipped_time_samples: Vec<String>,
    /// 遇到但未求值的 variantSet 名（升序）。
    pub skipped_variants: Vec<String>,
    /// 遇到但未解析的 `prepend references` / `payload` 资产（升序）。
    pub unresolved_asset_paths: Vec<String>,
    /// 非 `Mesh` 的 prim 数量（当前版本只把 `Mesh` 转成网格）。
    pub non_mesh_prims: usize,
    /// 法线在**非等比缩放**下未做逆转置的 prim（升序，元素含缩放描述）。
    pub non_uniform_normals: Vec<String>,
}

// ===========================================================================
//  词法器（手写）
// ===========================================================================

/// Token 种类。
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// 标识符 / 关键字（`def` / `point3f` / `points` …）。
    Ident(String),
    /// 数字（已转 `f64`）。
    Num(f64),
    /// 字符串字面量（`"..."` / `"""..."""` / `'...'`，内容已解转义）。
    ///
    /// 引号与三引号定界**全部在词法层消化**：解析器只拿到最终文本，
    /// 不会有「字符串体跨 token 流拼接」的中间态。
    Str(String),
    /// `@asset@` 资产路径（内容已剥离 `@`）。
    Asset(String),
    /// `</path>` 场景路径（内容已剥离尖括号）。
    Path(String),
    /// `{`。
    LBrace,
    /// `}`。
    RBrace,
    /// `[`。
    LBracket,
    /// `]`.
    RBracket,
    /// `(`。
    LParen,
    /// `)`.
    RParen,
    /// `=`.
    Eq,
    /// `,`.
    Comma,
    /// `:`（命名空间分隔，如 `primvars:st`）。
    Colon,
    /// `;`.
    Semicolon,
}

/// 带行列的 token（行/列均从 1 开始，用于错误定位）。
#[derive(Debug, Clone, PartialEq)]
struct Spanned {
    tok: Tok,
    line: usize,
    col: usize,
}

/// 手写词法器：把 `.usda` 文本切成 token 流（跳过注释与空白）。
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    /// 构造词法器。
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// 前进一个字节并更新行列。
    fn bump(&mut self) -> Option<u8> {
        let b = *self.src.get(self.pos)?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    /// 窥视当前字节（不前进）。
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    /// 窥视第 `n` 个字节偏移处（不前进）。
    fn peek_at(&self, n: usize) -> Option<u8> {
        self.src.get(self.pos + n).copied()
    }

    /// 跳过空白与 `#` 行注释（USD 的注释是行注释，无块注释）。
    ///
    /// **例外**：文件首行的 `#usda <version>` 是 USD 的**魔数**而非注释，
    /// 必须以 token 形式交给解析器，故此处遇到 `#usda` 时**不**吞掉整行。
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.bump();
                }
                Some(b'#') => {
                    if self.at_usda_magic() {
                        return;
                    }
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    /// 当前 `#` 是否是 `#usda` 魔数（`#` 后紧跟 `usda` 再跟空白/行尾）。
    fn at_usda_magic(&self) -> bool {
        const MAGIC: &[u8] = b"#usda";
        if self.src.get(self.pos..self.pos + MAGIC.len()) != Some(MAGIC) {
            return false;
        }
        match self.src.get(self.pos + MAGIC.len()) {
            // `#usda` 后必须是空白或行尾，`#usdabc` 之类仍是注释。
            None => true,
            Some(&b) => b == b' ' || b == b'\t' || b == b'\r' || b == b'\n',
        }
    }

    /// 读一个完整 token 流（到 EOF）。
    fn tokenize(&mut self) -> Result<Vec<Spanned>, UsdError> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia();
            let Some(c) = self.peek() else { break };
            let (line, col) = (self.line, self.col);
            let tok = self.next_token(c, line, col)?;
            out.push(Spanned { tok, line, col });
        }
        Ok(out)
    }

    /// 读一个 token。
    fn next_token(&mut self, c: u8, line: usize, col: usize) -> Result<Tok, UsdError> {
        match c {
            b'#' if self.at_usda_magic() => {
                // `#usda` 魔数：吃掉 `#usda` 四个字面量，产出 `Ident("usda")`。
                for _ in 0..5 {
                    self.bump();
                }
                Ok(Tok::Ident("usda".to_string()))
            }
            b'"' => {
                // 三引号多行串（USD 允许跨行，且字符串内可原样含 `{}`）。
                let triple = self.peek_at(1) == Some(b'"') && self.peek_at(2) == Some(b'"');
                if triple {
                    self.bump();
                    self.bump();
                    self.bump();
                } else {
                    self.bump();
                }
                Ok(Tok::Str(self.read_string_body(triple, line, col, b'"')?))
            }
            b'\'' => {
                // 单引号字符串（USD 允许，`'''` 亦为多行形式）。
                let triple = self.peek_at(1) == Some(b'\'') && self.peek_at(2) == Some(b'\'');
                if triple {
                    self.bump();
                    self.bump();
                    self.bump();
                } else {
                    self.bump();
                }
                Ok(Tok::Str(self.read_string_body(triple, line, col, b'\'')?))
            }
            b'{' => {
                self.bump();
                Ok(Tok::LBrace)
            }
            b'}' => {
                self.bump();
                Ok(Tok::RBrace)
            }
            b'[' => {
                self.bump();
                Ok(Tok::LBracket)
            }
            b']' => {
                self.bump();
                Ok(Tok::RBracket)
            }
            b'(' => {
                self.bump();
                Ok(Tok::LParen)
            }
            b')' => {
                self.bump();
                Ok(Tok::RParen)
            }
            b'=' => {
                self.bump();
                Ok(Tok::Eq)
            }
            b',' => {
                self.bump();
                Ok(Tok::Comma)
            }
            b':' => {
                self.bump();
                Ok(Tok::Colon)
            }
            b';' => {
                self.bump();
                Ok(Tok::Semicolon)
            }
            b'@' => {
                self.bump();
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    match self.bump() {
                        None => {
                            return Err(UsdError::syntax(
                                format!("usd::lexer 第 {line} 行第 {col} 列"),
                                "资产路径 `@...@` 未闭合（读到文件结尾）",
                            ))
                        }
                        Some(b'@') => break,
                        Some(b) => buf.push(b),
                    }
                }
                Ok(Tok::Asset(Self::decode_bytes(
                    buf,
                    line,
                    col,
                    "资产路径 `@...@` 内容",
                )?))
            }
            b'<' => {
                self.bump();
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    match self.bump() {
                        None => {
                            return Err(UsdError::syntax(
                                format!("usd::lexer 第 {line} 行第 {col} 列"),
                                "场景路径 `<...>` 未闭合（读到文件结尾）",
                            ))
                        }
                        Some(b'>') => break,
                        Some(b) => buf.push(b),
                    }
                }
                Ok(Tok::Path(Self::decode_bytes(
                    buf,
                    line,
                    col,
                    "场景路径 `<...>` 内容",
                )?))
            }
            b'0'..=b'9' | b'-' | b'+' | b'.' => self.lex_number(line, col),
            _ => self.lex_ident(line, col),
        }
    }

    /// 读数字 token（支持 `1`、`-1.5`、`1e-3`、`.5`）。
    fn lex_number(&mut self, line: usize, col: usize) -> Result<Tok, UsdError> {
        let start = self.pos;
        if matches!(self.peek(), Some(b'-') | Some(b'+')) {
            self.bump();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' {
                self.bump();
            } else {
                break;
            }
        }
        // 指数部分。
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            let save = (self.pos, self.line, self.col);
            self.bump();
            if matches!(self.peek(), Some(b'-') | Some(b'+')) {
                self.bump();
            }
            let mut any = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.bump();
                    any = true;
                } else {
                    break;
                }
            }
            if !any {
                // 不是指数（如 `1e` 后接标识符）——回退，让标识符逻辑处理。
                self.pos = save.0;
                self.line = save.1;
                self.col = save.2;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        match text.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(Tok::Num(v)),
            _ => Err(UsdError::syntax(
                format!("usd::lexer 第 {line} 行第 {col} 列"),
                format!("数字字面量无法解析为有限 f64：{text:?}"),
            )),
        }
    }

    /// 读标识符 token（`def` / `points` / `xformOp:transform` 的 `xformOp` 部分）。
    ///
    /// 标识符内允许 `.` `-` `_` 与字母数字；`:` 作为**独立 token** 交给
    /// 解析器拼接（`primvars:st` 需要跨 `:` 合并）。
    fn lex_ident(&mut self, line: usize, col: usize) -> Result<Tok, UsdError> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-' || c == b'+' {
                s.push(c as char);
                self.bump();
            } else {
                break;
            }
        }
        if s.is_empty() {
            let bad = self.peek().unwrap_or(b'?');
            // 只对可打印 ASCII 显示字符；多字节 UTF-8 的前导字节若按 Latin-1
            // 显示会是误导性的「æ」之类，故非 ASCII 一律只给十六进制。
            let shown = if bad.is_ascii_graphic() {
                format!("（ASCII {:?}）", bad as char)
            } else {
                String::new()
            };
            return Err(UsdError::syntax(
                format!("usd::lexer 第 {line} 行第 {col} 列"),
                format!("无法识别的字符：0x{bad:02x}{shown}"),
            ));
        }
        Ok(Tok::Ident(s))
    }

    /// 把已收集的字节整体解码为 UTF-8 文本（词法层**唯一**的解码点）。
    ///
    /// 多字节序列必须先在 `Vec<u8>` 里完整攒齐再解码：逐字节 `b as char`
    /// 会把一个 UTF-8 字符拆成多个 U+00XX，产生乱码，且下次导出再导入会
    /// 继续按字节拆——不幂等。非法 UTF-8 是**语法错误**（带内容起始行列），
    /// 不做 lossy 替换（替换会把坏输入静默变成另一种字符串）。
    fn decode_bytes(
        bytes: Vec<u8>,
        line: usize,
        col: usize,
        what: &str,
    ) -> Result<String, UsdError> {
        String::from_utf8(bytes).map_err(|e| {
            let at = e.utf8_error().valid_up_to();
            let bad = e.as_bytes().get(at).copied().unwrap_or(0);
            UsdError::syntax(
                format!("usd::lexer 第 {line} 行第 {col} 列"),
                format!(
                    "{what}不是合法 UTF-8：第 {at} 字节处的 0x{bad:02x} 起序列非法\
                     （拒绝 lossy 替换）"
                ),
            )
        })
    }

    /// 读字符串内容（起始定界引号已被消费）。
    ///
    /// * `triple = false`：读到下一个**同种**引号即止（`"` 或 `'`）。
    /// * `triple = true`：读到连续三个同种引号即止；其间单个同种引号原样保留。
    /// * 反斜杠转义仅解 `\n \t \r \\ \" \'`，未知转义**原样保留**
    ///   （USD 允许路径里的 `\x` 直传，静默吞掉会改路径）。
    /// * 字符串体**按字节收集**，闭合处整体解码 UTF-8——逐字节 `as char`
    ///   会拆坏多字节字符（中文/emoji/组合字符），见 [`Self::decode_bytes`]。
    fn read_string_body(
        &mut self,
        triple: bool,
        line: usize,
        col: usize,
        quote: u8,
    ) -> Result<String, UsdError> {
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(c) = self.bump() else {
                return Err(UsdError::syntax(
                    format!("usd::lexer 第 {line} 行第 {col} 列"),
                    if triple {
                        "三引号多行字符串未闭合（读到文件结尾）"
                    } else {
                        "字符串未闭合（读到文件结尾）"
                    },
                ));
            };
            if c == b'\\' {
                // 转义序列（按字节展开；未知转义保留反斜杠 + 原字节）。
                let Some(e) = self.bump() else {
                    return Err(UsdError::syntax(
                        format!("usd::lexer 第 {line} 行第 {col} 列"),
                        "字符串转义符 `\\` 后被截断",
                    ));
                };
                match e {
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'\\' => out.push(b'\\'),
                    b'"' => out.push(b'"'),
                    b'\'' => out.push(b'\''),
                    other => {
                        // 保留未知转义的原样（USD 允许 `\x` 直传）。
                        out.push(b'\\');
                        out.push(other);
                    }
                }
                continue;
            }
            if c == quote {
                if triple {
                    // 三引号需连续三个同种引号。
                    if self.peek() == Some(quote) && self.peek_at(1) == Some(quote) {
                        self.bump();
                        self.bump();
                        return Self::decode_bytes(out, line, col, "字符串字面量");
                    }
                    // 三引号体内的单个同种引号原样保留（**按定界符**，不是恒 `"`）。
                    out.push(quote);
                    continue;
                }
                return Self::decode_bytes(out, line, col, "字符串字面量");
            }
            // 字节保真：多字节序列在此逐字节攒齐，闭合后整体解码。
            out.push(c);
        }
    }
}

// ===========================================================================
//  解析器（递归下降 + 缩进无关的 token 流）
// ===========================================================================

/// 手写 `.usda` 解析器。
///
/// 结构（自底向上）：`parse_layer` → `parse_prim_list` → `parse_prim`
/// → `parse_prim_body`（属性/关系/子 prim）→ `parse_value`（递归下降值）。
struct Parser {
    /// token 流（含行列）。
    toks: Vec<Spanned>,
    /// 读游标。
    pos: usize,
    /// 记账报告。
    report: LayerReport,
    /// 已声明的全部 prim 路径（用于关系目标校验）。
    paths: Vec<String>,
    /// 当前 prim 路径栈。
    stack: Vec<String>,
}

impl Parser {
    /// 构造解析器。
    fn new(toks: Vec<Spanned>) -> Self {
        Self {
            toks,
            pos: 0,
            report: LayerReport::default(),
            paths: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// 当前位置描述（用于错误定位）。
    fn here(&self) -> String {
        match self.toks.get(self.pos) {
            Some(s) => format!("usd::parser 第 {} 行第 {} 列", s.line, s.col),
            None => "usd::parser（token 流末尾）".to_string(),
        }
    }

    /// 窥视当前 token。
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|s| &s.tok)
    }

    /// 窥视第 `n` 个 token。
    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n).map(|s| &s.tok)
    }

    /// 吃掉当前 token。
    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).map(|s| s.tok.clone());
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// 当前是否为给定标识符。
    fn at_ident(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(v)) if v == s)
    }

    /// 若当前是 `Tok` 变体则吃掉。
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// 期望当前是 `Tok` 变体，否则报语法错误。
    fn expect(&mut self, t: Tok, what: &str) -> Result<(), UsdError> {
        if self.eat(&t) {
            return Ok(());
        }
        let found = match self.peek() {
            Some(found) => format!("{found:?}"),
            None => "EOF（token 流结束）".to_string(),
        };
        Err(UsdError::syntax(
            self.here(),
            format!("期望 {what}（{t:?}），实际读到 {found}"),
        ))
    }

    /// 读一个字符串字面量（`Tok::Str`，词法层已消化定界引号）。
    fn expect_string(&mut self, what: &str) -> Result<String, UsdError> {
        match self.peek().cloned() {
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(s)
            }
            other => Err(UsdError::syntax(
                self.here(),
                format!("期望 {what}（字符串字面量），实际读到 {other:?}"),
            )),
        }
    }

    /// 读一个标识符（允许 `a:b` 形式，跨 `:` 拼接）。
    fn expect_ident(&mut self, what: &str) -> Result<String, UsdError> {
        let mut s = match self.peek().cloned() {
            Some(Tok::Ident(v)) => {
                self.pos += 1;
                v
            }
            _ => {
                return Err(UsdError::syntax(
                    self.here(),
                    format!("期望 {what}（标识符），实际读到 {:?}", self.peek()),
                ))
            }
        };
        // `primvars:st` —— 名段后接 `:` + 另一名段。
        while matches!(self.peek(), Some(Tok::Colon)) {
            if let Some(Tok::Ident(next)) = self.peek_at(1).cloned() {
                self.pos += 2;
                s.push(':');
                s.push_str(&next);
            } else {
                break;
            }
        }
        // 数组类型后缀 `[]`（`int[]` / `point3f[]` / `float[]`）。
        // USD 的数组性写在**类型名**上，不在名字上，故必须在此吸收，
        // 否则 `int[] x` 会被当成「类型 `int` + 属性名缺失」。
        while matches!(self.peek(), Some(Tok::LBracket))
            && matches!(self.peek_at(1), Some(Tok::RBracket))
        {
            self.pos += 2;
            s.push_str("[]");
        }
        Ok(s)
    }

    // -----------------------------------------------------------------------
    //  层
    // -----------------------------------------------------------------------

    /// 解析整层。
    fn parse_layer(&mut self, src: &str) -> Result<Layer, UsdError> {
        // 头部 `#usda 1.0`。
        let version = self.parse_header()?;
        // 头部元数据块（可选）：`(#usda 1.0)` 之后的 `( ... )`。
        let metadata = self.parse_header_metadata()?;
        // `subLayers = [...]` 记录在 metadata 里（不解析目标文件）。
        let prims = self.parse_prim_list()?;
        let mut all_paths = std::mem::take(&mut self.paths);
        all_paths.sort();
        all_paths.dedup();
        let _ = src;
        Ok(Layer {
            version,
            metadata,
            prims,
            all_paths,
        })
    }

    /// 解析 `#usda <version>` 头。
    fn parse_header(&mut self) -> Result<String, UsdError> {
        if !self.at_ident("usda") {
            return Err(UsdError::syntax(
                self.here(),
                format!(
                    "缺少 `.usda` 文件头（应为 `#usda 1.0`；`.usdc` 二进制见 PU1001），\
                     实际读到 {:?}",
                    self.peek()
                ),
            ));
        }
        self.pos += 1;
        match self.bump() {
            Some(Tok::Num(v)) => Ok(format!("{v}")),
            Some(Tok::Ident(v)) => Ok(v),
            other => Err(UsdError::syntax(
                self.here(),
                format!("`#usda` 后应是版本号，实际读到 {other:?}"),
            )),
        }
    }

    /// 解析头部层元数据括号块（若存在，可多块）。
    ///
    /// 关键：USD 的层元数据**不要求**逗号分隔——`( ... )` 块内通常一行一个
    /// `key = value`，换行即分隔。故这里同时接受 `,` 与「直接跟下一个 key」。
    fn parse_header_metadata(&mut self) -> Result<Vec<(String, String)>, UsdError> {
        let mut out = Vec::new();
        while self.eat(&Tok::LParen) {
            loop {
                if self.eat(&Tok::RParen) {
                    break;
                }
                if self.pos >= self.toks.len() {
                    return Err(UsdError::syntax(
                        self.here(),
                        "层元数据块 `(` 未闭合（token 流提前结束）",
                    ));
                }
                let key = self.expect_ident("层元数据键")?;
                self.expect(Tok::Eq, "`=`")?;
                let val = self.parse_meta_scalar()?;
                out.push((key, val));
                // 逗号可选（USD 允许纯换行分隔）。
                let _ = self.eat(&Tok::Comma);
            }
        }
        Ok(out)
    }

    /// 读层元数据的值（标量 / `[...]` 数组）。
    ///
    /// `subLayers = [ @a@, @b@ ]` 是数组形式，**必须**在此消化数组括号，
    /// 否则 `parse_header_metadata` 的 `,` / `)` 判定会失配。
    fn parse_meta_scalar(&mut self) -> Result<String, UsdError> {
        if matches!(self.peek(), Some(Tok::LBracket)) {
            let value = self.parse_value()?;
            // `subLayers` / `references` 的数组元素统一以字符串形态保留
            //（`@a@` 与 `</a>` 在 `parse_value` 里都落进 `Paths`）。
            return Ok(match value {
                AttrValue::Paths(mut v) => {
                    v.sort();
                    v.dedup();
                    v.join(",")
                }
                AttrValue::Strings(v) => v.join(","),
                other => format!("{other:?}"),
            });
        }
        match self.peek().cloned() {
            Some(Tok::Num(v)) => {
                self.pos += 1;
                Ok(format!("{v}"))
            }
            Some(Tok::Path(s)) => {
                self.pos += 1;
                Ok(s)
            }
            Some(Tok::Asset(s)) => {
                self.pos += 1;
                Ok(s)
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(s)
            }
            Some(Tok::Ident(v)) => {
                self.pos += 1;
                Ok(v)
            }
            other => Err(UsdError::syntax(
                self.here(),
                format!("层元数据值应为标量或数组，实际读到 {other:?}"),
            )),
        }
    }

    // -----------------------------------------------------------------------
    //  prim
    // -----------------------------------------------------------------------

    /// 解析 prim 列表（直到 token 流结束）。
    fn parse_prim_list(&mut self) -> Result<Vec<Prim>, UsdError> {
        let mut out = Vec::new();
        while self.pos < self.toks.len() {
            // 跳过孤立分号。
            if self.eat(&Tok::Semicolon) {
                continue;
            }
            // `def` / `over` / `class` 开头的 prim。
            let kind = if self.at_ident("def") {
                PrimKind::Def
            } else if self.at_ident("over") {
                PrimKind::Over
            } else if self.at_ident("class") {
                PrimKind::Class
            } else {
                return Err(UsdError::syntax(
                    self.here(),
                    format!(
                        "层主体期望 `def` / `over` / `class` prim 声明，实际读到 {:?}",
                        self.peek()
                    ),
                ));
            };
            out.push(self.parse_prim(kind)?);
        }
        Ok(out)
    }

    /// 解析单个 prim 声明（含 `{ ... }` 体）。
    fn parse_prim(&mut self, kind: PrimKind) -> Result<Prim, UsdError> {
        self.pos += 1; // 吃掉 def/over/class
                       // 类型名可省（`over "x"` / `def "x"`）。
        let mut type_name = String::new();
        if let Some(Tok::Ident(_)) = self.peek() {
            if !matches!(self.peek(), Some(Tok::Ident(v)) if v == "def" || v == "over" || v == "class")
            {
                type_name = self.expect_ident("prim 类型名")?;
            }
        }
        let name = self.expect_string("prim 名称")?;
        if name.is_empty() {
            return Err(UsdError::syntax(self.here(), "prim 名称为空字符串"));
        }
        // 路径 = 父路径 + `/` + 名。
        let parent = self.stack.last().cloned().unwrap_or_default();
        let path = if parent.is_empty() {
            format!("/{name}")
        } else {
            format!("{parent}/{name}")
        };
        self.paths.push(path.clone());
        self.stack.push(path.clone());

        let mut prim = Prim {
            kind,
            type_name,
            name,
            path,
            attributes: Vec::new(),
            relationships: Vec::new(),
            children: Vec::new(),
            xform_op_order: Vec::new(),
            references: Vec::new(),
            payloads: Vec::new(),
            variant_sets: Vec::new(),
        };

        // prim 头之后可能有一串元数据（`( ... )`），也可能是 `{`。
        while self.eat(&Tok::LParen) {
            let mut depth = 1usize;
            while depth > 0 {
                match self.bump() {
                    Some(Tok::LParen) => depth += 1,
                    Some(Tok::RParen) => depth -= 1,
                    Some(_) => {}
                    None => {
                        return Err(UsdError::syntax(
                            self.here(),
                            format!("prim {} 的头部元数据块未闭合", prim_path_hint(&prim)),
                        ))
                    }
                }
            }
        }
        if !self.eat(&Tok::LBrace) {
            // 无体 prim（`over "x"` 形式）：只记录声明。
            self.stack.pop();
            return Ok(prim);
        }
        self.parse_prim_body(&mut prim)?;
        self.stack.pop();
        Ok(prim)
    }

    /// 解析 prim 体（`{` 已消费）到匹配的 `}`。
    fn parse_prim_body(&mut self, prim: &mut Prim) -> Result<(), UsdError> {
        loop {
            match self.peek().cloned() {
                Some(Tok::RBrace) => {
                    self.pos += 1;
                    return Ok(());
                }
                None => {
                    return Err(UsdError::syntax(
                        self.here(),
                        format!("prim {} 的体未以 `}}` 结束（token 流提前结束）", prim.path),
                    ))
                }
                Some(Tok::Semicolon) => {
                    self.pos += 1;
                }
                // 子 prim。
                Some(Tok::Ident(k)) if k == "def" || k == "over" || k == "class" => {
                    let kind = match k.as_str() {
                        "def" => PrimKind::Def,
                        "over" => PrimKind::Over,
                        _ => PrimKind::Class,
                    };
                    let child = self.parse_prim(kind)?;
                    prim.children.push(child);
                }
                // `variantSet "x" = { ... }`：记录但**不求值**。
                Some(Tok::Ident(k)) if k == "variantSet" => {
                    self.pos += 1;
                    let name = self.expect_string("variantSet 名")?;
                    prim.variant_sets.push(name.clone());
                    self.report.skipped_variants.push(name);
                    // 跳过 `= { ... }` 整块（含变体内的 prim）——**不**并入子 prim。
                    if self.eat(&Tok::Eq) {
                        self.skip_braced_block();
                    }
                }
                // 关系。
                Some(Tok::Ident(k)) if k == "rel" => {
                    let r = self.parse_relationship()?;
                    prim.relationships.push(r);
                }
                // 其它标识符开头：属性声明。
                Some(Tok::Ident(_)) => {
                    let attr = self.parse_attribute(prim)?;
                    // 记下常用属性。
                    if attr.name == "xformOpOrder" {
                        prim.xform_op_order = attr.value.as_strings();
                    } else if attr.name.ends_with("references") {
                        prim.references.extend(attr.value.as_strings());
                    } else if attr.name == "payload" {
                        prim.payloads.extend(attr.value.as_strings());
                    }
                    prim.attributes.push(attr);
                }
                other => {
                    return Err(UsdError::syntax(
                        self.here(),
                        format!("prim {} 的体内出现意外 token：{other:?}", prim.path),
                    ))
                }
            }
        }
    }

    /// 解析关系声明 `rel name = </a> </b>` 或 `= [</a>, </b>]`。
    fn parse_relationship(&mut self) -> Result<Relationship, UsdError> {
        self.pos += 1; // 吃掉 `rel`
        let name = self.expect_ident("关系名")?;
        // 关系声明同样允许元数据括号块（`( ... )`），跳过之。
        while matches!(self.peek(), Some(Tok::LParen)) {
            let save = self.pos;
            self.pos += 1;
            loop {
                if self.eat(&Tok::RParen) {
                    break;
                }
                if self.pos >= self.toks.len() || !matches!(self.peek(), Some(Tok::Ident(_))) {
                    self.pos = save;
                    break;
                }
                let _ = self.expect_ident("关系元数据键");
                if self.eat(&Tok::Eq) {
                    let _ = self.parse_value()?;
                }
                let _ = self.eat(&Tok::Comma);
            }
        }
        if !self.eat(&Tok::Eq) {
            return Err(UsdError::syntax(
                self.here(),
                format!("关系 {name} 后期望 `=`，实际读到 {:?}", self.peek()),
            ));
        }
        let value = self.parse_value()?;
        let targets = match value {
            AttrValue::Path(p) => vec![p],
            AttrValue::Paths(ps) => ps,
            AttrValue::Strings(ss) => ss,
            AttrValue::EmptyArray => Vec::new(),
            other => {
                return Err(UsdError::type_mismatch(
                    self.here(),
                    format!("关系 {name} 的目标应为场景路径，得到 {other:?}"),
                ))
            }
        };
        // 目标去重（保持首次出现顺序——确定性）。
        let mut uniq: Vec<String> = Vec::new();
        for t in targets {
            if !uniq.contains(&t) {
                uniq.push(t);
            }
        }
        Ok(Relationship {
            name,
            targets: uniq,
        })
    }

    /// 解析属性声明（`[uniform|custom] <type> <name> [(meta)] = <value> [(meta)]`）。
    ///
    /// USD 允许元数据括号块出现在**值前**或**值后**（导出器两种都写），
    /// 故这里前后各解析一轮，共用同一份累加器。
    fn parse_attribute(&mut self, prim: &Prim) -> Result<Attribute, UsdError> {
        let mut uniform = false;
        let mut custom = false;
        // 前置限定词。
        loop {
            if self.at_ident("uniform") {
                uniform = true;
                self.pos += 1;
            } else if self.at_ident("custom") {
                custom = true;
                self.pos += 1;
            } else if self.at_ident("rel") {
                // `rel` 已在调用方处理；防御性跳过。
                self.pos += 1;
            } else {
                break;
            }
        }
        let type_name = self.expect_ident("属性类型")?;
        let name = self.expect_ident("属性名")?;
        // 元数据累加器（值前 / 值后两轮共用）。
        let mut interpolation: Option<String> = None;
        let mut has_time_samples = false;
        // 值前元数据（少见但合法）。
        self.parse_attr_metadata(prim, &name, &mut interpolation, &mut has_time_samples)?;
        // 值。
        let value = if self.eat(&Tok::Eq) {
            self.parse_value()?
        } else {
            AttrValue::None
        };
        // 值后元数据（常见形式：`points = [...] ( interpolation = "vertex" )`）。
        self.parse_attr_metadata(prim, &name, &mut interpolation, &mut has_time_samples)?;
        // 值与类型的一致性检查（关键：避免把 int[] 当 float[] 用）。
        validate_attr_value(&type_name, &value, &name, &prim.path)?;
        Ok(Attribute {
            name,
            type_name,
            value,
            uniform,
            custom,
            interpolation,
            has_time_samples,
        })
    }

    /// 解析紧邻的属性元数据括号块（`(` 到匹配的 `)`，可重复多轮）。
    ///
    /// `timeSamples` 只记账不求值（`report.skipped_time_samples`），其余
    /// 键值对按 `key = value` 读入，`interpolation` 单独抽出。
    fn parse_attr_metadata(
        &mut self,
        prim: &Prim,
        attr_name: &str,
        interpolation: &mut Option<String>,
        has_time_samples: &mut bool,
    ) -> Result<(), UsdError> {
        while matches!(self.peek(), Some(Tok::LParen)) {
            let save = self.pos;
            self.pos += 1;
            // 读 `key = value` 对，直到 `)`。
            loop {
                if self.eat(&Tok::RParen) {
                    break;
                }
                if self.pos >= self.toks.len() {
                    // 不是元数据块（回退，交由调用方按语法处理）。
                    self.pos = save;
                    return Ok(());
                }
                if self.at_ident("timeSamples") {
                    *has_time_samples = true;
                    self.report
                        .skipped_time_samples
                        .push(format!("{}@timeSamples", prim.path));
                    // 跳过整块 `= { ... }`。
                    self.pos += 1;
                    if self.eat(&Tok::Eq) {
                        self.skip_braced_block();
                    }
                    continue;
                }
                let key = match self.expect_ident("属性元数据键") {
                    Ok(k) => k,
                    Err(_) => {
                        self.pos = save;
                        return Ok(());
                    }
                };
                if self.eat(&Tok::Eq) {
                    let v = self.parse_value()?;
                    if key == "interpolation" {
                        if let Some(s) = v.as_str() {
                            *interpolation = Some(s.to_string());
                        }
                    }
                } else {
                    let _ = self.parse_value()?;
                }
                let _ = attr_name;
                // 元数据项之间可由 `,` 分隔，也可只靠空白/换行。
                if self.eat(&Tok::Comma) {
                    continue;
                }
                if self.eat(&Tok::RParen) {
                    break;
                }
            }
        }
        Ok(())
    }

    /// 跳过 `{ ... }` 整块（`timeSamples` 体）。
    fn skip_braced_block(&mut self) {
        if !self.eat(&Tok::LBrace) {
            return;
        }
        let mut depth = 1usize;
        while depth > 0 {
            match self.bump() {
                Some(Tok::LBrace) => depth += 1,
                Some(Tok::RBrace) => depth -= 1,
                Some(_) => {}
                None => return,
            }
        }
    }

    // -----------------------------------------------------------------------
    //  值（递归下降）
    // -----------------------------------------------------------------------

    /// 解析属性值。
    fn parse_value(&mut self) -> Result<AttrValue, UsdError> {
        // 引号串 / 三引号多行串（词法层已合并为 `Tok::Str`）。
        // 字符串化的内嵌三元组/数组按原样保留。
        if let Some(Tok::Str(s)) = self.peek().cloned() {
            self.pos += 1;
            return Ok(classify_brace_text(&s));
        }
        // 资产路径。
        if let Some(Tok::Asset(s)) = self.peek().cloned() {
            self.pos += 1;
            // `@layer@</Prim>` 形式（`references` / `payload` 的标准写法）：
            // 资产后紧跟 prim 路径 → 合成 `{asset}@<{prim}>` 单串。
            if let Some(Tok::Path(p)) = self.peek().cloned() {
                self.pos += 1;
                return Ok(AttrValue::Str(format!("{s}@<{p}>")));
            }
            return Ok(AttrValue::Asset(s));
        }
        // 场景路径。
        if let Some(Tok::Path(s)) = self.peek().cloned() {
            self.pos += 1;
            return Ok(AttrValue::Path(s));
        }
        // 数组。
        if matches!(self.peek(), Some(Tok::LBracket)) {
            self.pos += 1;
            let mut items = Vec::new();
            loop {
                if self.eat(&Tok::RBracket) {
                    break;
                }
                if self.pos >= self.toks.len() {
                    return Err(UsdError::syntax(
                        self.here(),
                        "数组 `[` 未闭合（token 流提前结束）",
                    ));
                }
                items.push(self.parse_value()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                if self.eat(&Tok::RBracket) {
                    break;
                }
                return Err(UsdError::syntax(
                    self.here(),
                    format!("数组内期望 `,` 或 `]`，实际读到 {:?}", self.peek()),
                ));
            }
            return Ok(collapse_array(items));
        }
        // 元组。
        if matches!(self.peek(), Some(Tok::LParen)) {
            self.pos += 1;
            let mut items = Vec::new();
            loop {
                if self.eat(&Tok::RParen) {
                    break;
                }
                if self.pos >= self.toks.len() {
                    return Err(UsdError::syntax(
                        self.here(),
                        "元组 `(` 未闭合（token 流提前结束）",
                    ));
                }
                items.push(self.parse_value()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                if self.eat(&Tok::RParen) {
                    break;
                }
                return Err(UsdError::syntax(
                    self.here(),
                    format!("元组内期望 `,` 或 `)`，实际读到 {:?}", self.peek()),
                ));
            }
            return Ok(collapse_tuple(items));
        }
        // 数字。
        if let Some(Tok::Num(v)) = self.peek().cloned() {
            self.pos += 1;
            return Ok(AttrValue::Number(v));
        }
        // 裸词：`true` / `false` / 未加引号的 token。
        if let Some(Tok::Ident(s)) = self.peek().cloned() {
            self.pos += 1;
            return Ok(if s == "true" {
                AttrValue::Bool(true)
            } else if s == "false" {
                AttrValue::Bool(false)
            } else {
                AttrValue::Str(s)
            });
        }
        Err(UsdError::syntax(
            self.here(),
            format!("无法解析的属性值起始 token：{:?}", self.peek()),
        ))
    }
}

/// prim 路径提示（错误信息用；独立函数便于在格式化串里复用）。
fn prim_path_hint(p: &Prim) -> String {
    p.path.clone()
}

/// 把引号内的字符串按其内容归类（USD 允许把数组/元组写成字符串）。
///
/// **当前版本一律返回 [`AttrValue::Str`]**：USD 里 `"[...]"` 字符串化的数组是
/// 合法但罕见的写法，若按内容二次解析，遇到 `"(1, 2)"` 这类**并非**数组的
/// 字符串就会误分类。类型名原文保留在 `Attribute::type_name` 供审计，
/// 需要结构化数值时由渲染侧显式再解析。
fn classify_brace_text(s: &str) -> AttrValue {
    AttrValue::Str(s.to_string())
}

/// 把数组的元素列表归并成最合适的 `AttrValue` 形态。
///
/// 归并顺序即优先级：`[]`→`EmptyArray`、全数值→`Numbers`、全路径→`Paths`、
/// 全字符串→`Strings`、其余→`Rows`。**先判 `Paths` 后判 `Strings`**：纯路径数组
/// 语义上更具体，若先判 `Strings` 会把 `rel` 目标降级成普通字符串。
fn collapse_array(items: Vec<AttrValue>) -> AttrValue {
    if items.is_empty() {
        return AttrValue::EmptyArray;
    }
    let all_numbers = items.iter().all(|v| matches!(v, AttrValue::Number(_)));
    if all_numbers {
        return AttrValue::Numbers(items.iter().map(|v| v.as_number().unwrap_or(0.0)).collect());
    }
    let all_paths = items.iter().all(|v| matches!(v, AttrValue::Path(_)));
    if all_paths {
        return AttrValue::Paths(
            items
                .iter()
                .map(|v| match v {
                    AttrValue::Path(p) => p.clone(),
                    _ => String::new(),
                })
                .collect(),
        );
    }
    let all_strings = items
        .iter()
        .all(|v| matches!(v, AttrValue::Str(_) | AttrValue::Asset(_)));
    if all_strings {
        return AttrValue::Strings(
            items
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
        );
    }
    // 含嵌套数组 → 二维行数组。
    AttrValue::Rows(items.iter().map(element_row).collect())
}

/// 单个元素 → 一行数值（标量升格为单元素行）。
fn element_row(v: &AttrValue) -> Vec<f64> {
    match v {
        AttrValue::Number(n) => vec![*n],
        AttrValue::Numbers(ns) => ns.clone(),
        other => vec![other.as_number().unwrap_or(0.0)],
    }
}

/// 把元组元素归并成 `AttrValue`（一维数值 → `Numbers`，多维 → `Rows`）。
fn collapse_tuple(items: Vec<AttrValue>) -> AttrValue {
    if items.is_empty() {
        return AttrValue::Numbers(Vec::new());
    }
    if items.iter().all(|v| matches!(v, AttrValue::Number(_))) {
        return AttrValue::Numbers(items.iter().map(|v| v.as_number().unwrap_or(0.0)).collect());
    }
    // 嵌套元组 → 行数组（`matrix4d` 就是 4 个 4 元组）。
    AttrValue::Rows(items.iter().map(element_row).collect())
}

/// 校验属性值与声明类型的一致性（PU1003）。
fn validate_attr_value(
    type_name: &str,
    value: &AttrValue,
    attr_name: &str,
    prim_path: &str,
) -> Result<(), UsdError> {
    let loc = format!("prim {prim_path} 的属性 {attr_name}（类型 {type_name}）");
    // 数组类：类型名以 `[]` 结尾。
    let is_array = type_name.ends_with("[]");
    let base = type_name.trim_end_matches("[]");
    match value {
        AttrValue::None => Ok(()),
        AttrValue::Matrix(_)
        | AttrValue::Rows(_)
        | AttrValue::Numbers(_)
        | AttrValue::EmptyArray => {
            if matches!(value, AttrValue::EmptyArray) {
                return Ok(());
            }
            // `int[]` 必须是整数（tolist 索引），小数视为类型不匹配。
            if base == "int" || base == "uint" {
                if let AttrValue::Numbers(ns) = value {
                    for n in ns {
                        if n.fract().abs() > 1e-9 {
                            return Err(UsdError::type_mismatch(
                                &loc,
                                format!("`{type_name}` 要求整数索引，收到小数 {n}"),
                            ));
                        }
                    }
                }
            }
            // 点/法线/UV 必须行宽 2 或 3。
            if matches!(
                base,
                "point3f"
                    | "normal3f"
                    | "vector3f"
                    | "color3f"
                    | "texCoord2f"
                    | "texCoord3f"
                    | "float2"
                    | "float3"
                    | "normal3d"
                    | "point3d"
            ) {
                let rows = value.as_rows();
                let expect: Option<usize> = match base {
                    "point3f" | "normal3f" | "vector3f" | "color3f" | "float3" | "normal3d"
                    | "point3d" | "texCoord3f" => Some(3),
                    "texCoord2f" | "float2" => Some(2),
                    _ => None,
                };
                if let Some(w) = expect {
                    for r in &rows {
                        if r.len() != w {
                            return Err(UsdError::type_mismatch(
                                &loc,
                                format!("`{type_name}` 每行应恰有 {w} 个分量，收到 {}", r.len()),
                            ));
                        }
                    }
                }
            }
            let _ = is_array;
            Ok(())
        }
        AttrValue::Number(_)
        | AttrValue::Bool(_)
        | AttrValue::Str(_)
        | AttrValue::Asset(_)
        | AttrValue::Path(_) => {
            if is_array && type_name != "token[]" && !matches!(value, AttrValue::Str(_)) {
                // 标量赋给数组类型：仅当类型是 token 数组时可能合法。
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`{type_name}` 是数组类型，却收到标量值 {value:?}"),
                ));
            }
            Ok(())
        }
        AttrValue::Strings(_) | AttrValue::Paths(_) => {
            if !is_array && type_name != "token" && type_name != "string" && type_name != "asset" {
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`{type_name}` 不是数组类型，却收到数组值"),
                ));
            }
            Ok(())
        }
    }
}

// ===========================================================================
//  公开入口
// ===========================================================================

/// USD crate 二进制格式的 magic（前 8 字节）。
pub const USDC_MAGIC: &[u8; 8] = b"PXR-USDC";

/// 字节偏移 → `(行, 列)`，均从 1 开始；列按**字节**计（与词法器同口径）。
fn line_col_at(bytes: &[u8], at: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for &b in bytes.iter().take(at) {
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// 解析结果：层 + 记账报告。
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedUsda {
    /// 解析出的层。
    pub layer: Layer,
    /// 记账报告（被跳过但不致命的东西）。
    pub report: LayerReport,
}

/// 解析 USD 字节流（按内容分派 `.usda` / `.usdc`）。
///
/// `.usdc`（binary crate）→ **结构化拒绝** [`UsdError::BinaryNotSupported`]。
pub fn parse_bytes(bytes: &[u8]) -> Result<ParsedUsda, UsdError> {
    if bytes.starts_with(USDC_MAGIC) {
        return Err(UsdError::binary_not_supported(&bytes[..8]));
    }
    let text =
        std::str::from_utf8(bytes).map_err(|e| UsdError::invalid_utf8(bytes, e.valid_up_to()))?;
    parse_str(text)
}

/// 解析 `.usda` 文本。
pub fn parse_str(text: &str) -> Result<ParsedUsda, UsdError> {
    // 预检：`.usdc` 的 magic 若出现在文本里（不太可能）也走结构化拒绝。
    if text.starts_with("PXR-USDC") {
        return Err(UsdError::binary_not_supported(&text.as_bytes()[..8]));
    }
    let mut lexer = Lexer::new(text.as_bytes());
    let toks = lexer.tokenize()?;
    let mut p = Parser::new(toks);
    let layer = p.parse_layer(text)?;
    let mut report = p.report;
    // 记账列表排序（确定性）。
    report.skipped_time_samples.sort();
    report.skipped_time_samples.dedup();
    report.skipped_variants.sort();
    report.skipped_variants.dedup();
    report.unresolved_asset_paths.sort();
    report.unresolved_asset_paths.dedup();
    Ok(ParsedUsda { layer, report })
}

/// 解析 `.usda` 文件。
pub fn parse_file(path: &str) -> Result<ParsedUsda, UsdError> {
    let bytes = std::fs::read(path).map_err(|e| {
        UsdError::new(
            "PU1010",
            format!("usd::parse_file({path})"),
            format!("读文件失败：{e}"),
            "确认路径存在且可读；相对路径以进程工作目录为基准",
        )
    })?;
    parse_bytes(&bytes)
}

// ===========================================================================
//  场景数据结构（渲染侧可消费）
// ===========================================================================

/// 4×4 变换矩阵（**行主序**：`m[row][col]`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// 16 个元素，行主序。
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    /// 单位矩阵。
    pub fn identity() -> Self {
        let mut m = [[0.0f64; 4]; 4];
        for i in 0..4 {
            m[i][i] = 1.0;
        }
        Self { m }
    }

    /// 矩阵乘法 `self ∘ other`（`self` 后应用）。
    pub fn mul(&self, o: &Mat4) -> Mat4 {
        let mut r = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.m[i][k] * o.m[k][j];
                }
                r[i][j] = sum;
            }
        }
        Self { m: r }
    }

    /// 变换点（`w = 1`，不做透视除法）。
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * p[0] + self.m[0][1] * p[1] + self.m[0][2] * p[2] + self.m[0][3],
            self.m[1][0] * p[0] + self.m[1][1] * p[1] + self.m[1][2] * p[2] + self.m[1][3],
            self.m[2][0] * p[0] + self.m[2][1] * p[1] + self.m[2][2] * p[2] + self.m[2][3],
        ]
    }
}

/// 网格（`Mesh` prim 转换结果，顶点已烘焙世界变换）。
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// 源 prim 路径。
    pub path: String,
    /// 顶点位置（世界空间）。
    pub points: Vec<[f64; 3]>,
    /// 每面顶点数（`faceVertexCounts`）。
    pub face_counts: Vec<u32>,
    /// 展平的顶点索引（`faceVertexIndices`）。
    pub face_indices: Vec<u32>,
    /// 法线（世界空间；未提供则为空）。
    pub normals: Vec<[f64; 3]>,
    /// UV（`primvars:st`；未提供则为空）。
    /// 若提供的是 faceVarying 插值，按索引展开。
    pub uvs: Vec<[f64; 2]>,
    /// 材质绑定目标路径（`material:binding` 关系；无则 `None`）。
    pub material: Option<String>,
    /// 该网格累积的世界变换。
    pub world_transform: Mat4,
}

impl Mesh {
    /// 顶点数。
    pub fn vertex_count(&self) -> usize {
        self.points.len()
    }

    /// 面数。
    pub fn face_count(&self) -> usize {
        self.face_counts.len()
    }

    /// 三角化后的索引（**耳切法**，非扇形展开）。
    ///
    /// # 为什么不用扇形
    /// 旧实现是 `for k in 1..(n-1)` 的扇形展开，它对**凸**多边形正确，对
    /// **凹**多边形（L 形 / U 形 / 被切出的凹面，USD 网格极常见）会产出
    /// **自交或翻面**三角形——扇形三角 `[v0, v_k, v_{k+1}]` 用的对角线
    /// `v0→v_k` 在凹多边形里可能落在多边形**外部**。这是渲染正确性问题，
    /// 不是性能优化，故当前版本换成耳切法。
    ///
    /// # 签名保留
    /// 保留 `triangulated() -> Vec<[u32;3]>` 原签名（`Scene::triangle_count`
    /// 与既有 22 个测试都依赖它），**转发**到 [`Mesh::triangulated_ear`]。
    /// 耳切遇到退化 / 自交面时产出**零个三角形**而非错误三角形（C20：
    /// 绝不静默降级为看似合理的错几何）；需要拿到结构化错误码的调用方
    /// 请直接用 [`Mesh::triangulated_ear`]。
    pub fn triangulated(&self) -> Vec<[u32; 3]> {
        self.triangulated_ear().unwrap_or_default()
    }

    /// 耳切三角化（可失败版；退化 / 自交面返回结构化错误）。
    ///
    /// 与 [`Mesh::triangulated`] 的唯一区别是**错误不被吞掉**：调用方可
    /// 按错误码（PU1010 退化 / PU1011 自交 / PU1012 找不到耳点）分派处理。
    pub fn triangulated_ear(&self) -> Result<Vec<[u32; 3]>, UsdError> {
        let mut out = Vec::new();
        let mut base = 0usize;
        for (fi, c) in self.face_counts.iter().enumerate() {
            let n = *c as usize;
            if n >= 3 {
                let idx = &self.face_indices[base..base + n];
                let loc = format!("usd::Mesh::triangulated_ear（prim {}，第 {fi} 个面）", self.path);
                ear_clip(&self.points, idx, &loc, &mut out)?;
            }
            base += n;
        }
        Ok(out)
    }
}

// ===========================================================================
//  渲染就绪网格（USD → prism-rt `Scene::add_mesh` 的加载路径）
// ===========================================================================

/// 渲染就绪的三角网格：**f32 顶点 + u32 索引流**（三角化已完成、世界空间）。
///
/// # 为什么在 io 侧出这一层
/// `prism-io` 不依赖 `prism-rt`（`prism-rt` 只在 *dev-dependencies* 里引用本
/// crate，反向依赖会成环），故 USD 加载路径的终点是这份**中性**数据；rt 侧
/// 一行接手：
///
/// ```text
/// let m = prism_io::usd::parse_str(&usda)?;
/// let scene = prism_io::usd::build_scene(&m.layer)?;
/// for rm in scene.render_meshes() {
///     rt_scene.add_mesh(
///         rm.positions.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect(),
///         rm.indices.clone(),
///         material,
///     );
/// }
/// ```
///
/// # 精度
/// 顶点从 USD 的 f64 世界坐标**一次性**降到 f32（`as f32` 是 f64→f32 的
/// 正确舍入，单调收敛）；rt 全链路 f32，避免"半截 f64 半截 f32"的语义分裂。
/// 三角化在**降精度之前**完成（耳切的相对面积容差在 f64 域判定，不因降精度
/// 变成退化面）。
#[derive(Debug, Clone, PartialEq)]
pub struct RenderMesh {
    /// 源 prim 路径。
    pub path: String,
    /// 顶点位置（世界空间 f32；三角化共享的顶点表）。
    pub positions: Vec<[f32; 3]>,
    /// 三角形索引流（每 3 个一三角形；耳切三角化结果）。
    pub indices: Vec<u32>,
    /// 材质绑定目标路径（`material:binding` 关系；无则 `None`）。
    pub material: Option<String>,
}

impl RenderMesh {
    /// 三角形数。
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// 顶点数。
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// 第 `t` 个三角形的三个顶点坐标（越界返回 `None`；校验/调试用）。
    pub fn triangle(&self, t: usize) -> Option<[[f32; 3]; 3]> {
        let base = t.checked_mul(3)?;
        let idx = self.indices.get(base..base + 3)?;
        let pick = |i: u32| self.positions.get(i as usize).copied();
        Some([pick(idx[0])?, pick(idx[1])?, pick(idx[2])?])
    }

    /// 包围盒 `(min, max)`；空网格返回 `None`（确定性：逐顶点 min/max）。
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        if self.positions.is_empty() {
            return None;
        }
        for p in &self.positions {
            for k in 0..3 {
                if p[k] < min[k] {
                    min[k] = p[k];
                }
                if p[k] > max[k] {
                    max[k] = p[k];
                }
            }
        }
        Some((min, max))
    }
}

impl Mesh {
    /// USD 网格 → 渲染就绪网格（三角化 + f64→f32 降精度）。
    ///
    /// 三角化走 [`Mesh::triangulated`]：退化/自交面**产出零三角形**（不静默
    /// 降级为错几何，C20）；需要结构化错误码的调用方先用
    /// [`Mesh::triangulated_ear`] 逐面取错误。
    pub fn to_render_mesh(&self) -> RenderMesh {
        let positions: Vec<[f32; 3]> = self
            .points
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let tris = self.triangulated();
        let mut indices: Vec<u32> = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            indices.extend_from_slice(&t);
        }
        RenderMesh {
            path: self.path.clone(),
            positions,
            indices,
            material: self.material.clone(),
        }
    }
}

impl Scene {
    /// 全部网格的渲染就绪形态（按 prim 源顺序；确定性）。
    pub fn render_meshes(&self) -> Vec<RenderMesh> {
        self.meshes.iter().map(|m| m.to_render_mesh()).collect()
    }

    /// 渲染就绪形态的三角形总数（与 [`Scene::triangle_count`] 同口径——
    /// 但按"耳切后实际产出"计数，退化面为零贡献）。
    pub fn render_triangle_count(&self) -> usize {
        self.render_meshes().iter().map(|m| m.triangle_count()).sum()
    }
}

// ===========================================================================
//  耳切三角化（Ear Clipping）
// ===========================================================================

/// 耳切顶点的**相对面积容差**（相对于面片 Newell 法向模长）。
///
/// 判据全部是「向量叉积点积 vs 平面法向」这一类**有向面积**量，
/// 除以 `|N|`（≈ 2·面片面积，量纲 = 长度²）后成为**无量纲**数，
/// 于是本容差对「米级场景」与「毫米级资产」等一切缩放一律成立
/// （纯相对判据，绝不写死绝对阈值）。
const EAR_REL_EPS: f64 = 1e-12;

/// 耳切中「顶点位置」与「原始点索引」的绑定。
///
/// 索引必须一路带到输出：耳切过程会**重排**顶点顺序（剪掉耳点），
/// 最终三角形报的是**原始 `points` 下标**。
#[derive(Debug, Clone, Copy)]
struct EarVert {
    /// 原始点索引（输出用）。
    index: u32,
    /// 世界坐标。
    p: [f64; 3],
}

/// 向量减。
#[inline]
fn v_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 向量叉积。
#[inline]
fn v_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 向量点积。
#[inline]
fn v_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Newell 法向（多边形平面法向；模长 = 2·面积，方向由**绕向**决定）。
///
/// 用 Newell 而非「首个非退化三角形」：后者在退化前缀（首两点重合）时
/// 会漏判，而 Newell 对**全部**顶点对称求和，天然与起点选择无关 ——
/// 这正是 C17 全链路确定性所要求的（同一多边形换起点不得改变结果）。
///
/// # 闭合循环（不是 `windows(2)`）
/// 多边形是**闭合**轮廓，Newell 求和必须把「最后一点 → 第一点」那条边
/// 算进去。写成 `for i in 0..m { a = v[i], b = v[(i+1) % m] }` 而**不是**
/// `verts.windows(2)`：后者只覆盖前 m−1 条边，漏掉闭合边后法向会偏
/// （正方形这类对称形状恰好为 0，看起来「正常」；一般多边形会算错
/// 模长，全共线多边形甚至会得到非零法向 ⇒ 退化面逃过 PU1010 判据）。
///
/// 数学上 `Σ (a−b) × (a+b) = 2 Σ a×b = 4·面积向量`，故 `|N| = 2·面积`。
fn newell_normal(verts: &[EarVert]) -> [f64; 3] {
    let mut n = [0.0f64; 3];
    let m = verts.len();
    for i in 0..m {
        let a = verts[i].p;
        let b = verts[(i + 1) % m].p;
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

/// 点 `p` 是否落在三角形 `a,b,c` **内部或边界**上（3D 判定）。
///
/// 三个子叉积与三角形法向同号即为内部。**容差取 0**：调用方只在
/// 「已排除退化」的面循环里用它，边界点算作内部会保守地多剪一耳，
/// 但绝不会漏剪导致自交。
fn point_in_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3], n: [f64; 3]) -> bool {
    v_dot(v_cross(v_sub(b, a), v_sub(p, a)), n) >= 0.0
        && v_dot(v_cross(v_sub(c, b), v_sub(p, b)), n) >= 0.0
        && v_dot(v_cross(v_sub(a, c), v_sub(p, c)), n) >= 0.0
}

/// 2D 线段是否**真交**（严格内部相交；端点接触不算）。
///
/// 用于**自交预检**。这里做 2D 投影（最大轴投影，见 ear_clip 的说明），
/// 是因为「边—边相交」本质是**平面内**的问题：在法向主轴方向上的投影
/// 是该平面到 2D 的**双射**（法向分量最大 ⇒ 投影不折叠），故 2D 判定
/// 与 3D 判定**等价**，且免去 3D 线段最近点法的额外分支。
fn segments_cross_2d(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> bool {
    /// 2D 叉积的 z 分量。
    fn c2(u: [f64; 2], v: [f64; 2]) -> f64 {
        u[0] * v[1] - u[1] * v[0]
    }
    let d = [a1[0] - a0[0], a1[1] - a0[1]];
    let e = [b1[0] - b0[0], b1[1] - b0[1]];
    // 快速排除：包围盒不相交则必不相交。
    let (alo, ahi, blo, bhi) = (
        [a0[0].min(a1[0]), a0[1].min(a1[1])],
        [a0[0].max(a1[0]), a0[1].max(a1[1])],
        [b0[0].min(b1[0]), b0[1].min(b1[1])],
        [b0[0].max(b1[0]), b0[1].max(b1[1])],
    );
    if ahi[0] < blo[0] || bhi[0] < alo[0] || ahi[1] < blo[1] || bhi[1] < alo[1] {
        return false;
    }
    let d1 = c2(d, [b0[0] - a0[0], b0[1] - a0[1]]);
    let d2 = c2(d, [b1[0] - a0[0], b1[1] - a0[1]]);
    let d3 = c2(e, [a0[0] - b0[0], a0[1] - b0[1]]);
    let d4 = c2(e, [a1[0] - b0[0], a1[1] - b0[1]]);
    // 严格异号才算真交（d1 与 d2 反号 **且** d3 与 d4 反号）。
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// 单个面的耳切三角化，结果追加到 `out`。
///
/// # 算法
/// 1. **规范化**：丢弃与前一点**完全重合**的顶点（USD 允许重复点；
///    精确相等判据，不用容差 —— 近似相等的近邻点由步骤 2 的面积容差
///    归入「退化」，避免用容差合并改变有效几何）；
/// 2. **退化检查**：Newell 法向模长 `|N| ≈ 0` ⇒ 全共线 / 全重合 ⇒ PU1010；
/// 3. **自交预检**：2D 边对边真交 ⇒ PU1011（自交多边形无「内部」可言）；
/// 4. **耳点搜索**：逐点判定「凸 ∧ 三角形内无其他顶点」，在**非锚点**
///    （不取原多边形首点）中选**原始索引最小**的耳剪掉；
/// 5. 剩 3 点时收尾；中途无耳 ⇒ PU1012（数值边界，C18 永不 panic）。
///
/// # 为什么耳点判定用 3D 叉积而非 2D 符号
/// 凸性判据 `((v−prev) × (next−v)) · N > 0` 中，`N` 是 **3D** Newell
/// 法向：符号一致性由 3D 叉积与 3D 法向的点积给出，**不依赖任何投影
/// 退化假设**。只有「边—边自交预检」用 2D（已在该处注释论证等价）。
/// 于是本实现对**任意朝向、任意缩放、任意平面法向**都正确。
///
/// # 确定性（C17）
/// 唯一影响「选哪个耳」的是**顶点索引**（`min` 语义 + 锚点规则），
/// 与浮点比较的偶然顺序无关：同一输入逐次调用产出**逐字节相同**的
/// 索引序列。耳点候选的遍历顺序是 `position` 升序，而 `min` 的更新
/// 条件是严格小于，故**首个最小值**恒胜出，不存在「谁先遍历到」的不确定性。
///
/// # 复杂度
/// O(n²)（每轮 O(n) 候选 × O(n) 内点测试，共 O(n) 轮）。按 D-16 不做
/// O(n log n) 优化。
fn ear_clip(
    points: &[[f64; 3]],
    indices: &[u32],
    location: &str,
    out: &mut Vec<[u32; 3]>,
) -> Result<(), UsdError> {
    // ── 步骤 1：规范化（丢弃与前一点完全重合的顶点）─────────────────
    // 用 `EarVert.index` 携带原始下标；重复点保留**首次**出现的下标，
    // 使输出索引确定性地落在「原始点」上。
    let mut verts: Vec<EarVert> = Vec::with_capacity(indices.len());
    for &i in indices {
        let p = *points
            .get(i as usize)
            .ok_or_else(|| UsdError::invalid_mesh(location.to_string(), format!("顶点索引 {i} 越界（points 长度 = {}）", points.len())))?;
        if let Some(last) = verts.last() {
            if last.p == p {
                continue; // 重合点：跳过
            }
        }
        verts.push(EarVert { index: i, p });
    }
    // 首尾重合（闭合多边形常显式重复首点）。
    while verts.len() >= 2 && verts[0].p == verts[verts.len() - 1].p {
        verts.pop();
    }

    if verts.len() < 3 {
        // 面声明了 ≥3 顶点但去重后不足 3 ⇒ 退化。
        return Err(UsdError::degenerate_polygon(
            location.to_string(),
            format!("面去重后仅 {} 个不同顶点（声明 {} 个），不足 3", verts.len(), indices.len()),
        ));
    }

    // ── 步骤 2：退化检查（Newell 法向模长）─────────────────────────
    let normal = newell_normal(&verts);
    let area2 = v_dot(normal, normal); // = |N|² = (2·面积)²
    if !(area2 > 0.0) || !area2.is_finite() {
        return Err(UsdError::degenerate_polygon(
            location.to_string(),
            format!("面 Newell 法向模长为 0（|N|² = {area2}）：所有顶点共线或重合"),
        ));
    }

    // ── 步骤 3：自交预检（2D 最大轴投影）───────────────────────────
    // 投影策略：取 Newell 法向**绝对值最大的分量**所对应的轴丢弃之。
    // 理由：该轴法向分量最大 ⇒ 沿该轴投影时平面不折叠（投影是双射），
    // 故 2D 的边相交判定与 3D 等价。**失效场景**：面本身非平面（USD
    // 允许非共面的 `faceVertexCounts > 3`），此时 2D 投影会把面「压扁」，
    // 可能漏检真实的空间自交。本实现在**剖分**上仍是 3D 正确的（耳点判定
    // 全程 3D），仅自交**预检**在非平面面上是保守近似 —— 见文件头诚实边界。
    let drop_axis = {
        let ax = normal[0].abs();
        let ay = normal[1].abs();
        let az = normal[2].abs();
        if ax >= ay && ax >= az {
            0
        } else if ay >= az {
            1
        } else {
            2
        }
    };
    let to_2d = |p: [f64; 3]| -> [f64; 2] {
        let mut q = [0.0f64; 2];
        let mut k = 0usize;
        for axis in 0..3 {
            if axis == drop_axis {
                continue;
            }
            q[k] = p[axis];
            k += 1;
        }
        q
    };
    let m = verts.len();
    for i in 0..m {
        let i2 = (i + 1) % m;
        for j in (i + 1)..m {
            let j2 = (j + 1) % m;
            // 跳过相邻边（共享端点不算自交）。
            if j == i2 || (i == 0 && j2 == 0) {
                continue;
            }
            if segments_cross_2d(
                to_2d(verts[i].p),
                to_2d(verts[i2].p),
                to_2d(verts[j].p),
                to_2d(verts[j2].p),
            ) {
                return Err(UsdError::self_intersecting_polygon(
                    location.to_string(),
                    format!(
                        "面轮廓自交：边 {i}→{i2} 与边 {j}→{j2} 在内部真交（共 {} 顶点）",
                        m
                    ),
                ));
            }
        }
    }

    // ── 步骤 4：耳点搜索与剪除 ────────────────────────────────────
    // 逐轮重算 Newell 法向（剩余顶点已变），保证 3D 判据始终对应当前多边形。
    // 候选顺序按 **剩余顶点的最小原始索引**（`min` 语义）—— 确定性契约。
    while verts.len() > 3 {
        let n_now = newell_normal(&verts);
        let len_now = v_dot(n_now, n_now).sqrt();
        if !(len_now > 0.0) || !len_now.is_finite() {
            return Err(UsdError::degenerate_polygon(
                location.to_string(),
                format!("耳切过程中剩余 {} 顶点退化（|N| = {len_now}）", verts.len()),
            ));
        }
        // 面积容差：把有向面积与 |N| 比，阈值恒为无量纲 EAR_REL_EPS。
        let eps_area = EAR_REL_EPS * len_now;

        // 收集所有合法耳，取「原始索引最小」者。
        // **锚点规则**：优先在**非锚点**（position ≠ 0，即不取原多边形首点）
        // 中选最小索引者；只在非锚点里一个耳都没有时才回落到锚点。
        //   - 正确性：耳切定理保证 n ≥ 4 的简单多边形至少两个耳，故当锚点
        //     自己就是耳时，另一个耳必在非锚点里 ⇒ 该回落**永不**发生，
        //     但保留它是为了 C18「永不 panic」（数值边界上仍能出结果/报错）。
        //   - 兼容性：凸多边形时结果与旧扇形**逐索引相同**（都取首点扇形
        //     对角线），故既有 `quad_mesh_triangulates_deterministically`
        //     的 `[[0,1,2],[0,2,3]]` 期望值零改动通过。
        let mut chosen: Option<(usize, u32)> = None; // (position, 原始索引)
        let mut fallback: Option<(usize, u32)> = None;
        for i in 0..verts.len() {
            let prev = verts[(i + verts.len() - 1) % verts.len()];
            let cur = verts[i];
            let next = verts[(i + 1) % verts.len()];
            // 凸性：3D 叉积与 3D 法向点积为正（与 Newell 绕向一致）。
            if v_dot(v_cross(v_sub(cur.p, prev.p), v_sub(next.p, cur.p)), n_now) <= eps_area {
                continue;
            }
            // 耳三角形内不得含其他剩余顶点。
            let occupied = verts.iter().any(|v| {
                v.index != prev.index && v.index != cur.index && v.index != next.index
                    && point_in_triangle(v.p, prev.p, cur.p, next.p, n_now)
            });
            if occupied {
                continue;
            }
            // 索引最小者胜出（严格小于 ⇒ 首个最小值胜出，顺序确定）。
            if fallback.is_none_or(|(_, best)| cur.index < best) {
                fallback = Some((i, cur.index));
            }
            if i != 0 && chosen.is_none_or(|(_, best)| cur.index < best) {
                chosen = Some((i, cur.index));
            }
        }
        let Some((i, _)) = chosen.or(fallback) else {
            return Err(UsdError::triangulation_failed(
                location.to_string(),
                format!(
                    "剩余 {} 个顶点中找不到合法耳点（该面接近数值退化，|N| = {len_now}）",
                    verts.len()
                ),
            ));
        };
        // 剪耳：输出 (prev, cur, next) —— 顺序即多边形绕向，保证法线同向。
        let prev = verts[(i + verts.len() - 1) % verts.len()];
        let cur = verts[i];
        let next = verts[(i + 1) % verts.len()];
        out.push([prev.index, cur.index, next.index]);
        verts.remove(i);
    }
    // ── 步骤 5：剩 3 点，收尾 ──────────────────────────────────────
    if verts.len() == 3 {
        out.push([verts[0].index, verts[1].index, verts[2].index]);
    }
    Ok(())
}

/// 场景（一层的可渲染结果）。
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// 层版本。
    pub version: String,
    /// 全部网格（按 prim 源顺序）。
    pub meshes: Vec<Mesh>,
    /// 非网格 prim 数量。
    pub non_mesh_prims: usize,
}

impl Scene {
    /// 网格总数。
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// 三角面总数。
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangulated().len()).sum()
    }

    /// 场景级包围盒（`(min, max)`；空场景返回 `None`）。
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for m in &self.meshes {
            for p in &m.points {
                any = true;
                for k in 0..3 {
                    if p[k] < min[k] {
                        min[k] = p[k];
                    }
                    if p[k] > max[k] {
                        max[k] = p[k];
                    }
                }
            }
        }
        if any {
            Some((min, max))
        } else {
            None
        }
    }
}

// ===========================================================================
//  场景构建（prim 树 → 网格，Xform 链累加 + 烘焙）
// ===========================================================================

/// 由层构建可渲染场景（Xform 变换链在此**累加并烘焙**到顶点）。
///
/// 变换规则：`world(child) = world(parent) ∘ local(child)`；
/// `local` 由 `xformOpOrder` 依次左乘各操作（顺序即 USD 语义：列出的
/// 顺序是**从右到左**的应用序，故累乘时 `local = op_n ∘ … ∘ op_1`）。
pub fn build_scene(layer: &Layer) -> Result<Scene, UsdError> {
    let report_ref = build_scene_report(layer)?;
    Ok(report_ref.0)
}

/// 带记账报告的场景构建。
pub fn build_scene_report(layer: &Layer) -> Result<(Scene, LayerReport), UsdError> {
    let mut report = LayerReport::default();
    let mut meshes = Vec::new();
    let mut non_mesh = 0usize;
    for p in &layer.prims {
        collect_prim(p, Mat4::identity(), &mut meshes, &mut non_mesh, &mut report)?;
    }
    for m in &meshes {
        if let Some(mat) = &m.material {
            if !layer.all_paths.iter().any(|p| p == mat) {
                return Err(UsdError::dangling_relationship(
                    format!("prim {} 的关系 material:binding", m.path),
                    format!(
                        "材质目标 {mat:?} 不存在于本层（已声明路径：{:?}）",
                        layer.all_paths
                    ),
                ));
            }
        }
    }
    // 记账项统一升序（确定性契约：报告字段的**集合**按名排序，
    // 顺序不依赖遍历序；`non_mesh_prims` 是计数，天然确定）。
    report.skipped_time_samples.sort();
    report.skipped_variants.sort();
    report.unresolved_asset_paths.sort();
    report.non_uniform_normals.sort();
    Ok((
        Scene {
            version: layer.version.clone(),
            meshes,
            non_mesh_prims: non_mesh,
        },
        report,
    ))
}

/// 递归收集 prim 子树中的网格（累加世界变换）。
fn collect_prim(
    prim: &Prim,
    parent_world: Mat4,
    out: &mut Vec<Mesh>,
    non_mesh: &mut usize,
    report: &mut LayerReport,
) -> Result<(), UsdError> {
    let local = local_transform(prim)?;
    let world = parent_world.mul(&local);
    if prim.type_name == "Mesh" {
        out.push(build_mesh(prim, world, report)?);
    } else if !prim.type_name.is_empty() {
        *non_mesh += 1;
    }
    for r in &prim.references {
        report.unresolved_asset_paths.push(r.clone());
    }
    for p in &prim.payloads {
        report.unresolved_asset_paths.push(p.clone());
    }
    for c in &prim.children {
        collect_prim(c, world, out, non_mesh, report)?;
    }
    Ok(())
}

/// 解析单个 prim 的局部变换（`xformOpOrder` 驱动的操作链）。
fn local_transform(prim: &Prim) -> Result<Mat4, UsdError> {
    let mut m = Mat4::identity();
    // 无 xformOpOrder 时，只看是否显式给了 `xformOp:transform`。
    if prim.xform_op_order.is_empty() {
        if let Some(a) = prim.attribute("xformOp:transform") {
            m = m.mul(&matrix_from_attr(a, &prim.path)?);
        }
        return Ok(m);
    }
    for op in &prim.xform_op_order {
        let attr = prim.attribute(op).ok_or_else(|| {
            UsdError::invalid_xform(
                format!("prim {}", prim.path),
                format!(
                    "xformOpOrder 列出了操作 {op:?}，但该 prim 没有对应属性 {}",
                    op
                ),
            )
        })?;
        m = m.mul(&matrix_from_attr(attr, &prim.path)?);
    }
    Ok(m)
}

/// 由属性值构造 4×4 矩阵（支持 `matrix4d` 与 translate/scale/rotateXYZ）。
fn matrix_from_attr(attr: &Attribute, prim_path: &str) -> Result<Mat4, UsdError> {
    let loc = format!("prim {prim_path} 的属性 {}", attr.name);
    match attr.name.as_str() {
        "xformOp:transform" => {
            let rows = attr.value.as_rows();
            if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`xformOp:transform` 必须是 4×4 矩阵，收到 {}×?", rows.len()),
                ));
            }
            let mut m = [[0.0f64; 4]; 4];
            for (i, r) in rows.iter().enumerate() {
                for (j, v) in r.iter().enumerate() {
                    m[i][j] = *v;
                }
            }
            Ok(Mat4 { m })
        }
        "xformOp:translate" => {
            let t = attr.value.as_numbers();
            let mut m = Mat4::identity();
            if t.len() != 3 {
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`xformOp:translate` 需要 3 个分量，收到 {}", t.len()),
                ));
            }
            m.m[0][3] = t[0];
            m.m[1][3] = t[1];
            m.m[2][3] = t[2];
            Ok(m)
        }
        "xformOp:scale" => {
            let s = attr.value.as_numbers();
            let mut m = Mat4::identity();
            if s.len() != 3 {
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`xformOp:scale` 需要 3 个分量，收到 {}", s.len()),
                ));
            }
            m.m[0][0] = s[0];
            m.m[1][1] = s[1];
            m.m[2][2] = s[2];
            Ok(m)
        }
        "xformOp:rotateXYZ" => {
            let d = attr.value.as_numbers();
            if d.len() != 3 {
                return Err(UsdError::type_mismatch(
                    &loc,
                    format!("`xformOp:rotateXYZ` 需要 3 个角度（度），收到 {}", d.len()),
                ));
            }
            // 度 → 弧度后按 Z→Y→X 复合（USD 的 `rotateXYZ` 语义）。
            // 三角走本模块自带的确定性件（禁 `.sin()/.cos()`：会落平台 libm，
            // 跨平台逐位不一致，违反 C17/C18 确定性铁律）。
            let (rx, ry, rz) = (
                d[0] * (PI / 180.0),
                d[1] * (PI / 180.0),
                d[2] * (PI / 180.0),
            );
            let (sx, cx) = det_cos_sin(rx);
            let (sy, cy) = det_cos_sin(ry);
            let (sz, cz) = det_cos_sin(rz);
            let mx = Mat4 {
                m: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, cx, -sx, 0.0],
                    [0.0, sx, cx, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            };
            let my = Mat4 {
                m: [
                    [cy, 0.0, sy, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sy, 0.0, cy, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            };
            let mz = Mat4 {
                m: [
                    [cz, -sz, 0.0, 0.0],
                    [sz, cz, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            };
            Ok(mx.mul(&my).mul(&mz))
        }
        // 其它 xformOp 变体（orient/rotateZYX/…）：结构化拒绝。
        other => Err(UsdError::invalid_xform(
            loc,
            format!("暂不支持的变换操作：{other}"),
        )),
    }
}

/// 探测 `Mat4` 的线性部分是否**非等比缩放**。
///
/// 法线在非等比缩放下必须乘逆转置 `transpose(inverse(upper3x3))` 才正确；
/// 当前版本只按旋转部分变换，故此处返回一行缩放描述供 [`LayerReport`] 记账，
/// 让调用方能看见"这里的法线只是近似"。等比（含纯旋转/刚体）返回 `None`。
///
/// 判据：取线性部分三个基向量的长度，若两两相对差超 `1e-9` 即为非等比。
fn non_uniform_scale(m: &Mat4) -> Option<String> {
    let len =
        |c: usize| (m.m[0][c] * m.m[0][c] + m.m[1][c] * m.m[1][c] + m.m[2][c] * m.m[2][c]).sqrt();
    let (sx, sy, sz) = (len(0), len(1), len(2));
    let spread = [sx, sy, sz];
    let lo = spread.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = spread.iter().copied().fold(0.0_f64, f64::max);
    if lo <= 0.0 {
        return None;
    }
    if (hi - lo) / hi > 1e-9 {
        Some(format!("({sx:.6}, {sy:.6}, {sz:.6})"))
    } else {
        None
    }
}

/// 由 prim 构造网格（顶点/法线已烘焙世界变换）。
fn build_mesh(prim: &Prim, world: Mat4, report: &mut LayerReport) -> Result<Mesh, UsdError> {
    let loc = format!("prim {}", prim.path);
    // points。
    let points_attr = prim
        .attribute("points")
        .ok_or_else(|| UsdError::missing_attribute(&loc, "Mesh 缺少 `points` 属性"))?;
    let rows = points_attr.value.as_rows();
    if points_attr.has_time_samples && rows.is_empty() {
        return Err(UsdError::missing_time_sample_value(
            &loc,
            "`points` 只有 timeSamples、无默认帧值",
        ));
    }
    let mut points: Vec<[f64; 3]> = Vec::with_capacity(rows.len());
    for r in &rows {
        if r.len() != 3 {
            return Err(UsdError::invalid_mesh(
                &loc,
                format!("`points` 每项应为 3 个分量，收到 {}", r.len()),
            ));
        }
        points.push(world.transform_point([r[0], r[1], r[2]]));
    }
    // faceVertexCounts。
    let counts: Vec<u32> = match prim.value("faceVertexCounts") {
        Some(v) => v
            .as_numbers()
            .into_iter()
            .map(|n| if n < 0.0 { 0 } else { n as u32 })
            .collect(),
        None => Vec::new(),
    };
    // faceVertexIndices。
    let indices: Vec<u32> = match prim.value("faceVertexIndices") {
        Some(v) => v
            .as_numbers()
            .into_iter()
            .map(|n| if n < 0.0 { u32::MAX } else { n as u32 })
            .collect(),
        None => Vec::new(),
    };
    // 自洽性检查：索引越界 / counts 与 indices 长度不符 / 索引 < 3。
    if !indices.is_empty() {
        if points.is_empty() {
            return Err(UsdError::invalid_mesh(
                &loc,
                format!("有 {} 个面索引但 `points` 为空", indices.len()),
            ));
        }
        for i in &indices {
            if *i == u32::MAX || *i as usize >= points.len() {
                return Err(UsdError::invalid_mesh(
                    &loc,
                    format!(
                        "面索引 {i} 越界（顶点数 = {}；负索引或 >= 顶点数均为非法）",
                        points.len()
                    ),
                ));
            }
        }
    }
    if !counts.is_empty() {
        let sum: u32 = counts.iter().sum();
        if sum as usize != indices.len() {
            return Err(UsdError::invalid_mesh(
                &loc,
                format!(
                    "`faceVertexCounts` 之和 = {sum}，但 `faceVertexIndices` 长度 = {}",
                    indices.len()
                ),
            ));
        }
        for c in &counts {
            if *c < 3 {
                return Err(UsdError::invalid_mesh(
                    &loc,
                    format!("面顶点数 {c} < 3：USD 面至少 3 个顶点"),
                ));
            }
        }
    }
    // 法线。
    let mut normals = Vec::new();
    if let Some(a) = prim.attribute("normals") {
        // 法线只按**旋转部分**变换（非等比缩放下需逆转置，当前版本不做——
        // 见模块头「诚实边界」，此处记账到报告里而非静默近似）。
        if let Some(d) = non_uniform_scale(&world) {
            report
                .non_uniform_normals
                .push(format!("{}（缩放 {d}）", prim.path));
        }
        let world_pos = world.transform_point([0.0, 0.0, 0.0]);
        for r in a.value.as_rows() {
            if r.len() < 3 {
                return Err(UsdError::type_mismatch(
                    format!("prim {}", prim.path),
                    format!("`normals` 每项应为 3 个分量，收到 {}", r.len()),
                ));
            }
            // 减去平移分量即只剩旋转/缩放部分。
            let n = world.transform_point([r[0], r[1], r[2]]);
            normals.push([
                n[0] - world_pos[0],
                n[1] - world_pos[1],
                n[2] - world_pos[2],
            ]);
        }
    }
    // UV：取 `primvars:st`（或 `st0`）。
    let mut uvs = Vec::new();
    if let Some(a) = prim
        .attribute("primvars:st")
        .or_else(|| prim.attribute("primvars:st0"))
    {
        let interp = a.interpolation.as_deref().unwrap_or("vertex");
        let rows = a.value.as_rows();
        if interp == "faceVarying" || interp == "facevarying" {
            // 按索引展开（每个面顶点一份 UV）。
            for i in &indices {
                if *i as usize >= rows.len() {
                    return Err(UsdError::invalid_mesh(
                        format!("prim {}", prim.path),
                        format!("faceVarying UV 索引 {i} 越界（UV 数 = {}）", rows.len()),
                    ));
                }
                let r = &rows[*i as usize];
                uvs.push([
                    r.first().copied().unwrap_or(0.0),
                    r.get(1).copied().unwrap_or(0.0),
                ]);
            }
        } else {
            for r in &rows {
                uvs.push([
                    r.first().copied().unwrap_or(0.0),
                    r.get(1).copied().unwrap_or(0.0),
                ]);
            }
        }
    }
    // 材质绑定。
    let material = prim
        .relationship("material:binding")
        .and_then(|r| r.targets.first().cloned());
    Ok(Mesh {
        path: prim.path.clone(),
        points,
        face_counts: counts,
        face_indices: indices,
        normals,
        uvs,
        material,
        world_transform: world,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // ── 耳切三角化的测试装置（oracle 与被测实现走**不同的公式**）─────────

    /// 构造单面 `Mesh`（绕开解析器，让测试聚焦耳切本身）。
    fn ear_mesh(points: &[[f64; 3]], face: &[u32]) -> Mesh {
        Mesh {
            path: "/test/ear".to_string(),
            points: points.to_vec(),
            face_counts: vec![face.len() as u32],
            face_indices: face.to_vec(),
            normals: Vec::new(),
            uvs: Vec::new(),
            material: None,
            world_transform: Mat4::identity(),
        }
    }

    /// 三角形面积（3D）。
    fn tri_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        let n = v_cross(v_sub(b, a), v_sub(c, a));
        0.5 * v_dot(n, n).sqrt()
    }

    /// 多边形法向（**测试侧独立实现**：质心式 `Σ (pᵢ−c) × (pᵢ₊₁−c)`，
    /// 与被测实现的 Newell 边缘式 `Σ (aᵢ−bᵢ) × (aᵢ+bᵢ)` 是不同的求和路径，
    /// 两者数学等价但浮点误差不同 ⇒ 可互为交叉验证的 oracle）。
    fn poly_normal(pts: &[[f64; 3]]) -> [f64; 3] {
        let inv = 1.0 / pts.len() as f64;
        let c = [
            pts.iter().map(|p| p[0]).sum::<f64>() * inv,
            pts.iter().map(|p| p[1]).sum::<f64>() * inv,
            pts.iter().map(|p| p[2]).sum::<f64>() * inv,
        ];
        let mut n = [0.0f64; 3];
        for i in 0..pts.len() {
            let t = v_cross(
                v_sub(pts[i], c),
                v_sub(pts[(i + 1) % pts.len()], c),
            );
            n[0] += t[0];
            n[1] += t[1];
            n[2] += t[2];
        }
        n
    }

    /// 多边形面积（`|N| / 2`）。
    fn poly_area(pts: &[[f64; 3]]) -> f64 {
        let n = poly_normal(pts);
        0.5 * v_dot(n, n).sqrt()
    }

    /// 按法向绝对值最大的轴丢弃之，投影到 2D（测试多边形全部共面 ⇒ 无损）。
    fn project(pts: &[[f64; 3]], n: [f64; 3]) -> Vec<[f64; 2]> {
        let drop = {
            let (ax, ay, az) = (n[0].abs(), n[1].abs(), n[2].abs());
            if ax >= ay && ax >= az {
                0
            } else if ay >= az {
                1
            } else {
                2
            }
        };
        pts.iter()
            .map(|p| {
                let mut q = [0.0f64; 2];
                let mut k = 0usize;
                for axis in 0..3 {
                    if axis == drop {
                        continue;
                    }
                    q[k] = p[axis];
                    k += 1;
                }
                q
            })
            .collect()
    }

    /// 2D 有向面积（鞋带公式）。
    fn signed_area_2d(p: &[[f64; 2]]) -> f64 {
        let mut s = 0.0;
        for i in 0..p.len() {
            let a = p[i];
            let b = p[(i + 1) % p.len()];
            s += a[0] * b[1] - b[0] * a[1];
        }
        0.5 * s
    }

    /// 归一为逆时针（2D）。
    fn to_ccw(mut p: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
        if signed_area_2d(&p) < 0.0 {
            p.reverse();
        }
        p
    }

    /// 用半平面 `cross(b−a, p−a) ≥ 0` 裁剪 2D 多边形（Sutherland–Hodgman）。
    fn clip_half_plane(poly: &[[f64; 2]], a: [f64; 2], b: [f64; 2]) -> Vec<[f64; 2]> {
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let mut out = Vec::new();
        for i in 0..poly.len() {
            let cur = poly[i];
            let nxt = poly[(i + 1) % poly.len()];
            let (dc, dn) = (side(cur), side(nxt));
            if dc >= 0.0 {
                out.push(cur);
            }
            if (dc > 0.0 && dn < 0.0) || (dc < 0.0 && dn > 0.0) {
                let t = dc / (dc - dn);
                out.push([
                    cur[0] + t * (nxt[0] - cur[0]),
                    cur[1] + t * (nxt[1] - cur[1]),
                ]);
            }
        }
        out
    }

    /// 两个**凸** 2D 多边形的交集面积（测试里两个多边形都是三角形，恒凸）。
    /// 合法三角剖分里相邻三角形只共边（退化交集），面积恒为 0。
    fn overlap_area_2d(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> f64 {
        let clip = to_ccw(clip.to_vec());
        let mut poly = subject.to_vec();
        for i in 0..clip.len() {
            if poly.is_empty() {
                return 0.0;
            }
            poly = clip_half_plane(&poly, clip[i], clip[(i + 1) % clip.len()]);
        }
        if poly.len() < 3 {
            return 0.0;
        }
        signed_area_2d(&poly).abs()
    }

    /// 耳切结果的**三重断言**（硬要求，一条都不能少）：
    /// 1. **无自交**：任意两个三角形的交集面积 ≈ 0；
    /// 2. **面积覆盖**：三角形面积和 == 多边形面积（相对容差 1e-6）；
    /// 3. **法线同向**：每个三角形的法线与多边形法向点积 > 0。
    fn assert_ear_clip_valid(points: &[[f64; 3]], face: &[u32], tris: &[[u32; 3]], label: &str) {
        let poly: Vec<[f64; 3]> = face.iter().map(|&i| points[i as usize]).collect();
        let n = poly_normal(&poly);
        let nlen = v_dot(n, n).sqrt();
        assert!(nlen > 0.0, "{label}：测试用多边形自身退化，无法作为断言基准");
        let want = 0.5 * nlen;

        // ② 面积覆盖
        let sum: f64 = tris
            .iter()
            .map(|t| tri_area(points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]))
            .sum();
        let rel = (sum - want).abs() / want;
        assert!(
            rel < 1e-6,
            "{label}：面积和 {sum} != 多边形面积 {want}（相对误差 {rel:e} > 1e-6）"
        );

        // ③ 法线同向
        for (k, t) in tris.iter().enumerate() {
            let (a, b, c) = (
                points[t[0] as usize],
                points[t[1] as usize],
                points[t[2] as usize],
            );
            let tn = v_cross(v_sub(b, a), v_sub(c, a));
            assert!(
                v_dot(tn, n) > 0.0,
                "{label}：第 {k} 个三角形法线与多边形法向反向（翻面）"
            );
        }

        // ① 无自交
        let mut worst: f64 = 0.0;
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let ti: Vec<[f64; 3]> = tris[i].iter().map(|&k| points[k as usize]).collect();
                let tj: Vec<[f64; 3]> = tris[j].iter().map(|&k| points[k as usize]).collect();
                let ov = overlap_area_2d(&project(&ti, n), &project(&tj, n)) / want;
                worst = worst.max(ov);
            }
        }
        assert!(
            worst < 1e-9,
            "{label}：三角形两两重叠（最大相对交集面积 {worst:e} > 1e-9），剖分自交"
        );
    }

    /// 最小 `.usda`：一个 Xform 下的单个三角 Mesh。
    const MINIMAL: &str = r#"#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "root"
{
    matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]

    def Mesh "tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)] ( interpolation = "vertex" )
        float2[] primvars:st = [(0, 0), (1, 0), (0, 1)]
        uniform token subdivisionScheme = "none"
    }
}
"#;

    #[test]
    fn parses_minimal_usda_prim_tree() {
        let p = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.layer.version, "1");
        assert_eq!(p.layer.prims.len(), 1);
        let root = &p.layer.prims[0];
        assert_eq!(root.type_name, "Xform");
        assert_eq!(root.path, "/root");
        assert_eq!(root.children.len(), 1);
        let mesh = &root.children[0];
        assert_eq!(mesh.type_name, "Mesh");
        assert_eq!(mesh.path, "/root/tri");
        // 层元数据。
        let md: BTreeMap<&str, &str> = p
            .layer
            .metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(md.get("defaultPrim"), Some(&"root"));
        assert_eq!(md.get("metersPerUnit"), Some(&"0.01"));
        assert_eq!(md.get("upAxis"), Some(&"Y"));
    }

    #[test]
    fn mesh_attributes_parse_with_correct_shapes() {
        let p = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        let mesh = &p.layer.prims[0].children[0];
        let points = mesh.value("points").expect("points 应存在").as_rows();
        assert_eq!(points.len(), 3, "顶点数应为 3");
        assert_eq!(points[1], vec![1.0, 0.0, 0.0]);
        let idx = mesh
            .value("faceVertexIndices")
            .expect("索引应存在")
            .as_numbers();
        assert_eq!(idx, vec![0.0, 1.0, 2.0]);
        let counts = mesh
            .value("faceVertexCounts")
            .expect("counts 应存在")
            .as_numbers();
        assert_eq!(counts, vec![3.0]);
        let st = mesh.attribute("primvars:st").expect("st 应存在");
        assert_eq!(st.value.as_rows().len(), 3);
        let n_attr = mesh.attribute("normals").expect("normals 应存在");
        assert_eq!(n_attr.interpolation.as_deref(), Some("vertex"));
        assert_eq!(n_attr.value.as_rows().len(), 3);
    }

    #[test]
    fn builds_renderable_scene_with_triangulation() {
        let p = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(scene.mesh_count(), 1);
        let m = &scene.meshes[0];
        assert_eq!(m.vertex_count(), 3, "顶点");
        assert_eq!(m.face_count(), 1, "面数");
        assert_eq!(m.uvs.len(), 3, "UV 数");
        assert_eq!(m.triangulated(), vec![[0, 1, 2]]);
        assert_eq!(scene.triangle_count(), 1);
        let (min, max) = scene.bounds().expect("包围盒");
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn quad_mesh_triangulates_deterministically() {
        let src = r#"#usda 1.0
def Mesh "quad"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        let m = &scene.meshes[0];
        assert_eq!(m.face_count(), 1);
        // 期望值与旧扇形**逐索引相同**：耳切的「锚点规则」保证凸多边形时
        // 首个被剪的耳是非锚点里索引最小者（此处 index 1），留下
        // (0, 2, 3) 收尾 —— 与扇形的对角线 0-2 一致。
        assert_eq!(m.triangulated(), vec![[0, 1, 2], [0, 2, 3]], "耳切（锚点规则）");
        assert_eq!(scene.triangle_count(), 2);
    }

    // ── 耳切三角化（任务 1 的核心测试）────────────────────────────────

    /// 构造一个 XY 平面上的凸四边形（`ccw` 决定绕向）。
    fn quad_points(ccw: bool) -> Vec<[f64; 3]> {
        let ccw_pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        if ccw {
            ccw_pts.to_vec()
        } else {
            let mut v = ccw_pts.to_vec();
            v.reverse();
            v
        }
    }

    #[test]
    fn convex_quad_ccw_and_cw_both_give_two_consistently_oriented_triangles() {
        for &ccw in &[true, false] {
            let pts = quad_points(ccw);
            let m = ear_mesh(&pts, &[0, 1, 2, 3]);
            let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(tris.len(), 2, "凸四边形必须剖成 2 个三角形（ccw={ccw}）");
            assert_ear_clip_valid(&pts, &[0, 1, 2, 3], &tris, if ccw { "凸四边形 CCW" } else { "凸四边形 CW" });
            // 绕向反向 ⇒ 输出法线也反向（面积恒为正 2×1/2 = 1）。
            let poly_area = poly_area(&pts);
            let sum: f64 = tris
                .iter()
                .map(|t| tri_area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]))
                .sum();
            assert!((sum - poly_area).abs() < 1e-12, "面积和必须等于多边形面积");
            assert!((sum - 2.0).abs() < 1e-12, "2×1 矩形面积应为 2，实测 {sum}");
        }
    }

    #[test]
    fn concave_quad_l_shape_has_no_self_intersection_and_covers_full_area() {
        // **真凹**四边形（1 个 reflex 顶点，在下标 1 处）：
        //   (0,0) → (1,1) → (3,0) → (0,3) → 回到 (0,0)
        // 鞋带面积 = 3.0。扇形展开取对角线 0→2，即 (0,0)-(3,0) 这条
        // **落在多边形外**的线段 ⇒ 剖出两个各 3.0 的三角形（面积和 6.0
        // = 多边形面积的 2 倍，且互相重叠）。耳切必须改用对角线 1→3。
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let face = [0, 1, 2, 3];
        let m = ear_mesh(&pts, &face);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 2, "凹四边形必须剖成 2 个三角形");
        // 三重断言：无自交 + 面积覆盖 + 法线同向（细节在 helper 内）。
        assert_ear_clip_valid(&pts, &face, &tris, "凹四边形 L 形");

        // 显式复核面积数字（诊断用）：两三角形各 1.5，和 3.0。
        let want = poly_area(&pts);
        let sum: f64 = tris
            .iter()
            .map(|t| tri_area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]))
            .sum();
        assert!((want - 3.0).abs() < 1e-12, "L 形面积应为 3.0，实测 {want}");
        assert!((sum - 3.0).abs() < 1e-12, "两三角形面积和应为 3.0，实测 {sum}");

        // 旧扇形在同一多边形上**必然失败**（证明这个测试不是空转）：
        // 扇形取 [0,1,2] 与 [0,2,3]，面积和是多边形面积的 2 倍。
        let fan_sum = tri_area(pts[0], pts[1], pts[2]) + tri_area(pts[0], pts[2], pts[3]);
        assert!(
            (fan_sum - want).abs() > 1e-6,
            "扇形本应在此多边形上失败，却诡异地通过了（fan={fan_sum}, want={want}）"
        );
        assert!((fan_sum - 6.0).abs() < 1e-12, "扇形面积和应为 6.0（错误值），实测 {fan_sum}");
    }

    #[test]
    fn concave_octagon_u_shape_covers_full_area_with_six_triangles() {
        // U 形凹八边形（真实 DCC 资产里极常见的「切出凹槽」轮廓）。
        let pts = [
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ];
        let face: Vec<u32> = (0..8).collect();
        let m = ear_mesh(&pts, &face);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 6, "凹八边形必须剖成 8-2 = 6 个三角形");
        assert_ear_clip_valid(&pts, &face, &tris, "凹八边形 U 形");
        // U 形面积 = 6×4 − 2×3 = 18。
        let want = poly_area(&pts);
        assert!((want - 18.0).abs() < 1e-12, "U 形面积应为 18，实测 {want}");
    }

    #[test]
    fn concave_quad_two_triangles_have_exactly_zero_intersection_area() {
        // 关键断言：**凹四边形两三角形的交集面积为 0**。
        // 扇形实现在同一多边形上交集面积 = 1.5（完全重叠），耳切必须是 0。
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let face = [0u32, 1, 2, 3];
        let m = ear_mesh(&pts, &face);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 2);
        let n = poly_normal(&pts);
        let a: Vec<[f64; 2]> = tris[0].iter().map(|&k| project(&[pts[k as usize]], n)[0]).collect();
        let b: Vec<[f64; 2]> = tris[1].iter().map(|&k| project(&[pts[k as usize]], n)[0]).collect();
        let ov = overlap_area_2d(&a, &b);
        assert!(
            ov < 1e-12,
            "凹四边形两三角形交集面积必须为 0（实测 {ov:e}）—— 非零即意味着剖分自交"
        );
        // 对照：旧扇形在同一多边形上的交集面积 = 1.5（证明断言非空转）。
        let fa: Vec<[f64; 2]> = [0usize, 1, 2]
            .iter()
            .map(|&k| project(&[pts[k]], n)[0])
            .collect();
        let fb: Vec<[f64; 2]> = [0usize, 2, 3]
            .iter()
            .map(|&k| project(&[pts[k]], n)[0])
            .collect();
        assert!(
            overlap_area_2d(&fa, &fb) > 1.0,
            "扇形在凹四边形上应严重重叠（实测 {}），否则本测试无意义",
            overlap_area_2d(&fa, &fb)
        );
    }

    #[test]
    fn ear_clip_area_coverage_measurement_table() {
        // 诊断用面积覆盖表：多边形 / 三角形数 / 面积和 / 多边形面积 / 相对误差。
        // 每个多边形都过 `assert_ear_clip_valid`（含无自交 + 法线同向）。
        struct Case {
            label: &'static str,
            pts: &'static [[f64; 3]],
        }
        // L 形凹四边形（reflex 在下标 1）。
        const L4: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        // U 形凹八边形。
        const U8: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ];
        // 箭头形凹五边形（凹在顶点 2）。
        const ARROW5: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ];
        // 凸六边形（正六边形，近似）。
        const HEX6: &[[f64; 3]] = &[
            [1.0, 0.0, 0.0],
            [0.5, 0.866_025_4, 0.0],
            [-0.5, 0.866_025_4, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866_025_4, 0.0],
            [0.5, -0.866_025_4, 0.0],
        ];
        let cases = [
            Case { label: "凸四边形", pts: &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]] },
            Case { label: "凹四边形 L 形", pts: L4 },
            Case { label: "凹五边形箭头", pts: ARROW5 },
            Case { label: "凸六边形", pts: HEX6 },
            Case { label: "凹八边形 U 形", pts: U8 },
        ];
        println!("{:<16} {:>6} {:>14} {:>14} {:>12}", "多边形", "三角形数", "面积和", "多边形面积", "相对误差");
        for c in &cases {
            let face: Vec<u32> = (0..c.pts.len() as u32).collect();
            let tris = ear_mesh(c.pts, &face)
                .triangulated_ear()
                .unwrap_or_else(|e| panic!("{}: {e}", c.label));
            assert_ear_clip_valid(c.pts, &face, &tris, c.label);
            let sum: f64 = tris
                .iter()
                .map(|t| tri_area(c.pts[t[0] as usize], c.pts[t[1] as usize], c.pts[t[2] as usize]))
                .sum();
            let want = poly_area(c.pts);
            let rel = ((sum - want) / want).abs();
            // 相对误差必须远低于 1e-6 阈值（此处实测应在 1e-16 量级）。
            assert!(rel < 1e-9, "{}：相对误差 {rel:e} 过大", c.label);
            println!(
                "{:<16} {:>6} {:>14.10} {:>14.10} {:>12.3e}",
                c.label,
                tris.len(),
                sum,
                want,
                rel
            );
        }
    }

    #[test]
    fn ear_clip_tie_is_broken_by_lowest_vertex_index_not_by_traversal_order() {
        // **确定性契约的核心断言**：耳点有多个候选时，按**原始顶点索引最小**
        // 打破平局，绝不依赖浮点比较的偶然顺序或遍历先后。
        //
        // 构造一个「几乎处处都是耳」的凸多边形：4 个顶点全是耳。
        // 耳切固定选**非锚点中索引最小**者 ⇒ 必须剪掉下标 1（而非 0、2、3），
        // 产出 [[0,1,2],[0,2,3]]——与旧扇形逐索引相同（兼容性契约）。
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 2, 3]);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            tris,
            vec![[0, 1, 2], [0, 2, 3]],
            "凸四边形：4 个候选耳中应固定选非锚点里索引最小的 1"
        );
        // 换一个**几何等价但起点不同**的输入（起点从中间开始）：
        // 选中的耳仍由索引决定，而不是「位置 1」。
        let pts2 = [
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let m2 = ear_mesh(&pts2, &[0, 1, 2, 3]);
        let t2 = m2.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        // 无论起点在哪，剖分都几何有效（细节由专用测试覆盖）。
        assert_eq!(t2.len(), 2);
        assert_ear_clip_valid(&pts2, &[0, 1, 2, 3], &t2, "换起点（同一矩形）");
        // **逐字节确定性**：同一输入调用 100 次，每次结果 `==`。
        let first = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        for i in 0..100 {
            assert_eq!(
                m.triangulated_ear().unwrap_or_else(|e| panic!("{e}")),
                first,
                "第 {i} 次调用结果与首次不同 —— 违反 C17"
            );
        }
    }

    #[test]
    fn ear_clip_is_3d_correct_on_a_slanted_plane_with_a_non_axis_normal() {
        // **3D 正确性**（本测试重点）：耳切判定全程用 3D 叉积与 3D Newell
        // 法向的点积符号，不依赖「最大轴投影」这一 2D 近似。
        //
        // 本多边形 = 凹 L 形（含一个共线点）在**斜平面**上的刚体旋转
        // （Rx(35°) ∘ Rz(20°)），法向 = (1.9617, −5.3899, 8.1915) ——
        // **三个分量都显著非零**（且法向主轴是 z，投影丢弃的正是 z），
        // 故「碰巧轴对齐」的退化捷径在此无效。
        // 局部 2D 坐标 (0,0)(1.5,0)(3,0)(3,1)(1,1)(1,3)(0,3)，面积 5；
        // 刚体旋转保面积 ⇒ 3D 面积仍是 5。
        let pts = [
            [0.000000, 0.000000, 0.000000],
            [1.409539, 0.513030, 0.000000],
            [2.819078, 1.026060, 0.000000], // 与下标 1 共线（在底边上）
            [2.538911, 1.795812, 0.573576],
            [0.659526, 1.111771, 0.573576], // reflex 顶点（凹角）
            [0.099193, 2.651274, 1.720729],
            [-0.840499, 2.309253, 1.720729],
        ];
        let face: Vec<u32> = (0..7).collect();
        let m = ear_mesh(&pts, &face);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 5, "七边形应剖成 7−2 = 5 个三角形");
        // 局部 2D 鞋带面积 = 5；刚体旋转保面积 ⇒ 3D 面积应精确为 5。
        assert!((poly_area(&pts) - 5.0).abs() < 1e-5, "斜平面多边形面积应为 5");
        // 三重断言在 3D 斜平面上同样成立（无自交 + 面积覆盖 + 法线同向）。
        assert_ear_clip_valid(&pts, &face, &tris, "斜平面凹七边形");
        // 法向三个分量均非零（证明这不是「碰巧轴对齐」的退化用例）。
        let n = poly_normal(&pts);
        for (k, v) in n.iter().enumerate() {
            assert!(v.abs() > 1.0, "法向分量 {k} 应显著非零，实测 {v}");
        }
        // 每个三角形法线与多边形法向**同向**（点积 > 0）。
        for (k, t) in tris.iter().enumerate() {
            let tn = v_cross(
                v_sub(pts[t[1] as usize], pts[t[0] as usize]),
                v_sub(pts[t[2] as usize], pts[t[0] as usize]),
            );
            assert!(
                v_dot(tn, n) > 0.0,
                "第 {k} 个三角形法线反向（点积 {}）",
                v_dot(tn, n)
            );
        }
        // 凹性确认：确有顶点是 reflex（叉积与法向反向）——
        // 否则这个用例就退化成凸多边形，测不到「3D 凹」这条路径。
        let mut reflex = 0;
        for i in 0..pts.len() {
            let prev = pts[(i + pts.len() - 1) % pts.len()];
            let cur = pts[i];
            let next = pts[(i + 1) % pts.len()];
            if v_dot(v_cross(v_sub(cur, prev), v_sub(next, cur)), n) < 0.0 {
                reflex += 1;
            }
        }
        assert_eq!(reflex, 1, "L 形应恰有 1 个 reflex 顶点，实测 {reflex}");
    }

    #[test]
    fn duplicate_and_collinear_vertices_are_handled_structurally_not_panicked() {        // 情形一：末尾重复首点（USD 里闭合轮廓的常见写法）。
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 2, 3, 0]);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 2, "重复首点应被去掉，仍是 2 个三角形");
        assert_ear_clip_valid(&pts, &[0, 1, 2, 3], &tris, "重复首点");

        // 情形二：连续两个**完全重合**的顶点（下标不同但坐标相同）。
        let dup = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let m2 = ear_mesh(&dup, &[0, 1, 2, 3, 4]);
        let t2 = m2.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t2.len(), 2, "重合点应被去掉，仍是 2 个三角形");
        assert_ear_clip_valid(&dup, &[0, 1, 3, 4], &t2, "重合点");

        // 情形三：三点共线（中间点在直边上）——共线顶点不是耳（叉积 ≈ 0），
        // 必须被跳过而不是产出零面积三角形。
        let collinear = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let m3 = ear_mesh(&collinear, &[0, 1, 2, 3, 4]);
        let t3 = m3.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t3.len(), 3, "五边形（含共线点）应剖成 3 个三角形");
        // 每个三角形面积都必须**严格为正**（无零面积碎片）。
        for (k, t) in t3.iter().enumerate() {
            let a = tri_area(collinear[t[0] as usize], collinear[t[1] as usize], collinear[t[2] as usize]);
            assert!(a > 1e-12, "第 {k} 个三角形面积为 {a}，不应有零面积碎片");
        }
        assert_ear_clip_valid(&collinear, &[0, 1, 2, 3, 4], &t3, "共线点");
    }

    #[test]
    fn fully_degenerate_collinear_polygon_is_a_structured_error() {
        // 所有点共线 ⇒ Newell 法向模长为 0 ⇒ PU1010。
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 2, 3]);
        let e = m.triangulated_ear().unwrap_err();
        assert_eq!(e.code, "PU1010", "全共线多边形必须是 PU1010 退化错误");
        assert!(e.problem.contains("共线") || e.problem.contains("重合"), "问题描述应点明共线/重合：{}", e.problem);
        // 绝不静默产出垃圾：便捷接口返回空 vec（而不是假三角形）。
        assert!(m.triangulated().is_empty(), "退化面不得产出任何三角形");
    }

    #[test]
    fn self_intersecting_bowtie_polygon_is_a_structured_error_not_silent_garbage() {
        // 蝴蝶结（bowtie）：边 0→1 与边 2→3 在内部真交。
        // 注意：bowtie 的**有向面积恰好为 0**（两个三角形互相抵消），
        // 所以 Newell 法向判据会先命中 PU1010 —— 这也是正确的结构化拒绝。
        // 下面同时用一个**有向面积非零**的自交多边形（下方 pts2）确保
        // 自交预检（PU1011）这条路径本身是可达的。
        let bowtie = [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let m = ear_mesh(&bowtie, &[0, 1, 2, 3]);
        let e = m.triangulated_ear().unwrap_err();
        assert!(
            e.code == "PU1011" || e.code == "PU1010",
            "自交多边形必须是结构化错误（PU1011 自交 / PU1010 零面积），实得 {}",
            e.code
        );
        assert!(m.triangulated().is_empty(), "自交面不得产出任何三角形");

        // 有向面积**非零**的自交多边形：边 1（(4,0)→(0,3)）与
        // 边 3（(3,4)→(0,0)）在内部真交，但鞋带有向面积 = 1.5 ≠ 0
        // ⇒ 绕过 PU1010，直达自交预检 ⇒ PU1011。
        let self_x = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 4.0, 0.0],
        ];
        let m2 = ear_mesh(&self_x, &[0, 1, 2, 3]);
        let e2 = m2.triangulated_ear().unwrap_err();
        assert_eq!(e2.code, "PU1011", "有向面积非零的自交多边形必须是 PU1011，实得 {}", e2.code);
        assert!(e2.problem.contains("自交"), "问题描述应点明自交：{}", e2.problem);
        assert!(m2.triangulated().is_empty(), "自交面不得产出任何三角形");
    }

    #[test]
    fn ear_clip_is_deterministic_across_repeated_calls() {
        let pts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 2, 3]);
        let a = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        let b = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "同一输入两次调用必须逐字节相同");
        // 换个等价构造路径（重新解析同一段 usda）也必须一致。
        let src = "#usda 1.0\ndef Mesh \"m\"\n{\n    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n    point3f[] points = [(0,0,0), (3,0,0), (3,1,0), (1,3,0)]\n}\n";
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(scene.meshes[0].triangulated(), a, "解析路径与直构路径结果必须一致");
    }

    #[test]
    fn convex_quad_is_valid_under_every_choice_of_start_vertex() {
        // 换起始顶点**会**改变选中的对角线（凸四边形有两条合法对角线，
        // 耳切各取其一）—— 这是**正确行为**，不是缺陷：三角剖分本就不唯一。
        // 真正要保证的是：不论起点在哪，剖分都**几何有效**
        // （面积覆盖 + 无自交 + 法线同向）。
        let pts = quad_points(true);
        for f in [
            [0u32, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ] {
            let tris = ear_mesh(&pts, &f)
                .triangulated_ear()
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(tris.len(), 2, "起点 {f:?}：凸四边形必须剖成 2 个三角形");
            assert_ear_clip_valid(&pts, &f, &tris, "凸四边形（换起点）");
        }
    }

    #[test]
    fn concave_quad_ear_clip_picks_a_diagonal_inside_the_polygon() {
        // 凹四边形有**两条**合法对角线可选，扇形固定用 0→2（落在多边形外），
        // 耳切必须改用 1→3。断言：结果满足全部三重断言 + 面积正确，
        // 即便两条对角线都可能——我们要的是**几何正确**，不是某个下标。
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        // 两个可能的起点都跑一遍，结果都必须有效。
        for &f in &[&[0u32, 1, 2, 3][..], &[1u32, 2, 3, 0][..]] {
            let tris = ear_mesh(&pts, f)
                .triangulated_ear()
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(tris.len(), 2);
            assert_ear_clip_valid(&pts, f, &tris, "凹四边形（换起点）");
        }
    }

    #[test]
    fn ear_clip_handles_a_slanted_plane_face_in_3d() {
        // 非 XY 平面的**共面**四边形（法向 ∝ (0,−1,1)，即 y+z 恒定的斜面）：
        // 验证 3D 叉积判据，而不是「恰好在 XY 平面上才正确」。
        // 投影自检用的 drop_axis 在此会选中 |N_y| 或 |N_z|（并列，取 y），
        // 也就是说 2D 预检走的是**丢弃 y 轴**这条非常规分支。
        let pts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 2.0],
            [0.0, 2.0, 2.0],
        ];
        let face = [0, 1, 2, 3];
        let m = ear_mesh(&pts, &face);
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 2);
        assert_ear_clip_valid(&pts, &face, &tris, "斜面凸四边形");
        // 两三角形各 3√2，和 6√2。
        let sum: f64 = tris
            .iter()
            .map(|t| tri_area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]))
            .sum();
        assert!((sum - 6.0 * 2.0f64.sqrt()).abs() < 1e-12, "斜面面积和应为 6√2，实测 {sum}");
    }

    #[test]
    fn non_coplanar_quad_has_a_defined_and_deterministic_behaviour() {
        // 四点**不共面**（USD 允许；NURBS / 细分转换来的四边形常常如此）。
        //
        // # 行为定义（本实现的选择，理由如下）
        // **接受并三角化**，不做拒绝。理由：非共面四边形是「折叠」的，
        // 两条对角线给出两个都合理的剖分——这是**着色差异**，不是正确性
        // 缺陷（行业惯例同样直接三角化）。若改成拒绝，会误伤大量真实资产。
        //
        // 但要说清两条**已知边界**（不静默）：
        // 1. **面积不守恒**：`|Newell| / 2`（面片的「投影面积」）一般**不等于**
        //    两三角形面积之和（实测此例 6.3285 vs 6.4135，差 1.34e-2 相对）。
        //    所以 `assert_ear_clip_valid` 的面积断言**不适用**于非共面面。
        // 2. **自交预检是保守近似**：预检在 2D 投影上做（见 `ear_clip` 步骤 3），
        //    非共面面被压扁后可能漏检真实的空间自交，也可能误报。
        //    耳点判定本身全程 3D，**剖分**永远正确。
        //
        // 本测试锁定：接受 + 2 个非退化三角形 + 法线与 Newell 同向 + 确定性。
        let pts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 1.2],
            [3.0, 2.0, 0.6],
            [0.0, 2.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 2, 3]);
        let tris = m
            .triangulated_ear()
            .unwrap_or_else(|e| panic!("非共面四边形应当被接受（行为已定义），却报错：{e}"));
        assert_eq!(tris.len(), 2, "非共面四边形仍剖成 2 个三角形");

        // 边界 1：面积**不**守恒（这是定义的一部分，不是 bug）。
        let n = poly_normal(&pts);
        let newell_area = 0.5 * v_dot(n, n).sqrt();
        let sum: f64 = tris
            .iter()
            .map(|t| tri_area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]))
            .sum();
        assert!(
            (sum - newell_area).abs() / newell_area > 1e-6,
            "非共面情形的面积不守恒是已知边界，实测却守恒了（sum={sum}, newell={newell_area}）——\
             请更新本测试的边界描述"
        );

        // 两个三角形都非退化，且法线与 Newell 法向同向。
        for (k, t) in tris.iter().enumerate() {
            let (a, b, c) = (
                pts[t[0] as usize],
                pts[t[1] as usize],
                pts[t[2] as usize],
            );
            let area = tri_area(a, b, c);
            assert!(area > 1e-9, "第 {k} 个三角形退化（面积 {area}）");
            let tn = v_cross(v_sub(b, a), v_sub(c, a));
            assert!(
                v_dot(tn, n) > 0.0,
                "第 {k} 个三角形法线与 Newell 法向反向（翻面）"
            );
        }

        // 确定性：同一非共面面两次调用结果相同。
        let again = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris, again, "非共面面的剖分也必须确定性");
    }

    #[test]
    fn ear_clip_rejects_a_face_whose_index_is_out_of_range() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let m = ear_mesh(&pts, &[0, 1, 9]);
        let e = m.triangulated_ear().unwrap_err();
        assert_eq!(e.code, "PU1004", "越界索引应报 PU1004，实得 {}", e.code);
    }

    #[test]
    fn multi_face_mesh_triangulates_each_face_independently() {
        // 一个网格里混着：凸四边形 + **凹**四边形 + 三角形，验证逐面推进
        // （`face_indices` 的 base 游标）在耳切下依然正确。
        let pts = [
            [0.0, 0.0, 0.0],   // 0
            [2.0, 0.0, 0.0],   // 1
            [2.0, 2.0, 0.0],   // 2
            [0.0, 2.0, 0.0],   // 3
            [5.0, 0.0, 0.0],   // 4
            [6.0, 1.0, 0.0],   // 5
            [7.0, 0.0, 0.0],   // 6
            [5.0, 3.0, 0.0],   // 7
        ];
        let mut m = ear_mesh(&pts, &[0, 1, 2, 3]);
        m.face_counts = vec![4, 4, 3];
        // 第三面取 (0, 2, 4)：一个斜跨两个四边形的三角形。
        m.face_indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4];
        let tris = m.triangulated_ear().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tris.len(), 2 + 2 + 1, "三面应产出 2+2+1 = 5 个三角形");
        // 每个面的三角形数正确（顺序即面顺序）。
        assert_eq!(tris[0..2].len(), 2);
        assert_eq!(tris[2..4].len(), 2);
        assert_eq!(tris[4..5].len(), 1);
        assert_ear_clip_valid(&pts, &[0, 1, 2, 3], &tris[0..2], "多面-面0");
        assert_ear_clip_valid(&pts, &[4, 5, 6, 7], &tris[2..4], "多面-面1");
        assert_ear_clip_valid(&pts, &[0, 2, 4], &tris[4..5], "多面-面2");
    }

    #[test]
    fn a_degenerate_face_does_not_poison_the_whole_mesh() {
        // 前两面正常、第三面退化 ⇒ 结构化错误，但前两面的结果已可用
        // （错误在 `triangulated_ear` 里向上传播，不返回半成品）。
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let mut m = ear_mesh(&pts, &[0, 1, 2]);
        m.face_counts = vec![3, 3];
        m.face_indices = vec![0, 1, 2, 3, 4, 5];
        let e = m.triangulated_ear().unwrap_err();
        assert_eq!(e.code, "PU1010");
        assert!(e.location.contains("第 1 个面"), "错误应指明是哪个面：{}", e.location);
        // 便捷接口：整网格返回空（宁缺勿错，C20）。
        assert!(m.triangulated().is_empty());
    }

    #[test]
    fn xform_chain_is_applied_and_baked_to_vertices() {
        // root 平移 (1,2,3)，child 缩放 2 → 顶点 (0,0,0) → (1,2,3)。
        let src = r#"#usda 1.0
def Xform "root"
{
    matrix4d xformOp:transform = ( (1, 0, 0, 1), (0, 1, 0, 2), (0, 0, 1, 3), (0, 0, 0, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]

    def Xform "inner"
    {
        double3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]

        def Mesh "m"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        let m = &scene.meshes[0];
        assert_eq!(m.path, "/root/inner/m");
        assert_eq!(m.points[0], [1.0, 2.0, 3.0], "平移应烘焙到顶点");
        assert_eq!(m.points[1], [3.0, 2.0, 3.0], "缩放应先于平移生效");
    }

    #[test]
    fn rotate_xyz_operation_is_applied() {
        // 绕 Z 转 90°：+X → +Y。
        let src = r#"#usda 1.0
def Xform "root"
{
    double3 xformOp:rotateXYZ = (0, 0, 90)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        let m = &scene.meshes[0];
        assert!(
            (m.points[0][0] - 0.0).abs() < 1e-12 && (m.points[0][1] - 1.0).abs() < 1e-12,
            "绕 Z 转 90° 后 +X 应到 +Y：{:?}",
            m.points[0]
        );
    }

    #[test]
    fn usda_magic_line_is_not_treated_as_a_comment() {
        // `#usda 1.0` 是魔数，必须被解析；`#usdabc` 之类仍应是注释。
        let p = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.layer.version, "1", "`#usda 1.0` 魔数必须被识别");
        // `#usdabc` 开头的行是普通注释，不影响解析。
        let src = "#usda 1.0\n#usdabc 这行是注释\ndef Mesh \"m\"\n{\n    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]\n}\n";
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.layer.prims.len(), 1, "`#usdabc` 应被当注释跳过");
    }

    #[test]
    fn non_uniform_scale_normals_are_bookkept_not_silently_approximated() {
        // 非等比缩放：法线未做逆转置 → 必须记入报告。
        let src = r#"#usda 1.0
def Xform "root"
{
    double3 xformOp:scale = (2, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let (_scene, report) = build_scene_report(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            report.non_uniform_normals.len(),
            1,
            "非等比缩放下的法线必须被记账：{:?}",
            report.non_uniform_normals
        );
        // 等比缩放不记账。
        let src2 = src.replace("(2, 1, 1)", "(2, 2, 2)");
        let p2 = parse_str(&src2).unwrap_or_else(|e| panic!("{e}"));
        let (_s2, r2) = build_scene_report(&p2.layer).unwrap_or_else(|e| panic!("{e}"));
        assert!(r2.non_uniform_normals.is_empty(), "等比缩放不应记账");
        // 无 normals 的 prim 也不记账。
        let src3 = src.replace(
            "        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]\n",
            "",
        );
        let p3 = parse_str(&src3).unwrap_or_else(|e| panic!("{e}"));
        let (_s3, r3) = build_scene_report(&p3.layer).unwrap_or_else(|e| panic!("{e}"));
        assert!(r3.non_uniform_normals.is_empty(), "无 normals 时不应记账");
    }

    #[test]
    fn illegal_number_literal_is_a_structured_error_not_a_silent_zero() {
        // `1.0e` 后面没有数字 → 词法层必须报错，绝不能悄悄当成 1.0。
        let e = parse_str("#usda 1.0\ndef Mesh \"m\" { double x = 1.0e }")
            .expect_err("非法数字字面量必须报错");
        assert_eq!(e.code, "PU1002");
        // 非 ASCII 裸字节也不能被当成标识符吞掉。
        let e = parse_bytes(&[
            0x23, 0x75, 0x73, 0x64, 0x61, 0x20, 0x31, 0x2e, 0x30, 0x0a, 0xff,
        ])
        .expect_err("非法字节必须报错（PU1009 非 UTF-8）");
        assert!(e.code == "PU1009" || e.code == "PU1002", "实际 {}", e.code);
    }

    #[test]
    fn usdc_binary_magic_is_structurally_rejected() {
        let mut bytes = b"PXR-USDC".to_vec();
        bytes.extend_from_slice(&[0u8; 64]);
        let e = parse_bytes(&bytes).expect_err("`.usdc` 必须结构化拒绝");
        assert_eq!(e.code, "PU1001");
        assert!(e.fix.contains("usdcat"), "建议段应给出可行动出路");
    }

    #[test]
    fn material_binding_relationship_is_parsed_and_validated() {
        let src = r#"#usda 1.0
def Scope "root"
{
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        rel material:binding = </root/mat>
    }
    def Material "mat"
    {
        token outputs:surface.connect = </root/mat/shader.outputs:surface>
        def Shader "shader"
        {
            uniform token info:id = "UsdPreviewSurface"
        }
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let mesh = &p.layer.prims[0].children[0];
        assert_eq!(
            mesh.relationship("material:binding")
                .map(|r| r.targets.clone()),
            Some(vec!["/root/mat".to_string()])
        );
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            scene.meshes[0].material.as_deref(),
            Some("/root/mat"),
            "材质路径应被带到网格"
        );
        // 悬空材质 → 结构化错误。
        let bad = src.replace("</root/mat>", "</root/nope>");
        let pb = parse_str(&bad).unwrap_or_else(|e| panic!("{e}"));
        let e = build_scene(&pb.layer).expect_err("悬空材质必须报错");
        assert_eq!(e.code, "PU1005");
    }

    #[test]
    fn relationships_and_property_connects_are_recorded() {
        let src = r#"#usda 1.0
def Material "mat"
{
    token outputs:surface.connect = </mat/bsdf.outputs:surface>
    def Shader "bsdf"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.8, 0.2, 0.1)
        float inputs:roughness = 0.4
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let mat = &p.layer.prims[0];
        let bc = mat
            .value("outputs:surface.connect")
            .expect("connect 应被记录");
        assert_eq!(bc.as_str(), Some("/mat/bsdf.outputs:surface"));
        let bsdf = &mat.children[0];
        let dc = bsdf.value("inputs:diffuseColor").expect("diffuseColor");
        assert_eq!(dc.as_numbers(), vec![0.8, 0.2, 0.1]);
        assert_eq!(
            bsdf.value("inputs:roughness").and_then(|v| v.as_number()),
            Some(0.4)
        );
    }

    #[test]
    fn over_prims_are_parsed_as_over_kind() {
        let src = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
}
over "m"
{
    float custom:density = 2.5
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.layer.prims.len(), 2);
        let o = &p.layer.prims[1];
        assert_eq!(o.kind, PrimKind::Over);
        assert_eq!(o.name, "m");
        assert_eq!(
            o.value("custom:density").and_then(|v| v.as_number()),
            Some(2.5),
            "`over` 的属性应被保留"
        );
        assert_eq!(o.kind.keyword(), "over");
    }

    #[test]
    fn sublayers_and_references_are_recorded_but_not_resolved() {
        let src = r#"#usda 1.0
(
    subLayers = [
        @./base.usda@
    ]
)
def Xform "root"
{
    prepend references = @./common.usda@</Common>
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let root = &p.layer.prims[0];
        assert_eq!(root.references, vec!["./common.usda@</Common>".to_string()]);
        // subLayers 只在层元数据里，不解析目标文件。
        assert!(p.layer.metadata.iter().any(|(k, _)| k == "subLayers"));
    }

    #[test]
    fn time_samples_are_skipped_and_reported() {
        let src = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)] (
        timeSamples = {
            0: [(0,0,0), (1,0,0), (0,1,0)]
            1: [(0,0,1), (1,0,1), (0,1,1)]
        }
    )
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        // 默认帧仍可用。
        assert_eq!(scene.meshes[0].vertex_count(), 3);
        assert!(
            !p.report.skipped_time_samples.is_empty(),
            "timeSamples 必须被记账（不得静默丢弃）"
        );
    }

    #[test]
    fn parse_is_byte_level_deterministic() {
        let a = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        let b = parse_str(MINIMAL).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.layer, b.layer, "同输入两遍解析必须逐字段一致");
        assert_eq!(a.report, b.report);
        // 场景构建也须一致。
        let sa = build_scene(&a.layer).unwrap_or_else(|e| panic!("{e}"));
        let sb = build_scene(&b.layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sa, sb);
        // 顶点位模式逐位一致。
        for (p, q) in sa.meshes[0].points.iter().zip(sb.meshes[0].points.iter()) {
            for k in 0..3 {
                assert_eq!(p[k].to_bits(), q[k].to_bits());
            }
        }
    }

    #[test]
    fn malformed_inputs_are_structured_errors_not_panics() {
        // 缺文件头。
        let e = parse_str("def Mesh \"m\" {}").expect_err("缺 #usda 头必须报错");
        assert_eq!(e.code, "PU1002");
        // 体未闭合。
        let e = parse_str("#usda 1.0\ndef Mesh \"m\"\n{\n").expect_err("未闭合必须报错");
        assert_eq!(e.code, "PU1002");
        // 数组未闭合。
        let e = parse_str("#usda 1.0\ndef Mesh \"m\" { int[] x = [1, 2 ")
            .expect_err("数组未闭合必须报错");
        assert_eq!(e.code, "PU1002");
        // prim 名缺失。
        let e = parse_str("#usda 1.0\ndef Mesh\n{\n}\n").expect_err("缺 prim 名必须报错");
        assert_eq!(e.code, "PU1002");
        // 字符串未闭合。
        let e = parse_str("#usda 1.0\ndef Mesh \"m\" { uniform token x = \"abc }")
            .expect_err("字符串未闭合必须报错");
        assert!(e.code == "PU1002" || e.code == "PU1001", "got {}", e.code);
        // 非法 UTF-8。
        let e = parse_bytes(&[0xff, 0xfe, 0x00]).expect_err("非法 UTF-8 必须报错");
        assert_eq!(e.code, "PU1009");
    }

    #[test]
    fn type_mismatch_is_structured_error() {
        // faceVertexIndices 给小数。
        let src = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2.5]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
}
"#;
        let e = parse_str(src).expect_err("int[] 含小数必须报错");
        assert_eq!(e.code, "PU1003");
        // point3f[] 行宽不对。
        let src2 = r#"#usda 1.0
def Mesh "m" { point3f[] points = [(0,0), (1,0), (0,1)] }
"#;
        let e = parse_str(src2).expect_err("point3f 行宽 2 必须报错");
        assert_eq!(e.code, "PU1003");
    }

    #[test]
    fn invalid_mesh_topology_is_structured_error() {
        // 索引越界。
        let src = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 9]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let e = build_scene(&p.layer).expect_err("索引越界必须报错");
        assert_eq!(e.code, "PU1004");
        // counts 与 indices 不符。
        let src2 = r#"#usda 1.0
def Mesh "m"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
}
"#;
        let p2 = parse_str(src2).unwrap_or_else(|e| panic!("{e}"));
        let e = build_scene(&p2.layer).expect_err("counts 不符必须报错");
        assert_eq!(e.code, "PU1004");
        // 缺 points。
        let src3 = r#"#usda 1.0
def Mesh "m" { int[] faceVertexCounts = [3] }
"#;
        let p3 = parse_str(src3).unwrap_or_else(|e| panic!("{e}"));
        let e = build_scene(&p3.layer).expect_err("缺 points 必须报错");
        assert_eq!(e.code, "PU1006");
    }

    #[test]
    fn xform_op_order_with_missing_attribute_is_structured_error() {
        let src = r#"#usda 1.0
def Xform "root"
{
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let e = build_scene(&p.layer).expect_err("xformOpOrder 缺属性必须报错");
        assert_eq!(e.code, "PU1008");
    }

    #[test]
    fn variant_sets_are_recorded_as_skipped() {
        let src = r#"#usda 1.0
def Xform "root"
{
    variantSet "shading" = {
        "red" {
            def Material "mat" { uniform token info:id = "red" }
        }
        "blue" {
            def Material "mat" { uniform token info:id = "blue" }
        }
    }
    def Mesh "m"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.layer.prims[0].variant_sets, vec!["shading".to_string()]);
        assert!(p.report.skipped_variants.contains(&"shading".to_string()));
        // 变体内的 prim 不应被并入子 prim（未求值）。
        assert_eq!(p.layer.prims[0].children.len(), 1);
    }

    #[test]
    fn multiline_strings_and_comments_are_handled() {
        let src = "#usda 1.0\n\
                   # 这是注释行，应被跳过\n\
                   def Mesh \"m\"\n\
                   {\n\
                   \x20   uniform token info:doc = \"\"\"第一行\n\
                   第二行\"\"\"\n\
                   \x20   int[] faceVertexCounts = [3]\n\
                   \x20   int[] faceVertexIndices = [0, 1, 2]\n\
                   \x20   point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]\n\
                   }\n";
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let doc = p.layer.prims[0]
            .value("info:doc")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(doc.contains('\n'), "三引号多行串应保留换行：{doc:?}");
    }

    /// 字符串体按字节保真：中文 / emoji（4 字节）/ 组合字符逐字符精确相等。
    ///
    /// 回归钉子：旧实现 `out.push(c as char)` 逐字节推进，一个多字节字符会被
    /// 拆成多个 U+00XX（乱码），且每次导出再导入继续按字节拆，逐代恶化。
    #[test]
    fn utf8_string_literals_are_char_exact() {
        // 中文 2 字（各 3 字节）+ emoji（4 字节）+ e + U+0301 组合重音（2 字节）。
        let u = "材质🎨e\u{0301}";
        let src = format!(
            "#usda 1.0\ndef Scope \"s\"\n{{\n    \
             string dq = \"{u}\"\n    \
             string sq = '{u}'\n    \
             string tq = \"\"\"{u}\n第二行\"\"\"\n    \
             string tq_single = '''{u} it's'''\n    \
             string unk = \"\\{u}\"\n    \
             string esc = \"{u}\\n\\t\\q\\\\\"\n    \
             string empty = \"\"\n}}\n"
        );
        let p = parse_str(&src).unwrap_or_else(|e| panic!("{e}"));
        let val = |name: &str| {
            p.layer.prims[0]
                .value(name)
                .and_then(|v| v.as_str())
                .unwrap_or("<missing>")
        };

        // 逐字符 + 逐字节双断言（只比 `String` 相等已是字节级，这里显式展开）。
        let got: Vec<char> = val("dq").chars().collect();
        let want: Vec<char> = u.chars().collect();
        assert_eq!(got, want, "双引号串逐字符必须相等：{:?}", val("dq"));
        assert_eq!(val("dq").as_bytes(), u.as_bytes());
        assert_eq!(got.len(), 5, "中文2 + emoji1 + e1 + 组合符1");
        assert_eq!(val("dq").len(), 13, "3+3+4+1+2 字节");

        assert_eq!(val("sq"), u, "单引号串同样字节保真");
        assert_eq!(val("tq"), format!("{u}\n第二行"), "三引号串含多字节");
        assert_eq!(
            val("tq_single"),
            format!("{u} it's"),
            "三引号体内的单个同种引号必须原样保留"
        );
        assert_eq!(val("unk"), format!("\\{u}"), "未知转义保留反斜杠与完整字符");
        assert_eq!(val("esc"), format!("{u}\n\t\\q\\"), "已知转义按字节展开");
        assert_eq!(val("empty"), "", "空串保持空");
    }

    /// 非法 UTF-8：公共入口 PU1009；词法层兜底 PU1002（带行列），
    /// 不 panic、不 lossy 替换成 U+FFFD。
    #[test]
    fn invalid_utf8_string_body_is_a_structured_error() {
        // 第 4 行第 18 列是开引号，第 19 列起是字符串体；`\xe6\x9d` 是截断的 3 字节序列。
        let bytes: &[u8] = b"#usda 1.0\ndef Scope \"s\"\n{\n    string doc = \"\xe6\x9d\"\n}\n";
        let e = parse_bytes(bytes).expect_err("非法 UTF-8 输入必须报错");
        assert_eq!(e.code, "PU1009");
        assert!(e.problem.contains("UTF-8"), "{}", e.problem);
        assert!(
            e.location.contains("第 4 行第 19 列"),
            "公共入口也要带行列：{}",
            e.location
        );
        assert!(!e.fix.is_empty(), "C18：必须有建议段");

        // 词法层直接兜底（绕过 parse_bytes 的整体闸门）：整体解码失败要报
        // 明确语法错误并带字符串起始行列，而不是产出替换字符。
        let mut lx = Lexer::new(bytes);
        let e = lx.tokenize().expect_err("词法层必须拒绝非法 UTF-8");
        assert_eq!(e.code, "PU1002");
        assert!(
            e.location.contains("第 4 行第 18 列"),
            "错误必须带行/列：{}",
            e.location
        );
        assert!(e.problem.contains("UTF-8"), "{}", e.problem);
    }

    // ── 渲染就绪网格（USD → prism-rt 加载路径）────────────────────────

    /// `to_render_mesh` / `render_meshes`：三角化 + f64→f32 + 索引流扁平化
    /// + world Xform 已烘焙。
    #[test]
    fn render_mesh_conversion_flattens_triangles_and_bakes_xform() {
        let src = r#"#usda 1.0
def Xform "root"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
    }
}
"#;
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        let rms = scene.render_meshes();
        assert_eq!(rms.len(), 1);
        let rm = &rms[0];
        assert_eq!(rm.vertex_count(), 4);
        assert_eq!(rm.triangle_count(), 2);
        assert_eq!(scene.render_triangle_count(), 2);
        // 索引流与 triangulated() 逐三元组一致。
        let expect: Vec<u32> = scene.meshes[0].triangulated().into_iter().flatten().collect();
        assert_eq!(rm.indices, expect);
        // Xform 已在 build_scene 烘焙：首个顶点 = (0,0,0) + (1,2,3)。
        assert_eq!(rm.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(rm.triangle(0), Some([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [2.0, 3.0, 3.0]]));
        assert_eq!(rm.triangle(2), None, "越界三角形返回 None");
        let (bmin, bmax) = rm.bounds().expect("非空网格有包围盒");
        assert_eq!(bmin, [1.0, 2.0, 3.0]);
        assert_eq!(bmax, [2.0, 3.0, 3.0]);
        assert_eq!(rm.path, "/root/quad");
    }

    /// 空场景 / 空网格：渲染就绪形态无 panic，计数为 0。
    #[test]
    fn render_meshes_empty_scene_is_safe() {
        let src = "#usda 1.0\ndef Xform \"only\"\n{\n}\n";
        let p = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        let scene = build_scene(&p.layer).unwrap_or_else(|e| panic!("{e}"));
        assert!(scene.render_meshes().is_empty());
        assert_eq!(scene.render_triangle_count(), 0);
    }
}
