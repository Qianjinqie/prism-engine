//! 手写最小 SPIR-V 字节码（V2.0-G：无 glslang 环境的真实二进制供给）。
//!
//! # 为什么手写（如实记录做法）
//!
//! 正常路径是「WGSL → SPIR-V 由自研发射器生成（`crate::metal::msl` 头注的
//! 定案）」，构建期用 `glslangValidator`/`glslc` 产出 `.spv` 再 `include_bytes!`
//! 打进二进制。**本仓库构建环境没有 glslang/glslc**（已实测），于是本模块
//! 走预案：**手写 SPIR-V 指令流**，以 `static` 常量数组形式**编译期入二进制**
//! （零运行期文件 I/O、零第三方 crate —— D-15）。
//!
//! 手写对象是一个**语义真实**的 compute kernel：
//!
//! ```wgsl
//! @group(0) @binding(0) var<storage, read>       src: array<f32>;
//! @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
//! @compute @workgroup_size(64)
//! fn mul2(@builtin(global_invocation_id) gid: vec3<u32>) {
//!     dst[gid.x] = src[gid.x] * 2.0;
//! }
//! ```
//!
//! （即 [`cpu_reference_mul2`]，parity 由测试断言「二进制宣称的语义 == CPU
//! 参考」——二进制无法在本环境执行，故只能做**结构 + 语义锚定**双重校验。）
//!
//! # 供给边界（不撒谎）
//!
//! * [`MODULES`] 只有 `spv_mul2_probe` 一个模块，供 compute pipeline 装配
//!   路径的**真机 smoke** 使用；
//! * RT 语义 kernel（主射线发射 / wavefront 求交）的 SPIR-V **待真机工具链
//!   生成**，`KernelDesc::spv_module` 留空并由
//!   `crate::compute::kernel_spv_status` 如实报 `pending-toolchain`；
//! * 本模块**不做**完整 SPIR-V 语义验证（那需要 `spirv-val`，本环境没有），
//!   只做**结构验证**：[`validate`] 检查 magic/version/schema/bound、指令
//!   字长自洽、OpCode 白名单、入口名、`LocalSize`、描述符绑定数——足以拦截
//!   「字节被误改/截断/顺序错乱」一类回归。

// ---------------------------------------------------------------------------
// 编码辅助
// ---------------------------------------------------------------------------

/// 一条指令的首字：`(word_count << 16) | opcode`。
const fn inst(word_count: u32, opcode: u32) -> u32 {
    (word_count << 16) | opcode
}

/// 指令字长（首字高 16 位）。
const fn word_count(w: u32) -> u32 {
    w >> 16
}

/// 指令 OpCode（首字低 16 位）。
const fn opcode(w: u32) -> u32 {
    w & 0xFFFF
}

// OpCode（SPIR-V 1.0 公开规范）。
const OP_CAPABILITY: u32 = 17;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_F_MUL: u32 = 128;
const OP_LABEL: u32 = 248;
const OP_RETURN: u32 = 253;

/// OpCode 白名单（本模块产出的子集；出现表外 OpCode ⇒ 字节被改动）。
const ALLOWED_OPCODES: &[u32] = &[
    OP_CAPABILITY,
    OP_MEMORY_MODEL,
    OP_ENTRY_POINT,
    OP_EXECUTION_MODE,
    OP_TYPE_VOID,
    OP_TYPE_BOOL,
    OP_TYPE_INT,
    OP_TYPE_FLOAT,
    OP_TYPE_VECTOR,
    OP_TYPE_ARRAY,
    OP_TYPE_RUNTIME_ARRAY,
    OP_TYPE_STRUCT,
    OP_TYPE_POINTER,
    OP_TYPE_FUNCTION,
    OP_CONSTANT,
    OP_FUNCTION,
    OP_FUNCTION_END,
    OP_VARIABLE,
    OP_LOAD,
    OP_STORE,
    OP_ACCESS_CHAIN,
    OP_DECORATE,
    OP_MEMBER_DECORATE,
    OP_COMPOSITE_EXTRACT,
    OP_F_MUL,
    OP_LABEL,
    OP_RETURN,
];

/// OpCode 稳定短名（诊断/反汇编摘要用）。
pub fn opcode_name(op: u32) -> &'static str {
    match op {
        OP_CAPABILITY => "OpCapability",
        OP_MEMORY_MODEL => "OpMemoryModel",
        OP_ENTRY_POINT => "OpEntryPoint",
        OP_EXECUTION_MODE => "OpExecutionMode",
        OP_TYPE_VOID => "OpTypeVoid",
        OP_TYPE_BOOL => "OpTypeBool",
        OP_TYPE_INT => "OpTypeInt",
        OP_TYPE_FLOAT => "OpTypeFloat",
        OP_TYPE_VECTOR => "OpTypeVector",
        OP_TYPE_ARRAY => "OpTypeArray",
        OP_TYPE_RUNTIME_ARRAY => "OpTypeRuntimeArray",
        OP_TYPE_STRUCT => "OpTypeStruct",
        OP_TYPE_POINTER => "OpTypePointer",
        OP_TYPE_FUNCTION => "OpTypeFunction",
        OP_CONSTANT => "OpConstant",
        OP_FUNCTION => "OpFunction",
        OP_FUNCTION_END => "OpFunctionEnd",
        OP_VARIABLE => "OpVariable",
        OP_LOAD => "OpLoad",
        OP_STORE => "OpStore",
        OP_ACCESS_CHAIN => "OpAccessChain",
        OP_DECORATE => "OpDecorate",
        OP_MEMBER_DECORATE => "OpMemberDecorate",
        OP_COMPOSITE_EXTRACT => "OpCompositeExtract",
        OP_F_MUL => "OpFMul",
        OP_LABEL => "OpLabel",
        OP_RETURN => "OpReturn",
        _ => "OpUnknown",
    }
}

// ---------------------------------------------------------------------------
// magic / 版本 / 执行模型常量
// ---------------------------------------------------------------------------

/// SPIR-V magic（小端字节序 `0x03022307` 读成 `u32` 即此值）。
pub const MAGIC: u32 = 0x0723_0203;
/// 头部版本：SPIR-V 1.0，字 `0x00010000`。
pub const VERSION_1_0: u32 = 0x0001_0000;
/// 头部 schema（保留，必须 0）。
pub const SCHEMA: u32 = 0;
/// 执行模型：`GLCompute`。
pub const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
/// 执行模式：`LocalSize`。
pub const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
/// 存储类：`Input`。
pub const STORAGE_CLASS_INPUT: u32 = 1;
/// 存储类：`Uniform`（SPIR-V 1.0 的 SSBO 承载类，配合 `BufferBlock`）。
pub const STORAGE_CLASS_UNIFORM: u32 = 2;
/// 装饰：`BufferBlock`。
pub const DECORATION_BUFFER_BLOCK: u32 = 3;
/// 装饰：`Binding`。
pub const DECORATION_BINDING: u32 = 33;
/// 装饰：`DescriptorSet`。
pub const DECORATION_DESCRIPTOR_SET: u32 = 34;
/// 装饰：`Offset`。
pub const DECORATION_OFFSET: u32 = 35;

// ---------------------------------------------------------------------------
// 手写模块：spv_mul2_probe（`dst[i] = src[i] * 2.0`，LocalSize 64×1×1）
// ---------------------------------------------------------------------------
//
// 指令序严格按 SPIR-V 规范的 section 顺序：
//   ① capability → ② memory model → ③ entry point / execution mode →
//   ④ 注解（OpDecorate / OpMemberDecorate）→ ⑤ 类型/常量/全局变量 → ⑥ 函数。
//
// id 分配（bound = 26）：
//   1 %void  2 %fn_void_t  3 %f32  4 %u32  5 %v3u32
//   6 %ptr_Input_v3u32  7 %gl_GlobalInvocationID
//   8 %c0u32  9 %c2f32
//   10 %ptr_Uniform_f32  11 %rtarr_f32  12 %struct_src  13 %ptr_Uniform_struct_src
//   14 %src  15 %struct_dst  16 %ptr_Uniform_struct_dst  17 %dst
//   18 %main  19 %label  20 %gid  21 %i  22 %src_elem_ptr  23 %val
//   24 %doubled  25 %dst_elem_ptr
static MUL2: &[u32] = &[
    // --- 头部 ---
    MAGIC,
    VERSION_1_0,
    0,     // generator（0 = 自研手写，非工具生成 —— 如实标注）
    26,    // bound（max id + 1）
    SCHEMA,
    // ① OpCapability Shader
    inst(2, OP_CAPABILITY),
    1,
    // ② OpMemoryModel Logical GLSL450
    inst(3, OP_MEMORY_MODEL),
    0,
    1,
    // ③ OpEntryPoint GLCompute %18 "mul2" %7
    //    入口名 "mul2\0" 压入单字：'m'=0x6d 'u'=0x75 'l'=0x6c '2'=0x32（小端）。
    inst(5, OP_ENTRY_POINT),
    EXECUTION_MODEL_GLCOMPUTE,
    18,
    0x326C_756D, // "mul2\0"
    7,
    // ③ OpExecutionMode %18 LocalSize 64 1 1
    //    操作数：entry id + mode + 3 个分支尺寸 = 5 ⇒ 字长 6。
    inst(6, OP_EXECUTION_MODE),
    18,
    EXECUTION_MODE_LOCAL_SIZE,
    64,
    1,
    1,
    // ④ OpDecorate %12 BufferBlock
    inst(3, OP_DECORATE),
    12,
    DECORATION_BUFFER_BLOCK,
    // ④ OpMemberDecorate %12 0 Offset 0
    inst(5, OP_MEMBER_DECORATE),
    12,
    0,
    DECORATION_OFFSET,
    0,
    // ④ OpDecorate %14 DescriptorSet 0
    inst(4, OP_DECORATE),
    14,
    DECORATION_DESCRIPTOR_SET,
    0,
    // ④ OpDecorate %14 Binding 0
    inst(4, OP_DECORATE),
    14,
    DECORATION_BINDING,
    0,
    // ④ OpDecorate %15 BufferBlock
    inst(3, OP_DECORATE),
    15,
    DECORATION_BUFFER_BLOCK,
    // ④ OpMemberDecorate %15 0 Offset 0
    inst(5, OP_MEMBER_DECORATE),
    15,
    0,
    DECORATION_OFFSET,
    0,
    // ④ OpDecorate %17 DescriptorSet 0
    inst(4, OP_DECORATE),
    17,
    DECORATION_DESCRIPTOR_SET,
    0,
    // ④ OpDecorate %17 Binding 1
    inst(4, OP_DECORATE),
    17,
    DECORATION_BINDING,
    1,
    // ⑤ %1 = OpTypeVoid
    inst(2, OP_TYPE_VOID),
    1,
    // ⑤ %2 = OpTypeFunction %1
    inst(3, OP_TYPE_FUNCTION),
    2,
    1,
    // ⑤ %3 = OpTypeFloat 32
    inst(3, OP_TYPE_FLOAT),
    3,
    32,
    // ⑤ %4 = OpTypeInt 32 0
    inst(4, OP_TYPE_INT),
    4,
    32,
    0,
    // ⑤ %5 = OpTypeVector %4 3
    inst(4, OP_TYPE_VECTOR),
    5,
    4,
    3,
    // ⑤ %6 = OpTypePointer Input %5
    inst(4, OP_TYPE_POINTER),
    6,
    STORAGE_CLASS_INPUT,
    5,
    // ⑤ %10 = OpTypePointer Uniform %3
    inst(4, OP_TYPE_POINTER),
    10,
    STORAGE_CLASS_UNIFORM,
    3,
    // ⑤ %11 = OpTypeRuntimeArray %3
    inst(3, OP_TYPE_RUNTIME_ARRAY),
    11,
    3,
    // ⑤ %12 = OpTypeStruct %11
    inst(3, OP_TYPE_STRUCT),
    12,
    11,
    // ⑤ %13 = OpTypePointer Uniform %12
    inst(4, OP_TYPE_POINTER),
    13,
    STORAGE_CLASS_UNIFORM,
    12,
    // ⑤ %15 = OpTypeStruct %11
    inst(3, OP_TYPE_STRUCT),
    15,
    11,
    // ⑤ %16 = OpTypePointer Uniform %15
    inst(4, OP_TYPE_POINTER),
    16,
    STORAGE_CLASS_UNIFORM,
    15,
    // ⑤ %8 = OpConstant %4 0
    inst(4, OP_CONSTANT),
    4,
    8,
    0,
    // ⑤ %9 = OpConstant %3 2.0
    inst(4, OP_CONSTANT),
    3,
    9,
    0x4000_0000,
    // ⑤ %7 = OpVariable %6 Input
    inst(4, OP_VARIABLE),
    6,
    7,
    STORAGE_CLASS_INPUT,
    // ⑤ %14 = OpVariable %13 Uniform
    inst(4, OP_VARIABLE),
    13,
    14,
    STORAGE_CLASS_UNIFORM,
    // ⑤ %17 = OpVariable %16 Uniform
    inst(4, OP_VARIABLE),
    16,
    17,
    STORAGE_CLASS_UNIFORM,
    // ⑥ %18 = OpFunction %1 None %2
    inst(5, OP_FUNCTION),
    1,
    18,
    0,
    2,
    // ⑥ %19 = OpLabel
    inst(2, OP_LABEL),
    19,
    // ⑥ %20 = OpLoad %5 %7
    inst(4, OP_LOAD),
    5,
    20,
    7,
    // ⑥ %21 = OpCompositeExtract %4 %20 0
    inst(5, OP_COMPOSITE_EXTRACT),
    4,
    21,
    20,
    0,
    // ⑥ %22 = OpAccessChain %10 %14 %8 %21
    inst(6, OP_ACCESS_CHAIN),
    10,
    22,
    14,
    8,
    21,
    // ⑥ %23 = OpLoad %3 %22
    inst(4, OP_LOAD),
    3,
    23,
    22,
    // ⑥ %24 = OpFMul %3 %23 %9
    inst(5, OP_F_MUL),
    3,
    24,
    23,
    9,
    // ⑥ %25 = OpAccessChain %10 %17 %8 %21
    inst(6, OP_ACCESS_CHAIN),
    10,
    25,
    17,
    8,
    21,
    // ⑥ OpStore %25 %24
    inst(3, OP_STORE),
    25,
    24,
    // ⑥ OpReturn（无操作数）
    inst(1, OP_RETURN),
    // ⑥ OpFunctionEnd（无操作数）
    inst(1, OP_FUNCTION_END),
];

/// 手写模块注册表：模块名 → SPIR-V 字流（小端 `u32` 序）。
///
/// `static`（非 `const`）：`const` 会在每个引用点物化独立副本，导致
/// 「同一模块在不同调用点拿到不同地址」——与 `crate::shaders::SHADERS` 同一
/// 教训（wgpu/FFI 传指针时须全程序唯一实例）。
pub static MODULES: &[(&str, &[u32])] = &[("spv_mul2_probe", MUL2)];

/// 按名取模块字流。
pub fn module(name: &str) -> Option<&'static [u32]> {
    MODULES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, w)| *w)
}

/// 模块名全集（确定性顺序）。
pub fn module_names() -> Vec<&'static str> {
    MODULES.iter().map(|(n, _)| *n).collect()
}

// ---------------------------------------------------------------------------
// CPU 参考语义（二进制宣称行为的可执行对照）
// ---------------------------------------------------------------------------

/// `spv_mul2_probe` 的 CPU 参考语义：`dst[i] = src[i] * 2.0`（长度取二者较小值）。
///
/// 二进制的语义主张由本函数锚定；本环境无法执行 SPIR-V，故测试断言的是
/// 「结构校验通过 + 语义描述与参考一致」（不假装跑过 GPU）。
pub fn cpu_reference_mul2(src: &[f32], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = src[i] * 2.0;
    }
}

// ---------------------------------------------------------------------------
// 结构验证
// ---------------------------------------------------------------------------

/// 验证期望（入口名 + 工作组尺寸）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpvExpect {
    /// 期望入口名（`OpEntryPoint` 的名字）。
    pub entry: &'static str,
    /// 期望 `LocalSize`。
    pub local_size: [u32; 3],
    /// 期望的描述符集合（`DescriptorSet` 均为 0）的 binding 号集合。
    pub bindings: &'static [u32],
}

/// 结构验证失败原因（封闭枚举，便于测试逐项断言）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpvError {
    /// 字流过短（不足 5 字头部）。
    TooShort,
    /// magic 不符（字节被改/错序/不是 SPIR-V）。
    BadMagic {
        /// 实际读到的值。
        got: u32,
    },
    /// 版本不受支持。
    BadVersion {
        /// 实际读到的值。
        got: u32,
    },
    /// schema 非 0。
    BadSchema {
        /// 实际读到的值。
        got: u32,
    },
    /// bound 为 0。
    BadBound,
    /// 指令字长为 0（首字被破坏）。
    ZeroWordCount {
        /// 指令起始字下标。
        at: usize,
    },
    /// 指令越界（尾部被截断）。
    Truncated {
        /// 指令起始字下标。
        at: usize,
    },
    /// OpCode 不在白名单。
    UnknownOpcode {
        /// 实际 OpCode。
        got: u32,
        /// 指令起始字下标。
        at: usize,
    },
    /// 缺 `OpEntryPoint`。
    MissingEntryPoint,
    /// 入口名不符。
    EntryNameMismatch {
        /// 期望名。
        expected: &'static str,
        /// 实际名（`OpEntryPoint` 里读到的）。
        got: String,
    },
    /// 缺 `OpExecutionMode LocalSize`。
    MissingLocalSize,
    /// `LocalSize` 不符。
    LocalSizeMismatch {
        /// 期望值。
        expected: [u32; 3],
        /// 实际值。
        got: [u32; 3],
    },
    /// 描述符 binding 集合不符。
    BindingMismatch {
        /// 期望集合。
        expected: Vec<u32>,
        /// 实际集合。
        got: Vec<u32>,
    },
    /// 函数段结构不完整（无 OpFunctionEnd 或函数数≠1）。
    BadFunctionSection {
        /// 实际函数数。
        functions: u32,
    },
}

impl SpvError {
    /// 稳定短名（诊断日志契约）。
    pub fn as_str(&self) -> &'static str {
        match self {
            SpvError::TooShort => "too-short",
            SpvError::BadMagic { .. } => "bad-magic",
            SpvError::BadVersion { .. } => "bad-version",
            SpvError::BadSchema { .. } => "bad-schema",
            SpvError::BadBound => "bad-bound",
            SpvError::ZeroWordCount { .. } => "zero-word-count",
            SpvError::Truncated { .. } => "truncated",
            SpvError::UnknownOpcode { .. } => "unknown-opcode",
            SpvError::MissingEntryPoint => "missing-entry-point",
            SpvError::EntryNameMismatch { .. } => "entry-name-mismatch",
            SpvError::MissingLocalSize => "missing-local-size",
            SpvError::LocalSizeMismatch { .. } => "local-size-mismatch",
            SpvError::BindingMismatch { .. } => "binding-mismatch",
            SpvError::BadFunctionSection { .. } => "bad-function-section",
        }
    }
}

/// 结构验证摘要（成功时给出，供诊断打印与测试断言）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpvSummary {
    /// 版本字。
    pub version: u32,
    /// bound。
    pub bound: u32,
    /// 指令条数（不含 5 字头部）。
    pub instruction_count: usize,
    /// 入口名。
    pub entry: String,
    /// 执行模式 `LocalSize`。
    pub local_size: [u32; 3],
    /// 描述符 binding 号集合（升序去重）。
    pub bindings: Vec<u32>,
    /// 函数个数（`OpFunction` 计数）。
    pub functions: u32,
    /// 出现的 OpCode 集合（升序去重）。
    pub opcodes: Vec<u32>,
}

/// 结构验证：`words` 是完整模块（含 5 字头部）。
///
/// 只做**结构**校验（不做类型/语义验算——那需 `spirv-val`，本环境无）。
pub fn validate(words: &[u32], expect: &SpvExpect) -> Result<SpvSummary, SpvError> {
    if words.len() < 5 {
        return Err(SpvError::TooShort);
    }
    if words[0] != MAGIC {
        return Err(SpvError::BadMagic { got: words[0] });
    }
    if words[1] != VERSION_1_0 {
        return Err(SpvError::BadVersion { got: words[1] });
    }
    if words[4] != SCHEMA {
        return Err(SpvError::BadSchema { got: words[4] });
    }
    let bound = words[3];
    if bound == 0 {
        return Err(SpvError::BadBound);
    }

    let mut i = 5usize;
    let mut instruction_count = 0usize;
    let mut entry: Option<String> = None;
    let mut local_size: Option<[u32; 3]> = None;
    let mut bindings: Vec<u32> = Vec::new();
    let mut functions = 0u32;
    let mut opcodes: Vec<u32> = Vec::new();
    while i < words.len() {
        let first = words[i];
        let wc = word_count(first) as usize;
        let op = opcode(first);
        if wc == 0 {
            return Err(SpvError::ZeroWordCount { at: i });
        }
        if i + wc > words.len() {
            return Err(SpvError::Truncated { at: i });
        }
        if !ALLOWED_OPCODES.contains(&op) {
            return Err(SpvError::UnknownOpcode { got: op, at: i });
        }
        opcodes.push(op);
        instruction_count += 1;
        match op {
            OP_ENTRY_POINT => {
                // [0]=首字 [1]=执行模型 [2]=id [3..]="name\0" 字 [n-1..]=interface ids
                // 名字可能占多个字；本模块的名字单字即可（"main\0"）。
                if wc >= 5 {
                    let mut name_bytes: Vec<u8> = Vec::new();
                    for &w in &words[i + 3..i + wc - 1] {
                        name_bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    while name_bytes.last() == Some(&0) {
                        name_bytes.pop();
                    }
                    entry = Some(String::from_utf8_lossy(&name_bytes).into_owned());
                }
            }
            OP_EXECUTION_MODE => {
                if wc >= 6 && words[i + 2] == EXECUTION_MODE_LOCAL_SIZE {
                    local_size = Some([words[i + 3], words[i + 4], words[i + 5]]);
                }
            }
            OP_DECORATE => {
                if wc >= 4 && words[i + 2] == DECORATION_BINDING {
                    bindings.push(words[i + 3]);
                }
            }
            OP_FUNCTION => functions += 1,
            _ => {}
        }
        i += wc;
    }

    let Some(entry) = entry else {
        return Err(SpvError::MissingEntryPoint);
    };
    if entry != expect.entry {
        return Err(SpvError::EntryNameMismatch {
            expected: expect.entry,
            got: entry,
        });
    }
    let Some(ls) = local_size else {
        return Err(SpvError::MissingLocalSize);
    };
    if ls != expect.local_size {
        return Err(SpvError::LocalSizeMismatch {
            expected: expect.local_size,
            got: ls,
        });
    }
    bindings.sort_unstable();
    bindings.dedup();
    let mut expected_bindings = expect.bindings.to_vec();
    expected_bindings.sort_unstable();
    expected_bindings.dedup();
    if bindings != expected_bindings {
        return Err(SpvError::BindingMismatch {
            expected: expected_bindings,
            got: bindings,
        });
    }
    if functions != 1 {
        return Err(SpvError::BadFunctionSection { functions });
    }
    opcodes.sort_unstable();
    opcodes.dedup();

    Ok(SpvSummary {
        version: words[1],
        bound,
        instruction_count,
        entry,
        local_size: ls,
        bindings,
        functions,
        opcodes,
    })
}

/// 反汇编摘要（诊断用：每行 `[下标] OpName (wc)`）。
///
/// 不是完整反汇编（不展开操作数语义），只用于「字节被改动」时快速定位。
pub fn disassemble(words: &[u32]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "; spirv magic=0x{:08x} version=0x{:08x} generator=0x{:08x} bound={} schema={}\n",
        words.first().copied().unwrap_or(0),
        words.get(1).copied().unwrap_or(0),
        words.get(2).copied().unwrap_or(0),
        words.get(3).copied().unwrap_or(0),
        words.get(4).copied().unwrap_or(0),
    ));
    let mut i = 5usize;
    while i < words.len() {
        let wc = word_count(words[i]) as usize;
        if wc == 0 {
            out.push_str(&format!("[{i}] <zero word-count>\n"));
            break;
        }
        let end = (i + wc).min(words.len());
        let ops: Vec<String> = words[i + 1..end].iter().map(|w| w.to_string()).collect();
        out.push_str(&format!(
            "[{i}] {} ({wc}) {}\n",
            opcode_name(opcode(words[i])),
            ops.join(" ")
        ));
        i = end;
    }
    out
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ENTRY_SPV_MUL2, KERNEL_SPV_MUL2_PROBE};

    fn mul2_expect() -> SpvExpect {
        SpvExpect {
            entry: ENTRY_SPV_MUL2,
            local_size: [64, 1, 1],
            bindings: &[0, 1],
        }
    }

    /// 手写字节码结构自洽：magic/版本/bound/指令字长/入口/LocalSize/绑定。
    #[test]
    fn handwritten_module_is_structurally_valid() {
        let words = module(KERNEL_SPV_MUL2_PROBE).expect("探针模块已注册");
        let sum = validate(words, &mul2_expect()).expect("手写 SPIR-V 应通过结构验证");
        assert_eq!(sum.entry, "mul2", "入口名须与 KernelDesc.entry 一致");
        assert_eq!(sum.local_size, [64, 1, 1]);
        assert_eq!(sum.bindings, vec![0, 1]);
        assert_eq!(sum.functions, 1, "模块应恰有一个函数");
        assert_eq!(sum.bound, 26);
        // 关键指令必须在场（防误删）。
        for op in [
            OP_CAPABILITY,
            OP_ENTRY_POINT,
            OP_EXECUTION_MODE,
            OP_F_MUL,
            OP_ACCESS_CHAIN,
            OP_FUNCTION_END,
        ] {
            assert!(sum.opcodes.contains(&op), "缺指令 {}", opcode_name(op));
        }
        // 字流总量自洽：5 字头 + 各指令字长之和 == 数组长度。
        let mut i = 5usize;
        let mut total = 0usize;
        while i < words.len() {
            let wc = word_count(words[i]) as usize;
            assert!(wc > 0);
            total += wc;
            i += wc;
        }
        assert_eq!(5 + total, words.len(), "指令字长之和须铺满整个字流");
        println!(
            "SPIR-V 结构验证通过：{} 字，{} 条指令，bound={}",
            words.len(),
            sum.instruction_count,
            sum.bound
        );
    }

    /// 探针模块与 `KernelDesc` 的工作组尺寸/绑定一一对应（跨模块口径一致）。
    #[test]
    fn module_matches_kernel_desc() {
        let kernel = crate::compute::spirv_probe_kernel();
        let words = module(kernel.spv_module).expect("KernelDesc 引用的模块必须存在");
        let expect = SpvExpect {
            entry: kernel.entry,
            local_size: [kernel.workgroup.x, kernel.workgroup.y, kernel.workgroup.z],
            bindings: &[0, 1],
        };
        let sum = validate(words, &expect).expect("KernelDesc 与字节码须一致");
        assert_eq!(sum.bindings.len(), kernel.buffers.len());
        assert_eq!(
            sum.local_size,
            [
                kernel.workgroup.x,
                kernel.workgroup.y,
                kernel.workgroup.z
            ]
        );
    }

    /// 语义锚定：二进制宣称 `dst = src * 2`，CPU 参考必须同式。
    #[test]
    fn cpu_reference_matches_declared_semantics() {
        let src = [0.0f32, 1.0, -2.5, 1024.0, 0.25];
        let mut dst = [0.0f32; 5];
        cpu_reference_mul2(&src, &mut dst);
        for i in 0..src.len() {
            assert_eq!(dst[i], src[i] * 2.0);
        }
        // 长度不等时按较小值截断（不越界、不 panic）。
        let mut short = vec![0.0f32; 3];
        cpu_reference_mul2(&src, &mut short);
        assert_eq!(short, vec![0.0f32, 2.0, -5.0]);
    }

    /// 结构验证抓得住真实缺陷（负向用例，非恒真）。
    #[test]
    fn validator_catches_corruptions() {
        let good = module(KERNEL_SPV_MUL2_PROBE).expect("模块存在");

        // ① 截断。
        assert_eq!(
            validate(&good[..4], &mul2_expect()),
            Err(SpvError::TooShort)
        );
        let truncated = &good[..good.len() - 3];
        assert!(matches!(
            validate(truncated, &mul2_expect()),
            Err(SpvError::Truncated { .. })
        ));

        // ② magic 被改。
        let mut bad = good.to_vec();
        bad[0] = 0xDEAD_BEEF;
        assert_eq!(
            validate(&bad, &mul2_expect()),
            Err(SpvError::BadMagic { got: 0xDEAD_BEEF })
        );

        // ③ schema 非 0。
        let mut bad = good.to_vec();
        bad[4] = 7;
        assert_eq!(
            validate(&bad, &mul2_expect()),
            Err(SpvError::BadSchema { got: 7 })
        );

        // ④ 指令首字被清零（字长 0）。
        let mut bad = good.to_vec();
        bad[5] = 0;
        assert!(matches!(
            validate(&bad, &mul2_expect()),
            Err(SpvError::ZeroWordCount { at: 5 })
        ));

        // ⑤ OpCode 被换成白名单外的值。
        let mut bad = good.to_vec();
        bad[5] = inst(2, 0x0FFF);
        assert!(matches!(
            validate(&bad, &mul2_expect()),
            Err(SpvError::UnknownOpcode { got: 0x0FFF, .. })
        ));

        // ⑥ 期望入口名不符（模拟「模块与 KernelDesc 错配」）。
        let wrong = SpvExpect {
            entry: "not_main",
            local_size: [64, 1, 1],
            bindings: &[0, 1],
        };
        assert!(matches!(
            validate(good, &wrong),
            Err(SpvError::EntryNameMismatch { .. })
        ));

        // ⑦ LocalSize 不符（模拟 workgroup 与字节码不一致的真机坑）。
        let wrong_ls = SpvExpect {
            entry: ENTRY_SPV_MUL2,
            local_size: [128, 1, 1],
            bindings: &[0, 1],
        };
        assert!(matches!(
            validate(good, &wrong_ls),
            Err(SpvError::LocalSizeMismatch { .. })
        ));

        // ⑧ 绑定集合不符（模拟描述符布局与字节码错配）。
        let wrong_b = SpvExpect {
            entry: ENTRY_SPV_MUL2,
            local_size: [64, 1, 1],
            bindings: &[0],
        };
        assert!(matches!(
            validate(good, &wrong_b),
            Err(SpvError::BindingMismatch { .. })
        ));

        // 失败原因短名非空。
        for e in [
            SpvError::TooShort,
            SpvError::BadMagic { got: 0 },
            SpvError::BadVersion { got: 0 },
            SpvError::BadSchema { got: 1 },
            SpvError::BadBound,
            SpvError::ZeroWordCount { at: 0 },
            SpvError::Truncated { at: 0 },
            SpvError::UnknownOpcode { got: 0, at: 0 },
            SpvError::MissingEntryPoint,
            SpvError::MissingLocalSize,
            SpvError::BadFunctionSection { functions: 0 },
        ] {
            assert!(!e.as_str().is_empty(), "失败原因短名不得为空");
        }
    }

    /// 反汇编摘要可读且稳定（诊断用）。
    #[test]
    fn disassembly_is_stable_and_covers_all_instructions() {
        let words = module(KERNEL_SPV_MUL2_PROBE).expect("模块存在");
        let text = disassemble(words);
        assert!(text.contains("OpCapability"));
        assert!(text.contains("OpEntryPoint"));
        assert!(text.contains("OpFMul"));
        assert!(text.contains("OpFunctionEnd"));
        assert!(!text.contains("OpUnknown"), "不得出现未知 OpCode");
        assert_eq!(text, disassemble(words), "反汇编应确定性");
        // 每行都以 `[下标]` 开头（除首行注）。
        for line in text.lines().skip(1) {
            assert!(line.starts_with('['), "行格式异常：{line}");
        }
    }

    /// 注册表查询契约。
    #[test]
    fn module_lookup_contract() {
        assert!(module("spv_mul2_probe").is_some());
        assert!(module("no_such_module").is_none());
        assert_eq!(module_names(), vec!["spv_mul2_probe"]);
        // 只声明了唯一真实二进制 —— 不虚报 RT kernel 已有 SPIR-V。
        assert_eq!(MODULES.len(), 1);
    }
}
