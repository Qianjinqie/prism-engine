//! 手写最小 SPIR-V 字节码（无 glslang 工具链时的真实二进制供给）。
//!
//! # 为什么手写（如实记录做法）
//!
//! 正常路径是「WGSL → SPIR-V 由自研发射器生成（`crate::metal::msl` 头注的
//! 定案）」，构建期用 `glslangValidator`/`glslc` 产出 `.spv` 再 `include_bytes!`
//! 打进二进制。构建环境未提供 `glslangValidator`/`glslc` 时，本模块
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
//! 参考」——有 lavapipe（llvmpipe）软件 ICD 时**可真实执行**该 SPIR-V，
//! 结构 + 语义锚定与设备读回三重校验。）
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

/// 指令首字：`(word_count << 16) | opcode`。
pub const fn inst(word_count: u32, opcode: u32) -> u32 {
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
pub const OP_CAPABILITY: u32 = 17;
pub const OP_MEMORY_MODEL: u32 = 14;
pub const OP_ENTRY_POINT: u32 = 15;
pub const OP_EXECUTION_MODE: u32 = 16;
/// `OpExtInstImport`（导入扩展指令集，如 `GLSL.std.450`）。
pub const OP_EXT_INST_IMPORT: u32 = 11;
/// `OpExtInst`（扩展指令调用）。
pub const OP_EXT_INST: u32 = 12;
pub const OP_TYPE_VOID: u32 = 19;
pub const OP_TYPE_BOOL: u32 = 20;
pub const OP_TYPE_INT: u32 = 21;
pub const OP_TYPE_FLOAT: u32 = 22;
pub const OP_TYPE_VECTOR: u32 = 23;
pub const OP_TYPE_ARRAY: u32 = 28;
pub const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
pub const OP_TYPE_STRUCT: u32 = 30;
pub const OP_TYPE_POINTER: u32 = 32;
pub const OP_TYPE_FUNCTION: u32 = 33;
pub const OP_CONSTANT: u32 = 43;
pub const OP_FUNCTION: u32 = 54;
pub const OP_FUNCTION_END: u32 = 56;
pub const OP_VARIABLE: u32 = 59;
pub const OP_LOAD: u32 = 61;
pub const OP_STORE: u32 = 62;
pub const OP_ACCESS_CHAIN: u32 = 65;
pub const OP_DECORATE: u32 = 71;
pub const OP_MEMBER_DECORATE: u32 = 72;
pub const OP_COMPOSITE_EXTRACT: u32 = 81;
/// `OpConvertFToU`（浮点 → 无符号整；**向零截断**，越界为 UB，
/// 故调用方必须先把值钳进 `[0, u32::MAX]`）。
pub const OP_CONVERT_F_TO_U: u32 = 109;
/// `OpConvertUToF`（无符号整 → 浮点；就近舍入到偶数）。
pub const OP_CONVERT_U_TO_F: u32 = 112;
/// `OpBitcast`（位型重解释；f32 ↔ u32 零开销，无舍入）。
pub const OP_BITCAST: u32 = 124;
/// `OpFNegate`（浮点取负；**逐位等于** Rust 的一元 `-x`，含 ±0 与 NaN 符号位）。
///
/// 光追 kernel 的 `-b - sqrt(disc)` 若写成 `0.0 - b - sqrt` 会在 `b == ±0.0`
/// 时改动零的符号（`0.0 - 0.0 = +0.0`，而 `-0.0` 是 `-0.0`），故显式取负。
pub const OP_F_NEGATE: u32 = 127;
/// `OpIAdd`（整数加，模 2^32 回绕）。
pub const OP_I_ADD: u32 = 128;
/// 指令 `OpFAdd`（浮点加）。
pub const OP_F_ADD: u32 = 129;
/// 指令 `OpFSub`（浮点减）。
pub const OP_F_SUB: u32 = 131;
/// 指令 `OpIMul`（整数乘，模 2^32 回绕）。
pub const OP_I_MUL: u32 = 132;
/// 指令 `OpFMul`（浮点乘）。
///
/// lavapipe 实测修正：旧值 128 实为 `OpIAdd`。算术指令段自 128 起
/// 按「整/浮」交替编号：128 OpIAdd / 129 OpFAdd / 130 OpISub / 131 OpFSub /
/// 132 OpIMul / **133 OpFMul**——错位使 `mul2` kernel 语义改变且结果类型
/// 非法（spirv-val: "Expected int scalar or vector type as Result Type"）。
pub const OP_F_MUL: u32 = 133;
/// 指令 `OpFDiv`（浮点除；IEEE 正确舍入）。
pub const OP_F_DIV: u32 = 136;
/// `OpLogicalOr`。
pub const OP_LOGICAL_OR: u32 = 166;
/// `OpLogicalAnd`（布尔与；组合「disc ≥ 0 ∧ t > 0 ∧ t < closest」三重判据）。
pub const OP_LOGICAL_AND: u32 = 167;
/// `OpLogicalNot`（布尔取反）。
/// `OpSelect`（条件选择；无分支，故不需要结构化控制流）。
pub const OP_SELECT: u32 = 169;
/// `OpUGreaterThanEqual`（无符号 ≥）。
pub const OP_U_GREATER_THAN_EQUAL: u32 = 174;
/// `OpULessThan`（无符号 <；用于 ray/sphere 与 light 动态循环的上界判断、
/// 以及「命中下标 < 球数」= 是否命中的判据）。
///
/// ⚠️ **编号勘误（V2.1 T-A1）**：上一轮中断施工遗留的值 `170` 实为
/// `OpIEqual`（整数相等），用它会得到「球数 == 循环下标」这一**完全不同**的
/// 谓词——类型仍合法（u32 比 u32），驱动不报错，只会静默算错图。
/// 本值经 `spirv-as`/`spirv-dis`（SPIR-V 1.0 官方工具链）实测复核为 `176`。
pub const OP_U_LESS_THAN: u32 = 176;
/// `OpINotEqual`（整数 ≠；用于把 u32 标志位（0/1）转成布尔谓词）。
pub const OP_I_NOT_EQUAL: u32 = 171;
/// `OpFOrdLessThan`（有序浮点 <；NaN 为假）。
pub const OP_F_ORD_LESS_THAN: u32 = 184;
/// `OpFOrdGreaterThan`（有序浮点 >；NaN 为假）。
pub const OP_F_ORD_GREATER_THAN: u32 = 186;
/// `OpFOrdGreaterThanEqual`（有序浮点 ≥；NaN 为假）。
///
/// 与 Rust 的 `a >= b` 语义逐位一致（NaN 参与时两边都为 `false`）——这是
/// 光追 kernel「`disc >= 0` 才算相交」与 CPU 参考对拍的前提。
pub const OP_F_ORD_GREATER_THAN_EQUAL: u32 = 190;
/// `OpShiftRightLogical`（逻辑右移，高位补 0）。
pub const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
/// `OpShiftLeftLogical`。
pub const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
/// `OpBitwiseOr`。
pub const OP_BITWISE_OR: u32 = 197;
/// `OpBitwiseXor`。
pub const OP_BITWISE_XOR: u32 = 198;
/// `OpSelectionMerge`（结构化选择头；**必须紧邻**其后的分支指令）。
pub const OP_SELECTION_MERGE: u32 = 247;
/// `OpLoopMerge`（结构化循环头；**必须紧邻**其后的分支指令）。
///
/// 光追 megakernel 用它对球体 / 光源做**动态循环**（循环次数 = 实体个数，
/// 运行时取自参数块，不能展开），是背景 kernel（纯逐点、无循环）之外的第二
/// 种结构化控制流。lavapipe 实机真实执行。
pub const OP_LOOP_MERGE: u32 = 246;
/// `OpLogicalNot`（布尔取反；把 `a >= 0` 类的判据改写成 `!(a < 0)` 时用）。
///
/// ⚠️ **编号勘误（V2.1 T-A1）**：上一轮中断施工遗留的值 `167` 实为
/// `OpLogicalAnd`——把 `!(a < b)` 发射成 `a && b` 是**类型合法但语义全错**的
/// 指令（两者都接两个 bool），驱动不会报错，只会静默算错。本值经
/// `spirv-as`/`spirv-dis`（SPIR-V 1.0 官方工具链）实测复核为 `168`。
pub const OP_LOGICAL_NOT: u32 = 168;
/// `OpBranch`。
pub const OP_BRANCH: u32 = 249;
/// `OpBranchConditional`。
pub const OP_BRANCH_CONDITIONAL: u32 = 250;
pub const OP_LABEL: u32 = 248;
pub const OP_RETURN: u32 = 253;

/// OpCode 白名单（本模块产出的子集；出现表外 OpCode ⇒ 字节被改动）。
const ALLOWED_OPCODES: &[u32] = &[
    OP_CAPABILITY,
    OP_MEMORY_MODEL,
    OP_ENTRY_POINT,
    OP_EXECUTION_MODE,
    OP_EXT_INST_IMPORT,
    OP_EXT_INST,
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
    OP_CONVERT_F_TO_U,
    OP_CONVERT_U_TO_F,
    OP_BITCAST,
    OP_F_NEGATE,
    OP_I_ADD,
    OP_F_ADD,
    OP_F_SUB,
    OP_I_MUL,
    OP_F_MUL,
    OP_F_DIV,
    OP_LOGICAL_OR,
    OP_LOGICAL_AND,
    OP_LOGICAL_NOT,
    OP_SELECT,
    OP_U_GREATER_THAN_EQUAL,
    OP_U_LESS_THAN,
    OP_I_NOT_EQUAL,
    OP_F_ORD_LESS_THAN,
    OP_F_ORD_GREATER_THAN,
    OP_F_ORD_GREATER_THAN_EQUAL,
    OP_SHIFT_RIGHT_LOGICAL,
    OP_SHIFT_LEFT_LOGICAL,
    OP_BITWISE_OR,
    OP_BITWISE_XOR,
    OP_SELECTION_MERGE,
    OP_LOOP_MERGE,
    OP_LABEL,
    OP_BRANCH,
    OP_BRANCH_CONDITIONAL,
    OP_RETURN,
];

/// OpCode 稳定短名（诊断/反汇编摘要用）。
pub fn opcode_name(op: u32) -> &'static str {
    match op {
        OP_CAPABILITY => "OpCapability",
        OP_MEMORY_MODEL => "OpMemoryModel",
        OP_ENTRY_POINT => "OpEntryPoint",
        OP_EXECUTION_MODE => "OpExecutionMode",
        OP_EXT_INST_IMPORT => "OpExtInstImport",
        OP_EXT_INST => "OpExtInst",
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
        OP_CONVERT_F_TO_U => "OpConvertFToU",
        OP_CONVERT_U_TO_F => "OpConvertUToF",
        OP_BITCAST => "OpBitcast",
        OP_F_NEGATE => "OpFNegate",
        OP_I_ADD => "OpIAdd",
        OP_F_ADD => "OpFAdd",
        OP_F_SUB => "OpFSub",
        OP_I_MUL => "OpIMul",
        OP_F_MUL => "OpFMul",
        OP_F_DIV => "OpFDiv",
        OP_LOGICAL_OR => "OpLogicalOr",
        OP_LOGICAL_AND => "OpLogicalAnd",
        OP_LOGICAL_NOT => "OpLogicalNot",
        OP_SELECT => "OpSelect",
        OP_U_GREATER_THAN_EQUAL => "OpUGreaterThanEqual",
        OP_U_LESS_THAN => "OpULessThan",
        OP_I_NOT_EQUAL => "OpINotEqual",
        OP_F_ORD_LESS_THAN => "OpFOrdLessThan",
        OP_F_ORD_GREATER_THAN => "OpFOrdGreaterThan",
        OP_F_ORD_GREATER_THAN_EQUAL => "OpFOrdGreaterThanEqual",
        OP_SHIFT_RIGHT_LOGICAL => "OpShiftRightLogical",
        OP_SHIFT_LEFT_LOGICAL => "OpShiftLeftLogical",
        OP_BITWISE_OR => "OpBitwiseOr",
        OP_BITWISE_XOR => "OpBitwiseXor",
        OP_SELECTION_MERGE => "OpSelectionMerge",
        OP_LOOP_MERGE => "OpLoopMerge",
        OP_LABEL => "OpLabel",
        OP_BRANCH => "OpBranch",
        OP_BRANCH_CONDITIONAL => "OpBranchConditional",
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
/// 存储类：`Uniform`（UBO 承载类；Vulkan SSBO 不用它，见 `STORAGE_CLASS_STORAGE_BUFFER`）。
pub const STORAGE_CLASS_UNIFORM: u32 = 2;
/// 存储类：`StorageBuffer`（Vulkan 存储缓冲（SSBO）承载类；与
/// `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` 对应，可读写。修正：
/// 旧版误用 `Uniform`+`BufferBlock`（OpenGL 口味）导致 lavapipe
/// `vkCreateComputePipelines` 返回 `VK_ERROR_INVALID_SHADER_NV`）。
pub const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
/// 存储类：`Function`（函数局部变量）。
///
/// 光追 kernel 的循环携带值（最近命中距离、命中下标、循环计数器、颜色累加器）
/// 用 `OpVariable %ptr_Function_*` + `OpStore`/`OpLoad` 存取，**刻意不用
/// `OpPhi`**：不需要为每个循环头维护前驱列表（少一整类「前驱写错 ⇒ 驱动崩溃」
/// 的缺陷面），且与 CPU 参考的「可变局部量」逐句对应，可读性更好。
/// 函数局部变量必须位于函数**首块**且在所有其它指令之前（SPIR-V 规范）。
pub const STORAGE_CLASS_FUNCTION: u32 = 7;
/// 装饰：`Block`（Vulkan SSBO 结构体装饰；配对 `StorageBuffer` 存储类）。
pub const DECORATION_BLOCK: u32 = 2;
/// 装饰：`BufferBlock`（OpenGL 口味 UBO/SSBO 装饰；Vulkan 路径不用）。
pub const DECORATION_BUFFER_BLOCK: u32 = 3;
/// 装饰：`Binding`。
pub const DECORATION_BINDING: u32 = 33;
/// 装饰：`DescriptorSet`。
pub const DECORATION_DESCRIPTOR_SET: u32 = 34;
/// 装饰：`Offset`。
pub const DECORATION_OFFSET: u32 = 35;
/// 装饰：`NoContraction`（禁止把该浮点结果与相邻运算熔成 FMA）。
///
/// GPU 驱动**默认允许**收缩（Vulkan 规范），而 CPU 参考不会收缩——
/// 一旦熔成 FMA，末位就与参考不同。逐位对拍要求下必须显式禁止。
pub const DECORATION_NO_CONTRACTION: u32 = 42;
/// 装饰：`ArrayStride`（BufferBlock 内数组/运行期数组的显式步长——
/// Vulkan 消费方要求 Block 内数组必须显式声明步长，否则管线编译拒绝）。
pub const DECORATION_ARRAY_STRIDE: u32 = 6;
/// 装饰：`BuiltIn`（Input 存储类变量在 Vulkan compute 中**必须**以
/// BuiltIn 注解声明语义，缺此装饰 ⇒ 模块非法，`vkCreateComputePipelines`
/// 编译期拒绝——lavapipe 实测教训）。
pub const DECORATION_BUILT_IN: u32 = 11;
/// BuiltIn 语义：`GlobalInvocationId`。
pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

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
//
// SPIR-V 1.0 / Vulkan 1.0 口径：SSBO = `Uniform` 存储类 + `BufferBlock` 装饰
// （`Block` 才是 UBO；`StorageBuffer` 存储类是 SPIR-V 1.1+ 口味，本模块
// 钉 1.0 不用）。lavapipe 实测补齐了三处此前缺失的强制件：
// OpEntryPoint 入口名的 null 终结字、Input 变量的 BuiltIn 注解、
// BufferBlock 内运行期数组的 ArrayStride。
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
    //    入口名 "mul2\0" 占**两个字**：'m'=0x6d 'u'=0x75 'l'=0x6c '2'=0x32
    //    （小端）压首字，次字全零作 null 终结——少这个零字时入口名会
    //    "越界"吞掉接口表首 id（0x07）的字节，驱动侧查不到 entry point
    //    （lavapipe 实测：VUID-VkPipelineShaderStageCreateInfo-
    //    pName-00707 "entrypoint not found"）。
    //    操作数：model + entry id + 名 2 字 + 接口 1 id = 5 ⇒ 字长 6。
    inst(6, OP_ENTRY_POINT),
    EXECUTION_MODEL_GLCOMPUTE,
    18,
    0x326C_756D, // "mul2"
    0x0000_0000, // "\0" 终结 + 补齐
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
    // ④ OpDecorate %11 ArrayStride 4（BufferBlock 内运行期数组的显式步长；
    //    Vulkan 消费方对 Block 内数组强制要求 ArrayStride，缺它管线编译拒绝）
    inst(4, OP_DECORATE),
    11,
    DECORATION_ARRAY_STRIDE,
    4,
    // ④ OpDecorate %7 BuiltIn GlobalInvocationId（Vulkan compute 的 Input
    //    变量必须 BuiltIn 注解；缺此装饰 ⇒ 模块非法，lavapipe 实测拒绝建管线）
    inst(4, OP_DECORATE),
    7,
    DECORATION_BUILT_IN,
    BUILTIN_GLOBAL_INVOCATION_ID,
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
///
/// 本表只放**静态字面字流**；由发射器生成的模块（[`super::spirv_bg`] 的
/// 背景 kernel）走 [`generated_modules`]，两表由 [`module`] 统一寻址。
pub static MODULES: &[(&str, &[u32])] = &[("spv_mul2_probe", MUL2)];

/// 由自研发射器**在运行期生成一次**的模块表（`OnceLock` 缓存，全程序唯一）。
///
/// 生成是纯函数（同输入必得同字节流），因此「运行期生成」不等于「不确定」：
/// [`build_is_deterministic`](super::spirv_bg) 直接断言两次构建逐字相同。
///
/// 表内顺序 = 注册顺序（确定性）：背景帧 kernel 在前，光追场景 kernel 在后。
pub fn generated_modules() -> &'static [(&'static str, &'static [u32])] {
    static GENERATED: std::sync::OnceLock<Vec<(&'static str, &'static [u32])>> =
        std::sync::OnceLock::new();
    GENERATED.get_or_init(|| {
        vec![
            (
                super::spirv_bg::MODULE_BACKGROUND,
                super::spirv_bg::background_words(),
            ),
            (
                super::spirv_rt::MODULE_SCENE,
                super::spirv_rt::scene_words(),
            ),
        ]
    })
}

/// 按名取模块字流（静态表 + 生成表）。
pub fn module(name: &str) -> Option<&'static [u32]> {
    if let Some((_, w)) = MODULES.iter().find(|(n, _)| *n == name) {
        return Some(w);
    }
    generated_modules()
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, w)| *w)
}

/// 模块名全集（确定性顺序：静态表在前，生成表在后）。
pub fn module_names() -> Vec<&'static str> {
    MODULES
        .iter()
        .chain(generated_modules().iter())
        .map(|(n, _)| *n)
        .collect()
}

// ---------------------------------------------------------------------------
// CPU 参考语义（二进制宣称行为的可执行对照）
// ---------------------------------------------------------------------------

/// `spv_mul2_probe` 的 CPU 参考语义：`dst[i] = src[i] * 2.0`（长度取二者较小值）。
///
/// 二进制的语义主张由本函数锚定：结构校验（[`validate`]）+ CPU 参考对拍
/// + lavapipe 真机读回对拍（`compute::tests::real_device_compute_dispatch_smoke`）。
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
    /// `OpAccessChain` 的索引链在**非复合类型**上继续下钻
    /// （驱动侧表现是编译错误，激进实现直接崩溃）。
    ///
    /// 真实缺陷来源：把「指向 4 成员结构体的指针」误当成「指向数组的指针」
    /// 多写了一个索引——`spirv-val` 报 "reached non-composite type"，
    /// 而 lavapipe 在 W4 首次上机时直接 SIGSEGV。
    AccessChainNotComposite {
        /// 指令起始字下标。
        at: usize,
    },
    /// `OpAccessChain` 的结构体成员下标越界。
    AccessChainMemberOutOfRange {
        /// 指令起始字下标。
        at: usize,
        /// 请求的成员下标。
        member: u32,
        /// 结构体成员数。
        count: usize,
    },
    /// 结构化控制流被破坏：块未以终结指令收尾、块终结后仍有指令、
    /// `OpLoopMerge`/`OpSelectionMerge` 未紧邻其后的分支、或分支目标不是 label。
    ///
    /// 光追 megakernel 引入动态循环（`OpLoopMerge` + 回边）后，这类缺陷的
    /// 驱动侧表现是**进程级崩溃**（lavapipe 建管线时 SIGSEGV），故必须在
    /// 无设备阶段静态拦下。
    MalformedControlFlow {
        /// 指令起始字下标。
        at: usize,
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
            SpvError::AccessChainNotComposite { .. } => "access-chain-not-composite",
            SpvError::AccessChainMemberOutOfRange { .. } => "access-chain-member-out-of-range",
            SpvError::MalformedControlFlow { .. } => "malformed-control-flow",
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

/// `OpAccessChain` 校验需要的「类型形状」子集。
enum SpvShape {
    /// 结构体：成员类型 id（按成员序）。
    Struct(Vec<u32>),
    /// 数组 / 运行期数组：元素类型 id。
    Sequence(u32),
    /// 标量 / 向量 / 其它不可下钻的类型。
    Leaf,
    /// 指针：指向的类型 id。
    Pointer { pointee: u32 },
}

/// **索引链形状校验**：`OpAccessChain` 的每个下标都必须作用在复合类型上。
///
/// 这是 W4 真机 SIGSEGV 的直接产物：`Params` 是「指向 4 成员结构体的指针」，
/// 访问成员只需**一个**下标；多写一个下标就落在标量 `u32` 上。`spirv-val`
/// 报 "reached non-composite type"，而 lavapipe 在建管线时崩溃——一个纯
/// 结构缺陷把整个测试进程带走。此检查让同类缺陷在**无设备**时就被拦下。
///
/// 只跟踪结构体/数组/指针/标量的形状，不做类型匹配（那需要完整类型检查器）。
fn check_access_chains(words: &[u32]) -> Result<(), SpvError> {
    let mut shapes: Vec<(u32, SpvShape)> = Vec::new();
    let mut vars: Vec<(u32, u32)> = Vec::new();
    let mut consts: Vec<(u32, u32)> = Vec::new();

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if wc == 0 || i + wc > words.len() {
            // 字长异常由 `validate` 主循环报更精确的错误。
            break;
        }
        match words[i] & 0xFFFF {
            OP_TYPE_STRUCT if wc >= 3 => {
                shapes.push((words[i + 1], SpvShape::Struct(words[i + 2..i + wc].to_vec())));
            }
            OP_TYPE_RUNTIME_ARRAY if wc >= 3 => {
                shapes.push((words[i + 1], SpvShape::Sequence(words[i + 2])));
            }
            OP_TYPE_ARRAY if wc >= 4 => {
                shapes.push((words[i + 1], SpvShape::Sequence(words[i + 2])));
            }
            OP_TYPE_POINTER if wc >= 4 => {
                shapes.push((words[i + 1], SpvShape::Pointer { pointee: words[i + 3] }));
            }
            OP_TYPE_VOID | OP_TYPE_BOOL | OP_TYPE_INT | OP_TYPE_FLOAT | OP_TYPE_VECTOR => {
                shapes.push((words[i + 1], SpvShape::Leaf));
            }
            OP_VARIABLE if wc >= 3 => vars.push((words[i + 2], words[i + 1])),
            OP_CONSTANT if wc >= 4 => consts.push((words[i + 2], words[i + 3])),
            _ => {}
        }
        i += wc;
    }

    let shape_of = |id: u32| -> Option<&SpvShape> {
        shapes.iter().find(|(t, _)| *t == id).map(|(_, s)| s)
    };

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        if words[i] & 0xFFFF == OP_ACCESS_CHAIN && wc >= 5 {
            let base = words[i + 3];
            if let Some(&(_, base_ty)) = vars.iter().find(|(v, _)| *v == base) {
                if let Some(SpvShape::Pointer { pointee }) = shape_of(base_ty) {
                    let mut cur = *pointee;
                    for &idx in &words[i + 4..i + wc] {
                        match shape_of(cur) {
                            Some(SpvShape::Struct(members)) => {
                                // 结构体下标必须是常量；取不到常量时跳过该项检查
                                // （宁可漏检，不误报）。
                                let Some(&(_, member)) =
                                    consts.iter().find(|(c, _)| *c == idx)
                                else {
                                    break;
                                };
                                let m = member as usize;
                                if m >= members.len() {
                                    return Err(SpvError::AccessChainMemberOutOfRange {
                                        at: i,
                                        member,
                                        count: members.len(),
                                    });
                                }
                                cur = members[m];
                            }
                            Some(SpvShape::Sequence(elem)) => cur = *elem,
                            _ => return Err(SpvError::AccessChainNotComposite { at: i }),
                        }
                    }
                }
            }
        }
        i += wc;
    }
    Ok(())
}

/// **结构化控制流静态校验**：块终结性 + merge 指令位置 + 分支目标存在性。
///
/// 光追 megakernel 是**本仓第一个含动态循环**的模块（`OpLoopMerge` + 回边 +
/// 嵌套选择）。这类模块的一处结构性错误（少一条 `OpBranch`、merge 指令后
/// 直接跟 label、分支指向不存在的 id）在 lavapipe 上的表现不是「管线创建
/// 失败」而是**建管线时进程崩溃**——整个测试二进制被带走。故把「块是否
/// 以终结指令收尾」「merge 是否紧邻分支」「分支目标是否是已声明 label」
/// 这三条最容易写错的结构契约静态钉死。
///
/// 只查结构，不查支配关系/嵌套合法性（完整验证需 `spirv-val`，本环境无）。
fn check_control_flow(words: &[u32]) -> Result<(), SpvError> {
    // ① 收集全部 label id（分支目标必须落在集合内）。
    let mut labels: Vec<u32> = Vec::new();
    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        if words[i] & 0xFFFF == OP_LABEL && wc >= 2 {
            labels.push(words[i + 1]);
        }
        i += wc;
    }

    // ② 逐块检查终结性。
    let mut i = 5usize;
    let mut in_block = false;
    let mut terminated = true;
    let mut prev_op: u32 = 0;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        let op = words[i] & 0xFFFF;
        let next = i + wc;
        match op {
            OP_LABEL => {
                // 上一块必须以终结指令收尾（否则控制流会「掉进」下一块）。
                if in_block && !terminated {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                in_block = true;
                terminated = false;
            }
            OP_FUNCTION => {
                in_block = false;
                terminated = true;
            }
            OP_FUNCTION_END => {
                if !terminated {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                in_block = false;
                terminated = true;
            }
            OP_LOOP_MERGE | OP_SELECTION_MERGE => {
                // merge 指令必须是块的倒数第二条：紧跟一条分支。
                if next >= words.len() {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                let next_op = words[next] & 0xFFFF;
                if next_op != OP_BRANCH && next_op != OP_BRANCH_CONDITIONAL {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                // 目标 label（LoopMerge: continue + merge；SelectionMerge: merge）必须存在。
                let mut targets: &[u32] = &words[i + 1..i + 2];
                if op == OP_LOOP_MERGE {
                    if wc < 4 {
                        return Err(SpvError::MalformedControlFlow { at: i });
                    }
                    targets = &words[i + 1..i + 3];
                }
                for &t in targets {
                    if !labels.contains(&t) {
                        return Err(SpvError::MalformedControlFlow { at: i });
                    }
                }
            }
            OP_BRANCH => {
                if wc < 2 || !labels.contains(&words[i + 1]) {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                terminated = true;
            }
            OP_BRANCH_CONDITIONAL => {
                // 操作数序：条件 + 真 label + 假 label（可选权重）。
                if wc < 4 || !labels.contains(&words[i + 2]) || !labels.contains(&words[i + 3]) {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                // 条件分支的前一条必须是 merge 指令（结构化控制流的硬要求）。
                if prev_op != OP_LOOP_MERGE && prev_op != OP_SELECTION_MERGE {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
                terminated = true;
            }
            OP_RETURN => terminated = true,
            _ => {
                // 终结指令之后不得再有普通指令（块已结束）。
                if in_block && terminated {
                    return Err(SpvError::MalformedControlFlow { at: i });
                }
            }
        }
        prev_op = op;
        i = next;
    }
    Ok(())
}

/// 结构验证：`words` 是完整模块（含 5 字头部）。
///
/// 只做**结构**校验（不做类型/语义验算——那需 `spirv-val`）；额外检查
/// `OpAccessChain` 的索引链形状（见私有函数 `check_access_chains`）与
/// 结构化控制流的块终结性（见私有函数 `check_control_flow`）。
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
    // 索引链形状：最后一道（也是最像驱动的一道）静态门禁。
    check_access_chains(words)?;
    // 结构化控制流（循环/选择）：本仓首个含动态循环的模块即光追 megakernel，
    // 结构错在驱动上表现为崩溃，故同样在无设备阶段拦下。
    check_control_flow(words)?;
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

    /// **真机崩溃的静态复现**：`OpAccessChain` 多一个下标落在标量上。
    ///
    /// 背景 kernel 的 `Params` 是 4 成员结构体，访问成员只该有 1 个下标。
    /// 施工中曾误写成 `[0, member]`（照抄 BufferBlock 内的数组访问），
    /// `spirv-val` 报 "reached non-composite type"，lavapipe **SIGSEGV**。
    /// 这里把该缺陷原样注回模块，断言结构校验能拦下它——无设备、无崩溃。
    #[test]
    fn validator_catches_access_chain_through_scalar() {
        use crate::vulkan::spirv_bg::{background_words, ENTRY_BACKGROUND, WORKGROUP_BACKGROUND};
        let expect = SpvExpect {
            entry: ENTRY_BACKGROUND,
            local_size: WORKGROUP_BACKGROUND,
            bindings: &[0, 1],
        };
        let good = background_words();
        validate(good, &expect).expect("原始模块应合法");

        // 定位：第一条访问链（Params 成员，1 个下标）与最后一条（输出，2 个下标）。
        let mut params_var = 0u32;
        let mut last_chain = 0usize;
        let mut first_chain_seen = false;
        let mut i = 5usize;
        while i < good.len() {
            let wc = (good[i] >> 16) as usize;
            assert!(wc > 0);
            if good[i] & 0xFFFF == OP_ACCESS_CHAIN {
                if !first_chain_seen && wc == 5 {
                    params_var = good[i + 3];
                    first_chain_seen = true;
                }
                last_chain = i;
            }
            i += wc;
        }
        assert_ne!(params_var, 0, "未找到参数块访问链");

        // ① 把输出访问链的基址换成 Params（struct{4×u32}）：
        //    [0, idx] 中 idx 落在 u32 上 ⇒ non-composite。
        let mut bad = good.to_vec();
        bad[last_chain + 3] = params_var;
        assert!(
            matches!(
                validate(&bad, &expect),
                Err(SpvError::AccessChainNotComposite { .. })
            ),
            "索引链穿过标量必须被静态拦下（否则真机会崩驱动）"
        );

        // ② 结构体成员下标越界：把某条 Params 链的下标常量换成一个巨大的常量 id。
        let big_const = {
            let mut found = 0u32;
            let mut j = 5usize;
            while j < good.len() {
                let wc = (good[j] >> 16) as usize;
                if good[j] & 0xFFFF == OP_CONSTANT && wc >= 4 && good[j + 3] > 16 {
                    found = good[j + 2];
                    break;
                }
                j += wc;
            }
            found
        };
        assert_ne!(big_const, 0, "模块中应存在大常量（hash 乘子）");
        let mut bad2 = good.to_vec();
        // 第一条 Params 访问链的最后一个操作数就是成员下标。
        let mut i = 5usize;
        let mut patched = false;
        while i < good.len() {
            let wc = (good[i] >> 16) as usize;
            if good[i] & 0xFFFF == OP_ACCESS_CHAIN && wc == 5 && good[i + 3] == params_var {
                bad2[i + 4] = big_const;
                patched = true;
                break;
            }
            i += wc;
        }
        assert!(patched, "未找到 Params 访问链");
        assert!(
            matches!(
                validate(&bad2, &expect),
                Err(SpvError::AccessChainMemberOutOfRange { .. })
            ),
            "越界成员下标必须被拦下"
        );

        // 失败原因短名非空。
        for e in [
            SpvError::AccessChainNotComposite { at: 0 },
            SpvError::AccessChainMemberOutOfRange {
                at: 0,
                member: 9,
                count: 4,
            },
        ] {
            assert!(!e.as_str().is_empty());
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
        assert_eq!(
            module_names(),
            vec!["spv_mul2_probe", "spv_background", "spv_scene"],
            "静态模块 + 发射器模块，顺序固定"
        );
        // 手写字流只有一个；两个发射器产物（背景帧 / 光追场景）——两张表口径不混。
        assert_eq!(MODULES.len(), 1);
        assert_eq!(generated_modules().len(), 2);
        // 生成表指向 `OnceLock` 中的同一实例（指针稳定，FFI 传得出去）。
        let a = module("spv_background").expect("背景模块已注册");
        let b = module("spv_background").expect("背景模块已注册");
        assert_eq!(a.as_ptr(), b.as_ptr(), "同一模块须全程序唯一实例");
        let c = module("spv_scene").expect("场景模块已注册");
        let d = module("spv_scene").expect("场景模块已注册");
        assert_eq!(c.as_ptr(), d.as_ptr(), "同一模块须全程序唯一实例");
        assert_ne!(a.as_ptr(), c.as_ptr(), "两个模块不得共享同一字流");
    }
}
