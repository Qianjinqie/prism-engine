//! 背景渲染 kernel 的 SPIR-V 生成（自研发射器，零第三方依赖 — D-15）。
//!
//! # 为什么需要发射器而不是手写字流
//!
//! 上一个波的 `spv_mul2_probe` 是**逐字手写**的 79 字模块——对一条乘法够用，
//! 对背景渲染（垂直渐变 + 二次辉光 + value noise + 暗角 + 呼吸，实测 398 条
//! 指令）不可维护。本模块把「SPIR-V 汇编」抽成最小发射器：id 分配、段序
//! （capability → 扩展 → 内存模型 → 入口 → 注解 → 类型/常量/全局 → 函数）、
//! 常量去重（同 bits 只发一次）全部由代码保证，语义仍**逐式对应**
//! [`crate::background`] 的 CPU 参考。
//!
//! # 正确性判据（`docs/设计/研究/能力对齐的数学分层.md` §6 命题 P1）
//!
//! 背景算子是**逐点算子**：输出 `A(x)[p]` 只依赖 `x[p]` 与全局参数
//! `(w, h, t, seed)`，支撑半径 `r = 0`。按 P1，tile 分解在 `halo ≥ r = 0` 时
//! **与全局逐位相同**——即 GPU 与 CPU 的判等口径必须是**逐字节**，不是容差。
//! 本发射器为此做了三件事：
//!
//! 1. **同式**：每个乘加都按 CPU 的求值树（左结合、先乘后加）发射，不做
//!    代数重排；
//! 2. **禁收缩**：每条浮点算术结果都打 `NoContraction` 装饰——否则驱动可把
//!    `a*b + c` 熔成 FMA，末位就会与 CPU 不同（P1 的「逐位」判据被破坏）；
//! 3. **无超越函数**（D-29）：`floor`/`sqrt` 走 `GLSL.std.450`（两者都是
//!    IEEE 精确/正确舍入运算），正弦走 CPU 同款 Maclaurin 逼近。
//!
//! # 落点
//!
//! * binding 0：`Params { u32 w, u32 h, u32 t_bits, u32 seed }`（只读）；
//!   `t` 以 **f32 位型**传入，shader 用 `OpBitcast` 还原——避免第二套浮点
//!   文本/十进制往返口径。
//! * binding 1：`array<u32>`，每像素一个 little-endian 打包 RGBA8
//!   （`r | g<<8 | b<<16 | 255<<24`），与 [`crate::background`] 的字节序一致。
//! * 工作组 `8 × 8 × 1`；越界调用由 `OpSelectionMerge` 守卫丢弃（尾组不写
//!   越界内存，也不 panic）。

use std::sync::OnceLock;

use super::spirv::{
    inst, DECORATION_ARRAY_STRIDE, DECORATION_BINDING, DECORATION_BUFFER_BLOCK,
    DECORATION_DESCRIPTOR_SET, DECORATION_NO_CONTRACTION, DECORATION_OFFSET,
    EXECUTION_MODE_LOCAL_SIZE, EXECUTION_MODEL_GLCOMPUTE, MAGIC, OP_ACCESS_CHAIN, OP_BITCAST,
    OP_BITWISE_OR, OP_BITWISE_XOR, OP_BRANCH, OP_BRANCH_CONDITIONAL, OP_CAPABILITY,
    OP_COMPOSITE_EXTRACT, OP_CONSTANT, OP_CONVERT_F_TO_U, OP_CONVERT_U_TO_F, OP_DECORATE,
    OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_EXT_INST, OP_EXT_INST_IMPORT, OP_F_ADD, OP_F_DIV,
    OP_F_MUL, OP_F_ORD_GREATER_THAN, OP_F_ORD_LESS_THAN, OP_F_SUB, OP_FUNCTION,
    OP_FUNCTION_END, OP_I_ADD, OP_I_MUL, OP_LABEL, OP_LOAD, OP_LOGICAL_OR, OP_MEMBER_DECORATE,
    OP_MEMORY_MODEL, OP_RETURN, OP_SELECT, OP_SELECTION_MERGE, OP_SHIFT_LEFT_LOGICAL,
    OP_SHIFT_RIGHT_LOGICAL, OP_STORE, OP_TYPE_BOOL, OP_TYPE_FLOAT, OP_TYPE_FUNCTION, OP_TYPE_INT,
    OP_TYPE_POINTER, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_VECTOR, OP_TYPE_VOID,
    OP_U_GREATER_THAN_EQUAL, OP_VARIABLE, SCHEMA, STORAGE_CLASS_INPUT, STORAGE_CLASS_UNIFORM,
    VERSION_1_0,
};
use crate::background::{BG_BOTTOM, BG_TOP, GLOW, INV_U32, TAU};

/// 背景 kernel 的 SPIR-V 模块名（注册到 [`super::spirv::module`]）。
pub const MODULE_BACKGROUND: &str = "spv_background";
/// 背景 kernel 的入口名（`OpEntryPoint`）。
pub const ENTRY_BACKGROUND: &str = "bg_main";
/// 工作组尺寸（x, y, z）；1080p 一帧 = 240 × 135 组。
pub const WORKGROUP_BACKGROUND: [u32; 3] = [8, 8, 1];

/// `OpCapability Shader` 的 capability 号。
const CAPABILITY_SHADER: u32 = 1;
/// 逻辑寻址模型 `Logical`。
const ADDRESSING_LOGICAL: u32 = 0;
/// 内存模型 `GLSL450`。
const MEMORY_MODEL_GLSL450: u32 = 1;
/// 函数控制位 `None`。
const FUNCTION_CONTROL_NONE: u32 = 0;
/// 选择控制位 `None`。
const SELECTION_CONTROL_NONE: u32 = 0;

// GLSL.std.450 扩展指令号（1.00 修订版公开编号）。
/// `Floor(x)`：IEEE 精确取整。
const GLSL_FLOOR: u32 = 8;
/// `Sqrt(x)`：正确舍入平方根。
const GLSL_SQRT: u32 = 31;
/// `FMax(x, y)`。
const GLSL_FMAX: u32 = 40;
/// `FClamp(x, min, max)`。
const GLSL_FCLAMP: u32 = 43;

/// 生成（并缓存）背景 kernel 的 SPIR-V 字流。
///
/// `OnceLock` 保证**全程序唯一实例**——`vkCreateShaderModule` 拿到的指针
/// 地址在多次调用间必须一致（与 `SHADERS` 用 `static` 而非 `const` 同一教训）。
pub fn background_words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(build_background_module)
}

/// 发射一条指令：`operands` 为首字之外的全部字，字长由此推出。
fn push(dst: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    dst.push(inst((operands.len() + 1) as u32, opcode));
    dst.extend_from_slice(operands);
}

/// 把字符串按 SPIR-V 的 literal string 编码（UTF-8 + `\0` 终结 + 4 字节补齐）。
fn push_str(words: &mut Vec<u32>, s: &str) {
    let mut buf: Vec<u8> = s.as_bytes().to_vec();
    buf.push(0);
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
    for chunk in buf.chunks(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

/// 最小 SPIR-V 发射器：按规范段序累积指令，末尾拼装成完整模块。
struct Emitter {
    // --- 逻辑段（SPIR-V §2.4 的固定顺序；段内再分）---
    caps: Vec<u32>,
    ext_imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    constants: Vec<u32>,
    globals: Vec<u32>,
    funcs: Vec<u32>,
    // --- id 分配 ---
    next_id: u32,
    // --- 类型 id ---
    t_void: u32,
    t_bool: u32,
    t_f32: u32,
    t_u32: u32,
    t_v3u32: u32,
    p_in_v3: u32,
    p_u32: u32,
    p_params: u32,
    p_out: u32,
    t_struct_params: u32,
    t_struct_out: u32,
    t_fn: u32,
    // --- 全局变量 id ---
    v_gid: u32,
    v_params: u32,
    v_out: u32,
    // --- 其它 ---
    glsl: u32,
    fn_main: u32,
    // --- 常量去重（bits → id，线性查找：常量种类少，无需哈希）---
    f32_consts: Vec<(u32, u32)>,
    u32_consts: Vec<(u32, u32)>,
}

impl Emitter {
    fn fresh(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// 发射头三段 + 入口 + 类型/全局变量（全部 id 在 `new` 里已分配）。
    fn new() -> Self {
        let mut e = Emitter {
            caps: Vec::new(),
            ext_imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            annotations: Vec::new(),
            types: Vec::new(),
            constants: Vec::new(),
            globals: Vec::new(),
            funcs: Vec::new(),
            next_id: 1,
            t_void: 0,
            t_bool: 0,
            t_f32: 0,
            t_u32: 0,
            t_v3u32: 0,
            p_in_v3: 0,
            p_u32: 0,
            p_params: 0,
            p_out: 0,
            t_struct_params: 0,
            t_struct_out: 0,
            t_fn: 0,
            v_gid: 0,
            v_params: 0,
            v_out: 0,
            glsl: 0,
            fn_main: 0,
            f32_consts: Vec::new(),
            u32_consts: Vec::new(),
        };
        e.declare_prologue();
        e.declare_types();
        e.declare_globals();
        e
    }

    /// ① capability → ② 扩展/内存模型 → ③ 入口/执行模式 → ④ 注解。
    fn declare_prologue(&mut self) {
        push(&mut self.caps, OP_CAPABILITY, &[CAPABILITY_SHADER]);

        self.glsl = self.fresh();
        let mut ops = vec![self.glsl];
        push_str(&mut ops, "GLSL.std.450");
        push(&mut self.ext_imports, OP_EXT_INST_IMPORT, &ops);

        push(
            &mut self.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        self.fn_main = self.fresh();
        self.v_gid = self.fresh();
        self.v_params = self.fresh();
        self.v_out = self.fresh();

        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, self.fn_main];
        push_str(&mut ops, ENTRY_BACKGROUND);
        ops.push(self.v_gid);
        push(&mut self.entry_points, OP_ENTRY_POINT, &ops);

        push(
            &mut self.entry_points,
            OP_EXECUTION_MODE,
            &[
                self.fn_main,
                EXECUTION_MODE_LOCAL_SIZE,
                WORKGROUP_BACKGROUND[0],
                WORKGROUP_BACKGROUND[1],
                WORKGROUP_BACKGROUND[2],
            ],
        );

        // ④ 注解：Input 变量必须 BuiltIn（缺它模块非法，W3 已实测）。
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_gid, super::spirv::DECORATION_BUILT_IN, super::spirv::BUILTIN_GLOBAL_INVOCATION_ID],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_params, DECORATION_DESCRIPTOR_SET, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_params, DECORATION_BINDING, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_out, DECORATION_DESCRIPTOR_SET, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_out, DECORATION_BINDING, 1],
        );
    }

    /// ⑤ 类型段（全部类型一次发完，避免「常量后置类型」的段序违规）。
    fn declare_types(&mut self) {
        self.t_void = self.fresh();
        push(&mut self.types, OP_TYPE_VOID, &[self.t_void]);

        self.t_bool = self.fresh();
        push(&mut self.types, OP_TYPE_BOOL, &[self.t_bool]);

        self.t_f32 = self.fresh();
        push(&mut self.types, OP_TYPE_FLOAT, &[self.t_f32, 32]);

        self.t_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_INT, &[self.t_u32, 32, 0]);

        self.t_v3u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_VECTOR,
            &[self.t_v3u32, self.t_u32, 3],
        );

        self.p_in_v3 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_in_v3, STORAGE_CLASS_INPUT, self.t_v3u32],
        );

        self.p_u32 = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_u32, STORAGE_CLASS_UNIFORM, self.t_u32],
        );

        let rt_arr = self.fresh();
        push(&mut self.types, OP_TYPE_RUNTIME_ARRAY, &[rt_arr, self.t_u32]);

        self.t_struct_out = self.fresh();
        push(&mut self.types, OP_TYPE_STRUCT, &[self.t_struct_out, rt_arr]);

        self.p_out = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_out, STORAGE_CLASS_UNIFORM, self.t_struct_out],
        );

        self.t_struct_params = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_STRUCT,
            &[
                self.t_struct_params,
                self.t_u32,
                self.t_u32,
                self.t_u32,
                self.t_u32,
            ],
        );

        self.p_params = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_POINTER,
            &[self.p_params, STORAGE_CLASS_UNIFORM, self.t_struct_params],
        );

        self.t_fn = self.fresh();
        push(
            &mut self.types,
            OP_TYPE_FUNCTION,
            &[self.t_fn, self.t_void],
        );

        // SSBO 装饰（SPIR-V 1.0 / Vulkan 1.0 口径：Uniform + BufferBlock，
        // 与 W3 实测跑通的 mul2 模块同款）。
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.t_struct_params, DECORATION_BUFFER_BLOCK],
        );
        for (i, off) in [0u32, 4, 8, 12].into_iter().enumerate() {
            push(
                &mut self.annotations,
                OP_MEMBER_DECORATE,
                &[self.t_struct_params, i as u32, DECORATION_OFFSET, off],
            );
        }
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.t_struct_out, DECORATION_BUFFER_BLOCK],
        );
        push(
            &mut self.annotations,
            OP_MEMBER_DECORATE,
            &[self.t_struct_out, 0, DECORATION_OFFSET, 0],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[rt_arr, DECORATION_ARRAY_STRIDE, 4],
        );
    }

    /// ⑤ 全局变量段（在类型之后）。
    fn declare_globals(&mut self) {
        push(
            &mut self.globals,
            OP_VARIABLE,
            &[self.p_in_v3, self.v_gid, STORAGE_CLASS_INPUT],
        );
        push(
            &mut self.globals,
            OP_VARIABLE,
            &[self.p_params, self.v_params, STORAGE_CLASS_UNIFORM],
        );
        push(
            &mut self.globals,
            OP_VARIABLE,
            &[self.p_out, self.v_out, STORAGE_CLASS_UNIFORM],
        );
    }

    // ------------------------------------------------------------------
    // 常量（同 bits 去重：同一常量全模块只发一次）
    // ------------------------------------------------------------------

    fn f32_const(&mut self, v: f32) -> u32 {
        let bits = v.to_bits();
        if let Some(&(_, id)) = self.f32_consts.iter().find(|(b, _)| *b == bits) {
            return id;
        }
        let id = self.fresh();
        push(&mut self.constants, OP_CONSTANT, &[self.t_f32, id, bits]);
        self.f32_consts.push((bits, id));
        id
    }

    fn u32_const(&mut self, v: u32) -> u32 {
        if let Some(&(_, id)) = self.u32_consts.iter().find(|(b, _)| *b == v) {
            return id;
        }
        let id = self.fresh();
        push(&mut self.constants, OP_CONSTANT, &[self.t_u32, id, v]);
        self.u32_consts.push((v, id));
        id
    }

    // ------------------------------------------------------------------
    // 浮点指令（全部打 NoContraction：禁止驱动把 mul+add 熔成 FMA，
    // 否则末位与 CPU 不同 —— P1 的「逐位」判据要求）
    // ------------------------------------------------------------------

    fn fbin(&mut self, opcode: u32, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, opcode, &[self.t_f32, id, a, b]);
        push(&mut self.annotations, OP_DECORATE, &[id, DECORATION_NO_CONTRACTION]);
        id
    }

    fn fadd(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_ADD, a, b)
    }

    fn fsub(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_SUB, a, b)
    }

    fn fmul(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_MUL, a, b)
    }

    fn fdiv(&mut self, a: u32, b: u32) -> u32 {
        self.fbin(OP_F_DIV, a, b)
    }

    /// `GLSL.std.450` 一元扩展指令。
    fn ext1(&mut self, inst_no: u32, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, inst_no, x],
        );
        id
    }

    /// `GLSL.std.450` 二元扩展指令。
    fn ext2(&mut self, inst_no: u32, x: u32, y: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, inst_no, x, y],
        );
        id
    }

    fn fmax(&mut self, x: u32, y: u32) -> u32 {
        self.ext2(GLSL_FMAX, x, y)
    }

    fn fclamp(&mut self, x: u32, lo: f32, hi: f32) -> u32 {
        let lo = self.f32_const(lo);
        let hi = self.f32_const(hi);
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_EXT_INST,
            &[self.t_f32, id, self.glsl, GLSL_FCLAMP, x, lo, hi],
        );
        id
    }

    fn fgt(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_F_ORD_GREATER_THAN,
            &[self.t_bool, id, a, b],
        );
        id
    }

    fn flt(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_F_ORD_LESS_THAN, &[self.t_bool, id, a, b]);
        id
    }

    fn select_f32(&mut self, cond: u32, if_true: u32, if_false: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_SELECT,
            &[self.t_f32, id, cond, if_true, if_false],
        );
        id
    }

    // ------------------------------------------------------------------
    // 整数指令 / 类型转换
    // ------------------------------------------------------------------

    fn ubin(&mut self, opcode: u32, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, opcode, &[self.t_u32, id, a, b]);
        id
    }

    fn uadd(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_I_ADD, a, b)
    }

    fn umul(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_I_MUL, a, b)
    }

    fn ushr(&mut self, a: u32, shift: u32) -> u32 {
        let s = self.u32_const(shift);
        self.ubin(OP_SHIFT_RIGHT_LOGICAL, a, s)
    }

    fn ushl(&mut self, a: u32, shift: u32) -> u32 {
        let s = self.u32_const(shift);
        self.ubin(OP_SHIFT_LEFT_LOGICAL, a, s)
    }

    fn uxor(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_BITWISE_XOR, a, b)
    }

    fn uor(&mut self, a: u32, b: u32) -> u32 {
        self.ubin(OP_BITWISE_OR, a, b)
    }

    fn u2f(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_CONVERT_U_TO_F,
            &[self.t_f32, id, x],
        );
        id
    }

    fn f2u(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(
            &mut self.funcs,
            OP_CONVERT_F_TO_U,
            &[self.t_u32, id, x],
        );
        id
    }

    fn bitcast_to_f32(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_BITCAST, &[self.t_f32, id, x]);
        id
    }

    fn load(&mut self, ty: u32, ptr: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[ty, id, ptr]);
        id
    }

    /// 读 `Params` 的第 `member` 个 u32 成员。
    ///
    /// 索引只有一个：基址是「指向 4 成员结构体的指针」，`OpAccessChain` 的
    /// 首个索引就是**成员下标**（不是数组下标）。多加一个索引会作用在标量
    /// 上——`spirv-val` 直接报 "reached non-composite type"。
    fn load_param(&mut self, member: u32) -> u32 {
        let cm = self.u32_const(member);
        let ptr = self.fresh();
        push(
            &mut self.funcs,
            OP_ACCESS_CHAIN,
            &[self.p_u32, ptr, self.v_params, cm],
        );
        self.load(self.t_u32, ptr)
    }

    // ------------------------------------------------------------------
    // 语义件（逐式对应 background.rs）
    // ------------------------------------------------------------------

    /// `t * t * (3 - 2t)`（**不**含 clamp；调用方按 CPU 的 `smooth01` 先 clamp）。
    fn smooth_poly(&mut self, t: u32) -> u32 {
        let tt = self.fmul(t, t);
        let two = self.f32_const(2.0);
        let two_t = self.fmul(two, t);
        let three = self.f32_const(3.0);
        let poly = self.fsub(three, two_t);
        self.fmul(tt, poly)
    }

    /// `hash2i`：格点 + seed → [0,1)。
    fn hash2i(&mut self, ix: u32, iy: u32, seed: u32) -> u32 {
        let ka = self.u32_const(0x9e37_79b1);
        let kb = self.u32_const(0x85eb_ca77);
        let kc = self.u32_const(0xc2b2_ae3d);
        let t0 = self.umul(ix, ka);
        let t1 = self.umul(iy, kb);
        let t2 = self.umul(seed, kc);
        let sum01 = self.uadd(t0, t1);
        let sum = self.uadd(sum01, t2);

        // hash_u32（wrapping u32 位运算，与 CPU 位级同式）。
        let s16 = self.ushr(sum, 16);
        let h1 = self.uxor(sum, s16);
        let ka2 = self.u32_const(0x7feb_352d);
        let h2 = self.umul(h1, ka2);
        let s15 = self.ushr(h2, 15);
        let h3 = self.uxor(h2, s15);
        let kb2 = self.u32_const(0x846c_a68b);
        let h4 = self.umul(h3, kb2);
        let s16b = self.ushr(h4, 16);
        let h5 = self.uxor(h4, s16b);

        let hf = self.u2f(h5);
        let inv = self.f32_const(INV_U32);
        self.fmul(hf, inv)
    }

    /// 2D value noise（双线性 + smoothstep 权重），与 CPU `value_noise` 同式。
    fn value_noise(&mut self, px: u32, py: u32, seed: u32) -> u32 {
        let flx = self.ext1(GLSL_FLOOR, px);
        let fly = self.ext1(GLSL_FLOOR, py);
        // Rust 的 `f32 as u32` 对负值饱和到 0；OpConvertFToU 对负值是 UB，
        // 故先按 0 下界钳制，使两端对 t < 0 的输入也一致（P1 的判据不能有洞）。
        let zero = self.f32_const(0.0);
        let ix_f = self.fmax(flx, zero);
        let iy_f = self.fmax(fly, zero);
        let ix = self.f2u(ix_f);
        let iy = self.f2u(iy_f);
        let fx = self.fsub(px, flx);
        let fy = self.fsub(py, fly);
        let ux = self.smooth_poly(fx);
        let uy = self.smooth_poly(fy);

        let v00 = self.hash2i(ix, iy, seed);
        let one_u = self.u32_const(1);
        let x1 = self.uadd(ix, one_u);
        let y1 = self.uadd(iy, one_u);
        let v10 = self.hash2i(x1, iy, seed);
        let v01 = self.hash2i(ix, y1, seed);
        let v11 = self.hash2i(x1, y1, seed);

        let d10 = self.fsub(v10, v00);
        let m10 = self.fmul(d10, ux);
        let a = self.fadd(v00, m10);
        let d11 = self.fsub(v11, v01);
        let m11 = self.fmul(d11, ux);
        let b = self.fadd(v01, m11);
        let dba = self.fsub(b, a);
        let mba = self.fmul(dba, uy);
        self.fadd(a, mba)
    }

    /// `sin_approx`：Maclaurin 5 阶 + 区间折叠，与 CPU 同式（D-29 无 libm）。
    fn sin_approx(&mut self, x: u32) -> u32 {
        let tau = self.f32_const(TAU);
        let pi = self.f32_const(std::f32::consts::PI);
        let q = self.fdiv(x, tau);
        let qf = self.ext1(GLSL_FLOOR, q);
        let tq = self.fmul(tau, qf);
        let a0 = self.fsub(x, tq);
        // CPU: if a > PI { a -= TAU } if a < -PI { a += TAU }
        let cond_hi = self.fgt(a0, pi);
        let hi_sub = self.fsub(a0, tau);
        let a1 = self.select_f32(cond_hi, hi_sub, a0);
        let neg_pi = self.f32_const(-std::f32::consts::PI);
        let cond_lo = self.flt(a1, neg_pi);
        let lo_add = self.fadd(a1, tau);
        let a2 = self.select_f32(cond_lo, lo_add, a1);

        let xx = self.fmul(a2, a2);
        let a3 = self.fmul(a2, xx);
        let a5 = self.fmul(a3, xx);
        let a7 = self.fmul(a5, xx);
        let a9 = self.fmul(a7, xx);

        // a - a3/6 + a5/120 - a7/5040 + a9/362880（严格左结合，与 CPU 同序）。
        let c6 = self.f32_const(6.0);
        let t3 = self.fdiv(a3, c6);
        let s1 = self.fsub(a2, t3);
        let c120 = self.f32_const(120.0);
        let t5 = self.fdiv(a5, c120);
        let s2 = self.fadd(s1, t5);
        let c5040 = self.f32_const(5040.0);
        let t7 = self.fdiv(a7, c5040);
        let s3 = self.fsub(s2, t7);
        let c362880 = self.f32_const(362_880.0);
        let t9 = self.fdiv(a9, c362880);
        self.fadd(s3, t9)
    }

    /// 单通道合成：`col = top + (bottom-top)*g + glow_c*glow`，再乘 `k` 并量化。
    fn channel_rgba(&mut self, c: usize, g: u32, glow: u32, k: u32) -> u32 {
        // 常量折叠用 f32 减法，与 CPU 运行期同一运算（IEEE 确定性）。
        let top = self.f32_const(BG_TOP[c]);
        let delta = self.f32_const(BG_BOTTOM[c] - BG_TOP[c]);
        let glow_coeff = self.f32_const(GLOW[c]);

        let lerp = self.fmul(delta, g);
        let col0 = self.fadd(top, lerp);
        let add_glow = self.fmul(glow_coeff, glow);
        let col = self.fadd(col0, add_glow);
        let out = self.fmul(col, k);
        let clamped = self.fclamp(out, 0.0, 1.0);
        let c255 = self.f32_const(255.0);
        let scaled = self.fmul(clamped, c255);
        self.f2u(scaled)
    }

    /// ⑥ 函数体：一像素一 invocation。
    fn build_body(&mut self) {
        let l_entry = self.fresh();
        let l_body = self.fresh();
        let l_merge = self.fresh();

        // 先把「字段读取」落到局部，避免 `self.foo(self.bar(..))` 的双重可变借用。
        let t_void = self.t_void;
        let t_fn = self.t_fn;
        let fn_main = self.fn_main;
        push(
            &mut self.funcs,
            OP_FUNCTION,
            &[t_void, fn_main, FUNCTION_CONTROL_NONE, t_fn],
        );
        push(&mut self.funcs, OP_LABEL, &[l_entry]);

        let t_v3u32 = self.t_v3u32;
        let v_gid = self.v_gid;
        let gid = self.load(t_v3u32, v_gid);
        let t_u32 = self.t_u32;
        let gx = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[t_u32, gx, gid, 0]);
        let gy = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[t_u32, gy, gid, 1]);

        let w = self.load_param(0);
        let h = self.load_param(1);

        // 越界守卫（尾组调用丢弃）：if (gx >= w || gy >= h) return;
        let t_bool = self.t_bool;
        let cond_x = self.fresh();
        push(
            &mut self.funcs,
            OP_U_GREATER_THAN_EQUAL,
            &[t_bool, cond_x, gx, w],
        );
        let cond_y = self.fresh();
        push(
            &mut self.funcs,
            OP_U_GREATER_THAN_EQUAL,
            &[t_bool, cond_y, gy, h],
        );
        let oob = self.fresh();
        push(
            &mut self.funcs,
            OP_LOGICAL_OR,
            &[t_bool, oob, cond_x, cond_y],
        );
        push(
            &mut self.funcs,
            OP_SELECTION_MERGE,
            &[l_merge, SELECTION_CONTROL_NONE],
        );
        push(
            &mut self.funcs,
            OP_BRANCH_CONDITIONAL,
            &[oob, l_merge, l_body],
        );
        push(&mut self.funcs, OP_LABEL, &[l_body]);

        // 参数（t 以 f32 位型传输，bitcast 还原；seed 是 u32）。
        let t_bits = self.load_param(2);
        let t = self.bitcast_to_f32(t_bits);
        let seed = self.load_param(3);

        // vx = f32(x) / max(f32(w) - 1, 1)
        let one = self.f32_const(1.0);
        let gxf = self.u2f(gx);
        let wf = self.u2f(w);
        let wm1 = self.fsub(wf, one);
        let den_x = self.fmax(wm1, one);
        let vx = self.fdiv(gxf, den_x);

        let gyf = self.u2f(gy);
        let hf = self.u2f(h);
        let hm1 = self.fsub(hf, one);
        let den_y = self.fmax(hm1, one);
        let vy = self.fdiv(gyf, den_y);

        // g = smooth01(vy)
        let cs = self.fclamp(vy, 0.0, 1.0);
        let g = self.smooth_poly(cs);

        // 辉光：d2 = (vx-.5)^2 + (vy-.4)^2; glow = max(1-3*d2, 0)^2
        let half = self.f32_const(0.5);
        let sx = self.fsub(vx, half);
        let c040 = self.f32_const(0.40);
        let sy = self.fsub(vy, c040);
        let sxsx = self.fmul(sx, sx);
        let sysy = self.fmul(sy, sy);
        let d2 = self.fadd(sxsx, sysy);
        let three = self.f32_const(3.0);
        let d2_3 = self.fmul(d2, three);
        let g0 = self.fsub(one, d2_3);
        let zero = self.f32_const(0.0);
        let g1 = self.fmax(g0, zero);
        let glow = self.fmul(g1, g1);

        // 噪声：nx = vx*2.2 + t*0.05, ny = vy*1.3 - t*0.035
        let c22 = self.f32_const(2.2);
        let vx22 = self.fmul(vx, c22);
        let c005 = self.f32_const(0.05);
        let t005 = self.fmul(t, c005);
        let nx = self.fadd(vx22, t005);
        let c13 = self.f32_const(1.3);
        let vy13 = self.fmul(vy, c13);
        let c0035 = self.f32_const(0.035);
        let t0035 = self.fmul(t, c0035);
        let ny = self.fsub(vy13, t0035);
        let vn = self.value_noise(nx, ny, seed);
        let c20 = self.f32_const(2.0);
        let vn2 = self.fmul(vn, c20);
        let n = self.fsub(vn2, one);
        let c003 = self.f32_const(0.03);
        let n003 = self.fmul(c003, n);
        let modulation = self.fadd(one, n003);

        // 暗角：rad = sqrt((vx-.5)^2 + (vy-.5)^2)
        let cy = self.fsub(vy, half);
        let cycy = self.fmul(cy, cy);
        let rad2 = self.fadd(sxsx, cycy);
        let rad = self.ext1(GLSL_SQRT, rad2);
        let c045 = self.f32_const(0.45);
        let rad_m = self.fsub(rad, c045);
        let c055 = self.f32_const(0.55);
        let va = self.fdiv(rad_m, c055);
        let va_c = self.fclamp(va, 0.0, 1.0);
        let vs = self.smooth_poly(va_c);
        let vs_p = self.fmul(vs, c055);
        let vig = self.fsub(one, vs_p);

        // 呼吸：1 + 0.015 * sin_approx(t * TAU/5)
        let tau_over_5 = self.f32_const(TAU / 5.0);
        let arg = self.fmul(t, tau_over_5);
        let sin_v = self.sin_approx(arg);
        let c0015 = self.f32_const(0.015);
        let s0015 = self.fmul(c0015, sin_v);
        let breath = self.fadd(one, s0015);

        let mb = self.fmul(modulation, breath);
        let k = self.fmul(mb, vig);

        // 三通道 + 打包（r | g<<8 | b<<16 | 255<<24）。
        let r = self.channel_rgba(0, g, glow, k);
        let gg = self.channel_rgba(1, g, glow, k);
        let bb = self.channel_rgba(2, g, glow, k);

        let alpha = self.u32_const(0xFF00_0000);
        let g_shift = self.ushl(gg, 8);
        let b_shift = self.ushl(bb, 16);
        let pack0 = self.uor(alpha, b_shift);
        let pack1 = self.uor(pack0, g_shift);
        let packed = self.uor(pack1, r);

        // out[gy * w + gx] = packed
        let row = self.umul(gy, w);
        let idx = self.uadd(row, gx);
        let c0 = self.u32_const(0);
        let p_u32 = self.p_u32;
        let v_out = self.v_out;
        let out_ptr = self.fresh();
        push(
            &mut self.funcs,
            OP_ACCESS_CHAIN,
            &[p_u32, out_ptr, v_out, c0, idx],
        );
        push(&mut self.funcs, OP_STORE, &[out_ptr, packed]);

        push(&mut self.funcs, OP_BRANCH, &[l_merge]);
        push(&mut self.funcs, OP_LABEL, &[l_merge]);
        push(&mut self.funcs, OP_RETURN, &[]);
        push(&mut self.funcs, OP_FUNCTION_END, &[]);
    }

    /// 拼装完整模块：5 字头 + 六个逻辑段。
    fn assemble(self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        out.push(MAGIC);
        out.push(VERSION_1_0);
        out.push(0); // generator = 0（自研手写，非工具生成 —— 如实标注）
        out.push(self.next_id); // bound = max id + 1
        out.push(SCHEMA);
        for seg in [
            self.caps,
            self.ext_imports,
            self.memory_model,
            self.entry_points,
            self.annotations,
            self.types,
            self.constants,
            self.globals,
            self.funcs,
        ] {
            out.extend_from_slice(&seg);
        }
        out
    }
}

/// 构建背景 kernel 的完整 SPIR-V 模块（确定性：同输入必得同字节流）。
pub fn build_background_module() -> Vec<u32> {
    let mut e = Emitter::new();
    e.build_body();
    e.assemble()
}

// ---------------------------------------------------------------------------
// 测试（无设备即可执行：结构校验 + 与 CPU 常量口径的一致性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::spirv::{disassemble, validate, SpvExpect};

    /// 结构验证：入口 / LocalSize / 绑定 / 单函数 / 指令可反汇编。
    #[test]
    fn background_module_is_structurally_valid() {
        let words = background_words();
        assert!(words.len() > 200, "背景 kernel 不应是玩具模块");
        let summary = validate(
            words,
            &SpvExpect {
                entry: ENTRY_BACKGROUND,
                local_size: WORKGROUP_BACKGROUND,
                bindings: &[0, 1],
            },
        )
        .expect("发射器产出的模块须通过结构验证");
        assert_eq!(summary.entry, "bg_main");
        assert_eq!(summary.local_size, [8, 8, 1]);
        assert_eq!(summary.bindings, vec![0, 1]);
        assert_eq!(summary.functions, 1, "模块应恰有一个函数");

        // 指令字长之和必须铺满整个字流（发射器的段序/字长自洽性）。
        let mut i = 5usize;
        let mut total = 0usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            assert!(wc > 0, "零字长指令 @{i}");
            total += wc;
            i += wc;
        }
        assert_eq!(5 + total, words.len(), "指令字长之和须铺满字流");

        // 关键件必须在场。
        for op in [
            OP_EXT_INST,
            OP_F_MUL,
            OP_F_DIV,
            OP_CONVERT_U_TO_F,
            OP_CONVERT_F_TO_U,
            OP_BITCAST,
            OP_SELECTION_MERGE,
            OP_BRANCH_CONDITIONAL,
            OP_FUNCTION_END,
        ] {
            assert!(
                summary.opcodes.contains(&op),
                "缺指令 {}",
                crate::vulkan::spirv::opcode_name(op)
            );
        }
        // 反汇编不得出现未知 OpCode（白名单与发射器同步）。
        let text = disassemble(words);
        assert!(!text.contains("OpUnknown"), "白名单漏了发射器用到的 OpCode：\n{text}");
        println!(
            "背景 SPIR-V：{} 字 / {} 条指令 / bound={}",
            words.len(),
            summary.instruction_count,
            summary.bound
        );
    }

    /// 发射确定性：同一次构建两次调用逐字相同（`OnceLock` 只构建一次，
    /// 这里直接调 `build_background_module` 验证构建本身无随机/无序来源）。
    #[test]
    fn build_is_deterministic() {
        let a = build_background_module();
        let b = build_background_module();
        assert_eq!(a, b, "同输入两次构建必须逐字相同");
        assert_eq!(background_words(), a.as_slice(), "缓存实例须与构建结果一致");
    }

    /// 常量口径：发射器引用的颜色常量必须与 CPU 参考同一组（单一权威）。
    #[test]
    fn color_constants_come_from_cpu_reference() {
        // 直接引用同一常量表：若两端漂移，本断言与上游 `background` 的测试同时红。
        assert_eq!(BG_TOP, [0.035, 0.055, 0.130]);
        assert_eq!(BG_BOTTOM, [0.058, 0.090, 0.205]);
        assert_eq!(GLOW, [0.030, 0.070, 0.120]);
        // TAU = 2π（f32 位型固定，两端同一字）。
        assert_eq!(TAU.to_bits(), (std::f32::consts::PI * 2.0).to_bits());
        // 1/2^32 是精确的 2 的幂，u32→f32 归一化不引入舍入。
        assert_eq!(INV_U32, 1.0 / 4_294_967_296.0);
        assert_eq!(INV_U32.to_bits(), 0x2F80_0000);
    }

    /// 模块名/入口名是跨后端稳定契约。
    #[test]
    fn module_identity_is_stable() {
        assert_eq!(MODULE_BACKGROUND, "spv_background");
        assert_eq!(ENTRY_BACKGROUND, "bg_main");
        assert_eq!(WORKGROUP_BACKGROUND, [8, 8, 1]);
        // 注册表可见（`spirv::module` 是唯一取用口）。
        assert_eq!(
            crate::vulkan::spirv::module(MODULE_BACKGROUND).map(|w| w.len()),
            Some(background_words().len())
        );
    }
}
