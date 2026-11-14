//! 2D 矢量路径光栅 kernel 的 SPIR-V 生成（自研发射器，零第三方依赖）。
//!
//! # 与 `spirv_bg` 的关系
//!
//! 沿用同一套最小发射器骨架（段序 / id 分配 / 常量去重 / `NoContraction`），
//! 差别只在**语义件**：背景 kernel 是逐点算子（无循环），本 kernel 是
//! 「每像素遍历该行的解析边片段」（一个结构化动态循环 + 6 字段片段读取）。
//! 所有 storage buffer 统一声明为 `array<u32>`：f32 片段/输出走 `OpBitcast`
//! 位型往返（无文本、无第二套舍入口径），一套指针类型覆盖四个绑定，发射器
//! 因此不必为 f32/u32 各发一份 RuntimeArray/Struct。
//!
//! # 逐式对应（判据 P1 的基础）
//!
//! 函数体逐句对应 [`crate::path_raster::segment_area`]：
//! `pos_area` 的三角/梯形分支全部用 `OpSelect`（无控制流分歧），
//! `tri_factor` 的 `Δ=0` 守卫用 `safe_den` 避免驱动侧除零，被 `Select` 丢弃的
//! 那一支不参与结果——CPU 参考里该分支同样直接返回 0.0。每条浮点算术都打
//! `NoContraction`，禁止 FMA 熔合改末位。
//!
//! # 扰动版（负控用，反假绿）
//!
//! [`build_path_module_with_one`] 允许把**输出增益常量** `1.0` 换成 `1.0 + ε`
//! （在写回覆盖率前对累加结果做一次 `OpFMul`）。产品版增益 = `1.0`，是 IEEE
//! 754 下的**逐位恒等**（`x·1.0 = x`，含 ±0 / 次正规数），因此不改变任何结果；
//! 而饱和短路会把大量像素收敛到「满覆盖 / 零」支路，若把扰动放在 `a1` 的
//! `−1` 位移上，这些支路根本不读它 —— 负控会假绿。改用输出增益则**每个非零
//! 像素必然改变**，判据必 FAIL，从而稳健证明「绿灯来自真实 kernel 执行」而非
//! CPU 兜底或恒真断言。
//!
//! 实现上 `one_value` 是 kernel 的**唯一「单位常量」**：`a1` 的 `−1` 位移、饱和
//! 阈值 `> 1`、`cov · one` 输出增益、以及 `tri_factor` 里被 Select 丢弃的除零
//! 保护位四处共用同一个 `f32_const(self.one_value)` 槽。产品版 = `1.0`（四处都
//! 是数学正确值 / 恒等），扰动版 = `1.001`（四处一起错）。去重后两版都只占
//! **一个**常量 id，其余 `0.0 / 0.5 / COORD_BIAS / u32` 常量逐字相同，因此
//! `fresh()` 次数相同、`next_id` 相同 —— SPIR-V 头 5 字逐字相同，只有那个常量
//! 的位型不同。产品路径永远用 [`build_path_module`]（`ε = 0`）。

use std::sync::OnceLock;

use super::spirv::{
    inst, DECORATION_ARRAY_STRIDE, DECORATION_BINDING, DECORATION_BUFFER_BLOCK,
    DECORATION_BUILT_IN, DECORATION_DESCRIPTOR_SET, DECORATION_NO_CONTRACTION,
    DECORATION_OFFSET, BUILTIN_GLOBAL_INVOCATION_ID, EXECUTION_MODE_LOCAL_SIZE,
    EXECUTION_MODEL_GLCOMPUTE, MAGIC, OP_ACCESS_CHAIN, OP_BITCAST, OP_BRANCH,
    OP_BRANCH_CONDITIONAL, OP_CAPABILITY, OP_COMPOSITE_EXTRACT, OP_CONSTANT,
    OP_CONVERT_U_TO_F, OP_DECORATE, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_F_ADD,
    OP_F_DIV, OP_F_MUL, OP_F_NEGATE, OP_F_ORD_EQUAL, OP_F_ORD_GREATER_THAN,
    OP_F_ORD_GREATER_THAN_EQUAL, OP_F_ORD_LESS_THAN, OP_F_SUB, OP_FUNCTION,
    OP_FUNCTION_END, OP_I_ADD, OP_I_MUL, OP_LABEL, OP_LOAD, OP_LOGICAL_OR,
    OP_MEMBER_DECORATE, OP_MEMORY_MODEL, OP_RETURN, OP_SELECT, OP_SELECTION_MERGE,
    OP_STORE, OP_TYPE_BOOL, OP_TYPE_FLOAT, OP_TYPE_FUNCTION, OP_TYPE_INT,
    OP_TYPE_POINTER, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_VECTOR,
    OP_TYPE_VOID, OP_U_GREATER_THAN_EQUAL, OP_U_LESS_THAN, OP_VARIABLE, SCHEMA,
    STORAGE_CLASS_FUNCTION, STORAGE_CLASS_INPUT, STORAGE_CLASS_UNIFORM, VERSION_1_0,
    OP_LOOP_MERGE,
};
use crate::path_raster::COORD_BIAS;

/// 路径光栅 kernel 的 SPIR-V 模块名。
pub const MODULE_PATH_RASTER: &str = "spv_path_raster";
/// 入口名（`OpEntryPoint` / WGSL `fn`）。
pub const ENTRY_PATH_RASTER: &str = "path_main";
/// 工作组尺寸（与 `path_raster.wgsl` 的 `@workgroup_size` 一致）。
pub const WORKGROUP_PATH_RASTER: [u32; 3] = [8, 8, 1];
/// 片段字段数（与 `SEGMENT_FLOATS` 同源，写死为 SPIR-V 常量）。
const SEGMENT_FLOATS: u32 = 6;

const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
const FUNCTION_CONTROL_NONE: u32 = 0;
const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;

/// 生成（并缓存）产品版字流：`1.0` 常量逐位精确。
pub fn path_words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(build_path_module)
}

/// 构建产品版模块（确定性：同输入两次构建逐字相同）。
pub fn build_path_module() -> Vec<u32> {
    build_path_module_with_one(1.0)
}

/// 构建模块，`one` 为 kernel 的**唯一单位常量**。
///
/// 它被四处共用，且四处都通过 `f32_const(self.one_value)` **同一个去重常量槽**：
/// ① `a1` 的 `u0 − one` 位移；② 饱和阈值 `> one`；③ `tri_factor` 的 safe_den
/// 保护位（那一支被 Select 丢弃，值无关紧要）；④ 写回前输出增益 `cov · one`。
///
/// 产品调用恒传 `1.0`：四处都是数学正确值 / IEEE 逐位恒等（`x·1.0 = x`），
/// P1 逐字节判据成立。负控传 `1.0 + ε`：④使**每个非零像素**都改变 ⇒ 判据必
/// FAIL（反假绿）。由于产品版与扰动版只改这一个常量的位型、**不增删任何 id**，
/// `next_id`/bound 相同 ⇒ SPIR-V 头 5 字逐字相同（见模块头注释）。
pub fn build_path_module_with_one(one: f32) -> Vec<u32> {
    let mut e = Emitter::new(one);
    e.build_body();
    e.assemble()
}

fn push(dst: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    dst.push(inst((operands.len() + 1) as u32, opcode));
    dst.extend_from_slice(operands);
}

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

/// 最小 SPIR-V 发射器：九个逻辑段按规范顺序累积。
struct Emitter {
    caps: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    constants: Vec<u32>,
    globals: Vec<u32>,
    funcs: Vec<u32>,
    next_id: u32,
    t_void: u32,
    t_bool: u32,
    t_f32: u32,
    t_u32: u32,
    t_v3u32: u32,
    p_in_v3: u32,
    p_u32: u32,
    p_buf: u32,
    pf_u32: u32,
    pf_f32: u32,
    t_fn: u32,
    v_gid: u32,
    v_params: u32,
    v_segs: u32,
    v_rows: u32,
    v_out: u32,
    fn_main: u32,
    one_value: f32,
    f32_consts: Vec<(u32, u32)>,
    u32_consts: Vec<(u32, u32)>,
}

impl Emitter {
    fn fresh(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn new(one_value: f32) -> Self {
        let mut e = Emitter {
            caps: Vec::new(),
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
            p_buf: 0,
            pf_u32: 0,
            pf_f32: 0,
            t_fn: 0,
            v_gid: 0,
            v_params: 0,
            v_segs: 0,
            v_rows: 0,
            v_out: 0,
            fn_main: 0,
            one_value,
            f32_consts: Vec::new(),
            u32_consts: Vec::new(),
        };
        e.declare_prologue();
        e.declare_types();
        e.declare_globals();
        e
    }

    fn declare_prologue(&mut self) {
        push(&mut self.caps, OP_CAPABILITY, &[CAPABILITY_SHADER]);
        push(
            &mut self.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        self.fn_main = self.fresh();
        self.v_gid = self.fresh();
        self.v_params = self.fresh();
        self.v_segs = self.fresh();
        self.v_rows = self.fresh();
        self.v_out = self.fresh();

        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, self.fn_main];
        push_str(&mut ops, ENTRY_PATH_RASTER);
        ops.push(self.v_gid);
        push(&mut self.entry_points, OP_ENTRY_POINT, &ops);
        push(
            &mut self.entry_points,
            OP_EXECUTION_MODE,
            &[
                self.fn_main,
                EXECUTION_MODE_LOCAL_SIZE,
                WORKGROUP_PATH_RASTER[0],
                WORKGROUP_PATH_RASTER[1],
                WORKGROUP_PATH_RASTER[2],
            ],
        );
        push(
            &mut self.annotations,
            OP_DECORATE,
            &[self.v_gid, DECORATION_BUILT_IN, BUILTIN_GLOBAL_INVOCATION_ID],
        );
        let bindings = [
            (self.v_params, 0u32),
            (self.v_segs, 1),
            (self.v_rows, 2),
            (self.v_out, 3),
        ];
        for (v, b) in bindings {
            push(&mut self.annotations, OP_DECORATE, &[v, DECORATION_DESCRIPTOR_SET, 0]);
            push(&mut self.annotations, OP_DECORATE, &[v, DECORATION_BINDING, b]);
        }
    }

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
        push(&mut self.types, OP_TYPE_VECTOR, &[self.t_v3u32, self.t_u32, 3]);

        self.p_in_v3 = self.fresh();
        push(&mut self.types, OP_TYPE_POINTER, &[self.p_in_v3, STORAGE_CLASS_INPUT, self.t_v3u32]);
        self.p_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_POINTER, &[self.p_u32, STORAGE_CLASS_UNIFORM, self.t_u32]);

        let rt_arr = self.fresh();
        push(&mut self.types, OP_TYPE_RUNTIME_ARRAY, &[rt_arr, self.t_u32]);
        let t_struct = self.fresh();
        push(&mut self.types, OP_TYPE_STRUCT, &[t_struct, rt_arr]);
        self.p_buf = self.fresh();
        push(&mut self.types, OP_TYPE_POINTER, &[self.p_buf, STORAGE_CLASS_UNIFORM, t_struct]);

        self.pf_u32 = self.fresh();
        push(&mut self.types, OP_TYPE_POINTER, &[self.pf_u32, STORAGE_CLASS_FUNCTION, self.t_u32]);
        self.pf_f32 = self.fresh();
        push(&mut self.types, OP_TYPE_POINTER, &[self.pf_f32, STORAGE_CLASS_FUNCTION, self.t_f32]);

        self.t_fn = self.fresh();
        push(&mut self.types, OP_TYPE_FUNCTION, &[self.t_fn, self.t_void]);

        push(&mut self.annotations, OP_DECORATE, &[t_struct, DECORATION_BUFFER_BLOCK]);
        push(&mut self.annotations, OP_MEMBER_DECORATE, &[t_struct, 0, DECORATION_OFFSET, 0]);
        push(&mut self.annotations, OP_DECORATE, &[rt_arr, DECORATION_ARRAY_STRIDE, 4]);
    }

    fn declare_globals(&mut self) {
        push(&mut self.globals, OP_VARIABLE, &[self.p_in_v3, self.v_gid, STORAGE_CLASS_INPUT]);
        for v in [self.v_params, self.v_segs, self.v_rows, self.v_out] {
            push(&mut self.globals, OP_VARIABLE, &[self.p_buf, v, STORAGE_CLASS_UNIFORM]);
        }
    }

    // ---- 常量 ----

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

    // ---- 浮点指令（全部 NoContraction） ----

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
    fn fneg(&mut self, a: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_F_NEGATE, &[self.t_f32, id, a]);
        push(&mut self.annotations, OP_DECORATE, &[id, DECORATION_NO_CONTRACTION]);
        id
    }

    fn cmp(&mut self, opcode: u32, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, opcode, &[self.t_bool, id, a, b]);
        id
    }

    fn select_f32(&mut self, cond: u32, t: u32, f: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_SELECT, &[self.t_f32, id, cond, t, f]);
        id
    }

    fn bitcast_f32(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_BITCAST, &[self.t_f32, id, x]);
        id
    }

    fn bitcast_u32(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_BITCAST, &[self.t_u32, id, x]);
        id
    }

    // ---- 整数 / 内存 ----

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
    fn ugte(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_U_GREATER_THAN_EQUAL, &[self.t_bool, id, a, b]);
        id
    }
    fn ult(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_U_LESS_THAN, &[self.t_bool, id, a, b]);
        id
    }
    fn lor(&mut self, a: u32, b: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_LOGICAL_OR, &[self.t_bool, id, a, b]);
        id
    }
    fn u2f(&mut self, x: u32) -> u32 {
        let id = self.fresh();
        push(&mut self.funcs, OP_CONVERT_U_TO_F, &[self.t_f32, id, x]);
        id
    }

    /// `slot(v, index)`：读 `v` 的 runtime array 第 `index` 个 u32。
    fn slot(&mut self, v: u32, index: u32) -> u32 {
        let c0 = self.u32_const(0);
        let ptr = self.fresh();
        push(&mut self.funcs, OP_ACCESS_CHAIN, &[self.p_u32, ptr, v, c0, index]);
        let id = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_u32, id, ptr]);
        id
    }

    fn store_out(&mut self, index: u32, value: u32) {
        let c0 = self.u32_const(0);
        let ptr = self.fresh();
        push(&mut self.funcs, OP_ACCESS_CHAIN, &[self.p_u32, ptr, self.v_out, c0, index]);
        push(&mut self.funcs, OP_STORE, &[ptr, value]);
    }

    // ---- 语义件 ----

    /// `pos_area(u0,u1,h,fac)`：梯形（两端非负）或三角形（`fac·max(max(u0,u1),0)²`）。
    fn pos_area(&mut self, u0: u32, u1: u32, h: u32, fac: u32) -> u32 {
        let half = self.f32_const(0.5);
        let zero = self.f32_const(0.0);
        let trap0 = self.fadd(u0, u1);
        let trap1 = self.fmul(trap0, half);
        let trap = self.fmul(trap1, h);

        let gt = self.cmp(OP_F_ORD_GREATER_THAN, u0, u1);
        let pmax = self.select_f32(gt, u0, u1);
        let pgt = self.cmp(OP_F_ORD_GREATER_THAN, pmax, zero);
        let pp = self.select_f32(pgt, pmax, zero);
        let tri0 = self.fmul(fac, pp);
        let tri = self.fmul(tri0, pp);

        let lt = self.cmp(OP_F_ORD_LESS_THAN, u0, u1);
        let pmin = self.select_f32(lt, u0, u1);
        let nonneg = self.cmp(OP_F_ORD_GREATER_THAN_EQUAL, pmin, zero);
        self.select_f32(nonneg, trap, tri)
    }

    /// `tri_factor(u0,u1,h)`：`0.5·h/|u0−u1|`，`Δ=0` 时为 0（用 safe_den 避免除零）。
    fn tri_factor(&mut self, u0: u32, u1: u32, h: u32) -> u32 {
        let zero = self.f32_const(0.0);
        // safe_den 仅用于 `Δ=0` 时不产生非有限值；那一支的结果被末尾 Select 丢弃，
        // 值无关紧要。刻意复用 `self.one_value`（与输出增益 / a1 位移同一个去重
        // 常量槽），保证产品版与扰动版分配的常量 id 数完全一致（头 5 字相同）。
        let one = self.f32_const(self.one_value);
        let half = self.f32_const(0.5);
        let den = self.fsub(u0, u1);
        let neg = self.fneg(den);
        let is_neg = self.cmp(OP_F_ORD_LESS_THAN, den, zero);
        let abs_den = self.select_f32(is_neg, neg, den);
        let den_zero = self.cmp(OP_F_ORD_EQUAL, den, zero);
        let safe_den = self.select_f32(den_zero, one, abs_den);
        let num = self.fmul(half, h);
        let div = self.fdiv(num, safe_den);
        self.select_f32(den_zero, zero, div)
    }

    /// 单片段的面积贡献（与 `path_raster::segment_area` 逐句同式）。
    ///
    /// 片段字段 `(h, 0, base, fp, fq, delta)`：字段 0 直接是 y 子区间高度 `h`
    /// （CPU 已用 f64 算好、只舍入一次），字段 1 保留占位恒 0。**饱和短路**
    /// （`bias_px+lo > 1 ⇒ delta·h` / `bias_px+hi < 0 ⇒ 0`）是精确结论，把
    /// f32 大数消去挡在一般路径之外；一般路径只在 `u∈(−3,5)` 的小量上做
    /// `a0−a1`。全部用 `OpSelect`（无控制流分歧），与 CPU 参考的「三目 + 提前
    /// 返回」逐位等价。`one` 同时充当 `a1` 的 `−1` 位移与饱和阈值（产品恒 1.0）。
    fn segment_area(&mut self, seg_base: u32, gx_f: u32, one: u32) -> u32 {
        let v_segs = self.v_segs;
        let c = |e: &mut Emitter, k: u32| -> u32 {
            let kc = e.u32_const(k);
            let idx = e.uadd(seg_base, kc);
            let raw = e.slot(v_segs, idx);
            e.bitcast_f32(raw)
        };
        let h = c(self, 0);
        let base = c(self, 2);
        let fp = c(self, 3);
        let fq = c(self, 4);
        let delta = c(self, 5);

        let bias = self.f32_const(COORD_BIAS);
        let zero = self.f32_const(0.0);
        let bias0 = self.fsub(base, bias);
        let bias_px = self.fsub(bias0, gx_f);

        // lo = min(fp, fq)，hi = max(fp, fq)：一次比较 + 两个 OpSelect。
        let fp_lt = self.cmp(OP_F_ORD_LESS_THAN, fp, fq);
        let lo = self.select_f32(fp_lt, fp, fq);
        let hi = self.select_f32(fp_lt, fq, fp);

        let u0 = self.fadd(bias_px, fp);
        let u1 = self.fadd(bias_px, fq);
        let fac = self.tri_factor(u0, u1, h);
        let a0 = self.pos_area(u0, u1, h, fac);
        let u0m = self.fsub(u0, one);
        let u1m = self.fsub(u1, one);
        let a1 = self.pos_area(u0m, u1m, h, fac);
        let area = self.fsub(a0, a1);
        let normal = self.fmul(delta, area);
        let saturated = self.fmul(delta, h);

        // 饱和短路：bias_lo = bias_px + lo > one ⇒ saturated；
        //           bias_hi = bias_px + hi < 0   ⇒ 0；否则 normal。
        let bias_lo = self.fadd(bias_px, lo);
        let bias_hi = self.fadd(bias_px, hi);
        let sat_cond = self.cmp(OP_F_ORD_GREATER_THAN, bias_lo, one);
        let zero_cond = self.cmp(OP_F_ORD_LESS_THAN, bias_hi, zero);
        let inner = self.select_f32(zero_cond, zero, normal);
        self.select_f32(sat_cond, saturated, inner)
    }

    /// ⑥ 函数体：一像素一 invocation，遍历该行片段。
    fn build_body(&mut self) {
        let l_entry = self.fresh();
        let l_guard_merge = self.fresh();
        let l_body = self.fresh();
        let l_loop_hdr = self.fresh();
        let l_loop_body = self.fresh();
        let l_loop_cont = self.fresh();
        let l_loop_merge = self.fresh();
        let v_i = self.fresh();
        let v_cov = self.fresh();

        let t_void = self.t_void;
        let t_fn = self.t_fn;
        let fn_main = self.fn_main;
        push(&mut self.funcs, OP_FUNCTION, &[t_void, fn_main, FUNCTION_CONTROL_NONE, t_fn]);
        push(&mut self.funcs, OP_LABEL, &[l_entry]);

        // 函数局部变量必须紧随入口 label，先于一切其它指令（SPIR-V 规范）。
        push(&mut self.funcs, OP_VARIABLE, &[self.pf_u32, v_i, STORAGE_CLASS_FUNCTION]);
        push(&mut self.funcs, OP_VARIABLE, &[self.pf_f32, v_cov, STORAGE_CLASS_FUNCTION]);

        let gid = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_v3u32, gid, self.v_gid]);
        let gx = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[self.t_u32, gx, gid, 0]);
        let gy = self.fresh();
        push(&mut self.funcs, OP_COMPOSITE_EXTRACT, &[self.t_u32, gy, gid, 1]);

        let c0 = self.u32_const(0);
        let c1 = self.u32_const(1);
        let c2 = self.u32_const(2);
        let c6 = self.u32_const(SEGMENT_FLOATS);
        let zero_f = self.f32_const(0.0);
        let one_f = self.f32_const(self.one_value);

        let w = self.slot(self.v_params, c0);
        let h = self.slot(self.v_params, c1);

        // 越界守卫（尾组调用丢弃）。
        let cx = self.ugte(gx, w);
        let cy = self.ugte(gy, h);
        let oob = self.lor(cx, cy);
        push(&mut self.funcs, OP_SELECTION_MERGE, &[l_guard_merge, SELECTION_CONTROL_NONE]);
        push(&mut self.funcs, OP_BRANCH_CONDITIONAL, &[oob, l_guard_merge, l_body]);
        push(&mut self.funcs, OP_LABEL, &[l_body]);

        // rows[2*gy] = start, rows[2*gy+1] = count
        let gy2 = self.umul(gy, c2);
        let gy2p1 = self.uadd(gy2, c1);
        let start = self.slot(self.v_rows, gy2);
        let count = self.slot(self.v_rows, gy2p1);

        push(&mut self.funcs, OP_STORE, &[v_i, c0]);
        push(&mut self.funcs, OP_STORE, &[v_cov, zero_f]);
        push(&mut self.funcs, OP_BRANCH, &[l_loop_hdr]);

        // --- 循环头 ---
        push(&mut self.funcs, OP_LABEL, &[l_loop_hdr]);
        let i_cur = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_u32, i_cur, v_i]);
        let in_range = self.ult(i_cur, count);
        push(&mut self.funcs, OP_LOOP_MERGE, &[l_loop_merge, l_loop_cont, LOOP_CONTROL_NONE]);
        push(&mut self.funcs, OP_BRANCH_CONDITIONAL, &[in_range, l_loop_body, l_loop_merge]);

        // --- 循环体 ---
        push(&mut self.funcs, OP_LABEL, &[l_loop_body]);
        let si = self.uadd(start, i_cur);
        let seg_base = self.umul(si, c6);
        let gx_f = self.u2f(gx);
        let term = self.segment_area(seg_base, gx_f, one_f);
        let cov_cur = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_f32, cov_cur, v_cov]);
        let cov_next = self.fadd(cov_cur, term);
        push(&mut self.funcs, OP_STORE, &[v_cov, cov_next]);
        push(&mut self.funcs, OP_BRANCH, &[l_loop_cont]);

        // --- continue ---
        push(&mut self.funcs, OP_LABEL, &[l_loop_cont]);
        let i_next = self.uadd(i_cur, c1);
        push(&mut self.funcs, OP_STORE, &[v_i, i_next]);
        push(&mut self.funcs, OP_BRANCH, &[l_loop_hdr]);

        // --- merge：写输出 ---
        push(&mut self.funcs, OP_LABEL, &[l_loop_merge]);
        let cov = self.fresh();
        push(&mut self.funcs, OP_LOAD, &[self.t_f32, cov, v_cov]);
        // 输出增益：产品 one=1.0 ⇒ cov·1.0 逐位恒等（P1 不受影响）；负控
        // one=1.001 ⇒ 每个非零覆盖率像素都改变，逐字节判据必 FAIL（反假绿）。
        // 与 a1 位移 / 饱和阈值 / safe_den 共用同一个去重常量槽，id 数不变。
        let cov_out = self.fmul(cov, one_f);
        let cov_bits = self.bitcast_u32(cov_out);
        let row = self.umul(gy, w);
        let idx = self.uadd(row, gx);
        self.store_out(idx, cov_bits);
        push(&mut self.funcs, OP_BRANCH, &[l_guard_merge]);

        push(&mut self.funcs, OP_LABEL, &[l_guard_merge]);
        push(&mut self.funcs, OP_RETURN, &[]);
        push(&mut self.funcs, OP_FUNCTION_END, &[]);
    }

    fn assemble(self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        out.push(MAGIC);
        out.push(VERSION_1_0);
        out.push(0);
        out.push(self.next_id);
        out.push(SCHEMA);
        for seg in [
            self.caps,
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

// ---------------------------------------------------------------------------
// 测试（无设备即可执行）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::spirv::{disassemble, validate, SpvExpect};

    #[test]
    fn path_module_is_structurally_valid() {
        let words = path_words();
        let summary = validate(
            words,
            &SpvExpect {
                entry: ENTRY_PATH_RASTER,
                local_size: WORKGROUP_PATH_RASTER,
                bindings: &[0, 1, 2, 3],
            },
        )
        .expect("路径光栅 SPIR-V 须通过结构验证");
        assert_eq!(summary.entry, ENTRY_PATH_RASTER);
        assert_eq!(summary.local_size, [8, 8, 1]);
        assert_eq!(summary.bindings, vec![0, 1, 2, 3]);
        assert_eq!(summary.functions, 1);
        let mut i = 5usize;
        let mut total = 0usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            assert!(wc > 0, "零字长指令 @{i}");
            total += wc;
            i += wc;
        }
        assert_eq!(5 + total, words.len(), "指令字长之和须铺满字流");
        for op in [OP_LOOP_MERGE, OP_SELECT, OP_F_DIV, OP_BITCAST, OP_STORE, OP_LOAD, OP_ACCESS_CHAIN] {
            assert!(summary.opcodes.contains(&op), "缺指令 {}", crate::vulkan::spirv::opcode_name(op));
        }
        let text = disassemble(words);
        assert!(!text.contains("OpUnknown"), "白名单漏了 OpCode：\n{text}");
        println!("路径 SPIR-V：{} 字 / {} 条指令 / bound={}", words.len(), summary.instruction_count, summary.bound);
    }

    #[test]
    fn build_is_deterministic_and_perturbation_is_distinct() {
        assert_eq!(build_path_module(), build_path_module());
        assert_eq!(path_words(), build_path_module().as_slice());
        let perturbed = build_path_module_with_one(1.001);
        assert_ne!(perturbed, build_path_module(), "扰动版必须与产品版不同（负控前提）");
        assert_eq!(perturbed[..5], build_path_module()[..5], "头部不变");
    }

    #[test]
    fn module_identity_is_stable() {
        assert_eq!(MODULE_PATH_RASTER, "spv_path_raster");
        assert_eq!(ENTRY_PATH_RASTER, "path_main");
        assert_eq!(WORKGROUP_PATH_RASTER, [8, 8, 1]);
        assert_eq!(
            crate::vulkan::spirv::module(MODULE_PATH_RASTER).map(|w| w.len()),
            Some(path_words().len())
        );
    }
}
