//! Buffer 布局与 kernel 描述（设备无关，std430 子集）。
//!
//! # 定位
//!
//! GPU 与 CPU 看到同一块内存的前提是**布局口径唯一**。本模块定义引擎
//! compute kernel 的绑定模型（binding → buffer → 元素类型/个数/对齐），
//! 并给出**确定性**的字节布局推导（std430 子集：标量 4 字节对齐、
//! `vec3`/`vec4` 16 字节对齐、storage buffer 绑定起点 16 字节对齐）。
//!
//! 该布局同时被三处消费：
//! * CPU 参考实现按它打包/解包（`crate::compute` 的 kernel 语义）；
//! * Vulkan backend 按它创建 buffer / 描述符（`crate::vulkan::compute`）；
//! * WGSL / MSL 一侧的 `var<storage>` 数组与之一一对应（`crate::shaders`）。
//!
//! 不引入 `bytemuck` 之类第三方 crate（D-15）：打包/解包全部手写。

use super::dispatch::WorkgroupSize;

/// storage buffer 绑定起点的对齐（std430 + Vulkan
/// `minStorageBufferOffsetAlignment` 的保守上界）。
pub const BINDING_ALIGNMENT: u32 = 16;

/// 元素类型（引擎 compute kernel 用到的封闭集合）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 32 位无符号整数。
    U32,
    /// 32 位浮点。
    F32,
    /// `vec3<f32>`（std430 下 12 字节数据 + 4 字节补齐 = 16）。
    Vec3,
    /// `vec4<f32>`（16 字节）。
    Vec4,
}

impl ElementType {
    /// 元素字节大小（stride）。
    pub fn size(self) -> u32 {
        match self {
            ElementType::U32 | ElementType::F32 => 4,
            ElementType::Vec3 => 16,
            ElementType::Vec4 => 16,
        }
    }

    /// 元素对齐（std430）。
    pub fn align(self) -> u32 {
        match self {
            ElementType::U32 | ElementType::F32 => 4,
            ElementType::Vec3 | ElementType::Vec4 => 16,
        }
    }

    /// 标量分量个数（打包辅助用）。
    pub fn components(self) -> usize {
        match self {
            ElementType::U32 | ElementType::F32 => 1,
            ElementType::Vec3 => 3,
            ElementType::Vec4 => 4,
        }
    }
}

/// 绑定的访问方向（决定 Vulkan access mask / WGSL 的 `read`/`read_write`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingAccess {
    /// 只读（`var<storage, read>`）。
    ReadOnly,
    /// 只写（`var<storage, read_write>` 的保守子集：kernel 不读回）。
    WriteOnly,
    /// 读写。
    ReadWrite,
}

impl BindingAccess {
    /// 稳定短名（诊断日志 / 生成 WGSL 访问限定符）。
    pub fn as_str(self) -> &'static str {
        match self {
            BindingAccess::ReadOnly => "read",
            BindingAccess::WriteOnly => "write",
            BindingAccess::ReadWrite => "read_write",
        }
    }
}

/// 一个 storage buffer 绑定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferSpec {
    /// 描述符 binding 号（`@group(0) @binding(n)`）。
    pub binding: u32,
    /// 元素类型。
    pub element: ElementType,
    /// 元素个数（定长；0 表示运行时数组长度由 dispatch/extent 决定）。
    pub elements: u32,
    /// 访问方向。
    pub access: BindingAccess,
}

impl BufferSpec {
    /// 定长绑定。
    pub const fn new(
        binding: u32,
        element: ElementType,
        elements: u32,
        access: BindingAccess,
    ) -> Self {
        BufferSpec {
            binding,
            element,
            elements,
            access,
        }
    }

    /// 该绑定的数据字节大小（不含尾部对齐补齐）。
    pub fn data_bytes(&self) -> u32 {
        self.element.size().saturating_mul(self.elements)
    }
}

/// kernel 描述（设备无关的「kernel 是什么」）。
///
/// `spv_module` 指向嵌入二进制的 SPIR-V 模块名（见
/// [`crate::vulkan::spirv`]）；`wgsl_source` 指向 WGSL 源名（见
/// [`crate::shaders`]）。两者都是**编译期**常量，无运行期文件 I/O。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDesc {
    /// kernel 名（跨后端稳定标识）。
    pub name: &'static str,
    /// 入口函数名（SPIR-V `OpEntryPoint` / WGSL `fn` / MSL `kernel void`）。
    pub entry: &'static str,
    /// 工作组尺寸。
    pub workgroup: WorkgroupSize,
    /// storage buffer 绑定表（按 binding 升序存储）。
    pub buffers: Vec<BufferSpec>,
    /// 嵌入二进制中的 SPIR-V 模块名。
    pub spv_module: &'static str,
    /// 对应的 WGSL 源名。
    pub wgsl_source: &'static str,
}

/// kernel 描述校验失败原因（封闭枚举，便于测试逐项断言）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    /// kernel 名为空。
    EmptyName,
    /// 入口名为空。
    EmptyEntry,
    /// 工作组尺寸含零维。
    InvalidWorkgroup,
    /// 绑定号重复。
    DuplicateBinding,
    /// 绑定号为 0 个元素。
    ZeroElements,
    /// 未声明任何绑定。
    NoBindings,
}

impl LayoutError {
    /// 稳定短名。
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutError::EmptyName => "empty-name",
            LayoutError::EmptyEntry => "empty-entry",
            LayoutError::InvalidWorkgroup => "invalid-workgroup",
            LayoutError::DuplicateBinding => "duplicate-binding",
            LayoutError::ZeroElements => "zero-elements",
            LayoutError::NoBindings => "no-bindings",
        }
    }
}

impl KernelDesc {
    /// 构造描述（不做校验；调用 [`KernelDesc::validate`]）。
    pub const fn new(
        name: &'static str,
        entry: &'static str,
        workgroup: WorkgroupSize,
        buffers: Vec<BufferSpec>,
        spv_module: &'static str,
        wgsl_source: &'static str,
    ) -> Self {
        KernelDesc {
            name,
            entry,
            workgroup,
            buffers,
            spv_module,
            wgsl_source,
        }
    }

    /// 结构校验：名字非空、workgroup 合法、binding 唯一且元素数非零。
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.name.is_empty() {
            return Err(LayoutError::EmptyName);
        }
        if self.entry.is_empty() {
            return Err(LayoutError::EmptyEntry);
        }
        if !self.workgroup.is_valid() {
            return Err(LayoutError::InvalidWorkgroup);
        }
        if self.buffers.is_empty() {
            return Err(LayoutError::NoBindings);
        }
        for (i, b) in self.buffers.iter().enumerate() {
            if b.elements == 0 {
                return Err(LayoutError::ZeroElements);
            }
            if self.buffers[..i].iter().any(|p| p.binding == b.binding) {
                return Err(LayoutError::DuplicateBinding);
            }
        }
        Ok(())
    }

    /// 按 binding 号取绑定。
    pub fn binding(&self, binding: u32) -> Option<&BufferSpec> {
        self.buffers.iter().find(|b| b.binding == binding)
    }

    /// 字节布局：[`BindingLayout`] 列表（按 binding 升序，确定性）。
    pub fn layout(&self) -> Vec<BindingLayout> {
        let mut sorted: Vec<&BufferSpec> = self.buffers.iter().collect();
        sorted.sort_by_key(|b| b.binding);
        let mut out = Vec::with_capacity(sorted.len());
        let mut cursor: u32 = 0;
        for b in sorted {
            // 绑定起点对齐到 BINDING_ALIGNMENT（std430 storage buffer 起点）。
            let offset = align_up(cursor, BINDING_ALIGNMENT);
            out.push(BindingLayout {
                binding: b.binding,
                offset,
                bytes: b.data_bytes(),
                access: b.access,
                element: b.element,
                elements: b.elements,
            });
            cursor = offset + b.data_bytes();
        }
        out
    }

    /// 全部绑定打包后的总字节数（含尾部对齐补齐到 [`BINDING_ALIGNMENT`]）。
    pub fn total_bytes(&self) -> u32 {
        let layout = self.layout();
        match layout.last() {
            None => 0,
            Some(last) => align_up(last.offset + last.bytes, BINDING_ALIGNMENT),
        }
    }

    /// 生成该 kernel 的 WGSL 绑定声明（与布局同源的单一权威）。
    ///
    /// 供 `crate::shaders` 的源宿主交叉核对；返回的字符串确定性（同输入
    /// 同输出），可直接做字节级比对。
    pub fn wgsl_bindings(&self) -> String {
        let mut out = String::new();
        for l in self.layout() {
            let ty = match l.element {
                ElementType::U32 => "u32",
                ElementType::F32 => "f32",
                ElementType::Vec3 => "vec3<f32>",
                ElementType::Vec4 => "vec4<f32>",
            };
            out.push_str(&format!(
                "@group(0) @binding({}) var<storage, {}> b{}: array<{}>;\n",
                l.binding,
                l.access.as_str(),
                l.binding,
                ty
            ));
        }
        out
    }
}

/// 单个绑定的字节布局项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingLayout {
    /// 绑定号。
    pub binding: u32,
    /// 相对 buffer 起点的字节偏移。
    pub offset: u32,
    /// 数据字节数。
    pub bytes: u32,
    /// 访问方向。
    pub access: BindingAccess,
    /// 元素类型。
    pub element: ElementType,
    /// 元素个数。
    pub elements: u32,
}

impl BindingLayout {
    /// 该绑定覆盖的字节区间 `[offset, offset + bytes)`。
    pub fn range(&self) -> (u32, u32) {
        (self.offset, self.offset + self.bytes)
    }
}

/// 向上对齐（`align` 须为 2 的幂且非零）。
pub fn align_up(v: u32, align: u32) -> u32 {
    if align == 0 {
        return v;
    }
    let mask = align - 1;
    (v + mask) & !mask
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn primary_ray_gen_desc() -> KernelDesc {
        KernelDesc::new(
            "rt_primary_ray_gen",
            "primary_ray_gen",
            WorkgroupSize::new(64, 1, 1),
            vec![
                BufferSpec::new(0, ElementType::Vec4, 2, BindingAccess::ReadOnly), // [eye, forward]
                BufferSpec::new(1, ElementType::Vec4, 4, BindingAccess::ReadOnly), // 视锥
                BufferSpec::new(2, ElementType::Vec4, 1024 * 4, BindingAccess::WriteOnly),
            ],
            "rt_primary_ray_gen",
            "rt_primary_ray_gen.wgsl",
        )
    }

    #[test]
    fn element_sizes_and_alignments_are_std430() {
        assert_eq!(ElementType::U32.size(), 4);
        assert_eq!(ElementType::F32.size(), 4);
        assert_eq!(ElementType::Vec3.size(), 16, "std430 vec3 占 16 字节槽");
        assert_eq!(ElementType::Vec4.size(), 16);
        assert_eq!(ElementType::Vec3.align(), 16);
        assert_eq!(ElementType::U32.align(), 4);
        assert_eq!(ElementType::Vec3.components(), 3);
    }

    #[test]
    fn layout_offsets_are_binding_aligned_and_non_overlapping() {
        let desc = primary_ray_gen_desc();
        assert!(desc.validate().is_ok());
        let layout = desc.layout();
        assert_eq!(layout.len(), 3);
        // 按 binding 升序。
        assert_eq!(
            layout.iter().map(|l| l.binding).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // 起点 16 字节对齐 + 区间不重叠。
        let mut prev_end = 0u32;
        for l in &layout {
            assert_eq!(l.offset % BINDING_ALIGNMENT, 0, "绑定起点须 16 字节对齐");
            assert!(l.offset >= prev_end, "绑定区间不得重叠");
            let (s, e) = l.range();
            assert_eq!(e - s, l.bytes);
            prev_end = e;
        }
        assert_eq!(layout[0].bytes, 32, "2 × vec4 = 32 字节");
        assert_eq!(layout[1].bytes, 64, "4 × vec4 = 64 字节");
        assert_eq!(layout[2].bytes, 1024 * 4 * 16);
        assert_eq!(desc.total_bytes() % BINDING_ALIGNMENT, 0);
        assert_eq!(
            desc.total_bytes(),
            align_up(prev_end, BINDING_ALIGNMENT),
            "总字节 = 末绑定末尾对齐到 16"
        );
    }

    #[test]
    fn validate_rejects_malformed_kernels() {
        let mut d = primary_ray_gen_desc();
        d.name = "";
        assert_eq!(d.validate(), Err(LayoutError::EmptyName));
        let mut d = primary_ray_gen_desc();
        d.entry = "";
        assert_eq!(d.validate(), Err(LayoutError::EmptyEntry));
        let mut d = primary_ray_gen_desc();
        d.workgroup = WorkgroupSize::new(0, 1, 1);
        assert_eq!(d.validate(), Err(LayoutError::InvalidWorkgroup));
        let mut d = primary_ray_gen_desc();
        d.buffers.clear();
        assert_eq!(d.validate(), Err(LayoutError::NoBindings));
        let mut d = primary_ray_gen_desc();
        d.buffers.push(BufferSpec::new(0, ElementType::F32, 1, BindingAccess::ReadOnly));
        assert_eq!(d.validate(), Err(LayoutError::DuplicateBinding));
        let mut d = primary_ray_gen_desc();
        d.buffers.push(BufferSpec::new(7, ElementType::F32, 0, BindingAccess::ReadOnly));
        assert_eq!(d.validate(), Err(LayoutError::ZeroElements));
        // 失败原因短名非空（诊断契约）。
        for e in [
            LayoutError::EmptyName,
            LayoutError::EmptyEntry,
            LayoutError::InvalidWorkgroup,
            LayoutError::DuplicateBinding,
            LayoutError::ZeroElements,
            LayoutError::NoBindings,
        ] {
            assert!(!e.as_str().is_empty());
        }
    }

    #[test]
    fn wgsl_bindings_reflect_layout() {
        let desc = primary_ray_gen_desc();
        let wgsl = desc.wgsl_bindings();
        assert!(wgsl.contains("@group(0) @binding(0) var<storage, read> b0: array<vec4<f32>>;"));
        assert!(wgsl.contains("@group(0) @binding(2) var<storage, write> b2: array<vec4<f32>>;"));
        // 确定性：同输入两遍一致。
        assert_eq!(desc.wgsl_bindings(), wgsl);
        // 绑定号全集齐备。
        for l in desc.layout() {
            assert!(wgsl.contains(&format!("@binding({})", l.binding)));
        }
    }

    #[test]
    fn align_up_basics() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(5, 0), 5, "align=0 时原样返回（不 panic）");
    }
}
