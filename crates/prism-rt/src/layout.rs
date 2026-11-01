//! W0-A 定桩的**静态布局门禁**（只在 `cfg(test)` 下存在）。
//!
//! # 为什么要有这个模块
//!
//! 选型报告 §11 的定桩请求写的是「新增字段只用 `f32` / `u32` / `[f32;N]` /
//! `[u32;N]`；存储数组里**禁止** `vec3<f32>`」。这句话若只写在文档里，
//! 它会以两种方式失效：
//!
//! 1. 有人把 `uv: Vec<[f32; 2]>` 改成 `Vec<Vec3>` —— `size_of` 从 8 变 12，
//!    slot 断言确实会拦；
//! 2. 但更常见的是**新加**一个字段，谁都没想起来它也得遵守类型纪律
//!    —— 没有任何 `size_of` 断言会拦，因为新结构的尺寸本来就没被钉过。
//!
//! 第 2 类才是 V2.2 `vec3<f32>` 事故的真正形态：门禁"以为在那儿"，
//! 实际只覆盖了当时想到的那几个结构体。故本模块的门禁**从源码文本读**，
//! 遍历**白名单里每一个字段的声明类型**逐条判定，新增字段必须先进白名单
//! 才能过 —— 想偷偷加一个 `Vec3` 字段，得先改白名单，而白名单里写着规矩。
//!
//! # 与 `prism-gpu/web/abi.rs` 同款门禁的关系
//!
//! 那里扫的是 **WGSL 源码**（真机 ABI 侧，`gi_compute.wgsl` 的四个布局
//! 结构体）；这里扫的是 **Rust 结构体**（CPU 侧存储）。两边不能合并：
//! `prism-rt` 不依赖 `prism-gpu`（反向依赖会成环），且两者失败含义不同
//! ——WGSL 侧错了是驱动读到垃圾，Rust 侧错了是打包进缓冲的字节错位。
//!
//! # 它抓不住什么（诚实边界）
//!
//! * 抓不住「驱动不按规范实现」；
//! * 抓不住「顺序对但打包代码写错偏移」（那是 `scene_abi` 编码器单测的活）；
//! * 定长数组的**长度**用常量名（如 `[f32; INSTANCE_AFFINE_WORDS]`）时，
//!   文本门禁只验"是个定长数组"，验不了常量值 —— 那一层由
//!   `size_of` / `offset_of!` 编译期断言兜（更强：它看的是真实布局）。
//!
//! 它保证的是：**新增存储字段的声明类型落在保守集合里**。

#[cfg(test)]
mod tests {
    /// 一个被门禁覆盖的字段：`(文件, 结构体, 字段, 允许的类型)`。
    ///
    /// 「允许的类型」是**完整**声明类型（不做前缀匹配）：门禁比的是等价，
    /// 少一个空格都算改过。
    const GUARDED_FIELDS: &[(&str, &str, &str, &str)] = &[
        // ---- Scene（W1-A 新增三字段）----
        ("integrator.rs", "Scene", "textures", "Vec<TextureRef>"),
        ("integrator.rs", "Scene", "instances", "Vec<Instance>"),
        ("integrator.rs", "Scene", "media", "Vec<MediumRegion>"),
        // ---- Mesh（W1-A 新增两字段：顶点域 uv + 预烘焙切线）----
        ("integrator.rs", "Mesh", "uv", "Vec<[f32; 2]>"),
        ("integrator.rs", "Mesh", "tangent", "Vec<[f32; 4]>"),
        // ---- Bsdf（W1-A 新增一字段：四张贴图句柄）----
        ("bsdf.rs", "Bsdf", "maps", "[u32; MAP_SLOTS]"),
        // ---- Light::Environment（W1-A 新增字段）----
        ("light.rs", "Light", "map", "u32"),
    ];

    /// 允许的标量类型。
    const PRIMITIVES: &[&str] = &["u32", "f32"];

    /// 允许作为 `Vec<…>` 元素的**具名结构体**。
    ///
    /// 只收"自身字段已被逐条检查过"的结构体：
    /// * [`crate::texture::TextureRef`] / [`crate::instance::Instance`] 由
    ///   `every_field_of_new_structs_is_conservative` 全量检查；
    /// * [`crate::integrator::MediumRegion`] 是既有结构体（W1-A 之前就在），
    ///   本段不动它（`media` 字段只是把它装进 `Vec`）。
    const STRUCT_LEAVES: &[&str] = &["TextureRef", "Instance", "MediumRegion"];

    /// 判 `[T; N]` 里的 `N` 是否可接受：字面整数，或常量名。
    fn count_is_ok(count: &str) -> bool {
        let c = count.trim();
        if !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit()) {
            return true;
        }
        // 常量名（`MAP_SLOTS` / `INSTANCE_AFFINE_WORDS`）：长度由 `size_of`
        // 断言兜，这里只要求它是合法标识符（防 `[f32; ]` 之类的残缺声明）。
        !c.is_empty() && c.chars().all(|ch| ch.is_alphanumeric() || ch == '_')
    }

    /// 保守类型判定（**核心规则**）。
    ///
    /// `allow_struct_leaves = true` 时接受 [`STRUCT_LEAVES`] 里的具名结构体
    /// （用于 `Vec<TextureRef>` 这类"存储数组"）；`false` 时只接受标量与定长
    /// 数组（用于新结构体自身的字段）。
    fn check(ty: &str, allow_struct_leaves: bool) -> bool {
        let t = ty.trim();
        // `Vec<T>`：存储数组 —— 递归判元素。
        if let Some(inner) = t.strip_prefix("Vec<").and_then(|r| r.strip_suffix('>')) {
            return check(inner, allow_struct_leaves);
        }
        // `[T; N]`：定长数组 —— 递归判元素 + 判长度。
        if let Some(inner) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            let (elem, count) = match inner.split_once(';') {
                Some(x) => x,
                None => return false, // 切片 `[T]`：长度不定，stride 无从钉起
            };
            let elem = elem.trim();
            // **`[f32; 3]` 就是 Rust 侧的 `vec3<f32>`，一律禁**。
            // V2.2 事故形态：三元组进存储布局 ⇒ 真机驱动的 size≠12 ⇒
            // 后续字段偏移连锁错位。三元组请用 `[f32; 4]`（w 补 0）。
            if elem == "f32" && count.trim() == "3" {
                return false;
            }
            return check(elem, allow_struct_leaves) && count_is_ok(count);
        }
        if PRIMITIVES.contains(&t) {
            return true;
        }
        allow_struct_leaves && STRUCT_LEAVES.contains(&t)
    }

    /// 严格判定：只许标量与定长数组（新结构体自身字段用）。
    fn type_is_conservative(ty: &str) -> bool {
        check(ty, false)
    }

    /// 宽松判定：额外允许 [`STRUCT_LEAVES`]（`Scene`/`Mesh` 的容器字段用）。
    fn type_is_conservative_container(ty: &str) -> bool {
        check(ty, true)
    }

    /// 取源码里 `struct <name> { … }` 或 `enum <name> { … }` 的**顶层字段声明**。
    ///
    /// 只取 `name: type` 形态（enum 里是变体内部字段），注释先剥掉——
    /// 本仓注释里成篇引用 `vec3<f32>` 与类型表，不剥会把文档当声明解析
    /// （`prism-gpu/web/abi.rs` 的同款门禁与 `tools/wgsl_lint.mjs` 都踩过这个坑）。
    fn field_types(src: &str, name: &str) -> Vec<(String, String)> {
        // 剥 `//` 与 `///` 行注释。
        let stripped: String = src
            .lines()
            .map(|l| if l.trim_start().starts_with("//") { "" } else { l })
            .collect::<Vec<_>>()
            .join("\n");

        let head = match stripped.find(&format!("struct {name} ")) {
            Some(i) => i,
            None => match stripped.find(&format!("enum {name} ")) {
                Some(i) => i,
                None => panic!("源码里找不到 `struct {name}` / `enum {name}`"),
            },
        };
        let open = stripped[head..]
            .find('{')
            .map(|i| head + i)
            .expect("结构体缺 `{`");
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in stripped[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &stripped[open + 1..close.expect("结构体括号不配平")];
        let mut out = Vec::new();
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((lhs, rhs)) = line.split_once(':') {
                // 去掉可见性前缀（`pub` / `pub(crate)`）：取 lhs 的最后一个词。
                let fname = lhs.trim().split_whitespace().last().unwrap_or("");
                if fname.is_empty() || !fname.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                let ty = rhs.trim().trim_end_matches(',').trim();
                if ty.is_empty() {
                    continue;
                }
                out.push((fname.to_string(), ty.to_string()));
            }
        }
        out
    }

    /// 门禁主体：白名单里的每个字段，其声明类型必须在保守集合里。
    #[test]
    fn new_storage_fields_use_conservative_types_only() {
        let sources: &[(&str, &str)] = &[
            ("integrator.rs", include_str!("integrator.rs")),
            ("bsdf.rs", include_str!("bsdf.rs")),
            ("light.rs", include_str!("light.rs")),
        ];
        for (file, owner, field, want_ty) in GUARDED_FIELDS {
            let src = sources
                .iter()
                .find(|(f, _)| f == file)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| panic!("门禁引用了未知文件 {file}"));
            let fields = field_types(src, owner);
            let got = fields
                .iter()
                .find(|(n, _)| n == field)
                .unwrap_or_else(|| {
                    panic!(
                        "{file}: `{owner}` 找不到字段 `{field}`（改名了？白名单必须同步，\
                         否则门禁静默失效 —— 这正是 V2.2 事故的形态）"
                    )
                });
            assert_eq!(
                &got.1, want_ty,
                "{file}: `{owner}::{field}` 的声明类型变了（期望 `{want_ty}`，实际 `{}`）。\
                 存储字段只许 `f32` / `u32` / `[f32; N]` / `[u32; N]` 及其 `Vec`：\
                 `vec3<f32>` 的 size 在真机驱动上不是 12，`array` 的 stride 随之错位",
                got.1
            );
            assert!(
                type_is_conservative_container(&got.1),
                "{file}: `{owner}::{field}: {}` 不在保守类型集合里",
                got.1
            );
        }
    }

    /// 新增结构体 [`crate::texture::TextureRef`] / [`crate::instance::Instance`]
    /// 的**每一个**字段都必须过严格集合（不是白名单，是全量）。
    ///
    /// 理由是这两个结构体整条都是新的：没有"旧字段豁免"这回事。
    #[test]
    fn every_field_of_new_structs_is_conservative() {
        let cases: &[(&str, &str)] = &[("texture.rs", "TextureRef"), ("instance.rs", "Instance")];
        let sources: &[(&str, &str)] = &[
            ("texture.rs", include_str!("texture.rs")),
            ("instance.rs", include_str!("instance.rs")),
        ];
        for (file, name) in cases {
            let src = sources
                .iter()
                .find(|(f, _)| f == file)
                .map(|(_, s)| *s)
                .unwrap();
            let fields = field_types(src, name);
            assert!(
                !fields.is_empty(),
                "`{name}` 解析出 0 个字段（解析器与源码不同步 ⇒ 门禁静默失效）"
            );
            for (fname, ty) in fields {
                assert!(
                    type_is_conservative(&ty),
                    "{file}: `{name}::{fname}: {ty}` 不在保守类型集合里\
                     （只许 f32 / u32 / [f32; N] / [u32; N]）"
                );
            }
        }
    }

    /// 门禁自身会被"解析器悄悄失效"骗过：给一段**已知违规**的源码，
    /// 断言解析器确实解析出了该字段、且判定为不合规。
    ///
    /// 没有这条，上面两条测试可能在"什么都没扫到"的情况下全绿。
    #[test]
    fn gate_detects_known_violation() {
        let bad = r#"
pub struct Bogus {
    /// 注释里写 `vec3<f32>` 不该被当成声明
    pub uv: Vec<Vec3>,
    pub slots: Vec<[f32; 3]>,
    pub good: [u32; 4],
}
"#;
        let fields = field_types(bad, "Bogus");
        let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["uv", "slots", "good"], "解析器没解析出三个字段");
        let lookup = |n: &str| fields.iter().find(|(f, _)| f == n).unwrap().1.clone();
        assert!(!type_is_conservative(&lookup("uv")), "`Vec<Vec3>` 必须判违规");
        assert!(
            !type_is_conservative(&lookup("slots")),
            "`Vec<[f32; 3]>` 必须判违规（Rust 侧的 vec3 就是它）"
        );
        assert!(type_is_conservative(&lookup("good")), "`[u32; 4]` 必须判合规");
        // 容器模式：具名结构体叶子只认白名单。
        assert!(type_is_conservative_container("Vec<TextureRef>"));
        assert!(!type_is_conservative_container("Vec<Vec3>"), "`Vec3` 永不在白名单");
        assert!(!type_is_conservative_container("Vec<[f32; 3]>"));
        // 切片不许（stride 不定，无法钉）。
        assert!(!type_is_conservative("[f32]"));
        // 嵌套仍按递归判。
        assert!(type_is_conservative("Vec<[u32; 4]>"));
        assert!(type_is_conservative("[f32; 4]"), "三元组请用 [f32; 4]（w 补 0）");
        // 常量名长度可接受（值由 size_of 断言兜）。
        assert!(type_is_conservative("[f32; MAP_SLOTS]"));
    }

    /// `Bsdf` 必须是 `Copy`（选型报告 §2.2：不引入生命周期）。
    ///
    /// 用 trait bound 静态证：不满足 `Copy` 就编不过，而不是等运行期。
    #[test]
    fn bsdf_stays_copy_and_lifetime_free() {
        fn assert_copy<T: Copy + 'static>() {}
        assert_copy::<crate::bsdf::Bsdf>();
        assert_copy::<crate::instance::Instance>();
        assert_copy::<crate::texture::TextureRef>();
        // 句柄字段是 `[u32; 4]`（无生命周期参数）—— `Copy` 的结构性前提。
        assert_eq!(core::mem::size_of::<crate::bsdf::Bsdf>(), 72);
    }
}
