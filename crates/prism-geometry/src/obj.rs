//! OBJ/MTL 精简导入（白皮书 §12；架构 v2.4 S12 mesh 导入特解）。
//!
//! # 范围（从简）
//!
//! - 几何：`v`/`vn`/`vt` 顶点表 + `f` 面（四种顶点格式 `v`、`v/vt`、`v//vn`、`v/vt/vn`，
//!   1 基索引与负相对索引，>3 顶点扇形三角化），`(v, vt, vn)` 组合去重成统一顶点流
//!   （positions/normals/uvs 与 [`crate::gltf::GltfMesh`] 同构）。
//! - 分组：`o`/`g` 开新组，`usemtu` 在已有面之后出现时开新组（材质归属唯一）。
//! - 材质：[`parse_mtl`] 近似映射——`Kd` → base_color、`Ns` → roughness（Phong→GGX
//!   标准近似 `sqrt(2/(Ns+2))`）、metalness 固定 0 且报告注明；`map_Kd` 等纹理不导入。
//!
//! # 诚实纪律（宪章 C7/C20）
//!
//! 未支持的行/字段逐条聚合计数写入加载报告，**永不静默丢弃**；坏资产返回结构化
//! [`ObjError`]（三段式：在哪/什么事/怎么改），不 panic。
//!
//! # 确定性
//!
//! 输出顺序全部由文本行序决定；去重表只做查找不影响顺序；同输入同字节。

use crate::gltf::PbrMaterial;
use prism_math::Vec3;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// 结构化错误（宪章 C18/C19：三段式——在哪/什么事/怎么改）
// ─────────────────────────────────────────────────────────────────────────────

/// OBJ/MTL 解析错误。
#[derive(Debug, Clone, PartialEq)]
pub struct ObjError {
    /// 出错行号（1 基）。
    pub line: usize,
    /// 什么事。
    pub msg: String,
    /// 怎么改。
    pub hint: String,
}

impl std::fmt::Display for ObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prism-geometry/obj：解析失败\n  位置：第 {} 行\n  什么事：{}\n  建议：{}",
            self.line, self.msg, self.hint
        )
    }
}

impl std::error::Error for ObjError {}

// ─────────────────────────────────────────────────────────────────────────────
// 导入产物
// ─────────────────────────────────────────────────────────────────────────────

/// 导入的网格组（顶点流与 [`crate::gltf::GltfMesh`] 同构，材质近似映射）。
#[derive(Debug, Clone, PartialEq)]
pub struct ObjMesh {
    /// `o`/`g` 组名（无组名资产为 None）。
    pub name: Option<String>,
    /// 顶点位置。
    pub positions: Vec<Vec3>,
    /// 顶点法线（组内从未引用 `vn` 时为空；个别缺失时以零向量占位并出报告）。
    pub normals: Vec<Vec3>,
    /// 顶点 UV（组内从未引用 `vt` 时为空；个别缺失时以 (0,0) 占位并出报告）。
    pub uvs: Vec<[f32; 2]>,
    /// 三角形索引（>3 顶点面已扇形三角化）。
    pub indices: Vec<u32>,
    /// 材质（`usemtu` 查 [`parse_mtl`] 映射表；未命中或无 MTL 时为 None）。
    pub material: Option<PbrMaterial>,
}

/// `(v, vt, vn)` 组合去重键（0 基局部索引）。
type FaceKey = (u32, Option<u32>, Option<u32>);

/// 解析中的组。
struct Group {
    name: Option<String>,
    mat_name: Option<String>,
    dedup: HashMap<FaceKey, u32>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    /// 组内是否引用过 vn / vt（决定 normals/uvs 流是否输出）。
    used_vn: bool,
    used_vt: bool,
    /// 有 vn/vt 引用的顶点中缺失的数量（>0 则出占位警告）。
    missing_vn: usize,
    missing_vt: usize,
}

impl Group {
    fn new(name: Option<String>) -> Group {
        Group {
            name,
            mat_name: None,
            dedup: HashMap::new(),
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            used_vn: false,
            used_vt: false,
            missing_vn: 0,
            missing_vt: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.positions.is_empty() && self.indices.is_empty()
    }

    /// 面顶点 → 统一顶点流索引（组合去重）。
    fn intern(
        &mut self,
        vi: usize,
        ti: Option<usize>,
        ni: Option<usize>,
        gpos: &[Vec3],
        gnorm: &[Vec3],
        guv: &[[f32; 2]],
        line: usize,
    ) -> Result<u32, ObjError> {
        let key: FaceKey = (
            vi as u32,
            ti.map(|t| t as u32),
            ni.map(|n| n as u32),
        );
        if let Some(&id) = self.dedup.get(&key) {
            return Ok(id);
        }
        let id = self.positions.len() as u32;
        self.positions.push(gpos[vi]);
        // vn / vt 引用合法性在此统一校验（gpos 已在读取时查过）。
        match ni {
            Some(n) => {
                if n >= gnorm.len() {
                    return Err(ObjError {
                        line,
                        msg: format!("vn 索引 {} 越界（共 {} 条）", n + 1, gnorm.len()),
                        hint: "vn 索引必须在 1..vn 数（负索引相对末尾）".into(),
                    });
                }
                self.normals.push(gnorm[n]);
            }
            None => {
                self.normals.push(Vec3::splat(0.0));
                self.missing_vn += 1;
            }
        }
        match ti {
            Some(t) => {
                if t >= guv.len() {
                    return Err(ObjError {
                        line,
                        msg: format!("vt 索引 {} 越界（共 {} 条）", t + 1, guv.len()),
                        hint: "vt 索引必须在 1..vt 数（负索引相对末尾）".into(),
                    });
                }
                self.uvs.push(guv[t]);
            }
            None => {
                self.uvs.push([0.0, 0.0]);
                self.missing_vt += 1;
            }
        }
        if ni.is_some() {
            self.used_vn = true;
        }
        if ti.is_some() {
            self.used_vt = true;
        }
        self.dedup.insert(key, id);
        Ok(id)
    }

    /// 收尾：按"流是否被引用"裁剪 normals/uvs，返回（网格, 材质名, 占位警告）。
    ///
    /// 占位警告只对"组内引用了 vn/vt 但部分顶点缺失"的混合情况有意义；
    /// 组内完全未引用时整条流裁剪，无占位、无警告。
    fn finish(mut self) -> (ObjMesh, Option<String>, Vec<String>) {
        let mut warn = Vec::new();
        if self.used_vn && self.missing_vn > 0 {
            warn.push(format!(
                "组 {:?}：{} 个顶点未引用 vn，法线以零向量占位（请重新导出法线）",
                self.name, self.missing_vn
            ));
        }
        if self.used_vt && self.missing_vt > 0 {
            warn.push(format!(
                "组 {:?}：{} 个顶点未引用 vt，UV 以 (0,0) 占位",
                self.name, self.missing_vt
            ));
        }
        if !self.used_vn {
            self.normals.clear();
        }
        if !self.used_vt {
            self.uvs.clear();
        }
        let name = self.name;
        (
            ObjMesh {
                name: name.clone(),
                positions: self.positions,
                normals: self.normals,
                uvs: self.uvs,
                indices: self.indices,
                material: None,
            },
            self.mat_name,
            warn,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 主解析
// ─────────────────────────────────────────────────────────────────────────────

/// 解析 OBJ 文本（无 MTL 映射；材质全部 None）。
pub fn parse_obj(obj: &str) -> Result<(Vec<ObjMesh>, Vec<String>), ObjError> {
    parse_obj_inner(obj, &HashMap::new())
}

/// 解析 OBJ 文本并按 `usemtu` 名挂接 MTL 近似材质。
///
/// `mtl` 通常来自与 OBJ 同名的 .mtl 文件文本（本导入器不读外部文件，
/// 由调用方负责加载后传入）。
pub fn parse_obj_with_mtl(obj: &str, mtl: &str) -> Result<(Vec<ObjMesh>, Vec<String>), ObjError> {
    let (table, mut warn) = parse_mtl(mtl);
    let (meshes, mut obj_warn) = parse_obj_inner(obj, &table)?;
    warn.append(&mut obj_warn);
    Ok((meshes, warn))
}

/// 解析 MTL 文本 → (材质名 → PbrMaterial 近似映射, 报告)。
///
/// 近似映射（白皮书 §12：与 B2 Disney 参数对齐）：
/// - `Kd r g b` → base_color（alpha 取 1）；
/// - `Ns` → roughness = `sqrt(2/(Ns+2))`（Phong 指数→GGX 粗糙度标准近似）；
/// - metalness 固定 0（MTL 无金属度概念，逐材质报告注明）；
/// - `map_Kd` 等纹理行不导入，聚合进报告。
pub fn parse_mtl(mtl: &str) -> (HashMap<String, PbrMaterial>, Vec<String>) {
    let mut table: HashMap<String, PbrMaterial> = HashMap::new();
    let mut warn: Vec<String> = Vec::new();
    let mut unsupported: Vec<(String, usize)> = Vec::new(); // (token, 计数)，首现序
    let mut current: Option<String> = None;

    for (li, raw) in mtl.lines().enumerate() {
        let line_no = li + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let mut it = text.split_whitespace();
        let token = it.next().unwrap_or("");
        let args: Vec<&str> = it.collect();
        match token {
            "newmtl" => {
                let name = args.join(" ");
                if name.is_empty() {
                    warn.push(format!("mtl 第 {line_no} 行：newmtl 缺名字，材质未记录"));
                } else {
                    // MTL 无金属度概念：入场即以 metalness=0 记录（保证无 Kd/Ns 的材质也存在）。
                    table
                        .entry(name.clone())
                        .or_insert_with(|| PbrMaterial { metalness: 0.0, ..PbrMaterial::default() });
                    current = Some(name);
                }
            }
            "Kd" => {
                if let (Some(name), Some(rgb)) = (&current, read_rgb(&args)) {
                    let mat = table.entry(name.clone()).or_default();
                    mat.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
                } else if current.is_none() {
                    warn.push(format!("mtl 第 {line_no} 行：Kd 出现在 newmtl 之前，已忽略"));
                }
            }
            "Ns" => {
                if let (Some(name), Some(v)) = (&current, args.first().and_then(|s| s.parse::<f32>().ok())) {
                    let mat = table.entry(name.clone()).or_default();
                    mat.roughness = (2.0 / (v + 2.0)).sqrt();
                }
            }
            "map_Kd" => {
                warn.push(format!("mtl 第 {line_no} 行：map_Kd 纹理导入不支持，仅用 Kd 纯色"));
            }
            _ => count_unsupported(&mut unsupported, token),
        }
    }
    // metalness=0 注明：逐材质一条（MTL 无金属度概念）。
    let mut names: Vec<&String> = table.keys().collect();
    names.sort();
    for name in names {
        warn.push(format!("材质 {name}：MTL 无金属度概念，metalness 固定 0"));
    }
    for (token, count) in unsupported {
        warn.push(format!("mtl：token '{token}' 未支持，已忽略（{count} 处）"));
    }
    (table, warn)
}

/// 未支持 token 聚合计数（保持首现序）。
fn count_unsupported(list: &mut Vec<(String, usize)>, token: &str) {
    if let Some(e) = list.iter_mut().find(|(t, _)| t == token) {
        e.1 += 1;
    } else {
        list.push((token.to_string(), 1));
    }
}

/// Kd 的 RGB 参数（3 个有限浮点；非法返回 None——由聚合报告兜底）。
fn read_rgb(args: &[&str]) -> Option<[f32; 3]> {
    if args.len() < 3 {
        return None;
    }
    let mut c = [0.0_f32; 3];
    for (i, a) in args[..3].iter().enumerate() {
        let v = a.parse::<f32>().ok()?;
        if !v.is_finite() {
            return None;
        }
        c[i] = v;
    }
    Some(c)
}

/// 内部主解析（带材质查表）。
fn parse_obj_inner(
    obj: &str,
    materials: &HashMap<String, PbrMaterial>,
) -> Result<(Vec<ObjMesh>, Vec<String>), ObjError> {
    let mut warn: Vec<String> = Vec::new();
    let mut unsupported: Vec<(String, usize)> = Vec::new();

    // 全局顶点表（OBJ 的 v/vn/vt 是文件级共享）。
    let mut gpos: Vec<Vec3> = Vec::new();
    let mut gnorm: Vec<Vec3> = Vec::new();
    let mut guv: Vec<[f32; 2]> = Vec::new();

    let mut group = Group::new(None);
    let mut groups: Vec<Group> = Vec::new();
    let mut saw_geometry = false; // 是否出现过面/顶点引用（决定空组是否输出）

    for (li, raw) in obj.lines().enumerate() {
        let line_no = li + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let mut it = text.split_whitespace();
        let token = it.next().unwrap_or("");
        let args: Vec<&str> = it.collect();
        match token {
            "v" => {
                let (x, y, z) = read_xyz(&args, line_no, "v")?;
                if args.len() > 3 {
                    warn.push(format!("obj 第 {line_no} 行：v 的第 4 分量（w）已忽略"));
                }
                gpos.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let (x, y, z) = read_xyz(&args, line_no, "vn")?;
                let n = Vec3::new(x, y, z);
                let len = n.length();
                if (len - 1.0).abs() > 1e-3 {
                    warn.push(format!("obj 第 {line_no} 行：vn 长度 {len:.4} ≠ 1，已按原值导入"));
                }
                gnorm.push(n);
            }
            "vt" => {
                if args.is_empty() {
                    return Err(ObjError {
                        line: line_no,
                        msg: "vt 缺少 u 分量".into(),
                        hint: "vt 行至少要有 1 个浮点数".into(),
                    });
                }
                let u = parse_f32(args[0], line_no, "vt.u")?;
                let v = if args.len() > 1 { parse_f32(args[1], line_no, "vt.v")? } else { 0.0 };
                if args.len() > 2 {
                    warn.push(format!("obj 第 {line_no} 行：vt 的第 3 分量（w）已忽略"));
                }
                guv.push([u, v]);
            }
            "f" => {
                saw_geometry = true;
                if args.len() < 3 {
                    return Err(ObjError {
                        line: line_no,
                        msg: format!("面只有 {} 个顶点，无法构成三角形", args.len()),
                        hint: "OBJ 面至少 3 个顶点（f v1 v2 v3 ...）".into(),
                    });
                }
                // 面顶点解析：四种格式 + 1 基/负相对索引。
                let mut refs: Vec<(i64, Option<i64>, Option<i64>)> = Vec::with_capacity(args.len());
                for tok in args {
                    refs.push(parse_face_vert(tok, line_no)?);
                }
                // 组合去重 → 统一顶点流索引（先全部 intern，再扇形三角化）。
                let mut local: Vec<u32> = Vec::with_capacity(refs.len());
                for &(vi, ti, ni) in &refs {
                    let vi = resolve_index(vi, gpos.len(), line_no, "v")?;
                    let ti = match ti {
                        Some(t) => Some(resolve_index(t, guv.len(), line_no, "vt")?),
                        None => None,
                    };
                    let ni = match ni {
                        Some(n) => Some(resolve_index(n, gnorm.len(), line_no, "vn")?),
                        None => None,
                    };
                    let id = group.intern(vi, ti, ni, &gpos, &gnorm, &guv, line_no)?;
                    local.push(id);
                }
                // 扇形三角化：(0, i, i+1)。
                for k in 1..local.len() - 1 {
                    group.indices.push(local[0]);
                    group.indices.push(local[k]);
                    group.indices.push(local[k + 1]);
                }
            }
            "o" | "g" => {
                let name = args.join(" ");
                let name = if name.is_empty() { None } else { Some(name) };
                if group.is_empty() {
                    // 组还没内容：只改名，不拆组。
                    group.name = name;
                } else {
                    groups.push(std::mem::replace(&mut group, Group::new(name)));
                }
            }
            "usemtu" => {
                let name = args.join(" ");
                if name.is_empty() {
                    warn.push(format!("obj 第 {line_no} 行：usemtu 缺名字，已忽略"));
                } else if group.indices.is_empty() {
                    // 面还没开始：只挂材质名，不拆组。
                    group.mat_name = Some(name);
                } else {
                    // 已有面：材质归属要唯一 → 开新组。
                    let mut next = Group::new(None);
                    next.mat_name = Some(name);
                    groups.push(std::mem::replace(&mut group, next));
                }
            }
            "mtllib" => {
                warn.push(format!(
                    "obj 第 {line_no} 行：mtllib 外部材质库不加载，材质由调用方传入的 MTL 文本映射"
                ));
            }
            _ => count_unsupported(&mut unsupported, token),
        }
    }
    // 收尾组（空组不输出——如结尾多一个空 o/usemtu 的情形）。
    if !group.is_empty() {
        groups.push(group);
    }

    if !saw_geometry {
        return Err(ObjError {
            line: obj.lines().count(),
            msg: "资产没有任何面数据（无 f 行）".into(),
            hint: "确认这是三角网格 OBJ；本导入器不支持纯点/线资产".into(),
        });
    }

    // 聚合未支持行报告。
    for (token, count) in unsupported {
        warn.push(format!("obj：行首 token '{token}' 未支持，已忽略（{count} 处）"));
    }

    // 组收尾 + 材质挂接（材质名在 Group 里，收尾时查表）。
    let mut out = Vec::new();
    for g in groups {
        let (mut mesh, mat_name, mut gw) = g.finish();
        warn.append(&mut gw);
        match mat_name {
            Some(mn) => match materials.get(&mn) {
                Some(mat) => mesh.material = Some(mat.clone()),
                None => {
                    if !materials.is_empty() {
                        warn.push(format!(
                            "组 {:?}：usemtu '{mn}' 在 MTL 中无定义，材质留空",
                            mesh.name
                        ));
                    }
                }
            },
            None => {
                if !materials.is_empty() {
                    warn.push(format!("组 {:?}：未指定 usemtu，材质留空", mesh.name));
                }
            }
        }
        out.push(mesh);
    }
    Ok((out, warn))
}

/// 读 3 个有限浮点（v/vn 共用）。
fn read_xyz(args: &[&str], line: usize, what: &str) -> Result<(f32, f32, f32), ObjError> {
    if args.len() < 3 {
        return Err(ObjError {
            line,
            msg: format!("{what} 行需要 3 个分量，得 {}", args.len()),
            hint: format!("{what} x y z（三个十进制浮点数）"),
        });
    }
    let x = parse_f32(args[0], line, what)?;
    let y = parse_f32(args[1], line, what)?;
    let z = parse_f32(args[2], line, what)?;
    Ok((x, y, z))
}

/// 解析有限浮点（inf/NaN 视为坏资产）。
fn parse_f32(tok: &str, line: usize, what: &str) -> Result<f32, ObjError> {
    let v = tok.parse::<f32>().map_err(|_| ObjError {
        line,
        msg: format!("{what} '{tok}' 不是合法数字"),
        hint: "OBJ 数值字段是十进制浮点数".into(),
    })?;
    if !v.is_finite() {
        return Err(ObjError {
            line,
            msg: format!("{what} '{tok}' 非有限值"),
            hint: "坐标必须是有限数".into(),
        });
    }
    Ok(v)
}

/// 面顶点 token → (v, vt, vn) 的 1 基/负相对索引（未引用为 None）。
///
/// 四种合法格式：`v`、`v/vt`、`v//vn`、`v/vt/vn`。
fn parse_face_vert(tok: &str, line: usize) -> Result<(i64, Option<i64>, Option<i64>), ObjError> {
    let parts: Vec<&str> = tok.split('/').collect();
    if parts.len() > 3 {
        return Err(ObjError {
            line,
            msg: format!("面顶点 '{tok}' 段数超过 3"),
            hint: "面顶点格式是 v、v/vt、v//vn 或 v/vt/vn".into(),
        });
    }
    let v = parts[0]
        .parse::<i64>()
        .map_err(|_| ObjError {
            line,
            msg: format!("面顶点 '{tok}' 的 v 段 '{}` 不是整数", parts[0]),
            hint: "顶点索引是 1 基整数（负数表示相对末尾）".into(),
        })?;
    let vt = match parts.get(1) {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s.parse::<i64>().map_err(|_| ObjError {
            line,
            msg: format!("面顶点 '{tok}' 的 vt 段 '{s}' 不是整数"),
            hint: "纹理坐标索引是 1 基整数".into(),
        })?),
        None => None,
    };
    let vn = match parts.get(2) {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s.parse::<i64>().map_err(|_| ObjError {
            line,
            msg: format!("面顶点 '{tok}' 的 vn 段 '{s}' 不是整数"),
            hint: "法线索引是 1 基整数".into(),
        })?),
        None => None,
    };
    Ok((v, vt, vn))
}

/// OBJ 索引换算：>0 → i-1；<0 → len+i；0 或越界 → Err。
fn resolve_index(idx: i64, len: usize, line: usize, what: &str) -> Result<usize, ObjError> {
    if idx == 0 {
        return Err(ObjError {
            line,
            msg: format!("{what} 索引 0 非法（OBJ 索引 1 基，负数相对末尾）"),
            hint: format!("把 {what} 索引改为 1..={len} 或负相对索引"),
        });
    }
    let r = if idx > 0 {
        (idx - 1) as usize
    } else {
        let r = len as i64 + idx;
        if r < 0 {
            return Err(ObjError {
                line,
                msg: format!("{what} 相对索引 {idx} 越界（共 {len} 条）"),
                hint: format!("{what} 索引须落在 1..={len}"),
            });
        }
        r as usize
    };
    if r >= len {
        return Err(ObjError {
            line,
            msg: format!("{what} 索引 {idx} 越界（共 {len} 条）"),
            hint: format!("{what} 索引须落在 1..={len}"),
        });
    }
    Ok(r)
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小往返：3 顶点 + vn + f v//vn → 顶点流与索引逐项断言。
    #[test]
    fn minimal_obj_roundtrip() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
";
        let (meshes, warn) = parse_obj(src).expect("最小资产应解析");
        assert!(warn.is_empty(), "最小资产不应有警告：{warn:?}");
        assert_eq!(meshes.len(), 1);
        let m = &meshes[0];
        assert_eq!(m.positions, vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]);
        assert_eq!(m.normals, vec![Vec3::new(0.0, 0.0, 1.0); 3]);
        assert!(m.uvs.is_empty(), "未引用 vt → uv 流为空");
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert!(m.material.is_none());
    }

    /// 四种面格式混合 + vt 缺 v 分量补 0。
    #[test]
    fn face_formats() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0
vt 1.0 1.0
f 1/1/1 2/2/1 3//1
f 4 1 2
";
        let (meshes, warn) = parse_obj(src).expect("应解析");
        // vt[0] 只有 u → v 补 0。
        let text = warn.join("\n");
        let m = &meshes[0];
        assert_eq!(m.uvs[0], [0.0, 0.0], "vt 0.5 单分量 → v 补 0");
        assert_eq!(m.uvs[1], [1.0, 1.0]);
        // 组合去重语义：面 2 的顶点不带 vt/vn 引用 → 与面 1 的 (v,vt,vn) 组合不同 → 新顶点。
        assert_eq!(m.indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(m.positions.len(), 6);
        // 面 2 的顶点无 vn/vt 引用 → 占位警告。
        assert!(text.contains("占位"), "缺 vn/vt 引用应出占位警告：{text}");
    }

    /// 4 顶点面扇形三角化 + 组合去重（共享顶点只入流一次）。
    #[test]
    fn quad_fan_and_dedup() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
f 1 2 3
";
        let (meshes, _) = parse_obj(src).expect("应解析");
        let m = &meshes[0];
        // 去重：4 顶点只入流一次（第二个面复用 0/1/2）。
        assert_eq!(m.positions.len(), 4);
        assert_eq!(m.indices, vec![0, 1, 2, 0, 2, 3, 0, 1, 2]);
    }

    /// 负相对索引等价正索引。
    #[test]
    fn negative_indices() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f -3 -2 -1
f 1 2 3
";
        let (meshes, _) = parse_obj(src).expect("应解析");
        let m = &meshes[0];
        assert_eq!(m.indices, vec![0, 1, 2, 0, 1, 2], "f -3 -2 -1 == f 1 2 3");
    }

    /// o 分组拆 mesh；组间顶点表独立（同名顶点各自入流）。
    #[test]
    fn multiple_objects_split() {
        let src = "\
o first
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
o second
v 5.0 0.0 0.0
v 6.0 0.0 0.0
v 5.0 1.0 0.0
f 4 5 6
";
        let (meshes, warn) = parse_obj(src).expect("应解析");
        assert!(warn.is_empty(), "无未支持行：{warn:?}");
        assert_eq!(meshes.len(), 2);
        assert_eq!(meshes[0].name.as_deref(), Some("first"));
        assert_eq!(meshes[1].name.as_deref(), Some("second"));
        // 第二组用全局索引 4/5/6 → 组内 0/1/2，顶点表独立。
        assert_eq!(meshes[1].positions[0], Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(meshes[1].indices, vec![0, 1, 2]);
    }

    /// MTL 近似映射：Kd→base_color、Ns→roughness、metalness=0 注明、map_Kd 报告。
    #[test]
    fn mtl_mapping() {
        let mtl = "\
newmtl plastic
Kd 0.8 0.4 0.2
Ns 10.0
map_Kd albedo.png
newmtl blank
";
        let (table, warn) = parse_mtl(mtl);
        let text = warn.join("\n");
        let m = table.get("plastic").expect("plastic 应在表");
        assert_eq!(m.base_color, [0.8, 0.4, 0.2, 1.0]);
        assert!((m.roughness - (2.0 / 12.0_f32).sqrt()).abs() < 1e-6, "Ns=10 → sqrt(2/12)");
        assert_eq!(m.metalness, 0.0, "MTL metalness 固定 0");
        assert!(text.contains("map_Kd"), "纹理入报告：{text}");
        assert!(text.contains("metalness 固定 0"), "逐材质注明：{text}");
        // blank 材质取规范默认。
        let b = table.get("blank").expect("blank 应在表");
        assert_eq!(b.base_color, [1.0, 1.0, 1.0, 1.0], "无 Kd → 默认白");
        assert_eq!(b.roughness, 1.0, "无 Ns → 默认 roughness 1");

        // OBJ + MTL 挂接。
        let obj = "\
mtllib model.mtl
usemtu plastic
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        let (meshes, warn) = parse_obj_with_mtl(obj, mtl).expect("应解析");
        let text = warn.join("\n");
        assert!(text.contains("mtllib"), "mtllib 入报告：{text}");
        let m = &meshes[0];
        let mat = m.material.as_ref().expect("usemtu 应挂上材质");
        assert_eq!(mat.base_color, [0.8, 0.4, 0.2, 1.0]);
        // usemtu 未命中路径。
        let obj2 = obj.replace("plastic", "ghost");
        let (meshes2, warn2) = parse_obj_with_mtl(&obj2, mtl).expect("应解析");
        assert!(meshes2[0].material.is_none(), "未命中 → None");
        assert!(warn2.join("\n").contains("ghost"), "未命中材质名入报告");
    }

    /// usemtu 出现在已有面之后 → 拆新组。
    #[test]
    fn usemtu_splits_group() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 1.0 1.0 0.0
usemtu red
f 1 2 3
usemtu blue
f 2 3 4
";
        let (meshes, _) = parse_obj(src).expect("应解析");
        assert_eq!(meshes.len(), 2, "材质切换拆组");
        assert_eq!(meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(meshes[1].indices, vec![0, 1, 2], "第二组顶点流独立");
        assert_eq!(meshes[1].positions[0], Vec3::new(1.0, 0.0, 0.0), "f 2 3 4 → 组内 0/1/2");
    }

    /// 坏资产：全部结构化 Err（不 panic），Display 含建议段（宪章 C19）。
    #[test]
    fn bad_assets_structured_errors() {
        // 非法浮点。
        let e = parse_obj("v a b c\n").unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.to_string().contains("建议"), "三段式：{e}");
        // 非有限值。
        let e = parse_obj("v 1 2 nan\n").unwrap_err();
        assert!(e.msg.contains("非有限"), "NaN 拒绝：{e}");
        // 索引 0。
        let e = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n").unwrap_err();
        assert!(e.msg.contains("0 非法"), "索引 0：{e}");
        // 索引越界。
        let e = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n").unwrap_err();
        assert!(e.msg.contains("越界"), "越界：{e}");
        // 负相对索引越界。
        let e = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n").unwrap_err();
        assert!(e.msg.contains("越界"), "负相对越界：{e}");
        // 面顶点段数超限。
        let e = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/2/3/4 2 3\n").unwrap_err();
        assert!(e.msg.contains("段数"), "四段拒绝：{e}");
        // 退化面。
        let e = parse_obj("v 0 0 0\nv 1 0 0\nf 1 2\n").unwrap_err();
        assert!(e.msg.contains("无法构成三角形"), "退化面：{e}");
        // 无 f 行。
        let e = parse_obj("v 0 0 0\n").unwrap_err();
        assert!(e.msg.contains("没有任何面"), "纯点资产：{e}");
    }

    /// 未支持行聚合进报告（永不静默）。
    #[test]
    fn unsupported_lines_aggregated() {
        let src = "\
s 1
l 1 2
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
s off
f 1 2 3
";
        let (meshes, warn) = parse_obj(src).expect("应解析");
        assert_eq!(meshes.len(), 1);
        let text = warn.join("\n");
        assert!(text.contains("'s'") && text.contains("2 处"), "s 行聚合计数：{text}");
        assert!(text.contains("'l'"), "l 行入报告：{text}");
    }

    /// 确定性：同输入两遍逐位一致（含去重表路径）。
    #[test]
    fn deterministic() {
        let src = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1/1 2/2/1 3/3/1 4/4/1
";
        let a = parse_obj(src).expect("应解析");
        let b = parse_obj(src).expect("应解析");
        assert_eq!(a, b, "同输入必须同字节");
    }
}
