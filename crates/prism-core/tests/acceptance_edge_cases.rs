//! W1 波次验收测试（QA 严过关）：A1 节点图与注册表的边界反例攻击。
//!
//! 覆盖：
//! - C① `reduction_shape(n)` 边界 n = 0/1/2/3/7/9（树形结构 + 结合序 + 浮点位）；
//! - C② SampleIndex 像素域跨像素流唯一性（大范围抽样）；
//! - C③ Params canonical 字节序（乱序插入 / 嵌套 / 类型 tag / 大端键序）；
//! - C④ 快照装配 frame=0 初值路径与同帧重求值幂等（含 seek 回放）；
//! - 补充：Merge(Plus) 归约走 reduction_shape（spp=7 奇数尾进位 + n=3 槽位）。

use prism_core::{
    eval, payload_from_bytes, payload_to_bytes, reduction_shape, reduce, snapshot_read,
    EvalCtx, Graph, Node, NodeRegistry, Params, Payload, SamplePlane, SampleIndex,
    StructureId,
};
use prism_core::node::{
    Composition, DualityPair, KernelKind, LatticePlan, MathStructure, MergeOp, OpCapabilities,
    ParamValue, StructureMeta,
};

// ---------------------------------------------------------------------------
// C① reduction_shape 边界
// ---------------------------------------------------------------------------

#[test]
fn reduction_shape_boundaries_n0_to_n9() {
    // n=0：空树（空集归约 = 单位元）。
    let s0 = reduction_shape(0);
    assert_eq!(s0.depth(), 0);
    assert!(s0.layers().is_empty());

    // n=1：无归约层。
    let s1 = reduction_shape(1);
    assert_eq!(s1.depth(), 0);

    // n=2：单层单对。
    assert_eq!(reduction_shape(2).layers(), vec![vec![[0, 1]]]);

    // n=3：层0 (0,1)，尾 2 进位；层1 (0,1)。⌈log₂3⌉ = 2。
    let s3 = reduction_shape(3);
    assert_eq!(s3.depth(), 2);
    assert_eq!(s3.layers()[0], vec![[0, 1]]);
    assert_eq!(s3.layers()[1], vec![[0, 1]]);

    // n=7：层0 三对，尾 6 进位 → 层1 两对 → 层2 一对。⌈log₂7⌉ = 3。
    let s7 = reduction_shape(7);
    assert_eq!(s7.depth(), 3);
    assert_eq!(s7.layers()[0], vec![[0, 1], [2, 3], [4, 5]]);
    assert_eq!(s7.layers()[1], vec![[0, 1], [2, 3]]);
    assert_eq!(s7.layers()[2], vec![[0, 1]]);

    // n=9：层0 四对，尾 8 进位 → 层1（len5）两对，尾 4 进位 → 层2（len3）
    // 一对，尾 2 进位 → 层3（len2）一对。⌈log₂9⌉ = 4。
    let s9 = reduction_shape(9);
    assert_eq!(s9.depth(), 4);
    assert_eq!(s9.layers()[0], vec![[0, 1], [2, 3], [4, 5], [6, 7]]);
    assert_eq!(s9.layers()[1], vec![[0, 1], [2, 3]]);
    assert_eq!(s9.layers()[2], vec![[0, 1]]);
    assert_eq!(s9.layers()[3], vec![[0, 1]]);

    // 形状唯一（工程四律 1）：同 n 必得同树。
    for n in [0usize, 1, 2, 3, 7, 9] {
        assert_eq!(reduction_shape(n), reduction_shape(n), "n={n} 形状必须稳定");
    }
}

#[test]
fn reduce_association_order_at_boundaries_matches_manual_tree() {
    // 结合序逐边界手写对照（pairwise + 奇数尾进位）。
    let letter = |n: usize| ((b'a' + n as u8) as char).to_string();

    // n=0 → 单位元；n=1 → 原样。
    assert_eq!(reduce(Vec::<String>::new(), |a, b| format!("({a}+{b})"), "U".into()), "U");
    assert_eq!(reduce(vec![letter(0)], |a, b| format!("({a}+{b})"), "U".into()), "a");

    // n=3 = ((a+b)+c)。
    let v3: Vec<String> = (0..3).map(letter).collect();
    assert_eq!(reduce(v3, |a, b| format!("({a}+{b})"), String::new()), "((a+b)+c)");

    // n=7 = (((a+b)+(c+d))+((e+f)+g))。
    let v7: Vec<String> = (0..7).map(letter).collect();
    assert_eq!(
        reduce(v7, |a, b| format!("({a}+{b})"), String::new()),
        "(((a+b)+(c+d))+((e+f)+g))"
    );

    // n=9 = ((((a+b)+(c+d))+((e+f)+(g+h)))+i)。
    let v9: Vec<String> = (0..9).map(letter).collect();
    assert_eq!(
        reduce(v9, |a, b| format!("({a}+{b})"), String::new()),
        "((((a+b)+(c+d))+((e+f)+(g+h)))+i)"
    );

    // 浮点位级：n=7、n=9 串行归约 == 手工按同树展开（f32 位模式）。
    let vals: Vec<f32> = (0..9).map(|i| 1.0 + i as f32 * 0.1).collect();
    let r7 = reduce(vals[..7].to_vec(), |a, b| a + b, 0.0_f32);
    let manual7 = ((vals[0] + vals[1]) + (vals[2] + vals[3])) + ((vals[4] + vals[5]) + vals[6]);
    assert_eq!(r7.to_bits(), manual7.to_bits(), "n=7 结合序必须与树一致");
    let r9 = reduce(vals.clone(), |a, b| a + b, 0.0_f32);
    let manual9 = (((vals[0] + vals[1]) + (vals[2] + vals[3]))
        + ((vals[4] + vals[5]) + (vals[6] + vals[7])))
        + vals[8];
    assert_eq!(r9.to_bits(), manual9.to_bits(), "n=9 结合序必须与树一致");
}

// ---------------------------------------------------------------------------
// C② SampleIndex 像素域跨像素流唯一性（大范围抽样）
// ---------------------------------------------------------------------------

/// 参考派生（纯整数 splitmix 终化链；本文件独立实现）。
fn hash_mix(seed: u64, x: u64) -> u64 {
    let mut z = seed ^ x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn sample_stream_key(si: &SampleIndex, global_seed: u64) -> u64 {
    let mut h = hash_mix(global_seed, si.node_id);
    h = hash_mix(h, si.frame);
    h = hash_mix(h, ((si.pixel.0 as u64) << 32) | si.pixel.1 as u64);
    h = hash_mix(h, si.sample as u64);
    h = hash_mix(h, si.dimension as u64);
    h
}

#[test]
fn sample_index_pixel_domain_is_collision_free_over_large_grid() {
    // 大范围抽样：256×64 = 16384 像素 × 4 样本 = 65536 条流，节点/帧/维度
    // 固定——D-28 语义：同节点不同像素必不同流（防网格伪影）。
    let mut seen = std::collections::HashSet::new();
    for px in 0..256_u32 {
        for py in 0..64_u32 {
            for s in 0..4_u32 {
                let si = SampleIndex {
                    frame: 11,
                    pixel: (px, py),
                    sample: s,
                    dimension: 0,
                    node_id: 42,
                };
                assert!(
                    seen.insert(sample_stream_key(&si, 7)),
                    "样本流碰撞：pixel=({px},{py}) sample={s}"
                );
            }
        }
    }
    assert_eq!(seen.len(), 65536);

    // 结构体语义：单字段差异均使 SampleIndex 不等（派生键五要素齐备）。
    let a = SampleIndex { frame: 1, pixel: (0, 0), sample: 0, dimension: 0, node_id: 1 };
    assert_ne!(a, SampleIndex { pixel: (0, 1), ..a });
    assert_ne!(a, SampleIndex { sample: 1, ..a });
    assert_ne!(a, SampleIndex { dimension: 1, ..a });
    assert_ne!(a, SampleIndex { node_id: 2, ..a });
    assert_ne!(a, SampleIndex { frame: 2, ..a });
}

// ---------------------------------------------------------------------------
// C③ Params canonical 字节序
// ---------------------------------------------------------------------------

#[test]
fn params_canonical_bytes_byte_level_expectations() {
    // 手算期望：name_len(BE u16) + name + tag + value(BE)。
    let mut p = Params::new();
    p.set("k", ParamValue::F32(1.0));
    let expected: Vec<u8> = {
        let mut v = vec![0x00, 0x01, b'k', 0x00];
        v.extend_from_slice(&1.0_f32.to_bits().to_be_bytes());
        v
    };
    assert_eq!(p.to_canonical_bytes(), expected, "F32 = tag 0 + 大端位模式");

    // 类型 tag 互异：同"数值感"不同类型 → 字节不同。
    let f = |p: &Params| p.to_canonical_bytes();
    let mut pf = Params::new();
    pf.set("x", ParamValue::F32(2.0));
    let mut pu = Params::new();
    pu.set("x", ParamValue::U32(2));
    let mut pb = Params::new();
    pb.set("x", ParamValue::Bool(true));
    let mut ps = Params::new();
    ps.set("x", ParamValue::Str("2".into()));
    assert_ne!(f(&pf), f(&pu));
    assert_ne!(f(&pf), f(&pb));
    assert_ne!(f(&pf), f(&ps));
    assert_ne!(f(&pu), f(&pb));

    // BTreeMap 键序 = 字节序：大写在小写前（'Z'(0x5A) < 'a'(0x61)）。
    let mut p1 = Params::new();
    p1.set("alpha", ParamValue::U32(1));
    p1.set("Zulu", ParamValue::U32(2));
    let mut p2 = Params::new();
    p2.set("Zulu", ParamValue::U32(2));
    p2.set("alpha", ParamValue::U32(1));
    assert_eq!(f(&p1), f(&p2), "插入序无关");
    let direct = {
        let mut v = vec![0x00, 0x04];
        v.extend_from_slice(b"Zulu");
        v.push(1);
        v.extend_from_slice(&2_u32.to_be_bytes());
        v.extend_from_slice(&[0x00, 0x05]);
        v.extend_from_slice(b"alpha");
        v.push(1);
        v.extend_from_slice(&1_u32.to_be_bytes());
        v
    };
    assert_eq!(f(&p1), direct, "键按字节序升序输出");

    // 嵌套值：UTF-8 多字节键/值的长度前缀按字节数。
    let mut p3 = Params::new();
    p3.set("名", ParamValue::Str("棱镜引擎".into()));
    let b3 = p3.to_canonical_bytes();
    assert_eq!(&b3[0..2], &3_u16.to_be_bytes(), "UTF-8 键长按字节计（名 = 3 字节）");
    assert_eq!(&b3[2..5], "名".as_bytes());
    assert_eq!(b3[5], 3, "Str tag");
    assert_eq!(&b3[6..8], &12_u16.to_be_bytes(), "值长 = 4 字 × 3 字节");

    // 同名覆盖后字节稳定（无残留旧值）。
    let mut p4 = Params::new();
    p4.set("v", ParamValue::F32(9.0));
    p4.set("v", ParamValue::F32(3.5));
    let mut p5 = Params::new();
    p5.set("v", ParamValue::F32(3.5));
    assert_eq!(f(&p4), f(&p5));
    assert_eq!(p4.len(), 1);

    // 空包 = 空字节。
    assert!(Params::new().to_canonical_bytes().is_empty());
}

// ---------------------------------------------------------------------------
// C④ 快照装配：frame=0 初值路径 / 同帧重求值幂等 / seek 回放
// ---------------------------------------------------------------------------

/// 计数投影特解（S3）：Empty → 0，标量 v → v+1（公开 API 注册，走真实汇入校验）。
#[derive(Debug, Default)]
struct CountStep;

fn constraint_meta() -> StructureMeta {
    StructureMeta::new(
        StructureId::ConstraintFlow,
        KernelKind::Green,
        DualityPair::PrimalMultiplier,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: false },
    )
}

impl MathStructure for CountStep {
    fn meta(&self) -> &StructureMeta {
        // 单测生命周期内的进程级缓存：meta 只读常量。
        use std::sync::OnceLock;
        static META: OnceLock<StructureMeta> = OnceLock::new();
        META.get_or_init(constraint_meta)
    }
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        match x {
            Payload::Empty => Payload::Scalar(0.0),
            Payload::Scalar(v) => Payload::Scalar(v + 1.0),
            other => *other,
        }
    }
}

fn step_setup() -> (Graph, NodeRegistry) {
    let mut reg = NodeRegistry::new();
    reg.collect("count_step", constraint_meta(), || {
        Box::new(CountStep) as Box<dyn MathStructure>
    })
    .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::new(7, StructureId::ConstraintFlow, Composition::Feed)).unwrap();
    (g, reg)
}

#[test]
fn snapshot_assembly_frame0_initial_and_same_frame_idempotent() {
    let (g, reg) = step_setup();
    let plane = SamplePlane { pixel: (0, 0), spp: 1 };
    let sig = [0x5E; 32];

    // frame=0：无上一帧快照 → 图输入（Empty）作初值，投影一步得 0。
    let mut ctx = EvalCtx::new(1, 0, 0);
    let o0 = eval(&g, &reg, &plane, sig, &mut ctx).unwrap();
    assert_eq!(o0.get(7), Some(Payload::Scalar(0.0)), "frame=0 初值路径");
    assert_eq!(snapshot_read(&ctx, sig, 0), Some(Payload::Scalar(0.0)));

    // frame=1、2 顺序推进：0 → 1 → 2。
    ctx.frame = 1;
    assert_eq!(
        eval(&g, &reg, &plane, sig, &mut ctx).unwrap().get(7),
        Some(Payload::Scalar(1.0))
    );
    ctx.frame = 2;
    assert_eq!(
        eval(&g, &reg, &plane, sig, &mut ctx).unwrap().get(7),
        Some(Payload::Scalar(2.0))
    );

    // 同帧重求值幂等：读键 = 上一帧，写键 = 当前帧；重跑 frame=2 仍得 2。
    ctx.frame = 2;
    let again = eval(&g, &reg, &plane, sig, &mut ctx).unwrap();
    assert_eq!(again.get(7), Some(Payload::Scalar(2.0)), "同帧重求值幂等");
    assert_eq!(ctx.snapshots.len(), 3, "重求值不新增快照");

    // seek 回放：回到 frame=0（快照库已含 (sig,0)）仍按初值语义得 0。
    ctx.frame = 0;
    let replay0 = eval(&g, &reg, &plane, sig, &mut ctx).unwrap();
    assert_eq!(replay0.get(7), Some(Payload::Scalar(0.0)), "frame=0 恒用图输入作初值");
    // 再推进到 1：读 (sig,0)=0 → 1，与首次推进逐位一致。
    ctx.frame = 1;
    let replay1 = eval(&g, &reg, &plane, sig, &mut ctx).unwrap();
    assert_eq!(replay1.get(7), Some(Payload::Scalar(1.0)), "seek 回放与顺序推进一致");
}

#[test]
fn snapshot_payload_bytes_reject_truncated_and_unknown_tag() {
    // 快照字节契约：截断/未知 tag → None（绝不静默猜测，宪章 C20）。
    let scalar = payload_to_bytes(&Payload::Scalar(0.25));
    assert_eq!(scalar.len(), 5);
    assert_eq!(payload_from_bytes(&scalar[..4]), None, "截断标量必拒");
    assert_eq!(payload_from_bytes(&[]), None, "空字节必拒");
    assert_eq!(payload_from_bytes(&[7]), None, "未知 tag 必拒");
    // tag 2（Empty）多余字节必拒。
    assert_eq!(payload_from_bytes(&[2, 0]), None);
}

// ---------------------------------------------------------------------------
// 补充：Merge(Plus) 归约走 reduction_shape（spp=7 奇数尾进位 + n=3 槽位）
// ---------------------------------------------------------------------------

/// 采样特解：样本值 = sample 序号 + 1（可手算归约期望）。
#[derive(Debug, Default)]
struct OrdinalSampler;

fn sampling_meta() -> StructureMeta {
    StructureMeta::new(
        StructureId::DeterministicSampling,
        KernelKind::Transfer,
        DualityPair::SampleMeasure,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: true },
    )
}

impl MathStructure for OrdinalSampler {
    fn meta(&self) -> &StructureMeta {
        use std::sync::OnceLock;
        static META: OnceLock<StructureMeta> = OnceLock::new();
        META.get_or_init(sampling_meta)
    }
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        *x
    }
    fn sample(&self, si: SampleIndex, _ctx: &EvalCtx) -> Payload {
        Payload::Scalar(si.sample as f32 + 1.0)
    }
}

/// 常量特解族（const 泛型编码值，Empty → M/8）。
#[derive(Debug, Default)]
struct Const<const M: u32>;

impl<const M: u32> MathStructure for Const<M> {
    fn meta(&self) -> &StructureMeta {
        use std::sync::OnceLock;
        static META: OnceLock<StructureMeta> = OnceLock::new();
        META.get_or_init(|| {
            StructureMeta::new(
                StructureId::ScalarField,
                KernelKind::Covariance,
                DualityPair::SampleMeasure,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            )
        })
    }
    fn apply(&self, _x: &Payload, _ctx: &EvalCtx) -> Payload {
        Payload::Scalar(M as f32 / 8.0)
    }
}

#[test]
fn merge_plus_reduction_and_sample_mean_follow_fixed_shape() {
    let mut reg = NodeRegistry::new();
    reg.collect("ordinal", sampling_meta(), || {
        Box::new(OrdinalSampler) as Box<dyn MathStructure>
    })
    .unwrap();

    let mut g = Graph::new();
    g.add_node(Node::new(1, StructureId::DeterministicSampling, Composition::Feed)).unwrap();
    let plane = SamplePlane { pixel: (0, 0), spp: 7 };
    let sig = [0u8; 32];
    let mut ctx = EvalCtx::new(0, 0, 0);
    let out = eval(&g, &reg, &plane, sig, &mut ctx).unwrap();

    // spp=7：样本 1..=7，串行参考 = 对 reduction_shape(7) 树的固定顺序求值
    // = ((1+2)+(3+4))+((5+6)+7)，均值 = 28/7。浮点位断言（非数值比较）。
    let expect_sum = ((1.0_f32 + 2.0) + (3.0 + 4.0)) + ((5.0 + 6.0) + 7.0);
    assert_eq!(out.get(1), Some(Payload::Scalar(expect_sum / 7.0)), "spp=7 固定形状均值");

    // 两遍求值（spp=7 边界补测关卡 1）：canonical 字节逐字节一致。
    let mut ctx3 = EvalCtx::new(0, 0, 0);
    let o3 = eval(&g, &reg, &plane, sig, &mut ctx3).unwrap();
    let canon = |o: &prism_core::EvalOutput| {
        o.iter()
            .flat_map(|(id, p)| {
                let mut v = id.to_be_bytes().to_vec();
                v.extend_from_slice(&payload_to_bytes(&p));
                v
            })
            .collect::<Vec<u8>>()
    };
    assert_eq!(canon(&out), canon(&o3), "spp=7 两遍求值逐字节一致");
}

#[test]
fn merge_plus_three_inputs_reduce_in_slot_order_pairwise_tree() {
    // 三个不同结构的常量源（1、2、3）→ Merge(Plus) 汇点。
    let mut reg = NodeRegistry::new();
    for (name, milli) in [("c1", 8_u32), ("c2", 16), ("c3", 24)] {
        // 三个常量特解共用 ScalarField 结构时同 id 冲突——按 milli 派生三个
        // 结构位是错误做法；正确做法 = 三个节点引用同一已注册结构。此处注册
        // 一个常量结构，三个节点均引用它（值由 const 泛型区分不可行——工厂
        // 无参；改用三个不同 const 类型的 meta 需要三个不同 StructureId）。
        let _ = (name, milli);
    }
    // 注册三个结构（ScalarField/Diffusion/Convolution 各承载一个常量特解）。
    let specs: [(StructureId, u32); 3] = [
        (StructureId::ScalarField, 8),
        (StructureId::Diffusion, 16),
        (StructureId::Convolution, 24),
    ];
    for (i, (sid, milli)) in specs.iter().enumerate() {
        let meta = StructureMeta::new(
            sid.clone(),
            KernelKind::Covariance,
            DualityPair::SampleMeasure,
            LatticePlan::LatticeAnchored,
            OpCapabilities { invert: false, sample: false },
        );
        let name = format!("const.{}", milli);
        match i {
            0 => reg.collect(&name, meta, || Box::new(Const::<8>) as Box<dyn MathStructure>),
            1 => reg.collect(&name, meta, || Box::new(Const::<16>) as Box<dyn MathStructure>),
            _ => reg.collect(&name, meta, || Box::new(Const::<24>) as Box<dyn MathStructure>),
        }
        .unwrap();
    }

    let mut g = Graph::new();
    g.add_node(Node::new(1, StructureId::ScalarField, Composition::Feed)).unwrap();
    g.add_node(Node::new(2, StructureId::Diffusion, Composition::Feed)).unwrap();
    g.add_node(Node::new(3, StructureId::Convolution, Composition::Feed)).unwrap();
    g.add_node(Node::new(
        4,
        StructureId::EntityRepresentation,
        Composition::Merge(MergeOp::Plus),
    ))
    .unwrap();
    g.connect(1, 4, 0).unwrap();
    g.connect(2, 4, 1).unwrap();
    g.connect(3, 4, 2).unwrap();

    // EntityRepresentation 也需注册（汇点结构）。
    reg.collect(
        "sink",
        StructureMeta::new(
            StructureId::EntityRepresentation,
            KernelKind::NoKernel,
            DualityPair::ContentDigest,
            LatticePlan::LatticeAnchored,
            OpCapabilities { invert: false, sample: false },
        ),
        || Box::new(Identity) as Box<dyn MathStructure>,
    )
    .unwrap();

    let plane = SamplePlane { pixel: (0, 0), spp: 1 };
    let mut ctx = EvalCtx::new(0, 0, 0);
    let o = eval(&g, &reg, &plane, [0u8; 32], &mut ctx).unwrap();
    // (1 + 2) + 3 = 6（2^-k 倍数，加法精确）。
    assert_eq!(o.get(4), Some(Payload::Scalar(6.0)), "n=3 槽位归约按 pairwise 树");
}

#[derive(Debug, Default)]
struct Identity;

impl MathStructure for Identity {
    fn meta(&self) -> &StructureMeta {
        use std::sync::OnceLock;
        static META: OnceLock<StructureMeta> = OnceLock::new();
        META.get_or_init(|| {
            StructureMeta::new(
                StructureId::EntityRepresentation,
                KernelKind::NoKernel,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            )
        })
    }
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        *x
    }
}
