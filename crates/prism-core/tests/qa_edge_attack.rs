//! QA 边界反例攻击（边界攻击清单 C-⑦；prism-core 侧）。
//!
//! 覆盖：循环图喂 canonical 序列化（应报错不 panic）、自环、Remap 退化与
//! 倒序区间行为固化，以及清单 D 的 core 侧 SHA-256 锚定——
//! `channel_definition_hash`（走 signature.rs 的**模块私有** SHA-256）与
//! Python hashlib 基线对拍；与 prism-io 侧 qa_edge_attack 的六向量（同一
//! 数据模式、同基线族）互为镜像，共同证明双实现同向量一致。

use prism_anim::{EasingKind, Keyframe, Track};
use prism_core::node::Composition;
use prism_core::signature::{
    channel_definition_hash, graph_canonical_bytes, graph_signatures, ChannelDef, ChannelDefs,
    SignatureError,
};
use prism_core::{Graph, Node, PropertyChannel, StructureId};

// ---------------------------------------------------------------------------
// C-⑦ 循环图
// ---------------------------------------------------------------------------

/// 构造 n 节点 Feed 环（1→2→…→n→1）。
fn cycle_graph(n: u64) -> Graph {
    let mut g = Graph::new();
    for id in 1..=n {
        g.add_node(Node::new(id, StructureId::ScalarField, Composition::Feed)).unwrap();
    }
    for id in 1..n {
        g.connect(id, id + 1, 0).unwrap();
    }
    g.connect(n, 1, 0).unwrap();
    g
}

/// C-⑦：环图喂 `graph_canonical_bytes` **不得 panic/死循环**——canonical
/// 序列化是纯结构展开（不依赖拓扑序），应产出字节；签名侧才拒绝环。
#[test]
fn qa_cycle_graph_canonical_bytes_no_panic() {
    for n in [2_u64, 3, 8] {
        let g = cycle_graph(n);
        let bytes = graph_canonical_bytes(&g);
        assert!(!bytes.is_empty(), "{n} 节点环的 canonical 字节不应为空");
        assert_eq!(&bytes[..10], b"PRISMCANON", "canonical 头魔数（宪章 C8）");
        // 魔数后是 schema 版本 u32 大端（SIGNATURE_SCHEMA_VERSION = 1）。
        assert_eq!(
            u32::from_be_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]),
            1,
            "schema 版本头"
        );
    }
}

/// C-⑦：三节点环走 `graph_signatures` → 结构化报错 P3005（不 panic、
/// 不死循环、Display 三段式含"建议"）。
#[test]
fn qa_cycle_graph_signatures_error_p3005() {
    let g = cycle_graph(3);
    let err = graph_signatures(&g, &ChannelDefs::new()).unwrap_err();
    match &err {
        SignatureError::Graph(ge) => {
            assert_eq!(ge.code, "P3005", "环图透传拓扑错误码");
            assert!(ge.fix.contains("建议") || !ge.fix.is_empty(), "C19：修复建议非空");
        }
        other => panic!("应为 Graph(P3005)，实际 {other:?}"),
    }
    assert!(err.to_string().contains("建议"), "C19：错误 Display 必含建议段");
}

/// C-⑦：自环在**构造层**即被防呆拒绝——`Graph::connect(1→1)` 直接返回
/// P3005（"自环连线：节点不允许连接自身"），环根本进不了图；拒绝后图仍
/// 可正常签名（连接层防呆优先于签名层拓扑检查，双层防护）。
#[test]
fn qa_self_loop_rejected() {
    let mut g = Graph::new();
    g.add_node(Node::new(1, StructureId::ScalarField, Composition::Feed)).unwrap();
    let err = g.connect(1, 1, 0).unwrap_err();
    assert_eq!(err.code, "P3005", "自环连线在 connect 层被拒绝");
    assert!(err.fix.contains("建议") || !err.fix.is_empty(), "C19：修复建议非空");
    // 拒绝自环后图仍然健康（无环）→ 签名正常。
    assert!(graph_signatures(&g, &ChannelDefs::new()).is_ok(), "拒绝自环后图应可签名");
}

// ---------------------------------------------------------------------------
// Remap 组合子：退化与倒序区间行为固化
// ---------------------------------------------------------------------------

/// 0s:0 → 10s:100 的线性轨道（斜率 10/s，便于读出内层时间）。
fn ramp_track() -> PropertyChannel {
    PropertyChannel::Track(Track::new(vec![
        Keyframe::new(0.0, 0.0, EasingKind::Linear),
        Keyframe::new(10.0, 100.0, EasingKind::Linear),
    ]))
}

/// Remap 正向：out [0,1] → in [0,10]，端点严格对齐、中段线性保序。
#[test]
fn qa_remap_forward_endpoints_and_monotonic() {
    let ch = PropertyChannel::Remap {
        inner: Box::new(ramp_track()),
        in_lo: 0.0,
        in_hi: 10.0,
        out_lo: 0.0,
        out_hi: 1.0,
    };
    assert!((ch.sample(0.0) - 0.0).abs() < 1e-5, "t=out_lo → inner(in_lo)");
    assert!((ch.sample(0.5) - 50.0).abs() < 1e-5, "中点线性");
    assert!((ch.sample(1.0) - 100.0).abs() < 1e-4, "t=out_hi → inner(in_hi)");
    let mut prev = ch.sample(0.0);
    for i in 1..=20 {
        let v = ch.sample(i as f32 / 20.0);
        assert!(v >= prev, "正向 remap 应保序：t={}", i as f32 / 20.0);
        prev = v;
    }
}

/// Remap 倒序 in 区间（in_hi < in_lo）：固化**当前实现行为**——映射为线性
/// 递减、端点仍严格对齐（t=out_lo → inner(in_lo)，t=out_hi → inner(in_hi)）。
/// 注：模块文档称"线性映射保序（单调）"，倒序下为递减方向——文档措辞与
/// 实现的出入已列为观察项；本测试固化事实行为防无声漂移。
#[test]
fn qa_remap_reversed_in_interval_endpoints_align() {
    let ch = PropertyChannel::Remap {
        inner: Box::new(ramp_track()),
        in_lo: 10.0,
        in_hi: 0.0, // 倒序
        out_lo: 0.0,
        out_hi: 1.0,
    };
    assert!((ch.sample(0.0) - 100.0).abs() < 1e-4, "t=out_lo → inner(in_lo=10s) = 100");
    assert!((ch.sample(0.5) - 50.0).abs() < 1e-5, "中点 inner(5s) = 50（递减方向）");
    assert!((ch.sample(1.0) - 0.0).abs() < 1e-5, "t=out_hi → inner(in_hi=0s) = 0");
}

/// Remap 退化 out 区间（out_hi == out_lo）：恒映射到 in_lo（文档化行为）。
#[test]
fn qa_remap_degenerate_out_span() {
    let ch = PropertyChannel::Remap {
        inner: Box::new(ramp_track()),
        in_lo: 0.0,
        in_hi: 10.0,
        out_lo: 3.0,
        out_hi: 3.0, // span == 0
    };
    for t in [0.0_f32, 3.0, 3.5, 100.0, -7.0] {
        assert!((ch.sample(t) - 0.0).abs() < 1e-6, "span=0 时任意 t 恒 inner(in_lo)=0，t={t}");
    }
}

// ---------------------------------------------------------------------------
// 清单 D：core 侧 SHA-256（模块私有实现）同向量锚定
// ---------------------------------------------------------------------------

/// 数据生成器：data[i] = (i*7+3) mod 256（与 prism-io 侧 / Python 基线同模式）。
fn attack_data(n: usize) -> Vec<u8> {
    (0..n).map(|i: usize| ((i * 7 + 3) % 256) as u8).collect()
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// D 项（core 侧）：`channel_definition_hash(tag=0)` = sha256(00 00 ++ data)，
/// 走 signature.rs 模块私有 SHA-256，与 Python hashlib 基线对拍（块边界六
/// 长度顺带覆盖）。与 prism-io 侧六向量测试共同构成双实现同向量对拍。
#[test]
fn qa_core_sha256_matches_hashlib_via_channel_definition_hash() {
    let cases: [(usize, &str); 6] = [
        (55, "b22a6e13b3b9acc9c97273d5f6b61e6fa446f0cb8a8ccb1e0400e8792fce0fe9"),
        (56, "6bd07af46ad47d1c5f6d3d56955fc35d69dfe3ddd302665009e531600e7daff8"),
        (57, "c803e4d4fa5e08c39745628eba7a9f30c4373465ee40c9a56c461168496445db"),
        (63, "7a4b9e898fa80be66dadb2191e25d690952a42029d7af247792539f53f98ce19"),
        (64, "0c28395c73663830bbd63d12290b41c2013939c8b867a0481ce811f286433b8f"),
        (65, "c286d4157d889581a1b0e32f822218422bb90bb0199a1b7c4b9fbe4e2d3fc8a4"),
    ];
    for (n, want) in cases {
        let def = ChannelDef { tag: 0, def_bytes: attack_data(n) };
        let got = hex(&channel_definition_hash(&def));
        assert_eq!(got, want, "core 侧 SHA-256（经 channel_definition_hash）{n} 字节向量不符");
    }
}
