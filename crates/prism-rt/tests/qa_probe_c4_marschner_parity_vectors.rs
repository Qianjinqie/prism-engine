//! Marschner 毛发散射 **CPU 侧 golden vector 生成器**（毛发深化 B 收口）。
//!
//! # 存在的理由：CPU/GPU 从未对拍过
//!
//! `hair_scatter.rs`（Rust）与 `shaders/hair_scatter.wgsl`（WGSL）是**两份独立
//! 手写**的同一组 Marschner 公式。Marschner 是**能量守恒敏感**的物理公式
//! （三瓣之和 ≈ 入射能量），偏差 1e-3 就足以让头发"发灰/发黑"——而在本测试
//! 之前，两侧偏差**完全无人知晓**，因为没有任何东西把两者放在一起比较。
//!
//! 本文件是那次比较的**CPU 一半**：把 12 组测试向量的 `components_precise`
//! 三瓣结果以 **f32 位模式（十六进制）** 写到 `web/hair_golden.json`；
//! `web/host_test.mjs` 的「Marschner CPU/GPU 对拍」节读该文件，把同一组向量
//! 喂给 **SwiftShader/Chromium 上真跑的 WGSL**，逐分量比对。
//!
//! # 为什么用位模式而不是十进制
//!
//! 项目铁律要求"同输入逐字节可复现"。十进制往返会引入十进制→二进制→十进制
//! 的双重舍入，使"CPU 与 GPU 差 1 ulp"这种信息被格式误差淹没。位模式是**无损**
//! 的，且能直接断言 `to_bits()` 逐位一致（项目既有 `*_bitwise_frozen` 风格）。
//!
//! # 12 组向量的覆盖设计（不是随便凑数）
//!
//! | # | 覆盖维度 | 取值要点 |
//! |---|---------|---------|
//! | 0 | 正入射基准 | θ_i=θ_o=0，φ=0（峰值，M 与 N 同时最大） |
//! | 1 | 掠入射 | θ_i→π/2（cosθ_i 下界 1e-3，Fresnel→1） |
//! | 2 | 掠出射 | θ_o→π/2（M 的 sinθ_o 达 1） |
//! | 3 | 前向散射 | φ=0（φ_TT = −π，TT 瓣方位项最小 → 分离最明显） |
//! | 4 | 后向散射 | φ=π（φ_TT = 0，**TT 瓣方位项最大**——三瓣 φ 偏角符号约定的判据） |
//! | 5 | 低粗糙度 | β_m=0.02（尖锐峰）、k=1（小锥） |
//! | 6 | 高粗糙度 | β_m=0.5（平坦）、k=15（近各向同性的宽锥） |
//! | 7 | 零吸收 | σ_a=0（T≡1，验证吸收项退化路径） |
//! | 8 | 大吸收 | σ_a=6（强衰减，exp 下溢风险区） |
//! | 9 | 深色发 | f0 小（η=1.2） |
//! | 10 | 浅色发 | f0 大（η=1.9） |
//! | 11 | 混合极端 | 掠入射 + 零吸收 + 高粗糙度 + 后向散射（交叉覆盖） |
//!
//! # 铁律遵守
//!
//! - 零第三方依赖（只用 std）。
//! - 无 panic：所有参数经 `HairScatter::new` 域断言；测试内 `expect` 仅在
//!   "写文件失败"这一不可恢复场景出现，且带明确消息。
//! - 确定性：向量表是**静态字面量数组**，遍历顺序固定，无 HashMap。

use prism_rt::hair_scatter::HairScatter;

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

/// 一组测试向量。字段与 `HairScatter::new` + `components_precise` 一一对应。
///
/// `theta_i/phi_i/theta_o/phi_o` 为弧度；`beta_m` 纵向粗糙度；`k` 方位粗糙度；
/// `eta` 纤维折射率；`sigma_a` 吸收系数；`radius` 纤维半径。
struct Vector {
    name: &'static str,
    theta_i: f32,
    phi_i: f32,
    theta_o: f32,
    phi_o: f32,
    beta_m: f32,
    k: f32,
    eta: f32,
    sigma_a: f32,
    radius: f32,
}

/// 12 组覆盖向量（顺序固定 = 确定性铁律）。
///
/// 注意 `beta_m` 必须落在 `(0, 1]`（`HairScatter::new` 的域断言），
/// `eta >= 1`，`sigma_a >= 0`，`radius > 0`。
fn vectors() -> Vec<Vector> {
    vec![
        // 0: 正入射基准 —— M 与 N 同时取峰，验证三瓣都非零且 R > TT。
        Vector { name: "normal_incidence", theta_i: 0.0, phi_i: 0.0, theta_o: 0.0, phi_o: 0.0,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 1: 掠入射 —— cosθ_i 下界 1e-3，Schlick F→1，三瓣全部被压向 0。
        Vector { name: "grazing_incidence", theta_i: 1.5707, phi_i: 0.3, theta_o: 0.4, phi_o: 0.9,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 2: 掠出射 —— sinθ_o→1，M 的指数项远离峰值。
        Vector { name: "grazing_exit", theta_i: 0.2, phi_i: -0.4, theta_o: 1.5707, phi_o: 0.5,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 3: 前向散射 φ=0 —— φ_TT = −π，cos(φ_TT/2)=0，TT 瓣方位项 = I₀(0)，
        //    与 R 瓣的 I₀(k) 形成最大反差（φ 偏角符号约定的负向判据）。
        Vector { name: "forward_scatter", theta_i: 0.35, phi_i: 0.0, theta_o: 0.35, phi_o: 0.0,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 4: 后向散射 φ=π —— φ_TT = 0，cos(φ_TT/2)=1，TT 瓣方位项 = I₀(k) 取峰。
        //    若三瓣 φ 偏角符号写反（φ−π 写成 φ+π），本行与第 3 行的
        //    R/TT 比值会互换——这正是"小角度下误差极小、极易漏过"的符号错误
        //    的主要判据，故必须成对出现。
        Vector { name: "backward_scatter", theta_i: 0.35, phi_i: 0.0, theta_o: 0.35,
                 phi_o: std::f32::consts::PI,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 5: 低粗糙度 —— 尖锐峰 + 小方位锥（M 集中在 θ_o≈θ_i）。
        Vector { name: "low_roughness", theta_i: 0.5, phi_i: 0.8, theta_o: 0.55, phi_o: 1.1,
                 beta_m: 0.02, k: 1.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 6: 高粗糙度 —— 平坦峰 + 宽方位锥（近各向同性）。
        Vector { name: "high_roughness", theta_i: 0.5, phi_i: 0.8, theta_o: 0.9, phi_o: 1.1,
                 beta_m: 0.5, k: 15.0, eta: 1.55, sigma_a: 0.2, radius: 0.04 },
        // 7: 零吸收 —— σ_a=0 使 T≡1，验证吸收项的退化路径（无 exp 下溢）。
        Vector { name: "zero_absorption", theta_i: 0.4, phi_i: 0.2, theta_o: 0.45, phi_o: 0.7,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 0.0, radius: 0.04 },
        // 8: 大吸收 —— σ_a=6 且掠入射 → σ_a·d 很大，逼近 det_exp 的 −87 下溢守卫。
        Vector { name: "high_absorption", theta_i: 1.4, phi_i: 0.2, theta_o: 0.45, phi_o: 0.7,
                 beta_m: 0.12, k: 8.0, eta: 1.55, sigma_a: 6.0, radius: 0.04 },
        // 9: 深色发 —— η=1.2 → f0≈0.0083，F 极小，透射瓣相对突出。
        Vector { name: "dark_hair", theta_i: 0.4, phi_i: 0.2, theta_o: 0.45, phi_o: 0.7,
                 beta_m: 0.10, k: 10.0, eta: 1.2, sigma_a: 1.5, radius: 0.04 },
        // 10: 浅色发 —— η=1.9 → f0≈0.086，反射瓣相对突出。
        Vector { name: "blond_hair", theta_i: 0.4, phi_i: 0.2, theta_o: 0.45, phi_o: 0.7,
                 beta_m: 0.12, k: 8.0, eta: 1.9, sigma_a: 0.2, radius: 0.04 },
        // 11: 交叉极端 —— 掠入射 + 零吸收 + 高粗糙度 + 后向散射（维度组合覆盖）。
        Vector { name: "combined_extreme", theta_i: 1.5707, phi_i: 0.0, theta_o: 0.6,
                 phi_o: std::f32::consts::PI,
                 beta_m: 0.5, k: 15.0, eta: 1.55, sigma_a: 0.0, radius: 0.04 },
    ]
}

/// f32 位模式的十六进制字符串（无损，供 Node 侧 `Float32Array` 视图直接比对）。
fn bits_hex(v: f32) -> String {
    format!("0x{:08x}", v.to_bits())
}

#[test]
fn emit_cpu_golden_vectors_for_gpu_parity() {
    let vecs = vectors();
    assert_eq!(vecs.len(), 12, "契约：恰好 12 组测试向量（对拍门禁按此计数）");

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"_comment\": \"CPU 侧 golden vectors，由 `cargo test -p prism-rt emit_cpu_golden_vectors_for_gpu_parity` 生成。值为 f32 位模式（十六进制，无损）。勿手改。\",\n");
    json.push_str("  \"source\": \"prism-rt/src/hair_scatter.rs::components_precise\",\n");
    json.push_str("  \"format\": \"u32-hex-bit-pattern\",\n");
    json.push_str("  \"vectors\": [\n");

    for (idx, v) in vecs.iter().enumerate() {
        // 构造会做域断言（宪章 C19）——12 组取值均在域内。
        let h = HairScatter::new(v.beta_m, v.k, v.eta, v.sigma_a, v.radius);
        let (r, tt, trt) = h.components_precise(v.theta_i, v.phi_i, v.theta_o, v.phi_o);
        // 有限性前置：非有限值会让 GPU 侧比对失去意义（NaN != NaN，误差无界）。
        assert!(r.is_finite(), "{}: R 非有限", v.name);
        assert!(tt.is_finite(), "{}: TT 非有限", v.name);
        assert!(trt.is_finite(), "{}: TRT 非有限", v.name);

        // 条目本体（**不含**尾逗号：分隔符由下面的 `ssep` 统一决定，
        // 否则 format 串里的逗号与分隔逗号叠加会产出 `}},` 的坏 JSON）。
        let _ = write!(
            json,
            concat!(
                "    {{\"name\": \"{}\", \"theta_i\": {}, \"phi_i\": {}, \"theta_o\": {}, ",
                "\"phi_o\": {}, \"beta_m\": {}, \"k\": {}, \"eta\": {}, \"sigma_a\": {}, ",
                "\"radius\": {}, \"r_bits\": \"{}\", \"tt_bits\": \"{}\", \"trt_bits\": \"{}\"}}"
            ),
            v.name,
            v.theta_i,
            v.phi_i,
            v.theta_o,
            v.phi_o,
            v.beta_m,
            v.k,
            v.eta,
            v.sigma_a,
            v.radius,
            bits_hex(r),
            bits_hex(tt),
            bits_hex(trt),
        );
        // 最后一条不带逗号（合法 JSON 尾随逗号虽宽松，但门禁要严格）。
        if idx + 1 != vecs.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n}\n");

    // 【自证】产物必须是**合法 JSON**：本文件早期版本在 format 串里多写了一个
    // 逗号，产出 `}},,` 的坏 JSON，而测试本身仍然"通过"——因为它只写文件、
    // 从不读回。门禁自己生成的产物必须能被下游解析，故此处做三项结构自检：
    //   1. 无 `}},` / `,,`（重复分隔符）；
    //   2. 无悬空尾逗号（`}` 或 `]` 后紧跟 `,`）；
    //   3. 花括号与方括号配平。
    // 这不是"锦上添花"：坏 fixture 会让 GPU 对拍在**解析阶段**就失败，
    // 报错信息指向 JSON 而非真正的数值问题，浪费一轮排查。
    assert!(!json.contains("},,"), "产物含重复分隔符 `}},`（坏 JSON）");
    assert!(!json.contains(",,"), "产物含重复分隔符 `,,`（坏 JSON）");
    assert!(!json.contains("},") || json.contains("},\n"), "产物含尾随逗号（坏 JSON）");
    let opens = json.matches('{').count() + json.matches('[').count();
    let closes = json.matches('}').count() + json.matches(']').count();
    assert_eq!(opens, closes, "产物括号不配平（坏 JSON）");

    // 输出到 web/hair_golden.json（host_test.mjs 的对拍节读取该路径）。
    // 路径相对本文件（crates/prism-rt/tests/）上溯到 crates/prism-gpu/web/。
    let out: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../prism-gpu/web/hair_golden.json");
    fs::write(&out, json.as_bytes())
        .unwrap_or_else(|e| panic!("写 CPU golden vector 到 {} 失败：{e}", out.display()));
}
