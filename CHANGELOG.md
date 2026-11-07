# Changelog

本文件遵循兼容性宪章 C10/C15：PATCH 级 bug 修复若造成输出字节变化，
必须逐条进顶部固定栏目《渲染行为变化清单》；每条含
「受影响特性 / 变化原因 / 视觉影响描述 / 是否有开关」四要素。

---

## 渲染行为变化清单

### R-2026-xx-14：`config.ocio` 生效时色彩管理改走 Q24 定点 1D/3D LUT 链（V2.5-B1）

- **受影响特性**：`prism-app render --ocio <CONFIG.OCIO>` 与 v1 CLI /
  `prism_core::Scene.attach.ocio_config` 显式指定的 RGBA8 出片帧；
  `prism-postfx` 既有 `ColorManage::apply_color_with_ocio` 色彩链。
- **变化原因**：E-7 / 需求池 V2.5-B1 收官——此前 ACES 路径无任何 OCIO 配置入口。
  现自研 `config.ocio` 解析（`colorspaces` / `roles` / `displays`+`views` / `looks`
  + File / Matrix / Exponent / Log / Range / Group 六类 transform），烘焙为 Q24 定点
  1D shaper + 3D cube LUT（纯整数求值、零 libm、零线程），有 config 时在既有
  OETF 编码前插一遍 LUT 变换。
- **视觉影响描述**：**旧值 → 新值**，仅当显式给出非空 config 时帧字节变化（按
  config 的 view/look 链做色彩变换）；**无 config / 空 config 时 `apply_color_with_ocio(c, None)`
  直调 `apply_color(c)`，与既有 ACES 路径 `f32::to_bits()` 逐位相同**，
  `v25_b1_ocio_color::v1_no_config_matches_aces_path_bitwise` 与
  `v25_b1_ocio::v1_empty_config_is_noop` 双证。LUT vs 直接求值在 17³=4913 网格点
  最大偏差 **3.590e-7 ≤ 1e-6**（`v3_lut_vs_direct_evaluation`），双跑逐字节一致。
- **是否有开关**：有——仅 `--ocio <CONFIG.OCIO>` 或 `Scene.attach.ocio_config: Some(path)`
  且 config 非空时进入 LUT 分支；`None` / 空 config 走原 ACES 路径，字节不变。

### R-2026-xx-12：`MaterialKind::Hair` 续弹改走按 f 的 Marschner sampler（旧 Disney 提案退役）

- **受影响特性**：`prism-rt` 路径积分（`integrator.rs`）与 CPU wavefront
  （`wavefront.rs`）里毛发材质的间接续弹方向 / pdf / throughput。
- **变化原因**：R-6② 治本——旧实现方向用 Disney 提案、取值用 Marschner，
  估计量无偏但 pdf 不贴 f；新 sampler 按三瓣权重选瓣、纵向高斯采角、方位
  偏移采角，pdf 与 `eval_full` 同源并传入 MIS / throughput。f 数学零改动。
- **视觉影响描述**：**旧值 → 新值**，毛发场景间接光方向分布与收敛速度改变；
  `t10_marschner_sampler` S5 单股 4×4 spp=2 旧实现帧 vs 新 sampler 帧
  `differing_pixels = 16/16`、最大通道差 1.597087e0；S2 白炉相对差由
  2.472e-4 降到 8.181e-5，S1 ∫pdf dω = 0.999999955，两跑逐字节一致。
- **是否有开关**：无（`MaterialKind::Hair` 单一路径；非毛发材质输出逐位不变）。

### R-2026-xx-13：wavefront 求交接入 112 B 实例层级（Vulkan + WebGPU 双后端）

- **受影响特性**：`prism-gpu` wavefront 路径模式的 `shade` / `extend` /
  `shadow_ray` 求交（`rt_scene.wgsl::scene_intersect`）；含实例的场景帧。
- **变化原因**：G-8① 治本——此前 wavefront 只线性扫描 `tris`，实例表不进求交。
  现按 `prim_mat` 尾部 `W8I1` 描述符逐条逆仿射射线、按 `geom_id` 扫局部
  BLAS、法线 `det(A)·A⁻ᵀ·n` 回世界；`vulkan/wavefront.rs` 与
  `web/scene_gpu.rs` 同步消费同一份 112 B 编码与描述符校验。
- **视觉影响描述**：**旧值 → 新值**，实例场景从渲染错误变为与 CPU 权威
  逐位一致：I1 llvmpipe `0/256`、I2 浏览器 WebGPU `0/256`、I3 双跑 `0/256`；
  故意改错仿射 `13/256` 必 FAIL。无实例场景 `plain == n`，与改动前逐位相同。
- **是否有开关**：有——仅当 `prim_mat` 尾部存在 `W8I1` 描述符时进入实例分支；
  无实例的请求走原线性扫描，字节不变。

### R-2026-xx-10：`rt_wavefront.wgsl` 补环境光 NEE 腿（此前只有逃逸臂 ⇒ 环境直接光整条缺失）

- **受影响特性**：`prism-gpu` wavefront 路径模式 `shade` 段的环境光贡献。
- **变化原因**：CPU 权威 `integrator::nee_env_surface`（CDF 采样环境直接光 +
  `power_heuristic` MIS）在 WGSL wavefront 里没有对应腿——只有 `env_escape`
  逃逸臂。R-10 V1 白炉逐位对拍暴露（像素 18：CPU ~1.011 vs GPU ~0.972）。
- **视觉影响描述**：带环境贴图的 wavefront 帧直接光从缺失到正确，能量分布改变；
  修复后与 CPU 参考 8 帧逐位一致。7 份 `.spv` 已随源同步重生成（`spv_freshness` 7/7）。
- **是否有开关**：无（缺陷修复；无环境贴图的场景该分支不进入，输出不变）。

### R-2026-xx-11：wavefront 路径模式 golden 对拍判据从「逐位」明确为「相对差 ≤ 2e-6」（FMA 收缩跨设备）

- **受影响特性**：`gpu.wavefront_path` 浏览器 golden 判据（`qa_wavefront_golden.rs`
  的 CPU 侧生成器与 `web/abi.rs` 的 `c_gpu_wavefront_path`）。
- **变化原因**：SwiftShader（Dawn/WebGPU）与 naga→SPIR-V→llvmpipe 对**同一条
  WGSL** 的 FMA 收缩点不同，逐位判据在跨设备场景 46/256 采样点出现 1–3 ULP
  偏差——非实现错误，是后端编译器优化点的合法分歧（W2 跨设备判据同先例）。
- **视觉影响描述**：无（判据口径变化；相对差 ≤ 2e-6 远低于 8 bit 量化的 1 LSB），
  判据同时检查符号与有限性一致性。
- **是否有开关**：无。

### R-2026-xx-09：修复 `rt_wavefront` 路径模式的 NEE 口径（入射方向翻了两次 ⇒ 直接光恒为 0；`cos` 乘了两次）

- **受影响特性**：`prism-gpu` 的 **wavefront 五段式路径模式**（`rt_wavefront.wgsl` 的
  `shade` 段，跑在 `crates/prism-gpu/src/vulkan/wavefront.rs` 的派发器上）。
  受影响的是 `rt_shade.wgsl::nee_surface` 一条函数：
  旧式 `eval_bsdf(load_material(0u), -w_in, ls.wi, n) * cos_l`，
  新式 `eval_bsdf(load_material(prim), w_in, ls.wi, n)`。
  **CPU 产品路径（`prism-rt::integrator`）与 megakernel（`spirv_rt.rs`）的代码零改动**，
  故它们的输出字节不变。
- **变化原因**：两处口径错误，都是"编译通过、跑得出图、只是数字不对"的那一类：
  1. `bsdf.rs::eval` 的约定是第 2 参收**射线方向**（函数体内第一件事就是 `-w_in`）。
     旧调用传 `-w_in`，等于把入射方向翻了两次 ⇒ `cos_i = dot(dir, n) < 0` ⇒
     `eval_bsdf` 恒返回 0 ⇒ **NEE 整段失效**。真机现象：4 弹射图整幅 `rms = 0`
     （只剩自发光命中与逃逸背景）。
  2. `eval_bsdf` 的返回值已含 `* cos_o`（= `cos_l`），旧式再乘一次 `cos_l`，
     等效把 Lambert 项按 `cos²` 加权。
  3. 材质旧式硬编码 `load_material(0u)`，多材质场景下所有反射都被算成 0 号材质的颜色；
     改为**命中图元**的材质（`nee_surface` 新增 `prim` 形参）。
- **视觉影响描述**：**旧值 → 新值**（`prism-gpu/tests/qa_wavefront.rs`，64×48、
  1 盏球面灯、2 种材质、spp=1、seed `0x5EED1234`）：
  - 4 弹射：参照 RMS **0.0 → 0.08473**（旧值整幅全黑，`rms = 0` 会让 W3 的 PSNR 退化
    成"两张黑图相等"的 ∞，判据因此失去意义）；
  - 有光像素 **0/3072 → 1826/3072**；
  - 4 vs 8 弹射：**PSNR 47.23 dB / 相对 RMSE 0.0513**（旧值不可测）。
  能量方向只增不减（旧式少乘一次 `cos` 且 NEE 为 0），故这是**修复**而不是"调色"。
- **是否有开关**：有——①`WavefrontMode::Path` 才会走到 `nee_surface`；兼容模式
  （`CpuRefCompat`，W1 逐位对拍专用）与 megakernel 都不经过它，字节不受影响；
  ②`WFParams.nee_only != 0` 时 `shade` 跳过 MNEE 焦散段（与 CPU 侧同一条开关语义）；
  ③整条 wavefront 路径只在 `VulkanBackend::render_wavefront_frame` /
  `WebGpuBackend::render_wavefront` 被显式调用时执行，既有产品出片路径
  （`GpuDevice::render_scene` 走 megakernel）不受影响。

### R-2026-xx-08：修复 GI 存储布局对 `vec3<f32>` 的依赖（真机驱动 size 与规范不一致 → 字段整体错位）

- **受影响特性**：`prism-gpu` 实时 GI 的 GPU 路径存储布局——`gi_compute.wgsl` 的四个
  布局结构体 `GiPrim` / `GiParams` / `GiProbe` / `GiPoint`，以及新增的布局自检入口
  `gi_layout_census` + 新测试项 `gpu.gi_layout`。Rust 侧 `gi_pack_probes` /
  `gi_params_bytes` / `gi_unpack_sh` 的**偏移一个字节都没改**（布局零变化）。
- **变化原因**：真机（Android 12 / Chrome 144 内核 / Quark 10.18 / 真 GPU）实测：
  驱动对**结构体成员** `vec3<f32>` 的 `size` 处理与 WGSL 规范不一致（规范
  `size=12 / align=16`，真机表现为更大）→ 后续字段偏移整体重排：
  `GiProbe.normal`（应 `@16`）被推移到 `samples` 区（读到哨兵 `-1`），`samples`/`sh`
  被压缩（`samples[6]` 落到 `sh[0]` 的位置）。判定实验（沙箱内用真实 Rust 代码）：
  用候选 normal `(-1,-1,-1)` 跑 CPU 参考 `gi_probe_trace_ref`，**逐方向位掩码**
  `[0,0,1,0,1,0,0,0]` 与真机 GPU 的位掩码**逐位完全相同**；用正确 normal `(0,1,0)`
  得到全 1；用 `(0.9,0.8,0.7)`（`GI_TEST.env`）与 `(0,0,1)` 均不吻合。同时
  `gpu.params_roundtrip` 真机 **PASS**（256/256 字节回环一致）→ uniform 传输链无问题，
  排除"参数没到 GPU"。旁证：`gi_compute.wgsl` 是全仓**唯一**用 `array<vec3<f32>>`
  做存储布局的 shader，也是**唯一**真机 FAIL 的项；其余全部用 `array<vec4<f32>>` 的
  storage shader（`rt_wavefront` / `sim_particles` / `sim_sph` / `sim_cloth` /
  `scene_frame`）真机全 PASS。（`rt_wavefront.wgsl:79` 的 `array<vec3<f32>, 2>` 是
  函数返回值，不涉 buffer 布局，不受影响。）
  修复：把参与存储布局的 `vec3<f32>` 全部换成 `vec4<f32>`（第 4 分量恒 0）。
  `vec4` 的 `align/size` 都是 16，与原「`vec3` + 隐式/隐式填充」**逐字节等价**，
  故 Rust 侧全部偏移常量不变；shader 内对 vec3 字段的访问一律加 `.xyz`、写入
  一律 `vec4(xyz, 0.0)`，数学语义一分不改（`w=0` 不参与任何运算）。
  同时新增两侧静态门禁：`web/wgsl_lint.mjs` **规则 H**（存储布局禁 `array<vec3<f32>>`，
  函数返回类型白名单仅 `rt_wavefront.wgsl::cast_ray` 3 处）+ `web/host_test.mjs` 第 4 节
  同口径断言 + `web::abi::tests::gi_wgsl_layout_matches_abi_constants`（从 WGSL 源码
  按规范算逐字段偏移与 size，与 `GI_*_STRIDE_BYTES` 交叉核对）。
- **视觉影响描述**：这是**布局修复**，不改任何数学。**旧值 → 新值**（同一组固定输入）：
  - 新测试项 `gpu.gi_layout`：真机 `samples[1]`（= `GiProbe.normal`）**旧值读到 `-1`
    （samples 的哨兵）→ 新值应为 `(0,1,0)`**；`samples[3]`（`probe2.normal`）
    **旧值 `-1` → 新值 `(1,0,0)`**；`samples[6]` = `(probe_count, prim_count,
    env_radiance.x, albedo.x)` **旧值读到错位内容 → 新值 `(3, 2, 0.9, 0.8)`**。
  - `gpu.gi_compute` 遮挡统计 `visible_total`：真机**旧值 2/8 · 2/8 · 2/8（= 2-2-2）
    → 新值应为 8/8 · 4/8 · 4/8（= 8-4-4）**（与 CPU 参考一致）。
  - 本地（SwiftShader）预期**无变化**：`gpu.gi_compute` 的判据量与
    `gi_cpu_reference_fingerprint_matches_independent_js` 的指纹
    （`0x6adb7c92`）都**不变**——SwiftShader 严格按规范实现，vec3/vec4 两种写法
    在它上面逐字节等价。**这正是"客户端自洽对拍"的盲区：本地只能证明「没改坏」，
    证明不了「真机修好了」**，故本轮专门交付 `gpu.gi_layout` 作为真机判读锚点。
  - wasm 体积：**439,539 B → 453,223 B**（+13,684 B，上限 458,752 B，余量 5,529 B）。
    增量归因：新增 entry `gi_layout_census` 及四个结构体的布局表注释（WGSL 源码
    原文进数据段）+ `gpu.gi_layout` 的 run/collect 代码与诊断文案。已做无损压缩
    （布局表由 Markdown 表格改为单行偏移表，诊断成功路径文案精简），未改任何阈值。
- **是否有开关**：无。旧行为不保留——`vec3<f32>` 做存储布局在真机上是**不可用**的
  （驱动与规范分歧），保留开关等于保留崩溃路径。若某设备真的严格按规范实现，
  vec4 写法与旧写法逐字节等价，行为完全相同，故不需要开关。
- **真机验证（2026-10-30，已闭环，4 个浏览器环境 / 2 台设备 / 2 个产物版本，各 19/19）**：
  旧产物 `?v=54d66a8e` 在 **三星 Galaxy S10（SM-G9730 / 骁龙 855 / Adreno 640 / Android 12 / Quark）**、
  **荣耀 X40（骁龙 695 / Adreno 619）**、**Edge for Android 153（`EdgA/153`，Chrome/153 内核）**
  上各 19/19；新产物 `?v=8772008b` 在 **SamsungBrowser 30.1（Chromium 143）** 上 19/19。
  设备间是**同一份 wasm 字节**——换机型换驱动代际仍全绿，排除「针对某台设备调参」。
  （两台的 GPU 同为 Qualcomm Adreno，**不构成"跨厂商"证据**；Mali / PowerVR 未覆盖。）
  **新产物那轮首次拿到 `adapter` 字段**：`vendor=qualcomm` / `architecture=adreno-6xx` /
  `isFallbackAdapter=false` / `subgroup 64·128`（SwiftShader 为 `4·4`）——**硬件适配器直接确认**，
  不再依赖推断。上述"新值"全部命中：`gpu.gi_layout` 的 `samples[1]` 读出
  `(0,1,0)`（不再是 `-1`）、`gpu.gi_compute` 的 `visible_total` 为 `8/8 · 4/8 · 4/8`、
  `gpu.scene_frame` PSNR 52 dB 档。**桌面档仍零实测**（四环境均移动端）。
  **结论：责任在驱动侧**（偏离 WGSL 对 `vec3<f32>` 的 `size=12/align=16` 规定），
  本仓按规范打包的字节自始正确；改为 `vec4` 后同一设备同一链路立刻全绿。
  证据：`docs/QA/real-device-webgpu-2026-10-30.md` §9。**未改判据、未降 `F32_TOL`、
  未把失败项改 Skip。**

### R-2026-xx-07：修复 GI SDF 的球分支（`q.x` 是沿 x 的无限大板，不是球）+ 容量填充图元改为不可能命中

- **受影响特性**：`prism-gpu` 实时 GI 的 SDF 遮挡求值——CPU 参考
  `crates/prism-gpu/src/web/abi.rs::gi_prim_dist_ref` 与 GPU 镜像
  `crates/prism-gpu/shaders/gi_compute.wgsl::gi_prim_dist` 的**球分支**（`kind == 0`）。
  连带影响 `gpu.gi_compute` 的全部判据量：逐方向可见性 → SH-L2 系数 → 查询点辐照度；
  以及测试侧的容量填充图元 `GI_PRIMS[2..4]`。
- **变化原因**：球分支返回 `q[0]`，即 `|p.x − c.x| − r`——**只与 x 有关，是沿 x 的无限大板**。
  原注释「q 的三分量相等，任取一分量即 |p−c|」不成立：`abs(p−c) − r` 的三个分量
  只有在「仅一个坐标有差」时才相等。后果是测试里**「球」这个图元从未被真实测到**，
  它实际是一个贯穿 y/z 的板：任意 x∈[c.x−r, c.x+r] 的点都被判为"在体内"。
  真机实测（`gpu.gi_compute` FAIL、GPU 侧 `sh` 整块为零）暴露后，两端**同步**改为
  标准球 SDF `|p − c| − r`（同式同精度路径：先差、再平方和、再 `sqrt`、再减半径；
  **不用 `powi`**，以与 WGSL 的 `x*x` 同一条代码生成路径）。两端必须一起改——
  否则 CPU 参考与 WGSL 不再同式，跨端判对失去意义。
  同时把**容量填充图元**从「球心原点 + r=1」改为「半径 0 + 心在 −Y 轴 1e6」：
  填充数据只占 `GiParams.prims` 的定长容量（`prim_count=2` 时不用），但一旦
  `prim_count` 在任一环节被读成 4，原点 r=1 的球就会把探针起点判成"在球内"
  （`|p|−1 ≈ 5e−7 ≤ hit_eps`）→ 该探针 8/8 全遮挡 → `samples`/`sh`/`irradiance`
  整块零值。安全填充的 SDF 恒 ≈1e6（远大于 `t_max`=20），即使被误用也绝不命中。
- **视觉影响描述**：bug 修复导致**判据参考值变化**（不是画面变化——本路径的 GPU 结果
  与 CPU 参考比对，两者现在都按真球求值）。**旧值 → 新值**（同一组 `GI_TEST` 固定输入）：
  - `web::abi::tests::gi_diag_block_distinguishes_zero_from_reference`：
    逐探针可见方向数 `[5, 0, 1]` → **`[8, 4, 4]`**（probe 0 由 5/8 变 8/8；
    probe 1 由 0/8 变 4/8——旧板把起点 (1.001, 1, 0) 判成 −1e−3「在体内」，
    真球距离是 0.414「在体外」；probe 2 由 1/8 变 4/8）。
    probe 0 的 `sh[0]` `(0.997005, 0.886227, 0.775449)` → **`(1.595208, 1.417963, 1.240718)`**。
  - `gpu.gi_compute` 的 `visible_total`（24 个方向里的可见数）：**6/24 → 16/24**。
  - `web::abi::tests::gi_cpu_reference_fingerprint_matches_independent_js` 的跨实现锚：
    **`0xaf8429a3` → `0x6adb7c92`**。Rust CPU 参考与 `web/host_test.mjs` 的 JS 第三份
    实现**两端同步更新**，指纹仍逐位相等（`js_gi_fingerprint=0x6adb7c92`）。
  - 新增回归钉子 `web::abi::tests::gi_prim_count_misread_does_not_blind_the_scene`
    钉住机制：旧球 SDF + 旧填充在 `prim_count=4` 时 CPU 侧实测 `[0, 0, 0]`
    （三探针全 8/8 遮挡、9 个 SH 系数整块为零），而 `prim_count=2` 是 `[5, 0, 1]`
    ——与真机「GPU 全零 vs CPU `0.9970052`，60 个字段超容差」的形态一致；
    换成安全填充后 `prim_count=2` 与 `=4` **逐位相同**（`[8, 4, 4]`）。
- **为什么 V1.0 真机能 PASS、V2.2 真机不能（本版最重要的一条结论）**：
  把 V1.0 原件（错误球 SDF，未修复）重新构建后在 **SwiftShader 软件适配器**上跑
  `web/realgpu_check.mjs`：**17/17 全 PASS，含 `gpu.gi_compute`**。而同一份构建
  （432,052 B，字节可比）在真机（Android 10 / Chromium 143 内核 / 真 GPU）上 FAIL、
  GPU 侧 `samples`/`sh`/`irradiance` 整块零值。
  ⇒ 该缺陷的性质是**病态 SDF 的球 tracing**：`q.x` 这张板在 y/z 方向梯度恒为 0
  （SDF 不随射线前进变化），迭代收敛与否对浮点实现细节极度敏感——软件适配器的浮点
  行为接近 CPU，两端"同错得一致"；真 GPU 的 FMA 收缩/精度路径一变，就整体翻到
  "全遮挡"一侧。**V1.0 那次全 PASS 是碰巧落在与 CPU 同一侧，不是验对了。**
  ⇒ 方法论结论：**「CPU 参考 vs GPU 镜像」的自洽对拍，只有在两端算法本身正确时才有
  判别力。两端同错时它恒绿，且软件适配器（SwiftShader）永远无法暴露——只有真硬件
  GPU 能。** 该结论已写入 `docs/QA/real-device-webgpu-2026-10-30.md`，并作为
  `gpu.gi_compute` 保留「真机档」而非降级为软件适配器档的理由。
- **是否有开关**：无。旧行为不保留——`q.x` 不是球的 SDF，是缺陷；保留开关等于保留
  一个"看起来测了球、实际测了板"的假判据。

### R-2026-xx-06：修复折射界面法线朝向与 Fresnel 余弦口径（玻璃/折射体曾是「不透明遮挡」，ior 零效应）

- **受影响特性**：`prism-rt` 电介质（`transmission > 0`）的折射瓣——`crates/prism-rt/src/bsdf.rs`
  的 `Bsdf::sample` 中 transmission 分支（`n_in` / `cos_i_s` 两处口径），以及
  `crates/prism-rt/src/wavefront.rs` 的镜像同步实现。影响一切含玻璃/折射体的路径：
  CPU 参考后端、wavefront 及经它们出图的作品帧。
- **变化原因**：`sample` 收到的 `n` 是**几何外向**法线，但介质内出射时入射侧的法线
  应指向介质内部。旧实现两处都直接用了 `n`：
  ① `refract_dir(w_in, n, eta)` 在 `from_inside` 下 `cos_i = −w_in·n < 0`，返回一条
  **折回介质内部**的伪方向（`t·n = −cos_t < 0`），出射面永远走不通；
  ② Fresnel 余弦 `cos_v = wo·n` 在介质内为负、被 `.max(0.0)` 钳成 0，于是
  `schlick(F0, 0) = 1.0` ⇒ `p_reflect ≈ 0.999`（出射面几乎**恒走全反射**），路径在玻璃内
  反复弹射直到 `max_bounces` 耗尽。两条叠加的表现即「玻璃球是一个不透明遮挡体」，
  且 **ior 对图像零效应**。修复：`from_inside` 时把界面法线翻转为 `−n`（`n_in`），
  折射调用与所有 Fresnel 余弦统一用该法线口径。反射对翻转不敏感
  （`reflect_dir(w, −n) ≡ reflect_dir(w, n)`），故只改折射侧。
- **视觉影响描述**：bug 修复导致**输出字节变化**，仅限含折射体的场景——玻璃从
  「不透明遮挡」变为可透视、可聚焦，色散随之可见。**旧值 → 新值**（测试
  `qa_spectral_degeneracy::dispersion_glass_centroid_separation`）：
  旧版 `ior 1.0 / 1.5 / 2.5` 三帧**逐位相同**（色散完全不可见，等效分离度 = 0）→
  新版色心分离 **2.504 px**（阈值 1.0；红簇权 994.41 / 蓝簇权 353.53 / 主导像素 360，
  40×40 spp256 确定性锚定）；`cauchy_b = 0` 对照组 0.917 px 证明判据本身有效。
  非折射材质（金属 / 纯漫反射 / 不透射电介质）输出不变：白炉能量、电介质上界、
  T12 多散射四钉子（`multiscatter_compensation_monotonic_in_f0` /
  `white_metal_compensation_equals_white_furnace_form` /
  `old_colored_multiscatter_form_is_dead` / `eval_is_deterministic`）零回归，
  `prism-rt` 275 passed / 0 failed。**既有 golden 零位移的真相**：现有 golden 未覆盖
  玻璃出射路径（与 R-2026-xx-05 同属测试盲区），故本次位移由新增的
  `qa_spectral_degeneracy`（3/3）承接，而非既有基线漂移。
- **是否有开关**：无。旧行为不保留——旧形式是缺陷（折射方向折回介质内部），
  不是可选口径。

### R-2026-xx-05：修复 Disney BSDF 多散射补偿形式错误（高饱和金属低粗糙度掠射角偏补色）

- **受影响特性**：`prism-rt` 的 Disney BSDF 多散射能量补偿——`crates/prism-rt/src/bsdf.rs`
  的 `Bsdf::eval` 补偿项 `ms`，以及 `energy_dir` / `energy_avg` 的出口。影响一切经
  BSDF 着色的路径：CPU 参考后端光追、Vulkan 光追 kernel 的材质求值，以及
  `metalness→1` 的高饱和金属工作流（金、铜）。
- **变化原因**：旧实现按通道回填
  `ms_ch = (1−E_o)(1−E_i)·f0_ch / (π·(1−E_avg))`，其中 `E` 取自
  `energy_dir(·, ·, f0_ch)`——**每个通道用自己的 F0 查表**。低 F0 通道的 `(1−E)`
  更大、`1−E_avg` 更小，二者相除把回填量在 `f0→0` 侧放大，补偿幅度不随 F0 单调。
  修复改为 Fdez-Agüera 2019 彩色导体形式：幅度取**白炉表**（F0=1）的
  `(1−E_o)(1−E_i)/(π(1−E_avg))`，色相由半球平均 Fresnel
  `F_avg = f0 + (1−f0)/21` 承担；`F_avg=1` 时逐字退化为白炉标准形。
  同时在 `energy_dir` / `energy_avg` 出口加 `clamp(0,1)` 作**防御**——当前 LUT 上
  该 clamp **零触发**（实测金/铜/银在 α∈[0.05,0.35]、μ∈[0.05,0.25] 全域 raw < 1），
  它不是本次修复的主因，主因是补偿形式置换。
- **视觉影响描述**：bug 修复导致**输出字节变化**，仅限彩色金属（尤其高饱和）在
  低粗糙度、掠射角下的多散射补偿项。实测金（f0 = 1.000/0.766/0.336）在
  α=0.05、cosθo=0.25、cosθi=0.25 时，补偿量旧 → 新为
  `[0.019914, 0.046192, 0.050627]` → `[0.019914, 0.012003, 0.002676]`：
  **R 通道逐位不变（比值 1.000）、G 通道旧/新 = 3.85×、B 通道 = 18.92×**；
  更极端的掠射（cosθo=0.1、cosθi=0.8）B 通道达 **49.83×**。即旧版把蓝绿通道
  补了 4–50 倍于正确值的能量，整球偏成补色（金球发青绿）。铜
  （0.955/0.638/0.538）同向，α=0.35、cosθo=0.25、cosθi=0.25 时三通道比
  1.19/3.35/4.52。白金属（f0=1）逐位不变，既有白炉断言与电介质上界断言不受影响。
- **是否有开关**：无。旧行为不保留——旧形式本身是缺陷，不是可选口径。
- **既有 golden 零位移的真相**：本修复未触碰任何 golden 文件，全量回归
  3929 passed / 0 failed。但这不是「无位移」，而是**测试盲区**：现有测试只覆盖
  白金属（f0=1）与彩色电介质（metal=0），从未覆盖高饱和彩色金属。钉子补齐
  归 V2.2-C-4。
- **分发包说明**：`prism-engine-v2.1-source.txt` 打包于本修复之前，**含此缺陷**；
  源码树修复按计划随 v2.2 分发，不单独发 v2.1.1。v2.2 分发说明须显式载明此行。
- **验收（已闭环）**：钉子随 V2.2-C-4（T12）落地于 `crates/prism-rt/src/bsdf.rs` 测试段 ——
  `multiscatter_compensation_monotonic_in_f0`（金/铜 α×cosθ 网格补偿随 F0 单调不减）、
  `white_metal_compensation_equals_white_furnace_form`（白金属 f0=1 补偿逐位等于白炉标准形）、
  `old_colored_multiscatter_form_is_dead`（旧逐通道 F0 查表回填形式负向自证，必须判红）、
  `eval_is_deterministic`（两遍求值逐位一致，G3）。既有白炉 / 电介质上界断言零回归。
  发布说明草稿见 `docs/v2.2-release-note-draft.md`；九语言 README 同步归 PM 的 T02。

### R-2026-xx-03：T4 结清——sRGB ↔ ACEScg（AP1）矩阵由 Hill 拟合件换为官方原色推导（V2.1 流 C）

- **受影响特性**：`prism-math` 矩阵常量 `colorspace::SRGB_TO_ACESCG` /
  `colorspace::ACESCG_TO_SRGB`（`crates/prism-math/src/colorspace.rs`），
  以及一切消费这两块矩阵的路径：`aces.rs` 的 ACES 风格化调色（`ACESInput`/
  `ACESOutput` 拟合链）、`colorspace_registry` 的 legacy/derived 对拍照、
  经 `TransformChain` 的 `Srgb ↔ AcesCg` 互转。
- **变化原因**：T4 债务结清（作者拍板，四件套）。旧主常量是 Stephen Hill
  拟合件，并非由官方 AP1 原色 + Bradford(D65→D60) 推导，实测偏差
  **3.2045e-2 @ [2][2]**（f32/f64 双口径）。本版按官方原色重录主常量
  （同名换值，先例 R-2026-xx-02），逆阵 `ACESCG_TO_SRGB` 由新主常量经
  f64 精确求逆后收窄到 f32；旧 Hill 值保留为
  `colorspace::SRGB_TO_ACESCG_FITTED_HILL` **显式可选口径**（数值一字未改，
  行和仍精确为 1）。
- **视觉影响描述**：bug 修复导致**输出字节变化**——凡经 sRGB↔AP1 的线性域
  互转结果改变。矩阵逐元素最大位移 3.2e-2 量级（`[2][2]` 0.8378→0.8698，
  `[0][0]` 0.5972→0.6131；逆阵 `[0][0]` 1.7647→1.7051）；ACES 指纹 27 字
  全量重录（非黑样本全部位移）。
- **是否有开关**：无运行时开关。旧口径以 `SRGB_TO_ACESCG_FITTED_HILL`
  常量保留（编译期显式引用），二者**不可混用**。
- **黄金值重录清单（G6 逐条对应）**：

  | # | 测试名 | 旧 → 新 | 性质 |
  |---|--------|---------|------|
  | 1 | `c2_color_golden::aces_golden_file_stable`（指纹常量 `ACES_FINGERPRINT`） | `ACES_FINGERPRINT` 27 字全量重录：18% 灰 `3DD8_403D/3A/3B`→`3DD8_402C/3D/44`；中灰 `3EBF_A557/A556/A556`→`3EBF_A558/A556/A554`；白 `3F1E_7E5A/59/5B`→`3F1E_7E63/59/54`；红 `3F41_0AF6`→`3F3F_786E`；绿 `3D6A_D79F/3F24_5E27`→`3D52_E8EA/3F23_F4F4`；蓝 `3F2B_7813`→`3F28_D13F`；超亮 `3F68_B522/20/21`→`3F68_B53A/21/12`；混合 `3D1A_CC7B/3E5B_0065/3F02_9C41`→`3D1B_8FA0/3E5A_E656/3F01_D3D0` | 真实输出位移 |
  | 2 | `qa_edge_attack_math::aces_fitted_vs_independent_reference` | 测试内独立参考矩阵常量同步重录为新口径 | 测试装置同步（无输出位移） |
  | 3 | `aces::tests::reference_aces_fitted` | 同上 | 测试装置同步（无输出位移） |
  | 4 | `colorspace::tests::legacy_srgb_acescg_matrices_debt_t4_is_named` | 结清锁更名 `..._settled_t4_is_named`：判定由「偏差必须 > 1e-2（债务存在）」翻转为「偏差 ≤ 1e-2（已收敛）+ 逆阵可重算 + Hill 旧值未改」 | 债务锁 → 结清锁 |
  | 5 | `colorspace_registry::tests::legacy_constant_is_a_fitted_matrix_not_a_primaries_derivation` | 取值对象由 `SRGB_TO_ACESCG` 改为 `SRGB_TO_ACESCG_FITTED_HILL`，仍锁 3.2e-2 差异 | 参照物随结清换位 |
  | 6 | `colorspace_registry::tests::srgb_to_acescg_fitted_hill_alias_is_bit_identical_to_the_legacy_name` | **删除**（别名同值语义随结清失效），由 `..._preserved_and_diverged_from_recut_main` 接替（锁旧值一字未改 + 二者已分流） | 断言语义翻转 |

验收：`cargo test -p prism-math --release` 全绿；`legacy_srgb_acescg_matrices_settled_t4_is_named`
每次运行重算复核（官方公布矩阵对齐 < 1e-6、主常量收敛 ≤ 1e-2、逆阵 5e-7）。

### R-2026-xx-04：修复 BakedSdf 查询的 brick 边缘插值缺陷（查询点被吸附到格点）

- **受影响特性**：`prism-geometry` 的 `BakedSdf::dist`（烘焙 SDF 的任意点查询，
  `crates/prism-geometry/src/mesh_sdf.rs`），影响一切消费烘焙 SDF 的几何/渲染
  路径（SDF 图元求值、mesh→SDF 距离查询）。
- **变化原因**：查询把 **brick 局部**坐标 clamp 到 `BRICK−2`、权重 clamp 到 1——
  单元落在 brick 末格（局部 `[7,8)`）时查询点被吸附到最近格点，插值不是在被查
  点上求值；brick 无 apron，跨 brick 单元完全无插值。1-Lipschitz 场下定位偏差
  直接变成距离误差，实测最坏 **|SDF| = 0.0693（h=0.05）@ 真实表面点**
  （误差恰等于 `|p−q|`，q 为被吸附后的求值点）。
- **视觉影响描述**：bug 修复导致**几何查询结果变化**——brick 边界邻域的 SDF
  值/等值面位置修正，最坏 6.9e-2（h=0.05）量级的定位误差消除（修复后同点
  1.4e-3）。既有 golden 无位移（本版全仓测试证实），但 SDF 精度行为已变化。
- **是否有开关**：无。
- **验收**：`mesh_sdf::tests::r3_hausdorff_bound_holds_and_tightens_with_resolution`
  （新增 `≤ √3/2·h` 更严断言 + 实测值表）+
  `mesh_sdf::tests::brick_edge_cells_resolve_global_lattice_not_clamped_local`
  （专挑 ≥2 轴同时落末格的 177 个表面点：旧实现最坏 0.06929246 超阈，新实现
  1.4464e-3，余量 ~30×）。

### R-2026-xx-02：修复 ACEScg（AP1）→ ACES2065-1（AP0）色彩矩阵（按官方 S-2014-004 重录 + 精确逆阵重算）

- **受影响特性**：`prism-math` 矩阵常量 `colorspace::ACESCG_TO_ACES2065` /
  `colorspace::ACES2065_TO_ACESCG`（`crates/prism-math/src/colorspace.rs`），
  以及一切经 `TransformChain` 的 `AcesCg ↔ Aces2065` 线性域互转
  （渲染侧可达 postfx color 相关路径，如 `prism-postfx` `ColorManage`
  经 `effects::color` 的 `apply_color` 消费这两块矩阵）。
- **变化原因**：旧轨常量与 ACES 官方 S-2014-004 矩阵**坐实不符**——
  第三行整体抄错（旧 `[0.0283, 0.1292, 0.8593]` vs 官方
  `[-0.0055, 0.0040, 1.0015]`，逐元素 max ≈ 1.42e-1），第一二行另有
  4 位小数级偏差（~3e-4 ~ 4e-3）；两常量互为精确逆，属「自洽地错了
  一对」。本修复按 S-2014-004 官方全矩阵**重录三行**（逐位取官方值），
  逆矩阵由官方矩阵经 f64 求逆（`colorspace_registry::invert3`）**精确
  重算**（非照抄旧值），并由测试 `aces2065_to_acescg_is_recomputed_inverse`
  每次运行重算复核。
- **视觉影响描述**：bug 修复导致**输出字节变化**——凡经 AP1↔AP0 转换的
  结果改变（旧第三行错误使 AP0 输出的绿/蓝分量偏差达 ~1e-1 量级）。
  修复后旧轨直接矩阵与新轨 `colorspace_registry`（原色推导）一致性实测：
  正向 worst ≈ 1.9e-6、逆向 worst ≈ 2.9e-6（预期 1e-5 内，测试
  `aces_ap1_ap0_matches_registry_derivation` 锁定，不调容差）。
  白点守恒：AP1/AP0 同为 D60 名义白点，官方矩阵行和恰为 1.0（实测偏离
  < 1e-7），白点仍映射白点（`white_point_is_preserved` 容差由 1e-4
  加严至 1e-6）。
- **是否有开关**：无开关。属 bug 修复（宪章 C10：PATCH 修 bug——本修复
  导致输出字节变化，故按 C15 逐条进本清单）。重渲兜底：引擎为静态
  二进制，旧版本可并存安装（C10 逃生门），可用旧版本重渲既有基线。

验收：`crates/prism-math/src/colorspace.rs` 单测（互逆对
`matrices_are_inverse_pairs` / 白点守恒加严 / 逆阵重算验证 / 新旧轨
一致性对拍）。

### R-2026-xx-01：`MergeOp::Over` 半透明层叠半透明层改用标准 Porter-Duff 预乘混合

- **受影响特性**：`prism-core` 合成算子 `MergeOp::Over`
  （`compose::blend_pixel` / `compose::merge` / `ComposeNode` 全部求值路径），
  以及渲染侧经 `nodes::ComposeMergeNode`、`nodes::DepthCompositeNode` 的
  over 合成。
- **变化原因**：修复 over 的两处偏离标准 Porter-Duff 的缺陷——缺 αa
  权重（下层色按 `1` 而非 `αa` 加权）与缺归一化（输出未除以合成 alpha）。
  修复按规格《值域层-预乘混合规格》定义 2.1–2.3 在预乘混合域
  （`C = α·c`）实施；对外 `LayerBuf` 仍为直通 RGBA，接口与存储格式不变。
- **视觉影响描述**：仅**半透明层叠半透明层**（`αa < 1` 且 `αb < 1`）的
  象限输出改变（修复前该象限结果非标准 Porter-Duff；嵌套越深误差复利
  越深，实测 legacy 两层即 ~2e-2）。以下象限**逐位不变**：
  上层不透明（`αb = 1` ⟹ 输出 = 上层，最常见用例）、双不透明
  （⟹ 上层）、下层不透明 + 上层半透明、任一层全透明（⟹ 另一层，
  幂二 alpha 下逐位）。其余算子（plus/screen/multiply/mask）公式零
  改动，黄金值不变。修复后 over 满足结合律（规格命题 2.4），误差结构
  从复利 O(nε) 退化为单层 O(ε)（注 2.4.1），覆盖类子树可走
  `eval_flat` 扁平化路径。
- **是否有开关**：无开关。属 bug 修复（宪章 C10：PATCH 修 bug 不改
  行为——本修复导致输出字节变化，故按 C15 逐条进本清单；影响面为
  「半透明叠半透明」单一象限，维持 PATCH 级）。重渲兜底：引擎为静态
  二进制，旧版本可并存安装（C10 逃生门），可用旧版本重渲既有基线。

验收：`crates/prism-core/tests/qa_over_premultiplied.rs`（规格 §5 六条
全量断言，含 f32 逐位结合不可达的如实记录）+
`crates/prism-core/src/compose.rs` 单测（新语义钉子 4 枚）。

## 性能变更清单（非 C15：输出**逐字节不变**，不改画质）

### P-2026-xx-01：W0-C 性能——1080p/16spp 单帧 10.16 s → 0.72 s（4 核 14.0×，单线程 3.68×）

- **受影响特性**：默认档（`prism-rt` 路径追踪，`RtBackend`）的求值速度与线程模型。
  新增 `[profile.release]`、`prism-render::rt_tiles`（tile 并行）、`prism-rt` 采样器
  的三处整数件重写、相机零长快门快路径、`Scene::intersect` 内联。
- **为何非 C15**：C15 只登记"导致输出字节变化的修复"。本条**输出零字节变化**——
  4 核与 1 核、与改动前的原始单线程版，`cmp` 8,294,400 B **全等**
  （`fnv64 = 2f21205ca22f225b`）。改的是"怎么算"，不是"算出什么"。
- **变化原因与实测**（1080p / 16 spp / 6 弹射，单球 + 方向光）：
  | 层 | 改动 | 实测 |
  |---|---|---|
  | L1 | `[profile.release]` | **原案 `lto="fat"`+`cgu=1` 是 −15% 回归**，改取 `thin`+`cgu=16` ⇒ 1.03× |
  | L2 | 无 `perf`，改「只计数 + 单位成本」双测 | 定位热点：**采样器 60.7%**、`cast_ray` 24.7%、求交 7.0%、BSDF 5.7% |
  | L3 | 固定 32×32 tile + 固定序合并 | 1→4 线程 **3.82×**（并行效率 95.5%） |
  | L4 | 像素级 SIMD | **判定不做**：求交+BSDF 合计仅 12.7%，SIMD 上限 1.06×；且 sampling 的位扫描是数据相关循环，向量化更慢 |
  | L5 | 采样器三件套 + 零长快门 + `intersect` 内联 | 单线程 9.87 → **2.77 s**（3.56×） |
- **确定性口径**：新增 7 项 `rt_tiles` 关卡（`cargo test -p prism-render --lib rt_tiles`）。
  其中 `tiled_matches_pre_tiling_reference_loop` 把 **W0-C 之前的逐像素循环原样写在
  测试里**当参考实现，与 tile 版在 1/4/16 线程下比字节。归约（spp 均值、点积、
  BRDF 求和）**一行未动**，tile 边界 = 像素边界 ⇒ 无跨 tile 浮点合并。
- **是否有开关**：无。`RtBackend::threads` 可调（`0` = 自动 = `available_parallelism`），
  但**线程数不进结果**——它只决定谁在算，不决定算出什么，故不构成"行为开关"。
- **性能数字**（判据口径）：W0-C 前 10.1596 s → 后 **0.7239 s**（4 核）/ **2.7664 s**
  （1 核）。3 分钟 4320 帧：**12.8 h → 52.2 min**。`--fast` 未受影响
  （CLI 实测 78 ms，基线 79.6 ms）。
- **未达标项（如实登记）**：出口判据 1080p/16spp **≤ 0.5 s 未达成**（本机 0.724 s）。
  根因是**运行容器 `cpu.max = 400000 100000`，只有 4 核配额**（`nproc` 报 32 但配额 4），
  4 核理论上限 0.692 s。按实测 95.5% 并行效率外推，**0.5 s 需要约 6 核**。
  详见 `docs/v2.5-W0-算法选型报告.md` §14。

## 能力扩容清单（非 C15：特性扩容——此前被拒场景现在可渲染）

### F-2026-xx-05：V2.5 能力扩容 T9（R-13 自适应采样接产品路径 + 浏览器 wavefront 运行时腿）

- **扩容内容**：①`prism-app render --adaptive <TOTAL>` 新档——
  `prism-postfx::AdaptiveRtBackend`（同一 `prism-rt` 渲染核 + A5/C2 逐像素预算
  分配；与均匀档共用 `translate_scene`/相机/aesthetic 全部实现；预算不足
  结构化报错不静默降级）；②浏览器 WebGPU wavefront 运行时腿全通
  （realgpu 21/21）：`gpu.wavefront_path` golden 对拍、`gpu.scene_frame_v3`
  V3 剖面（含实例表 + 实例 TLAS）、`gpu.rt_wavefront_multi`——期间修复
  `BG_CAP` 越界、`layout:"auto"` 稀疏绑定（`u32::MAX` 哨兵 + 宿主跳过）、
  `WF_STAGE_GROUPS` 组前缀、`Material.kind@40` 打包、槽位 pixel-major 映射
  五处接线缺陷（详见报告 §29.3）；③`wavefront_path_golden.bin`（WPG1，
  15,580 B，三配置）与 `qa_wavefront_parity` / `qa_wavefront_r10` /
  `qa_wavefront_golden` 三套判据入库。
- **为何非 C15**：全部为新增能力/接线；既有可渲染工程输出逐字节不变。
- **是否有开关**：`--adaptive` 为显式 CLI 档位（默认档不变）；浏览器腿无开关。
- **验收**：`cargo test -p prism-gpu --test qa_wavefront_parity`（R-1/R-6 逐位）、
  `qa_wavefront_r10`（白炉 |L−1|≤1e-2 @1024 样本 + 无偏 ≤5e-3 @8192 样本 + GPU↔CPU 逐位）、
  `qa_wavefront_golden`（三配置）、`cargo test -p prism-postfx`（C2 回路 9 条）、
  `node crates/prism-gpu/web/ci_wasm_verify.mjs`（realgpu 21/21）。

### F-2026-xx-04：V2.5 能力扩容 W3（几何实例化，C1 两级 BVH）

- **扩容内容**：
  1. **共享几何可真实例化**（此前只有"展平"一条路）：`Scene.instances` 的 112 B 定桩记录
     由"无人读"变为**真渲染**。TLAS 复用 `aabb_bvh::AabbBvh`（SAH，12 桶），BLAS 复用
     `bvh::Bvh`（SAH + 严格 `<` tie-break）；**同一原型几何只建一次 BLAS**，构建成本与
     实例数解耦（10,000 实例实测 TLAS 构建 **4.1–4.3 ms**（3 次运行）、TLAS 0.179 MiB / 4095 节点，
     实例表恰 **112.0 B/条**）。剪枝有效性：10k 实例 512 射线，TLAS **3.142 ms** vs
     线性兜底 **202.2 ms**（**64.3×**）。
  2. **逆矩阵只在 CPU 算一次并存进记录**（3×4 仿射 + 3×4 逆，`from_affine` 余子式求逆），
     三个后端读同一份字节——**禁止各后端自行求逆**（§4.2 约束）。`det < 0`（镜像）在
     flags 里记 `INSTANCE_FLAG_MIRRORED`，法线走向按 `det(A)·A⁻ᵀn` 反号，非均匀缩放 +
     剪切下与展平几何**逐位对拍**（含反向自证：直接乘 `A` 的 dot < 0.99 会被判错）。
  3. **`t` 参数不换算**：局部射线取 `o_l = M⁻¹o`、`d_l = A⁻¹d`（**不归一化**），
     故 `o_l + t·d_l ≡ M⁻¹(o + t·d)` 对**同一个 `t`** 成立 ⇒ 世界空间 `T_MIN` /
     `t_max` / `SURFACE_EPS` 直接用于局部求交，零转换、零精度损失（§4 里的"最难点 2"
     被**消掉**而不是"算对"，见报告 §16.2/§16.4-A）。
  4. **glTF 改为保实例**：`parse_gltf_scene` 返回「共享几何表 + 实例记录」，节点变换不再
     烘进顶点；3 节点引用同一 mesh ⇒ 几何 1 份 + 实例 3 条（`saved_copies = 2`）。
     顶点数据本身**不再复制**，`parse_gltf` 变兼容壳（`GltfScene::flatten`，
     与 W3 之前**逐位等价**：合成变换末行恒 `(0,0,0,1)`，`w` 精确为 1）。
  5. **USD 停止展平**：新增 `build_scene_instanced[_report]`，Xform 链留在实例记录里、
     原型顶点保持局部坐标，共享原型按 **FNV-1a 内容指纹**去重。
  6. **运行时校验**：`Scene::validate_layout()` 第 5 步新增 `validate_instances`
     ——`geom_id`/`mat_id` 越界、矩阵非有限、identity 标志与数据不符、`pad != 0`、
     `det(A)·det(A⁻¹) ≉ 1` 全部拒绝。
- **未点亮 / 待验分述**：
  - **GPU 三后端不渲染实例**（`scene_abi` V3 `nodes` 槽仍"只有表、无编码器"）。
    这不是"静默少画"：`SceneRenderRequest` **没有实例字段**，实例从 `prism-render` /
    `prism-gpu` 侧**结构上不可达**，故 CPU/GPU 不会分歧；将来把实例接进 DSL 时，
    须同时接三个后端 + `scene_abi` 编码器，缺一不可。
  - **USD `instanceable` 语义未解析**：本次只做到"不再展平 + 内容指纹去重"，
    `instanceable` 元数据本身仍未读（去重是**保守**的：几何相同即共享，与 `instanceable`
    无关，故不会误共享不同几何）。
  - **glTF 材质映射仍是近似**：`gltf_default_material` 只把 `baseColorFactor` 映成
    Lambert，PBR 全量映射未做（与 W1-B 纹理同批）。
- **开关**：无。老入口输出逐位不变（CLI 默认档 `a81a6b4f…aad5`、`--fast` `ef012e5b…ade6`
  与 §13.4 回归锚完全相同；`parse_gltf` 走 `flatten` 兼容壳）。**唯一偏离规格的是非掠射
  判据的适用条件**：`t` 相对误差 ≤ 1e-6 在 `|cos| ≥ 0.4` 下 **0 条超阈**，掠射
  （`|cos| < 0.37`）下 f32 条件数本身不允许，已按实测登记为偏离（报告 §16.4-B）。

### F-2026-xx-03：V2.2 能力扩容（曲线图元 GPU 覆盖 / 材质库十种 + 各向异性 / SVGF 第二阶段 / 光谱色散切片）

- **扩容内容**：
  1. **曲线图元可渲染**（#A-2）：圆柱 / 圆锥 / 圆环 / 旋转曲面四类解析图元此前**不可渲染**
     （场景只有球 + 灯各 8 槽）。新增 `prism-geometry/curve_tess.rs` 弦高受控三角化
     （`hausdorff_bound = r_max·(1−cos(π/seg))`）+ 扁平 BVH，并**同时接通两条渲染路径**：
     CPU 参考（`CpuRefBackend` 三角化 → `Bvh` → `ray_intersect`，含 tie-break）与
     Vulkan megakernel（`spirv_rt` 参数块 22→24 字、绑定 4→6、`OpTypeArray` 存储数组、
     双栈 BVH 遍历、Möller–Trumbore）。**lavapipe 实机硬判据**：四图元全部
     `max_channel_delta=0 && differing_pixels=0`（`software_icd=true`），双跑逐位一致；
     WebGPU 侧 `scene_frame.wgsl` 同语义（栈式遍历），两目标锚定 + 槽位 ABI 钉偏移。
     场景 ABI 由 V1（4 绑定）升 **V2（6 绑定 `params/spheres/lights/triangles/nodes/out`，
     输出恒为末位）**，`scene_abi` 为唯一真源、`web/device.rs` 全量从它派生。
  2. **材质库扩至十余种 + 各向异性**（#B-3）：`MaterialKind` 由 4 变体扩至 ≥10 预设
     （可渲染口径，测试钉住计数）；principled 新增 **GGX 各向异性**（能量补偿闭式近似，
     无迭代）——`α_t==α_b` 时与既有各向同性 GGX **逐位一致**（退化判据）；DSL `material`
     块新增各向异性字段（含行列号诊断），**老 3 字段场景输出逐位不变**；能量守恒
     furnace 测试常驻（白炉半球积分反射率 ∈ `[1−ε,1+ε]`，含反向自证）。
  3. **SVGF 空域降噪第二阶段**（#B-2）：新增 G-buffer 引导**时域重投影**（motion vector
     闭式给出，不用光流、不引第三方）+ **方差引导 à-trous**（保持有理核、零超越函数）；
     静态相机 + 静态场景下重投影累积与「帧 0 起重放」**逐字节一致**。动态场景实测
     **PSNR = 43.4 dB**（判据 ≥40）、同等画质 **sample 数下降 91.7%**（判据 ≥50%）。
  4. **光谱渲染切片**（#B-1）：采样维度新增 λ 轴，自研 **CMF 分段有理式**拟合
     （零 crate、零超越函数），RGB↔光谱升采样；退化一致性——单波长 + 恒定谱反射率下
     与 RGB 路径**逐字节相等**；色散棱镜 golden 帧红/蓝质心分离 **2.504 px**
     （阈值 1.0，`cauchy_b=0` 对照组 0.917 证明判据对色散真响应）。
  5. **计数与帧池**（#D-2，非渲染能力，随版交付）：自研 `CountingAllocator`（零 crate）
     + `Framebuffer` 帧池 + `as_ptr()` 零拷贝身份判据；同场景两跑计数逐项相等、池中性逐字节。
- **未点亮 / 待验分述（G4/G7）**：
  - **WebGPU 渲染路径**（#A-1）：`scene_frame.wgsl` + `web/device.rs` 真实路径 + 4 绑定
    + `abi.rs` 场景项均已交付（**代码就绪**）；**真机 Chrome/Edge 未验**（沙箱仅 SwiftShader
    软件实现，语义已判对，剩余风险是驱动浮点差异）。README 边界行**不因本项改写**。
  - **Metal**：本仓**不提供独立 Metal 后端（不点亮）**，不引入任何第三方翻译层依赖；
    `metal/`（含 1,484 行 MSL）退为**历史资产**，`is_available()` 恒 false。
  - **硬件编码**（#C-2）：VAAPI 侧推进到「**绑定就绪**」（`extern "C"` 函数表 + `dlopen`
    符号解析 + 结构化 `Unsupported`），**不声称可编码**；真机出码待验。
- **开关**：无。以上均为行为扩容；唯一的行为变化项（折射法线口径）见 `R-2026-xx-06`。

### F-2026-xx-02：V2.1 能力扩容（Vulkan 光追实帧 / USD R1·R3 判据 / 签名闭包 S3 / 异质体积 / Metal 门控）

- **扩容内容**：
  1. **Vulkan 光追管线真实化**（#11）：`render_scene` 由「恒返回空帧、调用方
     回退 CPU」升级为 **lavapipe 实机真实出图**（自研发射器 `vulkan/spirv_rt.rs`
     光追 megakernel + `vulkan/scene.rs` dispatch 编排）。实测 64×48 / 113×47
     与 `CpuRefBackend` **max_channel_delta=0、differing_pixels=0（逐字节
     一致）**，GPU 双跑逐位一致；16×16 空场景全帧背景色 256/256 像素一致。
     `is_available()` 升级为 **compute ∧ render 双自检**（可用却出空帧的组合
     被测试排除）。Metal 同构移植（设备无关部分 + MSL kernel 源）+ 编译门控，
     `is_available()` 保持 false（**未实机验证、不点亮**，诚实留名）；WebGPU
     可行性结论见 `docs/v2.1-WebGPU可行性评估.md`。
  2. **USD 资产 R1/R3 判据**（#9）：新增「同构保持」（表示签名等价）与
     「保真度界」（Hausdorff `≤ √3/2·h`）的形式化与测试；R2 字节往返不动点
     不回归。
  3. **签名闭包 S3 收尾**（#8）：`cache_key_with_mode` —— 迭代节点键含
     `(ε,n)`（经 ChannelDef 0x5333 编码）、流节点键含 frame、S4 逐出中性；
     `mode=None` 时逐字节退化为旧 `cache_key`（零回归）。
  4. **异质体积**（#10）：`HeterogeneousMedium`（Woodcock delta tracking，
     σt 空间变化）+ 格心采样口径修正；σt 恒定场与均匀实现**逐位一致**
     （退化一致性）。**已接入渲染链路**（QA-F2 结清）：场景介质盒统一为
     `enum Medium { Homogeneous, Heterogeneous }`，integrator 与 wavefront
     双路径均可消费；均匀路径的采样流消耗与像素输出**逐字节不变**（单均匀
     分支），异质路径按需多抽一个接受均匀。无偏性口径如实分述：`shadow_transmittance_tracked`
     （调用方喂独立均匀）300k 样本内对 Beer 定律偏差 < 1%；生产入口
     `shadow_transmittance`（2-uniform 签名）的空碰撞派生链已由乘法链改为
     splitmix64 去相关（QA-F1：旧链实测 +1.08% 正偏），修复后由测试
     `shadow_transmittance_production_entry_unbiased_vs_beer` 钉在 2% 内。
- **为何非 C15**：全部为新增能力；既有可渲染工程输出逐字节不变（GPU 出帧与
  CPU 参考 0 差异；新增签名键只在带模式调用时生效）。R3 修复本身另有
  C15 条目（R-2026-xx-04）。
- **是否有开关**：无（新增能力，不改变既有路径行为）。
- **验收**：`prism-gpu/tests/qa_render_scene.rs`、`prism-io/tests/qa_r1_representation_signature.rs`、
  `prism-core/tests/qa_signature_s3_closure.rs`、`prism-render/tests/qa_incremental_s3.rs`、
  `prism-rt/tests/qa_heterogeneous_volume.rs` +
  `integrator::tests::heterogeneous_medium_render_path_is_reachable_and_matches_beer`
  （渲染路径可达性 + 整画面 Beer + 两遍逐位确定性）。

### F-2026-xx-01：movie 成片路径新增 sim（S3 约束梯度流，cloth）节点渲染支持

- **扩容内容**：`prism-render` movie 路径（`render_movie` /
  `render_movie_frame`，`crates/prism-render/src/movie.rs`）新增
  `StructureId::ConstraintFlow`（sim，当前种类仅 cloth）消费：自包含
  XPBD 求解器（半隐式欧拉 + 距离约束投影，数值纪律逐位对齐
  `prism-physics` 的 xpbd.rs）+ 帧循环驱动（`SimTimeline` 跨帧快照库）。
  此前含 sim 节点的场景在 `classify_node` 按 `R5010`「sim 暂不支持」
  显式拒绝；现在该类场景可渲染（粒子叠加为暖橙圆点）。`R5010` 对
  particles / compose / 未知结构的拒绝不变——仅 sim 一族解禁。
- **为何非 C15**：C15 只登记「bug 修复导致的输出字节变化」；本条是
  能力扩容——此前直接报错的场景现在成功出图，既有可渲染工程的输出
  逐字节不变（无字节变化可登记，故不进《渲染行为变化清单》。
- **确定性口径**：模式元数据（substeps / iterations / residual_eps）
  经通道定义（tag = 0x5333）哈希进帧签名——改迭代参数 ⇒ 签名变 ⇒
  快照/缓存不误复用；同参数两遍渲染逐字节一致；快照装配跳帧 ≡
  顺序推进；冷 seek（无前驱快照）按 `P3012` 结构化报错，绝不静默
  fallback 到初态。
- **是否有开关**：无开关（新增能力，不改变既有路径行为）。

验收：`crates/prism-render/tests/movie_sim_timeline.rs`（帧循环出图 /
确定性 / seek 等价 / 冷 seek P3012 / 模式元数据进签名与缓存键）+
`crates/prism-render/src/movie.rs` 单测。

---

## 语言前端与门禁（非 C15：新增 v3 前端，既有渲染字节不变）

### V3.0-W1-01：定桩 + 语言内核 + 确定性求值（W1 收口）

- **扩容内容**：
  - 定桩先行：`docs/v3.0-语言规格.md` 在解析器动工前冻结求值顺序 F-1~F-5
    与种子方案 F-6~F-10；`docs/v3.0-触达规格表.md` 39 行逐行填实
    `api_path:line` 与引擎侧公开参数全集（门禁逐行回读源码复核）。
  - `prism-dsl` 新增 `// prism-dsl 2` 前端：独立词法面（缩进栈 / 三引号 /
    插值）、AST、LL(1) 语句 + Pratt 表达式、确定性受限子集执行器
    （`src/eval.rs`）、降级装配（`codegen::v3`）。`emit` 把计算结果展开为
    **v2 块文本**后交既有 `compile_v2`，语言层不重建节点图、不重实现数学。
  - `prism-app render` 的编译入口改走 `compile_auto` 分派：无头 / 版本 1
    与旧 `compile_v2` 完全等价；版本 2 走 v3 并降级成同一份 `CompileV2Output`。
  - `prism-core::eval::hash_mix` 由测试模块提升为生产 API（算法逐字不变），
    新增 `hash_mix_bytes` / `hash_mix_unit_f32` / `hash_mix_unit_f64`；
    `prism-math::det_scalar` 补齐 15 个零 libm 标量函数（含固定 6 步牛顿 `det_sqrt`）。
- **为何非 C15**：旧 `.prism`（v1 / 版本 1）路径一行未改，三个黄金文件 md5
  对账一致、既有 golden 全绿；`compile_auto` 对旧文件与 `compile_v2` 逐字节等价。
  既有可渲染工程输出帧**无任何字节变化**，故不进《渲染行为变化清单》。
- **确定性口径**：Q32.32 定点数值域、零 libm、零线程；`rand(tag)` 唯一走
  `hash_mix(seed, tag, eval_ordinal)`；seed 与求值轨迹 `trace` 一起进
  `artifact_hash`；求值严格左→右深度优先、容器按声明序、`for` 上界编译期
  可判定（上限 4096 步）；换序 / 换种子 / 换语句序三类反例均可检出。
- **是否有开关**：有——仅首行 `// prism-dsl 2` 进入 v3 前端；版本 1 / 无头
  文档永久走旧路径，零损失契约不变。
- **验收**：`cargo build --workspace` 通过；受影响 5 crate 共 **906 passed /
  0 failed / 0 ignored**；`coverage_gate` 9 条（含 6 类错误注入全判红）、
  `v3_language` 12 条（A1 八类正例 + 逐条负控共 25 例、非法语义静态拒绝、
  版本头 / 求值 / 降级三段式负控、两跑逐字节、换序、md5 对账）、
  `v3_homologous` 4 条（`render_movie` 同源对拍逐字节）全绿。
  覆盖率实录：结构 39/39 = 100.00%，DSL 字段映射行 15/39 = 38.46%，
  参数级全名 0/225 = 0.00% / 末段 18/225 = 8.00%（W1 如实输出，不设阈值）。
