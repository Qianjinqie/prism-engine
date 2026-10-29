# DSL v2 节点盘点表 × render_movie 覆盖状态（V1.5-D）

> 维护人：寇豆码（工程师） · 波次：W2 执行闭合波·D 单顶 · 日期：2026-10-18
>
> **V2.0-W4 更新**（RQ-3 曲线图元完善）：S12 实体曲线四族
> （cylinder / cone / torus / revolution）进入已支持子集——见 §1 第 2 行
> 与 §2 帧求值路径；GPU 曲线槽位未建，走 rt CPU 波前路径（诚实口径，
> 不假装 GPU 已支持）。
>
> 权威来源：节点全集取 `crates/prism-dsl/src/codegen.rs` 的
> `codegen_v2`（v2 编译产物 `CompileV2Output`）与各 `build_*` 函数；
> render 侧覆盖状态取 `crates/prism-render/src/movie.rs`（`render_movie`
> 直达路径）的 `build_models` / `render_scene_frame` / 时间线求值。
>
> **纪律**：render_movie 碰到不支持的节点 = 明确报错（R5xxx 三段式：
> 哪个节点 / 为什么不支持 / 怎么改），**绝不静默跳过**。

## 1. 节点全集 × 覆盖矩阵

| # | DSL v2 写法 | 节点类型 | 结构段 | render_movie | 报错码 / 说明 |
|---|---|---|---|---|---|
| 1 | `stage3d { camera … }` | camera | S5 Reparameterization | **已支持** | orbit center / radius / fov；绑定通道可覆盖 |
| 2 | `stage3d { sdf { kind sphere … } }` | sphere | S12 EntityRepresentation | **已支持** | `kind ∉ {sphere, cylinder, cone, torus, revolution}`（box/…）→ R5013 |
| 2b | `stage3d { sdf { kind cylinder / cone / torus / revolution … } }` | 曲线图元族 | S12 EntityRepresentation | **已支持**（V2.0-W4） | 多曲线节点同场（按节点序入 rt 场景）；求值走 rt CPU 波前路径 + SAH-BVH；尺寸退化（≤0 / 轮廓点 <2）→ R5016；GPU 曲线槽位留待后续波次 |
| 3 | `stage3d { material … }` | material | S8 RadiativeTransfer | **已支持** | albedo / roughness / metalness |
| 4 | `stage3d { light … }` | light | —（校验后不建节点） | 不适用 | render 侧固定缺省暖色灯（与 v1 演示一致） |
| 5 | `layer text { … }` | text | S12 EntityRepresentation | **已支持** | prism-text 排版 + 光栅化 + u8 src-over；`rotation ≠ 0` → R5014 |
| 6 | `layer particles { … }` | particles | S4 DeterministicSampling | **不支持** | R5010：缺粒子域求值器（样本→像素映射管线） |
| 7 | `sim { … }` | sim | S3 ConstraintFlow | **不支持** | R5010：缺约束梯度流求解器（XPBD 步进） |
| 8 | `compose { … }` | compose | S2 Convolution（Merge 汇点） | **不支持** | R5010：其输入只能是 layer/sim（均不在已支持子集）；分屏/形变请用组合子 |
| 9 | `audio { … }` / `post { … }` / `note` | 元数据 | —（不建节点） | 不适用 | 仅登记进产物（`SceneV2` 摘要），render 不消费 |
| 10 | layer 前置 `morph from … to …` | 图层几何变形 | 矢量图层能力 | **本轮不覆盖** | MorphSpec 几何变形属矢量图层能力（W1-C 已下沉 core 类型，待矢量波次接入求值） |
| 11 | `transition { … }` | 转场 | TransitionV2（非节点） | **已支持** | crossfade / fade / cut / black / white / wipe / dissolve / push |
| 12 | 字段级关键帧 / `anim` / `expr` | 命名通道 | NamedChannel（非节点） | **已支持** | `sample_binding` 逐绑定覆盖节点参数 |

已支持集合（R5010 建议段的原话）：**stage3d camera / material、sdf sphere 与曲线
四族（cylinder / cone / torus / revolution）、layer text**。
即 DSL v2 的 8 类可建节点写法中 4 类已支持（camera / material / sdf / text，
其中 sdf 覆盖 5 种 kind），
3 类明确报错（particles / sim / compose），1 类不建节点（light）；外加 2 类非节点
能力（transition / 通道）已支持、1 项矢量能力（layer morph）不在本轮覆盖面。

## 2. render_movie 时间线语义（数据驱动，非场景顺序）

`render_movie` 对 `CompileV2Output.scenes` 的求值规则（按帧取活跃场景集）：

| 时间线形态 | 语义 | 说明 |
|---|---|---|
| 单场景活跃 | 直接求值该场景 | 最常见形态 |
| 两场景部分重叠 + 匹配 transition | 过渡混合 | crossfade/fade=全局权重混合；cut=中点硬切；black/white=向目标色淡出；wipe=带状擦除；dissolve=哈希溶解；push=推进（越界钳边） |
| 两场景起止**完全相同** | 左右分屏 | split_screen 组合子展开形态。**注意**：codegen_v2 产物永不产生此形态（P1011 禁止 transition 重叠 ≥ 场景时长 ⇒ 后场 start 恒 > 前场 start），该形态只能经 `prism-author` 组合子 API 构造 |
| 空档（timeline 长于内容） | 黑帧 | 确定性（固定背景色） |
| 其余多场景同时活跃 | 明确报错 | R5020：三场景同时活跃、或两场景同时活跃但既非同程也无匹配过渡 |

帧数 = `ceil(fps × duration)`；每帧 = 底片场景经 `render_frame`（复用 v1 求值器；
场景含曲线图元节点时走 `render_curve_frame`：rt CPU 波前路径 + SAH-BVH，
固定 spp/弹射深度常量，tone-map 与 GPU 路径同口径）
+ 按节点序 `overlay_text` 文本合成。确定性：两跑 MD5 一致（有单测凭证，
含曲线四族同场端到端用例）。

## 3. 组合子库 × 展开形态（V1.5-D 新增）

展开函数落 `prism-dsl/src/combinators.rs`（注册面 `COMBINATOR_TABLE`：
name → 展开 fn，与 codegen 同 crate）；`prism-author/src/combinators.rs`
为类型化建造者薄封装。**DSL v2 语法本轮不加新指令。**

| 组合子 | 展开产物节点图形状 | 涉及的节点类 | DSL v2 能否等价表达 |
|---|---|---|---|
| `title_card` | 单 S12 文本节点 + `title_card/opacity` 淡入淡出轨道 | text | **能**：`layer text` + opacity 关键帧块（示例 `examples/w2d_showcase.prism` 形态一） |
| `lower_third` | 两 S12 文本节点（headline + subhead）共用一条 opacity 轨道 | text | **能**：两个 `layer text` 各绑 opacity 关键帧 |
| `split_screen` | 两子片前缀吸收 + 两场景同程（起止相同） | camera/material/sphere | **不能**：时间线规则禁止同程（见 §2）；只能经 Rust API 构造 |
| `ken_burns` | 剔除底片相机，新相机 radius/center_x/y/z 各绑起止轨道 | camera | **不能**：`camera.orbit.radius` 只接受裸数字（关键帧块被 P2100 拒）；这正是组合子作为 Rust API 入口的存在理由 |
| `morph_between` | 后片平移 `fd − overlap` + `TransitionV2` | camera/material/sphere + transition | **能**：两场景 + 一条 overlap < 两侧时长的 `transition`（示例形态二） |

## 4. 报错码速查（render_movie，R5xxx 段，三段式宪章 C18/C19）

| 码 | 触发条件 |
|---|---|
| R5001 | 输出规格非法（宽高含 0 / fps 0 / 时长非正） |
| R5002 | 场景引用悬空节点 id（node_ids 指向不存在的节点） |
| R5003 | 节点绑定的通道在产物 channels 中缺失 |
| R5010 | 不支持的节点类型（particles / sim / compose / 未知结构） |
| R5011 | 同一场景多重相机 |
| R5012 | 同一场景多重材质 |
| R5013 | sdf kind ∉ {sphere, cylinder, cone, torus, revolution}，或多球体场景未拆分 |
| R5014 | 文本节点 rotation ≠ 0，或 S12 节点无法归类 |
| R5015 | 文本 size ≤ 0 |
| R5016 | 曲线图元尺寸退化（radius / half_height / major / minor ≤ 0，或 revolution 轮廓点 < 2）（V2.0-W4 新增） |
| R5020 | 时间线出现无法解释的多场景同时活跃 |

组合子参数错误为 C5xxx 段（`CombinatorError`，同为三段式），未登记组合子
名为 C5000。

## 5. CLI 出口（V1.5-D）

```text
prism-app render <file.prism> [--out <FILE>] [--codec <h264|mjpeg>]
```

读文件 → `compile_v2`（语法错误带**行号** + 出错行原文回显，列位置可视化；
`DslError` 仅携带行号，列号以原文回显代替）→ `render_movie`（R5xxx 三段式）
→ 写容器：请求 H.264 → 探测 ffmpeg → 管道编码；**任一环节失败不静默**，
明确告知后降级内置 MP4(MJPEG)（`prism-mux::write_mp4_jpeg_color`，零外部
依赖、永远可用）。
