# calculus-film-v2 T05 剩余工作 brief（B–E）

> 主理人齐活林编制，2026-10-14。状态：A 步（scene3d.rs 含玻璃切平面 Create 入场）已完成并经主理人核验。本文件是剩余 B–E 的唯一任务书；执行中如需分镜/数学细节，按 §6 指针读原文档指定行段，**不要通读大文件**。

## 0. 执行协议（最高优先级）

**一回合只干一件事**：每个回合只做一步（读一处 API / 写一个文件 / 跑一次验证 / 启动一次后台渲染），落盘并自验通过后才进入下一步。严禁「读完全部文档再一次性写多个大文件」。任一步 build/test 失败：就地修复重跑，禁止带病前进。

步骤：B（scenes_math.rs）→ C（主入口扩建）→ D（全片渲染，后台）→ E（验收六项）。最终回复汇总全部证据链。

## 1. 已完成事实（直接依赖，勿重做）

- `crates/prism-author/examples/calculus_film_v2/scene3d.rs`：`build_s7_scene(with_glass: bool)`、`camera_at(t)`、`prerender_frames(n) -> SequenceFrames{width,height,frames}`、`VERIFY_PARALLEL`、预算探测 + §8-R2 降级、玻璃平面 Create 入场（源帧 60=4.0s 前无平面、60..68 混合 k=(f−60+1)/8、≥68 全平面）、淡黑尾（f≥120 渐黑至 132 全黑）、后处理链（Fbuf16F→AtrousSpatial{3,0.1}→Reinhard→sRGB OETF）。实测 2.27s/帧 ⇒ 155 帧 ≈5.9 分钟（裸场景）；含透射代表帧更慢，`plan_budget` 已按慢者保守外推。
- `crates/prism-author/examples/calculus_film_v2/{common,symbols,axes,bgm}.rs`：T04 资产层（25 条测试）。
- `crates/prism-author/examples/calculus_film_v2.rs`：占位根（`#[path]` 挂接六模块 + 冒烟 main；scene3d 带占位 `#[allow(dead_code)]`，C 步清除）。
- 引擎原语（T02/T03）全部可用：`Track::DynPath/Counter/Sequence`、E3 `center_x`、prism-rt `Paraboloid/Plane`。
- Cargo.toml dev-deps 已含 prism-audio/postfx/render/rt/core/math（路径依赖，零第三方）。
- 当前测试：`cargo test -p prism-author --example calculus_film_v2` 33 passed / 0 failed。
- 已知实现事实（scene3d 实测）：`EvalCtx::new(seed, node_id, frame)` 实参序为 (SEED_3D, NODE_3D, frame_idx)；相机 height 为世界绝对 z。

## 2. B 步：scenes_math.rs（S3–S6 四场景）

新建 `crates/prism-author/examples/calculus_film_v2/scenes_math.rs`，每个场景一个构造函数返回 `ClipSpec`（建议 `fn scene_s3() -> ClipSpec` … `fn scene_s6() -> ClipSpec`）。时长一律 `scene_duration(帧数)`；帧数：S3=496 / S4=496 / S5=496 / S6=436。

逐拍分镜读 `docs/calculus-film-v2-prd.md` 行 128–221（S3–S6 四节）；数学正确性读行 306–324（§6 清单）。核心铁律**读数恒等**：每个 Counter 的 `(from,to,dur_sec,easing)` 与对应 DynPath 几何参数派生自**同一纯函数**（`secant_h(t)` / `riemann_sum(n)` / `tangent_a(t)` / `chain_x(t)`），禁两处写常数。关键数学（PRD §6 已验算，直接采用）：

- S3（f=x², x₀=1, 深蓝黑渐变）：割线斜率 m(h)=2+h ⇒ 3.0000/2.5000/2.2500/2.1250→2.0000；第二端点 (2,4)/(1.5,2.25)/(1.25,1.5625)/(1.125,1.265625)；ε=1.00→δ=√2−1≈0.41、ε=0.25→δ=√1.25−1≈0.12；定义式 ∀ε>0,∃δ>0 用 symbols 组装；HUD「m = 」Counter 与割线几何同函数。
- S4（f=x²/2, f′=x, 深青黑渐变）：切线 y=ax−a²/2；a: −2→−0.5→2.0→定格 1（CubicInOut/QuadInOut 按分镜）；动点 A 在 f 上、B 在 f′ 上、A-B 竖直引导虚线 alpha 0.3；HUD m 与切线斜率恒等；导函数曲线 f′(x)=x 亮紫 4px。
- S5（f=x² on [0,2], 深紫黑渐变）：右端黎曼和 Sₙ=4(n+1)(2n+1)/(3n²) ⇒ n=2: 5.0000 / 4: 3.7500 / 8: 3.1875 / 32: 2.7930 → 滚至 2.6667；矩形用 `riemann_rects(n, f, vp)`（积分域=视口 x 域 [0,2]）；矩形淡出+面积填充淡入「融进曲线」；积分式 ∫₀² x² dx = 8/3 用 symbols 组装；Σ→∫ 演化。
- S6（F=x³/3, Aurora）：双链 A=f（青）、B=F（亮紫）；x: 0→2 联动（QuadInOut）；HUD 左 f(x) 与右 F′(x) 恒等（x=1.5 ⇒ 2.2500）；F′ 重绘与 f 完全重合；FTC 公式 d/dx ∫₀ˣ f(t)dt = f(x) 用 symbols 组装。

API：T04 资产（`axes_tracks(vp)` / `sample_curve` / `riemann_rects` / `assemble_formula(parts, center_x, baseline_y)` / `symbol_path(kind,size)` / `SymbolKind` / `FormulaPart`——symbols.rs 头部 1–90 行）；Viewport（common.rs:189–300）；Track 字段速查见 §5。颜色一律 `hex_to_display`；坐标一律 `Viewport::to_px`；落位用 E3 `center_x`。

## 3. C 步：扩建 calculus_film_v2.rs 主入口

保持 `#[path]` 挂接六模块；清除 scene3d 占位 `#[allow(dead_code)]`；main 流程：
`build_s7_scene`/`prerender_frames(155)`（scene3d 已有，直接调）→ 组装 7×ClipSpec → `render_short_concat_faded_frames(&specs, 1.0, 1.5)` → `synth_bgm(90.0,44100)` → 断言 → `write_avi_mjpeg_audio(&frames,960,540,30,82,&bgm)` → 写盘 `/workspace/prism_calculus_90s.avi`。

- **S1（248 帧，DarkGlow）**：小标签「PRISM ENGINE 出品」Static 20px y=120；lim/dy/dx/∫ 三符号矢量描画（symbols，金/青/白，draw 1.0–1.2s）；5.0–6.0 三符号呼吸+上浮 8px（DynPath 闭包：scale 1.00→1.05→1.00 + y 偏移，CubicInOut）；主标题「微积分」kinetic Bold 96px stagger 0.12 y=380；副题「从割线，到光追」打字机 12cps y=460 带光标；7.5–8.0 符号阵淡出。
- **S2（308 帧，Aurora）**：视口 x∈[−4,4]、y∈[−2.4,2.4]、85px/unit、绘图区 140–820×66–474；axes_tracks 网格（13 竖 9 横 alpha 0.10 stagger 0.08）/轴/箭头/刻度/标签「x」「y」；y=sin x 49 点青 3.5px draw 2.0s；**相机缓推：视口 scale 1.00→1.08 关于绘图区中心（CubicInOut，4.5–6.5s）——全部几何经缩放视口 to_px，DynPath 闭包捕获缩放视口；禁 postfx::Warp**；公式标签「y = sin x」assemble_formula；文案「每一条光滑曲线，都是变化的故事」打字机 y=500；曲线 9.0s 淡出、网格 9.5s 淡出。
- **S7（310 帧）**：`Track::Sequence{frames: scene3d::prerender_frames(155).frames, width: 960, height: 540, src_fps: 15, at: 0.0, in_sec: 0.5, out_sec: 0.0}`；叠加文案轨：1.0–4.0 打字机「同一条曲线，旋转成一个面」y=500；4.0–5.0 公式「z = 1.6x + 1.6y − 1.28」assemble_formula + 切点 P 金点脉冲（ShapeTrack Ellipse）；5.0–6.5 打字机「切平面——切线在两个方向上的推广」；6.5–8.0 打字机「从割线到光追，同一件事：变化」；8.8–9.3 落版两行 Static（「Made with Prism Engine」Bold 44px /「Zero dependencies. Video as code.」20px，center_x=None 居中）；淡黑由预渲染渐黑帧承担。
- 背景色：S3 深蓝黑 / S4 深青黑 / S5 深紫黑 / S6 Aurora——用 common.rs 既有调色板常量（无则 Background::Gradient + hex_to_linear 贴近 v1 先例）。
- **main 装配断言（缺一不可）**：帧表 [248,308,496,496,496,436,310] 且 Σ2790−6×15=2700；总帧 2700；BGM `frames()==3_969_000`、`peak()<=1.0`、首/尾 0.05s 幅度 <0.01（PcmBuffer API：prism-audio/src/buffer.rs）；Sequence 轨 155 帧且 960×540；S7 ClipSpec 时长 == `scene_duration(310)`。

Track 字段速查（prism-author/src/spec.rs，已核对）：`ClipSpec{width,height,fps,duration,background,jpeg_quality,tracks}`；`StaticText{content,font,size,color,at,in_sec,letter_spacing,y,out_sec,center_x}`；`KineticText{content,font,size,color,at,stagger,per_char,scale_from,rise_px,easing,y,in_sec,out_sec,center_x}`；`TypewriterText{content,font,size,color,at,y,cps,cursor,cursor_blink_hz,in_sec,out_sec,letter_spacing,center_x}`；`PathTrack{points,color,width,fill,at,in_sec,out_sec,draw_sec,closed}`；`DynPathTrack{geom:Arc<dyn Fn(f32)->Vec<[f32;2]>+Send+Sync>,color,width,fill,at,in_sec,out_sec,draw_sec,closed}`（参数为场景绝对时间 t；fill=true 忽略 width/draw_sec/closed；color[3] 固定 alpha，渲染层乘 env）；`CounterText{prefix,from,to,at,dur_sec,easing,decimals,font,size,color,center_x,y,letter_spacing,in_sec,out_sec}`；`SequenceTrack{frames,width,height,src_fps,at,in_sec,out_sec}`；`AccentLine{center_x,y,w,h,color,at,in_sec,out_sec}`；`ShapeTrack{kind:ShapeKind::{Rect|RoundRect{radius}|Ellipse},x,y,w,h,color,at,in_sec,out_sec}`；`FontKind::{Bold,Regular}`；`EasingKind` 读 crates/prism-author/src/easing.rs（小文件全文）。

## 4. D 步：全片渲染（后台）

`cargo run --release --example calculus_film_v2` 用**后台任务**启动并轮询日志（S7 ≈6–9 分钟 + 2D 六场景数分钟 + mux；全程预计 10–20 分钟，**禁止前台干等**——单次前台超过约 15 分钟会被上游超时掐断）。渲染日志存 /tmp/t05/render.log。

## 5. E 步：验收六项（逐条给证据）

1. `cargo run --release --example calculus_film_v2` 退出码 0，落盘 `/workspace/prism_calculus_90s.avi`；
2. main 装配断言全过（帧表/2700/BGM 三项/Sequence 155 帧/S7 时长）；
3. AVI 读回自校验：ffprobe 确认 MJPG+pcm_s16le 双流、30fps、960×540、1470 音频样本/帧、体积 ≈145MB（输出存 /tmp/t05/ffprobe.txt）；
4. S7 预渲染 ≤30 分钟（plan_budget 外推记录在案）；
5. 确定性抽查：同参数完整重跑一次（第二次也走后台），抽 5 帧 MD5 一致（记录两次 MD5；抽帧方式自定，如 ffmpeg 抽帧或 prism-mux 读回）；
6. PRD §6 数学清单逐条核对通过（最终回复逐条列出）。

## 6. 文档指针（只读指定行段）

- `docs/calculus-film-v2-prd.md`：S1/S2 分镜 89–124；S3–S6 分镜 128–221；S7 分镜 224–244；§6 数学清单 306–324。
- `docs/calculus-film-v2-architecture.md`：§2.6 338–348；§5.1 617–660；§6.5 验收 775–791；§7 共享约定 810–851。
- `crates/prism-author/examples/calculus_film_v2/symbols.rs` 头部 1–90 行（FormulaPart/SymbolKind）；`common.rs` 189–300（Viewport）；`axes.rs` 56/156/201（三个入口签名）。
- `crates/prism-author/examples/calculus_film.rs`（v1 成片：ClipSpec 组装/转场/BGM/mux 范式）。

## 7. 铁律

零第三方依赖；纯函数确定性；禁 TODO/占位符/`#[ignore]`；`cargo build -p prism-author --all-targets` 0 warning；`cargo test -p prism-author --example calculus_film_v2` 全绿（33 条须保持）；每步自验通过才下一步。
