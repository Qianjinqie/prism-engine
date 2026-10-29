# V2.0-M3 PE-1 分布式渲染：帧级任务分发 + 本地多进程真跑 + 确定性合并

分支：`v2-m3-perf-eco`（完工不自行合并，等主理人）。

## 1. 范围与口径

| 项 | 口径 |
|---|---|
| 分发单位 | **帧级**：任务 = 帧号区间 + 配置哈希（线格式不载源码） |
| worker 形态 | `farm-worker` 子进程，本机 N 进程**真跑**（非线程替身） |
| 载荷 | **真实渲染帧像素**（RGBA 字节，`w·h·4/帧`），不是 32 字节替身 |
| 合并判据 | 按帧号归位 + 覆盖校验 + MD5：**N worker 合并 MD5 == 单机串行 MD5** |
| 断点续算 | 每 worker 一个独立 `farm-worker-<i>.cache`，重启跳过已完成帧 |
| 加速比 | 1 / 2 / 4 worker 三档实数，轻 / 重两档负载 |

口径边界（诚实）：`farm/worker.rs` 的 32 字节 splitmix64 载荷是**成本替身**，
验调度 / 传输 / 续算的确定性；本模块（`farm/movie.rs`）把载荷换成**真实渲染
产物**——"并行 == 串行"只有在真实像素上成立，才配叫"分布式渲染"。

## 2. 落点

| 文件 | 行数 | 内容 |
|---|---|---|
| `crates/prism-render/src/farm/movie.rs` | 551（新文件） | `MovieRenderer`（编译一次逐帧渲）/ `MovieJob`（切分 / 参考渲染 / 分发 / 续算分发）/ `MOVIE_SRC_ENV` 源码通道 / `MovieError`（P5011 三段式） |
| `crates/prism-render/src/farm/worker.rs` | 807（+182） | `RenderMode{Synthetic,Movie}` / `run_task_with_mode` / `run_worker_stream_with_mode` / 影片模式同源校验（异源任务 P5007 显式拒绝）/ 影片模式续算测试 |
| `crates/prism-render/src/farm/scheduler.rs` | 1275（+71） | `dispatch_to_local_workers_movie{,_cached}` / `SUBPROCESS_DISPATCH_LOCK` 进程级分发互斥锁 / `dispatch_with_worker_cache(…, movie_source)` |
| `crates/prism-render/src/farm/mod.rs` | 267（+13） | 新公开面导出（movie 模块 + 三个新函数 + `RenderMode`） |
| `crates/prism-render/src/bin/farm-worker.rs` | 132（+7 文档） | 用法表 / 帮助补 `PRISM_FARM_MOVIE_SRC` 行 |

源码怎么到 worker：`FrameTask` 线格式只载"帧区间 + 配置哈希"（与替身路径
同一份协议），DSL 源码走环境变量 `PRISM_FARM_MOVIE_SRC`（主节点分发时由
**调度器在进程级分发互斥锁内**注入，spawn 时被子进程继承，用完即删；范式同
既有 `PRISM_FARM_COST`）。worker 启动时读一次、编译一次，渲染前校验任务
`config_hash` 与自己编译产物一致——不渲染"手头恰好有的"另一部影片（C20）。

配置寻址：`ConfigHash::of_bytes(source)`——同源码同哈希，断点续算缓存按
它分键，坏源码不会污染好作业。

## 3. 确定性证据（CPU 路径，测试钉死）

测试：`farm::movie::tests::parallel_movie_digest_matches_serial_on_real_pixels`
（4 worker 子进程真跑）：

- `report.digest() == manifest_digest(串行参考)`——合并 MD5 与单机逐字节一致；
- `report.payloads_in_frame_order() == reference`——逐字节级证据（不止 MD5）；
- 1 / 2 / 4 worker 三档全对拍，MD5 恒等；
- `node_ids().len() >= 2`——真起了多个进程；
- 载荷长度 == `w·h·4`——真实像素而非替身。

为什么这是**定理**而非巧合：切分是 `(start, end, workers)` 纯函数（余数前置）；
渲染是 `(源码, 帧号)` 纯函数（DSL 编译无全局态、`CpuRefBackend` 无 RNG）；
合并按**帧号**归位 + 覆盖校验 ⇒ worker 数 / 完成顺序 / 哪个进程算哪块都不进
结果。帧采样公式 `ceil(fps·duration)` 与时刻 `frame/fps` 与 `render_video`
逐位一致（测试 `frame_sampling_matches_render_video_exactly` 钉死）——否则
"并行第 k 帧"与"串行第 k 帧"渲染的不是同一时刻，MD5 一致也无意义。

## 4. 加速比实数（`throughput_speedup_measured_across_worker_counts`）

实测环境：32 核可见，但 **cgroup CPU 配额 400000/100000 = 4 CPU**
（`/sys/fs/cgroup/cpu.max`）——这是本机集群式并行的硬顶。基准 DSL：
球体 + 轨道相机 + 一盏光，真实 CPU 参考渲染，无空转注入。

| 负载 | 1 worker | 2 worker | 4 worker |
|---|---|---|---|
| 轻载 20 帧 @128x96 | 160.92ms (1.00x) | 106.24ms (**1.51x**) | 83.01ms (**1.94x**) |
| 重载 20 帧 @256x192 | 659.83ms (1.00x) | 428.59ms (**1.54x**) | 322.22ms (**2.05x**) |

MD5（不随 worker 数改变，已断言）：轻载 `4044db3a…`，重载 `38173a81…`。

对照目标 ≥0.7N：**2 worker 达标**（1.51–1.54x ≥ 1.4）；**4 worker 未达标**
（1.94–2.05x < 2.8，约为 0.49–0.51N）——如实报告，不修饰。

归因（同一测试内解出来的 Amdahl 模型，`T(n) = R/n + F`）：

- 轻载：R≈109ms（渲染）/ F≈52ms（固定）；重载：R≈462ms / F≈197ms（载荷 ×4
  后主节点侧的收包 / SHA-256 校验 / 内存流量随之上升，F 不是常数——协议与
  数据量成正比这部分同样进 F）。
- 天花板：重载 4 worker 的理想值 = R/4 + F ≈ 313ms（实测 322ms，几乎贴模
  型）⇒ 并路径本身没有额外损耗，短板是 F 与 CPU 配额：
  4 worker + 主节点共 5 个吃 CPU 的进程抢 4 个 CPU 配额，天然封顶。
- 趋势符合预期：负载越重（R/F 越大），4 worker 从 1.94x → 2.05x——农场收益
  来自"帧成本 >> 传输/启动成本"，这正是两档基准要展示的实证形态。

（吞吐数字只打印不做脆断言——数值随机器波动，把测试变成跑分没有意义；
硬门槛只有 MD5 不随 worker 数改变。）

## 5. 断点续算证据

测试 `movie_resume_from_disk_cache_skips_done_frames`：4 帧作业 4 切分，
第一轮只下发前 2 块（模拟中断，worker 0 缓存落盘 2 帧）→ 第二轮完整任务 +
同缓存目录 → 续算结果与一次算完逐字节一致，缓存收敛到全部 4 帧，且缓存里
每帧载荷 == 参考渲染。缓存键 = `sha256("PRISM-FARM-CACHE" ++ cfg_be ++
frame_be)`，值是**任意长度真实像素**（`FrameCache` 是 `prism_io::CacheManifest`
薄封装，断点续算覆盖真实渲染而非仅替身）。

## 6. 并发的正确性：进程级分发互斥锁

载荷模式由**进程级环境变量**决定（全局态）。若两条子进程分发并发，后 spawn
的 worker 会读到前者的源码（或相反），载荷模式串味。`scheduler.rs` 用
`SUBPROCESS_DISPATCH_LOCK` 把"源码注入 + spawn + 收发 + 收工"收敛为临界区：
任一刻至多一条分发现置环境变量。相应地，进程内 worker 入口
`run_worker_stream` **恒 Synthetic、不读环境**（读它就是竞态）；模式选择
只发生在真实 worker 进程的启动处（`run_stdio_worker → RenderMode::from_env`）。

## 7. 环境受限项（方法论降级声明，如实）

1. **无多机**：本机沙箱无多机环境，N 进程 = 本机多进程真跑。协议 / 编解码 /
   合并与跨机是同一份代码，仅介质从 socket 换管道（`TcpTransport` 已留接口、
   本轮未实装，见 `FarmError::TransportUnavailable` 的建议段）。加速比实测
   反映本机多进程视角，**不代表**跨机集群数字。
2. **CPU 配额 4 核**：32 核可见但 cgroup quota=4 CPU，4 worker 加速比的上限
   主要由此决定（见 §4 归因）。
3. **CPU 参考渲染路径**：`CpuRefBackend` 软件渲染，无 GPU / 硬编路径参与
   （EC-1 三后端在本机均 Unsupported，已入能力矩阵文档）。

## 8. 本轮修掉的 bug（含既有）

1. `movie.rs` `MovieJob::dispatch` / `dispatch_with_cache` 在**锁外**手置
   `MOVIE_SRC_ENV`——与并发的其它子进程分发（及进程内 worker）竞态串味；
   改走调度器锁内注入的 `dispatch_to_local_workers_movie{,_cached}`。
2. `movie.rs` `reference_digest` 调用 `manifest_digest` 但顶层缺 import
   （lib 编译失败，E0425）。
3. `worker.rs` 影片模式异源校验错误消息对 `ConfigHash` 用 `{}`（无 Display，
   E0277）——改 `{:#x}` 打印裸哈希值。
4. `worker.rs` 新测试字节串 `b"另一部影片的源码"` 含非 ASCII（字节串字面量
   不允许）——改 `b"foreign-movie-source"`。
5. `movie.rs` 续算测试对 `FrameCache::open(&cache_dir)` 传**目录**（应为
   worker 缓存文件，运行时 CacheIo "Is a directory"）。
6. `farm-worker` bin 用法文档缺 `PRISM_FARM_MOVIE_SRC` 行。

## 9. 测试清单（prism-render，129 lib + 34 集成，全绿；0 ignored）

`frame_sampling_matches_render_video_exactly` /
`render_pixels_is_pure_and_config_hash_identifies_source` /
`tasks_cover_all_frames_without_gaps` /
`parallel_movie_digest_matches_serial_on_real_pixels`（主判据）/
`movie_resume_from_disk_cache_skips_done_frames` /
`throughput_speedup_measured_across_worker_counts` /
`worker_stream_movie_mode_renders_real_pixels_and_rejects_foreign_source` /
既有 farm 全量（split / merge / queue / transport / evidence / scheduler）。
