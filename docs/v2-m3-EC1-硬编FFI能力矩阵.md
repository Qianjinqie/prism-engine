# EC-1 硬编 FFI 能力矩阵（V2.0-M3）

> 数据同源：本文表格由 `prism-ffi-hwenc::capability_matrix_text()` 渲染口径手工
> 同步；判据代码在 `crates/prism-ffi-hwenc/`（探测链 `env_probe.rs`、汇入 `lib.rs`）。
> 复现命令：`cargo test -p prism-ffi-hwenc --features videotoolbox,nvenc,vaapi`。

## 1. 能力矩阵（本机实测）

本机（沙箱）环境探测实录（探测链真实读本机，非占位话术）：

| 探测点 | 实测结果 |
|---|---|
| 平台 `std::env::consts::OS` | `linux` |
| VideoToolbox 平台判定 | **不符**（仅 macOS/iOS） |
| NVENC `/proc/driver/nvidia/version` | **不存在** |
| NVENC `/dev/nvidia*`（nvidiactl/nvidia-uvm/nvidia0..7） | **无任何节点** |
| VAAPI `/dev/dri` 目录 | **不存在**（`renderD*` 更无从谈起） |

| 后端 | feature 状态（本构建） | 平台/环境证据 | 探测结论（如实） |
|---|---|---|---|
| `videotoolbox` | 未编入（默认）；`--features videotoolbox` 可编入 | 平台不符：OS = linux | `Unsupported`：平台不符——VideoToolbox 仅 macOS/iOS，本机无该平台硬件 |
| `nvenc` | 未编入（默认）；`--features nvenc` 可编入 | 无 NVIDIA 证据：驱动版本文件不存在、`/dev/nvidia*` 无节点 | `Unsupported`：未发现 NVIDIA 硬件证据——无驱动即无 NVENC |
| `vaapi` | 未编入（默认）；`--features vaapi` 可编入 | 无 DRM 证据：`/dev/dri` 不存在 | `Unsupported`：未发现可用渲染节点——无 DRM 设备即无 VA-API |

**三平台真机启用均缺硬件**（macOS 机、NVIDIA 机、带 `/dev/dri` 的 Linux 机都不在
本环境），故本机最好结论只到"环境证据在/不在"；`ProbeStatus::Available` 需要
真机 + FFI 绑定，本机不可达——**缺硬件不伪装成已做**。

## 2. 探测链（真探测清单）

`crates/prism-ffi-hwenc/src/env_probe.rs`（纯 `std`、零依赖）：

| 后端 | 真探测动作 | 证据结构 |
|---|---|---|
| `videotoolbox` | 读 `std::env::consts::OS`，与 `{macos, ios}` 比对 | `EnvironmentReport::videotoolbox_platform` |
| `nvenc` | 读 `/proc/driver/nvidia/version` 首行；逐个 `stat` `/dev/nvidia{ctl,-uvm,0..7}` | `NvencEvidence { driver_version_line, device_nodes }` |
| `vaapi` | `read_dir("/dev/dri")` 收集 `renderD*` 节点 | `VaapiEvidence { dri_dir_present, render_nodes }` |

纪律：无副作用（只读文件系统 + 编译期常量）、可重复（两遍逐字段相同）、
不 panic、不静默补默认值；每条 `Unsupported` 理由必带真实证据文本
（平台串 / 驱动文件 / 设备节点）。

## 3. feature gate 编译矩阵（本机验证记录）

| 命令 | 结果 |
|---|---|
| `cargo check -p prism-ffi-hwenc`（默认全关） | ✅ Finished |
| `cargo check -p prism-ffi-hwenc --features videotoolbox` | ✅ Finished |
| `cargo check -p prism-ffi-hwenc --features nvenc` | ✅ Finished |
| `cargo check -p prism-ffi-hwenc --features vaapi` | ✅ Finished |
| `cargo test -p prism-ffi-hwenc --features videotoolbox` | ✅ 11 passed |
| `cargo test -p prism-ffi-hwenc --features nvenc` | ✅ 11 passed |
| `cargo test -p prism-ffi-hwenc --features vaapi` | ✅ 11 passed |
| `cargo test -p prism-ffi-hwenc --features videotoolbox,nvenc,vaapi` | ✅ 11 passed |

主干隔离纪律保持：`[dependencies]` 为空，任何主干 crate 不依赖本 crate——
`cargo build -p prism-mux` 等的依赖图不受影响（零第三方依赖铁律不破）。

## 4. 降级路径（明确报错，不静默失败）

两级：

1. **探测级**：`HwEncoderProbe::probe()` 永远返回结构化 `HwProbe`——
   不可用时 `status = Unsupported` 且 `reason` 非空、带真实证据；
2. **调用级**：`require_available() -> Result<HwProbe, HwUnavailableError>`——
   无可用后端时 `Err`，错误文本列出**全部**后端的环境证据与结论，并含建议段
   （C19：确认目标机有对应硬件并开 feature；无硬件时改走软件编码路径）。
   测试 `require_available_errors_explicitly_without_silent_success` 钉住
   "本机必须显式报错"。

## 5. 环境受限项（如实清单，不拿缺硬件当少做理由）

| 受限项 | 证据 | 处理 |
|---|---|---|
| 真机硬编启用（三平台） | 矩阵第 1 节全部探测点缺席 | 不伪装；探测链/gate 矩阵/降级路径/文档四件齐交付 |
| ffprobe 体积对比（硬编 vs 软编） | 缺硬件即无硬编码产物可比 | **不写 ffprobe 堆料测试**（用户砍过式冗余）；标注为环境受限项，待真机波次补 |
| 平台库 link / extern 声明 | 需要目标机头文件与真机 | 本波次刻意不做（`PENDING_BINDING_NOTE` 在理由中如实点明） |

## 6. 复现

```bash
cargo test -p prism-ffi-hwenc --features videotoolbox,nvenc,vaapi
# capability_matrix_covers_all_three_backends_with_real_rows
# every_enabled_probe_carries_real_environment_evidence
# require_available_errors_explicitly_without_silent_success
```
