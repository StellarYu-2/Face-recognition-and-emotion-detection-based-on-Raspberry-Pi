# ASDUN 企业级嵌入式项目竞争力建设路线

## 1. 项目定位

本项目建议定位为：

> 基于 Raspberry Pi 的边缘端智能门禁系统，使用 C++17 在嵌入式 Linux 环境中完成摄像头采集、本地轻量推理、目标跟踪、身份识别、云端 GPU 协同推理、异常降级、设备状态上报和性能优化。

这个定位比“树莓派人脸识别项目”更贴近企业招聘中的嵌入式软件、驱动、系统联调、性能优化和产品质量岗位要求。

当前项目已经具备较好的基础：

- C++17 + CMake 工程结构。
- Raspberry Pi 端本地推理链路。
- OpenCV / ncnn / SQLite / libcurl 等工程组件。
- 摄像头采集线程、推理管线、目标跟踪、云端异步客户端。
- Windows GPU 云端 FastAPI 推理服务。
- 本地 fallback 与 hybrid cloud inference 设计。

后续重点不是继续堆功能，而是把项目做成“企业级软件系统”：可测、可观测、可调优、可降级、可维护、可部署。

## 2. JD 能力映射

| JD 要求 | 项目中可对应的能力 | 后续应强化的方向 |
|---|---|---|
| 嵌入式软件方案设计 | 边缘端 + 云端协同架构 | 补系统架构文档、模块边界、数据流、异常流 |
| C/C++ 编码 | Raspberry Pi 端 C++17 主程序 | 强化 RAII、接口抽象、错误处理、单元测试 |
| 开发者测试 | 当前测试体系较弱 | 增加 CTest、单元测试、回放测试、故障注入 |
| 联调 | Pi 与 Windows GPU 服务联调 | 增加网络诊断、健康检查、自动重连、链路日志 |
| 问题定界/定位 | 当前有部分 debug 日志 | 增加 PerfMonitor、trace id、模块耗时、错误码 |
| 持续提升产品质量 | 已有 tracking / quality gate | 增加 CI、静态检查、ASan/TSan、性能基线 |
| 组件/模块看护 | Camera / Pipeline / Cloud / Tracking | 为每个核心模块建立指标、测试和维护文档 |
| CPU / Linux 理解 | 运行在 Raspberry Pi Linux | 增加 V4L2、GPIO、systemd、perf、CPU affinity |
| 驱动 / 操作系统经验 | 当前偏应用层 | 增加 V4L2 backend、GPIO 控制、可选字符设备驱动 |
| 大规模并行化软件 | 当前有采集线程和云端线程 | 改造为多线程流水线和 bounded queue |
| AI 辅助软件开发 | 项目天然适合 AI 工具参与 | 记录 AI 辅助调试、生成测试、性能分析过程 |

## 3. 企业级项目亮点目标

最终希望这个项目在简历和面试中能体现以下亮点：

1. 设计并实现嵌入式 Linux 边缘端 AI 系统。
2. 使用 C++17 完成实时视频采集、推理调度、目标跟踪和设备控制。
3. 针对 Raspberry Pi ARM CPU 进行性能分析和推理调度优化。
4. 设计多线程流水线，保障 UI、采集、推理和网络请求互不阻塞。
5. 实现云端 GPU 协同推理、健康检查、超时保护、失败降级和自动重连。
6. 建立端到端性能观测体系，能够定位 FPS、延迟、CPU、内存、温度瓶颈。
7. 建立开发者测试体系，包括单元测试、回放测试、故障注入和性能回归测试。
8. 接入真实外设控制，例如 GPIO 继电器、LED、蜂鸣器或传感器。
9. 支持 systemd 部署、日志轮转、配置管理和开机自启动。
10. 具备工程化文档，包括架构设计、联调手册、问题定位手册和性能报告。

## 4. 推荐后续建设路线

### 阶段 A：性能观测与问题定位

优先级最高。没有指标，就无法证明优化效果。

建议新增 `PerfMonitor` 模块，记录以下指标：

- 采集耗时：camera capture latency。
- 检测耗时：face detect latency。
- 识别耗时：recognition latency。
- 表情耗时：emotion latency。
- 云端耗时：cloud roundtrip latency。
- 渲染耗时：render latency。
- 端到端耗时：frame total latency。
- FPS。
- active tracks 数量。
- dropped frames 数量。
- cloud queue depth。
- CPU 使用率。
- 内存占用。
- Raspberry Pi 温度。
- 是否发生降频。

建议输出格式：

```text
frame=1024 fps=22.8 capture=3.1ms detect=38.4ms recog=12.7ms cloud_rtt=86.2ms render=4.5ms cpu=72% mem=184MB temp=61C dropped=3
```

也可以增加 CSV/JSON 输出：

```text
logs/perf_2026-04-25.csv
logs/perf_latest.json
```

推荐验收标准：

- 能连续运行 30 分钟并输出稳定性能数据。
- 能区分本地推理耗时和云端推理耗时。
- 能定位卡顿来自采集、推理、网络还是渲染。
- 能形成一份 `docs/performance_report.md`，记录优化前后对比。

### 阶段 B：多线程实时流水线

当前项目已有摄像头采集线程和云端 worker，但主推理链路仍可以进一步工程化。

推荐目标架构：

```text
Capture Thread
  -> Frame Queue
Detect / Preprocess Thread
  -> Detection Queue
Recognition / Emotion Worker
  -> Result Queue
Render / Main Thread
Cloud Worker
```

关键点：

- 使用 bounded queue，限制内存增长。
- 支持 drop-old / keep-latest 策略，优先保证实时性。
- 推理慢时允许丢旧帧，不阻塞摄像头采集。
- 每个队列暴露 depth、push count、drop count。
- 每个线程支持优雅退出。
- 每个 stage 都写入性能指标。
- 可选支持 CPU affinity，把采集、推理、渲染绑定到不同 CPU 核。

建议新增或重构：

- `include/core/BoundedQueue.hpp`
- `include/core/PerfMonitor.hpp`
- `src/core/PerfMonitor.cpp`
- `include/core/PipelineRuntime.hpp`
- `src/core/PipelineRuntime.cpp`

推荐验收标准：

- 摄像头采集不因云端请求阻塞。
- 云端断开时 UI 仍然流畅。
- 推理耗时升高时队列不会无限增长。
- 日志中可以看到每个 stage 的耗时和丢帧数量。

### 阶段 C：嵌入式 Linux 能力强化

这一阶段用来贴近“驱动、Linux、CPU、操作系统”方向。

推荐方向 1：V4L2 摄像头 backend

当前采集主要依赖 OpenCV `VideoCapture`。企业面试中，如果能讲清楚 V4L2，会明显更有嵌入式味道。

建议新增：

- `/dev/video0` 打开和能力查询。
- `ioctl` 配置分辨率、FPS、pixel format。
- `mmap` 分配视频缓冲区。
- 支持 MJPEG / YUYV / NV12。
- buffer 入队、出队和帧时间戳。
- 采集失败自动恢复。

推荐模块名：

```text
include/camera/ICameraBackend.hpp
include/camera/OpenCVCameraBackend.hpp
include/camera/V4L2CameraBackend.hpp
src/camera/V4L2CameraBackend.cpp
```

推荐方向 2：GPIO 门禁控制

接入真实外设，让项目从 AI demo 变成嵌入式系统。

可实现：

- 识别成功后控制继电器开门。
- 未知人员蜂鸣器告警。
- 云端离线 LED 黄灯。
- 系统异常 LED 红灯。
- 支持配置开门持续时间、冷却时间、防重复触发。

推荐模块名：

```text
include/device/GpioController.hpp
include/device/DoorLockController.hpp
src/device/GpioController.cpp
src/device/DoorLockController.cpp
```

推荐方向 3：systemd 部署

企业项目必须能稳定部署。

建议增加：

- `deploy/asdun-access.service`
- 开机自启动。
- 崩溃自动重启。
- 日志输出到 journald。
- 配置路径通过命令行参数指定。
- graceful shutdown。

推荐验收标准：

- Raspberry Pi 重启后程序自动启动。
- 程序崩溃后 systemd 自动拉起。
- `journalctl -u asdun-access` 能看到关键日志。

### 阶段 D：云端协同与网络可靠性

当前 `CloudClient` 已经支持候选 URL、健康检查、异步请求和本地 fallback。后续可以继续增强为企业级链路。

建议增强点：

- 运行中连续失败后自动重新执行 `/health`。
- 支持多个云端地址的故障切换。
- 支持指数退避重连。
- 每个请求带 `request_id` / `frame_id` / `track_id`。
- 统计成功率、平均 RTT、P95 RTT、超时次数。
- 对 HTTP 500、超时、DNS 失败、JSON 异常分别打错误码。
- 云端恢复后自动恢复 hybrid 模式。

推荐新增诊断输出：

```text
cloud_status connected=true active=http://asdun-cloud:8000 rtt_avg=72ms rtt_p95=130ms timeout=3 failover=1
```

推荐验收标准：

- 拔掉 Windows GPU 服务后，Pi 本地识别不中断。
- 恢复 Windows GPU 服务后，Pi 能自动重新使用云端结果。
- 网络异常不会卡死 UI 或采集线程。
- 日志可以明确区分 DNS 问题、连接超时、服务错误、响应解析错误。

### 阶段 E：开发者测试体系

这是企业招聘中很容易体现成熟度的部分。

建议引入：

- CTest。
- GoogleTest 或 Catch2。
- fake detector / fake recognizer / fake cloud server。
- 测试数据集和图片序列回放。
- 性能回归测试。
- AddressSanitizer / ThreadSanitizer。
- clang-tidy / cppcheck。

建议优先测试模块：

| 模块 | 测试重点 |
|---|---|
| `TrackManager` | IOU 匹配、TTL、身份切换、云端结果过期 |
| `ConfidenceMapper` | 距离到置信度映射、边界值 |
| `CloudClient` | 队列满、超时、服务不可达、异常 JSON |
| `InferencePipeline` | detect interval、recognition interval、emotion interval 调度 |
| `FaceQualityGate` | 模糊度、人脸面积、稳定帧判断 |
| `Database` | 建表、增删改查、重复人员、异常路径 |

推荐目录：

```text
tests/
  test_track_manager.cpp
  test_confidence_mapper.cpp
  test_cloud_client.cpp
  test_quality_gate.cpp
  fixtures/
```

推荐验收标准：

- `ctest` 可以一键运行。
- 核心模块测试覆盖正常路径和异常路径。
- 每次重构后能快速确认行为没有退化。

### 阶段 F：配置、日志与可维护性

当前配置解析是手写 pseudo-YAML，云端 JSON 也是手写字符串解析。为了企业级质量，建议升级。

建议改造：

- 使用 `yaml-cpp` 解析 `config/app.yaml`。
- 使用 `nlohmann/json` 或 RapidJSON 解析云端响应。
- 使用 `spdlog` 替代散落的 `std::cout` / `std::cerr`。
- 日志分级：trace / debug / info / warn / error。
- 日志带模块名、frame_id、track_id、request_id。
- 支持日志轮转。

推荐日志格式：

```text
2026-04-25 21:30:12.345 [INFO] [Pipeline] frame=1024 tracks=2 detect_ms=36.8 recog_ms=11.2
2026-04-25 21:30:12.431 [WARN] [CloudClient] request_id=abc123 track=3 error=timeout rtt_ms=2000
```

推荐验收标准：

- 面对线上问题时，可以通过日志还原一次识别请求的完整路径。
- 配置错误能给出明确提示。
- JSON 响应异常不会导致程序崩溃。

## 5. 企业级功能优先级

| 优先级 | 功能 | 对招聘竞争力的价值 |
|---|---|---|
| P0 | PerfMonitor 性能观测 | 能证明你会定位问题和优化系统 |
| P0 | BoundedQueue + 多线程流水线 | 对应并行化、实时性、嵌入式调度 |
| P0 | CTest 单元测试 | 对应开发者测试和质量保障 |
| P1 | CloudClient 自动重连和故障切换 | 对应联调、可靠性、异常处理 |
| P1 | V4L2 摄像头 backend | 对应 Linux、驱动、底层接口 |
| P1 | GPIO 门禁控制 | 对应真实嵌入式外设控制 |
| P1 | systemd 部署 | 对应工程交付和长期运行 |
| P2 | MQTT 平台上报 | 对应 IoT 设备管理和平台化 |
| P2 | Web dashboard | 对应可视化和产品完整度 |
| P2 | 模型量化与 ncnn 性能优化 | 对应 AI 工程和 ARM CPU 优化 |

## 6. 可量化验收指标

建议后续每次优化都保留数据，形成前后对比。

| 指标 | 目标 |
|---|---|
| 稳定运行时长 | Raspberry Pi 连续运行 2 小时无崩溃 |
| UI 流畅性 | 云端断开时 UI 不阻塞 |
| 采集实时性 | 摄像头采集线程不被推理和网络阻塞 |
| FPS | 在目标分辨率下稳定达到可接受 FPS，例如 15+ 或 20+ |
| 端到端延迟 | 本地模式和 hybrid 模式分别统计 P50 / P95 |
| 云端超时处理 | 超时后自动 fallback，不影响主流程 |
| 内存稳定性 | 长时间运行内存无持续增长 |
| CPU 温度 | 高负载下记录温度与降频情况 |
| 单元测试 | 核心模块可通过 `ctest` 一键验证 |
| 故障注入 | 云端断开、恢复、异常响应均可验证 |

## 7. 推荐项目文档体系

建议逐步补齐这些文档：

```text
docs/
  architecture.md          # 系统架构、模块边界、数据流
  performance_report.md    # 性能数据和优化前后对比
  debugging_guide.md       # 常见问题定位手册
  deployment_pi.md         # Raspberry Pi 部署手册
  cloud_integration.md     # Pi 与 Windows GPU 云端联调手册
  test_plan.md             # 开发者测试和故障注入计划
  v4l2_backend.md          # V4L2 采集实现说明
```

企业面试中，文档不是装饰，它能证明你有系统化工程思维。

## 8. 简历表达模板

项目名称可以写为：

> 基于 Raspberry Pi 的边缘端智能门禁与云端协同推理系统

简历描述可以按下面方式组织：

```text
基于 Raspberry Pi 4 设计并实现边缘端智能门禁系统，使用 C++17、OpenCV、ncnn、SQLite、libcurl 完成本地视频采集、人脸检测、身份识别、目标跟踪、云端 GPU 协同推理和本地 fallback。
```

亮点 bullet 可以写：

```text
- 设计多线程实时推理流水线，将摄像头采集、检测、识别、渲染和云端请求解耦，通过 bounded queue 和丢帧策略保障实时性。
- 针对 Raspberry Pi ARM CPU 建立性能观测体系，统计 FPS、端到端延迟、模块耗时、CPU、内存、温度和 dropped frames，并基于数据优化推理调度参数。
- 实现 Pi 与 Windows/NVIDIA 云端推理服务的 hybrid inference，支持健康检查、超时保护、异步请求、本地 fallback 和云端恢复后的自动切换。
- 建立核心模块开发者测试，包括目标跟踪、质量门控、云端异常、推理调度和数据库读写，提升重构安全性和问题定位效率。
- 接入 GPIO 门禁外设，支持识别成功开门、未知人员告警、云端离线状态提示和 systemd 开机自启动部署。
```

当你完成真实指标后，可以把描述升级为：

```text
- 通过性能埋点定位检测模型为主要瓶颈，调整 detect_interval、crop_size、ncnn 线程数和云端请求频率，使端侧 FPS 从 X 提升到 Y，P95 延迟从 A ms 降低到 B ms。
- 通过异步云端请求和 bounded queue，将网络异常对主流程的影响控制在 0 阻塞，云端断开时系统可自动降级为本地模式并持续运行。
```

## 9. 面试讲述思路

建议按这个闭环讲项目：

1. 背景：为什么要做边缘端智能门禁，而不是纯云端识别。
2. 约束：Raspberry Pi CPU 性能有限、摄像头实时性要求高、网络不稳定。
3. 架构：本地轻量推理 + Windows GPU 云端增强 + 本地 fallback。
4. 难点：实时性、云端延迟、识别稳定性、多线程同步、异常恢复。
5. 方案：流水线、bounded queue、性能埋点、质量门控、目标跟踪、健康检查。
6. 数据：优化前后 FPS、延迟、CPU、内存、温度、超时恢复时间。
7. 质量：单元测试、回放测试、故障注入、日志追踪。
8. 复盘：哪些地方还能继续做，比如 V4L2、GPIO、systemd、MQTT 平台化。

## 10. 推荐下一步实际任务

最建议马上开始的三个任务：

1. 新增 `PerfMonitor`，先把性能数据跑出来。
2. 新增 `BoundedQueue`，为多线程流水线重构做准备。
3. 新增 `tests/` 和 CTest，先覆盖 `TrackManager`、`ConfidenceMapper`、`FaceQualityGate`。

完成这三件事后，这个项目会从“能展示的 AI 项目”开始变成“能面向企业招聘讲工程能力的嵌入式项目”。

