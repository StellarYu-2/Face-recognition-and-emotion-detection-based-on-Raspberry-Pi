# ASDUN Network Technical Plan

## 1. 目标

本文档定义项目后续的网络通信方案，用于支撑以下设备和系统协同工作：

- Raspberry Pi 4：边缘端，负责摄像头采集、本地轻量检测、UI 显示、离线 fallback。
- Windows/NVIDIA 主机：推理端，负责高精度人脸识别、情绪识别和人员库维护。
- ESP32：IoT 端，负责传感器数据采集、设备状态上报和轻量控制。
- 云平台服务器：平台端，负责数据汇聚、存储、设备管理、消息转发和 Web/App 数据接口。
- Web/App：可视化端，负责展示设备状态、识别结果、传感器数据、日志和告警。

核心目标：

- 避免依赖手机热点随机分配的局域网 IP。
- 保证 Pi 到 Windows 推理服务的低延迟通信。
- 支持 ESP32、Pi、Windows 多设备统一上报数据。
- 支持网站或 App 统一查看全部信息。
- 网络异常时，Pi 端 UI 不阻塞，系统可自动降级。
- 后续扩展设备时，不需要重构整体通信架构。

## 2. 总体架构

最终系统采用“边缘推理 + GPU 推理 + IoT 数据平台”的混合网络架构。

```text
                 +----------------------+
                 |      Web / App       |
                 | dashboard / control  |
                 +----------+-----------+
                            |
                            | HTTPS / WebSocket
                            v
                 +----------------------+
                 |  Platform Server     |
                 |  REST API            |
                 |  WebSocket Gateway   |
                 |  MQTT Broker         |
                 |  Database            |
                 +----+------------+----+
                      ^            ^
                      |            |
             MQTT/HTTP|            |MQTT/HTTP
                      |            |
        +-------------+--+      +--+-------------+
        |     ESP32      |      | Raspberry Pi 4 |
        | telemetry      |      | status/result  |
        | device status  |      | local UI       |
        +----------------+      +-------+--------+
                                         |
                                         | HTTP over Tailscale
                                         v
                              +----------------------+
                              | Windows/NVIDIA       |
                              | Inference Server     |
                              | FastAPI / ONNX GPU   |
                              +----------------------+
```

系统中有两个不同的“云”概念，需要明确区分：

| 名称 | 部署位置 | 职责 |
|---|---|---|
| Inference Server | Windows/NVIDIA 主机 | GPU 推理、人脸识别、情绪识别、人员库 |
| Platform Server | 云服务器或主机服务 | 数据汇聚、设备管理、数据库、Web/App 接口 |

Windows/NVIDIA 不建议直接承担完整平台职责。它主要负责高性能 AI 推理；平台服务负责统一数据展示和设备管理。

## 3. 网络分层设计

### 3.1 实时推理链路

实时推理链路用于 Pi 向 Windows/NVIDIA 上传人脸 crop，并接收识别结果。

```text
Pi -> Windows/NVIDIA
```

推荐协议：

```text
HTTP REST + Tailscale
```

推荐地址形式：

```yaml
cloud_server_url: "http://asdun-cloud:8000"
```

或：

```yaml
cloud_server_url: "http://100.x.x.x:8000"
```

其中 `asdun-cloud` 是 Windows/NVIDIA 主机在 Tailscale 中的设备名，`100.x.x.x` 是 Tailscale 分配的稳定虚拟 IP。

设计原则：

- Pi 不写死手机热点分配的 `192.168.x.x` 或 `172.x.x.x` 地址。
- Windows 推理服务监听 `0.0.0.0:8000`。
- Pi 端通过 Tailscale 设备名或稳定虚拟 IP 访问 Windows。
- HTTP 请求必须异步执行，不能阻塞摄像头采集和 UI 渲染。
- 请求失败时保留本地结果或旧结果，不能导致主程序崩溃。

### 3.2 IoT 数据链路

IoT 数据链路用于 ESP32、Pi、Windows 向平台服务器上报状态、传感器数据、识别结果和日志。

```text
ESP32 / Pi / Windows -> Platform Server
```

推荐协议：

```text
MQTT
```

原因：

- MQTT 轻量，适合 ESP32。
- 支持多设备长期在线。
- 支持主题划分，方便扩展。
- 支持实时消息订阅，适合 Web/App 展示。
- 网络抖动时可结合 QoS、保活和重连机制提高稳定性。

### 3.3 可视化访问链路

Web/App 不直接连接 Pi、ESP32 或 Windows，而是只连接 Platform Server。

```text
Web/App -> Platform Server
```

推荐协议：

```text
REST API + WebSocket
```

其中：

- REST API 用于查询历史数据、设备列表、人员库、配置项。
- WebSocket 用于推送实时识别结果、设备在线状态、传感器数据和告警。

这样 Web/App 不需要知道设备当前 IP，也不需要关心设备在哪个网络里。

## 4. 推荐最终方案

本项目推荐采用以下最终网络方案：

```text
Pi -> Windows/NVIDIA:
  HTTP over Tailscale

ESP32 -> Platform Server:
  MQTT

Pi -> Platform Server:
  MQTT or HTTP

Windows/NVIDIA -> Platform Server:
  MQTT or HTTP

Web/App -> Platform Server:
  HTTPS REST API + WebSocket
```

优先级：

1. Pi 到 Windows 的实时推理优先走 Tailscale。
2. ESP32、Pi、Windows 的数据上报统一走 Platform Server。
3. Web/App 只访问 Platform Server。
4. 手机热点只作为临时联网方式，不作为固定网络基础设施。

## 5. 设备职责

### 5.1 Raspberry Pi 4

职责：

- 摄像头采集。
- 本地轻量人脸检测。
- 目标跟踪。
- UI 实时显示。
- 向 Windows/NVIDIA 上传人脸 crop。
- 接收云端身份和情绪结果。
- 网络异常时启用本地 fallback。
- 向 Platform Server 上报运行状态和识别摘要。

Pi 端不应该承担：

- 长期数据库服务。
- 多设备统一管理。
- Web/App 后端。
- 重模型推理主力任务。

### 5.2 Windows/NVIDIA Inference Server

职责：

- 运行 FastAPI 推理服务。
- 使用 NVIDIA GPU 执行 ONNX Runtime / InsightFace 等模型。
- 提供 `/health`、`/analyze`、`/gallery/enroll`、`/gallery` 等接口。
- 维护高精度 embedding gallery。
- 将推理结果或服务状态上报到 Platform Server。

服务监听建议：

```text
host: 0.0.0.0
port: 8000
```

原因：

- 不依赖 Windows 当前局域网 IP。
- Pi 可以通过 Tailscale 地址访问。
- 本地调试也可以通过 `localhost:8000` 访问。

### 5.3 ESP32

职责：

- 采集环境传感器数据，例如温度、湿度、光照、距离、人体感应等。
- 上报设备心跳。
- 接收轻量控制指令。
- 在断网后自动重连。

ESP32 推荐只连接 Platform Server，不直接连接 Windows/NVIDIA 推理服务。

原因：

- ESP32 不适合处理复杂 AI 推理链路。
- 推理服务重启或迁移时，不应该影响 IoT 设备。
- 所有 IoT 数据通过平台统一管理更清晰。

### 5.4 Platform Server

职责：

- MQTT Broker。
- REST API。
- WebSocket 实时推送。
- 数据库存储。
- 设备注册和在线状态管理。
- 日志、告警和历史数据查询。
- Web/App 后端接口。

第一阶段可以部署在 Windows 主机或一台轻量云服务器上。后期如果需要外网访问和长期运行，建议迁移到云服务器。

### 5.5 Web/App

职责：

- 展示 Pi 运行状态。
- 展示 ESP32 传感器数据。
- 展示 Windows 推理服务状态。
- 展示实时识别结果。
- 展示历史记录、日志和告警。
- 后续可扩展设备控制功能。

Web/App 只访问 Platform Server，不直接访问设备端。

## 6. 地址与发现策略

### 6.1 不推荐方式

不推荐在配置中长期写死手机热点分配的局域网 IP：

```yaml
cloud_server_url: "http://192.168.43.120:8000"
```

原因：

- 手机热点重启后 IP 可能变化。
- 不同手机、不同系统、不同地点可能使用不同网段。
- 教室、宿舍、答辩现场的网络环境不可控。
- 每次改配置会降低工程可靠性。

### 6.2 推荐方式

Pi 到 Windows 推理服务推荐使用 Tailscale 设备名：

```yaml
cloud_server_url: "http://asdun-cloud:8000"
```

备用方式使用 Tailscale 固定虚拟 IP：

```yaml
cloud_server_url: "http://100.x.x.x:8000"
```

局域网内调试时可保留 mDNS 备用地址：

```yaml
cloud_server_url: "http://windows-hostname.local:8000"
```

### 6.3 多地址候选机制

为了让系统更专业，建议后续将单个 `cloud_server_url` 升级为候选列表：

```yaml
cloud_server_urls:
  - "http://asdun-cloud:8000"
  - "http://100.x.x.x:8000"
  - "http://windows-hostname.local:8000"

cloud_health_check_path: "/health"
cloud_connect_timeout_ms: 100
cloud_timeout_ms: 300
cloud_fallback_local: true
```

程序启动时依次请求：

```text
GET /health
```

选择第一个可用服务作为当前推理服务。

运行过程中如果当前服务连续失败，可以重新执行健康检查并切换可用地址。

## 7. API 设计

### 7.1 Windows/NVIDIA 推理服务 API

健康检查：

```http
GET /health
```

响应示例：

```json
{
  "ok": true,
  "device": "cuda",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "gallery_count": 2
}
```

分析单个人脸 crop：

```http
POST /analyze
Content-Type: multipart/form-data
```

字段：

```text
track_id: int
frame_id: int
ts_ms: int
source: string
image: jpeg file
local_name: optional string
local_conf: optional float
```

响应示例：

```json
{
  "ok": true,
  "track_id": 3,
  "frame_id": 1024,
  "identity": {
    "name": "yuqin",
    "known": true,
    "confidence": 96.4,
    "distance": 0.42
  },
  "emotion": {
    "label": "Happy",
    "confidence": 91.8
  },
  "latency_ms": 38
}
```

### 7.2 Platform Server API

设备列表：

```http
GET /api/devices
```

最新状态：

```http
GET /api/status/latest
```

识别事件：

```http
GET /api/events/recognition
```

传感器历史：

```http
GET /api/telemetry?device_id=esp32-01&limit=100
```

实时推送：

```text
WebSocket /ws
```

推送内容：

- 设备上线/离线。
- ESP32 传感器数据。
- Pi FPS 和运行模式。
- Windows GPU 推理状态。
- 人脸识别结果。
- 情绪识别结果。
- 告警事件。

## 8. MQTT Topic 规划

推荐统一使用 `asdun/` 作为项目 topic 前缀。

### 8.1 ESP32 上报

传感器数据：

```text
asdun/device/esp32-01/telemetry
```

消息示例：

```json
{
  "device_id": "esp32-01",
  "temperature": 28.4,
  "humidity": 61.0,
  "light": 730,
  "ts_ms": 1710000000000
}
```

设备状态：

```text
asdun/device/esp32-01/status
```

消息示例：

```json
{
  "device_id": "esp32-01",
  "online": true,
  "rssi": -52,
  "uptime_ms": 125000,
  "ts_ms": 1710000000000
}
```

### 8.2 Pi 上报

运行状态：

```text
asdun/device/pi-01/status
```

消息示例：

```json
{
  "device_id": "pi-01",
  "mode": "hybrid",
  "fps": 22.5,
  "cloud_connected": true,
  "active_tracks": 2,
  "ts_ms": 1710000000000
}
```

识别摘要：

```text
asdun/device/pi-01/recognition
```

消息示例：

```json
{
  "device_id": "pi-01",
  "track_id": 3,
  "name": "yuqin",
  "identity_confidence": 96.4,
  "emotion": "Happy",
  "emotion_confidence": 91.8,
  "source": "cloud",
  "ts_ms": 1710000000000
}
```

### 8.3 Windows/NVIDIA 上报

推理服务状态：

```text
asdun/device/asdun-cloud/status
```

消息示例：

```json
{
  "device_id": "asdun-cloud",
  "service": "inference",
  "online": true,
  "provider": "CUDAExecutionProvider",
  "gallery_count": 5,
  "avg_latency_ms": 45,
  "ts_ms": 1710000000000
}
```

推理结果：

```text
asdun/inference/result
```

消息示例：

```json
{
  "source_device": "pi-01",
  "track_id": 3,
  "frame_id": 1024,
  "identity": {
    "name": "yuqin",
    "known": true,
    "confidence": 96.4
  },
  "emotion": {
    "label": "Happy",
    "confidence": 91.8
  },
  "latency_ms": 38,
  "ts_ms": 1710000000000
}
```

### 8.4 控制指令

平台下发控制指令：

```text
asdun/device/{device_id}/command
```

示例：

```json
{
  "command": "set_mode",
  "value": "hybrid",
  "request_id": "cmd-001",
  "ts_ms": 1710000000000
}
```

设备执行结果：

```text
asdun/device/{device_id}/command_result
```

示例：

```json
{
  "request_id": "cmd-001",
  "ok": true,
  "message": "mode updated",
  "ts_ms": 1710000000100
}
```

## 9. 数据存储建议

第一阶段推荐使用 SQLite，降低部署复杂度。

```text
platform_server/data/asdun_platform.sqlite
```

建议表：

| 表名 | 用途 |
|---|---|
| devices | 设备注册信息 |
| device_status | 最新设备状态 |
| telemetry | ESP32 传感器数据 |
| recognition_events | 人脸识别和情绪识别事件 |
| alerts | 告警事件 |
| command_logs | 控制指令记录 |

后期如果需要多人访问、长期运行或大量数据，迁移到 PostgreSQL。

## 10. 安全设计

### 10.1 Tailscale 安全边界

Pi 到 Windows 推理服务通过 Tailscale 私有网络访问。

优点：

- 不需要公网暴露 Windows 8000 端口。
- 不需要依赖手机热点 IP。
- 设备间通信有稳定身份。
- 适合宿舍、教室、展示现场切换。

### 10.2 Platform Server 安全

Platform Server 如果部署到公网，需要：

- HTTPS。
- Web/App 登录认证。
- MQTT 用户名和密码。
- 设备 token。
- 后端接口鉴权。
- 控制指令权限限制。

第一阶段内网调试可以简化，但配置结构要预留认证字段。

推荐配置：

```yaml
platform:
  base_url: "https://api.example.com"
  mqtt_host: "mqtt.example.com"
  mqtt_port: 8883
  mqtt_tls: true
  device_id: "pi-01"
  device_token: "replace-me"
```

## 11. 故障处理策略

### 11.1 Pi 无法连接 Windows 推理服务

处理方式：

- 保持 UI 和本地检测继续运行。
- 保留上一条未过期结果。
- 自动降级为本地识别。
- 周期性重新执行 `/health`。
- 上报 `cloud_connected=false` 到 Platform Server。

### 11.2 ESP32 无法连接 Platform Server

处理方式：

- 本地短暂缓存最近若干条数据。
- 自动重连 MQTT。
- 重连后继续上报最新状态。
- 非关键历史数据可以丢弃，避免占满内存。

### 11.3 Platform Server 离线

处理方式：

- Pi 到 Windows 的推理链路不受影响。
- ESP32 和 Pi 进入离线上报等待状态。
- Web/App 显示平台离线或数据停止更新。
- 平台恢复后设备自动重连。

### 11.4 Windows 推理服务离线

处理方式：

- Pi 本地 fallback。
- Platform Server 标记 `asdun-cloud` 离线。
- Web/App 显示推理端不可用。
- Windows 服务恢复后重新健康检查并恢复 hybrid 模式。

## 12. 配置建议

### 12.1 Pi 配置

```yaml
device_id: "pi-01"
inference_mode: "hybrid"

cloud_server_urls:
  - "http://asdun-cloud:8000"
  - "http://100.x.x.x:8000"
  - "http://windows-hostname.local:8000"

cloud_health_check_path: "/health"
cloud_connect_timeout_ms: 100
cloud_timeout_ms: 300
cloud_min_interval_ms: 600
cloud_max_queue_size: 2
cloud_jpeg_quality: 85
cloud_crop_size: 256
cloud_fallback_local: true

platform_enabled: true
platform_mqtt_host: "mqtt.example.com"
platform_mqtt_port: 8883
platform_mqtt_tls: true
platform_device_token: "replace-me"
```

### 12.2 Windows/NVIDIA 配置

```yaml
device_id: "asdun-cloud"
host: "0.0.0.0"
port: 8000
preferred_provider: "CUDAExecutionProvider"

platform_enabled: true
platform_mqtt_host: "mqtt.example.com"
platform_mqtt_port: 8883
platform_mqtt_tls: true
platform_device_token: "replace-me"
```

### 12.3 ESP32 配置

```text
device_id=esp32-01
wifi_ssid=your_wifi_or_hotspot
wifi_password=replace-me
mqtt_host=mqtt.example.com
mqtt_port=8883
mqtt_tls=true
device_token=replace-me
telemetry_interval_ms=1000
heartbeat_interval_ms=5000
```

## 13. 部署阶段

### 阶段 A：解决 Pi 到 Windows 的稳定访问

目标：

- Pi 和 Windows 安装并登录 Tailscale。
- Windows FastAPI 服务监听 `0.0.0.0:8000`。
- Pi 使用 `http://asdun-cloud:8000/health` 测试连通。
- Pi 配置不再依赖手机热点 IP。

完成标准：

```text
Pi 在宿舍和教室切换网络后，仍然可以访问 Windows /health。
```

### 阶段 B：平台服务器骨架

目标：

- 新增 Platform Server。
- 提供 REST API。
- 提供 WebSocket。
- 接入数据库。
- 接收 Pi 和 Windows 的状态上报。

完成标准：

```text
Web 页面可以看到 Pi 和 Windows 是否在线。
```

### 阶段 C：ESP32 MQTT 上报

目标：

- ESP32 连接 MQTT Broker。
- ESP32 上报 telemetry 和 status。
- Platform Server 写入数据库。
- Web/App 实时显示传感器数据。

完成标准：

```text
Web/App 可以实时看到 ESP32 数据变化。
```

### 阶段 D：识别结果统一展示

目标：

- Pi 或 Windows 将识别结果上报到 Platform Server。
- Web/App 实时显示人脸身份、情绪、置信度和时间。
- 历史识别事件可查询。

完成标准：

```text
Web/App 可以统一查看设备状态、传感器数据和识别结果。
```

### 阶段 E：控制与告警

目标：

- Web/App 可下发控制指令。
- ESP32 或 Pi 执行指令并返回结果。
- 平台支持告警规则。

完成标准：

```text
Web/App 可以控制设备，并能接收异常告警。
```

## 14. 当前阶段建议

当前项目最应该优先完成：

1. Pi 到 Windows 推理服务改用 Tailscale 设备名。
2. 保留 Pi 本地 fallback。
3. 将 `cloud_server_url` 升级为 `cloud_server_urls` 候选列表。
4. 增加启动时 `/health` 自动探测。
5. 后续再引入 Platform Server 和 MQTT。

当前命名建议：

```text
Raspberry Pi hostname / Tailscale name: asdun
Windows GPU Tailscale name: asdun-cloud
Pi 访问推理服务: http://asdun-cloud:8000
```

这条路线可以先解决当前 IP 变化问题，同时不会阻碍后续接入 ESP32、Web/App 和真正的平台云端。

## 15. 成功标准

最终系统应达到：

- Pi 和 Windows 换网络后不需要手动修改 IP。
- Pi 实时画面不会因为网络请求卡顿。
- Windows GPU 推理服务可以稳定被 Pi 发现和调用。
- ESP32 可以独立上报传感器数据。
- Web/App 可以统一观察 Pi、Windows、ESP32 的运行状态。
- Web/App 可以查看实时识别结果和历史记录。
- Platform Server 离线时，不影响 Pi 到 Windows 的核心推理链路。
- Windows 推理服务离线时，Pi 可以降级为本地模式继续运行。
