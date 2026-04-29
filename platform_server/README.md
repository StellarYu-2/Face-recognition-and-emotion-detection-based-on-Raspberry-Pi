# ASDUN Platform Server

这是 `network.md` 阶段 B 的最小平台服务：负责接收设备状态、保存到 SQLite，并通过 REST API + WebSocket 提供给 Web 页面查看。

当前先使用 HTTP 上报状态，后续可以把同一套数据库和 WebSocket 推送接到 MQTT Broker。

阶段 B 的目标是：

```text
Web 页面可以看到 asdun@asdun 这台树莓派和 asdun-cloud 这台 Windows 推理端是否在线。
```

## Run

从项目根目录启动：

```powershell
.\scripts\run_platform_server.ps1
```

默认地址：

```text
http://127.0.0.1:9000
```

如果树莓派通过 Tailscale 访问 Windows 主机，Pi 端配置使用：

```yaml
platform_base_url: "http://asdun-cloud:9000"
```

健康检查：

```powershell
curl.exe http://127.0.0.1:9000/health
```

## Post Device Status

正常情况下不需要手动模拟：

- Windows 推理服务启动后会自动上报 `asdun-cloud`。
- Platform Server 会主动探测 `asdun:22`，只要树莓派连上 Tailscale/SSH 可达，`asdun@asdun` 就会显示在线。
- Raspberry Pi 程序启动后会额外上报 `mode`、`fps`、`cloud_connected` 等运行状态。

手动模拟 Windows 推理服务在线：

```powershell
.\scripts\post_platform_status.ps1 -Preset cloud
```

手动模拟 Raspberry Pi 在线：

```powershell
.\scripts\post_platform_status.ps1 -Preset pi
```

浏览器打开：

```text
http://127.0.0.1:9000
```

如果两条状态都已上报，页面会显示 `pi-01` 和 `asdun-cloud`。

## API

设备状态上报：

```http
POST /api/status
Content-Type: application/json
```

示例：

```json
{
  "device_id": "pi-01",
  "role": "raspberry_pi",
  "display_name": "Raspberry Pi 4",
  "online": true,
  "status": {
    "mode": "hybrid",
    "fps": 22.5,
    "cloud_connected": true,
    "active_tracks": 2
  }
}
```

识别事件上报：

```http
POST /api/events/recognition
Content-Type: application/json
```

示例：

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
  "latency_ms": 38
}
```

常用读取接口：

```text
GET /api/snapshot
GET /api/devices
GET /api/status/latest
GET /api/people
GET /api/events/recognition?limit=100
GET /api/events/recognition?person=yuqin&limit=500
WebSocket /ws
```

`/api/people` 会按已知识别人员聚合历史事件，并计算每种情绪的出现次数占比：

```text
emotion percent = emotion event count / total known recognition events for that person * 100
```

Dashboard 的 People 表可以点选人员，或使用人员下拉框筛选 Recognition 明细。
