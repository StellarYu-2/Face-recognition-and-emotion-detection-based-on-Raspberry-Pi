# ASDUN Platform Server

## 目录定位

这个目录是 Platform Server 的程序本体，包含 FastAPI 后端、网页静态文件、依赖文件和 Docker 镜像构建文件。

云服务器怎么启动、怎么配 Nginx、怎么配 systemd、怎么写环境变量，统一放在：

```text
deploy/platform_server/
```

`platform_server` 是 ASDUN 的平台后端，负责：

- 接收设备在线状态。
- 接收 Windows 推理服务上报的人脸识别/情绪事件。
- 接收 ESP32 温湿度等遥测数据。
- 保存到 SQLite。
- 给网页通过 REST API 和 WebSocket 展示实时数据。

当前建议先继续使用 FastAPI，后续 React 前端、云端部署、设备控制命令都可以围绕这套 API 扩展。

云服务器部署说明见：

```text
deploy/platform_server/README.md
```

## 启动

如果是新 clone 的仓库，先复制本地配置模板：

```powershell
Copy-Item .\config\app.example.yaml .\config\app.yaml
Copy-Item .\cloud_server\config.example.yaml .\cloud_server\config.yaml
```

然后把 `config/app.yaml`、`cloud_server/config.yaml` 中的私有地址和 token 改成你自己的值。这两个真实配置文件被 `.gitignore` 忽略，不应该提交到 GitHub。

从项目根目录运行：

```powershell
.\scripts\run_platform_server.ps1
```

默认地址：

```text
http://127.0.0.1:9000
```

健康检查：

```powershell
curl.exe http://127.0.0.1:9000/health
```

网页：

```text
http://127.0.0.1:9000
```

## 设备 Token

本地调试默认不开启 token，所以现有 Pi、Windows、ESP32 不会被拦住。

准备公网部署前，建议开启：

```powershell
$env:ASDUN_DEVICE_AUTH_ENABLED = "true"
$env:ASDUN_DEVICE_TOKENS = "pi-01=pi-token,gpu-server=cloud-token,esp32-01=esp32-token"
.\scripts\run_platform_server.ps1
```

公网云服务器通常不能直接访问你的 Tailscale 主机名，例如 `raspberry-pi.local`，建议在云端环境变量里关闭主动探测：

```text
ASDUN_PROBE_PI_ENABLED=false
```

设备上报时带请求头：

```http
X-ASDUN-Device-Id: esp32-01
X-ASDUN-Device-Token: esp32-token
```

也可以用 JSON 格式配置 token：

```powershell
$env:ASDUN_DEVICE_TOKENS = '{"pi-01":"pi-token","gpu-server":"cloud-token","esp32-01":"esp32-token"}'
```

## 常用 API

读取：

```text
GET /health
GET /api/config/public
GET /api/snapshot
GET /api/devices
GET /api/status/latest
GET /api/people
GET /api/events/recognition?limit=100
GET /api/events/recognition?person=yuqin&limit=500
GET /api/telemetry?device_id=esp32-01&limit=100
GET /api/commands?device_id=esp32-01&limit=100
GET /api/commands/pending?device_id=esp32-01
WebSocket /ws
```

写入：

```text
POST /api/status
POST /api/events/recognition
POST /api/telemetry
POST /api/commands
POST /api/commands/{command_id}/result
```

## 状态上报

```http
POST /api/status
Content-Type: application/json
```

```json
{
  "device_id": "pi-01",
  "role": "raspberry_pi",
  "display_name": "raspberry-pi",
  "online": true,
  "status": {
    "mode": "hybrid",
    "fps": 22.5,
    "cloud_connected": true,
    "active_tracks": 2
  }
}
```

手动模拟：

```powershell
.\scripts\post_platform_status.ps1 -Preset pi
.\scripts\post_platform_status.ps1 -Preset cloud
```

如果开启 token：

```powershell
.\scripts\post_platform_status.ps1 -Preset pi -DeviceToken "pi-token"
```

## 识别事件

```http
POST /api/events/recognition
Content-Type: application/json
```

```json
{
  "source_device": "pi-01",
  "producer_device": "gpu-server",
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

Unknown 结果默认不由 Windows 推理服务上报到平台。

## ESP32 遥测

推荐 ESP32 温湿度使用专门接口：

```http
POST /api/telemetry
Content-Type: application/json
```

```json
{
  "device_id": "esp32-01",
  "role": "esp32",
  "display_name": "esp32-01",
  "online": true,
  "telemetry": {
    "temperature": 28.4,
    "humidity": 61.0,
    "light": 730,
    "rssi": -52,
    "uptime_ms": 125000
  }
}
```

平台会同时：

- 保存一条 telemetry 历史记录。
- 更新 Devices 表里的 `esp32-01` 最新状态。
- 通过 WebSocket 推送到网页。

## 控制命令

平台支持命令队列：

```text
Web / API 创建命令 -> 设备轮询 pending 命令 -> 设备执行 -> 设备回传结果
```

创建命令：

```http
POST /api/commands
Content-Type: application/json
```

```json
{
  "device_id": "esp32-01",
  "command": "set_mode",
  "payload": {
    "mode": "test"
  }
}
```

设备轮询：

```http
GET /api/commands/pending?device_id=esp32-01&limit=10
```

设备回传结果：

```http
POST /api/commands/{command_id}/result
Content-Type: application/json
```

```json
{
  "device_id": "esp32-01",
  "ok": true,
  "message": "done",
  "result": {
    "mode": "test"
  }
}
```

本地模拟完整流程：

```powershell
.\scripts\test_platform_command_flow.ps1 -PlatformUrl http://127.0.0.1:9000 -DeviceId esp32-01
```

如果设置了 `ASDUN_ADMIN_TOKEN`，创建命令时需要请求头：

```http
X-ASDUN-Admin-Token: your-admin-token
```

Dashboard 的 Commands 区域也可以直接创建命令。公网环境如果设置了 `ASDUN_ADMIN_TOKEN`，先在 `Admin token` 输入框填入管理 token，再点击 `Send` 或常用命令按钮。网页会把 token 保存在当前浏览器的 `localStorage`，下次打开同一个浏览器会自动填入；需要清除时点击 `Clear`。

常用命令按钮：

| 按钮 | command |
|---|---|
| `Ping` | `ping` |
| `Status` | `status` |
| `Reload gallery` | `reload_gallery` |

Raspberry Pi 端现在会按 `config/app.yaml` 中的配置轮询命令：

```yaml
platform_command_poll_enabled: true
platform_command_poll_interval_ms: 5000
platform_command_poll_limit: 5
```

Pi 当前支持的命令：

| command | 说明 |
|---|---|
| `ping` | 连通性测试，返回 `pong` |
| `status` | 返回当前推理模式和云端连接状态 |
| `reload_gallery` | 重新加载本地人脸库 |
| `reload_people` | `reload_gallery` 的别名 |
| `set_mode` | 仅检查/确认模式；运行时切换模式会返回需要改配置并重启 |

例如让 Pi 重新加载人脸库：

```json
{
  "device_id": "pi-01",
  "command": "reload_gallery",
  "payload": {}
}
```

真实测试 Pi 命令执行：

```powershell
.\scripts\post_platform_command.ps1 -PlatformUrl https://api.asdun.example.com -DeviceId pi-01 -Command ping -AdminToken your-admin-token
.\scripts\post_platform_command.ps1 -PlatformUrl https://api.asdun.example.com -DeviceId pi-01 -Command reload_gallery -AdminToken your-admin-token
```

## 人员统计

`/api/people` 只统计已知人员，不统计 Unknown。

情绪占比现在按出现次数计算：

```text
emotion percent = emotion event count / total known recognition events for that person * 100
```

例如 yuqin 一共出现 10 次，其中 Happy 出现 3 次，则 Happy 为 30%。
