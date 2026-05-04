# ASDUN Platform Upgrade Plan

## 0. 本轮已开始落地的内容

本轮先从风险最低、后续依赖最多的部分开始：

- Platform Server 新增 `GET /api/config/public`。
- Platform Server 新增 `POST /api/telemetry` 和 `GET /api/telemetry`，用于 ESP32 温湿度等传感器数据。
- Platform Server 新增设备 token 校验能力，默认关闭，本地调试不受影响。
- Windows Cloud Server 上报平台时支持 `device_token`。
- Raspberry Pi 的 PlatformClient 上报平台时支持 `platform_device_token`。
- Dashboard 新增 Telemetry 区域，可以看到 ESP32 最近上报的数据。
- ESP32 对接文档改为推荐 `/api/telemetry`，并补充 token header 写法。
- 新增 Platform Server 云端部署模板：Docker Compose、systemd、Nginx、环境变量样例。
- 新增 `scripts/test_platform_telemetry.ps1`，用于模拟 ESP32 上传温湿度。
- 新增控制命令队列 API：`POST /api/commands`、`GET /api/commands/pending`、`POST /api/commands/{command_id}/result`。
- Dashboard 新增 Commands 区域，可以创建命令并查看执行状态。
- 新增 `scripts/test_platform_command_flow.ps1`，用于模拟完整命令下发和回传流程。
- ESP32 对接文档新增命令轮询和结果回传流程。
- 新增 `examples/esp32_platform_client/esp32_platform_client.ino`，给 ESP32 同学参考温湿度上传和命令轮询。
- Dashboard People 区域新增心情日报弹窗和导出功能，可按人员生成当天情绪占比总结。
- Raspberry Pi 的 PlatformClient 新增命令轮询执行能力，支持 `ping`、`status`、`reload_gallery`，并会把执行结果回传到平台。
- 新增 `scripts/post_platform_command.ps1`，用于创建真实平台命令并等待设备回传结果。
- Dashboard Commands 区域新增 Admin token 输入框和 `Ping`、`Status`、`Reload gallery` 常用命令按钮。

下一步建议继续做：让 ESP32 真机按示例接入轮询、设备注册页面、告警规则、React 前端重构。

## 1. 当前结论

当前项目建议采用路线一：

```text
Platform Server / 网站 / 数据库：部署到云平台长期运行
Windows/NVIDIA Cloud Server：继续放在本机，负责 GPU 推理
Raspberry Pi：继续通过 Tailscale 调用 Windows 推理服务
ESP32：后续接入云端平台
Web/App：只访问云端 Platform Server
```

这样可以先解决网站长期可访问和数据集中存储的问题，同时不需要马上购买云 GPU。

需要明确区分两个服务：

| 服务 | 当前端口 | 建议部署位置 | 职责 |
|---|---:|---|---|
| Platform Server | 9000 | 云服务器 | 网站、REST API、WebSocket、数据库、设备管理 |
| Cloud Server | 8000 | Windows/NVIDIA 本机 | 人脸识别、情绪识别、人员库、GPU 推理 |

迁移后，本机不再需要长期运行网站后端 `9000`，但仍然需要运行 Windows GPU 推理服务 `8000`，除非后续购买云 GPU。

## 2. 目标架构

```text
                         +----------------------+
                         |      Web / App       |
                         | React dashboard      |
                         +----------+-----------+
                                    |
                                    | HTTPS / WebSocket
                                    v
                         +----------------------+
                         |  Cloud Platform      |
                         |  REST API            |
                         |  WebSocket           |
                         |  Database            |
                         |  Device commands     |
                         +----+------------+----+
                              ^            ^
                 HTTP/MQTT    |            | HTTP
                              |            |
              +---------------+--+      +--+----------------+
              |      ESP32       |      | Raspberry Pi       |
              | telemetry/status |      | status/result/UI   |
              +------------------+      +---------+----------+
                                                   |
                                                   | HTTP over Tailscale
                                                   v
                                      +-----------------------+
                                      | Windows/NVIDIA        |
                                      | Cloud Server :8000    |
                                      | GPU inference         |
                                      +-----------------------+
```

## 3. 当前已完成能力

当前项目已经完成：

- Pi 到 Windows 推理服务的 Tailscale 链路。
- Windows Cloud Server 的 `/health`、`/analyze`、gallery 相关接口。
- Platform Server 的 REST API、WebSocket、SQLite 存储。
- Dashboard 显示设备状态、识别记录、人员统计、按人员筛选。
- Windows Cloud Server 将识别结果上报到 Platform Server。
- Unknown 识别结果默认不进入网页识别记录。
- 情绪占比已改为按出现次数统计。

尚未完成：

- ESP32 MQTT/telemetry 接入。
- 设备控制命令下发。
- 告警规则。
- 公网部署、HTTPS、鉴权。
- React 前端重构。
- Platform Server 云端数据库迁移。

## 4. 升级阶段

### 阶段 1：整理 Platform Server 为正式后端

目标：

- 保持现有 FastAPI 后端。
- 固定当前 API，不急着重写 Java。
- 为后续 React 前端和云端部署打基础。

需要保留并稳定的接口：

```text
GET  /health
GET  /api/snapshot
GET  /api/devices
GET  /api/people
GET  /api/status/latest
GET  /api/events/recognition
POST /api/status
POST /api/events/recognition
WebSocket /ws
```

建议新增或完善的接口：

```text
GET  /api/config/public
POST /api/devices/register
POST /api/telemetry
GET  /api/telemetry?device_id=esp32-01&limit=100
POST /api/commands
GET  /api/commands/pending?device_id=pi-01
POST /api/commands/{command_id}/result
GET  /api/alerts
POST /api/alerts/rules
```

完成标准：

```text
所有设备和前端都只依赖 Platform Server API，不直接访问其他设备。
```

### 阶段 2：加入设备身份认证

公网部署前必须做设备鉴权，否则任何人都可以伪造设备状态和识别记录。

建议每台设备配置：

```yaml
device_id: "pi-01"
device_token: "replace-with-random-token"
```

请求头建议：

```http
X-ASDUN-Device-Id: pi-01
X-ASDUN-Device-Token: replace-with-random-token
```

第一阶段可以先做简单 token 校验：

- Platform Server 保存设备 token。
- Pi、Windows、ESP32 上报时带 token。
- token 错误则拒绝写入。

后续再升级：

- 用户登录。
- Web 管理员权限。
- 控制指令权限。
- token 轮换。

完成标准：

```text
公网 API 不接受无 token 的设备写入请求。
```

### 阶段 3：云端部署 Platform Server

推荐先使用轻量云服务器：

```text
1 核 / 2 GB 内存即可起步
Ubuntu 22.04 或 24.04
Python 3.11+
Nginx
HTTPS 证书
```

部署服务：

```text
FastAPI Platform Server
SQLite 或 PostgreSQL
Nginx 反向代理
systemd 后台守护
```

建议公网域名：

```text
https://api.asdun.example.com
```

云端部署后，设备配置改为：

```yaml
platform_base_url: "https://api.asdun.example.com"
```

Windows Cloud Server 配置：

```yaml
platform:
  enabled: true
  base_url: "https://api.asdun.example.com"
  device_id: "asdun-cloud"
  role: "inference_server"
```

Pi 配置：

```yaml
platform_enabled: true
platform_base_url: "https://api.asdun.example.com"
platform_device_id: "pi-01"
```

注意：Pi 到 Windows 的推理链路仍然保持：

```yaml
cloud_server_urls:
  - "http://asdun-cloud:8000"
```

完成标准：

```text
即使本机不运行 Platform Server 9000，网页仍然可以通过公网域名访问。
Pi 和 Windows 可以向公网平台上报数据。
```

### 阶段 4：数据库升级准备

当前 SQLite 适合调试和初期部署。

如果后续需要长期运行，建议迁移到 PostgreSQL。

建议保留的数据表：

| 表名 | 用途 |
|---|---|
| devices | 设备注册信息 |
| device_status | 当前最新状态 |
| status_events | 状态历史 |
| recognition_events | 识别事件 |
| people | 人员资料，可后续独立出来 |
| telemetry | ESP32 传感器数据 |
| command_logs | 控制命令记录 |
| alerts | 告警事件 |
| alert_rules | 告警规则 |

短期可以继续 SQLite，但代码层面建议做一个清晰的 Store 层，避免 API 直接写 SQL。

完成标准：

```text
后续切换 SQLite/PostgreSQL 时，不需要重写 API 和前端。
```

### 阶段 5：React 前端重构

不建议现在立刻用 Java 重写后端。

更推荐：

```text
后端：继续 FastAPI
前端：React + TypeScript + Vite
实时：WebSocket
样式：Tailwind CSS 或普通 CSS modules
```

React 前端页面建议：

```text
Dashboard
  - 设备在线状态
  - Pi / Windows / ESP32 总览
  - 实时识别记录
  - 人员统计

People
  - 人员列表
  - 情绪占比
  - 历史识别记录

Devices
  - 设备注册
  - 状态详情
  - 最近心跳

Telemetry
  - ESP32 传感器曲线
  - 温湿度/光照/人体感应

Commands
  - 设备控制
  - 命令执行结果

Alerts
  - 告警列表
  - 告警规则配置
```

前端不应直接访问 Pi、ESP32、Windows。

前端只访问：

```text
https://api.asdun.example.com
```

完成标准：

```text
React 页面可以替代当前 platform_server/static/index.html，但不影响设备和后端 API。
```

### 阶段 6：设备控制设计

后续 Web 控制设备时，不建议网页直接连设备。

推荐模型：

```text
Web -> Platform Server -> command_logs -> Device -> command_result -> Platform Server -> Web
```

HTTP 轮询版本：

```text
POST /api/commands
GET  /api/commands/pending?device_id=pi-01
POST /api/commands/{command_id}/result
```

命令示例：

```json
{
  "device_id": "pi-01",
  "command": "reload_gallery",
  "request_id": "cmd-001"
}
```

执行结果：

```json
{
  "request_id": "cmd-001",
  "ok": true,
  "message": "local gallery reloaded"
}
```

当前 Pi 端已经支持的 HTTP 轮询命令：

```text
ping             连通性测试，返回 pong
status           返回当前推理模式和云端连接状态
reload_gallery   重新加载本地人脸库
reload_people    reload_gallery 的别名
set_mode         可检查当前模式；运行时切换 local/hybrid/cloud 暂不自动执行，需要改 config/app.yaml 后重启
```

MQTT 版本后续可升级为：

```text
asdun/device/pi-01/command
asdun/device/pi-01/command_result
```

完成标准：

```text
网页可以下发控制命令，设备执行后返回结果，平台可追踪命令历史。
```

### 阶段 7：ESP32 接入

ESP32 建议优先使用 MQTT。

Topic：

```text
asdun/device/esp32-01/status
asdun/device/esp32-01/telemetry
asdun/device/esp32-01/command
asdun/device/esp32-01/command_result
```

第一版也可以先用 HTTP：

```text
POST /api/status
POST /api/telemetry
```

ESP32 配置：

```text
device_id=esp32-01
wifi_ssid=...
wifi_password=...
platform_url=https://api.asdun.example.com
device_token=...
telemetry_interval_ms=1000
heartbeat_interval_ms=5000
```

完成标准：

```text
Dashboard 能实时看到 ESP32 在线状态和传感器数据变化。
```

## 5. 近期最推荐执行顺序

建议按下面顺序做：

```text
1. 整理 API 文档
2. 加设备 token
3. 保持 FastAPI 后端，暂不 Java 重写
4. 把 Platform Server 部署到云服务器
5. 修改 Pi 和 Windows 的 platform_base_url
6. 稳定运行一段时间
7. 再开始 React 前端
8. 再接 ESP32 / MQTT / 控制命令
```

不要一开始就同时做：

```text
云部署 + React + Java 后端 + MQTT + 设备控制
```

这样风险太大，也不方便排错。

## 6. 部署后的日常运行方式

云平台长期运行：

```text
Platform Server
Database
Dashboard
```

Windows 本机长期运行：

```text
Cloud Server :8000
```

Raspberry Pi 长期运行：

```text
本地识别 UI
Pi -> Windows Cloud Server
Pi -> Cloud Platform
```

如果不购买云 GPU，Windows 的 `8000` 仍然需要保持运行。

如果未来购买云 GPU，可以迁移为：

```text
Cloud Server :8000 也部署到 GPU 云服务器
Windows 本机不再需要运行推理服务
```

## 7. 风险点

### 7.1 公网安全

上线前必须避免裸接口：

```text
POST /api/status
POST /api/events/recognition
POST /api/commands
```

这些都需要鉴权。

### 7.2 数据丢失

SQLite 在云端早期可以使用，但要定期备份：

```text
platform_server/data/asdun_platform.sqlite
```

长期建议 PostgreSQL。

### 7.3 服务守护

云服务器上不要用普通终端手动运行。

建议：

```text
systemd
Docker Compose
Nginx
```

Windows 本机推理服务短期可以用 PowerShell，后续建议：

```text
Windows Task Scheduler
NSSM / WinSW Windows Service
```

## 8. 当前建议结论

当前最优路线：

```text
先把 Platform Server 云端化
继续使用 Windows 本机 GPU 做推理
保持 FastAPI 后端
后续使用 React 重写前端
最后再接 ESP32、控制命令、告警规则
```

这样项目会从“本地演示系统”逐步升级成“多设备云平台系统”，同时每一步都能单独验证，不会一次性把复杂度拉爆。
