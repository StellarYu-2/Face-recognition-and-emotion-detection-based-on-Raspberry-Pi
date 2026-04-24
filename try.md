# 云端混合推理新方案

## 1. 目标

本方案用于替代“树莓派 4 全本地重模型推理”的路线，在不更换硬件的前提下，同时提升实时帧率、人脸识别准确率和情绪识别稳定性。

核心目标：

- 树莓派端保持实时画面流畅，UI 不等待云端结果。
- Windows/NVIDIA 主机负责高精度人脸识别和情绪识别。
- 当前树莓派本地版本保留为 fallback，不破坏已验证可运行链路。
- 后续所有云端改造工作以本文档为准。

## 2. 已确认决策

| 项目 | 决策 |
|---|---|
| 云端运行位置 | Windows 主机，使用 NVIDIA GPU |
| 树莓派职责 | 摄像头采集、本地轻量检测、跟踪、UI 显示、异步请求云端 |
| 云端职责 | 高精度 embedding 库、人脸识别、情绪识别 |
| 第一版模型策略 | 先复用当前情绪模型，链路跑通后再考虑更强情绪模型 |
| 本地方案 | 保留，不删除；作为断网或云端失败时的 fallback |
| 开发分支 | `feature/cloud-hybrid-inference` |
| 稳定备份 | `backup/local-pi-fast-stable` 和 `local-pi-fast-stable-v1` |

## 3. 总体架构

```text
USB Camera
    |
    v
Raspberry Pi 4 C++ Client
    - camera capture
    - lightweight face detection
    - track manager
    - UI rendering
    - async cloud client
    |
    | JPEG face crop + track_id + quality metadata
    v
Windows/NVIDIA Cloud Server
    - FastAPI HTTP service
    - InsightFace / ONNXRuntime GPU
    - high-quality face alignment
    - high-accuracy identity matching
    - emotion inference
    - cloud embedding gallery
    |
    | JSON result
    v
Raspberry Pi UI
    - update name/confidence/emotion by track_id
    - never block video display
```

## 4. 为什么采用混合架构

### 4.1 树莓派全本地的瓶颈

树莓派 4 没有专用 NPU，重模型只能依赖 CPU。SCRFD 640x640、关键点对齐、MobileFaceNet、情绪识别同时运行时，推理会抢占画面刷新资源。

全本地轻量模型可以提高 FPS，但代价是：

- 人脸角度变化时识别准确率下降。
- 情绪识别受裁剪质量影响明显。
- 多人场景更容易牺牲刷新率或准确率。

### 4.2 全云端的瓶颈

全云端如果上传整帧并等待返回，会让 UI 延迟受网络影响。网络抖动时，画面和框都会卡。

因此本方案不上传整帧，不同步等待结果，只上传人脸 crop，并异步更新标签。

### 4.3 混合方案优势

- 画面 FPS 由树莓派本地轻量链路保证。
- 准确率由 Windows/NVIDIA 高性能模型保证。
- 网络波动时，UI 仍能显示本地检测框和旧结果。
- 云端模型后续可替换，不影响树莓派主循环。

## 5. 树莓派端设计

### 5.1 保留模块

继续保留当前 C++ 工程中的模块：

- `CameraManager`
- `FaceDetector`
- `TrackManager`
- `Renderer`
- `EmbeddingStore`
- `EmotionRecognizer`
- `InferencePipeline`

这些模块保证本地模式仍然可用。

### 5.2 新增模块

计划新增：

```text
include/cloud/CloudClient.hpp
src/cloud/CloudClient.cpp
include/cloud/CloudTypes.hpp
src/cloud/CloudTypes.cpp
```

职责：

- 后台线程发送 HTTP 请求。
- 管理请求队列。
- 限制同一个 track 的请求频率。
- 丢弃过期云端结果。
- 将云端返回结果写入 track 状态。

### 5.3 不阻塞原则

树莓派主循环必须满足：

```text
camera -> local detect/track -> render
```

这条路径不能等待云端 HTTP。

云端请求必须是：

```text
best crop -> queue -> worker thread -> cloud -> result cache -> track update
```

如果云端超时，当前帧直接继续显示。

## 6. 云端服务设计

### 6.1 目录规划

```text
cloud_server/
  app.py
  requirements.txt
  config.yaml
  models/
  data/
    gallery/
    cloud_gallery.sqlite
  services/
    face_service.py
    emotion_service.py
    gallery_store.py
  README.md
```

### 6.2 推荐技术栈

第一版使用 Python：

- FastAPI：HTTP 服务。
- Uvicorn：服务启动。
- OpenCV：图像解码和预处理。
- NumPy：数组处理。
- ONNX Runtime GPU：NVIDIA GPU 推理。
- InsightFace：高精度人脸检测、关键点、ArcFace embedding。

云端 Windows 主机优先使用 conda/mamba 环境管理：

```text
cloud_server/environment.yml
env name: asdun-cloud
```

原因是后续会接入 GPU 推理依赖，conda 环境比直接使用系统 Python 或普通 venv 更容易隔离版本。脚本 `scripts/run_cloud_server.ps1` 默认优先使用 conda/mamba，如果找不到再退回 `.venv`。

启动时必须打印：

```text
available_providers = [...]
using_provider = CUDAExecutionProvider / CPUExecutionProvider
```

如果没有 `CUDAExecutionProvider`，先不继续做性能判断。

## 7. 云端 API 设计

### 7.1 健康检查

```http
GET /health
```

返回：

```json
{
  "ok": true,
  "device": "cuda",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "gallery_count": 2
}
```

### 7.2 分析单个人脸 crop

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

返回：

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
    "confidence": 91.8,
    "probs": {
      "Calm": 0.03,
      "Happy": 0.918,
      "Sad": 0.02,
      "Angry": 0.032
    }
  },
  "latency_ms": 38
}
```

失败返回：

```json
{
  "ok": false,
  "track_id": 3,
  "frame_id": 1024,
  "error": "no_face"
}
```

### 7.3 云端人员录入

第一版可以先不做复杂 Web 页面，提供 HTTP 接口：

```http
POST /gallery/enroll
```

字段：

```text
name: string
images: multiple jpeg files
```

返回：

```json
{
  "ok": true,
  "name": "yuqin",
  "samples": 8,
  "embedding_dim": 512
}
```

### 7.4 云端人员列表

```http
GET /gallery
```

返回：

```json
{
  "people": [
    {
      "name": "yuqin",
      "samples": 8
    }
  ]
}
```

## 8. 配置项设计

在 `config/app.yaml` 中新增：

```yaml
inference_mode: "hybrid"   # local / hybrid / cloud

cloud_server_url: "http://10.247.28.48:8000"
cloud_timeout_ms: 300
cloud_connect_timeout_ms: 100
cloud_min_interval_ms: 600
cloud_max_queue_size: 2
cloud_jpeg_quality: 85
cloud_crop_size: 256
cloud_require_known_track: false
cloud_fallback_local: true
cloud_debug: false
```

含义：

| 配置 | 作用 |
|---|---|
| `inference_mode` | 选择本地、混合、云端模式 |
| `cloud_server_url` | Windows 主机服务地址 |
| `cloud_timeout_ms` | 单次云端请求超时 |
| `cloud_min_interval_ms` | 同一个 track 最小上传间隔 |
| `cloud_max_queue_size` | 请求队列上限，防止积压 |
| `cloud_jpeg_quality` | crop JPEG 压缩质量 |
| `cloud_crop_size` | 上传前 resize 尺寸 |
| `cloud_fallback_local` | 云端失败时是否保留本地识别 |

如果 Windows 主机 IP 变化，正常只需要改 `cloud_server_url` 这一项。云端服务继续监听 `0.0.0.0:8000`，不用因为 IP 变化改服务端代码。

## 9. 融合策略

### 9.1 身份融合

云端结果优先级高于本地结果。

如果云端返回：

```text
known=true and confidence >= 80
```

则更新 track identity。

如果云端超时或失败：

- `hybrid` 模式保留本地结果。
- `cloud` 模式显示上一条未过期结果。

### 9.2 情绪融合

情绪结果不要求每帧更新。

推荐策略：

- 云端情绪置信度高于 `60%` 才更新。
- UI 显示使用平滑概率，避免表情抖动。
- 结果超过 `3s` 未更新，显示旧结果但降低可信度。

### 9.3 结果过期

云端结果必须校验：

```text
track_id 一致
frame_id 不落后太多
track 仍然存在
```

否则丢弃，避免把旧结果贴到新的人脸上。

## 10. 第一阶段开发计划

### 阶段 A：云端服务骨架

目标：

- Windows 上启动 FastAPI。
- `/health` 返回 GPU provider。
- `/analyze` 可以接收 JPEG 并返回假数据。

完成标准：

```text
curl /health 正常
Pi 可以连通 Windows 服务
```

当前实现：

```text
cloud_server/app.py
cloud_server/services/runtime.py
cloud_server/services/fake_analyzer.py
cloud_server/environment.yml
cloud_server/requirements.txt
scripts/run_cloud_server.ps1
include/cloud/CloudTypes.hpp
include/cloud/CloudClient.hpp
src/cloud/CloudClient.cpp
```

阶段 A 暂时不接真实模型，`/analyze` 只用于验证 HTTP 上传、人脸 crop 编码、track_id/frame_id 透传和树莓派异步客户端联调。

树莓派端编译云端客户端需要 libcurl 开发包：

```bash
sudo apt update
sudo apt install -y libcurl4-openssl-dev
```

当前 C++ 客户端以观察模式运行：

```yaml
cloud_apply_identity: false
cloud_apply_emotion: false
cloud_debug: true
```

原因是阶段 A 云端返回的是假数据，先只验证异步请求和结果回传，不覆盖本地 UI 结果。等阶段 B/C 接入真实云端模型后，再打开对应 apply 开关。

### 阶段 B：云端情绪识别

目标：

- 云端复用当前 emotion 模型。
- Pi 上传 crop。
- 云端返回四分类情绪。
- Pi UI 显示云端情绪。

完成标准：

```text
Happy / Angry / Calm 响应延迟可接受
UI 不明显卡顿
断开云端后本地模式仍可跑
```

当前实现：

```text
cloud_server/services/config.py
cloud_server/services/emotion_service.py
models/emotion-ferplus-8.onnx
```

云端 `/analyze` 现在会返回真实情绪结果，身份结果仍然沿用本地传来的假/回显信息。树莓派端已设置：

```yaml
max_emotion_faces: 0
cloud_apply_identity: false
cloud_apply_emotion: true
cloud_emotion_min_confidence: 58.0
cloud_emotion_min_gap: 0.14
```

这样树莓派不再本地跑情绪模型，释放 CPU；身份识别仍保留本地链路，避免阶段 B 引入过多变量。
云端情绪结果还会经过置信度和 top1/top2 差距过滤，低置信或模棱两可时不刷新 UI，保留上一条稳定情绪。

### 阶段 C：云端人脸识别

目标：

- 云端维护 embedding 库。
- 云端 enroll 人员。
- 云端返回 name/confidence。

完成标准：

```text
同一人多姿态识别稳定
未录入人员不误识别
Pi 端 FPS 不因云端请求明显下降
```

当前实现：

```text
cloud_server/services/gallery_store.py
cloud_server/services/identity_service.py
models/arcfaceresnet100-8.onnx
```

云端新增接口：

```http
GET /gallery
POST /gallery/enroll
POST /gallery/reload
```

树莓派录入流程仍然是主入口：用户在树莓派上按原来的姿态提示拍照，本地录入成功后，程序会把这些已保存的人脸图片批量上传到云端 `/gallery/enroll`，自动同步云端 gallery。

树莓派端已打开：

```yaml
cloud_apply_identity: true
cloud_identity_apply_unknown: false
cloud_identity_min_confidence: 55.0
```

含义：云端只在识别为“已知人员且置信度足够”时覆盖本地身份；云端返回 Unknown 不会清空本地身份，避免云端库为空或低质量 crop 时造成闪烁。

### 阶段 D：高精度模型替换

目标：

- 情绪模型从当前 FER+ 换为更适合真实表情的模型。
- 可选使用更强 ArcFace / InsightFace 模型。

完成标准：

```text
情绪准确率明显优于 Pi 本地
身份识别稳定性明显优于 Pi 本地
```

## 11. 测试方案

### 11.1 网络连通测试

Pi 执行：

```bash
curl http://WINDOWS_IP:8000/health
```

预期：

```json
{"ok": true}
```

### 11.2 延迟测试

记录：

```text
Pi encode cost
HTTP roundtrip cost
Cloud inference cost
Total latency
```

目标：

```text
局域网单次 analyze 总延迟 < 200ms
理想状态 < 80ms
```

### 11.3 帧率测试

测试三组：

```text
local mode
hybrid mode with cloud online
hybrid mode with cloud offline
```

要求：

```text
hybrid online 不应显著低于 local mode
cloud offline 不应卡死或崩溃
```

### 11.4 准确率测试

身份：

- 已录入人员正脸。
- 已录入人员轻微侧脸。
- 未录入人员。
- 两人同时入镜。

情绪：

- 10 秒 Calm。
- 10 秒 Happy。
- 10 秒 Angry。
- 10 秒 Sad 或低落表情。

## 12. 风险和应对

| 风险 | 应对 |
|---|---|
| 云端网络抖动 | 异步请求，旧结果保留，本地 fallback |
| 云端 GPU 环境复杂 | 先用 `/health` 验证 provider，再接模型 |
| crop 质量影响云端准确率 | Pi 上传稍大 crop，云端二次检测对齐 |
| 结果贴错人 | track_id + frame_id + 过期检查 |
| 云端 embedding 和本地 embedding 不兼容 | 云端单独维护高精度 embedding 库 |
| 情绪模型不够准 | 第一阶段复用，第二阶段替换更强模型 |

## 13. 暂不做的事情

第一版不做：

- 公网部署。
- 用户登录系统。
- Web 管理后台。
- 全帧视频流推送。
- 每帧云端推理。
- 删除本地推理链路。

这些会增加复杂度，不利于先把混合推理跑通。

## 14. 当前推荐执行顺序

1. 新增 `cloud_server/`。
2. Windows 主机跑通 FastAPI `/health`。
3. Windows 主机确认 NVIDIA GPU provider 可用。
4. Pi 端新增 `CloudClient`，先请求假数据。
5. 接入云端情绪识别。
6. 接入云端人员库和身份识别。
7. 做本地/云端结果融合。
8. 根据测试结果替换更强情绪模型。

## 15. 判断成功的标准

本方案完成后，应达到：

- Pi 端 UI 流畅度接近当前轻量本地版本。
- 身份识别准确率接近或超过当前 SCRFD + MobileFaceNet 本地版本。
- 情绪识别响应比低频本地版本更稳定。
- 云端断开时程序不崩溃，可退回本地模式。
- 后续可以通过替换云端模型继续提升准确率。
