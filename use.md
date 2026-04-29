# ASDUN 使用说明

本文档说明如何启动两种运行方式：

- 云端混合模式：树莓派负责摄像头、检测、跟踪、UI；Windows/NVIDIA 主机负责 GPU 人脸识别和情绪识别。
- 树莓派本地模式：树莓派独立完成摄像头、人脸识别和情绪检测，不依赖 Windows 云端服务。

默认设备命名：

```text
Raspberry Pi / Tailscale name: asdun
Windows GPU / Tailscale name: asdun-cloud
Raspberry Pi user: asdun
```

如果你的用户名或设备名不同，请替换命令中的 `asdun` 或 `asdun-cloud`。

## 1. 云端混合模式

云端混合模式是当前推荐模式。

### 1.0 Windows 启动平台状态页面

阶段 B 需要先启动 Platform Server，用来显示树莓派和 Windows 推理端的在线状态。

在 Windows PowerShell 中进入项目根目录：

```powershell
cd C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi
.\scripts\run_platform_server.ps1
```

浏览器打开：

```text
http://127.0.0.1:9000
```

然后再启动 Windows 云端推理服务和树莓派程序。正常情况下页面会看到：

```text
asdun@asdun
asdun-cloud
```

### 1.1 Windows 启动 GPU 云端服务

在 Windows PowerShell 中进入项目根目录：

```powershell
cd C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi
```

第一次配置环境或依赖变化后运行：

```powershell
.\scripts\run_cloud_server.ps1
```

平时环境已经配好时运行：

```powershell
.\scripts\run_cloud_server.ps1 -SkipInstall
```

这个窗口会一直占用，不要关闭。关闭窗口后云端服务会停止。

另开一个 PowerShell 测试 Windows 本机服务：

```powershell
curl.exe http://127.0.0.1:8000/health
```

正常结果应该包含：

```json
{
  "ok": true,
  "device": "cuda",
  "active_provider": "CUDAExecutionProvider"
}
```

查看云端人员库：

```powershell
curl.exe http://127.0.0.1:8000/gallery
```

查看云端人员库质量诊断：

```powershell
curl.exe http://127.0.0.1:8000/gallery/diagnostics
```

### 1.2 Windows 检查树莓派到云端链路

确认 Windows 云端服务已经启动后，在 Windows 项目根目录运行：

```powershell
.\scripts\check_pi_cloud_link.ps1 -PiHost asdun -PiUser asdun -CloudUrl http://asdun-cloud:8000
```

如果成功，会看到：

```text
[network] Pi-to-cloud health check passed.
```

如果失败，先分别检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

以及在树莓派上检查：

```bash
curl http://asdun-cloud:8000/health
curl http://100.120.250.49:8000/health
```

### 1.3 Windows 同步代码到树莓派

如果你在 Windows 修改了代码或配置，运行：

```powershell
cd C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi
.\scripts\sync_to_pi.ps1 -PiHost asdun -PiUser asdun -RemoteDir ~/asdun_pi
```

该脚本会同步树莓派端需要的源码、配置、脚本和轻量模型，不会删除树莓派上的运行数据。

### 1.4 树莓派配置云端混合模式

在树莓派上进入项目目录：

```bash
cd ~/asdun_pi
```

确认 `config/app.yaml` 中使用混合模式：

```yaml
inference_mode: "hybrid"
cloud_server_url: "http://asdun-cloud:8000"
cloud_server_urls:
  - "http://asdun-cloud:8000"
  - "http://100.120.250.49:8000"
max_emotion_faces: 0
cloud_apply_identity: true
cloud_apply_emotion: true
```

说明：

- `inference_mode: "hybrid"` 表示启用本地检测和云端识别。
- `max_emotion_faces: 0` 表示树莓派本地不跑情绪模型，情绪结果由 Windows 云端返回。
- `cloud_apply_identity: true` 表示将云端身份结果应用到 UI。
- `cloud_apply_emotion: true` 表示将云端情绪结果应用到 UI。

### 1.5 树莓派编译并运行

如果代码有更新，先编译：

```bash
cd ~/asdun_pi
cmake -S . -B build_rpi -DCMAKE_BUILD_TYPE=Release
cmake --build build_rpi -j4
```

运行：

```bash
./scripts/run_rpi_turbo.sh
```

启动时如果云端连接正常，会看到类似：

```text
[CloudClient] probing: http://asdun-cloud:8000/health
[CloudClient] selected server: http://asdun-cloud:8000
[CloudClient] enabled: http://asdun-cloud:8000/analyze
```

如果 `asdun-cloud` 名称偶尔超时，程序会继续尝试备用 Tailscale IP。

### 1.6 菜单操作

程序启动后会显示：

```text
ASDUN Main Menu
1) Enroll or update person
2) Live recognition + emotion
3) Delete person
0) Exit
```

录入或更新人员：

```text
输入 1
输入人员姓名
按提示看向摄像头
画面提示 ready 后按 s 采集
采集完成后自动上传到 Windows 云端 gallery
```

实时识别和情绪检测：

```text
输入 2
```

删除人员：

```text
输入 3
```

退出：

```text
输入 0
```

## 2. 树莓派纯本地模式

本地模式用于没有 Windows 云端服务、Tailscale 网络异常、或需要离线运行时。

### 2.1 修改树莓派配置

在树莓派上编辑：

```bash
cd ~/asdun_pi
nano config/app.yaml
```

将关键配置改为：

```yaml
inference_mode: "local"
max_emotion_faces: 1
cloud_apply_identity: false
cloud_apply_emotion: false
```

说明：

- `inference_mode: "local"` 表示不启用云端 `CloudClient`。
- `max_emotion_faces: 1` 表示树莓派本地启用情绪检测。
- `cloud_apply_identity` 和 `cloud_apply_emotion` 在本地模式下不会使用，但建议关闭，避免混淆。

如果希望本地情绪检测同时处理更多人脸，可以改为：

```yaml
max_emotion_faces: 2
```

但树莓派负载会更高，画面帧率可能下降。

### 2.2 树莓派编译并运行

如果只是改配置，通常不需要重新编译，直接运行：

```bash
cd ~/asdun_pi
./scripts/run_rpi_turbo.sh
```

如果代码更新过，再编译：

```bash
cd ~/asdun_pi
cmake -S . -B build_rpi -DCMAKE_BUILD_TYPE=Release
cmake --build build_rpi -j4
./scripts/run_rpi_turbo.sh
```

本地模式启动时不应该看到：

```text
[CloudClient] enabled
```

如果仍然看到 `CloudClient` 日志，说明 `inference_mode` 没有改成 `local`，或者配置文件没有同步到树莓派。

## 3. 常用模式切换

### 3.1 切回云端混合模式

在 `config/app.yaml` 中设置：

```yaml
inference_mode: "hybrid"
max_emotion_faces: 0
cloud_apply_identity: true
cloud_apply_emotion: true
```

然后运行：

```bash
cd ~/asdun_pi
./scripts/run_rpi_turbo.sh
```

Windows 云端服务需要保持运行：

```powershell
.\scripts\run_cloud_server.ps1 -SkipInstall
```

### 3.2 切回树莓派本地模式

在 `config/app.yaml` 中设置：

```yaml
inference_mode: "local"
max_emotion_faces: 1
cloud_apply_identity: false
cloud_apply_emotion: false
```

然后运行：

```bash
cd ~/asdun_pi
./scripts/run_rpi_turbo.sh
```

本地模式不需要启动 Windows 云端服务。

## 4. 推荐日常开发流程

### 4.1 Windows 修改代码并同步到树莓派

Windows PowerShell：

```powershell
cd C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi
.\scripts\sync_to_pi.ps1 -PiHost asdun -PiUser asdun -RemoteDir ~/asdun_pi
```

树莓派：

```bash
cd ~/asdun_pi
cmake --build build_rpi -j4
./scripts/run_rpi_turbo.sh
```

### 4.2 Windows 云端服务重启

在运行云端服务的 PowerShell 窗口按：

```text
Ctrl + C
```

重新启动：

```powershell
cd C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi
.\scripts\run_cloud_server.ps1 -SkipInstall
```

## 5. 常见问题

### 5.1 Windows 本机 `/health` 不通

检查云端服务是否已经启动：

```powershell
curl.exe http://127.0.0.1:8000/health
```

如果不通，重新启动：

```powershell
.\scripts\run_cloud_server.ps1 -SkipInstall
```

### 5.2 树莓派访问 `asdun-cloud` 超时

在树莓派上测试：

```bash
curl http://asdun-cloud:8000/health
curl http://100.120.250.49:8000/health
```

如果设备名不通但 IP 能通，说明 Tailscale MagicDNS 或名称解析不稳定，可以继续依赖 `cloud_server_urls` 中的备用 IP。

如果两者都不通，检查：

- Windows 云端服务是否运行。
- Windows 和树莓派是否都在 Tailscale 中显示 `Connected`。
- Windows 防火墙是否允许 8000 端口访问。

### 5.3 云端 gallery 是空的

Windows PowerShell：

```powershell
curl.exe http://127.0.0.1:8000/gallery
```

如果返回：

```json
{"gallery_count": 0}
```

需要在树莓派程序中选择 `1) Enroll or update person` 录入人员。

### 5.4 云端识别返回 Unknown

常见原因：

- 人脸没有正对摄像头。
- 上传给云端的人脸 crop 质量较低。
- 人员没有录入云端 gallery。
- `cloud_identity_min_confidence` 阈值过高。

可以先检查 gallery：

```powershell
curl.exe http://127.0.0.1:8000/gallery
curl.exe http://127.0.0.1:8000/gallery/diagnostics
```

### 5.5 本地模式帧率下降

本地模式下树莓派会同时做人脸识别和情绪检测，CPU 压力更高。可以降低：

```yaml
max_emotion_faces: 1
emotion_interval: 24
recognition_interval: 24
```

如果仍然卡顿，优先使用云端混合模式。

### 5.6 运行 `.sh` 脚本提示 `bash\r`

如果树莓派运行脚本时出现：

```text
env: 'bash\r': No such file or directory
```

说明脚本被 Windows CRLF 换行污染了。先在树莓派项目目录修复：

```bash
cd ~/asdun_pi
sed -i 's/\r$//' scripts/*.sh
chmod +x scripts/*.sh
```

然后重新运行：

```bash
./scripts/run_rpi_turbo.sh
```

仓库中的 `.gitattributes` 已设置 `.sh` 文件使用 LF 换行，后续同步前尽量保持这条规则。
