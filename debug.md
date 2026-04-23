# Windows 开发 + 树莓派调试说明

本文档用于你当前工作模式：在 Windows 写代码，在树莓派 `asdun (10.192.30.252)` 上编译与调试。

## 1. 基础约定

- Windows 本地项目路径：`C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi`
- 树莓派远端目录建议：`/home/pi/asdun_pi`
- 下文默认树莓派用户名是 `pi`，若你实际用户名不同，请替换。

## 2. 首次连接检查（Windows PowerShell）

```powershell
ssh pi@10.192.30.252 "hostname && uname -a"
```

若能看到主机名 `asdun` 和系统信息，说明 SSH 网络通路正常。

## 3. 树莓派安装依赖（详细）

你的当前输出说明基础依赖已经基本装好，状态是正常的。  
`libncnn-dev` 在 Debian 13 / Raspberry Pi 官方源里经常没有，这时用源码安装 `ncnn` 是标准做法。

### 3.1 基础依赖（你已经执行过）

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config gdb \
  libopencv-dev libsqlite3-dev libcamera-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

说明：
- 不要求现在就 `apt full-upgrade`，先保证项目可编译可运行。
- 你日志里 `libcamera` 已升级到 `0.7`，这是正常升级路径。

### 3.2 先尝试系统包（多数情况下会失败）

```bash
sudo apt install -y libncnn-dev
```

若提示 `Unable to locate package libncnn-dev`，继续执行下面的源码安装。

### 3.3 源码安装 NCNN（推荐）

```bash
cd ~
git clone --depth=1 https://github.com/Tencent/ncnn.git
cmake -S ncnn -B ncnn/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=ON
cmake --build ncnn/build -j$(nproc)
sudo cmake --install ncnn/build
sudo ldconfig
```

参数说明：
- `NCNN_VULKAN=OFF`：首版先用 CPU，稳定优先。
- `NCNN_OPENMP=ON`：让 ARM 多核并行更容易跑起来。
- 关闭 `tools/examples/tests`：缩短编译时间并减少依赖。

### 3.4 安装后验证（建议逐条执行）

```bash
ls -l /usr/local/lib/libncnn*
ls -l /usr/local/lib/cmake/ncnn
pkg-config --modversion opencv4
pkg-config --modversion gstreamer-1.0
pkg-config --modversion gstreamer-app-1.0
pkg-config --modversion gstreamer-video-1.0
pkg-config --modversion libcamera || true
```

预期：
- 前两条能看到 `libncnn.so` 和 `ncnnConfig.cmake`。
- `opencv4`、`gstreamer-*` 能返回版本号。

### 3.5 如果 CMake 仍找不到 ncnn

在项目构建时加 `CMAKE_PREFIX_PATH`：

```bash
cd /home/pi/asdun_pi
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=/usr/local
cmake --build build -j4
```

如果你是源码装在其它前缀（比如 `/opt/ncnn`），把路径替换成对应前缀。

### 3.6 如果你需要把 ONNX 模型转换成 NCNN

你当前项目已经使用 NCNN 推理，所以如果下载到的是 `.onnx` 模型，还需要 `pnnx` 工具。

推荐直接安装：

```bash
python3 -m pip install --user -U pnnx
```

确认工具存在：

```bash
command -v pnnx || ls -l ~/.local/bin/pnnx
ls -l ~/ncnn/build/tools/ncnnoptimize || true
```

如果你必须源码构建 `pnnx`，它通常需要单独处理，不一定会随着 `~/ncnn/build` 一起生成，所以不要再用 `~/ncnn/build/tools/pnnx` 这个路径判断。

当前项目已经附带转换脚本：

```bash
cd /home/pi/asdun_pi
chmod +x scripts/convert_models.sh
./scripts/convert_models.sh
```

该脚本会把：
- `models/version-RFB-320.onnx`
- `models/arcfaceresnet100-8.onnx`
- `models/emotion-ferplus-8.onnx`

转换为：
- `models/face_detector.param` / `models/face_detector.bin`
- `models/face_recognizer.param` / `models/face_recognizer.bin`
- `models/emotion.param` / `models/emotion.bin`

## 4. 代码同步（Windows -> 树莓派）

```powershell
$LOCAL = "C:\Users\Yu.WIN-H4MB5VM86KC.000\Desktop\asdun_pi"
$REMOTE = "pi@10.192.30.252:/home/pi/asdun_pi"

ssh pi@10.192.30.252 "mkdir -p /home/pi/asdun_pi"
scp -r "$LOCAL\*" $REMOTE
```

提示：如果文件较多，后续可改用 `rsync` 提升增量同步效率。

## 5. 树莓派构建

```bash
cd /home/pi/asdun_pi
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
```

可执行文件默认为：`/home/pi/asdun_pi/build/asdun_access`

## 6. 运行与参数

```bash
cd /home/pi/asdun_pi
./build/asdun_access
```

配置文件：`config/app.yaml`  
常用项：
- `camera_source`：`"0"` 或 `gst:<pipeline>`
- `detect_interval` / `emotion_interval`
- `match_threshold` / `sigmoid_tau`

## 7. gdb 本地调试（在树莓派终端）

```bash
cd /home/pi/asdun_pi
gdb ./build/asdun_access
```

常用命令：
- `run`
- `bt`（崩溃后打印调用栈）
- `break asdun::App::handleRecognition`
- `next`

## 8. gdbserver 远端调试（可选）

树莓派端：

```bash
cd /home/pi/asdun_pi
gdbserver :2345 ./build/asdun_access
```

Windows 端（WSL 或远端 gdb 客户端）连接 `10.192.30.252:2345` 即可。

## 9. 常见问题排查

- 相机打不开：确认 `camera_source`、`libcamera` 权限、是否被其它进程占用。
- 检测不到脸：检查 `face_cascade_path` 是否存在，或者降低 `min_face_area_ratio`。
- 误识别偏多：调低 `match_threshold` 或增加录入样本质量阈值。
- FPS 不够：提高 `detect_interval`、降低分辨率、降低 `emotion_interval` 频率。

## 10. 建议日常流程

1. Windows 修改代码。  
2. `scp` 到树莓派。  
3. 树莓派 `cmake --build`。  
4. 运行并观察日志/FPS。  
5. 必要时 `gdb` 抓栈并回传修复。
## 11.我是强无敌的阿斯顿1
111