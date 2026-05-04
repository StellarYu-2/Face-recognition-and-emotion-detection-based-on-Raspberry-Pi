# Models Setup

模型权重和转换后的 `.bin/.param/.onnx` 文件不提交到 GitHub。请按需自行下载或转换，并放到本地 `models/` 目录。

当前项目默认从 `models/onnx_to_ncnn/` 读取以下 NCNN 模型文件：

- `face_detector.param`
- `face_detector.bin`
- `mobilefacenet.param`
- `mobilefacenet.bin`
- `emotion.param`
- `emotion.bin`

Windows GPU/InsightFace 路径通常还需要自行放入：

- `models/emotion-ferplus-8.onnx`
- `models/arcfaceresnet100-8.onnx`
- `models/buffalo_l/`

## 推荐做法

在树莓派上运行：

```bash
cd /home/asdun/asdun_pi
chmod +x scripts/convert_models.sh
./scripts/convert_models.sh
```

该脚本会把上述 3 个 ONNX 转换为程序默认读取的 NCNN 文件名。

## 注意事项

1. 需要 `pnnx` 工具。  
Debian 13 / Raspberry Pi OS 新版本通常不允许直接对系统 Python 做 `pip install --user`，推荐使用虚拟环境安装：

```bash
sudo apt install -y python3-venv python3-pip python3-full
python3 -m venv ~/.venvs/pnnx
source ~/.venvs/pnnx/bin/activate
pip install -U pip
pip install pnnx
```

2. 若你不想用 `pip`，也可以单独构建 `ncnn/tools/pnnx`。  
3. 当前代码默认假设：
- 检测模型输出为 `scores` 和 `boxes`
- 人脸识别 embedding 输出 blob 为 `fc1`
- 情绪分类输出类别顺序为：
  `0 Neutral, 1 Happy, 2 Sad, 3 Angry, 4 Surprise, 5 Fear, 6 Disgust`

如果你的模型 blob 名或类别顺序不同，需要同步修改 `config/app.yaml`。

## 如果找不到 pnnx

树莓派上执行：

```bash
chmod +x scripts/setup_pnnx_venv.sh
./scripts/setup_pnnx_venv.sh
ls -l ~/.venvs/pnnx/bin/pnnx
```
