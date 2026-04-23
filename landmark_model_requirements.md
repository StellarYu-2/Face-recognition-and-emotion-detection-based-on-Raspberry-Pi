# 关键点模型要求

当前代码已经接入了：

- `检测框 -> 5 点关键点 -> ArcFace 标准对齐 -> 人脸 embedding`

也就是说，你现在需要找的不是普通人脸检测模型，而是一个**5 点关键点回归模型**。

## 推荐规格

- 格式：`ONNX` 或已经转换好的 `NCNN`
- 输入：`112x112` 人脸裁剪图
- 输出：`10` 个浮点数
- 点位顺序：
  - `left_eye_x, left_eye_y`
  - `right_eye_x, right_eye_y`
  - `nose_x, nose_y`
  - `left_mouth_x, left_mouth_y`
  - `right_mouth_x, right_mouth_y`

## 当前代码支持的输出坐标模式

- `zero_one`
  - 输出坐标归一化到 `[0, 1]`
- `minus_one_one`
  - 输出坐标归一化到 `[-1, 1]`
- `pixel`
  - 输出坐标直接是输入图像像素坐标

通过 `config/app.yaml` 的 `landmark_coord_mode` 配置切换。

## 当前代码支持的输入预处理配置

在 `config/app.yaml` 中可以调：

- `landmark_color_order`
  - `rgb` 或 `bgr`
- `landmark_mean`
- `landmark_norm`

默认值适合很多 `[-1, 1]` 风格的人脸关键点模型：

- `landmark_color_order: "rgb"`
- `landmark_mean: 127.5`
- `landmark_norm: 0.0078125`

## 文件放置位置

默认配置路径：

- `./models/onnx_to_ncnn/face_landmark_5.param`
- `./models/onnx_to_ncnn/face_landmark_5.bin`

如果你的文件名不同，改 `config/app.yaml` 即可。

## 启用成功后的表现

程序启动时会看到：

- 启用成功：关键点对齐已启用
- 未启用：未找到关键点模型

识别日志里如果出现：

- `decision=skip_alignment`

说明关键点模型已经进入流程，但这一帧关键点估计失败了。

## 如果你找不到完全符合的模型

最容易兼容的是：

- 轻量级 `5-point face landmark regression`
- 输入 `112x112`
- 输出 `10 floats`

如果你只能找到 `68/98/106` 点模型，我也可以继续帮你把当前代码改成支持多点模型并从中抽取 5 点对齐。
