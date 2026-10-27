# INT8 加速与单帧局部映射使用说明

本版本新增 `POST /v1/scene-mapping`、显式 INT8 Q/DQ 量化、Jetson TensorRT 引擎构建与模型对比脚本。原有 `POST /v1/weed-segmentation` 和 `/health` 保持兼容。中文测试页仍在 `/docs`，勾选“同时进行传感器映射”后可提交图像和传感器 JSON。

当前代码可在 Windows 验证映射和 ONNX 量化；Jetson 系统版本、实际标定和现场校准图像尚未提供。未宣称目标设备引擎、运行速度或杂草识别精度已验证。

## 一、先确认 Jetson 环境

在设备上读取版本，不按 Windows 快照安装 GPU 库：

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack libnvinfer10 python3-libnvinfer
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

项目虚拟环境与 GPU 软件栈的安装步骤见 [Jetson 部署说明](jetson-migration.md)。保留已有 `.env`，按需要添加 `WEED_CALIBRATION_DIR=calibrations`、`WEED_MAX_SENSOR_SKEW_MS=50`。

新的 `scripts.build_tensorrt` 仅接受目标 Jetson 的 TensorRT 10.x。旧版环境请先报告版本并按 NVIDIA 的兼容关系处理，不要直接 pip 升级 TensorRT。显式 Q/DQ 量化图决定量化位置，引擎中可以存在浮点层。[NVIDIA 平台说明](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-jetson.html)

## 二、生成 INT8 校准模型

这一步可以在当前 Windows 电脑进行，把 ONNX 与同名 `.onnx.json` 一起复制到 Jetson。使用真实摄像头图片建立 `datasets/calibration`，覆盖光照、阴影、距离和运动条件；不要使用测试集作训练或调参。校准本身不需要标签，但后续任务精度评价需要另备有标注测试集。

在当前项目目录执行（本机虚拟环境位于项目上一级）：

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-quantization.txt
..\.venv\Scripts\python.exe -m scripts.export_yolov5_onnx --weights models/yolov5n-seg.pt --output models/yolov5n-seg.fp32.onnx
..\.venv\Scripts\python.exe -m scripts.quantize_yolov5 --onnx models/yolov5n-seg.fp32.onnx --output models/yolov5n-seg.int8.onnx --calibration-dir datasets/calibration --max-images 300
```

新电脑或 Jetson 若虚拟环境位于项目内，分别使用 `.\.venv\Scripts\python.exe` 或 `.venv/bin/python`。量化额外依赖只用于离线工具，不需要装进只运行 TensorRT 的服务环境。已在本机执行的版本是 ONNX 1.23.2、ONNX Runtime 1.31.1；这不是 Jetson 安装版本要求。

实现采用 ONNX Runtime 静态量化，Conv 使用对称有符号 INT8、权重逐通道量化，默认保留最终分割 head 的浮点计算。可通过重复 `--exclude-prefix` 增加敏感层排除规则。输出记录实际排除节点、量化节点数量、模型摘要、图片清单摘要、工具和预处理版本。[ONNX Runtime 量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

校准与推理使用同一个 RGB、letterbox、NCHW、除以 255 处理模块。空目录、损坏图像、校准期间变更的图片、重复量化或已有输出文件都会明确报错。当前实现每次重新校准，不复用旧校准缓存，因此更换权重/图片/尺寸后重新运行即可；选用新的输出文件名保留旧模型。未实现跨构建的 TensorRT timing cache。

当前仓库的 `yolov5n-seg.pt` 仍是 COCO 通用模型。换成自己训练的草坪/杂草 `best.pt` 后，必须重新导出、校准、构建、评估。

## 三、在目标 Jetson 构建 FP16 与 INT8 引擎

先停止 API 释放 GPU 内存。假定上一步的 ONNX 和两个 `.onnx.json` 已放到设备的 `models` 目录：

```bash
.venv/bin/python -m scripts.check_jetson --output artifacts/jetson-environment.json --write-constraints artifacts/jetson-constraints.txt
.venv/bin/python -m pip install -r deploy/jetson/requirements-export.txt -c artifacts/jetson-constraints.txt
.venv/bin/python -m scripts.build_tensorrt --onnx models/yolov5n-seg.fp32.onnx --precision fp16 --output models/yolov5n-seg.fp16.engine
.venv/bin/python -m scripts.build_tensorrt --onnx models/yolov5n-seg.int8.onnx --precision int8 --output models/yolov5n-seg.int8.engine
```

任何命令失败时先修复该步骤，不继续使用未验证产物。工作空间默认 2 GiB，可加 `--workspace-gib 1`，这不是总 GPU 内存上限。新脚本不会覆盖现有引擎，失败后再次运行应选新文件名或检查已生成的诊断文件。

每个引擎对应 `.engine.json` 和 `.layers.json`：前者用于运行时验证模型、类别、输入尺寸和摘要；后者来自 TensorRT 引擎检查器，记录实际层信息。INT8 构建必须包含 Q/DQ，且检查器需观察到 INT8 张量格式；随后加载与预热成功才打印完成。观察到 INT8 不表示所有层都是 INT8，也不表示整体更快。

在 `.env` 中切换：

```dotenv
WEED_BACKEND=yolov5
WEED_MODEL_PATH=models/yolov5n-seg.int8.engine
WEED_DEVICE=0
WEED_IMAGE_SIZE=640
WEED_HALF=true
```

若已有自定义 `WEED_MODEL_VERSION`，同步更新或移除以使用文件摘要。运行时按引擎实际输入类型准备数据；`WEED_HALF=true` 不会把 FP16 引擎变成 INT8。回切只需把路径换为经过验证的 `.fp16.engine` 并重启。

## 四、比较性能和输出

用同一批图片分别运行模型。第一份模型作为结果对照；下面 PyTorch 默认 FP32，TensorRT 使用引擎实际精度：

```bash
.venv/bin/python -m scripts.benchmark_models \
  --model models/yolov5n-seg.pt \
  --model models/yolov5n-seg.fp16.engine \
  --model models/yolov5n-seg.int8.engine \
  --images-dir datasets/validation-images --max-images 100 \
  --warmup 10 --repeats 10 --device 0 --output artifacts/precision-comparison.json
```

报告区分网络耗时与预处理/推理/掩膜后处理耗时，给出 p50/p95，并记录同类别实例的掩膜 IoU 匹配数量和未匹配数量。匹配采用最大总 IoU 分配，IoU ≥ 0.5 计为匹配；这些是浮点/量化输出一致性指标，不是带标注 mAP 或杂草识别准确率。无目标时不编造平均 IoU。

另一次测当前运行服务的端到端耗时，可增加 `--http-url http://127.0.0.1:8000`。它单独标识当前服务模型，不会自动切换服务到上述所有引擎。API 进程与离线基准共享 GPU 会产生争用，应分别测量并记录条件。PyTorch 峰值分配统计不含全部 TensorRT 或系统内存；在设备上同时用 `tegrastats` 记录功耗、温度、系统/GPU 内存和降频状态。

是否使用 INT8，要结合真实草坪标注集的 mask mAP、杂草召回、草坪误识别率与设备实测延迟确定。小网络和较重后处理可能使 FP16 更合适，不能预先承诺固定 FPS。[NVIDIA 性能测量方法](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

## 五、发送摄像头、雷达与海拔数据

URL：`POST http://127.0.0.1:8000/v1/scene-mapping`。multipart 字段为 `file`（JPEG/PNG/WebP）和 `context`（JSON 文本）。局域网请求把地址换成 Jetson 实际 IP，并按部署文档配置监听地址。

```powershell
..\.venv\Scripts\python.exe -m scripts.mapping_client --image vendor/yolov5/data/images/bus.jpg --context examples/mapping-context.json
```

`examples/mapping-context.json` 的数值全是联调样例，不是传感器采集记录；默认未选标定，避免产生伪物理坐标。实际请求应替换相机/雷达编号、数据和采集时间。新的映射接口所有时间戳采用带引号的十进制字符串，响应也保留字符串；旧分割接口继续使用原整数字段。

输入约定：

- `clock_id` 必须表示已对齐的同一个时钟域。不同设备都用纳秒不意味着已经同步；采集端需先对时。
- 雷达 `cartesian3d` 提供 `[x,y,z]` 米；`polar2d` 提供 `range_m` 和 `azimuth_rad`，后者仅保存观测并返回 `insufficient_geometry`，不默认补高度。
- 每帧最多 256 个雷达目标，ID 不可重复；context 最多 48 KiB。径向速度字段只是径向标量，当前不进行运动补偿。
- 海拔 `reference` 为 `msl`（海拔）、`ellipsoid`（椭球高）、`relative` 或 `unknown`。本版本只记录海拔及时间差，不把它当作像素深度。离地高度自动修正地面平面的功能待明确传感器安装和测量方向后接入。
- 相机图像需保持与标定一致的方向、分辨率和去畸变状态；非平凡 EXIF 旋转在新映射接口被拒绝。旧图像接口仍支持 EXIF 方向校正。

## 六、安装真实标定并理解返回值

将实际标定文件放入 `WEED_CALIBRATION_DIR`，文件名为 `<calibration_id>.json`。只允许引用目录内受控文件，不允许客户端传入路径。字段示例见 `examples/synthetic-calibration.json`；**该文件是模拟几何，只用于软件测试，不能作为机器的实际标定**，服务不会自动启用它。

- `T_base_from_camera`：把相机光学坐标转换到车体坐标。
- `T_base_from_radar`：把雷达坐标转换到车体坐标。
- `base_link`：x 前、y 左、z 上；相机光学坐标 x 右、y 下、z 前；单位米。
- `camera_matrix`、`distortion`：标准 OpenCV 针孔相机模型；鱼眼需先生成匹配的去畸变图像和新内参，当前不直接接受鱼眼模型。
- `ground_plane`：车体坐标下单位法向量与偏移，满足 `normal_base · point + offset_m = 0`。只有请求 `ground_plane_valid=true` 且地面条件与该配置一致时才用于估计。姿态、坡度变化后不能继续盲用平地标定。

响应保留原 `detections`（像素识别，原地面字段仍 null），另在 `mapping` 内提供：

| 字段 | 含义 |
| --- | --- |
| `status`、`reasons` | ready / partial / unavailable 及原因；缺传感器、失配或未标定不会伪造成功坐标 |
| `calibration_id`、`calibration_kind` | 标定版本标识及 measured/synthetic |
| `radar_time_offset_ms` | 雷达减相机时间；不同 clock_id 时为 null |
| `radar_targets[].radar_position_base_m` | 雷达反射点的车体坐标，不能直接当成杂草根部位置 |
| `radar_targets[].pixel_px` | 原始图像上的投影；`status` 标记后方、越界或可关联状态 |
| `candidate_detection_indices` | 从 0 开始的候选分割实例索引；真实掩膜孔洞不算关联，多个候选保留歧义 |
| `mapping.detections[].ground_position_m` | 图像参考点射线与有效地面平面的交点估计；无法计算时 null |
| `position_method`、`reference_pixel_px` | 位置的计算方式和实际采用的掩膜内像素；不是刀具落点保证 |
| `altitude` | 海拔原值、参考基准、误差字段与时间戳，不自动转换全球坐标 |

本次没有跨帧跟踪、全局建图、车辆定位、设备驱动或执行机构控制。要获得全局杂草地图，后续还需车辆位置/姿态及地图原点定义。

## 七、本地验证与复现

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m pip check
```

安装 `requirements-quantization.txt` 后才执行实际 ONNX 量化测试；未安装时对应测试明确跳过。几何测试采用有已知真值的模拟相机与雷达，覆盖时差、坐标轴、掩膜孔洞、错误标定及缺少几何数据。模拟测试不替代现场联合标定。
