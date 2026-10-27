# YOLOv5 分割版部署到 Jetson Orin Nano Super 8GB

当前已完成 Windows 上的原版 YOLOv5 分割适配与真实模型测试。用户已安装 Jetson 系统，实际版本和运行结果待提供；尚未在目标设备验证 CUDA/TensorRT、性能或精度。v0.3 新增 INT8 Q/DQ 与局部映射，详见 [新功能部署说明](int8-and-mapping.md)。本文原 FP16 路径继续可用。

## 1. 模型与部署顺序

默认使用官方 `yolov5n-seg.pt`（nano 实例分割模型、COCO 通用预训练权重）验证部署。它不是已训练好的草坪杂草模型；后续用自己的草坪/杂草分割数据微调得到 `best.pt` 后，再替换权重。[YOLOv5 官方仓库](https://github.com/ultralytics/yolov5#segmentation)

顺序为：准备 JetPack → 验证 GPU 环境 → 运行 YOLOv5 PyTorch API → 在目标 Jetson 构建 TensorRT FP16 引擎 → 比较输出 → 启动服务。

`yolov5n.pt` 是普通检测模型，不能提供本接口所需的掩膜。此项目会拒绝检测版权重，不会用矩形框伪造分割结果。

## 2. 先安装匹配的 JetPack、PyTorch 与 torchvision

为 Orin Nano Super 选择 NVIDIA 明确支持该设备的 JetPack 镜像，记录实际版本。JetPack 6.2.2 的官方组件示例是 CUDA 12.6、TensorRT 10.3、cuDNN 9.3；这不表示应盲目固定该版本，也不表示最新版已在本项目验证。[JetPack 下载版本表](https://developer.nvidia.com/embedded/jetpack-archive)、[6.2.2 组件说明](https://developer.nvidia.com/embedded/jetpack-sdk-622)

按照实际 JetPack/Python 版本安装 NVIDIA 提供的 GPU PyTorch，并配套安装可在 CUDA 上执行 NMS 的 torchvision。TensorRT 使用 JetPack 自带的 Python 绑定。具体版本对应关系以 [NVIDIA PyTorch 安装文档](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) 和 [兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) 为准。由于设备版本尚未提供，这里不硬编码某个 GPU wheel 下载地址。

把项目源码放到 Jetson，例如 `~/lawn-weed-vision`。以下命令在 Jetson 的终端执行：

```bash
cd ~/lawn-weed-vision
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv --system-site-packages .venv
.venv/bin/python -m scripts.check_jetson \
  --output artifacts/jetson-environment.json \
  --write-constraints artifacts/jetson-constraints.txt
```

`--system-site-packages` 用于访问已配置好的系统 GPU 库和 TensorRT。若 NVIDIA PyTorch 安装在系统 Python 不能访问的另一个虚拟环境，需要在这个项目环境中按匹配版本重新安装。

检查脚本验证 Linux ARM64/L4T、Python、CUDA 可用性、NumPy 与 Torch 转换、torchvision CUDA NMS、TensorRT 导入。**脚本返回非零退出码时，先修复环境，不继续安装或导出。** 成功后才会写入约束文件，把当前工作的 torch/torchvision/numpy 版本固定下来，避免后续 pip 安装覆盖 GPU 栈。

## 3. 安装应用并准备模型

```bash
.venv/bin/python -m pip install -r requirements.txt -c artifacts/jetson-constraints.txt
.venv/bin/python -m scripts.check_jetson --output artifacts/jetson-environment-after-install.json
.venv/bin/python -m scripts.prepare_yolov5
cp deploy/jetson/.env.example .env
```

已有 `.env` 时编辑它，不重复覆盖。准备脚本获取以下固定来源：

- 官方仓库提交：`af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce`。
- 官方 v7.0 发布的 `yolov5n-seg.pt`，下载后校验 SHA-256。
- 源码位于 `vendor/yolov5`，权重位于 `models/yolov5n-seg.pt`。

不要安装 Windows 的完整依赖快照，也不要复制 Windows 虚拟环境。约束冲突应通过选择兼容依赖解决，不应先移除 GPU 版本约束。准备脚本不修改已有的不同权重或不同版本源码。离线部署时可从当前 Windows 项目复制已准备好的源码目录（含其 Git 元数据）和模型目录；Python/CUDA/TensorRT 环境仍须在 Jetson 单独安装。

初始配置使用 PyTorch：

```dotenv
WEED_BACKEND=yolov5
WEED_YOLOV5_REPO=vendor/yolov5
WEED_MODEL_PATH=models/yolov5n-seg.pt
WEED_DEVICE=0
WEED_HALF=true
WEED_IMAGE_SIZE=640
WEED_CONFIDENCE=0.25
WEED_IOU_THRESHOLD=0.45
WEED_MAX_DETECTIONS=100
WEED_MAX_IMAGE_PIXELS=3000000
```

先启动服务：

```bash
bash deploy/jetson/run.sh
```

另一个终端执行：

```bash
curl --fail-with-body http://127.0.0.1:8000/health
.venv/bin/python scripts/client.py --image vendor/yolov5/data/images/bus.jpg --camera-id front_rgb
```

也可运行 `.venv/bin/python -m scripts.smoke_test`。这些操作只验证通用模型部署；不验证草坪杂草识别精度。

## 4. 在这台 Jetson 上导出 TensorRT

先停止 API 服务以释放模型内存，然后安装 ONNX 导出依赖：

```bash
.venv/bin/python -m pip install -r deploy/jetson/requirements-export.txt -c artifacts/jetson-constraints.txt
.venv/bin/python -m scripts.export_yolov5_engine \
  --weights models/yolov5n-seg.pt --image-size 640 --workspace-gib 2
```

脚本调用固定版本的原版 YOLOv5 导出器，设置 batch 1、固定输入、`half=True` 请求 FP16，并在导出后加载和预热引擎。这是原版 YOLOv5 的导出方式，不使用新版 YOLO 的 `quantize=16` 接口。官方 YOLOv5 的基础命令形式为 `python export.py --weights yolov5n-seg.pt --include engine --device 0 --half`；本项目封装额外保存并校验了类别元数据。[官方导出器](https://github.com/ultralytics/yolov5/blob/master/export.py)

输出是一对文件：

```text
models/yolov5n-seg.engine
models/yolov5n-seg.engine.json
```

两个文件必须一起保留。元数据包含类别顺序、输入尺寸、权重和引擎摘要、源码版本、导出环境。原版 YOLOv5 的引擎本身不能保证保留类别名称，本服务通过元数据校验避免误用 COCO 类别或把类别显示成无意义编号。

引擎已存在时脚本会停止，确实要覆盖可加 `--overwrite`。8GB 为系统和 GPU 共用内存；若导出内存不足，停止其他模型进程，并尝试 `--workspace-gib 1`。这是工作空间上限，不是完整内存预算。

**Windows 的引擎不可直接复制到 Jetson 使用。** TensorRT 引擎依赖平台、GPU 与运行时，必须在目标设备或严格匹配环境中构建；本项目导出脚本要求在 Jetson 上运行。[NVIDIA TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

切换 `.env`：

```dotenv
WEED_MODEL_PATH=models/yolov5n-seg.engine
WEED_MODEL_VERSION=yolov5n-seg-coco-trt-fp16
WEED_DEVICE=0
WEED_IMAGE_SIZE=640
```

其他 YOLOv5 参数保留，重新启动 API。以后换为自训 `best.pt`，需要重新导出对应引擎和元数据；不要复用通用模型的类别表。

先对同一批固定测试图比较 PyTorch/FP32、PyTorch/FP16 和 TensorRT 的类别、置信度、框、掩膜和面积，再测延迟分位数、端到端时间、内存与热降频。当前没有给出 Orin Nano Super 的实测 FPS。INT8 工具现已提供；正式校准与部署仍需代表性草坪数据和精度回归基线，不能把通用图片流程测试当作任务验收。

## 5. 局域网请求与开机启动

默认只监听 Jetson 本机。需要从 Windows 通过局域网调用时：

```bash
WEED_HTTP_HOST=0.0.0.0 bash deploy/jetson/run.sh
hostname -I
```

假设 Jetson 实际 IP 为 `192.168.1.50`，Windows 请求：

```powershell
curl.exe -X POST 'http://192.168.1.50:8000/v1/weed-segmentation' -F 'file=@D:/images/lawn.jpg;type=image/jpeg' -F 'camera_id=front_rgb'
```

中文测试页面为 `http://192.168.1.50:8000/docs`。用实际 IP 替换示例地址；此原型面向受信任局域网，未配置认证。

需要开机启动时，先编辑 `deploy/jetson/lawn-weed-vision.service.example` 的 User、WorkingDirectory 和 ExecStart，使用户名、项目路径与设备一致。默认只监听本机；按需要改为局域网监听。然后：

```bash
sudo cp deploy/jetson/lawn-weed-vision.service.example /etc/systemd/system/lawn-weed-vision.service
sudo systemctl daemon-reload
sudo systemctl enable --now lawn-weed-vision
systemctl status lawn-weed-vision
journalctl -u lawn-weed-vision -f
```

健康检查返回 503 时，查看日志并修复模型路径/依赖。进程存在不代表模型就绪；本服务保留诊断接口，systemd 的进程重启策略不会修复缺失的模型。

## 6. 草坪标签、标定和真实任务

建议先明确 `turfgrass`（保留草坪）和 `weed`（清除杂草），统一成簇/植株的实例标注粒度；背景不伪装成杂草实例。按地块、日期拆分训练与测试集，覆盖阴影、露水、遮挡、不同草高及运动模糊。训练使用原版 YOLOv5 的 `segment/train.py`，得到分割 `best.pt` 后再配置本服务。

`mask_polygon_px` 是可视化近似轮廓，可能连接不相连区域、丢失孔洞；面积和中心点来自真实二值掩膜。未做相机内外参、地面/深度和机械执行机构标定前，原分割接口的 `ground_position_m` 为 `null`；新映射接口的地面估计须满足标定与地面条件，详见新功能说明。不生成刀具或机械臂动作。
