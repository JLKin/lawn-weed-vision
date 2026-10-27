# 智能除草机视觉原型：YOLOv5 分割版 → Jetson Orin Nano Super

**v0.3 新增**：显式 INT8 Q/DQ 校准、Jetson TensorRT 10.x 引擎构建、性能/掩膜一致性对比，以及 `POST /v1/scene-mapping` 单帧摄像头/雷达/海拔输入。中文测试页增加映射模式。使用命令、标定要求与验证边界见 [INT8 与传感器映射说明](docs/int8-and-mapping.md)。用户已安装 Jetson 系统，但实际版本与设备端验证结果尚未提供。

这是一个本地运行的 Python / FastAPI 实例分割服务，先在当前 Windows 开发电脑验证“上传图像 → 模型推理 → JSON 结果”，以后再把相同 API 迁移到 Jetson Orin Nano Super 8GB。

**当前默认模型为原版 YOLOv5 的 `yolov5n-seg.pt`，属于 COCO 通用预训练分割模型，只用于部署冒烟测试。本项目没有训练好的草坪/杂草专用模型。** 它可能输出 `person`、`car` 等 COCO 类别，也可能在草坪照片上返回空数组；不能把这些结果当作杂草识别效果。[YOLOv5 官方分割说明](https://github.com/ultralytics/yolov5#segmentation)

项目使用固定版本的原版 YOLOv5 源码加载模型、执行 NMS 和还原掩膜；`ultralytics` Python 包是该版本源码所需的共享工具依赖。普通 `yolov5n.pt` 检测权重和 `yolov5nu.pt` 不可用于本分割接口。设备端 GPU 软件栈、引擎和性能验证待完成。

原分割接口提供像素坐标且 `ground_position_m` 始终为 `null`。新映射接口仅在标定与当前地面条件有效时，在 `mapping.detections` 中提供地面坐标估计。不控制机械刀具或机械臂；真实除草前需要完成草坪/杂草标注与微调、独立现场测试、相机标定、执行机构坐标转换和安全控制。

## Windows 安装与启动

本次已在当前电脑准备原版 YOLOv5 源码、通用分割权重及依赖，虚拟环境位于项目上一级的 `.venv`。在项目目录执行 `..\.venv\Scripts\python.exe -m uvicorn weed_vision.main:app --host 127.0.0.1 --port 8000 --workers 1` 即可运行。最新结果见 [v0.3 验证记录](docs/v03-verification.md)，基础模型记录见 [YOLOv5 验证记录](docs/yolov5-verification.md)。下载压缩包或换电脑后，先安装 Git，再按以下步骤建立环境。

在 PowerShell 中进入项目根目录。当前电脑可使用 Python 3.12；Python 3.11 也可用于本项目。以下命令始终直接调用虚拟环境解释器，无需修改 PowerShell 执行策略：

```powershell
Set-Location -LiteralPath 'D:\世界院校职业技能大赛\模型部署\lawn-weed-vision'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.prepare_yolov5
Copy-Item -LiteralPath '.env.example' -Destination '.env'
.\.venv\Scripts\python.exe -m uvicorn weed_vision.main:app --host 127.0.0.1 --port 8000 --workers 1
```

已有 `.env` 时先保留配置再修改 `WEED_BACKEND=yolov5`、`WEED_MODEL_PATH=models/yolov5n-seg.pt` 和 `WEED_YOLOV5_REPO=vendor/yolov5`。准备脚本会联网下载固定版本的官方源码与通用权重并校验 SHA-256，不会覆盖不同的已有权重或源码。离线使用可复制已经准备好的 `vendor/yolov5` 和 `models` 目录。服务启动阶段要求模型已存在；只加载自己训练或来源可信的权重。

默认设备为 CPU，不要求 NVIDIA 显卡。依赖包首次安装可能较大；先确保网络、磁盘空间充足。模型无法加载时进程仍提供诊断接口：`/health` 返回 HTTP 503，推理接口返回结构化错误；修复路径或依赖后重启服务。配置有误则启动失败，并在终端显示校验原因。

在另一个 PowerShell 窗口检查：

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

就绪时 HTTP 200，例如：

```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "yolov5n-seg.pt@sha256:1ee00e5e14e3",
  "model_path": "yolov5n-seg.pt",
  "detail": null
}
```

`model_version` 由实际权重标识产生，上面仅说明格式，以实际响应为准。未就绪时为 `status: "unavailable"`、`model_ready: false`、HTTP 503。中文测试页面位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)，可查看模型状态、上传与预览图片、填写相机参数，并查看中文结果表格和错误提示；原始 JSON 保留接口字段名和纳秒时间戳精度。页面资源随项目本地提供，无需加载外部 CDN。Swagger 接口文档位于 [http://127.0.0.1:8000/api-docs](http://127.0.0.1:8000/api-docs)。停止服务按 `Ctrl+C`。

## 上传一张图像

接口：`POST /v1/weed-segmentation`，请求类型为 `multipart/form-data`。

| 字段 | 要求 |
| --- | --- |
| `file` | 必填；JPEG、PNG 或 WebP；声明的 MIME 必须与实际解码格式一致 |
| `camera_id` | 必填；1–64 个字符；首字符为英文字母或数字，其余允许英文字母、数字、`_ . : -` |
| `capture_timestamp_ns` | 可选；范围 `0` 到 `9223372036854775807` 的整数；原样回传；省略时返回 `null` |

时间戳由采集端提供，用于以后与传感器时间对齐；服务和示例客户端不会自动生成采集时间，也不推断时间戳的时钟来源。纳秒时间戳通常超过 JavaScript `Number` 的安全整数范围；浏览器等消费者应使用支持大整数的 JSON 解析方式，避免精度丢失。

PowerShell 使用 `curl.exe`，避免命中旧版 PowerShell 的 `curl` 别名。把图片路径替换成实际文件；下面指定 PNG 的 MIME，JPEG 改为 `image/jpeg`，WebP 改为 `image/webp`：

```powershell
curl.exe --fail-with-body -X POST 'http://127.0.0.1:8000/v1/weed-segmentation' -F 'file=@D:/images/lawn.png;type=image/png' -F 'camera_id=front_rgb'
```

Python 客户端会根据扩展名选择 MIME，检查 HTTP 状态并把失败响应以 JSON 输出到标准错误：

```powershell
.\.venv\Scripts\python.exe scripts/client.py --image 'D:\images\lawn.png' --camera-id front_rgb
.\.venv\Scripts\python.exe scripts/client.py --image 'D:\images\lawn.png' --camera-id front_rgb --capture-timestamp-ns 1792700000000000000 --url http://127.0.0.1:8000
```

第二条命令的时间戳仅演示参数格式，实际应使用相机采集时间。CPU 推理较慢时可增加客户端 `--timeout 180`。

以下成功结果仅用于说明字段和坐标形状，数值不是性能测量或杂草模型预测：

```json
{
  "request_id": "0123456789abcdef0123456789abcdef",
  "camera_id": "front_rgb",
  "capture_timestamp_ns": null,
  "model_version": "demo-version",
  "image_width": 640,
  "image_height": 480,
  "inference_ms": 123.4,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox_xyxy": [100.0, 80.0, 140.0, 160.0],
      "centroid_px": [119.5, 119.5],
      "area_px": 3200,
      "mask_polygon_px": [[100.0, 80.0], [139.0, 80.0], [139.0, 159.0], [100.0, 159.0]],
      "ground_position_m": null
    }
  ]
}
```

字段含义：

- `request_id`：关联该请求的响应与服务日志；错误也包含此字段。
- `class_name`：直接来自已加载模型的类别表。只有替换成经过微调的权重后，才可能输出 `weed` 等自定义类别；服务不把 COCO 类别重命名为杂草。
- `bbox_xyxy`：`[左, 上, 右, 下]`，像素坐标；`centroid_px` 为二值掩膜所有前景像素中心的平均坐标；`area_px` 为前景像素数量。
- `mask_polygon_px`：一个轮廓的点列表，来自 Ultralytics `masks.xy`；是可视化用近似，可能连接不相连区域且不保留孔洞。因此轮廓面积未必等于 `area_px`，不得直接作为刀具边界。
- `image_width`、`image_height` 和全部像素坐标以应用 EXIF 方向校正后的图像为准。YOLOv5 后端通过 `process_mask_native` 还原原图分辨率上的掩膜；面积和中心点也在该分辨率计算。
- `inference_ms`：服务在取得模型串行锁之后，测量模型 `predict` 加结果后处理所用时间；不包含等锁、文件上传、图像解码和网络传输。它不是端到端延迟或固定 FPS 承诺。
- `detections: []`：合法的“没有达到置信度阈值的分割实例”结果，不能据此认定区域安全或没有杂草。

## 校验与错误处理

默认最多上传 10 MiB 的图像文件，最多 12,000,000 个解码像素。整个 multipart 请求另留 64 KiB 元数据开销；超出请求总限制也会被拒绝。空文件、损坏图像、MIME 与实际格式不符、GIF 等不支持格式、无效相机标识、超范围时间戳均会被校验。精确 API 约束以 `/docs` 和源码为准。

| HTTP 状态 | 常见原因 |
| --- | --- |
| `400` | 空文件、解码失败或请求内容无效 |
| `413` | 上传大小或像素数超出限制 |
| `415` | 图像格式或 MIME 不支持/不匹配 |
| `422` | 缺少字段、`camera_id` 不合法或时间戳校验失败 |
| `503` | 模型未能加载，当前不能推理 |
| `500` | 推理或服务内部失败；使用 `request_id` 关联终端日志 |

错误结构示意：

```json
{
  "request_id": "...",
  "error": {
    "code": "validation_error",
    "message": "Invalid request fields.",
    "details": []
  }
}
```

具体 `code`、`message` 以实际响应为准，`details` 可省略。模型初始化失败的完整原因写入服务终端，API 仅返回经过处理的诊断信息。

## 配置与项目结构

在项目根目录运行时读取 `.env`；同名环境变量优先。改变配置后重启。相对模型路径按启动时的工作目录解析。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `WEED_BACKEND` | `yolov5` | 原版 YOLOv5 分割后端；旧版通用适配器可显式选 `ultralytics` |
| `WEED_YOLOV5_REPO` | `vendor/yolov5` | 准备脚本下载的固定版本源码目录 |
| `WEED_MODEL_PATH` | `models/yolov5n-seg.pt` | 原版 YOLOv5 分割权重；TensorRT 引擎须附带 `.engine.json` 元数据 |
| `WEED_MODEL_VERSION` | 自动生成 | 可选的模型发布版本；建议包括数据集/训练批次标识 |
| `WEED_DEVICE` | `cpu` | 本地 CPU；具有合适 CUDA 环境时可改为 `0` |
| `WEED_HALF` | `false` | YOLOv5 PyTorch 在 CUDA 上使用 FP16；CPU 使用 FP32，引擎以实际输入类型为准 |
| `WEED_CONFIDENCE` | `0.25` | 置信度阈值，`0 < 值 <= 1` |
| `WEED_IOU_THRESHOLD` | `0.45` | YOLOv5 NMS 的交并比阈值 |
| `WEED_IMAGE_SIZE` | `640` | 网络输入尺寸；32 的倍数，范围 32–2048 |
| `WEED_MAX_DETECTIONS` | `100` | 单张图像最多实例数，范围 1–300 |
| `WEED_MAX_UPLOAD_BYTES` | `10485760` | 文件字节数上限 |
| `WEED_MAX_IMAGE_PIXELS` | `12000000` | 图像解码像素数上限 |

```text
weed_vision/          配置、图像校验、响应类型、模型封装、FastAPI 路由
weed_vision/web/      中文测试页的 HTML、CSS、JavaScript
scripts/client.py    单张图像上传客户端
scripts/prepare_yolov5.py  下载固定版本源码与通用分割权重
scripts/check_jetson.py   设备环境检查与 GPU 依赖约束生成
scripts/export_yolov5_engine.py  在 Jetson 导出引擎及类别元数据
deploy/jetson/       Jetson 配置、启动脚本和 systemd 模板
tests/               不依赖下载模型的 API 与推理后处理测试
docs/jetson-migration.md  草坪微调与 Jetson/TensorRT 迁移说明
.env.example         配置模板
requirements.txt     本地运行依赖范围
requirements-dev.txt 测试依赖
requirements-lock-windows-py312.txt 本机通过验证的完整依赖快照（含测试依赖）
pyproject.toml       Python 包与 pytest 配置
```

服务每个进程只加载一份模型，并将访问模型的预测串行化。当前建议 `--workers 1`；多 worker 会重复加载权重并占用更多内存，尤其不适合先在 8GB Jetson 上直接增加 worker。当前单图 API 不包含摄像头采集、追踪、传感器融合或自动执行控制。

## 测试与部署验收

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

自动化测试使用可控的模型替身验证接口、输入校验、后处理和引擎元数据校验，不证明预训练权重能识别草坪杂草。真实模型测试可启动服务，检查 `/health`，再用客户端上传图像。

也可使用随包脚本，在进程内通过完整 API 调用默认模型，并使用 Ultralytics 自带的公交车示例照片：

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

该脚本读取同一套 `WEED_*` 配置，预期通用 COCO 模型在示例图中产生实例；切换为草坪专用模型后应改用自己的验证图像和期望结果。输出保存到 `artifacts/smoke-response.json`。本次真实返回的样本见 [示例响应](examples/deployment-smoke-response.json)。该结果只表示软件链路已运行。

需要复现本次 Windows / Python 3.12 依赖时，在新建环境中使用 `python -m pip install -r requirements-lock-windows-py312.txt`。此快照包含测试依赖，不能直接用于 Jetson ARM64。常规安装仍可使用 `requirements.txt` 中的兼容版本范围。

进入业务训练前，建议首先明确两类标签 `turfgrass`（需保留草坪草）与 `weed`（需清除杂草）；有可靠标注时再细分杂草种类。泥土等背景不必伪装成杂草实例。草坪通常连成一片，若主要目标是可通行草坪区域，可以另设语义分割任务；本服务输出的是实例分割。按拍摄地块/日期拆分训练、验证和测试集，避免相邻视频帧泄漏；覆盖阴影、露水、遮挡、不同草高和运动模糊。障碍物、人员、宠物、工具及地形风险需要独立验证与停止逻辑。完整迁移步骤见 [Jetson/TensorRT 迁移说明](docs/jetson-migration.md)。

Ultralytics 组件与权重的发布/使用许可请查阅其 [官方许可说明](https://www.ultralytics.com/license)。
