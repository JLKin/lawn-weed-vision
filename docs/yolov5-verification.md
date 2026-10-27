# YOLOv5 分割版本机验证记录

验证日期：2026-10-24（Asia/Shanghai）。项目位置：`D:\世界院校职业技能大赛\模型部署\lawn-weed-vision`；当前已安装的虚拟环境在上一级 `.venv`。

## 已验证的内容

| 检查 | 实际结果 |
| --- | --- |
| 自动测试 | 58 项通过；覆盖 API 校验、错误响应、掩膜几何、YOLOv5 输出结构与引擎类别元数据 |
| 依赖完整性 | `pip check` 通过 |
| Python 编译与启动脚本 | Python 文件编译通过；Jetson `run.sh` 的 Bash 语法检查通过 |
| 官方源码及权重准备 | 固定提交和权重 SHA-256 校验通过 |
| 真实模型进程内测试 | 4 个实例，类别为 `bus`、`person`，单次 `inference_ms=178.395` |
| 实际 HTTP 服务 | `GET /health` 返回 200、`model_ready=true` |
| 示例客户端上传 | `POST /v1/weed-segmentation` 成功，4 个实例，单次 `inference_ms=209.597` |
| 时间戳与坐标 | `1792735200123456789` 精确回传；掩膜非空；所有 `ground_position_m=null` |
| 测试页面 | `/docs` 返回中文页面并标识 YOLOv5；`/api-docs` 返回 200 |
| Windows 上误用 Jetson 脚本 | 环境检查报告未就绪；引擎导出明确拒绝在非 Jetson 平台执行 |

运行环境为 Windows、Python 3.12.2、PyTorch 2.14.1+cpu、Ultralytics 8.4.184，网络输入尺寸 640，默认设备 CPU。完整依赖快照见 `requirements-lock-windows-py312.txt`，不得将该 Windows 快照用于 Jetson。

## 实际模型与样例

- 原版 YOLOv5 官方源码提交：`af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce`。
- 权重：`models/yolov5n-seg.pt`，来自官方 v7.0 发布。
- 权重 SHA-256：`1ee00e5e14e3863898c5446c62734776759e29be08fd1c174774992f7aaa9985`。
- API 模型标识：`yolov5n-seg.pt@sha256:1ee00e5e14e3`。
- 输入：官方仓库 `data/images/bus.jpg`，图像尺寸 810 × 1080。
- 实际 HTTP 响应保存在 `examples/deployment-smoke-response.json`，可核对字段、真实轮廓与时间戳。

这些耗时是本机单次观测，不是稳定性能基准，也不能推算 Jetson FPS。模型为 COCO 通用预训练分割模型；检测到公交车和人只说明部署链路正常，没有验证草坪或杂草识别能力。

## 本机使用

本次验证后保留服务运行，打开 `http://127.0.0.1:8000/docs` 即可使用中文测试页。电脑重启或进程结束后，在项目目录执行：

```powershell
..\.venv\Scripts\python.exe -m uvicorn weed_vision.main:app --host 127.0.0.1 --port 8000 --workers 1
```

这条命令适用于当前电脑的父目录虚拟环境。新电脑按照 README 创建项目内 `.venv`，并使用 `.\.venv\Scripts\python.exe`。

## Jetson 验证边界

已准备 GPU 环境检查、依赖约束生成、模型下载校验、TensorRT FP16 导出及类别元数据校验、启动脚本和 systemd 示例。Jetson 尚未配置，未在目标设备运行这些流程，未生成 TensorRT 引擎，也未测量设备端性能或精度。下一步按 [Jetson 部署说明](jetson-migration.md) 在设备上验证。

交付源码 ZIP 不包含虚拟环境、YOLOv5 仓库或模型二进制。安装后运行 `python -m scripts.prepare_yolov5` 联网获取固定版本；当前电脑已准备好的 `vendor/yolov5` 和 `models` 可另外复制，离线准备的细节见部署说明。
