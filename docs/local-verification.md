# 首次版本验证记录（历史）

本文保留 2026-10-23 的 YOLO26 首次部署及中文页面验证记录。当前默认模型已改为原版 YOLOv5 分割版；最新状态与实际结果请看 [YOLOv5 验证记录](yolov5-verification.md)。下文版本、耗时和测试数量仅对应当时版本，依赖快照已更新到当前版本。

验证日期：2026-10-23（Asia/Shanghai）。

项目位置：`D:\世界院校职业技能大赛\模型部署\lawn-weed-vision`。
已安装的独立环境：`D:\世界院校职业技能大赛\模型部署\.venv`。

当前电脑可直接启动：

```powershell
Set-Location -LiteralPath 'D:\世界院校职业技能大赛\模型部署\lawn-weed-vision'
..\.venv\Scripts\python.exe -m uvicorn weed_vision.main:app --host 127.0.0.1 --port 8000 --workers 1
```

浏览器打开 `http://127.0.0.1:8000/docs`，在中文测试页中填写相机编号、选择图片并点击“开始识别”。Swagger 接口文档已移至 `/api-docs`。首次交付时测试使用的临时 HTTP 服务已停止；本记录下方的测试数据来自首次交付。

## 已完成的检查

| 检查 | 实际结果 |
| --- | --- |
| `python -m pytest -q` | 45 项通过：40 项 API 测试，5 项掩膜几何测试 |
| `python -m pip check` | 没有损坏的依赖关系 |
| 真实模型进程内 API 冒烟测试 | HTTP 200，5 个实例，类别 `bus`、`person` |
| 临时 Uvicorn HTTP 服务 | `/health` 与 `/openapi.json` 均返回 200 |
| `scripts/client.py` 上传真实图片 | 退出码 0，返回非空实例；`1792735200123456789` 时间戳精确回传 |
| `curl.exe` 健康检查 | 退出码 0，`model_ready: true` |

运行环境为 Windows / Python 3.12，默认 `WEED_DEVICE=cpu`，网络输入尺寸 640。已验证 Ultralytics 8.4.184、PyTorch 2.14.1、FastAPI 0.161.1；完整版本见 `requirements-lock-windows-py312.txt`。

使用的权重为官方通用 `yolo26n-seg.pt`，响应中的版本为 `yolo26n-seg.pt@sha256:361fbfabab28`。示例输入为 Ultralytics 安装包自带 `bus.jpg`，输出图像尺寸为 810 × 1080。进程内单次推理测得 `inference_ms=62.274`，另一次 HTTP 客户端请求为 `77.238`；这些是本机单次观测，不是稳定性能基准，也不是 Jetson 性能预测。

测试时 Starlette 对测试客户端使用 httpx 发出一条弃用提示；本次 45 项测试均通过。未来升级测试依赖时需关注其 httpx2 迁移说明。

## 尚未完成的业务验证

没有训练或验证草坪/杂草专用模型，没有测量杂草识别精度，也未在 Jetson/TensorRT 硬件上执行。未实施相机标定或执行机构控制，所有实例的 `ground_position_m` 均为 `null`。

源代码压缩包不包含虚拟环境和模型权重。当前本机已缓存默认权重；在新环境首次运行需要下载，或按 README 配置已有的兼容本地分割权重。

## 中文测试页更新验证

2026-10-23：`/docs` 已改为中文测试页，原始 Swagger 文档位于 `/api-docs`。已重启本项目原先运行在 8000 端口的服务；本次启动日志位于 `artifacts/server-zh.stdout.log` 和 `artifacts/server-zh.stderr.log`。

- 原有 45 项自动化测试通过，前端 JavaScript 语法检查通过。
- 在 Edge 浏览器完成真实图片上传，页面显示 5 个实例、中文状态与结果表格。
- 时间戳 `1792735200123456789` 在页面精确展示；原始响应直接显示服务返回的文本，避免 JSON 数字转换丢失精度。
- 未选图片时显示中文提示；上传损坏图片时显示中文错误与请求编号。
- 检查桌面 1440px 和手机 390px 宽度，手机页面无整体横向溢出，结果表格可横向滚动。
- 浏览器截图保存在 `output/playwright/`，不包含在源码压缩包内。
