# v0.3 本地验证记录

日期：2026-10-27，Windows / Python 3.12.2，默认 CPU。用户已批准方案 A（INT8 加速与单帧局部映射）。设备端 JetPack 版本、设备连接、现场传感器标定和代表性校准图像尚未提供。

| 检查 | 实际结果 |
| --- | --- |
| 自动测试 | 90 项通过，包含一次小型 ONNX 网络的真实 Q/DQ 量化与 CPU 执行；1 条现有 Starlette/httpx 弃用警告 |
| 依赖、语法 | `pip check`、Python 编译、浏览器脚本语法检查通过 |
| 完整 YOLOv5 FP32 ONNX | 实际导出成功，固定 640×640、batch 1，两个分割输出 |
| 完整 YOLOv5 Q/DQ 量化 | 用官方 2 张通用示例图片进行流程测试，106 个 QuantizeLinear 节点；保留最终分割 head 的浮点计算 |
| 量化 ONNX 运行 | ONNX Runtime CPU 实际执行成功；输出形状 `[1,25200,117]`、`[1,32,160,160]`；所有输出有限 |
| 原有分割 HTTP | 200，通用模型对 bus.jpg 返回 4 个实例 |
| 新映射 HTTP | 200，同样返回 4 个实例；纳秒字符串 `1792735200123456789` 精确保留；没有标定时映射 unavailable、地面坐标 null |
| 真实模型 + 模拟几何 | 使用明确标识 synthetic 的模拟标定，车体原点投影为 `[405,540]`；地面估计返回；未当作真实机器标定 |
| 基准脚本 | 本机 CPU 网络/流程计时可运行；同一模型与自身比较的匹配掩膜 IoU=1；独立 HTTP 计时可运行 |
| TensorRT 平台检查 | 在 Windows 上构建被明确拒绝，没有生成虚假的 Jetson 引擎 |
| 中文页面 | 实际浏览器上传并请求新接口；显示标定缺失原因和精确时间戳；无控制台错误 |
| 投影可视化与手机布局 | 回放“真实模型 + 模拟几何”响应检查蓝色投影点；390px 宽度无页面横向溢出，表格可局部滚动 |

完整依赖见 `requirements-lock-windows-py312.txt`。本机量化工具为 ONNX 1.23.2、ONNX Runtime 1.31.1，不能据此推断 Jetson 的可用版本。生成的测试模型放在 `artifacts/`，不打包为正式草坪模型。

本机证据文件：

- `artifacts/int8-onnx-verification.json`、`onnx-export.log`、`quantization-smoke.log`。
- `artifacts/mapping-http-response.json`、`mapping-synthetic-response.json`。
- `artifacts/benchmark-cpu.json`、`benchmark-regression.json`。
- `output/playwright/mapping-desktop.png`、`mapping-synthetic-desktop.png`、`mapping-mobile.png`。

部署包保留实际 HTTP 响应样例 `examples/mapping-response-uncalibrated.json`。其中海拔和雷达输入来自标明用途的示例 JSON，并非实测传感器数据。

边界：未在 Jetson 构建或运行 FP16/INT8 引擎，未测得 Orin Nano Super FPS；官方 2 张图片仅测试校准代码，不满足草坪任务校准覆盖；未验证杂草精度、实际距离、定位或机械控制。现场验收按 [使用说明](int8-and-mapping.md) 执行。

验证后本机服务保留在 `http://127.0.0.1:8000`，中文页为 `/docs`；进程结束后按 README 重启。当前没有启用模拟标定。
