# 完整目录上传记录

此仓库是原 lawn-weed-vision 目录的完整文件快照，包含源码、模型、ONNX、测试结果、截图、日志与缓存。原目录未修改。source-manifest.json 记录每个原文件的路径、大小及 SHA-256。

vendor/yolov5 的源码以普通文件保存；Git 不允许直接嵌套跟踪另一份 .git，其内部历史完整保存在 yolov5-git-metadata.zip。解压该 ZIP 到仓库根目录，即可恢复 vendor/yolov5/.git，再运行项目的 prepare_yolov5 校验脚本。请保留该项目自带的上游 LICENSE 和说明。

仓库使用私有可见性。原 .gitignore 保留；本次完整快照已强制纳入此前被忽略的文件，之后新产生的文件仍遵循忽略规则。模型/样例验证边界见项目文档，上传不代表完成了 Jetson 实机验证。

如需恢复空目录，可依据清单中的 directories 创建；Git 本身不保存空目录。
