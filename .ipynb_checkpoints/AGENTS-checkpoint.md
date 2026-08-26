# AGENTS.md

## 适用范围

本仓库包含 TeleStyle 图像、全景图和视频风格迁移的推理代码。它主要是研究与推理仓库：没有包构建系统，也没有完整的自动化测试套件。

## 仓库导航

- `telestyleimage_inference.py`：Qwen Image Edit 与 TeleStyle 图像 LoRA。其 `__main__` 块是一个硬编码的小型示例。
- `telestylepanorama_inference.py`：接缝感知 ERP 全景图命令行流程，导入图像脚本中的 `ImageStyleInference`。
- `telestyle_spherical.py`：无模型依赖的 ERP 球面重投影、X 轴旋转与纬度融合工具。
- `telestylevideo_inference.py`：视频推理入口及 CLI 参数解析。
- `telestylevideo_pipeline.py` 和 `telestylevideo_transformer.py`：视频模型内部实现。
- `inputs/` 与 `assets/`：已跟踪的示例和文档素材。
- `weights/`、`Wan2.1-T2V-1.3B-Diffusers/`、模型缓存与生成结果目录：本地运行数据；不要将大型产物提交到 Git。

## 环境与权重

- 目标环境为 Python 3.11 和支持 CUDA 的 PyTorch。图像与视频推理均依赖 CUDA 和大模型权重，不应预期它们能在仅 CPU 的 CI 环境中完整运行。
- 使用 `pip install -r requirements.txt` 安装依赖；DiffSynth 依赖固定到一个 Git commit。
- 图像推理的基础模型路径目前在 `ImageStyleInference._load_models()` 中与机器路径绑定。除非任务明确要求路径可配置化，否则保持该行为；任何路径变更都要同步更新 `README.md`。
- 权重文件名及预期目录结构需与 README 保持一致。除非明确要求，绝不提交权重、模型缓存或生成媒体文件。

## 修改规范

- 保持修改聚焦。除非任务涉及接口调整，否则保留现有公开 CLI 参数名称和默认值。
- 图像尺寸必须符合 DiT/VAE 对齐规则。全景脚本采用 16 像素图像对齐，并在接缝同步前将坐标转换至 latent 空间。
- 修改全景逻辑时，必须保持环形顺序：`[右边缘 | 中心全景 | 左边缘]`，并在调用模型前校验宽度、边距和融合范围。
- 球面重投影必须通过 ERP 像素中心、单位球坐标和重采样完成；禁止用 `torch.rot90` 替代球面旋转。双分支模式中 A/B 保留独立 scheduler 和 latent 轨迹，只在最后若干步将对齐后的 B 去噪预测融合进 A；不要在每一步将 A latent 回写给 B。仅解码最终 A 分支。
- 验证小型纯函数时，避免导入或初始化重量级模型代码；尽可能使辅助函数保持确定性。
- 代码标识符、docstring、CLI 帮助信息使用英文；README 和 AGENTS 文档使用中文。除非在澄清被修改代码，否则可以保留已有中文注释。

## 验证

修改后运行最轻量且相关的检查：

```bash
python -m py_compile telestyleimage_inference.py telestylepanorama_inference.py telestylevideo_inference.py telestyle_spherical.py
python -m unittest tests/test_spherical_reprojection.py
```

只改文档时，核对 Markdown 链接、命令、权重名称和路径是否与当前源码一致。修改全景辅助逻辑时，添加或运行 CPU 测试，确认环形画布顺序、球面映射、极区权重和输出尺寸。除非任务范围、权重和 GPU 显存均合适，否则不要运行完整模型推理。

交付前检查 `git diff --check` 与 `git status --short`。不要还原或覆盖预先存在的用户修改。
