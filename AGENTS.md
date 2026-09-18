# AGENTS.md

## 仓库概况

TeleStyle 是基于 Qwen Image Edit 的 ERP 全景风格迁移研究仓库。当前生图入口仅保留无 A1 的南北双 chart 流程；`training/` 中仍包含 A1 SphereAdapter 的数据准备、训练和评估代码。完整推理与训练依赖 Python 3.11、CUDA、大模型权重及 `requirements.txt` 中固定版本的依赖。

## 当前生图约定

- 输入内容必须是严格 2:1 的 ERP，至少为 `32×16`。
- 每次运行按文件名顺序遍历 `inputs/style/` 或 `--style-dir` 指定目录中的全部支持图片。
- 每张风格图只输出一张 `{style_stem}_erp.png`；不得恢复 chart、中间结果、baseline、对比图或报告输出。
- 唯一流程是南北 stereographic chart 独立去噪、无 A1、赤道 RGB 硬拼、低频颜色匹配和赤道残差修复。

## 主要文件

- `telestylepanorama_inference.py`：唯一全景 CLI、风格目录遍历、参数验证和最终 ERP 保存。
- `telestyleimage_inference.py`：Qwen/LoRA 加载、南北独立 scheduler 与 latent 轨迹、chart 解码和 RGB 合成。
- `telestyle_spherical.py`：ERP 像素中心、单位球方向、stereographic chart 提取和重采样纯函数。
- `training/geometry.py`：A1 与当前 RGB 硬拼共用的 chart-to-ERP 重投影、颜色匹配和接缝残差修复。
- `training/`：A1 数据缓存、条件、损失、SphereAdapter、Qwen 封装、训练与评估。
- `configs/`：A1 基础、pilot、pilot v2 和 TeleStyle pilot 配置。
- `data/a1/README.md`：A1 本地 ERP 数据目录与准备说明。
- `tests/test_panorama_inference.py`：当前批量入口和唯一生图路由的轻量回归。
- `tests/test_spherical_reprojection.py`：当前 chart 提取和重叠同步回归。
- `inputs/`：示例 ERP 与本地风格图。权重、模型缓存和生成结果均为本地数据，不要提交。

## 修改约束

- 修改保持聚焦；除非任务明确要求，否则保留现有 CLI 参数、默认值、权重名称和硬编码基础模型路径。路径变化需同步更新 `README.md`。
- 图像尺寸遵守 DiT/VAE 对齐规则；内部 ERP 和 chart 尺寸按 16 像素对齐，最终 ERP 恢复输入尺寸。
- 球面变换必须基于 ERP 像素中心、单位球方向和重采样；不得用 `torch.rot90` 代替球面旋转。
- 双分支必须保留各自 scheduler 和 latent 轨迹；不得用一个分支的逐步 latent 覆盖另一个分支。
- 最终合成保持南半球 yaw 为 `0°`，先以 2 倍采样重投影，再通过 area 抗锯齿缩小。
- 接缝优化仅作用于配置的赤道带；带外应保持原始 RGB 硬拼结果。
- 纯几何辅助函数应确定、可在 CPU 测试，测试中避免初始化重量级模型。
- 代码标识符、docstring 和 CLI help 使用英文；README、AGENTS 和设计文档使用中文。
- 不要覆盖用户已有修改，尤其是 `inputs/` 下的本地图片和未提交数据。

## 验证

只运行与改动相关的最小检查。基础语法检查：

```bash
python -m py_compile telestyleimage_inference.py telestylepanorama_inference.py telestyle_spherical.py training/*.py
```

当前全景入口、球面几何或接缝逻辑改动至少运行：

```bash
python -m unittest tests/test_panorama_inference.py tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py
```

A1 或共享推理逻辑改动按需运行对应 `tests/test_a1_*.py`、`tests/test_sphere_adapter.py` 和 `tests/test_telestyle_pilot.py`。当前维护中的 CPU 回归集合为：

```bash
python -m unittest tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py tests/test_sphere_adapter.py tests/test_a1_data.py tests/test_a1_training_dtype.py tests/test_a1_training_loop.py tests/test_a1_comparison.py tests/test_telestyle_pilot.py tests/test_panorama_inference.py
```

除非权重、GPU 和任务范围都合适，否则不要运行完整模型推理；只通过 CPU 测试时不要宣称实际生成质量已验证。

只改文档时核对链接、命令和源码名称即可。交付前运行 `git diff --check` 和 `git status --short`。
