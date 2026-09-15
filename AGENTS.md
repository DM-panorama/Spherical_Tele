# AGENTS.md

## 仓库概况

TeleStyle 是基于 Qwen Image Edit 的图像与 ERP 全景风格迁移研究仓库，同时包含 A1 SphereAdapter 的数据准备、训练、评估和推理代码。完整推理与训练依赖 Python 3.11、CUDA、大模型权重及 `requirements.txt` 中固定版本的依赖。

## 主要文件

- `telestyleimage_inference.py`：图像推理，以及全景流程共用的 `ImageStyleInference`。
- `telestylepanorama_inference.py`：全景 CLI；默认 `spherope` 完整 ERP 模式，另有 `hemisphere` 和 `legacy`；A1 checkpoint 仅用于显式 `hemisphere`。
- `telestyle_spherical.py`：ERP 旋转、stereographic chart、球面重投影和 latent 融合纯函数。
- `telestyle_spherope.py`：SpheRoPE 相位纯函数与临时 Qwen 位置编码适配。
- `training/`：A1 数据缓存、几何、损失、SphereAdapter、Qwen 封装、训练与评估。
- `configs/`：A1 基础、pilot 和 TeleStyle pilot 配置。
- `tests/`：CPU 单元测试与轻量 fake-model 推理测试。
- `inputs/`：示例输入。权重、模型缓存和生成结果均为本地数据，不要提交。

## 修改约束

- 修改保持聚焦；除非任务明确要求，否则保留现有 CLI 参数、默认值、权重名称和硬编码基础模型路径。路径变化需同步更新 `README.md`。
- 图像尺寸遵守 DiT/VAE 对齐规则；全景输入和 chart 尺寸按 16 像素对齐。
- `legacy` 环形画布顺序必须保持为 `[右边缘 | 中心全景 | 左边缘]`。
- 球面变换必须基于 ERP 像素中心、单位球方向和重采样；不得用 `torch.rot90` 代替球面旋转。
- 双分支推理保留各自 scheduler 和 latent 轨迹；不要无意中把一个分支 latent 逐步覆盖到另一个分支。
- 纯几何辅助函数应确定、可在 CPU 测试，测试中避免初始化重量级模型。
- 代码标识符、docstring 和 CLI help 使用英文；README、AGENTS 和设计文档使用中文。

## 验证

只运行与改动相关的最小检查。基础语法检查：

```bash
python -m py_compile telestyleimage_inference.py telestylepanorama_inference.py telestyle_spherical.py training/*.py
```

球面或全景几何改动至少运行：

```bash
python -m unittest tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py
```

SpheRoPE 改动还需运行 `python -m unittest tests.test_spherope tests.test_spherope_inference`；只验证 CPU 接入时不要宣称实际生成质量已验证。

A1 或共享推理逻辑改动按需运行对应 `tests/test_a1_*.py`、`tests/test_sphere_adapter.py` 和 `tests/test_telestyle_pilot.py`。当前维护中的 CPU 回归集合为：

```bash
python -m unittest tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py tests/test_sphere_adapter.py tests/test_a1_data.py tests/test_a1_training_dtype.py tests/test_a1_training_loop.py tests/test_a1_inference.py tests/test_a1_comparison.py tests/test_telestyle_pilot.py
```

`tests/test_atlas_inference.py` 与 `tests/test_hemisphere_rgb_inference.py` 仍引用已移除或改名的接口，修复前不属于通过基线。除非权重、GPU 和任务范围都合适，否则不要运行完整模型推理。

只改文档时核对链接、命令和源码名称即可。交付前运行 `git diff --check` 和 `git status --short`，不要覆盖已有用户修改。
