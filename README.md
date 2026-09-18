# TeleStyle

TeleStyle 是基于 Qwen Image Edit 的 ERP 全景风格迁移研究仓库。当前对外生图入口只保留一种流程：南北双 stereographic chart 独立去噪、无 A1、重投影后在赤道 RGB 硬拼，并在赤道带执行低频颜色匹配与残差修复。

`training/` 中仍保留 A1 SphereAdapter 的数据准备、训练和评估研究代码，但 A1 不接入当前全景生图 CLI。

## 项目结构

```text
TeleStyle/
├── telestylepanorama_inference.py   # 批量全景生图 CLI
├── telestyleimage_inference.py      # 双 chart 模型推理与 RGB 合成
├── telestyle_spherical.py           # chart 提取、球面坐标和重采样
├── inputs/
│   ├── content.png                  # 示例 2:1 ERP
│   ├── panorama.png                 # 示例 2:1 ERP
│   └── style/                       # 批量风格参考图
├── weights/                         # 本地 LoRA 权重，不提交 Git
├── configs/                         # A1 与 pilot 配置
├── training/                        # A1 数据、模型、训练和评估代码
├── data/a1/                         # A1 本地数据目录及说明
└── tests/                           # CPU 几何、训练和轻量推理测试
```

模型缓存、权重、A1 数据缓存和生成结果均为本地文件，不应提交到仓库。

## 环境与权重

完整生图需要 Python 3.11、CUDA 和 `requirements.txt` 中固定版本的依赖：

```bash
pip install -r requirements.txt
```

当前代码使用以下固定路径和权重名称：

- 基础模型：`/root/autodl-tmp/Qwen-Image-Edit-2509`
- TeleStyle LoRA：`weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors`
- Lightning LoRA：`weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`

## 生图流程

1. 读取严格为 2:1 的 ERP 内容图，并按 DiT/VAE 约束计算 16 像素对齐尺寸。
2. 以 ERP 像素中心和单位球方向提取南、北两个方形 stereographic chart；默认各自越过赤道 `15°`。
3. 两个 chart 使用相同 seed，但保留各自的 scheduler 和 latent 轨迹，独立完成去噪。
4. 分别解码南北 chart，以 2 倍采样重投影回 ERP，并在赤道硬拼后做 area 抗锯齿缩小。
5. 在赤道局部匹配低频颜色并修复异常梯度；优化带外保持原硬拼结果。
6. 恢复输入分辨率，只保存最终 ERP。

当前入口不会生成或保存南北 chart、中间 latent、baseline、对比图或 JSON 报告。

## 批量生成

把所有风格参考图放入 `inputs/style/`，然后运行：

```bash
python telestylepanorama_inference.py --content inputs/content.png --style-dir inputs/style --output-dir qwen_style_output
```

`--style-dir` 支持 `.bmp`、`.jpeg`、`.jpg`、`.png`、`.tif`、`.tiff` 和 `.webp`。程序按文件名排序并依次生成；每张风格图只输出一张 PNG：

```text
qwen_style_output/
├── style_1_erp.png
├── style_2_erp.png
├── style_3_erp.png
└── style_4_erp.png
```

若多个输入文件拥有相同 stem，它们会对应同一个输出文件名，因此风格目录内应使用不同的文件主名。

### CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--content` | 必填 | 输入 2:1 ERP 路径 |
| `--style-dir` | `inputs/style` | 风格参考图目录 |
| `--output-dir` | `qwen_style_output` | 最终 ERP 输出目录 |
| `--prompt` | 内置 ERP 风格迁移提示词 | 编辑指令 |
| `--seed` | `123` | 所有风格图依次使用的随机种子 |
| `--steps` | `4` | 去噪步数，必须大于 0 |
| `--hemisphere-size` | ERP 高度对齐到 16 | 单张方形 chart 的边长 |
| `--hemisphere-overlap-degrees` | `15` | 南北 chart 越过赤道的角度，范围为 `0°–45°` |
| `--hemisphere-color-match-degrees` | `6` | 赤道低频颜色匹配半宽；`0` 关闭 |
| `--hemisphere-seam-residual-degrees` | `2` | 赤道残差修复半宽；`0` 关闭 |
| `--hemisphere-seam-residual-blur-degrees` | `0.5` | 残差沿经度环形平滑的角尺度 |

输入 ERP 至少为 `32×16`，且必须严格为 2:1。chart 和内部 ERP 尺寸按 16 像素对齐，最终图片恢复到输入尺寸。风格图进入模型前统一缩放为 `1024×1024`。

## A1 研究代码

A1 数据目录布局和准备说明见 [`data/a1/README.md`](data/a1/README.md)。常用入口：

```bash
python -m training.prepare_a1_data --config configs/a1.yaml
python -m training.prepare_a1_data --config configs/a1.yaml --build-cache --variants 4
python -m training.train_sphere_adapter --config configs/a1.yaml
python -m training.evaluate_a1 --config configs/a1.yaml --checkpoint <checkpoint-path>
```

配置还包括 `a1_pilot.yaml`、`a1_pilot_v2.yaml` 和 `a1_telestyle_pilot.yaml`。训练和评估依赖对应数据、缓存、模型权重与 GPU 环境。

## 验证

基础语法检查：

```bash
python -m py_compile telestyleimage_inference.py telestylepanorama_inference.py telestyle_spherical.py training/*.py
```

当前全景入口与球面几何的 CPU 回归：

```bash
python -m unittest tests/test_panorama_inference.py tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py
```

完整维护中的 CPU 回归集合：

```bash
python -m unittest tests/test_spherical_reprojection.py tests/test_a1_geometry_losses.py tests/test_sphere_adapter.py tests/test_a1_data.py tests/test_a1_training_dtype.py tests/test_a1_training_loop.py tests/test_a1_comparison.py tests/test_telestyle_pilot.py tests/test_panorama_inference.py
```

CPU 测试只验证几何、路由和训练组件，不代表实际 GPU 生图质量已经验证。
