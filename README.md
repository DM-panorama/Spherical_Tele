# TeleStyle 图片与 ERP 全景图风格迁移

## 当前生图方法

| 方法 | 入口 | 用途 |
| --- | --- | --- |
| 普通图片风格迁移 | `telestyleimage_inference.py` | 普通平面图片，脚本内使用硬编码示例路径 |
| Hemisphere 双半球（默认、推荐） | `telestylepanorama_inference.py --panorama-mode hemisphere` | 南北半球 chart 独立去噪、赤道重叠区逐步同步，最后合成一个 ERP latent |
| Legacy 环形画布 | `telestylepanorama_inference.py --panorama-mode legacy` | 通过 `[右边缘｜中心全景｜左边缘]` 环形画布同步左右接缝 |
| Legacy 旋转双分支极区融合 | legacy 加 `--enable-polar-fusion` | 使用旋转后的 B 分支辅助 A 分支改善极区结构 |

推荐优先使用 `hemisphere`。两种 ERP legacy 路径主要用于兼容和对比。

## 环境与模型

推荐使用 Python 3.11、支持 CUDA 的 PyTorch 和 NVIDIA GPU。安装依赖：

```bash
pip install -r requirements.txt
```

图像基础模型路径当前固定在 `ImageStyleInference._load_models()` 中：

```text
/root/autodl-tmp/Qwen-Image-Edit-2509
```

该目录需要包含 Qwen-Image-Edit 的 transformer、text encoder、VAE 和 processor。TeleStyle 权重放在项目根目录的 `weights/`：

```text
weights/
├── diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors
└── diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors
```

`weights/` 已被 Git 忽略，不要把模型权重提交到仓库。

## Hemisphere（双半球）方法（核心）

### 适用输入

内容图应为 ERP 等距柱状全景图，通常采用 `2:1` 宽高比，例如 `2048×1024`。代码要求输入至少为 `32×16` 像素。为了符合 DiT/VAE 网格，内部工作尺寸会对齐到 16 的倍数，最终结果仍恢复为原始输入分辨率。

风格图可以是普通图片，进入模型前会缩放为 `1024×1024`。

### 核心思路

直接把 ERP 当作普通平面图片生成容易出现三个问题：

- ERP 左右边缘在球面上实际相连，普通图片模型却会把它们视为两个边界。
- ERP 在南北极附近存在严重的经度拉伸，平面处理不容易维持正确结构。
- 分块生成后再拼 RGB，容易在赤道、左右接缝或极点留下可见拼接痕迹。

`hemisphere` 方法先通过 ERP 像素中心和单位球方向，把同一张全景图重投影为南、北两个方形立体投影 chart。每个 chart 默认越过赤道 `15°`，因此双方在赤道附近观察到同一片球面区域，可以在 latent 空间持续同步。

### 完整生图流程

1. **读取并对齐尺寸**：内容图和风格图转换为 RGB；ERP 目标宽高对齐到 16 的倍数。
2. **建立南北 chart**：以南极、北极为中心生成两个方形立体投影 chart。chart 的球面覆盖角为 `90° + overlap`，默认覆盖到对方半球 `15°`。
3. **准备两份推理状态**：北、南 chart 分别建立 edit condition、scheduler 和 latent 轨迹；南北分支使用相同的 `seed`。
4. **同步初始 latent**：通过球面方向映射，把两个 chart 的共同赤道带重投影到彼此坐标系，并按纬度置信权重融合。
5. **逐步去噪并同步**：每个 scheduler step 中，两条分支分别预测并更新自己的 latent；更新后只同步双方共同的赤道带，不把一条分支的整份 latent 覆盖给另一条分支。
6. **合成 ERP latent**：去噪结束后，把南北 chart 按 ERP 像素中心和单位球坐标重投影为一张 ERP latent，并在重叠纬度带平滑融合。
7. **增加球面 padding**：左右方向使用循环 padding；跨越南北极时使用“纬度反射 + 经度半周平移”，保持真实球面拓扑。
8. **只解码一次**：对带球面 padding 的最终 ERP latent 执行一次 VAE 解码，再裁掉 padding。默认流程不依赖生成后的 RGB 拼接。
9. **恢复原尺寸并保存**：如果内部对齐改变了尺寸，结果会缩放回输入 ERP 的原始宽高；输出目录会自动创建。

南北分支各自完成一次方形 chart 去噪，因此计算量和显存需求通常高于普通单图推理。

### 最简指令

`hemisphere` 是默认模式，下面的命令即可运行：

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_result.png
```

### 完整指令

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_result.png \
  --panorama-mode hemisphere \
  --prompt "Transfer the style of Figure 2 to the equirectangular panorama in Figure 1. Preserve the panorama geometry and seamless horizontal wrap-around continuity." \
  --seed 123 \
  --steps 4 \
  --hemisphere-size 1024 \
  --hemisphere-overlap-degrees 15 \
  --decode-padding-px 128
```

### Hemisphere 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--content` | 必填 | ERP 内容图路径 |
| `--style` | 必填 | 风格参考图路径 |
| `--output` | 必填 | 输出路径，父目录不存在时自动创建 |
| `--prompt` | 内置 ERP 提示词 | 编辑指令 |
| `--seed` | `123` | 南北分支共用的随机种子 |
| `--steps` | `4` | 去噪步数，必须大于 0 |
| `--panorama-mode` | `hemisphere` | 全景图方法，可选 `hemisphere` 或 `legacy` |
| `--hemisphere-size` | 输入 ERP 高度对齐到 16 | 单个方形 chart 的边长；显式设置时必须为正且能被 16 整除 |
| `--hemisphere-overlap-degrees` | `15` | 南北 chart 越过赤道的角度，必须严格位于 `0°–45°` 之间 |
| `--decode-padding-px` | `128` | 最终 ERP VAE 解码前使用的球面 padding；会限制到有效尺寸并向下对齐到 16 |

如果输入是 `2048×1024` ERP，省略 `--hemisphere-size` 时，chart 默认就是 `1024×1024`。输入高度较大时可以显式降低 chart 尺寸以节省计算量，但细节也可能减少。

`--enable-polar-fusion` 只允许配合 `--panorama-mode legacy`，不能用于 `hemisphere`。

### Hemisphere 问题

有问题的主要是 **完整生图流程** 里的第 6 步：**合成 ERP latent**：去噪结束后，把南北 chart 按 ERP 像素中心和单位球坐标重投影为一张 ERP latent，并在重叠纬度带平滑融合。

这导致了两个问题：

1. 这里把两张极坐标的 latent 做了重投影，合成一张 ERP latent ，但是 latent 不能像 RGB 图像一样非线性变换，拉伸、压缩、插值都会影响高频，所以生成的效果图物体结构（低频）很好，但是笔触细节（高频）很碎，抖动感明显，有锯齿形纹路

2. 这个方法本质上，两张半球的图是独立的，知识在 VAE 编码的时候有一点联系，这样的联系并补紧密，所以最终融合的时候，赤道处接缝不太理想，如果要保持锐利就对不齐，如果要融合就会偏软。

   此方法处理赤道接缝与环形画布处理左右接缝处比较，环形画布表现很好的原因之一是，它的左右是可以直接平移并拼接的，最终得到的是一整张图，而这个方法把球面分成两份，二者之间的联系就减弱了，所以接缝处不如环形画布那么自然

## 其他生图方法

### 普通图片风格迁移

普通图片入口是硬编码示例。运行前在 `telestyleimage_inference.py` 的 `__main__` 中修改：

```python
content_ref = "inputs/content.png"
style_ref = "inputs/style.jpg"
```

然后执行：

```bash
python telestyleimage_inference.py
```

默认使用 4 个推理步、随机种子 `123`，并把内容图短边对齐为 `1024`。结果保存到：

```text
qwen_style_output/<风格图文件名>_result.png
```

### Legacy 环形 ERP

Legacy 基础模式把内容图扩展为 `[右边缘 | 中心全景 | 左边缘]`，初始化 latent 后以及每次 scheduler 更新后同步重复的左右边缘区域，最后裁出中心 ERP 并恢复原分辨率。

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_legacy.png \
  --panorama-mode legacy \
  --margin-px 256 \
  --blend-px 96 \
  --steps 4 \
  --seed 123
```

`--margin-px` 必须大于 0，实际值会限制到 ERP 宽度的一半并对齐到 16；`--blend-px` 不能为负，也不能大于有效 margin。

### Legacy 旋转双分支极区融合

该模式为原始 ERP 建立 A 分支，并把 ERP 绕 X 轴做球面旋转后建立 B 分支。A/B 保持独立 scheduler 和 latent 轨迹；B 分支对齐后的去噪预测用于引导 A 的极区，最终只解码 A。

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_polar_fusion.png \
  --panorama-mode legacy \
  --enable-polar-fusion \
  --margin-px 256 \
  --blend-px 96 \
  --polar-rotation-degrees 90 \
  --polar-blend-start-degrees 45 \
  --polar-blend-end-degrees 75 \
  --polar-fusion-steps 2 \
  --polar-fusion-strength 1.0
```

其他可调参数可通过 `python telestylepanorama_inference.py --help` 查看，包括极区低通半径和最终细节限制参数。

## A1 SphereAdapter 纯球面几何训练

A1 只使用高质量、连续的 2:1 ERP，不加载 TeleStyle 或 Lightning LoRA，也不需要风格图和风格化真值。数据和缓存的本地目录为：

```text
data/a1/
├── erp/Structured3D/  # scene_xxxxx/<view_id>/rgb_*.png
├── manifests/    # 自动生成 train/validation/test 清单
├── cache/        # VAE latent 和固定文本条件缓存
└── stress/       # 程序生成的高频压力样本
```

放入 ERP 后先扫描数据并生成清单：

```bash
python -m training.prepare_a1_data --config configs/a1.yaml
```

CUDA 可用且磁盘空间满足估算后，构建四个确定性球面增强缓存：

```bash
python -m training.prepare_a1_data --config configs/a1.yaml --build-cache --variants 4
```

启动或恢复 20k-step A1 训练，并对检查点执行硬接缝消融评估：

```bash
python -m training.train_sphere_adapter --config configs/a1.yaml
python -m training.train_sphere_adapter --config configs/a1.yaml --resume outputs/a1/a1_step_002000.pt
python -m training.evaluate_a1 --config configs/a1.yaml --checkpoint outputs/a1/a1_step_020000.pt
```

为固定 validation 样本输出零门控基线、完整 Adapter 和关闭跨球面通信三组 ERP，
并生成横向三联图：

```bash
python -m training.evaluate_a1 \
    --config configs/a1_pilot_v2.yaml \
    --checkpoint outputs/a1_pilot_v2/a1_step_000075.pt \
    --max-samples 4 \
    --comparison-samples 4 \
    --save-comparison-images
```

该可选评估功能不改变原有内容图加风格图的全景生图命令或默认输出。

### TeleStyle + A1 单命令 A/B 验证

下面的命令在同一进程中固定内容图、风格图、prompt、seed、步数和半球尺寸，依次生成原版 TeleStyle 与加载 EMA SphereAdapter 的实验版本：

```bash
conda activate telestyle311
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

python telestylepanorama_inference.py \
    --content inputs/panorama.png \
    --style inputs/style.jpg \
    --output outputs/ab_test/telestyle_a1.png \
    --panorama-mode hemisphere \
    --hemisphere-size 1024 \
    --steps 4 \
    --seed 123 \
    --a1-config configs/a1_pilot_v2.yaml \
    --a1-checkpoint outputs/a1_pilot_v2/a1_step_000075.pt \
    --a1-ab-test \
    --save-a1-chart-images
```

A/B 模式一次生成五张图片：`telestyle_a1_rgb_hardcut_baseline.png`（原版 latent 合成 baseline）、`telestyle_no_a1_rgb_hardcut.png`（无 A1 的独立双半球 RGB 硬拼）、`telestyle_a1_rgb_hardcut.png`（A1 RGB 硬拼）、`telestyle_a1_rgb_hardcut_north_chart.png` 和 `telestyle_a1_rgb_hardcut_south_chart.png`。另外写出 `telestyle_a1_rgb_hardcut_report.json`，不再生成 comparison 拼图。南北 chart 是去噪后分别解码的 1024×1024 方形图。A1 最终图不再合成 ERP latent，而是把两张 decoded RGB chart 按原生球面坐标以 2 倍分辨率重投影到 ERP，在赤道硬拼后使用 area 抗锯齿缩小。最终 ERP 不对南半球额外施加 yaw；单独保存的 south chart 会旋转 180°，以便与 north chart 按相同观察方向比较。报告中的左右接缝误差与像素差只能辅助比较，风格强度、极区纹理和结构稳定性仍需查看原尺寸 ERP。

A1 训练时没有加载 TeleStyle/Lightning LoRA，因此这一组合属于实验性推理。checkpoint 中的 chart size 必须与 `--hemisphere-size` 一致；不传任何 A1 参数时，原有生图路径不变。

训练期间的进度条按累计缓存样本计数，并显示当前 step、阶段、loss、gate 和预计剩余时间；样本会跨 epoch 重复抽取，该计数不是去重图片数。

训练入口会在加载大模型前检查 CUDA、显存档位、模型文件、缓存清单和可用空间。检查点只包含 SphereAdapter、EMA、注入 gates、优化器状态及基础模型哈希，不会合并或写回基础 Qwen 模型。
