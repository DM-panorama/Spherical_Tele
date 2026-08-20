# 基于TeleStyled 球面改造

## 摘要

内容保持风格迁移，即根据内容参考图和风格参考图生成风格化结果，由于扩散 Transformer（DiT）内部表征中内容与风格特征的天然耦合，始终是一项重要挑战。本技术报告提出 TeleStyle：一个轻量且有效的图像与视频风格化模型。TeleStyle 基于 Qwen-Image-Edit，利用基础模型强大的内容保持与风格定制能力。为实现有效训练，我们构建了高质量的特定风格数据集，并从数千种多样的真实世界风格类别中进一步合成三元组数据。我们引入课程式持续学习框架，在由干净（人工整理）和噪声（合成）三元组构成的混合数据集上训练 TeleStyle。该方法使模型能泛化到未见过的风格，同时维持精确的内容一致性。此外，我们还引入视频到视频的风格化模块，以提升时间一致性与视觉质量。TeleStyle 在风格相似度、内容一致性和美学质量三项核心评估指标上达到当前最优水平。

## 最新动态

- 已发布 TeleStyleV2-SenseNova：[代码](https://huggingface.co/spaces/witcherderivia/TeleStyle-SenseNova/tree/main)、[模型](https://huggingface.co/Tele-AI/TeleStyleV2)、[在线演示](https://huggingface.co/spaces/witcherderivia/TeleStyle-SenseNova)。它增强了 SenseNova U1 的内容保持风格迁移能力，同时保留通用图像编辑能力。该实验模型在像素空间训练，监督微调难度较高，仍有较大改进空间，支持 1MP 至 4MP 分辨率。
- 2026 年 6 月 10 日：发布 TeleStyleV2，通过自蒸馏支持艺术内容参考。已发布[代码](https://github.com/Tele-AI/TeleStyleV2)、[模型](https://huggingface.co/Tele-AI/TeleStyleV2)和[在线演示](https://huggingface.co/spaces/witcherderivia/TeleStyleV2)。
- 2026 年 1 月 30 日：完善代码并更新 `requirements.txt`；上传性能更优的新版本 TeleStyle-Image 模型；同时发布 [TeleStyle-Image 免费在线演示](https://huggingface.co/spaces/witcherderivia/TeleStyle)。如果该项目对你有帮助，欢迎点 Star 支持。
- 2026 年 1 月 28 日：发布 TeleStyle 的<a href="http://arxiv.org/abs/2601.20175" target="_blank">技术报告</a>、<a href="https://github.com/Tele-AI/TeleStyle" target="_blank">代码</a>和<a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">模型</a>。

## 使用方法

### 1. 安装

```
pip install -r requirements.txt
```

已验证的运行环境：
- Python 3.11
- PyTorch 2.9.1 + CUDA 12.1
- diffusers 0.36.0
- transformers 4.57.3

### 2. 下载检查点

将 [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) 下载至本地路径，例如 `./`。

将 [TeleStyle 检查点](https://huggingface.co/Tele-AI/TeleStyle/tree/main) 下载至本地路径，例如 `./weights/`：

我们提供图像和视频检查点：

- **图像（风格参考图 + 内容图 → 风格化图像）**
  diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors; diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors

- **视频（风格化首帧 + 内容视频 → 风格化视频）**
  dit.ckpt; prompt_embeds.pth

### 3. 推理

我们提供了运行 TeleStyle-Image 和 TeleStyle-Video 的推理脚本：

#### 图像风格化

在 `telestyleimage_inference.py` 中将 `content_ref` 和 `style_ref` 改为你自己的内容图与风格图路径，然后运行：

```
python telestyleimage_inference.py
```

#### 接缝感知全景图风格化
对等距柱状全景图（ERP），请使用专用脚本。默认单分支模式会在生成前环形扩展左右边缘，
并在每个去噪步骤同步重复的边缘 latent，最后裁取中心全景图，不进行 RGB 空间羽化。

如需改善南北极附近的畸变，可启用双分支极区融合：脚本会额外生成绕 X 轴球面旋转后的 ERP，
在最初若干步将其去噪预测旋回原坐标系并仅引导高纬区域，后续步骤仅由原始视角细化。该模式约增加一倍去噪时间。
```
python telestylepanorama_inference.py \
  --content <content-path> \
  --style <style-path> \
  --output <output-path> \
  --margin-px 256 \
  --blend-px 96

# 可选：启用双分支极区融合（默认关闭）
python telestylepanorama_inference.py \
  --content <content-path> \
  --style <style-path> \
  --output <output-path> \
  --enable-polar-fusion \
  --polar-rotation-degrees 90 \
  --polar-blend-start-degrees 45 \
  --polar-blend-end-degrees 75 \
  --polar-fusion-steps 2 \
  --polar-lowpass-radius-latent 8
```
输出保持输入全景图分辨率。双分支模式默认以 `0.35`、`0.20` 的系数引导最初两步；旋回后的 B 预测会在高纬做经度环形低通，`--polar-lowpass-radius-latent` 默认 `8`，设为 `0` 可关闭。`--polar-fusion-strength` 为引导调度的倍率。其余参数可通过 `--help` 查看。

#### 批量全景风格迁移

批量工具会将内容目录内的每张 ERP 全景图，与风格目录内的每张风格图做笛卡尔积。默认输出至 `qwen_style_output/batch/<style-name>/<content-name>.png`；`--styles style1` 可只补跑指定风格，不会触碰其他风格目录。每次运行会覆盖所选风格的同名输出，并在 `输出目录/manifests/` 写入 JSONL 清单，记录风格文件哈希、结果状态和耗时。

可使用下载脚本获取 Zillow ZInD `sample_tour/000` 的 ERP 内容图：

```
python download_zind_sample_panos.py
```

准备内容图目录和风格图目录后运行：

```
python telestylepanorama_batch.py
```

可通过 `--content-dir`、`--style-dir`、`--output-dir` 指定目录；批处理默认启用极区融合，支持与单图相同的 `--polar-*` 参数。下载脚本默认跳过已有内容图，传入 `--overwrite` 才会重新下载。

#### 视频风格化
```
python telestylevideo_inference.py --video_path <video-path> --image_path <style-image-path> --output_path <output-path>
```
