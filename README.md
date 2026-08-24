# <div align="center">TeleStyle:  Content-Preserving Style Transfer in Images and Videos</div>
<div align="center">
    Shiwen Zhang, Xiaoyan Yang, Bojia Zi, Haibin Huang, Chi Zhang, Xuelong Li
    <br>
    Institute of Artificial Intelligence, China Telecom (TeleAI) 
</div>
<br>
<div align="center">
    [<a href="https://tele-ai.github.io/TeleStyle/" target="_blank">Project Page</a>]
    [<a href="http://arxiv.org/abs/2601.20175" target="_blank">arXiv</a>]
    [<a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">Hugging Face</a>]
    [<a href="https://github.com/Tele-AI/TeleStyle" target="_blank">GitHub</a>]
    [<a href="https://huggingface.co/spaces/witcherderivia/TeleStyle" target="_blank">Demo</a>]
</div>

## Abstract
Content-preserving style transfer—generating stylized outputs based on content and style references—remains a significant challenge for Diffusion Transformers (DiTs) due to the inherent entanglement of content and style features in their internal representations. In this technical report, we present TeleStyle, a lightweight yet effective model for both image and video stylization. Built upon Qwen-Image-Edit, TeleStyle leverages the base model’s robust capabilities in content preservation and style customization. To facilitate effective training, we curated a high-quality dataset of distinct specific styles and further synthesized triplets using thousands of diverse, in-the-wild style categories. We introduce a Curriculum Continual Learning framework to train TeleStyle on this hybrid dataset of clean (curated) and noisy (synthetic) triplets. This approach enables the model to generalize to unseen styles without compromising precise content fidelity. Additionally, we introduce a video-to-video stylization module to enhance temporal consistency and visual quality. TeleStyle achieves state-of-the-art performance across three core evaluation metrics: style similarity, content consistency, and aesthetic quality.

## Latest News
- Released TeleStyleV2-SenseNova, [Code](https://huggingface.co/spaces/witcherderivia/TeleStyle-SenseNova/tree/main), [Model](https://huggingface.co/Tele-AI/TeleStyleV2), [Demo](https://huggingface.co/spaces/witcherderivia/TeleStyle-SenseNova), reinforces SenseNova U1 for Content-Preserving Style Transfer and preserves its general image editing capability. This experimental model is trained in pixel space, making the sft quite hard and still having much space to improve. The model supports 1MP to 4MP.
- June 10, 2026: We release TeleStyleV2, supporting artistic content reference via self distillation. Released [Code](https://github.com/Tele-AI/TeleStyleV2), [Model](https://huggingface.co/Tele-AI/TeleStyleV2), [Demo](https://huggingface.co/spaces/witcherderivia/TeleStyleV2).
- Jan 30, 2026: We refine the code and update requirements.txt. In addition, a new version of TeleStyle-Image model with better performance has been uploaded. Finally, we release a [free online demo for TeleStyle-Image ](https://huggingface.co/spaces/witcherderivia/TeleStyle). Please light a star to support this project if you find the demo useful. 
- Jan 28, 2026: We release the <a href="http://arxiv.org/abs/2601.20175" target="_blank">technical report </a>, <a href="https://github.com/Tele-AI/TeleStyle" target="_blank">code</a> and <a href="https://huggingface.co/Tele-AI/TeleStyle" target="_blank">model</a> of TeleStyle.

## Todo List

- [x] Release inference code
- [x] Release models
- [x] Release technical report



## How to use

### 1. Installation

```
pip install -r requirements.txt
```

This environment is tested with:
- Python 3.11
- PyTorch 2.9.1 + CUDA 12.1
- diffusers 0.36.0
- transformers 4.57.3

### 2. Download Checkpoint

Download the [Wan2.1-T2V-1.3B-Diffusers]([https://huggingface.co/Tele-AI/TeleStyle/tree/main](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)) to a local path for example `./`.

Download the [TeleStyle checkpoint](https://huggingface.co/Tele-AI/TeleStyle/tree/main) to a local path for example `./weights/`:

We provide Image and Video checkpoint:

- **Image (reference style image + content image -> stylized image)**  
  diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors; diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors
  

- **Video (stylized first frame + content video -> stylized video)**  
  dit.ckpt; prompt_embeds.pth

### 3. Inference

We provide inference scripts for running TeleStyle-Image and TeleStyle-Video:

#### Image Stylization
```
python telestyleimage_inference.py
```

#### Spherical-chart Panorama Stylization
对等距柱状全景图（ERP），请使用专用脚本。默认 `hemisphere` 模式会把输入分别投影成以南北极为中心的方形极射投影 chart；两张 chart 默认各越过赤道 `15°`。初始化 latent 后以及每个 scheduler step 后，程序都会在共同的赤道带内按球面坐标同步两份 latent。

去噪完成后，南北 chart 先在潜空间重投影成一张 ERP latent，再增加左右循环 padding 和“纬度反射 + 经度半周平移”的极点 padding，最后只执行一次 VAE 解码。因此赤道、左右接缝和极点都不依赖生成后的 RGB 拼接。两张 chart 使用独立 scheduler 和 latent 轨迹，计算量约为两个同尺寸方形图的去噪。
```
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_result.png

# 可选：覆盖默认 chart 尺寸、15° 重叠带和 128px 解码 padding
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_custom.png \
  --hemisphere-size 1024 \
  --hemisphere-overlap-degrees 15 \
  --decode-padding-px 128
```
输出保持输入全景图分辨率。`--hemisphere-size` 默认取输入 ERP 高度并对齐到 16；显式值必须为正且能被 16 整除。重叠角必须在 `0°–45°` 之间。方形 chart 的四角继续采样有效球面内容，不使用黑色圆外遮罩。

旧环形 ERP、旋转双分支极区融合和最终 RGB 极区 patch 仍可通过显式 legacy 模式使用：
```
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_legacy.png \
  --panorama-mode legacy \
  --margin-px 256 \
  --blend-px 96

# legacy 模式下仍可附加 --enable-polar-fusion 或 --enable-polar-patches
```
旧极区参数仅在 `--panorama-mode legacy` 下生效；其余参数可通过 `--help` 查看。

#### Video Stylization
```
python telestylevideo_inference.py --video_path assets/example/1.mp4 --image_path assets/example/1-0.png --output_path results
```

### ComfyUI
Thanks to the community for providing the ComfyUI implementation:
- [aistudynow/Comfyui-tetestyle-image-video](https://github.com/aistudynow/Comfyui-tetestyle-image-video)
- [neurodanzelus-cmd/ComfyUI-TeleStyle](https://github.com/neurodanzelus-cmd/ComfyUI-TeleStyle)

## Citation
If you find TeleStyle useful in your research, please light a star for the project and cite our paper, thank you:
```bibtex
@article{teleai2026telestyle,
    title={TeleStyle: Content-Preserving Style Transfer in Images and Videos}, 
    author={Shiwen Zhang and Xiaoyan Yang and Bojia Zi and Haibin Huang and Chi Zhang and Xuelong Li},
    journal={arXiv preprint arXiv:2601.20175},
    year={2026}
}
