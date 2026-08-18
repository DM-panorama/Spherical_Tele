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

#### Seam-aware Panorama Stylization
对等距柱状全景图（ERP），请使用专用脚本。默认单分支模式会在生成前环形扩展左右边缘，
并在每个去噪步骤同步重复的边缘 latent，最后裁取中心全景图，不进行 RGB 空间羽化。

如需改善南北极附近的畸变，可启用双分支极区融合：脚本会额外生成绕 X 轴球面旋转后的 ERP，
在最初若干步将其去噪预测旋回原坐标系并仅引导高纬区域，后续步骤仅由原始视角细化。该模式约增加一倍去噪时间。
```
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_result.png \
  --margin-px 256 \
  --blend-px 96

# 可选：启用双分支极区融合（默认关闭）
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_polar_result.png \
  --enable-polar-fusion \
  --polar-rotation-degrees 90 \
  --polar-blend-start-degrees 45 \
  --polar-blend-end-degrees 75 \
  --polar-fusion-steps 2 \
  --polar-lowpass-radius-latent 8 \
  --polar-detail-start-degrees 65 \
  --polar-detail-end-degrees 88 \
  --polar-detail-radius-latent 24 \
  --polar-detail-steps 2
```
输出保持输入全景图分辨率。双分支模式默认以 `0.35`、`0.20` 的系数引导最初两步；旋回后的 B 预测会在高纬做经度环形低通，`--polar-lowpass-radius-latent` 默认 `8`，设为 `0` 可关闭。A 在最后两步默认以 `65°–88°` 的纬度权重限制过采样极区细节，可用 `--no-polar-detail-limiter` 关闭。`--polar-fusion-strength` 为引导调度的倍率。其余参数可通过 `--help` 查看。

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
