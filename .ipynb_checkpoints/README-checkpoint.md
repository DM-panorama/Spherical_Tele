# <div align="center">TeleStyle: Content-Preserving Style Transfer in Images and Videos</div>

<div align="center">
  Shiwen Zhang, Xiaoyan Yang, Bojia Zi, Haibin Huang, Chi Zhang, Xuelong Li<br>
  Institute of Artificial Intelligence, China Telecom (TeleAI)
</div>

<br>

<div align="center">
  <a href="https://tele-ai.github.io/TeleStyle/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2601.20175">arXiv</a> ·
  <a href="https://huggingface.co/Tele-AI/TeleStyle">Hugging Face</a> ·
  <a href="https://github.com/Tele-AI/TeleStyle">GitHub</a> ·
  <a href="https://huggingface.co/spaces/witcherderivia/TeleStyle">Demo</a>
</div>

## Overview

TeleStyle performs reference-based, content-preserving stylization for images and videos. Image stylization uses a content image and a style image; video stylization uses a source video and a stylized first-frame reference. The repository also includes a seam-aware workflow for equirectangular panoramas (ERP).

For model releases and examples, see the [Hugging Face collection](https://huggingface.co/Tele-AI/TeleStyle). TeleStyleV2 is maintained in its [own repository](https://github.com/Tele-AI/TeleStyleV2).

## Setup

Tested environment:

- Python 3.11
- PyTorch 2.9.1 with CUDA 12.1
- `diffusers` 0.36.0
- `transformers` 4.57.3
- A CUDA-capable GPU (the provided inference scripts use CUDA)

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` installs DiffSynth from a pinned Git commit. If Git is unavailable in the target environment, use `requirements.no_git.txt` after installing the matching DiffSynth package by another method.

## Checkpoints

Download the following checkpoints before inference.

| Purpose | Source | Expected local location |
| --- | --- | --- |
| Qwen Image Edit base model | Follow the model layout required by DiffSynth | `/root/autodl-tmp/Qwen-Image-Edit-2509/` for the current image script |
| Wan 2.1 T2V base model | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | `./Wan2.1-T2V-1.3B-Diffusers/` for video |
| TeleStyle image LoRAs | [Tele-AI/TeleStyle](https://huggingface.co/Tele-AI/TeleStyle/tree/main) | `weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors` and `weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` |
| TeleStyle video weights | [Tele-AI/TeleStyle](https://huggingface.co/Tele-AI/TeleStyle/tree/main) | `weights/dit.ckpt` and `weights/prompt_embeds.pth` |

The image script currently defines its base-model path in `ImageStyleInference._load_models()` as `/root/autodl-tmp/Qwen-Image-Edit-2509`. Update that constant if your checkpoint lives elsewhere. The directory must contain `transformer/`, `text_encoder/`, `vae/`, and `processor/` subdirectories expected by the script.

## Inference

Sample assets are provided under `inputs/` and `assets/example/`.

### Image stylization

Set `content_ref` and `style_ref` in `telestyleimage_inference.py`, then run:

```bash
python telestyleimage_inference.py
```

The script resizes the content image so its short side is 1024 pixels (aligned to 16 pixels), resizes the style reference to 1024×1024, and writes the result to `qwen_style_output/<style-name>_result.png`.

### Seam-aware panorama stylization

For ERP panoramas, use the dedicated command-line script:

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_result.png \
  --margin-px 256 \
  --blend-px 96
```

It creates a circular working canvas (`right edge | panorama | left edge`), keeps duplicate edge regions synchronized in latent space during denoising, and crops the centre image back to the source resolution. Use `python telestylepanorama_inference.py --help` to view prompt, seed, step, and seam parameters.

#### Optional spherical polar fusion

ERP projection over-samples the north and south poles. Enable the optional second branch to rotate the panorama on the unit sphere around the fixed X axis, denoise that reoriented view alongside the original view, and fuse its inverse-projected latent into the original branch near the poles. This is a true ERP → sphere → rotation → ERP resampling path, not an image-space 90° rotation. The final image is decoded once from the fused original branch.

```bash
python telestylepanorama_inference.py \
  --content inputs/panorama.png \
  --style inputs/style.jpg \
  --output qwen_style_output/panorama_polar_fusion.png \
  --spherical-polar-fusion \
  --spherical-rotation-deg 90 \
  --polar-blend-start-deg 45 \
  --polar-blend-end-deg 75
```

The feature is disabled by default, so existing commands keep their original single-branch behavior. `--polar-blend-start-deg` and `--polar-blend-end-deg` define the absolute-latitude cosine transition: A dominates below the start latitude and the rotated B branch dominates at and beyond the end latitude. Dual-branch mode evaluates the DiT twice per denoising step and therefore needs additional runtime and VRAM headroom.

### Video stylization

```bash
python telestylevideo_inference.py \
  --video_path assets/example/1.mp4 \
  --image_path assets/example/1-0.png \
  --output_path results_video
```

By default, video inference uses 129 frames, a 1248×720 target size, 25 denoising steps, and writes `results_video/generated_video.mp4`. Pass command-line options such as `--video_length`, `--height`, `--width`, `--num_inference_steps`, and checkpoint paths to override these values.

## Repository layout

| Path | Description |
| --- | --- |
| `telestyleimage_inference.py` | Image reference stylization and latent seam synchronization helper |
| `telestylepanorama_inference.py` | CLI for seam-aware equirectangular panorama stylization |
| `telestylevideo_inference.py` | Video inference entry point |
| `telestylevideo_pipeline.py` | Wan-based video diffusion pipeline |
| `telestylevideo_transformer.py` | TeleStyle video transformer implementation |
| `inputs/`, `assets/` | Example input images, videos, and project-site media |
| `weights/` | User-downloaded TeleStyle weights; not committed |

## Citation

```bibtex
@article{teleai2026telestyle,
  title={TeleStyle: Content-Preserving Style Transfer in Images and Videos},
  author={Shiwen Zhang and Xiaoyan Yang and Bojia Zi and Haibin Huang and Chi Zhang and Xuelong Li},
  journal={arXiv preprint arXiv:2601.20175},
  year={2026}
}
```

## Acknowledgements

Community ComfyUI implementations:

- [aistudynow/Comfyui-tetestyle-image-video](https://github.com/aistudynow/Comfyui-tetestyle-image-video)
- [neurodanzelus-cmd/ComfyUI-TeleStyle](https://github.com/neurodanzelus-cmd/ComfyUI-TeleStyle)
