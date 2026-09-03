"""Load Qwen Image Edit with optional, frozen inference LoRAs."""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


def load_base_pipeline(
    model_dir: Path,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    vram_limit: float | None = None,
    cpu_offload: bool = False,
    lora_paths: list[Path] | tuple[Path, ...] | None = None,
) -> QwenImagePipeline:
    """Load the base pipeline and optionally merge configured LoRAs into its DiT.

    `None` preserves the historical A1 behavior: no LoRA is loaded.
    """
    transformer = sorted(
        glob.glob(str(model_dir / "transformer" / "diffusion_pytorch_model*.safetensors"))
    )
    text_encoder = sorted(
        glob.glob(str(model_dir / "text_encoder" / "model*.safetensors"))
    )
    if not transformer or not text_encoder:
        raise FileNotFoundError(f"incomplete Qwen model at {model_dir}")
    offload_options = {}
    if cpu_offload:
        offload_options = {
            "offload_device": "cpu",
            "offload_dtype": torch_dtype,
            "onload_device": "cpu",
            "onload_dtype": torch_dtype,
            "preparing_device": "cpu",
            "preparing_dtype": torch_dtype,
            "computation_device": device,
            "computation_dtype": torch_dtype,
        }
    def model_config(path):
        return ModelConfig(path=path, **offload_options)

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        vram_limit=vram_limit,
        model_configs=[
            model_config(transformer),
            model_config(text_encoder),
            model_config(str(model_dir / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(path=str(model_dir / "processor")),
    )
    if pipe.tokenizer is None and pipe.processor is not None:
        pipe.tokenizer = pipe.processor.tokenizer
    for lora_path in lora_paths or ():
        path = Path(lora_path)
        if not path.is_file():
            raise FileNotFoundError(f"configured LoRA does not exist: {path}")
        pipe.load_lora(pipe.dit, str(path))
    for parameter in pipe.dit.parameters():
        parameter.requires_grad_(False)
    return pipe


def configured_lora_paths(config, config_path: str | Path) -> list[Path]:
    """Resolve optional LoRAs while retaining old config behavior."""
    values = config.model.get("lora_paths", [])
    root = Path(config_path).resolve().parent.parent
    paths = []
    for value in values:
        path = Path(str(value))
        paths.append(path if path.is_absolute() else (root / path).resolve())
    if len(paths) != len(set(paths)):
        raise ValueError("model.lora_paths must not contain duplicates.")
    return paths
