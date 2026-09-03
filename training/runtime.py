"""Runtime selection and fail-fast checks for full A1 training."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RuntimeProfile:
    chart_size: int
    gradient_accumulation: int
    cpu_offload: bool
    vram_gb: float


def require_python_311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "A1 full-model caching/training requires Python 3.11; "
            f"detected {sys.version_info.major}.{sys.version_info.minor}."
        )


def select_runtime_profile(config) -> RuntimeProfile:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. CPU is supported only for unit tests and toy smoke tests."
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    vram_gb = properties.total_memory / (1024**3)
    profiles = sorted(
        config.runtime.chart_profiles,
        key=lambda item: float(item.minimum_vram_gb),
        reverse=True,
    )
    for item in profiles:
        if vram_gb >= float(item.minimum_vram_gb):
            return RuntimeProfile(
                chart_size=int(item.chart_size),
                gradient_accumulation=int(item.gradient_accumulation),
                cpu_offload=bool(item.cpu_offload),
                vram_gb=vram_gb,
            )
    raise RuntimeError(
        f"A1 requires at least {config.runtime.minimum_training_vram_gb} GiB VRAM; "
        f"detected {vram_gb:.1f} GiB."
    )


def require_model_files(model_dir: Path) -> None:
    required = (
        model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        model_dir / "text_encoder" / "model.safetensors.index.json",
        model_dir / "vae" / "diffusion_pytorch_model.safetensors",
        model_dir / "processor" / "preprocessor_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing base-model files: " + ", ".join(missing))


def require_free_space(path: Path, estimated_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    reserve = 1024**3
    if free < estimated_bytes + reserve:
        raise RuntimeError(
            f"insufficient space at {path}: need about "
            f"{(estimated_bytes + reserve) / 1024**3:.1f} GiB including reserve, "
            f"have {free / 1024**3:.1f} GiB."
        )


def estimate_cache_bytes(
    records: int, chart_size: int, variants: int = 4,
) -> int:
    latent_side = chart_size // 8
    two_latents_bf16 = 2 * 16 * latent_side * latent_side * 2
    overhead = 64 * 1024
    global_prompt_allowance = 32 * 1024 * 1024
    return records * variants * (two_latents_bf16 + overhead) + global_prompt_allowance
