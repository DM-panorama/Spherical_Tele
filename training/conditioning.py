"""TeleStyle prompt and style-image conditioning caches."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImageUnit_PromptEmbedder

from .checkpoint import sha256_file


TELESTYLE_PROMPT = (
    "Style Transfer the style of Figure 2 to Figure 1, and keep the content "
    "and characteristics of Figure 1."
)


def split_style_paths(paths: list[Path]) -> tuple[list[Path], Path]:
    """Return style_1..3 training references and held-out style_4."""
    by_stem = {path.stem: path for path in paths}
    required = [f"style_{index}" for index in range(1, 5)]
    missing = [name for name in required if name not in by_stem]
    if missing:
        raise ValueError("missing required style images: " + ", ".join(missing))
    return [by_stem[name] for name in required[:3]], by_stem[required[3]]


def discover_style_paths(style_root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path for path in style_root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


@torch.no_grad()
def compute_prompt_conditioning(pipe, prompt: str) -> dict[str, torch.Tensor]:
    """Compute live TeleStyle text conditioning instead of using cache text."""
    output = QwenImageUnit_PromptEmbedder().process(
        pipe, prompt=prompt, edit_image=None
    )
    return {
        "prompt_emb": output["prompt_emb"].cpu(),
        "prompt_mask": output["prompt_emb_mask"].cpu(),
    }


def style_cache_key(path: Path, size: int) -> str:
    return f"{sha256_file(path)}-{size}x{size}"


@torch.no_grad()
def cache_style_latent(pipe, path: Path, cache_root: Path, size: int) -> torch.Tensor:
    """Encode one square style reference and reuse a content-addressed cache."""
    if size <= 0 or size % 16:
        raise ValueError("style size must be a positive multiple of 16.")
    key = style_cache_key(path, size)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{key}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") == "telestyle-style-latent-v1"
            and payload.get("key") == key
        ):
            return payload["latent"]
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size))
    pixels = pipe.preprocess_image(image).to(
        device=pipe.device, dtype=pipe.torch_dtype
    )
    pipe.load_models_to_device(["vae"])
    latent = pipe.vae.encode(
        pixels, tiled=False, tile_size=128, tile_stride=64
    ).cpu()
    torch.save(
        {
            "format": "telestyle-style-latent-v1",
            "key": key,
            "source": str(path.resolve()),
            "source_sha256": sha256_file(path),
            "size": size,
            "latent": latent,
        },
        cache_path,
    )
    return latent


def style_manifest(paths: list[Path], size: int) -> list[dict[str, str | int]]:
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path), "size": size}
        for path in paths
    ]
