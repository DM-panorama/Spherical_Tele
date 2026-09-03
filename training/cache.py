"""Build deterministic native-chart latent and prompt caches."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import (
    QwenImageUnit_PromptEmbedder,
)

from telestyle_spherical import extract_stereographic_hemisphere

from .data import ErpRecord
from .geometry import rotate_erp


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor[0].permute(1, 2, 0).clamp(0, 1) * 255
    ).round().byte().cpu().numpy()
    return Image.fromarray(array, "RGB")


def _variant_parameters(
    fingerprint: str,
    variant: int,
    overlap_range: tuple[float, float],
) -> tuple[float, float, float, float, bool]:
    seed = int(fingerprint[:16], 16) + variant * 104729
    generator = torch.Generator().manual_seed(seed)
    yaw = float(torch.rand((), generator=generator) * 360 - 180)
    pitch = float(torch.rand((), generator=generator) * 120 - 60)
    roll = float(torch.rand((), generator=generator) * 120 - 60)
    overlap_steps = round((overlap_range[1] - overlap_range[0]) / 0.5)
    overlap_index = int(
        torch.randint(overlap_steps + 1, (), generator=generator)
    )
    overlap = overlap_range[0] + overlap_index * 0.5
    swap = bool(torch.randint(0, 2, (), generator=generator))
    return yaw, pitch, roll, overlap, swap


@torch.no_grad()
def _encode_chart(pipe, chart: Image.Image) -> dict[str, torch.Tensor]:
    pipe.load_models_to_device(["vae"])
    pixels = pipe.preprocess_image(chart).to(
        device=pipe.device, dtype=pipe.torch_dtype
    )
    latent = pipe.vae.encode(pixels, tiled=False, tile_size=128, tile_stride=64)
    return {"latent": latent.cpu()}


def build_cache(
    records: list[ErpRecord],
    cache_root: Path,
    pipe,
    chart_size: int,
    prompt: str,
    overlap_range: tuple[float, float],
    variants: int = 4,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    index_path = cache_root / "index.jsonl"
    prompt_unit = QwenImageUnit_PromptEmbedder()
    prompt_output = prompt_unit.process(pipe, prompt=prompt, edit_image=None)
    conditioning_path = cache_root / "conditioning.pt"
    torch.save(
        {
            "prompt_emb": prompt_output["prompt_emb"].cpu(),
            "prompt_mask": prompt_output["prompt_emb_mask"].cpu(),
        }, conditioning_path,
    )
    index_items = []
    for record in records:
        with Image.open(record.path) as source_image:
            source = _pil_to_tensor(source_image)
        for variant in range(variants):
            yaw, pitch, roll, overlap, swap = _variant_parameters(
                record.fingerprint, variant, overlap_range
            )
            if swap:
                pitch += 180.0
            cache_id = hashlib.sha256(
                f"{record.fingerprint}:{variant}:{chart_size}".encode()
            ).hexdigest()[:24]
            cache_path = cache_root / f"{cache_id}.pt"
            is_stress = record.scene_id.startswith("stress:")
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if cached.get("format") == "telestyle-a1-cache-v1":
                    index_items.append(
                        {"path": str(cache_path.resolve()), "split": record.split, "stress": is_stress}
                    )
                    continue
            rotated = _tensor_to_pil(rotate_erp(source, yaw, pitch, roll))
            north = extract_stereographic_hemisphere(
                rotated, chart_size, overlap, north=True
            )
            south = extract_stereographic_hemisphere(
                rotated, chart_size, overlap, north=False
            )
            north_encoded = _encode_chart(pipe, north)
            south_encoded = _encode_chart(pipe, south)
            payload = {
                "format": "telestyle-a1-cache-v1",
                "stress": is_stress,
                "source_path": record.path,
                "source_fingerprint": record.fingerprint,
                "scene_id": record.scene_id,
                "split": record.split,
                "variant": variant,
                "rotation_degrees": [yaw, pitch, roll],
                "overlap_degrees": overlap,
                "swapped": swap,
                "chart_size": chart_size,
                "north": north_encoded,
                "south": south_encoded,
            }
            torch.save(payload, cache_path)
            index_items.append(
                {"path": str(cache_path.resolve()), "split": record.split, "stress": is_stress}
            )
    with index_path.open("w", encoding="utf-8") as handle:
        for item in index_items:
            handle.write(json.dumps(item) + "\n")
    metadata_path = cache_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {"conditioning": str(conditioning_path.resolve()), "prompt": prompt}
        ),
        encoding="utf-8",
    )
    return index_path
