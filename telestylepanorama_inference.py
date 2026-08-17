"""Seam-aware stylization for equirectangular panoramas (ERP).

The content panorama is extended circularly before a single TeleStyle pass:
``[right edge | panorama | left edge]``.  The generated duplicate edges are
then feathered back into the centre crop so that the output wraps smoothly
from its last column to its first column.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from telestyleimage_inference import ImageStyleInference


DEFAULT_PROMPT = (
    "Transfer the style of Figure 2 to the equirectangular panorama in "
    "Figure 1. Preserve the panorama geometry and seamless horizontal "
    "wrap-around continuity."
)


def _nearest_multiple_of_16(value: int) -> int:
    """Return the nearest valid DiT dimension, never smaller than 16."""
    return max(16, int(round(value / 16.0)) * 16)


def _aligned_margin(width: int, requested_margin: int) -> int:
    """Clamp a circular extension to half the panorama width and align it."""
    if requested_margin <= 0:
        raise ValueError("margin-px must be greater than zero.")
    max_margin = width // 2
    margin = min(requested_margin, max_margin)
    margin -= margin % 16
    if margin < 16:
        raise ValueError(
            "Panorama is too narrow for a 16-pixel circular margin; "
            "use an image at least 32 pixels wide."
        )
    return margin


def make_wrapped_canvas(panorama: Image.Image, margin: int) -> Image.Image:
    """Create ``[right edge | panorama | left edge]`` without blending."""
    panorama = panorama.convert("RGB")
    width, height = panorama.size
    if not 0 < margin <= width // 2:
        raise ValueError("margin must be positive and no larger than half the width.")

    canvas = Image.new("RGB", (width + 2 * margin, height))
    canvas.paste(panorama.crop((width - margin, 0, width, height)), (0, 0))
    canvas.paste(panorama, (margin, 0))
    canvas.paste(panorama.crop((0, 0, margin, height)), (margin + width, 0))
    return canvas


def resize_for_pipeline(canvas: Image.Image) -> Image.Image:
    """Resize only the working canvas to DiT-compatible dimensions."""
    width, height = canvas.size
    target_size = (_nearest_multiple_of_16(width), _nearest_multiple_of_16(height))
    if target_size == canvas.size:
        return canvas
    return canvas.resize(target_size, Image.Resampling.LANCZOS)


def extract_panorama(
    generated_canvas: Image.Image,
    source_size: Tuple[int, int],
    source_margin: int,
) -> Image.Image:
    """Extract the centre ERP after latent-space seam synchronization."""
    source_width, _ = source_size
    generated_canvas = generated_canvas.convert("RGB")
    canvas_width, canvas_height = generated_canvas.size
    source_canvas_width = source_width + 2 * source_margin
    x0 = round(source_margin * canvas_width / source_canvas_width)
    x1 = round((source_margin + source_width) * canvas_width / source_canvas_width)
    x0 = max(1, min(x0, canvas_width - 1))
    x1 = max(x0 + 1, min(x1, canvas_width - 1))
    result = generated_canvas.crop((x0, 0, x1, canvas_height))
    return result.resize(source_size, Image.Resampling.LANCZOS) if result.size != source_size else result


def stylize_panorama(
    engine: ImageStyleInference,
    content: Image.Image,
    style: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    margin_px: int,
    blend_px: int,
) -> tuple[Image.Image, int, Tuple[int, int]]:
    """Run one wrapped inference pass and return the seam-blended ERP."""
    content = content.convert("RGB")
    source_size = content.size
    if source_size[0] < 32 or source_size[1] < 16:
        raise ValueError("Content panorama must be at least 32x16 pixels.")
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")

    margin = _aligned_margin(source_size[0], margin_px)
    if blend_px < 0:
        raise ValueError("blend-px cannot be negative.")
    if blend_px > margin:
        raise ValueError("blend-px cannot be larger than the effective margin.")

    wrapped = make_wrapped_canvas(content, margin)
    working_content = resize_for_pipeline(wrapped)
    working_style = style.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    x0 = round(margin * working_content.width / wrapped.width)
    x1 = round((margin + source_size[0]) * working_content.width / wrapped.width)
    centre_x_latent = x0 // 8
    centre_width_latent = x1 // 8 - centre_x_latent
    blend_width_latent = min(round(blend_px * working_content.width / wrapped.width / 8), centre_x_latent, centre_width_latent // 2)
    generated = engine.inference_with_latent_seam_sync(prompt, working_content, working_style, seed, steps, centre_x_latent, centre_width_latent, blend_width_latent)
    result = extract_panorama(generated, source_size, margin)
    return result, margin, working_content.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seam-aware TeleStyle ERP stylization")
    parser.add_argument("--content", required=True, help="Input equirectangular panorama")
    parser.add_argument("--style", required=True, help="Style reference image")
    parser.add_argument("--output", required=True, help="Output panorama path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Editing instruction")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--margin-px", type=int, default=256, help="Circular extension width")
    parser.add_argument("--blend-px", type=int, default=96, help="Latent seam synchronization width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.content, "content"), (args.style, "style")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} image does not exist: {path}")

    with Image.open(args.content) as image:
        content = image.convert("RGB")
    with Image.open(args.style) as image:
        style = image.convert("RGB")

    engine = ImageStyleInference()
    with torch.no_grad():
        result, margin, working_size = stylize_panorama(
            engine, content, style, args.prompt, args.seed, args.steps,
            args.margin_px, args.blend_px,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(
        f"Saved {output_path} | input={content.size} | "
        f"model_canvas={working_size} | margin={margin}px | blend={args.blend_px}px"
    )


if __name__ == "__main__":
    main()