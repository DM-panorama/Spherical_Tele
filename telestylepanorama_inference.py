"""Spherical-chart stylization for equirectangular panoramas (ERP).

The default path denoises overlapping north/south stereographic charts,
synchronizes their shared equatorial latents after every scheduler step, and
decodes one spherical-padded ERP latent.  The previous wrapped ERP path remains
available as an explicit legacy mode.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from telestyleimage_inference import ImageStyleInference
from telestyle_spherical import (
    extract_stereographic_hemisphere,
    rotate_erp_image,
)


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


def _stylize_panorama_legacy(
    engine: ImageStyleInference,
    content: Image.Image,
    style: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    margin_px: int,
    blend_px: int,
    enable_polar_fusion: bool = False,
    polar_rotation_degrees: float = 90.0,
    polar_blend_start_degrees: float = 45.0,
    polar_blend_end_degrees: float = 75.0,
    polar_fusion_steps: int = 2,
    polar_fusion_strength: float = 1.0,
    polar_lowpass_radius_latent: int = 8,
    polar_detail_limiter: bool = True,
    polar_detail_start_degrees: float = 65.0,
    polar_detail_end_degrees: float = 88.0,
    polar_detail_radius_latent: int = 24,
    polar_detail_steps: int = 2,
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
    if enable_polar_fusion:
        if polar_fusion_steps <= 0:
            raise ValueError("polar-fusion-steps must be greater than zero.")
        if not 0.0 <= polar_fusion_strength <= 1.0:
            raise ValueError("polar-fusion-strength must be between zero and one.")
        if not 0 <= polar_blend_start_degrees < polar_blend_end_degrees < 90:
            raise ValueError("polar blend degrees must satisfy 0 <= start < end < 90.")
        if polar_lowpass_radius_latent < 0:
            raise ValueError("polar-lowpass-radius-latent cannot be negative.")
        if polar_detail_radius_latent < 0:
            raise ValueError("polar-detail-radius-latent cannot be negative.")
        if polar_detail_steps <= 0:
            raise ValueError("polar-detail-steps must be greater than zero.")
        if not 0 <= polar_detail_start_degrees < polar_detail_end_degrees < 90:
            raise ValueError("polar detail degrees must satisfy 0 <= start < end < 90.")

    wrapped = make_wrapped_canvas(content, margin)
    working_content = resize_for_pipeline(wrapped)
    working_content_b = None
    if enable_polar_fusion:
        rotated_content = rotate_erp_image(content, polar_rotation_degrees)
        working_content_b = resize_for_pipeline(make_wrapped_canvas(rotated_content, margin))
        if working_content_b.size != working_content.size:
            raise ValueError("Rotated and original working canvases must have identical dimensions.")
    working_style = style.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    x0 = round(margin * working_content.width / wrapped.width)
    x1 = round((margin + source_size[0]) * working_content.width / wrapped.width)
    centre_x_latent = x0 // 8
    centre_width_latent = x1 // 8 - centre_x_latent
    blend_width_latent = min(round(blend_px * working_content.width / wrapped.width / 8), centre_x_latent, centre_width_latent // 2)
    if enable_polar_fusion:
        generated = engine.inference_with_latent_polar_fusion(
            prompt, working_content, working_content_b, working_style, seed, steps,
            centre_x_latent, centre_width_latent, blend_width_latent,
            polar_rotation_degrees, polar_blend_start_degrees,
            polar_blend_end_degrees, polar_fusion_steps, polar_fusion_strength,
            polar_lowpass_radius_latent, polar_detail_limiter,
            polar_detail_start_degrees, polar_detail_end_degrees,
            polar_detail_radius_latent, polar_detail_steps,
        )
    else:
        generated = engine.inference_with_latent_seam_sync(
            prompt, working_content, working_style, seed, steps,
            centre_x_latent, centre_width_latent, blend_width_latent,
        )
    return extract_panorama(generated, source_size, margin), margin, working_content.size


def _effective_decode_padding(
    target_size: Tuple[int, int], requested_padding: int,
) -> int:
    """Clamp decode padding to the aligned ERP dimensions."""
    if requested_padding < 0:
        raise ValueError("decode-padding-px cannot be negative.")
    padding = min(requested_padding, min(target_size) // 2)
    return padding - padding % 16


def stylize_panorama(
    engine: ImageStyleInference,
    content: Image.Image,
    style: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    margin_px: int,
    blend_px: int,
    enable_polar_fusion: bool = False,
    polar_rotation_degrees: float = 90.0,
    polar_blend_start_degrees: float = 45.0,
    polar_blend_end_degrees: float = 75.0,
    polar_fusion_steps: int = 2,
    polar_fusion_strength: float = 1.0,
    polar_lowpass_radius_latent: int = 8,
    polar_detail_limiter: bool = True,
    polar_detail_start_degrees: float = 65.0,
    polar_detail_end_degrees: float = 88.0,
    polar_detail_radius_latent: int = 24,
    polar_detail_steps: int = 2,
    panorama_mode: str = "hemisphere",
    hemisphere_size: int | None = None,
    hemisphere_overlap_degrees: float = 15.0,
    decode_padding_px: int = 128,
) -> tuple[Image.Image, int, Tuple[int, int]]:
    """Stylize an ERP with synchronized hemisphere charts or the legacy path."""
    if panorama_mode == "legacy":
        return _stylize_panorama_legacy(
            engine, content, style, prompt, seed, steps, margin_px, blend_px,
            enable_polar_fusion, polar_rotation_degrees,
            polar_blend_start_degrees, polar_blend_end_degrees,
            polar_fusion_steps, polar_fusion_strength,
            polar_lowpass_radius_latent, polar_detail_limiter,
            polar_detail_start_degrees, polar_detail_end_degrees,
            polar_detail_radius_latent, polar_detail_steps,
        )
    if panorama_mode != "hemisphere":
        raise ValueError("panorama_mode must be 'hemisphere' or 'legacy'.")
    if enable_polar_fusion:
        raise ValueError("legacy polar options require --panorama-mode legacy.")

    content = content.convert("RGB")
    source_size = content.size
    if source_size[0] < 32 or source_size[1] < 16:
        raise ValueError("Content panorama must be at least 32x16 pixels.")
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if not 0 < hemisphere_overlap_degrees < 45:
        raise ValueError(
            "hemisphere-overlap-degrees must be between zero and 45."
        )
    if hemisphere_size is None:
        chart_size = _nearest_multiple_of_16(source_size[1])
    else:
        if hemisphere_size <= 0 or hemisphere_size % 16:
            raise ValueError(
                "hemisphere-size must be positive and divisible by 16."
            )
        chart_size = hemisphere_size

    target_size = (
        _nearest_multiple_of_16(source_size[0]),
        _nearest_multiple_of_16(source_size[1]),
    )
    decode_padding = _effective_decode_padding(target_size, decode_padding_px)
    content_north = extract_stereographic_hemisphere(
        content, chart_size, hemisphere_overlap_degrees, north=True
    )
    content_south = extract_stereographic_hemisphere(
        content, chart_size, hemisphere_overlap_degrees, north=False
    )
    working_style = style.convert("RGB").resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    generated = engine.inference_with_hemisphere_latent_sync(
        prompt, content_north, content_south, working_style, seed, steps,
        target_size[1], target_size[0], hemisphere_overlap_degrees,
        decode_padding // 8,
    )
    if generated.size != source_size:
        generated = generated.resize(source_size, Image.Resampling.LANCZOS)
    return generated, decode_padding, (chart_size, chart_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spherical-chart TeleStyle ERP stylization")
    parser.add_argument("--content", required=True, help="Input equirectangular panorama")
    parser.add_argument("--style", required=True, help="Style reference image")
    parser.add_argument("--output", required=True, help="Output panorama path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Editing instruction")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--panorama-mode", choices=("hemisphere", "legacy"),
        default="hemisphere", help="Panorama generation method",
    )
    parser.add_argument(
        "--hemisphere-size", type=int, default=None,
        help="Square chart size; defaults to the aligned ERP height",
    )
    parser.add_argument(
        "--hemisphere-overlap-degrees", type=float, default=15.0,
        help="Angular overlap across the equator for latent synchronization",
    )
    parser.add_argument(
        "--decode-padding-px", type=int, default=128,
        help="Spherical padding used only for the final ERP VAE decode",
    )
    parser.add_argument("--margin-px", type=int, default=256, help="Circular extension width")
    parser.add_argument("--blend-px", type=int, default=96, help="Latent seam synchronization width")
    parser.add_argument("--enable-polar-fusion", action="store_true", help="Enable rotated dual-branch polar prediction fusion")
    parser.add_argument("--polar-rotation-degrees", type=float, default=90.0, help="Fixed X-axis ERP rotation for branch B")
    parser.add_argument("--polar-blend-start-degrees", type=float, default=45.0, help="Latitude where polar fusion begins")
    parser.add_argument("--polar-blend-end-degrees", type=float, default=75.0, help="Latitude where polar fusion reaches full weight")
    parser.add_argument("--polar-fusion-steps", type=int, default=2, help="Number of initial denoising steps guided by branch B")
    parser.add_argument("--polar-fusion-strength", type=float, default=1.0, help="Multiplier for the early B-to-A guidance schedule")
    parser.add_argument("--polar-lowpass-radius-latent", type=int, default=8, help="Maximum polar circular low-pass radius in latent pixels; zero disables it")
    parser.add_argument("--polar-detail-limiter", action=argparse.BooleanOptionalAction, default=True, help="Limit oversampled polar detail during final A refinement")
    parser.add_argument("--polar-detail-start-degrees", type=float, default=65.0, help="Latitude where final A detail limiting begins")
    parser.add_argument("--polar-detail-end-degrees", type=float, default=88.0, help="Latitude where final A detail limiting reaches full weight")
    parser.add_argument("--polar-detail-radius-latent", type=int, default=24, help="Maximum final A polar low-pass radius in latent pixels")
    parser.add_argument("--polar-detail-steps", type=int, default=2, help="Number of final A denoising steps to limit polar detail")
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
            engine=engine, content=content, style=style, prompt=args.prompt,
            seed=args.seed, steps=args.steps, margin_px=args.margin_px,
            blend_px=args.blend_px,
            enable_polar_fusion=args.enable_polar_fusion,
            polar_rotation_degrees=args.polar_rotation_degrees,
            polar_blend_start_degrees=args.polar_blend_start_degrees,
            polar_blend_end_degrees=args.polar_blend_end_degrees,
            polar_fusion_steps=args.polar_fusion_steps,
            polar_fusion_strength=args.polar_fusion_strength,
            polar_lowpass_radius_latent=args.polar_lowpass_radius_latent,
            polar_detail_limiter=args.polar_detail_limiter,
            polar_detail_start_degrees=args.polar_detail_start_degrees,
            polar_detail_end_degrees=args.polar_detail_end_degrees,
            polar_detail_radius_latent=args.polar_detail_radius_latent,
            polar_detail_steps=args.polar_detail_steps,
            panorama_mode=args.panorama_mode,
            hemisphere_size=args.hemisphere_size,
            hemisphere_overlap_degrees=args.hemisphere_overlap_degrees,
            decode_padding_px=args.decode_padding_px,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    if args.panorama_mode == "hemisphere":
        mode_details = (
            f"chart={working_size} | overlap={args.hemisphere_overlap_degrees}deg | "
            f"decode_padding={margin}px"
        )
    else:
        mode_details = (
            f"model_canvas={working_size} | margin={margin}px | "
            f"blend={args.blend_px}px | polar_fusion={args.enable_polar_fusion}"
        )
    print(
        f"Saved {output_path} | input={content.size} | "
        f"mode={args.panorama_mode} | {mode_details}"
    )


if __name__ == "__main__":
    main()
