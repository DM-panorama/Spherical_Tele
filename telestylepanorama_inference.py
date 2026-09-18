"""Batch ERP stylization with north/south charts and an optimized RGB hard-cut."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from telestyleimage_inference import ImageStyleInference
from telestyle_spherical import extract_stereographic_hemisphere


DEFAULT_PROMPT = (
    "Transfer the style of Figure 2 to the equirectangular panorama in "
    "Figure 1. Preserve the panorama geometry and seamless horizontal "
    "wrap-around continuity."
)
IMAGE_SUFFIXES = frozenset({
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})


def _nearest_multiple_of_16(value: int) -> int:
    """Return the nearest valid DiT dimension, never smaller than 16."""
    return max(16, int(round(value / 16.0)) * 16)


def _style_paths(style_dir: Path) -> list[Path]:
    """Return supported style images in deterministic filename order."""
    if not style_dir.is_dir():
        raise FileNotFoundError(f"style directory does not exist: {style_dir}")
    paths = sorted(
        (
            path for path in style_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise FileNotFoundError(
            f"style directory contains no supported images: {style_dir}"
        )
    return paths


def _restore_outside_equatorial_band(
    repaired: Image.Image, baseline: Image.Image, band_degrees: float,
) -> Image.Image:
    """Keep the original hard cut exactly outside the repaired latitude band."""
    if repaired.size != baseline.size:
        raise ValueError("repaired and baseline images must have identical sizes.")
    if band_degrees == 0:
        return baseline.copy()
    repaired_array = np.asarray(repaired.convert("RGB"))
    baseline_array = np.asarray(baseline.convert("RGB")).copy()
    height = repaired_array.shape[0]
    latitude = 90.0 - (np.arange(height, dtype=np.float32) + 0.5) * (
        180.0 / height
    )
    inside = np.abs(latitude) < band_degrees
    baseline_array[inside, :, :] = repaired_array[inside, :, :]
    return Image.fromarray(baseline_array, "RGB")


def stylize_panorama(
    engine: ImageStyleInference,
    content: Image.Image,
    style: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    hemisphere_size: int | None = None,
    hemisphere_overlap_degrees: float = 15.0,
    hemisphere_color_match_degrees: float = 6.0,
    hemisphere_seam_residual_degrees: float = 2.0,
    hemisphere_seam_residual_blur_degrees: float = 0.5,
) -> Image.Image:
    """Generate one ERP with two charts, an equator hard cut, and seam repair."""
    content = content.convert("RGB")
    source_size = content.size
    if source_size[0] < 32 or source_size[1] < 16:
        raise ValueError("Content panorama must be at least 32x16 pixels.")
    if source_size[0] != 2 * source_size[1]:
        raise ValueError("Content panorama must be a 2:1 ERP.")
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if not 0 < hemisphere_overlap_degrees < 45:
        raise ValueError("hemisphere-overlap-degrees must be between zero and 45.")
    if not 0.0 <= hemisphere_color_match_degrees <= hemisphere_overlap_degrees:
        raise ValueError(
            "hemisphere-color-match-degrees must be between zero and "
            "hemisphere-overlap-degrees."
        )
    if not 0.0 <= hemisphere_seam_residual_degrees <= hemisphere_overlap_degrees:
        raise ValueError(
            "hemisphere-seam-residual-degrees must be between zero and "
            "hemisphere-overlap-degrees."
        )

    if hemisphere_size is None:
        chart_size = _nearest_multiple_of_16(source_size[1])
    elif hemisphere_size <= 0 or hemisphere_size % 16:
        raise ValueError("hemisphere-size must be positive and divisible by 16.")
    else:
        chart_size = hemisphere_size

    output_height = _nearest_multiple_of_16(source_size[1])
    output_width = 2 * output_height
    content_north = extract_stereographic_hemisphere(
        content, chart_size, hemisphere_overlap_degrees, north=True
    )
    content_south = extract_stereographic_hemisphere(
        content, chart_size, hemisphere_overlap_degrees, north=False
    )
    working_style = style.convert("RGB").resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    generated, hard_cut = engine.inference_with_hemisphere_rgb_hard_cut(
        prompt, content_north, content_south, working_style, seed, steps,
        output_height, output_width, hemisphere_overlap_degrees,
        color_match_degrees=hemisphere_color_match_degrees,
        seam_residual_degrees=hemisphere_seam_residual_degrees,
        seam_residual_blur_degrees=hemisphere_seam_residual_blur_degrees,
    )
    if generated.size != source_size:
        generated = generated.resize(source_size, Image.Resampling.LANCZOS)
        hard_cut = hard_cut.resize(source_size, Image.Resampling.LANCZOS)
    return _restore_outside_equatorial_band(
        generated, hard_cut,
        max(hemisphere_color_match_degrees, hemisphere_seam_residual_degrees),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one ERP for every style image in a directory"
    )
    parser.add_argument("--content", required=True, help="Input 2:1 ERP panorama")
    parser.add_argument(
        "--style-dir", default="inputs/style",
        help="Directory of style reference images",
    )
    parser.add_argument(
        "--output-dir", default="qwen_style_output",
        help="Directory for generated ERP images",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Editing instruction")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--hemisphere-size", type=int, default=None,
        help="Square chart size; defaults to the aligned ERP height",
    )
    parser.add_argument(
        "--hemisphere-overlap-degrees", type=float, default=15.0,
        help="Chart overlap across the equator",
    )
    parser.add_argument(
        "--hemisphere-color-match-degrees", type=float, default=6.0,
        help="Low-frequency color matching half-width; zero disables it",
    )
    parser.add_argument(
        "--hemisphere-seam-residual-degrees", type=float, default=2.0,
        help="Equator residual correction half-width; zero disables it",
    )
    parser.add_argument(
        "--hemisphere-seam-residual-blur-degrees", type=float, default=0.5,
        help="Longitude smoothing scale for equator residual correction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content_path = Path(args.content)
    if not content_path.is_file():
        raise FileNotFoundError(f"content image does not exist: {content_path}")
    style_paths = _style_paths(Path(args.style_dir))
    with Image.open(content_path) as image:
        content = image.convert("RGB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = ImageStyleInference()
    for index, style_path in enumerate(style_paths, start=1):
        with Image.open(style_path) as image:
            style = image.convert("RGB")
        with torch.no_grad():
            result = stylize_panorama(
                engine=engine, content=content, style=style,
                prompt=args.prompt, seed=args.seed, steps=args.steps,
                hemisphere_size=args.hemisphere_size,
                hemisphere_overlap_degrees=args.hemisphere_overlap_degrees,
                hemisphere_color_match_degrees=args.hemisphere_color_match_degrees,
                hemisphere_seam_residual_degrees=(
                    args.hemisphere_seam_residual_degrees
                ),
                hemisphere_seam_residual_blur_degrees=(
                    args.hemisphere_seam_residual_blur_degrees
                ),
            )
        output_path = output_dir / f"{style_path.stem}_erp.png"
        result.save(output_path)
        print(
            f"[{index}/{len(style_paths)}] Saved {output_path} | "
            f"style={style_path.name} | input={content.size}"
        )


if __name__ == "__main__":
    main()
