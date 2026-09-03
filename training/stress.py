"""Deterministic high-frequency ERP stress images for A1."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_stress_erps(root: Path, width: int = 1024) -> list[Path]:
    if width <= 0 or width % 2:
        raise ValueError("stress ERP width must be positive and even.")
    root.mkdir(parents=True, exist_ok=True)
    height = width // 2
    outputs = []

    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]
    patterns = {
        "checker_4": ((x // 4 + y // 4) % 2) * 255,
        "checker_16": ((x // 16 + y // 16) % 2) * 255,
        "vertical_lines": ((x % 12) < 2) * 255 + np.zeros((height, 1)),
        "horizontal_lines": ((y % 12) < 2) * 255 + np.zeros((1, width)),
        "diagonal_lines": (((x + y) % 20) < 3) * 255,
        "sine_multifrequency": (
            127.5 + 42.5 * np.sin(x * 2 * math.pi / 8)
            + 42.5 * np.sin(x * 2 * math.pi / 31)
            + 42.5 * np.sin(y * 2 * math.pi / 17)
        ),
    }
    for name, pattern in patterns.items():
        gray = np.clip(pattern, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        path = root / f"{name}.png"
        Image.fromarray(rgb, "RGB").save(path)
        outputs.append(path)

    text_path = root / "fine_text_and_curves.png"
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(height // 32):
        draw.text((8 + (row % 3) * 37, row * 32 + height // 2 - 80), "TeleStyle ERP 0123456789", fill="black")
    for offset in range(-80, 81, 8):
        points = [
            (column, int(height / 2 + offset + 35 * math.sin(column / 40)))
            for column in range(width)
        ]
        draw.line(points, fill=(20, 20, 20), width=1)
    image.save(text_path)
    outputs.append(text_path)
    return outputs
