"""Pixel-centered SpheRoPE adaptation for Qwen Image Edit.

See https://arxiv.org/html/2606.32033v1, Section 3.2. Only the width
axis changes; the low-frequency subspace converges at the polar limit.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn


def spherical_width_phases(
    height, width, dim, theta=10000, *, device=None, tolerance=0.10,
    centered=True, rows=None, columns=None,
):
    """Return width phases [H, W, dim/2] at ERP token pixel centers.

    Optional fractional rows/columns allow evaluation at the pole limits
    and beyond the wrap boundary. Frequencies use Qwen's original ladder.
    """
    if height <= 0 or width <= 0 or dim <= 0 or dim % 2:
        raise ValueError("Grid dimensions must be positive and dim must be even.")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive.")
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between zero and one.")
    rows = torch.arange(height, device=device) if rows is None else rows
    columns = torch.arange(width, device=device) if columns is None else columns
    rows = torch.as_tensor(rows, device=device, dtype=torch.float32).reshape(-1)
    columns = torch.as_tensor(columns, device=device, dtype=torch.float32).reshape(-1)
    frequencies = 1.0 / torch.pow(
        theta, torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    fundamental = 2 * math.pi / width
    cycles = frequencies / fundamental
    quantizable = (cycles >= 1) & (
        (cycles - cycles.round()).abs() / cycles <= tolerance
    )
    # Once a channel fails, all lower frequencies use the spherical path.
    cyclic = quantizable.to(torch.int64).cumprod(0).bool()
    latitude = math.pi / 2 - (rows + 0.5) * (math.pi / height)
    longitude = (columns + 0.5) * (2 * math.pi / width) - math.pi
    radius = width / 2
    offset = width - width // 2 if centered else 0
    x = (latitude.cos()[:, None] * longitude.cos()[None, :] + 1) * radius - offset
    y = (latitude.cos()[:, None] * longitude.sin()[None, :] + 1) * radius - offset
    slots = torch.arange(dim // 2, device=device)
    spherical = torch.where(slots % 2 == 0, x[..., None], y[..., None]) * frequencies
    linear = (columns - offset)[None, :, None] * (cycles.round() * fundamental)
    return torch.where(cyclic, linear, spherical)


class QwenSphericalRoPE(nn.Module):
    """Replace output/content width phases while preserving other embeddings."""

    def __init__(self, original):
        super().__init__()
        if list(original.axes_dim) != [16, 56, 56]:
            raise ValueError("SpheRoPE requires Qwen RoPE axes [16, 56, 56].")
        self.original = original
        self._width_cache = {}

    def forward(self, image_shapes, text_lengths, device):
        if len(image_shapes) != 3 or image_shapes[0] != image_shapes[1]:
            raise ValueError("SpheRoPE requires matching output/content grids and one style image.")
        if any(frame != 1 for frame, _, _ in image_shapes):
            raise ValueError("SpheRoPE currently supports batch size one.")
        if image_shapes[0][2] != 2 * image_shapes[0][1]:
            raise ValueError("SpheRoPE requires a 2:1 ERP token grid.")
        image, text = self.original(image_shapes, text_lengths, device)
        result = image.clone()
        start = 0
        width_start = sum(self.original.axes_dim[:2]) // 2
        for index, (frame, height, width) in enumerate(image_shapes):
            count = frame * height * width
            if index < 2:
                key = (height, width, str(image.device), image.dtype)
                if key not in self._width_cache:
                    phases = spherical_width_phases(
                        height, width, self.original.axes_dim[2], self.original.theta,
                        device=image.device, centered=self.original.scale_rope,
                    )
                    self._width_cache[key] = torch.polar(
                        torch.ones_like(phases), phases
                    ).reshape(count, -1).to(image.dtype)
                result[start:start + count, width_start:] = self._width_cache[key]
            start += count
        return result, text


@contextmanager
def spherical_rope_context(dit, enabled=True):
    """Scope positional encoding replacement to a single inference call."""
    if not enabled:
        yield
        return
    original = dit.pos_embed
    dit.pos_embed = QwenSphericalRoPE(original)
    try:
        yield
    finally:
        dit.pos_embed = original
