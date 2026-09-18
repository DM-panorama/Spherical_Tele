"""Spherical geometry shared by chart inference and A1 training.

All resampling uses ERP pixel centres and unit-sphere directions.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image



def _erp_directions(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Return unit vectors for ERP pixel centres, shaped ``[H, W, 3]``."""
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    latitude = math.pi / 2 - math.pi * (y[:, None] + 0.5) / height
    longitude = 2 * math.pi * (x[None, :] + 0.5) / width - math.pi
    cos_latitude = torch.cos(latitude)
    return torch.stack(
        (
            cos_latitude * torch.sin(longitude),
            torch.sin(latitude).expand(height, width),
            cos_latitude * torch.cos(longitude),
        ),
        dim=-1,
    )


def _directions_to_grid(directions: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert unit vectors to a grid for circular-horizontal ERP sampling."""
    x, y, z = directions.unbind(dim=-1)
    longitude = torch.atan2(x, z)
    latitude = torch.asin(y.clamp(-1.0, 1.0))
    source_x = ((longitude + math.pi) / (2 * math.pi) * width - 0.5).remainder(width)
    source_y = (math.pi / 2 - latitude) / math.pi * height - 0.5

    # The source is padded by one circular column on each side.  align_corners
    # is false, so normalized coordinates are derived from pixel centres.
    grid_x = 2 * ((source_x + 1.0) + 0.5) / (width + 2) - 1.0
    grid_y = 2 * (source_y + 0.5) / height - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def _sample_circular_erp(source: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``source`` with circular longitude handling."""
    if source.ndim != 4:
        raise ValueError("source must have shape [batch, channels, height, width].")
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [output_height, output_width, 2].")
    padded = F.pad(source, (1, 1, 0, 0), mode="circular")
    return F.grid_sample(
        padded,
        grid.unsqueeze(0).expand(source.shape[0], -1, -1, -1).to(source.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _sample_tensor(source: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a tensor using an ``align_corners=False`` grid."""
    if source.ndim != 4:
        raise ValueError("source must have shape [batch, channels, height, width].")
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [output_height, output_width, 2].")
    return F.grid_sample(
        source,
        grid.unsqueeze(0).expand(source.shape[0], -1, -1, -1).to(source.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _polar_stereographic_directions(size: int, cap_degrees: float, north: bool, device: torch.device) -> torch.Tensor:
    radius = math.tan(math.radians(cap_degrees) / 2.0)
    axis = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) * (2.0 / size) - 1.0
    u, v = torch.meshgrid(axis * radius, axis * radius, indexing="xy")
    r = torch.sqrt(u.square() + v.square())
    theta = 2.0 * torch.atan(r)
    scale = torch.where(r > 0, torch.sin(theta) / r, torch.zeros_like(r))
    y = torch.cos(theta)
    if not north:
        y = -y
    return torch.stack((u * scale, y, v * scale), dim=-1)

def _directions_to_stereographic_grid(
    directions: torch.Tensor,
    cap_degrees: float,
    north: bool,
) -> torch.Tensor:
    """Map unit directions to a north- or south-centred stereographic grid."""
    x, y, z = directions.unbind(dim=-1)
    denominator = 1.0 + y if north else 1.0 - y
    radius = math.tan(math.radians(cap_degrees) / 2.0)
    return torch.stack(
        (
            x / denominator.clamp_min(1e-6) / radius,
            z / denominator.clamp_min(1e-6) / radius,
        ),
        dim=-1,
    )


def _north_confidence(
    latitude_degrees: torch.Tensor, overlap_degrees: float,
) -> torch.Tensor:
    """Return smooth south-to-north confidence across the equatorial overlap."""
    phase = latitude_degrees.clamp(-overlap_degrees, overlap_degrees)
    phase = phase * (math.pi / (2.0 * overlap_degrees))
    return 0.5 + 0.5 * torch.sin(phase)


def extract_stereographic_hemisphere(
    image: Image.Image,
    size: int,
    overlap_degrees: float,
    north: bool,
) -> Image.Image:
    """Project an ERP into a square polar chart extending across the equator."""
    if size <= 0 or size % 16:
        raise ValueError("hemisphere size must be positive and divisible by 16.")
    if not 0 < overlap_degrees < 45:
        raise ValueError("hemisphere overlap must be between zero and 45 degrees.")
    image = image.convert("RGB")
    source = (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    cap_degrees = 90.0 + overlap_degrees
    directions = _polar_stereographic_directions(
        size, cap_degrees, north, source.device
    )
    chart = _sample_circular_erp(
        source, _directions_to_grid(directions, image.height, image.width)
    )
    array = (
        chart[0].permute(1, 2, 0).clamp(0, 1) * 255
    ).round().byte().numpy()
    return Image.fromarray(array, "RGB")


class HemisphereLatentProjector:
    """Cached grids for synchronizing two overlapping stereographic charts."""

    def __init__(
        self,
        chart_size: int,
        overlap_degrees: float,
        device: torch.device,
    ) -> None:
        if chart_size <= 0:
            raise ValueError("chart_size must be positive.")
        if not 0 < overlap_degrees < 45:
            raise ValueError("hemisphere overlap must be between zero and 45 degrees.")
        self.chart_size = chart_size
        self.overlap_degrees = overlap_degrees
        self.cap_degrees = 90.0 + overlap_degrees

        north_directions = _polar_stereographic_directions(
            chart_size, self.cap_degrees, True, device
        )
        south_directions = _polar_stereographic_directions(
            chart_size, self.cap_degrees, False, device
        )
        self.south_to_north_grid = _directions_to_stereographic_grid(
            north_directions, self.cap_degrees, False
        )
        self.north_to_south_grid = _directions_to_stereographic_grid(
            south_directions, self.cap_degrees, True
        )
        north_latitude = north_directions[..., 1].asin() * (180.0 / math.pi)
        south_latitude = south_directions[..., 1].asin() * (180.0 / math.pi)
        self.north_overlap = (
            north_latitude.abs() <= overlap_degrees
        ).unsqueeze(0).unsqueeze(0)
        self.south_overlap = (
            south_latitude.abs() <= overlap_degrees
        ).unsqueeze(0).unsqueeze(0)
        self.north_weight = _north_confidence(
            north_latitude, overlap_degrees
        ).unsqueeze(0).unsqueeze(0)
        self.south_weight = _north_confidence(
            south_latitude, overlap_degrees
        ).unsqueeze(0).unsqueeze(0)

    def synchronize(
        self,
        north: torch.Tensor,
        south: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return order-independent latent charts sharing one overlap field."""
        expected = (self.chart_size, self.chart_size)
        if north.ndim != 4 or south.ndim != 4:
            raise ValueError("hemisphere latents must have four dimensions.")
        if north.shape != south.shape or north.shape[-2:] != expected:
            raise ValueError("hemisphere latents must have identical square shapes.")

        north_source = north.clone()
        south_source = south.clone()
        south_on_north = _sample_tensor(south_source, self.south_to_north_grid)
        north_on_south = _sample_tensor(north_source, self.north_to_south_grid)
        north_weight = self.north_weight.to(north.device, north.dtype)
        south_weight = self.south_weight.to(south.device, south.dtype)
        north_fused = (
            north_source * north_weight + south_on_north * (1.0 - north_weight)
        )
        south_fused = (
            north_on_south * south_weight + south_source * (1.0 - south_weight)
        )
        north_mask = self.north_overlap.to(north.device)
        south_mask = self.south_overlap.to(south.device)
        return (
            torch.where(north_mask, north_fused, north_source),
            torch.where(south_mask, south_fused, south_source),
        )
