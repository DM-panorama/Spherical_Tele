"""Spherical ERP reprojection utilities used by panorama inference.

All resampling is performed from equirectangular pixel centres through unit
sphere coordinates.  This deliberately avoids image-space ``rot90`` shortcuts,
which do not represent a rotation on an equirectangular panorama.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _rotation_x(directions: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """Rotate ``[..., 3]`` unit vectors about the fixed X axis."""
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = directions.unbind(dim=-1)
    return torch.stack((x, cosine * y - sine * z, sine * y + cosine * z), dim=-1)


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
    if grid.shape[:2] != source.shape[-2:] or grid.shape[-1] != 2:
        raise ValueError("grid shape must be [height, width, 2] for source.")
    padded = F.pad(source, (1, 1, 0, 0), mode="circular")
    return F.grid_sample(
        padded,
        grid.unsqueeze(0).expand(source.shape[0], -1, -1, -1).to(source.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def make_circular_latent_canvas(
    centre: torch.Tensor, left_width: int, right_width: int,
) -> torch.Tensor:
    """Build a ``[right | centre | left]`` latent canvas from an ERP centre."""
    if centre.ndim != 4:
        raise ValueError("centre must have shape [batch, channels, height, width].")
    if left_width < 0 or right_width < 0:
        raise ValueError("extension widths cannot be negative.")
    if left_width > centre.shape[-1] or right_width > centre.shape[-1]:
        raise ValueError("extension widths cannot exceed the centre width.")
    return torch.cat(
        (centre[..., -left_width:] if left_width else centre[..., :0], centre,
         centre[..., :right_width]),
        dim=-1,
    )


def make_polar_latitude_weight(
    height: int,
    width: int,
    start_degrees: float,
    end_degrees: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a cosine-ramped polar weight for an ERP latent grid."""
    if height <= 0 or width <= 0:
        raise ValueError("latent height and width must be positive.")
    if not (0 <= start_degrees < end_degrees < 90):
        raise ValueError("polar degrees must satisfy 0 <= start < end < 90.")
    directions = _erp_directions(height, width, device)
    latitude = directions[..., 1].asin().abs() * (180.0 / math.pi)
    transition = ((latitude - start_degrees) / (end_degrees - start_degrees)).clamp(0, 1)
    return (0.5 - 0.5 * torch.cos(math.pi * transition)).unsqueeze(0).unsqueeze(0)


def limit_polar_latent_detail(
    source: torch.Tensor,
    polar_weight: torch.Tensor,
    max_radius: int,
    pole_mean_rows: int = 1,
) -> torch.Tensor:
    """Suppress oversampled polar longitude detail while preserving the equator."""
    if pole_mean_rows < 0:
        raise ValueError("pole_mean_rows cannot be negative.")
    limited = latitude_adaptive_circular_lowpass(source, polar_weight, max_radius)
    if pole_mean_rows == 0:
        return limited

    row_weight = polar_weight[0, 0, :, 0]
    fully_polar = row_weight >= (1.0 - 1e-6)
    top_rows = 0
    while top_rows < min(pole_mean_rows, source.shape[-2]) and fully_polar[top_rows]:
        top_rows += 1
    bottom_rows = 0
    while bottom_rows < min(pole_mean_rows, source.shape[-2]) and fully_polar[-1 - bottom_rows]:
        bottom_rows += 1
    if top_rows == 0 and bottom_rows == 0:
        return limited

    result = limited.clone()
    if top_rows:
        top_mean = result[..., :top_rows, :].float().mean(dim=-1, keepdim=True).to(result.dtype)
        result[..., :top_rows, :] = top_mean.expand_as(result[..., :top_rows, :])
    if bottom_rows:
        bottom_mean = result[..., -bottom_rows:, :].float().mean(dim=-1, keepdim=True).to(result.dtype)
        result[..., -bottom_rows:, :] = bottom_mean.expand_as(result[..., -bottom_rows:, :])
    return result


def latitude_adaptive_circular_lowpass(
    source: torch.Tensor,
    polar_weight: torch.Tensor,
    max_radius: int,
) -> torch.Tensor:
    """Circularly low-pass ERP longitude with strength increasing toward poles."""
    if source.ndim != 4:
        raise ValueError("source must have shape [batch, channels, height, width].")
    if max_radius < 0:
        raise ValueError("max_radius cannot be negative.")
    if polar_weight.ndim != 4 or polar_weight.shape[-2:] != source.shape[-2:]:
        raise ValueError("polar_weight must match source height and width.")
    if polar_weight.shape[0] not in (1, source.shape[0]) or polar_weight.shape[1] not in (1, source.shape[1]):
        raise ValueError("polar_weight batch and channel dimensions must broadcast to source.")
    radius = min(max_radius, max(0, (source.shape[-1] - 1) // 2))
    if radius == 0:
        return source

    working = source.float()
    offsets = torch.arange(-radius, radius + 1, device=source.device, dtype=torch.float32)
    sigma = max(radius / 2.0, 0.5)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, 1, -1)
    kernel = kernel.expand(source.shape[1], 1, 1, -1)
    padded = F.pad(working, (radius, radius, 0, 0), mode="circular")
    blurred = F.conv2d(padded, kernel, groups=source.shape[1])
    weight = polar_weight.to(device=source.device, dtype=working.dtype)
    return (working * (1.0 - weight) + blurred * weight).to(dtype=source.dtype)


def early_polar_guidance_strength(
    progress_id: int,
    guidance_steps: int,
    strength: float = 1.0,
) -> float:
    """Return the early-step B→A prediction guidance coefficient."""
    if progress_id < 0:
        raise ValueError("progress_id cannot be negative.")
    if guidance_steps <= 0:
        raise ValueError("guidance_steps must be greater than zero.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between zero and one.")
    if progress_id >= guidance_steps:
        return 0.0
    if progress_id == 0:
        return 0.35 * strength
    return 0.20 * (0.6 ** (progress_id - 1)) * strength


def make_rotated_latent_canvas(
    latents_a: torch.Tensor,
    projector: "SphericalLatentProjector",
    centre_x: int,
    centre_width: int,
) -> torch.Tensor:
    """Rotate A's centre ERP latents into a wrapped B latent canvas."""
    if latents_a.ndim != 4:
        raise ValueError("latents_a must have shape [batch, channels, height, width].")
    if centre_x < 0 or centre_width <= 0:
        raise ValueError("centre latent coordinates must be non-negative and non-empty.")
    right_width = latents_a.shape[-1] - centre_x - centre_width
    if right_width < 0:
        raise ValueError("centre latent region does not fit inside latents_a.")
    centre_a = latents_a[..., centre_x : centre_x + centre_width]
    if centre_a.shape[-1] != centre_width:
        raise ValueError("centre latent region has an unexpected width.")
    centre_b = projector.a_to_b(centre_a)
    return make_circular_latent_canvas(centre_b, centre_x, right_width)


class SphericalLatentProjector:
    """Cached A↔B ERP grids and latitude weights for one latent resolution."""

    def __init__(
        self,
        height: int,
        width: int,
        rotation_degrees: float,
        polar_blend_start_degrees: float,
        polar_blend_end_degrees: float,
        device: torch.device,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("latent height and width must be positive.")
        if not math.isfinite(rotation_degrees):
            raise ValueError("spherical rotation angle must be finite.")
        if not (0 <= polar_blend_start_degrees < polar_blend_end_degrees < 90):
            raise ValueError("polar blend degrees must satisfy 0 <= start < end < 90.")

        directions = _erp_directions(height, width, device)
        # B(d) = A(R^-1 d), while returning B to A samples B(R d).
        self.a_to_b_grid = _directions_to_grid(
            _rotation_x(directions, -rotation_degrees), height, width
        )
        self.b_to_a_grid = _directions_to_grid(
            _rotation_x(directions, rotation_degrees), height, width
        )
        latitude = directions[..., 1].asin().abs() * (180.0 / math.pi)
        transition = ((latitude - polar_blend_start_degrees) /
                      (polar_blend_end_degrees - polar_blend_start_degrees)).clamp(0, 1)
        self.polar_weight = (0.5 - 0.5 * torch.cos(math.pi * transition)).unsqueeze(0).unsqueeze(0)

    def a_to_b(self, latents: torch.Tensor) -> torch.Tensor:
        return _sample_circular_erp(latents, self.a_to_b_grid)

    def b_to_a(self, latents: torch.Tensor) -> torch.Tensor:
        return _sample_circular_erp(latents, self.b_to_a_grid)

    def fuse_a_with_b(
        self,
        latents_a: torch.Tensor,
        latents_b_aligned: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if latents_a.shape != latents_b_aligned.shape:
            raise ValueError("A and aligned B latents must have identical shapes.")
        if not 0 <= strength <= 1:
            raise ValueError("fusion strength must be between zero and one.")
        weight = self.polar_weight.to(device=latents_a.device, dtype=latents_a.dtype) * strength
        return latents_a * (1 - weight) + latents_b_aligned * weight


def rotate_erp_image(image: Image.Image, rotation_degrees: float) -> Image.Image:
    """Rotate an RGB ERP about the X axis using spherical resampling."""
    image = image.convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    directions = _erp_directions(image.height, image.width, tensor.device)
    grid = _directions_to_grid(
        _rotation_x(directions, -rotation_degrees), image.height, image.width
    )
    rotated = _sample_circular_erp(tensor, grid)
    output = (rotated[0].permute(1, 2, 0).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(output, "RGB")
