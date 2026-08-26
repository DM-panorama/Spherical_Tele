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

def make_spherical_latent_canvas(
    centre: torch.Tensor,
    horizontal_padding: int,
    vertical_padding: int,
) -> torch.Tensor:
    """Pad an ERP latent using exact horizontal and cross-pole topology."""
    if centre.ndim != 4:
        raise ValueError("centre must have shape [batch, channels, height, width].")
    if horizontal_padding < 0 or vertical_padding < 0:
        raise ValueError("spherical padding widths cannot be negative.")
    if horizontal_padding > centre.shape[-1] or vertical_padding > centre.shape[-2]:
        raise ValueError("spherical padding cannot exceed the corresponding ERP dimension.")
    if centre.shape[-1] % 2:
        raise ValueError("ERP latent width must be even for half-turn polar padding.")

    if vertical_padding:
        half_turn = centre.shape[-1] // 2
        north = torch.roll(
            centre[..., :vertical_padding, :].flip(-2), shifts=half_turn, dims=-1
        )
        south = torch.roll(
            centre[..., -vertical_padding:, :].flip(-2), shifts=half_turn, dims=-1
        )
        vertically_padded = torch.cat((north, centre, south), dim=-2)
    else:
        vertically_padded = centre
    return make_circular_latent_canvas(
        vertically_padded, horizontal_padding, horizontal_padding
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

    def compose_erp(
        self,
        north: torch.Tensor,
        south: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """Reproject synchronized charts into one ERP latent before decoding."""
        if output_height <= 0 or output_width <= 0:
            raise ValueError("ERP output dimensions must be positive.")
        if output_width % 2:
            raise ValueError("ERP output width must be even.")
        if (
            north.shape != south.shape
            or north.shape[-2:] != (self.chart_size, self.chart_size)
        ):
            raise ValueError("hemisphere latents must match the projector chart size.")
        directions = _erp_directions(output_height, output_width, north.device)
        north_grid = _directions_to_stereographic_grid(
            directions, self.cap_degrees, True
        )
        south_grid = _directions_to_stereographic_grid(
            directions, self.cap_degrees, False
        )
        north_erp = _sample_tensor(north, north_grid)
        south_erp = _sample_tensor(south, south_grid)
        latitude = directions[..., 1].asin() * (180.0 / math.pi)
        weight = _north_confidence(latitude, self.overlap_degrees)
        weight = weight.unsqueeze(0).unsqueeze(0).to(north.dtype)
        return north_erp * weight + south_erp * (1.0 - weight)


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
