"""Differentiable spherical geometry used by A1 training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from telestyle_spherical import (
    _directions_to_grid,
    _directions_to_stereographic_grid,
    _erp_directions,
    _polar_stereographic_directions,
    _sample_circular_erp,
    _sample_tensor,
)


@dataclass(frozen=True)
class SphereGeometry:
    north_directions: torch.Tensor
    south_directions: torch.Tensor
    north_overlap: torch.Tensor
    south_overlap: torch.Tensor
    north_to_south_indices: torch.Tensor
    south_to_north_indices: torch.Tensor
    north_to_south_cosine: torch.Tensor
    south_to_north_cosine: torch.Tensor


def swap_sphere_geometry(geometry: SphereGeometry) -> SphereGeometry:
    """Exchange chart labels while retaining each token sphere direction."""
    return SphereGeometry(
        north_directions=geometry.south_directions,
        south_directions=geometry.north_directions,
        north_overlap=geometry.south_overlap,
        south_overlap=geometry.north_overlap,
        north_to_south_indices=geometry.south_to_north_indices,
        south_to_north_indices=geometry.north_to_south_indices,
        north_to_south_cosine=geometry.south_to_north_cosine,
        south_to_north_cosine=geometry.north_to_south_cosine,
    )


@dataclass(frozen=True)
class ReprojectedCharts:
    north: torch.Tensor
    south: torch.Tensor
    hard_cut: torch.Tensor
    consistency_mask: torch.Tensor
    latitude_degrees: torch.Tensor


def rotation_matrix(
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    device: torch.device,
) -> torch.Tensor:
    """Return an active 3-D rotation in yaw, pitch, roll order."""
    yaw, pitch, roll = [
        math.radians(value) for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    yaw_matrix = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device,
        dtype=torch.float32,
    )
    pitch_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        device=device,
        dtype=torch.float32,
    )
    roll_matrix = torch.tensor(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    return roll_matrix @ pitch_matrix @ yaw_matrix


def rotate_erp(
    source: torch.Tensor,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> torch.Tensor:
    """Rotate an ERP via output pixel centres and inverse sphere sampling."""
    if source.ndim != 4 or source.shape[-1] != 2 * source.shape[-2]:
        raise ValueError("source must have shape [B, C, H, 2H].")
    height, width = source.shape[-2:]
    directions = _erp_directions(height, width, source.device)
    rotation = rotation_matrix(
        yaw_degrees, pitch_degrees, roll_degrees, source.device
    )
    source_directions = directions @ rotation
    grid = _directions_to_grid(source_directions, height, width)
    return _sample_circular_erp(source, grid)


def build_sphere_geometry(
    token_size: int,
    overlap_degrees: float,
    consistency_degrees: float,
    neighbors: int,
    device: torch.device,
) -> SphereGeometry:
    """Build token-centre directions and angular nearest-neighbour maps."""
    if token_size <= 0:
        raise ValueError("token_size must be positive.")
    if not 0 < consistency_degrees <= overlap_degrees < 45:
        raise ValueError("degrees must satisfy 0 < consistency <= overlap < 45.")
    if neighbors <= 0 or neighbors > token_size * token_size:
        raise ValueError("neighbors must fit inside the opposite chart.")
    cap_degrees = 90.0 + overlap_degrees
    north = _polar_stereographic_directions(
        token_size, cap_degrees, True, device
    ).reshape(-1, 3)
    south = _polar_stereographic_directions(
        token_size, cap_degrees, False, device
    ).reshape(-1, 3)
    north_latitude = north[:, 1].asin() * (180.0 / math.pi)
    south_latitude = south[:, 1].asin() * (180.0 / math.pi)
    north_overlap = north_latitude.abs() <= consistency_degrees
    south_overlap = south_latitude.abs() <= consistency_degrees

    similarity = north @ south.transpose(0, 1)
    north_cosine, north_indices = similarity.topk(neighbors, dim=1)
    south_cosine, south_indices = similarity.transpose(0, 1).topk(neighbors, dim=1)
    return SphereGeometry(
        north_directions=north,
        south_directions=south,
        north_overlap=north_overlap,
        south_overlap=south_overlap,
        north_to_south_indices=north_indices,
        south_to_north_indices=south_indices,
        north_to_south_cosine=north_cosine,
        south_to_north_cosine=south_cosine,
    )


def reproject_native_charts(
    north: torch.Tensor,
    south: torch.Tensor,
    overlap_degrees: float,
    consistency_degrees: float,
    output_height: int,
    output_width: int,
    antialias_scale: int = 2,
    south_yaw_degrees: float = 0.0,
) -> ReprojectedCharts:
    """Project native RGB charts to one supersampled ERP grid and hard-cut."""
    if north.ndim != 4 or north.shape != south.shape:
        raise ValueError("north and south must have identical [B, C, H, W] shapes.")
    if north.shape[-1] != north.shape[-2]:
        raise ValueError("native charts must be square.")
    if output_width != 2 * output_height:
        raise ValueError("ERP output must have a 2:1 aspect ratio.")
    if antialias_scale <= 0:
        raise ValueError("antialias_scale must be positive.")
    if not math.isfinite(south_yaw_degrees):
        raise ValueError("south_yaw_degrees must be finite.")

    high_height = output_height * antialias_scale
    high_width = output_width * antialias_scale
    directions = _erp_directions(high_height, high_width, north.device)
    cap_degrees = 90.0 + overlap_degrees
    north_grid = _directions_to_stereographic_grid(
        directions, cap_degrees, True
    )
    south_directions = directions
    if south_yaw_degrees:
        south_directions = directions @ rotation_matrix(
            south_yaw_degrees, 0.0, 0.0, north.device
        )
    south_grid = _directions_to_stereographic_grid(
        south_directions, cap_degrees, False
    )
    north_high = _sample_tensor(north, north_grid)
    south_high = _sample_tensor(south, south_grid)
    latitude_high = directions[..., 1].asin() * (180.0 / math.pi)
    north_owner = latitude_high >= 0
    hard_high = torch.where(
        north_owner.unsqueeze(0).unsqueeze(0), north_high, south_high
    )
    mask_high = latitude_high.abs() <= consistency_degrees

    if antialias_scale > 1:
        size = (output_height, output_width)
        north_erp = F.interpolate(north_high, size=size, mode="area")
        south_erp = F.interpolate(south_high, size=size, mode="area")
        hard_cut = F.interpolate(hard_high, size=size, mode="area")
        mask = F.interpolate(
            mask_high.float()[None, None], size=size, mode="area"
        ) >= 0.5
        latitude = F.interpolate(
            latitude_high[None, None], size=size, mode="bilinear",
            align_corners=False,
        )[0, 0]
    else:
        north_erp, south_erp, hard_cut = north_high, south_high, hard_high
        mask = mask_high[None, None]
        latitude = latitude_high
    return ReprojectedCharts(
        north=north_erp,
        south=south_erp,
        hard_cut=hard_cut,
        consistency_mask=mask,
        latitude_degrees=latitude,
    )
