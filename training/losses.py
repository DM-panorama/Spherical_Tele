"""Normalized decoded-RGB geometry losses for A1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    mask = mask.expand(value.shape[0], value.shape[1], -1, -1)
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


def charbonnier_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return _masked_mean(torch.sqrt((first - second).square() + epsilon**2), mask)


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = torch.roll(image, shifts=-1, dims=-1) - image
    vertical = image[..., 1:, :] - image[..., :-1, :]
    vertical = F.pad(vertical, (0, 0, 0, 1), mode="replicate")
    return horizontal, vertical


def gradient_consistency_loss(
    first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    first_x, first_y = image_gradients(first)
    second_x, second_y = image_gradients(second)
    return 0.5 * (
        _masked_mean((first_x - second_x).abs(), mask)
        + _masked_mean((first_y - second_y).abs(), mask)
    )


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def laplacian_pyramid_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 3,
) -> torch.Tensor:
    losses = []
    current_first, current_second = first, second
    current_mask = mask.float()
    for _ in range(levels):
        losses.append(
            _masked_mean((_laplacian(current_first) - _laplacian(current_second)).abs(), current_mask)
        )
        if min(current_first.shape[-2:]) < 4:
            break
        current_first = F.avg_pool2d(current_first, 2)
        current_second = F.avg_pool2d(current_second, 2)
        current_mask = F.interpolate(current_mask, size=current_first.shape[-2:], mode="area")
    return torch.stack(losses).mean()


def hard_seam_loss(hard_cut: torch.Tensor) -> torch.Tensor:
    """Penalize abnormal first/second normal differences at the equator."""
    height = hard_cut.shape[-2]
    if height < 6:
        raise ValueError("hard-cut ERP must have at least six rows.")
    split = height // 2
    seam_first = hard_cut[..., split, :] - hard_cut[..., split - 1, :]
    nearby_first = torch.cat(
        (
            hard_cut[..., split - 2, :] - hard_cut[..., split - 3, :],
            hard_cut[..., split + 2, :] - hard_cut[..., split + 1, :],
        ),
        dim=-1,
    )
    first_loss = (seam_first.abs().mean() - nearby_first.abs().mean()).abs()
    seam_second = (
        hard_cut[..., split, :] - 2 * hard_cut[..., split - 1, :]
        + hard_cut[..., split - 2, :]
    )
    return first_loss + 0.5 * seam_second.abs().mean()


def high_frequency_energy(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    horizontal, vertical = image_gradients(image)
    return _masked_mean(horizontal.abs() + vertical.abs(), mask)


def high_frequency_energy_loss(
    hard_cut: torch.Tensor,
    consistency_mask: torch.Tensor,
    near_degrees_mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    overlap_energy = high_frequency_energy(hard_cut, consistency_mask)
    near_energy = high_frequency_energy(hard_cut, near_degrees_mask)
    return (overlap_energy - near_energy).abs() / (near_energy + epsilon)


def geometric_losses(
    north_erp: torch.Tensor,
    south_erp: torch.Tensor,
    hard_cut: torch.Tensor,
    consistency_mask: torch.Tensor,
    latitude_degrees: torch.Tensor,
) -> dict[str, torch.Tensor]:
    near = (
        (latitude_degrees.abs() > 10.0)
        & (latitude_degrees.abs() <= 25.0)
    )[None, None]
    return {
        "rgb": charbonnier_loss(north_erp, south_erp, consistency_mask),
        "gradient": gradient_consistency_loss(
            north_erp, south_erp, consistency_mask
        ),
        "laplacian": laplacian_pyramid_loss(
            north_erp, south_erp, consistency_mask
        ),
        "seam": hard_seam_loss(hard_cut),
        "energy": high_frequency_energy_loss(
            hard_cut, consistency_mask, near
        ),
    }
