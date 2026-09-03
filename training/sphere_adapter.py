"""Trainable cross-chart SphereAdapter with zero-gated residual injection."""

from __future__ import annotations

import math

import torch
from torch import nn

from .geometry import SphereGeometry


class SphericalPositionEncoder(nn.Module):
    def __init__(self, adapter_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(9, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim),
        )

    def forward(
        self, directions: torch.Tensor, overlap: torch.Tensor, chart_id: float,
    ) -> torch.Tensor:
        longitude = torch.atan2(directions[:, 0], directions[:, 2])
        latitude = directions[:, 1].clamp(-1, 1).asin()
        features = torch.cat(
            (
                directions,
                longitude.sin()[:, None],
                longitude.cos()[:, None],
                latitude[:, None],
                latitude.abs()[:, None],
                overlap.float()[:, None],
                torch.full_like(latitude[:, None], chart_id),
            ),
            dim=-1,
        )
        return self.projection(features.to(self.projection[0].weight.dtype))


class CorrespondenceAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.angular_bias = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        indices: torch.Tensor,
        cosine: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(query_tokens)
        key = self.key(source_tokens)[:, indices]
        value = self.value(source_tokens)[:, indices]
        logits = (query[:, :, None] * key).sum(dim=-1) / math.sqrt(query.shape[-1])
        angular_distance = (1.0 - cosine)[None, :, :, None].to(query.dtype)
        logits = logits + self.angular_bias(angular_distance).squeeze(-1)
        attended = (logits.softmax(dim=-1)[..., None] * value).sum(dim=2)
        attended = self.output(attended)
        return attended * overlap[None, :, None].to(attended.dtype)


class SphereAdapter(nn.Module):
    """One shared recurrent adapter used at multiple frozen DiT depths."""

    def __init__(
        self,
        hidden_dim: int = 3072,
        adapter_dim: int = 512,
        num_heads: int = 8,
        global_tokens: int = 32,
        equator_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.position = SphericalPositionEncoder(adapter_dim)
        self.position_up = nn.Linear(adapter_dim, hidden_dim)
        self.position_gate = nn.Parameter(torch.zeros(()))

        self.global_seed = nn.Parameter(torch.randn(global_tokens, adapter_dim) * 0.02)
        self.equator_seed = nn.Parameter(torch.randn(equator_tokens, adapter_dim) * 0.02)
        self.global_gather = nn.MultiheadAttention(
            adapter_dim, num_heads, batch_first=True
        )
        self.global_broadcast = nn.MultiheadAttention(
            adapter_dim, num_heads, batch_first=True
        )
        self.equator_gather = nn.MultiheadAttention(
            adapter_dim, num_heads, batch_first=True
        )
        self.equator_broadcast = nn.MultiheadAttention(
            adapter_dim, num_heads, batch_first=True
        )
        self.correspondence = CorrespondenceAttention(adapter_dim)
        self.output_norm = nn.LayerNorm(adapter_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )

    def add_position(
        self,
        north: torch.Tensor,
        south: torch.Tensor,
        geometry: SphereGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        north_id = float(torch.sign(geometry.north_directions[:, 1].mean()))
        south_id = float(torch.sign(geometry.south_directions[:, 1].mean()))
        north_position = self.position(
            geometry.north_directions, geometry.north_overlap, north_id
        )
        south_position = self.position(
            geometry.south_directions, geometry.south_overlap, south_id
        )
        scale = torch.tanh(self.position_gate)
        return (
            north + scale * self.position_up(north_position)[None],
            south + scale * self.position_up(south_position)[None],
        )

    def forward(
        self,
        north: torch.Tensor,
        south: torch.Tensor,
        geometry: SphereGeometry,
        global_state: torch.Tensor | None = None,
        equator_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        north_small = self.down(self.input_norm(north))
        south_small = self.down(self.input_norm(south))
        batch = north.shape[0]
        if global_state is None:
            global_state = self.global_seed[None].expand(batch, -1, -1)
        if equator_state is None:
            equator_state = self.equator_seed[None].expand(batch, -1, -1)

        combined = torch.cat((north_small, south_small), dim=1)
        global_update, _ = self.global_gather(
            global_state, combined, combined, need_weights=False
        )
        global_state = global_state + global_update
        north_global, _ = self.global_broadcast(
            north_small, global_state, global_state, need_weights=False
        )
        south_global, _ = self.global_broadcast(
            south_small, global_state, global_state, need_weights=False
        )

        overlap_source = torch.cat(
            (
                north_small * geometry.north_overlap[None, :, None],
                south_small * geometry.south_overlap[None, :, None],
            ),
            dim=1,
        )
        valid_overlap = torch.cat(
            (geometry.north_overlap, geometry.south_overlap), dim=0
        )[None].expand(batch, -1)
        if valid_overlap.any():
            equator_update, _ = self.equator_gather(
                equator_state, overlap_source, overlap_source,
                key_padding_mask=~valid_overlap, need_weights=False,
            )
            equator_state = equator_state + equator_update
        north_equator, _ = self.equator_broadcast(
            north_small, equator_state, equator_state, need_weights=False
        )
        south_equator, _ = self.equator_broadcast(
            south_small, equator_state, equator_state, need_weights=False
        )
        north_equator = north_equator * geometry.north_overlap[None, :, None]
        south_equator = south_equator * geometry.south_overlap[None, :, None]

        north_corr = self.correspondence(
            north_small,
            south_small,
            geometry.north_to_south_indices,
            geometry.north_to_south_cosine,
            geometry.north_overlap,
        )
        south_corr = self.correspondence(
            south_small,
            north_small,
            geometry.south_to_north_indices,
            geometry.south_to_north_cosine,
            geometry.south_overlap,
        )
        north_update = self.output_norm(
            north_global + north_equator + north_corr
        )
        south_update = self.output_norm(
            south_global + south_equator + south_corr
        )
        north_update = north_update + self.feed_forward(north_update)
        south_update = south_update + self.feed_forward(south_update)
        return (
            self.up(north_update),
            self.up(south_update),
            global_state,
            equator_state,
        )
