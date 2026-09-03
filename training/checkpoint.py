"""Adapter-only checkpoints, EMA, and reproducibility metadata."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdapterEma:
    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one.")
        self.decay = decay
        self.shadow = {
            name: value.detach().float().cpu().clone()
            for name, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            target = self.shadow[name]
            target.mul_(self.decay).add_(
                value.detach().float().cpu(), alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]


def save_adapter_checkpoint(
    path: Path,
    adapter: nn.Module,
    injection_gates: torch.Tensor,
    ema: AdapterEma,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "telestyle-sphere-adapter-a1-v1",
        "step": int(step),
        "adapter": adapter.state_dict(),
        "injection_gates": injection_gates.detach().cpu(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_adapter_checkpoint(
    path: Path,
    adapter: nn.Module,
    injection_gates: torch.Tensor,
    ema: AdapterEma | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "telestyle-sphere-adapter-a1-v1":
        raise ValueError("unsupported SphereAdapter checkpoint format.")
    adapter.load_state_dict(payload["adapter"], strict=True)
    with torch.no_grad():
        injection_gates.copy_(
            payload["injection_gates"].to(
                injection_gates.device, injection_gates.dtype
            )
        )
    if ema is not None:
        ema.load_state_dict(payload["ema"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"]), dict(payload["metadata"])

def validate_adapter_metadata(
    metadata: dict[str, Any],
    *,
    base_transformer_sha256: str | None = None,
    lora_sha256: dict[str, str] | None = None,
    prompt: str | None = None,
    lightning_timesteps: list[float] | None = None,
    south_yaw_degrees: float | None = None,
) -> None:
    """Validate pilot dependencies; legacy checkpoints without schema stay valid."""
    if int(metadata.get("schema_version", 1)) < 2:
        return
    checks = {
        "base_transformer_index_sha256": base_transformer_sha256,
        "lora_sha256": lora_sha256,
        "prompt": prompt,
        "lightning_timesteps": lightning_timesteps,
        "south_yaw_degrees": south_yaw_degrees,
    }
    for key, actual in checks.items():
        if actual is None:
            continue
        expected = metadata.get(key)
        if expected != actual:
            raise ValueError(
                f"A1 checkpoint metadata mismatch for {key}: "
                f"checkpoint={expected!r}, runtime={actual!r}."
            )


def dependency_hashes(paths: list[Path]) -> dict[str, str]:
    """Return stable filename-to-SHA256 metadata for configured LoRAs."""
    return {path.name: sha256_file(path) for path in paths}
