"""Dataset for immutable A1 VAE/text caches."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class CachedA1Dataset(torch.utils.data.Dataset):
    def __init__(
        self, index_path: Path, split: str,
        conditioning: dict[str, torch.Tensor] | None = None,
    ) -> None:
        with index_path.open(encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        selected = [item for item in items if item["split"] == split]
        self.paths = [Path(item["path"]) for item in selected]
        self.stress_flags = [bool(item.get("stress", False)) for item in selected]
        metadata = json.loads(
            (index_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        self.conditioning = conditioning or torch.load(
            metadata["conditioning"], map_location="cpu", weights_only=False
        )
        if not self.paths:
            raise ValueError(f"cache contains no {split} samples: {index_path}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        payload = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        if payload.get("format") != "telestyle-a1-cache-v1":
            raise ValueError(f"unsupported cache file: {self.paths[index]}")
        for branch_name in ("north", "south"):
            payload[branch_name]["prompt_emb"] = self.conditioning["prompt_emb"]
            payload[branch_name]["prompt_mask"] = self.conditioning["prompt_mask"]
        return payload
