"""ERP discovery, validation, deterministic splitting, and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ErpRecord:
    path: str
    scene_id: str
    width: int
    height: int
    split: str
    fingerprint: str
    seam_error: float


@dataclass(frozen=True)
class RejectedErp:
    path: str
    reason: str


def discover_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    normalized = {extension.lower() for extension in extensions}
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def infer_scene_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_scene(scene_id: str, percentages=(90, 5, 5)) -> str:
    if len(percentages) != 3 or sum(percentages) != 100:
        raise ValueError("split percentages must contain three values summing to 100.")
    bucket = int.from_bytes(
        hashlib.sha256(scene_id.encode("utf-8")).digest()[:8], "big"
    ) % 100
    if bucket < percentages[0]:
        return "train"
    if bucket < percentages[0] + percentages[1]:
        return "validation"
    return "test"


def _seam_error(image: Image.Image, sample_rows: int = 256) -> float:
    resized = image.resize((512, min(sample_rows, image.height)), Image.Resampling.BOX)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return float(np.abs(array[:, 0] - array[:, -1]).mean())


def inspect_erp(
    path: Path,
    root: Path,
    min_height: int,
    percentages=(90, 5, 5),
) -> ErpRecord | RejectedErp:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width != 2 * height:
                return RejectedErp(str(path), f"expected 2:1 ERP, got {width}x{height}")
            if height < min_height:
                return RejectedErp(str(path), f"height {height} is below {min_height}")
            seam_error = _seam_error(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        return RejectedErp(str(path), f"cannot read image: {error}")
    scene_id = infer_scene_id(path, root)
    return ErpRecord(
        path=str(path.resolve()),
        scene_id=scene_id,
        width=width,
        height=height,
        split=split_for_scene(scene_id, percentages),
        fingerprint=file_fingerprint(path),
        seam_error=seam_error,
    )


def write_manifests(
    records: Iterable[ErpRecord], rejected: Iterable[RejectedErp], output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grouped = {"train": [], "validation": [], "test": []}
    for record in records:
        grouped[record.split].append(record)
    for split, items in grouped.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    with (output / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ErpRecord]:
    with path.open(encoding="utf-8") as handle:
        return [ErpRecord(**json.loads(line)) for line in handle if line.strip()]
