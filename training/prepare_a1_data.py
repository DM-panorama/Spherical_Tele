"""Scan ERP data, generate manifests/stress cases, and optionally build caches."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from .cache import build_cache
from .config import load_config, resolve_path
from .data import (
    ErpRecord,
    RejectedErp,
    discover_images,
    inspect_erp,
    write_manifests,
)
from .models import load_base_pipeline
from .runtime import (
    estimate_cache_bytes,
    require_free_space,
    require_model_files,
    require_python_311,
    select_runtime_profile,
)
from .stress import generate_stress_erps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pure-geometry A1 ERP data")
    parser.add_argument("--config", default="configs/a1.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--variants", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root else resolve_path(args.config, config.paths.dataset_root)
    )
    manifest_root = resolve_path(args.config, config.paths.manifest_root)
    cache_root = (
        Path(args.cache_root).resolve()
        if args.cache_root else resolve_path(args.config, config.paths.cache_root)
    )
    stress_root = resolve_path(args.config, config.paths.stress_root)
    for path in (dataset_root, manifest_root, cache_root, stress_root):
        path.mkdir(parents=True, exist_ok=True)

    paths = discover_images(dataset_root, config.data.extensions)
    accepted: list[ErpRecord] = []
    rejected: list[RejectedErp] = []
    for path in paths:
        result = inspect_erp(
            path,
            dataset_root,
            int(config.data.min_erp_height),
            tuple(config.data.split_percentages),
        )
        (accepted if isinstance(result, ErpRecord) else rejected).append(result)
    write_manifests(accepted, rejected, manifest_root)
    stress = generate_stress_erps(
        stress_root, width=max(1024, 2 * int(config.data.min_erp_height))
    )
    stress_records = []
    for path in stress:
        result = inspect_erp(
            path,
            stress_root,
            int(config.data.min_erp_height),
            tuple(config.data.split_percentages),
        )
        if isinstance(result, ErpRecord):
            stress_records.append(
                replace(result, scene_id=f"stress:{path.stem}", split="train")
            )
    counts = Counter(record.split for record in accepted)
    warnings = sum(
        record.seam_error > float(config.data.seam_error_warn_threshold)
        for record in accepted
    )
    print(
        f"Scanned {len(paths)} files: {len(accepted)} accepted, "
        f"{len(rejected)} rejected, splits={dict(counts)}, seam_warnings={warnings}."
    )
    print(f"Generated {len(stress)} deterministic stress ERPs at {stress_root}.")

    if not args.build_cache:
        print("Manifest scan complete. Add --build-cache after CUDA is available.")
        return
    if args.variants <= 0:
        raise ValueError("--variants must be positive.")
    if not accepted:
        raise RuntimeError(f"no valid ERP images found in {dataset_root}")
    profile = select_runtime_profile(config)
    require_python_311()
    model_dir = Path(config.paths.model_dir)
    require_model_files(model_dir)
    cache_records = accepted + stress_records
    estimated = estimate_cache_bytes(
        len(cache_records), profile.chart_size, args.variants
    )
    require_free_space(cache_root, estimated)
    vram_limit = profile.vram_gb * 0.85 if profile.cpu_offload else None
    pipe = load_base_pipeline(
        model_dir, device="cuda", torch_dtype=torch.bfloat16,
        vram_limit=vram_limit, cpu_offload=profile.cpu_offload,
    )
    index_path = build_cache(
        cache_records,
        cache_root,
        pipe,
        profile.chart_size,
        str(config.training.prompt),
        tuple(float(value) for value in config.data.overlap_degrees_range),
        variants=args.variants,
    )
    print(f"A1 cache index written to {index_path}.")


if __name__ == "__main__":
    main()
