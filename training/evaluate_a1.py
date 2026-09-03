"""Evaluate A1 full/cross-chart-off ablations on cached validation samples."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .cached_dataset import CachedA1Dataset
from .checkpoint import (
    dependency_hashes, load_adapter_checkpoint, sha256_file,
    validate_adapter_metadata,
)
from .conditioning import (
    TELESTYLE_PROMPT, cache_style_latent, compute_prompt_conditioning,
    discover_style_paths, split_style_paths,
)
from .config import load_config, resolve_path
from .geometry import build_sphere_geometry, reproject_native_charts
from .joint_qwen import JointQwenSphereModel
from .losses import geometric_losses
from .models import configured_lora_paths, load_base_pipeline
from .runtime import (
    require_model_files,
    require_python_311,
    select_runtime_profile,
)
from .sphere_adapter import SphereAdapter
from .train_sphere_adapter import _synchronize_overlap_noise


COMPARISON_MODES = (
    "zero_gate_baseline",
    "adapter_full",
    "cross_chart_off",
)
COMPARISON_LABELS = {
    "zero_gate_baseline": "TeleStyle Teacher",
    "adapter_full": "Adapter full",
    "cross_chart_off": "Cross-chart off",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A1 hard-seam geometry")
    parser.add_argument("--config", default="configs/a1.yaml")
    parser.add_argument("--checkpoint", help="Omit for the zero-gate baseline")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--save-comparison-images",
        action="store_true",
        help="Save zero-gate/full/cross-chart-off ERP comparison images",
    )
    parser.add_argument(
        "--comparison-samples",
        type=int,
        default=4,
        help="Number of validation samples rendered when comparison output is enabled",
    )
    parser.add_argument(
        "--comparison-output",
        help="Comparison directory; defaults beside the evaluation JSON",
    )
    return parser.parse_args()


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: sum(record[key] for record in records) / len(records)
        for key in records[0]
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if not args.save_comparison_images:
        return
    if not args.checkpoint:
        raise ValueError("--save-comparison-images requires --checkpoint.")
    if args.comparison_samples <= 0:
        raise ValueError("--comparison-samples must be positive.")
    if args.comparison_samples > args.max_samples:
        raise ValueError("--comparison-samples cannot exceed --max-samples.")


@contextmanager
def _adapter_mode(model: JointQwenSphereModel, mode: str):
    """Temporarily configure one ablation mode and restore all live gates."""
    if mode not in COMPARISON_MODES:
        raise ValueError(f"unsupported comparison mode: {mode}")
    injection_gates = model.injection_gates.detach().clone()
    position_gate = model.adapter.position_gate.detach().clone()
    try:
        if mode == "zero_gate_baseline":
            model.injection_gates.zero_()
            model.adapter.position_gate.zero_()
        yield mode != "adapter_full"
    finally:
        model.injection_gates.copy_(injection_gates)
        model.adapter.position_gate.copy_(position_gate)


def _comparison_sheet(images: dict[str, Image.Image]) -> Image.Image:
    """Build a labelled left-to-right triptych from the three ERP modes."""
    ordered = [images[name].convert("RGB") for name in COMPARISON_MODES]
    if len({image.size for image in ordered}) != 1:
        raise ValueError("comparison ERP images must have identical sizes.")
    width, height = ordered[0].size
    header_height = 32
    sheet = Image.new("RGB", (width * len(ordered), height + header_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (mode, image) in enumerate(zip(COMPARISON_MODES, ordered)):
        offset = index * width
        draw.text((offset + 8, 8), COMPARISON_LABELS[mode], fill="black")
        sheet.paste(image, (offset, header_height))
    return sheet


def _comparison_root(
    args: argparse.Namespace, output_root: Path, run_name: str,
) -> Path:
    if not args.comparison_output:
        return output_root / f"{run_name}_comparison"
    path = Path(args.comparison_output)
    if path.is_absolute():
        return path
    return (Path(args.config).resolve().parent.parent / path).resolve()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    require_python_311()
    profile = select_runtime_profile(config)
    model_dir = Path(config.paths.model_dir)
    require_model_files(model_dir)
    cache_root = resolve_path(args.config, config.paths.cache_root)
    device, dtype = torch.device("cuda"), torch.bfloat16
    vram_limit = profile.vram_gb * 0.85 if profile.cpu_offload else None
    lora_paths = configured_lora_paths(config, args.config)
    pipe = load_base_pipeline(
        model_dir, device="cuda", torch_dtype=dtype, vram_limit=vram_limit,
        cpu_offload=profile.cpu_offload, lora_paths=lora_paths,
    )
    pipe.load_models_to_device(["dit", "vae"])
    telestyle_pilot = bool(lora_paths)
    prompt = str(config.training.get("prompt", TELESTYLE_PROMPT))
    style_latent = None
    if telestyle_pilot:
        conditioning = compute_prompt_conditioning(pipe, prompt)
        dataset = CachedA1Dataset(
            cache_root / "index.jsonl", "validation", conditioning=conditioning
        )
        _, held_out_style = split_style_paths(
            discover_style_paths(resolve_path(args.config, config.paths.style_root))
        )
        style_size = int(config.training.get("style_size", 1024))
        style_latent = cache_style_latent(
            pipe, held_out_style,
            resolve_path(args.config, config.paths.style_cache_root), style_size,
        ).to(device=device, dtype=dtype)
    else:
        dataset = CachedA1Dataset(cache_root / "index.jsonl", "validation")
    adapter = SphereAdapter(
        hidden_dim=int(config.model.hidden_dim),
        adapter_dim=int(config.model.adapter_dim),
        num_heads=int(config.model.num_heads),
        global_tokens=int(config.model.global_tokens),
        equator_tokens=int(config.model.equator_tokens),
    ).to(device=device, dtype=dtype)
    model = JointQwenSphereModel(
        pipe.dit, adapter, int(config.model.injection_stride), False
    )
    model.eval()
    checkpoint_metadata = {}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        _, checkpoint_metadata = load_adapter_checkpoint(
            checkpoint_path, model.adapter, model.injection_gates
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "ema" in checkpoint and "shadow" in checkpoint["ema"]:
            model.adapter.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
    output_root = resolve_path(args.config, config.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = Path(args.checkpoint).stem if args.checkpoint else "zero_gate_baseline"
    comparison_root = None
    comparison_results = {name: [] for name in COMPARISON_MODES}
    comparison_manifest = []
    if args.save_comparison_images:
        comparison_root = _comparison_root(args, output_root, run_name)
        for directory in (*COMPARISON_MODES, "contact_sheets"):
            (comparison_root / directory).mkdir(parents=True, exist_ok=True)
    if telestyle_pilot:
        pipe.scheduler.set_timesteps(
            4, denoising_strength=1.0,
            dynamic_shift_len=(profile.chart_size // 16) ** 2,
        )
        schedule_timesteps = list(pipe.scheduler.timesteps)
        if len(schedule_timesteps) != 4:
            raise RuntimeError("Lightning evaluation requires four timesteps.")
        validate_adapter_metadata(
            checkpoint_metadata,
            base_transformer_sha256=sha256_file(
                model_dir / "transformer"
                / "diffusion_pytorch_model.safetensors.index.json"
            ),
            lora_sha256=dependency_hashes(lora_paths), prompt=prompt,
            lightning_timesteps=[
                float(value.detach().cpu()) for value in schedule_timesteps
            ],
            south_yaw_degrees=float(
                config.training.get("south_yaw_degrees", 0.0)
            ),
        )
        timestep_id = 2
    else:
        pipe.scheduler.set_timesteps(
            1000, training=True,
            dynamic_shift_len=(profile.chart_size // 16) ** 2,
        )
        schedule_timesteps = list(pipe.scheduler.timesteps)
        timestep_id = 250
    timestep = schedule_timesteps[timestep_id]
    sigma = pipe.scheduler.sigmas[timestep_id].to(device=device, dtype=dtype)
    timestep_device = timestep.reshape(1).to(device=device, dtype=dtype)
    results = {name: [] for name in COMPARISON_MODES}

    for sample_id in range(min(len(dataset), args.max_samples)):
        sample = dataset[sample_id]
        north = {
            key: value.to(device=device, dtype=dtype)
            if key != "prompt_mask" else value.to(device)
            for key, value in sample["north"].items()
        }
        south = {
            key: value.to(device=device, dtype=dtype)
            if key != "prompt_mask" else value.to(device)
            for key, value in sample["south"].items()
        }
        generator_n = torch.Generator(device=device).manual_seed(10000 + sample_id * 2)
        generator_s = torch.Generator(device=device).manual_seed(10000 + sample_id * 2)
        noise_n = torch.randn(
            north["latent"].shape, generator=generator_n,
            device=device, dtype=dtype,
        )
        noise_s = torch.randn(
            south["latent"].shape, generator=generator_s,
            device=device, dtype=dtype,
        )
        overlap = float(sample["overlap_degrees"])
        geometry = build_sphere_geometry(
            north["latent"].shape[-1] // 2,
            overlap,
            float(config.data.consistency_degrees),
            int(config.model.correspondence_neighbors),
            device,
        )
        noise_n, noise_s = _synchronize_overlap_noise(noise_n, noise_s, geometry)
        zt_n = (1 - sigma) * north["latent"] + sigma * noise_n
        zt_s = (1 - sigma) * south["latent"] + sigma * noise_s
        render_comparison = (
            comparison_root is not None and sample_id < args.comparison_samples
        )
        modes = COMPARISON_MODES
        rendered_images = {}
        rendered_paths = {}
        teacher_prediction = None
        for mode in modes:
            with _adapter_mode(model, mode) as disabled:
                pred_n, pred_s = model(
                    zt_n, zt_s,
                    ([north["latent"], style_latent] if telestyle_pilot else north["latent"]),
                    ([south["latent"], style_latent] if telestyle_pilot else south["latent"]),
                    north["prompt_emb"], south["prompt_emb"],
                    north["prompt_mask"], south["prompt_mask"],
                    timestep_device, geometry,
                    disable_cross_chart=disabled,
                    adapter_enabled=mode != "zero_gate_baseline",
                )
            if mode == "zero_gate_baseline":
                teacher_prediction = (pred_n, pred_s)
            x0_n, x0_s = zt_n - sigma * pred_n, zt_s - sigma * pred_s
            rgb_n = pipe.vae.decode(x0_n, device=device, tiled=False)
            rgb_s = pipe.vae.decode(x0_s, device=device, tiled=False)
            projected = reproject_native_charts(
                rgb_n, rgb_s, overlap,
                float(config.data.consistency_degrees),
                profile.chart_size // 2, profile.chart_size,
                south_yaw_degrees=float(config.training.get("south_yaw_degrees", 0.0)),
            )
            losses = geometric_losses(
                projected.north, projected.south, projected.hard_cut,
                projected.consistency_mask, projected.latitude_degrees,
            )
            if teacher_prediction is None:
                teacher_deviation = pred_n.new_zeros(())
            else:
                teacher_deviation = 0.5 * (
                    F.mse_loss(pred_n, teacher_prediction[0])
                    + F.mse_loss(pred_s, teacher_prediction[1])
                )
            losses["teacher_deviation"] = teacher_deviation
            record = {key: float(value) for key, value in losses.items()}
            results[mode].append(record)
            if render_comparison:
                comparison_results[mode].append(record)
                image = pipe.vae_output_to_image(projected.hard_cut)
                image_path = (
                    comparison_root / mode / f"sample_{sample_id:04d}.png"
                )
                image.save(image_path)
                rendered_images[mode] = image
                rendered_paths[mode] = str(
                    image_path.relative_to(comparison_root)
                )
        if render_comparison:
            sheet_path = (
                comparison_root / "contact_sheets"
                / f"sample_{sample_id:04d}.png"
            )
            _comparison_sheet(rendered_images).save(sheet_path)
            comparison_manifest.append(
                {
                    "sample_id": sample_id,
                    "scene_id": sample.get("scene_id"),
                    "source_path": sample.get("source_path"),
                    "variant": sample.get("variant"),
                    "rotation_degrees": sample.get("rotation_degrees"),
                    "overlap_degrees": overlap,
                    "images": rendered_paths,
                    "contact_sheet": str(sheet_path.relative_to(comparison_root)),
                }
            )

    summary = {name: _mean(items) for name, items in results.items()}
    summary["gate_strength"] = {
        "position": float(model.adapter.position_gate.detach().abs()),
        "injection_mean": float(model.injection_gates.detach().abs().mean()),
        "injection_max": float(model.injection_gates.detach().abs().max()),
    }
    output_path = output_root / f"{run_name}_evaluation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Evaluation written to {output_path}")
    if comparison_root is not None:
        comparison_summary = {
            name: _mean(items) for name, items in comparison_results.items()
        }
        (comparison_root / "metrics.json").write_text(
            json.dumps(comparison_summary, indent=2), encoding="utf-8"
        )
        (comparison_root / "manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "timestep_id": timestep_id,
                    "sample_count": len(comparison_manifest),
                    "modes": list(COMPARISON_MODES),
                    "samples": comparison_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Comparison images written to {comparison_root}")


if __name__ == "__main__":
    main()
