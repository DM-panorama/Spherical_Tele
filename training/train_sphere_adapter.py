"""Full A1 SphereAdapter training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm.auto import tqdm

from .cached_dataset import CachedA1Dataset
from .checkpoint import (
    AdapterEma,
    dependency_hashes,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    sha256_file,
)
from .conditioning import (
    TELESTYLE_PROMPT, cache_style_latent, compute_prompt_conditioning,
    discover_style_paths, split_style_paths, style_manifest,
)
from .config import load_config, resolve_path
from .geometry import (
    build_sphere_geometry,
    reproject_native_charts,
    swap_sphere_geometry,
)
from .joint_qwen import JointQwenSphereModel
from .losses import geometric_losses
from .models import configured_lora_paths, load_base_pipeline
from .runtime import (
    require_model_files,
    require_python_311,
    select_runtime_profile,
)
from .sphere_adapter import SphereAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A1 SphereAdapter")
    parser.add_argument("--config", default="configs/a1.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _stage(config, step: int) -> tuple[str, int, dict[str, float]]:
    if step < int(config.training.warmup_end):
        name, index = "warmup", 0
    elif step < int(config.training.geometry_end):
        name, index = "geometry", 1
    else:
        name, index = "highfreq", 2
    weights = {
        key: float(value) for key, value in config.loss[name].items()
    }
    return name, index, weights


def _optimizer(model: JointQwenSphereModel, config):
    position, gates, cross = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "gate" in name:
            gates.append(parameter)
        elif "position" in name or "seed" in name:
            position.append(parameter)
        else:
            cross.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": position, "lr": float(config.training.lr_position_tokens)},
            {"params": cross, "lr": float(config.training.lr_cross_modules)},
            {"params": gates, "lr": float(config.training.lr_gates)},
        ],
        weight_decay=float(config.training.weight_decay),
    )


def _corrupt_condition(
    latent: torch.Tensor,
    overlap: torch.Tensor,
    asymmetric_probability: float = 0.20,
    dropout_probability: float = 0.10,
) -> torch.Tensor:
    result = latent
    side = int(math.sqrt(overlap.numel()))
    mask = overlap.reshape(1, 1, side, side).float()
    mask = F.interpolate(
        mask, size=latent.shape[-2:], mode="nearest"
    ).to(device=latent.device, dtype=latent.dtype)
    draw = random.random()
    if draw < dropout_probability:
        result = result * (1.0 - mask)
    elif draw < dropout_probability + asymmetric_probability:
        result = result + torch.randn_like(result) * mask * 0.10
    return result


def _to_device(branch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    return {
        "latent": branch["latent"].to(device=device, dtype=dtype),
        "prompt_emb": branch["prompt_emb"].to(device=device, dtype=dtype),
        "prompt_mask": branch["prompt_mask"].to(device=device),
    }

def _decode_vae_checkpointed(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Decode a latent while recomputing VAE activations during backward."""
    return torch_checkpoint(
        lambda value: vae.decode(value, device=device, tiled=False),
        latent,
        use_reentrant=False,
    )


def _resize_latent_for_decode(
    latent: torch.Tensor,
    chart_size: int,
    decode_chart_size: int,
) -> torch.Tensor:
    """Resize only the auxiliary RGB-loss latent to reduce VAE memory."""
    if decode_chart_size == chart_size:
        return latent
    scale = decode_chart_size / chart_size
    target_size = tuple(round(size * scale) for size in latent.shape[-2:])
    return F.interpolate(
        latent, size=target_size, mode="bilinear", align_corners=False
    )


def _next_sample(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _prune_checkpoints(
    output_root: Path, keep_last: int, protected_steps: set[int],
) -> None:
    checkpoints = sorted(output_root.glob("a1_step_*.pt"))
    removable = [
        path for path in checkpoints
        if int(path.stem.rsplit("_", 1)[-1]) not in protected_steps
    ]
    for path in removable[:-keep_last]:
        path.unlink()



def _set_lightning_schedule(scheduler, chart_size: int) -> tuple[list, list]:
    """Configure the exact four denoising points used by Lightning-4step."""
    scheduler.set_timesteps(
        4, denoising_strength=1.0, dynamic_shift_len=(chart_size // 16) ** 2
    )
    timesteps = list(scheduler.timesteps)
    sigmas = list(scheduler.sigmas[:len(timesteps)])
    if len(timesteps) != 4 or len(sigmas) != 4:
        raise RuntimeError("Lightning schedule must expose exactly four timesteps.")
    return timesteps, sigmas


def _schedule_metadata(timesteps: list) -> list[float]:
    return [float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for value in timesteps]


def _synchronize_overlap_noise(
    north: torch.Tensor, south: torch.Tensor, geometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synchronize aligned overlap noise using token correspondence maps."""
    north_tokens = F.avg_pool2d(north.float(), 2).flatten(2).transpose(1, 2)
    south_tokens = F.avg_pool2d(south.float(), 2).flatten(2).transpose(1, 2)
    north_mask = geometry.north_overlap
    if north_mask.any():
        opposite = geometry.north_to_south_indices[:, 0]
        aligned_south = south_tokens[:, opposite]
        merged = 0.5 * (north_tokens + aligned_south)
        north_delta = torch.zeros_like(north_tokens)
        south_delta = torch.zeros_like(south_tokens)
        north_delta[:, north_mask] = merged[:, north_mask] - north_tokens[:, north_mask]
        target = opposite[north_mask]
        south_delta[:, target] = merged[:, north_mask] - south_tokens[:, target]
        side = north.shape[-1] // 2
        north = north + F.interpolate(
            north_delta.transpose(1, 2).reshape(north.shape[0], north.shape[1], side, side),
            size=north.shape[-2:], mode="nearest",
        ).to(north.dtype)
        south = south + F.interpolate(
            south_delta.transpose(1, 2).reshape(south.shape[0], south.shape[1], side, side),
            size=south.shape[-2:], mode="nearest",
        ).to(south.dtype)
    return north, south


def _gate_l2(model: JointQwenSphereModel) -> torch.Tensor:
    return model.injection_gates.square().mean() + model.adapter.position_gate.square()


def _only_adapter_trainable(model: JointQwenSphereModel) -> bool:
    """CPU-testable invariant for the frozen shared Teacher/Student backbone."""
    return all(
        parameter.requires_grad == (
            name.startswith("adapter.") or name == "injection_gates"
        )
        for name, parameter in model.named_parameters()
    )
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    profile = select_runtime_profile(config)
    model_dir = Path(config.paths.model_dir)
    require_model_files(model_dir)
    cache_root = resolve_path(args.config, config.paths.cache_root)
    require_python_311()
    output_root = resolve_path(args.config, config.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(config.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    decode_chart_size = int(
        config.training.get("decode_chart_size", profile.chart_size)
    )
    if (
        decode_chart_size <= 0
        or decode_chart_size > profile.chart_size
        or decode_chart_size % 16
    ):
        raise ValueError(
            "training.decode_chart_size must be a positive multiple of 16 "
            f"not exceeding the runtime chart size ({profile.chart_size})."
        )
    vram_limit = profile.vram_gb * 0.85 if profile.cpu_offload else None
    lora_paths = configured_lora_paths(config, args.config)
    pipe = load_base_pipeline(
        model_dir, device="cuda", torch_dtype=dtype, vram_limit=vram_limit,
        cpu_offload=profile.cpu_offload, lora_paths=lora_paths,
    )
    pipe.load_models_to_device(["dit", "vae"])
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)

    telestyle_pilot = bool(lora_paths)
    prompt = str(config.training.get("prompt", TELESTYLE_PROMPT))
    training_styles, held_out_style = [], None
    style_latents = {}
    style_size = int(config.training.get("style_size", 1024))
    if telestyle_pilot:
        conditioning = compute_prompt_conditioning(pipe, prompt)
        dataset = CachedA1Dataset(
            cache_root / "index.jsonl", "train", conditioning=conditioning
        )
        style_root = resolve_path(args.config, config.paths.style_root)
        training_styles, held_out_style = split_style_paths(
            discover_style_paths(style_root)
        )
        style_cache_root = resolve_path(args.config, config.paths.style_cache_root)
        style_latents = {
            path: cache_style_latent(pipe, path, style_cache_root, style_size)
            for path in training_styles
        }
    else:
        dataset = CachedA1Dataset(cache_root / "index.jsonl", "train")

    adapter = SphereAdapter(
        hidden_dim=int(config.model.hidden_dim),
        adapter_dim=int(config.model.adapter_dim),
        num_heads=int(config.model.num_heads),
        global_tokens=int(config.model.global_tokens),
        equator_tokens=int(config.model.equator_tokens),
    ).to(device=device, dtype=dtype)
    model = JointQwenSphereModel(
        pipe.dit,
        adapter,
        injection_stride=int(config.model.injection_stride),
        use_gradient_checkpointing=True,
    )
    model.train()
    if not _only_adapter_trainable(model):
        raise RuntimeError("only SphereAdapter and injection gates may be trainable.")
    optimizer = _optimizer(model, config)
    ema = AdapterEma(model.adapter, float(config.training.ema_decay))
    start_step = 0
    resume_metadata = {}
    if args.resume:
        start_step, resume_metadata = load_adapter_checkpoint(
            Path(args.resume), model.adapter, model.injection_gates, ema, optimizer
        )
        print(f"Resumed step {start_step} from {args.resume}: {resume_metadata}")

    total_steps = args.max_steps or int(config.training.total_steps)
    lr_warmup_steps = int(config.training.get("lr_warmup_steps", 500))
    if lr_warmup_steps <= 0:
        raise ValueError("training.lr_warmup_steps must be positive.")
    base_learning_rates = [
        float(config.training.lr_position_tokens),
        float(config.training.lr_cross_modules),
        float(config.training.lr_gates),
    ]
    if telestyle_pilot:
        schedule_timesteps, schedule_sigmas = _set_lightning_schedule(
            pipe.scheduler, profile.chart_size
        )
    else:
        pipe.scheduler.set_timesteps(
            1000, training=True,
            dynamic_shift_len=(profile.chart_size // 16) ** 2,
        )
        schedule_timesteps = list(pipe.scheduler.timesteps)
        schedule_sigmas = list(pipe.scheduler.sigmas[:len(schedule_timesteps)])
    normal_indices = [
        index for index, stress in enumerate(dataset.stress_flags) if not stress
    ]
    stress_indices = [
        index for index, stress in enumerate(dataset.stress_flags) if stress
    ]
    if not normal_indices:
        raise RuntimeError("A1 training requires at least one natural ERP cache.")
    loader_options = {"batch_size": 1, "shuffle": True, "num_workers": 1,
                      "collate_fn": lambda items: items[0]}
    normal_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, normal_indices), **loader_options
    )
    stress_loader = (
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, stress_indices), **loader_options
        ) if stress_indices else None
    )
    normal_iterator = iter(normal_loader)
    stress_iterator = iter(stress_loader) if stress_loader is not None else None
    optimizer.zero_grad(set_to_none=True)
    accumulation = profile.gradient_accumulation
    geometry_cache = {}
    log_path = output_root / "train.jsonl"
    samples_seen = int(
        resume_metadata.get("samples_seen", start_step * accumulation)
    )
    progress_bar = tqdm(
        total=total_steps * accumulation,
        initial=samples_seen,
        desc="A1 training",
        unit="sample",
        dynamic_ncols=True,
    )

    for step in range(start_step, total_steps):
        stage_name, stage_index, weights = _stage(config, step)
        total_loss_value = 0.0
        progress = step / max(total_steps - 1, 1)
        learning_rate_factor = (
            min(1.0, (step + 1) / lr_warmup_steps)
            * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        for group, base_rate in zip(optimizer.param_groups, base_learning_rates):
            group["lr"] = base_rate * learning_rate_factor
        for _ in range(accumulation):
            use_stress = (
                stage_name == "highfreq"
                and stress_loader is not None
                and random.random() < float(config.data.stress_fraction_highfreq)
            )
            if use_stress:
                sample, stress_iterator = _next_sample(
                    stress_loader, stress_iterator
                )
            else:
                sample, normal_iterator = _next_sample(
                    normal_loader, normal_iterator
                )
            north = _to_device(sample["north"], device, dtype)
            south = _to_device(sample["south"], device, dtype)
            overlap = float(sample["overlap_degrees"])
            token_size = north["latent"].shape[-1] // 2
            key = (token_size, round(overlap, 2))
            if key not in geometry_cache:
                if len(geometry_cache) >= 16:
                    geometry_cache.pop(next(iter(geometry_cache)))
                geometry_cache[key] = build_sphere_geometry(
                    token_size,
                    overlap,
                    float(config.data.consistency_degrees),
                    int(config.model.correspondence_neighbors),
                    device,
                )
            geometry = geometry_cache[key]
            timestep_id = random.randrange(len(schedule_timesteps))
            timestep = schedule_timesteps[timestep_id]
            sigma = schedule_sigmas[timestep_id].to(device=device, dtype=dtype)
            timestep_device = timestep.reshape(1).to(device=device, dtype=dtype)
            noise_seed = random.randrange(2**31)
            generator_n = torch.Generator(device=device).manual_seed(noise_seed)
            generator_s = torch.Generator(device=device).manual_seed(
                noise_seed if telestyle_pilot else noise_seed + 1
            )
            noise_n = torch.randn(
                north["latent"].shape, generator=generator_n,
                device=device, dtype=dtype,
            )
            noise_s = torch.randn(
                south["latent"].shape, generator=generator_s,
                device=device, dtype=dtype,
            )
            if telestyle_pilot:
                noise_n, noise_s = _synchronize_overlap_noise(
                    noise_n, noise_s, geometry
                )
            zt_n = (1.0 - sigma) * north["latent"] + sigma * noise_n
            zt_s = (1.0 - sigma) * south["latent"] + sigma * noise_s
            if telestyle_pilot:
                style_path = random.choice(training_styles)
                style_latent = style_latents[style_path].to(
                    device=device, dtype=dtype
                )
                edit_n = [north["latent"], style_latent]
                edit_s = [south["latent"], style_latent]
                with torch.no_grad():
                    teacher_n, teacher_s = model(
                        zt_n, zt_s, edit_n, edit_s,
                        north["prompt_emb"], south["prompt_emb"],
                        north["prompt_mask"], south["prompt_mask"],
                        timestep_device, geometry, adapter_enabled=False,
                    )
            else:
                edit_n = _corrupt_condition(
                    north["latent"], geometry.north_overlap
                )
                edit_s = _corrupt_condition(
                    south["latent"], geometry.south_overlap
                )
            prediction_n, prediction_s = model(
                zt_n, zt_s, edit_n, edit_s,
                north["prompt_emb"], south["prompt_emb"],
                north["prompt_mask"], south["prompt_mask"],
                timestep_device, geometry,
            )
            if telestyle_pilot:
                primary = 0.5 * (
                    F.mse_loss(prediction_n, teacher_n)
                    + F.mse_loss(prediction_s, teacher_s)
                )
            else:
                primary = 0.5 * (
                    F.mse_loss(prediction_n, noise_n - north["latent"])
                    + F.mse_loss(prediction_s, noise_s - south["latent"])
                )
            x0_n = zt_n - sigma * prediction_n
            x0_s = zt_s - sigma * prediction_s
            geometry_losses = {
                name: primary.new_zeros(())
                for name in ("rgb", "gradient", "laplacian", "seam", "energy")
            }
            if telestyle_pilot:
                projected = reproject_native_charts(
                    x0_n.float(), x0_s.float(), overlap,
                    float(config.data.consistency_degrees),
                    x0_n.shape[-2] // 2, x0_n.shape[-1],
                    antialias_scale=1,
                    south_yaw_degrees=float(
                        config.training.get("south_yaw_degrees", 0.0)
                    ),
                )
                geometry_losses = geometric_losses(
                    projected.north, projected.south, projected.hard_cut,
                    projected.consistency_mask, projected.latitude_degrees,
                )
            losses = {
                "teacher": primary,
                "diffusion": primary,
                **geometry_losses,
                "gate": _gate_l2(model),
                "symmetry": primary.new_zeros(()),
            }
            decode_probability = float(
                config.training.decode_loss_probability[stage_index]
            )
            if random.random() < decode_probability:
                decode_x0_n = _resize_latent_for_decode(
                    x0_n, profile.chart_size, decode_chart_size
                )
                decode_x0_s = _resize_latent_for_decode(
                    x0_s, profile.chart_size, decode_chart_size
                )
                rgb_n = _decode_vae_checkpointed(pipe.vae, decode_x0_n, device)
                rgb_s = _decode_vae_checkpointed(pipe.vae, decode_x0_s, device)
                decoded = reproject_native_charts(
                    rgb_n, rgb_s, overlap,
                    float(config.data.consistency_degrees),
                    decode_chart_size // 2, decode_chart_size,
                    south_yaw_degrees=float(config.training.get("south_yaw_degrees", 0.0)),
                )
                decoded_losses = geometric_losses(
                    decoded.north, decoded.south, decoded.hard_cut,
                    decoded.consistency_mask, decoded.latitude_degrees,
                )
                for name, value in decoded_losses.items():
                    losses[name] = (
                        losses[name] + value if telestyle_pilot else value
                    )
            loss = sum(
                weights[name] * losses[name]
                for name in weights
            )
            loss = loss + float(config.training.get("gate_l2", 0.0)) * losses["gate"]
            (loss / accumulation).backward()
            total_loss_value += float(loss.detach()) / accumulation
            samples_seen += 1
            progress_bar.update(1)

        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config.training.gradient_clip),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model.adapter)
        completed = step + 1
        record = {
            "step": completed,
            "samples_seen": samples_seen,
            "stage": stage_name,
            "loss": total_loss_value,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "gate_max": float(model.injection_gates.detach().abs().max()),
        }
        progress_bar.set_postfix(
            step=completed,
            stage=stage_name,
            loss=f"{total_loss_value:.4f}",
            gate=f"{record['gate_max']:.4f}",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if completed % 10 == 0:
            progress_bar.write(str(record))
        if completed % int(config.training.save_steps) == 0 or completed == total_steps:
            transformer_index = (
                model_dir / "transformer"
                / "diffusion_pytorch_model.safetensors.index.json"
            )
            metadata = {
                "base_transformer_index_sha256": sha256_file(transformer_index),
                "chart_size": profile.chart_size,
                "config": str(Path(args.config).resolve()),
                "stage": stage_name,
                "samples_seen": samples_seen,
                "gradient_accumulation": accumulation,
            }
            if telestyle_pilot:
                metadata.update({
                    "schema_version": 2,
                    "base_model": str(model_dir.resolve()),
                    "lora_sha256": dependency_hashes(lora_paths),
                    "prompt": prompt,
                    "styles": style_manifest(
                        [*training_styles, held_out_style], style_size
                    ),
                    "training_styles": [path.name for path in training_styles],
                    "validation_style": held_out_style.name,
                    "lightning_timesteps": _schedule_metadata(schedule_timesteps),
                    "south_yaw_degrees": float(
                        config.training.get("south_yaw_degrees", 0.0)
                    ),
                })
            save_adapter_checkpoint(
                output_root / f"a1_step_{completed:06d}.pt",
                model.adapter,
                model.injection_gates,
                ema,
                optimizer,
                completed,
                metadata,
            )
            _prune_checkpoints(
                output_root,
                int(config.training.keep_last),
                {
                    int(config.training.warmup_end),
                    int(config.training.geometry_end),
                    total_steps,
                },
            )
    progress_bar.close()


if __name__ == "__main__":
    main()
