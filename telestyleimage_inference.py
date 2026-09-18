"""TeleStyle inference for independent north/south charts and RGB ERP composition."""

import copy
import glob
import math
import os

import torch
from tqdm import tqdm

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from telestyle_spherical import HemisphereLatentProjector


RGB_HARD_CUT_SOUTH_YAW_DEGREES = 0.0


def _prepare_edit_inputs(pipe, prompt, content, style, seed, num_inference_steps):
    """Run pipeline units and return one independent edit-inference state."""
    height, width = content.height, content.width
    inputs_posi, inputs_nega = {"prompt": prompt}, {"negative_prompt": ""}
    inputs_shared = {
        "cfg_scale": 1.0, "input_image": None, "denoising_strength": 1.0,
        "inpaint_mask": None, "inpaint_blur_size": None,
        "inpaint_blur_sigma": None, "height": height, "width": width,
        "seed": seed, "rand_device": "cpu",
        "num_inference_steps": num_inference_steps,
        "blockwise_controlnet_inputs": None, "tiled": False,
        "tile_size": 128, "tile_stride": 64,
        "eligen_entity_prompts": None, "eligen_entity_masks": None,
        "eligen_enable_on_negative": False,
        "edit_image": [content, style], "edit_image_auto_resize": False,
        "edit_rope_interpolation": False, "context_image": None,
        "zero_cond_t": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )
    return inputs_shared, inputs_posi, inputs_nega


class ImageStyleInference:
    """Generate ERP images from independent north/south chart trajectories."""

    def __init__(self):
        self._load_models()

    def _load_models(self):
        model_dir = "/root/autodl-tmp/Qwen-Image-Edit-2509"
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=glob.glob(os.path.join(
                    model_dir, "transformer/diffusion_pytorch_model*.safetensors"
                ))),
                ModelConfig(path=glob.glob(os.path.join(
                    model_dir, "text_encoder/model*.safetensors"
                ))),
                ModelConfig(path=os.path.join(
                    model_dir, "vae/diffusion_pytorch_model.safetensors"
                )),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(path=os.path.join(model_dir, "processor")),
        )
        self.pipe.load_lora(
            self.pipe.dit,
            "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
        )
        self.pipe.load_lora(
            self.pipe.dit,
            "weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        )

    @torch.no_grad()
    def inference_with_hemisphere_rgb_hard_cut(
        self, prompt, content_north, content_south, style, seed,
        num_inference_steps, output_height, output_width,
        overlap_degrees=15.0, consistency_degrees=10.0,
        color_match_degrees=6.0, seam_residual_degrees=2.0,
        seam_residual_blur_degrees=0.5,
    ):
        """Denoise two charts, hard-cut at the equator, and repair the seam."""
        if content_north.size != content_south.size:
            raise ValueError("north and south hemisphere charts must have identical sizes.")
        if content_north.width != content_north.height or content_north.width % 16:
            raise ValueError("hemisphere charts must be square and divisible by 16.")
        if output_height <= 0 or output_width != 2 * output_height:
            raise ValueError("RGB hard-cut output must have a positive 2:1 size.")
        if output_height % 16 or output_width % 16:
            raise ValueError("ERP output dimensions must be divisible by 16.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if not 0.0 <= color_match_degrees <= overlap_degrees:
            raise ValueError("color_match_degrees must be between zero and overlap_degrees.")
        if not 0.0 <= seam_residual_degrees <= overlap_degrees:
            raise ValueError("seam_residual_degrees must be between zero and overlap_degrees.")
        if seam_residual_degrees > 0 and (
            not math.isfinite(seam_residual_blur_degrees)
            or seam_residual_blur_degrees <= 0
        ):
            raise ValueError("seam_residual_blur_degrees must be positive when enabled.")

        from training.geometry import (
            correct_equatorial_seam_residual,
            match_equatorial_low_frequency,
            reproject_native_charts,
        )

        pipe = self.pipe
        chart_size = content_north.width
        pipe.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=1.0,
            dynamic_shift_len=(chart_size // 16) ** 2,
        )
        scheduler_north = pipe.scheduler
        scheduler_south = copy.deepcopy(scheduler_north)
        inputs_north, posi_north, nega_north = _prepare_edit_inputs(
            pipe, prompt, content_north, style, seed, num_inference_steps
        )
        inputs_south, posi_south, nega_south = _prepare_edit_inputs(
            pipe, prompt, content_south, style, seed, num_inference_steps
        )
        if inputs_north["latents"].shape != inputs_south["latents"].shape:
            raise ValueError("hemisphere latent charts must have identical shapes.")
        latent_size = inputs_north["latents"].shape[-1]
        if inputs_north["latents"].shape[-2] != latent_size:
            raise ValueError("hemisphere latent charts must be square.")

        projector = HemisphereLatentProjector(
            latent_size, overlap_degrees, inputs_north["latents"].device
        )
        inputs_north["latents"], inputs_south["latents"] = projector.synchronize(
            inputs_north["latents"], inputs_south["latents"]
        )
        inputs_north["noise"] = inputs_north["latents"]
        inputs_south["noise"] = inputs_south["latents"]

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(tqdm(
            scheduler_north.timesteps, desc="TeleStyle hemisphere RGB hard-cut"
        )):
            timestep = timestep.unsqueeze(0).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            noise_pred_north = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0, inputs_north, posi_north, nega_north,
                **models, timestep=timestep, progress_id=progress_id,
            )
            noise_pred_south = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0, inputs_south, posi_south, nega_south,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_north["latents"] = pipe.step(
                scheduler_north, progress_id=progress_id,
                noise_pred=noise_pred_north, **inputs_north,
            )
            inputs_south["latents"] = pipe.step(
                scheduler_south, progress_id=progress_id,
                noise_pred=noise_pred_south, **inputs_south,
            )

        pipe.load_models_to_device(["vae"])
        north_decoded = pipe.vae.decode(
            inputs_north["latents"], device=pipe.device, tiled=False
        )
        south_decoded = pipe.vae.decode(
            inputs_south["latents"], device=pipe.device, tiled=False
        )
        projected = reproject_native_charts(
            north_decoded, south_decoded, overlap_degrees,
            consistency_degrees, output_height, output_width,
            antialias_scale=2,
            south_yaw_degrees=RGB_HARD_CUT_SOUTH_YAW_DEGREES,
        )
        hard_cut_image = pipe.vae_output_to_image(projected.hard_cut)
        matched = match_equatorial_low_frequency(
            projected.north, projected.south, projected.hard_cut,
            projected.latitude_degrees, color_match_degrees,
        )
        matched = correct_equatorial_seam_residual(
            matched, projected.latitude_degrees, seam_residual_degrees,
            seam_residual_blur_degrees,
        )
        image = pipe.vae_output_to_image(matched)
        del north_decoded, south_decoded, projected, matched
        pipe.load_models_to_device([])
        return image, hard_cut_image
