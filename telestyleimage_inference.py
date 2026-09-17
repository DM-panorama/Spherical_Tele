import copy
import torch
import os
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from telestyle_spherical import (
    HemisphereLatentProjector,
    SphericalLatentProjector,
    early_polar_guidance_strength,
    latitude_adaptive_circular_lowpass,
    limit_polar_latent_detail,
    make_polar_latitude_weight,
    make_circular_latent_canvas,
    make_rotated_latent_canvas,
    make_spherical_latent_canvas,
)


RGB_HARD_CUT_SOUTH_YAW_DEGREES = 0.0
SOUTH_CHART_DISPLAY_ROTATION_DEGREES = 180


def synchronize_wrapped_latents(latents, centre_x, centre_width, blend_width):
    """Synchronize duplicate ERP edge strips in-place in VAE latent space."""
    if blend_width == 0:
        return latents
    if latents.ndim != 4:
        raise ValueError("latents must have shape [batch, channels, height, width].")
    if (
        centre_x < blend_width
        or centre_width < 2 * blend_width
        or centre_x + centre_width + blend_width > latents.shape[-1]
    ):
        raise ValueError("The requested seam strips do not fit inside the latent canvas.")
    left_main = latents[..., centre_x : centre_x + blend_width].clone()
    right_extension = latents[..., centre_x + centre_width : centre_x + centre_width + blend_width].clone()
    alpha_left = torch.linspace(0.0, 1.0, blend_width, device=latents.device, dtype=latents.dtype).view(1, 1, 1, -1)
    left_merged = right_extension * (1.0 - alpha_left) + left_main * alpha_left
    latents[..., centre_x : centre_x + blend_width] = left_merged
    latents[..., centre_x + centre_width : centre_x + centre_width + blend_width] = left_merged
    right_main_start = centre_x + centre_width - blend_width
    right_main = latents[..., right_main_start : right_main_start + blend_width].clone()
    left_extension = latents[..., centre_x - blend_width : centre_x].clone()
    alpha_right = torch.linspace(1.0, 0.0, blend_width, device=latents.device, dtype=latents.dtype).view(1, 1, 1, -1)
    right_merged = right_main * alpha_right + left_extension * (1.0 - alpha_right)
    latents[..., right_main_start : right_main_start + blend_width] = right_merged
    latents[..., centre_x - blend_width : centre_x] = right_merged
    return latents


def _prepare_edit_inputs(
    pipe, prompt, content, style, seed, num_inference_steps,
    spherical_padding_px=None,
):
    """Run pipeline units and return one independent edit-inference state."""
    height, width = content.height, content.width
    inputs_posi, inputs_nega = {"prompt": prompt}, {"negative_prompt": ""}
    inputs_shared = {
        "cfg_scale": 1.0, "input_image": None, "denoising_strength": 1.0,
        "inpaint_mask": None, "inpaint_blur_size": None, "inpaint_blur_sigma": None,
        "height": height, "width": width, "seed": seed, "rand_device": "cpu",
        "num_inference_steps": num_inference_steps,
        "blockwise_controlnet_inputs": None, "tiled": False, "tile_size": 128,
        "tile_stride": 64, "eligen_entity_prompts": None,
        "eligen_entity_masks": None, "eligen_enable_on_negative": False,
        "edit_image": [content, style], "edit_image_auto_resize": False,
        "edit_rope_interpolation": False, "context_image": None,
        "zero_cond_t": False,
    }
    spherical_embedder_used = False
    for unit in pipe.units:
        if spherical_padding_px is not None and isinstance(unit, QwenImageUnit_EditImageEmbedder):
            unit = _SphericalEditImageEmbedder(spherical_padding_px)
            spherical_embedder_used = True
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )
    if spherical_padding_px is not None and not spherical_embedder_used:
        raise RuntimeError("The pipeline has no supported Qwen edit image embedder.")
    return inputs_shared, inputs_posi, inputs_nega

from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline, ModelConfig, QwenImageUnit_EditImageEmbedder,
)
from telestyle_spherope import spherical_rope_context


class _SphericalEditImageEmbedder(QwenImageUnit_EditImageEmbedder):
    """Encode padded ERP content and an ordinary style image without resizing."""

    def __init__(self, padding_px):
        super().__init__()
        self.padding_px = padding_px

    def process(self, pipe, edit_image, tiled, tile_size, tile_stride, edit_image_auto_resize=False):
        if edit_image_auto_resize or not isinstance(edit_image, (list, tuple)) or len(edit_image) != 2:
            raise ValueError("Spherical editing requires content and style images without auto-resize.")
        pipe.load_models_to_device(self.onload_model_names)
        content, style = edit_image
        tensor = pipe.preprocess_image(content).to(device=pipe.device, dtype=pipe.torch_dtype)
        padded = make_spherical_latent_canvas(tensor, self.padding_px, self.padding_px)
        content_latent = pipe.vae.encode(padded, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pad = self.padding_px // 8
        content_latent = content_latent[
            ..., pad:pad + content.height // 8, pad:pad + content.width // 8
        ].contiguous()
        style_result = super().process(pipe, style, tiled, tile_size, tile_stride, False)
        return {"edit_latents": [content_latent, style_result["edit_latents"]], "edit_image": [content, style]}




class ImageStyleInference:
   
    def __init__(self,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sphere_model = None
        self.sphere_config = None
        self.sphere_checkpoint_info = None

        self._load_models()
    
    def _load_models(self):

          model_dir = "/root/autodl-tmp/Qwen-Image-Edit-2509"
          self.model_dir = model_dir

          self.pipe = QwenImagePipeline.from_pretrained(
              torch_dtype=torch.bfloat16,
              device="cuda",
              model_configs=[
                  ModelConfig(
                      path=glob.glob(
                          os.path.join(
                              model_dir,
                              "transformer/diffusion_pytorch_model*.safetensors",
                          )
                      )
                  ),
                  ModelConfig(
                      path=glob.glob(
                          os.path.join(
                              model_dir,
                              "text_encoder/model*.safetensors",
                          )
                      )
                  ),
                  ModelConfig(
                      path=os.path.join(
                          model_dir,
                          "vae/diffusion_pytorch_model.safetensors",
                      )
                  ),
              ],
              tokenizer_config=None,
              processor_config=ModelConfig(
                  path=os.path.join(model_dir, "processor")
              ),
          )

          telestyle_image = (
              "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors"
          )
          speedup = (
              "weights/"
              "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
          )

          self.pipe.load_lora(self.pipe.dit, telestyle_image)
          self.pipe.load_lora(self.pipe.dit, speedup)

    def load_sphere_adapter(
        self, config_path, checkpoint_path, expected_chart_size=None,
    ):
        """Load an EMA A1 SphereAdapter beside the LoRA-merged DiT."""
        from training.checkpoint import (
            dependency_hashes, sha256_file, validate_adapter_metadata,
        )
        from training.config import load_config
        from training.models import configured_lora_paths
        from training.joint_qwen import JointQwenSphereModel
        from training.sphere_adapter import SphereAdapter

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"A1 checkpoint does not exist: {checkpoint_path}"
            )
        config = load_config(config_path)
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload.get("format") != "telestyle-sphere-adapter-a1-v1":
            raise ValueError("unsupported SphereAdapter checkpoint format.")
        metadata = dict(payload.get("metadata", {}))
        checkpoint_chart_size = int(metadata.get("chart_size", 0))
        if expected_chart_size is not None and (
            checkpoint_chart_size != int(expected_chart_size)
        ):
            raise ValueError(
                "A1 checkpoint chart size does not match hemisphere-size: "
                f"checkpoint={checkpoint_chart_size}, "
                f"hemisphere-size={expected_chart_size}."
            )
        transformer_index = (
            Path(self.model_dir) / "transformer"
            / "diffusion_pytorch_model.safetensors.index.json"
        )
        expected_hash = metadata.get("base_transformer_index_sha256")
        if expected_hash:
            if not transformer_index.is_file():
                raise FileNotFoundError(
                    f"base transformer index does not exist: {transformer_index}"
                )
            actual_hash = sha256_file(transformer_index)
            if actual_hash != expected_hash:
                raise ValueError(
                    "A1 checkpoint was trained against a different base "
                    "transformer index."
                )
        if int(metadata.get("schema_version", 1)) >= 2:
            lora_paths = configured_lora_paths(config, config_path)
            chart_size = checkpoint_chart_size or int(expected_chart_size)
            self.pipe.scheduler.set_timesteps(
                4, denoising_strength=1.0, dynamic_shift_len=(chart_size // 16) ** 2
            )
            validate_adapter_metadata(
                metadata,
                base_transformer_sha256=sha256_file(transformer_index),
                lora_sha256=dependency_hashes(lora_paths),
                prompt=str(config.training.prompt),
                lightning_timesteps=[
                    float(value.detach().cpu())
                    for value in self.pipe.scheduler.timesteps
                ],
                south_yaw_degrees=float(config.training.south_yaw_degrees),
            )
        ema = payload.get("ema", {})
        if "shadow" not in ema:
            raise ValueError("A1 checkpoint does not contain EMA adapter weights.")

        adapter = SphereAdapter(
            hidden_dim=int(config.model.hidden_dim),
            adapter_dim=int(config.model.adapter_dim),
            num_heads=int(config.model.num_heads),
            global_tokens=int(config.model.global_tokens),
            equator_tokens=int(config.model.equator_tokens),
        ).to(device=self.device, dtype=self.pipe.torch_dtype)
        model = JointQwenSphereModel(
            self.pipe.dit, adapter,
            injection_stride=int(config.model.injection_stride),
            use_gradient_checkpointing=False,
        )
        model.adapter.load_state_dict(ema["shadow"], strict=True)
        with torch.no_grad():
            model.injection_gates.copy_(
                payload["injection_gates"].to(
                    device=self.device, dtype=self.pipe.torch_dtype
                )
            )
        model.eval()
        self.sphere_model = model
        self.sphere_config = config
        self.sphere_checkpoint_info = {
            "path": str(checkpoint_path.resolve()),
            "step": int(payload.get("step", 0)),
            "metadata": metadata,
        }
        del payload
        return dict(self.sphere_checkpoint_info)

    def inference(self,
        prompt,
        content_ref,
        style_ref,
        seed=123,
        num_inference_steps=4,
        minedge=1024,
        ):
        w, h = Image.open(content_ref).convert("RGB").size
        minedge=minedge-minedge%16

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        images = [
            Image.open(content_ref).convert("RGB").resize((w, h)),
            Image.open(style_ref).convert("RGB").resize((minedge, minedge)),
        ]

        image = self.pipe(
            prompt, 
            edit_image=images, 
            seed=seed, 
            num_inference_steps=num_inference_steps, 
            height=h, 
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0
        )  # lightning

        return image



    @torch.no_grad()
    def inference_with_latent_seam_sync(self, prompt, content, style, seed, num_inference_steps, centre_x_latent, centre_width_latent, blend_width_latent, return_latents=False):
        """Run Qwen-Image-Edit while synchronizing duplicate ERP latents."""
        pipe = self.pipe
        height, width = content.height, content.width
        pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, dynamic_shift_len=(height // 16) * (width // 16))
        inputs_posi, inputs_nega = {"prompt": prompt}, {"negative_prompt": ""}
        inputs_shared = {"cfg_scale": 1.0, "input_image": None, "denoising_strength": 1.0, "inpaint_mask": None, "inpaint_blur_size": None, "inpaint_blur_sigma": None, "height": height, "width": width, "seed": seed, "rand_device": "cpu", "num_inference_steps": num_inference_steps, "blockwise_controlnet_inputs": None, "tiled": False, "tile_size": 128, "tile_stride": 64, "eligen_entity_prompts": None, "eligen_entity_masks": None, "eligen_enable_on_negative": False, "edit_image": [content, style], "edit_image_auto_resize": False, "edit_rope_interpolation": False, "context_image": None, "zero_cond_t": False}
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        synchronize_wrapped_latents(inputs_shared["latents"], centre_x_latent, centre_width_latent, blend_width_latent)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(pipe.model_fn, 1.0, inputs_shared, inputs_posi, inputs_nega, **models, timestep=timestep, progress_id=progress_id)
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
            synchronize_wrapped_latents(inputs_shared["latents"], centre_x_latent, centre_width_latent, blend_width_latent)
        if return_latents:
            pipe.load_models_to_device([])
            return inputs_shared["latents"]
        pipe.load_models_to_device(["vae"])
        image = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=False)
        image = pipe.vae_output_to_image(image)
        pipe.load_models_to_device([])
        return image


    @torch.no_grad()
    def inference_with_latent_polar_fusion(
        self,
        prompt,
        content_a,
        content_b,
        style,
        seed,
        num_inference_steps,
        centre_x_latent,
        centre_width_latent,
        blend_width_latent,
        rotation_degrees=90.0,
        polar_blend_start_degrees=45.0,
        polar_blend_end_degrees=75.0,
        polar_fusion_steps=2,
        polar_fusion_strength=1.0,
        polar_lowpass_radius_latent=8,
        polar_detail_limiter=True,
        polar_detail_start_degrees=65.0,
        polar_detail_end_degrees=88.0,
        polar_detail_radius_latent=24,
        polar_detail_steps=2,
        return_latents=False,
    ):
        """Use early B predictions to guide A's polar geometry, then refine A alone."""
        if content_a.size != content_b.size:
            raise ValueError("A and B working canvases must have identical dimensions.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if polar_fusion_steps <= 0:
            raise ValueError("polar_fusion_steps must be greater than zero.")
        if not 0.0 <= polar_fusion_strength <= 1.0:
            raise ValueError("polar_fusion_strength must be between zero and one.")
        if polar_lowpass_radius_latent < 0:
            raise ValueError("polar_lowpass_radius_latent cannot be negative.")
        if polar_detail_radius_latent < 0:
            raise ValueError("polar_detail_radius_latent cannot be negative.")
        if polar_detail_steps <= 0:
            raise ValueError("polar_detail_steps must be greater than zero.")
        if not 0 <= polar_detail_start_degrees < polar_detail_end_degrees < 90:
            raise ValueError("polar detail degrees must satisfy 0 <= start < end < 90.")

        pipe = self.pipe
        height, width = content_a.height, content_a.width
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            dynamic_shift_len=(height // 16) * (width // 16),
        )
        scheduler_a = pipe.scheduler
        scheduler_b = copy.deepcopy(scheduler_a)
        inputs_a, inputs_posi_a, inputs_nega_a = _prepare_edit_inputs(
            pipe, prompt, content_a, style, seed, num_inference_steps
        )
        inputs_b, inputs_posi_b, inputs_nega_b = _prepare_edit_inputs(
            pipe, prompt, content_b, style, seed, num_inference_steps
        )
        if inputs_a["latents"].shape != inputs_b["latents"].shape:
            raise ValueError("A and B latent canvases must have identical shapes.")
        latent_width = inputs_a["latents"].shape[-1]
        right_extension_width = latent_width - centre_x_latent - centre_width_latent
        if centre_x_latent < 0 or centre_width_latent <= 0 or right_extension_width < 0:
            raise ValueError("The centre latent region does not fit inside the latent canvas.")

        projector = SphericalLatentProjector(
            inputs_a["latents"].shape[-2],
            centre_width_latent,
            rotation_degrees,
            polar_blend_start_degrees,
            polar_blend_end_degrees,
            inputs_a["latents"].device,
        )
        # Align B's starting noise with A in spherical coordinates.  B keeps
        # its own rotated edit condition, scheduler, and subsequent trajectory.
        inputs_b["latents"] = make_rotated_latent_canvas(
            inputs_a["latents"], projector, centre_x_latent, centre_width_latent
        )
        inputs_b["noise"] = inputs_b["latents"]
        synchronize_wrapped_latents(
            inputs_a["latents"], centre_x_latent, centre_width_latent, blend_width_latent
        )
        synchronize_wrapped_latents(
            inputs_b["latents"], centre_x_latent, centre_width_latent, blend_width_latent
        )
        detail_weight = make_polar_latitude_weight(
            inputs_a["latents"].shape[-2], centre_width_latent,
            polar_detail_start_degrees, polar_detail_end_degrees,
            inputs_a["latents"].device,
        )
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        guidance_steps = min(polar_fusion_steps, num_inference_steps)
        for progress_id, timestep in enumerate(tqdm(scheduler_a.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_a = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0, inputs_a, inputs_posi_a, inputs_nega_a,
                **models, timestep=timestep, progress_id=progress_id
            )
            noise_pred_b = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0, inputs_b, inputs_posi_b, inputs_nega_b,
                **models, timestep=timestep, progress_id=progress_id
            )
            guidance_strength = early_polar_guidance_strength(
                progress_id, guidance_steps, polar_fusion_strength
            )
            if guidance_strength:
                b_centre = noise_pred_b[
                    ..., centre_x_latent : centre_x_latent + centre_width_latent
                ]
                b_aligned = projector.b_to_a(b_centre)
                b_aligned = latitude_adaptive_circular_lowpass(
                    b_aligned, projector.polar_weight, polar_lowpass_radius_latent
                )
                b_canvas = make_circular_latent_canvas(
                    b_aligned, centre_x_latent, right_extension_width
                )
                polar_weight = make_circular_latent_canvas(
                    projector.polar_weight.to(
                        device=noise_pred_a.device, dtype=noise_pred_a.dtype
                    ),
                    centre_x_latent,
                    right_extension_width,
                ) * guidance_strength
                noise_pred_a = noise_pred_a * (1.0 - polar_weight) + b_canvas * polar_weight
            inputs_a["latents"] = pipe.step(
                scheduler_a, progress_id=progress_id, noise_pred=noise_pred_a, **inputs_a
            )
            inputs_b["latents"] = pipe.step(
                scheduler_b, progress_id=progress_id, noise_pred=noise_pred_b, **inputs_b
            )
            if polar_detail_limiter and progress_id >= num_inference_steps - min(polar_detail_steps, num_inference_steps):
                a_centre = inputs_a["latents"][
                    ..., centre_x_latent : centre_x_latent + centre_width_latent
                ]
                a_limited = limit_polar_latent_detail(
                    a_centre, detail_weight, polar_detail_radius_latent
                )
                inputs_a["latents"] = make_circular_latent_canvas(
                    a_limited, centre_x_latent, right_extension_width
                )
            synchronize_wrapped_latents(
                inputs_a["latents"], centre_x_latent, centre_width_latent, blend_width_latent
            )
            synchronize_wrapped_latents(
                inputs_b["latents"], centre_x_latent, centre_width_latent, blend_width_latent
            )
        if return_latents:
            pipe.load_models_to_device([])
            return inputs_a["latents"]
        pipe.load_models_to_device(["vae"])
        image = pipe.vae.decode(inputs_a["latents"], device=pipe.device, tiled=False)
        image = pipe.vae_output_to_image(image)
        pipe.load_models_to_device([])
        return image

    @torch.no_grad()
    def inference_with_spherope(
        self, prompt, content, style, seed, num_inference_steps,
        padding_px=128, return_latents=False, enable_spherope=True,
    ):
        """Denoise a complete ERP with spherical RoPE and padded VAE boundaries."""
        height, width = content.height, content.width
        if height < 16 or width != 2 * height or height % 16:
            raise ValueError("SpheRoPE requires a 2:1 ERP with dimensions divisible by 16.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if padding_px < 0 or padding_px % 16 or padding_px > min(height, width):
            raise ValueError("padding_px must be nonnegative, divisible by 16, and fit the ERP dimensions.")
        pipe = self.pipe
        pipe.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=1.0,
            dynamic_shift_len=(height // 16) * (width // 16),
        )
        try:
            shared, posi, nega = _prepare_edit_inputs(
                pipe, prompt, content, style, seed, num_inference_steps,
                spherical_padding_px=padding_px,
            )
            if shared["latents"].shape != shared["edit_latents"][0].shape:
                raise ValueError("ERP content and output latent shapes must match.")
            pipe.load_models_to_device(pipe.in_iteration_models)
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            with spherical_rope_context(pipe.dit, enabled=enable_spherope):
                for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="TeleStyle SpheRoPE")):
                    timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    prediction = pipe.cfg_guided_model_fn(
                        pipe.model_fn, 1.0, shared, posi, nega,
                        **models, timestep=timestep, progress_id=progress_id,
                    )
                    shared["latents"] = pipe.step(
                        pipe.scheduler, progress_id=progress_id, noise_pred=prediction, **shared
                    )
            if return_latents:
                return shared["latents"]
            pad = padding_px // 8
            latent = make_spherical_latent_canvas(shared["latents"], pad, pad)
            pipe.load_models_to_device(["vae"])
            decoded = pipe.vae.decode(latent, device=pipe.device, tiled=False)
            image = pipe.vae_output_to_image(decoded)
            return image.crop((padding_px, padding_px, padding_px + width, padding_px + height))
        finally:
            pipe.load_models_to_device([])

    @torch.no_grad()
    def inference_with_hemisphere_latent_sync(
        self,
        prompt,
        content_north,
        content_south,
        style,
        seed,
        num_inference_steps,
        output_height,
        output_width,
        overlap_degrees=15.0,
        decode_padding_latent=16,
        return_latents=False,
    ):
        """Denoise overlapping polar charts and decode one synchronized ERP."""
        if content_north.size != content_south.size:
            raise ValueError("north and south hemisphere charts must have identical sizes.")
        if content_north.width != content_north.height or content_north.width % 16:
            raise ValueError("hemisphere charts must be square and divisible by 16.")
        if output_height <= 0 or output_width <= 0:
            raise ValueError("ERP output dimensions must be positive.")
        if output_height % 16 or output_width % 16:
            raise ValueError("ERP output dimensions must be divisible by 16.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if decode_padding_latent < 0:
            raise ValueError("decode_padding_latent cannot be negative.")

        pipe = self.pipe
        chart_size = content_north.width
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
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

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(tqdm(scheduler_north.timesteps)):
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
                noise_pred=noise_pred_north, **inputs_north
            )
            inputs_south["latents"] = pipe.step(
                scheduler_south, progress_id=progress_id,
                noise_pred=noise_pred_south, **inputs_south
            )
            inputs_north["latents"], inputs_south["latents"] = projector.synchronize(
                inputs_north["latents"], inputs_south["latents"]
            )

        erp_latents = projector.compose_erp(
            inputs_north["latents"], inputs_south["latents"],
            output_height // 8, output_width // 8,
        )
        if return_latents:
            pipe.load_models_to_device([])
            return erp_latents
        if decode_padding_latent > min(erp_latents.shape[-2:]):
            raise ValueError("decode padding cannot exceed the ERP latent dimensions.")
        decode_latents = make_spherical_latent_canvas(
            erp_latents, decode_padding_latent, decode_padding_latent
        )
        pipe.load_models_to_device(["vae"])
        image = pipe.vae.decode(decode_latents, device=pipe.device, tiled=False)
        image = pipe.vae_output_to_image(image)
        pipe.load_models_to_device([])

        padding_px = decode_padding_latent * 8
        if padding_px:
            image = image.crop((
                padding_px, padding_px,
                padding_px + output_width, padding_px + output_height,
            ))
        return image


    @torch.no_grad()
    def inference_with_hemisphere_rgb_hard_cut(
        self,
        prompt,
        content_north,
        content_south,
        style,
        seed,
        num_inference_steps,
        output_height,
        output_width,
        overlap_degrees=15.0,
        consistency_degrees=10.0,
        return_latents=False,
        color_match_degrees=6.0,
        return_hard_cut_baseline=False,
    ):
        """Denoise independent charts and locally match the RGB hard cut."""
        if content_north.size != content_south.size:
            raise ValueError("north and south hemisphere charts must have identical sizes.")
        if content_north.width != content_north.height or content_north.width % 16:
            raise ValueError("hemisphere charts must be square and divisible by 16.")
        if output_height <= 0 or output_width <= 0:
            raise ValueError("ERP output dimensions must be positive.")
        if output_height % 16 or output_width % 16:
            raise ValueError("ERP output dimensions must be divisible by 16.")
        if output_width != 2 * output_height:
            raise ValueError("RGB hard-cut output must have a 2:1 aspect ratio.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if not 0.0 <= color_match_degrees <= overlap_degrees:
            raise ValueError(
                "color_match_degrees must be between zero and overlap_degrees."
            )

        from training.geometry import (
            match_equatorial_low_frequency, reproject_native_charts,
        )

        pipe = self.pipe
        chart_size = content_north.width
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
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
        for progress_id, timestep in enumerate(
            tqdm(scheduler_north.timesteps, desc="TeleStyle RGB hard-cut (no A1)")
        ):
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
                noise_pred=noise_pred_north, **inputs_north
            )
            inputs_south["latents"] = pipe.step(
                scheduler_south, progress_id=progress_id,
                noise_pred=noise_pred_south, **inputs_south
            )

        if return_latents:
            pipe.load_models_to_device([])
            return inputs_north["latents"], inputs_south["latents"]
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
            # Keep the native chart longitude convention for final ERP composition.
            south_yaw_degrees=RGB_HARD_CUT_SOUTH_YAW_DEGREES,
        )
        hard_cut_image = (
            pipe.vae_output_to_image(projected.hard_cut)
            if return_hard_cut_baseline else None
        )
        matched = match_equatorial_low_frequency(
            projected.north, projected.south, projected.hard_cut,
            projected.latitude_degrees, color_match_degrees,
        )
        image = pipe.vae_output_to_image(matched)
        del north_decoded, south_decoded, projected, matched
        pipe.load_models_to_device([])
        if return_hard_cut_baseline:
            return image, hard_cut_image
        return image


    @torch.no_grad()
    def inference_with_hemisphere_sphere_adapter(
        self,
        prompt,
        content_north,
        content_south,
        style,
        seed,
        num_inference_steps,
        output_height,
        output_width,
        overlap_degrees=15.0,
        decode_padding_latent=16,
        return_latents=False,
        return_chart_images=False,
    ):
        """Denoise independent hemispheres and hard-cut decoded RGB charts."""
        if return_latents and return_chart_images:
            raise ValueError(
                "return_latents and return_chart_images cannot both be enabled."
            )
        if self.sphere_model is None or self.sphere_config is None:
            raise RuntimeError("load_sphere_adapter must be called before A1 inference.")
        if content_north.size != content_south.size:
            raise ValueError("north and south hemisphere charts must have identical sizes.")
        if content_north.width != content_north.height or content_north.width % 16:
            raise ValueError("hemisphere charts must be square and divisible by 16.")
        if output_height <= 0 or output_width <= 0:
            raise ValueError("ERP output dimensions must be positive.")
        if output_height % 16 or output_width % 16:
            raise ValueError("ERP output dimensions must be divisible by 16.")
        if output_width != 2 * output_height:
            raise ValueError("A1 RGB hard-cut output must have a 2:1 aspect ratio.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero.")
        if decode_padding_latent < 0:
            raise ValueError("decode padding cannot be negative.")

        from training.geometry import build_sphere_geometry, reproject_native_charts

        pipe = self.pipe
        chart_size = content_north.width
        checkpoint_chart_size = int(
            self.sphere_checkpoint_info["metadata"].get("chart_size", 0)
        )
        if checkpoint_chart_size != chart_size:
            raise ValueError(
                "A1 checkpoint chart size does not match the input charts: "
                f"checkpoint={checkpoint_chart_size}, charts={chart_size}."
            )
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            dynamic_shift_len=(chart_size // 16) ** 2,
        )
        scheduler_north = pipe.scheduler
        scheduler_south = copy.deepcopy(scheduler_north)
        inputs_north, posi_north, _ = _prepare_edit_inputs(
            pipe, prompt, content_north, style, seed, num_inference_steps
        )
        inputs_south, posi_south, _ = _prepare_edit_inputs(
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
        geometry = build_sphere_geometry(
            latent_size // 2,
            overlap_degrees,
            float(self.sphere_config.data.consistency_degrees),
            int(self.sphere_config.model.correspondence_neighbors),
            inputs_north["latents"].device,
        )

        pipe.load_models_to_device(pipe.in_iteration_models)
        for progress_id, timestep in enumerate(
            tqdm(scheduler_north.timesteps, desc="TeleStyle + A1")
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            noise_pred_north, noise_pred_south = self.sphere_model(
                inputs_north["latents"],
                inputs_south["latents"],
                inputs_north["edit_latents"],
                inputs_south["edit_latents"],
                posi_north["prompt_emb"],
                posi_south["prompt_emb"],
                posi_north["prompt_emb_mask"],
                posi_south["prompt_emb_mask"],
                timestep,
                geometry,
                disable_cross_chart=False,
            )
            inputs_north["latents"] = pipe.step(
                scheduler_north, progress_id=progress_id,
                noise_pred=noise_pred_north, **inputs_north
            )
            inputs_south["latents"] = pipe.step(
                scheduler_south, progress_id=progress_id,
                noise_pred=noise_pred_south, **inputs_south
            )

        if return_latents:
            pipe.load_models_to_device([])
            return inputs_north["latents"], inputs_south["latents"]

        pipe.load_models_to_device(["vae"])
        north_decoded = pipe.vae.decode(
            inputs_north["latents"], device=pipe.device, tiled=False
        )
        south_decoded = pipe.vae.decode(
            inputs_south["latents"], device=pipe.device, tiled=False
        )
        projected = reproject_native_charts(
            north_decoded,
            south_decoded,
            overlap_degrees,
            float(self.sphere_config.data.consistency_degrees),
            output_height,
            output_width,
            antialias_scale=2,
            # Keep the native chart longitude convention for final ERP composition.
            south_yaw_degrees=RGB_HARD_CUT_SOUTH_YAW_DEGREES,
        )
        image = pipe.vae_output_to_image(projected.hard_cut)
        north_image = None
        south_image = None
        if return_chart_images:
            north_image = pipe.vae_output_to_image(north_decoded)
            south_image = pipe.vae_output_to_image(south_decoded).rotate(
                SOUTH_CHART_DISPLAY_ROTATION_DEGREES
            )
        del north_decoded, south_decoded, projected
        pipe.load_models_to_device([])
        if return_chart_images:
            return image, north_image, south_image
        return image


if __name__ == "__main__":
    inference_engine = ImageStyleInference()

    prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
        
    content_ref = "inputs/content.png"
    style_ref = "inputs/style.jpg"
    
    with torch.no_grad():
        generated_image = inference_engine.inference(prompt, content_ref, style_ref, seed=123, num_inference_steps=4, minedge=1024)

    save_dir=f'./qwen_style_output/'

    os.makedirs(save_dir,exist_ok=True)
    prefix=style_ref.split('/')[-1].split('.')[0]


    generated_image.save(os.path.join(save_dir, f'{prefix}_result.png'))


    print(f"saved to {os.path.join(save_dir, f'{prefix}_result.png')}")
            
