import copy
import torch
import os
import glob
from PIL import Image
from tqdm import tqdm

from telestyle_spherical import (
    SphericalLatentProjector,
    early_polar_guidance_strength,
    latitude_adaptive_circular_lowpass,
    make_circular_latent_canvas,
    make_rotated_latent_canvas,
)


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


def _prepare_edit_inputs(pipe, prompt, content, style, seed, num_inference_steps):
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
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )
    return inputs_shared, inputs_posi, inputs_nega

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig



class ImageStyleInference:
   
    def __init__(self,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_models()
    
    def _load_models(self):

          model_dir = "/root/autodl-tmp/Qwen-Image-Edit-2509"

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
    def inference_with_latent_seam_sync(self, prompt, content, style, seed, num_inference_steps, centre_x_latent, centre_width_latent, blend_width_latent):
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
            synchronize_wrapped_latents(
                inputs_a["latents"], centre_x_latent, centre_width_latent, blend_width_latent
            )
            synchronize_wrapped_latents(
                inputs_b["latents"], centre_x_latent, centre_width_latent, blend_width_latent
            )
        pipe.load_models_to_device(["vae"])
        image = pipe.vae.decode(inputs_a["latents"], device=pipe.device, tiled=False)
        image = pipe.vae_output_to_image(image)
        pipe.load_models_to_device([])
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
            
