"""Joint north/south execution wrapper for the frozen Qwen Image DiT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn

from diffsynth.core import gradient_checkpoint_forward

from .geometry import SphereGeometry
from .sphere_adapter import SphereAdapter


@dataclass
class _BranchState:
    image: torch.Tensor
    text: torch.Tensor
    rotary: tuple[torch.Tensor, torch.Tensor]
    conditioning: torch.Tensor
    output_tokens: int
    latent_height: int
    latent_width: int


class JointQwenSphereModel(nn.Module):
    """Run two native chart trajectories through one frozen shared DiT."""

    def __init__(
        self,
        dit: nn.Module,
        adapter: SphereAdapter,
        injection_stride: int = 4,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if injection_stride <= 0:
            raise ValueError("injection_stride must be positive.")
        self.dit = dit
        self.adapter = adapter
        self.injection_stride = injection_stride
        self.use_gradient_checkpointing = use_gradient_checkpointing
        for parameter in self.dit.parameters():
            parameter.requires_grad_(False)
        injection_count = (
            len(self.dit.transformer_blocks) + injection_stride - 1
        ) // injection_stride
        reference = next(self.adapter.parameters())
        self.injection_gates = nn.Parameter(
            torch.zeros(injection_count, device=reference.device, dtype=reference.dtype)
        )

    def _prepare_branch(
        self,
        latents: torch.Tensor,
        edit_latents: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _BranchState:
        if latents.shape[0] != 1:
            raise ValueError("A1 currently requires per-device batch size one.")
        latent_height, latent_width = latents.shape[-2:]
        if latent_height % 2 or latent_width % 2:
            raise ValueError("Qwen latent dimensions must be divisible by two.")
        token_height, token_width = latent_height // 2, latent_width // 2
        output = rearrange(
            latents,
            "B C (H P) (W Q) -> B (H W) (C P Q)",
            H=token_height,
            W=token_width,
            P=2,
            Q=2,
        )
        edit_latents_list = (
            list(edit_latents)
            if isinstance(edit_latents, (list, tuple))
            else [edit_latents]
        )
        if not edit_latents_list:
            raise ValueError("A1 requires at least one edit latent.")
        edit_tokens = []
        image_shapes = [(latents.shape[0], token_height, token_width)]
        for edit_latent in edit_latents_list:
            if edit_latent.shape[0] != latents.shape[0]:
                raise ValueError("edit latent batch sizes must match the output latent.")
            if edit_latent.shape[-2] % 2 or edit_latent.shape[-1] % 2:
                raise ValueError("edit latent dimensions must be divisible by two.")
            edit_height = edit_latent.shape[-2] // 2
            edit_width = edit_latent.shape[-1] // 2
            edit_tokens.append(rearrange(
                edit_latent,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=edit_height,
                W=edit_width,
                P=2,
                Q=2,
            ))
            image_shapes.append(
                (edit_latent.shape[0], edit_height, edit_width)
            )
        image = self.dit.img_in(torch.cat([output, *edit_tokens], dim=1))
        text = self.dit.txt_in(self.dit.txt_norm(prompt_emb))
        conditioning = self.dit.time_text_embed(timestep / 1000, image.dtype)
        text_lengths = prompt_emb_mask.sum(dim=1).tolist()
        rotary = self.dit.pos_embed(image_shapes, text_lengths, latents.device)
        return _BranchState(
            image=image,
            text=text,
            rotary=rotary,
            conditioning=conditioning,
            output_tokens=output.shape[1],
            latent_height=latent_height,
            latent_width=latent_width,
        )

    def _run_block(self, block: nn.Module, state: _BranchState) -> None:
        state.text, state.image = gradient_checkpoint_forward(
            block,
            self.use_gradient_checkpointing,
            False,
            image=state.image,
            text=state.text,
            temb=state.conditioning,
            image_rotary_emb=state.rotary,
            attention_mask=None,
            enable_fp8_attention=False,
            modulate_index=None,
        )

    def _finish(self, state: _BranchState) -> torch.Tensor:
        image = self.dit.norm_out(state.image, state.conditioning)
        image = self.dit.proj_out(image[:, :state.output_tokens])
        return rearrange(
            image,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=state.latent_height // 2,
            W=state.latent_width // 2,
            P=2,
            Q=2,
        )

    def forward(
        self,
        north_latents: torch.Tensor,
        south_latents: torch.Tensor,
        north_edit_latents: torch.Tensor | list[torch.Tensor],
        south_edit_latents: torch.Tensor | list[torch.Tensor],
        north_prompt_emb: torch.Tensor,
        south_prompt_emb: torch.Tensor,
        north_prompt_mask: torch.Tensor,
        south_prompt_mask: torch.Tensor,
        timestep: torch.Tensor,
        geometry: SphereGeometry,
        disable_cross_chart: bool = False,
        adapter_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        north = self._prepare_branch(
            north_latents, north_edit_latents,
            north_prompt_emb, north_prompt_mask, timestep,
        )
        south = self._prepare_branch(
            south_latents, south_edit_latents,
            south_prompt_emb, south_prompt_mask, timestep,
        )
        north_output = north.image[:, :north.output_tokens]
        south_output = south.image[:, :south.output_tokens]
        if adapter_enabled:
            north_output, south_output = self.adapter.add_position(
                north_output, south_output, geometry
            )
        north.image = torch.cat(
            (north_output, north.image[:, north.output_tokens:]), dim=1
        )
        south.image = torch.cat(
            (south_output, south.image[:, south.output_tokens:]), dim=1
        )

        global_state = None
        equator_state = None
        injection_id = 0
        for block_id, block in enumerate(self.dit.transformer_blocks):
            self._run_block(block, north)
            self._run_block(block, south)
            if (block_id + 1) % self.injection_stride != 0 and (
                block_id + 1 != len(self.dit.transformer_blocks)
            ):
                continue
            if adapter_enabled and not disable_cross_chart:
                north_output = north.image[:, :north.output_tokens]
                south_output = south.image[:, :south.output_tokens]
                (
                    north_update,
                    south_update,
                    global_state,
                    equator_state,
                ) = self.adapter(
                    north_output,
                    south_output,
                    geometry,
                    global_state,
                    equator_state,
                )
                gate = torch.tanh(self.injection_gates[injection_id])
                north.image = torch.cat(
                    (
                        north_output + gate * north_update,
                        north.image[:, north.output_tokens:],
                    ),
                    dim=1,
                )
                south.image = torch.cat(
                    (
                        south_output + gate * south_update,
                        south.image[:, south.output_tokens:],
                    ),
                    dim=1,
                )
            injection_id += 1
        return self._finish(north), self._finish(south)
