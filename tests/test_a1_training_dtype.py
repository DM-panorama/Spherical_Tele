"""Dtype regression checks for A1 training condition corruption."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from training.train_sphere_adapter import (
    _corrupt_condition,
    _decode_vae_checkpointed,
    _resize_latent_for_decode,
)


class A1TrainingDtypeTests(unittest.TestCase):
    def test_condition_corruption_preserves_latent_dtype(self):
        latent = torch.ones(1, 16, 8, 8, dtype=torch.bfloat16)
        overlap = torch.ones(4, dtype=torch.bool)

        cases = {
            "dropout": 0.05,
            "asymmetric_noise": 0.15,
            "unchanged": 0.50,
        }
        for name, draw in cases.items():
            with self.subTest(name=name), patch(
                "training.train_sphere_adapter.random.random",
                return_value=draw,
            ):
                result = _corrupt_condition(latent, overlap)
            self.assertEqual(result.dtype, latent.dtype)
            self.assertEqual(result.device, latent.device)
            self.assertEqual(result.shape, latent.shape)

    def test_checkpointed_vae_decode_preserves_latent_gradients(self):
        class ToyVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decode_calls = 0

            def decode(self, value, *, device, tiled):
                self.decode_calls += 1
                self.decode_arguments = (device, tiled)
                return value.square()

        vae = ToyVae()
        latent = torch.tensor([2.0], requires_grad=True)
        decoded = _decode_vae_checkpointed(vae, latent, torch.device("cpu"))
        decoded.sum().backward()

        self.assertEqual(vae.decode_arguments, (torch.device("cpu"), False))
        self.assertGreaterEqual(vae.decode_calls, 2)
        self.assertTrue(torch.equal(latent.grad, torch.tensor([4.0])))

    def test_auxiliary_decode_resize_preserves_dtype_and_gradients(self):
        latent = torch.ones(1, 16, 8, 8, dtype=torch.bfloat16, requires_grad=True)

        resized = _resize_latent_for_decode(latent, 1024, 512)
        resized.float().sum().backward()

        self.assertEqual(resized.shape, (1, 16, 4, 4))
        self.assertEqual(resized.dtype, latent.dtype)
        self.assertIsNotNone(latent.grad)
