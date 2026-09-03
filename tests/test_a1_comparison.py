"""CPU checks for A1 comparison modes and contact sheets."""

from __future__ import annotations

import argparse
import unittest

import torch
from PIL import Image

from training.evaluate_a1 import (
    COMPARISON_MODES,
    _adapter_mode,
    _comparison_sheet,
    _validate_args,
)


class _ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_gate = torch.nn.Parameter(torch.tensor(0.25))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = _ToyAdapter()
        self.injection_gates = torch.nn.Parameter(torch.tensor([0.5, -0.5]))


class A1ComparisonTests(unittest.TestCase):
    def test_modes_return_expected_cross_chart_state_and_restore_gates(self):
        model = _ToyModel()
        original_injection = model.injection_gates.detach().clone()
        original_position = model.adapter.position_gate.detach().clone()

        for mode, expected_disabled in (
            ("zero_gate_baseline", True),
            ("adapter_full", False),
            ("cross_chart_off", True),
        ):
            with self.subTest(mode=mode), torch.no_grad():
                with _adapter_mode(model, mode) as disabled:
                    self.assertEqual(disabled, expected_disabled)
                    if mode == "zero_gate_baseline":
                        self.assertEqual(torch.count_nonzero(model.injection_gates), 0)
                        self.assertEqual(float(model.adapter.position_gate.detach()), 0.0)
                self.assertTrue(torch.equal(model.injection_gates, original_injection))
                self.assertTrue(torch.equal(model.adapter.position_gate, original_position))

    def test_zero_gate_mode_restores_gates_after_exception(self):
        model = _ToyModel()
        with self.assertRaisesRegex(RuntimeError, "expected"), torch.no_grad():
            with _adapter_mode(model, "zero_gate_baseline"):
                raise RuntimeError("expected")
        self.assertTrue(
            torch.equal(model.injection_gates, torch.tensor([0.5, -0.5]))
        )
        self.assertEqual(float(model.adapter.position_gate.detach()), 0.25)

    def test_comparison_sheet_has_three_labelled_columns(self):
        images = {
            mode: Image.new("RGB", (32, 16), (index * 40, 0, 0))
            for index, mode in enumerate(COMPARISON_MODES)
        }
        sheet = _comparison_sheet(images)
        self.assertEqual(sheet.size, (96, 48))
        self.assertEqual(sheet.getpixel((16, 40)), (0, 0, 0))
        self.assertEqual(sheet.getpixel((48, 40)), (40, 0, 0))
        self.assertEqual(sheet.getpixel((80, 40)), (80, 0, 0))

    def test_comparison_validation_requires_checkpoint_and_valid_count(self):
        base = {
            "max_samples": 4,
            "save_comparison_images": True,
            "comparison_samples": 4,
            "checkpoint": "adapter.pt",
        }
        _validate_args(argparse.Namespace(**base))
        for update, message in (
            ({"checkpoint": None}, "requires --checkpoint"),
            ({"comparison_samples": 0}, "must be positive"),
            ({"comparison_samples": 5}, "cannot exceed"),
        ):
            values = {**base, **update}
            with self.subTest(update=update), self.assertRaisesRegex(
                ValueError, message
            ):
                _validate_args(argparse.Namespace(**values))


if __name__ == "__main__":
    unittest.main()
