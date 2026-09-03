"""Static regression checks for the A1 training loop structure."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class A1TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (
            Path(__file__).parents[1] / "training" / "train_sphere_adapter.py"
        ).read_text(encoding="utf-8")
        cls.tree = ast.parse(source)

    @staticmethod
    def _calls(node: ast.AST) -> set[str]:
        return {
            ast.unparse(item.func)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }

    def test_backward_and_progress_update_are_inside_accumulation_loop(self):
        loops = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.For)
            and ast.unparse(node.target) == "_"
            and ast.unparse(node.iter) == "range(accumulation)"
        ]
        self.assertEqual(len(loops), 1)
        calls = self._calls(loops[0])
        self.assertTrue(any(name.endswith(".backward") for name in calls))
        self.assertIn("progress_bar.update", calls)

    def test_parameter_group_loop_does_not_process_samples(self):
        loops = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.For)
            and "optimizer.param_groups" in ast.unparse(node.iter)
        ]
        self.assertEqual(len(loops), 1)
        calls = self._calls(loops[0])
        self.assertFalse(any(name.endswith(".backward") for name in calls))

    def test_learning_rate_warmup_is_configurable(self):
        source = ast.unparse(self.tree)
        self.assertIn("config.training.get('lr_warmup_steps', 500)", source)
        self.assertIn("(step + 1) / lr_warmup_steps", source)
