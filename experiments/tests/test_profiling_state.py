from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation import MemoryTracker, PhaseTimer
from experiments.state import build_portable_state, load_portable_state, restore_rng_state, save_portable_state


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "fake"
        self.creation_date = "now"
        self.output_directory = Path("output")
        self.gaussians = torch.nn.Module()
        self.gaussians.means = torch.nn.Parameter(torch.zeros(2, 3))
        self.gaussians.active_sh_degree = 0
        self.gaussians.max_sh_degree = 3


class ProfilingStateTest(unittest.TestCase):
    def test_cpu_timing_and_memory_smoke(self):
        with PhaseTimer("cpu", device="cpu") as timer:
            _ = torch.ones(8).sum()
        with MemoryTracker("cpu") as memory:
            _ = torch.ones(8)
        self.assertGreaterEqual(timer.record.milliseconds, 0.0)
        self.assertGreaterEqual(timer.record.cpu_milliseconds, 0.0)
        self.assertEqual(memory.record.peak_allocated_bytes, 0)

    def test_portable_state_roundtrip_and_rng(self):
        model = FakeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        state = build_portable_state(model, optimizer, 12, None, {"test": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_portable_state(path, state)
            loaded = load_portable_state(path)
        self.assertEqual(loaded["model_metadata"]["num_iterations_trained"], 12)
        restore_rng_state(loaded["rng_state"])
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        restore_rng_state(loaded["rng_state"])
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
