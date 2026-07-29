"""Synchronized CPU/CUDA phase timing and allocator memory records."""

from __future__ import annotations

from contextlib import AbstractContextManager
from time import perf_counter

import torch

from ..types import MemoryRecord, TimingRecord


class PhaseTimer(AbstractContextManager):
    def __init__(self, phase: str, *, device: torch.device | str, warmup: bool = False) -> None:
        self.phase = phase
        self.device = torch.device(device)
        self.warmup = warmup
        self.record: TimingRecord | None = None

    def __enter__(self):
        self.cpu_start_time = perf_counter()
        if self.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            self.start_event.record()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            milliseconds = float(self.start_event.elapsed_time(self.end_event))
        else:
            milliseconds = (perf_counter() - self.cpu_start_time) * 1000.0
        cpu_milliseconds = (perf_counter() - self.cpu_start_time) * 1000.0
        self.record = TimingRecord(self.phase, milliseconds, self.device.type, self.warmup, cpu_milliseconds)
        return False


class MemoryTracker(AbstractContextManager):
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.record: MemoryRecord | None = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.record = MemoryRecord(
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
                torch.cuda.max_memory_allocated(self.device),
                torch.cuda.max_memory_reserved(self.device),
            )
        else:
            self.record = MemoryRecord(0, 0, 0, 0)
        return False
