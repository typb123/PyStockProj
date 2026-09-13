"""Low-overhead, opt-in phase timing helpers for performance measurement."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator


def phase_seconds(collector: dict | None, phase_name: str) -> float:
    """Return the currently recorded total for one phase."""
    if collector is None:
        return 0.0
    return float(collector.get("phase_totals_seconds", {}).get(phase_name, 0.0))


def record_phase_timing(
    collector: dict | None,
    phase_name: str,
    elapsed_seconds: float,
) -> None:
    """Accumulate a phase duration in a process-local, serializable mapping."""
    if collector is None:
        return
    totals = collector.setdefault("phase_totals_seconds", {})
    counts = collector.setdefault("phase_call_counts", {})
    totals[phase_name] = float(totals.get(phase_name, 0.0)) + float(
        elapsed_seconds
    )
    counts[phase_name] = int(counts.get(phase_name, 0)) + 1


@contextmanager
def phase_timing(collector: dict | None, phase_name: str) -> Iterator[None]:
    """Time a phase only when collection was explicitly requested."""
    if collector is None:
        yield
        return

    started_at = time.perf_counter()
    try:
        yield
    finally:
        record_phase_timing(collector, phase_name, time.perf_counter() - started_at)
