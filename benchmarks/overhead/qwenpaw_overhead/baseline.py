"""Low-overhead WSL/Linux host baseline sampling."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _psi(resource: str) -> dict[str, float]:
    path = Path("/proc/pressure") / resource
    result: dict[str, float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        prefix = parts[0]
        for part in parts[1:]:
            key, separator, value = part.partition("=")
            if separator and key in {"avg10", "avg60", "avg300", "total"}:
                try:
                    result[f"psi_{resource}_{prefix}_{key}"] = float(value)
                except ValueError:
                    continue
    return result


def sample_system_baseline(
    duration_s: float = 60.0,
    *,
    interval_s: float = 1.0,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Collect whole-WSL CPU, memory and pressure without spawning QwenPaw."""

    if duration_s < 0 or interval_s <= 0:
        raise ValueError("duration_s must be non-negative and interval_s positive")
    import psutil

    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration_s
    psutil.cpu_percent(interval=None)
    while True:
        now = time.monotonic()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        row: dict[str, Any] = {
            "sample_type": "system_baseline",
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_percent_machine_normalized": psutil.cpu_percent(interval=None),
            "load1": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None,
            "memory_total_bytes": memory.total,
            "memory_available_bytes": memory.available,
            "memory_used_bytes": memory.used,
            "swap_used_bytes": swap.used,
        }
        for resource in ("cpu", "memory", "io"):
            row.update(_psi(resource))
        rows.append(row)
        if on_sample is not None:
            on_sample(row)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval_s, remaining))
    return rows

