"""Host and benchmark manifest collection without importing QwenPaw."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "credential",
)


def _read_first(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default


def _mem_total_bytes() -> int | None:
    for line in _read_first(Path("/proc/meminfo")).splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _is_wsl() -> bool:
    text = " ".join(
        (
            platform.release(),
            _read_first(Path("/proc/version")),
            os.environ.get("WSL_DISTRO_NAME", ""),
        ),
    ).lower()
    return "microsoft" in text or "wsl" in text


def _command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def sanitized_config_hash(path: Path) -> str | None:
    """Hash config structure after replacing values of sensitive fields.

    The digest is useful for detecting configuration drift without writing a
    provider key (or even its ciphertext) into benchmark artifacts.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in sorted(item.items()):
                lowered = key.lower()
                if any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS):
                    result[key] = "<redacted>"
                else:
                    result[key] = scrub(child)
            return result
        if isinstance(item, list):
            return [scrub(child) for child in item]
        return item

    payload = json.dumps(
        scrub(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_manifest(
    *,
    run_id: str,
    python_executable: Path,
    seed_working_dir: Path,
    profiles: list[str],
    scenarios: list[str],
    backend: str,
    model: str,
    cpu_affinity: str = "0-3",
) -> dict[str, Any]:
    """Collect stable provenance for one benchmark result directory."""

    # Query the exact benchmark interpreter.  The harness may itself be run
    # from a source checkout or a different Python environment.
    version = _command_version(
        [
            str(python_executable),
            "-c",
            "import importlib.metadata as m; print(m.version('qwenpaw'))",
        ],
    )

    cpu_model = ""
    for line in _read_first(Path("/proc/cpuinfo")).splitlines():
        if line.lower().startswith(("model name", "hardware")):
            cpu_model = line.partition(":")[2].strip()
            break

    return {
        "schema_version": "1",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "wsl": _is_wsl(),
            "cpu_model": cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "memory_total_bytes": _mem_total_bytes(),
            "cgroup_v2": Path("/sys/fs/cgroup/cgroup.controllers").exists(),
        },
        "runtime": {
            "python_executable": str(python_executable),
            "python_version": _command_version([str(python_executable), "--version"]),
            "qwenpaw_version": version,
            "taskset": shutil.which("taskset"),
        },
        "experiment": {
            "backend": backend,
            "provider": {
                "mock": "mock",
                "offline": "mock/local-only",
                "offline+dashscope": "mock/local-only + dashscope",
                "static": "none",
                "dashscope": "dashscope",
            }.get(backend, "unknown"),
            "model": model,
            "profiles": profiles,
            "scenarios": scenarios,
            "cpu_affinity": cpu_affinity,
            "controlled_backend_warmup": False,
            "stock_tui_backend_warmup": "enabled only when TUI phase runs",
            "page_cache_dropped": False,
            "seed_config_sha256": sanitized_config_hash(
                seed_working_dir / "config.json",
            ),
        },
    }
