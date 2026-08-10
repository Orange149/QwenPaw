"""Best-effort Windows host metrics for WSL benchmark context."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_POWERSHELL_QUERY = r"""
$items = @(
  Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -in @("VmmemWSL", "vmmem") }
)
[pscustomobject]@{
  process_count = $items.Count
  working_set_bytes = ($items | Measure-Object -Property WorkingSet64 -Sum).Sum
} | ConvertTo-Json -Compress
""".strip()


def _is_wsl() -> bool:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False
    return "microsoft" in release.casefold()


def sample_vmmemwsl(
    *,
    powershell_executable: str | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Return a coarse VmmemWSL snapshot without exposing host process data."""

    observed = datetime.now(timezone.utc).isoformat()
    if not _is_wsl():
        return {
            "available": False,
            "source": "windows_vmmemwsl_working_set",
            "observed_at_utc": observed,
            "error_type": "NotWSL",
        }
    executable = powershell_executable or shutil.which("powershell.exe")
    if not executable:
        return {
            "available": False,
            "source": "windows_vmmemwsl_working_set",
            "observed_at_utc": observed,
            "error_type": "PowerShellUnavailable",
        }
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _POWERSHELL_QUERY,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "source": "windows_vmmemwsl_working_set",
            "observed_at_utc": observed,
            "error_type": type(exc).__name__,
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "source": "windows_vmmemwsl_working_set",
            "observed_at_utc": observed,
            "error_type": "PowerShellQueryFailed",
            "returncode": completed.returncode,
        }
    try:
        payload = json.loads(completed.stdout.strip())
        process_count = int(payload["process_count"])
        working_set = int(payload["working_set_bytes"] or 0)
        if process_count < 1 or working_set < 1:
            raise ValueError("VmmemWSL process was not found")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "source": "windows_vmmemwsl_working_set",
            "observed_at_utc": observed,
            "error_type": type(exc).__name__,
        }
    return {
        "available": True,
        "source": "windows_vmmemwsl_working_set",
        "observed_at_utc": observed,
        "process_count": process_count,
        "working_set_bytes": working_set,
    }


def vmmemwsl_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Build an explicitly coarse, signed before/after observation."""

    available = before.get("available") is True and after.get("available") is True
    before_bytes = before.get("working_set_bytes") if available else None
    after_bytes = after.get("working_set_bytes") if available else None
    return {
        "available": available,
        "attribution": "whole_wsl_vm_not_qwenpaw_process",
        "before": before,
        "after": after,
        "working_set_before_bytes": before_bytes,
        "working_set_after_bytes": after_bytes,
        "working_set_delta_bytes": (
            int(after_bytes) - int(before_bytes) if available else None
        ),
    }


__all__ = ["sample_vmmemwsl", "vmmemwsl_delta"]
