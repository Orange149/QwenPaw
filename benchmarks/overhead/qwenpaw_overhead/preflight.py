"""Read-only prerequisite checks for the benchmark harness."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_QWENPAW_VERSION = "2.0.1"
DEFAULT_PYTHON = Path("/home/orange/.qwenpaw/venv/bin/python")
DEFAULT_WORKING_DIR = Path("/home/orange/.qwenpaw")
DEFAULT_SECRET_DIR = Path("/home/orange/.qwenpaw.secret")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return result.returncode, (result.stdout or result.stderr).strip()


def _allowed_cpus() -> str:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return "unknown"


def _parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = part.split("-", 1)
                result.update(range(int(start), int(end) + 1))
            else:
                result.add(int(part))
        except ValueError:
            return set()
    return result


def _running_qwenpaw_processes() -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    own_pid = os.getpid()
    matches: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or ())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if process.info["pid"] != own_pid and (
            "qwenpaw acp" in cmdline
            or cmdline.rstrip().endswith("/qwenpaw")
            or " qwenpaw tui" in cmdline
        ):
            matches.append(int(process.info["pid"]))
    return sorted(matches)


def run_preflight(
    *,
    python_executable: Path = DEFAULT_PYTHON,
    working_dir: Path = DEFAULT_WORKING_DIR,
    secret_dir: Path = DEFAULT_SECRET_DIR,
    expected_version: str = EXPECTED_QWENPAW_VERSION,
) -> list[Check]:
    """Return checks without changing the host or QwenPaw configuration."""

    checks: list[Check] = []
    if not python_executable.is_file():
        checks.append(Check("python", "fail", f"missing: {python_executable}"))
    else:
        code, version = _run(
            [
                str(python_executable),
                "-c",
                "import importlib.metadata as m; print(m.version('qwenpaw'))",
            ],
        )
        status = "ok" if code == 0 and version == expected_version else "fail"
        checks.append(
            Check(
                "qwenpaw_version",
                status,
                f"found={version or 'unknown'} expected={expected_version}",
            ),
        )

    config = working_dir / "config.json"
    checks.append(
        Check(
            "seed_working_dir",
            "ok" if config.is_file() else "fail",
            str(config),
        ),
    )
    try:
        parsed = json.loads(config.read_text(encoding="utf-8"))
        profiles = parsed.get("agents", {}).get("profiles", {})
        profile_ok = "default" in profiles
    except (OSError, ValueError, TypeError):
        profile_ok = False
    checks.append(
        Check(
            "default_agent",
            "ok" if profile_ok else "fail",
            "default profile is referenced" if profile_ok else "default profile missing",
        ),
    )

    active_model = secret_dir / "providers" / "active_model.json"
    master_key = secret_dir / ".master_key"
    secret_ok = active_model.is_file() and master_key.is_file()
    checks.append(
        Check(
            "encrypted_provider_seed",
            "ok" if secret_ok else "fail",
            f"active_model={active_model.is_file()} master_key={master_key.is_file()}",
        ),
    )

    cgroup = Path("/sys/fs/cgroup/cgroup.controllers")
    checks.append(
        Check(
            "cgroup_v2",
            "ok" if cgroup.is_file() else "warn",
            str(cgroup),
        ),
    )
    allowed_text = _allowed_cpus()
    allowed = _parse_cpu_list(allowed_text)
    target = {0, 1, 2, 3}
    checks.append(
        Check(
            "cpu_affinity",
            "ok" if target.issubset(allowed) else "fail",
            f"allowed={allowed_text} target=0-3",
        ),
    )

    proxies = sorted(key for key in os.environ if key.lower().endswith("_proxy"))
    checks.append(
        Check(
            "proxy_environment",
            "ok",
            "runner will clear: " + (", ".join(proxies) if proxies else "none set"),
        ),
    )

    running = _running_qwenpaw_processes()
    checks.append(
        Check(
            "existing_qwenpaw_processes",
            "warn" if running else "ok",
            f"pids={running}" if running else "none detected",
        ),
    )
    return checks


def preflight_ok(checks: list[Check]) -> bool:
    return all(check.status != "fail" for check in checks)
