"""Disk-footprint, portability, and result-secret inspection helpers.

All discovery functions are read-only.  The aarch64 resolver probe is also
read-only by default: callers receive the exact pip command and must pass
``execute=True`` before a subprocess (and therefore network access) is used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import re
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Iterable, Iterator, Mapping, MutableSet, Sequence
from pathlib import Path

from .schemas import (
    ArmDryRunResult,
    DistributionRecord,
    FootprintRecord,
    FootprintTarget,
    SecretFinding,
    SharedObjectRecord,
    StateLayer,
)

_BLOCK_SIZE = 512  # POSIX st_blocks is expressed in 512-byte blocks.
_SECRET_ENV_NAME = re.compile(
    r"(?:api[_-]?key|token|secret|password|passwd|authorization|credential)",
    re.IGNORECASE,
)
_REDACTED_VALUES = {
    "",
    "***",
    "<redacted>",
    "redacted",
    "masked",
    "none",
    "null",
}
_FORBIDDEN_SECRET_FILENAMES = {
    ".env",
    ".master_key",
    "credentials.yaml",
    "envs.json",
}
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_key",
        re.compile(r"\b(?:sk|sk-ws|sk-or-v1)-[A-Za-z0-9_-]{12,}\b"),
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "bearer_token",
        re.compile(
            r"\bBearer\s+([A-Za-z0-9._~+/=-]{12,})",
            re.IGNORECASE,
        ),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
    (
        "generic_secret_assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|token|secret|password|passwd)"
            r"\b\s*[:=]\s*[\"']?([^\"'\s,;}]{8,})",
            re.IGNORECASE,
        ),
    ),
)
_OUTPUT_REDACTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(https?://[^:/\s]+:)[^@/\s]+@"),
    re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:sk|sk-ws|sk-or-v1)-[A-Za-z0-9_-]{8,}\b"),
)


def _inode_key(file_stat: os.stat_result) -> tuple[int, int]:
    return (file_stat.st_dev, file_stat.st_ino)


def _allocated_bytes(file_stat: os.stat_result) -> int:
    return int(getattr(file_stat, "st_blocks", 0)) * _BLOCK_SIZE


def _is_shared_object(path: Path) -> bool:
    name = path.name
    return name.endswith(".so") or ".so." in name


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def measure_tree(
    target: FootprintTarget,
    *,
    seen_inodes: MutableSet[tuple[int, int]] | None = None,
) -> FootprintRecord:
    """Measure a target without following symlinks.

    Hard-linked entries are charged once.  Supplying ``seen_inodes`` extends
    that de-duplication across several disjoint categories; callers that need
    overlapping rollups should give each rollup a fresh set.
    """

    root = Path(target.path).expanduser()
    present = root.exists() or root.is_symlink()
    if not present:
        return FootprintRecord(
            category=target.category,
            path=str(root),
            layer=target.layer,
            present=False,
            rollup=target.rollup,
            excluded_root_names=tuple(target.exclude_root_names),
            error=None if target.optional else "required path is missing",
        )

    seen = seen_inodes if seen_inodes is not None else set()
    allocated = 0
    apparent = 0
    files = 0
    directories = 0
    symlinks = 0
    other = 0
    unique_inodes = 0
    duplicate_inodes = 0
    inaccessible = 0
    first_error: str | None = None
    stack = [root]

    while stack:
        path = stack.pop()
        try:
            file_stat = path.lstat()
        except OSError as exc:
            inaccessible += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
            continue

        key = _inode_key(file_stat)
        if key in seen:
            duplicate_inodes += 1
            continue
        seen.add(key)
        unique_inodes += 1
        allocated += _allocated_bytes(file_stat)
        apparent += int(file_stat.st_size)

        mode = file_stat.st_mode
        if stat.S_ISREG(mode):
            files += 1
        elif stat.S_ISDIR(mode):
            directories += 1
            try:
                with os.scandir(path) as entries:
                    stack.extend(
                        Path(entry.path)
                        for entry in entries
                        if not (
                            path == root
                            and entry.name in set(target.exclude_root_names)
                        )
                    )
            except OSError as exc:
                inaccessible += 1
                first_error = first_error or f"{type(exc).__name__}: {exc}"
        elif stat.S_ISLNK(mode):
            symlinks += 1
        else:
            other += 1

    return FootprintRecord(
        category=target.category,
        path=str(root),
        layer=target.layer,
        present=True,
        rollup=target.rollup,
        excluded_root_names=tuple(target.exclude_root_names),
        allocated_bytes=allocated,
        apparent_bytes=apparent,
        file_count=files,
        directory_count=directories,
        symlink_count=symlinks,
        other_entry_count=other,
        unique_inode_count=unique_inodes,
        duplicate_inode_count=duplicate_inodes,
        inaccessible_entry_count=inaccessible,
        error=first_error,
    )


def measure_targets(
    targets: Iterable[FootprintTarget],
    *,
    deduplicate_across_targets: bool = True,
) -> list[FootprintRecord]:
    """Measure targets in input order with deterministic inode ownership."""

    shared_seen: set[tuple[int, int]] = set()
    records: list[FootprintRecord] = []
    for target in targets:
        if target.rollup or not deduplicate_across_targets:
            seen: set[tuple[int, int]] = set()
        else:
            seen = shared_seen
        records.append(measure_tree(target, seen_inodes=seen))
    return records


def classify_state_path(
    path: str | os.PathLike[str],
    *,
    working_dir: str | os.PathLike[str],
    install_root: str | os.PathLike[str] | None = None,
    secret_dir: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    cache_roots: Iterable[str | os.PathLike[str]] = (),
) -> StateLayer:
    """Classify a QwenPaw-owned or adjacent path by persistence role."""

    candidate = Path(path).expanduser()
    working = Path(working_dir).expanduser()
    if secret_dir and _within(candidate, Path(secret_dir).expanduser()):
        return StateLayer.SECRETS
    if install_root and _within(candidate, Path(install_root).expanduser()):
        return StateLayer.INSTALL
    if state_dir and _within(candidate, Path(state_dir).expanduser()):
        return StateLayer.LOGS
    if any(_within(candidate, Path(root).expanduser()) for root in cache_roots):
        return StateLayer.CACHE
    if _within(candidate, working / "local_models") or _within(
        candidate,
        working / "models",
    ):
        return StateLayer.MODELS
    if candidate.name.endswith(".log") or _within(candidate, working / "logs"):
        return StateLayer.LOGS
    if _within(candidate, working / "npm-cache"):
        return StateLayer.CACHE
    if _within(candidate, working):
        return StateLayer.MUTABLE_STATE
    return StateLayer.EXTERNAL


def _find_qwenpaw_package() -> Path | None:
    spec = importlib.util.find_spec("qwenpaw")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def default_footprint_targets(
    working_dir: str | os.PathLike[str],
    *,
    install_root: str | os.PathLike[str] | None = None,
    package_dir: str | os.PathLike[str] | None = None,
    secret_dir: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
    python_stdlib: str | os.PathLike[str] | None = None,
) -> list[FootprintTarget]:
    """Build the standard install/state/cache footprint target list."""

    working = Path(working_dir).expanduser()
    install = Path(install_root or sys.prefix).expanduser()
    package = Path(package_dir).expanduser() if package_dir else _find_qwenpaw_package()
    user_home = Path(home).expanduser() if home else Path.home()
    secrets = Path(secret_dir).expanduser() if secret_dir else Path(f"{working}.secret")
    paw_state = (
        Path(state_dir).expanduser()
        if state_dir
        else user_home / ".local" / "state" / "paw"
    )
    python_entrypoint = Path(python_executable or sys.executable).expanduser()
    stdlib = Path(
        python_stdlib or sysconfig.get_path("stdlib") or "/missing-python-stdlib",
    ).expanduser()

    targets = [
        FootprintTarget(
            "install.venv_total",
            str(install),
            StateLayer.INSTALL,
            rollup=True,
            optional=False,
        ),
        FootprintTarget(
            "runtime.python_entrypoint",
            str(python_entrypoint),
            StateLayer.INSTALL,
            rollup=True,
            optional=False,
        ),
        FootprintTarget(
            "runtime.python_executable_resolved",
            str(python_entrypoint.resolve(strict=False)),
            StateLayer.INSTALL,
            rollup=True,
            optional=False,
        ),
        FootprintTarget(
            "runtime.python_stdlib",
            str(stdlib),
            StateLayer.INSTALL,
            rollup=True,
            optional=False,
        ),
        FootprintTarget(
            "state.total",
            str(working),
            StateLayer.MUTABLE_STATE,
            rollup=True,
            optional=False,
            exclude_root_names=("venv", "bin"),
        ),
    ]
    if package is not None:
        targets.extend(
            [
                FootprintTarget(
                    "install.qwenpaw_distribution",
                    str(package),
                    StateLayer.INSTALL,
                    rollup=True,
                ),
                FootprintTarget(
                    "install.console",
                    str(package / "console"),
                    StateLayer.INSTALL,
                    rollup=True,
                ),
                FootprintTarget(
                    "install.tokenizer",
                    str(package / "tokenizer"),
                    StateLayer.INSTALL,
                    rollup=True,
                ),
                FootprintTarget(
                    "install.bundled_skills",
                    str(package / "agents" / "skills"),
                    StateLayer.INSTALL,
                    rollup=True,
                ),
                FootprintTarget(
                    "install.docs",
                    str(package / "docs"),
                    StateLayer.INSTALL,
                    rollup=True,
                ),
            ],
        )

    state_categories = (
        ("state.workspaces", working / "workspaces", StateLayer.MUTABLE_STATE),
        ("state.skill_pool", working / "skill_pool", StateLayer.MUTABLE_STATE),
        ("state.governance", working / "governance", StateLayer.MUTABLE_STATE),
        ("state.plugins", working / "plugins", StateLayer.MUTABLE_STATE),
        ("state.media", working / "media", StateLayer.MUTABLE_STATE),
        ("state.local_models", working / "local_models", StateLayer.MODELS),
        ("state.models", working / "models", StateLayer.MODELS),
        ("state.npm_cache", working / "npm-cache", StateLayer.CACHE),
        ("state.qwenpaw_log", working / "qwenpaw.log", StateLayer.LOGS),
        ("state.tui_logs", paw_state, StateLayer.LOGS),
        ("state.secrets", secrets, StateLayer.SECRETS),
        (
            "state.backups",
            Path(f"{working}.backups"),
            StateLayer.MUTABLE_STATE,
        ),
        (
            "cache.huggingface",
            user_home / ".cache" / "huggingface",
            StateLayer.CACHE,
        ),
        (
            "cache.modelscope",
            user_home / ".cache" / "modelscope",
            StateLayer.CACHE,
        ),
        (
            "cache.playwright",
            user_home / ".cache" / "ms-playwright",
            StateLayer.CACHE,
        ),
    )
    targets.extend(
        FootprintTarget(category, str(path), layer, rollup=True)
        for category, path, layer in state_categories
    )
    targets.extend(_workspace_state_targets(working))
    for path in sorted(working.glob("token_usage*")):
        targets.append(
            FootprintTarget(
                f"state.token_usage.{path.name}",
                str(path),
                StateLayer.MUTABLE_STATE,
                rollup=True,
            ),
        )
    return targets


def _workspace_state_targets(working_dir: Path) -> list[FootprintTarget]:
    """Discover independently reportable session/history/memory state."""

    workspaces = working_dir / "workspaces"
    if not workspaces.is_dir():
        return []
    targets: list[FootprintTarget] = []
    for workspace in sorted(
        (path for path in workspaces.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", workspace.name)
        prefix = f"workspace.{label}"
        targets.append(
            FootprintTarget(
                f"{prefix}.total",
                str(workspace),
                StateLayer.MUTABLE_STATE,
                rollup=True,
            ),
        )
        fixed_paths = (
            ("sessions", workspace / "sessions"),
            ("history_db", workspace / "history.db"),
            ("memory", workspace / "memory"),
            ("memory_markdown", workspace / "MEMORY.md"),
        )
        for suffix, path in fixed_paths:
            targets.append(
                FootprintTarget(
                    f"{prefix}.{suffix}",
                    str(path),
                    StateLayer.MUTABLE_STATE,
                    rollup=True,
                ),
            )
        discovered: set[Path] = set()
        for pattern in (
            "history.db-*",
            "mem_*",
            "token_usage*",
            "*.log",
            "*.log.*",
        ):
            discovered.update(workspace.glob(pattern))
        for path in sorted(discovered, key=lambda item: item.name):
            suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)
            layer = StateLayer.LOGS if ".log" in path.name else StateLayer.MUTABLE_STATE
            targets.append(
                FootprintTarget(
                    f"{prefix}.{suffix}",
                    str(path),
                    layer,
                    rollup=True,
                ),
            )
    return targets


def _default_allowed_root(site_packages: Path) -> Path:
    # Typical layout: <venv>/lib/python3.12/site-packages.  Falling back to
    # site-packages itself remains safe for non-venv layouts.
    try:
        return site_packages.parents[2]
    except IndexError:
        return site_packages


def scan_distributions(
    site_packages: str | os.PathLike[str],
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> list[DistributionRecord]:
    """Measure installed distributions using their wheel ``RECORD`` files."""

    site_path = Path(site_packages).expanduser().resolve()
    allowed = (
        Path(allowed_root).expanduser().resolve()
        if allowed_root
        else _default_allowed_root(site_path)
    )
    distributions = sorted(
        importlib.metadata.distributions(path=[str(site_path)]),
        key=lambda item: (
            (item.metadata.get("Name") or "").casefold(),
            item.version or "",
        ),
    )
    globally_seen: set[tuple[int, int]] = set()
    records: list[DistributionRecord] = []

    for distribution in distributions:
        allocated = 0
        apparent = 0
        files = 0
        native_files = 0
        duplicates = 0
        missing = 0
        outside = 0
        for relative in distribution.files or ():
            candidate = Path(distribution.locate_file(relative))
            resolved = candidate.resolve(strict=False)
            if not _within(resolved, allowed):
                outside += 1
                continue
            try:
                file_stat = candidate.lstat()
            except OSError:
                missing += 1
                continue
            if stat.S_ISDIR(file_stat.st_mode):
                continue
            key = _inode_key(file_stat)
            if key in globally_seen:
                duplicates += 1
                continue
            globally_seen.add(key)
            files += 1
            allocated += _allocated_bytes(file_stat)
            apparent += int(file_stat.st_size)
            if _is_shared_object(candidate):
                native_files += 1

        records.append(
            DistributionRecord(
                name=distribution.metadata.get("Name") or "unknown",
                version=distribution.version or "",
                location=str(Path(distribution.locate_file("")).resolve()),
                allocated_bytes=allocated,
                apparent_bytes=apparent,
                file_count=files,
                native_file_count=native_files,
                duplicate_inode_count=duplicates,
                missing_file_count=missing,
                outside_root_file_count=outside,
            ),
        )
    return records


def distribution_file_index(
    site_packages: str | os.PathLike[str],
) -> dict[str, str]:
    """Return absolute installed-file path to normalized distribution name."""

    site_path = Path(site_packages).expanduser().resolve()
    index: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_path)]):
        name = distribution.metadata.get("Name") or "unknown"
        for relative in distribution.files or ():
            candidate = Path(distribution.locate_file(relative))
            index.setdefault(str(candidate.resolve(strict=False)), name)
    return index


_ELF_MACHINES = {
    3: "x86",
    8: "mips",
    20: "powerpc",
    21: "powerpc64",
    40: "arm",
    62: "x86_64",
    183: "aarch64",
    243: "riscv",
}


def parse_elf_identity(
    path: str | os.PathLike[str],
) -> tuple[int | None, str | None, int | None]:
    """Return ``(ELF class bits, machine name, machine id)``."""

    try:
        with Path(path).open("rb") as file_obj:
            header = file_obj.read(20)
    except OSError:
        return (None, None, None)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return (None, None, None)
    elf_class = {1: 32, 2: 64}.get(header[4])
    byte_order = {1: "little", 2: "big"}.get(header[5])
    if byte_order is None:
        return (elf_class, None, None)
    machine_id = int.from_bytes(header[18:20], byte_order)
    return (
        elf_class,
        _ELF_MACHINES.get(machine_id, f"machine_{machine_id}"),
        machine_id,
    )


def _iter_shared_objects(root: Path) -> Iterator[Path]:
    if root.is_file() or root.is_symlink():
        if _is_shared_object(root):
            yield root
        return
    if not root.is_dir():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for filename in sorted(filenames):
            candidate = Path(directory) / filename
            if _is_shared_object(candidate):
                yield candidate


def scan_shared_objects(
    roots: Iterable[str | os.PathLike[str]],
    *,
    distribution_index: Mapping[str, str] | None = None,
) -> list[SharedObjectRecord]:
    """Inventory ELF architecture for ``.so`` files below *roots*."""

    ownership = distribution_index or {}
    seen: set[tuple[int, int]] = set()
    records: list[SharedObjectRecord] = []
    for root in sorted((Path(item).expanduser() for item in roots), key=str):
        for candidate in _iter_shared_objects(root):
            try:
                file_stat = candidate.lstat()
            except OSError:
                continue
            key = _inode_key(file_stat)
            if key in seen:
                continue
            seen.add(key)
            elf_class, machine, machine_id = parse_elf_identity(candidate)
            resolved = str(candidate.resolve(strict=False))
            records.append(
                SharedObjectRecord(
                    path=str(candidate),
                    distribution=ownership.get(resolved),
                    allocated_bytes=_allocated_bytes(file_stat),
                    apparent_bytes=int(file_stat.st_size),
                    elf_class=elf_class,
                    elf_machine=machine,
                    elf_machine_id=machine_id,
                    is_symlink=stat.S_ISLNK(file_stat.st_mode),
                ),
            )
    return sorted(records, key=lambda item: item.path)


def build_arm_cp312_pip_command(
    requirement: str,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Return a no-install pip resolver command for aarch64 CPython 3.12."""

    return (
        python_executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary",
        ":all:",
        "--platform",
        "manylinux_2_17_aarch64",
        "--platform",
        "manylinux2014_aarch64",
        "--implementation",
        "cp",
        "--python-version",
        "3.12",
        "--abi",
        "cp312",
        "--report",
        "-",
        requirement,
    )


def _redact_subprocess_output(text: str) -> str:
    redacted = text
    for pattern in _OUTPUT_REDACTIONS:
        if "https?" in pattern.pattern:
            redacted = pattern.sub(r"\1<redacted>@", redacted)
        elif "Bearer" in pattern.pattern:
            redacted = pattern.sub(r"\1<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted-api-key>", redacted)
    return redacted


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    return (match.group(1) if match else requirement).lower().replace("_", "-")


def _arm_resolution_blockers(
    requirement: str,
    stdout: str,
    stderr: str,
) -> tuple[tuple[dict[str, str], ...], bool]:
    text = "\n".join((stdout, stderr))
    names: list[str] = []
    for pattern in (
        r"Could not find a version that satisfies the requirement\s+([^\s(;]+)",
        r"No matching distribution found for\s+([^\s(;]+)",
    ):
        names.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    top_level = _requirement_name(requirement)
    blockers: list[dict[str, str]] = []
    seen: set[str] = set()
    limited = False
    for raw_name in names:
        name = _requirement_name(raw_name)
        if name in seen:
            continue
        seen.add(name)
        is_top = name == top_level
        limited = limited or is_top
        blockers.append(
            {
                "package": name,
                "kind": (
                    "top_level_requirement_unavailable"
                    if is_top
                    else "dependency_no_matching_aarch64_cp312_wheel"
                ),
                "reason": (
                    "resolver stopped at the top-level requirement; dependency "
                    "wheel compatibility remains untested"
                    if is_top
                    else "pip reported no matching binary distribution"
                ),
            },
        )
    if not blockers and text.strip():
        blockers.append(
            {
                "package": "unknown",
                "kind": "resolver_error_unparsed",
                "reason": "inspect the redacted pip output",
            },
        )
    return tuple(blockers), limited


def arm_cp312_pip_dry_run(
    requirement: str,
    *,
    execute: bool = False,
    python_executable: str = sys.executable,
    timeout_seconds: float = 300.0,
    env: Mapping[str, str] | None = None,
) -> ArmDryRunResult:
    """Plan or explicitly execute the aarch64/cp312 pip compatibility check."""

    command = build_arm_cp312_pip_command(
        requirement,
        python_executable=python_executable,
    )
    if not execute:
        return ArmDryRunResult(command=command, executed=False)

    child_env = dict(os.environ if env is None else env)
    child_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        blockers, limited = _arm_resolution_blockers(
            requirement,
            stdout,
            stderr,
        )
        if not blockers:
            blockers = (
                {
                    "package": _requirement_name(requirement),
                    "kind": "resolver_timeout",
                    "reason": (
                        "pip did not finish before the compatibility probe "
                        "timeout"
                    ),
                },
            )
        return ArmDryRunResult(
            command=command,
            executed=True,
            compatible=False,
            stdout=_redact_subprocess_output(stdout),
            stderr=_redact_subprocess_output(stderr),
            timed_out=True,
            blockers=blockers,
            resolution_limited_by_top_level=limited,
        )
    blockers, limited = _arm_resolution_blockers(
        requirement,
        completed.stdout,
        completed.stderr,
    ) if completed.returncode else ((), False)
    return ArmDryRunResult(
        command=command,
        executed=True,
        compatible=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=_redact_subprocess_output(completed.stdout),
        stderr=_redact_subprocess_output(completed.stderr),
        blockers=blockers,
        resolution_limited_by_top_level=limited,
    )


def _secret_fingerprint(value: bytes) -> str:
    return hashlib.sha256(b"qwenpaw-overhead-secret\0" + value).hexdigest()[:16]


def _iter_scan_files(paths: Iterable[str | os.PathLike[str]]) -> Iterator[Path]:
    for raw_path in paths:
        root = Path(raw_path).expanduser()
        if root.is_symlink():
            continue
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames if name not in {".git", "__pycache__"}
            )
            for filename in sorted(filenames):
                candidate = Path(directory) / filename
                if not candidate.is_symlink():
                    yield candidate


def _known_secret_values(
    supplied: Iterable[str | bytes],
    environment: Mapping[str, str] | None,
) -> list[bytes]:
    values: list[bytes] = []
    for value in supplied:
        encoded = value if isinstance(value, bytes) else value.encode()
        if len(encoded) >= 8:
            values.append(encoded)
    source_environment = os.environ if environment is None else environment
    if source_environment is not None:
        for name, value in source_environment.items():
            if _SECRET_ENV_NAME.search(name) and len(value) >= 8:
                values.append(value.encode())
    return sorted(set(values), key=lambda item: (len(item), item))


def _usable_secret_match(value: str) -> bool:
    normalized = value.strip().strip("\"'")
    if normalized.casefold() in _REDACTED_VALUES:
        return False
    if normalized.startswith(("${", "ENC:", "<")):
        return False
    return len(normalized) >= 8


def scan_for_secrets(
    paths: Iterable[str | os.PathLike[str]],
    *,
    known_secret_values: Iterable[str | bytes] = (),
    environment: Mapping[str, str] | None = None,
    max_file_bytes: int | None = 64 * 1024 * 1024,
) -> list[SecretFinding]:
    """Scan benchmark artifacts without ever returning matching plaintext."""

    needles = _known_secret_values(known_secret_values, environment)
    findings: list[SecretFinding] = []
    emitted: set[tuple[str, str, int | None, str]] = set()

    def emit(path: Path, rule: str, value: bytes, line: int | None) -> None:
        fingerprint = _secret_fingerprint(value)
        key = (str(path), rule, line, fingerprint)
        if key not in emitted:
            emitted.add(key)
            findings.append(
                SecretFinding(
                    path=str(path),
                    rule=rule,
                    fingerprint=fingerprint,
                    line_number=line,
                ),
            )

    for path in _iter_scan_files(paths):
        if path.name in _FORBIDDEN_SECRET_FILENAMES:
            emit(path, "forbidden_secret_file", str(path).encode(), None)
        try:
            file_stat = path.stat()
            if max_file_bytes is not None and file_stat.st_size > max_file_bytes:
                continue
            with path.open("rb") as file_obj:
                for line_number, raw_line in enumerate(file_obj, start=1):
                    for needle in needles:
                        if needle in raw_line:
                            emit(path, "known_secret_value", needle, line_number)
                    text_line = raw_line.decode("utf-8", errors="replace")
                    for rule, pattern in _SECRET_PATTERNS:
                        for match in pattern.finditer(text_line):
                            value = (
                                match.group(1) if match.lastindex else match.group(0)
                            )
                            if _usable_secret_match(value):
                                emit(path, rule, value.encode(), line_number)
        except OSError:
            continue
    return sorted(
        findings,
        key=lambda item: (item.path, item.line_number or -1, item.rule),
    )


class SecretLeakError(RuntimeError):
    """Raised when benchmark output fails the no-secret contract."""


def assert_no_secrets(findings: Sequence[SecretFinding]) -> None:
    """Fail closed using fingerprints and locations, never secret contents."""

    if findings:
        first = findings[0]
        raise SecretLeakError(
            f"secret scan found {len(findings)} finding(s); "
            f"first={first.rule} at {first.path}:{first.line_number or 0}",
        )


__all__ = [
    "SecretLeakError",
    "arm_cp312_pip_dry_run",
    "assert_no_secrets",
    "build_arm_cp312_pip_command",
    "classify_state_path",
    "default_footprint_targets",
    "distribution_file_index",
    "measure_targets",
    "measure_tree",
    "parse_elf_identity",
    "scan_distributions",
    "scan_for_secrets",
    "scan_shared_objects",
]
