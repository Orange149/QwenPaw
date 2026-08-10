"""Create disposable QwenPaw benchmark homes without touching user state.

The public entry point, :func:`create_isolated_run`, copies a prepared seed
working directory and optional secret directory into a freshly-created 0700
directory below ``/tmp`` (or an explicitly supplied temporary base).  The
returned :class:`IsolationPaths` is a context manager and removes only the
directory it created.

QwenPaw 2.0.1 resolves its mutable roots at import time.  Callers must pass
``paths.env`` to every benchmark subprocess; importing QwenPaw in the harness
process before applying that environment defeats the isolation guarantee.
``venv`` and ``bin`` at the seed root are deployment artifacts, not runtime
state, and are intentionally not copied.  The installed executable is chosen
separately by the benchmark runner.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


_ROOT_COPY_IGNORES = frozenset({"bin", "venv"})
_RUN_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_MARKER = ".qwenpaw-overhead-isolation"
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:api[_-]?key|token|secret|password|passwd|authorization|credential)",
    re.IGNORECASE,
)


class IsolationError(RuntimeError):
    """Raised when a seed cannot be copied without escaping isolation."""


@dataclass
class IsolationPaths:
    """Paths and subprocess environment for one disposable benchmark run."""

    root: Path
    working_dir: Path
    secret_dir: Path
    state_dir: Path
    project_dir: Path
    cache_dir: Path
    home_dir: Path
    temp_dir: Path
    backup_dir: Path
    seed_working_dir: Path
    seed_secret_dir: Path | None
    env: dict[str, str]
    _base_tmp: Path
    _closed: bool = False

    def __enter__(self) -> "IsolationPaths":
        if self._closed:
            raise RuntimeError("an IsolationPaths instance cannot be reused")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove this run's generated root after strict ownership checks."""

        if self._closed:
            return
        root = self.root
        marker = root / _MARKER
        try:
            resolved_root = root.resolve(strict=True)
            resolved_root.relative_to(self._base_tmp)
        except (FileNotFoundError, ValueError) as exc:
            self._closed = True
            if not root.exists():
                return
            raise IsolationError(
                f"refusing to clean unverified isolation root: {root}",
            ) from exc
        if resolved_root == self._base_tmp or not marker.is_file():
            raise IsolationError(
                f"refusing to clean unmarked isolation root: {resolved_root}",
            )
        shutil.rmtree(resolved_root)
        self._closed = True


def _validated_seed(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise IsolationError(f"{label} must not itself be a symlink: {path}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise IsolationError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise IsolationError(f"{label} is not a directory: {resolved}")
    return resolved


def _validate_seed_tree(seed: Path, *, ignored_root_names: set[str]) -> None:
    """Reject links so a copied runtime cannot write back through the seed."""

    for current, dirnames, filenames in os.walk(seed, followlinks=False):
        current_path = Path(current)
        if current_path == seed:
            dirnames[:] = [
                name for name in dirnames if name not in ignored_root_names
            ]
            filenames = [
                name for name in filenames if name not in ignored_root_names
            ]
        for name in [*dirnames, *filenames]:
            candidate = current_path / name
            if candidate.is_symlink():
                relative = candidate.relative_to(seed)
                raise IsolationError(
                    "seed contains a symlink; prepare a self-contained seed "
                    f"before benchmarking: {relative}",
                )


def _root_ignore(seed: Path):
    def ignore(current: str, names: list[str]) -> set[str]:
        if Path(current).resolve() != seed:
            return set()
        return set(names).intersection(_ROOT_COPY_IGNORES)

    return ignore


def _chmod_tree_private(root: Path, *, files_private: bool) -> None:
    """Make copied state private while retaining executable file bits."""

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        os.chmod(current_path, 0o700)
        for name in dirnames:
            os.chmod(current_path / name, 0o700)
        if not files_private:
            continue
        for name in filenames:
            path = current_path / name
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            os.chmod(path, 0o700 if mode & 0o111 else 0o600)


def _copy_seed(seed: Path, destination: Path, *, ignore_install: bool) -> None:
    ignored = set(_ROOT_COPY_IGNORES) if ignore_install else set()
    _validate_seed_tree(seed, ignored_root_names=ignored)
    shutil.copytree(
        seed,
        destination,
        copy_function=shutil.copy2,
        ignore=_root_ignore(seed) if ignore_install else None,
    )


def _private_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def create_isolated_run(
    seed_working_dir: Path,
    seed_secret_dir: Path | None,
    run_id: str,
    base_tmp: Path = Path("/tmp"),
) -> IsolationPaths:
    """Copy seed state into a unique, private, disposable run directory.

    The returned ``env`` is a complete copy of the current environment with
    QwenPaw, legacy CoPaw, XDG, npm, backup, and temporary roots redirected
    below the generated directory.  ``HOME`` is redirected too, closing
    fallback paths in QwenPaw and third-party libraries.  No source path is
    ever removed or opened for writing.
    """

    seed_working = _validated_seed(
        Path(seed_working_dir),
        label="seed working directory",
    )
    seed_secret = None
    if seed_secret_dir is not None:
        seed_secret = _validated_seed(
            Path(seed_secret_dir),
            label="seed secret directory",
        )
    if seed_secret is not None and seed_secret == seed_working:
        raise IsolationError("working and secret seeds must be distinct")

    base = _validated_seed(Path(base_tmp), label="temporary base")
    for seed, label in (
        (seed_working, "working seed"),
        (seed_secret, "secret seed"),
    ):
        if seed is None:
            continue
        try:
            base.relative_to(seed)
        except ValueError:
            pass
        else:
            raise IsolationError(
                f"temporary base must not be inside the {label}: {base}",
            )
    safe_id = _RUN_ID_SAFE.sub("-", str(run_id).strip()).strip(".-")
    if not safe_id:
        raise IsolationError("run_id must contain at least one safe character")
    safe_id = safe_id[:64]

    root = Path(
        tempfile.mkdtemp(
            prefix=f"qwenpaw-overhead.{safe_id}.",
            dir=str(base),
        ),
    ).resolve()
    os.chmod(root, 0o700)
    try:
        (root / _MARKER).write_text("owned\n", encoding="utf-8")
        os.chmod(root / _MARKER, 0o600)

        working_dir = root / "working"
        secret_dir = root / "secret"
        _copy_seed(seed_working, working_dir, ignore_install=True)
        if seed_secret is None:
            _private_mkdir(secret_dir)
        else:
            _copy_seed(seed_secret, secret_dir, ignore_install=False)

        state_dir = root / "state"
        project_dir = root / "project"
        cache_dir = root / "cache"
        home_dir = root / "home"
        temp_dir = root / "tmp"
        backup_dir = root / "backups"
        for path in (
            state_dir,
            project_dir,
            cache_dir,
            home_dir,
            temp_dir,
            backup_dir,
            root / "config",
            root / "data",
        ):
            _private_mkdir(path)
        _chmod_tree_private(working_dir, files_private=False)
        _chmod_tree_private(secret_dir, files_private=True)

        env = {
            name: value
            for name, value in os.environ.items()
            if not _SENSITIVE_ENV_NAME.search(name)
        }
        overrides = {
            "HOME": str(home_dir),
            "QWENPAW_WORKING_DIR": str(working_dir),
            "COPAW_WORKING_DIR": str(working_dir),
            "QWENPAW_SECRET_DIR": str(secret_dir),
            "COPAW_SECRET_DIR": str(secret_dir),
            "QWENPAW_BACKUP_DIR": str(backup_dir),
            "COPAW_BACKUP_DIR": str(backup_dir),
            "PAW_STATE_DIR": str(state_dir),
            "XDG_STATE_HOME": str(state_dir / "xdg"),
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "NPM_CONFIG_CACHE": str(cache_dir / "npm"),
            "TMPDIR": str(temp_dir),
            # A copied .master_key is the only permitted key source.  Avoid
            # reading or updating the host desktop keyring from a benchmark.
            "QWENPAW_DISABLE_KEYRING": "1",
        }
        for proxy_name in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        ):
            env.pop(proxy_name, None)
        env.update(overrides)
        return IsolationPaths(
            root=root,
            working_dir=working_dir,
            secret_dir=secret_dir,
            state_dir=state_dir,
            project_dir=project_dir,
            cache_dir=cache_dir,
            home_dir=home_dir,
            temp_dir=temp_dir,
            backup_dir=backup_dir,
            seed_working_dir=seed_working,
            seed_secret_dir=seed_secret,
            env=env,
            _base_tmp=base,
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


__all__ = [
    "IsolationError",
    "IsolationPaths",
    "create_isolated_run",
]
