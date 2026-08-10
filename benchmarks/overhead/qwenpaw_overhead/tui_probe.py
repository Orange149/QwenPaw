"""One-shot PTY probe for the separately reported stock Textual TUI."""

from __future__ import annotations

import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import termios
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


_ANSI = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(frozen=True)
class TuiProbeResult:
    success: bool
    pid: int
    acp_pid: int | None
    start_ns: int
    acp_seen_ns: int | None
    ui_ready_ns: int | None
    exit_ns: int
    exit_code: int | None
    output_bytes: int
    graceful_exit: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["startup_to_acp_ms"] = (
            (self.acp_seen_ns - self.start_ns) / 1_000_000
            if self.acp_seen_ns is not None
            else None
        )
        value["startup_to_ui_ready_ms"] = (
            (self.ui_ready_ns - self.start_ns) / 1_000_000
            if self.ui_ready_ns is not None
            else None
        )
        value["exit_ms"] = (
            (self.exit_ns - (self.ui_ready_ns or self.start_ns)) / 1_000_000
        )
        return value


def _find_acp_child(pid: int) -> int | None:
    try:
        import psutil

        root = psutil.Process(pid)
        for child in root.children(recursive=True):
            command = " ".join(child.cmdline())
            if "qwenpaw acp" in command or (
                "-m qwenpaw acp" in command
            ):
                return child.pid
    except (ImportError, OSError):
        return None
    except Exception:
        return None
    return None


def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (OSError, ProcessLookupError):
        pass


def run_tui_probe(
    *,
    python_executable: Path,
    env: dict[str, str],
    project_dir: Path,
    agent: str = "default",
    affinity: str = "0-3",
    startup_timeout_s: float = 120.0,
    idle_s: float = 90.0,
    on_process_started: Callable[[int], None] | None = None,
) -> TuiProbeResult:
    """Launch the real TUI in a PTY and discard all terminal content.

    ``ui_ready`` is the rendered status-bar transition after the stock backend
    warmup, not ACP ``AvailableCommands``.  The headless ACP benchmark remains
    authoritative for Runtime Ready; this probe exists only to quantify the
    parent Textual process, its timers, and the default warmup.
    """

    if not python_executable.is_file():
        raise FileNotFoundError(python_executable)
    if not project_dir.is_dir():
        raise FileNotFoundError(project_dir)
    command = [str(python_executable), "-m", "qwenpaw", "tui", "--agent", agent]
    taskset = shutil.which("taskset")
    if affinity:
        if taskset is None:
            raise RuntimeError("taskset is required for an affinity-controlled probe")
        command = [taskset, "-c", affinity, *command]
    command.append(str(project_dir))

    master, slave = pty.openpty()
    try:
        # A stable terminal size avoids resize-driven Textual work.
        import fcntl

        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
        start_ns = time.monotonic_ns()
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(slave)
    if on_process_started is not None:
        on_process_started(process.pid)

    acp_pid = None
    acp_seen_ns = None
    ready_ns = None
    output_bytes = 0
    rolling = b""
    deadline = time.monotonic() + startup_timeout_s
    error = None
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if acp_pid is None:
                acp_pid = _find_acp_child(process.pid)
                if acp_pid is not None:
                    acp_seen_ns = time.monotonic_ns()
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master, 65_536)
                except OSError:
                    chunk = b""
                output_bytes += len(chunk)
                rolling = (rolling + chunk)[-131_072:]
                plain = _ANSI.sub(b"", rolling).lower()
                if b" ready" in plain or b"ready " in plain:
                    ready_ns = time.monotonic_ns()
                    break
        if ready_ns is None:
            error = (
                f"TUI exited before ready (code={process.returncode})"
                if process.poll() is not None
                else "TUI readiness timeout"
            )
        else:
            end_idle = time.monotonic() + max(0.0, idle_s)
            while process.poll() is None and time.monotonic() < end_idle:
                readable, _, _ = select.select([master], [], [], 0.1)
                if readable:
                    try:
                        output_bytes += len(os.read(master, 65_536))
                    except OSError:
                        break

        if process.poll() is None:
            os.write(master, b"\x11")  # Ctrl+Q: documented TUI graceful exit.
        try:
            process.wait(timeout=5.0)
            graceful = True
        except subprocess.TimeoutExpired:
            graceful = False
            _signal_group(process, signal.SIGINT)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    _signal_group(process, signal.SIGKILL)
                    process.wait(timeout=2.0)
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if process.poll() is None:
            _signal_group(process, signal.SIGKILL)
            process.wait(timeout=2.0)

    exit_ns = time.monotonic_ns()
    success = ready_ns is not None and process.returncode == 0 and graceful
    return TuiProbeResult(
        success=success,
        pid=process.pid,
        acp_pid=acp_pid,
        start_ns=start_ns,
        acp_seen_ns=acp_seen_ns,
        ui_ready_ns=ready_ns,
        exit_ns=exit_ns,
        exit_code=process.returncode,
        output_bytes=output_bytes,
        graceful_exit=graceful,
        error=error,
    )

