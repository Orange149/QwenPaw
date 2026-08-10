"""Low-overhead Linux process-tree and mutable-state measurements.

``ResourceSampler`` follows the launched PID and every descendant it observes,
using psutil for process metadata/counters and ``/proc/<pid>/smaps_rollup`` for
PSS, USS, and SwapPss.  Samples are streamed to CSV so a long idle run does not
grow harness memory.  PIDs are always paired with process creation time to
avoid attributing a reused PID to QwenPaw.

The sampler is intentionally independent of QwenPaw.  It cannot see a process
whose entire lifetime falls between two samples, and it cannot automatically
attribute a pre-existing external model daemon.  Use cgroup counters as the
whole-tree cross-check when available and report external model processes as a
separate boundary.
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import psutil


_CSV_FIELDS = (
    "monotonic_ns",
    "wall_time_ns",
    "pid",
    "ppid",
    "create_time",
    "start_ticks",
    "role",
    "name",
    "command",
    "cmdline_sha256",
    "status",
    "cpu_user_s",
    "cpu_system_s",
    "cpu_percent_single_core",
    "rss_bytes",
    "pss_bytes",
    "uss_bytes",
    "swap_pss_bytes",
    "threads",
    "num_fds",
    "ctx_voluntary",
    "ctx_involuntary",
    "read_bytes",
    "write_bytes",
    "socket_scan",
    "inet_connection_count",
    "remote_endpoints",
)

_SECRET_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+|"
    r"(?:api[_-]?key|token|secret|password)=\S+)",
)
_SECRET_FLAGS = frozenset(
    {
        "--api-key",
        "--apikey",
        "--authorization",
        "--password",
        "--secret",
        "--token",
    },
)
_SECRET_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    """A PID plus its creation timestamp, safe against PID reuse."""

    pid: int
    create_time: float
    start_ticks: int | None = None


def _process_start_ticks(pid: int) -> int | None:
    """Read Linux's immutable process-start tick for PID-reuse checks."""

    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8",
            errors="replace",
        )
        # comm is parenthesized and may contain spaces or ``)`` characters;
        # field 3 starts after the final closing parenthesis. starttime is
        # field 22, hence offset 19 in this suffix.
        suffix = text[text.rfind(")") + 1 :].split()
        return int(suffix[19])
    except (OSError, IndexError, ValueError):
        return None


def _identity(process: psutil.Process) -> ProcessIdentity:
    return ProcessIdentity(
        process.pid,
        process.create_time(),
        _process_start_ticks(process.pid),
    )


def _same_process(identity: ProcessIdentity) -> psutil.Process | None:
    try:
        process = psutil.Process(identity.pid)
        current_ticks = _process_start_ticks(identity.pid)
        if identity.start_ticks is not None and current_ticks is not None:
            if current_ticks != identity.start_ticks:
                return None
        elif abs(process.create_time() - identity.create_time) > 0.1:
            # WSL can expose small boot-time rounding jitter through psutil.
            # The kernel start tick is preferred; this is only a fallback.
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return None


def _command_tokens(cmdline: Sequence[str], executable: str = "") -> list[str]:
    tokens = [str(token) for token in cmdline if str(token)]
    if not tokens and executable:
        tokens = [executable]
    return tokens


def classify_process_role(
    cmdline: Sequence[str],
    *,
    executable: str = "",
    is_root: bool = False,
) -> str:
    """Classify one process without persisting its potentially secret args."""

    tokens = _command_tokens(cmdline, executable)
    lowered = [token.lower() for token in tokens]
    joined = " ".join(lowered)
    basename = Path(executable or (tokens[0] if tokens else "")).name.lower()

    if "-m qwenpaw acp" in joined or re.search(r"\bqwenpaw\s+acp\b", joined):
        return "acp"
    if any(
        marker in joined
        for marker in (
            "playwright",
            "ms-playwright",
            "chromium",
            "chrome --",
            "google-chrome",
        )
    ) or basename in {"chromium", "chromium-browser", "chrome", "google-chrome"}:
        return "browser"
    if any(
        marker in joined
        for marker in (
            "modelcontextprotocol",
            "tavily-mcp",
            "mcp-server",
            "mcp_server",
        )
    ):
        return "mcp"
    if basename in {"ollama", "llama-server", "vllm", "qwenpaw-local"}:
        return "model"
    if is_root:
        if re.search(r"\bqwenpaw\s+app\b", joined):
            return "app"
        if re.search(r"\bqwenpaw\s+task\b", joined):
            return "task"
        if "qwenpaw" in joined:
            return "tui"
        return "root"
    if basename in {
        "bash",
        "cmd.exe",
        "dash",
        "fish",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sh",
        "sleep",
        "zsh",
    }:
        return "tool"
    if basename in {"node", "nodejs", "npx", "npm"} and "mcp" in joined:
        return "mcp"
    return "other"


def _safe_command(cmdline: Sequence[str], executable: str) -> str:
    tokens = _command_tokens(cmdline, executable)
    safe: list[str] = []
    hide_next = False
    for token in tokens[:12]:
        if hide_next:
            safe.append("<redacted>")
            hide_next = False
            continue
        lowered = token.lower()
        if lowered in _SECRET_FLAGS or (
            "=" not in lowered
            and any(fragment in lowered for fragment in _SECRET_FRAGMENTS)
        ):
            safe.append(token)
            hide_next = True
            continue
        scrubbed = _SECRET_VALUE.sub("<redacted>", token)
        if "://" in scrubbed:
            parsed = urlsplit(scrubbed)
            if parsed.hostname:
                try:
                    port = parsed.port
                except ValueError:
                    scrubbed = "<redacted-url>"
                else:
                    host = (
                        f"[{parsed.hostname}]"
                        if ":" in parsed.hostname
                        else parsed.hostname
                    )
                    netloc = f"{host}:{port}" if port else host
                    scrubbed = urlunsplit(
                        (parsed.scheme, netloc, parsed.path, "", ""),
                    )
        safe.append(scrubbed)
    if len(tokens) > 12:
        safe.append("…")
    return " ".join(safe)


def _smaps_rollup(pid: int) -> tuple[int | None, int | None, int | None]:
    path = Path("/proc") / str(pid) / "smaps_rollup"
    values: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, separator, raw = line.partition(":")
                if not separator:
                    continue
                fields = raw.split()
                if fields and fields[0].isdigit():
                    values[key] = int(fields[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None, None, None
    pss = values.get("Pss")
    private_keys = ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
    uss = sum(values.get(key, 0) for key in private_keys)
    return pss, uss, values.get("SwapPss")


def _optional(callable_value, default: Any = None) -> Any:
    try:
        return callable_value()
    except (
        AttributeError,
        OSError,
        ProcessLookupError,
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
    ):
        return default


def _cgroup2_mount() -> Path | None:
    """Resolve the cgroup2 mount without assuming ``/sys/fs/cgroup``."""

    try:
        lines = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator or not after.startswith("cgroup2 "):
            continue
        fields = before.split()
        if len(fields) >= 5:
            return Path(fields[4])
    return None


def _process_cgroup_path(pid: int) -> Path | None:
    mount = _cgroup2_mount()
    if mount is None:
        return None
    try:
        lines = (Path("/proc") / str(pid) / "cgroup").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        if hierarchy == "0" and not controllers:
            return mount / relative.lstrip("/")
    return None


def _key_value_file(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if len(fields) >= 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return result


def _integer_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
        return int(value)
    except (OSError, ValueError):
        return None


def _io_stat(path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return totals
    for line in lines:
        for field in line.split()[1:]:
            key, separator, raw = field.partition("=")
            if not separator:
                continue
            try:
                totals[key] = totals.get(key, 0) + int(raw)
            except ValueError:
                continue
    return totals


def _snapshot_cgroup(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_dir():
        return {"available": False, "path": str(path) if path else None}
    try:
        members = [
            int(value)
            for value in (path / "cgroup.procs").read_text(
                encoding="utf-8",
                errors="replace",
            ).split()
        ]
    except (OSError, ValueError):
        members = []
    return {
        "available": True,
        "path": str(path),
        "member_count": len(members),
        "cpu_stat": _key_value_file(path / "cpu.stat"),
        "memory_current_bytes": _integer_file(path / "memory.current"),
        "memory_peak_bytes": _integer_file(path / "memory.peak"),
        "memory_events": _key_value_file(path / "memory.events"),
        "io_stat": _io_stat(path / "io.stat"),
        "pids_current": _integer_file(path / "pids.current"),
    }


def _numeric_delta(before: Any, after: Any) -> Any:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        keys = set(before).union(after)
        return {
            key: _numeric_delta(before.get(key), after.get(key))
            for key in sorted(keys)
            if key not in {"available", "path"}
        }
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _format_endpoint(address: Any) -> tuple[str, str, int] | None:
    if not address:
        return None
    try:
        ip = str(address.ip)
        port = int(address.port)
    except AttributeError:
        try:
            ip = str(address[0])
            port = int(address[1])
        except (IndexError, TypeError, ValueError):
            return None
    display = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
    return display, ip, port


class ResourceSampler:
    """Stream psutil/proc samples for one process tree to a CSV file."""

    def __init__(
        self,
        root_pid: int,
        interval_s: float,
        output_path: Path,
    ) -> None:
        if int(root_pid) <= 0:
            raise ValueError("root_pid must be positive")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.root_pid = int(root_pid)
        self.interval_s = float(interval_s)
        self.output_path = Path(output_path)
        self._root_identity: ProcessIdentity | None = None
        self._observed: dict[ProcessIdentity, dict[str, Any]] = {}
        self._first_counters: dict[ProcessIdentity, dict[str, float]] = {}
        self._last_counters: dict[ProcessIdentity, dict[str, float]] = {}
        self._previous_cpu: dict[ProcessIdentity, tuple[int, float]] = {}
        self._roles: dict[str, dict[str, Any]] = {}
        self._tree_peak = {
            "processes": 0,
            "rss_bytes": 0,
            "pss_bytes": None,
            "uss_bytes": None,
            "swap_pss_bytes": None,
            "threads": 0,
            "num_fds": 0,
        }
        self._coverage_min: float | None = None
        self._nonempty_sample_count = 0
        self._empty_sample_count = 0
        self._tree_series: dict[str, list[float]] = {
            "rss_bytes": [],
            "pss_bytes": [],
            "uss_bytes": [],
            "cpu_percent_single_core": [],
        }
        self._remote_endpoints: dict[str, dict[str, Any]] = {}
        self._last_socket_scan_ns = 0
        self._socket_scan_count = 0
        self._sample_count = 0
        self._errors: list[str] = []
        self._started_monotonic_ns: int | None = None
        self._ended_monotonic_ns: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._aggregate: dict[str, Any] | None = None
        self._sample_lock = threading.Lock()
        self._cgroup_path: Path | None = None
        self._cgroup_start: dict[str, Any] | None = None

    def start(self) -> "ResourceSampler":
        """Capture the root identity, create the CSV, and start sampling."""

        if self._thread is not None or self._started_monotonic_ns is not None:
            raise RuntimeError("ResourceSampler can only be started once")
        try:
            root = psutil.Process(self.root_pid)
            self._root_identity = _identity(root)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            raise ProcessLookupError(
                f"root process is unavailable: {self.root_pid}",
            ) from exc

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        self._started_monotonic_ns = time.monotonic_ns()
        self._cgroup_path = _process_cgroup_path(self.root_pid)
        self._cgroup_start = _snapshot_cgroup(self._cgroup_path)
        try:
            with self._sample_lock:
                self._sample_once(force_sockets=True)
            self._thread = threading.Thread(
                target=self._run,
                name=f"qwenpaw-overhead-sampler-{self.root_pid}",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._handle.close()
            self._handle = None
            raise
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                with self._sample_lock:
                    # Network policy is part of the benchmark's acceptance
                    # evidence.  Scan on every scheduled sample so a short
                    # model connection is not deliberately hidden behind the
                    # old one-second socket cadence.  This is still sampled
                    # observation, not packet-complete enforcement.
                    self._sample_once(force_sockets=True)
            except Exception as exc:  # keep evidence from partial runs
                self._record_error(f"sample failed: {type(exc).__name__}: {exc}")

    def _record_error(self, message: str) -> None:
        if len(self._errors) < 100:
            self._errors.append(message)

    def _discover(self) -> dict[ProcessIdentity, psutil.Process]:
        seeds: dict[ProcessIdentity, psutil.Process] = {}
        if self._root_identity is not None:
            root = _same_process(self._root_identity)
            if root is not None:
                seeds[self._root_identity] = root
        for identity in list(self._observed):
            process = _same_process(identity)
            if process is not None:
                seeds[identity] = process

        discovered = dict(seeds)
        queue = list(seeds.values())
        while queue:
            parent = queue.pop()
            try:
                children = parent.children(recursive=False)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for child in children:
                try:
                    identity = _identity(child)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if identity in discovered:
                    continue
                discovered[identity] = child
                queue.append(child)
        return discovered

    def _process_row(
        self,
        identity: ProcessIdentity,
        process: psutil.Process,
        monotonic_ns: int,
        wall_time_ns: int,
        scan_sockets: bool,
    ) -> dict[str, Any] | None:
        try:
            with process.oneshot():
                ppid = process.ppid()
                name = process.name()
                executable = _optional(process.exe, "") or ""
                cmdline = _optional(process.cmdline, []) or []
                status = _optional(process.status, "unknown") or "unknown"
                cpu = process.cpu_times()
                memory = process.memory_info()
                threads = _optional(process.num_threads, None)
                fds = _optional(process.num_fds, None)
                context = _optional(process.num_ctx_switches, None)
                io = _optional(process.io_counters, None)
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
            ProcessLookupError,
        ):
            return None

        is_root = identity == self._root_identity
        role = classify_process_role(
            cmdline,
            executable=executable,
            is_root=is_root,
        )
        endpoint_names: list[str] = []
        connection_count: int | None = None
        if scan_sockets:
            connections = _optional(
                lambda: process.net_connections(kind="inet"),
                None,
            )
            if connections is not None:
                connection_count = len(connections)
                for connection in connections:
                    endpoint = _format_endpoint(connection.raddr)
                    if endpoint is None:
                        continue
                    display, ip, port = endpoint
                    endpoint_names.append(display)
                    try:
                        loopback = ipaddress.ip_address(ip).is_loopback
                    except ValueError:
                        loopback = False
                    item = self._remote_endpoints.setdefault(
                        display,
                        {
                            "endpoint": display,
                            "ip": ip,
                            "port": port,
                            "loopback": loopback,
                            "first_seen_monotonic_ns": monotonic_ns,
                            "last_seen_monotonic_ns": monotonic_ns,
                            "roles": set(),
                            "statuses": set(),
                        },
                    )
                    item["last_seen_monotonic_ns"] = monotonic_ns
                    item["roles"].add(role)
                    item["statuses"].add(str(connection.status))
        pss, uss, swap_pss = _smaps_rollup(identity.pid)
        total_cpu = float(cpu.user) + float(cpu.system)
        previous = self._previous_cpu.get(identity)
        cpu_percent = None
        if previous is not None and monotonic_ns > previous[0]:
            elapsed = (monotonic_ns - previous[0]) / 1_000_000_000
            cpu_percent = max(0.0, (total_cpu - previous[1]) / elapsed * 100.0)
        self._previous_cpu[identity] = (monotonic_ns, total_cpu)

        raw_cmdline = "\0".join(str(token) for token in cmdline)
        command_hash = hashlib.sha256(raw_cmdline.encode("utf-8")).hexdigest()
        row = {
            "monotonic_ns": monotonic_ns,
            "wall_time_ns": wall_time_ns,
            "pid": identity.pid,
            "ppid": ppid,
            "create_time": identity.create_time,
            "start_ticks": identity.start_ticks,
            "role": role,
            "name": name,
            "command": _safe_command(cmdline, executable),
            "cmdline_sha256": command_hash,
            "status": status,
            "cpu_user_s": float(cpu.user),
            "cpu_system_s": float(cpu.system),
            "cpu_percent_single_core": cpu_percent,
            "rss_bytes": int(memory.rss),
            "pss_bytes": pss,
            "uss_bytes": uss,
            "swap_pss_bytes": swap_pss,
            "threads": threads,
            "num_fds": fds,
            "ctx_voluntary": getattr(context, "voluntary", None),
            "ctx_involuntary": getattr(context, "involuntary", None),
            "read_bytes": getattr(io, "read_bytes", None),
            "write_bytes": getattr(io, "write_bytes", None),
            "socket_scan": scan_sockets,
            "inet_connection_count": connection_count,
            "remote_endpoints": ";".join(sorted(set(endpoint_names))),
        }
        counters = {
            "cpu_user_s": float(cpu.user),
            "cpu_system_s": float(cpu.system),
            "read_bytes": float(getattr(io, "read_bytes", 0) or 0),
            "write_bytes": float(getattr(io, "write_bytes", 0) or 0),
            "ctx_voluntary": float(getattr(context, "voluntary", 0) or 0),
            "ctx_involuntary": float(getattr(context, "involuntary", 0) or 0),
        }
        self._first_counters.setdefault(identity, counters)
        self._last_counters[identity] = counters
        observed = self._observed.setdefault(
            identity,
            {
                "pid": identity.pid,
                "create_time": identity.create_time,
                "start_ticks": identity.start_ticks,
                "first_seen_monotonic_ns": monotonic_ns,
                "last_seen_monotonic_ns": monotonic_ns,
                "role": role,
                "name": name,
                "command": row["command"],
                "cmdline_sha256": command_hash,
            },
        )
        observed.update(
            {
                "last_seen_monotonic_ns": monotonic_ns,
                "role": role,
                "name": name,
                "command": row["command"],
                "cmdline_sha256": command_hash,
            },
        )
        return row

    @staticmethod
    def _complete_sum(rows: list[dict[str, Any]], key: str) -> int | None:
        values = [row[key] for row in rows]
        if not values or any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    @staticmethod
    def _update_optional_peak(
        target: dict[str, Any],
        key: str,
        value: int | None,
    ) -> None:
        if value is None:
            return
        current = target.get(key)
        target[key] = value if current is None else max(int(current), value)

    def _update_peaks(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            # Empty samples after normal process exit say nothing about smaps
            # availability and must not turn otherwise complete coverage into
            # zero.
            self._empty_sample_count += 1
            return
        self._nonempty_sample_count += 1
        self._tree_peak["processes"] = max(
            self._tree_peak["processes"],
            len(rows),
        )
        self._tree_peak["rss_bytes"] = max(
            self._tree_peak["rss_bytes"],
            sum(int(row["rss_bytes"]) for row in rows),
        )
        self._tree_series["rss_bytes"].append(
            float(sum(int(row["rss_bytes"]) for row in rows)),
        )
        self._tree_peak["threads"] = max(
            self._tree_peak["threads"],
            sum(int(row["threads"] or 0) for row in rows),
        )
        self._tree_peak["num_fds"] = max(
            self._tree_peak["num_fds"],
            sum(int(row["num_fds"] or 0) for row in rows),
        )
        for key in ("pss_bytes", "uss_bytes", "swap_pss_bytes"):
            value = self._complete_sum(rows, key)
            self._update_optional_peak(
                self._tree_peak,
                key,
                value,
            )
            if key in self._tree_series and value is not None:
                self._tree_series[key].append(float(value))
        cpu_values = [
            float(row["cpu_percent_single_core"])
            for row in rows
            if row["cpu_percent_single_core"] is not None
        ]
        if cpu_values:
            self._tree_series["cpu_percent_single_core"].append(sum(cpu_values))
        covered = sum(row["pss_bytes"] is not None for row in rows)
        coverage = covered / len(rows)
        self._coverage_min = (
            coverage
            if self._coverage_min is None
            else min(self._coverage_min, coverage)
        )

        by_role: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_role.setdefault(str(row["role"]), []).append(row)
        for role, role_rows in by_role.items():
            stats = self._roles.setdefault(
                role,
                {
                    "peak_processes": 0,
                    "peak_rss_bytes": 0,
                    "peak_pss_bytes": None,
                    "peak_uss_bytes": None,
                    "peak_swap_pss_bytes": None,
                    "peak_threads": 0,
                    "peak_num_fds": 0,
                },
            )
            stats["peak_processes"] = max(stats["peak_processes"], len(role_rows))
            stats["peak_rss_bytes"] = max(
                stats["peak_rss_bytes"],
                sum(int(row["rss_bytes"]) for row in role_rows),
            )
            stats["peak_threads"] = max(
                stats["peak_threads"],
                sum(int(row["threads"] or 0) for row in role_rows),
            )
            stats["peak_num_fds"] = max(
                stats["peak_num_fds"],
                sum(int(row["num_fds"] or 0) for row in role_rows),
            )
            for source_key, target_key in (
                ("pss_bytes", "peak_pss_bytes"),
                ("uss_bytes", "peak_uss_bytes"),
                ("swap_pss_bytes", "peak_swap_pss_bytes"),
            ):
                self._update_optional_peak(
                    stats,
                    target_key,
                    self._complete_sum(role_rows, source_key),
                )

    def _sample_once(self, *, force_sockets: bool) -> dict[str, Any]:
        monotonic_ns = time.monotonic_ns()
        wall_time_ns = time.time_ns()
        scan_sockets = force_sockets or (
            monotonic_ns - self._last_socket_scan_ns >= 1_000_000_000
        )
        if scan_sockets:
            self._last_socket_scan_ns = monotonic_ns
            self._socket_scan_count += 1
        rows: list[dict[str, Any]] = []
        for identity, process in self._discover().items():
            row = self._process_row(
                identity,
                process,
                monotonic_ns,
                wall_time_ns,
                scan_sockets,
            )
            if row is not None:
                rows.append(row)
        rows.sort(key=lambda item: (int(item["pid"]), float(item["create_time"])))
        writer = self._writer
        if writer is None:
            raise RuntimeError("sampler CSV is not open")
        for row in rows:
            writer.writerow(row)
        if self._handle is not None:
            self._handle.flush()
        self._sample_count += 1
        self._update_peaks(rows)
        cpu_values = [
            float(row["cpu_percent_single_core"])
            for row in rows
            if row["cpu_percent_single_core"] is not None
        ]
        return {
            "monotonic_ns": monotonic_ns,
            "wall_time_ns": wall_time_ns,
            "processes": len(rows),
            "pids": [int(row["pid"]) for row in rows],
            "rss_bytes": sum(int(row["rss_bytes"]) for row in rows),
            "pss_bytes": self._complete_sum(rows, "pss_bytes"),
            "uss_bytes": self._complete_sum(rows, "uss_bytes"),
            "swap_pss_bytes": self._complete_sum(rows, "swap_pss_bytes"),
            "threads": sum(int(row["threads"] or 0) for row in rows),
            "num_fds": sum(int(row["num_fds"] or 0) for row in rows),
            "cpu_percent_single_core": sum(cpu_values) if cpu_values else None,
            "socket_scan": scan_sockets,
        }

    def sample_now(self) -> dict[str, Any]:
        """Synchronously append one sample and return its tree aggregate."""

        if self._started_monotonic_ns is None or self._handle is None:
            raise RuntimeError("ResourceSampler is not running")
        with self._sample_lock:
            return self._sample_once(force_sockets=True)

    def _counter_totals(self) -> dict[str, float | int]:
        totals: dict[str, float] = {
            "cpu_user_s": 0.0,
            "cpu_system_s": 0.0,
            "read_bytes": 0.0,
            "write_bytes": 0.0,
            "ctx_voluntary": 0.0,
            "ctx_involuntary": 0.0,
        }
        for identity, first in self._first_counters.items():
            last = self._last_counters.get(identity, first)
            for key in totals:
                totals[key] += max(0.0, last[key] - first[key])
        return {
            "cpu_user_s": totals["cpu_user_s"],
            "cpu_system_s": totals["cpu_system_s"],
            "read_bytes": int(totals["read_bytes"]),
            "write_bytes": int(totals["write_bytes"]),
            "ctx_voluntary": int(totals["ctx_voluntary"]),
            "ctx_involuntary": int(totals["ctx_involuntary"]),
        }

    def _role_totals(self) -> dict[str, dict[str, Any]]:
        result = {role: dict(stats) for role, stats in self._roles.items()}
        identities_by_role: dict[str, list[ProcessIdentity]] = {}
        for identity, item in self._observed.items():
            identities_by_role.setdefault(str(item["role"]), []).append(identity)
        for role, identities in identities_by_role.items():
            stats = result.setdefault(role, {})
            stats["observed_process_count"] = len(identities)
            for key in ("cpu_user_s", "cpu_system_s", "read_bytes", "write_bytes"):
                delta = 0.0
                for identity in identities:
                    first = self._first_counters[identity][key]
                    last = self._last_counters.get(
                        identity,
                        self._first_counters[identity],
                    )[key]
                    delta += max(0.0, last - first)
                stats[f"observed_{key}"] = (
                    delta if key.startswith("cpu_") else int(delta)
                )
        return dict(sorted(result.items()))

    @staticmethod
    def _series_summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "p95": None, "max": None}
        ordered = sorted(values)
        rank = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999) - 1))
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "p95": ordered[rank],
            "max": ordered[-1],
        }

    def _serialized_endpoints(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for endpoint in sorted(self._remote_endpoints):
            item = dict(self._remote_endpoints[endpoint])
            item["roles"] = sorted(item["roles"])
            item["statuses"] = sorted(item["statuses"])
            result.append(item)
        return result

    def stop(self) -> dict[str, Any]:
        """Stop sampling, close the CSV, and return a JSON-safe aggregate."""

        if self._aggregate is not None:
            return self._aggregate
        if self._started_monotonic_ns is None:
            raise RuntimeError("ResourceSampler has not been started")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
            if self._thread.is_alive():
                raise RuntimeError("ResourceSampler thread did not stop")
        try:
            with self._sample_lock:
                self._sample_once(force_sockets=True)
        except Exception as exc:
            self._record_error(
                f"final sample failed: {type(exc).__name__}: {exc}",
            )
        self._ended_monotonic_ns = time.monotonic_ns()
        cgroup_end = _snapshot_cgroup(self._cgroup_path)
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

        observed = [self._observed[key] for key in sorted(self._observed)]
        endpoints = self._serialized_endpoints()
        limitations = [
            "sub-interval processes may be missed",
            "socket destinations are sampled and are not packet-complete enforcement",
            "counter totals are deltas between first and last observation",
            "external model daemons are outside the descendant tree",
        ]
        if not (self._cgroup_start or {}).get("available"):
            limitations.append("cgroup v2 counters unavailable for root process")
        elif int((self._cgroup_start or {}).get("member_count", 0)) > 1:
            limitations.append(
                "existing cgroup is shared; counters are a cross-check, not "
                "exclusive QwenPaw attribution",
            )
        self._aggregate = {
            "root_pid": self.root_pid,
            "root_create_time": (
                self._root_identity.create_time if self._root_identity else None
            ),
            "root_start_ticks": (
                self._root_identity.start_ticks if self._root_identity else None
            ),
            "interval_s": self.interval_s,
            "sample_count": self._sample_count,
            "socket_scan_count": self._socket_scan_count,
            "started_monotonic_ns": self._started_monotonic_ns,
            "ended_monotonic_ns": self._ended_monotonic_ns,
            "duration_s": (
                self._ended_monotonic_ns - self._started_monotonic_ns
            )
            / 1_000_000_000,
            "tree_peak": dict(self._tree_peak),
            "tree_summary": {
                key: self._series_summary(values)
                for key, values in self._tree_series.items()
            },
            "smaps_coverage_min": self._coverage_min,
            "nonempty_sample_count": self._nonempty_sample_count,
            "empty_sample_count": self._empty_sample_count,
            "totals": self._counter_totals(),
            "roles": self._role_totals(),
            "observed_processes": observed,
            "observed_pids": sorted({item["pid"] for item in observed}),
            "observed_remote_endpoints": endpoints,
            "observed_non_loopback_remote_endpoints": [
                item["endpoint"] for item in endpoints if not item["loopback"]
            ],
            "observed_loopback_only": (
                all(item["loopback"] for item in endpoints) if endpoints else None
            ),
            "cgroup_start": self._cgroup_start,
            "cgroup_end": cgroup_end,
            "cgroup_delta": _numeric_delta(self._cgroup_start, cgroup_end),
            "errors": list(self._errors),
            "output_path": str(self.output_path),
            "limitations": limitations,
        }
        return self._aggregate


def snapshot_state(root: Path) -> dict[str, Any]:
    """Snapshot path metadata without reading file contents or following links."""

    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError(f"state root must not be a symlink: {root_path}")
    resolved = root_path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)

    entries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    apparent = 0
    allocated = 0
    file_count = 0
    directory_count = 1
    for current, dirnames, filenames in os.walk(resolved, followlinks=False):
        current_path = Path(current)
        for name in sorted([*dirnames, *filenames]):
            path = current_path / name
            relative = path.relative_to(resolved).as_posix()
            try:
                stat_result = path.lstat()
            except OSError as exc:
                errors.append(f"{relative}: {type(exc).__name__}")
                continue
            if path.is_symlink():
                kind = "symlink"
            elif path.is_dir():
                kind = "directory"
                directory_count += 1
            elif path.is_file():
                kind = "file"
                file_count += 1
            else:
                kind = "other"
            size = int(stat_result.st_size) if kind == "file" else 0
            blocks = getattr(stat_result, "st_blocks", None)
            allocated_size = (
                int(blocks * 512)
                if kind == "file" and blocks is not None
                else size
            )
            apparent += size
            allocated += allocated_size
            entries[relative] = {
                "kind": kind,
                "size_bytes": size,
                "allocated_bytes": allocated_size,
                "mtime_ns": int(stat_result.st_mtime_ns),
                "mode": int(stat_result.st_mode & 0o7777),
            }
    return {
        "root": str(resolved),
        "captured_wall_time_ns": time.time_ns(),
        "entries": entries,
        "totals": {
            "file_count": file_count,
            "directory_count": directory_count,
            "entry_count": len(entries),
            "apparent_bytes": apparent,
            "allocated_bytes": allocated,
        },
        "errors": errors,
    }


def diff_state(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return added/removed/modified paths and signed footprint deltas."""

    before_entries = before.get("entries")
    after_entries = after.get("entries")
    if not isinstance(before_entries, Mapping) or not isinstance(
        after_entries,
        Mapping,
    ):
        raise ValueError("state snapshots must contain entry mappings")
    before_paths = set(before_entries)
    after_paths = set(after_entries)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    modified = sorted(
        path
        for path in before_paths & after_paths
        if before_entries[path] != after_entries[path]
    )

    before_totals = before.get("totals", {})
    after_totals = after.get("totals", {})
    keys = (
        "file_count",
        "directory_count",
        "entry_count",
        "apparent_bytes",
        "allocated_bytes",
    )
    delta = {
        key: int(after_totals.get(key, 0)) - int(before_totals.get(key, 0))
        for key in keys
    }
    details = {
        path: {
            "before": before_entries.get(path),
            "after": after_entries.get(path),
        }
        for path in [*added, *removed, *modified]
    }
    return {
        "root_before": before.get("root"),
        "root_after": after.get("root"),
        "added": added,
        "removed": removed,
        "modified": modified,
        "details": details,
        "before_totals": dict(before_totals),
        "after_totals": dict(after_totals),
        "delta": delta,
        "changed_path_count": len(details),
    }


def _coerce_identity(value: Any) -> ProcessIdentity | None:
    if isinstance(value, ProcessIdentity):
        return value
    if isinstance(value, Mapping):
        try:
            raw_ticks = value.get("start_ticks")
            return ProcessIdentity(
                int(value["pid"]),
                float(value["create_time"]),
                int(raw_ticks) if raw_ticks is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return ProcessIdentity(int(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    try:
        process = psutil.Process(int(value))
        return _identity(process)
    except (TypeError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def wait_for_descendants_exit(
    pids: Iterable[int | tuple[int, float] | ProcessIdentity | Mapping[str, Any]],
    timeout: float,
) -> bool:
    """Wait until recorded processes disappear, returning ``False`` on timeout.

    Pass ``observed_processes`` from :meth:`ResourceSampler.stop` when
    possible; those records include creation time and are safe against PID
    reuse.  Plain integer PIDs are bound to their current identity on entry.
    """

    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    identities = {
        identity
        for value in pids
        if (identity := _coerce_identity(value)) is not None
    }
    deadline = time.monotonic() + timeout
    while identities:
        running: set[ProcessIdentity] = set()
        for identity in identities:
            process = _same_process(identity)
            if process is None:
                continue
            status = _optional(process.status, None)
            # A zombie has completed execution and owns no sampled resources;
            # reaping its process-table entry is the launcher's responsibility.
            if status != psutil.STATUS_ZOMBIE:
                running.add(identity)
        identities = running
        if not identities:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


__all__ = [
    "ProcessIdentity",
    "ResourceSampler",
    "classify_process_role",
    "diff_state",
    "snapshot_state",
    "wait_for_descendants_exit",
]
