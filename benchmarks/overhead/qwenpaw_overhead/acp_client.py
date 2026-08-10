"""Lifecycle-aware ACP benchmark client for QwenPaw 2.0.1.

Unlike the interactive TUI transport, this client performs no hidden warmup.
It exposes the subprocess PID immediately after spawn, keeps the ACP session
alive for idle sampling, and records protocol milestones with monotonic clocks.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import hmac
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    Implementation,
    RequestPermissionResponse,
)

from .scenarios import SHELL_COMMAND, get_scenario
from .schemas import EventRecord, SCHEMA_VERSION


_STDIO_BUFFER_LIMIT = 50 * 1024 * 1024


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attr(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _block_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    value = _attr(content, "text")
    return str(value or "")


def _raw_input(tool_call: Any) -> Any:
    return _attr(tool_call, "raw_input", "rawInput")


def _option_id(option: Any) -> str:
    return str(_attr(option, "option_id", "optionId") or "")


def _is_allow_once(option: Any) -> bool:
    return _option_id(option) == "allow_once" and str(
        _attr(option, "kind") or "",
    ) == "allow_once"


@dataclass
class ACPConfig:
    """Configuration for :class:`ACPBenchmarkClient`."""

    command: list[str] = field(
        default_factory=lambda: [
            sys.executable,
            "-m",
            "qwenpaw",
            "acp",
            "--local-diagnostics",
        ],
    )
    cwd: str = field(default_factory=os.getcwd)
    prompt: str = ""
    agent: str | None = None
    cpu_affinity: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    profile: str = "full"
    scenario: str = "S1"
    backend: str = "acp"
    timeout_s: float = 180.0
    available_commands_timeout_s: float = 120.0
    idle_seconds: float = 0.0
    expected_command: str = SHELL_COMMAND
    auto_approve_printf: bool = True
    env: dict[str, str] = field(default_factory=dict, repr=False)
    jsonl_path: Path | None = None
    stderr_path: Path | None = None

    @classmethod
    def from_value(
        cls,
        value: "ACPConfig | Mapping[str, Any] | None",
        overrides: Mapping[str, Any] | None = None,
    ) -> "ACPConfig":
        if isinstance(value, cls):
            data = {
                key: copy.deepcopy(getattr(value, key))
                for key in value.__dataclass_fields__
            }
        elif value is None:
            data = {}
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError("config must be ACPConfig, mapping, or None")
        data.update(dict(overrides or {}))
        if data.get("jsonl_path") is not None:
            data["jsonl_path"] = Path(data["jsonl_path"])
        if data.get("stderr_path") is not None:
            data["stderr_path"] = Path(data["stderr_path"])
        if data.get("command") is not None:
            data["command"] = list(data["command"])
        return cls(**data)

    def resolved_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        try:
            return get_scenario(self.scenario).prompts[0]
        except ValueError:
            return "Reply with exactly: ready"

    def spawn_command(self) -> list[str]:
        command = list(self.command)
        if self.agent and "--agent" not in command:
            command += ["--agent", self.agent]
        if self.cpu_affinity:
            taskset = shutil.which("taskset")
            if taskset is None:
                raise RuntimeError(
                    "cpu_affinity was requested but taskset is not installed",
                )
            command = [taskset, "-c", self.cpu_affinity, *command]
        return command


@dataclass
class ACPStartResult:
    session_id: str
    pid: int
    protocol_version: int
    agent_version: str | None
    available_commands_count: int
    ready: bool
    timestamps_ns: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ACPResult:
    schema_version: str
    run_id: str
    profile: str
    scenario: str
    backend: str
    session_id: str | None
    command: list[str]
    pid: int | None
    success: bool
    timestamps_ns: dict[str, int]
    events: list[dict[str, Any]]
    stop_reason: str | None
    reasoning_text: str
    answer_text: str
    tool_calls: list[dict[str, Any]]
    permission_requests: list[dict[str, Any]]
    available_commands_count: int
    usage: dict[str, int]
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Recorder:
    def __init__(self, config: ACPConfig) -> None:
        self.config = config
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, data: Mapping[str, Any] | None = None) -> int:
        now = time.monotonic_ns()
        row = EventRecord(
            run_id=self.config.run_id,
            profile=self.config.profile,
            scenario=self.config.scenario,
            backend=self.config.backend,
            event=event,
            monotonic_ns=now,
            wall_time_utc=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            data=dict(data or {}),
        ).to_dict()
        self.events.append(row)
        if self.config.jsonl_path is not None:
            self.config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                )
        return now


class _ACPWireClient:
    """ACP callbacks used by ``spawn_agent_process``."""

    def __init__(self, config: ACPConfig, recorder: _Recorder) -> None:
        self.config = config
        self.recorder = recorder
        self.connection: Any = None
        self.session_id: str | None = None
        self.available_commands_event = asyncio.Event()
        self.available_commands_count = 0
        self.timestamps_ns: dict[str, int] = {}
        self.reasoning_parts: list[str] = []
        self.answer_parts: list[str] = []
        self.tools: dict[str, dict[str, Any]] = {}
        self.permission_requests: list[dict[str, Any]] = []
        self.usage: dict[str, int] = {}
        self._safe_permission_consumed = False

    def on_connect(self, connection: Any) -> None:
        self.connection = connection

    def reset_turn(self) -> None:
        for key in (
            "prompt_start",
            "first_reasoning",
            "first_answer",
            "first_tool_call",
            "turn_end",
        ):
            self.timestamps_ns.pop(key, None)
        self.reasoning_parts.clear()
        self.answer_parts.clear()
        self.tools.clear()
        self.permission_requests.clear()
        self.usage.clear()
        self._safe_permission_consumed = False

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **_kwargs: Any,
    ) -> None:
        if self.session_id is not None and session_id != self.session_id:
            return
        kind = str(_attr(update, "session_update", "sessionUpdate") or "")
        if kind == "available_commands_update":
            commands = _attr(update, "available_commands", "availableCommands")
            commands = commands if isinstance(commands, list) else []
            self.available_commands_count = len(commands)
            hashes = []
            for command in commands:
                name = str(_attr(command, "name") or "")
                hashes.append(
                    {
                        "name_len": len(name),
                        "name_sha256": _sha256_text(name) if name else None,
                    },
                )
            timestamp = self.recorder.record(
                "AvailableCommands",
                {"command_count": len(commands), "commands": hashes},
            )
            self.timestamps_ns.setdefault("available_commands", timestamp)
            self.available_commands_event.set()
            return

        if kind == "agent_thought_chunk":
            text = _block_text(_attr(update, "content"))
            if not text:
                return
            if "first_reasoning" not in self.timestamps_ns:
                timestamp = self.recorder.record(
                    "FirstReasoning",
                    {"text_len": len(text), "text_sha256": _sha256_text(text)},
                )
                self.timestamps_ns["first_reasoning"] = timestamp
            self.reasoning_parts.append(text)
            return

        if kind == "agent_message_chunk":
            metadata = _attr(update, "field_meta", "_meta")
            if isinstance(metadata, dict) and isinstance(metadata.get("usage"), dict):
                usage = {
                    key: value
                    for key, value in metadata["usage"].items()
                    if key
                    in {"inputTokens", "outputTokens", "totalTokens"}
                    and isinstance(value, int)
                    and value >= 0
                }
                self.usage = usage
                self.recorder.record("Usage", usage)
            text = _block_text(_attr(update, "content"))
            if not text:
                return
            if "first_answer" not in self.timestamps_ns:
                timestamp = self.recorder.record(
                    "FirstAnswer",
                    {"text_len": len(text), "text_sha256": _sha256_text(text)},
                )
                self.timestamps_ns["first_answer"] = timestamp
            self.answer_parts.append(text)
            return

        if kind in {"tool_call", "tool_call_update"}:
            self._record_tool(update, kind)
            return

        if kind == "usage_update":
            used = _attr(update, "used")
            size = _attr(update, "size")
            data = {
                "used": used if isinstance(used, int) and used >= 0 else None,
                "size": size if isinstance(size, int) and size >= 0 else None,
            }
            self.recorder.record("ContextUsage", data)

    def _record_tool(self, update: Any, kind: str) -> None:
        tool_call_id = str(
            _attr(update, "tool_call_id", "toolCallId") or f"tool-{len(self.tools)}",
        )
        entry = self.tools.setdefault(
            tool_call_id,
            {
                "tool_call_id_sha256": _sha256_text(tool_call_id),
                "name": "",
                "status": None,
                "command_len": 0,
                "command_sha256": None,
            },
        )
        title = str(_attr(update, "title") or "")
        if title:
            entry["name"] = title
        status = _attr(update, "status")
        if status:
            entry["status"] = str(status)
        raw = _raw_input(update)
        if isinstance(raw, dict) and isinstance(raw.get("command"), str):
            command = raw["command"]
            entry["command_len"] = len(command)
            entry["command_sha256"] = _sha256_text(command)
        if "first_tool_call" not in self.timestamps_ns:
            timestamp = self.recorder.record(
                "FirstToolCall",
                {
                    "tool_name_len": len(entry["name"]),
                    "tool_name_sha256": (
                        _sha256_text(entry["name"]) if entry["name"] else None
                    ),
                    "status": entry["status"],
                },
            )
            self.timestamps_ns["first_tool_call"] = timestamp
        else:
            self.recorder.record(
                "ToolCall",
                {
                    "update_kind": kind,
                    "tool_call_id_sha256": entry["tool_call_id_sha256"],
                    "status": entry["status"],
                },
            )

    async def request_permission(
        self,
        options: list[Any],
        session_id: str,
        tool_call: Any,
        **_kwargs: Any,
    ) -> RequestPermissionResponse:
        raw = _raw_input(tool_call)
        title = str(_attr(tool_call, "title") or "")
        kind = str(_attr(tool_call, "kind") or "")
        command = raw.get("command") if isinstance(raw, dict) else None
        allow_options = [option for option in options if _is_allow_once(option)]
        deny_options = [
            option
            for option in options
            if _option_id(option) == "deny"
            and str(_attr(option, "kind") or "") == "reject_once"
        ]
        exact_shape = isinstance(raw, dict) and set(raw) == {"command"}
        exact_card = (
            title == "Bash requires approval (INFO)" and kind == "other"
        )
        exact_command = (
            isinstance(command, str)
            and self.config.expected_command == SHELL_COMMAND
            and hmac.compare_digest(command, self.config.expected_command)
        )
        expected_command_hash = _sha256_text(self.config.expected_command)
        matching_tools = [
            item
            for item in self.tools.values()
            if item.get("name") == "execute_shell_command"
            and item.get("command_sha256") == expected_command_hash
            and item.get("status") == "in_progress"
        ]
        approved = bool(
            self.config.auto_approve_printf
            and not self._safe_permission_consumed
            and session_id == self.session_id
            and exact_card
            and exact_shape
            and exact_command
            and len(matching_tools) == 1
            and len(options) == 2
            and len(allow_options) == 1
            and len(deny_options) == 1
        )
        if approved:
            self._safe_permission_consumed = True
        record = {
            "approved": approved,
            "title_len": len(title),
            "title_sha256": _sha256_text(title) if title else None,
            "kind": kind,
            "command_len": len(command) if isinstance(command, str) else 0,
            "command_sha256": (
                _sha256_text(command) if isinstance(command, str) else None
            ),
            "option_count": len(options),
        }
        self.permission_requests.append(record)
        self.recorder.record("PermissionRequest", record)
        if approved:
            return RequestPermissionResponse(
                outcome=AllowedOutcome(
                    outcome="selected",
                    option_id="allow_once",
                ),
            )
        return RequestPermissionResponse(
            outcome=DeniedOutcome(outcome="cancelled"),
        )

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        del params
        self.recorder.record(
            "ExtensionNotification",
            {
                "method_len": len(method),
                "method_sha256": _sha256_text(method),
            },
        )

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del params
        self.recorder.record(
            "ExtensionMethod",
            {
                "method_len": len(method),
                "method_sha256": _sha256_text(method),
            },
        )
        return {}


class ACPBenchmarkClient:
    """A reusable ACP subprocess/session lifecycle.

    Call ``start()``, optionally keep the instance idle while a sampler reads
    ``pid``, invoke ``prompt()`` one or more times, then ``close()``.
    """

    def __init__(self, config: ACPConfig | Mapping[str, Any] | None = None) -> None:
        self.config = ACPConfig.from_value(config)
        self.recorder = _Recorder(self.config)
        self._wire_client = _ACPWireClient(self.config, self.recorder)
        self._stack: AsyncExitStack | None = None
        self._connection: Any = None
        self._process: Any = None
        self._session_id: str | None = None
        self._spawn_command: list[str] = []
        self._stderr_file: Any = None
        self._started = False
        self._closed = False
        self._stop_reason: str | None = None
        self._error_type: str | None = None
        self._ready = False

    @property
    def pid(self) -> int | None:
        value = getattr(self._process, "pid", None)
        return int(value) if value is not None else None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(
        self,
        process_started_callback: Callable[[int], Any] | None = None,
    ) -> ACPStartResult:
        if self._started:
            raise RuntimeError("ACP benchmark client is already started")
        if self._closed:
            raise RuntimeError("ACP benchmark client is closed")
        self._spawn_command = self.config.spawn_command()
        if not self._spawn_command:
            raise ValueError("ACP command cannot be empty")
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        transport_kwargs: dict[str, Any] = {"limit": _STDIO_BUFFER_LIMIT}
        if self.config.stderr_path is not None:
            self.config.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = self.config.stderr_path.open("ab", buffering=0)
            transport_kwargs["stderr"] = self._stderr_file
        else:
            transport_kwargs["stderr"] = subprocess.DEVNULL
        # Isolation builds a complete environment with proxy variables removed.
        # Merging it back into os.environ would silently undo that cleanup.
        env = dict(self.config.env) if self.config.env else dict(os.environ)
        env.setdefault("PAW_DISABLE_BACKEND_WARMUP", "1")
        try:
            command, *arguments = self._spawn_command
            self._wire_client.timestamps_ns["launch_start"] = self.recorder.record(
                "LaunchStart",
                {
                    "command_arg_count": len(self._spawn_command),
                    "cpu_affinity": self.config.cpu_affinity,
                },
            )
            self._connection, self._process = await self._stack.enter_async_context(
                spawn_agent_process(
                    self._wire_client,
                    command,
                    *arguments,
                    cwd=self.config.cwd,
                    env=env,
                    transport_kwargs=transport_kwargs,
                ),
            )
            pid = self.pid
            if pid is None:
                raise RuntimeError("ACP subprocess has no pid")
            if process_started_callback is not None:
                callback_result = process_started_callback(pid)
                if inspect.isawaitable(callback_result):
                    await callback_result
            self._wire_client.timestamps_ns["process_started"] = self.recorder.record(
                "ProcessStarted",
                {
                    "pid": pid,
                    "command_arg_count": len(self._spawn_command),
                    "cpu_affinity": self.config.cpu_affinity,
                },
            )

            self._wire_client.timestamps_ns["initialize_start"] = self.recorder.record(
                "InitializeStart",
            )
            initialized = await asyncio.wait_for(
                self._connection.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(
                        name="qwenpaw-overhead",
                        version=SCHEMA_VERSION,
                    ),
                ),
                timeout=self.config.timeout_s,
            )
            protocol_version = int(initialized.protocol_version)
            self._wire_client.timestamps_ns["initialize"] = self.recorder.record(
                "Initialize",
                {
                    "protocol_version": protocol_version,
                    "protocol_match": protocol_version == PROTOCOL_VERSION,
                },
            )
            session = await asyncio.wait_for(
                self._connection.new_session(cwd=self.config.cwd),
                timeout=self.config.timeout_s,
            )
            self._session_id = str(session.session_id)
            self._wire_client.session_id = self._session_id
            self._wire_client.timestamps_ns["connected"] = self.recorder.record(
                "Connected",
                {
                    "session_id_sha256": _sha256_text(self._session_id),
                },
            )
            try:
                await asyncio.wait_for(
                    self._wire_client.available_commands_event.wait(),
                    timeout=self.config.available_commands_timeout_s,
                )
                self._ready = True
            except asyncio.TimeoutError:
                self.recorder.record("AvailableCommandsTimeout")
            self._started = True
            agent_info = getattr(initialized, "agent_info", None)
            agent_version = getattr(agent_info, "version", None)
            return ACPStartResult(
                session_id=self._session_id,
                pid=pid,
                protocol_version=protocol_version,
                agent_version=str(agent_version) if agent_version else None,
                available_commands_count=self._wire_client.available_commands_count,
                ready=self._ready,
                timestamps_ns=dict(self._wire_client.timestamps_ns),
            )
        except BaseException:
            await self.close()
            raise

    async def idle(self, seconds: float) -> None:
        if not self._started:
            raise RuntimeError("ACP benchmark client is not started")
        if seconds < 0:
            raise ValueError("idle duration cannot be negative")
        self.recorder.record("IdleStart", {"duration_ms": int(seconds * 1000)})
        await asyncio.sleep(seconds)
        self.recorder.record("IdleEnd", {"duration_ms": int(seconds * 1000)})

    async def prompt(self, text: str | None = None) -> dict[str, Any]:
        if not self._started or self._connection is None or self._session_id is None:
            raise RuntimeError("ACP benchmark client is not started")
        self._wire_client.reset_turn()
        prompt = text if text is not None else self.config.resolved_prompt()
        self._wire_client.timestamps_ns["prompt_start"] = self.recorder.record(
            "PromptStart",
            {
                "prompt_len": len(prompt),
                "prompt_sha256": _sha256_text(prompt),
            },
        )
        try:
            response = await asyncio.wait_for(
                self._connection.prompt(
                    prompt=[text_block(prompt)],
                    session_id=self._session_id,
                ),
                timeout=self.config.timeout_s,
            )
            self._stop_reason = str(getattr(response, "stop_reason", "") or "")
            await self._settle()
            self._wire_client.timestamps_ns["turn_end"] = self.recorder.record(
                "TurnEnd",
                {
                    "stop_reason": self._stop_reason,
                    "tool_call_count": len(self._wire_client.tools),
                    "permission_request_count": len(
                        self._wire_client.permission_requests,
                    ),
                },
            )
            return self._turn_result()
        except BaseException as exc:
            self._error_type = type(exc).__name__
            self.recorder.record("ACPError", {"error_type": self._error_type})
            raise

    async def _settle(self, max_idle: int = 5) -> None:
        idle = 0
        previous_count = len(self.recorder.events)
        while idle < max_idle:
            await asyncio.sleep(0)
            current_count = len(self.recorder.events)
            if current_count == previous_count:
                idle += 1
            else:
                idle = 0
                previous_count = current_count

    def _turn_result(self) -> dict[str, Any]:
        timestamps = dict(self._wire_client.timestamps_ns)
        start = timestamps.get("prompt_start")

        def elapsed(key: str) -> float | None:
            value = timestamps.get(key)
            if start is None or value is None:
                return None
            return (value - start) / 1_000_000

        return {
            "stop_reason": self._stop_reason,
            "timestamps_ns": timestamps,
            "latency_ms": {
                "ttft_reasoning": elapsed("first_reasoning"),
                "ttft_answer": elapsed("first_answer"),
                "first_tool_call": elapsed("first_tool_call"),
                "total": elapsed("turn_end"),
            },
            "reasoning_text": "".join(self._wire_client.reasoning_parts),
            "answer_text": "".join(self._wire_client.answer_parts),
            "tool_calls": list(self._wire_client.tools.values()),
            "permission_requests": list(self._wire_client.permission_requests),
            "usage": dict(self._wire_client.usage),
        }

    def result(self) -> ACPResult:
        return ACPResult(
            schema_version=SCHEMA_VERSION,
            run_id=self.config.run_id,
            profile=self.config.profile,
            scenario=self.config.scenario,
            backend=self.config.backend,
            session_id=self._session_id,
            command=list(self._spawn_command),
            pid=self.pid,
            success=(
                self._error_type is None
                and self._ready
                and self._stop_reason is not None
            ),
            timestamps_ns=dict(self._wire_client.timestamps_ns),
            events=list(self.recorder.events),
            stop_reason=self._stop_reason,
            reasoning_text="".join(self._wire_client.reasoning_parts),
            answer_text="".join(self._wire_client.answer_parts),
            tool_calls=list(self._wire_client.tools.values()),
            permission_requests=list(self._wire_client.permission_requests),
            available_commands_count=self._wire_client.available_commands_count,
            usage=dict(self._wire_client.usage),
            error_type=self._error_type,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None and self._session_id is not None:
            try:
                await self._connection.close_session(session_id=self._session_id)
            except Exception:
                self.recorder.record("CloseSessionError")
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        self.recorder.record("ClientClosed")

    async def __aenter__(self) -> "ACPBenchmarkClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


async def run_acp_async(
    config: ACPConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ACPResult:
    """One-shot convenience wrapper around :class:`ACPBenchmarkClient`."""

    cfg = ACPConfig.from_value(config, kwargs)
    client = ACPBenchmarkClient(cfg)
    try:
        await client.start()
        if cfg.idle_seconds:
            await client.idle(cfg.idle_seconds)
        await client.prompt()
        return client.result()
    except BaseException as exc:
        client._error_type = type(exc).__name__  # noqa: SLF001
        return client.result()
    finally:
        await client.close()


def run_acp(
    config: ACPConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ACPResult:
    """Synchronously run one ACP benchmark turn."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_acp_async(config, **kwargs))
    raise RuntimeError("run_acp() cannot run in an active event loop; await run_acp_async()")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        nargs="+",
        help="ACP command; defaults to the current Python's qwenpaw module",
    )
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--agent")
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--stderr", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ACPConfig(
        command=args.command
        or [sys.executable, "-m", "qwenpaw", "acp", "--local-diagnostics"],
        cwd=args.cwd,
        agent=args.agent,
        cpu_affinity=args.cpu_affinity,
        prompt=args.prompt,
        scenario=args.scenario,
        profile=args.profile,
        run_id=args.run_id or uuid.uuid4().hex,
        idle_seconds=args.idle_seconds,
        timeout_s=args.timeout,
        jsonl_path=args.jsonl,
        stderr_path=args.stderr,
    )
    result = run_acp(config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
