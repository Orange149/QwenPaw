"""Deterministic OpenAI-compatible server for overhead measurements.

The server deliberately implements only the endpoints QwenPaw's OpenAI and
DashScope-compatible providers need: ``GET /models`` and
``POST /chat/completions``.  Responses are stable and synthetic, so benchmark
artifacts never need to contain user data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from .scenarios import S1, S2, S3, SHELL_COMMAND, SHELL_OUTPUT, WARMUP_PROMPT


_SCHEMA_VERSION = "1"
_TOOL_CALL_ID = "call_qwenpaw_overhead_printf_1"
_TITLE_PROMPT = (
    "You generate short titles for chat sessions. Given the first user "
    "message, reply with a concise title (at most 6 words, no quotes, no "
    "trailing punctuation, same language as the message) that captures the "
    "topic. Reply with the title only."
)
_MEMORY_SYSTEM_PREFIXES = (
    "You are an automatic memory system.",
    "你是自动记忆系统。",
)
_APPROVAL_GENERALIZE_SYSTEM_PROMPT = (
    "You generalize a single tool-call target into a conservative glob "
    "pattern so that future, similar calls are auto-approved without "
    "asking again. You MUST output ONLY the glob pattern — no "
    "explanation, no quotes, no backticks, no tool name, no "
    "parentheses, no leading/trailing whitespace."
)
_APPROVAL_GENERALIZE_USER_PROMPT = (
    "tool_name: Bash\n"
    "tool_type: shell\n"
    f"target: {SHELL_COMMAND}\n\n"
    "This is a shell command. Generalize CONSERVATIVELY: replace "
    "varying arguments with '*' while KEEPING the command name and "
    "any subcommand. Examples: 'git status' -> 'git *', "
    "'npm run build' -> 'npm run *'. Do NOT widen destructive "
    "commands (rm, dd, mkfs, sudo, chmod 777, > /dev/...) — return "
    "them unchanged. Never output a bare '*'.\n\n"
    "glob pattern:"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_request_summary(body: dict[str, Any], raw_size: int) -> dict[str, Any]:
    """Return structural request facts without retaining prompt/tool content."""

    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    roles: dict[str, int] = {}
    content_bytes = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
        content = message.get("content")
        if isinstance(content, str):
            content_bytes += len(content.encode("utf-8"))
        elif content is not None:
            encoded = json.dumps(
                content,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            content_bytes += len(encoded)

    tools = body.get("tools")
    tools = tools if isinstance(tools, list) else []
    tools_bytes = len(
        json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8",
        ),
    )
    model = str(body.get("model") or "")
    return {
        "body_bytes": raw_size,
        "message_count": len(messages),
        "message_roles": roles,
        "message_content_bytes": content_bytes,
        "tool_count": len(tools),
        "tool_schema_bytes": tools_bytes,
        "model_len": len(model),
        "model_sha256": _sha256_text(model) if model else None,
        "stream": body.get("stream") is True,
        "max_tokens": body.get("max_tokens"),
        "enable_thinking": body.get("enable_thinking"),
    }


def _tool_names(body: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    tools = body.get("tools")
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def _has_tool_result(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in messages
    )


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        value = block.get("text")
        if not isinstance(value, str):
            return None
        parts.append(value)
    return "".join(parts)


def _expected_s3_answers(before_turn: int) -> list[str]:
    answers: list[str] = []
    for turn in range(1, before_turn + 1):
        if turn == 1:
            answers.append("STORED.")
        else:
            previous = turn - 1
            value = f"V{previous * 7919 % 100000:05d}"
            answers.append(f"K{previous:02d}={value};STORED.")
    return answers


def classify_request_kind(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    system_texts = [
        text
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "system"
        and (text := _message_text(message)) is not None
    ]
    user_texts = [
        text
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and (text := _message_text(message)) is not None
    ]
    if user_texts == [WARMUP_PROMPT]:
        return "backend_warmup"
    if system_texts == [_APPROVAL_GENERALIZE_SYSTEM_PROMPT]:
        return "approval_generalization"
    if len(system_texts) == 1 and _TITLE_PROMPT in system_texts[0]:
        return "auto_title"
    if len(system_texts) == 1 and any(
        prefix in system_texts[0] for prefix in _MEMORY_SYSTEM_PREFIXES
    ):
        return "auto_memory"
    return "primary"


def _validate_auxiliary_request(
    kind: str,
    body: dict[str, Any],
) -> list[str]:
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    typed = [message for message in messages if isinstance(message, dict)]
    systems = [
        text
        for message in typed
        if message.get("role") == "system"
        and (text := _message_text(message)) is not None
    ]
    users = [
        text
        for message in typed
        if message.get("role") == "user"
        and (text := _message_text(message)) is not None
    ]
    if kind == "auto_title":
        if len(systems) != 1 or _TITLE_PROMPT not in systems[0]:
            return ["title_system_prompt_mismatch"]
        if len(users) != 1 or "QWENPAW_OVERHEAD_" not in users[0]:
            return ["title_user_prompt_mismatch"]
        return []
    if kind == "auto_memory":
        if len(systems) != 1 or not any(
            prefix in systems[0] for prefix in _MEMORY_SYSTEM_PREFIXES
        ):
            return ["memory_system_prompt_mismatch"]
        if len(users) != 1 or not (
            "QWENPAW_OVERHEAD_" in users[0]
            and (
                "# Recent Conversation" in users[0]
                or "# 最近的对话" in users[0]
            )
        ):
            return ["memory_user_prompt_mismatch"]
        return []
    if kind == "backend_warmup":
        if users != [WARMUP_PROMPT]:
            return ["backend_warmup_prompt_mismatch"]
        return []
    if kind == "approval_generalization":
        errors: list[str] = []
        if systems != [_APPROVAL_GENERALIZE_SYSTEM_PROMPT]:
            errors.append("approval_generalization_system_mismatch")
        if users != [_APPROVAL_GENERALIZE_USER_PROMPT]:
            errors.append("approval_generalization_target_mismatch")
        if _tool_names(body):
            errors.append("approval_generalization_tools_present")
        return errors
    return ["unknown_auxiliary_kind"]


def _validate_scenario_request(
    scenario: str,
    body: dict[str, Any],
    request_index: int,
) -> list[str]:
    """Validate the synthetic task without logging any message content."""

    messages = body.get("messages")
    if not isinstance(messages, list):
        return ["messages_not_list"]
    typed = [message for message in messages if isinstance(message, dict)]
    users = [
        text
        for message in typed
        if message.get("role") == "user"
        and (text := _message_text(message)) is not None
    ]
    assistants = [
        text
        for message in typed
        if message.get("role") == "assistant"
        and (text := _message_text(message)) is not None
        and text
    ]
    errors: list[str] = []
    if scenario == "WARMUP":
        if users != [WARMUP_PROMPT]:
            errors.append("warmup_user_prompt_mismatch")
    elif scenario == "S1":
        if users != [S1.prompts[0]]:
            errors.append("s1_user_prompt_mismatch")
    elif scenario == "S2":
        if users != [S2.prompts[0]]:
            errors.append("s2_user_prompt_mismatch")
        has_result = _has_tool_result(body)
        if request_index == 1 and has_result:
            errors.append("s2_unexpected_first_round_tool_result")
        elif request_index == 2:
            tool_messages = [
                message for message in typed if message.get("role") == "tool"
            ]
            if len(tool_messages) != 1 or _message_text(tool_messages[0]) != (
                SHELL_OUTPUT
            ):
                errors.append("s2_tool_result_mismatch")
            tool_calls = [
                call
                for message in typed
                if message.get("role") == "assistant"
                and isinstance(message.get("tool_calls"), list)
                for call in message["tool_calls"]
                if isinstance(call, dict)
            ]
            commands: list[str] = []
            for call in tool_calls:
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                try:
                    arguments = json.loads(str(function.get("arguments") or ""))
                except json.JSONDecodeError:
                    continue
                if (
                    function.get("name") == "execute_shell_command"
                    and isinstance(arguments, dict)
                    and isinstance(arguments.get("command"), str)
                ):
                    commands.append(arguments["command"])
            if commands != [SHELL_COMMAND]:
                errors.append("s2_tool_call_history_mismatch")
        elif request_index != 1:
            errors.append("s2_unexpected_request_index")
    elif scenario == "S3":
        if not 1 <= request_index <= len(S3.prompts):
            errors.append("s3_unexpected_request_index")
        else:
            if users != list(S3.prompts[:request_index]):
                errors.append("s3_user_history_mismatch")
            if assistants != _expected_s3_answers(request_index - 1):
                errors.append("s3_assistant_history_mismatch")
    return errors


@dataclass
class _MockState:
    run_id: str
    profile: str
    scenario: str
    backend: str
    model: str
    tool_command: str
    answer_text: str
    reasoning_text: str
    first_chunk_delay_s: float
    between_chunk_delay_s: float
    pause_after_first_chunk: bool
    pause_timeout_s: float
    jsonl_path: Path | None
    events: list[dict[str, Any]] = field(default_factory=list)
    request_count: int = 0
    primary_request_count: int = 0
    auxiliary_request_count: int = 0
    tool_call_emissions: int = 0
    request_bodies: dict[int, dict[str, Any]] = field(default_factory=dict)
    primary_request_bodies: dict[int, dict[str, Any]] = field(
        default_factory=dict,
    )
    validation_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    events_lock: threading.Lock = field(default_factory=threading.Lock)
    first_chunk_sent: threading.Event = field(default_factory=threading.Event)
    continue_stream: threading.Event = field(default_factory=threading.Event)

    def record(self, event: str, data: dict[str, Any] | None = None) -> None:
        row = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile": self.profile,
            "scenario": self.scenario,
            "backend": self.backend,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": _utc_now(),
            "data": data or {},
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self.events_lock:
            self.events.append(row)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded + "\n")

    def next_request_index(self) -> int:
        with self.lock:
            self.request_count += 1
            return self.request_count

    def retain_request_body(
        self,
        request_index: int,
        body: dict[str, Any],
        *,
        kind: str,
        kind_index: int,
    ) -> None:
        """Keep an in-memory-only copy for exact-wire replay."""

        with self.lock:
            self.request_bodies[request_index] = copy.deepcopy(body)
            if kind == "primary":
                self.primary_request_bodies[kind_index] = copy.deepcopy(body)

    def next_kind_index(self, kind: str) -> int:
        with self.lock:
            if kind == "primary":
                self.primary_request_count += 1
                return self.primary_request_count
            self.auxiliary_request_count += 1
            return self.auxiliary_request_count

    def reserve_tool_call(self) -> bool:
        with self.lock:
            if self.tool_call_emissions:
                return False
            self.tool_call_emissions = 1
            return True

    def answer_for_request(self, request_index: int) -> str:
        if self.scenario != "S3":
            return self.answer_text
        turn = max(1, min(request_index, 10))
        if turn == 1:
            return "STORED."
        previous = turn - 1
        previous_value = f"V{previous * 7919 % 100000:05d}"
        return f"K{previous:02d}={previous_value};STORED."

    def after_first_chunk(self) -> None:
        self.first_chunk_sent.set()
        if self.pause_after_first_chunk:
            self.continue_stream.wait(timeout=self.pause_timeout_s)
        if self.between_chunk_delay_s:
            time.sleep(self.between_chunk_delay_s)


class _MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: _MockState) -> None:
        super().__init__(address, _MockHandler)
        self.state = state


class _MockHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible request handler."""

    protocol_version = "HTTP/1.1"
    server: _MockHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.state.model,
                            "object": "model",
                        },
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.split("?", 1)[0].rstrip("/").endswith(
            "/chat/completions",
        ):
            self._send_json(404, {"error": {"type": "not_found"}})
            return

        raw = self._read_body()
        try:
            decoded = json.loads(raw or b"{}")
            body = decoded if isinstance(decoded, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": {"type": "invalid_json"}})
            return

        state = self.server.state
        request_index = state.next_request_index()
        request_kind = classify_request_kind(body)
        if state.scenario == "WARMUP" and request_kind == "backend_warmup":
            request_kind = "primary"
        kind_index = state.next_kind_index(request_kind)
        state.retain_request_body(
            request_index,
            body,
            kind=request_kind,
            kind_index=kind_index,
        )
        summary = _safe_request_summary(body, len(raw))
        summary["request_index"] = request_index
        summary["request_kind"] = request_kind
        summary["kind_request_index"] = kind_index
        state.record("mock_request_received", summary)

        if request_kind != "primary":
            auxiliary_errors = _validate_auxiliary_request(request_kind, body)
            state.record(
                "mock_auxiliary_request_validated",
                {
                    "request_index": request_index,
                    "request_kind": request_kind,
                    "kind_request_index": kind_index,
                    "valid": not auxiliary_errors,
                    "error_codes": auxiliary_errors,
                },
            )
            if auxiliary_errors:
                with state.lock:
                    state.validation_failures += 1
                self._send_json(
                    422,
                    {"error": {"type": "auxiliary_request_mismatch"}},
                )
                return
            if request_kind == "auto_title":
                auxiliary_answer = "QwenPaw Benchmark"
            elif request_kind == "backend_warmup":
                auxiliary_answer = "ready"
            elif request_kind == "approval_generalization":
                # Preserve the exact fixed target.  Widening it would make a
                # later shell call eligible for approval and invalidate the
                # benchmark's one-command safety boundary.
                auxiliary_answer = SHELL_COMMAND
            else:
                auxiliary_answer = "Skipped: synthetic benchmark data"
            if body.get("stream") is False:
                self._send_non_stream_text(
                    request_index,
                    answer_text=auxiliary_answer,
                    usage_round=1,
                )
            else:
                self._stream_text(
                    request_index,
                    answer_text=auxiliary_answer,
                    usage_round=1,
                )
            return

        validation_errors = _validate_scenario_request(
            state.scenario,
            body,
            kind_index,
        )
        state.record(
            "mock_request_validated",
            {
                "request_index": request_index,
                "primary_request_index": kind_index,
                "valid": not validation_errors,
                "error_codes": validation_errors,
            },
        )
        if validation_errors:
            with state.lock:
                state.validation_failures += 1
            self._send_json(
                422,
                {"error": {"type": "scenario_request_mismatch"}},
            )
            return

        if state.scenario == "S2" and not _has_tool_result(body):
            if "execute_shell_command" not in _tool_names(body):
                self._send_json(
                    422,
                    {"error": {"type": "required_tool_not_advertised"}},
                )
                return
            if not state.reserve_tool_call():
                # A retry or accidental third first-round request must never
                # execute the synthetic command twice.
                self._send_json(
                    409,
                    {"error": {"type": "tool_call_already_emitted"}},
                )
                return
            if body.get("stream") is False:
                self._send_non_stream_tool(request_index)
            else:
                self._stream_tool_call(request_index)
            return

        answer_text = state.answer_for_request(kind_index)
        usage_round = 2 if state.scenario == "S2" else 1
        if body.get("stream") is False:
            self._send_non_stream_text(
                request_index,
                answer_text=answer_text,
                usage_round=usage_round,
            )
        else:
            self._stream_text(
                request_index,
                answer_text=answer_text,
                usage_round=usage_round,
            )

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _send_headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)
        self.wfile.flush()

    def _write_sse(self, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            data = payload
        else:
            data = json.dumps(payload, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _chunk(
        self,
        *,
        delta: dict[str, Any],
        finish_reason: str | None = None,
        chunk_id: str = "chatcmpl-qwenpaw-overhead",
    ) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": self.server.state.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                },
            ],
        }

    @staticmethod
    def _usage(round_index: int) -> dict[str, int]:
        prompt = 64 if round_index == 1 else 96
        completion = 8 if round_index == 1 else 6
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _stream_text(
        self,
        request_index: int,
        *,
        answer_text: str,
        usage_round: int,
    ) -> None:
        state = self.server.state
        usage = self._usage(usage_round)
        self._send_headers(200, "text/event-stream")
        if state.first_chunk_delay_s:
            time.sleep(state.first_chunk_delay_s)

        first_written = False
        try:
            if state.reasoning_text:
                self._write_sse(
                    self._chunk(
                        delta={
                            "role": "assistant",
                            "reasoning_content": state.reasoning_text,
                        },
                    ),
                )
                state.record(
                    "mock_first_reasoning",
                    {
                        "request_index": request_index,
                        "text_len": len(state.reasoning_text),
                        "text_sha256": _sha256_text(state.reasoning_text),
                    },
                )
                first_written = True
                state.after_first_chunk()

            self._write_sse(
                self._chunk(
                    delta={"role": "assistant", "content": answer_text},
                ),
            )
            state.record(
                "mock_first_answer",
                {
                    "request_index": request_index,
                    "text_len": len(answer_text),
                    "text_sha256": _sha256_text(answer_text),
                },
            )
            if not first_written:
                state.after_first_chunk()

            self._write_sse(self._chunk(delta={}, finish_reason="stop"))
            self._write_sse(
                {
                    "id": "chatcmpl-qwenpaw-overhead",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": state.model,
                    "choices": [],
                    "usage": usage,
                },
            )
            self._write_sse("[DONE]")
            state.record(
                "mock_usage",
                {"request_index": request_index, "synthetic": True, **usage},
            )
            state.record(
                "mock_stream_done",
                {"request_index": request_index, "usage": usage},
            )
        except (BrokenPipeError, ConnectionResetError):
            state.record(
                "mock_client_disconnected",
                {"request_index": request_index},
            )

    def _stream_tool_call(self, request_index: int) -> None:
        state = self.server.state
        usage = self._usage(1)
        self._send_headers(200, "text/event-stream")
        if state.first_chunk_delay_s:
            time.sleep(state.first_chunk_delay_s)
        arguments = json.dumps(
            {"command": state.tool_command},
            separators=(",", ":"),
        )
        try:
            self._write_sse(
                self._chunk(
                    delta={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": _TOOL_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": "execute_shell_command",
                                    "arguments": arguments,
                                },
                            },
                        ],
                    },
                    chunk_id="chatcmpl-qwenpaw-overhead-tool",
                ),
            )
            state.record(
                "mock_tool_call",
                {
                    "request_index": request_index,
                    "tool_call_id_sha256": _sha256_text(_TOOL_CALL_ID),
                    "tool_name": "execute_shell_command",
                    "arguments_len": len(arguments),
                    "arguments_sha256": _sha256_text(arguments),
                },
            )
            state.after_first_chunk()
            self._write_sse(
                self._chunk(
                    delta={},
                    finish_reason="tool_calls",
                    chunk_id="chatcmpl-qwenpaw-overhead-tool",
                ),
            )
            self._write_sse(
                {
                    "id": "chatcmpl-qwenpaw-overhead-tool",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": state.model,
                    "choices": [],
                    "usage": usage,
                },
            )
            self._write_sse("[DONE]")
            state.record(
                "mock_usage",
                {"request_index": request_index, "synthetic": True, **usage},
            )
            state.record(
                "mock_stream_done",
                {"request_index": request_index, "usage": usage},
            )
        except (BrokenPipeError, ConnectionResetError):
            state.record(
                "mock_client_disconnected",
                {"request_index": request_index},
            )

    def _send_non_stream_text(
        self,
        request_index: int,
        *,
        answer_text: str,
        usage_round: int,
    ) -> None:
        state = self.server.state
        usage = self._usage(usage_round)
        self._send_json(
            200,
            {
                "id": "chatcmpl-qwenpaw-overhead",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": state.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer_text,
                            **(
                                {"reasoning_content": state.reasoning_text}
                                if state.reasoning_text
                                else {}
                            ),
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": usage,
            },
        )
        state.record(
            "mock_usage",
            {"request_index": request_index, "synthetic": True, **usage},
        )
        state.record(
            "mock_response_done",
            {"request_index": request_index, "usage": usage},
        )

    def _send_non_stream_tool(self, request_index: int) -> None:
        usage = self._usage(1)
        arguments = json.dumps(
            {"command": self.server.state.tool_command},
            separators=(",", ":"),
        )
        self._send_json(
            200,
            {
                "id": "chatcmpl-qwenpaw-overhead-tool",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": self.server.state.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": _TOOL_CALL_ID,
                                    "type": "function",
                                    "function": {
                                        "name": "execute_shell_command",
                                        "arguments": arguments,
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
                "usage": usage,
            },
        )
        self.server.state.record(
            "mock_tool_call",
            {
                "request_index": request_index,
                "tool_call_id_sha256": _sha256_text(_TOOL_CALL_ID),
                "tool_name": "execute_shell_command",
                "arguments_len": len(arguments),
                "arguments_sha256": _sha256_text(arguments),
            },
        )
        self.server.state.record(
            "mock_usage",
            {"request_index": request_index, "synthetic": True, **usage},
        )
        self.server.state.record(
            "mock_response_done",
            {"request_index": request_index, "usage": usage},
        )

    def log_message(self, _format: str, *_args: Any) -> None:
        return


@dataclass
class MockServerHandle:
    """Handle yielded by :func:`start_server`."""

    base_url: str
    events: list[dict[str, Any]]
    _server: _MockHTTPServer
    _thread: threading.Thread

    @property
    def request_count(self) -> int:
        return self._server.state.request_count

    @property
    def primary_request_count(self) -> int:
        return self._server.state.primary_request_count

    @property
    def auxiliary_request_count(self) -> int:
        return self._server.state.auxiliary_request_count

    @property
    def tool_call_emissions(self) -> int:
        return self._server.state.tool_call_emissions

    @property
    def validation_failures(self) -> int:
        return self._server.state.validation_failures

    def wait_for_first_chunk(self, timeout: float = 5.0) -> bool:
        return self._server.state.first_chunk_sent.wait(timeout)

    def release_stream(self) -> None:
        self._server.state.continue_stream.set()

    def snapshot_events(self) -> list[dict[str, Any]]:
        with self._server.state.events_lock:
            return list(self.events)

    @property
    def request_bodies(self) -> dict[int, dict[str, Any]]:
        """Copies of request JSON, retained only while the context is open."""

        with self._server.state.lock:
            return copy.deepcopy(self._server.state.request_bodies)

    def request_body(self, request_index: int) -> dict[str, Any] | None:
        with self._server.state.lock:
            body = self._server.state.request_bodies.get(request_index)
            return copy.deepcopy(body) if body is not None else None

    @property
    def primary_request_bodies(self) -> dict[int, dict[str, Any]]:
        with self._server.state.lock:
            return copy.deepcopy(self._server.state.primary_request_bodies)

    def primary_request_body(self, request_index: int) -> dict[str, Any] | None:
        with self._server.state.lock:
            body = self._server.state.primary_request_bodies.get(request_index)
            return copy.deepcopy(body) if body is not None else None

    def last_request_body(self) -> dict[str, Any] | None:
        with self._server.state.lock:
            if not self._server.state.request_bodies:
                return None
            request_index = max(self._server.state.request_bodies)
            return copy.deepcopy(self._server.state.request_bodies[request_index])

    def close(self) -> None:
        self.release_stream()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        with self._server.state.lock:
            self._server.state.request_bodies.clear()
            self._server.state.primary_request_bodies.clear()


@contextmanager
def start_server(
    *,
    scenario: str = "S1",
    host: str = "127.0.0.1",
    port: int = 0,
    run_id: str = "mock-run",
    profile: str = "mock",
    backend: str = "mock",
    model: str = "qwenpaw-overhead-mock",
    tool_command: str = SHELL_COMMAND,
    answer_text: str | None = None,
    reasoning_text: str = "",
    first_chunk_delay_s: float = 0.0,
    between_chunk_delay_s: float = 0.0,
    pause_after_first_chunk: bool = False,
    pause_timeout_s: float = 10.0,
    jsonl_path: str | Path | None = None,
) -> Iterator[MockServerHandle]:
    """Start a deterministic server in a background thread.

    ``WARMUP`` accepts only the stock TUI warmup and replies ``ready``.
    ``S1`` emits one fixed short answer. ``S2`` emits exactly one
    ``execute_shell_command`` call, then emits the fixed answer only after a
    subsequent request contains a ``role=tool`` message. ``S3`` emits the
    deterministic acknowledgement/recall answer for request indices 1..10.
    """

    normalized = scenario.upper()
    aliases = {"TEXT": "S1", "SHORT_TEXT": "S1", "SHELL": "S2", "TOOL": "S2"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"WARMUP", "S1", "S2", "S3"}:
        raise ValueError(
            "mock scenario must be WARMUP, S1/text, S2/shell, or S3",
        )
    if port < 0 or port > 65535:
        raise ValueError("port must be in the range 0..65535")

    if normalized == "WARMUP":
        default_answer = "ready"
    elif normalized == "S1":
        default_answer = S1.expected_text
    elif normalized == "S2":
        default_answer = S2.expected_text
    else:
        default_answer = "STORED."
    state = _MockState(
        run_id=run_id,
        profile=profile,
        scenario=normalized,
        backend=backend,
        model=model,
        tool_command=tool_command,
        answer_text=answer_text or default_answer or SHELL_OUTPUT,
        reasoning_text=reasoning_text,
        first_chunk_delay_s=max(0.0, first_chunk_delay_s),
        between_chunk_delay_s=max(0.0, between_chunk_delay_s),
        pause_after_first_chunk=pause_after_first_chunk,
        pause_timeout_s=max(0.1, pause_timeout_s),
        jsonl_path=Path(jsonl_path) if jsonl_path is not None else None,
    )
    server = _MockHTTPServer((host, port), state)
    thread = threading.Thread(
        target=server.serve_forever,
        name="qwenpaw-overhead-mock",
        daemon=True,
    )
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    handle = MockServerHandle(
        base_url=f"http://{bound_host}:{bound_port}/v1",
        events=state.events,
        _server=server,
        _thread=thread,
    )
    state.record(
        "mock_server_started",
        {"host": str(bound_host), "port": int(bound_port)},
    )
    try:
        yield handle
    finally:
        state.record("mock_server_stopping")
        handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="S1",
        choices=("WARMUP", "S1", "S2", "S3"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8009, type=int)
    parser.add_argument("--run-id", default="mock-cli")
    parser.add_argument("--jsonl", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    with start_server(
        scenario=args.scenario,
        host=args.host,
        port=args.port,
        run_id=args.run_id,
        jsonl_path=args.jsonl,
    ) as server:
        print(server.base_url, flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
