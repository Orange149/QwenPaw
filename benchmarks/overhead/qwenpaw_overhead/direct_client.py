"""Minimal OpenAI-compatible streaming client used as the direct baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import httpx

from .schemas import EventRecord, SCHEMA_VERSION
from .scenarios import SHELL_COMMAND


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return stripped + "/chat/completions"


def _safe_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            result[key] = raw
    details = value.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            result["cached_tokens"] = cached
    return result


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8",
        ),
    )


def _request_summary(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    tools = body.get("tools")
    tools = tools if isinstance(tools, list) else []
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
            content_bytes += _json_size(content)
    model = str(body.get("model") or "")
    raw = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "body_bytes": len(raw),
        "body_sha256": hashlib.sha256(raw).hexdigest(),
        "message_count": len(messages),
        "message_roles": roles,
        "message_content_bytes": content_bytes,
        "tool_count": len(tools),
        "tool_schema_bytes": _json_size(tools),
        "model_len": len(model),
        "model_sha256": _sha256_text(model) if model else None,
        "stream": body.get("stream") is True,
    }


@dataclass
class DirectRequestConfig:
    """Configuration accepted by :func:`run_request`."""

    base_url: str
    model: str = ""
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    request_body: dict[str, Any] | None = None
    api_key: str | None = field(default=None, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    profile: str = "direct"
    scenario: str = "unknown"
    backend: str = "direct"
    timeout_s: float = 120.0
    max_tokens: int | None = None
    enable_thinking: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = True
    jsonl_path: Path | None = None
    trust_env: bool = True
    allow_exact_shell: bool = False
    expected_command: str = SHELL_COMMAND
    tool_timeout_s: float = 5.0

    @classmethod
    def from_value(
        cls,
        value: "DirectRequestConfig | Mapping[str, Any] | None",
        overrides: Mapping[str, Any],
    ) -> "DirectRequestConfig":
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
            raise TypeError("config must be DirectRequestConfig, mapping, or None")
        data.update(overrides)
        if "body" in data and "request_body" not in data:
            data["request_body"] = data.pop("body")
        if "output_jsonl" in data and "jsonl_path" not in data:
            data["jsonl_path"] = data.pop("output_jsonl")
        if data.get("jsonl_path") is not None:
            data["jsonl_path"] = Path(data["jsonl_path"])
        return cls(**data)

    def build_body(self) -> dict[str, Any]:
        if self.request_body is not None:
            return copy.deepcopy(self.request_body)
        messages = self.messages
        if messages is None:
            messages = [{"role": "user", "content": self.prompt}]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "stream": self.stream,
        }
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        for key, value in (
            ("max_tokens", self.max_tokens),
            ("enable_thinking", self.enable_thinking),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
        ):
            if value is not None:
                body[key] = value
        return body


class _Recorder:
    def __init__(self, config: DirectRequestConfig) -> None:
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


def _iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        yield "\n".join(data_lines)


class _ResponseAccumulator:
    def __init__(self, recorder: _Recorder, attempt: int = 1) -> None:
        self.recorder = recorder
        self.attempt = attempt
        self.reasoning_parts: list[str] = []
        self.answer_parts: list[str] = []
        self.usage: dict[str, int] = {}
        self.tool_calls: dict[str, dict[str, Any]] = {}
        self.timestamps_ns: dict[str, int] = {}
        self.chunk_count = 0
        self.finish_reason: str | None = None
        self._tool_index_keys: dict[str, str] = {}

    def _first_text(self, kind: str, value: str) -> None:
        key = f"first_{kind}"
        if key in self.timestamps_ns:
            return
        timestamp = self.recorder.record(
            "FirstReasoning" if kind == "reasoning" else "FirstAnswer",
            {
                "attempt": self.attempt,
                "text_len": len(value),
                "text_sha256": _sha256_text(value),
            },
        )
        self.timestamps_ns[key] = timestamp

    def consume_stream_payload(self, payload: str) -> None:
        if payload.strip() == "[DONE]":
            self.timestamps_ns["stream_done"] = self.recorder.record(
                "StreamDone",
                {"attempt": self.attempt},
            )
            return
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            self.recorder.record(
                "MalformedSSE",
                {
                    "payload_len": len(payload),
                    "payload_sha256": _sha256_text(payload),
                },
            )
            return
        if not isinstance(decoded, dict):
            return
        self.chunk_count += 1
        usage = _safe_usage(decoded.get("usage"))
        if usage:
            self.usage = usage
            self.recorder.record("Usage", {"attempt": self.attempt, **usage})
        choices = decoded.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                self.finish_reason = finish
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._consume_delta(delta)

    def consume_completion(self, decoded: dict[str, Any]) -> None:
        usage = _safe_usage(decoded.get("usage"))
        if usage:
            self.usage = usage
            self.recorder.record("Usage", {"attempt": self.attempt, **usage})
        choices = decoded.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                self.finish_reason = finish
            message = choice.get("message")
            if isinstance(message, dict):
                self._consume_delta(message)

    def _consume_delta(self, delta: dict[str, Any]) -> None:
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self._first_text("reasoning", reasoning)
            self.reasoning_parts.append(reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._first_text("answer", content)
            self.answer_parts.append(content)
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            return
        for position, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            index = str(call.get("index", position))
            call_id = str(call.get("id") or "")
            if call_id:
                self._tool_index_keys[index] = call_id
            key = self._tool_index_keys.get(index) or call_id or index
            current = self.tool_calls.setdefault(
                key,
                {
                    "tool_call_id": call_id or key,
                    "tool_call_id_sha256": (
                        _sha256_text(call_id) if call_id else None
                    ),
                    "name": "",
                    "arguments": "",
                },
            )
            function = call.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    current["name"] += str(function["name"])
                if function.get("arguments"):
                    current["arguments"] += str(function["arguments"])
            if "first_tool_call" not in self.timestamps_ns:
                timestamp = self.recorder.record(
                    "FirstToolCall",
                    {"attempt": self.attempt, "tool_call_count": len(calls)},
                )
                self.timestamps_ns["first_tool_call"] = timestamp

    def sanitized_tool_calls(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for call in self.tool_calls.values():
            name = call["name"]
            arguments = call["arguments"]
            result.append(
                {
                    "tool_call_id_sha256": call["tool_call_id_sha256"],
                    "name": name,
                    "name_len": len(name),
                    "name_sha256": _sha256_text(name) if name else None,
                    "arguments_len": len(arguments),
                    "arguments_sha256": (
                        _sha256_text(arguments) if arguments else None
                    ),
                },
            )
        return result

    def raw_tool_calls(self) -> list[dict[str, str]]:
        return [
            {
                "id": str(call["tool_call_id"]),
                "name": str(call["name"]),
                "arguments": str(call["arguments"]),
            }
            for call in self.tool_calls.values()
        ]


def _latency_ms(start: int, end: int | None) -> float | None:
    if end is None:
        return None
    return (end - start) / 1_000_000


@dataclass
class _AttemptResult:
    attempt: int
    status_code: int | None
    error_type: str | None
    accumulator: _ResponseAccumulator
    request_start_ns: int
    response_headers_ns: int | None
    request_end_ns: int


def _run_attempt(
    *,
    client: httpx.Client,
    cfg: DirectRequestConfig,
    body: dict[str, Any],
    headers: dict[str, str],
    recorder: _Recorder,
    attempt: int,
) -> _AttemptResult:
    accumulator = _ResponseAccumulator(recorder, attempt=attempt)
    status_code: int | None = None
    error_type: str | None = None
    summary = _request_summary(body)
    request_start = recorder.record(
        "ModelRequestStart",
        {"attempt": attempt, **summary},
    )
    response_headers_ns: int | None = None
    try:
        with client.stream(
            "POST",
            _endpoint(cfg.base_url),
            headers=headers,
            json=body,
        ) as response:
            status_code = response.status_code
            response_headers_ns = recorder.record(
                "ResponseHeaders",
                {
                    "attempt": attempt,
                    "status_code": status_code,
                    "is_sse": "text/event-stream"
                    in response.headers.get("content-type", "").lower(),
                },
            )
            if response.is_error:
                raw_error = response.read()
                error_type = f"http_{status_code}"
                recorder.record(
                    "Error",
                    {
                        "attempt": attempt,
                        "error_type": error_type,
                        "response_bytes": len(raw_error),
                        "response_sha256": hashlib.sha256(raw_error).hexdigest(),
                    },
                )
            elif "text/event-stream" in response.headers.get(
                "content-type",
                "",
            ).lower():
                for payload in _iter_sse_data(response.iter_lines()):
                    accumulator.consume_stream_payload(payload)
            else:
                raw_response = response.read()
                try:
                    decoded = json.loads(raw_response)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, dict):
                    accumulator.consume_completion(decoded)
                else:
                    error_type = "invalid_json_response"
                    recorder.record(
                        "Error",
                        {
                            "attempt": attempt,
                            "error_type": error_type,
                            "response_bytes": len(raw_response),
                            "response_sha256": hashlib.sha256(
                                raw_response,
                            ).hexdigest(),
                        },
                    )
    except (httpx.HTTPError, OSError, ValueError) as exc:
        error_type = type(exc).__name__
        recorder.record(
            "Error",
            {"attempt": attempt, "error_type": error_type},
        )
    request_end = recorder.record(
        "ModelRequestEnd",
        {
            "attempt": attempt,
            "status_code": status_code,
            "success": (
                error_type is None
                and status_code is not None
                and status_code < 400
            ),
        },
    )
    return _AttemptResult(
        attempt=attempt,
        status_code=status_code,
        error_type=error_type,
        accumulator=accumulator,
        request_start_ns=request_start,
        response_headers_ns=response_headers_ns,
        request_end_ns=request_end,
    )


def _validate_exact_shell_call(
    call: dict[str, str],
    expected_command: str,
) -> dict[str, str] | None:
    if expected_command != SHELL_COMMAND:
        return None
    if call.get("name") != "execute_shell_command":
        return None
    arguments = call.get("arguments", "")
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict) or set(decoded) != {"command"}:
        return None
    command = decoded.get("command")
    if not isinstance(command, str) or not hmac.compare_digest(
        command,
        expected_command,
    ):
        return None
    return call


def _execute_exact_printf(
    cfg: DirectRequestConfig,
    recorder: _Recorder,
) -> tuple[str | None, str | None]:
    executable = shutil.which("printf")
    if executable is None:
        return None, "printf_not_found"
    recorder.record(
        "ToolExecutionStart",
        {
            "tool_name": "execute_shell_command",
            "command_len": len(cfg.expected_command),
            "command_sha256": _sha256_text(cfg.expected_command),
        },
    )
    try:
        completed = subprocess.run(
            [executable, "QWENPAW_BENCH_42"],
            check=False,
            capture_output=True,
            text=True,
            timeout=cfg.tool_timeout_s,
            shell=False,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        error_type = type(exc).__name__
        recorder.record("ToolExecutionEnd", {"error_type": error_type})
        return None, error_type
    output = completed.stdout
    recorder.record(
        "ToolExecutionEnd",
        {
            "returncode": completed.returncode,
            "stdout_len": len(output),
            "stdout_sha256": _sha256_text(output),
            "stderr_len": len(completed.stderr),
            "stderr_sha256": (
                _sha256_text(completed.stderr) if completed.stderr else None
            ),
        },
    )
    if completed.returncode != 0 or output != "QWENPAW_BENCH_42":
        return None, "unexpected_printf_result"
    return output, None


def _second_round_body(
    body: dict[str, Any],
    call: dict[str, str],
    accumulator: _ResponseAccumulator,
    tool_output: str,
) -> dict[str, Any]:
    result = copy.deepcopy(body)
    messages = result.get("messages")
    if not isinstance(messages, list):
        messages = []
        result["messages"] = messages
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            },
        ],
    }
    reasoning = "".join(accumulator.reasoning_parts)
    if reasoning:
        assistant["reasoning_content"] = reasoning
    messages.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_output,
            },
        ],
    )
    return result


def _sum_usage(attempts: list[_AttemptResult]) -> dict[str, int]:
    result: dict[str, int] = {}
    for attempt in attempts:
        for key, value in attempt.accumulator.usage.items():
            result[key] = result.get(key, 0) + value
    return result


def run_request(
    config: DirectRequestConfig | Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one request and return raw timings plus provider usage.

    A captured ``request_body`` is replayed without adding or changing model
    parameters. When no body is supplied, a minimal user-only streaming body
    is constructed from ``model`` and ``prompt``.
    """

    cfg = DirectRequestConfig.from_value(config, kwargs)
    if not cfg.base_url:
        raise ValueError("base_url is required")
    body = cfg.build_body()
    summary = _request_summary(body)
    recorder = _Recorder(cfg)
    timestamps: dict[str, int] = {}
    error_type: str | None = None
    tool_execution_count = 0

    headers = dict(cfg.headers)
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("Accept", "text/event-stream")
    if cfg.api_key is not None:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    timestamps["request_start"] = recorder.record("RequestStart", summary)
    attempts: list[_AttemptResult] = []
    timeout = httpx.Timeout(cfg.timeout_s)
    with httpx.Client(timeout=timeout, trust_env=cfg.trust_env) as client:
        first = _run_attempt(
            client=client,
            cfg=cfg,
            body=body,
            headers=headers,
            recorder=recorder,
            attempt=1,
        )
        attempts.append(first)
        error_type = first.error_type
        if cfg.allow_exact_shell and error_type is None:
            raw_calls = first.accumulator.raw_tool_calls()
            call = (
                _validate_exact_shell_call(raw_calls[0], cfg.expected_command)
                if len(raw_calls) == 1
                else None
            )
            if call is None:
                error_type = "unsafe_or_unexpected_tool_call"
                recorder.record("ToolRejected", {"tool_call_count": len(raw_calls)})
            else:
                tool_output, tool_error = _execute_exact_printf(cfg, recorder)
                if tool_error is not None or tool_output is None:
                    error_type = tool_error or "tool_execution_failed"
                else:
                    tool_execution_count = 1
                    second_body = _second_round_body(
                        body,
                        call,
                        first.accumulator,
                        tool_output,
                    )
                    second = _run_attempt(
                        client=client,
                        cfg=cfg,
                        body=second_body,
                        headers=headers,
                        recorder=recorder,
                        attempt=2,
                    )
                    attempts.append(second)
                    error_type = second.error_type
                    if second.accumulator.raw_tool_calls() and error_type is None:
                        error_type = "unexpected_second_tool_call"
                        recorder.record(
                            "ToolRejected",
                            {
                                "attempt": 2,
                                "tool_call_count": len(
                                    second.accumulator.raw_tool_calls(),
                                ),
                            },
                        )

    for attempt in attempts:
        timestamps[f"model_{attempt.attempt}_start"] = attempt.request_start_ns
        timestamps[f"model_{attempt.attempt}_end"] = attempt.request_end_ns
        if attempt.response_headers_ns is not None:
            timestamps[f"model_{attempt.attempt}_headers"] = (
                attempt.response_headers_ns
            )
        for key, value in attempt.accumulator.timestamps_ns.items():
            timestamps.setdefault(key, value)
            timestamps[f"model_{attempt.attempt}_{key}"] = value
    status_code = attempts[-1].status_code if attempts else None
    timestamps["request_end"] = recorder.record(
        "RequestEnd",
        {
            "status_code": status_code,
            "success": error_type is None and status_code is not None and status_code < 400,
        },
    )
    start = timestamps["request_start"]
    success = error_type is None and status_code is not None and status_code < 400
    accumulators = [attempt.accumulator for attempt in attempts]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "profile": cfg.profile,
        "scenario": cfg.scenario,
        "backend": cfg.backend,
        "success": success,
        "status_code": status_code,
        "error_type": error_type,
        "request_summary": summary,
        "timestamps_ns": timestamps,
        "latency_ms": {
            "response_headers": _latency_ms(start, timestamps.get("model_1_headers")),
            "ttft_reasoning": _latency_ms(start, timestamps.get("first_reasoning")),
            "ttft_answer": _latency_ms(start, timestamps.get("first_answer")),
            "first_tool_call": _latency_ms(start, timestamps.get("first_tool_call")),
            "total": _latency_ms(start, timestamps["request_end"]),
        },
        "reasoning_text": "".join(
            part for acc in accumulators for part in acc.reasoning_parts
        ),
        "answer_text": "".join(
            part for acc in accumulators for part in acc.answer_parts
        ),
        "tool_calls": [
            call for acc in accumulators for call in acc.sanitized_tool_calls()
        ],
        "tool_execution_count": tool_execution_count,
        "model_attempts": len(attempts),
        "usage": _sum_usage(attempts),
        "usage_by_attempt": [attempt.accumulator.usage for attempt in attempts],
        "chunk_count": sum(acc.chunk_count for acc in accumulators),
        "finish_reason": (
            accumulators[-1].finish_reason if accumulators else None
        ),
        "events": recorder.events,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with exactly: ready")
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile", default="direct")
    parser.add_argument("--scenario", default="unknown")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-exact-shell",
        action="store_true",
        help=(
            "execute only the fixed benchmark command and perform the second "
            "model round"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    request_body = None
    if args.request_json is not None:
        request_body = json.loads(args.request_json.read_text(encoding="utf-8"))
    result = run_request(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        request_body=request_body,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        run_id=args.run_id or uuid.uuid4().hex,
        profile=args.profile,
        scenario=args.scenario,
        jsonl_path=args.jsonl,
        timeout_s=args.timeout,
        allow_exact_shell=args.allow_exact_shell,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
