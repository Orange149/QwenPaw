"""Non-buffering, privacy-preserving OpenAI-compatible HTTP relay.

The relay exists to timestamp the exact wire request QwenPaw sends while
keeping the provider credential and synthetic prompt out of benchmark logs.
It forwards request/response bytes, but persists only structural counts,
hashes, numeric usage, and monotonic timestamps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import SplitResult, urlsplit


_SCHEMA_VERSION = "1"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_KNOWN_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encoded_size(value: Any) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
    except (TypeError, ValueError):
        return 0


def _message_content_size(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return _encoded_size(content) if content is not None else 0


def summarize_request(raw: bytes) -> dict[str, Any]:
    """Summarize an OpenAI request without returning content or credentials."""

    try:
        decoded = json.loads(raw or b"{}")
        body = decoded if isinstance(decoded, dict) else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "body_bytes": len(raw),
            "body_sha256": _sha256_bytes(raw),
            "valid_json": False,
        }

    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    roles: dict[str, int] = {}
    content_bytes = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
        content_bytes += _message_content_size(message)

    tools = body.get("tools")
    tools = tools if isinstance(tools, list) else []
    model = str(body.get("model") or "")
    result: dict[str, Any] = {
        "body_bytes": len(raw),
        "body_sha256": _sha256_bytes(raw),
        "valid_json": True,
        "message_count": len(messages),
        "message_roles": roles,
        "message_content_bytes": content_bytes,
        "tool_count": len(tools),
        "tool_schema_bytes": _encoded_size(tools),
        "model_len": len(model),
        "model_sha256": _sha256_text(model) if model else None,
        "stream": body.get("stream") is True,
    }
    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "enable_thinking",
        "thinking_budget",
        "parallel_tool_calls",
    ):
        value = body.get(key)
        if isinstance(value, (bool, int, float)) or value is None:
            result[key] = value
    return result


def sanitize_usage(value: Any) -> dict[str, int]:
    """Keep only numeric, provider-reported usage counters."""

    if not isinstance(value, dict):
        return {}
    usage: dict[str, int] = {}
    for key in _KNOWN_USAGE_KEYS:
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            usage[key] = raw
    details = value.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            usage["cached_tokens"] = cached
    return usage


@dataclass
class _RelayState:
    upstream: SplitResult
    run_id: str
    profile: str
    scenario: str
    backend: str
    timeout_s: float
    verify_tls: bool
    upstream_api_key: str | None
    upstream_headers: dict[str, str]
    local_prefix: str
    jsonl_path: Path | None
    max_requests: int | None
    events: list[dict[str, Any]] = field(default_factory=list)
    event_lock: threading.Lock = field(default_factory=threading.Lock)
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    request_count: int = 0
    request_bodies: dict[int, dict[str, Any]] = field(default_factory=dict)

    def record(self, event: str, data: dict[str, Any] | None = None) -> int:
        monotonic_ns = time.monotonic_ns()
        row = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile": self.profile,
            "scenario": self.scenario,
            "backend": self.backend,
            "event": event,
            "monotonic_ns": monotonic_ns,
            "wall_time_utc": _utc_now(),
            "data": data or {},
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self.event_lock:
            self.events.append(row)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded + "\n")
        return monotonic_ns

    def next_request_index(self) -> int:
        with self.request_lock:
            self.request_count += 1
            return self.request_count

    def retain_request_body(self, request_index: int, raw: bytes) -> None:
        """Retain parsed wire JSON in memory only for exact replay."""

        try:
            decoded = json.loads(raw or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(decoded, dict):
            return
        with self.request_lock:
            self.request_bodies[request_index] = copy.deepcopy(decoded)

    def upstream_path(self, incoming_path: str) -> str:
        incoming = urlsplit(incoming_path)
        path = incoming.path
        prefix = self.local_prefix.rstrip("/")
        if prefix and (path == prefix or path.startswith(prefix + "/")):
            path = path[len(prefix) :]
        combined = self.upstream.path.rstrip("/") + "/" + path.lstrip("/")
        query_parts = [part for part in (self.upstream.query, incoming.query) if part]
        if query_parts:
            combined += "?" + "&".join(query_parts)
        return combined


class _SSETracker:
    """Parse SSE event boundaries while bytes continue downstream."""

    def __init__(self, state: _RelayState, request_index: int) -> None:
        self.state = state
        self.request_index = request_index
        self.data_lines: list[str] = []
        self.first_reasoning = False
        self.first_answer = False
        self.first_tool = False
        self.chunk_count = 0

    def feed(self, raw_line: bytes) -> None:
        try:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        except Exception:
            return
        if not line:
            self._process_event()
            return
        if line.startswith("data:"):
            self.data_lines.append(line[5:].lstrip(" "))

    def finish(self) -> None:
        self._process_event()

    def _process_event(self) -> None:
        if not self.data_lines:
            return
        payload = "\n".join(self.data_lines)
        self.data_lines.clear()
        if payload.strip() == "[DONE]":
            self.state.record(
                "relay_upstream_done",
                {
                    "request_index": self.request_index,
                    "chunk_count": self.chunk_count,
                },
            )
            return
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            self.state.record(
                "relay_non_json_sse",
                {
                    "request_index": self.request_index,
                    "payload_bytes": len(payload.encode("utf-8")),
                    "payload_sha256": _sha256_text(payload),
                },
            )
            return
        if not isinstance(decoded, dict):
            return
        self.chunk_count += 1
        usage = sanitize_usage(decoded.get("usage"))
        if usage:
            self.state.record(
                "relay_usage",
                {"request_index": self.request_index, **usage},
            )

        choices = decoded.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning and not self.first_reasoning:
                self.first_reasoning = True
                self.state.record(
                    "relay_first_reasoning",
                    {
                        "request_index": self.request_index,
                        "text_len": len(reasoning),
                        "text_sha256": _sha256_text(reasoning),
                    },
                )
            content = delta.get("content")
            if isinstance(content, str) and content and not self.first_answer:
                self.first_answer = True
                self.state.record(
                    "relay_first_answer",
                    {
                        "request_index": self.request_index,
                        "text_len": len(content),
                        "text_sha256": _sha256_text(content),
                    },
                )
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                self._record_tool_calls(tool_calls)

    def _record_tool_calls(self, calls: list[Any]) -> None:
        if not self.first_tool:
            self.first_tool = True
            self.state.record(
                "relay_first_tool_call",
                {
                    "request_index": self.request_index,
                    "tool_delta_count": len(calls),
                },
            )
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            function = function if isinstance(function, dict) else {}
            name = str(function.get("name") or "")
            arguments = str(function.get("arguments") or "")
            if not name and not arguments:
                continue
            self.state.record(
                "relay_tool_delta",
                {
                    "request_index": self.request_index,
                    "tool_name_len": len(name),
                    "tool_name_sha256": _sha256_text(name) if name else None,
                    "arguments_len": len(arguments),
                    "arguments_sha256": (
                        _sha256_text(arguments) if arguments else None
                    ),
                },
            )


class _RelayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: _RelayState) -> None:
        super().__init__(address, _RelayHandler)
        self.state = state


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _RelayHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy()

    def _read_request_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _upstream_connection(self) -> http.client.HTTPConnection:
        state = self.server.state
        host = state.upstream.hostname
        if host is None:
            raise ValueError("upstream URL has no hostname")
        if state.upstream.scheme == "https":
            context = (
                ssl.create_default_context()
                if state.verify_tls
                else ssl._create_unverified_context()  # noqa: SLF001
            )
            return http.client.HTTPSConnection(
                host,
                state.upstream.port or 443,
                timeout=state.timeout_s,
                context=context,
            )
        return http.client.HTTPConnection(
            host,
            state.upstream.port or 80,
            timeout=state.timeout_s,
        )

    def _request_headers(self, raw: bytes) -> dict[str, str]:
        state = self.server.state
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            lowered = name.lower()
            if lowered in _HOP_BY_HOP_HEADERS or lowered in {
                "host",
                "content-length",
                "accept-encoding",
            }:
                continue
            headers[name] = value
        # Compression would prevent safe incremental SSE inspection.
        headers["Accept-Encoding"] = "identity"
        if raw:
            headers["Content-Length"] = str(len(raw))
        if state.upstream_api_key is not None:
            headers["Authorization"] = f"Bearer {state.upstream_api_key}"
        headers.update(state.upstream_headers)
        return headers

    def _proxy(self) -> None:
        state = self.server.state
        request_index = state.next_request_index()
        raw = self._read_request_body()
        state.retain_request_body(request_index, raw)
        summary = summarize_request(raw)
        summary.update(
            {
                "request_index": request_index,
                "method": self.command,
            },
        )
        state.record("relay_request_received", summary)
        if state.max_requests is not None and request_index > state.max_requests:
            state.record(
                "relay_budget_blocked",
                {
                    "request_index": request_index,
                    "max_requests": state.max_requests,
                },
            )
            self._send_budget_error()
            return

        connection: http.client.HTTPConnection | None = None
        try:
            connection = self._upstream_connection()
            state.record(
                "relay_upstream_request_start",
                {"request_index": request_index},
            )
            connection.request(
                self.command,
                state.upstream_path(self.path),
                body=raw if raw else None,
                headers=self._request_headers(raw),
            )
            upstream = connection.getresponse()
            content_type = upstream.getheader("Content-Type", "")
            state.record(
                "relay_upstream_headers",
                {
                    "request_index": request_index,
                    "status_code": upstream.status,
                    "is_sse": "text/event-stream" in content_type.lower(),
                },
            )
            self._send_response_headers(upstream)
            if self.command == "HEAD":
                return
            if "text/event-stream" in content_type.lower():
                self._stream_sse(upstream, request_index)
            else:
                self._stream_bytes(upstream, request_index)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            state.record(
                "relay_error",
                {
                    "request_index": request_index,
                    "error_type": type(exc).__name__,
                },
            )
            if not self.wfile.closed:
                try:
                    self._send_safe_gateway_error()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            if connection is not None:
                connection.close()

    def _send_response_headers(self, upstream: http.client.HTTPResponse) -> None:
        self.send_response(upstream.status, upstream.reason)
        for name, value in upstream.getheaders():
            lowered = name.lower()
            if lowered in _HOP_BY_HOP_HEADERS or lowered in {
                "content-length",
                "content-encoding",
            }:
                continue
            self.send_header(name, value)
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _stream_sse(
        self,
        upstream: http.client.HTTPResponse,
        request_index: int,
    ) -> None:
        state = self.server.state
        tracker = _SSETracker(state, request_index)
        downstream_bytes = 0
        try:
            while True:
                line = upstream.readline()
                if not line:
                    break
                tracker.feed(line)
                self.wfile.write(line)
                self.wfile.flush()
                downstream_bytes += len(line)
            tracker.finish()
            state.record(
                "relay_response_complete",
                {
                    "request_index": request_index,
                    "downstream_bytes": downstream_bytes,
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            state.record(
                "relay_client_disconnected",
                {
                    "request_index": request_index,
                    "downstream_bytes": downstream_bytes,
                },
            )

    def _stream_bytes(
        self,
        upstream: http.client.HTTPResponse,
        request_index: int,
    ) -> None:
        downstream_bytes = 0
        try:
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                downstream_bytes += len(chunk)
            self.server.state.record(
                "relay_response_complete",
                {
                    "request_index": request_index,
                    "downstream_bytes": downstream_bytes,
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            self.server.state.record(
                "relay_client_disconnected",
                {
                    "request_index": request_index,
                    "downstream_bytes": downstream_bytes,
                },
            )

    def _send_safe_gateway_error(self) -> None:
        payload = b'{"error":{"type":"relay_upstream_error"}}'
        self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_budget_error(self) -> None:
        payload = b'{"error":{"type":"relay_request_budget_exhausted"}}'
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Retry-After", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)
        self.wfile.flush()

    def log_message(self, _format: str, *_args: Any) -> None:
        return


@dataclass
class RelayHandle:
    """Handle yielded by :func:`start_server`."""

    base_url: str
    events: list[dict[str, Any]]
    _server: _RelayHTTPServer
    _thread: threading.Thread

    @property
    def request_count(self) -> int:
        return self._server.state.request_count

    def snapshot_events(self) -> list[dict[str, Any]]:
        with self._server.state.event_lock:
            return list(self.events)

    @property
    def request_bodies(self) -> dict[int, dict[str, Any]]:
        """Parsed bodies captured in memory; never written to event JSONL."""

        with self._server.state.request_lock:
            return copy.deepcopy(self._server.state.request_bodies)

    def request_body(self, request_index: int) -> dict[str, Any] | None:
        with self._server.state.request_lock:
            body = self._server.state.request_bodies.get(request_index)
            return copy.deepcopy(body) if body is not None else None

    def last_request_body(self) -> dict[str, Any] | None:
        with self._server.state.request_lock:
            if not self._server.state.request_bodies:
                return None
            request_index = max(self._server.state.request_bodies)
            return copy.deepcopy(self._server.state.request_bodies[request_index])

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        with self._server.state.request_lock:
            self._server.state.request_bodies.clear()


@contextmanager
def start_server(
    upstream_base_url: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    run_id: str = "relay-run",
    profile: str = "relay",
    scenario: str = "unknown",
    backend: str = "relay",
    timeout_s: float = 120.0,
    verify_tls: bool = True,
    upstream_api_key: str | None = None,
    upstream_headers: dict[str, str] | None = None,
    local_prefix: str = "/v1",
    jsonl_path: str | Path | None = None,
    max_requests: int | None = None,
) -> Iterator[RelayHandle]:
    """Start the relay and yield its local OpenAI-compatible base URL."""

    upstream = urlsplit(upstream_base_url)
    if upstream.scheme not in {"http", "https"} or not upstream.hostname:
        raise ValueError("upstream_base_url must be an http(s) URL with a host")
    if upstream.username is not None or upstream.password is not None:
        raise ValueError("credentials must not be embedded in upstream URL")
    if port < 0 or port > 65535:
        raise ValueError("port must be in the range 0..65535")
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests cannot be negative")

    state = _RelayState(
        upstream=upstream,
        run_id=run_id,
        profile=profile,
        scenario=scenario,
        backend=backend,
        timeout_s=max(0.1, timeout_s),
        verify_tls=verify_tls,
        upstream_api_key=upstream_api_key,
        upstream_headers=dict(upstream_headers or {}),
        local_prefix="/" + local_prefix.strip("/") if local_prefix else "",
        jsonl_path=Path(jsonl_path) if jsonl_path is not None else None,
        max_requests=max_requests,
    )
    server = _RelayHTTPServer((host, port), state)
    thread = threading.Thread(
        target=server.serve_forever,
        name="qwenpaw-overhead-relay",
        daemon=True,
    )
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    base_url = f"http://{bound_host}:{bound_port}{state.local_prefix}"
    handle = RelayHandle(base_url, state.events, server, thread)
    state.record(
        "relay_server_started",
        {
            "host": str(bound_host),
            "port": int(bound_port),
            "upstream_host": upstream.hostname,
            "upstream_port": upstream.port or (443 if upstream.scheme == "https" else 80),
        },
    )
    try:
        yield handle
    finally:
        state.record("relay_server_stopping")
        handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--run-id", default="relay-cli")
    parser.add_argument("--profile", default="relay")
    parser.add_argument("--scenario", default="unknown")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable containing the upstream key; never logged",
    )
    parser.add_argument("--insecure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    with start_server(
        args.upstream_base_url,
        host=args.host,
        port=args.port,
        run_id=args.run_id,
        profile=args.profile,
        scenario=args.scenario,
        jsonl_path=args.jsonl,
        verify_tls=not args.insecure,
        upstream_api_key=key,
        max_requests=args.max_requests,
    ) as relay:
        print(relay.base_url, flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
