"""Tests for relay streaming, privacy, and hard request budgets."""

from __future__ import annotations

import json

import httpx

from benchmarks.overhead.qwenpaw_overhead.mock_server import start_server as mock
from benchmarks.overhead.qwenpaw_overhead.relay import start_server as relay
from benchmarks.overhead.qwenpaw_overhead.scenarios import S1


def test_relay_forwards_first_sse_chunk_without_buffering_and_blocks_budget() -> None:
    api_key = "relay-test-secret-key"
    body = {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": S1.prompts[0]}],
        "stream": True,
    }
    relay_handle = None
    with mock(scenario="S1", pause_after_first_chunk=True) as upstream:
        with relay(
            upstream.base_url,
            upstream_api_key=api_key,
            max_requests=1,
        ) as proxy:
            relay_handle = proxy
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                with client.stream(
                    "POST",
                    proxy.base_url + "/chat/completions",
                    json=body,
                ) as response:
                    assert response.status_code == 200
                    lines = response.iter_lines()
                    first_line = next(lines)
                    assert first_line.startswith("data:")
                    assert upstream.wait_for_first_chunk(1.0)
                    # The upstream is still gated, so receiving this line proves
                    # the relay did not wait for the complete response body.
                    upstream.release_stream()
                    list(lines)

                blocked = client.post(
                    proxy.base_url + "/chat/completions",
                    json={**body, "stream": False},
                )

            assert blocked.status_code == 429
            assert upstream.request_count == 1
            assert proxy.request_count == 2
            assert any(
                event["event"] == "relay_budget_blocked"
                for event in proxy.snapshot_events()
            )
            serialized = json.dumps(proxy.snapshot_events())
            assert api_key not in serialized
            assert S1.prompts[0] not in serialized

    assert relay_handle is not None
    assert relay_handle.request_bodies == {}


def test_relay_rejects_negative_request_budget() -> None:
    try:
        with relay("http://127.0.0.1:9/v1", max_requests=-1):
            raise AssertionError("context should not start")
    except ValueError as exc:
        assert "max_requests" in str(exc)
