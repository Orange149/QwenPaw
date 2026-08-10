"""End-to-end contract test for the persistent ACP benchmark client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from benchmarks.overhead.qwenpaw_overhead.acp_client import (
    ACPBenchmarkClient,
    ACPConfig,
)
from benchmarks.overhead.qwenpaw_overhead.scenarios import SHELL_COMMAND


FIXTURE = Path(__file__).with_name("_fake_benchmark_acp_agent.py")


def test_acp_lifecycle_milestones_and_unique_safe_permission() -> None:
    async def exercise() -> None:
        config = ACPConfig(
            command=[sys.executable, str(FIXTURE)],
            cwd=str(Path.cwd()),
            prompt="QWENPAW_OVERHEAD_S2",
            scenario="S2",
            timeout_s=10.0,
            available_commands_timeout_s=5.0,
            expected_command=SHELL_COMMAND,
        )
        client = ACPBenchmarkClient(config)
        callback_pids: list[int] = []

        def spawned(pid: int) -> None:
            callback_pids.append(pid)
            assert [event["event"] for event in client.recorder.events] == [
                "LaunchStart",
            ]

        try:
            started = await client.start(process_started_callback=spawned)
            assert started.ready is True
            assert started.pid == client.pid == callback_pids[0]
            assert started.available_commands_count == 1
            assert "connected" in started.timestamps_ns
            assert "available_commands" in started.timestamps_ns

            turn = await client.prompt()
            result = client.result()

            assert turn["stop_reason"] == "end_turn"
            assert turn["reasoning_text"] == "fixture reasoning"
            assert turn["answer_text"] == "QWENPAW_BENCH_42"
            assert turn["usage"] == {
                "inputTokens": 101,
                "outputTokens": 7,
                "totalTokens": 108,
            }
            assert [item["approved"] for item in turn["permission_requests"]] == [
                True,
                False,
            ]
            assert result.success is True
            assert result.usage == turn["usage"]
            events = [event["event"] for event in result.events]
            for expected in (
                "LaunchStart",
                "ProcessStarted",
                "Initialize",
                "Connected",
                "AvailableCommands",
                "FirstReasoning",
                "FirstAnswer",
                "FirstToolCall",
                "TurnEnd",
            ):
                assert expected in events
        finally:
            await client.close()

    asyncio.run(exercise())


def test_acp_cpu_affinity_fails_explicitly_without_taskset(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.overhead.qwenpaw_overhead.acp_client.shutil.which",
        lambda _name: None,
    )
    config = ACPConfig(command=[sys.executable, str(FIXTURE)], cpu_affinity="0-3")

    try:
        config.spawn_command()
    except RuntimeError as exc:
        assert "taskset" in str(exc)
    else:
        raise AssertionError("missing taskset must be an explicit failure")
