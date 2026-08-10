"""Contract tests for overhead benchmark records."""

from __future__ import annotations

from benchmarks.overhead.qwenpaw_overhead.schemas import (
    SCHEMA_VERSION,
    CoreThresholds,
    EventRecord,
    StateLayer,
    SummaryStats,
)


def test_event_record_matches_common_jsonl_envelope() -> None:
    event = EventRecord(
        run_id="run-1",
        profile="minimal",
        scenario="idle",
        backend="mock",
        event="resource_sample",
        monotonic_ns=123,
        wall_time_utc="2026-08-06T00:00:00Z",
        data={"layer": StateLayer.CACHE, "values": (1, 2)},
    )

    assert event.to_dict() == {
        "run_id": "run-1",
        "profile": "minimal",
        "scenario": "idle",
        "backend": "mock",
        "event": "resource_sample",
        "monotonic_ns": 123,
        "wall_time_utc": "2026-08-06T00:00:00Z",
        "data": {"layer": "cache", "values": [1, 2]},
        "schema_version": SCHEMA_VERSION,
    }


def test_summary_stats_serializes_range_with_csv_name() -> None:
    stats = SummaryStats(3, 2.0, 1.0, 3.0, 2.0)

    data = stats.to_dict()

    assert data["range"] == 2.0
    assert "value_range" not in data


def test_core_threshold_defaults_are_protocol_values() -> None:
    thresholds = CoreThresholds()

    assert thresholds.package_pass_bytes == 512 * 1024 * 1024
    assert thresholds.package_fail_bytes == 1024 * 1024 * 1024
    assert thresholds.idle_pss_pass_bytes == 256 * 1024 * 1024
    assert thresholds.idle_pss_fail_bytes == 512 * 1024 * 1024
    assert thresholds.framework_context_pass_tokens == 4_000
    assert thresholds.framework_context_fail_tokens == 8_000
