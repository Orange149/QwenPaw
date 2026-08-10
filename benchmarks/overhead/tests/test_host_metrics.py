from __future__ import annotations

from types import SimpleNamespace

from benchmarks.overhead.qwenpaw_overhead import host_metrics


def test_vmmemwsl_query_returns_only_aggregate_fields(monkeypatch) -> None:
    monkeypatch.setattr(host_metrics, "_is_wsl", lambda: True)
    monkeypatch.setattr(host_metrics.shutil, "which", lambda _name: "powershell.exe")
    monkeypatch.setattr(
        host_metrics.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"process_count":1,"working_set_bytes":123456}',
        ),
    )

    result = host_metrics.sample_vmmemwsl()

    assert result["available"] is True
    assert result["process_count"] == 1
    assert result["working_set_bytes"] == 123456
    assert set(result) == {
        "available",
        "source",
        "observed_at_utc",
        "process_count",
        "working_set_bytes",
    }


def test_vmmemwsl_delta_is_signed_and_requires_two_valid_snapshots() -> None:
    before = {"available": True, "working_set_bytes": 200}
    after = {"available": True, "working_set_bytes": 150}

    result = host_metrics.vmmemwsl_delta(before, after)

    assert result["available"] is True
    assert result["working_set_delta_bytes"] == -50
    unavailable = host_metrics.vmmemwsl_delta(
        before,
        {"available": False},
    )
    assert unavailable["working_set_delta_bytes"] is None
