from __future__ import annotations

from benchmarks.overhead.qwenpaw_overhead.preflight import (
    Check,
    preflight_ok,
    run_preflight,
)


def test_preflight_reports_missing_inputs_without_writing(tmp_path) -> None:
    checks = run_preflight(
        python_executable=tmp_path / "python",
        working_dir=tmp_path / "working",
        secret_dir=tmp_path / "secret",
    )
    by_name = {check.name: check for check in checks}
    assert by_name["python"].status == "fail"
    assert by_name["seed_working_dir"].status == "fail"
    assert not preflight_ok(checks)
    assert list(tmp_path.iterdir()) == []


def test_preflight_ok_allows_warnings() -> None:
    assert preflight_ok([Check("x", "ok", ""), Check("y", "warn", "")])

