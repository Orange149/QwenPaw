"""Offline contracts for the benchmark CLI and high-level runner helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.overhead.qwenpaw_overhead.artifacts import write_csv
from benchmarks.overhead.qwenpaw_overhead.cli import _parser, _settings, main
from benchmarks.overhead.qwenpaw_overhead.report import evaluate_core_gates
from benchmarks.overhead.qwenpaw_overhead.runner import (
    HarnessRunner,
    HarnessSettings,
    _build_trim_decisions,
    _correctness,
    _extract_core_metrics,
    _quality_status,
)
from benchmarks.overhead.qwenpaw_overhead.scenarios import S2, S3
from benchmarks.overhead.qwenpaw_overhead.schemas import (
    DecisionStatus,
    DistributionRecord,
    FootprintRecord,
    GateStatus,
    StateLayer,
)


@pytest.mark.parametrize(
    ("flags", "authorized"),
    [
        ([], False),
        (["--allow-remote-api"], False),
        (["--confirm-free-quota-stop"], False),
        (["--allow-remote-api", "--confirm-free-quota-stop"], True),
    ],
)
def test_dashscope_cli_requires_both_remote_confirmations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    authorized: bool,
) -> None:
    """The two paid-API acknowledgements must survive argparse and be ANDed."""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-placeholder")
    args = _parser().parse_args(["run", "--phase", "dashscope", *flags])
    runner = HarnessRunner(_settings(args, tmp_path / "result"))

    assert runner.settings.allow_remote_api is (
        "--allow-remote-api" in flags
    )
    assert runner.settings.confirm_free_quota_stop is (
        "--confirm-free-quota-stop" in flags
    )
    if authorized:
        runner._validate_remote_authorization()
    else:
        with pytest.raises(RuntimeError, match="pass both"):
            runner._validate_remote_authorization()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com.evil.example/v1",
        "https://aliyuncs.com/v1",
        "https://127.0.0.1/v1",
    ],
)
def test_remote_authorization_rejects_non_dashscope_https_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-placeholder")
    runner = HarnessRunner(
        HarnessSettings(
            run_id="host-guard",
            result_dir=tmp_path / "host-guard",
            allow_remote_api=True,
            confirm_free_quota_stop=True,
            dashscope_base_url=endpoint,
        ),
    )

    with pytest.raises(RuntimeError, match="HTTPS DashScope"):
        runner._validate_remote_authorization()


def test_remote_authorization_requires_key_but_does_not_contact_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    runner = HarnessRunner(
        HarnessSettings(
            run_id="key-guard",
            result_dir=tmp_path / "key-guard",
            allow_remote_api=True,
            confirm_free_quota_stop=True,
            dashscope_base_url=(
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY is required"):
        runner._validate_remote_authorization()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-placeholder")
    runner._validate_remote_authorization()


def test_s2_correctness_requires_one_exact_approved_shell_call() -> None:
    expected_hash = hashlib.sha256(S2.expected_command.encode("utf-8")).hexdigest()
    valid = {
        "answer_text": S2.expected_text,
        "tool_calls": [{"command_sha256": expected_hash}],
        "permission_requests": [{"approved": True}],
    }
    wire_bodies = {
        2: {
            "messages": [
                {"role": "tool", "content": S2.expected_text},
            ],
        },
    }

    assert _correctness(S2, [valid], wire_bodies=wire_bodies) is True
    assert _correctness(S2, [valid]) is False

    wrong_command = copy.deepcopy(valid)
    wrong_command["tool_calls"][0]["command_sha256"] = "0" * 64
    assert _correctness(S2, [wrong_command], wire_bodies=wire_bodies) is False

    extra_permission = copy.deepcopy(valid)
    extra_permission["permission_requests"].append({"approved": False})
    assert _correctness(
        S2,
        [extra_permission],
        wire_bodies=wire_bodies,
    ) is False


def test_s3_correctness_checks_all_ten_ordered_answers() -> None:
    answers = ["STORED."] + [
        f"K{turn:02d}=V{turn * 7919 % 100000:05d};STORED."
        for turn in range(1, 10)
    ]
    turns = [{"answer_text": answer} for answer in answers]

    assert len(turns) == len(S3.prompts) == 10
    assert _correctness(S3, turns) is True
    assert _correctness(S3, turns[:-1]) is False

    corrupted = copy.deepcopy(turns)
    corrupted[5]["answer_text"] = "K05=WRONG;STORED."
    assert _correctness(S3, corrupted) is False


def test_current_venv_does_not_fill_future_minimal_package_gate() -> None:
    current_install = FootprintRecord(
        category="install.venv_total",
        path="/synthetic/current-venv",
        layer=StateLayer.INSTALL,
        present=True,
        rollup=True,
        allocated_bytes=1008 * 1024 * 1024,
    )

    metrics = _extract_core_metrics([], [current_install], [])
    package_gate = next(
        gate
        for gate in evaluate_core_gates(metrics)
        if gate.metric == "package_bytes"
    )

    assert metrics["package_bytes"] is None
    assert package_gate.value is None
    assert package_gate.status is GateStatus.UNKNOWN


def test_qwenpaw_distribution_is_not_nominated_for_self_deletion() -> None:
    distribution = DistributionRecord(
        name="QwenPaw",
        version="2.0.1",
        location="/synthetic/site-packages",
        allocated_bytes=64 * 1024 * 1024,
        apparent_bytes=60 * 1024 * 1024,
        file_count=100,
    )

    decisions = _build_trim_decisions([], [], [distribution])

    assert len(decisions) == 1
    assert decisions[0].component == "distribution:QwenPaw"
    assert decisions[0].status is DecisionStatus.KEEP


def test_remote_budget_skip_does_not_count_as_a_real_model_trial() -> None:
    rows = []
    for profile in ("full", "core"):
        for scenario in ("S1", "S2"):
            for _ in range(3):
                rows.append(
                    {
                        "record_type": "measurement",
                        "profile": profile,
                        "scenario": scenario,
                        "backend": "mock",
                        "success": True,
                        "metrics": {"orphan_free": True},
                    },
                )
            rows.extend(
                [
                    {
                        "record_type": "measurement",
                        "profile": profile,
                        "scenario": scenario,
                        "backend": "dashscope",
                        "success": True,
                        "metrics": {"orphan_free": True},
                    },
                    {
                        "record_type": "measurement",
                        "profile": profile,
                        "scenario": scenario,
                        "backend": "dashscope",
                        "success": True,
                        "metrics": {"orphan_free": True},
                    },
                    {
                        "record_type": "measurement",
                        "profile": profile,
                        "scenario": scenario,
                        "backend": "dashscope",
                        "success": False,
                        "error_type": "ApiBudgetExhausted",
                        "metrics": {"orphan_free": True},
                    },
                ],
            )

    status, note = _quality_status(rows)

    assert status == "unknown"
    assert "budget skips do not count" in note


def test_core_mock_tax_gate_uses_s1_not_guard_augmented_s2() -> None:
    measurements = [
        {
            "profile": "core",
            "scenario": "S1",
            "backend": "mock",
            "success": True,
            "metrics": {"mock_orchestration_tax_ms": 100.0},
        },
        {
            "profile": "core",
            "scenario": "S2",
            "backend": "mock",
            "success": True,
            "metrics": {"mock_orchestration_tax_ms": 900.0},
        },
    ]

    metrics = _extract_core_metrics(measurements, [], [])

    assert metrics["mock_tax_ms"] == 100.0


def test_report_command_rebuilds_minimal_offline_bundle(tmp_path: Path) -> None:
    result_dir = tmp_path / "minimal-result"
    result_dir.mkdir()
    (result_dir / "manifest.json").write_text(
        json.dumps({"run_id": "minimal-result", "arm_resolver": None}),
        encoding="utf-8",
    )
    measurement = {
        "run_id": "minimal-result",
        "profile": "core",
        "scenario": "idle",
        "backend": "mock",
        "record_type": "measurement",
        "success": True,
        "metrics": {
            "startup_ready_ms": 500.0,
            "idle_tree_pss_median_bytes": 64 * 1024 * 1024,
            "idle_cpu_avg_core_pct": 0.1,
            "idle_cpu_p95_core_pct": 0.2,
        },
    }
    (result_dir / "measurements.jsonl").write_text(
        json.dumps(measurement) + "\n",
        encoding="utf-8",
    )
    (result_dir / "requests.jsonl").write_text("", encoding="utf-8")
    current_install = FootprintRecord(
        category="install.venv_total",
        path="/synthetic/current-venv",
        layer=StateLayer.INSTALL,
        present=True,
        rollup=True,
        allocated_bytes=1008 * 1024 * 1024,
    )
    write_csv(
        result_dir / "footprint.csv",
        [{"record_type": "target", **current_install.to_dict()}],
    )

    assert main(["report", str(result_dir)]) == 0

    summary = (result_dir / "summary.csv").read_text(encoding="utf-8")
    report = (result_dir / "report.md").read_text(encoding="utf-8")
    assert "idle_tree_pss_median_bytes" in summary
    assert "| package_bytes | — | bytes | unknown |" in report
    assert "minimal-package gate stays unknown" in report
