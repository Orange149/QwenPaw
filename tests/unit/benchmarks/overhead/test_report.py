"""Tests for statistical policy, gates, and trimming rules."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.overhead.qwenpaw_overhead.report import (
    evaluate_core_gates,
    evaluate_trim_candidate,
    overall_gate_status,
    render_markdown_report,
    summarize_records,
    summarize_values,
    write_report_bundle,
    write_summary_csv,
)
from benchmarks.overhead.qwenpaw_overhead.schemas import (
    ArmDryRunResult,
    DecisionStatus,
    GateStatus,
    ReportThresholds,
    SecretFinding,
    TrimEvidence,
)


def test_three_runs_report_only_median_and_range() -> None:
    stats = summarize_values([3.0, 1.0, 2.0])

    assert stats.n == 3
    assert stats.median == 2.0
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0
    assert stats.value_range == 2.0
    assert stats.mean is None
    assert stats.p95 is None
    assert stats.policy == "median_range_n_le_3"


def test_p95_is_only_emitted_for_long_idle_time_series() -> None:
    values = list(range(1, 21))

    idle = summarize_values(values, idle_timeseries=True)
    repeated_runs = summarize_values(values, idle_timeseries=False)

    assert idle.p95 == 19.0
    assert idle.mean == 10.5
    assert repeated_runs.p95 is None


def test_idle_p95_requires_minimum_sample_count() -> None:
    stats = summarize_values(list(range(19)), idle_timeseries=True)

    assert stats.p95 is None


def test_summarize_records_requires_explicit_idle_sample_event() -> None:
    idle_records = [
        {
            "profile": "minimal",
            "scenario": "idle_30m",
            "backend": "dashscope",
            "event": "resource_sample",
            "data": {"cpu": index},
        }
        for index in range(20)
    ]
    non_sample_records = [{**record, "event": "run_result"} for record in idle_records]

    idle_summary = summarize_records(idle_records)[0]
    run_summary = summarize_records(non_sample_records)[0]

    assert idle_summary.stats.p95 == 18.0
    assert run_summary.stats.p95 is None


def test_write_summary_csv_uses_long_form_schema(tmp_path: Path) -> None:
    summaries = summarize_records(
        [
            {
                "profile": "full",
                "scenario": "request",
                "backend": "mock",
                "metrics": {"duration_ms": value},
            }
            for value in (10, 20, 30)
        ],
    )
    output = tmp_path / "summary.csv"

    write_summary_csv(output, summaries)

    with output.open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 1
    assert rows[0]["metric"] == "duration_ms"
    assert rows[0]["median"] == "20.0"
    assert rows[0]["p95"] == ""


def test_core_gates_apply_boundaries_and_linear_growth_failure() -> None:
    gates = evaluate_core_gates(
        {
            "package_bytes": 512 * 1024 * 1024,
            "framework_idle_pss_bytes": 513 * 1024 * 1024,
            "idle_cpu_avg_core_pct": 2.0,
            "idle_cpu_p95_core_pct": 5.0,
            "wsl_ready_seconds": 3.0,
            "arm_ready_seconds": 5.1,
            "mock_tax_ms": 251.0,
            "framework_context_tokens": 8_000,
            "ten_turn_pss_growth_bytes": 1,
            "fd_growth": 2,
            "thread_growth": 0,
        },
        linear_memory_growth=True,
    )
    by_metric = {gate.metric: gate for gate in gates}

    assert by_metric["package_bytes"].status is GateStatus.PASS
    assert by_metric["framework_idle_pss_bytes"].status is GateStatus.FAIL
    assert by_metric["idle_cpu_avg_core_pct"].status is GateStatus.WARN
    assert by_metric["arm_ready_seconds"].status is GateStatus.FAIL
    assert by_metric["mock_tax_ms"].status is GateStatus.FAIL
    assert by_metric["framework_context_tokens"].status is GateStatus.WARN
    assert by_metric["ten_turn_pss_growth_bytes"].status is GateStatus.FAIL


def test_sustained_fd_or_thread_growth_is_automatic_failure() -> None:
    gates = evaluate_core_gates(
        {"fd_growth": 0, "thread_growth": 0},
        sustained_fd_growth=True,
        sustained_thread_growth=True,
    )
    by_metric = {gate.metric: gate for gate in gates}

    assert by_metric["fd_growth"].status is GateStatus.FAIL
    assert by_metric["thread_growth"].status is GateStatus.FAIL


def test_unused_static_component_over_20_mib_is_trim_candidate() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="console",
            feature_used=False,
            unused_in_s1_s2=True,
            baseline_runs=3,
            variant_runs=3,
            disk_savings_bytes=20 * 1024 * 1024,
        ),
    )

    assert decision.status is DecisionStatus.TRIM_CANDIDATE
    assert "disk_savings_bytes" in decision.triggered_thresholds


def test_quality_regression_blocks_trim_even_with_large_saving() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="memory",
            feature_used=False,
            unused_in_s1_s2=True,
            baseline_runs=3,
            variant_runs=3,
            disk_savings_bytes=100 * 1024 * 1024,
            success_rate_drop_pp=0.1,
        ),
    )

    assert decision.status is DecisionStatus.BLOCKED


def test_cpu_and_percentage_deltas_do_not_nominate_trim_candidate() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="scheduler",
            feature_used=False,
            unused_in_s1_s2=True,
            baseline_runs=3,
            variant_runs=3,
            idle_cpu_savings_core_pct=99.0,
            input_token_savings_ratio=0.99,
            latency_savings_ratio=0.99,
        ),
    )

    assert decision.status is DecisionStatus.KEEP
    assert decision.triggered_thresholds == ()


def test_fixed_prompt_500_token_saving_is_allowed_trim_trigger() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="prompt_section",
            feature_used=False,
            unused_in_s1_s2=True,
            baseline_runs=3,
            variant_runs=3,
            fixed_prompt_token_savings=500,
        ),
    )

    assert decision.status is DecisionStatus.TRIM_CANDIDATE
    assert decision.triggered_thresholds == ("fixed_prompt_token_savings",)


def test_trim_requires_explicit_unused_in_s1_s2_evidence() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="unknown-usage",
            feature_used=False,
            baseline_runs=3,
            variant_runs=3,
            disk_savings_bytes=20 * 1024 * 1024,
        ),
    )

    assert decision.status is DecisionStatus.MEASURE_MORE


def test_used_expensive_component_is_optimize_not_trim_candidate() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="governance",
            feature_used=True,
            unused_in_s1_s2=False,
            baseline_runs=3,
            variant_runs=3,
            idle_pss_savings_bytes=20 * 1024 * 1024,
        ),
    )

    assert decision.status is DecisionStatus.OPTIMIZE_CANDIDATE


def test_dynamic_candidate_with_insufficient_repeats_requests_more_data() -> None:
    decision = evaluate_trim_candidate(
        TrimEvidence(
            component="memory",
            feature_used=False,
            unused_in_s1_s2=True,
            baseline_runs=2,
            variant_runs=2,
            idle_pss_savings_bytes=20 * 1024 * 1024,
        ),
    )

    assert decision.status is DecisionStatus.MEASURE_MORE


def test_markdown_contains_policy_and_no_secret_plaintext() -> None:
    finding = SecretFinding(
        path="result.json",
        rule="known_secret_value",
        fingerprint="0123456789abcdef",
        line_number=4,
    )
    markdown = render_markdown_report(
        [],
        arm_result=ArmDryRunResult(
            command=("python", "-m", "pip", "install", "--dry-run"),
            executed=False,
        ),
        secret_findings=[finding],
        report_thresholds=ReportThresholds(),
    )

    assert "n≤3" in markdown
    assert "p95" in markdown
    assert "0123456789abcdef" in markdown
    assert "secret plaintext" not in markdown
    assert overall_gate_status([], secret_findings=[finding]) is GateStatus.FAIL


def test_report_bundle_writes_stable_csv_and_markdown_names(
    tmp_path: Path,
) -> None:
    summaries = summarize_records(
        [
            {
                "profile": "minimal",
                "scenario": "request",
                "backend": "mock",
                "metrics": {"duration_ms": 10},
            }
        ],
    )

    paths = write_report_bundle(tmp_path, summaries)

    assert set(paths) == {
        "summary",
        "core_gates",
        "footprint",
        "distributions",
        "shared_objects",
        "trim_candidates",
        "secret_findings",
        "markdown",
    }
    assert all(path.exists() for path in paths.values())
    assert (
        paths["markdown"]
        .read_text(encoding="utf-8")
        .startswith(
            "# QwenPaw Overhead Report",
        )
    )
