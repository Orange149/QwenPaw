"""Small-sample-safe aggregation and Markdown/CSV reporting."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import write_csv
from .schemas import (
    ArmDryRunResult,
    CoreGateResult,
    CoreThresholds,
    DecisionStatus,
    DistributionRecord,
    FootprintRecord,
    GateStatus,
    MetricSummary,
    ReportThresholds,
    SecretFinding,
    SharedObjectRecord,
    SummaryStats,
    TrimDecision,
    TrimEvidence,
)

_METADATA_KEYS = {
    "backend",
    "command",
    "data",
    "event",
    "events",
    "monotonic_ns",
    "profile",
    "record_type",
    "run_id",
    "scenario",
    "schema_version",
    "wall_time_utc",
}
_IDLE_SAMPLE_EVENTS = {
    "idle_sample",
    "resource_sample",
    "sample",
    "timeseries_sample",
}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_values(
    values: Iterable[int | float],
    *,
    idle_timeseries: bool = False,
    thresholds: ReportThresholds | None = None,
) -> SummaryStats:
    """Summarize values without presenting an unstable three-run p95.

    Repeated-run experiments with ``n <= 3`` intentionally report only the
    median and range.  p95 is reserved for a sufficiently long idle time
    series and uses the deterministic nearest-rank definition.
    """

    limits = thresholds or ReportThresholds()
    clean = [
        number for value in values if (number := _finite_number(value)) is not None
    ]
    if not clean:
        return SummaryStats(
            n=0,
            median=None,
            minimum=None,
            maximum=None,
            value_range=None,
            policy="no_data",
        )

    minimum = min(clean)
    maximum = max(clean)
    small_sample = len(clean) <= limits.minimum_repeated_runs
    enough_idle = idle_timeseries and len(clean) >= limits.minimum_idle_samples_for_p95
    if small_sample:
        policy = "median_range_n_le_3"
    elif enough_idle:
        policy = "idle_timeseries_median_range_mean_p95"
    else:
        policy = "median_range_mean_no_p95"
    return SummaryStats(
        n=len(clean),
        median=float(statistics.median(clean)),
        minimum=minimum,
        maximum=maximum,
        value_range=maximum - minimum,
        mean=None if small_sample else float(statistics.fmean(clean)),
        p95=_nearest_rank(clean, 0.95) if enough_idle else None,
        policy=policy,
    )


def flatten_numeric_metrics(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten nested numeric leaves using dotted metric names."""

    flattened: dict[str, float] = {}
    for key in sorted(value):
        item = value[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_numeric_metrics(item, prefix=name))
            continue
        number = _finite_number(item)
        if number is not None:
            flattened[name] = number
    return flattened


def _record_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("metrics", "aggregate", "data"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return {key: value for key, value in record.items() if key not in _METADATA_KEYS}


def _is_idle_timeseries_record(record: Mapping[str, Any]) -> bool:
    explicit = record.get("is_idle_timeseries")
    if isinstance(explicit, bool):
        return explicit
    scenario = str(record.get("scenario") or "").casefold()
    event = str(record.get("event") or record.get("record_type") or "").casefold()
    return "idle" in scenario and event in _IDLE_SAMPLE_EVENTS


def summarize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: ReportThresholds | None = None,
) -> list[MetricSummary]:
    """Aggregate plain runner/sampler dictionaries by profile/scenario/backend."""

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    idle_flags: dict[tuple[str, str, str, str], bool] = defaultdict(bool)
    for record in records:
        profile = str(record.get("profile") or "unknown")
        scenario = str(record.get("scenario") or "unknown")
        backend = str(record.get("backend") or "unknown")
        idle_sample = _is_idle_timeseries_record(record)
        for metric, value in flatten_numeric_metrics(_record_metrics(record)).items():
            key = (profile, scenario, backend, metric)
            grouped[key].append(value)
            idle_flags[key] = idle_flags[key] or idle_sample

    summaries = [
        MetricSummary(
            profile=profile,
            scenario=scenario,
            backend=backend,
            metric=metric,
            stats=summarize_values(
                values,
                idle_timeseries=idle_flags[key],
                thresholds=thresholds,
            ),
        )
        for key, values in grouped.items()
        for profile, scenario, backend, metric in [key]
    ]
    return sorted(
        summaries,
        key=lambda item: (item.profile, item.scenario, item.backend, item.metric),
    )


def write_summary_csv(
    path: str | Path,
    summaries: Iterable[MetricSummary],
) -> None:
    """Write deterministic long-form summary CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile",
        "scenario",
        "backend",
        "metric",
        "n",
        "median",
        "minimum",
        "maximum",
        "range",
        "mean",
        "p95",
        "policy",
    ]
    with output.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())


def _upper_bound_gate(
    metric: str,
    value: float | None,
    *,
    unit: str,
    pass_threshold: float,
    fail_threshold: float | None,
    force_fail: bool = False,
    note: str = "",
) -> CoreGateResult:
    if value is None:
        status = GateStatus.UNKNOWN
    elif force_fail or (fail_threshold is not None and value > fail_threshold):
        status = GateStatus.FAIL
    elif value <= pass_threshold:
        status = GateStatus.PASS
    else:
        status = GateStatus.WARN
    return CoreGateResult(
        metric=metric,
        value=value,
        unit=unit,
        status=status,
        pass_threshold=pass_threshold,
        fail_threshold=fail_threshold,
        note=note,
    )


def evaluate_core_gates(
    metrics: Mapping[str, int | float | None],
    *,
    thresholds: CoreThresholds | None = None,
    linear_memory_growth: bool = False,
    sustained_fd_growth: bool = False,
    sustained_thread_growth: bool = False,
) -> list[CoreGateResult]:
    """Evaluate the protocol's absolute framework-overhead thresholds."""

    limits = thresholds or CoreThresholds()
    specifications = (
        (
            "package_bytes",
            "bytes",
            limits.package_pass_bytes,
            limits.package_fail_bytes,
        ),
        (
            "framework_idle_pss_bytes",
            "bytes",
            limits.idle_pss_pass_bytes,
            limits.idle_pss_fail_bytes,
        ),
        (
            "idle_cpu_avg_core_pct",
            "single_core_percent",
            limits.idle_cpu_avg_pass_core_pct,
            limits.idle_cpu_avg_fail_core_pct,
        ),
        (
            "idle_cpu_p95_core_pct",
            "single_core_percent",
            limits.idle_cpu_p95_pass_core_pct,
            limits.idle_cpu_p95_fail_core_pct,
        ),
        (
            "wsl_ready_seconds",
            "seconds",
            limits.wsl_ready_pass_seconds,
            limits.wsl_ready_fail_seconds,
        ),
        (
            "arm_ready_seconds",
            "seconds",
            limits.arm_ready_pass_seconds,
            limits.arm_ready_fail_seconds,
        ),
        (
            "mock_tax_ms",
            "milliseconds",
            limits.mock_tax_pass_ms,
            limits.mock_tax_fail_ms,
        ),
        (
            "framework_context_tokens",
            "tokens",
            limits.framework_context_pass_tokens,
            limits.framework_context_fail_tokens,
        ),
        (
            "ten_turn_pss_growth_bytes",
            "bytes",
            limits.ten_turn_pss_growth_pass_bytes,
            limits.ten_turn_pss_growth_fail_bytes,
        ),
    )
    gates = [
        _upper_bound_gate(
            metric,
            metrics.get(metric),
            unit=unit,
            pass_threshold=pass_value,
            fail_threshold=fail_value,
            force_fail=metric == "ten_turn_pss_growth_bytes" and linear_memory_growth,
            note=(
                "linear growth is an automatic failure"
                if metric == "ten_turn_pss_growth_bytes"
                else ""
            ),
        )
        for metric, unit, pass_value, fail_value in specifications
    ]
    gates.extend(
        [
            _upper_bound_gate(
                "fd_growth",
                metrics.get("fd_growth"),
                unit="descriptors",
                pass_threshold=limits.fd_growth_pass,
                fail_threshold=None,
                force_fail=sustained_fd_growth,
                note=("values above the pass gate warn; sustained growth fails"),
            ),
            _upper_bound_gate(
                "thread_growth",
                metrics.get("thread_growth"),
                unit="threads",
                pass_threshold=limits.thread_growth_pass,
                fail_threshold=None,
                force_fail=sustained_thread_growth,
                note=("values above the pass gate warn; sustained growth fails"),
            ),
        ],
    )
    return gates


def evaluate_trim_candidate(
    evidence: TrimEvidence,
    *,
    thresholds: ReportThresholds | None = None,
) -> TrimDecision:
    """Apply explicit quality, repeatability, savings, and portability rules."""

    limits = thresholds or ReportThresholds()
    reasons: list[str] = []
    triggered: list[str] = []

    if evidence.required:
        if evidence.arm_compatible is False:
            return TrimDecision(
                evidence.component,
                DecisionStatus.BLOCKED,
                ("required component has no compatible aarch64 wheel",),
                ("arm_compatible",),
            )
        return TrimDecision(
            evidence.component,
            DecisionStatus.KEEP,
            ("component is required by the target profile",),
        )

    if evidence.success_rate_drop_pp > limits.maximum_success_rate_drop_pp:
        return TrimDecision(
            evidence.component,
            DecisionStatus.BLOCKED,
            ("ablation exceeds the allowed task-success regression",),
            ("maximum_success_rate_drop_pp",),
        )
    if evidence.failure_rate_increase_pp > limits.maximum_failure_rate_increase_pp:
        return TrimDecision(
            evidence.component,
            DecisionStatus.BLOCKED,
            ("ablation exceeds the allowed failure-rate regression",),
            ("maximum_failure_rate_increase_pp",),
        )
    if (
        evidence.uss_growth_bytes_per_request
        > limits.maximum_uss_growth_bytes_per_request
        or evidence.fd_growth > limits.maximum_fd_growth
        or evidence.thread_growth > limits.maximum_thread_growth
    ):
        return TrimDecision(
            evidence.component,
            DecisionStatus.BLOCKED,
            ("ablation introduces a resource-growth regression",),
            ("resource_growth",),
        )

    fixed_prompt_savings = (
        evidence.fixed_prompt_token_savings
        if evidence.fixed_prompt_token_savings is not None
        else evidence.input_token_savings
    )
    # The trim protocol deliberately limits entry criteria to four material
    # costs.  CPU and percentage deltas remain reportable evidence but cannot
    # nominate a component for deletion by themselves.
    savings = (
        (
            "disk_savings_bytes",
            evidence.disk_savings_bytes >= limits.disk_savings_bytes,
        ),
        (
            "idle_pss_savings_bytes",
            evidence.idle_pss_savings_bytes >= limits.idle_pss_savings_bytes,
        ),
        (
            "fixed_prompt_token_savings",
            fixed_prompt_savings >= limits.input_token_savings,
        ),
        (
            "latency_savings_ms",
            evidence.latency_savings_ms >= limits.latency_savings_ms,
        ),
    )
    triggered.extend(name for name, matched in savings if matched)
    if evidence.arm_compatible is False:
        reasons.append("component is unavailable for aarch64/cp312")

    repeated = (
        evidence.baseline_runs >= limits.minimum_repeated_runs
        and evidence.variant_runs >= limits.minimum_repeated_runs
    )
    if triggered and not repeated:
        if (
            evidence.unused_in_s1_s2 is None
            and evidence.feature_used is None
        ):
            reasons.append("S1/S2 feature usage is unknown")
        return TrimDecision(
            evidence.component,
            DecisionStatus.MEASURE_MORE,
            tuple(reasons + ["fewer than three baseline/variant runs"]),
            tuple(triggered),
        )
    if not triggered:
        if evidence.arm_compatible is False:
            return TrimDecision(
                evidence.component,
                DecisionStatus.OPTIMIZE_CANDIDATE,
                tuple(reasons + ["replace or repackage for aarch64"]),
                ("arm_compatible",),
            )
        return TrimDecision(
            evidence.component,
            DecisionStatus.KEEP,
            ("no saving or portability threshold was triggered",),
        )
    if evidence.unused_in_s1_s2 is True:
        reasons.append("feature was unused in both S1 and S2")
        return TrimDecision(
            evidence.component,
            DecisionStatus.TRIM_CANDIDATE,
            tuple(reasons),
            tuple(triggered),
        )
    if evidence.unused_in_s1_s2 is False or evidence.feature_used is True:
        reasons.append("feature is used; optimize or replace instead of deleting")
        return TrimDecision(
            evidence.component,
            DecisionStatus.OPTIMIZE_CANDIDATE,
            tuple(reasons),
            tuple(triggered),
        )
    return TrimDecision(
        evidence.component,
        DecisionStatus.MEASURE_MORE,
        tuple(reasons + ["S1/S2 feature usage is unknown"]),
        tuple(triggered),
    )


def overall_gate_status(
    gates: Iterable[CoreGateResult],
    *,
    secret_findings: Sequence[SecretFinding] = (),
) -> GateStatus:
    if secret_findings:
        return GateStatus.FAIL
    statuses = {gate.status for gate in gates}
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.WARN in statuses or GateStatus.UNKNOWN in statuses:
        return GateStatus.WARN
    return GateStatus.PASS


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def render_markdown_report(
    summaries: Sequence[MetricSummary],
    *,
    core_gates: Sequence[CoreGateResult] = (),
    footprint: Sequence[FootprintRecord] = (),
    distributions: Sequence[DistributionRecord] = (),
    shared_objects: Sequence[SharedObjectRecord] = (),
    trim_decisions: Sequence[TrimDecision] = (),
    arm_result: ArmDryRunResult | None = None,
    secret_findings: Sequence[SecretFinding] = (),
    report_thresholds: ReportThresholds | None = None,
) -> str:
    """Render a compact report with explicit statistical and trimming policy."""

    limits = report_thresholds or ReportThresholds()
    lines = [
        "# QwenPaw Overhead Report",
        "",
        f"Overall gate: **{overall_gate_status(core_gates, secret_findings=secret_findings).value.upper()}**",
        "",
        "## Statistical policy",
        "",
        (
            "Repeated experiments with n≤3 report median and range only. "
            "p95 is emitted only for idle time series with at least "
            f"{limits.minimum_idle_samples_for_p95} samples."
        ),
        "",
    ]
    if core_gates:
        lines.extend(
            [
                "## Absolute overhead gates",
                "",
                "| Metric | Value | Unit | Status | Pass ≤ | Fail > | Note |",
                "|---|---:|---|---|---:|---:|---|",
            ],
        )
        for gate in core_gates:
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        gate.metric,
                        gate.value,
                        gate.unit,
                        gate.status.value,
                        gate.pass_threshold,
                        gate.fail_threshold,
                        gate.note,
                    )
                )
                + " |",
            )
        lines.append("")

    if summaries:
        lines.extend(
            [
                "## Metric summaries",
                "",
                "| Profile | Scenario | Backend | Metric | n | Median | Range | Mean | p95 | Policy |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---|",
            ],
        )
        for summary in summaries:
            stats = summary.stats
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        summary.profile,
                        summary.scenario,
                        summary.backend,
                        summary.metric,
                        stats.n,
                        stats.median,
                        stats.value_range,
                        stats.mean,
                        stats.p95,
                        stats.policy,
                    )
                )
                + " |",
            )
        lines.append("")

    network_metric_names = {
        "network_sample_count",
        "network_socket_scan_sample_count",
        "pre_request_non_loopback_count",
        "runtime_non_loopback_destination_count",
        "runtime_loopback_destination_count",
        "relay_upstream_host_policy_enforced_count",
    }
    network_summaries = [
        summary for summary in summaries if summary.metric in network_metric_names
    ]
    if network_summaries:
        lines.extend(
            [
                "## Network observation limits",
                "",
                (
                    "Process INET sockets are sampled on every resource interval. "
                    "A zero non-loopback count means no violation was observed; "
                    "it is not packet-complete proof and is not a firewall. "
                    "For remote runs, the relay separately enforces an HTTPS "
                    "DashScope aliyuncs.com upstream before any request starts."
                ),
                "",
                "| Profile | Scenario | Backend | Observation | n | Median | Range |",
                "|---|---|---|---|---:|---:|---:|",
            ],
        )
        for summary in network_summaries:
            lines.append(
                f"| {_cell(summary.profile)} | {_cell(summary.scenario)} | "
                f"{_cell(summary.backend)} | {_cell(summary.metric)} | "
                f"{summary.stats.n} | {_cell(summary.stats.median)} | "
                f"{_cell(summary.stats.value_range)} |",
            )
        lines.append("")

    shared_cgroup = [
        summary
        for summary in summaries
        if summary.metric.startswith("shared_cgroup_crosscheck_")
    ]
    if shared_cgroup:
        lines.extend(
            [
                "## cgroup attribution limits",
                "",
                (
                    "The process inherited a cgroup with pre-existing members. "
                    "Its CPU/I/O/memory deltas are retained only as a shared "
                    "whole-cgroup cross-check; QwenPaw gates use the sampled "
                    "descendant process tree, not these shared counters."
                ),
                "",
            ],
        )
    if footprint:
        lines.extend(
            [
                "## Footprint",
                "",
                "| Category | Layer | Allocated | Apparent | Files | Present |",
                "|---|---|---:|---:|---:|---|",
            ],
        )
        for record in footprint:
            lines.append(
                f"| {_cell(record.category)} | {record.layer.value} | "
                f"{_bytes(record.allocated_bytes)} | {_bytes(record.apparent_bytes)} | "
                f"{record.file_count} | {record.present} |",
            )
        lines.append("")

    if distributions:
        lines.extend(
            [
                "## Largest installed distributions (RECORD)",
                "",
                "| Distribution | Version | Allocated | Apparent | Files | Native .so |",
                "|---|---|---:|---:|---:|---:|",
            ],
        )
        for record in sorted(
            distributions,
            key=lambda item: item.allocated_bytes,
            reverse=True,
        )[:25]:
            lines.append(
                f"| {_cell(record.name)} | {_cell(record.version)} | "
                f"{_bytes(record.allocated_bytes)} | {_bytes(record.apparent_bytes)} | "
                f"{record.file_count} | {record.native_file_count} |",
            )
        lines.append("")

    if shared_objects:
        architectures: dict[str, int] = defaultdict(int)
        for record in shared_objects:
            architectures[record.elf_machine or "not_elf"] += 1
        lines.extend(["## Native-object architecture", ""])
        for architecture, count in sorted(architectures.items()):
            lines.append(f"- {architecture}: {count}")
        lines.append("")

    if arm_result is not None:
        if not arm_result.executed:
            arm_status = "planned (not executed)"
        elif arm_result.compatible:
            arm_status = "compatible"
        else:
            arm_status = "incompatible or unresolved"
        lines.extend(
            [
                "## aarch64 / CPython 3.12 resolver",
                "",
                f"Status: **{arm_status}**",
                "",
                "```text",
                " ".join(arm_result.command),
                "```",
                "",
            ],
        )
        if arm_result.blockers:
            lines.extend(
                [
                    "Resolver blockers:",
                    "",
                    "| Package | Kind | Meaning |",
                    "|---|---|---|",
                ],
            )
            for blocker in arm_result.blockers:
                lines.append(
                    f"| {_cell(blocker.get('package'))} | "
                    f"{_cell(blocker.get('kind'))} | "
                    f"{_cell(blocker.get('reason'))} |",
                )
            lines.append("")
        if arm_result.resolution_limited_by_top_level:
            lines.extend(
                [
                    (
                        "The resolver failed at the QwenPaw top-level "
                        "requirement, so this run does not establish whether "
                        "its transitive native dependencies have ARM wheels."
                    ),
                    "",
                ],
            )

    if trim_decisions:
        lines.extend(
            [
                "## Trimming candidates",
                "",
                (
                    "A component is a trim candidate only when three repeated "
                    "baseline/variant runs show it unused in both S1 and S2, "
                    "success/failure rates do not regress, resource-growth "
                    "guards pass, and one allowed material-saving gate fires."
                ),
                "",
                "Allowed trim-entry gates:",
                "",
                f"- disk saving ≥ {_bytes(limits.disk_savings_bytes)}",
                f"- idle PSS saving ≥ {_bytes(limits.idle_pss_savings_bytes)}",
                f"- fixed-prompt saving ≥ {limits.input_token_savings} tokens",
                f"- ready/request latency saving ≥ {limits.latency_savings_ms:.0f} ms",
                (
                    "- CPU and percentage deltas are reported but never "
                    "nominate trimming by themselves"
                ),
                "",
                "| Component | Decision | Triggered thresholds | Reasons |",
                "|---|---|---|---|",
            ],
        )
        for decision in trim_decisions:
            lines.append(
                f"| {_cell(decision.component)} | {decision.status.value} | "
                f"{_cell(', '.join(decision.triggered_thresholds))} | "
                f"{_cell('; '.join(decision.reasons))} |",
            )
        lines.append("")

    lines.extend(["## Secret scan", ""])
    if not secret_findings:
        lines.extend(["PASS — no credential material detected.", ""])
    else:
        lines.extend(
            [
                (
                    f"FAIL — {len(secret_findings)} finding(s). Values are "
                    "omitted; only salted fingerprints are shown."
                ),
                "",
            ],
        )
        for finding in secret_findings:
            lines.append(
                f"- `{_cell(finding.path)}:{finding.line_number or 0}` "
                f"{finding.rule} `{finding.fingerprint}`",
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    path: str | Path,
    summaries: Sequence[MetricSummary],
    **kwargs: Any,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown_report(summaries, **kwargs),
        encoding="utf-8",
    )


def write_report_bundle(
    output_dir: str | Path,
    summaries: Sequence[MetricSummary],
    *,
    core_gates: Sequence[CoreGateResult] = (),
    footprint: Sequence[FootprintRecord] = (),
    distributions: Sequence[DistributionRecord] = (),
    shared_objects: Sequence[SharedObjectRecord] = (),
    trim_decisions: Sequence[TrimDecision] = (),
    arm_result: ArmDryRunResult | None = None,
    secret_findings: Sequence[SecretFinding] = (),
    report_thresholds: ReportThresholds | None = None,
) -> dict[str, Path]:
    """Write the complete public CSV/Markdown report set."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / "summary.csv",
        "core_gates": output / "core_gates.csv",
        "footprint": output / "footprint.csv",
        "distributions": output / "distributions.csv",
        "shared_objects": output / "shared_objects.csv",
        "trim_candidates": output / "trim_candidates.csv",
        "secret_findings": output / "secret_findings.csv",
        "markdown": output / "report.md",
    }
    write_summary_csv(paths["summary"], summaries)
    for name, records in (
        ("core_gates", core_gates),
        ("footprint", footprint),
        ("distributions", distributions),
        ("shared_objects", shared_objects),
        ("trim_candidates", trim_decisions),
        ("secret_findings", secret_findings),
    ):
        write_csv(paths[name], (record.to_dict() for record in records))
    write_markdown_report(
        paths["markdown"],
        summaries,
        core_gates=core_gates,
        footprint=footprint,
        distributions=distributions,
        shared_objects=shared_objects,
        trim_decisions=trim_decisions,
        arm_result=arm_result,
        secret_findings=secret_findings,
        report_thresholds=report_thresholds,
    )
    return paths


__all__ = [
    "evaluate_core_gates",
    "evaluate_trim_candidate",
    "flatten_numeric_metrics",
    "overall_gate_status",
    "render_markdown_report",
    "summarize_records",
    "summarize_values",
    "write_markdown_report",
    "write_report_bundle",
    "write_summary_csv",
]
