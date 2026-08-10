"""Stable, JSON-safe records used by the QwenPaw overhead benchmark.

The benchmark intentionally keeps its result format independent from QwenPaw's
runtime models.  That makes raw measurements readable even when the installed
QwenPaw version changes, and lets every collector emit plain dictionaries at
the process boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any

SCHEMA_VERSION = "1"


class StateLayer(str, Enum):
    """Storage ownership layer for a footprint target."""

    INSTALL = "install"
    MUTABLE_STATE = "mutable_state"
    SECRETS = "secrets"
    LOGS = "logs"
    CACHE = "cache"
    MODELS = "models"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    """Outcome of a component-level trimming rule."""

    TRIM_CANDIDATE = "trim_candidate"
    OPTIMIZE_CANDIDATE = "optimize_candidate"
    KEEP = "keep"
    BLOCKED = "blocked"
    MEASURE_MORE = "measure_more"


class GateStatus(str, Enum):
    """Three-level verdict for the framework's absolute overhead gates."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


def _json_safe(value: Any) -> Any:
    """Recursively convert benchmark records to JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    return value


class JsonRecord:
    """Mixin for records that cross the JSON/CSV boundary."""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True, slots=True)
class EventRecord(JsonRecord):
    """One line in the common benchmark event JSONL stream."""

    run_id: str
    profile: str
    scenario: str
    backend: str
    event: str
    monotonic_ns: int
    wall_time_utc: str
    data: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FootprintTarget(JsonRecord):
    """A path and the storage category it represents."""

    category: str
    path: str
    layer: StateLayer = StateLayer.UNKNOWN
    # Rollups are measured independently instead of competing for ownership
    # of inodes with their child categories.
    rollup: bool = False
    optional: bool = True
    exclude_root_names: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class FootprintRecord(JsonRecord):
    """Allocated and apparent usage for one target tree."""

    category: str
    path: str
    layer: StateLayer
    present: bool
    rollup: bool = False
    excluded_root_names: Sequence[str] = field(default_factory=tuple)
    allocated_bytes: int = 0
    apparent_bytes: int = 0
    file_count: int = 0
    directory_count: int = 0
    symlink_count: int = 0
    other_entry_count: int = 0
    unique_inode_count: int = 0
    duplicate_inode_count: int = 0
    inaccessible_entry_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DistributionRecord(JsonRecord):
    """Files attributed to one installed distribution through ``RECORD``."""

    name: str
    version: str
    location: str
    allocated_bytes: int
    apparent_bytes: int
    file_count: int
    native_file_count: int = 0
    duplicate_inode_count: int = 0
    missing_file_count: int = 0
    outside_root_file_count: int = 0


@dataclass(frozen=True, slots=True)
class SharedObjectRecord(JsonRecord):
    """One native shared object relevant to a future ARM deployment."""

    path: str
    distribution: str | None
    allocated_bytes: int
    apparent_bytes: int
    elf_class: int | None
    elf_machine: str | None
    elf_machine_id: int | None
    is_symlink: bool = False


@dataclass(frozen=True, slots=True)
class ArmDryRunResult(JsonRecord):
    """Plan or result of the opt-in CPython 3.12/aarch64 pip resolver check."""

    command: Sequence[str]
    executed: bool
    compatible: bool | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blockers: Sequence[Mapping[str, str]] = field(default_factory=tuple)
    resolution_limited_by_top_level: bool = False


@dataclass(frozen=True, slots=True)
class SecretFinding(JsonRecord):
    """A leak indicator that never includes the original secret value."""

    path: str
    rule: str
    fingerprint: str
    line_number: int | None = None


@dataclass(frozen=True, slots=True)
class SummaryStats(JsonRecord):
    """Small-sample-safe summary for one numeric metric."""

    n: int
    median: float | None
    minimum: float | None
    maximum: float | None
    value_range: float | None
    mean: float | None = None
    p95: float | None = None
    policy: str = "median_range"

    def to_dict(self) -> dict[str, Any]:
        # Explicit base dispatch avoids zero-argument ``super()`` interacting
        # badly with the replacement class created by ``dataclass(slots=True)``.
        data = JsonRecord.to_dict(self)
        data["range"] = data.pop("value_range")
        return data


@dataclass(frozen=True, slots=True)
class MetricSummary(JsonRecord):
    """A metric summary together with its experimental grouping."""

    profile: str
    scenario: str
    backend: str
    metric: str
    stats: SummaryStats

    def to_dict(self) -> dict[str, Any]:
        data = {
            "profile": self.profile,
            "scenario": self.scenario,
            "backend": self.backend,
            "metric": self.metric,
        }
        data.update(self.stats.to_dict())
        return data


@dataclass(frozen=True, slots=True)
class ReportThresholds(JsonRecord):
    """Configurable default gates for identifying trimming candidates."""

    minimum_repeated_runs: int = 3
    minimum_idle_samples_for_p95: int = 20
    disk_savings_bytes: int = 20 * 1024 * 1024
    idle_pss_savings_bytes: int = 10 * 1024 * 1024
    idle_cpu_savings_core_pct: float = 1.0
    input_token_savings: int = 500
    input_token_savings_ratio: float = 0.10
    latency_savings_ms: float = 100.0
    latency_savings_ratio: float = 0.10
    maximum_success_rate_drop_pp: float = 0.0
    maximum_failure_rate_increase_pp: float = 0.0
    maximum_uss_growth_bytes_per_request: float = 64 * 1024
    maximum_fd_growth: int = 2
    maximum_thread_growth: int = 0


@dataclass(frozen=True, slots=True)
class CoreThresholds(JsonRecord):
    """Absolute pass/warn/fail gates fixed by the benchmark protocol."""

    package_pass_bytes: int = 512 * 1024 * 1024
    package_fail_bytes: int = 1024 * 1024 * 1024
    idle_pss_pass_bytes: int = 256 * 1024 * 1024
    idle_pss_fail_bytes: int = 512 * 1024 * 1024
    idle_cpu_avg_pass_core_pct: float = 1.0
    idle_cpu_avg_fail_core_pct: float = 3.0
    idle_cpu_p95_pass_core_pct: float = 5.0
    idle_cpu_p95_fail_core_pct: float = 10.0
    wsl_ready_pass_seconds: float = 3.0
    wsl_ready_fail_seconds: float = 5.0
    arm_ready_pass_seconds: float = 5.0
    arm_ready_fail_seconds: float = 5.0
    mock_tax_pass_ms: float = 100.0
    mock_tax_fail_ms: float = 250.0
    framework_context_pass_tokens: int = 4_000
    framework_context_fail_tokens: int = 8_000
    ten_turn_pss_growth_pass_bytes: int = 10 * 1024 * 1024
    ten_turn_pss_growth_fail_bytes: int = 50 * 1024 * 1024
    fd_growth_pass: int = 2
    thread_growth_pass: int = 0


@dataclass(frozen=True, slots=True)
class CoreGateResult(JsonRecord):
    """One evaluated absolute framework-overhead threshold."""

    metric: str
    value: float | int | None
    unit: str
    status: GateStatus
    pass_threshold: float | int
    fail_threshold: float | int | None
    note: str = ""


@dataclass(frozen=True, slots=True)
class TrimEvidence(JsonRecord):
    """Normalized evidence for a module/package/feature ablation."""

    component: str
    feature_used: bool | None = None
    unused_in_s1_s2: bool | None = None
    required: bool = False
    baseline_runs: int = 0
    variant_runs: int = 0
    disk_savings_bytes: int = 0
    idle_pss_savings_bytes: int = 0
    idle_cpu_savings_core_pct: float = 0.0
    input_token_savings: int = 0
    fixed_prompt_token_savings: int | None = None
    input_token_savings_ratio: float = 0.0
    latency_savings_ms: float = 0.0
    latency_savings_ratio: float = 0.0
    success_rate_drop_pp: float = 0.0
    failure_rate_increase_pp: float = 0.0
    uss_growth_bytes_per_request: float = 0.0
    fd_growth: int = 0
    thread_growth: int = 0
    arm_compatible: bool | None = None


@dataclass(frozen=True, slots=True)
class TrimDecision(JsonRecord):
    """Rule result for one potential trimming target."""

    component: str
    status: DecisionStatus
    reasons: Sequence[str] = field(default_factory=tuple)
    triggered_thresholds: Sequence[str] = field(default_factory=tuple)


__all__ = [
    "SCHEMA_VERSION",
    "ArmDryRunResult",
    "CoreGateResult",
    "CoreThresholds",
    "DecisionStatus",
    "DistributionRecord",
    "EventRecord",
    "FootprintRecord",
    "FootprintTarget",
    "GateStatus",
    "JsonRecord",
    "MetricSummary",
    "ReportThresholds",
    "SecretFinding",
    "SharedObjectRecord",
    "StateLayer",
    "SummaryStats",
    "TrimDecision",
    "TrimEvidence",
]
