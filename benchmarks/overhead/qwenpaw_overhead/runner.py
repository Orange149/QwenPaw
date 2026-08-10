"""High-level orchestration for the QwenPaw overhead experiment matrix."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import ipaddress
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import psutil

from .acp_client import ACPBenchmarkClient, ACPConfig
from .artifacts import atomic_write_json, write_csv
from .baseline import sample_system_baseline
from .direct_client import run_request
from .footprint import (
    arm_cp312_pip_dry_run,
    default_footprint_targets,
    distribution_file_index,
    measure_targets,
    scan_distributions,
    scan_for_secrets,
    scan_shared_objects,
)
from .host_metrics import vmmemwsl_delta
from .isolation import create_isolated_run
from .manifest import collect_manifest
from .matrix import (
    CONTROLLED_PROFILES,
    REQUEST_SCENARIOS,
    ApiBudget,
    MatrixJob,
    dashscope_jobs,
    mock_jobs,
    startup_jobs,
)
from .mock_server import classify_request_kind
from .mock_server import start_server as start_mock_server
from .profiles import generate_profile
from .relay import start_server as start_relay_server
from .report import (
    evaluate_core_gates,
    evaluate_trim_candidate,
    render_markdown_report,
    summarize_records,
    write_summary_csv,
)
from .sampler import (
    ResourceSampler,
    diff_state,
    snapshot_state,
    wait_for_descendants_exit,
)
from .scenarios import S1, S2, S3, Scenario, get_scenario
from .schemas import TrimEvidence
from .token_accounting import analyze_request
from .tui_probe import run_tui_probe


DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@dataclass
class HarnessSettings:
    """Paths and durations for one result directory."""

    run_id: str
    result_dir: Path
    python_executable: Path = Path("/home/orange/.qwenpaw/venv/bin/python")
    seed_working_dir: Path = Path("/home/orange/.qwenpaw")
    seed_secret_dir: Path = Path("/home/orange/.qwenpaw.secret")
    model: str = "dashscope/qwen3.7-plus"
    profiles: tuple[str, ...] = CONTROLLED_PROFILES
    repetitions: int = 3
    idle_seconds: float = 90.0
    baseline_seconds: float = 60.0
    sample_interval_s: float = 0.25
    cpu_affinity: str = "0-3"
    startup_timeout_s: float = 180.0
    execute_arm_resolver: bool = False
    arm_requirement: str = "qwenpaw==2.0.1"
    allow_remote_api: bool = False
    confirm_free_quota_stop: bool = False
    dashscope_base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    remote_max_attempts: int = 30
    remote_max_tokens: int = 500_000
    remote_scenarios: tuple[str, ...] = REQUEST_SCENARIOS
    smoke: bool = False


@dataclass
class HarnessRunner:
    """Collect artifacts while keeping all copied credentials in ``/tmp``."""

    settings: HarnessSettings
    measurements: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    request_records: list[dict[str, Any]] = field(default_factory=list)
    sample_files: list[tuple[Path, dict[str, str]]] = field(default_factory=list)
    footprint_records: list[Any] = field(default_factory=list)
    distribution_records: list[Any] = field(default_factory=list)
    shared_object_records: list[Any] = field(default_factory=list)
    arm_result: Any = None
    remote_budget: ApiBudget = field(init=False)
    _captured_tool_schema: list[dict[str, Any]] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.settings.result_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.result_dir / "raw_samples").mkdir(exist_ok=True)
        self.remote_budget = ApiBudget(
            max_attempts=self.settings.remote_max_attempts,
            max_tokens=self.settings.remote_max_tokens,
        )

    @property
    def result_dir(self) -> Path:
        return self.settings.result_dir

    def initialize_manifest(self, *, backend: str, scenarios: list[str]) -> None:
        manifest = collect_manifest(
            run_id=self.settings.run_id,
            python_executable=self.settings.python_executable,
            seed_working_dir=self.settings.seed_working_dir,
            profiles=list(self.settings.profiles),
            scenarios=scenarios,
            backend=backend,
            model=self.settings.model,
            cpu_affinity=self.settings.cpu_affinity,
        )
        manifest["settings"] = {
            "repetitions": self.settings.repetitions,
            "idle_seconds": self.settings.idle_seconds,
            "baseline_seconds": self.settings.baseline_seconds,
            "sample_interval_s": self.settings.sample_interval_s,
            "startup_timeout_s": self.settings.startup_timeout_s,
            "smoke": self.settings.smoke,
        }
        manifest["status"] = "running"
        atomic_write_json(self.result_dir / "manifest.json", manifest)

    def collect_footprint(self) -> None:
        # Do not resolve the venv's python symlink before deriving its prefix:
        # resolving would turn .../venv/bin/python into /usr/bin/python3.12.
        python = self.settings.python_executable.expanduser().absolute()
        venv = python.parent.parent
        candidates = sorted((venv / "lib").glob("python*/site-packages"))
        site_packages = next(
            (path for path in candidates if (path / "qwenpaw").is_dir()),
            candidates[0] if candidates else venv / "missing-site-packages",
        )
        package_dir = site_packages / "qwenpaw"
        targets = default_footprint_targets(
            self.settings.seed_working_dir,
            install_root=venv,
            package_dir=package_dir,
            secret_dir=self.settings.seed_secret_dir,
            python_executable=python,
        )
        self.footprint_records = measure_targets(targets)
        self.distribution_records = scan_distributions(
            site_packages,
            allowed_root=venv,
        )
        index = distribution_file_index(site_packages)
        self.shared_object_records = scan_shared_objects(
            [site_packages],
            distribution_index=index,
        )
        self.arm_result = arm_cp312_pip_dry_run(
            self.settings.arm_requirement,
            execute=self.settings.execute_arm_resolver,
            python_executable=str(self.settings.python_executable),
        )

    def collect_baseline(self) -> None:
        rows = sample_system_baseline(
            self.settings.baseline_seconds,
            interval_s=min(1.0, max(0.1, self.settings.sample_interval_s)),
        )
        for row in rows:
            self.measurements.append(
                {
                    "run_id": self.settings.run_id,
                    "profile": "wsl_baseline",
                    "scenario": "idle",
                    "backend": "none",
                    "record_type": "timeseries_sample",
                    "is_idle_timeseries": True,
                    "metrics": row,
                },
            )

    def record_vmmemwsl_boundaries(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        """Record coarse Windows VM memory outside measured sample windows."""

        observation = vmmemwsl_delta(before, after)
        self.measurements.append(
            {
                "run_id": self.settings.run_id,
                "profile": "wsl_vmmem",
                "scenario": "whole_run_boundary",
                "backend": "windows-host",
                "record_type": "host_boundary",
                "metrics": {
                    "working_set_before_bytes": observation.get(
                        "working_set_before_bytes",
                    ),
                    "working_set_after_bytes": observation.get(
                        "working_set_after_bytes",
                    ),
                    "working_set_delta_bytes": observation.get(
                        "working_set_delta_bytes",
                    ),
                },
            },
        )
        manifest_path = self.result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["wsl_vmmemwsl"] = observation
        atomic_write_json(manifest_path, manifest)

    async def run_startup_matrix(self) -> None:
        for job in startup_jobs(
            self.settings.profiles,
            repetitions=self.settings.repetitions,
        ):
            # Even with backend warmup disabled, never leave the cloned seed's
            # cloud URL reachable during an offline idle test.  A loopback
            # mock is a deny-by-construction sink here; zero requests is part
            # of startup correctness.
            with start_mock_server(
                scenario="S1",
                run_id=self.settings.run_id,
                profile=job.profile,
                backend="mock-startup-sink",
            ) as server:
                record = await self._run_acp_job(
                    job,
                    scenario=None,
                    server=server,
                )
                server_events = server.snapshot_events()
                self.events.extend(_tag_events(server_events, job))
                self._record_wire_requests(
                    server.request_bodies,
                    server_events,
                    job,
                )
                if record is not None:
                    count = int(server.request_count)
                    record["metrics"]["idle_model_request_count"] = count
                    if count:
                        record["success"] = False
                        record["error_type"] = "UnexpectedIdleModelRequest"

    async def run_mock_matrix(self, *, include_s3: bool = True) -> None:
        for job in mock_jobs(
            self.settings.profiles,
            repetitions=self.settings.repetitions,
            include_s3=include_s3,
        ):
            scenario = get_scenario(job.scenario)
            record: dict[str, Any] | None = None
            try:
                with start_mock_server(
                    scenario=job.scenario,
                    run_id=self.settings.run_id,
                    profile=job.profile,
                    backend="mock",
                ) as server:
                    record = await self._run_acp_job(
                        job,
                        scenario=scenario,
                        server=server,
                    )
                    server_events = server.snapshot_events()
                    self.events.extend(_tag_events(server_events, job))
                    self._record_wire_requests(
                        server.request_bodies,
                        server_events,
                        job,
                    )
                    if job.scenario in {"S1", "S2"} and record is not None:
                        await self._run_mock_direct_replay(
                            job,
                            scenario,
                            server.primary_request_body(1),
                            record,
                        )
            except Exception as exc:  # one sample must not erase prior data
                if record is None:
                    self._record_failure(job, exc)
                else:
                    record["success"] = False
                    record["error_type"] = type(exc).__name__
                    record["postprocess_error"] = type(exc).__name__

    async def run_dashscope_matrix(self) -> None:
        """Run the opt-in paid/limited API matrix through the safe relay."""

        self._validate_remote_authorization()
        for job in dashscope_jobs(
            repetitions=self.settings.repetitions,
            scenarios=self.settings.remote_scenarios,
        ):
            if self.remote_budget.exhausted:
                self._record_budget_skip(job, "remote API budget exhausted")
                continue
            if job.direct:
                await self._run_remote_direct(job)
            else:
                await self._run_remote_acp(job)

    async def run_tui_reference(self) -> None:
        """Measure one stock Textual TUI parent plus its ACP descendant."""

        job = MatrixJob(
            phase="tui",
            profile="stock_reference",
            scenario="idle",
            backend="mock-tui",
            repetition=1,
        )
        sample_id = job.sample_id
        sample_csv = self.result_dir / "raw_samples" / f"{sample_id}.csv"
        metadata = {
            "run_id": self.settings.run_id,
            "sample_id": sample_id,
            "profile": "stock_reference",
            "scenario": "idle",
            "backend": "mock-tui",
        }
        self.sample_files.append((sample_csv, metadata))
        sampler: ResourceSampler | None = None
        aggregate: dict[str, Any] = {}
        observed: list[Any] = []
        try:
            with start_mock_server(
                scenario="WARMUP",
                run_id=self.settings.run_id,
                profile="stock_reference",
                backend="mock-tui",
            ) as server:
                with create_isolated_run(
                    self.settings.seed_working_dir,
                    self.settings.seed_secret_dir,
                    f"{self.settings.run_id}-{sample_id}",
                ) as paths:
                    profile = generate_profile(
                        paths,
                        "stock_reference",
                        model=None,
                        base_url=server.base_url,
                    )
                    before = _snapshot_layers(paths)

                    def process_started(pid: int) -> None:
                        nonlocal sampler
                        sampler = ResourceSampler(
                            pid,
                            self.settings.sample_interval_s,
                            sample_csv,
                        ).start()

                    result = await asyncio.to_thread(
                        run_tui_probe,
                        python_executable=self.settings.python_executable,
                        env=profile.env,
                        project_dir=paths.project_dir,
                        affinity=self.settings.cpu_affinity,
                        startup_timeout_s=self.settings.startup_timeout_s,
                        idle_s=self.settings.idle_seconds,
                        on_process_started=process_started,
                    )
                    if sampler is not None:
                        aggregate = sampler.stop()
                        observed = aggregate.get("observed_processes", [])
                    orphan_free = wait_for_descendants_exit(observed, timeout=5.0)
                    after = _snapshot_layers(paths)
                    state_delta = _diff_layers(before, after)
                    network = _network_metrics(
                        sample_csv,
                        prompt_start_ns=None,
                    )
                    server_events = server.snapshot_events()
                    self.events.extend(
                        _tag_events(
                            server_events,
                            job,
                        ),
                    )
                    self._record_wire_requests(
                        server.request_bodies,
                        server_events,
                        job,
                    )
                    payload = result.to_dict()
                    self.measurements.append(
                        {
                            **metadata,
                            "record_type": "measurement",
                            "phase": "tui",
                            "repetition": 1,
                            "success": bool(
                                result.success
                                and orphan_free
                                and server.validation_failures == 0
                                and network.get("network_allowlist_ok") is not False
                            ),
                            "error_type": result.error,
                            "retry": 0,
                            "metrics": {
                                "startup_to_acp_ms": payload.get(
                                    "startup_to_acp_ms",
                                ),
                                "startup_to_ui_ready_ms": payload.get(
                                    "startup_to_ui_ready_ms",
                                ),
                                "exit_ms": payload.get("exit_ms"),
                                "output_bytes": result.output_bytes,
                                "model_attempts": server.request_count,
                                "mock_validation_failures": (
                                    server.validation_failures
                                ),
                                "orphan_free": orphan_free,
                                "state_apparent_net_delta_bytes": state_delta[
                                    "delta"
                                ]["apparent_bytes"],
                                "state_allocated_net_delta_bytes": state_delta[
                                    "delta"
                                ]["allocated_bytes"],
                                **state_delta["activity"],
                                **network,
                                **_sampler_summary_metrics(aggregate),
                            },
                            "resource_aggregate": aggregate,
                            "state_delta": _state_delta_summary(state_delta),
                            "profile_metadata": profile.metadata,
                            "limitations": [
                                "TUI UI-ready is not ACP AvailableCommands",
                                "the official project-bound TUI enables Coding Mode",
                            ],
                        },
                    )
        except Exception as exc:
            if sampler is not None:
                aggregate = sampler.stop()
            self._record_failure(job, exc)

    def _validate_remote_authorization(self) -> None:
        if not (
            self.settings.allow_remote_api
            and self.settings.confirm_free_quota_stop
        ):
            raise RuntimeError(
                "DashScope is disabled: pass both --allow-remote-api and "
                "--confirm-free-quota-stop after configuring the console limit",
            )
        endpoint = urlsplit(self.settings.dashscope_base_url)
        host = (endpoint.hostname or "").lower()
        if endpoint.scheme != "https" or not (
            host == "dashscope.aliyuncs.com" or host.endswith(".dashscope.aliyuncs.com")
        ):
            raise RuntimeError(
                "remote upstream must be an HTTPS DashScope aliyuncs.com endpoint",
            )
        if not os.environ.get(self.settings.dashscope_api_key_env):
            raise RuntimeError(
                f"{self.settings.dashscope_api_key_env} is required for the "
                "direct DashScope baseline; it is never written to results",
            )

    async def _run_remote_acp(self, job: MatrixJob) -> None:
        remaining = self.settings.remote_max_attempts - self.remote_budget.attempts
        record: dict[str, Any] | None = None
        try:
            with start_relay_server(
                self.settings.dashscope_base_url,
                run_id=self.settings.run_id,
                profile=job.profile,
                scenario=job.scenario,
                backend="dashscope",
                max_requests=max(0, remaining),
            ) as relay:
                try:
                    record = await self._run_acp_job(
                        job,
                        scenario=get_scenario(job.scenario),
                        server=relay,
                    )
                finally:
                    # Account first, while relay bodies are still retained.
                    # This finally path also runs if ACP parsing, recording, or
                    # shutdown fails after an upstream request was attempted.
                    relay_events = relay.snapshot_events()
                    relay_bodies = relay.request_bodies
                    attempts, tokens = self._account_remote(relay_events)
                    self.events.extend(_tag_events(relay_events, job))
                    self._record_wire_requests(relay_bodies, relay_events, job)
                if record is not None:
                    record["metrics"]["model_attempts"] = attempts
                    record["metrics"]["remote_budget_tokens_after"] = (
                        self.remote_budget.tokens
                    )
                    if attempts and tokens <= 0:
                        record["success"] = False
                        record["error_type"] = "MissingProviderUsage"
        except Exception as exc:
            if record is None:
                self._record_failure(job, exc)
            else:
                record["success"] = False
                record["error_type"] = type(exc).__name__

    async def _run_remote_direct(self, job: MatrixJob) -> None:
        if job.scenario == "S2" and self._captured_tool_schema is None:
            await self._calibrate_tool_schema()
        remaining = self.settings.remote_max_attempts - self.remote_budget.attempts
        api_key = os.environ.get(self.settings.dashscope_api_key_env)
        direct: dict[str, Any] | None = None
        try:
            with start_relay_server(
                self.settings.dashscope_base_url,
                run_id=self.settings.run_id,
                profile="direct",
                scenario=job.scenario,
                backend="dashscope-direct",
                max_requests=max(0, remaining),
            ) as relay:
                try:
                    body = self._direct_remote_body(get_scenario(job.scenario))
                    direct = await asyncio.to_thread(
                        run_request,
                        base_url=relay.base_url,
                        request_body=body,
                        api_key=api_key,
                        run_id=self.settings.run_id,
                        profile="direct",
                        scenario=job.scenario,
                        backend="dashscope-direct",
                        trust_env=False,
                        allow_exact_shell=job.scenario == "S2",
                    )
                finally:
                    relay_events = relay.snapshot_events()
                    relay_bodies = relay.request_bodies
                    attempts, tokens = self._account_remote(relay_events)
                    if direct is not None:
                        self.events.extend(
                            _tag_events(direct.get("events", []), job),
                        )
                    self.events.extend(_tag_events(relay_events, job))
                    self._record_wire_requests(relay_bodies, relay_events, job)
                if direct is None:
                    raise RuntimeError("direct client returned no result")
                usage_complete = not attempts or tokens > 0
                self.measurements.append(
                    {
                        "run_id": self.settings.run_id,
                        "sample_id": job.sample_id,
                        "profile": "direct",
                        "scenario": job.scenario,
                        "backend": "dashscope",
                        "record_type": "measurement",
                        "phase": "request",
                        "repetition": job.repetition,
                        "success": bool(
                            direct.get("success")
                            and usage_complete
                            and _direct_correctness(
                                get_scenario(job.scenario),
                                direct,
                            )
                        ),
                        "error_type": (
                            direct.get("error_type")
                            if usage_complete
                            else "MissingProviderUsage"
                        ),
                        "retry": max(
                            0,
                            attempts - (2 if job.scenario == "S2" else 1),
                        ),
                        "metrics": {
                            "e2e_ms": direct.get("latency_ms", {}).get("total"),
                            "ttft_reasoning_ms": direct.get("latency_ms", {}).get(
                                "ttft_reasoning",
                            ),
                            "ttft_answer_ms": direct.get("latency_ms", {}).get(
                                "ttft_answer",
                            ),
                            "model_attempts": attempts,
                            "input_tokens": _sum_usage(relay_events).get(
                                "prompt_tokens",
                                0,
                            ),
                            "output_tokens": _sum_usage(relay_events).get(
                                "completion_tokens",
                                0,
                            ),
                            "total_tokens": tokens,
                            "tool_calls": len(direct.get("tool_calls", [])),
                            "remote_budget_tokens_after": self.remote_budget.tokens,
                        },
                    },
                )
        except Exception as exc:
            self._record_failure(job, exc)

    def _direct_remote_body(self, scenario: Scenario) -> dict[str, Any]:
        model_id = self.settings.model.split("/", 1)[-1]
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": scenario.prompts[0]}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 128,
            "temperature": 0,
            "enable_thinking": False,
        }
        if scenario is S2:
            if not self._captured_tool_schema:
                raise RuntimeError("S2 direct baseline has no captured tool schema")
            body["tools"] = json.loads(json.dumps(self._captured_tool_schema))
            body["tool_choice"] = "auto"
        return body

    async def _calibrate_tool_schema(self) -> None:
        """Capture QwenPaw's exact S2 schema locally without a cloud call."""

        with start_mock_server(
            scenario="S2",
            run_id=self.settings.run_id,
            profile="schema_calibration",
            backend="mock-calibration",
        ) as server:
            with create_isolated_run(
                self.settings.seed_working_dir,
                self.settings.seed_secret_dir,
                f"{self.settings.run_id}-schema-calibration-{uuid.uuid4().hex[:8]}",
            ) as paths:
                profile = generate_profile(
                    paths,
                    "core",
                    model=self.settings.model,
                    base_url=server.base_url,
                )
                client = ACPBenchmarkClient(
                    ACPConfig(
                        command=[
                            str(self.settings.python_executable),
                            "-m",
                            "qwenpaw",
                            "acp",
                            "--local-diagnostics",
                        ],
                        cwd=str(paths.project_dir),
                        agent="default",
                        cpu_affinity=self.settings.cpu_affinity,
                        run_id=self.settings.run_id,
                        profile="schema_calibration",
                        scenario="S2",
                        backend="mock-calibration",
                        timeout_s=self.settings.startup_timeout_s,
                        available_commands_timeout_s=self.settings.startup_timeout_s,
                        env=profile.env,
                    ),
                )
                try:
                    await client.start()
                    await client.prompt(S2.prompts[0])
                finally:
                    await client.close()
                body = server.primary_request_body(1)
                tools = body.get("tools") if isinstance(body, dict) else None
                if not isinstance(tools, list) or not tools:
                    raise RuntimeError("QwenPaw emitted no S2 tool schema")
                self._captured_tool_schema = _shell_tool_schema(tools)
                if not self._captured_tool_schema:
                    raise RuntimeError(
                        "QwenPaw emitted no execute_shell_command schema",
                    )

    def _account_remote(
        self,
        events: Iterable[Mapping[str, Any]],
    ) -> tuple[int, int]:
        materialized = list(events)
        attempts = len(
            {
                data.get("request_index")
                for event in materialized
                if event.get("event") == "relay_upstream_request_start"
                and isinstance((data := event.get("data")), Mapping)
                and isinstance(data.get("request_index"), int)
            },
        )
        usage = _sum_usage(materialized)
        tokens = int(usage.get("total_tokens", 0))
        self.remote_budget.account(attempts=attempts, tokens=tokens)
        if attempts and tokens <= 0:
            # Provider usage is authoritative.  Without it the token stop
            # condition cannot be evaluated, so prevent any subsequent
            # remote request.
            self.remote_budget.tokens = self.remote_budget.max_tokens
        return attempts, tokens

    def _record_budget_skip(self, job: MatrixJob, reason: str) -> None:
        self.measurements.append(
            {
                "run_id": self.settings.run_id,
                "sample_id": job.sample_id,
                "profile": job.profile,
                "scenario": job.scenario,
                "backend": job.backend,
                "record_type": "measurement",
                "phase": job.phase,
                "repetition": job.repetition,
                "success": False,
                "error_type": "ApiBudgetExhausted",
                "retry": 0,
                "metrics": {},
                "skip_reason": reason,
            },
        )

    async def _run_acp_job(
        self,
        job: MatrixJob,
        *,
        scenario: Scenario | None,
        server: Any,
    ) -> dict[str, Any] | None:
        sample_id = job.sample_id
        sample_csv = self.result_dir / "raw_samples" / f"{sample_id}.csv"
        metadata = {
            "run_id": self.settings.run_id,
            "sample_id": sample_id,
            "profile": job.profile,
            "scenario": job.scenario,
            "backend": job.backend,
        }
        self.sample_files.append((sample_csv, metadata))
        isolation_id = f"{self.settings.run_id}-{sample_id}"
        sampler: ResourceSampler | None = None
        aggregate: dict[str, Any] = {}
        turn_results: list[dict[str, Any]] = []
        turn_resources: list[dict[str, int | None]] = []
        close_start_ns = None
        close_end_ns = None
        client: ACPBenchmarkClient | None = None

        with create_isolated_run(
            self.settings.seed_working_dir,
            self.settings.seed_secret_dir,
            isolation_id,
        ) as paths:
            profile = generate_profile(
                paths,
                job.profile,
                model=(
                    None
                    if job.profile == "stock_reference"
                    else self.settings.model
                ),
                base_url=server.base_url if server is not None else None,
            )
            before = _snapshot_layers(paths)
            config = ACPConfig(
                command=[
                    str(self.settings.python_executable),
                    "-m",
                    "qwenpaw",
                    "acp",
                    "--local-diagnostics",
                ],
                cwd=str(paths.project_dir),
                agent="default",
                cpu_affinity=self.settings.cpu_affinity,
                run_id=self.settings.run_id,
                profile=job.profile,
                scenario=job.scenario,
                backend=job.backend,
                timeout_s=self.settings.startup_timeout_s,
                available_commands_timeout_s=self.settings.startup_timeout_s,
                env=profile.env,
                stderr_path=paths.state_dir / "acp.stderr.log",
            )
            client = ACPBenchmarkClient(config)

            def process_started(pid: int) -> None:
                nonlocal sampler
                sampler = ResourceSampler(
                    pid,
                    self.settings.sample_interval_s,
                    sample_csv,
                ).start()

            error_type = None
            start_result = None
            try:
                start_result = await client.start(process_started)
                turn_resources.append(_tree_snapshot(start_result.pid))
                if scenario is None:
                    await client.idle(self.settings.idle_seconds)
                else:
                    prompts = scenario.prompts[: job.turns]
                    for prompt in prompts:
                        turn_results.append(await client.prompt(prompt))
                        await asyncio.sleep(
                            min(0.25, self.settings.sample_interval_s * 1.2),
                        )
                        turn_resources.append(_tree_snapshot(start_result.pid))
            except Exception as exc:
                error_type = type(exc).__name__
            finally:
                close_start_ns = time.monotonic_ns()
                await client.close()
                close_end_ns = time.monotonic_ns()
                if sampler is not None:
                    aggregate = sampler.stop()

            observed = aggregate.get("observed_processes", [])
            all_exited = wait_for_descendants_exit(observed, timeout=5.0)
            after = _snapshot_layers(paths)
            state_delta = _diff_layers(before, after)
            client_result = client.result().to_dict()
            self.events.extend(_tag_events(client_result["events"], job))

            ready_ns = client_result["timestamps_ns"].get(
                "available_commands",
            )
            launch_ns = client_result["timestamps_ns"].get("launch_start")
            connected_ns = client_result["timestamps_ns"].get("connected")
            resource_metrics = _resource_metrics(
                sample_csv,
                ready_ns=ready_ns,
                end_ns=close_start_ns,
            )
            wire_bodies = (
                server.request_bodies
                if server is not None and hasattr(server, "request_bodies")
                else {}
            )
            primary_wire_bodies = _primary_wire_bodies(wire_bodies)
            first_wire = primary_wire_bodies.get(1, {})
            wire_tool_names = _wire_tool_names(first_wire)
            configured_local_tools = {
                "read_file",
                "execute_shell_command",
                "get_current_time",
            }
            core_tool_schema_ok = (
                wire_tool_names == configured_local_tools
                if job.profile == "core" and scenario is not None
                else True
            )
            correctness = _correctness(
                scenario,
                turn_results,
                wire_bodies=primary_wire_bodies,
            )
            attempts = int(getattr(server, "request_count", 0)) if server else 0
            primary_attempts = len(primary_wire_bodies)
            auxiliary_attempts = max(0, attempts - primary_attempts)
            mock_validation_failures = int(
                getattr(server, "validation_failures", 0),
            ) if server else 0
            usage = _sum_usage(
                server.snapshot_events() if server is not None else [],
            )
            metrics: dict[str, Any] = {
                "startup_ready_ms": _elapsed_ms(launch_ns, ready_ns),
                "startup_connected_ms": _elapsed_ms(launch_ns, connected_ns),
                "exit_ms": _elapsed_ms(close_start_ns, close_end_ns),
                "available_commands_count": (
                    start_result.available_commands_count if start_result else 0
                ),
                "model_attempts": attempts,
                "primary_model_attempts": primary_attempts,
                "auxiliary_model_attempts": auxiliary_attempts,
                "mock_validation_failures": mock_validation_failures,
                "wire_tool_count": len(wire_tool_names),
                "wire_tool_names": sorted(wire_tool_names),
                "wire_dynamic_tool_count": len(
                    wire_tool_names - configured_local_tools,
                ),
                "core_tool_schema_exact": core_tool_schema_ok,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "tool_calls": sum(
                    len(turn.get("tool_calls", [])) for turn in turn_results
                ),
                "turn_count": len(turn_results),
                "state_apparent_net_delta_bytes": state_delta["delta"][
                    "apparent_bytes"
                ],
                "state_allocated_net_delta_bytes": state_delta["delta"][
                    "allocated_bytes"
                ],
                "state_file_growth": state_delta["delta"]["file_count"],
                **state_delta["activity"],
                "orphan_free": all_exited,
                **resource_metrics,
                **_sampler_summary_metrics(aggregate),
            }
            network_metrics = _network_metrics(
                sample_csv,
                prompt_start_ns=(
                    turn_results[0].get("timestamps_ns", {}).get("prompt_start")
                    if turn_results
                    else None
                ),
            )
            metrics.update(network_metrics)
            if job.backend == "dashscope" and server is not None:
                metrics["relay_upstream_host_policy_enforced"] = True
                metrics["relay_upstream_host_policy_enforced_count"] = 1
                metrics.update(
                    _relay_latency_metrics(
                        {"turn_results": turn_results},
                        server.snapshot_events(),
                        request_indices=_primary_wire_indexes(wire_bodies),
                    ),
                )
            if turn_results:
                metrics.update(_turn_latency_metrics(turn_results))
            if scenario is S3 and len(turn_resources) >= 2:
                metrics.update(
                    _growth_metrics(turn_resources[0], turn_resources[-1]),
                )
                metrics.update(_trend_metrics(turn_resources))
            elif len(turn_resources) >= 2:
                one_turn_growth = _growth_metrics(
                    turn_resources[0],
                    turn_resources[-1],
                )
                metrics.update(
                    {
                        "request_pss_delta_bytes": one_turn_growth[
                            "ten_turn_pss_growth_bytes"
                        ],
                        "request_uss_delta_bytes": one_turn_growth[
                            "ten_turn_uss_growth_bytes"
                        ],
                        "request_fd_delta": one_turn_growth["fd_growth"],
                        "request_thread_delta": one_turn_growth[
                            "thread_growth"
                        ],
                    },
                )
            network_ok = metrics.get("network_allowlist_ok") is not False
            success = bool(
                error_type is None
                and ready_ns is not None
                and all_exited
                and correctness
                and mock_validation_failures == 0
                and core_tool_schema_ok
                and network_ok
            )
            record = {
                **metadata,
                "record_type": "measurement",
                "phase": job.phase,
                "repetition": job.repetition,
                "success": success,
                "error_type": error_type,
                "retry": max(0, primary_attempts - _expected_attempts(job)),
                "metrics": metrics,
                "resource_aggregate": aggregate,
                "turn_resource_series": turn_resources,
                "state_delta": _state_delta_summary(state_delta),
                "profile_metadata": profile.metadata,
            }
            self.measurements.append(record)
            return record

    async def _run_mock_direct_replay(
        self,
        job: MatrixJob,
        scenario: Scenario,
        request_body: dict[str, Any] | None,
        paired_record: dict[str, Any],
    ) -> None:
        if request_body is None:
            return
        direct_profile = f"direct_wire_{job.profile}"
        direct_job = MatrixJob(
            phase="request",
            profile=direct_profile,
            scenario=job.scenario,
            backend="mock-direct",
            repetition=job.repetition,
            direct=False,
        )
        with start_mock_server(
            scenario=job.scenario,
            run_id=self.settings.run_id,
            profile=direct_profile,
            backend="mock-direct",
        ) as direct_server:
            direct = await asyncio.to_thread(
                run_request,
                base_url=direct_server.base_url,
                request_body=request_body,
                run_id=self.settings.run_id,
                profile=direct_profile,
                scenario=job.scenario,
                backend="mock-direct",
                trust_env=False,
                allow_exact_shell=job.scenario == "S2",
            )
            server_events = direct_server.snapshot_events()
            self.events.extend(_tag_events(direct.get("events", []), direct_job))
            self.events.extend(_tag_events(server_events, direct_job))
            self._record_wire_requests(
                direct_server.request_bodies,
                server_events,
                direct_job,
            )
            direct_total = direct.get("latency_ms", {}).get("total")
            qwen_total = paired_record["metrics"].get("e2e_ms")
            tax = (
                qwen_total - direct_total
                if isinstance(qwen_total, (int, float))
                and isinstance(direct_total, (int, float))
                else None
            )
            paired_record["metrics"]["direct_wire_e2e_ms"] = direct_total
            paired_record["metrics"]["mock_orchestration_tax_ms"] = tax
            paired_record["metrics"]["framework_path_overhead_ms"] = tax
            paired_record["metrics"]["framework_path_overhead_includes_aux_model"] = (
                int(
                    paired_record["metrics"].get(
                        "auxiliary_model_attempts",
                        0,
                    ),
                )
                > 0
            )
            paired_record["paired_direct_sample_id"] = direct_job.sample_id
            correctness = _direct_correctness(scenario, direct)
            mock_valid = direct_server.validation_failures == 0
            self.measurements.append(
                {
                    "run_id": self.settings.run_id,
                    "sample_id": direct_job.sample_id,
                    "paired_sample_id": job.sample_id,
                    "profile": direct_profile,
                    "scenario": job.scenario,
                    "backend": "mock-direct",
                    "record_type": "measurement",
                    "phase": "request",
                    "repetition": job.repetition,
                    "success": bool(
                        direct.get("success") and correctness and mock_valid
                    ),
                    "error_type": direct.get("error_type"),
                    "retry": 0,
                    "metrics": {
                        "e2e_ms": direct_total,
                        "ttft_reasoning_ms": direct.get("latency_ms", {}).get(
                            "ttft_reasoning"
                        ),
                        "ttft_answer_ms": direct.get("latency_ms", {}).get(
                            "ttft_answer"
                        ),
                        "model_attempts": direct_server.request_count,
                        "mock_validation_failures": (
                            direct_server.validation_failures
                        ),
                        "tool_calls": len(direct.get("tool_calls", [])),
                        **_sum_usage(server_events),
                    },
                },
            )

    def _record_wire_requests(
        self,
        bodies: Mapping[int, dict[str, Any]],
        events: Iterable[Mapping[str, Any]],
        job: MatrixJob,
    ) -> None:
        event_rows = list(events)
        usage_by_request = _usage_by_request(event_rows)
        event_metadata = {
            data["request_index"]: data
            for event in event_rows
            if event.get("event") in {
                "mock_request_received",
                "relay_request_received",
            }
            and isinstance((data := event.get("data")), Mapping)
            and isinstance(data.get("request_index"), int)
        }
        for request_index, body in sorted(bodies.items()):
            if (
                job.scenario == "S2"
                and job.profile == "core"
                and request_index == 1
                and self._captured_tool_schema is None
                and isinstance(body.get("tools"), list)
                and body["tools"]
            ):
                self._captured_tool_schema = _shell_tool_schema(body["tools"])
            prompt_tokens = usage_by_request.get(request_index, {}).get(
                "prompt_tokens",
            )
            provider_prompt_tokens = (
                prompt_tokens
                if any(
                    event.get("event") == "relay_usage"
                    and isinstance(event.get("data"), Mapping)
                    and event["data"].get("request_index") == request_index
                    for event in event_rows
                )
                else None
            )
            request_kind = classify_request_kind(body)
            source_metadata = event_metadata.get(request_index, {})
            self.request_records.append(
                {
                    "schema_version": "1",
                    "run_id": self.settings.run_id,
                    "profile": job.profile,
                    "scenario": job.scenario,
                    "backend": job.backend,
                    "repetition": job.repetition,
                    "sample_id": job.sample_id,
                    "request_index": request_index,
                    "request_kind": request_kind,
                    "kind_request_index": source_metadata.get(
                        "kind_request_index",
                    ),
                    "usage_source": (
                        "provider" if provider_prompt_tokens is not None else "estimate"
                    ),
                    **analyze_request(
                        body,
                        provider_prompt_tokens=provider_prompt_tokens,
                    ),
                },
            )

    def _record_failure(self, job: MatrixJob, exc: BaseException) -> None:
        self.measurements.append(
            {
                "run_id": self.settings.run_id,
                "sample_id": job.sample_id,
                "profile": job.profile,
                "scenario": job.scenario,
                "backend": job.backend,
                "record_type": "measurement",
                "phase": job.phase,
                "repetition": job.repetition,
                "success": False,
                "error_type": type(exc).__name__,
                "retry": 0,
                "metrics": {},
            },
        )

    def finalize(self, *, completed: bool = True) -> list[Any]:
        """Write all public artifacts and fail closed on detected secrets."""

        _validate_unique_measurement_ids(self.measurements)

        _write_footprint_csv(
            self.result_dir / "footprint.csv",
            self.footprint_records,
            self.distribution_records,
            self.shared_object_records,
        )
        _merge_sample_files(self.result_dir / "samples.csv", self.sample_files)
        with (self.result_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in sorted(
                self.events,
                key=lambda row: int(row.get("monotonic_ns", 0)),
            ):
                stream.write(
                    json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                )
        with (self.result_dir / "requests.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for request in self.request_records:
                stream.write(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                )
        with (self.result_dir / "measurements.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for measurement in self.measurements:
                stream.write(
                    json.dumps(
                        measurement,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n",
                )

        summaries = summarize_records(self.measurements)
        write_summary_csv(self.result_dir / "summary.csv", summaries)
        core_metrics = _extract_core_metrics(
            self.measurements,
            self.footprint_records,
            self.request_records,
        )
        growth_flags = _growth_flags(self.measurements)
        gates = evaluate_core_gates(core_metrics, **growth_flags)
        trim_decisions = _build_trim_decisions(
            self.measurements,
            self.request_records,
            self.distribution_records,
        )

        manifest_path = self.result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "scanning" if completed else "partial"
        manifest["remote_budget"] = {
            "attempts": self.remote_budget.attempts,
            "tokens": self.remote_budget.tokens,
            "max_attempts": self.remote_budget.max_attempts,
            "max_tokens": self.remote_budget.max_tokens,
        }
        manifest["arm_resolver"] = (
            self.arm_result.to_dict() if self.arm_result is not None else None
        )
        atomic_write_json(manifest_path, manifest)

        findings = scan_for_secrets(
            [self.result_dir],
            environment=os.environ,
        )
        report = render_markdown_report(
            summaries,
            core_gates=gates,
            footprint=self.footprint_records,
            distributions=self.distribution_records,
            shared_objects=self.shared_object_records,
            trim_decisions=trim_decisions,
            arm_result=self.arm_result,
            secret_findings=findings,
        )
        report += _decision_section(
            gates,
            self.measurements,
            self.arm_result,
            request_records=self.request_records,
            footprints=self.footprint_records,
        )
        (self.result_dir / "report.md").write_text(report, encoding="utf-8")

        manifest["status"] = (
            "secret_scan_failed"
            if findings
            else "completed" if completed else "partial"
        )
        manifest["secret_finding_count"] = len(findings)
        atomic_write_json(manifest_path, manifest)
        final_findings = scan_for_secrets(
            [self.result_dir],
            environment=os.environ,
        )
        if final_findings and not findings:
            manifest["status"] = "secret_scan_failed"
            manifest["secret_finding_count"] = len(final_findings)
            atomic_write_json(manifest_path, manifest)
        if final_findings:
            raise RuntimeError(
                "result secret scan failed with "
                f"{len(final_findings)} finding(s)",
            )
        return gates


def _elapsed_ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) / 1_000_000


def _validate_unique_measurement_ids(
    measurements: Iterable[Mapping[str, Any]],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in measurements:
        if row.get("record_type") != "measurement":
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every measurement row must have a sample_id")
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(
            "duplicate measurement sample_id(s): " + ", ".join(sorted(duplicates)),
        )


def _tag_events(
    events: Iterable[Mapping[str, Any]],
    job: MatrixJob,
) -> list[dict[str, Any]]:
    return [
        {
            **dict(event),
            "sample_id": job.sample_id,
            "repetition": job.repetition,
            "phase": job.phase,
        }
        for event in events
    ]


def _shell_tool_schema(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") == (
            "execute_shell_command"
        ):
            selected.append(json.loads(json.dumps(tool)))
    return selected


def _wire_tool_names(body: Mapping[str, Any]) -> set[str]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(
            function.get("name"),
            str,
        ):
            names.add(str(function["name"]))
    return names


def _primary_wire_bodies(
    bodies: Mapping[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    primary: dict[int, dict[str, Any]] = {}
    for _wire_index, body in sorted(bodies.items()):
        if classify_request_kind(body) != "primary":
            continue
        primary[len(primary) + 1] = body
    return primary


def _primary_wire_indexes(
    bodies: Mapping[int, dict[str, Any]],
) -> set[int]:
    return {
        wire_index
        for wire_index, body in bodies.items()
        if classify_request_kind(body) == "primary"
    }


def _expected_attempts(job: MatrixJob) -> int:
    if job.phase != "request":
        return 0
    return job.turns * (2 if job.scenario == "S2" else 1)


def _correctness(
    scenario: Scenario | None,
    turns: list[dict[str, Any]],
    *,
    wire_bodies: Mapping[int, Mapping[str, Any]] | None = None,
) -> bool:
    if scenario is None:
        return True
    if len(turns) != len(scenario.prompts):
        return False
    if scenario is S1:
        return turns[0].get("answer_text", "").strip() == S1.expected_text
    if scenario is S2:
        answer = turns[0].get("answer_text", "").strip()
        tools = turns[0].get("tool_calls", [])
        permissions = turns[0].get("permission_requests", [])
        expected_hash = hashlib.sha256(
            S2.expected_command.encode("utf-8"),
        ).hexdigest()
        bodies = wire_bodies or {}
        second = bodies.get(2, {})
        messages = second.get("messages") if isinstance(second, Mapping) else None
        exact_tool_result = bool(
            isinstance(messages, list)
            and any(
                isinstance(message, Mapping)
                and message.get("role") == "tool"
                and str(message.get("content") or "").strip()
                == S2.expected_text
                for message in messages
            )
        )
        return bool(
            answer == S2.expected_text
            and len(tools) == 1
            and tools[0].get("command_sha256") == expected_hash
            and len(permissions) == 1
            and permissions[0].get("approved") is True
            and exact_tool_result
        )
    if scenario is S3:
        expected = ["STORED."] + [
            f"K{turn:02d}=V{turn * 7919 % 100000:05d};STORED."
            for turn in range(1, 10)
        ]
        return [turn.get("answer_text", "").strip() for turn in turns] == expected
    return False


def _direct_correctness(scenario: Scenario, result: Mapping[str, Any]) -> bool:
    answer = str(result.get("answer_text") or "").strip()
    if scenario is S1:
        return answer == S1.expected_text
    if scenario is S2:
        return answer == S2.expected_text and len(result.get("tool_calls", [])) == 1
    return bool(answer)


def _sum_usage(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in _usage_by_request(events).values():
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return totals


def _usage_by_request(
    events: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for event in events:
        if str(event.get("event")) not in {"relay_usage", "mock_usage"}:
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        index = data.get("request_index")
        if not isinstance(index, int):
            continue
        usage = result.setdefault(index, {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
    return result


def _turn_latency_metrics(turns: list[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "ttft_reasoning_ms": [],
        "ttft_answer_ms": [],
        "e2e_ms": [],
    }
    mapping = {
        "ttft_reasoning_ms": "ttft_reasoning",
        "ttft_answer_ms": "ttft_answer",
        "e2e_ms": "total",
    }
    for turn in turns:
        latency = turn.get("latency_ms")
        if not isinstance(latency, Mapping):
            continue
        for output_name, source_name in mapping.items():
            value = latency.get(source_name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[output_name].append(float(value))
    return {
        name: statistics.median(items) if items else None
        for name, items in values.items()
    }


def _relay_latency_metrics(
    context: Mapping[str, Any],
    relay_events: Iterable[Mapping[str, Any]],
    *,
    request_indices: set[int] | None = None,
) -> dict[str, float | None]:
    """Decompose one remote ACP turn using monotonic relay milestones."""

    turns = context.get("turn_results")
    if not isinstance(turns, list) or not turns:
        return {}
    turn = turns[0]
    if not isinstance(turn, Mapping):
        return {}
    timestamps = turn.get("timestamps_ns")
    if not isinstance(timestamps, Mapping):
        return {}

    by_name: dict[str, list[int]] = {}
    for event in relay_events:
        if request_indices is not None:
            data = event.get("data")
            request_index = (
                data.get("request_index") if isinstance(data, Mapping) else None
            )
            if request_index not in request_indices:
                continue
        name = str(event.get("event") or "")
        value = event.get("monotonic_ns")
        if isinstance(value, int):
            by_name.setdefault(name, []).append(value)

    def earliest(*names: str) -> int | None:
        values = [value for name in names for value in by_name.get(name, [])]
        return min(values) if values else None

    def latest(*names: str) -> int | None:
        values = [value for name in names for value in by_name.get(name, [])]
        return max(values) if values else None

    acp_submit = timestamps.get("prompt_start")
    acp_first = min(
        value
        for key in ("first_reasoning", "first_answer", "first_tool_call")
        if isinstance((value := timestamps.get(key)), int)
    ) if any(
        isinstance(timestamps.get(key), int)
        for key in ("first_reasoning", "first_answer", "first_tool_call")
    ) else None
    acp_end = timestamps.get("turn_end")
    relay_received = earliest("relay_request_received")
    upstream_start = earliest("relay_upstream_request_start")
    upstream_first = earliest(
        "relay_first_reasoning",
        "relay_first_answer",
        "relay_first_tool_call",
    )
    upstream_end = latest("relay_upstream_done", "relay_response_complete")
    return {
        "framework_preprocess_ms": _elapsed_ms(acp_submit, relay_received),
        "model_side_ttft_ms": _elapsed_ms(upstream_start, upstream_first),
        "relay_to_acp_first_ms": _elapsed_ms(upstream_first, acp_first),
        "framework_postprocess_ms": _elapsed_ms(upstream_end, acp_end),
    }


def _sampler_summary_metrics(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    tree = aggregate.get("tree_summary")
    tree = tree if isinstance(tree, Mapping) else {}
    peak = aggregate.get("tree_peak")
    peak = peak if isinstance(peak, Mapping) else {}

    def stat(metric: str, name: str) -> Any:
        value = tree.get(metric)
        return value.get(name) if isinstance(value, Mapping) else None

    totals = aggregate.get("totals")
    totals = totals if isinstance(totals, Mapping) else {}
    result: dict[str, Any] = {
        "tree_sample_count": aggregate.get("sample_count"),
        "tree_socket_scan_count": aggregate.get("socket_scan_count"),
        "tree_duration_s": aggregate.get("duration_s"),
        "tree_rss_mean_bytes": stat("rss_bytes", "mean"),
        "tree_rss_p95_bytes": stat("rss_bytes", "p95"),
        "tree_rss_max_bytes": stat("rss_bytes", "max"),
        "tree_pss_mean_bytes": stat("pss_bytes", "mean"),
        "tree_pss_p95_bytes": stat("pss_bytes", "p95"),
        "tree_pss_max_bytes": stat("pss_bytes", "max"),
        "tree_uss_mean_bytes": stat("uss_bytes", "mean"),
        "tree_uss_p95_bytes": stat("uss_bytes", "p95"),
        "tree_uss_max_bytes": stat("uss_bytes", "max"),
        "tree_cpu_avg_core_pct": stat("cpu_percent_single_core", "mean"),
        "tree_cpu_p95_core_pct": stat("cpu_percent_single_core", "p95"),
        "tree_cpu_max_core_pct": stat("cpu_percent_single_core", "max"),
        "tree_peak_process_count": peak.get("processes"),
        "tree_peak_rss_bytes": peak.get("rss_bytes"),
        "tree_peak_pss_bytes": peak.get("pss_bytes"),
        "tree_peak_uss_bytes": peak.get("uss_bytes"),
        "tree_peak_swap_pss_bytes": peak.get("swap_pss_bytes"),
        "tree_peak_threads": peak.get("threads"),
        "tree_peak_fds": peak.get("num_fds"),
        "smaps_coverage_min": aggregate.get("smaps_coverage_min"),
        "tree_nonempty_sample_count": aggregate.get("nonempty_sample_count"),
        "tree_empty_sample_count": aggregate.get("empty_sample_count"),
        "cpu_user_s": totals.get("cpu_user_s"),
        "cpu_system_s": totals.get("cpu_system_s"),
        "cpu_total_s": (
            float(totals.get("cpu_user_s", 0))
            + float(totals.get("cpu_system_s", 0))
            if isinstance(totals.get("cpu_user_s"), (int, float))
            and isinstance(totals.get("cpu_system_s"), (int, float))
            else None
        ),
        "read_bytes": totals.get("read_bytes"),
        "write_bytes": totals.get("write_bytes"),
        "ctx_voluntary": totals.get("ctx_voluntary"),
        "ctx_involuntary": totals.get("ctx_involuntary"),
        "sampler_error_count": len(aggregate.get("errors", [])),
    }

    roles = aggregate.get("roles")
    if isinstance(roles, Mapping):
        role_fields = (
            "peak_processes",
            "peak_rss_bytes",
            "peak_pss_bytes",
            "peak_uss_bytes",
            "peak_swap_pss_bytes",
            "peak_threads",
            "peak_num_fds",
            "observed_process_count",
            "observed_cpu_user_s",
            "observed_cpu_system_s",
            "observed_read_bytes",
            "observed_write_bytes",
        )
        for role, raw_stats in sorted(roles.items()):
            if not isinstance(raw_stats, Mapping):
                continue
            safe_role = "".join(
                character if character.isalnum() else "_"
                for character in str(role)
            )
            for field_name in role_fields:
                result[f"role_{safe_role}_{field_name}"] = raw_stats.get(
                    field_name,
                )

    cgroup = aggregate.get("cgroup_delta")
    if isinstance(cgroup, Mapping):
        cgroup_start = aggregate.get("cgroup_start")
        cgroup_start = (
            cgroup_start if isinstance(cgroup_start, Mapping) else {}
        )
        member_count = cgroup_start.get("member_count")
        exclusive = bool(
            cgroup_start.get("available") is True and member_count == 1
        )
        result["cgroup_start_member_count"] = member_count
        result["cgroup_exclusive"] = exclusive
        result["cgroup_exclusive_count"] = 1 if exclusive else 0
        prefix = "cgroup_exclusive" if exclusive else "shared_cgroup_crosscheck"
        cpu_stat = cgroup.get("cpu_stat")
        io_stat = cgroup.get("io_stat")
        memory_events = cgroup.get("memory_events")
        for source, output_name in (
            (cpu_stat, f"{prefix}_cpu"),
            (io_stat, f"{prefix}_io"),
            (memory_events, f"{prefix}_memory_event"),
        ):
            if not isinstance(source, Mapping):
                continue
            for name, value in sorted(source.items()):
                result[f"{output_name}_{name}_delta"] = value
        result[f"{prefix}_memory_current_bytes_delta"] = cgroup.get(
            "memory_current_bytes",
        )
        result[f"{prefix}_memory_peak_bytes_delta"] = cgroup.get(
            "memory_peak_bytes",
        )
        result[f"{prefix}_pids_current_delta"] = cgroup.get("pids_current")
    return result


def _tree_snapshot(root_pid: int) -> dict[str, int | None]:
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"pss_bytes": None, "uss_bytes": None, "fds": None, "threads": None}
    pss = 0
    uss = 0
    fds = 0
    threads = 0
    memory_complete = True
    for process in processes:
        try:
            memory = process.memory_full_info()
            pss += int(memory.pss)
            uss += int(memory.uss)
            fds += process.num_fds()
            threads += process.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            memory_complete = False
    return {
        "pss_bytes": pss if memory_complete else None,
        "uss_bytes": uss if memory_complete else None,
        "fds": fds,
        "threads": threads,
    }


def _growth_metrics(
    first: Mapping[str, int | None],
    last: Mapping[str, int | None],
) -> dict[str, int | None]:
    def delta(key: str) -> int | None:
        before = first.get(key)
        after = last.get(key)
        if before is None or after is None:
            return None
        return int(after) - int(before)

    return {
        "ten_turn_pss_growth_bytes": delta("pss_bytes"),
        "ten_turn_uss_growth_bytes": delta("uss_bytes"),
        "fd_growth": delta("fds"),
        "thread_growth": delta("threads"),
    }


def _trend_metrics(
    series: list[Mapping[str, int | None]],
) -> dict[str, Any]:
    """Describe ten-turn growth and flag only sustained, material trends."""

    def numeric(key: str) -> list[float]:
        return [
            float(value)
            for row in series
            if isinstance((value := row.get(key)), (int, float))
            and not isinstance(value, bool)
        ]

    pss = numeric("pss_bytes")
    slope = None
    r_squared = None
    positive_ratio = None
    linear = False
    if len(pss) >= 6:
        x_values = list(range(len(pss)))
        x_mean = statistics.fmean(x_values)
        y_mean = statistics.fmean(pss)
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = (
            sum(
                (x - x_mean) * (y - y_mean)
                for x, y in zip(x_values, pss, strict=True)
            )
            / denominator
            if denominator
            else 0.0
        )
        fitted = [y_mean + slope * (x - x_mean) for x in x_values]
        total_variance = sum((y - y_mean) ** 2 for y in pss)
        residual = sum(
            (y - fit) ** 2 for y, fit in zip(pss, fitted, strict=True)
        )
        r_squared = (
            max(0.0, 1.0 - residual / total_variance)
            if total_variance
            else 0.0
        )
        deltas = [after - before for before, after in zip(pss, pss[1:])]
        positive_ratio = sum(delta > 0 for delta in deltas) / len(deltas)
        linear = bool(
            pss[-1] - pss[0] >= 1024 * 1024
            and slope > 0
            and r_squared >= 0.8
            and positive_ratio >= 0.8
        )

    def sustained(key: str) -> bool:
        values = numeric(key)
        if len(values) < 4:
            return False
        tail = values[-3:]
        return bool(
            all(value > values[0] for value in tail)
            and all(after >= before for before, after in zip(tail, tail[1:]))
        )

    return {
        "turn_pss_slope_bytes_per_turn": slope,
        "turn_pss_linear_r_squared": r_squared,
        "turn_pss_positive_delta_ratio": positive_ratio,
        "ten_turn_pss_linear_growth": linear,
        "fd_sustained_growth": sustained("fds"),
        "thread_sustained_growth": sustained("threads"),
    }


def _resource_metrics(
    path: Path,
    *,
    ready_ns: int | None,
    end_ns: int | None,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = int(row["monotonic_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if ready_ns is not None and timestamp < ready_ns:
                continue
            if end_ns is not None and timestamp > end_ns:
                continue
            grouped.setdefault(timestamp, []).append(row)
    pss_values: list[float] = []
    uss_values: list[float] = []
    cpu_values: list[float] = []
    process_values: list[float] = []
    for rows in grouped.values():
        pss = _complete_row_sum(rows, "pss_bytes")
        uss = _complete_row_sum(rows, "uss_bytes")
        cpu = _complete_row_sum(rows, "cpu_percent_single_core")
        if pss is not None:
            pss_values.append(pss)
        if uss is not None:
            uss_values.append(uss)
        if cpu is not None:
            cpu_values.append(cpu)
        process_values.append(float(len(rows)))
    return {
        "idle_tree_pss_median_bytes": _median(pss_values),
        "idle_tree_pss_peak_bytes": max(pss_values) if pss_values else None,
        "idle_tree_uss_median_bytes": _median(uss_values),
        "idle_cpu_avg_core_pct": (
            statistics.fmean(cpu_values) if cpu_values else None
        ),
        "idle_cpu_p95_core_pct": (
            _p95(cpu_values) if len(cpu_values) >= 20 else None
        ),
        "idle_process_count_median": _median(process_values),
        "idle_sample_count": len(grouped),
    }


def _endpoint_is_loopback(endpoint: str) -> bool:
    host = endpoint.rsplit(":", 1)[0].strip("[]")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _network_metrics(
    path: Path,
    *,
    prompt_start_ns: int | None,
) -> dict[str, Any]:
    non_loopback_before: set[str] = set()
    non_loopback_any: set[str] = set()
    loopback_any: set[str] = set()
    sample_timestamps: set[int] = set()
    socket_scan_timestamps: set[int] = set()
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    timestamp = int(row.get("monotonic_ns") or 0)
                except ValueError:
                    continue
                sample_timestamps.add(timestamp)
                if str(row.get("socket_scan", "")).casefold() in {
                    "1",
                    "true",
                    "yes",
                }:
                    socket_scan_timestamps.add(timestamp)
                for endpoint in filter(None, row.get("remote_endpoints", "").split(";")):
                    if _endpoint_is_loopback(endpoint):
                        loopback_any.add(endpoint)
                    else:
                        non_loopback_any.add(endpoint)
                        if prompt_start_ns is None or timestamp < prompt_start_ns:
                            non_loopback_before.add(endpoint)
    observed_violation = bool(non_loopback_any)
    return {
        "pre_request_non_loopback_count": len(non_loopback_before),
        "runtime_non_loopback_destination_count": len(non_loopback_any),
        "runtime_loopback_destination_count": len(loopback_any),
        "network_sample_count": len(sample_timestamps),
        "network_socket_scan_sample_count": len(socket_scan_timestamps),
        "network_observed_allowlist_ok": not observed_violation,
        "network_observed_violation": observed_violation,
        # A clean sampled trace is useful evidence, but it is not a firewall
        # or packet capture.  Only an observed violation is a definitive
        # allowlist failure.
        "network_allowlist_ok": False if observed_violation else None,
        "network_observation_complete": False,
        "network_observation_method": "sampled_process_inet_sockets",
    }


def _snapshot_layers(paths: Any) -> dict[str, Any]:
    roots = {
        "working": paths.working_dir,
        "secret": paths.secret_dir,
        "state": paths.state_dir,
        "cache": paths.cache_dir,
        "home": paths.home_dir,
        "temp": paths.temp_dir,
        "backups": paths.backup_dir,
        "project": paths.project_dir,
        "xdg_config": paths.root / "config",
        "xdg_data": paths.root / "data",
    }
    return {name: snapshot_state(path) for name, path in roots.items()}


def _diff_layers(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    layers = {
        name: diff_state(before[name], after[name])
        for name in sorted(before.keys() & after.keys())
    }
    keys = (
        "file_count",
        "directory_count",
        "entry_count",
        "apparent_bytes",
        "allocated_bytes",
    )
    total = {
        key: sum(int(value.get("delta", {}).get(key, 0)) for value in layers.values())
        for key in keys
    }
    activity = {
        "state_positive_apparent_bytes": 0,
        "state_reclaimed_apparent_bytes": 0,
        "state_positive_allocated_bytes": 0,
        "state_reclaimed_allocated_bytes": 0,
        "state_created_paths": 0,
        "state_removed_paths": 0,
        "state_modified_paths": 0,
    }
    for layer in layers.values():
        activity["state_created_paths"] += len(layer.get("added", []))
        activity["state_removed_paths"] += len(layer.get("removed", []))
        activity["state_modified_paths"] += len(layer.get("modified", []))
        for detail in layer.get("details", {}).values():
            before_item = detail.get("before") or {}
            after_item = detail.get("after") or {}
            for field, positive_key, reclaimed_key in (
                (
                    "size_bytes",
                    "state_positive_apparent_bytes",
                    "state_reclaimed_apparent_bytes",
                ),
                (
                    "allocated_bytes",
                    "state_positive_allocated_bytes",
                    "state_reclaimed_allocated_bytes",
                ),
            ):
                delta = int(after_item.get(field, 0)) - int(
                    before_item.get(field, 0),
                )
                if delta >= 0:
                    activity[positive_key] += delta
                else:
                    activity[reclaimed_key] += -delta
    return {"delta": total, "activity": activity, "layers": layers}


def _complete_row_sum(rows: list[dict[str, str]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in {None, "", "None"}:
            return None
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None
    return sum(values)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _state_delta_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("layers"), Mapping):
        return {
            "delta": value.get("delta", {}),
            "activity": value.get("activity", {}),
            "layers": {
                name: _state_delta_summary(layer)
                for name, layer in value["layers"].items()
                if isinstance(layer, Mapping)
            },
        }
    return {
        "delta": value.get("delta", {}),
        "added": value.get("added", []),
        "removed": value.get("removed", []),
        "modified": value.get("modified", []),
        "details": value.get("details", {}),
        "changed_path_count": value.get("changed_path_count", 0),
    }


def _merge_sample_files(
    output: Path,
    files: Iterable[tuple[Path, Mapping[str, str]]],
) -> None:
    rows: list[dict[str, Any]] = []
    for path, metadata in files:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append({**metadata, **row})
    write_csv(output, rows)


def _write_footprint_csv(
    path: Path,
    footprints: Iterable[Any],
    distributions: Iterable[Any],
    shared_objects: Iterable[Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for record in footprints:
        rows.append({"record_type": "target", **record.to_dict()})
    for record in distributions:
        rows.append({"record_type": "distribution", **record.to_dict()})
    for record in shared_objects:
        rows.append({"record_type": "shared_object", **record.to_dict()})
    write_csv(path, rows)


def _extract_core_metrics(
    measurements: Iterable[Mapping[str, Any]],
    footprints: Iterable[Any],
    request_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    return _extract_profile_metrics(
        "core",
        measurements,
        footprints,
        request_records,
    )


def _extract_profile_metrics(
    profile: str,
    measurements: Iterable[Mapping[str, Any]],
    footprints: Iterable[Any],
    request_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        record
        for record in measurements
        if record.get("profile") == profile and record.get("success") is True
    ]

    def metric(name: str, *, scenario: str | None = None) -> float | None:
        values = []
        for record in selected:
            if scenario is not None and record.get("scenario") != scenario:
                continue
            metrics = record.get("metrics")
            value = metrics.get(name) if isinstance(metrics, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return statistics.median(values) if values else None

    # A future minimal deployment package does not exist yet.  The current
    # full venv remains visible in footprint.csv as an upper bound, but must
    # not be mislabeled as the pruned package this gate is defined for.
    package = next(
        (
            getattr(footprint, "allocated_bytes", None)
            for footprint in footprints
            if getattr(footprint, "category", "")
            == "install.future_minimal_deployment"
        ),
        None,
    )
    return {
        # Current venv is an upper bound; the report must not call it a future
        # minimal package until an actual pruned artifact exists.
        "package_bytes": package,
        "framework_idle_pss_bytes": metric(
            "idle_tree_pss_median_bytes",
            scenario="idle",
        ),
        "idle_cpu_avg_core_pct": metric("idle_cpu_avg_core_pct", scenario="idle"),
        "idle_cpu_p95_core_pct": metric("idle_cpu_p95_core_pct", scenario="idle"),
        "wsl_ready_seconds": (
            metric("startup_ready_ms", scenario="idle") / 1000
            if metric("startup_ready_ms", scenario="idle") is not None
            else None
        ),
        "arm_ready_seconds": None,
        # S2 may contain an additional Guard/governance model request that the
        # direct two-round tool baseline does not make.  Only S1 is eligible
        # for the host-framework tax gate; S2 remains visible as the broader
        # framework-path overhead in the detailed metrics.
        "mock_tax_ms": metric("mock_orchestration_tax_ms", scenario="S1"),
        "framework_context_tokens": _median_request_metric(
            (
                record
                for record in request_records
                if record.get("profile") == profile
                and record.get("scenario") == "S1"
                and record.get("backend") == "mock"
                and record.get("request_index") == 1
            ),
            "framework_fixed_tokens_estimate",
        ),
        "ten_turn_pss_growth_bytes": metric(
            "ten_turn_pss_growth_bytes",
            scenario="S3",
        ),
        "fd_growth": metric("fd_growth", scenario="S3"),
        "thread_growth": metric("thread_growth", scenario="S3"),
    }


def _median_request_metric(
    records: Iterable[Mapping[str, Any]],
    name: str,
) -> float | None:
    values = [
        float(record[name])
        for record in records
        if isinstance(record.get(name), (int, float))
        and not isinstance(record.get(name), bool)
    ]
    return statistics.median(values) if values else None


def _growth_flags(
    measurements: Iterable[Mapping[str, Any]],
) -> dict[str, bool]:
    metrics = [
        row.get("metrics")
        for row in measurements
        if row.get("profile") == "core" and row.get("scenario") == "S3"
    ]

    def any_true(name: str) -> bool:
        return any(
            isinstance(value, Mapping) and value.get(name) is True
            for value in metrics
        )

    return {
        "linear_memory_growth": any_true("ten_turn_pss_linear_growth"),
        "sustained_fd_growth": any_true("fd_sustained_growth"),
        "sustained_thread_growth": any_true("thread_sustained_growth"),
    }


def _build_trim_decisions(
    measurements: Iterable[Mapping[str, Any]],
    requests: Iterable[Mapping[str, Any]],
    distributions: Iterable[Any] = (),
) -> list[Any]:
    """Turn the cumulative profile matrix into conservative trim evidence."""

    rows = list(measurements)
    request_rows = list(requests)
    comparisons = (
        ("skills", "full", "no_skills"),
        ("non_core_tools", "no_skills", "local_tools"),
        ("optional_services_memory_extra_agents", "local_tools", "core"),
    )

    def values(profile: str, scenario: str, metric: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            if (
                row.get("profile") != profile
                or row.get("scenario") != scenario
                or row.get("success") is not True
            ):
                continue
            expected_backend = "none" if scenario == "idle" else "mock"
            if row.get("backend") != expected_backend:
                continue
            metrics = row.get("metrics")
            value = metrics.get(metric) if isinstance(metrics, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result.append(float(value))
        return result

    def median(profile: str, scenario: str, metric: str) -> float | None:
        found = values(profile, scenario, metric)
        return statistics.median(found) if found else None

    def prompt(profile: str) -> float | None:
        return _median_request_metric(
            (
                row
                for row in request_rows
                if row.get("profile") == profile
                and row.get("scenario") == "S1"
                and row.get("backend") == "mock"
                and row.get("request_index") == 1
            ),
            "framework_fixed_tokens_estimate",
        )

    def request_rows_for(profile: str) -> list[Mapping[str, Any]]:
        return [
            row
            for row in rows
            if row.get("profile") == profile
            and row.get("scenario") in {"S1", "S2"}
            and row.get("backend") == "mock"
            and row.get("record_type") == "measurement"
        ]

    decisions = []
    for component, baseline, variant in comparisons:
        baseline_rows = request_rows_for(baseline)
        variant_rows = request_rows_for(variant)
        if not baseline_rows or not variant_rows:
            continue
        baseline_success = (
            sum(row.get("success") is True for row in baseline_rows)
            / len(baseline_rows)
            if baseline_rows
            else 0.0
        )
        variant_success = (
            sum(row.get("success") is True for row in variant_rows)
            / len(variant_rows)
            if variant_rows
            else 0.0
        )
        expected_tool_counts = {"S1": 0, "S2": 1}
        unused: bool | None = bool(
            baseline_rows
            and all(row.get("success") is True for row in baseline_rows)
            and all(
                row.get("metrics", {}).get("tool_calls")
                == expected_tool_counts.get(str(row.get("scenario")))
                for row in baseline_rows
            )
        )
        if component == "optional_services_memory_extra_agents":
            # Absence of a tool call cannot prove that background memory or
            # service machinery was unused; keep this candidate conservative.
            unused = None
        pss_before = median(baseline, "idle", "idle_tree_pss_median_bytes")
        pss_after = median(variant, "idle", "idle_tree_pss_median_bytes")
        ready_before = median(baseline, "idle", "startup_ready_ms")
        ready_after = median(variant, "idle", "startup_ready_ms")
        e2e_before = median(baseline, "S1", "e2e_ms")
        e2e_after = median(variant, "S1", "e2e_ms")
        latency_savings = max(
            [
                0.0,
                *(
                before - after
                for before, after in (
                    (ready_before, ready_after),
                    (e2e_before, e2e_after),
                )
                if before is not None and after is not None
                ),
            ],
        )
        prompt_before = prompt(baseline)
        prompt_after = prompt(variant)
        evidence = TrimEvidence(
            component=component,
            unused_in_s1_s2=unused if baseline_rows else None,
            baseline_runs=min(
                (sum(row.get("scenario") == name for row in baseline_rows) for name in ("S1", "S2")),
                default=0,
            ),
            variant_runs=min(
                (sum(row.get("scenario") == name for row in variant_rows) for name in ("S1", "S2")),
                default=0,
            ),
            idle_pss_savings_bytes=int(
                max(0.0, (pss_before or 0.0) - (pss_after or 0.0)),
            ),
            fixed_prompt_token_savings=int(
                max(0.0, (prompt_before or 0.0) - (prompt_after or 0.0)),
            ),
            latency_savings_ms=latency_savings,
            success_rate_drop_pp=max(
                0.0,
                (baseline_success - variant_success) * 100.0,
            ),
            failure_rate_increase_pp=max(
                0.0,
                (baseline_success - variant_success) * 100.0,
            ),
        )
        decisions.append(evaluate_trim_candidate(evidence))
    for distribution in sorted(
        distributions,
        key=lambda item: int(getattr(item, "allocated_bytes", 0)),
        reverse=True,
    ):
        allocated = int(getattr(distribution, "allocated_bytes", 0))
        if allocated < 20 * 1024 * 1024:
            continue
        distribution_name = str(
            getattr(distribution, "name", "unknown"),
        )
        normalized_name = distribution_name.casefold().replace("_", "-")
        evidence = TrimEvidence(
            component=f"distribution:{distribution_name}",
            feature_used=None,
            unused_in_s1_s2=None,
            required=normalized_name == "qwenpaw",
            baseline_runs=0,
            variant_runs=0,
            # This is the distribution's measured allocated footprint, an
            # upper bound until a dependency-pruned package is actually built.
            disk_savings_bytes=allocated,
        )
        decisions.append(evaluate_trim_candidate(evidence))
    return decisions


def _decision_section(
    gates: Iterable[Any],
    measurements: Iterable[Mapping[str, Any]],
    arm_result: Any,
    *,
    request_records: Iterable[Mapping[str, Any]] = (),
    footprints: Iterable[Any] = (),
) -> str:
    """Append the protocol's four-way interpretation without hiding unknowns."""

    gate_map = {str(gate.metric): str(gate.status.value) for gate in gates}
    measured = {
        key: value
        for key, value in gate_map.items()
        if key not in {"package_bytes", "arm_ready_seconds"}
        and value != "unknown"
    }
    measurement_rows = list(measurements)
    request_rows = list(request_records)
    full_gates = evaluate_core_gates(
        _extract_profile_metrics(
            "full",
            measurement_rows,
            footprints,
            request_rows,
        ),
    )
    full_measured = {
        str(gate.metric): str(gate.status.value)
        for gate in full_gates
        if gate.metric not in {"package_bytes", "arm_ready_seconds"}
        and gate.status.value != "unknown"
    }
    full_fails = any(value == "fail" for value in full_measured.values())
    quality_status, quality_note = _quality_status(measurement_rows)
    required_dynamic = {
        "framework_idle_pss_bytes",
        "idle_cpu_avg_core_pct",
        "idle_cpu_p95_core_pct",
        "wsl_ready_seconds",
        "mock_tax_ms",
        "framework_context_tokens",
        "ten_turn_pss_growth_bytes",
        "fd_growth",
        "thread_growth",
    }
    protocol_data_complete = bool(
        required_dynamic.issubset(measured)
        and all(key in full_measured for key in required_dynamic)
    )

    if getattr(arm_result, "executed", False) and not getattr(
        arm_result,
        "compatible",
        False,
    ):
        classification = (
            "4 — ARM dependency blocking: resolve packaging/native wheels "
            "before board performance testing."
        )
    elif not protocol_data_complete or quality_status == "unknown":
        classification = (
            "Inconclusive — the full n=3 startup/idle/S1/S2/S3 protocol is "
            "incomplete (including smoke runs); gate rows are provisional only."
        )
    elif quality_status == "fail":
        classification = (
            "2 — Core task correctness or process-exit acceptance failed: "
            "the measured runtime is not yet suitable for the target."
        )
    elif gate_map.get("framework_context_tokens") == "fail" and all(
        gate_map.get(name) == "pass"
        for name in (
            "framework_idle_pss_bytes",
            "idle_cpu_avg_core_pct",
            "idle_cpu_p95_core_pct",
        )
    ):
        classification = (
            "3 — Host memory/CPU pass while prompt cost fails: trim prompt, "
            "tool schemas, and skills first."
        )
    elif any(value == "fail" for value in measured.values()):
        classification = (
            "2 — Core still fails at least one measured overhead gate: split "
            "dependencies or refactor the runtime."
        )
    elif (
        measured
        and all(value == "pass" for value in measured.values())
        and full_fails
        and quality_status == "pass"
    ):
        classification = (
            "1 — Core passes while full fails at least one measured gate: the "
            "core is usable and the default capability set is overweight."
        )
    else:
        classification = (
            "Inconclusive — required gates are unknown or warning; do not force "
            "the run into classes 1–4."
        )

    sample_rows = [
        row
        for row in measurement_rows
        if row.get("record_type") == "measurement"
    ]
    budget_skips = sum(
        row.get("error_type") == "ApiBudgetExhausted" for row in sample_rows
    )
    failures = sum(
        row.get("success") is not True
        and row.get("error_type") != "ApiBudgetExhausted"
        for row in sample_rows
    )
    orphan_failures = sum(
        row.get("metrics", {}).get("orphan_free") is False
        for row in sample_rows
        if isinstance(row.get("metrics"), Mapping)
    )
    return (
        "\n## Protocol decision\n\n"
        f"{classification}\n\n"
        f"- Independent measurement rows: {len(sample_rows)}\n"
        f"- Failed/incorrect rows: {failures}\n"
        f"- Budget-skipped rows: {budget_skips}\n"
        f"- Rows with surviving descendants: {orphan_failures}\n"
        f"- Correctness/exit acceptance: {quality_status} — {quality_note}\n"
        "- The current full venv is an installation upper bound; the future "
        "minimal-package gate stays unknown until a pruned artifact is built.\n"
    )


def _quality_status(
    measurements: Iterable[Mapping[str, Any]],
) -> tuple[str, str]:
    rows = [
        row
        for row in measurements
        if row.get("record_type") == "measurement"
    ]
    if any(
        isinstance(row.get("metrics"), Mapping)
        and row["metrics"].get("orphan_free") is False
        for row in rows
    ):
        return "fail", "one or more descendants survived shutdown"

    mock = [
        row
        for row in rows
        if row.get("profile") in {"full", "core"}
        and row.get("scenario") in {"S1", "S2"}
        and row.get("backend") == "mock"
    ]
    if mock:
        groups = {
            (profile, scenario): [
                row
                for row in mock
                if row.get("profile") == profile and row.get("scenario") == scenario
            ]
            for profile in ("full", "core")
            for scenario in ("S1", "S2")
        }
        if any(len(group) < 3 for group in groups.values()):
            return "unknown", "fewer than 3 mock samples (for example, smoke mode)"
        if any(
            sum(row.get("success") is True for row in group) < 3
            for group in groups.values()
        ):
            return "fail", "mock correctness is below 3/3"

    remote = [
        row
        for row in rows
        if row.get("profile") in {"full", "core"}
        and row.get("scenario") in {"S1", "S2"}
        and row.get("backend") == "dashscope"
        and row.get("error_type") != "ApiBudgetExhausted"
    ]
    if not remote:
        return (
            "unknown",
            "offline-only: DashScope correctness was not run; conclusions are provisional",
        )
    groups = {
        (profile, scenario): [
            row
            for row in remote
            if row.get("profile") == profile and row.get("scenario") == scenario
        ]
        for profile in ("full", "core")
        for scenario in ("S1", "S2")
    }
    if any(len(group) < 3 for group in groups.values()):
        return (
            "unknown",
            "fewer than 3 executed remote samples; budget skips do not count",
        )
    if any(
        sum(row.get("success") is True for row in group) < 2
        for group in groups.values()
    ):
        return "fail", "real-model correctness is below 2/3"
    if mock:
        return "pass", "mock 3/3 and real-model groups at least 2/3"
    return "unknown", "no complete S1/S2 correctness matrix is present"
