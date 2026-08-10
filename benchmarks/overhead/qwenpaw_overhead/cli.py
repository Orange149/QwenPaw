"""Command-line entry point for the external QwenPaw overhead harness."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifacts import result_directory
from .footprint import scan_for_secrets
from .host_metrics import sample_vmmemwsl
from .matrix import REQUEST_SCENARIOS
from .preflight import (
    DEFAULT_PYTHON,
    DEFAULT_SECRET_DIR,
    DEFAULT_WORKING_DIR,
    preflight_ok,
    run_preflight,
)
from .report import (
    evaluate_core_gates,
    render_markdown_report,
    summarize_records,
    write_summary_csv,
)
from .runner import (
    HarnessRunner,
    HarnessSettings,
    _build_trim_decisions,
    _decision_section,
    _extract_core_metrics,
    _growth_flags,
)
from .schemas import (
    ArmDryRunResult,
    DistributionRecord,
    FootprintRecord,
    SharedObjectRecord,
    StateLayer,
)


DEFAULT_RESULT_ROOT = Path("benchmark_results")
PHASES = ("all", "static", "startup", "mock", "dashscope", "tui")
PROFILES = ("full", "no_skills", "local_tools", "core")


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--secret-dir", type=Path, default=DEFAULT_SECRET_DIR)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="read-only checks; does not import or start QwenPaw",
    )
    _add_paths(preflight)
    preflight.add_argument("--json", action="store_true")

    footprint = subparsers.add_parser(
        "footprint",
        help="collect static install/state/dependency footprint",
    )
    _add_paths(footprint)
    footprint.add_argument("--run-id")
    footprint.add_argument(
        "--arm-dry-run",
        action="store_true",
        help="execute the network-capable aarch64/cp312 pip resolver dry-run",
    )

    run = subparsers.add_parser("run", help="execute a benchmark phase")
    _add_paths(run)
    run.add_argument("--run-id")
    run.add_argument("--phase", choices=PHASES, default="all")
    run.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
    )
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--idle-seconds", type=float, default=90.0)
    run.add_argument("--baseline-seconds", type=float, default=60.0)
    run.add_argument("--sample-interval", type=float, default=0.25)
    run.add_argument("--startup-timeout", type=float, default=180.0)
    run.add_argument("--cpu-affinity", default="0-3")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--arm-dry-run", action="store_true")
    run.add_argument("--allow-remote-api", action="store_true")
    run.add_argument("--confirm-free-quota-stop", action="store_true")
    run.add_argument(
        "--remote-scenarios",
        nargs="+",
        choices=REQUEST_SCENARIOS,
        default=list(REQUEST_SCENARIOS),
        help="limit the paid DashScope phase to S1 and/or S2",
    )

    report = subparsers.add_parser(
        "report",
        help="rebuild summary.csv and report.md without starting QwenPaw",
    )
    report.add_argument("result_dir", type=Path)
    return parser


def _print_checks(checks: Iterable[Any], *, as_json: bool) -> None:
    materialized = [check.as_dict() for check in checks]
    if as_json:
        print(json.dumps(materialized, ensure_ascii=False, indent=2))
        return
    width = max((len(row["name"]) for row in materialized), default=0)
    for row in materialized:
        print(f"{row['status'].upper():4} {row['name']:<{width}}  {row['detail']}")


def _preflight(args: argparse.Namespace) -> int:
    checks = run_preflight(
        python_executable=args.python,
        working_dir=args.working_dir,
        secret_dir=args.secret_dir,
    )
    _print_checks(checks, as_json=args.json)
    return 0 if preflight_ok(checks) else 2


def _settings(
    args: argparse.Namespace,
    output: Path,
) -> HarnessSettings:
    repetitions = args.repetitions
    idle = args.idle_seconds
    baseline = args.baseline_seconds
    interval = args.sample_interval
    timeout = args.startup_timeout
    if args.smoke:
        repetitions = 1
        idle = min(idle, 2.0)
        baseline = min(baseline, 2.0)
        interval = min(interval, 0.2)
        timeout = min(timeout, 60.0)
    if repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if min(idle, baseline, interval, timeout) <= 0:
        raise ValueError("durations and sample interval must be positive")
    return HarnessSettings(
        run_id=output.name,
        result_dir=output,
        python_executable=args.python,
        seed_working_dir=args.working_dir,
        seed_secret_dir=args.secret_dir,
        profiles=tuple(dict.fromkeys(args.profiles)),
        repetitions=repetitions,
        idle_seconds=idle,
        baseline_seconds=baseline,
        sample_interval_s=interval,
        cpu_affinity=args.cpu_affinity,
        startup_timeout_s=timeout,
        execute_arm_resolver=args.arm_dry_run,
        allow_remote_api=args.allow_remote_api,
        confirm_free_quota_stop=args.confirm_free_quota_stop,
        remote_scenarios=tuple(dict.fromkeys(args.remote_scenarios)),
        smoke=args.smoke,
    )


def _new_result(root: Path, requested_run_id: str | None) -> Path:
    return result_directory(root, requested_run_id or _run_id())


async def _execute_phases(
    runner: HarnessRunner,
    phase: str,
    *,
    include_remote: bool = False,
) -> None:
    if phase in {"all", "startup"}:
        await runner.run_startup_matrix()
    if phase in {"all", "mock"}:
        await runner.run_mock_matrix(include_s3=True)
    if phase in {"all", "tui"}:
        await runner.run_tui_reference()
    if phase == "dashscope" or (phase == "all" and include_remote):
        await runner.run_dashscope_matrix()


def _run(args: argparse.Namespace) -> int:
    checks = run_preflight(
        python_executable=args.python,
        working_dir=args.working_dir,
        secret_dir=args.secret_dir,
    )
    _print_checks(checks, as_json=False)
    if not preflight_ok(checks):
        print("Preflight failed; no result directory was created.", file=sys.stderr)
        return 2

    remote_flags_present = bool(
        args.allow_remote_api or args.confirm_free_quota_stop
    )
    if args.phase not in {"all", "dashscope"} and remote_flags_present:
        raise ValueError(
            "remote confirmations are valid only with --phase all or dashscope",
        )
    if remote_flags_present and not (
        args.allow_remote_api and args.confirm_free_quota_stop
    ):
        raise ValueError("remote execution requires both confirmation flags")
    include_remote = args.phase == "dashscope" or (
        args.phase == "all" and remote_flags_present
    )
    if include_remote and args.profiles != list(PROFILES):
        raise ValueError("the DashScope protocol fixes profiles to direct/full/core/stock")

    output = _new_result(args.result_root, args.run_id)
    runner = HarnessRunner(_settings(args, output))
    backend = (
        "offline+dashscope"
        if args.phase == "all" and include_remote
        else "dashscope"
        if args.phase == "dashscope"
        else "offline"
    )
    runner.initialize_manifest(backend=backend, scenarios=["idle", "S1", "S2", "S3"])
    vmmem_before = sample_vmmemwsl()
    completed = False
    caught: BaseException | None = None
    try:
        if args.phase in {"all", "static"}:
            runner.collect_footprint()
        if args.phase == "all":
            runner.collect_baseline()
        asyncio.run(
            _execute_phases(
                runner,
                args.phase,
                include_remote=include_remote,
            ),
        )
        completed = True
    except BaseException as exc:
        caught = exc
    finally:
        try:
            runner.record_vmmemwsl_boundaries(
                vmmem_before,
                sample_vmmemwsl(),
            )
            gates = runner.finalize(completed=completed)
        except BaseException as exc:
            caught = caught or exc
            gates = []

    print(f"Results: {output}")
    if gates:
        verdicts = ", ".join(f"{gate.metric}={gate.status.value}" for gate in gates)
        print(f"Core gates: {verdicts}")
    if caught is not None:
        if isinstance(caught, KeyboardInterrupt):
            print("Interrupted; partial artifacts were retained.", file=sys.stderr)
            return 130
        print(f"Benchmark failed: {type(caught).__name__}: {caught}", file=sys.stderr)
        return 1
    failed_samples = sum(
        row.get("record_type") == "measurement" and row.get("success") is not True
        for row in runner.measurements
    )
    if failed_samples:
        print(f"Completed with {failed_samples} failed sample(s).", file=sys.stderr)
        return 1
    return 0


def _footprint(args: argparse.Namespace) -> int:
    checks = run_preflight(
        python_executable=args.python,
        working_dir=args.working_dir,
        secret_dir=args.secret_dir,
    )
    _print_checks(checks, as_json=False)
    if not preflight_ok(checks):
        return 2
    output = _new_result(args.result_root, args.run_id)
    settings = HarnessSettings(
        run_id=output.name,
        result_dir=output,
        python_executable=args.python,
        seed_working_dir=args.working_dir,
        seed_secret_dir=args.secret_dir,
        execute_arm_resolver=args.arm_dry_run,
    )
    runner = HarnessRunner(settings)
    runner.initialize_manifest(backend="static", scenarios=[])
    try:
        runner.collect_footprint()
        runner.finalize(completed=True)
    except Exception as exc:
        print(f"Footprint failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Partial results: {output}", file=sys.stderr)
        return 1
    print(f"Results: {output}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _optional_int(value: str | None) -> int | None:
    return int(value) if value not in {None, "", "None"} else None


def _load_footprint(
    path: Path,
) -> tuple[list[FootprintRecord], list[DistributionRecord], list[SharedObjectRecord]]:
    footprints: list[FootprintRecord] = []
    distributions: list[DistributionRecord] = []
    objects: list[SharedObjectRecord] = []
    if not path.is_file() or path.stat().st_size == 0:
        return footprints, distributions, objects
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            kind = row.get("record_type")
            if kind == "target":
                footprints.append(
                    FootprintRecord(
                        category=row["category"],
                        path=row["path"],
                        layer=StateLayer(row["layer"]),
                        present=_bool(row["present"]),
                        rollup=_bool(row.get("rollup", "false")),
                        excluded_root_names=tuple(
                            json.loads(row.get("excluded_root_names") or "[]"),
                        ),
                        allocated_bytes=int(row.get("allocated_bytes") or 0),
                        apparent_bytes=int(row.get("apparent_bytes") or 0),
                        file_count=int(row.get("file_count") or 0),
                        directory_count=int(row.get("directory_count") or 0),
                        symlink_count=int(row.get("symlink_count") or 0),
                        other_entry_count=int(row.get("other_entry_count") or 0),
                        unique_inode_count=int(row.get("unique_inode_count") or 0),
                        duplicate_inode_count=int(row.get("duplicate_inode_count") or 0),
                        inaccessible_entry_count=int(
                            row.get("inaccessible_entry_count") or 0,
                        ),
                        error=row.get("error") or None,
                    ),
                )
            elif kind == "distribution":
                distributions.append(
                    DistributionRecord(
                        name=row["name"],
                        version=row["version"],
                        location=row["location"],
                        allocated_bytes=int(row.get("allocated_bytes") or 0),
                        apparent_bytes=int(row.get("apparent_bytes") or 0),
                        file_count=int(row.get("file_count") or 0),
                        native_file_count=int(row.get("native_file_count") or 0),
                        duplicate_inode_count=int(row.get("duplicate_inode_count") or 0),
                        missing_file_count=int(row.get("missing_file_count") or 0),
                        outside_root_file_count=int(
                            row.get("outside_root_file_count") or 0,
                        ),
                    ),
                )
            elif kind == "shared_object":
                objects.append(
                    SharedObjectRecord(
                        path=row["path"],
                        distribution=row.get("distribution") or None,
                        allocated_bytes=int(row.get("allocated_bytes") or 0),
                        apparent_bytes=int(row.get("apparent_bytes") or 0),
                        elf_class=_optional_int(row.get("elf_class")),
                        elf_machine=row.get("elf_machine") or None,
                        elf_machine_id=_optional_int(row.get("elf_machine_id")),
                        is_symlink=_bool(row.get("is_symlink", "false")),
                    ),
                )
    return footprints, distributions, objects


def _arm_from_manifest(value: Any) -> ArmDryRunResult | None:
    if not isinstance(value, dict) or not isinstance(value.get("command"), list):
        return None
    return ArmDryRunResult(
        command=tuple(str(item) for item in value["command"]),
        executed=bool(value.get("executed")),
        compatible=value.get("compatible"),
        returncode=value.get("returncode"),
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
        timed_out=bool(value.get("timed_out")),
        blockers=tuple(
            dict(item)
            for item in value.get("blockers", [])
            if isinstance(item, dict)
        ),
        resolution_limited_by_top_level=bool(
            value.get("resolution_limited_by_top_level"),
        ),
    )


def _report(args: argparse.Namespace) -> int:
    directory = args.result_dir.expanduser().resolve()
    measurements = _read_jsonl(directory / "measurements.jsonl")
    requests = _read_jsonl(directory / "requests.jsonl")
    footprints, distributions, objects = _load_footprint(
        directory / "footprint.csv",
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arm = _arm_from_manifest(manifest.get("arm_resolver"))
    summaries = summarize_records(measurements)
    write_summary_csv(directory / "summary.csv", summaries)
    gates = evaluate_core_gates(
        _extract_core_metrics(measurements, footprints, requests),
        **_growth_flags(measurements),
    )
    decisions = _build_trim_decisions(measurements, requests, distributions)
    findings = scan_for_secrets([directory], environment=os.environ)
    report = render_markdown_report(
        summaries,
        core_gates=gates,
        footprint=footprints,
        distributions=distributions,
        shared_objects=objects,
        trim_decisions=decisions,
        arm_result=arm,
        secret_findings=findings,
    )
    report += _decision_section(
        gates,
        measurements,
        arm,
        request_records=requests,
        footprints=footprints,
    )
    (directory / "report.md").write_text(report, encoding="utf-8")
    final_findings = scan_for_secrets([directory], environment=os.environ)
    if final_findings:
        print(
            f"Secret scan failed with {len(final_findings)} finding(s).",
            file=sys.stderr,
        )
        return 1
    print(f"Rebuilt: {directory / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "footprint":
            return _footprint(args)
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            return _report(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
