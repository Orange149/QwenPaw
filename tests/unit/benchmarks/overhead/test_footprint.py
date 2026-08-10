"""Tests for static footprint and portability discovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.overhead.qwenpaw_overhead.footprint import (
    SecretLeakError,
    arm_cp312_pip_dry_run,
    assert_no_secrets,
    build_arm_cp312_pip_command,
    classify_state_path,
    default_footprint_targets,
    distribution_file_index,
    measure_targets,
    measure_tree,
    parse_elf_identity,
    scan_distributions,
    scan_for_secrets,
    scan_shared_objects,
)
from benchmarks.overhead.qwenpaw_overhead.schemas import (
    FootprintTarget,
    StateLayer,
)


def test_measure_tree_deduplicates_hardlinks_and_does_not_follow_symlinks(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"0123456789")
    os.link(payload, tmp_path / "hardlink.bin")
    (tmp_path / "symlink.bin").symlink_to(payload)

    record = measure_tree(
        FootprintTarget("fixture", str(tmp_path), StateLayer.CACHE),
    )

    assert record.present is True
    assert record.file_count == 1
    assert record.directory_count == 1
    assert record.symlink_count == 1
    assert record.duplicate_inode_count == 1
    assert record.apparent_bytes >= 10
    assert record.allocated_bytes >= payload.stat().st_blocks * 512


def test_measure_targets_deduplicates_across_categories_unless_rollup(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload"
    payload.write_text("data", encoding="utf-8")
    targets = [
        FootprintTarget("owner", str(payload), StateLayer.CACHE),
        FootprintTarget("duplicate", str(payload), StateLayer.CACHE),
        FootprintTarget(
            "rollup",
            str(payload),
            StateLayer.CACHE,
            rollup=True,
        ),
    ]

    owner, duplicate, rollup = measure_targets(targets)

    assert owner.file_count == 1
    assert duplicate.file_count == 0
    assert duplicate.duplicate_inode_count == 1
    assert rollup.file_count == 1
    assert rollup.rollup is True


def test_classify_state_path_uses_specific_layers_before_working_dir(
    tmp_path: Path,
) -> None:
    working = tmp_path / ".qwenpaw"
    secrets = tmp_path / ".qwenpaw.secret"
    install = tmp_path / "venv"

    assert (
        classify_state_path(
            working / "models" / "model.gguf",
            working_dir=working,
            install_root=install,
            secret_dir=secrets,
        )
        is StateLayer.MODELS
    )
    assert (
        classify_state_path(
            secrets / "envs.json",
            working_dir=working,
            install_root=install,
            secret_dir=secrets,
        )
        is StateLayer.SECRETS
    )
    assert (
        classify_state_path(
            install / "lib" / "package.py",
            working_dir=working,
            install_root=install,
            secret_dir=secrets,
        )
        is StateLayer.INSTALL
    )


def test_default_targets_discover_workspace_state_and_python_runtime(
    tmp_path: Path,
) -> None:
    working = tmp_path / ".qwenpaw"
    workspace = working / "workspaces" / "default"
    (workspace / "sessions").mkdir(parents=True)
    (workspace / "history.db").write_bytes(b"db")
    (workspace / "mem_agent").mkdir()
    (workspace / "token_usage.json").write_text("{}", encoding="utf-8")
    python = tmp_path / "python3.12"
    python.write_bytes(b"python")
    stdlib = tmp_path / "python3.12-stdlib"
    stdlib.mkdir()

    targets = default_footprint_targets(
        working,
        install_root=tmp_path / "venv",
        package_dir=tmp_path / "qwenpaw",
        python_executable=python,
        python_stdlib=stdlib,
        home=tmp_path,
    )
    by_category = {target.category: target for target in targets}

    assert by_category["runtime.python_entrypoint"].path == str(python)
    assert by_category["runtime.python_stdlib"].path == str(stdlib)
    assert "workspace.default.sessions" in by_category
    assert "workspace.default.history_db" in by_category
    assert "workspace.default.mem_agent" in by_category
    assert "workspace.default.token_usage.json" in by_category
    assert all(
        by_category[name].rollup
        for name in by_category
        if name.startswith("workspace.default")
    )


def _make_fake_distribution(site_packages: Path) -> tuple[Path, Path]:
    package = site_packages / "demo"
    dist_info = site_packages / "demo-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata = dist_info / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
        encoding="utf-8",
    )
    record = dist_info / "RECORD"
    record.write_text(
        "demo/__init__.py,,\n"
        "demo-1.0.dist-info/METADATA,,\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    return module, dist_info


def test_distribution_record_scan_and_file_index(tmp_path: Path) -> None:
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    module, _ = _make_fake_distribution(site_packages)

    records = scan_distributions(site_packages)
    index = distribution_file_index(site_packages)

    assert len(records) == 1
    assert records[0].name == "demo"
    assert records[0].version == "1.0"
    assert records[0].file_count == 3
    assert records[0].apparent_bytes > module.stat().st_size
    assert index[str(module.resolve())] == "demo"


def _elf_header(machine_id: int, *, elf_class: int = 64) -> bytes:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4] = 2 if elf_class == 64 else 1
    header[5] = 1  # little endian
    header[18:20] = machine_id.to_bytes(2, "little")
    return bytes(header)


def test_shared_object_inventory_reports_elf_architecture(tmp_path: Path) -> None:
    library = tmp_path / "package" / "native.abi3.so"
    library.parent.mkdir()
    library.write_bytes(_elf_header(183))

    assert parse_elf_identity(library) == (64, "aarch64", 183)
    records = scan_shared_objects(
        [tmp_path],
        distribution_index={str(library.resolve()): "native-demo"},
    )

    assert len(records) == 1
    assert records[0].distribution == "native-demo"
    assert records[0].elf_machine == "aarch64"


def test_arm_pip_probe_is_plan_only_by_default(monkeypatch) -> None:
    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("subprocess must not run without execute=True")

    monkeypatch.setattr(subprocess, "run", forbidden_run)

    result = arm_cp312_pip_dry_run("qwenpaw==2.0.1")
    command = build_arm_cp312_pip_command("qwenpaw==2.0.1")

    assert result.executed is False
    assert result.compatible is None
    assert tuple(result.command) == command
    assert "--dry-run" in command
    assert "manylinux_2_17_aarch64" in command
    assert "cp312" in command


def test_arm_pip_probe_executes_only_with_explicit_switch(
    monkeypatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="resolved https://user:password@example.invalid/simple",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = arm_cp312_pip_dry_run("demo==1", execute=True)

    assert result.executed is True
    assert result.compatible is True
    assert result.returncode == 0
    assert "password" not in result.stdout
    assert captured["kwargs"]["check"] is False


def test_arm_pip_probe_reports_timeout_as_a_blocker(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = arm_cp312_pip_dry_run(
        "qwenpaw==2.0.1",
        execute=True,
        timeout_seconds=0.01,
    )

    assert result.executed is True
    assert result.compatible is False
    assert result.timed_out is True
    assert result.blockers == (
        {
            "package": "qwenpaw",
            "kind": "resolver_timeout",
            "reason": "pip did not finish before the compatibility probe timeout",
        },
    )


def test_secret_scan_never_returns_or_raises_with_plaintext(tmp_path: Path) -> None:
    secret = "dashscope-secret-value-12345"
    artifact = tmp_path / "result.json"
    artifact.write_text(
        f'{{"authorization":"Bearer live-token-value-12345","copied":"{secret}"}}',
        encoding="utf-8",
    )

    findings = scan_for_secrets(
        [tmp_path],
        known_secret_values=[secret],
    )

    assert findings
    assert secret not in repr(findings)
    with pytest.raises(SecretLeakError) as exc_info:
        assert_no_secrets(findings)
    assert secret not in str(exc_info.value)


def test_secret_scan_accepts_redacted_placeholders(tmp_path: Path) -> None:
    artifact = tmp_path / "manifest.json"
    artifact.write_text(
        '{"api_key":"<redacted>","token":"${TOKEN}"}',
        encoding="utf-8",
    )

    assert scan_for_secrets([artifact]) == []


def test_secret_scan_rejects_secret_storage_files(tmp_path: Path) -> None:
    secret_file = tmp_path / ".master_key"
    secret_file.write_text("not-printed", encoding="utf-8")

    findings = scan_for_secrets([tmp_path])

    assert any(item.rule == "forbidden_secret_file" for item in findings)
