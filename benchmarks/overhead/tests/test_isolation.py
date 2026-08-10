from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.overhead.qwenpaw_overhead.isolation import (
    IsolationError,
    create_isolated_run,
)


def _seed(tmp_path: Path) -> tuple[Path, Path, Path]:
    working = tmp_path / "seed-working"
    secret = tmp_path / "seed-secret"
    base = tmp_path / "runs"
    (working / "workspaces" / "default").mkdir(parents=True)
    (working / "workspaces" / "default" / "agent.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (working / "venv" / "bin").mkdir(parents=True)
    (working / "venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    (working / "bin").mkdir()
    (working / "bin" / "qwenpaw").write_text("wrapper\n", encoding="utf-8")
    secret.mkdir()
    (secret / ".master_key").write_text("not-a-real-key\n", encoding="utf-8")
    base.mkdir()
    return working, secret, base


def test_isolated_run_redirects_every_mutable_root_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working, secret, base = _seed(tmp_path)
    seed_agent = working / "workspaces" / "default" / "agent.json"
    original = seed_agent.read_bytes()
    monkeypatch.setenv("HTTPS_PROXY", "http://should-not-leak.invalid")
    monkeypatch.setenv("no_proxy", "localhost")

    with create_isolated_run(working, secret, "run / 01", base) as paths:
        root = paths.root
        assert root.parent == base.resolve()
        assert paths.working_dir != working.resolve()
        assert (paths.working_dir / "workspaces/default/agent.json").is_file()
        assert not (paths.working_dir / "venv").exists()
        assert not (paths.working_dir / "bin").exists()
        assert (paths.secret_dir / ".master_key").is_file()
        assert paths.env["QWENPAW_WORKING_DIR"] == str(paths.working_dir)
        assert paths.env["QWENPAW_SECRET_DIR"] == str(paths.secret_dir)
        assert paths.env["PAW_STATE_DIR"] == str(paths.state_dir)
        assert paths.env["QWENPAW_DISABLE_KEYRING"] == "1"
        assert "HTTPS_PROXY" not in paths.env
        assert "no_proxy" not in paths.env
        for directory in (
            paths.root,
            paths.working_dir,
            paths.secret_dir,
            paths.state_dir,
            paths.project_dir,
            paths.cache_dir,
            paths.home_dir,
            paths.temp_dir,
            paths.backup_dir,
        ):
            assert os.stat(directory).st_mode & 0o777 == 0o700
        assert os.stat(paths.secret_dir / ".master_key").st_mode & 0o777 == 0o600
        (paths.working_dir / "runtime-write.txt").write_text(
            "isolated\n",
            encoding="utf-8",
        )

    assert not root.exists()
    assert seed_agent.read_bytes() == original
    assert not (working / "runtime-write.txt").exists()


def test_isolated_run_rejects_non_ignored_symlink(tmp_path: Path) -> None:
    working, secret, base = _seed(tmp_path)
    (working / "escape").symlink_to(tmp_path / "outside")

    with pytest.raises(IsolationError, match="symlink"):
        create_isolated_run(working, secret, "bad-link", base)
    assert not list(base.iterdir())


def test_isolated_run_rejects_temporary_base_inside_seed(tmp_path: Path) -> None:
    working, secret, _ = _seed(tmp_path)
    nested_base = working / "runs"
    nested_base.mkdir()

    with pytest.raises(IsolationError, match="must not be inside"):
        create_isolated_run(working, secret, "recursive", nested_base)

