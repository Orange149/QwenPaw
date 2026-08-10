from __future__ import annotations

import csv
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

from benchmarks.overhead.qwenpaw_overhead.sampler import (
    ProcessIdentity,
    ResourceSampler,
    classify_process_role,
    diff_state,
    snapshot_state,
    wait_for_descendants_exit,
)


def test_process_role_classification() -> None:
    assert classify_process_role(
        [sys.executable, "-m", "qwenpaw", "acp", "--local-diagnostics"],
    ) == "acp"
    assert classify_process_role(["npx", "-y", "tavily-mcp@latest"]) == "mcp"
    assert classify_process_role(["chromium", "--headless"]) == "browser"
    assert classify_process_role(["bash", "-c", "true"]) == "tool"
    assert classify_process_role(["qwenpaw"], is_root=True) == "tui"


def test_resource_sampler_streams_csv_and_aggregates_descendants(
    tmp_path: Path,
) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    code = (
        "import socket,subprocess,time; "
        f"s=socket.create_connection(('127.0.0.1',{port})); "
        "p=subprocess.Popen(['bash','-c','sleep 0.35']); "
        "time.sleep(0.35); p.wait(); s.close()"
    )
    process = subprocess.Popen([sys.executable, "-c", code])
    listener.settimeout(2)
    connection, _ = listener.accept()
    output = tmp_path / "samples.csv"
    sampler = ResourceSampler(process.pid, 0.02, output).start()
    try:
        time.sleep(0.16)
        immediate = sampler.sample_now()
        aggregate = sampler.stop()
        process.wait(timeout=2)
    finally:
        connection.close()
        listener.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)

    assert aggregate["root_pid"] == process.pid
    assert aggregate["sample_count"] >= 2
    assert aggregate["tree_peak"]["processes"] >= 2
    assert aggregate["tree_peak"]["rss_bytes"] > 0
    assert aggregate["tree_peak"]["pss_bytes"] is not None
    assert aggregate["smaps_coverage_min"] == 1.0
    assert aggregate["nonempty_sample_count"] >= 1
    assert immediate["processes"] >= 2
    assert immediate["socket_scan"] is True
    assert aggregate["observed_processes"]
    assert process.pid in aggregate["observed_pids"]
    assert Path(aggregate["output_path"]) == output
    assert sampler.stop() is aggregate
    assert aggregate["observed_loopback_only"] is True
    assert any(
        endpoint["port"] == port
        for endpoint in aggregate["observed_remote_endpoints"]
    )
    assert "cgroup_start" in aggregate
    assert "cgroup_delta" in aggregate
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"monotonic_ns", "pss_bytes", "role", "cmdline_sha256"} <= set(
        rows[0],
    )
    assert rows[0]["start_ticks"]


def test_snapshot_and_diff_state_report_signed_growth(tmp_path: Path) -> None:
    root = tmp_path / "state"
    root.mkdir()
    old = root / "old.txt"
    changed = root / "changed.txt"
    old.write_text("old", encoding="utf-8")
    changed.write_text("a", encoding="utf-8")
    before = snapshot_state(root)

    old.unlink()
    changed.write_text("a much longer value", encoding="utf-8")
    (root / "new.txt").write_text("new", encoding="utf-8")
    after = snapshot_state(root)
    delta = diff_state(before, after)

    assert delta["added"] == ["new.txt"]
    assert delta["removed"] == ["old.txt"]
    assert "changed.txt" in delta["modified"]
    assert delta["delta"]["apparent_bytes"] > 0
    assert delta["changed_path_count"] >= 3


def test_wait_for_descendants_exit_uses_creation_identity() -> None:
    short = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.1)"])
    short_identity = ProcessIdentity(short.pid, psutil.Process(short.pid).create_time())
    assert wait_for_descendants_exit([short_identity], timeout=1.0)
    short.wait(timeout=1)

    long = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        identity = {
            "pid": long.pid,
            "create_time": psutil.Process(long.pid).create_time(),
        }
        assert wait_for_descendants_exit([identity], timeout=0.02) is False
    finally:
        long.terminate()
        long.wait(timeout=1)
    assert wait_for_descendants_exit([identity], timeout=0.5)
