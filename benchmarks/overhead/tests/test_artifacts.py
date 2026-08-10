from __future__ import annotations

import csv
import json

import pytest

from benchmarks.overhead.qwenpaw_overhead.artifacts import (
    append_jsonl,
    atomic_write_json,
    result_directory,
    write_csv,
)


def test_result_directory_rejects_traversal_and_existing_run(tmp_path) -> None:
    with pytest.raises(ValueError):
        result_directory(tmp_path, "../escape")
    created = result_directory(tmp_path, "run-1")
    assert created.parent == tmp_path.resolve()
    with pytest.raises(FileExistsError):
        result_directory(tmp_path, "run-1")


def test_json_and_csv_writers(tmp_path) -> None:
    atomic_write_json(tmp_path / "x.json", {"b": 2, "a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1, "b": 2}
    append_jsonl(tmp_path / "events.jsonl", {"event": "ready"})
    assert json.loads((tmp_path / "events.jsonl").read_text()) == {
        "event": "ready",
    }
    write_csv(tmp_path / "rows.csv", [{"a": 1}, {"b": [2]}])
    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2

