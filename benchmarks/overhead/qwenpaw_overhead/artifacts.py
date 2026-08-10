"""Safe writes for public benchmark artifacts."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def result_directory(output_root: Path, run_id: str) -> Path:
    """Create a non-ambiguous result directory below *output_root*."""

    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or dash",
        )
    root = output_root.expanduser().resolve()
    result = root / run_id
    root.mkdir(parents=True, exist_ok=True)
    result.mkdir(mode=0o750, exist_ok=False)
    return result


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON via an adjacent temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temp_path, 0o640)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a union-of-fields CSV; nested values become compact JSON."""

    materialized = [dict(row) for row in rows]
    keys: list[str] = []
    for row in materialized:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not keys:
            return
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            encoded = {
                key: (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(encoded)

