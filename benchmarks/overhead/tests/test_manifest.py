from __future__ import annotations

import json

from benchmarks.overhead.qwenpaw_overhead.manifest import sanitized_config_hash


def test_sanitized_hash_ignores_secret_values(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"api_key": "first", "nested": {"value": 3}}),
        encoding="utf-8",
    )
    first = sanitized_config_hash(config)
    config.write_text(
        json.dumps({"api_key": "second", "nested": {"value": 3}}),
        encoding="utf-8",
    )
    assert sanitized_config_hash(config) == first


def test_sanitized_hash_detects_non_secret_drift(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "a"}), encoding="utf-8")
    first = sanitized_config_hash(config)
    config.write_text(json.dumps({"model": "b"}), encoding="utf-8")
    assert sanitized_config_hash(config) != first

