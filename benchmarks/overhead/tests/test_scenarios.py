from __future__ import annotations

from benchmarks.overhead.qwenpaw_overhead.scenarios import (
    S1,
    S2,
    S3,
    SHELL_COMMAND,
    get_scenario,
)


def test_synthetic_scenarios_are_stable_and_safe() -> None:
    assert len(S1.prompts) == 1
    assert "QP7F3A" in S1.prompts[0]
    assert S2.expected_command == SHELL_COMMAND
    assert len(S3.prompts) == 10
    assert get_scenario("s2") is S2
    assert len(S1.prompt_sha256) == 64

