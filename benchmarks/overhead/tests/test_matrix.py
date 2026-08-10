from __future__ import annotations

import pytest

from benchmarks.overhead.qwenpaw_overhead.matrix import (
    ApiBudget,
    dashscope_jobs,
    mock_jobs,
    startup_jobs,
)


def test_default_matrix_sizes_and_s3_scope() -> None:
    assert len(startup_jobs()) == 12
    jobs = mock_jobs()
    assert len(jobs) == 26
    s3 = [job for job in jobs if job.scenario == "S3"]
    assert [(job.profile, job.turns) for job in s3] == [
        ("full", 10),
        ("core", 10),
    ]


def test_dashscope_matrix_is_rotated_and_bounded() -> None:
    jobs = dashscope_jobs()
    assert len(jobs) == 19
    first_targets = [
        (job.profile, job.direct)
        for job in jobs
        if job.scenario == "S1" and job.repetition == 1
    ]
    second_targets = [
        (job.profile, job.direct)
        for job in jobs
        if job.scenario == "S1" and job.repetition == 2
    ]
    assert first_targets != second_targets

    s1_only = dashscope_jobs(scenarios=("S1",))
    assert len(s1_only) == 10
    assert {job.scenario for job in s1_only} == {"S1"}

    with pytest.raises(ValueError):
        dashscope_jobs(scenarios=())


def test_budget_stops_before_either_limit() -> None:
    budget = ApiBudget(max_attempts=2, max_tokens=100)
    budget.account(attempts=1, tokens=90)
    assert budget.can_start()
    budget.account(attempts=0, tokens=10)
    assert budget.exhausted
    with pytest.raises(ValueError):
        budget.account(attempts=-1, tokens=0)
