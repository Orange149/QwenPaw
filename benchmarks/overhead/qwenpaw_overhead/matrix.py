"""Declarative execution matrix and safety budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CONTROLLED_PROFILES = ("full", "no_skills", "local_tools", "core")
ALL_PROFILES = (*CONTROLLED_PROFILES, "stock_reference")
REQUEST_SCENARIOS = ("S1", "S2")


@dataclass(frozen=True)
class MatrixJob:
    """One independently reportable benchmark sample."""

    phase: str
    profile: str
    scenario: str
    backend: str
    repetition: int
    turns: int = 1
    direct: bool = False

    @property
    def sample_id(self) -> str:
        target = "direct" if self.direct else self.profile
        return (
            f"{self.phase}-{self.backend}-{target}-"
            f"{self.scenario.lower()}-{self.repetition:02d}"
        )


@dataclass
class ApiBudget:
    """Local stop guard for paid/limited remote API use.

    The attempt limit is enforceable before a request.  Provider token usage
    is known only after a response, so the request that reaches the token
    limit can make the observed total exceed ``max_tokens``.
    """

    max_attempts: int = 30
    max_tokens: int = 500_000
    attempts: int = 0
    tokens: int = 0

    def can_start(self) -> bool:
        return self.attempts < self.max_attempts and self.tokens < self.max_tokens

    def account(self, *, attempts: int, tokens: int) -> None:
        if attempts < 0 or tokens < 0:
            raise ValueError("budget counters cannot be negative")
        self.attempts += attempts
        self.tokens += tokens

    @property
    def exhausted(self) -> bool:
        return not self.can_start()


def startup_jobs(
    profiles: Iterable[str] = CONTROLLED_PROFILES,
    *,
    repetitions: int = 3,
) -> list[MatrixJob]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return [
        MatrixJob("startup", profile, "idle", "none", repetition)
        for profile in profiles
        for repetition in range(1, repetitions + 1)
    ]


def mock_jobs(
    profiles: Iterable[str] = CONTROLLED_PROFILES,
    *,
    repetitions: int = 3,
    include_s3: bool = True,
) -> list[MatrixJob]:
    """Return the deterministic mock matrix plus exact-wire direct replay."""

    jobs: list[MatrixJob] = []
    for profile in profiles:
        for scenario in REQUEST_SCENARIOS:
            for repetition in range(1, repetitions + 1):
                jobs.append(
                    MatrixJob(
                        "request",
                        profile,
                        scenario,
                        "mock",
                        repetition,
                    ),
                )
    if include_s3:
        for profile in ("full", "core"):
            if profile in profiles:
                jobs.append(
                    MatrixJob("request", profile, "S3", "mock", 1, turns=10),
                )
    return jobs


def dashscope_jobs(
    *,
    repetitions: int = 3,
    scenarios: Iterable[str] = REQUEST_SCENARIOS,
) -> list[MatrixJob]:
    """Return remote jobs in a rotated order to reduce time-of-day bias."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    selected_scenarios = tuple(dict.fromkeys(scenarios))
    if not selected_scenarios or any(
        scenario not in REQUEST_SCENARIOS for scenario in selected_scenarios
    ):
        raise ValueError("DashScope scenarios must be a non-empty subset of S1/S2")
    targets = (("direct", True), ("full", False), ("core", False))
    jobs: list[MatrixJob] = []
    for scenario_index, scenario in enumerate(selected_scenarios):
        for repetition in range(1, repetitions + 1):
            offset = (repetition - 1 + scenario_index) % len(targets)
            rotated = targets[offset:] + targets[:offset]
            for profile, direct in rotated:
                jobs.append(
                    MatrixJob(
                        "request",
                        profile,
                        scenario,
                        "dashscope",
                        repetition,
                        direct=direct,
                    ),
                )
    if "S1" in selected_scenarios:
        jobs.append(
            MatrixJob(
                "request",
                "stock_reference",
                "S1",
                "dashscope",
                1,
            ),
        )
    return jobs
