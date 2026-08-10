"""Deterministic synthetic workloads used by the overhead benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


SHELL_COMMAND = "printf QWENPAW_BENCH_42"
SHELL_OUTPUT = "QWENPAW_BENCH_42"
WARMUP_PROMPT = (
    "Warm up the QwenPaw backend for an interactive terminal session. "
    "Reply with exactly: ready. Do not call tools."
)


@dataclass(frozen=True)
class Scenario:
    """A benchmark workload with public, non-sensitive expected output."""

    name: str
    prompts: tuple[str, ...]
    expected_text: str | None = None
    expected_tool: str | None = None
    expected_command: str | None = None

    @property
    def prompt_sha256(self) -> str:
        joined = "\n\0\n".join(self.prompts).encode("utf-8")
        return sha256(joined).hexdigest()


# Kept stable so results from different builds are comparable.  The filler is
# deliberately mundane and contains no machine/user data.  With common Qwen
# tokenizers the complete prompt is close to 256 tokens; the authoritative
# prompt token count still comes from the provider's usage record.
_S1_FILLER = " ".join(
    (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    ).split()
    * 9
)

S1 = Scenario(
    name="S1",
    prompts=(
        "QWENPAW_OVERHEAD_S1. Do not call tools. Read the following fixed "
        "synthetic text and reply with only the checksum after CHECKSUM=. "
        f"TEXT={_S1_FILLER} CHECKSUM=QP7F3A",
    ),
    expected_text="QP7F3A",
)

S2 = Scenario(
    name="S2",
    prompts=(
        "QWENPAW_OVERHEAD_S2. Call execute_shell_command exactly once with "
        f"the exact command `{SHELL_COMMAND}`. Do not execute any other "
        "command. After the tool result, reply with only QWENPAW_BENCH_42.",
    ),
    expected_text=SHELL_OUTPUT,
    expected_tool="execute_shell_command",
    expected_command=SHELL_COMMAND,
)


def _s3_prompts() -> tuple[str, ...]:
    prompts: list[str] = []
    for turn in range(1, 11):
        key = f"K{turn:02d}"
        value = f"V{turn * 7919 % 100000:05d}"
        if turn == 1:
            recall = "Acknowledge by replying only STORED."
        else:
            previous = turn - 1
            previous_value = f"V{previous * 7919 % 100000:05d}"
            recall = (
                f"Also recall K{previous:02d}; reply only "
                f"K{previous:02d}={previous_value};STORED."
            )
        prompts.append(
            "QWENPAW_OVERHEAD_S3. Do not call tools. "
            f"Remember {key}={value} for this session. {recall}",
        )
    return tuple(prompts)


S3 = Scenario(name="S3", prompts=_s3_prompts())
SCENARIOS = {scenario.name: scenario for scenario in (S1, S2, S3)}


def get_scenario(name: str) -> Scenario:
    """Return a named scenario or raise a useful error."""

    normalized = name.upper()
    try:
        return SCENARIOS[normalized]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"unknown scenario {name!r}; choose {choices}") from exc
