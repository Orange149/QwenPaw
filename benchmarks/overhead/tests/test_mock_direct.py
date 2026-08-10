"""Contract tests for the deterministic model and direct baseline client."""

from __future__ import annotations

import json

import httpx

from benchmarks.overhead.qwenpaw_overhead.direct_client import run_request
from benchmarks.overhead.qwenpaw_overhead.mock_server import start_server
from benchmarks.overhead.qwenpaw_overhead.scenarios import (
    S1,
    S2,
    S3,
    SHELL_COMMAND,
    SHELL_OUTPUT,
)


def _shell_body() -> dict:
    return {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": S2.prompts[0]}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell_command",
                    "description": "Execute the fixed benchmark command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_mock_s1_stream_reports_synthetic_usage_and_clears_wire_body() -> None:
    handle = None
    with start_server(scenario="S1", reasoning_text="fixture thought") as server:
        handle = server
        result = run_request(
            base_url=server.base_url,
            model="fixture-model",
            prompt=S1.prompts[0],
            trust_env=False,
        )

        assert result["success"] is True
        assert result["answer_text"] == S1.expected_text
        assert result["reasoning_text"] == "fixture thought"
        assert result["usage"] == {
            "prompt_tokens": 64,
            "completion_tokens": 8,
            "total_tokens": 72,
        }
        done = [
            event
            for event in server.snapshot_events()
            if event["event"] == "mock_stream_done"
        ]
        assert done[0]["data"]["usage"] == result["usage"]
        usage_events = [
            event
            for event in server.snapshot_events()
            if event["event"] == "mock_usage"
        ]
        assert usage_events[0]["data"]["total_tokens"] == 72
        assert usage_events[0]["data"]["synthetic"] is True
        assert server.last_request_body()["messages"][0]["content"] == (
            S1.prompts[0]
        )

    assert handle is not None
    assert handle.request_bodies == {}


def test_direct_s2_executes_only_exact_printf_and_sums_two_round_usage() -> None:
    with start_server(scenario="S2") as server:
        result = run_request(
            base_url=server.base_url,
            request_body=_shell_body(),
            allow_exact_shell=True,
            trust_env=False,
        )

        assert result["success"] is True
        assert result["answer_text"] == SHELL_OUTPUT
        assert result["model_attempts"] == 2
        assert result["tool_execution_count"] == 1
        assert result["usage_by_attempt"] == [
            {
                "prompt_tokens": 64,
                "completion_tokens": 8,
                "total_tokens": 72,
            },
            {
                "prompt_tokens": 96,
                "completion_tokens": 6,
                "total_tokens": 102,
            },
        ]
        assert result["usage"] == {
            "prompt_tokens": 160,
            "completion_tokens": 14,
            "total_tokens": 174,
        }
        assert server.request_count == 2
        assert server.tool_call_emissions == 1
        second = server.request_body(2)
        assert [message["role"] for message in second["messages"]] == [
            "user",
            "assistant",
            "tool",
        ]
        arguments = second["messages"][1]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert json.loads(arguments) == {"command": SHELL_COMMAND}
        assert second["messages"][2]["content"] == SHELL_OUTPUT


def test_direct_does_not_execute_tool_without_explicit_opt_in() -> None:
    with start_server(scenario="S2") as server:
        result = run_request(
            base_url=server.base_url,
            request_body=_shell_body(),
            trust_env=False,
        )

        assert result["success"] is True
        assert result["model_attempts"] == 1
        assert result["tool_execution_count"] == 0
        assert server.request_count == 1


def test_direct_rejects_non_fixed_expected_command_without_execution() -> None:
    with start_server(scenario="S2") as server:
        result = run_request(
            base_url=server.base_url,
            request_body=_shell_body(),
            allow_exact_shell=True,
            expected_command="printf SOMETHING_ELSE",
            trust_env=False,
        )

        assert result["success"] is False
        assert result["error_type"] == "unsafe_or_unexpected_tool_call"
        assert result["tool_execution_count"] == 0
        assert result["model_attempts"] == 1


def test_mock_rejects_a_prompt_mismatch_instead_of_returning_expected_text() -> None:
    with start_server(scenario="S1") as server:
        result = run_request(
            base_url=server.base_url,
            model="fixture-model",
            prompt="not the benchmark prompt",
            trust_env=False,
        )

        assert result["success"] is False
        assert server.validation_failures == 1
        validation = [
            event
            for event in server.snapshot_events()
            if event["event"] == "mock_request_validated"
        ]
        assert validation[-1]["data"]["valid"] is False
        assert validation[-1]["data"]["error_codes"] == [
            "s1_user_prompt_mismatch",
        ]


def test_mock_s3_requires_ordered_user_and_assistant_history() -> None:
    with start_server(scenario="S3") as server:
        history: list[dict[str, str]] = []
        for turn, prompt in enumerate(S3.prompts, 1):
            history.append({"role": "user", "content": prompt})
            response = httpx.post(
                f"{server.base_url}/chat/completions",
                json={
                    "model": "fixture-model",
                    "messages": history,
                    "stream": False,
                },
                timeout=5.0,
                trust_env=False,
            )
            assert response.status_code == 200
            answer = response.json()["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": answer})
            assert server.request_count == turn
        assert server.validation_failures == 0

    with start_server(scenario="S3") as server:
        first = httpx.post(
            f"{server.base_url}/chat/completions",
            json={
                "model": "fixture-model",
                "messages": [{"role": "user", "content": S3.prompts[0]}],
                "stream": False,
            },
            timeout=5.0,
            trust_env=False,
        )
        assert first.status_code == 200
        missing_history = httpx.post(
            f"{server.base_url}/chat/completions",
            json={
                "model": "fixture-model",
                "messages": [{"role": "user", "content": S3.prompts[1]}],
                "stream": False,
            },
            timeout=5.0,
            trust_env=False,
        )
        assert missing_history.status_code == 422
        assert server.validation_failures == 1


def test_direct_rejects_mock_supplied_unsafe_command_without_execution(
    tmp_path,
) -> None:
    marker = tmp_path / "must-not-exist"
    with start_server(
        scenario="S2",
        tool_command=f"touch {marker}",
    ) as server:
        result = run_request(
            base_url=server.base_url,
            request_body=_shell_body(),
            allow_exact_shell=True,
            trust_env=False,
        )

        assert result["success"] is False
        assert result["error_type"] == "unsafe_or_unexpected_tool_call"
        assert result["tool_execution_count"] == 0
        assert marker.exists() is False


def _approval_generalization_body(target: str) -> dict:
    return {
        "model": "fixture-model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generalize a single tool-call target into a "
                    "conservative glob pattern so that future, similar "
                    "calls are auto-approved without asking again. You "
                    "MUST output ONLY the glob pattern — no explanation, "
                    "no quotes, no backticks, no tool name, no parentheses, "
                    "no leading/trailing whitespace."
                ),
            },
            {
                "role": "user",
                "content": (
                    "tool_name: Bash\n"
                    "tool_type: shell\n"
                    f"target: {target}\n\n"
                    "This is a shell command. Generalize CONSERVATIVELY: "
                    "replace varying arguments with '*' while KEEPING the "
                    "command name and any subcommand. Examples: 'git "
                    "status' -> 'git *', 'npm run build' -> 'npm run *'. "
                    "Do NOT widen destructive commands (rm, dd, mkfs, "
                    "sudo, chmod 777, > /dev/...) — return them unchanged. "
                    "Never output a bare '*'.\n\n"
                    "glob pattern:"
                ),
            },
        ],
        "stream": False,
    }


def test_mock_treats_exact_shell_approval_generalization_as_auxiliary() -> None:
    with start_server(scenario="S2") as server:
        response = httpx.post(
            f"{server.base_url}/chat/completions",
            json=_approval_generalization_body(SHELL_COMMAND),
            timeout=5.0,
            trust_env=False,
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == (
            SHELL_COMMAND
        )
        assert server.primary_request_count == 0
        assert server.auxiliary_request_count == 1
        assert server.validation_failures == 0


def test_mock_rejects_generalization_for_any_other_shell_target() -> None:
    with start_server(scenario="S2") as server:
        response = httpx.post(
            f"{server.base_url}/chat/completions",
            json=_approval_generalization_body("touch /tmp/not-allowed"),
            timeout=5.0,
            trust_env=False,
        )

        assert response.status_code == 422
        assert server.primary_request_count == 0
        assert server.auxiliary_request_count == 1
        assert server.validation_failures == 1
