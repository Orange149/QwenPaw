from __future__ import annotations

from benchmarks.overhead.qwenpaw_overhead.token_accounting import (
    TokenEstimator,
    analyze_request,
)


def _words(value: str) -> int:
    return len(value.split())


def test_request_decomposition_splits_skills_history_and_tools() -> None:
    body = {
        "messages": [
            {
                "role": "system",
                "content": "base words <agent-skills>skill words</agent-skills>",
            },
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "new question"},
        ],
        "tools": [{"type": "function", "function": {"name": "demo"}}],
    }
    result = analyze_request(
        body,
        provider_prompt_tokens=100,
        estimator=TokenEstimator("words", _words),
    )
    assert result["system_tokens_estimate"] == 2
    assert result["skills_tokens_estimate"] == 2
    assert result["tool_results_tokens_estimate"] == 2
    assert result["current_user_tokens_estimate"] == 2
    assert result["tokenizer_residual_tokens"] == (
        100 - result["estimated_subtotal_tokens"]
    )

