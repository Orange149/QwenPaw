"""Sanitized, approximate prompt-cost decomposition for captured wire bodies."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_SKILLS_BLOCK = re.compile(
    r"<agent-skills>.*?</agent-skills>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class TokenEstimator:
    """A local tokenizer when available, otherwise QwenPaw's byte heuristic."""

    name: str
    count: Callable[[str], int]


def load_estimator() -> TokenEstimator:
    """Load QwenPaw's bundled tokenizer without importing QwenPaw itself."""

    spec = importlib.util.find_spec("qwenpaw")
    if spec is not None and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        tokenizer_json = root / "tokenizer" / "tokenizer.json"
        if tokenizer_json.is_file():
            try:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(tokenizer_json))
                return TokenEstimator(
                    "qwenpaw-bundled-tokenizer",
                    lambda text: len(tokenizer.encode(text).ids) if text else 0,
                )
            except (ImportError, OSError, ValueError):
                pass

    return TokenEstimator(
        "utf8-bytes-div-4",
        lambda text: int(len(text.encode("utf-8")) / 4 + 0.5) if text else 0,
    )


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def analyze_request(
    body: dict[str, Any],
    *,
    provider_prompt_tokens: int | None = None,
    estimator: TokenEstimator | None = None,
) -> dict[str, Any]:
    """Return counts only; raw message text is never included in the result."""

    token_counter = estimator or load_estimator()
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    tools = body.get("tools")
    tools = tools if isinstance(tools, list) else []

    last_user_index = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index

    buckets = {
        "system": 0,
        "skills": 0,
        "tool_schema": 0,
        "history": 0,
        "tool_results": 0,
        "current_user": 0,
    }
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = _content_text(message.get("content"))
        if role == "system":
            skill_blocks = _SKILLS_BLOCK.findall(content)
            skills_text = "\n".join(skill_blocks)
            system_text = _SKILLS_BLOCK.sub("", content)
            buckets["skills"] += token_counter.count(skills_text)
            buckets["system"] += token_counter.count(system_text)
        elif role == "tool":
            buckets["tool_results"] += token_counter.count(content)
        elif role == "user" and index == last_user_index:
            buckets["current_user"] += token_counter.count(content)
        else:
            # Include assistant tool-call JSON and prior user turns.
            serialized = json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            buckets["history"] += token_counter.count(serialized)

    tools_text = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    buckets["tool_schema"] = token_counter.count(tools_text)
    tool_names = sorted(
        {
            str(function["name"])
            for tool in tools
            if isinstance(tool, dict)
            and isinstance((function := tool.get("function")), dict)
            and isinstance(function.get("name"), str)
        },
    )
    subtotal = sum(buckets.values())
    residual = (
        provider_prompt_tokens - subtotal
        if provider_prompt_tokens is not None
        else None
    )
    return {
        "estimator": token_counter.name,
        **{f"{name}_tokens_estimate": value for name, value in buckets.items()},
        "framework_fixed_tokens_estimate": (
            buckets["system"] + buckets["skills"] + buckets["tool_schema"]
        ),
        "estimated_subtotal_tokens": subtotal,
        "provider_prompt_tokens": provider_prompt_tokens,
        "tokenizer_residual_tokens": residual,
        "message_count": len(messages),
        "tool_count": len(tools),
        "tool_names": tool_names,
    }
