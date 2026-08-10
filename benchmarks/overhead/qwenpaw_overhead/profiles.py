"""Generate the QwenPaw 2.0.1 overhead ablation profiles.

This module deliberately edits only an :class:`~.isolation.IsolationPaths`
copy.  It does not import :mod:`qwenpaw`, because QwenPaw binds its working and
secret directories during import.  The generated JSON matches the 2.0.1
schema:

* agent ``channels``/``mcp`` may be null, preventing those services from being
  constructed;
* root ``channels``/``mcp``/``tools`` are non-optional, so ``core`` retains
  valid fallback objects and explicitly disables their optional components.

The controlled matrix is cumulative: ``full`` keeps the seed's features,
``no_skills`` removes skills, ``local_tools`` retains three deterministic
builtin tools (the active Memory manager still injects ``memory_search``), and
``core`` additionally removes Memory and optional runtime services so its wire
schema is exactly those three tools. All four disable TUI backend warmup and
pin generation parameters. The separate
``stock_reference`` profile rebases paths only and otherwise preserves the
seed, including default warmup and model parameters.

Configuration cannot remove the always-created CronManager or the two 2-second
config watchers, nor can it prevent built-in tool modules and skill metadata
from being imported/scanned.  Those residual costs must remain visible in a
core measurement.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from .isolation import IsolationPaths


PROFILE_NAMES = (
    "full",
    "no_skills",
    "local_tools",
    "core",
    "stock_reference",
)
_PROFILE_ALIASES = {"minimal": "core"}
LOCAL_TOOL_NAMES = frozenset(
    {"read_file", "execute_shell_command", "get_current_time"},
)

# QwenPaw 2.0.1 descriptor registry.  Explicit entries are required because a
# missing ToolsConfig entry is re-added with its code-defined default.
QWENPAW_2_0_1_BUILTIN_TOOLS = frozenset(
    {
        "append_file",
        "ast_search",
        "browser_use",
        "chat_with_agent",
        "check_agent_task",
        "delegate_external_agent",
        "desktop_screenshot",
        "edit_file",
        "execute_shell_command",
        "get_current_time",
        "get_token_usage",
        "glob_search",
        "grep_search",
        "list_agents",
        "materialize_skill",
        "read_file",
        "send_file_to_user",
        "set_user_timezone",
        "spawn_subagent",
        "submit_to_agent",
        "view_image",
        "view_video",
        "web_fetch",
        "web_search",
        "write_file",
    },
)

QWENPAW_2_0_1_CHANNELS = frozenset(
    {
        "console",
        "dingtalk",
        "discord",
        "feishu",
        "imessage",
        "mattermost",
        "matrix",
        "mqtt",
        "onebot",
        "qq",
        "sip",
        "slack",
        "telegram",
        "voice",
        "wechat",
        "wecom",
        "xiaoyi",
        "yuanbao",
    },
)

QWENPAW_2_0_1_EXTERNAL_ACP_AGENTS = frozenset(
    {"opencode", "qwen_code", "claude_code", "codex"},
)


class ProfileError(RuntimeError):
    """Raised when an isolated seed cannot form a safe benchmark profile."""


@dataclass(frozen=True)
class ProfileResult:
    """Canonical profile name, subprocess environment, and safe metadata."""

    name: str
    env: dict[str, str]
    metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"required configuration is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read JSON configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProfileError(f"JSON configuration must be an object: {path}")
    return payload


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = None
    try:
        previous_mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode or previous_mode or 0o600)
        os.replace(temporary, path)
    except Exception:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _rebase_string(value: str, paths: IsolationPaths) -> str:
    for source, destination in (
        (paths.seed_working_dir, paths.working_dir),
        (paths.seed_secret_dir, paths.secret_dir),
    ):
        if source is None:
            continue
        source_text = str(source)
        if value == source_text:
            return str(destination)
        if value.startswith(source_text + os.sep):
            return str(destination / value[len(source_text + os.sep) :])
    return value


def _rebase_value(value: Any, paths: IsolationPaths) -> Any:
    if isinstance(value, dict):
        return {key: _rebase_value(child, paths) for key, child in value.items()}
    if isinstance(value, list):
        return [_rebase_value(child, paths) for child in value]
    if isinstance(value, str):
        return _rebase_string(value, paths)
    return value


def _workspace_targets(
    root_config: dict[str, Any],
    paths: IsolationPaths,
) -> dict[str, Path]:
    agents = root_config.get("agents")
    if not isinstance(agents, dict):
        raise ProfileError("config.json is missing the agents object")
    profiles = agents.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ProfileError("config.json has no agent profiles")

    targets: dict[str, Path] = {}
    for raw_id, reference in profiles.items():
        agent_id = str(raw_id)
        if not isinstance(reference, dict):
            raise ProfileError(f"invalid profile reference for {agent_id!r}")
        candidate = paths.working_dir / "workspaces" / agent_id
        configured = reference.get("workspace_dir")
        if isinstance(configured, str) and configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = paths.working_dir / configured_path
            if _under(configured_path, paths.working_dir):
                candidate = configured_path
        resolved = candidate.resolve()
        if not _under(resolved, paths.working_dir) or not resolved.is_dir():
            raise ProfileError(
                f"agent {agent_id!r} workspace is not present in the isolated "
                f"seed: {resolved}",
            )
        reference["workspace_dir"] = str(resolved)
        targets[agent_id] = resolved
    return targets


def _disable_channels(value: Any) -> dict[str, Any]:
    channels = dict(value) if isinstance(value, dict) else {}
    for name in QWENPAW_2_0_1_CHANNELS.union(channels):
        current = channels.get(name)
        config = dict(current) if isinstance(current, dict) else {}
        config["enabled"] = False
        channels[name] = config
    return channels


def _configure_tools(
    value: Any,
    enabled_names: frozenset[str],
) -> tuple[dict[str, Any], int]:
    config = dict(value) if isinstance(value, dict) else {}
    raw_tools = config.get("builtin_tools")
    tools = dict(raw_tools) if isinstance(raw_tools, dict) else {}
    names = QWENPAW_2_0_1_BUILTIN_TOOLS.union(tools)
    for name in names:
        current = tools.get(name)
        entry = dict(current) if isinstance(current, dict) else {"name": name}
        entry["name"] = name
        entry["enabled"] = name in enabled_names
        tools[name] = entry
    config["builtin_tools"] = tools
    return config, len(names - enabled_names)


def _disable_acp(value: Any) -> dict[str, Any]:
    config = dict(value) if isinstance(value, dict) else {}
    raw_agents = config.get("agents")
    agents = dict(raw_agents) if isinstance(raw_agents, dict) else {}
    for name in QWENPAW_2_0_1_EXTERNAL_ACP_AGENTS.union(agents):
        current = agents.get(name)
        entry = dict(current) if isinstance(current, dict) else {}
        entry["enabled"] = False
        agents[name] = entry
    config["agents"] = agents
    return config


def _minimal_running(value: Any) -> dict[str, Any]:
    running = dict(value) if isinstance(value, dict) else {}
    running["memory_manager_backend"] = "none"
    light_context = running.get("light_context_config")
    light_context = dict(light_context) if isinstance(light_context, dict) else {}
    # Scroll injects recall_history tools after the builtin-tool allowlist.
    # Native context is required for core's wire schema to contain exactly
    # the three configured local tools.
    light_context["strategy"] = "native"
    running["light_context_config"] = light_context
    reme = running.get("reme_light_memory_config")
    reme = dict(reme) if isinstance(reme, dict) else {}
    reme["auto_memory_interval"] = 0
    reme["dream_cron_enabled"] = False
    search = reme.get("auto_memory_search_config")
    search = dict(search) if isinstance(search, dict) else {}
    search["enabled"] = False
    reme["auto_memory_search_config"] = search
    running["reme_light_memory_config"] = reme
    title = running.get("auto_title_config")
    title = dict(title) if isinstance(title, dict) else {}
    title["enabled"] = False
    running["auto_title_config"] = title
    return running


def _native_context_running(value: Any) -> dict[str, Any]:
    running = dict(value) if isinstance(value, dict) else {}
    light_context = running.get("light_context_config")
    light_context = dict(light_context) if isinstance(light_context, dict) else {}
    light_context["strategy"] = "native"
    running["light_context_config"] = light_context
    return running


def _core_root(config: dict[str, Any]) -> int:
    config["channels"] = _disable_channels(config.get("channels"))
    raw_mcp = config.get("mcp")
    mcp = dict(raw_mcp) if isinstance(raw_mcp, dict) else {}
    mcp["clients"] = {}
    config["mcp"] = mcp
    config["tools"], tool_count = _configure_tools(
        config.get("tools"),
        LOCAL_TOOL_NAMES,
    )
    config["acp"] = _disable_acp(config.get("acp"))
    config["plugins"] = {}
    config["skill_paths"] = []

    agents = config["agents"]
    agents["running"] = _minimal_running(agents.get("running"))
    defaults = agents.get("defaults")
    defaults = dict(defaults) if isinstance(defaults, dict) else {}
    defaults["heartbeat"] = {"enabled": False}
    agents["defaults"] = defaults
    return tool_count


def _core_agent(agent: dict[str, Any], workspace: Path) -> int:
    agent["workspace_dir"] = str(workspace)
    agent["channels"] = None
    agent["mcp"] = None
    agent["heartbeat"] = {"enabled": False}
    agent["running"] = _minimal_running(agent.get("running"))
    agent["tools"], tool_count = _configure_tools(
        agent.get("tools"),
        LOCAL_TOOL_NAMES,
    )
    agent["acp"] = _disable_acp(agent.get("acp"))
    plan = agent.get("plan")
    plan = dict(plan) if isinstance(plan, dict) else {}
    plan["enabled"] = False
    agent["plan"] = plan
    coding = agent.get("coding_mode")
    coding = dict(coding) if isinstance(coding, dict) else {}
    coding["enabled"] = False
    coding["project_dir"] = None
    agent["coding_mode"] = coding
    return tool_count


def _disable_skill_manifest(path: Path) -> int:
    if not path.is_file():
        return 0
    payload = _load_json(path)
    raw_skills = payload.get("skills")
    if not isinstance(raw_skills, dict):
        return 0
    changed = 0
    for name, value in raw_skills.items():
        entry = dict(value) if isinstance(value, dict) else {}
        if entry.get("enabled") is not False:
            changed += 1
        entry["enabled"] = False
        raw_skills[name] = entry
    _atomic_json(path, payload)
    return changed


def _disable_driver_cards(workspace: Path) -> int:
    root = workspace / "drivers"
    if not root.is_dir():
        return 0
    count = 0
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ProfileError(f"cannot parse DriverCard {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProfileError(f"DriverCard must be a mapping: {path}")
        if payload.get("enabled") is not False:
            count += 1
        payload["enabled"] = False
        serialized = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        )
        temporary = path.with_name(f".{path.name}.profile.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return count


def _empty_jobs(workspace: Path) -> int:
    path = workspace / "jobs.json"
    previous = 0
    if path.is_file():
        payload = _load_json(path)
        jobs = payload.get("jobs")
        previous = len(jobs) if isinstance(jobs, list) else 0
    _atomic_json(path, {"version": 1, "jobs": []})
    return previous


def _active_selection(
    root: dict[str, Any],
    agents: dict[str, dict[str, Any]],
    paths: IsolationPaths,
) -> tuple[str, str] | None:
    active_path = paths.secret_dir / "providers" / "active_model.json"
    if active_path.is_file():
        payload = _load_json(active_path)
        provider = str(payload.get("provider_id") or "")
        model = str(payload.get("model") or "")
        if provider and model:
            return provider, model
    active_id = str(root.get("agents", {}).get("active_agent") or "")
    candidates = [agents.get(active_id), *agents.values()]
    for agent in candidates:
        if not isinstance(agent, dict):
            continue
        value = agent.get("active_model")
        if not isinstance(value, dict):
            continue
        provider = str(value.get("provider_id") or "")
        model = str(value.get("model") or "")
        if provider and model:
            return provider, model
    return None


def _resolve_selection(
    requested: str | None,
    current: tuple[str, str] | None,
) -> tuple[str, str] | None:
    if requested is None:
        return current
    requested = requested.strip()
    if not requested or any(ord(character) < 32 for character in requested):
        raise ProfileError("model must be a non-empty printable value")
    if "/" in requested:
        provider, model = requested.split("/", 1)
    elif current is not None:
        provider, model = current[0], requested
    else:
        raise ProfileError(
            "model must use provider/model when the seed has no active model",
        )
    if not provider or not model:
        raise ProfileError("model must use provider/model")
    return provider, model


def _apply_model_selection(
    selection: tuple[str, str] | None,
    agents: dict[str, dict[str, Any]],
    paths: IsolationPaths,
) -> None:
    if selection is None:
        return
    provider, model = selection
    value = {"provider_id": provider, "model": model}
    for agent in agents.values():
        agent["active_model"] = dict(value)
    active_path = paths.secret_dir / "providers" / "active_model.json"
    _atomic_json(active_path, value, mode=0o600)


def _validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProfileError("base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ProfileError("base_url must not embed credentials")
    return base_url.strip().rstrip("/")


def _provider_config_path(secret_dir: Path, provider: str) -> Path:
    if not provider or provider in {".", ".."} or "/" in provider or "\\" in provider:
        raise ProfileError(f"unsafe provider id: {provider!r}")
    candidates = [
        secret_dir / "providers" / kind / f"{provider}.json"
        for kind in ("builtin", "custom", "plugin")
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        detail = "not found" if not existing else "ambiguous"
        raise ProfileError(
            f"provider config for {provider!r} is {detail}; base_url override "
            "requires one existing cloned provider JSON",
        )
    return existing[0]


def _apply_controlled_model_parameters(
    selection: tuple[str, str],
    paths: IsolationPaths,
) -> None:
    """Pin parameters at both provider and model precedence levels."""

    provider, model_id = selection
    path = _provider_config_path(paths.secret_dir, provider)
    payload = _load_json(path)
    controlled_kwargs = {
        "max_tokens": 128,
        "temperature": 0,
        # DashScopeProvider consumes this spelling and translates it to the
        # native/OpenAI-compatible request shape.  ``enable_thinking`` would
        # leak through as an unsupported top-level OpenAI client argument.
        "thinking_enable": False,
    }
    provider_kwargs = payload.get("generate_kwargs")
    provider_kwargs = (
        dict(provider_kwargs) if isinstance(provider_kwargs, dict) else {}
    )
    provider_kwargs.update(controlled_kwargs)
    provider_kwargs.pop("enable_thinking", None)
    provider_kwargs.pop("stream", None)
    provider_kwargs.pop("thinking_budget", None)
    provider_kwargs.pop("reasoning_effort", None)
    payload["generate_kwargs"] = provider_kwargs

    found = False
    for collection_name in ("models", "extra_models"):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            continue
        for raw_entry in collection:
            if not isinstance(raw_entry, dict) or raw_entry.get("id") != model_id:
                continue
            found = True
            raw_entry["max_tokens"] = 128
            raw_entry["thinking_enabled"] = False
            raw_entry["thinking_budget"] = None
            raw_entry["reasoning_effort"] = None
            model_kwargs = raw_entry.get("generate_kwargs")
            model_kwargs = (
                dict(model_kwargs) if isinstance(model_kwargs, dict) else {}
            )
            model_kwargs.update(controlled_kwargs)
            model_kwargs.pop("enable_thinking", None)
            model_kwargs.pop("stream", None)
            model_kwargs.pop("thinking_budget", None)
            model_kwargs.pop("reasoning_effort", None)
            raw_entry["generate_kwargs"] = model_kwargs
    if not found:
        raise ProfileError(
            f"model {provider}/{model_id} is absent from the cloned provider "
            "models/extra_models lists",
        )
    _atomic_json(path, payload, mode=0o600)


def _apply_base_url(
    base_url: str | None,
    selection: tuple[str, str] | None,
    paths: IsolationPaths,
) -> str | None:
    if base_url is None:
        return None
    if selection is None:
        raise ProfileError("base_url override requires an active model/provider")
    normalized = _validate_base_url(base_url)
    path = _provider_config_path(paths.secret_dir, selection[0])
    payload = _load_json(path)
    payload["base_url"] = normalized
    _atomic_json(path, payload, mode=0o600)
    parsed = urlsplit(normalized)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def generate_profile(
    paths: IsolationPaths,
    name: str,
    model: str | None = None,
    base_url: str | None = None,
) -> ProfileResult:
    """Materialize a named profile inside one isolated run.

    Controlled profiles accept ``provider/model`` or a model id that reuses
    the seed's active provider.  ``stock_reference`` rejects model overrides,
    but may receive a relay ``base_url`` as transport-only instrumentation.
    A base URL is written only to an existing cloned provider JSON; this helper
    never invents credentials or a provider schema.  Each ablation must begin
    from a fresh seed copy.
    """

    if paths._closed:  # pylint: disable=protected-access
        raise ProfileError("cannot generate a profile in a closed isolation")
    requested_name = name.strip().lower()
    normalized_name = _PROFILE_ALIASES.get(requested_name, requested_name)
    if normalized_name not in PROFILE_NAMES:
        raise ProfileError(
            f"unknown profile {name!r}; choose {', '.join(PROFILE_NAMES)}",
        )
    is_stock = normalized_name == "stock_reference"
    if is_stock and model is not None:
        raise ProfileError(
            "stock_reference does not accept a model override",
        )

    config_path = paths.working_dir / "config.json"
    root_config = _rebase_value(_load_json(config_path), paths)
    workspaces = _workspace_targets(root_config, paths)
    agent_configs: dict[str, dict[str, Any]] = {}
    for agent_id, workspace in workspaces.items():
        agent_path = workspace / "agent.json"
        agent = _rebase_value(_load_json(agent_path), paths)
        agent["workspace_dir"] = str(workspace)
        agent_configs[agent_id] = agent

    if normalized_name == "core":
        if "default" not in agent_configs:
            raise ProfileError("core profile requires a default agent")
        root_agents = root_config["agents"]
        root_agents["profiles"] = {
            "default": root_agents["profiles"]["default"],
        }
        root_agents["agent_order"] = ["default"]
        root_agents["active_agent"] = "default"
        workspaces = {"default": workspaces["default"]}
        agent_configs = {"default": agent_configs["default"]}

    counts: dict[str, int] = {
        "agents": len(agent_configs),
        "tools_disabled": 0,
        "skills_disabled": 0,
        "driver_cards_disabled": 0,
        "cron_jobs_removed": 0,
    }

    if normalized_name in {"no_skills", "local_tools", "core"}:
        for agent_id, agent in agent_configs.items():
            workspace = workspaces[agent_id]
            counts["skills_disabled"] += _disable_skill_manifest(
                workspace / "skill.json",
            )
        counts["skills_disabled"] += _disable_skill_manifest(
            paths.working_dir / "skill_pool" / "skill.json",
        )

    if normalized_name in {"local_tools", "core"}:
        root_config["tools"], counts["tools_disabled"] = _configure_tools(
            root_config.get("tools"),
            LOCAL_TOOL_NAMES,
        )
        for agent in agent_configs.values():
            agent["tools"], disabled = _configure_tools(
                agent.get("tools"),
                LOCAL_TOOL_NAMES,
            )
            counts["tools_disabled"] = max(
                counts["tools_disabled"],
                disabled,
            )
        if normalized_name == "local_tools":
            root_agents = root_config.get("agents")
            if isinstance(root_agents, dict):
                root_agents["running"] = _native_context_running(
                    root_agents.get("running"),
                )
            for agent in agent_configs.values():
                agent["running"] = _native_context_running(
                    agent.get("running"),
                )

    if normalized_name == "core":
        counts["tools_disabled"] = max(
            counts["tools_disabled"],
            _core_root(root_config),
        )
        for agent_id, agent in agent_configs.items():
            workspace = workspaces[agent_id]
            counts["tools_disabled"] = max(
                counts["tools_disabled"],
                _core_agent(agent, workspace),
            )
            counts["driver_cards_disabled"] += _disable_driver_cards(workspace)
            counts["cron_jobs_removed"] += _empty_jobs(workspace)

    current = _active_selection(root_config, agent_configs, paths)
    if is_stock:
        selection = current
        safe_base_url = _apply_base_url(base_url, selection, paths)
    else:
        # Every model-authored tool call must cross the ACP permission bridge.
        # The benchmark client then approves only the exact fixed printf.  A
        # post-hoc command check would be too late for remote-model S2.
        for agent in agent_configs.values():
            agent["approval_level"] = "STRICT"
            running = agent.get("running")
            running = dict(running) if isinstance(running, dict) else {}
            running["approval_level"] = "STRICT"
            agent["running"] = running
        selection = _resolve_selection(model, current)
        if selection is None:
            raise ProfileError(
                "controlled profiles require an active model or model=provider/model",
            )
        _apply_model_selection(selection, agent_configs, paths)
        _apply_controlled_model_parameters(selection, paths)
        safe_base_url = _apply_base_url(base_url, selection, paths)

    _atomic_json(config_path, root_config)
    for agent_id, agent in agent_configs.items():
        _atomic_json(workspaces[agent_id] / "agent.json", agent)

    env = dict(paths.env)
    if is_stock:
        env.pop("PAW_DISABLE_BACKEND_WARMUP", None)
    else:
        env["PAW_DISABLE_BACKEND_WARMUP"] = "1"

    metadata: dict[str, Any] = {
        "profile": normalized_name,
        "requested_profile": requested_name,
        "schema_target": "qwenpaw-2.0.1",
        "workspace_count": len(workspaces),
        "warmup_enabled": is_stock,
        "ablation_counts": counts,
        "active_model": (
            {"provider_id": selection[0], "model": selection[1]}
            if selection is not None
            else None
        ),
        "base_url_override": safe_base_url,
        "transport_instrumented": base_url is not None,
        "generation_parameters": (
            None
            if is_stock
            else {
                "max_tokens": 128,
                "temperature": 0,
                "thinking": False,
                "thinking_enabled": False,
                "thinking_budget": None,
                "reasoning_effort": None,
                "stream": True,
            }
        ),
        "tool_approval_level": None if is_stock else "STRICT",
        "configured_builtin_tools": (
            sorted(LOCAL_TOOL_NAMES)
            if normalized_name in {"local_tools", "core"}
            else None
        ),
        "runtime_dynamic_tools_expected": (
            ["memory_search"]
            if normalized_name == "local_tools"
            else []
            if normalized_name == "core"
            else None
        ),
        "known_residual_services": (
            [
                "cron_manager",
                "agent_config_watcher",
                "driver_config_watcher",
                "builtin_imports",
                "skill_metadata_scan",
            ]
            if normalized_name == "core"
            else []
        ),
    }
    return ProfileResult(
        name=normalized_name,
        env=env,
        metadata=metadata,
    )


__all__ = [
    "PROFILE_NAMES",
    "ProfileError",
    "ProfileResult",
    "generate_profile",
]
