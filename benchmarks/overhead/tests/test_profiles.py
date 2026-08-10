from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.overhead.qwenpaw_overhead.isolation import create_isolated_run
from benchmarks.overhead.qwenpaw_overhead.profiles import (
    LOCAL_TOOL_NAMES,
    PROFILE_NAMES,
    ProfileError,
    generate_profile,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _agent(agent_id: str, workspace: Path) -> dict:
    return {
        "id": agent_id,
        "name": agent_id.title(),
        "workspace_dir": str(workspace),
        "channels": {"console": {"enabled": True}},
        "mcp": {
            "clients": {
                "demo": {
                    "name": "demo",
                    "enabled": True,
                    "transport": "stdio",
                    "command": "echo",
                    "args": [],
                    "env": {},
                },
            },
        },
        "heartbeat": {"enabled": True},
        "running": {
            "memory_manager_backend": "remelight",
            "light_context_config": {"strategy": "scroll"},
            "reme_light_memory_config": {
                "auto_memory_interval": 5,
                "dream_cron_enabled": True,
                "auto_memory_search_config": {"enabled": True},
            },
            "auto_title_config": {"enabled": True},
        },
        "active_model": {
            "provider_id": "dashscope",
            "model": "qwen3.7-plus",
        },
        "tools": {
            "builtin_tools": {
                "browser_use": {"name": "browser_use", "enabled": True},
                "read_file": {"name": "read_file", "enabled": True},
                "execute_shell_command": {
                    "name": "execute_shell_command",
                    "enabled": True,
                },
                "get_current_time": {
                    "name": "get_current_time",
                    "enabled": True,
                },
            },
        },
        "acp": {"agents": {"opencode": {"enabled": True}}},
        "plan": {"enabled": True},
        "coding_mode": {"enabled": True, "project_dir": str(workspace)},
    }


def _profile_seed(tmp_path: Path) -> tuple[Path, Path, Path]:
    working = tmp_path / "seed-working"
    secret = tmp_path / "seed-secret"
    base = tmp_path / "runs"
    base.mkdir()
    profiles = {}
    for agent_id in ("default", "qa"):
        workspace = working / "workspaces" / agent_id
        workspace.mkdir(parents=True)
        profiles[agent_id] = {
            "id": agent_id,
            "workspace_dir": str(workspace),
            "enabled": True,
        }
        _write_json(workspace / "agent.json", _agent(agent_id, workspace))
        _write_json(
            workspace / "skill.json",
            {
                "skills": {
                    "guidance": {"enabled": True, "channels": ["all"]},
                },
            },
        )
        _write_json(
            workspace / "jobs.json",
            {"version": 1, "jobs": [{"id": "scheduled"}]},
        )
        card = {
            "name": "demo",
            "protocol": "mcp",
            "endpoint": {
                "transport": "stdio",
                "command": "echo",
                "args": [],
            },
            "credentials": {},
            "config": {},
            "enabled": True,
        }
        driver_path = workspace / "drivers" / "mcp" / "demo.yaml"
        driver_path.parent.mkdir(parents=True)
        driver_path.write_text(
            yaml.safe_dump(card, sort_keys=False),
            encoding="utf-8",
        )

    _write_json(
        working / "config.json",
        {
            "channels": {"console": {"enabled": True}},
            "mcp": {"clients": {}},
            "agents": {
                "active_agent": "default",
                "agent_order": ["default", "qa"],
                "profiles": profiles,
            },
            "tools": {
                "builtin_tools": {
                    "browser_use": {"name": "browser_use", "enabled": True},
                },
            },
        },
    )
    _write_json(
        working / "skill_pool" / "skill.json",
        {"skills": {"guidance": {"enabled": True}}},
    )
    _write_json(
        secret / "providers" / "active_model.json",
        {"provider_id": "dashscope", "model": "qwen3.7-plus"},
    )
    _write_json(
        secret / "providers" / "builtin" / "dashscope.json",
        {
            "id": "dashscope",
            "base_url": "https://dashscope.example/v1",
            "api_key": "ENC:not-a-real-secret",
            "generate_kwargs": {
                "top_p": 0.8,
                "max_tokens": 999,
                "enable_thinking": True,
                "stream": False,
            },
            "models": [
                {
                    "id": "qwen3.7-plus",
                    "name": "Qwen3.7 Plus",
                    "max_tokens": 8192,
                    "generate_kwargs": {
                        "max_tokens": 4096,
                        "enable_thinking": True,
                        "stream": False,
                    },
                    "thinking_enabled": True,
                    "thinking_budget": 2048,
                    "reasoning_effort": "high",
                },
            ],
            "extra_models": [],
        },
    )
    return working, secret, base


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profiles_are_valid_qwenpaw_2_0_1_configs(
    tmp_path: Path,
    profile: str,
) -> None:
    working, secret, base = _profile_seed(tmp_path)
    with create_isolated_run(working, secret, profile, base) as paths:
        result = generate_profile(paths, profile)
        root = _read(paths.working_dir / "config.json")
        validation = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "from pathlib import Path; "
                    "from qwenpaw.config.config import AgentProfileConfig,Config; "
                    "root=json.loads(Path(sys.argv[1]).read_text()); "
                    "Config.model_validate(root); "
                    "[AgentProfileConfig.model_validate(json.loads((Path(v['workspace_dir'])/'agent.json').read_text())) "
                    "for v in root['agents']['profiles'].values()]"
                ),
                str(paths.working_dir / "config.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=result.env,
            timeout=30,
        )
        assert validation.returncode == 0, validation.stderr
        assert result.env.get("PAW_DISABLE_BACKEND_WARMUP") == (
            None if profile == "stock_reference" else "1"
        )


def test_controlled_profile_pins_effective_provider_and_model_parameters(
    tmp_path: Path,
) -> None:
    working, secret, base = _profile_seed(tmp_path)
    with create_isolated_run(working, secret, "full", base) as paths:
        result = generate_profile(
            paths,
            "full",
            model="dashscope/qwen3.7-plus",
            base_url="http://127.0.0.1:9123/v1?ignored=metadata",
        )
        provider = _read(
            paths.secret_dir / "providers/builtin/dashscope.json",
        )
        kwargs = provider["generate_kwargs"]
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 0
        assert kwargs["thinking_enable"] is False
        assert "enable_thinking" not in kwargs
        assert "stream" not in kwargs
        assert kwargs["top_p"] == 0.8
        model = provider["models"][0]
        assert model["max_tokens"] == 128
        assert model["thinking_enabled"] is False
        assert model["thinking_budget"] is None
        assert model["reasoning_effort"] is None
        assert model["generate_kwargs"]["max_tokens"] == 128
        assert model["generate_kwargs"]["thinking_enable"] is False
        assert provider["base_url"] == "http://127.0.0.1:9123/v1?ignored=metadata"
        assert result.metadata["base_url_override"] == "http://127.0.0.1:9123/v1"
        assert result.metadata["generation_parameters"]["stream"] is True
        assert "api_key" not in json.dumps(result.metadata)


def test_cumulative_ablation_matrix(tmp_path: Path) -> None:
    for profile in PROFILE_NAMES:
        case = tmp_path / profile
        case.mkdir()
        working, secret, base = _profile_seed(case)
        with create_isolated_run(working, secret, profile, base) as paths:
            generate_profile(paths, profile)
            root = _read(paths.working_dir / "config.json")
            default_ws = paths.working_dir / "workspaces/default"
            agent = _read(default_ws / "agent.json")
            skill = _read(default_ws / "skill.json")["skills"]["guidance"]
            enabled_tools = {
                name
                for name, config in agent["tools"]["builtin_tools"].items()
                if config["enabled"]
            }

            if profile in {"full", "stock_reference"}:
                assert skill["enabled"] is True
                assert "browser_use" in enabled_tools
            else:
                assert skill["enabled"] is False
            if profile in {"local_tools", "core"}:
                assert enabled_tools == LOCAL_TOOL_NAMES
            if profile == "core":
                assert list(root["agents"]["profiles"]) == ["default"]
                assert agent["channels"] is None
                assert agent["mcp"] is None
                assert agent["heartbeat"]["enabled"] is False
                assert agent["running"]["memory_manager_backend"] == "none"
                assert (
                    agent["running"]["light_context_config"]["strategy"]
                    == "native"
                )
                assert agent["running"]["auto_title_config"]["enabled"] is False
                assert agent["plan"]["enabled"] is False
                assert agent["coding_mode"]["enabled"] is False
                assert _read(default_ws / "jobs.json")["jobs"] == []
                card = yaml.safe_load(
                    (default_ws / "drivers/mcp/demo.yaml").read_text(
                        encoding="utf-8",
                    ),
                )
                assert card["enabled"] is False
            else:
                assert len(root["agents"]["profiles"]) == 2


def test_stock_reference_allows_transport_only_override(tmp_path: Path) -> None:
    working, secret, base = _profile_seed(tmp_path)
    original_provider = _read(
        secret / "providers/builtin/dashscope.json",
    )
    with create_isolated_run(working, secret, "stock", base) as paths:
        result = generate_profile(
            paths,
            "stock_reference",
            base_url="http://127.0.0.1:9333/v1",
        )
        provider = _read(
            paths.secret_dir / "providers/builtin/dashscope.json",
        )
        assert provider["base_url"] == "http://127.0.0.1:9333/v1"
        assert provider["generate_kwargs"] == original_provider["generate_kwargs"]
        assert provider["models"] == original_provider["models"]
        assert result.metadata["transport_instrumented"] is True
        assert result.metadata["generation_parameters"] is None
    with create_isolated_run(working, secret, "stock-model", base) as paths:
        with pytest.raises(ProfileError, match="model override"):
            generate_profile(
                paths,
                "stock_reference",
                model="dashscope/qwen3.7-plus",
            )


def test_core_alias_minimal_is_supported(tmp_path: Path) -> None:
    working, secret, base = _profile_seed(tmp_path)
    with create_isolated_run(working, secret, "alias", base) as paths:
        result = generate_profile(paths, "minimal")
        assert result.name == "core"
        assert result.metadata["requested_profile"] == "minimal"
