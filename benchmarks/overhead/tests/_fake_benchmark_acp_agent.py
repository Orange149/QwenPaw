"""Small deterministic ACP agent used by overhead-client contract tests."""

from __future__ import annotations

import asyncio

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PROTOCOL_VERSION,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    Implementation,
    PermissionOption,
    ToolCallUpdate,
)


SHELL_COMMAND = "printf QWENPAW_BENCH_42"


class BenchmarkAgent(Agent):
    def __init__(self) -> None:
        self._connection = None

    def on_connect(self, connection) -> None:  # noqa: ANN001
        self._connection = connection

    async def initialize(
        self,
        protocol_version,
        client_capabilities=None,
        client_info=None,
        **_kwargs,
    ) -> InitializeResponse:  # noqa: ANN001
        del protocol_version, client_capabilities, client_info
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(name="overhead-fixture", version="1"),
        )

    async def new_session(
        self,
        cwd,
        additional_directories=None,
        mcp_servers=None,
        **_kwargs,
    ) -> NewSessionResponse:  # noqa: ANN001
        del cwd, additional_directories, mcp_servers
        session_id = "overhead-session"
        await self._connection.session_update(
            session_id=session_id,
            update=AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[
                    AvailableCommand(
                        name="benchmark",
                        description="Run the deterministic benchmark",
                    ),
                ],
            ),
        )
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        prompt,
        session_id,
        message_id=None,
        **_kwargs,
    ) -> PromptResponse:  # noqa: ANN001
        del prompt, message_id
        await self._connection.session_update(
            session_id=session_id,
            update=update_agent_thought(text_block("fixture reasoning")),
        )

        tool = ToolCallUpdate(
            tool_call_id="fixture-tool-1",
            title="Bash requires approval (INFO)",
            kind="other",
            raw_input={"command": SHELL_COMMAND},
        )
        options = [
            PermissionOption(
                option_id="allow_once",
                name="Allow once",
                kind="allow_once",
            ),
            PermissionOption(
                option_id="deny",
                name="Deny",
                kind="reject_once",
            ),
        ]
        await self._connection.session_update(
            session_id=session_id,
            update=start_tool_call(
                "fixture-tool-1",
                "execute_shell_command",
                kind="execute",
                status="in_progress",
                raw_input={"command": SHELL_COMMAND},
            ),
        )
        await self._connection.request_permission(
            options=options,
            session_id=session_id,
            tool_call=tool,
        )
        # An identical duplicate proves that the benchmark client grants only
        # one permission per turn.
        await self._connection.request_permission(
            options=options,
            session_id=session_id,
            tool_call=tool,
        )
        await self._connection.session_update(
            session_id=session_id,
            update=update_tool_call(
                "fixture-tool-1",
                status="completed",
                raw_output="QWENPAW_BENCH_42",
            ),
        )
        await self._connection.session_update(
            session_id=session_id,
            update=AgentMessageChunk(
                session_update="agent_message_chunk",
                content=text_block(""),
                field_meta={
                    "usage": {
                        "inputTokens": 101,
                        "outputTokens": 7,
                        "totalTokens": 108,
                    },
                },
            ),
        )
        await self._connection.session_update(
            session_id=session_id,
            update=update_agent_message(text_block("QWENPAW_BENCH_42")),
        )
        return PromptResponse(stop_reason="end_turn")

    async def close_session(self, session_id, **_kwargs):  # noqa: ANN001
        del session_id
        return None

    async def cancel(self, session_id, **_kwargs) -> None:  # noqa: ANN001
        del session_id


if __name__ == "__main__":
    asyncio.run(run_agent(BenchmarkAgent()))
