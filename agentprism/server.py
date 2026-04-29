"""agentprism MCP server entrypoint.

Wires the :class:`SessionRegistry` and :class:`ToolDispatcher` to the
``mcp`` SDK's stdio server. Run via the ``agentprism`` console script
(see ``pyproject.toml``) or ``python -m agentprism.server``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agentprism.notifications import MCPContextHolder, notify_session_complete
from agentprism.session import Session, SessionRegistry
from agentprism.tools import ToolDispatcher, tool_definitions

log = logging.getLogger("agentprism")


def _configure_logging() -> None:
    level_name = os.environ.get("AGENTPRISM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    # IMPORTANT: log to stderr so we never corrupt the stdio MCP channel.
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_server() -> tuple[Server, SessionRegistry, MCPContextHolder]:
    """Construct the MCP server and its dependencies.

    Also returns the :class:`MCPContextHolder` used to bridge the lowlevel
    SDK's per-request ``ServerSession`` reference into the long-running
    completion-watcher tasks owned by :class:`SessionRegistry`. The holder
    starts empty and is populated lazily on the first tool call (the
    earliest moment the SDK exposes the session via its ``ContextVar``).
    """
    holder = MCPContextHolder()
    server: Server = Server("agentprism")

    async def _on_session_complete(session: Session, output: str) -> None:
        # Best-effort wake-up nudge to the orchestrating client.
        await notify_session_complete(session, output, holder)

    registry = SessionRegistry(on_complete=_on_session_complete)
    dispatcher = ToolDispatcher(registry)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        # Capture the live ServerSession on every entry — the first call
        # populates the holder so background notifications can use it.
        try:
            holder.capture(server.request_context.session)
        except LookupError:  # pragma: no cover — request context missing
            pass
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in tool_definitions()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        # Capture the active session for outbound sampling/log notifications.
        try:
            holder.capture(server.request_context.session)
        except LookupError:  # pragma: no cover — request context missing
            pass
        try:
            result = await dispatcher.call(name, arguments or {})
        except Exception as e:
            log.exception("Tool %s failed", name)
            return [TextContent(type="text", text=f"ERROR: {type(e).__name__}: {e}")]
        return [TextContent(type="text", text=result)]

    return server, registry, holder


async def run() -> None:
    _configure_logging()
    server, registry, holder = build_server()
    log.info("agentprism starting (pid=%d)", os.getpid())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        log.info("agentprism shutting down — killing %d sessions", len(registry.list()))
        await registry.shutdown()
        holder.clear()


def main() -> None:
    """Console-script entrypoint."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
