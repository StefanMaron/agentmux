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

from agentprism.dashboard import start_dashboard
from agentprism.lockfile import remove_lock, write_lock
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


async def run(dashboard_port: int | None = None) -> None:
    """Run the MCP stdio server.

    Always starts an HTTP API server on a random free port and registers a
    lockfile in ``~/.agentprism/{pid}.json`` so the standalone dashboard can
    discover this instance. If ``dashboard_port`` is explicitly given, an
    additional HTTP server is bound to that port for backwards compatibility
    with the legacy per-instance dashboard.
    """
    _configure_logging()
    server, registry, holder = build_server()
    log.info("agentprism starting (pid=%d)", os.getpid())

    # Always start the auto-API on a random free port so the global
    # standalone dashboard can discover and aggregate this instance.
    api_port = await start_dashboard(0, registry)
    cwd = os.getcwd()
    try:
        write_lock(api_port, cwd)
        log.info("registered instance: port=%d cwd=%s", api_port, cwd)
    except Exception as e:  # pragma: no cover — best effort
        log.warning("failed to write lockfile: %s", e)

    # Backwards-compat: explicit per-instance dashboard on a fixed port.
    if dashboard_port is not None and dashboard_port != api_port:
        await start_dashboard(dashboard_port, registry)

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
        remove_lock()


async def _run_standalone_dashboard(port: int) -> None:
    """Run the standalone aggregator dashboard until interrupted."""
    _configure_logging()
    # Imported here so the MCP path doesn't pay the cost.
    from agentprism.standalone_dashboard import start_standalone_dashboard

    bound = await start_standalone_dashboard(port)
    log.info("standalone dashboard listening on http://127.0.0.1:%d", bound)
    # Block forever (until cancelled).
    stop = asyncio.Event()
    try:
        await stop.wait()
    except asyncio.CancelledError:
        pass


def main() -> None:
    """Console-script entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="agentprism", add_help=True)
    parser.add_argument(
        "--dashboard",
        metavar="PORT",
        type=int,
        default=None,
        help="(legacy) start an extra per-instance dashboard on this port",
    )
    sub = parser.add_subparsers(dest="command")

    dash = sub.add_parser(
        "dashboard",
        help="run the standalone global dashboard that aggregates all running agentprism instances",
    )
    dash.add_argument(
        "--port",
        type=int,
        default=7070,
        help="port to bind (default: 7070)",
    )

    args, _ = parser.parse_known_args()

    try:
        if args.command == "dashboard":
            asyncio.run(_run_standalone_dashboard(args.port))
        else:
            asyncio.run(run(dashboard_port=args.dashboard))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
