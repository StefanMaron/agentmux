"""Server → client push notifications when subagent sessions complete.

When a session spawned via ``agent_spawn`` finishes (the wrapped adapter
flips to a terminal ``done``/``error`` state), agentmux proactively pokes
the orchestrating MCP client so the LLM can wake up and process the
result without polling.

Two delivery channels, in preference order:

1. **``sampling/createMessage``** — the MCP server-to-client request that
   asks the client's LLM to run an inference. This is the ideal nudge:
   Claude Code (and other compliant clients) will surface the message
   into the active conversation. Requires the client to have advertised
   the ``sampling`` capability in its ``initialize`` handshake.

2. **``notifications/message``** — a plain log notification (level
   ``info``). Always available. Used as a fallback when the client did
   not advertise ``sampling``. Hooks / extensions on the client side can
   pick up the structured JSON payload to react to completions.

The ``MCPContextHolder`` is the bridge between the long-lived MCP
session (owned by ``server.run``) and the background asyncio tasks that
detect adapter completion. The lowlevel SDK only exposes the session via
a ``ContextVar`` set during request handling, so we capture it on the
first tool call and stash it on the holder for later out-of-band use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp import types

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.session import ServerSession

    from agentmux.session import Session

log = logging.getLogger("agentmux.notifications")


# ---------------------------------------------------------------------------
# Context holder
# ---------------------------------------------------------------------------


@dataclass
class MCPContextHolder:
    """Mutable handle on the active MCP server session.

    Populated lazily — the lowlevel SDK creates the ``ServerSession`` inside
    ``server.run()`` and only exposes it through a ``ContextVar`` while a
    request is being handled. We snapshot it on the first tool call and
    reuse the reference for asynchronous notifications fired later.
    """

    session: "ServerSession | None" = None

    def capture(self, session: "ServerSession") -> None:
        """Record the active session if we haven't already."""
        if self.session is None:
            self.session = session

    def clear(self) -> None:
        self.session = None

    def client_supports_sampling(self) -> bool:
        """True if the connected client advertised ``sampling`` capability."""
        sess = self.session
        if sess is None:
            return False
        params = getattr(sess, "_client_params", None) or getattr(
            sess, "client_params", None
        )
        if params is None:
            return False
        caps = getattr(params, "capabilities", None)
        return getattr(caps, "sampling", None) is not None


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


def _build_message(session: "Session", output: str) -> str:
    """Compose the human-readable wake-up text sent to the client."""
    return (
        "Subagent worker finished.\n\n"
        f"Session: {session.session_id}\n"
        f"Provider: {session.provider}\n"
        f'Task: "{session.initial_task}"\n'
        f"CWD: {session.cwd}\n\n"
        "Output:\n"
        f"{output}\n\n"
        "The worker has been decommissioned. You can now process these results, "
        "spawn additional workers, or continue with the next step."
    )


async def notify_session_complete(
    session: "Session",
    output: str,
    holder: MCPContextHolder,
) -> None:
    """Push a wake-up notification to the MCP client.

    Tries ``sampling/createMessage`` first; falls back to
    ``notifications/message`` (log INFO) if the client does not advertise
    sampling. Errors are swallowed and logged — completion notifications
    are best-effort and must never break the spawn pipeline.
    """
    server_session = holder.session
    if server_session is None:
        log.debug(
            "no MCP session captured yet — skipping notify for %s",
            session.session_id,
        )
        return

    text = _build_message(session, output)

    if holder.client_supports_sampling():
        try:
            await server_session.create_message(
                messages=[
                    types.SamplingMessage(
                        role="user",
                        content=types.TextContent(type="text", text=text),
                    )
                ],
                max_tokens=1024,
            )
            log.info(
                "sampling/createMessage delivered for session %s", session.session_id
            )
            return
        except Exception as exc:  # noqa: BLE001
            # The client may reject sampling (rate limits, user denial, etc.).
            # Fall through to the log-notification fallback so something is
            # still visible.
            log.warning(
                "sampling/createMessage failed for %s: %r — falling back to log",
                session.session_id,
                exc,
            )

    # Fallback: structured log notification. Hooks / clients without
    # sampling can still parse the JSON payload to react.
    payload: dict[str, Any] = {
        "event": "session_complete",
        "session_id": session.session_id,
        "provider": session.provider,
        "cwd": session.cwd,
        "initial_task": session.initial_task,
        "output": output,
        "message": text,
    }
    try:
        await server_session.send_log_message(
            level="info",
            data=payload,
            logger="agentmux.session_complete",
        )
        log.info(
            "notifications/message delivered for session %s", session.session_id
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "notifications/message failed for %s: %r (payload=%s)",
            session.session_id,
            exc,
            json.dumps(payload)[:200],
        )
