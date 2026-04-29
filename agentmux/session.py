"""SessionRegistry — tracks active agent sessions across providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agentmux.adapters.base import AgentAdapter
from agentmux.adapters.claude_code import ClaudeCodeAdapter
from agentmux.adapters.codex import CodexAdapter
from agentmux.adapters.copilot import CopilotAdapter

# Provider name → adapter class.
PROVIDERS: dict[str, type[AgentAdapter]] = {
    "copilot": CopilotAdapter,
    "claude": ClaudeCodeAdapter,
    "codex": CodexAdapter,
}


@dataclass
class Session:
    """One live agent session managed by the registry."""

    session_id: str
    provider: str
    adapter: AgentAdapter
    cwd: str
    model: str | None
    mode: str | None
    initial_task: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "cwd": self.cwd,
            "model": self.model,
            "mode": self.mode,
            "created_at": self.created_at.isoformat(),
        }


class SessionRegistry:
    """In-memory map of ``session_id`` → :class:`Session`.

    The registry is the single source of truth for spawned agents during the
    MCP server's lifetime. It's intentionally process-local — restarting
    agentmux drops all sessions.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def adapter_class(provider: str) -> type[AgentAdapter]:
        try:
            return PROVIDERS[provider]
        except KeyError as e:
            raise ValueError(
                f"Unknown provider {provider!r}. Known: {sorted(PROVIDERS)}"
            ) from e

    async def spawn(
        self,
        provider: str,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> Session:
        cls = self.adapter_class(provider)
        adapter = cls()
        session_id = await adapter.spawn(task=task, cwd=cwd, model=model, mode=mode)

        session = Session(
            session_id=session_id,
            provider=provider,
            adapter=adapter,
            cwd=cwd,
            model=model,
            mode=mode,
            initial_task=task,
        )
        async with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as e:
            raise ValueError(f"Unknown session_id: {session_id}") from e

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    async def kill(self, session_id: str) -> None:
        session = self.get(session_id)
        try:
            await session.adapter.kill(session_id)
        finally:
            async with self._lock:
                self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        """Kill every active session — call on server shutdown."""
        for session in list(self._sessions.values()):
            try:
                await session.adapter.kill(session.session_id)
            except Exception:  # noqa: BLE001
                pass
        self._sessions.clear()
