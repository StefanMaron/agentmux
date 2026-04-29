"""SessionRegistry — tracks active agent sessions across providers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentprism.adapters.aider_adapter import AiderAdapter
from agentprism.adapters.base import AgentAdapter
from agentprism.adapters.claude_code import ClaudeCodeAdapter
from agentprism.adapters.codex import CodexAdapter
from agentprism.adapters.copilot import CopilotAdapter
from agentprism.adapters.gemini import GeminiAdapter
from agentprism.adapters.ollama import OllamaAdapter
from agentprism.adapters.opencode import OpenCodeAdapter

log = logging.getLogger("agentprism.session")

#: Callback signature fired when an adapter session reaches a terminal state.
#: Receives the :class:`Session` and the adapter's accumulated output.
OnCompleteCallback = Callable[["Session", str], Awaitable[None]]

# Provider name → adapter class.
PROVIDERS: dict[str, type[AgentAdapter]] = {
    "copilot": CopilotAdapter,
    "claude": ClaudeCodeAdapter,
    "codex": CodexAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
    "opencode": OpenCodeAdapter,
    "aider": AiderAdapter,
}


def _git_head(cwd: str) -> str | None:
    """Return the current HEAD SHA if cwd is inside a git repo, else None."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_delta(cwd: str, base_sha: str | None) -> dict:
    """Return new commits and working-tree summary since base_sha."""
    import subprocess

    result: dict = {}
    if not base_sha:
        return result

    try:
        # New commits since spawn
        log = subprocess.run(
            ["git", "log", "--oneline", f"{base_sha}..HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        commits = [line.strip() for line in log.stdout.splitlines() if line.strip()]
        result["new_commits"] = commits
        result["new_commit_count"] = len(commits)

        # Working tree status (uncommitted changes)
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        changed = [line.strip() for line in status.stdout.splitlines() if line.strip()]
        result["working_tree_changes"] = changed
    except Exception:
        pass

    return result


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
    git_base_sha: str | None = None  # HEAD at spawn time for delta tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

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
    agentprism drops all sessions.
    """

    def __init__(
        self,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._on_complete = on_complete
        # Per-session watcher tasks. Kept so we can cancel them on
        # shutdown / kill without leaving stray coroutines pending.
        self._watchers: dict[str, asyncio.Task] = {}

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
        base_sha = await asyncio.get_event_loop().run_in_executor(None, _git_head, cwd)
        session_id = await adapter.spawn(task=task, cwd=cwd, model=model, mode=mode)

        session = Session(
            session_id=session_id,
            provider=provider,
            adapter=adapter,
            cwd=cwd,
            model=model,
            mode=mode,
            initial_task=task,
            git_base_sha=base_sha,
        )
        async with self._lock:
            self._sessions[session_id] = session

        # Background watcher: when the adapter's initial turn completes,
        # fire the on_complete callback. We treat the initial spawn-turn's
        # completion as "session done" for notification purposes — that's
        # the moment the orchestrator wants to be woken up.
        if self._on_complete is not None:
            self._watchers[session_id] = asyncio.create_task(
                self._watch_completion(session)
            )
        return session

    async def _watch_completion(self, session: Session) -> None:
        """Await terminal state on a session and fire the on_complete callback.

        Errors here are logged but never re-raised — completion notification
        is a best-effort side channel and must not poison the spawn flow.
        """
        try:
            try:
                output = await session.adapter.wait(session.session_id)
            except Exception as exc:
                # Surface the error in the callback payload rather than
                # silently dropping the notification — the orchestrator
                # likely still wants to know the worker is gone.
                output = f"[adapter error] {type(exc).__name__}: {exc}"
            if self._on_complete is not None:
                try:
                    await self._on_complete(session, output)
                except Exception:
                    log.exception(
                        "on_complete callback failed for session %s",
                        session.session_id,
                    )
        except asyncio.CancelledError:
            raise
        finally:
            self._watchers.pop(session.session_id, None)

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as e:
            raise ValueError(f"Unknown session_id: {session_id}") from e

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    async def kill(self, session_id: str) -> None:
        session = self.get(session_id)
        watcher = self._watchers.pop(session_id, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
        try:
            await session.adapter.kill(session_id)
        finally:
            async with self._lock:
                self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        """Kill every active session — call on server shutdown."""
        for task in list(self._watchers.values()):
            if not task.done():
                task.cancel()
        self._watchers.clear()
        for session in list(self._sessions.values()):
            try:
                await session.adapter.kill(session.session_id)
            except Exception:
                pass
        self._sessions.clear()
