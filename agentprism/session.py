"""SessionRegistry — tracks active agent sessions across providers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentprism import session_store
from agentprism.adapters.aider_adapter import AiderAdapter
from agentprism.adapters.base import AgentAdapter
from agentprism.adapters.claude_code import ClaudeCodeAdapter
from agentprism.adapters.codex import CodexAdapter
from agentprism.adapters.copilot import CopilotAdapter
from agentprism.adapters.gemini import GeminiAdapter
from agentprism.adapters.ollama import OllamaAdapter
from agentprism.adapters.opencode import OpenCodeAdapter
from agentprism.adapters.orphan import OrphanAdapter

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
    recovered: bool = False  # True if rehydrated from disk after a restart

    def summary(self) -> dict:
        out = {
            "session_id": self.session_id,
            "provider": self.provider,
            "cwd": self.cwd,
            "model": self.model,
            "mode": self.mode,
            "created_at": self.created_at.isoformat(),
        }
        if self.recovered:
            out["recovered"] = True
        return out


class SessionRegistry:
    """In-memory map of ``session_id`` → :class:`Session`.

    Sessions are also persisted to ``~/.agentprism/sessions/{session_id}.json``
    so that subprocesses can be re-attached after an agentprism restart.
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

        # Persist the session so it can be recovered if agentprism restarts
        # while the worker is still running.
        try:
            child_pid = adapter.child_pid(session_id)
            if child_pid:
                # PGID == PID for processes started with start_new_session=True.
                pgid = child_pid
                session_store.write_session({
                    "session_id": session_id,
                    "instance_pid": os.getpid(),
                    "provider": provider,
                    "model": model,
                    "mode": mode,
                    "cwd": cwd,
                    "initial_task": task,
                    "git_base_sha": base_sha,
                    "child_pid": child_pid,
                    "child_pgid": pgid,
                    "created_at": session.created_at.isoformat(),
                    "status": "active",
                })
        except Exception as e:
            log.warning("could not persist session %s: %s", session_id, e)

        # Background watcher: when the adapter's initial turn completes,
        # fire the on_complete callback. We treat the initial spawn-turn's
        # completion as "session done" for notification purposes.
        if self._on_complete is not None:
            self._watchers[session_id] = asyncio.create_task(
                self._watch_completion(session)
            )
        return session

    async def _watch_completion(self, session: Session) -> None:
        """Await terminal state on a session and fire the on_complete callback."""
        try:
            try:
                output = await session.adapter.wait(session.session_id)
            except Exception as exc:
                output = f"[adapter error] {type(exc).__name__}: {exc}"
            # Mark the session done on disk. We keep the file around briefly
            # so a status query right after completion still finds it; the
            # next clean kill / shutdown removes it.
            try:
                session_store.update_session(session.session_id, status="done")
            except Exception:
                pass
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
            try:
                session_store.remove_session(session_id)
            except Exception:
                pass

    async def shutdown(self) -> None:
        """Cancel watchers and detach from sessions, leaving subprocesses running.

        This is the key policy change: we no longer kill spawned workers
        when agentprism stops. Each session's record stays on disk so the
        next agentprism boot can rehydrate it as a recovered orphan.
        """
        for task in list(self._watchers.values()):
            if not task.done():
                task.cancel()
        self._watchers.clear()
        # Intentionally do not call adapter.kill() — let workers run to
        # completion. The next instance recovers them via session_store.
        self._sessions.clear()

    def recover_orphans(self) -> int:
        """Rehydrate sessions whose worker is still alive but whose parent died.

        Stale records (worker also dead) are pruned. Returns the number of
        sessions added back to the in-memory map. Safe to call once at
        startup before stdio takes over.
        """
        records = session_store.discover_sessions()
        orphans, dead = session_store.classify_orphans(records, current_pid=os.getpid())

        for rec in dead:
            try:
                session_store.remove_session(rec["session_id"])
            except Exception:
                pass

        recovered = 0
        for rec in orphans:
            try:
                self._reattach_orphan(rec)
                recovered += 1
            except Exception as e:
                log.warning("failed to recover session %s: %s", rec.get("session_id"), e)
        if recovered:
            log.info("recovered %d orphaned session(s) from previous instance(s)", recovered)
        return recovered

    def _reattach_orphan(self, rec: dict) -> None:
        session_id = rec["session_id"]
        adapter = OrphanAdapter(
            original_provider=rec.get("provider", "unknown"),
            pid=int(rec["child_pid"]),
            pgid=int(rec.get("child_pgid", rec["child_pid"])),
        )
        adapter.bind_session_id(session_id)

        created_at_raw = rec.get("created_at")
        try:
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if created_at_raw
                else datetime.now(UTC)
            )
        except Exception:
            created_at = datetime.now(UTC)

        session = Session(
            session_id=session_id,
            provider=rec.get("provider", "orphan"),
            adapter=adapter,
            cwd=rec.get("cwd", ""),
            model=rec.get("model"),
            mode=rec.get("mode"),
            initial_task=rec.get("initial_task", ""),
            git_base_sha=rec.get("git_base_sha"),
            created_at=created_at,
            recovered=True,
        )
        self._sessions[session_id] = session

        # Re-claim ownership in the lockfile so a future agentprism doesn't
        # also try to recover this same session.
        try:
            session_store.update_session(session_id, instance_pid=os.getpid())
        except Exception:
            pass
