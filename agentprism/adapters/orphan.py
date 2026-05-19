"""OrphanAdapter — represents a recovered session whose original parent died.

When agentprism restarts and finds a session record on disk whose
``instance_pid`` is dead but whose ``child_pid`` is still alive, it
re-registers the session with this adapter. The original stdout pipe is
gone, so we can't replay output or send follow-up messages — but we can
still report whether the worker is alive (via PID + git activity) and
kill it cleanly via its process group.
"""

from __future__ import annotations

import asyncio
import time

from agentprism.adapters.base import AgentAdapter, ProviderStatus
from agentprism.lockfile import is_pid_alive
from agentprism.proc_utils import IS_WINDOWS, kill_process_group


class OrphanAdapter(AgentAdapter):
    """Adapter that wraps an already-running PID/PGID."""

    provider = "orphan"

    def __init__(
        self,
        original_provider: str,
        pid: int,
        pgid: int | None = None,
        spawn_time: float | None = None,
    ) -> None:
        self.original_provider = original_provider
        self.pid = pid
        self.pgid = pgid if pgid is not None else pid
        self.spawn_time = spawn_time or time.time()
        self._session_id: str | None = None

    # The registry sets this once it knows the recovered session_id.
    def bind_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    @property
    def _all_chunks(self) -> list[dict]:
        """Single explanatory chunk so the dashboard shows context instead of an empty terminal.

        The original stdout pipe died with the previous agentprism instance,
        so we can't replay or stream the worker's actual output. We can
        still report aliveness via PID and progress via git delta — see
        ``agent_status``.
        """
        return [{
            "kind": "text",
            "text": (
                "[recovered session]\n"
                f"This session was rehydrated from disk after the previous agentprism "
                f"instance exited. The worker (pid {self.pid}) is still being tracked by "
                f"PID and process group, but its stdout pipe was lost — live output is not "
                f"available here.\n"
                f"To check progress: call agent_status (process_alive + new_commits + "
                f"working_tree_changes), or watch the cwd's git log directly. "
                f"To stop the worker: agent_kill (sends SIGTERM to the whole process group).\n"
            ),
        }]

    async def spawn(self, task, cwd, model=None, mode=None) -> str:  # pragma: no cover
        raise RuntimeError("OrphanAdapter cannot spawn — it wraps an existing process")

    async def resume(self, session_id: str, message: str) -> str:
        raise RuntimeError(
            "Cannot resume a recovered session — the stdin pipe was lost when the "
            "original agentprism instance exited. Kill it and spawn a fresh worker."
        )

    async def status(self, session_id: str) -> str:
        return "working" if is_pid_alive(self.pid) else "done"

    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while is_pid_alive(self.pid):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out after {timeout}s")
            await asyncio.sleep(1.0)
        return "[recovered session — original output not captured by this agentprism instance]"

    async def kill(self, session_id: str) -> None:
        if not is_pid_alive(self.pid):
            return
        if not IS_WINDOWS:
            await kill_process_group(self.pid)
        else:  # pragma: no cover — Windows fallback
            import os
            import signal as _sig
            try:
                os.kill(self.pid, _sig.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def child_pid(self, session_id: str) -> int | None:
        return self.pid

    def activity_info(self) -> dict:
        return {
            "process_alive": is_pid_alive(self.pid),
            "uptime_seconds": round(time.time() - self.spawn_time),
            "last_activity_seconds_ago": None,
            "status": "working" if is_pid_alive(self.pid) else "done",
            "recovered": True,
            "original_provider": self.original_provider,
            "child_pid": self.pid,
        }

    @classmethod
    def models(cls) -> list[dict]:  # pragma: no cover
        return []

    @classmethod
    def check_available(cls) -> ProviderStatus:  # pragma: no cover
        return ProviderStatus(
            provider="orphan",
            installed=True,
            authenticated=True,
            note="internal — wraps recovered subprocesses",
        )
