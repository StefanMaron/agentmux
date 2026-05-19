"""Process-group utilities for adapter subprocess management.

Adapters spawn their CLI workers with ``start_new_session=True`` so each
worker becomes the leader of its own process group. That lets us kill the
worker *and* every grandchild it spawned (npm, git, language servers,
etc.) with a single ``killpg`` instead of leaving orphans behind when
agentprism dies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time

log = logging.getLogger("agentprism.proc_utils")

IS_WINDOWS = sys.platform == "win32"


def _pgid(pid: int) -> int | None:
    """Return the process group id for ``pid`` or None if it's already gone."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


async def kill_process_group(pid: int, timeout: float = 3.0) -> bool:
    """Kill ``pid`` and every member of its process group.

    Sends SIGTERM to the group, waits up to ``timeout`` for the leader to
    exit, then escalates to SIGKILL. Returns True if the group is gone by
    the time we return.

    On Windows there are no POSIX process groups; the caller should fall
    back to ``proc.terminate()`` / ``proc.kill()`` on the immediate child.
    """
    if IS_WINDOWS:
        return False

    pgid = _pgid(pid)
    if pgid is None:
        return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError as e:
        log.warning("killpg(%d, SIGTERM) denied: %s", pgid, e)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.05)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError as e:
        log.warning("killpg(%d, SIGKILL) denied: %s", pgid, e)
        return False

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
