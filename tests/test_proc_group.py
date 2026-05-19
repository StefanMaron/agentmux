"""Smoke test for kill_process_group — verifies grandchildren die too."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from agentprism.lockfile import is_pid_alive
from agentprism.proc_utils import IS_WINDOWS, kill_process_group


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX-only: process groups")
def test_kill_process_group_reaps_grandchildren():
    async def main() -> None:
        # bash -c 'sleep 90 & sleep 90 & wait' — leader plus two grandchildren
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            "sleep 90 & sleep 90 & wait",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Give bash a moment to fork the sleep grandchildren.
        await asyncio.sleep(0.3)
        pgid = os.getpgid(proc.pid)
        # Find sleep grandchildren in the same process group via /proc.
        sleep_pids = _find_pgid_members(pgid, exclude={proc.pid})
        assert sleep_pids, "expected at least one grandchild in the process group"

        ok = await kill_process_group(proc.pid, timeout=2.0)
        assert ok, "kill_process_group should report success"

        # Wait briefly for kernel cleanup.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not is_pid_alive(proc.pid) and all(
                not is_pid_alive(pid) for pid in sleep_pids
            ):
                break
            await asyncio.sleep(0.05)

        assert not is_pid_alive(proc.pid), "leader still alive"
        for pid in sleep_pids:
            assert not is_pid_alive(pid), f"grandchild {pid} still alive"

        # Reap so we don't leave a zombie.
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            pass

    asyncio.run(main())


def _find_pgid_members(pgid: int, exclude: set[int]) -> list[int]:
    """Return PIDs whose process group is ``pgid`` (Linux /proc only)."""
    if not sys.platform.startswith("linux"):
        return []
    out: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in exclude:
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
        except (FileNotFoundError, PermissionError):
            continue
        # stat format: pid (comm) state ppid pgrp ...
        # comm can contain spaces/parens, so split off the trailing fields.
        rparen = stat.rfind(")")
        if rparen < 0:
            continue
        rest = stat[rparen + 2 :].split()
        if len(rest) < 3:
            continue
        try:
            entry_pgrp = int(rest[2])
        except ValueError:
            continue
        if entry_pgrp == pgid:
            out.append(pid)
    return out
