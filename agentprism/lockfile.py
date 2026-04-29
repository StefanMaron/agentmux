"""Cross-platform lockfile registry for running agentprism instances.

Each agentprism process writes a JSON lockfile to ``~/.agentprism/{pid}.json``
on startup describing its HTTP API port and working directory. The standalone
dashboard discovers live instances by listing this directory and pruning
stale lockfiles whose PIDs no longer exist.

All operations are stdlib-only and work on Linux, macOS, and Windows.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("agentprism.lockfile")


def lockfile_dir() -> Path:
    """Return the directory used to register running instances.

    Uses ``~/.agentprism/`` on every platform. Directory is created if missing.
    """
    d = Path.home() / ".agentprism"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _own_lockfile() -> Path:
    return lockfile_dir() / f"{os.getpid()}.json"


def write_lock(port: int, cwd: str) -> Path:
    """Write a lockfile for the current process. Returns the path."""
    path = _own_lockfile()
    payload = {"pid": os.getpid(), "port": int(port), "cwd": str(cwd)}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    # Atomic replace — works on POSIX and Windows (Python 3.3+).
    os.replace(tmp, path)
    log.debug("wrote lockfile %s -> %s", path, payload)
    return path


def remove_lock() -> None:
    """Remove the current process's lockfile, if present. Best-effort."""
    path = _own_lockfile()
    try:
        path.unlink()
        log.debug("removed lockfile %s", path)
    except FileNotFoundError:
        pass
    except Exception as e:  # pragma: no cover — best effort
        log.warning("failed to remove lockfile %s: %s", path, e)


def is_pid_alive(pid: int) -> bool:
    """Return True if ``pid`` refers to a live process.

    Cross-platform: ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the
    PID is gone, ``PermissionError`` if it exists but we lack rights to signal
    it (still alive!), and on Windows the same call (Python 3) returns success
    when the process exists or raises ``OSError`` otherwise.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists; we just can't signal it.
        return True
    except OSError:
        # On Windows, OSError with winerror==87 means the PID is invalid.
        return False
    return True


def discover() -> list[dict]:
    """Return one dict per live agentprism instance.

    Each dict has keys ``pid`` (int), ``port`` (int), ``cwd`` (str), and
    ``lockfile`` (str). Stale lockfiles whose PIDs are dead are removed as a
    side effect.
    """
    out: list[dict] = []
    d = lockfile_dir()
    for entry in d.glob("*.json"):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
            port = int(data.get("port", 0))
            cwd = str(data.get("cwd", ""))
        except Exception:
            # Corrupt lockfile — drop it.
            try:
                entry.unlink()
            except Exception:
                pass
            continue
        if not is_pid_alive(pid):
            try:
                entry.unlink()
            except Exception:
                pass
            continue
        out.append({"pid": pid, "port": port, "cwd": cwd, "lockfile": str(entry)})
    return out
