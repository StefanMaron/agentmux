"""Per-session JSON persistence for the SessionRegistry.

Each spawned session gets a file at ``~/.agentprism/sessions/{session_id}.json``.
On startup, agentprism rehydrates these files: if the original instance
PID is dead but the child PID is still alive, the session is re-registered
as a recovered orphan so ``agent_status`` and ``agent_kill`` keep working.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from agentprism.lockfile import is_pid_alive, lockfile_dir

log = logging.getLogger("agentprism.session_store")


def sessions_dir() -> Path:
    d = lockfile_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_file(session_id: str) -> Path:
    safe = session_id.replace("/", "_").replace("\\", "_")
    return sessions_dir() / f"{safe}.json"


def write_session(payload: dict) -> Path:
    """Atomically write the session record. Returns the path."""
    sid = payload["session_id"]
    path = _session_file(sid)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def update_session(session_id: str, **changes) -> None:
    """Read-modify-write a session record. Best-effort."""
    path = _session_file(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception as e:
        log.warning("could not read session file %s: %s", path, e)
        return
    data.update(changes)
    try:
        write_session(data)
    except Exception as e:
        log.warning("could not update session file %s: %s", path, e)


def remove_session(session_id: str) -> None:
    path = _session_file(session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("could not remove session file %s: %s", path, e)


def discover_sessions() -> list[dict]:
    """Return every persisted session record. No filtering applied."""
    out: list[dict] = []
    d = sessions_dir()
    for entry in d.glob("*.json"):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except Exception:
            try:
                entry.unlink()
            except Exception:
                pass
            continue
        data["_path"] = str(entry)
        out.append(data)
    return out


def classify_orphans(records: list[dict], current_pid: int) -> tuple[list[dict], list[dict]]:
    """Split persisted records into (recoverable_orphans, dead).

    A record is a recoverable orphan when its ``instance_pid`` is dead (or
    is not us) and its ``child_pid`` is still alive. A record is dead when
    the child is gone — those files are stale and should be removed by the
    caller.
    """
    orphans: list[dict] = []
    dead: list[dict] = []
    for rec in records:
        instance_pid = int(rec.get("instance_pid", 0))
        child_pid = int(rec.get("child_pid", 0))
        if instance_pid == current_pid:
            continue
        if is_pid_alive(instance_pid):
            continue
        if child_pid > 0 and is_pid_alive(child_pid):
            orphans.append(rec)
        else:
            dead.append(rec)
    return orphans, dead
