"""Ollama adapter — direct REST API for local LLMs.

Targets brainstorming and reasoning, not file editing. Uses the streaming
``/api/chat`` endpoint over plain HTTP (stdlib only) and keeps the whole
conversation in-memory so each turn is just an append-and-send.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from agentprism.adapters.base import (
    AgentAdapter,
    ProviderStatus,
    QuotaExceededError,
    detect_quota_error,
)

log = logging.getLogger(__name__)

OLLAMA_BINARY = os.environ.get("AGENTPRISM_OLLAMA_BIN", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = "qwen2.5-coder:14b"

# Static fallback list — overridden by live /api/tags when the server is up.
OLLAMA_FALLBACK_MODELS: list[dict[str, str]] = [
    {"id": "qwen2.5-coder:14b", "multiplier": "0x", "note": "local default — code reasoning"},
    {"id": "qwen2.5-coder:7b",  "multiplier": "0x", "note": "smaller, faster"},
    {"id": "llama3.1:8b",       "multiplier": "0x", "note": "general-purpose"},
    {"id": "deepseek-r1:14b",   "multiplier": "0x", "note": "reasoning"},
]


def _http_get(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _list_live_models() -> list[dict] | None:
    """Query ``/api/tags`` for installed models. Returns None if unreachable."""
    raw = _http_get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    out: list[dict] = []
    for m in data.get("models") or []:
        name = m.get("name") or m.get("model")
        if not name:
            continue
        size = m.get("size")
        note = ""
        if isinstance(size, int) and size > 0:
            note = f"{size / 1e9:.1f} GB"
        out.append({"id": name, "multiplier": "0x", "note": note})
    return out


@dataclass
class _OllamaSession:
    session_id: str
    cwd: str
    model: str
    messages: list[dict] = field(default_factory=list)  # {"role", "content"}
    output: str = ""                                    # latest turn's text
    status: str = "working"                             # working | idle | done | error
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    all_chunks: list[dict] = field(default_factory=list)
    in_flight: bool = False


class OllamaAdapter(AgentAdapter):
    """Local Ollama adapter — streaming HTTP chat, no subprocess."""

    provider = "ollama"

    def __init__(self) -> None:
        self._session: _OllamaSession | None = None
        self._turn_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- classmethods

    @classmethod
    def models(cls) -> list[dict]:
        live = _list_live_models()
        if live:
            return live
        return [dict(m) for m in OLLAMA_FALLBACK_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        installed = cls._binary_installed(OLLAMA_BINARY)
        if not installed:
            return ProviderStatus(
                "ollama", False, False, f"'{OLLAMA_BINARY}' not found in PATH"
            )

        live = _list_live_models()
        if live is None:
            return ProviderStatus(
                "ollama", True, False,
                f"server unreachable at {OLLAMA_HOST} — start ollama server first",
            )
        if not live:
            return ProviderStatus(
                "ollama", True, True,
                f"server up at {OLLAMA_HOST} but no models installed yet",
            )
        names = ", ".join(m["id"] for m in live[:5])
        more = "" if len(live) <= 5 else f" (+{len(live) - 5} more)"
        return ProviderStatus(
            "ollama", True, True,
            f"server up at {OLLAMA_HOST}; models: {names}{more}",
        )

    # ------------------------------------------------------------------ public

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        sess = _OllamaSession(
            session_id=session_id,
            cwd=cwd,
            model=model or DEFAULT_MODEL,
        )
        self._session = sess
        sess.messages.append({"role": "user", "content": task})
        self._start_turn(sess)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        sess = self._require(session_id)
        # Wait for any in-flight turn to finish before queueing the next.
        await sess.done_event.wait()
        sess.done_event.clear()
        sess.status = "working"
        sess.messages.append({"role": "user", "content": message})
        self._start_turn(sess)
        return "message sent — use agent_wait or agent_status to observe"

    async def status(self, session_id: str) -> str:
        sess = self._require(session_id)
        return sess.status

    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        sess = self._require(session_id)
        try:
            await asyncio.wait_for(sess.done_event.wait(), timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(f"Timed out after {timeout}s") from e
        return sess.output

    async def kill(self, session_id: str) -> None:
        sess = self._require(session_id)
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        sess.in_flight = False
        sess.status = "done"
        sess.done_event.set()

    @property
    def _all_chunks(self) -> list[dict]:
        """Expose session chunks under the name the dashboard expects."""
        return self._session.all_chunks if self._session else []

    def activity_info(self) -> dict:
        sess = self._session
        if sess is None:
            return {}
        return {
            "process_alive": sess.in_flight,
            "uptime_seconds": round(time.time() - sess.spawn_time),
            "last_activity_seconds_ago": round(time.time() - sess.last_activity, 1),
            "status": sess.status,
        }

    # ---------------------------------------------------------------- private

    def _require(self, session_id: str) -> _OllamaSession:
        if self._session is None or self._session.session_id != session_id:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._session

    def _start_turn(self, sess: _OllamaSession) -> None:
        sess.done_event.clear()
        sess.status = "working"
        sess.in_flight = True
        sess.output = ""
        sess.last_activity = time.time()
        self._turn_task = asyncio.create_task(
            self._run_turn(sess), name=f"ollama-turn-{sess.session_id[:8]}"
        )

    async def _run_turn(self, sess: _OllamaSession) -> None:
        """Stream one assistant response from /api/chat into sess."""
        payload = {
            "model": sess.model,
            "messages": sess.messages,
            "stream": True,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{OLLAMA_HOST}/api/chat"
        log.info("ollama turn: model=%s msgs=%d", sess.model, len(sess.messages))

        loop = asyncio.get_running_loop()
        text_parts: list[str] = []
        error: str | None = None

        def _stream_blocking() -> None:
            nonlocal error
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for raw in resp:
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in ev:
                            error = str(ev["error"])
                            return

                        msg = ev.get("message") or {}
                        chunk = msg.get("content") or ""
                        if chunk:
                            text_parts.append(chunk)
                            # Schedule chunk append on the loop thread.
                            loop.call_soon_threadsafe(
                                _append_chunk, sess, chunk
                            )

                        if ev.get("done"):
                            return
            except urllib.error.URLError as e:
                error = f"ollama HTTP error: {e.reason}"
            except (TimeoutError, OSError) as e:
                error = f"ollama connection error: {e}"

        def _append_chunk(s: _OllamaSession, c: str) -> None:
            s.last_activity = time.time()
            s.all_chunks.append({"kind": "text", "text": c})

        try:
            await loop.run_in_executor(None, _stream_blocking)
        except asyncio.CancelledError:
            sess.in_flight = False
            sess.status = "done"
            sess.output = "".join(text_parts)
            sess.done_event.set()
            raise

        sess.in_flight = False
        sess.output = "".join(text_parts)

        if error and not sess.output.strip():
            error_lower = error.lower()
            if "connection" in error_lower or "refused" in error_lower or "unreachable" in error_lower:
                quota_err: QuotaExceededError | None = QuotaExceededError(
                    "ollama", sess.model, "Ollama server not running or no models available"
                )
            else:
                quota_err = detect_quota_error(error, "ollama", sess.model)
            if quota_err:
                sess.status = "error"
                sess.output = str(quota_err)
                sess.all_chunks.append({"kind": "text", "text": f"[error] {sess.output}\n"})
            else:
                sess.status = "error"
                sess.output = error
                sess.all_chunks.append({"kind": "text", "text": f"[error] {error}\n"})
        else:
            sess.status = "done"
            # Persist assistant turn into history for follow-ups.
            if sess.output:
                sess.messages.append({"role": "assistant", "content": sess.output})

        sess.done_event.set()
        log.info(
            "ollama turn done (status=%s, output=%d chars)",
            sess.status, len(sess.output),
        )
