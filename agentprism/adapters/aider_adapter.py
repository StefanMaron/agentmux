"""Aider CLI adapter — local coding tasks via Aider + Ollama.

Aider is a terminal coding agent that wraps an LLM with a proven file-editing
framework (SEARCH/REPLACE blocks). We drive it headlessly with
``aider --message <task> --yes-always --no-pretty …`` per turn, against an
Ollama backend for free local coding.

Each :meth:`spawn` / :meth:`send` call runs a one-shot ``aider`` process in
the same ``cwd``. Aider preserves chat context between runs in the same
directory via its ``.aider.chat.history.md`` file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from agentprism.adapters.base import AgentAdapter, ProviderStatus

log = logging.getLogger(__name__)

AIDER_BINARY = os.environ.get("AGENTPRISM_AIDER_BIN", "aider")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = "qwen2.5-coder:14b"

AIDER_FALLBACK_MODELS: list[dict[str, str]] = [
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
    """Query Ollama's ``/api/tags`` for installed models."""
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
class _AiderSession:
    session_id: str
    cwd: str
    model: str | None
    proc: asyncio.subprocess.Process | None = None
    output: str = ""
    status: str = "working"  # working | idle | done | error
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    all_chunks: list[dict] = field(default_factory=list)


class AiderAdapter(AgentAdapter):
    """Aider adapter — one-shot ``aider --message`` subprocess per turn."""

    provider = "aider"

    def __init__(self) -> None:
        self._session: _AiderSession | None = None
        self._drain_task: asyncio.Task | None = None

    @classmethod
    def models(cls) -> list[dict]:
        live = _list_live_models()
        if live:
            return live
        return [dict(m) for m in AIDER_FALLBACK_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        installed = shutil.which(AIDER_BINARY) is not None
        if not installed:
            return ProviderStatus(
                "aider", False, False,
                f"'{AIDER_BINARY}' not found in PATH — install with 'uv tool install aider-chat'",
            )
        # Authenticated == Ollama server reachable (no API key needed).
        ollama_up = _http_get(f"{OLLAMA_HOST}/api/tags", timeout=2.0) is not None
        if not ollama_up:
            return ProviderStatus(
                "aider", True, False,
                f"ollama server unreachable at {OLLAMA_HOST} — start ollama first",
            )
        return ProviderStatus(
            "aider", True, True,
            f"aider + ollama ready at {OLLAMA_HOST}",
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
        sess = _AiderSession(
            session_id=session_id,
            cwd=cwd,
            model=model or DEFAULT_MODEL,
        )
        self._session = sess
        await self._run_turn(sess, task)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        sess = self._require(session_id)
        await sess.done_event.wait()
        sess.done_event.clear()
        sess.status = "working"
        await self._run_turn(sess, message)
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
        if sess.proc and sess.proc.returncode is None:
            try:
                sess.proc.terminate()
                await asyncio.wait_for(sess.proc.wait(), timeout=3.0)
            except Exception:
                try:
                    sess.proc.kill()
                except Exception:
                    pass
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
        sess.status = "done"
        sess.done_event.set()

    @property
    def _all_chunks(self) -> list[dict]:
        return self._session.all_chunks if self._session else []

    def activity_info(self) -> dict:
        sess = self._session
        if sess is None:
            return {}
        return {
            "process_alive": sess.proc is not None and sess.proc.returncode is None,
            "uptime_seconds": round(time.time() - sess.spawn_time),
            "last_activity_seconds_ago": round(time.time() - sess.last_activity, 1),
            "status": sess.status,
        }

    # ---------------------------------------------------------------- private

    def _require(self, session_id: str) -> _AiderSession:
        if self._session is None or self._session.session_id != session_id:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._session

    def _build_argv(self, sess: _AiderSession, prompt: str) -> list[str]:
        model = sess.model or DEFAULT_MODEL
        # Allow an explicit "ollama/foo" or LiteLLM-style override; otherwise
        # assume an Ollama tag.
        if "/" in model:
            model_arg = model
        else:
            model_arg = f"ollama/{model}"
        return [
            AIDER_BINARY,
            "--model", model_arg,
            "--message", prompt,
            "--yes-always",          # auto-confirm everything
            "--no-pretty",           # no ANSI for subprocess capture
            "--no-stream",           # easier line-buffered drain
            "--no-auto-commits",     # let agentprism decide when to commit
            "--no-suggest-shell-commands",
        ]

    async def _run_turn(self, sess: _AiderSession, prompt: str) -> None:
        argv = self._build_argv(sess, prompt)
        log.info("aider spawn: %s … (cwd=%s)", " ".join(argv[:5]), sess.cwd)

        env = os.environ.copy()
        env.setdefault("OLLAMA_API_BASE", OLLAMA_HOST)
        env.setdefault("OLLAMA_HOST", OLLAMA_HOST)

        sess.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sess.cwd,
            env=env,
        )
        sess.done_event.clear()
        sess.status = "working"
        sess.last_activity = time.time()
        self._drain_task = asyncio.create_task(
            self._drain(sess), name=f"aider-drain-{sess.session_id[:8]}"
        )

    async def _drain(self, sess: _AiderSession) -> None:
        """Drain aider's plain-text stdout/stderr; emit one chunk per line."""
        assert sess.proc is not None
        text_parts: list[str] = []
        stderr_chunks: list[bytes] = []

        async def read_stderr() -> None:
            assert sess.proc is not None
            while True:
                line = await sess.proc.stderr.readline()
                if not line:
                    break
                stderr_chunks.append(line)

        async def read_stdout() -> None:
            assert sess.proc is not None
            while True:
                line = await sess.proc.stdout.readline()
                if not line:
                    break
                sess.last_activity = time.time()
                raw = line.decode("utf-8", errors="replace")
                text_parts.append(raw)
                sess.all_chunks.append({"kind": "text", "text": raw})

        await asyncio.gather(read_stdout(), read_stderr())
        await sess.proc.wait()

        sess.output = "".join(text_parts)

        if sess.proc.returncode != 0 and not sess.output.strip():
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            sess.status = "error"
            sess.output = err or f"aider exited with code {sess.proc.returncode}"
        else:
            sess.status = "done"

        sess.done_event.set()
        log.info(
            "aider turn done (rc=%s, output=%d chars)",
            sess.proc.returncode, len(sess.output),
        )
