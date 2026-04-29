"""Google Gemini CLI adapter — subprocess-based.

Uses ``gemini -p <task> -y --output-format json`` per turn.
Resume via ``gemini --resume latest``.

Auth: set GEMINI_API_KEY (free tier: 1,500 Gemini 2.5 Pro requests/day via
Google AI Studio at https://aistudio.google.com/).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field

from agentprism.adapters.base import AgentAdapter, ProviderStatus

log = logging.getLogger(__name__)

GEMINI_BINARY = os.environ.get("AGENTPRISM_GEMINI_BIN", "gemini")

GEMINI_MODELS: list[dict[str, str]] = [
    {"id": "gemini-2.5-pro",   "multiplier": "0x (free tier)", "note": "flagship, 1500 req/day free"},
    {"id": "gemini-2.5-flash", "multiplier": "0x (free tier)", "note": "fast, higher free quota"},
    {"id": "gemini-2.0-flash", "multiplier": "0x (free tier)", "note": "previous gen, very fast"},
]


@dataclass
class _GeminiSession:
    session_id: str
    cwd: str
    model: str | None
    proc: asyncio.subprocess.Process | None = None
    output: str = ""
    status: str = "working"
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    all_chunks: list[dict] = field(default_factory=list)
    gemini_session_id: str = ""   # captured from first JSON event


class GeminiAdapter(AgentAdapter):
    """Gemini CLI adapter using ``gemini -p … -y --output-format json``."""

    provider = "gemini"

    def __init__(self) -> None:
        self._session: _GeminiSession | None = None
        self._drain_task: asyncio.Task | None = None

    @classmethod
    def models(cls) -> list[dict]:
        return [dict(m) for m in GEMINI_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        installed = cls._binary_installed(GEMINI_BINARY)
        if not installed:
            return ProviderStatus("gemini", False, False,
                                  f"'{GEMINI_BINARY}' not found — install: npm install -g @google/gemini-cli")
        authenticated = bool(os.environ.get("GEMINI_API_KEY") or
                             (pathlib.Path.home() / ".gemini" / "settings.json").exists())
        note = "" if authenticated else (
            "set GEMINI_API_KEY (free key at https://aistudio.google.com/)"
        )
        return ProviderStatus("gemini", True, authenticated, note)

    # ------------------------------------------------------------------ public

    async def spawn(self, task: str, cwd: str, model: str | None = None,
                    mode: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        sess = _GeminiSession(session_id=session_id, cwd=cwd, model=model)
        self._session = sess
        await self._run_turn(sess, task, is_first=True)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        sess = self._require(session_id)
        await sess.done_event.wait()
        sess.done_event.clear()
        sess.status = "working"
        await self._run_turn(sess, message, is_first=False)
        await sess.done_event.wait()
        return sess.output

    async def status(self, session_id: str) -> str:
        return self._require(session_id).status

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

    def _require(self, session_id: str) -> _GeminiSession:
        if self._session is None or self._session.session_id != session_id:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._session

    def _build_argv(self, sess: _GeminiSession, prompt: str, is_first: bool) -> list[str]:
        argv = [GEMINI_BINARY]
        if not is_first and sess.gemini_session_id:
            argv += ["--resume", sess.gemini_session_id]
        argv += [
            "-p", prompt,
            "-y",                               # yolo: auto-approve all tool calls
            "--skip-trust",                     # trust cwd without interactive prompt
            "--output-format", "stream-json",   # JSONL stream: message/tool/result events
        ]
        if sess.model:
            argv += ["--model", sess.model]
        return argv

    async def _run_turn(self, sess: _GeminiSession, prompt: str, is_first: bool) -> None:
        argv = self._build_argv(sess, prompt, is_first)
        log.info("gemini spawn: %s (cwd=%s)", " ".join(argv[:5]) + " …", sess.cwd)
        sess.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,     # gemini needs a pipe (not DEVNULL) to detect non-TTY
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sess.cwd,
        )
        # Write empty line + close — equivalent to `echo "" | gemini`
        if sess.proc.stdin:
            sess.proc.stdin.write(b"\n")
            await sess.proc.stdin.drain()
            sess.proc.stdin.close()
        sess.done_event.clear()
        sess.status = "working"
        sess.last_activity = time.time()
        self._drain_task = asyncio.create_task(
            self._drain(sess), name=f"gemini-drain-{sess.session_id[:8]}"
        )

    async def _drain(self, sess: _GeminiSession) -> None:
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

        async def read_stdout_jsonl() -> None:
            assert sess.proc is not None
            while True:
                line = await sess.proc.stdout.readline()
                if not line:
                    break
                sess.last_activity = time.time()
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue

                # Gemini outputs pretty-printed JSON blocks separated by newlines
                # Try to collect full object if needed
                try:
                    ev = _json.loads(raw)
                except _json.JSONDecodeError:
                    sess.all_chunks.append({"kind": "text", "text": raw + "\n"})
                    continue

                ev_type = ev.get("type") or ev.get("role") or ""

                # Capture session ID for resume
                if "session_id" in ev and not sess.gemini_session_id:
                    sess.gemini_session_id = ev["session_id"]

                # stream-json event types
                if ev_type == "init":
                    model = ev.get("model", "")
                    sess.all_chunks.append({"kind": "think", "text": f"[gemini/{model}]\n"})

                elif ev_type == "message":
                    role = ev.get("role", "")
                    content = ev.get("content") or ""
                    if role == "assistant" and content:
                        text_parts.append(str(content))
                        sess.all_chunks.append({"kind": "text", "text": str(content)})

                elif ev_type == "tool_call":
                    name = ev.get("name") or ev.get("tool_name") or "tool"
                    args = ev.get("args") or ev.get("arguments") or {}
                    cmd = args.get("command") or args.get("query") or _json.dumps(args)[:80]
                    sess.all_chunks.append({"kind": "tool", "text": f"⚙ {name}({cmd})\n"})

                elif ev_type == "tool_result":
                    result = ev.get("result") or ev.get("output") or ev.get("content") or ""
                    if result:
                        preview = str(result)[:300].replace("\n", " ")
                        sess.all_chunks.append({"kind": "tool", "text": f"  → {preview}\n"})

                elif ev_type == "error":
                    msg = ev.get("message") or ev.get("error") or str(ev)
                    sess.all_chunks.append({"kind": "error", "text": f"error: {msg}\n"})
                    text_parts.append(f"ERROR: {msg}")

        await asyncio.gather(read_stdout_jsonl(), read_stderr())
        await sess.proc.wait()

        sess.output = "\n".join(text_parts) if text_parts else ""

        if sess.proc.returncode != 0 and not sess.output.strip():
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            sess.status = "error"
            sess.output = err or f"gemini exited with code {sess.proc.returncode}"
        else:
            sess.status = "done"

        sess.done_event.set()
        log.info("gemini turn done (rc=%s, output=%d chars)", sess.proc.returncode, len(sess.output))
