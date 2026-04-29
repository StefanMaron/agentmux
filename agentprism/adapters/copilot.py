"""GitHub Copilot CLI adapter — subprocess-based (no ACP).

Uses ``copilot -p <task> --yolo`` per turn, with ``--resume`` for follow-ups.
Much simpler than ACP and actually works.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field

from agentprism.adapters.base import AgentAdapter, ProviderStatus

log = logging.getLogger(__name__)

COPILOT_BINARY = os.environ.get("AGENTPRISM_COPILOT_BIN", "copilot")

COPILOT_MODELS: list[dict[str, str]] = [
    {"id": "auto",              "multiplier": "1x",    "note": "default (resolves to claude-sonnet-4.6)"},
    {"id": "claude-sonnet-4.6", "multiplier": "1x",    "note": "best default for most tasks"},
    {"id": "claude-sonnet-4.5", "multiplier": "1x",    "note": ""},
    {"id": "claude-haiku-4.5",  "multiplier": "0.33x", "note": "fastest/cheapest Claude"},
    {"id": "claude-opus-4.7",   "multiplier": "7.5x",  "note": "deep reasoning only"},
    {"id": "claude-sonnet-4",   "multiplier": "1x",    "note": ""},
    {"id": "gpt-5.5",           "multiplier": "7.5x",  "note": "GPT flagship"},
    {"id": "gpt-5.4",           "multiplier": "1x",    "note": ""},
    {"id": "gpt-5.3-codex",     "multiplier": "1x",    "note": "code-focused"},
    {"id": "gpt-5.2-codex",     "multiplier": "1x",    "note": "code-focused"},
    {"id": "gpt-5.2",           "multiplier": "1x",    "note": ""},
    {"id": "gpt-5.4-mini",      "multiplier": "0.33x", "note": ""},
    {"id": "gpt-5-mini",        "multiplier": "0x",    "note": "free"},
    {"id": "gpt-4.1",           "multiplier": "0x",    "note": "free"},
]


@dataclass
class _CopilotSession:
    session_id: str
    session_name: str        # passed to --name / --resume
    cwd: str
    model: str | None
    output_file: str = ""    # unused, kept for compat
    proc: asyncio.subprocess.Process | None = None
    output: str = ""
    status: str = "working"  # working | idle | done | error
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    all_chunks: list[dict] = field(default_factory=list)


class CopilotAdapter(AgentAdapter):
    """Copilot adapter using ``copilot -p … --yolo`` subprocess per turn."""

    provider = "copilot"

    def __init__(self) -> None:
        self._session: _CopilotSession | None = None
        self._drain_task: asyncio.Task | None = None

    @classmethod
    def models(cls) -> list[dict]:
        return [dict(m) for m in COPILOT_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        installed = cls._binary_installed(COPILOT_BINARY)
        if not installed:
            return ProviderStatus("copilot", False, False, f"'{COPILOT_BINARY}' not found in PATH")
        auth_dir = pathlib.Path.home() / ".copilot"
        authenticated = auth_dir.is_dir() and any(auth_dir.iterdir())
        note = "" if authenticated else "run 'copilot login' to authenticate"
        return ProviderStatus("copilot", True, authenticated, note)

    # ------------------------------------------------------------------ public

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        session_name = f"agentprism-{session_id[:8]}"
        sess = _CopilotSession(
            session_id=session_id,
            session_name=session_name,
            cwd=cwd,
            model=model,
        )
        self._session = sess

        await self._run_turn(sess, task, is_first=True)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        sess = self._require(session_id)
        await sess.done_event.wait()  # wait for any in-flight turn to finish
        sess.done_event.clear()
        sess.status = "working"
        await self._run_turn(sess, message, is_first=False)
        # Non-blocking — return immediately so Claude isn't stuck waiting.
        # Use agent_wait/agent_status to observe the result.
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
        try:
            pathlib.Path(sess.output_file).unlink(missing_ok=True)
        except Exception:
            pass
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
            "process_alive": sess.proc is not None and sess.proc.returncode is None,
            "uptime_seconds": round(time.time() - sess.spawn_time),
            "last_activity_seconds_ago": round(time.time() - sess.last_activity, 1),
            "status": sess.status,
        }

    # ---------------------------------------------------------------- private

    def _require(self, session_id: str) -> _CopilotSession:
        if self._session is None or self._session.session_id != session_id:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._session

    def _build_argv(self, sess: _CopilotSession, prompt: str, is_first: bool) -> list[str]:
        argv = [COPILOT_BINARY]
        if not is_first:
            argv += ["--resume", sess.session_name]
        else:
            argv += ["--name", sess.session_name]
        argv += [
            "-p", prompt,
            "--yolo",
            "--output-format", "json",  # JSONL stream with tool calls, results, messages
        ]
        if sess.model:
            argv += ["--model", sess.model]
        return argv

    async def _run_turn(self, sess: _CopilotSession, prompt: str, is_first: bool) -> None:
        argv = self._build_argv(sess, prompt, is_first)
        log.info("copilot spawn: %s (cwd=%s)", " ".join(argv[:6]) + " …", sess.cwd)

        sess.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,   # must not touch parent stdio (MCP channel)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sess.cwd,
        )
        sess.done_event.clear()
        sess.status = "working"
        sess.last_activity = time.time()
        self._drain_task = asyncio.create_task(
            self._drain(sess), name=f"copilot-drain-{sess.session_id[:8]}"
        )

    async def _drain(self, sess: _CopilotSession) -> None:
        """Read JSONL stdout, parse events into chunks, collect final output."""
        import json as _json
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
                try:
                    ev = _json.loads(raw)
                except _json.JSONDecodeError:
                    sess.all_chunks.append({"kind": "text", "text": raw + "\n"})
                    continue

                ev_type = ev.get("type", "")
                data = ev.get("data") or {}

                if ev_type == "tool.execution_start":
                    name = data.get("toolName", "tool")
                    args = data.get("arguments") or {}
                    cmd = args.get("command") or args.get("query") or _json.dumps(args)[:80]
                    sess.all_chunks.append({"kind": "tool", "text": f"⚙ {name}({cmd})\n"})

                elif ev_type == "tool.execution_complete":
                    result = data.get("result") or {}
                    out = result.get("content") or result.get("output") or ""
                    if out:
                        preview = str(out)[:300].replace("\n", " ")
                        sess.all_chunks.append({"kind": "tool", "text": f"  → {preview}\n"})

                elif ev_type == "assistant.message":
                    content = data.get("content") or ""
                    if content:
                        text_parts.append(content)
                        sess.all_chunks.append({"kind": "text", "text": content})
                    # Surface tool requests as upcoming tool calls
                    for tr in data.get("toolRequests") or []:
                        name = tr.get("name", "tool")
                        args = tr.get("arguments") or {}
                        cmd = args.get("command") or args.get("query") or _json.dumps(args)[:80]
                        sess.all_chunks.append({"kind": "think", "text": f"→ calling {name}({cmd})\n"})

                elif ev_type == "assistant.thinking":
                    think = data.get("thinking") or data.get("content") or ""
                    if think:
                        sess.all_chunks.append({"kind": "think", "text": think[:200] + "\n"})

        await asyncio.gather(read_stdout_jsonl(), read_stderr())
        await sess.proc.wait()

        sess.output = "\n".join(text_parts) if text_parts else ""

        if sess.proc.returncode != 0 and not sess.output.strip():
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            sess.status = "error"
            sess.output = err or f"copilot exited with code {sess.proc.returncode}"
        else:
            sess.status = "done"

        sess.done_event.set()
        log.info("copilot turn done (rc=%s, output=%d chars)", sess.proc.returncode, len(sess.output))
