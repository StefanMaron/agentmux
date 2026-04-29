"""GitHub Copilot CLI adapter — subprocess-based (no ACP).

Uses ``copilot -p <task> --yolo`` per turn, with ``--resume`` for follow-ups.
Much simpler than ACP and actually works.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import tempfile
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
    output_file: str         # --share target
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
        output_file = tempfile.mktemp(prefix="agentprism-", suffix=".md")

        sess = _CopilotSession(
            session_id=session_id,
            session_name=session_name,
            cwd=cwd,
            model=model,
            output_file=output_file,
        )
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
            "--share", sess.output_file,
            "-s",  # silent: agent response only, no stats
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
        """Read stdout/stderr, update chunks, mark done on exit."""
        assert sess.proc is not None
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        async def read_stream(stream: asyncio.StreamReader, buf: list[bytes], kind: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                buf.append(line)
                sess.last_activity = time.time()
                text = line.decode("utf-8", errors="replace")
                sess.all_chunks.append({"kind": kind, "text": text})

        await asyncio.gather(
            read_stream(sess.proc.stdout, stdout_chunks, "text"),
            read_stream(sess.proc.stderr, stderr_chunks, "tool"),
        )
        await sess.proc.wait()

        # Prefer the --share file (full markdown output) over stdout
        share = pathlib.Path(sess.output_file)
        if share.exists() and share.stat().st_size > 0:
            sess.output = share.read_text(encoding="utf-8", errors="replace")
        else:
            sess.output = b"".join(stdout_chunks).decode("utf-8", errors="replace")

        if sess.proc.returncode != 0 and not sess.output.strip():
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            sess.status = "error"
            sess.output = err or f"copilot exited with code {sess.proc.returncode}"
        else:
            sess.status = "done"

        sess.done_event.set()
        log.info("copilot turn done (rc=%s, output=%d chars)", sess.proc.returncode, len(sess.output))
