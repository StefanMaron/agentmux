"""OpenCode CLI adapter — subprocess-based.

OpenCode is a terminal-based AI coding agent from the SST team
(https://opencode.ai). This adapter shells out to ``opencode run`` per turn
and uses ``--session <id>`` for follow-up messages, mirroring the structure
of the Copilot adapter.

Key flags used:
- ``opencode run "<msg>"`` — non-interactive run
- ``--format json`` — emit JSONL stream of events (text, tool, step_*)
- ``--session <id>`` — continue a specific session id (returned in the JSON
  events as ``sessionID``)
- ``--dangerously-skip-permissions`` — auto-approve permissions (the YOLO
  equivalent), required for unattended runs
- ``--model provider/model`` — provider-prefixed model selection
- ``--dir`` — working directory

OpenCode supports many providers including Anthropic, OpenAI, GitHub Copilot,
GitHub Models, OpenRouter, and Ollama (configured via
``~/.config/opencode/opencode.json`` or ``opencode providers login``).
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

OPENCODE_BINARY = os.environ.get("AGENTPRISM_OPENCODE_BIN", "opencode")

_INSTALL_HINT = (
    "OpenCode CLI not found. Install with one of:\n"
    "  npm install -g opencode-ai\n"
    "  curl -fsSL https://opencode.ai/install | bash\n"
    "  brew install sst/tap/opencode\n"
    "Then authenticate with: opencode providers login"
)


class NotInstalledError(RuntimeError):
    """Raised when the OpenCode binary is missing."""


# A representative selection of models commonly available via OpenCode's
# bundled "opencode" zero-config provider plus popular configured providers.
# Users can pass any "provider/model" string via --model.
OPENCODE_MODELS: list[dict[str, str]] = [
    {"id": "opencode/gpt-5-nano",          "multiplier": "0x", "note": "free, zero-config default"},
    {"id": "opencode/big-pickle",          "multiplier": "0x", "note": "free, zero-config"},
    {"id": "opencode/hy3-preview-free",    "multiplier": "0x", "note": "free preview"},
    {"id": "opencode/ling-2.6-flash-free", "multiplier": "0x", "note": "free flash"},
    {"id": "opencode/minimax-m2.5-free",   "multiplier": "0x", "note": "free"},
    {"id": "opencode/nemotron-3-super-free", "multiplier": "0x", "note": "free"},
    {"id": "anthropic/claude-sonnet-4-5",  "multiplier": "1x", "note": "requires anthropic auth"},
    {"id": "anthropic/claude-opus-4-7",    "multiplier": "7.5x", "note": "deep reasoning"},
    {"id": "openai/gpt-5",                 "multiplier": "1x", "note": "requires openai auth"},
    {"id": "openai/gpt-5-mini",            "multiplier": "0.33x", "note": ""},
    {"id": "github-copilot/claude-sonnet-4.5", "multiplier": "1x", "note": "via copilot subscription"},
    {"id": "ollama/qwen2.5-coder:14b-8k",  "multiplier": "0x", "note": "local Ollama — configure provider in ~/.config/opencode/config.json"},
    {"id": "ollama/qwen2.5:14b",           "multiplier": "0x", "note": "local Ollama reasoning"},
]


@dataclass
class _OpenCodeSession:
    session_id: str           # agentprism session id (uuid)
    cwd: str
    model: str | None
    opencode_session_id: str = ""   # captured from first run's JSON events
    output_file: str = ""           # unused, kept for compat
    proc: asyncio.subprocess.Process | None = None
    output: str = ""
    status: str = "working"         # working | idle | done | error
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    all_chunks: list[dict] = field(default_factory=list)


class OpenCodeAdapter(AgentAdapter):
    """OpenCode adapter using ``opencode run … --format json`` per turn."""

    provider = "opencode"

    def __init__(self) -> None:
        self._session: _OpenCodeSession | None = None
        self._drain_task: asyncio.Task | None = None

    @classmethod
    def models(cls) -> list[dict]:
        return [dict(m) for m in OPENCODE_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        installed = cls._binary_installed(OPENCODE_BINARY)
        if not installed:
            return ProviderStatus(
                "opencode", False, False,
                f"'{OPENCODE_BINARY}' not found in PATH — {_INSTALL_HINT.splitlines()[0]}",
            )
        # OpenCode stores credentials in ~/.local/share/opencode/auth.json.
        # The bundled "opencode" provider works without auth (free models),
        # so we treat the binary as authenticated by default but flag if no
        # explicit credentials exist.
        auth_file = pathlib.Path.home() / ".local" / "share" / "opencode" / "auth.json"
        has_creds = auth_file.is_file() and auth_file.stat().st_size > 2
        # Free zero-config provider works without creds, so authenticated=True.
        note = "" if has_creds else "no provider credentials — only free 'opencode/*' models available (run 'opencode providers login' for more)"
        return ProviderStatus("opencode", True, True, note)

    # ------------------------------------------------------------------ public

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        if not self._binary_installed(OPENCODE_BINARY):
            raise NotInstalledError(_INSTALL_HINT)

        session_id = str(uuid.uuid4())
        sess = _OpenCodeSession(
            session_id=session_id,
            cwd=cwd,
            model=model,
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
            if sess.output_file:
                pathlib.Path(sess.output_file).unlink(missing_ok=True)
        except Exception:
            pass
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
            "opencode_session_id": sess.opencode_session_id,
        }

    # ---------------------------------------------------------------- private

    def _require(self, session_id: str) -> _OpenCodeSession:
        if self._session is None or self._session.session_id != session_id:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._session

    def _build_argv(self, sess: _OpenCodeSession, prompt: str, is_first: bool) -> list[str]:
        argv = [OPENCODE_BINARY, "run"]
        if not is_first and sess.opencode_session_id:
            argv += ["--session", sess.opencode_session_id]
        argv += [
            "--format", "json",
            "--dangerously-skip-permissions",
            "--dir", sess.cwd,
        ]
        if sess.model:
            argv += ["--model", sess.model]
        # Positional message argument(s)
        argv.append(prompt)
        return argv

    async def _run_turn(self, sess: _OpenCodeSession, prompt: str, is_first: bool) -> None:
        argv = self._build_argv(sess, prompt, is_first)
        log.info("opencode spawn: %s (cwd=%s)", " ".join(argv[:6]) + " …", sess.cwd)

        sess.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sess.cwd,
        )
        sess.done_event.clear()
        sess.status = "working"
        sess.last_activity = time.time()
        self._drain_task = asyncio.create_task(
            self._drain(sess), name=f"opencode-drain-{sess.session_id[:8]}"
        )

    async def _drain(self, sess: _OpenCodeSession) -> None:
        """Read JSONL stdout, parse OpenCode events into chunks."""
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

                # Capture session id from first event we see
                if not sess.opencode_session_id:
                    sid = ev.get("sessionID")
                    if sid:
                        sess.opencode_session_id = sid

                ev_type = ev.get("type", "")
                part = ev.get("part") or {}
                part_type = part.get("type", "")

                if ev_type == "text" or part_type == "text":
                    content = part.get("text") or ev.get("text") or ""
                    if content:
                        text_parts.append(content)
                        sess.all_chunks.append({"kind": "text", "text": content})

                elif ev_type == "tool" or part_type == "tool":
                    tool_name = part.get("tool") or part.get("name") or "tool"
                    state = part.get("state") or {}
                    inp = state.get("input") or part.get("input") or {}
                    if isinstance(inp, dict):
                        cmd = (
                            inp.get("command")
                            or inp.get("query")
                            or inp.get("filePath")
                            or inp.get("path")
                            or _json.dumps(inp)[:80]
                        )
                    else:
                        cmd = str(inp)[:80]
                    status = state.get("status") or ""
                    if status in ("running", "pending", ""):
                        sess.all_chunks.append({"kind": "tool", "text": f"⚙ {tool_name}({cmd})\n"})
                    elif status == "completed":
                        out = state.get("output") or state.get("result") or ""
                        if out:
                            preview = str(out)[:300].replace("\n", " ")
                            sess.all_chunks.append({"kind": "tool", "text": f"  → {preview}\n"})

                elif ev_type == "reasoning" or part_type == "reasoning":
                    think = part.get("text") or part.get("content") or ""
                    if think:
                        sess.all_chunks.append({"kind": "think", "text": think[:200] + "\n"})

                elif ev_type == "step_start":
                    # marker — ignore
                    pass
                elif ev_type == "step_finish":
                    # marker — ignore
                    pass

        await asyncio.gather(read_stdout_jsonl(), read_stderr())
        await sess.proc.wait()

        sess.output = "\n".join(text_parts) if text_parts else ""

        if sess.proc.returncode != 0 and not sess.output.strip():
            err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            sess.status = "error"
            sess.output = err or f"opencode exited with code {sess.proc.returncode}"
        else:
            sess.status = "done"

        sess.done_event.set()
        log.info(
            "opencode turn done (rc=%s, output=%d chars, opencode_sid=%s)",
            sess.proc.returncode, len(sess.output), sess.opencode_session_id,
        )


__all__ = ["NotInstalledError", "OpenCodeAdapter"]
