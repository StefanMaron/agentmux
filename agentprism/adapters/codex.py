"""Codex CLI adapter for agentprism.

Wraps the OpenAI ``codex`` CLI behind the :class:`AgentAdapter` interface.

Programmatic interface notes
----------------------------
The Codex CLI (`codex`, installed at ``$(which codex)``) exposes several
non-interactive entry points. For agentprism we use ``codex exec``, which is
the documented headless mode:

    codex exec [OPTIONS] [PROMPT]
    codex exec resume [SESSION_ID] [PROMPT]   # follow-up turn

Relevant flags we rely on:

* ``--json``                         -- emit one JSON event per line on stdout
                                        (``thread.started`` carries the
                                        ``thread_id``/session id; ``turn.completed``
                                        signals the end of a turn; ``turn.failed``
                                        and ``error`` carry failure info.)
* ``-m, --model``                    -- choose the model (e.g. ``o4-mini``,
                                        ``gpt-5-codex``, ``o3``).
* ``-C, --cd <DIR>``                 -- working directory.
* ``--skip-git-repo-check``          -- allow running outside a git repo.
* ``--full-auto`` /
  ``--dangerously-bypass-approvals-and-sandbox`` -- non-interactive sandbox modes.
* ``-o, --output-last-message FILE`` -- final assistant message goes here.
* ``codex exec resume <SESSION_ID> <PROMPT>`` -- continue a previous session.

There is no persistent bidirectional stdio "session" mode for ``codex exec``
(unlike Copilot's ACP). Each turn is a fresh subprocess. We get conversational
continuity by recording the ``thread_id`` from the first run and then using
``codex exec resume`` for each subsequent ``send()``.

(``codex mcp-server`` exposes Codex over MCP and ``codex app-server`` /
``codex exec-server`` are experimental long-lived protocols, but they are
not stable enough to depend on right now.)

Authentication: ``codex login`` is required, or ``OPENAI_API_KEY`` set in
the environment. We surface the error verbatim if a turn fails with 401.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentprism.adapters.base import AgentAdapter, ProviderStatus, detect_quota_error


class NotInstalledError(RuntimeError):
    """Raised when the underlying provider CLI is not available on PATH."""


_INSTALL_HINT = (
    "Codex CLI not found on PATH. Install it with one of:\n"
    "  npm install -g @openai/codex\n"
    "  brew install codex            # if available on your platform\n"
    "Then run `codex login` (or export OPENAI_API_KEY) before using this adapter.\n"
    "See https://github.com/openai/codex for details."
)


def _find_codex() -> str | None:
    return shutil.which("codex")


# ---------------------------------------------------------------------------
# Session bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _CodexSession:
    """In-memory state for one logical Codex conversation."""

    session_id: str
    cwd: str
    model: str | None
    mode: str | None
    thread_id: str | None = None             # Codex-side conversation id
    proc: asyncio.subprocess.Process | None = None
    reader_task: asyncio.Task[None] | None = None
    output_buf: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    state: str = "working"                   # working | idle | done | error
    last_error: str | None = None
    last_message_file: str | None = None     # path passed via -o
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CodexAdapter(AgentAdapter):
    """Adapter that drives the OpenAI Codex CLI via ``codex exec --json``."""

    provider = "codex"

    # One adapter instance owns one session (matches Copilot adapter contract),
    # but we keep a class-level registry so the AgentAdapter API can look up by
    # session_id even though the same instance is the lookup target.
    def __init__(self) -> None:
        self._session: _CodexSession | None = None

    # ----- public API -------------------------------------------------------

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        """Start a new Codex turn for ``task`` in ``cwd``.

        Returns immediately with a ``session_id``; the turn keeps running
        in the background. ``mode`` is mapped to a sandbox profile:

        * ``"safe"`` / ``None``  -> ``--full-auto`` (workspace-write sandbox,
                                     no approvals).
        * ``"read-only"``        -> ``-s read-only``.
        * ``"yolo"`` / ``"unsafe"`` -> ``--dangerously-bypass-approvals-and-sandbox``.
        """
        codex_bin = _find_codex()
        if not codex_bin:
            raise NotInstalledError(_INSTALL_HINT)

        session_id = uuid.uuid4().hex
        sess = _CodexSession(
            session_id=session_id,
            cwd=cwd,
            model=model,
            mode=mode,
        )
        self._session = sess

        cmd = self._build_exec_cmd(codex_bin, task, sess, resume_thread_id=None)
        await self._launch(cmd, sess)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        """Send a follow-up turn, blocking until it completes.

        Uses ``codex exec resume <thread_id>`` so the new turn shares the
        same conversation history as the original ``spawn()`` call. If the
        previous turn is still running we wait for it to finish first.
        """
        sess = self._require_session(session_id)
        codex_bin = _find_codex()
        if not codex_bin:
            raise NotInstalledError(_INSTALL_HINT)

        # Make sure the previous turn settled before starting a new one.
        if sess.state == "working":
            await sess.done_event.wait()

        # Reset per-turn state.
        sess.events.clear()
        sess.output_buf.clear()
        sess.last_error = None
        sess.state = "working"
        sess.done_event = asyncio.Event()

        cmd = self._build_exec_cmd(
            codex_bin,
            message,
            sess,
            resume_thread_id=sess.thread_id,
        )
        await self._launch(cmd, sess)
        await sess.done_event.wait()
        return self._collect_output(sess)

    async def status(self, session_id: str) -> str:
        sess = self._require_session(session_id)
        return sess.state

    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        sess = self._require_session(session_id)
        try:
            await asyncio.wait_for(sess.done_event.wait(), timeout=timeout)
        except TimeoutError:
            return self._collect_output(sess)
        return self._collect_output(sess)

    async def kill(self, session_id: str) -> None:
        sess = self._require_session(session_id)
        proc = sess.proc
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass
        if sess.reader_task and not sess.reader_task.done():
            sess.reader_task.cancel()
        sess.state = "done" if sess.state != "error" else "error"
        sess.done_event.set()

    @classmethod
    def models(cls) -> list[dict]:
        """Known Codex/OpenAI models with rough cost tiers.

        ``multiplier`` is a relative cost weight (1.0 = baseline ~ ``o4-mini``-class
        token pricing). These are approximate and intended only for relative
        comparison/budget heuristics inside agentprism.
        """
        return [
            {
                "id": "gpt-5-codex",
                "multiplier": 5.0,
                "note": "Default Codex flagship (frontier reasoning + tool use). "
                        "Best quality, highest cost.",
            },
            {
                "id": "o3",
                "multiplier": 4.0,
                "note": "Strong reasoning model; good for complex multi-step "
                        "refactors. Slower and pricier than o4-mini.",
            },
            {
                "id": "o4-mini",
                "multiplier": 1.0,
                "note": "Fast, cheap reasoning baseline. Good default for routine "
                        "edits and exploration.",
            },
            {
                "id": "gpt-4.1",
                "multiplier": 2.0,
                "note": "General-purpose chat/coding model; non-reasoning. Useful "
                        "when latency matters more than deep planning.",
            },
            {
                "id": "gpt-4.1-mini",
                "multiplier": 0.6,
                "note": "Cheapest hosted option; suitable for small or boilerplate "
                        "tasks.",
            },
        ]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        import pathlib
        binary = "codex"
        installed = cls._binary_installed(binary)
        if not installed:
            return ProviderStatus("codex", False, False, f"'{binary}' not found — install with: npm install -g @openai/codex")
        # Auth: API key env var or ~/.codex/ config
        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        has_config = (pathlib.Path.home() / ".codex").is_dir()
        authenticated = has_key or has_config
        note = "" if authenticated else "set OPENAI_API_KEY or run 'codex login'"
        return ProviderStatus("codex", True, authenticated, note)

    # ----- internals --------------------------------------------------------

    def _require_session(self, session_id: str) -> _CodexSession:
        sess = self._session
        if sess is None or sess.session_id != session_id:
            raise KeyError(f"unknown codex session: {session_id}")
        return sess

    def _build_exec_cmd(
        self,
        codex_bin: str,
        prompt: str,
        sess: _CodexSession,
        resume_thread_id: str | None,
    ) -> list[str]:
        # Per-turn file for the final assistant message.
        tmp = tempfile.NamedTemporaryFile(
            prefix="codex-last-",
            suffix=".txt",
            delete=False,
        )
        tmp.close()
        sess.last_message_file = tmp.name

        cmd: list[str] = [codex_bin, "exec"]
        if resume_thread_id:
            cmd += ["resume", resume_thread_id]

        cmd += ["--json", "--skip-git-repo-check", "-C", sess.cwd]
        cmd += ["-o", sess.last_message_file]

        # Sandbox / approval mode.
        mode = (sess.mode or "safe").lower()
        if mode in ("safe", "auto", "full-auto", "default", ""):
            cmd += ["--full-auto"]
        elif mode in ("read-only", "readonly", "ro"):
            cmd += ["-s", "read-only"]
        elif mode in ("yolo", "unsafe", "dangerous", "bypass"):
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            # Pass through verbatim if it looks like a sandbox value.
            cmd += ["-s", sess.mode]  # type: ignore[list-item]

        if sess.model:
            cmd += ["-m", sess.model]

        cmd.append(prompt)
        return cmd

    async def _launch(self, cmd: list[str], sess: _CodexSession) -> None:
        env = os.environ.copy()
        # Force non-TTY behavior; codex respects this for color/progress.
        env.setdefault("NO_COLOR", "1")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=sess.cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        sess.proc = proc
        sess.reader_task = asyncio.create_task(self._drain(proc, sess))

    async def _drain(
        self,
        proc: asyncio.subprocess.Process,
        sess: _CodexSession,
    ) -> None:
        """Consume JSONL events from stdout and stderr until the process exits."""
        assert proc.stdout is not None
        assert proc.stderr is not None

        stderr_lines: list[str] = []

        async def read_stdout() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    # Non-JSON line (warnings, ANSI noise) -- keep raw.
                    sess.output_buf.append(text)
                    continue
                sess.events.append(event)
                self._handle_event(event, sess)

        async def read_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                stderr_lines.append(line.decode("utf-8", errors="replace").rstrip("\n"))
                # Codex logs go to stderr; keep them for diagnostics but don't
                # surface them as model output unless we have nothing else.

        try:
            await asyncio.gather(read_stdout(), read_stderr())
            await proc.wait()
        finally:
            # If the CLI wrote a "last message" file, prefer it as the
            # canonical assistant output for this turn.
            if sess.last_message_file and os.path.exists(sess.last_message_file):
                try:
                    with open(sess.last_message_file, encoding="utf-8") as f:
                        last = f.read().strip()
                    if last:
                        sess.output_buf.append(last)
                except OSError:
                    pass
                try:
                    os.unlink(sess.last_message_file)
                except OSError:
                    pass

            if sess.state == "working":
                # Process exited without an explicit terminal event.
                if proc.returncode and proc.returncode != 0:
                    sess.state = "error"
                    if not sess.last_error:
                        sess.last_error = (
                            f"codex exited with status {proc.returncode}"
                        )
                else:
                    sess.state = "done"

            # Check for quota errors in stderr + last_error.
            if sess.state == "error":
                stderr_text = "\n".join(stderr_lines)
                combined = stderr_text + "\n" + (sess.last_error or "")
                quota_err = detect_quota_error(combined, "codex", sess.model)
                if quota_err:
                    sess.last_error = str(quota_err)

            sess.done_event.set()

    def _handle_event(self, event: dict[str, Any], sess: _CodexSession) -> None:
        etype = event.get("type", "")

        # ``thread.started`` -> capture the conversation id for resume.
        if etype == "thread.started":
            tid = event.get("thread_id") or event.get("session_id")
            if tid and not sess.thread_id:
                sess.thread_id = tid
            return

        if etype == "turn.completed":
            # Some Codex versions inline the final message here.
            msg = event.get("output_text") or event.get("message")
            if isinstance(msg, str) and msg.strip():
                sess.output_buf.append(msg)
            sess.state = "done"
            sess.done_event.set()
            return

        if etype == "turn.failed" or etype == "error":
            err = event.get("error") or {}
            if isinstance(err, dict):
                sess.last_error = err.get("message") or json.dumps(err)
            else:
                sess.last_error = event.get("message") or str(err)
            sess.state = "error"
            # Check for quota error in the error message.
            if sess.last_error:
                quota_err = detect_quota_error(sess.last_error, "codex", sess.model)
                if quota_err:
                    sess.last_error = str(quota_err)
            sess.done_event.set()
            return

        # Streaming assistant text deltas (best-effort across versions).
        if etype in ("assistant.delta", "message.delta", "agent_message_delta"):
            delta = event.get("delta") or event.get("text") or ""
            if isinstance(delta, str) and delta:
                sess.output_buf.append(delta)
            return

        if etype in ("assistant.message", "agent_message"):
            text = event.get("text") or event.get("message") or ""
            if isinstance(text, str) and text:
                sess.output_buf.append(text)
            return

    def _collect_output(self, sess: _CodexSession) -> str:
        if sess.state == "error" and sess.last_error:
            if sess.last_error.startswith("[quota_exceeded]"):
                return sess.last_error
            joined = "".join(sess.output_buf).strip()
            if joined:
                return f"{joined}\n\n[codex error] {sess.last_error}"
            return f"[codex error] {sess.last_error}"
        return "".join(sess.output_buf).strip()


__all__ = ["CodexAdapter", "NotInstalledError"]
