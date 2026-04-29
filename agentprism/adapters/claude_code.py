"""Claude Code adapter — wraps the ``claude`` CLI as an agentprism subagent.

Programmatic interface used
---------------------------
Claude Code does **not** speak ACP (the Agent Client Protocol used by the
GitHub Copilot CLI). It instead exposes a streaming-JSON I/O mode on the
``claude`` binary itself::

    claude -p \\
        --session-id <uuid> \\
        --input-format stream-json \\
        --output-format stream-json \\
        --verbose \\
        --model <model> \\
        --permission-mode <mode>

In this mode the CLI behaves like a long-lived JSON-RPC peer:

* **stdin** receives one JSON object per line. To submit a turn, write::

      {"type":"user",
       "message":{"role":"user",
                  "content":[{"type":"text","text":"<prompt>"}]}}

* **stdout** emits one JSON object per line. Important event types:

  - ``system`` (subtype ``init``): handshake, includes ``session_id``,
    available tools, model, etc. Sent once at startup.
  - ``assistant``: a model message (text content blocks, tool_use blocks,
    or thinking blocks). Multiple per turn.
  - ``user``: an injected user message echoing tool_results from the
    harness back into the conversation.
  - ``result``: emitted **once per user turn** when the agent stops. Has
    ``"subtype": "success"`` or ``"is_error": true``. Contains a flat
    ``result`` string with the final assistant text. **This is our
    "turn complete" marker.**
  - ``rate_limit_event``, ``hook_started``/``hook_response`` — ignored.

The process stays alive after a ``result`` event and accepts more user
messages on stdin. Closing stdin causes it to exit cleanly.

Why not ``--print`` one-shot mode?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A pure ``claude -p "prompt"`` invocation runs a single turn and exits.
You can chain turns with ``--resume <session-id>`` but each turn pays
the full process startup cost (~1s) and re-loads CLAUDE.md / hooks /
plugins. Stream-json keeps one process warm for the whole session and
gives us an event stream we can poll for status — the same shape the
Copilot ACP adapter uses.

Limitations vs the Copilot ACP adapter
--------------------------------------
* **No mid-turn cancellation message.** ACP has ``session/cancel``;
  Claude Code stream-json has no equivalent JSON message. ``kill()``
  here SIGTERMs the process, which aborts the in-flight turn but also
  ends the session — there is no graceful "stop just this turn".
* **No structured permission prompts over the wire** — we set
  ``--permission-mode bypassPermissions`` by default (callers can
  override via the ``mode`` arg) so the subagent runs autonomously.
* **No fine-grained progress events** beyond assistant/tool blocks. We
  expose the same ``working``/``idle``/``done``/``error`` status the
  base class requires.
* **One MCP-style tool catalogue per process.** Tools are fixed at
  spawn time via the CLI flags; we cannot register new tools mid-session.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentprism.adapters.base import AgentAdapter, ProviderStatus

# ---------------------------------------------------------------------------
# Internal session bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    """Per-session state held by :class:`ClaudeCodeAdapter`."""

    session_id: str
    proc: asyncio.subprocess.Process
    cwd: str
    model: str | None
    mode: str | None
    # Output of the most recently completed turn (the ``result`` event's
    # flat ``result`` string, or assistant text concatenated as a fallback).
    last_result: str = ""
    # Accumulated text for the *currently running* turn.
    pending_text: list[str] = field(default_factory=list)
    # Lifecycle: "working" while a turn is in flight, "idle" between turns,
    # "done" after the process exited cleanly, "error" otherwise.
    state: str = "working"
    # An asyncio.Event set whenever ``state`` transitions out of "working".
    turn_done: asyncio.Event = field(default_factory=asyncio.Event)
    # Stash the most recent error message, if any.
    error: str | None = None
    # The reader task draining stdout.
    reader_task: asyncio.Task | None = None
    # Lock serialising send() calls so we never interleave turns.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter(AgentAdapter):
    """Adapter that drives ``claude`` via its stream-json I/O mode.

    One adapter instance owns one ``claude`` subprocess and one logical
    conversation. Sessions are keyed by a UUID we generate in :meth:`spawn`
    and pass to the CLI via ``--session-id`` so the subprocess and the
    adapter agree on identity (the ``result`` events echo it back).
    """

    provider = "claude_code"

    #: Path / name of the binary. Overridable for tests.
    binary: str = "claude"

    #: Default model alias (matches ``claude --help`` accepted aliases).
    DEFAULT_MODEL = "sonnet"

    #: Default permission mode. ``bypassPermissions`` makes the subagent
    #: fully autonomous, which is what callers expect when they delegate
    #: a task to a wrapped agent. Override with the ``mode`` arg.
    DEFAULT_PERMISSION_MODE = "bypassPermissions"

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------------
    # Public API (AgentAdapter)
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        """Launch a fresh ``claude`` subprocess and submit the initial task.

        Returns the ``session_id`` immediately. The first turn continues
        running in the background; use :meth:`wait` to block on it.
        """
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"`{self.binary}` not found in PATH — install Claude Code first"
            )
        if not os.path.isdir(cwd):
            raise ValueError(f"cwd does not exist: {cwd}")

        session_id = str(uuid.uuid4())
        argv = [
            self.binary,
            "-p",
            "--session-id",
            session_id,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",  # required by stream-json output mode
            "--model",
            model or self.DEFAULT_MODEL,
            "--permission-mode",
            mode or self.DEFAULT_PERMISSION_MODE,
        ]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        sess = _Session(
            session_id=session_id,
            proc=proc,
            cwd=cwd,
            model=model,
            mode=mode,
        )
        self._sessions[session_id] = sess
        sess.reader_task = asyncio.create_task(self._read_stdout(sess))

        # Submit the initial task. Don't await its completion — the contract
        # is fire-and-return-session-id.
        await self._write_user_message(sess, task)
        return session_id

    async def send(self, session_id: str, message: str) -> str:
        """Send a follow-up message and wait for the agent's response."""
        sess = self._require(session_id)
        async with sess.send_lock:
            # Make sure no earlier turn is still running. (The registry
            # normally serialises this for us, but be defensive.)
            if sess.state == "working":
                await self._await_turn(sess, timeout=None)
            sess.turn_done.clear()
            sess.pending_text.clear()
            sess.state = "working"
            await self._write_user_message(sess, message)
            await self._await_turn(sess, timeout=None)
            if sess.state == "error":
                raise RuntimeError(sess.error or "claude subprocess errored")
            return sess.last_result

    async def status(self, session_id: str) -> str:
        sess = self._require(session_id)
        # Reap exit if the process is already gone but we haven't noticed.
        if sess.proc.returncode is not None and sess.state == "working":
            sess.state = "error" if sess.proc.returncode != 0 else "done"
            sess.turn_done.set()
        return sess.state

    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        """Block until the current turn finishes; return its output."""
        sess = self._require(session_id)
        await self._await_turn(sess, timeout=timeout)
        if sess.state == "error":
            raise RuntimeError(sess.error or "claude subprocess errored")
        return sess.last_result

    async def kill(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        # Try a graceful EOF first: closing stdin lets the CLI exit cleanly.
        try:
            if sess.proc.stdin is not None and not sess.proc.stdin.is_closing():
                sess.proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(sess.proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                sess.proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(sess.proc.wait(), timeout=2.0)
            except (TimeoutError, ProcessLookupError):
                try:
                    sess.proc.kill()
                except ProcessLookupError:
                    pass
        if sess.reader_task is not None:
            sess.reader_task.cancel()
            try:
                await sess.reader_task
            except (asyncio.CancelledError, Exception):
                pass

    @classmethod
    def models(cls) -> list[dict]:
        """Models we expose to MCP clients.

        ``multiplier`` is a rough cost-vs-haiku weight used by callers to
        budget runs; ``note`` documents intended use.
        """
        return [
            {
                "id": "haiku",
                "multiplier": 0.2,
                "note": "Fastest/cheapest. Good for trivial edits and lookups.",
            },
            {
                "id": "sonnet",
                "multiplier": 1.0,
                "note": "Default. Balanced reasoning and speed for most coding tasks.",
            },
            {
                "id": "opus",
                "multiplier": 5.0,
                "note": "Strongest reasoning. Use for complex refactors and design.",
            },
        ]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        import pathlib
        installed = cls._binary_installed(cls.binary)
        if not installed:
            return ProviderStatus("claude", False, False, f"'{cls.binary}' not found in PATH")
        # Auth: claude stores credentials in ~/.claude/
        auth_file = pathlib.Path.home() / ".claude" / ".credentials.json"
        authenticated = auth_file.exists()
        note = "" if authenticated else "run 'claude' and log in to authenticate"
        return ProviderStatus("claude", True, authenticated, note)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, session_id: str) -> _Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"unknown session_id: {session_id}")
        return sess

    async def _write_user_message(self, sess: _Session, text: str) -> None:
        if sess.proc.stdin is None or sess.proc.stdin.is_closing():
            raise RuntimeError("claude subprocess stdin is closed")
        payload = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }
        line = (json.dumps(payload) + "\n").encode("utf-8")
        sess.proc.stdin.write(line)
        await sess.proc.stdin.drain()

    async def _await_turn(self, sess: _Session, timeout: float | None) -> None:
        if sess.turn_done.is_set():
            return
        if timeout is None:
            await sess.turn_done.wait()
            return
        deadline = time.monotonic() + timeout
        try:
            await asyncio.wait_for(sess.turn_done.wait(), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"claude session {sess.session_id} did not finish in "
                f"{timeout:.1f}s (deadline={deadline})"
            ) from exc

    async def _read_stdout(self, sess: _Session) -> None:
        """Drain the subprocess's stdout, dispatching JSON events.

        Sets ``sess.turn_done`` whenever a ``result`` event arrives or the
        process dies. Survives malformed lines (logs and moves on).
        """
        assert sess.proc.stdout is not None
        try:
            while True:
                raw = await sess.proc.stdout.readline()
                if not raw:
                    # EOF — process exited.
                    rc = await sess.proc.wait()
                    if sess.state == "working":
                        sess.state = "error" if rc != 0 else "done"
                        if rc != 0 and sess.error is None:
                            stderr = b""
                            if sess.proc.stderr is not None:
                                try:
                                    stderr = await sess.proc.stderr.read()
                                except Exception:
                                    pass
                            sess.error = (
                                f"claude exited with code {rc}: "
                                f"{stderr.decode('utf-8', errors='replace')[:500]}"
                            )
                        sess.turn_done.set()
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Stray non-JSON output (rare); ignore.
                    continue
                self._handle_event(sess, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover — defensive
            sess.state = "error"
            sess.error = f"reader crashed: {exc!r}"
            sess.turn_done.set()

    def _handle_event(self, sess: _Session, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "assistant":
            # Accumulate text blocks for fallback in case ``result`` is sparse.
            content = event.get("message", {}).get("content") or []
            for block in content:
                if block.get("type") == "text":
                    txt = block.get("text") or ""
                    if txt:
                        sess.pending_text.append(txt)
        elif etype == "result":
            # End-of-turn marker.
            if event.get("is_error"):
                sess.state = "error"
                sess.error = event.get("result") or event.get("subtype") or "error"
                sess.last_result = sess.error or ""
            else:
                # ``result`` is the flat assistant text for this turn.
                sess.last_result = event.get("result") or "".join(sess.pending_text)
                sess.state = "idle"
            sess.turn_done.set()
        elif etype == "system" and event.get("subtype") == "error":
            sess.state = "error"
            sess.error = json.dumps(event)[:500]
            sess.turn_done.set()
        # Other event types (system/init, hook_*, rate_limit_event, user
        # tool-result echoes, etc.) are intentionally ignored — they do
        # not affect turn lifecycle for our purposes.
