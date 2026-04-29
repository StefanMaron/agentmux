"""GitHub Copilot CLI adapter.

Speaks the Agent Client Protocol (ACP) — JSON-RPC 2.0 over stdio — with
a ``copilot --acp`` subprocess. Each adapter instance owns exactly one
subprocess and one ACP session.

Lifecycle
---------

1. ``spawn(task, cwd, ...)``
   * launches ``copilot --acp``
   * sends ``initialize``
   * sends ``session/new`` with ``cwd``
   * (optional) sends ``session/set_mode``
   * sends ``session/prompt`` with the initial task — does **not** wait
   * returns the ACP session id

2. A background reader coroutine demuxes stdout:
   * frames with an ``id`` resolve the matching ``Future`` in ``_pending``
   * notifications (no ``id``) named ``session/update`` append text chunks
     to ``_output_buffer`` and flip ``_done`` when ``stopReason`` arrives

3. ``send`` / ``wait`` / ``status`` / ``kill`` operate on this state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from agentprism.adapters.base import AgentAdapter, ProviderStatus

log = logging.getLogger(__name__)

COPILOT_BINARY = os.environ.get("AGENTPRISM_COPILOT_BIN", "copilot")

# Live-probed model catalogue. Multipliers are quoted strings to preserve
# the human-readable "0x" / "7.5x" form Copilot uses.
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

ACP_MODE_URI = "https://agentclientprotocol.com/protocol/session-modes"
KNOWN_MODES = {"agent", "plan", "autopilot"}


def _mode_uri(mode: str) -> str:
    """Expand a short mode name (``"plan"``) to the full ACP URI."""
    if mode.startswith("http"):
        return mode
    short = mode.lstrip("#")
    if short not in KNOWN_MODES:
        raise ValueError(f"Unknown Copilot mode '{mode}'. Expected one of {sorted(KNOWN_MODES)}.")
    return f"{ACP_MODE_URI}#{short}"


class CopilotAdapter(AgentAdapter):
    """Adapter for the ``copilot --acp`` JSON-RPC stdio agent."""

    provider = "copilot"

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None

        # JSON-RPC plumbing.
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()

        # Session state.
        self._acp_session_id: str | None = None
        self._output_buffer: list[str] = []
        self._done = asyncio.Event()
        self._done.set()  # idle by default; cleared when a prompt is in flight
        self._current_prompt_id: int | None = None
        self._last_stop_reason: str | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------ public

    @classmethod
    def models(cls) -> list[dict]:
        return [dict(m) for m in COPILOT_MODELS]

    @classmethod
    def check_available(cls) -> ProviderStatus:
        import pathlib
        installed = cls._binary_installed(COPILOT_BINARY)
        if not installed:
            return ProviderStatus("copilot", False, False, f"'{COPILOT_BINARY}' not found in PATH")
        # Auth: copilot stores session in ~/.copilot/
        auth_dir = pathlib.Path.home() / ".copilot"
        authenticated = auth_dir.is_dir() and any(auth_dir.iterdir())
        note = "" if authenticated else "run 'copilot login' to authenticate"
        return ProviderStatus("copilot", True, authenticated, note)

    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        if self._proc is not None:
            raise RuntimeError("CopilotAdapter already spawned; create a new instance per session.")

        if not os.path.isabs(cwd):
            raise ValueError(f"cwd must be an absolute path, got: {cwd!r}")
        if not os.path.isdir(cwd):
            raise ValueError(f"cwd does not exist or is not a directory: {cwd}")

        argv = [COPILOT_BINARY, "--acp"]
        if model:
            argv.extend(["--model", model])

        log.info("Spawning copilot: %s (cwd=%s)", " ".join(argv), cwd)
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        self._reader_task = asyncio.create_task(self._read_stdout(), name="copilot-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="copilot-stderr")

        # 1. initialize
        await self._request(
            "initialize",
            {
                "protocolVersion": 1,
                "capabilities": {},
                "clientInfo": {"name": "agentprism", "version": "0.1.0"},
            },
        )

        # 2. session/new
        new_resp = await self._request("session/new", {"cwd": cwd, "mcpServers": []})
        session_id = new_resp.get("sessionId")
        if not session_id:
            raise RuntimeError(f"copilot session/new returned no sessionId: {new_resp!r}")
        self._acp_session_id = session_id

        # 3. optional set_mode
        if mode:
            await self._request(
                "session/set_mode",
                {"sessionId": session_id, "mode": _mode_uri(mode)},
            )

        # 4. fire the initial prompt — do NOT await its completion
        self._start_prompt(task)

        return session_id

    async def send(self, session_id: str, message: str) -> str:
        self._check_session(session_id)
        # Wait for any in-flight turn to finish before starting a new one.
        await self._done.wait()
        prompt_id = self._start_prompt(message)
        await self._await_prompt(prompt_id)
        return self._drain_output()

    async def status(self, session_id: str) -> str:
        self._check_session(session_id)
        if self._error:
            return "error"
        if self._proc is None or self._proc.returncode is not None:
            return "done"
        if not self._done.is_set():
            return "working"
        return "idle"

    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        self._check_session(session_id)
        if self._current_prompt_id is None:
            return self._drain_output()
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for copilot session {session_id}"
            ) from e
        return self._drain_output()

    async def kill(self, session_id: str) -> None:
        if self._acp_session_id and session_id != self._acp_session_id:
            raise ValueError(f"Unknown session_id {session_id}")

        # Best-effort graceful close.
        if self._proc and self._proc.returncode is None and self._acp_session_id:
            try:
                await asyncio.wait_for(
                    self._request("session/close", {"sessionId": self._acp_session_id}),
                    timeout=2.0,
                )
            except (TimeoutError, Exception) as e:
                log.debug("session/close failed (ignored): %s", e)

        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=3.0)
                except TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass

        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Fail any still-pending futures.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("copilot subprocess terminated"))
        self._pending.clear()
        self._done.set()

    # ----------------------------------------------------------------- internals

    def _check_session(self, session_id: str) -> None:
        if self._acp_session_id is None:
            raise RuntimeError("CopilotAdapter has not been spawned yet")
        if session_id != self._acp_session_id:
            raise ValueError(
                f"session_id mismatch: adapter owns {self._acp_session_id!r}, got {session_id!r}"
            )

    def _start_prompt(self, text: str) -> int:
        """Send a ``session/prompt`` and register its future without awaiting."""
        assert self._acp_session_id is not None
        self._output_buffer.clear()
        self._last_stop_reason = None
        self._done.clear()

        prompt_id = self._next_id
        self._next_id += 1
        self._current_prompt_id = prompt_id

        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[prompt_id] = fut

        params = {
            "sessionId": self._acp_session_id,
            "prompt": [{"type": "text", "text": text}],
        }
        asyncio.create_task(  # noqa: RUF006
            self._send_frame(
                {"jsonrpc": "2.0", "id": prompt_id, "method": "session/prompt", "params": params}
            )
        )
        return prompt_id

    async def _await_prompt(self, prompt_id: int) -> Any:
        fut = self._pending.get(prompt_id)
        if fut is None:
            raise RuntimeError(f"No pending future for prompt id {prompt_id}")
        try:
            return await fut
        finally:
            self._done.set()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and await its response."""
        if self._proc is None:
            raise RuntimeError("subprocess not started")
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._send_frame(
            {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        )
        return await fut

    async def _send_frame(self, frame: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("copilot stdin not available")
        data = (json.dumps(frame) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
        log.debug("→ copilot %s", frame.get("method") or f"response#{frame.get('id')}")

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("copilot non-JSON stdout line: %r", line[:200])
                    continue
                self._dispatch(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("copilot reader crashed: %s", e)
            self._error = str(e)
        finally:
            # Reader exited — fail any pending futures so callers don't hang.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("copilot stdout closed"))
            self._pending.clear()
            self._done.set()

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                log.debug("copilot stderr: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _dispatch(self, frame: dict[str, Any]) -> None:
        """Route a single JSON-RPC frame to the right handler."""
        if "id" in frame and ("result" in frame or "error" in frame):
            # Response to one of our requests.
            req_id = frame["id"]
            fut = self._pending.pop(req_id, None)
            if fut is None or fut.done():
                log.debug("copilot response with no waiter: id=%s", req_id)
                return
            if "error" in frame:
                err = frame["error"]
                fut.set_exception(
                    RuntimeError(f"copilot error {err.get('code')}: {err.get('message')}")
                )
            else:
                result = frame.get("result") or {}
                if req_id == self._current_prompt_id:
                    self._last_stop_reason = result.get("stopReason")
                    self._current_prompt_id = None
                fut.set_result(result)
            return

        # Notification.
        method = frame.get("method")
        params = frame.get("params") or {}
        if method == "session/update":
            self._handle_session_update(params)
        else:
            log.debug("copilot unhandled notification: %s", method)

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        update = params.get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            if content.get("type") == "text":
                text = content.get("text") or ""
                if text:
                    self._output_buffer.append(text)
        elif kind == "agent_thought_chunk":
            # We ignore thinking chunks for the user-visible buffer.
            pass
        # Other update kinds (tool_call, plan, etc.) are silently dropped for now.

    def _drain_output(self) -> str:
        text = "".join(self._output_buffer)
        self._output_buffer.clear()
        return text
