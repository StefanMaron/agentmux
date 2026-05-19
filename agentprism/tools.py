"""MCP tool definitions and dispatch for agentprism.

Each tool is described by a JSON schema (consumed by the MCP SDK to
advertise capabilities) and a coroutine handler that operates on the
shared :class:`SessionRegistry`.

Tools
-----
* ``agent_providers`` — which providers are installed and authenticated
* ``agent_models``    — list models for a provider (or all providers)
* ``agent_spawn``     — start an agent in the background
* ``agent_resume``    — start a new turn on an existing session (non-blocking)
* ``agent_status``    — working / idle / done / error
* ``agent_wait``      — block until the current turn finishes
* ``agent_list``      — enumerate active sessions
* ``agent_kill``      — terminate a session
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentprism.session import PROVIDERS, SessionRegistry, git_delta

DEFAULT_PROVIDER = os.environ.get("AGENTPRISM_DEFAULT_PROVIDER", "")


def _wait_cap_seconds() -> float:
    """Server-side cap on a single ``agent_wait`` / ``agent_run`` call.

    Long blocking waits outlive the host's per-tool-call timeout and tear
    down the MCP stdio channel. We cap each call at this value and return
    ``status: "still_running"`` so the caller can poll again.
    """
    raw = os.environ.get("AGENTPRISM_WAIT_CAP_SECONDS", "60")
    try:
        v = float(raw)
        return v if v > 0 else 60.0
    except ValueError:
        return 60.0


def _quota_error_response(output: str, session: Any) -> dict | None:
    """Return a structured quota-exceeded error dict if output signals a quota error."""
    if not output.startswith("[quota_exceeded]"):
        return None
    return {
        "error": "quota_exceeded",
        "provider": session.provider,
        "model": session.model or "unknown",
        "message": output,
        "suggestion": (
            "Try provider='ollama', model='qwen2.5-coder:14b-8k' for free local inference, "
            "or provider='gemini', model='gemini-2.5-flash' for free cloud inference."
        ),
    }

# ---------------------------------------------------------------------- schemas


def tool_definitions() -> list[dict[str, Any]]:
    """Return the JSON-schema definitions for every agentprism tool."""
    return [
        {
            "name": "agent_providers",
            "description": (
                "Check which coding-agent providers are installed and authenticated "
                "on this machine. Call this before agent_spawn when you don't know "
                "which providers are available. Only spawn workers for providers "
                "where available=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_models",
            "description": (
                "List available models for a coding-agent provider, including the "
                "premium-request multiplier for each. Pass no provider to list models "
                "for every supported provider."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider id (e.g. 'copilot'). Omit for all providers.",
                        "enum": sorted(PROVIDERS.keys()),
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_spawn",
            "description": (
                "Delegate a coding task to an external coding agent (Copilot, Claude Code, or Codex) "
                "running as a background worker. Returns immediately with a session_id. "
                "USE THIS when: the user asks to delegate/offload/hand off a task to Copilot or another agent; "
                "you want to run multiple tasks in parallel without blocking; "
                "the task is large and you want to preserve your own context window. "
                "After the current turn ends, use agent_resume(session_id, message) to start another "
                "turn on the same session. Use agent_wait to block until done. "
                "For a simpler one-shot pattern with no session tracking, use agent_run instead. "
                "Provider guide: 'copilot' for most implementation tasks (1x cost); "
                "'claude' for deep reasoning; 'codex' for OpenAI models. "
                f"Default if omitted: '{DEFAULT_PROVIDER or 'copilot'}'."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Initial prompt / task for the agent.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Absolute working directory for the agent.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": sorted(PROVIDERS.keys()),
                        "description": (
                            "Which coding agent to use: 'copilot', 'claude', or 'codex'. "
                            "Omit to use the default (AGENTPRISM_DEFAULT_PROVIDER env var, or 'copilot')."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model id (see agent_models).",
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "Optional session mode: 'agent' (default), 'plan', or 'autopilot'."
                        ),
                    },
                },
                "required": ["task", "cwd"],
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_run",
            "description": (
                "Delegate a coding task to an external coding agent and return the result. "
                "One-shot: spawns the agent, blocks until done, cleans up — no session tracking needed. "
                "USE THIS when: the user asks to 'let Copilot handle this', 'delegate to Copilot', "
                "'offload to another agent', or 'use Copilot for X'; "
                "the task is self-contained and needs no mid-task corrections; "
                "you want to offload implementation work to preserve your own context window. "
                "Use agent_spawn instead when you need to send follow-up turns or run workers in parallel. "
                "Each call blocks at most AGENTPRISM_WAIT_CAP_SECONDS (default 60s); if the worker is "
                "still running, returns status='still_running' with a session_id — call agent_wait on "
                "that session_id to continue polling."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task for the agent to complete.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Absolute working directory for the agent.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": sorted(PROVIDERS.keys()),
                        "description": "Which coding agent to use. Omit for default.",
                    },
                    "model": {"type": "string", "description": "Optional model id."},
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Max seconds to wait. Omit to wait indefinitely.",
                    },
                },
                "required": ["task", "cwd"],
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_resume",
            "description": (
                "Start a new turn on an existing agent session with a follow-up message. "
                "Non-blocking — returns immediately; use agent_wait or agent_status to observe the result. "
                "The session must NOT be currently working — call agent_wait first to drain the in-flight turn. "
                "There is no way to deliver a message to a running subprocess; each turn is a fresh "
                "subprocess invocation that resumes the prior conversation via the provider's own resume "
                "mechanism (--resume / thread_id). If you need to abort the current turn, use agent_kill "
                "and spawn a new session."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "message":    {"type": "string"},
                },
                "required": ["session_id", "message"],
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_status",
            "description": (
                "Report the current state of an agent session. Returns: "
                "status (working/idle/done/error), "
                "new_commits and working_tree_changes (git activity since spawn), "
                "activity.process_alive (is the subprocess still running), "
                "activity.last_activity_seconds_ago (seconds since last output line — rises if stuck), "
                "activity.uptime_seconds (total time running). "
                "DO NOT kill a session just because output text is empty or last_activity is high — "
                "the agent may be waiting for an LLM API response (normal). "
                "Only kill if process_alive is false, or working_tree_changes is empty after many minutes with no commits."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_wait",
            "description": (
                "Block until the agent's current turn finishes (or timeout), then return its output "
                "plus git context: new_commits made since spawn and working_tree_changes. "
                "No need to run git log or git status after this — the delta is included. "
                "Each call is capped server-side at AGENTPRISM_WAIT_CAP_SECONDS (default 60s) so the "
                "MCP connection never times out; if the cap fires before the worker finishes, the "
                "response has status='still_running' and the caller should call agent_wait again to "
                "continue polling. Pass a small timeout_seconds (e.g. 30) for a quick check, or omit "
                "it entirely to wait one cap-window at a time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id":      {"type": "string"},
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Optional timeout. Omit to wait indefinitely.",
                    },
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_list",
            "description": "List every active agent session managed by this server.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "agent_kill",
            "description": "Terminate an agent session and free its subprocess.",
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
    ]


# --------------------------------------------------------------------- dispatch


class ToolDispatcher:
    """Dispatches MCP tool calls to async handlers."""

    def __init__(self, registry: SessionRegistry) -> None:
        self.registry = registry

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a single tool call. Always returns a string for MCP text content."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        result = await handler(**(arguments or {}))
        return result if isinstance(result, str) else json.dumps(result, indent=2)

    # -- handlers ------------------------------------------------------------

    async def _tool_agent_providers(self) -> dict:
        results = []
        for name, cls in PROVIDERS.items():
            status = cls.check_available()
            results.append({
                "provider": name,
                "available": status.available,
                "installed": status.installed,
                "authenticated": status.authenticated,
                "note": status.note,
            })
        return {"providers": results}

    async def _tool_agent_models(self, provider: str | None = None) -> dict:
        if provider is not None:
            cls = self.registry.adapter_class(provider)
            return {"provider": provider, "models": cls.models()}
        return {
            "providers": {
                name: cls.models() for name, cls in PROVIDERS.items()
            }
        }

    async def _tool_agent_run(
        self,
        task: str,
        cwd: str,
        provider: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        if not provider:
            provider = DEFAULT_PROVIDER or "copilot"
        session = await self.registry.spawn(
            provider=provider, task=task, cwd=cwd, model=model
        )
        import asyncio
        cap = _wait_cap_seconds()
        effective = cap if timeout_seconds is None else min(timeout_seconds, cap)
        try:
            output = await session.adapter.wait(session.session_id, timeout=effective)
        except TimeoutError:
            # If the caller didn't set a timeout, or there's still time on
            # their budget, return still_running and leave the worker alive.
            if timeout_seconds is None or timeout_seconds > effective:
                return {
                    "session_id": session.session_id,
                    "provider": provider,
                    "status": "still_running",
                    "elapsed_seconds": effective,
                    "hint": (
                        "Capped at AGENTPRISM_WAIT_CAP_SECONDS so the MCP connection "
                        "stays alive. Call agent_wait(session_id) to continue polling, "
                        "or agent_kill to stop the worker."
                    ),
                }
            # Caller's own deadline expired — kill and report.
            try:
                await self.registry.kill(session.session_id)
            except Exception:
                pass
            return {"provider": provider, "status": "timeout", "error": f"Timed out after {timeout_seconds}s"}

        quota_resp = _quota_error_response(output, session)
        if quota_resp:
            try:
                await self.registry.kill(session.session_id)
            except Exception:
                pass
            return quota_resp
        delta = await asyncio.get_event_loop().run_in_executor(
            None, git_delta, session.cwd, session.git_base_sha
        )
        try:
            await self.registry.kill(session.session_id)
        except Exception:
            pass
        return {"provider": provider, "output": output, **delta}

    async def _tool_agent_spawn(
        self,
        task: str,
        cwd: str,
        provider: str | None = None,
        model: str | None = None,
        mode: str | None = None,
    ) -> dict:
        if not provider:
            provider = DEFAULT_PROVIDER or "copilot"
        session = await self.registry.spawn(
            provider=provider, task=task, cwd=cwd, model=model, mode=mode
        )
        return {
            "session_id": session.session_id,
            "provider":   session.provider,
            "status":     "spawned",
            "message":    f"Agent {provider} started; use agent_wait or agent_status to observe.",
        }

    async def _tool_agent_resume(self, session_id: str, message: str) -> dict:
        session = self.registry.get(session_id)
        current = await session.adapter.status(session_id)
        if current == "working":
            return {
                "session_id": session_id,
                "error": "session_busy",
                "status": current,
                "hint": (
                    "Call agent_wait(session_id) until the current turn finishes, "
                    "then call agent_resume again. Or agent_kill to abort."
                ),
            }
        try:
            confirmation = await session.adapter.resume(session_id, message)
        except RuntimeError as e:
            return {"session_id": session_id, "error": str(e)}
        return {
            "session_id": session_id,
            "status": "resumed",
            "message": confirmation,
        }

    async def _tool_agent_status(self, session_id: str) -> dict:
        import asyncio
        session = self.registry.get(session_id)
        status = await session.adapter.status(session_id)
        delta = await asyncio.get_event_loop().run_in_executor(
            None, git_delta, session.cwd, session.git_base_sha
        )
        if status == "error":
            try:
                output = await asyncio.wait_for(
                    session.adapter.wait(session_id), timeout=2.0
                )
                quota_resp = _quota_error_response(output, session)
                if quota_resp:
                    return {"session_id": session_id, "status": status, **delta, **quota_resp}
            except Exception:
                pass
        result: dict = {"session_id": session_id, "status": status, **delta}
        # Add provider-specific activity info if available (e.g. log tailing, frame counts)
        if hasattr(session.adapter, "activity_info"):
            result["activity"] = session.adapter.activity_info()
        return result

    async def _tool_agent_wait(
        self, session_id: str, timeout_seconds: float | None = None
    ) -> dict:
        import asyncio
        session = self.registry.get(session_id)
        cap = _wait_cap_seconds()
        effective = cap if timeout_seconds is None else min(timeout_seconds, cap)
        try:
            output = await session.adapter.wait(session_id, timeout=effective)
        except TimeoutError:
            # If the caller hadn't set their own deadline (or has time left),
            # this is a server-side cap hit, not a real timeout. Tell the
            # caller to poll again. Subprocess stays alive.
            if timeout_seconds is None or timeout_seconds > effective:
                delta = await asyncio.get_event_loop().run_in_executor(
                    None, git_delta, session.cwd, session.git_base_sha
                )
                return {
                    "session_id": session_id,
                    "status": "still_running",
                    "elapsed_seconds": effective,
                    "hint": (
                        "Capped at AGENTPRISM_WAIT_CAP_SECONDS so the MCP connection "
                        "stays alive. Call agent_wait again to keep polling."
                    ),
                    **delta,
                }
            return {
                "session_id": session_id,
                "status": "timeout",
                "error": f"Timed out after {timeout_seconds}s",
            }
        quota_resp = _quota_error_response(output, session)
        if quota_resp:
            return {"session_id": session_id, **quota_resp}
        delta = await asyncio.get_event_loop().run_in_executor(
            None, git_delta, session.cwd, session.git_base_sha
        )
        return {"session_id": session_id, "status": "done", "output": output, **delta}

    async def _tool_agent_list(self) -> dict:
        return {"sessions": [s.summary() for s in self.registry.list()]}

    async def _tool_agent_kill(self, session_id: str) -> dict:
        await self.registry.kill(session_id)
        return {"session_id": session_id, "status": "killed"}
