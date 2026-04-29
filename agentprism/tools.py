"""MCP tool definitions and dispatch for agentprism.

Each tool is described by a JSON schema (consumed by the MCP SDK to
advertise capabilities) and a coroutine handler that operates on the
shared :class:`SessionRegistry`.

Tools
-----
* ``agent_providers`` — which providers are installed and authenticated
* ``agent_models``    — list models for a provider (or all providers)
* ``agent_spawn``     — start an agent in the background
* ``agent_send``      — send a message and block for the response
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
                "Use agent_send to correct the worker mid-task. Use agent_wait to block until done. "
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
                "Use agent_spawn instead when you need to send corrections or run workers in parallel."
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
            "name": "agent_send",
            "description": (
                "Send a correction or follow-up message to a running agent session. "
                "Non-blocking — returns immediately. Use agent_wait or agent_status to observe the result."
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
                "No need to run git log or git status after this — the delta is included."
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
        try:
            import asyncio
            output = await session.adapter.wait(session.session_id, timeout=timeout_seconds)
            delta = await asyncio.get_event_loop().run_in_executor(
                None, git_delta, session.cwd, session.git_base_sha
            )
            return {"provider": provider, "output": output, **delta}
        except TimeoutError as e:
            return {"provider": provider, "status": "timeout", "error": str(e)}
        finally:
            try:
                await self.registry.kill(session.session_id)
            except Exception:
                pass

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

    async def _tool_agent_send(self, session_id: str, message: str) -> dict:
        session = self.registry.get(session_id)
        output = await session.adapter.send(session_id, message)
        return {"session_id": session_id, "output": output}

    async def _tool_agent_status(self, session_id: str) -> dict:
        import asyncio
        session = self.registry.get(session_id)
        status = await session.adapter.status(session_id)
        delta = await asyncio.get_event_loop().run_in_executor(
            None, git_delta, session.cwd, session.git_base_sha
        )
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
        try:
            output = await session.adapter.wait(session_id, timeout=timeout_seconds)
            delta = await asyncio.get_event_loop().run_in_executor(
                None, git_delta, session.cwd, session.git_base_sha
            )
            return {"session_id": session_id, "status": "done", "output": output, **delta}
        except TimeoutError as e:
            return {"session_id": session_id, "status": "timeout", "error": str(e)}

    async def _tool_agent_list(self) -> dict:
        return {"sessions": [s.summary() for s in self.registry.list()]}

    async def _tool_agent_kill(self, session_id: str) -> dict:
        await self.registry.kill(session_id)
        return {"session_id": session_id, "status": "killed"}
