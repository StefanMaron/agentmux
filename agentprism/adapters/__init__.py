"""Provider-specific adapters for agentprism.

Each adapter implements the :class:`AgentAdapter` interface defined in
:mod:`agentprism.adapters.base`, wrapping a coding agent's native protocol
(stdio JSON-RPC, HTTP, etc.) behind a uniform async API.
"""

from agentprism.adapters.aider_adapter import AiderAdapter
from agentprism.adapters.base import AgentAdapter
from agentprism.adapters.copilot import CopilotAdapter
from agentprism.adapters.ollama import OllamaAdapter

__all__ = ["AgentAdapter", "AiderAdapter", "CopilotAdapter", "OllamaAdapter"]
