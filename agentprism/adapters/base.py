"""Abstract base class for agentprism provider adapters."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderStatus:
    provider: str
    installed: bool
    authenticated: bool
    note: str = ""

    @property
    def available(self) -> bool:
        return self.installed and self.authenticated


class AgentAdapter(ABC):
    """Abstract interface every provider adapter must implement.

    A single adapter instance owns a single agent session (one subprocess,
    one logical conversation). The :class:`SessionRegistry` creates one
    adapter per ``agent_spawn`` call and tracks them by ``session_id``.

    All methods are async because adapters typically wrap stdio/network IO.
    """

    #: Provider identifier used in tool dispatch (e.g. ``"copilot"``).
    provider: str = ""

    @abstractmethod
    async def spawn(
        self,
        task: str,
        cwd: str,
        model: str | None = None,
        mode: str | None = None,
    ) -> str:
        """Start the agent subprocess and submit the initial task.

        Returns immediately with a ``session_id`` — the prompt continues
        running in the background. Use :meth:`wait` or :meth:`status` to
        observe progress.
        """

    @abstractmethod
    async def send(self, session_id: str, message: str) -> str:
        """Send a follow-up message and block until the agent responds."""

    @abstractmethod
    async def status(self, session_id: str) -> str:
        """Return one of ``"working"``, ``"idle"``, ``"done"``, ``"error"``."""

    @abstractmethod
    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        """Block until the current turn finishes, then return accumulated output."""

    @abstractmethod
    async def kill(self, session_id: str) -> None:
        """Terminate the subprocess and release resources."""

    @classmethod
    @abstractmethod
    def models(cls) -> list[dict]:
        """Return the list of models this provider supports.

        Each entry is a dict with at least ``id`` and ``multiplier`` keys,
        and optionally a ``note`` describing intended use.
        """

    @classmethod
    @abstractmethod
    def check_available(cls) -> ProviderStatus:
        """Return availability/auth status without making any API calls."""

    @classmethod
    def _binary_installed(cls, binary: str) -> bool:
        return shutil.which(binary) is not None
