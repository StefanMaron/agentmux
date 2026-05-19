"""Abstract base class for agentprism provider adapters."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass


class QuotaExceededError(RuntimeError):
    def __init__(self, provider: str, model: str, retry_after: str | None = None):
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        msg = f"[quota_exceeded] {provider} quota exceeded for model {model}."
        if retry_after:
            msg += f" Retry after: {retry_after}."
        msg += " Try a different provider or model."
        super().__init__(msg)


QUOTA_PATTERNS = [
    "429", "quota exceeded", "rate limit", "rate_limit", "exhausted",
    "too many requests", "resource_exhausted", "resource exhausted",
    "insufficient_quota", "exceeded your current quota",
]


def detect_quota_error(text: str, provider: str, model: str | None = None) -> QuotaExceededError | None:
    lower = text.lower()
    if any(p in lower for p in QUOTA_PATTERNS):
        retry = None
        for line in text.splitlines():
            if "retry" in line.lower() and any(c.isdigit() for c in line):
                retry = line.strip()[:80]
                break
        return QuotaExceededError(provider, model or "unknown", retry)
    return None


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
    async def resume(self, session_id: str, message: str) -> str:
        """Start a new turn on an existing session with ``message``.

        Non-blocking: kicks off the next turn (typically via the provider's
        ``--resume`` / thread-id mechanism) and returns a short confirmation
        immediately. Callers observe progress via :meth:`wait` / :meth:`status`.

        Implementations MUST raise if the session is currently ``"working"`` —
        the caller is expected to ``agent_wait`` first. There is no in-flight
        message delivery: every provider's protocol is turn-based, so a new
        message can only become its own turn after the current one ends.
        """

    @abstractmethod
    async def status(self, session_id: str) -> str:
        """Return one of ``"working"``, ``"idle"``, ``"done"``, ``"error"``."""

    @abstractmethod
    async def wait(self, session_id: str, timeout: float | None = None) -> str:
        """Block until the current turn finishes, then return accumulated output."""

    @abstractmethod
    async def kill(self, session_id: str) -> None:
        """Terminate the subprocess and release resources."""

    def child_pid(self, session_id: str) -> int | None:
        """Return the PID of the spawned worker, or None for non-subprocess adapters.

        Subprocess adapters override this so the registry can persist the PID
        for orphan recovery after agentprism restarts.
        """
        return None

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
