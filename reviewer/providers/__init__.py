"""Review provider implementations.

Each provider is a concrete :class:`khonliang_reviewer.ReviewProvider`
implementation wrapping a specific backend transport. Providers stay
transport- and storage-agnostic beyond their own adapter code — the bus
skill wiring and SQLite usage persistence live one layer up, in
``reviewer.agent``.
"""

from reviewer.providers._prompt import REVIEW_RESPONSE_SCHEMA, build_review_prompt
from reviewer.providers.claude_cli import (
    ClaudeCliAuthError,
    ClaudeCliProvider,
    ClaudeCliProviderConfig,
)
from reviewer.providers.codex_cli import (
    CodexCliAuthError,
    CodexCliProvider,
    CodexCliProviderConfig,
)
from reviewer.providers.gh_copilot import (
    GhCopilotAuthError,
    GhCopilotProvider,
    GhCopilotProviderConfig,
)
from reviewer.providers.ollama import (
    OllamaAuthError,
    OllamaHealthcheckError,
    OllamaProvider,
    OllamaProviderConfig,
)


__all__ = [
    "REVIEW_RESPONSE_SCHEMA",
    "ClaudeCliAuthError",
    "ClaudeCliProvider",
    "ClaudeCliProviderConfig",
    "CodexCliAuthError",
    "CodexCliProvider",
    "CodexCliProviderConfig",
    "GhCopilotAuthError",
    "GhCopilotProvider",
    "GhCopilotProviderConfig",
    "OllamaAuthError",
    "OllamaHealthcheckError",
    "OllamaProvider",
    "OllamaProviderConfig",
    "build_review_prompt",
]
