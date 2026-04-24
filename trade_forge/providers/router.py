"""Auto-failover LLM router: tries primary provider, falls back on error."""
import logging
from trade_forge.providers.base import LLMProvider
from trade_forge.providers.claude_provider import ClaudeProvider
from trade_forge.providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    pass


class LLMRouter:
    """Routes LLM requests with automatic failover."""

    def __init__(self, primary: str = "auto"):
        self._providers: list[LLMProvider] = []
        self._primary = primary
        self._init_providers()

    def _init_providers(self):
        claude, gemini = ClaudeProvider(), GeminiProvider()
        order = [gemini, claude] if self._primary == "gemini" else [claude, gemini]
        self._providers = [p for p in order if p.is_available()]

    @property
    def available_providers(self) -> list[str]:
        return [p.name for p in self._providers]

    def has_providers(self) -> bool:
        return len(self._providers) > 0

    async def analyze(self, role_prompt: str, data: dict) -> dict:
        if not self._providers:
            raise AllProvidersFailedError("No LLM provider configured")
        errors = []
        for provider in self._providers:
            try:
                result = await provider.analyze(role_prompt, data)
                result["_provider"] = provider.name
                return result
            except Exception as e:
                logger.warning("%s failed: %s -- trying next", provider.name, e)
                errors.append(f"{provider.name}: {e}")
                continue
        raise AllProvidersFailedError(f"All providers failed: {'; '.join(errors)}")

    def __repr__(self):
        names = [p.name for p in self._providers]
        return f"<LLMRouter providers={names}>"
