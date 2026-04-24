"""Claude API provider via Anthropic SDK."""
import logging
from trade_forge.providers.base import LLMProvider
from trade_forge.providers.sanitizer import DataSanitizer
from trade_forge.secrets import SecretManager

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                key = SecretManager.get("ANTHROPIC_API_KEY")
                if not key:
                    return None
                self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                logger.warning("anthropic package not installed")
                return None
        return self._client

    def is_available(self) -> bool:
        return SecretManager.is_configured("ANTHROPIC_API_KEY")

    async def analyze(self, role_prompt: str, data: dict) -> dict:
        client = self._get_client()
        if not client:
            raise RuntimeError("Claude provider not configured")
        sanitized = DataSanitizer.sanitize_for_llm(data)
        try:
            import json
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=role_prompt,
                messages=[{"role": "user", "content": json.dumps(sanitized, default=str)}],
            )
            return {"content": response.content[0].text, "_provider": self.name}
        except Exception as e:
            # Never expose raw error -- may contain request echo
            error_type = type(e).__name__
            logger.warning("Claude API error: %s", error_type)
            raise RuntimeError(f"Claude API error: {error_type}") from None
