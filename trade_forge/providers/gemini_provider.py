"""Gemini API provider via Google GenAI SDK."""
import logging
from trade_forge.providers.base import LLMProvider
from trade_forge.providers.sanitizer import DataSanitizer
from trade_forge.secrets import SecretManager

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-pro"):
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.genai as genai
                key = SecretManager.get("GOOGLE_AI_API_KEY")
                if not key:
                    return None
                self._client = genai.Client(api_key=key)
            except ImportError:
                logger.warning("google-genai package not installed")
                return None
        return self._client

    def is_available(self) -> bool:
        return SecretManager.is_configured("GOOGLE_AI_API_KEY")

    async def analyze(self, role_prompt: str, data: dict) -> dict:
        client = self._get_client()
        if not client:
            raise RuntimeError("Gemini provider not configured")
        sanitized = DataSanitizer.sanitize_for_llm(data)
        try:
            import json
            prompt = f"{role_prompt}\n\nData:\n{json.dumps(sanitized, default=str)}"
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            return {"content": response.text, "_provider": self.name}
        except Exception as e:
            error_type = type(e).__name__
            logger.warning("Gemini API error: %s", error_type)
            raise RuntimeError(f"Gemini API error: {error_type}") from None
