"""Abstract LLM provider interface. repr blocked, pickle blocked."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for Claude and Gemini providers."""

    name: str = "base"

    @abstractmethod
    async def analyze(self, role_prompt: str, data: dict) -> dict:
        """Send role prompt + market data, get structured analysis back."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and can accept requests."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} configured={self.is_available()}>"

    def __str__(self):
        return self.__repr__()

    def __getstate__(self):
        raise TypeError(f"{self.__class__.__name__} cannot be serialized (contains secrets)")
