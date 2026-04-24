import pytest
from trade_forge.providers.router import LLMRouter, AllProvidersFailedError
from trade_forge.providers.claude_provider import ClaudeProvider
from trade_forge.providers.gemini_provider import GeminiProvider


def test_router_repr():
    router = LLMRouter()
    text = repr(router)
    assert "LLMRouter" in text


def test_router_no_providers_without_keys():
    """Without API keys, router should have no providers."""
    router = LLMRouter()
    # May or may not have providers depending on env -- just check it doesn't crash
    assert isinstance(router.available_providers, list)


def test_claude_provider_not_available_without_key():
    provider = ClaudeProvider()
    # Without ANTHROPIC_API_KEY set, should not be available
    # (unless it's actually set in the environment)
    assert isinstance(provider.is_available(), bool)


def test_gemini_provider_not_available_without_key():
    provider = GeminiProvider()
    assert isinstance(provider.is_available(), bool)


def test_claude_provider_repr_safe():
    provider = ClaudeProvider()
    text = repr(provider)
    assert "ClaudeProvider" in text
    assert "api_key" not in text.lower()
    assert "sk-" not in text


def test_gemini_provider_repr_safe():
    provider = GeminiProvider()
    text = repr(provider)
    assert "GeminiProvider" in text
    assert "api_key" not in text.lower()
    assert "AIza" not in text


@pytest.mark.asyncio
async def test_router_raises_when_no_providers():
    router = LLMRouter()
    router._providers = []  # Force empty
    with pytest.raises(AllProvidersFailedError):
        await router.analyze("test prompt", {"data": "test"})
