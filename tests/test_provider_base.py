import pickle
import pytest
from trade_forge.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    name = "dummy"
    async def analyze(self, role_prompt, data):
        return {"result": "test"}
    def is_available(self):
        return True


def test_repr_does_not_expose_internals():
    p = DummyProvider()
    text = repr(p)
    assert "DummyProvider" in text
    assert "configured=True" in text


def test_str_same_as_repr():
    p = DummyProvider()
    assert str(p) == repr(p)


def test_pickle_blocked():
    p = DummyProvider()
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(p)
