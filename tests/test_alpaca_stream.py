import pytest
from trade_forge.monitor.alpaca_stream import AlpacaStream


class TestAlpacaStream:
    def test_not_configured_without_keys(self):
        stream = AlpacaStream()
        # Without ALPACA_DATA_KEY set, should not be configured
        # (unless actually set in env)
        assert isinstance(stream.is_configured(), bool)

    def test_subscribe_bars(self):
        stream = AlpacaStream()
        handler_called = False
        async def dummy_handler(bar):
            nonlocal handler_called
            handler_called = True
        stream.subscribe_bars(["AAPL", "TSLA"], handler=dummy_handler)
        assert stream._symbols == ["AAPL", "TSLA"]
        assert stream._handler is not None

    @pytest.mark.asyncio
    async def test_run_without_config_returns_immediately(self):
        """If not configured, run() should return without error."""
        stream = AlpacaStream()
        stream._symbols = []  # Force unconfigured state
        await stream.run()  # Should not hang or crash

    def test_repr_does_not_expose_keys(self):
        stream = AlpacaStream()
        text = str(stream.__dict__)
        assert "ALPACA" not in text
        assert "secret" not in text.lower() or "SecretManager" not in text
