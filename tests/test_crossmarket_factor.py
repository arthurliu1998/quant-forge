from quantforge.factors.crossmarket_factor import CrossMarketFactor


def test_strong_positive():
    f = CrossMarketFactor(weight=0.15)
    score = f.compute("2330", {"sox_return": 2.5, "adr_spread": 0.015, "fx_change": 0.008})
    assert score.normalized > 0.7


def test_strong_negative():
    f = CrossMarketFactor(weight=0.15)
    score = f.compute("2330", {"sox_return": -2.5, "adr_spread": -0.02, "fx_change": -0.01})
    assert score.normalized < 0.3


def test_no_data():
    f = CrossMarketFactor(weight=0.15)
    score = f.compute("AAPL", {})
    assert score.normalized == 0.5


def test_partial_data():
    f = CrossMarketFactor(weight=0.20)
    score = f.compute("2454", {"sox_return": 1.5})
    assert 0.5 < score.normalized < 1.0
    assert score.weight == 0.20
