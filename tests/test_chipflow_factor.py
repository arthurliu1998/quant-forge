from quantforge.factors.chipflow_factor import ChipflowFactor


def test_all_bullish():
    f = ChipflowFactor(weight=0.30)
    data = {"institutional_flow": {
        "foreign_net_consecutive_days": 5, "trust_same_direction": True,
        "foreign_net_volume_ratio": 0.15, "margin_change_pct": -6.0,
        "short_change_pct": -15.0,
    }}
    score = f.compute("2330", data)
    assert score.normalized > 0.7


def test_all_bearish():
    f = ChipflowFactor(weight=0.30)
    data = {"institutional_flow": {
        "foreign_net_consecutive_days": -4, "trust_same_direction": False,
        "foreign_net_volume_ratio": -0.12, "margin_change_pct": 15.0,
        "short_change_pct": 25.0,
    }}
    score = f.compute("2330", data)
    assert score.normalized < 0.3


def test_missing_data():
    f = ChipflowFactor(weight=0.30)
    score = f.compute("2330", {})
    assert score.normalized == 0.5


def test_us_returns_none():
    f = ChipflowFactor(weight=0.30)
    result = f.compute("AAPL", {"market": "US"})
    assert result is None
