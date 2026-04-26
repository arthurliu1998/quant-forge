from quantforge.factors.synthesizer import SignalSynthesizer
from quantforge.core.models import Regime, FactorScore


def test_tw_stock():
    s = SignalSynthesizer()
    signal = s.synthesize("2330", {
        "market": "TW", "regime": Regime.BULL_TREND,
        "technical_score": FactorScore("tech", 0.75, 0.75, 0.35),
        "chipflow_score": FactorScore("chip", 0.70, 0.70, 0.30),
        "crossmarket_score": FactorScore("cross", 0.80, 0.80, 0.15),
        "sentiment_score": FactorScore("sent", 0.67, 0.67, 0.20),
    })
    assert signal.symbol == "2330"
    assert signal.market == "TW"
    expected = (0.75*0.35 + 0.70*0.30 + 0.80*0.15 + 0.67*0.20) * 100
    assert abs(signal.quant_score - round(expected, 1)) < 0.2
    assert signal.signal_level == "BUY"


def test_us_no_chipflow():
    s = SignalSynthesizer()
    signal = s.synthesize("AAPL", {
        "market": "US", "regime": Regime.BULL_TREND,
        "technical_score": FactorScore("tech", 0.80, 0.80, 0.45),
        "chipflow_score": None,
        "crossmarket_score": FactorScore("cross", 0.70, 0.70, 0.20),
        "sentiment_score": FactorScore("sent", 0.60, 0.60, 0.35),
    })
    assert signal.edge_scores.chipflow is None
    expected = (0.80*0.45 + 0.70*0.20 + 0.60*0.35) * 100
    assert abs(signal.quant_score - round(expected, 1)) < 0.2


def test_bear_raises_threshold():
    s = SignalSynthesizer()
    signal = s.synthesize("2330", {
        "market": "TW", "regime": Regime.BEAR_TREND,
        "technical_score": FactorScore("tech", 0.75, 0.75, 0.35),
        "chipflow_score": FactorScore("chip", 0.75, 0.75, 0.30),
        "crossmarket_score": FactorScore("cross", 0.75, 0.75, 0.15),
        "sentiment_score": FactorScore("sent", 0.75, 0.75, 0.20),
    })
    assert signal.buy_threshold == 85.0
    assert signal.signal_level in ("WATCHLIST", "NO_SIGNAL")


def test_advisor_bonus():
    s = SignalSynthesizer()
    signal = s.synthesize("2330", {
        "market": "TW", "regime": Regime.BULL_TREND,
        "technical_score": FactorScore("tech", 0.65, 0.65, 0.35),
        "chipflow_score": FactorScore("chip", 0.65, 0.65, 0.30),
        "crossmarket_score": FactorScore("cross", 0.65, 0.65, 0.15),
        "sentiment_score": FactorScore("sent", 0.65, 0.65, 0.20),
        "advisor_bonus": 8.0,
    })
    assert signal.quant_score == 65.0
    assert signal.advisor_bonus == 8.0
    assert signal.combined_score == 73.0
    assert signal.signal_level == "ADVISOR_ASSISTED_BUY"


def test_advisor_cannot_override_low():
    s = SignalSynthesizer()
    signal = s.synthesize("TSLA", {
        "market": "US", "regime": Regime.BULL_TREND,
        "technical_score": FactorScore("tech", 0.50, 0.50, 0.45),
        "chipflow_score": None,
        "crossmarket_score": FactorScore("cross", 0.50, 0.50, 0.20),
        "sentiment_score": FactorScore("sent", 0.50, 0.50, 0.35),
        "advisor_bonus": 10.0,
    })
    assert signal.quant_score == 50.0
    assert signal.signal_level == "NO_SIGNAL"


def test_advisor_cap():
    s = SignalSynthesizer(advisor_cap=10.0)
    signal = s.synthesize("X", {
        "market": "US", "regime": Regime.NEUTRAL,
        "technical_score": FactorScore("t", 0.7, 0.7, 0.45),
        "chipflow_score": None,
        "crossmarket_score": FactorScore("c", 0.7, 0.7, 0.20),
        "sentiment_score": FactorScore("s", 0.7, 0.7, 0.35),
        "advisor_bonus": 20.0,
    })
    assert signal.advisor_bonus == 10.0
