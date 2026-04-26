from quantforge.factors.sentiment_factor import SentimentFactor


def test_with_score():
    f = SentimentFactor(weight=0.20)
    score = f.compute("AAPL", {"finbert_sentiment": 0.72})
    expected = (0.72 + 1) / 2
    assert abs(score.normalized - expected) < 0.01


def test_negative():
    f = SentimentFactor(weight=0.20)
    score = f.compute("AAPL", {"finbert_sentiment": -0.8})
    assert score.normalized < 0.2


def test_no_data():
    f = SentimentFactor(weight=0.20)
    score = f.compute("AAPL", {})
    assert score.normalized == 0.5


def test_multiple_scores():
    f = SentimentFactor(weight=0.35)
    score = f.compute("AAPL", {"finbert_scores": [0.5, -0.8, 0.3]})
    assert score.weight == 0.35
    assert 0.0 <= score.normalized <= 1.0
