"""Edge Layer 4: Sentiment factor scoring.
Accepts pre-computed FinBERT scores. Actual FinBERT integration in Plan 2.
"""
import math
from quantforge.factors.base import Factor
from quantforge.core.models import FactorScore


class SentimentFactor(Factor):
    def __init__(self, weight: float = 0.20, neg_boost: float = 1.5,
                 decay_lambda: float = 0.5):
        super().__init__("sentiment", weight)
        self.neg_boost = neg_boost
        self.decay_lambda = decay_lambda

    def compute(self, symbol: str, data: dict) -> FactorScore:
        single = data.get("finbert_sentiment")
        if single is not None:
            normalized = (single + 1.0) / 2.0
            return FactorScore(self.name, single, max(0.0, min(1.0, normalized)), self.weight)

        scores = data.get("finbert_scores")
        if scores and len(scores) > 0:
            weighted_sum = 0.0
            weight_sum = 0.0
            for i, s in enumerate(scores):
                time_weight = math.exp(-self.decay_lambda * i)
                boost = self.neg_boost if s < 0 else 1.0
                weighted_sum += s * time_weight * boost
                weight_sum += time_weight * boost
            raw = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            normalized = (raw + 1.0) / 2.0
            return FactorScore(self.name, raw, max(0.0, min(1.0, normalized)), self.weight)

        return FactorScore(self.name, 0.0, 0.5, self.weight)
