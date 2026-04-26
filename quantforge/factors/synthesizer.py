"""Signal synthesizer: combines all factor scores into a QuantSignal.
Implements dual-score system (Option C):
- Quant score: deterministic, backtestable
- Advisor bonus: non-deterministic, capped at ±10
"""
from datetime import datetime
from typing import Optional
from quantforge.core.models import FactorScore, EdgeScores, QuantSignal, Regime


class SignalSynthesizer:
    def __init__(self, advisor_cap: float = 10.0):
        self.advisor_cap = advisor_cap

    def synthesize(self, symbol: str, factors: dict) -> QuantSignal:
        market = factors.get("market", "US")
        regime = factors.get("regime", Regime.NEUTRAL)

        tech = factors.get("technical_score")
        chip = factors.get("chipflow_score")
        cross = factors.get("crossmarket_score")
        sent = factors.get("sentiment_score")

        edge_scores = EdgeScores(
            technical=tech,
            chipflow=chip,
            crossmarket=cross,
            sentiment=sent,
        )

        quant_score = edge_scores.quant_score

        advisor_bonus = factors.get("advisor_bonus", 0.0)
        advisor_bonus = max(-self.advisor_cap, min(self.advisor_cap, advisor_bonus))

        buy_threshold = 85.0 if regime == Regime.BEAR_TREND else 70.0

        return QuantSignal(
            symbol=symbol,
            market=market,
            quant_score=round(quant_score, 1),
            advisor_bonus=advisor_bonus,
            regime=regime,
            edge_scores=edge_scores,
            timestamp=datetime.now().isoformat(),
            buy_threshold=buy_threshold,
        )
