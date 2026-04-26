"""Edge Layer 3: Cross-market correlation factor."""
from quantforge.factors.base import Factor
from quantforge.core.models import FactorScore


class CrossMarketFactor(Factor):
    def __init__(self, weight: float = 0.15):
        super().__init__("crossmarket", weight)

    def compute(self, symbol: str, data: dict) -> FactorScore:
        scores = []

        sox = data.get("sox_return")
        if sox is not None:
            if sox > 2.0: scores.append(1.0)
            elif sox > 1.0: scores.append(0.7)
            elif sox > -1.0: scores.append(0.5)
            elif sox > -2.0: scores.append(0.3)
            else: scores.append(0.0)

        adr = data.get("adr_spread")
        if adr is not None:
            if adr > 0.02: scores.append(1.0)
            elif adr > 0.01: scores.append(0.7)
            elif adr > -0.01: scores.append(0.5)
            elif adr > -0.02: scores.append(0.3)
            else: scores.append(0.0)

        fx = data.get("fx_change")
        if fx is not None:
            if fx > 0.01: scores.append(1.0)
            elif fx > 0.0: scores.append(0.6)
            elif fx > -0.01: scores.append(0.4)
            else: scores.append(0.0)

        if not scores:
            return FactorScore(self.name, 0.5, 0.5, self.weight)

        avg = sum(scores) / len(scores)
        return FactorScore(self.name, avg, avg, self.weight)
