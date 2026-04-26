"""Inter-holding correlation analysis.

Computes pairwise price correlation between holdings to detect
concentration risk from correlated positions.
"""
import pandas as pd
import numpy as np
from typing import Optional


class CorrelationAnalyzer:
    def __init__(self, warning_threshold: float = 0.7, lookback_days: int = 60):
        self.warning_threshold = warning_threshold
        self.lookback_days = lookback_days

    def compute_matrix(self, price_data: dict[str, pd.Series]) -> pd.DataFrame:
        """Compute correlation matrix from daily close prices.

        Args:
            price_data: {symbol: pd.Series of close prices}
        Returns:
            correlation matrix as DataFrame
        """
        if len(price_data) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(price_data)
        returns = df.pct_change().dropna()
        if len(returns) < 10:
            return pd.DataFrame()
        return returns.corr()

    def get_warnings(self, price_data: dict[str, pd.Series]) -> list[dict]:
        """Return pairs with correlation above threshold.

        Returns:
            list of {"pair": "A-B", "correlation": 0.85} dicts
        """
        matrix = self.compute_matrix(price_data)
        if matrix.empty:
            return []
        warnings = []
        symbols = list(matrix.columns)
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = matrix.iloc[i, j]
                if not pd.isna(corr) and abs(corr) > self.warning_threshold:
                    warnings.append({
                        "pair": f"{symbols[i]}-{symbols[j]}",
                        "correlation": round(corr, 3),
                    })
        return warnings

    def get_correlation_dict(self, price_data: dict[str, pd.Series]) -> dict[str, float]:
        """Return correlations as {pair_string: value} for RiskController."""
        warnings = self.get_warnings(price_data)
        return {w["pair"]: w["correlation"] for w in warnings}
