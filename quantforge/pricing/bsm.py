"""Black-Scholes-Merton option pricing without scipy.

Uses Abramowitz & Stegun rational approximation (7.1.26) for the
standard normal CDF.  Max absolute error < 7.5e-8.
"""
import math


def norm_cdf(x: float) -> float:
    """Standard normal CDF via rational approximation."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


def bsm_put_price(spot: float, strike: float, vol: float,
                   T: float, r: float = 0.03) -> float:
    """BSM European put price.  *T* in years."""
    if T <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return strike * math.exp(-r * T) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bsm_call_price(spot: float, strike: float, vol: float,
                    T: float, r: float = 0.03) -> float:
    """BSM European call price.  *T* in years."""
    if T <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)
