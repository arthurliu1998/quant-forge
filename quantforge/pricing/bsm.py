"""Option pricing models: BSM, Bates, Heston (approx), SVI.

BSM uses Abramowitz & Stegun rational approximation (7.1.26) for the
standard normal CDF.  Max absolute error < 7.5e-8.

Bates: Merton jump-diffusion closed-form (Poisson-weighted BSM sum).
Heston: Quadratic skew approximation (adjusted vol → BSM).
SVI:    Gatheral's Stochastic Volatility Inspired parameterisation
        (implied vol surface → BSM).
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


# ── Bates (Merton jump-diffusion) ────────────────────────────────

def bates_put_price(spot: float, strike: float, vol: float,
                    T: float, r: float = 0.03,
                    lam: float = 0.1, mu_j: float = -0.05,
                    sigma_j: float = 0.10, n_terms: int = 10) -> float:
    """Merton jump-diffusion put price (Poisson-weighted BSM sum).

    Parameters
    ----------
    lam : jump intensity (avg jumps/year)
    mu_j : mean jump size (log-normal)
    sigma_j : jump volatility
    n_terms : Poisson expansion terms
    """
    if T <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    price = 0.0
    lam_prime = lam * (1.0 + mu_j)
    for n in range(n_terms):
        # Poisson weight
        weight = math.exp(-lam_prime * T) * (lam_prime * T) ** n / math.factorial(n)
        # Adjusted volatility
        sigma_n2 = vol * vol + n * sigma_j * sigma_j / T
        sigma_n = math.sqrt(sigma_n2)
        # Adjusted drift
        r_n = r - lam * mu_j + n * math.log(1.0 + mu_j) / T
        price += weight * bsm_put_price(spot, strike, sigma_n, T, r_n)
    return price


def bates_call_price(spot: float, strike: float, vol: float,
                     T: float, r: float = 0.03,
                     lam: float = 0.1, mu_j: float = -0.05,
                     sigma_j: float = 0.10, n_terms: int = 10) -> float:
    """Merton jump-diffusion call price."""
    if T <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    price = 0.0
    lam_prime = lam * (1.0 + mu_j)
    for n in range(n_terms):
        weight = math.exp(-lam_prime * T) * (lam_prime * T) ** n / math.factorial(n)
        sigma_n2 = vol * vol + n * sigma_j * sigma_j / T
        sigma_n = math.sqrt(sigma_n2)
        r_n = r - lam * mu_j + n * math.log(1.0 + mu_j) / T
        price += weight * bsm_call_price(spot, strike, sigma_n, T, r_n)
    return price


# ── Heston (quadratic skew approximation) ────────────────────────

def _heston_adjusted_vol(spot: float, strike: float, vol: float,
                         T: float, kappa: float = 2.0,
                         theta: float = 0.0, xi: float = 0.3,
                         rho: float = -0.7) -> float:
    """Approximate Heston implied vol via quadratic skew.

    vol is the ATM vol (sqrt of v0).
    theta defaults to vol**2 if passed as 0.
    """
    if vol <= 0 or spot <= 0 or strike <= 0:
        return vol
    if theta <= 0:
        theta = vol * vol
    moneyness = math.log(strike / spot)
    # First-order skew correction
    adj = vol + rho * xi / (2.0 * vol) * moneyness
    return max(adj, 0.01)


def heston_put_price(spot: float, strike: float, vol: float,
                     T: float, r: float = 0.03,
                     kappa: float = 2.0, theta: float = 0.0,
                     xi: float = 0.3, rho: float = -0.7) -> float:
    """Heston-approximate put: adjusted vol → BSM."""
    adj_vol = _heston_adjusted_vol(spot, strike, vol, T, kappa, theta, xi, rho)
    return bsm_put_price(spot, strike, adj_vol, T, r)


def heston_call_price(spot: float, strike: float, vol: float,
                      T: float, r: float = 0.03,
                      kappa: float = 2.0, theta: float = 0.0,
                      xi: float = 0.3, rho: float = -0.7) -> float:
    """Heston-approximate call: adjusted vol → BSM."""
    adj_vol = _heston_adjusted_vol(spot, strike, vol, T, kappa, theta, xi, rho)
    return bsm_call_price(spot, strike, adj_vol, T, r)


# ── SVI (Stochastic Volatility Inspired) ─────────────────────────

def _svi_implied_vol(spot: float, strike: float, vol: float, T: float,
                     b: float = 0.1, rho_svi: float = -0.4,
                     m: float = 0.0, sigma_svi: float = 0.1) -> float:
    """Gatheral SVI implied vol from the raw parameterisation.

    w(k) = a + b * (rho_svi * (k - m) + sqrt((k - m)^2 + sigma_svi^2))

    *a* is auto-calibrated so that ATM (k=0) matches the input *vol*:
      a = vol^2 * T - b * sqrt(m^2 + sigma_svi^2)
    """
    if spot <= 0 or strike <= 0 or T <= 0 or vol <= 0:
        return max(vol, 0.01)
    # Auto-calibrate a so ATM implied vol = vol
    atm_wing = b * (rho_svi * (0.0 - m) + math.sqrt((0.0 - m) ** 2 + sigma_svi ** 2))
    a = vol * vol * T - atm_wing

    k = math.log(strike / spot)
    w = a + b * (rho_svi * (k - m) + math.sqrt((k - m) ** 2 + sigma_svi ** 2))
    implied_var = w / T
    if implied_var <= 0:
        return 0.01
    return math.sqrt(implied_var)


def svi_put_price(spot: float, strike: float, vol: float,
                  T: float, r: float = 0.03,
                  a: float = 0.04, b: float = 0.1,
                  rho_svi: float = -0.4, m: float = 0.0,
                  sigma_svi: float = 0.1) -> float:
    """SVI put: implied vol from SVI surface → BSM.

    *a* parameter is ignored — auto-calibrated from *vol*.
    """
    iv = _svi_implied_vol(spot, strike, vol, T, b, rho_svi, m, sigma_svi)
    return bsm_put_price(spot, strike, iv, T, r)


def svi_call_price(spot: float, strike: float, vol: float,
                   T: float, r: float = 0.03,
                   a: float = 0.04, b: float = 0.1,
                   rho_svi: float = -0.4, m: float = 0.0,
                   sigma_svi: float = 0.1) -> float:
    """SVI call: implied vol from SVI surface → BSM.

    *a* parameter is ignored — auto-calibrated from *vol*.
    """
    iv = _svi_implied_vol(spot, strike, vol, T, b, rho_svi, m, sigma_svi)
    return bsm_call_price(spot, strike, iv, T, r)
