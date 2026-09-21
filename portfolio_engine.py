"""Portfolio risk mathematics for the multi-asset VaR application.

The model is deliberately transparent and easy to inspect:
- daily adjusted-close simple returns
- constant target portfolio weights
- correlated multivariate-normal Monte Carlo OR joint historical bootstrap
- Historical simulation, Parametric Normal VaR and CVaR comparison
- covariance-aware volatility and Euler volatility-risk contributions

All VaR/CVaR values are reported as positive loss numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class PortfolioStats:
    mean_daily: float
    vol_daily: float
    vol_annual: float
    skew: float
    excess_kurtosis: float
    n_obs: int


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns for an aligned price matrix."""
    if prices.shape[1] < 1:
        raise ValueError("Price matrix must contain at least one asset.")
    return prices.pct_change().dropna(how="any")


def validate_weights(weights: np.ndarray | pd.Series, tol: float = 1e-6) -> np.ndarray:
    """Return validated 1-D long-only portfolio weights that sum to one."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("Weights must be a non-empty one-dimensional vector.")
    if not np.isfinite(w).all():
        raise ValueError("Weights must be finite numbers.")
    if (w < 0).any():
        raise ValueError("This learning version supports long-only weights (no negatives).")
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"Weights must sum to 100%; current total is {w.sum():.2%}.")
    return w


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray | pd.Series) -> pd.Series:
    """Daily portfolio return under constant target weights."""
    w = validate_weights(weights)
    if returns.shape[1] != len(w):
        raise ValueError("Number of weights must match number of assets.")
    rp = returns.to_numpy(dtype=float) @ w
    return pd.Series(rp, index=returns.index, name="Portfolio")


def portfolio_stats(returns: pd.DataFrame, weights: np.ndarray | pd.Series) -> PortfolioStats:
    rp = portfolio_returns(returns, weights)
    return PortfolioStats(
        mean_daily=float(rp.mean()),
        vol_daily=float(rp.std()),
        vol_annual=float(rp.std() * sqrt(TRADING_DAYS)),
        skew=float(rp.skew()),
        excess_kurtosis=float(rp.kurt()),
        n_obs=int(len(rp)),
    )


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def simulate_portfolio_paths(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    investment: float,
    horizon: int,
    n_sims: int,
    method: str = "normal",
    zero_drift: bool = False,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate portfolio-value paths while preserving cross-asset dependence.

    normal:
        Draws each day's asset-return vector from a multivariate Normal using the
        historical covariance matrix.
    bootstrap:
        Resamples entire historical return rows.  Sampling rows jointly preserves
        the realised cross-sectional dependence and non-normal tails in history.

    The portfolio is modelled with constant target weights.  Each day's portfolio
    return is the weighted sum of that day's asset returns.
    """
    w = validate_weights(weights)
    if investment <= 0:
        raise ValueError("Investment must be positive.")
    if horizon < 1 or n_sims < 1:
        raise ValueError("Horizon and simulations must be positive integers.")
    if returns.shape[1] != len(w):
        raise ValueError("Number of weights must match number of assets.")

    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy(dtype=float)
    if zero_drift:
        mu = np.zeros_like(mu)

    if method == "normal":
        cov = returns.cov().to_numpy(dtype=float)
        daily_asset = rng.multivariate_normal(mu, cov, size=(n_sims, horizon))
    elif method == "bootstrap":
        hist = returns.to_numpy(dtype=float)
        centred = hist - hist.mean(axis=0) + mu
        picks = rng.integers(0, len(centred), size=(n_sims, horizon))
        daily_asset = centred[picks]
    else:
        raise ValueError("method must be 'normal' or 'bootstrap'")

    daily_port = np.einsum("sha,a->sh", daily_asset, w)
    # Guard against an impossible <-100% daily portfolio return from a Normal draw.
    daily_port = np.maximum(daily_port, -0.999999)
    growth = np.cumprod(1.0 + daily_port, axis=1)
    values = investment * growth
    return np.hstack([np.full((n_sims, 1), investment), values])


def pnl_from_paths(paths: np.ndarray) -> np.ndarray:
    """Terminal P&L for each simulated portfolio path."""
    return paths[:, -1] - paths[:, 0]


def var_cvar(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """Empirical VaR and CVaR / Expected Shortfall as positive loss numbers."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")
    cutoff = float(np.percentile(pnl, (1.0 - confidence) * 100.0))
    tail = pnl[pnl <= cutoff]
    if len(tail) == 0:
        return max(0.0, -cutoff), max(0.0, -cutoff)
    return max(0.0, -cutoff), max(0.0, -float(tail.mean()))


def historical_var_cvar(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    horizon: int,
    investment: float,
    confidence: float,
) -> tuple[float, float]:
    """Historical simulation using realised rolling compounded portfolio returns."""
    rp = portfolio_returns(returns, weights)
    if horizon == 1:
        horizon_returns = rp
    else:
        horizon_returns = (1.0 + rp).rolling(horizon).apply(np.prod, raw=True).dropna() - 1.0
    pnl = investment * horizon_returns.to_numpy(dtype=float)
    return var_cvar(pnl, confidence)


def parametric_var_cvar(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    horizon: int,
    investment: float,
    confidence: float,
    zero_drift: bool = False,
) -> tuple[float, float]:
    """Delta-normal portfolio VaR/CVaR using mean and covariance.

    This is intentionally labelled an approximation: horizon portfolio returns are
    approximated as Normal with mean scaled by time and volatility by sqrt(time).
    """
    w = validate_weights(weights)
    mean_daily = 0.0 if zero_drift else float(returns.mean().to_numpy() @ w)
    cov = returns.cov().to_numpy(dtype=float)
    vol_daily = float(np.sqrt(w @ cov @ w))

    mu_h = mean_daily * horizon
    sigma_h = vol_daily * sqrt(horizon)
    alpha = 1.0 - confidence
    nd = NormalDist()
    z = nd.inv_cdf(alpha)

    q = mu_h + z * sigma_h
    var = max(0.0, -investment * q)

    # Conditional lower-tail mean of a Normal random variable.
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    tail_mean = mu_h - sigma_h * phi / alpha
    cvar = max(0.0, -investment * tail_mean)
    return float(var), float(cvar)


def compare_methods(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    mc_pnl: np.ndarray,
    mc_label: str,
    horizon: int,
    investment: float,
    confidence: float,
    zero_drift: bool,
) -> pd.DataFrame:
    rows: list[tuple[str, float, float]] = []

    v, c = var_cvar(mc_pnl, confidence)
    rows.append((mc_label, v, c))

    v, c = historical_var_cvar(returns, weights, horizon, investment, confidence)
    rows.append(("Historical simulation", v, c))

    v, c = parametric_var_cvar(
        returns, weights, horizon, investment, confidence, zero_drift=zero_drift
    )
    rows.append(("Parametric (Normal approx.)", v, c))

    df = pd.DataFrame(rows, columns=["Method", "VaR", "CVaR"])
    df["VaR %"] = df["VaR"] / investment * 100.0
    df["CVaR %"] = df["CVaR"] / investment * 100.0
    return df


def volatility_contributions(
    returns: pd.DataFrame, weights: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Euler contribution of each holding to portfolio volatility.

    Component contribution_i = w_i * (Sigma w)_i / sigma_p.
    Percentage contributions sum to ~100% when portfolio volatility is positive.
    """
    w = validate_weights(weights)
    cov = returns.cov().to_numpy(dtype=float)
    sigma_p = float(np.sqrt(w @ cov @ w))
    if sigma_p <= 0:
        contrib = np.zeros_like(w)
        pct = np.zeros_like(w)
    else:
        marginal = cov @ w / sigma_p
        contrib = w * marginal
        pct = contrib / sigma_p

    return pd.DataFrame(
        {
            "Ticker": list(returns.columns),
            "Weight": w,
            "Vol contribution (daily)": contrib,
            "Risk contribution %": pct,
        }
    )


def diversification_diagnostics(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    horizon: int,
    investment: float,
    confidence: float,
    zero_drift: bool = False,
) -> dict[str, float]:
    """Simple, explainable diversification diagnostics under a Normal approximation."""
    w = validate_weights(weights)
    individual_vol = returns.std().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float)
    portfolio_vol = float(np.sqrt(w @ cov @ w))
    weighted_avg_vol = float(w @ individual_vol)
    diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else np.nan

    portfolio_var, _ = parametric_var_cvar(
        returns, w, horizon, investment, confidence, zero_drift=zero_drift
    )

    standalone_sum = 0.0
    for i, col in enumerate(returns.columns):
        one = returns[[col]]
        v, _ = parametric_var_cvar(
            one,
            np.array([1.0]),
            horizon,
            investment * w[i],
            confidence,
            zero_drift=zero_drift,
        )
        standalone_sum += v

    return {
        "portfolio_vol_daily": portfolio_vol,
        "weighted_avg_standalone_vol_daily": weighted_avg_vol,
        "diversification_ratio": diversification_ratio,
        "portfolio_parametric_var": portfolio_var,
        "sum_standalone_parametric_var": standalone_sum,
        "var_diversification_benefit": standalone_sum - portfolio_var,
    }
