import numpy as np
import pandas as pd
import pytest

import portfolio_engine as pe


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    n = 1200
    # Three correlated assets from a stable covariance matrix.
    cov = np.array(
        [
            [0.000225, 0.000090, 0.000045],
            [0.000090, 0.000196, 0.000035],
            [0.000045, 0.000035, 0.000100],
        ]
    )
    mu = np.array([0.0004, 0.0003, 0.0002])
    x = rng.multivariate_normal(mu, cov, size=n)
    return pd.DataFrame(x, columns=["A", "B", "C"])


@pytest.fixture
def weights():
    return np.array([0.4, 0.35, 0.25])


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        pe.validate_weights(np.array([0.5, 0.4]))


def test_portfolio_returns_shape(returns, weights):
    rp = pe.portfolio_returns(returns, weights)
    assert len(rp) == len(returns)
    assert rp.name == "Portfolio"


def test_simulation_shape_and_start(returns, weights):
    paths = pe.simulate_portfolio_paths(
        returns, weights, 100_000, horizon=5, n_sims=2000, seed=1
    )
    assert paths.shape == (2000, 6)
    assert np.all(paths[:, 0] == 100_000)
    assert np.all(paths > 0)


def test_seed_reproducible(returns, weights):
    a = pe.simulate_portfolio_paths(returns, weights, 50_000, 3, 1000, seed=9)
    b = pe.simulate_portfolio_paths(returns, weights, 50_000, 3, 1000, seed=9)
    assert np.array_equal(a, b)


def test_cvar_at_least_var(returns, weights):
    paths = pe.simulate_portfolio_paths(returns, weights, 100_000, 5, 50_000, seed=2)
    var, cvar = pe.var_cvar(pe.pnl_from_paths(paths), 0.95)
    assert cvar >= var >= 0


def test_higher_confidence_means_higher_var(returns, weights):
    paths = pe.simulate_portfolio_paths(returns, weights, 100_000, 5, 100_000, seed=2)
    pnl = pe.pnl_from_paths(paths)
    assert pe.var_cvar(pnl, 0.99)[0] > pe.var_cvar(pnl, 0.95)[0]


def test_bootstrap_preserves_shape(returns, weights):
    paths = pe.simulate_portfolio_paths(
        returns, weights, 100_000, 10, 5000, method="bootstrap", seed=3
    )
    assert paths.shape == (5000, 11)


def test_risk_contributions_sum_to_100_percent(returns, weights):
    rc = pe.volatility_contributions(returns, weights)
    assert rc["Risk contribution %"].sum() == pytest.approx(1.0, abs=1e-10)


def test_diversification_reduces_vol_for_imperfect_correlation(returns, weights):
    d = pe.diversification_diagnostics(returns, weights, 5, 100_000, 0.95)
    assert d["weighted_avg_standalone_vol_daily"] >= d["portfolio_vol_daily"]
    assert d["diversification_ratio"] >= 1.0


def test_method_comparison_has_expected_rows(returns, weights):
    paths = pe.simulate_portfolio_paths(returns, weights, 100_000, 5, 20_000, seed=5)
    pnl = pe.pnl_from_paths(paths)
    df = pe.compare_methods(
        returns, weights, pnl, "MC", 5, 100_000, 0.95, zero_drift=False
    )
    assert list(df["Method"]) == [
        "MC",
        "Historical simulation",
        "Parametric (Normal approx.)",
    ]
    assert (df["CVaR"] >= df["VaR"]).all()
