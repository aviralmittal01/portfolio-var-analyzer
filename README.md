# Portfolio VaR & CVaR Analyzer

An educational Streamlit application for understanding **portfolio market risk** through Value at Risk (VaR), Expected Shortfall / CVaR, correlation, covariance, diversification and scenario simulation.

This version extends a single-asset VaR learning app into a **multi-asset, covariance-aware portfolio risk lab**. The goal is not to imitate a production risk platform; it is to make the core risk concepts visible, testable and interview-defendable.

## Why this project exists

VaR is easy to memorise and harder to internalise. A single-stock VaR number also misses one of the most important ideas in asset management: **portfolio risk depends on how positions move together, not just on each position's standalone risk**.

This project therefore focuses on the questions:

- How do portfolio weights affect risk?
- Why does correlation matter?
- Why is portfolio VaR not simply the sum of standalone VaRs?
- How do Historical, Parametric and Monte Carlo approaches differ?
- What does CVaR tell us that VaR does not?
- Where can model assumptions understate tail risk?

## Features

- Multi-asset portfolio with editable tickers and weights
- Weight validation (100% total, long-only learning scope)
- Adjusted historical price download via Yahoo Finance / `yfinance`
- Common-date alignment and a data-quality report
- Portfolio return, volatility, skewness and excess kurtosis
- Correlation matrix / heatmap
- Covariance-aware portfolio volatility
- Correlated multivariate-Normal Monte Carlo simulation
- Joint historical bootstrap simulation that preserves realised cross-asset dependence
- Historical simulation VaR / CVaR
- Parametric Normal-approximation VaR / CVaR
- Monte Carlo VaR / CVaR
- Portfolio P&L distribution and simulated portfolio-value paths
- Euler contribution to portfolio volatility
- Diversification diagnostics
- Method-comparison table and CSV export
- Explicit methodology, assumptions and limitations inside the app
- Unit tests for the core risk engine

## Risk methodology

### Portfolio return

For portfolio weights `w` and asset-return vector `r_t`:

`R_p,t = w' r_t`

The app assumes constant target weights for the learning horizon.

### Portfolio volatility

Using the historical covariance matrix `Σ`:

`σ_p = sqrt(w' Σ w)`

This is the key reason portfolio risk is not simply the weighted average of standalone risks: **covariances matter**.

### Monte Carlo: Correlated Normal

For each simulated day, the model draws an asset-return vector from a multivariate Normal distribution using the historical mean vector and covariance matrix. The weighted asset returns become the portfolio return for that day.

### Monte Carlo: Joint Historical Bootstrap

The model resamples entire historical return rows. Because all asset returns from a chosen historical day are sampled together, the observed cross-sectional dependence and non-Normal tail observations are retained.

### Historical simulation

Historical portfolio returns are replayed directly. For multi-day horizons, rolling compounded portfolio returns are used.

### Parametric VaR

The model uses a Delta-Normal style approximation based on portfolio mean and covariance-aware volatility, scaling mean by time and volatility by the square root of time.

### VaR and CVaR

- **VaR:** loss threshold exceeded in approximately `(1 - confidence)` of modelled scenarios.
- **CVaR / Expected Shortfall:** average loss in scenarios beyond the VaR cutoff.

VaR is **not** the maximum possible loss.

## Model scope / limitations

This project deliberately stays transparent and defendable:

- Long-only positions
- Constant target weights
- Adjusted closing-price data
- Same-currency holdings only (known mixed-currency portfolios are blocked)
- Historical behaviour used as the risk-estimation base
- No transaction costs, taxes or liquidity effects
- No intraday jumps or derivatives positions
- Normal Monte Carlo assumes a stable mean/covariance structure
- Bootstrap only resamples observed history
- Correlations can change sharply in stressed markets
- Yahoo Finance is an educational data source, not institutional market data

A production risk platform would require much stronger market-data controls, FX conversion, factor and stress models, liquidity modelling, independent model validation, governance and monitoring.

## Project structure

```text
BlackRock_Portfolio_VaR_Analyzer/
├── app.py                     # Streamlit user interface
├── portfolio_engine.py        # Portfolio risk maths and simulation
├── market_data.py             # Market-data access, alignment and quality report
├── tests/
│   └── test_portfolio_engine.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── INTERVIEW_DEFENSE_GUIDE.md
├── ATTRIBUTION.md
└── LICENSE
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and add the project files.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app.
4. Select the repository and `app.py` as the entry point.
5. Deploy.

No API key is required for the current Yahoo Finance data path.

## Suggested interview framing

Do **not** present this as proof of advanced Python expertise if that is not your background. A stronger and more accurate framing is:

> I was learning VaR and realised I understood the definition more than the intuition, so I used an existing educational single-stock implementation as a starting point and, with AI-assisted development, extended the idea into a multi-asset portfolio risk tool. The extension forced me to understand why covariance and correlation matter, why portfolio risk is not the sum of individual risks, and how Historical, Parametric and Monte Carlo VaR differ. I then worked through the methodology, assumptions and code flow so I could explain and validate what the tool was doing rather than treating the output as a black box.

See `INTERVIEW_DEFENSE_GUIDE.md` for the concepts and likely follow-up questions to prepare.

## Disclaimer

Educational project only. Not investment advice. Actual losses can exceed VaR, particularly during market stress.
