"""Portfolio VaR & CVaR Analyzer — Streamlit application.

This project extends a single-asset educational VaR implementation into a
covariance-aware multi-asset portfolio risk lab.  The design goal is clarity:
show how weights, volatility, correlation, confidence and horizon affect risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import market_data
import portfolio_engine as pe

st.set_page_config(
    page_title="Portfolio VaR & CVaR Analyzer",
    page_icon="📉",
    layout="wide",
)

CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CNY": "¥", "CAD": "C$", "AUD": "A$", "HKD": "HK$", "CHF": "CHF ",
}


def money(x: float, currency: str) -> str:
    prefix = CURRENCY_SYMBOLS.get(currency, f"{currency} " if currency else "")
    return f"{prefix}{x:,.2f}"


@st.cache_data(ttl=3600, show_spinner="Downloading and aligning price history…")
def cached_portfolio_prices(symbols: tuple[str, ...], years: float):
    return market_data.load_portfolio_prices(symbols, years)


# --------------------------------------------------------------------------- #
# Sidebar — portfolio construction and model controls
# --------------------------------------------------------------------------- #
st.sidebar.title("📉 Portfolio Risk Lab")
st.sidebar.caption("Multi-asset VaR, CVaR, correlation and diversification")

st.sidebar.subheader("1) Portfolio")
default_holdings = pd.DataFrame(
    {
        "Ticker": ["AAPL", "MSFT", "JPM", "SPY"],
        "Weight %": [30.0, 25.0, 20.0, 25.0],
    }
)

holdings_editor = st.sidebar.data_editor(
    default_holdings,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Ticker", help="Yahoo Finance ticker, e.g. AAPL, MSFT, RELIANCE.NS"
        ),
        "Weight %": st.column_config.NumberColumn(
            "Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"
        ),
    },
    key="holdings_editor",
)

holdings = holdings_editor.copy()
holdings["Ticker"] = holdings["Ticker"].fillna("").astype(str).str.strip().str.upper()
holdings["Weight %"] = pd.to_numeric(holdings["Weight %"], errors="coerce")
holdings = holdings[(holdings["Ticker"] != "") & holdings["Weight %"].notna()].copy()

investment = st.sidebar.number_input(
    "Portfolio value", min_value=1.0, value=100_000.0, step=10_000.0, format="%.2f"
)

st.sidebar.subheader("2) Risk settings")
confidence = st.sidebar.select_slider(
    "Confidence level",
    options=[0.90, 0.95, 0.975, 0.99],
    value=0.95,
    format_func=lambda x: f"{x:.1%}".replace(".0%", "%"),
)
horizon = st.sidebar.slider("Horizon (trading days)", 1, 30, 5)
years = st.sidebar.slider("Years of price history", 1, 10, 3)

st.sidebar.subheader("3) Simulation")
method = st.sidebar.radio(
    "Monte Carlo method",
    ["normal", "bootstrap"],
    format_func=lambda m: {
        "normal": "Correlated Normal",
        "bootstrap": "Joint Historical Bootstrap",
    }[m],
)
n_sims = st.sidebar.select_slider(
    "Simulations", options=[10_000, 25_000, 50_000, 100_000, 200_000], value=50_000
)
zero_drift = st.sidebar.checkbox(
    "Assume zero drift",
    value=False,
    help="Useful for short-horizon risk because estimated average return is noisy.",
)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

with st.sidebar.expander("Model scope"):
    st.write(
        "Long-only portfolio; constant target weights; adjusted closing prices; "
        "same-currency holdings; no transaction costs or derivatives positions."
    )

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
st.title("Portfolio Value at Risk & Expected Shortfall")
st.caption(
    "A learning-focused market-risk tool showing how volatility, correlation and "
    "diversification shape portfolio VaR."
)

if holdings.empty:
    st.info("Add at least one ticker and weight in the sidebar.")
    st.stop()

if holdings["Ticker"].duplicated().any():
    st.error("Each ticker should appear only once. Combine duplicate positions first.")
    st.stop()

weight_total = float(holdings["Weight %"].sum())
if abs(weight_total - 100.0) > 0.01:
    st.error(f"Portfolio weights must sum to 100%. Current total: **{weight_total:.2f}%**")
    st.stop()

symbols = tuple(holdings["Ticker"].tolist())
weights = holdings["Weight %"].to_numpy(dtype=float) / 100.0

try:
    prices, quality = cached_portfolio_prices(symbols, float(years))
except Exception as exc:
    st.error(f"Could not build the portfolio price history: {exc}")
    st.stop()

known_currencies = sorted(set(quality.loc[quality["Currency"] != "Unknown", "Currency"]))
if len(known_currencies) > 1:
    st.error(
        "This version deliberately blocks mixed-currency portfolios because combining "
        "them without FX conversion would create misleading risk numbers. "
        f"Detected currencies: {', '.join(known_currencies)}."
    )
    st.stop()

currency = known_currencies[0] if known_currencies else ""
if not known_currencies:
    st.warning("Trading currency could not be verified. Interpret currency amounts carefully.")

returns = pe.simple_returns(prices)
stats = pe.portfolio_stats(returns, weights)

mc_paths = pe.simulate_portfolio_paths(
    returns,
    weights,
    investment,
    horizon,
    n_sims,
    method=method,
    zero_drift=zero_drift,
    seed=int(seed),
)
mc_pnl = pe.pnl_from_paths(mc_paths)
var, cvar = pe.var_cvar(mc_pnl, confidence)

mc_label = (
    "Monte Carlo (Correlated Normal)"
    if method == "normal"
    else "Monte Carlo (Joint Bootstrap)"
)
comparison = pe.compare_methods(
    returns,
    weights,
    mc_pnl,
    mc_label,
    horizon,
    investment,
    confidence,
    zero_drift,
)
risk_contrib = pe.volatility_contributions(returns, weights)
diversification = pe.diversification_diagnostics(
    returns, weights, horizon, investment, confidence, zero_drift
)

conf_txt = f"{confidence:.1%}".replace(".0%", "%")
day_txt = "1 day" if horizon == 1 else f"{horizon} days"

# --------------------------------------------------------------------------- #
# Headline risk view
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio value", money(investment, currency))
c2.metric("Annualised volatility", f"{stats.vol_annual:.1%}")
c3.metric(f"{conf_txt} VaR ({day_txt})", money(var, currency), f"{var/investment:.2%} of portfolio")
c4.metric(f"{conf_txt} CVaR ({day_txt})", money(cvar, currency), f"{cvar/investment:.2%} of portfolio")
c5.metric("Diversification ratio", f"{diversification['diversification_ratio']:.2f}x")

var_text = f"{currency} {var:,.2f}" if currency else f"{var:,.2f}"
cvar_text = f"{currency} {cvar:,.2f}" if currency else f"{cvar:,.2f}"
st.success(
    f"At {conf_txt} confidence over {day_txt}, the model estimates that portfolio losses "
    f"should not exceed {var_text}. In the worst {1-confidence:.1%} of modelled outcomes, "
    f"the average loss is {cvar_text} (CVaR / Expected Shortfall)."
)

if stats.excess_kurtosis > 1:
    st.warning(
        f"The historical portfolio return series has fat tails (excess kurtosis "
        f"{stats.excess_kurtosis:.1f}). Normal-based methods can understate extreme risk; "
        "compare them with Historical and Joint Bootstrap results."
    )

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs(
    [
        "🏠 Overview",
        "📊 Loss distribution",
        "🌀 Simulated paths",
        "🔗 Correlation",
        "⚖️ Risk contribution",
        "🧪 Method comparison",
        "✅ Data quality",
        "📚 Methodology",
    ]
)

with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Portfolio holdings")
        holdings_show = holdings.copy()
        holdings_show["Position value"] = holdings_show["Weight %"] / 100.0 * investment
        holdings_show["Position value"] = holdings_show["Position value"].map(
            lambda x: money(x, currency)
        )
        st.dataframe(holdings_show, hide_index=True, use_container_width=True)

        fig = go.Figure(
            go.Pie(
                labels=holdings["Ticker"],
                values=holdings["Weight %"],
                hole=0.55,
                textinfo="label+percent",
            )
        )
        fig.update_layout(title="Allocation", height=380, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Diversification diagnostics")
        st.metric(
            "Weighted average standalone volatility",
            f"{diversification['weighted_avg_standalone_vol_daily'] * np.sqrt(252):.1%} annualised",
        )
        st.metric(
            "Portfolio volatility after correlations",
            f"{diversification['portfolio_vol_daily'] * np.sqrt(252):.1%} annualised",
        )
        benefit = diversification["var_diversification_benefit"]
        st.metric(
            "Estimated VaR diversification benefit",
            money(benefit, currency),
            help=(
                "Estimated as the sum of stand-alone parametric VaRs minus portfolio parametric VaR. "
                "This is a diversification diagnostic; VaR is not guaranteed to be subadditive under all models."
            ),
        )
        st.info(
            "**Diversification insight:** Portfolio risk is not the simple sum of individual risks. "
            "Covariance and correlation determine how positions move together, which is the "
            "mechanism behind diversification."
        )

    st.subheader("Normalised price history")
    norm = prices / prices.iloc[0] * 100.0
    fig = go.Figure()
    for col in norm.columns:
        fig.add_scatter(x=norm.index, y=norm[col], mode="lines", name=col)
    fig.update_layout(height=420, yaxis_title="Growth of 100", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader(f"{day_txt} portfolio P&L distribution ({n_sims:,} simulations)")
    fig = go.Figure()
    fig.add_histogram(x=mc_pnl, nbinsx=80, opacity=0.85, name="P&L")
    fig.add_vline(x=-var, line_width=3, annotation_text=f"VaR {money(var, currency)}")
    fig.add_vline(
        x=-cvar,
        line_width=3,
        line_dash="dash",
        annotation_text=f"CVaR {money(cvar, currency)}",
    )
    fig.update_layout(
        xaxis_title=f"Profit / Loss ({currency})" if currency else "Profit / Loss",
        yaxis_title="Scenarios",
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected simulated P&L", money(float(mc_pnl.mean()), currency))
    m2.metric("Worst simulated", money(float(mc_pnl.min()), currency))
    m3.metric("Best simulated", money(float(mc_pnl.max()), currency))

with tabs[2]:
    st.subheader("Simulated portfolio-value paths")
    if horizon < 2:
        st.info("Increase the horizon to at least 2 trading days to view paths.")
    else:
        x = list(range(horizon + 1))
        p5, p25, p50, p75, p95 = np.percentile(mc_paths, [5, 25, 50, 75, 95], axis=0)
        fig = go.Figure()
        for i in range(min(50, mc_paths.shape[0])):
            fig.add_scatter(
                x=x,
                y=mc_paths[i],
                mode="lines",
                showlegend=False,
                line=dict(width=0.6),
                opacity=0.18,
                hoverinfo="skip",
            )
        fig.add_scatter(x=x, y=p95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=p5, mode="lines", line=dict(width=0), fill="tonexty", name="5th–95th percentile")
        fig.add_scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=p25, mode="lines", line=dict(width=0), fill="tonexty", name="25th–75th percentile")
        fig.add_scatter(x=x, y=p50, mode="lines", line=dict(width=3), name="Median")
        fig.add_hline(y=investment, line_dash="dot", annotation_text="Starting portfolio value")
        fig.update_layout(
            xaxis_title="Trading days ahead",
            yaxis_title=f"Portfolio value ({currency})" if currency else "Portfolio value",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Historical return correlation")
    corr = pe.correlation_matrix(returns)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Correlation measures how asset returns move together. Lower or negative correlations "
        "can reduce portfolio volatility; correlations can also rise during market stress."
    )

with tabs[4]:
    st.subheader("Contribution to portfolio volatility")
    rc = risk_contrib.copy()
    rc["Weight %"] = rc["Weight"] * 100
    rc["Risk contribution %"] = rc["Risk contribution %"] * 100
    st.dataframe(
        rc[["Ticker", "Weight %", "Risk contribution %"]].round(2),
        hide_index=True,
        use_container_width=True,
    )
    fig = go.Figure(
        go.Bar(x=rc["Ticker"], y=rc["Risk contribution %"], text=rc["Risk contribution %"].round(1))
    )
    fig.update_layout(yaxis_title="Contribution to portfolio volatility (%)", height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is Euler contribution to volatility, not Component VaR. A holding can contribute "
        "more or less risk than its capital weight depending on volatility and covariance."
    )

with tabs[5]:
    st.subheader(f"Risk-method comparison — {conf_txt}, {day_txt}")
    shown = comparison.copy()
    shown["VaR"] = shown["VaR"].map(lambda v: money(v, currency))
    shown["CVaR"] = shown["CVaR"].map(lambda v: money(v, currency))
    shown["VaR %"] = shown["VaR %"].map("{:.2f}%".format)
    shown["CVaR %"] = shown["CVaR %"].map("{:.2f}%".format)
    st.dataframe(shown, hide_index=True, use_container_width=True)

    st.write(
        "**How to read the differences:** Parametric VaR compresses the portfolio into mean, "
        "volatility and a Normal assumption. Historical simulation replays realised portfolio "
        "returns. Correlated Normal Monte Carlo creates many covariance-consistent scenarios, "
        "while the joint bootstrap resamples actual historical cross-asset return vectors."
    )

    st.download_button(
        "⬇️ Download method comparison (CSV)",
        comparison.round(6).to_csv(index=False).encode(),
        file_name="portfolio_var_method_comparison.csv",
        mime="text/csv",
    )

with tabs[6]:
    st.subheader("Data quality & provenance")
    st.dataframe(quality, hide_index=True, use_container_width=True)
    st.write(
        f"**Aligned sample:** {len(prices):,} common trading dates from "
        f"{prices.index.min().date()} to {prices.index.max().date()}."
    )
    st.caption(
        "Source: Yahoo Finance via yfinance. Prices are adjusted closes. An inner join is used so "
        "each portfolio date contains valid observations for every holding. This prevents missing "
        "data from silently creating artificial returns."
    )
    if (quality["Dropped for alignment"] > 20).any():
        st.warning(
            "Some securities lost many observations during alignment. Review listing dates, "
            "market holidays and data availability before interpreting results."
        )

with tabs[7]:
    st.subheader("Methodology, assumptions & limitations")
    st.markdown(
        """
### Core workflow
1. Download adjusted closing prices for every holding.
2. Align all securities on common trading dates.
3. Compute daily simple returns and the historical covariance/correlation matrix.
4. Apply portfolio weights to estimate portfolio return and volatility.
5. Generate portfolio scenarios using either correlated Normal draws or a joint historical bootstrap.
6. Convert terminal portfolio values into P&L.
7. VaR is the loss threshold at the selected tail percentile; CVaR is the average loss beyond VaR.

### Why covariance matters
If assets do not move perfectly together, total portfolio volatility can be lower than the weighted
average of standalone volatilities. That is the statistical mechanism behind diversification.

### Model assumptions
- Long-only portfolio and constant target weights.
- Holdings should be in the same currency; this version blocks known mixed-currency portfolios.
- Historical return behaviour is used to estimate future risk.
- Normal Monte Carlo assumes a stable mean/covariance structure.
- Bootstrap preserves observed historical cross-sectional dependence and tails, but only resamples history.
- No transaction costs, taxes, liquidity effects or intraday gaps.
- VaR is **not** a maximum possible loss. Losses can exceed VaR.

### Why CVaR matters
VaR tells you where the tail begins. CVaR / Expected Shortfall asks: *once I am in that bad tail, how
large is the average loss?* This gives more information about severity beyond the VaR cutoff.
        """
    )
    st.info(
        "**Scope note:** This tool is designed to make market-risk concepts visible, not to "
        "replace a production risk platform. A production implementation would require stronger "
        "market-data controls, currency conversion, factor/stress models, liquidity risk, validation, "
        "model governance and independent testing."
    )

st.divider()
st.caption(
    "⚠️ Educational project, not financial advice. Risk estimates depend on historical data and "
    "model assumptions; actual losses can exceed VaR, especially during market stress."
)
