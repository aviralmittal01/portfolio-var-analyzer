"""Market data helpers for the Portfolio VaR app.

Data is sourced from Yahoo Finance via yfinance.  This module intentionally
contains no Streamlit code so that data access and alignment logic can be
reused and tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-var-app)"}
ALLOWED_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}


@dataclass(frozen=True)
class PriceQuality:
    symbol: str
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    raw_observations: int
    aligned_observations: int
    dropped_for_alignment: int
    currency: str


def _clean_quotes(quotes: list[dict]) -> list[dict]:
    results: list[dict] = []
    for q in quotes:
        if q.get("quoteType") not in ALLOWED_TYPES or not q.get("symbol"):
            continue
        results.append(
            {
                "symbol": q["symbol"],
                "name": q.get("longname") or q.get("shortname") or q["symbol"],
                "exchange": q.get("exchDisp") or q.get("exchange") or "",
                "type": q.get("quoteType", ""),
            }
        )
    return results


def search_symbols(query: str, max_results: int = 8) -> list[dict]:
    """Search Yahoo Finance by company name or ticker."""
    query = query.strip()
    if not query:
        return []

    try:
        quotes = yf.Search(query, max_results=max_results).quotes
        results = _clean_quotes(quotes)
        if results:
            return results
    except Exception:
        pass

    try:
        resp = requests.get(
            YAHOO_SEARCH_URL,
            params={"q": query, "quotesCount": max_results, "newsCount": 0},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return _clean_quotes(resp.json().get("quotes", []))
    except Exception:
        return []


def load_prices(symbol: str, years: float) -> pd.Series:
    """Adjusted daily closing prices for one symbol."""
    days = int(years * 365) + 10
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    df = yf.Ticker(symbol).history(start=start, auto_adjust=True)
    if df.empty or "Close" not in df:
        raise ValueError(f"No price data found for '{symbol}'.")

    prices = df["Close"].dropna().astype(float)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.name = symbol
    return prices


def get_currency(symbol: str) -> str:
    """Trading currency code (for example USD or INR). Empty if unavailable."""
    try:
        return str(yf.Ticker(symbol).fast_info["currency"] or "")
    except Exception:
        return ""


def load_portfolio_prices(
    symbols: Iterable[str], years: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and align prices for a portfolio.

    Returns
    -------
    aligned_prices:
        Inner-joined adjusted closes.  Every retained date has a price for every
        security, which prevents artificial portfolio returns caused by missing
        observations.
    quality_report:
        One row per symbol showing raw observations, aligned observations,
        observations dropped by alignment, date range and trading currency.
    """
    cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not cleaned:
        raise ValueError("At least one ticker is required.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Duplicate tickers are not allowed.")

    series_map: dict[str, pd.Series] = {}
    currencies: dict[str, str] = {}
    for symbol in cleaned:
        series_map[symbol] = load_prices(symbol, years)
        currencies[symbol] = get_currency(symbol)

    outer = pd.concat(series_map.values(), axis=1, join="outer").sort_index()
    outer.columns = cleaned
    aligned = outer.dropna(how="any").copy()

    if len(aligned) < 60:
        raise ValueError(
            "The securities do not have enough overlapping history. "
            "Try a longer history window or a different set of tickers."
        )

    report_rows: list[dict] = []
    aligned_count = len(aligned)
    for symbol in cleaned:
        raw = series_map[symbol]
        report_rows.append(
            {
                "Ticker": symbol,
                "Currency": currencies[symbol] or "Unknown",
                "First date": raw.index.min().date(),
                "Last date": raw.index.max().date(),
                "Raw observations": len(raw),
                "Aligned observations": aligned_count,
                "Dropped for alignment": max(0, len(raw) - aligned_count),
            }
        )

    return aligned, pd.DataFrame(report_rows)
