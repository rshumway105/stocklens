"""
Fundamental data fetcher — extracts valuation, profitability, growth, and
balance sheet metrics from yfinance.

yfinance exposes fundamentals via Ticker.info, Ticker.financials,
Ticker.quarterly_financials, Ticker.balance_sheet, and Ticker.cashflow.
This module normalizes those into a flat dict of key metrics, suitable for
conversion into model features.

Note: yfinance fundamental data can be spotty for some tickers (especially
ETFs, which have no fundamentals).  The fetcher returns what's available
and fills gaps with NaN.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """
    Fetch a snapshot of fundamental metrics for a single ticker.

    Returns a flat dictionary with keys like:
        pe_ratio, forward_pe, pb_ratio, ps_ratio, ev_ebitda,
        gross_margin, operating_margin, net_margin, roe, roa,
        debt_to_equity, current_ratio, fcf_yield, revenue_growth, ...

    Missing values are set to NaN.
    """
    logger.info("Fetching fundamentals for {}", ticker)

    try:
        yf_ticker = yf.Ticker(ticker)
        info: dict = yf_ticker.info or {}
    except Exception as e:
        logger.error("Failed to get info for {}: {}", ticker, e)
        return _empty_fundamentals(ticker)

    # Helper: safely pull from info dict, defaulting to NaN
    def g(key: str, default: float = np.nan) -> float:
        val = info.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    fundamentals = {
        "ticker": ticker.upper(),
        "fetch_date": pd.Timestamp.now().isoformat(),

        # Valuation
        "pe_ratio": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "pb_ratio": g("priceToBook"),
        "ps_ratio": g("priceToSalesTrailing12Months"),
        "ev_ebitda": g("enterpriseToEbitda"),
        "peg_ratio": g("pegRatio"),
        "market_cap": g("marketCap"),
        "enterprise_value": g("enterpriseValue"),

        # Profitability
        "gross_margin": g("grossMargins"),
        "operating_margin": g("operatingMargins"),
        "net_margin": g("profitMargins"),
        "roe": g("returnOnEquity"),
        "roa": g("returnOnAssets"),

        # Growth
        "revenue_growth": g("revenueGrowth"),
        "earnings_growth": g("earningsGrowth"),
        "earnings_quarterly_growth": g("earningsQuarterlyGrowth"),

        # Health
        "debt_to_equity": g("debtToEquity"),
        "current_ratio": g("currentRatio"),
        "quick_ratio": g("quickRatio"),
        "total_debt": g("totalDebt"),
        "total_cash": g("totalCash"),
        "free_cashflow": g("freeCashflow"),
        "operating_cashflow": g("operatingCashflow"),

        # Dividends
        "dividend_yield": g("dividendYield"),
        "payout_ratio": g("payoutRatio"),

        # Analyst
        "target_mean_price": g("targetMeanPrice"),
        "target_median_price": g("targetMedianPrice"),
        "recommendation_mean": g("recommendationMean"),
        "number_of_analysts": g("numberOfAnalystOpinions"),

        # Meta
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "beta": g("beta"),
        "fifty_two_week_high": g("fiftyTwoWeekHigh"),
        "fifty_two_week_low": g("fiftyTwoWeekLow"),
    }

    # Derived: FCF yield = free_cashflow / market_cap
    if not np.isnan(fundamentals["free_cashflow"]) and fundamentals["market_cap"] > 0:
        fundamentals["fcf_yield"] = (
            fundamentals["free_cashflow"] / fundamentals["market_cap"]
        )
    else:
        fundamentals["fcf_yield"] = np.nan

    non_nan = sum(1 for v in fundamentals.values() if v is not np.nan and v != "")
    logger.info("Fetched {} non-empty fundamental fields for {}", non_nan, ticker)

    return fundamentals


def fetch_fundamentals_df(ticker: str) -> pd.DataFrame:
    """
    Fetch fundamentals and return as a single-row DataFrame.
    Useful for appending to a historical fundamentals table.
    """
    data = fetch_fundamentals(ticker)
    return pd.DataFrame([data])


def fetch_multiple_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fundamentals for multiple tickers, returning a DataFrame
    with one row per ticker.
    """
    rows = []
    for t in tickers:
        data = fetch_fundamentals(t)
        rows.append(data)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("ticker")
    return df


def _empty_fundamentals(ticker: str) -> dict[str, Any]:
    """Return a fundamentals dict filled with NaN for a failed fetch."""
    return {
        "ticker": ticker.upper(),
        "fetch_date": pd.Timestamp.now().isoformat(),
        "pe_ratio": np.nan,
        "forward_pe": np.nan,
        "pb_ratio": np.nan,
        "ps_ratio": np.nan,
        "ev_ebitda": np.nan,
        "peg_ratio": np.nan,
        "market_cap": np.nan,
        "enterprise_value": np.nan,
        "gross_margin": np.nan,
        "operating_margin": np.nan,
        "net_margin": np.nan,
        "roe": np.nan,
        "roa": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "earnings_quarterly_growth": np.nan,
        "debt_to_equity": np.nan,
        "current_ratio": np.nan,
        "quick_ratio": np.nan,
        "total_debt": np.nan,
        "total_cash": np.nan,
        "free_cashflow": np.nan,
        "operating_cashflow": np.nan,
        "dividend_yield": np.nan,
        "payout_ratio": np.nan,
        "target_mean_price": np.nan,
        "target_median_price": np.nan,
        "recommendation_mean": np.nan,
        "number_of_analysts": np.nan,
        "sector": "",
        "industry": "",
        "beta": np.nan,
        "fifty_two_week_high": np.nan,
        "fifty_two_week_low": np.nan,
        "fcf_yield": np.nan,
    }
