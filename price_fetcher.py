"""
Price data fetcher — wraps yfinance to download OHLCV history for stocks and ETFs.

Key design decisions:
- Always returns a DataFrame with a DatetimeIndex named 'Date'.
- Columns are standardized to: Open, High, Low, Close, Adj Close, Volume.
- Handles yfinance quirks (MultiIndex columns for single tickers, timezone-aware
  indices) so downstream code never has to worry about them.
- Fetches ticker info (name, sector, industry) for watchlist enrichment.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from backend.config import get_settings


def fetch_price_history(
    ticker: str,
    years: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker.

    Args:
        ticker: Stock or ETF symbol (e.g. "AAPL", "SPY").
        years: Number of years of history to fetch. Ignored if `start` is set.
        start: Start date string (YYYY-MM-DD). Overrides `years`.
        end: End date string (YYYY-MM-DD). Defaults to today.

    Returns:
        DataFrame indexed by Date with columns:
        [Open, High, Low, Close, Adj Close, Volume].
        Empty DataFrame on failure.
    """
    settings = get_settings()
    if start is None:
        lookback = years or settings.price_history_years
        start = (datetime.now() - timedelta(days=lookback * 365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    logger.info("Fetching price data for {} from {} to {}", ticker, start, end)

    try:
        yf_ticker = yf.Ticker(ticker)
        df: pd.DataFrame = yf_ticker.history(start=start, end=end, auto_adjust=False)

        if df.empty:
            logger.warning("No price data returned for {}", ticker)
            return pd.DataFrame()

        # yfinance sometimes returns timezone-aware index — normalize to tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "Date"

        # Keep only the columns we care about (yfinance may add Dividends, Stock Splits)
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        available = [c for c in expected_cols if c in df.columns]
        df = df[available]

        # Drop rows where Close is NaN (delistings, holidays)
        df = df.dropna(subset=["Close"])

        logger.info(
            "Fetched {} rows for {} ({} → {})",
            len(df), ticker,
            df.index.min().date(), df.index.max().date(),
        )
        return df

    except Exception as e:
        logger.error("Failed to fetch price data for {}: {}", ticker, e)
        return pd.DataFrame()


def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch basic info about a ticker (name, sector, industry, market cap, etc.).

    Returns a flat dictionary.  Missing fields default to empty strings.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "quote_type": info.get("quoteType", ""),
        }
    except Exception as e:
        logger.error("Failed to fetch info for {}: {}", ticker, e)
        return {"ticker": ticker.upper(), "name": "", "sector": "", "industry": ""}


def fetch_multiple_tickers(
    tickers: list[str],
    years: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper: fetch price data for a list of tickers.

    Returns a dict mapping ticker -> DataFrame.
    Tickers that fail are logged and excluded from the result.
    """
    results: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = fetch_price_history(t, years=years)
        if not df.empty:
            results[t.upper()] = df
    logger.info("Successfully fetched {}/{} tickers", len(results), len(tickers))
    return results
