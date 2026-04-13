"""
Macroeconomic data fetcher — wraps the FRED API via fredapi.

Downloads key macro indicators used as model features:
- Interest rates (Fed Funds, Treasury yields, yield curve)
- Inflation (CPI, Core CPI, PCE)
- Employment (unemployment rate, jobless claims, nonfarm payrolls)
- Activity (GDP, ISM PMI, consumer confidence)
- Market indicators (VIX, credit spreads, USD index)

Each series is identified by its FRED series ID.  The fetcher handles
missing API keys gracefully (logs a warning and returns empty data).
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from backend.log import logger

from backend.config import get_settings


# ---------------------------------------------------------------------------
# FRED series catalog — maps human-readable names to FRED series IDs
# ---------------------------------------------------------------------------

MACRO_SERIES: dict[str, dict[str, str]] = {
    # Interest rates
    "fed_funds_rate": {
        "series_id": "DFF",
        "description": "Effective Federal Funds Rate (daily)",
    },
    "treasury_2y": {
        "series_id": "DGS2",
        "description": "2-Year Treasury Constant Maturity Rate",
    },
    "treasury_10y": {
        "series_id": "DGS10",
        "description": "10-Year Treasury Constant Maturity Rate",
    },
    "treasury_30y": {
        "series_id": "DGS30",
        "description": "30-Year Treasury Constant Maturity Rate",
    },
    "yield_curve_10y2y": {
        "series_id": "T10Y2Y",
        "description": "10-Year minus 2-Year Treasury Spread",
    },
    # Inflation
    "cpi_yoy": {
        "series_id": "CPIAUCSL",
        "description": "Consumer Price Index for All Urban Consumers (SA)",
    },
    "core_cpi": {
        "series_id": "CPILFESL",
        "description": "CPI Less Food and Energy (SA)",
    },
    "pce": {
        "series_id": "PCEPI",
        "description": "Personal Consumption Expenditures Price Index",
    },
    # Employment
    "unemployment_rate": {
        "series_id": "UNRATE",
        "description": "Civilian Unemployment Rate",
    },
    "initial_claims": {
        "series_id": "ICSA",
        "description": "Initial Jobless Claims (weekly)",
    },
    "nonfarm_payrolls": {
        "series_id": "PAYEMS",
        "description": "Total Nonfarm Payrolls (SA)",
    },
    # Activity
    "real_gdp": {
        "series_id": "GDPC1",
        "description": "Real GDP (quarterly, SAAR)",
    },
    "ism_manufacturing": {
        "series_id": "MANEMP",
        "description": "Manufacturing Employment (proxy for ISM)",
    },
    "consumer_confidence": {
        "series_id": "UMCSENT",
        "description": "University of Michigan Consumer Sentiment",
    },
    # Market-wide
    "vix": {
        "series_id": "VIXCLS",
        "description": "CBOE Volatility Index (VIX)",
    },
    "credit_spread_baa": {
        "series_id": "BAA10Y",
        "description": "Moody's BAA minus 10-Year Treasury Spread",
    },
    "usd_index": {
        "series_id": "DTWEXBGS",
        "description": "Trade Weighted US Dollar Index (Broad)",
    },
}


def _get_fred_client():
    """
    Create a fredapi.Fred client.  Returns None if the API key is missing,
    so callers can degrade gracefully.
    """
    settings = get_settings()
    if not settings.fred_api_key:
        logger.warning(
            "FRED_API_KEY not set — macro data will be unavailable. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return None

    try:
        from fredapi import Fred
        return Fred(api_key=settings.fred_api_key)
    except ImportError:
        logger.error("fredapi package not installed. Run: pip install fredapi")
        return None


def fetch_single_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch a single FRED series.

    Args:
        series_id: FRED series identifier (e.g. "DGS10").
        start: Start date (YYYY-MM-DD).  Defaults to 10 years ago.
        end: End date.  Defaults to today.

    Returns:
        DataFrame with DatetimeIndex and a single column named after the series_id.
        Empty DataFrame on failure.
    """
    fred = _get_fred_client()
    if fred is None:
        return pd.DataFrame()

    if start is None:
        start = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    try:
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        if series is None or series.empty:
            logger.warning("No data returned for FRED series {}", series_id)
            return pd.DataFrame()

        df = series.to_frame(name=series_id)
        df.index.name = "Date"
        df = df.dropna()

        logger.info(
            "Fetched {} observations for FRED:{} ({} → {})",
            len(df), series_id,
            df.index.min().date(), df.index.max().date(),
        )
        return df

    except Exception as e:
        logger.error("Failed to fetch FRED series {}: {}", series_id, e)
        return pd.DataFrame()


def fetch_all_macro_series(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all configured macro series from FRED.

    Returns a dict mapping the human-readable key (e.g. "fed_funds_rate")
    to its DataFrame.  Series that fail are excluded with a warning.
    """
    results: dict[str, pd.DataFrame] = {}

    for key, meta in MACRO_SERIES.items():
        df = fetch_single_series(meta["series_id"], start=start, end=end)
        if not df.empty:
            results[key] = df

    logger.info(
        "Fetched {}/{} macro series successfully",
        len(results), len(MACRO_SERIES),
    )
    return results


def get_macro_catalog() -> list[dict[str, str]]:
    """Return the catalog of macro series (for display in the UI)."""
    return [
        {"key": key, **meta}
        for key, meta in MACRO_SERIES.items()
    ]
