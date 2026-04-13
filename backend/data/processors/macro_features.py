"""
Macroeconomic feature engineering pipeline.

Transforms raw FRED time-series data into model-ready features.
For each macro indicator, we compute:
- Level: the current value (forward-filled to daily frequency)
- Change: 1-month and 3-month changes
- Direction: whether the indicator is rising or falling (sign of change)
- Regime: Z-score relative to its own history (is this value unusual?)

We also compute derived cross-series features:
- Yield curve slope and inversions
- Real rates (nominal - inflation)
- Financial conditions composite

All macro features are date-indexed and designed to be merged with
per-ticker price data via date alignment.  Macro data is typically
released with a lag, so we shift appropriately to avoid lookahead bias.
"""

from typing import Optional

import numpy as np
import pandas as pd
from backend.log import logger


# ---------------------------------------------------------------------------
# Per-series feature computation
# ---------------------------------------------------------------------------

def compute_series_features(
    series: pd.Series,
    name: str,
    publication_lag_days: int = 1,
) -> pd.DataFrame:
    """
    Compute level, change, direction, and regime features for a single
    macro time series.

    Args:
        series: Time series with DatetimeIndex (may be irregular — monthly,
                weekly, or daily).
        name: Base name for the features (e.g. "fed_funds").
        publication_lag_days: How many days after the observation date
            the data becomes publicly available.  Used to shift the series
            forward to prevent lookahead bias.  Default 1 day for daily
            series; set higher for monthly releases (e.g. 30 for GDP).

    Returns:
        DataFrame with daily DatetimeIndex and columns:
        {name}_level, {name}_chg_1m, {name}_chg_3m,
        {name}_direction, {name}_zscore_2y
    """
    # Ensure DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    # Resample to daily and forward-fill (macro data releases are irregular)
    daily = series.resample("D").last().ffill()

    # Apply publication lag to prevent lookahead bias
    # Shift forward = the value from N days ago is what you'd actually know today
    if publication_lag_days > 0:
        daily = daily.shift(publication_lag_days)

    df = pd.DataFrame(index=daily.index)

    # Level
    df[f"{name}_level"] = daily

    # Change over 1 month (~21 trading days) and 3 months (~63 trading days)
    df[f"{name}_chg_1m"] = daily - daily.shift(30)
    df[f"{name}_chg_3m"] = daily - daily.shift(90)

    # Percentage change (for series where it makes sense)
    prev_1m = daily.shift(30).replace(0, np.nan)
    prev_3m = daily.shift(90).replace(0, np.nan)
    df[f"{name}_pct_1m"] = (daily - daily.shift(30)) / prev_1m.abs()
    df[f"{name}_pct_3m"] = (daily - daily.shift(90)) / prev_3m.abs()

    # Direction: +1 rising, -1 falling, 0 flat (based on 1-month change)
    df[f"{name}_direction"] = np.sign(df[f"{name}_chg_1m"])

    # Regime z-score: how unusual is the current level vs trailing 2-year history?
    rolling_mean = daily.rolling(window=504, min_periods=252).mean()  # ~2 years
    rolling_std = daily.rolling(window=504, min_periods=252).std()
    df[f"{name}_zscore_2y"] = (
        (daily - rolling_mean) / rolling_std.replace(0, np.nan)
    ).clip(-4, 4)

    return df


# ---------------------------------------------------------------------------
# Publication lag estimates for FRED series
# ---------------------------------------------------------------------------

# Approximate days between observation date and when data is publicly available.
# This prevents lookahead bias — e.g., GDP for Q1 isn't released until late April.
PUBLICATION_LAGS: dict[str, int] = {
    "fed_funds_rate": 1,        # daily, next day
    "treasury_2y": 1,           # daily
    "treasury_10y": 1,          # daily
    "treasury_30y": 1,          # daily
    "yield_curve_10y2y": 1,     # daily
    "cpi_yoy": 15,              # monthly, ~2 week lag
    "core_cpi": 15,             # monthly, ~2 week lag
    "pce": 30,                  # monthly, ~1 month lag
    "unemployment_rate": 5,     # monthly, first Friday of next month
    "initial_claims": 5,        # weekly, Thursday release
    "nonfarm_payrolls": 5,      # monthly, first Friday
    "real_gdp": 60,             # quarterly, ~2 month lag
    "ism_manufacturing": 5,     # monthly, first business day
    "consumer_confidence": 3,   # monthly, released end of month
    "vix": 0,                   # real-time market data
    "credit_spread_baa": 1,     # daily
    "usd_index": 1,             # daily
}


# ---------------------------------------------------------------------------
# Cross-series derived features
# ---------------------------------------------------------------------------

def compute_derived_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that combine multiple macro series.

    Args:
        macro_df: DataFrame with columns from compute_series_features
                  for multiple series (all on daily frequency).

    Returns:
        DataFrame with additional derived columns appended.
    """
    df = macro_df.copy()

    # --- Yield curve features ---
    if "treasury_10y_level" in df.columns and "treasury_2y_level" in df.columns:
        df["yield_curve_slope"] = df["treasury_10y_level"] - df["treasury_2y_level"]
        df["yield_curve_inverted"] = (df["yield_curve_slope"] < 0).astype(float)

    # --- Real rates (10Y nominal minus CPI) ---
    if "treasury_10y_level" in df.columns and "cpi_yoy_level" in df.columns:
        df["real_rate_10y"] = df["treasury_10y_level"] - df["cpi_yoy_level"]

    # --- Credit conditions ---
    if "credit_spread_baa_level" in df.columns and "vix_level" in df.columns:
        # Simple financial stress indicator: both widening spreads and rising VIX = stress
        # Normalize each to z-scores, then average
        for col in ["credit_spread_baa_level", "vix_level"]:
            mean = df[col].rolling(504, min_periods=252).mean()
            std = df[col].rolling(504, min_periods=252).std().replace(0, np.nan)
            df[f"_z_{col}"] = (df[col] - mean) / std

        z_cols = [c for c in df.columns if c.startswith("_z_")]
        if z_cols:
            df["financial_stress_index"] = df[z_cols].mean(axis=1)
            df = df.drop(columns=z_cols)

    # --- Monetary policy stance ---
    if "fed_funds_rate_level" in df.columns and "cpi_yoy_level" in df.columns:
        df["real_fed_funds"] = df["fed_funds_rate_level"] - df["cpi_yoy_level"]

    logger.info("Computed derived macro features")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_macro_features(
    macro_series_dict: dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full macro feature pipeline.

    Takes a dict of raw FRED series (as returned by macro_fetcher.fetch_all_macro_series)
    and produces a single daily-frequency DataFrame with all macro features.

    Args:
        macro_series_dict: Dict mapping series key (e.g. "fed_funds_rate")
            to a DataFrame with a single value column.
        start_date: Optional start date to trim the output.
        end_date: Optional end date to trim the output.

    Returns:
        DataFrame with DatetimeIndex (daily) and all macro feature columns.
    """
    logger.info("Computing macro features from {} series", len(macro_series_dict))

    all_features: list[pd.DataFrame] = []

    for key, raw_df in macro_series_dict.items():
        if raw_df.empty:
            continue

        # Extract the single value column
        if len(raw_df.columns) == 1:
            series = raw_df.iloc[:, 0]
        else:
            # If multiple columns, take the first numeric one
            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
            series = raw_df[numeric_cols[0]]

        lag = PUBLICATION_LAGS.get(key, 1)
        features = compute_series_features(series, name=key, publication_lag_days=lag)
        all_features.append(features)

    if not all_features:
        logger.warning("No macro features computed — all series empty")
        return pd.DataFrame()

    # Join all series on date (outer join to keep all dates)
    macro_df = pd.concat(all_features, axis=1)

    # Forward-fill gaps (weekends, holidays)
    macro_df = macro_df.ffill()

    # Compute cross-series derived features
    macro_df = compute_derived_macro_features(macro_df)

    # Trim date range if requested
    if start_date:
        macro_df = macro_df.loc[start_date:]
    if end_date:
        macro_df = macro_df.loc[:end_date]

    feature_count = len(macro_df.columns)
    logger.info(
        "Macro feature matrix: {} features × {} days",
        feature_count, len(macro_df),
    )

    return macro_df
