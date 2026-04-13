"""
Feature assembly pipeline.

This is the orchestrator that brings together all feature sources
(technical, fundamental, macro, sentiment) into a single, model-ready
feature matrix for a given ticker.

Pipeline steps:
1. Load raw data (prices, fundamentals, macro, sentiment)
2. Compute technical features from price data
3. Compute fundamental features (sector z-scores)
4. Compute macro features (level, change, direction)
5. Compute sentiment features (aggregated scores)
6. Merge everything on date index
7. Compute target variables (forward returns, fair value)
8. Validate no lookahead bias
9. Handle missing data

The output is a single DataFrame per ticker with all features + targets,
ready for the walk-forward training loop.
"""

from typing import Optional

import numpy as np
import pandas as pd
from backend.log import logger

from backend.data.processors.technical_features import compute_technical_features
from backend.data.processors.fundamental_features import (
    compute_fundamental_features,
    compute_sector_zscores,
)
from backend.data.processors.macro_features import compute_macro_features
from backend.data.processors.sentiment_features import combine_sentiment_features
from backend.data.processors.target_builder import (
    compute_all_targets,
    get_target_columns,
    validate_no_lookahead,
)


def assemble_features(
    price_df: pd.DataFrame,
    fundamentals_df: Optional[pd.DataFrame] = None,
    macro_features_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    ticker: str = "",
    compute_targets: bool = True,
    min_feature_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Assemble the complete feature matrix for a single ticker.

    Merges all data sources on the date index and produces a DataFrame
    where each row is a trading day and columns are features + targets.

    Args:
        price_df: OHLCV DataFrame (required). Must have DatetimeIndex.
        fundamentals_df: Fundamental metrics (single row or time series).
            If a single row, values are broadcast to all dates.
        macro_features_df: Pre-computed macro features (daily DatetimeIndex).
        sentiment_df: Pre-computed sentiment features (daily DatetimeIndex).
        ticker: Ticker symbol (for logging).
        compute_targets: Whether to compute target variables.
        min_feature_coverage: Drop features missing more than this fraction
            of their values (default: drop if >50% NaN).

    Returns:
        DataFrame with DatetimeIndex, all feature columns, and target columns.
    """
    logger.info("Assembling features for {} ({} price rows)", ticker, len(price_df))

    # ── Step 1: Technical features from price data ──
    try:
        df = compute_technical_features(price_df)
    except Exception as e:
        logger.error("Technical features failed for {}: {}", ticker, e)
        df = price_df.copy()

    # ── Step 2: Merge fundamental features ──
    if fundamentals_df is not None and not fundamentals_df.empty:
        df = _merge_fundamentals(df, fundamentals_df)

    # ── Step 3: Merge macro features ──
    if macro_features_df is not None and not macro_features_df.empty:
        df = _merge_macro(df, macro_features_df)

    # ── Step 4: Merge sentiment features ──
    if sentiment_df is not None and not sentiment_df.empty:
        df = _merge_sentiment(df, sentiment_df)

    # ── Step 5: Compute targets ──
    if compute_targets:
        df = compute_all_targets(df, price_col="Close")

    # ── Step 6: Clean up ──
    # Drop raw price columns that shouldn't be features
    raw_cols = ["Open", "High", "Low", "Adj Close"]
    df = df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")
    # Keep Close and Volume as they may be useful references

    # Drop features with too many missing values
    if min_feature_coverage > 0:
        target_cols = set(get_target_columns()) if compute_targets else set()
        feature_cols = [c for c in df.columns if c not in target_cols]
        coverage = df[feature_cols].notna().mean()
        low_coverage = coverage[coverage < min_feature_coverage].index.tolist()
        if low_coverage:
            logger.info(
                "Dropping {} features with <{:.0%} coverage for {}",
                len(low_coverage), min_feature_coverage, ticker,
            )
            df = df.drop(columns=low_coverage)

    # ── Step 7: Validate no lookahead ──
    if compute_targets:
        feature_only = df.drop(columns=get_target_columns(), errors="ignore")
        try:
            validate_no_lookahead(feature_only)
        except AssertionError as e:
            logger.error("Lookahead bias check failed for {}: {}", ticker, e)
            raise

    total_features = len([c for c in df.columns if c not in set(get_target_columns())])
    total_targets = len([c for c in df.columns if c in set(get_target_columns())])
    valid_rows = df.dropna(subset=["target_return_21d"]).shape[0] if compute_targets else len(df)

    logger.info(
        "Feature matrix for {}: {} features, {} targets, {} total rows, {} valid rows",
        ticker, total_features, total_targets, len(df), valid_rows,
    )

    return df


# ---------------------------------------------------------------------------
# Internal merge helpers
# ---------------------------------------------------------------------------

def _merge_fundamentals(
    df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge fundamental data into the price-based DataFrame.

    Fundamentals are point-in-time snapshots.  If we have a single snapshot,
    broadcast it across all dates.  If we have a time series of snapshots,
    forward-fill to daily.
    """
    if len(fundamentals_df) == 1:
        # Single snapshot — add as constant columns
        for col in fundamentals_df.columns:
            if col in ("ticker", "fetch_date", "sector", "industry"):
                continue
            val = fundamentals_df.iloc[0][col]
            if isinstance(val, (int, float)):
                df[f"fund_{col}"] = val
    else:
        # Time series of snapshots — resample to daily and forward-fill
        fund = fundamentals_df.copy()
        if not isinstance(fund.index, pd.DatetimeIndex):
            if "fetch_date" in fund.columns:
                fund.index = pd.to_datetime(fund["fetch_date"])
            else:
                # Can't align without dates; broadcast latest
                for col in fund.select_dtypes(include=[np.number]).columns:
                    df[f"fund_{col}"] = fund.iloc[-1][col]
                return df

        fund = fund.select_dtypes(include=[np.number])
        fund = fund.add_prefix("fund_")
        fund = fund.resample("D").last().ffill()
        df = df.join(fund, how="left")
        # Forward-fill any gaps
        fund_cols = [c for c in df.columns if c.startswith("fund_")]
        df[fund_cols] = df[fund_cols].ffill()

    return df


def _merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro features into the price DataFrame via date join.

    Macro data is daily; we left-join to keep only trading days.
    """
    macro_cols = [c for c in macro_df.columns if c not in df.columns]
    if not macro_cols:
        return df

    df = df.join(macro_df[macro_cols], how="left")

    # Forward-fill macro data (weekends/holidays)
    df[macro_cols] = df[macro_cols].ffill()

    return df


def _merge_sentiment(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment features into the price DataFrame via date join.
    """
    sent_cols = [c for c in sentiment_df.columns if c not in df.columns]
    if not sent_cols:
        return df

    df = df.join(sentiment_df[sent_cols], how="left")

    # Forward-fill (sentiment may not be available every day)
    df[sent_cols] = df[sent_cols].ffill()

    # Fill remaining NaN with 0 (no news = neutral sentiment)
    df[sent_cols] = df[sent_cols].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------

def assemble_all_tickers(
    price_data: dict[str, pd.DataFrame],
    fundamentals: Optional[dict[str, pd.DataFrame]] = None,
    macro_features: Optional[pd.DataFrame] = None,
    sentiment_data: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Assemble feature matrices for multiple tickers.

    Args:
        price_data: Dict mapping ticker -> OHLCV DataFrame.
        fundamentals: Dict mapping ticker -> fundamentals DataFrame.
        macro_features: Pre-computed macro features (shared across tickers).
        sentiment_data: Dict mapping ticker -> sentiment DataFrame.

    Returns:
        Dict mapping ticker -> complete feature matrix.
    """
    results: dict[str, pd.DataFrame] = {}

    for ticker, price_df in price_data.items():
        try:
            fund_df = fundamentals.get(ticker) if fundamentals else None
            sent_df = sentiment_data.get(ticker) if sentiment_data else None

            feature_matrix = assemble_features(
                price_df=price_df,
                fundamentals_df=fund_df,
                macro_features_df=macro_features,
                sentiment_df=sent_df,
                ticker=ticker,
            )
            results[ticker] = feature_matrix

        except Exception as e:
            logger.error("Feature assembly failed for {}: {}", ticker, e)

    logger.info("Assembled features for {}/{} tickers", len(results), len(price_data))
    return results
