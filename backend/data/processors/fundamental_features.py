"""
Fundamental feature engineering pipeline.

Transforms raw fundamental snapshots into model-ready features.
Key design: sector-relative z-scores are preferred over raw values,
because a P/E of 30 means different things in tech vs utilities.

Feature groups:
- Raw fundamental metrics (cleaned and normalized)
- Sector-relative z-scores (each metric vs sector median)
- Composite quality and value scores
- Analyst signal features

No lookahead bias: all features use point-in-time fundamental data.
"""

from typing import Optional

import numpy as np
import pandas as pd
from backend.log import logger


# ---------------------------------------------------------------------------
# Metrics used for z-score computation
# ---------------------------------------------------------------------------

# Valuation metrics (lower = cheaper, so z-score inversion may be needed for some)
VALUATION_METRICS = [
    "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda", "peg_ratio",
]

# Profitability metrics (higher = better)
PROFITABILITY_METRICS = [
    "gross_margin", "operating_margin", "net_margin", "roe", "roa",
]

# Growth metrics (higher = better)
GROWTH_METRICS = [
    "revenue_growth", "earnings_growth", "earnings_quarterly_growth",
]

# Health metrics (context-dependent)
HEALTH_METRICS = [
    "debt_to_equity", "current_ratio", "quick_ratio", "fcf_yield",
]

ALL_FUNDAMENTAL_METRICS = (
    VALUATION_METRICS + PROFITABILITY_METRICS + GROWTH_METRICS + HEALTH_METRICS
)


# ---------------------------------------------------------------------------
# Sector-relative z-scores
# ---------------------------------------------------------------------------

def compute_sector_zscores(
    fundamentals_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute sector-relative z-scores for fundamental metrics.

    For each metric, the z-score is:
        z = (value - sector_median) / sector_std

    This tells us how many standard deviations a stock's metric is
    from its sector peers — far more useful than the raw number.

    Args:
        fundamentals_df: DataFrame with one row per ticker.
            Must have a 'sector' column and metric columns.
            Index should be the ticker symbol.
        metrics: List of column names to z-score. Defaults to all known metrics.

    Returns:
        DataFrame with original columns plus z-score columns named '{metric}_zscore'.
    """
    if metrics is None:
        metrics = [m for m in ALL_FUNDAMENTAL_METRICS if m in fundamentals_df.columns]

    df = fundamentals_df.copy()

    if "sector" not in df.columns or df["sector"].isna().all():
        logger.warning("No sector data — z-scores will use entire universe as peer group")
        df["sector"] = "ALL"

    for metric in metrics:
        if metric not in df.columns:
            continue

        col = df[metric].astype(float)
        zscore_col = f"{metric}_zscore"

        # Group by sector, compute median and std
        sector_median = col.groupby(df["sector"]).transform("median")
        sector_std = col.groupby(df["sector"]).transform("std")

        # Avoid division by zero (sectors with only 1 stock)
        sector_std = sector_std.replace(0, np.nan)

        df[zscore_col] = (col - sector_median) / sector_std

        # Clip extreme z-scores to avoid outlier domination
        df[zscore_col] = df[zscore_col].clip(-4, 4)

    n_zscores = sum(1 for c in df.columns if c.endswith("_zscore"))
    logger.info("Computed {} sector-relative z-scores", n_zscores)

    return df


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------

def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite fundamental scores by averaging z-scores within
    each category.  These give a single number summarizing value,
    quality, growth, and financial health.

    Args:
        df: DataFrame with z-score columns (output of compute_sector_zscores).

    Returns:
        DataFrame with added composite score columns.
    """
    df = df.copy()

    def _avg_zscores(columns: list[str], name: str) -> None:
        """Average available z-score columns into a composite."""
        available = [f"{c}_zscore" for c in columns if f"{c}_zscore" in df.columns]
        if available:
            df[name] = df[available].mean(axis=1)
        else:
            df[name] = np.nan

    # Value score: low valuation z-scores = cheap stock
    # Invert sign because lower P/E = better value
    val_cols = [f"{m}_zscore" for m in VALUATION_METRICS if f"{m}_zscore" in df.columns]
    if val_cols:
        df["composite_value"] = -df[val_cols].mean(axis=1)  # negative = cheap = good
    else:
        df["composite_value"] = np.nan

    # Quality score: high profitability z-scores = high quality
    _avg_zscores(PROFITABILITY_METRICS, "composite_quality")

    # Growth score: high growth z-scores = fast growing
    _avg_zscores(GROWTH_METRICS, "composite_growth")

    # Health score: mix of debt (lower = better) and liquidity (higher = better)
    # Debt-to-equity is inverted (lower = healthier)
    health_cols = []
    if "debt_to_equity_zscore" in df.columns:
        df["_health_debt"] = -df["debt_to_equity_zscore"]
        health_cols.append("_health_debt")
    for m in ["current_ratio", "quick_ratio", "fcf_yield"]:
        zc = f"{m}_zscore"
        if zc in df.columns:
            health_cols.append(zc)
    if health_cols:
        df["composite_health"] = df[health_cols].mean(axis=1)
        df = df.drop(columns=["_health_debt"], errors="ignore")
    else:
        df["composite_health"] = np.nan

    # Overall fundamental score (equal-weighted blend)
    composite_cols = [
        "composite_value", "composite_quality",
        "composite_growth", "composite_health",
    ]
    available_composites = [c for c in composite_cols if c in df.columns and not df[c].isna().all()]
    if available_composites:
        df["composite_fundamental"] = df[available_composites].mean(axis=1)
    else:
        df["composite_fundamental"] = np.nan

    logger.info("Computed composite fundamental scores")
    return df


# ---------------------------------------------------------------------------
# Analyst signal features
# ---------------------------------------------------------------------------

def compute_analyst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute analyst-derived features.

    - Target price upside/downside vs current price
    - Recommendation strength (1=strong buy, 5=sell)

    Args:
        df: DataFrame with columns like 'target_mean_price', 'recommendation_mean',
            and the current price.

    Returns:
        DataFrame with added analyst feature columns.
    """
    df = df.copy()

    # Analyst target upside (requires knowing the current price)
    # This will be merged with price data in the main pipeline
    if "target_mean_price" in df.columns and "current_price" in df.columns:
        df["analyst_target_upside"] = (
            (df["target_mean_price"] - df["current_price"]) / df["current_price"]
        )
    elif "target_mean_price" in df.columns:
        # Will be filled later when merged with price data
        df["analyst_target_upside"] = np.nan

    if "target_median_price" in df.columns and "current_price" in df.columns:
        df["analyst_median_upside"] = (
            (df["target_median_price"] - df["current_price"]) / df["current_price"]
        )

    # Recommendation score (already 1-5 from yfinance; normalize to 0-1 where 1 = strong buy)
    if "recommendation_mean" in df.columns:
        df["analyst_rec_normalized"] = 1 - (df["recommendation_mean"] - 1) / 4
        df["analyst_rec_normalized"] = df["analyst_rec_normalized"].clip(0, 1)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_fundamental_features(
    fundamentals_df: pd.DataFrame,
    include_composites: bool = True,
    include_analyst: bool = True,
) -> pd.DataFrame:
    """
    Run the full fundamental feature pipeline.

    Args:
        fundamentals_df: DataFrame with one row per ticker (index = ticker).
            Must include fundamental metric columns and 'sector'.
        include_composites: Whether to compute composite scores.
        include_analyst: Whether to compute analyst features.

    Returns:
        DataFrame with all original columns plus engineered features.
    """
    logger.info("Computing fundamental features for {} tickers", len(fundamentals_df))

    df = fundamentals_df.copy()

    # Step 1: Sector-relative z-scores
    df = compute_sector_zscores(df)

    # Step 2: Composite scores
    if include_composites:
        df = compute_composite_scores(df)

    # Step 3: Analyst features
    if include_analyst:
        df = compute_analyst_features(df)

    feature_cols = [c for c in df.columns if c.endswith(("_zscore", "_normalized"))
                    or c.startswith("composite_") or c.startswith("analyst_")]
    logger.info("Generated {} fundamental features", len(feature_cols))

    return df
