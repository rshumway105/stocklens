"""
Target variable builder.

Computes the labels that the ML models will learn to predict:

1. **Forward returns** (Model 1 — Return Forecaster):
   Log returns at multiple horizons: 5d, 21d, 63d, 126d.
   These are computed at each point in time using *future* data,
   so they must ONLY be used as training targets — never as features.

2. **Smoothed price target** (Model 2 — Fair Value Estimator):
   The 63-day moving average of price serves as the "fair value" anchor.
   The model learns to predict this smoothed price from fundamentals + macro.

CRITICAL: These columns contain future information and are TARGETS ONLY.
They are clearly prefixed with 'target_' to prevent accidental use as features.
The training loop enforces this separation.
"""

import numpy as np
import pandas as pd
from backend.log import logger


def compute_forward_returns(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Compute forward log returns at multiple horizons.

    For each horizon, the forward return at time t is:
        log(price[t + horizon] / price[t])

    This uses FUTURE data — target only, never a feature.

    Args:
        df: DataFrame with a price column and DatetimeIndex.
        price_col: Column name for the closing price.

    Returns:
        DataFrame with added target columns:
        target_return_5d, target_return_21d, target_return_63d, target_return_126d
    """
    df = df.copy()
    price = df[price_col]

    horizons = {
        "5d": 5,
        "21d": 21,
        "63d": 63,
        "126d": 126,
    }

    for label, days in horizons.items():
        # shift(-days) looks into the future
        future_price = price.shift(-days)
        df[f"target_return_{label}"] = np.log(future_price / price)

    # Also add a binary direction target (up/down) for each horizon
    for label in horizons:
        df[f"target_direction_{label}"] = (
            df[f"target_return_{label}"] > 0
        ).astype(float)

    logger.info(
        "Computed forward returns at {} horizons ({} → {})",
        len(horizons),
        df.index.min().date(),
        df.index.max().date(),
    )

    return df


def compute_fair_value_target(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute the fair value target: a smoothed trailing price.

    The 63-day (3-month) moving average removes short-term noise
    and gives a "what the price should be" anchor.  The model
    learns to predict this from fundamentals and macro alone.

    Args:
        df: DataFrame with a price column.
        price_col: Column for closing price.
        window: Smoothing window in trading days (default 63 = ~3 months).

    Returns:
        DataFrame with added columns:
        - target_fair_value: smoothed price (SMA)
        - target_valuation_gap: (current - fair) / fair as percentage
    """
    df = df.copy()

    df["target_fair_value"] = df[price_col].rolling(
        window=window, min_periods=window
    ).mean()

    # Valuation gap: how far current price deviates from smoothed "fair value"
    fair = df["target_fair_value"].replace(0, np.nan)
    df["target_valuation_gap"] = (df[price_col] - fair) / fair

    logger.info(
        "Computed fair value target ({}d window)", window
    )

    return df


def compute_all_targets(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Compute all target variables for a ticker's DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        price_col: Closing price column name.

    Returns:
        DataFrame with all target columns added.
    """
    df = compute_forward_returns(df, price_col=price_col)
    df = compute_fair_value_target(df, price_col=price_col)
    return df


def get_target_columns() -> list[str]:
    """Return the list of target column names (for separation from features)."""
    return [
        "target_return_5d",
        "target_return_21d",
        "target_return_63d",
        "target_return_126d",
        "target_direction_5d",
        "target_direction_21d",
        "target_direction_63d",
        "target_direction_126d",
        "target_fair_value",
        "target_valuation_gap",
    ]


def validate_no_lookahead(
    features_df: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> bool:
    """
    Assert that no target columns leaked into the feature set.

    This is a safety check that should be called before training.
    Raises AssertionError if any target column is found in the feature DataFrame.

    Args:
        features_df: The feature matrix (should NOT contain targets).
        target_columns: List of target column names.  Uses defaults if None.

    Returns:
        True if validation passes.
    """
    if target_columns is None:
        target_columns = get_target_columns()

    leaked = set(features_df.columns) & set(target_columns)
    if leaked:
        raise AssertionError(
            f"LOOKAHEAD BIAS DETECTED: Target columns found in features: {leaked}. "
            f"These must be separated before training."
        )

    logger.debug("No lookahead bias detected — {} target columns properly excluded", len(target_columns))
    return True
