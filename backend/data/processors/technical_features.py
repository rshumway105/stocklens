"""
Technical feature engineering pipeline.

Transforms raw OHLCV price data into model-ready technical features.
All features use only data available at prediction time (no lookahead bias).

Feature groups:
- Moving averages (SMA, EMA at multiple periods)
- Momentum indicators (RSI, MACD, Stochastic, Williams %R)
- Volatility measures (Bollinger Bands, ATR, historical vol)
- Volume signals (OBV, volume MA ratios)
- Price patterns (multi-horizon returns, drawdown, price vs SMA ratios)

Dependencies: pandas, numpy, ta (technical analysis library)
"""

from typing import Optional

import numpy as np
import pandas as pd
from backend.log import logger


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """
    Add SMA and EMA at standard periods: 5, 10, 20, 50, 100, 200.

    Also adds price-to-SMA ratios, which capture how far the current
    price deviates from its trend — often more predictive than raw MAs.
    """
    periods = [5, 10, 20, 50, 100, 200]

    for p in periods:
        df[f"sma_{p}"] = df[col].rolling(window=p, min_periods=p).mean()
        df[f"ema_{p}"] = df[col].ewm(span=p, adjust=False, min_periods=p).mean()

        # Price relative to SMA — captures mean-reversion tendency
        df[f"price_to_sma_{p}"] = df[col] / df[f"sma_{p}"] - 1

    # SMA crossover signals (fast vs slow)
    df["sma_cross_20_50"] = df["sma_20"] / df["sma_50"] - 1
    df["sma_cross_50_200"] = df["sma_50"] / df["sma_200"] - 1

    return df


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators: RSI, MACD, Stochastic Oscillator, Williams %R.

    These measure the speed and magnitude of price movements to identify
    overbought/oversold conditions and trend strength.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD (12/26/9) ---
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- Stochastic Oscillator (14-period) ---
    low_14 = low.rolling(window=14, min_periods=14).min()
    high_14 = high.rolling(window=14, min_periods=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df["stochastic_k"] = 100 * (close - low_14) / denom
    df["stochastic_d"] = df["stochastic_k"].rolling(window=3, min_periods=3).mean()

    # --- Williams %R (14-period) ---
    df["williams_r"] = -100 * (high_14 - close) / denom

    return df


# ---------------------------------------------------------------------------
# Volatility Measures
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility features: Bollinger Bands, ATR, historical volatility.

    Volatility is a key input for both return prediction and risk estimation.
    Higher volatility typically means wider prediction intervals.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- Bollinger Bands (20-period, 2 std) ---
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20  # normalized width
    # Position within bands (0 = at lower, 1 = at upper)
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"] = (close - df["bb_lower"]) / bb_range

    # --- ATR (14-period) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    # ATR as % of price (normalized)
    df["atr_pct"] = df["atr_14"] / close

    # --- Historical Volatility (20-day rolling annualized) ---
    log_returns = np.log(close / close.shift(1))
    df["hvol_20"] = log_returns.rolling(window=20, min_periods=20).std() * np.sqrt(252)
    df["hvol_60"] = log_returns.rolling(window=60, min_periods=60).std() * np.sqrt(252)

    # Volatility ratio — short-term vs long-term (spikes indicate regime change)
    df["vol_ratio_20_60"] = df["hvol_20"] / df["hvol_60"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Volume Features
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features: OBV, volume MA ratios.

    Volume confirms price moves — a breakout on high volume is more
    meaningful than one on low volume.
    """
    close = df["Close"]
    volume = df["Volume"]

    # --- On-Balance Volume (OBV) ---
    direction = np.sign(close.diff())
    df["obv"] = (volume * direction).cumsum()
    # OBV rate of change (20-day) — more useful than raw OBV level
    df["obv_roc_20"] = df["obv"].pct_change(periods=20)

    # --- Volume moving average ratios ---
    vol_ma_20 = volume.rolling(window=20, min_periods=20).mean()
    vol_ma_50 = volume.rolling(window=50, min_periods=50).mean()

    df["vol_ratio_20"] = volume / vol_ma_20.replace(0, np.nan)  # today vs 20-day avg
    df["vol_ratio_50"] = volume / vol_ma_50.replace(0, np.nan)  # today vs 50-day avg
    df["vol_trend"] = vol_ma_20 / vol_ma_50.replace(0, np.nan)  # short vs long vol trend

    return df


# ---------------------------------------------------------------------------
# Price Pattern Features
# ---------------------------------------------------------------------------

def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return-based and pattern features:
    - Multi-horizon returns (1d through 252d lookback)
    - Drawdown from 52-week high
    - Distance from 52-week low

    These capture momentum, mean-reversion, and drawdown risk.
    """
    close = df["Close"]

    # --- Multi-horizon lookback returns (log returns) ---
    for period, label in [
        (1, "1d"), (5, "5d"), (21, "21d"), (63, "63d"), (126, "126d"), (252, "252d")
    ]:
        df[f"return_{label}"] = np.log(close / close.shift(period))

    # --- Drawdown from rolling 52-week (252-day) high ---
    rolling_high_252 = close.rolling(window=252, min_periods=63).max()
    df["drawdown_from_high"] = (close - rolling_high_252) / rolling_high_252

    # --- Distance from rolling 52-week low ---
    rolling_low_252 = close.rolling(window=252, min_periods=63).min()
    df["distance_from_low"] = (close - rolling_low_252) / rolling_low_252.replace(0, np.nan)

    # --- Intraday range ---
    df["intraday_range"] = (df["High"] - df["Low"]) / close

    # --- Gap (open vs previous close) ---
    df["gap"] = (df["Open"] - close.shift(1)) / close.shift(1)

    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def compute_technical_features(
    df: pd.DataFrame,
    drop_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Run the full technical feature pipeline on an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and columns:
            [Open, High, Low, Close, Volume] (Adj Close optional).
        drop_intermediate: If True, drop raw MA columns (keep only ratios
            and indicators) to reduce feature count.

    Returns:
        DataFrame with all original columns plus computed features.
        Early rows will have NaN where insufficient lookback data exists.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Computing technical features for {} rows", len(df))

    df = df.copy()  # avoid mutating the input

    df = add_moving_averages(df)
    df = add_momentum_indicators(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_patterns(df)

    if drop_intermediate:
        # Drop raw SMA/EMA columns (keep price_to_sma ratios and crossovers)
        ma_cols = [c for c in df.columns if c.startswith(("sma_", "ema_")) and "cross" not in c and "to_sma" not in c]
        # Also drop raw Bollinger Band levels (keep width and position)
        bb_cols = ["bb_upper", "bb_lower"]
        drop_cols = [c for c in ma_cols + bb_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

    feature_cols = [c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}]
    logger.info("Generated {} technical features", len(feature_cols))

    return df


def get_technical_feature_names() -> list[str]:
    """
    Return the list of technical feature column names (for documentation).

    Generates a dummy DataFrame to extract the names — only call this
    for introspection, not in the hot path.
    """
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(0)
    dummy = pd.DataFrame({
        "Open": 100 + np.cumsum(np.random.randn(300) * 0.5),
        "High": 101 + np.cumsum(np.random.randn(300) * 0.5),
        "Low": 99 + np.cumsum(np.random.randn(300) * 0.5),
        "Close": 100 + np.cumsum(np.random.randn(300) * 0.5),
        "Volume": np.random.randint(1e6, 5e7, 300),
    }, index=dates)
    # Make High >= Low
    dummy["High"] = dummy[["High", "Low"]].max(axis=1) + 0.5
    result = compute_technical_features(dummy)
    base_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    return [c for c in result.columns if c not in base_cols]
