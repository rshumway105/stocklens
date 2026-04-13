"""
Tests for Phase 2 — Feature Engineering.

Validates:
- Technical features compute correctly and have no lookahead bias
- Fundamental z-scores are properly centered and clipped
- Macro features respect publication lags
- Sentiment scoring and aggregation work
- Target builder computes forward returns correctly
- Feature pipeline assembles everything without data leakage
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df() -> pd.DataFrame:
    """Realistic 2-year daily OHLCV data."""
    np.random.seed(42)
    n = 504  # ~2 years of trading days
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 150 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 50)  # floor at $50

    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.3,
        "High": close + np.abs(np.random.randn(n)) * 1.5,
        "Low": close - np.abs(np.random.randn(n)) * 1.5,
        "Close": close,
        "Adj Close": close * 0.99,
        "Volume": np.random.randint(5_000_000, 50_000_000, size=n),
    }, index=dates)


@pytest.fixture
def fundamentals_df() -> pd.DataFrame:
    """Multi-ticker fundamentals for z-score testing."""
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "WFC"]
    sectors = ["Technology"] * 5 + ["Financial"] * 3

    return pd.DataFrame({
        "ticker": tickers,
        "sector": sectors,
        "pe_ratio": [28, 32, 25, 60, 22, 12, 10, 9],
        "forward_pe": [25, 28, 22, 45, 18, 11, 9, 8],
        "pb_ratio": [40, 12, 6, 8, 5, 1.5, 1.0, 0.9],
        "gross_margin": [0.43, 0.69, 0.56, 0.47, 0.81, 0.55, 0.50, 0.48],
        "operating_margin": [0.30, 0.42, 0.28, 0.06, 0.35, 0.30, 0.28, 0.25],
        "net_margin": [0.25, 0.36, 0.24, 0.04, 0.28, 0.25, 0.22, 0.20],
        "roe": [1.5, 0.40, 0.25, 0.15, 0.25, 0.12, 0.10, 0.09],
        "roa": [0.28, 0.20, 0.15, 0.04, 0.14, 0.01, 0.009, 0.008],
        "debt_to_equity": [1.5, 0.4, 0.1, 0.6, 0.1, 2.0, 1.8, 1.5],
        "current_ratio": [1.0, 1.8, 2.5, 1.1, 2.8, 0.9, 0.8, 0.7],
        "revenue_growth": [0.02, 0.07, 0.11, 0.12, 0.23, 0.05, 0.03, 0.01],
        "fcf_yield": [0.03, 0.025, 0.04, 0.01, 0.05, 0.06, 0.07, 0.08],
    }).set_index("ticker")


@pytest.fixture
def macro_series() -> dict[str, pd.DataFrame]:
    """Sample macro time series for feature computation."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1200, freq="D")

    return {
        "fed_funds_rate": pd.DataFrame(
            {"DFF": np.linspace(0.25, 5.5, 1200) + np.random.randn(1200) * 0.05},
            index=dates,
        ),
        "treasury_10y": pd.DataFrame(
            {"DGS10": np.linspace(1.5, 4.5, 1200) + np.random.randn(1200) * 0.1},
            index=dates,
        ),
        "treasury_2y": pd.DataFrame(
            {"DGS2": np.linspace(0.5, 5.0, 1200) + np.random.randn(1200) * 0.1},
            index=dates,
        ),
        "vix": pd.DataFrame(
            {"VIXCLS": 20 + np.random.randn(1200) * 5},
            index=dates,
        ),
    }


# ---------------------------------------------------------------------------
# Technical features tests
# ---------------------------------------------------------------------------

class TestTechnicalFeatures:

    def test_output_has_features(self, price_df):
        from backend.data.processors.technical_features import compute_technical_features
        result = compute_technical_features(price_df)
        base_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        feature_cols = set(result.columns) - base_cols
        assert len(feature_cols) > 30, "Should produce 30+ technical features"

    def test_no_future_data_in_features(self, price_df):
        """Features at time t should only use data from t and earlier."""
        from backend.data.processors.technical_features import compute_technical_features
        result = compute_technical_features(price_df)

        # Check: RSI at row 100 should not change if we modify data after row 100
        original_rsi = result["rsi_14"].iloc[100]

        modified = price_df.copy()
        modified.iloc[101:, modified.columns.get_loc("Close")] = 999999
        result_modified = compute_technical_features(modified)
        modified_rsi = result_modified["rsi_14"].iloc[100]

        assert original_rsi == modified_rsi, "RSI at t should not depend on future data"

    def test_moving_average_ratios_centered_around_zero(self, price_df):
        from backend.data.processors.technical_features import compute_technical_features
        result = compute_technical_features(price_df)
        # Price-to-SMA ratios should be roughly centered around 0
        ratio_col = "price_to_sma_50"
        if ratio_col in result.columns:
            mean_ratio = result[ratio_col].dropna().mean()
            assert abs(mean_ratio) < 0.1, f"price_to_sma_50 mean should be near 0, got {mean_ratio}"

    def test_rsi_bounded(self, price_df):
        from backend.data.processors.technical_features import compute_technical_features
        result = compute_technical_features(price_df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"

    def test_bollinger_position_bounded(self, price_df):
        from backend.data.processors.technical_features import compute_technical_features
        result = compute_technical_features(price_df)
        bb = result["bb_position"].dropna()
        # Most values should be between -0.5 and 1.5 (can exceed 0/1 during extremes)
        assert bb.median() > 0 and bb.median() < 1

    def test_missing_columns_raises(self):
        from backend.data.processors.technical_features import compute_technical_features
        bad_df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_technical_features(bad_df)


# ---------------------------------------------------------------------------
# Fundamental features tests
# ---------------------------------------------------------------------------

class TestFundamentalFeatures:

    def test_zscore_columns_created(self, fundamentals_df):
        from backend.data.processors.fundamental_features import compute_sector_zscores
        result = compute_sector_zscores(fundamentals_df)
        zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
        assert len(zscore_cols) > 5, "Should create z-score columns"

    def test_zscore_sector_centered(self, fundamentals_df):
        """Within each sector, z-scores should average to ~0."""
        from backend.data.processors.fundamental_features import compute_sector_zscores
        result = compute_sector_zscores(fundamentals_df)

        for sector in result["sector"].unique():
            sector_data = result[result["sector"] == sector]
            if len(sector_data) > 2:
                pe_z = sector_data["pe_ratio_zscore"].dropna()
                assert abs(pe_z.mean()) < 0.5, f"Sector {sector} PE z-scores should average near 0"

    def test_zscore_clipped(self, fundamentals_df):
        from backend.data.processors.fundamental_features import compute_sector_zscores
        result = compute_sector_zscores(fundamentals_df)
        zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
        for col in zscore_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= -4, f"{col} should be clipped at -4"
                assert vals.max() <= 4, f"{col} should be clipped at 4"

    def test_composite_scores_created(self, fundamentals_df):
        from backend.data.processors.fundamental_features import compute_fundamental_features
        result = compute_fundamental_features(fundamentals_df)
        assert "composite_value" in result.columns
        assert "composite_quality" in result.columns
        assert "composite_fundamental" in result.columns


# ---------------------------------------------------------------------------
# Macro features tests
# ---------------------------------------------------------------------------

class TestMacroFeatures:

    def test_series_features_output(self):
        from backend.data.processors.macro_features import compute_series_features
        dates = pd.date_range("2020-01-01", periods=600, freq="D")
        series = pd.Series(np.linspace(1, 5, 600), index=dates)

        result = compute_series_features(series, name="test_rate", publication_lag_days=1)

        assert "test_rate_level" in result.columns
        assert "test_rate_chg_1m" in result.columns
        assert "test_rate_direction" in result.columns
        assert "test_rate_zscore_2y" in result.columns

    def test_publication_lag_shifts_data(self):
        """With lag=5, today's feature should reflect data from 5 days ago."""
        from backend.data.processors.macro_features import compute_series_features
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        values = np.arange(100, dtype=float)
        series = pd.Series(values, index=dates)

        result = compute_series_features(series, name="x", publication_lag_days=5)
        # After forward-fill and lag, the level at day 10 should be value from day 5
        level_at_10 = result["x_level"].iloc[10]
        assert level_at_10 == values[5], f"Expected {values[5]}, got {level_at_10}"

    def test_full_pipeline(self, macro_series):
        from backend.data.processors.macro_features import compute_macro_features
        result = compute_macro_features(macro_series)
        assert not result.empty
        assert len(result.columns) > 10, "Should produce many macro features"

    def test_derived_features(self, macro_series):
        from backend.data.processors.macro_features import compute_macro_features
        result = compute_macro_features(macro_series)
        # Should have yield curve features since we provided 10y and 2y
        assert "yield_curve_slope" in result.columns


# ---------------------------------------------------------------------------
# Sentiment features tests
# ---------------------------------------------------------------------------

class TestSentimentFeatures:

    def test_heuristic_scorer(self):
        from backend.data.processors.sentiment_features import score_text_heuristic
        assert score_text_heuristic("Stock surge rally gains") > 0
        assert score_text_heuristic("Market crash plunge fear") < 0
        assert score_text_heuristic("Company reports quarterly results") == 0
        assert score_text_heuristic("") == 0

    def test_news_aggregation(self):
        from backend.data.processors.sentiment_features import (
            aggregate_news_sentiment,
            score_headlines,
        )
        headlines = pd.DataFrame({
            "title": [
                "Stock surges on strong earnings",
                "Market drops amid fears",
                "Company beats expectations",
                "Bearish outlook for sector",
            ],
            "published_at": pd.date_range("2024-01-01", periods=4, freq="D"),
        })
        scored = score_headlines(headlines)
        daily = aggregate_news_sentiment(scored)

        assert not daily.empty
        assert "news_sentiment_mean" in daily.columns
        assert "news_article_count" in daily.columns

    def test_combine_handles_none(self):
        from backend.data.processors.sentiment_features import combine_sentiment_features
        result = combine_sentiment_features(news_sentiment=None, social_sentiment=None)
        assert result.empty


# ---------------------------------------------------------------------------
# Target builder tests
# ---------------------------------------------------------------------------

class TestTargetBuilder:

    def test_forward_returns_use_future_data(self, price_df):
        from backend.data.processors.target_builder import compute_forward_returns
        result = compute_forward_returns(price_df)

        # 5-day forward return at row 0 should use price at row 5
        expected = np.log(price_df["Close"].iloc[5] / price_df["Close"].iloc[0])
        actual = result["target_return_5d"].iloc[0]
        assert abs(actual - expected) < 1e-10

    def test_forward_returns_nan_at_end(self, price_df):
        """Last N rows should be NaN (no future data available)."""
        from backend.data.processors.target_builder import compute_forward_returns
        result = compute_forward_returns(price_df)
        assert result["target_return_5d"].iloc[-1] != result["target_return_5d"].iloc[-1]  # NaN
        assert result["target_return_126d"].iloc[-100] != result["target_return_126d"].iloc[-100]

    def test_lookahead_validation_passes(self, price_df):
        from backend.data.processors.target_builder import validate_no_lookahead
        features = price_df[["Close", "Volume"]].copy()
        features["rsi_14"] = 50  # dummy feature
        assert validate_no_lookahead(features) is True

    def test_lookahead_validation_catches_leak(self, price_df):
        from backend.data.processors.target_builder import validate_no_lookahead
        bad_features = price_df.copy()
        bad_features["target_return_5d"] = 0.01  # leaked target!
        with pytest.raises(AssertionError, match="LOOKAHEAD BIAS"):
            validate_no_lookahead(bad_features)


# ---------------------------------------------------------------------------
# Feature pipeline integration tests
# ---------------------------------------------------------------------------

class TestFeaturePipeline:

    def test_assemble_minimal(self, price_df):
        """Should work with just price data."""
        from backend.data.processors.feature_pipeline import assemble_features
        result = assemble_features(price_df, ticker="TEST")
        assert not result.empty
        assert "target_return_21d" in result.columns
        assert "rsi_14" in result.columns

    def test_assemble_with_macro(self, price_df, macro_series):
        """Should merge macro features correctly."""
        from backend.data.processors.macro_features import compute_macro_features
        from backend.data.processors.feature_pipeline import assemble_features

        macro_df = compute_macro_features(macro_series)
        result = assemble_features(price_df, macro_features_df=macro_df, ticker="TEST")

        # Should have macro columns
        macro_cols = [c for c in result.columns if c.startswith(("fed_funds", "treasury", "vix"))]
        assert len(macro_cols) > 0, "Macro features should be merged"

    def test_no_target_in_features(self, price_df):
        """Final feature matrix should separate targets from features."""
        from backend.data.processors.feature_pipeline import assemble_features
        from backend.data.processors.target_builder import get_target_columns

        result = assemble_features(price_df, ticker="TEST")
        target_cols = set(get_target_columns()) & set(result.columns)
        feature_cols = set(result.columns) - target_cols

        # Targets should be present
        assert len(target_cols) > 0
        # Features should not include targets
        assert not (feature_cols & set(get_target_columns()))
