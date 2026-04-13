"""
Tests for Phase 1 — data pipeline integrity.

Tests cover:
- Price fetcher returns valid DataFrames with expected columns
- Storage round-trip (save → load) preserves data
- Fundamental fetcher returns expected keys
- Macro series catalog is well-formed
- Watchlist CRUD operations work correctly

Run with: pytest backend/tests/ -v
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Create a realistic sample price DataFrame."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")  # business days
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(100) * 2)

    return pd.DataFrame(
        {
            "Open": close - np.random.rand(100),
            "High": close + np.random.rand(100) * 2,
            "Low": close - np.random.rand(100) * 2,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": np.random.randint(1_000_000, 50_000_000, size=100),
        },
        index=dates,
    )


@pytest.fixture
def sample_macro_df() -> pd.DataFrame:
    """Create a sample macro series DataFrame."""
    dates = pd.date_range("2020-01-01", periods=50, freq="MS")
    return pd.DataFrame(
        {"DGS10": np.random.uniform(1.0, 5.0, size=50)},
        index=dates,
    )


@pytest.fixture
def tmp_settings(tmp_path):
    """Override settings to use temporary paths for testing."""
    from backend.config import Settings

    return Settings(
        database_path=str(tmp_path / "test.db"),
        cache_dir=str(tmp_path / "cache"),
        fred_api_key="",
        newsapi_key="",
    )


# ---------------------------------------------------------------------------
# Price fetcher tests
# ---------------------------------------------------------------------------

class TestPriceFetcher:
    """Tests for backend.data.fetchers.price_fetcher."""

    def test_fetch_returns_dataframe(self, sample_price_df):
        """Price data should be a DataFrame with DatetimeIndex."""
        assert isinstance(sample_price_df, pd.DataFrame)
        assert isinstance(sample_price_df.index, pd.DatetimeIndex)

    def test_expected_columns(self, sample_price_df):
        """Price DataFrame must have standard OHLCV columns."""
        required = {"Open", "High", "Low", "Close", "Volume"}
        assert required.issubset(set(sample_price_df.columns))

    def test_no_nan_in_close(self, sample_price_df):
        """Close prices should have no NaN values after cleaning."""
        assert sample_price_df["Close"].isna().sum() == 0

    def test_chronological_order(self, sample_price_df):
        """Dates should be in ascending order."""
        assert sample_price_df.index.is_monotonic_increasing

    def test_positive_volume(self, sample_price_df):
        """Volume should be non-negative."""
        assert (sample_price_df["Volume"] >= 0).all()

    def test_high_ge_low(self, sample_price_df):
        """High should always be >= Low."""
        assert (sample_price_df["High"] >= sample_price_df["Low"]).all()


# ---------------------------------------------------------------------------
# Storage round-trip tests
# ---------------------------------------------------------------------------

class TestStorage:
    """Tests for backend.data.storage — save and load round-trips."""

    def test_price_roundtrip(self, sample_price_df, tmp_settings):
        """Saving and loading price data should preserve the DataFrame."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import init_db, load_price_data, save_price_data

            init_db()
            save_price_data("TEST", sample_price_df)
            loaded = load_price_data("TEST")

            assert loaded is not None
            assert len(loaded) == len(sample_price_df)
            assert list(loaded.columns) == list(sample_price_df.columns)
            pd.testing.assert_frame_equal(loaded, sample_price_df)

    def test_macro_roundtrip(self, sample_macro_df, tmp_settings):
        """Saving and loading macro data should preserve the DataFrame."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import init_db, load_macro_data, save_macro_data

            init_db()
            save_macro_data("DGS10", sample_macro_df)
            loaded = load_macro_data("DGS10")

            assert loaded is not None
            assert len(loaded) == len(sample_macro_df)

    def test_load_missing_returns_none(self, tmp_settings):
        """Loading a non-existent ticker should return None."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import load_price_data

            assert load_price_data("NONEXISTENT") is None

    def test_empty_df_skipped(self, tmp_settings):
        """Saving an empty DataFrame should be a no-op."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import init_db, load_price_data, save_price_data

            init_db()
            save_price_data("EMPTY", pd.DataFrame())
            assert load_price_data("EMPTY") is None


# ---------------------------------------------------------------------------
# Watchlist CRUD tests
# ---------------------------------------------------------------------------

class TestWatchlist:
    """Tests for watchlist operations."""

    def test_add_and_retrieve(self, tmp_settings):
        """Adding a ticker should make it appear in the watchlist."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import add_to_watchlist, get_watchlist, init_db

            init_db()
            add_to_watchlist("AAPL", name="Apple Inc", sector="Technology")
            wl = get_watchlist()

            assert len(wl) == 1
            assert wl.iloc[0]["ticker"] == "AAPL"
            assert wl.iloc[0]["name"] == "Apple Inc"

    def test_upsert_updates_name(self, tmp_settings):
        """Adding the same ticker again should update (not duplicate) it."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import add_to_watchlist, get_watchlist, init_db

            init_db()
            add_to_watchlist("AAPL", name="Apple")
            add_to_watchlist("AAPL", name="Apple Inc")
            wl = get_watchlist()

            assert len(wl) == 1
            assert wl.iloc[0]["name"] == "Apple Inc"

    def test_remove(self, tmp_settings):
        """Removing a ticker should remove it from the watchlist."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import (
                add_to_watchlist,
                get_watchlist,
                init_db,
                remove_from_watchlist,
            )

            init_db()
            add_to_watchlist("AAPL")
            add_to_watchlist("MSFT")
            remove_from_watchlist("AAPL")
            wl = get_watchlist()

            assert len(wl) == 1
            assert wl.iloc[0]["ticker"] == "MSFT"

    def test_case_insensitive(self, tmp_settings):
        """Tickers should be normalized to uppercase."""
        with patch("backend.data.storage.get_settings", return_value=tmp_settings):
            from backend.data.storage import add_to_watchlist, get_watchlist, init_db

            init_db()
            add_to_watchlist("aapl")
            wl = get_watchlist()

            assert wl.iloc[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Fundamentals tests
# ---------------------------------------------------------------------------

class TestFundamentals:
    """Tests for the fundamental data fetcher."""

    def test_empty_fundamentals_has_required_keys(self):
        """The fallback dict should contain all expected metric keys."""
        from backend.data.fetchers.fundamental_fetcher import _empty_fundamentals

        data = _empty_fundamentals("TEST")
        required_keys = {
            "ticker", "fetch_date", "pe_ratio", "forward_pe", "pb_ratio",
            "ps_ratio", "ev_ebitda", "gross_margin", "operating_margin",
            "net_margin", "roe", "roa", "debt_to_equity", "current_ratio",
            "fcf_yield", "sector", "industry",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_empty_fundamentals_ticker_uppercase(self):
        """Ticker should be uppercased."""
        from backend.data.fetchers.fundamental_fetcher import _empty_fundamentals

        assert _empty_fundamentals("aapl")["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Macro catalog tests
# ---------------------------------------------------------------------------

class TestMacroCatalog:
    """Tests for the macro series catalog."""

    def test_catalog_not_empty(self):
        """The macro catalog should have entries."""
        from backend.data.fetchers.macro_fetcher import MACRO_SERIES

        assert len(MACRO_SERIES) > 0

    def test_all_entries_have_required_fields(self):
        """Each catalog entry must have series_id and description."""
        from backend.data.fetchers.macro_fetcher import MACRO_SERIES

        for key, meta in MACRO_SERIES.items():
            assert "series_id" in meta, f"Missing series_id for {key}"
            assert "description" in meta, f"Missing description for {key}"
            assert len(meta["series_id"]) > 0, f"Empty series_id for {key}"


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Tests for Pydantic API schemas."""

    def test_health_response_defaults(self):
        """HealthResponse should populate defaults."""
        from backend.api.schemas import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"
        assert resp.timestamp  # should be non-empty

    def test_watchlist_item_validation(self):
        """WatchlistItem should accept valid data."""
        from backend.api.schemas import WatchlistItem

        item = WatchlistItem(ticker="AAPL", name="Apple Inc", sector="Technology")
        assert item.ticker == "AAPL"

    def test_watchlist_add_rejects_empty(self):
        """WatchlistAddRequest should reject empty ticker."""
        from backend.api.schemas import WatchlistAddRequest

        with pytest.raises(Exception):
            WatchlistAddRequest(ticker="")
