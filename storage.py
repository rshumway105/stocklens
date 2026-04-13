"""
Storage layer for StockLens.

Handles two storage backends:
- **SQLite**: Structured data (watchlist, predictions, model metadata, feature snapshots).
- **Parquet**: Bulk time-series data (price history, feature matrices) for fast columnar reads.

All write operations ensure parent directories exist.  All reads return
pandas DataFrames (or None / empty DataFrame on miss).
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from backend.config import get_settings


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_db_connection() -> sqlite3.Connection:
    """Open a SQLite connection, creating the DB file + parent dirs if needed."""
    settings = get_settings()
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read perf
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist.  Safe to call on every startup."""
    conn = _get_db_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        logger.info("Database initialized at {}", get_settings().db_path)
    finally:
        conn.close()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS price_history_meta (
    ticker      TEXT PRIMARY KEY,
    first_date  TEXT NOT NULL,
    last_date   TEXT NOT NULL,
    row_count   INTEGER NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS macro_series_meta (
    series_id   TEXT PRIMARY KEY,
    description TEXT,
    first_date  TEXT,
    last_date   TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    pred_date   TEXT NOT NULL,
    horizon     TEXT NOT NULL,           -- '5d', '21d', '63d', '126d'
    predicted_return REAL NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    fair_value  REAL,
    valuation_gap REAL,
    confidence  REAL,
    model_version TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, pred_date, horizon)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    version     TEXT NOT NULL,
    params      TEXT,                    -- JSON blob of hyperparameters
    metrics     TEXT,                    -- JSON blob of evaluation metrics
    trained_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def get_watchlist() -> pd.DataFrame:
    """Return all tickers on the watchlist."""
    conn = _get_db_connection()
    try:
        return pd.read_sql("SELECT * FROM watchlist ORDER BY ticker", conn)
    finally:
        conn.close()


def add_to_watchlist(
    ticker: str,
    name: str = "",
    sector: str = "",
    industry: str = "",
) -> None:
    """Add a ticker to the watchlist (upsert)."""
    conn = _get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO watchlist (ticker, name, sector, industry)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = excluded.name,
                sector = excluded.sector,
                industry = excluded.industry
            """,
            (ticker.upper(), name, sector, industry),
        )
        conn.commit()
        logger.debug("Added {} to watchlist", ticker.upper())
    finally:
        conn.close()


def remove_from_watchlist(ticker: str) -> None:
    """Remove a ticker from the watchlist."""
    conn = _get_db_connection()
    try:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Parquet cache for time-series data
# ---------------------------------------------------------------------------

def _parquet_path(category: str, key: str) -> Path:
    """Build the parquet file path: <cache_dir>/<category>/<KEY>.parquet"""
    settings = get_settings()
    path = settings.parquet_cache_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key.upper()}.parquet"


def save_price_data(ticker: str, df: pd.DataFrame) -> None:
    """Persist price history to parquet and update metadata."""
    if df.empty:
        logger.warning("Empty DataFrame for {}; skipping save", ticker)
        return

    path = _parquet_path("prices", ticker)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info("Saved {} price rows for {} → {}", len(df), ticker, path)

    # Update metadata in SQLite
    conn = _get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO price_history_meta (ticker, first_date, last_date, row_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                first_date = excluded.first_date,
                last_date = excluded.last_date,
                row_count = excluded.row_count,
                updated_at = excluded.updated_at
            """,
            (
                ticker.upper(),
                str(df.index.min().date()),
                str(df.index.max().date()),
                len(df),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_price_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached price history from parquet.  Returns None on miss."""
    path = _parquet_path("prices", ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug("Loaded {} rows for {} from cache", len(df), ticker)
    return df


def save_macro_data(series_id: str, df: pd.DataFrame) -> None:
    """Persist a FRED macro series to parquet and update metadata."""
    if df.empty:
        logger.warning("Empty DataFrame for macro series {}; skipping", series_id)
        return

    path = _parquet_path("macro", series_id)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info("Saved {} rows for macro {} → {}", len(df), series_id, path)

    conn = _get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO macro_series_meta (series_id, first_date, last_date, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                first_date = excluded.first_date,
                last_date = excluded.last_date,
                updated_at = excluded.updated_at
            """,
            (
                series_id,
                str(df.index.min().date()) if hasattr(df.index.min(), "date") else str(df.index.min()),
                str(df.index.max().date()) if hasattr(df.index.max(), "date") else str(df.index.max()),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_macro_data(series_id: str) -> Optional[pd.DataFrame]:
    """Load cached macro series from parquet.  Returns None on miss."""
    path = _parquet_path("macro", series_id)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def save_fundamental_data(ticker: str, df: pd.DataFrame) -> None:
    """Persist fundamental data snapshot to parquet."""
    if df.empty:
        return
    path = _parquet_path("fundamentals", ticker)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.info("Saved fundamental data for {} → {}", ticker, path)


def load_fundamental_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached fundamental data.  Returns None on miss."""
    path = _parquet_path("fundamentals", ticker)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")
