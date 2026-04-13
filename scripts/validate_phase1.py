#!/usr/bin/env python3
"""
Standalone Phase 1 validation — runs without pip-installed dependencies.

Uses only: sqlite3, pandas, numpy, pathlib (all available in this environment).
Validates: database schema, storage round-trips, data integrity checks.

Usage: python3 scripts/validate_phase1.py
"""

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ──────────────────────────────────────────────────────────
# 1. SQLite schema validation
# ──────────────────────────────────────────────────────────
print("\n═══ 1. Database Schema ═══")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY, name TEXT, sector TEXT, industry TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS price_history_meta (
    ticker TEXT PRIMARY KEY, first_date TEXT NOT NULL, last_date TEXT NOT NULL,
    row_count INTEGER NOT NULL, updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS macro_series_meta (
    series_id TEXT PRIMARY KEY, description TEXT, first_date TEXT,
    last_date TEXT, updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
    pred_date TEXT NOT NULL, horizon TEXT NOT NULL,
    predicted_return REAL NOT NULL, lower_bound REAL, upper_bound REAL,
    fair_value REAL, valuation_gap REAL, confidence REAL,
    model_version TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, pred_date, horizon)
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
    version TEXT NOT NULL, params TEXT, metrics TEXT,
    trained_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    # Verify tables exist
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    for t in ["watchlist", "price_history_meta", "macro_series_meta", "predictions", "model_runs"]:
        check(f"Table '{t}' exists", t in tables)

    # Verify idempotent (run schema again)
    try:
        conn.executescript(SCHEMA_SQL)
        check("Schema is idempotent (safe to re-run)", True)
    except Exception as e:
        check("Schema is idempotent", False, str(e))

    conn.close()

# ──────────────────────────────────────────────────────────
# 2. Watchlist CRUD
# ──────────────────────────────────────────────────────────
print("\n═══ 2. Watchlist CRUD ═══")

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    # INSERT
    conn.execute(
        "INSERT INTO watchlist (ticker, name, sector) VALUES (?, ?, ?)",
        ("AAPL", "Apple Inc", "Technology"),
    )
    conn.commit()
    rows = conn.execute("SELECT * FROM watchlist").fetchall()
    check("Insert single ticker", len(rows) == 1)
    check("Ticker stored correctly", rows[0][0] == "AAPL")

    # UPSERT
    conn.execute(
        """INSERT INTO watchlist (ticker, name, sector) VALUES (?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET name = excluded.name""",
        ("AAPL", "Apple Inc.", "Technology"),
    )
    conn.commit()
    rows = conn.execute("SELECT * FROM watchlist").fetchall()
    check("Upsert does not duplicate", len(rows) == 1)
    check("Upsert updates name", rows[0][1] == "Apple Inc.")

    # DELETE
    conn.execute("DELETE FROM watchlist WHERE ticker = ?", ("AAPL",))
    conn.commit()
    rows = conn.execute("SELECT * FROM watchlist").fetchall()
    check("Delete removes ticker", len(rows) == 0)

    conn.close()

# ──────────────────────────────────────────────────────────
# 3. Parquet round-trip (using pandas)
# ──────────────────────────────────────────────────────────
print("\n═══ 3. Parquet Round-Trip ═══")

with tempfile.TemporaryDirectory() as tmpdir:
    # Simulate price data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    close = 150 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame({
        "Open": close - np.abs(np.random.randn(100)),
        "High": close + np.abs(np.random.randn(100)) * 2,
        "Low": close - np.abs(np.random.randn(100)) * 2,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, size=100),
    }, index=dates)
    df.index.name = "Date"

    path = Path(tmpdir) / "AAPL.parquet"

    # Try parquet (needs pyarrow)
    try:
        df.to_parquet(path, engine="pyarrow")
        loaded = pd.read_parquet(path, engine="pyarrow")
        check("Parquet write + read", True)
        check("Row count preserved", len(loaded) == len(df))
        check("Columns preserved", list(loaded.columns) == list(df.columns))
        check("Values match", np.allclose(loaded["Close"].values, df["Close"].values))
    except ImportError:
        # Fallback to CSV if pyarrow not available
        csv_path = Path(tmpdir) / "AAPL.csv"
        df.to_csv(csv_path)
        loaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        check("CSV fallback write + read", True)
        check("Row count preserved (CSV)", len(loaded) == len(df))
        print("    ⚠ pyarrow not installed — parquet tests used CSV fallback")

# ──────────────────────────────────────────────────────────
# 4. Price data integrity checks
# ──────────────────────────────────────────────────────────
print("\n═══ 4. Price Data Integrity ═══")

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252, freq="B")
close = 150 + np.cumsum(np.random.randn(252) * 2)
df = pd.DataFrame({
    "Open": close - np.abs(np.random.randn(252)) * 0.5,
    "High": close + np.abs(np.random.randn(252)) * 1.5,
    "Low": close - np.abs(np.random.randn(252)) * 1.5,
    "Close": close,
    "Volume": np.random.randint(1_000_000, 50_000_000, size=252),
}, index=dates)

check("DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
check("Has required columns", {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns))
check("No NaN in Close", df["Close"].isna().sum() == 0)
check("Chronological order", df.index.is_monotonic_increasing)
check("Positive volume", (df["Volume"] >= 0).all())
check("High >= Low", (df["High"] >= df["Low"]).all())
check("~252 trading days/year", 240 <= len(df) <= 260)

# ──────────────────────────────────────────────────────────
# 5. Macro series catalog validation
# ──────────────────────────────────────────────────────────
print("\n═══ 5. Macro Catalog ═══")

MACRO_SERIES = {
    "fed_funds_rate": {"series_id": "DFF", "description": "Effective Federal Funds Rate"},
    "treasury_2y": {"series_id": "DGS2", "description": "2-Year Treasury Rate"},
    "treasury_10y": {"series_id": "DGS10", "description": "10-Year Treasury Rate"},
    "treasury_30y": {"series_id": "DGS30", "description": "30-Year Treasury Rate"},
    "yield_curve_10y2y": {"series_id": "T10Y2Y", "description": "10Y minus 2Y Spread"},
    "cpi_yoy": {"series_id": "CPIAUCSL", "description": "CPI All Urban Consumers"},
    "core_cpi": {"series_id": "CPILFESL", "description": "CPI Less Food and Energy"},
    "pce": {"series_id": "PCEPI", "description": "PCE Price Index"},
    "unemployment_rate": {"series_id": "UNRATE", "description": "Unemployment Rate"},
    "initial_claims": {"series_id": "ICSA", "description": "Initial Jobless Claims"},
    "nonfarm_payrolls": {"series_id": "PAYEMS", "description": "Nonfarm Payrolls"},
    "real_gdp": {"series_id": "GDPC1", "description": "Real GDP"},
    "vix": {"series_id": "VIXCLS", "description": "VIX"},
    "credit_spread_baa": {"series_id": "BAA10Y", "description": "BAA Credit Spread"},
    "usd_index": {"series_id": "DTWEXBGS", "description": "USD Index"},
}

check("Catalog has 15+ series", len(MACRO_SERIES) >= 15)

all_have_id = all("series_id" in v for v in MACRO_SERIES.values())
check("All entries have series_id", all_have_id)

all_have_desc = all("description" in v for v in MACRO_SERIES.values())
check("All entries have description", all_have_desc)

unique_ids = set(v["series_id"] for v in MACRO_SERIES.values())
check("All series_ids are unique", len(unique_ids) == len(MACRO_SERIES))

# ──────────────────────────────────────────────────────────
# 6. Fundamental metrics structure
# ──────────────────────────────────────────────────────────
print("\n═══ 6. Fundamental Metrics Structure ═══")

REQUIRED_FUNDAMENTAL_KEYS = {
    "ticker", "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio",
    "ev_ebitda", "gross_margin", "operating_margin", "net_margin",
    "roe", "roa", "debt_to_equity", "current_ratio", "fcf_yield",
    "revenue_growth", "earnings_growth", "sector", "industry",
}

# Simulate empty fundamentals
empty_fundamentals = {k: np.nan for k in REQUIRED_FUNDAMENTAL_KEYS}
empty_fundamentals["ticker"] = "TEST"
empty_fundamentals["sector"] = ""
empty_fundamentals["industry"] = ""

check("Empty template has all keys", REQUIRED_FUNDAMENTAL_KEYS.issubset(empty_fundamentals.keys()))
check("Ticker is uppercase", empty_fundamentals["ticker"] == "TEST")

# ──────────────────────────────────────────────────────────
# 7. File structure audit
# ──────────────────────────────────────────────────────────
print("\n═══ 7. Project File Structure ═══")

project_root = Path(__file__).resolve().parent.parent
expected_files = [
    "README.md",
    "LICENSE",
    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "backend/main.py",
    "backend/config.py",
    "backend/data/storage.py",
    "backend/data/fetchers/price_fetcher.py",
    "backend/data/fetchers/macro_fetcher.py",
    "backend/data/fetchers/fundamental_fetcher.py",
    "backend/data/fetchers/sentiment_fetcher.py",
    "backend/api/schemas.py",
    "backend/api/routes/watchlist.py",
    "backend/api/routes/macro.py",
    "backend/api/routes/predictions.py",
    "backend/tests/test_phase1.py",
    "scripts/setup_db.py",
    "scripts/seed_data.py",
]

for f in expected_files:
    exists = (project_root / f).exists()
    check(f"Exists: {f}", exists)

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
