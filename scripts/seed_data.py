#!/usr/bin/env python3
"""
Seed the StockLens database with initial data.

Adds default tickers to the watchlist, fetches their price history and
fundamental snapshots, and pulls macro series from FRED.

Usage:
    python scripts/seed_data.py
    python scripts/seed_data.py --tickers AAPL MSFT GOOGL
    python scripts/seed_data.py --skip-macro    # skip FRED (if no API key)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from backend.config import get_settings
from backend.data.fetchers.fundamental_fetcher import fetch_fundamentals
from backend.data.fetchers.macro_fetcher import fetch_all_macro_series
from backend.data.fetchers.price_fetcher import fetch_price_history, fetch_ticker_info
from backend.data.storage import (
    add_to_watchlist,
    init_db,
    save_fundamental_data,
    save_macro_data,
    save_price_data,
)

import pandas as pd


def seed_watchlist(tickers: list[str]) -> None:
    """Add tickers to the watchlist with info from yfinance."""
    logger.info("Seeding watchlist with {} tickers", len(tickers))

    for ticker in tickers:
        info = fetch_ticker_info(ticker)
        add_to_watchlist(
            ticker=ticker.upper(),
            name=info.get("name", ""),
            sector=info.get("sector", ""),
            industry=info.get("industry", ""),
        )
        logger.info("  ✓ {} — {}", ticker, info.get("name", "unknown"))


def seed_prices(tickers: list[str], years: int = 5) -> None:
    """Fetch and cache price history for all tickers."""
    logger.info("Fetching {} years of price data for {} tickers", years, len(tickers))

    for ticker in tickers:
        df = fetch_price_history(ticker, years=years)
        if not df.empty:
            save_price_data(ticker, df)
            logger.info("  ✓ {} — {} rows", ticker, len(df))
        else:
            logger.warning("  ✗ {} — no data", ticker)


def seed_fundamentals(tickers: list[str]) -> None:
    """Fetch and cache fundamental snapshots for all tickers."""
    logger.info("Fetching fundamentals for {} tickers", len(tickers))

    for ticker in tickers:
        data = fetch_fundamentals(ticker)
        df = pd.DataFrame([data])
        if not df.empty:
            save_fundamental_data(ticker, df)
            logger.info("  ✓ {}", ticker)


def seed_macro() -> None:
    """Fetch and cache all configured macro series from FRED."""
    logger.info("Fetching macro series from FRED...")

    results = fetch_all_macro_series()
    for key, df in results.items():
        # Use the FRED series ID column name as the series_id for storage
        series_id = df.columns[0] if len(df.columns) == 1 else key
        save_macro_data(series_id, df)
        logger.info("  ✓ {} — {} observations", key, len(df))

    if not results:
        logger.warning("No macro data fetched — is FRED_API_KEY set in .env?")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed StockLens with initial data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers to seed (defaults to config.default_tickers)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of price history to fetch (default: 5)",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip fetching macro data from FRED",
    )
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip fetching fundamental data",
    )
    args = parser.parse_args()

    settings = get_settings()
    tickers = args.tickers or settings.default_tickers

    # Ensure DB is ready
    init_db()

    print(f"\n{'='*60}")
    print(f"  StockLens Data Seeder")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  History: {args.years} years")
    print(f"{'='*60}\n")

    # Step 1: Watchlist
    seed_watchlist(tickers)
    print()

    # Step 2: Price data
    seed_prices(tickers, years=args.years)
    print()

    # Step 3: Fundamentals
    if not args.skip_fundamentals:
        seed_fundamentals(tickers)
        print()

    # Step 4: Macro data
    if not args.skip_macro:
        seed_macro()
        print()

    print(f"\n{'='*60}")
    print(f"  ✓ Seeding complete!")
    print(f"  Database: {settings.db_path}")
    print(f"  Cache: {settings.parquet_cache_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
