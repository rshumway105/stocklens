"""
Scheduled tasks for StockLens.

Defines the jobs that run on a schedule:
- Data refresh: update price history, fundamentals, macro data
- Model retraining: re-run the training pipeline with fresh data
- Report generation: pre-compute valuation reports for the watchlist

These tasks are designed to be called by the scheduler (scheduler.py)
or run manually via scripts.
"""

from datetime import datetime
from typing import Optional

from backend.log import logger


def refresh_price_data(tickers: Optional[list[str]] = None) -> dict:
    """
    Refresh price data for watchlist tickers.

    Fetches the latest OHLCV data and updates the parquet cache.
    Returns a summary dict with success/failure counts.
    """
    from backend.data.fetchers.price_fetcher import fetch_price_history
    from backend.data.storage import get_watchlist, save_price_data

    if tickers is None:
        watchlist = get_watchlist()
        tickers = watchlist["ticker"].tolist()

    logger.info("Refreshing price data for {} tickers", len(tickers))
    results = {"success": 0, "failed": 0, "tickers": []}

    for ticker in tickers:
        try:
            df = fetch_price_history(ticker)
            if not df.empty:
                save_price_data(ticker, df)
                results["success"] += 1
                results["tickers"].append({"ticker": ticker, "status": "ok", "rows": len(df)})
            else:
                results["failed"] += 1
                results["tickers"].append({"ticker": ticker, "status": "empty"})
        except Exception as e:
            results["failed"] += 1
            results["tickers"].append({"ticker": ticker, "status": "error", "error": str(e)})
            logger.error("Price refresh failed for {}: {}", ticker, e)

    logger.info("Price refresh complete: {}/{} succeeded", results["success"], len(tickers))
    return results


def refresh_fundamentals(tickers: Optional[list[str]] = None) -> dict:
    """Refresh fundamental data for watchlist tickers."""
    from backend.data.fetchers.fundamental_fetcher import fetch_fundamentals
    from backend.data.storage import get_watchlist, save_fundamental_data
    import pandas as pd

    if tickers is None:
        watchlist = get_watchlist()
        tickers = watchlist["ticker"].tolist()

    logger.info("Refreshing fundamentals for {} tickers", len(tickers))
    results = {"success": 0, "failed": 0}

    for ticker in tickers:
        try:
            data = fetch_fundamentals(ticker)
            df = pd.DataFrame([data])
            save_fundamental_data(ticker, df)
            results["success"] += 1
        except Exception as e:
            results["failed"] += 1
            logger.error("Fundamentals refresh failed for {}: {}", ticker, e)

    return results


def refresh_macro_data() -> dict:
    """Refresh all macro series from FRED."""
    from backend.data.fetchers.macro_fetcher import fetch_all_macro_series
    from backend.data.storage import save_macro_data

    logger.info("Refreshing macro data from FRED")
    results = {"success": 0, "failed": 0}

    try:
        series_dict = fetch_all_macro_series()
        for key, df in series_dict.items():
            series_id = df.columns[0] if len(df.columns) == 1 else key
            save_macro_data(series_id, df)
            results["success"] += 1

        logger.info("Macro refresh complete: {} series updated", results["success"])
    except Exception as e:
        results["failed"] += 1
        logger.error("Macro refresh failed: {}", e)

    return results


def refresh_all_data(tickers: Optional[list[str]] = None) -> dict:
    """Run all data refresh tasks."""
    logger.info("Starting full data refresh at {}", datetime.utcnow().isoformat())

    price_result = refresh_price_data(tickers)
    fund_result = refresh_fundamentals(tickers)
    macro_result = refresh_macro_data()

    summary = {
        "prices": price_result,
        "fundamentals": fund_result,
        "macro": macro_result,
        "completed_at": datetime.utcnow().isoformat(),
    }

    logger.info("Full data refresh complete")
    return summary
