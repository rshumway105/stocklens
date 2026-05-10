"""
Watchlist API routes.

Endpoints for managing the user's tracked tickers.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.api.schemas import WatchlistAddRequest, WatchlistItem, WatchlistResponse
from backend.data.fetchers.price_fetcher import fetch_ticker_info
from backend.data.storage import add_to_watchlist, get_watchlist, remove_from_watchlist

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get("", response_model=WatchlistResponse)
async def list_watchlist():
    """Return all tickers on the watchlist."""
    df = get_watchlist()
    items = [
        WatchlistItem(
            ticker=row["ticker"],
            name=row.get("name", ""),
            sector=row.get("sector", ""),
            industry=row.get("industry", ""),
            added_at=row.get("added_at"),
        )
        for _, row in df.iterrows()
    ]
    return WatchlistResponse(tickers=items, count=len(items))


@router.post("", response_model=WatchlistItem, status_code=201)
async def add_ticker(req: WatchlistAddRequest, background_tasks: BackgroundTasks):
    """
    Add a ticker to the watchlist and kick off background training.

    Returns immediately. Training status is available at GET /api/status/{ticker}.
    """
    ticker = req.ticker.upper()
    info = fetch_ticker_info(ticker)

    if not info.get("name"):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found or has no data on Yahoo Finance.",
        )

    add_to_watchlist(
        ticker=ticker,
        name=info.get("name", ""),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
    )

    # Kick off the full data-fetch + model-training pipeline in the background
    from backend.api.training_manager import start_training, models_exist
    if not models_exist(ticker):
        background_tasks.add_task(start_training, ticker)

    return WatchlistItem(
        ticker=ticker,
        name=info.get("name", ""),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
    )


@router.delete("/{ticker}", status_code=204)
async def delete_ticker(ticker: str):
    """Remove a ticker from the watchlist."""
    remove_from_watchlist(ticker.upper())
