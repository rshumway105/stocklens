"""
Watchlist API routes.

Endpoints for managing the user's tracked tickers.
"""

from fastapi import APIRouter, HTTPException

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
async def add_ticker(req: WatchlistAddRequest):
    """
    Add a ticker to the watchlist.

    Fetches basic info (name, sector) from yfinance before saving.
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
