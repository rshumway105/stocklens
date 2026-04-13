"""
Predictions & reports API routes.

Phase 1 stub — serves cached price data and fundamentals.
Full prediction endpoints will be wired up in Phase 3.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import (
    FundamentalsResponse,
    PriceBar,
    PriceHistoryResponse,
)
from backend.data.fetchers.fundamental_fetcher import fetch_fundamentals
from backend.data.fetchers.price_fetcher import fetch_price_history
from backend.data.storage import load_price_data, save_price_data

router = APIRouter(tags=["data"])


@router.get("/prices/{ticker}", response_model=PriceHistoryResponse)
async def get_price_history(
    ticker: str,
    years: int = Query(5, ge=1, le=20, description="Years of history"),
    refresh: bool = Query(False, description="Force re-fetch from Yahoo Finance"),
):
    """
    Get OHLCV price history for a ticker.

    Returns cached data if available, or fetches from yfinance.
    """
    ticker = ticker.upper()

    df = None if refresh else load_price_data(ticker)

    if df is None:
        df = fetch_price_history(ticker, years=years)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")
        save_price_data(ticker, df)

    bars = [
        PriceBar(
            date=str(idx.date()),
            open=round(row.get("Open", 0), 4),
            high=round(row.get("High", 0), 4),
            low=round(row.get("Low", 0), 4),
            close=round(row.get("Close", 0), 4),
            adj_close=round(row["Adj Close"], 4) if "Adj Close" in row and not np.isnan(row["Adj Close"]) else None,
            volume=int(row.get("Volume", 0)),
        )
        for idx, row in df.iterrows()
    ]

    return PriceHistoryResponse(
        ticker=ticker,
        bars=bars,
        first_date=str(df.index.min().date()),
        last_date=str(df.index.max().date()),
        count=len(bars),
    )


@router.get("/fundamentals/{ticker}", response_model=FundamentalsResponse)
async def get_fundamentals(ticker: str):
    """
    Get a current snapshot of fundamental metrics for a ticker.

    Fetches live from yfinance (not cached, since fundamentals change quarterly).
    """
    ticker = ticker.upper()
    data = fetch_fundamentals(ticker)

    if not data.get("ticker"):
        raise HTTPException(status_code=404, detail=f"No fundamental data for {ticker}")

    # Convert NaN to None for JSON serialization
    def clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    return FundamentalsResponse(
        ticker=data["ticker"],
        fetch_date=data["fetch_date"],
        pe_ratio=clean(data.get("pe_ratio")),
        forward_pe=clean(data.get("forward_pe")),
        pb_ratio=clean(data.get("pb_ratio")),
        ps_ratio=clean(data.get("ps_ratio")),
        ev_ebitda=clean(data.get("ev_ebitda")),
        peg_ratio=clean(data.get("peg_ratio")),
        market_cap=clean(data.get("market_cap")),
        gross_margin=clean(data.get("gross_margin")),
        operating_margin=clean(data.get("operating_margin")),
        net_margin=clean(data.get("net_margin")),
        roe=clean(data.get("roe")),
        roa=clean(data.get("roa")),
        revenue_growth=clean(data.get("revenue_growth")),
        earnings_growth=clean(data.get("earnings_growth")),
        debt_to_equity=clean(data.get("debt_to_equity")),
        current_ratio=clean(data.get("current_ratio")),
        fcf_yield=clean(data.get("fcf_yield")),
        sector=data.get("sector", ""),
        industry=data.get("industry", ""),
    )
