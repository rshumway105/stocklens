"""
Valuation reports API routes.

Serves complete valuation reports and watchlist overview data.
These endpoints orchestrate the full pipeline: data → features → models → report.
"""

from fastapi import APIRouter, HTTPException

from backend.api.schemas import ValuationReport, WatchlistOverviewItem, WatchlistOverviewResponse
from backend.log import logger

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{ticker}", response_model=ValuationReport)
async def get_valuation_report(ticker: str):
    """
    Generate a complete valuation report for a ticker.

    This is the main endpoint for the Ticker Deep Dive page.
    Runs the full pipeline: fetch data → compute features → run models → build report.

    In production, cached predictions are served unless ?refresh=true.
    Currently returns a demo report since models require xgboost to be installed.
    """
    ticker = ticker.upper()

    try:
        report = _build_demo_report(ticker)
        return report

    except Exception as e:
        logger.error("Failed to build report for {}: {}", ticker, e)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("", response_model=WatchlistOverviewResponse)
async def get_watchlist_overview():
    """
    Get the watchlist overview with valuation signals for all tracked tickers.

    This powers the main Watchlist page — a table showing each ticker's
    current price, fair value, signal, and predicted returns.
    """
    from backend.data.storage import get_watchlist

    watchlist = get_watchlist()
    items = []

    for _, row in watchlist.iterrows():
        ticker = row["ticker"]
        try:
            item = _build_demo_overview_item(ticker, row)
            items.append(item)
        except Exception as e:
            logger.warning("Failed to build overview for {}: {}", ticker, e)
            items.append(WatchlistOverviewItem(
                ticker=ticker,
                name=row.get("name", ""),
                sector=row.get("sector", ""),
            ))

    return WatchlistOverviewResponse(items=items, count=len(items))


# ---------------------------------------------------------------------------
# Demo report builders (used until models are trained with real data)
# ---------------------------------------------------------------------------

def _build_demo_report(ticker: str) -> ValuationReport:
    """
    Build a demo report with realistic structure but placeholder predictions.

    This demonstrates the full report format.  Once xgboost is installed
    and models are trained, this is replaced with real inference.
    """
    from backend.api.schemas import (
        FeatureExplanation,
        FundamentalWithZscore,
        ReturnForecast,
        RiskFlag,
        SentimentSummary,
    )

    # Try to get real price and fundamental data
    current_price = None
    name = ""
    sector = ""

    try:
        from backend.data.storage import load_price_data
        price_data = load_price_data(ticker)
        if price_data is not None and not price_data.empty:
            current_price = round(float(price_data["Close"].iloc[-1]), 2)
    except Exception:
        pass

    try:
        from backend.data.fetchers.price_fetcher import fetch_ticker_info
        info = fetch_ticker_info(ticker)
        name = info.get("name", "")
        sector = info.get("sector", "")
    except Exception:
        pass

    return ValuationReport(
        ticker=ticker,
        name=name,
        sector=sector,
        signal="fairly_valued",
        confidence=None,
        current_price=current_price,
        fair_value=None,
        valuation_gap_pct=None,
        forecasts=[
            ReturnForecast(horizon="5d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="21d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="63d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="126d", predicted_return=0.0, lower_bound=None, upper_bound=None),
        ],
        top_drivers=[],
        fundamentals=[],
        sentiment=None,
        macro_context=[],
        risk_flags=[
            RiskFlag(
                flag="demo_mode",
                severity="info",
                description="Models not yet trained. Install xgboost and run the training pipeline for real predictions.",
            )
        ],
        historical_accuracy=None,
    )


def _build_demo_overview_item(ticker: str, watchlist_row: dict) -> WatchlistOverviewItem:
    """Build a watchlist overview item with available data."""
    current_price = None

    try:
        from backend.data.storage import load_price_data
        price_data = load_price_data(ticker)
        if price_data is not None and not price_data.empty:
            current_price = round(float(price_data["Close"].iloc[-1]), 2)

            # Calculate actual price changes
            close = price_data["Close"]
            change_1d = None
            change_1m = None

            if len(close) > 1:
                change_1d = round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2)
            if len(close) > 21:
                change_1m = round(float((close.iloc[-1] / close.iloc[-22] - 1) * 100), 2)

            return WatchlistOverviewItem(
                ticker=ticker,
                name=watchlist_row.get("name", ""),
                sector=watchlist_row.get("sector", ""),
                current_price=current_price,
                price_change_1d_pct=change_1d,
                price_change_1m_pct=change_1m,
                signal="unknown",
            )
    except Exception:
        pass

    return WatchlistOverviewItem(
        ticker=ticker,
        name=watchlist_row.get("name", ""),
        sector=watchlist_row.get("sector", ""),
        current_price=current_price,
    )
