"""
Pydantic schemas for the StockLens API.

Defines request/response models for all endpoints.
Using strict validation — malformed requests get clear error messages.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

class TickerBase(BaseModel):
    """Base model for a ticker on the watchlist."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock/ETF symbol")


class WatchlistItem(TickerBase):
    """Full watchlist entry returned by the API."""
    name: str = ""
    sector: str = ""
    industry: str = ""
    added_at: Optional[str] = None


class WatchlistAddRequest(TickerBase):
    """Request body when adding a ticker to the watchlist."""
    pass


class WatchlistResponse(BaseModel):
    """Response containing the full watchlist."""
    tickers: list[WatchlistItem]
    count: int


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

class PriceBar(BaseModel):
    """Single OHLCV bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float] = None
    volume: int


class PriceHistoryResponse(BaseModel):
    """Response for price history queries."""
    ticker: str
    bars: list[PriceBar]
    first_date: str
    last_date: str
    count: int


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

class MacroSeriesMeta(BaseModel):
    """Metadata for a single macro series."""
    key: str
    series_id: str
    description: str


class MacroDataPoint(BaseModel):
    """Single observation in a macro time series."""
    date: str
    value: float


class MacroSeriesResponse(BaseModel):
    """Response for a macro series query."""
    key: str
    series_id: str
    description: str
    data: list[MacroDataPoint]
    count: int


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

class FundamentalsResponse(BaseModel):
    """Fundamental metrics snapshot for a ticker."""
    ticker: str
    fetch_date: str
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    peg_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    fcf_yield: Optional[float] = None
    sector: str = ""
    industry: str = ""


# ---------------------------------------------------------------------------
# Predictions (placeholder for Phase 3+)
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Prediction output for a single ticker/horizon."""
    ticker: str
    prediction_date: str
    horizon: str
    predicted_return: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    fair_value: Optional[float] = None
    valuation_gap: Optional[float] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
