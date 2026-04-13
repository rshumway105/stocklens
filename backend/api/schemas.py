"""
Pydantic schemas for the StockLens API.

Defines request/response models for all endpoints.
Using strict validation — malformed requests get clear error messages.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

class TickerBase(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock/ETF symbol")


class WatchlistItem(TickerBase):
    name: str = ""
    sector: str = ""
    industry: str = ""
    added_at: Optional[str] = None


class WatchlistAddRequest(TickerBase):
    pass


class WatchlistResponse(BaseModel):
    tickers: list[WatchlistItem]
    count: int


class WatchlistOverviewItem(BaseModel):
    """Single row in the watchlist overview table."""
    ticker: str
    name: str = ""
    sector: str = ""
    current_price: Optional[float] = None
    fair_value: Optional[float] = None
    valuation_gap_pct: Optional[float] = None
    signal: str = "unknown"
    confidence: Optional[float] = None
    predicted_return_1m: Optional[float] = None
    price_change_1d_pct: Optional[float] = None
    price_change_1m_pct: Optional[float] = None


class WatchlistOverviewResponse(BaseModel):
    items: list[WatchlistOverviewItem]
    count: int
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float] = None
    volume: int


class PriceHistoryResponse(BaseModel):
    ticker: str
    bars: list[PriceBar]
    first_date: str
    last_date: str
    count: int


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

class MacroSeriesMeta(BaseModel):
    key: str
    series_id: str
    description: str


class MacroDataPoint(BaseModel):
    date: str
    value: float


class MacroSeriesResponse(BaseModel):
    key: str
    series_id: str
    description: str
    data: list[MacroDataPoint]
    count: int


class MacroDashboardItem(BaseModel):
    key: str
    name: str
    current_value: Optional[float] = None
    change_1m: Optional[float] = None
    change_3m: Optional[float] = None
    direction: str = "flat"
    zscore: Optional[float] = None
    description: str = ""


class MacroDashboardResponse(BaseModel):
    indicators: list[MacroDashboardItem]
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

class FundamentalsResponse(BaseModel):
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


class FundamentalWithZscore(BaseModel):
    metric: str
    value: Optional[float] = None
    zscore: Optional[float] = None
    sector_median: Optional[float] = None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class ReturnForecast(BaseModel):
    horizon: str
    predicted_return: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    forecasts: list[ReturnForecast]
    fair_value: Optional[float] = None
    valuation_gap: Optional[float] = None
    signal: str = "unknown"
    confidence: Optional[float] = None
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature Explanation (SHAP)
# ---------------------------------------------------------------------------

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: Optional[float] = None
    importance: Optional[float] = None
    feature_value: Optional[float] = None
    direction: str = "unknown"
    explanation: str = ""


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class SentimentSummary(BaseModel):
    news_sentiment: Optional[float] = None
    news_trend: str = "neutral"
    social_sentiment: Optional[float] = None
    social_mention_volume: Optional[int] = None
    combined_sentiment: Optional[float] = None


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

class RiskFlag(BaseModel):
    flag: str
    severity: str = "info"
    description: str = ""


# ---------------------------------------------------------------------------
# Valuation Report
# ---------------------------------------------------------------------------

class ValuationReport(BaseModel):
    """Complete valuation report for a single ticker."""
    ticker: str
    name: str = ""
    sector: str = ""
    report_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    signal: str = "unknown"
    confidence: Optional[float] = None

    current_price: Optional[float] = None
    fair_value: Optional[float] = None
    fair_value_lower: Optional[float] = None
    fair_value_upper: Optional[float] = None
    valuation_gap_pct: Optional[float] = None

    forecasts: list[ReturnForecast] = []
    top_drivers: list[FeatureExplanation] = []
    fundamentals: list[FundamentalWithZscore] = []
    sentiment: Optional[SentimentSummary] = None
    macro_context: list[dict[str, Any]] = []
    risk_flags: list[RiskFlag] = []
    historical_accuracy: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestMetrics(BaseModel):
    horizon: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    direction_accuracy: Optional[float] = None
    information_coefficient: Optional[float] = None
    interval_coverage: Optional[float] = None


class BacktestResponse(BaseModel):
    model_name: str
    n_folds: int
    metrics: list[BacktestMetrics]
    fold_details: list[dict[str, Any]] = []
    completed_at: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    models_loaded: bool = False
    data_status: dict[str, Any] = {}
