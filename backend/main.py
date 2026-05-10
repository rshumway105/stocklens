"""
StockLens — FastAPI application entrypoint.

Initializes the app, registers routes, and sets up middleware.
Run with: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.log import logger

from backend.api.routes import macro, predictions, reports, signals, watchlist
from backend.api.schemas import HealthResponse
from backend.api import training_manager
from backend.config import get_settings
from backend.data.storage import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    settings = get_settings()
    logger.info("Starting StockLens v0.1.0")
    logger.info("Database: {}", settings.db_path)
    logger.info("Cache dir: {}", settings.parquet_cache_dir)

    # Initialize database tables
    init_db()

    yield

    logger.info("StockLens shutting down")


app = FastAPI(
    title="StockLens",
    description="Stock & ETF valuation and forecasting API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server (localhost:5173 for Vite, 3000 for CRA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(watchlist.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(macro.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(signals.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse()


@app.get("/api/status/{ticker}", tags=["system"])
async def training_status(ticker: str):
    """
    Get the training pipeline status for a ticker.

    States: idle | fetching | training | ready | error
    """
    return training_manager.get_status(ticker.upper())


@app.get("/api/status", tags=["system"])
async def all_training_statuses():
    """Get training status for all tickers."""
    return training_manager.get_all_statuses()


@app.get("/api/backtest/{ticker}", tags=["system"])
async def get_backtest_results(ticker: str):
    """Return saved walk-forward backtest results for a ticker."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "models" / ticker.upper() / "backtest_results.json"
    if not path.exists():
        return {"ticker": ticker.upper(), "available": False}
    data = json.loads(path.read_text())
    data["available"] = True
    return data


@app.get("/api/backtest", tags=["system"])
async def list_backtest_results():
    """Return backtest results for all tickers that have them."""
    import json
    from pathlib import Path
    models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    results = {}
    if models_dir.exists():
        for ticker_dir in models_dir.iterdir():
            path = ticker_dir / "backtest_results.json"
            if path.exists():
                results[ticker_dir.name] = json.loads(path.read_text())
    return results



# ---------------------------------------------------------------------------
# Market movers — top gainers / losers from a broad universe of tickers
# ---------------------------------------------------------------------------

# Representative universe: ~120 liquid names across all sectors
_MOVERS_UNIVERSE = [
    # Mega-cap Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "IBM",
    # Semiconductors
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "ASML", "TSM", "ARM", "MRVL",
    # Software & Cloud
    "CRM", "ADBE", "NOW", "SNOW", "PLTR", "NET", "DDOG", "CRWD", "PANW", "FTNT", "WDAY",
    # Internet & Consumer Tech
    "UBER", "ABNB", "COIN", "NFLX", "DIS", "SPOT", "PINS", "SNAP", "RBLX",
    "BABA", "JD", "PDD", "MELI", "SE",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "COF", "AXP", "V", "MA", "PYPL", "BLK", "SCHW",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "GILD",
    "ISRG", "MRNA", "REGN", "VRTX", "BSX", "MDT", "SYK",
    # Consumer Discretionary
    "HD", "LOW", "NKE", "SBUX", "MCD", "CMG", "TJX", "ROST", "LULU",
    "F", "GM", "RIVN", "ETSY", "EBAY",
    # Consumer Staples
    "WMT", "COST", "TGT", "PG", "KO", "PEP", "PM", "MDLZ", "CL", "EL", "ULTA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "DVN",
    # Industrials
    "CAT", "DE", "BA", "HON", "GE", "MMM", "RTX", "LMT", "NOC", "UPS", "FDX",
    # Financials extras / RE / Utilities
    "AMT", "PLD", "EQIX", "NEE", "DUK", "SO",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT", "HYG", "VNQ",
]

# Friendly names for display (ticker → name)
_TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "ORCL": "Oracle",
    "IBM": "IBM", "AMD": "AMD", "INTC": "Intel", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "MU": "Micron", "AMAT": "Applied Materials", "LRCX": "Lam Research", "ASML": "ASML",
    "TSM": "TSMC", "ARM": "Arm Holdings", "MRVL": "Marvell",
    "CRM": "Salesforce", "ADBE": "Adobe", "NOW": "ServiceNow", "SNOW": "Snowflake",
    "PLTR": "Palantir", "NET": "Cloudflare", "DDOG": "Datadog", "CRWD": "CrowdStrike",
    "PANW": "Palo Alto", "FTNT": "Fortinet", "WDAY": "Workday",
    "UBER": "Uber", "ABNB": "Airbnb", "COIN": "Coinbase", "NFLX": "Netflix", "DIS": "Disney",
    "SPOT": "Spotify", "PINS": "Pinterest", "SNAP": "Snap", "RBLX": "Roblox",
    "BABA": "Alibaba", "JD": "JD.com", "PDD": "PDD Holdings", "MELI": "MercadoLibre", "SE": "Sea Ltd",
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "C": "Citigroup", "COF": "Capital One", "AXP": "Amex",
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "BLK": "BlackRock", "SCHW": "Schwab",
    "UNH": "UnitedHealth", "JNJ": "J&J", "LLY": "Eli Lilly", "ABBV": "AbbVie", "MRK": "Merck",
    "PFE": "Pfizer", "TMO": "Thermo Fisher", "ABT": "Abbott", "AMGN": "Amgen", "GILD": "Gilead",
    "ISRG": "Intuitive Surgical", "MRNA": "Moderna", "REGN": "Regeneron", "VRTX": "Vertex",
    "BSX": "Boston Scientific", "MDT": "Medtronic", "SYK": "Stryker",
    "HD": "Home Depot", "LOW": "Lowe's", "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonald's",
    "CMG": "Chipotle", "TJX": "TJX", "ROST": "Ross Stores", "LULU": "Lululemon",
    "F": "Ford", "GM": "General Motors", "RIVN": "Rivian", "ETSY": "Etsy", "EBAY": "eBay",
    "WMT": "Walmart", "COST": "Costco", "TGT": "Target", "PG": "P&G", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "PM": "Philip Morris", "MDLZ": "Mondelez", "CL": "Colgate", "EL": "Estee Lauder",
    "ULTA": "Ulta Beauty",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger",
    "EOG": "EOG Resources", "MPC": "Marathon Petroleum", "PSX": "Phillips 66",
    "VLO": "Valero", "OXY": "Occidental", "DVN": "Devon Energy",
    "CAT": "Caterpillar", "DE": "Deere", "BA": "Boeing", "HON": "Honeywell", "GE": "GE",
    "MMM": "3M", "RTX": "RTX Corp", "LMT": "Lockheed Martin", "NOC": "Northrop Grumman",
    "UPS": "UPS", "FDX": "FedEx",
    "AMT": "American Tower", "PLD": "Prologis", "EQIX": "Equinix",
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Co",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF", "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF", "GLD": "Gold ETF", "SLV": "Silver ETF",
    "USO": "Oil ETF", "TLT": "20Y Treasury ETF", "HYG": "High Yield Bond ETF", "VNQ": "Real Estate ETF",
}

# Simple in-memory cache: {"data": ..., "expires": timestamp}
_movers_cache: dict = {}
_MOVERS_CACHE_TTL = 600  # 10 minutes


@app.get("/api/movers", tags=["market"])
async def get_market_movers(top_n: int = 10):
    """
    Return top gainers and losers from a broad universe of ~120 tickers.
    Results are cached for 10 minutes to avoid hammering Yahoo Finance.
    """
    import time
    import asyncio

    # Serve from cache if still fresh
    now = time.time()
    if _movers_cache.get("expires", 0) > now:
        cached = _movers_cache["data"]
        return {**cached, "cached": True}

    def _fetch():
        import yfinance as yf
        import pandas as pd

        try:
            # Batch download — much faster than individual requests
            raw = yf.download(
                tickers=_MOVERS_UNIVERSE,
                period="5d",       # grab 5 days so we always have at least 2 trading days
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error("yf.download failed in movers: {}", e)
            return {"gainers": [], "losers": [], "universe_size": 0}

        if raw.empty:
            return {"gainers": [], "losers": [], "universe_size": 0}

        # yf.download with multiple tickers returns a MultiIndex: (field, ticker)
        close = raw["Close"] if "Close" in raw.columns else raw.get("Adj Close")
        if close is None:
            return {"gainers": [], "losers": [], "universe_size": 0}

        # Drop tickers with no data
        close = close.dropna(axis=1, how="all")
        if len(close) < 2:
            return {"gainers": [], "losers": [], "universe_size": 0}

        last = close.iloc[-1]
        prev = close.iloc[-2]
        pct_change = ((last - prev) / prev * 100).dropna()

        results = []
        for ticker, chg in pct_change.items():
            price = last.get(ticker)
            if price is None or pd.isna(price) or pd.isna(chg):
                continue
            results.append({
                "ticker": ticker,
                "name": _TICKER_NAMES.get(ticker, ticker),
                "price": round(float(price), 2),
                "change_pct": round(float(chg), 2),
            })

        results.sort(key=lambda x: x["change_pct"], reverse=True)
        return {
            "gainers": results[:top_n],
            "losers": list(reversed(results))[:top_n],
            "universe_size": len(results),
        }

    try:
        # Run blocking yfinance call in a thread pool
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.error("Movers fetch failed: {}", e)
        data = {"gainers": [], "losers": [], "universe_size": 0}

    # Cache the result
    _movers_cache["data"] = data
    _movers_cache["expires"] = now + _MOVERS_CACHE_TTL

    return {**data, "cached": False}


@app.get("/api/news", tags=["news"])
async def get_market_news(limit: int = 20):
    """
    Fetch recent market news headlines from Yahoo Finance RSS feed.
    Returns up to `limit` articles sorted newest-first.
    """
    try:
        import feedparser
        from datetime import timezone
        import time

        RSS_FEEDS = [
            ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
            ("Reuters Markets", "https://feeds.reuters.com/reuters/businessNews"),
        ]

        articles = []
        for source_name, url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:15]:
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = time.strftime("%Y-%m-%dT%H:%M:%SZ", entry.published_parsed)
                    articles.append({
                        "title": entry.get("title", "").strip(),
                        "summary": entry.get("summary", "").strip()[:280],
                        "source": source_name,
                        "url": entry.get("link", ""),
                        "published_at": published,
                    })
            except Exception:
                pass

        # Sort by published date descending, put None dates last
        articles.sort(key=lambda a: a["published_at"] or "", reverse=True)
        return {"articles": articles[:limit], "count": len(articles[:limit])}
    except Exception as e:
        logger.error("News fetch failed: {}", e)
        return {"articles": [], "count": 0}
