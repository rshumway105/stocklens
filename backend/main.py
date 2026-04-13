"""
StockLens — FastAPI application entrypoint.

Initializes the app, registers routes, and sets up middleware.
Run with: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.log import logger

from backend.api.routes import macro, predictions, reports, watchlist
from backend.api.schemas import HealthResponse
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


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse()
