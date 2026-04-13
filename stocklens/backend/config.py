"""
StockLens configuration module.

Loads settings from environment variables and .env file.
All configuration is centralized here to avoid scattered env lookups.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root is two levels up from this file (stocklens/backend/config.py -> stocklens/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    fred_api_key: str = Field(default="", description="FRED API key for macro data")
    newsapi_key: str = Field(default="", description="NewsAPI key for news headlines")
    reddit_client_id: str = Field(default="", description="Reddit API client ID")
    reddit_client_secret: str = Field(default="", description="Reddit API client secret")
    reddit_user_agent: str = Field(default="stocklens/1.0", description="Reddit API user agent")

    # --- Paths ---
    database_path: str = Field(
        default="data/stocklens.db",
        description="SQLite database path relative to project root",
    )
    cache_dir: str = Field(
        default="data/cache",
        description="Directory for parquet cache files, relative to project root",
    )

    # --- App Settings ---
    log_level: str = Field(default="INFO", description="Logging level")
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")

    # --- Default Watchlist ---
    default_tickers: list[str] = Field(
        default=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "JPM", "V", "SPY", "QQQ",
        ],
        description="Default tickers to track on first run",
    )

    # --- Data Settings ---
    price_history_years: int = Field(
        default=5, description="Years of price history to fetch"
    )
    min_training_years: int = Field(
        default=3, description="Minimum years of data before first prediction"
    )

    @property
    def db_path(self) -> Path:
        """Absolute path to the SQLite database."""
        return PROJECT_ROOT / self.database_path

    @property
    def parquet_cache_dir(self) -> Path:
        """Absolute path to the parquet cache directory."""
        return PROJECT_ROOT / self.cache_dir

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
