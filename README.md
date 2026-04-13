# StockLens

**Open-source stock & ETF valuation and forecasting system.**

StockLens combines machine learning models with fundamental analysis, macroeconomic indicators, and sentiment signals to estimate fair value and forecast returns for stocks and ETFs. It provides a local web dashboard for exploring predictions, feature explanations, and macro context.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions.

---

## Features

- **Fair Value Estimation** — ML-driven fair value with confidence intervals and valuation gap signals
- **Return Forecasting** — Forward return predictions at 1-week, 1-month, 3-month, and 6-month horizons
- **SHAP Explainability** — Top feature drivers for every prediction, explained in plain English
- **Macro Dashboard** — Real-time macroeconomic indicators (rates, inflation, employment, VIX)
- **Sentiment Analysis** — News and social media sentiment via FinBERT
- **Walk-Forward Backtesting** — No lookahead bias; time-series-aware validation throughout

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  React Dashboard                │
│  Watchlist · Ticker Detail · Macro · Backtest   │
└──────────────────────┬──────────────────────────┘
                       │ HTTP (Axios)
                       ▼
┌─────────────────────────────────────────────────┐
│                FastAPI Backend                   │
│  /api/watchlist · /api/prices · /api/macro       │
│  /api/predictions · /api/reports                 │
└──────┬──────────┬──────────┬────────────────────┘
       │          │          │
       ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │yfinance│ │ FRED   │ │NewsAPI │
   │ Prices │ │ Macro  │ │Sentim. │
   └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       ▼          ▼          ▼
   ┌─────────────────────────────┐
   │  SQLite + Parquet Storage   │
   └─────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the dashboard)
- A FRED API key ([get one free](https://fred.stlouisfed.org/docs/api/api_key.html))

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/stocklens.git
cd stocklens

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your FRED_API_KEY (required) and NEWSAPI_KEY (optional)
```

### 3. Initialize database and seed data

```bash
# Create database tables
python scripts/setup_db.py

# Fetch initial data (prices, fundamentals, macro)
python scripts/seed_data.py

# Or seed specific tickers only:
python scripts/seed_data.py --tickers AAPL MSFT NVDA SPY
```

### 4. Start the API server

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive API documentation.

### 5. Start the dashboard (Phase 5)

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Path                       | Description                        |
|--------|----------------------------|------------------------------------|
| GET    | `/api/health`              | Health check                       |
| GET    | `/api/watchlist`           | List all tracked tickers           |
| POST   | `/api/watchlist`           | Add a ticker                       |
| DELETE | `/api/watchlist/{ticker}`  | Remove a ticker                    |
| GET    | `/api/prices/{ticker}`     | OHLCV price history                |
| GET    | `/api/fundamentals/{ticker}` | Fundamental metrics snapshot    |
| GET    | `/api/macro/catalog`       | List available macro series        |
| GET    | `/api/macro/{series_key}`  | Get macro series data              |

---

## Project Structure

```
stocklens/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings & env loading
│   ├── data/
│   │   ├── fetchers/           # Data source wrappers
│   │   │   ├── price_fetcher.py
│   │   │   ├── macro_fetcher.py
│   │   │   ├── fundamental_fetcher.py
│   │   │   └── sentiment_fetcher.py
│   │   ├── processors/         # Feature engineering (Phase 2)
│   │   └── storage.py          # SQLite + Parquet I/O
│   ├── models/                 # ML models (Phase 3)
│   ├── api/
│   │   ├── routes/             # Endpoint modules
│   │   └── schemas.py          # Pydantic models
│   ├── jobs/                   # Scheduled tasks (Phase 4)
│   └── tests/
├── frontend/                   # React dashboard (Phase 5)
├── notebooks/                  # Exploration & analysis
├── scripts/                    # CLI utilities
├── pyproject.toml
└── README.md
```

---

## Development Phases

| Phase | Focus                         | Status      |
|-------|-------------------------------|-------------|
| 1     | Data pipeline & project setup | ✅ Complete |
| 2     | Feature engineering           | ⬜ Planned  |
| 3     | Model training & backtesting  | ⬜ Planned  |
| 4     | API & valuation reports       | ⬜ Planned  |
| 5     | React dashboard               | ⬜ Planned  |
| 6     | Polish & documentation        | ⬜ Planned  |

---

## Running Tests

```bash
pytest backend/tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
