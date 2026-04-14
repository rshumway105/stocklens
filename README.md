# StockLens

**Open-source stock & ETF valuation and forecasting system.**

StockLens combines XGBoost models with fundamental analysis, macroeconomic indicators, and sentiment signals to estimate fair value and forecast returns for stocks and ETFs. A local React dashboard lets you explore predictions, feature explanations, and macro context.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Stock predictions are inherently uncertain. Past model performance does not guarantee future results. Always do your own research before making investment decisions.

---

## Features

| Feature | Description |
|---------|-------------|
| **Fair Value Estimation** | ML-driven fair value from fundamentals + macro. Overvalued/undervalued signals with configurable thresholds. |
| **Return Forecasting** | Forward return predictions at 1-week, 1-month, 3-month, and 6-month horizons with 80% prediction intervals. |
| **SHAP Explainability** | Top 10 feature drivers for every prediction, with plain-English descriptions (e.g., "P/E is 2.1σ above sector median"). |
| **70+ Engineered Features** | Technical indicators, sector-relative fundamental z-scores, macro regime signals, and sentiment scores. |
| **Macro Dashboard** | 17 FRED macro indicators (rates, inflation, employment, activity, market) with interactive charts. |
| **Sentiment Analysis** | News headline sentiment (FinBERT-ready), Reddit mention volume, combined signal with trend tracking. |
| **Walk-Forward Backtesting** | Expanding window validation with purge gaps — zero lookahead bias, guaranteed. |
| **Risk Flags** | Automatic detection of high volatility, drawdowns, overbought/oversold, low liquidity, high leverage. |
| **Scheduled Refresh** | APScheduler jobs for daily price/macro updates and weekly fundamentals refresh. |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              React Dashboard (Vite + Tailwind)       │
│                                                      │
│  ┌──────────┐ ┌────────────┐ ┌───────┐ ┌─────────┐ │
│  │Watchlist │ │Ticker Deep │ │ Macro │ │ Model   │ │
│  │Overview  │ │   Dive     │ │ Dash  │ │ Perf.   │ │
│  └──────────┘ └────────────┘ └───────┘ └─────────┘ │
└────────────────────────┬─────────────────────────────┘
                         │ Axios → /api/*
                         ▼
┌──────────────────────────────────────────────────────┐
│                  FastAPI Backend                      │
│                                                      │
│  Routes:  /watchlist  /prices  /fundamentals         │
│           /macro      /reports                       │
│                                                      │
│  Services:  ReportBuilder  ←  Ensemble               │
│             ↑                   ↑      ↑             │
│       Feature Pipeline    ReturnForecaster            │
│       (Tech+Fund+Macro    FairValueEstimator         │
│        +Sentiment)        SHAP Explainer             │
│                                                      │
│  Scheduler:  Daily prices · Daily macro · Weekly fund│
└──────┬──────────┬──────────┬─────────────────────────┘
       │          │          │
       ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │yfinance│ │ FRED   │ │NewsAPI │
   │        │ │ API    │ │ + RSS  │
   └───┬────┘ └───┬────┘ └───┬────┘
       └──────────┼──────────┘
                  ▼
       ┌──────────────────┐
       │ SQLite + Parquet │
       │   Local Storage  │
       └──────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for the dashboard)
- **FRED API key** — [get one free here](https://fred.stlouisfed.org/docs/api/api_key.html)

### 1. Clone and install

```bash
git clone https://github.com/rshumway105/stocklens.git
cd stocklens

# Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# For ML models (Phase 3):
pip install -e ".[ml]"           # xgboost, lightgbm, optuna, shap

# For NLP sentiment (optional):
pip install -e ".[nlp]"          # transformers, torch
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
FRED_API_KEY=your_key_here       # Required for macro data
NEWSAPI_KEY=your_key_here        # Optional, for news sentiment
```

### 3. Initialize and seed data

```bash
python scripts/setup_db.py                        # Create database tables
python scripts/seed_data.py                        # Fetch data for default tickers
python scripts/seed_data.py --tickers AAPL NVDA   # Or specific tickers
python scripts/seed_data.py --skip-macro           # Skip FRED if no API key yet
```

### 4. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

### 5. Start the dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at **http://localhost:5173**

### 6. Run backtests (optional)

```bash
python scripts/run_backtest.py                     # Synthetic data backtest
python scripts/run_backtest.py --horizons 5d 21d   # Specific horizons
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check with system status |
| `GET` | `/api/watchlist` | List all tracked tickers |
| `POST` | `/api/watchlist` | Add a ticker (`{ "ticker": "AAPL" }`) |
| `DELETE` | `/api/watchlist/{ticker}` | Remove a ticker |
| `GET` | `/api/prices/{ticker}` | OHLCV price history (`?years=5&refresh=true`) |
| `GET` | `/api/fundamentals/{ticker}` | Current fundamental metrics snapshot |
| `GET` | `/api/macro/catalog` | List all available macro series |
| `GET` | `/api/macro/{series_key}` | Get macro series data (`?refresh=true`) |
| `GET` | `/api/reports` | Watchlist overview with signals and price changes |
| `GET` | `/api/reports/{ticker}` | Full valuation report for a ticker |

---

## Feature Catalog

### Technical Features (37 features per ticker)

| Category | Features |
|----------|----------|
| Moving Averages | SMA/EMA at 5, 10, 20, 50, 100, 200 periods; price-to-SMA ratios; SMA crossovers (20/50, 50/200) |
| Momentum | RSI (14), MACD (12/26/9) + signal + histogram, Stochastic K/D, Williams %R |
| Volatility | Bollinger Band width + position, ATR (14) + ATR%, 20d/60d historical vol, vol ratio |
| Volume | OBV + 20d rate of change, volume/MA ratios (20d, 50d), volume trend |
| Price Patterns | Lookback returns (1d–252d), drawdown from 52w high, distance from 52w low, intraday range, gap |

### Fundamental Features (17+ features per ticker)

All metrics computed as **sector-relative z-scores** (how many σ from sector median):

| Category | Metrics |
|----------|---------|
| Valuation | P/E, Forward P/E, P/B, P/S, EV/EBITDA, PEG |
| Profitability | Gross margin, operating margin, net margin, ROE, ROA |
| Growth | Revenue growth, earnings growth, quarterly earnings growth |
| Health | Debt/equity, current ratio, quick ratio, FCF yield |
| Composites | Value score, quality score, growth score, health score, overall fundamental score |

### Macro Features (38 features from 17 FRED series)

Each series produces: level, 1m/3m change, % change, direction, 2-year regime z-score.

Plus derived cross-series features: yield curve slope/inversion, real rates, financial stress index, real fed funds rate.

**Publication lag enforcement** prevents lookahead bias (e.g., GDP shifted 60 days).

### Sentiment Features

| Source | Features |
|--------|----------|
| News (NewsAPI) | Daily mean/std/volume, positive ratio, 7d/21d rolling, trend |
| Social (Reddit) | Mention count, engagement score, score-weighted sentiment |
| Combined | Blended signal (70% news / 30% social), sentiment momentum |

---

## Model Architecture

### Model 1 — Return Forecaster

XGBoost regression predicting forward log returns at 4 horizons.

- **Targets:** 5-day, 21-day, 63-day, 126-day forward returns
- **Features:** All technical + fundamental + macro + sentiment
- **Intervals:** Quantile regression (10th/90th percentile) for 80% prediction bands
- **Validation:** Walk-forward expanding window, 3-year minimum, 5-day purge gap

### Model 2 — Fair Value Estimator

XGBoost regression predicting smoothed price from intrinsic value signals only.

- **Target:** 63-day moving average of price
- **Features:** Fundamental z-scores + macro only (no technicals, no current price)
- **Signal:** Overvalued if gap > +15%, undervalued if gap < -15%

### Ensemble

Blends both models. When they agree (negative returns + overvalued), the signal is stronger. Composite confidence score (0–100) based on model agreement, interval width, signal magnitude, and historical accuracy.

### Explainability

SHAP TreeExplainer for every prediction:
- Top 10 features with direction and magnitude
- Plain-English explanations (e.g., "RSI at 28 — technically oversold, pushing prediction higher")
- Falls back to feature importances if SHAP is not installed

---

## Project Structure

```
stocklens/
├── backend/
│   ├── main.py                        # FastAPI app entrypoint
│   ├── config.py                      # Settings from .env
│   ├── log.py                         # Logging (loguru with stdlib fallback)
│   ├── data/
│   │   ├── fetchers/
│   │   │   ├── price_fetcher.py       # yfinance OHLCV + ticker info
│   │   │   ├── macro_fetcher.py       # FRED API (17 series)
│   │   │   ├── fundamental_fetcher.py # yfinance fundamentals (30+ metrics)
│   │   │   └── sentiment_fetcher.py   # NewsAPI + RSS + Reddit
│   │   ├── processors/
│   │   │   ├── technical_features.py  # 37 technical indicators
│   │   │   ├── fundamental_features.py# Sector z-scores + composites
│   │   │   ├── macro_features.py      # Level/change/direction + derived
│   │   │   ├── sentiment_features.py  # Scoring + aggregation
│   │   │   ├── target_builder.py      # Forward returns + fair value targets
│   │   │   └── feature_pipeline.py    # Assembles everything into one matrix
│   │   └── storage.py                 # SQLite + Parquet I/O
│   ├── models/
│   │   ├── return_forecaster.py       # XGBoost return prediction
│   │   ├── fair_value_estimator.py    # XGBoost fair value + feature filter
│   │   ├── ensemble.py               # Signal blending + confidence scoring
│   │   ├── explainer.py              # SHAP wrapper + plain-English
│   │   └── trainer.py                # Walk-forward loop + backtest
│   ├── api/
│   │   ├── schemas.py                # 16 Pydantic response models
│   │   ├── report_builder.py         # Orchestrates data → report
│   │   └── routes/
│   │       ├── watchlist.py           # CRUD for tracked tickers
│   │       ├── predictions.py         # Price + fundamental data endpoints
│   │       ├── macro.py              # Macro series catalog + data
│   │       └── reports.py            # Valuation reports + overview
│   ├── jobs/
│   │   ├── scheduler.py             # APScheduler cron config
│   │   └── tasks.py                 # Data refresh functions
│   └── tests/
│       ├── test_phase1.py
│       └── test_phase2.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js                # Dev server + API proxy
│   ├── tailwind.config.js            # Dark terminal theme
│   ├── index.html
│   └── src/
│       ├── App.jsx                    # Router + nav
│       ├── index.css                  # Tailwind + custom components
│       ├── pages/
│       │   ├── Watchlist.jsx          # Sortable table with signals
│       │   ├── TickerDetail.jsx       # Full valuation report page
│       │   ├── MacroDashboard.jsx     # Macro indicators + charts
│       │   └── ModelPerformance.jsx   # Backtest results + methodology
│       ├── components/
│       │   ├── PriceChart.jsx         # OHLCV + fair value overlay
│       │   ├── ShapWaterfall.jsx      # Feature contribution bars
│       │   ├── ForecastFanChart.jsx   # Return predictions by horizon
│       │   ├── SentimentTimeline.jsx  # Sentiment area chart
│       │   └── ValuationBadge.jsx     # Color-coded signal badge
│       ├── hooks/
│       │   └── useApi.js             # Axios wrapper with loading/error
│       └── utils/
│           └── formatters.js         # Price, percent, color helpers
├── scripts/
│   ├── setup_db.py                   # Create database tables
│   ├── seed_data.py                  # Fetch initial data
│   ├── run_backtest.py               # Walk-forward backtest CLI
│   ├── validate_phase1.py            # Phase 1 tests (45 checks)
│   ├── validate_phase2.py            # Phase 2 tests (67 checks)
│   ├── validate_phase3.py            # Phase 3 tests (73 checks)
│   ├── validate_phase4.py            # Phase 4 tests (44 checks)
│   └── validate_phase5.py            # Phase 5 tests (81 checks)
├── notebooks/                        # Exploration notebooks (future)
├── pyproject.toml                    # Python deps + tool config
├── .env.example                      # Template for API keys
├── .gitignore
├── LICENSE                           # MIT
└── README.md
```

---

## Development

### Running tests

```bash
# All validation suites (no external deps needed)
python scripts/validate_phase1.py
python scripts/validate_phase2.py
python scripts/validate_phase3.py
python scripts/validate_phase4.py
python scripts/validate_phase5.py

# pytest (requires dev dependencies)
pytest backend/tests/ -v
```

### Code style

The project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check backend/
ruff format backend/
```

### Key design rules

1. **No lookahead bias** — every feature uses only past data; enforced by `validate_no_lookahead()` assertions
2. **Walk-forward only** — never random train/test splits on time-series data
3. **Sector-relative** — raw fundamental values are z-scored against sector peers
4. **Fail gracefully** — missing data sources log warnings, never crash the pipeline
5. **Type hints everywhere** — Pydantic models for API, type annotations for all functions
6. **Reproducibility** — random seeds fixed, model metadata saved with hyperparameters

### Adding a new data source

1. Create a fetcher in `backend/data/fetchers/`
2. Create a processor in `backend/data/processors/`
3. Wire it into `feature_pipeline.py` → `assemble_features()`
4. Add storage helpers in `storage.py` if needed
5. Add a refresh task in `jobs/tasks.py`

### Adding a new model

1. Create the model class in `backend/models/`
2. Add it to the `trainer.py` walk-forward loop
3. Wire into `ensemble.py` for signal blending
4. Update `report_builder.py` to include the new signal

---

## Development Phases

All 6 phases are ✅ Complete:

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Data pipeline & project setup | ✅ Complete |
| 2 | Feature engineering (70+ features) | ✅ Complete |
| 3 | Model training & walk-forward backtesting | ✅ Complete |
| 4 | API, valuation reports, scheduler | ✅ Complete |
| 5 | React dashboard (4 pages, 5 chart components) | ✅ Complete |
| 6 | Polish, documentation, quickstart | ✅ Complete |

---

## Roadmap

- [ ] Integrate FinBERT for real NLP sentiment scoring
- [ ] Add LightGBM as secondary model for ensembling
- [ ] Optuna hyperparameter tuning with time-series CV
- [ ] LSTM sequence model via PyTorch for ensemble diversity
- [ ] Earnings calendar integration for risk flag timing
- [ ] Portfolio-level analysis (correlation, diversification)
- [ ] Docker Compose for one-command deployment
- [ ] GitHub Actions CI for automated testing

---

## License

MIT — see [LICENSE](LICENSE).
