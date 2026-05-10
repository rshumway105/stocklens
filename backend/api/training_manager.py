"""
Background training manager.

Tracks per-ticker training state and runs the full pipeline
(fetch data → build features → train models) in a background thread.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.log import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# In-memory status store: ticker -> status dict
_status: dict[str, dict] = {}
_lock = threading.Lock()


def get_status(ticker: str) -> dict:
    with _lock:
        return dict(_status.get(ticker.upper(), {"state": "idle"}))


def get_all_statuses() -> dict:
    with _lock:
        return {k: dict(v) for k, v in _status.items()}


def _set(ticker: str, **kwargs):
    with _lock:
        _status[ticker] = {"ticker": ticker, "updated_at": datetime.utcnow().isoformat(), **kwargs}


def models_exist(ticker: str) -> bool:
    mdir = PROJECT_ROOT / "data" / "models" / ticker.upper()
    return (mdir / "return_forecaster" / "metadata.json").exists()


def run_training_pipeline(ticker: str):
    """
    Full pipeline: fetch data, build features, train and save models.
    Runs synchronously — call from a background thread.
    """
    ticker = ticker.upper()
    _set(ticker, state="fetching", message="Fetching price data...")
    logger.info("Training pipeline started for {}", ticker)

    try:
        from backend.data.storage import (
            add_to_watchlist, save_price_data, save_fundamental_data,
            save_macro_data, load_price_data, load_fundamental_data, load_macro_data,
        )
        from backend.data.fetchers.price_fetcher import fetch_price_history, fetch_ticker_info
        from backend.data.fetchers.fundamental_fetcher import fetch_fundamentals
        from backend.data.fetchers.macro_fetcher import fetch_all_macro_series, MACRO_SERIES
        from backend.data.processors.feature_pipeline import assemble_features
        from backend.data.processors.macro_features import compute_macro_features
        from backend.models.trainer import train_production_model
        import pandas as pd

        # ── Prices ──────────────────────────────────────────────
        price_df = load_price_data(ticker)
        if price_df is None:
            price_df = fetch_price_history(ticker, years=5)
            if price_df.empty:
                _set(ticker, state="error", message=f"No price data found for {ticker}")
                return
            save_price_data(ticker, price_df)
        logger.info("{}: {} price rows loaded", ticker, len(price_df))

        # ── Fundamentals ─────────────────────────────────────────
        _set(ticker, state="fetching", message="Fetching fundamentals...")
        fund_df = load_fundamental_data(ticker)
        if fund_df is None:
            fund_data = fetch_fundamentals(ticker)
            fund_df = pd.DataFrame([fund_data])
            save_fundamental_data(ticker, fund_df)

        info = fetch_ticker_info(ticker)
        add_to_watchlist(
            ticker, name=info.get("name", ""),
            sector=info.get("sector", ""), industry=info.get("industry", ""),
        )

        # ── Macro data ───────────────────────────────────────────
        _set(ticker, state="fetching", message="Fetching macro data from FRED...")
        macro_dfs = {}
        for key, meta in MACRO_SERIES.items():
            cached = load_macro_data(meta["series_id"])
            if cached is not None:
                macro_dfs[key] = cached

        if not macro_dfs:
            fetched = fetch_all_macro_series()
            for key, df in fetched.items():
                series_id = df.columns[0] if len(df.columns) == 1 else key
                save_macro_data(series_id, df)
                macro_dfs[key] = df

        macro_features_df = compute_macro_features(macro_dfs) if macro_dfs else None

        # ── Feature matrix ───────────────────────────────────────
        _set(ticker, state="training", message="Building feature matrix...")
        feature_matrix = assemble_features(
            price_df=price_df,
            fundamentals_df=fund_df,
            macro_features_df=macro_features_df,
            ticker=ticker,
            compute_targets=True,
        )

        valid_rows = feature_matrix.dropna(subset=["target_return_21d"]).shape[0]
        if valid_rows < 50:
            _set(ticker, state="error", message=f"Only {valid_rows} valid rows — not enough data to train")
            return

        # ── Train models ─────────────────────────────────────────
        _set(ticker, state="training", message=f"Training models on {valid_rows} rows...")
        model_dir = PROJECT_ROOT / "data" / "models" / ticker
        results = train_production_model(
            feature_matrix=feature_matrix,
            horizons=["5d", "21d", "63d", "126d"],
            save_path=str(model_dir),
        )

        rf = results.get("return_forecaster")
        fve = results.get("fair_value_estimator")

        if not (rf and rf._fitted):
            _set(ticker, state="error", message="ReturnForecaster failed to train")
            return

        # ── Walk-forward backtest ─────────────────────────────────
        _set(ticker, state="training", message="Running walk-forward backtest...")
        try:
            import json
            from backend.models.trainer import WalkForwardTrainer, WalkForwardConfig

            config = WalkForwardConfig(
                min_train_days=756, test_window_days=63, step_days=63, purge_days=5
            )
            trainer = WalkForwardTrainer(config)

            rf_bt = trainer.backtest_return_forecaster(
                feature_matrix, horizons=["5d", "21d", "63d", "126d"]
            )
            fv_bt = trainer.backtest_fair_value(feature_matrix)

            bt_out = {
                "ticker": ticker,
                "return_forecaster": {
                    "metrics": rf_bt.metrics,
                    "n_folds": len(rf_bt.fold_results),
                    "n_predictions": len(rf_bt.predictions) if rf_bt.predictions is not None else 0,
                    "horizons": ["5d", "21d", "63d", "126d"],
                },
                "fair_value_estimator": {
                    "metrics": fv_bt.metrics,
                    "n_folds": len(fv_bt.fold_results),
                    "n_predictions": len(fv_bt.predictions) if fv_bt.predictions is not None else 0,
                },
            }
            bt_path = model_dir / "backtest_results.json"
            bt_path.write_text(json.dumps(bt_out, indent=2))
            logger.info("Backtest results saved for {}", ticker)
        except Exception as bt_err:
            logger.warning("Backtest failed for {} (non-fatal): {}", ticker, bt_err)

        _set(
            ticker,
            state="ready",
            message="Models trained successfully",
            trained_at=datetime.utcnow().isoformat(),
            rows=valid_rows,
        )
        logger.info("Training pipeline complete for {}", ticker)

    except Exception as e:
        logger.error("Training pipeline failed for {}: {}", ticker, e)
        _set(ticker, state="error", message=str(e))


def start_training(ticker: str):
    """Start training in a background thread (non-blocking)."""
    ticker = ticker.upper()
    current = get_status(ticker)
    if current.get("state") in ("fetching", "training"):
        logger.info("Training already in progress for {}", ticker)
        return

    _set(ticker, state="fetching", message="Starting pipeline...")
    thread = threading.Thread(target=run_training_pipeline, args=(ticker,), daemon=True)
    thread.start()
    logger.info("Training thread started for {}", ticker)
