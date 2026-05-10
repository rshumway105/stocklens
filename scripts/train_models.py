#!/usr/bin/env python3
"""
Train StockLens models for a ticker using real data.

Steps:
  1. Fetch & cache prices, fundamentals, and macro data
  2. Build the full feature matrix
  3. Train ReturnForecaster + FairValueEstimator on all available data
  4. Save models to data/models/<ticker>/

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --ticker MSFT
    python scripts/train_models.py --ticker AAPL --skip-macro
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train StockLens models")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to train on")
    parser.add_argument("--years", type=int, default=5, help="Years of price history")
    parser.add_argument("--skip-macro", action="store_true", help="Skip FRED macro fetch")
    parser.add_argument("--horizons", nargs="+", default=["5d", "21d", "63d", "126d"])
    args = parser.parse_args()

    ticker = args.ticker.upper()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    model_dir = PROJECT_ROOT / "data" / "models" / ticker

    from backend.data.storage import (
        init_db, add_to_watchlist, save_price_data, save_fundamental_data,
        save_macro_data, load_price_data, load_fundamental_data, load_macro_data,
    )
    from backend.data.fetchers.price_fetcher import fetch_price_history, fetch_ticker_info
    from backend.data.fetchers.fundamental_fetcher import fetch_fundamentals
    from backend.data.processors.feature_pipeline import assemble_features
    from backend.data.processors.macro_features import compute_macro_features

    init_db()

    print(f"\n{'='*60}")
    print(f"  StockLens Model Trainer — {ticker}")
    print(f"{'='*60}\n")

    # ── Step 1: Prices ──────────────────────────────────────────
    print(f"[1/5] Loading price data for {ticker}...")
    price_df = load_price_data(ticker)
    if price_df is None:
        print("  Fetching from Yahoo Finance...")
        price_df = fetch_price_history(ticker, years=args.years)
        if price_df.empty:
            print(f"  ERROR: No price data found for {ticker}")
            sys.exit(1)
        save_price_data(ticker, price_df)
    print(f"  {len(price_df)} rows  ({price_df.index[0].date()} to {price_df.index[-1].date()})")

    # ── Step 2: Fundamentals ────────────────────────────────────
    print(f"\n[2/5] Loading fundamentals for {ticker}...")
    import pandas as pd
    fund_df = load_fundamental_data(ticker)
    if fund_df is None:
        print("  Fetching from Yahoo Finance...")
        fund_data = fetch_fundamentals(ticker)
        fund_df = pd.DataFrame([fund_data])
        save_fundamental_data(ticker, fund_df)

    # Add to watchlist
    info = fetch_ticker_info(ticker)
    add_to_watchlist(ticker, name=info.get("name",""), sector=info.get("sector",""), industry=info.get("industry",""))
    print(f"  PE={fund_df.get('pe_ratio', [None])[0]}, Sector={fund_df.get('sector', [''])[0] if 'sector' in fund_df.columns else ''}")

    # ── Step 3: Macro data ──────────────────────────────────────
    macro_features_df = None
    if not args.skip_macro:
        print(f"\n[3/5] Loading macro data from FRED...")
        from backend.data.fetchers.macro_fetcher import fetch_all_macro_series, MACRO_SERIES
        macro_dfs = {}
        for key, meta in MACRO_SERIES.items():
            cached = load_macro_data(meta["series_id"])
            if cached is not None:
                macro_dfs[key] = cached

        if not macro_dfs:
            print("  Fetching from FRED (this may take a minute)...")
            fetched = fetch_all_macro_series()
            for key, df in fetched.items():
                series_id = df.columns[0] if len(df.columns) == 1 else key
                save_macro_data(series_id, df)
                macro_dfs[key] = df
            print(f"  Fetched {len(macro_dfs)} macro series")
        else:
            print(f"  Loaded {len(macro_dfs)} cached macro series")

        if macro_dfs:
            macro_features_df = compute_macro_features(macro_dfs)
            print(f"  Macro feature matrix: {macro_features_df.shape}")
    else:
        print("\n[3/5] Skipping macro data")

    # ── Step 4: Build feature matrix ────────────────────────────
    print(f"\n[4/5] Building feature matrix...")
    feature_matrix = assemble_features(
        price_df=price_df,
        fundamentals_df=fund_df,
        macro_features_df=macro_features_df,
        ticker=ticker,
        compute_targets=True,
    )
    valid_rows = feature_matrix.dropna(subset=["target_return_21d"]).shape[0]
    print(f"  Shape: {feature_matrix.shape}  ({valid_rows} rows with valid 21d target)")

    if valid_rows < 200:
        print(f"  WARNING: Only {valid_rows} valid rows — need at least 200 for training")
        if valid_rows < 50:
            print("  ERROR: Too little data to train. Try --years 7 or a different ticker.")
            sys.exit(1)

    # ── Step 5: Train & save models ──────────────────────────────
    print(f"\n[5/5] Training models (horizons: {', '.join(args.horizons)})...")
    from backend.models.trainer import train_production_model

    results = train_production_model(
        feature_matrix=feature_matrix,
        horizons=args.horizons,
        save_path=str(model_dir),
    )

    rf = results.get("return_forecaster")
    fve = results.get("fair_value_estimator")

    if rf and rf._fitted:
        print(f"  ReturnForecaster trained — {len(rf.models)} horizon models")
        print(f"  Saved to {model_dir}/return_forecaster/")
    else:
        print("  WARNING: ReturnForecaster did not fit")

    if fve and fve._fitted:
        print(f"  FairValueEstimator trained")
        print(f"  Saved to {model_dir}/fair_value_estimator/")
    else:
        print("  WARNING: FairValueEstimator did not fit")

    # Quick sanity prediction
    if rf and rf._fitted:
        import numpy as np
        target_cols = [c for c in feature_matrix.columns if c.startswith("target_")]
        feature_cols = [c for c in feature_matrix.columns if c not in target_cols]
        X_latest = feature_matrix[feature_cols].iloc[[-1]].fillna(feature_matrix[feature_cols].median())
        preds = rf.predict(X_latest, include_intervals=True)
        print(f"\n  Live prediction for {ticker}:")
        for h, pred_df in preds.items():
            ret = pred_df["predicted_return"].iloc[0]
            print(f"    {h}: {ret:+.2%} predicted return")

    print(f"\n{'='*60}")
    print(f"  Training complete! Models saved to:")
    print(f"  {model_dir}")
    print(f"\n  Restart the backend to load the trained models:")
    print(f"  (Ctrl+C then: python -m uvicorn backend.main:app --reload --port 8000)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
