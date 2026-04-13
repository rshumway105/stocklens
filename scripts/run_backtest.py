#!/usr/bin/env python3
"""
Run walk-forward backtests for StockLens models.

Generates a synthetic feature matrix (or loads real data if available),
runs the walk-forward training loop, and prints performance metrics.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --horizons 5d 21d
    python scripts/run_backtest.py --folds 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.log import logger


def generate_synthetic_data(n_days: int = 1500, n_features: int = 40) -> pd.DataFrame:
    """
    Generate a synthetic feature matrix for testing the training pipeline.

    The synthetic data has realistic properties:
    - Price series with drift and volatility
    - Technical features that correlate with price movements
    - Forward return targets computed from the price series
    - NaN in early rows (insufficient lookback)
    """
    np.random.seed(42)
    dates = pd.date_range("2019-01-02", periods=n_days, freq="B")

    # Simulate a price series
    returns = np.random.randn(n_days) * 0.015 + 0.0003  # slight positive drift
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(index=dates)
    df["Close"] = prices
    df["Volume"] = np.random.randint(5_000_000, 50_000_000, size=n_days)

    # Simulate technical features
    for i in range(n_features):
        if i < 10:
            # Features correlated with future returns (with noise)
            future_ret = np.roll(returns, -21)  # 21-day forward return
            df[f"feature_{i}"] = future_ret * (0.3 + np.random.rand()) + np.random.randn(n_days) * 0.1
        elif i < 20:
            # Mean-reverting features
            df[f"feature_{i}"] = np.random.randn(n_days).cumsum() * 0.01
        else:
            # Noise features (should get low importance)
            df[f"feature_{i}"] = np.random.randn(n_days)

    # Add some realistic feature names
    rename_map = {
        "feature_0": "rsi_14",
        "feature_1": "macd",
        "feature_2": "price_to_sma_50",
        "feature_3": "bb_position",
        "feature_4": "return_21d",
        "feature_5": "hvol_20",
        "feature_6": "vol_ratio_20",
        "feature_7": "fund_pe_ratio_zscore",
        "feature_8": "fed_funds_rate_level",
        "feature_9": "composite_value",
    }
    df = df.rename(columns=rename_map)

    # Compute targets (forward returns)
    for horizon, days in [("5d", 5), ("21d", 21), ("63d", 63), ("126d", 126)]:
        future_price = df["Close"].shift(-days)
        df[f"target_return_{horizon}"] = np.log(future_price / df["Close"])
        df[f"target_direction_{horizon}"] = (df[f"target_return_{horizon}"] > 0).astype(float)

    # Fair value target
    df["target_fair_value"] = df["Close"].rolling(63, min_periods=63).mean()
    fv = df["target_fair_value"].replace(0, np.nan)
    df["target_valuation_gap"] = (df["Close"] - fv) / fv

    # Null out early rows (simulate insufficient lookback)
    df.iloc[:200, df.columns.get_indexer([c for c in df.columns if c.startswith("feature_")])] = np.nan

    return df


def run_return_backtest(feature_matrix: pd.DataFrame, horizons: list[str], max_folds: int) -> dict:
    """Run walk-forward backtest for the return forecaster."""
    from backend.models.trainer import WalkForwardTrainer, WalkForwardConfig

    config = WalkForwardConfig(
        min_train_days=756,
        test_window_days=63,
        step_days=63,
        purge_days=5,
        max_folds=max_folds if max_folds > 0 else None,
    )

    trainer = WalkForwardTrainer(config)
    result = trainer.backtest_return_forecaster(feature_matrix, horizons=horizons)

    return {
        "metrics": result.metrics,
        "n_folds": len(result.fold_results),
        "n_predictions": len(result.predictions) if result.predictions is not None else 0,
        "fold_results": result.fold_results,
    }


def run_fair_value_backtest(feature_matrix: pd.DataFrame, max_folds: int) -> dict:
    """Run walk-forward backtest for the fair value estimator."""
    from backend.models.trainer import WalkForwardTrainer, WalkForwardConfig

    config = WalkForwardConfig(
        min_train_days=756,
        test_window_days=63,
        step_days=63,
        purge_days=5,
        max_folds=max_folds if max_folds > 0 else None,
    )

    trainer = WalkForwardTrainer(config)
    result = trainer.backtest_fair_value(feature_matrix)

    return {
        "metrics": result.metrics,
        "n_folds": len(result.fold_results),
        "n_predictions": len(result.predictions) if result.predictions is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run StockLens walk-forward backtest")
    parser.add_argument("--horizons", nargs="+", default=["21d"], help="Return horizons")
    parser.add_argument("--folds", type=int, default=0, help="Max folds (0=all)")
    parser.add_argument("--skip-fv", action="store_true", help="Skip fair value backtest")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  StockLens Walk-Forward Backtest")
    print(f"  Horizons: {', '.join(args.horizons)}")
    print(f"{'='*60}\n")

    # Generate or load data
    if args.synthetic:
        print("Generating synthetic feature matrix...")
        feature_matrix = generate_synthetic_data()
        print(f"  Shape: {feature_matrix.shape}")
        print(f"  Date range: {feature_matrix.index[0].date()} → {feature_matrix.index[-1].date()}")
        print()

    # Return forecaster backtest
    print("Running return forecaster backtest...")
    try:
        rf_results = run_return_backtest(feature_matrix, args.horizons, args.folds)
        print(f"\n  Folds completed: {rf_results['n_folds']}")
        print(f"  Total predictions: {rf_results['n_predictions']}")
        print(f"\n  Metrics:")
        for key, value in rf_results["metrics"].items():
            print(f"    {key}: {value:.4f}")
    except ImportError as e:
        print(f"  ⚠ Skipped — missing dependency: {e}")
        print(f"  Install with: pip install xgboost")
        rf_results = None
    print()

    # Fair value backtest
    if not args.skip_fv:
        print("Running fair value estimator backtest...")
        try:
            fv_results = run_fair_value_backtest(feature_matrix, args.folds)
            print(f"\n  Folds completed: {fv_results['n_folds']}")
            print(f"  Total predictions: {fv_results['n_predictions']}")
            print(f"\n  Metrics:")
            for key, value in fv_results["metrics"].items():
                print(f"    {key}: {value:.4f}")
        except ImportError as e:
            print(f"  ⚠ Skipped — missing dependency: {e}")
            fv_results = None
        print()

    print(f"\n{'='*60}")
    print(f"  Backtest complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
