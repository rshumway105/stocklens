"""
Walk-Forward Trainer — time-series-aware training and backtesting.

Implements expanding window (walk-forward) cross-validation:
1. Start with a minimum training window (e.g., 3 years).
2. Train the model on all data up to time t.
3. Predict the next period (test window).
4. Slide forward and repeat.

This ensures:
- No future data ever leaks into training.
- Model performance is measured on truly out-of-sample data.
- We can track how the model performs over time.

The trainer handles both the ReturnForecaster and FairValueEstimator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger
from backend.data.processors.target_builder import get_target_columns


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward training."""

    # Minimum training window in trading days (~3 years)
    min_train_days: int = 756

    # How many days to predict before retraining
    test_window_days: int = 63  # ~3 months

    # Step size: how many days to slide forward each iteration
    step_days: int = 63

    # Purge gap: days between train end and test start (avoids leakage
    # from overlapping forward return windows)
    purge_days: int = 5

    # Maximum number of walk-forward folds (None = use all available)
    max_folds: Optional[int] = None


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""

    # Per-fold results
    fold_results: list[dict[str, Any]] = field(default_factory=list)

    # Aggregated predictions (concatenated out-of-sample predictions)
    predictions: Optional[pd.DataFrame] = None

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Training metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class WalkForwardTrainer:
    """
    Walk-forward training loop for time-series models.

    Supports both return forecasting and fair value estimation.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def generate_folds(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Generate walk-forward train/test split indices.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            List of fold dicts, each with:
            - fold: fold number
            - train_start, train_end: training period
            - test_start, test_end: testing period
            - train_idx, test_idx: integer index arrays
        """
        n = len(df)
        folds = []
        fold_num = 0

        train_end = self.config.min_train_days

        while train_end + self.config.purge_days + self.config.test_window_days <= n:
            test_start = train_end + self.config.purge_days
            test_end = min(test_start + self.config.test_window_days, n)

            folds.append({
                "fold": fold_num,
                "train_start": 0,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_start_date": str(df.index[0].date()),
                "train_end_date": str(df.index[train_end - 1].date()),
                "test_start_date": str(df.index[test_start].date()),
                "test_end_date": str(df.index[test_end - 1].date()),
                "train_size": train_end,
                "test_size": test_end - test_start,
            })

            train_end += self.config.step_days
            fold_num += 1

            if self.config.max_folds and fold_num >= self.config.max_folds:
                break

        logger.info(
            "Generated {} walk-forward folds ({}→{} days, step={})",
            len(folds), self.config.min_train_days, n, self.config.step_days,
        )

        return folds

    def backtest_return_forecaster(
        self,
        feature_matrix: pd.DataFrame,
        horizons: Optional[list[str]] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest for the return forecaster.

        Args:
            feature_matrix: Complete feature matrix with targets.
            horizons: Which horizons to backtest.  Defaults to ["21d"].

        Returns:
            BacktestResult with per-fold results and aggregated metrics.
        """
        from backend.models.return_forecaster import ReturnForecaster

        if horizons is None:
            horizons = ["21d"]

        target_cols = get_target_columns()
        feature_cols = [c for c in feature_matrix.columns if c not in target_cols]
        X = feature_matrix[feature_cols]
        targets = {h: feature_matrix[f"target_return_{h}"] for h in horizons
                    if f"target_return_{h}" in feature_matrix.columns}

        folds = self.generate_folds(feature_matrix)

        if not folds:
            logger.warning("No valid folds generated — not enough data")
            return BacktestResult()

        all_predictions = []
        fold_results = []

        for fold_info in folds:
            train_idx = slice(fold_info["train_start"], fold_info["train_end"])
            test_idx = slice(fold_info["test_start"], fold_info["test_end"])

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = {h: t.iloc[train_idx] for h, t in targets.items()}
            y_test = {h: t.iloc[test_idx] for h, t in targets.items()}

            # Fill NaN features with training median (no test leakage)
            train_median = X_train.median()
            X_train = X_train.fillna(train_median)
            X_test = X_test.fillna(train_median)

            try:
                model = ReturnForecaster()
                model.config.horizons = horizons
                model.fit(X_train, y_train, fit_quantiles=True)

                preds = model.predict(X_test)

                for horizon in horizons:
                    if horizon in preds and horizon in y_test:
                        pred_df = preds[horizon].copy()
                        pred_df["actual_return"] = y_test[horizon].values[:len(pred_df)]
                        pred_df["horizon"] = horizon
                        pred_df["fold"] = fold_info["fold"]
                        all_predictions.append(pred_df)

                fold_result = {
                    **fold_info,
                    "status": "success",
                }

                # Compute per-fold metrics
                for horizon in horizons:
                    if horizon in preds and horizon in y_test:
                        actual = y_test[horizon].dropna()
                        predicted = preds[horizon]["predicted_return"].reindex(actual.index).dropna()
                        common = actual.index.intersection(predicted.index)
                        if len(common) > 0:
                            a = actual.loc[common]
                            p = predicted.loc[common]
                            fold_result[f"rmse_{horizon}"] = float(np.sqrt(((a - p) ** 2).mean()))
                            fold_result[f"direction_accuracy_{horizon}"] = float(
                                (np.sign(a) == np.sign(p)).mean()
                            )

                fold_results.append(fold_result)

            except Exception as e:
                logger.error("Fold {} failed: {}", fold_info["fold"], e)
                fold_results.append({**fold_info, "status": "failed", "error": str(e)})

        # Aggregate predictions
        if all_predictions:
            combined = pd.concat(all_predictions)
        else:
            combined = pd.DataFrame()

        # Compute overall metrics
        metrics = self._compute_backtest_metrics(combined, horizons)

        result = BacktestResult(
            fold_results=fold_results,
            predictions=combined,
            metrics=metrics,
            metadata={
                "n_folds": len(folds),
                "horizons": horizons,
                "config": {
                    "min_train_days": self.config.min_train_days,
                    "test_window_days": self.config.test_window_days,
                    "step_days": self.config.step_days,
                    "purge_days": self.config.purge_days,
                },
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            "Backtest complete: {} folds, {} total predictions",
            len(folds), len(combined),
        )

        return result

    def backtest_fair_value(
        self,
        feature_matrix: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run walk-forward backtest for the fair value estimator.

        Args:
            feature_matrix: Complete feature matrix with targets.

        Returns:
            BacktestResult with valuation signal accuracy.
        """
        from backend.models.fair_value_estimator import (
            FairValueEstimator,
            filter_fair_value_features,
        )

        target_cols = get_target_columns()
        feature_cols = [c for c in feature_matrix.columns if c not in target_cols]
        X_full = feature_matrix[feature_cols]
        X_fv = filter_fair_value_features(X_full)

        target_col = "target_fair_value"
        if target_col not in feature_matrix.columns:
            logger.error("target_fair_value not found in feature matrix")
            return BacktestResult()

        y = feature_matrix[target_col]
        folds = self.generate_folds(feature_matrix)

        all_predictions = []
        fold_results = []

        for fold_info in folds:
            train_idx = slice(fold_info["train_start"], fold_info["train_end"])
            test_idx = slice(fold_info["test_start"], fold_info["test_end"])

            X_train = X_fv.iloc[train_idx]
            X_test = X_fv.iloc[test_idx]
            y_train = y.iloc[train_idx]

            train_median = X_train.median()
            X_train = X_train.fillna(train_median)
            X_test = X_test.fillna(train_median)

            try:
                model = FairValueEstimator()
                model.fit(X_train, y_train)

                if not model._fitted:
                    fold_results.append({**fold_info, "status": "skipped"})
                    continue

                pred_df = model.predict(X_test)

                # Add actual price and valuation gap for evaluation
                if "Close" in feature_matrix.columns:
                    actual_prices = feature_matrix["Close"].iloc[test_idx]
                    pred_df["actual_price"] = actual_prices.values[:len(pred_df)]
                    fv = pred_df["fair_value"].replace(0, np.nan)
                    pred_df["valuation_gap"] = (
                        (pred_df["actual_price"] - fv) / fv
                    )

                pred_df["fold"] = fold_info["fold"]
                all_predictions.append(pred_df)
                fold_results.append({**fold_info, "status": "success"})

            except Exception as e:
                logger.error("Fair value fold {} failed: {}", fold_info["fold"], e)
                fold_results.append({**fold_info, "status": "failed", "error": str(e)})

        combined = pd.concat(all_predictions) if all_predictions else pd.DataFrame()

        metrics = {}
        if not combined.empty and "fair_value" in combined.columns and "actual_price" in combined.columns:
            valid = combined.dropna(subset=["fair_value", "actual_price"])
            if len(valid) > 0:
                errors = valid["actual_price"] - valid["fair_value"]
                metrics["mae"] = float(errors.abs().mean())
                metrics["rmse"] = float(np.sqrt((errors ** 2).mean()))
                metrics["mean_valuation_gap"] = float(valid["valuation_gap"].mean())

        return BacktestResult(
            fold_results=fold_results,
            predictions=combined,
            metrics=metrics,
            metadata={
                "n_folds": len(folds),
                "model": "fair_value",
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

    def _compute_backtest_metrics(
        self,
        predictions: pd.DataFrame,
        horizons: list[str],
    ) -> dict[str, float]:
        """Compute aggregate backtest metrics across all folds."""
        metrics: dict[str, float] = {}

        if predictions.empty:
            return metrics

        for horizon in horizons:
            horizon_data = predictions[predictions["horizon"] == horizon].copy()
            valid = horizon_data.dropna(subset=["predicted_return", "actual_return"])

            if len(valid) == 0:
                continue

            pred = valid["predicted_return"]
            actual = valid["actual_return"]

            # RMSE
            metrics[f"rmse_{horizon}"] = float(np.sqrt(((actual - pred) ** 2).mean()))

            # MAE
            metrics[f"mae_{horizon}"] = float((actual - pred).abs().mean())

            # Direction accuracy (did we predict the right sign?)
            direction_correct = (np.sign(pred) == np.sign(actual)).mean()
            metrics[f"direction_accuracy_{horizon}"] = float(direction_correct)

            # Information coefficient (rank correlation)
            if len(valid) > 10:
                ic = pred.corr(actual, method="spearman")
                metrics[f"ic_{horizon}"] = float(ic) if not np.isnan(ic) else 0.0

            # Prediction interval coverage (if available)
            if "lower_bound" in valid.columns and "upper_bound" in valid.columns:
                in_interval = (
                    (actual >= valid["lower_bound"]) &
                    (actual <= valid["upper_bound"])
                )
                coverage = in_interval.mean()
                metrics[f"interval_coverage_{horizon}"] = float(coverage)

            logger.info(
                "Backtest {} — RMSE: {:.4f}, Direction: {:.1%}, IC: {:.3f}",
                horizon,
                metrics.get(f"rmse_{horizon}", 0),
                metrics.get(f"direction_accuracy_{horizon}", 0),
                metrics.get(f"ic_{horizon}", 0),
            )

        return metrics


def train_production_model(
    feature_matrix: pd.DataFrame,
    horizons: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Train both models on all available data (for production use).

    This trains on the FULL dataset (not walk-forward) to get the
    best possible model for generating live predictions.

    Args:
        feature_matrix: Complete feature matrix with targets.
        horizons: Return horizons to forecast.
        save_path: Directory to save model artifacts.

    Returns:
        Dict with 'return_forecaster', 'fair_value_estimator', and metadata.
    """
    from backend.models.return_forecaster import ReturnForecaster
    from backend.models.fair_value_estimator import (
        FairValueEstimator,
        filter_fair_value_features,
    )
    from pathlib import Path

    if horizons is None:
        horizons = ["5d", "21d", "63d", "126d"]

    target_cols = get_target_columns()
    feature_cols = [c for c in feature_matrix.columns if c not in target_cols]
    X = feature_matrix[feature_cols].copy()
    X = X.fillna(X.median())

    results: dict[str, Any] = {}

    # --- Train return forecaster ---
    targets = {h: feature_matrix[f"target_return_{h}"] for h in horizons
                if f"target_return_{h}" in feature_matrix.columns}

    rf = ReturnForecaster()
    rf.config.horizons = horizons
    rf.fit(X, targets)
    results["return_forecaster"] = rf

    # --- Train fair value estimator ---
    X_fv = filter_fair_value_features(X)
    fv_target = feature_matrix.get("target_fair_value")

    if fv_target is not None:
        fve = FairValueEstimator()
        fve.fit(X_fv, fv_target)
        results["fair_value_estimator"] = fve
    else:
        logger.warning("No fair_value target — skipping FV model")

    # --- Save if requested ---
    if save_path:
        path = Path(save_path)
        rf.save(path / "return_forecaster")
        if "fair_value_estimator" in results:
            results["fair_value_estimator"].save(path / "fair_value_estimator")
        logger.info("Models saved to {}", path)

    return results
