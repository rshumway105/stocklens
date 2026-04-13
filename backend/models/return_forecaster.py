"""
Return Forecaster — Model 1.

Predicts forward log returns at multiple horizons (5d, 21d, 63d, 126d)
using XGBoost regression.  Optionally uses LightGBM as a secondary model
for ensembling.

Key design decisions:
- Walk-forward expanding window training only (no random splits).
- Quantile regression for prediction intervals (confidence bounds).
- Feature importance tracked per training run.
- Model artifacts saved with metadata for reproducibility.

The model operates on the feature matrix produced by the Phase 2 pipeline.
Target columns are prefixed with 'target_' and strictly separated from features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ReturnForecasterConfig:
    """Configuration for the return forecaster model."""

    # Target horizons to predict
    horizons: list[str] = field(default_factory=lambda: ["5d", "21d", "63d", "126d"])

    # XGBoost hyperparameters (defaults; overridden by tuning)
    xgb_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    })

    # Quantile regression params for prediction intervals
    quantile_lower: float = 0.1   # 10th percentile
    quantile_upper: float = 0.9   # 90th percentile

    # Walk-forward settings
    min_train_days: int = 756     # ~3 years minimum training window
    retrain_every_days: int = 63  # retrain every ~3 months

    # Feature selection
    max_features: Optional[int] = None  # None = use all; set to limit
    drop_features: list[str] = field(default_factory=lambda: [
        "Close", "Volume",  # raw prices shouldn't be features
    ])


# ---------------------------------------------------------------------------
# Return Forecaster model
# ---------------------------------------------------------------------------

class ReturnForecaster:
    """
    XGBoost-based return forecaster.

    Predicts forward returns at configurable horizons.
    Supports walk-forward training and quantile prediction intervals.
    """

    def __init__(self, config: Optional[ReturnForecasterConfig] = None):
        self.config = config or ReturnForecasterConfig()
        self.models: dict[str, Any] = {}            # horizon -> fitted model
        self.models_lower: dict[str, Any] = {}      # horizon -> lower quantile model
        self.models_upper: dict[str, Any] = {}      # horizon -> upper quantile model
        self.feature_names: list[str] = []
        self.feature_importances: dict[str, pd.Series] = {}
        self.training_metadata: dict[str, Any] = {}
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: dict[str, pd.Series],
        fit_quantiles: bool = True,
    ) -> "ReturnForecaster":
        """
        Train the model on feature matrix X and target dict y.

        Args:
            X: Feature matrix (rows = observations, columns = features).
                Must NOT contain any target columns.
            y: Dict mapping horizon label (e.g. "21d") to target Series.
            fit_quantiles: Whether to also fit quantile models for intervals.

        Returns:
            self (fitted model).
        """
        self._validate_features(X)
        self.feature_names = list(X.columns)

        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            raise

        for horizon in self.config.horizons:
            if horizon not in y:
                logger.warning("No target for horizon {} — skipping", horizon)
                continue

            target = y[horizon]

            # Align X and y, drop NaN targets
            mask = target.notna()
            X_train = X.loc[mask]
            y_train = target.loc[mask]

            if len(X_train) < 100:
                logger.warning(
                    "Only {} samples for horizon {} — skipping (need >= 100)",
                    len(X_train), horizon,
                )
                continue

            logger.info(
                "Training return forecaster for {} — {} samples, {} features",
                horizon, len(X_train), len(self.feature_names),
            )

            # Main model (mean prediction)
            model = xgb.XGBRegressor(**self.config.xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False,
            )
            self.models[horizon] = model

            # Feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=self.feature_names,
            ).sort_values(ascending=False)
            self.feature_importances[horizon] = importance

            # Quantile models for prediction intervals
            if fit_quantiles:
                for q, q_models, label in [
                    (self.config.quantile_lower, self.models_lower, "lower"),
                    (self.config.quantile_upper, self.models_upper, "upper"),
                ]:
                    q_params = {**self.config.xgb_params}
                    q_params["objective"] = "reg:quantileerror"
                    q_params["quantile_alpha"] = q

                    q_model = xgb.XGBRegressor(**q_params)
                    q_model.fit(X_train, y_train, verbose=False)
                    q_models[horizon] = q_model

                logger.info("  Fitted quantile models for {}", horizon)

        self._fitted = True
        self.training_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "horizons": list(self.models.keys()),
        }

        return self

    def predict(
        self,
        X: pd.DataFrame,
        include_intervals: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate predictions for each horizon.

        Args:
            X: Feature matrix (same columns as training).
            include_intervals: Whether to include prediction intervals.

        Returns:
            Dict mapping horizon -> DataFrame with columns:
            [predicted_return, lower_bound, upper_bound]
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        self._validate_features(X)
        predictions: dict[str, pd.DataFrame] = {}

        for horizon, model in self.models.items():
            pred_df = pd.DataFrame(index=X.index)
            pred_df["predicted_return"] = model.predict(X)

            if include_intervals and horizon in self.models_lower:
                pred_df["lower_bound"] = self.models_lower[horizon].predict(X)
                pred_df["upper_bound"] = self.models_upper[horizon].predict(X)
            else:
                pred_df["lower_bound"] = np.nan
                pred_df["upper_bound"] = np.nan

            predictions[horizon] = pred_df

        return predictions

    def predict_single(
        self,
        X: pd.DataFrame,
        horizon: str = "21d",
    ) -> pd.DataFrame:
        """Predict a single horizon. Convenience wrapper."""
        preds = self.predict(X)
        if horizon not in preds:
            raise ValueError(f"Horizon '{horizon}' not available. Trained: {list(preds.keys())}")
        return preds[horizon]

    def get_feature_importance(self, horizon: str = "21d", top_n: int = 20) -> pd.Series:
        """Return top feature importances for a given horizon."""
        if horizon not in self.feature_importances:
            raise ValueError(f"No importances for horizon '{horizon}'")
        return self.feature_importances[horizon].head(top_n)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Check that no target columns leaked into the features."""
        target_cols = [c for c in X.columns if c.startswith("target_")]
        if target_cols:
            raise ValueError(
                f"Target columns found in feature matrix: {target_cols}. "
                "These must be removed before training/prediction."
            )

    def save(self, path: Path) -> None:
        """Save model artifacts to disk."""
        import json
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        for horizon, model in self.models.items():
            with open(path / f"return_forecaster_{horizon}.pkl", "wb") as f:
                pickle.dump(model, f)
            if horizon in self.models_lower:
                with open(path / f"return_forecaster_{horizon}_lower.pkl", "wb") as f:
                    pickle.dump(self.models_lower[horizon], f)
            if horizon in self.models_upper:
                with open(path / f"return_forecaster_{horizon}_upper.pkl", "wb") as f:
                    pickle.dump(self.models_upper[horizon], f)

        # Save metadata
        meta = {
            **self.training_metadata,
            "feature_names": self.feature_names,
            "config": {
                "horizons": self.config.horizons,
                "xgb_params": {k: v for k, v in self.config.xgb_params.items()
                               if isinstance(v, (int, float, str, bool))},
            },
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved return forecaster to {}", path)

    @classmethod
    def load(cls, path: Path) -> "ReturnForecaster":
        """Load a saved model from disk."""
        import json
        import pickle

        path = Path(path)
        model = cls()

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        model.feature_names = meta.get("feature_names", [])
        model.training_metadata = meta

        for horizon in meta.get("horizons", []):
            pkl_path = path / f"return_forecaster_{horizon}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    model.models[horizon] = pickle.load(f)

            lower_path = path / f"return_forecaster_{horizon}_lower.pkl"
            if lower_path.exists():
                with open(lower_path, "rb") as f:
                    model.models_lower[horizon] = pickle.load(f)

            upper_path = path / f"return_forecaster_{horizon}_upper.pkl"
            if upper_path.exists():
                with open(upper_path, "rb") as f:
                    model.models_upper[horizon] = pickle.load(f)

        model._fitted = bool(model.models)
        logger.info("Loaded return forecaster from {} ({} horizons)", path, len(model.models))
        return model
