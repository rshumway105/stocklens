"""
Fair Value Estimator — Model 2.

Predicts a fundamentals-derived "fair price" using only fundamental and
macroeconomic features (excludes current market price and technical indicators).

The target is the 63-day moving average of price, which acts as a smoothed
"what the price should be" anchor.  The valuation gap between current price
and estimated fair value is the key signal.

Design:
- Only uses fundamental + macro features (no technicals, no current price).
- This ensures the model captures intrinsic value rather than momentum.
- Output: fair value estimate + valuation gap percentage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger


# ---------------------------------------------------------------------------
# Feature filters — which columns the fair value model is allowed to see
# ---------------------------------------------------------------------------

# Prefixes for features the fair value model CAN use
ALLOWED_PREFIXES = (
    "fund_",              # fundamental metrics and z-scores
    "composite_",         # composite fundamental scores
    "analyst_",           # analyst signals
    # Macro features
    "fed_funds_", "treasury_", "yield_curve_", "cpi_", "core_cpi_",
    "pce_", "unemployment_", "initial_claims_", "nonfarm_",
    "real_gdp_", "ism_", "consumer_confidence_",
    "vix_", "credit_spread_", "usd_index_",
    "real_rate_", "real_fed_funds_", "financial_stress_",
)

# Columns that are explicitly EXCLUDED (even if they match a prefix)
EXCLUDED_COLUMNS = {
    "Close", "Volume", "Open", "High", "Low", "Adj Close",
}

# Patterns to exclude (technical indicators that sneak in)
EXCLUDED_PATTERNS = (
    "sma_", "ema_", "rsi_", "macd", "stochastic_", "williams_",
    "bb_", "atr_", "hvol_", "vol_ratio_", "obv", "return_",
    "drawdown_", "distance_from_", "intraday_", "gap",
    "price_to_sma_", "sma_cross_",
    # Sentiment (could include but keeping FV model purely fundamental)
    "news_", "social_", "combined_sentiment", "sentiment_",
)


def filter_fair_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a feature DataFrame to only include columns allowed
    for the fair value model.

    This prevents the model from learning price patterns or momentum —
    it should only see fundamental value and macro context.

    Args:
        df: Full feature DataFrame.

    Returns:
        Filtered DataFrame with only fundamental + macro columns.
    """
    allowed = []

    for col in df.columns:
        # Skip explicit exclusions
        if col in EXCLUDED_COLUMNS:
            continue

        # Skip target columns
        if col.startswith("target_"):
            continue

        # Skip technical/sentiment patterns
        if any(col.startswith(p) or col.endswith(p) for p in EXCLUDED_PATTERNS):
            continue

        # Include if it matches an allowed prefix
        if any(col.startswith(p) for p in ALLOWED_PREFIXES):
            allowed.append(col)

    logger.info(
        "Fair value feature filter: {} → {} columns",
        len(df.columns), len(allowed),
    )
    return df[allowed]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FairValueConfig:
    """Configuration for the fair value estimator."""

    xgb_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 15,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
    })

    # Valuation gap thresholds
    overvalued_threshold: float = 0.15    # +15% above fair value
    undervalued_threshold: float = -0.15  # -15% below fair value

    # Minimum training samples
    min_train_samples: int = 500


# ---------------------------------------------------------------------------
# Fair Value Estimator
# ---------------------------------------------------------------------------

class FairValueEstimator:
    """
    XGBoost model that estimates fair value from fundamentals + macro.

    The model predicts the smoothed (63-day MA) price level.
    The valuation gap = (current_price - fair_value) / fair_value.
    """

    def __init__(self, config: Optional[FairValueConfig] = None):
        self.config = config or FairValueConfig()
        self.model: Any = None
        self.feature_names: list[str] = []
        self.feature_importances: pd.Series = pd.Series(dtype=float)
        self.training_metadata: dict[str, Any] = {}
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "FairValueEstimator":
        """
        Train the fair value model.

        Args:
            X: Feature matrix (fundamental + macro features only).
            y: Target — smoothed price (63-day MA).

        Returns:
            self (fitted model).
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            raise

        # Drop NaN targets
        mask = y.notna()
        X_train = X.loc[mask]
        y_train = y.loc[mask]

        # Also drop rows where all features are NaN
        valid_rows = X_train.notna().any(axis=1)
        X_train = X_train.loc[valid_rows]
        y_train = y_train.loc[valid_rows]

        if len(X_train) < self.config.min_train_samples:
            logger.warning(
                "Only {} samples (need {}) — fair value model not trained",
                len(X_train), self.config.min_train_samples,
            )
            return self

        self.feature_names = list(X_train.columns)

        logger.info(
            "Training fair value estimator — {} samples, {} features",
            len(X_train), len(self.feature_names),
        )

        # Fill remaining NaN in features with column median
        X_train = X_train.fillna(X_train.median())

        self.model = xgb.XGBRegressor(**self.config.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )

        self.feature_importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        self._fitted = True
        self.training_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": len(X_train),
            "n_features": len(self.feature_names),
        }

        logger.info("Fair value estimator trained successfully")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fair value and compute valuation gap.

        Args:
            X: Feature matrix (same columns as training).

        Returns:
            DataFrame with columns: [fair_value, valuation_gap, signal].
            valuation_gap = (current_price - fair_value) / fair_value
            signal: "overvalued", "undervalued", or "fairly_valued"
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X_pred = X[self.feature_names].copy()
        X_pred = X_pred.fillna(X_pred.median())

        result = pd.DataFrame(index=X.index)
        result["fair_value"] = self.model.predict(X_pred)

        return result

    def compute_valuation_gap(
        self,
        fair_value: pd.Series,
        current_price: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute valuation gap and signal from fair value and current price.

        Args:
            fair_value: Predicted fair value series.
            current_price: Actual current price series.

        Returns:
            DataFrame with [valuation_gap, signal].
        """
        result = pd.DataFrame(index=fair_value.index)

        fv = fair_value.replace(0, np.nan)
        result["valuation_gap"] = (current_price - fv) / fv

        # Classify signal
        result["signal"] = "fairly_valued"
        result.loc[
            result["valuation_gap"] > self.config.overvalued_threshold,
            "signal"
        ] = "overvalued"
        result.loc[
            result["valuation_gap"] < self.config.undervalued_threshold,
            "signal"
        ] = "undervalued"

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Return top feature importances."""
        return self.feature_importances.head(top_n)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import json
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            with open(path / "fair_value_estimator.pkl", "wb") as f:
                pickle.dump(self.model, f)

        meta = {
            **self.training_metadata,
            "feature_names": self.feature_names,
            "config": {
                "overvalued_threshold": self.config.overvalued_threshold,
                "undervalued_threshold": self.config.undervalued_threshold,
            },
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved fair value estimator to {}", path)

    @classmethod
    def load(cls, path: Path) -> "FairValueEstimator":
        """Load model from disk."""
        import json
        import pickle

        path = Path(path)
        model = cls()

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        model.feature_names = meta.get("feature_names", [])
        model.training_metadata = meta

        pkl_path = path / "fair_value_estimator.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                model.model = pickle.load(f)
            model._fitted = True

        return model
