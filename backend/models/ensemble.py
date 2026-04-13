"""
Ensemble — blends Return Forecaster and Fair Value Estimator signals.

Combines predictions from both models into a unified signal with
a composite confidence score (0-100).

Signal logic:
- If both models agree (e.g., predicted negative returns AND overvalued),
  the signal is stronger and confidence is higher.
- If models disagree, confidence is lower and the signal reflects
  the stronger of the two.

Confidence score factors:
1. Model agreement (both bullish/bearish = higher confidence)
2. Prediction interval width (tighter = more confident)
3. Magnitude of valuation gap (larger = more confident)
4. Number of supporting features (via SHAP)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble."""

    # Weight of return forecaster vs fair value in the final signal
    return_weight: float = 0.5
    fair_value_weight: float = 0.5

    # Confidence score weights
    agreement_weight: float = 0.35    # how much model agreement matters
    interval_weight: float = 0.25     # how much prediction interval width matters
    magnitude_weight: float = 0.25    # how much signal magnitude matters
    historical_weight: float = 0.15   # how much backtest accuracy matters

    # Thresholds
    overvalued_threshold: float = 0.15
    undervalued_threshold: float = -0.15


class EnsembleModel:
    """
    Combines return forecaster and fair value estimator into a unified signal.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def combine_predictions(
        self,
        return_predictions: Optional[pd.DataFrame] = None,
        fair_value_predictions: Optional[pd.DataFrame] = None,
        current_prices: Optional[pd.Series] = None,
        horizon: str = "21d",
    ) -> pd.DataFrame:
        """
        Combine model predictions into a unified signal.

        Args:
            return_predictions: DataFrame with [predicted_return, lower_bound, upper_bound]
                from the return forecaster.
            fair_value_predictions: DataFrame with [fair_value] from the FV estimator.
            current_prices: Current price series (for valuation gap calculation).
            horizon: Which return horizon to use.

        Returns:
            DataFrame with columns:
            - predicted_return: blended return forecast
            - fair_value: estimated fair value
            - valuation_gap: (price - fair_value) / fair_value
            - signal: "overvalued" / "undervalued" / "fairly_valued"
            - confidence: 0-100 composite score
            - lower_bound, upper_bound: prediction interval
        """
        result = pd.DataFrame()

        # --- Return forecaster signal ---
        has_returns = return_predictions is not None and not return_predictions.empty
        if has_returns:
            result["predicted_return"] = return_predictions["predicted_return"]
            result["lower_bound"] = return_predictions.get("lower_bound", np.nan)
            result["upper_bound"] = return_predictions.get("upper_bound", np.nan)
            result["return_signal"] = np.where(
                result["predicted_return"] > 0, 1, -1
            )
        else:
            result["predicted_return"] = np.nan
            result["return_signal"] = 0

        # --- Fair value signal ---
        has_fv = (
            fair_value_predictions is not None
            and not fair_value_predictions.empty
            and current_prices is not None
        )
        if has_fv:
            result["fair_value"] = fair_value_predictions["fair_value"]
            fv = result["fair_value"].replace(0, np.nan)

            # Align current prices with result index
            if isinstance(current_prices, pd.Series):
                aligned_prices = current_prices.reindex(result.index)
            else:
                aligned_prices = current_prices

            result["valuation_gap"] = (aligned_prices - fv) / fv
            result["fv_signal"] = np.where(
                result["valuation_gap"] > self.config.overvalued_threshold, -1,
                np.where(
                    result["valuation_gap"] < self.config.undervalued_threshold, 1, 0
                )
            )
        else:
            result["fair_value"] = np.nan
            result["valuation_gap"] = np.nan
            result["fv_signal"] = 0

        # --- Combined signal ---
        result["combined_score"] = (
            self.config.return_weight * result["return_signal"] +
            self.config.fair_value_weight * result["fv_signal"]
        )

        result["signal"] = "fairly_valued"
        result.loc[result["combined_score"] > 0.3, "signal"] = "undervalued"
        result.loc[result["combined_score"] < -0.3, "signal"] = "overvalued"

        # --- Confidence score (0-100) ---
        result["confidence"] = self._compute_confidence(result)

        # Clean up intermediate columns
        result = result.drop(columns=["return_signal", "fv_signal", "combined_score"], errors="ignore")

        logger.info(
            "Ensemble: {} predictions generated",
            len(result.dropna(subset=["predicted_return"])),
        )

        return result

    def _compute_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a 0-100 confidence score based on multiple factors.

        Higher confidence when:
        - Both models agree on direction
        - Prediction interval is narrow
        - Valuation gap magnitude is large
        """
        scores = pd.DataFrame(index=df.index)

        # Factor 1: Model agreement (0 or 1)
        if "return_signal" in df.columns and "fv_signal" in df.columns:
            # Agreement: both positive, both negative, or both neutral
            same_sign = (
                (df["return_signal"] * df["fv_signal"] > 0) |
                (df["return_signal"] == 0) & (df["fv_signal"] == 0)
            )
            scores["agreement"] = same_sign.astype(float)
        else:
            scores["agreement"] = 0.5

        # Factor 2: Prediction interval width (narrower = more confident)
        if "lower_bound" in df.columns and "upper_bound" in df.columns:
            interval_width = (df["upper_bound"] - df["lower_bound"]).abs()
            # Normalize: width of 0.05 (5%) = high confidence, 0.20 (20%) = low
            scores["interval"] = 1 - (interval_width.clip(0, 0.3) / 0.3)
            scores["interval"] = scores["interval"].fillna(0.5)
        else:
            scores["interval"] = 0.5

        # Factor 3: Signal magnitude (stronger signal = more confident)
        if "valuation_gap" in df.columns:
            gap_mag = df["valuation_gap"].abs()
            # Normalize: gap of 0.30 (30%) = max confidence
            scores["magnitude"] = (gap_mag.clip(0, 0.3) / 0.3)
            scores["magnitude"] = scores["magnitude"].fillna(0)
        else:
            scores["magnitude"] = 0.5

        # Weighted average → scale to 0-100
        confidence = (
            self.config.agreement_weight * scores["agreement"] +
            self.config.interval_weight * scores["interval"] +
            self.config.magnitude_weight * scores["magnitude"] +
            self.config.historical_weight * 0.5  # placeholder for backtest accuracy
        )

        return (confidence * 100).clip(0, 100).round(1)
