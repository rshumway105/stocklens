"""
SHAP Explainer — feature importance and per-prediction explanations.

Wraps SHAP's TreeExplainer to produce:
- Global feature importance rankings
- Per-prediction SHAP waterfall data (top N drivers)
- Plain-English feature explanations

When SHAP is not installed, falls back to the model's built-in
feature_importances_ (which is less informative but functional).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger


class ModelExplainer:
    """
    Produces SHAP-based explanations for tree model predictions.

    Falls back to model.feature_importances_ if SHAP is not available.
    """

    def __init__(self, model: Any, feature_names: list[str]):
        """
        Args:
            model: A fitted tree model (XGBoost, LightGBM, etc.).
            feature_names: List of feature column names.
        """
        self.model = model
        self.feature_names = feature_names
        self._shap_available = False
        self._explainer: Any = None

        try:
            import shap
            self._explainer = shap.TreeExplainer(model)
            self._shap_available = True
            logger.info("SHAP TreeExplainer initialized ({} features)", len(feature_names))
        except ImportError:
            logger.warning("SHAP not installed — using built-in feature importances")
        except Exception as e:
            logger.warning("SHAP initialization failed ({}), using fallback", e)

    def explain_prediction(
        self,
        X_row: pd.DataFrame,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Explain a single prediction with the top contributing features.

        Args:
            X_row: Single-row DataFrame (one observation to explain).
            top_n: Number of top features to include.

        Returns:
            List of dicts, each with:
            - feature: feature name
            - shap_value: contribution magnitude (positive = pushes prediction up)
            - feature_value: actual feature value for this observation
            - direction: "positive" or "negative"
        """
        if X_row.shape[0] != 1:
            X_row = X_row.iloc[[0]]

        if self._shap_available and self._explainer is not None:
            return self._explain_with_shap(X_row, top_n)
        else:
            return self._explain_with_importance(X_row, top_n)

    def _explain_with_shap(
        self,
        X_row: pd.DataFrame,
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Use SHAP TreeExplainer for detailed explanation."""
        import shap

        shap_values = self._explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            # Multi-output; take first
            shap_values = shap_values[0]

        if len(shap_values.shape) > 1:
            sv = shap_values[0]
        else:
            sv = shap_values

        # Build explanation list sorted by absolute SHAP value
        explanations = []
        indices = np.argsort(np.abs(sv))[::-1][:top_n]

        for idx in indices:
            feat_name = self.feature_names[idx]
            feat_val = float(X_row.iloc[0, idx])
            shap_val = float(sv[idx])

            explanations.append({
                "feature": feat_name,
                "shap_value": round(shap_val, 6),
                "feature_value": round(feat_val, 4) if not np.isnan(feat_val) else None,
                "direction": "positive" if shap_val > 0 else "negative",
                "explanation": self._generate_explanation(feat_name, feat_val, shap_val),
            })

        return explanations

    def _explain_with_importance(
        self,
        X_row: pd.DataFrame,
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Fallback: use model's feature_importances_ attribute."""
        if not hasattr(self.model, "feature_importances_"):
            return []

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        explanations = []
        for idx in indices:
            feat_name = self.feature_names[idx]
            feat_val = float(X_row.iloc[0, idx]) if idx < X_row.shape[1] else np.nan

            explanations.append({
                "feature": feat_name,
                "importance": round(float(importances[idx]), 6),
                "feature_value": round(feat_val, 4) if not np.isnan(feat_val) else None,
                "direction": "unknown",  # importance doesn't tell direction
                "explanation": f"{feat_name} is a top driver (importance: {importances[idx]:.4f})",
            })

        return explanations

    def global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return global feature importance as a DataFrame.

        Uses SHAP mean absolute values if available, otherwise model importances.
        """
        if hasattr(self.model, "feature_importances_"):
            imp = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names,
            ).sort_values(ascending=False)

            return pd.DataFrame({
                "feature": imp.index[:top_n],
                "importance": imp.values[:top_n],
            })

        return pd.DataFrame(columns=["feature", "importance"])

    def _generate_explanation(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
    ) -> str:
        """
        Generate a plain-English explanation for a feature's contribution.

        Translates feature names and values into human-readable statements.
        """
        direction = "pushing the prediction higher" if shap_value > 0 else "pushing the prediction lower"

        # Map feature names to readable descriptions
        readable_names = {
            "rsi_14": "RSI (14-day momentum)",
            "macd": "MACD trend signal",
            "bb_position": "Bollinger Band position",
            "atr_pct": "volatility (ATR %)",
            "hvol_20": "20-day historical volatility",
            "return_21d": "1-month past return",
            "return_63d": "3-month past return",
            "drawdown_from_high": "drawdown from 52-week high",
            "price_to_sma_50": "price vs 50-day average",
            "price_to_sma_200": "price vs 200-day average",
            "vol_ratio_20": "volume vs 20-day average",
        }

        # Z-score features
        if feature_name.endswith("_zscore"):
            base = feature_name.replace("_zscore", "").replace("fund_", "")
            readable = base.replace("_", " ").title()
            if not np.isnan(feature_value):
                return (
                    f"{readable} is {abs(feature_value):.1f} std devs "
                    f"{'above' if feature_value > 0 else 'below'} sector median, "
                    f"{direction}"
                )
            return f"{readable} z-score is {direction}"

        # Macro features
        if feature_name.endswith("_level"):
            base = feature_name.replace("_level", "").replace("_", " ").title()
            return f"{base} at {feature_value:.2f}, {direction}"

        if feature_name.endswith("_direction"):
            base = feature_name.replace("_direction", "").replace("_", " ").title()
            trend = "rising" if feature_value > 0 else "falling" if feature_value < 0 else "flat"
            return f"{base} is {trend}, {direction}"

        # Known features
        readable = readable_names.get(feature_name, feature_name.replace("_", " "))
        if not np.isnan(feature_value):
            return f"{readable} at {feature_value:.3f}, {direction}"
        return f"{readable} is {direction}"
