"""
Valuation report builder service.

Orchestrates data fetching, feature computation, model inference,
and explanation to produce a complete ValuationReport for a ticker.

This is the core service that the /api/reports endpoint calls.
It assembles all the pieces built in Phases 1-3 into a single,
user-facing report.
"""

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.log import logger
from backend.api.schemas import (
    FeatureExplanation,
    FundamentalWithZscore,
    ReturnForecast,
    RiskFlag,
    SentimentSummary,
    ValuationReport,
)


class ReportBuilder:
    """
    Builds a complete valuation report for a given ticker.

    Requires:
    - Price data (OHLCV)
    - Feature matrix (from Phase 2 pipeline)
    - Trained models (return forecaster + fair value estimator)
    - Optional: sentiment data, macro context
    """

    def __init__(
        self,
        return_forecaster: Any = None,
        fair_value_estimator: Any = None,
    ):
        self.rf = return_forecaster
        self.fve = fair_value_estimator

    def build_report(
        self,
        ticker: str,
        feature_matrix: pd.DataFrame,
        price_df: pd.DataFrame,
        fundamentals: Optional[dict] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        macro_snapshot: Optional[dict] = None,
        backtest_metrics: Optional[dict] = None,
        ticker_info: Optional[dict] = None,
    ) -> ValuationReport:
        """
        Build a complete valuation report.

        Args:
            ticker: Stock/ETF symbol.
            feature_matrix: Full feature matrix (from feature pipeline).
            price_df: Raw OHLCV price data.
            fundamentals: Raw fundamental data dict.
            sentiment_data: Sentiment features DataFrame.
            macro_snapshot: Current macro indicator values.
            backtest_metrics: Historical accuracy metrics.
            ticker_info: Basic ticker info (name, sector, etc.).

        Returns:
            ValuationReport with all sections populated.
        """
        logger.info("Building valuation report for {}", ticker)

        info = ticker_info or {}
        current_price = float(price_df["Close"].iloc[-1]) if not price_df.empty else None

        # --- Separate features from targets ---
        target_cols = [c for c in feature_matrix.columns if c.startswith("target_")]
        feature_cols = [c for c in feature_matrix.columns if c not in target_cols]
        X = feature_matrix[feature_cols]

        # Use the most recent row for prediction
        X_latest = X.iloc[[-1]]

        # --- Return forecasts ---
        forecasts = self._build_forecasts(X_latest)

        # --- Fair value ---
        fair_value = None
        valuation_gap = None
        signal = "unknown"
        confidence = None

        if self.fve is not None and self.fve._fitted and current_price is not None:
            fv_result = self._build_fair_value(X_latest, current_price)
            fair_value = fv_result.get("fair_value")
            valuation_gap = fv_result.get("valuation_gap")
            signal = fv_result.get("signal", "unknown")

        # --- Ensemble signal ---
        if forecasts and fair_value is not None:
            signal, confidence = self._compute_ensemble_signal(
                forecasts, fair_value, current_price
            )
        elif forecasts:
            # Return-only signal
            ret_21d = next((f for f in forecasts if f.horizon == "21d"), None)
            if ret_21d:
                signal = "undervalued" if ret_21d.predicted_return > 0.02 else (
                    "overvalued" if ret_21d.predicted_return < -0.02 else "fairly_valued"
                )
                confidence = 40.0  # lower confidence with single model

        # --- Feature explanations ---
        top_drivers = self._build_explanations(X_latest)

        # --- Fundamentals with z-scores ---
        fund_items = self._build_fundamentals_section(fundamentals)

        # --- Sentiment summary ---
        sentiment = self._build_sentiment_summary(sentiment_data)

        # --- Macro context ---
        macro_ctx = self._build_macro_context(macro_snapshot)

        # --- Risk flags ---
        risk_flags = self._detect_risk_flags(price_df, fundamentals, feature_matrix)

        report = ValuationReport(
            ticker=ticker.upper(),
            name=info.get("name", ""),
            sector=info.get("sector", ""),
            signal=signal,
            confidence=confidence,
            current_price=round(current_price, 2) if current_price else None,
            fair_value=round(fair_value, 2) if fair_value else None,
            valuation_gap_pct=round(valuation_gap * 100, 2) if valuation_gap else None,
            forecasts=forecasts,
            top_drivers=top_drivers,
            fundamentals=fund_items,
            sentiment=sentiment,
            macro_context=macro_ctx,
            risk_flags=risk_flags,
            historical_accuracy=backtest_metrics,
        )

        logger.info("Report built for {} — signal: {}, confidence: {}",
                     ticker, signal, confidence)
        return report

    def _build_forecasts(self, X_latest: pd.DataFrame) -> list[ReturnForecast]:
        """Generate return forecasts from the return forecaster model."""
        if self.rf is None or not self.rf._fitted:
            return []

        try:
            preds = self.rf.predict(X_latest, include_intervals=True)
            forecasts = []

            horizon_labels = {"5d": "1-week", "21d": "1-month", "63d": "3-month", "126d": "6-month"}

            for horizon, pred_df in preds.items():
                row = pred_df.iloc[0]
                forecasts.append(ReturnForecast(
                    horizon=horizon,
                    predicted_return=round(float(row["predicted_return"]), 6),
                    lower_bound=_safe_round(row.get("lower_bound")),
                    upper_bound=_safe_round(row.get("upper_bound")),
                ))

            return forecasts
        except Exception as e:
            logger.error("Failed to generate forecasts: {}", e)
            return []

    def _build_fair_value(
        self,
        X_latest: pd.DataFrame,
        current_price: float,
    ) -> dict:
        """Predict fair value and valuation gap."""
        try:
            from backend.models.fair_value_estimator import filter_fair_value_features

            X_fv = filter_fair_value_features(X_latest)
            pred_df = self.fve.predict(X_fv)
            fv = float(pred_df["fair_value"].iloc[0])

            gap_df = self.fve.compute_valuation_gap(
                fair_value=pd.Series([fv]),
                current_price=pd.Series([current_price]),
            )

            return {
                "fair_value": fv,
                "valuation_gap": float(gap_df["valuation_gap"].iloc[0]),
                "signal": gap_df["signal"].iloc[0],
            }
        except Exception as e:
            logger.error("Fair value prediction failed: {}", e)
            return {}

    def _compute_ensemble_signal(
        self,
        forecasts: list[ReturnForecast],
        fair_value: float,
        current_price: float,
    ) -> tuple[str, float]:
        """Combine return and FV signals into ensemble verdict."""
        from backend.models.ensemble import EnsembleModel

        # Get 21d forecast
        ret_21d = next((f for f in forecasts if f.horizon == "21d"), None)
        if ret_21d is None:
            return "unknown", 0.0

        return_preds = pd.DataFrame({
            "predicted_return": [ret_21d.predicted_return],
            "lower_bound": [ret_21d.lower_bound or np.nan],
            "upper_bound": [ret_21d.upper_bound or np.nan],
        })

        fv_preds = pd.DataFrame({"fair_value": [fair_value]})
        prices = pd.Series([current_price])

        ensemble = EnsembleModel()
        result = ensemble.combine_predictions(
            return_predictions=return_preds,
            fair_value_predictions=fv_preds,
            current_prices=prices,
        )

        signal = result["signal"].iloc[0]
        confidence = float(result["confidence"].iloc[0])
        return signal, confidence

    def _build_explanations(self, X_latest: pd.DataFrame) -> list[FeatureExplanation]:
        """Generate SHAP-based feature explanations."""
        if self.rf is None or not self.rf._fitted:
            return []

        try:
            from backend.models.explainer import ModelExplainer

            # Use the 21d model for explanations
            model = self.rf.models.get("21d")
            if model is None:
                model = next(iter(self.rf.models.values()), None)
            if model is None:
                return []

            explainer = ModelExplainer(model, self.rf.feature_names)
            raw = explainer.explain_prediction(X_latest, top_n=10)

            return [
                FeatureExplanation(
                    feature=item["feature"],
                    shap_value=item.get("shap_value"),
                    importance=item.get("importance"),
                    feature_value=item.get("feature_value"),
                    direction=item.get("direction", "unknown"),
                    explanation=item.get("explanation", ""),
                )
                for item in raw
            ]
        except Exception as e:
            logger.error("Explanation generation failed: {}", e)
            return []

    def _build_fundamentals_section(
        self,
        fundamentals: Optional[dict],
    ) -> list[FundamentalWithZscore]:
        """Build fundamentals list with z-score context."""
        if not fundamentals:
            return []

        metrics_to_show = [
            "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
            "gross_margin", "operating_margin", "net_margin", "roe", "roa",
            "revenue_growth", "debt_to_equity", "current_ratio", "fcf_yield",
        ]

        items = []
        for metric in metrics_to_show:
            val = fundamentals.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                items.append(FundamentalWithZscore(
                    metric=metric,
                    value=round(float(val), 4),
                    zscore=None,  # would come from sector comparison
                ))

        return items

    def _build_sentiment_summary(
        self,
        sentiment_data: Optional[pd.DataFrame],
    ) -> Optional[SentimentSummary]:
        """Build sentiment summary from latest data."""
        if sentiment_data is None or sentiment_data.empty:
            return None

        latest = sentiment_data.iloc[-1]

        news_sent = _safe_float(latest.get("news_sentiment_7d"))
        trend = "neutral"
        trend_val = _safe_float(latest.get("news_sentiment_trend"))
        if trend_val is not None:
            trend = "improving" if trend_val > 0.05 else ("worsening" if trend_val < -0.05 else "neutral")

        return SentimentSummary(
            news_sentiment=news_sent,
            news_trend=trend,
            social_sentiment=_safe_float(latest.get("social_sentiment_7d")),
            social_mention_volume=int(latest.get("social_mention_count", 0)) if "social_mention_count" in latest.index else None,
            combined_sentiment=_safe_float(latest.get("combined_sentiment")),
        )

    def _build_macro_context(
        self,
        macro_snapshot: Optional[dict],
    ) -> list[dict[str, Any]]:
        """Build macro context section."""
        if not macro_snapshot:
            return []

        context = []
        key_indicators = [
            ("fed_funds_rate_level", "Fed Funds Rate"),
            ("treasury_10y_level", "10Y Treasury"),
            ("yield_curve_slope", "Yield Curve (10Y-2Y)"),
            ("vix_level", "VIX"),
            ("cpi_yoy_level", "CPI YoY"),
            ("unemployment_rate_level", "Unemployment"),
        ]

        for key, label in key_indicators:
            if key in macro_snapshot:
                val = macro_snapshot[key]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    direction_key = key.replace("_level", "_direction").replace("_slope", "_direction")
                    direction_val = macro_snapshot.get(direction_key, 0)
                    direction = "rising" if direction_val > 0 else ("falling" if direction_val < 0 else "flat")

                    context.append({
                        "indicator": label,
                        "value": round(float(val), 3),
                        "direction": direction,
                    })

        return context

    def _detect_risk_flags(
        self,
        price_df: pd.DataFrame,
        fundamentals: Optional[dict],
        feature_matrix: pd.DataFrame,
    ) -> list[RiskFlag]:
        """Detect unusual conditions that warrant risk warnings."""
        flags = []

        if price_df.empty:
            return flags

        # High volatility
        if "hvol_20" in feature_matrix.columns:
            hvol = feature_matrix["hvol_20"].iloc[-1]
            if not np.isnan(hvol) and hvol > 0.40:
                flags.append(RiskFlag(
                    flag="high_volatility",
                    severity="warning",
                    description=f"Annualized volatility is {hvol:.0%}, above the 40% threshold",
                ))

        # Large drawdown
        if "drawdown_from_high" in feature_matrix.columns:
            dd = feature_matrix["drawdown_from_high"].iloc[-1]
            if not np.isnan(dd) and dd < -0.20:
                flags.append(RiskFlag(
                    flag="significant_drawdown",
                    severity="warning",
                    description=f"Price is {dd:.0%} below its 52-week high",
                ))

        # Low volume
        if "vol_ratio_20" in feature_matrix.columns:
            vol_r = feature_matrix["vol_ratio_20"].iloc[-1]
            if not np.isnan(vol_r) and vol_r < 0.5:
                flags.append(RiskFlag(
                    flag="low_liquidity",
                    severity="info",
                    description="Volume is less than half the 20-day average",
                ))

        # Extreme RSI
        if "rsi_14" in feature_matrix.columns:
            rsi = feature_matrix["rsi_14"].iloc[-1]
            if not np.isnan(rsi):
                if rsi > 80:
                    flags.append(RiskFlag(
                        flag="overbought",
                        severity="info",
                        description=f"RSI at {rsi:.0f} — technically overbought",
                    ))
                elif rsi < 20:
                    flags.append(RiskFlag(
                        flag="oversold",
                        severity="info",
                        description=f"RSI at {rsi:.0f} — technically oversold",
                    ))

        # High debt
        if fundamentals and fundamentals.get("debt_to_equity"):
            dte = fundamentals["debt_to_equity"]
            if isinstance(dte, (int, float)) and not np.isnan(dte) and dte > 300:
                flags.append(RiskFlag(
                    flag="high_leverage",
                    severity="warning",
                    description=f"Debt-to-equity ratio is {dte:.0f}%",
                ))

        return flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_round(val: Any, digits: int = 6) -> Optional[float]:
    """Round a value, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, digits)
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return None
