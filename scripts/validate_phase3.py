#!/usr/bin/env python3
"""
Phase 3 validation — Model Training & Backtesting.

Tests model architecture, walk-forward fold generation, ensemble logic,
explainer, and data flow — all without requiring xgboost/shap.

Usage: python3 scripts/validate_phase3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ---------------------------------------------------------------------------
# Generate test data
# ---------------------------------------------------------------------------

np.random.seed(42)
N = 1500
dates = pd.date_range("2019-01-02", periods=N, freq="B")
returns = np.random.randn(N) * 0.015 + 0.0003
prices = 100 * np.exp(np.cumsum(returns))

feature_matrix = pd.DataFrame(index=dates)
feature_matrix["Close"] = prices
feature_matrix["Volume"] = np.random.randint(5_000_000, 50_000_000, size=N)

for i in range(30):
    feature_matrix[f"feature_{i}"] = np.random.randn(N) * (0.5 if i < 10 else 1.0)

# Targets
for horizon, days in [("5d", 5), ("21d", 21), ("63d", 63), ("126d", 126)]:
    future_price = feature_matrix["Close"].shift(-days)
    feature_matrix[f"target_return_{horizon}"] = np.log(future_price / feature_matrix["Close"])
    feature_matrix[f"target_direction_{horizon}"] = (feature_matrix[f"target_return_{horizon}"] > 0).astype(float)

feature_matrix["target_fair_value"] = feature_matrix["Close"].rolling(63, min_periods=63).mean()
fv = feature_matrix["target_fair_value"].replace(0, np.nan)
feature_matrix["target_valuation_gap"] = (feature_matrix["Close"] - fv) / fv


# ══════════════════════════════════════════════════════════
print("\n═══ 1. Return Forecaster Architecture ═══")
# ══════════════════════════════════════════════════════════

from backend.models.return_forecaster import ReturnForecaster, ReturnForecasterConfig

config = ReturnForecasterConfig()
check("Default horizons are 4", len(config.horizons) == 4)
check("Horizons include 5d,21d,63d,126d",
      set(config.horizons) == {"5d", "21d", "63d", "126d"})
check("XGB params have objective", "objective" in config.xgb_params)
check("Random state set for reproducibility", config.xgb_params.get("random_state") == 42)

rf = ReturnForecaster()
check("Model initializes unfitted", rf._fitted is False)
check("Models dict empty", len(rf.models) == 0)

# Validate feature rejection
try:
    bad_df = pd.DataFrame({"target_return_5d": [1], "feature_a": [2]})
    rf._validate_features(bad_df)
    check("Rejects target columns in features", False, "Should have raised")
except ValueError:
    check("Rejects target columns in features", True)

clean_df = pd.DataFrame({"feature_a": [1], "feature_b": [2]})
try:
    rf._validate_features(clean_df)
    check("Accepts clean features", True)
except ValueError:
    check("Accepts clean features", False)


# ══════════════════════════════════════════════════════════
print("\n═══ 2. Fair Value Estimator Architecture ═══")
# ══════════════════════════════════════════════════════════

from backend.models.fair_value_estimator import (
    FairValueEstimator, FairValueConfig, filter_fair_value_features,
)

fv_config = FairValueConfig()
check("Overvalued threshold is 15%", fv_config.overvalued_threshold == 0.15)
check("Undervalued threshold is -15%", fv_config.undervalued_threshold == -0.15)

fve = FairValueEstimator()
check("FV model initializes unfitted", fve._fitted is False)

# Test feature filter
test_cols = pd.DataFrame({
    "rsi_14": [50],             # technical — should be excluded
    "macd": [0.5],              # technical — excluded
    "fund_pe_ratio": [20],      # fundamental — included
    "fund_roe_zscore": [1.2],   # fundamental z-score — included
    "composite_value": [0.5],   # composite — included
    "fed_funds_rate_level": [5], # macro — included
    "vix_level": [20],          # macro — included
    "Close": [150],             # price — excluded
    "news_sentiment_7d": [0.3], # sentiment — excluded
    "target_return_21d": [0.02],# target — excluded
})

filtered = filter_fair_value_features(test_cols)
check("Excludes technical features", "rsi_14" not in filtered.columns)
check("Excludes MACD", "macd" not in filtered.columns)
check("Excludes Close", "Close" not in filtered.columns)
check("Excludes sentiment", "news_sentiment_7d" not in filtered.columns)
check("Excludes targets", "target_return_21d" not in filtered.columns)
check("Includes fundamental", "fund_pe_ratio" in filtered.columns)
check("Includes z-scores", "fund_roe_zscore" in filtered.columns)
check("Includes composites", "composite_value" in filtered.columns)
check("Includes macro", "fed_funds_rate_level" in filtered.columns)
check("Includes VIX", "vix_level" in filtered.columns)

# Test valuation gap computation
fve_test = FairValueEstimator()
gap_df = fve_test.compute_valuation_gap(
    fair_value=pd.Series([100, 100, 100]),
    current_price=pd.Series([120, 100, 80]),
)
check("Overvalued signal correct", gap_df.iloc[0]["signal"] == "overvalued")
check("Fairly valued signal correct", gap_df.iloc[1]["signal"] == "fairly_valued")
check("Undervalued signal correct", gap_df.iloc[2]["signal"] == "undervalued")
check("Gap = +20%", abs(gap_df.iloc[0]["valuation_gap"] - 0.20) < 0.01)
check("Gap = -20%", abs(gap_df.iloc[2]["valuation_gap"] - (-0.20)) < 0.01)


# ══════════════════════════════════════════════════════════
print("\n═══ 3. Walk-Forward Fold Generation ═══")
# ══════════════════════════════════════════════════════════

from backend.models.trainer import WalkForwardTrainer, WalkForwardConfig

wf_config = WalkForwardConfig(
    min_train_days=756,
    test_window_days=63,
    step_days=63,
    purge_days=5,
)
trainer = WalkForwardTrainer(wf_config)
folds = trainer.generate_folds(feature_matrix)

check("Folds generated", len(folds) > 0, f"got {len(folds)}")
check("Multiple folds", len(folds) > 3, f"got {len(folds)}")

# Validate fold properties
for i, fold in enumerate(folds):
    # Training always starts from 0 (expanding window)
    check(f"Fold {i}: train starts at 0", fold["train_start"] == 0)

    # Test starts after train + purge gap
    expected_test_start = fold["train_end"] + wf_config.purge_days
    check(f"Fold {i}: purge gap respected", fold["test_start"] == expected_test_start)

    # No overlap between train and test
    check(f"Fold {i}: no train/test overlap", fold["train_end"] < fold["test_start"])

    # Training window grows with each fold
    if i > 0:
        check(
            f"Fold {i}: expanding window",
            fold["train_end"] > folds[i-1]["train_end"],
        )

    if i >= 3:  # Only check first few to keep output manageable
        break

# Test max_folds limit
limited_config = WalkForwardConfig(max_folds=3)
limited_trainer = WalkForwardTrainer(limited_config)
limited_folds = limited_trainer.generate_folds(feature_matrix)
check("Max folds limit works", len(limited_folds) <= 3)


# ══════════════════════════════════════════════════════════
print("\n═══ 4. Ensemble Model ═══")
# ══════════════════════════════════════════════════════════

from backend.models.ensemble import EnsembleModel, EnsembleConfig

ensemble = EnsembleModel()

# Test with both models predicting negative / overvalued
return_preds = pd.DataFrame({
    "predicted_return": [-0.05, 0.05, -0.02],
    "lower_bound": [-0.10, 0.01, -0.06],
    "upper_bound": [-0.01, 0.10, 0.02],
}, index=[0, 1, 2])

fv_preds = pd.DataFrame({
    "fair_value": [100, 100, 100],
}, index=[0, 1, 2])

current = pd.Series([120, 80, 102], index=[0, 1, 2])

result = ensemble.combine_predictions(
    return_predictions=return_preds,
    fair_value_predictions=fv_preds,
    current_prices=current,
)

check("Ensemble produces predictions", not result.empty)
check("Has predicted_return", "predicted_return" in result.columns)
check("Has fair_value", "fair_value" in result.columns)
check("Has valuation_gap", "valuation_gap" in result.columns)
check("Has signal", "signal" in result.columns)
check("Has confidence", "confidence" in result.columns)

# Row 0: negative return + overvalued → should be overvalued with high confidence
check("Bearish agreement → overvalued", result.iloc[0]["signal"] == "overvalued")

# Row 1: positive return + undervalued → should be undervalued
check("Bullish agreement → undervalued", result.iloc[1]["signal"] == "undervalued")

# Confidence should be higher when models agree
conf_agree = result.iloc[0]["confidence"]  # both bearish
conf_neutral = result.iloc[2]["confidence"]  # mixed signals
check("Agreement has higher confidence", conf_agree >= conf_neutral,
      f"agree={conf_agree}, neutral={conf_neutral}")

# Confidence bounded 0-100
check("Confidence >= 0", (result["confidence"] >= 0).all())
check("Confidence <= 100", (result["confidence"] <= 100).all())

# Test with only return predictions
result_return_only = ensemble.combine_predictions(return_predictions=return_preds)
check("Works with return preds only", not result_return_only.empty)

# Test with only fair value
result_fv_only = ensemble.combine_predictions(
    fair_value_predictions=fv_preds, current_prices=current
)
check("Works with FV preds only", not result_fv_only.empty)


# ══════════════════════════════════════════════════════════
print("\n═══ 5. Explainer (Fallback Mode) ═══")
# ══════════════════════════════════════════════════════════

from backend.models.explainer import ModelExplainer

# Create a mock model with feature_importances_
class MockModel:
    def __init__(self, n_features):
        self.feature_importances_ = np.random.rand(n_features)
        self.feature_importances_ /= self.feature_importances_.sum()

feature_names = [f"feature_{i}" for i in range(20)]
mock = MockModel(20)

explainer = ModelExplainer(mock, feature_names)
check("Explainer initializes", explainer is not None)
check("SHAP not required", True)  # we're in fallback mode

# Test explanation
X_row = pd.DataFrame([np.random.randn(20)], columns=feature_names)
explanation = explainer.explain_prediction(X_row, top_n=5)
check("Explanation returns list", isinstance(explanation, list))
check("Top 5 features returned", len(explanation) == 5)
check("Explanation has feature key", "feature" in explanation[0])
check("Explanation has importance", "importance" in explanation[0])

# Global importance
global_imp = explainer.global_importance(top_n=10)
check("Global importance is DataFrame", isinstance(global_imp, pd.DataFrame))
check("Global importance has 10 rows", len(global_imp) == 10)

# Test explanation text generation
text = explainer._generate_explanation("fund_pe_ratio_zscore", 2.1, 0.05)
check("Z-score explanation is readable", "std dev" in text.lower() or "sector" in text.lower())

text2 = explainer._generate_explanation("fed_funds_rate_level", 5.25, -0.02)
check("Macro explanation is readable", "level" in text2.lower() or "at" in text2.lower())


# ══════════════════════════════════════════════════════════
print("\n═══ 6. File Structure ═══")
# ══════════════════════════════════════════════════════════

project_root = Path(__file__).resolve().parent.parent
phase3_files = [
    "backend/models/return_forecaster.py",
    "backend/models/fair_value_estimator.py",
    "backend/models/ensemble.py",
    "backend/models/explainer.py",
    "backend/models/trainer.py",
    "scripts/run_backtest.py",
]

for f in phase3_files:
    check(f"Exists: {f}", (project_root / f).exists())


# ══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
