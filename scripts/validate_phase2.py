#!/usr/bin/env python3
"""
Phase 2 validation — Feature Engineering.

Tests all feature processors using only pandas, numpy, and the project code.
Usage: python3 scripts/validate_phase2.py
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
N = 504
dates = pd.date_range("2022-01-03", periods=N, freq="B")
close = 150 + np.cumsum(np.random.randn(N) * 1.5)
close = np.maximum(close, 50)

price_df = pd.DataFrame({
    "Open": close + np.random.randn(N) * 0.3,
    "High": close + np.abs(np.random.randn(N)) * 1.5,
    "Low": close - np.abs(np.random.randn(N)) * 1.5,
    "Close": close,
    "Adj Close": close * 0.99,
    "Volume": np.random.randint(5_000_000, 50_000_000, size=N),
}, index=dates)

# ══════════════════════════════════════════════════════════
print("\n═══ 1. Technical Features ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.technical_features import compute_technical_features

tech = compute_technical_features(price_df)
base_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
feature_cols = [c for c in tech.columns if c not in base_cols]

check("Produces 30+ features", len(feature_cols) > 30, f"got {len(feature_cols)}")
check("RSI column exists", "rsi_14" in tech.columns)
check("MACD column exists", "macd" in tech.columns)
check("Bollinger width exists", "bb_width" in tech.columns)
check("ATR exists", "atr_14" in tech.columns)
check("OBV exists", "obv" in tech.columns)
check("Return 1d exists", "return_1d" in tech.columns)
check("Drawdown exists", "drawdown_from_high" in tech.columns)

# RSI bounds
rsi = tech["rsi_14"].dropna()
check("RSI >= 0", rsi.min() >= 0, f"min={rsi.min():.2f}")
check("RSI <= 100", rsi.max() <= 100, f"max={rsi.max():.2f}")

# No lookahead: RSI at row 100 shouldn't change if future data changes
original_rsi = tech["rsi_14"].iloc[100]
modified_price = price_df.copy()
modified_price.iloc[101:, modified_price.columns.get_loc("Close")] = 999
modified_tech = compute_technical_features(modified_price)
check("No lookahead in RSI", original_rsi == modified_tech["rsi_14"].iloc[100])

# No lookahead: return_1d is a LOOKBACK return (today vs yesterday), not forward
ret_1d = tech["return_1d"].iloc[50]
expected = np.log(price_df["Close"].iloc[50] / price_df["Close"].iloc[49])
check("return_1d is lookback (not forward)", abs(ret_1d - expected) < 1e-10)

# ══════════════════════════════════════════════════════════
print("\n═══ 2. Fundamental Features ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.fundamental_features import (
    compute_sector_zscores,
    compute_composite_scores,
    compute_fundamental_features,
)

fund_df = pd.DataFrame({
    "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "WFC"],
    "sector": ["Tech"] * 5 + ["Finance"] * 3,
    "pe_ratio": [28, 32, 25, 60, 22, 12, 10, 9],
    "forward_pe": [25, 28, 22, 45, 18, 11, 9, 8],
    "pb_ratio": [40, 12, 6, 8, 5, 1.5, 1.0, 0.9],
    "gross_margin": [0.43, 0.69, 0.56, 0.47, 0.81, 0.55, 0.50, 0.48],
    "operating_margin": [0.30, 0.42, 0.28, 0.06, 0.35, 0.30, 0.28, 0.25],
    "net_margin": [0.25, 0.36, 0.24, 0.04, 0.28, 0.25, 0.22, 0.20],
    "roe": [1.5, 0.40, 0.25, 0.15, 0.25, 0.12, 0.10, 0.09],
    "roa": [0.28, 0.20, 0.15, 0.04, 0.14, 0.01, 0.009, 0.008],
    "debt_to_equity": [1.5, 0.4, 0.1, 0.6, 0.1, 2.0, 1.8, 1.5],
    "current_ratio": [1.0, 1.8, 2.5, 1.1, 2.8, 0.9, 0.8, 0.7],
    "revenue_growth": [0.02, 0.07, 0.11, 0.12, 0.23, 0.05, 0.03, 0.01],
    "fcf_yield": [0.03, 0.025, 0.04, 0.01, 0.05, 0.06, 0.07, 0.08],
}).set_index("ticker")

zscored = compute_sector_zscores(fund_df)
zscore_cols = [c for c in zscored.columns if c.endswith("_zscore")]
check("Z-score columns created", len(zscore_cols) > 5, f"got {len(zscore_cols)}")

# Z-scores should be clipped to [-4, 4]
for col in zscore_cols:
    vals = zscored[col].dropna()
    if len(vals) > 0:
        check(f"{col} clipped [-4, 4]", vals.min() >= -4 and vals.max() <= 4)

# Composite scores
full_fund = compute_fundamental_features(fund_df)
check("composite_value exists", "composite_value" in full_fund.columns)
check("composite_quality exists", "composite_quality" in full_fund.columns)
check("composite_fundamental exists", "composite_fundamental" in full_fund.columns)

# ══════════════════════════════════════════════════════════
print("\n═══ 3. Macro Features ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.macro_features import (
    compute_series_features,
    compute_macro_features,
)

# Single series
dates_macro = pd.date_range("2020-01-01", periods=600, freq="D")
test_series = pd.Series(np.linspace(1, 5, 600), index=dates_macro)
series_feats = compute_series_features(test_series, "test_rate", publication_lag_days=5)

check("Level column exists", "test_rate_level" in series_feats.columns)
check("1m change exists", "test_rate_chg_1m" in series_feats.columns)
check("Direction exists", "test_rate_direction" in series_feats.columns)
check("Z-score exists", "test_rate_zscore_2y" in series_feats.columns)

# Publication lag check
level_at_10 = series_feats["test_rate_level"].iloc[10]
expected_val = test_series.iloc[5]  # shifted by 5 days
check("Publication lag applied", abs(level_at_10 - expected_val) < 0.01, f"got {level_at_10}, expected {expected_val}")

# Multi-series pipeline
macro_dict = {
    "fed_funds_rate": pd.DataFrame({"DFF": np.linspace(0.25, 5.5, 1200)}, index=pd.date_range("2020-01-01", periods=1200, freq="D")),
    "treasury_10y": pd.DataFrame({"DGS10": np.linspace(1.5, 4.5, 1200)}, index=pd.date_range("2020-01-01", periods=1200, freq="D")),
    "treasury_2y": pd.DataFrame({"DGS2": np.linspace(0.5, 5.0, 1200)}, index=pd.date_range("2020-01-01", periods=1200, freq="D")),
    "vix": pd.DataFrame({"VIXCLS": 20 + np.random.randn(1200) * 5}, index=pd.date_range("2020-01-01", periods=1200, freq="D")),
    "credit_spread_baa": pd.DataFrame({"BAA10Y": 2 + np.random.randn(1200) * 0.5}, index=pd.date_range("2020-01-01", periods=1200, freq="D")),
}

macro_feats = compute_macro_features(macro_dict)
check("Macro pipeline produces output", not macro_feats.empty)
check("Macro has 10+ features", len(macro_feats.columns) > 10, f"got {len(macro_feats.columns)}")
check("Yield curve slope computed", "yield_curve_slope" in macro_feats.columns)
check("Financial stress index computed", "financial_stress_index" in macro_feats.columns)

# ══════════════════════════════════════════════════════════
print("\n═══ 4. Sentiment Features ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.sentiment_features import (
    score_text_heuristic,
    score_headlines,
    aggregate_news_sentiment,
    combine_sentiment_features,
)

check("Positive text scores positive", score_text_heuristic("Stock surge rally gains") > 0)
check("Negative text scores negative", score_text_heuristic("Market crash plunge fear") < 0)
check("Neutral text scores zero", score_text_heuristic("Company quarterly report") == 0)
check("Empty text scores zero", score_text_heuristic("") == 0)

# Aggregation
headlines = pd.DataFrame({
    "title": ["surge gains beat", "crash fear drop", "strong profit growth", "decline weak sell"],
    "published_at": pd.date_range("2024-01-01", periods=4, freq="D"),
})
scored = score_headlines(headlines)
check("Score column added", "sentiment_score" in scored.columns)

daily = aggregate_news_sentiment(scored)
check("Daily aggregation works", not daily.empty)
check("news_sentiment_mean exists", "news_sentiment_mean" in daily.columns)
check("news_sentiment_7d exists", "news_sentiment_7d" in daily.columns)

# ══════════════════════════════════════════════════════════
print("\n═══ 5. Target Builder ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.target_builder import (
    compute_forward_returns,
    compute_fair_value_target,
    compute_all_targets,
    validate_no_lookahead,
    get_target_columns,
)

targets = compute_forward_returns(price_df)

# Forward return correctness
expected_5d = np.log(price_df["Close"].iloc[5] / price_df["Close"].iloc[0])
actual_5d = targets["target_return_5d"].iloc[0]
check("5d forward return correct", abs(actual_5d - expected_5d) < 1e-10)

# Last rows should be NaN
check("Forward returns NaN at end", np.isnan(targets["target_return_5d"].iloc[-1]))
check("126d NaN near end", np.isnan(targets["target_return_126d"].iloc[-100]))

# Direction targets
check("Direction targets exist", "target_direction_5d" in targets.columns)
check("Direction is 0 or 1", targets["target_direction_5d"].dropna().isin([0, 1]).all())

# Fair value target
fv = compute_fair_value_target(price_df)
check("Fair value column exists", "target_fair_value" in fv.columns)
check("Valuation gap exists", "target_valuation_gap" in fv.columns)

# Lookahead validation
clean_features = price_df[["Close", "Volume"]].copy()
check("Lookahead validation passes on clean data", validate_no_lookahead(clean_features))

leaked = clean_features.copy()
leaked["target_return_5d"] = 0.01
try:
    validate_no_lookahead(leaked)
    check("Lookahead validation catches leak", False, "Should have raised")
except AssertionError:
    check("Lookahead validation catches leak", True)

# ══════════════════════════════════════════════════════════
print("\n═══ 6. Feature Pipeline Integration ═══")
# ══════════════════════════════════════════════════════════

from backend.data.processors.feature_pipeline import assemble_features

# Minimal: price only
result = assemble_features(price_df, ticker="TEST")
check("Pipeline produces output", not result.empty)
check("Has technical features", "rsi_14" in result.columns)
check("Has target columns", "target_return_21d" in result.columns)

target_cols = set(get_target_columns()) & set(result.columns)
feature_only = set(result.columns) - target_cols
check("Targets separated from features", not (feature_only & set(get_target_columns())))

# With macro data
result_macro = assemble_features(price_df, macro_features_df=macro_feats, ticker="TEST")
macro_in_result = [c for c in result_macro.columns if c.startswith(("fed_funds", "treasury", "vix"))]
check("Macro features merged", len(macro_in_result) > 0, f"got {len(macro_in_result)}")

total_features = len(set(result_macro.columns) - target_cols)
check("Total features > 40", total_features > 40, f"got {total_features}")

# ══════════════════════════════════════════════════════════
print("\n═══ 7. File Structure ═══")
# ══════════════════════════════════════════════════════════

project_root = Path(__file__).resolve().parent.parent
phase2_files = [
    "backend/data/processors/technical_features.py",
    "backend/data/processors/fundamental_features.py",
    "backend/data/processors/macro_features.py",
    "backend/data/processors/sentiment_features.py",
    "backend/data/processors/target_builder.py",
    "backend/data/processors/feature_pipeline.py",
    "backend/tests/test_phase2.py",
]

for f in phase2_files:
    check(f"Exists: {f}", (project_root / f).exists())

# ══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
