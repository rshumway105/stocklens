"""
Valuation reports API routes.

Serves complete valuation reports and watchlist overview data.
These endpoints orchestrate the full pipeline: data → features → models → report.
"""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import ValuationReport, WatchlistOverviewItem, WatchlistOverviewResponse
from backend.config import get_settings
from backend.log import logger

router = APIRouter(prefix="/reports", tags=["reports"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _models_dir(ticker: str) -> Path:
    return PROJECT_ROOT / "data" / "models" / ticker.upper()


def _load_models(ticker: str):
    """Load trained models for a ticker if they exist. Returns (rf, fve) or (None, None)."""
    mdir = _models_dir(ticker)
    rf = None
    fve = None
    try:
        from backend.models.return_forecaster import ReturnForecaster
        rf_path = mdir / "return_forecaster"
        if rf_path.exists():
            rf = ReturnForecaster.load(rf_path)
            logger.info("Loaded ReturnForecaster for {}", ticker)
    except Exception as e:
        logger.warning("Could not load ReturnForecaster for {}: {}", ticker, e)

    try:
        from backend.models.fair_value_estimator import FairValueEstimator
        fve_path = mdir / "fair_value_estimator"
        if fve_path.exists():
            fve = FairValueEstimator.load(fve_path)
            logger.info("Loaded FairValueEstimator for {}", ticker)
    except Exception as e:
        logger.warning("Could not load FairValueEstimator for {}: {}", ticker, e)

    return rf, fve


@router.get("/{ticker}", response_model=ValuationReport)
async def get_valuation_report(ticker: str):
    """
    Generate a complete valuation report for a ticker.

    Uses trained models if available; falls back to demo mode otherwise.
    """
    ticker = ticker.upper()

    try:
        rf, fve = _load_models(ticker)
        if rf is not None and rf._fitted:
            report = _build_live_report(ticker, rf, fve)
        else:
            report = _build_demo_report(ticker)
        return report

    except Exception as e:
        logger.error("Failed to build report for {}: {}", ticker, e)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("", response_model=WatchlistOverviewResponse)
async def get_watchlist_overview():
    """
    Get the watchlist overview with valuation signals for all tracked tickers.

    This powers the main Watchlist page — a table showing each ticker's
    current price, fair value, signal, and predicted returns.
    """
    from backend.data.storage import get_watchlist

    watchlist = get_watchlist()
    items = []

    for _, row in watchlist.iterrows():
        ticker = row["ticker"]
        try:
            item = _build_overview_item(ticker, row)
            items.append(item)
        except Exception as e:
            logger.warning("Failed to build overview for {}: {}", ticker, e)
            items.append(WatchlistOverviewItem(
                ticker=ticker,
                name=row.get("name", ""),
                sector=row.get("sector", ""),
            ))

    return WatchlistOverviewResponse(items=items, count=len(items))


# ---------------------------------------------------------------------------
# Live report builder (used when trained models are present)
# ---------------------------------------------------------------------------

def _build_live_report(ticker: str, rf, fve) -> ValuationReport:
    """Build a full report using trained models and real cached data."""
    import numpy as np
    import pandas as pd
    from backend.data.storage import load_price_data, load_fundamental_data, load_macro_data
    from backend.data.fetchers.price_fetcher import fetch_ticker_info
    from backend.data.processors.feature_pipeline import assemble_features
    from backend.data.processors.macro_features import compute_macro_features
    from backend.data.fetchers.macro_fetcher import MACRO_SERIES
    from backend.api.report_builder import ReportBuilder

    price_df = load_price_data(ticker)
    if price_df is None or price_df.empty:
        return _build_demo_report(ticker)

    fund_df = load_fundamental_data(ticker)

    # Load macro series — stored by FRED series ID (e.g. "DFF"), keyed by
    # human-readable name (e.g. "fed_funds_rate") for compute_macro_features
    macro_dfs = {}
    for key, meta in MACRO_SERIES.items():
        series_id = meta["series_id"]
        cached = load_macro_data(series_id)
        if cached is not None:
            macro_dfs[key] = cached
    macro_features_df = compute_macro_features(macro_dfs) if macro_dfs else None

    feature_matrix = assemble_features(
        price_df=price_df,
        fundamentals_df=fund_df,
        macro_features_df=macro_features_df,
        ticker=ticker,
        compute_targets=True,
    )

    target_cols = [c for c in feature_matrix.columns if c.startswith("target_")]
    X = feature_matrix.drop(columns=target_cols)

    # Align to the model's exact training features — add missing cols as NaN,
    # drop extra cols — so XGBoost doesn't reject mismatched feature sets.
    if rf is not None and rf.feature_names:
        missing = [c for c in rf.feature_names if c not in X.columns]
        for col in missing:
            X[col] = float("nan")
        X = X[rf.feature_names]

    X = X.fillna(X.median())
    feature_matrix_clean = pd.concat([X, feature_matrix[target_cols]], axis=1)

    info = fetch_ticker_info(ticker)

    # Macro snapshot for context
    macro_snapshot = None
    if macro_features_df is not None and not macro_features_df.empty:
        macro_snapshot = macro_features_df.iloc[-1].to_dict()

    builder = ReportBuilder(return_forecaster=rf, fair_value_estimator=fve)
    return builder.build_report(
        ticker=ticker,
        feature_matrix=feature_matrix_clean,
        price_df=price_df,
        fundamentals=fund_df.iloc[0].to_dict() if fund_df is not None and not fund_df.empty else None,
        macro_snapshot=macro_snapshot,
        ticker_info=info,
    )


# ---------------------------------------------------------------------------
# Demo report builders (used until models are trained with real data)
# ---------------------------------------------------------------------------

def _build_demo_report(ticker: str) -> ValuationReport:
    """
    Build a demo report with realistic structure but placeholder predictions.

    This demonstrates the full report format.  Once xgboost is installed
    and models are trained, this is replaced with real inference.
    """
    from backend.api.schemas import (
        FeatureExplanation,
        FundamentalWithZscore,
        ReturnForecast,
        RiskFlag,
        SentimentSummary,
    )

    # Try to get real price and fundamental data
    current_price = None
    name = ""
    sector = ""

    try:
        from backend.data.storage import load_price_data
        price_data = load_price_data(ticker)
        if price_data is not None and not price_data.empty:
            current_price = round(float(price_data["Close"].iloc[-1]), 2)
    except Exception:
        pass

    try:
        from backend.data.fetchers.price_fetcher import fetch_ticker_info
        info = fetch_ticker_info(ticker)
        name = info.get("name", "")
        sector = info.get("sector", "")
    except Exception:
        pass

    return ValuationReport(
        ticker=ticker,
        name=name,
        sector=sector,
        signal="fairly_valued",
        confidence=None,
        current_price=current_price,
        fair_value=None,
        valuation_gap_pct=None,
        forecasts=[
            ReturnForecast(horizon="5d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="21d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="63d", predicted_return=0.0, lower_bound=None, upper_bound=None),
            ReturnForecast(horizon="126d", predicted_return=0.0, lower_bound=None, upper_bound=None),
        ],
        top_drivers=[],
        fundamentals=[],
        sentiment=None,
        macro_context=[],
        risk_flags=[
            RiskFlag(
                flag="demo_mode",
                severity="info",
                description="Models not yet trained. Install xgboost and run the training pipeline for real predictions.",
            )
        ],
        historical_accuracy=None,
    )


def _build_overview_item(ticker: str, watchlist_row: dict) -> WatchlistOverviewItem:
    """
    Build a watchlist overview item.

    Uses trained models for signal/confidence/fair_value when available;
    falls back to prices-only when models haven't been trained yet.
    """
    import numpy as np
    from backend.data.storage import load_price_data

    current_price = None
    change_1d = None
    change_1m = None

    price_data = load_price_data(ticker)
    if price_data is not None and not price_data.empty:
        close = price_data["Close"]
        current_price = round(float(close.iloc[-1]), 2)
        if len(close) > 1:
            change_1d = round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2)
        if len(close) > 21:
            change_1m = round(float((close.iloc[-1] / close.iloc[-22] - 1) * 100), 2)

    base = dict(
        ticker=ticker,
        name=watchlist_row.get("name", ""),
        sector=watchlist_row.get("sector", ""),
        current_price=current_price,
        price_change_1d_pct=change_1d,
        price_change_1m_pct=change_1m,
    )

    # If trained models exist, run a quick prediction for signal/confidence
    rf, fve = _load_models(ticker)
    if rf is None or not rf._fitted or price_data is None or price_data.empty:
        return WatchlistOverviewItem(**base, signal="unknown")

    try:
        import pandas as pd
        from backend.data.storage import load_fundamental_data, load_macro_data
        from backend.data.processors.feature_pipeline import assemble_features
        from backend.data.processors.macro_features import compute_macro_features
        from backend.data.fetchers.macro_fetcher import MACRO_SERIES
        from backend.models.fair_value_estimator import filter_fair_value_features

        fund_df = load_fundamental_data(ticker)
        macro_dfs = {k: load_macro_data(m["series_id"]) for k, m in MACRO_SERIES.items()
                     if load_macro_data(m["series_id"]) is not None}
        macro_features_df = compute_macro_features(macro_dfs) if macro_dfs else None

        fm = assemble_features(price_data, fund_df, macro_features_df,
                               ticker=ticker, compute_targets=False)

        # Align to training features
        missing = [c for c in rf.feature_names if c not in fm.columns]
        for col in missing:
            fm[col] = float("nan")
        X = fm[rf.feature_names].fillna(fm[rf.feature_names].median())
        X_latest = X.iloc[[-1]]

        # Return forecast (21d)
        preds = rf.predict(X_latest, include_intervals=True)
        ret_21d = None
        if "21d" in preds:
            ret_21d = float(preds["21d"]["predicted_return"].iloc[0])

        # Fair value + valuation gap
        fair_value = None
        valuation_gap = None
        fv_signal = "unknown"
        if fve is not None and fve._fitted and current_price:
            X_fv = filter_fair_value_features(X_latest)
            fv_pred = fve.predict(X_fv)
            fv_raw = float(fv_pred["fair_value"].iloc[0])
            if not np.isnan(fv_raw):
                fair_value = round(fv_raw, 2)
                gap = (current_price - fv_raw) / fv_raw
                valuation_gap = round(gap * 100, 2)
                fv_signal = "overvalued" if gap > 0.15 else ("undervalued" if gap < -0.15 else "fairly_valued")

        # Ensemble signal + confidence
        if ret_21d is not None and fair_value is not None:
            from backend.models.ensemble import EnsembleModel
            return_preds = pd.DataFrame({
                "predicted_return": [ret_21d],
                "lower_bound": [preds["21d"]["lower_bound"].iloc[0]],
                "upper_bound": [preds["21d"]["upper_bound"].iloc[0]],
            })
            fv_preds = pd.DataFrame({"fair_value": [fair_value]})
            prices = pd.Series([current_price])
            ensemble = EnsembleModel()
            result = ensemble.combine_predictions(return_preds, fv_preds, prices)
            signal = result["signal"].iloc[0]
            confidence = round(float(result["confidence"].iloc[0]), 1)
        elif ret_21d is not None:
            signal = "undervalued" if ret_21d > 0.02 else ("overvalued" if ret_21d < -0.02 else "fairly_valued")
            confidence = 40.0
        else:
            signal = fv_signal
            confidence = None

        return WatchlistOverviewItem(
            **base,
            fair_value=fair_value,
            valuation_gap_pct=valuation_gap,
            signal=signal,
            confidence=confidence,
            predicted_return_1m=ret_21d,
        )

    except Exception as e:
        logger.warning("Model inference failed for {} overview: {}", ticker, e)
        return WatchlistOverviewItem(**base, signal="unknown")
