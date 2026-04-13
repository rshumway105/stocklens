"""
Macro data API routes.

Endpoints for fetching and serving macroeconomic indicator data.
"""

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import MacroDataPoint, MacroSeriesMeta, MacroSeriesResponse
from backend.data.fetchers.macro_fetcher import (
    MACRO_SERIES,
    fetch_single_series,
    get_macro_catalog,
)
from backend.data.storage import load_macro_data, save_macro_data

router = APIRouter(prefix="/macro", tags=["macro"])


@router.get("/catalog", response_model=list[MacroSeriesMeta])
async def list_macro_series():
    """Return the catalog of available macro series with their FRED IDs."""
    return [
        MacroSeriesMeta(key=item["key"], series_id=item["series_id"], description=item["description"])
        for item in get_macro_catalog()
    ]


@router.get("/{series_key}", response_model=MacroSeriesResponse)
async def get_macro_series(
    series_key: str,
    refresh: bool = Query(False, description="Force re-fetch from FRED"),
):
    """
    Get data for a specific macro series.

    Uses cached parquet data if available; pass ?refresh=true to force
    a fresh pull from FRED.
    """
    if series_key not in MACRO_SERIES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown macro series key '{series_key}'. Use /macro/catalog for valid keys.",
        )

    meta = MACRO_SERIES[series_key]
    series_id = meta["series_id"]

    # Try cache first
    df = None if refresh else load_macro_data(series_id)

    if df is None:
        df = fetch_single_series(series_id)
        if df.empty:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch FRED series {series_id}. Check your FRED_API_KEY.",
            )
        save_macro_data(series_id, df)

    # Convert to response
    data_points = [
        MacroDataPoint(date=str(idx.date()), value=float(row.iloc[0]))
        for idx, row in df.iterrows()
        if not row.isna().any()
    ]

    return MacroSeriesResponse(
        key=series_key,
        series_id=series_id,
        description=meta["description"],
        data=data_points,
        count=len(data_points),
    )
