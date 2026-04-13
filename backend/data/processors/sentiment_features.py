"""
Sentiment feature engineering pipeline.

Transforms raw news headlines and social media posts into model-ready
sentiment features.  In the full system, FinBERT scores each headline;
this module handles the aggregation logic that turns per-headline scores
into per-ticker, per-day features.

Feature groups:
- News sentiment: daily average, 7-day trend, article volume
- Social sentiment: Reddit mention volume, score-weighted sentiment
- Combined sentiment: blended signal across sources

Phase 2 note: The FinBERT scoring function is stubbed with a simple
keyword-based heuristic.  It will be replaced with the real model in
Phase 3 when torch/transformers are integrated.
"""

from typing import Optional

import numpy as np
import pandas as pd
from backend.log import logger


# ---------------------------------------------------------------------------
# Sentiment scoring (FinBERT stub)
# ---------------------------------------------------------------------------

# Keyword lists for the heuristic scorer (placeholder until FinBERT)
_POSITIVE_WORDS = {
    "surge", "jump", "gain", "rally", "beat", "upgrade", "bullish",
    "growth", "profit", "record", "strong", "outperform", "buy",
    "positive", "optimistic", "boost", "soar", "breakout", "innovation",
    "expand", "revenue", "success", "exceed", "upside",
}

_NEGATIVE_WORDS = {
    "crash", "drop", "fall", "plunge", "miss", "downgrade", "bearish",
    "loss", "decline", "weak", "underperform", "sell", "negative",
    "pessimistic", "risk", "fear", "recession", "layoff", "cut",
    "warning", "debt", "bankruptcy", "fraud", "investigation",
}


def score_text_heuristic(text: str) -> float:
    """
    Simple keyword-based sentiment scoring.

    Returns a score in [-1, 1] where:
        -1 = very negative
         0 = neutral
        +1 = very positive

    This is a placeholder — will be replaced with FinBERT in Phase 3.
    """
    if not text or not isinstance(text, str):
        return 0.0

    words = set(text.lower().split())
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def score_headlines(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """
    Score each row's text for sentiment.

    Args:
        df: DataFrame with a text column (e.g. news headlines).
        text_col: Name of the column containing text to score.

    Returns:
        Same DataFrame with an added 'sentiment_score' column in [-1, 1].
    """
    df = df.copy()
    df["sentiment_score"] = df[text_col].apply(score_text_heuristic)
    return df


# ---------------------------------------------------------------------------
# News sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_news_sentiment(
    headlines_df: pd.DataFrame,
    date_col: str = "published_at",
) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment into daily features.

    Args:
        headlines_df: DataFrame with columns [published_at, sentiment_score].
            Should already be scored via score_headlines().

    Returns:
        DataFrame indexed by date with columns:
        - news_sentiment_mean: average sentiment that day
        - news_sentiment_std: sentiment dispersion (disagreement)
        - news_article_count: number of articles
        - news_positive_ratio: fraction of positive articles
        - news_sentiment_7d: 7-day rolling average sentiment
        - news_sentiment_trend: sentiment direction (7d MA minus 21d MA)
    """
    if headlines_df.empty or "sentiment_score" not in headlines_df.columns:
        logger.warning("No scored headlines to aggregate")
        return pd.DataFrame()

    df = headlines_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date

    daily = df.groupby("date").agg(
        news_sentiment_mean=("sentiment_score", "mean"),
        news_sentiment_std=("sentiment_score", "std"),
        news_article_count=("sentiment_score", "count"),
        news_positive_ratio=("sentiment_score", lambda x: (x > 0.1).mean()),
    ).fillna({"news_sentiment_std": 0})

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    # Rolling features
    daily["news_sentiment_7d"] = (
        daily["news_sentiment_mean"].rolling(7, min_periods=1).mean()
    )
    daily["news_sentiment_21d"] = (
        daily["news_sentiment_mean"].rolling(21, min_periods=3).mean()
    )
    daily["news_sentiment_trend"] = (
        daily["news_sentiment_7d"] - daily["news_sentiment_21d"]
    )

    # Volume trend
    daily["news_volume_7d_avg"] = (
        daily["news_article_count"].rolling(7, min_periods=1).mean()
    )

    logger.info("Aggregated news sentiment: {} days", len(daily))
    return daily


# ---------------------------------------------------------------------------
# Social media sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_social_sentiment(
    posts_df: pd.DataFrame,
    date_col: str = "created_utc",
) -> pd.DataFrame:
    """
    Aggregate per-post social media data into daily features.

    Args:
        posts_df: DataFrame with columns [created_utc, title, score, num_comments].
            Will be scored for sentiment if 'sentiment_score' is missing.

    Returns:
        DataFrame indexed by date with columns:
        - social_sentiment_mean: average sentiment
        - social_mention_count: number of posts
        - social_engagement: total score + comments (weighted)
        - social_sentiment_7d: 7-day rolling average
    """
    if posts_df.empty:
        logger.debug("No social posts to aggregate")
        return pd.DataFrame()

    df = posts_df.copy()

    # Score if not already done
    if "sentiment_score" not in df.columns:
        df = score_headlines(df, text_col="title")

    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.date

    daily = df.groupby("date").agg(
        social_sentiment_mean=("sentiment_score", "mean"),
        social_mention_count=("sentiment_score", "count"),
        social_total_score=("score", "sum") if "score" in df.columns else ("sentiment_score", "count"),
        social_total_comments=("num_comments", "sum") if "num_comments" in df.columns else ("sentiment_score", "count"),
    )

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    # Engagement = upvotes + comments (proxy for attention/virality)
    if "social_total_score" in daily.columns and "social_total_comments" in daily.columns:
        daily["social_engagement"] = (
            daily["social_total_score"] + daily["social_total_comments"]
        )

    # Rolling
    daily["social_sentiment_7d"] = (
        daily["social_sentiment_mean"].rolling(7, min_periods=1).mean()
    )
    daily["social_volume_7d_avg"] = (
        daily["social_mention_count"].rolling(7, min_periods=1).mean()
    )

    logger.info("Aggregated social sentiment: {} days", len(daily))
    return daily


# ---------------------------------------------------------------------------
# Combined sentiment
# ---------------------------------------------------------------------------

def combine_sentiment_features(
    news_sentiment: Optional[pd.DataFrame] = None,
    social_sentiment: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge news and social sentiment into a single feature DataFrame.

    Also computes a blended overall sentiment signal.

    Args:
        news_sentiment: Output of aggregate_news_sentiment().
        social_sentiment: Output of aggregate_social_sentiment().

    Returns:
        DataFrame indexed by date with all sentiment columns + combined signal.
    """
    frames = []
    if news_sentiment is not None and not news_sentiment.empty:
        frames.append(news_sentiment)
    if social_sentiment is not None and not social_sentiment.empty:
        frames.append(social_sentiment)

    if not frames:
        logger.warning("No sentiment data to combine")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()

    # Combined sentiment: weighted average of news and social
    # News gets higher weight (more reliable, less noisy)
    news_col = "news_sentiment_7d"
    social_col = "social_sentiment_7d"

    if news_col in combined.columns and social_col in combined.columns:
        combined["combined_sentiment"] = (
            0.7 * combined[news_col].fillna(0) +
            0.3 * combined[social_col].fillna(0)
        )
    elif news_col in combined.columns:
        combined["combined_sentiment"] = combined[news_col]
    elif social_col in combined.columns:
        combined["combined_sentiment"] = combined[social_col]

    # Sentiment momentum (is overall sentiment improving or deteriorating?)
    if "combined_sentiment" in combined.columns:
        combined["sentiment_momentum"] = (
            combined["combined_sentiment"] -
            combined["combined_sentiment"].shift(7)
        )

    logger.info(
        "Combined sentiment features: {} columns × {} days",
        len(combined.columns), len(combined),
    )
    return combined
