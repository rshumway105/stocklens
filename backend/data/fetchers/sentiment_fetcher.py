"""
Sentiment data fetcher — news headlines and social media signals.

This module fetches raw text data from news and social sources.  In Phase 2,
these will be scored with FinBERT for sentiment polarity.

Data sources:
- NewsAPI (requires free API key)
- RSS feeds from major financial outlets (no key needed)
- Reddit via PRAW (optional API key)

For Phase 1, this module provides the data-fetching scaffolding.
Actual NLP sentiment scoring is deferred to Phase 2.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from backend.log import logger

from backend.config import get_settings


# ---------------------------------------------------------------------------
# News headlines via NewsAPI
# ---------------------------------------------------------------------------

def fetch_news_headlines(
    query: str,
    days_back: int = 7,
    page_size: int = 50,
) -> pd.DataFrame:
    """
    Fetch recent news headlines for a query (typically a ticker or company name).

    Args:
        query: Search term (e.g. "AAPL" or "Apple Inc").
        days_back: How many days of history to fetch.
        page_size: Max articles per request (NewsAPI free tier caps at 100).

    Returns:
        DataFrame with columns: [title, description, source, url, published_at].
        Empty DataFrame if the API key is missing or the request fails.
    """
    settings = get_settings()
    if not settings.newsapi_key:
        logger.warning("NEWSAPI_KEY not set — news sentiment unavailable")
        return pd.DataFrame()

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": page_size,
        "language": "en",
        "apiKey": settings.newsapi_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            logger.info("No news articles found for query '{}'", query)
            return pd.DataFrame()

        rows = [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles
            if a.get("title")  # skip articles with no title
        ]

        df = pd.DataFrame(rows)
        if not df.empty and "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

        logger.info("Fetched {} headlines for '{}'", len(df), query)
        return df

    except Exception as e:
        logger.error("NewsAPI request failed for '{}': {}", query, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Financial RSS feeds (no API key needed)
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "reuters_markets": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
    "cnbc_economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}


def fetch_rss_headlines(feed_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch headlines from financial RSS feeds.

    Args:
        feed_key: Specific feed to fetch (key from RSS_FEEDS).
                  If None, fetches all feeds.

    Returns:
        DataFrame with columns: [title, summary, source, link, published].
    """
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser not installed. Run: pip install feedparser")
        return pd.DataFrame()

    feeds_to_fetch = (
        {feed_key: RSS_FEEDS[feed_key]} if feed_key and feed_key in RSS_FEEDS
        else RSS_FEEDS
    )

    all_entries: list[dict] = []

    for name, url in feeds_to_fetch.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                all_entries.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "source": name,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            logger.warning("Failed to parse RSS feed '{}': {}", name, e)

    df = pd.DataFrame(all_entries)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    logger.info("Fetched {} RSS entries from {} feeds", len(df), len(feeds_to_fetch))
    return df


# ---------------------------------------------------------------------------
# Reddit (via PRAW) — optional
# ---------------------------------------------------------------------------

def fetch_reddit_mentions(
    ticker: str,
    subreddits: Optional[list[str]] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """
    Fetch recent Reddit posts mentioning a ticker.

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.
    Returns an empty DataFrame if credentials are missing or PRAW
    is not installed.

    Args:
        ticker: Stock ticker symbol.
        subreddits: Subreddits to search. Defaults to finance-related subs.
        limit: Max posts per subreddit.

    Returns:
        DataFrame with columns: [title, selftext, subreddit, score, created_utc, url].
    """
    settings = get_settings()
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        logger.debug("Reddit credentials not set — skipping Reddit sentiment")
        return pd.DataFrame()

    try:
        import praw
    except ImportError:
        logger.warning("praw not installed. Run: pip install praw")
        return pd.DataFrame()

    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]

    try:
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

        posts: list[dict] = []
        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.search(ticker, limit=limit, sort="new"):
                    posts.append({
                        "title": post.title,
                        "selftext": post.selftext[:500],  # truncate long posts
                        "subreddit": sub_name,
                        "score": post.score,
                        "created_utc": datetime.utcfromtimestamp(post.created_utc),
                        "url": f"https://reddit.com{post.permalink}",
                        "num_comments": post.num_comments,
                    })
            except Exception as e:
                logger.warning("Failed to search r/{}: {}", sub_name, e)

        df = pd.DataFrame(posts)
        logger.info("Fetched {} Reddit posts for {}", len(df), ticker)
        return df

    except Exception as e:
        logger.error("Reddit fetch failed for {}: {}", ticker, e)
        return pd.DataFrame()
