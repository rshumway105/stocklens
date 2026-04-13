"""
Job scheduler for StockLens.

Configures recurring tasks using APScheduler:
- Daily price refresh (after market close)
- Weekly fundamentals refresh (weekends)
- Daily macro data refresh

The scheduler is started with the FastAPI app and stopped on shutdown.
"""

from backend.log import logger


def create_scheduler():
    """
    Create and configure the APScheduler instance.

    Returns the scheduler (not yet started).  Call scheduler.start()
    to begin running jobs.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("apscheduler not installed — scheduled jobs disabled")
        return None

    scheduler = BackgroundScheduler()

    # Daily price refresh — 6:30 PM ET (after market close)
    scheduler.add_job(
        _run_price_refresh,
        CronTrigger(hour=18, minute=30, timezone="US/Eastern"),
        id="daily_price_refresh",
        name="Daily price data refresh",
        replace_existing=True,
    )

    # Daily macro refresh — 7:00 PM ET
    scheduler.add_job(
        _run_macro_refresh,
        CronTrigger(hour=19, minute=0, timezone="US/Eastern"),
        id="daily_macro_refresh",
        name="Daily macro data refresh",
        replace_existing=True,
    )

    # Weekly fundamentals refresh — Saturday 10 AM ET
    scheduler.add_job(
        _run_fundamentals_refresh,
        CronTrigger(day_of_week="sat", hour=10, minute=0, timezone="US/Eastern"),
        id="weekly_fundamentals_refresh",
        name="Weekly fundamentals refresh",
        replace_existing=True,
    )

    logger.info("Scheduler configured with 3 recurring jobs")
    return scheduler


def _run_price_refresh():
    """Wrapper for scheduled price refresh."""
    from backend.jobs.tasks import refresh_price_data
    logger.info("Scheduled job: refreshing price data")
    refresh_price_data()


def _run_macro_refresh():
    """Wrapper for scheduled macro refresh."""
    from backend.jobs.tasks import refresh_macro_data
    logger.info("Scheduled job: refreshing macro data")
    refresh_macro_data()


def _run_fundamentals_refresh():
    """Wrapper for scheduled fundamentals refresh."""
    from backend.jobs.tasks import refresh_fundamentals
    logger.info("Scheduled job: refreshing fundamentals")
    refresh_fundamentals()
