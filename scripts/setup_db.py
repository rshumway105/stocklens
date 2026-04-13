#!/usr/bin/env python3
"""
Setup the StockLens database.

Creates all tables and ensures the data directories exist.
Safe to run multiple times — uses CREATE IF NOT EXISTS.

Usage:
    python scripts/setup_db.py
"""

import sys
from pathlib import Path

# Add project root to path so imports work when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.data.storage import init_db


def main() -> None:
    settings = get_settings()

    # Ensure directories exist
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.parquet_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Database path: {settings.db_path}")
    print(f"Cache directory: {settings.parquet_cache_dir}")

    init_db()
    print("✓ Database initialized successfully")


if __name__ == "__main__":
    main()
