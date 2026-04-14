#!/usr/bin/env python3
"""
StockLens quickstart — one-command setup.

Creates the virtual environment, installs dependencies,
initializes the database, and optionally seeds data.

Usage:
    python scripts/quickstart.py
    python scripts/quickstart.py --skip-seed
    python scripts/quickstart.py --tickers AAPL MSFT SPY
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd=None, check=True):
    """Run a shell command and print it."""
    print(f"\n  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or PROJECT_ROOT, capture_output=False)
    if check and result.returncode != 0:
        print(f"\n  ✗ Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="StockLens quickstart setup")
    parser.add_argument("--skip-seed", action="store_true", help="Skip data seeding")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend install")
    parser.add_argument("--tickers", nargs="+", help="Tickers to seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  StockLens Quickstart")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 10):
        print(f"\n  ✗ Python 3.10+ required (you have {sys.version})")
        sys.exit(1)
    print(f"\n  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check for .env
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        env_example = PROJECT_ROOT / ".env.example"
        if env_example.exists():
            print("\n  Creating .env from .env.example...")
            import shutil
            shutil.copy(env_example, env_file)
            print("  ⚠ Edit .env to add your FRED_API_KEY before seeding macro data")
        else:
            print("  ⚠ No .env file found — create one from .env.example")

    # Install Python dependencies
    print("\n── Installing Python dependencies ──")
    run(f"{sys.executable} -m pip install -r requirements.txt")

    # Try installing ML extras
    print("\n── Installing ML dependencies ──")
    run(f"{sys.executable} -m pip install xgboost", check=False)

    # Initialize database
    print("\n── Initializing database ──")
    run(f"{sys.executable} scripts/setup_db.py")

    # Seed data
    if not args.skip_seed:
        print("\n── Seeding data ──")
        ticker_arg = ""
        if args.tickers:
            ticker_arg = f" --tickers {' '.join(args.tickers)}"
        run(f"{sys.executable} scripts/seed_data.py{ticker_arg}", check=False)

    # Frontend
    if not args.skip_frontend:
        frontend_dir = PROJECT_ROOT / "frontend"
        if (frontend_dir / "package.json").exists():
            # Check for Node.js
            node_check = subprocess.run("node --version", shell=True, capture_output=True)
            if node_check.returncode == 0:
                print("\n── Installing frontend dependencies ──")
                run("npm install", cwd=frontend_dir)
            else:
                print("\n  ⚠ Node.js not found — skip frontend install")
                print("    Install Node.js 18+ and run: cd frontend && npm install")

    # Run validation
    print("\n── Running validation ──")
    run(f"{sys.executable} scripts/validate_phase1.py", check=False)

    # Done
    print("\n" + "=" * 60)
    print("  ✓ Setup complete!")
    print()
    print("  Start the backend:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Start the frontend:")
    print("    cd frontend && npm run dev")
    print()
    print("  Then open http://localhost:5173")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
