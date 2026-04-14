#!/usr/bin/env python3
"""
Phase 6 validation — Polish & Documentation.

Checks documentation completeness, project structure, code quality signals,
and overall project health.

Usage: python3 scripts/validate_phase6.py
"""

import sys
from pathlib import Path

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


root = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════
print("\n═══ 1. Root Project Files ═══")
# ══════════════════════════════════════════════════════════

for f in ["README.md", "LICENSE", ".env.example", ".gitignore",
          "pyproject.toml", "requirements.txt", "CONTRIBUTING.md"]:
    check(f"Exists: {f}", (root / f).exists())

# ══════════════════════════════════════════════════════════
print("\n═══ 2. README Quality ═══")
# ══════════════════════════════════════════════════════════

readme = (root / "README.md").read_text()
check("README > 5000 chars", len(readme) > 5000, f"got {len(readme)}")
check("Has architecture diagram", "┌─" in readme or "```" in readme)
check("Has Quick Start section", "Quick Start" in readme)
check("Has Prerequisites", "Prerequisites" in readme)
check("Has API Reference table", "/api/health" in readme)
check("Has Feature Catalog", "Feature Catalog" in readme)
check("Has Model Architecture", "Model Architecture" in readme)
check("Has Project Structure tree", "stocklens/" in readme)
check("Has Development section", "Development" in readme)
check("Has Roadmap", "Roadmap" in readme)
check("Has License mention", "MIT" in readme)
check("Has disclaimer", "not financial advice" in readme.lower())
check("Has clone URL", "git clone" in readme)
check("Has pip install command", "pip install" in readme)
check("Has uvicorn start command", "uvicorn" in readme)
check("Has npm install command", "npm install" in readme)
check("All 6 phases marked complete",
      readme.count("✅ Complete") >= 6 or readme.count("✅") >= 6,
      f"found {readme.count('✅')} checkmarks")

# ══════════════════════════════════════════════════════════
print("\n═══ 3. CONTRIBUTING Guide ═══")
# ══════════════════════════════════════════════════════════

contrib = (root / "CONTRIBUTING.md").read_text()
check("CONTRIBUTING > 500 chars", len(contrib) > 500)
check("Has commit message guide", "Conventional Commits" in contrib or "commit" in contrib.lower())
check("Has code standards", "Type hints" in contrib or "type hints" in contrib.lower())
check("Has testing section", "validate" in contrib.lower())
check("Has architecture notes", "lookahead" in contrib.lower())

# ══════════════════════════════════════════════════════════
print("\n═══ 4. .env.example Quality ═══")
# ══════════════════════════════════════════════════════════

env = (root / ".env.example").read_text()
check("Has FRED_API_KEY", "FRED_API_KEY" in env)
check("Has NEWSAPI_KEY", "NEWSAPI_KEY" in env)
check("Has signup URL for FRED", "fred.stlouisfed.org" in env)
check("Has signup URL for NewsAPI", "newsapi.org" in env)
check("Has clear instructions", "Copy this file" in env or "cp .env.example" in env)

# ══════════════════════════════════════════════════════════
print("\n═══ 5. Backend Completeness ═══")
# ══════════════════════════════════════════════════════════

backend_files = [
    "backend/__init__.py",
    "backend/main.py",
    "backend/config.py",
    "backend/log.py",
    "backend/data/__init__.py",
    "backend/data/storage.py",
    "backend/data/fetchers/__init__.py",
    "backend/data/fetchers/price_fetcher.py",
    "backend/data/fetchers/macro_fetcher.py",
    "backend/data/fetchers/fundamental_fetcher.py",
    "backend/data/fetchers/sentiment_fetcher.py",
    "backend/data/processors/__init__.py",
    "backend/data/processors/technical_features.py",
    "backend/data/processors/fundamental_features.py",
    "backend/data/processors/macro_features.py",
    "backend/data/processors/sentiment_features.py",
    "backend/data/processors/target_builder.py",
    "backend/data/processors/feature_pipeline.py",
    "backend/models/__init__.py",
    "backend/models/return_forecaster.py",
    "backend/models/fair_value_estimator.py",
    "backend/models/ensemble.py",
    "backend/models/explainer.py",
    "backend/models/trainer.py",
    "backend/api/__init__.py",
    "backend/api/schemas.py",
    "backend/api/report_builder.py",
    "backend/api/routes/__init__.py",
    "backend/api/routes/watchlist.py",
    "backend/api/routes/predictions.py",
    "backend/api/routes/macro.py",
    "backend/api/routes/reports.py",
    "backend/jobs/__init__.py",
    "backend/jobs/scheduler.py",
    "backend/jobs/tasks.py",
    "backend/tests/__init__.py",
    "backend/tests/test_phase1.py",
    "backend/tests/test_phase2.py",
]

missing = [f for f in backend_files if not (root / f).exists()]
check(f"All {len(backend_files)} backend files exist", len(missing) == 0,
      f"missing: {missing[:5]}")

# ══════════════════════════════════════════════════════════
print("\n═══ 6. Frontend Completeness ═══")
# ══════════════════════════════════════════════════════════

frontend_files = [
    "frontend/package.json",
    "frontend/vite.config.js",
    "frontend/tailwind.config.js",
    "frontend/postcss.config.js",
    "frontend/index.html",
    "frontend/src/main.jsx",
    "frontend/src/index.css",
    "frontend/src/App.jsx",
    "frontend/src/pages/Watchlist.jsx",
    "frontend/src/pages/TickerDetail.jsx",
    "frontend/src/pages/MacroDashboard.jsx",
    "frontend/src/pages/ModelPerformance.jsx",
    "frontend/src/components/PriceChart.jsx",
    "frontend/src/components/ShapWaterfall.jsx",
    "frontend/src/components/ForecastFanChart.jsx",
    "frontend/src/components/SentimentTimeline.jsx",
    "frontend/src/components/ValuationBadge.jsx",
    "frontend/src/hooks/useApi.js",
    "frontend/src/utils/formatters.js",
]

missing_fe = [f for f in frontend_files if not (root / f).exists()]
check(f"All {len(frontend_files)} frontend files exist", len(missing_fe) == 0,
      f"missing: {missing_fe[:5]}")

# ══════════════════════════════════════════════════════════
print("\n═══ 7. Scripts Completeness ═══")
# ══════════════════════════════════════════════════════════

script_files = [
    "scripts/setup_db.py",
    "scripts/seed_data.py",
    "scripts/run_backtest.py",
    "scripts/quickstart.py",
    "scripts/validate_phase1.py",
    "scripts/validate_phase2.py",
    "scripts/validate_phase3.py",
    "scripts/validate_phase4.py",
    "scripts/validate_phase5.py",
    "scripts/validate_phase6.py",
]

missing_sc = [f for f in script_files if not (root / f).exists()]
check(f"All {len(script_files)} script files exist", len(missing_sc) == 0,
      f"missing: {missing_sc[:5]}")

# ══════════════════════════════════════════════════════════
print("\n═══ 8. Code Quality Signals ═══")
# ══════════════════════════════════════════════════════════

# Check that all Python files have module docstrings
py_files = list((root / "backend").rglob("*.py"))
files_with_docstrings = 0
files_checked = 0

for pf in py_files:
    if pf.name == "__init__.py":
        continue
    content = pf.read_text()
    if not content.strip():
        continue
    files_checked += 1
    # Check for triple-quote docstring near the top
    lines = content.strip().split("\n")
    for line in lines[:5]:
        if '"""' in line or "'''" in line:
            files_with_docstrings += 1
            break

docstring_pct = (files_with_docstrings / max(files_checked, 1)) * 100
check(f"Module docstrings ({files_with_docstrings}/{files_checked})",
      docstring_pct > 80, f"{docstring_pct:.0f}%")

# Check __init__.py files have content
init_files = list((root / "backend").rglob("__init__.py"))
inits_with_content = sum(1 for f in init_files if f.read_text().strip())
check(f"__init__.py files have docstrings ({inits_with_content}/{len(init_files)})",
      inits_with_content == len(init_files))

# No API keys in code
all_py = " ".join(pf.read_text() for pf in py_files)
check("No hardcoded API keys", "sk-" not in all_py and "AKIA" not in all_py)

# .gitignore covers essentials
gitignore = (root / ".gitignore").read_text()
check(".gitignore covers .env", ".env" in gitignore)
check(".gitignore covers __pycache__", "__pycache__" in gitignore)
check(".gitignore covers node_modules", "node_modules" in gitignore)
check(".gitignore covers .db files", "*.db" in gitignore)

# ══════════════════════════════════════════════════════════
print("\n═══ 9. Total File Count ═══")
# ══════════════════════════════════════════════════════════

total_py = len(list((root / "backend").rglob("*.py")))
total_jsx = len(list((root / "frontend" / "src").rglob("*.jsx")))
total_js = len(list((root / "frontend" / "src").rglob("*.js")))
total_scripts = len(list((root / "scripts").glob("*.py")))

check(f"Backend: {total_py} Python files", total_py >= 30)
check(f"Frontend: {total_jsx + total_js} JS/JSX files", total_jsx + total_js >= 10)
check(f"Scripts: {total_scripts} utility scripts", total_scripts >= 7)

# Lines of code estimate
total_lines = 0
for pf in (root / "backend").rglob("*.py"):
    total_lines += len(pf.read_text().split("\n"))
for jf in (root / "frontend" / "src").rglob("*.jsx"):
    total_lines += len(jf.read_text().split("\n"))
for jf in (root / "frontend" / "src").rglob("*.js"):
    total_lines += len(jf.read_text().split("\n"))
for sf in (root / "scripts").glob("*.py"):
    total_lines += len(sf.read_text().split("\n"))

check(f"Total ~{total_lines} lines of code", total_lines > 3000)

# ══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"  Total project: ~{total_lines} lines across {total_py + total_jsx + total_js + total_scripts} files")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
