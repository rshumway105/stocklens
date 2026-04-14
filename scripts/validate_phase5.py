#!/usr/bin/env python3
"""
Phase 5 validation — React Dashboard.

Validates all frontend files exist, have expected content, and are well-formed.
Usage: python3 scripts/validate_phase5.py
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


root = Path(__file__).resolve().parent.parent / "frontend"

# ══════════════════════════════════════════════════════════
print("\n═══ 1. Project Configuration ═══")
# ══════════════════════════════════════════════════════════

check("package.json exists", (root / "package.json").exists())
pkg = (root / "package.json").read_text()
check("Has react dependency", '"react"' in pkg)
check("Has react-router-dom", '"react-router-dom"' in pkg)
check("Has recharts", '"recharts"' in pkg)
check("Has axios", '"axios"' in pkg)
check("Has tailwindcss", '"tailwindcss"' in pkg)

check("vite.config.js exists", (root / "vite.config.js").exists())
vite = (root / "vite.config.js").read_text()
check("Vite proxies /api", "'/api'" in vite)

check("tailwind.config.js exists", (root / "tailwind.config.js").exists())
tw = (root / "tailwind.config.js").read_text()
check("Tailwind has terminal colors", "terminal" in tw)
check("Tailwind has accent colors", "accent" in tw)
check("Tailwind has custom fonts", "JetBrains Mono" in tw)

check("postcss.config.js exists", (root / "postcss.config.js").exists())
check("index.html exists", (root / "index.html").exists())
html = (root / "index.html").read_text()
check("HTML loads Google Fonts", "fonts.googleapis.com" in html)
check("HTML has dark class", 'class="dark"' in html)

# ══════════════════════════════════════════════════════════
print("\n═══ 2. Entry Points ═══")
# ══════════════════════════════════════════════════════════

check("main.jsx exists", (root / "src" / "main.jsx").exists())
main = (root / "src" / "main.jsx").read_text()
check("main.jsx imports App", "import App" in main)
check("main.jsx renders root", "createRoot" in main)

check("index.css exists", (root / "src" / "index.css").exists())
css = (root / "src" / "index.css").read_text()
check("CSS has Tailwind directives", "@tailwind base" in css)
check("CSS has card component", ".card" in css)
check("CSS has badge components", "badge-overvalued" in css)
check("CSS has table styles", "table-header" in css)

# ══════════════════════════════════════════════════════════
print("\n═══ 3. App & Routing ═══")
# ══════════════════════════════════════════════════════════

check("App.jsx exists", (root / "src" / "App.jsx").exists())
app = (root / "src" / "App.jsx").read_text()
check("App uses BrowserRouter", "BrowserRouter" in app)
check("App has Watchlist route", "Watchlist" in app)
check("App has TickerDetail route", "TickerDetail" in app)
check("App has MacroDashboard route", "MacroDashboard" in app)
check("App has ModelPerformance route", "ModelPerformance" in app)
check("App has /ticker/:ticker route", "/ticker/:ticker" in app)
check("App has navigation", "NavLink" in app)
check("App has StockLens branding", "StockLens" in app)

# ══════════════════════════════════════════════════════════
print("\n═══ 4. Pages ═══")
# ══════════════════════════════════════════════════════════

pages_dir = root / "src" / "pages"

# Watchlist
check("Watchlist.jsx exists", (pages_dir / "Watchlist.jsx").exists())
wl = (pages_dir / "Watchlist.jsx").read_text()
check("Watchlist has add ticker", "addTicker" in wl or "handleAdd" in wl)
check("Watchlist has sorting", "handleSort" in wl)
check("Watchlist has remove", "handleRemove" in wl)
check("Watchlist uses ValuationBadge", "ValuationBadge" in wl)
check("Watchlist navigates to detail", "navigate" in wl)
check("Watchlist has loading state", "skeleton" in wl or "loading" in wl)
check("Watchlist has error state", "error" in wl)

# TickerDetail
check("TickerDetail.jsx exists", (pages_dir / "TickerDetail.jsx").exists())
td = (pages_dir / "TickerDetail.jsx").read_text()
check("Detail has PriceChart", "PriceChart" in td)
check("Detail has ShapWaterfall", "ShapWaterfall" in td)
check("Detail has ForecastFanChart", "ForecastFanChart" in td)
check("Detail has ValuationBadge", "ValuationBadge" in td)
check("Detail shows fundamentals", "fundamentals" in td)
check("Detail shows risk flags", "risk_flags" in td)
check("Detail shows macro context", "macro_context" in td)
check("Detail has back link", "Watchlist" in td)

# MacroDashboard
check("MacroDashboard.jsx exists", (pages_dir / "MacroDashboard.jsx").exists())
md = (pages_dir / "MacroDashboard.jsx").read_text()
check("Macro has category grid", "CATEGORY_MAP" in md)
check("Macro has series selector", "selectedSeries" in md)
check("Macro has chart", "LineChart" in md)
check("Macro has friendly names", "FRIENDLY_NAMES" in md)

# ModelPerformance
check("ModelPerformance.jsx exists", (pages_dir / "ModelPerformance.jsx").exists())
mp = (pages_dir / "ModelPerformance.jsx").read_text()
check("Performance has system status", "System Status" in mp)
check("Performance has model cards", "ModelCard" in mp)
check("Performance has ensemble info", "Ensemble" in mp)
check("Performance has walk-forward", "Walk-Forward" in mp)

# ══════════════════════════════════════════════════════════
print("\n═══ 5. Components ═══")
# ══════════════════════════════════════════════════════════

comp_dir = root / "src" / "components"

check("ValuationBadge.jsx exists", (comp_dir / "ValuationBadge.jsx").exists())
check("PriceChart.jsx exists", (comp_dir / "PriceChart.jsx").exists())
pc = (comp_dir / "PriceChart.jsx").read_text()
check("PriceChart uses Recharts", "ResponsiveContainer" in pc)
check("PriceChart has fair value line", "fairValue" in pc)

check("ShapWaterfall.jsx exists", (comp_dir / "ShapWaterfall.jsx").exists())
sw = (comp_dir / "ShapWaterfall.jsx").read_text()
check("ShapWaterfall uses BarChart", "BarChart" in sw)
check("ShapWaterfall colors by direction", "direction" in sw)

check("ForecastFanChart.jsx exists", (comp_dir / "ForecastFanChart.jsx").exists())
fc = (comp_dir / "ForecastFanChart.jsx").read_text()
check("ForecastFanChart has horizons", "HORIZON_LABELS" in fc)

check("SentimentTimeline.jsx exists", (comp_dir / "SentimentTimeline.jsx").exists())

# ══════════════════════════════════════════════════════════
print("\n═══ 6. Hooks & Utils ═══")
# ══════════════════════════════════════════════════════════

check("useApi.js exists", (root / "src" / "hooks" / "useApi.js").exists())
api = (root / "src" / "hooks" / "useApi.js").read_text()
check("useApi has GET fetch", "api.get" in api)
check("useApi has POST", "api.post" in api)
check("useApi has DELETE", "api.delete" in api)
check("useApi has error handling", "catch" in api)

check("formatters.js exists", (root / "src" / "utils" / "formatters.js").exists())
fmt = (root / "src" / "utils" / "formatters.js").read_text()
check("Has formatPrice", "formatPrice" in fmt)
check("Has formatPercent", "formatPercent" in fmt)
check("Has formatCompact", "formatCompact" in fmt)
check("Has valueColor", "valueColor" in fmt)
check("Has signalBadgeClass", "signalBadgeClass" in fmt)

# ══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

sys.exit(1 if FAIL > 0 else 0)
