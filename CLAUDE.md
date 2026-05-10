# StockLens

Personal stock valuation tool. Python/FastAPI backend, React/Vite frontend.

## Stack
- Backend: FastAPI, SQLite, APScheduler, yfinance, FRED API
- Frontend: React, Vite, Tailwind, Recharts

## Conventions
- Backend changes go in /backend, frontend in /frontend/src
- All new API endpoints need a corresponding schema in schemas.py
- Commit after each working feature, not in bulk

## Never do
- Don't add new npm packages without asking
- Don't change the Tailwind color tokens in tailwind.config.js
- Don't modify the scheduler jobs without noting the change in a comment
