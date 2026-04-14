# Contributing to StockLens

Thanks for your interest in contributing! StockLens is an educational project and welcomes contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/stocklens.git`
3. Create a feature branch: `git checkout -b feat/your-feature`
4. Follow the setup instructions in `README.md`

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Standards

- **Python**: Type hints on all functions. Docstrings on all public functions and classes.
- **Formatting**: Use Ruff (`ruff format backend/`). Line length: 100 characters.
- **Naming**: snake_case for Python, camelCase for JavaScript.
- **Imports**: Group as stdlib → third-party → local. Ruff handles sorting.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add RSI divergence feature
fix: handle missing FRED data gracefully
docs: update API endpoint table
test: add walk-forward fold validation
refactor: simplify feature pipeline merge logic
```

## Testing

Run the validation suites before submitting a PR:

```bash
python scripts/validate_phase1.py
python scripts/validate_phase2.py
python scripts/validate_phase3.py
python scripts/validate_phase4.py
python scripts/validate_phase5.py
```

## Pull Request Guidelines

- One feature or fix per PR
- Include tests for new functionality
- Update README if adding new endpoints, features, or configuration
- No API keys or data files in commits
- Ensure all validation scripts pass

## Architecture Notes

- **No lookahead bias**: This is the #1 rule. Every feature must use only data available at prediction time.
- **Sector-relative features**: Raw fundamental values should be z-scored against sector peers.
- **Fail gracefully**: If a data source is down, log a warning and continue with available data.
- **Walk-forward only**: Never use random train/test splits on time-series data.

## Questions?

Open an issue on GitHub. There are no dumb questions — this is a learning project.
