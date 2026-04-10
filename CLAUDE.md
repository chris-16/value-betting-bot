# Value Betting Bot

AI-powered football betting bot that detects value bets by comparing ML model probability predictions against bookmaker odds. Paper trading MVP.

## Tech Stack

- **Language:** Python 3.11+
- **Framework:** Streamlit
- **Database:** PostgreSQL 16 (via Docker)
- **ORM:** SQLAlchemy + Alembic migrations
- **ML:** scikit-learn
- **Hosting:** Docker / Docker Compose

## Project Structure

```
src/
  app.py              # Streamlit entrypoint
  config.py           # Settings & env vars
  db/                 # Database models & connection
  models/             # ML model training & inference
  scrapers/           # Odds & match data fetchers
  strategies/         # Betting logic & value detection
  pages/              # Streamlit multi-page views
tests/                # pytest test suite
alembic/              # DB migrations
```

## Commands

```bash
# Run locally
docker compose up --build

# Run tests
pytest

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# DB migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
```

## Conventions

- Use type hints everywhere
- Keep functions small and focused
- SQL through SQLAlchemy ORM, never raw strings
- All monetary values as Decimal, never float
- Config via environment variables (python-dotenv)
- Ruff for linting and formatting (line length 100)
- Tests mirror src/ structure in tests/
