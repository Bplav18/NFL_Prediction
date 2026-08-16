# NFL Predictive Analytics & Market Modeling Platform

An end-to-end machine learning platform that ingests historical NFL game and
player data, trains probability-calibrated models to project game and player
outcomes, and compares model-implied probabilities against publicly available
market-implied probabilities to surface potential value.

## What this project demonstrates

- **Data engineering** — ingestion, cleaning, and feature pipelines for NFL data
- **Machine learning** — model training, comparison, and probability calibration
- **Statistics** — calibration, confidence intervals, expected value, Brier score
- **Software engineering** — a REST API and full-stack web application
- **Deployment** — a live, hosted application with CI/CD
- **Automation** — scheduled data refresh and prediction generation
- **Visualization** — a dashboard comparing model vs. market probabilities

## Architecture

```
NFL Data Sources
      │
      ▼
Data Ingestion
      │
      ▼
PostgreSQL / Supabase
      │
      ├───────────────┐
      ▼               ▼
Feature Engineering   Historical Dataset
      │
      ▼
ML Models (Game Winner / Spread / Total / Player Props)
      │
      ▼
Probability Calibration
      │
      ▼
Prediction Engine (Model Prob. vs Market Prob. → Edge, Confidence)
      │
      ▼
FastAPI Backend
      │
      ▼
React / Next.js Frontend → Live Dashboard
```

## Repository structure

```
nfl-predictive-market/
│
├── data/
│   ├── raw/            # Unmodified source data (not committed)
│   ├── processed/      # Cleaned, joined datasets (not committed)
│   └── features/       # Model-ready feature tables (not committed)
│
├── src/
│   ├── ingestion/       # Pulling data from external sources
│   ├── preprocessing/   # Cleaning and joining raw data
│   ├── features/        # Feature engineering
│   ├── models/           # Model training code
│   ├── evaluation/       # Metrics, calibration, backtesting
│   └── predictions/      # Turning model output into probabilities/edge
│
├── api/                 # FastAPI backend
├── frontend/             # React / Next.js dashboard
├── notebooks/            # Exploratory analysis & model evaluation
├── tests/                 # Unit / integration tests
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── .github/workflows/    # CI/CD
```

## Development phases

1. **Data + ML** — reliable NFL data → feature engineering → trained,
   properly evaluated models.
2. **Predictive market** — convert predictions to calibrated probabilities,
   compute market-implied probabilities, compute edge, backtest historically.
3. **Product** — database, API, dashboard, deployment, automated updates.

## Getting started

```bash
# Clone
git clone https://github.com/Bplav18/NFL_Prediction.git
cd NFL_Prediction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # then fill in real values

# Run the API locally
uvicorn api.main:app --reload
```

## Model evaluation approach

Models are compared using probabilistic and ranking metrics — not just raw
accuracy — including log loss, Brier score, ROC-AUC, and calibration curves,
using walk-forward (time-respecting) validation to avoid data leakage.

## Deployment

| Layer      | Service            |
|------------|---------------------|
| Frontend   | Vercel               |
| Backend    | Render                |
| Database   | Supabase / PostgreSQL |
| CI/CD      | GitHub Actions        |

## Status

🚧 Under active development. See `notebooks/` for current exploratory work.
