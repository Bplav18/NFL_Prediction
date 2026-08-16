"""
FastAPI entrypoint for the NFL Predictive Analytics platform.

Run locally with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="NFL Predictive Analytics API",
    description="Serves model-implied and market-implied win probabilities, "
    "spreads, totals, and player props for NFL games.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"status": "ok", "service": "nfl-predictive-analytics-api"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Routers for predictions, games, and players will be included here, e.g.:
# from api.routers import predictions
# app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
