"""
FastAPI entrypoint for the NFL Predictive Analytics platform.

Run locally with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predictions

app = FastAPI(
    title="NFL Predictive Analytics API",
    description="Serves model-implied and market-implied win probabilities, "
    "spreads, totals, and player props for NFL games.",
    version="0.1.0",
)

# Local-dev CORS: the frontend is a static file (opened directly or served
# on a different port), so the browser needs explicit permission to call
# this API from a different origin. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])


@app.get("/")
def read_root():
    return {"status": "ok", "service": "nfl-predictive-analytics-api"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}