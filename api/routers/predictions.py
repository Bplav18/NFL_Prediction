"""
Serves model-vs-market predictions from the precomputed edge report.

The edge report (data/features/market_edge.parquet) is generated offline by
src/predictions/generate_edge_report.py using out-of-fold walk-forward
predictions — this API layer only reads and serves that file, it doesn't
retrain anything live.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import roc_auc_score, accuracy_score

router = APIRouter()

EDGE_REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "market_edge.parquet"


def load_edge_report() -> pd.DataFrame:
    if not EDGE_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Edge report not found. Run "
                "`python -m src.predictions.generate_edge_report` first."
            ),
        )
    return pd.read_parquet(EDGE_REPORT_PATH)


@router.get("/edge")
def get_edge_report(
    season: int | None = Query(None, description="Filter to a specific season"),
    min_abs_edge: float = Query(0.0, description="Only return games with |edge| >= this value (0-1 scale)"),
    limit: int = Query(100, le=1000),
):
    """Returns games ranked by |model_prob - market_prob|, largest disagreement first."""
    df = load_edge_report()

    if season is not None:
        df = df[df["season"] == season]
    if min_abs_edge > 0:
        df = df[df["edge"].abs() >= min_abs_edge]

    df = df.sort_values("edge", key=lambda s: s.abs(), ascending=False).head(limit)
    return df.to_dict(orient="records")


@router.get("/stats")
def get_summary_stats():
    """Overall model vs. market performance comparison."""
    df = load_edge_report()

    model_auc = roc_auc_score(df["home_win"], df["model_home_prob"])
    market_auc = roc_auc_score(df["home_win"], df["market_home_prob"])
    model_acc = accuracy_score(df["home_win"], df["model_home_prob"] > 0.5)
    market_acc = accuracy_score(df["home_win"], df["market_home_prob"] > 0.5)

    high_edge = df[df["edge"].abs() > 0.05].copy()
    if len(high_edge) > 0:
        # "Model favors home" here means the model rates home MORE likely than
        # the market does (edge > 0) — i.e. the direction you'd bet if trading
        # against the market's own price. This intentionally matches the
        # definition used in generate_edge_report.py, NOT simply whether the
        # model's raw probability is > 50% (a different, less relevant question
        # for a market-comparison product).
        model_favors_home = high_edge["edge"] > 0
        model_correct = (
            (model_favors_home & (high_edge["home_win"] == 1))
            | (~model_favors_home & (high_edge["home_win"] == 0))
        )
        high_disagreement_model_accuracy = float(model_correct.mean())
    else:
        high_disagreement_model_accuracy = None

    return {
        "n_games": len(df),
        "seasons": sorted(df["season"].unique().tolist()),
        "model_roc_auc": round(model_auc, 3),
        "market_roc_auc": round(market_auc, 3),
        "model_accuracy": round(model_acc, 3),
        "market_accuracy": round(market_acc, 3),
        "n_high_disagreement_games": len(high_edge),
        "model_accuracy_on_high_disagreement_games": (
            round(high_disagreement_model_accuracy, 3)
            if high_disagreement_model_accuracy is not None
            else None
        ),
    }


@router.get("/games/{season}/{week}")
def get_week_games(season: int, week: int):
    """Returns all games for a specific season/week with model and market probabilities."""
    df = load_edge_report()
    df = df[(df["season"] == season) & (df["week"] == week)]

    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No games found for season={season}, week={week}")

    return df.sort_values("game_id").to_dict(orient="records")