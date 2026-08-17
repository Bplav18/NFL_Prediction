"""
Generates the core "edge" report: model-implied win probability vs.
de-vigged market-implied win probability, for every game with walk-forward
out-of-fold predictions.

Critically, this uses OUT-OF-FOLD predictions from walk-forward validation —
never a model's predictions on data it was trained on — so the "edge" shown
here reflects genuine predictive signal, not the model just memorizing
outcomes it already saw.

Usage:
    python -m src.predictions.generate_edge_report
"""

from pathlib import Path

import pandas as pd

from src.evaluation.walk_forward import season_walk_forward_splits
from src.models.train import build_diff_features, get_models
from src.predictions.market_odds import add_market_probabilities

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "game_features.parquet"
SCHEDULES_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "schedules.parquet"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "market_edge.parquet"

MODEL_NAME = "random_forest"  # keep in sync with the best model from train.py
MIN_TRAIN_SEASONS = 2


def generate_out_of_fold_predictions(df: pd.DataFrame) -> pd.Series:
    """Runs walk-forward validation and returns out-of-fold model_home_prob per game."""
    X, y = build_diff_features(df)
    model = get_models()[MODEL_NAME]

    predictions = pd.Series(index=df.index, dtype=float)

    for test_season, train_idx, test_idx in season_walk_forward_splits(
        df, min_train_seasons=MIN_TRAIN_SEASONS
    ):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test = X.loc[test_idx]

        model.fit(X_train, y_train)
        predictions.loc[test_idx] = model.predict_proba(X_test)[:, 1]

    return predictions


def main():
    print("Loading features and schedules...")
    games = pd.read_parquet(FEATURES_PATH)
    schedules = pd.read_parquet(SCHEDULES_PATH)

    print(f"Generating out-of-fold predictions with {MODEL_NAME}...")
    games["model_home_prob"] = generate_out_of_fold_predictions(games)

    # Drop early-history games with no out-of-fold prediction
    before = len(games)
    games = games.dropna(subset=["model_home_prob"])
    print(f"Dropped {before - len(games)} games with no out-of-fold prediction "
          f"(first {MIN_TRAIN_SEASONS} seasons, used only for training)")

    print("Merging market moneylines and computing de-vigged probabilities...")
    odds_cols = schedules[["game_id", "home_moneyline", "away_moneyline"]]
    games = games.merge(odds_cols, on="game_id", how="left")

    before = len(games)
    games = games.dropna(subset=["home_moneyline", "away_moneyline"])
    print(f"Dropped {before - len(games)} games with no market odds available")

    games = add_market_probabilities(games)

    games["edge"] = games["model_home_prob"] - games["market_home_prob"]
    games["edge_pct"] = (games["edge"] * 100).round(1)

    report_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_win", "model_home_prob", "market_home_prob", "edge", "edge_pct",
    ]
    report = games[report_cols].sort_values("edge", key=abs, ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(report)} games -> {OUTPUT_PATH}")

    print("\nTop 10 largest edges (model vs. market disagreement):")
    print(report.head(10).to_string(index=False))

    # Quick check: does following the model's DISAGREEMENT with the market actually
    # pick winners? (edge > 0 -> model rates home more likely than market does,
    # i.e. the direction you'd bet if trading against the market's price)
    report["model_favors_home"] = report["edge"] > 0
    report["model_correct"] = (
        (report["model_favors_home"] & (report["home_win"] == 1))
        | (~report["model_favors_home"] & (report["home_win"] == 0))
    )
    print(f"\nWhen model disagreed with market (|edge| > 5%), model picked the "
          f"winner {report[report['edge'].abs() > 0.05]['model_correct'].mean():.1%} of the time "
          f"(n={len(report[report['edge'].abs() > 0.05])})")


if __name__ == "__main__":
    main()