"""
Trains and compares models for predicting NFL game winners (home_win).

Feature strategy: rather than feeding raw home_* and away_* rolling stats
separately, this builds DIFFERENTIAL features (home - away) for each rolling
metric. A team's absolute rolling passing yards mean much less than how it
compares to the opponent's — this is a standard, well-understood approach for
head-to-head prediction problems, and it also keeps feature count low, which
helps logistic regression avoid overfitting on ~1,800 games.

Models compared:
- Logistic Regression (baseline, scaled features)
- Random Forest
- XGBoost
- LightGBM

Evaluated with season-based walk-forward validation (see src/evaluation/
walk_forward.py) using probabilistic metrics — log loss, Brier score, ROC-AUC
— rather than raw accuracy, since well-calibrated probabilities matter more
than the model's arbitrary 0.5 decision threshold for this use case.

Usage:
    python -m src.models.train
    python -m src.models.train --min-train-seasons 3
"""

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.evaluation.walk_forward import season_walk_forward_splits

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "game_features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "features" / "models"


def build_diff_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Builds home-minus-away differential features from paired rolling columns."""
    rolling_pattern = re.compile(r"^home_(.+)_last(\d+)$")
    metrics, window = [], None
    for col in df.columns:
        m = rolling_pattern.match(col)
        if m:
            metrics.append(m.group(1))
            window = int(m.group(2))
    metrics = sorted(set(metrics))

    if not metrics:
        raise ValueError(
            "No rolling (_lastN) columns found. Did you run "
            "src.preprocessing.build_features first?"
        )

    features = pd.DataFrame(index=df.index)
    for metric in metrics:
        features[f"{metric}_diff"] = df[f"home_{metric}_last{window}"] - df[f"away_{metric}_last{window}"]

    features["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    features["win_streak_diff"] = df["home_win_streak"] - df["away_win_streak"]
    features["div_game"] = df["div_game"]

    target = df["home_win"]
    return features, target


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=-1,
        ),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, seasons: pd.Series, min_train_seasons: int) -> dict:
    """Runs walk-forward validation for one model, returns aggregate metrics."""
    df_for_split = pd.DataFrame({"season": seasons})
    all_true, all_pred_proba = [], []

    for test_season, train_idx, test_idx in season_walk_forward_splits(
        df_for_split, min_train_seasons=min_train_seasons
    ):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X_test)[:, 1]

        all_true.extend(y_test.tolist())
        all_pred_proba.extend(pred_proba.tolist())

    all_pred = [1 if p >= 0.5 else 0 for p in all_pred_proba]

    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "log_loss": log_loss(all_true, all_pred_proba),
        "brier_score": brier_score_loss(all_true, all_pred_proba),
        "roc_auc": roc_auc_score(all_true, all_pred_proba),
        "n_predictions": len(all_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and compare NFL game-winner models")
    parser.add_argument("--min-train-seasons", type=int, default=2,
                         help="Minimum seasons of history before the first test season (default: 2)")
    args = parser.parse_args()

    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)

    print("Building differential features...")
    X, y = build_diff_features(df)
    print(f"  -> {X.shape[1]} features, {X.shape[0]} games")
    print(f"  Features: {X.columns.tolist()}")

    seasons_available = sorted(df["season"].unique())
    print(f"  Seasons available: {seasons_available}")

    results = {}
    models = get_models()
    for name, model in models.items():
        print(f"\nEvaluating {name} (walk-forward validation)...")
        metrics = evaluate_model(model, X, y, df["season"], args.min_train_seasons)
        results[name] = metrics
        print(
            f"  accuracy={metrics['accuracy']:.3f}  "
            f"log_loss={metrics['log_loss']:.3f}  "
            f"brier={metrics['brier_score']:.3f}  "
            f"roc_auc={metrics['roc_auc']:.3f}  "
            f"(n={metrics['n_predictions']})"
        )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (lower log_loss / brier is better; higher auc is better)")
    print("=" * 70)
    results_df = pd.DataFrame(results).T.sort_values("log_loss")
    print(results_df.to_string())

    best_model_name = results_df.index[0]
    print(f"\nBest model by log loss: {best_model_name}")

    # Refit the best model on ALL available data and save it
    print(f"Refitting {best_model_name} on full dataset and saving...")
    best_model = models[best_model_name]
    best_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{best_model_name}.joblib"
    joblib.dump({"model": best_model, "feature_columns": X.columns.tolist()}, model_path)
    print(f"Saved -> {model_path}")

    results_path = MODELS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path)
    print(f"Saved comparison table -> {results_path}")


if __name__ == "__main__":
    main()