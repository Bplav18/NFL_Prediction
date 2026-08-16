"""
Builds a game-level feature table from raw schedules + weekly player stats.

The core design principle: every feature for a given game must be computable
using ONLY information available before that game was played. All rolling/
form features are shifted by one game so a team's upcoming game is never
included in its own "recent form" — this is the #1 way sports models leak
future information and look artificially good in testing.

Pipeline:
1. Load schedules (game-level) and weekly player stats (player-level).
2. Aggregate weekly stats up to team-game level (sum across players).
3. Reshape schedules into a long "team-game" table (one row per team per game).
4. Merge team-game stats onto the team-game table.
5. Sort by team + date, compute rolling form features using data strictly
   BEFORE the current game (via .shift(1)).
6. Pivot back to one row per game (home_* / away_* columns).
7. Build prediction targets (home_win, point differential, total points).
8. Save to data/features/game_features.parquet.

Usage:
    python -m src.preprocessing.build_features
    python -m src.preprocessing.build_features --rolling-window 5
    python -m src.preprocessing.build_features --include-playoffs
"""

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"

# nflverse's weekly stats retroactively use each franchise's CURRENT abbreviation
# for all historical seasons, but the schedules data uses the abbreviation that
# was actually in use at the time. Left unresolved, this fractures a relocated
# franchise's game history into two "teams" for feature purposes, silently
# breaking rolling-form features for every game they play. Normalize schedules
# to match the weekly-stats convention before doing anything else.
TEAM_CODE_FIXES = {
    "OAK": "LV",  # Raiders: Oakland -> Las Vegas (2020)
}

# Weekly player stats to sum up to team-game level.
STAT_COLUMNS = [
    "passing_yards", "passing_tds", "interceptions", "passing_epa",
    "rushing_yards", "rushing_tds", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_epa",
    "sacks", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
]


def load_raw():
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    weekly = pd.read_parquet(RAW_DIR / "weekly_player_stats.parquet")

    schedules["home_team"] = schedules["home_team"].replace(TEAM_CODE_FIXES)
    schedules["away_team"] = schedules["away_team"].replace(TEAM_CODE_FIXES)

    return schedules, weekly


def aggregate_team_week_stats(weekly: pd.DataFrame, include_playoffs: bool) -> pd.DataFrame:
    """Sum player-level stats up to one row per team per season/week."""
    season_types = ["REG", "POST"] if include_playoffs else ["REG"]
    weekly = weekly[weekly["season_type"].isin(season_types)].copy()
    agg = (
        weekly.groupby(["season", "week", "recent_team"])[STAT_COLUMNS]
        .sum()
        .reset_index()
        .rename(columns={"recent_team": "team"})
    )
    agg["turnovers"] = (
        agg["interceptions"]
        + agg["sack_fumbles_lost"]
        + agg["rushing_fumbles_lost"]
        + agg["receiving_fumbles_lost"]
    )
    return agg


def build_team_game_table(schedules: pd.DataFrame, include_playoffs: bool) -> pd.DataFrame:
    """Reshape one-row-per-game schedules into one-row-per-team-per-game."""
    df = schedules.copy()
    if not include_playoffs:
        df = df[df["game_type"] == "REG"]

    keep_cols = [
        "game_id", "season", "week", "gameday", "div_game",
        "home_team", "away_team", "home_score", "away_score",
        "home_rest", "away_rest",
    ]
    df = df[keep_cols].dropna(subset=["home_score", "away_score"])

    home = df.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest_days",
    }).drop(columns=["away_rest"])
    home["is_home"] = 1

    away = df.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest_days",
    }).drop(columns=["home_rest"])
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    return team_games


def add_rolling_form_features(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Adds rolling-average form features computed ONLY from a team's prior games.
    .shift(1) before .rolling() ensures the current game is excluded —
    this is the key leakage-prevention step.
    """
    team_games = team_games.sort_values(["team", "gameday"]).reset_index(drop=True)

    roll_cols = [
        "points_for", "points_against", "win",
        "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
        "passing_epa", "rushing_epa", "receiving_epa", "turnovers", "sacks",
    ]

    grouped = team_games.groupby("team", group_keys=False)
    for col in roll_cols:
        team_games[f"{col}_last{window}"] = grouped[col].apply(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).reset_index(drop=True)

    # Win/loss streak entering this game (also shifted to avoid leakage)
    team_games["win_streak"] = grouped["win"].apply(
        lambda s: s.shift(1).groupby((s.shift(1) != s.shift(1).shift(1)).cumsum()).cumcount() + 1
    ).reset_index(drop=True)
    team_games["win_streak"] = team_games["win_streak"].fillna(0)

    return team_games


def pivot_to_game_level(team_games: pd.DataFrame) -> pd.DataFrame:
    """Turn the team-game table back into one row per game, home_*/away_* columns."""
    home = team_games[team_games["is_home"] == 1].add_prefix("home_")
    away = team_games[team_games["is_home"] == 0].add_prefix("away_")

    games = home.merge(
        away,
        left_on="home_game_id",
        right_on="away_game_id",
        suffixes=("", ""),
    )

    games = games.rename(columns={"home_game_id": "game_id"})
    games = games.drop(columns=["away_game_id"])

    # season, week, div_game are identical for both teams in a game — collapse to one column
    for col in ["season", "week", "div_game"]:
        games[col] = games[f"home_{col}"]
        games = games.drop(columns=[f"home_{col}", f"away_{col}"])

    # Targets
    games["home_win"] = (games["home_points_for"] > games["home_points_against"]).astype(int)
    games["point_diff"] = games["home_points_for"] - games["home_points_against"]
    games["total_points"] = games["home_points_for"] + games["home_points_against"]

    return games


def main():
    parser = argparse.ArgumentParser(description="Build the game-level feature table")
    parser.add_argument("--rolling-window", type=int, default=4,
                         help="Number of prior games to average for form features (default: 4)")
    parser.add_argument("--include-playoffs", action="store_true",
                         help="Include playoff games (default: regular season only)")
    args = parser.parse_args()

    print("Loading raw data...")
    schedules, weekly = load_raw()

    print("Aggregating weekly player stats to team-game level...")
    team_week_stats = aggregate_team_week_stats(weekly, args.include_playoffs)

    print("Building team-game table...")
    team_games = build_team_game_table(schedules, args.include_playoffs)

    print("Merging team stats onto team-game table...")
    team_games = team_games.merge(
        team_week_stats,
        on=["season", "week", "team"],
        how="left",
    )

    print(f"Computing rolling form features (window={args.rolling_window})...")
    team_games = add_rolling_form_features(team_games, args.rolling_window)

    print("Pivoting back to one row per game...")
    games = pivot_to_game_level(team_games)

    # Drop games where either team has no prior-game history yet
    # (first game of the dataset for that team — no way to know their form)
    form_col = f"home_points_for_last{args.rolling_window}"
    before = len(games)
    games = games.dropna(subset=[form_col])
    print(f"Dropped {before - len(games)} games with no team history (start of dataset)")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "game_features.parquet"
    games.to_parquet(out_path, index=False)
    print(f"\nSaved {games.shape[0]} games x {games.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    main()