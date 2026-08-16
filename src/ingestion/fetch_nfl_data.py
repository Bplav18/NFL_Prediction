"""
Pulls raw NFL data via nfl_data_py and saves it to data/raw/ as parquet.

Sources (all via nfl_data_py, backed by the nflverse project):
- Schedules & results: game-level info (teams, scores, dates, weather, etc.)
- Weekly player stats: per-player, per-week box score stats
- Play-by-play: every play of every game (large; pulled per-season)

Usage:
    python -m src.ingestion.fetch_nfl_data --years 2019 2020 2021 2022 2023 2024
    python -m src.ingestion.fetch_nfl_data --years 2024 --skip-pbp   # faster, skip play-by-play
"""

import argparse
import sys
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_schedules(years: list[int]) -> pd.DataFrame:
    """Game-level schedule and results data."""
    print(f"Fetching schedules for {years}...")
    df = nfl.import_schedules(years)
    print(f"  -> {df.shape[0]} games, {df.shape[1]} columns")
    return df


def fetch_weekly_data(years: list[int]) -> pd.DataFrame:
    """Per-player, per-week stat lines."""
    print(f"Fetching weekly player data for {years}...")
    df = nfl.import_weekly_data(years)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def fetch_pbp_data(years: list[int]) -> pd.DataFrame:
    """Play-by-play data. Large — one season at a time to limit memory use."""
    frames = []
    for year in years:
        print(f"Fetching play-by-play for {year}...")
        df = nfl.import_pbp_data([year])
        print(f"  -> {df.shape[0]} plays, {df.shape[1]} columns")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, name: str) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {name} -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Fetch raw NFL data into data/raw/")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Seasons to pull, e.g. --years 2022 2023 2024",
    )
    parser.add_argument(
        "--skip-pbp",
        action="store_true",
        help="Skip play-by-play data (it's the slowest and largest pull)",
    )
    args = parser.parse_args()

    try:
        schedules = fetch_schedules(args.years)
        save(schedules, "schedules")

        weekly = fetch_weekly_data(args.years)
        save(weekly, "weekly_player_stats")

        if not args.skip_pbp:
            pbp = fetch_pbp_data(args.years)
            save(pbp, "play_by_play")
        else:
            print("Skipping play-by-play (--skip-pbp set)")

    except Exception as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone. Raw data written to data/raw/")


if __name__ == "__main__":
    main()