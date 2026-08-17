"""
Converts American moneyline odds into de-vigged implied win probabilities.

Sportsbooks price both sides of a game so their combined implied probability
exceeds 100% — the difference is the "vig" (their built-in margin). Comparing
a model's probability directly against a single side's raw implied probability
would be misleading, since the raw number is inflated by the vig. This module
removes it so the market probability represents a fair, break-even estimate.
"""

import pandas as pd


def american_odds_to_raw_prob(odds: float) -> float:
    """Converts a single American moneyline to its raw (vig-inflated) implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def devig_two_way(home_odds: float, away_odds: float) -> tuple[float, float]:
    """
    Normalizes home/away moneylines into fair (de-vigged) probabilities
    that sum to exactly 1.0.
    """
    raw_home = american_odds_to_raw_prob(home_odds)
    raw_away = american_odds_to_raw_prob(away_odds)
    overround = raw_home + raw_away  # > 1.0 due to the vig

    fair_home = raw_home / overround
    fair_away = raw_away / overround
    return fair_home, fair_away


def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `market_home_prob` and `market_away_prob` columns to a DataFrame
    that has `home_moneyline` and `away_moneyline` columns.
    """
    df = df.copy()
    probs = df.apply(
        lambda row: devig_two_way(row["home_moneyline"], row["away_moneyline"]),
        axis=1,
    )
    df["market_home_prob"] = probs.apply(lambda p: p[0])
    df["market_away_prob"] = probs.apply(lambda p: p[1])
    return df