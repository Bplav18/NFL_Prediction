"""
Season-based walk-forward validation.

Standard k-fold cross-validation shuffles rows randomly, which for time-series
data like NFL seasons would let a model "see the future" — e.g. training on
2023 data to help predict a 2020 game. Walk-forward validation instead trains
only on seasons strictly before the test season, mimicking how the model
would actually be used in production: predicting a season you haven't seen
yet using only what happened before it.
"""

from typing import Iterator

import pandas as pd


def season_walk_forward_splits(
    df: pd.DataFrame,
    season_col: str = "season",
    min_train_seasons: int = 2,
) -> Iterator[tuple[int, pd.Index, pd.Index]]:
    """
    Yields (test_season, train_index, test_index) tuples.

    For each season after the first `min_train_seasons`, train on every
    season strictly before it and test on that season alone.

    Example with seasons [2018, 2019, 2020, 2021] and min_train_seasons=2:
        test=2020, train=[2018, 2019]
        test=2021, train=[2018, 2019, 2020]
    """
    seasons = sorted(df[season_col].unique())

    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Need more than {min_train_seasons} seasons of data; "
            f"got {len(seasons)}: {seasons}"
        )

    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]

        train_idx = df.index[df[season_col].isin(train_seasons)]
        test_idx = df.index[df[season_col] == test_season]

        yield test_season, train_idx, test_idx