import math
import numpy as np
import pandas as pd
from typing import Tuple, Union

from backlight.datasource.marketdata import MarketData
from backlight.positions import calculate_positions
from backlight.positions.positions import Positions
from backlight.portfolio.portfolio import Portfolio


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def calculate_pl(positions_or_portfolio: Union[Positions, Portfolio]) -> pd.Series:
    """Compute pl of a positions or portfolio.

    Args:
        positions_or_portfolio: Positions or Portfolio to be evaluated.

    Returns:
        Series of pl.
    """
    next_value = positions_or_portfolio.value.shift(periods=-1)
    pl = (next_value - positions_or_portfolio.value).shift(periods=1)[
        1:
    ]  # drop first nan
    return pl.rename("pl")


def calculate_sharpe(
    positions_or_portfolio: Union[Positions, Portfolio], freq: pd.Timedelta
) -> float:
    """Compute the yearly Sharpe ratio, a measure of risk adjusted returns.

    Args:
        positions_or_portfolio: Their `value` should always be positive.
        freq: Frequency to calculate mean and std of returns.

    Returns:
        sharpe ratio.
    """
    value = positions_or_portfolio.value.resample(freq).first().dropna()
    previous_value = value.shift(periods=1)
    log_return = np.log((value.values / previous_value.values)[1:])

    days_in_year = pd.Timedelta("252D")
    annual_factor = math.sqrt(days_in_year / freq)
    return annual_factor * np.mean(log_return) / np.std(log_return)


def calculate_drawdown(
    positions_or_portfolio: Union[Positions, Portfolio]
) -> pd.Series:
    """Compute drawdown c.f. https://en.wikipedia.org/wiki/Drawdown_(economics)

    Args:
        positions_or_portfolio: Positions or Portfolio.

    Returns:
        Drawdown in the periods.
    """
    histrical_max = positions_or_portfolio.value.cummax()
    value = positions_or_portfolio.value
    return histrical_max - value


def calculate_performance(
    positions_or_portfolio: Union[Positions, Portfolio],
    window: pd.Timedelta = pd.Timedelta("1D"),
) -> pd.DataFrame:
    """Evaluate the pl perfomance of positions

    Args:
        positions_or_portfolio: Positions or Portfolio to be evaluated.
        window: Window for `calculate_sharpe`.

    Returns:
        DataFrame of perfomance
    """
    pl = calculate_pl(positions_or_portfolio)

    total_pl = _sum(pl)
    win_pl = _sum(pl[pl > 0.0])
    lose_pl = _sum(pl[pl < 0.0])
    sharpe = calculate_sharpe(positions_or_portfolio, window)
    drawdown = calculate_drawdown(positions_or_portfolio)

    m = pd.DataFrame.from_records(
        [
            ("total_pl", total_pl),
            ("total_win_pl", win_pl),
            ("total_lose_pl", lose_pl),
            ("sharpe", sharpe),
            ("max_drawdown", drawdown.max()),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
