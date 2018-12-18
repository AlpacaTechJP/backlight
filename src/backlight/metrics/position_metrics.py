import math
import numpy as np
import pandas as pd
from typing import Tuple

from backlight.datasource.marketdata import MarketData
from backlight.positions import calc_positions
from backlight.positions.positions import Positions
from backlight.trades.trades import Trade, Trades


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def _trade_amount(amount: pd.Series) -> pd.Series:
    previous_amount = amount.shift(periods=1)
    amount_diff = (amount - previous_amount)[1:]  # drop first nan
    return _sum(amount_diff.abs())


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def calc_pl(positions: Positions) -> pd.Series:
    next_value = positions.value.shift(periods=-1)
    pl = (next_value - positions.value).shift(periods=1)[1:]  # drop first nan
    return pl.rename("pl")


def calc_sharpe(positions: Positions, freq: pd.Timedelta) -> float:
    """Compute the yearly Sharpe ratio, a measure of risk adjusted returns.

    Args:
        positions: Their `value` should always be positive.
        freq: Frequency to calculate mean and std of returns.

    Returns:
        sharpe ratio.
    """
    value = positions.value.resample(freq).first().dropna()
    previous_value = value.shift(periods=1)
    log_return = np.log((value.values / previous_value.values)[1:])

    days_in_year = pd.Timedelta("252D")
    annual_factor = math.sqrt(days_in_year / freq)
    return annual_factor * np.mean(log_return) / np.std(log_return)


def calc_drawdown(positions: Positions) -> pd.Series:
    """Compute drawdown c.f. https://en.wikipedia.org/wiki/Drawdown_(economics)

    Args:
        positions: Positions.

    Returns:
        Drawdown in the periods.
    """
    histrical_max = positions.value.cummax()
    value = positions.value
    return histrical_max - value


def calc_position_performance(
    positions: Positions, window: pd.Timedelta = pd.Timedelta("1D")
) -> pd.DataFrame:
    """Evaluate the pl perfomance of positions"""
    pl = calc_pl(positions)
    trade_amount = _trade_amount(positions.amount)

    total_pl = _sum(pl)
    win_pl = _sum(pl[pl > 0.0])
    lose_pl = _sum(pl[pl < 0.0])
    average_pl = _divide(total_pl, trade_amount)
    sharpe = calc_sharpe(positions, window)
    drawdown = calc_drawdown(positions)

    m = pd.DataFrame.from_records(
        [
            ("avg_pl_per_amount", average_pl),
            ("total_pl", total_pl),
            ("total_win_pl", win_pl),
            ("total_lose_pl", lose_pl),
            ("cnt_amount", trade_amount),
            ("sharpe", sharpe),
            ("max_drawdown", drawdown.max()),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
