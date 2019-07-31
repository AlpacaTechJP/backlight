import math
import numpy as np
import pandas as pd
from typing import Tuple

from backlight.datasource.marketdata import MarketData
from backlight.positions import calc_positions
from backlight.positions.positions import Positions
from backlight.metrics.evaluation_metrics import calc_pl, calc_sharpe, calc_drawdown


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _trade_amount(amount: pd.Series) -> pd.Series:
    previous_amount = amount.shift(periods=1)
    amount_diff = (amount - previous_amount)[1:]  # drop first nan
    return _sum(amount_diff.abs())


# calculate_position_performance?
def calc_position_performance(
    positions: Positions, window: pd.Timedelta = pd.Timedelta("1D")
) -> pd.DataFrame:
    """Evaluate the pl perfomance of positions

    Args:
        positions: Positions to be evaluated.
        window: Window for `calc_sharpe`.

    Returns:
        DataFrame of perfomance
    """
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
