import pandas as pd

from typing import List

from backlight.datasource import MarketData
from backlight.positions.positions import Positions
from backlight.trades import Trade, count, flatten, evaluate_pl


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def _pl(positions: Positions) -> pd.Series:
    next_price = positions.price.shift(periods=-1)
    price_diff = next_price - positions.price
    pl = (price_diff * positions.amount).shift(periods=1)[1:]  # drop first nan
    return pl.rename("pl")


def _trade_amount(amount: pd.Series) -> pd.Series:
    previous_amount = amount.shift(periods=1)
    amount_diff = (amount - previous_amount)[1:]  # drop first nan
    return _sum(amount_diff.abs())


def calc_trade_peformance(trades: List[Trade], mkt: MarketData) -> pd.DataFrame:
    total_count, win_count, lose_count = count(trades, mkt)
    trade = flatten(trades)
    pl = evaluate_pl(trade, mkt)

    m = pd.DataFrame.from_records(
        [
            ("cnt_total", total_count),
            ("cnt_win", win_count),
            ("cnt_lose", lose_count),
            ("win_ratio", win_count / total_count),
            ("lose_ratio", lose_count / total_count),
            ("pl", pl),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T


def calc_position_performance(positions: Positions) -> pd.DataFrame:
    """Evaluate the pl perfomance of positions"""
    pl = _pl(positions)
    trade_amount = _trade_amount(positions.amount)

    total_pl = _sum(pl)
    average_pl = total_pl / trade_amount

    m = pd.DataFrame.from_records(
        [("avg_pl", average_pl), ("total_pl", total_pl), ("trade_amount", trade_amount)]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T
