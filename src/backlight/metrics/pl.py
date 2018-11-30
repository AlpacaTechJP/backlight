import pandas as pd

from backlight.positions.positions import Positions


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def calc_position_performance(positions: Positions):
    """Evaluate the pl perfomance of positions"""
    next_price = positions.price.shift(periods=-1)
    price_diff = next_price - positions.price
    pl = (price_diff * positions.amount).shift(periods=1)[1:]  # drop first nan

    previous_amount = positions.amount.shift(periods=1)
    amount_diff = (positions.amount - previous_amount)[1:]  # drop first nan

    total_pl = _sum(pl)
    trade_amount = _sum(amount_diff.abs())
    average_pl = total_pl / trade_amount

    m = pd.DataFrame.from_records(
        [
            ("avg_pl", average_pl),
            ("total_pl", total_pl),
            ("trade_amount", trade_amount),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T

