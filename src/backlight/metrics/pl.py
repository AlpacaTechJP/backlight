import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.positions import calc_positions
from backlight.positions.positions import Positions
from backlight.trades.trades import Trade, Trades, count


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


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def calc_position_performance(positions: Positions) -> pd.DataFrame:
    """Evaluate the pl perfomance of positions"""
    pl = _pl(positions)
    trade_amount = _trade_amount(positions.amount)

    total_pl = _sum(pl)
    win_pl = _sum(pl[pl > 0.0])
    lose_pl = _sum(pl[pl < 0.0])
    average_pl = _divide(total_pl, trade_amount)

    m = pd.DataFrame.from_records(
        [
            ("avg_pl_per_amount", average_pl),
            ("total_pl", total_pl),
            ("total_win_pl", win_pl),
            ("total_lose_pl", lose_pl),
            ("cnt_amount", trade_amount),
        ]
    ).set_index(0)

    del m.index.name
    m.columns = ["metrics"]

    return m.T


def calc_trade_performance(trades: Trades, mkt: MarketData) -> pd.DataFrame:
    total_count, win_count, lose_count = count(trades, mkt)

    m = pd.DataFrame.from_records(
        [
            ("cnt_trade", total_count),
            ("cnt_win", win_count),
            ("cnt_lose", lose_count),
            ("win_ratio", _divide(win_count, total_count)),
            ("lose_ratio", _divide(lose_count, total_count)),
        ]
    ).set_index(0)
    del m.index.name
    m.columns = ["metrics"]

    positions = calc_positions(trades, mkt)
    m = pd.concat([m.T, calc_position_performance(positions)], axis=1)

    m.loc[:, "avg_win_pl"] = _divide(m["total_win_pl"], m["cnt_win"])
    m.loc[:, "avg_lose_pl"] = _divide(m["total_lose_pl"], m["cnt_lose"])
    m.loc[:, "avg_pl_per_trade"] = _divide(m["total_pl"], m["cnt_trade"])

    return m
