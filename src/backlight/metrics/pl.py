import math
import numpy as np
import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.positions import calc_positions
from backlight.positions.positions import Positions, calc_pl
from backlight.trades.trades import Trade, Trades, count


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def _trade_amount(amount: pd.Series) -> pd.Series:
    previous_amount = amount.shift(periods=1)
    amount_diff = (amount - previous_amount)[1:]  # drop first nan
    return _sum(amount_diff.abs())


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


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


def calc_trade_performance(
    trades: Trades, mkt: MarketData, principal: float = 0.0
) -> pd.DataFrame:
    """Evaluate the pl perfomance of trades

    Args:
        trades:  Trades. All the index of `trades` should be in `mkt`.
        mkt: Market data.
        principal: Positions' principal is initialized by this.

    Returns:
        metrics
    """
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

    positions = calc_positions(trades, mkt, principal=principal)
    m = pd.concat([m.T, calc_position_performance(positions)], axis=1)

    m.loc[:, "avg_win_pl"] = _divide(
        m.loc["metrics", "total_win_pl"], m.loc["metrics", "cnt_win"]
    )
    m.loc[:, "avg_lose_pl"] = _divide(
        m.loc["metrics", "total_lose_pl"], m.loc["metrics", "cnt_lose"]
    )
    m.loc[:, "avg_pl_per_trade"] = _divide(
        m.loc["metrics", "total_pl"], m.loc["metrics", "cnt_trade"]
    )

    return m
