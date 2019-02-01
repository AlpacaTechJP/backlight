import math
import numpy as np
import pandas as pd
from typing import Tuple

import backlight.positions
from backlight.datasource.marketdata import MarketData
from backlight.trades.trades import Trades, make_trades
from backlight.metrics.position_metrics import calc_pl, calc_position_performance


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0.0


def _calc_pl(trade: pd.Series, mkt: MarketData) -> float:
    mkt = mkt.loc[trade.index, :]
    trades = make_trades(mkt.symbol, [trade])
    positions = backlight.positions.calc_positions(trades, mkt)
    pl = calc_pl(positions)
    return _sum(pl)


def count_trades(trades: Trades, mkt: MarketData) -> Tuple[int, int, int]:
    """Count winning, losing and total number of trades.

    Args:
        trades : Trades to evaluate. Each trade is evaluated
                 only if it contains more than one transactions,
                 because we can define pl in that case.
        mkt: Market data. The index should contains all trades' index.

    Returns:
        total count, win count, lose count
    """
    pls = [
        _calc_pl(trades.get_trade(i), mkt)
        for i in trades.ids
        if len(trades.get_trade(i).index) > 1
    ]
    total = len(trades.ids)
    win = sum([pl > 0.0 for pl in pls])
    lose = sum([pl < 0.0 for pl in pls])
    return total, win, lose


def calc_trade_performance(
    trades: Trades, mkt: MarketData, principal: float = 0.0
) -> pd.DataFrame:
    """Evaluate the pl perfomance of trades and positions.

    Args:
        trades:  Trades to evaluate. Trades will be flattened as Positions.
        mkt: Market data. The index should contains all trades' index.
        principal: Positions' principal is initialized by this value.

    Returns:
        Metrics of trades and positions.
    """
    total_count, win_count, lose_count = count_trades(trades, mkt)

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

    positions = backlight.positions.calc_positions(trades, mkt, principal=principal)
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
