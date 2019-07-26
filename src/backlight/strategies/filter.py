import pandas as pd
from typing import List

from backlight.trades.trades import Trades, remove_trades
from backlight.datasource.marketdata import MarketData


def limit_max_amount(trades: Trades, max_amount: int) -> Trades:
    """Limit trade by max amount.

    Args:
        trades: Trades
        max_amount: Max amount in absolute value 
    Result:
        Trades
    """
    assert max_amount > 0.0

    current_amount = 0.0
    deleted_ids = []  # type: List[int]
    for index, row in trades.iterrows():

        if row["_id"] in deleted_ids:
            continue

        next_amount = current_amount + row["amount"]
        if abs(next_amount) > max_amount:
            deleted_ids.append(row["_id"])
            continue

        current_amount = next_amount

    return trades[~trades["_id"].isin(deleted_ids)]


def filter_entry_by_indicator(
    trades: Trades, mkt: MarketData
) -> Trades:
    """ TODO
    """
    assert trades.index.unique().isin(mkt.index).all()

    def indicator_func(mkt):
        window = "30min"
        moving_average = mkt.mid.rolling(window).mean()
        return ((mkt.mid - moving_average) / moving_average).abs()

    indicator = indicator_func(mkt)

    threshold = 0.0

    deleted_ids = []  # type: List[int]
    for i in trades.ids:
        entry = trades.get_trade(i).index[0]
        if indicator.loc[entry] > threshold:
            deleted_ids.append(i)

    return trades[~trades["_id"].isin(deleted_ids)]
