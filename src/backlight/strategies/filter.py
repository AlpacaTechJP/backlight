import pandas as pd
from typing import List

from backlight.trades.trades import Trades
from backlight.datasource.marketdata import AskBidMarketData


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


def skip_entry_by_spread(
    trades: Trades, mkt: AskBidMarketData, max_spread: float
) -> Trades:
    """Skip entry by spread.

    Args:
        trades: Trades
        mkt: Market data for ask/bid prices
        max_spread: More than the value, skip entry
    Result:
        Trades
    """
    assert max_spread >= 0.0
    assert trades.index.unique().isin(mkt.index).all()

    spread = mkt.spread

    deleted_ids = []  # type: List[int]
    for i in trades.ids:
        entry = trades.get_trade(i).index[0]
        if spread.at[entry] > max_spread:
            deleted_ids.append(i)

    return trades[~trades["_id"].isin(deleted_ids)]
