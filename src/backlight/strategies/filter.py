import pandas as pd
from typing import List

from backlight.trades.trades import Trades


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
