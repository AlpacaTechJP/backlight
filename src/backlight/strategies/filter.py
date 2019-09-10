import pandas as pd
import numpy as np
from typing import List, Type


from backlight.trades.trades import Trades, from_dataframe
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


def filter_entry_by_time(
    trades: Trades, unit: str, container_set: tuple
) -> Type["Trades"]:
    """Filter trade which match conditions at least one element. 
        -> e.g. for container_set = [1,2] and unit = 'hour' Trades of hour 1 or 2 
            will be return.
    Args:
        container_set: The results will only contain elements of time in this set
        unit: Can be 'hour', 'minute'... The results.time will be in the set

    Returns:
        Trades.
    """

    sort = trades.sort_values("_id")

    df = pd.DataFrame(
        data=np.zeros((sort.shape[0], 3)), columns=["amount", "_id", "index"]
    )

    j = 0

    for i in range(0, sort.index.size, 2):
        entry_index = sort.index[i]
        exit_index = sort.index[i + 1]
        if (
            getattr(entry_index, unit) in container_set
            or getattr(exit_index, unit) in container_set
        ):
            # This code is faster than an iloc.
            df.at[j, "amount"] = sort.iat[i, 0]
            df.at[j + 1, "amount"] = sort.iat[i + 1, 0]
            df.at[j, "_id"] = sort.iat[i, 1]
            df.at[j + 1, "_id"] = sort.iat[i + 1, 1]
            df.at[j, "index"] = entry_index
            df.at[j + 1, "index"] = exit_index

            j += 2

    df = df.iloc[0:j, :].sort_values("index").set_index("index")
    df.index.name = None

    return from_dataframe(df, trades.symbol, trades.currency_unit)


def skip_entry_by_hours(trades: Trades, hours: List[int]) -> Type["Trades"]:
    """Skip entry by hours.

    Args:
        trades: Trades
        hours: Hours which will be filtered out from entry.
    Result:
        Trades
    """
    deleted_ids = []
    for i in trades.ids:
        entry = trades.get_trade(i).index[0]
        if entry.hour in hours:
            deleted_ids.append(i)

    return trades[~trades["_id"].isin(deleted_ids)]
