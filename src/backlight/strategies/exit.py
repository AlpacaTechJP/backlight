import pandas as pd

from typing import Callable, List

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import make_trade
from backlight.trades.trades import Transaction, Trade, Trades
from backlight.strategies.common import Action, concat


def _exit_transaction(
    df: pd.DataFrame, amount: float, exit_condition: Callable[[pd.DataFrame], pd.Series]
) -> Transaction:
    exit_indices = df[exit_condition(df)].index
    if exit_indices.empty:
        exit_index = df.index[-1]
    else:
        exit_index = exit_indices[0]
    return Transaction(timestamp=exit_index, amount=-amount)


def _exit(
    trade: Trade,
    df: pd.DataFrame,
    max_holding_time: pd.Timedelta,
    exit_condition: Callable[[pd.DataFrame], pd.Series],
) -> Trade:
    idx = trade.index[0]
    amount = trade.amount.sum()
    df_exit = df[(idx <= df.index) & (df.index <= idx + max_holding_time)]
    transaction = _exit_transaction(df_exit, amount, exit_condition)
    trade.add(transaction)
    return trade


def direction_based_exit(
    mkt: MarketData,
    sig: Signal,
    entries: Trades,
    max_holding_time: pd.Timedelta,
    exit_condition: Callable[[pd.DataFrame], pd.Series],
) -> Trades:
    """Take positions.

    Args:
        mkt: Market data
        sig: Signal data
        entries: Tuple of entry trades.
        max_holding_time: maximum holding time
        exit_condition: The entry is closed most closest time which
                        condition is `True`.
    Result:
        Trades
    """
    df = concat(mkt, sig)

    trades = tuple(
        _exit(trade, df, max_holding_time, exit_condition) for trade in entries
    )
    return trades
