import numpy as np
import pandas as pd

from typing import Callable, List

from backlight.datasource.marketdata import MarketData
from backlight.labelizer.common import TernaryDirection
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


def _exit_by_max_holding_time(
    trade: Trade, df: pd.DataFrame, exit_condition: Callable[[pd.DataFrame], pd.Series]
) -> Trade:
    idx = trade.index[0]
    amount = trade.amount.sum()
    df_exit = df[idx <= df.index]
    transaction = _exit_transaction(df_exit, amount, exit_condition)
    trade.add(transaction)
    return trade


def _exit_with_max_holding_time(
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


def _no_exit_condition(df: pd.DataFrame) -> pd.Series:
    return pd.Series(index=df.index, data=False)


def exit_by_max_holding_time(
    mkt: MarketData,
    sig: Signal,
    entries: Trades,
    max_holding_time: pd.Timedelta,
    exit_condition: Callable[[pd.DataFrame], pd.Series],
) -> Trades:
    """Exit trade at max holding time or satisfying condition.

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
        _exit_with_max_holding_time(trade, df, max_holding_time, exit_condition)
        for trade in entries
    )
    return trades


def exit_at_max_holding_time(
    mkt: MarketData, sig: Signal, entries: Trades, max_holding_time: pd.Timedelta
) -> Trades:
    """Exit at max holding time.

    Args:
        mkt: Market data
        sig: Signal data
        entries: Tuple of entry trades.
        max_holding_time: maximum holding time
    Result:
        Trades
    """
    return exit_by_max_holding_time(
        mkt, sig, entries, max_holding_time, _no_exit_condition
    )


def _exit_at_opposite_signals_condition(
    df: pd.DataFrame, opposite_signals_dict: dict
) -> pd.Series:
    current_signal = TernaryDirection(df["pred"][0])
    opposite_signals = opposite_signals_dict[current_signal]
    return df["pred"].isin(opposite_signals)


def exit_at_opposite_signals(
    mkt: MarketData,
    sig: Signal,
    entries: Trades,
    max_holding_time: pd.Timedelta,
    opposite_signals_dict: dict,
) -> Trades:
    """Exit at max holding time or opposite signals.

    Args:
        mkt: Market data
        sig: Signal data
        entries: Tuple of entry trades.
        max_holding_time: maximum holding time
        opposite_signals_dict: Dictionary to define opposite signals for each signal.
    Result:
        Trades
    """

    def _exit_condition(df: pd.DataFrame) -> pd.Series:
        return _exit_at_opposite_signals_condition(df, opposite_signals_dict)

    return exit_by_max_holding_time(
        mkt, sig, entries, max_holding_time, _exit_condition
    )


def _exit_by_expectation_condition(df: pd.DataFrame) -> pd.Series:
    current_signal = TernaryDirection(df["pred"][0])
    v = np.array([1.0, 0.0, -1.0])
    expectation = np.dot(df[["up", "neutral", "down"]].values, v)
    expectation = current_signal.value * expectation
    sign = expectation < 0.0
    return pd.Series(index=df.index, data=sign)


def exit_by_expectation(
    mkt: MarketData, sig: Signal, entries: Trades, max_holding_time: pd.Timedelta
) -> Trades:
    """Exit at max holding time or by expectation.

    Args:
        mkt: Market data
        sig: Signal data
        entries: Tuple of entry trades.
        max_holding_time: maximum holding time
    Result:
        Trades
    """

    def _exit_condition(df: pd.DataFrame) -> pd.Series:
        return _exit_by_expectation_condition(df)

    return exit_by_max_holding_time(
        mkt, sig, entries, max_holding_time, _exit_condition
    )


def exit_by_trailing_stop(
    mkt: MarketData,
    entries: Trades,
    initial_stop: float,
    trailing_stop: float,
    draw_positions: bool = True,
) -> Trades:
    """Trailing stop exit strategy.

      Given the list of entries, it simulates exits by using the trailing stop logic.
      The marketdata defines the range for simulation. In case you want to clear all
      positions at the end of the day, you have to limit the end edge of marketdata,
      and call this function for each day.

      Args:
        mkt            : Market data
        entries        : List of entries
        initial_stop   : Initial stop in absolute price.
        trailing_stop  : Trailing stop in absolute price.
        draw_positions : Close all positions at the end of marketdata.

      Returns:
        trades : All trades for entry and exit.
    """
    trades = tuple(
        _exit_with_max_holding_time(trade, df, exit_condition) for trade in entries
    )
    return trades
