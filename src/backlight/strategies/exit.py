import numpy as np
import pandas as pd

from typing import Callable, List, Optional, Tuple

from backlight.datasource.marketdata import MarketData
from backlight.labelizer.common import TernaryDirection
from backlight.signal.signal import Signal
from backlight.trades import make_trade
from backlight.trades.trades import Transaction, Trades, concat, from_dataframe
from backlight.strategies.common import Action


def _concat(mkt: MarketData, sig: Optional[Signal]) -> pd.DataFrame:
    if sig is None:
        return mkt

    assert mkt.symbol == sig.symbol
    # Assume sig is less frequent than mkt.
    assert all([idx in mkt.index for idx in sig.index])
    df = pd.concat([mkt, sig], axis=1, join="inner")
    df.symbol = mkt.symbol
    return df


def _exit_transaction(
    df: pd.DataFrame,
    trade: pd.Series,
    exit_condition: Callable[[pd.DataFrame, pd.Series, pd.Timestamp], bool],
) -> Transaction:
    for index in df.index:
        if exit_condition(df, trade, index):
            return Transaction(timestamp=index, amount=-trade.sum())
    return Transaction(timestamp=df.index[-1], amount=-trade.sum())


def _no_exit_condition(df: pd.DataFrame, trade: pd.Series, index: pd.Timestamp) -> bool:
    return False


def exit(
    mkt: MarketData,
    sig: Optional[Signal],
    entries: Trades,
    exit_condition: Callable[[pd.DataFrame, pd.Series, pd.Timestamp], bool],
) -> Trades:
    """Exit trade when satisfying condition.

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
    df = _concat(mkt, sig)

    def _exit(
        trades: Trades,
        df: pd.DataFrame,
        exit_condition: Callable[[pd.DataFrame, pd.Series, pd.Timestamp], bool],
    ) -> pd.Series:

        indices = []  # type: List[pd.Timestamp]
        exits = []  # type: List[Tuple[float, int]]
        for i in trades.ids:
            trade = trades.get_trade(i)
            if trade.sum() == 0:
                continue

            idx = trade.index[0]
            df_exit = df[idx <= df.index]
            transaction = _exit_transaction(df_exit, trade, exit_condition)

            indices.append(transaction.timestamp)
            exits.append((transaction.amount, i))

        df = pd.DataFrame(index=indices, data=exits, columns=["amount", "_id"])

        return from_dataframe(df, symbol)

    symbol = entries.symbol
    exits = _exit(entries, df, exit_condition)
    return concat([entries, exits])


def exit_by_max_holding_time(
    mkt: MarketData,
    sig: Optional[Signal],
    entries: Trades,
    max_holding_time: pd.Timedelta,
    exit_condition: Callable[[pd.DataFrame, pd.Series, pd.Timestamp], bool],
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
    df = _concat(mkt, sig)

    def _exit_by_max_holding_time(
        trades: Trades,
        df: pd.DataFrame,
        max_holding_time: pd.Timedelta,
        exit_condition: Callable[[pd.DataFrame, pd.Series, pd.Timestamp], bool],
    ) -> Trades:

        indices = []  # type: List[pd.Timestamp]
        exits = []  # type: List[Tuple[float, int]]
        for i in trades.ids:
            trade = trades.get_trade(i)
            if trade.sum() == 0:
                continue

            idx = trade.index[0]

            start = max(idx, df.index[0])
            end = min(idx + max_holding_time, df.index[-1])

            df_exit = df.loc[start:end]
            transaction = _exit_transaction(df_exit, trade, exit_condition)

            indices.append(transaction.timestamp)
            exits.append((transaction.amount, i))

        df = pd.DataFrame(index=indices, data=exits, columns=["amount", "_id"])
        return from_dataframe(df, symbol)

    symbol = entries.symbol
    exits = _exit_by_max_holding_time(entries, df, max_holding_time, exit_condition)
    return concat([entries, exits])


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

    def _exit_at_opposite_signals_condition(
        df: pd.DataFrame, opposite_signals_dict: dict, index: pd.Timestamp
    ) -> bool:
        opposite_signals = opposite_signals_dict[TernaryDirection(df["pred"][0])]
        return df["pred"].at[index] in opposite_signals

    def _exit_condition(df: pd.DataFrame, trade: pd.Series, index: pd.Timestamp) -> bool:
        return _exit_at_opposite_signals_condition(df, opposite_signals_dict, index)

    return exit_by_max_holding_time(
        mkt, sig, entries, max_holding_time, _exit_condition
    )


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

    def _exit_by_expectation_condition(
        df: pd.DataFrame, trade: pd.Series, index: pd.Timestamp
    ) -> bool:
        return (
            TernaryDirection(df["pred"].at[df.index[0]]).value
            * (df["up"].at[index] - df["down"].at[index])
            < 0.0
        )

    return exit_by_max_holding_time(
        mkt, sig, entries, max_holding_time, _exit_by_expectation_condition
    )


def exit_by_trailing_stop(
    mkt: MarketData, entries: Trades, initial_stop: float, trailing_stop: float
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

      Returns:
        trades : All trades for entry and exit.
    """
    assert initial_stop >= 0.0
    assert trailing_stop >= 0.0

    def _exit_by_trailing_stop(
        df: pd.DataFrame, trade: pd.Series, index: pd.Timestamp
    ) -> bool:

        amount = trade.sum()
        entry_price = df.mid.iat[0]

        if np.sign(amount) * (df.mid.at[index] - entry_price) <= -initial_stop:
            return True

        prices = df.mid
        pl_per_amount = np.sign(amount) * (prices - entry_price)
        historical_max_pl = pl_per_amount.cummax()
        drawdown = historical_max_pl - pl_per_amount

        if (
            historical_max_pl.at[index] >= trailing_stop
            and drawdown.at[index] >= trailing_stop
        ):
            return True

        return False

    return exit(mkt, None, entries, _exit_by_trailing_stop)


def exit_at_loss_and_gain(
    mkt: MarketData,
    sig: Optional[Signal],
    entries: Trades,
    max_holding_time: pd.Timedelta,
    loss_threshold: float,
    gain_threshold: float,
) -> Trades:

    df = _concat(mkt, sig)

    def _exit_at_loss_and_gain(
        df: pd.DataFrame, trade: pd.Series, index: pd.Timestamp
    ) -> bool:

        amount = trade.sum()
        entry_price = df.mid.iat[0]
        value = np.sign(amount) * (df.mid.at[index] - entry_price)

        if value <= -loss_threshold or value >= gain_threshold:
            return True

        return False

    return exit_by_max_holding_time(
        mkt, None, entries, max_holding_time, _exit_at_loss_and_gain
    )
