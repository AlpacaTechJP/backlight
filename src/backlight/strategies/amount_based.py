import pandas as pd
import numpy as np

from typing import Callable

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import Trade, Trades, Transaction, make_trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


def _concat(mkt: MarketData, sig: Signal) -> pd.DataFrame:
    assert mkt.symbol == sig.symbol
    # Assume sig is less frequent than mkt.
    assert all([idx in mkt.index for idx in sig.index])
    return pd.concat([mkt, sig], axis=1, join="inner")


def direction_based_trades(
    mkt: MarketData, sig: Signal, direction_action_dict: dict
) -> Trades:
    """Just take trades without closing them.

    Args:
        mkt: Market data
        sig: Signal data
        direction_action_dict: Dictionary from signals to actions
    Result:
        Trades
    """
    df = _concat(mkt, sig)
    trades = pd.DataFrame(index=df.index, columns=["amount"]).astype(np.float64)
    trades = []
    for direction, action in direction_action_dict.items():
        index = df[df.pred == direction.value].index
        for idx in index:
            trade = Trade().add(Transaction(timestamp=idx, amount=action.act_on_amount))
            trades.append(trade)
    t = make_trades(trades, mkt)
    return t


def _no_exit(df: pd.DataFrame) -> pd.Series:
    return pd.Series(index=df.index, data=False)


def _exit_opposite_signals(df: pd.DataFrame, opposite_signals_dict: dict) -> pd.Series:
    current_signal = TernaryDirection(df["pred"][0])
    opposite_signals = opposite_signals_dict[current_signal]
    return df["pred"].isin(opposite_signals)


def entry_exit_trades(
    mkt: MarketData,
    sig: Signal,
    direction_action_dict: dict,
    max_holding_time: pd.Timedelta,
    exit_condition: Callable[[pd.DataFrame], pd.Series] = _no_exit,
) -> Trades:
    """Take positions and close them within maximum holding time.

    Args:
        mkt: Market data
        sig: Signal data
        direction_action_dict: Dictionary from signals to actions
        max_holding_time: maximum holding time
        exit_condition: The entry is closed most closest time which 
                        condition is `True`.
    Result:
        Trades
    """
    df = _concat(mkt, sig)

    trades = []
    for idx, row in df.iterrows():
        action = direction_action_dict[TernaryDirection(row["pred"])]
        amount = action.act_on_amount()
        trade = Trade().add(Transaction(timestamp=idx, amount=amount))
        df_to_max_holding_time = df[
            (idx <= df.index) & (df.index <= idx + max_holding_time)
        ]
        exit_indices = df_to_max_holding_time[
            exit_condition(df_to_max_holding_time)
        ].index
        if exit_indices.empty:
            exit_index = df_to_max_holding_time.index[-1]
        else:
            exit_index = exit_indices[0]
        trade.add(Transaction(timestamp=exit_index, amount=-amount))
        trades.append(trade)

    t = make_trades(trades, mkt)
    return t


def only_take_long(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return direction_based_trades(mkt, sig, direction_action_dict)


def only_take_short(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, direction_action_dict)


def simple_buy_sell(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, direction_action_dict)


def only_entry_long(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take only long positions and close them within maximum holding time.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)


def only_entry_short(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take only short positions and close them within maximum holding time.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)


def simple_entry(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take both positions and close them within maximum holding time. """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)


def exit_on_oppsite_signals(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """
    Take both positions and close them within maximum holding time.
    If opposite signals appear, also close positions.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }

    opposite_signals_dict = {
        TernaryDirection.UP: [TernaryDirection.DOWN.value],
        TernaryDirection.NEUTRAL: [],
        TernaryDirection.DOWN: [TernaryDirection.UP.value],
    }

    def _exit_condition(mkt: MarketData) -> pd.Series:
        return _exit_opposite_signals(mkt, opposite_signals_dict)

    return entry_exit_trades(
        mkt, sig, direction_action_dict, max_holding_time, _exit_condition
    )


def exit_on_other_signals(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """
    Take both positions and close them within maximum holding time.
    If other signals appear, also close positions.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }

    opposite_signals_dict = {
        TernaryDirection.UP: [
            TernaryDirection.DOWN.value,
            TernaryDirection.NEUTRAL.value,
        ],
        TernaryDirection.NEUTRAL: [],
        TernaryDirection.DOWN: [
            TernaryDirection.UP.value,
            TernaryDirection.NEUTRAL.value,
        ],
    }

    def _exit_condition(mkt: MarketData) -> pd.Series:
        return _exit_opposite_signals(mkt, opposite_signals_dict)

    return entry_exit_trades(
        mkt, sig, direction_action_dict, max_holding_time, _exit_condition
    )
