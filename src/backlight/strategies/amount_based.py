import pandas as pd
import numpy as np

from typing import Callable, List

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import make_trade
from backlight.trades.trades import Trades, make_trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action
from backlight.strategies.entry import direction_based_entry
from backlight.strategies.exit import (
    exit_at_max_holding_time,
    exit_at_opposite_signals,
    exit_by_expectation,
    exit_at_loss_at_gain,
)


def only_take_long(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return direction_based_entry(mkt, sig, direction_action_dict)


def only_take_short(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_entry(mkt, sig, direction_action_dict)


def simple_buy_sell(mkt: MarketData, sig: Signal) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_entry(mkt, sig, direction_action_dict)


def _entry_and_exit_at_max_holding_time(
    mkt: MarketData,
    sig: Signal,
    direction_action_dict: dict,
    max_holding_time: pd.Timedelta,
) -> Trades:
    """Take positions and close them within maximum holding time.

    Args:
        mkt: Market data
        sig: Signal data
        direction_action_dict: Dictionary from signals to actions
        max_holding_time: maximum holding time
        exit_condition: The entry is closed most closest time which condition is `True`.

    Result:
        Trades
    """
    entries = direction_based_entry(mkt, sig, direction_action_dict)
    trades = exit_at_max_holding_time(mkt, sig, entries, max_holding_time)
    return trades


def only_entry_long_and_exit(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take only long positions and close them within maximum holding time.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return _entry_and_exit_at_max_holding_time(
        mkt, sig, direction_action_dict, max_holding_time
    )


def only_entry_short_and_exit(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take only short positions and close them within maximum holding time.
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return _entry_and_exit_at_max_holding_time(
        mkt, sig, direction_action_dict, max_holding_time
    )


def simple_entry_and_exit(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    """Take both positions and close them within maximum holding time. """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return _entry_and_exit_at_max_holding_time(
        mkt, sig, direction_action_dict, max_holding_time
    )


def entry_and_exit_at_opposite_signals(
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
    entries = direction_based_entry(mkt, sig, direction_action_dict)

    opposite_signals_dict = {
        TernaryDirection.UP: [TernaryDirection.DOWN.value],
        TernaryDirection.NEUTRAL: [],
        TernaryDirection.DOWN: [TernaryDirection.UP.value],
    }
    trades = exit_at_opposite_signals(
        mkt, sig, entries, max_holding_time, opposite_signals_dict
    )
    return trades


def entry_and_exit_at_other_signals(
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
    entries = direction_based_entry(mkt, sig, direction_action_dict)

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
    trades = exit_at_opposite_signals(
        mkt, sig, entries, max_holding_time, opposite_signals_dict
    )
    return trades


def entry_and_exit_by_expectation(
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

    entries = direction_based_entry(mkt, sig, direction_action_dict)

    trades = exit_by_expectation(mkt, sig, entries, max_holding_time)
    return trades


def enter_exit_loss_gain(
    mkt: MarketData,
    sig: Signal,
    gain_threshold: float,
    loss_threshold: float,
    max_holding: pd.Timedelta,
) -> Trades:
    """Take positions and close them when a certain threshold of gain is attained OR
    loss in attained. Otherwise close it after max_holding
    Args:
        mkt: Market data
        sig: Signal data
        direction_action_dict: Dictionary from signals to actions
        max_holding_time: maximum holding time
        gain_threshold: postive float when to take gain,
        loss_threshold: negative float when to stop loss,
        max_holding: max holding time,
    Result:
        Trades
    """
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    entries = direction_based_entry(mkt, sig, direction_action_dict)
    trades = exit_at_loss_at_gain(
        mkt, sig, entries, max_holding, loss_threshold, gain_threshold
    )
    return trades
