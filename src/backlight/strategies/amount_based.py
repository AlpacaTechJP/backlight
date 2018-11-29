import pandas as pd
import numpy as np

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import Trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


def _make_trades(trades: pd.DataFrame, mkt: MarketData, symbol: str) -> Trades:
    """
    Order of concat is important, if mkt in front of trades,
    it will have error since it will use the type of mkt(MarketData)
    to concat, but trades is normal DataFrame.
    """
    t = Trades(pd.concat([trades, mkt], axis=1, join="inner"))
    t.symbol = symbol
    return t


def _check_inputs(mkt: MarketData, sig: Signal) -> None:
    assert mkt.symbol == sig.symbol
    # Assume sig is less frequent than mkt.
    assert all([idx in mkt.index for idx in sig.index])


def direction_based_trades(
    mkt: MarketData, sig: Signal, direction_action_dict: dict
) -> Trades:
    """

    Args:
        ternary_action_dict(dict<TernaryDirection,Action>)
    """
    _check_inputs(mkt, sig)
    trades = pd.DataFrame(index=sig.index, columns=["amount"]).astype(np.float64)
    for direction, action in direction_action_dict.items():
        trades.loc[sig["pred"] == direction.value, "amount"] = action.act_on_amount()
    t = _make_trades(trades, mkt, sig.symbol)
    return t


def _no_condition(df: pd.DataFrame) -> pd.Series:
    return pd.Series(index=df.index, data=False)


def entry_exit_trades(
    mkt: MarketData,
    sig: Signal,
    direction_action_dict: dict,
    max_holding_time: pd.Timedelta,
    exit_condition=_no_condition,
) -> Trades:
    """
    """
    _check_inputs(mkt, sig)
    df = pd.concat([mkt, sig], axis=1, join="inner")
    trades = pd.DataFrame(index=df.index, data=0, columns=["amount"]).astype(np.float64)
    for idx, row in df.iterrows():
        action = direction_action_dict[TernaryDirection(row["pred"])]
        amount = action.act_on_amount()
        trades.loc[idx, "amount"] += amount

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
        trades.loc[exit_index, "amount"] -= amount

    t = _make_trades(trades, mkt, sig.symbol)
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
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)


def only_entry_short(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)


def simple_entry(
    mkt: MarketData, sig: Signal, max_holding_time: pd.Timedelta
) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return entry_exit_trades(mkt, sig, direction_action_dict, max_holding_time)
