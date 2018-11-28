import pandas as pd
import numpy as np

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import Trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


def direction_based_trades(
    mkt: MarketData, sig: Signal, direction_action_dict: dict
) -> Trades:
    """

    Args:
        ternary_action_dict(dict<TernaryDirection,Action>)
    """
    trades = pd.DataFrame(index=sig.index, columns=["amount"]).astype(np.float64)
    for direction, action in direction_action_dict.items():
        trades.loc[sig["pred"] == direction.value, "amount"] = action.act_on_amount()
    # Order of concat is important, if mkt in front of trades, will have error
    # since it will use the type of mkt(MarketData) to concat, but trades is
    # normal DataFrame
    t = Trades(pd.concat([trades, mkt], axis=1, join="inner"))
    t.symbol = sig.symbol
    return t


def _no_condition(df: pd.DataFrame):
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
    df = pd.concat([mkt, sig], axis=1, join="inner")
    df.loc[:, "amount"] = 0.0
    for idx, row in df.iterrows():
        action = direction_action_dict[row["pred"]]
        amount = action.act_on_amount
        df[idx, "amount"] += amount

        df_to_max_holding_time = df[
            (idx <= df.index) & (df.index <= idx + max_holding_time)
        ]
        exit_indices = df_to_max_holding_time[
            exit_condition(df_to_max_holding_time)
        ].index
        if exit_indices.empty():
            exit_index = df_to_max_holding_time.index[-1]
        else:
            exit_index = exit_indices[0]
        df[exit_index, "amount"] -= amount

    # Order of concat is important, if mkt in front of trades, will have error
    # since it will use the type of mkt(MarketData) to concat, but trades is
    # normal DataFrame
    t = Trades(df)
    t.symbol = sig.symbol
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
