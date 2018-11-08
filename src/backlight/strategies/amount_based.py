import pandas as pd
import numpy as np

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import Trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


def direction_based_trades(
    mkt: MarketData, sig: Signal, target_column_name: str, direction_action_dict: dict
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
    t.target_column_name = target_column_name
    t.symbol = sig.symbol
    return t


def only_take_long(mkt: MarketData, sig: Signal, target_column_name: str) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)


def only_take_short(mkt: MarketData, sig: Signal, target_column_name: str) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)


def simple_buy_sell(mkt: MarketData, sig: Signal, target_column_name: str) -> Trades:
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)
