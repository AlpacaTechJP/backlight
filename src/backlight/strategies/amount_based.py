import pandas as pd
import numpy as np

from backlight.trades import Trades
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


def direction_based_trades(mkt, sig, target_column_name, direction_action_dict):
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
    return Trades(
        target_column_name=target_column_name,
        df=pd.concat([trades, mkt], axis=1, join="inner"),
        symbol=sig.symbol,
    )


def only_take_long(mkt, sig, target_column_name):
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.Donothing,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)


def only_take_short(mkt, sig, target_column_name):
    direction_action_dict = {
        TernaryDirection.UP: Action.Donothing,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)


def simple_buy_sell(mkt, sig, target_column_name):
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    return direction_based_trades(mkt, sig, target_column_name, direction_action_dict)
