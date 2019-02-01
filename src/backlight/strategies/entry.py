import pandas as pd

from typing import List

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import make_trade
from backlight.trades.trades import Trades, from_dataframe
from backlight.strategies.common import Action


def direction_based_entry(
    mkt: MarketData, sig: Signal, direction_action_dict: dict
) -> Trades:
    """Take positions.

    Args:
        mkt: Market data
        sig: Signal data
        direction_action_dict: Dictionary from signals to actions
    Result:
        Trades
    """
    assert all([idx in mkt.index for idx in sig.index])
    df = sig

    trades = []  # type: List[pd.Dataframe]
    for direction, action in direction_action_dict.items():

        amount = action.act_on_amount()
        if amount == 0.0:
            continue

        trades.append(
            pd.DataFrame(
                index=df[df["pred"] == direction.value].index,
                data=direction.value,
                columns=["amount"],
            )
        )

    df_trades = pd.concat(trades, axis=0).sort_index()
    df_trades.loc[:, "_id"] = range(len(df_trades.index))

    t = from_dataframe(df_trades, df.symbol)
    return t
