import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal
from backlight.trades import make_trade
from backlight.trades.trades import Transaction, Trade, Trades, make_trades, make_trade
from backlight.strategies.common import Action


def _entry(amount: float, idx: pd.Timestamp, symbol: str) -> Trade:
    t = Transaction(timestamp=idx, amount=amount)
    trade = make_trade([t])
    return trade


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

    trades = ()  # type: Trades
    for direction, action in direction_action_dict.items():
        amount = action.act_on_amount()
        if amount == 0.0:
            continue
        target_index = df[df["pred"] == direction.value].index
        trades += tuple(_entry(amount, idx, df.symbol) for idx in target_index)

    return make_trades(df.symbol, trades)
