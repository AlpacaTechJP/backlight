import pandas as pd
import numpy as np
from typing import Type, Callable

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.trades import flatten
from backlight.trades.trades import Trade, Trades


def _freq(idx: pd.Index) -> pd.Timedelta:
    if idx.freq is not None:
        return idx.freq
    if len(idx) > 1:
        return idx[1] - idx[0]
    return pd.Timedelta("1s")


class Positions(pd.DataFrame):
    """Positions container which inherits pd.DataFrame.

    They have following columns:
        - `amount`: Amount of the asset you are holding at that moment.
        - `price`: Price per unit of the asset at that moment.
        - `principal`: Principal in your bank account at that moment.
    """

    _metadata = ["symbol"]

    _target_columns = ["amount", "price", "principal"]

    def reset_cols(self) -> None:
        """ Keep only _target_columns"""
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def value(self) -> pd.Series:
        """ Series of the position valuation"""
        return self.amount * self.price + self.principal

    @property
    def _constructor(self) -> Type["Positions"]:
        return Positions


def _pricer(trade: Trade, mkt: MarketData, principal: float) -> Positions:

    # historical data
    idx = mkt.index[trade.index[0] <= mkt.index]  # only after first trades
    positions = pd.DataFrame(index=idx)
    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid.loc[idx]
    fee = mkt.fee(trade.amount)
    positions.loc[:, "principal"] = -fee.cumsum() + principal
    positions = positions.ffill()

    # add initial data
    initial_idx = idx[0] - _freq(idx)
    positions.loc[initial_idx, "amount"] = 0.0
    positions.loc[initial_idx, "price"] = 0.0
    positions.loc[initial_idx, "principal"] = principal

    pos = Positions(positions.sort_index())
    pos.reset_cols()
    pos.symbol = trade.symbol
    return pos


def calc_positions(
    trades: Trades, mkt: MarketData, principal: float = 0.0
) -> Positions:
    """Create Positions from Trades and MarketData.
    Positions' frequency is determined by MarketData's frequency.

    Args:
        trades: Tuple of trades.
        mkt: Market data.
        principal: The initial principal value.
    """
    trade = flatten(trades)

    assert trade.symbol == mkt.symbol
    assert (trade.index.isin(mkt.index)).all()

    positions = _pricer(trade, mkt, principal)
    positions.symbol = trade.symbol
    return positions
