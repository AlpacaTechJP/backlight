import pandas as pd
import numpy as np
from typing import Type, Callable

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.trades.trades import Trades


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


def _pricer(trades: Trades, mkt: MarketData, principal: float) -> pd.DataFrame:
    trade = trades.amount

    # historical data
    idx = mkt.index[trade.index[0] <= mkt.index]  # only after first trades
    fee = mkt.fee(trade)
    positions = pd.DataFrame(
        index=idx,
        data={
            "amount": trade.cumsum(),
            "price": mkt.mid.loc[idx],
            "principal": -fee.cumsum() + principal,
        },
        columns=["amount", "price", "principal"],
    )
    positions = positions.ffill()

    # add initial data
    initial_idx = idx[0] - _freq(idx)
    positions.at[initial_idx, :] = [0.0, 0.0, principal]

    return positions.sort_index()


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
    assert trades.symbol == mkt.symbol
    assert trades.index.isin(mkt.index).all()

    pos = Positions(_pricer(trades, mkt, principal))
    pos.reset_cols()
    pos.symbol = trades.symbol
    return pos
