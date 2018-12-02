import pandas as pd
from typing import Type, Callable, List

from backlight.datasource.marketdata import MarketData
from backlight.trades import flatten
from backlight.trades.trades import Trade


class Positions(pd.DataFrame):
    """Dataframe for Positions.

    The ``price`` should be the price to evaluate the positions, not the cost to
    acquire. In some cases (e.g. ask/bid pricing), ``Trades``\ ' price and
    ``Positions``\ ' price are different.
    """

    _metadata = ["symbol"]

    @property
    def amount(self) -> pd.Series:
        if "amount" in self.columns:
            return self["amount"]
        raise NotImplementedError

    @property
    def price(self) -> pd.Series:
        if "price" in self.columns:
            return self["price"]
        raise NotImplementedError

    @property
    def _constructor(self) -> Type["Positions"]:
        return Positions


def _mid_trader(trade: Trade, mkt: MarketData) -> Positions:
    positions = pd.DataFrame(index=mkt.index)
    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid
    pos = Positions(positions)
    pos.symbol = trade.symbol
    return pos


def calc_positions(
    trades: List[Trade],
    mkt: MarketData,
    trader: Callable[[Trade, MarketData], Positions] = _mid_trader,
) -> Positions:
    trade = flatten(trades)
    assert trade.symbol == mkt.symbol
    positions = trader(trade, mkt)
    return positions
