import pandas as pd
from typing import Type, Callable

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.trades import flatten
from backlight.trades.trades import Trade, Trades


class Positions(pd.DataFrame):
    """Dataframe for Positions.

    The ``price`` should be the price to evaluate the positions, not the cost to
    acquire. In some cases (e.g. ask/bid pricing), ``Trades``\ ' price and
    ``Positions``\ ' price are different.
    """

    _target_columns = ["amount", "price", "fee"]  # TODO: better name for fee

    def reset_cols(self) -> None:
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def value(self) -> pd.Series:
        return self.amount * self.price + self.fee

    @property
    def _constructor(self) -> Type["Positions"]:
        return Positions


def _mid_pricer(trade: Trade, mkt: MarketData) -> Positions:
    positions = pd.DataFrame(index=mkt.index)

    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "amount"] = positions["amount"].ffill()

    positions.loc[:, "price"] = mkt.mid

    positions.loc[:, "fee"] = (-mkt.mid * trade.amount).cumsum()
    positions.loc[:, "fee"] = positions["fee"].ffill()


    pos = Positions(positions)
    pos.reset_cols()
    pos.symbol = trade.symbol
    return pos


def _askbid_pricer(trade: Trade, mkt: MarketData) -> Positions:
    positions = pd.DataFrame(index=mkt.index)
    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "amount"] = positions["amount"].ffill()
    positions.loc[positions.amount > 0, "price"] = mkt.bid
    positions.loc[positions.amount < 0, "price"] = mkt.ask
    positions.loc[positions.amount == 0, "price"] = mkt.mid
    pos = Positions(positions)
    pos.reset_cols()
    pos.symbol = trade.symbol
    return pos


def _get_pricer(mkt: MarketData) -> Callable[[Trade, MarketData], Positions]:
    if isinstance(mkt, MidMarketData):
        return _mid_pricer
    if isinstance(mkt, AskBidMarketData):
        return _askbid_pricer
    raise NotImplementedError()


def calc_positions(
    trades: Trades,
    mkt: MarketData,
) -> Positions:
    trade = flatten(trades)
    pricer = _get_pricer(mkt)
    assert trade.symbol == mkt.symbol
    positions = pricer(trade, mkt)
    return positions


def calc_pl(positions: Positions) -> pd.Series:
    next_value = positions.value.shift(periods=-1)
    pl = (next_value - positions.value).shift(periods=1)[1:]  # drop first nan
    return pl.rename("pl")
