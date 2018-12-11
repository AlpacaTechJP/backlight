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


def _pricer(trade: Trade, mkt: MarketData) -> Positions:
    positions = pd.DataFrame(index=mkt.index)

    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "amount"] = positions["amount"].ffill()

    positions.loc[:, "price"] = mkt.mid

    fee = _calc_trade_fee(trade.amount, mkt)
    positions.loc[:, "fee"] = (fee * trade.amount).cumsum()
    positions.loc[:, "fee"] = positions["fee"].ffill()

    pos = Positions(positions)
    pos.reset_cols()
    pos.symbol = trade.symbol
    return pos


def _calc_trade_fee(trade_amount: pd.Series, mkt: MarketData) -> pd.Series:
    """
    This functionality should be included in Market.
    """
    if isinstance(mkt, MidMarketData):
        return -mkt.mid[trade_amount.index]

    if isinstance(mkt, AskBidMarketData):
        fee = pd.Series(data=0.0, index=trade_amount.index)
        fee.loc[trade_amount > 0.0] = -mkt.loc[trade_amount > 0.0, "ask"]
        fee.loc[trade_amount < 0.0] = -mkt.loc[trade_amount < 0.0, "bid"]
        return fee
    raise NotImplementedError()


def calc_positions(trades: Trades, mkt: MarketData) -> Positions:
    trade = flatten(trades)
    assert trade.symbol == mkt.symbol
    assert (trade.index.isin(mkt.index)).all()

    positions = _pricer(trade, mkt)
    return positions


def calc_pl(positions: Positions) -> pd.Series:
    next_value = positions.value.shift(periods=-1)
    pl = (next_value - positions.value).shift(periods=1)[1:]  # drop first nan
    return pl.rename("pl")
