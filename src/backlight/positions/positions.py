import pandas as pd
from typing import Type, Callable

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.trades import flatten
from backlight.trades.trades import Trade, Trades


class Positions(pd.DataFrame):
    """Dataframe for Positions.
    """

    _target_columns = ["amount", "price", "principal"]

    def reset_cols(self) -> None:
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def value(self) -> pd.Series:
        return self.amount * self.price + self.principal

    @property
    def _constructor(self) -> Type["Positions"]:
        return Positions


def _pricer(trade: Trade, mkt: MarketData, principal: float) -> Positions:
    positions = pd.DataFrame(index=mkt.index)

    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "amount"] = positions["amount"].ffill()

    positions.loc[:, "price"] = mkt.mid

    fee = mkt.fee(trade.amount)
    positions.loc[:, "principal"] = -fee.cumsum() + principal
    positions.loc[:, "principal"] = positions["principal"].ffill()

    pos = Positions(positions)
    pos.reset_cols()
    pos.symbol = trade.symbol
    return pos


def calc_positions(
    trades: Trades, mkt: MarketData, principal: float = 0.0
) -> Positions:
    trade = flatten(trades)

    assert trade.symbol == mkt.symbol
    assert (trade.index.isin(mkt.index)).all()

    positions = _pricer(trade, mkt, principal)
    return positions


def calc_pl(positions: Positions) -> pd.Series:
    next_value = positions.value.shift(periods=-1)
    pl = (next_value - positions.value).shift(periods=1)[1:]  # drop first nan
    return pl.rename("pl")
