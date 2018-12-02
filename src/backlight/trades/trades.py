import pandas as pd
from collections import namedtuple
from typing import List, Type, Tuple  # noqa

from backlight.datasource.marketdata import MarketData

Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade(pd.Series):

    _metadata = ["symbol"]

    def add(self, t: Transaction) -> "Trade":
        if t.timestamp in self.index:
            self.loc[t.timestamp] += t.amount
            return self

        self.loc[t.timestamp] = t.amount
        return self

    @property
    def amount(self) -> pd.Series:
        return self.rename("amount").sort_index()

    @property
    def _constructor(self) -> Type["Trade"]:
        return Trade


def _sum(a: list) -> int:
    return sum(a) if len(a) != 0 else 0


def _make_trade(sr: pd.Series, symbol: str) -> Trade:
    t = Trade(sr)
    t.symbol = symbol
    return t


def make_trade(symbol: str) -> Trade:
    t = Trade()
    t.symbol = symbol
    return t


def _evaluate_pl(trade: Trade, mkt: MarketData) -> float:
    amount = trade.amount.cumsum()
    mkt, amount = mkt.align(amount, axis=0, join="inner")
    # TODO: ask bid pricing
    next_price = mkt.mid.shift(periods=-1)
    price_diff = next_price - mkt.mid
    pl = (price_diff * amount).shift(periods=1)[1:]  # drop first nan
    return _sum(pl)


def count(trades: List[Trade], mkt: MarketData) -> Tuple[int, int, int]:
    pls = [_evaluate_pl(t, mkt) for t in trades if len(t.index) > 1]
    total = len(trades)
    win = _sum([pl > 0.0 for pl in pls])
    lose = _sum([pl < 0.0 for pl in pls])
    return total, win, lose


def flatten(trades: List[Trade]) -> Trade:
    symbol = trades[0].symbol
    assert all([t.symbol == symbol for t in trades])

    # compute amount
    amounts = [t.amount for t in trades]
    amount = pd.Series()
    for a in amounts:
        amount = amount.add(a, fill_value=0.0)

    return _make_trade(amount.sort_index(), symbol)
