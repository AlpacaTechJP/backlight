import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, Tuple  # noqa

from backlight.datasource.marketdata import MarketData


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade:
    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._index = ()  # type: tuple
        self._amount = ()  # type: tuple

    def __eq__(self, other: Any) -> bool:
        return self.__class__ == other.__class__ and self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        return hash((self._index, self._amount, self.symbol))

    def add(self, t: Transaction) -> None:
        self._index += (t.timestamp,)
        self._amount += (t.amount,)

    @property
    def amount(self) -> pd.Series:
        amount = pd.Series(data=self._amount, index=self._index)
        return amount.groupby(amount.index).sum().sort_index()

    @property
    def index(self) -> pd.Index:
        return pd.Index(self._index).drop_duplicates().sort_values()

    @property
    def symbol(self) -> str:
        return self._symbol


Trades = Tuple[Trade, ...]


def _sum(a: list) -> int:
    return sum(a) if len(a) != 0 else 0


def from_series(sr: pd.Series, symbol: str) -> Trade:
    t = Trade(symbol)
    t._index = tuple([i for i in sr.index])
    t._amount = tuple(sr.values.tolist())
    return t


def make_trade(symbol: str) -> Trade:
    t = Trade(symbol)
    return t


def _evaluate_pl(trade: Trade, mkt: MarketData) -> float:
    amount = trade.amount.cumsum()
    mkt, amount = mkt.align(amount, axis=0, join="inner")
    # TODO: ask bid pricing
    next_price = mkt.mid.shift(periods=-1)
    price_diff = next_price - mkt.mid
    pl = (price_diff * amount).shift(periods=1)[1:]  # drop first nan
    return _sum(pl)


def count(trades: Trades, mkt: MarketData) -> Tuple[int, int, int]:
    pls = [_evaluate_pl(t, mkt) for t in trades if len(t.index) > 1]
    total = len(trades)
    win = _sum([pl > 0.0 for pl in pls])
    lose = _sum([pl < 0.0 for pl in pls])
    return total, win, lose


@lru_cache()
def flatten(trades: Trades) -> Trade:
    symbol = trades[0].symbol
    assert all([t.symbol == symbol for t in trades])

    # compute amount
    amounts = [t.amount for t in trades]
    amount = pd.Series()
    for a in amounts:
        amount = amount.add(a, fill_value=0.0)

    return from_series(amount.sort_index(), symbol)
