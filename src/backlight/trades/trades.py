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

    def __repr__(self) -> str:
        return str(self.amount)

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


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0


def from_series(sr: pd.Series, symbol: str) -> Trade:
    t = Trade(symbol)
    t._index = tuple([i for i in sr.index])
    t._amount = tuple(sr.values.tolist())
    return t


def make_trade(symbol: str) -> Trade:
    t = Trade(symbol)
    return t


@lru_cache()
def flatten(trades: Trades) -> Trade:
    symbol = trades[0].symbol
    assert all([t.symbol == symbol for t in trades])

    amounts = pd.concat([t.amount for t in trades], axis=1, join="outer")
    amount = amounts.sum(axis=1, skipna=True).sort_index()
    return from_series(amount, symbol)
