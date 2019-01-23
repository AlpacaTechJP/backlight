import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, Tuple  # noqa

from backlight.datasource.marketdata import MarketData


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade:
    """Series object like instance for Trade. The purpose of the class is
        to improve computation speed.
    """

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
        """Add transaction"""
        self._index += (t.timestamp,)
        self._amount += (t.amount,)

    @property
    def amount(self) -> pd.Series:
        """Amount of transactions at that moment"""
        amount = pd.Series(data=self._amount, index=self._index)
        return amount.groupby(amount.index).sum().sort_index()

    @property
    def index(self) -> pd.Index:
        """Index of transactions"""
        return pd.Index(self._index).drop_duplicates().sort_values()

    @property
    def symbol(self) -> str:
        """Asset symbol"""
        return self._symbol


Trades = Tuple[Trade, ...]
"""A collection of trades.

This is designed to achieve following purposes
1. Compute metrics which need individual trade perfomance s.t. win_rate and lose_rate.
2. Filter the trades s.t. `[t for t in trades if trades.index[0].hour in [0, 1, 2])]`.
"""


def _sum(a: pd.Series) -> float:
    return a.sum() if len(a) != 0 else 0


def from_series(sr: pd.Series, symbol: str) -> Trade:
    """Create a Trade instance from pd.Series.

    Args:
        sr :  Series
        symbol :  A symbol

    Returns:
        Trade
    """
    t = Trade(symbol)
    t._index = tuple([i for i in sr.index])
    t._amount = tuple(sr.values.tolist())
    return t


def make_trade(symbol: str) -> Trade:
    """Initialize Trade instance"""
    t = Trade(symbol)
    return t


@lru_cache()
def flatten(trades: Trades) -> Trade:
    """Flatten tuple of trade to a trade."""
    symbol = trades[0].symbol
    assert all([t.symbol == symbol for t in trades])

    amounts = pd.concat([t.amount for t in trades], axis=0)
    amount = amounts.groupby(amounts.index).sum().sort_index()
    return from_series(amount, symbol)
