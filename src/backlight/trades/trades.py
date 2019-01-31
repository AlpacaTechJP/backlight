import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, Tuple, List, Iterable  # noqa

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

    def __repr__(self) -> str:
        return str(self.amount)

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


def _max(s: pd.Series) -> int:
    if len(s) == 0:
        return 0

    return max(s)


class Trades(pd.DataFrame):
    """A collection of trades.

    This is designed to achieve following purposes
    1. Compute metrics which need individual trade perfomance
       s.t. win_rate and lose_rate.
    2. Filter the trades
       s.t. `[t for t in trades if trades.index[0].hour in [0, 1, 2])]`.
    """

    _metadata = ["symbol"]

    _target_columns = ["amount", "_id"]

    @property
    def ids(self) -> List[int]:
        if "_id" not in self.columns:
            return []
        return self._id.unique().tolist()

    def add_trade(self, trade: Trade) -> Type["Trades"]:
        assert self.symbol == trade.symbol

        next_id = _max(self.ids) + 1
        df = trade.amount.to_frame(name="amount")
        df.loc[:, "_id"] = next_id

        return make_trades(self.symbol, pd.concat([self, df], axis=0))

    def get_trade(self, trade_id: int) -> pd.Series:
        return self.loc[self._id == trade_id, "amount"]

    def reset_cols(self) -> None:
        """Keep only _target_columns"""
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def _constructor(self) -> Type["Trades"]:
        return Trades


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


def from_tuple(trades: Iterable[Trade]) -> Trades:
    symbol = trades[0].symbol
    trs = make_trades(symbol)
    for t in trades:
        trs = trs.add_trade(t)
    return trs


def make_trade(symbol: str) -> Trade:
    """Initialize Trade instance"""
    t = Trade(symbol)
    return t


def make_trades(symbol: str, df: pd.DataFrame = None) -> Trades:
    """Initialize Trades instance"""
    if df is None:
        df = pd.DataFrame()

    t = Trades(df)
    t.symbol = symbol
    t.reset_cols()
    return t


def flatten(trades: Trades) -> Trade:
    """Flatten tuple of trade to a trade."""
    amounts = trades.amount
    amount = amounts.groupby(amounts.index).sum().sort_index()
    return from_series(amount, trades.symbol)
