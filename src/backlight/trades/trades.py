import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, Tuple, List, Iterable  # noqa

from backlight.datasource.marketdata import MarketData


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


Trade = pd.Series


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

    @property
    def amount(self) -> Trade:
        a = self["amount"]
        return a.groupby(a.index).sum().sort_index()

    def add_trade(self, trade: Trade) -> Type["Trades"]:
        next_id = _max(self.ids) + 1
        return self.append_trade(trade, next_id)

    def get_trade(self, trade_id: int) -> Trade:
        trade = self.loc[self._id == trade_id, "amount"]
        return trade.groupby(trade.index).sum().sort_index()

    def append_trade(self, trade: Trade, trade_id: int) -> Type["Trades"]:
        df = trade.to_frame(name="amount")
        df.loc[:, "_id"] = trade_id

        return _sort(concat([self, df]))

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


def _sort(t: Trades) -> Trades:
    t["ind"] = t.index
    t = t.sort_values(by=["ind", "_id"])
    t.reset_cols()
    return t


def make_trade(transactions: Iterable[Transaction]) -> Trade:
    """Initialize Trade instance"""
    index = [t.timestamp for t in transactions]
    data = [t.amount for t in transactions]
    sr = pd.Series(index=index, data=data, name="amount")
    return sr.groupby(sr.index).sum().sort_index()


def concat(trades: Iterable[Trades]):
    t = Trades(pd.concat(trades, axis=0))
    t.symbol = trades[0].symbol
    return _sort(t)


def make_trades(
    symbol: str, trades: Iterable[Trade], ids: Iterable[int] = None
) -> Trades:
    if ids is None:
        ids = range(len(trades))

    assert len(ids) == len(trades)

    trs = Trades()
    trs.symbol = symbol
    for i, t in zip(ids, trades):
        trs = trs.append_trade(t, i)

    return _sort(trs)
