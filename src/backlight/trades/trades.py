import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, List, Iterable, Optional  # noqa

from backlight.datasource.marketdata import MarketData


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


def _max(s: pd.Series) -> int:
    if len(s) == 0:
        return 0

    return max(s)


class Trades(pd.DataFrame):
    """A collection of trades.

    This is designed to achieve following purposes
    1. Compute metrics which need individual trade perfomance
       s.t. win_rate and lose_rate.
    2. Filter the trades.
    """

    _metadata = ["symbol"]

    _target_columns = ["amount", "_id"]

    @property
    def ids(self) -> List[int]:
        """Return all unique ids"""
        if "_id" not in self.columns:
            return []
        return self._id.unique().tolist()

    @property
    def amount(self) -> pd.Series:
        """Flattend as one Trade"""
        a = self["amount"]
        return a.groupby(a.index).sum().sort_index()

    def get_trade(self, trade_id: int) -> pd.Series:
        """Get trade.

        Args:
            trade_id: Id for the trade.
                      Trades of the same id are recognized as one individual trade.
        Returns:
            Trade of pd.Series.
        """
        return self.loc[self._id == trade_id, "amount"]

    def get_any(self, key: Any) -> Type["Trades"]:
        """Filter trade which match conditions at least one element.

        Args:
            key: Same arguments with pd.DataFrame.__getitem__.

        Returns:
            Trades.
        """
        filterd_ids = self[key].ids
        trades = [self.get_trade(i) for i in filterd_ids]
        return make_trades(self.symbol, trades, filterd_ids)

    def get_all(self, key: Any) -> Type["Trades"]:
        """Filter trade which match conditions for all elements.

        Args:
            key: Same arguments with pd.DataFrame.__getitem__.

        Returns:
            Trades.
        """
        filterd = self[key]
        ids = []
        trades = []
        for i in filterd.ids:
            t = self.get_trade(i)
            if t.equals(filterd.get_trade(i)):
                ids.append(i)
                trades.append(t)
        return make_trades(self.symbol, trades, ids)

    def add_trade(self, trade: pd.Series, trade_id: int) -> Type["Trades"]:
        """Register new trade.

        Args:
            trade: Trade.
            trade_id: Id for the trade.
                      Trades of the same id are recognized as one individual trade.
        Returns:
            Trades.
        """
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


def make_trade(transactions: Iterable[Transaction]) -> pd.Series:
    """Create Trade instance from transacsions"""
    index = [t.timestamp for t in transactions]
    data = [t.amount for t in transactions]
    sr = pd.Series(index=index, data=data, name="amount")
    return sr.groupby(sr.index).sum().sort_index()


def from_dataframe(df: pd.DataFrame, symbol: str) -> Trades:
    """Create a Trades instance out of a DataFrame object

    Args:
        df:  DataFrame
        symbol: symbol to query

    Returns:
        Trades
    """

    trades = Trades(df.copy())
    trades.symbol = symbol
    trades.reset_cols()

    return _sort(trades)


def concat(trades: List[Trades]) -> Trades:
    """Concatenate some fo Trades"""
    t = Trades(pd.concat(trades, axis=0))
    t.symbol = trades[0].symbol
    return _sort(t)


def make_trades(
    symbol: str, trades: List[pd.Series], ids: Optional[List[int]] = None
) -> Trades:
    """Create Trades from some of trades"""
    if ids is None:
        _ids = list(range(len(trades)))
    else:
        _ids = ids

    assert len(_ids) == len(trades)

    trs = Trades()
    trs.symbol = symbol
    for i, t in zip(_ids, trades):
        trs = trs.add_trade(t, i)  # type: ignore

    return _sort(trs)
