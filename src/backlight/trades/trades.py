import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, Tuple, List, Iterable  # noqa

from backlight.datasource.marketdata import MarketData


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


Trade = pd.Series


def add_transaction(trade: Trade, t: Transaction) -> Trade:
    """Add transaction"""
    sr = pd.Series(index=[t.timestamp], data=[t.amount], name="amount")
    sr = pd.concat([trade, sr])
    return from_series(sr)


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

    def add_trade(self, trade: pd.Series) -> Type["Trades"]:
        next_id = _max(self.ids) + 1
        df = trade.to_frame(name="amount")
        df.loc[:, "_id"] = next_id

        return make_trades(self.symbol, pd.concat([self, df], axis=0))

    def get_trade(self, trade_id: int) -> Trade:
        trade = self.loc[self._id == trade_id, "amount"]
        return from_series(trade.groupby(trade.index).sum().sort_index())

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


def from_series(sr: pd.Series) -> Trade:
    """Create a Trade instance from pd.Series.

    Args:
        sr :  Series
        symbol :  A symbol

    Returns:
        Trade
    """
    sr = sr.groupby(sr.index).sum().sort_index()
    t = Trade(sr)
    return t


def from_tuple(trades: Iterable[Trade], symbol: str) -> Trades:
    trs = make_trades(symbol)
    for t in trades:
        trs = trs.add_trade(t)
    return trs


def make_trade(symbol: str, transactions: Iterable[Transaction] = None) -> Trade:
    """Initialize Trade instance"""
    if transactions is None:
        sr = pd.Series(name="amount")
        return from_series(sr)

    index = [t.timestamp for t in transactions]
    data = [t.amount for t in transactions]
    sr = pd.Series(index=index, data=data, name="amount")
    return from_series(sr)


def make_trades(symbol: str, df: pd.DataFrame = None) -> Trades:
    """Initialize Trades instance"""
    if df is None:
        df = pd.DataFrame()

    t = Trades(df)
    t.symbol = symbol
    t.reset_cols()
    return t


def concat(trades: Iterable[Trade]):
    return pd


def flatten(trades: Trades) -> Trade:
    """Flatten tuple of trade to a trade."""
    amounts = trades.amount
    amount = amounts.groupby(amounts.index).sum().sort_index()
    return from_series(amount)
