import pandas as pd
from collections import namedtuple
from functools import lru_cache
from typing import Any, Type, List, Iterable, Optional  # noqa
from backlight.asset.currency import Currency

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

    _metadata = ["symbol", "currency_unit"]

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

    def get_any(self, interval: tuple, time: str) -> Type["Trades"]:
        """Filter trade which match conditions at least one element.

        Args:
            interval: The results will only contain elements of time in this interval
            time: Can be 'hour', 'minute'... The results.time will be in the interval

        Returns:
            Trades.
        """

        # The function is getting complicated to read, but its way faster.
        # Also, I had to change the arguments, so it is less flexible. I can work on
        # it more to think about a just middle.

        key = getattr(self.index, time).isin(interval)
        filterd_ids = self[key].ids

        sort = self.sort_values("_id")

        trades = []
        for i in range(0, sort.index.size, 2):
            entry_index = sort.index[i]
            exit_index = sort.index[i + 1]
            if (
                getattr(entry_index, time) in interval
                or getattr(exit_index, time) in interval
            ):
                trade = pd.Series(
                    data=[sort.iat[i, 0], sort.iat[i + 1, 0]],
                    index=[entry_index, exit_index],
                )
                trades.append(trade)

        return make_trades(self.symbol, trades, self.currency_unit, filterd_ids)

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
        return make_trades(self.symbol, trades, self.currency_unit, ids)

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


def from_dataframe(df: pd.DataFrame, symbol: str, currency_unit: Currency) -> Trades:
    """Create a Trades instance out of a DataFrame object

    Args:
        df:  DataFrame
        symbol: symbol to query

    Returns:
        Trades
    """

    trades = Trades(df.copy())
    trades.symbol = symbol
    trades.currency_unit = currency_unit
    trades.reset_cols()

    return _sort(trades)


def concat(trades: List[Trades], refresh_id: bool = False) -> Trades:
    """Concatenate some of Trades

    Args:
        trades: List of trades
        refresh_id: Set true to re-assign ids for trades. Default: False

    Returns:
        Trades
    """
    if refresh_id:
        id_offset = 0
        list_of_trades = []
        for a_trades in trades:
            a_trades = a_trades.copy()
            a_trades._id += id_offset
            id_offset = a_trades._id.max() + 1
            list_of_trades.append(a_trades)
        trades = list_of_trades

    t = Trades(pd.concat(trades, axis=0))
    t.symbol = trades[0].symbol
    t.currency_unit = trades[0].currency_unit
    return _sort(t)


def make_trades(
    symbol: str,
    trades: List[pd.Series],
    currency_unit: Currency,
    ids: Optional[List[int]] = None,
) -> Trades:
    """Create Trades from some of trades"""
    if ids is None:
        _ids = list(range(len(trades)))
    else:
        _ids = ids

    assert len(_ids) == len(trades)

    df = pd.concat(trades, axis=0).to_frame(name="amount")
    df.loc[:, "_id"] = 0
    current = 0
    for i, t in zip(_ids, trades):
        l = len(t.index)
        df.iloc[current : current + l, 1] = i
        current += l

    return from_dataframe(df, symbol, currency_unit)
