import pandas as pd
from collections import namedtuple
from typing import Any, List, Type, Tuple  # noqa

from backlight.datasource.marketdata import MarketData

Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._index = []  # type: list
        self._amount = []  # type: list

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Trade):
            return False

        return self._index == other._index and self._amount == other._amount

    def add(self, t: Transaction) -> None:
        self._index.append(t.timestamp)
        self._amount.append(t.amount)

    @property
    def amount(self) -> pd.Series:
        amount = pd.Series(data=self._amount, index=self._index)
        return amount.groupby(amount.index).sum().sort_index()

    @property
    def size(self) -> int:
        return len(self._index)


def _sum(a: list) -> int:
    return sum(a) if len(a) != 0 else 0


def _make_trade(sr: pd.Series, symbol: str) -> Trade:
    t = Trade(symbol)
    t._index = [i for i in sr.index]
    t._amount = sr.values.tolist()
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


def count(trades: List[Trade], mkt: MarketData) -> Tuple[int, int, int]:
    pls = [_evaluate_pl(t, mkt) for t in trades if t.size > 1]
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
