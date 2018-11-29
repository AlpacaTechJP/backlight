import pandas as pd
from collections import namedtuple
from typing import List, Type  # noqa


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade:
    def __init__(self, transactions: List[Transaction], symbol: str) -> None:
        self._symbol = symbol
        index = [t.timestamp for t in transactions]
        amount = [t.amount for t in transactions]
        self._amount = (
            pd.Series(index=index, data=amount, name="amount")
            .groupby(index)
            .sum()
            .sort_index()
        )

    @property
    def amount(self) -> pd.Series:
        return self._amount

    @property
    def symbol(self) -> str:
        return self._symbol


class Trades(pd.DataFrame):

    _metadata = ["trades"]

    @property
    def amount(self) -> pd.Series:
        amounts = [t.amount for t in self.trades]
        amount = pd.Series()
        for a in amounts:
            amount = amount.add(a, fill_value=0.0)
        return amount

    @property
    def symbol(self) -> str:
        symbol = self.trades.symbol
        assert all([t.symbol == symbol for t in self.trades])
        return symbol

    @property
    def _constructor(self) -> Type["Trades"]:
        return Trades
