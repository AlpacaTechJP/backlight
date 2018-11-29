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

    _metadata = ["_trades", "symbol"]

    def reset(self) -> None:
        # check symbols
        symbol = self.trades[0].symbol
        assert all([t.symbol == symbol for t in self.trades])
        self.symbol = symbol

        # compute amount
        amounts = [t.amount for t in self.trades]
        amount = pd.Series()
        for a in amounts:
            amount = amount.add(a, fill_value=0.0)

        assert all(amount.index.isin(self.index))

        self.loc[:, "amount"] = amount

    @property
    def trades(self) -> List[Trade]:
        start = self.index[0]
        end = self.index[-1]
        return [
            t
            for t in self._trades
            if start <= t.amount.index[0] and t.amount.index[-1] <= end
        ]

    @property
    def _constructor(self) -> Type["Trades"]:
        return Trades
