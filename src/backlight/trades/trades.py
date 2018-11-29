import pandas as pd
from collections import namedtuple
from typing import List, Type  # noqa


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade(pd.Series):
    def add(self, t: Transaction) -> "Trade":
        if t.timestamp in self.index:
            self.loc[t.timestamp] += t.amount
            return self

        self.loc[t.timestamp] = t.amount
        return self

    @property
    def amount(self) -> pd.Series:
        return self.rename("amount").sort_index()

    @property
    def _constructor(self) -> Type["Trade"]:
        return Trade


class Trades(pd.DataFrame):

    _metadata = ["_trades", "symbol"]

    def reset(self) -> None:
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
