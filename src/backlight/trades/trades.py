import pandas as pd
from collections import namedtuple
from typing import List, Type  # noqa


Transaction = namedtuple("Transaction", ["timestamp", "amount"])


class Trade(pd.Series):

    _metadata = ["_trades", "symbol"]

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
