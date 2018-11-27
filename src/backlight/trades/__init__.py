import pandas as pd
from typing import Type


class Trades(pd.DataFrame):

    _metadata = ["symbol", "target_column_name"]

    @property
    def amount(self) -> pd.Series:
        if "amount" in self.columns:
            return self["amount"]
        raise NotImplementedError

    @property
    def _constructor(self) -> Type["Trades"]:
        return Trades
