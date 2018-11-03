import pandas as pd


class Trades(pd.DataFrame):

    _metadata = ["symbol", "target_column_name"]

    @property
    def amount(self):
        if "amount" in self.columns:
            return self["amount"]
        return NotImplementedError

    @property
    def _constructor(self):
        return Trades
