import pandas as pd


class Trades(pd.DataFrame):
    def __init__(self, df, target_column_name, symbol="", start_dt=None, end_dt=None):
        super(Trades, self).__init__(df)
        self._target_column_name = target_column_name
        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

    @property
    def target_column_name(self):
        """Is this interface needed?"""
        return self._target_column_name

    @property
    def symbol(self):
        return self._symbol

    @property
    def start_dt(self):
        return self._start_dt

    @property
    def end_dt(self):
        return self._end_dt

    @property
    def amount(self):
        if "amount" in self.columns:
            return self["amount"]
        return NotImplementedError
