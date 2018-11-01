import pandas as pd


class MarketData(pd.DataFrame):
    def __init__(self, df, symbol, start_dt=None, end_dt=None):

        super(MarketData, self).__init__(df)

        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

    @property
    def symbol(self):
        return self._symbol

    @property
    def mid(self):
        if "mid" in self.columns:
            return self["mid"]
        else:
            return (self.ask + self.bid) / 2
