import pandas as pd


class MarketData(pd.DataFrame):
    """MarketData container which inherits pd.DataFrame."""

    def __init__(self, df, symbol, start_dt=None, end_dt=None):
        """Initialize the MarketData instance.

        Args:
            df (DataFrame): dataframe
            symbol (str): symbol
            start_dt (pd.Timestamp): starting point. defaults to df.index[0]
            end_dt (pd.Timestamp): termination point. defaults to df.index[-1]
        """

        super(MarketData, self).__init__(df)

        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

    @property
    def symbol(self):
        """str: symbol"""
        return self._symbol

    @property
    def mid(self):
        """Series: mid price"""
        if "mid" in self.columns:
            return self["mid"]
        else:
            return (self.ask + self.bid) / 2
