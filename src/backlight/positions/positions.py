import pandas as pd


class Positions(pd.DataFrame):
    """Dataframe for Positions.

    The ``price`` should be the price to evaluate the positions, not the cost to
    acquire. In some cases (e.g. ask/bid pricing), ``Trades``\ ' price and
    ``Positions``\ ' price are different.
    """

    def __init__(self, df, symbol, start_dt=None, end_dt=None):
        """Wraps a DataFrame with some preperties."""
        super(Positions, self).__init__(df)
        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

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

    @property
    def price(self):
        if "price" in self.columns:
            return self["price"]
        return NotImplementedError
