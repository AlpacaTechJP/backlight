import pandas as pd


class Positions(pd.DataFrame):
    """Dataframe for Positions.

    The ``price`` should be the price to evaluate the positions, not the cost to
    acquire. In some cases (e.g. ask/bid pricing), ``Trades``\ ' price and
    ``Positions``\ ' price are different.
    """

    _metadata = ["symbol"]

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

    @property
    def _constructor(self):
        return Positions
