import pandas as pd
from typing import Type


class Positions(pd.DataFrame):
    """Dataframe for Positions.

    The ``price`` should be the price to evaluate the positions, not the cost to
    acquire. In some cases (e.g. ask/bid pricing), ``Trades``\ ' price and
    ``Positions``\ ' price are different.
    """

    _metadata = ["symbol"]

    @property
    def amount(self) -> pd.Series:
        if "amount" in self.columns:
            return self["amount"]
        raise NotImplementedError

    @property
    def price(self) -> pd.Series:
        if "price" in self.columns:
            return self["price"]
        raise NotImplementedError

    @property
    def _constructor(self) -> Type[Positions]:
        return Positions
