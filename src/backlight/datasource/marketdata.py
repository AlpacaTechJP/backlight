import pandas as pd
from typing import Type


class MarketData(pd.DataFrame):
    """MarketData container which inherits pd.DataFrame."""

    _metadata = ["symbol"]

    @property
    def mid(self) -> pd.Series:
        """Series: mid price"""
        if "mid" in self.columns:
            return self["mid"]
        else:
            return (self.ask + self.bid) / 2

    @property
    def _constructor(self) -> Type["MarketData"]:
        return MarketData
