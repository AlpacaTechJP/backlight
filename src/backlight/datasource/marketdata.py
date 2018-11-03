import pandas as pd


class MarketData(pd.DataFrame):
    """MarketData container which inherits pd.DataFrame."""

    _metadata = ['symbol', ]

    @property
    def mid(self):
        """Series: mid price"""
        if "mid" in self.columns:
            return self["mid"]
        else:
            return (self.ask + self.bid) / 2

    @property
    def _constructor(self):
        return MarketData
