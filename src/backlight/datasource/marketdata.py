import pandas as pd
from typing import Type


class MarketData(pd.DataFrame):
    """MarketData container which inherits pd.DataFrame."""

    _metadata = ["symbol", "_target_columns"]

    def reset_cols(self) -> None:
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def _constructor(self) -> Type["MarketData"]:
        return MarketData

    @property
    def start_dt(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end_dt(self) -> pd.Timestamp:
        return self.index[-1]


class MidMarketData(MarketData):

    _target_columns = ["mid"]

    @property
    def mid(self) -> pd.Series:
        """Series: mid price"""
        return self["mid"]

    @property
    def _constructor(self) -> Type["MidMarketData"]:
        return MidMarketData


class AskBidMarketData(MarketData):

    _target_columns = ["ask", "bid"]

    @property
    def mid(self) -> pd.Series:
        """Series: mid price"""
        return (self.ask + self.bid) / 2.0

    @property
    def _constructor(self) -> Type["AskBidMarketData"]:
        return AskBidMarketData
