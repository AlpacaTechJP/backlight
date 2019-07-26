import pandas as pd
from typing import Type
from backlight.asset.currency import Currency


class MarketData(pd.DataFrame):
    """MarketData container which inherits pd.DataFrame."""

    _metadata = ["symbol", "_target_columns", "currency_unit"]

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
    """MarketData container for mid price.
    """

    _target_columns = ["mid"]

    @property
    def mid(self) -> pd.Series:
        """Mid price"""
        return self["mid"]

    def fee(self, trade_amount: pd.Series) -> pd.Series:
        """Calculate trading fee when trading on mid prices."""
        return self.mid[trade_amount.index] * trade_amount

    @property
    def _constructor(self) -> Type["MidMarketData"]:
        return MidMarketData


class AskBidMarketData(MarketData):

    _target_columns = ["ask", "bid"]

    @property
    def mid(self) -> pd.Series:
        """Mid price"""
        return (self.ask + self.bid) / 2.0

    @property
    def spread(self) -> pd.Series:
        """Spread"""
        return (self.ask - self.bid).abs()

    def fee(self, trade_amount: pd.Series) -> pd.Series:
        """Calculate trading fee when trading on ask/bid prices."""
        fee = pd.Series(data=0.0, index=trade_amount.index)

        # TODO: avoid long codes
        fee.loc[trade_amount > 0.0] = self.loc[
            pd.Series(data=False, index=self.index) | (trade_amount > 0.0), "ask"
        ]
        fee.loc[trade_amount < 0.0] = self.loc[
            pd.Series(data=False, index=self.index) | (trade_amount < 0.0), "bid"
        ]
        return fee * trade_amount

    @property
    def _constructor(self) -> Type["AskBidMarketData"]:
        return AskBidMarketData
