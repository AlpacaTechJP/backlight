import pandas as pd
from enum import Enum

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal


class Action(Enum):
    TakeShort = -1
    Donothing = 0
    TakeLong = 1

    def act_on_amount(self) -> "Action":
        return self.value


def concat(mkt: MarketData, sig: Signal) -> pd.DataFrame:
    assert mkt.symbol == sig.symbol
    # Assume sig is less frequent than mkt.
    assert all([idx in mkt.index for idx in sig.index])
    df = pd.concat([mkt, sig], axis=1, join="inner")
    df.symbol = mkt.symbol
    return df
