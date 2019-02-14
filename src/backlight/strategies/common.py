import pandas as pd
from enum import Enum

from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import Signal


class Action(Enum):
    TakeShort = -1
    Donothing = 0
    TakeLong = 1

    def act_on_amount(self) -> float:
        return float(self.value)
