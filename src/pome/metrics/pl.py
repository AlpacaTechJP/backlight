import pandas as pd
import numpy as np
from ..positions.positions import Positions


class PlMetrics:
    def __init__(self, positions: Positions, start_dt: str, end_dt: str):
        next_price = positions.price.shift(periods=-1)
        price_diff = next_price - positions.price
        pl = (price_diff * positions.amount).shift(periods=1)[1:]  # drop first nan
        self._pl = pl[(start_dt < pl.index) & (pl.index <= end_dt)]

        previous_amount = positions.amount.shift(periods=1)
        amount_diff = (positions.amount - previous_amount)[1:]  # drop first nan
        amount_diff = amount_diff[
            (start_dt < amount_diff.index) & (amount_diff.index <= end_dt)
        ]
        self._trade_count = len(amount_diff[amount_diff != 0.0])

    @property
    def pl(self):
        return self._pl

    @property
    def total_pl(self):
        if len(self._pl) == 0:
            return 0.0
        return self._pl.sum()

    @property
    def average_pl(self):
        if len(self._pl) == 0:
            return 0.0
        return self.total_pl / self._trade_count

    @property
    def total_win(self):
        mask = self._pl > 0.0
        masked = self._pl[mask]
        if len(masked) == 0:
            return 0.0
        return masked.sum()

    @property
    def total_loss(self):
        mask = self._pl < 0.0
        masked = self._pl[mask]
        if len(masked) == 0:
            return 0.0
        return masked.sum()

    @property
    def trade_count(self):
        return self._trade_count
