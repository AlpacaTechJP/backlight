import numpy as np
import pandas as pd
from typing import Type

from backlight.labelizer.common import TernaryDirection


def _argmax(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Returns the indices of the maximum values along an axis.
    If multiple maximum values exist, return -1.
    """
    rows = np.where(a == a.max(axis=1)[:, None])[0]
    rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
    argmax = a.argmax(axis)
    argmax[rows_multiple_max] = -1
    return argmax


class Signal(pd.DataFrame):

    _metadata = ["symbol", "_target_columns", "currency_unit"]

    def reset_cols(self) -> None:
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def _constructor(self) -> Type["Signal"]:
        return Signal

    @property
    def start_dt(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end_dt(self) -> pd.Timestamp:
        return self.index[-1]


class TernarySignal(Signal):

    _target_columns = ["up", "neutral", "down"]

    def reset_pred(self, threshold: float = 0.0) -> None:
        argmax = _argmax(self[["up", "neutral", "down"]].values, axis=1)

        up_index = (argmax == 0) & (self.up >= threshold)
        down_index = (argmax == 2) & (self.down >= threshold)

        self.loc[up_index, "pred"] = TernaryDirection.UP.value
        self.loc[down_index, "pred"] = TernaryDirection.DOWN.value
        self.loc[~up_index & ~down_index, "pred"] = TernaryDirection.NEUTRAL.value

    @property
    def _constructor(self) -> Type["TernarySignal"]:
        return TernarySignal


class BinarySignal(Signal):

    _target_columns = ["up", "down"]

    def reset_pred(self) -> None:
        argmax = np.argmax(self[["up", "down"]].values, axis=1)
        self.loc[argmax == 0, "pred"] = TernaryDirection.UP.value
        self.loc[argmax == 1, "pred"] = TernaryDirection.DOWN.value

    @property
    def _constructor(self) -> Type["BinarySignal"]:
        return BinarySignal
