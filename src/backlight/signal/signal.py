import numpy as np
import pandas as pd

from backlight.labelizer.common import TernaryDirection


class Signal(pd.DataFrame):

    _metadata = ['symbol', '_target_columns']

    def reset_cols(self):
        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, axis=1, inplace=True)

    @property
    def _constructor(self):
        return Signal

    @property
    def start_dt(self):
        return self.index[0]

    @property
    def end_dt(self):
        return self.index[-1]


class TernarySignal(Signal):

    _target_columns = ["up", "neutral", "down"]

    def reset_pred(self):
        argmax = np.argmax(self[["up", "neutral", "down"]].values, axis=1)
        self.loc[argmax == 0, "pred"] = TernaryDirection.UP.value
        self.loc[argmax == 1, "pred"] = TernaryDirection.NEUTRAL.value
        self.loc[argmax == 2, "pred"] = TernaryDirection.DOWN.value

    @property
    def _constructor(self):
        return TernarySignal


class BinarySignal(Signal):

    _target_columns = ["up", "down"]

    def reset_pred(self):
        argmax = np.argmax(self[["up", "down"]].values, axis=1)
        self.loc[argmax == 0, "pred"] = TernaryDirection.UP.value
        self.loc[argmax == 1, "pred"] = TernaryDirection.DOWN.value

    @property
    def _constructor(self):
        return BinarySignal
