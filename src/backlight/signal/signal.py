import numpy as np
import pandas as pd

from backlight.labelizer.common import TernaryDirection


class Signal(pd.DataFrame):
    def __init__(self, df, symbol, start_dt=None, end_dt=None):
        """An abstraction interface for signals"""

        super(Signal, self).__init__(df)

        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

        for col in self.columns:
            if col not in self._target_columns:
                self.drop(col, inplace=True)

    @property
    def symbol(self):
        return self._symbol

    @property
    def start_dt(self):
        return self._start_dt

    @property
    def end_dt(self):
        return self._end_dt

    @property
    def pred(self):
        return NotImplementedError


class TernarySignal(Signal):

    _target_columns = ["up", "neutral", "down"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        argmax = np.argmax(self[["up", "neutral", "down"]].values, axis=1)
        self.loc[argmax == 0, "pred"] = TernaryDirection.UP.value
        self.loc[argmax == 1, "pred"] = TernaryDirection.NEUTRAL.value
        self.loc[argmax == 2, "pred"] = TernaryDirection.DOWN.value


class BinarySignal(Signal):

    _target_columns = ["up", "down"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        argmax = np.argmax(self[["up", "down"]].values, axis=1)
        self.loc[argmax == 0, "pred"] = TernaryDirection.UP.value
        self.loc[argmax == 1, "pred"] = TernaryDirection.DOWN.value
