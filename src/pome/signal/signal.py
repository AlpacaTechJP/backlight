import numpy as np
import pandas as pd
from ..labelizer.common import TernaryDirection


class Signal(pd.DataFrame):
    def __init__(self, df, symbol, start_dt=None, end_dt=None):
        """Wraps a DataFrame with some preperties."""

        super(Signal, self).__init__(df)

        self._symbol = symbol
        self._start_dt = df.index[0] if start_dt is None else start_dt
        self._end_dt = df.index[-1] if end_dt is None else end_dt

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred

    @property
    def pred(self):
        if "pred" not in self.columns:
            self.loc[:, "argmax"] = np.argmax(
                self[["up", "neutral", "down"]].values, axis=1
            )
            self.loc[self.argmax == 0, "pred"] = TernaryDirection.UP.value
            self.loc[self.argmax == 1, "pred"] = TernaryDirection.NEUTRAL.value
            self.loc[self.argmax == 2, "pred"] = TernaryDirection.DOWN.value
        return self[["pred"]]


class BinaryOneColumnLabelSignal(Signal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred

    @property
    def pred(self):
        if "pred" not in self.columns:
            self.loc[self.label == 1.0, "pred"] = TernaryDirection.UP.value
            self.loc[self.label == -1.0, "pred"] = TernaryDirection.DOWN.value
        return self[["pred"]]


class BinaryOneColumnUPProbaSignal(Signal):
    pass  # TODO


class BinaryTwoColumnsSignal(Signal):
    pass  # TODO
