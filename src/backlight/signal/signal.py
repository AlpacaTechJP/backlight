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
        self.loc[:, "argmax"] = np.argmax(
            self[["up", "neutral", "down"]].values, axis=1
        )
        self["pred"] = (
            self["argmax"]
            .replace(0, TernaryDirection.UP.value)
            .replace(1, TernaryDirection.NEUTRAL.value)
            .replace(2, TernaryDirection.DOWN.value)
        )

    @property
    def pred(self):
        return self[["pred"]]


class TernaryOneColumnLabelSignal(Signal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["pred"] = (
            self["label"]
            .replace(0.0, TernaryDirection.UP.value)
            .replace(1.0, TernaryDirection.NEUTRAL.value)
            .replace(2.0, TernaryDirection.DOWN.value)
        )

    @property
    def pred(self):
        return self[["pred"]]


class BinaryOneColumnLabelSignal(Signal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["pred"] = (
            self["label"]
            .replace(1.0, TernaryDirection.UP.value)
            .replace(-1.0, TernaryDirection.DOWN.value)
        )

    @property
    def pred(self):
        return self[["pred"]]


class BinaryOneColumnUPProbaSignal(Signal):
    pass  # TODO


class BinaryTwoColumnsSignal(Signal):
    pass  # TODO
