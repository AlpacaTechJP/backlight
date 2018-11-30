import pandas as pd

from backlight.labelizer.common import LabelType, Label
from backlight.metrics.ternary import calc_ternary_metrics
from backlight.signal.signal import Signal


def calc_metrics(sig: Signal, lbl: Label, dropna: bool = True) -> pd.DataFrame:
    if lbl.label_type == LabelType.TERNARY:
        return calc_ternary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError
