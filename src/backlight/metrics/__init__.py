import pandas as pd

from typing import Optional

from backlight.labelizer.common import LabelType, Label
from backlight.metrics.ternary import calc_ternary_metrics
from backlight.metrics import pl
from backlight.signal.signal import Signal
from backlight.positions import Positions


def calc_metrics(sig: Signal, lbl: Label, dropna: bool = True) -> pd.DataFrame:
    if lbl.label_type == LabelType.TERNARY:
        return calc_ternary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError


def calc_position_performance(
    positions: Positions,
    start_dt: Optional[pd.Timestamp] = None,
    end_dt: Optional[pd.Timestamp] = None,
):
    """Evaluate the pl perfomance of positions from start_dt to end_dt"""
    start_dt = positions.index[0] if start_dt is None else start_dt
    end_dt = positions.index[-1] if end_dt is None else end_dt
    return pl.calc_position_performance(positions, start_dt, end_dt)
