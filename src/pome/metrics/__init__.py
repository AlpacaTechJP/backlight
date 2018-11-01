from ..labelizer.common import LabelType
from .binary import calc_binary_metrics
from .binary_extend import calc_binary_trade_performance
from .pl import PlMetrics
from ..positions.positions import Positions


def calc_metrics(sig, lbl, dropna=True):
    if lbl.label_type == LabelType.BINARY:
        return calc_binary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError


def calc_trade_performance(sig, lbl, trades, dropna=True):
    """Wrong interface? sig, lbl is needed?"""
    if lbl.label_type == LabelType.BINARY:
        return calc_binary_trade_performance(sig.dropna(), lbl.dropna(), trades)
    else:
        raise NotImplementedError


def evaluate_position(positions: Positions, start_dt: str = None, end_dt: str = None):
    """Evaluate the pl perfomance of positions from start_dt to end_dt"""
    start_dt = positions.index[0] if start_dt is None else start_dt
    end_dt = positions.index[-1] if end_dt is None else end_dt
    return PlMetrics(positions, start_dt, end_dt)
