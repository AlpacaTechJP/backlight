from ..labelizer.common import LabelType
from ..positions.positions import Positions
from .binary import calc_binary_metrics
from .ternary import calc_ternary_metrics
from .binary_extend import calc_binary_trade_performance
from .pl import PlMetrics


def calc_metrics(sig, lbl, dropna=True):
    """Dispatch the task to the correct function according to label type"""
    if lbl.label_type == LabelType.BINARY:
        return calc_binary_metrics(sig.dropna(), lbl.dropna())
    if lbl.label_type == LabelType.TERNARY:
        return calc_ternary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError


def calc_trade_performance(sig, lbl, trades, dropna=True):
    """Dispatch the task to the correct function according to label type
    The result of this function will includes everything in calc_metrics
    plus addition information like PL(profit/loss)
    """
    if lbl.label_type == LabelType.BINARY:
        return calc_binary_trade_performance(sig.dropna(), lbl.dropna(), trades)
    else:
        raise NotImplementedError


def evaluate_position(positions: Positions, start_dt: str = None, end_dt: str = None):
    """Evaluate the pl perfomance of positions from start_dt to end_dt"""
    start_dt = positions.index[0] if start_dt is None else start_dt
    end_dt = positions.index[-1] if end_dt is None else end_dt
    return PlMetrics(positions, start_dt, end_dt)
