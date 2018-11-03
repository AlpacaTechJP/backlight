from backlight.labelizer.common import LabelType
from backlight.metrics.ternary import calc_ternary_metrics


def calc_metrics(sig, lbl, dropna=True):
    if lbl.label_type == LabelType.TERNARY:
        return calc_ternary_metrics(sig.dropna(), lbl.dropna())
    else:
        raise NotImplementedError
