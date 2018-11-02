import numpy as np
import pandas as pd


class metric_property(property):
    """Overrides property to:
        * capture warnings
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

    def __get__(self, obj, objtype=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            return super().__get__(obj)


def sync_index_and_get_list(lbl, sig):
    overlap_index = sig.index & lbl.index
    pred = sig.reindex(overlap_index)
    real = lbl.reindex(overlap_index)
    pred_arr = pred.pred.squeeze().values.tolist()
    real_arr = real.label.values.tolist()
    return real_arr, pred_arr
