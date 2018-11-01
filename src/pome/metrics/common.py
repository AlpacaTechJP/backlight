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
    def index_in_index(compared_index, standard_index):
        in_standards = []
        for i in compared_index:
            in_standard = "T" if i in standard_index else "F"
            in_standards.append(in_standard)
        return in_standards

    pred = sig.copy()
    real = lbl.copy()
    overlap_index = pd.concat([pred, real], join="inner", axis=1).index

    pred["in_overlap"] = index_in_index(pred.index, overlap_index)
    pred = pred[pred.in_overlap == "T"].drop("in_overlap", axis=1)

    real["in_overlap"] = index_in_index(real.index, overlap_index)
    real = real[real.in_overlap == "T"].drop("in_overlap", axis=1)

    pred_arr = pred.pred.squeeze().values.tolist()
    real_arr = real.label.values.tolist()
    return real_arr, pred_arr
