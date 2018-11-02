import numpy as np
import pandas as pd

from ..labelizer.common import TernaryDirection
from .common import metric_property, sync_index_and_get_list


def calc_ternary_metrics(sig, lbl):
    real_arr, pred_arr = sync_index_and_get_list(lbl, sig)
    t_metrics = TernaryMetrics(real_arr, pred_arr)
    return t_metrics


class TernaryMetrics:
    """Binary metrics.
    Confusion matrix:
                      |             Actual
        | Predicted   | Up     | Neutral     | Down     |
        | ----------- | ------ | ----------- | -------- |
        | Up          | a      | b           | c        |
        | Neutral     | d      | e           | f        |
        | Down        | g      | h           | i        |

                      |      Actual
        | Predicted   | Up             |      Down      |
        | ----------- | -------------- | -------------- |
        | Up          | True Positive  | False Positive |
        | Down        | False Negative | True Negative  |

                      |      Actual
        | Predicted   | Up             |      Down      |
        | ----------- | -------------- | -------------- |
        | Up          | UU             | UD             |
        | Down        | DU             | DD             |

    Metrics definitions:
        Scalar metrics:
        | Name         | Definition                                    |
        | ------------ | --------------------------------------------  |
        | ALL          | a + b + c + d + e + f + g + h + i             |
        | ------------ | --------------------------------------------  |
        | accuracy     | ( a + e + i ) / sum(ALL)                   |
        |              |                                               |
        | pred_ratio_u | ( a + b + c ) / sum(ALL)                      |
        | pred_ratio_n | ( d + e + f ) / sum(ALL)                      |
        | pred_ratio_d | ( g + h + i ) / sum(ALL)                      |
        | pred_ratio_ud| ( a + b + c + g + h + i ) / sum(ALL)          |
        |              |                                               |
        | pred_count_u | a + b + c                                     |
        | pred_count_n | d + e + f                                     |
        | pred_count_d | g + h + i                                     |
        | pred_count_ud| ( a + b + c ) + ( g + h + i )                 |
        |              |                                               |
        | real_ratio_u | ( a + d + g ) / sum(ALL)                      |
        | real_ratio_n | ( b + e + h ) / sum(ALL)                      |
        | real_ratio_d | ( c + f + i ) / sum(ALL)                      |
        | real_ratio_ud| ( a + d + g + c + f + i ) / sum(ALL)          |
        |              |                                               |
        | precision_u  | a  / ( a + b + c )                            |
        | precision_n  | e  / ( d + e + f )                            |
        | precision_d  | i  / ( g + h + i )                            |
        | precision_ud | ( a + i ) / ( a + b + c + g + h + i)          |
        |              |                                               |
        | recall_u     | a  / ( a + d + g )                            |
        | recall_n     | e  / ( b + e + h )                            |
        | recall_d     | i  / ( c + f + i )                            |
        | recall_ud    | ( a + i ) / ( a + d + g + c + f + i)          |
        |              |                                               |
        | hit_rate_u   | a  / ( a + c)                                 |
        | hit_rate_d   | i  / ( i + g )                                |
        | hit_rate_ud  | ( a + i ) / ( a +  c + i + g)                 |
        |              |                                               |
        | hedge_rate_u | ( a + b )  / ( a + b + c )                    |
        | hedge_rate_d | ( h + i )  / ( g + h + i )                    |
        | hedge_rate_ud| ( a + b + h + i ) / ( a + b + c + g + h + i)  |
        |              |                                               |
        | f1_u         | 2 * (recall_u * prec_u) / (recall_u + prec_u) |
        | f1_d         | 2 * (recall_d * prec_d) / (recall_d + prec_d) |
    """

    def __init__(self, y_true: list, y_pred: list):
        """
        Args:
            y_true     (list): Actual labels
            y_pred     (list): Predicted labels
        """
        assert all(
            [el in list([e.value for e in TernaryDirection]) for el in y_true]
        ), y_true
        assert all(
            [el in list([e.value for e in TernaryDirection]) for el in y_pred]
        ), y_pred
        assert len(y_true) == len(y_pred), (
            "Actual and predictred classes should have"
            "the same number of elements: {} != {}".format(len(y_true), len(y_pred))
        )
        self._y_true = np.array(y_true)
        self._y_pred = np.array(y_pred)
        self.metric_names = [
            attr
            for attr in dir(self)
            if not attr.startswith("__")
            and not attr.startswith("_")
            and attr not in ["get", "to_frame"]
        ]
        self._precompute()

    def _precompute(self):
        self._mask_a = (self._y_true == TernaryDirection.UP.value) & (
            self._y_pred == TernaryDirection.UP.value
        )
        self._mask_b = (self._y_true == TernaryDirection.NEUTRAL.value) & (
            self._y_pred == TernaryDirection.UP.value
        )
        self._mask_c = (self._y_true == TernaryDirection.DOWN.value) & (
            self._y_pred == TernaryDirection.UP.value
        )
        self._mask_d = (self._y_true == TernaryDirection.UP.value) & (
            self._y_pred == TernaryDirection.NEUTRAL.value
        )
        self._mask_e = (self._y_true == TernaryDirection.NEUTRAL.value) & (
            self._y_pred == TernaryDirection.NEUTRAL.value
        )
        self._mask_f = (self._y_true == TernaryDirection.DOWN.value) & (
            self._y_pred == TernaryDirection.NEUTRAL.value
        )
        self._mask_g = (self._y_true == TernaryDirection.UP.value) & (
            self._y_pred == TernaryDirection.DOWN.value
        )
        self._mask_h = (self._y_true == TernaryDirection.NEUTRAL.value) & (
            self._y_pred == TernaryDirection.DOWN.value
        )
        self._mask_i = (self._y_true == TernaryDirection.DOWN.value) & (
            self._y_pred == TernaryDirection.DOWN.value
        )
        self._a = np.sum(self._mask_a)
        self._b = np.sum(self._mask_b)
        self._c = np.sum(self._mask_c)
        self._d = np.sum(self._mask_d)
        self._e = np.sum(self._mask_e)
        self._f = np.sum(self._mask_f)
        self._g = np.sum(self._mask_g)
        self._h = np.sum(self._mask_h)
        self._i = np.sum(self._mask_i)
        # tn, fp, fn, tp = confusion_matrix(
        #    self._y_true, self._y_pred, labels=[
        # TernaryDirection.DOWN.value, TernaryDirection.UP.value]).ravel()
        # self._a = tp
        # self._b = fp
        # self._c = fn
        # self._d = tn
        self._sumall = np.sum(
            [
                self._a,
                self._b,
                self._c,
                self._d,
                self._e,
                self._f,
                self._g,
                self._h,
                self._i,
            ]
        )

    def _get_flip_count(self, signal_arr):
        previous_s = 0
        count_flip = 0
        for i, s in enumerate(signal_arr):
            if i == 0:
                previous_s = s
            if i < len(signal_arr):
                if s != previous_s:
                    count_flip += 1
                    previous_s = s
        return count_flip

    @metric_property
    def accuracy(self):
        return (self._a + self._e + self._i) / self._sumall

    @metric_property
    def real_flip_count(self):
        return self._get_flip_count(self._y_true)

    @metric_property
    def pred_flip_count(self):
        return self._get_flip_count(self._y_pred)

    @metric_property
    def pred_ratio_u(self):
        return (self._a + self._b + self._c) / self._sumall

    @metric_property
    def pred_ratio_n(self):
        return (self._d + self._e + self._f) / self._sumall

    @metric_property
    def pred_ratio_d(self):
        return (self._g + self._h + self._i) / self._sumall

    @metric_property
    def pred_ratio_ud(self):
        return (
            self._a + self._b + self._c + self._g + self._h + self._i
        ) / self._sumall

    @metric_property
    def pred_count_u(self):
        return self._a + self._b + self._c

    @metric_property
    def pred_count_n(self):
        return self._d + self._e + self._f

    @metric_property
    def pred_count_d(self):
        return self._g + self._h + self._i

    @metric_property
    def pred_count_ud(self):
        return self._a + self._b + self._c + self._g + self._h + self._i

    @metric_property
    def real_ratio_u(self):
        return (self._a + self._d + self._g) / self._sumall

    @metric_property
    def real_ratio_n(self):
        return (self._b + self._e + self._h) / self._sumall

    @metric_property
    def real_ratio_d(self):
        return (self._c + self._f + self._i) / self._sumall

    @metric_property
    def real_ratio_ud(self):
        return (
            self._a + self._d + self._g + self._c + self._f + self._i
        ) / self._sumall

    @metric_property
    def precision_u(self):
        return self._a / (self._a + self._b + self._c)

    @metric_property
    def precision_n(self):
        return self._e / (self._d + self._e + self._f)

    @metric_property
    def precision_d(self):
        return self._i / (self._g + self._h + self._i)

    @metric_property
    def precision_ud(self):
        return (self._a + self._i) / (
            self._a + self._b + self._c + self._g + self._h + self._i
        )

    @metric_property
    def recall_u(self):
        return self._a / (self._a + self._d + self._g)

    @metric_property
    def recall_n(self):
        return self._e / (self._b + self._e + self._h)

    @metric_property
    def recall_d(self):
        return self._i / (self._c + self._f + self._i)

    @metric_property
    def recall_ud(self):
        return (self._a + self._i) / (
            self._a + self._d + self._g + self._c + self._f + self._i
        )

    @metric_property
    def hit_rate_u(self):
        return self._a / (self._a + self._c)

    @metric_property
    def hit_rate_d(self):
        return self._i / (self._i + self._g)

    @metric_property
    def hit_rate_ud(self):
        return (self._a + self._i) / (self._a + self._c + self._i + self._g)

    @metric_property
    def hedge_rate_u(self):
        return (self._a + self._b) / (self._a + self._b + self._c)

    @metric_property
    def hedge_rate_d(self):
        return (self._h + self._i) / (self._g + self._h + self._i)

    @metric_property
    def hedge_rate_ud(self):
        return (self._a + self._b + self._h + self._i) / (
            self._a + self._b + self._c + self._g + self._h + self._i
        )

    @metric_property
    def f1_u(self):
        return (
            2 * (self.recall_u * self.precision_u) / (self.recall_u + self.precision_u)
        )

    @metric_property
    def f1_d(self):
        return (
            2 * (self.recall_d * self.precision_d) / (self.recall_d + self.precision_d)
        )

    def get(self, metric_names: list = []) -> dict:
        if metric_names is None or len(metric_names) == 0:
            metric_names = self.metric_names
        ret = {}
        for metric_name in metric_names:
            ret[metric_name] = getattr(self, metric_name)
        return ret

    def to_frame(self):
        dic = {}
        for metric_name in self.metric_names:
            dic[metric_name] = getattr(self, metric_name)
        return pd.DataFrame(dic, index=[0]).astype(np.float64)
