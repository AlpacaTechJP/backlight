import numpy as np
import pandas as pd

from ..labelizer.common import TernaryDirection
from .common import metric_property, sync_index_and_get_list


def calc_binary_metrics(sig, lbl):
    real_arr, pred_arr = sync_index_and_get_list(lbl, sig)
    b_metrics = BinaryMetrics(real_arr, pred_arr)
    return b_metrics


class BinaryMetrics:
    """Binary metrics.
    Confusion matrix:
                      |      Actual
        | Predicted   | Up             |  Down          |
        | ----------- | -------------- |  ------------- |
        | Up          | a              |  b             |
        | Down        | c              |  d             |

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
        | Name            | Definition                                    |
        | --------------- | --------------------------------------------  |
        | ALL             | a + b + c + d                                 |
        | --------------- | --------------------------------------------  |
        | accuracy        | ( a + d ) / sum(ALL)                          |
        | real_label_diff | ( a + c ) - ( b + d )                         |
        | pred_label_diff | ( a + b ) - ( c + d )                         |
        | diff_w_real_diff| 2b - 2c                                       |
        |                 |                                               |
        | PPV             | a / ( a + b )                                 |
        | NPV             | d / ( c + d )                                 |
        |                 |                                               |
        | pred_ratio_u    | ( a + b ) / sum(ALL)                          |
        | pred_ratio_d    | ( c + d ) / sum(ALL)                          |
        |                 |                                               |
        | real_ratio_u    | ( a + c ) / sum(ALL)                          |
        | real_ratio_d    | ( b + d ) / sum(ALL)                          |
        |                 |                                               |
        | pred_count_u    | a + b                                         |
        | pred_count_d    | c + d                                         |
        |                 |                                               |
        | precision_u     | a / ( a + b )                                 |
        | precision_d     | d / ( c + d )                                 |
        |                 |                                               |
        | recall_u        | a / ( a + c )                                 |
        | recall_d        | d / ( b + d )                                 |
        |                 |                                               |
        | f1_u            | 2 * (recall_u * prec_u) / (recall_u + prec_u) |
        | f1_d            | 2 * (recall_d * prec_d) / (recall_d + prec_d) |
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
        self._mask_b = (self._y_true == TernaryDirection.DOWN.value) & (
            self._y_pred == TernaryDirection.UP.value
        )
        self._mask_c = (self._y_true == TernaryDirection.UP.value) & (
            self._y_pred == TernaryDirection.DOWN.value
        )
        self._mask_d = (self._y_true == TernaryDirection.DOWN.value) & (
            self._y_pred == TernaryDirection.DOWN.value
        )
        self._a = np.sum(self._mask_a)
        self._b = np.sum(self._mask_b)
        self._c = np.sum(self._mask_c)
        self._d = np.sum(self._mask_d)
        self._tp = self._a
        self._fp = self._b
        self._fn = self._c
        self._tn = self._d
        self.uu = self._a
        self.ud = self._b
        self.du = self._c
        self.dd = self._d
        # tn, fp, fn, tp = confusion_matrix(
        #    self._y_true, self._y_pred, labels=[
        # TernaryDirection.DOWN.value, TernaryDirection.UP.value]).ravel()
        # self._a = tp
        # self._b = fp
        # self._c = fn
        # self._d = tn
        self._sumall = np.sum([self._a, self._b, self._c, self._d])

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
        return (self._a + self._d) / self._sumall

    @metric_property
    def real_label_diff(self):
        return (self._a + self._c) - (self._b + self._d)

    @metric_property
    def pred_label_diff(self):
        return (self._a + self._b) - (self._c + self._d)

    @metric_property
    def diff_w_real_diff(self):
        return (2 * self._c) - (2 * self._b)

    @metric_property
    def real_flip_count(self):
        return self._get_flip_count(self._y_true)

    @metric_property
    def pred_flip_count(self):
        return self._get_flip_count(self._y_pred)

    @metric_property
    def pred_ratio_u(self):
        return (self._a + self._b) / self._sumall

    @metric_property
    def pred_ratio_d(self):
        return (self._c + self._d) / self._sumall

    @metric_property
    def real_ratio_u(self):
        return (self._a + self._c) / self._sumall

    @metric_property
    def real_ratio_d(self):
        return (self._b + self._d) / self._sumall

    @metric_property
    def precision_u(self):
        return self._a / (self._a + self._b)

    @metric_property
    def precision_d(self):
        return self._d / (self._c + self._d)

    @metric_property
    def recall_u(self):
        return self._a / (self._a + self._c)

    @metric_property
    def recall_d(self):
        return self._d / (self._b + self._d)

    @metric_property
    def pred_count_u(self):
        return self._a + self._b

    @metric_property
    def pred_count_d(self):
        return self._c + self._d

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
