import unittest

import numpy as np
import pandas as pd
from backlight.metrics import ternary
from pandas.util.testing import assert_frame_equal


class TestBinaryMetricsClass(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.y_true = [-1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0, 0.0,
                       -1.0, 1.0, 1.0]
        self.y_pred = [-1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0,
                       0.0, -1.0, 1.0]
        self.b_metrics = ternary.TernaryMetrics(self.y_true, self.y_pred)
        self.ans_dict = {
            "accuracy": (2 + 1 + 3) / 13.0,
            "real_flip_count": 7,
            "pred_flip_count": 8,
            "pred_ratio_u": 5 / 13.0,
            "pred_ratio_n": 3 / 13.0,
            "pred_ratio_d": 5 / 13.0,
            "pred_ratio_ud": 10 / 13.0,
            "pred_count_u": 5,
            "pred_count_n": 3,
            "pred_count_d": 5,
            "pred_count_ud": 10,
            "real_ratio_u": 4 / 13.0,
            "real_ratio_n": 3 / 13.0,
            "real_ratio_d": 6 / 13.0,
            "real_ratio_ud": 10 / 13.0,
            "precision_u": 2 / 5.0,
            "precision_n": 1 / 3.0,
            "precision_d": 3 / 5.0,
            "precision_ud": 5 / 10.0,
            "recall_u": 2 / 4.0,
            "recall_n": 1 / 3.0,
            "recall_d": 3 / 6.0,
            "recall_ud": 5 / 10.0,
            "hit_rate_u": 2 / 4.0,
            "hit_rate_d": 3 / 4.0,
            "hit_rate_ud": 5 / 8.0,
            "hedge_rate_u": 3 / 5.0,
            "hedge_rate_d": 4 / 5.0,
            "hedge_rate_ud": 7 / 10.0,
            "f1_u": ((2 * 0.5 * 0.4) / 0.9),
            "f1_d": ((2 * 0.5 * 0.6) / 1.1),
        }

    def test_metric_properties(self):
        assert self.b_metrics._a == 2
        assert self.b_metrics._b == 1
        assert self.b_metrics._c == 2
        assert self.b_metrics._d == 1
        assert self.b_metrics._e == 1
        assert self.b_metrics._f == 1
        assert self.b_metrics._g == 1
        assert self.b_metrics._h == 1
        assert self.b_metrics._i == 3

        assert self.b_metrics.accuracy == (2 + 1 + 3) / 13.0
        assert self.b_metrics.real_flip_count == 7
        assert self.b_metrics.pred_flip_count == 8

        assert self.b_metrics.pred_ratio_u == 5 / 13.0
        assert self.b_metrics.pred_ratio_n == 3 / 13.0
        assert self.b_metrics.pred_ratio_d == 5 / 13.0
        assert self.b_metrics.pred_ratio_ud == 10 / 13.0

        assert self.b_metrics.pred_count_u == 5
        assert self.b_metrics.pred_count_n == 3
        assert self.b_metrics.pred_count_d == 5
        assert self.b_metrics.pred_count_ud == 10

        assert self.b_metrics.real_ratio_u == 4 / 13.0
        assert self.b_metrics.real_ratio_n == 3 / 13.0
        assert self.b_metrics.real_ratio_d == 6 / 13.0
        assert self.b_metrics.real_ratio_ud == 10 / 13.0

        assert self.b_metrics.precision_u == 2 / 5.0
        assert self.b_metrics.precision_n == 1 / 3.0
        assert self.b_metrics.precision_d == 3 / 5.0
        assert self.b_metrics.precision_ud == 5 / 10.0

        assert self.b_metrics.recall_u == 2 / 4.0
        assert self.b_metrics.recall_n == 1 / 3.0
        assert self.b_metrics.recall_d == 3 / 6.0
        assert self.b_metrics.recall_ud == 5 / 10.0

        assert self.b_metrics.hit_rate_u == 2 / 4.0
        assert self.b_metrics.hit_rate_d == 3 / 4.0
        assert self.b_metrics.hit_rate_ud == 5 / 8.0

        assert self.b_metrics.hedge_rate_u == 3 / 5.0
        assert self.b_metrics.hedge_rate_d == 4 / 5.0
        assert self.b_metrics.hedge_rate_ud == 7 / 10.0

        assert self.b_metrics.f1_u == ((2 * 0.5 * 0.4) / 0.9)
        assert self.b_metrics.f1_d == ((2 * 0.5 * 0.6) / 1.1)

    def test_get(self):
        rlt_dict = self.b_metrics.get()
        self.assertDictEqual(self.ans_dict, rlt_dict)

    def test_to_frame(self):
        ans_df = pd.DataFrame(self.ans_dict, index=[0]).astype(np.float64)
        rlt_df = self.b_metrics.to_frame()
        assert_frame_equal(ans_df, rlt_df)
