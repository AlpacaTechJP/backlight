import unittest

import pandas as pd
import numpy as np
from backlight.metrics import ternary
from pandas.util.testing import assert_frame_equal


class TestBinaryMetricsClass(unittest.TestCase):
    def setUp(self):
        self.y_true = [-1.0, 1.0, -1.0, -1.0, -1.0]
        self.y_pred = [-1.0, 1.0, 1.0, -1.0, -1.0]
        self.b_metrics = binary.BinaryMetrics(self.y_true, self.y_pred)
        self.ans_dict = {
            "accuracy": 0.8,
            "real_label_diff": -3,
            "pred_label_diff": -1,
            "diff_w_real_diff": -2,
            "real_flip_count": 2,
            "pred_flip_count": 2,
            "pred_ratio_u": 0.4,
            "pred_ratio_d": 0.6,
            "real_ratio_u": 0.2,
            "real_ratio_d": 0.8,
            "precision_u": 0.5,
            "precision_d": 1,
            "recall_u": 1,
            "recall_d": 0.75,
            "pred_count_u": 2,
            "pred_count_d": 3,
            "f1_u": ((2 * 1 * 0.5) / 1.5),
            "f1_d": ((2 * 1 * 0.75) / 1.75),
        }

    def test_metric_properties(self):
        assert self.b_metrics.accuracy == 0.8
        assert self.b_metrics.real_label_diff == -3
        assert self.b_metrics.pred_label_diff == -1
        assert self.b_metrics.diff_w_real_diff == -2
        assert self.b_metrics.real_flip_count == 2
        assert self.b_metrics.pred_flip_count == 2
        assert self.b_metrics.pred_ratio_u == 0.4
        assert self.b_metrics.pred_ratio_d == 0.6
        assert self.b_metrics.real_ratio_u == 0.2
        assert self.b_metrics.real_ratio_d == 0.8
        assert self.b_metrics.precision_u == 0.5
        assert self.b_metrics.precision_d == 1
        assert self.b_metrics.recall_u == 1
        assert self.b_metrics.recall_d == 0.75
        assert self.b_metrics.pred_count_u == 2
        assert self.b_metrics.pred_count_d == 3
        assert self.b_metrics.f1_u == ((2 * 1 * 0.5) / 1.5)
        assert self.b_metrics.f1_d == ((2 * 1 * 0.75) / 1.75)

    def test_get(self):
        rlt_dict = self.b_metrics.get()
        self.assertDictEqual(self.ans_dict, rlt_dict)

    def test_to_frame(self):
        ans_df = pd.DataFrame(self.ans_dict, index=[0]).astype(np.float64)
        rlt_df = self.b_metrics.to_frame()
        assert_frame_equal(ans_df, rlt_df)
