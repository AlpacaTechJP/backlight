import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from backlight.metrics import binary_extend
from backlight.trades import Trades


class TestBinaryMetricsClass(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.y_true = [-1.0, 1.0, -1.0, -1.0, -1.0]
        self.y_pred = [-1.0, 1.0, 1.0, -1.0, -1.0]
        self.trades = Trades(
            pd.DataFrame(
                {
                    "lag0_dt_close": [100, 102, 101, 103, 106],
                    #                 2    -1   2    3
                    # 1.02 * -0.99019608 * 1.01980198 *  * 1.02912621
                    "amount": [-1.0, 1.0, 1.0, -1.0, -1.0],
                }
            ),
            "lag0_dt_close",
        )
        self.abpl = binary_extend.AmountBasedPL(self.y_true, self.y_pred, self.trades)
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
            "total_pl": -4.0,
            "total_pl_percentage": (-4.0 / 100),
            "accumulate_pl_percentage": round(
                (
                    ((-1 * 0.02) + 1)
                    * ((1 * -0.00980392156863) + 1)
                    * ((1 * 0.019801980198) + 1)
                    * ((-1 * 0.0291262135922) + 1)
                )
                - 1,
                13,
            ),
            "baseline_pl": 6.0,
            "baseline_pl_percentage": (6.0 / 100),
            "average_win": 2.0,
            "average_loss": -2.0,
            "average_pl": -1.0,
        }

    def test_metric_properties(self):
        assert self.abpl.total_pl == -4.0
        assert self.abpl.total_pl_percentage == (-4.0 / 100)
        assert self.abpl.baseline_pl == 6.0
        assert self.abpl.baseline_pl_percentage == (6.0 / 100)
        assert self.abpl.average_win == 2.0
        assert self.abpl.average_loss == -2.0
        assert self.abpl.average_pl == -1.0

    def test_average_pl_only_take_short_case(self):
        trades = Trades(
            pd.DataFrame(
                {
                    "lag0_dt_close": [100, 102, 101, 103, 106],
                    #                      2    -1   2    3
                    "amount": [-1.0, 0.0, 0.0, -1.0, -1.0],
                }
            ),
            "lag0_dt_close",
        )
        abpl = binary_extend.AmountBasedPL(self.y_true, self.y_pred, trades)
        assert abpl.average_pl == -2.5

    def test_get(self):
        rlt_dict = self.abpl.get()
        for k, v in self.ans_dict.items():
            assert abs(rlt_dict[k] - v) < 0.00000001

    def test_to_frame(self):
        ans_df = pd.DataFrame(self.ans_dict, index=[0]).astype(np.float64)
        rlt_df = self.abpl.to_frame()
        assert_frame_equal(ans_df, rlt_df)
