import unittest

import pandas as pd
from backlight.labelizer.common import LabelType
from backlight.labelizer.labelizer import Label
from backlight.metrics import common
from backlight.signal.signal import BinaryOneColumnLabelSignal


class TestCommonModule(unittest.TestCase):
    def test_sync_index_and_get_list(self):
        sig = BinaryOneColumnLabelSignal(
            pd.DataFrame(
                {"label": [1.0, -1.0, 1.0, 1.0, -1.0]},
                index=[
                    pd.Timestamp("2017-12-01"),
                    pd.Timestamp("2017-12-02"),
                    pd.Timestamp("2017-12-05"),
                    pd.Timestamp("2017-12-06"),
                    pd.Timestamp("2017-12-07"),
                ],
            ),
            "Topix",
        )
        lbl = Label(
            LabelType.BINARY,
            pd.DataFrame(
                {"label": [-1.0, 1.0, -1.0, -1.0, 1.0]},
                index=[
                    pd.Timestamp("2017-12-02"),
                    pd.Timestamp("2017-12-05"),
                    pd.Timestamp("2017-12-06"),
                    pd.Timestamp("2017-12-07"),
                    pd.Timestamp("2017-12-08"),
                ],
            ),
        )
        ans_real_arr = [-1.0, 1.0, -1.0, -1.0]
        ans_pred_arr = [-1.0, 1.0, 1.0, -1.0]
        rlt_real_arr, rlt_pred_arr = common.sync_index_and_get_list(lbl, sig)
        self.assertListEqual(ans_real_arr, rlt_real_arr)
        self.assertListEqual(ans_pred_arr, rlt_pred_arr)
