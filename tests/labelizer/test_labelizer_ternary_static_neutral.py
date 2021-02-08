from backlight.labelizer.ternary.static_neutral import StaticNeutralLabelizer as module

import pytest
import pandas as pd
import numpy as np
import datetime


@pytest.fixture
def sample_df():
    index = pd.date_range(
        "2017-09-04 13:00:00+00:00", "2017-09-05 13:00:00+00:00", freq="1H"
    )
    return pd.DataFrame(
        index=index,
        data=np.array(
            [
                [109.68, 109.69, 109.685],
                [109.585, 109.595, 109.59],
                [109.525, 109.535, 109.53],
                [109.6, 109.61, 109.605],
                [109.695, 109.7, 109.6975],
                [109.565, 109.705, 109.635],
                [109.63, 109.685, 109.6575],
                [109.555, 109.675, 109.615],
                [109.7, 109.75, 109.725],
                [109.67, 109.72, 109.695],
                [109.66, 109.675, 109.6675],
                [109.8, 109.815, 109.8075],
                [109.565, 109.575, 109.57],
                [109.535, 109.545, 109.54],
                [109.32, 109.33, 109.325],
                [109.27, 109.275, 109.2725],
                [109.345, 109.355, 109.35],
                [109.305, 109.315, 109.31],
                [109.3, 109.31, 109.305],
                [109.445, 109.46, 109.4525],
                [109.42, 109.425, 109.4225],
                [109.385, 109.395, 109.39],
                [109.305, 109.315, 109.31],
                [109.365, 109.375, 109.37],
                [109.365, 109.375, 109.37],
            ]
        ),
        columns=["bid", "ask", "mid"],
    )


def test_create(sample_df):
    lbl_args = {
        "lookahead": "1H",
        "neutral_ratio": 0.5,
        "session_splits": [datetime.time(9), datetime.time(18)],
        "neutral_hard_limit": 0.00,
        "window_start": "20170904 12:00:00+0000",
        "window_end": "20170905 06:00:00+0000",
    }

    lbl = module(**lbl_args).create(sample_df)
    assert lbl.label.sum() == 1
    assert lbl.neutral_range.isna().sum() == 0
