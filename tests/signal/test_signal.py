from backlight.signal import signal as module
import pandas as pd
import numpy as np

from backlight.labelizer.common import TernaryDirection


def test_TernarySignal():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=7),
        data=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        columns=["up", "neutral", "down"],
    )
    signal = module.TernarySignal(df)
    signal.reset_pred()
    expected = np.array(
        [
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.UP.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.DOWN.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
        ]
    )
    np.testing.assert_array_equal(signal.pred.values, expected)


def test_TernarySignal_with_treshold():
    th = 0.5
    low = th - 0.1

    data = [
        [0, 0, 0],
        [th, 0, 0],
        [0, th, 0],
        [0, 0, th],
        [low, 0, 0],
        [0, low, 0],
        [0, 0, low],
        [th, th, 0],
        [th, 0, th],
        [0, th, th],
    ]

    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=len(data)),
        data=data,
        columns=["up", "neutral", "down"],
    )
    signal = module.TernarySignal(df)
    signal.reset_pred(threshold=th)
    expected = np.array(
        [
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.UP.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.DOWN.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
            TernaryDirection.NEUTRAL.value,
        ]
    )
    np.testing.assert_array_equal(signal.pred.values, expected)
