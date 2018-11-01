from pome.labelizer import common
import pandas as pd


def test_TernaryDirection():
    assert common.TernaryDirection.UP.value == 1
    assert common.TernaryDirection.NEUTRAL.value == 0
    assert common.TernaryDirection.DOWN.value == -1


def test_get_majority_label():
    se = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
    result = common.get_majority_label(se)
    assert result == 1.0
