from backlight.metrics import ternary as module
import pandas as pd
import pytest

import backlight.datasource
import backlight.labelizer
from backlight.asset.currency import Currency


@pytest.fixture
def symbol():
    return "USDJPY"


@pytest.fixture
def currency_unit():
    return Currency.JPY


@pytest.fixture
def signal(symbol, currency_unit):
    periods = 18
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        columns=["up", "neutral", "down"],
    )
    signal = backlight.signal.from_dataframe(df, symbol, currency_unit)
    return signal


@pytest.fixture
def label(currency_unit):
    periods = 18
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[
            [1, 1, 0.0],
            [1, 1, 0.0],
            [1, 1, 0.0],
            [1, 1, 0.0],
            [1, 1, 0.0],
            [1, 1, 0.0],
            [0, 0, 0.0],
            [0, 0, 0.0],
            [0, 0, 0.0],
            [0, 0, 0.0],
            [0, 0, 0.0],
            [0, 0, 0.0],
            [-1, -1, 0.0],
            [-1, -1, 0.0],
            [-1, -1, 0.0],
            [-1, -1, 0.0],
            [-1, -1, 0.0],
            [-1, -1, 0.0],
        ],
        columns=["label_diff", "label", "neutral_range"],
    )
    label = backlight.labelizer.from_dataframe(df, currency_unit)
    return label


def test_calculate_ternary_metrics(signal, label):
    total = len(signal)
    m = module.calculate_ternary_metrics(signal, label)
    uu = 3
    un = 3
    ud = 3
    nu = 1
    nn = 1
    nd = 1
    du = 2
    dn = 2
    dd = 2

    assert m.loc["metrics", "cnt_uu"] == uu
    assert m.loc["metrics", "cnt_un"] == un
    assert m.loc["metrics", "cnt_ud"] == ud
    assert m.loc["metrics", "cnt_nu"] == nu
    assert m.loc["metrics", "cnt_nn"] == nn
    assert m.loc["metrics", "cnt_nd"] == nd
    assert m.loc["metrics", "cnt_dd"] == dd
    assert m.loc["metrics", "cnt_du"] == du
    assert m.loc["metrics", "cnt_dn"] == dn
    assert m.loc["metrics", "cnt_total"] == total

    assert m.loc["metrics", "hit_ratio"] == 5 / 10
    assert m.loc["metrics", "hit_ratio_u"] == 3 / 6
    assert m.loc["metrics", "hit_ratio_d"] == 2 / 4

    assert m.loc["metrics", "hedge_ratio"] == 10 / 15
    assert m.loc["metrics", "hedge_ratio_u"] == 6 / 9
    assert m.loc["metrics", "hedge_ratio_d"] == 4 / 6

    assert m.loc["metrics", "neutral_ratio"] == 3 / total
    assert m.loc["metrics", "coverage"] == 15 / total
    assert m.loc["metrics", "coverage_u"] == 9 / total
    assert m.loc["metrics", "coverage_d"] == 6 / total
