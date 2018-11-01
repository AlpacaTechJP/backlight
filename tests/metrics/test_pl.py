from backlight.metrics import pl as module
import pandas as pd
import pytest


@pytest.fixture
def positions():
    return pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=5),
        data=[[1.0, 2], [2.0, 0], [0.5, -6], [3.0, -2], [1.0, -2]],
        columns=["price", "amount"],
    )


def test_pl_metrics_with_same_range(positions):
    # same range of positions index
    start_dt = "2018-06-06"
    end_dt = "2018-06-10"
    metrics = module.PlMetrics(positions, start_dt, end_dt)

    pls = [2.0, 0.0, -15.0, 4.0]
    expected_pl = pd.Series(
        index=pd.date_range(start="2018-06-07", periods=4), data=pls
    )
    assert metrics.pl.equals(expected_pl)

    expected_trade_count = 3
    assert metrics.trade_count == expected_trade_count

    expected_total_pl = sum(pls)
    assert metrics.total_pl == expected_total_pl

    expected_average_pl = expected_total_pl / expected_trade_count
    assert metrics.average_pl == expected_average_pl

    expected_total_win = 6.0
    assert metrics.total_win == expected_total_win

    expected_total_loss = -15.0
    assert metrics.total_loss == expected_total_loss


def test_pl_metrics_with_shorter_range(positions):
    # shorter range of positions index
    start_dt = "2018-06-07"
    end_dt = "2018-06-09"
    metrics = module.PlMetrics(positions, start_dt, end_dt)

    pls = [0.0, -15.0]
    expected_pl = pd.Series(
        index=pd.date_range(start="2018-06-08", periods=2), data=pls
    )
    assert metrics.pl.equals(expected_pl)

    expected_trade_count = 2
    assert metrics.trade_count == expected_trade_count


def test_pl_metrics_with_zero_range(positions):
    # shorter range of positions index
    start_dt = "2018-06-07"
    end_dt = "2018-06-07"
    metrics = module.PlMetrics(positions, start_dt, end_dt)

    expected_pl = pd.Series()
    assert metrics.pl.equals(expected_pl)

    expected_trade_count = 0
    assert metrics.trade_count == expected_trade_count

    expected_total_pl = 0.0
    assert metrics.total_pl == expected_total_pl

    expected_average_pl = 0.0
    assert metrics.average_pl == expected_average_pl

    expected_total_win = 0.0
    assert metrics.total_win == expected_total_win

    expected_total_loss = 0.0
    assert metrics.total_loss == expected_total_loss
