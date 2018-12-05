from backlight.metrics import pl as module
import pandas as pd
import pytest

import backlight.datasource
import backlight.positions
from backlight.trades.trades import from_series


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def market(symbol):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol)


@pytest.fixture
def trades(symbol):
    symbol = "usdjpy"
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = from_series(
            pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount"),
            symbol,
        )
        trades.append(trade)
    return tuple(trades)


@pytest.fixture
def positions(trades, market):
    # positions should be
    # data = [
    #     [1.0, 1.0],  # pl = None
    #     [-1.0, 2.0],  # pl = 1.0 * (2.0 - 1.0) = 1.0
    #     [0.0, 3.0],  # pl = -1.0 * (3.0 - 2.0) = -1.0
    #     [2.0, 4.0],  # pl = 0.0 * (4.0 - 3.0) = 0.0
    #     [-2.0, 5.0],  # pl = 2.0 * (5.0 - 4.0) = 2.0
    #     [0.0, 6.0],  # pl = -2.0 * (6.0 - 5.0) = -2.0
    #     [1.0, 7.0],  # pl = 0.0 * (7.0 - 6.0) = 0.0
    #     [1.0, 8.0],  # pl = 1.0 * (8.0 - 7.0) = 1.0
    #     [2.0, 9.0],  # pl = 1.0 * (9.0 - 8.0) = 1.0
    #     [2.0, 9.0],  # pl = 2.0 * (9.0 - 9.0) = 0.0
    # ]
    return backlight.positions.calc_positions(trades, market)


def test__trade_amount(positions):
    expected = 13.0
    assert module._trade_amount(positions.amount) == expected


def test_calc_position_performance(positions):
    metrics = module.calc_position_performance(positions)
    expected_total_pl = 2.0
    expected_win_pl = 5.0
    expected_lose_pl = -3.0
    expected_trade_amount = 13.0
    expected_avg_pl = expected_total_pl / expected_trade_amount
    assert metrics.loc["metrics", "total_pl"] == expected_total_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_lose_pl"] == expected_lose_pl
    assert metrics.loc["metrics", "cnt_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl_per_amount"] == expected_avg_pl


def test_calc_trade_performance(trades, market):
    metrics = module.calc_trade_performance(trades, market)
    expected_cnt_trade = 5
    expected_cnt_win = 3
    expected_cnt_lose = 1
    expected_total_pl = 2.0
    expected_win_pl = 5.0
    expected_lose_pl = -3.0
    expected_trade_amount = 13.0
    expected_avg_pl = expected_total_pl / expected_trade_amount
    assert metrics.loc["metrics", "cnt_trade"] == expected_cnt_trade
    assert metrics.loc["metrics", "cnt_win"] == expected_cnt_win
    assert metrics.loc["metrics", "cnt_lose"] == expected_cnt_lose
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_lose_pl"] == expected_lose_pl
    assert metrics.loc["metrics", "cnt_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl_per_amount"] == expected_avg_pl
