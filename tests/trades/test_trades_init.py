from backlight import trades as module
import pandas as pd

import backlight.trades as tr
import backlight.datasource as ds


def test_evaluate_pl():
    periods = 3
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = [1.0, -1.0]

    t00 = tr.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = tr.Transaction(timestamp=dates[1], amount=amounts[1])
    t10 = tr.Transaction(timestamp=dates[1], amount=amounts[0])
    t01 = tr.Transaction(timestamp=dates[0], amount=amounts[1])
    t20 = tr.Transaction(timestamp=dates[2], amount=amounts[0])

    mkt = ds.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [1], [2]], columns=["mid"]), symbol
    )

    trade = tr.make_trade(symbol).add(t00).add(t11)
    assert module.evaluate_pl(trade, mkt) == 1.0

    trade = tr.make_trade(symbol).add(t00).add(t01)
    assert module.evaluate_pl(trade, mkt) == 0.0

    trade = tr.make_trade(symbol).add(t11).add(t20)
    assert module.evaluate_pl(trade, mkt) == -1.0

    trade = tr.make_trade(symbol).add(t00).add(t10).add(t20)
    assert module.evaluate_pl(trade, mkt) == 3.0
