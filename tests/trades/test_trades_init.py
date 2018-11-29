from backlight import trades as module
import pandas as pd

import backlight.trades as tr
import backlight.datasource as ds


def test_Trades():
    periods = 2
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = range(periods)

    t00 = tr.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = tr.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = tr.Transaction(timestamp=dates[0], amount=amounts[1])
    trade1 = tr.Trade([t00, t11], symbol)
    trade2 = tr.Trade([t00, t01], symbol)

    mkt = ds.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [2]], columns=["mid"]), symbol
    )

    trades = module.make_trades([trade1, trade2], mkt)

    expected = pd.Series(
        index=dates, data=[amounts[0] * 2 + amounts[1], amounts[1]], name="amount"
    )
    assert trades.symbol == symbol
    assert trades.trades == [trade1, trade2]
    assert (trades.amount == expected).all()

    expected = pd.Series(
        index=[dates[1]], data=[amounts[1]], name="amount"
    )
    masked = trades[trades.index == dates[1]]
    assert masked.symbol == symbol
    assert masked.trades == []
    assert (masked.amount == expected).all()
