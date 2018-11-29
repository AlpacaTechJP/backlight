from backlight import trades as module
import pandas as pd

import backlight.trades as tr
import backlight.datasource as ds


def test__pl():
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

    trade = tr.Trade()
    trade.add(t00)
    trade.add(t11)
    assert module._pl(trade, mkt) == 1.0

    trade = tr.Trade()
    trade.add(t00)
    trade.add(t01)
    assert module._pl(trade, mkt) == 0.0

    trade = tr.Trade()
    trade.add(t11)
    trade.add(t20)
    assert module._pl(trade, mkt) == -1.0

    trade = tr.Trade()
    trade.add(t00).add(t10).add(t20)
    assert module._pl(trade, mkt) == 3.0


def test_make_trades():
    periods = 3
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = [1.0, -1.0]

    t00 = tr.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = tr.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = tr.Transaction(timestamp=dates[0], amount=amounts[1])
    t20 = tr.Transaction(timestamp=dates[2], amount=amounts[0])
    trade1 = tr.Trade().add(t00).add(t11)
    trade2 = tr.Trade().add(t00).add(t01)
    trade3 = tr.Trade().add(t11).add(t20)

    mkt = ds.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [1], [2]], columns=["mid"]), symbol
    )

    trades = module.make_trades([trade1, trade2, trade3], mkt)

    expected = pd.Series(
        index=dates,
        data=[amounts[0] * 2 + amounts[1], amounts[1] * 2, amounts[0]],
        name="amount",
    )
    assert trades.symbol == symbol
    assert trades.trades == [trade1, trade2, trade3]
    assert (trades.amount == expected).all()
    assert module.count(trades) == (3, 1, 1)

    expected = pd.Series(
        index=dates[1:], data=[amounts[1] * 2, amounts[0]], name="amount"
    )
    masked = trades[trades.index >= dates[1]]
    assert masked.symbol == symbol
    assert masked.trades == [trade3]
    assert (masked.amount == expected).all()
    assert module.count(masked) == (1, 0, 1)

    expected = pd.Series(
        index=dates[:2],
        data=[amounts[0] * 2 + amounts[1], amounts[1] * 2],
        name="amount",
    )
    masked = trades[trades.index <= dates[1]]
    assert masked.symbol == symbol
    assert masked.trades == [trade1, trade2]
    assert (masked.amount == expected).all()
    assert module.count(masked) == (2, 1, 0)
