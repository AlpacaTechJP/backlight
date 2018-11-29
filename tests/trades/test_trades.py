from backlight.trades import trades as module
import pandas as pd


def test_Trade():
    periods = 2
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = range(periods)

    t00 = module.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = module.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = module.Transaction(timestamp=dates[0], amount=amounts[1])

    trade = module.Trade([t00, t11], symbol)
    expected = pd.Series(index=dates, data=amounts[:2], name="amount")
    assert trade.symbol == symbol
    assert (trade.amount == expected).all()

    trade = module.Trade([t00, t01], symbol)
    expected = pd.Series(
        index=[dates[0]], data=[amounts[0] + amounts[1]], name="amount"
    )
    assert trade.symbol == symbol
    assert (trade.amount == expected).all()

    trade = module.Trade([t11, t01, t00], symbol)
    expected = pd.Series(
        index=dates, data=[amounts[0] + amounts[1], amounts[1]], name="amount"
    )
    assert trade.symbol == symbol
    assert (trade.amount == expected).all()
