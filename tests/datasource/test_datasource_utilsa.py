from backlight.datasource import utils as module
import pandas as pd
from backlight.asset.currency import Currency


def test_from_dataframe():
    symbol = "USDJPY"
    currency_unit = Currency.JPY
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )
    mkt = module.from_dataframe(df, symbol, currency_unit)
    assert mkt.symbol == symbol
    assert mkt.currency_unit == currency_unit
    assert all(mkt.mid.values == [1, 3, 5])

    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["aaa", "bbb"],
    )
    col_mapping = {"aaa": "ask", "bbb": "bid"}
    mkt = module.from_dataframe(df, symbol, currency_unit, col_mapping=col_mapping)
    assert mkt.symbol == symbol
    assert mkt.currency_unit == currency_unit
    assert all(mkt.mid.values == [1, 3, 5])


def test_mid2askbid():
    symbol = "USDJPY"
    currency_unit = Currency.JPY
    mid = module.from_dataframe(
        pd.DataFrame(
            index=pd.date_range(start="2018-06-06", periods=3),
            data=[[1], [3], [5]],
            columns=["mid"],
        ),
        symbol,
        currency_unit,
    )

    expected = module.from_dataframe(
        pd.DataFrame(
            index=pd.date_range(start="2018-06-06", periods=3),
            data=[[0, 2], [2, 4], [4, 6]],
            columns=["bid", "ask"],
        ),
        symbol,
        currency_unit,
    )

    mkt = module.mid2askbid(mid, 1.0)
    assert mkt.symbol == symbol
    assert mkt.currency_unit == currency_unit
    assert (mkt.ask == expected.ask).all()
    assert (mkt.bid == expected.bid).all()
