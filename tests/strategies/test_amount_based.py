import pandas as pd
from backlight.datasource.marketdata import MarketData
from backlight.signal.signal import BinaryOneColumnLabelSignal
from backlight.strategies import amount_based
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action
from backlight.trades import Trades
from pandas.util.testing import assert_frame_equal


def test_direction_based_trades():
    index = [
        pd.Timestamp("2017-12-01"),
        pd.Timestamp("2017-12-02"),
        pd.Timestamp("2017-12-05"),
        pd.Timestamp("2017-12-06"),
        pd.Timestamp("2017-12-07"),
        pd.Timestamp("2017-12-08"),
    ]
    sig = BinaryOneColumnLabelSignal(
        pd.DataFrame({"label": [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]}, index=index), "Topix"
    )
    mkt = MarketData(
        pd.DataFrame(
            {
                "lag0_dt_close": [1.0, 1.2, 1.5, 2.4, 0.75, 4.8],
                "lag0_styles_growth": [2.0, 2.0, 4.0, 3.0, 3.0, 6.0],
                "lag0_styles_value": [1.0, 4.0, 2.5, 7.0, 1.25, 14.0],
            },
            index=index,
        ),
        "Topix",
        sig._start_dt,
        sig._end_dt,
    )
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    ans_amount_df = pd.DataFrame(
        {"amount": [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]}, index=index
    )
    ans_trades_df = pd.concat([ans_amount_df, mkt], axis=1, join="inner")
    ans_trades = Trades(ans_trades_df, "lag0_dt_close")
    result_trades = amount_based.direction_based_trades(
        mkt, sig, "lag0_dt_close", direction_action_dict
    )
    ans_trades = ans_trades.sort_index(axis=1)
    result_trades = result_trades.sort_index(axis=1)
    assert_frame_equal(result_trades, ans_trades)
