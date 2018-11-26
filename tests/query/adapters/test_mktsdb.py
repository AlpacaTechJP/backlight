from backlight.query.adapters import mktsdb as module
from unittest import mock
import pandas as pd
import os


@mock.patch.dict(os.environ, {"TICK_MARKETSTORE_HOST": "8888"})
def test_MarketstoreAdapter():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )

    with mock.patch("backlight.query.adapters.mktsdb.client") as mocked:
        cli = mocked.Client()
        cli.query.return_value = df
        m = module.MarketstoreAdapter(url=None)
        res = m.query("ABC", "2018-06-06", "2018-06-10")
        cli.query.assert_called_with(
            symbol="ABC", timeframe="1Min", end_dt="2018-06-10", start_dt="2018-06-06"
        )
        assert df.equals(res)
