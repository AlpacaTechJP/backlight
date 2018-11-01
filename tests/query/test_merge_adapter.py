from pome.query import merge_adapter as module
from unittest import mock
import pandas as pd


def test_MergeAdapter():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )

    urls = ["file://hoge.csv", "s3://huga.csv"]
    with mock.patch("pome.query.csv_adapter.CSVAdapter.query") as csv_query, mock.patch(
        "pome.query.csv_adapter.S3CSVAdapter.query"
    ) as s3_query:
        csv_query.return_value = df
        s3_query.return_value = df
        m = module.MergeAdapter(urls)
        res = m.query("ABC", "2018-06-06", "2018-06-10")
        csv_query.assert_called_with("ABC", "2018-06-06", "2018-06-10")
        s3_query.assert_called_with("ABC", "2018-06-06", "2018-06-10")
        expected = pd.concat([df, df], axis=0).sort_index()
        assert res.equals(expected)
