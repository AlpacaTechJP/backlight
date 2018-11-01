from pome.query import csv_glob_adapter as module
from unittest import mock
import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )


def test_CSVGlobAdapter(df):
    path = "/hoge-path/*.csv"
    url = "file://" + path
    symbol = "ABC"
    paths = ["/hoge-path/ABC.csv", "/hoge-path/DEF.csv"]
    target_path = paths[0]

    with mock.patch("pandas.read_csv") as mocked_read_csv, mock.patch(
        "glob.glob"
    ) as mocked_glob:
        mocked_read_csv.return_value = df
        mocked_glob.return_value = paths
        m = module.CSVGlobAdapter(url=url)
        res = m.query(symbol, "2018-06-06", "2018-06-10")
        mocked_read_csv.assert_called_with(target_path, index_col=0, parse_dates=True)

        assert df.equals(res)


def test_CSVGlobAdapter_when_empty(df):
    path = "/hoge-path/*.csv"
    url = "file://" + path
    symbol = "ABC"
    paths = []

    with mock.patch("pandas.read_csv") as mocked_read_csv, mock.patch(
        "glob.glob"
    ) as mocked_glob:
        mocked_read_csv.return_value = df
        mocked_glob.return_value = paths
        m = module.CSVGlobAdapter(url=url)
        res = m.query(symbol, "2018-06-06", "2018-06-10")

        assert res.equals(pd.DataFrame())


def test_S3CSVGlobAdapter(df):
    path = "hoge-path/*.csv"
    bucket = "buzz-bucket"
    url = "s3://" + bucket + "/" + path
    symbol = "ABC"
    keys = ["hoge-path/ABC.csv", "hoge-path/DEF.csv"]
    target_key = keys[0]

    with mock.patch(
        "pome.query.csv_glob_adapter.S3CSVGlobAdapter._s3client"
    ) as mocked_client, mock.patch(
        "pome.query.csv_glob_adapter._list_s3_keys"
    ) as mocked_keys, mock.patch(
        "pandas.read_csv"
    ) as mocked_read_csv, mock.patch(
        "io.BytesIO"
    ) as mocked_io:  # noqa
        mocked_read_csv.return_value = df
        mocked_keys.return_value = keys
        m = module.S3CSVGlobAdapter(url=url)
        res = m.query(symbol, "2018-06-06", "2018-06-10")
        mocked_client.get_object.assert_called_with(Bucket=bucket, Key=target_key)
        assert df.equals(res)


def test_S3CSVGlobAdapter_when_empty(df):
    path = "hoge-path/*.csv"
    bucket = "buzz-bucket"
    url = "s3://" + bucket + "/" + path
    symbol = "ABC"
    keys = []

    with mock.patch(
        "pome.query.csv_glob_adapter._list_s3_keys"
    ) as mocked_keys, mock.patch("pandas.read_csv") as mocked_read_csv, mock.patch(
        "io.BytesIO"
    ) as mocked_io:  # noqa
        mocked_read_csv.return_value = df
        mocked_keys.return_value = keys
        m = module.S3CSVGlobAdapter(url=url)
        res = m.query(symbol, "2018-06-06", "2018-06-10")
        assert res.equals(pd.DataFrame())
