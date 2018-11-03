from backlight.query.adapters import csv as module
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


def test_CSVAdapter(df):
    path = "/hoge-path/fuba.csv"
    url = "file://" + path

    with mock.patch("pandas.read_csv") as mocked:
        mocked.return_value = df
        m = module.CSVAdapter(url=url)
        res = m.query("ABC", "2018-06-06", "2018-06-10")
        mocked.assert_called_with(path, parse_dates=True)

        assert df.equals(res)


def test_S3CSVAdapter(df):
    path = "hoge-path/fuba.csv"
    bucket = "buzz-bucket"
    url = "s3://" + bucket + "/" + path

    with mock.patch(
        "backlight.query.adapters.csv.S3CSVAdapter._s3client"
    ) as mocked_client, mock.patch("pandas.read_csv") as mocked_read_csv, mock.patch(
        "io.BytesIO"
    ) as mocked_io:  # noqa
        mocked_read_csv.return_value = df
        m = module.S3CSVAdapter(url=url)
        res = m.query("ABC", "2018-06-06", "2018-06-10")
        mocked_client.get_object.assert_called_with(Bucket=bucket, Key=path)
        assert df.equals(res)
