import io
import pandas as pd
from boto3 import Session
from urllib.parse import urlparse

from backlight.query.adapter import DataSourceAdapter


def read_csv_and_set_index(url: str):
    df = pd.read_csv(url, parse_dates=True)
    if "timestamp" in df:
        df = df.set_index("timestamp")
    elif df.columns[0] == "Unnamed: 0":
        df = df.set_index(df.columns[0])
        del df.index.name
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


class CSVAdapter(DataSourceAdapter):
    """Data source adapter for csv files"""

    def __init__(self, url: str):
        """Initializer.

        Args:
            url     : Url to specify local file path. It shoule start with "file".
        """
        self._url = urlparse(url)
        assert self._url.scheme in ("file",)

    def query(self, symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame:
        """Query pandas dataframe.

        See also :class:`backlight.query.adapter`.
        """
        return read_csv_and_set_index(self._url.path)[start_dt:end_dt]


class S3CSVAdapter(DataSourceAdapter):
    """Data source adapter for csv files on s3
    """

    _s3client = Session().client("s3")

    def __init__(self, url: str):
        """Initializer.

        Args:
            url     : Url to specify s3 file path. It shoule start with "s3".
        """
        self._url = urlparse(url)
        assert self._url.scheme in ("s3",)

    def query(self, symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame:
        """Query pandas dataframe.

        See also :class:`backlight.query.adapter`.
        """
        bucket = self._url.netloc
        key = self._url.path[1:]  # delete first '/'

        obj = S3CSVAdapter._s3client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(
            io.BytesIO(obj["Body"].read()), compression="gzip", parse_dates=True
        )
        df = df[(start_dt <= df.index) & (df.index <= end_dt)].sort_index()
        return df
