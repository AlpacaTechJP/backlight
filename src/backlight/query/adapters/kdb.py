import pandas as pd

from backlight.query.adapter import DataSourceAdapter


class KDBAdapter(DataSourceAdapter):
    """Data source adapter for KDB"""

    def __init__(self, url: str):
        self._url = url

    def query(self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
        pass
